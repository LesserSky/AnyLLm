//! Integration tests for the ported orchestration layer: targets, budgets,
//! and the seam, exercised end to end through the public API. The unit tests
//! inside each module cover the fine-grained Python behaviors; these tests
//! verify the pieces compose the same way the Python CLI does.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::{json, Value};

use anyllm_orch::budgets::{self, Budget, PlannedCall, estimate_cost, preflight, receipt};
use anyllm_orch::decimal::Decimal;
use anyllm_orch::seam::{
    self, ApprovedAttachment, RunCall, RunOptions, load_feature, make_snapshot, run_request,
    save_feature,
};
use anyllm_orch::sys::{
    builtins, Chunk, EmptyKeyStore, Error, KeyStore, Message, Provider, Usage,
};
use anyllm_orch::targets::{self, Target};

// One process-wide lock: these tests redirect ANYLLM_HOME and set key env
// vars, so they must not overlap each other.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct TestHome(std::path::PathBuf);

impl TestHome {
    fn new(tag: &str) -> TestHome {
        let dir = std::env::temp_dir().join(format!(
            "orch-it-{}-{}",
            tag,
            anyllm_orch::sys::new_id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TestHome(dir)
    }
}

impl Drop for TestHome {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn clear_key_env() {
    let keys: Vec<String> = std::env::vars()
        .map(|(key, _)| key)
        .filter(|key| key.ends_with("_API_KEY") || key.starts_with("ANYLLM_"))
        .collect();
    for key in keys {
        if key != "ANYLLM_HOME" {
            std::env::remove_var(&key);
        }
    }
}

/// A key store with fixed saved keys, standing in for the keystore.
struct MapStore(BTreeMap<String, String>);

impl KeyStore for MapStore {
    fn get(&self, name: &str) -> Result<Option<String>, Error> {
        Ok(self.0.get(name).cloned())
    }
}

// ---------------------------------------------------------------------------
// A fake ChatBackend, mirroring the mock API the Python suite spins up.

#[derive(Default)]
struct FakeBackend {
    chunks: Vec<Chunk>,
}

impl anyllm_orch::sys::ChatBackend for FakeBackend {
    fn stream(&self, _request: &anyllm_orch::sys::ChatRequest) -> Result<Vec<Chunk>, Error> {
        Ok(self.chunks.clone())
    }
}

#[test]
fn target_resolution_gates_credentials_by_host() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let home = TestHome::new("resolve");
    std::env::set_var("ANYLLM_HOME", &home.0);
    clear_key_env();

    let providers = builtins();
    let mut store = MapStore(BTreeMap::new());
    store.0.insert("target:lab".to_string(), "target-secret".to_string());

    // Saved target on the provider's own host: provider env key is used.
    std::env::set_var("OPENAI_API_KEY", "sk-openai-secret");
    targets::add_target(
        "official",
        "openai",
        "gpt-4o-mini",
        None,
        &providers,
        &store,
    )
    .unwrap();
    let resolved = targets::resolve_named("official", &providers, &store, true).unwrap();
    assert_eq!(resolved.credential.as_deref(), Some("sk-openai-secret"));
    assert!(resolved.host_is_default);
    assert!(resolved.http_warning.is_none());

    // Same provider pointed at a different host: provider key withheld, but
    // the dedicated target key is accepted there.
    targets::add_target(
        "lab",
        "openai",
        "local",
        Some("http://127.0.0.1:9/v1"),
        &providers,
        &store,
    )
    .unwrap();
    let lab = targets::resolve_named("lab", &providers, &store, true).unwrap();
    assert_eq!(lab.credential.as_deref(), Some("target-secret"));
    assert_eq!(lab.credential_source, "saved-target");
    assert!(!lab.host_is_default);

    // A target on a non-default host with no target key refuses to run.
    targets::add_target(
        "orphan",
        "openai",
        "local",
        Some("http://other.example/v1"),
        &providers,
        &EmptyKeyStore,
    )
    .unwrap();
    let error = targets::resolve_named("orphan", &providers, &EmptyKeyStore, true).unwrap_err();
    assert!(error.message.contains("No credential is configured"));
    assert!(error.hint.unwrap_or_default().contains("will not be sent"));

    std::env::remove_var("OPENAI_API_KEY");
}

#[test]
fn budget_preflight_and_receipt_compose() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let home = TestHome::new("budget");
    std::env::set_var("ANYLLM_HOME", &home.0);
    clear_key_env();

    let price = budgets::set_price(
        "fast",
        &Value::from("0.20"),
        &Value::from("0.40"),
        Some("usd"),
        "2026-01-01T00:00:00+00:00",
    )
    .unwrap();
    assert_eq!(price.currency, "USD");
    let stored = budgets::load_prices().unwrap()["fast"].clone();
    assert_eq!(
        estimate_cost(&stored, 1_000_000, 2_000_000).format_f(),
        "1.000000"
    );

    // Unknown prices are refused under a dollar gate, never treated as zero.
    let calls = vec![
        PlannedCall::new("fast", "fast", 1_000_000, Some(1_000_000)),
        PlannedCall::new("slow", "slow", 1_000_000, Some(1_000_000)),
    ];
    let mut prices = BTreeMap::new();
    prices.insert("fast".to_string(), stored.clone());
    let report = preflight(
        &calls,
        &prices,
        &Budget { estimated_usd: Some(Decimal::parse("10").unwrap()), ..Budget::new() },
        None,
    );
    assert!(!report.allowed);
    assert!(report.reason.unwrap_or_default().contains("slow"));

    // A complete estimate below the gate is allowed, and the disclosure says
    // it is a heuristic, not a bill.
    prices.insert(
        "slow".to_string(),
        budgets::Price {
            name: "slow".to_string(),
            input_per_million: Decimal::parse("0.20").unwrap(),
            output_per_million: Decimal::parse("0.40").unwrap(),
            currency: "USD".to_string(),
            updated: String::new(),
        },
    );
    let allowed = preflight(
        &calls,
        &prices,
        &Budget { estimated_usd: Some(Decimal::parse("2").unwrap()), ..Budget::new() },
        None,
    );
    assert!(allowed.allowed);
    assert!(allowed.disclosure.contains("not a hard spending cap"));
    assert_eq!(allowed.estimate.unwrap().format_f(), "1.200000");

    // Reported usage feeds the receipt; unknown usage stays "unknown".
    let usage = Usage {
        input_tokens: Some(1_000_000),
        output_tokens: Some(1_000_000),
        reasoning_tokens: Some(5_000_000),
        cached_tokens: Some(9),
    };
    let info = receipt(Some(&usage), prices.get("fast"));
    assert_eq!(info["estimated_cost"], Value::from("0.600000"));
    assert_eq!(info["cost_basis"], Value::from("reported"));
    let missing = receipt(None, prices.get("fast"));
    assert_eq!(missing["estimated_cost"], Value::from("unknown"));
}

#[test]
fn seam_run_request_and_feature_files() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let home = TestHome::new("seam");
    std::env::set_var("ANYLLM_HOME", &home.0);
    clear_key_env();

    // run_request against a scripted backend: status derives the same way.
    let backend = FakeBackend {
        chunks: vec![
            Chunk { text: "Hel".into(), ..Default::default() },
            Chunk { text: "lo".into(), ..Default::default() },
            Chunk {
                usage: Some(Usage { input_tokens: Some(7), output_tokens: Some(4), ..Default::default() }),
                ..Default::default()
            },
            Chunk { finish_reason: Some("stop".into()), ..Default::default() },
        ],
    };
    let provider = Provider::new("mock", "Mock", "openai", "http://127.0.0.1/v1");
    let target = Target::new("mock", "m", "http://127.0.0.1/v1", Some("fast"));
    let snapshot = make_snapshot(
        &[Message::new("user", "hi")],
        None,
        None,
        None,
        Some(vec![ApprovedAttachment::new("notes.txt", "abc", "hello-notes")]),
        false,
        None,
    );
    let result = run_request(
        &backend,
        &provider,
        &target,
        &snapshot,
        &RunOptions::new(5.0),
        RunCall {
            credential: Some("secret"),
            prompt: Some("hi"),
            messages: None,
            cancel: None,
            events: None,
        },
    );
    assert_eq!(result.text, "Hello");
    assert_eq!(result.status, "ok");
    let public = result.public_value();
    assert_eq!(public["alias"], "fast");
    assert_eq!(public["usage"]["input"], Value::from(7));

    // Feature files: version marker round-trips and future versions refuse.
    let path = seam::feature_path("targets.json");
    let mut payload = serde_json::Map::new();
    payload.insert("targets".to_string(), json!({}));
    save_feature(&path, payload).unwrap();
    let loaded = load_feature(&path, "targets").unwrap();
    assert_eq!(loaded["version"], Value::from(1));

    // Attachments survive a snapshot and are read only by exact path.
    let map = snapshot.attachment_map();
    assert_eq!(map["notes.txt"].content, "hello-notes");
}
