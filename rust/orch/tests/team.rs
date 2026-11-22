//! Team behavior tests: bounded fan-out, budget gates, and saved/adopted runs.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyllm_orch::budgets::{self, Budget};
use anyllm_orch::sys::{
    builtins, ChatBackend, ChatRequest, Chunk, EmptyKeyStore, Error, Message, Provider, Session,
};
use anyllm_orch::team;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct TestHome(std::path::PathBuf);
impl TestHome {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("orch-team-{}", anyllm_orch::sys::new_id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestHome {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct MockBackend {
    active: AtomicUsize,
    peak: AtomicUsize,
    calls: AtomicUsize,
}
impl MockBackend {
    fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
        }
    }
}
impl ChatBackend for MockBackend {
    fn stream(&self, request: &ChatRequest) -> Result<Vec<Chunk>, Error> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(15));
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(vec![
            Chunk {
                text: format!("answer:{}", request.model),
                ..Default::default()
            },
            Chunk {
                finish_reason: Some("stop".into()),
                ..Default::default()
            },
        ])
    }
}

fn setup() -> (TestHome, BTreeMap<String, Provider>) {
    let home = TestHome::new();
    std::env::set_var("ANYLLM_HOME", &home.0);
    let mut providers = builtins();
    let mut mock = Provider::new("mock", "Mock", "openai", "http://127.0.0.1:1/v1");
    mock.key_required = false;
    providers.insert("mock".into(), mock);
    (home, providers)
}

#[test]
fn fan_out_is_bounded_and_preserves_member_order() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (_home, providers) = setup();
    anyllm_orch::targets::add_target("shared", "mock", "first", None, &providers, &EmptyKeyStore)
        .unwrap();
    let members = vec![
        ("slow".into(), "shared".into()),
        ("fast".into(), "shared".into()),
        ("last".into(), "mock:third".into()),
    ];
    let backend = MockBackend::new();
    let artifact = team::execute_team_ask(
        &backend,
        &members,
        "compare",
        &providers,
        &EmptyKeyStore,
        &[Message::new("user", "history")],
        None,
        None,
        Some(32),
        5.0,
        Some(5.0),
        "stacked",
        None,
        None,
        2,
        true,
        false,
        anyllm_orch::sys::Style::new(false),
        80,
    )
    .unwrap();
    let results = artifact["results"].as_array().unwrap();
    assert_eq!(
        results
            .iter()
            .map(|r| r["alias"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["slow", "fast", "last"]
    );
    assert_eq!(results[0]["text"], "answer:first");
    assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
    assert_eq!(backend.peak.load(Ordering::SeqCst), 2);
}

#[test]
fn request_budget_gate_refuses_before_any_model_call() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (_home, providers) = setup();
    budgets::save_budget(&Budget {
        max_requests: Some(1),
        ..Budget::new()
    })
    .unwrap();
    let backend = MockBackend::new();
    let result = team::execute_team_ask(
        &backend,
        &[
            ("a".into(), "mock:first".into()),
            ("b".into(), "mock:second".into()),
        ],
        "compare",
        &providers,
        &EmptyKeyStore,
        &[],
        None,
        None,
        Some(16),
        5.0,
        None,
        "stacked",
        None,
        None,
        2,
        true,
        false,
        anyllm_orch::sys::Style::new(false),
        80,
    );
    assert!(result
        .unwrap_err()
        .message
        .contains("above the budget of 1"));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn saved_run_loads_and_adopt_appends_one_exchange() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (_home, providers) = setup();
    let backend = MockBackend::new();
    let artifact = team::execute_team_ask(
        &backend,
        &[
            ("a".into(), "mock:first".into()),
            ("b".into(), "mock:second".into()),
        ],
        "question",
        &providers,
        &EmptyKeyStore,
        &[],
        None,
        None,
        Some(16),
        5.0,
        Some(5.0),
        "stacked",
        None,
        None,
        2,
        true,
        false,
        anyllm_orch::sys::Style::new(false),
        80,
    )
    .unwrap();
    let loaded = team::load_last_run().unwrap();
    assert_eq!(loaded["id"], artifact["id"]);
    let provider = providers["mock"].clone();
    let mut session = Session::new(provider, "first");
    team::adopt_run(&mut session, &loaded, "a").unwrap();
    assert_eq!(session.messages.len(), 2);
    assert_eq!(session.messages[0].content, "question");
    assert_eq!(session.messages[1].content, "answer:first");
    assert_eq!(session.turns, 1);
}
