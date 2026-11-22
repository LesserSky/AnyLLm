//! Offline parity tests for `anyllm-core`.
//!
//! Ports the unit-testable parts of the Python suite
//! (`test_anyllm.py`, `test_ui.py`): config, catalog, keystore, SSE
//! parsing, provider chunk parsing, usage accounting, and session naming.
//! No network is used.

use std::io::Cursor;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyllm_core::adapters::{Chunk, ChunkStream, Usage};
use anyllm_core::catalog::{
    builtins, get_provider, load_providers, validate_base_url, validate_custom_name,
};
use anyllm_core::config::{
    parse_config_value, read_json, validate_max_tokens, validate_temperature, write_json, Config,
};
use anyllm_core::errors::ErrorKind;
use anyllm_core::keystore::{accepts_generic_api_key, mask_key, KeyStore};
use anyllm_core::session::{
    finish_notice, interrupted_content, sanitize_session_name, session_base_url, Session,
};
use anyllm_core::transport::{iter_sse, join_url, raise_for_status};
use serde_json::{json, Map, Value};

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "anyllm-core-test-{}-{}-{}",
        tag,
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn config_roundtrip_and_value_parsing() {
    let dir = temp_dir("config");
    let path = dir.join("config.json");
    let mut config = Config::new(path.clone(), None).expect("new config");
    assert_eq!(config.get("provider"), Value::Null);
    assert_eq!(config.get("show_reasoning"), json!(false));

    config
        .set("provider", json!("openai"))
        .expect("set provider");
    config
        .set("temperature", json!(0.3))
        .expect("set temperature");
    config.save().expect("save config");

    let reloaded = Config::new(
        path.clone(),
        Some(read_json(&path, "config").expect("read config")),
    )
    .expect("reload config");
    assert_eq!(reloaded.get("provider"), json!("openai"));
    assert_eq!(reloaded.get("temperature"), json!(0.3));

    assert_eq!(parse_config_value("true"), json!(true));
    assert_eq!(parse_config_value("false"), json!(false));
    assert_eq!(parse_config_value("0.5"), json!(0.5));
    assert_eq!(parse_config_value("hello"), json!("hello"));
    assert_eq!(parse_config_value("null"), Value::Null);

    assert_eq!(validate_temperature(&json!(0)).unwrap(), Some(0.0));
    assert_eq!(validate_temperature(&json!(2)).unwrap(), Some(2.0));
    assert_eq!(validate_temperature(&json!("0.25")).unwrap(), Some(0.25));
    assert_eq!(validate_temperature(&Value::Null).unwrap(), None);
    assert!(validate_temperature(&json!(2.1)).is_err());
    assert!(validate_temperature(&json!("hot")).is_err());

    assert_eq!(validate_max_tokens(&json!(1)).unwrap(), Some(1));
    assert_eq!(validate_max_tokens(&json!("16")).unwrap(), Some(16));
    assert_eq!(validate_max_tokens(&Value::Null).unwrap(), None);
    assert!(validate_max_tokens(&json!(0)).is_err());
    assert!(validate_max_tokens(&json!(-3)).is_err());
}

#[test]
fn json_files_roundtrip() {
    let dir = temp_dir("json");
    let path = dir.join("out.json");
    write_json(&path, &json!({"a": 1, "b": ["x"]}), true).expect("write json");
    let data = read_json(&path, "test").expect("read json");
    assert_eq!(data["a"], json!(1));
    assert_eq!(data["b"][0], json!("x"));
}

#[test]
fn catalog_builtins_and_custom_validation() {
    let providers = builtins();
    assert!(providers.contains_key("openai"));
    assert!(providers.contains_key("openscale"));
    assert!(providers.contains_key("custom"));

    let openai = get_provider("OpenAI", &providers).expect("case-insensitive lookup");
    assert_eq!(openai.kind, "openai");
    assert_eq!(openai.default_model(), "gpt-4o-mini");
    assert_eq!(
        openai.auth_headers(Some("sk-test")),
        vec![("Authorization".to_string(), "Bearer sk-test".to_string())]
    );

    let anthropic = get_provider("anthropic", &providers).expect("anthropic");
    assert_eq!(
        anthropic.auth_headers(Some("sk-test")),
        vec![("x-api-key".to_string(), "sk-test".to_string())]
    );

    assert!(validate_custom_name("openai").is_err());
    assert!(validate_custom_name("-bad").is_err());
    assert!(validate_custom_name("").is_err());
    assert_eq!(validate_custom_name(" My-Lab ").unwrap(), "my-lab");

    assert_eq!(
        validate_base_url(" https://api.example.com/v1/ ").unwrap(),
        "https://api.example.com/v1"
    );
    assert!(validate_base_url("ftp://example.com").is_err());
    assert!(validate_base_url("").is_err());
}

#[test]
fn custom_provider_spec_loads() {
    let mut data = Map::new();
    data.insert(
        "custom_providers".to_string(),
        json!({
            "lab": {
                "kind": "openai",
                "base_url": "https://lab.example/v1",
                "env": ["LAB_API_KEY"],
                "models": ["lab-small", "lab-large"],
                "label": "Lab"
            }
        }),
    );
    let config = Config::new(PathBuf::from("unused"), Some(data)).expect("config");
    let providers = load_providers(&config).expect("providers");
    let lab = get_provider("lab", &providers).expect("custom provider");
    assert_eq!(lab.label, "Lab");
    assert_eq!(lab.base_url, "https://lab.example/v1");
    assert_eq!(lab.env_vars, vec!["LAB_API_KEY"]);
    assert_eq!(lab.default_model(), "lab-small");
    assert!(lab.key_required);
    assert!(!lab.local);
}

#[test]
fn keystore_roundtrip_and_masking() {
    let dir = temp_dir("keys");
    let store = KeyStore::with_path(dir.join("keys.json"));
    let scheme = store.set("openai", "sk-test-abcdef").expect("set key");
    assert!(
        scheme == "dpapi" || scheme == "plain",
        "unexpected scheme {scheme}"
    );
    assert_eq!(
        store.get("openai").unwrap().as_deref(),
        Some("sk-test-abcdef")
    );
    assert_eq!(
        store.masked("openai").unwrap().as_deref(),
        Some("sk-t...cdef")
    );
    assert!(store.info().unwrap().contains_key("openai"));
    assert!(store.remove("openai").unwrap());
    assert_eq!(store.get("openai").unwrap(), None);

    assert_eq!(mask_key(""), "(empty)");
    assert_eq!(mask_key("abc"), "***");
    assert_eq!(mask_key("sk-1234567890abc"), "sk-1...0abc");
}

#[test]
fn generic_key_only_reaches_custom_openai_endpoints() {
    let providers = builtins();
    assert!(accepts_generic_api_key(&providers["custom"]));
    assert!(!accepts_generic_api_key(&providers["openai"]));
    assert!(!accepts_generic_api_key(&providers["groq"]));
    assert!(!accepts_generic_api_key(&providers["anthropic"]));
    assert!(!accepts_generic_api_key(&providers["google"]));
}

#[test]
fn join_url_preserves_base_query() {
    assert_eq!(
        join_url(
            "https://example.test/v1?api-version=2024-01-01",
            "chat/completions"
        ),
        "https://example.test/v1/chat/completions?api-version=2024-01-01"
    );
    assert_eq!(
        join_url("https://example.test/v1/", "/models"),
        "https://example.test/v1/models"
    );
}

#[test]
fn sse_handles_crlf_comments_and_multiline_data() {
    let body = b": keepalive\r\n\r\n\
                 data: {\"a\": 1}\r\n\r\n\
                 data: line1\r\ndata: line2\r\n\r\n\
                 data: [DONE]\r\n\r\n";
    let events: Vec<String> = iter_sse(Cursor::new(body.to_vec()))
        .map(|event| event.expect("event"))
        .collect();
    assert_eq!(
        events,
        vec!["{\"a\": 1}".to_string(), "line1\nline2".to_string()]
    );
}

#[test]
fn http_errors_map_to_actionable_kinds() {
    let auth = raise_for_status(
        401,
        br#"{"error":{"message":"bad key"}}"#,
        None,
        "https://api.example.com/v1",
    );
    assert_eq!(auth.kind, ErrorKind::Auth);
    assert!(auth.message.contains("authentication failed"));

    let rate = raise_for_status(429, b"", Some("3"), "https://api.example.com/v1");
    assert_eq!(rate.kind, ErrorKind::RateLimit);
    assert!(rate.hint.unwrap().contains("3 seconds"));

    let context = raise_for_status(
        400,
        br#"{"error":{"message":"maximum context length exceeded"}}"#,
        None,
        "https://api.example.com/v1",
    );
    assert_eq!(context.kind, ErrorKind::Api);
    assert!(context.hint.unwrap().contains("too long"));

    let missing = raise_for_status(404, b"", None, "https://api.example.com/v1");
    assert_eq!(missing.kind, ErrorKind::NotFound);
}

fn read_chunks(kind: &str, body: &[u8]) -> Vec<Chunk> {
    let mut stream = ChunkStream::from_reader(kind, "Mock", Box::new(Cursor::new(body.to_vec())));
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next_chunk().expect("chunk") {
        chunks.push(chunk);
    }
    chunks
}

fn fold(chunks: &[Chunk]) -> (String, String, Usage, Option<String>) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut usage = Usage::default();
    let mut finish = None;
    for chunk in chunks {
        text.push_str(&chunk.text);
        reasoning.push_str(&chunk.reasoning);
        if let Some(chunk_usage) = &chunk.usage {
            usage.update(chunk_usage);
        }
        if let Some(reason) = &chunk.finish_reason {
            finish = Some(reason.clone());
        }
    }
    (text, reasoning, usage, finish)
}

#[test]
fn openai_stream_parses_text_usage_and_finish() {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"from mock\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,",
        "\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
        "data: [DONE]\n\n",
    );
    let (text, _reasoning, usage, finish) = fold(&read_chunks("openai", body.as_bytes()));
    assert_eq!(text, "Hello from mock");
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.cached_tokens, Some(1));
    assert_eq!(usage.total(), Some(10));
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[test]
fn openai_stream_parses_reasoning_delta() {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (text, reasoning, _usage, _finish) = fold(&read_chunks("openai", body.as_bytes()));
    assert_eq!(text, "answer");
    assert_eq!(reasoning, "think");
}

#[test]
fn anthropic_stream_parses_text_usage_and_finish() {
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":",
        "{\"input_tokens\":5,\"cache_read_input_tokens\":2,\"output_tokens\":1}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":",
        "{\"type\":\"text_delta\",\"text\":\"Hi \"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":",
        "{\"type\":\"text_delta\",\"text\":\"from Claude\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},",
        "\"usage\":{\"output_tokens\":4}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let (text, _reasoning, usage, finish) = fold(&read_chunks("anthropic", body.as_bytes()));
    assert_eq!(text, "Hi from Claude");
    assert_eq!(usage.input_tokens, Some(5));
    assert_eq!(usage.output_tokens, Some(4));
    assert_eq!(usage.cached_tokens, Some(2));
    assert_eq!(finish.as_deref(), Some("end_turn"));
}

#[test]
fn gemini_stream_parses_text_usage_and_finish() {
    let body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Gemini \"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"says hi\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"finishReason\":\"STOP\"}],",
        "\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":2}}\n\n",
    );
    let (text, _reasoning, usage, finish) = fold(&read_chunks("gemini", body.as_bytes()));
    assert_eq!(text, "Gemini says hi");
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[test]
fn gemini_thought_parts_are_reasoning() {
    let body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[",
        "{\"text\":\"plan\",\"thought\":true}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}]}}]}\n\n",
    );
    let (text, reasoning, _usage, _finish) = fold(&read_chunks("gemini", body.as_bytes()));
    assert_eq!(text, "done");
    assert_eq!(reasoning, "plan");
}

#[test]
fn usage_accounting() {
    let mut usage = Usage {
        input_tokens: Some(2),
        output_tokens: None,
        reasoning_tokens: Some(1),
        cached_tokens: None,
    };
    usage.update(&Usage {
        input_tokens: Some(1),
        output_tokens: Some(3),
        reasoning_tokens: Some(0),
        cached_tokens: Some(4),
    });
    assert_eq!(usage.input_tokens, Some(2));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.reasoning_tokens, Some(1));
    assert_eq!(usage.cached_tokens, Some(4));
    assert_eq!(usage.total(), Some(5));

    let combined = usage.combined(&Usage {
        input_tokens: Some(5),
        output_tokens: Some(4),
        reasoning_tokens: None,
        cached_tokens: Some(1),
    });
    assert_eq!(combined.input_tokens, Some(7));
    assert_eq!(combined.output_tokens, Some(7));
    assert_eq!(combined.reasoning_tokens, Some(1));
    assert_eq!(combined.cached_tokens, Some(5));
    assert_eq!(combined.total(), Some(14));
    assert_eq!(Usage::from_dict(&combined.to_dict()), combined);
}

#[test]
fn session_names_and_finish_notices() {
    assert_eq!(
        sanitize_session_name(" My chat/one ").unwrap(),
        "My chat-one"
    );
    assert_eq!(sanitize_session_name("a//b").unwrap(), "a-b");
    assert!(sanitize_session_name("...").is_err());
    assert!(sanitize_session_name("").is_err());

    assert!(finish_notice(Some("length"))
        .unwrap()
        .contains("token limit"));
    assert!(finish_notice(Some("max_tokens"))
        .unwrap()
        .contains("token limit"));
    assert!(finish_notice(Some("content_filter"))
        .unwrap()
        .contains("content filter"));
    assert_eq!(finish_notice(Some("stop")), None);
    assert_eq!(finish_notice(None), None);

    assert_eq!(
        interrupted_content("Hello there"),
        "Hello there\n[interrupted]"
    );
    assert_eq!(
        interrupted_content("Hello there\n[interrupted]"),
        "Hello there\n[interrupted]"
    );
}

#[test]
fn session_serializes_expected_shape_and_target() {
    let provider = builtins().remove("openai").expect("openai");
    let mut session = Session::new(
        provider,
        "gpt-4o-mini",
        None,
        Some("sk-test"),
        "saved",
        Some("be brief"),
        Some(0.3),
        Some(128),
        60.0,
        true,
    );
    session
        .append_exchange("hi", "hello", None)
        .expect("append");
    let data = session.to_dict();
    assert_eq!(data["version"], json!(1));
    assert_eq!(data["provider"], json!("openai"));
    assert_eq!(data["model"], json!("gpt-4o-mini"));
    assert_eq!(data["system"], json!("be brief"));
    assert_eq!(data["show_reasoning"], json!(true));
    assert_eq!(data["messages"][0]["role"], json!("user"));
    assert_eq!(data["messages"][1]["content"], json!("hello"));
    assert_eq!(data["usage"]["total"], Value::Null);
    assert!(data.get("updated").and_then(Value::as_str).is_some());
    assert!(data.get("target").is_none());

    session.target_alias = Some("   ".to_string());
    assert!(session.to_dict().get("target").is_none());
    session.target_alias = Some("prod".to_string());
    assert_eq!(session.to_dict()["target"], json!("prod"));
}

#[test]
fn session_base_url_prefers_override_then_saved() {
    let provider = builtins().remove("openai").expect("openai");
    let mut data = Map::new();
    data.insert("base_url".to_string(), json!("https://saved.example/v1"));
    assert_eq!(
        session_base_url(&data, &provider, None).unwrap(),
        "https://saved.example/v1"
    );
    assert_eq!(
        session_base_url(&data, &provider, Some("https://override.example/v1")).unwrap(),
        "https://override.example/v1"
    );
    assert_eq!(
        session_base_url(&Map::new(), &provider, None).unwrap(),
        "https://api.openai.com/v1"
    );
}
