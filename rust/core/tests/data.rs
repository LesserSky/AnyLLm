//! Data module parity and cross-platform storage contracts.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyllm_core::config::{parse_config_value, write_json, Config};
use anyllm_core::errors::python_repr;
use anyllm_core::keystore::{dpapi_available, mask_key, KeyStore};
use anyllm_core::term::{format_table, Style};
use serde_json::json;
use serde_json::Map;

fn temp_dir() -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("anyllm-data-test-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp directory");
    dir
}

#[cfg(unix)]
#[test]
fn json_write_modes_match_private_and_public_contracts() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_dir();
    let private = dir.join("private.json");
    let public = dir.join("public.json");
    write_json(&private, &json!({"secret": true}), true).expect("private write");
    write_json(&public, &json!({"public": true}), false).expect("public write");

    assert_eq!(
        std::fs::metadata(private).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(public).unwrap().permissions().mode() & 0o777,
        0o644
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn key_masking_counts_unicode_characters_and_dpapi_is_platform_gated() {
    assert_eq!(mask_key("密钥abcdefgh"), "密钥ab...efgh");
    assert_eq!(dpapi_available(), cfg!(windows));
}

#[test]
fn malformed_dpapi_entries_are_rejected_off_windows_without_disclosing_data() {
    if cfg!(windows) {
        return;
    }
    let dir = temp_dir();
    let path = dir.join("keys.json");
    std::fs::write(
        &path,
        r#"{"keys":{"openai":{"scheme":"dpapi","value":"AQ=="}}}"#,
    )
    .unwrap();
    let err = KeyStore::with_path(path).get("openai").unwrap_err();
    assert!(err.message.contains("Windows DPAPI"));
    assert!(!err.message.contains("AQ=="));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn plain_config_values_keep_original_whitespace_and_errors_use_python_repr() {
    assert_eq!(parse_config_value(" hello "), json!(" hello "));
    assert_eq!(python_repr("bad key"), "'bad key'");
    let mut config = Config::new(PathBuf::from("unused"), Some(Map::new())).unwrap();
    let error = config.set("bad", json!(1)).unwrap_err();
    assert_eq!(error.message, "Unknown config key 'bad'.");
}

#[test]
fn tables_preserve_cell_whitespace() {
    let lines = format_table(
        &Style::plain(),
        &["Name", "Value"],
        &[vec!["  a".into(), "b  ".into()]],
    );
    assert_eq!(lines[1], "  a   b");
}
