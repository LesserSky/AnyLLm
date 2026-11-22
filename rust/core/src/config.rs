//! Configuration paths, defaults, and JSON persistence.
//!
//! Port of Python `anyllm/config.py`. `ANYLLM_HOME` overrides the platform
//! default. JSON files are written atomically (temp file + rename), with
//! owner-only permissions for private files.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::errors::{AnyLLMError, Result};

pub const APP_NAME: &str = "anyllm";

pub const ALLOWED_KEYS: &[&str] = &[
    "provider",
    "model",
    "system",
    "temperature",
    "max_tokens",
    "show_reasoning",
];

pub fn config_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("ANYLLM_HOME") {
        return PathBuf::from(shellexpand_home(&override_dir));
    }
    if cfg!(windows) {
        let base = std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = home_dir();
                home.join("AppData").join("Roaming")
            });
        base.join(APP_NAME)
    } else {
        let base = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home_dir().join(".config"));
        base.join(APP_NAME)
    }
}

fn home_dir() -> PathBuf {
    #[cfg(windows)]
    let var = "USERPROFILE";
    #[cfg(not(windows))]
    let var = "HOME";
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn shellexpand_home(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("~") {
        if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') {
            return format!("{}{}", home_dir().to_string_lossy(), rest);
        }
    }
    text.to_string()
}

pub fn ensure_config_dir() -> Result<PathBuf> {
    let directory = config_dir();
    fs::create_dir_all(&directory).map_err(|e| {
        AnyLLMError::config(format!("Could not create {}: {}", directory.display(), e))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&directory, fs::Permissions::from_mode(0o700));
    }
    Ok(directory)
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

pub fn keys_path() -> PathBuf {
    config_dir().join("keys.json")
}

pub fn sessions_dir() -> PathBuf {
    ensure_config_dir().expect("config dir").join("sessions")
}

pub fn history_path() -> PathBuf {
    config_dir().join("input-history")
}

/// Atomically write pretty JSON, optionally with owner-only permissions.
pub fn write_json(path: &Path, data: &Value, private: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            AnyLLMError::config(format!("Could not create {}: {}", parent.display(), e))
        })?;
    }
    let temp = path.with_extension(format!("{}.tmp", process_unique_suffix()));
    let body = serde_json::to_string_pretty(data).unwrap_or_default();
    let mut file = fs::File::create(&temp)
        .map_err(|e| AnyLLMError::config(format!("Could not write {}: {}", temp.display(), e)))?;
    if private {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&temp, fs::Permissions::from_mode(0o600));
        }
    }
    file.write_all(body.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|e| AnyLLMError::config(format!("Could not write {}: {}", temp.display(), e)))?;
    drop(file);
    fs::rename(&temp, path)
        .map_err(|e| AnyLLMError::config(format!("Could not replace {}: {}", path.display(), e)))?;
    if private {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        }
    }
    Ok(())
}

fn process_unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), now)
}

pub fn read_json(path: &Path, what: &str) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let raw = fs::read_to_string(path).map_err(|e| {
        AnyLLMError::config(format!("Could not read the {} file: {}", what, e))
            .with_hint(path.display().to_string())
    })?;
    let data: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(e) => {
            // serde reports line/column; mirror the Python hint shape.
            let line = e.line();
            return Err(AnyLLMError::config(format!(
                "The {} file is not valid JSON (line {}).",
                what, line
            ))
            .with_hint(format!("Fix or delete {}", path.display())));
        }
    };
    let map = match data {
        Value::Object(map) => map,
        _ => {
            return Err(AnyLLMError::config(format!(
                "The {} file must contain a JSON object.",
                what
            ))
            .with_hint(format!("Fix or delete {}", path.display())));
        }
    };
    Ok(map)
}

pub fn defaults() -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("version".into(), json!(1));
    map.insert("provider".into(), Value::Null);
    map.insert("model".into(), Value::Null);
    map.insert("system".into(), Value::Null);
    map.insert("temperature".into(), Value::Null);
    map.insert("max_tokens".into(), Value::Null);
    map.insert("show_reasoning".into(), json!(false));
    map.insert("custom_providers".into(), json!({}));
    map
}

/// Small wrapper around the config data with defaults applied.
pub struct Config {
    pub path: PathBuf,
    pub data: Map<String, Value>,
}

impl Config {
    pub fn new(path: PathBuf, data: Option<Map<String, Value>>) -> Result<Self> {
        let mut merged = defaults();
        if let Some(data) = data {
            for (key, value) in data {
                merged.insert(key, value);
            }
        }
        if !merged
            .get("custom_providers")
            .map(Value::is_object)
            .unwrap_or(false)
        {
            return Err(
                AnyLLMError::config("'custom_providers' in the config must be an object.")
                    .with_hint(format!("Fix or delete {}", path.display())),
            );
        }
        Ok(Config { path, data: merged })
    }

    pub fn get(&self, key: &str) -> Value {
        self.get_opt(key)
            .unwrap_or_else(|| defaults().get(key).cloned().unwrap_or(Value::Null))
    }

    /// Raw lookup without defaults applied (Python `dict.get` shape).
    pub fn get_opt(&self, key: &str) -> Option<Value> {
        self.data.get(key).cloned()
    }

    /// Load the config file from the platform config directory.
    pub fn load() -> Result<Self> {
        let path = config_path();
        let data = read_json(&path, "config")?;
        Config::new(path, Some(data))
    }

    pub fn set(&mut self, key: &str, value: Value) -> Result<()> {
        if !ALLOWED_KEYS.contains(&key) && key != "custom_providers" {
            return Err(
                AnyLLMError::config(format!("Unknown config key {:?}.", key))
                    .with_hint(format!("Known keys: {}", ALLOWED_KEYS.join(", "))),
            );
        }
        self.data.insert(key.into(), value);
        Ok(())
    }

    pub fn unset(&mut self, key: &str) -> Result<()> {
        if !ALLOWED_KEYS.contains(&key) {
            return Err(
                AnyLLMError::config(format!("Unknown config key {:?}.", key))
                    .with_hint(format!("Known keys: {}", ALLOWED_KEYS.join(", "))),
            );
        }
        let default = defaults().get(key).cloned().unwrap_or(Value::Null);
        self.data.insert(key.into(), default);
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        let mut payload = self.data.clone();
        payload.insert("version".into(), json!(1));
        write_json(&self.path, &Value::Object(payload), true)
    }
}

pub fn load_config() -> Result<Config> {
    Config::load()
}

// -- value parsing for `config set` -----------------------------------------

/// Parse a command-line value into a JSON-ish value.
pub fn parse_config_value(text: &str) -> Value {
    let stripped = text.trim();
    let lowered = stripped.to_lowercase();
    if lowered == "true" || lowered == "false" {
        return json!(lowered == "true");
    }
    if lowered == "null" || lowered == "none" {
        return Value::Null;
    }
    match serde_json::from_str(stripped) {
        Ok(value) => value,
        Err(_) => json!(stripped),
    }
}

fn as_float(value: &Value, name: &str) -> Result<Option<f64>> {
    match value {
        Value::Null => Ok(None),
        Value::Number(n) => Ok(n.as_f64()),
        Value::String(s) => s
            .parse::<f64>()
            .map(Some)
            .map_err(|_| bad_number(value, name, "Use a number, e.g. 0.7")),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        _ => Err(bad_number(value, name, "Use a number, e.g. 0.7")),
    }
}

fn bad_number(value: &Value, name: &str, hint: &str) -> AnyLLMError {
    AnyLLMError::config(format!("Invalid {} value {}.", name, value)).with_hint(hint)
}

fn as_int(value: &Value, name: &str) -> Result<Option<i64>> {
    match value {
        Value::Null => Ok(None),
        Value::Number(n) => Ok(n.as_i64()),
        Value::String(s) => s
            .parse::<i64>()
            .map(Some)
            .map_err(|_| bad_number(value, name, "Use a whole number, e.g. 1024")),
        Value::Bool(b) => Ok(Some(if *b { 1 } else { 0 })),
        _ => Err(bad_number(value, name, "Use a whole number, e.g. 1024")),
    }
}

/// Accept a sampling temperature in `[0, 2]`, or `None` for the default.
pub fn validate_temperature(value: &Value) -> Result<Option<f64>> {
    let number = as_float(value, "temperature")?;
    if let Some(n) = number {
        if !(0.0..=2.0).contains(&n) {
            return Err(AnyLLMError::config(format!(
                "temperature must be between 0 and 2 (got {}).",
                n
            ))
            .with_hint("Most models are happiest between 0 and 1."));
        }
    }
    Ok(number)
}

/// Accept a positive response-token cap, or `None` for the provider default.
pub fn validate_max_tokens(value: &Value) -> Result<Option<i64>> {
    let number = as_int(value, "max_tokens")?;
    if let Some(n) = number {
        if n <= 0 {
            return Err(AnyLLMError::config(format!(
                "max_tokens must be greater than 0 (got {}).",
                n
            ))
            .with_hint("Try 1024 or 4096."));
        }
    }
    Ok(number)
}
