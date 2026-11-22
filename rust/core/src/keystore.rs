//! API key storage.
//!
//! Port of Python `anyllm/keystore.py`. Keys live in a separate `keys.json`:
//!
//! * Windows: encrypted with DPAPI (current user only).
//! * Linux/macOS: base64 in a file with permissions 0600.
//!
//! Environment variables always win over saved keys.

use std::collections::BTreeMap;
use std::path::PathBuf;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{json, Map, Value};

use crate::catalog::{builtins, Provider, OPENAI};
use crate::config::{keys_path, read_json, write_json};
use crate::errors::{AnyLLMError, Result};

pub const SCHEME_DPAPI: &str = "dpapi";
pub const SCHEME_PLAIN: &str = "plain";

// -- DPAPI (Windows only) ----------------------------------------------------

#[cfg(windows)]
mod dpapi {
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    pub fn protect(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut blob_in = CRYPT_INTEGER_BLOB {
            cbData: data.len() as u32,
            pbData: data.as_ptr() as *mut u8,
        };
        let mut blob_out = CRYPT_INTEGER_BLOB {
            cbData: 0,
            pbData: std::ptr::null_mut(),
        };
        let ok = unsafe {
            CryptProtectData(
                &mut blob_in,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null() as *const c_void,
                std::ptr::null_mut(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut blob_out,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let out = unsafe { std::slice::from_raw_parts(blob_out.pbData, blob_out.cbData as usize) }
            .to_vec();
        unsafe { LocalFree(blob_out.pbData as *mut c_void) };
        Ok(out)
    }

    pub fn unprotect(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut blob_in = CRYPT_INTEGER_BLOB {
            cbData: data.len() as u32,
            pbData: data.as_ptr() as *mut u8,
        };
        let mut blob_out = CRYPT_INTEGER_BLOB {
            cbData: 0,
            pbData: std::ptr::null_mut(),
        };
        let ok = unsafe {
            CryptUnprotectData(
                &mut blob_in,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null() as *const c_void,
                std::ptr::null_mut(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut blob_out,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let out = unsafe { std::slice::from_raw_parts(blob_out.pbData, blob_out.cbData as usize) }
            .to_vec();
        unsafe { LocalFree(blob_out.pbData as *mut c_void) };
        Ok(out)
    }
}

pub fn dpapi_available() -> bool {
    cfg!(windows)
}

pub fn utc_now_iso() -> String {
    // 2026-11-22T12:34:56+00:00 shape, matching Python
    // datetime.now(timezone.utc).replace(microsecond=0).isoformat().
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00",
        year,
        month,
        day,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's civil-from-days algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn mask_key(value: &str) -> String {
    if value.is_empty() {
        return "(empty)".to_string();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    format!(
        "{}...{}",
        value[..4].to_string(),
        value[chars.len() - 4..].to_string()
    )
}

fn env_slug(name: &str) -> String {
    let mut out = String::new();
    let mut pending_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_underscore && !out.is_empty() {
                out.push('_');
            }
            pending_underscore = false;
            out.push(ch.to_ascii_uppercase());
        } else {
            pending_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Where a key came from.
#[derive(Debug, Clone)]
pub struct KeyResolution {
    pub value: Option<String>,
    pub source: String,
    pub env_var: Option<String>,
    pub stored: bool,
}

impl KeyResolution {
    pub fn present(&self) -> bool {
        self.value
            .as_deref()
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    pub fn display(&self) -> String {
        if let Some(var) = &self.env_var {
            return format!("env:{}", var);
        }
        if self.stored {
            return "saved".to_string();
        }
        if self
            .value
            .as_deref()
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            return "configured".to_string();
        }
        "missing".to_string()
    }
}

/// Read/write the per-user key file.
pub struct KeyStore {
    pub path: PathBuf,
}

impl Default for KeyStore {
    fn default() -> Self {
        KeyStore { path: keys_path() }
    }
}

impl KeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_path(path: PathBuf) -> Self {
        KeyStore { path }
    }

    fn load(&self) -> Result<Map<String, Value>> {
        let mut data = read_json(&self.path, "key store")?;
        if data.is_empty() {
            return Ok(serde_json::Map::new());
        }
        match data.get("keys") {
            None => {
                data.insert("keys".into(), json!({}));
            }
            Some(Value::Object(_)) => {}
            Some(_) => {
                return Err(AnyLLMError::config(
                    "The key store is malformed: 'keys' must be an object.",
                )
                .with_hint(format!("Fix or delete {}", self.path.display())));
            }
        }
        Ok(data)
    }

    fn save(&self, mut data: Map<String, Value>) -> Result<()> {
        data.insert("version".into(), json!(1));
        write_json(&self.path, &Value::Object(data), true).map_err(|e| {
            AnyLLMError::config(format!("Could not save the API key: {}", e.message)).with_hint(
                format!(
                    "Check that you can write to {}",
                    self.path
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ),
            )
        })
    }

    fn keys_of(data: &Map<String, Value>) -> Map<String, Value> {
        data.get("keys")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    }

    /// Decrypted key for a provider, or None when nothing is stored.
    pub fn get(&self, provider: &str) -> Result<Option<String>> {
        let data = self.load()?;
        let keys = Self::keys_of(&data);
        let entry = match keys.get(provider) {
            None => return Ok(None),
            Some(entry) => entry,
        };
        let entry = entry.as_object().ok_or_else(|| {
            AnyLLMError::config(format!("The saved key for {:?} is malformed.", provider))
                .with_hint(format!("Run `anyllm keys set {}` again.", provider))
        })?;
        let scheme = entry.get("scheme").and_then(Value::as_str).unwrap_or("");
        let encoded = entry.get("value").and_then(Value::as_str).unwrap_or("");
        if encoded.is_empty() {
            return Err(AnyLLMError::config(format!(
                "The saved key for {:?} is empty or malformed.",
                provider
            ))
            .with_hint(format!("Run `anyllm keys set {}` again.", provider)));
        }
        let raw = BASE64.decode(encoded.as_bytes()).map_err(|_| {
            AnyLLMError::config(format!("The saved key for {:?} is corrupted.", provider))
                .with_hint(format!("Run `anyllm keys set {}` again.", provider))
        })?;
        if scheme == SCHEME_PLAIN {
            return String::from_utf8(raw)
                .map(Some)
                .map_err(|_| corrupted(provider));
        }
        if scheme == SCHEME_DPAPI {
            if !dpapi_available() {
                return Err(AnyLLMError::config(format!(
                    "The key for {:?} was encrypted with Windows DPAPI \
                     and cannot be read on this system.",
                    provider
                ))
                .with_hint(format!(
                    "Set it again on this machine: anyllm keys set {}",
                    provider
                )));
            }
            #[cfg(windows)]
            {
                return dpapi::unprotect(&raw)
                    .map_err(|e| {
                        AnyLLMError::config(format!(
                            "The saved key for {:?} could not be decrypted: {}",
                            provider, e
                        ))
                        .with_hint(
                            "It was probably saved by a different Windows user. \
                                    Run `anyllm keys set {}` again.",
                        )
                    })
                    .and_then(|bytes| {
                        String::from_utf8(bytes)
                            .map(Some)
                            .map_err(|_| corrupted(provider))
                    });
            }
            #[cfg(not(windows))]
            return Err(corrupted(provider));
        }
        Err(AnyLLMError::config(format!(
            "The saved key for {:?} uses unknown encryption {:?}.",
            provider, scheme
        ))
        .with_hint(format!("Run `anyllm keys set {}` again.", provider)))
    }

    /// Save a key; returns the storage scheme that was used.
    pub fn set(&self, provider: &str, key: &str) -> Result<String> {
        let key = key.trim();
        if key.is_empty() {
            return Err(AnyLLMError::config(
                "The API key is empty; nothing was saved.",
            ));
        }
        if key.chars().any(char::is_whitespace) {
            return Err(AnyLLMError::config("That API key contains whitespace.")
                .with_hint("Paste the key without spaces or quotes."));
        }
        let mut scheme = SCHEME_PLAIN.to_string();
        let mut payload = key.as_bytes().to_vec();
        if dpapi_available() {
            #[cfg(windows)]
            {
                match dpapi::protect(&payload) {
                    Ok(protected) => {
                        payload = protected;
                        scheme = SCHEME_DPAPI.to_string();
                    }
                    Err(_) => {
                        scheme = SCHEME_PLAIN.to_string();
                    }
                }
            }
        }
        let mut data = self.load()?;
        if !data.contains_key("keys") {
            data.insert("keys".into(), json!({}));
        }
        if let Some(keys) = data.get_mut("keys").and_then(Value::as_object_mut) {
            keys.insert(
                provider.to_string(),
                json!({
                    "scheme": scheme,
                    "value": BASE64.encode(&payload),
                    "updated": utc_now_iso(),
                }),
            );
        }
        self.save(data)?;
        Ok(scheme)
    }

    /// Remove a saved key; returns whether one was there.
    pub fn remove(&self, provider: &str) -> Result<bool> {
        let mut data = self.load()?;
        let had = data
            .get("keys")
            .and_then(Value::as_object)
            .map(|keys| keys.contains_key(provider))
            .unwrap_or(false);
        if !had {
            return Ok(false);
        }
        if let Some(keys) = data.get_mut("keys").and_then(Value::as_object_mut) {
            keys.remove(provider);
        }
        self.save(data)?;
        Ok(true)
    }

    /// Non-secret metadata for every saved key: (scheme, updated).
    pub fn info(&self) -> Result<BTreeMap<String, (String, Option<String>)>> {
        let data = self.load()?;
        let mut result = BTreeMap::new();
        for (name, entry) in Self::keys_of(&data) {
            if let Some(entry) = entry.as_object() {
                let scheme = entry
                    .get("scheme")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
                let updated = entry
                    .get("updated")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                result.insert(name, (scheme, updated));
            }
        }
        Ok(result)
    }

    pub fn masked(&self, provider: &str) -> Result<Option<String>> {
        Ok(self
            .get(provider)?
            .filter(|v| !v.is_empty())
            .map(|v| mask_key(&v)))
    }
}

fn corrupted(provider: &str) -> AnyLLMError {
    AnyLLMError::config(format!("The saved key for {:?} is corrupted.", provider))
        .with_hint(format!("Run `anyllm keys set {}` again.", provider))
}

/// Whether the generic `ANYLLM_API_KEY` may be sent to this provider.
pub fn accepts_generic_api_key(provider: &Provider) -> bool {
    if provider.kind != OPENAI {
        return false;
    }
    if provider.name == "custom" {
        return true;
    }
    !builtins().contains_key(&provider.name)
}

/// Find a key for a provider, preferring environment variables.
pub fn resolve_key(provider: &Provider, store: &KeyStore) -> KeyResolution {
    let mut candidates: Vec<String> = vec![format!("ANYLLM_{}_API_KEY", env_slug(&provider.name))];
    candidates.extend(provider.env_vars.iter().cloned());
    if accepts_generic_api_key(provider) {
        candidates.push("ANYLLM_API_KEY".to_string());
    }
    let mut seen = std::collections::HashSet::new();
    for variable in candidates {
        if variable.is_empty() || !seen.insert(variable.clone()) {
            continue;
        }
        if let Ok(value) = std::env::var(&variable) {
            if !value.trim().is_empty() {
                return KeyResolution {
                    value: Some(value.trim().to_string()),
                    source: format!("env:{}", variable),
                    env_var: Some(variable),
                    stored: false,
                };
            }
        }
    }
    if let Ok(Some(stored)) = store.get(&provider.name) {
        if !stored.is_empty() {
            return KeyResolution {
                value: Some(stored),
                source: "saved".to_string(),
                env_var: None,
                stored: true,
            };
        }
    }
    if !provider.key_required {
        return KeyResolution {
            value: None,
            source: "not required".to_string(),
            env_var: None,
            stored: false,
        };
    }
    KeyResolution {
        value: None,
        source: "missing".to_string(),
        env_var: None,
        stored: false,
    }
}
