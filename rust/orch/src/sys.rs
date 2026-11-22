//! Self-contained stand-ins for the `anyllm` core types the orchestration
//! layer needs (errors, providers, usage, chat backend, key resolution).
//!
//! `anyllm-core` is being written in parallel and its API will move, so this
//! crate keeps its own minimal copies here. `rust/cli` will swap them for the
//! real ones once both crates settle.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The provider catalog keyed by provider name.
pub type ProviderMap = BTreeMap<String, Provider>;

/// Result alias used across the orchestration layer.
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Errors

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Usage,
    Config,
    Auth,
    NotFound,
    RateLimit,
    Network,
    Api,
    Provider,
}

/// Error with a short message and an optional user hint, like `errors.py`.
#[derive(Clone, Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub hint: Option<String>,
    pub status: Option<u16>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl Error {
    fn new(kind: ErrorKind, message: impl Into<String>, hint: Option<String>) -> Error {
        Error { kind, message: message.into(), hint, status: None, param: None, code: None }
    }

    pub fn usage(message: impl Into<String>, hint: Option<String>) -> Error {
        Error::new(ErrorKind::Usage, message, hint)
    }

    pub fn usage_hint(message: impl Into<String>, hint: impl Into<String>) -> Error {
        Error::new(ErrorKind::Usage, message, Some(hint.into()))
    }

    pub fn config(message: impl Into<String>, hint: Option<String>) -> Error {
        Error::new(ErrorKind::Config, message, hint)
    }

    pub fn config_hint(message: impl Into<String>, hint: impl Into<String>) -> Error {
        Error::new(ErrorKind::Config, message, Some(hint.into()))
    }

    pub fn auth(message: impl Into<String>, hint: Option<String>) -> Error {
        Error::new(ErrorKind::Auth, message, hint)
    }

    pub fn api(message: impl Into<String>, hint: Option<String>) -> Error {
        Error::new(ErrorKind::Api, message, hint)
    }

    pub fn provider(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Provider, message, None)
    }

    pub fn network(message: impl Into<String>, hint: Option<String>) -> Error {
        Error::new(ErrorKind::Network, message, hint)
    }

    pub fn is_usage(&self) -> bool {
        self.kind == ErrorKind::Usage
    }

    /// Errors rendered as warnings (yellow) rather than failures (red),
    /// matching `_WARN_ERRORS` in `term.py`.
    pub fn is_warn(&self) -> bool {
        matches!(self.kind, ErrorKind::RateLimit | ErrorKind::Network | ErrorKind::Usage)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// Providers (mirrors the parts of `catalog.py` that orchestration uses)

pub const OPENAI: &str = "openai";
pub const ANTHROPIC: &str = "anthropic";
pub const GEMINI: &str = "gemini";

#[derive(Clone, Debug, PartialEq)]
pub struct Provider {
    pub name: String,
    pub label: String,
    pub kind: String,
    pub base_url: String,
    pub env_vars: Vec<String>,
    pub models: Vec<String>,
    pub key_required: bool,
    pub local: bool,
    pub docs: String,
    pub auth_header: String,
    pub auth_prefix: String,
}

impl Provider {
    pub fn new(name: &str, label: &str, kind: &str, base_url: &str) -> Provider {
        Provider {
            name: name.to_string(),
            label: label.to_string(),
            kind: kind.to_string(),
            base_url: base_url.to_string(),
            env_vars: Vec::new(),
            models: Vec::new(),
            key_required: true,
            local: false,
            docs: String::new(),
            auth_header: "Authorization".to_string(),
            auth_prefix: "Bearer ".to_string(),
        }
    }

    pub fn default_model(&self) -> String {
        self.models.first().cloned().unwrap_or_default()
    }
}

fn provider(
    name: &str,
    label: &str,
    kind: &str,
    base_url: &str,
    env_vars: &[&str],
    models: &[&str],
    key_required: bool,
    local: bool,
    docs: &str,
    auth_header: &str,
    auth_prefix: &str,
) -> Provider {
    Provider {
        name: name.to_string(),
        label: label.to_string(),
        kind: kind.to_string(),
        base_url: base_url.to_string(),
        env_vars: env_vars.iter().map(|s| s.to_string()).collect(),
        models: models.iter().map(|s| s.to_string()).collect(),
        key_required,
        local,
        docs: docs.to_string(),
        auth_header: auth_header.to_string(),
        auth_prefix: auth_prefix.to_string(),
    }
}

/// The built-in provider catalog from `catalog.py`.
pub fn builtins() -> BTreeMap<String, Provider> {
    let mut map = BTreeMap::new();
    let entries = [
        provider(
            "openai", "OpenAI", OPENAI, "https://api.openai.com/v1",
            &["OPENAI_API_KEY"],
            &["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            true, false, "https://platform.openai.com/docs",
            "Authorization", "Bearer ",
        ),
        provider(
            "anthropic", "Anthropic Claude", ANTHROPIC, "https://api.anthropic.com/v1",
            &["ANTHROPIC_API_KEY"],
            &["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
            true, false, "https://docs.anthropic.com",
            "x-api-key", "",
        ),
        provider(
            "google", "Google Gemini", GEMINI, "https://generativelanguage.googleapis.com/v1beta",
            &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            &["gemini-2.5-flash", "gemini-2.5-pro"],
            true, false, "https://ai.google.dev/gemini-api/docs",
            "x-goog-api-key", "",
        ),
        provider(
            "deepseek", "DeepSeek", OPENAI, "https://api.deepseek.com/v1",
            &["DEEPSEEK_API_KEY"],
            &["deepseek-chat", "deepseek-reasoner"],
            true, false, "https://api-docs.deepseek.com",
            "Authorization", "Bearer ",
        ),
        provider(
            "groq", "Groq", OPENAI, "https://api.groq.com/openai/v1",
            &["GROQ_API_KEY"],
            &["llama-3.3-70b-versatile"],
            true, false, "https://console.groq.com/docs",
            "Authorization", "Bearer ",
        ),
        provider(
            "mistral", "Mistral", OPENAI, "https://api.mistral.ai/v1",
            &["MISTRAL_API_KEY"],
            &["mistral-large-latest", "mistral-small-latest"],
            true, false, "https://docs.mistral.ai",
            "Authorization", "Bearer ",
        ),
        provider(
            "xai", "xAI Grok", OPENAI, "https://api.x.ai/v1",
            &["XAI_API_KEY"],
            &["grok-3", "grok-3-mini"],
            true, false, "https://docs.x.ai",
            "Authorization", "Bearer ",
        ),
        provider(
            "openrouter", "OpenRouter", OPENAI, "https://openrouter.ai/api/v1",
            &["OPENROUTER_API_KEY"],
            &["openrouter/auto"],
            true, false, "https://openrouter.ai/docs",
            "Authorization", "Bearer ",
        ),
        provider(
            "openscale", "OpenScale", OPENAI, "https://api.openscale.so/v1",
            &["OPENSCALE_API_KEY"],
            &[
                "deepseek/deepseek-v4.1-flash",
                "z-ai/glm-5.3-flash",
                "deepseek/deepseek-v4.1-flash:fast",
            ],
            true, false, "https://openscale.so/docs",
            "Authorization", "Bearer ",
        ),
        provider(
            "together", "Together AI", OPENAI, "https://api.together.xyz/v1",
            &["TOGETHER_API_KEY"],
            &["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
            true, false, "https://docs.together.ai",
            "Authorization", "Bearer ",
        ),
        provider(
            "ollama", "Ollama (local)", OPENAI, "http://127.0.0.1:11434/v1",
            &[], &["llama3.2"],
            false, true,
            "https://github.com/ollama/ollama/blob/main/docs/openai.md",
            "Authorization", "Bearer ",
        ),
        provider(
            "lmstudio", "LM Studio (local)", OPENAI, "http://127.0.0.1:1234/v1",
            &[], &[],
            false, true, "https://lmstudio.ai/docs",
            "Authorization", "Bearer ",
        ),
        provider(
            "custom", "Custom OpenAI-compatible endpoint", OPENAI, "",
            &["ANYLLM_API_KEY"], &[],
            true, false, "",
            "Authorization", "Bearer ",
        ),
    ];
    for entry in entries {
        map.insert(entry.name.clone(), entry);
    }
    map
}

pub fn normalize_name(name: &str) -> String {
    name.trim().to_lowercase()
}

/// Look up a provider, or fail with the catalog's error text.
pub fn get_provider(name: &str, providers: &BTreeMap<String, Provider>) -> Result<Provider> {
    let key = normalize_name(name);
    if let Some(found) = providers.get(&key) {
        return Ok(found.clone());
    }
    let available = providers.keys().cloned().collect::<Vec<_>>().join(", ");
    Err(Error::config(
        format!("Unknown provider {:?}.", name),
        Some(format!(
            "Available providers: {}. Add your own with `anyllm providers add NAME --base-url URL`.",
            available
        )),
    ))
}

/// A minimal URL validator matching `catalog.validate_base_url`.
/// Returns (scheme, host, port, path, query, fragment, userinfo).
#[derive(Clone, Debug, Default)]
pub struct UrlParts {
    pub scheme: String,
    pub netloc: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

/// Split a URL. No external crates: http(s) URLs and simple netlocs only.
pub fn urlsplit(url: &str) -> UrlParts {
    let mut parts = UrlParts::default();
    let value = url.trim();
    let (scheme, rest) = match value.find("://") {
        Some(index) => (value[..index].to_string(), &value[index + 3..]),
        None => return UrlParts { path: value.to_string(), ..Default::default() },
    };
    parts.scheme = scheme.to_lowercase();
    let (authority, tail) = match rest.find(['/', '?', '#']) {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, ""),
    };
    parts.netloc = authority.to_string();
    let mut userinfo = String::new();
    let mut hostport = authority;
    if let Some(at) = authority.rfind('@') {
        userinfo = authority[..at].to_string();
        hostport = &authority[at + 1..];
    }
    if !userinfo.is_empty() {
        match userinfo.find(':') {
            Some(index) => {
                parts.username = Some(userinfo[..index].to_string());
                parts.password = Some(userinfo[index + 1..].to_string());
            }
            None => parts.username = Some(userinfo),
        }
    }
    let (host, port) = match hostport.rfind(':') {
        Some(index) if !hostport.starts_with('[') || hostport[index + 1..].chars().all(|c| c.is_ascii_digit()) => {
            let port_text = &hostport[index + 1..];
            let port = port_text.parse::<u16>().ok();
            if port.is_some() {
                (&hostport[..index], port)
            } else {
                (hostport, None)
            }
        }
        _ => (hostport, None),
    };
    parts.host = host.trim_start_matches('[').trim_end_matches(']').to_string();
    parts.port = port;
    let mut tail_text = tail;
    if let Some(hash) = tail_text.find('#') {
        parts.fragment = tail_text[hash + 1..].to_string();
        tail_text = &tail_text[..hash];
    }
    if let Some(question) = tail_text.find('?') {
        parts.query = tail_text[question + 1..].to_string();
        tail_text = &tail_text[..question];
    }
    parts.path = tail_text.to_string();
    parts
}

pub fn validate_base_url(url: &str) -> Result<String> {
    let value = url.trim().trim_end_matches('/').to_string();
    if value.is_empty() {
        return Err(Error::config(
            "A base URL is required.",
            Some("Example: --base-url https://api.example.com/v1".to_string()),
        ));
    }
    let parts = urlsplit(&value);
    if (parts.scheme != "http" && parts.scheme != "https") || parts.netloc.is_empty() {
        return Err(Error::config(
            format!("Invalid base URL {:?}.", url),
            Some("Use an http:// or https:// URL, for example https://api.example.com/v1".to_string()),
        ));
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Usage / Chunk (mirrors `adapters.py` data types)

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
}

impl Usage {
    /// Keep the latest (usually cumulative) values from a stream.
    pub fn update(&mut self, other: &Usage) {
        let pick = |current: Option<i64>, incoming: Option<i64>| match (current, incoming) {
            (None, incoming) => incoming,
            (current, None) => current,
            (Some(current), Some(incoming)) => Some(current.max(incoming)),
        };
        self.input_tokens = pick(self.input_tokens, other.input_tokens);
        self.output_tokens = pick(self.output_tokens, other.output_tokens);
        self.reasoning_tokens = pick(self.reasoning_tokens, other.reasoning_tokens);
        self.cached_tokens = pick(self.cached_tokens, other.cached_tokens);
    }

    /// Add two usage reports, for session totals.
    pub fn combined(&self, other: &Usage) -> Usage {
        let add = |left: Option<i64>, right: Option<i64>| match (left, right) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        Usage {
            input_tokens: add(self.input_tokens, other.input_tokens),
            output_tokens: add(self.output_tokens, other.output_tokens),
            reasoning_tokens: add(self.reasoning_tokens, other.reasoning_tokens),
            cached_tokens: add(self.cached_tokens, other.cached_tokens),
        }
    }

    pub fn total(&self) -> Option<i64> {
        if self.input_tokens.is_none() && self.output_tokens.is_none() {
            return None;
        }
        Some(self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0))
    }

    /// Same shape as `adapters.Usage.to_dict`.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::json!({
            "input": self.input_tokens,
            "output": self.output_tokens,
            "reasoning": self.reasoning_tokens,
            "cached": self.cached_tokens,
            "total": self.total(),
        })
    }

    pub fn from_value(raw: &serde_json::Value) -> Usage {
        if !raw.is_object() {
            return Usage::default();
        }
        let int = |key: &str| -> Option<i64> {
            match raw.get(key) {
                Some(serde_json::Value::Number(number)) => number.as_i64(),
                Some(serde_json::Value::String(text)) if !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()) => {
                    text.parse::<i64>().ok()
                }
                _ => None,
            }
        };
        Usage {
            input_tokens: int("input"),
            output_tokens: int("output"),
            reasoning_tokens: int("reasoning"),
            cached_tokens: int("cached"),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.input_tokens.is_none()
            && self.output_tokens.is_none()
            && self.reasoning_tokens.is_none()
            && self.cached_tokens.is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCallDelta {
    pub index: i64,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub reasoning: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub tool_call: Option<ToolCallDelta>,
}

// ---------------------------------------------------------------------------
// Messages, sessions, and the chat backend seam

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message { role: role.to_string(), content: content.to_string(), tool_calls: None, tool_call_id: None }
    }
}

/// One completion request as the orchestration layer issues it.
/// Contains no credential-echoing fields; the key lives only in `credential`.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub provider: Provider,
    pub model: String,
    pub base_url: String,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub credential: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub timeout: f64,
    pub tools: Option<serde_json::Value>,
    /// Absolute deadline, enforced between chunks.
    pub deadline: Option<std::time::Instant>,
    /// Cooperative cancellation flag.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// The one point where orchestration touches a provider. `rust/cli` will
/// implement this with the real transport; tests implement it with fakes.
pub trait ChatBackend: Sync {
    fn stream(&self, request: &ChatRequest) -> Result<Vec<Chunk>>;
}

/// A live chat session, as seen by the orchestration layer. Mirrors the
/// fields `targets.py`, `team.py`, and `budgets.py` read and mutate.
#[derive(Clone, Debug)]
pub struct Session {
    pub provider: Provider,
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub key_source: String,
    pub system: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub timeout: f64,
    pub show_reasoning: bool,
    pub target_alias: Option<String>,
    /// In-memory only. Set by an explicit `--base-url` on this process.
    pub explicit_endpoint: Option<String>,
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub turns: u64,
}

impl Session {
    pub fn new(provider: Provider, model: &str) -> Session {
        Session {
            base_url: provider.base_url.clone(),
            provider,
            model: model.to_string(),
            api_key: None,
            key_source: String::new(),
            system: None,
            temperature: None,
            max_tokens: None,
            timeout: 120.0,
            show_reasoning: false,
            target_alias: None,
            explicit_endpoint: None,
            messages: Vec::new(),
            usage: Usage::default(),
            turns: 0,
        }
    }

    /// Append one finished exchange without calling a provider
    /// (`session.append_exchange`, used by `/team adopt`).
    pub fn append_exchange(
        &mut self,
        user_text: &str,
        assistant_text: &str,
        usage: Option<&Usage>,
    ) -> Result<()> {
        if user_text.trim().is_empty() {
            return Err(Error::usage("The message is empty.", None));
        }
        if assistant_text.trim().is_empty() {
            return Err(Error::usage("The answer is empty.", None));
        }
        self.messages.push(Message::new("user", user_text));
        self.messages.push(Message::new("assistant", assistant_text));
        if let Some(usage) = usage {
            self.usage = self.usage.combined(usage);
        }
        self.turns += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Key resolution (mirrors the parts of `keystore.py` orchestration uses)

/// Reads stored API keys. Implemented by the core keystore; tests fake it.
pub trait KeyStore: Sync {
    fn get(&self, name: &str) -> Result<Option<String>>;
}

/// An empty keystore, for calls that never read saved keys.
pub struct EmptyKeyStore;

impl KeyStore for EmptyKeyStore {
    fn get(&self, _name: &str) -> Result<Option<String>> {
        Ok(None)
    }
}

#[derive(Clone, Debug)]
pub struct KeyResolution {
    pub value: Option<String>,
    pub source: String,
    pub env_var: Option<String>,
    pub stored: bool,
}

impl KeyResolution {
    pub fn present(&self) -> bool {
        self.value.as_deref().map(|value| !value.is_empty()).unwrap_or(false)
    }

    pub fn display(&self) -> String {
        if let Some(env_var) = &self.env_var {
            return format!("env:{}", env_var);
        }
        if self.stored {
            return "saved".to_string();
        }
        if self.value.is_some() {
            return "configured".to_string();
        }
        "missing".to_string()
    }
}

/// `keystore.mask_key`.
pub fn mask_key(value: &str) -> String {
    if value.is_empty() {
        return "(empty)".to_string();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    format!("{}...{}", value[..4].to_string(), value[value.len() - 4..].to_string())
}

pub fn env_slug(name: &str) -> String {
    let mut slug = String::new();
    let mut underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_uppercase());
            underscore = false;
        } else if !underscore {
            slug.push('_');
            underscore = true;
        }
    }
    while slug.starts_with('_') {
        slug.remove(0);
    }
    while slug.ends_with('_') {
        slug.pop();
    }
    slug
}

/// Whether the generic `ANYLLM_API_KEY` may be sent to `provider`.
pub fn accepts_generic_api_key(provider: &Provider) -> bool {
    if provider.kind != OPENAI {
        return false;
    }
    if provider.name == "custom" {
        return true;
    }
    !builtins().contains_key(&provider.name)
}

/// Find a key for `provider`, preferring environment variables.
pub fn resolve_key(provider: &Provider, store: &dyn KeyStore) -> KeyResolution {
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
        return KeyResolution { value: Some(stored), source: "saved".to_string(), env_var: None, stored: true };
    }
    if !provider.key_required {
        return KeyResolution { value: None, source: "not required".to_string(), env_var: None, stored: false };
    }
    KeyResolution { value: None, source: "missing".to_string(), env_var: None, stored: false }
}

// ---------------------------------------------------------------------------
// Config-directory plumbing (mirrors `config.py` paths, for feature files)

/// Directory for config, keys, and feature files. `ANYLLM_HOME` overrides
/// the platform default.
pub fn config_dir() -> std::path::PathBuf {
    if let Ok(home) = std::env::var("ANYLLM_HOME") {
        if !home.trim().is_empty() {
            return std::path::PathBuf::from(expand_user(&home));
        }
    }
    if cfg!(windows) {
        let base = std::env::var("APPDATA").unwrap_or_default();
        let root = if base.is_empty() {
            home_dir().join("AppData").join("Roaming")
        } else {
            std::path::PathBuf::from(base)
        };
        return root.join("anyllm");
    }
    let base = std::env::var("XDG_CONFIG_HOME").unwrap_or_default();
    let root = if base.is_empty() {
        home_dir().join(".config")
    } else {
        std::path::PathBuf::from(base)
    };
    root.join("anyllm")
}

fn home_dir() -> std::path::PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
}

fn expand_user(path: &str) -> String {
    if path == "~" {
        return home_dir().to_string_lossy().into_owned();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir().join(rest).to_string_lossy().into_owned();
    }
    if cfg!(windows) {
        if let Some(rest) = path.strip_prefix("~\\") {
            return home_dir().join(rest).to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

// ---------------------------------------------------------------------------
// Terminal style (minimal stand-in for `term.py`, enough for orchestration)

#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub enabled: bool,
}

impl Default for Style {
    fn default() -> Style {
        Style { enabled: true }
    }
}

impl Style {
    pub fn new(enabled: bool) -> Style {
        Style { enabled }
    }

    fn wrap(&self, text: &str, code: &str) -> String {
        if !self.enabled || text.is_empty() {
            return text.to_string();
        }
        format!("{}{}\u{1b}[0m", code, text)
    }

    pub fn bold(&self, text: &str) -> String {
        self.wrap(text, "\u{1b}[1m")
    }

    pub fn dim(&self, text: &str) -> String {
        self.wrap(text, "\u{1b}[2m")
    }

    pub fn yellow(&self, text: &str) -> String {
        self.wrap(text, "\u{1b}[33m")
    }

    pub fn green(&self, text: &str) -> String {
        self.wrap(text, "\u{1b}[32m")
    }

    pub fn red(&self, text: &str) -> String {
        self.wrap(text, "\u{1b}[1;31m")
    }
}

// ---------------------------------------------------------------------------
// Small utilities

/// `uuid.uuid4().hex`-shaped identifier: 32 hex characters, unique per
/// process. Seeded from wall clock, a counter, and the process id.
pub fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut state = nanos ^ (count.wrapping_mul(0x9E3779B97F4A7C15)) ^ (pid << 32);
    let mut out = String::with_capacity(32);
    for _ in 0..4 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D049BB133111EB);
        mixed ^= mixed >> 31;
        out.push_str(&format!("{:08x}", (mixed >> 32) as u32));
    }
    out
}

/// A POSIX-ish `shlex.split` for `/budget`, `/tools`, and `/team` lines.
/// Supports single and double quotes and a backslash escape; comments are
/// not treated specially because a prompt may contain `#`.
pub fn split_words(text: &str) -> std::result::Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut has_word = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                if has_word {
                    words.push(std::mem::take(&mut current));
                    has_word = false;
                }
            }
            '\'' => {
                has_word = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(other) => current.push(other),
                        None => return Err("No closing quotation".to_string()),
                    }
                }
            }
            '"' => {
                has_word = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(escaped) => current.push(escaped),
                            None => return Err("No closing quotation".to_string()),
                        },
                        Some(other) => current.push(other),
                        None => return Err("No closing quotation".to_string()),
                    }
                }
            }
            '\\' => {
                has_word = true;
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
            }
            other => {
                has_word = true;
                current.push(other);
            }
        }
    }
    if has_word {
        words.push(current);
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_key_matches_python() {
        assert_eq!(mask_key(""), "(empty)");
        assert_eq!(mask_key("short"), "*****");
        assert_eq!(mask_key("sk-abcdefghijklmnop"), "sk-a...mnop");
    }

    #[test]
    fn env_slug_matches_python() {
        assert_eq!(env_slug("open.scale"), "OPEN_SCALE");
        assert_eq!(env_slug("lab-2"), "LAB_2");
        assert_eq!(env_slug("-x-"), "X");
    }

    #[test]
    fn validate_base_url_matches_python() {
        assert_eq!(validate_base_url(" https://api.example.com/v1/ ").unwrap(), "https://api.example.com/v1");
        assert!(validate_base_url("").is_err());
        assert!(validate_base_url("ftp://example.com").is_err());
        assert!(validate_base_url("https://").is_err());
    }

    #[test]
    fn urlsplit_handles_userinfo_and_query() {
        let parts = urlsplit("https://user:pw@api.openai.com:8443/v1?api-version=2024-10-21#frag");
        assert_eq!(parts.scheme, "https");
        assert_eq!(parts.username.as_deref(), Some("user"));
        assert_eq!(parts.password.as_deref(), Some("pw"));
        assert_eq!(parts.host, "api.openai.com");
        assert_eq!(parts.port, Some(8443));
        assert_eq!(parts.path, "/v1");
        assert_eq!(parts.query, "api-version=2024-10-21");
        assert_eq!(parts.fragment, "frag");
    }

    #[test]
    fn split_words_quoting() {
        assert_eq!(split_words("a 'b c' \"d\\\"e\"").unwrap(), vec!["a", "b c", "d\"e"]);
        assert!(split_words("'unclosed").is_err());
        assert!(split_words("").unwrap().is_empty());
    }

    #[test]
    fn ids_are_unique_and_hex() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn usage_update_keeps_max_and_combined_adds() {
        let mut base = Usage { input_tokens: Some(10), ..Default::default() };
        base.update(&Usage { input_tokens: Some(5), output_tokens: Some(3), ..Default::default() });
        assert_eq!(base.input_tokens, Some(10));
        assert_eq!(base.output_tokens, Some(3));
        let sum = base.combined(&Usage { input_tokens: Some(2), ..Default::default() });
        assert_eq!(sum.input_tokens, Some(12));
    }

    #[test]
    fn append_exchange_updates_turns_and_usage() {
        let mut session = Session::new(builtins()["openai"].clone(), "gpt-4o-mini");
        let usage = Usage { input_tokens: Some(3), output_tokens: Some(2), ..Default::default() };
        session.append_exchange("hi", "hello", Some(&usage)).unwrap();
        assert_eq!(session.turns, 1);
        assert_eq!(session.usage.input_tokens, Some(3));
        assert!(session.append_exchange(" ", "x", None).is_err());
    }

    #[test]
    fn accepts_generic_api_key_gates_builtin_hosts() {
        assert!(accepts_generic_api_key(&builtins()["custom"]));
        assert!(!accepts_generic_api_key(&builtins()["openai"]));
        let mut lab = Provider::new("lab", "Lab", OPENAI, "http://lab.internal/v1");
        lab.env_vars.push("LAB_API_KEY".to_string());
        assert!(accepts_generic_api_key(&lab));
        assert!(!accepts_generic_api_key(&Provider::new("a", "A", ANTHROPIC, "u")));
    }
}
