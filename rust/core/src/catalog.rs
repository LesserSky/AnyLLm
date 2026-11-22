//! Known providers and the custom-provider registry.
//!
//! Port of Python `anyllm/catalog.py`.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::errors::{AnyLLMError, ErrorKind, Result};

pub const OPENAI: &str = "openai";
pub const ANTHROPIC: &str = "anthropic";
pub const GEMINI: &str = "gemini";
pub const KINDS: &[&str] = &[OPENAI, ANTHROPIC, GEMINI];

/// One API endpoint plus the details needed to talk to it.
#[derive(Debug, Clone, PartialEq)]
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Self {
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

    pub fn default_model(&self) -> &str {
        self.models.first().map(String::as_str).unwrap_or("")
    }

    pub fn auth_headers(&self, key: Option<&str>) -> Vec<(String, String)> {
        match key {
            None => Vec::new(),
            Some(key) if key.is_empty() => Vec::new(),
            Some(key) => vec![(
                self.auth_header.clone(),
                format!("{}{}", self.auth_prefix, key),
            )],
        }
    }
}

/// The built-in catalog. Custom providers cannot reuse these names.
pub fn builtins() -> BTreeMap<String, Provider> {
    let mut map = BTreeMap::new();
    for provider in [
        Provider::new(
            "openai",
            "OpenAI",
            OPENAI,
            "https://api.openai.com/v1",
            &["OPENAI_API_KEY"],
            &["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            true,
            false,
            "https://platform.openai.com/docs",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "anthropic",
            "Anthropic Claude",
            ANTHROPIC,
            "https://api.anthropic.com/v1",
            &["ANTHROPIC_API_KEY"],
            &["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
            true,
            false,
            "https://docs.anthropic.com",
            "x-api-key",
            "",
        ),
        Provider::new(
            "google",
            "Google Gemini",
            GEMINI,
            "https://generativelanguage.googleapis.com/v1beta",
            &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            &["gemini-2.5-flash", "gemini-2.5-pro"],
            true,
            false,
            "https://ai.google.dev/gemini-api/docs",
            "x-goog-api-key",
            "",
        ),
        Provider::new(
            "deepseek",
            "DeepSeek",
            OPENAI,
            "https://api.deepseek.com/v1",
            &["DEEPSEEK_API_KEY"],
            &["deepseek-chat", "deepseek-reasoner"],
            true,
            false,
            "https://api-docs.deepseek.com",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "groq",
            "Groq",
            OPENAI,
            "https://api.groq.com/openai/v1",
            &["GROQ_API_KEY"],
            &["llama-3.3-70b-versatile"],
            true,
            false,
            "https://console.groq.com/docs",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "mistral",
            "Mistral",
            OPENAI,
            "https://api.mistral.ai/v1",
            &["MISTRAL_API_KEY"],
            &["mistral-large-latest", "mistral-small-latest"],
            true,
            false,
            "https://docs.mistral.ai",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "xai",
            "xAI Grok",
            OPENAI,
            "https://api.x.ai/v1",
            &["XAI_API_KEY"],
            &["grok-3", "grok-3-mini"],
            true,
            false,
            "https://docs.x.ai",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "openrouter",
            "OpenRouter",
            OPENAI,
            "https://openrouter.ai/api/v1",
            &["OPENROUTER_API_KEY"],
            &["openrouter/auto"],
            true,
            false,
            "https://openrouter.ai/docs",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "openscale",
            "OpenScale",
            OPENAI,
            "https://api.openscale.so/v1",
            &["OPENSCALE_API_KEY"],
            &[
                "deepseek/deepseek-v4.1-flash",
                "z-ai/glm-5.3-flash",
                "deepseek/deepseek-v4.1-flash:fast",
            ],
            true,
            false,
            "https://openscale.so/docs",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "together",
            "Together AI",
            OPENAI,
            "https://api.together.xyz/v1",
            &["TOGETHER_API_KEY"],
            &["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
            true,
            false,
            "https://docs.together.ai",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "ollama",
            "Ollama (local)",
            OPENAI,
            "http://127.0.0.1:11434/v1",
            &[],
            &["llama3.2"],
            false,
            true,
            "https://github.com/ollama/ollama/blob/main/docs/openai.md",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "lmstudio",
            "LM Studio (local)",
            OPENAI,
            "http://127.0.0.1:1234/v1",
            &[],
            &[],
            false,
            true,
            "https://lmstudio.ai/docs",
            "Authorization",
            "Bearer ",
        ),
        Provider::new(
            "custom",
            "Custom OpenAI-compatible endpoint",
            OPENAI,
            "",
            &["ANYLLM_API_KEY"],
            &[],
            true,
            false,
            "",
            "Authorization",
            "Bearer ",
        ),
    ] {
        map.insert(provider.name.clone(), provider);
    }
    map
}

/// Lowercase and trim a provider name.
pub fn normalize_name(name: &str) -> String {
    name.trim().to_lowercase()
}

/// Validate a name for a custom provider (Python `validate_custom_name`).
pub fn validate_custom_name(name: &str) -> Result<String> {
    let cleaned = normalize_name(name);
    if !valid_name_shape(&cleaned) {
        return Err(
            AnyLLMError::config(format!("Invalid provider name {:?}.", name)).with_hint(
                "Use 1-39 letters, digits, dots, dashes, or underscores, \
                 starting with a letter or digit.",
            ),
        );
    }
    if builtins().contains_key(&cleaned) {
        return Err(AnyLLMError::config(format!(
            "{:?} is a built-in provider and cannot be redefined.",
            cleaned
        ))
        .with_hint("Pick another name, or use it directly with --provider."));
    }
    Ok(cleaned)
}

/// `^[a-z0-9][a-z0-9._-]{0,38}$`
fn valid_name_shape(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit() => {}
        _ => return false,
    }
    let rest: Vec<char> = chars.collect();
    rest.len() <= 38
        && rest.iter().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '.' || *c == '_' || *c == '-'
        })
}

/// Validate an http(s) base URL (Python `validate_base_url`).
pub fn validate_base_url(url: &str) -> Result<String> {
    let value = url.trim().trim_end_matches('/');
    if value.is_empty() {
        return Err(AnyLLMError::config("A base URL is required.")
            .with_hint("Example: --base-url https://api.example.com/v1"));
    }
    let (scheme, rest) = match value.split_once("://") {
        Some(parts) => parts,
        None => ("", ""),
    };
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    if (scheme != "http" && scheme != "https") || host.is_empty() {
        return Err(AnyLLMError::config(format!("Invalid base URL {:?}.", url))
            .with_hint("Use an http:// or https:// URL, for example https://api.example.com/v1"));
    }
    Ok(value.to_string())
}

/// Build a `Provider` from a stored JSON spec (Python `_provider_from_spec`).
pub fn provider_from_spec(name: &str, spec: &Value) -> Result<Provider> {
    let spec_map = spec.as_object().ok_or_else(|| {
        AnyLLMError::config(format!(
            "Provider {:?} in the config is not an object.",
            name
        ))
        .with_hint("Fix the custom_providers entry in the config file, or remove it.")
    })?;
    let kind = spec_map
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or(OPENAI)
        .trim()
        .to_lowercase();
    let kind = if kind.is_empty() {
        OPENAI.to_string()
    } else {
        kind
    };
    if !KINDS.contains(&kind.as_str()) {
        return Err(AnyLLMError::config(format!(
            "Provider {:?} has unsupported kind {:?}.",
            name, kind
        ))
        .with_hint(format!("Use one of: {}", KINDS.join(", "))));
    }
    let base_url_raw = spec_map
        .get("base_url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let base_url = validate_base_url(&base_url_raw).map_err(|exc| {
        AnyLLMError::config(format!("Provider {:?} has an invalid base URL.", name))
            .with_hint(exc.hint.unwrap_or_default())
    })?;

    let string_list = |key: &str| -> Result<Vec<String>> {
        match spec_map.get(key) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(items)) => {
                if items.iter().all(|item| item.is_string()) {
                    Ok(items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect())
                } else {
                    Err(AnyLLMError::config(format!(
                        "Provider {:?} has an invalid {:?} list in the config.",
                        name, key
                    )))
                }
            }
            Some(_) => Err(AnyLLMError::config(format!(
                "Provider {:?} has an invalid {:?} list in the config.",
                name, key
            ))),
        }
    };
    let env = string_list("env")?;
    let models = string_list("models")?;
    let label = spec_map
        .get("label")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(name)
        .to_string();
    let key_required = match spec_map.get("key_required") {
        None => true,
        Some(value) => value.as_bool().unwrap_or(true),
    };
    let auth_header = spec_map
        .get("auth_header")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("Authorization")
        .to_string();
    let auth_prefix = match spec_map.get("auth_prefix") {
        Some(Value::String(s)) => s.clone(),
        _ => "Bearer ".to_string(),
    };
    Ok(Provider {
        name: name.to_string(),
        label,
        kind,
        base_url,
        env_vars: env,
        models,
        key_required,
        local: false,
        docs: String::new(),
        auth_header,
        auth_prefix,
    })
}

/// Merge built-ins with custom providers stored in the config.
pub fn load_providers(config: &Config) -> Result<BTreeMap<String, Provider>> {
    let mut providers = builtins();
    let custom = config.get("custom_providers");
    if custom.is_null() {
        return Ok(providers);
    }
    let custom = match custom {
        Value::Object(map) => map,
        _ => {
            return Err(
                AnyLLMError::config("'custom_providers' in the config must be an object.")
                    .with_hint(
                        "Run `anyllm config path` and fix the file, or delete it to start over.",
                    ),
            );
        }
    };
    for (raw_name, spec) in &custom {
        let name = normalize_name(raw_name);
        if name.is_empty() || builtins().contains_key(&name) {
            return Err(AnyLLMError::config(format!(
                "Custom provider name {:?} is reserved or empty.",
                raw_name
            ))
            .with_hint("Remove or rename that entry in the config file."));
        }
        let provider = provider_from_spec(&name, spec)?;
        providers.insert(name, provider);
    }
    Ok(providers)
}

/// Look up one provider by (case-insensitive) name.
pub fn get_provider(name: &str, providers: &BTreeMap<String, Provider>) -> Result<Provider> {
    let key = normalize_name(name);
    if let Some(provider) = providers.get(&key) {
        return Ok(provider.clone());
    }
    let available: Vec<String> = providers.keys().cloned().collect();
    Err(
        AnyLLMError::config(format!("Unknown provider {:?}.", name)).with_hint(format!(
            "Available providers: {}. Add your own with \
         `anyllm providers add NAME --base-url URL`.",
            available.join(", ")
        )),
    )
}

/// Options for `providers add` (Python `provider_spec_from_args` inputs).
pub struct ProviderSpecArgs {
    pub kind: String,
    pub base_url: String,
    pub env: Vec<String>,
    pub models: Vec<String>,
    pub no_key: bool,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
    pub label: Option<String>,
}

/// Build the JSON spec stored for a custom provider.
pub fn provider_spec_from_args(name: &str, args: &ProviderSpecArgs) -> Result<Value> {
    let kind = args.kind.clone();
    let auth_header = if let Some(header) = &args.auth_header {
        header.clone()
    } else if kind == ANTHROPIC {
        "x-api-key".to_string()
    } else if kind == GEMINI {
        "x-goog-api-key".to_string()
    } else {
        "Authorization".to_string()
    };
    let auth_prefix = if let Some(prefix) = &args.auth_prefix {
        prefix.clone()
    } else if kind == ANTHROPIC || kind == GEMINI {
        String::new()
    } else {
        "Bearer ".to_string()
    };
    Ok(json!({
        "kind": kind,
        "base_url": validate_base_url(&args.base_url)?,
        "env": args.env,
        "models": args.models,
        "key_required": !args.no_key,
        "auth_header": auth_header,
        "auth_prefix": auth_prefix,
        "label": args.label.clone().unwrap_or_else(|| name.to_string()),
    }))
}

/// Store a custom-provider spec map back into a `Map<String, Value>`.
pub fn spec_map(spec: &Value) -> Option<Map<String, Value>> {
    spec.as_object().cloned()
}

/// Error kind re-export used by callers building provider errors.
pub fn unknown_kind_error(kind: &str) -> AnyLLMError {
    AnyLLMError::new(
        ErrorKind::Api,
        format!("Unsupported provider kind {:?}.", kind),
    )
}
