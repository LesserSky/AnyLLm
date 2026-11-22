//! Named provider/model/base-URL targets. Port of `anyllm.targets`.
//!
//! A target stores no secrets. Credentials are resolved at request time. A
//! target whose URL is not the provider's configured host does not inherit
//! that provider's saved key or usual environment variable. That stops a key
//! for api.openai.com from being sent to some other host just because the
//! provider name is `openai`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{json, Map, Value};

use crate::seam::{feature_path, load_feature, save_feature, Repl, ReplRegistry};
use crate::sys::{
    env_slug, get_provider, mask_key, resolve_key, urlsplit, validate_base_url, Error, KeyStore,
    Provider, ProviderMap, Result, Session, Style,
};

const LOCAL_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

/// One concrete endpoint. `alias` is a local label, not a secret.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub alias: Option<String>,
}

impl Target {
    pub fn new(provider: &str, model: &str, base_url: &str, alias: Option<&str>) -> Target {
        Target {
            provider: provider.to_string(),
            model: model.to_string(),
            base_url: base_url.to_string(),
            alias: alias.map(|value| value.to_string()),
        }
    }
}

/// Everything `targets.py` stores for one alias. `base_url` is None when the
/// provider's own endpoint is used.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetSpec {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
}

impl TargetSpec {
    fn to_value(&self) -> Value {
        json!({
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        })
    }

    fn from_value(name: &str, spec: &Value, path: &std::path::Path) -> Result<TargetSpec> {
        let malformed = |message: String| {
            Error::config(message, Some(format!("Fix or remove {}.", path.display())))
        };
        let object = match spec.as_object() {
            Some(map) => map,
            None => return Err(malformed(format!("Target {:?} is malformed.", name))),
        };
        let provider = object.get("provider").and_then(Value::as_str).unwrap_or("");
        if provider.trim().is_empty() {
            return Err(malformed(format!("Target {:?} has no provider.", name)));
        }
        let model = object.get("model").and_then(Value::as_str).unwrap_or("");
        if model.trim().is_empty() {
            return Err(malformed(format!("Target {:?} has no model.", name)));
        }
        let base_url = match object.get("base_url") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text.clone()),
            Some(_) => {
                return Err(malformed(format!(
                    "Target {:?} has an invalid base URL.",
                    name
                )))
            }
        };
        Ok(TargetSpec {
            provider: provider.to_string(),
            model: model.to_string(),
            base_url,
        })
    }
}

/// A resolved target plus the credential it may use.
#[derive(Clone, Debug)]
pub struct ResolvedTarget {
    pub target: Target,
    pub provider: Provider,
    pub credential: Option<String>,
    pub credential_source: String,
    pub host_is_default: bool,
    pub http_warning: Option<String>,
}

/// The name of the alias that is also a provider name, if any.
type Collision = Option<String>;

pub fn targets_path() -> PathBuf {
    feature_path("targets.json")
}

/// Load every target or fail. A bad entry rejects the whole file.
/// Mirrors `targets.load_targets`.
pub fn load_targets(providers: &ProviderMap) -> Result<BTreeMap<String, TargetSpec>> {
    let path = targets_path();
    let data = load_feature(&path, "targets")?;
    let targets = match data.get("targets") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(_) => {
            return Err(Error::config(
                "targets.json is malformed.",
                Some(format!(
                    "'targets' must be an object. Fix or remove {}.",
                    path.display()
                )),
            ))
        }
    };
    // Validate the full map before returning any of it.
    let mut checked: BTreeMap<String, TargetSpec> = BTreeMap::new();
    for (raw_name, spec) in &targets {
        let alias = validate_target_alias(raw_name, providers)?;
        let parsed = TargetSpec::from_value(raw_name, spec, &path)?;
        if let Some(existing) = checked.get(&alias) {
            return Err(Error::config(
                format!(
                    "Target names {:?} and {:?} normalize to the same alias.",
                    existing.provider, alias
                ),
                Some(format!("Fix or remove {}.", path.display())),
            ));
        }
        checked.insert(alias, parsed);
    }
    Ok(checked)
}

/// Persist the target map, adding the version marker.
pub fn save_targets(data: &BTreeMap<String, TargetSpec>) -> Result<()> {
    let mut targets = Map::new();
    for (alias, spec) in data {
        targets.insert(alias.clone(), spec.to_value());
    }
    let mut payload = Map::new();
    payload.insert("targets".to_string(), Value::Object(targets));
    save_feature(&targets_path(), payload)
}

/// Case-fold a label. This does not check provider collisions.
/// Mirrors `targets.normalize_alias`.
pub fn normalize_alias(name: &str) -> Result<String> {
    let cleaned = name.trim().to_lowercase();
    let valid = is_valid_alias(&cleaned);
    if !valid || cleaned.contains(':') {
        return Err(Error::config(
            format!("Invalid target name {:?}.", name),
            Some(
                "Use 1-32 letters, digits, dots, dashes, or underscores. \
Names cannot contain a colon."
                    .to_string(),
            ),
        ));
    }
    Ok(cleaned)
}

fn is_valid_alias(cleaned: &str) -> bool {
    let bytes = cleaned.as_bytes();
    if bytes.is_empty() || bytes.len() > 32 {
        return false;
    }
    if !bytes[0].is_ascii_alphanumeric() {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'.' || *b == b'_' || *b == b'-')
}

/// Environment variable for a target. Distinct from ANYLLM_<PROVIDER>_API_KEY.
pub fn target_env_var(alias: &str) -> String {
    format!("ANYLLM_TARGET_{}_API_KEY", env_slug(alias))
}

/// Key-store id for a target. Distinct from a provider's saved key.
pub fn target_store_name(alias: &str) -> Result<String> {
    Ok(format!("target:{}", normalize_alias(alias)?))
}

/// Return a short reason when `alias` shares a provider name or env slug.
/// Mirrors `targets.alias_collides`.
pub fn alias_collides(alias: &str, providers: &ProviderMap) -> Collision {
    let folded = alias.to_lowercase();
    let slug = env_slug(alias);
    for name in providers.keys() {
        if name.to_lowercase() == folded {
            return Some(format!("provider {:?}", name));
        }
        if env_slug(name) == slug {
            return Some(format!(
                "the environment name of provider {:?} (ANYLLM_{}_API_KEY)",
                name,
                env_slug(name)
            ));
        }
    }
    None
}

/// Mirrors `targets.validate_target_alias`.
pub fn validate_target_alias(name: &str, providers: &ProviderMap) -> Result<String> {
    let cleaned = normalize_alias(name)?;
    if let Some(collision) = alias_collides(&cleaned, providers) {
        return Err(Error::config(
            format!("Target name {:?} collides with {}.", cleaned, collision),
            Some(
                "Choose a name that is not a provider and does not normalize to a provider's \
ANYLLM_<PROVIDER>_API_KEY variable. Target keys use ANYLLM_TARGET_<NAME>_API_KEY \
or `anyllm keys set target:<name>`."
                    .to_string(),
            ),
        ));
    }
    Ok(cleaned)
}

/// Save a target. Mirrors `targets.add_target`; the store is passed in so the
/// caller supplies the real keystore.
pub fn add_target(
    name: &str,
    provider: &str,
    model: &str,
    base_url: Option<&str>,
    providers: &ProviderMap,
    store: &dyn KeyStore,
) -> Result<Target> {
    let alias = validate_target_alias(name, providers)?;
    if model.trim().is_empty() {
        return Err(Error::config(
            "A target needs a model name.",
            Some("Pass --model NAME.".to_string()),
        ));
    }
    let provider = get_provider(provider, providers)?;
    let stored_url = match base_url {
        Some(url) if !url.trim().is_empty() => Some(validate_base_url(url)?),
        _ => None,
    };
    let mut data = load_targets(providers)?;
    data.insert(
        alias.clone(),
        TargetSpec {
            provider: provider.name.clone(),
            model: model.trim().to_string(),
            base_url: stored_url,
        },
    );
    save_targets(&data)?;
    let resolved = resolve_named(&alias, providers, store, false)?;
    Ok(resolved.target)
}

/// Remove a saved target. Mirrors `targets.remove_target`.
pub fn remove_target(name: &str, providers: &ProviderMap) -> Result<()> {
    let alias = normalize_alias(name)?;
    let mut data = load_targets(providers)?;
    if data.remove(&alias).is_none() {
        return Err(Error::config(
            format!("Unknown target {:?}.", alias),
            Some("Run `anyllm targets list` to see saved names.".to_string()),
        ));
    }
    save_targets(&data)
}

pub fn list_target_specs(providers: &ProviderMap) -> Result<BTreeMap<String, TargetSpec>> {
    load_targets(providers)
}

/// Split `provider:model` at the first colon. Model IDs may contain slashes
/// and colons. Mirrors `targets.split_provider_model`.
pub fn split_provider_model(spec: &str) -> Result<(String, String)> {
    let text = spec.trim();
    if !text.contains(':') {
        return Err(Error::config(
            format!("Could not read a provider:model target from {:?}.", spec),
            Some(
                "Use provider:model, for example openscale:deepseek/deepseek-v4.1-flash, \
or a saved alias."
                    .to_string(),
            ),
        ));
    }
    let mut parts = text.splitn(2, ':');
    let provider = parts.next().unwrap_or("").trim();
    let model = parts.next().unwrap_or("").trim();
    if provider.is_empty() || model.is_empty() {
        return Err(Error::config(
            format!("Could not read a provider:model target from {:?}.", spec),
            Some("Put the provider before the first colon and the model after it.".to_string()),
        ));
    }
    Ok((provider.to_string(), model.to_string()))
}

/// Warn before sending a key to non-local plain HTTP. Localhost is allowed.
pub fn plain_http_warning(url: &str) -> Option<String> {
    let parts = urlsplit(url);
    if parts.scheme != "http" {
        return None;
    }
    let host = parts.host.to_lowercase();
    if LOCAL_HOSTS.contains(&host.as_str()) {
        return None;
    }
    Some(format!(
        "{} is plain HTTP on a non-local host. The request, including any API key, \
would not be encrypted.",
        url
    ))
}

/// Normalize a base URL for host comparison. Query strings stay significant.
/// Host case and a default port (80 or 443) do not make a different endpoint.
/// A query, fragment, userinfo, or any other port does.
pub fn canonical_endpoint(url: &str) -> Result<String> {
    let value = validate_base_url(url)?;
    let parts = urlsplit(&value);
    let scheme = parts.scheme.to_lowercase();
    let host = parts.host.to_lowercase();
    let default_port: u16 = if scheme == "https" { 443 } else { 80 };
    let host_text = if host.contains(':') {
        format!("[{}]", host)
    } else {
        host
    };
    let netloc = match parts.port {
        Some(port) if port != default_port => format!("{}:{}", host_text, port),
        _ => host_text,
    };
    let netloc = match (&parts.username, &parts.password) {
        (Some(user), Some(password)) => format!("{}:{}@{}", user, password, netloc),
        (Some(user), None) => format!("{}@{}", user, netloc),
        _ => netloc,
    };
    let mut canonical = format!("{}://{}{}", scheme, netloc, parts.path);
    if !parts.query.is_empty() {
        canonical.push('?');
        canonical.push_str(&parts.query);
    }
    if !parts.fragment.is_empty() {
        canonical.push('#');
        canonical.push_str(&parts.fragment);
    }
    Ok(canonical)
}

fn urls_match(left: &str, right: &str) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    match (canonical_endpoint(left), canonical_endpoint(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// One stderr line when an explicit `--base-url` sends a provider key
/// elsewhere. Mirrors `targets.explicit_override_warning`.
pub fn explicit_override_warning(provider: &Provider, base_url: Option<&str>) -> Option<String> {
    let base_url = base_url?;
    if base_url.is_empty() || host_uses_provider_key(provider, Some(base_url)) {
        return None;
    }
    let mut text = format!(
        "warning: --base-url {} is not {}'s configured endpoint; this command still sends the {} key there.",
        base_url, provider.name, provider.name
    );
    if plain_http_warning(base_url).is_some() {
        text.push_str(" The connection is plain HTTP, so that key is not encrypted.");
    }
    Some(text)
}

/// Read only the target namespace. Never a provider key or
/// ANYLLM_<PROVIDER>_API_KEY. Mirrors `targets._alias_credential`.
fn alias_credential(alias: Option<&str>, store: &dyn KeyStore) -> (Option<String>, String) {
    let alias = match alias {
        Some(value) if !value.trim().is_empty() => value,
        _ => return (None, String::new()),
    };
    let cleaned = match normalize_alias(alias) {
        Ok(value) => value,
        Err(_) => return (None, String::new()),
    };
    let env_name = target_env_var(&cleaned);
    if let Ok(value) = std::env::var(&env_name) {
        if !value.trim().is_empty() {
            return (Some(value.trim().to_string()), format!("env:{}", env_name));
        }
    }
    // Provider keys live under the bare provider name. Do not read that slot.
    if let Ok(Some(saved)) = target_store_name(&cleaned).and_then(|slot| store.get(&slot)) {
        return (Some(saved), "saved-target".to_string());
    }
    (None, String::new())
}

/// Return `(credential, source, host_is_default)`. A non-default host never
/// receives the builtin or saved provider key.
pub fn resolve_credentials(
    provider: &Provider,
    target: &Target,
    store: &dyn KeyStore,
) -> (Option<String>, String, bool) {
    let provider_url = provider.base_url.trim();
    let host_is_default = !provider_url.is_empty() && urls_match(&target.base_url, provider_url);
    let custom_endpoint = provider_url.is_empty();
    let (alias_key, alias_source) = alias_credential(target.alias.as_deref(), store);
    if let Some(key) = alias_key {
        return (Some(key), alias_source, host_is_default || custom_endpoint);
    }
    if host_is_default || custom_endpoint {
        let resolution = resolve_key(provider, store);
        return (
            resolution.value,
            resolution.source,
            host_is_default || custom_endpoint,
        );
    }
    // Named builtin (or a user provider) pointed at a different host.
    (None, "withheld".to_string(), false)
}

/// Resolve a saved alias. Mirrors `targets.resolve_named`.
pub fn resolve_named(
    name: &str,
    providers: &ProviderMap,
    store: &dyn KeyStore,
    require_credential: bool,
) -> Result<ResolvedTarget> {
    let alias = validate_target_alias(name, providers)?;
    let spec = load_targets(providers)?.remove(&alias);
    let spec = match spec {
        Some(spec) => spec,
        None => {
            return Err(Error::config(
                format!("Unknown target {:?}.", alias),
                Some("Add it with `anyllm targets add`, or run `anyllm targets list`.".to_string()),
            ))
        }
    };
    resolve_spec(
        &spec.provider,
        &spec.model,
        spec.base_url.as_deref(),
        Some(&alias),
        providers,
        store,
        require_credential,
    )
}

/// Resolve a `provider:model` string. Mirrors `targets.resolve_provider_model`.
pub fn resolve_provider_model(
    spec: &str,
    providers: &ProviderMap,
    store: &dyn KeyStore,
    require_credential: bool,
) -> Result<ResolvedTarget> {
    let (provider_name, model) = split_provider_model(spec)?;
    resolve_spec(
        &provider_name,
        &model,
        None,
        None,
        providers,
        store,
        require_credential,
    )
}

/// Resolve an alias, or a `provider:model` string. Never guess between them.
/// Mirrors `targets.resolve_member_spec`.
pub fn resolve_member_spec(
    spec: &str,
    providers: &ProviderMap,
    store: &dyn KeyStore,
    require_credential: bool,
) -> Result<ResolvedTarget> {
    let text = spec.trim();
    if text.contains(':') {
        resolve_provider_model(text, providers, store, require_credential)
    } else {
        resolve_named(text, providers, store, require_credential)
    }
}

/// Shared body of the resolve functions. Mirrors `targets._resolve_spec`.
fn resolve_spec(
    provider_name: &str,
    model: &str,
    base_url: Option<&str>,
    alias: Option<&str>,
    providers: &ProviderMap,
    store: &dyn KeyStore,
    require_credential: bool,
) -> Result<ResolvedTarget> {
    let provider = get_provider(provider_name, providers)?;
    let concrete = match base_url {
        Some(url) if !url.trim().is_empty() => validate_base_url(url)?,
        _ => {
            if provider.base_url.trim().is_empty() {
                return Err(Error::config(
                    format!("Provider {:?} has no base URL.", provider.name),
                    Some(
                        "Add the target with --base-url, or choose a provider that has an endpoint."
                            .to_string(),
                    ),
                ));
            }
            provider.base_url.trim().to_string()
        }
    };
    let target = Target::new(&provider.name, model.trim(), &concrete, alias);
    let (credential, source, host_is_default) = resolve_credentials(&provider, &target, store);
    if require_credential && provider.key_required && credential.is_none() {
        if host_is_default || provider.base_url.trim().is_empty() {
            let hint = match alias {
                Some(alias) => format!(
                    "Set {} or `anyllm keys set {}`. The {} key is also accepted on that provider's own endpoint.",
                    target_env_var(alias),
                    target_store_name(alias)?,
                    provider.name
                ),
                None => format!(
                    "Run `anyllm keys set {}`, or set the provider's environment variable.",
                    provider.name
                ),
            };
            return Err(Error::config(
                format!("No API key found for {}.", provider.name),
                Some(hint),
            ));
        }
        let display_alias = alias
            .map(|value| value.to_string())
            .unwrap_or_else(|| format!("{:?}", provider.name));
        return Err(Error::config(
            format!(
                "No credential is configured for target {} at {}.",
                display_alias, concrete
            ),
            Some(format!(
                "This URL is not the provider's default host, so the provider's usual key \
will not be sent there. Save a target key with `anyllm keys set {}` or set {}.",
                match alias {
                    Some(alias) => target_store_name(alias)?,
                    None => "target:NAME".to_string(),
                },
                target_env_var(alias.unwrap_or("NAME"))
            )),
        ));
    }
    Ok(ResolvedTarget {
        host_is_default: if provider.base_url.trim().is_empty() {
            true
        } else {
            host_is_default
        },
        http_warning: plain_http_warning(&concrete),
        target,
        provider,
        credential,
        credential_source: source,
    })
}

/// True when this URL is the provider's own configured endpoint. An empty
/// provider URL (the built-in `custom` provider) means the given URL is that
/// endpoint. Any other non-matching URL must not inherit the provider key.
pub fn host_uses_provider_key(provider: &Provider, base_url: Option<&str>) -> bool {
    let provider_url = provider.base_url.trim();
    if provider_url.is_empty() {
        return true;
    }
    let base_url = match base_url {
        Some(value) if !value.trim().is_empty() => value,
        _ => return true,
    };
    urls_match(base_url, provider_url)
}

/// Pick the key for a saved target, a restored session, or a live rebind.
/// Mirrors `targets.resolve_endpoint_key`.
pub fn resolve_endpoint_key(
    provider: &Provider,
    base_url: Option<&str>,
    target_alias: Option<&str>,
    store: &dyn KeyStore,
    providers: &ProviderMap,
) -> (Option<String>, String) {
    let alias = match target_alias {
        Some(value) if !value.trim().is_empty() => value,
        _ => {
            if host_uses_provider_key(provider, base_url) {
                let resolution = resolve_key(provider, store);
                return (resolution.value, resolution.source);
            }
            return (None, "withheld".to_string());
        }
    };
    // Invalid alias names behave like no alias (Python's resolve_named with a
    // bad alias raises; the CLI only stores valid ones, and a failure here is
    // surfaced as withheld rather than crashing the /info path).
    match resolve_named(alias, providers, store, false) {
        Ok(resolved) => {
            if let Some(url) = base_url {
                if !url.trim().is_empty() && !urls_match(url, &resolved.target.base_url) {
                    return (None, "withheld".to_string());
                }
            }
            (resolved.credential, resolved.credential_source)
        }
        Err(_) => (None, "withheld".to_string()),
    }
}

/// The error raised when a live request has no key. Mirrors
/// `targets.missing_endpoint_key`.
pub fn missing_endpoint_key(
    provider: &Provider,
    base_url: Option<&str>,
    target_alias: Option<&str>,
) -> Error {
    let base = base_url.unwrap_or("");
    if let Some(alias) = target_alias {
        return Error::config(
            format!("No target key for {:?} at {}.", alias, base),
            Some(format!(
                "The provider key is not used for this host. Set {} or `anyllm keys set {}`.",
                target_env_var(alias),
                target_store_name(alias).unwrap_or_else(|_| format!("target:{}", alias)),
            )),
        );
    }
    if !host_uses_provider_key(provider, base_url) {
        return Error::config(
            format!("Refusing to send the {} key to {}.", provider.name, base),
            Some(
                "That URL is not the provider's default host. Save a target for it and set \
ANYLLM_TARGET_<NAME>_API_KEY, or use the provider's normal endpoint."
                    .to_string(),
            ),
        );
    }
    Error::config(
        format!("No API key found for {}.", provider.name),
        Some(format!(
            "Run `anyllm keys set {}`, or set its environment variable.",
            provider.name
        )),
    )
}

/// Read a persisted target name. Reject anything that is not a string.
/// Mirrors `targets.saved_target_alias`.
pub fn saved_target_alias(data: &Value) -> Result<Option<String>> {
    let raw = match data.get("target") {
        None => return Ok(None),
        Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    match raw.as_str() {
        Some(text) if !text.trim().is_empty() => Ok(Some(text.trim().to_string())),
        _ => Err(Error::config(
            "Saved session has an invalid target name.",
            Some(
                "The target field must be a string alias, or be omitted. It must not contain a key."
                    .to_string(),
            ),
        )),
    }
}

/// True only for this process's explicit `--base-url` on this exact URL.
/// Mirrors `targets.explicit_endpoint_applies`.
pub fn explicit_endpoint_applies(session: &Session) -> bool {
    if session.target_alias.is_some() {
        return false;
    }
    match &session.explicit_endpoint {
        Some(explicit) if !explicit.trim().is_empty() => urls_match(explicit, &session.base_url),
        _ => false,
    }
}

/// Key for a live session. Mirrors `targets.resolve_live_credential`.
pub fn resolve_live_credential(
    session: &Session,
    providers: &ProviderMap,
    store: &dyn KeyStore,
) -> (Option<String>, String) {
    if explicit_endpoint_applies(session) {
        let resolution = resolve_key(&session.provider, store);
        return (resolution.value, resolution.source);
    }
    resolve_endpoint_key(
        &session.provider,
        Some(session.base_url.as_str()),
        session.target_alias.as_deref(),
        store,
        providers,
    )
}

/// One line for /info and /keys. Never includes the secret itself.
/// Mirrors `targets.describe_live_key`.
pub fn describe_live_key(
    session: &Session,
    providers: &ProviderMap,
    store: &dyn KeyStore,
) -> String {
    let (credential, source) = resolve_live_credential(session, providers, store);
    if !session.provider.key_required && credential.is_none() {
        return "not needed".to_string();
    }
    let credential = credential.unwrap_or_default();
    if credential.is_empty() || source == "withheld" {
        return "withheld (non-default host)".to_string();
    }
    let explicit = explicit_endpoint_applies(session);
    if explicit && source.starts_with("env:") {
        return format!(
            "explicit --base-url, environment ({})",
            source.split(':').nth(1).unwrap_or("")
        );
    }
    if explicit && source == "saved" {
        return format!("explicit --base-url, saved ({})", mask_key(&credential));
    }
    if source.starts_with("env:") {
        return format!("environment ({})", source.split(':').nth(1).unwrap_or(""));
    }
    if source == "saved-target" {
        return format!("target key ({})", mask_key(&credential));
    }
    if source == "saved" {
        return format!("saved ({})", mask_key(&credential));
    }
    if explicit {
        return "explicit --base-url".to_string();
    }
    if !source.is_empty() {
        return source;
    }
    "configured".to_string()
}

/// Refresh `session.api_key` from the target, explicit URL, or host rule.
/// Mirrors `targets.rebind_session_credentials`.
pub fn rebind_session_credentials(
    session: &mut Session,
    providers: &ProviderMap,
    store: &dyn KeyStore,
) -> Result<()> {
    let alias = session.target_alias.clone();
    let explicit = explicit_endpoint_applies(session);
    let (credential, source) = resolve_live_credential(session, providers, store);
    if let Some(key) = credential {
        session.api_key = Some(key);
        session.key_source = source;
        return Ok(());
    }
    // An in-memory key is kept only for the provider's own endpoint, and not
    // as a stand-in for an explicit URL whose key is now missing.
    let keeps_in_memory = !explicit
        && alias.is_none()
        && host_uses_provider_key(&session.provider, Some(&session.base_url))
        && session.api_key.is_some();
    if keeps_in_memory {
        if session.key_source.is_empty() {
            session.key_source = "session".to_string();
        }
        return Ok(());
    }
    session.api_key = None;
    session.key_source = if source.is_empty() {
        "withheld".to_string()
    } else {
        source
    };
    if explicit && session.provider.key_required {
        return Err(missing_endpoint_key(
            &session.provider,
            Some(if session.provider.base_url.trim().is_empty() {
                session.base_url.as_str()
            } else {
                session.provider.base_url.as_str()
            }),
            None,
        ));
    }
    if session.provider.key_required {
        return Err(missing_endpoint_key(
            &session.provider,
            Some(session.base_url.as_str()),
            alias.as_deref(),
        ));
    }
    Ok(())
}

/// Reject `--target` combined with an explicit provider, model, or base URL.
/// Mirrors `targets.conflicting_target_flags`.
pub fn conflicting_target_flags(
    target: Option<&str>,
    provider: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
) -> Result<()> {
    let chosen = match target {
        Some(value) if !value.is_empty() => value,
        _ => return Ok(()),
    };
    let _ = chosen;
    let mut clashes: Vec<&str> = Vec::new();
    if provider.map(|value| !value.is_empty()).unwrap_or(false) {
        clashes.push("--provider");
    }
    if model.map(|value| !value.is_empty()).unwrap_or(false) {
        clashes.push("--model");
    }
    if base_url.map(|value| !value.is_empty()).unwrap_or(false) {
        clashes.push("--base-url");
    }
    if !clashes.is_empty() {
        return Err(Error::usage(
            format!("Use either --target or {}, not both.", clashes.join(", ")),
            Some("A saved target already names the provider, model, and base URL.".to_string()),
        ));
    }
    Ok(())
}

/// Point a live session at a saved target without copying the key into saved
/// transcripts. Mirrors `targets.apply_target_to_session`.
pub fn apply_target_to_session(
    session: &mut Session,
    name: &str,
    providers: &ProviderMap,
    store: &dyn KeyStore,
) -> Result<ResolvedTarget> {
    let resolved = resolve_named(name, providers, store, true)?;
    session.provider = resolved.provider.clone();
    session.model = resolved.target.model.clone();
    session.base_url = resolved.target.base_url.clone();
    session.api_key = resolved.credential.clone();
    session.key_source = resolved.credential_source.clone();
    session.target_alias = resolved.target.alias.clone();
    // A target is its own credential identity. Drop one-shot --base-url consent.
    session.explicit_endpoint = None;
    Ok(resolved)
}

/// `targets.render_target_list`, collected as lines instead of printed.
pub fn render_target_list(style: &Style, providers: &ProviderMap) -> Result<Vec<String>> {
    let specs = list_target_specs(providers)?;
    let mut lines = Vec::new();
    if specs.is_empty() {
        lines.push(style.dim("no saved targets"));
        return Ok(lines);
    }
    lines.push(style.bold("NAME          PROVIDER/MODEL"));
    for (name, spec) in &specs {
        lines.push(format!("{:<12}  {}/{}", name, spec.provider, spec.model));
        let url = spec
            .base_url
            .clone()
            .unwrap_or_else(|| "(provider default)".to_string());
        lines.push(style.dim(&format!("              {}", url)));
    }
    Ok(lines)
}

// ------------------------------------------------------------ REPL commands

/// `/target NAME`. Mirrors `targets.repl_target`.
pub fn repl_target(repl: &mut dyn Repl, rest: &str) -> Result<()> {
    let name = rest.trim();
    if name.is_empty() {
        return Err(Error::usage(
            "Usage: /target NAME",
            Some("Save one with `anyllm targets add`.".to_string()),
        ));
    }
    let providers = repl.providers().clone();
    let resolved = {
        let store = repl.store();
        resolve_named(name, &providers, store, true)?
    };
    let session = repl.session();
    session.provider = resolved.provider.clone();
    session.model = resolved.target.model.clone();
    session.base_url = resolved.target.base_url.clone();
    session.api_key = resolved.credential.clone();
    session.key_source = resolved.credential_source.clone();
    session.target_alias = resolved.target.alias.clone();
    // A target is its own credential identity. Drop one-shot --base-url consent.
    session.explicit_endpoint = None;
    let style = *repl.style();
    repl.say(&style.dim(&format!(
        "target → {} · {}/{}",
        resolved.target.alias.clone().unwrap_or_default(),
        resolved.target.provider,
        resolved.target.model
    )));
    repl.say(&style.dim(&format!("  endpoint {}", resolved.target.base_url)));
    if let Some(warning) = &resolved.http_warning {
        repl.say(&style.yellow(&format!("  {}", warning)));
    }
    if !resolved.host_is_default && resolved.credential_source == "withheld" {
        repl.say(&style.dim("  provider key withheld because this is not the default host"));
    }
    Ok(())
}

/// `/targets`. Mirrors `targets.repl_targets`.
pub fn repl_targets(repl: &mut dyn Repl, _rest: &str) -> Result<()> {
    let providers = repl.providers().clone();
    let specs = list_target_specs(&providers)?;
    let style = *repl.style();
    if specs.is_empty() {
        repl.say(&style.dim("no saved targets"));
        return Ok(());
    }
    repl.say(&style.bold("Targets"));
    for (name, spec) in &specs {
        let url = spec
            .base_url
            .clone()
            .unwrap_or_else(|| "(provider default)".to_string());
        repl.say(&format!(
            "  {}  {}/{}  {}",
            name, spec.provider, spec.model, url
        ));
    }
    repl.say("");
    Ok(())
}

/// Register `/target` and `/targets`. Mirrors `targets.register`.
pub fn register(registry: &mut ReplRegistry) {
    use std::sync::Arc;
    registry.register(
        "target",
        Arc::new(|repl: &mut dyn Repl, rest: &str| repl_target(repl, rest)),
        Some(("/target NAME", "use a saved model target")),
    );
    registry.register(
        "targets",
        Arc::new(|repl: &mut dyn Repl, rest: &str| repl_targets(repl, rest)),
        Some(("/targets", "list saved targets")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::save_feature;
    use crate::sys::builtins;
    use crate::sys::{EmptyKeyStore, Session};
    use serde_json::json;

    // A mutex so env-mutating tests do not race each other.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct TestHome(PathBuf);

    impl TestHome {
        fn new(tag: &str) -> TestHome {
            let dir =
                std::env::temp_dir().join(format!("orch-targets-{}-{}", tag, crate::sys::new_id()));
            std::fs::create_dir_all(&dir).unwrap();
            TestHome(dir)
        }
    }

    impl Drop for TestHome {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn set_home(home: &TestHome) {
        std::env::set_var("ANYLLM_HOME", &home.0);
    }

    fn clear_keys_from_env() {
        for (key, _) in std::env::vars() {
            if key.ends_with("_API_KEY") || key.starts_with("ANYLLM_") {
                if key != "ANYLLM_HOME" {
                    std::env::remove_var(&key);
                }
            }
        }
    }

    #[test]
    fn roundtrip_and_remove() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("roundtrip");
        set_home(&home);
        clear_keys_from_env();
        let providers = builtins();
        let store = EmptyKeyStore;

        add_target(
            "fast",
            "openscale",
            "deepseek/deepseek-v4.1-flash",
            None,
            &providers,
            &store,
        )
        .unwrap();
        add_target(
            "dev",
            "custom",
            "local",
            Some("http://localhost:1234/v1"),
            &providers,
            &store,
        )
        .unwrap();
        let specs = list_target_specs(&providers).unwrap();
        assert_eq!(specs["fast"].provider, "openscale");
        assert_eq!(specs["fast"].base_url, None);
        assert_eq!(
            specs["dev"].base_url.as_deref(),
            Some("http://localhost:1234/v1")
        );
        let text = std::fs::read_to_string(targets_path()).unwrap();
        assert!(!text.contains("sk-"));
        assert!(!text.to_lowercase().contains("api_key"));
        remove_target("fast", &providers).unwrap();
        assert!(!list_target_specs(&providers).unwrap().contains_key("fast"));
    }

    #[test]
    fn split_keeps_model_colons_and_slashes() {
        let (provider, model) =
            split_provider_model("openscale:deepseek/deepseek-v4.1-flash:fast").unwrap();
        assert_eq!(provider, "openscale");
        assert_eq!(model, "deepseek/deepseek-v4.1-flash:fast");
        assert!(split_provider_model("no-colon").is_err());
    }

    #[test]
    fn unknown_alias_fails_before_a_request() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("unknown");
        set_home(&home);
        clear_keys_from_env();
        let providers = builtins();
        let error = resolve_named("missing", &providers, &EmptyKeyStore, true).unwrap_err();
        assert!(error.message.contains("Unknown target"));
    }

    #[test]
    fn conflicting_flags() {
        assert!(conflicting_target_flags(Some("fast"), Some("openai"), None, None).is_err());
        assert!(conflicting_target_flags(Some("fast"), None, Some("x"), None).is_err());
        assert!(conflicting_target_flags(Some("fast"), None, None, Some("http://x")).is_err());
        assert!(conflicting_target_flags(None, Some("openai"), Some("m"), None).is_ok());
    }

    #[test]
    fn malformed_and_future_version() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("versions");
        set_home(&home);
        clear_keys_from_env();
        let path = targets_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, json!({"version": 1, "targets": []}).to_string()).unwrap();
        let providers = builtins();
        assert!(load_targets(&providers).is_err());
        let mut payload = serde_json::Map::new();
        payload.insert("targets".to_string(), json!({}));
        payload.insert("version".to_string(), json!(2));
        crate::seam::write_json(&path, &Value::Object(payload), true).unwrap();
        let error = load_targets(&providers).unwrap_err();
        assert!(error.message.to_lowercase().contains("version"));
        let _ = save_feature; // referenced import
    }

    #[test]
    fn default_host_uses_provider_key_and_other_host_does_not() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("hosts");
        set_home(&home);
        clear_keys_from_env();
        std::env::set_var("OPENAI_API_KEY", "sk-openai-secret");
        let providers = builtins();
        let store = EmptyKeyStore;

        add_target(
            "official",
            "openai",
            "gpt-4o-mini",
            None,
            &providers,
            &store,
        )
        .unwrap();
        let official = resolve_named("official", &providers, &store, true).unwrap();
        assert_eq!(official.credential.as_deref(), Some("sk-openai-secret"));
        assert!(official.host_is_default);

        add_target(
            "dev",
            "openai",
            "local",
            Some("http://127.0.0.1:1234/v1"),
            &providers,
            &store,
        )
        .unwrap();
        let error = resolve_named("dev", &providers, &store, true).unwrap_err();
        assert!(!format!("{:?}", error).contains("sk-openai-secret"));
        assert!(error.hint.unwrap_or_default().contains("will not be sent"));
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn alias_key_is_sent_to_a_non_default_host() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("aliaskey");
        set_home(&home);
        clear_keys_from_env();
        std::env::set_var("OPENAI_API_KEY", "sk-openai-secret");
        std::env::set_var("ANYLLM_TARGET_DEV_API_KEY", "dev-secret");
        let providers = builtins();
        let store = EmptyKeyStore;
        add_target(
            "dev",
            "openai",
            "local",
            Some("http://127.0.0.1:1234/v1"),
            &providers,
            &store,
        )
        .unwrap();
        let resolved = resolve_named("dev", &providers, &store, true).unwrap();
        assert_eq!(resolved.credential.as_deref(), Some("dev-secret"));
        assert!(!resolved.host_is_default);
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ANYLLM_TARGET_DEV_API_KEY");
    }

    #[test]
    fn custom_endpoint_can_use_generic_key() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("generic");
        set_home(&home);
        clear_keys_from_env();
        std::env::set_var("ANYLLM_API_KEY", "generic-secret");
        let providers = builtins();
        add_target(
            "box",
            "custom",
            "local",
            Some("http://localhost:1234/v1"),
            &providers,
            &EmptyKeyStore,
        )
        .unwrap();
        let resolved = resolve_named("box", &providers, &EmptyKeyStore, true).unwrap();
        assert_eq!(resolved.credential.as_deref(), Some("generic-secret"));
        std::env::remove_var("ANYLLM_API_KEY");
    }

    #[test]
    fn user_provider_key_is_not_forwarded_to_a_different_url() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("labkey");
        set_home(&home);
        clear_keys_from_env();
        let mut lab = Provider::new("lab", "Lab", crate::sys::OPENAI, "http://lab.internal/v1");
        lab.env_vars.push("LAB_API_KEY".to_string());
        let mut providers = builtins();
        providers.insert("lab".to_string(), lab);
        std::env::set_var("LAB_API_KEY", "lab-secret");
        let same = resolve_member_spec("lab:model", &providers, &EmptyKeyStore, true).unwrap();
        assert_eq!(same.credential.as_deref(), Some("lab-secret"));
        add_target(
            "elsewhere",
            "lab",
            "model",
            Some("http://other.internal/v1"),
            &providers,
            &EmptyKeyStore,
        )
        .unwrap();
        let error = resolve_named("elsewhere", &providers, &EmptyKeyStore, true).unwrap_err();
        assert!(!format!("{:?}", error).contains("lab-secret"));
        std::env::remove_var("LAB_API_KEY");
    }

    #[test]
    fn normalized_default_endpoint_is_the_same_host() {
        let providers = builtins();
        let openai = &providers["openai"];
        let default = openai.base_url.clone();
        for url in [
            default.clone(),
            format!("{}/", default),
            "https://API.OpenAI.com/v1".to_string(),
            "https://api.openai.com:443/v1/".to_string(),
            "HTTPS://api.openai.com/v1".to_string(),
        ] {
            assert_eq!(
                canonical_endpoint(&url).unwrap(),
                canonical_endpoint(&default).unwrap()
            );
            assert!(host_uses_provider_key(openai, Some(&url)));
            assert!(explicit_override_warning(openai, Some(&url)).is_none());
        }
        for url in [
            "https://api.openai.com/v1?api-version=2024-10-21",
            "https://api.openai.com:443/v1?x=1",
            "http://api.openai.com/v1",
            "https://api.openai.com:8443/v1",
            "https://proxy.example/v1",
            "https://user:secret@api.openai.com/v1",
        ] {
            assert_ne!(
                canonical_endpoint(url).unwrap(),
                canonical_endpoint(&default).unwrap()
            );
            assert!(!host_uses_provider_key(openai, Some(url)));
            let note = explicit_override_warning(openai, Some(url)).unwrap();
            assert!(note.contains("still sends the openai key"));
            assert!(!note.contains('\n'));
            assert!(!note.contains("sk-"));
        }
        let remote_http =
            explicit_override_warning(openai, Some("http://proxy.example/v1")).unwrap();
        assert!(remote_http.contains("not encrypted"));
        let local_http = explicit_override_warning(openai, Some("http://127.0.0.1:9/v1")).unwrap();
        assert!(local_http.contains("still sends the openai key"));
        assert!(!local_http.contains("not encrypted"));
        let custom = &providers["custom"];
        assert!(explicit_override_warning(custom, Some("https://other.example/v1")).is_none());
        assert!(host_uses_provider_key(
            custom,
            Some("http://proxy.example/v1?q=1")
        ));
    }

    #[test]
    fn plain_http_warning_is_only_for_nonlocal_hosts() {
        assert!(plain_http_warning("http://127.0.0.1:1234/v1").is_none());
        assert!(plain_http_warning("http://localhost:1234/v1").is_none());
        assert!(plain_http_warning("https://example.com/v1").is_none());
        let warning = plain_http_warning("http://example.com/v1").unwrap();
        assert!(warning.contains("not be encrypted"));
    }

    #[test]
    fn alias_cannot_reuse_a_provider_name_or_env_slug() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("collide");
        set_home(&home);
        clear_keys_from_env();
        let mut lab = Provider::new(
            "open.scale",
            "Open Scale",
            crate::sys::OPENAI,
            "https://open.scale.example/v1",
        );
        lab.env_vars.push("OPEN_SCALE_API_KEY".to_string());
        let mut providers = builtins();
        providers.insert("open.scale".to_string(), lab);

        struct MemStore(BTreeMap<String, String>);
        impl KeyStore for MemStore {
            fn get(&self, name: &str) -> Result<Option<String>> {
                Ok(self.0.get(name).cloned())
            }
        }
        let store = MemStore(BTreeMap::new());
        let _ = store;

        for name in ["openscale", "OpenScale", "OPENSCALE"] {
            let error = add_target(
                name,
                "openscale",
                "m",
                Some("http://127.0.0.1:9/v1"),
                &providers,
                &EmptyKeyStore,
            )
            .unwrap_err();
            assert!(error.message.to_lowercase().contains("collides"));
        }
        assert!(add_target(
            "open-scale",
            "openai",
            "m",
            Some("http://127.0.0.1:9/v1"),
            &providers,
            &EmptyKeyStore
        )
        .is_err());

        // A target alias never receives the provider's own key.
        std::env::set_var("OPENSCALE_API_KEY", "provider-sentinel");
        std::env::set_var("ANYLLM_OPENSCALE_API_KEY", "also-provider");
        let target = Target::new("openscale", "m", "http://127.0.0.1:9/v1", Some("openscale"));
        let (credential, _source, _default) =
            resolve_credentials(&providers["openscale"], &target, &EmptyKeyStore);
        assert_ne!(credential.as_deref(), Some("provider-sentinel"));
        assert_ne!(credential.as_deref(), Some("also-provider"));
        assert_ne!(credential.as_deref(), Some("saved-provider-sentinel"));
        std::env::remove_var("OPENSCALE_API_KEY");
        std::env::remove_var("ANYLLM_OPENSCALE_API_KEY");
    }

    #[test]
    fn describe_live_key_and_rebind_rules() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = TestHome::new("live");
        set_home(&home);
        clear_keys_from_env();
        let providers = builtins();
        std::env::set_var("OPENAI_API_KEY", "sk-openai-secret");
        let mut session = Session::new(providers["openai"].clone(), "gpt-4o-mini");
        assert_eq!(
            describe_live_key(&session, &providers, &EmptyKeyStore),
            "environment (OPENAI_API_KEY)"
        );
        session.explicit_endpoint = Some("http://elsewhere.example/v1".to_string());
        session.base_url = "http://elsewhere.example/v1".to_string();
        let described = describe_live_key(&session, &providers, &EmptyKeyStore);
        assert_eq!(
            described,
            "explicit --base-url, environment (OPENAI_API_KEY)"
        );
        rebind_session_credentials(&mut session, &providers, &EmptyKeyStore).unwrap();
        assert_eq!(session.api_key.as_deref(), Some("sk-openai-secret"));
        std::env::remove_var("OPENAI_API_KEY");

        // Non-default host without consent refuses to leak the provider key.
        std::env::set_var("OPENAI_API_KEY", "sk-openai-secret");
        let mut other = Session::new(providers["openai"].clone(), "gpt-4o-mini");
        other.base_url = "http://other.example/v1".to_string();
        other.api_key = Some("sk-openai-secret".to_string());
        assert_eq!(
            describe_live_key(&other, &providers, &EmptyKeyStore),
            "withheld (non-default host)"
        );
        let error = rebind_session_credentials(&mut other, &providers, &EmptyKeyStore).unwrap_err();
        assert!(error.message.contains("Refusing to send"));
        assert_eq!(other.api_key, None);
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn saved_target_alias_rejects_non_strings() {
        assert_eq!(saved_target_alias(&json!({})).unwrap(), None);
        assert_eq!(saved_target_alias(&json!({"target": null})).unwrap(), None);
        assert_eq!(
            saved_target_alias(&json!({"target": " lab "})).unwrap(),
            Some("lab".to_string())
        );
        assert!(saved_target_alias(&json!({"target": 5})).is_err());
        assert!(saved_target_alias(&json!({"target": "  "})).is_err());
    }
}
