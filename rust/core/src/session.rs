//! Conversation state and on-disk sessions.
//!
//! Port of Python `anyllm/session.py`. The streaming send loop is the same:
//! a user message is appended, chunks accumulate with usage kept per-chunk,
//! and a failed request pops the message back off the history.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::adapters::{stream_chat, Chunk, Message, StreamArgs, Usage};
use crate::catalog::{validate_base_url, Provider};
use crate::config::{read_json, sessions_dir, write_json};
use crate::errors::{AnyLLMError, Result};
use crate::keystore::utc_now_iso;

/// Stored on a partial assistant turn when the user hits Ctrl-C mid-stream.
pub const INTERRUPTED_MARK: &str = "[interrupted]";

/// One finished exchange.
#[derive(Debug, Clone)]
pub struct Turn {
    pub text: String,
    pub reasoning: String,
    pub usage: Usage,
    pub elapsed: f64,
    pub finish_reason: Option<String>,
}

/// Mark a partial answer kept after the user interrupted a stream.
pub fn interrupted_content(text: &str) -> String {
    let body = text.trim_end();
    if body.ends_with(INTERRUPTED_MARK) {
        return body.to_string();
    }
    format!("{}\n{}", body, INTERRUPTED_MARK)
}

/// Choose the endpoint when a saved conversation is restored.
pub fn session_base_url(
    data: &Map<String, Value>,
    provider: &Provider,
    override_url: Option<&str>,
) -> Result<String> {
    if let Some(override_url) = override_url {
        if !override_url.trim().is_empty() {
            return validate_base_url(override_url);
        }
    }
    let saved = data.get("base_url");
    match saved {
        None | Some(Value::Null) => Ok(provider.base_url.clone()),
        Some(Value::String(saved)) if saved.trim().is_empty() => Ok(provider.base_url.clone()),
        Some(Value::String(saved)) => validate_base_url(saved).map_err(|exc| {
            AnyLLMError::config(format!(
                "Saved session has an invalid base URL {:?}.",
                saved
            ))
            .with_hint(exc.hint.unwrap_or_default())
        }),
        Some(_) => Err(
            AnyLLMError::config("Saved session has an invalid base URL.")
                .with_hint("Expected a string. Fix the session file, or delete it."),
        ),
    }
}

/// Explain a finish reason that means the answer is not whole.
pub fn finish_notice(finish_reason: Option<&str>) -> Option<&'static str> {
    let reason = finish_reason?.to_lowercase();
    match reason.as_str() {
        "length" | "max_tokens" | "max_completion_tokens" => {
            Some("the response hit the token limit and was cut off (try a larger --max-tokens)")
        }
        "content_filter" | "safety" | "blocklist" | "prohibited_content" | "recitation" => {
            Some("the response was stopped by the provider's content filter")
        }
        _ => None,
    }
}

/// One conversation: provider, model, history, and token totals.
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
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub turns: usize,
    /// Optional target alias shown by the CLI (never the API key).
    pub target_alias: Option<String>,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Provider,
        model: &str,
        base_url: Option<&str>,
        api_key: Option<&str>,
        key_source: &str,
        system: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<i64>,
        timeout: f64,
        show_reasoning: bool,
    ) -> Self {
        let messages: Vec<Message> = Vec::new();
        let base_url = base_url
            .filter(|url| !url.trim().is_empty())
            .unwrap_or(&provider.base_url)
            .to_string();
        Session {
            base_url,
            provider,
            model: model.to_string(),
            api_key: api_key.filter(|k| !k.is_empty()).map(str::to_string),
            key_source: key_source.to_string(),
            system: system.map(str::to_string),
            temperature,
            max_tokens,
            timeout,
            show_reasoning,
            messages,
            usage: Usage::default(),
            turns: 0,
            target_alias: None,
        }
    }

    /// Send a user message and stream the answer back.
    pub fn send(&mut self, text: &str, observer: &mut dyn SessionObserver) -> Result<Turn> {
        if text.trim().is_empty() {
            return Err(AnyLLMError::usage("The message is empty."));
        }
        if self.model.trim().is_empty() {
            return Err(
                AnyLLMError::usage("No model selected.").with_hint("Set one with /model NAME.")
            );
        }

        self.messages.push(Message::new("user", text));
        let started = Instant::now();
        let mut text_parts: Vec<String> = Vec::new();
        let mut thought_parts: Vec<String> = Vec::new();
        let mut usage = Usage::default();
        let mut finish_reason: Option<String> = None;

        let args = StreamArgs {
            model: &self.model,
            messages: &self.messages,
            system: self.system.as_deref(),
            key: self.api_key.as_deref(),
            base_url: Some(&self.base_url),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout: self.timeout,
            include_usage: true,
        };
        let mut stream = match stream_chat(&self.provider, &args) {
            Ok(stream) => stream,
            Err(err) => {
                self.messages.pop();
                return Err(err);
            }
        };
        loop {
            let chunk = match stream.next_chunk() {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(err) => {
                    self.messages.pop();
                    return Err(err);
                }
            };
            consume_chunk(
                &chunk,
                &mut usage,
                &mut finish_reason,
                &mut text_parts,
                &mut thought_parts,
                observer,
            );
        }

        let answer: String = text_parts.concat();
        let reasoning: String = thought_parts.concat();
        if answer.trim().is_empty() {
            self.messages.pop();
            if let Some(notice) = finish_notice(finish_reason.as_deref()) {
                return Err(AnyLLMError::provider(capitalize_first(notice))
                    .with_hint("The finish reason is the evidence for that limit."));
            }
            if !reasoning.trim().is_empty() {
                return Err(AnyLLMError::provider(
                    "The model returned reasoning but no answer text.",
                )
                .with_hint("That does not by itself show that the output cap was hit."));
            }
            return Err(
                AnyLLMError::provider("The provider returned an empty response.")
                    .with_hint("Try another model with /model, or rephrase the prompt."),
            );
        }

        self.messages.push(Message::new("assistant", &answer));
        self.usage = self.usage.combined(&usage);
        self.turns += 1;
        Ok(Turn {
            text: answer,
            reasoning,
            usage,
            elapsed: started.elapsed().as_secs_f64(),
            finish_reason,
        })
    }

    /// Append one finished exchange without calling a provider.
    pub fn append_exchange(
        &mut self,
        user_text: &str,
        assistant_text: &str,
        usage: Option<&Usage>,
    ) -> Result<()> {
        if user_text.trim().is_empty() {
            return Err(AnyLLMError::usage("The message is empty."));
        }
        if assistant_text.trim().is_empty() {
            return Err(AnyLLMError::usage("The answer is empty."));
        }
        self.messages.push(Message::new("user", user_text));
        self.messages
            .push(Message::new("assistant", assistant_text));
        if let Some(usage) = usage {
            self.usage = self.usage.combined(usage);
        }
        self.turns += 1;
        Ok(())
    }

    pub fn restore(&mut self, messages: Vec<Message>, usage: Usage) {
        self.turns = messages.iter().filter(|m| m.role == "assistant").count();
        self.messages = messages;
        self.usage = usage;
    }

    /// Serializable payload, matching the Python `to_dict` shape.
    pub fn to_dict(&self) -> Value {
        let mut payload = json!({
            "version": 1,
            "provider": self.provider.name,
            "model": self.model,
            "system": self.system,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "show_reasoning": self.show_reasoning,
            "usage": self.usage.to_dict(),
            "updated": utc_now_iso(),
        });
        if let Some(map) = payload.as_object_mut() {
            map.insert(
                "messages".into(),
                Value::Array(
                    self.messages
                        .iter()
                        .map(|m| json!({"role": m.role, "content": m.content}))
                        .collect(),
                ),
            );
            if !self.base_url.trim().is_empty() {
                map.insert("base_url".into(), json!(self.base_url));
            }
            if let Some(alias) = &self.target_alias {
                if !alias.trim().is_empty() {
                    map.insert("target".into(), json!(alias));
                }
            }
        }
        payload
    }
}

fn capitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Observer callbacks used by `Session::send`.
pub trait SessionObserver {
    fn on_text(&mut self, _text: &str) {}
    fn on_reasoning(&mut self, _text: &str) {}
}

/// A no-op observer.
pub struct NoObserver;
impl SessionObserver for NoObserver {}

fn consume_chunk(
    chunk: &Chunk,
    usage: &mut Usage,
    finish_reason: &mut Option<String>,
    text_parts: &mut Vec<String>,
    thought_parts: &mut Vec<String>,
    observer: &mut dyn SessionObserver,
) {
    if let Some(chunk_usage) = &chunk.usage {
        usage.update(chunk_usage);
    }
    if !chunk.text.is_empty() {
        text_parts.push(chunk.text.clone());
        observer.on_text(&chunk.text);
    }
    if !chunk.reasoning.is_empty() {
        thought_parts.push(chunk.reasoning.clone());
        observer.on_reasoning(&chunk.reasoning);
    }
    if let Some(finish) = &chunk.finish_reason {
        *finish_reason = Some(finish.clone());
    }
}

// -- saved sessions -------------------------------------------------------------

/// Allow letters, digits, spaces, dots, dashes, and underscores.
pub fn sanitize_session_name(name: &str) -> Result<String> {
    let mut cleaned = String::new();
    let mut in_bad_run = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | ' ') {
            cleaned.push(ch);
            in_bad_run = false;
        } else if !in_bad_run {
            cleaned.push('-');
            in_bad_run = true;
        }
    }
    let cleaned = cleaned
        .trim_matches(|c: char| c == ' ' || c == '.' || c == '-')
        .to_string();
    if cleaned.is_empty() {
        return Err(AnyLLMError::config("Invalid session name.")
            .with_hint("Use letters, digits, spaces, dots, dashes, or underscores."));
    }
    Ok(cleaned.chars().take(64).collect())
}

pub fn session_path(name: &str) -> Result<PathBuf> {
    Ok(sessions_dir().join(format!("{}.json", sanitize_session_name(name)?)))
}

/// `YYYYmmdd-HHMMSS` in local time (Python uses `datetime.now()`).
pub fn default_session_name() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = local_civil(secs);
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, mo, d, h, mi, s)
}

/// Local civil time. Reads `TZ`/`LOCALTIME` is out of scope here; without a
/// tz database the port uses UTC (a deviation from the Python local clock).
fn local_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        m,
        d,
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// Save a session; returns the file path.
pub fn save_session(session: &Session, name: Option<&str>) -> Result<PathBuf> {
    let generated;
    let name = match name {
        Some(name) => name,
        None => {
            generated = default_session_name();
            generated.as_str()
        }
    };
    let target = session_path(name)?;
    write_json(&target, &session.to_dict(), true).map_err(|e| {
        AnyLLMError::config(format!("Could not save the session: {}", e.message)).with_hint(
            format!(
                "Check that you can write to {}",
                target
                    .parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            ),
        )
    })?;
    Ok(target)
}

/// Read and validate one saved session's JSON data.
pub fn load_session_data(name: &str) -> Result<Map<String, Value>> {
    let path = session_path(name)?;
    if !path.exists() {
        return Err(
            AnyLLMError::config(format!("Session {:?} was not found.", name))
                .with_hint("Run `/sessions` in chat to list saved sessions."),
        );
    }
    let data = read_json(&path, "session")?;
    if let Some(messages) = data.get("messages") {
        if let Some(list) = messages.as_array() {
            for item in list {
                let ok = item
                    .as_object()
                    .map(|m| {
                        matches!(
                            m.get("role").and_then(Value::as_str),
                            Some("user") | Some("assistant")
                        ) && m.get("content").and_then(Value::as_str).is_some()
                    })
                    .unwrap_or(false);
                if !ok {
                    return Err(AnyLLMError::config(format!(
                        "Session {:?} contains an invalid message.",
                        name
                    ))
                    .with_hint(format!("Fix or delete {}", path.display())));
                }
            }
        } else {
            return Err(AnyLLMError::config(format!(
                "Session {:?} has an invalid message list.",
                name
            ))
            .with_hint(format!("Fix or delete {}", path.display())));
        }
    }
    let provider = data
        .get("provider")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty());
    if provider.is_none() {
        return Err(
            AnyLLMError::config(format!("Session {:?} does not name a provider.", name))
                .with_hint(format!("Fix or delete {}", path.display())),
        );
    }
    Ok(data)
}

/// One row of `/sessions` (and of `list_sessions`).
#[derive(Debug, Clone)]
pub struct SessionEntry {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub messages: usize,
    pub updated: String,
    pub path: PathBuf,
}

/// List saved sessions, newest first.
pub fn list_sessions() -> Vec<SessionEntry> {
    let directory = sessions_dir();
    let mut entries: Vec<SessionEntry> = Vec::new();
    let Ok(dir) = std::fs::read_dir(&directory) else {
        return entries;
    };
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let data = match read_json(&path, "session") {
            Ok(data) => data,
            Err(_) => continue,
        };
        let message_count = data
            .get("messages")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let updated = data
            .get("updated")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_default();
        entries.push(SessionEntry {
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            provider: data
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            model: data
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            messages: message_count,
            updated,
            path,
        });
    }
    entries.sort_by(|a, b| b.updated.cmp(&a.updated));
    entries
}

pub fn latest_session_name() -> Option<String> {
    list_sessions().into_iter().next().map(|e| e.name)
}

/// Messages parsed from a saved session payload.
pub fn messages_from_data(data: &Map<String, Value>) -> Vec<Message> {
    data.get("messages")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|item| {
                    let map = item.as_object()?;
                    Some(Message::new(
                        map.get("role").and_then(Value::as_str)?,
                        map.get("content").and_then(Value::as_str)?,
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Usage parsed from a saved session payload.
pub fn usage_from_data(data: &Map<String, Value>) -> Usage {
    Usage::from_dict(data.get("usage").unwrap_or(&Value::Null))
}
