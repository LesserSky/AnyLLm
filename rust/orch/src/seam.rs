//! Integration seam for model orchestration. Port of `anyllm.seam`.
//!
//! Contracts, unchanged from the Python:
//!
//! * [`RequestSnapshot`] is immutable and contains no credentials.
//! * [`run_request`] performs one completion. It never prints. Streaming
//!   events are queued for the main thread; workers do not touch the terminal.
//! * [`Presentation`] is the main-thread rendering interface.
//! * Feature JSON under the config directory is versioned and separate from
//!   conversation transcripts. Attachments approved for a process live only
//!   in [`AttachmentStore`] (memory).
//! * REPL handlers register through [`ReplRegistry::register`].
//!
//! Cancellation stays cooperative: a blocking HTTP read cannot be killed, an
//! in-flight provider request may still finish and incur charges. Callers
//! bound that window with [`RunOptions::timeout`] and an overall deadline.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::sys::{
    config_dir, split_words, ChatBackend, ChatRequest, Error, ErrorKind, KeyStore, Message,
    Provider, ProviderMap, Style, Usage,
};
use crate::targets::Target;

pub const SNAPSHOT_VERSION: u64 = 1;
pub const FEATURE_STATE_VERSION: u64 = 1;

pub const CANCELLATION_NOTE: &str = "Cancellation stops scheduling and printing. An HTTP read \
already in flight cannot be killed, so that provider request may still finish and incur \
charges. Timeouts bound the wait; this is not a hard cancel.";

// ----------------------------------------------------------- attachments

/// File text the user already approved for this process. `content` stays in
/// memory; persisted artifacts store a path and hash only, never the body.
#[derive(Clone, Debug, PartialEq)]
pub struct ApprovedAttachment {
    pub path: String,
    pub sha256: String,
    pub content: String,
}

impl ApprovedAttachment {
    pub fn new(path: &str, sha256: &str, content: &str) -> ApprovedAttachment {
        ApprovedAttachment {
            path: path.to_string(),
            sha256: sha256.to_string(),
            content: content.to_string(),
        }
    }

    pub fn ref_value(&self) -> Value {
        json!({
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.content.len(),
        })
    }
}

/// Process-local approved attachments. Not a filesystem walker.
#[derive(Default)]
pub struct AttachmentStore {
    items: BTreeMap<String, ApprovedAttachment>,
}

impl AttachmentStore {
    pub fn approve(&mut self, attachment: ApprovedAttachment) -> Result<(), Error> {
        if attachment.path.is_empty() || attachment.path.trim() != attachment.path {
            return Err(Error::usage(
                "Approved attachment paths must be exact and non-empty.",
                None,
            ));
        }
        if attachment
            .path
            .replace('\\', "/")
            .split('/')
            .any(|part| part == "..")
        {
            return Err(Error::usage(
                "Approved attachment paths cannot contain '..'.",
                None,
            ));
        }
        self.items.insert(attachment.path.clone(), attachment);
        Ok(())
    }

    pub fn revoke(&mut self, path: &str) {
        self.items.remove(path);
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn list(&self) -> Vec<ApprovedAttachment> {
        self.items.values().cloned().collect()
    }
}

fn global_store() -> &'static Mutex<AttachmentStore> {
    static STORE: OnceLock<Mutex<AttachmentStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(AttachmentStore::default()))
}

/// The process-wide store Package B should fill after consent.
pub fn attachment_store() -> &'static Mutex<AttachmentStore> {
    global_store()
}

// -------------------------------------------------------------- snapshot

/// Frozen view of a conversation at the moment a fan-out starts.
#[derive(Clone, Debug)]
pub struct RequestSnapshot {
    pub version: u64,
    pub snapshot_id: String,
    pub history: Vec<(String, String)>,
    pub system: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub attachments: Vec<ApprovedAttachment>,
    pub record_refs: bool,
}

impl RequestSnapshot {
    pub fn attachment_map(&self) -> BTreeMap<String, ApprovedAttachment> {
        self.attachments
            .iter()
            .map(|item| (item.path.clone(), item.clone()))
            .collect()
    }
}

/// Copy history into an immutable snapshot. Later edits to the source are
/// ignored. Mirrors `seam.make_snapshot`.
#[allow(clippy::too_many_arguments)]
pub fn make_snapshot(
    history: &[Message],
    system: Option<&str>,
    temperature: Option<f64>,
    max_tokens: Option<i64>,
    attachments: Option<Vec<ApprovedAttachment>>,
    record_refs: bool,
    snapshot_id: Option<&str>,
) -> RequestSnapshot {
    let frozen: Vec<(String, String)> = history
        .iter()
        .filter(|item| item.role == "user" || item.role == "assistant")
        .map(|item| (item.role.clone(), item.content.clone()))
        .collect();
    RequestSnapshot {
        version: SNAPSHOT_VERSION,
        snapshot_id: snapshot_id
            .map(|id| id.to_string())
            .unwrap_or_else(crate::sys::new_id),
        history: frozen,
        system: system.map(|value| value.to_string()),
        temperature,
        max_tokens,
        attachments: attachments.unwrap_or_default(),
        record_refs,
    }
}

/// Copy a session without reading `api_key` or `key_source`.
pub fn snapshot_from_session(
    session: &crate::sys::Session,
    attachments: Option<Vec<ApprovedAttachment>>,
    record_refs: bool,
) -> RequestSnapshot {
    let attachments = attachments.unwrap_or_else(|| attachment_store().lock().unwrap().list());
    make_snapshot(
        &session.messages,
        session.system.as_deref(),
        session.temperature,
        session.max_tokens,
        Some(attachments),
        record_refs,
        None,
    )
}

/// Private message list for one request. Mutating it does not change the
/// snapshot. Mirrors `seam.messages_for`.
pub fn messages_for(
    snapshot: &RequestSnapshot,
    prompt: &str,
    extra: Option<&[Message]>,
) -> Vec<Message> {
    let mut copied: Vec<Message> = snapshot
        .history
        .iter()
        .map(|(role, content)| Message::new(role, content))
        .collect();
    if !prompt.is_empty() {
        copied.push(Message::new("user", prompt));
    }
    if let Some(extra) = extra {
        copied.extend(extra.iter().cloned());
    }
    copied
}

// ----------------------------------------------------------- run plumbing

/// Per-call limits. `deadline` is an absolute time, enforced between chunks.
#[derive(Clone, Debug)]
pub struct RunOptions {
    pub timeout: f64,
    pub deadline: Option<Instant>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub show_reasoning: bool,
    pub tools_payload: Option<Value>,
}

impl Default for RunOptions {
    /// Python's dataclass default: 120 s timeout, everything else unset.
    fn default() -> RunOptions {
        RunOptions {
            timeout: 120.0,
            deadline: None,
            temperature: None,
            max_tokens: None,
            show_reasoning: false,
            tools_payload: None,
        }
    }
}

impl RunOptions {
    pub fn new(timeout: f64) -> RunOptions {
        RunOptions {
            timeout,
            ..Default::default()
        }
    }
}

/// Status for the main thread. Workers enqueue these and do not print.
#[derive(Clone, Debug, PartialEq)]
pub struct RunEvent {
    pub kind: String,
    pub target: String,
    pub status: String,
    pub detail: String,
}

impl RunEvent {
    pub fn status(target: &str, status: &str) -> RunEvent {
        RunEvent {
            kind: "status".to_string(),
            target: target.to_string(),
            status: status.to_string(),
            detail: String::new(),
        }
    }
}

/// One completion. `usage` is `None` when the provider reported nothing.
/// Mirrors `seam.RunResult`, including the `public_dict` JSON shape.
#[derive(Clone, Debug, Default)]
pub struct RunResult {
    pub status: String,
    pub text: String,
    pub reasoning: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub elapsed: f64,
    pub incomplete: bool,
    pub tool_deltas: Vec<crate::sys::ToolCallDelta>,
    pub tool_trace: Vec<Value>,
    pub request_count: u64,
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub alias: String,
}

impl RunResult {
    /// `RunResult.public_dict`, plus the `alias` the team artifact adds.
    pub fn public_value(&self) -> Value {
        json!({
            "status": self.status,
            "text": self.text,
            "reasoning": self.reasoning,
            "usage": self.usage.as_ref().map(|usage| usage.to_value()),
            "finish_reason": self.finish_reason,
            "error": self.error,
            "elapsed": (self.elapsed * 1000.0).round() / 1000.0,
            "incomplete": self.incomplete,
            "request_count": self.request_count,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "alias": self.alias,
        })
    }
}

/// Main-thread output. Package B can wrap `block` for Markdown without
/// changing team scheduling; the write target is injected here.
pub struct Presentation<'w> {
    pub style: Style,
    output: &'w mut dyn Write,
}

impl<'w> Presentation<'w> {
    pub fn new(style: Style, output: &'w mut dyn Write) -> Presentation<'w> {
        Presentation { style, output }
    }

    pub fn line(&mut self, text: &str) {
        let _ = writeln!(self.output, "{}", text);
    }

    pub fn status_line(&mut self, text: &str) {
        self.line(text);
    }

    pub fn warn(&mut self, text: &str) {
        self.line(&self.style.yellow(text));
    }

    pub fn block(&mut self, title: &str, body: &str, meta: &str) {
        self.line(&self.style.bold(title));
        if !meta.is_empty() {
            self.line(&self.style.dim(meta));
        }
        if !body.is_empty() {
            self.line(body);
        }
        self.line("");
    }
}

/// Presentation that records lines in memory, for tests and dry runs.
#[derive(Default)]
pub struct CapturedPresentation {
    pub style: Style,
    pub lines: Vec<String>,
}

impl CapturedPresentation {
    pub fn new(style: Style) -> CapturedPresentation {
        CapturedPresentation {
            style,
            lines: Vec::new(),
        }
    }

    pub fn line(&mut self, text: &str) {
        self.lines.push(text.to_string());
    }

    pub fn status_line(&mut self, text: &str) {
        self.line(text);
    }

    pub fn warn(&mut self, text: &str) {
        self.line(&self.style.yellow(text));
    }

    pub fn block(&mut self, title: &str, body: &str, meta: &str) {
        self.line(&self.style.bold(title));
        if !meta.is_empty() {
            self.line(&self.style.dim(meta));
        }
        if !body.is_empty() {
            self.line(body);
        }
        self.line("");
    }
}

// ------------------------------------------------------------ run_request

/// Arguments for [`run_request`] that the Python passes as keyword args.
pub struct RunCall<'a> {
    pub credential: Option<&'a str>,
    /// `Some` sends snapshot history plus the prompt; `None` sends `messages`
    /// as-is (tool rounds).
    pub prompt: Option<&'a str>,
    pub messages: Option<Vec<Message>>,
    /// Shared cooperative-cancellation flag, cloned into the backend request.
    pub cancel: Option<Arc<AtomicBool>>,
    pub events: Option<&'a mpsc::Sender<RunEvent>>,
}

/// Run one completion. `messages`, when passed, is copied and sent as-is
/// (tool rounds). Otherwise the snapshot history plus `prompt` is used.
/// Credentials stay in the call and are never written onto the result.
pub fn run_request(
    backend: &dyn ChatBackend,
    provider: &Provider,
    target: &Target,
    snapshot: &RequestSnapshot,
    options: &RunOptions,
    call: RunCall<'_>,
) -> RunResult {
    let label = target
        .alias
        .clone()
        .unwrap_or_else(|| format!("{}:{}", target.provider, target.model));
    let cancel = call.cancel.clone();

    if let Some(flag) = &cancel {
        if flag.load(Ordering::SeqCst) {
            emit(call.events, RunEvent::status(&label, "cancelled"));
            return blank(
                "cancelled",
                target,
                "Cancelled before the request was sent.",
            );
        }
    }

    let now = Instant::now();
    let mut timeout = options.timeout.max(0.1);
    if let Some(deadline) = options.deadline {
        let remaining = deadline.duration_since(now).as_secs_f64();
        if remaining <= 0.0 {
            emit(call.events, RunEvent::status(&label, "timeout"));
            return blank(
                "timeout",
                target,
                "The team deadline elapsed before this request started.",
            );
        }
        timeout = timeout.min(remaining).max(0.1);
    }

    let outbound = match &call.messages {
        Some(messages) => messages.clone(),
        None => messages_for(snapshot, call.prompt.unwrap_or(""), None),
    };
    let temperature = options.temperature.or(snapshot.temperature);
    let max_tokens = options.max_tokens.or(snapshot.max_tokens);
    emit(call.events, RunEvent::status(&label, "running"));

    let request = ChatRequest {
        provider: provider.clone(),
        model: target.model.clone(),
        base_url: target.base_url.clone(),
        messages: outbound,
        system: snapshot.system.clone(),
        credential: call.credential.map(|value| value.to_string()),
        temperature,
        max_tokens,
        timeout,
        tools: options.tools_payload.clone(),
        deadline: options.deadline,
        cancel: cancel.clone(),
    };

    let started = Instant::now();
    let mut text_parts: Vec<String> = Vec::new();
    let mut thought_parts: Vec<String> = Vec::new();
    let mut deltas: Vec<crate::sys::ToolCallDelta> = Vec::new();
    let mut usage: Option<Usage> = None;
    let mut finish_reason: Option<String> = None;

    let streamed = backend.stream(&request);
    match streamed {
        Ok(chunks) => {
            for chunk in &chunks {
                if let Some(flag) = &cancel {
                    if flag.load(Ordering::SeqCst) {
                        let elapsed = started.elapsed().as_secs_f64();
                        emit(call.events, RunEvent::status(&label, "cancelled"));
                        return RunResult {
                            status: "cancelled".to_string(),
                            text: text_parts.concat(),
                            reasoning: thought_parts.concat(),
                            usage,
                            finish_reason: finish_reason.clone(),
                            error: Some(CANCELLATION_NOTE.to_string()),
                            elapsed,
                            incomplete: true,
                            tool_deltas: deltas.clone(),
                            tool_trace: Vec::new(),
                            request_count: 1,
                            provider: target.provider.clone(),
                            model: target.model.clone(),
                            base_url: target.base_url.clone(),
                            alias: target.alias.clone().unwrap_or_default(),
                        };
                    }
                }
                if let Some(chunk_usage) = &chunk.usage {
                    usage.get_or_insert_with(Usage::default).update(chunk_usage);
                }
                if !chunk.text.is_empty() {
                    text_parts.push(chunk.text.clone());
                }
                if !chunk.reasoning.is_empty() {
                    thought_parts.push(chunk.reasoning.clone());
                }
                if let Some(delta) = &chunk.tool_call {
                    deltas.push(delta.clone());
                }
                if let Some(finish) = &chunk.finish_reason {
                    finish_reason = Some(finish.clone());
                }
            }
        }
        Err(error) => {
            let status = match error.kind {
                ErrorKind::Network => {
                    let cancelled = cancel
                        .as_ref()
                        .map(|flag| flag.load(Ordering::SeqCst))
                        .unwrap_or(false);
                    let timed_out = options
                        .deadline
                        .map(|deadline| Instant::now() >= deadline)
                        .unwrap_or(false);
                    let lowered = error.message.to_lowercase();
                    if cancelled {
                        "cancelled"
                    } else if timed_out
                        || lowered.contains("timed out")
                        || lowered.contains("deadline")
                    {
                        "timeout"
                    } else {
                        "error"
                    }
                }
                _ => "error",
            };
            return finished_error(
                status,
                target,
                &error.message,
                started,
                &text_parts,
                &thought_parts,
                usage,
                finish_reason,
                deltas,
            );
        }
    }

    let text = text_parts.concat();
    let reasoning = thought_parts.concat();
    let length_capped = matches!(
        finish_reason.as_deref(),
        Some("length") | Some("max_tokens") | Some("max_completion_tokens")
    );
    let mut incomplete = length_capped;
    let mut status = if incomplete && deltas.is_empty() {
        "incomplete"
    } else {
        "ok"
    }
    .to_string();
    let mut error: Option<String> = None;
    if text.trim().is_empty() && deltas.is_empty() {
        if length_capped {
            // finish_reason is evidence the output cap stopped the reply.
            status = "incomplete".to_string();
            let mut message = "The output cap was exhausted before any answer text.".to_string();
            if !reasoning.trim().is_empty() {
                message.push_str(" Reasoning was returned.");
            }
            error = Some(message);
            incomplete = true;
        } else if !reasoning.trim().is_empty() {
            status = "incomplete".to_string();
            error = Some(
                "The model returned reasoning but no answer text. That does not by itself \
show that the output cap was hit."
                    .to_string(),
            );
            incomplete = true;
        } else {
            status = "error".to_string();
            error = Some("The provider returned an empty response.".to_string());
            incomplete = true;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    emit(call.events, RunEvent::status(&label, &status));
    let incomplete = incomplete || status != "ok";
    RunResult {
        status,
        text,
        reasoning,
        usage,
        finish_reason,
        error,
        elapsed,
        incomplete,
        tool_deltas: deltas,
        tool_trace: Vec::new(),
        request_count: 1,
        provider: target.provider.clone(),
        model: target.model.clone(),
        base_url: target.base_url.clone(),
        alias: target.alias.clone().unwrap_or_default(),
    }
}

fn emit(events: Option<&mpsc::Sender<RunEvent>>, event: RunEvent) {
    if let Some(sender) = events {
        let _ = sender.send(event);
    }
}

fn blank(status: &str, target: &Target, message: &str) -> RunResult {
    RunResult {
        status: status.to_string(),
        error: Some(message.to_string()),
        incomplete: true,
        provider: target.provider.clone(),
        model: target.model.clone(),
        base_url: target.base_url.clone(),
        alias: target.alias.clone().unwrap_or_default(),
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn finished_error(
    status: &str,
    target: &Target,
    message: &str,
    started: Instant,
    text_parts: &[String],
    thought_parts: &[String],
    usage: Option<Usage>,
    finish_reason: Option<String>,
    deltas: Vec<crate::sys::ToolCallDelta>,
) -> RunResult {
    RunResult {
        status: status.to_string(),
        text: text_parts.concat(),
        reasoning: thought_parts.concat(),
        usage,
        finish_reason,
        error: Some(message.to_string()),
        elapsed: started.elapsed().as_secs_f64(),
        incomplete: true,
        tool_deltas: deltas,
        request_count: 1,
        provider: target.provider.clone(),
        model: target.model.clone(),
        base_url: target.base_url.clone(),
        alias: target.alias.clone().unwrap_or_default(),
        ..Default::default()
    }
}

// --------------------------------------------------------- feature files

/// Load a versioned feature file. Missing files are empty version-1 objects.
/// Mirrors `seam.load_feature` (including the on-disk `version` check).
pub fn load_feature(path: &Path, what: &str) -> Result<Map<String, Value>, Error> {
    let data = read_json(path, what)?;
    if data.is_empty() {
        let mut empty = Map::new();
        empty.insert("version".to_string(), Value::from(FEATURE_STATE_VERSION));
        return Ok(empty);
    }
    let version = data
        .get("version")
        .and_then(|value| value.as_u64())
        .unwrap_or(1);
    if version != FEATURE_STATE_VERSION {
        return Err(Error::config(
            format!(
                "The {} file uses version {}, which this build cannot read.",
                what, version
            ),
            Some(format!(
                "Expected version {}. Fix or remove {}.",
                FEATURE_STATE_VERSION,
                path.display()
            )),
        ));
    }
    Ok(data)
}

/// Save a versioned feature file, owner-only.
pub fn save_feature(path: &Path, data: Map<String, Value>) -> Result<(), Error> {
    let mut payload = data;
    payload.insert("version".to_string(), Value::from(FEATURE_STATE_VERSION));
    write_json(path, &Value::Object(payload), true)
}

/// Path of a feature file inside the config directory.
pub fn feature_path(name: &str) -> PathBuf {
    config_dir().join(name)
}

/// Read a JSON object from disk. Missing files read as empty objects.
/// Same error texts as `config.read_json`.
pub fn read_json(path: &Path, what: &str) -> Result<Map<String, Value>, Error> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let raw = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(exc) => {
            return Err(Error::config(
                format!("Could not read the {} file: {}", what, exc),
                Some(path.display().to_string()),
            ))
        }
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(exc) => {
            return Err(Error::config(
                format!("The {} file is not valid JSON (line {}).", what, exc.line()),
                Some(format!("Fix or delete {}", path.display())),
            ))
        }
    };
    match parsed {
        Value::Object(map) => Ok(map),
        _ => Err(Error::config(
            format!("The {} file must contain a JSON object.", what),
            Some(format!("Fix or delete {}", path.display())),
        )),
    }
}

/// Atomically write JSON, optionally with owner-only permissions.
/// Mirrors `config.write_json`.
pub fn write_json(path: &Path, data: &Value, private: bool) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|exc| {
            Error::config(
                format!("Could not create {}: {}", parent.display(), exc),
                None,
            )
        })?;
    }
    let mut body = serde_json::to_string_pretty(data).unwrap_or_default();
    body.push('\n');
    let temp_name = format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("state"),
        std::process::id()
    );
    let temp_path = path.with_file_name(temp_name);
    let write_result = std::fs::write(&temp_path, body.as_bytes())
        .and_then(|_| set_permissions(&temp_path, private));
    if let Err(exc) = write_result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(Error::config(
            format!("Could not write {}: {}", path.display(), exc),
            None,
        ));
    }
    match std::fs::rename(&temp_path, path) {
        Ok(()) => Ok(()),
        Err(exc) => {
            let _ = std::fs::remove_file(&temp_path);
            Err(Error::config(
                format!("Could not write {}: {}", path.display(), exc),
                None,
            ))
        }
    }
}

#[cfg(unix)]
fn set_permissions(path: &Path, private: bool) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = if private { 0o600 } else { 0o644 };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_permissions(_path: &Path, _private: bool) -> std::io::Result<()> {
    Ok(())
}

// ------------------------------------------------------------ repl seam

/// What a REPL handler needs from the host REPL. The concrete REPL lives in
/// the CLI crate; tests provide their own implementation.
pub trait Repl {
    fn say(&mut self, text: &str);
    fn style(&self) -> &Style;
    fn providers(&self) -> &ProviderMap;
    fn store(&self) -> &dyn KeyStore;
    fn session(&mut self) -> &mut crate::sys::Session;
    /// Whether stdin is an interactive terminal.
    fn is_interactive(&self) -> bool {
        false
    }
}

/// One registered REPL command: `(repl, rest) -> Result`.
pub type ReplHandler = Arc<dyn Fn(&mut dyn Repl, &str) -> Result<(), Error> + Send + Sync>;

/// Registered REPL commands, mirroring `seam.REPL_HANDLERS` + `HELP_ROWS`.
#[derive(Default)]
pub struct ReplRegistry {
    pub handlers: BTreeMap<String, ReplHandler>,
    pub help_rows: Vec<(String, String)>,
}

impl ReplRegistry {
    /// Register a REPL command. Safe to call for the same row twice.
    pub fn register(&mut self, name: &str, handler: ReplHandler, help_row: Option<(&str, &str)>) {
        self.handlers.insert(name.to_string(), handler);
        if let Some(row) = help_row {
            let row = (row.0.to_string(), row.1.to_string());
            if !self.help_rows.contains(&row) {
                self.help_rows.push(row);
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&ReplHandler> {
        self.handlers.get(name)
    }
}

/// Convenience wrapper so handler bodies register without boilerplate.
pub fn handler<F>(function: F) -> ReplHandler
where
    F: Fn(&mut dyn Repl, &str) -> Result<(), Error> + Send + Sync + 'static,
{
    Arc::new(function)
}

/// Parse a handler's argument line the way the Python handlers use shlex.
pub fn handler_words(rest: &str) -> Result<Vec<String>, Error> {
    split_words(rest)
        .map_err(|exc| Error::usage(format!("Could not parse the command: {}", exc), None))
}

/// Wire the orchestration commands that live in this crate. Safe to call
/// more than once. `team` and `tool_protocol` register themselves from their
/// own modules (owned by the sibling worker); the CLI calls their
/// `register` after this one.
pub fn install_repl_features(registry: &mut ReplRegistry) {
    if registry.get("target").is_some() {
        return;
    }
    crate::targets::register(registry);
    crate::budgets::register(registry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::{Chunk, EmptyKeyStore, Provider, Session};

    struct FakeBackend(Vec<Chunk>);

    impl ChatBackend for FakeBackend {
        fn stream(&self, _request: &ChatRequest) -> Result<Vec<Chunk>, Error> {
            Ok(self.0.clone())
        }
    }

    fn target() -> Target {
        Target::new("mock", "m", "http://127.0.0.1/v1", Some("fast"))
    }

    #[test]
    fn snapshot_freezes_history_and_ignores_later_edits() {
        let mut session = Session::new(
            Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1"),
            "m",
        );
        session.messages.push(Message::new("user", "earlier"));
        let snapshot = snapshot_from_session(&session, None, false);
        session.messages.push(Message::new("assistant", "late"));
        assert_eq!(snapshot.history.len(), 1);
        assert_eq!(snapshot.history[0].1, "earlier");
        let _ = EmptyKeyStore;
    }

    #[test]
    fn attachment_store_rejects_dotdot_and_keeps_order() {
        let mut store = AttachmentStore::default();
        store
            .approve(ApprovedAttachment::new("b.txt", "hash", "B"))
            .unwrap();
        store
            .approve(ApprovedAttachment::new("a.txt", "hash", "A"))
            .unwrap();
        assert!(store
            .approve(ApprovedAttachment::new("../x", "h", "C"))
            .is_err());
        let list = store.list();
        assert_eq!(list[0].path, "a.txt");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn run_request_collects_text_usage_and_finish_reason() {
        let backend = FakeBackend(vec![
            Chunk {
                text: "hel".into(),
                ..Default::default()
            },
            Chunk {
                text: "lo".into(),
                ..Default::default()
            },
            Chunk {
                usage: Some(Usage {
                    input_tokens: Some(3),
                    output_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Chunk {
                finish_reason: Some("stop".into()),
                ..Default::default()
            },
        ]);
        let provider = Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1");
        let snapshot = make_snapshot(
            &[Message::new("user", "hi")],
            None,
            None,
            None,
            None,
            false,
            None,
        );
        let result = run_request(
            &backend,
            &provider,
            &target(),
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
        assert_eq!(result.status, "ok");
        assert_eq!(result.text, "hello");
        assert_eq!(result.usage.as_ref().map(|u| u.input_tokens), Some(Some(3)));
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        let public = result.public_value();
        assert_eq!(public["alias"], "fast");
        assert!(public["usage"].is_object());
    }

    #[test]
    fn run_request_marks_cancel_before_send_and_empty_answers() {
        let backend = FakeBackend(vec![]);
        let provider = Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1");
        let snapshot = make_snapshot(&[], None, None, None, None, false, None);
        let flag = Arc::new(AtomicBool::new(true));
        let cancelled = run_request(
            &backend,
            &provider,
            &target(),
            &snapshot,
            &RunOptions::new(5.0),
            RunCall {
                credential: None,
                prompt: Some("x"),
                messages: None,
                cancel: Some(flag),
                events: None,
            },
        );
        assert_eq!(cancelled.status, "cancelled");

        let empty = run_request(
            &backend,
            &provider,
            &target(),
            &snapshot,
            &RunOptions::new(5.0),
            RunCall {
                credential: None,
                prompt: Some("x"),
                messages: None,
                cancel: None,
                events: None,
            },
        );
        assert_eq!(empty.status, "error");
        assert!(empty.incomplete);
        assert_eq!(
            empty.error.as_deref(),
            Some("The provider returned an empty response.")
        );
    }

    #[test]
    fn run_request_reasoning_only_is_incomplete_not_error() {
        let backend = FakeBackend(vec![Chunk {
            reasoning: "thinking".into(),
            ..Default::default()
        }]);
        let provider = Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1");
        let snapshot = make_snapshot(&[], None, None, None, None, false, None);
        let result = run_request(
            &backend,
            &provider,
            &target(),
            &snapshot,
            &RunOptions::new(5.0),
            RunCall {
                credential: None,
                prompt: Some("x"),
                messages: None,
                cancel: None,
                events: None,
            },
        );
        assert_eq!(result.status, "incomplete");
        assert!(result.text.is_empty());
    }

    #[test]
    fn run_request_length_cap_with_no_text_is_incomplete() {
        let backend = FakeBackend(vec![Chunk {
            finish_reason: Some("length".into()),
            ..Default::default()
        }]);
        let provider = Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1");
        let snapshot = make_snapshot(&[], None, None, None, None, false, None);
        let result = run_request(
            &backend,
            &provider,
            &target(),
            &snapshot,
            &RunOptions::new(5.0),
            RunCall {
                credential: None,
                prompt: Some("x"),
                messages: None,
                cancel: None,
                events: None,
            },
        );
        assert_eq!(result.status, "incomplete");
        assert_eq!(
            result.error.as_deref(),
            Some("The output cap was exhausted before any answer text.")
        );
    }

    #[test]
    fn run_request_network_error_maps_to_timeout_or_error() {
        struct FailBackend;
        impl ChatBackend for FailBackend {
            fn stream(&self, _request: &ChatRequest) -> Result<Vec<Chunk>, Error> {
                Err(Error::network("connection timed out", None))
            }
        }
        let provider = Provider::new("mock", "Mock", crate::sys::OPENAI, "http://x/v1");
        let snapshot = make_snapshot(&[], None, None, None, None, false, None);
        let result = run_request(
            &FailBackend,
            &provider,
            &target(),
            &snapshot,
            &RunOptions::new(5.0),
            RunCall {
                credential: None,
                prompt: Some("x"),
                messages: None,
                cancel: None,
                events: None,
            },
        );
        assert_eq!(result.status, "timeout");
        assert!(result.incomplete);
    }

    #[test]
    fn feature_files_version_roundtrip() {
        let dir = std::env::temp_dir().join(format!("orch-seam-{}", crate::sys::new_id()));
        let path = dir.join("feature.json");
        let mut data = Map::new();
        data.insert("members".to_string(), Value::from(1));
        save_feature(&path, data).unwrap();
        let loaded = load_feature(&path, "team").unwrap();
        assert_eq!(loaded["members"], Value::from(1));
        assert_eq!(loaded["version"], Value::from(1));
        let _ = std::fs::remove_dir_all(&dir);

        let bad = std::env::temp_dir().join(format!("orch-seam-{}", crate::sys::new_id()));
        std::fs::create_dir_all(&bad).unwrap();
        let bad_path = bad.join("feature.json");
        std::fs::write(&bad_path, "{\"version\": 2, \"x\": 1}").unwrap();
        let error = load_feature(&bad_path, "team").unwrap_err();
        assert!(error.message.contains("version 2"));
        let _ = std::fs::remove_dir_all(&bad);
    }
}
