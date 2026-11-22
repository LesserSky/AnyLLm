//! Provider adapters.
//!
//! Port of Python `anyllm/adapters.py`. Every adapter turns one provider's
//! streaming protocol into the same small `Chunk` stream. The rest of the app
//! never sees provider-specific JSON.

use std::collections::VecDeque;
use std::io::Read;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::catalog::{Provider, ANTHROPIC, GEMINI, OPENAI};
use crate::errors::{AnyLLMError, ErrorKind, Result};
use crate::transport::{
    iter_sse_controlled, join_url, open_stream_controlled, request_json, ControlledSseIterator,
    SseReader,
};

/// One conversation turn as sent to a provider.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// OpenAI-compatible assistant tool call records, preserved when tools are enabled.
    pub tool_calls: Option<Value>,
    /// The call ID for a role=`tool` result message.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn with_tool_calls(mut self, calls: Value) -> Self {
        self.tool_calls = Some(calls);
        self
    }

    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

/// Token usage reported by a provider. Missing numbers stay `None` and are
/// never shown as zero.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
}

impl Usage {
    /// Keep the latest (usually cumulative) values from a stream.
    pub fn update(&mut self, other: &Usage) {
        self.input_tokens = max_of(self.input_tokens, other.input_tokens);
        self.output_tokens = max_of(self.output_tokens, other.output_tokens);
        self.reasoning_tokens = max_of(self.reasoning_tokens, other.reasoning_tokens);
        self.cached_tokens = max_of(self.cached_tokens, other.cached_tokens);
    }

    /// Add two usage reports, for session totals.
    pub fn combined(&self, other: &Usage) -> Usage {
        Usage {
            input_tokens: sum_of(self.input_tokens, other.input_tokens),
            output_tokens: sum_of(self.output_tokens, other.output_tokens),
            reasoning_tokens: sum_of(self.reasoning_tokens, other.reasoning_tokens),
            cached_tokens: sum_of(self.cached_tokens, other.cached_tokens),
        }
    }

    pub fn total(&self) -> Option<i64> {
        if self.input_tokens.is_none() && self.output_tokens.is_none() {
            return None;
        }
        Some(self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0))
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "input": self.input_tokens,
            "output": self.output_tokens,
            "reasoning": self.reasoning_tokens,
            "cached": self.cached_tokens,
            "total": self.total(),
        })
    }

    pub fn from_dict(raw: &Value) -> Usage {
        let get = |key: &str| int_field(raw.get(key));
        Usage {
            input_tokens: get("input"),
            output_tokens: get("output"),
            reasoning_tokens: get("reasoning"),
            cached_tokens: get("cached"),
        }
    }
}

fn max_of(current: Option<i64>, incoming: Option<i64>) -> Option<i64> {
    match (current, incoming) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (None, Some(b)) => Some(b),
        (a, None) => a,
    }
}

fn sum_of(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// One streamed fragment of a tool call. Fragments share an index.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallDelta {
    pub index: i64,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One piece of a streamed response.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub reasoning: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub tool_call: Option<ToolCallDelta>,
}

impl Chunk {
    pub fn text(value: impl Into<String>) -> Self {
        Chunk {
            text: value.into(),
            ..Default::default()
        }
    }
    pub fn reasoning(value: impl Into<String>) -> Self {
        Chunk {
            reasoning: value.into(),
            ..Default::default()
        }
    }
}

/// JSON int-or-digit-string-or-float coercion (Python `_int`).
pub fn int_field(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Bool(_)) | None => None,
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Some(Value::String(s)) if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) => {
            s.parse().ok()
        }
        _ => None,
    }
}

// -- message shaping -----------------------------------------------------------

/// Keep user/assistant turns with non-empty text content.
fn normalize_messages(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|m| (m.role == "user" || m.role == "assistant") && !m.content.trim().is_empty())
        .cloned()
        .collect()
}

/// Anthropic wants user/assistant turns that alternate.
fn anthropic_messages(messages: &[Message]) -> Vec<Message> {
    let normalized = normalize_messages(messages);
    let mut result: Vec<Message> = Vec::new();
    for message in normalized {
        if let Some(last) = result.last_mut() {
            if last.role == message.role {
                last.content.push_str("\n\n");
                last.content.push_str(&message.content);
                continue;
            }
        }
        result.push(message);
    }
    if result.first().map(|m| m.role.as_str()) == Some("assistant") {
        result.insert(0, Message::new("user", "(continuing the conversation)"));
    }
    result
}

/// Gemini contents: `user` and `model` roles.
fn gemini_contents(messages: &[Message]) -> Vec<Value> {
    let mut contents: Vec<Value> = Vec::new();
    for message in normalize_messages(messages) {
        let role = if message.role == "user" {
            "user"
        } else {
            "model"
        };
        if contents.is_empty() && role == "model" {
            contents.push(json!({
                "role": "user",
                "parts": [{"text": "(continuing the conversation)"}],
            }));
        }
        contents.push(json!({
            "role": role,
            "parts": [{"text": message.content}],
        }));
    }
    contents
}

fn messages_value(messages: &[Message], system: Option<&str>) -> Vec<Value> {
    let mut conversation: Vec<Value> = Vec::new();
    if let Some(system) = system {
        if !system.trim().is_empty() {
            conversation.push(json!({"role": "system", "content": system}));
        }
    }
    for message in normalize_messages(messages) {
        conversation.push(json!({"role": message.role, "content": message.content}));
    }
    conversation
}

/// Keep tool-call turns for OpenAI-compatible requests when tools are enabled.
fn openai_tool_messages_value(messages: &[Message], system: Option<&str>) -> Vec<Value> {
    let mut conversation: Vec<Value> = Vec::new();
    if let Some(system) = system {
        if !system.trim().is_empty() {
            conversation.push(json!({"role": "system", "content": system}));
        }
    }
    for message in messages {
        match message.role.as_str() {
            "user" | "assistant" => {
                let calls = message
                    .tool_calls
                    .as_ref()
                    .filter(|calls| calls.as_array().is_some_and(|items| !items.is_empty()));
                if !message.content.trim().is_empty() || calls.is_some() {
                    let mut item = json!({"role": message.role, "content": message.content});
                    if let Some(calls) = calls {
                        item["tool_calls"] = calls.clone();
                    }
                    conversation.push(item);
                }
            }
            "tool" => conversation.push(json!({
                "role": "tool",
                "tool_call_id": message.tool_call_id.as_deref().unwrap_or(""),
                "content": message.content,
            })),
            _ => {}
        }
    }
    conversation
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().map(|n| n != 0.0).unwrap_or(true),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

// -- usage parsers ---------------------------------------------------------------

fn openai_usage(raw: Option<&Value>) -> Option<Usage> {
    let raw = raw?;
    let usage = Usage {
        input_tokens: int_field(raw.get("prompt_tokens").or_else(|| raw.get("input_tokens"))),
        output_tokens: int_field(
            raw.get("completion_tokens")
                .or_else(|| raw.get("output_tokens")),
        ),
        reasoning_tokens: raw
            .get("completion_tokens_details")
            .and_then(|d| int_field(d.get("reasoning_tokens"))),
        cached_tokens: raw
            .get("prompt_tokens_details")
            .and_then(|d| int_field(d.get("cached_tokens"))),
    };
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() {
        return None;
    }
    Some(usage)
}

fn anthropic_usage(raw: Option<&Value>) -> Option<Usage> {
    let raw = raw?;
    let usage = Usage {
        input_tokens: int_field(raw.get("input_tokens")),
        output_tokens: int_field(raw.get("output_tokens")),
        cached_tokens: int_field(raw.get("cache_read_input_tokens")),
        reasoning_tokens: None,
    };
    if usage.input_tokens.is_none()
        && usage.output_tokens.is_none()
        && usage.cached_tokens.is_none()
    {
        return None;
    }
    Some(usage)
}

fn gemini_usage(raw: Option<&Value>) -> Option<Usage> {
    let raw = raw?;
    let usage = Usage {
        input_tokens: int_field(raw.get("promptTokenCount")),
        output_tokens: int_field(raw.get("candidatesTokenCount")),
        reasoning_tokens: int_field(raw.get("thoughtsTokenCount")),
        cached_tokens: int_field(raw.get("cachedContentTokenCount")),
    };
    if usage.input_tokens.is_none()
        && usage.output_tokens.is_none()
        && usage.reasoning_tokens.is_none()
    {
        return None;
    }
    Some(usage)
}

fn error_text(detail: &Value) -> String {
    fn python_string(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Null => "None".to_string(),
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::Number(n) => n.to_string(),
            _ => value.to_string(),
        }
    }
    fn truthy(value: &Value) -> bool {
        match value {
            Value::Null => false,
            Value::Bool(v) => *v,
            Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(true),
            Value::String(s) => !s.is_empty(),
            Value::Array(v) => !v.is_empty(),
            Value::Object(v) => !v.is_empty(),
        }
    }
    match detail {
        Value::Object(map) => map
            .get("message")
            .filter(|value| truthy(value))
            .or_else(|| map.get("type").filter(|value| truthy(value)))
            .map(python_string)
            .unwrap_or_else(|| python_string(detail)),
        other => python_string(other),
    }
}

fn content_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.as_object()
                    .and_then(|m| m.get("text"))
                    .and_then(Value::as_str)
            })
            .collect(),
        _ => String::new(),
    }
}

// -- stream_options rejection ----------------------------------------------------

fn compact_alnum(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn mentions_stream_options(parts: &[&str]) -> bool {
    parts
        .iter()
        .any(|part| !part.is_empty() && compact_alnum(part).contains("streamoption"))
}

fn mentions_include_usage(parts: &[&str]) -> bool {
    parts
        .iter()
        .any(|part| !part.is_empty() && compact_alnum(part).contains("includeusage"))
}

/// A 400/422 that says `stream_options` itself was rejected. The follow-up
/// request omits the field; other client errors are not replayed.
fn stream_options_rejected(err: &AnyLLMError) -> bool {
    let status = err.status.unwrap_or(0);
    if status != 400 && status != 422 {
        return false;
    }
    let param = err.param.clone().unwrap_or_default();
    let code = err.code.clone().unwrap_or_default();
    let message = err.message.clone();
    let blob_parts = [param.as_str(), code.as_str(), message.as_str()];
    if !mentions_stream_options(&blob_parts) && !mentions_include_usage(&blob_parts) {
        return false;
    }
    if mentions_stream_options(&[param.as_str()]) || mentions_stream_options(&[code.as_str()]) {
        return true;
    }
    let compact_message = compact_alnum(&message);
    if ["streamoption", "streamoptions", "includeusage"].contains(&compact_message.as_str()) {
        return true;
    }
    let blob = blob_parts.join(" ");
    let compact = compact_alnum(&blob);
    [
        "unsupported",
        "notsupported",
        "unrecognized",
        "unexpected",
        "unknownparameter",
        "unknownparam",
        "unknownargument",
        "unknownfield",
        "unknownproperty",
        "additionalpropert",
        "extrafield",
        "extrainput",
        "extrapropert",
        "notpermitted",
        "notallowed",
        "invalidparameter",
        "invalidparam",
        "invalidargument",
        "invalidfield",
        "invalidrequestargument",
    ]
    .iter()
    .any(|needle| compact.contains(needle))
}

fn token_field(model: &str) -> &'static str {
    let lowered = model.to_lowercase();
    if ["o1", "o3", "o4", "gpt-5"]
        .iter()
        .any(|prefix| lowered.starts_with(prefix))
    {
        "max_completion_tokens"
    } else {
        "max_tokens"
    }
}

fn percent_encode(value: &str) -> String {
    let mut out = String::new();
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

fn check_base_url(base_url: &str, provider: &Provider) -> Result<String> {
    if base_url.is_empty() {
        return Err(AnyLLMError::api(
            format!("Provider {:?} has no base URL.", provider.name),
            None,
            None,
            None,
        )
        .with_hint("Pass --base-url, or add the provider with `anyllm providers add`."));
    }
    Ok(base_url.to_string())
}

fn accept_sse() -> Vec<(String, String)> {
    vec![("Accept".to_string(), "text/event-stream".to_string())]
}

fn controlled_stop_error(cancelled: bool) -> AnyLLMError {
    let message = if cancelled {
        "Streaming stopped locally after a cancel. The provider request was not cancelled."
    } else {
        "The overall deadline elapsed while streaming. The provider request was not cancelled."
    };
    AnyLLMError::network(message).with_hint(
        "The provider may still be generating and may still charge for this request. Stopping the local read does not cancel it on the server.",
    )
}

fn request_timeout(args: &StreamArgs<'_>) -> Result<f64> {
    if args
        .cancel
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
    {
        return Err(controlled_stop_error(true));
    }
    let Some(deadline) = args.deadline else {
        return Ok(args.timeout);
    };
    let remaining = deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f64();
    if remaining <= 0.0 {
        return Err(controlled_stop_error(false));
    }
    Ok(args.timeout.min(remaining))
}

fn open_adapter_stream(
    url: &str,
    headers: &[(String, String)],
    payload: &Value,
    args: &StreamArgs<'_>,
) -> Result<Box<dyn Read + Send + Sync>> {
    open_stream_controlled(
        url,
        headers,
        payload,
        request_timeout(args)?,
        2,
        args.deadline,
        args.cancel.clone(),
    )
}

fn chunk_stream(
    kind: &str,
    label: &str,
    reader: Box<dyn Read + Send + Sync>,
    args: &StreamArgs<'_>,
) -> ChunkStream {
    if args.deadline.is_some() || args.cancel.is_some() {
        ChunkStream::from_controlled_reader(
            kind,
            label,
            reader,
            args.deadline,
            args.cancel
                .clone()
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
        )
    } else {
        ChunkStream::from_reader(kind, label, reader)
    }
}

// -- streaming -------------------------------------------------------------------

/// Arguments for one completion request.
#[derive(Debug, Clone)]
pub struct StreamArgs<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub system: Option<&'a str>,
    pub key: Option<&'a str>,
    pub base_url: Option<&'a str>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub timeout: f64,
    /// Include `stream_options.include_usage` on OpenAI-kind requests.
    pub include_usage: bool,
    /// Optional OpenAI-compatible tools payload. Empty JSON arrays are omitted.
    pub tools: Option<&'a Value>,
    /// Absolute deadline shared by request setup and streamed reads.
    pub deadline: Option<Instant>,
    /// Cooperative cancellation flag shared with the orchestration request.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl<'a> Default for StreamArgs<'a> {
    fn default() -> Self {
        StreamArgs {
            model: "",
            messages: &[],
            system: None,
            key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            timeout: 120.0,
            include_usage: true,
            tools: None,
            deadline: None,
            cancel: None,
        }
    }
}

/// A live chunk stream. Pull chunks with `next_chunk` until it returns `None`.
pub struct ChunkStream {
    reader: ChunkEventReader,
    kind: String,
    label: String,
    pending: VecDeque<Chunk>,
    finished: bool,
}

enum ChunkEventReader {
    Blocking(SseReader<Box<dyn Read + Send + Sync>>),
    Controlled(ControlledSseIterator),
}

impl ChunkEventReader {
    fn next_event(&mut self) -> Option<Result<String>> {
        match self {
            Self::Blocking(reader) => match reader.next_event() {
                Ok(Some(payload)) => Some(Ok(payload)),
                Ok(None) => None,
                Err(error) => Some(Err(AnyLLMError::network(format!(
                    "connection broke while streaming: {}",
                    error
                ))
                .with_hint(
                    "The provider may have timed out. Retry, or use --timeout with a larger value.",
                ))),
            },
            Self::Controlled(reader) => reader.next(),
        }
    }
}

impl ChunkStream {
    /// Wrap an already-open SSE body (used by tests with canned bytes).
    pub fn from_reader(kind: &str, label: &str, reader: Box<dyn Read + Send + Sync>) -> Self {
        ChunkStream {
            reader: ChunkEventReader::Blocking(SseReader::new(reader)),
            kind: kind.to_string(),
            label: label.to_string(),
            pending: VecDeque::new(),
            finished: false,
        }
    }

    /// Wrap an open SSE response with cooperative deadline and cancellation.
    ///
    /// Cancellation stops local delivery; an underlying blocking read may
    /// remain active until the transport returns or its I/O timeout expires.
    pub fn from_controlled_reader(
        kind: &str,
        label: &str,
        reader: Box<dyn Read + Send + Sync>,
        deadline: Option<Instant>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        ChunkStream {
            reader: ChunkEventReader::Controlled(iter_sse_controlled(reader, deadline, cancelled)),
            kind: kind.to_string(),
            label: label.to_string(),
            pending: VecDeque::new(),
            finished: false,
        }
    }

    pub fn next_chunk(&mut self) -> Result<Option<Chunk>> {
        loop {
            if let Some(chunk) = self.pending.pop_front() {
                return Ok(Some(chunk));
            }
            if self.finished {
                return Ok(None);
            }
            let payload = match self.reader.next_event() {
                Some(Ok(payload)) => payload,
                None => {
                    self.finished = true;
                    return Ok(None);
                }
                Some(Err(error)) => {
                    self.finished = true;
                    return Err(error);
                }
            };
            let event: Value = match serde_json::from_str(&payload) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let event = if event.is_array() {
                event
                    .as_array()
                    .and_then(|items| items.last())
                    .cloned()
                    .unwrap_or_else(|| json!({}))
            } else {
                event
            };
            let map = match event.as_object() {
                Some(map) => map,
                None => continue,
            };
            match self.kind.as_str() {
                OPENAI => self.parse_openai(&self.label.clone(), map)?,
                ANTHROPIC => self.parse_anthropic(&self.label.clone(), map)?,
                GEMINI => self.parse_gemini(&self.label.clone(), map)?,
                other => {
                    self.finished = true;
                    return Err(AnyLLMError::api(
                        format!("Provider kind {:?} is not supported for streaming.", other),
                        None,
                        None,
                        None,
                    ));
                }
            }
        }
    }

    fn provider_error(&self, label: &str, detail: &str) -> AnyLLMError {
        AnyLLMError::provider(format!("{} error: {}", label, detail))
    }

    fn parse_openai(&mut self, label: &str, event: &Map<String, Value>) -> Result<()> {
        if let Some(error) = event.get("error").filter(|v| !v.is_null()) {
            self.finished = true;
            return Err(self.provider_error(label, &error_text(error)));
        }
        if let Some(usage) = openai_usage(event.get("usage")) {
            self.pending.push_back(Chunk {
                usage: Some(usage),
                ..Default::default()
            });
        }
        if let Some(Value::Array(choices)) = event.get("choices") {
            for choice in choices {
                let choice = match choice.as_object() {
                    Some(choice) => choice,
                    None => continue,
                };
                if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                    let text = content_text(delta.get("content").unwrap_or(&Value::Null));
                    if !text.is_empty() {
                        self.pending.push_back(Chunk::text(text));
                    }
                    let reasoning = delta
                        .get("reasoning_content")
                        .or_else(|| delta.get("reasoning"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !reasoning.is_empty() {
                        self.pending.push_back(Chunk::reasoning(reasoning));
                    }
                    if let Some(Value::Array(calls)) = delta.get("tool_calls") {
                        for call in calls {
                            let call = match call.as_object() {
                                Some(call) => call,
                                None => continue,
                            };
                            let function = call
                                .get("function")
                                .and_then(Value::as_object)
                                .cloned()
                                .unwrap_or_default();
                            let index = int_field(call.get("index")).unwrap_or(0);
                            self.pending.push_back(Chunk {
                                tool_call: Some(ToolCallDelta {
                                    index,
                                    id: call
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string(),
                                    name: function
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string(),
                                    arguments: function
                                        .get("arguments")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string(),
                                }),
                                ..Default::default()
                            });
                        }
                    }
                }
                if let Some(finish) = choice.get("finish_reason").and_then(Value::as_str) {
                    if !finish.is_empty() {
                        self.pending.push_back(Chunk {
                            finish_reason: Some(finish.to_string()),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_anthropic(&mut self, label: &str, event: &Map<String, Value>) -> Result<()> {
        let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");
        match event_type {
            "error" => {
                self.finished = true;
                return Err(self.provider_error(
                    label,
                    &error_text(event.get("error").unwrap_or(&Value::Null)),
                ));
            }
            "message_start" => {
                if let Some(usage) = event
                    .get("message")
                    .and_then(|m| m.as_object())
                    .and_then(|m| anthropic_usage(m.get("usage")))
                {
                    self.pending.push_back(Chunk {
                        usage: Some(usage),
                        ..Default::default()
                    });
                }
            }
            "content_block_delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let delta_type = delta.get("type").and_then(Value::as_str).unwrap_or("");
                if delta_type == "text_delta" {
                    let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                    if !text.is_empty() {
                        self.pending.push_back(Chunk::text(text));
                    }
                } else if delta_type == "thinking_delta" {
                    let thought = delta.get("thinking").and_then(Value::as_str).unwrap_or("");
                    if !thought.is_empty() {
                        self.pending.push_back(Chunk::reasoning(thought));
                    }
                }
            }
            "message_delta" => {
                let usage = anthropic_usage(event.get("usage"));
                let finish = event
                    .get("delta")
                    .and_then(Value::as_object)
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if usage.is_some() || finish.is_some() {
                    self.pending.push_back(Chunk {
                        usage,
                        finish_reason: finish,
                        ..Default::default()
                    });
                }
            }
            "message_stop" => {
                self.finished = true;
            }
            _ => {}
        }
        Ok(())
    }

    fn parse_gemini(&mut self, label: &str, event: &Map<String, Value>) -> Result<()> {
        if let Some(error) = event.get("error").filter(|v| !v.is_null()) {
            self.finished = true;
            return Err(self.provider_error(label, &error_text(error)));
        }
        let feedback = event
            .get("promptFeedback")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(blocked) = feedback.get("blockReason").and_then(Value::as_str) {
            self.finished = true;
            return Err(AnyLLMError::provider(format!(
                "{} blocked the prompt ({}).",
                label, blocked
            ))
            .with_hint("Rephrase the prompt without restricted content."));
        }
        if let Some(usage) = gemini_usage(event.get("usageMetadata")) {
            self.pending.push_back(Chunk {
                usage: Some(usage),
                ..Default::default()
            });
        }
        if let Some(Value::Array(candidates)) = event.get("candidates") {
            for candidate in candidates {
                let candidate = match candidate.as_object() {
                    Some(candidate) => candidate,
                    None => continue,
                };
                let content = candidate
                    .get("content")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if let Some(Value::Array(parts)) = content.get("parts") {
                    for part in parts {
                        let part = match part.as_object() {
                            Some(part) => part,
                            None => continue,
                        };
                        let text = match part.get("text").and_then(Value::as_str) {
                            Some(text) if !text.is_empty() => text,
                            _ => continue,
                        };
                        if part
                            .get("thought")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            self.pending.push_back(Chunk::reasoning(text));
                        } else {
                            self.pending.push_back(Chunk::text(text));
                        }
                    }
                }
                if let Some(finish) = candidate.get("finishReason").and_then(Value::as_str) {
                    let mapped = match finish {
                        "STOP" => "stop".to_string(),
                        "MAX_TOKENS" => "length".to_string(),
                        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" => {
                            "content_filter".to_string()
                        }
                        "OTHER" => "stop".to_string(),
                        other => other.to_lowercase(),
                    };
                    self.pending.push_back(Chunk {
                        finish_reason: Some(mapped),
                        ..Default::default()
                    });
                }
            }
        }
        Ok(())
    }
}

/// Yield response chunks for one completion request.
///
/// Errors from opening the stream surface before the first chunk. `kind`
/// dispatching matches Python `stream_chat`.
pub fn stream_chat(provider: &Provider, args: &StreamArgs) -> Result<ChunkStream> {
    let base = check_base_url(
        args.base_url
            .filter(|base| !base.is_empty())
            .unwrap_or(&provider.base_url),
        provider,
    )?;
    let tools = args.tools.filter(|tools| json_truthy(tools));
    if tools.is_some() && provider.kind != OPENAI {
        return Err(AnyLLMError::api(
            format!(
                "Tool calls are not implemented for provider kind '{}'.",
                provider.kind
            ),
            None,
            None,
            None,
        )
        .with_hint("Support stays unknown for this kind. No tools payload was sent."));
    }
    let mut headers = provider.auth_headers(args.key);
    headers.extend(accept_sse());
    match provider.kind.as_str() {
        OPENAI => {
            let url = join_url(&base, "chat/completions");
            let mut payload = json!({
                "model": args.model,
                "messages": if tools.is_some() {
                    openai_tool_messages_value(args.messages, args.system)
                } else {
                    messages_value(args.messages, args.system)
                },
                "stream": true,
            });
            if let Some(tools) = tools {
                payload["tools"] = tools.clone();
            }
            if let Some(temperature) = args.temperature {
                payload["temperature"] = json!(temperature);
            }
            if let Some(max_tokens) = args.max_tokens {
                if max_tokens != 0 {
                    payload[token_field(args.model)] = json!(max_tokens);
                }
            }
            if args.include_usage {
                payload["stream_options"] = json!({"include_usage": true});
            }
            let reader = match open_adapter_stream(&url, &headers, &payload, args) {
                Ok(reader) => reader,
                Err(err) => {
                    // Some OpenAI-compatible servers reject stream_options.
                    // Retry once without it when the 400/422 names that field.
                    if args.include_usage
                        && payload.get("stream_options").is_some()
                        && stream_options_rejected(&err)
                    {
                        let mut payload = payload;
                        payload
                            .as_object_mut()
                            .expect("payload is an object")
                            .remove("stream_options");
                        open_adapter_stream(&url, &headers, &payload, args)?
                    } else {
                        return Err(err);
                    }
                }
            };
            Ok(chunk_stream(OPENAI, &provider.label, reader, args))
        }
        ANTHROPIC => {
            if args.key.map(|k| k.is_empty()).unwrap_or(true) {
                return Err(AnyLLMError::auth("Anthropic requires an API key.")
                    .with_hint("Run `anyllm keys set anthropic`, or export ANTHROPIC_API_KEY."));
            }
            let url = join_url(&base, "messages");
            let messages = anthropic_messages(args.messages)
                .into_iter()
                .map(|m| json!({"role": m.role, "content": m.content}))
                .collect::<Vec<_>>();
            let mut payload = json!({
                "model": args.model,
                "max_tokens": args.max_tokens.filter(|value| *value != 0).unwrap_or(4096),
                "messages": messages,
                "stream": true,
            });
            if let Some(system) = args.system {
                if !system.trim().is_empty() {
                    payload["system"] = json!(system);
                }
            }
            if let Some(temperature) = args.temperature {
                payload["temperature"] = json!(temperature);
            }
            let mut headers = provider.auth_headers(args.key);
            headers.push(("anthropic-version".to_string(), "2023-06-01".to_string()));
            headers.extend(accept_sse());
            let reader = open_adapter_stream(&url, &headers, &payload, args)?;
            Ok(chunk_stream(ANTHROPIC, &provider.label, reader, args))
        }
        GEMINI => {
            if args.key.map(|k| k.is_empty()).unwrap_or(true) {
                return Err(AnyLLMError::auth("Google Gemini requires an API key.")
                    .with_hint("Run `anyllm keys set google`, or export GEMINI_API_KEY."));
            }
            let model = args.model.strip_prefix("models/").unwrap_or(args.model);
            let url = join_url(
                &base,
                &format!(
                    "models/{}:streamGenerateContent?alt=sse",
                    percent_encode(model)
                ),
            );
            let mut payload = json!({
                "contents": gemini_contents(args.messages),
            });
            if let Some(system) = args.system {
                if !system.trim().is_empty() {
                    payload["systemInstruction"] = json!({"parts": [{"text": system}]});
                }
            }
            let mut generation = Map::new();
            if let Some(temperature) = args.temperature {
                generation.insert("temperature".into(), json!(temperature));
            }
            if let Some(max_tokens) = args.max_tokens {
                if max_tokens != 0 {
                    generation.insert("maxOutputTokens".into(), json!(max_tokens));
                }
            }
            if !generation.is_empty() {
                payload["generationConfig"] = Value::Object(generation);
            }
            let mut headers = provider.auth_headers(args.key);
            headers.extend(accept_sse());
            let reader = open_adapter_stream(&url, &headers, &payload, args)?;
            Ok(chunk_stream(GEMINI, &provider.label, reader, args))
        }
        other => Err(AnyLLMError::api(
            format!(
                "Provider {:?} uses unsupported kind {:?}.",
                provider.name, other
            ),
            None,
            None,
            None,
        )
        .with_hint("Supported kinds: openai, anthropic, gemini.")),
    }
}

/// Ask the provider which models this key can use.
pub fn list_models(
    provider: &Provider,
    key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
) -> Result<Vec<String>> {
    let base = check_base_url(
        base_url
            .filter(|base| !base.is_empty())
            .unwrap_or(&provider.base_url),
        provider,
    )?;
    let mut headers = provider.auth_headers(key);
    headers.push(("Accept".to_string(), "application/json".to_string()));

    let items: Vec<Value> = match provider.kind.as_str() {
        OPENAI => {
            let data = request_json(
                &join_url(&base, "models"),
                &headers,
                None,
                "GET",
                timeout,
                2,
            )?;
            match data {
                Value::Object(map) => map
                    .get("data")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                Value::Array(items) => items,
                _ => Vec::new(),
            }
        }
        ANTHROPIC => {
            if key.map(|k| k.is_empty()).unwrap_or(true) {
                return Err(
                    AnyLLMError::auth("Anthropic requires an API key to list models.")
                        .with_hint("Run `anyllm keys set anthropic`."),
                );
            }
            headers.push(("anthropic-version".to_string(), "2023-06-01".to_string()));
            let data = request_json(
                &join_url(&base, "models?limit=1000"),
                &headers,
                None,
                "GET",
                timeout,
                2,
            )?;
            match data {
                Value::Object(map) => map
                    .get("data")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                Value::Array(items) => items,
                _ => Vec::new(),
            }
        }
        GEMINI => {
            if key.map(|k| k.is_empty()).unwrap_or(true) {
                return Err(
                    AnyLLMError::auth("Google Gemini requires an API key to list models.")
                        .with_hint("Run `anyllm keys set google`."),
                );
            }
            let data = request_json(
                &join_url(&base, "models?pageSize=1000"),
                &headers,
                None,
                "GET",
                timeout,
                2,
            )?;
            let items = match data {
                Value::Object(map) => map
                    .get("models")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                Value::Array(items) => items,
                _ => Vec::new(),
            };
            items
                .into_iter()
                .filter(|item| {
                    match item
                        .as_object()
                        .and_then(|m| m.get("supportedGenerationMethods"))
                    {
                        Some(Value::Array(methods)) if !methods.is_empty() => methods
                            .iter()
                            .any(|method| method.as_str() == Some("generateContent")),
                        Some(Value::String(methods)) if !methods.is_empty() => {
                            methods.contains("generateContent")
                        }
                        _ => true,
                    }
                })
                .collect()
        }
        other => {
            return Err(AnyLLMError::api(
                format!("Unsupported provider kind {:?}.", other),
                None,
                None,
                None,
            ));
        }
    };

    let mut names: Vec<String> = Vec::new();
    for item in items {
        match item {
            Value::String(name) => names.push(name),
            Value::Object(map) => {
                let value = map
                    .get("id")
                    .or_else(|| map.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .strip_prefix("models/")
                    .unwrap_or_else(|| {
                        map.get("id")
                            .or_else(|| map.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    })
                    .to_string();
                if !value.is_empty() {
                    names.push(value);
                }
            }
            _ => {}
        }
    }
    names.sort_by_key(|name| name.to_lowercase());
    names.dedup();
    Ok(names)
}

/// Kind marker for callers that classify stream errors.
#[allow(dead_code)]
fn kind_of(err: &AnyLLMError) -> ErrorKind {
    err.kind
}
