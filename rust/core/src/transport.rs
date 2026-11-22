//! HTTP, retries, error mapping, and server-sent-event parsing.
//!
//! Port of Python `anyllm/transport.py` over `ureq` 2.x. Streaming reads the
//! response body incrementally via `response.into_reader()`.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use serde_json::{Map, Value};

use crate::errors::{AnyLLMError, Result};
use crate::VERSION;

/// The server asked the client to try again and did not apply the request.
/// Safe for a completion POST, which is not idempotent.
const SAFE_RETRY_STATUS: [u16; 4] = [408, 425, 429, 503];

/// Retry only for idempotent calls (GET, HEAD): the request may have landed.
const IDEMPOTENT_RETRY_STATUS: [u16; 3] = [500, 502, 504];

pub const MAX_RETRIES: usize = 2;
pub const MAX_RETRY_WAIT: f64 = 8.0;

pub fn user_agent() -> String {
    format!("anyllm/{} ({}; rust)", VERSION, std::env::consts::OS)
}

/// Join a base URL and path, preserving any query string on the base.
pub fn join_url(base: &str, path: &str) -> String {
    let (scheme, rest) = match base.split_once("://") {
        Some((scheme, rest)) => (scheme.to_string(), rest.to_string()),
        None => (String::new(), base.to_string()),
    };
    let (netloc, remainder) = match rest.find(['/', '?', '#']) {
        Some(i) => (rest[..i].to_string(), &rest[i..]),
        None => (rest.clone(), ""),
    };
    let (base_path, query_and_fragment) = match remainder.find(['?', '#']) {
        Some(i) => (remainder[..i].to_string(), &remainder[i..]),
        None => (remainder.to_string(), ""),
    };
    let (query, fragment) = match query_and_fragment.strip_prefix('?') {
        Some(q) => match q.find('#') {
            Some(i) => (q[..i].to_string(), q[i..].to_string()),
            None => (q.to_string(), String::new()),
        },
        None => (String::new(), query_and_fragment.to_string()),
    };
    let new_path = format!(
        "{}/{}",
        base_path.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    let mut out = String::new();
    if !scheme.is_empty() {
        out.push_str(&scheme);
        out.push_str("://");
    }
    out.push_str(&netloc);
    out.push_str(&new_path);
    if !query.is_empty() {
        out.push('?');
        out.push_str(&query);
    }
    out.push_str(&fragment);
    out
}

fn netloc_of(url: &str) -> String {
    match url.find("://") {
        Some(i) => url[i + 3..].split('/').next().unwrap_or(url).to_string(),
        None => url.to_string(),
    }
}

fn trunc400(text: &str) -> String {
    if text.chars().count() > 400 {
        text.chars().take(400).collect()
    } else {
        text.to_string()
    }
}

fn text_field(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
                    Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("."))
            }
        }
        _ => None,
    }
}

fn format_detail_list(items: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Value::Array(list) = items {
        for item in list {
            match item {
                Value::String(s) if !s.trim().is_empty() => parts.push(s.trim().to_string()),
                Value::Object(map) => {
                    let loc_text = match map.get("loc") {
                        Some(Value::Array(loc)) => loc
                            .iter()
                            .filter_map(|p| match p {
                                Value::String(s) if s != "body" && s != "root" => Some(s.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("."),
                        _ => String::new(),
                    };
                    let msg = map
                        .get("msg")
                        .or_else(|| map.get("message"))
                        .or_else(|| map.get("type"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !loc_text.is_empty() && !msg.is_empty() {
                        parts.push(format!("{}: {}", loc_text, msg));
                    } else if !msg.is_empty() {
                        parts.push(msg);
                    } else if !loc_text.is_empty() {
                        parts.push(loc_text);
                    }
                }
                _ => {}
            }
        }
    }
    parts.join("; ")
}

/// Return `(message, param, code)` from an error response body.
fn parse_error_body(body: &[u8]) -> (String, Option<String>, Option<String>) {
    let text = String::from_utf8_lossy(body).trim().to_string();
    if text.is_empty() {
        return (String::new(), None, None);
    }
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(_) => return (trunc400(&text), None, None),
    };
    let map = match parsed.as_object() {
        Some(map) => map,
        None => return (trunc400(&text), None, None),
    };

    let mut param = text_field(map.get("param").unwrap_or(&Value::Null));
    let mut code = text_field(map.get("code").unwrap_or(&Value::Null));
    let mut message = String::new();

    match map.get("error") {
        Some(Value::Object(detail)) => {
            param = text_field(detail.get("param").unwrap_or(&Value::Null)).or(param);
            code = text_field(detail.get("code").unwrap_or(&Value::Null)).or(code);
            let raw = detail
                .get("message")
                .or_else(|| detail.get("code"))
                .or_else(|| detail.get("type"));
            if let Some(raw) = raw {
                if !raw.is_null() {
                    message = raw
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| raw.to_string());
                }
            }
        }
        Some(Value::String(detail)) if !detail.is_empty() => message = detail.clone(),
        Some(Value::Array(_)) => message = format_detail_list(map.get("error").unwrap()),
        _ => {}
    }

    if message.is_empty() {
        for key in ["message", "detail", "error_description", "msg"] {
            match map.get(key) {
                Some(Value::String(s)) if !s.is_empty() => {
                    message = s.clone();
                    break;
                }
                Some(value @ Value::Array(_)) => {
                    let formatted = format_detail_list(value);
                    if !formatted.is_empty() {
                        message = formatted;
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    if message.is_empty() {
        message = text.clone();
    }
    if let Some(p) = &param {
        if !message.to_lowercase().contains(&p.to_lowercase()) {
            message = if message.is_empty() {
                p.clone()
            } else {
                format!("{} ({})", message, p)
            };
        }
    }
    (trunc400(&message), param, code)
}

/// Turn an HTTP error into a specific, actionable error.
pub fn raise_for_status(
    status: u16,
    body: &[u8],
    retry_after: Option<&str>,
    url: &str,
) -> AnyLLMError {
    let host = netloc_of(url);
    let (message, param, code) = parse_error_body(body);
    let detail = if !message.is_empty() {
        format!("{} (HTTP {})", message, status)
    } else {
        format!("HTTP {} from {}", status, host)
    };
    let lowered = message.to_lowercase();

    match status {
        401 | 403 => {
            let hint = if status == 401 {
                "The API key was rejected. Run `anyllm keys set <provider>` again, \
                 or check the environment variable in `anyllm doctor`."
            } else {
                "The key was accepted but access was denied. Check that the key \
                 may use this model, region, or account."
            };
            AnyLLMError::auth(format!("authentication failed: {}", detail)).with_hint(hint)
        }
        404 => AnyLLMError::not_found(format!("endpoint or model not found: {}", detail))
            .with_hint(
                "Check the model name and --base-url. \
             Run `anyllm models <provider>` to see what the key can access.",
            ),
        429 => {
            let mut hint = "You hit a rate limit or quota. Wait a moment and retry.".to_string();
            if let Some(retry_after) = retry_after {
                hint.push_str(&format!(
                    " The provider asked to wait {} seconds.",
                    retry_after
                ));
            }
            AnyLLMError::rate_limit(format!("rate limited: {}", detail)).with_hint(hint)
        }
        _ => {
            let hint = if status == 400 {
                if [
                    "context length",
                    "maximum context",
                    "too many tokens",
                    "too long",
                ]
                .iter()
                .any(|word| lowered.contains(word))
                {
                    Some(
                        "The conversation may be too long. \
                         Use /clear to start fresh, or shorten the prompt.",
                    )
                } else if lowered.contains("max_tokens")
                    || lowered.contains("max_completion_tokens")
                {
                    Some(
                        "The token limit was rejected. \
                         Try without --max-tokens, or use a smaller value.",
                    )
                } else if lowered.contains("temperature") {
                    Some("This model may not support temperature. Try without --temperature.")
                } else if lowered.contains("model")
                    && (lowered.contains("not exist") || lowered.contains("not found"))
                {
                    Some("Run `anyllm models <provider>` to list available model names.")
                } else {
                    None
                }
            } else if status >= 500 {
                Some("The provider had a server error. Retry shortly.")
            } else {
                None
            };
            let mut err = AnyLLMError::api(
                format!("request failed: {}", detail),
                Some(status),
                param,
                code,
            );
            if let Some(hint) = hint {
                err = err.with_hint(hint);
            }
            err
        }
    }
}

fn retry_delay(retry_after: Option<&str>, attempt: usize) -> Option<f64> {
    if let Some(header) = retry_after {
        if let Ok(seconds) = header.trim().parse::<f64>() {
            if seconds > MAX_RETRY_WAIT {
                return None;
            }
            return Some(seconds.max(0.0));
        }
    }
    Some(0.8 * (2_u32).pow(attempt as u32) as f64)
}

fn should_retry_status(status: u16, replay_safe: bool) -> bool {
    if SAFE_RETRY_STATUS.contains(&status) {
        return true;
    }
    replay_safe && IDEMPOTENT_RETRY_STATUS.contains(&status)
}

/// True when the TCP connection never came up, so nothing was sent.
fn failed_before_send(message: &str) -> bool {
    let lowered = message.to_lowercase();
    lowered.contains("connection refused")
        || lowered.contains("dns")
        || lowered.contains("no such host")
        || lowered.contains("unreachable")
}

fn is_tls_failure(message: &str) -> bool {
    let lowered = message.to_lowercase();
    ["tls", "ssl", "certificate", "invalid peer cert"]
        .iter()
        .any(|marker| lowered.contains(marker))
}

fn network_error(message: &str, url: &str) -> AnyLLMError {
    let host = netloc_of(url);
    let hint = if is_tls_failure(message) {
        "TLS failed. Check your system certificates, proxy, or VPN."
    } else {
        "Check your internet connection, proxy settings, and the base URL."
    };
    AnyLLMError::network(format!("could not reach {}: {}", host, message)).with_hint(hint)
}

/// Open a URL, retrying only failures that are safe to send again.
fn open(
    method: &str,
    url: &str,
    data: Option<&Vec<u8>>,
    headers: &[(String, String)],
    timeout: f64,
    retries: usize,
    replay_safe: bool,
) -> Result<Box<dyn Read + Send + Sync>> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs_f64(timeout.max(0.1)))
        .timeout_read(Duration::from_secs_f64(timeout.max(0.1)))
        .timeout_write(Duration::from_secs_f64(timeout.max(0.1)))
        .build();
    let mut attempt = 0;
    loop {
        let mut request = agent
            .request(method, url)
            .set("User-Agent", &user_agent())
            .set("Content-Type", "application/json");
        for (key, value) in headers {
            request = request.set(key, value);
        }
        let outcome = match &data {
            Some(bytes) => request.send_bytes(bytes),
            None => request.call(),
        };
        match outcome {
            Ok(response) => {
                return Ok(response.into_reader());
            }
            Err(ureq::Error::Status(status, response)) => {
                let retry_after = response.header("Retry-After").map(str::to_string);
                let mut body = Vec::new();
                let mut reader = response.into_reader();
                let _ = reader.read_to_end(&mut body);
                if attempt < retries && should_retry_status(status, replay_safe) {
                    if let Some(delay) = retry_delay(retry_after.as_deref(), attempt) {
                        std::thread::sleep(Duration::from_secs_f64(delay));
                        attempt += 1;
                        continue;
                    }
                }
                return Err(raise_for_status(status, &body, retry_after.as_deref(), url));
            }
            Err(ureq::Error::Transport(transport)) => {
                let text = transport.to_string();
                if attempt < retries
                    && !is_tls_failure(&text)
                    && (replay_safe || failed_before_send(&text))
                {
                    std::thread::sleep(Duration::from_secs_f64(0.6 * (attempt as f64 + 1.0)));
                    attempt += 1;
                    continue;
                }
                return Err(network_error(&text, url));
            }
        }
    }
}

fn serialize_payload(payload: &Value) -> Vec<u8> {
    serde_json::to_vec(payload).unwrap_or_default()
}

/// POST JSON and return an open streaming response.
///
/// The POST is not replayed after the server has seen it. Retries are limited
/// to 408/425/429/503 and to connection failures that happen before send.
pub fn open_stream(
    url: &str,
    headers: &[(String, String)],
    payload: &Value,
    timeout: f64,
    retries: usize,
) -> Result<Box<dyn Read + Send + Sync>> {
    let data = serialize_payload(payload);
    let reader = open("POST", url, Some(&data), headers, timeout, retries, false)?;
    Ok(reader)
}

/// Perform a request and decode a JSON object or array.
pub fn request_json(
    url: &str,
    headers: &[(String, String)],
    payload: Option<&Value>,
    method: &str,
    timeout: f64,
    retries: usize,
) -> Result<Value> {
    let data = payload.map(serialize_payload);
    let replay_safe = matches!(method.to_uppercase().as_str(), "GET" | "HEAD" | "OPTIONS");
    let mut reader = open(
        method,
        url,
        data.as_ref(),
        headers,
        timeout,
        retries,
        replay_safe,
    )?;
    let mut raw = Vec::new();
    reader.read_to_end(&mut raw).map_err(|e| {
        AnyLLMError::network(format!(
            "connection broke while reading from {}: {}",
            netloc_of(url),
            e
        ))
        .with_hint("Retry the request.")
    })?;
    if raw.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(Value::Object(Map::new()));
    }
    match serde_json::from_slice::<Value>(&raw) {
        Ok(value) => Ok(value),
        Err(_) => {
            let snippet: String = String::from_utf8_lossy(&raw)
                .trim()
                .chars()
                .take(300)
                .collect();
            Err(AnyLLMError::api(
                format!("{} returned a non-JSON response.", netloc_of(url)),
                None,
                None,
                None,
            )
            .with_hint(format!("Response started with: {:?}", snippet)))
        }
    }
}

// -- server-sent events --------------------------------------------------------

/// Incremental SSE parser over any byte stream.
///
/// Reads the body in 8 KiB chunks, splits complete events, and stops after a
/// `[DONE]` payload. The deadline/cancel reader thread from the Python
/// version is not ported; a blocking read cannot be interrupted here.
pub struct SseReader<R: Read> {
    inner: R,
    buf: Vec<u8>,
    pending: std::collections::VecDeque<String>,
    done: bool,
}

impl<R: Read> SseReader<R> {
    pub fn new(inner: R) -> Self {
        SseReader {
            inner,
            buf: Vec::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
        }
    }

    /// Extract complete SSE events out of the buffer.
    fn take_events(&mut self, flush: bool) {
        loop {
            let boundary = find_boundary(&self.buf);
            match boundary {
                Some((start, end)) => {
                    let block: Vec<u8> = self.buf.drain(..end).collect();
                    let block = block[..start].to_vec();
                    if let Some(payload) = event_payload(&block) {
                        self.pending.push_back(payload);
                    }
                }
                None => break,
            }
        }
        if flush {
            let block = std::mem::take(&mut self.buf);
            if let Some(payload) = event_payload(&block) {
                self.pending.push_back(payload);
            }
        }
    }

    /// Next data payload, or None at end of stream (or after `[DONE]`).
    pub fn next_event(&mut self) -> std::io::Result<Option<String>> {
        while self.pending.is_empty() && !self.done {
            let mut chunk = [0u8; 8192];
            let read = self.inner.read(&mut chunk)?;
            if read == 0 {
                self.take_events(true);
                self.done = true;
                break;
            }
            self.buf.extend_from_slice(&chunk[..read]);
            self.take_events(false);
        }
        while let Some(payload) = self.pending.pop_front() {
            if payload.trim() == "[DONE]" {
                self.done = true;
                self.pending.clear();
                return Ok(None);
            }
            return Ok(Some(payload));
        }
        Ok(None)
    }
}

/// Find the next SSE blank-line boundary, including mixed LF/CRLF endings.
/// Returns (block_len, next_pos).
fn find_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' {
            if buf[i + 1] == b'\n' {
                return Some((i, i + 2));
            }
            if buf[i + 1] == b'\r' && buf.get(i + 2) == Some(&b'\n') {
                return Some((i, i + 3));
            }
        } else if buf[i] == b'\r' && buf.get(i + 1) == Some(&b'\n') {
            if buf.get(i + 2) == Some(&b'\n') {
                return Some((i, i + 3));
            }
            if buf.get(i + 2) == Some(&b'\r') && buf.get(i + 3) == Some(&b'\n') {
                return Some((i, i + 4));
            }
        }
        i += 1;
    }
    None
}

/// Decode the `data:` payload out of one SSE block.
fn event_payload(block: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(block);
    let mut parts: Vec<String> = Vec::new();
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            parts.push(value.to_string());
        } else if line.starts_with(':') {
            continue;
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("\n"))
}

/// Blocking SSE iteration, mapping read errors to `NetworkError`.
pub fn iter_sse<R: Read>(reader: R) -> SseIterator<R> {
    SseIterator {
        reader: SseReader::new(reader),
        finished: false,
    }
}

/// Iterate SSE events while observing a cancellation flag and absolute deadline.
///
/// The reader runs on a background thread because generic `Read` implementations
/// can block. Stopping this iterator stops local delivery; it cannot cancel the
/// remote request. A read already blocked in the operating system may finish
/// later, as in the Python transport's local cancellation behavior.
pub fn iter_sse_controlled<R>(
    reader: R,
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
) -> ControlledSseIterator
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let reader_stopped = Arc::clone(&stopped);
    thread::spawn(move || {
        let mut parser = SseReader::new(reader);
        loop {
            if reader_stopped.load(Ordering::Relaxed) {
                break;
            }
            let item = match parser.next_event() {
                Ok(Some(payload)) => Some(Ok(payload)),
                Ok(None) => {
                    let _ = sender.send(None);
                    break;
                }
                Err(error) => Some(Err(AnyLLMError::network(format!(
                    "connection broke while streaming: {}",
                    error
                ))
                .with_hint(
                    "The provider may have timed out. Retry, or use --timeout with a larger value.",
                ))),
            };
            if let Some(item) = item {
                let failed = item.is_err();
                if sender.send(Some(item)).is_err() {
                    break;
                }
                if parser.done || failed {
                    break;
                }
            }
        }
    });
    ControlledSseIterator {
        receiver,
        cancelled,
        deadline,
        stopped,
        finished: false,
    }
}

pub struct ControlledSseIterator {
    receiver: mpsc::Receiver<Option<Result<String>>>,
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
    stopped: Arc<AtomicBool>,
    finished: bool,
}

impl Iterator for ControlledSseIterator {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let cancelled = self.cancelled.load(Ordering::Relaxed);
            let deadline_elapsed = self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline);
            if cancelled || deadline_elapsed {
                match self.receiver.try_recv() {
                    Ok(Some(item)) => return Some(item),
                    Ok(None) | Err(mpsc::TryRecvError::Disconnected) => {
                        self.finished = true;
                        return None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                self.finished = true;
                self.stopped.store(true, Ordering::Relaxed);
                let message = if cancelled {
                    "Streaming stopped locally after a cancel. The provider request was not cancelled."
                } else {
                    "The overall deadline elapsed while streaming. The provider request was not cancelled."
                };
                return Some(Err(local_stop_error(message)));
            }
            let wait = self
                .deadline
                .map(|deadline| {
                    deadline
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(50))
                })
                .unwrap_or(Duration::from_millis(50));
            match self.receiver.recv_timeout(wait) {
                Ok(Some(item)) => return Some(item),
                Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.finished = true;
                    return None;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

impl Drop for ControlledSseIterator {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
    }
}

fn local_stop_error(message: &str) -> AnyLLMError {
    AnyLLMError::network(message).with_hint(
        "The provider may still be generating and may still charge for this request. Stopping the local read does not cancel it on the server.",
    )
}

pub struct SseIterator<R: Read> {
    reader: SseReader<R>,
    finished: bool,
}

impl<R: Read> Iterator for SseIterator<R> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.reader.next_event() {
            Ok(Some(payload)) => Some(Ok(payload)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(e) => {
                self.finished = true;
                Some(Err(AnyLLMError::network(format!(
                    "connection broke while streaming: {}",
                    e
                ))
                .with_hint(
                    "The provider may have timed out. Retry, or use --timeout \
                     with a larger value.",
                )))
            }
        }
    }
}

/// Read a whole response body as bytes (used for error status bodies).
pub fn drain(reader: Box<dyn Read + Send + Sync>) -> Vec<u8> {
    let mut out = Vec::new();
    let _ = reader.take(100_000).read_to_end(&mut out);
    out
}
