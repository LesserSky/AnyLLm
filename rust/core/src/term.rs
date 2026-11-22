//! Terminal theme: ANSI color, plain fallback, endpoint redaction, tables.
//!
//! Port of Python `anyllm/term.py`.

use std::io::IsTerminal;
use std::sync::OnceLock;

use crate::adapters::Usage;
use crate::errors::{AnyLLMError, ErrorKind};

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const ITALIC: &str = "\x1b[3m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const CYAN: &str = "\x1b[36m";
pub const GREY: &str = "\x1b[90m";

pub const ACCENT: &str = "\x1b[36m";
pub const ACCENT_BOLD: &str = "\x1b[1;36m";
pub const ERROR: &str = "\x1b[1;31m";
pub const HINT: &str = "\x1b[2;33m";
pub const REASONING: &str = "\x1b[2;3m";

#[cfg(windows)]
const ENABLE_VT: u32 = 0x0004;

/// Query keys whose values are credentials. `api-version` is not secret.
const SECRET_QUERY: [&str; 12] = [
    "key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "secret",
    "password",
    "auth",
    "authorization",
    "sig",
    "signature",
    "client_secret",
];

const STATUS_ROLES_WARN: [&str; 2] = ["missing", "no key"];
const STATUS_ROLES_OK: [&str; 1] = ["saved"];
const STATUS_ROLES_MUTED: [&str; 3] = ["not needed", "no key needed", "not required"];

// -- Windows VT support -------------------------------------------------------

static VT_READY: OnceLock<bool> = OnceLock::new();

#[cfg(windows)]
fn enable_windows_vt() -> bool {
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, SetConsoleMode, CONSOLE_MODE, STD_ERROR_HANDLE,
        STD_OUTPUT_HANDLE,
    };
    // Styling is cosmetic; never fail because a console is unusual.
    let mut supported = false;
    unsafe {
        for handle_id in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            let handle = GetStdHandle(handle_id as u32);
            if handle.is_null() {
                continue;
            }
            let mut mode: CONSOLE_MODE = 0;
            if GetConsoleMode(handle, &mut mode) == 0 {
                continue;
            }
            if SetConsoleMode(handle, mode | ENABLE_VT) == 0 {
                continue;
            }
            let mut check: CONSOLE_MODE = 0;
            if GetConsoleMode(handle, &mut check) != 0 && (check & ENABLE_VT) != 0 {
                supported = true;
            }
        }
    }
    supported
}

#[cfg(not(windows))]
fn enable_windows_vt() -> bool {
    true
}

fn windows_vt_supported() -> bool {
    *VT_READY.get_or_init(enable_windows_vt)
}

/// Prepare the terminal once at startup (Python `term.setup`).
pub fn setup() {
    let _ = windows_vt_supported();
}

pub fn is_tty() -> bool {
    std::io::stdout().is_terminal()
}

pub fn stderr_is_tty() -> bool {
    std::io::stderr().is_terminal()
}

fn force_color() -> bool {
    match std::env::var("FORCE_COLOR") {
        Ok(value) => !matches!(
            value.trim().to_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// Decide whether ANSI styling should be emitted.
///
/// `NO_COLOR` wins over `FORCE_COLOR`; a dumb `TERM`, a non-TTY stream, or a
/// Windows console without VT processing disable color.
pub fn color_enabled(no_color: bool, tty: bool) -> bool {
    if no_color {
        return false;
    }
    if std::env::var("NO_COLOR")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return false;
    }
    if force_color() {
        return true;
    }
    if std::env::var("TERM")
        .map(|t| t.trim().to_lowercase() == "dumb")
        .unwrap_or(false)
    {
        return false;
    }
    if cfg!(windows) && !windows_vt_supported() {
        return false;
    }
    tty
}

/// Apply ANSI styles only when enabled.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub enabled: bool,
}

impl Style {
    pub fn new(enabled: bool) -> Self {
        Style { enabled }
    }

    pub fn plain() -> Self {
        Style { enabled: false }
    }

    fn wrap(&self, text: &str, code: &str) -> String {
        if !self.enabled || text.is_empty() {
            return text.to_string();
        }
        format!("{}{}{}", code, text, RESET)
    }

    pub fn bold(&self, text: &str) -> String {
        self.wrap(text, BOLD)
    }
    pub fn dim(&self, text: &str) -> String {
        self.wrap(text, DIM)
    }
    pub fn red(&self, text: &str) -> String {
        self.wrap(text, RED)
    }
    pub fn green(&self, text: &str) -> String {
        self.wrap(text, GREEN)
    }
    pub fn yellow(&self, text: &str) -> String {
        self.wrap(text, YELLOW)
    }
    pub fn grey(&self, text: &str) -> String {
        self.wrap(text, GREY)
    }
    pub fn accent(&self, text: &str) -> String {
        self.wrap(text, ACCENT)
    }
    pub fn accent_bold(&self, text: &str) -> String {
        self.wrap(text, ACCENT_BOLD)
    }
    pub fn muted(&self, text: &str) -> String {
        self.dim(text)
    }
    pub fn hint(&self, text: &str) -> String {
        self.wrap(text, HINT)
    }
    pub fn reasoning(&self, text: &str) -> String {
        self.wrap(text, REASONING)
    }
    pub fn code(&self, text: &str) -> String {
        self.grey(text)
    }
    pub fn error(&self, text: &str) -> String {
        self.wrap(text, ERROR)
    }
    pub fn ok(&self, text: &str) -> String {
        self.green(text)
    }
    pub fn warn(&self, text: &str) -> String {
        self.yellow(text)
    }
    pub fn prompt(&self, text: &str) -> String {
        self.accent_bold(text)
    }
}

/// Show an endpoint without userinfo, key query values, or a known secret.
///
/// `api-version` and other non-secret query parameters are preserved.
pub fn redact_endpoint(url: &str, secrets: &[&str]) -> String {
    if url.trim().is_empty() {
        return "(not set)".to_string();
    }
    let raw = url.trim().to_string();
    let mut changed = false;

    let (prefix, rest) = match raw.find("://") {
        Some(i) => (raw[..i + 3].to_string(), raw[i + 3..].to_string()),
        None => (String::new(), raw.clone()),
    };
    let (netloc, tail) = {
        match rest.find(['/', '?', '#']) {
            Some(i) => (rest[..i].to_string(), rest[i..].to_string()),
            None => (rest.clone(), String::new()),
        }
    };
    let mut netloc_out = netloc.clone();
    if let Some(at) = netloc.rfind('@') {
        let hostport = &netloc[at + 1..];
        let (host, port) = match hostport.rfind(':') {
            // Do not treat the colon inside an IPv6 literal as a port.
            Some(i) if !hostport[..i].contains(']') => (&hostport[..i], Some(&hostport[i..])),
            _ => (hostport, None),
        };
        let host = if host.contains(':') && !host.starts_with('[') {
            format!("[{}]", host)
        } else {
            host.to_string()
        };
        netloc_out = match port {
            Some(p) => format!("{}{}", host, p),
            None => host,
        };
        changed = true;
    }
    let mut tail_out = tail.clone();
    if let Some(query_start) = tail.find('?') {
        let before = tail[..=query_start].to_string();
        let query = &tail[query_start + 1..];
        let after = match query.find('#') {
            Some(i) => query[i..].to_string(),
            None => String::new(),
        };
        let query = match query.find('#') {
            Some(i) => &query[..i],
            None => query,
        };
        let mut rebuilt: Vec<String> = Vec::new();
        for pair in query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (key, _value) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => (pair, ""),
            };
            if sensitive_query_key(key) {
                rebuilt.push(format!("{}=***", key));
                changed = true;
            } else {
                rebuilt.push(pair.to_string());
            }
        }
        if changed {
            tail_out = format!("{}{}{}", before, rebuilt.join("&"), after);
        }
    }

    let mut out = if changed {
        format!("{}{}{}", prefix, netloc_out, tail_out)
    } else {
        raw
    };
    for secret in secrets {
        if secret.len() < 8 {
            continue;
        }
        if out.contains(secret) {
            out = out.replace(secret, "***");
        }
    }
    out
}

fn sensitive_query_key(key: &str) -> bool {
    let name = key.trim().to_lowercase().replace('-', "_");
    if SECRET_QUERY.contains(&name.as_str()) {
        return true;
    }
    name.ends_with("_key")
        || name.ends_with("_token")
        || name.ends_with("_secret")
        || name.ends_with("_password")
}

/// Concise startup block: product, provider, model, and endpoint.
pub fn startup_banner(
    style: &Style,
    version: &str,
    provider: &str,
    model: &str,
    endpoint: &str,
) -> Vec<String> {
    let endpoint = if endpoint.is_empty() {
        "(not set)"
    } else {
        endpoint
    };
    if !style.enabled {
        return vec![
            String::new(),
            format!("anyllm {}", version),
            format!("{}/{}", provider, model),
            endpoint.to_string(),
            "Type /help for commands, /exit to quit.".to_string(),
            String::new(),
        ];
    }
    let bar = "─".repeat(32);
    vec![
        String::new(),
        style.accent(&bar),
        format!(
            "{}{}",
            style.accent_bold("anyllm"),
            style.muted(&format!("  {}", version))
        ),
        format!(
            "{}{}{}",
            style.bold(provider),
            style.muted("  ·  "),
            style.accent(model)
        ),
        style.muted(endpoint),
        style.muted("Type /help for commands, /exit to quit."),
        style.accent(&bar),
        String::new(),
    ]
}

/// One compact status line. Token counts appear only when usage exists.
pub fn format_reply_status(style: &Style, elapsed: f64, usage: Option<&Usage>) -> String {
    let separator = if style.enabled { " · " } else { " | " };
    let mut pieces = vec![format!("{:.1}s", elapsed)];
    if let Some(usage) = usage {
        if usage_present(usage) {
            if let Some(input) = usage.input_tokens {
                pieces.push(format!("{} in", input));
            }
            if let Some(output) = usage.output_tokens {
                pieces.push(format!("{} out", output));
            }
            if let Some(reasoning) = usage.reasoning_tokens {
                if reasoning > 0 {
                    pieces.push(format!("{} reasoning", reasoning));
                }
            }
            if let Some(cached) = usage.cached_tokens {
                if cached > 0 {
                    pieces.push(format!("{} cached", cached));
                }
            }
        }
    }
    style.muted(&format!("  {}", pieces.join(separator)))
}

fn usage_present(usage: &Usage) -> bool {
    usage.input_tokens.is_some()
        || usage.output_tokens.is_some()
        || usage.reasoning_tokens.is_some()
        || usage.cached_tokens.is_some()
}

/// Color an error without changing its message or hint text.
pub fn format_error(style: &Style, exc: &AnyLLMError) -> Vec<String> {
    let painter = if exc.kind.is_warn() {
        Style::warn
    } else {
        Style::error
    };
    let mut lines = vec![painter(style, &format!("error: {}", exc.message))];
    if let Some(hint) = &exc.hint {
        lines.push(style.hint(&format!("  {}", hint)));
    }
    lines
}

/// Aligned text table. A dim rule and accent names appear only in color.
pub fn format_table(style: &Style, headers: &[&str], rows: &[Vec<String>]) -> Vec<String> {
    if headers.is_empty() {
        return Vec::new();
    }
    let width_count = headers.len();
    let mut normalized: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut cells: Vec<String> = row.clone();
        cells.truncate(width_count);
        while cells.len() < width_count {
            cells.push(String::new());
        }
        normalized.push(cells);
    }
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for cells in &normalized {
        for (index, cell) in cells.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let paint = |index: usize, raw: &str, shown: &str| -> String {
        if !style.enabled {
            return shown.to_string();
        }
        if index == 0 {
            return style.accent(shown);
        }
        if STATUS_ROLES_WARN.contains(&raw) {
            return style.warn(shown);
        }
        if STATUS_ROLES_OK.contains(&raw) {
            return style.ok(shown);
        }
        if STATUS_ROLES_MUTED.contains(&raw) {
            return style.muted(shown);
        }
        shown.to_string()
    };

    let header_shown: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| pad(h, widths[i]))
        .collect();
    let mut lines = vec![style.bold(header_shown.join("  ").trim_end())];
    if style.enabled && !normalized.is_empty() {
        let rule: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
        lines.push(style.muted(&rule.join("  ")));
    }
    for cells in &normalized {
        let last = cells.len() - 1;
        let mut painted: Vec<String> = Vec::with_capacity(cells.len());
        for (index, cell) in cells.iter().enumerate() {
            let shown = if index == last {
                cell.clone()
            } else {
                pad(cell, widths[index])
            };
            painted.push(paint(index, cell, &shown));
        }
        lines.push(painted.join("  ").trim_end().to_string());
    }
    lines
}

fn pad(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        text.to_string()
    } else {
        format!("{}{}", text, " ".repeat(width - len))
    }
}

/// Suppress an unused-import warning for ErrorKind in non-test builds.
#[allow(dead_code)]
fn _kind_marker(_: ErrorKind) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_strips_userinfo_and_secret_queries() {
        let url = "https://user:pass@example.com/v1?api_key=abc12345&api-version=2024";
        let out = redact_endpoint(url, &[]);
        assert!(out.contains("api-version=2024"));
        assert!(out.contains("api_key=***"));
        assert!(!out.contains("user:pass"));
        assert!(out.starts_with("https://example.com"));
    }

    #[test]
    fn redact_replaces_known_secrets() {
        let url = "https://api.example.com/v1/sk-abcdef123456";
        let out = redact_endpoint(url, &["sk-abcdef123456"]);
        assert!(!out.contains("sk-abcdef123456"));
        assert!(out.contains("***"));
    }

    #[test]
    fn redact_empty_is_placeholder() {
        assert_eq!(redact_endpoint("", &[]), "(not set)");
        assert_eq!(redact_endpoint("   ", &[]), "(not set)");
    }

    #[test]
    fn banner_plain_mode_is_ascii() {
        let style = Style::plain();
        let lines = startup_banner(&style, "1.0.0", "openai", "gpt-4o-mini", "https://x/v1");
        assert_eq!(lines[1], "anyllm 1.0.0");
        assert_eq!(lines[2], "openai/gpt-4o-mini");
        assert_eq!(lines[3], "https://x/v1");
    }

    #[test]
    fn table_pads_columns() {
        let style = Style::plain();
        let lines = format_table(
            &style,
            &["NAME", "KIND"],
            &[
                vec!["openai".into(), "openai".into()],
                vec!["x".into(), "openai".into()],
            ],
        );
        assert_eq!(lines[0], "NAME    KIND");
        assert_eq!(lines[1], "openai  openai");
        assert_eq!(lines[2], "x       openai");
    }

    #[test]
    fn status_line_plain_and_colored() {
        let usage = Usage {
            input_tokens: Some(20),
            output_tokens: Some(14),
            reasoning_tokens: Some(6),
            cached_tokens: None,
        };
        let plain = format_reply_status(&Style::plain(), 1.1, Some(&usage));
        assert_eq!(plain, "  1.1s | 20 in | 14 out | 6 reasoning");
    }
}
