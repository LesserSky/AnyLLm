//! User-facing error types.
//!
//! Every error carries a short message plus an optional hint that tells the
//! user what to do next. The CLI prints both, so errors stay readable.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl ErrorKind {
    /// Errors printed in yellow instead of red (see Python `_WARN_ERRORS`).
    pub fn is_warn(&self) -> bool {
        matches!(
            self,
            ErrorKind::RateLimit | ErrorKind::Network | ErrorKind::Usage
        )
    }
}

/// Port of Python `AnyLLMError`: shown without a traceback.
/// `status`/`param`/`code` carry the extra data of Python `APIError`.
#[derive(Debug, Clone)]
pub struct AnyLLMError {
    pub kind: ErrorKind,
    pub message: String,
    pub hint: Option<String>,
    pub status: Option<u16>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl AnyLLMError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        AnyLLMError {
            kind,
            message: message.into(),
            hint: None,
            status: None,
            param: None,
            code: None,
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Attach a hint only when `hint` is `Some`.
    pub fn with_hint_opt(mut self, hint: Option<String>) -> Self {
        self.hint = hint;
        self
    }

    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Usage, message)
    }
    pub fn config(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Config, message)
    }
    pub fn auth(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Auth, message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::RateLimit, message)
    }
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Network, message)
    }
    pub fn provider(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Provider, message)
    }

    /// Python `APIError`: keeps HTTP status, param, and code.
    pub fn api(
        message: impl Into<String>,
        status: Option<u16>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        let mut err = Self::new(ErrorKind::Api, message);
        err.status = status;
        err.param = param;
        err.code = code;
        err
    }
}

impl fmt::Display for AnyLLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AnyLLMError {}

impl From<std::io::Error> for AnyLLMError {
    fn from(e: std::io::Error) -> Self {
        AnyLLMError::config(format!("{}", e))
    }
}

pub type Result<T> = std::result::Result<T, AnyLLMError>;
