//! `anyllm-core`: the foundation + wire layer for the anyllm CLI.
//!
//! Ports of the Python package's core modules:
//! `errors`, `config`, `catalog`, `keystore`, `term`, `transport`,
//! `adapters`, and `session`. Behavior, error text, and the JSON file
//! formats (`config.json`, `keys.json`, `sessions/*.json`) match the
//! original Python implementation.

pub mod adapters;
pub mod catalog;
pub mod config;
pub mod errors;
pub mod keystore;
pub mod session;
pub mod term;
pub mod transport;

pub const VERSION: &str = "1.0.0";

pub use adapters::{
    list_models, stream_chat, Chunk, ChunkStream, Message, StreamArgs, ToolCallDelta, Usage,
};
pub use catalog::{
    builtins, get_provider, load_providers, normalize_name, provider_from_spec, validate_base_url,
    validate_custom_name, Provider, ProviderSpecArgs, ANTHROPIC, GEMINI, KINDS, OPENAI,
};
pub use config::{
    config_dir, config_path, keys_path, load_config, parse_config_value, read_json, sessions_dir,
    validate_max_tokens, validate_temperature, write_json, Config, ALLOWED_KEYS,
};
pub use errors::{AnyLLMError, ErrorKind, Result};
pub use keystore::{
    accepts_generic_api_key, dpapi_available, mask_key, resolve_key, KeyResolution, KeyStore,
};
pub use session::{
    default_session_name, finish_notice, interrupted_content, latest_session_name, list_sessions,
    load_session_data, sanitize_session_name, save_session, session_base_url, session_path,
    NoObserver, Session, SessionObserver, Turn, INTERRUPTED_MARK,
};
pub use term::{
    color_enabled, format_error, format_reply_status, format_table, is_tty, redact_endpoint, setup,
    startup_banner, Style,
};
pub use transport::{
    join_url, open_stream, raise_for_status, request_json, user_agent, SseReader, MAX_RETRIES,
};
