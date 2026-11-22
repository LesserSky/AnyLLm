# AnyLLM Rust port contracts

Phase 1 follows the Python implementation in `anyllm/anyllm/` for commands,
terminal output, exit codes, network behavior, and file formats.

## Crates and ownership

| Package | Rust name | Responsibility |
| --- | --- | --- |
| `anyllm-core` | `anyllm_core` | Provider catalog, config, keystore, sessions, terminal primitives, HTTP/SSE, provider adapters |
| `anyllm-orch` | `anyllm_orch` | Targets and credential policy, budgets and prices, request snapshots, attachments, team runs, tool protocol, REPL feature registry |
| `anyllm-cli` | `anyllm_cli` | Argument parsing and dispatch, interaction and rendering, concrete REPL, core/orchestration bridge |

The binary is `anyllm` (`anyllm.exe` on Windows). From the repository root,
run `./rust/target/release/anyllm` after a release build in `rust/`.

`cli` depends on `core` and `orch`. Both libraries remain independently
testable and have no dependency on each other or on `cli`. `orch` receives its
network and key-store implementations through `orch::sys` traits. Reuse the
workspace's existing dependencies and ask the top integrator before adding
others. `rustyline` is approved for the CLI, and `crossterm` is approved only if
Windows console handling requires it. Crate owners maintain their manifests.
The top integrator owns the workspace manifest and final lockfile resolution.

## Public surfaces used by the CLI

- Core data uses `Config`, `Provider`, `ProviderSpecArgs`, `KeyStore`, and
  `KeyResolution`, with the config, catalog, and keystore functions exposed
  by `anyllm_core`. Use `load_providers` to include named custom providers.
- Core chat uses `Session`, `SessionObserver`, and `Turn` for ordinary
  conversation, `stream_chat` with `StreamArgs` for direct streaming, and
  `list_models` for discovery. `ChunkStream::next_chunk` yields `Chunk` values
  with text, reasoning, `Usage`, finish reason, and `ToolCallDelta`.
- Session save and load use core's `save_session`, `load_session_data`, and
  `latest_session_name`. Core `Style`, `format_error`, `format_table`,
  `format_reply_status`, `startup_banner`, and `redact_endpoint` provide shared
  rendering primitives. User-facing errors use `AnyLLMError` and `ErrorKind`.
- Orchestration uses `targets::{Target, TargetSpec, ResolvedTarget}` and the
  target resolution functions. Use `budgets::{Budget, Price, PlannedCall,
  Preflight, preflight, receipt}` and `decimal::Decimal` for budgets and costs.
- Use `seam::{RequestSnapshot, RunOptions, RunCall, RunResult, RunEvent,
  Presentation, run_request}` for isolated runs. The `team` and `tool_protocol`
  modules expose the Python entry points `execute_team_ask`, `probe_tools`,
  `run_tool_conversation`, and their respective `register` functions. The
  orchestration lead coordinates their signatures with the interface lead.

## Bridge owned by the CLI

- Convert the core provider map into `orch::sys::ProviderMap` field for field.
  Preserve custom providers, auth headers and prefixes, URLs, key requirements,
  model defaults, and environment variable order. Keep the catalog's default
  URL separate from the resolved request URL.
- Wrap or borrow the real core `KeyStore` to implement `orch::sys::KeyStore`.
  Map `get` errors into `sys::Error`. Use `orch::targets` for named targets,
  endpoint-specific credentials, and live rebinding. Pass only the credential
  selected for the resolved endpoint to the transport. Preserve explicit
  `--base-url` behavior and its warning where Python applies it.
- Implement `orch::sys::ChatBackend::stream(&ChatRequest) -> sys::Result<Vec<Chunk>>`
  over core streaming. Pass the resolved model and URL, selected credential,
  system prompt, messages, temperature, token limit, timeout, and tool payload.
  Forward the cancellation flag and absolute deadline to core and check them
  before sending and during consumption. Preserve the chunk order and every
  text, reasoning, optional usage counter, finish reason, and tool delta field.
- Core keeps `stream_chat`, `StreamArgs`, and `Message::new` usable as they
  stand. The extended adapter boundary is
  `adapters::stream_chat_with_options(&Provider, &StreamArgs, &StreamOptions)`.
  `StreamOptions<'a>` defaults to no overrides and provides `messages:
  Option<&'a [serde_json::Value]>`, `tools: Option<&'a serde_json::Value>`,
  `deadline: Option<std::time::Instant>`, and `cancel: Option<&'a
  std::sync::atomic::AtomicBool>`. The JSON message override preserves OpenAI
  `tool_calls` and `tool_call_id`, with `StreamArgs::system` still sent by the
  adapter. Keep the normal provider-specific mapping for plain messages.
  Only OpenAI-kind endpoints support the tool conversation in Python.
- Convert `core::AnyLLMError` and `sys::Error` in both directions, mapping kind,
  message, hint, status, param, and code. Follow Python's rendering and exit
  codes for usage errors (2), other application errors (1), interrupts (130),
  and successful commands (0). Credentials stay out of diagnostics and saved
  or public output.
- Implement `seam::Repl` over the CLI REPL, including `say`, provider map,
  store, interactive status, and `sys::Style` with the same `enabled` flag.
  Use `ReplRegistry` and the feature registrations for slash commands and help.
  Expose the CLI backend to orchestration handlers through the REPL seam when
  needed, with the orchestration lead coordinating any additive seam methods.
- Keep one authoritative live session. When bridging to `sys::Session`,
  synchronize provider, model, effective endpoint, credential and source,
  system, generation settings, timeout, reasoning flag, target alias, history,
  usage, and turn count. Apply mutations such as `/target` rebind and `/team
  adopt` back to the shared live state. Track `explicit_endpoint` in memory
  for the current invocation and use the core session schema for saved chats.

Keep Python launchers and existing JSON shapes and permissions through Phase 1.
