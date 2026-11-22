# Rust port contracts

Python under `anyllm/anyllm` defines behavior, formats, and user-facing text.
Phase 1 preserves the Python launchers and builds the binary named `anyllm`.

## Crate boundaries

- `anyllm-core` owns provider catalog, configuration, keys, sessions, terminal
  primitives, HTTP transport, provider adapters, and their public errors.
- `anyllm-orch` owns targets, prices and budgets, request snapshots, team and
  tool orchestration. Its `sys` module defines its independent integration
  types and traits; it does not depend on core or CLI.
- `anyllm-cli` owns argument dispatch, interaction, rendering, and the bridge
  between core and orchestration. It may depend on both crates; neither lower
  crate may depend on CLI. Avoid cyclic dependencies.
- The workspace manifest and lockfile are integrated by the top agent. Crate
  owners manage their own manifests. New dependencies require top approval;
  `rustyline` is pre-approved, and `crossterm` only if Windows requires it.

## CLI-facing types

- Core exports `Provider`, `Config`, `KeyStore`, `KeyResolution`, `Session`,
  `Turn`, `SessionObserver`, `Message`, `Usage`, `Chunk`, `ToolCallDelta`,
  `StreamArgs`, `ChunkStream`, `AnyLLMError`, `ErrorKind`, and `Style`, plus
  catalog, storage, rendering, and transport functions from `core::lib`.
- Orchestration exposes `sys::{Provider, ProviderMap, Session, Message, Usage,
  Chunk, ToolCallDelta, Error, ErrorKind, ChatRequest, ChatBackend, KeyStore,
  Style}`, `targets::{Target, TargetSpec, ResolvedTarget}`, `budgets::{Price,
  Budget, PlannedCall, Preflight}`, and `seam::{RequestSnapshot, RunOptions,
  RunResult, RunEvent, Presentation, Repl, ReplRegistry}`. Team and tool
  protocol APIs are owned by orchestration and exposed by their modules.

## Bridge

CLI implements `orch::sys::ChatBackend` using core's `stream_chat` and
`ChunkStream`, and implements `orch::sys::KeyStore` as a wrapper around the
core keystore. Convert provider fields, messages, usage, chunks, tool deltas,
errors (including kind, hint, status, param, code), styles, and session state
explicitly at the boundary; preserve credentials, target aliases and history.
Convert `ChatRequest` model, endpoint, system, options, timeout, and credential
into core request arguments; propagate streaming errors. `ChatBackend::stream`
returns collected chunks, and its deadline and cancellation must be honored.
The current core `Message` and `StreamArgs` lack tool-message fields and tool
payloads, deadline and cancellation; coordinate additions with the core lead
rather than silently discarding them. Keep orchestration's extra session fields
(such as `explicit_endpoint`) intact when synchronizing state.
