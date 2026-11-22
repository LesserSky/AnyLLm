# AnyLLM Rust port: 50-agent execution plan

Phase 1 only: finish the Rust port at behavioral parity with the Python CLI.
Do not start the later structure refactor or the visual redesign.

## Current state

- Python is the spec: `anyllm/anyllm/*.py`, 143 tests, all green.
- Rust workspace: `rust/`.
  - `rust/core`: errors, config, catalog, keystore, session, term, transport,
    adapters. 24 tests.
  - `rust/orch`: targets, budgets, decimal, seam, sys. 39 tests.
  - `rust/cli`: compiling skeleton, no real commands yet.
- `cargo test --workspace` is green (63 tests) at the start of this plan.
- Missing: the real CLI, REPL, team, tool protocol, parity tests, CI,
  packaging, docs.
- The Python launchers stay in place until the top agent proves parity. Do not
  switch them.

## Toolchain

Every fresh sandbox starts without Rust. Install it once per sandbox before
any cargo command:

```sh
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
```

The top, each lead, and each manager must do this in its own sandbox. Workers
share their manager's sandbox and use the toolchain installed there.

## Mission

Port every remaining Python behavior to Rust, with the same user-visible text,
JSON shapes, file formats, exit codes, and edge cases. Improve robustness only
where the Python behavior is clearly a bug; record each such change in the
final report. No new features.

## Hard rules

1. Read the Python module before porting it. The Python source wins over this
   plan when they disagree.
2. Match exact strings, spacing, ANSI behavior, JSON keys, exit codes, and file
   permissions. Plain output must stay parseable.
3. Never edit Python source or tests in Phase 1.
4. No new crate dependencies without the top agent's approval. Proposed and
   pre-approved for the CLI: `rustyline` for line editing and history.
   `crossterm` only if needed for Windows console handling. Nothing else.
5. Never commit secrets, keys, sessions, `.env`, or generated `target/`.
6. `cargo fmt` before every commit. Run `cargo test -p <your crate>` before
   reporting. Use `flock /tmp/anyllm-build.lock <command>` in any shared
   sandbox so only one heavy build runs at a time.
7. Commit often. Push your branch. Never force-push, reset, or delete branches
   you did not create.
8. Stay inside your assigned paths. If a change needs another crate, ask the
   parent instead of editing it.

## Ownership

| Area | Exclusive paths |
| --- | --- |
| Top | `rust/Cargo.toml`, `rust/Cargo.lock`, `rust/PORT_PLAN.md`, final integration |
| L1-A core | `rust/core/**` |
| L1-B interface | `rust/cli/**` |
| L1-C orchestration/quality | `rust/orch/**`, `rust/tests/**`, `.github/workflows/**`, `rust/README.md` |
| A1 adapters | `rust/core/src/adapters.rs`, `rust/core/tests/adapters.rs` |
| A2 transport | `rust/core/src/transport.rs`, `rust/core/tests/transport.rs` |
| A3 data | `rust/core/src/{config,keystore,session,catalog,term,errors}.rs`, `rust/core/tests/data.rs` |
| B1 CLI | `rust/cli/src/{main,args,commands}.rs` |
| B2 REPL | `rust/cli/src/repl.rs` |
| B3 render | `rust/cli/src/render.rs` |
| C1 team | `rust/orch/src/team.rs`, `rust/orch/tests/team.rs` |
| C2 tools | `rust/orch/src/tool_protocol.rs`, `rust/orch/tests/tools.rs` |
| C3 quality | `rust/tests/**`, `.github/workflows/**`, `rust/README.md` |

`Cargo.toml` files inside a crate may be edited by that crate's owner. The top
owns the workspace manifest. If you need a dependency, ask your parent; the top
approves and updates the workspace.

## Hierarchy

50 agents, 13 sandboxes:

```
top (1)
├── L1-A core (1)
│   ├── A1 adapters (1 + 5 workers)
│   ├── A2 transport (1 + 4 workers)
│   └── A3 data (1 + 4 workers)
├── L1-B interface (1)
│   ├── B1 CLI (1 + 5 workers)
│   ├── B2 REPL (1 + 4 workers)
│   └── B3 render (1 + 4 workers)
└── L1-C orchestration/quality (1)
    ├── C1 team (1 + 4 workers)
    ├── C2 tools (1 + 4 workers)
    └── C3 quality (1 + 3 workers)
```

Count: 1 + 3 + 9 + 37 = 50. The top, 3 leads, and 9 managers each own a
sandbox. Workers share their manager's sandbox with `--shared-sandbox`.

Models:

- Top and 3 leads: `codex` / `gpt-6.1-terra`, reasoning `medium`.
- 9 managers: `codex` / `gpt-6-luna`, reasoning `medium`.
- 37 workers: `opencode` / `opencode/deepseek-v4.1-flash-free`, reasoning
  `medium`.

If Codex returns a plan-limit error, stop the tree and report to the parent.
Do not retry, and do not switch models without the parent's approval.

Spawn examples (flags in any order; do not add unsupported flags):

```sh
room-cli thread spawn --provider codex --model gpt-6.1-terra \
  --reasoning-level medium --title "Rust L1-A core" \
  --prompt-file rust/agents/l1-core.md

room-cli thread spawn --provider codex --model gpt-6-luna \
  --reasoning-level medium --title "Rust A1 adapters" \
  --prompt-file /tmp/a1.md

room-cli thread spawn --provider opencode \
  --model opencode/deepseek-v4.1-flash-free --reasoning-level medium \
  --shared-sandbox --title "Rust A1-w1" --prompt-file /tmp/a1-w1.md
```

Do not use `--parent-self`, `--project`, or `--top-level`.

## Work packages

### A1 adapters

Port `anyllm/adapters.py` against the existing `rust/core/src/adapters.rs`.
Cover OpenAI, Anthropic, Gemini, and custom OpenAI-compatible providers:
request payloads, streaming deltas, reasoning text, usage, tool-call deltas,
finish reasons, model listing, and error mapping. Add tests with the local mock
server from C3 or a small `TcpListener` fixture.

### A2 transport

Port `anyllm/transport.py`: URL joining, SSE framing across arbitrary chunk
boundaries, retries (`MAX_RETRIES`, backoff cap), timeouts, cancellation,
proxy environment variables, HTTP status mapping, JSON requests, user agent.
Add deadline, retry, malformed-SSE, disconnect, and HTTP error tests.

### A3 data

Port `anyllm/config.py`, `keystore.py`, `session.py`, `catalog.py`, `term.py`,
and `errors.py`. Verify env-var precedence, DPAPI on Windows, `0600`/`0700`
modes, redaction, table layout, color/plain detection, session naming and
latest-session selection, catalog aliases, and exact error text. Add parity
tests.

### B1 CLI

Port the argparse surface in `anyllm/cli.py` to `rust/cli`. Commands: `chat`,
`ask`, `models`, `providers`, `keys`, `config`, `doctor`, plus the orchestration
subcommands (`targets`, `prices`, `team`, `tools`, `capabilities`). Match flags,
defaults, aliases, stdin pipes, `--json`, exit codes, and error formatting.
Build `anyllm` as a single binary. The first milestone is:
`anyllm --version`, `anyllm providers list`, `anyllm config show`,
`anyllm ask --json` against the mock server.

### B2 REPL

Port `anyllm/repl.py`: prompt, slash commands, history file, backslash
continuation, `//text` escape, Ctrl+C streaming interruption with the
`[interrupted]` marker, Ctrl+D exit, save/load, model/provider switching, and
the streaming printer. Use `rustyline` (pre-approved). Keep the exact command
names and help text.

### B3 render

Port the UI behavior of `anyllm/term.py` and the display parts of `repl.py`:
startup banner, spinner until first token, status line, reasoning block,
tables, errors and hints, ANSI/plain switching, `NO_COLOR`/`FORCE_COLOR`,
Windows console fallback. Phase 1 is parity only. Do not redesign the look.

### C1 team

Port `anyllm/team.py`: members file, runs directory, pool/fan-out, budgets and
price gates, synthesis, stacked/columns rendering, save/load/adopt, `/team`
commands, and cancellation. Add tests for fan-out, budget gates, and artifacts.

### C2 tools

Port `anyllm/tool_protocol.py`: tool payloads, capability records, delta
assembly, read-attached-file limits, tool conversation loop, probe, `/tools`
commands, and the local read/edit/undo path. Add tests for assembly, path
parsing, gating, and limits.

### C3 quality

Build the Rust mock HTTP server for OpenAI, Anthropic, and Gemini streaming.
Port the Python suite's behaviors to Rust integration tests, including golden
CLI output. Add CI for fmt, clippy, test, and release builds on Linux, Windows,
and macOS. Add `rust/README.md` with build/run instructions and a parity
checklist. Package a release binary and install notes. Do not switch the
launchers.

## Coordination protocol

Every child must:

1. Work only in its assigned paths.
2. Commit with a clear message.
3. `git push -u origin HEAD`.
4. Report to its parent with exactly one line:
   `DONE role=<id> branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<short>`
   Use `room-cli thread tell --parent "..."` when available. If `--parent` is
   unavailable, write the same line to `rust/.agents/<role>.done` and include it
   in the final message.

Every parent must:

1. Spawn only its assigned children. Never exceed the count.
2. After spawning, freeze code edits until children report.
3. Poll with `room-cli thread wait --children` (up to 100 s; rerun on exit 2).
4. Read reports with `room-cli thread output CHILD_ID --all` or `thread log`.
5. `git fetch origin <branch>` then `git merge --no-edit FETCH_HEAD`.
6. On a `Cargo.lock` conflict, take either side and run
   `cargo test --workspace` to regenerate; never hand-edit the lock.
7. Run `cargo test --workspace` after all merges. Fix integration breakage.
8. `git push -u origin HEAD`.
9. Report the same `DONE` line upward.

If a child fails or stalls, read its log, fix or reassign its work, and report
the gap. Do not spawn replacements beyond the fixed count.

## Budget guard

Before spawning, run `room-cli usage`. If fewer than 1200 Cloud Credits remain,
spawn one lead at a time and report before continuing. Every agent stops
spawning if fewer than 300 credits remain. Do not start Phase 2 or Phase 3.

## Definition of done (Phase 1)

- `cargo test --workspace` green, with new parity tests.
- `cargo build --release` produces `rust/target/release/anyllm`.
- `anyllm --version`, `providers list`, `config show`, `keys list`, `doctor`,
  `ask --json`, `chat` (piped), and the mock streaming tests work.
- The Python tests still pass unchanged.
- The launchers are unchanged; the top reports the exact command to run the
  Rust binary.
- Final report: branch, commit, test counts, parity checklist, known gaps,
  credits used.

## Out of scope

- Phase 2: structural refactor and deduplication.
- Phase 3: macOS-inspired visual redesign.
- New features, new providers, or Python changes.
