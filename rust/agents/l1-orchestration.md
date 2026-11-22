# L1-C: orchestration and quality lead

Project root: `/code/AnyLLm`. Read `rust/PORT_PLAN.md` and
`rust/CONTRACTS.md` first. The Python package is the spec.

You own `rust/orch/**`, `rust/tests/**`, `.github/workflows/**`, and
`rust/README.md`. Your branch is a copy of the top agent's branch.

## Step 0: install Rust

```sh
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
```

## Step 1: verify

```sh
cd rust && cargo test -p anyllm-orch
```

It must be green before you spawn.

## Step 2: spawn exactly 3 managers

Write each prompt file in `/tmp`, then spawn. Do not use built-in subagents,
`codex`, or `claude` CLIs.

```sh
room-cli thread spawn --provider codex --model gpt-6-luna \
  --reasoning-level medium --title "Rust C1 team" \
  --prompt-file /tmp/c1.md
room-cli thread spawn --provider codex --model gpt-6-luna \
  --reasoning-level medium --title "Rust C2 tools" \
  --prompt-file /tmp/c2.md
room-cli thread spawn --provider codex --model gpt-6-luna \
  --reasoning-level medium --title "Rust C3 quality" \
  --prompt-file /tmp/c3.md
```

Manager prompt template:

```
You are <role> in the AnyLLM Rust port. Project root: /code/AnyLLm.
Read rust/PORT_PLAN.md and rust/CONTRACTS.md first. The Python package is the spec.
If cargo is missing, install Rust first:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  . "$HOME/.cargo/env"
  rustup component add rustfmt clippy
Mission: <work package from the plan>.
Own only: <exclusive paths>.
Spawn exactly <N> workers, all in your sandbox:
room-cli thread spawn --provider opencode \
  --model opencode/deepseek-v4.1-flash-free --reasoning-level medium \
  --shared-sandbox --title "<role>-w1" --prompt-file /tmp/<role>-w1.md
... repeat for each worker with a distinct narrow slice ...
Write each worker prompt with: the exact files it owns, the Python functions
to read, the test command, and "commit nothing; report your diff and test
result to me". Workers share your sandbox, so assign disjoint files and use
flock /tmp/anyllm-build.lock for every cargo command.
After all workers report: review every diff, run cargo fmt and
cargo test -p anyllm-orch, commit, push, and report:
DONE role=<role> branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
Do not spawn more workers than assigned. Do not edit other crates.
If Codex returns a plan-limit error, stop and report to your parent; do not retry.
```

Assignments:

- C1 team: 4 workers. Slices: members + runs storage, pool/fan-out +
  concurrency, budgets/prices/synthesis, save/load/adopt + `/team` commands.
  Files: `rust/orch/src/team.rs`, `rust/orch/tests/team.rs`.
- C2 tools: 4 workers. Slices: payloads + capability records, delta assembly +
  path parsing, tool conversation loop, probe + `/tools` commands + local
  read/edit/undo. Files: `rust/orch/src/tool_protocol.rs`,
  `rust/orch/tests/tools.rs`.
- C3 quality: 3 workers. Slices: mock HTTP server + streaming fixtures,
  CLI/integration parity tests + golden output, CI/release packaging +
  `rust/README.md` parity checklist. Files: `rust/tests/**`,
  `.github/workflows/**`, `rust/README.md`.

## Step 3: integrate

After spawning, freeze code edits until reports arrive. Poll
`room-cli thread wait --children` (rerun on exit 2). For each manager:

```sh
git fetch origin <branch> && git merge --no-edit FETCH_HEAD
```

Merge C1, C2, C3 in that order. C3's mock server is the shared test fixture;
if C1/C2 wrote their own, keep C3's and adapt. On `Cargo.lock` conflicts, take
either side and run `cargo test -p anyllm-orch` to regenerate.

Then run `cargo fmt --all`, `cargo test --workspace`, commit, push, and report:

```
DONE role=l1-orchestration branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
```

Stop if fewer than 300 Cloud Credits remain. Do not start Phase 2 or 3.
