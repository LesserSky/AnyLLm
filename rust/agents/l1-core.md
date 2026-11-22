# L1-A: core crate lead

Project root: `/code/AnyLLm`. Read `rust/PORT_PLAN.md` and
`rust/CONTRACTS.md` first. The Python package is the spec.

You own `rust/core/**`. Your branch is a copy of the top agent's branch.

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
cd rust && cargo test -p anyllm-core
```

It must be green before you spawn. If not, fix only the baseline.

## Step 2: spawn exactly 3 managers

Write each prompt file in `/tmp` using the template below, then spawn. Do not
use built-in subagents, `codex`, or `claude` CLIs.

```sh
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust A1 adapters" \
  --prompt-file /tmp/a1.md
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust A2 transport" \
  --prompt-file /tmp/a2.md
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust A3 data" \
  --prompt-file /tmp/a3.md
```

Manager prompt template (fill in the role, mission, paths, worker count, and
worker slices):

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
cargo test -p anyllm-core, commit, push, and report:
DONE role=<role> branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
Do not spawn more workers than assigned. Do not edit other crates.
```

Assignments:

- A1 adapters: 5 workers. Slices: OpenAI streaming, Anthropic streaming,
  Gemini streaming, usage/reasoning/finish mapping, model listing and custom
  provider payloads. Files: `rust/core/src/adapters.rs`,
  `rust/core/tests/adapters.rs`.
- A2 transport: 4 workers. Slices: SSE framing and chunk boundaries, retries
  and timeouts, proxy/TLS/URL/status mapping, cancellation and JSON requests.
  Files: `rust/core/src/transport.rs`, `rust/core/tests/transport.rs`.
- A3 data: 4 workers. Slices: config + keystore, session + catalog, term +
  errors, cross-platform permissions/DPAPI. Files:
  `rust/core/src/{config,keystore,session,catalog,term,errors}.rs`,
  `rust/core/tests/data.rs`.

## Step 3: integrate

After spawning, freeze code edits until reports arrive. Poll
`room-cli thread wait --children` (rerun on exit 2). For each manager:

```sh
git fetch origin <branch> && git merge --no-edit FETCH_HEAD
```

Merge A1, A2, A3 in that order. Resolve conflicts in `rust/core` by matching
the Python behavior. On `Cargo.lock` conflicts, take either side and run
`cargo test -p anyllm-core` to regenerate.

Then run `cargo fmt -p anyllm-core`, `cargo test -p anyllm-core`, commit, and
`git push -u origin HEAD`. Report to your parent:

```
DONE role=l1-core branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
```

Stop if fewer than 300 Cloud Credits remain. Do not start Phase 2 or 3.
