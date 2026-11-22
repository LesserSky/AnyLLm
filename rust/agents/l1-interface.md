# L1-B: interface crate lead

Project root: `/code/AnyLLm`. Read `rust/PORT_PLAN.md` and
`rust/CONTRACTS.md` first. The Python package is the spec.

You own `rust/cli/**`. Your branch is a copy of the top agent's branch.

## Step 1: verify

```sh
cd rust && cargo test -p anyllm-cli && cargo build -p anyllm-cli
```

The skeleton must build. `./target/debug/anyllm --version` must print
`anyllm 1.0.0`.

## Step 2: spawn exactly 3 managers

Write each prompt file in `/tmp`, then spawn. Do not use built-in subagents,
`codex`, or `claude` CLIs.

```sh
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust B1 CLI" \
  --prompt-file /tmp/b1.md
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust B2 REPL" \
  --prompt-file /tmp/b2.md
room-cli thread spawn --provider codex --model gpt-6.2-astra \
  --reasoning-level high --title "Rust B3 render" \
  --prompt-file /tmp/b3.md
```

Manager prompt template:

```
You are <role> in the AnyLLM Rust port. Project root: /code/AnyLLm.
Read rust/PORT_PLAN.md and rust/CONTRACTS.md first. The Python package is the spec.
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
cargo test -p anyllm-cli, commit, push, and report:
DONE role=<role> branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
Do not spawn more workers than assigned. Do not edit other crates.
```

Assignments:

- B1 CLI: 5 workers. Slices: argument parser + help/version, `ask` one-shot +
  `--json` + stdin pipes, `models`/`providers`, `keys`/`config`/`doctor`,
  `targets`/`prices`/`team`/`tools`/`capabilities` subcommands. Files:
  `rust/cli/src/{main,args,commands}.rs`.
- B2 REPL: 4 workers. Slices: prompt/history/multiline, slash command registry,
  streaming printer + Ctrl+C/Ctrl+D, session save/load/model/provider switching.
  Files: `rust/cli/src/repl.rs`.
- B3 render: 4 workers. Slices: banner/spinner, status line/reasoning, tables,
  color/plain/Windows console. Files: `rust/cli/src/render.rs`; coordinate any
  change to `rust/core/src/term.rs` with L1-A, do not edit it directly.

## Step 3: integrate

After spawning, freeze code edits until reports arrive. Poll
`room-cli thread wait --children` (rerun on exit 2). For each manager:

```sh
git fetch origin <branch> && git merge --no-edit FETCH_HEAD
```

Merge B1, B2, B3 in that order. B1 owns the command skeleton; B2 and B3 plug
into it. If they created conflicting entry points, keep B1's dispatch and
adapt B2/B3. On `Cargo.lock` conflicts, take either side and run
`cargo test -p anyllm-cli` to regenerate.

Then run `cargo fmt -p anyllm-cli`, `cargo test -p anyllm-cli`,
`cargo build --release -p anyllm-cli`, and smoke test:

```sh
./target/release/anyllm --version
./target/release/anyllm providers list
./target/release/anyllm config show
```

Commit, push, and report:

```
DONE role=l1-interface branch=<branch> commit=<sha> tests=<passed>/<total> gaps=<...>
```

Stop if fewer than 300 Cloud Credits remain. Do not start Phase 2 or 3.
