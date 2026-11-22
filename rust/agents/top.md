# Top integrator: AnyLLM Rust port

You are the top of a 50-agent tree. Project root: `/code/AnyLLm` (run `pwd`
first; use the root Cloudroom gives you). Read `rust/PORT_PLAN.md` before
anything else. The Python package is the spec.

## First 30 minutes

0. Install Rust if it is missing:

```sh
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
```

1. Verify the baseline: `cd rust && cargo test --workspace`. It must be green
   (63 tests). If it is not, fix only the baseline before spawning.
2. Read `rust/core/src/lib.rs`, `rust/orch/src/lib.rs`, `rust/Cargo.toml`,
   `anyllm/cli.py` (top-level structure), `anyllm/repl.py` (class Repl),
   `anyllm/team.py`, and `anyllm/tool_protocol.py`.
3. Write `rust/CONTRACTS.md`: crate boundaries, the public types the CLI uses,
   how `cli` bridges `orch::sys` types to `core` types, dependency rules, and
   the exact binary name (`anyllm`). Keep it under 120 lines. Commit it.
4. `git add -A && git commit -m "rust: freeze port contracts"` and
   `git push -u origin HEAD`. Note your branch with `git branch --show-current`.

## Spawn exactly 3 leads

Use these files, verbatim. Do not rewrite them.

```sh
room-cli thread spawn --provider codex --model gpt-6.1-terra \
  --reasoning-level medium --title "Rust L1-A core" \
  --prompt-file rust/agents/l1-core.md

room-cli thread spawn --provider codex --model gpt-6.1-terra \
  --reasoning-level medium --title "Rust L1-B interface" \
  --prompt-file rust/agents/l1-interface.md

room-cli thread spawn --provider codex --model gpt-6.1-terra \
  --reasoning-level medium --title "Rust L1-C orchestration" \
  --prompt-file rust/agents/l1-orchestration.md
```

Check `room-cli usage` before spawning. If fewer than 1200 Cloud Credits
remain, spawn one lead, wait for it, then decide. Stop spawning below 300.
If Codex returns a plan-limit error, stop and report to your parent. Do not
retry, and do not switch models.

## While children run

- Freeze code edits. Do not implement their areas yourself.
- Poll `room-cli thread wait --children` (up to 100 s; rerun on exit 2).
- Read reports with `room-cli thread output CHILD_ID --all`.
- If a child is stuck, read `room-cli thread log CHILD_ID` and send a short,
  specific instruction with `room-cli thread tell CHILD_ID "..."`.
- If a child cannot start because a login is missing or expired, tell your
  parent which login to fix. Do not switch provider or model.

## Integration

1. Merge the three lead branches into your branch, one at a time:
   `git fetch origin <branch> && git merge --no-edit FETCH_HEAD`.
2. On `Cargo.lock` conflicts, take either side and run
   `cargo test --workspace` to regenerate. Never hand-edit the lock.
3. Run `cargo test --workspace` and `cargo build --release`.
4. Smoke test: `./target/release/anyllm --version`,
   `./target/release/anyllm providers list`, `./target/release/anyllm config show`.
5. If a whole area is broken, send the exact failing output to the responsible
   lead and ask for a fix commit. Do not silently rewrite their area.
6. `cargo fmt --all`, commit, `git push -u origin HEAD`.

## Report

Send your parent exactly one final message, under 30 lines:

```
DONE role=top branch=<branch> commit=<sha> tests=<passed>/<total> \
  gaps=<short list> credits=<used/remaining>
```

Include the command to run the Rust binary, and any parity gaps. Do not start
Phase 2 (structure) or Phase 3 (visual redesign). Do not spawn more than the
50-agent tree. Do not spawn replacement agents beyond the fixed count.
