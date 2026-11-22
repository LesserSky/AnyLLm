# AnyLLM Rust CLI

The Rust workspace is the in-progress native port of the Python AnyLLM CLI. The
Python package and launchers remain the behavioral reference and stay in place
while parity work continues.

## Build and run

Install stable Rust with `rustup` and the `rustfmt` and `clippy` components.
From the repository root:

```sh
cargo build --manifest-path rust/Cargo.toml --release -p anyllm-cli --bin anyllm
rust/target/release/anyllm --version
```

On Windows, run `rust\target\release\anyllm.exe --version`. The binary is
self-contained; copy it to a directory on `PATH` to install it. The Python
launcher is not replaced by this build.

## Release packaging

The Rust GitHub Actions workflow builds a release binary on Linux, Windows,
and macOS and uploads each platform's executable as a workflow artifact. Select
the artifact for your operating system and architecture, then place `anyllm`
(`anyllm.exe` on Windows) in a directory on `PATH`. The workflow artifacts are
CI outputs; a tagged public release still needs to be published separately.

## Parity checklist

Python under `anyllm/anyllm` defines the expected command behavior, output,
JSON/file formats, and exit codes.

- [x] `--version` output has a golden check.
- [x] OpenAI, Anthropic, and Gemini streaming adapters have local HTTP fixture
  coverage in `rust/tests`.
- [ ] CLI command and option surface matches Python `build_parser`.
- [ ] `providers list`, `config show`, `keys list`, and `doctor` match Python
  output and exit behavior.
- [ ] `ask --json` and piped `chat` match Python against a local mock server.
- [ ] REPL, orchestration commands, and streaming interruption match Python.
- [ ] Python launchers remain unchanged until the top-level parity gate passes.

Run the current Rust checks from the repository root:

```sh
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo test --manifest-path rust/Cargo.toml --workspace
cargo test --manifest-path rust/tests/Cargo.toml
sh rust/tests/cli_parity.sh
```

The standalone mock fixture package uses only workspace crates and standard
library networking; it adds no third-party dependencies.
