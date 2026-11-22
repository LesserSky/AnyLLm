#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

flock /tmp/anyllm-build.lock cargo build --manifest-path rust/Cargo.toml -p anyllm-cli --bin anyllm
actual=$(rust/target/debug/anyllm --version)
expected=$(cat rust/tests/golden/version.txt)
if [ "$actual" != "$expected" ]; then
    printf 'version output mismatch\nexpected: %s\nactual:   %s\n' "$expected" "$actual" >&2
    exit 1
fi
