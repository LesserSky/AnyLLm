#!/usr/bin/env sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
if command -v python3 >/dev/null 2>&1; then
    exec python3 -m anyllm "$@"
fi
exec python -m anyllm "$@"
