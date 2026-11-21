#!/bin/sh
# Builds the reference-solver images (once) and compares every case.
# Usage: bench/compare/run_all.sh [OUT_DIR] [extra compare.py options]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-compare_out}
[ $# -gt 0 ] && shift
for img in fastercap palace; do
    docker image inspect "faradice-bench/$img" >/dev/null 2>&1 \
        || docker build -t "faradice-bench/$img" "$here/docker/$img"
done
for case in "$here"/cases/*.faradice.toml; do
    python3 "$here/compare.py" "$case" --out "$out" "$@"
done
cat "$out"/*/summary.md > "$out/summary.md"
echo "→ $out/summary.md"
