#!/usr/bin/env bash
# Build the C harness against the RELEASE cdylibs of both crates.
# (CI's recipes link target/debug for correctness runs; benchmarking debug
# builds would be meaningless, so this script deliberately diverges.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"
STAGE="$LR/benchmarks/cross-language/.stage/c"
mkdir -p "$STAGE"

# Single source of truth for the library version: the repo-root manifest.
LIBLEV_VERSION="$(grep -m1 '^version = "' "$LR/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')"

# _POSIX_C_SOURCE: strict -std=c17 hides clock_gettime/CLOCK_MONOTONIC/gmtime_r.
cc -std=c17 -O2 -DNDEBUG -Wall -Wextra -Werror \
    -D_POSIX_C_SOURCE=200809L \
    -DBENCH_LIBLEV_VERSION="\"$LIBLEV_VERSION\"" \
    -I"$LR/include" \
    -I"$LD_REPO/include" \
    -I"$LR/vinary-tree-interop/include" \
    "$SCRIPT_DIR/bench.c" \
    -L"$LR/target/release" -lliblevenshtein \
    -L"$LD_REPO/target/release" -llibdictenstein \
    -Wl,-rpath,"$LR/target/release" \
    -Wl,-rpath,"$LD_REPO/target/release" \
    -o "$STAGE/bench"

echo "built: $STAGE/bench" >&2
