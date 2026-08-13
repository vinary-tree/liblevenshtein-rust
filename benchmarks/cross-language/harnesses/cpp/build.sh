#!/usr/bin/env bash
# Build the C++ harness against the RELEASE cdylibs (see c/build.sh for the
# debug-vs-release rationale). Compiles are pinned away from the timing
# cores (2-9) so an in-flight timed run is never perturbed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"
STAGE="$LR/benchmarks/cross-language/.stage/cpp"
mkdir -p "$STAGE"

LIBLEV_VERSION="$(grep -m1 '^version = "' "$LR/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')"

taskset -c 16-31 c++ -std=c++23 -O2 -DNDEBUG -Wall -Wextra -Werror \
    -DBENCH_LIBLEV_VERSION="\"$LIBLEV_VERSION\"" \
    -I"$LR/include" \
    -I"$LD_REPO/include" \
    -I"$LR/vinary-tree-interop/include" \
    "$SCRIPT_DIR/bench.cpp" \
    -L"$LR/target/release" -lliblevenshtein \
    -L"$LD_REPO/target/release" -llibdictenstein \
    -Wl,-rpath,"$LR/target/release" \
    -Wl,-rpath,"$LD_REPO/target/release" \
    -o "$STAGE/bench"

echo "built: $STAGE/bench" >&2
