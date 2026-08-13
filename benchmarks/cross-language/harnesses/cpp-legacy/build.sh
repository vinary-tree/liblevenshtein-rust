#!/usr/bin/env bash
# Build the legacy liblevenshtein-cpp baseline and its harness.
#
# The upstream repo is built OUT-OF-SOURCE into this program's staging area
# so the checkout at $CPP_REPO is never written to (it carries the user's
# uncommitted work). Its own build/ dir references a conda toolchain
# (envs/ll-cpp) that no longer exists; we build with the system toolchain
# instead and record exactly what we used in provenance.json.
#
# CMAKE_COMPILE_WARNING_AS_ERROR=OFF: the project sets warnings-as-errors,
# and a 2024 codebase compiled by a much newer GCC would otherwise fail on
# new diagnostics unrelated to correctness. This changes no code generation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP_REPO="${XL_LEGACY_CPP_REPO:-/home/dylon/Workspace/universal-automata/liblevenshtein-cpp}"
STAGE="$XL/.stage/cpp-legacy"
BUILD="$STAGE/lib-build"
BUILD_CPUSET="${XL_BUILD_CPUSET:-16-31}"

[ -d "$CPP_REPO" ] || { echo "cpp-legacy: repo not found: $CPP_REPO" >&2; exit 4; }
mkdir -p "$STAGE"

# ---------------------------------------------------------------------------
# Provenance: the checkout has uncommitted modifications (including to the
# DAWG hot path), so the commit alone does not identify what we measured.
# Record commit + dirty flag + a digest of the working-tree diff.
# ---------------------------------------------------------------------------
COMMIT="$(git -C "$CPP_REPO" rev-parse HEAD)"
DIRTY=false
DIFF_SHA256="-"
if [ -n "$(git -C "$CPP_REPO" status --porcelain)" ]; then
    DIRTY=true
    DIFF_SHA256="$(git -C "$CPP_REPO" diff HEAD | sha256sum | cut -d' ' -f1)"
fi
VERSION="$(grep -m1 -E '^\s+VERSION [0-9]' "$CPP_REPO/CMakeLists.txt" | tr -dc '0-9.')"
LEGACY_ID="universal-automata/liblevenshtein-cpp@${COMMIT:0:12}$([ "$DIRTY" = true ] && echo "+dirty")"

cat > "$STAGE/provenance.json" <<EOF
{
  "source_repo": "$CPP_REPO",
  "upstream": "https://github.com/universal-automata/liblevenshtein-cpp",
  "git_commit": "$COMMIT",
  "working_tree_dirty": $DIRTY,
  "working_tree_diff_sha256": "$DIFF_SHA256",
  "declared_version": "$VERSION",
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "toolchain": "$(c++ --version | head -1)",
  "protobuf": "$(pkg-config --modversion protobuf 2>/dev/null || echo unknown)",
  "cmake_flags": "-DCMAKE_BUILD_TYPE=Release -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
  "note": "Benchmarks the working tree (the version under active development), not clean HEAD; the upstream repo has no release tags. Set XL_LEGACY_CPP_REPO to a clean 'git worktree add' path to measure published HEAD instead."
}
EOF

# ---------------------------------------------------------------------------
# Library, then harness. Both pinned off the timing cores.
# ---------------------------------------------------------------------------
if [ ! -f "$BUILD/proto/liblevenshtein.so" ]; then
    taskset -c "$BUILD_CPUSET" cmake -S "$CPP_REPO" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF >&2
    taskset -c "$BUILD_CPUSET" cmake --build "$BUILD" -j8 >&2
fi

taskset -c "$BUILD_CPUSET" c++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra \
    -DBENCH_LEGACY_CPP_VERSION="\"$VERSION\"" \
    -DBENCH_LEGACY_CPP_ID="\"$LEGACY_ID\"" \
    -I"$CPP_REPO/src" \
    -I"$CPP_REPO/third-party" \
    -I"$BUILD/generated" \
    "$SCRIPT_DIR/bench.cpp" \
    -L"$BUILD/proto" -llevenshtein \
    -Wl,-rpath,"$BUILD/proto" \
    -o "$STAGE/bench"

echo "built: $STAGE/bench ($LEGACY_ID)" >&2
