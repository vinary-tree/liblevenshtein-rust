#!/usr/bin/env bash
# Build the Lua 5.4 harness artifacts into benchmarks/cross-language/.stage/lua:
#   vinary_tree/libdictenstein.so   facade C module vs the RELEASE liblibdictenstein
#   vinary_tree/liblevenshtein.so   facade C module vs the RELEASE libliblevenshtein
#   bench_clock.so                  monotonic-clock shim (PROTOCOL.md section 9)
# Mirrors .github/workflows/ci.yml (lua-bindings job) but links target/release,
# adds -O2, and pins the compile to a benchmark-safe cpuset.
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_DIR="$(cd "$HARNESS_DIR/../.." && pwd)"
LR_DIR="$(cd "$XL_DIR/../.." && pwd)"
LD_DIR="$(cd "$LR_DIR/../libdictenstein" && pwd)"
STAGE_DIR="$XL_DIR/.stage/lua"
CPUSET="${XL_BUILD_CPUSET:-16-31}"

mkdir -p "$STAGE_DIR/vinary_tree"

lua_pkg=""
for candidate in lua5.4 lua54 lua-5.4; do
    if pkg-config --exists "$candidate" 2>/dev/null; then
        lua_pkg="$candidate"
        break
    fi
done
if [[ -z "$lua_pkg" ]]; then
    if pkg-config --exists lua 2>/dev/null \
        && [[ "$(pkg-config --modversion lua)" == 5.4* ]]; then
        lua_pkg="lua"
    else
        echo "build-lua: no Lua 5.4 pkg-config module found" >&2
        exit 1
    fi
fi
read -r -a lua_cflags <<< "$(pkg-config --cflags "$lua_pkg")"

taskset -c "$CPUSET" cc -std=c17 -O2 -Wall -Wextra -Werror -shared -fPIC \
    "${lua_cflags[@]}" \
    -I"$LD_DIR/include" -I"$LR_DIR/vinary-tree-interop/include" \
    -I"$LR_DIR/vinary-tree-interop/bindings/lua" \
    "$LD_DIR/bindings/lua/src/libdictenstein_lua.c" \
    -L"$LD_DIR/target/release" -llibdictenstein \
    -o "$STAGE_DIR/vinary_tree/libdictenstein.so"

taskset -c "$CPUSET" cc -std=c17 -O2 -Wall -Wextra -Werror -shared -fPIC \
    "${lua_cflags[@]}" \
    -I"$LR_DIR/include" -I"$LR_DIR/vinary-tree-interop/include" \
    -I"$LR_DIR/vinary-tree-interop/bindings/lua" \
    "$LR_DIR/bindings/lua/src/liblevenshtein_lua.c" \
    -L"$LR_DIR/target/release" -lliblevenshtein \
    -o "$STAGE_DIR/vinary_tree/liblevenshtein.so"

taskset -c "$CPUSET" cc -std=c17 -O2 -Wall -Wextra -Werror -shared -fPIC \
    "${lua_cflags[@]}" \
    "$HARNESS_DIR/bench_clock.c" \
    -o "$STAGE_DIR/bench_clock.so"

echo "build-lua: staged Lua harness modules in $STAGE_DIR" >&2
