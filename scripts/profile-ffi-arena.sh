#!/usr/bin/env bash
# Arena-reuse / zero-copy heap profile for the resource-ABI cursor (wave W8).
#
# Builds the liblevenshtein + libdictenstein cdylibs, compiles the C arena driver
# (bindings/c/tests/arena_profile.c), runs it standalone (which asserts the
# zero-copy packing and cross-batch arena-reuse invariants directly), and then --
# if valgrind is available -- runs it under massif and dhat so the steady-state
# allocation behavior can be read: after warm-up, draining a batch should perform
# ~0 heap allocations because the cursor arena is cleared-and-refilled, not
# reallocated. This is the dynamic corroboration of the LLEV-ARENA invariants
# proven in docs/verification/verus/ffi_batch_arena.rs.
#
# Usage:
#   scripts/profile-ffi-arena.sh            # build, run, and (if present) valgrind
#   SKIP_VALGRIND=1 scripts/profile-ffi-arena.sh   # assertions only, no profiler
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DICT="${LIBDICTENSTEIN_DIR:-$REPO/../libdictenstein}"
INTEROP="${VINARY_TREE_INTEROP_ROOT:-$REPO/../vinary-tree-interop}"
OUT="$REPO/target/ffi-arena-profile"
mkdir -p "$OUT"

echo "== building cdylibs =="
cargo build --features native-bindings-full
cargo build --manifest-path "$DICT/Cargo.toml" --no-default-features --features ffi

echo "== compiling arena_profile.c =="
cc -std=c17 -Wall -Wextra -Werror -g \
  -Iinclude -I"$DICT/include" -I"$INTEROP/include" \
  bindings/c/tests/arena_profile.c \
  -Ltarget/debug -lliblevenshtein \
  -L"$DICT/target/debug" -llibdictenstein \
  -Wl,-rpath,"$REPO/target/debug" \
  -Wl,-rpath,"$DICT/target/debug" \
  -o "$OUT/arena_profile"

echo "== standalone run (asserts zero-copy packing + arena reuse) =="
"$OUT/arena_profile"

if [[ "${SKIP_VALGRIND:-0}" == "1" ]] || ! command -v valgrind >/dev/null 2>&1; then
  echo "valgrind not run (SKIP_VALGRIND set or valgrind absent); assertions passed."
  exit 0
fi

echo "== massif heap profile =="
valgrind --tool=massif --massif-out-file="$OUT/massif.out" "$OUT/arena_profile" >/dev/null 2>&1 || true
if command -v ms_print >/dev/null 2>&1; then
  ms_print "$OUT/massif.out" | sed -n '1,40p' | tee "$OUT/massif.txt"
fi

echo "== dhat allocation profile =="
# dhat reports total allocation events; with WARMUP_QUERIES warm drains, a
# non-reusing arena would show allocations scaling with the drain count, whereas
# a reused arena keeps total allocations flat in the warm loop.
valgrind --tool=dhat --dhat-out-file="$OUT/dhat.out.json" "$OUT/arena_profile" 2>"$OUT/dhat.txt" || true
grep -iE "total:|at t-gmax|allocs" "$OUT/dhat.txt" | tee -a "$OUT/dhat.summary" || true

echo "arena profile complete; artifacts in $OUT"
