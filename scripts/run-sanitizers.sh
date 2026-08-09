#!/usr/bin/env bash
# Dynamic-analysis harness for the resource-ABI boundary (wave W8).
#
# Runs the binding-integration test suite under AddressSanitizer (+
# LeakSanitizer) and ThreadSanitizer to catch, at the FFI boundary, the classes
# of defect that safe Rust and `catch_unwind` cannot: use-after-free,
# out-of-bounds access, leaks of leased/retained resources, and data races
# across the call gate. These exercise the same VT-LIFE/LLEV-LEASE/LLEV-ARENA
# invariants the correspondence tests pin, but dynamically at machine level.
#
# Requires a nightly toolchain with the `rust-src` component (for
# `-Zbuild-std`, which rebuilds std with the sanitizer runtime).
#
# Usage:
#   scripts/run-sanitizers.sh                 # asan+lsan then tsan, whole suite
#   SANITIZER_ONLY=address scripts/run-sanitizers.sh --test abi_paging_correspondence
#   SANITIZER_NIGHTLY=nightly-2026-04-21 scripts/run-sanitizers.sh
set -euo pipefail

TARGET="${SANITIZER_TARGET:-x86_64-unknown-linux-gnu}"
NIGHTLY="${SANITIZER_NIGHTLY:-nightly}"
FEATURES="${SANITIZER_FEATURES:-binding-integration-tests}"
ONLY="${SANITIZER_ONLY:-address thread}"

run_one() {
  local san="$1"; shift
  echo "== ${san}sanitizer =="
  RUSTFLAGS="-Zsanitizer=${san}" \
  RUSTDOCFLAGS="-Zsanitizer=${san}" \
  ASAN_OPTIONS="detect_leaks=1:detect_stack_use_after_return=1" \
    cargo +"$NIGHTLY" test -Zbuild-std \
      --target "$TARGET" --features "$FEATURES" "$@"
}

for san in $ONLY; do
  run_one "$san" "$@"
done

echo "sanitizers: all requested runs completed cleanly"
