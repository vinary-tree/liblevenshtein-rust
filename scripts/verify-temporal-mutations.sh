#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${TEMPORAL_MUTANTS_OUTPUT:-$ROOT/target/temporal-mutants}"
SCRATCH="${TEMPORAL_MUTANTS_SCRATCH:-$OUTPUT/scratch}"
EXPECTED_VERSION="27.1.0"

if ! command -v cargo-mutants >/dev/null 2>&1; then
  echo "error: cargo-mutants $EXPECTED_VERSION is required" >&2
  exit 2
fi

actual_version="$(cargo mutants --version | awk '{print $2}')"
if [[ "$actual_version" != "$EXPECTED_VERSION" ]]; then
  echo "error: cargo-mutants $EXPECTED_VERSION required; found $actual_version" >&2
  exit 2
fi

mkdir -p "$OUTPUT" "$SCRATCH"
export TMPDIR="$SCRATCH"
for group in \
  msm-cutoff \
  interner-collision \
  interner-reuse \
  transition-cache \
  timestamped-subsumption \
  exact-workspace-storage \
  exact-workspace-core \
  exact-workspace-bounded-paths \
  timestamped-knn-finalization; do
  cargo mutants \
    --manifest-path "$ROOT/Cargo.toml" \
    --in-place \
    --config "$ROOT/.cargo/temporal-mutants/$group.toml" \
    --output "$OUTPUT/$group" \
    --jobserver true \
    --jobserver-tasks 2 \
    --no-shuffle \
    --no-times
done
