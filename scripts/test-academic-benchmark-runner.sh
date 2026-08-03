#!/usr/bin/env bash
# Focused regression checks for the academic benchmark runner's shell guards.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_ROOT/scripts/run-academic-benchmarks.sh"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# Sourcing exposes the pure guard functions without starting a benchmark.
# shellcheck source=run-academic-benchmarks.sh
source "$RUNNER"

printf '999999.0 0.0 0.0 1/1 1\n' >"$SCRATCH/high-loadavg"
if (LOADAVG_PATH="$SCRATCH/high-loadavg" \
        LOAD_LIMIT_FACTOR="0.75" \
        DRY_RUN="0" \
        ALLOW_HIGH_LOAD="0" \
        load_guard) >"$SCRATCH/high.out" 2>&1; then
    printf 'load_guard accepted a deliberately excessive load\n' >&2
    exit 1
fi
if ! grep -Fq 'refusing wall-time-producing benchmark under high load' "$SCRATCH/high.out"; then
    printf 'load_guard did not emit its refusal diagnostic\n' >&2
    cat "$SCRATCH/high.out" >&2
    exit 1
fi
if grep -Fq 'awk: fatal' "$SCRATCH/high.out"; then
    printf 'load_guard reused an awk-reserved variable\n' >&2
    cat "$SCRATCH/high.out" >&2
    exit 1
fi

printf '0.0 0.0 0.0 1/1 1\n' >"$SCRATCH/low-loadavg"
LOADAVG_PATH="$SCRATCH/low-loadavg" \
    LOAD_LIMIT_FACTOR="0.75" \
    DRY_RUN="0" \
    ALLOW_HIGH_LOAD="0" \
    load_guard >"$SCRATCH/low.out" 2>&1

printf 'academic benchmark runner guard tests: PASS\n'
