#!/usr/bin/env bash
# Chain the remaining DATA COLLECTION behind the parallel streams so it
# completes without supervision. The streams can run for many hours and the
# steps after them must not depend on an interactive session outliving them.
#
#   1. wait for every stream's completion sentinel (.stream-<ccd>-done)
#   2. run the >=51-sample hypothesis arms (pgmcp experiments 178-184)
#   3. aggregate
#   4. publish every raw cell to pgmcp (idempotent, content-addressed)
#
# Every step is individually resumable and skips work already on disk, so
# re-running this script is safe. Failures are recorded in state.tsv and do
# not abort the chain — a later step's data is still worth collecting.
#
# EXECUTION IS STRICTLY SERIAL: exactly one single-threaded benchmark process
# runs at a time, pinned by the manifest cpuset. Parallel streams were tried
# and reverted — measured anchors showed MAD inflated 2-4x (2.1-3.8% vs 1.2%
# on a quiet machine), i.e. concurrency perturbs the very quantity being
# measured. Throughput is not worth biased numbers.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"

[ $# -ge 1 ] || { echo "usage: run-remaining.sh <results_dir> [ccd...]" >&2; exit 2; }
RESULTS_DIR="$(cd "$1" && pwd)"; shift
STREAMS=("$@")
[ ${#STREAMS[@]} -gt 0 ] || STREAMS=(8-15 16-23 24-31)
STATE_TSV="$RESULTS_DIR/state.tsv"

log() { printf '[remaining %s] %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; }
state_row() { printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$STATE_TSV"; }

# ---------------------------------------------------------------------------
# 1. Wait for the serial atlas sweep
# ---------------------------------------------------------------------------
if [ ! -f "$RESULTS_DIR/.atlas-done" ]; then
    log "waiting for the serial atlas sweep to finish..."
    while [ ! -f "$RESULTS_DIR/.atlas-done" ]; do
        if ! pgrep -f 'run-all.sh --results' >/dev/null 2>&1; then
            sleep 10
            [ -f "$RESULTS_DIR/.atlas-done" ] && break
            log "atlas sweep vanished without its sentinel"
            state_row "atlas-sweep" "vanished" "see logs/atlas-sweep.log"
            break
        fi
        sleep 120
    done
fi
log "atlas sweep finished"

# ---------------------------------------------------------------------------
# 2. Hypothesis arms, serial and contention-free on one CCD
# ---------------------------------------------------------------------------
log "running >=51-sample hypothesis cells (serial)"
if bash "$SCRIPT_DIR/run-hypothesis-cells.sh" "$RESULTS_DIR" \
        >> "$RESULTS_DIR/logs/hypothesis-cells.log" 2>&1; then
    state_row "hypothesis-cells" "ok" ""
else
    state_row "hypothesis-cells" "failed" "see logs/hypothesis-cells.log"
    log "hypothesis cells FAILED (continuing)"
fi

# ---------------------------------------------------------------------------
# 3-4. Aggregate, then publish raw cells to pgmcp (the system of record)
# ---------------------------------------------------------------------------
log "aggregating"
python3 "$SCRIPT_DIR/aggregate.py" "$RESULTS_DIR" \
    >> "$RESULTS_DIR/logs/aggregate.log" 2>&1 \
    && state_row "aggregate" "ok" "" \
    || state_row "aggregate" "failed" "see logs/aggregate.log"

log "publishing to pgmcp"
python3 "$SCRIPT_DIR/pgmcp-upload.py" "$RESULTS_DIR" \
    >> "$RESULTS_DIR/logs/pgmcp-upload.log" 2>&1 \
    && state_row "pgmcp-upload" "ok" "" \
    || state_row "pgmcp-upload" "failed" "re-run scripts/pgmcp-upload.py"

log "ALL DATA COLLECTION COMPLETE: $RESULTS_DIR"
