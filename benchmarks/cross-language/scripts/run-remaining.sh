#!/usr/bin/env bash
# Chain the remaining DATA COLLECTION behind the atlas sweep so it completes
# without supervision: the sweep can run for a day or more, and the steps
# after it must not depend on an interactive session still being attached.
#
#   1. wait for the atlas sweep's completion sentinel
#   2. time the C++ legacy pair (cpp-legacy), which the sweep did not include
#   3. run the >=51-sample hypothesis arms (pgmcp experiments 178-184)
#   4. aggregate
#   5. publish every raw cell to pgmcp (idempotent, content-addressed)
#
# Every step is individually resumable and skips work already on disk, so
# re-running this script is safe. Failures are recorded in state.tsv and do
# not abort the chain — a later step's data is still worth collecting.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"

[ $# -eq 1 ] || { echo "usage: run-remaining.sh <results_dir>" >&2; exit 2; }
RESULTS_DIR="$(cd "$1" && pwd)"
STATE_TSV="$RESULTS_DIR/state.tsv"

log() { printf '[remaining %s] %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; }
state_row() { printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$STATE_TSV"; }

# ---------------------------------------------------------------------------
# 1. Wait for the atlas sweep (poll; it writes .atlas-done when finished)
# ---------------------------------------------------------------------------
if [ ! -f "$RESULTS_DIR/.atlas-done" ]; then
    log "waiting for the atlas sweep to finish..."
    while [ ! -f "$RESULTS_DIR/.atlas-done" ]; do
        if ! pgrep -f 'run-all.sh --results' >/dev/null 2>&1; then
            sleep 10
            [ -f "$RESULTS_DIR/.atlas-done" ] && break
            log "atlas sweep vanished without its sentinel — continuing anyway"
            state_row "atlas-sweep" "vanished" "no sentinel; see logs/atlas-sweep.log"
            break
        fi
        sleep 120
    done
fi
log "atlas sweep done (exit $(cat "$RESULTS_DIR/.atlas-done" 2>/dev/null || echo '?'))"

# ---------------------------------------------------------------------------
# 2. C++ legacy pair — the third head-to-head, added after the sweep launched
# ---------------------------------------------------------------------------
log "timing cpp-legacy (45 query cells + construct + memory)"
if bash "$SCRIPT_DIR/run-all.sh" --results "$RESULTS_DIR" --targets cpp-legacy \
        --no-aggregate >> "$RESULTS_DIR/logs/cpp-legacy-sweep.log" 2>&1; then
    state_row "cpp-legacy-timing" "ok" ""
else
    state_row "cpp-legacy-timing" "failed" "see logs/cpp-legacy-sweep.log"
    log "cpp-legacy timing FAILED (continuing)"
fi

# ---------------------------------------------------------------------------
# 3. Hypothesis arms at the protocol-prescribed >=51 replicates
# ---------------------------------------------------------------------------
log "running >=51-sample hypothesis cells"
if bash "$SCRIPT_DIR/run-hypothesis-cells.sh" "$RESULTS_DIR" \
        >> "$RESULTS_DIR/logs/hypothesis-cells.log" 2>&1; then
    state_row "hypothesis-cells" "ok" ""
else
    state_row "hypothesis-cells" "failed" "see logs/hypothesis-cells.log"
    log "hypothesis cells FAILED (continuing)"
fi

# ---------------------------------------------------------------------------
# 4-5. Aggregate, then publish raw cells to pgmcp (the system of record)
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
