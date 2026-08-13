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
# CCD 0-7 is deliberately LEFT FREE for the user's other development work
# (their processes are unpinned and would otherwise contend with measurement
# cores). The hypothesis arms run SERIALLY on one owned CCD afterwards, so
# they are measured free of cross-stream contention: they are the arms that
# formally decide the preregistered experiments.
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
# 1. Wait for all parallel streams
# ---------------------------------------------------------------------------
log "waiting for streams: ${STREAMS[*]}"
while :; do
    pending=0
    for ccd in "${STREAMS[@]}"; do
        [ -f "$RESULTS_DIR/.stream-${ccd}-done" ] && continue
        if pgrep -f "run-stream.sh $RESULTS_DIR $ccd" >/dev/null 2>&1; then
            pending=$((pending + 1))
        else
            sleep 10
            if [ ! -f "$RESULTS_DIR/.stream-${ccd}-done" ]; then
                log "stream $ccd vanished without its sentinel"
                state_row "stream-${ccd}" "vanished" "see logs/stream-${ccd}.log"
                : > "$RESULTS_DIR/.stream-${ccd}-done"
            fi
        fi
    done
    [ "$pending" -eq 0 ] && break
    sleep 120
done
log "all streams finished"

# ---------------------------------------------------------------------------
# 2. Hypothesis arms, serial and contention-free on one CCD
# ---------------------------------------------------------------------------
log "running >=51-sample hypothesis cells (serial, CCD 16-23)"
if XL_STREAM_CCD="16-23" bash "$SCRIPT_DIR/run-hypothesis-cells.sh" "$RESULTS_DIR" \
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
