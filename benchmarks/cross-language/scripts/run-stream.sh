#!/usr/bin/env bash
# Run a subset of targets as one PARALLEL STREAM pinned to a single CCD.
#
#   run-stream.sh <results_dir> <ccd-core-range> <target>[,<target>...]
#   e.g. run-stream.sh results/phase0 16-23 python,go,ruby
#
# Streams exist because the machine has 32 cores and serial execution of the
# full matrix projects to 25-50 h. Each stream owns one CCD, so streams never
# share an L3; targets within a stream still run strictly serially.
#
# WHY EACH STREAM MEASURES ITS OWN ANCHOR. A calibration probe on
# 2026-08-13 ran identical work 4.74% faster on core 24 than on core 2, with
# a MAD of only 1.2% — core placement shifts results by more than the noise
# floor. Cross-stream absolute times are therefore not interchangeable. Each
# stream measures the pure-Rust anchor on ITS OWN cores into
# <results>/anchors/<ccd>/, and the atlas's "target / anchor" overhead ratios
# are computed within-stream. Head-to-head pairs are unaffected because both
# arms of a pair are always assigned to the same stream.
#
# The anchor is measured FIRST so that, if a stream is interrupted, whatever
# target cells it produced already have their normalization basis on disk.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"

[ $# -eq 3 ] || { echo "usage: run-stream.sh <results_dir> <ccd-range> <targets-csv>" >&2; exit 2; }
RESULTS_DIR="$(cd "$1" && pwd)"
CCD="$2"
IFS=',' read -r -a TARGETS <<< "$3"

CELL_DIR="$RESULTS_DIR/cells"
ANCHOR_DIR="$RESULTS_DIR/anchors/$CCD"
STATE_TSV="$RESULTS_DIR/state.tsv"
mkdir -p "$CELL_DIR" "$ANCHOR_DIR" "$RESULTS_DIR/tsv"

log() { printf '[stream %s %s] %s\n' "$CCD" "$(date '+%H:%M:%S')" "$*" >&2; }
state_row() { printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$STATE_TSV"; }

export XL_STREAM_CCD="$CCD"
# Builds run on this stream's own cores so compilation never perturbs another
# stream's measurements.
export XL_BUILD_CPUSET="$CCD"

# ---------------------------------------------------------------------------
# 1. Per-stream anchor: the pure Rust core on this CCD (normalization basis)
# ---------------------------------------------------------------------------
anchor_tsv="$RESULTS_DIR/tsv/anchor__${CCD}.tsv"
anchor_missing="$(python3 "$SCRIPT_DIR/matrix.py" cells --target rust --backend dynamic_dawg \
    --mode query --missing-from "$ANCHOR_DIR" --tsv-out "$anchor_tsv" --cell-dir "$ANCHOR_DIR")"
if [ "$anchor_missing" -gt 0 ]; then
    log "anchor: $anchor_missing rust cells on this CCD"
    if XL_CELL_DIR="$ANCHOR_DIR" "$SCRIPT_DIR/run-one.sh" cells "$RESULTS_DIR" \
            rust dynamic_dawg query "$anchor_tsv" >/dev/null; then
        state_row "stream-${CCD}-anchor" "ok" ""
    else
        state_row "stream-${CCD}-anchor" "failed" "see failures/"
        log "anchor FAILED — target cells from this stream will lack a normalization basis"
    fi
else
    log "anchor already complete"
fi

# ---------------------------------------------------------------------------
# 2. Assigned targets, strictly serial within the stream
# ---------------------------------------------------------------------------
for target in "${TARGETS[@]}"; do
    backends="$(awk -F'\t' -v t="$target" '!/^#/ && $1 == t { print $4 }' "$XL/targets.tsv")"
    [ -n "$backends" ] || { log "unknown target $target — skipping"; state_row "stream-${CCD}-${target}" "unknown" ""; continue; }
    IFS=',' read -r -a backend_list <<< "$backends"
    for backend in "${backend_list[@]}"; do
        tsv="$RESULTS_DIR/tsv/${target}__${backend}__query__${CCD}.tsv"
        count="$(python3 "$SCRIPT_DIR/matrix.py" cells --target "$target" --backend "$backend" \
            --mode query --missing-from "$CELL_DIR" --tsv-out "$tsv" --cell-dir "$CELL_DIR")"
        if [ "$count" -gt 0 ]; then
            log "$target x $backend: $count query cells"
            if ! "$SCRIPT_DIR/run-one.sh" cells "$RESULTS_DIR" "$target" "$backend" query "$tsv" \
                    >/dev/null; then
                state_row "${target}__${backend}__query" "failed" "stream $CCD"
                log "$target x $backend query batch FAILED (continuing)"
                continue
            fi
        else
            log "$target x $backend: query cells already complete"
        fi

        if [ ! -f "$CELL_DIR/${target}__${backend}__construct__standard__d1__hits.json" ]; then
            reps=10
            [ "$backend" = "double_array_trie" ] && reps=3
            XL_REPS="$reps" "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" "$target" "$backend" \
                construct standard 1 hits >/dev/null \
                || state_row "${target}__${backend}__construct" "failed" "stream $CCD"
        fi
        if [ ! -f "$CELL_DIR/${target}__${backend}__memory__standard__d2__std-d2.json" ]; then
            "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" "$target" "$backend" \
                memory standard 2 std-d2 >/dev/null \
                || state_row "${target}__${backend}__memory" "failed" "stream $CCD"
        fi
    done
    log "$target complete"
done

log "stream complete: ${TARGETS[*]}"
