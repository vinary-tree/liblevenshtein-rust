#!/usr/bin/env bash
# Orchestrate the full cross-language benchmark matrix.
#
# Usage:
#   run-all.sh [--results DIR] [--targets a,b,c|all] [--phase N]
#              [--quick] [--samples N] [--no-aggregate]
#              [--allow-gate-failure CELL=EXPLANATION]...
#
# Flow: environment capture -> correctness gate (hard precondition) ->
# per-target timed cells (query batches, construct, memory) in targets.tsv
# phase order -> drift sentinels every ~90 min -> state accounting ->
# aggregate. Every cell that exists and validates is skipped on resume, so
# interrupting this script at any point is free.
#
# A target whose build or gate fails lands in failures/ and state.tsv as
# build_failed/gate_failed — the run continues with the remaining targets,
# and the aggregate renders the failure rows. Nothing is silently skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR=""
TARGETS="all"
PHASE=""
QUICK=0
SAMPLES=""
NO_AGGREGATE=0
declare -a ALLOWED_FAILURES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --results) RESULTS_DIR="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --no-aggregate) NO_AGGREGATE=1; shift ;;
        --allow-gate-failure) ALLOWED_FAILURES+=("$2"); shift 2 ;;
        *) echo "run-all: unknown flag $1" >&2; exit 2 ;;
    esac
done

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="$XL/results/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
CELL_DIR="$RESULTS_DIR/cells"
mkdir -p "$CELL_DIR" "$RESULTS_DIR/sentinels" "$RESULTS_DIR/tsv"
STATE_TSV="$RESULTS_DIR/state.tsv"

log() { printf '[run-all %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

state_row() { # cell_or_scope status detail
    printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$STATE_TSV"
}

# ---------------------------------------------------------------------------
# Target selection, phase-ordered
# ---------------------------------------------------------------------------

manifest_targets() {
    awk -F'\t' '!/^#/ && $1 != "target" && NF >= 7 { print $7 "\t" $1 }' "$XL/targets.tsv" \
        | sort -n | cut -f2
}

declare -a RUN_TARGETS=()
if [ "$TARGETS" = "all" ]; then
    while IFS= read -r t; do RUN_TARGETS+=("$t"); done < <(manifest_targets)
else
    IFS=',' read -r -a RUN_TARGETS <<< "$TARGETS"
fi

if [ -n "$PHASE" ]; then
    declare -a FILTERED=()
    for t in "${RUN_TARGETS[@]}"; do
        p="$(awk -F'\t' -v t="$t" '!/^#/ && $1 == t { print $7 }' "$XL/targets.tsv")"
        [ "$p" = "$PHASE" ] && FILTERED+=("$t")
    done
    RUN_TARGETS=("${FILTERED[@]}")
fi
[ ${#RUN_TARGETS[@]} -gt 0 ] || { echo "run-all: no targets selected" >&2; exit 2; }

target_field() { # target column-name
    awk -F'\t' -v target="$1" -v col="$2" '
        !/^#/ && $1 == "target" { for (i = 1; i <= NF; i++) idx[$i] = i; next }
        !/^#/ && $1 == target { print $(idx[col]) }
    ' "$XL/targets.tsv"
}

log "targets (phase order): ${RUN_TARGETS[*]}"
log "results: $RESULTS_DIR"

# ---------------------------------------------------------------------------
# 1. Environment capture + doctor (readiness is a hard precondition)
# ---------------------------------------------------------------------------

python3 "$SCRIPT_DIR/env-capture.py" "$RESULTS_DIR"

if ! bash "$SCRIPT_DIR/doctor.sh" --targets "$(IFS=,; echo "${RUN_TARGETS[*]}")" \
    --report "$RESULTS_DIR/doctor-report.json"; then
    state_row "doctor" "red" "see doctor-report.json"
    echo "run-all: doctor is red for requested targets; refusing to time." >&2
    exit 1
fi
state_row "doctor" "green" ""

# ---------------------------------------------------------------------------
# 2. Correctness gate (hard precondition for timing)
# ---------------------------------------------------------------------------

if [ ${#ALLOWED_FAILURES[@]} -gt 0 ]; then
    python3 - "$RESULTS_DIR/gate-report.json" "${ALLOWED_FAILURES[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text()) if report_path.exists() else {}
documented = report.get("documented_mismatches", [])
for spec in sys.argv[2:]:
    cell, _, explanation = spec.partition("=")
    if not explanation:
        raise SystemExit(f"--allow-gate-failure needs CELL=EXPLANATION, got {spec!r}")
    if not any(d.get("cell") == cell for d in documented):
        documented.append({"cell": cell, "explanation": explanation,
                           "accepted_via": "run-all --allow-gate-failure"})
report["documented_mismatches"] = documented
report_path.write_text(json.dumps(report, indent=2) + "\n")
PYEOF
fi

GATE_TARGETS="$(IFS=,; echo "${RUN_TARGETS[*]}")"
if ! python3 "$SCRIPT_DIR/gate.py" "$RESULTS_DIR" --targets "$GATE_TARGETS"; then
    state_row "gate" "gate_failed" "$GATE_TARGETS"
    echo "run-all: correctness gate is not green; timing refused." >&2
    echo "         Investigate gate-report.json; document a genuine legacy delta" >&2
    echo "         with --allow-gate-failure 'CELL=explanation' to proceed." >&2
    exit 1
fi
state_row "gate" "green" "$GATE_TARGETS"

# ---------------------------------------------------------------------------
# 3. Drift sentinels: 3 fixed rust cells, re-run every ~90 minutes
# ---------------------------------------------------------------------------

SENTINEL_INTERVAL_SECONDS=5400
LAST_SENTINEL=0

run_sentinels() {
    local stamp epoch
    epoch="$(date +%s)"
    stamp="$(date +%Y%m%dT%H%M%S)"
    log "sentinel run @$stamp"
    local spec
    for spec in "query standard 2 std-d2" "query transposition 3 tr-d3" "query standard 1 hits"; do
        # shellcheck disable=SC2086
        set -- $spec
        XL_CELL_DIR="$RESULTS_DIR/sentinels" XL_SAMPLES=10 \
            "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" rust dynamic_dawg "$1" "$2" "$3" "$4" \
            >/dev/null
        mv "$RESULTS_DIR/sentinels/rust__dynamic_dawg__$1__$2__d$3__$4.json" \
           "$RESULTS_DIR/sentinels/${stamp}__rust__$2__d$3__$4.json"
    done
    LAST_SENTINEL="$epoch"
    python3 - "$RESULTS_DIR/sentinels" <<'PYEOF'
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
groups = defaultdict(list)
for path in sorted(Path(sys.argv[1]).glob("*__rust__*.json")):
    stamp, _, rest = path.name.partition("__rust__")
    data = json.loads(path.read_text())
    med = statistics.median(data["measurements"]["samples_ns"])
    groups[rest].append((stamp, med))
for rest, series in groups.items():
    if len(series) < 2:
        continue
    first = series[0][1]
    last = series[-1][1]
    shift = abs(last - first) / first * 100.0
    marker = "DRIFT-WARNING" if shift > 5.0 else "drift-ok"
    print(f"[sentinel] {marker} {rest}: {shift:.2f}% (first {first:.0f}ns, last {last:.0f}ns)",
          file=sys.stderr)
PYEOF
}

maybe_sentinels() {
    local now
    now="$(date +%s)"
    if [ $((now - LAST_SENTINEL)) -ge $SENTINEL_INTERVAL_SECONDS ]; then
        run_sentinels
    fi
}

# ---------------------------------------------------------------------------
# 4. Timed cells per target×backend
# ---------------------------------------------------------------------------

QUICK_FLAG=()
if [ "$QUICK" -eq 1 ]; then QUICK_FLAG=(--quick); fi

run_target() {
    local target="$1" backend mode
    local backends
    backends="$(target_field "$target" backends)"
    IFS=',' read -r -a backend_list <<< "$backends"

    # The Java pair's query timing belongs to JMH (forks, managed warmup);
    # the driver is idempotent and covers BOTH jvm targets' missing cells,
    # so the second target's call is a fast no-op. Construct/memory cells
    # still flow through the generic per-backend path below.
    if [ "$target" = "jvm-vinary" ] || [ "$target" = "jvm-legacy" ]; then
        log "$target: query cells via JMH driver"
        if ! bash "$SCRIPT_DIR/run-jvm-pair.sh" jmh "$RESULTS_DIR" "${QUICK_FLAG[@]}"; then
            state_row "${target}__jmh__query" "failed" "see logs/jmh-matrix-driver.log"
            log "$target JMH query matrix FAILED (continuing with other targets)"
        fi
    fi

    for backend in "${backend_list[@]}"; do
        # -- query cells (batched; resume = only missing cells enter the TSV)
        if [ "$target" != "jvm-vinary" ] && [ "$target" != "jvm-legacy" ]; then
            local tsv="$RESULTS_DIR/tsv/${target}__${backend}__query.tsv"
            local count
            count="$(python3 "$SCRIPT_DIR/matrix.py" cells --target "$target" --backend "$backend" \
                --mode query --missing-from "$CELL_DIR" --tsv-out "$tsv" --cell-dir "$CELL_DIR" \
                "${QUICK_FLAG[@]}")"
            if [ "$count" -gt 0 ]; then
                log "$target×$backend: $count query cells"
                if XL_SAMPLES="${SAMPLES}" "$SCRIPT_DIR/run-one.sh" cells "$RESULTS_DIR" \
                    "$target" "$backend" query "$tsv" >/dev/null; then
                    :
                else
                    state_row "${target}__${backend}__query" "failed" "see failures/"
                    log "$target×$backend query batch FAILED (continuing with other targets)"
                    continue
                fi
            else
                log "$target×$backend: query cells already complete"
            fi
        fi
        # -- construct cell
        if [ ! -f "$CELL_DIR/${target}__${backend}__construct__standard__d1__hits.json" ]; then
            local reps=10
            [ "$backend" = "double_array_trie" ] && reps=3
            log "$target×$backend: construct (reps=$reps)"
            if ! XL_REPS="$reps" "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" \
                "$target" "$backend" construct standard 1 hits >/dev/null; then
                state_row "${target}__${backend}__construct" "failed" "see failures/"
                log "$target×$backend construct FAILED (continuing)"
            fi
        fi
        # -- memory cell
        if [ ! -f "$CELL_DIR/${target}__${backend}__memory__standard__d2__std-d2.json" ]; then
            log "$target×$backend: memory"
            if ! "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" \
                "$target" "$backend" memory standard 2 std-d2 >/dev/null; then
                state_row "${target}__${backend}__memory" "failed" "see failures/"
                log "$target×$backend memory FAILED (continuing)"
            fi
        fi
        maybe_sentinels
    done
}

run_sentinels   # baseline sentinel before any timing
for target in "${RUN_TARGETS[@]}"; do
    run_target "$target"
done
run_sentinels   # closing sentinel

# ---------------------------------------------------------------------------
# 5. State accounting: every expected cell -> ok|degraded|failed|missing
# ---------------------------------------------------------------------------

python3 - "$RESULTS_DIR" "$CELL_DIR" "$GATE_TARGETS" "$SCRIPT_DIR" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
results_dir, cell_dir, targets, scripts = (Path(sys.argv[1]), Path(sys.argv[2]),
                                           sys.argv[3], Path(sys.argv[4]))
expected = subprocess.run(
    [sys.executable, str(scripts / "matrix.py"), "expected", "--targets", targets],
    capture_output=True, text=True, check=True).stdout.splitlines()
rows = []
for line in expected:
    target, backend, mode, algorithm, distance, queryset = line.split("\t")
    name = f"{target}__{backend}__{mode}__{algorithm}__d{distance}__{queryset}.json"
    path = cell_dir / name
    if path.exists():
        status = json.loads(path.read_text()).get("status", "unknown")
    elif (results_dir / "failures" / f"{target}__{backend}.build_failed").exists():
        status = "build_failed"
    else:
        status = "missing"
    rows.append((name, status))
with (results_dir / "state.tsv").open("a") as fh:
    for name, status in rows:
        fh.write(f"{name}\t{status}\t\n")
counts = {}
for _, status in rows:
    counts[status] = counts.get(status, 0) + 1
print(f"[state] {counts}", file=sys.stderr)
PYEOF

# ---------------------------------------------------------------------------
# 6. Aggregate
# ---------------------------------------------------------------------------

if [ "$NO_AGGREGATE" -eq 0 ]; then
    python3 "$SCRIPT_DIR/aggregate.py" "$RESULTS_DIR"
fi

# ---------------------------------------------------------------------------
# 7. Publish raw cells to pgmcp (the system of record for benchmark data).
#    Content-addressed and idempotent, so this is safe to re-run; a failure
#    here is reported but never discards local results.
# ---------------------------------------------------------------------------

if python3 "$SCRIPT_DIR/pgmcp-upload.py" "$RESULTS_DIR"; then
    state_row "pgmcp-upload" "ok" ""
else
    state_row "pgmcp-upload" "failed" "re-run scripts/pgmcp-upload.py $RESULTS_DIR"
    log "pgmcp upload FAILED — local cells intact; re-run the uploader"
fi
log "done: $RESULTS_DIR"
