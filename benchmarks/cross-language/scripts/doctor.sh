#!/usr/bin/env bash
# Readiness probe: toolchains, native artifacts, environment, and a one-query
# smoke per requested target. Emits doctor-report.json (to --report PATH or
# .stage/doctor-report.json) and exits nonzero if any REQUESTED target is
# unready — run-all refuses to time against a red doctor.
#
# Usage: doctor.sh [--targets a,b,c] [--report PATH] [--skip-governor]
#
# --skip-governor exists for authoring iterations only; run-all never passes
# it, so timing always runs under the performance governor.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"
LR="$(cd "$XL/../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"

TARGETS="rust,c"
REPORT="$XL/.stage/doctor-report.json"
SKIP_GOVERNOR=0
while [ $# -gt 0 ]; do
    case "$1" in
        --targets) TARGETS="$2"; shift 2 ;;
        --report) REPORT="$2"; shift 2 ;;
        --skip-governor) SKIP_GOVERNOR=1; shift ;;
        *) echo "doctor: unknown flag $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$(dirname "$REPORT")"

declare -a CHECKS=()   # each entry: name<TAB>status<TAB>detail
FAILURES=0

check() { # name status detail
    CHECKS+=("$1	$2	$3")
    if [ "$2" = "fail" ]; then
        FAILURES=$((FAILURES + 1))
        echo "doctor: FAIL $1: $3" >&2
    else
        echo "doctor: $2  $1: $3" >&2
    fi
}

probe_tool() { # name command...
    local name="$1"; shift
    local out
    if out="$("$@" 2>&1 | head -1)"; then
        check "tool:$name" "ok" "$out"
        return 0
    else
        check "tool:$name" "fail" "not runnable: $*"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Global environment
# ---------------------------------------------------------------------------

if [ "$SKIP_GOVERNOR" -eq 0 ]; then
    BAD_GOVERNORS=""
    for cpu in 2 3 4 5 6 7 8 9; do
        gov="$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || echo missing)"
        [ "$gov" = "performance" ] || BAD_GOVERNORS="$BAD_GOVERNORS cpu$cpu=$gov"
    done
    if [ -n "$BAD_GOVERNORS" ]; then
        check "governor" "fail" "not performance on:$BAD_GOVERNORS (sudo cpupower frequency-set -g performance)"
    else
        check "governor" "ok" "performance on cpus 2-9"
    fi
else
    check "governor" "skipped" "--skip-governor (authoring only; never for timing)"
fi

SIBLINGS="$(cat /sys/devices/system/cpu/cpu2/topology/thread_siblings_list 2>/dev/null || echo unknown)"
check "smt-sibling-map" "ok" "cpu2 siblings: $SIBLINGS (runner never schedules the sibling; keep it idle during timing)"

AVAIL_GIB="$(df --output=avail -BG "$XL" | tail -1 | tr -dc '0-9')"
if [ "${AVAIL_GIB:-0}" -lt 20 ]; then
    check "disk" "fail" "only ${AVAIL_GIB}G free (< 20G floor)"
else
    check "disk" "ok" "${AVAIL_GIB}G free"
fi

for artifact in "$LR/target/release/libliblevenshtein.so" "$LD_REPO/target/release/liblibdictenstein.so"; do
    if [ -f "$artifact" ]; then
        check "artifact:$(basename "$artifact")" "ok" "$artifact"
    else
        check "artifact:$(basename "$artifact")" "fail" "missing (build with RUSTFLAGS='-C target-cpu=native' cargo build --release --features …)"
    fi
done

if python3 "$XL/workload/generate_workload.py" --verify >/dev/null 2>&1; then
    check "workload" "ok" "artifacts match provenance.json"
else
    check "workload" "fail" "workload drift or missing (run generate_workload.py, review provenance)"
fi

# ---------------------------------------------------------------------------
# Per-target readiness
# ---------------------------------------------------------------------------

target_tools_ok() { # target -> 0/1 via required tool probes
    case "$1" in
        rust) probe_tool cargo cargo --version ;;
        c) probe_tool cc cc --version ;;
        cpp) probe_tool c++ c++ --version ;;
        jvm-vinary|jvm-legacy)
            probe_tool java java --version || return 1
            local major
            major="$(java --version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1)"
            if [ "${major:-0}" -lt 22 ]; then
                check "tool:java>=22" "fail" "FFM needs JDK >= 22, found $major"
                return 1
            fi
            [ -x "$LR/bindings/jvm/gradlew" ] \
                && check "tool:gradlew" "ok" "$LR/bindings/jvm/gradlew" \
                || { check "tool:gradlew" "fail" "missing bindings/jvm/gradlew"; return 1; }
            ;;
        js-native|js-legacy) probe_tool node node --version && probe_tool npm npm --version ;;
        js-wasm) probe_tool node node --version && probe_tool wasm-pack wasm-pack --version ;;
        js-wasi) probe_tool node node --version && probe_tool wasm-opt wasm-opt --version ;;
        python) probe_tool python3 python3 --version ;;
        clojure) probe_tool clojure clojure --version ;;
        clojurescript) probe_tool clojure clojure --version && probe_tool node node --version ;;
        dotnet) probe_tool dotnet dotnet --version ;;
        go) probe_tool go go version ;;
        swift) probe_tool swift swift --version ;;
        ruby) probe_tool ruby ruby --version ;;
        fortran) probe_tool gfortran gfortran --version && probe_tool fpm fpm --version ;;
        ocaml) probe_tool opam opam --version && probe_tool dune dune --version ;;
        haskell) probe_tool ghc ghc --version && probe_tool cabal cabal --version \
            && probe_tool pkg-config pkg-config --version ;;
        lua) probe_tool lua5.4 lua5.4 -v && probe_tool cc cc --version ;;
        *) check "target:$1" "fail" "unknown target"; return 1 ;;
    esac
}

SMOKE_DIR="$XL/.stage/doctor-smoke"
rm -rf "$SMOKE_DIR" && mkdir -p "$SMOKE_DIR"

smoke_target() { # target -> one-query verify through the real runner
    local target="$1" backend
    backend="$(awk -F'\t' -v t="$target" '!/^#/ && $1 == t { split($4, b, ","); print b[1] }' "$XL/targets.tsv")"
    if [ -z "$backend" ]; then
        check "smoke:$target" "fail" "no backends in targets.tsv"
        return 1
    fi
    if XL_GATE_LIMIT=1 XL_CELL_DIR="$SMOKE_DIR" \
        "$SCRIPT_DIR/run-one.sh" cell "$SMOKE_DIR" "$target" "$backend" verify standard 1 hits \
        >/dev/null 2>"$SMOKE_DIR/$target.smoke.log"; then
        check "smoke:$target" "ok" "1-query verify via $backend"
    else
        check "smoke:$target" "fail" "see $SMOKE_DIR/$target.smoke.log"
        return 1
    fi
}

IFS=',' read -r -a TARGET_LIST <<< "$TARGETS"
for target in "${TARGET_LIST[@]}"; do
    if target_tools_ok "$target"; then
        smoke_target "$target" || true
    fi
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

python3 - "$REPORT" "$TARGETS" "${CHECKS[@]}" <<'PYEOF'
import json, sys
from datetime import datetime, timezone
report_path, targets, *checks = sys.argv[1:]
rows = []
failures = 0
for check in checks:
    name, status, detail = check.split("\t", 2)
    rows.append({"name": name, "status": status, "detail": detail})
    failures += status == "fail"
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "targets": targets.split(","),
    "checks": rows,
    "failures": failures,
    "verdict": "green" if failures == 0 else "red",
}
with open(report_path, "w") as fh:
    json.dump(report, fh, indent=2)
    fh.write("\n")
print(f"doctor: {report['verdict']} ({failures} failure(s)) -> {report_path}", file=sys.stderr)
PYEOF

exit $((FAILURES > 0 ? 1 : 0))
