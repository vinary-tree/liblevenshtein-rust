#!/usr/bin/env bash
# Run one timed command while continuously classifying competing benchmark
# processes. The monitor is evidence-bearing: any protocol violation, foreign
# harness, or ledger-write failure makes the command unusable even when the
# benchmark process itself exits successfully.
set -uo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: run-with-contention-monitor.sh <results-dir> -- <command> [args...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$1"
shift 2
mkdir -p "$RESULTS_DIR"

LOG="${XL_CONTENTION_LOG:-$RESULTS_DIR/contention-monitor.log}"
STAMP="$(date -u +%Y%m%dT%H%M%S).$$"
OBSERVATIONS="$RESULTS_DIR/.contention-observations.$STAMP"
ONLINE_CPUS="$(cat /sys/devices/system/cpu/online 2>/dev/null || echo 0)"
MONITOR_CPU="${XL_MONITOR_CPU:-${ONLINE_CPUS##*[,-]}}"

"$@" &
CHILD_PID=$!

# Invoked indirectly by the signal trap below.
# shellcheck disable=SC2329
forward_signal() {
    kill -TERM "$CHILD_PID" 2>/dev/null || true
}
trap forward_signal HUP INT TERM

monitor() {
    # Keep the observer itself off the benchmark cpuset. Children inherit this
    # affinity, so the guard and its one-second timer cannot migrate onto a
    # measured core and become the interference they are meant to detect.
    taskset -pc "$MONITOR_CPU" "$BASHPID" >/dev/null
    while kill -0 "$CHILD_PID" 2>/dev/null; do
        XL_TIMED_CHILD_PID="$CHILD_PID" \
            "$SCRIPT_DIR/timed-proc-guard.sh" "$RESULTS_DIR" >>"$OBSERVATIONS" 2>&1
        sleep 1
    done
    # Close any interval whose process ended during the last sampling period.
    XL_TIMED_CHILD_PID="$CHILD_PID" \
        "$SCRIPT_DIR/timed-proc-guard.sh" "$RESULTS_DIR" >>"$OBSERVATIONS" 2>&1
}

monitor &
MONITOR_PID=$!
wait "$CHILD_PID"
CHILD_STATUS=$?
wait "$MONITOR_PID"
MONITOR_STATUS=$?
trap - HUP INT TERM

if [ -s "$OBSERVATIONS" ]; then
    {
        printf '[%s] command:' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf ' %q' "$@"
        printf '\n'
        cat "$OBSERVATIONS"
    } >>"$LOG"
fi

if [ "$MONITOR_STATUS" -ne 0 ]; then
    echo "contention monitor failed; timed result is invalid (see $LOG)" >&2
    rm -f "$OBSERVATIONS"
    exit 74
fi
if grep -Eq 'PROTOCOL-VIOLATION|FOREIGN-CONTENTION|GUARD-ERROR' "$OBSERVATIONS"; then
    echo "contention monitor rejected timed result (see $LOG)" >&2
    rm -f "$OBSERVATIONS"
    exit 75
fi
rm -f "$OBSERVATIONS"
exit "$CHILD_STATUS"
