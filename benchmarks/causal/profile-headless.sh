#!/usr/bin/env bash
# Reproducible, GUI-free profiler wrapper for the Java parity causal campaign.

set -euo pipefail

usage() {
    printf 'usage: %s <uprof|heaptrack|perf-stat> <output-dir> [--] <command> [args...]\n' "$0" >&2
    exit 2
}

[[ $# -ge 3 ]] || usage
mode=$1
output_dir=$2
shift 2
if [[ ${1:-} == -- ]]; then
    shift
fi
[[ $# -ge 1 ]] || usage

cpu=${CAUSAL_PROFILE_CPU:-3}
if [[ -e $output_dir ]]; then
    printf 'profile-headless: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

{
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'mode=%s\n' "$mode"
    printf 'cpu=%s\n' "$cpu"
    printf 'kernel=%s\n' "$(uname -srvmo)"
    printf 'cpu_model=%s\n' "$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
    printf 'governor=%s\n' "$(sed -n '1p' "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor" 2>/dev/null || true)"
    printf 'rustc=%s\n' "$(rustc --version 2>/dev/null || true)"
    printf 'java=%s\n' "$(java --version 2>&1 | head -n 1 || true)"
    printf 'uprof=%s\n' "$(AMDuProfCLI --version 2>&1 | head -n 1 || true)"
    printf 'heaptrack=%s\n' "$(heaptrack --version 2>&1 | head -n 1 || true)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
} >"$output_dir/environment.txt"

case $mode in
    uprof)
        command -v AMDuProfCLI >/dev/null
        AMDuProfCLI collect \
            --config hotspots \
            --call-graph-depth 64 \
            --affinity "$cpu" \
            --output-dir "$output_dir/uprof" \
            "$@" >"$output_dir/collect.log" 2>&1
        AMDuProfCLI report \
            --input-dir "$output_dir/uprof" \
            --detail \
            --show-percentage \
            --cutoff 0 \
            --report-output "$output_dir/report.csv" \
            >"$output_dir/report.log" 2>&1
        ;;
    heaptrack)
        command -v heaptrack >/dev/null
        command -v heaptrack_print >/dev/null
        # --record-only is mandatory: without it this distro auto-launches
        # heaptrack_gui after collection. Analysis remains text-only.
        # heaptrack appends its own compression suffix to the output stem.
        taskset --cpu-list "$cpu" heaptrack --record-only \
            --output "$output_dir/heaptrack" \
            "$@" \
            >"$output_dir/program.stdout" 2>"$output_dir/collect.stderr"
        heaptrack_print \
            --file "$output_dir/heaptrack.zst" \
            --print-allocators 1 \
            --print-temporary 1 \
            --print-leaks 0 \
            --peak-limit 40 \
            >"$output_dir/report.txt" 2>"$output_dir/report.stderr"
        ;;
    perf-stat)
        command -v perf >/dev/null
        perf stat -d -d -d \
            --output "$output_dir/report.txt" \
            -- taskset --cpu-list "$cpu" "$@" \
            >"$output_dir/program.stdout" 2>"$output_dir/program.stderr"
        ;;
    *)
        usage
        ;;
esac

printf '%s\n' "$output_dir"
