#!/usr/bin/env bash
# Compile a measurement-only classpath shadow and run the published Java query.

set -euo pipefail

if [[ $# -ne 5 ]]; then
    printf 'usage: %s <sorted-dictionary> <queries> <algorithm> <distance> <output-dir>\n' "$0" >&2
    exit 2
fi

dictionary=$(realpath "$1")
queries=$(realpath "$2")
algorithm=$3
distance=$4
output_dir=$5
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
classpath_file="$repo/benchmarks/cross-language/harnesses/jvm/legacy/build/runtime-classpath.txt"
cpu=${CAUSAL_PROFILE_CPU:-3}

if [[ -e $output_dir ]]; then
    printf 'run-legacy-subsumption-probe: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
if [[ ! -f $classpath_file ]]; then
    printf 'run-legacy-subsumption-probe: stage the JVM legacy harness first\n' >&2
    exit 2
fi

mkdir -p "$output_dir/classes"
classpath=$(<"$classpath_file")
javac -cp "$classpath" \
    -d "$output_dir/classes" \
    "$repo/benchmarks/causal/java-shadow/com/github/liblevenshtein/transducer/UnsubsumeFunction.java" \
    "$repo/benchmarks/causal/LegacySubsumptionProbe.java" \
    >"$output_dir/compile.log" 2>&1
taskset --cpu-list "$cpu" java \
    -cp "$output_dir/classes:$classpath" \
    LegacySubsumptionProbe "$dictionary" "$queries" "$algorithm" "$distance" \
    >"$output_dir/result.json" 2>"$output_dir/run.log"
printf '%s\n' "$output_dir/result.json"
