#!/usr/bin/env bash
# Compile and run the identity graph census against the published Java jar.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <sorted-dictionary> <output-dir>\n' "$0" >&2
    exit 2
fi

dictionary=$(realpath "$1")
output_dir=$2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
classpath_file="$repo/benchmarks/cross-language/harnesses/jvm/legacy/build/runtime-classpath.txt"
cpu=${CAUSAL_PROFILE_CPU:-3}
if [[ -e $output_dir ]]; then
    printf 'run-legacy-structure-probe: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
if [[ ! -f $classpath_file ]]; then
    printf 'run-legacy-structure-probe: stage the JVM legacy harness first\n' >&2
    exit 2
fi
mkdir -p "$output_dir/classes"
classpath=$(<"$classpath_file")
javac -cp "$classpath" \
    -d "$output_dir/classes" \
    "$repo/benchmarks/causal/LegacyStructureProbe.java"
taskset --cpu-list "$cpu" java \
    -cp "$output_dir/classes:$classpath" \
    LegacyStructureProbe "$dictionary" >"$output_dir/result.json"
printf '%s\n' "$output_dir/result.json"
