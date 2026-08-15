#!/usr/bin/env bash
# Time uninstrumented construction over every generated corpus/order cell.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <corpus-manifest.json> <output-dir>\n' "$0" >&2
    exit 2
fi

manifest=$(realpath "$1")
output_dir=$2
corpus_dir=$(dirname "$manifest")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary=${CAUSAL_CONSTRUCTION_BINARY:-$repo/target/release/causal_construction_bench}
cpu=${CAUSAL_PROFILE_CPU:-3}
warmups=${CAUSAL_CONSTRUCTION_WARMUPS:-5}
reps=${CAUSAL_CONSTRUCTION_REPS:-30}

if [[ -e $output_dir ]]; then
    printf 'run-construction-matrix: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
if [[ ! -x $binary ]]; then
    printf 'run-construction-matrix: build %s first\n' "$binary" >&2
    exit 2
fi
mkdir -p "$output_dir/cells"

while IFS=$'\t' read -r name domain order dictionary; do
    case $domain in byte|unicode|u64) ;; *)
        printf 'run-construction-matrix: unsupported unit domain: %s\n' "$domain" >&2
        exit 2
    esac
    # Shuffled corpora exercise the public unordered bulk constructor. The
    # incremental `stream` path is a different API with online publication
    # semantics and must not stand in for unordered bulk construction.
    constructor=from_terms
    if [[ $order == sorted ]]; then
        constructor=from_sorted_terms
    fi
    taskset --cpu-list "$cpu" "$binary" \
        --dictionary "$corpus_dir/$dictionary" \
        --domain "$domain" \
        --constructor "$constructor" \
        --warmups "$warmups" \
        --reps "$reps" >"$output_dir/cells/${name}-${order}.json"
done < <(jq -r '.records[] | [.name, .unit_domain, .order, .dictionary] | @tsv' "$manifest")

jq -c '{cell:(input_filename | split("/") | last | rtrimstr(".json"))} + .' \
    "$output_dir"/cells/*.json \
    | jq -s '
        def median:
            sort as $s
            | ($s | length) as $n
            | if $n % 2 == 1 then $s[$n / 2 | floor]
              else ($s[$n / 2 - 1] + $s[$n / 2]) / 2 end;
        {
            schema: "liblevenshtein.java-parity-construction-matrix.v1",
            cells: map(. + {
                median_ns: (.samples_ns | median),
                min_ns: (.samples_ns | min),
                max_ns: (.samples_ns | max)
            } | del(.samples_ns))
        }
    ' >"$output_dir/summary.json"

cp "$manifest" "$output_dir/corpus-manifest.json"
printf '%s\n' "$output_dir/summary.json"
