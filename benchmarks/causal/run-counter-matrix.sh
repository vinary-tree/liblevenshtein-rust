#!/usr/bin/env bash
# Run causal work counters over every generated corpus/order cell.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <corpus-manifest.json> <output-dir>\n' "$0" >&2
    exit 2
fi

manifest=$(realpath "$1")
output_dir=$2
corpus_dir=$(dirname "$manifest")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary=${CAUSAL_QUERY_BINARY:-$repo/target/release/causal_query_profile}
cpu=${CAUSAL_PROFILE_CPU:-3}

if [[ -e $output_dir ]]; then
    printf 'run-counter-matrix: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
if [[ ! -x $binary ]]; then
    printf 'run-counter-matrix: build %s first\n' "$binary" >&2
    exit 2
fi
mkdir -p "$output_dir/cells"

while IFS=$'\t' read -r name domain order dictionary queries; do
    case $domain in byte|unicode|u64) ;; *)
        printf 'run-counter-matrix: unsupported unit domain: %s\n' "$domain" >&2
        exit 2
    esac
    constructor=from_terms
    if [[ $order == sorted ]]; then
        constructor=from_sorted_terms
    fi
    cell="$output_dir/cells/${name}-${order}.json"
    taskset --cpu-list "$cpu" "$binary" \
        --dictionary "$corpus_dir/$dictionary" \
        --queries "$corpus_dir/$queries" \
        --domain "$domain" \
        --constructor "$constructor" \
        --algorithm standard \
        --max-distance 2 \
        --passes 1 >"$cell"
done < <(jq -r '.records[] | [.name, .unit_domain, .order, .dictionary, .queries] | @tsv' "$manifest")

jq -s '{
    schema: "liblevenshtein.java-parity-counter-matrix.v1",
    cells: map({
        domain,
        constructor,
        term_count,
        query_count,
        build_ns,
        query_ns,
        matches,
        checksum_u64,
        construction_work,
        work
    })
}' "$output_dir"/cells/*.json >"$output_dir/summary.json"

cp "$manifest" "$output_dir/corpus-manifest.json"
printf '%s\n' "$output_dir/summary.json"
