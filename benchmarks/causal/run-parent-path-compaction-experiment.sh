#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 CONTROL_BINARY TREATMENT_BINARY DICTIONARY QUERIES OUTPUT_CSV [SAMPLES] [PASSES] [CPU]" >&2
  exit 2
fi

control_binary=$1
treatment_binary=$2
dictionary=$3
queries=$4
output=$5
samples=${6:-51}
passes=${7:-1}
cpu=${8:-3}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
load_ledger="${output%.csv}-host-load.jsonl"

for binary in "$control_binary" "$treatment_binary"; do
  if [[ ! -x $binary ]]; then
    echo "binary is not executable: $binary" >&2
    exit 2
  fi
done
if [[ ! -f $dictionary || ! -f $queries ]]; then
  echo "dictionary or query workload does not exist" >&2
  exit 2
fi
if [[ -e $output || -e $load_ledger ]]; then
  echo "output evidence already exists: $output or $load_ledger" >&2
  exit 2
fi
if ! [[ $samples =~ ^[1-9][0-9]*$ && $passes =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ ]]; then
  echo "samples and passes must be positive integers; CPU must be non-negative" >&2
  exit 2
fi

control_sha=$(sha256sum "$control_binary" | awk '{print $1}')
treatment_sha=$(sha256sum "$treatment_binary" | awk '{print $1}')
scratch=$(mktemp -d /tmp/liblev-parent-path-compaction.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

run_arm() {
  local arm=$1
  local destination=$2
  local binary=$treatment_binary
  if [[ $arm == control ]]; then
    binary=$control_binary
  fi
  taskset -c "$cpu" "$binary" \
    --dictionary "$dictionary" \
    --queries "$queries" \
    --max-distance 2 \
    --batch-size 256 \
    --passes "$passes" \
    --constructor batch \
    --backend dynamic-dawg-unicode \
    --algorithm standard >"$destination"
}

append_sample() {
  local sample=$1
  local arm=$2
  local source=$3
  local binary_sha=$treatment_sha
  if [[ $arm == control ]]; then
    binary_sha=$control_sha
  fi
  jq -r --argjson sample "$sample" --arg arm "$arm" --arg binary_sha "$binary_sha" \
    '[
      $sample,
      $arm,
      $binary_sha,
      .query_ns,
      .matches,
      .term_bytes,
      .distance_sum,
      .checksum_u64,
      .order_checksum_u64,
      .nonempty_batches
    ] | @csv' "$source" >>"$output"
}

admit_pair() {
  local label=$1
  "$host_gate" --cpu "$cpu" --label "$label" --output "$load_ledger" >/dev/null
}

admit_pair warmup
for arm in control treatment; do
  for warmup in 1 2 3; do
    run_arm "$arm" "$scratch/warmup-$arm-$warmup.json"
  done
done

printf '%s\n' \
  'sample,arm,binary_sha256,query_ns,matches,term_bytes,distance_sum,checksum_u64,order_checksum_u64,nonempty_batches' >"$output"

for ((sample = 1; sample <= samples; sample++)); do
  admit_pair "pair-$sample"
  if ((sample % 2 == 1)); then
    arms=(control treatment)
  else
    arms=(treatment control)
  fi
  for arm in "${arms[@]}"; do
    result="$scratch/$sample-$arm.json"
    run_arm "$arm" "$result"
    append_sample "$sample" "$arm" "$result"
  done
  echo "pair $sample/$samples complete" >&2
done
