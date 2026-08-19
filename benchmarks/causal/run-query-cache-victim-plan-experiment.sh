#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "usage: $0 BINARY OUTPUT_CSV [SAMPLES] [CPU] [CAPACITY] [REQUESTS]" >&2
  exit 2
fi

binary=$1
output=$2
samples=${3:-51}
cpu=${4:-3}
capacity=${5:-256}
requests=${6:-4096}
control_env=LIBLEVENSHTEIN_CAUSAL_ALLOCATING_QUERY_CACHE_VICTIM_PLAN
fixed_seed_env=LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
load_ledger="${output%.csv}-host-load.jsonl"

if [[ ! -x $binary ]]; then
  echo "binary is not executable: $binary" >&2
  exit 2
fi
if [[ -e $output || -e $load_ledger ]]; then
  echo "output evidence already exists: $output or $load_ledger" >&2
  exit 2
fi
if ! [[ $samples =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ && $capacity =~ ^[1-9][0-9]*$ && $requests =~ ^[1-9][0-9]*$ ]]; then
  echo "samples, capacity, and requests must be positive integers; CPU must be non-negative" >&2
  exit 2
fi

scratch=$(mktemp -d /tmp/liblev-query-cache-plan.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

run_arm() {
  local arm=$1
  local destination=$2
  if [[ $arm == control ]]; then
    env "$control_env=1" "$fixed_seed_env=1" taskset -c "$cpu" "$binary" \
      --capacity "$capacity" --requests "$requests" >"$destination"
  else
    env -u "$control_env" "$fixed_seed_env=1" taskset -c "$cpu" "$binary" \
      --capacity "$capacity" --requests "$requests" >"$destination"
  fi
}

append_sample() {
  local sample=$1
  local arm=$2
  local source=$3
  jq -r --argjson sample "$sample" --arg arm "$arm" \
    '[
      $sample,
      $arm,
      .elapsed_ns,
      .checksum_u64,
      .hot_retained,
      .resident_entries,
      .resident_weight,
      .hits,
      .misses,
      .admissions,
      .rejections,
      .evictions
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
  'sample,arm,elapsed_ns,checksum_u64,hot_retained,resident_entries,resident_weight,hits,misses,admissions,rejections,evictions' >"$output"

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
