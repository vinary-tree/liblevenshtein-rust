#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 BINARY DICTIONARY QUERIES OUTPUT_CSV CONTROL_ENV [SAMPLES] [PASSES] [CPU]" >&2
  exit 2
fi

binary=$1
dictionary=$2
queries=$3
output=$4
control_env=$5
samples=${6:-51}
passes=${7:-1}
cpu=${8:-3}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
load_ledger="${output%.csv}-host-load.jsonl"

if [[ ! -x $binary ]]; then
  echo "binary is not executable: $binary" >&2
  exit 2
fi
if [[ ! -f $dictionary || ! -f $queries ]]; then
  echo "dictionary or query workload does not exist" >&2
  exit 2
fi
if [[ -e $output || -e $load_ledger ]]; then
  echo "output evidence already exists: $output or $load_ledger" >&2
  exit 2
fi
if [[ -z $control_env || $control_env == *=* ]]; then
  echo "CONTROL_ENV must be one non-empty environment-variable name" >&2
  exit 2
fi
if ! [[ $samples =~ ^[1-9][0-9]*$ && $passes =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ ]]; then
  echo "samples and passes must be positive integers; CPU must be non-negative" >&2
  exit 2
fi

binary_sha=$(sha256sum "$binary" | awk '{print $1}')
scratch=$(mktemp -d /tmp/liblev-resource-env-experiment.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

run_arm() {
  local arm=$1
  local destination=$2
  local -a command=(
    taskset -c "$cpu" "$binary"
    --dictionary "$dictionary"
    --queries "$queries"
    --max-distance 2
    --batch-size 256
    --passes "$passes"
    --constructor batch
    --backend dynamic-dawg-unicode
    --algorithm standard
  )
  if [[ $arm == control ]]; then
    env "$control_env=1" "${command[@]}" >"$destination"
  else
    env -u "$control_env" "${command[@]}" >"$destination"
  fi
}

append_sample() {
  local sample=$1
  local arm=$2
  local source=$3
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
      .nonempty_batches,
      .consumer_work.dictionary_intersections,
      .consumer_work.edges_enumerated,
      .consumer_work.transition_attempts,
      .consumer_work.transition_accepted,
      .consumer_work.packed_dfa_transition_hits,
      .consumer_work.packed_dfa_transition_misses,
      .consumer_work.packed_dfa_physical_target_probes,
      .provider_work.arena_locks,
      .provider_work.graph_projections,
      .provider_work.graph_calls,
      .provider_work.value_calls
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
  'sample,arm,binary_sha256,query_ns,matches,term_bytes,distance_sum,checksum_u64,order_checksum_u64,nonempty_batches,dictionary_intersections,edges_enumerated,transition_attempts,transition_accepted,packed_dfa_transition_hits,packed_dfa_transition_misses,packed_dfa_physical_target_probes,arena_locks,graph_projections,graph_calls,value_calls' >"$output"

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
