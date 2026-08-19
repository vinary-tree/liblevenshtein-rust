#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "usage: $0 BINARY DICTIONARY OUTPUT_CSV [SAMPLES] [ITERATIONS] [CPU] [PATTERN]" >&2
  exit 2
fi

binary=$1
dictionary=$2
output=$3
samples=${4:-51}
iterations=${5:-1000000}
cpu=${6:-3}
pattern=${7:-(ph|f)one}
retention_env=LIBLEVENSHTEIN_CAUSAL_RETAIN_LEGACY_PHONETIC_STATE
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
if [[ ! -f $dictionary ]]; then
  echo "dictionary does not exist: $dictionary" >&2
  exit 2
fi
if ! [[ $samples =~ ^[1-9][0-9]*$ && $iterations =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ ]]; then
  echo "samples and iterations must be positive integers; CPU must be a non-negative integer" >&2
  exit 2
fi

scratch=$(mktemp -d /tmp/liblev-phonetic-retention.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

run_arm() {
  local arm=$1
  local destination=$2
  if [[ $arm == control ]]; then
    env "$retention_env=1" taskset -c "$cpu" "$binary" \
      --dictionary "$dictionary" \
      --pattern "$pattern" \
      --max-distance 1 \
      --iterations "$iterations" \
      --workload construct-drop >"$destination"
  else
    env -u "$retention_env" taskset -c "$cpu" "$binary" \
      --dictionary "$dictionary" \
      --pattern "$pattern" \
      --max-distance 1 \
      --iterations "$iterations" \
      --workload construct-drop >"$destination"
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
      .iterator_inline_bytes,
      .retained_inline_bytes,
      .first_gate_matches,
      .first_gate_term_bytes,
      .first_gate_distance_sum,
      .first_gate_checksum_u64,
      .first_gate_order_checksum_u64,
      .full_gate_matches,
      .full_gate_term_bytes,
      .full_gate_distance_sum,
      .full_gate_checksum_u64,
      .full_gate_order_checksum_u64
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
  'sample,arm,construct_drop_ns,iterator_inline_bytes,retained_inline_bytes,first_gate_matches,first_gate_term_bytes,first_gate_distance_sum,first_gate_checksum_u64,first_gate_order_checksum_u64,full_gate_matches,full_gate_term_bytes,full_gate_distance_sum,full_gate_checksum_u64,full_gate_order_checksum_u64' >"$output"

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
