#!/usr/bin/env bash
set -euo pipefail

resume=0
if [[ $# -gt 0 && ${!#} == --resume ]]; then
  resume=1
  set -- "${@:1:$(($# - 1))}"
fi
if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 BINARY OUTPUT_CSV CONTROL_ARM TREATMENT_ARM ENTRIES [SAMPLES] [PASSES] [CPU] [--resume]" >&2
  exit 2
fi

binary=$1
output=$2
control_arm=$3
treatment_arm=$4
entries=$5
samples=${6:-51}
passes=${7:-8}
cpu=${8:-3}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
resume_helper="$repo/benchmarks/causal/prepare-collection-traversal-resume.py"
load_ledger="${output%.csv}-host-load.jsonl"
rejected_ledger="${output%.csv}-rejected-host-load.jsonl"

if [[ ! -x $binary ]]; then
  echo "binary is not executable: $binary" >&2
  exit 2
fi
if [[ $control_arm == "$treatment_arm" ]]; then
  echo "control and treatment arms must differ" >&2
  exit 2
fi
if ((resume == 0)) && [[ -e $output || -e $load_ledger ]]; then
  echo "output evidence already exists: $output or $load_ledger" >&2
  exit 2
fi
if ((resume == 1)) && [[ ! -f $output || ! -f $load_ledger ]]; then
  echo "resume requires existing sample CSV and host-load ledger" >&2
  exit 2
fi
if ! [[ $entries =~ ^[1-9][0-9]*$ && $samples =~ ^[1-9][0-9]*$ && $passes =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ ]]; then
  echo "entries, samples, and passes must be positive integers; CPU must be non-negative" >&2
  exit 2
fi

output_directory=$(dirname "$output")
mkdir -p "$output_directory"
binary_sha=$(sha256sum "$binary" | awk '{print $1}')
scratch=$(mktemp -d /tmp/libdictenstein-collection-experiment.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

run_arm() {
  local arm=$1
  local destination=$2
  taskset -c "$cpu" "$binary" \
    --arm "$arm" \
    --entries "$entries" \
    --passes "$passes" >"$destination"
  jq -e --arg arm "$arm" --argjson entries "$entries" \
    '.schema == "libdictenstein.collection-traversal.v1" and .arm == $arm and .dictionary_entries == $entries' \
    "$destination" >/dev/null
}

append_sample() {
  local sample=$1
  local role=$2
  local source=$3
  jq -r \
    --argjson sample "$sample" \
    --arg role "$role" \
    --arg binary_sha "$binary_sha" \
    '[
      $sample,
      $role,
      .arm,
      $binary_sha,
      .dictionary_entries,
      .consumed_entries_per_pass,
      .passes,
      .elapsed_ns,
      .checksum,
      .boundary_calls
    ] | @csv' "$source" >>"$output"
}

admit() {
  local label=$1
  "$host_gate" --cpu "$cpu" --label "$label" --output "$load_ledger" >/dev/null
}

if ((resume == 1)); then
  start_sample=$(
    "$resume_helper" \
      "$output" "$load_ledger" "$rejected_ledger" "$binary_sha" \
      "$control_arm" "$treatment_arm" "$entries" "$passes"
  )
  if ((start_sample > samples)); then
    echo "all $samples pairs are already complete" >&2
    exit 0
  fi
  echo "resuming at pair $start_sample/$samples" >&2
else
  admit warmup-before
  for arm in "$control_arm" "$treatment_arm"; do
    for warmup in 1 2 3; do
      run_arm "$arm" "$scratch/warmup-$arm-$warmup.json"
    done
  done
  admit warmup-after

  printf '%s\n' \
    'sample,role,arm,binary_sha256,dictionary_entries,consumed_entries_per_pass,passes,elapsed_ns,checksum,boundary_calls' >"$output"
  start_sample=1
fi

for ((sample = start_sample; sample <= samples; sample++)); do
  admit "pair-$sample-before"
  if ((sample % 2 == 1)); then
    roles=(control treatment)
  else
    roles=(treatment control)
  fi
  for role in "${roles[@]}"; do
    if [[ $role == control ]]; then
      arm=$control_arm
    else
      arm=$treatment_arm
    fi
    run_arm "$arm" "$scratch/$sample-$role.json"
  done
  admit "pair-$sample-after"
  append_sample "$sample" control "$scratch/$sample-control.json"
  append_sample "$sample" treatment "$scratch/$sample-treatment.json"
  echo "pair $sample/$samples complete" >&2
done
