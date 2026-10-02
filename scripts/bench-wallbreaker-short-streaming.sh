#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 || ! -x "$1" || ! -x "$2" || -e "$5" ]]; then
  echo "usage: $0 <baseline-probe> <streaming-probe> <baseline-git-ref> <streaming-git-ref> <new-disk-backed-output-dir> [timed-iterations-per-process]" >&2
  exit 2
fi

baseline=$1 streaming=$2 baseline_ref=$3 streaming_ref=$4 output=$5
iterations=${6:-500}
if [[ ! $iterations =~ ^[1-9][0-9]{0,9}$ ]] || ((iterations > 2147483647)); then
  echo "timed-iterations-per-process must be a positive 32-bit decimal integer" >&2
  exit 2
fi
mkdir -p "$output"
printf '%s\n' "$iterations" > "$output/iterations.txt"
git rev-parse --verify "$baseline_ref^{commit}" > "$output/baseline-commit.txt"
git rev-parse --verify "$streaming_ref^{commit}" > "$output/streaming-commit.txt"
sha256sum "$baseline" "$streaming" > "$output/binaries.sha256"
git show "$baseline_ref:examples/wallbreaker_oracle_probe.rs" | sha256sum > "$output/baseline-source.sha256"
git show "$streaming_ref:examples/wallbreaker_oracle_probe.rs" | sha256sum > "$output/streaming-source.sha256"
sha256sum "$0" > "$output/runner.sha256"
uname -a > "$output/uname.txt"
lscpu > "$output/lscpu.txt"

run_one() {
  local case=$1 arm=$2 seed=$3 binary=$4 command_arm=$5
  taskset -c 2 "$binary" "$case" "$command_arm" "$seed" "$iterations" \
    > "$output/$case-$arm-$seed.txt"
}

# Thirty-two prespecified seeds, paired in sixteen ABBA/BAAB blocks. The
# previous source revision supplies the original eager WallBreaker control;
# the candidate revision supplies the borrowed-term streaming implementation.
for ((block = 0; block < 16; block++)); do
  first_seed=$((1001 + 2 * block))
  second_seed=$((first_seed + 1))
  if ((block % 2 == 0)); then
    cases=(short selective)
  else
    cases=(selective short)
  fi
  for case in "${cases[@]}"; do
    if ((block % 2 == 0)); then
      run_one "$case" control "$first_seed" "$baseline" wallbreaker
      run_one "$case" treatment "$first_seed" "$streaming" wallbreaker
      run_one "$case" treatment "$second_seed" "$streaming" wallbreaker
      run_one "$case" control "$second_seed" "$baseline" wallbreaker
    else
      run_one "$case" treatment "$first_seed" "$streaming" wallbreaker
      run_one "$case" control "$first_seed" "$baseline" wallbreaker
      run_one "$case" control "$second_seed" "$baseline" wallbreaker
      run_one "$case" treatment "$second_seed" "$streaming" wallbreaker
    fi
  done

  # The old probe had no empty-query case. The new probe retains the original
  # eager fallback as an in-binary control, avoiding a post-hoc omission.
  if ((block % 2 == 0)); then
    run_one empty control "$first_seed" "$streaming" eager
    run_one empty treatment "$first_seed" "$streaming" wallbreaker
    run_one empty treatment "$second_seed" "$streaming" wallbreaker
    run_one empty control "$second_seed" "$streaming" eager
  else
    run_one empty treatment "$first_seed" "$streaming" wallbreaker
    run_one empty control "$first_seed" "$streaming" eager
    run_one empty control "$second_seed" "$streaming" eager
    run_one empty treatment "$second_seed" "$streaming" wallbreaker
  fi
  printf 'completed block %d/16\n' "$((block + 1))"
done
