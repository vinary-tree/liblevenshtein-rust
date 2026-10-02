#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ! -x "$1" || -e "$2" ]]; then
  echo "usage: $0 <built-wallbreaker-oracle-probe> <new-disk-backed-output-dir>" >&2
  exit 2
fi

binary=$1 output=$2
mkdir -p "$output"
sha256sum "$binary" > "$output/binary.sha256"
sha256sum examples/wallbreaker_oracle_probe.rs "$0" > "$output/sources.sha256"
uname -a > "$output/uname.txt"
lscpu > "$output/lscpu.txt"

run_one() {
  local case=$1 arm=$2 seed=$3
  taskset -c 2 "$binary" "$case" "$arm" "$seed" 500 \
    > "$output/$case-$arm-$seed.txt"
}

for ((block = 0; block < 16; block++)); do
  first_seed=$((1001 + 2 * block))
  second_seed=$((first_seed + 1))
  if ((block % 2 == 0)); then
    cases=(selective short)
  else
    cases=(short selective)
  fi
  for case in "${cases[@]}"; do
    if ((block % 2 == 0)); then
      run_one "$case" wallbreaker "$first_seed"
      run_one "$case" exhaustive "$first_seed"
      run_one "$case" exhaustive "$second_seed"
      run_one "$case" wallbreaker "$second_seed"
    else
      run_one "$case" exhaustive "$first_seed"
      run_one "$case" wallbreaker "$first_seed"
      run_one "$case" wallbreaker "$second_seed"
      run_one "$case" exhaustive "$second_seed"
    fi
  done
  printf 'completed block %d/16\n' "$((block + 1))"
done
