#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 && ! -e "$1" ]] || {
  echo "usage: $0 <new-disk-backed-scratch-dir>" >&2
  exit 2
}
scratch=$1
mkdir -p "$scratch/valid/staged/native-a" "$scratch/valid/staged/maven"
printf 'native\n' > "$scratch/valid/staged/native-a/native.tbz"
printf 'jar\n' > "$scratch/valid/staged/maven/library.jar"
printf 'metadata\n' > "$scratch/valid/staged/maven/README"
bash scripts/flatten-release-assets.sh \
  "$scratch/valid/staged" "$scratch/valid/assets" maven
[[ -f "$scratch/valid/assets/native.tbz" &&
   -f "$scratch/valid/assets/library.jar" &&
   ! -e "$scratch/valid/assets/README" ]] ||
  { echo "filtered asset inventory differs" >&2; exit 1; }
(cd "$scratch/valid/assets" && sha256sum --check SHA256SUMS)

mkdir -p "$scratch/duplicate/staged/one" "$scratch/duplicate/staged/two"
printf 'one\n' > "$scratch/duplicate/staged/one/same.tbz"
printf 'two\n' > "$scratch/duplicate/staged/two/same.tbz"
if bash scripts/flatten-release-assets.sh \
    "$scratch/duplicate/staged" "$scratch/duplicate/assets"; then
  echo "accepted a duplicate release asset basename" >&2
  exit 1
fi

mkdir -p "$scratch/reserved/staged/one"
printf 'forged\n' > "$scratch/reserved/staged/one/SHA256SUMS"
if bash scripts/flatten-release-assets.sh \
    "$scratch/reserved/staged" "$scratch/reserved/assets"; then
  echo "accepted an artifact-provided checksum manifest" >&2
  exit 1
fi

echo "release asset flattening controls passed"
