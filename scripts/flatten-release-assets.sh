#!/usr/bin/env bash
# Flatten independently downloaded artifact groups without losing provenance.
set -euo pipefail

die() { echo "release asset staging: $*" >&2; exit 1; }
[[ $# -ge 2 ]] || die "usage: $0 <staging-root> <new-assets-dir> [jar-pom-only-group ...]"
source_root=${1%/}
output=$2
shift 2
[[ -d "$source_root" && ! -e "$output" ]] ||
  die "staging root must exist and output must be new"
for group in "$@"; do
  [[ "$group" =~ ^[A-Za-z0-9_-]+$ && -d "$source_root/$group" ]] ||
    die "invalid or missing filtered group: $group"
done
[[ -z $(find "$source_root" -type l -print -quit) ]] ||
  die "symlinked release assets are not supported"

mkdir "$output"
count=0
while IFS= read -r -d '' file; do
  relative=${file#"$source_root"/}
  group=${relative%%/*}
  name=${file##*/}
  for filtered in "$@"; do
    if [[ "$group" == "$filtered" ]]; then
      case "$name" in *.jar|*.pom) ;; *) continue 2 ;; esac
    fi
  done
  [[ "$name" != SHA256SUMS ]] || die "artifact attempted to supply SHA256SUMS"
  [[ ! -e "$output/$name" && ! -L "$output/$name" ]] ||
    die "release asset basename collision: $name"
  cp -- "$file" "$output/$name"
  ((count += 1))
done < <(find "$source_root" -type f -print0 | LC_ALL=C sort -z)
[[ $count -gt 0 ]] || die "no publishable assets were staged"

(
  cd "$output"
  mapfile -d '' -t files < <(
    find . -mindepth 1 -maxdepth 1 -type f -print0 | LC_ALL=C sort -z
  )
  [[ ${#files[@]} == "$count" ]] ||
    die "flat asset inventory changed while producing manifest"
  printf '%s\0' "${files[@]}" | xargs -0 -r sha256sum > SHA256SUMS
)
echo "staged $count collision-free release assets"
