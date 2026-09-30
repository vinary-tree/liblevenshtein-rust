#!/usr/bin/env bash
set -euo pipefail

die() { echo "opam public asset: $*" >&2; exit 1; }

# The pure half is exercised by the offline cross-run and negative tests.
verify_bytes() {
  local repository=$1 tag=$2 package=$3 staged=$4 public=$5 api_digest=$6 output=$7
  local name="$package.tbz"
  local url="https://github.com/$repository/releases/download/$tag/$name"
  local digest
  [[ -f "$staged/$name" && -f "$staged/opam" &&
     -f "$public/$name" && -f "$public/SHA256SUMS" ]] ||
    die "required staged or public bytes are missing"
  digest=$(sha256sum "$public/$name")
  digest=${digest%% *}
  [[ "$api_digest" == "sha256:$digest" ]] || die "release API digest differs from public bytes"
  [[ $(grep -Fxc "$digest  ./$name" "$public/SHA256SUMS") == 1 ]] ||
    die "public SHA256SUMS does not name exactly these bytes"
  [[ $(awk -v plain="$name" -v dotted="./$name" \
      '$2 == plain || $2 == dotted { count++ } END { print count+0 }' \
      "$public/SHA256SUMS") == 1 ]] ||
    die "public SHA256SUMS has multiple entries for the source archive"
  cmp -s "$staged/$name" "$public/$name" || die "staged/public archive divergence"
  [[ $(grep -Ec '^[[:space:]]*url[[:space:]]*\{' "$staged/opam") == 1 &&
     $(grep -Ec '^[[:space:]]*src[[:space:]]*:' "$staged/opam") == 1 &&
     $(grep -Ec '^[[:space:]]*checksum[[:space:]]*:' "$staged/opam") == 1 ]] ||
    die "expected exactly one opam URL block, source, and checksum"
  [[ $(grep -Fxc "  src: \"$url\"" "$staged/opam") == 1 ]] ||
    die "opam source URL differs from exact release asset"
  [[ $(grep -Fxc "  checksum: \"sha256=$digest\"" "$staged/opam") == 1 ]] ||
    die "staged opam digest differs from public bytes"
  # Re-derive the submitted checksum from the downloaded public asset, never
  # from a separately rebuilt staging archive.
  sed -E "s|^  checksum: \"sha256=[0-9a-f]{64}\"$|  checksum: \"sha256=$digest\"|" \
    "$staged/opam" > "$output"
  [[ $(grep -Fxc "  checksum: \"sha256=$digest\"" "$output") == 1 ]] ||
    die "could not bind public checksum"
  printf '%s\n' "$digest"
}

if [[ "${1:-}" == --verify-local ]]; then
  [[ $# == 8 ]] || die "usage: $0 --verify-local <repo> <tag> <package> <staged> <public> <api-digest> <output-opam>"
  verify_bytes "${@:2}"
  exit
fi

[[ $# == 5 ]] || die "usage: $0 <repo> <tag> <package> <staged-dir> <new-output-dir>"
repository=$1 tag=$2 package=$3 staged=$4 output_dir=$5
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "invalid repository"
[[ "$tag" =~ ^v[0-9A-Za-z._-]+$ && "$package" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid tag or package identity"
[[ ! -e "$output_dir" ]] || die "output already exists"

head_sha=$(git rev-parse HEAD)
remote_refs=$(git ls-remote --tags "https://github.com/$repository.git" \
  "refs/tags/$tag" "refs/tags/$tag^{}")
remote_sha=$(awk -v ref="refs/tags/$tag^{}" '$2 == ref { print $1 }' <<< "$remote_refs")
if [[ -z "$remote_sha" ]]; then
  remote_sha=$(awk -v ref="refs/tags/$tag" '$2 == ref { print $1 }' <<< "$remote_refs")
fi
[[ -n "$remote_sha" && "$remote_sha" == "$head_sha" ]] ||
  die "public tag does not resolve to checked-out source commit"

release=$(gh api "repos/$repository/releases/tags/$tag")
[[ $(jq -r '.tag_name' <<< "$release") == "$tag" &&
   $(jq -r '.draft' <<< "$release") == false ]] || die "release identity or publication state differs"
immutable=$(jq -r '.immutable // false' <<< "$release")
if [[ "${VINARY_REQUIRE_IMMUTABLE_RELEASE:-0}" == 1 ]]; then
  [[ "$immutable" == true ]] || die "release immutability is required"
elif [[ "$immutable" != true ]]; then
  echo "warning: public release is mutable; digest/readback cannot prevent a later asset replacement" >&2
fi
name="$package.tbz"
[[ $(jq -r --arg name "$name" '[.assets[] | select(.name == $name and .state == "uploaded")] | length' <<< "$release") == 1 ]] ||
  die "expected exactly one uploaded source archive"
api_digest=$(jq -r --arg name "$name" '.assets[] | select(.name == $name) | .digest' <<< "$release")
[[ "$api_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "release asset lacks a SHA-256 digest"
expected_url="https://github.com/$repository/releases/download/$tag/$name"
actual_url=$(jq -r --arg name "$name" '.assets[] | select(.name == $name) | .browser_download_url' <<< "$release")
[[ "$actual_url" == "$expected_url" ]] || die "release asset URL differs from canonical identity"

mkdir -p "$output_dir/public"
gh release download "$tag" --repo "$repository" --pattern "$name" --dir "$output_dir/public"
gh release download "$tag" --repo "$repository" --pattern SHA256SUMS --dir "$output_dir/public"
verify_bytes "$repository" "$tag" "$package" "$staged" "$output_dir/public" \
  "$api_digest" "$output_dir/opam"
