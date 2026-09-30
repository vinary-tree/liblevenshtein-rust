#!/usr/bin/env bash
set -euo pipefail

die() { echo "opam archive contract: $*" >&2; exit 1; }

[[ $# == 3 ]] || die "usage: $0 <new-disk-backed-scratch-dir> <owner/repo> <source-tag>"
scratch=$1 repository=$2 tag=$3
[[ ! -e "$scratch" ]] || die "scratch directory already exists"
mkdir -p "$scratch"

# Independent staging runs intentionally differ in process timezone, umask,
# and copied-file mtimes. The resulting compressed bytes must not.
(umask 022; TZ=UTC ./scripts/stage-ocaml-package.sh "$scratch/run1" "$tag")
sleep 2
(umask 077; TZ=Pacific/Honolulu ./scripts/stage-ocaml-package.sh "$scratch/run2" "$tag")
shopt -s nullglob
archives=("$scratch/run1/"*.tbz)
[[ ${#archives[@]} == 1 ]] || die "expected one source archive"
name=${archives[0]##*/}
package=${name%.tbz}
cmp -s "$scratch/run1/$name" "$scratch/run2/$name" ||
  die "archive bytes differ across independent staging runs"
cmp -s "$scratch/run1/opam" "$scratch/run2/opam" ||
  die "opam metadata differs across independent staging runs"

mkdir -p "$scratch/public" "$scratch/bad-public" "$scratch/bad-stage"
cp "$scratch/run1/$name" "$scratch/public/$name"
(cd "$scratch/public" && sha256sum "./$name" > SHA256SUMS)
digest=$(sha256sum "$scratch/public/$name")
digest=${digest%% *}
./scripts/verify-opam-public-asset.sh --verify-local \
  "$repository" "$tag" "$package" "$scratch/run1" "$scratch/public" \
  "sha256:$digest" "$scratch/verified.opam"
cmp -s "$scratch/run1/opam" "$scratch/verified.opam" ||
  die "public-byte-derived opam differs from staged metadata"

if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/run1" "$scratch/public" \
    "sha256:$(printf '%064d' 0)" "$scratch/rejected-digest.opam"; then
  die "accepted a release API digest mismatch"
fi
cp "$scratch/public/$name" "$scratch/bad-public/$name"
printf '%064d  ./%s\n' 0 "$name" > "$scratch/bad-public/SHA256SUMS"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/run1" "$scratch/bad-public" \
    "sha256:$digest" "$scratch/rejected-manifest.opam"; then
  die "accepted an incorrect public checksum manifest"
fi
cp "$scratch/public/SHA256SUMS" "$scratch/bad-public/SHA256SUMS"
printf '%064d  ./%s\n' 0 "$name" >> "$scratch/bad-public/SHA256SUMS"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/run1" "$scratch/bad-public" \
    "sha256:$digest" "$scratch/rejected-duplicate-manifest.opam"; then
  die "accepted conflicting public checksum entries"
fi
cp "$scratch/run2/$name" "$scratch/bad-stage/$name"
cp "$scratch/run2/opam" "$scratch/bad-stage/opam"
printf x >> "$scratch/bad-stage/$name"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/bad-stage" "$scratch/public" \
    "sha256:$digest" "$scratch/rejected-archive.opam"; then
  die "accepted staged/public archive divergence"
fi
sed 's@/releases/download/@/releases/invalid/@' \
  "$scratch/run2/opam" > "$scratch/bad-stage/opam"
cp "$scratch/run2/$name" "$scratch/bad-stage/$name"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/bad-stage" "$scratch/public" \
    "sha256:$digest" "$scratch/rejected-url.opam"; then
  die "accepted an incorrect opam source URL"
fi
cp "$scratch/run2/opam" "$scratch/bad-stage/opam"
printf '  src: "https://github.com/%s/releases/download/%s/%s"\n' \
  "$repository" "$tag" "$name" >> "$scratch/bad-stage/opam"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/bad-stage" "$scratch/public" \
    "sha256:$digest" "$scratch/rejected-duplicate.opam"; then
  die "accepted duplicate opam source metadata"
fi
cp "$scratch/run2/opam" "$scratch/bad-stage/opam"
printf '  checksum: "sha256=%s"\n' "$digest" >> "$scratch/bad-stage/opam"
if ./scripts/verify-opam-public-asset.sh --verify-local \
    "$repository" "$tag" "$package" "$scratch/bad-stage" "$scratch/public" \
    "sha256:$digest" "$scratch/rejected-duplicate-checksum.opam"; then
  die "accepted duplicate opam checksum metadata"
fi
echo "opam archive contract passed: $name sha256:$digest"
