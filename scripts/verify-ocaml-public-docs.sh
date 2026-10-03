#!/usr/bin/env bash
set -euo pipefail

die() { echo "OCaml public docs: $*" >&2; exit 1; }

[[ $# == 5 ]] ||
  die "usage: $0 <package> <opam-version> <public-module> <guide-marker> <api-marker>"
package=$1 version=$2 module=$3 guide_marker=$4 api_marker=$5
[[ "$package" =~ ^[a-z][a-z0-9-]*$ ]] || die "invalid opam package name"
[[ "$version" =~ ^[0-9A-Za-z.~_-]+$ ]] || die "invalid opam version"
[[ "$module" =~ ^[A-Z][A-Za-z0-9_]*$ ]] || die "invalid OCaml module"
[[ -n "$guide_marker" && -n "$api_marker" ]] || die "empty readback marker"

# Use disk-backed repo scratch, not the host's RAM-backed /tmp.
mkdir -p target
scratch=$(mktemp -d -p target ocaml-doc-readback.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
root="https://ocaml.org/p/$package/$version"
page="$root"
guide="$root/doc/index.html"
api="$root/doc/$package/$module/index.html"

curl -fLsS --retry 8 --retry-all-errors --retry-delay 10 --max-time 30 \
  "$page" -o "$scratch/package.html" ||
  die "version-specific package page is not public: $page"
curl -fLsS --retry 8 --retry-all-errors --retry-delay 10 --max-time 30 \
  "$guide" -o "$scratch/guide.html" ||
  die "version-specific odoc guide is not public: $guide"
curl -fLsS --retry 8 --retry-all-errors --retry-delay 10 --max-time 30 \
  "$api" -o "$scratch/api.html" ||
  die "version-specific module API is not public: $api"

grep -Fq "$package" "$scratch/package.html" ||
  die "package page lacks exact package identity"
grep -Fq "$version" "$scratch/package.html" ||
  die "package page lacks exact version identity"
grep -Fq "$guide_marker" "$scratch/guide.html" ||
  die "odoc guide lacks its source-specific content"
grep -Fq "$api_marker" "$scratch/api.html" ||
  die "module API lacks its documented interface"
echo "OCaml package, guide, and API readback passed: $root"
