#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <new-output-directory> [source-tag]" >&2
  exit 2
fi

output=$1
if [ -e "$output" ]; then
  echo "output already exists: $output" >&2
  exit 1
fi

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)
source_tag=${2:-v$version}
package="liblevenshtein-$version"
source="$output/source/$package"
mkdir -p "$source/include" "$source/test"
cp bindings/ocaml/dune bindings/ocaml/dune-project "$source/"
cp bindings/ocaml/vinary_tree_liblevenshtein.ml "$source/"
cp bindings/ocaml/vinary_tree_liblevenshtein.mli "$source/"
cp bindings/ocaml/liblevenshtein_stubs.c "$source/"
cp bindings/ocaml/README.md "$source/README.md"
cp bindings/ocaml/liblevenshtein.opam.template \
  "$source/liblevenshtein.opam"
cp bindings/ocaml/include/* "$source/include/"
cp bindings/ocaml/test/dune bindings/ocaml/test/snapshot.ml "$source/test/"
cp LICENSE "$source/LICENSE"

archive="$output/$package.tbz"
tar -cjf "$archive" -C "$output/source" "$package"
cp bindings/ocaml/liblevenshtein.opam.template "$output/opam"
read -r checksum _ < <(sha256sum "$archive")
printf '\nurl {\n  src: "https://github.com/vinary-tree/liblevenshtein-rust/releases/download/%s/%s.tbz"\n  checksum: "sha256=%s"\n}\n' \
  "$source_tag" "$package" "$checksum" >> "$output/opam"
