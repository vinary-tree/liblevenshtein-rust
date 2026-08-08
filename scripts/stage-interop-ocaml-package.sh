#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <new-output-directory>" >&2
  exit 2
fi

output=$1
if [ -e "$output" ]; then
  echo "output already exists: $output" >&2
  exit 1
fi

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' vinary-tree-interop/Cargo.toml | head -n 1)
package="vinary-tree-interop-$version"
source="$output/source/$package"
mkdir -p "$source"
cp vinary-tree-interop/bindings/ocaml/dune "$source/"
cp vinary-tree-interop/bindings/ocaml/dune-project "$source/"
cp vinary-tree-interop/bindings/ocaml/vinary_tree_interop.ml "$source/"
cp vinary-tree-interop/bindings/ocaml/vinary_tree_interop.mli "$source/"
cp vinary-tree-interop/bindings/ocaml/vinary-tree-interop.opam.template \
  "$source/vinary-tree-interop.opam"
cp vinary-tree-interop/README.md "$source/README.md"
cp LICENSE "$source/LICENSE"

archive="$output/$package.tbz"
tar -cjf "$archive" -C "$output/source" "$package"
cp vinary-tree-interop/bindings/ocaml/vinary-tree-interop.opam.template "$output/opam"
read -r checksum _ < <(sha256sum "$archive")
printf '\nurl {\n  src: "https://github.com/vinary-tree/liblevenshtein-rust/releases/download/interop-v%s/%s.tbz"\n  checksum: "sha256=%s"\n}\n' \
  "$version" "$package" "$checksum" >> "$output/opam"
