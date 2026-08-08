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
prefix="$output/$package"
mkdir -p "$prefix/include" "$prefix/lib/cmake/vinary-tree-interop" "$prefix/lib/pkgconfig"
cp vinary-tree-interop/include/vinary_tree_interop.h "$prefix/include/"
cp cmake/vinary-tree-interopConfig.cmake cmake/vinary-tree-interopConfigVersion.cmake \
  "$prefix/lib/cmake/vinary-tree-interop/"
cp pkgconfig/vinary-tree-interop.pc "$prefix/lib/pkgconfig/"
cp vinary-tree-interop/README.md "$prefix/README.md"
cp LICENSE "$prefix/LICENSE"
tar -czf "$output/$package.tar.gz" -C "$output" "$package"
printf '%s\n' "$prefix"
