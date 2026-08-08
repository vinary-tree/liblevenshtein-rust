#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <interop|liblevenshtein> <new-output-directory>" >&2
  exit 2
fi

package=$1
output=$2
if [ -e "$output" ]; then
  echo "output already exists: $output" >&2
  exit 1
fi

mkdir -p "$output/src"
case "$package" in
  interop)
    cp "$repository_root/vinary-tree-interop/bindings/fortran/fpm.toml" \
      "$output/fpm.toml"
    cp "$repository_root/vinary-tree-interop/bindings/fortran/src/vinary_tree_interop.f90" \
      "$output/src/"
    cp "$repository_root/vinary-tree-interop/README.md" "$output/README.md"
    cp "$repository_root/LICENSE" "$output/LICENSE"
    ;;
  liblevenshtein)
    cp "$repository_root/bindings/fortran/fpm.publish.toml" "$output/fpm.toml"
    cp "$repository_root/bindings/fortran/src/vinary_tree_liblevenshtein.f90" \
      "$output/src/"
    cp "$repository_root/bindings/fortran/README.md" "$output/README.md"
    cp "$repository_root/bindings/fortran/LICENSE" "$output/LICENSE"
    ;;
  *)
    echo "unknown fpm package: $package" >&2
    exit 2
    ;;
esac

git -C "$output" init --quiet
git -C "$output" add fpm.toml src README.md LICENSE
git -C "$output" \
  -c user.name="Vinary Tree release automation" \
  -c user.email="dylon.devo@gmail.com" \
  commit --quiet -m "Package $package"
