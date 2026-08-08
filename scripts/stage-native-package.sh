#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <rust-target> <cargo-release-directory> <output-directory>" >&2
  exit 2
fi

target=$1
release_dir=$2
output_dir=$3
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)

if [ -z "$version" ]; then
  echo "could not read package version from Cargo.toml" >&2
  exit 1
fi

package_name="liblevenshtein-${version}-${target}"
prefix="${output_dir}/${package_name}"

mkdir -p \
  "${prefix}/bin" \
  "${prefix}/include" \
  "${prefix}/lib/cmake/liblevenshtein" \
  "${prefix}/lib/cmake/vinary-tree-interop" \
  "${prefix}/lib/pkgconfig"

cp include/liblevenshtein.h include/liblevenshtein_abi.h include/liblevenshtein.hpp \
  "${prefix}/include/"
cp vinary-tree-interop/include/vinary_tree_interop.h "${prefix}/include/"
cp cmake/liblevenshteinConfig.cmake cmake/liblevenshteinConfigVersion.cmake \
  "${prefix}/lib/cmake/liblevenshtein/"
cp cmake/vinary-tree-interopConfig.cmake cmake/vinary-tree-interopConfigVersion.cmake \
  "${prefix}/lib/cmake/vinary-tree-interop/"
cp pkgconfig/liblevenshtein.pc pkgconfig/vinary-tree-interop.pc \
  "${prefix}/lib/pkgconfig/"
cp LICENSE README.md "${prefix}/"
cp bindings/cpp/README.md "${prefix}/C_CPP_BINDINGS.md"

case "$target" in
  *-pc-windows-msvc)
    dll=$(find "$release_dir" -maxdepth 2 -type f -name 'liblevenshtein.dll' -print -quit)
    implib=$(find "$release_dir" -maxdepth 2 -type f \
      \( -name 'liblevenshtein.dll.lib' -o -name 'liblevenshtein.lib' \) -print -quit)
    static_library=$(find "$release_dir" -maxdepth 2 -type f \
      -name 'liblevenshtein.lib' -print -quit)
    test -n "$dll" && test -n "$implib" && test -n "$static_library"
    cp "$dll" "${prefix}/bin/liblevenshtein.dll"
    cp "$implib" "${prefix}/lib/liblevenshtein.dll.lib"
    cp "$static_library" "${prefix}/lib/liblevenshtein.lib"
    private_libs='-lbcrypt -luserenv -lws2_32 -lntdll -lsynchronization -ladvapi32'
    ;;
  *-apple-darwin)
    library=$(find "$release_dir" -maxdepth 2 -type f \
      -name 'libliblevenshtein.dylib' -print -quit)
    static_library=$(find "$release_dir" -maxdepth 2 -type f \
      -name 'libliblevenshtein.a' -print -quit)
    test -n "$library" && test -n "$static_library"
    cp "$library" "${prefix}/lib/libliblevenshtein.dylib"
    cp "$static_library" "${prefix}/lib/libliblevenshtein.a"
    private_libs='-ldl -lpthread -lm -liconv -framework CoreFoundation -framework Security'
    ;;
  *-linux-gnu)
    library=$(find "$release_dir" -maxdepth 2 -type f \
      -name 'libliblevenshtein.so' -print -quit)
    static_library=$(find "$release_dir" -maxdepth 2 -type f \
      -name 'libliblevenshtein.a' -print -quit)
    test -n "$library" && test -n "$static_library"
    cp "$library" "${prefix}/lib/libliblevenshtein.so"
    cp "$static_library" "${prefix}/lib/libliblevenshtein.a"
    private_libs='-ldl -lpthread -lm'
    ;;
  *)
    echo "unsupported release target: $target" >&2
    exit 1
    ;;
esac

sed -i.bak \
  "s|^Libs.private:.*|Libs.private: ${private_libs}|" \
  "${prefix}/lib/pkgconfig/liblevenshtein.pc"
rm -f "${prefix}/lib/pkgconfig/liblevenshtein.pc.bak"

tar -czf "${output_dir}/${package_name}.tar.gz" -C "$output_dir" "$package_name"
(
  cd "$prefix"
  pwd -P
)
