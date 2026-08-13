#!/usr/bin/env bash
# Deterministic symlink farm for the JS harness — mirrors the repo's own
# committed layout (bindings/javascript/node_modules/@vinary-tree/* are
# symlinks), so local packages never resolve against the npm registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"
FARM="$SCRIPT_DIR/node_modules/@vinary-tree"
mkdir -p "$FARM"

link() { # name target
    ln -sfn "$2" "$FARM/$1"
}

link liblevenshtein "$LR/bindings/javascript"
link libdictenstein "$LD_REPO/bindings/javascript"
link interop "$LR/vinary-tree-interop/bindings/javascript"
link vinary-tree "$LR/bindings/javascript-runtime"

echo "symlink farm ready under $FARM" >&2
