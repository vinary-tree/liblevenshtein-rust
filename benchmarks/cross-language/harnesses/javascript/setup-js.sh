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

# libdictenstein's facade resolves @vinary-tree/{vinary-tree,interop} from
# wherever IT lives; with default (realpath) resolution that is the sibling
# repo, which has no node_modules (CI provisions one with npm ci, and that
# repo does not gitignore it — we must not write there). The harness wrapper
# therefore runs node with --preserve-symlinks, so the facade's package walk
# starts at the SYMLINK path inside this farm and finds the umbrella here.

echo "symlink farm ready under $FARM" >&2
