#!/usr/bin/env bash
# ClojureScript harness launcher.
#
# The compiled out/harness.js is a Node CJS script; its @vinary-tree/*
# requires must resolve against the deterministic symlink farm the JS
# harness owns (harnesses/javascript/node_modules — setup-js.sh is
# idempotent). Two mechanisms cover the whole require chain:
#   1. the harness-local node_modules symlink (created below) resolves the
#      facades from out/harness.js's own directory chain, and
#   2. NODE_PATH covers the facades' INTERNAL @vinary-tree/* requires,
#      which node otherwise resolves against each package's realpath
#      (the sibling repos carry no node_modules of their own).
# We also cd into harnesses/javascript so any cwd-relative resolution
# matches the js-native target exactly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/../.." && pwd)"
JS="$XL/harnesses/javascript"

[ -d "$JS/node_modules/@vinary-tree" ] || bash "$JS/setup-js.sh" >&2
if [ ! -e "$SCRIPT_DIR/node_modules" ]; then
    ln -sfn "$JS/node_modules" "$SCRIPT_DIR/node_modules"
fi
[ -f "$SCRIPT_DIR/out/harness.js" ] || {
    echo "run.sh: out/harness.js missing; compile with: taskset -c 16-31 clojure -M:compile (in $SCRIPT_DIR)" >&2
    exit 4
}

cd "$JS"
exec env NODE_PATH="$JS/node_modules" node "$SCRIPT_DIR/out/harness.js" "$@"
