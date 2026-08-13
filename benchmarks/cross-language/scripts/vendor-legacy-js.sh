#!/usr/bin/env bash
# Vendor the legacy compiled-CoffeeScript JavaScript 2.0.4 artifacts, byte-
# exactly as the old interactive demo served them, with full provenance.
#
# Source of truth (user directive): the gh-pages branch of
# universal-automata/liblevenshtein — the GitHub static pages linked from the
# liblevenshtein-java README's demo. Cross-check: the npm tarball
# liblevenshtein@2.0.4 (published 2015-07-04) whose registry dist shasum is
# pinned below; both carry the same compiled 2.0.4 output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR="$XL/legacy/javascript/vendor"
mkdir -p "$VENDOR"

GH_COMMIT="59934da81e56662bfe317aefd6877fc461a64438"   # refs/heads/gh-pages, resolved 2026-08-13
BASE="https://raw.githubusercontent.com/universal-automata/liblevenshtein/$GH_COMMIT/javascripts/2.0.4"
NPM_EXPECTED_SHA1="caa72ae5d4bc79261ba4d4941a7116834f370dd1"  # registry dist.shasum for 2.0.4

fetch() { # filename
    local name="$1"
    curl -fsSL "$BASE/$name" -o "$VENDOR/$name"
    echo "fetched $name ($(wc -c < "$VENDOR/$name") bytes)" >&2
}

fetch levenshtein-transducer.js
fetch levenshtein-transducer.min.js

# npm cross-check: same compiled output published to the registry.
SCRATCH="$(mktemp -d "$XL/.stage/vendor-js.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
(cd "$SCRATCH" && npm pack liblevenshtein@2.0.4 --silent >/dev/null)
TGZ="$SCRATCH/liblevenshtein-2.0.4.tgz"
ACTUAL_SHA1="$(sha1sum "$TGZ" | cut -d' ' -f1)"
VERDICT="tarball-sha1-match"
if [ "$ACTUAL_SHA1" != "$NPM_EXPECTED_SHA1" ]; then
    VERDICT="tarball-sha1-MISMATCH ($ACTUAL_SHA1)"
fi
tar -xzf "$TGZ" -C "$SCRATCH"
NPM_BUNDLE="$SCRATCH/package/lib/liblevenshtein/levenshtein-transducer.js"
CONTENT_VERDICT="npm-bundle-absent"
if [ -f "$NPM_BUNDLE" ]; then
    if cmp -s "$NPM_BUNDLE" "$VENDOR/levenshtein-transducer.js"; then
        CONTENT_VERDICT="identical-to-gh-pages"
    else
        CONTENT_VERDICT="DIFFERS-from-gh-pages"
    fi
fi

SHA_FULL="$(sha256sum "$VENDOR/levenshtein-transducer.js" | cut -d' ' -f1)"
SHA_MIN="$(sha256sum "$VENDOR/levenshtein-transducer.min.js" | cut -d' ' -f1)"

cat > "$XL/legacy/javascript/vendor/provenance.json" <<EOF
{
  "source_repo": "https://github.com/universal-automata/liblevenshtein",
  "branch": "gh-pages",
  "git_commit": "$GH_COMMIT",
  "path": "javascripts/2.0.4/",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "files": [
    { "name": "levenshtein-transducer.js", "sha256": "$SHA_FULL",
      "url": "$BASE/levenshtein-transducer.js" },
    { "name": "levenshtein-transducer.min.js", "sha256": "$SHA_MIN",
      "url": "$BASE/levenshtein-transducer.min.js" }
  ],
  "npm_cross_check": {
    "package": "liblevenshtein",
    "version": "2.0.4",
    "sha1": "$ACTUAL_SHA1",
    "expected_sha1": "$NPM_EXPECTED_SHA1",
    "tarball_verdict": "$VERDICT",
    "content_verdict": "$CONTENT_VERDICT"
  },
  "license": "MIT (c) 2014 Dylon Edwards",
  "notes": "Compiled CoffeeScript 1.7.1 output; exports-aware IIFE loadable via CommonJS require() on modern Node."
}
EOF

cat > "$XL/legacy/javascript/vendor/LICENSE.vendored.md" <<'EOF'
# Vendored artifact license

The files in this directory are the compiled JavaScript build of
liblevenshtein-coffeescript 2.0.4, vendored byte-exactly from the `gh-pages`
branch of https://github.com/universal-automata/liblevenshtein (see
`provenance.json` for the commit, URLs, and digests).

They are distributed under their original license:

The MIT License (MIT)

Copyright (c) 2014 Dylon Edwards

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
EOF

echo "vendored to $VENDOR (npm cross-check: $VERDICT, content: $CONTENT_VERDICT)" >&2
