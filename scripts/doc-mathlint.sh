#!/usr/bin/env bash
#
# doc-mathlint.sh — MathJax-conformance gate for the liblevenshtein LIVING documentation.
#
# It runs the fence-aware scanner (scripts/doc-math-prescan.raku) in --lint mode over the
# set of documents this campaign rewrites (Bucket A), and fails if any of them still carries
# old-style math: Unicode-literal formulae (backticked or bare), bare undelimited `O(...)`
# in prose, leaked bare-dollar LaTeX, or a letter abutting a `$ opening delimiter.
#
# The living-doc set is read from a manifest (one repo-relative path per line, `#` comments
# allowed). This is deliberately an *allow-list*: the append-only scientific records (Bucket
# B) keep their notation and are never listed here.
#
#   Manifest resolution order:
#     1. --manifest FILE
#     2. any FILE/glob arguments given on the command line
#     3. docs/.mathlint-include.txt   (the campaign's curated living-doc manifest)
#
# Usage:
#   scripts/doc-mathlint.sh                        # lint the curated manifest
#   scripts/doc-mathlint.sh --manifest FILE
#   scripts/doc-mathlint.sh docs/user-guide/*.md   # lint an explicit set
#
# Exit status: 0 = clean, 1 = violations found, 2 = usage / environment error.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scanner="scripts/doc-math-prescan.raku"
default_manifest="docs/.mathlint-include.txt"
manifest=""
declare -a files=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="${2:?--manifest needs a FILE}"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) files+=("$1"); shift ;;
  esac
done

command -v raku >/dev/null 2>&1 || { echo "error: raku not found on PATH" >&2; exit 2; }
[[ -f "$scanner" ]] || { echo "error: scanner missing: $scanner" >&2; exit 2; }

# Resolve the file set.
if [[ -n "$manifest" ]]; then
  [[ -f "$manifest" ]] || { echo "error: manifest not found: $manifest" >&2; exit 2; }
  mapfile -t files < <(grep -vE '^\s*(#|$)' "$manifest")
elif [[ ${#files[@]} -eq 0 ]]; then
  [[ -f "$default_manifest" ]] || {
    echo "error: no files given and $default_manifest is missing." >&2
    echo "       pass paths/globs, or --manifest FILE, or create the manifest." >&2
    exit 2
  }
  mapfile -t files < <(grep -vE '^\s*(#|$)' "$default_manifest")
fi

# Keep only existing files (a manifest may list a path that was archived/renamed).
declare -a present=()
for f in "${files[@]}"; do
  [[ -f "$f" ]] && present+=("$f")
done
[[ ${#present[@]} -gt 0 ]] || { echo "error: no existing files to lint" >&2; exit 2; }

echo "doc-mathlint: scanning ${#present[@]} living document(s) for MathJax conformance…"
echo "──────────────────────────────────────────────────────────────────────────────"

# Run the fence-aware scanner in lint mode; capture output and status.
set +e
violations="$(raku "$scanner" --lint "${present[@]}")"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "✅ PASS — 0 old-style math constructs across ${#present[@]} living document(s)."
  echo "         (all inline math is \`\$…\$\`; display math is \`\`\`math; no bare O(...) in prose.)"
  exit 0
else
  echo "$violations"
  echo "──────────────────────────────────────────────────────────────────────────────"
  n="$(printf '%s\n' "$violations" | grep -c ':' || true)"
  echo "❌ FAIL — $n residual old-style-math construct(s). Convert per scripts/doc-math-prescan.raku --key."
  echo "   Kinds:"
  printf '%s\n' "$violations" | sed -E 's/^[^ ]+ ([a-z-]+):.*/\1/' | sort | uniq -c | sed 's/^/     /'
  exit 1
fi
