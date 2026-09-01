#!/usr/bin/env bash
#
# doc-mathlint.sh — repository-wide MathJax and documentation-classification gate.
#
# The default mode discovers every Markdown and Rust source. It scans all
# classified living Markdown plus all Rustdoc, excludes classified append-only
# evidence, and fails if any Markdown path is unclassified. It also runs the
# scanner's static positive/negative contract fixtures before the repository.
#
#   Manifest resolution order:
#     1. --manifest FILE
#     2. any FILE/glob arguments given on the command line
#     3. repository discovery and classification
#
# Usage:
#   scripts/doc-mathlint.sh                        # lint living docs + Rustdoc
#   scripts/doc-mathlint.sh --manifest FILE
#   scripts/doc-mathlint.sh docs/user-guide/*.md   # lint an explicit set
#
# Exit status: 0 = clean, 1 = violations found, 2 = usage / environment error.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scanner="scripts/doc-math-prescan.raku"
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
fi

# Keep only existing explicit files (a compatibility manifest may list a path
# that was archived or renamed).
declare -a present=()
for f in "${files[@]}"; do
  [[ -f "$f" ]] && present+=("$f")
done

echo "doc-mathlint: testing the delimiter, Rustdoc, fence, and classification contract…"
echo "──────────────────────────────────────────────────────────────────────────────"
raku scripts/test-doc-math-prescan.raku

declare -a scanner_args=(--lint)
if [[ ${#present[@]} -gt 0 ]]; then
  scanner_args+=("${present[@]}")
  scope="${#present[@]} explicit file(s)"
elif [[ -n "$manifest" || ${#files[@]} -gt 0 ]]; then
  echo "error: no existing files to lint" >&2
  exit 2
else
  scanner_args+=(--repository-root=.)
  scope="all classified living Markdown and Rustdoc"
fi

echo "doc-mathlint: scanning $scope for MathJax conformance…"
echo "──────────────────────────────────────────────────────────────────────────────"

# Run the fence-aware scanner in lint mode; capture output and status.
set +e
violations="$(raku "$scanner" "${scanner_args[@]}")"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "✅ PASS — 0 mathematical-syntax or Markdown-classification violations across $scope."
  echo "         (inline dollars surround backticks; display math uses a math fence.)"
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
