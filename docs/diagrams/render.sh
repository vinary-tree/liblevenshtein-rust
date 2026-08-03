#!/usr/bin/env bash
#
# render.sh — render every diagram source under docs/diagrams/ to a committed
# sibling SVG with the repository's reproducible diagram toolchain.
#
#   Source ext   Tool          Concept
#   ----------   -----------   ------------------------------------------------
#   .puml        PlantUML      component / sequence / state / class diagrams
#   .dsl         Structurizr   C4 model views (exported via PlantUML)
#   .d2          D2            polished conceptual architecture
#   .dot         Graphviz      dependency graphs, decision trees, Hasse lattices
#   .mmd         Mermaid       GitHub-native sketches
#   .pikchr      Pikchr        precise small structural diagrams
#   .asy         Asymptote     publication-grade mathematical figures
#
# Every source has exactly one committed sibling `<stem>.svg`. Never hand-edit an
# SVG: edit the source and re-run this script. CI runs `render.sh --check`, which
# re-renders to a temp tree and diffs against the committed SVGs (non-zero exit on
# drift), guaranteeing the committed SVGs always match their sources.
#
# Usage:
#   ./render.sh                 # render every source in place
#   ./render.sh <substring>     # render only sources whose path matches <substring>
#   ./render.sh --check         # render to a temp tree; fail if any SVG differs
#   ./render.sh --list          # list every discovered source and its target SVG
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUPPETEER_CFG="$DIR/puppeteer-config.json"

# Source extensions, paired with the binary that renders them.
declare -A TOOL=(
  [puml]=plantuml [dsl]=structurizr [d2]=d2
  [dot]=dot [mmd]=mmdc [pikchr]=pikchr [asy]=asy
)

die()  { printf 'render.sh: %s\n' "$*" >&2; exit 1; }
warn() { printf 'render.sh: %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

# Discover every diagram source (sorted, stable order).
discover() {
  find "$DIR" -type f \( \
    -name '*.puml' -o -name '*.dsl' -o -name '*.d2' -o \
    -name '*.dot'  -o -name '*.mmd' -o -name '*.pikchr' -o -name '*.asy' \
  \) | sort
}

# Render one Structurizr workspace: export its primary view to PlantUML, then SVG.
render_structurizr() {
  local src="$1" out="$2" tmp puml
  tmp="$(mktemp -d)"
  if ! structurizr export -workspace "$src" -format plantuml -output "$tmp" >/dev/null 2>&1; then
    rm -rf "$tmp"; warn "structurizr export failed for $src"; return 1
  fi
  # The primary view is the first non-legend (.puml that is not *-key.puml).
  puml="$(find "$tmp" -name '*.puml' ! -name '*-key.puml' | sort | head -1)"
  [ -n "$puml" ] || { rm -rf "$tmp"; warn "structurizr produced no view for $src"; return 1; }
  JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djava.awt.headless=true" \
    plantuml -tsvg -nometadata -o "$tmp" "$puml"
  mv "${puml%.puml}.svg" "$out"
  rm -rf "$tmp"
}

# Render a single source into <outdir>/<stem>.svg.
render_one() {
  local src="$1" outdir="$2" stem ext out
  stem="$(basename "${src%.*}")"
  ext="${src##*.}"
  out="$outdir/$stem.svg"
  mkdir -p "$outdir"
  case "$ext" in
    puml)   JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djava.awt.headless=true" \
              plantuml -tsvg -nometadata -o "$outdir" "$src" ;;
    d2)     d2 --layout=elk --pad=20 "$src" "$out" >/dev/null 2>&1 ;;
    mmd)    mmdc -i "$src" -o "$out" -b transparent -p "$PUPPETEER_CFG" -q >/dev/null 2>&1 ;;
    dot)    dot -Tsvg -o "$out" "$src" ;;
    pikchr) pikchr --svg-only "$src" > "$out" ;;
    asy)    asy -f svg -o "$outdir/$stem" "$src" >/dev/null 2>&1 ;;
    dsl)    render_structurizr "$src" "$out" || return 1 ;;
    *)      warn "unknown extension: .$ext ($src)"; return 1 ;;
  esac
  [ -s "$out" ] || { warn "empty SVG produced for $src"; return 1; }
}

# Preflight: only require a tool if at least one matching source exists.
preflight() {
  local -A needed=()
  local src ext
  for src in "$@"; do ext="${src##*.}"; needed[$ext]=1; done
  local missing=0
  for ext in "${!needed[@]}"; do
    if ! have "${TOOL[$ext]}"; then
      warn "missing renderer '${TOOL[$ext]}' for .$ext sources"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || die "install the missing renderer(s) listed above"
}

main() {
  local mode="build" filter=""
  case "${1:-}" in
    --check) mode="check" ;;
    --list)  mode="list" ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "")      ;;
    *)       filter="$1" ;;
  esac

  local all=() srcs=()
  mapfile -t all < <(discover)
  [ "${#all[@]}" -gt 0 ] || { warn "no diagram sources found"; exit 0; }
  if [ -n "$filter" ]; then
    mapfile -t srcs < <(printf '%s\n' "${all[@]}" | grep -F -- "$filter" || true)
    [ "${#srcs[@]}" -gt 0 ] || die "no sources match '$filter'"
  else
    srcs=("${all[@]}")
  fi

  if [ "$mode" = "list" ]; then
    local s; for s in "${srcs[@]}"; do printf '%s  ->  %s.svg\n' "$s" "${s%.*}"; done
    exit 0
  fi

  preflight "${srcs[@]}"

  if [ "$mode" = "check" ]; then
    local tmp drift=0 s rel ref
    tmp="$(mktemp -d)"
    for s in "${srcs[@]}"; do
      rel="${s#"$DIR"/}"
      render_one "$s" "$tmp/$(dirname "$rel")" || { warn "FAILED render: $rel"; drift=1; continue; }
      ref="${s%.*}.svg"
      if [ ! -f "$ref" ]; then
        warn "MISSING committed SVG: $ref"; drift=1
      elif ! diff -q "$ref" "$tmp/$(dirname "$rel")/$(basename "${s%.*}").svg" >/dev/null; then
        warn "DRIFT: $ref is out of date with its source"; drift=1
      fi
    done
    rm -rf "$tmp"
    [ "$drift" -eq 0 ] || die "committed SVGs are out of sync — run ./render.sh"
    echo "render.sh: all ${#srcs[@]} SVG(s) are in sync."
    exit 0
  fi

  local s n=0 fails=0
  for s in "${srcs[@]}"; do
    if render_one "$s" "$(dirname "$s")"; then
      printf '  rendered %s\n' "${s#"$DIR"/}"; n=$((n + 1))
    else
      warn "FAILED  ${s#"$DIR"/}"; fails=$((fails + 1))
    fi
  done
  echo "render.sh: rendered $n diagram(s), $fails failed."
  [ "$fails" -eq 0 ] || exit 1
}

main "$@"
