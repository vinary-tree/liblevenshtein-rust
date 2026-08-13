#!/usr/bin/env bash
# Dedicated >=51-sample runs for the pgmcp hypothesis-deciding cells
# (experiments 178-183 prescribe >=51 replicates per arm; the regular matrix
# runs 30). One cell per target at the preregistered coordinates:
#
#   H-J1  (exp 178): jvm pair, transposition d2 tr-d2 — JMH 3 forks x 17
#                    iterations, driven by run-jvm-pair.sh hypothesis
#   H-J2  (exp 179): construct cells at reps=51 for jvm-vinary DD + legacy
#   H-JS1 (exp 180): js-native + js-legacy, standard d2 std-d2
#   H-JS2 (exp 181): js-wasm (+ js-native control), standard d2 std-d2
#   H-A1  (exp 182): every atlas target + rust control, standard d2 std-d2
#   H-M1  (exp 183): memory cells already deterministic one-shots (no rerun)
#
# Outputs land in <results>/hypothesis/ so the 30-sample matrix cells remain
# the canonical atlas data and these deeper runs are clearly the experiment
# arms. Resumable: existing cells are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"

[ $# -eq 1 ] || { echo "usage: run-hypothesis-cells.sh <results_dir>" >&2; exit 2; }
RESULTS_DIR="$(cd "$1" && pwd)"
HYP_DIR="$RESULTS_DIR/hypothesis"
mkdir -p "$HYP_DIR"

log() { printf '[hypothesis %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

query_cell() { # target backend algorithm distance queryset
    local target="$1" backend="$2" algorithm="$3" distance="$4" queryset="$5"
    local name="${target}__${backend}__query__${algorithm}__d${distance}__${queryset}.json"
    if [ -f "$HYP_DIR/$name" ]; then
        log "skip (exists): $name"
        return 0
    fi
    log "51-sample query cell: $target $backend $algorithm d$distance $queryset"
    XL_CELL_DIR="$HYP_DIR" XL_SAMPLES=51 \
        "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" "$target" "$backend" \
        query "$algorithm" "$distance" "$queryset" >/dev/null
}

construct_cell() { # target backend
    local target="$1" backend="$2"
    local name="${target}__${backend}__construct__standard__d1__hits.json"
    if [ -f "$HYP_DIR/$name" ]; then
        log "skip (exists): $name"
        return 0
    fi
    log "51-rep construct cell: $target $backend"
    XL_CELL_DIR="$HYP_DIR" XL_REPS=51 \
        "$SCRIPT_DIR/run-one.sh" cell "$RESULTS_DIR" "$target" "$backend" \
        construct standard 1 hits >/dev/null
}

# H-J1: JMH deep profile (3 forks x 17 iterations = 51 samples/side).
if [ ! -f "$RESULTS_DIR/hypothesis/jvm-vinary__dynamic_dawg__query__transposition__d2__tr-d2.json" ] \
    || [ ! -f "$RESULTS_DIR/hypothesis/jvm-legacy__own__query__transposition__d2__tr-d2.json" ]; then
    log "H-J1: JMH hypothesis profile"
    bash "$SCRIPT_DIR/run-jvm-pair.sh" hypothesis "$RESULTS_DIR"
fi

# H-J2: construction arms at 51 reps.
construct_cell jvm-vinary dynamic_dawg
construct_cell jvm-legacy own

# H-JS1 + H-JS2 arms.
query_cell js-legacy own standard 2 std-d2
query_cell js-native dynamic_dawg standard 2 std-d2
query_cell js-wasm dynamic_dawg standard 2 std-d2

# H-A1: rust control + every atlas target at the deciding coordinate.
# H-C1 (exp 184) shares this coordinate: its arms are the `cpp` and
# `cpp-legacy` rows below. Each target's backend comes from targets.tsv —
# the legacy targets build their own dictionary and reject `dynamic_dawg`.
primary_backend() {
    awk -F'\t' -v t="$1" '!/^#/ && $1 == t { split($4, b, ","); print b[1] }' "$XL/targets.tsv"
}
for target in rust c cpp cpp-legacy python go ruby lua dotnet swift fortran ocaml haskell clojure clojurescript; do
    query_cell "$target" "$(primary_backend "$target")" standard 2 std-d2
done

log "hypothesis cells complete under $HYP_DIR"
