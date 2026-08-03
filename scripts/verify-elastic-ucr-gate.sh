#!/usr/bin/env bash
# Verify the preregistered five-measure UCR result schema and shared slice.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-$PROJECT_ROOT/target/academic-benchmarks/results}"
MAX_CELLS="${ELASTIC_UCR_MAX_CELLS:-1000000000}"
MAX_DATASETS="${ELASTIC_UCR_MAX_DATASETS:-1000}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

MEASURES=(msm erp twed frechet dtw)

result_path() {
    local measure="$1"
    printf '%s/elastic_ucr_%s_%s_%s.csv' \
        "$RESULTS_DIR" "$measure" "$MAX_CELLS" "$MAX_DATASETS"
}

for measure in "${MEASURES[@]}"; do
    input="$(result_path "$measure")"
    if [[ ! -f "$input" ]]; then
        printf 'missing elastic UCR result: %s\n' "$input" >&2
        exit 1
    fi

    awk -F, -v expected_measure="$measure" '
        NR == 1 {
            if (NF != 32 || $1 != "record_type" || $32 != "parameters") {
                print FILENAME ": invalid 32-field header" > "/dev/stderr";
                exit 1;
            }
            next;
        }
        $1 == "summary" {
            summaries++;
            if (NF != 32 || $2 != expected_measure) {
                print FILENAME ": malformed summary row " NR > "/dev/stderr";
                exit 1;
            }
            if ($15 != $16 + $17 || $18 > $17) {
                print FILENAME ": flat accounting failure at row " NR > "/dev/stderr";
                exit 1;
            }
            if ($20 != $21 + $22 || $23 > $22 || $25 != $26 + $27 || $28 > $27) {
                print FILENAME ": trie accounting failure at row " NR > "/dev/stderr";
                exit 1;
            }
            print $3 > datasets;
            next;
        }
        $1 == "case" {
            if (NF != 6 || $2 != expected_measure) {
                print FILENAME ": malformed case row " NR > "/dev/stderr";
                exit 1;
            }
            if ($5 == "majority") {
                majority_cases++;
                print $3 "," $4 "," $6 > majority;
            } else if ($5 == expected_measure) {
                measure_cases++;
            } else {
                print FILENAME ": unexpected case arm at row " NR > "/dev/stderr";
                exit 1;
            }
            next;
        }
        {
            print FILENAME ": unexpected record type at row " NR > "/dev/stderr";
            exit 1;
        }
        END {
            if (summaries != 51 || majority_cases != 13754 || measure_cases != 13754) {
                printf "%s: expected 51 summaries and 13754 paired cases; got %d, %d, %d\n",
                    FILENAME, summaries, majority_cases, measure_cases > "/dev/stderr";
                exit 1;
            }
        }
    ' datasets="$SCRATCH/$measure.datasets" \
      majority="$SCRATCH/$measure.majority" \
      "$input"
done

for measure in erp twed frechet dtw; do
    diff -u "$SCRATCH/msm.datasets" "$SCRATCH/$measure.datasets"
    diff -u "$SCRATCH/msm.majority" "$SCRATCH/$measure.majority"
done

awk -F, '
    $1 == "summary" {
        majority += $11;
        correct += $12;
        total += $13;
        bound_pruned += $16;
        exact += $17;
        abandoned += $18;
    }
    END {
        if (majority != 5664 || correct != 11653 || total != 13754 ||
            bound_pruned != 152272 || exact != 1154677 || abandoned != 1087933) {
            printf "MSM compatibility failure: majority=%d correct=%d total=%d bound=%d exact=%d abandoned=%d\n",
                majority, correct, total, bound_pruned, exact, abandoned > "/dev/stderr";
            exit 1;
        }
    }
' "$(result_path msm)"

printf 'elastic UCR gate: PASS — five measures, 51 datasets, 13754 paired cases\n'
