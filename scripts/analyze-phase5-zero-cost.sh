#!/usr/bin/env bash
# Aggregate Criterion relative-change confidence intervals for one Phase 5 suite.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: scripts/analyze-phase5-zero-cost.sh SUITE MANIFEST\n' >&2
    exit 2
fi

readonly suite="$1"
readonly manifest="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root
criterion_root="$(realpath "$repo_root/target/criterion")"
readonly criterion_root

if [[ ! "$suite" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'invalid suite name: %s\n' "$suite" >&2
    exit 2
fi
if [[ ! -f "$manifest" ]]; then
    printf 'missing change manifest: %s\n' "$manifest" >&2
    exit 2
fi

count=0
sum_point=0
sum_upper=0
printf 'suite\tcase\tmean_change\tupper95_change\n'

while IFS= read -r estimate_file; do
    [[ -z "$estimate_file" ]] && continue
    canonical="$(realpath "$estimate_file")"
    case "$canonical" in
        "$criterion_root"/*/change/estimates.json) ;;
        *)
            printf 'estimate escapes Criterion change tree: %s\n' "$estimate_file" >&2
            exit 2
            ;;
    esac

    point="$(jq -er '.mean.point_estimate | numbers' "$canonical")"
    upper="$(jq -er '.mean.confidence_interval.upper_bound | numbers' "$canonical")"
    case_name="${canonical#"$criterion_root"/}"
    case_name="${case_name%/change/estimates.json}"

    printf '%s\t%s\t%.9f\t%.9f\n' "$suite" "$case_name" "$point" "$upper"
    sum_point="$(awk -v a="$sum_point" -v b="$point" 'BEGIN { printf "%.17g", a + b }')"
    sum_upper="$(awk -v a="$sum_upper" -v b="$upper" 'BEGIN { printf "%.17g", a + b }')"
    count=$((count + 1))
done < "$manifest"

if (( count == 0 )); then
    printf 'no Criterion change estimates listed for %s\n' "$suite" >&2
    exit 1
fi

mean_point="$(awk -v sum="$sum_point" -v n="$count" 'BEGIN { printf "%.17g", sum / n }')"
mean_upper="$(awk -v sum="$sum_upper" -v n="$count" 'BEGIN { printf "%.17g", sum / n }')"
printf '%s\t__SUITE_MEAN__\t%.9f\t%.9f\n' "$suite" "$mean_point" "$mean_upper"

if awk -v point="$mean_point" -v upper="$mean_upper" \
    'BEGIN { exit ! (point < 0.015 && upper < 0.03) }'; then
    printf 'PASS: %s has mean change %.4f%% and conservative upper95 %.4f%% across %d cases.\n' \
        "$suite" \
        "$(awk -v x="$mean_point" 'BEGIN { print 100*x }')" \
        "$(awk -v x="$mean_upper" 'BEGIN { print 100*x }')" \
        "$count"
else
    printf 'FAIL: %s has mean change %.4f%% and conservative upper95 %.4f%% across %d cases.\n' \
        "$suite" \
        "$(awk -v x="$mean_point" 'BEGIN { print 100*x }')" \
        "$(awk -v x="$mean_upper" 'BEGIN { print 100*x }')" \
        "$count" >&2
    exit 1
fi
