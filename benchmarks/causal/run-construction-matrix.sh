#!/usr/bin/env bash
# Time uninstrumented construction over every generated corpus/order cell.

set -euo pipefail

resume=0
if [[ ${3:-} == --resume ]]; then
    resume=1
    set -- "$1" "$2"
fi
if [[ $# -ne 2 ]]; then
    printf 'usage: %s <corpus-manifest.json> <output-dir> [--resume]\n' "$0" >&2
    exit 2
fi

manifest=$(realpath "$1")
output_dir=$2
corpus_dir=$(dirname "$manifest")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary=${CAUSAL_CONSTRUCTION_BINARY:-$repo/target/release/causal_construction_bench}
cpu=${CAUSAL_PROFILE_CPU:-3}
warmups=${CAUSAL_CONSTRUCTION_WARMUPS:-5}
reps=${CAUSAL_CONSTRUCTION_REPS:-30}
host_gate="$repo/benchmarks/causal/host-load-admission.py"
contention_monitor="$repo/benchmarks/cross-language/scripts/run-with-contention-monitor.sh"
schema="$repo/benchmarks/causal/schemas/construction-matrix-cell.schema.json"
schema_check="$repo/benchmarks/cross-language/scripts/schema_check.py"
load_ledger="$output_dir/host-load.jsonl"
load_rejections="$output_dir/host-load-rejections.jsonl"
admissions_dir="$output_dir/admissions"
contention_ledger="$output_dir/foreign-contention.jsonl"
contention_log="$output_dir/contention-monitor.log"
run_config="$output_dir/run-config.json"

if ((resume)); then
    if [[ ! -d $output_dir || ! -f $output_dir/corpus-manifest.json || ! -f $run_config ]]; then
        printf 'run-construction-matrix: resume requires an initialized output directory\n' >&2
        exit 2
    fi
    if [[ -s $contention_ledger ]] \
        || { [[ -s $contention_log ]] \
            && grep -Eq 'PROTOCOL-VIOLATION|FOREIGN-CONTENTION|GUARD-ERROR' "$contention_log"; }; then
        printf 'run-construction-matrix: resume refuses recorded foreign contention\n' >&2
        exit 2
    fi
elif [[ -e $output_dir ]]; then
    printf 'run-construction-matrix: output already exists: %s\n' "$output_dir" >&2
    exit 2
fi
if [[ ! -x $binary ]]; then
    printf 'run-construction-matrix: build %s first\n' "$binary" >&2
    exit 2
fi
if ! [[ $cpu =~ ^[0-9]+$ && $warmups =~ ^[0-9]+$ && $reps =~ ^[1-9][0-9]*$ ]]; then
    printf 'run-construction-matrix: CPU and warmups must be non-negative integers; reps must be positive\n' >&2
    exit 2
fi

binary_sha=$(sha256sum "$binary" | awk '{print $1}')
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')

verify_pinned_inputs() {
    local dictionary=$1 dictionary_sha=$2 queries=$3 queries_sha=$4
    local current_manifest_sha current_dictionary_sha current_queries_sha
    current_manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
    if [[ $current_manifest_sha != "$manifest_sha" ]]; then
        printf 'run-construction-matrix: corpus manifest mutated during the run\n' >&2
        exit 1
    fi
    if [[ ! -f $corpus_dir/$dictionary || ! -f $corpus_dir/$queries ]]; then
        printf 'run-construction-matrix: corpus file disappeared: %s or %s\n' \
            "$dictionary" "$queries" >&2
        exit 1
    fi
    current_dictionary_sha=$(sha256sum "$corpus_dir/$dictionary" | awk '{print $1}')
    current_queries_sha=$(sha256sum "$corpus_dir/$queries" | awk '{print $1}')
    if [[ $current_dictionary_sha != "$dictionary_sha" ]]; then
        printf 'run-construction-matrix: dictionary digest mismatch for %s: expected %s, got %s\n' \
            "$dictionary" "$dictionary_sha" "$current_dictionary_sha" >&2
        exit 1
    fi
    if [[ $current_queries_sha != "$queries_sha" ]]; then
        printf 'run-construction-matrix: query digest mismatch for %s: expected %s, got %s\n' \
            "$queries" "$queries_sha" "$current_queries_sha" >&2
        exit 1
    fi
}

require_compiler_quiet() {
    local active
    active="$({ pgrep -ax cargo || true; pgrep -ax rustc || true; pgrep -ax cargo-llvm-cov || true; } | sort -u)"
    if [[ -n $active ]]; then
        printf 'run-construction-matrix: compiler-load gate is red:\n%s\n' "$active" >&2
        exit 3
    fi
}

verify_binary_unchanged() {
    local current_binary_sha
    current_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
    if [[ $current_binary_sha != "$binary_sha" ]]; then
        printf 'run-construction-matrix: benchmark binary mutated during the run: expected %s, got %s\n' \
            "$binary_sha" "$current_binary_sha" >&2
        exit 1
    fi
}
if ((resume)); then
    if [[ $(sha256sum "$output_dir/corpus-manifest.json" | awk '{print $1}') != "$manifest_sha" ]]; then
        printf 'run-construction-matrix: manifest changed since the interrupted run\n' >&2
        exit 2
    fi
    jq -e \
        --arg binary "$binary" --arg binary_sha "$binary_sha" --arg manifest_sha "$manifest_sha" \
        --argjson cpu "$cpu" --argjson warmups "$warmups" --argjson reps "$reps" '
        .schema == "liblevenshtein.java-parity-construction-run.v1"
        and .binary == $binary and .binary_sha256 == $binary_sha
        and .manifest_sha256 == $manifest_sha and .cpu == $cpu
        and .warmups == $warmups and .reps == $reps
    ' "$run_config" >/dev/null || {
        printf 'run-construction-matrix: resume configuration or binary digest mismatch\n' >&2
        exit 2
    }
else
    mkdir -p "$output_dir/cells" "$admissions_dir"
    cp "$manifest" "$output_dir/corpus-manifest.json"
    jq -n \
        --arg binary "$binary" --arg binary_sha "$binary_sha" --arg manifest_sha "$manifest_sha" \
        --argjson cpu "$cpu" --argjson warmups "$warmups" --argjson reps "$reps" \
        '{schema:"liblevenshtein.java-parity-construction-run.v1",binary:$binary,binary_sha256:$binary_sha,manifest_sha256:$manifest_sha,cpu:$cpu,warmups:$warmups,reps:$reps}' \
        >"$run_config"
fi

mkdir -p "$admissions_dir"

rebuild_load_ledger() {
    local admission temporary="$load_ledger.tmp"
    : >"$temporary"
    for admission in "$admissions_dir"/*.jsonl; do
        [[ -f $admission ]] || continue
        cat "$admission" >>"$temporary"
    done
    mv "$temporary" "$load_ledger"
}

validate_cell_admission() {
    local admission=$1 cell=$2
    jq -s -e --argjson cpu "$cpu" --arg pre "pre-$cell" --arg post "post-$cell" '
        length == 2
        and all(.[];
            .schema == "liblevenshtein.causal-host-load.v1"
            and .admitted == true
            and .selected_cpu == $cpu
            and .selected_cpus == [$cpu]
            and .thresholds.max_selected_busy_percent == 10
            and .thresholds.max_sibling_busy_percent == 10
            and .thresholds.max_llc_mean_busy_percent == 10
            and .thresholds.max_llc_peer_busy_percent == 20)
        and (([.[] | .label] | sort) == ([$pre, $post] | sort))
    ' "$admission" >/dev/null
}

rebuild_load_ledger

scratch=$(mktemp -d /tmp/liblev-construction-matrix.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

admit() {
    local label=$1 attempt_ledger=$2
    if "$host_gate" --cpu "$cpu" \
        --max-selected-busy 10 \
        --max-sibling-busy 10 \
        --max-llc-mean-busy 10 \
        --max-llc-peer-busy 20 \
        --label "$label" --output "$attempt_ledger" >/dev/null; then
        return 0
    fi
    tail -n 1 "$attempt_ledger" >>"$load_rejections"
    return 3
}

while IFS=$'\t' read -r name domain order term_count dictionary dictionary_sha queries queries_sha; do
    case $domain in byte|unicode|u64) ;; *)
        printf 'run-construction-matrix: unsupported unit domain: %s\n' "$domain" >&2
        exit 2
    esac
    # Shuffled corpora exercise the public unordered bulk constructor. The
    # incremental `stream` path is a different API with online publication
    # semantics and must not stand in for unordered bulk construction.
    constructor=from_terms
    if [[ $order == sorted ]]; then
        constructor=from_sorted_terms
    fi
    cell="${name}-${order}"
    destination="$output_dir/cells/$cell.json"
    admission_file="$admissions_dir/$cell.jsonl"
    verify_binary_unchanged
    verify_pinned_inputs "$dictionary" "$dictionary_sha" "$queries" "$queries_sha"
    if [[ -f $destination ]]; then
        jq -e \
            --arg cell "$cell" --arg domain "$domain" --arg order "$order" \
            --arg constructor "$constructor" --arg dictionary_sha "$dictionary_sha" \
            --arg queries "$queries" --arg queries_sha "$queries_sha" \
            --arg binary_sha "$binary_sha" --argjson term_count "$term_count" \
            --argjson warmups "$warmups" --argjson reps "$reps" '
            .schema == "liblevenshtein.causal-construction.v1"
            and .cell == $cell and .domain == $domain and .order == $order
            and .constructor == $constructor and .term_count == $term_count
            and .semantic_term_count == $term_count
            and .semantic_membership_checks == $term_count
            and (.semantic_checksum_hex | test("^[0-9a-f]{16}$"))
            and .dictionary_sha256 == $dictionary_sha and .binary_sha256 == $binary_sha
            and .queries == $queries and .queries_sha256 == $queries_sha
            and .warmups == $warmups and (.samples_ns | length) == $reps
            and all(.samples_ns[]; . > 0)
        ' "$destination" >/dev/null || {
            printf 'run-construction-matrix: invalid retained cell: %s\n' "$destination" >&2
            exit 2
        }
        "$schema_check" "$schema" "$destination" >/dev/null
        if [[ ! -f $admission_file ]] || ! validate_cell_admission "$admission_file" "$cell"; then
            printf 'run-construction-matrix: invalid retained admission: %s\n' \
                "$admission_file" >&2
            exit 2
        fi
        printf 'run-construction-matrix: retained complete cell %s\n' "$cell" >&2
        continue
    fi

    require_compiler_quiet
    verify_binary_unchanged
    cell_admission_attempt="$scratch/$cell.host-load.jsonl"
    rm -f "$cell_admission_attempt"
    admit "pre-$cell" "$cell_admission_attempt" || exit $?
    raw="$scratch/$cell.raw.json"
    XL_CONTENTION_LEDGER="$contention_ledger" \
        XL_CONTENTION_LOG="$contention_log" \
        "$contention_monitor" "$output_dir" -- \
        taskset --cpu-list "$cpu" "$binary" \
            --dictionary "$corpus_dir/$dictionary" \
            --domain "$domain" \
            --constructor "$constructor" \
            --warmups "$warmups" \
            --reps "$reps" >"$raw"
    require_compiler_quiet
    admit "post-$cell" "$cell_admission_attempt" || exit $?
    verify_binary_unchanged
    verify_pinned_inputs "$dictionary" "$dictionary_sha" "$queries" "$queries_sha"
    jq -e \
        --arg cell "$cell" --arg order "$order" --arg dictionary "$dictionary" \
        --arg dictionary_sha "$dictionary_sha" --arg binary_sha "$binary_sha" \
        --arg queries "$queries" --arg queries_sha "$queries_sha" \
        --argjson expected_terms "$term_count" --argjson expected_warmups "$warmups" \
        --argjson expected_reps "$reps" '
        if .schema != "liblevenshtein.causal-construction.v1"
           or .term_count != $expected_terms or .warmups != $expected_warmups
           or .semantic_term_count != $expected_terms
           or .semantic_membership_checks != $expected_terms
           or (.semantic_checksum_hex | test("^[0-9a-f]{16}$") | not)
           or (.samples_ns | length) != $expected_reps or any(.samples_ns[]; . <= 0)
        then error("construction cell contract failed")
        else . + {
            cell:$cell,order:$order,dictionary:$dictionary,
            dictionary_sha256:$dictionary_sha,queries:$queries,
            queries_sha256:$queries_sha,binary_sha256:$binary_sha
        }
        end
    ' "$raw" >"$destination.tmp"
    "$schema_check" "$schema" "$destination.tmp" >/dev/null
    mv "$destination.tmp" "$destination"
    validate_cell_admission "$cell_admission_attempt" "$cell"
    mv "$cell_admission_attempt" "$admission_file"
    rebuild_load_ledger
    printf 'run-construction-matrix: completed cell %s\n' "$cell" >&2
done < <(jq -r '.records[] | [.name, .unit_domain, .order, .term_count, .dictionary, .dictionary_sha256, .queries, .queries_sha256] | @tsv' "$manifest")

require_compiler_quiet
verify_binary_unchanged

expected_admissions=$(jq '.records | length * 2' "$manifest")
jq -s -e --argjson expected "$expected_admissions" '
    length == $expected and all(.[]; .admitted == true)
' "$load_ledger" >/dev/null || {
    printf 'run-construction-matrix: accepted admission ledger is incomplete\n' >&2
    exit 2
}
if [[ -s $load_rejections ]]; then
    jq -s -e 'length > 0 and all(.[]; .admitted == false)' \
        "$load_rejections" >/dev/null || {
        printf 'run-construction-matrix: rejection ledger is malformed\n' >&2
        exit 2
    }
fi
if [[ -s $contention_ledger ]] \
    || { [[ -s $contention_log ]] \
        && grep -Eq 'PROTOCOL-VIOLATION|FOREIGN-CONTENTION|GUARD-ERROR' "$contention_log"; }; then
    printf 'run-construction-matrix: foreign contention evidence exists\n' >&2
    exit 2
fi

jq -c '{cell:(input_filename | split("/") | last | rtrimstr(".json"))} + .' \
    "$output_dir"/cells/*.json \
    | jq -s \
        --arg binary_sha "$binary_sha" --arg manifest_sha "$manifest_sha" \
        --argjson cpu "$cpu" --argjson warmups "$warmups" --argjson reps "$reps" '
        def median:
            sort as $s
            | ($s | length) as $n
            | if $n % 2 == 1 then $s[$n / 2 | floor]
              else ($s[$n / 2 - 1] + $s[$n / 2]) / 2 end;
        {
            schema: "liblevenshtein.java-parity-construction-matrix.v1",
            binary_sha256: $binary_sha,
            manifest_sha256: $manifest_sha,
            cpu: $cpu,
            warmups: $warmups,
            reps: $reps,
            cells: map(. + {
                median_ns: (.samples_ns | median),
                min_ns: (.samples_ns | min),
                max_ns: (.samples_ns | max)
            } | del(.samples_ns))
        }
        ' >"$output_dir/summary.json.tmp"
mv "$output_dir/summary.json.tmp" "$output_dir/summary.json"
printf '%s\n' "$output_dir/summary.json"
