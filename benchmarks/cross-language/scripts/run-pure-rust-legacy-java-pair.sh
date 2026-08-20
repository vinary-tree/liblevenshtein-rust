#!/usr/bin/env bash
# Strict direct-language pair: pure Rust core versus legacy pure Java.
#
# This runner never builds. One child process contributes one sample, arms are
# adjacent and alternate first position, and every child is admitted before
# and after timing against the selected CPU's complete LLC-sharing group.
set -euo pipefail

resume=0
if [[ ${!#:-} == --resume ]]; then
    resume=1
    set -- "${@:1:$(($# - 1))}"
fi
if [[ $# -lt 7 || $# -gt 12 ]]; then
    echo "usage: $0 MODE RUST_BINARY JAVA_BINARY DICTIONARY QUERIES MANIFEST OUTPUT_DIR [SAMPLES] [CPU] [ALGORITHM] [DISTANCE] [RUST_CONSTRUCTOR] [--resume]" >&2
    exit 2
fi

mode=$1
rust_binary=$(realpath "$2")
java_binary=$(realpath "$3")
dictionary=$(realpath "$4")
queries=$(realpath "$5")
manifest=$(realpath "$6")
output_arg=$7
samples=${8:-51}
cpu=${9:-0}
algorithm=${10:-standard}
distance=${11:-2}
rust_constructor=${12:-from_terms}
warmup_seconds=${XL_PAIR_WARMUP_SECONDS:-3}

case $mode in query|construct) ;; *)
    echo "mode must be query or construct" >&2
    exit 2
esac
case $algorithm in standard|transposition|merge_and_split) ;; *)
    echo "legacy Java supports standard, transposition, or merge_and_split" >&2
    exit 2
esac
case $rust_constructor in from_terms|from_sorted_terms) ;; *)
    echo "Rust constructor must be from_terms or from_sorted_terms" >&2
    exit 2
esac
if ! [[ $samples =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ && $distance =~ ^[123]$ \
        && $warmup_seconds =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "samples must be positive, CPU non-negative, distance 1..3, and warmup seconds non-negative" >&2
    exit 2
fi
for executable in "$rust_binary" "$java_binary"; do
    [[ -x $executable ]] || { echo "not executable: $executable" >&2; exit 2; }
done
for input in "$dictionary" "$queries" "$manifest"; do
    [[ -f $input ]] || { echo "missing input: $input" >&2; exit 2; }
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
contention_monitor="$script_dir/run-with-contention-monitor.sh"
java_closure_helper="$script_dir/java-execution-closure.py"
pair_schema="$repo/benchmarks/cross-language/schema/pure-rust-legacy-java-pair.schema.json"
schema_check="$script_dir/schema_check.py"
output_parent=$(realpath -m "$(dirname "$output_arg")")
output="$output_parent/$(basename "$output_arg")"
load_ledger="$output/host-load-admission.jsonl"
load_rejections="$output/host-load-rejections.jsonl"
contention_ledger="$output/foreign-contention.jsonl"
contention_log="$output/contention-monitor.log"
run_config="$output/run-config.json"
summary="$output/summary.json"

rust_sha=$(sha256sum "$rust_binary" | awk '{print $1}')
java_sha=$(sha256sum "$java_binary" | awk '{print $1}')
dictionary_sha=$(sha256sum "$dictionary" | awk '{print $1}')
queries_sha=$(sha256sum "$queries" | awk '{print $1}')
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
java_closure_json=$(python3 "$java_closure_helper" "$java_binary")
java_closure_sha=$(jq -r '.closure_sha256' <<<"$java_closure_json")

verify_manifest_artifact() {
    local path=$1 actual=$2 relative expected
    relative=$(realpath --relative-to "$(dirname "$manifest")" "$path")
    if jq -e '.artifacts | type == "object"' "$manifest" >/dev/null 2>&1; then
        expected=$(jq -r --arg relative "$relative" '.artifacts[$relative].sha256 // empty' "$manifest")
        if [[ -z $expected || $expected != "$actual" ]]; then
            echo "manifest does not pin $relative to its current SHA-256" >&2
            exit 2
        fi
    fi
}
verify_manifest_artifact "$dictionary" "$dictionary_sha"
verify_manifest_artifact "$queries" "$queries_sha"

assert_inputs_unchanged() {
    [[ $(sha256sum "$rust_binary" | awk '{print $1}') == "$rust_sha" ]] || {
        echo "Rust binary changed during the run" >&2; exit 1; }
    [[ $(sha256sum "$java_binary" | awk '{print $1}') == "$java_sha" ]] || {
        echo "Java launcher changed during the run" >&2; exit 1; }
    [[ $(python3 "$java_closure_helper" "$java_binary" | jq -r '.closure_sha256') == "$java_closure_sha" ]] || {
        echo "Java executable, version, classpath, or classpath content changed during the run" >&2
        exit 1
    }
    [[ $(sha256sum "$dictionary" | awk '{print $1}') == "$dictionary_sha" ]] || {
        echo "dictionary changed during the run" >&2; exit 1; }
    [[ $(sha256sum "$queries" | awk '{print $1}') == "$queries_sha" ]] || {
        echo "queries changed during the run" >&2; exit 1; }
    [[ $(sha256sum "$manifest" | awk '{print $1}') == "$manifest_sha" ]] || {
        echo "manifest changed during the run" >&2; exit 1; }
}

mkdir -p "$output_parent"
if ((resume)); then
    [[ -d $output && -f $run_config && -d $output/pairs ]] || {
        echo "resume requires an initialized output directory" >&2; exit 2; }
    [[ ! -s $contention_ledger && ! -s $contention_log ]] || {
        echo "resume refuses evidence with recorded foreign contention" >&2; exit 2; }
    jq -e \
        --arg mode "$mode" --arg rust_binary "$rust_binary" --arg rust_sha "$rust_sha" \
        --arg java_binary "$java_binary" --arg java_sha "$java_sha" \
        --arg java_closure_sha "$java_closure_sha" \
        --arg dictionary "$dictionary" --arg dictionary_sha "$dictionary_sha" \
        --arg queries "$queries" --arg queries_sha "$queries_sha" \
        --arg manifest "$manifest" --arg manifest_sha "$manifest_sha" \
        --arg algorithm "$algorithm" --arg constructor "$rust_constructor" \
        --argjson samples "$samples" --argjson cpu "$cpu" --argjson distance "$distance" \
        --argjson warmup "$warmup_seconds" '
        .schema == "liblevenshtein.pure-rust-legacy-java-pair-config.v1"
        and .mode == $mode and .rust_binary == $rust_binary
        and .rust_binary_sha256 == $rust_sha and .java_binary == $java_binary
        and .java_binary_sha256 == $java_sha and .dictionary == $dictionary
        and .java_execution_closure.closure_sha256 == $java_closure_sha
        and .dictionary_sha256 == $dictionary_sha and .queries == $queries
        and .queries_sha256 == $queries_sha and .manifest == $manifest
        and .manifest_sha256 == $manifest_sha and .samples == $samples
        and .cpu == $cpu and .algorithm == $algorithm and .distance == $distance
        and .rust_constructor == $constructor and .warmup_seconds == $warmup
        ' "$run_config" >/dev/null || {
        echo "resume configuration or pinned digest mismatch" >&2
        exit 2
    }
else
    [[ ! -e $output ]] || { echo "output already exists: $output" >&2; exit 2; }
    mkdir -p "$output/pairs"
    jq -n \
        --arg mode "$mode" --arg rust_binary "$rust_binary" --arg rust_sha "$rust_sha" \
        --arg java_binary "$java_binary" --arg java_sha "$java_sha" \
        --argjson java_closure "$java_closure_json" \
        --arg dictionary "$dictionary" --arg dictionary_sha "$dictionary_sha" \
        --arg queries "$queries" --arg queries_sha "$queries_sha" \
        --arg manifest "$manifest" --arg manifest_sha "$manifest_sha" \
        --arg algorithm "$algorithm" --arg constructor "$rust_constructor" \
        --argjson samples "$samples" --argjson cpu "$cpu" --argjson distance "$distance" \
        --argjson warmup "$warmup_seconds" '
        {
          schema:"liblevenshtein.pure-rust-legacy-java-pair-config.v1",
          mode:$mode,rust_binary:$rust_binary,rust_binary_sha256:$rust_sha,
          java_binary:$java_binary,java_binary_sha256:$java_sha,
          java_execution_closure:$java_closure,
          dictionary:$dictionary,dictionary_sha256:$dictionary_sha,
          queries:$queries,queries_sha256:$queries_sha,
          manifest:$manifest,manifest_sha256:$manifest_sha,
          samples:$samples,cpu:$cpu,algorithm:$algorithm,distance:$distance,
          rust_constructor:$constructor,warmup_seconds:$warmup
        }' >"$run_config.tmp"
    mv "$run_config.tmp" "$run_config"
fi
config_sha=$(sha256sum "$run_config" | awk '{print $1}')

require_compiler_quiet() {
    local active
    active="$({ pgrep -ax cargo || true; pgrep -ax rustc || true; pgrep -ax cargo-llvm-cov || true; } | sort -u)"
    if [[ -n $active ]]; then
        printf 'compiler-load gate is red:\n%s\n' "$active" >&2
        exit 3
    fi
}

admit() {
    local label=$1 accepted_ledger=$2 attempt status
    attempt=$(mktemp "$output_parent/.host-load-attempt.XXXXXX")
    if "$host_gate" --cpu "$cpu" \
        --max-selected-busy 10 --max-sibling-busy 10 \
        --max-llc-mean-busy 10 --max-llc-peer-busy 20 \
        --label "$label" --output "$attempt" >/dev/null
    then
        cat "$attempt" >>"$accepted_ledger"
        rm -f "$attempt"
        return 0
    else
        status=$?
        # Rejections are evidence, but they are not admissions for a completed
        # pair. Keep them in a separate append-only ledger so a post-timing
        # rejection cannot make a later --resume mistake a discarded sample
        # for an accepted one.
        cat "$attempt" >>"$load_rejections"
        rm -f "$attempt"
        return "$status"
    fi
}

validate_pair_load_ledger() {
    local replicate=$1 ledger=$2
    [[ -s $ledger ]] || {
        echo "missing host-load admissions for replicate $replicate" >&2
        exit 2
    }
    jq -s -e --argjson replicate "$replicate" --argjson cpu "$cpu" '
        def expected_labels:
          ["rust", "java"][] as $arm
          | ["pre", "post"][] as $phase
          | "replicate-\($replicate)-\($arm)-\($phase)";
        length == 4
        and all(.[];
          .schema == "liblevenshtein.causal-host-load.v1"
          and .admitted == true
          and .selected_cpu == $cpu
          and .selected_cpus == [$cpu]
          and .thresholds.max_selected_busy_percent == 10
          and .thresholds.max_sibling_busy_percent == 10
          and .thresholds.max_llc_mean_busy_percent == 10
          and .thresholds.max_llc_peer_busy_percent == 20)
        and (([.[] | .label] | sort) == ([expected_labels] | sort))
        ' "$ledger" >/dev/null || {
        echo "host-load admissions are incomplete or invalid for replicate $replicate" >&2
        exit 2
    }
}

rebuild_load_ledger() {
    local expected_pairs=$1 replicate pair_dir
    if ((expected_pairs == 0)); then
        rm -f "$load_ledger"
        return
    fi
    : >"$load_ledger.tmp"
    for ((replicate = 1; replicate <= expected_pairs; replicate++)); do
        pair_dir=$(printf '%s/pairs/replicate-%06d' "$output" "$replicate")
        validate_pair_load_ledger "$replicate" "$pair_dir/host-load-admission.jsonl"
        cat "$pair_dir/host-load-admission.jsonl" >>"$load_ledger.tmp"
    done
    mv "$load_ledger.tmp" "$load_ledger"
}

validate_load_ledger() {
    local expected_pairs=$1
    rebuild_load_ledger "$expected_pairs"
    if ((expected_pairs == 0)); then
        return
    fi
    [[ -s $load_ledger ]] || {
        echo "missing host-load admission ledger" >&2
        exit 2
    }
    jq -s -e --argjson pairs "$expected_pairs" --argjson cpu "$cpu" '
        def expected_labels:
          [range(1; $pairs + 1) as $replicate
           | ["rust", "java"][] as $arm
           | ["pre", "post"][] as $phase
           | "replicate-\($replicate)-\($arm)-\($phase)"];
        length == ($pairs * 4)
        and all(.[];
          .schema == "liblevenshtein.causal-host-load.v1"
          and .admitted == true
          and .selected_cpu == $cpu
          and .selected_cpus == [$cpu]
          and .thresholds.max_selected_busy_percent == 10
          and .thresholds.max_sibling_busy_percent == 10
          and .thresholds.max_llc_mean_busy_percent == 10
          and .thresholds.max_llc_peer_busy_percent == 20)
        and (([.[] | .label] | sort) == (expected_labels | sort))
        ' "$load_ledger" >/dev/null || {
        echo "host-load admission ledger is incomplete, rejected, or inconsistent" >&2
        exit 2
    }
}

validate_rejection_ledger() {
    [[ -s $load_rejections ]] || return 0
    jq -s -e --argjson cpu "$cpu" '
        length > 0
        and all(.[];
          .schema == "liblevenshtein.causal-host-load.v1"
          and .admitted == false
          and .selected_cpu == $cpu
          and .selected_cpus == [$cpu]
          and .thresholds.max_selected_busy_percent == 10
          and .thresholds.max_sibling_busy_percent == 10
          and .thresholds.max_llc_mean_busy_percent == 10
          and .thresholds.max_llc_peer_busy_percent == 20
          and (.rejection_reasons | length) > 0)
        ' "$load_rejections" >/dev/null || {
        echo "host-load rejection ledger is malformed" >&2
        exit 2
    }
}

extract_signature() {
    local result=$1
    if [[ $mode == query ]]; then
        jq -S -c '.measurements | {
            matches_per_pass,term_bytes_per_pass,distance_sum_per_pass,checksum_hex
        }' "$result"
    else
        jq -S -c '.construct | {
            term_count,semantic_term_count,semantic_membership_checks,semantic_checksum_hex
        }' "$result"
    fi
}

run_arm() {
    local arm=$1 replicate=$2 pair_scratch=$3 destination log_file
    destination="$pair_scratch/$arm.json"
    log_file="$pair_scratch/$arm.log"
    local -a command
    if [[ $arm == rust ]]; then
        command=(taskset -c "$cpu" "$rust_binary" --mode "$mode"
            --algorithm "$algorithm" --max-distance "$distance"
            --dictionary "$dictionary" --queries "$queries"
            --backend dynamic_dawg --constructor "$rust_constructor" --out "$destination")
    else
        command=(taskset -c "$cpu" "$java_binary" --mode "$mode"
            --algorithm "$algorithm" --max-distance "$distance"
            --dictionary "$dictionary" --queries "$queries"
            --backend own --out "$destination")
    fi
    if [[ $mode == query ]]; then
        command+=(--samples 1 --warmup-seconds "$warmup_seconds")
    else
        command+=(--reps 1)
    fi
    require_compiler_quiet
    assert_inputs_unchanged
    admit "replicate-$replicate-$arm-pre" "$pair_scratch/host-load-admission.jsonl"
    XL_MONITOR_CPU=${XL_MONITOR_CPU:-31} \
    XL_CONTENTION_LEDGER="$contention_ledger" \
    XL_CONTENTION_LOG="$contention_log" \
        "$contention_monitor" "$output" -- "${command[@]}" >"$log_file" 2>&1
    require_compiler_quiet
    admit "replicate-$replicate-$arm-post" "$pair_scratch/host-load-admission.jsonl"
    assert_inputs_unchanged
    [[ -s $destination ]] || { echo "$arm produced no result" >&2; exit 1; }
    if [[ $mode == query ]]; then
        jq -e --arg arm "$arm" --arg constructor "$rust_constructor" '
            .status == "ok" and .mode == "query"
            and .protocol.samples_requested == 1
            and .measurements.sample_count == 1
            and (.measurements.samples_ns | length) == 1
            and .measurements.samples_ns[0] > 0
            and (.measurements.checksum_hex | test("^[0-9a-f]{16}$"))
            and (if $arm == "rust" then
                    .target.implementation == "rust-core"
                    and any(.notes[]; . == ("dynamic DAWG constructor: " + $constructor))
                 else .target.implementation == "legacy" end)
            ' "$destination" >/dev/null
    else
        jq -e --arg arm "$arm" --arg constructor "$rust_constructor" '
            .status == "ok" and .mode == "construct"
            and .construct.reps == 1 and (.construct.times_ns | length) == 1
            and .construct.times_ns[0] > 0
            and .construct.semantic_term_count == .construct.term_count
            and .construct.semantic_membership_checks == .construct.term_count
            and (.construct.semantic_checksum_hex | test("^[0-9a-f]{16}$"))
            and (if $arm == "rust" then
                    .target.implementation == "rust-core"
                    and any(.notes[]; . == ("dynamic DAWG constructor: " + $constructor))
                 else .target.implementation == "legacy" end)
            ' "$destination" >/dev/null
    fi
}

completed=0
expected_signature=""
shopt -s nullglob
pair_dirs=("$output"/pairs/replicate-*)
for pair_dir in "${pair_dirs[@]}"; do
    completed=$((completed + 1))
    expected=$(printf '%s/pairs/replicate-%06d' "$output" "$completed")
    [[ $pair_dir == "$expected" && -f $pair_dir/pair.json \
        && -f $pair_dir/rust.json && -f $pair_dir/java.json ]] || {
        echo "resume found a gap or incomplete pair at $pair_dir" >&2; exit 2; }
    jq -e --argjson replicate "$completed" --arg config_sha "$config_sha" \
        --arg rust_raw_sha "$(sha256sum "$pair_dir/rust.json" | awk '{print $1}')" \
        --arg java_raw_sha "$(sha256sum "$pair_dir/java.json" | awk '{print $1}')" '
        .schema == "liblevenshtein.pure-rust-legacy-java-pair.v1"
        and .replicate == $replicate and .run_config_sha256 == $config_sha
        and .rust.raw_sha256 == $rust_raw_sha and .java.raw_sha256 == $java_raw_sha
        and .exact_signature_equal == true
        ' "$pair_dir/pair.json" >/dev/null || {
        echo "resume pair digest or semantic validation failed: $pair_dir" >&2; exit 2; }
    rust_signature=$(extract_signature "$pair_dir/rust.json")
    java_signature=$(extract_signature "$pair_dir/java.json")
    pair_signature=$(jq -S -c '.signature' "$pair_dir/pair.json")
    [[ $rust_signature == "$java_signature" && $pair_signature == "$rust_signature" ]] || {
        echo "resume pair raw and recorded signatures disagree: $pair_dir" >&2
        exit 2
    }
    if [[ -z $expected_signature ]]; then
        expected_signature=$pair_signature
    elif [[ $pair_signature != "$expected_signature" ]]; then
        echo "resume signatures drift across replicates at $pair_dir" >&2
        exit 2
    fi
done
validate_load_ledger "$completed"
validate_rejection_ledger
if ((completed > samples)); then
    echo "output contains more complete pairs than requested" >&2
    exit 2
fi
if ((completed == samples)); then
    echo "requested sample count is already complete; validating evidence" >&2
fi

pair_scratch=""
trap '[[ -z ${pair_scratch:-} ]] || rm -rf -- "$pair_scratch"' EXIT
for ((replicate = completed + 1; replicate <= samples; replicate++)); do
    pair_scratch=$(mktemp -d "$output_parent/.pure-rust-java-pair.XXXXXX")
    if ((replicate % 2 == 1)); then arms=(rust java); else arms=(java rust); fi
    for arm in "${arms[@]}"; do
        run_arm "$arm" "$replicate" "$pair_scratch"
    done

    rust_raw_sha=$(sha256sum "$pair_scratch/rust.json" | awk '{print $1}')
    java_raw_sha=$(sha256sum "$pair_scratch/java.json" | awk '{print $1}')
    rust_signature=$(extract_signature "$pair_scratch/rust.json")
    java_signature=$(extract_signature "$pair_scratch/java.json")
    [[ $rust_signature == "$java_signature" ]] || {
        echo "Rust/Java exact semantic signature mismatch" >&2
        exit 1
    }
    signature_equal=true
    signature=$rust_signature
    if [[ -z $expected_signature ]]; then
        expected_signature=$signature
    elif [[ $signature != "$expected_signature" ]]; then
        echo "semantic signature drifted across replicates" >&2
        exit 1
    fi
    jq -n --slurpfile rust "$pair_scratch/rust.json" --slurpfile java "$pair_scratch/java.json" \
        --argjson replicate "$replicate" --arg mode "$mode" \
        --arg config_sha "$config_sha" --arg rust_sha "$rust_raw_sha" \
        --arg java_sha "$java_raw_sha" --argjson signature_equal "$signature_equal" \
        --argjson signature "$signature" '
        {
          schema:"liblevenshtein.pure-rust-legacy-java-pair.v1",
          replicate:$replicate,mode:$mode,run_config_sha256:$config_sha,
          exact_signature_equal:$signature_equal,
          signature:$signature,
          rust:{raw_file:"rust.json",raw_sha256:$rust_sha,
                elapsed_ns:(if $mode == "query" then $rust[0].measurements.samples_ns[0]
                            else $rust[0].construct.times_ns[0] end)},
          java:{raw_file:"java.json",raw_sha256:$java_sha,
                elapsed_ns:(if $mode == "query" then $java[0].measurements.samples_ns[0]
                            else $java[0].construct.times_ns[0] end)}
        }' >"$pair_scratch/pair.json"
    "$schema_check" "$pair_schema" "$pair_scratch/pair.json" >/dev/null
    destination=$(printf '%s/pairs/replicate-%06d' "$output" "$replicate")
    mv "$pair_scratch" "$destination"
    pair_scratch=""
    rebuild_load_ledger "$replicate"

    jq -s --arg mode "$mode" --arg config_sha "$config_sha" --argjson requested "$samples" '
        def median:
          sort as $s | ($s | length) as $n
          | if $n % 2 == 1 then $s[$n / 2 | floor]
            else ($s[$n / 2 - 1] + $s[$n / 2]) / 2 end;
        def stats:
          . as $values | ($values | median) as $median
          | {
              samples:($values | length),
              median_ns:$median,
              mean_ns:(($values | add) / ($values | length)),
              mad_ns:($values | map((. - $median) | if . < 0 then -. else . end) | median)
            };
        . as $pairs
        | $pairs[0].signature as $signature
        | if (($pairs | length) > 0
              and ($pairs | all(.[];
                   .mode == $mode
                   and .run_config_sha256 == $config_sha
                   and .exact_signature_equal == true
                   and .signature == $signature)))
          then . else error("pair signatures or run configuration drifted") end
        | ($pairs | map(.rust.elapsed_ns) | stats) as $rust
        | ($pairs | map(.java.elapsed_ns) | stats) as $java
        | {
          schema:"liblevenshtein.pure-rust-legacy-java-pair-summary.v1",
          mode:$mode,run_config_sha256:$config_sha,samples_requested:$requested,
          samples_completed:($pairs | length),exact_signature:$pairs[0].signature,
          rust:$rust,java:$java,
          java_over_rust_x:($java.median_ns / $rust.median_ns),
          rust_latency_change_percent:(100 * ($rust.median_ns / $java.median_ns - 1)),
          pairs:$pairs
        }' "$output"/pairs/replicate-*/pair.json >"$summary.tmp"
    mv "$summary.tmp" "$summary"
    echo "replicate $replicate/$samples complete" >&2
done

require_compiler_quiet
assert_inputs_unchanged
[[ $(sha256sum "$run_config" | awk '{print $1}') == "$config_sha" ]] || {
    echo "run configuration changed during the run" >&2
    exit 1
}
validate_load_ledger "$samples"
validate_rejection_ledger
[[ ! -s $contention_ledger && ! -s $contention_log ]] || {
    echo "foreign-contention evidence exists; refusing to declare the run complete" >&2
    exit 1
}
jq -e --argjson samples "$samples" --argjson expected_signature "$expected_signature" '
    .samples_requested == $samples
    and .samples_completed == $samples
    and (.pairs | length) == $samples
    and ([.pairs[].replicate] == [range(1; $samples + 1)])
    and all(.pairs[]; .signature == $expected_signature)
    and .exact_signature == $expected_signature
    ' "$summary" >/dev/null || {
    echo "final summary is incomplete or its signatures drifted" >&2
    exit 1
}
printf '%s\n' "$summary"
