#!/usr/bin/env bash
# Reproducible academic benchmark runner for MSM and phonetic experiments.
#
# This script intentionally keeps corpora and outputs under target/ so that
# downloads are never staged in /tmp. Heavy Rust commands run through
# systemd-run by default to cap resident memory.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${ACADEMIC_BENCH_DIR:-$PROJECT_ROOT/target/academic-benchmarks}"
RESULTS_DIR="${ACADEMIC_RESULTS_DIR:-$BENCH_ROOT/results}"
MSM_DIR="$BENCH_ROOT/msm"
PHONETIC_DIR="$BENCH_ROOT/phonetic"
UCR_ARCHIVE_ZIP="${UCR_ARCHIVE_ZIP:-$MSM_DIR/Univariate2018_ts.zip}"
UCR_ARCHIVE_ROOT="${UCR_ARCHIVE_ROOT:-$MSM_DIR/Univariate_ts}"
CMUDICT_PATH="${CMUDICT_PATH:-$PHONETIC_DIR/cmudict.dict}"

MEMORY_MAX="${MEMORY_MAX:-4G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNCAPPED="${ALLOW_UNCAPPED:-0}"
ALLOW_HIGH_LOAD="${ALLOW_HIGH_LOAD:-0}"
LOAD_LIMIT_FACTOR="${LOAD_LIMIT_FACTOR:-0.75}"

MSM_MAX_CELLS="${MSM_MAX_CELLS:-1000000000}"
MSM_MAX_DATASETS="${MSM_MAX_DATASETS:-1000}"
PHONETIC_CASES="${PHONETIC_CASES:-2048}"
PHONETIC_RECALL_K="${PHONETIC_RECALL_K:-5}"
PHONETIC_MAX_DISTANCE="${PHONETIC_MAX_DISTANCE:-2}"

CMUDICT_URL="${CMUDICT_URL:-https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict}"
CMUDICT_SHA256_EXPECTED="${CMUDICT_SHA256_EXPECTED:-81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22}"

DEFAULT_UCR_ARCHIVE_URLS=(
    "https://timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
    "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
    "https://timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip"
    "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip"
)

usage() {
    cat <<'USAGE'
Usage: scripts/run-academic-benchmarks.sh <command>

Commands:
  all                 Download/prepare corpora and run MSM + phonetic benchmarks.
  msm-ucr             Run the UCR/UEA 2018 univariate archive exact 1-NN MSM benchmark.
  phonetic-cmudict    Run CMUdict homophone recall profiles and diagnostics.
  clean-raw           Remove downloaded/extracted corpora, preserving result artifacts.
  help                Show this help text.

Important environment variables:
  ACADEMIC_BENCH_DIR      Corpus/result root. Default: target/academic-benchmarks
  MEMORY_MAX              systemd-run MemoryMax. Default: 4G
  MEMORY_SWAP_MAX         systemd-run MemorySwapMax. Default: 0
  CARGO_BUILD_JOBS        Cargo parallelism for benchmark builds. Default: 1
  MSM_MAX_CELLS           Dataset selector for UCR archive. Default: 1000000000
  MSM_MAX_DATASETS        Dataset cap for UCR archive. Default: 1000
  PHONETIC_CASES          CMUdict homophone sample/corpus limit. Default: 2048
  PHONETIC_RECALL_K       Homophone recall cutoff. Default: 5
  PHONETIC_MAX_DISTANCE   Phonetic query edit-distance limit. Default: 2
  UCR_ARCHIVE_URL         Override UCR archive download URL.
  UCR_ARCHIVE_PATH        Use an existing local UCR archive zip instead of downloading.
  UCR_ARCHIVE_ROOT        Use an existing extracted UCR archive directory.
  CMUDICT_URL             Override CMUdict download URL.
  CMUDICT_PATH            Use an existing CMUdict dictionary path.
  DRY_RUN=1               Print benchmark commands without downloading or executing.
  ALLOW_HIGH_LOAD=1       Bypass the load guard for wall-time-producing runs.
  ALLOW_UNCAPPED=1        Run without systemd-run if systemd-run is unavailable.

Outputs:
  target/academic-benchmarks/results/

The script does not use /tmp for benchmark corpora or result artifacts.
USAGE
}

log() {
    printf '[academic-bench] %s\n' "$*" >&2
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        printf 'required command not found: %s\n' "$cmd" >&2
        exit 1
    fi
}

ensure_dirs() {
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    mkdir -p "$RESULTS_DIR" "$MSM_DIR" "$PHONETIC_DIR"
}

stamp() {
    date -u '+%Y%m%dT%H%M%SZ'
}

write_metadata() {
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN skip metadata write"
        return 0
    fi
    ensure_dirs
    local path="$RESULTS_DIR/run-metadata-$(stamp).txt"
    {
        printf 'timestamp_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'project_root=%s\n' "$PROJECT_ROOT"
        printf 'academic_bench_dir=%s\n' "$BENCH_ROOT"
        printf 'git_commit=%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)"
        printf 'git_branch=%s\n' "$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        printf 'git_dirty=%s\n' "$(git -C "$PROJECT_ROOT" diff --quiet 2>/dev/null && printf false || printf true)"
        printf 'rustc=%s\n' "$(rustc --version 2>/dev/null || true)"
        printf 'cargo=%s\n' "$(cargo --version 2>/dev/null || true)"
        printf 'memory_max=%s\n' "$MEMORY_MAX"
        printf 'memory_swap_max=%s\n' "$MEMORY_SWAP_MAX"
        printf 'cargo_build_jobs=%s\n' "$CARGO_BUILD_JOBS"
        printf 'msm_max_cells=%s\n' "$MSM_MAX_CELLS"
        printf 'msm_max_datasets=%s\n' "$MSM_MAX_DATASETS"
        printf 'phonetic_cases=%s\n' "$PHONETIC_CASES"
        printf 'phonetic_recall_k=%s\n' "$PHONETIC_RECALL_K"
        printf 'phonetic_max_distance=%s\n' "$PHONETIC_MAX_DISTANCE"
    } >"$path"
    log "wrote metadata: $path"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'sha256-unavailable'
    fi
}

verify_sha256_if_set() {
    local path="$1"
    local expected="$2"
    local label="$3"
    if [[ -z "$expected" ]]; then
        return 0
    fi
    local actual
    actual="$(sha256_of "$path")"
    if [[ "$actual" != "$expected" ]]; then
        printf '%s checksum mismatch: expected %s, got %s\n' "$label" "$expected" "$actual" >&2
        exit 1
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN download $url -> $dest"
        return 0
    fi
    require_cmd curl
    mkdir -p "$(dirname "$dest")"
    log "downloading $url"
    curl --fail --location --retry 3 --connect-timeout 15 --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

prepare_ucr_archive() {
    ensure_dirs
    if [[ -d "$UCR_ARCHIVE_ROOT" ]]; then
        log "using extracted UCR archive: $UCR_ARCHIVE_ROOT"
        return 0
    fi

    if [[ -n "${UCR_ARCHIVE_PATH:-}" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            log "DRY_RUN copy $UCR_ARCHIVE_PATH -> $UCR_ARCHIVE_ZIP"
        else
            cp "$UCR_ARCHIVE_PATH" "$UCR_ARCHIVE_ZIP"
        fi
    fi

    if [[ ! -f "$UCR_ARCHIVE_ZIP" ]]; then
        if [[ -n "${UCR_ARCHIVE_URL:-}" ]]; then
            download_file "$UCR_ARCHIVE_URL" "$UCR_ARCHIVE_ZIP"
        else
            local downloaded=0
            local url
            for url in "${DEFAULT_UCR_ARCHIVE_URLS[@]}"; do
                if download_file "$url" "$UCR_ARCHIVE_ZIP"; then
                    downloaded=1
                    break
                fi
                rm -f "$UCR_ARCHIVE_ZIP.part"
            done
            if [[ "$downloaded" != "1" ]]; then
                printf 'failed to download UCR archive; set UCR_ARCHIVE_URL, UCR_ARCHIVE_PATH, or UCR_ARCHIVE_ROOT\n' >&2
                exit 1
            fi
        fi
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN unzip $UCR_ARCHIVE_ZIP -> $MSM_DIR"
        return 0
    fi

    require_cmd unzip
    log "extracting UCR archive into $MSM_DIR"
    unzip -q -o "$UCR_ARCHIVE_ZIP" -d "$MSM_DIR"
    if [[ ! -d "$UCR_ARCHIVE_ROOT" ]]; then
        local found
        found="$(find "$MSM_DIR" -maxdepth 3 -type d -name Univariate_ts -print -quit)"
        if [[ -n "$found" && "$found" != "$UCR_ARCHIVE_ROOT" ]]; then
            ln -s "$found" "$UCR_ARCHIVE_ROOT"
        fi
    fi
    if [[ ! -d "$UCR_ARCHIVE_ROOT" ]]; then
        printf 'UCR archive extraction did not produce %s\n' "$UCR_ARCHIVE_ROOT" >&2
        exit 1
    fi
    printf 'ucr_archive_sha256=%s\n' "$(sha256_of "$UCR_ARCHIVE_ZIP")" >"$RESULTS_DIR/ucr-archive-checksum.txt"
}

prepare_cmudict() {
    ensure_dirs
    if [[ ! -f "$CMUDICT_PATH" ]]; then
        download_file "$CMUDICT_URL" "$CMUDICT_PATH"
    else
        log "using CMUdict: $CMUDICT_PATH"
    fi
    if [[ "$DRY_RUN" != "1" ]]; then
        verify_sha256_if_set "$CMUDICT_PATH" "$CMUDICT_SHA256_EXPECTED" "CMUdict"
        printf 'cmudict_sha256=%s\n' "$(sha256_of "$CMUDICT_PATH")" >"$RESULTS_DIR/cmudict-checksum.txt"
    fi
}

load_guard() {
    if [[ "$DRY_RUN" == "1" || "$ALLOW_HIGH_LOAD" == "1" ]]; then
        return 0
    fi
    local cores load
    cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || printf '0')"
    if awk -v load="$load" -v cores="$cores" -v factor="$LOAD_LIMIT_FACTOR" 'BEGIN { exit !(load > cores * factor) }'; then
        printf 'refusing wall-time-producing benchmark under high load: load=%s cores=%s factor=%s\n' "$load" "$cores" "$LOAD_LIMIT_FACTOR" >&2
        printf 'set ALLOW_HIGH_LOAD=1 only when you intentionally want to bypass this guard\n' >&2
        exit 1
    fi
}

run_to_file() {
    local output="$1"
    shift
    ensure_dirs
    load_guard
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN output=%s command=' "$output" >&2
        printf '%q ' "$@" >&2
        printf '\n' >&2
        return 0
    fi
    log "running -> $output"
    if command -v systemd-run >/dev/null 2>&1; then
        (
            cd "$PROJECT_ROOT"
            systemd-run --user --scope --quiet \
                -p "MemoryMax=$MEMORY_MAX" \
                -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
                env "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS" "$@"
        ) >"$output"
    elif [[ "$ALLOW_UNCAPPED" == "1" ]]; then
        (
            cd "$PROJECT_ROOT"
            env "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS" "$@"
        ) >"$output"
    else
        printf 'systemd-run is required for heavy benchmark commands; set ALLOW_UNCAPPED=1 to override\n' >&2
        exit 1
    fi
}

summarize_msm() {
    local input="$1"
    local output="$RESULTS_DIR/msm_ucr_archive_summary.tsv"
    awk -F, '
        BEGIN {
            print "datasets\ttotal\tmajority_correct\tmsm_correct\tmajority_accuracy\tmsm_accuracy\texact_evaluations\tlower_bound_pruned\tcutoff_abandoned\testimated_cells"
        }
        NR > 1 && $1 == "summary" {
            datasets++;
            cells += $6;
            majority += $7;
            msm += $8;
            total += $9;
            evals += $11;
            pruned += $12;
            abandoned += $13;
        }
        END {
            if (total > 0) {
                printf "%d\t%d\t%d\t%d\t%.12f\t%.12f\t%d\t%d\t%d\t%.0f\n",
                    datasets, total, majority, msm, majority / total, msm / total,
                    evals, pruned, abandoned, cells;
            }
        }
    ' "$input" >"$output"
    log "wrote MSM summary: $output"
}

summarize_phonetic() {
    local label="$1"
    local input="$2"
    local output="$RESULTS_DIR/phonetic_cmudict_summary.tsv"
    if [[ ! -f "$output" ]]; then
        printf 'profile\tsamples\tmatched_count_sum\trecall_at_k_mean\treciprocal_rank_mean\n' >"$output"
    fi
    awk -v label="$label" '
        function metric(key, regex, value) {
            regex = "\"" key "\":[^,}]+";
            if (match($0, regex)) {
                value = substr($0, RSTART + length(key) + 3, RLENGTH - length(key) - 3);
                return value + 0;
            }
            return 0;
        }
        /"workload":"cmudict_phonetic_homophones"/ && /"warmup":false/ {
            count++;
            matched += metric("matched_count");
            recall += metric("recall_at_k");
            reciprocal += metric("reciprocal_rank");
        }
        END {
            if (count > 0) {
                printf "%s\t%d\t%d\t%.15f\t%.15f\n",
                    label, count, matched, recall / count, reciprocal / count;
            }
        }
    ' "$input" >>"$output"
}

run_msm_ucr() {
    prepare_ucr_archive
    write_metadata
    local summary_csv="$RESULTS_DIR/msm_ucr_archive_summary_${MSM_MAX_CELLS}_${MSM_MAX_DATASETS}.csv"
    local results_csv="$RESULTS_DIR/msm_ucr_archive_1nn_${MSM_MAX_CELLS}_${MSM_MAX_DATASETS}.csv"

    run_to_file "$summary_csv" \
        cargo run --release --example msm_experiment -- \
        ucr-archive-summary 0 0 "$UCR_ARCHIVE_ROOT" "$MSM_MAX_CELLS" "$MSM_MAX_DATASETS"

    run_to_file "$results_csv" \
        cargo run --release --example msm_experiment -- \
        ucr-archive-1nn 0 0 "$UCR_ARCHIVE_ROOT" "$MSM_MAX_CELLS" "$MSM_MAX_DATASETS"

    if [[ "$DRY_RUN" != "1" ]]; then
        summarize_msm "$results_csv"
    fi
}

run_phonetic_profile() {
    local label="$1"
    local output="$2"
    shift 2
    run_to_file "$output" \
        cargo run --release --features phonetic-rules,embedded-rules --example scientific_eval -- \
        --workload cmudict-phonetic \
        --cmudict "$CMUDICT_PATH" \
        --corpus-limit "$PHONETIC_CASES" \
        --samples "$PHONETIC_CASES" \
        --warmups 0 \
        --max-distance "$PHONETIC_MAX_DISTANCE" \
        --recall-k "$PHONETIC_RECALL_K" \
        "$@"
    if [[ "$DRY_RUN" != "1" ]]; then
        summarize_phonetic "$label" "$output"
    fi
}

run_phonetic_diagnostic() {
    local output="$RESULTS_DIR/phonetic_cmudict_en_us_cmudict_diag_${PHONETIC_CASES}.jsonl"
    run_to_file "$output" \
        cargo run --release --features phonetic-rules,embedded-rules --example scientific_eval -- \
        --workload cmudict-phonetic-diagnostic \
        --cmudict "$CMUDICT_PATH" \
        --corpus-limit "$PHONETIC_CASES" \
        --diagnostic-limit "$PHONETIC_CASES" \
        --max-distance "$PHONETIC_MAX_DISTANCE" \
        --recall-k "$PHONETIC_RECALL_K" \
        --phonetic-dialect en-us-cmudict
    if [[ "$DRY_RUN" != "1" ]]; then
        sed -n 's/.*"root_cause":"\([^"]*\)".*/\1/p' "$output" \
            | sort \
            | uniq -c \
            >"$RESULTS_DIR/phonetic_cmudict_diagnostic_root_causes.txt"
        log "wrote phonetic diagnostic summary: $RESULTS_DIR/phonetic_cmudict_diagnostic_root_causes.txt"
    fi
}

run_phonetic_cmudict() {
    prepare_cmudict
    write_metadata
    rm -f "$RESULTS_DIR/phonetic_cmudict_summary.tsv"

    run_phonetic_profile \
        "zompist-default" \
        "$RESULTS_DIR/phonetic_cmudict_zompist_${PHONETIC_CASES}.jsonl" \
        --phonetic-dialect zompist-default

    run_phonetic_profile \
        "american+llev-homophones+names-before" \
        "$RESULTS_DIR/phonetic_cmudict_american_homophones_names_before_${PHONETIC_CASES}.jsonl" \
        --phonetic-rules-file data/rules/english/american.llev \
        --phonetic-rules-extension data/rules/english/homophones.llev \
        --phonetic-rules-extension data/rules/english/names.llev \
        --phonetic-rules-extension-order before

    run_phonetic_profile \
        "en-us-cmudict" \
        "$RESULTS_DIR/phonetic_cmudict_en_us_cmudict_${PHONETIC_CASES}.jsonl" \
        --phonetic-dialect en-us-cmudict

    run_phonetic_diagnostic
}

clean_raw() {
    log "removing downloaded and extracted corpora under $BENCH_ROOT"
    rm -rf "$UCR_ARCHIVE_ROOT" \
        "$UCR_ARCHIVE_ZIP" \
        "$UCR_ARCHIVE_ZIP.part" \
        "$MSM_DIR/Univariate2018_ts" \
        "$MSM_DIR/Archives" \
        "$CMUDICT_PATH" \
        "$CMUDICT_PATH.part"
}

main() {
    case "${1:-help}" in
        all)
            run_msm_ucr
            run_phonetic_cmudict
            ;;
        msm-ucr)
            run_msm_ucr
            ;;
        phonetic-cmudict)
            run_phonetic_cmudict
            ;;
        clean-raw)
            clean_raw
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
