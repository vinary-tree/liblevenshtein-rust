#!/usr/bin/env bash
set -euo pipefail

resume=0
if [[ ${!#:-} == --resume ]]; then
  resume=1
  set -- "${@:1:$(($# - 1))}"
fi
if [[ $# -lt 3 || $# -gt 9 ]]; then
  echo "usage: $0 CONTROL_BINARY TREATMENT_BINARY OUTPUT_CSV [SAMPLES] [CPU] [TERMS] [QUERIES] [REPETITIONS] [DISTANCE] [--resume]" >&2
  exit 2
fi

control_binary=$1
treatment_binary=$2
output=$3
samples=${4:-51}
cpu=${5:-3}
terms=${6:-256}
queries=${7:-64}
repetitions=${8:-1}
distance=${9:-2}
expected_rows=150
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
contention_monitor="$repo/benchmarks/cross-language/scripts/run-with-contention-monitor.sh"
load_ledger="${output%.csv}-host-load.jsonl"
rejection_ledger="${output%.csv}-host-load-rejections.jsonl"
admission_dir="${output%.csv}-admissions"
legacy_load_archive="${output%.csv}-host-load-pre-transactional.jsonl"
contention_ledger="${output%.csv}-foreign-contention.jsonl"
contention_log="${output%.csv}-contention-monitor.log"

for binary in "$control_binary" "$treatment_binary"; do
  if [[ ! -x $binary ]]; then
    echo "binary is not executable: $binary" >&2
    exit 2
  fi
done
if ((resume)); then
  if [[ ! -f $output || (! -d $admission_dir && ! -f $load_ledger) ]]; then
    echo "resume requires an existing CSV and committed or legacy host-load evidence" >&2
    exit 2
  fi
  if [[ -s $contention_ledger ]]; then
    echo "resume refuses a run with recorded foreign contention" >&2
    exit 2
  fi
  if [[ -s $contention_log ]] \
      && grep -Eq 'PROTOCOL-VIOLATION|FOREIGN-CONTENTION|GUARD-ERROR' "$contention_log"; then
    echo "resume refuses fatal contention-monitor diagnostics" >&2
    exit 2
  fi
elif [[ -e $output || -e $load_ledger || -e $rejection_ledger || -e $admission_dir \
    || -e $legacy_load_archive \
    || -e $contention_ledger || -e $contention_log ]]; then
  echo "output evidence already exists for $output" >&2
  exit 2
fi
if ! [[ $samples =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ && $terms =~ ^[1-9][0-9]*$ && $queries =~ ^[1-9][0-9]*$ && $repetitions =~ ^[1-9][0-9]*$ && $distance =~ ^[0-9]+$ ]]; then
  echo "samples, terms, queries, and repetitions must be positive integers; CPU and distance must be non-negative" >&2
  exit 2
fi
if ((queries > terms || distance > 255)); then
  echo "queries must not exceed terms and distance must not exceed 255" >&2
  exit 2
fi

scratch=$(mktemp -d /tmp/liblev-backend-propagation.XXXXXX)
output_transaction="${output}.transaction.$$"
ledger_transaction="${load_ledger}.transaction.$$"
trap 'rm -rf -- "$scratch"; rm -f -- "$output_transaction" "$ledger_transaction"' EXIT
mkdir -p "$admission_dir"
control_sha=$(sha256sum "$control_binary" | awk '{print $1}')
treatment_sha=$(sha256sum "$treatment_binary" | awk '{print $1}')

require_compiler_quiet() {
  local active
  active="$({ pgrep -ax cargo || true; pgrep -ax rustc || true; pgrep -ax cargo-llvm-cov || true; } | sort -u)"
  if [[ -n $active ]]; then
    printf 'compiler-load gate is red:\n%s\n' "$active" >&2
    exit 3
  fi
}

assert_binaries_unchanged() {
  local current_control_sha current_treatment_sha
  current_control_sha=$(sha256sum "$control_binary" | awk '{print $1}')
  current_treatment_sha=$(sha256sum "$treatment_binary" | awk '{print $1}')
  if [[ $current_control_sha != "$control_sha" ]]; then
    echo "control binary changed during the run" >&2
    exit 1
  fi
  if [[ $current_treatment_sha != "$treatment_sha" ]]; then
    echo "treatment binary changed during the run" >&2
    exit 1
  fi
}

append_rejection() {
  local record=$1
  local transaction="${rejection_ledger}.transaction.$$"
  local combined="$scratch/rejection-merge-$$-$RANDOM.jsonl"
  if [[ -f $rejection_ledger ]]; then
    cp "$rejection_ledger" "$combined"
  else
    : >"$combined"
  fi
  cat "$record" >>"$combined"
  jq -s -c '
    unique_by([.timestamp_utc, .pid, .label, .admitted,
               (.rejection_reasons // [] | tostring)])[]
  ' "$combined" >"$transaction"
  rm -f "$combined"
  mv "$transaction" "$rejection_ledger"
}

admit() {
  local label=$1 transaction_ledger=$2
  local record="$scratch/admission-${label//[^a-zA-Z0-9_.-]/_}-$$-$RANDOM.jsonl"
  local gate_status
  if "$host_gate" --cpu "$cpu" \
    --max-selected-busy 10 \
    --max-sibling-busy 10 \
    --max-llc-mean-busy 10 \
    --max-llc-peer-busy 20 \
    --label "$label" --output "$record" >/dev/null; then
    jq -e --arg label "$label" '
      .schema == "liblevenshtein.causal-host-load.v1"
      and .label == $label and .admitted == true
    ' "$record" >/dev/null || {
      echo "host gate emitted malformed accepted evidence for $label" >&2
      return 1
    }
    cat "$record" >>"$transaction_ledger"
    rm -f "$record"
    return 0
  else
    gate_status=$?
  fi
  if [[ -s $record ]]; then
    append_rejection "$record"
  fi
  rm -f "$record"
  return "$gate_status"
}

validate_admission_pair() {
  local evidence=$1 replicate=$2
  jq -s -e --argjson replicate "$replicate" --argjson cpu "$cpu" '
    def expected_labels:
      ["replicate-\($replicate)-control",
       "replicate-\($replicate)-control-post",
       "replicate-\($replicate)-treatment",
       "replicate-\($replicate)-treatment-post"];
    length == 4
    and all(.[];
      .schema == "liblevenshtein.causal-host-load.v1"
      and .admitted == true and .selected_cpu == $cpu)
    and (([.[] | .label] | sort) == (expected_labels | sort))
  ' "$evidence" >/dev/null
}

committed_admission_path() {
  printf '%s/replicate-%06d.jsonl' "$admission_dir" "$1"
}

commit_admissions() {
  local evidence=$1 replicate=$2 target transaction
  target=$(committed_admission_path "$replicate")
  transaction="${target}.transaction.$$"
  cp "$evidence" "$transaction"
  mv "$transaction" "$target"
}

rebuild_accepted_ledger() {
  local completed=$1 replicate evidence
  : >"$ledger_transaction"
  for ((replicate = 1; replicate <= completed; replicate++)); do
    evidence=$(committed_admission_path "$replicate")
    [[ -f $evidence ]] || {
      echo "missing committed admission evidence for replicate $replicate" >&2
      exit 2
    }
    validate_admission_pair "$evidence" "$replicate" || {
      echo "invalid committed admission evidence for replicate $replicate" >&2
      exit 2
    }
    cat "$evidence" >>"$ledger_transaction"
  done
  mv "$ledger_transaction" "$load_ledger"
}

migrate_legacy_rejections() {
  [[ -f $load_ledger ]] || return 0
  local rejected="$scratch/legacy-host-load-rejections.jsonl"
  jq -c 'select(.admitted == false)' "$load_ledger" >"$rejected"
  if [[ -s $rejected ]]; then
    append_rejection "$rejected"
  fi
}

archive_legacy_load_ledger() {
  local completed=$1 first_committed
  first_committed=$(committed_admission_path 1)
  if [[ -f $load_ledger && ! -f $legacy_load_archive \
      && ($completed -eq 0 || ! -f $first_committed) ]]; then
    cp "$load_ledger" "${legacy_load_archive}.transaction.$$"
    mv "${legacy_load_archive}.transaction.$$" "$legacy_load_archive"
  fi
}

recover_committed_admissions() {
  local completed=$1 replicate target transaction
  ((completed == 0)) && return 0
  for ((replicate = 1; replicate <= completed; replicate++)); do
    target=$(committed_admission_path "$replicate")
    if [[ -f $target ]]; then
      validate_admission_pair "$target" "$replicate" || {
        echo "invalid committed admission evidence for replicate $replicate" >&2
        exit 2
      }
      continue
    fi
    [[ -f $load_ledger ]] || {
      echo "cannot recover admission evidence for replicate $replicate" >&2
      exit 2
    }
    transaction="${target}.transaction.$$"
    jq -s -c --argjson replicate "$replicate" '
      def last_admitted($label):
        map(select(.admitted == true and .label == $label)) | last;
      [last_admitted("replicate-\($replicate)-control"),
       last_admitted("replicate-\($replicate)-control-post"),
       last_admitted("replicate-\($replicate)-treatment"),
       last_admitted("replicate-\($replicate)-treatment-post")]
      | if any(.[]; . == null) then error("missing legacy admission row")
        else .[] end
    ' "$load_ledger" >"$transaction"
    validate_admission_pair "$transaction" "$replicate" || {
      echo "legacy host-load ledger cannot prove replicate $replicate" >&2
      exit 2
    }
    mv "$transaction" "$target"
  done
}

commit_output_pair() {
  local control=$1 treatment=$2
  cp "$output" "$output_transaction"
  cat "$control" >>"$output_transaction"
  cat "$treatment" >>"$output_transaction"
  mv "$output_transaction" "$output"
}

run_arm() {
  local binary=$1
  local sha=$2
  local arm=$3
  local profile=$4
  local replicate=$5
  local destination=$6
  require_compiler_quiet
  assert_binaries_unchanged
  XL_CONTENTION_LEDGER="$contention_ledger" \
    XL_CONTENTION_LOG="$contention_log" \
    "$contention_monitor" "$(dirname "$output")" -- \
      taskset -c "$cpu" "$binary" \
      --replicate "$replicate" \
      --arm "$arm" \
      --profile "$profile" \
      --terms "$terms" \
      --queries "$queries" \
      --repetitions "$repetitions" \
      --distance "$distance" \
      --binary-sha "$sha" \
      --no-header >"$destination"
  require_compiler_quiet
  assert_binaries_unchanged
}

validate_arm() {
  local result=$1
  if [[ $(wc -l <"$result") -ne $expected_rows ]]; then
    echo "unexpected row count in $result" >&2
    exit 1
  fi
  awk -F, -v expected="$expected_rows" '
    NF != 32 { print "unexpected column count: " $0 > "/dev/stderr"; exit 1 }
    $32 != expected { print "row contract mismatch: " $0 > "/dev/stderr"; exit 1 }
    $12 == "applicable" {
      if ($14 > $26 || $15 > $27 || $16 > $28 || $20 > $29 || ($24 != "" && $24 > $30)) {
        print "hard bound exceeded: " $0 > "/dev/stderr"; exit 1
      }
      if ($17 == "" || $18 == "" || $19 == "" || $20 == "" || $21 == "" || $22 == "" || $23 == "" || $25 == "") {
        print "applicable row is missing evidence: " $0 > "/dev/stderr"; exit 1
      }
    }
    $12 == "inapplicable" {
      if ($15 != 0 || $16 != 0 || $17 != "" || $18 != "" || $19 != "" || $20 != "" || $21 != "" || $22 != "" || $23 != "" || $24 != "" || $25 != 0) {
        print "inapplicable row contains fabricated measurements: " $0 > "/dev/stderr"; exit 1
      }
    }
    $12 != "applicable" && $12 != "inapplicable" {
      print "unknown applicability: " $0 > "/dev/stderr"; exit 1
    }
    {
      cell = $7 SUBSEP $9
      if (!(cell in cells)) {
        cell_count++
      }
      cells[cell]++
    }
    END {
      if (cell_count != 30) {
        print "matrix does not contain 30 family/domain cells" > "/dev/stderr"; exit 1
      }
      for (cell in cells) {
        if (cells[cell] != 5) {
          print "family/domain cell does not contain construction plus four query rows" > "/dev/stderr"; exit 1
        }
      }
    }
  ' "$result"
}

validate_pair() {
  local control=$1
  local treatment=$2
  awk -F, '
    FNR == NR {
      key = $7 SUBSEP $8 SUBSEP $9 SUBSEP $10 SUBSEP $11
      applicability[key] = $12
      reason[key] = $13
      result_count[key] = $16
      checksum[key] = $25
      seen[key] = 1
      next
    }
    {
      key = $7 SUBSEP $8 SUBSEP $9 SUBSEP $10 SUBSEP $11
      if (!(key in seen) || applicability[key] != $12 || reason[key] != $13 || result_count[key] != $16 || checksum[key] != $25) {
        print "control/treatment semantic mismatch: " $0 > "/dev/stderr"; exit 1
      }
      matched[key] = 1
    }
    END {
      for (key in seen) {
        if (!(key in matched)) {
          print "treatment is missing a control matrix cell" > "/dev/stderr"; exit 1
        }
      }
    }
  ' "$control" "$treatment"
}

first_sample=1
if ((resume)); then
  header=$("$treatment_binary" --header-only)
  if [[ $(head -n 1 "$output") != "$header" ]]; then
    echo "resume CSV header does not match the treatment binary" >&2
    exit 2
  fi
  data_rows=$(($(wc -l <"$output") - 1))
  rows_per_pair=$((expected_rows * 2))
  if ((data_rows < 0)); then
    echo "resume CSV has no header" >&2
    exit 2
  fi
  completed=$((data_rows / rows_per_pair))
  trailing_rows=$((data_rows % rows_per_pair))
  if ((trailing_rows > 0)); then
    head -n $((completed * rows_per_pair + 1)) "$output" >"$output_transaction"
    mv "$output_transaction" "$output"
    echo "discarded $trailing_rows uncommitted trailing CSV rows; retained $completed complete pairs" >&2
  fi
  awk -F, -v rows="$expected_rows" -v control_sha="$control_sha" -v treatment_sha="$treatment_sha" '
    NR == 1 { next }
    {
      offset = (NR - 2) % (rows * 2)
      expected_replicate = int((NR - 2) / (rows * 2)) + 1
      expected_arm = offset < rows ? "control" : "treatment"
      expected_profile = offset < rows ? "legacy-shared-kernels" : "treatment"
      expected_sha = offset < rows ? control_sha : treatment_sha
      if ($2 != expected_replicate || $3 != expected_arm || $4 != expected_profile || $31 != expected_sha || $32 != rows) {
        print "resume CSV replicate, arm, profile, binary SHA, or row contract mismatch" > "/dev/stderr"
        exit 1
      }
    }
  ' "$output"
  archive_legacy_load_ledger "$completed"
  migrate_legacy_rejections
  recover_committed_admissions "$completed"
  rebuild_accepted_ledger "$completed"
  if ((completed >= samples)); then
    echo "requested sample count is already complete; accepted ledger rebuilt" >&2
    exit 2
  fi
  first_sample=$((completed + 1))
else
  "$treatment_binary" --header-only >"$output_transaction"
  mv "$output_transaction" "$output"
fi

require_compiler_quiet
assert_binaries_unchanged
warmup_admissions="$scratch/warmup-admissions.jsonl"
: >"$warmup_admissions"
admit warmup-control "$warmup_admissions"
for warmup in 1 2 3; do
  run_arm "$control_binary" "$control_sha" control legacy-shared-kernels "$warmup" "$scratch/warmup-control-$warmup.csv"
done

require_compiler_quiet
assert_binaries_unchanged
admit warmup-treatment "$warmup_admissions"
for warmup in 1 2 3; do
  run_arm "$treatment_binary" "$treatment_sha" treatment treatment "$warmup" "$scratch/warmup-treatment-$warmup.csv"
done

for ((sample = first_sample; sample <= samples; sample++)); do
  if ((sample % 2 == 0)); then
    arms=(treatment control)
  else
    arms=(control treatment)
  fi
  pair_admissions="$scratch/replicate-$sample-admissions.jsonl"
  : >"$pair_admissions"
  for arm in "${arms[@]}"; do
    require_compiler_quiet
    assert_binaries_unchanged
    admit "replicate-$sample-$arm" "$pair_admissions"
    result="$scratch/replicate-$sample-$arm.csv"
    if [[ $arm == control ]]; then
      run_arm "$control_binary" "$control_sha" control legacy-shared-kernels "$sample" "$result"
    else
      run_arm "$treatment_binary" "$treatment_sha" treatment treatment "$sample" "$result"
    fi
    admit "replicate-$sample-$arm-post" "$pair_admissions"
    validate_arm "$result"
  done
  validate_pair "$scratch/replicate-$sample-control.csv" "$scratch/replicate-$sample-treatment.csv"
  validate_admission_pair "$pair_admissions" "$sample"
  commit_admissions "$pair_admissions" "$sample"
  commit_output_pair "$scratch/replicate-$sample-control.csv" "$scratch/replicate-$sample-treatment.csv"
  rebuild_accepted_ledger "$sample"
  echo "replicate $sample/$samples complete" >&2
done

require_compiler_quiet
assert_binaries_unchanged
