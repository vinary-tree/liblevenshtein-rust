#!/usr/bin/env bash
set -euo pipefail

resume=0
if [[ ${!#:-} == --resume ]]; then
  resume=1
  set -- "${@:1:$(($# - 1))}"
fi
if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "usage: $0 BINARY OUTPUT_CSV [SAMPLES] [CPU] [HOT_OPERATIONS] [ZIPF_OPERATIONS] [--resume]" >&2
  exit 2
fi

binary=$1
output=$2
samples=${3:-51}
cpu=${4:-3}
hot_operations=${5:-100000}
zipf_operations=${6:-200000}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_gate="$repo/benchmarks/causal/host-load-admission.py"
contention_monitor="$repo/benchmarks/cross-language/scripts/run-with-contention-monitor.sh"
load_ledger="${output%.csv}-host-load.jsonl"
rejection_ledger="${output%.csv}-host-load-rejections.jsonl"
admission_dir="${output%.csv}-admissions"
legacy_load_archive="${output%.csv}-host-load-pre-transactional.jsonl"
contention_ledger="${output%.csv}-foreign-contention.jsonl"
contention_log="${output%.csv}-contention-monitor.log"

if [[ ! -x $binary ]]; then
  echo "binary is not executable: $binary" >&2
  exit 2
fi
if ((resume)); then
  if [[ ! -f $output || (! -d $admission_dir && ! -f $load_ledger) ]]; then
    echo "resume requires existing CSV and committed or legacy host-load evidence" >&2
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
if ! [[ $samples =~ ^[1-9][0-9]*$ && $cpu =~ ^[0-9]+$ && $hot_operations =~ ^[1-9][0-9]*$ && $zipf_operations =~ ^[1-9][0-9]*$ ]]; then
  echo "samples and operation counts must be positive integers; CPU must be non-negative" >&2
  exit 2
fi
if ((zipf_operations < 10)); then
  echo "ZIPF_OPERATIONS must be at least 10" >&2
  exit 2
fi

scratch=$(mktemp -d /tmp/liblev-query-cache-policy.XXXXXX)
output_transaction="${output}.transaction.$$"
ledger_transaction="${load_ledger}.transaction.$$"
trap 'rm -rf -- "$scratch"; rm -f -- "$output_transaction" "$ledger_transaction"' EXIT
mkdir -p "$admission_dir"
binary_sha=$(sha256sum "$binary" | awk '{print $1}')

run_replicate() {
  local replicate=$1
  local destination=$2
  XL_CONTENTION_LEDGER="$contention_ledger" \
    XL_CONTENTION_LOG="$contention_log" \
    "$contention_monitor" "$(dirname "$output")" -- taskset -c "$cpu" "$binary" \
    --replicate "$replicate" \
    --hot-operations "$hot_operations" \
    --zipf-operations "$zipf_operations" \
    --binary-sha "$binary_sha" \
    --no-header >"$destination"
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

validate_admission_replicate() {
  local evidence=$1 replicate=$2
  jq -s -e --argjson replicate "$replicate" --argjson cpu "$cpu" '
    def expected_labels:
      ["replicate-\($replicate)", "replicate-\($replicate)-post"];
    length == 2
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
    validate_admission_replicate "$evidence" "$replicate" || {
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
      validate_admission_replicate "$target" "$replicate" || {
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
      [last_admitted("replicate-\($replicate)"),
       last_admitted("replicate-\($replicate)-post")]
      | if any(.[]; . == null) then error("missing legacy admission row")
        else .[] end
    ' "$load_ledger" >"$transaction"
    validate_admission_replicate "$transaction" "$replicate" || {
      echo "legacy host-load ledger cannot prove replicate $replicate" >&2
      exit 2
    }
    mv "$transaction" "$target"
  done
}

commit_output_replicate() {
  local result=$1
  cp "$output" "$output_transaction"
  cat "$result" >>"$output_transaction"
  mv "$output_transaction" "$output"
}

first_sample=1
if ((resume)); then
  header=$($binary --header-only)
  if [[ $(head -n 1 "$output") != "$header" ]]; then
    echo "resume CSV header does not match this binary" >&2
    exit 2
  fi
  data_rows=$(($(wc -l <"$output") - 1))
  if ((data_rows < 0)); then
    echo "resume CSV has no header" >&2
    exit 2
  fi
  completed=$((data_rows / 20))
  trailing_rows=$((data_rows % 20))
  if ((trailing_rows > 0)); then
    head -n $((completed * 20 + 1)) "$output" >"$output_transaction"
    mv "$output_transaction" "$output"
    echo "discarded $trailing_rows uncommitted trailing CSV rows; retained $completed complete replicates" >&2
  fi
  awk -F, -v sha="$binary_sha" '
    NR == 1 { next }
    $2 < 1 || $2 != int((NR - 2) / 20) + 1 || $26 != sha {
      print "resume CSV replicate sequence or binary SHA mismatch" > "/dev/stderr"
      exit 1
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
  "$binary" --header-only >"$output_transaction"
  mv "$output_transaction" "$output"
fi

warmup_admissions="$scratch/warmup-admissions.jsonl"
: >"$warmup_admissions"
admit warmup "$warmup_admissions"
for warmup in 1 2 3; do
  run_replicate "$warmup" "$scratch/warmup-$warmup.csv"
done

for ((sample = first_sample; sample <= samples; sample++)); do
  replicate_admissions="$scratch/replicate-$sample-admissions.jsonl"
  : >"$replicate_admissions"
  admit "replicate-$sample" "$replicate_admissions"
  result="$scratch/replicate-$sample.csv"
  run_replicate "$sample" "$result"
  admit "replicate-$sample-post" "$replicate_admissions"
  if [[ $(wc -l <"$result") -ne 20 ]]; then
    echo "replicate $sample emitted an unexpected number of policy/workload rows" >&2
    exit 1
  fi
  awk -F, '
    $19 > $21 { print "resident entry bound exceeded: " $0 > "/dev/stderr"; exit 1 }
    $20 > $22 { print "resident weight bound exceeded: " $0 > "/dev/stderr"; exit 1 }
    {
      key = $2 SUBSEP $6
      if (key in checksum && checksum[key] != $25) {
        print "policy result checksum mismatch: " $0 > "/dev/stderr"
        exit 1
      }
      checksum[key] = $25
      count[key]++
    }
    END {
      for (key in count) {
        if (count[key] != 5) {
          print "missing policy row for replicate/workload" > "/dev/stderr"
          exit 1
        }
      }
    }
  ' "$result"
  validate_admission_replicate "$replicate_admissions" "$sample"
  commit_admissions "$replicate_admissions" "$sample"
  commit_output_replicate "$result"
  rebuild_accepted_ledger "$sample"
  echo "replicate $sample/$samples complete" >&2
done
