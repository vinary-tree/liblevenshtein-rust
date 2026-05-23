#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/docs/verification/FORMAL_VERIFICATION_MANIFEST.tsv"
ASSUMPTIONS="$ROOT/docs/verification/ASSUMPTIONS.tsv"
TLA_JAR="${TLA_JAR:-/home/dylon/.tla/tla2tools.jar}"
MODE="${1:-audit}"

GAP_RE='^\s*(Axiom\s+|Admitted\.|admit\.|Parameter\s+|Conjecture\s+|Hypothesis\s+)'
ADMIT_RE='^\s*(Admitted\.|admit\.)'
ASSUME_RE='^\s*(Axiom\s+|Parameter\s+|Conjecture\s+|Hypothesis\s+)'

profile_memory() {
  case "$1" in
    light) echo "8G" ;;
    standard) echo "32G" ;;
    heavy) echo "96G" ;;
    exceptional) echo "128G" ;;
    *) echo "32G" ;;
  esac
}

profile_memory_bytes() {
  case "$1" in
    # OCaml 5/Rocq reserves substantial virtual address space for domain
    # heaps even when actual RSS stays well below the cgroup MemoryMax.
    # prlimit --as is only a non-cgroup fallback, so keep it conservative
    # enough to allow Rocq startup while still bounding runaway processes.
    light) echo "$((64 * 1024 * 1024 * 1024))" ;;
    standard) echo "$((64 * 1024 * 1024 * 1024))" ;;
    heavy) echo "$((96 * 1024 * 1024 * 1024))" ;;
    exceptional) echo "$((128 * 1024 * 1024 * 1024))" ;;
    *) echo "$((32 * 1024 * 1024 * 1024))" ;;
  esac
}

profile_cpu() {
  case "$1" in
    light) echo "400%" ;;
    standard) echo "800%" ;;
    heavy|exceptional) echo "1200%" ;;
    *) echo "800%" ;;
  esac
}

profile_tasks() {
  case "$1" in
    light) echo "80" ;;
    standard) echo "160" ;;
    heavy|exceptional) echo "240" ;;
    *) echo "160" ;;
  esac
}

systemd_scope_available() {
  [[ "${FORMAL_VERIFY_DISABLE_SYSTEMD_SCOPE:-}" != "1" ]] || return 1
  command -v systemd-run >/dev/null 2>&1 || return 1
  systemd-run --user --scope --quiet \
    -p "MemoryMax=64M" \
    -p "CPUQuota=100%" \
    true >/dev/null 2>&1
}

run_capped() {
  local profile="$1"
  shift
  local mem mem_bytes cpu tasks
  mem="$(profile_memory "$profile")"
  mem_bytes="$(profile_memory_bytes "$profile")"
  cpu="$(profile_cpu "$profile")"
  tasks="$(profile_tasks "$profile")"

  if systemd_scope_available; then
    systemd-run --user --scope --quiet \
      -p "MemoryMax=$mem" \
      -p "CPUQuota=$cpu" \
      -p "IOWeight=40" \
      -p "TasksMax=$tasks" \
      /usr/bin/time -v "$@"
  elif command -v prlimit >/dev/null 2>&1; then
    echo "warning: systemd-run scope unavailable; using prlimit --as=$mem_bytes fallback" >&2
    prlimit --as="$mem_bytes" /usr/bin/time -v "$@"
  elif [[ "${FORMAL_VERIFY_ALLOW_UNCAPPED:-}" == "1" ]]; then
    echo "warning: running uncapped because FORMAL_VERIFY_ALLOW_UNCAPPED=1" >&2
    /usr/bin/time -v "$@"
  else
    echo "error: refusing uncapped proof/model run; systemd-run not available" >&2
    exit 2
  fi
}

trusted_files() {
  awk -F '\t' '
    $0 !~ /^#/ && NF >= 4 && $1 == "trusted" && $3 == "coq" { print $4 }
  ' "$MANIFEST"
}

trusted_entries() {
  awk -F '\t' '
    $0 !~ /^#/ && NF >= 4 && $1 == "trusted" && $3 == "coq" { print $2 "\t" $4 }
  ' "$MANIFEST"
}

allowed_symbols() {
  awk -F '\t' '
    $0 !~ /^#/ && NF >= 6 && $1 == "allowed" { print $3 }
  ' "$ASSUMPTIONS"
}

audit_manifest() {
  local missing=0
  while IFS=$'\t' read -r status profile kind path _rest; do
    [[ -z "${status:-}" || "$status" == \#* ]] && continue
    if [[ ! -e "$ROOT/$path" ]]; then
      echo "manifest missing path: $path" >&2
      missing=1
    fi
  done < "$MANIFEST"
  return "$missing"
}

audit_all_gaps() {
  echo "== Active proof gaps across docs/verification and rocq =="
  local tmp_hits
  tmp_hits="$(mktemp)"

  find "$ROOT/docs/verification" "$ROOT/rocq" -name '*.v' -print0 \
    | while IFS= read -r -d '' file; do
        awk -v file="$file" '
          {
            line = $0
            out = ""
            i = 1
            while (i <= length(line)) {
              two = substr(line, i, 2)
              if (depth > 0) {
                if (two == "(*") {
                  depth++
                  i += 2
                } else if (two == "*)") {
                  depth--
                  i += 2
                } else {
                  i++
                }
              } else if (two == "(*") {
                depth++
                i += 2
              } else {
                out = out substr(line, i, 1)
                i++
              }
            }
            if (out ~ /^[[:space:]]*(Axiom[[:space:]]+|Admitted\.|admit\.|Parameter[[:space:]]+|Conjecture[[:space:]]+|Hypothesis[[:space:]]+)/) {
              print file ":" NR ":" out
            }
          }
        ' "$file"
      done > "$tmp_hits"

  if [[ -s "$tmp_hits" ]]; then
    cat "$tmp_hits"
  else
    echo "No active proof gaps found."
  fi
  echo
  echo "== Gap counts by file =="
  cut -d: -f1 "$tmp_hits" \
    | cut -d: -f1 \
    | sed "s#^$ROOT/##" \
    | sort \
    | uniq -c \
    | sort -nr || true
  rm -f "$tmp_hits"
}

audit_gaps_tsv() {
  printf 'status\tkind\tsymbol\tpath\tline\tclassification\tnote\n'

  find "$ROOT/docs/verification" "$ROOT/rocq" -name '*.v' -print0 \
    | while IFS= read -r -d '' file; do
        awk -v file="$file" -v root="$ROOT/" '
          {
            line = $0
            out = ""
            i = 1
            while (i <= length(line)) {
              two = substr(line, i, 2)
              if (depth > 0) {
                if (two == "(*") {
                  depth++
                  i += 2
                } else if (two == "*)") {
                  depth--
                  i += 2
                } else {
                  i++
                }
              } else if (two == "(*") {
                depth++
                i += 2
              } else {
                out = out substr(line, i, 1)
                i++
              }
            }
            if (out ~ /^[[:space:]]*(Axiom[[:space:]]+|Admitted\.|admit\.|Parameter[[:space:]]+|Conjecture[[:space:]]+|Hypothesis[[:space:]]+)/) {
              rel = file
              sub(root, "", rel)
              kind = out
              sub(/^[[:space:]]*/, "", kind)
              sub(/[[:space:]].*/, "", kind)
              gsub(/\./, "", kind)

              symbol = "<anonymous>"
              if (kind != "Admitted" && kind != "admit") {
                symbol = out
                sub(/^[[:space:]]*(Axiom|Parameter|Conjecture|Hypothesis)[[:space:]]+/, "", symbol)
                sub(/[[:space:]:\(].*/, "", symbol)
              }

              printf "partial\t%s\t%s\t%s\t%d\tprove\tactive proof gap\n", kind, symbol, rel, NR
            }
          }
        ' "$file"
      done
}

check_trusted_no_admitted() {
  local failed=0
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    if rg -n "$ADMIT_RE" "$ROOT/$rel"; then
      echo "trusted file contains Admitted.: $rel" >&2
      failed=1
    fi
  done < <(trusted_files)
  return "$failed"
}

check_trusted_assumptions() {
  local tmp_allowed tmp_hits failed
  tmp_allowed="$(mktemp)"
  tmp_hits="$(mktemp)"
  failed=0
  allowed_symbols > "$tmp_allowed"

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    rg -n "$ASSUME_RE" "$ROOT/$rel" >> "$tmp_hits" || true
  done < <(trusted_files)

  if [[ -s "$tmp_hits" ]]; then
    while IFS= read -r hit; do
      local symbol
      symbol="$(sed -E 's/.*(Axiom|Parameter|Conjecture|Hypothesis)[[:space:]]+([^[:space:]:]+).*/\2/' <<<"$hit")"
      if ! grep -Fxq "$symbol" "$tmp_allowed"; then
        echo "unallowlisted trusted assumption: $hit" >&2
        failed=1
      fi
    done < "$tmp_hits"
  fi

  rm -f "$tmp_allowed" "$tmp_hits"
  return "$failed"
}

coq_compile_trusted() {
  while IFS=$'\t' read -r profile rel; do
    [[ -z "$rel" ]] && continue
    echo "== Coq trusted compile [$profile]: $rel =="
    coq_compile_file "$profile" "$rel"
  done < <(trusted_entries)
}

coq_compile_file() {
  local profile="$1"
  local rel="$2"
  case "$rel" in
    docs/verification/core/theories/*)
      local file="${rel#docs/verification/core/theories/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/core/theories' && coqc -Q . Liblevenshtein.Core '$file'"
      ;;
    docs/verification/articulatory/theories/*)
      local file="${rel#docs/verification/articulatory/theories/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/articulatory/theories' && coqc -R . Liblevenshtein.Articulatory '$file'"
      ;;
    docs/verification/wallbreaker/theories/*)
      local file="${rel#docs/verification/wallbreaker/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/wallbreaker' && coqc -R theories WallBreaker '$file'"
      ;;
    docs/verification/grammar/theories/*)
      local file="${rel#docs/verification/grammar/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/grammar' && coqc -Q ../core/theories Liblevenshtein.Core -Q theories Liblevenshtein.Grammar.Verification '$file'"
      ;;
    docs/verification/msm/theories/*)
      local file="${rel#docs/verification/msm/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/msm' && coqc -R theories Liblevenshtein.MSM '$file'"
      ;;
    docs/verification/phonetic/*)
      local file="${rel#docs/verification/phonetic/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/phonetic' && coqc -Q . PhoneticRewrites -R theories Liblevenshtein.Phonetic.Verification '$file'"
      ;;
    docs/verification/llre/theories/*)
      local file="${rel#docs/verification/llre/theories/}"
      run_capped "$profile" bash -lc "cd '$ROOT/docs/verification/llre/theories' && coqc -R . Liblevenshtein.LLRE '$file'"
      ;;
    rocq/liblevenshtein/*)
      local file="${rel#rocq/liblevenshtein/}"
      run_capped "$profile" bash -lc "cd '$ROOT/rocq/liblevenshtein' && coqc -R . LevensteinAutomata '$file'"
      ;;
    *)
      local dir file
      dir="$(dirname "$ROOT/$rel")"
      file="$(basename "$rel")"
      run_capped "$profile" bash -lc "cd '$dir' && coqc '$file'"
      ;;
  esac
}

tla_check() {
  if [[ ! -f "$TLA_JAR" ]]; then
    echo "error: TLA+ jar not found at $TLA_JAR" >&2
    exit 2
  fi

  local cfg module profile jvm_mem
  for cfg in "$ROOT"/docs/verification/tla/*.cfg; do
    module="${cfg%.cfg}.tla"
    profile="standard"
    [[ "$(basename "$cfg")" == "Subsumption.cfg" ]] && profile="light"
    jvm_mem="$(profile_memory "$profile")"
    echo "== TLC [$profile]: $(basename "$module") =="
    run_capped "$profile" java "-Xmx$jvm_mem" -jar "$TLA_JAR" -config "$cfg" "$module"
  done
}

case "$MODE" in
  audit)
    audit_manifest
    audit_all_gaps
    ;;
  audit-tsv)
    audit_manifest >/dev/null
    audit_gaps_tsv
    ;;
  trusted)
    audit_manifest
    check_trusted_no_admitted
    check_trusted_assumptions
    ;;
  coq-trusted)
    audit_manifest
    check_trusted_no_admitted
    check_trusted_assumptions
    coq_compile_trusted
    ;;
  coq-file)
    profile="${2:-standard}"
    rel="${3:-}"
    if [[ -z "$rel" ]]; then
      echo "usage: scripts/verify-formal.sh coq-file <profile> <path>" >&2
      exit 2
    fi
    coq_compile_file "$profile" "$rel"
    ;;
  tla)
    tla_check
    ;;
  all)
    "$0" trusted
    "$0" coq-trusted
    "$0" tla
    ;;
  *)
    cat >&2 <<USAGE
usage: scripts/verify-formal.sh [audit|audit-tsv|trusted|coq-trusted|coq-file|tla|all]

All proof/model execution is memory-capped with systemd-run unless
FORMAL_VERIFY_ALLOW_UNCAPPED=1 is set.
USAGE
    exit 2
    ;;
esac
