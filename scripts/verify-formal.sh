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

declare -A CLEANED_COQ_DIRS=()

validate_rss_override() {
  [[ -z "${FORMAL_VERIFY_RSS_MB:-}" ]] && return
  if [[ ! "$FORMAL_VERIFY_RSS_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: FORMAL_VERIFY_RSS_MB must be a positive integer number of MiB" >&2
    exit 2
  fi
  if [[ -n "${FORMAL_VERIFY_RSS_GUARD_PERCENT:-}" ]] &&
     ! [[ "$FORMAL_VERIFY_RSS_GUARD_PERCENT" =~ ^[1-9][0-9]?$|^100$ ]]; then
    echo "error: FORMAL_VERIFY_RSS_GUARD_PERCENT must be an integer from 1 to 100" >&2
    exit 2
  fi
}

validate_rss_override

rss_override_mb() {
  [[ -n "${FORMAL_VERIFY_RSS_MB:-}" ]] || return 1
  echo "$FORMAL_VERIFY_RSS_MB"
}

profile_memory() {
  local rss_mb
  if rss_mb="$(rss_override_mb)"; then
    echo "${rss_mb}M"
    return
  fi
  case "$1" in
    light) echo "2G" ;;
    standard) echo "4G" ;;
    heavy) echo "12G" ;;
    exceptional) echo "24G" ;;
    *) echo "4G" ;;
  esac
}

profile_rss_kb() {
  local rss_mb
  if rss_mb="$(rss_override_mb)"; then
    echo "$((rss_mb * 1024))"
    return
  fi
  case "$1" in
    light) echo "$((2 * 1024 * 1024))" ;;
    standard) echo "$((4 * 1024 * 1024))" ;;
    heavy) echo "$((12 * 1024 * 1024))" ;;
    exceptional) echo "$((24 * 1024 * 1024))" ;;
    *) echo "$((4 * 1024 * 1024))" ;;
  esac
}

profile_memory_bytes() {
  case "$1" in
    # OCaml 5/Rocq reserves substantial virtual address space for domain
    # heaps even when actual RSS stays well below the cgroup MemoryMax.
    # prlimit --as is not an RSS limit. Keep it large enough for Rocq startup;
    # run_rss_monitored below enforces the actual RSS cap when systemd-run is
    # unavailable.
    light) echo "$((64 * 1024 * 1024 * 1024))" ;;
    standard) echo "$((64 * 1024 * 1024 * 1024))" ;;
    heavy) echo "$((96 * 1024 * 1024 * 1024))" ;;
    exceptional) echo "$((128 * 1024 * 1024 * 1024))" ;;
    *) echo "$((32 * 1024 * 1024 * 1024))" ;;
  esac
}

rss_tree_kb() {
  local root="$1"
  ps -eo pid=,ppid=,rss= | awk -v root="$root" '
    {
      pid = $1
      parent[pid] = $2
      rss[pid] = $3
      pids[++n] = pid
    }
    function is_descendant(pid, cur, seen) {
      cur = pid
      seen = 0
      while (cur != "" && cur != 0 && seen++ < 4096) {
        if (cur == root) return 1
        cur = parent[cur]
      }
      return 0
    }
    END {
      total = 0
      for (i = 1; i <= n; i++) {
        if (is_descendant(pids[i])) total += rss[pids[i]]
      }
      print total + 0
    }'
}

terminate_process_tree() {
  local pid="$1"
  if command -v setsid >/dev/null 2>&1; then
    kill -TERM "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
  else
    kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
}

run_rss_monitored() {
  local profile="$1"
  shift
  local rss_limit_kb rss_guard_percent rss_kill_kb exceeded_file pid monitor status
  rss_limit_kb="$(profile_rss_kb "$profile")"
  rss_guard_percent="${FORMAL_VERIFY_RSS_GUARD_PERCENT:-80}"
  rss_kill_kb="$((rss_limit_kb * rss_guard_percent / 100))"
  [[ "$rss_kill_kb" -gt 0 ]] || rss_kill_kb="$rss_limit_kb"
  exceeded_file="$(mktemp)"
  rm -f "$exceeded_file"

  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    "$@" &
  fi
  pid="$!"

  (
    local rss_kb
    while kill -0 "$pid" >/dev/null 2>&1; do
      rss_kb="$(rss_tree_kb "$pid")"
      if [[ "$rss_kb" -gt "$rss_kill_kb" ]]; then
        printf '%s\n' "$rss_kb" > "$exceeded_file"
        echo "error: RSS guard exceeded for profile '$profile': ${rss_kb}KiB > ${rss_kill_kb}KiB (${rss_guard_percent}% of ${rss_limit_kb}KiB cap)" >&2
        terminate_process_tree "$pid"
        break
      fi
      sleep "${FORMAL_VERIFY_RSS_POLL_SECONDS:-0.05}"
    done
  ) &
  monitor="$!"

  status=0
  wait "$pid" || status="$?"
  kill "$monitor" >/dev/null 2>&1 || true
  wait "$monitor" >/dev/null 2>&1 || true

  if [[ -s "$exceeded_file" ]]; then
    rm -f "$exceeded_file"
    return 137
  fi
  rm -f "$exceeded_file"
  return "$status"
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
    echo "warning: systemd-run scope unavailable; using prlimit --as=$mem_bytes plus RSS monitor cap=$mem fallback" >&2
    run_rss_monitored "$profile" prlimit --as="$mem_bytes" /usr/bin/time -v "$@"
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

contract_symbol_allowed() {
  local symbol="$1"
  allowed_symbols | grep -Fxq "$symbol"
}

evidence_symbol_allowed() {
  local symbol="$1"
  allowed_symbols | grep -Fxq "$symbol"
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

# Advisory: flag statements whose conclusion is the trivial proposition True.
# The gap audit (GAP_RE) only catches Axiom/Admitted/admit/Parameter/etc.; a
# theorem can still be vacuous if its goal is True (e.g. "... -> True", a bare
# "True." goal closed by trivial, "... /\ True", or ":= True"). Such proofs pass
# the gap audit while proving nothing, so they are surfaced here for review.
audit_vacuous() {
  echo "== Vacuous / placeholder (True) conclusions across docs/verification and rocq =="
  echo "(advisory) These pass the gap audit but prove nothing; replace with a real"
  echo "statement or remove."
  echo
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
                if (two == "(*") { depth++; i += 2 }
                else if (two == "*)") { depth--; i += 2 }
                else { i++ }
              } else if (two == "(*") { depth++; i += 2 }
              else { out = out substr(line, i, 1); i++ }
            }
            if (out ~ /(\/\\|->|\\\/)[[:space:]]*True([^[:alnum:]_]|$)/ \
                || out ~ /^[[:space:]]*True\.[[:space:]]*$/ \
                || out ~ /:=[[:space:]]*True([^[:alnum:]_]|$)/) {
              print file ":" NR ":" out
            }
          }
        ' "$file"
      done > "$tmp_hits"

  if [[ -s "$tmp_hits" ]]; then
    cat "$tmp_hits"
  else
    echo "No vacuous placeholder conclusions found."
  fi
  echo
  echo "== Vacuous-conclusion counts by file =="
  cut -d: -f1 "$tmp_hits" \
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

audit_contracts_tsv() {
  printf 'status\tkind\tsymbol\tpath\tline\tclassification\tnote\n'

  find "$ROOT/docs/verification" "$ROOT/rocq" -name '*.v' -print0 \
    | while IFS= read -r -d '' file; do
        awk -v file="$file" -v root="$ROOT/" '
          function emit(kind, symbol, note) {
            rel = file
            sub(root, "", rel)
            status = "partial"
            classification = "prove"
            printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\n",
              status, kind, symbol, rel, NR, classification, note
          }
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

            if (out ~ /^[[:space:]]*Definition[[:space:]]+[A-Za-z0-9_]*_contract[[:space:]]+/) {
              symbol = out
              sub(/^[[:space:]]*Definition[[:space:]]+/, "", symbol)
              sub(/[[:space:]:\(].*/, "", symbol)
              emit("Definition", symbol, "explicit proof contract")
            } else if (out ~ /^[[:space:]]*Record[[:space:]]+[A-Za-z0-9_]*Contracts[[:space:]]*[:{]/) {
              symbol = out
              sub(/^[[:space:]]*Record[[:space:]]+/, "", symbol)
              sub(/[[:space:]:{].*/, "", symbol)
              emit("Record", symbol, "record of explicit proof contracts")
            } else if (out ~ /^[[:space:]]*[A-Za-z0-9_]+_contract[[:space:]]*:/) {
              symbol = out
              sub(/^[[:space:]]*/, "", symbol)
              sub(/[[:space:]:].*/, "", symbol)
              emit("Field", symbol, "contract field in proof-obligation record")
            } else if (out ~ /^[[:space:]]*[A-Za-z0-9_]+_ax[[:space:]]*:/) {
              symbol = out
              sub(/^[[:space:]]*/, "", symbol)
              sub(/[[:space:]:].*/, "", symbol)
              emit("Field", symbol, "axiom-shaped field in proof-obligation record")
            } else if (out ~ /^[[:space:]]*(Lemma|Theorem|Corollary)[[:space:]]+[A-Za-z0-9_]+_ax[[:space:]]*:/) {
              symbol = out
              sub(/^[[:space:]]*(Lemma|Theorem|Corollary)[[:space:]]+/, "", symbol)
              sub(/[[:space:]:].*/, "", symbol)
              emit("NamedTheorem", symbol, "axiom-shaped theorem name")
            }
          }
        ' "$file"
      done \
    | while IFS=$'\t' read -r status kind symbol path line classification note; do
        if contract_symbol_allowed "$symbol"; then
          printf 'allowed\t%s\t%s\t%s\t%s\tassumption\tallowlisted cited contract\n' \
            "$kind" "$symbol" "$path" "$line"
        else
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$status" "$kind" "$symbol" "$path" "$line" "$classification" "$note"
        fi
      done
}

audit_contracts() {
  echo "== Explicit proof contracts across docs/verification and rocq =="
  local tmp_hits
  tmp_hits="$(mktemp)"
  audit_contracts_tsv | awk -F '\t' 'NR > 1 && $1 != "allowed" { print }' > "$tmp_hits"
  if [[ -s "$tmp_hits" ]]; then
    cat "$tmp_hits"
  else
    echo "No unallowlisted explicit proof contracts found."
  fi
  echo
  echo "== Contract counts by file =="
  cut -f4 "$tmp_hits" | sort | uniq -c | sort -nr || true
  rm -f "$tmp_hits"
}

audit_evidence_tsv() {
  printf 'status\tkind\tsymbol\tpath\tline\tclassification\tnote\n'

  find "$ROOT/docs/verification" "$ROOT/rocq" -name '*.v' -print0 \
    | while IFS= read -r -d '' file; do
        awk -v file="$file" -v root="$ROOT/" '
          function emit(kind, symbol, note) {
            rel = file
            sub(root, "", rel)
            printf "partial\t%s\t%s\t%s\t%d\tprove\t%s\n",
              kind, symbol, rel, NR, note
          }
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

            if (out ~ /^[[:space:]]*Record[[:space:]]+[A-Za-z0-9_]*Evidence[[:space:]]*[:{]/) {
              symbol = out
              sub(/^[[:space:]]*Record[[:space:]]+/, "", symbol)
              sub(/[[:space:]:{].*/, "", symbol)
              emit("Record", symbol, "neutral evidence record")
              in_evidence_record = 1
            } else if (out ~ /^[[:space:]]*Definition[[:space:]]+[A-Za-z0-9_]*_premise[[:space:]]*[:=]/) {
              symbol = out
              sub(/^[[:space:]]*Definition[[:space:]]+/, "", symbol)
              sub(/[[:space:]:\(].*/, "", symbol)
              emit("Definition", symbol, "explicit premise parameter")
            } else if (in_evidence_record && out ~ /^[[:space:]]*[A-Za-z0-9_]+[[:space:]]*:/) {
              symbol = out
              sub(/^[[:space:]]*/, "", symbol)
              sub(/[[:space:]:].*/, "", symbol)
              emit("Field", symbol, "field in evidence record")
            } else if (out ~ /^[[:space:]]*[A-Za-z0-9_]+_(proof|bridge|premise)[[:space:]]*:/) {
              symbol = out
              sub(/^[[:space:]]*/, "", symbol)
              sub(/[[:space:]:].*/, "", symbol)
              emit("Field", symbol, "field in evidence or premise record")
            }

            if (in_evidence_record && out ~ /^[[:space:]]*}\./) {
              in_evidence_record = 0
            }
          }
        ' "$file"
      done \
    | while IFS=$'\t' read -r status kind symbol path line classification note; do
        if evidence_symbol_allowed "$symbol"; then
          printf 'allowed\t%s\t%s\t%s\t%s\tassumption\tallowlisted cited evidence\n' \
            "$kind" "$symbol" "$path" "$line"
        else
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$status" "$kind" "$symbol" "$path" "$line" "$classification" "$note"
        fi
      done
}

audit_evidence() {
  echo "== Explicit evidence and premise surfaces across docs/verification and rocq =="
  local tmp_hits
  tmp_hits="$(mktemp)"
  audit_evidence_tsv | awk -F '\t' 'NR > 1 && $1 != "allowed" { print }' > "$tmp_hits"
  if [[ -s "$tmp_hits" ]]; then
    cat "$tmp_hits"
  else
    echo "No unallowlisted evidence or premise surfaces found."
  fi
  echo
  echo "== Evidence counts by file =="
  cut -f4 "$tmp_hits" | sort | uniq -c | sort -nr || true
  rm -f "$tmp_hits"
}

check_trusted_evidence() {
  local tmp_hits failed
  tmp_hits="$(mktemp)"
  failed=0
  audit_evidence_tsv | awk -F '\t' 'NR > 1 && $1 != "allowed" { print }' > "$tmp_hits"

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    local matches
    matches="$(awk -F '\t' -v rel="$rel" '$4 == rel { print }' "$tmp_hits")"
    if [[ -n "$matches" ]]; then
      printf '%s\n' "$matches"
      echo "trusted file contains unallowlisted evidence or premise surface: $rel" >&2
      failed=1
    fi
  done < <(trusted_files)

  rm -f "$tmp_hits"
  return "$failed"
}

check_trusted_contracts() {
  local tmp_hits failed
  tmp_hits="$(mktemp)"
  failed=0
  audit_contracts_tsv | awk -F '\t' 'NR > 1 && $1 != "allowed" { print }' > "$tmp_hits"

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    local matches
    matches="$(awk -F '\t' -v rel="$rel" '$4 == rel { print }' "$tmp_hits")"
    if [[ -n "$matches" ]]; then
      printf '%s\n' "$matches"
      echo "trusted file contains unallowlisted explicit proof contract: $rel" >&2
      failed=1
    fi
  done < <(trusted_files)

  rm -f "$tmp_hits"
  return "$failed"
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

clean_coq_artifacts_under() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  find "$dir" -type f \( \
    -name '*.vo' -o \
    -name '*.vos' -o \
    -name '*.vok' -o \
    -name '*.glob' -o \
    -name '.*.aux' -o \
    -name '.lia.cache' -o \
    -name '.nia.cache' -o \
    -name '.nra.cache' -o \
    -name 'Makefile.coq' -o \
    -name 'Makefile.coq.conf' -o \
    -name '.Makefile.coq.d' \
  \) -delete
}

clean_coq_artifacts_once() {
  local dir="$1"
  [[ -z "${CLEANED_COQ_DIRS[$dir]:-}" ]] || return 0
  clean_coq_artifacts_under "$dir"
  CLEANED_COQ_DIRS["$dir"]=1
}

clean_trusted_coq_artifacts() {
  clean_coq_artifacts_once "$ROOT/docs/verification/core/theories"
  clean_coq_artifacts_once "$ROOT/docs/verification/articulatory/theories"
  clean_coq_artifacts_once "$ROOT/docs/verification/wallbreaker"
}

coq_compile_trusted() {
  while IFS=$'\t' read -r profile rel; do
    [[ -z "$rel" ]] && continue
    echo "== Coq trusted compile [$profile]: $rel =="
    coq_compile_file "$profile" "$rel"
  done < <(trusted_entries)
}

coq_project_target() {
  local profile="$1"
  local project_dir="$2"
  local coq_project="$3"
  local target_v="$4"
  local target_vo="${target_v%.v}.vo"
  clean_coq_artifacts_once "$project_dir"
  run_capped "$profile" bash -lc "cd '$project_dir' && coq_makefile -f '$coq_project' -o Makefile.coq >/dev/null && make -f Makefile.coq -j1 '$target_vo'"
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
      coq_project_target "$profile" "$ROOT/docs/verification/msm" "_CoqProject" "$file"
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
  audit-contracts)
    audit_manifest
    audit_contracts
    ;;
  audit-contracts-tsv)
    audit_manifest >/dev/null
    audit_contracts_tsv
    ;;
  audit-evidence)
    audit_manifest
    audit_evidence
    ;;
  audit-evidence-tsv)
    audit_manifest >/dev/null
    audit_evidence_tsv
    ;;
  audit-vacuous)
    audit_vacuous
    ;;
  trusted)
    audit_manifest
    check_trusted_no_admitted
    check_trusted_assumptions
    check_trusted_contracts
    check_trusted_evidence
    ;;
  coq-trusted)
    audit_manifest
    check_trusted_no_admitted
    check_trusted_assumptions
    check_trusted_contracts
    check_trusted_evidence
    clean_trusted_coq_artifacts
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
usage: scripts/verify-formal.sh [audit|audit-tsv|audit-contracts|audit-contracts-tsv|audit-evidence|audit-evidence-tsv|audit-vacuous|trusted|coq-trusted|coq-file|tla|all]

All proof/model execution is memory-capped with systemd-run unless
FORMAL_VERIFY_ALLOW_UNCAPPED=1 is set.
USAGE
    exit 2
    ;;
esac
