#!/usr/bin/env bash
# Concurrency guard for the timing sweep.
#
# The protocol admits exactly ONE timed benchmark process at a time (see
# harnesses/common/PROTOCOL.md, "Fairness"). This guard distinguishes three
# populations that a naive `pgrep -f bench` conflates:
#
#   MINE     a harness binary whose ancestry reaches run-all.sh / run-one.sh.
#            More than one of these at once is a PROTOCOL VIOLATION: the runner
#            has lost serialization and the affected cells are invalid.
#
#   FOREIGN  a harness binary invoked by something else (another agent, an
#            interactive shell, a profiler). Not a protocol violation — the
#            runner is still serial — but it IS CPU contention that biases
#            whatever cell is in flight. Recorded, not treated as a violation.
#
#   WRAPPER  heaptrack / valgrind / perf / massif shims and the `sh -c` layers
#            around them. These carry the harness path in their command line
#            but are not themselves timed passes; counting them inflated the
#            old guard's tally (a single foreign heaptrack run presents as
#            four matching processes).
#
# Foreign observations are appended to <results>/foreign-contention.jsonl so the
# aggregator and the ledger can mark the affected measurement window rather than
# silently averaging contended samples in with clean ones.
#
# Usage: timed-proc-guard.sh <results-dir>
# Prints one line per poll ONLY when there is something to report; silence means
# a single well-behaved timed process (or none).
set -uo pipefail

RESULTS="${1:?usage: timed-proc-guard.sh <results-dir>}"
LEDGER="$RESULTS/foreign-contention.jsonl"

# A timed pass always carries --mode; profiler shims never run the pass
# themselves, they exec the binary that does.
HARNESS_RE='bench-cross-rust|\.stage/[a-z-]+/bench|bench\.mjs|VinaryBench|LegacyBench'
WRAPPER_RE='heaptrack|valgrind|massif|perf record|/bin/sh |/bin/zsh -c|/bin/bash -c'

# Walk the parent chain looking for the runner. Depth-capped so a pid-reuse
# cycle cannot wedge the guard.
descends_from_runner() {
    local pid="$1" depth=0 cmd
    while [ "$pid" -gt 1 ] && [ "$depth" -lt 12 ]; do
        cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
        case "$cmd" in
            *run-one.sh*|*run-all.sh*|*run-remaining.sh*|*run-hypothesis-cells*|*run-jvm-pair*|*run-stream.sh*)
                return 0 ;;
        esac
        pid="$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
        [ -n "$pid" ] || return 1
        depth=$((depth + 1))
    done
    return 1
}

mine=0
declare -a foreign=()

for proc in /proc/[0-9]*; do
    pid="${proc#/proc/}"
    [ "$pid" = "$$" ] && continue
    # Processes routinely exit between the glob and the read; a vanished pid is
    # normal, not an error, so the redirect itself must not emit to stderr.
    [ -r "$proc/cmdline" ] || continue
    cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null)" || continue
    [ -n "$cmd" ] || continue
    printf '%s' "$cmd" | grep -qE "$HARNESS_RE" || continue
    # Drop the profiler/shell shims and this guard's own probes.
    printf '%s' "$cmd" | grep -qE "$WRAPPER_RE" && {
        # Still worth recording: the wrapper's child IS eating cores.
        printf '%s' "$cmd" | grep -qE 'heaptrack|valgrind|massif|perf record' \
            && foreign+=("$pid	$(stat -c %Y "$proc" 2>/dev/null || echo 0)	$(printf '%s' "$cmd" | cut -c1-160)")
        continue
    }
    printf '%s' "$cmd" | grep -q 'timed-proc-guard' && continue

    if descends_from_runner "$pid"; then
        mine=$((mine + 1))
    else
        # The /proc/<pid> directory's mtime is the process start time, so
        # contention is dated from when it actually began rather than from the
        # poll that happened to notice it (polls are minutes apart).
        started="$(stat -c %Y "$proc" 2>/dev/null || echo 0)"
        foreign+=("$pid	$started	$(printf '%s' "$cmd" | cut -c1-160)")
    fi
done

now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Protocol violation: the runner is timing more than one thing at once.
if [ "$mine" -gt 1 ]; then
    echo "PROTOCOL-VIOLATION $now: $mine concurrent timed processes owned by the runner (expected 1)"
fi

# Contention: someone else is running the harness binaries on our cores.
# The ledger records an interval per pid (first_seen/last_seen) rather than a
# row per poll, so a foreign run that spans an hour costs one line, and the
# aggregator can intersect [first_seen, last_seen] against cell mtimes to mark
# exactly which cells were measured under contention.
# Runs whenever there is anything to record OR an existing ledger to close out.
# A poll that sees NOTHING is precisely the poll that can bound a departed
# process's end time, so gating this on a non-empty `foreign` would never close
# an interval.
if [ "${#foreign[@]}" -gt 0 ] || [ -f "$LEDGER" ]; then
    # Entries go through argv, not a pipe: a heredoc script already occupies
    # stdin, so piping the rows in would silently deliver nothing.
    python3 - "$LEDGER" "$now" ${foreign[@]+"${foreign[@]}"} <<'PY'
import json, os, sys
from datetime import datetime, timezone

ledger, ts, *entries = sys.argv[1:]
os.makedirs(os.path.dirname(ledger), exist_ok=True)


def iso(epoch):
    if not epoch:
        return None
    return datetime.fromtimestamp(int(epoch), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

rows = {}
order = []
if os.path.exists(ledger):
    with open(ledger) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["pid"], rec["cmdline"])
            if key not in rows:
                order.append(key)
            rows[key] = rec

seen_now = set()
for entry in entries:
    if not entry:
        continue
    pid_s, started_s, cmd = entry.split("\t", 2)
    key = (int(pid_s), cmd)
    seen_now.add(key)
    if key in rows:
        rows[key]["last_seen_utc"] = ts
        rows[key].pop("ended_by_utc", None)  # it is alive again this poll
    else:
        order.append(key)
        rows[key] = {"pid": int(pid_s), "cmdline": cmd,
                     "started_utc": iso(started_s),
                     "first_seen_utc": ts, "last_seen_utc": ts,
                     "kind": "foreign_harness_invocation"}

# Close out intervals for pids that were recorded before and are gone now. The
# process ended somewhere in (last_seen_utc, this poll], so ended_by_utc is the
# only sound upper bound on its contention. Consumers must use it rather than
# last_seen_utc, which would understate the window by up to one poll interval
# and let cells measured in that gap escape the contended flag.
for key, rec in rows.items():
    if key in seen_now or "ended_by_utc" in rec:
        continue
    if not os.path.exists(f"/proc/{rec['pid']}"):
        rec["ended_by_utc"] = ts

tmp = ledger + ".tmp"
with open(tmp, "w") as fh:
    for key in order:
        fh.write(json.dumps(rows[key]) + "\n")
os.replace(tmp, ledger)
PY
    status=$?
    if [ "${#foreign[@]}" -eq 0 ]; then
        : # nothing foreign this poll; any departed pids were just bounded above
    elif [ "$status" -eq 0 ]; then
        echo "FOREIGN-CONTENTION $now: ${#foreign[@]} harness process(es) not owned by the runner (recorded in foreign-contention.jsonl; runner still serial, mine=$mine)"
    else
        # Never claim the window was recorded when the write failed: an
        # unrecorded contention window is a silent hole in the evidence chain.
        echo "GUARD-ERROR $now: detected ${#foreign[@]} foreign harness process(es) but FAILED to record them in $LEDGER"
    fi
fi

exit 0
