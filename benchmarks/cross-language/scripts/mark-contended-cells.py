#!/usr/bin/env python3
"""Intersect the foreign-contention ledger with cell measurement windows.

``timed-proc-guard.sh`` records *when* a harness binary ran outside the runner's
control (another agent, a profiler, an interactive shell).  Those processes do
not break the runner's serialization, but they do compete for the pinned cores,
so any cell whose timed region overlapped one of them was measured under load
that its clean siblings never saw.  Averaging the two populations together would
quietly bias every ratio computed from them.

This tool reconstructs each cell's measurement window and reports — or, with
``--annotate``, records in the cell itself — which cells overlap a contention
interval.  It never deletes or rewrites measurements: it only appends a note and
sets ``contended: true`` so the aggregator can separate the populations and the
ledger can state exactly which numbers are affected.

Window reconstruction
---------------------
A cell records ``timestamp_utc`` at the moment it is written, which is the *end*
of its measured region.  The start is recovered by subtracting everything the
harness did while timing:

```
start = timestamp_utc − (Σ samples_ns + warmup_ns + construct_ns)
```

``construct_ns`` is included because the dictionary build happens inside the same
process invocation and competes for the same cores.  The result is a conservative
(over-wide) window: it can flag a cell whose *timed* region merely sat near a
contention interval, which is the safe direction to err — a false positive costs
one re-measurement, a false negative silently corrupts a published ratio.

Usage:
    mark-contended-cells.py <results-dir> [--annotate] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

NS_PER_S = 1_000_000_000


def parse_iso(value: str) -> datetime:
    """Parse the ``...Z`` form the harnesses and guard both emit."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_intervals(ledger: Path) -> list[tuple[datetime, datetime, str]]:
    """Union-free list of (start, end, cmdline) foreign-contention intervals."""
    if not ledger.exists():
        return []
    intervals: list[tuple[datetime, datetime, str]] = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # started_utc is the true process start; first_seen_utc is only when the
        # guard's poll happened to notice it. Prefer the former.
        start = parse_iso(rec.get("started_utc") or rec["first_seen_utc"])
        # ended_by_utc is the first poll that found the process GONE, so the
        # true end lies in (last_seen_utc, ended_by_utc]. Using last_seen_utc
        # would understate the window by up to one poll interval and let cells
        # measured in that gap escape the flag; take the sound upper bound.
        end = parse_iso(rec.get("ended_by_utc") or rec["last_seen_utc"])
        intervals.append((start, end, rec["cmdline"]))
    return intervals


def cell_window(cell: dict) -> tuple[datetime, datetime] | None:
    """Reconstruct (start, end) of the cell's measured region, or None."""
    stamp = cell.get("timestamp_utc")
    if not stamp:
        return None
    end = parse_iso(stamp)

    elapsed_ns = 0
    measurements = cell.get("measurements") or {}
    samples = measurements.get("samples_ns") or []
    elapsed_ns += sum(samples)

    protocol = cell.get("protocol") or {}
    warmup_s = protocol.get("warmup_seconds_min") or 0
    elapsed_ns += int(warmup_s * NS_PER_S)

    construct = cell.get("construct") or {}
    elapsed_ns += sum(construct.get("times_ns") or [])
    dictionary = cell.get("dictionary") or {}
    elapsed_ns += dictionary.get("construct_ns") or 0

    return end - timedelta(seconds=elapsed_ns / NS_PER_S), end


def overlaps(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--annotate", action="store_true",
                        help="record contended:true + a note in each affected cell")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    ledger = args.results / "foreign-contention.jsonl"
    intervals = load_intervals(ledger)
    if not intervals:
        print(f"no foreign-contention intervals recorded in {ledger}", file=sys.stderr)
        return 0

    span_start = min(i[0] for i in intervals)
    span_end = max(i[1] for i in intervals)
    print(f"{len(intervals)} contention interval(s) spanning "
          f"{span_start.isoformat()} .. {span_end.isoformat()}", file=sys.stderr)

    affected = []
    cells_dir = args.results / "cells"
    for path in sorted(cells_dir.glob("*.json")):
        try:
            cell = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  SKIP unparseable: {path.name}", file=sys.stderr)
            continue
        window = cell_window(cell)
        if window is None:
            continue
        hits = [c for (s, e, c) in intervals if overlaps(window, (s, e))]
        if not hits:
            continue
        affected.append({
            "cell": path.name,
            "window_start_utc": window[0].isoformat().replace("+00:00", "Z"),
            "window_end_utc": window[1].isoformat().replace("+00:00", "Z"),
            "foreign_processes": sorted({c.split()[0] + " …" for c in hits}),
        })
        if args.annotate:
            cell["contended"] = True
            notes = cell.setdefault("notes", [])
            note = ("measured while foreign harness process(es) ran outside the runner's "
                    "control; see foreign-contention.jsonl")
            if note not in notes:
                notes.append(note)
            path.write_text(json.dumps(cell, indent=2) + "\n")

    if args.json:
        json.dump({"intervals": len(intervals), "affected": affected}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for row in affected:
            print(f"CONTENDED {row['cell']}")
    print(f"{len(affected)} cell(s) overlap a contention interval"
          f"{' (annotated)' if args.annotate else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
