#!/usr/bin/env python3
"""Topology-aware CPU admission gate for causal benchmarks.

The benchmark core and every core sharing its last-level cache are hard gates.
Other CCDs are recorded, but do not reject a run on a many-core machine. This
distinguishes direct scheduling/cache contention from socket-wide background
load while retaining an auditable record of both.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CPU_SYSFS = Path("/sys/devices/system/cpu")
PROC_STAT = Path("/proc/stat")


def parse_cpu_list(specification: str) -> list[int]:
    cpus: set[int] = set()
    for part in specification.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        else:
            start, end = map(int, bounds)
            if end < start:
                raise ValueError(f"descending CPU range: {part}")
            cpus.update(range(start, end + 1))
    return sorted(cpus)


def read_cpu_list(path: Path, fallback: int) -> list[int]:
    try:
        return parse_cpu_list(path.read_text())
    except FileNotFoundError:
        return [fallback]


def read_cpu_times(path: Path = PROC_STAT) -> dict[int, tuple[int, int]]:
    times: dict[int, tuple[int, int]] = {}
    with path.open() as source:
        for line in source:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
                continue
            suffix = fields[0][3:]
            if not suffix.isdigit():
                continue
            values = [int(value) for value in fields[1:]]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            times[int(suffix)] = (total, idle)
    if not times:
        raise RuntimeError(f"no per-CPU counters found in {path}")
    return times


def busy_percentages(
    before: dict[int, tuple[int, int]],
    after: dict[int, tuple[int, int]],
) -> dict[int, float]:
    busy: dict[int, float] = {}
    for cpu in sorted(before.keys() & after.keys()):
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0 or idle_delta < 0:
            raise RuntimeError(f"non-monotonic /proc/stat counters for CPU {cpu}")
        busy[cpu] = 100.0 * (total_delta - idle_delta) / total_delta
    return busy


def group_statistics(busy: dict[int, float], cpus: list[int]) -> dict[str, float]:
    samples = [busy[cpu] for cpu in cpus if cpu in busy]
    if not samples:
        raise RuntimeError(f"no utilization samples for CPUs {cpus}")
    return {
        "mean_busy_percent": sum(samples) / len(samples),
        "max_busy_percent": max(samples),
    }


def topology(cpu: int) -> tuple[list[int], list[int], int, int]:
    root = CPU_SYSFS / f"cpu{cpu}"
    if not root.exists():
        raise ValueError(f"CPU {cpu} does not exist")
    siblings = read_cpu_list(root / "topology/thread_siblings_list", cpu)
    l3 = read_cpu_list(root / "cache/index3/shared_cpu_list", cpu)
    core_id = int((root / "topology/core_id").read_text())
    package_id = int((root / "topology/physical_package_id").read_text())
    return siblings, l3, core_id, package_id


def sample(cpu: int, interval: float) -> dict[str, object]:
    siblings, l3, core_id, package_id = topology(cpu)
    before = read_cpu_times()
    time.sleep(interval)
    after = read_cpu_times()
    busy = busy_percentages(before, after)
    all_cpus = sorted(busy)
    return {
        "schema": "liblevenshtein.causal-host-load.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": os.getpid(),
        "interval_seconds": interval,
        "selected_cpu": cpu,
        "core_id": core_id,
        "physical_package_id": package_id,
        "thread_siblings": siblings,
        "last_level_cache_cpus": l3,
        "selected_cpu_busy_percent": busy[cpu],
        "thread_siblings_busy": group_statistics(busy, siblings),
        "last_level_cache_busy": group_statistics(busy, l3),
        "package_busy": group_statistics(busy, all_cpus),
        "per_cpu_busy_percent": {str(key): value for key, value in busy.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-selected-busy", type=float, default=20.0)
    parser.add_argument("--max-sibling-busy", type=float, default=20.0)
    parser.add_argument("--max-llc-mean-busy", type=float, default=25.0)
    parser.add_argument("--max-llc-peer-busy", type=float, default=50.0)
    parser.add_argument("--label", help="caller-defined phase or pair identifier")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    for name in (
        "max_selected_busy",
        "max_sibling_busy",
        "max_llc_mean_busy",
        "max_llc_peer_busy",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 100.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 100")

    record = sample(args.cpu, args.interval)
    record["label"] = args.label
    selected = float(record["selected_cpu_busy_percent"])
    siblings = record["thread_siblings_busy"]
    llc = record["last_level_cache_busy"]
    assert isinstance(siblings, dict) and isinstance(llc, dict)
    reasons: list[str] = []
    if selected > args.max_selected_busy:
        reasons.append(
            f"selected CPU busy {selected:.2f}% > {args.max_selected_busy:.2f}%"
        )
    if float(siblings["max_busy_percent"]) > args.max_sibling_busy:
        reasons.append(
            "hardware-thread sibling busy "
            f"{float(siblings['max_busy_percent']):.2f}% > {args.max_sibling_busy:.2f}%"
        )
    if float(llc["mean_busy_percent"]) > args.max_llc_mean_busy:
        reasons.append(
            f"LLC-group mean busy {float(llc['mean_busy_percent']):.2f}% "
            f"> {args.max_llc_mean_busy:.2f}%"
        )
    if float(llc["max_busy_percent"]) > args.max_llc_peer_busy:
        reasons.append(
            f"LLC-group peer busy {float(llc['max_busy_percent']):.2f}% "
            f"> {args.max_llc_peer_busy:.2f}%"
        )

    record["thresholds"] = {
        "max_selected_busy_percent": args.max_selected_busy,
        "max_sibling_busy_percent": args.max_sibling_busy,
        "max_llc_mean_busy_percent": args.max_llc_mean_busy,
        "max_llc_peer_busy_percent": args.max_llc_peer_busy,
    }
    record["admitted"] = not reasons
    record["rejection_reasons"] = reasons
    encoded = json.dumps(record, separators=(",", ":"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as destination:
            destination.write(encoded + "\n")
    print(encoded)
    if reasons:
        print("host-load admission rejected: " + "; ".join(reasons), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"host-load admission failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
