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


def sample(cpus: list[int], interval: float) -> dict[str, object]:
    if not cpus:
        raise ValueError("at least one CPU is required")
    topology_by_cpu = {cpu: topology(cpu) for cpu in cpus}
    before = read_cpu_times()
    time.sleep(interval)
    after = read_cpu_times()
    busy = busy_percentages(before, after)
    all_cpus = sorted(busy)
    llc_groups = sorted(
        {tuple(details[1]) for details in topology_by_cpu.values()},
        key=lambda group: group[0],
    )
    sibling_groups = sorted(
        {tuple(details[0]) for details in topology_by_cpu.values()},
        key=lambda group: group[0],
    )
    record: dict[str, object] = {
        "schema": "liblevenshtein.causal-host-load.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": os.getpid(),
        "interval_seconds": interval,
        "selected_cpus": cpus,
        "selected_cpus_busy": group_statistics(busy, cpus),
        "thread_sibling_groups": [
            {
                "cpus": list(group),
                **group_statistics(busy, list(group)),
            }
            for group in sibling_groups
        ],
        "last_level_cache_groups": [
            {
                "cpus": list(group),
                **group_statistics(busy, list(group)),
            }
            for group in llc_groups
        ],
        "core_ids": sorted({details[2] for details in topology_by_cpu.values()}),
        "physical_package_ids": sorted(
            {details[3] for details in topology_by_cpu.values()}
        ),
        "package_busy": group_statistics(busy, all_cpus),
        "per_cpu_busy_percent": {str(key): value for key, value in busy.items()},
    }
    # Preserve the original scalar fields for the existing single-CPU callers
    # and their already-committed evidence readers.
    if len(cpus) == 1:
        cpu = cpus[0]
        siblings, l3, core_id, package_id = topology_by_cpu[cpu]
        record.update(
            {
                "selected_cpu": cpu,
                "core_id": core_id,
                "physical_package_id": package_id,
                "thread_siblings": siblings,
                "last_level_cache_cpus": l3,
                "selected_cpu_busy_percent": busy[cpu],
                "thread_siblings_busy": group_statistics(busy, siblings),
                "last_level_cache_busy": group_statistics(busy, l3),
            }
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--cpu", type=int)
    selected.add_argument(
        "--cpuset",
        help="CPU list/ranges whose complete LLC-sharing groups must be quiet",
    )
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

    try:
        cpus = [args.cpu] if args.cpu is not None else parse_cpu_list(args.cpuset)
    except ValueError as error:
        parser.error(str(error))
    record = sample(cpus, args.interval)
    record["label"] = args.label
    selected_busy = record["selected_cpus_busy"]
    sibling_groups = record["thread_sibling_groups"]
    llc_groups = record["last_level_cache_groups"]
    assert isinstance(selected_busy, dict)
    assert isinstance(sibling_groups, list) and isinstance(llc_groups, list)
    reasons: list[str] = []
    if float(selected_busy["max_busy_percent"]) > args.max_selected_busy:
        reasons.append(
            "selected CPU busy "
            f"{float(selected_busy['max_busy_percent']):.2f}% > "
            f"{args.max_selected_busy:.2f}%"
        )
    for group in sibling_groups:
        assert isinstance(group, dict)
        if float(group["max_busy_percent"]) > args.max_sibling_busy:
            reasons.append(
                f"hardware-thread sibling group {group['cpus']} busy "
                f"{float(group['max_busy_percent']):.2f}% > "
                f"{args.max_sibling_busy:.2f}%"
            )
    for group in llc_groups:
        assert isinstance(group, dict)
        if float(group["mean_busy_percent"]) > args.max_llc_mean_busy:
            reasons.append(
                f"LLC group {group['cpus']} mean busy "
                f"{float(group['mean_busy_percent']):.2f}% > "
                f"{args.max_llc_mean_busy:.2f}%"
            )
        if float(group["max_busy_percent"]) > args.max_llc_peer_busy:
            reasons.append(
                f"LLC group {group['cpus']} peer busy "
                f"{float(group['max_busy_percent']):.2f}% > "
                f"{args.max_llc_peer_busy:.2f}%"
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
