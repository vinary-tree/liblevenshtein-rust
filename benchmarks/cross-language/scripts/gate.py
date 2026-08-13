#!/usr/bin/env python3
"""Correctness pre-gate (PROTOCOL.md §6.4; plan §7).

Every target×backend must reproduce the Rust oracle's result multiset —
compared as (matches, term_bytes, distance_sum, checksum) — before any of
its timing runs are accepted. The gate covers the FULL query-cell family
(the same coordinates the timed matrix runs), so every timed cell has a
verify twin carrying its correctness evidence — the JMH-converted Java
cells copy their checksums from these twins:

    3 shared algorithms (standard, transposition, merge_and_split)
  × 3 distances (1, 2, 3)
  × 5 query sets (hits, X-d1, X-d2, X-d3, oov)                  = 45 cells
  + damerau_levenshtein × 3 × 5, for targets whose manifest
    says damerau=yes                                             (+15 cells)

X is the algorithm-matched mutation family: std for standard/
merge_and_split, tr for transposition/damerau_levenshtein. The oracle is
rust×dynamic_dawg; rust×double_array_trie must match it exactly (internal
consistency) before anything else is compared.

Execution uses the batched `--cells --mode verify` path through run-one.sh
so one expensive DoubleArrayTrie construction serves all of a backend's
verify cells.

Usage:
    gate.py <results_dir> --targets rust,c[,...]
    gate.py <results_dir> --targets ... --report-only

Exit 0 iff every compared cell matches or is covered by an entry a human
added to `documented_mismatches` in gate-report.json (this script never
auto-documents a delta).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
XL = SCRIPTS_DIR.parent

SHARED_ALGORITHMS = ("standard", "transposition", "merge_and_split")
BONUS_ALGORITHM = "damerau_levenshtein"
DISTANCES = (1, 2, 3)
ORACLE_TARGET = "rust"
ORACLE_BACKEND = "dynamic_dawg"


def mut_prefix(algorithm: str) -> str:
    return "tr" if algorithm in ("transposition", BONUS_ALGORITHM) else "std"


def cells_for(algorithms: tuple[str, ...]) -> list[tuple[str, int, str]]:
    return [
        (algorithm, distance, queryset)
        for algorithm in algorithms
        for distance in DISTANCES
        for queryset in (
            "hits",
            f"{mut_prefix(algorithm)}-d1",
            f"{mut_prefix(algorithm)}-d2",
            f"{mut_prefix(algorithm)}-d3",
            "oov",
        )
    ]


def load_manifest() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    header: list[str] = []
    for line in (XL / "targets.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split("\t")
        if fields[0] == "target":
            header = fields
            continue
        rows[fields[0]] = dict(zip(header, fields))
    return rows


def cell_json_path(
    results_dir: Path, target: str, backend: str, algorithm: str, distance: int, queryset: str
) -> Path:
    return (
        results_dir
        / "gate"
        / f"{target}__{backend}__verify__{algorithm}__d{distance}__{queryset}.json"
    )


def run_verify_batch(
    results_dir: Path,
    target: str,
    backend: str,
    cells: list[tuple[str, int, str]],
) -> None:
    """Runs all missing verify cells for target×backend in ONE process."""
    missing = [
        (algorithm, distance, queryset)
        for algorithm, distance, queryset in cells
        if not cell_json_path(results_dir, target, backend, algorithm, distance, queryset).exists()
    ]
    if not missing:
        return
    tsv_dir = results_dir / "gate" / "tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = tsv_dir / f"{target}__{backend}.tsv"
    with tsv_path.open("w") as fh:
        for algorithm, distance, queryset in missing:
            queries = XL / "workload" / "queries" / f"{queryset}.txt"
            out = cell_json_path(results_dir, target, backend, algorithm, distance, queryset)
            fh.write(f"{algorithm}\t{distance}\t{queries}\t{out}\n")
    env = dict(os.environ)
    env["XL_CELL_DIR"] = str(results_dir / "gate")
    # Verify cells are untimed correctness runs: route them off the timing
    # cores so gating can proceed concurrently with a timed run.
    env.setdefault("XL_CPUSET_OVERRIDE", "16-31")
    result = subprocess.run(
        [
            str(SCRIPTS_DIR / "run-one.sh"),
            "cells",
            str(results_dir),
            target,
            backend,
            "verify",
            str(tsv_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"gate: verify batch failed for {target}×{backend}:\n"
            f"{result.stdout}{result.stderr}"
        )


def signature(cell_json: Path) -> tuple[int, int, int, str]:
    data = json.loads(cell_json.read_text())
    measurements = data["measurements"]
    return (
        measurements["matches_per_pass"],
        measurements["term_bytes_per_pass"],
        measurements["distance_sum_per_pass"],
        measurements["checksum_hex"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--targets",
        default="rust,c",
        help="comma-separated targets to gate (oracle 'rust' is always included)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="re-render gate-report.json from existing verify cells; run nothing",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    (results_dir / "gate").mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if ORACLE_TARGET not in targets:
        targets.insert(0, ORACLE_TARGET)
    for target in targets:
        if target not in manifest:
            raise SystemExit(f"gate: unknown target {target!r}")

    report_path = results_dir / "gate-report.json"
    documented: list[dict] = []
    if report_path.exists():
        documented = json.loads(report_path.read_text()).get("documented_mismatches", [])

    shared_cells = cells_for(SHARED_ALGORITHMS)
    bonus_cells = cells_for((BONUS_ALGORITHM,))

    def target_cells(target: str) -> list[tuple[str, int, str]]:
        if manifest[target]["damerau"] == "yes":
            return shared_cells + bonus_cells
        return shared_cells

    # 1. Oracle cells (dynamic_dawg, incl. damerau — the oracle supports it).
    if not args.report_only:
        run_verify_batch(results_dir, ORACLE_TARGET, ORACLE_BACKEND, shared_cells + bonus_cells)
    oracle_signatures = {
        cell: signature(cell_json_path(results_dir, ORACLE_TARGET, ORACLE_BACKEND, *cell))
        for cell in shared_cells + bonus_cells
    }

    # 2. Every requested target×backend, batched per backend.
    comparisons: list[dict] = []
    failures = 0
    for target in targets:
        backends = manifest[target]["backends"].split(",")
        for backend in backends:
            if target == ORACLE_TARGET and backend == ORACLE_BACKEND:
                continue
            if backend == "own" and target not in manifest:
                continue
            cells = target_cells(target)
            if not args.report_only:
                run_verify_batch(results_dir, target, backend, cells)
            for cell in cells:
                algorithm, distance, queryset = cell
                cell_key = f"{target}__{backend}__{algorithm}__d{distance}__{queryset}"
                observed = signature(cell_json_path(results_dir, target, backend, *cell))
                expected = oracle_signatures[cell]
                matched = observed == expected
                excused = any(d.get("cell") == cell_key for d in documented)
                comparisons.append(
                    {
                        "cell": cell_key,
                        "expected": dict(
                            zip(("matches", "term_bytes", "distance_sum", "checksum_hex"), expected)
                        ),
                        "observed": dict(
                            zip(("matches", "term_bytes", "distance_sum", "checksum_hex"), observed)
                        ),
                        "verdict": "match" if matched else ("documented" if excused else "MISMATCH"),
                    }
                )
                if not matched and not excused:
                    failures += 1
                    print(f"gate: MISMATCH {cell_key}", file=sys.stderr)
                    print(f"  expected {expected}", file=sys.stderr)
                    print(f"  observed {observed}", file=sys.stderr)

    report = {
        "oracle": f"{ORACLE_TARGET}x{ORACLE_BACKEND}",
        "gate_limit_queries": int(os.environ.get("XL_GATE_LIMIT", "200")),
        "targets": targets,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "documented_mismatches": documented,
        "verdict": "green" if failures == 0 else f"{failures} undocumented mismatch(es)",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"gate: {len(comparisons)} comparisons, verdict: {report['verdict']} "
        f"(report: {report_path})",
        file=sys.stderr,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
