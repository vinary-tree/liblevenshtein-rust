#!/usr/bin/env python3
"""Aggregate a results directory into summary.json + tables/*.md.

Stdlib-only. For every cell JSON under <results>/cells/:
  - schema-validate (schema_check.py)
  - refuse to aggregate cells whose input digests disagree with
    environment.json (a run must be internally consistent)
  - compute robust statistics: median, MAD, p10/p90, mean, and a bootstrap
    95% confidence interval of the median (10,000 resamples, SplitMix64
    seeded 42 — same PRNG family as the workload generator; PROTOCOL.md is
    the normative definition)
  - derive µs/query and queries/second

Tables rendered (data artifacts consumed by the Phase 5 docs):
  query_summary.md        every query cell
  pair_java.md            legacy-java ÷ jvm-vinary speedups per shared cell
  pair_javascript.md      legacy-js ÷ {js-native, js-wasm, js-wasi} speedups
  atlas_overhead.md       target ÷ rust anchor overhead factors (dynamic_dawg)
  construct.md            dictionary construction times
  memory.md               peak RSS per target×backend
  not_measured.md         every expected-but-absent cell with its status

Every expected cell (matrix.py) is accounted for: measured, degraded,
failed, build_failed, or missing — nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
XL = SCRIPTS_DIR.parent
MASK64 = 0xFFFF_FFFF_FFFF_FFFF
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

sys.path.insert(0, str(SCRIPTS_DIR))
from schema_check import validate_file  # noqa: E402  # type: ignore[import-not-found]


class SplitMix64:
    """Normative PRNG (PROTOCOL.md / workload generator); used here so the
    bootstrap is reproducible across aggregate re-runs."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        threshold = (2**64 // n) * n
        while True:
            draw = self.next()
            if draw < threshold:
                return draw % n


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile on pre-sorted data."""
    if not sorted_values:
        raise ValueError("empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_median_ci(samples: list[int]) -> tuple[float, float]:
    rng = SplitMix64(BOOTSTRAP_SEED)
    n = len(samples)
    medians = [0.0] * BOOTSTRAP_RESAMPLES
    for i in range(BOOTSTRAP_RESAMPLES):
        resample = sorted(samples[rng.below(n)] for _ in range(n))
        mid = n // 2
        medians[i] = (
            float(resample[mid])
            if n % 2
            else (resample[mid - 1] + resample[mid]) / 2.0
        )
    medians.sort()
    return percentile(medians, 0.025), percentile(medians, 0.975)


def robust_stats(samples: list[int]) -> dict:
    ordered = sorted(samples)
    med = statistics.median(ordered)
    mad = statistics.median([abs(x - med) for x in ordered])
    ci_low, ci_high = bootstrap_median_ci(samples)
    return {
        "count": len(samples),
        "median_ns": med,
        "mad_ns": mad,
        "mean_ns": statistics.fmean(samples),
        "p10_ns": percentile(ordered, 0.10),
        "p90_ns": percentile(ordered, 0.90),
        "median_ci95_ns": [ci_low, ci_high],
    }


def load_cells(results_dir: Path) -> tuple[list[dict], list[str]]:
    schema = XL / "schema" / "result.schema.json"
    cells = []
    errors = []
    for path in sorted((results_dir / "cells").glob("*.json")):
        cell_errors = validate_file(path, schema)
        if cell_errors:
            errors.extend(cell_errors)
            continue
        data = json.loads(path.read_text())
        data["_file"] = path.name
        cells.append(data)
    return cells, errors


def check_digest_consistency(cells: list[dict], environment: dict) -> list[str]:
    """Refuse to summarize cells that did not all run against the same inputs.

    Two independent hazards are checked. The workload digest catches a
    regenerated dictionary or query set. The native-library digests catch a
    rebuild partway through a multi-hour run — cells on either side of it would
    otherwise be averaged together under one run_id while actually measuring two
    different binaries. Cells predating the native-digest field, and pure-legacy
    cells that link none of these libraries, are skipped rather than faulted.
    """
    problems = []
    expected_dict = environment["workload"]["dictionary_sha256"]
    for cell in cells:
        if cell["dictionary"]["sha256"] != expected_dict:
            problems.append(
                f"{cell['_file']}: dictionary digest {cell['dictionary']['sha256'][:12]}… "
                f"differs from environment {expected_dict[:12]}…"
            )

    pinned = {
        name: spec["sha256"]
        for name, spec in (environment.get("native_artifacts") or {}).items()
        if isinstance(spec, dict) and "sha256" in spec
    }
    for cell in cells:
        for name, digest in (cell.get("native_digests") or {}).items():
            expected = pinned.get(name)
            if expected is not None and digest != expected:
                problems.append(
                    f"{cell['_file']}: {name} digest {digest[:12]}… differs from the "
                    f"environment pin {expected[:12]}… (library rebuilt mid-run?)"
                )
    return problems


def key_of(cell: dict) -> tuple:
    return (
        cell["target"]["language"],
        cell["_file"].split("__")[0],  # target id from the filename convention
        cell["dictionary"]["structure"],
        cell["algorithm"],
        cell["max_distance"],
        cell["workload"]["queryset"],
    )


def cell_target_id(cell: dict) -> str:
    return cell["_file"].split("__")[0]


def summarize_query_cells(cells: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cell in cells:
        if cell["mode"] != "query":
            continue
        m = cell["measurements"]
        stats = robust_stats(m["samples_ns"])
        query_count = cell["workload"]["query_count"]
        entry = {
            "target": cell_target_id(cell),
            "backend": cell["dictionary"]["structure"],
            "algorithm": cell["algorithm"],
            "max_distance": cell["max_distance"],
            "queryset": cell["workload"]["queryset"],
            "status": cell["status"],
            "harness": cell["protocol"]["harness"],
            "query_count": query_count,
            "matches_per_pass": m["matches_per_pass"],
            "checksum_hex": m["checksum_hex"],
            "stats": stats,
            "us_per_query": stats["median_ns"] / 1000.0 / query_count,
            "queries_per_second": 1e9 * query_count / stats["median_ns"],
            "notes": cell["notes"],
        }
        out[cell["_file"].removesuffix(".json")] = entry
    return out


def coordinate(entry: dict) -> tuple:
    return (entry["algorithm"], entry["max_distance"], entry["queryset"])


def pair_table(
    query_cells: dict[str, dict], legacy_target: str, new_targets: list[tuple[str, str]]
) -> list[dict]:
    """Rows comparing a legacy target against new-stack target×backend legs."""
    legacy = {
        coordinate(e): e
        for e in query_cells.values()
        if e["target"] == legacy_target
    }
    rows = []
    for coord, legacy_entry in sorted(legacy.items()):
        row = {
            "algorithm": coord[0],
            "max_distance": coord[1],
            "queryset": coord[2],
            "legacy_us_per_query": legacy_entry["us_per_query"],
        }
        for target, backend in new_targets:
            match = next(
                (
                    e
                    for e in query_cells.values()
                    if e["target"] == target
                    and e["backend"] == backend
                    and coordinate(e) == coord
                ),
                None,
            )
            column = f"{target}({backend})"
            if match:
                row[f"{column}_us_per_query"] = match["us_per_query"]
                row[f"{column}_speedup"] = (
                    legacy_entry["us_per_query"] / match["us_per_query"]
                )
            else:
                row[f"{column}_us_per_query"] = None
                row[f"{column}_speedup"] = None
        rows.append(row)
    return rows


def pair_ratio_summary(rows: list[dict], speedup_key: str) -> dict:
    """Summarize one paired matrix without weighting slow cells more heavily.

    ``speedup_key`` is a control/treatment time ratio (legacy/new for the Java
    table), so values above one are treatment wins. The geometric mean is the
    multiplicatively symmetric breadth statistic; the median and full range
    are retained so one aggregate cannot conceal heterogeneous coordinates.
    """

    def summarize(selected: list[dict]) -> dict:
        ratios = [row[speedup_key] for row in selected if row.get(speedup_key) is not None]
        if not ratios:
            return {
                "cells": 0,
                "geometric_mean": None,
                "median": None,
                "minimum": None,
                "maximum": None,
                "treatment_wins": 0,
                "control_wins": 0,
                "ties": 0,
            }
        return {
            "cells": len(ratios),
            "geometric_mean": math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
            "median": statistics.median(ratios),
            "minimum": min(ratios),
            "maximum": max(ratios),
            "treatment_wins": sum(value > 1.0 for value in ratios),
            "control_wins": sum(value < 1.0 for value in ratios),
            "ties": sum(value == 1.0 for value in ratios),
        }

    algorithms = sorted({row["algorithm"] for row in rows})
    return {
        "ratio": "control_time_over_treatment_time",
        "overall": summarize(rows),
        "by_algorithm": {
            algorithm: summarize([row for row in rows if row["algorithm"] == algorithm])
            for algorithm in algorithms
        },
    }


def atlas_table(query_cells: dict[str, dict]) -> list[dict]:
    anchor = {
        coordinate(e): e
        for e in query_cells.values()
        if e["target"] == "rust" and e["backend"] == "dynamic_dawg"
    }
    # This table measures BINDING overhead: how much a language facade costs on
    # top of the same Rust core. Competing implementations do not belong in it —
    # their ratio against the anchor is a different quantity (implementation
    # speed, not binding cost) and belongs in the pair tables. cpp-legacy is
    # listed explicitly because it is currently excluded only incidentally, by
    # the dynamic_dawg backend filter below; relaxing that filter without this
    # line would silently relabel a rival implementation as binding overhead.
    rows = []
    targets = sorted(
        {e["target"] for e in query_cells.values()}
        - {"rust", "jvm-legacy", "js-legacy", "cpp-legacy"}
    )
    for target in targets:
        entries = [
            e
            for e in query_cells.values()
            if e["target"] == target and e["backend"] == "dynamic_dawg"
        ]
        ratios = []
        for entry in entries:
            base = anchor.get(coordinate(entry))
            if base:
                ratios.append(entry["us_per_query"] / base["us_per_query"])
        if not ratios:
            continue
        log_sum = sum(__import__("math").log(r) for r in ratios)
        rows.append(
            {
                "target": target,
                "cells_compared": len(ratios),
                "overhead_geomean": __import__("math").exp(log_sum / len(ratios)),
                "overhead_min": min(ratios),
                "overhead_max": max(ratios),
            }
        )
    return rows


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], title: str) -> str:
    lines = [f"# {title}", ""]
    header = " | ".join(label for _, label in columns)
    divider = " | ".join("---" for _ in columns)
    lines.append(f"| {header} |")
    lines.append(f"| {divider} |")
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif value is None:
                cells.append("—")
            else:
                cells.append(str(value))
        lines.append(f"| {' | '.join(cells)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--targets",
        help="expected-target accounting list (default: targets present in state.tsv/cells)",
    )
    parser.add_argument("--quick", action="store_true", help="expectation uses the quick matrix")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()

    environment_path = results_dir / "environment.json"
    if not environment_path.exists():
        raise SystemExit(f"aggregate: {environment_path} missing (run env-capture.py)")
    environment = json.loads(environment_path.read_text())

    # Stamp `contended: true` on any cell whose timed region overlapped a
    # foreign harness invocation before the cells are read, so the flag is on
    # disk and in this summary rather than discovered after publication. Load
    # average alone misses this: a profiler pinned to one core barely moves the
    # 1-minute average yet competes directly for the cell's core.
    # The marker's own report is authoritative for this list rather than a scan
    # of the loaded cells: load_cells() drops schema-invalid cells, and a cell
    # awaiting post-fill is invalid, so deriving the count from loaded cells
    # would silently report "0 contended" for exactly the in-flight batch that
    # the contention hit.
    foreign_contended: list[str] = []
    marker = Path(__file__).resolve().parent / "mark-contended-cells.py"
    if (results_dir / "foreign-contention.jsonl").exists() and marker.exists():
        marked = subprocess.run(
            [sys.executable, str(marker), str(results_dir), "--annotate", "--json"],
            capture_output=True,
            text=True,
        )
        for line in marked.stderr.splitlines():
            print(f"aggregate: contention: {line}", file=sys.stderr)
        if marked.returncode != 0:
            raise SystemExit("aggregate: foreign-contention marking failed; refusing to "
                             "summarize cells whose contention status is unknown")
        report = json.loads(marked.stdout or '{"affected": []}')
        foreign_contended = sorted(
            row["cell"].removesuffix(".json") for row in report.get("affected", [])
        )

    cells, validation_errors = load_cells(results_dir)
    for error in validation_errors:
        print(f"aggregate: INVALID {error}", file=sys.stderr)

    digest_problems = check_digest_consistency(cells, environment)
    if digest_problems:
        for problem in digest_problems:
            print(f"aggregate: DIGEST {problem}", file=sys.stderr)
        raise SystemExit("aggregate: refusing to mix workload digests in one summary")

    query_cells = summarize_query_cells(cells)

    construct_rows = []
    memory_rows = []
    for cell in cells:
        if cell["mode"] == "construct":
            times = cell["construct"]["times_ns"]
            stats = robust_stats(times)
            construct_rows.append(
                {
                    "target": cell_target_id(cell),
                    "backend": cell["dictionary"]["structure"],
                    "reps": cell["construct"]["reps"],
                    "median_ms": stats["median_ns"] / 1e6,
                    "mad_ms": stats["mad_ns"] / 1e6,
                    "terms": cell["construct"]["term_count"],
                }
            )
        elif cell["mode"] == "memory":
            memory_rows.append(
                {
                    "target": cell_target_id(cell),
                    "backend": cell["dictionary"]["structure"],
                    "max_rss_mib": cell["memory"]["max_rss_kib"] / 1024.0,
                    "matches_per_pass": cell["measurements"]["matches_per_pass"],
                }
            )

    # ------------------------------------------------------------------
    # Accounting: every expected cell must resolve to a status
    # ------------------------------------------------------------------
    present = {cell["_file"] for cell in cells}
    if args.targets:
        expected_targets = args.targets
    else:
        seen = sorted({cell_target_id(cell) for cell in cells})
        expected_targets = ",".join(seen) if seen else "rust"
    matrix_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "matrix.py"),
        "expected",
        "--targets",
        expected_targets,
    ]
    if args.quick:
        matrix_cmd.append("--quick")
    expected_lines = subprocess.run(
        matrix_cmd, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    not_measured = []
    for line in expected_lines:
        target, backend, mode, algorithm, distance, queryset = line.split("\t")
        name = f"{target}__{backend}__{mode}__{algorithm}__d{distance}__{queryset}.json"
        if name in present:
            continue
        build_failed = (results_dir / "failures" / f"{target}__{backend}.build_failed").exists()
        not_measured.append(
            {
                "cell": name.removesuffix(".json"),
                "status": "build_failed" if build_failed else "missing",
            }
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    query_rows = [
        {
            "cell": name,
            "target": entry["target"],
            "backend": entry["backend"],
            "algorithm": entry["algorithm"],
            "d": entry["max_distance"],
            "queryset": entry["queryset"],
            "us_per_query": entry["us_per_query"],
            "qps": entry["queries_per_second"],
            "ci_low_us": entry["stats"]["median_ci95_ns"][0] / 1000.0 / entry["query_count"],
            "ci_high_us": entry["stats"]["median_ci95_ns"][1] / 1000.0 / entry["query_count"],
            "matches": entry["matches_per_pass"],
            "samples": entry["stats"]["count"],
            "status": entry["status"],
        }
        for name, entry in sorted(query_cells.items())
    ]
    (tables_dir / "query_summary.md").write_text(
        markdown_table(
            query_rows,
            [
                ("target", "target"),
                ("backend", "backend"),
                ("algorithm", "algorithm"),
                ("d", "d"),
                ("queryset", "query set"),
                ("us_per_query", "µs/query (median)"),
                ("ci_low_us", "CI95 low"),
                ("ci_high_us", "CI95 high"),
                ("qps", "queries/s"),
                ("matches", "matches/pass"),
                ("samples", "samples"),
                ("status", "status"),
            ],
            "Query performance — all cells",
        )
    )

    java_rows = pair_table(
        query_cells,
        "jvm-legacy",
        [("jvm-vinary", "dynamic_dawg"), ("jvm-vinary", "double_array_trie")],
    )
    java_pair_summary = pair_ratio_summary(
        java_rows, "jvm-vinary(dynamic_dawg)_speedup"
    )
    (tables_dir / "pair_java.md").write_text(
        markdown_table(
            java_rows,
            [
                ("algorithm", "algorithm"),
                ("max_distance", "d"),
                ("queryset", "query set"),
                ("legacy_us_per_query", "legacy µs/q"),
                ("jvm-vinary(dynamic_dawg)_us_per_query", "vinary DD µs/q"),
                ("jvm-vinary(dynamic_dawg)_speedup", "speedup (DD)"),
                ("jvm-vinary(double_array_trie)_us_per_query", "vinary DAT µs/q"),
                ("jvm-vinary(double_array_trie)_speedup", "speedup (DAT)"),
            ],
            "Java vs Java — legacy 3.0.0 ÷ vinary-tree 0.10.0",
        )
    )

    # The C++ pair has a single new-stack leg: the vinary C++ facade is built
    # against DynamicDawg only (see targets.tsv), unlike the JVM pair which
    # carries both dictionary backends.
    cpp_rows = pair_table(query_cells, "cpp-legacy", [("cpp", "dynamic_dawg")])
    (tables_dir / "pair_cpp.md").write_text(
        markdown_table(
            cpp_rows,
            [
                ("algorithm", "algorithm"),
                ("max_distance", "d"),
                ("queryset", "query set"),
                ("legacy_us_per_query", "legacy C++ µs/q"),
                ("cpp(dynamic_dawg)_us_per_query", "vinary C++ µs/q"),
                ("cpp(dynamic_dawg)_speedup", "speedup"),
            ],
            "C++ vs C++ — liblevenshtein-cpp ÷ vinary-tree 0.10.0 C++ facade",
        )
    )

    js_rows = pair_table(
        query_cells,
        "js-legacy",
        [
            ("js-native", "dynamic_dawg"),
            ("js-wasm", "dynamic_dawg"),
            ("js-wasi", "dynamic_dawg"),
        ],
    )
    (tables_dir / "pair_javascript.md").write_text(
        markdown_table(
            js_rows,
            [
                ("algorithm", "algorithm"),
                ("max_distance", "d"),
                ("queryset", "query set"),
                ("legacy_us_per_query", "legacy µs/q"),
                ("js-native(dynamic_dawg)_us_per_query", "native µs/q"),
                ("js-native(dynamic_dawg)_speedup", "speedup (native)"),
                ("js-wasm(dynamic_dawg)_us_per_query", "wasm µs/q"),
                ("js-wasm(dynamic_dawg)_speedup", "speedup (wasm)"),
                ("js-wasi(dynamic_dawg)_us_per_query", "wasi µs/q"),
                ("js-wasi(dynamic_dawg)_speedup", "speedup (wasi)"),
            ],
            "JavaScript vs JavaScript — legacy 2.0.4 ÷ @vinary-tree/liblevenshtein 0.10.0",
        )
    )

    atlas_rows = atlas_table(query_cells)
    (tables_dir / "atlas_overhead.md").write_text(
        markdown_table(
            atlas_rows,
            [
                ("target", "target"),
                ("cells_compared", "cells"),
                ("overhead_geomean", "overhead ×rust (geomean)"),
                ("overhead_min", "min"),
                ("overhead_max", "max"),
            ],
            "Binding overhead vs pure-Rust anchor (dynamic_dawg cells)",
        )
    )

    (tables_dir / "construct.md").write_text(
        markdown_table(
            sorted(construct_rows, key=lambda r: (r["target"], r["backend"])),
            [
                ("target", "target"),
                ("backend", "backend"),
                ("median_ms", "median build ms"),
                ("mad_ms", "MAD ms"),
                ("reps", "reps"),
                ("terms", "terms"),
            ],
            "Dictionary construction (pre-sorted input; sorting excluded)",
        )
    )

    (tables_dir / "memory.md").write_text(
        markdown_table(
            sorted(memory_rows, key=lambda r: (r["target"], r["backend"])),
            [
                ("target", "target"),
                ("backend", "backend"),
                ("max_rss_mib", "peak RSS MiB"),
                ("matches_per_pass", "matches/pass"),
            ],
            "Peak RSS — construct + one full pass (whole process, /usr/bin/time -v)",
        )
    )

    # ------------------------------------------------------------------
    # Contention accounting. This machine is shared: external workloads run
    # unpinned and can be scheduled onto the pinned measurement core, which
    # inflates dispersion (observed: MAD 2.1-3.8% under load vs 1.2% on a
    # quiet machine). run-one.sh records the 1-minute load average with every
    # cell, so affected cells are IDENTIFIABLE rather than silently averaged
    # in. Threshold 8.0 ~ 25% of this host's 32 cores busy.
    # ------------------------------------------------------------------
    CONTENTION_THRESHOLD = 8.0
    contended = []
    missing_snapshot = []
    for cell in cells:
        load = cell.get("cell_snapshot", {}).get("load_avg_1m")
        if load is None:
            missing_snapshot.append(cell["_file"].removesuffix(".json"))
        elif load > CONTENTION_THRESHOLD:
            contended.append(
                {
                    "cell": cell["_file"].removesuffix(".json"),
                    "load_avg_1m": load,
                    "status": cell["status"],
                }
            )
    contended.sort(key=lambda r: -r["load_avg_1m"])
    quiet_count = len(cells) - len(contended) - len(missing_snapshot)

    # A second, independent contention signal (computed above from the marker's
    # own report): cells whose timed region overlapped a harness binary invoked
    # outside the runner — another agent, a profiler. These need not show an
    # elevated load average, since a single core-pinned profiler competes
    # directly for the measured core while barely moving a 32-core average, so
    # they are counted separately rather than folded into that population.
    foreign_note = (
        f"{len(foreign_contended)} cell(s) overlapped a harness process invoked outside "
        f"the runner's control (see foreign-contention.jsonl for the offending command "
        f"lines and their intervals). These are flagged independently of load average "
        f"because a core-pinned profiler contends for the measured core without moving "
        f"a 32-core load average."
        if foreign_contended
        else "No cell overlapped a harness process invoked outside the runner's control."
    )

    contention_note = (
        f"{len(contended)} of {len(cells)} cells were measured with a 1-minute load "
        f"average above {CONTENTION_THRESHOLD} on a 32-core host; {quiet_count} were "
        f"measured quiet and {len(missing_snapshot)} lack a snapshot (a cell only "
        f"receives one when its batch completes, so in-flight batches appear here). "
        f"Contention inflates dispersion rather than shifting medians systematically, "
        f"which is why medians with MAD and bootstrap intervals are reported instead "
        f"of means; cells listed below can be re-measured on a quiet machine. "
        f"{foreign_note}"
    )
    contention_md = f"# Cells measured under external load\n\n{contention_note}\n"
    if contended:
        contention_md += "\n" + markdown_table(
            contended,
            [("cell", "cell"), ("load_avg_1m", "1-min load"), ("status", "status")],
            "Cells with load average above the threshold",
        )
    if foreign_contended:
        contention_md += "\n\n## Cells overlapping a foreign harness invocation\n\n"
        contention_md += "\n".join(f"- `{name}`" for name in foreign_contended) + "\n"
    (tables_dir / "contention.md").write_text(contention_md)

    (tables_dir / "not_measured.md").write_text(
        markdown_table(
            not_measured,
            [("cell", "cell"), ("status", "status")],
            "Not measured (every expected-but-absent cell)",
        )
        if not_measured
        else "# Not measured\n\nAll expected cells are present.\n"
    )

    summary = {
        "run_id": results_dir.name,
        "environment_ref": "environment.json",
        "cell_count": len(cells),
        "validation_errors": validation_errors,
        "query_cells": query_cells,
        "construct": construct_rows,
        "memory": memory_rows,
        "pair_java": java_rows,
        "pair_java_summary": java_pair_summary,
        "pair_javascript": js_rows,
        "pair_cpp": cpp_rows,
        "atlas_overhead": atlas_rows,
        "not_measured": not_measured,
        "contention": {
            "threshold_load_avg_1m": CONTENTION_THRESHOLD,
            "cells_under_load": len(contended),
            "cells_quiet": quiet_count,
            "cells_without_snapshot": len(missing_snapshot),
            "note": contention_note,
            "cells": contended,
            "foreign_harness_overlap": {
                "cells": foreign_contended,
                "count": len(foreign_contended),
                "ledger": "foreign-contention.jsonl",
            },
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"aggregate: {len(cells)} cells, {len(not_measured)} not measured, "
        f"{len(validation_errors)} invalid — summary.json + tables/ written",
        file=sys.stderr,
    )
    return 1 if validation_errors else 0


if __name__ == "__main__":
    sys.exit(main())
