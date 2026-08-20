#!/usr/bin/env python3
"""Validate and summarize paired all-backend propagation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

SCHEMA = "liblevenshtein.backend-propagation-matrix.v1"
ROWS_PER_ARM = 150
BOOTSTRAP_RESAMPLES = 10_000


def cell_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["backend_family"],
        row["backend"],
        row["unit_domain"],
        row["algorithm"],
        row["stage"],
    )


def median_ci(values: list[float], seed_material: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    random_source = random.Random(seed)
    size = len(values)
    medians = sorted(
        statistics.median(random_source.choices(values, k=size))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return [medians[249], medians[9749]]


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize_ratios(values: list[float], seed_material: str) -> dict[str, object]:
    median = statistics.median(values)
    return {
        "samples": len(values),
        "geometric_mean": geometric_mean(values),
        "median": median,
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
        "bootstrap_95_ci_of_median": median_ci(values, seed_material),
        "minimum": min(values),
        "maximum": max(values),
        "treatment_wins": sum(value > 1.0 for value in values),
        "ties": sum(value == 1.0 for value in values),
        "control_wins": sum(value < 1.0 for value in values),
    }


def read_and_validate(path: Path) -> tuple[list[dict[str, str]], int]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("evidence CSV contains no data rows")
    if any(row["schema"] != SCHEMA for row in rows):
        raise ValueError("unexpected evidence schema")

    by_replicate_arm: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_replicate_arm[(int(row["replicate"]), row["arm"])].append(row)
    replicates = sorted({replicate for replicate, _ in by_replicate_arm})
    if replicates != list(range(1, len(replicates) + 1)):
        raise ValueError("replicates are not contiguous from one")

    semantic_fields = ("applicability", "reason", "result_count", "checksum_u64")
    for replicate in replicates:
        arms: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
        for arm in ("control", "treatment"):
            arm_rows = by_replicate_arm.get((replicate, arm), [])
            if len(arm_rows) != ROWS_PER_ARM:
                raise ValueError(
                    f"replicate {replicate} {arm} has {len(arm_rows)} rows, expected {ROWS_PER_ARM}"
                )
            keyed = {cell_key(row): row for row in arm_rows}
            if len(keyed) != ROWS_PER_ARM:
                raise ValueError(f"replicate {replicate} {arm} contains duplicate cells")
            arms[arm] = keyed
        if arms["control"].keys() != arms["treatment"].keys():
            raise ValueError(f"replicate {replicate} arm inventories differ")
        for key, control in arms["control"].items():
            treatment = arms["treatment"][key]
            if any(control[field] != treatment[field] for field in semantic_fields):
                raise ValueError(f"replicate {replicate} semantic mismatch for {key}")
    return rows, len(replicates)


def analyze(path: Path) -> dict[str, object]:
    rows, replicate_count = read_and_validate(path)
    paired: dict[tuple[str, ...], dict[int, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["applicability"] == "applicable":
            paired[cell_key(row)][int(row["replicate"])][row["arm"]] = row

    cells: list[dict[str, object]] = []
    for key in sorted(paired):
        elapsed_ratios: list[float] = []
        allocation_ratios: list[float] = []
        control_ns: list[float] = []
        treatment_ns: list[float] = []
        for replicate in range(1, replicate_count + 1):
            pair = paired[key][replicate]
            control = float(pair["control"]["ns_per_operation"])
            treatment = float(pair["treatment"]["ns_per_operation"])
            control_ns.append(control)
            treatment_ns.append(treatment)
            elapsed_ratios.append(control / treatment)
            treatment_allocated = int(pair["treatment"]["allocated_bytes"])
            if treatment_allocated:
                allocation_ratios.append(
                    int(pair["control"]["allocated_bytes"]) / treatment_allocated
                )
        family, backend, domain, algorithm, stage = key
        label = "/".join(key)
        cells.append(
            {
                "backend_family": family,
                "backend": backend,
                "unit_domain": domain,
                "algorithm": algorithm,
                "stage": stage,
                "control_median_ns_per_operation": statistics.median(control_ns),
                "treatment_median_ns_per_operation": statistics.median(treatment_ns),
                "paired_speedup_control_over_treatment": summarize_ratios(
                    elapsed_ratios, label + "/elapsed"
                ),
                "paired_allocated_bytes_ratio_control_over_treatment": (
                    summarize_ratios(allocation_ratios, label + "/allocated")
                    if allocation_ratios
                    else None
                ),
            }
        )

    query_cells = [cell for cell in cells if cell["stage"] == "query"]
    construction_cells = [cell for cell in cells if cell["stage"] == "construction"]

    def across_cells(selected: list[dict[str, object]]) -> dict[str, object]:
        ratios = [
            float(cell["paired_speedup_control_over_treatment"]["median"])
            for cell in selected
        ]
        return {
            "cells": len(selected),
            "geometric_mean_of_cell_paired_medians": geometric_mean(ratios),
            "median_of_cell_paired_medians": statistics.median(ratios),
            "minimum_cell_paired_median": min(ratios),
            "maximum_cell_paired_median": max(ratios),
            "cells_treatment_won": sum(ratio > 1.0 for ratio in ratios),
            "cells_tied": sum(ratio == 1.0 for ratio in ratios),
            "cells_control_won": sum(ratio < 1.0 for ratio in ratios),
        }

    family_summary = {}
    for family in sorted({str(cell["backend_family"]) for cell in query_cells}):
        family_summary[family] = across_cells(
            [cell for cell in query_cells if cell["backend_family"] == family]
        )

    algorithm_summary = {}
    for algorithm in sorted({str(cell["algorithm"]) for cell in query_cells}):
        algorithm_summary[algorithm] = across_cells(
            [cell for cell in query_cells if cell["algorithm"] == algorithm]
        )

    return {
        "schema": "liblevenshtein.backend-propagation-analysis.v1",
        "source": str(path),
        "replicates": replicate_count,
        "applicable_cells": len(cells),
        "query": across_cells(query_cells),
        "construction": across_cells(construction_cells),
        "query_by_backend_family": family_summary,
        "query_by_algorithm": algorithm_summary,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze(arguments.evidence)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
