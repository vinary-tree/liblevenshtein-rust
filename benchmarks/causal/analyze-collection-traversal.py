#!/usr/bin/env python3
"""Validate and summarize one paired collection-traversal experiment."""

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


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def bootstrap_median_interval(
    values: list[float], seed_material: str, replicates: int = 20_000
) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    generator = random.Random(seed)
    size = len(values)
    medians = [
        statistics.median(generator.choices(values, k=size)) for _ in range(replicates)
    ]
    medians.sort()
    return [medians[int(0.025 * replicates)], medians[int(0.975 * replicates)]]


def summarize(values: list[float], seed_material: str) -> dict[str, object]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mad": mad(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_ci_of_median": bootstrap_median_interval(
            values, seed_material
        ),
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("sample CSV is empty")
    required = {
        "sample",
        "role",
        "arm",
        "binary_sha256",
        "dictionary_entries",
        "consumed_entries_per_pass",
        "passes",
        "elapsed_ns",
        "checksum",
        "boundary_calls",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"sample CSV lacks columns: {sorted(missing)}")
    return rows


def validate_admissions(path: Path, sample_ids: list[int]) -> dict[str, object]:
    with path.open() as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records:
        raise ValueError("host-admission ledger is empty")
    rejected = [record for record in records if not record.get("admitted")]
    if rejected:
        raise ValueError(f"host-admission ledger contains {len(rejected)} rejected samples")
    cpus = {tuple(record.get("selected_cpus", [])) for record in records}
    if len(cpus) != 1:
        raise ValueError("host-admission ledger changed selected CPUs")
    expected_labels = ["warmup-before", "warmup-after"]
    expected_labels.extend(
        label
        for sample in sample_ids
        for label in (f"pair-{sample}-before", f"pair-{sample}-after")
    )
    observed_labels = [record.get("label") for record in records]
    if observed_labels != expected_labels:
        raise ValueError(
            "host-admission labels do not exactly bracket every measured pair"
        )
    return {
        "sha256": digest(path),
        "records": len(records),
        "selected_cpus": list(next(iter(cpus))),
        "all_admitted": True,
    }


def analyze(samples: Path, admissions: Path) -> dict[str, object]:
    rows = load_rows(samples)
    paired: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        role = row["role"]
        if role not in {"control", "treatment"}:
            raise ValueError(f"unknown role {role!r}")
        sample = int(row["sample"])
        if role in paired[sample]:
            raise ValueError(f"duplicate {role} row for sample {sample}")
        paired[sample][role] = row

    controls: list[float] = []
    treatments: list[float] = []
    ratios: list[float] = []
    differences: list[float] = []
    binary_digests: set[str] = set()
    control_arms: set[str] = set()
    treatment_arms: set[str] = set()
    entry_counts: set[int] = set()
    consumed_counts: set[int] = set()
    pass_counts: set[int] = set()
    control_calls: list[int] = []
    treatment_calls: list[int] = []
    checksums: set[int] = set()

    sample_ids = sorted(paired)
    if sample_ids != list(range(1, len(sample_ids) + 1)):
        raise ValueError(f"sample identifiers are not contiguous from 1: {sample_ids}")

    for sample in sample_ids:
        pair = paired[sample]
        if set(pair) != {"control", "treatment"}:
            raise ValueError(f"sample {sample} is not a complete pair")
        control = pair["control"]
        treatment = pair["treatment"]
        invariant_fields = (
            "binary_sha256",
            "dictionary_entries",
            "consumed_entries_per_pass",
            "passes",
            "checksum",
        )
        for field in invariant_fields:
            if control[field] != treatment[field]:
                raise ValueError(f"sample {sample} disagrees on {field}")

        passes = int(control["passes"])
        consumed = int(control["consumed_entries_per_pass"])
        denominator = passes * consumed
        if denominator <= 0:
            raise ValueError(f"sample {sample} has a non-positive work denominator")
        control_per_entry = int(control["elapsed_ns"]) / denominator
        treatment_per_entry = int(treatment["elapsed_ns"]) / denominator
        if control_per_entry <= 0 or treatment_per_entry <= 0:
            raise ValueError(f"sample {sample} has a non-positive duration")

        controls.append(control_per_entry)
        treatments.append(treatment_per_entry)
        ratios.append(control_per_entry / treatment_per_entry)
        differences.append(control_per_entry - treatment_per_entry)
        binary_digests.add(control["binary_sha256"])
        control_arms.add(control["arm"])
        treatment_arms.add(treatment["arm"])
        entry_counts.add(int(control["dictionary_entries"]))
        consumed_counts.add(consumed)
        pass_counts.add(passes)
        control_calls.append(int(control["boundary_calls"]))
        treatment_calls.append(int(treatment["boundary_calls"]))
        checksums.add(int(control["checksum"]))

    for label, values in {
        "binary digests": binary_digests,
        "control arms": control_arms,
        "treatment arms": treatment_arms,
        "dictionary sizes": entry_counts,
        "consumed counts": consumed_counts,
        "pass counts": pass_counts,
        "semantic checksums": checksums,
    }.items():
        if len(values) != 1:
            raise ValueError(f"experiment changed {label}: {sorted(values)}")

    samples_digest = digest(samples)
    return {
        "schema": "libdictenstein.collection-traversal-analysis.v1",
        "inputs": {
            "samples": {"path": str(samples), "sha256": samples_digest},
            "host_admission": validate_admissions(admissions, sample_ids),
        },
        "cell": {
            "control_arm": next(iter(control_arms)),
            "treatment_arm": next(iter(treatment_arms)),
            "binary_sha256": next(iter(binary_digests)),
            "dictionary_entries": next(iter(entry_counts)),
            "consumed_entries_per_pass": next(iter(consumed_counts)),
            "passes": next(iter(pass_counts)),
            "pairs": len(paired),
        },
        "nanoseconds_per_consumed_entry": {
            "control": summarize(controls, samples_digest + ":control"),
            "treatment": summarize(treatments, samples_digest + ":treatment"),
            "paired_difference_control_minus_treatment": summarize(
                differences, samples_digest + ":difference"
            ),
        },
        "paired_speedup_control_over_treatment": {
            **summarize(ratios, samples_digest + ":ratio"),
            "geometric_mean": math.exp(
                sum(math.log(ratio) for ratio in ratios) / len(ratios)
            ),
        },
        "foreign_boundary_calls_per_invocation": {
            "control": summarize(
                [float(value) for value in control_calls], samples_digest + ":ccalls"
            ),
            "treatment": summarize(
                [float(value) for value in treatment_calls],
                samples_digest + ":tcalls",
            ),
        },
        "semantic_invariants": {
            "paired_checksums_equal": True,
            "paired_work_counts_equal": True,
            "same_binary": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("host_admission", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(analyze(args.samples, args.host_admission), indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"collection traversal analysis failed: {error}") from error
