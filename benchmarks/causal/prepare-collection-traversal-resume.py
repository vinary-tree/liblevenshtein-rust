#!/usr/bin/env python3
"""Validate and prepare an interrupted collection experiment for strict resume."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


HEADER = [
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
]


def load_pairs(
    path: Path,
    binary_sha: str,
    control_arm: str,
    treatment_arm: str,
    entries: int,
    passes: int,
) -> int:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != HEADER:
            raise ValueError("sample CSV header does not match the v1 contract")
        rows = list(reader)
    if not rows or len(rows) % 2:
        raise ValueError("resume requires at least one complete control/treatment pair")

    pairs: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        sample = int(row["sample"])
        role = row["role"]
        if role not in {"control", "treatment"}:
            raise ValueError(f"sample {sample} has an invalid role")
        pair = pairs.setdefault(sample, {})
        if role in pair:
            raise ValueError(f"sample {sample} repeats role {role}")
        pair[role] = row

    sample_ids = sorted(pairs)
    if sample_ids != list(range(1, len(sample_ids) + 1)):
        raise ValueError("sample identifiers are not contiguous from one")
    for sample in sample_ids:
        pair = pairs[sample]
        if set(pair) != {"control", "treatment"}:
            raise ValueError(f"sample {sample} is incomplete")
        control = pair["control"]
        treatment = pair["treatment"]
        expected = ((control, control_arm), (treatment, treatment_arm))
        for row, arm in expected:
            if row["arm"] != arm:
                raise ValueError(f"sample {sample} changed the {row['role']} arm")
            if row["binary_sha256"] != binary_sha:
                raise ValueError(f"sample {sample} changed the binary digest")
            if int(row["dictionary_entries"]) != entries:
                raise ValueError(f"sample {sample} changed dictionary size")
            if int(row["passes"]) != passes:
                raise ValueError(f"sample {sample} changed pass count")
        for field in ("consumed_entries_per_pass", "checksum"):
            if control[field] != treatment[field]:
                raise ValueError(f"sample {sample} disagrees on {field}")
    return len(sample_ids)


def prepare_ledger(
    path: Path, rejected_path: Path, complete_pairs: int
) -> None:
    with path.open() as source:
        records = [json.loads(line) for line in source if line.strip()]
    expected_labels = ["warmup-before", "warmup-after"]
    expected_labels.extend(
        label
        for sample in range(1, complete_pairs + 1)
        for label in (f"pair-{sample}-before", f"pair-{sample}-after")
    )
    if len(records) < len(expected_labels):
        raise ValueError("host ledger does not bracket every retained pair")
    retained = records[: len(expected_labels)]
    for record, label in zip(retained, expected_labels, strict=True):
        if record.get("label") != label or not record.get("admitted"):
            raise ValueError(f"host ledger has invalid retained admission {label}")

    trailing = records[len(expected_labels) :]
    next_sample = complete_pairs + 1
    permitted = {f"pair-{next_sample}-before", f"pair-{next_sample}-after"}
    if any(record.get("label") not in permitted for record in trailing):
        raise ValueError("host ledger has unexplained records after the retained prefix")
    if trailing:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with rejected_path.open("a") as destination:
            for record in trailing:
                destination.write(json.dumps(record, separators=(",", ":")) + "\n")

    temporary = path.with_name(path.name + ".resume.tmp")
    with temporary.open("w") as destination:
        for record in retained:
            destination.write(json.dumps(record, separators=(",", ":")) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("host_ledger", type=Path)
    parser.add_argument("rejected_ledger", type=Path)
    parser.add_argument("binary_sha")
    parser.add_argument("control_arm")
    parser.add_argument("treatment_arm")
    parser.add_argument("entries", type=int)
    parser.add_argument("passes", type=int)
    args = parser.parse_args()

    complete = load_pairs(
        args.samples,
        args.binary_sha,
        args.control_arm,
        args.treatment_arm,
        args.entries,
        args.passes,
    )
    prepare_ledger(args.host_ledger, args.rejected_ledger, complete)
    print(complete + 1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"collection traversal resume failed: {error}") from error
