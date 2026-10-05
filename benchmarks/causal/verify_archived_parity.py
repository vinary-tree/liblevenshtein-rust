"""Read-only integrity and outcome audit for the archived post-optimization pairs.

The checker consumes the committed 51-pair raw observations. It validates their
hash-linked identity, signature, per-arm work counts, sample table and recorded
medians, then checks the stated parity and construction outcomes. It does not
claim that the currently checked-out binaries were timed by these old samples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent / "evidence" / "2026-08-19"
WORKLOAD = Path(__file__).resolve().parents[1] / "cross-language" / "workload"
CASES = (
    ("direct-standard-d1-hits", "query"),
    ("direct-construction-from-terms", "construct"),
    ("direct-construction-from-sorted-terms", "construct"),
)
MAX_RUST_QUERY_US = 51.2
JMH_UNIT_TO_NS = {"ns/op": 1, "us/op": 1_000, "ms/op": 1_000_000, "s/op": 1_000_000_000}
SIGNATURE_FIELDS = (
    "matches_per_pass",
    "term_bytes_per_pass",
    "distance_sum_per_pass",
    "checksum_hex",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def audit_case(name: str, expected_mode: str) -> dict:
    directory = EVIDENCE / name
    config = read_json(directory / "run-config.json")
    summary = read_json(directory / "summary.json")
    require(config["mode"] == summary["mode"] == expected_mode, f"{name}: mode")
    count = int(config["samples"])
    require(count >= 51, f"{name}: insufficient pairs: {count}")
    require(
        summary["samples_requested"] == summary["samples_completed"] == count,
        f"{name}: incomplete summary",
    )
    require(
        summary["run_config_sha256"] == digest(directory / "run-config.json"),
        f"{name}: run config digest",
    )
    for key, expected_path in (
        ("dictionary", WORKLOAD / "dictionary.txt"),
        ("queries", WORKLOAD / "queries" / "hits.txt"),
        ("manifest", WORKLOAD / "provenance.json"),
    ):
        require(
            config[key].endswith(
                f"/benchmarks/cross-language/workload/{expected_path.relative_to(WORKLOAD)}"
            )
            and config[f"{key}_sha256"] == digest(expected_path),
            f"{name}: committed {key} identity",
        )
    require(len(summary["pairs"]) == count, f"{name}: summary pair count")

    with (directory / "host-load-admission.jsonl").open(encoding="utf-8") as source:
        admissions = [json.loads(line) for line in source if line.strip()]
    expected_labels = [
        f"replicate-{replicate}-{arm}-{boundary}"
        for replicate in range(1, count + 1)
        for arm in ("rust", "java")
        for boundary in ("pre", "post")
    ]
    require(
        len(admissions) == len(expected_labels)
        and {row["label"] for row in admissions} == set(expected_labels)
        and all(
            row["admitted"] is True and row["rejection_reasons"] == []
            for row in admissions
        ),
        f"{name}: incomplete or rejected host-load admission",
    )

    samples: dict[str, list[int]] = {"rust": [], "java": []}
    observed_rows: list[tuple[int, str, int, int, str]] = []
    work_count: int | None = None
    for replicate in range(1, count + 1):
        pair_dir = directory / "pairs" / f"replicate-{replicate:06d}"
        pair = read_json(pair_dir / "pair.json")
        label = f"{name}/{replicate}"
        require(pair == summary["pairs"][replicate - 1], f"{label}: summary pair")
        require(pair["replicate"] == replicate, f"{label}: identity")
        require(pair["mode"] == expected_mode, f"{label}: mode")
        require(pair["exact_signature_equal"] is True, f"{label}: unequal signature")
        require(pair["signature"] == summary["exact_signature"], f"{label}: signature")
        require(
            pair["run_config_sha256"] == summary["run_config_sha256"],
            f"{label}: config digest",
        )
        for arm in ("rust", "java"):
            raw_path = pair_dir / f"{arm}.json"
            raw = read_json(raw_path)
            raw_sha = digest(raw_path)
            require(pair[arm]["raw_sha256"] == raw_sha, f"{label}/{arm}: raw digest")
            elapsed = int(pair[arm]["elapsed_ns"])
            if expected_mode == "query":
                current_work = int(raw["workload"]["query_count"])
                raw_samples = raw["measurements"]["samples_ns"]
                raw_signature = raw["measurements"]
            else:
                current_work = int(raw["construct"]["term_count"])
                raw_samples = raw["construct"]["times_ns"]
                raw_signature = raw["construct"]
            require(
                all(
                    raw_signature[key] == value
                    for key, value in summary["exact_signature"].items()
                ),
                f"{label}/{arm}: raw signature",
            )
            require(raw_samples == [elapsed], f"{label}/{arm}: raw sample")
            require(current_work > 0, f"{label}/{arm}: empty workload")
            if work_count is None:
                work_count = current_work
            require(current_work == work_count, f"{label}/{arm}: workload count")
            samples[arm].append(elapsed)
            observed_rows.append((replicate, arm, elapsed, current_work, raw_sha))

    with (directory / "samples.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    require(len(rows) == 2 * count, f"{name}: sample table length")
    for row, expected in zip(rows, observed_rows, strict=True):
        replicate, arm, elapsed, current_work, raw_sha = expected
        require(int(row["replicate"]) == replicate, f"{name}: sample identity")
        require(row["arm"] == arm, f"{name}: sample arm")
        require(int(row["elapsed_ns"]) == elapsed, f"{name}: sample elapsed")
        require(int(row["work_items"]) == current_work, f"{name}: sample workload")
        require(row["raw_sha256"] == raw_sha, f"{name}: sample digest")
        require(
            row["pair_order"] == ("rust-java" if replicate % 2 else "java-rust"),
            f"{name}: pair order",
        )

    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    for arm in ("rust", "java"):
        require(medians[arm] == summary[arm]["median_ns"], f"{name}/{arm}: median")
        require(len(samples[arm]) == summary[arm]["samples"], f"{name}/{arm}: count")
    require(
        max(samples["rust"]) < min(samples["java"]),
        f"{name}: latency distributions overlap",
    )
    require(work_count is not None, f"{name}: missing workload count")
    if expected_mode == "query":
        require(
            medians["rust"] / work_count / 1_000 <= MAX_RUST_QUERY_US,
            f"{name}: Rust query median exceeds parity target",
        )
    return {
        "case": name,
        "mode": expected_mode,
        "pairs": count,
        "work_items_per_sample": work_count,
        "rust_median_ns": medians["rust"],
        "java_median_ns": medians["java"],
        "all_rust_samples_below_all_java_samples": True,
    }


def audit_jvm_matrix() -> dict:
    directory = EVIDENCE / "jvm-parity-full"
    summary = read_json(directory / "summary.json")
    pairs = summary["pair_java"]
    require(summary["validation_errors"] == [], "JVM matrix: validation errors")
    require(summary["cell_count"] == 90, "JVM matrix: timed cell count")
    require(len(pairs) == 45, "JVM matrix: pair count")
    seen: set[tuple[str, int, str]] = set()
    ratios: list[float] = []
    for pair in pairs:
        algorithm = pair["algorithm"]
        distance = int(pair["max_distance"])
        queryset = pair["queryset"]
        key = (algorithm, distance, queryset)
        require(key not in seen, f"JVM matrix: duplicate {key}")
        seen.add(key)
        prefix = f"{algorithm}__d{distance}__{queryset}"
        medians: dict[str, float] = {}
        signatures: dict[str, tuple] = {}
        for target, backend in (
            ("jvm-legacy", "own"),
            ("jvm-vinary", "dynamic_dawg"),
        ):
            stem = f"{target}__{backend}"
            cell_key = f"{stem}__query__{prefix}"
            cell = read_json(directory / "cells" / f"{cell_key}.json")
            twin = read_json(
                directory / "verify-full" / f"{stem}__verify__{prefix}.json"
            )
            jmh_records = read_json(directory / "jmh" / f"{stem}__{prefix}.json")
            require(len(jmh_records) == 1, f"{cell_key}: JMH record count")
            jmh = jmh_records[0]
            params = jmh["params"]
            require(
                params["algorithm"] == algorithm
                and int(params["distance"]) == distance
                and params["queryset"] == queryset,
                f"{cell_key}: JMH identity",
            )
            require(
                params.get("resultMode", "materialized") == "materialized",
                f"{cell_key}: result transport",
            )
            require(
                jmh["forks"] == 2 and jmh["measurementIterations"] == 10,
                f"{cell_key}: JMH protocol",
            )
            unit = jmh["primaryMetric"]["scoreUnit"]
            require(unit in JMH_UNIT_TO_NS, f"{cell_key}: JMH time unit")
            raw_data = jmh["primaryMetric"]["rawData"]
            require(
                len(raw_data) == 2 and all(len(fork) == 10 for fork in raw_data),
                f"{cell_key}: JMH sample shape",
            )
            jmh_samples = [
                int(value * JMH_UNIT_TO_NS[unit]) for fork in raw_data for value in fork
            ]
            observed = cell["measurements"]
            require(observed["samples_ns"] == jmh_samples, f"{cell_key}: timed samples")
            require(observed["sample_count"] == 20, f"{cell_key}: timed count")
            require(
                cell["workload"]["query_count"]
                == twin["workload"]["query_count"]
                == 1_000,
                f"{cell_key}: full query coverage",
            )
            require(
                cell["workload"]["sha256"] == twin["workload"]["sha256"],
                f"{cell_key}: workload identity",
            )
            require(
                cell["dictionary"]["sha256"] == twin["dictionary"]["sha256"],
                f"{cell_key}: dictionary identity",
            )
            query_path = WORKLOAD / "queries" / f"{queryset}.txt"
            require(
                cell["workload"]["file"].endswith(f"/queries/{queryset}.txt")
                and cell["workload"]["sha256"] == digest(query_path)
                and cell["dictionary"]["file"].endswith("/dictionary.txt")
                and cell["dictionary"]["sha256"] == digest(WORKLOAD / "dictionary.txt"),
                f"{cell_key}: committed workload identity",
            )
            signature = tuple(observed[field] for field in SIGNATURE_FIELDS)
            require(
                signature
                == tuple(twin["measurements"][field] for field in SIGNATURE_FIELDS),
                f"{cell_key}: full-coverage correctness twin",
            )
            aggregate = summary["query_cells"][cell_key]
            median = statistics.median(jmh_samples)
            require(aggregate["stats"]["count"] == 20, f"{cell_key}: summary count")
            require(
                aggregate["stats"]["median_ns"] == median, f"{cell_key}: summary median"
            )
            require(
                aggregate["matches_per_pass"] == signature[0], f"{cell_key}: matches"
            )
            require(aggregate["checksum_hex"] == signature[3], f"{cell_key}: checksum")
            medians[target] = median
            signatures[target] = signature
        require(
            signatures["jvm-legacy"] == signatures["jvm-vinary"],
            f"{prefix}: cross-language signature",
        )
        ratio = medians["jvm-legacy"] / medians["jvm-vinary"]
        require(ratio > 1, f"{prefix}: parity loss")
        require(
            math.isclose(
                pair["jvm-vinary(dynamic_dawg)_speedup"], ratio, rel_tol=1e-12
            ),
            f"{prefix}: pair ratio",
        )
        ratios.append(ratio)

    require(
        {algorithm for algorithm, _, _ in seen}
        == {"standard", "transposition", "merge_and_split"},
        "JVM matrix: algorithm coverage",
    )
    for algorithm in ("standard", "transposition", "merge_and_split"):
        expected_querysets = (
            {"hits", "oov", "tr-d1", "tr-d2", "tr-d3"}
            if algorithm == "transposition"
            else {"hits", "oov", "std-d1", "std-d2", "std-d3"}
        )
        require(
            {queryset for a, _, queryset in seen if a == algorithm}
            == expected_querysets,
            f"JVM matrix: {algorithm} queryset coverage",
        )
        require(
            len(
                {
                    (distance, queryset)
                    for a, distance, queryset in seen
                    if a == algorithm
                }
            )
            == 15,
            f"JVM matrix: {algorithm} shape coverage",
        )
        require(
            {distance for a, distance, _ in seen if a == algorithm} == {1, 2, 3},
            f"JVM matrix: {algorithm} distance coverage",
        )
    geometric_mean = math.exp(math.fsum(map(math.log, ratios)) / len(ratios))
    recorded = summary["pair_java_summary"]["overall"]
    require(
        recorded["cells"] == 45 and recorded["treatment_wins"] == 45,
        "JVM matrix: summary wins",
    )
    require(
        math.isclose(recorded["geometric_mean"], geometric_mean, rel_tol=1e-12),
        "JVM matrix: summary geometric mean",
    )
    return {
        "pairs": len(pairs),
        "timed_cells": 2 * len(pairs),
        "full_coverage_twins": 2 * len(pairs),
        "vinary_wins": len(ratios),
        "legacy_over_vinary_geometric_mean": geometric_mean,
    }


def main() -> None:
    result = [audit_case(name, mode) for name, mode in CASES]
    print(
        json.dumps(
            {
                "schema": "liblevenshtein.archived-parity-audit.v1",
                "cases": result,
                "jvm_matrix": audit_jvm_matrix(),
            }
        )
    )


if __name__ == "__main__":
    main()
