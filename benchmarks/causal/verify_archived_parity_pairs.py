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
import statistics
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent / "evidence" / "2026-08-19"
CASES = (
    ("direct-standard-d1-hits", "query"),
    ("direct-construction-from-terms", "construct"),
    ("direct-construction-from-sorted-terms", "construct"),
)
MAX_RUST_QUERY_US = 51.2


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
    require(len(summary["pairs"]) == count, f"{name}: summary pair count")

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


def main() -> None:
    result = [audit_case(name, mode) for name, mode in CASES]
    print(
        json.dumps(
            {"schema": "liblevenshtein.archived-parity-audit.v1", "cases": result}
        )
    )


if __name__ == "__main__":
    main()
