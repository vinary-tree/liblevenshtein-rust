#!/usr/bin/env python3
"""Audit and summarize a strict pure-Rust/legacy-Java pair directory.

The pair runner deliberately stores raw one-process samples.  This companion
turns those immutable rows into a flat CSV for statistical tooling and adds
the protocol-implementation closure that cannot be embedded recursively in
the runner itself.  It never edits raw pair data or the runner summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aggregate_module(scripts: Path):
    path = scripts / "aggregate.py"
    spec = importlib.util.spec_from_file_location("cross_language_aggregate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aggregate helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pooled_cohens_d(left: list[int], right: list[int]) -> float:
    left_variance = statistics.variance(left)
    right_variance = statistics.variance(right)
    degrees = len(left) + len(right) - 2
    pooled = math.sqrt(
        ((len(left) - 1) * left_variance + (len(right) - 1) * right_variance)
        / degrees
    )
    return (statistics.fmean(left) - statistics.fmean(right)) / pooled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    scripts = Path(__file__).resolve().parent
    repository = scripts.parents[2]
    run_config_path = result_dir / "run-config.json"
    summary_path = result_dir / "summary.json"
    if not run_config_path.is_file() or not summary_path.is_file():
        parser.error("result directory must contain run-config.json and summary.json")

    run_config = json.loads(run_config_path.read_text())
    summary = json.loads(summary_path.read_text())
    requested = int(run_config["samples"])
    if summary["samples_requested"] != requested or summary["samples_completed"] != requested:
        raise RuntimeError("summary is not complete for the configured sample count")

    mode = run_config["mode"]
    if mode == "query":
        work_item_kind = "query"
    elif mode == "construct":
        work_item_kind = "term"
    else:
        raise RuntimeError(f"unsupported pair mode: {mode}")

    rows: list[dict[str, object]] = []
    samples: dict[str, list[int]] = {"rust": [], "java": []}
    work_items_per_sample: int | None = None
    expected_signature = summary["exact_signature"]
    for replicate in range(1, requested + 1):
        pair_dir = result_dir / "pairs" / f"replicate-{replicate:06d}"
        pair = json.loads((pair_dir / "pair.json").read_text())
        if pair["replicate"] != replicate or pair["signature"] != expected_signature:
            raise RuntimeError(f"pair identity or signature drift at replicate {replicate}")
        for arm in ("rust", "java"):
            raw_path = pair_dir / f"{arm}.json"
            raw = json.loads(raw_path.read_text())
            if sha256(raw_path) != pair[arm]["raw_sha256"]:
                raise RuntimeError(f"raw digest mismatch: {raw_path}")
            if mode == "query":
                arm_work_items = int(raw["workload"]["query_count"])
                raw_samples = raw["measurements"]["samples_ns"]
            else:
                arm_work_items = int(raw["construct"]["term_count"])
                raw_samples = raw["construct"]["times_ns"]
            if work_items_per_sample is None:
                work_items_per_sample = arm_work_items
            elif arm_work_items != work_items_per_sample:
                raise RuntimeError(
                    f"{work_item_kind} count drifted between arms or replicates"
                )
            elapsed = int(pair[arm]["elapsed_ns"])
            if raw_samples != [elapsed]:
                raise RuntimeError(f"raw sample mismatch: {raw_path}")
            samples[arm].append(elapsed)
            rows.append(
                {
                    "replicate": replicate,
                    "pair_order": "rust-java" if replicate % 2 else "java-rust",
                    "arm": arm,
                    "elapsed_ns": elapsed,
                    "work_item_kind": work_item_kind,
                    "work_items": arm_work_items,
                    "ns_per_work_item": elapsed / arm_work_items,
                    "raw_sha256": pair[arm]["raw_sha256"],
                }
            )

    assert work_items_per_sample is not None
    sample_path = result_dir / "samples.csv"
    sample_tmp = sample_path.with_suffix(".csv.tmp")
    with sample_tmp.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sample_tmp.replace(sample_path)

    aggregate = load_aggregate_module(scripts)
    rust_stats = aggregate.robust_stats(samples["rust"])
    java_stats = aggregate.robust_stats(samples["java"])
    paired_differences = [
        rust - java for rust, java in zip(samples["rust"], samples["java"], strict=True)
    ]

    protocol_paths = [
        scripts / "run-pure-rust-legacy-java-pair.sh",
        scripts / "run-with-contention-monitor.sh",
        scripts / "timed-proc-guard.sh",
        scripts / "java-execution-closure.py",
        scripts / "schema_check.py",
        scripts / "aggregate.py",
        repository / "benchmarks" / "causal" / "host-load-admission.py",
        repository
        / "benchmarks"
        / "cross-language"
        / "schema"
        / "pure-rust-legacy-java-pair.schema.json",
        repository / "benchmarks" / "cross-language" / "schema" / "result.schema.json",
        Path(__file__).resolve(),
    ]
    protocol_closure = [
        {
            "path": str(path.relative_to(repository)),
            "sha256": sha256(path),
        }
        for path in protocol_paths
    ]
    closure_digest = hashlib.sha256(
        "".join(f"{entry['path']}\0{entry['sha256']}\n" for entry in protocol_closure).encode()
    ).hexdigest()

    rejection_path = result_dir / "host-load-rejections.jsonl"
    analysis = {
        "schema": "liblevenshtein.pure-rust-legacy-java-pair-analysis.v2",
        "mode": mode,
        "samples": requested,
        "work_item_kind": work_item_kind,
        "work_items_per_sample": work_items_per_sample,
        "exact_signature": expected_signature,
        "rust": rust_stats,
        "java": java_stats,
        "rust_median_ns_per_work_item": (
            rust_stats["median_ns"] / work_items_per_sample
        ),
        "java_median_ns_per_work_item": (
            java_stats["median_ns"] / work_items_per_sample
        ),
        "java_over_rust_median_x": java_stats["median_ns"] / rust_stats["median_ns"],
        "rust_minus_java_pooled_cohens_d": pooled_cohens_d(
            samples["rust"], samples["java"]
        ),
        "paired_difference_ns": aggregate.robust_stats(paired_differences),
        "median_ci95_nonoverlap": (
            rust_stats["median_ci95_ns"][1] < java_stats["median_ci95_ns"][0]
        ),
        "accepted_admissions": sum(
            1 for line in (result_dir / "host-load-admission.jsonl").read_text().splitlines()
            if line.strip()
        ),
        "rejected_admissions": (
            sum(1 for line in rejection_path.read_text().splitlines() if line.strip())
            if rejection_path.is_file()
            else 0
        ),
        "evidence": {
            "run_config_sha256": sha256(run_config_path),
            "runner_summary_sha256": sha256(summary_path),
            "samples_csv_sha256": sha256(sample_path),
            "accepted_admissions_sha256": sha256(
                result_dir / "host-load-admission.jsonl"
            ),
            "rejected_admissions_sha256": (
                sha256(rejection_path) if rejection_path.is_file() else None
            ),
        },
        "protocol_closure_sha256": closure_digest,
        "protocol_closure": protocol_closure,
    }
    analysis_path = result_dir / "analysis.json"
    analysis_tmp = analysis_path.with_suffix(".json.tmp")
    analysis_tmp.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    analysis_tmp.replace(analysis_path)
    print(analysis_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
