#!/usr/bin/env python3
"""Convert JMH JSON output into unified result-schema cell files.

The Java pair is the one harness whose timing JMH owns (forks, managed
warmup). Its benchmark method executes ONE FULL PASS over the query set per
invocation, so `primaryMetric.rawData` (per-fork arrays of per-iteration
average times) flattens directly into `measurements.samples_ns`. The vinary
benchmark also records whether results were materialized as managed objects or
reduced through the borrowed descriptor view; the emitted sample definition
must preserve that distinction.

Checksums/triples are NOT produced by JMH runs; they come from the twin
verify cells (VerifyMain/LegacyVerifyMain runs recorded by the gate). This
converter looks the matching verify cell up in <results>/gate/ and copies
its (matches, bytes, distsum, checksum) into the converted cell, so the
timed cell carries the same correctness evidence as every other target's.

Usage:
  jmh_to_result.py <results_dir> <jmh_json> --target jvm-vinary|jvm-legacy \
      --backend dynamic_dawg|double_array_trie|own \
      --runtime-version "openjdk 26.0.2" --library-version 0.10.0|3.0.0 \
      [--artifact-id io.vinarytree:liblevenshtein:0.10.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
XL = SCRIPTS_DIR.parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

TIME_UNIT_TO_NS = {
    "ns/op": 1.0,
    "us/op": 1e3,
    "ms/op": 1e6,
    "s/op": 1e9,
}


def queryset_file(queryset: str) -> Path:
    return XL / "workload" / "queries" / f"{queryset}.txt"


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_bytes().split(b"\n") if line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("jmh_json", type=Path)
    parser.add_argument("--target", required=True, choices=("jvm-vinary", "jvm-legacy"))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--library-version", required=True)
    parser.add_argument("--artifact-id", default="")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    cell_dir = results_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(args.jmh_json.read_text())
    if not records:
        raise SystemExit(f"jmh_to_result: {args.jmh_json} contains no benchmark records")

    converted = 0
    for record in records:
        params = record.get("params", {})
        algorithm = params["algorithm"]
        distance = int(params["distance"])
        queryset = params["queryset"]
        result_mode = (
            params.get("resultMode", "materialized")
            if args.target == "jvm-vinary"
            else "materialized"
        )
        if result_mode not in ("borrowed", "materialized"):
            raise SystemExit(
                f"jmh_to_result: unknown resultMode {result_mode!r} in "
                f"{record['benchmark']}"
            )
        metric = record["primaryMetric"]
        unit = metric["scoreUnit"]
        if unit not in TIME_UNIT_TO_NS:
            raise SystemExit(f"jmh_to_result: unhandled JMH score unit {unit!r}")
        scale = TIME_UNIT_TO_NS[unit]
        samples_ns = [
            int(value * scale)
            for fork in metric["rawData"]
            for value in fork
        ]
        if not samples_ns:
            raise SystemExit(f"jmh_to_result: empty rawData in {record['benchmark']}")

        # Correctness evidence from the twin verify cell. Prefer the
        # full-coverage twin (verify-full/, same query count as the timed
        # pass); fall back to the gate twin only when the full one is absent,
        # and then say so in the notes — a gate twin covers only the first
        # --gate-limit queries, so its triple is NOT a per-pass count.
        twin_name = (
            f"{args.target}__{args.backend}__verify__{algorithm}__d{distance}__{queryset}.json"
        )
        full_path = results_dir / "verify-full" / twin_name
        gate_path = results_dir / "gate" / twin_name
        partial_twin = False
        if full_path.exists():
            verify_path = full_path
        elif gate_path.exists():
            verify_path = gate_path
            partial_twin = True
        else:
            raise SystemExit(
                f"jmh_to_result: missing twin verify cell {twin_name} — run "
                "`run-jvm-pair.sh verify-full <results>` (or the gate) first"
            )
        twin = json.loads(verify_path.read_text())
        verify = twin["measurements"]
        twin_queries = twin["workload"]["query_count"]

        forks = int(record.get("forks", 0))
        warmup_iterations = int(record.get("warmupIterations", 0))
        measurement_iterations = int(record.get("measurementIterations", 0))
        structure = {
            "dynamic_dawg": "dynamic_dawg",
            "double_array_trie": "double_array_trie",
            "own": "legacy_sorted_dawg",
        }[args.backend]

        cell = {
            "schema_version": "1.0.0",
            "suite": "cross-language-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": {
                "language": "java",
                "implementation": "legacy" if args.target == "jvm-legacy" else "vinary-tree",
                "backend": "jvm-legacy-jar" if args.target == "jvm-legacy" else "jvm-ffm",
                "runtime_version": args.runtime_version,
                "library_version": args.library_version,
                "artifact": {
                    "kind": "maven" if args.target == "jvm-legacy" else "local-build",
                    "id": args.artifact_id
                    or (
                        "com.github.universal-automata:liblevenshtein:3.0.0"
                        if args.target == "jvm-legacy"
                        else "io.vinarytree:liblevenshtein:0.10.0"
                    ),
                },
            },
            "dictionary": {
                "file": str(XL / "workload" / "dictionary.txt"),
                "sha256": sha256_of(XL / "workload" / "dictionary.txt"),
                "term_count": count_lines(XL / "workload" / "dictionary.txt"),
                "structure": structure,
                "unit_domain": "jvm_string" if args.target == "jvm-legacy" else "unicode_scalar",
            },
            "workload": {
                "queryset": queryset,
                "file": str(queryset_file(queryset)),
                "sha256": sha256_of(queryset_file(queryset)),
                "query_count": count_lines(queryset_file(queryset)),
            },
            "algorithm": algorithm,
            "max_distance": distance,
            "mode": "query",
            "protocol": {
                "timer": "monotonic",
                "harness": "jmh",
                "warmup_seconds_min": 0,
                "warmup_passes": warmup_iterations,
                "samples_requested": forks * measurement_iterations,
                "sample_definition": (
                    "one full pass over the query set; every cursor fully drained and "
                    + (
                        "(term bytes, distance) consumed through borrowed descriptors"
                        if result_mode == "borrowed"
                        else "(term, distance) materialized"
                    )
                ),
                "batch_size": None if args.target == "jvm-legacy" else 256,
                "wall_cap_seconds": 0,
                "jmh": {
                    "forks": forks,
                    "warmup_iterations": warmup_iterations,
                    "measurement_iterations": measurement_iterations,
                    "iteration_seconds": 2,
                },
            },
            "measurements": {
                "samples_ns": samples_ns,
                "sample_count": len(samples_ns),
                "matches_per_pass": verify["matches_per_pass"],
                "term_bytes_per_pass": verify["term_bytes_per_pass"],
                "distance_sum_per_pass": verify["distance_sum_per_pass"],
                "checksum_hex": verify["checksum_hex"],
            },
            "status": "ok",
            "notes": [
                f"converted from JMH record {record['benchmark']}",
                f"JVM result transport: {result_mode}",
                (
                    "correctness evidence (triple+checksum) copied from the full-coverage "
                    f"verify twin over all {twin_queries} queries — the same pass JMH timed"
                    if not partial_twin
                    else (
                        "PARTIAL EVIDENCE: no full-coverage twin available; triple+checksum "
                        f"copied from the {twin_queries}-query gate twin and therefore describe "
                        "a subset of the timed pass"
                    )
                ),
            ],
        }
        out = (
            cell_dir
            / f"{args.target}__{args.backend}__query__{algorithm}__d{distance}__{queryset}.json"
        )
        out.write_text(json.dumps(cell, indent=2) + "\n")
        converted += 1

    print(f"jmh_to_result: converted {converted} record(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
