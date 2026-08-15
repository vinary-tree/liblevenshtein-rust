#!/usr/bin/env python3
"""Validate causal-counter identities used by the Java parity gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--construction", required=True, type=Path)
    parser.add_argument("--sorted-construction", required=True, type=Path)
    parser.add_argument("--batch-arm", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native = load(args.native)
    resource = load(args.resource)
    construction = load(args.construction)
    sorted_construction = load(args.sorted_construction)
    failures: list[str] = []

    def equal(label: str, left: Any, right: Any) -> None:
        if left != right:
            failures.append(f"{label}: {left!r} != {right!r}")

    for field in ("matches", "term_bytes", "distance_sum", "checksum_u64"):
        equal(f"native/resource {field}", native[field], resource[field])

    native_work = native["work"]
    consumer = resource["consumer_work"]
    for field in (
        "dictionary_intersections",
        "final_checks",
        "edges_enumerated",
        "transition_attempts",
        "transition_accepted",
        "characteristic_vectors",
        "characteristic_units",
        "state_bytes_copied",
        "state_bytes_enqueued",
        "pool_misses",
        "matches_materialized",
    ):
        equal(f"native/resource work {field}", native_work[field], consumer[field])

    provider = resource["provider_work"]
    equal("resource edge callbacks", consumer["foreign_edge_callbacks"], provider["edges_calls"])
    equal("resource edge pages", consumer["foreign_edge_pages"], provider["edges_calls"])
    equal(
        "resource descriptors consumer/provider",
        consumer["foreign_edge_descriptors"],
        provider["descriptors_cloned"],
    )
    equal(
        "resource descriptors/transitions",
        provider["descriptors_cloned"],
        consumer["transition_attempts"],
    )
    equal(
        "resource snapshot count",
        provider["snapshots_created"],
        resource["query_count"],
    )

    build = construction["construction_work"]
    sorted_build = sorted_construction["construction_work"]
    for field in (
        "term_insert_attempts",
        "input_units",
        "version_loads",
        "path_units_walked",
        "edge_lists_cloned",
        "edge_arcs_cloned",
        "nodes_created",
        "nodes_dropped",
        "graph_versions_created",
        "cas_publications",
        "cas_retries",
    ):
        equal(f"ordered input does not alter path-copy work: {field}", build[field], sorted_build[field])
    equal("one graph publication per insertion", build["cas_publications"], build["term_insert_attempts"])
    equal("single-threaded construction has no CAS retries", build["cas_retries"], 0)
    equal("one cloned edge list per walked unit", build["edge_lists_cloned"], build["path_units_walked"])
    if build["nodes_created"] <= build["nodes_dropped"]:
        failures.append("construction should retain a positive live-node population")

    reference = (
        resource["matches"],
        resource["term_bytes"],
        resource["distance_sum"],
        resource["checksum_u64"],
        resource["consumer_work"],
        resource["provider_work"],
    )
    for arm_path in args.batch_arm:
        arm = load(arm_path)
        observed = (
            arm["matches"],
            arm["term_bytes"],
            arm["distance_sum"],
            arm["checksum_u64"],
            arm["consumer_work"],
            arm["provider_work"],
        )
        equal(f"batch-size arm {arm_path}", observed, reference)

    result = {
        "schema": "liblevenshtein.java-parity-causal-gate.v1",
        "status": "fail" if failures else "pass",
        "checks": 22 + 12 + len(args.batch_arm),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
