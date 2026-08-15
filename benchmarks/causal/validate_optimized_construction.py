#!/usr/bin/env python3
"""Validate the accepted freeze-once construction invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_RESULT = {
    "matches": 18_514,
    "term_bytes": 82_131,
    "distance_sum": 36_201,
    "checksum_u64": 7_775_666_136_087_164_888,
}

EXPECTED_COMMON_WORK = {
    "term_insert_attempts": 79_343,
    "input_units": 673_918,
    "version_loads": 0,
    "path_units_walked": 0,
    "edge_lists_cloned": 0,
    "edge_arcs_cloned": 0,
    "nodes_created": 29_133,
    "nodes_dropped": 0,
    "graph_versions_created": 1,
    "cas_publications": 0,
    "cas_retries": 0,
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorted", required=True, type=Path)
    parser.add_argument("--unordered", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = {"sorted": load(args.sorted), "unordered": load(args.unordered)}
    failures: list[str] = []

    def equal(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            failures.append(f"{label}: expected {expected!r}, observed {actual!r}")

    for label, arm in arms.items():
        equal(f"{label} schema", arm["schema"], "liblevenshtein.causal-work.v1")
        equal(f"{label} term_count", arm["term_count"], 79_343)
        for field, expected in EXPECTED_RESULT.items():
            equal(f"{label} {field}", arm[field], expected)
        for field, expected in EXPECTED_COMMON_WORK.items():
            equal(f"{label} construction_work.{field}", arm["construction_work"][field], expected)

    equal("sorted constructor", arms["sorted"]["constructor"], "from_sorted_terms")
    equal("sorted batch_sort_calls", arms["sorted"]["construction_work"]["batch_sort_calls"], 0)
    equal("sorted batch_sort_terms", arms["sorted"]["construction_work"]["batch_sort_terms"], 0)
    equal("sorted batch_sort_units", arms["sorted"]["construction_work"]["batch_sort_units"], 0)

    equal("unordered constructor", arms["unordered"]["constructor"], "from_terms")
    equal("unordered batch_sort_calls", arms["unordered"]["construction_work"]["batch_sort_calls"], 1)
    equal("unordered batch_sort_terms", arms["unordered"]["construction_work"]["batch_sort_terms"], 79_343)
    equal("unordered batch_sort_units", arms["unordered"]["construction_work"]["batch_sort_units"], 673_918)
    equal("sorted/unordered query work", arms["sorted"]["work"], arms["unordered"]["work"])

    result = {
        "schema": "liblevenshtein.optimized-construction-gate.v1",
        "status": "fail" if failures else "pass",
        "arms": sorted(arms),
        "checks": 2 * (2 + len(EXPECTED_RESULT) + len(EXPECTED_COMMON_WORK)) + 8,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
