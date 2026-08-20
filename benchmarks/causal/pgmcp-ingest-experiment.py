#!/usr/bin/env python3
"""Ingest filtered CSV sample arms into a pre-registered pgmcp experiment.

The pgmcp command-line adapter accepts scalar ``KEY=VALUE`` arguments only;
experiment samples, host metadata, and command specifications are structured
values. This helper speaks the same local MCP transport as the cross-language
uploader and keeps the submitted distributions raw and auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def load_mcp_client(repo: Path):
    module_path = repo / "benchmarks/cross-language/scripts/pgmcp-upload.py"
    spec = importlib.util.spec_from_file_location("pgmcp_upload", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MCP client from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.McpClient


def parse_mapping(value: str) -> tuple[str, tuple[str, str]]:
    source, separator, destination = value.partition("=")
    label, kind_separator, kind = destination.partition(":")
    if not separator or not kind_separator or kind not in {"control", "treatment", "baseline"}:
        raise argparse.ArgumentTypeError(
            "arm mappings must be SOURCE=LABEL:control|treatment|baseline"
        )
    return source, (label, kind)


def parse_filter(value: str) -> tuple[str, str]:
    column, separator, expected = value.partition("=")
    if not separator or not column:
        raise argparse.ArgumentTypeError("filters must be COLUMN=VALUE")
    return column, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--hypothesis-id", type=int, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--arm-column", required=True)
    parser.add_argument("--arm", action="append", type=parse_mapping, required=True)
    parser.add_argument("--filter", action="append", type=parse_filter, default=[])
    parser.add_argument("--unit")
    parser.add_argument("--git-ref")
    parser.add_argument("--source", default="external_benchmark")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--unit-key-column")
    parser.add_argument("--host-meta", type=json.loads, default={})
    parser.add_argument("--run-plan", type=json.loads, default={})
    parser.add_argument("--endpoint", default="http://127.0.0.1:3100/mcp")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    artifact = args.csv_path.resolve()
    if not artifact.is_file():
        parser.error(f"CSV does not exist: {artifact}")
    mappings = dict(args.arm)
    filters = dict(args.filter)
    samples: dict[str, list[float]] = {source: [] for source in mappings}
    unit_keys: dict[str, list[str]] = {source: [] for source in mappings}

    with artifact.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {args.value_column, args.arm_column, *filters}
        if args.unit_key_column:
            required.add(args.unit_key_column)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if any(row[column] != expected for column, expected in filters.items()):
                continue
            source_arm = row[args.arm_column]
            if source_arm not in mappings:
                continue
            samples[source_arm].append(float(row[args.value_column]))
            if args.unit_key_column:
                unit_keys[source_arm].append(row[args.unit_key_column])

    if any(not values for values in samples.values()):
        empty = sorted(source for source, values in samples.items() if not values)
        parser.error(f"no samples found for arms: {empty}")

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    client_type = load_mcp_client(repo)
    client = client_type(args.endpoint)
    client.initialize()
    results = []
    for source_arm, values in samples.items():
        label, kind = mappings[source_arm]
        payload = {
            "experiment_id": args.experiment_id,
            "hypothesis_id": args.hypothesis_id,
            "arm_label": label,
            "arm_kind": kind,
            "metric": args.metric,
            "samples": values,
            "source": args.source,
            "is_warmup": False,
            "command_spec": {
                "artifact": str(artifact.relative_to(repo)),
                "sha256": digest,
            },
            "host_meta": args.host_meta,
            "run_plan": args.run_plan,
        }
        if args.unit:
            payload["unit"] = args.unit
        if args.git_ref:
            payload["git_ref"] = args.git_ref
        if args.seed is not None:
            payload["seed"] = args.seed
        if args.unit_key_column:
            payload["unit_keys"] = unit_keys[source_arm]
        results.append(client._call("experiment_record_measurement", payload))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
