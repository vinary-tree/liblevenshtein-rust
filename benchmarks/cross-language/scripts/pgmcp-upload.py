#!/usr/bin/env python3
"""Incremental raw-cell uploader: results/<run>/cells/*.json (and
hypothesis/*.json) → the pgmcp data table `xlang_bench_cells`.

Speaks MCP Streamable HTTP directly to the local pgmcp daemon
(http://127.0.0.1:3100/mcp): initialize → notifications/initialized →
tools/call {data_table_insert, data_table_delete}. The pgmcp table is the
system of record for raw benchmark data (standing user directive), so this
runs automatically at the end of every timed sweep — never only by hand.

CONTENT-ADDRESSED LEDGER. `.pgmcp-uploaded.txt` records `<key>\\t<sha256>`
per cell. A cell whose file changed after upload (e.g. JMH cells
re-converted against full-coverage verify twins) is *replaced*: its existing
row is deleted by (run_id, cell) filter, then the new row is inserted. So
re-running is idempotent and the table never accumulates stale duplicates
of the same coordinate.

Usage: pgmcp-upload.py <results_dir> [--endpoint URL] [--batch N] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import urllib.request
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
TABLE = "xlang_bench_cells"
PROJECT = "liblevenshtein-rust"


class McpClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.session_id: str | None = None
        self.next_id = 1

    def _post(self, payload: dict, expect_response: bool) -> dict | None:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **({"Mcp-Session-Id": self.session_id} if self.session_id else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if self.session_id is None:
                self.session_id = response.headers.get("Mcp-Session-Id")
            if not expect_response:
                response.read()
                return None
            content_type = response.headers.get("Content-Type", "")
            body = response.read().decode()
        if "text/event-stream" in content_type:
            # Parse the final non-empty data: frame (the stream opens with an
            # empty priming frame before the JSON-RPC response).
            message = None
            for line in body.splitlines():
                if line.startswith("data:"):
                    payload_text = line[5:].strip()
                    if payload_text:
                        message = json.loads(payload_text)
            if message is None:
                raise RuntimeError("SSE stream carried no data frames")
            return message
        return json.loads(body)

    def request(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params,
        }
        self.next_id += 1
        message = self._post(payload, expect_response=True)
        if message is None or "error" in message:
            raise RuntimeError(f"{method} failed: {message}")
        return message["result"]

    def notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method}, expect_response=False)

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "xlang-bench-uploader", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized")

    def _call(self, name: str, arguments: dict) -> dict:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(f"{name} error: {result}")
        content = result.get("content", [])
        text = content[0].get("text", "{}") if content else "{}"
        return json.loads(text)

    def insert_rows(self, rows: list[dict], source: str) -> int:
        payload = self._call(
            "data_table_insert",
            {"table": TABLE, "project": PROJECT, "source": source, "rows": rows},
        )
        return payload.get("inserted", 0)

    def delete_cell(self, run_id: str, cell: str) -> int:
        """Remove any existing row for this coordinate (supersede-in-place).

        pgmcp filters are a SEQUENCE of {field, op, value} conditions (a map
        is rejected with "invalid type: map, expected a sequence").
        """
        payload = self._call(
            "data_table_delete",
            {
                "table": TABLE,
                "project": PROJECT,
                "filter": [
                    {"field": "run_id", "op": "eq", "value": run_id},
                    {"field": "cell", "op": "eq", "value": cell},
                ],
            },
        )
        return payload.get("deleted", 0)


def cell_to_row(run_id: str, path: Path, arm: str | None = None) -> dict:
    data = json.loads(path.read_text())
    name = path.stem
    row: dict = {
        "run_id": run_id if arm is None else f"{run_id}/{arm}",
        "cell": name,
        "target": name.split("__")[0],
        "backend": data["dictionary"]["structure"],
        "mode": data["mode"],
        "algorithm": data["algorithm"],
        "max_distance": data["max_distance"],
        "queryset": data["workload"]["queryset"],
        "status": data["status"],
        "harness": data["protocol"]["harness"],
        "notes": data["notes"],
    }
    if "measurements" in data:
        measurements = data["measurements"]
        row.update(
            {
                "query_count": data["workload"]["query_count"],
                "matches_per_pass": measurements["matches_per_pass"],
                "checksum_hex": measurements["checksum_hex"],
                "sample_count": measurements["sample_count"],
            }
        )
        samples = measurements["samples_ns"]
        if samples:
            ordered = sorted(samples)
            median = statistics.median(ordered)
            row.update(
                {
                    "median_ns": median,
                    "mad_ns": statistics.median([abs(x - median) for x in ordered]),
                    "p10_ns": ordered[int(0.10 * (len(ordered) - 1))],
                    "p90_ns": ordered[int(0.90 * (len(ordered) - 1))],
                    "us_per_query": median / 1000.0 / data["workload"]["query_count"],
                    "queries_per_second": 1e9 * data["workload"]["query_count"] / median,
                    "samples_ns": samples,
                }
            )
    if "construct" in data:
        row["construct_median_ns"] = statistics.median(data["construct"]["times_ns"])
        row["samples_ns"] = data["construct"]["times_ns"]
    if "memory" in data:
        row["max_rss_kib"] = data["memory"]["max_rss_kib"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:3100/mcp")
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    run_id = results_dir.name

    ledger_path = results_dir / ".pgmcp-uploaded.txt"
    ledger: dict[str, str] = {}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            key, _, digest = line.partition("\t")
            ledger[key] = digest  # empty digest = pre-content-addressing entry

    def digest_of(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # (key, path, is_replacement)
    pending: list[tuple[str, Path, bool]] = []
    sources: list[tuple[str, Path]] = []
    for path in sorted((results_dir / "cells").glob("*.json")):
        sources.append((f"cells/{path.stem}", path))
    hypothesis_dir = results_dir / "hypothesis"
    if hypothesis_dir.is_dir():
        for path in sorted(hypothesis_dir.glob("*.json")):
            sources.append((f"hypothesis/{path.stem}", path))

    for key, path in sources:
        current = digest_of(path)
        if key not in ledger:
            pending.append((key, path, False))
        elif ledger[key] != current:
            # Content changed since upload (e.g. re-converted JMH cell):
            # supersede the stored row rather than duplicating the coordinate.
            pending.append((key, path, True))

    if not pending:
        print(
            f"pgmcp-upload: up to date ({len(sources)} cell(s) already in {TABLE})",
            file=sys.stderr,
        )
        return 0
    replacements = sum(1 for _, _, replace in pending if replace)
    print(
        f"pgmcp-upload: {len(pending)} cell(s) to upload "
        f"({len(pending) - replacements} new, {replacements} superseded)",
        file=sys.stderr,
    )
    if args.dry_run:
        for key, _, replace in pending:
            print(f"{'replace' if replace else 'new    '} {key}")
        return 0

    client = McpClient(args.endpoint)
    client.initialize()

    total = 0
    for start in range(0, len(pending), args.batch):
        batch = pending[start : start + args.batch]
        for key, path, replace in batch:
            if replace:
                client.delete_cell(run_id if not key.startswith("hypothesis/") else f"{run_id}/hypothesis", path.stem)
        rows = [
            cell_to_row(run_id, path, arm="hypothesis" if key.startswith("hypothesis/") else None)
            for key, path, _ in batch
        ]
        inserted = client.insert_rows(
            rows, source=f"benchmarks/cross-language results/{run_id} (pgmcp-upload.py)"
        )
        if inserted != len(rows):
            raise RuntimeError(f"batch insert mismatch: {inserted} != {len(rows)}")
        for key, path, _ in batch:
            ledger[key] = digest_of(path)
        with ledger_path.open("w") as fh:
            for key in sorted(ledger):
                fh.write(f"{key}\t{ledger[key]}\n")
        total += inserted
        print(f"pgmcp-upload: {total}/{len(pending)} uploaded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
