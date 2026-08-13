#!/usr/bin/env python3
"""Runner post-fill step (PROTOCOL.md §11).

Merges runner-known facts into a harness-emitted cell JSON, then validates
the merged result against schema/result.schema.json:

  - run_id (results directory name)
  - dictionary.sha256 / workload.sha256 (computed once, cached per run)
  - environment_ref
  - cell_snapshot {cpuset, cpu_mhz_start, cpu_mhz_end, load_avg_1m}
  - memory {max_rss_kib, measured_by, child_cmdline}  (memory mode only,
    parsed from the `/usr/bin/time -v` log the runner captured)

Usage:
  postfill.py <results_dir> <cell.json> --cpuset LIST \
      --mhz-start N --mhz-end N --loadavg X \
      [--time-v-log FILE --child-cmdline STR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SCHEMA = SCRIPTS_DIR.parent / "schema" / "result.schema.json"

sys.path.insert(0, str(SCRIPTS_DIR))
from schema_check import validate_file  # noqa: E402  # type: ignore[import-not-found]


def sha256_cached(results_dir: Path, path: Path) -> str:
    """Digest cache: one hash per distinct input file per results dir."""
    cache_path = results_dir / ".sha256-cache.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    key = str(path.resolve())
    if key not in cache:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return cache[key]


def parse_max_rss_kib(time_v_log: Path) -> int:
    text = time_v_log.read_text(errors="replace")
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if not match:
        raise SystemExit(f"postfill: no max-RSS line in {time_v_log}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("cell_json", type=Path)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--mhz-start", type=float, required=True)
    parser.add_argument("--mhz-end", type=float, required=True)
    parser.add_argument("--loadavg", type=float, required=True)
    parser.add_argument("--time-v-log", type=Path)
    parser.add_argument("--child-cmdline")
    args = parser.parse_args()

    cell = json.loads(args.cell_json.read_text())
    cell["run_id"] = args.results_dir.name
    cell["environment_ref"] = "../environment.json"
    cell["cell_snapshot"] = {
        "cpuset": args.cpuset,
        "cpu_mhz_start": args.mhz_start,
        "cpu_mhz_end": args.mhz_end,
        "load_avg_1m": args.loadavg,
    }

    dictionary_file = Path(cell["dictionary"]["file"])
    workload_file = Path(cell["workload"]["file"])
    cell["dictionary"]["sha256"] = sha256_cached(args.results_dir, dictionary_file)
    cell["workload"]["sha256"] = sha256_cached(args.results_dir, workload_file)

    if cell["mode"] == "memory":
        if not args.time_v_log:
            raise SystemExit("postfill: memory mode requires --time-v-log")
        cell["memory"] = {
            "max_rss_kib": parse_max_rss_kib(args.time_v_log),
            "measured_by": "/usr/bin/time -v",
        }
        if args.child_cmdline:
            cell["memory"]["child_cmdline"] = args.child_cmdline

    args.cell_json.write_text(json.dumps(cell, indent=2) + "\n")

    errors = validate_file(args.cell_json, SCHEMA)
    if errors:
        for error in errors:
            print(f"postfill: INVALID {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
