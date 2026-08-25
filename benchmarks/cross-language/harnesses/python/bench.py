#!/usr/bin/env python3
"""Python harness for the cross-language benchmark program.

Implements harnesses/common/PROTOCOL.md over the ctypes facades
(liblevenshtein + libdictenstein). The runner
provides PYTHONPATH to the three binding source trees and the
LIBLEVENSHTEIN_LIBRARY / LIBDICTENSTEIN_LIBRARY release overrides.

Fairness notes (PROTOCOL.md §10): CPython, GC enabled, PYTHONHASHSEED=0
(runner-set); plain Match iteration — the migration-realistic path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import libdictenstein  # type: ignore[import-not-found]  # runner sets PYTHONPATH
import liblevenshtein  # type: ignore[import-not-found]

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
WALL_CAP_SECONDS = 300.0
SAMPLE_DEFINITION = (
    "one full pass over the query set; every cursor fully drained and "
    "(term, distance) materialized"
)

ALGORITHMS = {
    "standard": liblevenshtein.Algorithm.STANDARD,
    "transposition": liblevenshtein.Algorithm.TRANSPOSITION,
    "merge_and_split": liblevenshtein.Algorithm.MERGE_AND_SPLIT,
    "damerau_levenshtein": liblevenshtein.Algorithm.DAMERAU_LEVENSHTEIN,
}


def entry_hash(term: str, distance: int) -> int:
    hash_value = FNV_OFFSET
    for byte in term.encode("utf-8"):
        hash_value = ((hash_value ^ byte) * FNV_PRIME) & MASK64
    hash_value = (hash_value * FNV_PRIME) & MASK64  # XOR with 0x00 is identity
    for i in range(8):
        hash_value = ((hash_value ^ ((distance >> (8 * i)) & 0xFF)) * FNV_PRIME) & MASK64
    return hash_value


def self_test() -> None:
    def fnv(data: bytes) -> int:
        hash_value = FNV_OFFSET
        for byte in data:
            hash_value = ((hash_value ^ byte) * FNV_PRIME) & MASK64
        return hash_value

    assert fnv(b"") == 0xCBF29CE484222325, "fnv empty"
    assert fnv(b"a") == 0xAF63DC4C8601EC8C, "fnv 'a'"
    assert entry_hash("cat", 1) == 0x9697FA3E50464BC4, "entry(cat,1)"
    assert entry_hash("cat", 0) == 0xB592C1475B3595E5, "entry(cat,0)"
    assert entry_hash("cot", 1) == 0xB8ACC5D3816BCDEA, "entry(cot,1)"
    assert (entry_hash("cat", 0) + entry_hash("cot", 1)) & MASK64 == 0x6E3F871ADCA163CF


def die(message: str) -> NoReturn:
    print(f"bench-cross-python: {message}", file=sys.stderr)
    sys.exit(2)


def read_lines(path: Path) -> list[str]:
    data = path.read_text(encoding="utf-8")
    lines = [line for line in data.split("\n") if line]
    if not lines:
        die(f"{path} contains no lines")
    return lines


def assert_strictly_sorted(lines: list[str], path: Path) -> None:
    for i in range(len(lines) - 1):
        if lines[i].encode() >= lines[i + 1].encode():
            die(f"{path} is not strictly byte-sorted at line {i + 1}")


class Side:
    def __init__(self) -> None:
        self.dictionary = None
        self.transducer = None
        self._prepared_entries: list[tuple[str, None]] | None = None

    def build_dictionary(self, terms: list[str], backend: str) -> None:
        if backend == "dynamic_dawg":
            if self._prepared_entries is None:
                self._prepared_entries = [(term, None) for term in terms]
            dawg = libdictenstein.DynamicDawg()
            inserted = dawg.update_many(self._prepared_entries)
            if inserted != len(terms):
                die(f"batch insert count mismatch: {inserted} != {len(terms)}")
            self.dictionary = dawg
        elif backend == "double_array_trie":
            self.dictionary = libdictenstein.DoubleArrayTrie(terms)
        else:
            die(f"unknown backend: {backend}")

    def free_dictionary(self) -> None:
        if self.transducer is not None:
            self.transducer.close()
            self.transducer = None
        if self.dictionary is not None:
            self.dictionary.close()
            self.dictionary = None

    def create_transducer(self, algorithm: str) -> None:
        if self.transducer is not None:
            self.transducer.close()
        self.transducer = liblevenshtein.Transducer(self.dictionary, ALGORITHMS[algorithm])

    def full_pass(
        self, queries: list[str], max_distance: int, with_checksum: bool
    ) -> tuple[int, int, int, int]:
        matches = 0
        term_bytes = 0
        distance_sum = 0
        checksum = 0
        assert self.transducer is not None, "create_transducer must run before full_pass"
        query = self.transducer.query
        for term_query in queries:
            with query(term_query, max_distance) as cursor:
                for match in cursor:
                    term = match.term
                    matches += 1
                    term_bytes += len(term)  # ASCII workload: len == UTF-8 bytes
                    distance_sum += match.distance
                    if with_checksum:
                        checksum = (checksum + entry_hash(term, match.distance)) & MASK64
        return matches, term_bytes, distance_sum, checksum


def render_result(
    args: argparse.Namespace,
    mode: str,
    algorithm: str,
    max_distance: int,
    queries_path: Path,
    query_count: int,
    term_count: int,
    backend: str,
    construct_ns: int | None,
    warmup_passes: int,
    samples_ns: list[int],
    triple: tuple[int, int, int],
    checksum: int,
    construct_times: list[int] | None,
    status: str,
    notes: list[str],
) -> dict:
    result: dict = {
        "schema_version": "1.0.0",
        "suite": "cross-language-v1",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": {
            "language": "python",
            "implementation": "vinary-tree",
            "backend": "ctypes",
            "runtime_version": f"CPython {sys.version.split()[0]}",
            "library_version": "0.10.0",
            "artifact": {"kind": "local-build", "id": "liblevenshtein@0.10.0"},
        },
        "dictionary": {
            "file": str(args.dictionary),
            "term_count": term_count,
            "structure": backend,
            "unit_domain": "unicode_scalar",
        },
        "workload": {
            "queryset": queries_path.stem,
            "file": str(queries_path),
            "query_count": query_count,
        },
        "algorithm": algorithm,
        "max_distance": max_distance,
        "mode": "memory" if mode == "memory-child" else mode,
        "protocol": {
            "timer": "monotonic",
            "harness": "self-timed",
            "warmup_seconds_min": args.warmup_seconds,
            "warmup_passes": warmup_passes,
            "samples_requested": args.reps if mode == "construct" else (args.samples if mode == "query" else 0),
            "sample_definition": SAMPLE_DEFINITION,
            "batch_size": 256,
            "wall_cap_seconds": int(WALL_CAP_SECONDS),
        },
        "status": status,
        "notes": notes,
    }
    if construct_ns is not None:
        result["dictionary"]["construct_ns"] = construct_ns
    if construct_times is not None:
        result["construct"] = {
            "reps": len(construct_times),
            "times_ns": construct_times,
            "term_count": term_count,
        }
    else:
        result["measurements"] = {
            "samples_ns": samples_ns,
            "sample_count": len(samples_ns),
            "matches_per_pass": triple[0],
            "term_bytes_per_pass": triple[1],
            "distance_sum_per_pass": triple[2],
            "checksum_hex": f"{checksum:016x}",
        }
    return result


def write_result(out_path: Path, result: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> int:
    self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--algorithm")
    parser.add_argument("--max-distance", type=int, default=-1, dest="max_distance")
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=3.0, dest="warmup_seconds")
    parser.add_argument("--gate-limit", type=int, default=200, dest="gate_limit")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--cells", type=Path)
    args = parser.parse_args()

    terms = read_lines(args.dictionary)
    assert_strictly_sorted(terms, args.dictionary)
    side = Side()
    base_notes = ["ctypes facade (decision E3); GC enabled; PYTHONHASHSEED=0"]

    if args.mode == "construct":
        if args.out is None:
            die("--out is required for construct mode")
        side.build_dictionary(terms, args.backend)
        side.free_dictionary()
        times: list[int] = []
        for _ in range(args.reps):
            start = time.perf_counter_ns()
            side.build_dictionary(terms, args.backend)
            times.append(time.perf_counter_ns() - start)
            side.free_dictionary()
        result = render_result(
            args, "construct", "standard", 1,
            args.queries or Path("workload/queries/hits.txt"), 1, len(terms), args.backend,
            None, 1, [], (0, 0, 0), 0, times, "ok",
            base_notes + ["construct mode: timed region is the build from the pre-sorted in-memory list only"],
        )
        write_result(args.out, result)
        return 0

    build_start = time.perf_counter_ns()
    side.build_dictionary(terms, args.backend)
    construct_ns = time.perf_counter_ns() - build_start

    def run_one(algorithm: str, max_distance: int, queries_path: Path, out_path: Path) -> None:
        side.create_transducer(algorithm)
        queries = read_lines(queries_path)
        if args.mode == "verify":
            limit = min(args.gate_limit, len(queries))
            subset = queries[:limit]
            m, b, d, checksum = side.full_pass(subset, max_distance, True)
            result = render_result(
                args, "verify", algorithm, max_distance, queries_path, limit, len(terms),
                args.backend, construct_ns, 0, [], (m, b, d), checksum, None, "ok", base_notes,
            )
        elif args.mode == "memory-child":
            m, b, d, checksum = side.full_pass(queries, max_distance, True)
            result = render_result(
                args, "memory-child", algorithm, max_distance, queries_path, len(queries),
                len(terms), args.backend, construct_ns, 0, [], (m, b, d), checksum, None, "ok",
                base_notes,
            )
        elif args.mode == "query":
            gate = side.full_pass(queries, max_distance, True)
            gate_triple = gate[:3]

            warm_start = time.perf_counter_ns()
            warmup_ns = int(args.warmup_seconds * 1e9)
            warmup_passes = 0
            last_pass_ns = 0
            while time.perf_counter_ns() - warm_start < warmup_ns or warmup_passes < 2:
                t0 = time.perf_counter_ns()
                triple = side.full_pass(queries, max_distance, False)[:3]
                last_pass_ns = time.perf_counter_ns() - t0
                if triple != gate_triple:
                    die("nondeterministic result during warmup")
                warmup_passes += 1

            sample_count = args.samples
            status = "ok"
            notes = list(base_notes)
            last_pass_seconds = last_pass_ns / 1e9
            if sample_count * last_pass_seconds > WALL_CAP_SECONDS:
                reduced = max(10, int(WALL_CAP_SECONDS / last_pass_seconds))
                notes.append(
                    f"samples reduced from {sample_count} to {reduced} by the "
                    f"{WALL_CAP_SECONDS:.0f}s wall cap (estimated pass {last_pass_seconds:.3f}s)"
                )
                sample_count = reduced
                status = "degraded"

            samples_ns: list[int] = []
            for _ in range(sample_count):
                t0 = time.perf_counter_ns()
                triple = side.full_pass(queries, max_distance, False)[:3]
                samples_ns.append(time.perf_counter_ns() - t0)
                if triple != gate_triple:
                    die("nondeterministic result during measurement")

            result = render_result(
                args, "query", algorithm, max_distance, queries_path, len(queries), len(terms),
                args.backend, construct_ns, warmup_passes, samples_ns, gate_triple, gate[3],
                None, status, notes,
            )
        else:
            die(f"unknown mode: {args.mode}")
            return
        write_result(out_path, result)

    if args.cells is not None:
        for line in args.cells.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                die(f"cells row needs 4 fields: {line}")
            run_one(fields[0], int(fields[1]), Path(fields[2]), Path(fields[3]))
    else:
        if not args.algorithm or args.max_distance < 0 or args.queries is None or args.out is None:
            die("--algorithm, --max-distance, --queries, --out are required")
        run_one(args.algorithm, args.max_distance, args.queries, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
