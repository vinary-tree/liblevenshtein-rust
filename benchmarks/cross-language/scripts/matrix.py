#!/usr/bin/env python3
"""The benchmark matrix: single source of truth for which cells exist.

Both run-all.sh (to schedule work) and aggregate.py (to account for every
cell — measured, degraded, failed, or skipped) derive the expected cell list
from this module, so a cell can never silently vanish between scheduling and
reporting.

Rules (user decision 2026-08-13: FULL matrix for every language):

  query cells per target×backend:
      algorithms: standard, transposition, merge_and_split
                  (+ damerau_levenshtein where targets.tsv damerau=yes)
    × distances:  1, 2, 3
    × query sets: hits, X-d1, X-d2, X-d3, oov
                  where X = std for standard/merge_and_split,
                        X = tr  for transposition/damerau_levenshtein
      → 45 shared cells (+15 damerau bonus)

  construct: one cell per target×backend (reps: 10, DAT: 3)
  memory:    one cell per target×backend (standard, d=2, std-d2)

  --quick (off by default): the head-to-head pairs and the anchors keep the
  full matrix; every other target's query cells reduce to
  {standard, transposition} × d ∈ {1,2} × {hits, algorithm-matched mut-dK
  diagonal} = 8 cells. Construct and memory cells are unaffected.

CLI:
  matrix.py cells   --target T --backend B [--mode query|construct|memory]
                    [--missing-from DIR] [--tsv-out FILE --cell-dir DIR]
  matrix.py expected --targets a,b,c        # full accounting list, TSV to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
XL = SCRIPTS_DIR.parent

SHARED_ALGORITHMS = ("standard", "transposition", "merge_and_split")
BONUS_ALGORITHM = "damerau_levenshtein"
DISTANCES = (1, 2, 3)

# Targets that keep the full matrix even under --quick: the two migration
# pairs and the raw-performance anchors.
FULL_MATRIX_TARGETS = frozenset(
    {"rust", "c", "jvm-vinary", "jvm-legacy", "js-native", "js-wasm", "js-wasi", "js-legacy"}
)


def load_manifest() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    header: list[str] = []
    for line in (XL / "targets.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split("\t")
        if fields[0] == "target":
            header = fields
            continue
        rows[fields[0]] = dict(zip(header, fields))
    return rows


def mut_prefix(algorithm: str) -> str:
    return "tr" if algorithm in ("transposition", BONUS_ALGORITHM) else "std"


def query_sets(algorithm: str) -> tuple[str, ...]:
    prefix = mut_prefix(algorithm)
    return ("hits", f"{prefix}-d1", f"{prefix}-d2", f"{prefix}-d3", "oov")


def query_cells(target: str, backend: str, damerau: bool, quick: bool) -> list[tuple]:
    if quick and target not in FULL_MATRIX_TARGETS:
        return [
            (target, backend, "query", algorithm, distance, queryset)
            for algorithm in ("standard", "transposition")
            for distance in (1, 2)
            for queryset in ("hits", f"{mut_prefix(algorithm)}-d{distance}")
        ]
    algorithms = SHARED_ALGORITHMS + ((BONUS_ALGORITHM,) if damerau else ())
    return [
        (target, backend, "query", algorithm, distance, queryset)
        for algorithm in algorithms
        for distance in DISTANCES
        for queryset in query_sets(algorithm)
    ]


def all_cells_for(
    target: str, manifest: dict[str, dict[str, str]], quick: bool = False
) -> list[tuple]:
    row = manifest[target]
    damerau = row["damerau"] == "yes"
    cells: list[tuple] = []
    for backend in row["backends"].split(","):
        cells.extend(query_cells(target, backend, damerau, quick))
        cells.append((target, backend, "construct", "standard", 1, "hits"))
        cells.append((target, backend, "memory", "standard", 2, "std-d2"))
    return cells


def cell_filename(cell: tuple) -> str:
    target, backend, mode, algorithm, distance, queryset = cell
    return f"{target}__{backend}__{mode}__{algorithm}__d{distance}__{queryset}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    cells_cmd = sub.add_parser("cells")
    cells_cmd.add_argument("--target", required=True)
    cells_cmd.add_argument("--backend", required=True)
    cells_cmd.add_argument("--quick", action="store_true")
    cells_cmd.add_argument("--mode", choices=("query", "construct", "memory"))
    cells_cmd.add_argument(
        "--missing-from", type=Path, help="emit only cells whose JSON is absent from this dir"
    )
    cells_cmd.add_argument(
        "--tsv-out",
        type=Path,
        help="write a harness --cells TSV (query mode only) instead of listing",
    )
    cells_cmd.add_argument(
        "--cell-dir", type=Path, help="output dir the TSV's out_path column points into"
    )

    expected_cmd = sub.add_parser("expected")
    expected_cmd.add_argument("--targets", required=True)
    expected_cmd.add_argument("--quick", action="store_true")

    args = parser.parse_args()
    manifest = load_manifest()

    if args.command == "expected":
        for target in args.targets.split(","):
            target = target.strip()
            if target not in manifest:
                raise SystemExit(f"matrix: unknown target {target!r}")
            for cell in all_cells_for(target, manifest, args.quick):
                print("\t".join(str(part) for part in cell))
        return 0

    if args.target not in manifest:
        raise SystemExit(f"matrix: unknown target {args.target!r}")
    row = manifest[args.target]
    if args.backend not in row["backends"].split(","):
        raise SystemExit(f"matrix: {args.target} has no backend {args.backend!r}")

    cells = [
        cell
        for cell in all_cells_for(args.target, manifest, args.quick)
        if cell[1] == args.backend and (args.mode is None or cell[2] == args.mode)
    ]
    if args.missing_from:
        cells = [
            cell for cell in cells if not (args.missing_from / cell_filename(cell)).exists()
        ]

    if args.tsv_out:
        if args.mode != "query":
            raise SystemExit("matrix: --tsv-out requires --mode query")
        if not args.cell_dir:
            raise SystemExit("matrix: --tsv-out requires --cell-dir")
        args.cell_dir.mkdir(parents=True, exist_ok=True)
        args.tsv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.tsv_out.open("w") as fh:
            for cell in cells:
                _, _, _, algorithm, distance, queryset = cell
                queries = XL / "workload" / "queries" / f"{queryset}.txt"
                out = args.cell_dir / cell_filename(cell)
                fh.write(f"{algorithm}\t{distance}\t{queries}\t{out}\n")
        print(len(cells))
        return 0

    for cell in cells:
        print("\t".join(str(part) for part in cell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
