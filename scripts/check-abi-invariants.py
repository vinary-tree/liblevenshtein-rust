#!/usr/bin/env python3
"""Validate the ABI invariant registry (docs/verification/ABI_INVARIANTS.tsv).

The registry is the traceability spine of the ABI verification program: every
invariant row names its formal home (spec_path/spec_name), its strength on the
proof ladder, its mirroring executable test (test_path/test_name), and the
gates that run both. This checker enforces that the registry never drifts from
the artifacts it points at:

  1. column shape and unique, well-formed invariant ids;
  2. strength values drawn from the documented ladder;
  3. every spec_path and test_path exists;
  4. every spec_name is literally present in its spec file (TLA invariant /
     property name, SMT echo label, or test function for test-pinned rows);
  5. every test_name is a real test function in its test file;
  6. test-pinned rows justify themselves in the law column;
  7. every VT-* id referenced by an INVARIANT-HOOK comment anywhere under
     tests/ or vinary-tree-interop/tests/ resolves to a registry row, and
     every registry id is hooked by at least one test file.

Stdlib-only; exit 1 on any failure. Wired into `scripts/verify-formal.sh
audit` and run directly in CI.
"""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTEROP_ROOT = Path(
    os.environ.get("VINARY_TREE_INTEROP_ROOT", ROOT.parent / "vinary-tree-interop")
).resolve()
REGISTRY = ROOT / "docs" / "verification" / "ABI_INVARIANTS.tsv"

COLUMNS = [
    "id",
    "layer",
    "law",
    "spec_kind",
    "spec_path",
    "spec_name",
    "strength",
    "test_path",
    "test_name",
    "gate",
]
STRENGTHS = {
    "rocq-unbounded",
    "tlaps-unbounded",
    "apalache-inductive",
    "verus",
    "smt-dual",
    "tlc-bounded",
    "test-pinned",
}
SPEC_KINDS = {"tla", "coq", "smt", "verus", "test"}
ID_PATTERN = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)*-\d+$")
HOOK_PATTERN = re.compile(r"INVARIANT-HOOK:\s*([A-Z]+(?:-[A-Z0-9]+)*-\d+(?:\.\.\d+)?)")


class Failures:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        self.messages.append(message)


def resolve_path(relative: str) -> Path:
    prefix = "vinary-tree-interop/"
    if relative.startswith(prefix):
        return INTEROP_ROOT / relative.removeprefix(prefix)
    return ROOT / relative


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return f"vinary-tree-interop/{path.relative_to(INTEROP_ROOT)}"
        except ValueError:
            return str(path)


def parse_rows(failures: Failures) -> list[dict[str, str]]:
    if not REGISTRY.is_file():
        failures.add(f"registry missing: {REGISTRY.relative_to(ROOT)}")
        return []
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(
        REGISTRY.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) != len(COLUMNS):
            failures.add(
                f"line {line_number}: {len(cells)} columns, expected {len(COLUMNS)}"
            )
            continue
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def expand_hook(hook: str) -> list[str]:
    """`VT-LIFE-1..6` expands to VT-LIFE-1 ... VT-LIFE-6; plain ids pass through."""
    if ".." not in hook:
        return [hook]
    prefix, _, tail = hook.rpartition("-")
    low, _, high = tail.partition("..")
    return [f"{prefix}-{index}" for index in range(int(low), int(high) + 1)]


def main() -> int:
    failures = Failures()
    rows = parse_rows(failures)

    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if not ID_PATTERN.match(row_id):
            failures.add(f"{row_id}: malformed invariant id")
        if row_id in seen_ids:
            failures.add(f"{row_id}: duplicate id")
        seen_ids.add(row_id)

        if row["strength"] not in STRENGTHS:
            failures.add(f"{row_id}: unknown strength {row['strength']!r}")
        if row["spec_kind"] not in SPEC_KINDS:
            failures.add(f"{row_id}: unknown spec_kind {row['spec_kind']!r}")
        if row["strength"] == "test-pinned":
            if row["spec_kind"] != "test":
                failures.add(f"{row_id}: test-pinned rows must have spec_kind=test")
            if "exhaustively pinned" not in row["law"]:
                failures.add(
                    f"{row_id}: test-pinned rows must justify themselves in the "
                    "law column ('exhaustively pinned ...')"
                )
        if not row["gate"].strip():
            failures.add(f"{row_id}: empty gate column")

        spec_path = resolve_path(row["spec_path"])
        if not spec_path.is_file():
            failures.add(f"{row_id}: spec_path missing: {row['spec_path']}")
        else:
            spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
            # SMT rows point at echo-labelled blocks or constant families; the
            # bracketed tag is the stable token. Everything else must contain
            # the name verbatim.
            needle = row["spec_name"]
            if row["spec_kind"] == "smt" and "[" in needle:
                needle = needle[needle.index("[") : needle.index("]") + 1]
            if needle.split(" ")[0] not in spec_text:
                failures.add(
                    f"{row_id}: spec_name {row['spec_name']!r} not found in "
                    f"{row['spec_path']}"
                )

        test_path = resolve_path(row["test_path"])
        if not test_path.is_file():
            failures.add(f"{row_id}: test_path missing: {row['test_path']}")
        else:
            test_text = test_path.read_text(encoding="utf-8", errors="replace")
            if not re.search(rf"fn {re.escape(row['test_name'])}\b", test_text):
                failures.add(
                    f"{row_id}: test fn {row['test_name']!r} not found in "
                    f"{row['test_path']}"
                )

    # Hooks <-> registry closure.
    hooked: set[str] = set()
    hook_sources: dict[str, list[str]] = {}
    for base in (ROOT / "tests", INTEROP_ROOT / "tests"):
        for path in sorted(base.rglob("*.rs")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in HOOK_PATTERN.finditer(text):
                for invariant_id in expand_hook(match.group(1)):
                    hooked.add(invariant_id)
                    hook_sources.setdefault(invariant_id, []).append(
                        display_path(path)
                    )
    for invariant_id in sorted(hooked - seen_ids):
        failures.add(
            f"hook references unregistered invariant {invariant_id} "
            f"(in {', '.join(hook_sources[invariant_id])})"
        )
    for invariant_id in sorted(seen_ids - hooked):
        failures.add(f"registered invariant {invariant_id} has no INVARIANT-HOOK")

    if failures.messages:
        print(f"check-abi-invariants: {len(failures.messages)} failure(s)")
        for message in failures.messages:
            print(f"  FAIL {message}")
        return 1
    print(
        f"check-abi-invariants: {len(rows)} invariants verified "
        f"(specs, tests, hooks, and ladder all consistent)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
