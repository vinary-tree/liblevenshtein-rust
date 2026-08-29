#!/usr/bin/env python3
"""Generate public binding API → source → docs → test/example traceability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
API_PATH = ROOT / "bindings" / "api.json"
SURFACE_PATH = ROOT / "bindings" / "api-surface-map.json"
OUTPUT_PATH = ROOT / "bindings" / "conformance" / "public-api-traceability.tsv"


def fail(message: str) -> None:
    raise SystemExit(f"binding traceability error: {message}")


def clean(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\t\r\n")
    ):
        fail(f"{field} must be a non-empty single-line string")
    return value


def resolve(relative: str, field: str) -> Path:
    path = (ROOT / clean(relative, field)).resolve()
    try:
        path.relative_to(WORKSPACE)
    except ValueError:
        fail(f"{field} leaves the indexed workspace: {relative}")
    if not path.is_file():
        fail(f"{field} is missing: {path}")
    return path


def read_many(relatives: list[str], field: str) -> str:
    return "\n".join(
        resolve(relative, f"{field}[{index}]").read_text(
            encoding="utf-8", errors="ignore"
        )
        for index, relative in enumerate(relatives)
    )


def symbol_leaf(symbol: str) -> str:
    parts = [part for part in re.split(r"[\s.:#%$/()]+", symbol) if part]
    if not parts:
        fail(f"unusable public symbol {symbol!r}")
    return parts[-1]


def named_in(name: str, corpus: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    return re.search(pattern, corpus) is not None


def symbols(entry: object, field: str) -> tuple[list[str], str]:
    if not isinstance(entry, dict):
        fail(f"{field} must be an object")
    symbol = entry.get("symbol")
    if symbol is None:
        return [], clean(entry.get("_reason"), f"{field}._reason")
    result = symbol if isinstance(symbol, list) else [symbol]
    if not result or not all(isinstance(item, str) and item.strip() for item in result):
        fail(f"{field}.symbol must contain non-empty strings")
    return result, "-"


def evidence_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def direct_status(symbol: str, native_item: str, corpus: str) -> str:
    return (
        "complete"
        if named_in(symbol_leaf(symbol), corpus) or named_in(native_item, corpus)
        else "audit-required"
    )


def documentation_status(symbol: str, guide: str) -> str:
    escaped = symbol.replace("|", r"\|")
    return "complete" if f"`{escaped}`" in guide else "audit-required"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="reject a stale committed matrix"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="reject every exposed API item without direct source, docs, test, and example evidence",
    )
    args = parser.parse_args()

    api = json.loads(API_PATH.read_text(encoding="utf-8"))
    surface = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))
    if surface.get("modelVersion") != 1:
        fail("unsupported api-surface-map modelVersion")
    docs_facades = api["documentation"]["facades"]
    expected_functions = [entry["name"] for entry in api["cFunctions"]]
    expected_enums = list(api["enums"])

    rows: list[str] = []
    incomplete: list[str] = []
    exposed_count = 0
    absence_count = 0
    for facade_id, facade in surface["languages"].items():
        context = f"languages.{facade_id}"
        source_files = facade.get("sourceFiles")
        tests = facade.get("tests")
        if not isinstance(source_files, list) or not source_files:
            fail(f"{context}.sourceFiles must be a non-empty array")
        if not isinstance(tests, list) or not tests:
            fail(f"{context}.tests must be a non-empty array")
        source_corpus = read_many(source_files, f"{context}.sourceFiles")
        test_corpus = read_many(tests, f"{context}.tests")
        guide_relative = clean(facade.get("readme"), f"{context}.readme")
        guide_text = resolve(guide_relative, f"{context}.readme").read_text(
            encoding="utf-8"
        )
        if facade_id in docs_facades:
            example_relative = clean(
                docs_facades[facade_id].get("example"),
                f"documentation.facades.{facade_id}.example",
            )
        else:
            example_relative = tests[0]
        example_text = resolve(example_relative, f"{context}.example").read_text(
            encoding="utf-8", errors="ignore"
        )

        items: list[tuple[str, str, object]] = []
        functions = facade.get("functions")
        if not isinstance(functions, dict) or list(functions) != expected_functions:
            fail(f"{context}.functions differs from bindings/api.json order or set")
        items.extend(("function", name, functions[name]) for name in expected_functions)
        enums = facade.get("enums")
        if not isinstance(enums, dict) or list(enums) != expected_enums:
            fail(f"{context}.enums differs from bindings/api.json order or set")
        items.extend(("enum", f"enum:{name}", enums[name]) for name in expected_enums)
        items.extend(
            (
                ("protocol", "protocol:iterator", facade.get("iterator")),
                ("protocol", "protocol:reducer", facade.get("reducer")),
            )
        )

        for kind, native_item, entry in items:
            public_symbols, reason = symbols(entry, f"{context}.{native_item}")
            if not public_symbols:
                absence_count += 1
                rows.append(
                    "\t".join(
                        (
                            facade_id,
                            kind,
                            native_item,
                            "-",
                            "reasoned-absence",
                            evidence_json(source_files),
                            "inapplicable",
                            guide_relative,
                            "inapplicable",
                            evidence_json(tests),
                            "inapplicable",
                            example_relative,
                            "inapplicable",
                            reason,
                        )
                    )
                )
                continue
            for public_symbol in public_symbols:
                exposed_count += 1
                source_status = direct_status(public_symbol, native_item, source_corpus)
                docs_status = documentation_status(public_symbol, guide_text)
                test_status = direct_status(public_symbol, native_item, test_corpus)
                example_status = direct_status(public_symbol, native_item, example_text)
                statuses = (source_status, docs_status, test_status, example_status)
                if any(status != "complete" for status in statuses):
                    incomplete.append(
                        f"{facade_id}|{kind}|{native_item}|{public_symbol}"
                    )
                rows.append(
                    "\t".join(
                        (
                            facade_id,
                            kind,
                            native_item,
                            public_symbol,
                            "exposed",
                            evidence_json(source_files),
                            source_status,
                            guide_relative,
                            docs_status,
                            evidence_json(tests),
                            test_status,
                            example_relative,
                            example_status,
                            "-",
                        )
                    )
                )

    if args.require_complete and incomplete:
        fail(
            f"{len(incomplete)} exposed API traceability gaps remain; first: "
            + ", ".join(incomplete[:10])
        )
    header = (
        "facade\tkind\tnative_item\tpublic_symbol\tsurface_status\t"
        "source_evidence\tsource_status\tguide\tdocumentation_status\t"
        "test_evidence\ttest_status\tcanonical_example\texample_status\treason"
    )
    rendered = "\n".join((header, *rows, ""))
    if args.check:
        if (
            not OUTPUT_PATH.is_file()
            or OUTPUT_PATH.read_text(encoding="utf-8") != rendered
        ):
            fail(f"stale matrix: rerun {Path(__file__).relative_to(ROOT)}")
    else:
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(
        f"binding traceability: {exposed_count} public API mappings, "
        f"{absence_count} reasoned absences, {len(incomplete)} direct-evidence gaps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
