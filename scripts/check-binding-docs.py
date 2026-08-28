#!/usr/bin/env python3
"""Fail closed when a declared binding lacks complete, current documentation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "bindings/api.json").read_text(encoding="utf-8"))
SURFACE_MODEL = json.loads(
    (ROOT / "bindings/api-surface-map.json").read_text(encoding="utf-8")
)
DOCS = MODEL["documentation"]
LINK_RE = re.compile(r"!?(?:\[[^\]]+\])\(([^)]+)\)")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|STUB)\b", re.IGNORECASE)


def fail(message: str) -> None:
    print(f"binding-docs: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(relative: str) -> tuple[Path, str]:
    path = ROOT / relative
    if not path.is_file():
        fail(f"missing documented file: {relative}")
    return path, path.read_text(encoding="utf-8")


def check_links(path: Path, text: str) -> None:
    for raw in LINK_RE.findall(text):
        target = raw.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            fail(f"{path.relative_to(ROOT)} links outside the repository: {raw}")
        if not resolved.exists():
            fail(f"{path.relative_to(ROOT)} has broken local link: {raw}")


def public_symbols(facade: str) -> set[str]:
    entry = SURFACE_MODEL["languages"][facade]
    symbols: set[str] = set()
    for mapping in entry["functions"].values():
        symbol = mapping.get("symbol")
        if isinstance(symbol, str):
            symbols.add(symbol)
        elif isinstance(symbol, list):
            symbols.update(symbol)
    symbols.update(
        mapping["symbol"]
        for mapping in entry["enums"].values()
        if isinstance(mapping.get("symbol"), str)
    )
    for protocol in ("iterator", "reducer"):
        symbol = entry[protocol].get("symbol")
        if isinstance(symbol, str):
            symbols.add(symbol)
    return symbols


def check_guide(
    facade: str, relative: str, languages: list[str], example: str | None
) -> None:
    path, text = read(relative)
    if "BEGIN GENERATED BINDING OPERATIONS" not in text:
        fail(f"{relative} is not governed by the binding guide generator")
    for heading in DOCS["requiredTopics"]:
        if f"## {heading}" not in text:
            fail(f"{relative} is missing required section {heading!r}")
    for language in languages:
        if language.casefold() not in text.casefold():
            fail(f"{relative} does not identify represented language {language!r}")
    if "### Facade symbol index" not in text:
        fail(f"{relative} has no generated facade symbol index")
    if "### Intended usage paths" not in text:
        fail(f"{relative} has no intended-usage decision table")
    symbol_index = text.split("### Facade symbol index", 1)[1].split(
        "### Intended usage paths", 1
    )[0]
    missing_symbols = sorted(
        symbol
        for symbol in public_symbols(facade)
        if f"`{symbol.replace('|', r'\|')}`" not in symbol_index
    )
    if missing_symbols:
        fail(f"{relative} omits modeled public symbols: " + ", ".join(missing_symbols))
    if example is not None:
        read(example)
        if example not in text:
            fail(f"{relative} does not link canonical example {example}")
    fences: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if not line.startswith("```"):
            continue
        if in_fence:
            in_fence = False
            continue
        tag = line[3:].strip().split(maxsplit=1)[0] if line[3:].strip() else ""
        fences.append(tag)
        in_fence = True
    if in_fence:
        fail(f"{relative} has an unclosed fenced code block")
    if not fences or any(not language for language in fences):
        fail(f"{relative} must tag every fenced code block with a language")
    if not any(language in {"sh", "bash", "shell", "console"} for language in fences):
        fail(f"{relative} has no executable verification command")
    if PLACEHOLDER_RE.search(text):
        fail(f"{relative} contains a documentation placeholder")
    check_links(path, text)


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate-family-completeness-matrix.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / DOCS["generator"]), "--check"],
        cwd=ROOT,
        check=True,
    )

    facade_keys = set(DOCS["facades"])
    support_keys = {
        language for tier in MODEL["supportTiers"].values() for language in tier
    }
    grouped = {"typescript": "javascript", "clojurescript": "javascript"}
    normalized_support = {grouped.get(language, language) for language in support_keys}
    if facade_keys != normalized_support:
        fail(
            "documentation facade set differs from support tiers: "
            f"missing={sorted(normalized_support - facade_keys)}, "
            f"extra={sorted(facade_keys - normalized_support)}"
        )

    for facade, entry in DOCS["facades"].items():
        check_guide(facade, entry["guide"], entry["languages"], entry["example"])

    hub_path, hub = read(DOCS["hub"])
    architecture_path, architecture = read(DOCS["architecture"])
    collection_path, collection = read(DOCS["collectionProtocols"])
    optimization_path, optimization = read(DOCS["optimizationMethodology"])
    check_links(hub_path, hub)
    check_links(architecture_path, architecture)
    check_links(collection_path, collection)
    check_links(optimization_path, optimization)
    if PLACEHOLDER_RE.search(collection):
        fail(
            f"{collection_path.relative_to(ROOT)} contains a documentation placeholder"
        )
    if DOCS["collectionProtocols"] not in hub and collection_path.name not in hub:
        fail("binding hub does not route readers to the collection-protocol plan")
    for entry in DOCS["facades"].values():
        if Path(entry["guide"]).name not in hub and entry["guide"] not in hub:
            # The hub may link a binding directory rather than repeat every README.
            binding_root = str(Path(entry["guide"]).parent)
            if binding_root not in hub:
                fail(f"family hub does not route readers to {entry['guide']}")

    print(
        "binding-docs: ok "
        f"({len(DOCS['facades'])} project facades; "
        "standalone interop documentation linked externally)"
    )


if __name__ == "__main__":
    main()
