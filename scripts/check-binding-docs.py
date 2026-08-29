#!/usr/bin/env python3
"""Fail closed when a declared binding lacks complete, current documentation."""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "bindings/api.json").read_text(encoding="utf-8"))
SURFACE_MODEL = json.loads(
    (ROOT / "bindings/api-surface-map.json").read_text(encoding="utf-8")
)
DOCS = MODEL["documentation"]
LINK_RE = re.compile(r"!?(?:\[[^\]]+\])\(([^)]+)\)")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|STUB)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
EXPLICIT_ANCHOR_RE = re.compile(
    r"<a\s+[^>]*(?:id|name)\s*=\s*(['\"])([^'\"]+)\1[^>]*>", re.IGNORECASE
)
ANCHOR_CACHE: dict[Path, set[str]] = {}
FACADE_PACKAGE_KEYS = {
    "python": "pypi",
    "jvm": "maven",
    "clojure": "clojars",
    "javascript": "npm",
    "dotnet": "nuget",
    "go": "goModule",
    "swift": "swift",
    "ruby": "rubygems",
    "fortran": "fpm",
    "ocaml": "opam",
    "haskell": "hackage",
    "lua": "luarocks",
}
RETIRED_PACKAGE_PATTERNS = {
    "@vinary-tree/interop": re.compile(r"@vinary-tree/interop(?![A-Za-z0-9_-])"),
    "@vinary-tree/vinary-tree": re.compile(
        r"@vinary-tree/vinary-tree(?!-interop|[A-Za-z0-9_])"
    ),
    "vinary-tree-liblevenshtein": re.compile(
        r"(?<![A-Za-z0-9_-])vinary-tree-liblevenshtein(?![A-Za-z0-9_-])"
    ),
}


def fail(message: str) -> None:
    print(f"binding-docs: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(relative: str) -> tuple[Path, str]:
    path = ROOT / relative
    if not path.is_file():
        fail(f"missing documented file: {relative}")
    return path, path.read_text(encoding="utf-8")


def canonical_package(facade: str) -> str:
    if facade in {"c", "cpp"}:
        return "liblevenshtein"
    package_key = FACADE_PACKAGE_KEYS.get(facade)
    if package_key is None:
        fail(f"no canonical package mapping for documented facade {facade}")
    return MODEL["packages"][package_key]


def github_heading_slug(heading: str) -> str:
    """Approximate GitHub's documented heading-ID normalization."""
    heading = re.sub(r"!?(?:\[([^]]*)\])\([^)]*\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = html.unescape(heading)
    heading = re.sub(r"\\(.)", r"\1", heading)
    heading = heading.replace("`", "")
    slug: list[str] = []
    for character in heading.strip().lower():
        category = unicodedata.category(character)
        if character.isspace():
            slug.append("-")
        elif character in {"-", "_"} or category[0] in {"L", "M", "N"}:
            slug.append(character)
    return "".join(slug)


def markdown_anchors(path: Path) -> set[str]:
    cached = ANCHOR_CACHE.get(path)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8")
    anchors = {match[1] for match in EXPLICIT_ANCHOR_RE.findall(text)}
    occurrences: dict[str, int] = {}
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is not None:
            continue
        match = HEADING_RE.match(line)
        if match is None:
            continue
        base = github_heading_slug(match.group(1))
        if not base:
            continue
        duplicate = occurrences.get(base, 0)
        occurrences[base] = duplicate + 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    ANCHOR_CACHE[path] = anchors
    return anchors


def check_links(path: Path, text: str) -> None:
    for raw in LINK_RE.findall(text):
        target = raw.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, separator, fragment = target.partition("#")
        resolved = (path.parent / target_path).resolve() if target_path else path
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            fail(f"{path.relative_to(ROOT)} links outside the repository: {raw}")
        if not resolved.exists():
            fail(f"{path.relative_to(ROOT)} has broken local link: {raw}")
        if separator and fragment and resolved.suffix.casefold() == ".md":
            anchor = unquote(fragment)
            if anchor not in markdown_anchors(resolved):
                fail(
                    f"{path.relative_to(ROOT)} has broken local anchor "
                    f"{raw}: {anchor!r} is absent from {resolved.relative_to(ROOT)}"
                )


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
            "-m",
            "unittest",
            "discover",
            "-s",
            str(ROOT / "scripts" / "tests"),
            "-p",
            "test_*.py",
            "-q",
        ],
        cwd=ROOT,
        check=True,
    )
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
        [
            sys.executable,
            str(ROOT / "scripts" / "generate-binding-traceability.py"),
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
        surface = SURFACE_MODEL["languages"][facade]
        if surface["readme"] != entry["guide"]:
            fail(f"{facade} guide differs between api.json and api-surface-map.json")
        if entry["example"] not in surface["tests"]:
            fail(
                f"{facade} canonical example is not executable test evidence: "
                f"{entry['example']}"
            )
        _, guide_text = read(entry["guide"])
        package = canonical_package(facade)
        if package not in guide_text:
            fail(f"{facade} guide omits canonical package coordinate {package!r}")
        check_guide(facade, entry["guide"], entry["languages"], entry["example"])

    hub_path, hub = read(DOCS["hub"])
    architecture_path, architecture = read(DOCS["architecture"])
    collection_path, collection = read(DOCS["collectionProtocols"])
    optimization_path, optimization = read(DOCS["optimizationMethodology"])
    check_links(hub_path, hub)
    check_links(architecture_path, architecture)
    check_links(collection_path, collection)
    check_links(optimization_path, optimization)
    operational_documents = {
        DOCS["hub"]: hub,
        DOCS["architecture"]: architecture,
        DOCS["collectionProtocols"]: collection,
        DOCS["optimizationMethodology"]: optimization,
    }
    operational_documents.update(
        {entry["guide"]: read(entry["guide"])[1] for entry in DOCS["facades"].values()}
    )
    for relative, document in operational_documents.items():
        for retired, pattern in RETIRED_PACKAGE_PATTERNS.items():
            if pattern.search(document):
                fail(f"{relative} uses retired public package name {retired!r}")
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
