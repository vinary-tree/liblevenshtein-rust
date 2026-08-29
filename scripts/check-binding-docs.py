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
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
DOI_LINK_RE = re.compile(
    r"\[(?P<label>[^\]]+)\]\("
    r"(?:<https://doi\.org/(?P<angle>[^>\s]+)>|"
    r"https://doi\.org/(?P<plain>[^)\s]+))\)",
    re.IGNORECASE,
)
DOI_VALUE_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
DOI_TOKEN_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
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
    "bindings/javascript-runtime": re.compile(
        r"(?<![A-Za-z0-9_/-])bindings/javascript-runtime"
        r"(?:/|(?=[^A-Za-z0-9_-]|$))"
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


def normalized_citation_text(value: str) -> str:
    """Normalize bibliographic prose without erasing meaningful words."""
    return " ".join(
        re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", value).casefold()).split()
    )


def citation_context(text: str, position: int) -> str:
    """Return the numbered-reference item containing a DOI link."""
    starts = list(re.finditer(r"(?m)^\s*\d{1,3}\.\s+", text[:position]))
    start = starts[-1].start() if starts else max(0, position - 1200)
    return text[start:position]


def check_citation_identity(
    document: str,
    prose: str,
    link: re.Match[str],
    citation: dict[str, object],
) -> None:
    """Check that a DOI link's label and surrounding bibliography match metadata."""
    doi = link.group("angle") or link.group("plain")
    key = doi.casefold()
    if DOI_VALUE_RE.fullmatch(doi) is None:
        fail(f"{document} contains invalid DOI syntax {doi!r}")
    label_dois = DOI_TOKEN_RE.findall(link.group("label"))
    if len(label_dois) != 1 or label_dois[0].casefold() != key:
        fail(f"{document} DOI link label disagrees with URL {doi}")
    context = normalized_citation_text(citation_context(prose, link.start()))
    for field in ("title", "firstAuthor"):
        expected = normalized_citation_text(str(citation[field]))
        if expected not in context:
            fail(
                f"{document} DOI {doi} does not match ledger {field} "
                f"{citation[field]!r}"
            )
    if str(citation["year"]) not in context:
        fail(f"{document} DOI {doi} does not cite ledger year {citation['year']}")


def check_citation_ledger() -> None:
    relative = DOCS.get("citationLedger")
    if not isinstance(relative, str):
        fail("api.json does not declare documentation.citationLedger")
    _, raw_ledger = read(relative)
    ledger = json.loads(raw_ledger)
    if ledger.get("schemaVersion") != 1:
        fail(f"{relative} has an unsupported schemaVersion")
    if ledger.get("registry") != "Crossref":
        fail(f"{relative} must name Crossref as its metadata registry")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ledger.get("verifiedAt", "")):
        fail(f"{relative} has no ISO-8601 verifiedAt date")

    documents = ledger.get("documents")
    citations = ledger.get("citations")
    if not isinstance(documents, list) or not documents:
        fail(f"{relative} must declare at least one governed document")
    if len(documents) != len(set(documents)) or documents != sorted(documents):
        fail(f"{relative} documents must be unique and sorted")
    if not isinstance(citations, list) or not citations:
        fail(f"{relative} must declare at least one citation")

    by_doi: dict[str, dict[str, object]] = {}
    declared_order: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            fail(f"{relative} contains a non-object citation")
        doi = citation.get("doi")
        if not isinstance(doi, str) or DOI_VALUE_RE.fullmatch(doi) is None:
            fail(f"{relative} contains an invalid DOI: {doi!r}")
        key = doi.casefold()
        if key in by_doi:
            fail(f"{relative} contains duplicate DOI {doi}")
        for field in ("title", "firstAuthor", "publisher", "type", "registryUrl"):
            if not isinstance(citation.get(field), str) or not citation[field]:
                fail(f"{relative} DOI {doi} has no {field}")
        if not isinstance(citation.get("year"), int):
            fail(f"{relative} DOI {doi} has no numeric year")
        expected_registry_url = f"https://api.crossref.org/works/{doi}"
        if citation["registryUrl"].casefold() != expected_registry_url.casefold():
            fail(f"{relative} DOI {doi} has a non-canonical registryUrl")
        sources = citation.get("sources")
        if not isinstance(sources, list) or not sources:
            fail(f"{relative} DOI {doi} has no source documents")
        if len(sources) != len(set(sources)) or sources != sorted(sources):
            fail(f"{relative} DOI {doi} sources must be unique and sorted")
        by_doi[key] = citation
        declared_order.append(key)
    if declared_order != sorted(declared_order):
        fail(f"{relative} citations must be sorted by DOI")

    observed: dict[str, set[str]] = {doi: set() for doi in by_doi}
    for document in documents:
        if not isinstance(document, str):
            fail(f"{relative} contains a non-string governed document")
        _, text = read(document)
        prose = HTML_COMMENT_RE.sub("", text)
        links = list(DOI_LINK_RE.finditer(prose))
        if prose.casefold().count("https://doi.org/") != len(links):
            fail(
                f"{document} has a DOI URL that is not a labeled canonical Markdown link"
            )
        for link in links:
            doi = link.group("angle") or link.group("plain")
            key = doi.casefold()
            citation = by_doi.get(key)
            if citation is None:
                fail(f"{document} cites DOI {doi} absent from {relative}")
            check_citation_identity(document, prose, link, citation)
            observed[key].add(document)

    if set(documents) != {
        source
        for citation in citations
        for source in citation["sources"]
        if isinstance(source, str)
    }:
        fail(f"{relative} governed documents differ from citation sources")
    for key, citation in by_doi.items():
        expected_sources = set(citation["sources"])
        if observed[key] != expected_sources:
            fail(
                f"{relative} DOI {citation['doi']} source drift: "
                f"declared={sorted(expected_sources)}, observed={sorted(observed[key])}"
            )


def check_binding_diagrams() -> None:
    diagram_root = ROOT / "docs" / "diagrams" / "bindings"
    sources = {path.with_suffix("") for path in diagram_root.glob("*.puml")}
    rendered = {path.with_suffix("") for path in diagram_root.glob("*.svg")}
    missing_svg = sorted(path.with_suffix(".svg") for path in sources - rendered)
    orphan_svg = sorted(path.with_suffix(".svg") for path in rendered - sources)
    if missing_svg or orphan_svg:
        fail(
            "binding diagram source/render inventory differs: "
            f"missing={[_relative(path) for path in missing_svg]}, "
            f"orphaned={[_relative(path) for path in orphan_svg]}"
        )

    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (ROOT / "docs", ROOT / "bindings")
        for path in sorted(root.rglob("*.md"))
    )
    for stem in sorted(sources):
        source = stem.with_suffix(".puml")
        text = source.read_text(encoding="utf-8")
        for required in (
            "skinparam backgroundColor #FFFFFF",
            "skinparam shadowing false",
            'skinparam defaultFontName "DejaVu Sans"',
        ):
            if required not in text:
                fail(f"{_relative(source)} omits binding diagram style {required!r}")
        svg_name = stem.with_suffix(".svg").name
        embed = re.compile(rf"!\[[^\]\n]+\]\([^\)\n]*{re.escape(svg_name)}\)")
        if embed.search(markdown) is None:
            fail(
                f"{_relative(stem.with_suffix('.svg'))} is not embedded in documentation"
            )


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


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
        [
            sys.executable,
            str(ROOT / "scripts" / "check-package-documentation.py"),
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
    supplemental = DOCS.get("operationalDocuments")
    if (
        not isinstance(supplemental, list)
        or not supplemental
        or supplemental != sorted(set(supplemental))
    ):
        fail("documentation.operationalDocuments must be a non-empty sorted set")
    for relative in supplemental:
        path, document = read(relative)
        check_links(path, document)
        operational_documents[relative] = document
    for relative, document in operational_documents.items():
        if PLACEHOLDER_RE.search(document):
            fail(f"{relative} contains a documentation placeholder")
        for retired, pattern in RETIRED_PACKAGE_PATTERNS.items():
            if pattern.search(document):
                fail(f"{relative} uses retired public package name {retired!r}")
    check_citation_ledger()
    check_binding_diagrams()
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
