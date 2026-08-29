"""Regression tests for the binding-documentation link and anchor gate."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_binding_docs", ROOT / "scripts" / "check-binding-docs.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/check-binding-docs.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class BindingDocumentationAnchorTests(unittest.TestCase):
    def test_github_heading_slug_matches_project_heading_styles(self) -> None:
        cases = {
            "Notation & Terminology": "notation--terminology",
            "Time Series (Move–Split–Merge)": "time-series-movesplitmerge",
            "7. Transducer and cursor (11)": "7-transducer-and-cursor-11",
            "`VtStatus` — the one error currency": "vtstatus--the-one-error-currency",
        }
        for heading, expected in cases.items():
            with self.subTest(heading=heading):
                self.assertEqual(CHECKER.github_heading_slug(heading), expected)

    def test_same_document_anchor_is_accepted(self) -> None:
        CHECKER.check_links(ROOT / "README.md", "[Quick start](#quick-start)")

    def test_cross_document_anchor_is_accepted(self) -> None:
        CHECKER.check_links(
            ROOT / "README.md",
            "[Introspection](docs/bindings/c-abi-reference.md#4-introspection-4)",
        )

    def test_missing_markdown_anchor_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            CHECKER.check_links(
                ROOT / "README.md", "[Missing](#this-anchor-does-not-exist)"
            )
        self.assertIn("broken local anchor", stderr.getvalue())

    def test_documented_facades_share_guides_and_examples_with_surface_model(
        self,
    ) -> None:
        for facade, documentation in CHECKER.DOCS["facades"].items():
            surface = CHECKER.SURFACE_MODEL["languages"][facade]
            with self.subTest(facade=facade):
                self.assertEqual(surface["readme"], documentation["guide"])
                self.assertIn(documentation["example"], surface["tests"])

    def test_every_guide_names_its_exact_canonical_package(self) -> None:
        for facade, documentation in CHECKER.DOCS["facades"].items():
            guide = (ROOT / documentation["guide"]).read_text(encoding="utf-8")
            with self.subTest(facade=facade):
                self.assertIn(CHECKER.canonical_package(facade), guide)

    def test_retired_package_names_are_boundary_aware(self) -> None:
        mistaken = "Use @vinary-tree/interop and @vinary-tree/vinary-tree."
        canonical = "Use @vinary-tree/vinary-tree-interop."
        self.assertTrue(
            CHECKER.RETIRED_PACKAGE_PATTERNS["@vinary-tree/interop"].search(mistaken)
        )
        runtime_pattern = CHECKER.RETIRED_PACKAGE_PATTERNS["@vinary-tree/vinary-tree"]
        self.assertTrue(runtime_pattern.search(mistaken))
        self.assertIsNone(runtime_pattern.search(canonical))

    def test_doi_link_parser_accepts_plain_and_angle_wrapped_targets(self) -> None:
        text = (
            "[DOI:10.1145/1297027.1297033]"
            "(https://doi.org/10.1145/1297027.1297033)\n"
            "[10.1016/0022-0000(89)90034-2]"
            "(<https://doi.org/10.1016/0022-0000(89)90034-2>)"
        )
        self.assertEqual(
            [
                match.group("angle") or match.group("plain")
                for match in CHECKER.DOI_LINK_RE.finditer(text)
            ],
            ["10.1145/1297027.1297033", "10.1016/0022-0000(89)90034-2"],
        )

    def test_doi_labels_and_urls_are_consistent_in_governed_documents(self) -> None:
        CHECKER.check_citation_ledger()

    def test_resolving_but_unrelated_doi_is_rejected(self) -> None:
        prose = (
            "2. T. Kalibera, R. Jones. “Rigorous Benchmarking in Reasonable Time.” "
            "*ISMM*, 2013. "
            "[DOI:10.1145/2660193.2660196]"
            "(https://doi.org/10.1145/2660193.2660196)"
        )
        link = CHECKER.DOI_LINK_RE.search(prose)
        self.assertIsNotNone(link)
        wrong_work = {
            "title": "An experimental survey of energy management across the stack",
            "firstAuthor": "Kambadur",
            "year": 2014,
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            CHECKER.check_citation_identity("methodology.md", prose, link, wrong_work)
        self.assertIn("does not match ledger title", stderr.getvalue())

    def test_normalized_citation_text_preserves_bibliographic_words(self) -> None:
        self.assertEqual(
            CHECKER.normalized_citation_text(
                "*Bringing the web up to speed with WebAssembly.*"
            ),
            "bringing the web up to speed with webassembly",
        )


if __name__ == "__main__":
    unittest.main()
