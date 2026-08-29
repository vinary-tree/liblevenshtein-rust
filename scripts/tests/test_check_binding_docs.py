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


if __name__ == "__main__":
    unittest.main()
