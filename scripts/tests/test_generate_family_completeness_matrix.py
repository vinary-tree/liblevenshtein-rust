"""Regression tests for the family completeness inventory."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate-family-completeness-matrix.py"
SPEC = importlib.util.spec_from_file_location("family_completeness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiscoverBindingLanguagesTests(unittest.TestCase):
    def test_discovers_direct_and_shared_runtime_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "bindings/python",
                "bindings/jvm",
                "bindings/dotnet",
                "bindings/javascript/cljs",
                "bindings/raku",
                "bindings/julia",
            ):
                (root / relative).mkdir(parents=True, exist_ok=True)

            self.assertEqual(
                MODULE.discover_binding_languages(root),
                {
                    "python",
                    "java",
                    "kotlin",
                    "scala",
                    "csharp",
                    "fsharp",
                    "javascript",
                    "typescript",
                    "clojurescript",
                    "raku",
                    "julia",
                },
            )

    def test_ignores_non_binding_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bindings" / "generated").mkdir(parents=True)
            (root / "bindings" / "conformance").mkdir(parents=True)

            self.assertEqual(MODULE.discover_binding_languages(root), set())

    def test_project_without_bindings_has_no_discovered_languages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(MODULE.discover_binding_languages(Path(directory)), set())


class DocumentationTopicTests(unittest.TestCase):
    def test_inherited_complete_state_requires_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                SystemExit, "is complete without documentation evidence"
            ):
                MODULE.documentation_topic(
                    "overview",
                    "complete",
                    None,
                    "project|julia|capability",
                    Path(directory),
                )

    def test_inherited_inapplicable_state_remains_inapplicable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                MODULE.documentation_topic(
                    "overview",
                    "inapplicable",
                    None,
                    "project|julia|capability",
                    Path(directory),
                ),
                ("inapplicable", "-"),
            )


class KnownMissingCapabilityTests(unittest.TestCase):
    def test_normalizes_reviewed_gap_with_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.md").write_text("# Review\n", encoding="utf-8")

            self.assertEqual(
                MODULE.normalize_known_missing(
                    "project",
                    {
                        "julia": {
                            "capabilities": ["feature"],
                            "evidence": "review.md#finding",
                        }
                    },
                    {"feature"},
                    {"julia"},
                    {"julia"},
                    root,
                ),
                {"julia": ({"feature"}, "review.md#finding")},
            )

    def test_rejects_unknown_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.md").write_text("# Review\n", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unknown capabilities"):
                MODULE.normalize_known_missing(
                    "project",
                    {
                        "julia": {
                            "capabilities": ["omitted-feature"],
                            "evidence": "review.md",
                        }
                    },
                    {"feature"},
                    {"julia"},
                    {"julia"},
                    root,
                )

    def test_rejects_gap_without_declared_facade(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires declared facade evidence"):
            MODULE.normalize_known_missing(
                "project",
                {
                    "julia": {
                        "capabilities": ["feature"],
                        "evidence": "review.md",
                    }
                },
                {"feature"},
                {"julia"},
                set(),
            )


if __name__ == "__main__":
    unittest.main()
