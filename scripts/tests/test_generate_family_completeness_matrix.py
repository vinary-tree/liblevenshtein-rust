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


if __name__ == "__main__":
    unittest.main()
