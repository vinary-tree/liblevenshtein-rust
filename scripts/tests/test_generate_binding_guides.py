"""Regression tests for generated language-binding guide indexes."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_binding_guides", ROOT / "scripts" / "generate-binding-guides.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/generate-binding-guides.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class InteropRootIndexTests(unittest.TestCase):
    def test_distribution_cells_preserve_their_existing_code_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interop = root / "vinary-tree-interop"
            interop.mkdir()
            (interop / "README.md").write_text(
                "# interop\n\n"
                "<!-- BEGIN GENERATED INTEROP LANGUAGE INDEX; DO NOT EDIT -->\n"
                "stale\n"
                "<!-- END GENERATED INTEROP LANGUAGE INDEX -->\n",
                encoding="utf-8",
            )
            original_root = GENERATOR.ROOT
            try:
                GENERATOR.ROOT = root
                _, rendered = GENERATOR.render_interop_root()
            finally:
                GENERATOR.ROOT = original_root

        self.assertIn("| Python 3.10+ | PyPI package `vinary-tree-interop` |", rendered)
        self.assertNotIn("`PyPI package `vinary-tree-interop``", rendered)
        self.assertIn(
            "| C++20+ | Header-only native facade and CMake/pkg-config package |",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
