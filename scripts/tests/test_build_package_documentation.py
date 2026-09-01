from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "build_package_documentation",
    Path(__file__).parents[1] / "build-package-documentation.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PackageDocumentationBuildTests(unittest.TestCase):
    def test_every_exported_python_declaration_has_api_text(self) -> None:
        BUILDER.require_python_docstrings()

    def test_output_cleanup_cannot_escape_repository_target(self) -> None:
        with self.assertRaisesRegex(SystemExit, "refusing to clean output"):
            BUILDER.clean_output(BUILDER.ROOT / "bindings" / "python")


if __name__ == "__main__":
    unittest.main()
