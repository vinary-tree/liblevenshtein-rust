from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
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
    def test_documentation_entry_points_are_directly_executable(self) -> None:
        scripts = Path(__file__).parents[1]
        for entry_point in (
            scripts / "build-package-documentation.py",
            scripts / "package-documentation-site.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(entry_point), "--help"],
                cwd=BUILDER.ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{entry_point.name}: {completed.stderr or completed.stdout}",
            )

    def test_every_builder_is_required_by_the_release_archive(self) -> None:
        self.assertEqual(
            list(BUILDER.BUILDERS),
            list(BUILDER.GENERATED_SURFACE_LAYOUT),
        )

    def test_documenter_timestamp_is_normalized_to_source_epoch(self) -> None:
        target = BUILDER.ROOT / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as temporary:
            siteinfo = Path(temporary) / ".documenter-siteinfo.json"
            siteinfo.write_text(
                json.dumps(
                    {
                        "documenter": {
                            "documenter_version": "1.18.0",
                            "generation_timestamp": "2099-12-31T23:59:59",
                            "julia_version": "1.12.7",
                        }
                    }
                ),
                encoding="utf-8",
            )
            BUILDER.normalize_documenter_siteinfo(siteinfo, 0)
            self.assertEqual(
                json.loads(siteinfo.read_text(encoding="utf-8"))["documenter"][
                    "generation_timestamp"
                ],
                "1970-01-01T00:00:00",
            )

    def test_every_exported_python_declaration_has_api_text(self) -> None:
        BUILDER.require_python_docstrings()

    def test_output_cleanup_cannot_escape_repository_target(self) -> None:
        with self.assertRaisesRegex(SystemExit, "refusing to clean output"):
            BUILDER.clean_output(BUILDER.ROOT / "bindings" / "python")


if __name__ == "__main__":
    unittest.main()
