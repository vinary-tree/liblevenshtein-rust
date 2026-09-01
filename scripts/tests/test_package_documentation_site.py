from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "package_documentation_site",
    Path(__file__).parents[1] / "package-documentation-site.py",
)
assert SPEC is not None and SPEC.loader is not None
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)


class PackageDocumentationSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SITE.ROOT.joinpath("target").mkdir(parents=True, exist_ok=True)

    def test_semver_order_places_final_after_prereleases(self) -> None:
        versions = [
            "4.0.0-rc.10",
            "4.0.0",
            "4.0.0-rc.2",
            "3.9.9",
            "4.0.0-alpha.12",
        ]
        self.assertEqual(
            sorted(versions, key=SITE.semver_key, reverse=True),
            [
                "4.0.0",
                "4.0.0-rc.10",
                "4.0.0-rc.2",
                "4.0.0-alpha.12",
                "3.9.9",
            ],
        )

    def test_archive_bytes_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory(dir=SITE.ROOT / "target") as temporary:
            root = Path(temporary)
            version = root / "4.0.0-rc.5"
            (version / "native").mkdir(parents=True)
            (version / "native" / "index.html").write_text(
                "<h1>native API</h1>\n", encoding="utf-8"
            )
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            SITE.deterministic_archive(version, first)
            SITE.deterministic_archive(version, second)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )

    def test_archive_reader_rejects_parent_traversal(self) -> None:
        contents = io.BytesIO()
        with tarfile.open(fileobj=contents, mode="w") as archive:
            info = tarfile.TarInfo("4.0.0/../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        contents.seek(0)
        with tarfile.open(fileobj=contents, mode="r:") as archive:
            with self.assertRaisesRegex(SystemExit, "unsafe or unversioned"):
                SITE.safe_archive_members(archive)

    def test_archive_reader_rejects_duplicate_members(self) -> None:
        contents = io.BytesIO()
        with tarfile.open(fileobj=contents, mode="w") as archive:
            for payload in (b"first", b"second"):
                info = tarfile.TarInfo("4.0.0/index.html")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        contents.seek(0)
        with tarfile.open(fileobj=contents, mode="r:") as archive:
            with self.assertRaisesRegex(SystemExit, "duplicate archive member"):
                SITE.safe_archive_members(archive)

    def test_manifest_requires_every_documentation_surface(self) -> None:
        with tempfile.TemporaryDirectory(dir=SITE.ROOT / "target") as temporary:
            version = Path(temporary) / "4.0.0-rc.5"
            (version / "native").mkdir(parents=True)
            index = version / "native" / "index.html"
            index.write_text("native\n", encoding="utf-8")
            manifest = {
                "schemaVersion": 1,
                "component": "liblevenshtein",
                "version": version.name,
                "sourceRef": f"v{version.name}",
                "surfaces": ["native"],
                "sha256": {"native/index.html": SITE.digest(index)},
            }
            (version / "documentation-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "every required surface"):
                SITE.verify_extracted_version(version)

    def test_release_archive_requires_exact_clean_tag(self) -> None:
        with tempfile.TemporaryDirectory(dir=SITE.ROOT / "target") as temporary:
            repository = Path(temporary)
            tracked = repository / "tracked.txt"
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "core.fsmonitor", "false"],
                cwd=repository,
                check=True,
            )
            tracked.write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Documentation Test",
                    "-c",
                    "user.email=documentation@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "tag", "v1.0.0"], cwd=repository, check=True)
            original_root = SITE.ROOT
            SITE.ROOT = repository
            try:
                SITE.require_source_identity("v1.0.0")
                tracked.write_text("dirty\n", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "unstaged tracked"):
                    SITE.require_source_identity("v1.0.0")
                subprocess.run(
                    ["git", "add", "tracked.txt"], cwd=repository, check=True
                )
                with self.assertRaisesRegex(SystemExit, "staged tracked"):
                    SITE.require_source_identity("v1.0.0")
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=Documentation Test",
                        "-c",
                        "user.email=documentation@example.invalid",
                        "commit",
                        "-qm",
                        "second",
                    ],
                    cwd=repository,
                    check=True,
                )
                with self.assertRaisesRegex(SystemExit, "does not equal immutable"):
                    SITE.require_source_identity("v1.0.0")
            finally:
                SITE.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
