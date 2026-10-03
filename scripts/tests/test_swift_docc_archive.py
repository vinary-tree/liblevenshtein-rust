from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "package-swift-docc.py"
SPEC = importlib.util.spec_from_file_location("package_swift_docc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class SwiftDoccArchiveTests(unittest.TestCase):
    def test_round_trip_preserves_symbol_filenames_and_is_reproducible(self) -> None:
        ARCHIVE.TARGET.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ARCHIVE.TARGET) as temporary:
            root = Path(temporary)
            site, archive, output = (
                root / "site",
                root / "docs.tar.gz",
                root / "readback",
            )
            symbol = site / "data/documentation/liblevenshtein/algorithm/!=(_:_:).json"
            symbol.parent.mkdir(parents=True)
            symbol.write_text('{"identifier":{"url":"/algorithm/!=(_:_:).json"}}')
            (site / "index.html").write_text("<html>Swift DocC</html>")

            ARCHIVE.build(site, archive)
            first = archive.read_bytes()
            ARCHIVE.build(site, archive)
            self.assertEqual(first, archive.read_bytes())
            ARCHIVE.verify(archive, output)
            self.assertEqual(
                (output / symbol.relative_to(site)).read_bytes(), symbol.read_bytes()
            )

    def test_target_boundary_rejects_external_output(self) -> None:
        with self.assertRaisesRegex(SystemExit, "inside the repository target"):
            ARCHIVE.under_target(ARCHIVE.ROOT / "outside.tar.gz")

    def test_readback_rejects_path_traversal(self) -> None:
        ARCHIVE.TARGET.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ARCHIVE.TARGET) as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, mode="w:gz") as tar:
                body = b"bad"
                info = tarfile.TarInfo("site/../escape")
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
            with self.assertRaisesRegex(SystemExit, "unsafe archive path"):
                ARCHIVE.verify(archive, root / "readback")


if __name__ == "__main__":
    unittest.main()
