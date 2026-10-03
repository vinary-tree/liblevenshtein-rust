from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "check-swift-docc.py"
SPEC = importlib.util.spec_from_file_location("check_swift_docc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class SwiftDoccTests(unittest.TestCase):
    def test_public_source_types_are_inventoried(self) -> None:
        self.assertGreaterEqual(len(CHECK.public_types()), 16)
        self.assertIn("Transducer", CHECK.public_types())
        self.assertIn("QueryCache", CHECK.public_types())

    def test_missing_public_type_fails(self) -> None:
        target = CHECK.ROOT / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as temporary:
            site = Path(temporary)
            (site / "index.html").write_text("<html></html>", encoding="utf-8")
            module = site / "data" / "documentation" / "liblevenshtein.json"
            module.parent.mkdir(parents=True)
            module.write_text(
                json.dumps(
                    {
                        "identifier": {"url": "/documentation/liblevenshtein"},
                        "abstract": "Search a dictionary",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "omits public Swift types"):
                CHECK.verify(site)

    def test_missing_renderer_asset_is_copied_from_matching_toolchain(self) -> None:
        target = CHECK.ROOT / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as temporary:
            root = Path(temporary)
            site, template = root / "site", root / "template"
            (site / "css").mkdir(parents=True)
            (template / "css").mkdir(parents=True)
            (template / "js").mkdir(parents=True)
            (site / "index.html").write_text(
                '<html><script>var baseUrl = "/guide/"</script>'
                '<script src="/guide/js/index.js"></script>'
                '<link href="/guide/css/index.css" rel="stylesheet"></html>',
                encoding="utf-8",
            )
            (template / "index.html").write_text("<html></html>", encoding="utf-8")
            (site / "css/index.css").write_text("matched", encoding="utf-8")
            (template / "css/index.css").write_text("matched", encoding="utf-8")
            (template / "js/index.js").write_text("renderer", encoding="utf-8")

            CHECK.complete_toolchain_assets(site, template)
            self.assertEqual((site / "js/index.js").read_text(), "renderer")

    def test_mismatched_toolchain_assets_fail_closed(self) -> None:
        target = CHECK.ROOT / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as temporary:
            root = Path(temporary)
            site, template = root / "site", root / "template"
            (site / "css").mkdir(parents=True)
            (template / "css").mkdir(parents=True)
            (site / "index.html").write_text(
                '<html><script>var baseUrl = "/guide/"</script>'
                '<script src="/guide/js/index.js"></script>'
                '<link href="/guide/css/index.css" rel="stylesheet"></html>',
                encoding="utf-8",
            )
            (site / "css/index.css").write_text("old", encoding="utf-8")
            (template / "css/index.css").write_text("new", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "does not match"):
                CHECK.complete_toolchain_assets(site, template)


if __name__ == "__main__":
    unittest.main()
