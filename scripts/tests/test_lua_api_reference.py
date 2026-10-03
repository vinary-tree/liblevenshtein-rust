from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import lua_api_reference as reference


class LuaApiReferenceTests(unittest.TestCase):
    def test_every_registered_symbol_has_documented_semantics(self) -> None:
        model = reference.load_and_validate()
        exported = {
            entry["name"] for group in model["groups"] for entry in group["entries"]
        }
        self.assertIn("transducer", exported)
        self.assertIn("query_cache", exported)
        self.assertIn("true_damerau_distance_threshold", exported)

    def test_render_is_byte_reproducible(self) -> None:
        model = reference.load_and_validate()
        first = reference.render(model, "4.0.0-rc.6", "v4.0.0-rc.6")
        second = reference.render(model, "4.0.0-rc.6", "v4.0.0-rc.6")
        self.assertEqual(first.encode(), second.encode())
        self.assertIn("levenshtein.transducer", first)
        self.assertIn("cache:stats", first)

    def test_missing_registration_is_rejected(self) -> None:
        model = reference.load_and_validate()
        model["groups"][0]["entries"] = model["groups"][0]["entries"][1:]
        with tempfile.TemporaryDirectory(dir=reference.ROOT / "target") as temporary:
            alternate = Path(temporary) / "incomplete.json"
            alternate.write_text(json.dumps(model), encoding="utf-8")
            with (
                mock.patch.object(reference, "SPEC", alternate),
                self.assertRaisesRegex(ValueError, "functions differs"),
            ):
                reference.load_and_validate()


if __name__ == "__main__":
    unittest.main()
