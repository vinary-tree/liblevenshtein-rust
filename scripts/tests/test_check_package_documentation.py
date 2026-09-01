from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "check_package_documentation",
    Path(__file__).parents[1] / "check-package-documentation.py",
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class _Response:
    def __init__(self, body: bytes, *, encoding: str = "") -> None:
        self.status = 200
        self.headers = {"Content-Encoding": encoding}
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _ceiling: int) -> bytes:
        return self._body


class PackageDocumentationCheckerTests(unittest.TestCase):
    def test_iso_date_rejects_calendar_impossibilities(self) -> None:
        with self.assertRaisesRegex(SystemExit, "valid calendar date"):
            CHECKER.iso_date("2026-02-30", "observedAt")

    def test_readback_ceiling_applies_after_gzip_decompression(self) -> None:
        compressed = gzip.compress(b"x" * (CHECKER.MAX_READBACK_BYTES + 1))
        response = _Response(compressed, encoding="gzip")
        with mock.patch.object(
            CHECKER.urllib.request, "urlopen", return_value=response
        ):
            with self.assertRaisesRegex(SystemExit, "decompressed public readback"):
                CHECKER.read_url("https://example.invalid/reference", {})

    def test_strict_gate_rejects_pending_publication(self) -> None:
        release = json.loads(CHECKER.RELEASE_PATH.read_text(encoding="utf-8"))
        model = self._minimal_model(release, "pending-publication", "missing")
        with self._patched_models(model, release), mock.patch.object(
            sys, "argv", ["check-package-documentation.py", "--require-complete"]
        ):
            with self.assertRaisesRegex(SystemExit, "pending-publication"):
                CHECKER.main()

    def test_strict_gate_accepts_explicit_numeric_only_candidate(self) -> None:
        release = json.loads(CHECKER.RELEASE_PATH.read_text(encoding="utf-8"))
        model = self._minimal_model(release, "candidate-only", "deferred")
        with self._patched_models(model, release), mock.patch.object(
            sys, "argv", ["check-package-documentation.py", "--require-complete"]
        ), mock.patch("builtins.print"):
            self.assertEqual(CHECKER.main(), 0)

    @staticmethod
    def _minimal_model(
        release: dict[str, object], release_state: str, destination_state: str
    ) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "component": release["component"],
            "canonicalVersion": release["canonical"],
            "sourceRef": release["publication"]["sourceTag"],
            "sourceRepository": "https://github.com/vinary-tree/liblevenshtein-rust",
            "observedAt": dt.date.today().isoformat(),
            "packages": [
                {
                    "id": "candidate",
                    "languages": ["Example"],
                    "ecosystem": "Example registry",
                    "coordinate": "example",
                    "registryVersion": "4.0.0",
                    "releaseState": release_state,
                    "sourceEvidence": ["Cargo.toml"],
                    "releaseProof": "The ecosystem accepts only numeric final versions.",
                    "destinations": [
                        {
                            "kind": kind,
                            "service": "Example documentation",
                            "state": destination_state,
                            "reason": "The package has not been published.",
                        }
                        for kind in ("package-guide", "api-reference")
                    ],
                }
            ],
        }

    @staticmethod
    def _patched_models(
        model: dict[str, object], release: dict[str, object]
    ) -> mock._patch:
        original = Path.read_text

        def read_text(path: Path, *args: object, **kwargs: object) -> str:
            if path == CHECKER.MODEL_PATH:
                return json.dumps(model)
            if path == CHECKER.RELEASE_PATH:
                return json.dumps(release)
            return original(path, *args, **kwargs)

        return mock.patch.object(Path, "read_text", autospec=True, side_effect=read_text)


if __name__ == "__main__":
    unittest.main()
