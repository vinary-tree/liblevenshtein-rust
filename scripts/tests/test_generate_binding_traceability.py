"""Regression tests for the public binding API traceability generator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_binding_traceability",
    ROOT / "scripts" / "generate-binding-traceability.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/generate-binding-traceability.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class BindingTraceabilityTests(unittest.TestCase):
    def test_runtime_evidence_uses_configured_sibling_root(self) -> None:
        target = ROOT / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="traceability-runtime-", dir=target
        ) as directory:
            configured_root = Path(directory)
            declaration = configured_root / "index.d.ts"
            declaration.write_text("export {};\n", encoding="utf-8")
            with patch.object(GENERATOR, "RUNTIME_ROOT", configured_root):
                resolved = GENERATOR.resolve(
                    "../javascript-runtime/index.d.ts", "runtime declaration"
                )
                self.assertEqual(resolved, declaration.resolve())

    def test_symbol_leaf_preserves_host_method_identity(self) -> None:
        self.assertEqual(GENERATOR.symbol_leaf("QueryCursor.__next__"), "__next__")
        self.assertEqual(GENERATOR.symbol_leaf("Query.GetEnumerator"), "GetEnumerator")

    def test_direct_status_accepts_public_or_backing_native_reference(self) -> None:
        self.assertEqual(
            GENERATOR.direct_status(
                "Transducer.query", "llev_transducer_query_utf8", "query(pattern)"
            ),
            "complete",
        )
        self.assertEqual(
            GENERATOR.direct_status(
                "Transducer.query",
                "llev_transducer_query_utf8",
                "llev_transducer_query_utf8(handle, pattern)",
            ),
            "complete",
        )
        self.assertEqual(
            GENERATOR.direct_status(
                "Transducer.query", "llev_transducer_query_utf8", "close(handle)"
            ),
            "audit-required",
        )

    def test_documentation_requires_the_exact_public_symbol(self) -> None:
        self.assertEqual(
            GENERATOR.documentation_status(
                "QueryCursor.__next__", "Use `QueryCursor.__next__` to advance."
            ),
            "complete",
        )
        self.assertEqual(
            GENERATOR.documentation_status(
                "QueryCursor.__next__", "Use the cursor iterator to advance."
            ),
            "audit-required",
        )

    def test_evidence_paths_cannot_leave_the_workspace(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            GENERATOR.resolve("../../../etc/passwd", "test.evidence")
        self.assertIn("leaves the indexed workspace", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
