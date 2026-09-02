"""Regression tests for the exhaustive host-extension trait matrix."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_extension_provider_matrix",
    ROOT / "scripts" / "generate-extension-provider-matrix.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load generate-extension-provider-matrix.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class ExtensionProviderMatrixTests(unittest.TestCase):
    def temporary_workspace(self) -> tempfile.TemporaryDirectory[str]:
        target = ROOT / "target"
        target.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="extension-matrix-", dir=target)

    def test_root_environment_overrides_relative_default(self) -> None:
        with self.temporary_workspace() as directory:
            workspace = Path(directory)
            default = workspace / "siblings" / "lling-llang"
            override = workspace / "alternate" / "lling-llang"
            default.mkdir(parents=True)
            override.mkdir(parents=True)
            model = {
                "projects": {
                    "lling-llang": {
                        "root": "siblings/lling-llang",
                        "rootEnvironment": "LLING_LLANG_ROOT",
                    }
                },
                "evidenceProjects": {},
            }
            roots = GENERATOR.resolve_roots(
                workspace, model, {"LLING_LLANG_ROOT": str(override)}
            )
            self.assertEqual(roots["lling-llang"], override.resolve())

    def test_absolute_modeled_root_is_rejected(self) -> None:
        model = {
            "projects": {
                "lling-llang": {
                    "root": "/machine-specific/lling-llang",
                    "rootEnvironment": "LLING_LLANG_ROOT",
                }
            },
            "evidenceProjects": {},
        }
        with self.assertRaises(SystemExit) as raised:
            GENERATOR.resolve_roots(ROOT, model, {})
        self.assertIn("must be repository-relative", str(raised.exception))

    def test_scanner_ignores_comments_and_multiline_strings(self) -> None:
        with self.temporary_workspace() as directory:
            project = Path(directory)
            source = project / "src" / "traits.rs"
            source.parent.mkdir()
            source.write_text(
                """pub trait Visible<'a> {}
/*
pub trait BlockComment {}
*/
const ORDINARY: &str = "
pub trait OrdinaryString {}
";
const RAW: &str = r#"
pub trait RawString {}
"#;
// pub trait LineComment {}
pub trait AlsoVisible {}
pub unsafe trait UnsafeVisible {}
""",
                encoding="utf-8",
            )
            traits = GENERATOR.discover_traits("fixture", project)
            self.assertEqual(
                set(traits),
                {
                    "fixture:src/traits.rs:AlsoVisible",
                    "fixture:src/traits.rs:UnsafeVisible",
                    "fixture:src/traits.rs:Visible",
                },
            )

    def test_same_trait_name_in_distinct_modules_remains_distinct(self) -> None:
        with self.temporary_workspace() as directory:
            project = Path(directory)
            first = project / "src" / "first.rs"
            second = project / "src" / "second.rs"
            first.parent.mkdir()
            first.write_text("pub trait LanguageModel {}\n", encoding="utf-8")
            second.write_text("pub trait LanguageModel {}\n", encoding="utf-8")
            traits = GENERATOR.discover_traits("fixture", project)
            self.assertEqual(len(traits), 2)
            self.assertIn("fixture:src/first.rs:LanguageModel", traits)
            self.assertIn("fixture:src/second.rs:LanguageModel", traits)


if __name__ == "__main__":
    unittest.main()
