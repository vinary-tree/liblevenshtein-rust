#!/usr/bin/env python3
"""Build versioned API references from the authoritative binding surfaces."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "target" / "package-documentation"
RELEASE_PATH = ROOT / "release" / "version.json"


def fail(message: str) -> None:
    raise SystemExit(f"package-documentation-build: {message}")


def executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        fail(f"required documentation generator is unavailable: {name}")
    return resolved


def clean_output(output: Path) -> None:
    resolved = output.resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        fail(f"refusing to clean output outside {OUTPUT_ROOT}: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)
    resolved.parent.mkdir(parents=True, exist_ok=True)


def run(
    command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> None:
    rendered = " ".join(command)
    print(f"package-documentation-build: running {rendered}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_markers(path: Path, markers: tuple[str, ...]) -> None:
    if not path.is_file():
        fail(f"generator did not produce {path.relative_to(ROOT)}")
    body = path.read_text(encoding="utf-8", errors="strict")
    for marker in markers:
        if marker not in body:
            fail(f"{path.relative_to(ROOT)} is missing marker {marker!r}")


def require_doxygen_symbols(xml_root: Path) -> None:
    model = json.loads((ROOT / "bindings" / "api.json").read_text(encoding="utf-8"))
    required_c = {entry["name"] for entry in model["cFunctions"]}
    required_types = {
        "LlevStatus",
        "LlevAlgorithm",
        "LlevQueryOrder",
        "LlevPhoneticRuleSetKind",
        "LlevMatch",
        "LlevMatchBatchView",
        "LlevOwnedString",
    }
    required_cpp = {
        "vinary_tree::liblevenshtein::error",
        "vinary_tree::liblevenshtein::batch",
        "vinary_tree::liblevenshtein::query_cursor",
        "vinary_tree::liblevenshtein::transducer",
    }
    required_cpp_enums = {"algorithm", "query_order"}
    member_names: set[str] = set()
    compound_names: set[str] = set()
    for path in xml_root.glob("*.xml"):
        tree = ET.parse(path)
        member_names.update(
            node.text for node in tree.findall(".//memberdef/name") if node.text
        )
        compound_names.update(
            node.text for node in tree.findall(".//compoundname") if node.text
        )
    missing_c = sorted(required_c - member_names)
    declared_names = member_names | compound_names
    missing_types = sorted(required_types - declared_names)
    missing_cpp = sorted(required_cpp - compound_names)
    missing_cpp_enums = sorted(required_cpp_enums - member_names)
    if missing_c or missing_types or missing_cpp or missing_cpp_enums:
        fail(
            "Doxygen XML omits public declarations: "
            f"C functions={missing_c}, C types={missing_types}, "
            f"C++ types={missing_cpp}, C++ enums={missing_cpp_enums}"
        )


def require_python_docstrings() -> None:
    source_root = ROOT / "bindings" / "python" / "src" / "liblevenshtein"
    init_tree = ast.parse((source_root / "__init__.py").read_text(encoding="utf-8"))
    exports: set[str] | None = None
    for node in init_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            exports = set(value)
            break
    if not exports:
        fail("Python package lacks a non-empty literal __all__ API inventory")

    classes: dict[str, ast.ClassDef] = {}
    for source_name in ("_native.py", "_generated.py"):
        tree = ast.parse((source_root / source_name).read_text(encoding="utf-8"))
        classes.update(
            (node.name, node) for node in tree.body if isinstance(node, ast.ClassDef)
        )
    missing_classes = sorted(exports - set(classes))
    if missing_classes:
        fail(f"Python __all__ exports lack class definitions: {missing_classes}")

    public_dunders = {
        "__init__",
        "__iter__",
        "__next__",
        "__len__",
        "__getitem__",
        "__enter__",
        "__exit__",
    }
    missing_docs: list[str] = []
    for class_name in sorted(exports):
        class_node = classes[class_name]
        if ast.get_docstring(class_node, clean=False) is None:
            missing_docs.append(class_name)
        for node in class_node.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") and node.name not in public_dunders:
                continue
            if ast.get_docstring(node, clean=False) is None:
                missing_docs.append(f"{class_name}.{node.name}")
    if missing_docs:
        fail(f"Python public declarations lack docstrings: {missing_docs}")


def build_native(version: str, source_ref: str) -> None:
    output = OUTPUT_ROOT / "native"
    clean_output(output)
    input_root = ROOT / "target" / "package-documentation-input"
    input_root.mkdir(parents=True, exist_ok=True)
    mainpage = input_root / "native-mainpage.md"
    source_url = f"https://github.com/vinary-tree/liblevenshtein-rust/blob/{source_ref}"
    template = (ROOT / "docs" / "api" / "doxygen" / "mainpage.md.in").read_text(
        encoding="utf-8"
    )
    mainpage.write_text(
        template.replace("@VERSION@", version).replace("@SOURCE_URL@", source_url),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "VINARY_TREE_DOC_MAINPAGE": str(mainpage),
            "VINARY_TREE_DOC_OUTPUT": str(output),
            "VINARY_TREE_DOC_VERSION": version,
            "VINARY_TREE_SOURCE_ROOT": str(ROOT),
        }
    )
    run(
        [executable("doxygen"), str(ROOT / "docs" / "api" / "doxygen" / "Doxyfile")],
        env=environment,
    )
    require_markers(
        output / "html" / "index.html",
        ("liblevenshtein C and C++ API", version),
    )
    if not (output / "xml" / "index.xml").is_file():
        fail("Doxygen did not produce the machine-readable XML index")
    require_doxygen_symbols(output / "xml")


def build_python(version: str, source_ref: str) -> None:
    output = OUTPUT_ROOT / "python"
    clean_output(output)
    require_python_docstrings()
    environment = dict(os.environ)
    python_path = str(ROOT / "bindings" / "python" / "src")
    if inherited := environment.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{inherited}"
    environment["PYTHONPATH"] = python_path
    environment["VINARY_TREE_DOC_VERSION"] = version
    run(
        [
            executable("pdoc"),
            "--output-directory",
            str(output),
            "--docformat",
            "google",
            "--footer-text",
            f"liblevenshtein {version}",
            "--edit-url",
            "liblevenshtein="
            "https://github.com/vinary-tree/liblevenshtein-rust/blob/"
            f"{source_ref}/bindings/python/src/liblevenshtein/",
            "liblevenshtein",
        ],
        env=environment,
    )
    require_markers(
        output / "liblevenshtein.html",
        (version, "Transducer", "QueryCursor", "PhoneticRuleSet"),
    )


def build_javascript(version: str, _source_ref: str) -> None:
    output = OUTPUT_ROOT / "javascript"
    clean_output(output)
    run([executable("npm"), "run", "docs"], cwd=ROOT / "bindings" / "javascript")
    require_markers(
        output / "index.html",
        ("@vinary-tree/liblevenshtein API", version),
    )


BUILDERS: dict[str, Callable[[str, str], None]] = {
    "native": build_native,
    "python": build_python,
    "javascript": build_javascript,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        action="append",
        choices=[*BUILDERS, "all"],
        default=[],
        help="surface to build; repeat as needed (default: all)",
    )
    args = parser.parse_args()

    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    version = release.get("canonical")
    if not isinstance(version, str) or not version:
        fail("release/version.json lacks canonical")
    publication = release.get("publication")
    source_ref = publication.get("sourceTag") if isinstance(publication, dict) else None
    if not isinstance(source_ref, str) or not source_ref:
        fail("release/version.json lacks publication.sourceTag")
    selected = args.surface or ["all"]
    if "all" in selected and len(selected) != 1:
        fail("--surface all cannot be combined with another surface")
    surfaces = list(BUILDERS) if selected == ["all"] else selected
    for surface in surfaces:
        BUILDERS[surface](version, source_ref)
    print(f"package-documentation-build: built {', '.join(surfaces)} for {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
