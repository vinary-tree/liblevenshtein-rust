#!/usr/bin/env python3
"""Build versioned API references from the authoritative binding surfaces."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from html import escape
from pathlib import Path

try:
    from scripts.package_documentation_surfaces import GENERATED_SURFACE_LAYOUT
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from package_documentation_surfaces import GENERATED_SURFACE_LAYOUT


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


def source_date_epoch() -> int:
    configured = os.environ.get("SOURCE_DATE_EPOCH")
    if configured is not None:
        try:
            epoch = int(configured)
        except ValueError:
            fail("SOURCE_DATE_EPOCH must be an integer Unix timestamp")
        if epoch < 0:
            fail("SOURCE_DATE_EPOCH cannot be negative")
        return epoch
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "show",
            "-s",
            "--format=%ct",
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value.isdecimal():
        fail("cannot derive SOURCE_DATE_EPOCH from the source revision")
    return int(value)


def normalize_documenter_siteinfo(path: Path, epoch: int) -> None:
    if not path.is_file():
        fail("Documenter did not produce .documenter-siteinfo.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    documenter = value.get("documenter") if isinstance(value, dict) else None
    if not isinstance(documenter, dict):
        fail("Documenter site information lacks its metadata object")
    for field in ("documenter_version", "julia_version", "generation_timestamp"):
        if not isinstance(documenter.get(field), str):
            fail(f"Documenter site information lacks {field}")
    documenter["generation_timestamp"] = datetime.fromtimestamp(
        epoch, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def generated_digests(surfaces: list[str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for surface in surfaces:
        root = OUTPUT_ROOT / surface
        if not root.is_dir():
            fail(f"generated surface is absent: {surface}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                fail(f"unsupported generated documentation entry: {path}")
            if path.is_file():
                relative = f"{surface}/{path.relative_to(root).as_posix()}"
                digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digests:
        fail("documentation generators produced no files")
    return digests


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
            (
                "liblevenshtein="
                "https://github.com/vinary-tree/liblevenshtein-rust/blob/"
                f"{source_ref}/bindings/python/src/liblevenshtein/"
            ),
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


def build_julia(version: str, source_ref: str) -> None:
    output = OUTPUT_ROOT / "julia"
    clean_output(output)
    docs = ROOT / "bindings" / "julia" / "Liblevenshtein" / "docs"
    environment_files = (docs / "Project.toml", docs / "Manifest.toml")
    missing_environment = [path for path in environment_files if not path.is_file()]
    if missing_environment:
        fail(
            "Julia documentation environment is incomplete: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing_environment)
        )
    locked_environment = {path: path.read_bytes() for path in environment_files}
    environment = dict(os.environ)
    environment["LIBLEVENSHTEIN_DOCS_DEPLOY"] = "0"
    environment["VINARY_TREE_DOC_OUTPUT"] = str(output)
    environment["VINARY_TREE_DOC_SOURCE_REF"] = source_ref
    epoch = source_date_epoch()
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    environment["JULIA_DEPOT_PATH"] = str(
        ROOT / "target" / "package-documentation-julia-depot"
    )
    expression = "\n".join(
        (
            "using Pkg",
            f"Pkg.activate({json.dumps(str(docs))})",
            "Pkg.instantiate()",
            f"include({json.dumps(str(docs / 'make.jl'))})",
        )
    )
    run([executable("julia"), "--startup-file=no", "-e", expression], env=environment)
    changed_environment = [
        path.relative_to(ROOT)
        for path, original in locked_environment.items()
        if path.read_bytes() != original
    ]
    if changed_environment:
        fail(
            "Julia changed its locked documentation environment: "
            + ", ".join(map(str, changed_environment))
        )
    normalize_documenter_siteinfo(output / ".documenter-siteinfo.json", epoch)
    require_markers(output / "index.html", ("Liblevenshtein.jl", "Transducer"))


def build_raku(version: str, _source_ref: str) -> None:
    output = OUTPUT_ROOT / "raku"
    clean_output(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "bindings" / "raku" / "docs" / "Liblevenshtein.rakudoc"
    completed = subprocess.run(
        [executable("raku"), "--doc=Text", str(source)],
        cwd=ROOT,
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = completed.stdout
    if not rendered.strip():
        fail("Raku Pod renderer produced an empty API reference")
    (output / "Liblevenshtein.txt").write_text(rendered, encoding="utf-8")
    (output / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>Liblevenshtein Raku API {escape(version)}</title>"
        "<style>body{font:16px/1.5 system-ui;margin:2rem auto;max-width:72rem;"
        "padding:0 1rem;color:#18212b}pre{white-space:pre-wrap}</style>"
        "</head><body>"
        f"<h1>Liblevenshtein Raku API {escape(version)}</h1>"
        f"<pre>{escape(rendered)}</pre></body></html>\n",
        encoding="utf-8",
    )
    require_markers(output / "index.html", (version, "Transducer", "QueryCursor"))


BUILDERS: dict[str, Callable[[str, str], None]] = {
    "native": build_native,
    "python": build_python,
    "javascript": build_javascript,
    "julia": build_julia,
    "raku": build_raku,
}

if BUILDERS.keys() != GENERATED_SURFACE_LAYOUT.keys():
    fail(
        "builder/archive surface inventories disagree: "
        f"builders={list(BUILDERS)}, archive={list(GENERATED_SURFACE_LAYOUT)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        action="append",
        choices=[*BUILDERS, "all"],
        default=[],
        help="surface to build; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="build each selected surface twice and require byte-identical files",
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
    if args.verify_reproducible:
        first = generated_digests(surfaces)
        for surface in surfaces:
            BUILDERS[surface](version, source_ref)
        second = generated_digests(surfaces)
        changed = sorted(
            path
            for path in first.keys() | second.keys()
            if first.get(path) != second.get(path)
        )
        if changed:
            preview = ", ".join(changed[:20])
            suffix = "" if len(changed) <= 20 else f" (+{len(changed) - 20} more)"
            fail(f"generated documentation is not reproducible: {preview}{suffix}")
        print(
            "package-documentation-build: verified byte-reproducible output for "
            + ", ".join(surfaces)
        )
    print(f"package-documentation-build: built {', '.join(surfaces)} for {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
