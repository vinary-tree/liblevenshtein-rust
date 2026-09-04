#!/usr/bin/env python3
"""Dependency-free architectural and packaging checks for language bindings."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import tomllib
from release_source_refs import validate_source_refs

ROOT = Path(__file__).resolve().parents[1]
INTEROP_ROOT = Path(
    os.environ.get("VINARY_TREE_INTEROP_ROOT", ROOT.parent / "vinary-tree-interop")
).resolve()
RUNTIME_ROOT = Path(
    os.environ.get(
        "VINARY_TREE_JAVASCRIPT_RUNTIME_ROOT", ROOT.parent / "javascript-runtime"
    )
).resolve()
LIBDICT_ROOT = Path(
    os.environ.get("LIBDICTENSTEIN_ROOT", ROOT.parent / "libdictenstein")
).resolve()
LLING_ROOT = Path(
    os.environ.get("LLING_LLANG_ROOT", ROOT.parent / "lling-llang")
).resolve()
DUALLITY_ROOT = Path(
    os.environ.get("DUALLITY_ROOT", ROOT.parent / "duallity")
).resolve()
MODEL = json.loads((ROOT / "bindings" / "api.json").read_text(encoding="utf-8"))
RELEASE_MODEL = json.loads(
    (ROOT / "release" / "version.json").read_text(encoding="utf-8")
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--check",
    action="store_true",
    help="verify the committed completeness matrix instead of rewriting it",
)
ARGS = parser.parse_args()
GENERATED_TREE_PARTS = frozenset(
    {
        ".build",
        ".cpcache",
        ".gradle",
        ".pytest_cache",
        ".swiftpm",
        ".venv",
        "__pycache__",
        "_build",
        "bin",
        "build",
        "dist-newstyle",
        "node_modules",
        "obj",
        "target",
        "venv",
    }
)
PUBLISHABLE_EXCLUDED_PARTS = GENERATED_TREE_PARTS | {"smoke", "test", "tests"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def text(path: Path) -> str:
    require(path.is_file(), f"required binding file is missing: {display(path)}")
    return path.read_text(encoding="utf-8")


def surface_file(relative: str) -> Path:
    """Resolve modeled facade evidence against its configured repository root."""
    runtime_prefix = "../javascript-runtime/"
    if relative.startswith(runtime_prefix):
        return RUNTIME_ROOT / relative.removeprefix(runtime_prefix)
    return ROOT / relative


subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "generate-bindings.py"), "--check"],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "scripts" / "generate-family-completeness-matrix.py"),
        "--check",
    ],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "scripts" / "generate-extension-provider-matrix.py"),
        "--check",
    ],
    cwd=ROOT,
    check=True,
)

organization = MODEL["organization"]
interop = MODEL["interop"]
packages = MODEL["packages"]
require(organization["github"] == "vinary-tree", "wrong GitHub organization")
require(interop["cPrefix"] == "vt_", "shared ABI must use the vt_ prefix")
require(interop["crate"] == "vinary-tree-interop", "wrong interop crate")
require(
    interop["maven"] == "io.vinarytree:vinary-tree-interop",
    "wrong interop Maven coordinate",
)
release_coordinates = RELEASE_MODEL.get("coordinates", {})
require(
    release_coordinates.get("mavenGroup") == organization["mavenGroup"]
    and release_coordinates.get("mavenArtifact") == "liblevenshtein"
    and f"{release_coordinates.get('interopMavenGroup')}:{release_coordinates.get('interopMavenArtifact')}"
    == interop["maven"],
    "release model and generated binding model disagree on canonical Maven coordinates",
)
try:
    validate_source_refs(RELEASE_MODEL)
except (TypeError, ValueError) as error:
    raise SystemExit(f"invalid release source refs: {error}") from error
require(
    release_coordinates.get("javaPackage") == organization["javaPackage"],
    "release model and generated binding model disagree on the Java package",
)
require(
    release_coordinates.get("legacyMavenGroups")
    == ["com.github.dylon", "com.github.universal-automata"],
    "historical liblevenshtein Maven relocation groups are incomplete",
)
require(
    interop["scalarWfstInterfaceVersion"] == 1, "wrong scalar WFST interface version"
)
require(
    interop["dictionaryVisitInterfaceVersion"] == 1,
    "wrong dictionary visit interface version",
)
require(
    interop["dictionaryGraphInterfaceVersion"] == 1,
    "wrong dictionary graph interface version",
)
require(
    interop["dictionaryEntriesInterfaceVersion"] == 1,
    "wrong dictionary entries interface version",
)
require(
    interop["snapshotIdentityInterfaceVersion"] == 1,
    "wrong snapshot identity interface version",
)

# Public symbol model, Rust exports, and C declarations must agree exactly.
modeled = {item["name"] for item in MODEL["cFunctions"]}
exported: set[str] = set()
for module in (ROOT / "src" / "ffi").glob("*.rs"):
    exported.update(
        re.findall(
            r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(llev_[a-z0-9_]+)\s*\(',
            module.read_text(encoding="utf-8"),
        )
    )
require(
    exported == modeled,
    f"C symbol model mismatch: missing={sorted(modeled - exported)}, extra={sorted(exported - modeled)}",
)
header = text(ROOT / "include" / "liblevenshtein.h")
for symbol in modeled:
    require(
        re.search(rf"\b{re.escape(symbol)}\s*\(", header) is not None,
        f"public C header is missing {symbol}",
    )
abi_header = text(ROOT / "include" / "liblevenshtein_abi.h")
for marker in (
    "#ifndef VT_INTEROP_HEADER",
    '#define VT_INTEROP_HEADER "vinary_tree_interop.h"',
    "#include VT_INTEROP_HEADER",
):
    require(
        marker in abi_header,
        "liblevenshtein ABI must consume the overridable shared interop header",
    )
interop_header = text(INTEROP_ROOT / "include" / "vinary_tree_interop.h")
for marker in (
    "VT_WFST_INTERFACE_VERSION 1u",
    "VT_RECOMMENDED_ARC_BATCH 256u",
    "VtWeightDomain",
    "VtWfstArc",
    "VtWfstVTable",
    "state_info",
    "state_arcs",
):
    require(marker in interop_header, f"scalar WFST ABI is missing {marker}")

for marker in (
    "VT_DICTIONARY_GRAPH_INTERFACE_VERSION 1u",
    "VtDictionaryGraphNode",
    "VtDictionaryGraphEdge",
    "VtDictionaryGraphView",
    "VtDictionaryGraphVTable",
    "VT_DICTIONARY_GRAPH_INTERFACE_ID",
):
    require(marker in interop_header, f"dictionary graph ABI is missing {marker}")

for marker in (
    "VT_DICTIONARY_ENTRIES_INTERFACE_VERSION 1u",
    "VT_STATUS_BATCH_IN_USE = 9",
    "VT_DICTIONARY_ENTRY_ORDER_LEXICOGRAPHIC = 1",
    "VT_DICTIONARY_ENTRIES_INFO_FLAG_EXACT_LEN UINT64_C(1)",
    "VT_DICTIONARY_ENTRIES_INFO_FLAG_SNAPSHOT_IDENTITY UINT64_C(2)",
    "VtDictionaryEntry",
    "VtDictionaryEntryBatchLimits",
    "VtDictionaryEntryBatchView",
    "VtDictionaryEntriesInfo",
    "VtDictionaryEntriesCursor",
    "VtDictionaryEntriesVTable",
    "VtDictionaryEntryReducer",
    "next_batch",
    "release_batch",
    "reduce",
    "cancel",
    "close",
    "VT_DICTIONARY_ENTRIES_INTERFACE_ID",
):
    require(marker in interop_header, f"dictionary entries ABI is missing {marker}")

entries = interop["dictionaryEntries"]
require(entries["interfaceId"] == "vt.dict.entry.v1", "wrong entries interface ID")
require(
    entries["statusValues"].get("BATCH_IN_USE") == 9,
    "wrong dictionary entries live-batch status",
)
require(
    entries["orderValues"] == {"LEXICOGRAPHIC": 1},
    "wrong dictionary entries ordering contract",
)
require(
    entries["infoFlags"] == {"EXACT_LEN": 1, "SNAPSHOT_IDENTITY": 2},
    "wrong dictionary entries info flags",
)
require(
    entries["vtableOperations"]
    == ["open", "next_batch", "release_batch", "reduce", "cancel", "close"],
    "wrong dictionary entries operation order",
)
require(
    set(entries["layouts"]) == {"lp64", "arm32"},
    "dictionary entries layouts must pin LP64 and ARM32",
)
require(
    MODEL["objects"].get("DictionaryEntriesCursor")
    == {
        "interface": "vt.dict.entry.v1",
        "ownership": "move-only-captured-revision-with-one-live-batch-lease",
        "order": "lexicographic",
        "methods": ["next_batch", "release_batch", "reduce", "cancel", "close"],
    },
    "dictionary entries cursor surface is missing or malformed",
)

for mirror in (
    INTEROP_ROOT / "bindings" / "haskell" / "include" / "vinary_tree_interop.h",
):
    require(
        text(mirror) == interop_header,
        f"standalone interop header mirror is stale: {mirror}",
    )

entries_fixture = text(ROOT / "bindings" / "conformance" / "dictionary_entries_v1.tsv")
require(
    entries_fixture == text(INTEROP_ROOT / "conformance" / "dictionary_entries_v1.tsv"),
    "dictionary entries conformance fixtures differ",
)
for marker in (
    "interface\t-\tVtDictionaryEntriesVTable\tid\tvt.dict.entry.v1",
    "status\t-\tVtStatus\tBATCH_IN_USE\t9",
    "order\t-\tVtDictionaryEntryOrder\tLEXICOGRAPHIC\t1",
    "layout\tlp64\tVtDictionaryEntriesVTable\tsize\t64",
    "layout\tarm32\tVtDictionaryEntriesVTable\tsize\t36",
):
    require(marker in entries_fixture, f"entries fixture is missing {marker}")

# Clean ownership migration: no publishable liblevenshtein facade may construct
# libdictenstein dictionaries or expose old CRUD symbols.
publishable_roots = [
    ROOT / "src" / "bindings.rs",
    ROOT / "include" / "liblevenshtein.h",
    ROOT / "include" / "liblevenshtein.hpp",
    ROOT / "bindings" / "python",
    ROOT / "bindings" / "jvm",
    ROOT / "bindings" / "clojure",
    ROOT / "bindings" / "javascript",
]
publishable_files: list[Path] = []
for root in publishable_roots:
    if root.is_file():
        publishable_files.append(root)
    elif root.is_dir():
        publishable_files.extend(path for path in root.rglob("*") if path.is_file())
for path in publishable_files:
    if PUBLISHABLE_EXCLUDED_PARTS.intersection(path.parts):
        continue
    source = path.read_text(encoding="utf-8", errors="ignore")
    for forbidden in (
        "llev_index_",
        "llev_dat_",
        "llev_scdawg_",
        "llev_persistent_",
        "LlevIndex",
        "StringIndex",
        "PersistentArTrie",
        "DoubleArrayTrieBuilder",
    ):
        require(
            forbidden not in source,
            f"dictionary-owned API {forbidden!r} remains in {path.relative_to(ROOT)}",
        )

# Every Tier 2/3 package is independently publishable and must consume the
# project-owned iterator ABI, never the retired dictionary-owned symbol family.
for language in ("dotnet", "go", "swift", "ruby", "fortran", "ocaml", "haskell", "lua"):
    root = ROOT / "bindings" / language
    require(root.is_dir(), f"missing {language} binding")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".md", ".txt", ""}:
            continue
        if GENERATED_TREE_PARTS.intersection(path.parts):
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        require(
            not re.search(r"\bllev_(?:index|dat|scdawg|persistent)_", source),
            f"retired dictionary ABI remains in {path.relative_to(ROOT)}",
        )

# Repository/package identity guard.
identity_suffixes = {
    ".c",
    ".clj",
    ".cljs",
    ".cs",
    ".d.ts",
    ".edn",
    ".f90",
    ".go",
    ".h",
    ".hpp",
    ".hsc",
    ".java",
    ".json",
    ".kts",
    ".lua",
    ".md",
    ".mjs",
    ".ml",
    ".mli",
    ".opam",
    ".py",
    ".rb",
    ".rockspec",
    ".swift",
    ".toml",
    ".ts",
    ".tsv",
    ".yml",
}
identity_roots = [ROOT / "bindings", INTEROP_ROOT]
for sibling in (LIBDICT_ROOT, LLING_ROOT, DUALLITY_ROOT):
    for relative in ("bindings", "include"):
        candidate = sibling / relative
        if candidate.is_dir():
            identity_roots.append(candidate)
for root in identity_roots:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in identity_suffixes:
            continue
        if GENERATED_TREE_PARTS.intersection(path.parts):
            continue
        source = path.read_text(encoding="utf-8", errors="ignore").lower()
        forbidden_identities = ["f1r3fly", "universal-automata", "universal_automata"]
        if path == ROOT / "bindings" / "jvm" / "jreleaser.yml":
            # This release-only file intentionally owns a minimal relocation
            # POM for the historical pure-Java Maven coordinate. Public Java
            # packages and implementation artifacts must still reject that
            # retired project identity everywhere else.
            forbidden_identities.remove("universal-automata")
        for forbidden in forbidden_identities:
            require(
                forbidden not in source,
                f"unrelated identity {forbidden!r} in {path}",
            )

cargo = tomllib.loads(text(ROOT / "Cargo.toml"))
features = cargo["features"]
for language in (
    *MODEL["supportTiers"]["tier1"],
    *MODEL["supportTiers"]["tier2"],
    *MODEL["supportTiers"]["tier3"],
):
    normalized = {
        "c": "c-bindings",
        "cpp": "cpp-bindings",
        "javascript": "javascript-bindings",
        "typescript": "typescript-bindings",
        "clojurescript": "clojurescript-bindings",
        "jvm": "jvm-bindings",
        "clojure": "clojure-bindings",
    }.get(language, f"{language}-bindings")
    require(normalized in features, f"missing Cargo feature gate {normalized}")

# Snapshot and marshalling laws must be executable, not documentation-only.
snapshot_sources = "\n".join(
    text(ROOT / relative)
    for relative in (
        "tests/query_start_snapshot_semantics.rs",
        "tests/binding_snapshot_semantics.rs",
        "tests/ffi_resource_snapshot_semantics.rs",
    )
)
for marker in (
    "proptest!",
    "next_batch",
    "remove",
    "update",
    "clear",
    "compact",
    "checkpoint",
    "outliving",
    "BatchInUse",
):
    require(marker in snapshot_sources, f"snapshot test coverage is missing {marker}")
bindings_source = text(ROOT / "src" / "bindings.rs")
ffi_source = text(ROOT / "src" / "ffi" / "index.rs")
for marker in (
    "VT_RECOMMENDED_EDGE_BATCH",
    "PARALLEL_REENTRANT",
    "CallGate::Serial",
    "snapshot()?",
    "DEFAULT_MATCH_BATCH",
):
    require(marker in bindings_source, f"marshalling contract is missing {marker}")
for marker in ("byte_arena", "u64_arena", "leased", "llev_query_cursor_reduce"):
    require(marker in ffi_source, f"zero-copy ABI is missing {marker}")

# Tier-1 package coordinates and idiomatic facades.
python = tomllib.loads(text(ROOT / "bindings" / "python" / "pyproject.toml"))
require(python["project"]["name"] == packages["pypi"], "wrong PyPI package")
dotnet_project = text(
    ROOT
    / "bindings"
    / "dotnet"
    / "src"
    / "VinaryTree.Liblevenshtein"
    / "VinaryTree.Liblevenshtein.csproj"
)
require(
    f"<PackageId>{packages['nuget']}</PackageId>" in dotnet_project,
    "wrong NuGet package",
)
swift_package = text(ROOT / "Package.swift")
require(
    f'name: "{packages["swift"]}"' in swift_package,
    "wrong SwiftPM package",
)
ruby_gemspec = text(ROOT / "bindings" / "ruby" / "liblevenshtein.gemspec")
require(
    f'spec.name = "{packages["rubygems"]}"' in ruby_gemspec,
    "wrong RubyGems package",
)
fortran_package = tomllib.loads(text(ROOT / "bindings" / "fortran" / "fpm.toml"))
fortran_publish_package = tomllib.loads(
    text(ROOT / "bindings" / "fortran" / "fpm.publish.toml")
)
require(
    fortran_package["name"] == packages["fpm"]
    and fortran_publish_package["name"] == packages["fpm"],
    "wrong fpm package",
)
require(
    fortran_package.get("build", {}).get("module-naming") is False
    and fortran_publish_package.get("build", {}).get("module-naming") is False,
    "fpm manifests must disable module naming for the namespaced public module",
)
ocaml_project = text(ROOT / "bindings" / "ocaml" / "dune-project")
require(
    f"(name {packages['opam']})" in ocaml_project,
    "wrong opam package",
)
haskell_package = text(ROOT / "bindings" / "haskell" / "liblevenshtein.cabal")
require(
    re.search(
        rf"^name: {re.escape(packages['hackage'])}$",
        haskell_package,
        flags=re.MULTILINE,
    )
    is not None,
    "wrong Hackage package",
)
luarocks_version = RELEASE_MODEL["registries"]["luaRocks"]
luarocks_package = text(
    ROOT / "bindings" / "lua" / f"liblevenshtein-{luarocks_version}.rockspec"
)
require(
    re.search(
        rf'^package = "{re.escape(packages["luarocks"])}"$',
        luarocks_package,
        flags=re.MULTILINE,
    )
    is not None,
    "wrong LuaRocks package",
)
for marker in (
    'LIBLEVENSHTEIN = { header = "liblevenshtein.h", library = "liblevenshtein" }',
    '"$(LIBLEVENSHTEIN_INCDIR)"',
    '"$(LIBLEVENSHTEIN_LIBDIR)"',
):
    require(
        marker in luarocks_package,
        f"LuaRocks external-library contract is missing {marker}",
    )
require(
    '"target/release"' not in luarocks_package,
    "LuaRocks package must not link against a source-checkout target directory",
)
julia_project = tomllib.loads(
    text(ROOT / "bindings" / "julia" / "Liblevenshtein" / "Project.toml")
)
require(julia_project["name"] == packages["julia"], "wrong Julia package")
require(
    julia_project["version"] == RELEASE_MODEL["registries"]["julia"],
    "wrong Julia package version",
)
require(
    julia_project.get("deps", {}).get("VinaryTreeInterop")
    == "8d6503e5-4d65-4bd8-a8ee-293a0149584e"
    and julia_project.get("compat", {}).get("VinaryTreeInterop") == "4",
    "Julia package must depend on the canonical VinaryTreeInterop package",
)
raku_meta = json.loads(text(ROOT / "bindings" / "raku" / "META6.json"))
require(raku_meta["name"] == packages["zef"], "wrong Raku package")
require(
    raku_meta["version"] == RELEASE_MODEL["registries"]["zef"],
    "wrong Raku package version",
)
raku_interop_version = RELEASE_MODEL["dependencies"]["vinary-tree-interop"].replace(
    "-rc.", ".rc."
)
require(
    f"Vinary-Tree-Interop:ver<{raku_interop_version}>:auth<zef:vinary-tree>"
    in raku_meta.get("depends", []),
    "Raku package must depend on the canonical Vinary-Tree-Interop distribution",
)
raku_libdict_version = RELEASE_MODEL["dependencies"]["libdictenstein"].replace(
    "-rc.", ".rc."
)
require(
    f"Libdictenstein:ver<{raku_libdict_version}>:auth<zef:vinary-tree>"
    in raku_meta.get("test-depends", []),
    "Raku tests must pin the coordinated Libdictenstein distribution",
)
raku_abi = text(
    ROOT / "bindings" / "raku" / "lib" / "Liblevenshtein" / "GeneratedAbi.rakumod"
)
raku_facade = text(ROOT / "bindings" / "raku" / "lib" / "Liblevenshtein.rakumod")
generated_raku_exports = set(
    re.findall(
        r"^our (?:constant|enum) ([A-Z][A-Za-z0-9-]*) is export", raku_abi, re.MULTILINE
    )
)
generated_raku_exports.update(
    re.findall(r"^    ([A-Z][A-Z0-9-]*) =>", raku_abi, re.MULTILINE)
)
for name in generated_raku_exports:
    require(
        re.search(
            rf"^our constant {re.escape(name)} is export =",
            raku_facade,
            re.MULTILINE,
        )
        is not None,
        f"Raku facade does not re-export generated ABI name {name}",
    )
python_binding_roots = (
    ROOT / "bindings" / "python",
    INTEROP_ROOT / "bindings" / "python",
    LIBDICT_ROOT / "bindings" / "python",
)
python_manifests = {
    binding_root: tomllib.loads(text(binding_root / "pyproject.toml"))
    for binding_root in python_binding_roots
}
python_interop = python_manifests[INTEROP_ROOT / "bindings" / "python"]
require(
    python_interop["project"]["name"] == "vinary-tree-interop",
    "wrong interop PyPI package",
)
for binding_root, manifest in python_manifests.items():
    setup_path = binding_root / "setup.py"
    if setup_path.is_file():
        require(
            '"LICENSE"' in text(setup_path),
            f"Python native-wheel hook does not stage its license: {setup_path}",
        )
        continue

    license_files = manifest["project"].get("license-files", [])
    require(
        isinstance(license_files, list) and "LICENSE" in license_files,
        f"declarative Python package does not declare LICENSE: {binding_root / 'pyproject.toml'}",
    )
    require(
        (binding_root / "LICENSE").is_file(),
        f"declarative Python package license is missing: {binding_root / 'LICENSE'}",
    )
go_interop = text(INTEROP_ROOT / "bindings" / "go" / "go.mod")
go_project = text(ROOT / "bindings" / "go" / "go.mod")
go_interop_module = "github.com/vinary-tree/vinary-tree-interop/bindings/go/v4"
require(
    f"module {go_interop_module}" in go_interop,
    "shared Go module path must match its versioned repository subdirectory",
)
require(
    f"module {packages['goModule']}" in go_project,
    "project Go module path differs from the canonical package model",
)
require(
    go_interop_module in go_project, "project Go module uses the wrong interop path"
)
require(
    "\nreplace " not in go_project,
    "publishable Go module must not contain local replace directives",
)
jvm_build = text(ROOT / "bindings" / "jvm" / "build.gradle.kts")
require('group = "io.vinarytree"' in jvm_build, "wrong Maven group")
require(
    'artifactId = "liblevenshtein"' in jvm_build
    or 'artifactId.set("liblevenshtein")' in jvm_build,
    "wrong Maven artifact",
)
for build_path in (
    INTEROP_ROOT / "bindings" / "jvm" / "build.gradle.kts",
    ROOT / "bindings" / "jvm" / "build.gradle.kts",
    LIBDICT_ROOT / "bindings" / "jvm" / "build.gradle.kts",
):
    source = text(build_path)
    for marker in (
        'tasks.named<Jar>("sourcesJar")',
        'include("**/*.java")',
        "includeEmptyDirs = false",
        'providers.gradleProperty("stagingRepository")',
    ):
        require(marker in source, f"JVM publication {build_path} is missing {marker}")
native_java = text(
    ROOT
    / "bindings"
    / "jvm"
    / "src"
    / "main"
    / "java"
    / Path(*organization["javaPackage"].split("."))
    / "Native.java"
)
require("java.lang.foreign" in native_java, "JVM bindings must target FFM")
require(
    not re.search(
        r"\bnative\s+(?:void|int|long|boolean|[A-Z][A-Za-z0-9_]*)\s+", native_java
    ),
    "JVM native path must not declare JNI methods",
)
jvm_provider = text(
    INTEROP_ROOT
    / "bindings"
    / "jvm"
    / "src"
    / "main"
    / "java"
    / "io"
    / "vinarytree"
    / "interop"
    / "UnicodeDictionaryResource.java"
)
for marker in ("upcallStub", "PARALLEL_REENTRANT", "snapshot", "DICTIONARY_EDGE"):
    require(marker.lower() in jvm_provider.lower(), f"JVM provider is missing {marker}")
jvm_snapshot_smoke = text(
    ROOT
    / "bindings"
    / "jvm"
    / "src"
    / "smoke"
    / "java"
    / "io"
    / "vinarytree"
    / "liblevenshtein"
    / "ResourceSnapshotSmoke.java"
)
for marker in (
    "first = longLived.next()",
    "current.set(mutated())",
    "long-lived cursor changed after mutation",
):
    require(marker in jvm_snapshot_smoke, f"JVM snapshot smoke is missing {marker}")
clojure_project = text(ROOT / "bindings" / "clojure" / "project.clj")
require(packages["clojars"] in clojure_project, "wrong Clojars coordinate")
for clojure_path in (
    ROOT / "bindings" / "clojure" / "project.clj",
    ROOT / "bindings" / "clojure" / "deps.edn",
    LIBDICT_ROOT / "bindings" / "clojure" / "project.clj",
    LIBDICT_ROOT / "bindings" / "clojure" / "deps.edn",
):
    require(
        "--enable-native-access=ALL-UNNAMED" in text(clojure_path),
        f"Clojure FFM profile lacks native-access authorization: {clojure_path}",
    )
javascript = json.loads(text(ROOT / "bindings" / "javascript" / "package.json"))
require(javascript["name"] == packages["npm"], "wrong npm project package")
javascript_interop = json.loads(
    text(INTEROP_ROOT / "bindings" / "javascript" / "package.json")
)
require(javascript_interop["name"] == interop["npm"], "wrong interop npm package")
require(
    javascript["dependencies"][interop["npm"]] == javascript_interop["version"],
    "project npm facade must pin the interop package exactly",
)
require(
    javascript["dependencies"][MODEL["wasm"]["runtimePackage"]]
    == MODEL["packageVersion"],
    "project npm facade must consume the exact shared JavaScript runtime version",
)
for export in (".", "./typescript", "./clojurescript", "./wasm", "./wasi"):
    require(export in javascript["exports"], f"npm package lacks {export} export")
for facade in ("native.mjs", "wasm.mjs", "wasi.mjs"):
    facade_source = text(ROOT / "bindings" / "javascript" / "facades" / facade)
    require(
        "assertSameRuntime" in facade_source, f"{facade} lacks runtime identity guard"
    )
    require(
        "assertDictionaryResource" in facade_source,
        f"{facade} lacks dictionary interface guard",
    )

# Related projects own their modular packages while sharing one JS/WASM/WASI
# runtime instance. WFST composition must retain inputs in O(1), not import the
# complete component graphs.
related_packages = {
    "lling-llang": (LLING_ROOT, "@vinary-tree/lling-llang", "assertWfstResource"),
    "duallity": (DUALLITY_ROOT, "@vinary-tree/duallity", "assertDictionaryResource"),
}
related_versions = json.loads(text(ROOT / "bindings" / "related-projects.json"))
for project, (project_root, package_name, guard) in related_packages.items():
    package_root = project_root / "bindings" / "javascript"
    package = json.loads(text(package_root / "package.json"))
    require(package["name"] == package_name, f"wrong {project} npm package")
    require(
        package["version"] == related_versions[project]["version"],
        f"wrong {project} npm version",
    )
    require(
        package["dependencies"]["@vinary-tree/javascript-runtime"]
        == MODEL["packageVersion"],
        f"{project} must pin the shared JavaScript runtime exactly",
    )
    for export in (".", "./typescript", "./clojurescript", "./wasm", "./wasi"):
        require(export in package["exports"], f"{project} npm package lacks {export}")
    for facade in ("native.mjs", "wasm.mjs", "wasi.mjs"):
        source = text(package_root / "facades" / facade)
        require(
            "assertSameRuntime" in source, f"{project} {facade} lacks runtime guard"
        )
        require(guard in source, f"{project} {facade} lacks interface guard")

lling_bindings = text(LLING_ROOT / "src" / "bindings.rs")
for marker in (
    "CapturedWfst",
    "CompositionResource::capture",
    "ProviderCallGate::for_flags",
    "composition_construction_retains_inputs_without_expanding_them",
    "assert_eq!(left_expansions.load(Ordering::Relaxed), 0)",
):
    require(marker in lling_bindings, f"lazy retained composition is missing {marker}")
require(
    "import_tropical_wfst(first)?" not in lling_bindings
    and "import_tropical_wfst(second)?" not in lling_bindings,
    "composition must not eagerly import its component graphs",
)
duallity_bindings = text(DUALLITY_ROOT / "src" / "bindings.rs")
for marker in (
    "DictionaryProvider::capture(resource)?",
    "retained_duallity_snapshot_composes_after_all_source_handles_are_dropped",
    "LlingWfstResource::compose",
):
    require(
        marker in duallity_bindings,
        f"duallity retained-resource coverage is missing {marker}",
    )

# Active release metadata must name every implemented registry.
release = text(ROOT / ".github" / "workflows" / "release.yml").lower()
require(
    re.search(r"(?m)^\s*path:\s*['\"]?\.\./", release) is None,
    "release artifact uploads must stage sibling outputs inside the checkout",
)
checkout_siblings = text(
    ROOT / ".github" / "actions" / "checkout-dev-siblings" / "action.yml"
).lower()
require(
    "source-ref-manifest: release/version.json" in release,
    "binding contract must resolve exact sibling sources from the release manifest",
)
for marker in (
    "source-ref-manifest:",
    "scripts/release_source_refs.py",
    '--manifest "$manifest" --component "$repo"',
):
    require(
        marker in checkout_siblings,
        f"sibling checkout action does not fail closed on the source manifest: {marker}",
    )
require(
    "python3 scripts/release_source_refs.py --self-test" in release,
    "release contract does not execute the source-ref hostile-input tests",
)
for marker in (
    "crates-io",
    "pypi",
    "npm",
    "maven-central",
    "maven-relocation-dylon",
    "maven-relocation-universal-automata",
    "clojars",
):
    require(marker in release, f"release workflow does not cover {marker}")
for marker in (
    "bindings/jvm/build/staging-deploy/io/vinarytree/liblevenshtein",
    "name: interop-maven-staging",
    "path: release-artifacts/interop-maven-staging",
    "../vinary-tree-interop/bindings/jvm/build/staging-deploy",
    "for artifact in vinary-tree-interop liblevenshtein; do",
    "bindings/jvm/build/staging-relocations",
    "com-github-dylon",
    "com-github-universal-automata",
    "legacy_group: ${{ inputs.registry == 'maven-relocation-dylon' && 'com.github.dylon' || 'com.github.universal-automata' }}",
    "scripts/stage-maven-relocations.py",
    "deploy --git-root-search --deployer-name canonical",
    "deploy --git-root-search --deployer-name ${{ env.deployer_name }}",
    "jreleaser_deploy_maven_mavencentral_canonical_active: release",
    "jreleaser_deploy_maven_mavencentral_legacydylon_active:",
    "jreleaser_deploy_maven_mavencentral_legacyuniversalautomata_active:",
    "require exact canonical artifact bytes to be public first",
    "bindings/jvm/build/staging-deploy/io/vinarytree/liblevenshtein/$version/liblevenshtein-$version.jar",
):
    require(marker in release, f"Maven publication workflow is missing {marker}")
relocation_stager = text(ROOT / "scripts" / "stage-maven-relocations.py")
jreleaser_configuration = text(ROOT / "bindings" / "jvm" / "jreleaser.yml")
jvm_build = text(ROOT / "bindings" / "jvm" / "build.gradle.kts")
release_model = json.loads(text(ROOT / "release" / "version.json"))
jvm_description = release_model["metadata"]["description"]
require(
    f'description = "{jvm_description}"' in jvm_build,
    "Maven POM description does not explain the product's purpose",
)
require(
    f"description: {jvm_description}" in jreleaser_configuration,
    "JReleaser and Maven POM descriptions have drifted",
)
require(
    f':description "{jvm_description}"'
    in text(ROOT / "bindings" / "clojure" / "project.clj"),
    "Clojars and Maven descriptions have drifted",
)
require(
    not re.search(
        r"^        active: release$", jreleaser_configuration, flags=re.MULTILINE
    ),
    "Maven deployers must remain inactive unless one workflow lane selects them",
)
for marker in (
    "legacyMavenGroups",
    "<distributionManagement>",
    "<relocation>",
    "require_canonical_pom",
    '"--check"',
):
    require(marker in relocation_stager, f"Maven relocation stager is missing {marker}")
# Cargo resolves this repository's development-only path dependencies relative
# to the repository root.  JVM packaging must therefore mirror the documented
# sibling topology instead of nesting the dictionary producer below the root;
# the latter passes source inspection but fails in a clean GitHub runner before
# Gradle ever starts.  Keep Gradle's composite substitution disabled here so
# the consumer test resolves the exact Maven artifact staged by the interop
# producer, not its source checkout.
for marker in (
    "--manifest-path ../libdictenstein/cargo.toml",
    "../vinary-tree-interop/bindings/jvm/gradlew",
    '-pvinarytreeinteroproot="$runner_temp/no-composite-interop"',
):
    require(marker in release, f"JVM release staging is missing {marker}")
for forbidden in (
    ".release-deps/libdictenstein",
    ".release-deps/vinary-tree-interop/bindings/jvm",
    "${github_ref_name#v}",
):
    require(
        forbidden not in release,
        f"release staging contains a forbidden source-tag-derived path/version: {forbidden}",
    )
require(
    "numbered corrective release tag" in release,
    "release workflow does not recognize a provenance-safe corrective source tag",
)
libdictenstein_release = text(
    LIBDICT_ROOT / ".github" / "workflows" / "release-bindings.yml"
).lower()
for release_name, release_source in (
    ("liblevenshtein", release),
    ("libdictenstein", libdictenstein_release),
):
    require(
        "rubygems/configure-rubygems-credentials@dc5a8d8553e6ee01fc26761a49e99e733d17954a"
        in release_source,
        f"{release_name} RubyGems upload does not use the pinned OIDC credential exchange",
    )
    require(
        "rubygems_api_key" not in release_source,
        f"{release_name} RubyGems upload still references a long-lived API key",
    )
    require(
        "clojars_username: ${{ vars.clojars_username }}" in release_source,
        f"{release_name} Clojars upload must consume the organization username variable",
    )
    require(
        "secrets.clojars_username" not in release_source,
        f"{release_name} Clojars upload incorrectly stores the public username as a secret",
    )
    require(
        "secrets.luarocks_api_key" in release_source,
        f"{release_name} LuaRocks upload does not consume its repository environment key",
    )
    require(
        '--temp-key "$luarocks_api_key"' in release_source,
        f"{release_name} LuaRocks upload must use the non-persisting temporary-key mode",
    )
    require(
        "luarocks install dkjson" in release_source,
        f"{release_name} LuaRocks upload does not install its JSON transport dependency",
    )
    for marker in (
        "luarocks --lua-version 5.4 make --tree",
        "luarocks-tree",
        "_incdir=",
        "_libdir=",
        "lua5.4 bindings/lua/",
    ):
        require(
            marker in release_source,
            f"{release_name} LuaRocks lane does not prove installed-package behavior: {marker}",
        )
    require(
        '--api-key "$luarocks_api_key"' not in release_source,
        f"{release_name} LuaRocks upload persists the API key in runner configuration",
    )
    for marker in (
        'fork="vinary-tree/opam-repository"',
        "gh auth setup-git",
        '["registries"]["opam"]',
        '--head "vinary-tree:$branch"',
        "secrets.opam_github_token",
        "environment: github-release",
    ):
        require(
            marker in release_source,
            f"{release_name} opam publisher is missing {marker}",
        )
    for forbidden in (
        "account=$(gh api user",
        "gh repo fork ocaml/opam-repository",
    ):
        require(
            forbidden not in release_source,
            f"{release_name} opam publisher retains dynamic fork logic: {forbidden}",
        )
require(
    "environment: ${{ inputs.registry }}" in release,
    "historical Maven relocation jobs must use their registry-specific protected environment",
)
for marker in (
    "stage-native-package.sh",
    "python-wheels",
    "maven-central",
    "clojars",
    "npm publish",
    "nuget",
    "go-module",
    "swiftpm",
    "rubygems",
    "hackage final-version candidate (never published for rc)",
    "fpm final-version candidate (never published for rc)",
    "opam-repository",
    "luarocks upload",
):
    require(
        marker in libdictenstein_release,
        f"libdictenstein release workflow does not cover {marker}",
    )
for forbidden in ("name: publish hackage", "name: publish fpm"):
    require(
        forbidden not in libdictenstein_release,
        f"libdictenstein RC workflow violates the numeric-only registry embargo: {forbidden}",
    )
interop_release = text(INTEROP_ROOT / ".github" / "workflows" / "release.yml").lower()
for marker in (
    "workflow_dispatch:",
    "cargo publish --locked",
    "gh-action-pypi-publish",
    "npm publish",
    "maven-central",
    "vinarytree.interop",
    "hackage final-version candidate (never published for rc)",
    "fpm final-version candidate (never published for rc)",
    'module_tag="bindings/go/v$version"',
    "opam-repository",
):
    require(marker in interop_release, f"shared interop release is missing {marker}")
for name, workflow in (
    ("liblevenshtein", release),
    ("libdictenstein", libdictenstein_release),
    ("vinary-tree-interop", interop_release),
):
    require(
        "workflow_dispatch:" in workflow,
        f"{name} release workflow must support explicit immutable-tag dispatch",
    )
    require(
        "\n  push:" not in workflow,
        f"{name} release workflow must not auto-run before dependency tags are ready",
    )
for forbidden in (
    "cargo publish --locked -p vinary-tree-interop",
    "pypi-interop:",
    "npm-interop:",
    "dotnet nuget push dist/vinarytree.interop",
):
    require(
        forbidden not in release,
        f"project release must not republish immutable interop artifact {forbidden}",
    )
for marker in ("nuget", "rubygems", "hackage", "fpm", "go-module", "opam", "luarocks"):
    require(marker in release, f"release workflow does not cover {marker}")
for marker in (
    "hackage final-version candidate (never published for rc)",
    "fpm final-version candidate (never published for rc)",
):
    require(marker in release, f"project RC workflow is missing {marker}")
for forbidden in ("name: publish hackage", "name: publish fpm"):
    require(
        forbidden not in release,
        f"project RC workflow violates the numeric-only registry embargo: {forbidden}",
    )
for project, package_name in (
    ("lling-llang", "@vinary-tree/lling-llang"),
    ("duallity", "@vinary-tree/duallity"),
):
    project_root = LLING_ROOT if project == "lling-llang" else DUALLITY_ROOT
    project_release = text(
        project_root / ".github" / "workflows" / "release-bindings.yml"
    ).lower()
    require(
        "workflow_dispatch:" in project_release and "\n  push:" not in project_release,
        f"{project} release workflow must be explicit-dispatch only",
    )
    for marker in ("npm publish", "cargo publish", "stage-native-package.sh"):
        require(
            marker in project_release,
            f"{project} release workflow does not cover {marker}",
        )
ci = text(ROOT / ".github" / "workflows" / "ci.yml").lower()
portable_checkout = text(
    ROOT / ".github" / "actions" / "checkout-portable-siblings" / "action.yml"
).lower()
require(
    "cargo metadata --locked --format-version 1 --no-deps" in portable_checkout,
    "portable sibling checkout must validate the committed dependency resolution",
)
require(
    "cargo generate-lockfile" not in portable_checkout,
    "portable sibling checkout must not replace the committed dependency resolution",
)
for marker in (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "armv7-unknown-linux-gnueabihf",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "freebsd",
    "netbsd",
    "openbsd",
    "dragonfly",
):
    require(marker in ci, f"CI platform matrix is missing {marker}")

# Every maintained facade must run a real producer/consumer handoff in CI.
# These markers name executable tests or package-smoke projects, rather than
# merely checking that a language compiler is installed.
for marker in (
    "bindings/c/tests/cross_project_snapshot.c",
    "../libdictenstein/bindings/cpp/tests/cross_project_snapshot.cpp",
    "bindings/cpp/tests/package-cross",
    "../libdictenstein/bindings/python/tests",
    "../libdictenstein/bindings/jvm",
    "../libdictenstein/bindings/clojure",
    "../libdictenstein/bindings/javascript",
    "../libdictenstein/bindings/dotnet/tests",
    "../libdictenstein/bindings/go",
    "../libdictenstein/bindings/ruby/test/test_cross_project.rb",
    "bindings/fortran/integration",
    "flang-new-22",
    "../libdictenstein/bindings/ocaml",
    "cabal test --project-file=cabal.project",
    "bindings/lua/tests/snapshot.lua",
):
    require(marker in ci, f"CI cross-project conformance is missing {marker}")

runtime_package = json.loads(text(RUNTIME_ROOT / "package.json"))
for artifact in (
    "LICENSE",
    "README.md",
    "native/prebuilds/",
    "generated/wasm/vinary_tree.js",
    "generated/wasm/vinary_tree_bg.wasm",
    "generated/wasi/vinary_tree.wasm",
):
    require(
        artifact in runtime_package["files"],
        f"umbrella npm artifact does not publish {artifact}",
    )
require(
    "--strip-debug" in runtime_package["scripts"]["build:wasi"],
    "published WASI runtime must remove profiling debug sections",
)
native_stager = text(RUNTIME_ROOT / "scripts" / "stage-native-prebuild.mjs")
require(
    'execFileSync("strip"' in native_stager,
    "published Node prebuilds must remove profiling debug symbols",
)
# Panic-free WASM boundary (LLEV-B4): a panic in the umbrella-runtime WASM
# modules traps and kills the whole instance instead of surfacing a status or
# JsError, so non-test code in browser.rs and wasi.rs must never reach a
# panic path. The scanner strips string/char literals and comments, then
# skips `#[cfg(test)]` items by tracking their brace depth, so doc examples
# and test modules stay exempt while any reintroduced production panic site
# fails this gate.
WASM_PANIC_PATTERN = re.compile(
    r"\.unwrap\(\)|\.expect\(|\bunreachable!|\bpanic!|\btodo!|\bunimplemented!"
)


def wasm_panic_sites(path: Path) -> list[str]:
    """Return the non-test panic-capable lines of one WASM boundary module."""
    sites: list[str] = []
    depth = 0
    test_depths: list[int] = []
    cfg_test_pending = False
    for number, raw_line in enumerate(text(path).splitlines(), start=1):
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', raw_line)
        code = re.sub(r"'(?:[^'\\]|\\.)'", "''", code)
        code = code.split("//", 1)[0]
        stripped = code.strip()
        if stripped.startswith("#[cfg(test)]"):
            cfg_test_pending = True
        elif cfg_test_pending and stripped.startswith("#["):
            pass  # further attributes on the same test-only item
        elif cfg_test_pending:
            if "{" in code:
                test_depths.append(depth)
                cfg_test_pending = False
            elif stripped.endswith(";"):
                cfg_test_pending = False  # declaration-only item: `mod tests;`
        elif not test_depths and WASM_PANIC_PATTERN.search(code):
            sites.append(f"{display(path)}:{number}: {raw_line.strip()}")
        depth += code.count("{") - code.count("}")
        if test_depths and depth <= test_depths[-1]:
            test_depths.pop()
    return sites


for wasm_module in (
    RUNTIME_ROOT / "rust" / "src" / "browser.rs",
    RUNTIME_ROOT / "rust" / "src" / "wasi.rs",
):
    wasm_panics = wasm_panic_sites(wasm_module)
    require(
        not wasm_panics,
        "WASM boundary modules must stay panic-free (LLEV-B4); surface "
        "failures as a vt_* status or JsError instead of panicking:\n  "
        + "\n  ".join(wasm_panics),
    )

# Facade-coverage completeness: every modeled C function must map, in every
# maintained facade, either to a real public symbol found in that facade's
# sources or to an explicit reasoned absence. The per-language surface model
# lives in bindings/api-surface-map.json and regenerates the committed matrix
# bindings/conformance/completeness-matrix.tsv, which --check verifies
# byte-for-byte exactly as generate-bindings.py verifies its mirrors.
FACADE_LANGUAGES = (
    "c",
    "clojure",
    "cpp",
    "dotnet",
    "fortran",
    "go",
    "haskell",
    "javascript",
    "javascript-runtime",
    "julia",
    "jvm",
    "lua",
    "ocaml",
    "python",
    "raku",
    "ruby",
    "swift",
)
FACADE_KEYS = {
    "binding",
    "delegateTo",
    "sourceFiles",
    "readme",
    "tests",
    "enums",
    "iterator",
    "reducer",
    "phoneticGating",
    "functions",
}
SYMBOL_KEYS = {"symbol", "_reason", "note"}
ENUM_KEYS = {"symbol", "definedIn", "_reason", "note"}

surface = json.loads(text(ROOT / "bindings" / "api-surface-map.json"))
require(surface.get("modelVersion") == 1, "unknown api-surface-map model version")
facades = surface["languages"]
require(
    sorted(facades) == sorted(FACADE_LANGUAGES),
    "api-surface-map languages mismatch: "
    f"missing={sorted(set(FACADE_LANGUAGES) - set(facades))}, "
    f"extra={sorted(set(facades) - set(FACADE_LANGUAGES))}",
)
function_names = [item["name"] for item in MODEL["cFunctions"]]


def symbol_leaf(symbol: str) -> str:
    parts = [part for part in re.split(r"[\s.:#%$/()]+", symbol) if part]
    require(bool(parts), f"unusable facade symbol {symbol!r}")
    return parts[-1]


def named_in(leaf: str, corpus: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])"
    return re.search(pattern, corpus) is not None


def modeled_symbols(entry: object, context: str) -> list[str]:
    require(isinstance(entry, dict), f"{context} must be an object")
    assert isinstance(
        entry, dict
    )  # require() exits above; this narrows for type checkers
    require(
        not set(entry) - SYMBOL_KEYS,
        f"{context} carries unknown keys {sorted(set(entry) - SYMBOL_KEYS)}",
    )
    require("symbol" in entry, f"{context} is missing its symbol")
    symbol = entry["symbol"]
    if symbol is None:
        reason = entry.get("_reason")
        require(
            isinstance(reason, str) and bool(reason.strip()),
            f"{context} models an absence without a _reason",
        )
        return []
    symbols = symbol if isinstance(symbol, list) else [symbol]
    require(
        bool(symbols)
        and all(isinstance(item, str) and item.strip() for item in symbols),
        f"{context} has an empty symbol",
    )
    return symbols


matrix_rows: list[str] = []
matrix_exposed = 0
matrix_absent = 0
matrix_findings: list[str] = []
absence_report: list[str] = []
for language in FACADE_LANGUAGES:
    facade = facades[language]
    context = f"api-surface-map {language}"
    require(
        not set(facade) - FACADE_KEYS,
        f"{context} carries unknown keys {sorted(set(facade) - FACADE_KEYS)}",
    )
    binding = facade["binding"]
    require(
        binding in {"c-abi", "delegated"},
        f"{context} has unknown binding kind {binding!r}",
    )
    if binding == "delegated":
        require(
            facade.get("delegateTo") in FACADE_LANGUAGES,
            f"{context} delegates to an unmodeled facade",
        )
    require(bool(facade["sourceFiles"]), f"{context} lists no facade sources")
    corpus = "\n".join(
        text(surface_file(relative)) for relative in facade["sourceFiles"]
    )
    readme = facade["readme"]
    if readme is not None:
        require(
            surface_file(readme).is_file(), f"{context} README is missing: {readme}"
        )
    require(bool(facade["tests"]), f"{context} lists no executable tests")
    for relative in facade["tests"]:
        require(
            surface_file(relative).is_file(), f"{context} test is missing: {relative}"
        )

    functions = facade["functions"]
    require(
        sorted(functions) == sorted(function_names),
        f"{context} functions mismatch: "
        f"missing={sorted(set(function_names) - set(functions))}, "
        f"extra={sorted(set(functions) - set(function_names))}",
    )
    exposed = 0
    findings: list[str] = []
    absences: list[str] = []
    for name in function_names:
        symbols = modeled_symbols(functions[name], f"{context} {name}")
        if not symbols:
            absences.append(name)
            if functions[name]["_reason"].startswith("FINDING"):
                findings.append(name)
                matrix_findings.append(f"{language}:{name}")
            continue
        exposed += 1
        if binding == "c-abi":
            require(
                named_in(name, corpus),
                f"{context} claims {name} without binding the C symbol",
            )
        for symbol in symbols:
            require(
                named_in(symbol_leaf(symbol), corpus),
                f"{context} {name} names unlocatable symbol {symbol!r}",
            )

    enums = facade["enums"]
    require(
        sorted(enums) == sorted(MODEL["enums"]),
        f"{context} enums mismatch: "
        f"missing={sorted(set(MODEL['enums']) - set(enums))}, "
        f"extra={sorted(set(enums) - set(MODEL['enums']))}",
    )
    enums_exposed = 0
    for enum_name in sorted(MODEL["enums"]):
        entry = enums[enum_name]
        enum_context = f"{context} enum {enum_name}"
        require(isinstance(entry, dict), f"{enum_context} must be an object")
        require(
            not set(entry) - ENUM_KEYS,
            f"{enum_context} carries unknown keys {sorted(set(entry) - ENUM_KEYS)}",
        )
        require("symbol" in entry, f"{enum_context} is missing its symbol")
        if entry["symbol"] is None:
            reason = entry.get("_reason")
            require(
                isinstance(reason, str) and bool(reason.strip()),
                f"{enum_context} models an absence without a _reason",
            )
            continue
        defined_in = entry.get("definedIn")
        require(
            defined_in in facade["sourceFiles"],
            f"{enum_context} must be defined in a listed facade source",
        )
        require(
            named_in(symbol_leaf(entry["symbol"]), text(surface_file(defined_in))),
            f"{enum_context} names unlocatable symbol {entry['symbol']!r}",
        )
        enums_exposed += 1

    iterator_symbols = modeled_symbols(facade["iterator"], f"{context} iterator")
    require(
        len(iterator_symbols) == 1,
        f"{context} must name exactly one safe-iterator entry point",
    )
    require(
        named_in(symbol_leaf(iterator_symbols[0]), corpus),
        f"{context} iterator {iterator_symbols[0]!r} is not in its sources",
    )
    reducer_symbols = modeled_symbols(facade["reducer"], f"{context} reducer")
    require(
        len(reducer_symbols) <= 1,
        f"{context} must name at most one batch-reducer entry point",
    )
    for symbol in reducer_symbols:
        require(
            named_in(symbol_leaf(symbol), corpus),
            f"{context} reducer {symbol!r} is not in its sources",
        )
    gating = facade["phoneticGating"]
    require(
        isinstance(gating, str) and bool(gating.strip()),
        f"{context} is missing its phonetic gating mode",
    )
    if gating != "status-error":
        require(
            named_in(symbol_leaf(gating), corpus),
            f"{context} phonetic gate {gating!r} is not in its sources",
        )

    matrix_exposed += exposed
    matrix_absent += len(absences)
    absence_report.append(
        f"  {language}: {len(absences)} reasoned absences"
        + (f" ({', '.join(absences)})" if absences else "")
    )
    matrix_rows.append(
        "\t".join(
            (
                language,
                f"{exposed}/{len(function_names)}",
                str(len(absences)),
                f"{enums_exposed}/{len(MODEL['enums'])}",
                iterator_symbols[0],
                reducer_symbols[0] if reducer_symbols else "none",
                gating,
                "yes" if readme is not None else "no",
                "yes" if facade["tests"] else "no",
                ",".join(findings) if findings else "-",
            )
        )
    )

matrix_header = (
    "language\tfns_exposed\tfns_null_with_reason\tenums_ok\titerator\treducer\t"
    "phonetic_gating\treadme_present\ttests_present\tfindings"
)
matrix = "\n".join([matrix_header, *matrix_rows, ""])
matrix_path = ROOT / "bindings" / "conformance" / "completeness-matrix.tsv"
if ARGS.check:
    require(
        matrix_path.is_file() and matrix_path.read_text(encoding="utf-8") == matrix,
        "completeness matrix is stale: rerun scripts/check-bindings.py to "
        "regenerate bindings/conformance/completeness-matrix.tsv",
    )
else:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(matrix, encoding="utf-8")

print("facade completeness matrix:")
print("\n".join(absence_report))
print(
    f"facade completeness: {matrix_exposed}/"
    f"{len(FACADE_LANGUAGES) * len(function_names)} exposures, "
    f"{matrix_absent} reasoned absences, {len(matrix_findings)} findings"
    + (f": {', '.join(matrix_findings)}" if matrix_findings else "")
)
print(
    "binding model, ownership, marshalling, packages, snapshots, and CI are consistent"
)
