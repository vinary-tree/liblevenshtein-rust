#!/usr/bin/env python3
"""Dependency-free architectural and packaging checks for language bindings."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "bindings" / "api.json").read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def text(path: Path) -> str:
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    require(path.is_file(), f"required binding file is missing: {display_path}")
    return path.read_text(encoding="utf-8")


subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "generate-bindings.py"), "--check"],
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
require(
    interop["scalarWfstInterfaceVersion"] == 1, "wrong scalar WFST interface version"
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
        re.search(rf"\b{re.escape(symbol)}\s*\(", header),
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
interop_header = text(
    ROOT / "vinary-tree-interop" / "include" / "vinary_tree_interop.h"
)
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
    if any(
        part
        in {
            ".build",
            ".cpcache",
            ".gradle",
            ".pytest_cache",
            ".swiftpm",
            "__pycache__",
            "bin",
            "build",
            "dist-newstyle",
            "node_modules",
            "obj",
            "target",
            "test",
            "tests",
            "smoke",
        }
        for part in path.parts
    ):
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
        if any(
            part
            in {
                ".build",
                ".cpcache",
                ".gradle",
                ".pytest_cache",
                ".swiftpm",
                "__pycache__",
                "bin",
                "build",
                "dist-newstyle",
                "node_modules",
                "obj",
                "target",
            }
            for part in path.parts
        ):
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
identity_roots = [ROOT / "bindings", ROOT / "vinary-tree-interop"]
for sibling in (
    ROOT.parent / "libdictenstein",
    ROOT.parent / "lling-llang",
    ROOT.parent / "duallity",
):
    for relative in ("bindings", "include"):
        candidate = sibling / relative
        if candidate.is_dir():
            identity_roots.append(candidate)
for root in identity_roots:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in identity_suffixes:
            continue
        if any(
            part
            in {
                ".build",
                "_build",
                ".cpcache",
                ".gradle",
                ".pytest_cache",
                ".swiftpm",
                "__pycache__",
                "bin",
                "build",
                "dist-newstyle",
                "node_modules",
                "obj",
                "target",
            }
            for part in path.parts
        ):
            continue
        source = path.read_text(encoding="utf-8", errors="ignore").lower()
        for forbidden in ("f1r3fly", "universal-automata", "universal_automata"):
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
python_interop = tomllib.loads(
    text(ROOT / "vinary-tree-interop" / "bindings" / "python" / "pyproject.toml")
)
require(
    python_interop["project"]["name"] == "vinary-tree-interop",
    "wrong interop PyPI package",
)
for setup_path in (
    ROOT / "bindings" / "python" / "setup.py",
    ROOT / "vinary-tree-interop" / "bindings" / "python" / "setup.py",
    ROOT.parent / "libdictenstein" / "bindings" / "python" / "setup.py",
):
    require(
        '"LICENSE"' in text(setup_path),
        f"Python wheel does not stage its license: {setup_path}",
    )
go_interop = text(ROOT / "vinary-tree-interop" / "bindings" / "go" / "go.mod")
go_project = text(ROOT / "bindings" / "go" / "go.mod")
go_interop_module = (
    "github.com/vinary-tree/liblevenshtein-rust/vinary-tree-interop/bindings/go"
)
require(
    f"module {go_interop_module}" in go_interop,
    "shared Go module path must match its versioned repository subdirectory",
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
    ROOT / "vinary-tree-interop" / "bindings" / "jvm" / "build.gradle.kts",
    ROOT / "bindings" / "jvm" / "build.gradle.kts",
    ROOT.parent / "libdictenstein" / "bindings" / "jvm" / "build.gradle.kts",
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
    ROOT
    / "vinary-tree-interop"
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
    ROOT.parent / "libdictenstein" / "bindings" / "clojure" / "project.clj",
    ROOT.parent / "libdictenstein" / "bindings" / "clojure" / "deps.edn",
):
    require(
        "--enable-native-access=ALL-UNNAMED" in text(clojure_path),
        f"Clojure FFM profile lacks native-access authorization: {clojure_path}",
    )
javascript = json.loads(text(ROOT / "bindings" / "javascript" / "package.json"))
require(javascript["name"] == packages["npm"], "wrong npm project package")
javascript_interop = json.loads(
    text(ROOT / "vinary-tree-interop" / "bindings" / "javascript" / "package.json")
)
require(javascript_interop["name"] == interop["npm"], "wrong interop npm package")
require(
    javascript["dependencies"][interop["npm"]] == javascript_interop["version"],
    "project npm facade must pin the interop package exactly",
)
require(
    javascript["dependencies"][MODEL["wasm"]["umbrellaPackage"]]
    == MODEL["packageVersion"],
    "project npm facade must consume the exact umbrella runtime version",
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
    "lling-llang": ("@vinary-tree/lling-llang", "0.2.0", "assertWfstResource"),
    "duallity": ("@vinary-tree/duallity", "0.3.0", "assertDictionaryResource"),
}
for project, (package_name, version, guard) in related_packages.items():
    package_root = ROOT.parent / project / "bindings" / "javascript"
    package = json.loads(text(package_root / "package.json"))
    require(package["name"] == package_name, f"wrong {project} npm package")
    require(package["version"] == version, f"wrong {project} npm version")
    require(
        package["dependencies"]["@vinary-tree/vinary-tree"] == MODEL["packageVersion"],
        f"{project} must pin the umbrella runtime exactly",
    )
    for export in (".", "./typescript", "./clojurescript", "./wasm", "./wasi"):
        require(export in package["exports"], f"{project} npm package lacks {export}")
    for facade in ("native.mjs", "wasm.mjs", "wasi.mjs"):
        source = text(package_root / "facades" / facade)
        require(
            "assertSameRuntime" in source, f"{project} {facade} lacks runtime guard"
        )
        require(guard in source, f"{project} {facade} lacks interface guard")

lling_bindings = text(ROOT.parent / "lling-llang" / "src" / "bindings.rs")
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
duallity_bindings = text(ROOT.parent / "duallity" / "src" / "bindings.rs")
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
for marker in ("crates-io", "pypi", "npm", "maven-central", "clojars"):
    require(marker in release, f"release workflow does not cover {marker}")
require(
    "io/github/vinary-tree" not in release,
    "Maven local paths must follow io.vinarytree -> io/vinarytree",
)
libdictenstein_release = text(
    ROOT.parent / "libdictenstein" / ".github" / "workflows" / "release-bindings.yml"
).lower()
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
    "fpm publish",
    "opam-repository",
    "cabal upload",
    "luarocks upload",
):
    require(
        marker in libdictenstein_release,
        f"libdictenstein release workflow does not cover {marker}",
    )
interop_release = text(ROOT / ".github" / "workflows" / "release-interop.yml").lower()
for marker in (
    'tags: ["interop-v*.*.*"]',
    "cargo publish -p vinary-tree-interop",
    "gh-action-pypi-publish",
    "npm publish",
    "maven-central",
    "vinarytree.interop",
    "cabal upload",
    "fpm publish",
    "vinary-tree-interop/bindings/go/v$version",
    "opam-repository",
):
    require(marker in interop_release, f"shared interop release is missing {marker}")
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
for project, package_name in (
    ("lling-llang", "@vinary-tree/lling-llang"),
    ("duallity", "@vinary-tree/duallity"),
):
    project_release = text(
        ROOT.parent / project / ".github" / "workflows" / "release-bindings.yml"
    ).lower()
    for marker in ("npm publish", "cargo publish", "stage-native-package.sh"):
        require(
            marker in project_release,
            f"{project} release workflow does not cover {marker}",
        )
ci = text(ROOT / ".github" / "workflows" / "ci.yml").lower()
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

runtime_package = json.loads(
    text(ROOT / "bindings" / "javascript-runtime" / "package.json")
)
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
native_stager = text(
    ROOT / "bindings" / "javascript-runtime" / "scripts" / "stage-native-prebuild.mjs"
)
require(
    'execFileSync("strip"' in native_stager,
    "published Node prebuilds must remove profiling debug symbols",
)
text(ROOT / "bindings" / "javascript-runtime" / "generated" / "wasm" / ".npmignore")

print(
    "binding model, ownership, marshalling, packages, snapshots, and CI are consistent"
)
