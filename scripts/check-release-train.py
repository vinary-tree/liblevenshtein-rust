#!/usr/bin/env python3
"""Reject incoherent Vinary Tree release trains before any package is published."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_MODEL = json.loads((ROOT / "release/version.json").read_text(encoding="utf-8"))
DESCRIPTION_MODEL_PATH = ROOT / "release/package-descriptions.json"
DESCRIPTION_MODEL = json.loads(DESCRIPTION_MODEL_PATH.read_text(encoding="utf-8"))
EXPECTED = str(ROOT_MODEL.get("canonical", ""))
VERSION_MATCH = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-rc\.(\d+)", EXPECTED)
if VERSION_MATCH is None:
    raise SystemExit(
        "release train error: root canonical version is not a numbered release candidate"
    )
MAJOR, MINOR, PATCH, CANDIDATE = VERSION_MATCH.groups()
BASE = f"{MAJOR}.{MINOR}.{PATCH}"


def sibling(environment: str, directory: str) -> Path:
    return Path(os.environ.get(environment, ROOT.parent / directory)).resolve()


COMPONENTS = {
    "liblevenshtein": ROOT,
    "vinary-tree-interop": sibling("VINARY_TREE_INTEROP_ROOT", "vinary-tree-interop"),
    "javascript-runtime": sibling(
        "VINARY_TREE_JAVASCRIPT_RUNTIME_ROOT", "javascript-runtime"
    ),
    "libdictenstein": sibling("LIBDICTENSTEIN_ROOT", "libdictenstein"),
    "lling-llang": sibling("LLING_LLANG_ROOT", "lling-llang"),
    "duallity": sibling("DUALLITY_ROOT", "duallity"),
    "liblevenshtein-npm-compatibility": sibling(
        "LIBLEVENSHTEIN_NPM_ROOT", "liblevenshtein-npm"
    ),
}
LLATTICE_ROOT = sibling("LLATTICE_ROOT", "llattice")

REGISTRY_SPELLINGS = {
    "cargo": EXPECTED,
    "clojars": EXPECTED,
    "cmake": EXPECTED,
    "fpm": BASE,
    "goTag": f"v{EXPECTED}",
    "hackage": BASE,
    "julia": EXPECTED,
    "maven": EXPECTED,
    "npm": EXPECTED,
    "nuget": EXPECTED,
    "opam": f"{BASE}~rc{CANDIDATE}",
    "pkgConfig": EXPECTED,
    "pypi": f"{BASE}rc{CANDIDATE}",
    "rubygems": f"{BASE}.rc.{CANDIDATE}",
    "swiftTag": EXPECTED,
    "zef": EXPECTED,
}

NPM_PACKAGES = {
    "liblevenshtein": (
        "bindings/javascript/package.json",
        "@vinary-tree/liblevenshtein",
    ),
    "vinary-tree-interop": (
        "bindings/javascript/package.json",
        "@vinary-tree/vinary-tree-interop",
    ),
    "javascript-runtime": ("package.json", "@vinary-tree/javascript-runtime"),
    "libdictenstein": (
        "bindings/javascript/package.json",
        "@vinary-tree/libdictenstein",
    ),
    "lling-llang": ("bindings/javascript/package.json", "@vinary-tree/lling-llang"),
    "duallity": ("bindings/javascript/package.json", "@vinary-tree/duallity"),
    "liblevenshtein-npm-compatibility": ("package.json", "liblevenshtein"),
}

CANONICAL_NPM_DEPENDENCIES = {
    "@vinary-tree/vinary-tree-interop",
    "@vinary-tree/javascript-runtime",
    "@vinary-tree/libdictenstein",
    "@vinary-tree/liblevenshtein",
    "@vinary-tree/lling-llang",
    "@vinary-tree/duallity",
}
DEPRECATED_NPM_COORDINATES = {
    "@vinary-tree/" + "interop",
    "@vinary-tree/" + "vinary-tree",
    "@vinary-tree/" + "javascript-runtime-interop",
}
DEPRECATED_NPM_PATTERNS = {
    coordinate: re.compile(
        re.escape(coordinate) + r"(?=$|[^A-Za-z0-9._~-])",
        flags=re.MULTILINE,
    )
    for coordinate in DEPRECATED_NPM_COORDINATES
}


def fail(message: str) -> None:
    print(f"release train error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_descriptions() -> dict[str, dict[str, str]]:
    if DESCRIPTION_MODEL.get("schemaVersion") != 1:
        fail("package description manifest has an unsupported schema")
    maximum = DESCRIPTION_MODEL.get("summaryMaximumCharacters")
    if not isinstance(maximum, int) or maximum < 40:
        fail("package description summary limit must be an integer of at least 40")
    raw = DESCRIPTION_MODEL.get("components")
    if not isinstance(raw, dict):
        fail("package description manifest has no components object")
    expected_components = set(COMPONENTS) | {"liblevenshtein-macros", "llattice"}
    if set(raw) != expected_components:
        missing = sorted(expected_components - set(raw))
        unexpected = sorted(set(raw) - expected_components)
        fail(
            "package description component set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    descriptions: dict[str, dict[str, str]] = {}
    for component, metadata in raw.items():
        if not isinstance(metadata, dict):
            fail(f"{component}: package description entry is not an object")
        summary = metadata.get("summary")
        description = metadata.get("description")
        if not isinstance(summary, str) or not summary.strip():
            fail(f"{component}: canonical package summary is empty")
        if len(summary) > maximum:
            fail(
                f"{component}: canonical package summary has {len(summary)} characters; "
                f"the limit is {maximum}"
            )
        if summary.endswith("."):
            fail(f"{component}: canonical package summary must not end in a period")
        if not isinstance(description, str) or not description.strip():
            fail(f"{component}: canonical package description is empty")
        if not description.endswith("."):
            fail(f"{component}: canonical package description must end in a period")
        descriptions[component] = {
            "summary": summary,
            "description": description,
        }
    return descriptions


def load(component: str, root: Path) -> dict:
    path = root / "release/version.json"
    if not path.is_file():
        fail(f"{component}: missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != 1:
        fail(f"{component}: unsupported release manifest schema")
    if value.get("component") != component:
        fail(f"{component}: component identity is {value.get('component')!r}")
    if value.get("canonical") != EXPECTED:
        fail(f"{component}: canonical version is {value.get('canonical')!r}")
    return value


def check_dependency(owner: str, name: str, version: object) -> None:
    if name in DEPRECATED_NPM_COORDINATES:
        fail(f"{owner}: dependency uses deprecated or malformed npm coordinate {name}")
    if name.startswith("@vinary-tree/") and name not in CANONICAL_NPM_DEPENDENCIES:
        fail(f"{owner}: dependency uses unknown scoped npm coordinate {name}")
    expected = "0.1.0" if name == "llattice" else EXPECTED
    if version != expected:
        fail(f"{owner}: dependency {name} is {version!r}, expected {expected!r}")


def reject_deprecated_coordinates(component: str, root: Path) -> None:
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0:
        detail = tracked.stderr.decode("utf-8", errors="replace").strip()
        fail(f"{component}: cannot enumerate reviewed source files: {detail}")

    for raw_relative in tracked.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = Path(os.fsdecode(raw_relative))
        if (
            relative.name == "CHANGELOG.md"
            or relative.name == "FINDINGS_LEDGER.md"
            or relative.parts[:2] == ("docs", "releases")
            or relative == Path("docs/npm-coordinate-migration.md")
            or relative == Path("docs/releasing.md")
            or relative == Path("docs/releasing-language-bindings.md")
        ):
            continue
        try:
            source = (root / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for token in DEPRECATED_NPM_PATTERNS.values():
            if token.search(source):
                fail(f"{component}: deprecated npm coordinate remains in {relative}")


descriptions = load_descriptions()
manifests = {name: load(name, root) for name, root in COMPONENTS.items()}

for component, manifest in manifests.items():
    if manifest.get("metadata") != descriptions[component]:
        fail(
            f"{component}: release metadata differs from "
            f"{DESCRIPTION_MODEL_PATH.relative_to(ROOT)}"
        )

for component, component_root in COMPONENTS.items():
    reject_deprecated_coordinates(component, component_root)

if len(set(COMPONENTS.values())) != len(COMPONENTS):
    fail("two artifact owners resolve to the same repository root")
for standalone in (
    "vinary-tree-interop",
    "javascript-runtime",
    "liblevenshtein-npm-compatibility",
):
    if COMPONENTS[standalone] == ROOT or ROOT in COMPONENTS[standalone].parents:
        fail(
            f"{standalone}: standalone owner is still nested under liblevenshtein-rust"
        )
for obsolete in (ROOT / "vinary-tree-interop", ROOT / "bindings/javascript-runtime"):
    if obsolete.exists():
        fail(f"embedded artifact owner still exists: {obsolete}")

for component, component_root in COMPONENTS.items():
    python_sync = component_root / "scripts/sync-release-version.py"
    javascript_sync = component_root / "scripts/sync-release-version.mjs"
    if python_sync.is_file():
        command = [sys.executable, str(python_sync)]
    elif javascript_sync.is_file():
        command = ["node", str(javascript_sync)]
    else:
        fail(f"{component}: missing local release-version synchronizer")
    completed = subprocess.run(
        command,
        cwd=component_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        fail(f"{component}: local release-version validation failed: {detail}")

for component, manifest in manifests.items():
    for dependency, version in manifest.get("dependencies", {}).items():
        check_dependency(component, dependency, version)

    publication_model = manifest.get("publication")
    publication = publication_model or {}
    lua_rocks_revision = publication.get("luaRocksRevision", 1)
    if not isinstance(lua_rocks_revision, int) or lua_rocks_revision < 1:
        fail(f"{component}: publication.luaRocksRevision must be a positive integer")

    registries = manifest.get("registries")
    if registries is not None:
        required_package_registries = set()
        component_root = COMPONENTS[component]
        if any(component_root.glob("bindings/julia/*/Project.toml")):
            required_package_registries.add("julia")
        if (component_root / "bindings/raku/META6.json").is_file():
            required_package_registries.add("zef")
        missing_registries = required_package_registries - set(registries)
        if missing_registries:
            fail(
                f"{component}: release model omits supported registries "
                f"{sorted(missing_registries)}"
            )
        for registry, version in registries.items():
            expected = (
                f"{BASE}rc{CANDIDATE}-{lua_rocks_revision}"
                if registry == "luaRocks"
                else REGISTRY_SPELLINGS.get(registry)
            )
            if expected is None:
                fail(f"{component}: unknown registry spelling {registry!r}")
            if version != expected:
                fail(f"{component}: {registry} uses {version!r}, expected {expected!r}")

    if publication_model is not None:
        if publication.get("distTag") != "next":
            fail(f"{component}: npm prereleases must use the next dist-tag")
        for numeric_only in ("hackage", "fpm"):
            if numeric_only not in (registries or {}):
                continue
            if publication.get(numeric_only) is not False:
                fail(f"{component}: {numeric_only} must remain unpublished for the RC")
            if not publication.get(f"{numeric_only}Reason"):
                fail(f"{component}: {numeric_only} embargo requires an explanation")

    package_path, package_name = NPM_PACKAGES[component]
    coordinates = manifest.get("coordinates")
    if not isinstance(coordinates, dict):
        fail(f"{component}: release manifest has no coordinates object")
    if coordinates.get("npmPackage") != package_name:
        fail(
            f"{component}: release npm coordinate is "
            f"{coordinates.get('npmPackage')!r}, expected {package_name!r}"
        )
    package = json.loads(
        (COMPONENTS[component] / package_path).read_text(encoding="utf-8")
    )
    if package.get("name") != package_name:
        fail(f"{component}: npm package name is {package.get('name')!r}")
    if package.get("version") != EXPECTED:
        fail(f"{component}: npm package version is {package.get('version')!r}")
    if package.get("description") != descriptions[component]["description"]:
        fail(f"{component}: npm description differs from canonical metadata")
    publish_config = package.get("publishConfig", {})
    if publish_config.get("access") != "public":
        fail(f"{component}: npm package must publish with public access")
    if publish_config.get("provenance") is not True:
        fail(f"{component}: npm package must request provenance")
    if publish_config.get("tag") != "next":
        fail(f"{component}: npm package must protect latest with tag=next")
    for dependency, version in package.get("dependencies", {}).items():
        if dependency.startswith("@vinary-tree/"):
            check_dependency(component, dependency, version)

runtime = manifests["javascript-runtime"]
if runtime.get("npm") != EXPECTED or runtime.get("distTag") != "next":
    fail("javascript-runtime: npm version/dist-tag does not identify this RC")
expected_prebuilds = {
    "linux-x64",
    "linux-arm64",
    "darwin-x64",
    "darwin-arm64",
    "win32-x64",
    "win32-arm64",
}
if set(runtime.get("nativePrebuilds", [])) != expected_prebuilds:
    fail("javascript-runtime: native prebuild platform set is incomplete")

compatibility = manifests["liblevenshtein-npm-compatibility"]
if compatibility.get("npm") != EXPECTED or compatibility.get("distTag") != "next":
    fail("legacy npm facade: the RC must publish only to next")
legacy_latest = compatibility.get("legacyLatest", {})
if legacy_latest != {"version": "2.0.4", "mustRemainUnchangedDuringRc": True}:
    fail("legacy npm facade: liblevenshtein@latest=2.0.4 protection changed")

for component in ("lling-llang", "duallity"):
    dependencies = manifests[component].get("dependencies", {})
    if dependencies.get("vinary-tree-interop") != EXPECTED:
        fail(f"{component}: standalone interop dependency is not exact")

macro_manifest = (ROOT / "liblevenshtein-macros/Cargo.toml").read_text(encoding="utf-8")
macro_description = re.search(
    r'^description = "([^"]+)"$', macro_manifest, flags=re.MULTILINE
)
if (macro_description.group(1) if macro_description else None) != descriptions[
    "liblevenshtein-macros"
]["description"]:
    fail("liblevenshtein-macros: Cargo description differs from canonical metadata")

llattice_manifest = (LLATTICE_ROOT / "Cargo.toml").read_text(encoding="utf-8")
llattice_description = re.search(
    r'^description = "([^"]+)"$', llattice_manifest, flags=re.MULTILINE
)
if (llattice_description.group(1) if llattice_description else None) != descriptions[
    "llattice"
]["description"]:
    fail("llattice: Cargo description differs from canonical metadata")
llattice_raku = json.loads(
    (LLATTICE_ROOT / "bindings/raku/META6.json").read_text(encoding="utf-8")
)
if llattice_raku.get("description") != descriptions["llattice"]["description"]:
    fail("llattice: Raku description differs from canonical metadata")

print(
    f"release train is coherent: 7 standalone owners, exact {EXPECTED} edges, "
    "all local version surfaces valid, canonical descriptions aligned, npm next, "
    "Hackage/fpm embargoed, legacy latest protected"
)
