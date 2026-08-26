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

REGISTRY_SPELLINGS = {
    "cargo": EXPECTED,
    "clojars": EXPECTED,
    "cmake": EXPECTED,
    "fpm": BASE,
    "goTag": f"v{EXPECTED}",
    "hackage": BASE,
    "maven": EXPECTED,
    "npm": EXPECTED,
    "nuget": EXPECTED,
    "opam": f"{BASE}~rc{CANDIDATE}",
    "pkgConfig": EXPECTED,
    "pypi": f"{BASE}rc{CANDIDATE}",
    "rubygems": f"{BASE}.rc.{CANDIDATE}",
    "swiftTag": EXPECTED,
}

NPM_PACKAGES = {
    "liblevenshtein": (
        "bindings/javascript/package.json",
        "@vinary-tree/liblevenshtein",
    ),
    "vinary-tree-interop": ("bindings/javascript/package.json", "@vinary-tree/interop"),
    "javascript-runtime": ("package.json", "@vinary-tree/vinary-tree"),
    "libdictenstein": (
        "bindings/javascript/package.json",
        "@vinary-tree/libdictenstein",
    ),
    "lling-llang": ("bindings/javascript/package.json", "@vinary-tree/lling-llang"),
    "duallity": ("bindings/javascript/package.json", "@vinary-tree/duallity"),
    "liblevenshtein-npm-compatibility": ("package.json", "liblevenshtein"),
}


def fail(message: str) -> None:
    print(f"release train error: {message}", file=sys.stderr)
    raise SystemExit(1)


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
    expected = "0.1.0" if name == "llattice" else EXPECTED
    if version != expected:
        fail(f"{owner}: dependency {name} is {version!r}, expected {expected!r}")


manifests = {name: load(name, root) for name, root in COMPONENTS.items()}

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
    package = json.loads(
        (COMPONENTS[component] / package_path).read_text(encoding="utf-8")
    )
    if package.get("name") != package_name:
        fail(f"{component}: npm package name is {package.get('name')!r}")
    if package.get("version") != EXPECTED:
        fail(f"{component}: npm package version is {package.get('version')!r}")
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

print(
    f"release train is coherent: 7 standalone owners, exact {EXPECTED} edges, "
    "all local version surfaces valid, npm next, Hackage/fpm embargoed, "
    "legacy latest protected"
)
