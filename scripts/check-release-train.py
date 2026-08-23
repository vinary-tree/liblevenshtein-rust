#!/usr/bin/env python3
"""Reject incoherent Vinary Tree release trains before any package is published."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "4.0.0-rc.1"
BASE = "4.0.0"


def sibling(environment: str, directory: str) -> Path:
    return Path(os.environ.get(environment, ROOT.parent / directory)).resolve()


COMPONENTS = {
    "liblevenshtein": ROOT,
    "vinary-tree-interop": sibling("VINARY_TREE_INTEROP_ROOT", "vinary-tree-interop"),
    "javascript-runtime": sibling("VINARY_TREE_JAVASCRIPT_RUNTIME_ROOT", "javascript-runtime"),
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
    "luaRocks": f"{EXPECTED}-1",
    "maven": EXPECTED,
    "npm": EXPECTED,
    "nuget": EXPECTED,
    "opam": "4.0.0~rc1",
    "pkgConfig": EXPECTED,
    "pypi": "4.0.0rc1",
    "rubygems": "4.0.0.rc.1",
    "swiftTag": EXPECTED,
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

for component, manifest in manifests.items():
    for dependency, version in manifest.get("dependencies", {}).items():
        check_dependency(component, dependency, version)

    registries = manifest.get("registries")
    if registries is not None:
        for registry, version in registries.items():
            expected = REGISTRY_SPELLINGS.get(registry)
            if expected is None:
                fail(f"{component}: unknown registry spelling {registry!r}")
            if version != expected:
                fail(f"{component}: {registry} uses {version!r}, expected {expected!r}")

    publication = manifest.get("publication")
    if publication is not None:
        if publication.get("distTag") != "next":
            fail(f"{component}: npm prereleases must use the next dist-tag")
        for numeric_only in ("hackage", "fpm"):
            if numeric_only not in (registries or {}):
                continue
            if publication.get(numeric_only) is not False:
                fail(f"{component}: {numeric_only} must remain unpublished for the RC")
            if not publication.get(f"{numeric_only}Reason"):
                fail(f"{component}: {numeric_only} embargo requires an explanation")

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

if not re.fullmatch(r"\d+\.\d+\.\d+-rc\.\d+", EXPECTED):
    fail("the release-train checker itself contains an invalid RC")

print(
    "release train is coherent: 7 owners, exact 4.0.0-rc.1 edges, "
    "npm next, Hackage/fpm embargoed, legacy latest protected"
)
