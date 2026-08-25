#!/usr/bin/env python3
"""Validate and resolve the immutable sibling refs in a release-train model."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


def expected_source_versions(model: dict[str, Any]) -> dict[str, str]:
    """Return every standalone sibling owner and its required package version."""

    canonical = model.get("canonical")
    dependencies = model.get("dependencies")
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("release model requires a non-empty canonical version")
    if not isinstance(dependencies, dict):
        raise TypeError("release model requires a dependencies object")

    components = {
        "vinary-tree-interop": dependencies.get("vinary-tree-interop"),
        "javascript-runtime": canonical,
        "liblevenshtein-npm": canonical,
        "llattice": dependencies.get("llattice"),
        "libdictenstein": dependencies.get("libdictenstein"),
        "lling-llang": dependencies.get("lling-llang"),
        "duallity": dependencies.get("duallity"),
    }
    validated: dict[str, str] = {}
    for component, version in components.items():
        if not isinstance(version, str) or not version:
            raise ValueError(f"release model has no non-empty version for {component}")
        validated[component] = version
    return validated


def validate_source_refs(model: dict[str, Any]) -> dict[str, str]:
    """Return the complete source-ref map or reject any mutable/incoherent ref."""

    versions = expected_source_versions(model)
    source_refs = model.get("sourceRefs")
    if not isinstance(source_refs, dict) or set(source_refs) != set(versions):
        raise ValueError("sourceRefs must name every and only standalone sibling owner")

    validated: dict[str, str] = {}
    for component, version in versions.items():
        source_ref = source_refs.get(component)
        pattern = rf"v{re.escape(version)}(?:-release\.[1-9][0-9]*)?"
        if not isinstance(source_ref, str) or re.fullmatch(pattern, source_ref) is None:
            raise ValueError(
                f"sourceRefs.{component} must be an immutable {version} release tag"
            )
        validated[component] = source_ref
    return validated


def self_test() -> None:
    """Prove that incomplete, mutable, and version-incoherent maps fail closed."""

    valid: dict[str, Any] = {
        "canonical": "4.0.0-rc.4",
        "dependencies": {
            "vinary-tree-interop": "4.0.0-rc.4",
            "llattice": "0.1.0",
            "libdictenstein": "4.0.0-rc.4",
            "lling-llang": "4.0.0-rc.4",
            "duallity": "4.0.0-rc.4",
        },
        "sourceRefs": {
            "vinary-tree-interop": "v4.0.0-rc.4-release.2",
            "javascript-runtime": "v4.0.0-rc.4-release.2",
            "liblevenshtein-npm": "v4.0.0-rc.4-release.1",
            "llattice": "v0.1.0",
            "libdictenstein": "v4.0.0-rc.4-release.1",
            "lling-llang": "v4.0.0-rc.4-release.1",
            "duallity": "v4.0.0-rc.4-release.1",
        },
    }
    assert validate_source_refs(valid) == valid["sourceRefs"]

    mutations = (
        lambda model: model["sourceRefs"].pop("duallity"),
        lambda model: model["sourceRefs"].__setitem__("unexpected", "v4.0.0-rc.4"),
        lambda model: model["sourceRefs"].__setitem__("libdictenstein", "master"),
        lambda model: model["sourceRefs"].__setitem__(
            "libdictenstein", "208d9cd6ccfc4993acddd3c166bb314049dfb258"
        ),
        lambda model: model["sourceRefs"].__setitem__("libdictenstein", "v4.0.0-rc.3"),
        lambda model: model["sourceRefs"].__setitem__(
            "libdictenstein", "v4.0.0-rc.4-release.0"
        ),
        lambda model: model["sourceRefs"].__setitem__(
            "libdictenstein", "v4.0.0-rc.4-release.next"
        ),
    )
    for mutate in mutations:
        malformed = copy.deepcopy(valid)
        mutate(malformed)
        try:
            validate_source_refs(malformed)
        except ValueError:
            continue
        raise AssertionError(f"malformed sourceRefs passed validation: {malformed!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/version.json"),
        help="release model containing sourceRefs",
    )
    parser.add_argument("--component", help="print one validated component ref")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("release source-ref hostile-input tests passed")
        return 0

    model = json.loads(args.manifest.read_text(encoding="utf-8"))
    refs = validate_source_refs(model)
    if args.component is None:
        for component, source_ref in refs.items():
            print(f"{component}\t{source_ref}")
        return 0
    if args.component not in refs:
        raise SystemExit(f"release source manifest has no ref for {args.component}")
    print(refs[args.component])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
