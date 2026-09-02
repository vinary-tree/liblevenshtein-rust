#!/usr/bin/env python3
"""Write or validate every liblevenshtein 4.x release coordinate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from release_source_refs import validate_source_refs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "release/version.json"
GENERATED_TREE_PARTS = frozenset(
    {".git", ".venv", "_build", "build", "dist", "node_modules", "target", "venv"}
)


def derived(canonical: str, lua_rocks_revision: int = 1) -> dict[str, str]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-rc\.(\d+)", canonical)
    if match is None:
        raise ValueError(f"canonical version is not a numbered RC: {canonical}")
    major, minor, patch, candidate = match.groups()
    base = f"{major}.{minor}.{patch}"
    return {
        "cargo": canonical,
        "clojars": canonical,
        "cmake": canonical,
        "fpm": base,
        "goTag": f"v{canonical}",
        "hackage": base,
        "julia": canonical,
        "luaRocks": f"{base}rc{candidate}-{lua_rocks_revision}",
        "maven": canonical,
        "npm": canonical,
        "nuget": canonical,
        "opam": f"{base}~rc{candidate}",
        "pkgConfig": canonical,
        "pypi": f"{base}rc{candidate}",
        "rubygems": f"{base}.rc.{candidate}",
        "swiftTag": canonical,
        "zef": canonical,
    }


def maven_coordinates(model: dict[str, object]) -> dict[str, object]:
    coordinates = model.get("coordinates")
    if not isinstance(coordinates, dict):
        raise TypeError("release/version.json requires coordinates")
    expected_strings = (
        "npmPackage",
        "mavenGroup",
        "mavenArtifact",
        "interopMavenGroup",
        "interopMavenArtifact",
        "javaPackage",
    )
    for field in expected_strings:
        if not isinstance(coordinates.get(field), str):
            raise TypeError(f"release/version.json requires string coordinates.{field}")
    legacy = coordinates.get("legacyMavenGroups")
    if not isinstance(legacy, list) or not all(
        isinstance(group, str) for group in legacy
    ):
        raise TypeError(
            "release/version.json requires string array coordinates.legacyMavenGroups"
        )
    return coordinates


def release_description(model: dict[str, object]) -> str:
    metadata = model.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("description"), str
    ):
        raise TypeError("release/version.json requires string metadata.description")
    description = str(metadata["description"]).strip()
    if not description:
        raise ValueError("release/version.json metadata.description cannot be empty")
    return description


def release_summary(model: dict[str, object]) -> str:
    metadata = model.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("summary"), str):
        raise TypeError("release/version.json requires string metadata.summary")
    summary = str(metadata["summary"]).strip()
    if not summary:
        raise ValueError("release/version.json metadata.summary cannot be empty")
    if len(summary) > 80:
        raise ValueError("release/version.json metadata.summary must fit 80 characters")
    if summary.endswith("."):
        raise ValueError(
            "release/version.json metadata.summary must not end with a period"
        )
    return summary


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def replace(path: str, pattern: str, replacement: str, expected: int = 1) -> None:
    target = ROOT / path
    original = target.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, original, flags=re.MULTILINE)
    if count != expected:
        raise ValueError(
            f"{path}: expected {expected} matches for {pattern!r}, found {count}"
        )
    target.write_text(updated, encoding="utf-8")


def rewrite_cargo_lock(expected: dict[str, str], path: str = "Cargo.lock") -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    for package, version in expected.items():
        pattern = rf'(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = ")[^"]+'
        source, count = re.subn(pattern, rf"\g<1>{version}", source)
        if count != 1:
            raise ValueError(
                f"{path}: expected one {package} package entry, found {count}"
            )
    target.write_text(source, encoding="utf-8")


def rewrite_uv_lock(expected: dict[str, str]) -> None:
    """Keep the checked-in Python resolver state on the same release train."""

    target = ROOT / "bindings/python/uv.lock"
    source = target.read_text(encoding="utf-8")
    for package, version in expected.items():
        pattern = rf'(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = ")[^"]+'
        source, count = re.subn(pattern, rf"\g<1>{version}", source)
        if count != 1:
            raise ValueError(
                "bindings/python/uv.lock: expected one "
                f"{package} package entry, found {count}"
            )
    target.write_text(source, encoding="utf-8")


def rewrite_julia_docs_manifest(expected: dict[str, str]) -> None:
    """Synchronize path-developed family packages in the locked docs environment."""

    target = ROOT / "bindings/julia/Liblevenshtein/docs/Manifest.toml"
    source = target.read_text(encoding="utf-8")
    for package, version in expected.items():
        pattern = (
            rf"(\[\[deps\.{re.escape(package)}\]\]"
            rf'(?:(?!\n\[\[deps\.).)*?\nversion = ")[^"]+'
        )
        source, count = re.subn(
            pattern,
            rf"\g<1>{version}",
            source,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError(
                "bindings/julia/Liblevenshtein/docs/Manifest.toml: expected one "
                f"{package} dependency entry, found {count}"
            )
    target.write_text(source, encoding="utf-8")


def cargo_lock_versions(
    expected: dict[str, str], path: str = "Cargo.lock"
) -> dict[str, str | None]:
    source = read(path)
    return {
        package: (
            match.group(1)
            if (
                match := re.search(
                    rf'\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"',
                    source,
                )
            )
            else None
        )
        for package in expected
    }


def rewrite_candidate_tokens(
    patterns: tuple[str, ...],
    canonical: str,
    lua_rocks_revision: int,
    excluded: tuple[str, ...] = (),
) -> None:
    base, candidate = canonical.split("-rc.", 1)
    escaped = re.escape(base)
    replacements = (
        (rf"{escaped}\.rc\.\d+", f"{base}.rc.{candidate}"),
        (rf"{escaped}~rc\d+", f"{base}~rc{candidate}"),
        (rf"{escaped}\\textasciitilde rc\d+", rf"{base}\textasciitilde rc{candidate}"),
        (
            rf"{escaped}rc\d+-\d+",
            f"{base}rc{candidate}-{lua_rocks_revision}",
        ),
        (rf"{escaped}rc\d+", f"{base}rc{candidate}"),
        (rf"{escaped}-rc\.\d+", canonical),
        (r"RELEASE_4_RC_\d+", f"RELEASE_4_RC_{candidate}"),
        (r"x-release-candidate: rc\.\d+", f"x-release-candidate: rc.{candidate}"),
        (r"(?<=`\$`r = )\d+(?=`\$`)", candidate),
    )
    excluded_paths = {ROOT / path for path in excluded}
    for pattern in patterns:
        for target in ROOT.glob(pattern):
            relative = target.relative_to(ROOT)
            if (
                target in excluded_paths
                or not target.is_file()
                or GENERATED_TREE_PARTS.intersection(relative.parts)
                or relative.parts[:2] == ("docs", "releases")
            ):
                continue
            source = target.read_text(encoding="utf-8")
            for version_pattern, replacement in replacements:
                source = re.sub(
                    version_pattern,
                    lambda _match, value=replacement: value,
                    source,
                )
            target.write_text(source, encoding="utf-8")


def update_json(path: str, mutate) -> None:
    target = ROOT / path
    value = json.loads(target.read_text(encoding="utf-8"))
    mutate(value)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_versions(model: dict[str, object], versions: dict[str, str]) -> None:
    canonical = str(model["canonical"])
    candidate = canonical.rsplit(".", 1)[-1]
    dependencies = model["dependencies"]
    assert isinstance(dependencies, dict)
    coordinates = maven_coordinates(model)
    maven_group = str(coordinates["mavenGroup"])
    maven_artifact = str(coordinates["mavenArtifact"])
    interop_group = str(coordinates["interopMavenGroup"])
    interop_artifact = str(coordinates["interopMavenArtifact"])
    npm_package = str(coordinates["npmPackage"])
    description = release_description(model)
    summary = release_summary(model)

    replace("Cargo.toml", r'^version = "[^"]+"$', f'version = "{canonical}"')
    replace(
        "Cargo.toml",
        r'^description = "[^"]+"$',
        f'description = "{description}"',
    )
    replace(
        "Cargo.toml",
        r"^vinary-tree-interop = \{[^\n]+\}$",
        f'vinary-tree-interop = {{ path = "../vinary-tree-interop", version = "={dependencies["vinary-tree-interop"]}", optional = true }}',
    )
    replace(
        "Cargo.toml",
        r"^libdictenstein = \{[^\n]+\}$",
        f'libdictenstein = {{ path = "../libdictenstein", version = "={dependencies["libdictenstein"]}", features = ["parking_lot"] }}',
    )
    rewrite_cargo_lock(
        {
            "liblevenshtein": canonical,
            "libdictenstein": str(dependencies["libdictenstein"]),
            "vinary-tree-interop": str(dependencies["vinary-tree-interop"]),
        }
    )
    rewrite_cargo_lock(
        {
            "liblevenshtein": canonical,
            "libdictenstein": str(dependencies["libdictenstein"]),
        },
        "liblevenshtein-macros/Cargo.lock",
    )

    def api(value: dict) -> None:
        value["packageVersion"] = canonical
        value["release"] = {
            "canonical": canonical,
            "registries": versions,
            "distTag": model["publication"]["distTag"],
        }

    update_json("bindings/api.json", api)

    package_documentation_versions = {
        "rust": versions["cargo"],
        "native-c-cpp": canonical,
        "python": versions["pypi"],
        "jvm-java": versions["maven"],
        "jvm-kotlin-scala": versions["maven"],
        "clojure": versions["clojars"],
        "javascript": versions["npm"],
        "dotnet": versions["nuget"],
        "go": versions["goTag"],
        "swift": versions["swiftTag"],
        "ruby": versions["rubygems"],
        "lua": versions["luaRocks"],
        "ocaml": versions["opam"],
        "haskell": versions["hackage"],
        "fortran": versions["fpm"],
        "julia": versions["julia"],
        "raku": versions["zef"],
    }

    def package_documentation(value: dict) -> None:
        value["canonicalVersion"] = canonical
        value["sourceRef"] = model["publication"]["sourceTag"]
        packages = value.get("packages", [])
        if not isinstance(packages, list):
            raise TypeError(
                "release/package-documentation.json packages must be an array"
            )
        seen: set[str] = set()
        for package in packages:
            if not isinstance(package, dict) or not isinstance(package.get("id"), str):
                raise TypeError(
                    "release/package-documentation.json packages require string ids"
                )
            identifier = package["id"]
            if identifier not in package_documentation_versions:
                raise ValueError(f"unknown package-documentation id: {identifier}")
            seen.add(identifier)
            wanted = package_documentation_versions[identifier]
            if package.get("registryVersion") != wanted:
                package["registryVersion"] = wanted
                package["releaseState"] = "candidate-only"
                package["releaseProof"] = (
                    "The RC6 source and documentation are validated locally; the user "
                    "explicitly prohibited publishing this candidate until the coordinated "
                    "feature branches are reviewed."
                )
                package.pop("registryReadback", None)
                evidence = package.get("sourceEvidence", [])
                if not isinstance(evidence, list):
                    raise TypeError(
                        f"release/package-documentation.json {identifier} sourceEvidence "
                        "must be an array"
                    )
                destinations = package.get("destinations", [])
                if not isinstance(destinations, list):
                    raise TypeError(
                        f"release/package-documentation.json {identifier} destinations "
                        "must be an array"
                    )
                for destination in destinations:
                    if not isinstance(destination, dict):
                        raise TypeError(
                            f"release/package-documentation.json {identifier} destination "
                            "must be an object"
                        )
                    if destination.get("state") == "verified":
                        destination["state"] = "build-only"
                        destination["reason"] = (
                            "This RC6 documentation surface is validated from source; public "
                            "deployment is intentionally deferred until release approval."
                        )
                        destination["buildEvidence"] = evidence
                        for field in (
                            "url",
                            "readbackUrl",
                            "verifiedAt",
                            "markers",
                        ):
                            destination.pop(field, None)
        missing = set(package_documentation_versions) - seen
        if missing:
            raise ValueError(
                "release/package-documentation.json lacks package ids: "
                + ", ".join(sorted(missing))
            )

    update_json("release/package-documentation.json", package_documentation)

    def related(value: dict) -> None:
        value.clear()
        for name in ("llattice", "libdictenstein", "lling-llang", "duallity"):
            version = str(dependencies[name])
            value[name] = {"version": version, "ref": f"v{version}"}

    update_json("bindings/related-projects.json", related)

    def npm(value: dict) -> None:
        value["name"] = npm_package
        value["version"] = versions["npm"]
        value["description"] = description
        value["dependencies"]["@vinary-tree/vinary-tree-interop"] = dependencies[
            "@vinary-tree/vinary-tree-interop"
        ]
        value["dependencies"]["@vinary-tree/javascript-runtime"] = dependencies[
            "@vinary-tree/javascript-runtime"
        ]
        value.setdefault("publishConfig", {})["tag"] = model["publication"]["distTag"]

    update_json("bindings/javascript/package.json", npm)
    replace(
        "bindings/javascript/deps.cljs",
        r'"@vinary-tree/liblevenshtein" "[^"]+"',
        f'"@vinary-tree/liblevenshtein" "{versions["npm"]}"',
    )

    def npm_lock(value: dict) -> None:
        value["version"] = versions["npm"]
        for package in value.get("packages", {}).values():
            name = package.get("name")
            if name == "@vinary-tree/liblevenshtein":
                package["version"] = versions["npm"]
            elif name in {
                "@vinary-tree/vinary-tree-interop",
                "@vinary-tree/libdictenstein",
                "@vinary-tree/javascript-runtime",
            }:
                package["version"] = canonical
            package_dependencies = package.get("dependencies", {})
            if "@vinary-tree/vinary-tree-interop" in package_dependencies:
                package_dependencies["@vinary-tree/vinary-tree-interop"] = dependencies[
                    "@vinary-tree/vinary-tree-interop"
                ]
            if "@vinary-tree/javascript-runtime" in package_dependencies:
                package_dependencies["@vinary-tree/javascript-runtime"] = dependencies[
                    "@vinary-tree/javascript-runtime"
                ]

    update_json("bindings/javascript/package-lock.json", npm_lock)
    replace(
        "bindings/javascript/build.mjs",
        r'assert\.equal\(packageJson\.dependencies\["@vinary-tree/vinary-tree-interop"\], "[^"]+"\);',
        f'assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree-interop"], "{dependencies["@vinary-tree/vinary-tree-interop"]}");',
    )
    replace(
        "bindings/javascript/test/facades.test.mjs",
        r'assert\.equal\(packageJson\.dependencies\["@vinary-tree/vinary-tree-interop"\], "[^"]+"\);',
        f'assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree-interop"], "{dependencies["@vinary-tree/vinary-tree-interop"]}");',
    )

    replace(
        "bindings/python/pyproject.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["pypi"]}"',
    )
    replace(
        "bindings/python/pyproject.toml",
        r'^description = "[^"]+"$',
        f'description = "{description}"',
    )
    replace(
        "bindings/python/pyproject.toml",
        r'vinary-tree-interop==[^"]+',
        "vinary-tree-interop=="
        + derived(str(dependencies["vinary-tree-interop"]))["pypi"],
    )
    rewrite_uv_lock(
        {
            "liblevenshtein": versions["pypi"],
            "vinary-tree-interop": derived(str(dependencies["vinary-tree-interop"]))[
                "pypi"
            ],
        }
    )
    replace(
        "bindings/julia/Liblevenshtein/Project.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["julia"]}"',
    )
    julia_major = canonical.split(".", 1)[0]
    interop_julia_major = str(dependencies["vinary-tree-interop"]).split(".", 1)[0]
    replace(
        "bindings/julia/Liblevenshtein/Project.toml",
        r'^VinaryTreeInterop = "\d+"$',
        f'VinaryTreeInterop = "{interop_julia_major}"',
    )
    replace(
        "bindings/julia/Liblevenshtein/docs/Project.toml",
        r'^Liblevenshtein = "\d+"$',
        f'Liblevenshtein = "{julia_major}"',
    )
    replace(
        "bindings/julia/Liblevenshtein/docs/Project.toml",
        r'^VinaryTreeInterop = "\d+"$',
        f'VinaryTreeInterop = "{interop_julia_major}"',
    )
    rewrite_julia_docs_manifest(
        {
            "Liblevenshtein": versions["julia"],
            "VinaryTreeInterop": str(dependencies["vinary-tree-interop"]),
        }
    )

    raku_dependency_version = canonical.replace("-rc.", ".rc.")

    def raku(value: dict) -> None:
        value["version"] = versions["zef"]
        value["description"] = description
        for field in ("depends", "test-depends"):
            dependencies_for_field = value.get(field, [])
            if not isinstance(dependencies_for_field, list):
                raise TypeError(f"bindings/raku/META6.json {field} must be an array")
            value[field] = [
                re.sub(
                    r":ver<[^>]+>",
                    f":ver<{raku_dependency_version}>",
                    dependency,
                )
                if isinstance(dependency, str)
                else dependency
                for dependency in dependencies_for_field
            ]

    update_json("bindings/raku/META6.json", raku)
    replace(
        "bindings/jvm/build.gradle.kts",
        r'^version = "[^"]+"$',
        f'version = "{versions["maven"]}"',
    )
    replace(
        "bindings/jvm/build.gradle.kts",
        r'^                description = "[^"]+"$',
        f'                description = "{description}"',
    )
    replace(
        "bindings/jvm/build.gradle.kts",
        r'^group = "[^"]+"$',
        f'group = "{maven_group}"',
    )
    replace(
        "bindings/jvm/build.gradle.kts",
        r'api\("[^":]+:[^":]+:[^"]+"\)',
        f'api("{interop_group}:{interop_artifact}:{dependencies["vinary-tree-interop"]}")',
    )
    replace(
        "bindings/jvm/settings.gradle.kts",
        r'substitute\(module\("[^":]+:[^"]+"\)\)',
        f'substitute(module("{interop_group}:{interop_artifact}"))',
    )
    replace(
        "bindings/jvm/jreleaser.yml",
        r"^  version: \S+$",
        f"  version: {versions['maven']}",
    )
    replace(
        "bindings/jvm/jreleaser.yml",
        r"^  description: .+$",
        f"  description: {description}",
    )
    replace(
        "bindings/jvm/jreleaser.yml",
        r"^      groupId: \S+$",
        f"      groupId: {maven_group}",
    )
    replace(
        "bindings/jvm/jreleaser.yml",
        r"^      artifactId: \S+$",
        f"      artifactId: {maven_artifact}",
    )
    replace(
        "bindings/jvm/deps.edn",
        r":deps \{[^\s]+/vinary-tree-interop",
        f":deps {{{interop_group}/{interop_artifact}",
    )
    replace(
        "bindings/clojure/project.clj",
        r'^(\(defproject io\.vinarytree/liblevenshtein-clojure) "[^"]+"$',
        rf'\1 "{versions["clojars"]}"',
    )
    replace(
        "bindings/clojure/project.clj",
        r'^  :description "[^"]+"$',
        f'  :description "{description}"',
    )
    replace(
        "bindings/clojure/project.clj",
        r'\[[^\s]+/vinary-tree-interop "[^"]+"\]',
        f'[{interop_group}/{interop_artifact} "{dependencies["vinary-tree-interop"]}"]',
    )
    replace(
        "bindings/clojure/project.clj",
        r'\[[^\s]+/liblevenshtein "[^"]+"\]',
        f'[{maven_group}/{maven_artifact} "{versions["maven"]}"]',
    )
    replace(
        "bindings/dotnet/src/VinaryTree.Liblevenshtein/VinaryTree.Liblevenshtein.csproj",
        r"^    <Version>[^<]+</Version>$",
        f"    <Version>{versions['nuget']}</Version>",
    )
    replace(
        "bindings/dotnet/src/VinaryTree.Liblevenshtein/VinaryTree.Liblevenshtein.csproj",
        r"^    <Description>[^<]+</Description>$",
        f"    <Description>{description}</Description>",
    )
    replace(
        "bindings/dotnet/src/VinaryTree.Liblevenshtein/VinaryTree.Liblevenshtein.csproj",
        r'<PackageReference Include="VinaryTree\.Interop" Version="[^"]+" />',
        f'<PackageReference Include="VinaryTree.Interop" Version="{versions["nuget"]}" />',
    )
    replace(
        "bindings/ruby/lib/vinary_tree/liblevenshtein/version.rb",
        r'^    VERSION = "[^"]+"$',
        f'    VERSION = "{versions["rubygems"]}"',
    )
    replace(
        "bindings/ruby/liblevenshtein.gemspec",
        r'^  spec\.summary = "[^"]+"$',
        f'  spec.summary = "{summary}"',
    )
    replace(
        "bindings/ruby/liblevenshtein.gemspec",
        r'^  spec\.description = "[^"]+"$',
        f'  spec.description = "{description}"',
    )
    replace(
        "bindings/fortran/fpm.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["fpm"]}"',
    )
    replace(
        "bindings/fortran/fpm.toml",
        r'^description = "[^"]+"$',
        f'description = "{description}"',
    )
    replace(
        "bindings/fortran/fpm.publish.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["fpm"]}"',
    )
    replace(
        "bindings/fortran/fpm.publish.toml",
        r'^description = "[^"]+"$',
        f'description = "{description}"',
    )
    replace(
        "bindings/fortran/fpm.publish.toml",
        r'^v = "[^"]+"$',
        f'v = "{versions["fpm"]}"',
    )
    replace(
        "bindings/go/go.mod",
        r"^module \S+$",
        "module github.com/vinary-tree/liblevenshtein-rust/bindings/go/v4",
    )
    replace(
        "bindings/go/go.mod",
        r"github\.com/vinary-tree/(?:liblevenshtein-rust/vinary-tree-interop|vinary-tree-interop)/bindings/go(?:/v4)? v\S+",
        f"github.com/vinary-tree/vinary-tree-interop/bindings/go/v4 {versions['goTag']}",
    )
    for path in ("bindings/go/liblevenshtein.go",):
        replace(
            path,
            r"github\.com/vinary-tree/(?:liblevenshtein-rust/vinary-tree-interop|vinary-tree-interop)/bindings/go(?:/v4)?",
            "github.com/vinary-tree/vinary-tree-interop/bindings/go/v4",
        )
    for path in ("bindings/go/leak_test.go", "bindings/go/property_test.go"):
        replace(
            path,
            r"github\.com/vinary-tree/libdictenstein/bindings/go(?:/v4)?",
            "github.com/vinary-tree/libdictenstein/bindings/go/v4",
        )

    for path in (
        "bindings/ocaml/liblevenshtein.opam",
        "bindings/ocaml/liblevenshtein.opam.template",
    ):
        replace(path, r'^synopsis: "[^"]+"$', f'synopsis: "{summary}"')
        replace(path, r'^description: "[^"]+"$', f'description: "{description}"')
        replace(
            path,
            r'"vinary-tree-interop" \{[^}]+\}',
            f'"vinary-tree-interop" {{= "{versions["opam"]}"}}',
        )
        replace(
            path,
            r'"libdictenstein" \{with-test & [^}]+\}',
            f'"libdictenstein" {{with-test & = "{versions["opam"]}"}}',
        )
        replace(
            path,
            r'\["pkg-config" "--atleast-version=[^"]+" "liblevenshtein"\]',
            f'["pkg-config" "--atleast-version={versions["pkgConfig"]}" "liblevenshtein"]',
        )
    replace(
        "bindings/ocaml/dune-project",
        r'^ \(synopsis "[^"]+"\)$',
        f' (synopsis "{summary}")',
    )
    replace(
        "bindings/haskell/liblevenshtein.cabal",
        r"^version: \S+$",
        f"version: {versions['hackage']}",
    )
    replace(
        "bindings/haskell/liblevenshtein.cabal",
        r"^synopsis: .+$",
        f"synopsis: {summary}",
    )
    replace(
        "bindings/haskell/liblevenshtein.cabal",
        r"^description: .+$",
        f"description: {description}",
    )
    cabal_path = ROOT / "bindings/haskell/liblevenshtein.cabal"
    cabal = cabal_path.read_text(encoding="utf-8")
    if not re.search(r"^x-release-candidate:", cabal, re.MULTILINE):
        cabal = cabal.replace(
            f"version: {versions['hackage']}\n",
            f"version: {versions['hackage']}\nx-release-candidate: rc.1\n",
            1,
        )
    cabal = re.sub(
        r"^x-release-candidate: \S+$",
        f"x-release-candidate: rc.{candidate}",
        cabal,
        flags=re.MULTILINE,
    )
    cabal = re.sub(
        r"vinary-tree-interop >=\S+ && <\S+", "vinary-tree-interop >=4 && <5", cabal
    )
    cabal_path.write_text(cabal, encoding="utf-8")

    for path in ("Package.swift", "bindings/swift/liblevenshtein/Package.swift"):
        replace(
            path,
            r'(url: "https://github\.com/vinary-tree/vinary-tree-interop\.git",\n\s+exact: ")[^"]+("\n)',
            rf"\g<1>{versions['swiftTag']}\2",
        )

    replace(
        "bindings/clojure/deps.edn",
        r'[^\s{]+/liblevenshtein \{:mvn/version "[^"]+"\}',
        f'{maven_group}/{maven_artifact} {{:mvn/version "{versions["maven"]}"}}',
    )
    replace(
        "bindings/clojure/deps.edn",
        r"\{[^\s{]+/vinary-tree-interop\n",
        f"{{{interop_group}/{interop_artifact}\n",
    )
    replace(
        "bindings/clojure/deps.edn",
        r'\n    [^\s{]+/liblevenshtein \{:local/root "\.\./jvm"\}',
        f'\n    {maven_group}/{maven_artifact} {{:local/root "../jvm"}}',
    )
    replace(
        "bindings/clojure/README.md",
        r'(\[io\.vinarytree/liblevenshtein-clojure ")[^"]+("\])',
        rf"\g<1>{versions['clojars']}\2",
    )
    replace(
        "bindings/clojure/README.md",
        r'(io\.vinarytree/liblevenshtein-clojure \{:mvn/version ")[^"]+("\})',
        rf"\g<1>{versions['clojars']}\2",
    )
    replace(
        "docs/architecture/overview.md",
        r"(\| \*\*liblevenshtein\*\* \(this crate, `v)[^`]+(`\))",
        rf"\g<1>{canonical}\2",
    )
    replace(
        "docs/architecture/overview.md",
        r"(\| \*\*libdictenstein\*\* \(`v)[^`]+(`\))",
        rf"\g<1>{dependencies['libdictenstein']}\2",
    )

    lua_path = f"bindings/lua/liblevenshtein-{versions['luaRocks']}.rockspec"
    lua_target = ROOT / lua_path
    if not lua_target.exists():
        candidates = list((ROOT / "bindings/lua").glob("liblevenshtein-*.rockspec"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one LuaRocks source file, found {len(candidates)}"
            )
        candidates[0].rename(lua_target)
    replace(lua_path, r'^version = "[^"]+"$', f'version = "{versions["luaRocks"]}"')
    replace(
        lua_path,
        r'^(description = \{ summary = ")[^"]+(".*)$',
        rf"\g<1>{summary}\2",
    )
    replace(
        lua_path,
        r'^(source = \{ url = "[^"]+", tag = ")[^"]+(" \})$',
        rf"\g<1>{model['publication']['sourceTag']}\2",
    )
    replace(
        lua_path,
        r'"libdictenstein == [^"]+"',
        f'"libdictenstein == {versions["luaRocks"]}"',
    )

    replace(
        "cmake/liblevenshteinConfigVersion.cmake",
        r'^set\(PACKAGE_VERSION "[^"]+"\)$',
        f'set(PACKAGE_VERSION "{versions["cmake"]}")',
    )
    replace(
        "pkgconfig/liblevenshtein.pc",
        r"^Version: \S+$",
        f"Version: {versions['pkgConfig']}",
    )
    replace(
        "pkgconfig/liblevenshtein.pc",
        r"^Description: .+$",
        f"Description: {summary}",
    )
    replace(
        "pkgconfig/liblevenshtein.pc",
        r"^Requires: vinary-tree-interop = \S+$",
        f"Requires: vinary-tree-interop = {versions['pkgConfig']}",
    )
    rewrite_candidate_tokens(
        (
            ".github/actions/**/*.yml",
            ".github/workflows/*.yml",
            "benchmarks/cross-language/harnesses/go/go.mod",
            "bindings/**/*.md",
            "docs/**/*.md",
            "docs/**/*.puml",
            "release/package-documentation.json",
        ),
        canonical,
        int(model["publication"].get("luaRocksRevision", 1)),
        excluded=(
            "docs/diagrams/bindings/rejected-candidate-recovery.puml",
            "docs/releases/4.0.0-rc.1.md",
            "docs/releases/README.md",
        ),
    )


def validate(model: dict[str, object], versions: dict[str, str]) -> list[str]:
    failures: list[str] = []
    if model.get("registries") != versions:
        failures.append(
            "release/version.json registry spellings are not derived from canonical"
        )
    canonical = str(model["canonical"])
    candidate = canonical.rsplit(".", 1)[-1]
    publication = model.get("publication", {})
    if not isinstance(publication, dict) or publication.get("fpm") is not False:
        failures.append("fpm RC publication must remain embargoed")
    if not isinstance(publication, dict) or publication.get("hackage") is not False:
        failures.append("Hackage RC publication must remain embargoed")
    source_tag = publication.get("sourceTag") if isinstance(publication, dict) else None
    immutable_tag_pattern = rf"v{re.escape(canonical)}(?:-release\.[1-9][0-9]*)?"
    if (
        not isinstance(source_tag, str)
        or re.fullmatch(immutable_tag_pattern, source_tag) is None
    ):
        failures.append(
            "RC source tag must be canonical or an append-only numbered correction"
        )
    dependencies = model["dependencies"]
    assert isinstance(dependencies, dict)
    try:
        validate_source_refs(model)
    except (TypeError, ValueError) as error:
        failures.append(str(error))
    coordinates = maven_coordinates(model)
    maven_group = str(coordinates["mavenGroup"])
    maven_artifact = str(coordinates["mavenArtifact"])
    interop_group = str(coordinates["interopMavenGroup"])
    interop_artifact = str(coordinates["interopMavenArtifact"])
    npm_package = str(coordinates["npmPackage"])
    description = release_description(model)
    summary = release_summary(model)
    if npm_package != "@vinary-tree/liblevenshtein":
        failures.append("the canonical npm package must be @vinary-tree/liblevenshtein")
    if maven_group != "io.vinarytree":
        failures.append("the canonical Maven group must be io.vinarytree")
    if coordinates["javaPackage"] != "io.vinarytree.liblevenshtein":
        failures.append(
            "the canonical Java package must be io.vinarytree.liblevenshtein"
        )
    if coordinates["legacyMavenGroups"] != [
        "com.github.dylon",
        "com.github.universal-automata",
    ]:
        failures.append("legacy Maven relocation groups are incomplete or reordered")
    checks = {
        "Cargo crate": ("Cargo.toml", r'^version = "([^"]+)"$', canonical),
        "Cargo interop": (
            "Cargo.toml",
            r'^vinary-tree-interop = \{[^\n]*version = "=([^"]+)"',
            dependencies["vinary-tree-interop"],
        ),
        "Cargo libdictenstein": (
            "Cargo.toml",
            r'^libdictenstein = \{[^\n]*version = "=([^"]+)"',
            dependencies["libdictenstein"],
        ),
        "Python": (
            "bindings/python/pyproject.toml",
            r'^version = "([^"]+)"$',
            versions["pypi"],
        ),
        "Python interop": (
            "bindings/python/pyproject.toml",
            r'vinary-tree-interop==([^"]+)',
            derived(str(dependencies["vinary-tree-interop"]))["pypi"],
        ),
        "Julia": (
            "bindings/julia/Liblevenshtein/Project.toml",
            r'^version = "([^"]+)"$',
            versions["julia"],
        ),
        "Julia interop compatibility": (
            "bindings/julia/Liblevenshtein/Project.toml",
            r'^VinaryTreeInterop = "(\d+)"$',
            str(dependencies["vinary-tree-interop"]).split(".", 1)[0],
        ),
        "Julia docs self compatibility": (
            "bindings/julia/Liblevenshtein/docs/Project.toml",
            r'^Liblevenshtein = "(\d+)"$',
            canonical.split(".", 1)[0],
        ),
        "Julia docs interop compatibility": (
            "bindings/julia/Liblevenshtein/docs/Project.toml",
            r'^VinaryTreeInterop = "(\d+)"$',
            str(dependencies["vinary-tree-interop"]).split(".", 1)[0],
        ),
        "Raku": (
            "bindings/raku/META6.json",
            r'^  "version": "([^"]+)",$',
            versions["zef"],
        ),
        "JVM": (
            "bindings/jvm/build.gradle.kts",
            r'^version = "([^"]+)"$',
            versions["maven"],
        ),
        "JVM group": (
            "bindings/jvm/build.gradle.kts",
            r'^group = "([^"]+)"$',
            maven_group,
        ),
        "JVM interop": (
            "bindings/jvm/build.gradle.kts",
            r'api\("([^\"]+)"\)',
            f"{interop_group}:{interop_artifact}:{dependencies['vinary-tree-interop']}",
        ),
        "JVM substitution": (
            "bindings/jvm/settings.gradle.kts",
            r'substitute\(module\("([^"]+)"\)\)',
            f"{interop_group}:{interop_artifact}",
        ),
        "JReleaser group": (
            "bindings/jvm/jreleaser.yml",
            r"^      groupId: (\S+)$",
            maven_group,
        ),
        "JReleaser artifact": (
            "bindings/jvm/jreleaser.yml",
            r"^      artifactId: (\S+)$",
            maven_artifact,
        ),
        "JReleaser canonical namespace": (
            "bindings/jvm/jreleaser.yml",
            r"^        namespace: (\S+)$",
            maven_group,
        ),
        "JVM Clojure classpath": (
            "bindings/jvm/deps.edn",
            r":deps \{([^\s]+/vinary-tree-interop)",
            f"{interop_group}/{interop_artifact}",
        ),
        ".NET": (
            "bindings/dotnet/src/VinaryTree.Liblevenshtein/VinaryTree.Liblevenshtein.csproj",
            r"<Version>([^<]+)</Version>",
            versions["nuget"],
        ),
        "Ruby": (
            "bindings/ruby/lib/vinary_tree/liblevenshtein/version.rb",
            r'VERSION = "([^"]+)"',
            versions["rubygems"],
        ),
        "fpm": (
            "bindings/fortran/fpm.publish.toml",
            r'^version = "([^"]+)"$',
            versions["fpm"],
        ),
        "CMake": (
            "cmake/liblevenshteinConfigVersion.cmake",
            r'^set\(PACKAGE_VERSION "([^"]+)"\)$',
            versions["cmake"],
        ),
        "pkg-config": (
            "pkgconfig/liblevenshtein.pc",
            r"^Version: (\S+)$",
            versions["pkgConfig"],
        ),
        "pkg-config interop": (
            "pkgconfig/liblevenshtein.pc",
            r"^Requires: vinary-tree-interop = (\S+)$",
            versions["pkgConfig"],
        ),
        "Swift root interop": (
            "Package.swift",
            r'exact: "([^"]+)"',
            versions["swiftTag"],
        ),
        "Swift facade interop": (
            "bindings/swift/liblevenshtein/Package.swift",
            r'exact: "([^"]+)"',
            versions["swiftTag"],
        ),
        "Clojure CLI JVM": (
            "bindings/clojure/deps.edn",
            rf'{re.escape(maven_group)}/{re.escape(maven_artifact)} \{{:mvn/version "([^"]+)"\}}',
            versions["maven"],
        ),
        "Clojure README Leiningen": (
            "bindings/clojure/README.md",
            r'\[io\.vinarytree/liblevenshtein-clojure "([^"]+)"\]',
            versions["clojars"],
        ),
        "Clojure README tools.deps": (
            "bindings/clojure/README.md",
            r'io\.vinarytree/liblevenshtein-clojure \{:mvn/version "([^"]+)"\}',
            versions["clojars"],
        ),
        "Architecture root crate": (
            "docs/architecture/overview.md",
            r"\| \*\*liblevenshtein\*\* \(this crate, `v([^`]+)`\)",
            canonical,
        ),
        "Architecture dictionary crate": (
            "docs/architecture/overview.md",
            r"\| \*\*libdictenstein\*\* \(`v([^`]+)`\)",
            dependencies["libdictenstein"],
        ),
        "LuaRocks": (
            f"bindings/lua/liblevenshtein-{versions['luaRocks']}.rockspec",
            r'^version = "([^"]+)"$',
            versions["luaRocks"],
        ),
    }
    for name, (path, pattern, wanted) in checks.items():
        match = re.search(pattern, read(path), flags=re.MULTILINE)
        actual = match.group(1) if match else None
        if actual != wanted:
            failures.append(f"{name}: expected {wanted}, got {actual}")
    python_source = (
        "vinary-tree-interop = { path = "
        '"../../../vinary-tree-interop/bindings/python" }'
    )
    if python_source not in read("bindings/python/pyproject.toml"):
        failures.append(
            "Python uv source must resolve the exact checked-out interop sibling"
        )
    lua_rockspec = read(f"bindings/lua/liblevenshtein-{versions['luaRocks']}.rockspec")
    lua_source = re.search(
        r'^source = \{ url = "[^"]+", tag = "([^"]+)" \}$',
        lua_rockspec,
        flags=re.MULTILINE,
    )
    if (lua_source.group(1) if lua_source else None) != source_tag:
        failures.append("LuaRocks source tag is stale")
    jreleaser = read("bindings/jvm/jreleaser.yml")
    if re.search(r"^        active: RELEASE$", jreleaser, flags=re.MULTILINE):
        failures.append(
            "JReleaser Maven deployers must remain inactive until one workflow lane selects them"
        )
    for marker in ("canonical:", "legacyDylon:", "legacyUniversalAutomata:"):
        if marker not in jreleaser:
            failures.append(f"JReleaser named deployer is missing {marker}")
    for legacy_group in coordinates["legacyMavenGroups"]:
        repository = legacy_group.replace(".", "-")
        for marker in (
            f"namespace: {legacy_group}",
            f"groupId: {legacy_group}",
            f"build/staging-relocations/{repository}",
        ):
            if marker not in jreleaser:
                failures.append(f"JReleaser legacy relocation is missing {marker}")
    expected_locks = {
        "liblevenshtein": canonical,
        "libdictenstein": str(dependencies["libdictenstein"]),
        "vinary-tree-interop": str(dependencies["vinary-tree-interop"]),
    }
    lock_expectations = {
        "Cargo.lock": expected_locks,
        "liblevenshtein-macros/Cargo.lock": {
            "liblevenshtein": canonical,
            "libdictenstein": str(dependencies["libdictenstein"]),
        },
    }
    for path, expected in lock_expectations.items():
        for package, actual in cargo_lock_versions(expected, path).items():
            if actual != expected[package]:
                failures.append(
                    f"{path} {package}: expected {expected[package]}, got {actual}"
                )
    python_lock_expected = {
        "liblevenshtein": versions["pypi"],
        "vinary-tree-interop": derived(str(dependencies["vinary-tree-interop"]))[
            "pypi"
        ],
    }
    for package, actual in cargo_lock_versions(
        python_lock_expected, "bindings/python/uv.lock"
    ).items():
        if actual != python_lock_expected[package]:
            failures.append(
                "bindings/python/uv.lock "
                f"{package}: expected {python_lock_expected[package]}, got {actual}"
            )
    julia_manifest_expected = {
        "Liblevenshtein": versions["julia"],
        "VinaryTreeInterop": str(dependencies["vinary-tree-interop"]),
    }
    for package, expected in julia_manifest_expected.items():
        block = re.search(
            rf"\[\[deps\.{re.escape(package)}\]\]"
            rf'(?:(?!\n\[\[deps\.).)*?\nversion = "([^"]+)"',
            read("bindings/julia/Liblevenshtein/docs/Manifest.toml"),
            flags=re.DOTALL,
        )
        actual = block.group(1) if block else None
        if actual != expected:
            failures.append(
                "bindings/julia/Liblevenshtein/docs/Manifest.toml "
                f"{package}: expected {expected}, got {actual}"
            )
    api = json.loads(read("bindings/api.json"))
    if (
        api.get("packageVersion") != canonical
        or api.get("release", {}).get("registries") != versions
    ):
        failures.append("bindings/api.json release identity is stale")
    raku = json.loads(read("bindings/raku/META6.json"))
    expected_raku_dependency_version = canonical.replace("-rc.", ".rc.")
    for field in ("depends", "test-depends"):
        dependencies_for_field = raku.get(field, [])
        if not isinstance(dependencies_for_field, list):
            failures.append(f"Raku {field} must be an array")
            continue
        for dependency in dependencies_for_field:
            if (
                isinstance(dependency, str)
                and ":ver<" in dependency
                and f":ver<{expected_raku_dependency_version}>" not in dependency
            ):
                failures.append(f"Raku {field} dependency is stale: {dependency}")
    package = json.loads(read("bindings/javascript/package.json"))
    if package.get("name") != npm_package:
        failures.append("npm facade package coordinate is stale")
    if package.get("version") != versions["npm"]:
        failures.append("npm facade version is stale")
    if (
        package.get("dependencies", {}).get("@vinary-tree/javascript-runtime")
        != dependencies["@vinary-tree/javascript-runtime"]
    ):
        failures.append("npm facade runtime pin is stale")
    if package.get("publishConfig", {}).get("tag") != publication.get("distTag"):
        failures.append("npm facade dist-tag policy is stale")
    description_surfaces = {
        "Cargo": read("Cargo.toml"),
        "npm": read("bindings/javascript/package.json"),
        "PyPI": read("bindings/python/pyproject.toml"),
        "Maven POM": read("bindings/jvm/build.gradle.kts"),
        "JReleaser": read("bindings/jvm/jreleaser.yml"),
        "Clojars": read("bindings/clojure/project.clj"),
        "NuGet": read(
            "bindings/dotnet/src/VinaryTree.Liblevenshtein/VinaryTree.Liblevenshtein.csproj"
        ),
        "RubyGems": read("bindings/ruby/liblevenshtein.gemspec"),
        "fpm development": read("bindings/fortran/fpm.toml"),
        "fpm publication": read("bindings/fortran/fpm.publish.toml"),
        "opam": read("bindings/ocaml/liblevenshtein.opam"),
        "opam template": read("bindings/ocaml/liblevenshtein.opam.template"),
        "Dune": read("bindings/ocaml/dune-project"),
        "Hackage": read("bindings/haskell/liblevenshtein.cabal"),
        "LuaRocks": read(
            f"bindings/lua/liblevenshtein-{versions['luaRocks']}.rockspec"
        ),
        "Raku": read("bindings/raku/META6.json"),
        "pkg-config": read("pkgconfig/liblevenshtein.pc"),
    }
    description_markers = {
        "Cargo": f'description = "{description}"',
        "npm": f'"description": "{description}"',
        "PyPI": f'description = "{description}"',
        "Maven POM": f'description = "{description}"',
        "JReleaser": f"description: {description}",
        "Clojars": f':description "{description}"',
        "NuGet": f"<Description>{description}</Description>",
        "RubyGems": f'spec.summary = "{summary}"',
        "fpm development": f'description = "{description}"',
        "fpm publication": f'description = "{description}"',
        "opam": f'synopsis: "{summary}"',
        "opam template": f'synopsis: "{summary}"',
        "Dune": f'(synopsis "{summary}")',
        "Hackage": f"synopsis: {summary}",
        "LuaRocks": f'summary = "{summary}"',
        "Raku": f'"description": "{description}"',
        "pkg-config": f"Description: {summary}",
    }
    for surface, source in description_surfaces.items():
        if description_markers[surface] not in source:
            failures.append(f"{surface} description differs from release metadata")
    go_mod = read("bindings/go/go.mod")
    if "module github.com/vinary-tree/liblevenshtein-rust/bindings/go/v4" not in go_mod:
        failures.append("Go module lacks /v4 semantic import path")
    if (
        f"github.com/vinary-tree/vinary-tree-interop/bindings/go/v4 {versions['goTag']}"
        not in go_mod
    ):
        failures.append("Go interop dependency is stale")
    cabal = read("bindings/haskell/liblevenshtein.cabal")
    if (
        f"version: {versions['hackage']}" not in cabal
        or f"x-release-candidate: rc.{candidate}" not in cabal
    ):
        failures.append("Hackage source candidate metadata is stale")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    publication = model.get("publication", {})
    lua_rocks_revision = publication.get("luaRocksRevision", 1)
    if not isinstance(lua_rocks_revision, int) or lua_rocks_revision < 1:
        print(
            "release-version error: publication.luaRocksRevision must be a positive integer",
            file=sys.stderr,
        )
        return 1
    versions = derived(str(model["canonical"]), lua_rocks_revision)
    if args.write:
        write_versions(model, versions)
    failures = validate(model, versions)
    if failures:
        for failure in failures:
            print(f"release-version error: {failure}", file=sys.stderr)
        return 1
    print(f"release versions agree with {model['canonical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
