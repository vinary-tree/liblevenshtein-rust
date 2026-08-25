#!/usr/bin/env python3
"""Write or validate every liblevenshtein 4.x release coordinate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "release/version.json"
GENERATED_TREE_PARTS = frozenset(
    {".git", ".venv", "_build", "build", "dist", "node_modules", "target", "venv"}
)


def derived(canonical: str) -> dict[str, str]:
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
        "luaRocks": f"{base}rc{candidate}-1",
        "maven": canonical,
        "npm": canonical,
        "nuget": canonical,
        "opam": f"{base}~rc{candidate}",
        "pkgConfig": canonical,
        "pypi": f"{base}rc{candidate}",
        "rubygems": f"{base}.rc.{candidate}",
        "swiftTag": canonical,
    }


def maven_coordinates(model: dict[str, object]) -> dict[str, object]:
    coordinates = model.get("coordinates")
    if not isinstance(coordinates, dict):
        raise TypeError("release/version.json requires coordinates")
    expected_strings = (
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


def rewrite_cargo_lock(expected: dict[str, str]) -> None:
    target = ROOT / "Cargo.lock"
    source = target.read_text(encoding="utf-8")
    for package, version in expected.items():
        pattern = rf'(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = ")[^"]+'
        source, count = re.subn(pattern, rf"\g<1>{version}", source)
        if count != 1:
            raise ValueError(
                f"Cargo.lock: expected one {package} package entry, found {count}"
            )
    target.write_text(source, encoding="utf-8")


def cargo_lock_versions(expected: dict[str, str]) -> dict[str, str | None]:
    source = read("Cargo.lock")
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
    excluded: tuple[str, ...] = (),
) -> None:
    base, candidate = canonical.split("-rc.", 1)
    escaped = re.escape(base)
    replacements = (
        (rf"{escaped}\.rc\.\d+", f"{base}.rc.{candidate}"),
        (rf"{escaped}~rc\d+", f"{base}~rc{candidate}"),
        (rf"{escaped}\\textasciitilde rc\d+", rf"{base}\textasciitilde rc{candidate}"),
        (rf"{escaped}rc\d+-\d+", f"{base}rc{candidate}-1"),
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

    replace("Cargo.toml", r'^version = "[^"]+"$', f'version = "{canonical}"')
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

    def api(value: dict) -> None:
        value["packageVersion"] = canonical
        value["release"] = {
            "canonical": canonical,
            "registries": versions,
            "distTag": model["publication"]["distTag"],
        }

    update_json("bindings/api.json", api)

    def related(value: dict) -> None:
        value.clear()
        for name in ("llattice", "libdictenstein", "lling-llang", "duallity"):
            version = str(dependencies[name])
            value[name] = {"version": version, "ref": f"v{version}"}

    update_json("bindings/related-projects.json", related)

    def npm(value: dict) -> None:
        value["version"] = versions["npm"]
        value["dependencies"]["@vinary-tree/interop"] = dependencies[
            "@vinary-tree/interop"
        ]
        value["dependencies"]["@vinary-tree/vinary-tree"] = dependencies[
            "@vinary-tree/vinary-tree"
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
                "@vinary-tree/interop",
                "@vinary-tree/libdictenstein",
                "@vinary-tree/vinary-tree",
            }:
                package["version"] = canonical
            package_dependencies = package.get("dependencies", {})
            if "@vinary-tree/interop" in package_dependencies:
                package_dependencies["@vinary-tree/interop"] = dependencies[
                    "@vinary-tree/interop"
                ]
            if "@vinary-tree/vinary-tree" in package_dependencies:
                package_dependencies["@vinary-tree/vinary-tree"] = dependencies[
                    "@vinary-tree/vinary-tree"
                ]

    update_json("bindings/javascript/package-lock.json", npm_lock)
    replace(
        "bindings/javascript/build.mjs",
        r'assert\.equal\(packageJson\.dependencies\["@vinary-tree/interop"\], "[^"]+"\);',
        f'assert.equal(packageJson.dependencies["@vinary-tree/interop"], "{dependencies["@vinary-tree/interop"]}");',
    )
    replace(
        "bindings/javascript/test/facades.test.mjs",
        r'assert\.equal\(packageJson\.dependencies\["@vinary-tree/interop"\], "[^"]+"\);',
        f'assert.equal(packageJson.dependencies["@vinary-tree/interop"], "{dependencies["@vinary-tree/interop"]}");',
    )

    replace(
        "bindings/python/pyproject.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["pypi"]}"',
    )
    replace(
        "bindings/python/pyproject.toml",
        r'^dependencies = \["vinary-tree-interop==[^"]+"\]$',
        f'dependencies = ["vinary-tree-interop=={versions["pypi"]}"]',
    )
    replace(
        "bindings/jvm/build.gradle.kts",
        r'^version = "[^"]+"$',
        f'version = "{versions["maven"]}"',
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
        r'<PackageReference Include="VinaryTree\.Interop" Version="[^"]+" />',
        f'<PackageReference Include="VinaryTree.Interop" Version="{versions["nuget"]}" />',
    )
    replace(
        "bindings/ruby/lib/vinary_tree/liblevenshtein/version.rb",
        r'^    VERSION = "[^"]+"$',
        f'    VERSION = "{versions["rubygems"]}"',
    )
    replace(
        "bindings/fortran/fpm.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["fpm"]}"',
    )
    replace(
        "bindings/fortran/fpm.publish.toml",
        r'^version = "[^"]+"$',
        f'version = "{versions["fpm"]}"',
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
        "bindings/haskell/liblevenshtein.cabal",
        r"^version: \S+$",
        f"version: {versions['hackage']}",
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
        ),
        canonical,
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
    if source_tag != f"v{canonical}-release.2":
        failures.append(
            "RC.4 publishable source tag must remain the append-only release.2 correction"
        )
    dependencies = model["dependencies"]
    assert isinstance(dependencies, dict)
    coordinates = maven_coordinates(model)
    maven_group = str(coordinates["mavenGroup"])
    maven_artifact = str(coordinates["mavenArtifact"])
    interop_group = str(coordinates["interopMavenGroup"])
    interop_artifact = str(coordinates["interopMavenArtifact"])
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
    for package, actual in cargo_lock_versions(expected_locks).items():
        if actual != expected_locks[package]:
            failures.append(
                f"Cargo.lock {package}: expected {expected_locks[package]}, got {actual}"
            )
    api = json.loads(read("bindings/api.json"))
    if (
        api.get("packageVersion") != canonical
        or api.get("release", {}).get("registries") != versions
    ):
        failures.append("bindings/api.json release identity is stale")
    package = json.loads(read("bindings/javascript/package.json"))
    if package.get("version") != versions["npm"]:
        failures.append("npm facade version is stale")
    if (
        package.get("dependencies", {}).get("@vinary-tree/vinary-tree")
        != dependencies["@vinary-tree/vinary-tree"]
    ):
        failures.append("npm facade runtime pin is stale")
    if package.get("publishConfig", {}).get("tag") != publication.get("distTag"):
        failures.append("npm facade dist-tag policy is stale")
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
    versions = derived(str(model["canonical"]))
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
