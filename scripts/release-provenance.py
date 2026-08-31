#!/usr/bin/env python3
"""Generate or verify the exact local subject of a coordinated release tuple.

The JSON artifact deliberately identifies immutable source commits, repository
content, dependency/feature contracts, lockfiles, package file sets, and the
executing Rust toolchain without embedding mutable worktree status or
pretending that a release candidate already has final publication tags. A
final release attestation can bind these stable subjects to tags and registry
artifacts after those objects exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "release/rc6-candidate-provenance.json"
EXPECTED_RELEASE = "4.0.0-rc.6"
EXPECTED_RUST_VERSION = "1.95"
EXPECTED_RUST_CHANNEL = "1.95.0"

COMPONENT_SPECS = (
    ("liblevenshtein", "LIBLEVENSHTEIN_ROOT", ROOT, "liblevenshtein-rust"),
    (
        "libdictenstein",
        "LIBDICTENSTEIN_ROOT",
        ROOT.parent / "libdictenstein",
        "libdictenstein",
    ),
    (
        "vinary-tree-interop",
        "VINARY_TREE_INTEROP_ROOT",
        ROOT.parent / "vinary-tree-interop",
        "vinary-tree-interop",
    ),
)

BUILD_INPUT_SPECS = (
    ("llattice", "LLATTICE_ROOT", ROOT.parent / "llattice", "llattice", "0.1.0"),
)

COORDINATED_DEPENDENCIES = {
    "liblevenshtein": {
        "libdictenstein": (EXPECTED_RELEASE, "../libdictenstein"),
        "vinary-tree-interop": (EXPECTED_RELEASE, "../vinary-tree-interop"),
    },
    "libdictenstein": {
        "llattice": ("0.1.0", "../llattice"),
        "vinary-tree-interop": (EXPECTED_RELEASE, "../vinary-tree-interop"),
    },
    "vinary-tree-interop": {},
}

EXPECTED_LOCK_PACKAGES = {
    "liblevenshtein": {
        "Cargo.lock": {
            "liblevenshtein": EXPECTED_RELEASE,
            "libdictenstein": EXPECTED_RELEASE,
            "vinary-tree-interop": EXPECTED_RELEASE,
        },
        "liblevenshtein-macros/Cargo.lock": {
            "liblevenshtein": EXPECTED_RELEASE,
            "libdictenstein": EXPECTED_RELEASE,
        },
    },
    "libdictenstein": {
        "Cargo.lock": {
            "libdictenstein": EXPECTED_RELEASE,
            "vinary-tree-interop": EXPECTED_RELEASE,
        }
    },
    "vinary-tree-interop": {"Cargo.lock": {"vinary-tree-interop": EXPECTED_RELEASE}},
}


def fail(message: str) -> None:
    raise SystemExit(f"release provenance error: {message}")


def run(command: list[str], *, cwd: Path, check: bool = True) -> bytes:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        fail(f"{' '.join(command)} failed in {cwd}: {detail}")
    return completed.stdout


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_roots() -> list[tuple[str, Path, str]]:
    roots = []
    for name, environment, default, canonical_directory in COMPONENT_SPECS:
        root = Path(os.environ.get(environment, default)).resolve()
        if not (root / ".git").exists():
            fail(f"{name}: repository root does not exist: {root}")
        roots.append((name, root, canonical_directory))
    if len({root for _, root, _ in roots}) != len(roots):
        fail("two components resolve to the same repository root")
    return roots


def build_input_roots() -> list[tuple[str, Path, str, str]]:
    roots = []
    for name, environment, default, canonical_directory, version in BUILD_INPUT_SPECS:
        root = Path(os.environ.get(environment, default)).resolve()
        if not (root / ".git").exists():
            fail(f"{name}: build-input repository root does not exist: {root}")
        roots.append((name, root, canonical_directory, version))
    if len({root for _, root, _, _ in roots}) != len(roots):
        fail("two build inputs resolve to the same repository root")
    return roots


def read_release_model(name: str, root: Path) -> dict[str, object]:
    path = root / "release/version.json"
    if not path.is_file():
        fail(f"{name}: missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("component") != name:
        fail(f"{name}: release model component is {value.get('component')!r}")
    if value.get("canonical") != EXPECTED_RELEASE:
        fail(f"{name}: canonical release is {value.get('canonical')!r}")
    if value.get("registries", {}).get("cargo") != EXPECTED_RELEASE:
        fail(f"{name}: Cargo registry version is not {EXPECTED_RELEASE}")
    return value


def manifest_value(source: str, field: str) -> str:
    match = re.search(rf'^{re.escape(field)}\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if match is None:
        fail(f"Cargo.toml has no string {field!r} field")
    return match.group(1)


def check_manifest(name: str, root: Path, require_canonical_paths: bool) -> None:
    source = (root / "Cargo.toml").read_text(encoding="utf-8")
    if manifest_value(source, "version") != EXPECTED_RELEASE:
        fail(f"{name}: Cargo package version is not {EXPECTED_RELEASE}")
    if manifest_value(source, "rust-version") != EXPECTED_RUST_VERSION:
        fail(f"{name}: rust-version is not {EXPECTED_RUST_VERSION}")
    for dependency, (expected_version, canonical_path) in COORDINATED_DEPENDENCIES[
        name
    ].items():
        match = re.search(
            rf"^{re.escape(dependency)}\s*=\s*\{{([^\n]+)\}}$",
            source,
            re.MULTILINE,
        )
        if match is None:
            fail(f"{name}: missing {dependency} dependency")
        fields = match.group(1)
        if f'version = "={expected_version}"' not in fields:
            fail(f"{name}: {dependency} is not pinned to ={expected_version}")
        if require_canonical_paths and f'path = "{canonical_path}"' not in fields:
            fail(f"{name}: {dependency} does not use canonical path {canonical_path}")


def manifest_release_contract(name: str, root: Path) -> dict[str, object]:
    """Return the explicit Cargo feature and coordinated-dependency contract."""

    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    package = manifest.get("package", {})
    features = manifest.get("features", {})
    dependencies = manifest.get("dependencies", {})

    dependency_contract: dict[str, object] = {}
    for dependency in sorted(COORDINATED_DEPENDENCIES[name]):
        specification = dependencies.get(dependency)
        if not isinstance(specification, dict):
            fail(f"{name}: {dependency} dependency is not an explicit table")
        dependency_contract[dependency] = {
            "version": specification.get("version"),
            "optional": bool(specification.get("optional", False)),
            "defaultFeatures": bool(specification.get("default-features", True)),
            "features": sorted(specification.get("features", [])),
        }

    return {
        "packageVersion": package.get("version"),
        "cargoFeatures": {
            feature: list(members) for feature, members in sorted(features.items())
        },
        "validatedCargoSelection": "--all-features",
        "coordinatedDependencies": dependency_contract,
    }


def check_build_input(name: str, root: Path, expected_version: str) -> None:
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    package = manifest.get("package", {})
    if package.get("name") != name:
        fail(f"{name}: build-input Cargo package name is {package.get('name')!r}")
    if package.get("version") != expected_version:
        fail(
            f"{name}: build-input Cargo package version is "
            f"{package.get('version')!r}, expected {expected_version!r}"
        )


def build_input_contract(root: Path) -> dict[str, object]:
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    package = manifest.get("package", {})
    features = manifest.get("features", {})
    return {
        "packageVersion": package.get("version"),
        "rustVersion": package.get("rust-version"),
        "cargoFeatures": {
            feature: list(members) for feature, members in sorted(features.items())
        },
        "validatedCargoSelection": "dependency-selected feature set",
    }


def check_toolchain_file(name: str, root: Path) -> None:
    path = root / "rust-toolchain.toml"
    if not path.is_file():
        fail(f"{name}: missing rust-toolchain.toml")
    source = path.read_text(encoding="utf-8")
    if f'channel = "{EXPECTED_RUST_CHANNEL}"' not in source:
        fail(f"{name}: toolchain channel is not {EXPECTED_RUST_CHANNEL}")


def check_lock_versions(name: str, root: Path) -> None:
    for relative, expected_packages in EXPECTED_LOCK_PACKAGES[name].items():
        path = root / relative
        if not path.is_file():
            fail(f"{name}: missing independently validated lockfile {relative}")
        source = path.read_text(encoding="utf-8")
        if "[[patch.unused]]" in source or 'name = "creusot-std"' in source:
            fail(f"{name}: host-global unused patch leaked into {relative}")
        for package, expected_version in expected_packages.items():
            match = re.search(
                rf'\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"',
                source,
            )
            if match is None:
                fail(f"{name}: {relative} has no {package} package entry")
            if match.group(1) != expected_version:
                fail(
                    f"{name}: {relative} resolves {package} {match.group(1)}, "
                    f"expected {expected_version}"
                )


def source_paths(root: Path) -> list[Path]:
    output = run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    paths = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if root == ROOT and relative == ARTIFACT.relative_to(ROOT):
            continue
        if (root / relative).is_file() or (root / relative).is_symlink():
            paths.append(relative)
    return sorted(set(paths), key=lambda value: os.fsencode(value.as_posix()))


def source_snapshot(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = source_paths(root)
    for relative in paths:
        path = root / relative
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            kind = b"120000"
            content = os.fsencode(os.readlink(path))
        else:
            kind = b"100755" if mode & stat.S_IXUSR else b"100644"
            content = path.read_bytes()
        name = os.fsencode(relative.as_posix())
        digest.update(kind + b"\0" + name + b"\0")
        digest.update(str(len(content)).encode("ascii") + b"\0" + content)
    return digest.hexdigest(), len(paths)


def status_lines(root: Path) -> list[str]:
    output = run(
        ["git", "status", "--short", "--untracked-files=all"], cwd=root
    ).decode("utf-8")
    artifact = ARTIFACT.relative_to(ROOT).as_posix() if root == ROOT else None
    return [
        line
        for line in output.splitlines()
        if line and (artifact is None or not line.endswith(artifact))
    ]


def package_file_set(root: Path) -> tuple[str, int]:
    command = [
        "cargo",
        f"+{EXPECTED_RUST_CHANNEL}",
        "package",
        "--manifest-path",
        str(root / "Cargo.toml"),
        "--list",
        "--locked",
        "--offline",
        "--allow-dirty",
    ]
    output = run(command, cwd=Path("/"))
    files = sorted(line for line in output.decode("utf-8").splitlines() if line)
    required = {"Cargo.lock", "Cargo.toml", "LICENSE", "README.md", "src/lib.rs"}
    missing = sorted(required.difference(files))
    if missing:
        fail(f"{root.name}: package omits required files: {', '.join(missing)}")
    canonical = "".join(f"{name}\n" for name in files).encode("utf-8")
    return sha256_bytes(canonical), len(files)


def tool_output(command: list[str]) -> str:
    return run(command, cwd=Path("/")).decode("utf-8").strip()


def build_artifact(
    roots: list[tuple[str, Path, str]],
    build_inputs: list[tuple[str, Path, str, str]],
    *,
    include_package_files: bool,
) -> dict[str, object]:
    components = []
    for name, root, canonical_directory in roots:
        model = read_release_model(name, root)
        snapshot_sha, source_count = source_snapshot(root)
        release_contract = manifest_release_contract(name, root)
        locks = {}
        for lock in sorted(root.glob("**/Cargo.lock")):
            if any(part == "target" for part in lock.relative_to(root).parts):
                continue
            source = lock.read_text(encoding="utf-8")
            if "[[patch.unused]]" in source or 'name = "creusot-std"' in source:
                fail(f"{name}: host-global unused patch leaked into {lock}")
            locks[lock.relative_to(root).as_posix()] = sha256_file(lock)
        component: dict[str, object] = {
            "name": name,
            "canonicalDirectory": canonical_directory,
            "repository": manifest_value(
                (root / "Cargo.toml").read_text(encoding="utf-8"), "repository"
            ),
            "subjectCommit": tool_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "releaseVersion": model["canonical"],
            "cargoReleaseContract": release_contract,
            "sourceSnapshotSha256": snapshot_sha,
            "sourceFileCount": source_count,
            "cargoLocksSha256": locks,
        }
        if include_package_files:
            package_sha, package_count = package_file_set(root)
            component["cargoPackageFileListSha256"] = package_sha
            component["cargoPackageFileCount"] = package_count
        components.append(component)
    inputs = []
    for name, root, canonical_directory, expected_version in build_inputs:
        snapshot_sha, source_count = source_snapshot(root)
        locks = {}
        for lock in sorted(root.glob("**/Cargo.lock")):
            if "target" in lock.relative_to(root).parts:
                continue
            source = lock.read_text(encoding="utf-8")
            if "[[patch.unused]]" in source or 'name = "creusot-std"' in source:
                fail(f"{name}: host-global unused patch leaked into {lock}")
            locks[lock.relative_to(root).as_posix()] = sha256_file(lock)
        build_input: dict[str, object] = {
            "name": name,
            "canonicalDirectory": canonical_directory,
            "requiredVersion": expected_version,
            "repository": manifest_value(
                (root / "Cargo.toml").read_text(encoding="utf-8"), "repository"
            ),
            "subjectCommit": tool_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "cargoContract": build_input_contract(root),
            "sourceSnapshotSha256": snapshot_sha,
            "sourceFileCount": source_count,
            "cargoLocksSha256": locks,
        }
        if include_package_files:
            package_sha, package_count = package_file_set(root)
            build_input["cargoPackageFileListSha256"] = package_sha
            build_input["cargoPackageFileCount"] = package_count
        inputs.append(build_input)
    return {
        "schemaVersion": 1,
        "kind": "coordinated-release-candidate-subject",
        "release": EXPECTED_RELEASE,
        "finalCommitAttestationRequired": True,
        "toolchain": {
            "declaredRustVersion": EXPECTED_RUST_VERSION,
            "declaredChannel": EXPECTED_RUST_CHANNEL,
            "rustcVerbose": tool_output(
                ["rustc", f"+{EXPECTED_RUST_CHANNEL}", "--version", "--verbose"]
            ),
            "cargoVerbose": tool_output(
                ["cargo", f"+{EXPECTED_RUST_CHANNEL}", "--version", "--verbose"]
            ),
        },
        "components": components,
        "buildInputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--include-package-files", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-canonical-paths", action="store_true")
    arguments = parser.parse_args()

    roots = component_roots()
    build_inputs = build_input_roots()
    all_roots = [(name, root) for name, root, _ in roots]
    all_roots.extend((name, root) for name, root, _, _ in build_inputs)
    if len({root for _, root in all_roots}) != len(all_roots):
        fail("a release component and build input resolve to the same repository root")
    for name, root, _ in roots:
        read_release_model(name, root)
        check_manifest(name, root, arguments.require_canonical_paths)
        check_toolchain_file(name, root)
        check_lock_versions(name, root)
    for name, root, _, expected_version in build_inputs:
        check_build_input(name, root, expected_version)

    actual = build_artifact(
        roots,
        build_inputs,
        include_package_files=arguments.include_package_files,
    )
    if arguments.require_clean:
        dirty = [name for name, root in all_roots if status_lines(root)]
        if dirty:
            fail(f"release subject is dirty: {', '.join(dirty)}")

    serialized = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if arguments.write:
        ARTIFACT.write_text(serialized, encoding="utf-8")
        print(f"wrote {ARTIFACT}")
        return
    if not ARTIFACT.is_file():
        fail(f"missing {ARTIFACT}")
    expected = ARTIFACT.read_text(encoding="utf-8")
    if expected != serialized:
        fail("candidate provenance does not match the current release subject")
    print(f"release provenance agrees with {EXPECTED_RELEASE}")


if __name__ == "__main__":
    main()
