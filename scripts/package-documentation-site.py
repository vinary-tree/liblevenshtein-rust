#!/usr/bin/env python3
"""Create and assemble immutable, versioned package-documentation archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

try:
    from scripts.package_documentation_surfaces import GENERATED_SURFACE_LAYOUT
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from package_documentation_surfaces import GENERATED_SURFACE_LAYOUT


ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / "target" / "package-documentation"
RELEASE_ROOT = ROOT / "target" / "package-documentation-release"
ARTIFACT_ROOT = ROOT / "target" / "package-documentation-artifacts"
SITE_ROOT = ROOT / "target" / "package-documentation-site"
RELEASE_PATH = ROOT / "release" / "version.json"
SURFACES = {
    surface: (GENERATED_ROOT / relative, entry_point)
    for surface, (relative, entry_point) in GENERATED_SURFACE_LAYOUT.items()
}
SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)


def fail(message: str) -> None:
    raise SystemExit(f"package-documentation-site: {message}")


def clean_directory(path: Path) -> None:
    resolved = path.resolve()
    target = (ROOT / "target").resolve()
    try:
        resolved.relative_to(target)
    except ValueError:
        fail(f"refusing to clean outside repository target: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True, exist_ok=True)


def release_authority() -> tuple[str, str]:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    version = release.get("canonical")
    publication = release.get("publication")
    source_ref = publication.get("sourceTag") if isinstance(publication, dict) else None
    if not isinstance(version, str) or SEMVER_RE.fullmatch(version) is None:
        fail("release/version.json has an invalid canonical semantic version")
    if not isinstance(source_ref, str) or not source_ref:
        fail("release/version.json lacks publication.sourceTag")
    return version, source_ref


def require_source_identity(source_ref: str) -> None:
    def revision(rev: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "rev-parse",
                "--verify",
                f"{rev}^{{commit}}",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fail(f"cannot resolve immutable source revision {rev!r}")
        return result.stdout.strip()

    head = revision("HEAD")
    expected = revision(source_ref)
    if head != expected:
        fail(f"HEAD {head} does not equal immutable source {source_ref} ({expected})")
    for staged in (False, True):
        command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "diff",
            "--quiet",
            "--ignore-submodules",
        ]
        if staged:
            command.insert(4, "--cached")
        command.append("--")
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            kind = "staged" if staged else "unstaged"
            fail(f"refusing to archive {kind} tracked source changes")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def render_version_index(version: str, surfaces: list[str]) -> str:
    items = "\n".join(
        f'      <li><a href="{html.escape(surface)}/">'
        f"{html.escape(surface.title())} API reference</a></li>"
        for surface in surfaces
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>liblevenshtein {html.escape(version)} package documentation</title>
  </head>
  <body>
    <main>
      <h1>liblevenshtein {html.escape(version)}</h1>
      <p>Versioned API references generated from the immutable release source.</p>
      <ul>
{items}
      </ul>
    </main>
  </body>
</html>
"""


def render_site_index(versions: list[str]) -> str:
    items = "\n".join(
        f'      <li><a href="{html.escape(version)}/">'
        f"liblevenshtein {html.escape(version)}</a></li>"
        for version in versions
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>liblevenshtein package documentation</title>
  </head>
  <body>
    <main>
      <h1>liblevenshtein package documentation</h1>
      <p>Select an immutable package version.</p>
      <ul>
{items}
      </ul>
    </main>
  </body>
</html>
"""


def semver_key(
    version: str,
) -> tuple[int, int, int, int, tuple[tuple[int, object], ...]]:
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        fail(f"invalid semantic-version directory: {version}")
    pre = match.group("pre")
    identifiers: list[tuple[int, object]] = []
    if pre is not None:
        for identifier in pre.split("."):
            identifiers.append(
                (0, int(identifier)) if identifier.isdigit() else (1, identifier)
            )
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        1 if pre is None else 0,
        tuple(identifiers),
    )


def deterministic_archive(source: Path, destination: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                fail(f"unsupported documentation source type: {path}")
            relative = path.relative_to(source.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0))


def build_archive() -> Path:
    version, source_ref = release_authority()
    require_source_identity(source_ref)
    clean_directory(RELEASE_ROOT)
    version_root = RELEASE_ROOT / version
    version_root.mkdir(parents=True)
    for surface, (source, required_index) in SURFACES.items():
        if not (source / required_index).is_file():
            fail(
                f"{surface} reference has not been generated: {source / required_index}"
            )
        shutil.copytree(source, version_root / surface)
    surface_names = list(SURFACES)
    (version_root / "index.html").write_text(
        render_version_index(version, surface_names), encoding="utf-8"
    )
    files = {
        path.relative_to(version_root).as_posix(): digest(path)
        for path in sorted(version_root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schemaVersion": 1,
        "component": "liblevenshtein",
        "version": version,
        "sourceRef": source_ref,
        "surfaces": surface_names,
        "sha256": files,
    }
    (version_root / "documentation-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive = ARTIFACT_ROOT / f"liblevenshtein-package-documentation-{version}.tar.gz"
    deterministic_archive(version_root, archive)
    print(f"package-documentation-site: built {archive.relative_to(ROOT)}")
    return archive


def safe_archive_members(archive: tarfile.TarFile) -> tuple[str, list[tarfile.TarInfo]]:
    members = archive.getmembers()
    if not members:
        fail("documentation archive is empty")
    versions: set[str] = set()
    names: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
            fail(f"unsafe or unversioned archive member: {member.name}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            fail(f"unsupported archive member type: {member.name}")
        if member.name in names:
            fail(f"duplicate archive member: {member.name}")
        names.add(member.name)
        versions.add(path.parts[0])
    if len(versions) != 1:
        fail(f"archive must contain exactly one version root: {sorted(versions)}")
    version = versions.pop()
    if SEMVER_RE.fullmatch(version) is None:
        fail(f"archive root is not a semantic version: {version}")
    return version, members


def verify_extracted_version(version_root: Path) -> dict[str, object]:
    manifest_path = version_root / "documentation-manifest.json"
    if not manifest_path.is_file():
        fail(f"archive lacks {manifest_path.name}: {version_root.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("component") != "liblevenshtein"
    ):
        fail(f"invalid documentation manifest identity: {version_root.name}")
    if manifest.get("version") != version_root.name:
        fail(f"documentation manifest/version mismatch: {version_root.name}")
    source_ref = manifest.get("sourceRef")
    expected_source = re.compile(
        rf"^v{re.escape(version_root.name)}(?:-release\.[1-9]\d*)?$"
    )
    if not isinstance(source_ref, str) or expected_source.fullmatch(source_ref) is None:
        fail(f"documentation manifest has an invalid source ref: {source_ref!r}")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        fail(f"documentation manifest lacks hashes: {version_root.name}")
    expected_files: set[str] = set()
    for relative, expected in hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            fail(f"invalid documentation hash entry: {version_root.name}")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative == "documentation-manifest.json"
        ):
            fail(f"unsafe documentation hash entry: {version_root.name}/{relative}")
        expected_files.add(relative_path.as_posix())
        path = version_root / relative
        if not path.is_file() or digest(path) != expected:
            fail(f"documentation hash mismatch: {version_root.name}/{relative}")
    actual_files = {
        path.relative_to(version_root).as_posix()
        for path in version_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != expected_files:
        fail(
            f"documentation manifest/file-set mismatch: {version_root.name}; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    surfaces = manifest.get("surfaces")
    if (
        not isinstance(surfaces, list)
        or len(surfaces) != len(set(surfaces))
        or set(surfaces) != set(SURFACES)
    ):
        fail(
            f"documentation manifest does not contain every required surface: "
            f"{version_root.name}"
        )
    for surface in surfaces:
        if not isinstance(surface, str) or surface not in SURFACES:
            fail(f"documentation manifest names an unknown surface: {surface!r}")
        required_index = SURFACES[surface][1]
        if not (version_root / surface / required_index).is_file():
            fail(
                f"documentation surface lacks its entry point: "
                f"{version_root.name}/{surface}/{required_index}"
            )
    return manifest


def assemble_site(archives_root: Path) -> None:
    archives = sorted(archives_root.glob("*.tar.gz"))
    if not archives:
        fail(f"no documentation archives found in {archives_root}")
    clean_directory(SITE_ROOT)
    versions: list[str] = []
    for archive_path in archives:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            version, members = safe_archive_members(archive)
            if version in versions or (SITE_ROOT / version).exists():
                fail(f"duplicate documentation version: {version}")
            archive.extractall(SITE_ROOT, members=members, filter="data")
        verify_extracted_version(SITE_ROOT / version)
        versions.append(version)
    versions.sort(key=semver_key, reverse=True)
    (SITE_ROOT / "index.html").write_text(render_site_index(versions), encoding="utf-8")
    (SITE_ROOT / "versions.json").write_text(
        json.dumps({"schemaVersion": 1, "versions": versions}, indent=2) + "\n",
        encoding="utf-8",
    )
    (SITE_ROOT / ".nojekyll").write_text("", encoding="utf-8")
    latest = SITE_ROOT / "latest"
    latest.mkdir()
    (latest / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url=../{html.escape(versions[0])}/">'
        f'<link rel="canonical" href="../{html.escape(versions[0])}/">',
        encoding="utf-8",
    )
    print(
        f"package-documentation-site: assembled {len(versions)} version(s) in "
        f"{SITE_ROOT.relative_to(ROOT)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="build the current version archive")
    assemble = subparsers.add_parser(
        "assemble", help="assemble all immutable archives into one Pages tree"
    )
    assemble.add_argument(
        "--archives",
        type=Path,
        default=ARTIFACT_ROOT,
        help="directory containing package-documentation .tar.gz archives",
    )
    args = parser.parse_args()
    if args.command == "build":
        build_archive()
    else:
        assemble_site(args.archives.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
