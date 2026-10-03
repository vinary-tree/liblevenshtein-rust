"""Pack and read back the complete Swift DocC site without renaming symbols.

GitHub's artifact uploader rejects colon-bearing filenames on Windows. DocC
legitimately emits names such as ``!=(_:_:).json``; the archive preserves
those exact names while exposing only a portable ``.tar.gz`` to the uploader.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target"
SITE = TARGET / "swift-docc"
ARCHIVE = TARGET / "swift-docc-artifacts" / "liblevenshtein-swift-docc.tar.gz"
READBACK = TARGET / "swift-docc-readback"
VERSION = json.loads((ROOT / "release/version.json").read_text(encoding="utf-8"))[
    "canonical"
]


def fail(message: str) -> None:
    raise SystemExit(f"swift-docc-archive: {message}")


def under_target(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(TARGET.resolve()) or resolved == TARGET.resolve():
        fail(f"output must be inside the repository target directory: {resolved}")
    return resolved


def member(name: str, content: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info, io.BytesIO(content)


def build(site: Path = SITE, archive: Path = ARCHIVE) -> None:
    site = under_target(site)
    archive = under_target(archive)
    if not (site / "index.html").is_file():
        fail("DocC site has no index.html")
    files: dict[str, bytes] = {}
    for path in sorted(site.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            fail(f"unsupported DocC site entry: {path}")
        if path.is_file():
            files[path.relative_to(site).as_posix()] = path.read_bytes()
    manifest = {
        "schemaVersion": 1,
        "version": VERSION,
        "files": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        metadata = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        info, stream = member("MANIFEST.json", metadata)
        tar.addfile(info, stream)
        for name, content in files.items():
            info, stream = member(f"site/{name}", content)
            tar.addfile(info, stream)
    print(f"swift-docc-archive: packed {len(files)} files for {VERSION}: {archive}")


def verify(archive: Path = ARCHIVE, destination: Path = READBACK) -> None:
    archive = under_target(archive)
    destination = under_target(destination)
    if not archive.is_file():
        fail(f"archive is absent: {archive}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    seen: dict[str, str] = {}
    manifest: dict | None = None
    with tarfile.open(archive, mode="r:gz") as tar:
        for entry in tar:
            if not entry.isfile() or entry.issym() or entry.islnk():
                fail(f"unsafe archive member: {entry.name}")
            body = tar.extractfile(entry)
            if body is None:
                fail(f"unreadable archive member: {entry.name}")
            content = body.read()
            if entry.name == "MANIFEST.json":
                if manifest is not None:
                    fail("duplicate archive manifest")
                manifest = json.loads(content)
                continue
            path = PurePosixPath(entry.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != "site"
                or len(path.parts) < 2
                or ".." in path.parts
            ):
                fail(f"unsafe archive path: {entry.name}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in seen:
                fail(f"duplicate archive member: {relative}")
            seen[relative] = hashlib.sha256(content).hexdigest()
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        fail("archive manifest is absent or invalid")
    if manifest.get("version") != VERSION or manifest.get("files") != seen:
        fail("archive manifest version or content digests differ from readback")
    if not (destination / "index.html").is_file():
        fail("readback lacks DocC index.html")
    print(f"swift-docc-archive: read back {len(seen)} exact files for {VERSION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
