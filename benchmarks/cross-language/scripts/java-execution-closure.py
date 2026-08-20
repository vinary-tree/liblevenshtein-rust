#!/usr/bin/env python3
"""Hash the complete execution closure of a generated Java launcher."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content_sha = file_sha256(path).encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(content_sha)
        count += 1
    return digest.hexdigest(), count


def closure(launcher: Path) -> dict[str, object]:
    launcher = launcher.resolve(strict=True)
    text = launcher.read_text(encoding="utf-8")
    executable_match = re.search(r"\bexec\s+([^\s\\]+)", text)
    if not executable_match:
        raise ValueError(f"cannot resolve Java executable from {launcher}")
    executable_name = executable_match.group(1)
    executable_text = shutil.which(executable_name)
    if executable_text is None:
        raise ValueError(f"Java executable is unavailable: {executable_name}")
    executable = Path(executable_text).resolve(strict=True)
    version_result = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=True
    )
    version = (version_result.stdout or version_result.stderr).strip()

    classpath_match = re.search(
        r"\bcat\s+[\"']?([^\"')\s]+runtime-classpath\.txt)", text
    )
    if not classpath_match:
        raise ValueError(f"cannot resolve runtime classpath file from {launcher}")
    classpath_file = Path(classpath_match.group(1)).resolve(strict=True)
    classpath_text = classpath_file.read_text(encoding="utf-8").strip()
    entries = []
    for raw_entry in classpath_text.split(os.pathsep):
        if not raw_entry:
            continue
        path = Path(raw_entry).resolve(strict=False)
        if path.is_dir():
            digest, count = tree_sha256(path)
            entries.append(
                {
                    "path": str(path),
                    "kind": "directory-tree",
                    "sha256": digest,
                    "file_count": count,
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": str(path),
                    "kind": "file",
                    "sha256": file_sha256(path),
                    "file_count": 1,
                }
            )
        elif not path.exists():
            marker = b"missing\0" + str(path).encode()
            entries.append(
                {
                    "path": str(path),
                    "kind": "missing",
                    "sha256": hashlib.sha256(marker).hexdigest(),
                    "file_count": 0,
                }
            )
        else:
            raise ValueError(f"unsupported classpath entry: {path}")

    record: dict[str, object] = {
        "launcher": str(launcher),
        "launcher_sha256": file_sha256(launcher),
        "java_executable": str(executable),
        "java_executable_sha256": file_sha256(executable),
        "java_version": version,
        "classpath_file": str(classpath_file),
        "classpath_file_sha256": file_sha256(classpath_file),
        "classpath_entries": entries,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["closure_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: java-execution-closure.py LAUNCHER", file=sys.stderr)
        return 2
    try:
        print(json.dumps(closure(Path(sys.argv[1])), sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"java-execution-closure: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
