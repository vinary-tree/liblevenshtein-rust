#!/usr/bin/env python3
"""Stage reviewed Maven relocation POMs beside the canonical JVM artifact."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "release/version.json"
POM_NAMESPACE = "http://maven.apache.org/POM/4.0.0"


def coordinates(
    model: dict[str, object],
) -> tuple[str, str, str, str, str, str, tuple[str, ...]]:
    value = model.get("coordinates")
    if not isinstance(value, dict):
        raise SystemExit("release/version.json requires coordinates")
    group = value.get("mavenGroup")
    artifact = value.get("mavenArtifact")
    interop_group = value.get("interopMavenGroup")
    interop_artifact = value.get("interopMavenArtifact")
    legacy = value.get("legacyMavenGroups")
    version = (
        model.get("registries", {}).get("maven")
        if isinstance(model.get("registries"), dict)
        else None
    )
    interop_version = (
        model.get("dependencies", {}).get("vinary-tree-interop")
        if isinstance(model.get("dependencies"), dict)
        else None
    )
    if not all(
        isinstance(item, str)
        for item in (
            group,
            artifact,
            version,
            interop_group,
            interop_artifact,
            interop_version,
        )
    ):
        raise SystemExit("canonical Maven and interop coordinates must be strings")
    if (
        not isinstance(legacy, list)
        or not legacy
        or not all(isinstance(item, str) for item in legacy)
    ):
        raise SystemExit("legacyMavenGroups must be a non-empty string array")
    if group in legacy or len(legacy) != len(set(legacy)):
        raise SystemExit(
            "legacyMavenGroups must be unique and exclude the canonical group"
        )
    return (
        group,
        artifact,
        version,
        interop_group,
        interop_artifact,
        interop_version,
        tuple(legacy),
    )


def artifact_directory(root: Path, group: str, artifact: str, version: str) -> Path:
    return root.joinpath(*group.split("."), artifact, version)


def legacy_repository(root: Path, group: str) -> Path:
    """Return a namespace-isolated staging repository for one legacy group."""
    return root / group.replace(".", "-")


def relocation_pom(
    legacy_group: str, canonical_group: str, artifact: str, version: str
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="{POM_NAMESPACE}"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="{POM_NAMESPACE} https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>{legacy_group}</groupId>
  <artifactId>{artifact}</artifactId>
  <version>{version}</version>
  <packaging>pom</packaging>
  <name>liblevenshtein Maven coordinate relocation</name>
  <description>Relocates the historical liblevenshtein coordinate to the Vinary Tree namespace.</description>
  <url>https://github.com/vinary-tree/liblevenshtein-rust</url>
  <licenses>
    <license>
      <name>Apache License 2.0</name>
      <url>https://www.apache.org/licenses/LICENSE-2.0.txt</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <developers>
    <developer>
      <id>dylon</id>
      <name>Dylon Edwards</name>
      <email>dylon.devo@gmail.com</email>
    </developer>
  </developers>
  <scm>
    <connection>scm:git:https://github.com/vinary-tree/liblevenshtein-rust.git</connection>
    <developerConnection>scm:git:ssh://git@github.com/vinary-tree/liblevenshtein-rust.git</developerConnection>
    <url>https://github.com/vinary-tree/liblevenshtein-rust</url>
  </scm>
  <distributionManagement>
    <relocation>
      <groupId>{canonical_group}</groupId>
      <artifactId>{artifact}</artifactId>
      <version>{version}</version>
      <message>liblevenshtein is now maintained by Vinary Tree; update this dependency to {canonical_group}:{artifact}:{version}.</message>
    </relocation>
  </distributionManagement>
</project>
"""


def require_canonical_pom(
    staging: Path,
    group: str,
    artifact: str,
    version: str,
    interop_group: str,
    interop_artifact: str,
    interop_version: str,
) -> Path:
    path = (
        artifact_directory(staging, group, artifact, version)
        / f"{artifact}-{version}.pom"
    )
    if not path.is_file():
        raise SystemExit(f"canonical staged POM is missing: {path}")
    root = ET.parse(path).getroot()
    ns = {"m": POM_NAMESPACE}
    actual = (
        root.findtext("m:groupId", namespaces=ns),
        root.findtext("m:artifactId", namespaces=ns),
        root.findtext("m:version", namespaces=ns),
    )
    if actual != (group, artifact, version):
        raise SystemExit(
            "canonical staged POM coordinate mismatch: "
            f"expected {(group, artifact, version)}, got {actual}"
        )
    dependencies = {
        (
            dependency.findtext("m:groupId", namespaces=ns),
            dependency.findtext("m:artifactId", namespaces=ns),
            dependency.findtext("m:version", namespaces=ns),
        )
        for dependency in root.findall("m:dependencies/m:dependency", namespaces=ns)
    }
    expected_interop = (interop_group, interop_artifact, interop_version)
    if expected_interop not in dependencies:
        raise SystemExit(
            f"canonical staged POM is missing exact interop dependency {expected_interop}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_repository", type=Path)
    parser.add_argument("relocation_repositories", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing relocation POMs without writing them",
    )
    args = parser.parse_args()
    canonical_repository = args.canonical_repository.resolve()
    relocation_repositories = args.relocation_repositories.resolve()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    (
        group,
        artifact,
        version,
        interop_group,
        interop_artifact,
        interop_version,
        legacy_groups,
    ) = coordinates(model)
    require_canonical_pom(
        canonical_repository,
        group,
        artifact,
        version,
        interop_group,
        interop_artifact,
        interop_version,
    )

    for legacy_group in legacy_groups:
        expected = relocation_pom(legacy_group, group, artifact, version)
        destination = artifact_directory(
            legacy_repository(relocation_repositories, legacy_group),
            legacy_group,
            artifact,
            version,
        )
        path = destination / f"{artifact}-{version}.pom"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"staged Maven relocation is missing or stale: {path}")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
            print(
                f"staged {legacy_group}:{artifact}:{version} -> {group}:{artifact}:{version}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
