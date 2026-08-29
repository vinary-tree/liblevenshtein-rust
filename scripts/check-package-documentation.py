#!/usr/bin/env python3
"""Validate package-documentation ownership, evidence, and public readbacks."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "release" / "package-documentation.json"
RELEASE_PATH = ROOT / "release" / "version.json"
ALLOWED_RELEASE_STATES = {"released", "pending-publication", "candidate-only"}
ALLOWED_DESTINATION_STATES = {"verified", "build-only", "missing", "deferred"}
REQUIRED_DESTINATION_KINDS = {"package-guide", "api-reference"}
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|STUB)\b", re.IGNORECASE)
MAX_READBACK_BYTES = 16 * 1024 * 1024


def fail(message: str) -> None:
    raise SystemExit(f"package-documentation: {message}")


def text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\t"):
        fail(f"{field} must be a non-empty single-line string")
    if PLACEHOLDER_RE.search(value):
        fail(f"{field} contains a placeholder marker")
    return value


def string_list(value: object, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        fail(f"{field} must be a{' possibly empty' if allow_empty else ' non-empty'} array")
    result = [text(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        fail(f"{field} contains duplicates")
    return result


def https_url(value: object, field: str) -> str:
    result = text(value, field)
    if not result.startswith("https://"):
        fail(f"{field} must use HTTPS")
    return result


def iso_date(value: object, field: str) -> dt.date:
    result = text(value, field)
    if DATE_RE.fullmatch(result) is None:
        fail(f"{field} must be an ISO-8601 calendar date")
    try:
        return dt.date.fromisoformat(result)
    except ValueError:
        fail(f"{field} is not a valid calendar date")


def evidence_paths(value: object, field: str) -> list[str]:
    result = string_list(value, field)
    for relative in result:
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            fail(f"{field} leaves the repository: {relative}")
        if not candidate.exists():
            fail(f"{field} is missing: {relative}")
    return result


def read_url(url: str, cache: dict[str, bytes]) -> bytes:
    cached = cache.get(url)
    if cached is not None:
        return cached
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "vinary-tree-package-documentation-gate/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 200:
                fail(f"public readback returned HTTP {response.status}: {url}")
            body = response.read(MAX_READBACK_BYTES + 1)
            if len(body) > MAX_READBACK_BYTES:
                fail(f"public readback exceeds the 16 MiB verification ceiling: {url}")
            if response.headers.get("Content-Encoding", "").casefold() == "gzip":
                body = gzip.decompress(body)
                if len(body) > MAX_READBACK_BYTES:
                    fail(
                        "decompressed public readback exceeds the 16 MiB "
                        f"verification ceiling: {url}"
                    )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        fail(f"public readback failed for {url}: {error}")
    cache[url] = body
    return body


def check_markers(url: str, markers: list[str], cache: dict[str, bytes]) -> None:
    body = read_url(url, cache)
    for marker in markers:
        if marker.encode("utf-8") not in body:
            fail(f"public readback {url} is missing marker {marker!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public",
        action="store_true",
        help="fetch and verify released registry pages and verified documentation",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="reject any released package whose guide or API reference is not verified",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="limit public/strict checks to one package id; repeat as needed",
    )
    args = parser.parse_args()

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if model.get("schemaVersion") != 1:
        fail("unsupported schemaVersion")
    if model.get("component") != release.get("component"):
        fail("component disagrees with release/version.json")
    canonical = text(model.get("canonicalVersion"), "canonicalVersion")
    if canonical != release.get("canonical"):
        fail("canonicalVersion disagrees with release/version.json")
    if text(model.get("sourceRef"), "sourceRef") != release["publication"]["sourceTag"]:
        fail("sourceRef disagrees with the immutable release source tag")
    https_url(model.get("sourceRepository"), "sourceRepository")
    observed_at = iso_date(model.get("observedAt"), "observedAt")
    if observed_at > dt.date.today():
        fail("observedAt cannot be in the future")

    packages = model.get("packages")
    if not isinstance(packages, list) or not packages:
        fail("packages must be a non-empty array")
    selected = set(args.package)
    seen: set[str] = set()
    incomplete: list[str] = []
    verified_destinations = 0
    public_cache: dict[str, bytes] = {}

    for index, package in enumerate(packages):
        field = f"packages[{index}]"
        if not isinstance(package, dict):
            fail(f"{field} must be an object")
        identifier = text(package.get("id"), f"{field}.id")
        if identifier in seen:
            fail(f"duplicate package id {identifier}")
        seen.add(identifier)
        string_list(package.get("languages"), f"{field}.languages")
        text(package.get("ecosystem"), f"{field}.ecosystem")
        text(package.get("coordinate"), f"{field}.coordinate")
        registry_version = text(package.get("registryVersion"), f"{field}.registryVersion")
        release_state = text(package.get("releaseState"), f"{field}.releaseState")
        if release_state not in ALLOWED_RELEASE_STATES:
            fail(f"{field}.releaseState is invalid: {release_state}")
        evidence_paths(package.get("sourceEvidence"), f"{field}.sourceEvidence")

        registry = package.get("registryReadback")
        if release_state == "released":
            if not isinstance(registry, dict):
                fail(f"{field}.registryReadback is required for a released package")
            registry_url = https_url(registry.get("url"), f"{field}.registryReadback.url")
            registry_markers = string_list(
                registry.get("markers"),
                f"{field}.registryReadback.markers",
                allow_empty=True,
            )
            if not any(spelling in registry_url for spelling in {registry_version, registry_version.replace(".", "%2e")}):
                # Some registry indexes are coordinate-specific but not version-specific.
                if registry_markers == [] or registry_version not in registry_markers:
                    fail(
                        f"{field}.registryReadback must pin {registry_version} in its URL or markers"
                    )
            if args.public and (not selected or identifier in selected):
                check_markers(registry_url, registry_markers, public_cache)
        else:
            text(package.get("releaseProof"), f"{field}.releaseProof")
            if release_state == "pending-publication":
                incomplete.append(f"{identifier}:package:pending-publication")

        destinations = package.get("destinations")
        if not isinstance(destinations, list) or not destinations:
            fail(f"{field}.destinations must be a non-empty array")
        destination_kinds: set[str] = set()
        for destination_index, destination in enumerate(destinations):
            destination_field = f"{field}.destinations[{destination_index}]"
            if not isinstance(destination, dict):
                fail(f"{destination_field} must be an object")
            kind = text(destination.get("kind"), f"{destination_field}.kind")
            if kind in destination_kinds:
                fail(f"{field} declares duplicate destination kind {kind}")
            destination_kinds.add(kind)
            text(destination.get("service"), f"{destination_field}.service")
            state = text(destination.get("state"), f"{destination_field}.state")
            if state not in ALLOWED_DESTINATION_STATES:
                fail(f"{destination_field}.state is invalid: {state}")
            if state == "verified":
                verified_destinations += 1
                url = https_url(destination.get("url"), f"{destination_field}.url")
                verified_at = iso_date(
                    destination.get("verifiedAt"), f"{destination_field}.verifiedAt"
                )
                if verified_at > observed_at:
                    fail(f"{destination_field}.verifiedAt is later than observedAt")
                markers = string_list(
                    destination.get("markers"),
                    f"{destination_field}.markers",
                    allow_empty=True,
                )
                readback_url = https_url(
                    destination.get("readbackUrl", url),
                    f"{destination_field}.readbackUrl",
                )
                if args.public and (not selected or identifier in selected):
                    check_markers(readback_url, markers, public_cache)
            elif state == "build-only":
                text(destination.get("reason"), f"{destination_field}.reason")
                evidence_paths(
                    destination.get("buildEvidence"),
                    f"{destination_field}.buildEvidence",
                )
            else:
                text(destination.get("reason"), f"{destination_field}.reason")
                if destination.get("verifiedAt") is not None:
                    fail(f"{destination_field} cannot be unverified and have verifiedAt")
            if release_state == "released" and state != "verified":
                incomplete.append(f"{identifier}:{kind}:{state}")

        missing_kinds = REQUIRED_DESTINATION_KINDS - destination_kinds
        if missing_kinds:
            fail(f"{field} lacks destination kinds: {sorted(missing_kinds)}")

    unknown = selected - seen
    if unknown:
        fail(f"unknown --package ids: {sorted(unknown)}")
    if args.require_complete:
        relevant = [
            item for item in incomplete if not selected or item.split(":", 1)[0] in selected
        ]
        if relevant:
            fail(
                f"{len(relevant)} release-documentation obligations remain incomplete; "
                f"first: {', '.join(relevant[:10])}"
            )

    print(
        f"package-documentation: {len(packages)} package surfaces; "
        f"{verified_destinations} verified destinations; "
        f"{len(incomplete)} release-documentation obligations incomplete"
    )
    if args.public:
        print(f"package-documentation: {len(public_cache)} public URLs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
