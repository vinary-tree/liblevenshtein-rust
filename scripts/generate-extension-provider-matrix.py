#!/usr/bin/env python3
"""Generate the exhaustive llattice/lling-llang host-extension matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bindings" / "conformance" / "extension-provider-model.json"
FAMILY_MODEL_PATH = ROOT / "bindings" / "conformance" / "family-bindings.json"
MATRIX_PATH = ROOT / "bindings" / "conformance" / "extension-provider-matrix.tsv"

VALID_CLASSIFICATIONS = {
    "bounded-batch-protocol",
    "declared-law-marker",
    "derived-adapter",
    "direct-capability-vtable",
    "high-level-operation-interface",
    "resource-cursor-protocol",
    "reviewed-rust-only-proof",
}
TRAIT_PATTERN = re.compile(
    r"(?m)^[ \t]*pub[ \t]+(?:(?:unsafe|auto)[ \t]+)*"
    r"trait[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)


def fail(message: str) -> None:
    raise SystemExit(f"extension-provider matrix error: {message}")


def clean(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\t\r\n")
    ):
        fail(f"{field} must be a non-empty single-line string")
    return value


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot load {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def without_comments(source: str) -> str:
    """Mask nested Rust comments and strings while retaining source lines."""
    output = list(source)
    index = 0
    block_depth = 0
    line_comment = False
    string_literal = False
    raw_hashes: str | None = None
    escaped = False
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if current == "\n":
                line_comment = False
            else:
                output[index] = " "
            index += 1
            continue
        if block_depth:
            if current == "/" and following == "*":
                output[index] = output[index + 1] = " "
                block_depth += 1
                index += 2
            elif current == "*" and following == "/":
                output[index] = output[index + 1] = " "
                block_depth -= 1
                index += 2
            else:
                if current != "\n":
                    output[index] = " "
                index += 1
            continue
        if raw_hashes is not None:
            terminator = '"' + raw_hashes
            if source.startswith(terminator, index):
                for offset in range(len(terminator)):
                    output[index + offset] = " "
                index += len(terminator)
                raw_hashes = None
            else:
                if current != "\n":
                    output[index] = " "
                index += 1
            continue
        if string_literal:
            if current != "\n":
                output[index] = " "
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                string_literal = False
            index += 1
            continue
        raw_match = re.match(r'(?:br|r)(?P<hashes>#{0,255})"', source[index:])
        if raw_match is not None:
            literal_prefix = raw_match.group(0)
            for offset in range(len(literal_prefix)):
                output[index + offset] = " "
            raw_hashes = raw_match.group("hashes")
            index += len(literal_prefix)
        elif current == '"':
            output[index] = " "
            string_literal = True
            index += 1
        elif current == "/" and following == "/":
            output[index] = output[index + 1] = " "
            line_comment = True
            index += 2
        elif current == "/" and following == "*":
            output[index] = output[index + 1] = " "
            block_depth = 1
            index += 2
        else:
            index += 1
    return "".join(output)


def resolve_roots(
    workspace_root: Path, model: dict, environment: dict[str, str]
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    sections = ("projects", "evidenceProjects")
    for section in sections:
        entries = model.get(section, {})
        if not isinstance(entries, dict):
            fail(f"{section} must be an object")
        for project, raw_specification in entries.items():
            if project in roots:
                fail(f"duplicate project root {project}")
            if not isinstance(raw_specification, dict):
                fail(f"{section}.{project} must be an object")
            relative = Path(clean(raw_specification.get("root"), f"{project}.root"))
            if relative.is_absolute():
                fail(
                    f"{project}.root must be repository-relative; use its "
                    "rootEnvironment override for a machine-specific checkout"
                )
            variable = clean(
                raw_specification.get("rootEnvironment"),
                f"{project}.rootEnvironment",
            )
            configured = environment.get(variable)
            root = (
                Path(configured) if configured else workspace_root / relative
            ).resolve()
            if not root.is_dir():
                fail(f"{project} root does not exist: {root}")
            roots[project] = root
    return roots


def discover_traits(project: str, project_root: Path) -> dict[str, tuple[str, str]]:
    source_root = project_root / "src"
    if not source_root.is_dir():
        fail(f"{project} has no src directory: {source_root}")
    discovered: dict[str, tuple[str, str]] = {}
    for path in sorted(source_root.rglob("*.rs")):
        relative = path.relative_to(project_root).as_posix()
        source = without_comments(path.read_text(encoding="utf-8"))
        for match in TRAIT_PATTERN.finditer(source):
            name = match.group("name")
            key = f"{project}:{relative}:{name}"
            if key in discovered:
                fail(f"duplicate public trait declaration {key}")
            discovered[key] = (relative, name)
    return discovered


def classification_index(model: dict) -> dict[str, str]:
    raw = model.get("classifications")
    if not isinstance(raw, dict):
        fail("classifications must be an object")
    unknown = set(raw) - VALID_CLASSIFICATIONS
    if unknown:
        fail(f"unknown classifications: {sorted(unknown)}")
    index: dict[str, str] = {}
    for classification, keys in raw.items():
        if not isinstance(keys, list):
            fail(f"classification {classification} must be an array")
        for raw_key in keys:
            key = clean(raw_key, f"classifications.{classification}")
            if key in index:
                fail(f"{key} is classified as both {index[key]} and {classification}")
            index[key] = classification
    return index


def evidence_reference(
    raw: object, roots: dict[str, Path], context: str
) -> tuple[str, Path]:
    if not isinstance(raw, dict) or set(raw) != {"project", "path"}:
        fail(f"{context} must contain exactly project and path")
    project = clean(raw.get("project"), f"{context}.project")
    relative_text = clean(raw.get("path"), f"{context}.path")
    if project not in roots:
        fail(f"{context} names unknown evidence project {project}")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"{context}.path must stay repository-relative: {relative_text}")
    path = (roots[project] / relative).resolve()
    try:
        path.relative_to(roots[project])
    except ValueError:
        fail(f"{context}.path leaves {project}: {relative_text}")
    if not path.is_file():
        fail(f"{context} is missing: {path}")
    return f"{project}:{relative.as_posix()}", path


def capability_tokens(capability: str) -> tuple[str, ...]:
    return tuple(part for part in capability.split(".") if part)


def render() -> tuple[str, Counter[str], int]:
    model = load_json(MODEL_PATH)
    family = load_json(FAMILY_MODEL_PATH)
    if model.get("schemaVersion") != 1:
        fail("unsupported schemaVersion")
    expected_model_keys = {
        "schemaVersion",
        "projects",
        "evidenceProjects",
        "classifications",
        "rustOnlyProofs",
        "languageProfiles",
        "implementations",
    }
    if set(model) != expected_model_keys:
        fail(
            "model keys differ: "
            f"missing={sorted(expected_model_keys - set(model))}, "
            f"unknown={sorted(set(model) - expected_model_keys)}"
        )
    roots = resolve_roots(ROOT, model, dict(os.environ))
    projects = model.get("projects")
    assert isinstance(projects, dict)

    discovered: dict[str, tuple[str, str]] = {}
    for project in projects:
        discovered.update(discover_traits(project, roots[project]))
    classified = classification_index(model)
    missing_classifications = set(discovered) - set(classified)
    stale_classifications = set(classified) - set(discovered)
    if missing_classifications or stale_classifications:
        fail(
            "trait classifications differ from the authoritative Rust surface: "
            f"unclassified={sorted(missing_classifications)}, "
            f"stale={sorted(stale_classifications)}"
        )

    rust_only_proofs = model.get("rustOnlyProofs", {})
    if not isinstance(rust_only_proofs, dict):
        fail("rustOnlyProofs must be an object")
    rust_only_traits = {
        key
        for key, classification in classified.items()
        if classification == "reviewed-rust-only-proof"
    }
    if set(rust_only_proofs) != rust_only_traits:
        fail(
            "rustOnlyProofs must cover exactly the reviewed Rust-only traits: "
            f"missing={sorted(rust_only_traits - set(rust_only_proofs))}, "
            f"stale={sorted(set(rust_only_proofs) - rust_only_traits)}"
        )
    validated_proofs: dict[str, str] = {}
    for key, raw_reference in rust_only_proofs.items():
        reference = clean(raw_reference, f"rustOnlyProofs.{key}")
        relative = Path(reference.split("#", 1)[0])
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"rustOnlyProofs.{key} must stay repository-relative")
        proof_path = (ROOT / relative).resolve()
        try:
            proof_path.relative_to(ROOT)
        except ValueError:
            fail(f"rustOnlyProofs.{key} leaves this repository")
        if not proof_path.is_file():
            fail(f"rustOnlyProofs.{key} is missing: {proof_path}")
        validated_proofs[key] = reference

    raw_languages = family.get("languages")
    if not isinstance(raw_languages, list):
        fail("family-bindings.json languages must be an array")
    languages: dict[str, str] = {}
    for entry in raw_languages:
        if not isinstance(entry, dict):
            fail("each family language must be an object")
        language = clean(entry.get("id"), "family language id")
        if language in languages:
            fail(f"duplicate family language {language}")
        languages[language] = clean(entry.get("hostIdioms"), f"{language}.hostIdioms")
    if "rust" not in languages:
        fail("family language inventory omits Rust")

    implementations = model.get("implementations", {})
    if not isinstance(implementations, dict):
        fail("implementations must be an object")
    language_profiles = model.get("languageProfiles", {})
    if not isinstance(language_profiles, dict):
        fail("languageProfiles must be an object")
    for profile, entries in language_profiles.items():
        clean(profile, "languageProfiles key")
        if not isinstance(entries, dict) or not entries:
            fail(f"languageProfiles.{profile} must be a non-empty object")
        unknown_languages = set(entries) - set(languages)
        if unknown_languages:
            fail(
                f"languageProfiles.{profile} names unknown languages: "
                f"{sorted(unknown_languages)}"
            )
    unknown_implementations = set(implementations) - set(discovered)
    if unknown_implementations:
        fail(f"implementations name unknown traits: {sorted(unknown_implementations)}")

    rows: list[str] = []
    status_counts: Counter[str] = Counter()
    used_profiles: set[str] = set()
    for key in sorted(discovered):
        project, _separator, remainder = key.partition(":")
        source, _separator, trait = remainder.rpartition(":")
        classification = classified[key]
        raw_implementation = implementations.get(key, {})
        if not isinstance(raw_implementation, dict):
            fail(f"implementation {key} must be an object")
        unknown_keys = set(raw_implementation) - {
            "abiCapabilities",
            "languages",
            "profiles",
        }
        if unknown_keys:
            fail(f"implementation {key} has unknown keys: {sorted(unknown_keys)}")
        raw_capabilities = raw_implementation.get("abiCapabilities", [])
        raw_support = raw_implementation.get("languages", {})
        raw_profiles = raw_implementation.get("profiles", [])
        if (
            not isinstance(raw_capabilities, list)
            or not isinstance(raw_support, dict)
            or not isinstance(raw_profiles, list)
        ):
            fail(
                f"implementation {key} has malformed capabilities, profiles, "
                "or languages"
            )
        capabilities = [
            clean(value, f"{key}.abiCapabilities") for value in raw_capabilities
        ]
        profiles = [clean(value, f"{key}.profiles") for value in raw_profiles]
        if len(capabilities) != len(set(capabilities)):
            fail(f"implementation {key} repeats an ABI capability")
        if len(profiles) != len(set(profiles)):
            fail(f"implementation {key} repeats a language profile")
        merged_support: dict[str, object] = {}
        for profile in profiles:
            if profile not in language_profiles:
                fail(f"implementation {key} names unknown profile {profile}")
            overlap = set(merged_support) & set(language_profiles[profile])
            if overlap:
                fail(
                    f"implementation {key} profiles overlap for languages: "
                    f"{sorted(overlap)}"
                )
            merged_support.update(language_profiles[profile])
            used_profiles.add(profile)
        overlap = set(merged_support) & set(raw_support)
        if overlap:
            fail(
                f"implementation {key} overrides profiled languages: {sorted(overlap)}"
            )
        merged_support.update(raw_support)
        unknown_languages = set(merged_support) - set(languages)
        if unknown_languages:
            fail(
                f"implementation {key} names unknown languages: {sorted(unknown_languages)}"
            )
        if "rust" in merged_support:
            fail(f"implementation {key} must not override native Rust support")
        if classification == "reviewed-rust-only-proof" and raw_implementation:
            fail(f"Rust-only trait {key} cannot declare a foreign implementation")

        evidence_corpus: list[str] = []
        prepared_support: dict[str, tuple[str, str, list[str]]] = {}
        for language, raw_entry in merged_support.items():
            if not isinstance(raw_entry, dict):
                fail(f"{key}.{language} must be an object")
            unknown_entry_keys = set(raw_entry) - {"status", "scope", "evidence"}
            if unknown_entry_keys or not {"status", "evidence"} <= set(raw_entry):
                fail(
                    f"{key}.{language} must contain status and evidence, with "
                    f"only optional scope; unknown={sorted(unknown_entry_keys)}"
                )
            status = clean(raw_entry.get("status"), f"{key}.{language}.status")
            if status not in {"complete", "partial", "abi-available"}:
                fail(
                    f"{key}.{language} implementation status must be complete, "
                    f"partial, or abi-available, not {status}"
                )
            raw_scope = raw_entry.get("scope")
            if status == "partial":
                scope = clean(raw_scope, f"{key}.{language}.scope")
            elif raw_scope is None:
                scope = (
                    "full-modeled-provider-surface"
                    if status == "complete"
                    else "raw-c-capability-without-idiomatic-facade"
                )
            else:
                scope = clean(raw_scope, f"{key}.{language}.scope")
            raw_evidence = raw_entry.get("evidence")
            if not isinstance(raw_evidence, list) or not raw_evidence:
                fail(f"{key}.{language} must provide non-empty evidence")
            references: list[str] = []
            for offset, raw_reference in enumerate(raw_evidence):
                reference, path = evidence_reference(
                    raw_reference, roots, f"{key}.{language}.evidence[{offset}]"
                )
                references.append(reference)
                evidence_corpus.append(path.read_text(encoding="utf-8"))
            prepared_support[language] = (status, scope, references)

        corpus = "\n".join(evidence_corpus)
        for capability in capabilities:
            missing_tokens = [
                token for token in capability_tokens(capability) if token not in corpus
            ]
            if missing_tokens:
                fail(
                    f"{key} capability {capability} lacks evidence tokens "
                    f"{missing_tokens}"
                )

        for language, host_idioms in languages.items():
            if classification == "reviewed-rust-only-proof":
                status = "inapplicable"
                scope = "library-owned-type-level-access-marker"
                evidence = [f"proof:{validated_proofs[key]}"]
                next_work = "-"
            elif language == "rust":
                status = "complete"
                scope = "full-native-trait-surface"
                evidence = [f"{project}:{source}"]
                next_work = "-"
            elif language in prepared_support:
                status, scope, evidence = prepared_support[language]
                if status == "abi-available":
                    next_work = "add-an-idiomatic-safe-provider-facade"
                elif status == "partial":
                    next_work = "complete-the-generic-provider-surface"
                else:
                    next_work = "-"
            else:
                status = "missing"
                scope = "-"
                evidence = []
                next_work = f"implement-{classification}"
            status_counts[status] += 1
            rows.append(
                "\t".join(
                    (
                        project,
                        source,
                        trait,
                        classification,
                        ",".join(capabilities) if capabilities else "-",
                        language,
                        host_idioms,
                        status,
                        scope,
                        json.dumps(evidence, separators=(",", ":"))
                        if evidence
                        else "-",
                        next_work,
                    )
                )
            )

    unused_profiles = set(language_profiles) - used_profiles
    if unused_profiles:
        fail(f"unused language profiles: {sorted(unused_profiles)}")

    header = (
        "project\tsource\ttrait\tprovider_translation\tabi_capabilities\tlanguage\t"
        "host_idioms\tprovider_surface_status\tsupport_scope\tevidence\t"
        "next_required_work"
    )
    return "\n".join((header, *rows, "")), status_counts, len(discovered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="reject a stale committed matrix"
    )
    args = parser.parse_args()
    matrix, status_counts, trait_count = render()
    if args.check:
        if (
            not MATRIX_PATH.is_file()
            or MATRIX_PATH.read_text(encoding="utf-8") != matrix
        ):
            fail(
                "committed matrix is stale; rerun "
                "scripts/generate-extension-provider-matrix.py"
            )
    else:
        MATRIX_PATH.write_text(matrix, encoding="utf-8")
    summary = ", ".join(
        f"{status}={status_counts[status]}" for status in sorted(status_counts)
    )
    print(
        f"extension-provider matrix: {trait_count} traits x "
        f"{sum(status_counts.values()) // trait_count} languages; {summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
