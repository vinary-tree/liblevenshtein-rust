#!/usr/bin/env python3
"""Generate the cross-project language/capability completeness inventory."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bindings" / "conformance" / "family-bindings.json"
MATRIX_PATH = ROOT / "bindings" / "conformance" / "family-completeness-matrix.tsv"
VALID_CELL_STATES = {
    "audit-required",
    "complete",
    "inapplicable",
    "missing",
    "review-required",
}
REQUIRED_DOCUMENTATION_TOPICS = (
    "overview",
    "installation",
    "quick-start",
    "common-usage",
    "intended-usage",
    "api-reference",
    "semantics",
    "errors",
    "lifecycle",
    "ownership",
    "concurrency",
    "collections-iterators",
    "snapshots",
    "zero-copy-batching",
    "examples",
    "migration",
    "compatibility",
    "performance",
    "security",
    "release",
)
ALLOWED_OVERRIDE_KEYS = {
    "applicabilityProof",
    "benchmark",
    "conformance",
    "documentationTopics",
    "evidence",
    "freshConsumer",
    "state",
}
PROJECT_ROOT_ENVIRONMENTS = {
    "vinary-tree-interop": "VINARY_TREE_INTEROP_ROOT",
    "llattice": "LLATTICE_ROOT",
    "libdictenstein": "LIBDICTENSTEIN_ROOT",
    "lling-llang": "LLING_LLANG_ROOT",
    "duallity": "DUALLITY_ROOT",
    "javascript-runtime": "VINARY_TREE_JAVASCRIPT_RUNTIME_ROOT",
    "liblevenshtein-npm": "LIBLEVENSHTEIN_NPM_ROOT",
}
DISCOVERABLE_BINDING_DIRECTORIES = {
    "c": {"c"},
    "cpp": {"cpp"},
    "python": {"python"},
    "jvm": {"java", "kotlin", "scala"},
    "clojure": {"clojure"},
    "dotnet": {"csharp", "fsharp"},
    "go": {"go"},
    "javascript": {"javascript", "typescript"},
    "swift": {"swift"},
    "ruby": {"ruby"},
    "fortran": {"fortran"},
    "ocaml": {"ocaml"},
    "haskell": {"haskell"},
    "lua": {"lua"},
    "raku": {"raku"},
    "julia": {"julia"},
}


def fail(message: str) -> None:
    raise SystemExit(f"family binding inventory error: {message}")


def clean(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\t\r\n")
    ):
        fail(f"{field} must be a non-empty single-line string")
    return value


def dimension_state(name: str, state: str, override: dict, cell_id: str) -> str:
    default = "missing" if state == "missing" else state
    value = clean(override.get(name, default), f"{cell_id}.{name}")
    if value not in VALID_CELL_STATES:
        fail(f"{cell_id}.{name} has invalid state {value}")
    return value


def aggregate_documentation(states: list[str]) -> str:
    """Derive a truthful summary; no aggregate override may hide a topic gap."""
    if all(state == "inapplicable" for state in states):
        return "inapplicable"
    if all(state in {"complete", "inapplicable"} for state in states):
        return "complete"
    for state in ("missing", "review-required", "audit-required"):
        if state in states:
            return state
    fail("documentation topic states could not be aggregated")


def discover_binding_languages(project_root: Path) -> set[str]:
    """Return language claims made visible by conventional binding directories."""
    bindings_root = project_root / "bindings"
    if not bindings_root.is_dir():
        return set()

    discovered: set[str] = set()
    for directory, languages in DISCOVERABLE_BINDING_DIRECTORIES.items():
        if (bindings_root / directory).is_dir():
            discovered.update(languages)
    if (bindings_root / "javascript" / "cljs").is_dir():
        discovered.add("clojurescript")
    return discovered


def documentation_topic(
    topic_id: str,
    state: str,
    override: object,
    cell_id: str,
    project_root: Path,
) -> tuple[str, str]:
    """Validate one topic's state and the evidence needed to advance it."""
    default = "missing" if state == "missing" else state
    if override is None:
        if default == "complete":
            fail(
                f"{cell_id}.documentationTopics.{topic_id} is complete "
                "without documentation evidence"
            )
        return default, "-"
    if not isinstance(override, dict):
        fail(f"{cell_id}.documentationTopics.{topic_id} must be an object")
    unknown = set(override) - {"evidence", "proof", "state"}
    if unknown:
        fail(
            f"{cell_id}.documentationTopics.{topic_id} has unknown keys: "
            f"{sorted(unknown)}"
        )
    topic_cell_id = f"{cell_id}.documentationTopics.{topic_id}"
    topic_state = dimension_state("state", default, override, topic_cell_id)
    raw_evidence = override.get("evidence", [])
    if not isinstance(raw_evidence, list):
        fail(f"{topic_cell_id}.evidence must be an array")
    evidence = [
        clean(reference, f"{topic_cell_id}.evidence") for reference in raw_evidence
    ]
    if len(evidence) != len(set(evidence)):
        fail(f"{topic_cell_id}.evidence contains duplicates")
    for reference in evidence:
        if reference.startswith("https://"):
            continue
        if "://" in reference:
            fail(f"{topic_cell_id}.evidence must use HTTPS: {reference}")
        relative = reference.split("#", 1)[0]
        if not relative:
            fail(f"{topic_cell_id}.evidence must name a file: {reference}")
        evidence_path = (project_root / relative).resolve()
        try:
            evidence_path.relative_to(project_root)
        except ValueError:
            fail(f"{topic_cell_id}.evidence leaves the project: {reference}")
        if not evidence_path.is_file():
            fail(f"{topic_cell_id}.evidence is missing: {evidence_path}")
    if topic_state == "complete" and not evidence:
        fail(f"{topic_cell_id} is complete without documentation evidence")
    proof = override.get("proof")
    if topic_state == "inapplicable":
        proof = clean(proof, f"{topic_cell_id}.proof")
        proof_path = (ROOT / proof.split("#", 1)[0]).resolve()
        if not proof_path.is_file():
            fail(f"{topic_cell_id} applicability proof is missing: {proof_path}")
        evidence.append(f"proof:{proof}")
    elif proof is not None:
        fail(f"{topic_cell_id}.proof is valid only for an inapplicable topic")
    return (
        topic_state,
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        if evidence
        else "-",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="reject a stale committed matrix"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="reject every missing, review-required, or audit-required cell",
    )
    parser.add_argument(
        "--require-documentation-complete",
        action="store_true",
        help="reject every unfinished project/language/capability/topic tuple",
    )
    args = parser.parse_args()

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("schemaVersion") != 2:
        fail("unsupported schemaVersion")
    languages = model.get("languages")
    projects = model.get("projects")
    documentation_topics = model.get("documentationTopics")
    overrides = model.get("cellOverrides", {})
    if (
        not isinstance(languages, list)
        or not isinstance(projects, list)
        or not isinstance(documentation_topics, list)
        or not isinstance(overrides, dict)
    ):
        fail(
            "languages/projects/documentationTopics must be arrays and "
            "cellOverrides must be an object"
        )

    topic_ids: list[str] = []
    for topic in documentation_topics:
        if not isinstance(topic, dict):
            fail("each documentation topic must be an object")
        identifier = clean(topic.get("id"), "documentationTopic.id")
        clean(topic.get("description"), f"documentationTopic.{identifier}.description")
        if not re.fullmatch(r"[a-z]+(?:-[a-z]+)*", identifier):
            fail(f"invalid documentation topic id {identifier}")
        if identifier in topic_ids:
            fail(f"duplicate documentation topic {identifier}")
        topic_ids.append(identifier)
    if tuple(topic_ids) != REQUIRED_DOCUMENTATION_TOPICS:
        fail(
            "documentationTopics must contain the complete canonical sequence: "
            + ", ".join(REQUIRED_DOCUMENTATION_TOPICS)
        )

    language_by_id: dict[str, dict] = {}
    for language in languages:
        if not isinstance(language, dict):
            fail("each language must be an object")
        identifier = clean(language.get("id"), "language.id")
        if identifier in language_by_id:
            fail(f"duplicate language {identifier}")
        clean(language.get("packageManager"), f"{identifier}.packageManager")
        clean(language.get("hostIdioms"), f"{identifier}.hostIdioms")
        language_by_id[identifier] = language
    required_campaign_languages = {"rust", "raku", "julia"}
    missing_campaign_languages = required_campaign_languages - set(language_by_id)
    if missing_campaign_languages:
        fail(
            "the family inventory is missing required campaign languages: "
            + ", ".join(sorted(missing_campaign_languages))
        )

    rows: list[str] = []
    seen_projects: set[str] = set()
    seen_cells: set[str] = set()
    incomplete: list[str] = []
    incomplete_documentation: list[str] = []
    for project in projects:
        if not isinstance(project, dict):
            fail("each project must be an object")
        project_id = clean(project.get("id"), "project.id")
        if project_id in seen_projects:
            fail(f"duplicate project {project_id}")
        seen_projects.add(project_id)
        role = clean(project.get("role"), f"{project_id}.role")
        modeled_root = clean(project.get("root"), f"{project_id}.root")
        environment = PROJECT_ROOT_ENVIRONMENTS.get(project_id)
        configured_root = os.environ.get(environment) if environment else None
        modeled_project_root = (ROOT / modeled_root).resolve()
        project_root = (
            Path(configured_root) if configured_root else ROOT / modeled_root
        ).resolve()
        if not project_root.is_dir():
            fail(f"project root does not exist: {project_root}")
        capabilities = project.get("capabilities")
        unit_domains = project.get("unitDomains")
        evidence = project.get("declaredLanguageEvidence")
        reviews = project.get("reviewRequiredLanguages", [])
        if not isinstance(capabilities, list) or not capabilities:
            fail(f"{project_id} must declare capabilities")
        if not isinstance(unit_domains, list) or not unit_domains:
            fail(f"{project_id} must declare unitDomains")
        if not isinstance(evidence, dict) or not isinstance(reviews, list):
            fail(f"{project_id} has malformed language evidence/review lists")
        if len(reviews) != len(set(reviews)):
            fail(f"{project_id} has duplicate review-required languages")
        unknown = (set(evidence) | set(reviews)) - set(language_by_id)
        if unknown:
            fail(f"{project_id} names unknown languages: {sorted(unknown)}")
        overlap = set(evidence) & set(reviews)
        if overlap:
            fail(
                f"{project_id} both declares and requests review for: {sorted(overlap)}"
            )
        for language_id, relative in evidence.items():
            relative = clean(relative, f"{project_id}.{language_id}.evidence")
            if not (project_root / relative).exists():
                fail(f"declared evidence is missing: {project_root / relative}")
        undeclared_bindings = discover_binding_languages(project_root) - set(evidence)
        if undeclared_bindings:
            fail(
                f"{project_id} has binding directories omitted from "
                "declaredLanguageEvidence: " + ", ".join(sorted(undeclared_bindings))
            )

        seen_capabilities: set[str] = set()
        for capability_value in capabilities:
            capability = clean(capability_value, f"{project_id}.capability")
            if capability in seen_capabilities:
                fail(f"{project_id} has duplicate capability {capability}")
            seen_capabilities.add(capability)
            for language_id, language in language_by_id.items():
                cell_id = f"{project_id}|{language_id}|{capability}"
                if cell_id in seen_cells:
                    fail(f"duplicate cell {cell_id}")
                seen_cells.add(cell_id)
                override = overrides.get(cell_id, {})
                if not isinstance(override, dict):
                    fail(f"override for {cell_id} must be an object")
                unknown_override_keys = set(override) - ALLOWED_OVERRIDE_KEYS
                if unknown_override_keys:
                    fail(
                        f"override for {cell_id} has unknown keys: "
                        f"{sorted(unknown_override_keys)}"
                    )
                if language_id in evidence:
                    default_state = "audit-required"
                    default_evidence = str(
                        (modeled_project_root / evidence[language_id]).relative_to(
                            ROOT.parent
                        )
                    )
                elif language_id in reviews:
                    default_state = "review-required"
                    default_evidence = "-"
                else:
                    default_state = "missing"
                    default_evidence = "-"
                state = clean(override.get("state", default_state), f"{cell_id}.state")
                if state not in VALID_CELL_STATES:
                    fail(f"{cell_id} has invalid state {state}")
                proof = override.get("applicabilityProof", "-")
                proof = clean(proof, f"{cell_id}.applicabilityProof")
                if state == "inapplicable":
                    if proof == "-":
                        fail(f"{cell_id} is inapplicable without a reviewed proof")
                    proof_path = (ROOT / proof.split("#", 1)[0]).resolve()
                    if not proof_path.is_file():
                        fail(f"{cell_id} applicability proof is missing: {proof_path}")
                evidence_text = clean(
                    override.get("evidence", default_evidence), f"{cell_id}.evidence"
                )

                conformance = dimension_state("conformance", state, override, cell_id)
                benchmark = dimension_state("benchmark", state, override, cell_id)
                documentation_overrides = override.get("documentationTopics", {})
                if not isinstance(documentation_overrides, dict):
                    fail(f"{cell_id}.documentationTopics must be an object")
                unknown_topics = set(documentation_overrides) - set(topic_ids)
                if unknown_topics:
                    fail(
                        f"{cell_id}.documentationTopics names unknown topics: "
                        f"{sorted(unknown_topics)}"
                    )
                topic_results = [
                    documentation_topic(
                        topic_id,
                        state,
                        documentation_overrides.get(topic_id),
                        cell_id,
                        project_root,
                    )
                    for topic_id in topic_ids
                ]
                topic_states = [result[0] for result in topic_results]
                incomplete_documentation.extend(
                    f"{cell_id}|{topic_id}"
                    for topic_id, topic_state in zip(topic_ids, topic_states)
                    if topic_state not in {"complete", "inapplicable"}
                )
                documentation = aggregate_documentation(topic_states)
                fresh_consumer = dimension_state(
                    "freshConsumer", state, override, cell_id
                )
                if any(
                    value not in {"complete", "inapplicable"}
                    for value in (
                        state,
                        conformance,
                        benchmark,
                        documentation,
                        *topic_states,
                        fresh_consumer,
                    )
                ):
                    incomplete.append(cell_id)
                rows.append(
                    "\t".join(
                        (
                            project_id,
                            role,
                            capability,
                            language_id,
                            state,
                            ",".join(
                                clean(domain, f"{project_id}.unitDomain")
                                for domain in unit_domains
                            ),
                            language["hostIdioms"],
                            language["packageManager"],
                            "deterministic-close-plus-leak-containment",
                            conformance,
                            benchmark,
                            documentation,
                            *(
                                value
                                for topic_state, topic_evidence in topic_results
                                for value in (topic_state, topic_evidence)
                            ),
                            fresh_consumer,
                            proof,
                            evidence_text,
                        )
                    )
                )

    unknown_overrides = set(overrides) - seen_cells
    if unknown_overrides:
        fail(f"overrides name nonexistent cells: {sorted(unknown_overrides)}")
    if args.require_complete and incomplete:
        fail(
            f"{len(incomplete)} incomplete cells remain; first: {', '.join(incomplete[:10])}"
        )
    if args.require_documentation_complete and incomplete_documentation:
        fail(
            f"{len(incomplete_documentation)} incomplete documentation topics "
            f"remain; first: {', '.join(incomplete_documentation[:10])}"
        )

    header_fields = (
        "project",
        "role",
        "capability",
        "language",
        "surface_status",
        "unit_domains",
        "host_idioms",
        "package_manager",
        "lifecycle_model",
        "conformance_status",
        "benchmark_status",
        "documentation_status",
        *(
            field
            for topic_id in topic_ids
            for field in (
                f"doc_{topic_id.replace('-', '_')}_status",
                f"doc_{topic_id.replace('-', '_')}_evidence",
            )
        ),
        "fresh_consumer_status",
        "applicability_proof",
        "evidence",
    )
    malformed_rows = [
        index
        for index, row in enumerate(rows, start=2)
        if len(row.split("\t")) != len(header_fields)
    ]
    if malformed_rows:
        fail(f"matrix rows have the wrong field count: {malformed_rows[:10]}")
    header = "\t".join(header_fields)
    rendered = "\n".join((header, *rows, ""))
    if args.check:
        if (
            not MATRIX_PATH.is_file()
            or MATRIX_PATH.read_text(encoding="utf-8") != rendered
        ):
            fail(f"stale matrix: rerun {Path(__file__).relative_to(ROOT)}")
    else:
        MATRIX_PATH.write_text(rendered, encoding="utf-8")
    print(
        f"family completeness inventory: {len(projects)} projects x {len(languages)} "
        f"languages x capability catalogs = {len(rows)} cells; "
        f"{len(topic_ids)} documentation topics per cell"
    )
    print(
        f"completion gate: {'passed' if not incomplete else f'{len(incomplete)} cells incomplete'}"
    )
    print(
        "documentation gate: "
        + (
            "passed"
            if not incomplete_documentation
            else f"{len(incomplete_documentation)} topics incomplete"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
