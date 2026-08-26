#!/usr/bin/env python3
"""Generate the cross-project language/capability completeness inventory."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("schemaVersion") != 1:
        fail("unsupported schemaVersion")
    languages = model.get("languages")
    projects = model.get("projects")
    overrides = model.get("cellOverrides", {})
    if (
        not isinstance(languages, list)
        or not isinstance(projects, list)
        or not isinstance(overrides, dict)
    ):
        fail("languages/projects must be arrays and cellOverrides must be an object")

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
    if "raku" not in language_by_id or "rust" not in language_by_id:
        fail("the family inventory must include both native Rust and Raku")

    rows: list[str] = []
    seen_projects: set[str] = set()
    seen_cells: set[str] = set()
    incomplete: list[str] = []
    for project in projects:
        if not isinstance(project, dict):
            fail("each project must be an object")
        project_id = clean(project.get("id"), "project.id")
        if project_id in seen_projects:
            fail(f"duplicate project {project_id}")
        seen_projects.add(project_id)
        role = clean(project.get("role"), f"{project_id}.role")
        project_root = (
            ROOT / clean(project.get("root"), f"{project_id}.root")
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
                if language_id in evidence:
                    default_state = "audit-required"
                    default_evidence = str(
                        (project_root / evidence[language_id]).relative_to(ROOT.parent)
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
                documentation = dimension_state(
                    "documentation", state, override, cell_id
                )
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

    header = (
        "project\trole\tcapability\tlanguage\tsurface_status\tunit_domains\t"
        "host_idioms\tpackage_manager\tlifecycle_model\tconformance_status\t"
        "benchmark_status\tdocumentation_status\tfresh_consumer_status\t"
        "applicability_proof\tevidence"
    )
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
        f"family completeness inventory: {len(projects)} projects x {len(languages)} languages x capability catalogs = {len(rows)} cells"
    )
    print(
        f"completion gate: {'passed' if not incomplete else f'{len(incomplete)} cells incomplete'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
