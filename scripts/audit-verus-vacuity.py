#!/usr/bin/env python3
"""Reject reflexive or premise-repeating Verus proof obligations.

Verus quite correctly proves a postcondition such as ``x == x`` or a
postcondition copied verbatim from ``requires``.  Those statements are still
useless as evidence.  This scanner parses the contract header of every
``proof fn`` without attempting to parse executable Rust bodies, then rejects
mechanically recognizable forms of vacuity: repeated premises, literal truth,
reflexive relations, and self-implications.

The scanner is deliberately syntax-aware about nested ``()``, ``[]``, and
``{}`` delimiters.  A comma inside a call or enum constructor does not split a
contract expression, and an equality nested inside a larger proposition is
not mistaken for a reflexive top-level conclusion.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROOF_START = re.compile(r"\bproof\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    function: str
    reason: str
    expression: str


def strip_comments(source: str) -> str:
    """Replace comments with whitespace while preserving newlines/offsets."""

    output = list(source)
    index = 0
    block_depth = 0
    in_line = False
    in_string = False
    escaped = False
    while index < len(source):
        pair = source[index : index + 2]
        char = source[index]
        if in_line:
            if char == "\n":
                in_line = False
            else:
                output[index] = " "
            index += 1
            continue
        if block_depth:
            if pair == "/*":
                output[index] = output[index + 1] = " "
                block_depth += 1
                index += 2
            elif pair == "*/":
                output[index] = output[index + 1] = " "
                block_depth -= 1
                index += 2
            else:
                if char != "\n":
                    output[index] = " "
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if pair == "//":
            output[index] = output[index + 1] = " "
            in_line = True
            index += 2
        elif pair == "/*":
            output[index] = output[index + 1] = " "
            block_depth = 1
            index += 2
        else:
            if char == '"':
                in_string = True
            index += 1
    return "".join(output)


def matching_delimiter(source: str, opening: int, left: str, right: str) -> int:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == left:
            depth += 1
        elif source[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed {left!r} at byte {opening}")


def find_body_start(source: str, start: int) -> int:
    parens = brackets = braces = 0
    index = start
    while index < len(source):
        char = source[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "{":
            if parens == 0 and brackets == 0 and braces == 0:
                return index
            braces += 1
        elif char == "}":
            braces -= 1
        index += 1
    raise ValueError(f"proof body not found after byte {start}")


def split_top_level(expressions: str) -> list[str]:
    parts: list[str] = []
    start = 0
    parens = brackets = braces = 0
    for index, char in enumerate(expressions):
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "," and parens == brackets == braces == 0:
            part = expressions[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = expressions[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def normalize(expression: str) -> str:
    return re.sub(r"\s+", "", expression).rstrip(",")


def strip_outer_parentheses(expression: str) -> str:
    expression = expression.strip()
    while expression.startswith("("):
        try:
            closing = matching_delimiter(expression, 0, "(", ")")
        except ValueError:
            break
        if closing != len(expression) - 1:
            break
        expression = expression[1:-1].strip()
    return expression


def top_level_relation(expression: str) -> tuple[str, str, str] | None:
    expression = strip_outer_parentheses(expression)
    parens = brackets = braces = 0
    index = 0
    operators = ("<==>", "==>", "==", "<=", ">=")
    while index < len(expression):
        char = expression[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif parens == brackets == braces == 0:
            for operator in operators:
                if expression.startswith(operator, index):
                    return expression[:index], operator, expression[index + len(operator) :]
        index += 1
    return None


def contract_clauses(header: str, keyword: str) -> list[str]:
    match = re.search(rf"\b{keyword}\b", header)
    if not match:
        return []
    start = match.end()
    next_clause = re.search(r"\b(?:requires|ensures|recommends|decreases)\b", header[start:])
    end = start + next_clause.start() if next_clause else len(header)
    return split_top_level(header[start:end])


def audit_source(path: Path, source: str) -> list[Finding]:
    clean = strip_comments(source)
    findings: list[Finding] = []
    for match in PROOF_START.finditer(clean):
        function = match.group(1)
        parameter_open = clean.find("(", match.start())
        parameter_close = matching_delimiter(clean, parameter_open, "(", ")")
        body_start = find_body_start(clean, parameter_close + 1)
        header = clean[parameter_close + 1 : body_start]
        requirements = {normalize(item) for item in contract_clauses(header, "requires")}
        for conclusion in contract_clauses(header, "ensures"):
            normalized = normalize(conclusion)
            line = source.count("\n", 0, parameter_close + 1 + header.find(conclusion)) + 1
            if normalized in requirements:
                findings.append(
                    Finding(path, line, function, "postcondition repeats a precondition", conclusion)
                )
            if normalize(strip_outer_parentheses(conclusion)) == "true":
                findings.append(
                    Finding(path, line, function, "literal true postcondition", conclusion)
                )
            relation = top_level_relation(conclusion)
            if relation and normalize(relation[0]) == normalize(relation[2]):
                reasons = {
                    "==>": "self-implication",
                    "<==>": "reflexive equivalence",
                    "==": "reflexive equality",
                    "<=": "reflexive non-strict inequality",
                    ">=": "reflexive non-strict inequality",
                }
                findings.append(Finding(path, line, function, reasons[relation[1]], conclusion))
    return findings


def self_test() -> None:
    fixture = """
verus! {
proof fn repeated(x: nat)
    requires x > 0,
    ensures x > 0,
{}
proof fn reflexive(x: nat)
    ensures (f(x, 1) == f(x, 1)),
{}
proof fn reflexive_order(x: nat)
    ensures x <= x,
{}
proof fn self_implication(x: nat)
    ensures x > 0 ==> x > 0,
{}
proof fn literal_truth()
    ensures true,
{}
proof fn useful(x: nat, y: nat)
    ensures x + y == y + x,
{}
}
"""
    findings = audit_source(Path("fixture.rs"), fixture)
    assert [finding.function for finding in findings] == [
        "repeated",
        "reflexive",
        "reflexive_order",
        "self_implication",
        "literal_truth",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    paths = arguments.paths or sorted(Path("docs/verification/verus").glob("*.rs"))
    findings: list[Finding] = []
    for path in paths:
        findings.extend(audit_source(path, path.read_text(encoding="utf-8")))
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}: {finding.function}: "
            f"{finding.reason}: {finding.expression.strip()}"
        )
    if findings:
        print(f"error: {len(findings)} vacuous Verus contract conclusion(s)", file=sys.stderr)
        return 1
    print(f"No reflexive or premise-repeating Verus conclusions found in {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
