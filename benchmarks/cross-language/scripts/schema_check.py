#!/usr/bin/env python3
"""Focused JSON-Schema validator for the two schemas this program ships.

Stdlib-only by design (no jsonschema dependency). Implements exactly the
subset of JSON Schema draft 2020-12 those schemas use: type, required,
properties, additionalProperties, enum, const, pattern, minimum, minLength,
minItems, items, propertyNames-free maps via additionalProperties-as-schema,
and allOf/if-then keyed on property values. Anything outside that subset in
a future schema revision must extend this validator (it fails loudly on
unknown keywords rather than skipping them).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HANDLED_KEYWORDS = {
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "required",
    "properties",
    "additionalProperties",
    "enum",
    "const",
    "pattern",
    "minimum",
    "maximum",
    "minLength",
    "minItems",
    "items",
    "allOf",
    "if",
    "then",
}

TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class ValidationError(Exception):
    pass


def _check_type(schema_type, value, path: str) -> None:
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    if not any(TYPE_CHECKS[t](value) for t in types):
        raise ValidationError(f"{path}: expected type {types}, got {type(value).__name__}")


def _matches(schema: dict, value, path: str) -> bool:
    try:
        validate(value, schema, path)
        return True
    except ValidationError:
        return False


def validate(value, schema: dict, path: str = "$") -> None:
    unknown = set(schema) - HANDLED_KEYWORDS
    if unknown:
        raise ValidationError(f"{path}: schema uses unhandled keywords {sorted(unknown)}")

    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: {value!r} not in enum {schema['enum']}")
    if "type" in schema:
        _check_type(schema["type"], value, path)

    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValidationError(f"{path}: {value!r} does not match {schema['pattern']!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: shorter than minLength {schema['minLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationError(f"{path}: fewer than minItems {schema['minItems']}")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{i}]")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValidationError(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                validate(item, additional, f"{path}.{key}")
            elif additional is False:
                raise ValidationError(f"{path}: unexpected property {key!r}")

    for i, sub in enumerate(schema.get("allOf", [])):
        cond = sub.get("if")
        then = sub.get("then")
        if cond is not None:
            if _matches(cond, value, f"{path}<allOf[{i}].if>") and then is not None:
                validate(value, then, f"{path}<allOf[{i}].then>")
        else:
            validate(value, sub, f"{path}<allOf[{i}]>")


def validate_file(instance_path: Path, schema_path: Path) -> list[str]:
    """Returns a list of error strings; empty means valid."""
    try:
        instance = json.loads(instance_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{instance_path}: invalid JSON: {exc}"]
    schema = json.loads(schema_path.read_text())
    try:
        validate(instance, schema)
    except ValidationError as exc:
        return [f"{instance_path}: {exc}"]
    return []


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: schema_check.py <schema.json> <instance.json>...", file=sys.stderr)
        return 2
    schema_path = Path(sys.argv[1])
    failures = 0
    for name in sys.argv[2:]:
        errors = validate_file(Path(name), schema_path)
        for error in errors:
            print(f"INVALID  {error}", file=sys.stderr)
        failures += len(errors)
    if failures == 0:
        print(f"OK: {len(sys.argv) - 2} file(s) valid", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
