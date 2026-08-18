#!/usr/bin/env python3
"""Generate portable ABI constants from the language-neutral binding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bindings" / "api.json"
NOTICE = "Generated from bindings/api.json; do not edit numeric values manually."


def pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.lower().split("_"))


def camel(name: str) -> str:
    value = pascal(name)
    return value[:1].lower() + value[1:]


def validate(model: dict) -> None:
    expected_org = {
        "github": "vinary-tree",
        "npmScope": "vinary-tree",
        "mavenGroup": "io.vinarytree",
        "javaPackage": "io.vinarytree.liblevenshtein",
        "fpmNamespace": "vinary-tree",
    }
    if model.get("organization") != expected_org:
        raise SystemExit("binding organization must use vinary-tree coordinates")
    expected_interop = {
        "crate": "vinary-tree-interop",
        "cPrefix": "vt_",
        "cHeader": "vinary_tree_interop.h",
        "npm": "@vinary-tree/interop",
        "maven": "io.vinarytree:vinary-tree-interop",
        "javaPackage": "io.vinarytree.interop",
        "resourceLayout": ["context", "vtable"],
        "dictionaryInterfaceVersion": 1,
        "dictionaryVisitInterfaceVersion": 1,
        "dictionaryGraphInterfaceVersion": 1,
        "snapshotIdentityInterfaceVersion": 1,
        "scalarWfstInterfaceVersion": 1,
    }
    if model.get("interop") != expected_interop:
        raise SystemExit("binding model changed the shared interop identity")
    if model.get("abiVersion") != 1 or model.get("apiRevision") != 2:
        raise SystemExit("unexpected ABI/API revision")
    if model.get("defaultMatchBatch") != 256:
        raise SystemExit("default match batch must remain 256 in ABI v1")
    names = [item["name"] for item in model.get("cFunctions", [])]
    if not names or len(names) != len(set(names)):
        raise SystemExit("cFunctions must be non-empty and unique")
    for enum_name, enum in model.get("enums", {}).items():
        values = list(enum.get("values", {}).values())
        if not values or len(values) != len(set(values)):
            raise SystemExit(f"enum {enum_name} values must be non-empty and unique")
    owned = set(model.get("objects", {}))
    forbidden = set(model.get("forbiddenOwnedObjects", []))
    if owned & forbidden:
        raise SystemExit("liblevenshtein may not own dictionary binding objects")


def render_c(model: dict) -> str:
    lines = [
        f"/* {NOTICE} */",
        "#ifndef LIBLEVENSHTEIN_ABI_H",
        "#define LIBLEVENSHTEIN_ABI_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#ifndef VT_INTEROP_HEADER",
        '#define VT_INTEROP_HEADER "vinary_tree_interop.h"',
        "#endif",
        "#include VT_INTEROP_HEADER",
        "",
        f"#define LLEV_ABI_VERSION {model['abiVersion']}u",
        f"#define LLEV_API_REVISION {model['apiRevision']}u",
        f"#define LLEV_DEFAULT_MATCH_BATCH {model['defaultMatchBatch']}u",
        "",
    ]
    for name, value in model["buildFeatures"].items():
        lines.append(f"#define LLEV_BUILD_FEATURE_{name} UINT64_C({value})")
    lines.append("")
    for enum in model["enums"].values():
        lines.append(f"typedef enum {enum['cType']} {{")
        values = list(enum["values"].items())
        for index, (name, value) in enumerate(values):
            comma = "," if index + 1 < len(values) else ""
            lines.append(f"    {enum['cPrefix']}{name} = {value}{comma}")
        lines.append(f"}} {enum['cType']};")
        lines.append("")
    lines.extend(
        [
            "typedef struct LlevTransducer LlevTransducer;",
            "typedef struct LlevQueryCursor LlevQueryCursor;",
            "typedef struct LlevPhoneticPattern LlevPhoneticPattern;",
            "typedef struct LlevPhoneticRuleSet LlevPhoneticRuleSet;",
            "",
            "typedef struct LlevMatch {",
            "    const void* term_data;",
            "    size_t term_len;",
            "    size_t byte_len;",
            "    size_t distance;",
            "    uint64_t id;",
            "    VtUnitDomain unit_domain;",
            "    uint8_t has_id;",
            "    uint8_t reserved[3];",
            "} LlevMatch;",
            "",
            "typedef struct LlevMatchBatchView {",
            "    const LlevMatch* matches;",
            "    size_t len;",
            "    uint64_t generation;",
            "} LlevMatchBatchView;",
            "",
            "typedef struct LlevOwnedString {",
            "    char* data;",
            "    size_t len;",
            "} LlevOwnedString;",
            "",
            "typedef LlevStatus (*LlevBatchReducer)(void* context,",
            "                                       const LlevMatch* matches,",
            "                                       size_t len);",
            "",
            "#endif /* LIBLEVENSHTEIN_ABI_H */",
            "",
        ]
    )
    return "\n".join(lines)


RUST_DOCS = {
    "status": "Result of a fallible native operation.",
    "algorithm": "Edit-distance algorithm.",
    "queryOrder": "Lazy result order.",
    "phoneticRuleSetKind": "Built-in phonetic rewrite-rule set.",
}


def render_rust(model: dict) -> str:
    lines = [
        "//! Generated ABI constants and enums.",
        "",
        "/// Stable liblevenshtein native ABI version.",
        f"pub const LLEV_ABI_VERSION: u32 = {model['abiVersion']};",
        "/// Additive API revision within this ABI version.",
        f"pub const LLEV_API_REVISION: u32 = {model['apiRevision']};",
        "",
    ]
    for name, value in model["buildFeatures"].items():
        lines.extend(
            [
                f"/// Compiled binding feature: {name.lower()}.",
                f"pub const LLEV_BUILD_FEATURE_{name}: u64 = {value};",
            ]
        )
    lines.append("")
    for key, enum in model["enums"].items():
        lines.extend(
            [
                f"/// {RUST_DOCS[key]}",
                "#[repr(u32)]",
                "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
                f"pub enum {enum['cType']} {{",
            ]
        )
        for name, value in enum["values"].items():
            lines.append(f"    /// {name.lower().replace('_', ' ')}.")
            lines.append(f"    {pascal(name)} = {value},")
        lines.extend(
            [
                "}",
                "",
                f"impl TryFrom<u32> for {enum['cType']} {{",
                "    type Error = ();",
                "",
                "    fn try_from(value: u32) -> Result<Self, Self::Error> {",
                "        match value {",
            ]
        )
        for name, value in enum["values"].items():
            lines.append(f"            {value} => Ok(Self::{pascal(name)}),")
        lines.extend(["            _ => Err(()),", "        }", "    }", "}", ""])
    return "\n".join(lines)


def render_python(model: dict) -> str:
    lines = [
        f'"""{NOTICE}"""',
        "from enum import IntEnum",
        "",
        f"ABI_VERSION = {model['abiVersion']}",
        f"API_REVISION = {model['apiRevision']}",
        f"DEFAULT_MATCH_BATCH = {model['defaultMatchBatch']}",
        "",
    ]
    for enum in model["enums"].values():
        name = enum["cType"].removeprefix("Llev")
        lines.append(f"class {name}(IntEnum):")
        for item, value in enum["values"].items():
            lines.append(f"    {item} = {value}")
        lines.append("")
    return "\n".join(lines)


def render_java(model: dict) -> str:
    status = model["enums"]["status"]["values"]
    lines = [
        f"package {model['organization']['javaPackage']};",
        "",
        f"/** {NOTICE} */",
        "final class GeneratedAbi {",
        "    private GeneratedAbi() {}",
        f"    static final int ABI_VERSION = {model['abiVersion']};",
        f"    static final int API_REVISION = {model['apiRevision']};",
        f"    static final int DEFAULT_MATCH_BATCH = {model['defaultMatchBatch']};",
    ]
    for name, value in status.items():
        lines.append(f"    static final int STATUS_{name} = {value};")
    lines.extend(["}", ""])
    return "\n".join(lines)


def render_typescript(model: dict) -> str:
    lines = [
        f"/** {NOTICE} */",
        f"export const ABI_VERSION: {model['abiVersion']};",
        f"export const API_REVISION: {model['apiRevision']};",
        f"export const DEFAULT_MATCH_BATCH: {model['defaultMatchBatch']};",
    ]
    for key, enum in model["enums"].items():
        lines.append(f"export enum {pascal(key)} {{")
        for name, value in enum["values"].items():
            lines.append(f"  {pascal(name)} = {value},")
        lines.append("}")
    lines.append("")
    return "\n".join(lines)


def render_fixture(model: dict) -> str:
    fixture = model["snapshotFixture"]
    lines = ["phase\toperation\tterm\tid"]
    lines.extend(
        f"initial\tinsert\t{term}\t{'' if value is None else value}"
        for term, value in fixture["initial"]
    )
    lines.extend(
        f"mutation\t{operation}\t{term}\t{'' if value is None else value}"
        for operation, term, value in fixture["mutations"]
    )
    lines.append("")
    return "\n".join(lines)


def outputs(model: dict) -> dict[Path, str]:
    java_path = Path(*model["organization"]["javaPackage"].split("."))
    interop_header = (
        ROOT / "vinary-tree-interop" / "include" / "vinary_tree_interop.h"
    ).read_text(encoding="utf-8")
    public_header = (ROOT / "include" / "liblevenshtein.h").read_text(encoding="utf-8")
    abi_header = render_c(model)
    ocaml_header = (
        ROOT / "vinary-tree-interop" / "bindings" / "ocaml" / "vinary_tree_ocaml.h"
    ).read_text(encoding="utf-8")
    lua_header = (
        ROOT / "vinary-tree-interop" / "bindings" / "lua" / "vinary_tree_lua.h"
    ).read_text(encoding="utf-8")
    generated = {
        ROOT / "include" / "liblevenshtein_abi.h": render_c(model),
        ROOT / "src" / "ffi" / "generated.rs": render_rust(model),
        ROOT
        / "bindings"
        / "python"
        / "src"
        / "liblevenshtein"
        / "_generated.py": render_python(model),
        ROOT
        / "bindings"
        / "jvm"
        / "src"
        / "main"
        / "java"
        / java_path
        / "GeneratedAbi.java": render_java(model),
        ROOT / "bindings" / "javascript" / "abi.d.ts": render_typescript(model),
        ROOT / "bindings" / "conformance" / "query_start_snapshot.tsv": render_fixture(
            model
        ),
        # Placement-clean sharing of the conformance oracle: sibling repos may
        # read a DEPENDENCY's files, and vinary-tree-interop is everyone's
        # dependency — so the fixture is mirrored into the interop crate for
        # libdictenstein/lling-llang/duallity tests to consume.
        ROOT
        / "vinary-tree-interop"
        / "conformance"
        / "query_start_snapshot.tsv": render_fixture(model),
        ROOT
        / "vinary-tree-interop"
        / "bindings"
        / "haskell"
        / "include"
        / "vinary_tree_interop.h": interop_header,
        ROOT / "bindings" / "ocaml" / "include" / "liblevenshtein.h": public_header,
        ROOT / "bindings" / "ocaml" / "include" / "liblevenshtein_abi.h": abi_header,
        ROOT
        / "bindings"
        / "ocaml"
        / "include"
        / "vinary_tree_interop.h": interop_header,
        ROOT / "bindings" / "ocaml" / "include" / "vinary_tree_ocaml.h": ocaml_header,
    }
    # libdictenstein owns dictionary construction and its language packages,
    # but mirrors the shared ABI headers from this one canonical model. The
    # sibling is present in every cross-project development and release checkout.
    libdictenstein = ROOT.parent / "libdictenstein"
    if libdictenstein.is_dir():
        libdictenstein_header = (
            libdictenstein / "include" / "libdictenstein.h"
        ).read_text(encoding="utf-8")
        generated.update(
            {
                libdictenstein / "include" / "vinary_tree_interop.h": interop_header,
                libdictenstein
                / "bindings"
                / "ocaml"
                / "include"
                / "libdictenstein.h": libdictenstein_header,
                libdictenstein
                / "bindings"
                / "lua"
                / "include"
                / "vinary_tree_lua.h": lua_header,
                libdictenstein
                / "bindings"
                / "ocaml"
                / "include"
                / "vinary_tree_interop.h": interop_header,
                libdictenstein
                / "bindings"
                / "ocaml"
                / "include"
                / "vinary_tree_ocaml.h": ocaml_header,
            }
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    validate(model)
    stale = []
    for path, content in outputs(model).items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                try:
                    stale.append(path.relative_to(ROOT))
                except ValueError:
                    stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        raise SystemExit(
            "generated binding files are stale: " + ", ".join(map(str, stale))
        )


if __name__ == "__main__":
    main()
