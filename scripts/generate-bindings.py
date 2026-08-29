#!/usr/bin/env python3
"""Generate portable ABI constants from the language-neutral binding model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEROP_ROOT = Path(
    os.environ.get("VINARY_TREE_INTEROP_ROOT", ROOT.parent / "vinary-tree-interop")
).resolve()
MODEL_PATH = ROOT / "bindings" / "api.json"
NOTICE = "Generated from bindings/api.json; do not edit numeric values manually."

EXPECTED_DICTIONARY_ENTRIES = {
    "interfaceId": "vt.dict.entry.v1",
    "statusValues": {
        "OK": 0,
        "END": 1,
        "INVALID_ARGUMENT": 2,
        "NULL_POINTER": 3,
        "UNSUPPORTED": 4,
        "IO_ERROR": 5,
        "CLOSED": 6,
        "LIMIT_EXCEEDED": 7,
        "PROVIDER_ERROR": 8,
        "BATCH_IN_USE": 9,
    },
    "orderValues": {"LEXICOGRAPHIC": 1},
    "infoFlags": {"EXACT_LEN": 1, "SNAPSHOT_IDENTITY": 2},
    "vtableOperations": [
        "open",
        "next_batch",
        "release_batch",
        "reduce",
        "cancel",
        "close",
    ],
    "layouts": {
        "lp64": {
            "pointerWidth": 64,
            "types": {
                "VtDictionaryEntry": {
                    "size": 40,
                    "alignment": 8,
                    "offsets": {
                        "unit_offset": 0,
                        "unit_len": 8,
                        "value_offset": 16,
                        "value_len": 24,
                        "reserved": 32,
                    },
                },
                "VtDictionaryEntryBatchLimits": {
                    "size": 32,
                    "alignment": 8,
                    "offsets": {
                        "max_entries": 0,
                        "max_units": 8,
                        "max_values": 16,
                        "reserved": 24,
                    },
                },
                "VtDictionaryEntryBatchView": {
                    "size": 64,
                    "alignment": 8,
                    "offsets": {
                        "entries": 0,
                        "entry_count": 8,
                        "units": 16,
                        "unit_count": 24,
                        "values": 32,
                        "value_count": 40,
                        "generation": 48,
                        "reserved": 56,
                    },
                },
                "VtDictionaryEntriesInfo": {
                    "size": 64,
                    "alignment": 8,
                    "offsets": {
                        "unit_domain": 0,
                        "value_domain": 4,
                        "order": 8,
                        "reserved0": 12,
                        "flags": 16,
                        "exact_len": 24,
                        "identity": 32,
                        "reserved": 48,
                    },
                },
                "VtDictionaryEntriesCursor": {
                    "size": 16,
                    "alignment": 8,
                    "offsets": {"context": 0, "vtable": 8},
                },
                "VtDictionaryEntriesVTable": {
                    "size": 64,
                    "alignment": 8,
                    "offsets": {
                        "struct_size": 0,
                        "interface_version": 8,
                        "reserved": 12,
                        "open": 16,
                        "next_batch": 24,
                        "release_batch": 32,
                        "reduce": 40,
                        "cancel": 48,
                        "close": 56,
                    },
                },
            },
        },
        "arm32": {
            "pointerWidth": 32,
            "types": {
                "VtDictionaryEntry": {
                    "size": 24,
                    "alignment": 8,
                    "offsets": {
                        "unit_offset": 0,
                        "unit_len": 4,
                        "value_offset": 8,
                        "value_len": 12,
                        "reserved": 16,
                    },
                },
                "VtDictionaryEntryBatchLimits": {
                    "size": 24,
                    "alignment": 8,
                    "offsets": {
                        "max_entries": 0,
                        "max_units": 4,
                        "max_values": 8,
                        "reserved": 16,
                    },
                },
                "VtDictionaryEntryBatchView": {
                    "size": 40,
                    "alignment": 8,
                    "offsets": {
                        "entries": 0,
                        "entry_count": 4,
                        "units": 8,
                        "unit_count": 12,
                        "values": 16,
                        "value_count": 20,
                        "generation": 24,
                        "reserved": 32,
                    },
                },
                "VtDictionaryEntriesInfo": {
                    "size": 64,
                    "alignment": 8,
                    "offsets": {
                        "unit_domain": 0,
                        "value_domain": 4,
                        "order": 8,
                        "reserved0": 12,
                        "flags": 16,
                        "exact_len": 24,
                        "identity": 32,
                        "reserved": 48,
                    },
                },
                "VtDictionaryEntriesCursor": {
                    "size": 8,
                    "alignment": 4,
                    "offsets": {"context": 0, "vtable": 4},
                },
                "VtDictionaryEntriesVTable": {
                    "size": 36,
                    "alignment": 4,
                    "offsets": {
                        "struct_size": 0,
                        "interface_version": 4,
                        "reserved": 8,
                        "open": 12,
                        "next_batch": 16,
                        "release_batch": 20,
                        "reduce": 24,
                        "cancel": 28,
                        "close": 32,
                    },
                },
            },
        },
    },
}


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
        "npm": "@vinary-tree/vinary-tree-interop",
        "maven": "io.vinarytree:vinary-tree-interop",
        "javaPackage": "io.vinarytree.interop",
        "resourceLayout": ["context", "vtable"],
        "dictionaryInterfaceVersion": 1,
        "dictionaryVisitInterfaceVersion": 1,
        "dictionaryGraphInterfaceVersion": 1,
        "dictionaryEntriesInterfaceVersion": 1,
        "snapshotIdentityInterfaceVersion": 1,
        "scalarWfstInterfaceVersion": 1,
        "dictionaryEntries": EXPECTED_DICTIONARY_ENTRIES,
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

PUBLIC_ENUM_NAMES = {
    "status": "Status",
    "algorithm": "Algorithm",
    "queryOrder": "QueryOrder",
    "phoneticRuleSetKind": "PhoneticRuleSetKind",
}

JAVA_ENUM_DOCS = {
    "status": "Result of a fallible native operation.",
    "algorithm": "Edit-distance algorithm.",
    "queryOrder": "Lazy result ordering.",
    "phoneticRuleSetKind": "Built-in phonetic rewrite-rule set.",
}

JAVA_VALUE_DOCS = {
    "status": {
        "OK": "The operation completed successfully.",
        "END": "A finite cursor reached the end of its stream.",
        "INVALID_ARGUMENT": "An argument violated the operation's contract.",
        "INVALID_UTF8": "Input advertised as text was not valid UTF-8.",
        "NULL_POINTER": "A required native pointer was null.",
        "PANIC": "A contained Rust panic crossed the failure boundary.",
        "UNSUPPORTED": "The requested capability is unavailable in this build.",
        "IO_ERROR": "An input/output operation failed.",
        "CLOSED": "The target resource was already closed.",
        "LIMIT_EXCEEDED": "A configured resource or traversal limit was exceeded.",
        "PROVIDER_ERROR": "A foreign dictionary provider reported a failure.",
        "BATCH_IN_USE": "A cursor was advanced while its previous batch remained borrowed.",
        "DOMAIN_MISMATCH": "The query and dictionary use different unit domains.",
    },
    "algorithm": {
        "STANDARD": "Standard insert/delete/substitute distance.",
        "TRANSPOSITION": "Optimal string alignment with adjacent transposition.",
        "MERGE_AND_SPLIT": "Merge-and-split edit distance.",
        "DAMERAU_LEVENSHTEIN": "Unrestricted Damerau-Levenshtein distance.",
    },
    "queryOrder": {
        "TRAVERSAL": "Provider traversal order with bounded buffering.",
        "DISTANCE_THEN_TERM": "Distance then term, buffering at most one distance layer.",
    },
    "phoneticRuleSetKind": {
        "ENGLISH_ORTHOGRAPHY": "English orthography normalization.",
        "ENGLISH_PHONETIC": "English phonetic transformation.",
    },
}

GO_VALUE_NAMES = {
    "status": {
        "OK": "StatusOK",
        "END": "StatusEnd",
        "INVALID_ARGUMENT": "StatusInvalidArgument",
        "INVALID_UTF8": "StatusInvalidUTF8",
        "NULL_POINTER": "StatusNullPointer",
        "PANIC": "StatusPanic",
        "UNSUPPORTED": "StatusUnsupported",
        "IO_ERROR": "StatusIOError",
        "CLOSED": "StatusClosed",
        "LIMIT_EXCEEDED": "StatusLimitExceeded",
        "PROVIDER_ERROR": "StatusProviderError",
        "BATCH_IN_USE": "StatusBatchInUse",
        "DOMAIN_MISMATCH": "StatusDomainMismatch",
    },
    "algorithm": {
        "STANDARD": "Standard",
        "TRANSPOSITION": "Transposition",
        "MERGE_AND_SPLIT": "MergeAndSplit",
        "DAMERAU_LEVENSHTEIN": "DamerauLevenshtein",
    },
    "queryOrder": {
        "TRAVERSAL": "Traversal",
        "DISTANCE_THEN_TERM": "DistanceThenTerm",
    },
    "phoneticRuleSetKind": {
        "ENGLISH_ORTHOGRAPHY": "EnglishOrthography",
        "ENGLISH_PHONETIC": "EnglishPhonetic",
    },
}

RUBY_VALUE_NAMES = {
    "status": {
        # END is a Ruby parser keyword and cannot be assigned as a constant.
        "END": "END_OF_STREAM",
    }
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


def render_java_enum(model: dict, key: str) -> str:
    """Render one public JVM enum from the language-neutral ABI model."""

    enum = model["enums"][key]
    class_name = PUBLIC_ENUM_NAMES[key]
    lines = [
        f"package {model['organization']['javaPackage']};",
        "",
        f"/** {JAVA_ENUM_DOCS[key]}",
        f" * <p>{NOTICE}",
        " */",
        f"public enum {class_name} {{",
    ]
    descriptions = JAVA_VALUE_DOCS.get(key, {})
    values = list(enum["values"].items())
    for index, (name, value) in enumerate(values):
        description = descriptions.get(
            name, name.lower().replace("_", " ").capitalize() + "."
        )
        separator = "," if index + 1 < len(values) or key == "status" else ";"
        lines.extend([f"    /** {description} */", f"    {name}({value}){separator}"])
    if key == "status":
        lines.extend(
            [
                "    /** A status introduced by a newer compatible ABI revision. */",
                "    UNKNOWN(-1);",
            ]
        )
    lines.extend(
        [
            "",
            "    private final int nativeValue;",
            "",
            f"    {class_name}(int nativeValue) {{",
            "        this.nativeValue = nativeValue;",
            "    }",
            "",
        ]
    )
    if key == "status":
        lines.extend(
            [
                "    /**",
                "     * Return this known status's stable numeric ABI value.",
                "     *",
                "     * <p>{@link #UNKNOWN} returns {@code -1}; use",
                "     * {@link NativeException#statusCode()} to recover an unknown status's",
                "     * original forward-compatible value.",
                "     *",
                "     * @return stable ABI value, or {@code -1} for {@link #UNKNOWN}",
                "     */",
                "    public int code() {",
                "        return nativeValue;",
                "    }",
                "",
                "    static Status fromNativeValue(int value) {",
                "        return switch (value) {",
            ]
        )
        for name, value in values:
            lines.append(f"            case {value} -> {name};")
        lines.extend(
            ["            default -> UNKNOWN;", "        };", "    }", "}", ""]
        )
    else:
        lines.extend(
            [
                "    int nativeValue() {",
                "        return nativeValue;",
                "    }",
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def render_dotnet_enums(model: dict) -> str:
    """Render public .NET enums from the language-neutral ABI model."""

    lines = [
        "// <auto-generated />",
        f"// {NOTICE}",
        "",
        "namespace VinaryTree.Liblevenshtein;",
        "",
    ]
    for key, enum in model["enums"].items():
        class_name = PUBLIC_ENUM_NAMES[key]
        summary = JAVA_ENUM_DOCS[key]
        if key == "status":
            summary += (
                " The uint backing value preserves statuses introduced by a newer "
                "compatible API revision."
            )
        lines.extend(
            [
                "/// <summary>",
                f"/// {summary}",
                "/// </summary>",
                f"public enum {class_name} : uint",
                "{",
            ]
        )
        descriptions = JAVA_VALUE_DOCS.get(key, {})
        for name, value in enum["values"].items():
            description = descriptions.get(
                name, name.lower().replace("_", " ").capitalize() + "."
            )
            lines.extend(
                [
                    f"    /// <summary>{description}</summary>",
                    f"    {pascal(name)} = {value},",
                ]
            )
        lines.extend(["}", ""])
    return "\n".join(lines)


def render_go_enums(model: dict) -> str:
    """Render public Go enum types from the language-neutral ABI model."""

    lines = [
        "// Code generated by scripts/generate-bindings.py from bindings/api.json; DO NOT EDIT.",
        "",
        "package liblevenshtein",
        "",
    ]
    for key, enum in model["enums"].items():
        type_name = PUBLIC_ENUM_NAMES[key]
        lines.extend(
            [
                f"// {type_name} is the {JAVA_ENUM_DOCS[key].lower()}",
                f"type {type_name} uint32",
                "",
                "const (",
            ]
        )
        for name, value in enum["values"].items():
            public_name = GO_VALUE_NAMES[key][name]
            description = JAVA_VALUE_DOCS[key][name]
            lines.extend(
                [
                    f"\t// {public_name} indicates {description[:1].lower() + description[1:]}",
                    f"\t{public_name} {type_name} = {value}",
                ]
            )
        lines.extend([")", ""])
    return "\n".join(lines)


def render_ruby_enums(model: dict) -> str:
    """Render public Ruby constant modules from the language-neutral ABI model."""

    lines = [
        "# frozen_string_literal: true",
        "",
        "# Code generated by scripts/generate-bindings.py from bindings/api.json; DO NOT EDIT.",
        "module VinaryTree",
        "  module Liblevenshtein",
    ]
    for key, enum in model["enums"].items():
        lines.append(f"    module {PUBLIC_ENUM_NAMES[key]}")
        for name, value in enum["values"].items():
            public_name = RUBY_VALUE_NAMES.get(key, {}).get(name, name)
            lines.append(f"      {public_name} = {value}")
        lines.extend(["    end", ""])
    lines.extend(["  end", "end", ""])
    return "\n".join(lines)


def render_swift_enums(model: dict) -> str:
    """Render public Swift ABI selections while preserving unknown statuses."""

    lines = [
        "// Code generated by scripts/generate-bindings.py from bindings/api.json; DO NOT EDIT.",
        "",
    ]
    for key, enum in model["enums"].items():
        type_name = PUBLIC_ENUM_NAMES[key]
        summary = JAVA_ENUM_DOCS[key]
        descriptions = JAVA_VALUE_DOCS[key]
        if key == "status":
            lines.extend(
                [
                    f"/// {summary}",
                    "///",
                    "/// This is an open raw-value wrapper so statuses introduced by a newer",
                    "/// compatible API revision remain observable rather than being discarded.",
                    f"public struct {type_name}: RawRepresentable, Hashable, Sendable, CustomStringConvertible {{",
                    "    public let rawValue: UInt32",
                    "",
                    "    public init(rawValue: UInt32) { self.rawValue = rawValue }",
                    "",
                ]
            )
            for name, value in enum["values"].items():
                public_name = camel(name)
                lines.extend(
                    [
                        f"    /// {descriptions[name]}",
                        f"    public static let {public_name} = Status(rawValue: {value})",
                    ]
                )
            lines.extend(["", "    public var description: String {", "        switch self {"])
            for name in enum["values"]:
                public_name = camel(name)
                lines.append(f'        case .{public_name}: "{name}"')
            lines.extend(
                [
                    '        default: "UNKNOWN(\\(rawValue))"',
                    "        }",
                    "    }",
                    "}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"/// {summary}",
                f"public enum {type_name}: UInt32, Sendable {{",
            ]
        )
        for name, value in enum["values"].items():
            lines.extend(
                [
                    f"    /// {descriptions[name]}",
                    f"    case {camel(name)} = {value}",
                ]
            )
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


def render_dictionary_entries_fixture(model: dict) -> str:
    entries = model["interop"]["dictionaryEntries"]
    lines = ["kind\ttier\ttype\tmember\tvalue"]
    lines.extend(
        [
            f"interface\t-\tVtDictionaryEntriesVTable\tid\t{entries['interfaceId']}",
            f"interface\t-\tVtDictionaryEntriesVTable\tversion\t{model['interop']['dictionaryEntriesInterfaceVersion']}",
        ]
    )
    for name, value in entries["statusValues"].items():
        lines.append(f"status\t-\tVtStatus\t{name}\t{value}")
    for name, value in entries["orderValues"].items():
        lines.append(f"order\t-\tVtDictionaryEntryOrder\t{name}\t{value}")
    for name, value in entries["infoFlags"].items():
        lines.append(f"info_flag\t-\tVtDictionaryEntriesInfo\t{name}\t{value}")
    for index, operation in enumerate(entries["vtableOperations"]):
        lines.append(f"operation\t-\tVtDictionaryEntriesVTable\t{operation}\t{index}")
    for tier, layout in entries["layouts"].items():
        lines.append(f"layout\t{tier}\t-\tpointer_width\t{layout['pointerWidth']}")
        for type_name, type_layout in layout["types"].items():
            lines.append(f"layout\t{tier}\t{type_name}\tsize\t{type_layout['size']}")
            lines.append(
                f"layout\t{tier}\t{type_name}\talignment\t{type_layout['alignment']}"
            )
            for member, offset in type_layout["offsets"].items():
                lines.append(f"offset\t{tier}\t{type_name}\t{member}\t{offset}")
    lines.append("")
    return "\n".join(lines)


def outputs(model: dict, *, include_siblings: bool = False) -> dict[Path, str]:
    java_path = Path(*model["organization"]["javaPackage"].split("."))
    interop_header = (INTEROP_ROOT / "include" / "vinary_tree_interop.h").read_text(
        encoding="utf-8"
    )
    public_header = (ROOT / "include" / "liblevenshtein.h").read_text(encoding="utf-8")
    abi_header = render_c(model)
    ocaml_header = (
        INTEROP_ROOT / "bindings" / "ocaml" / "vinary_tree_ocaml.h"
    ).read_text(encoding="utf-8")
    lua_header = (INTEROP_ROOT / "bindings" / "lua" / "vinary_tree_lua.h").read_text(
        encoding="utf-8"
    )
    java_source_root = ROOT / "bindings" / "jvm" / "src" / "main" / "java" / java_path
    generated = {
        ROOT / "include" / "liblevenshtein_abi.h": render_c(model),
        ROOT / "src" / "ffi" / "generated.rs": render_rust(model),
        ROOT
        / "bindings"
        / "python"
        / "src"
        / "liblevenshtein"
        / "_generated.py": render_python(model),
        java_source_root / "GeneratedAbi.java": render_java(model),
        java_source_root / "Status.java": render_java_enum(model, "status"),
        java_source_root / "Algorithm.java": render_java_enum(model, "algorithm"),
        java_source_root / "QueryOrder.java": render_java_enum(model, "queryOrder"),
        java_source_root / "PhoneticRuleSetKind.java": render_java_enum(
            model, "phoneticRuleSetKind"
        ),
        ROOT
        / "bindings"
        / "dotnet"
        / "src"
        / "VinaryTree.Liblevenshtein"
        / "GeneratedEnums.cs": render_dotnet_enums(model),
        ROOT / "bindings" / "go" / "generated_enums.go": render_go_enums(model),
        ROOT
        / "bindings"
        / "ruby"
        / "lib"
        / "vinary_tree"
        / "liblevenshtein"
        / "generated_enums.rb": render_ruby_enums(model),
        ROOT
        / "bindings"
        / "swift"
        / "liblevenshtein"
        / "Sources"
        / "Liblevenshtein"
        / "GeneratedEnums.swift": render_swift_enums(model),
        ROOT / "bindings" / "javascript" / "abi.d.ts": render_typescript(model),
        ROOT / "bindings" / "conformance" / "query_start_snapshot.tsv": render_fixture(
            model
        ),
        ROOT
        / "bindings"
        / "conformance"
        / "dictionary_entries_v1.tsv": render_dictionary_entries_fixture(model),
        ROOT / "bindings" / "ocaml" / "include" / "liblevenshtein.h": public_header,
        ROOT / "bindings" / "ocaml" / "include" / "liblevenshtein_abi.h": abi_header,
        ROOT
        / "bindings"
        / "ocaml"
        / "include"
        / "vinary_tree_interop.h": interop_header,
        ROOT / "bindings" / "lua" / "include" / "vinary_tree_lua.h": lua_header,
        ROOT / "bindings" / "lua" / "include" / "vinary_tree_interop.h": interop_header,
        ROOT
        / "bindings"
        / "swift"
        / "liblevenshtein"
        / "Sources"
        / "CLiblevenshtein"
        / "vinary_tree_interop.h": interop_header,
    }
    # Cross-project release work may opt in to refreshing sibling mirrors. Normal
    # local generation is deliberately confined to this repository.
    libdictenstein = ROOT.parent / "libdictenstein"
    if include_siblings and libdictenstein.is_dir():
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
    parser.add_argument(
        "--include-siblings",
        action="store_true",
        help="also generate header mirrors in an adjacent libdictenstein checkout",
    )
    args = parser.parse_args()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    validate(model)
    stale = []
    for path, content in outputs(model, include_siblings=args.include_siblings).items():
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
