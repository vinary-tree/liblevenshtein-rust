#!/usr/bin/env python3
"""Render the checked operational sections of every shipped binding guide.

The prose before the generated marker remains the package-specific tutorial.
This generator owns the uniform operational contract below it: support and
package metadata, executable evidence, ownership, errors, concurrency,
marshalling, security, troubleshooting, and family navigation.  Keeping that
contract data-driven lets the binding gate reject a newly declared language
that has no corresponding documentation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->"
END_MARKER = "<!-- END GENERATED BINDING OPERATIONS -->"
SURFACE_MODEL = json.loads(
    (ROOT / "bindings/api-surface-map.json").read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class Guide:
    language: str
    languages: str
    tier: str
    package: str
    boundary: str
    cleanup: str
    errors: str
    concurrency: str
    units: str
    source: str
    evidence: str
    command: str


GUIDES: dict[str, Guide] = {
    "c": Guide(
        "C",
        "C17 and C23",
        "Tier 1",
        "CMake package `liblevenshtein` and `pkg-config` module `liblevenshtein`",
        "The public C header calls the `llev_*` ABI exported by the shared or static native library.",
        "Balance every successful constructor/retain with its documented free/release function and release each cursor batch before advancing.",
        "Functions return `LlevStatus`; inspect the enum first and copy `llev_last_error_message()` before making another call on that thread.",
        "Independent handles are reentrant. A query cursor and its current lease are single-consumer resources.",
        "All strings and arrays use pointer-plus-length descriptors; embedded zero bytes and empty terms are valid where the function contract permits them.",
        "include/liblevenshtein.h",
        "bindings/c/tests/cross_project_snapshot.c",
        'cc -std=c17 -Wall -Wextra -Werror -Iinclude -I../libdictenstein/include -I../vinary-tree-interop/include bindings/c/tests/cross_project_snapshot.c -Ltarget/debug -lliblevenshtein -L../libdictenstein/target/debug -llibdictenstein -Wl,-rpath,"$PWD/target/debug" -Wl,-rpath,"$PWD/../libdictenstein/target/debug" -o target/c-cross-project-snapshot && target/c-cross-project-snapshot',
    ),
    "cpp": Guide(
        "C and C++",
        "C17/C23 and C++20/C++23",
        "Tier 1",
        "CMake package `liblevenshtein` and `pkg-config` module `liblevenshtein`",
        "The C surface calls `llev_*` directly; the C++ header adds move-only RAII and exceptions without another native boundary.",
        "Pair every C constructor with its documented free function. Prefer C++ scope-bound wrappers and never copy an owning raw handle.",
        "C returns `LlevStatus` and a thread-local diagnostic; C++ converts non-OK statuses into `vinary_tree::liblevenshtein::error`.",
        "Independent handles and captured queries are reentrant. A cursor is single-consumer, and a leased batch must be released before the next cursor operation.",
        "C spans carry explicit lengths. C++ overloads accept byte, Unicode-scalar, and `uint64_t` views without sentinel termination.",
        "include/liblevenshtein.h and include/liblevenshtein.hpp",
        "bindings/cpp/tests/snapshot.cpp",
        "cmake -S bindings/cpp/tests/package -B target/cpp-package && cmake --build target/cpp-package && ctest --test-dir target/cpp-package",
    ),
    "python": Guide(
        "Python",
        "Python 3.10+",
        "Tier 1",
        "PyPI package `liblevenshtein`",
        "`ctypes` calls the stable C ABI; first-party native calls release the GIL, while Python callbacks reacquire it.",
        "Use `with` for `Transducer` and `QueryCursor`. Finalizers are leak containment, not a deterministic resource policy.",
        "Native statuses become typed Python exceptions with the native diagnostic preserved in the message.",
        "Separate transducers and cursors may run on separate threads. Do not advance one cursor concurrently or retain callback-scoped borrowed views.",
        "`str` selects Unicode scalars, `bytes` selects raw bytes, and integer sequences select the packed `u64` domain.",
        "bindings/python/src/liblevenshtein",
        "bindings/python/tests/test_api.py",
        "PYTHONPATH=bindings/python/src:../vinary-tree-interop/bindings/python/src pytest -q bindings/python/tests/test_api.py",
    ),
    "jvm": Guide(
        "JVM",
        "Java 22+, Kotlin, and Scala",
        "Tier 1",
        "Maven coordinate `io.vinarytree:liblevenshtein`",
        "The finalized Foreign Function & Memory API calls the stable C ABI. Kotlin and Scala consume the same Java classes and ownership contracts.",
        "Use try-with-resources, Kotlin `use`, or Scala `Using.resource`. `Cleaner` is only a leak-safety fallback.",
        "Every non-OK status becomes a Java exception containing the thread-local native diagnostic; callback exceptions still close the active lease.",
        "Transducers and independent cursors are reentrant. One cursor is single-consumer; borrowed `MemorySegment` views expire when their batch callback returns.",
        "Java `String` is UTF-8 encoded for Unicode queries; byte arrays and `long[]` select the byte and packed-token domains.",
        "bindings/jvm/src/main/java/io/vinarytree/liblevenshtein",
        "bindings/jvm/src/smoke/java/io/vinarytree/liblevenshtein/ResourceSnapshotSmoke.java",
        "./gradlew -p bindings/jvm test",
    ),
    "clojure": Guide(
        "Clojure",
        "Clojure 1.12+ on Java 22+",
        "Tier 1",
        "Clojars coordinate `io.vinarytree/liblevenshtein-clojure`",
        "The idiomatic namespace delegates to the JVM facade and therefore introduces no second native boundary.",
        "Wrap transducers and cursors in `with-open`; lazy sequences that are abandoned before EOF must be closed explicitly.",
        "Native failures retain the delegated JVM `NativeException`, generated `Status` enum, exact raw status code, and copied native diagnostic.",
        "Independent resources are reentrant. Consume a single reducible cursor from one thread, and never retain a `reduce-batches` view after its callback.",
        "Strings select Unicode traversal; byte arrays and long collections use the corresponding JVM overloads.",
        "bindings/clojure/src/vinary_tree/liblevenshtein.clj",
        "bindings/clojure/test/vinary_tree/liblevenshtein_test.clj",
        "clojure -M:test -m vinary-tree.liblevenshtein-test-runner",
    ),
    "javascript": Guide(
        "JavaScript family",
        "JavaScript, TypeScript, and ClojureScript on Node.js, browsers, or WASI",
        "Tier 1",
        "npm package `@vinary-tree/liblevenshtein`",
        "The facade delegates to the singleton `@vinary-tree/javascript-runtime` runtime: native N-API by default, WebAssembly or WASI through explicit exports.",
        "Dictionaries and query cursors implement `Symbol.dispose` and support `using`. Close transducers and phonetic resources in `finally`; ClojureScript uses `close!`. GC finalizers are fallback containment.",
        "Native failures become thrown `Error` values carrying a copied native diagnostic. The facade does not expose the numeric status, so callers handle failures by operation and error category rather than parsing message text.",
        "Wrapper objects belong to one JavaScript runtime instance and Worker and must not be transferred between Workers. A cursor is single-consumer; returned matches and batches are host-owned values.",
        "Strings are Unicode; `Uint8Array` and `BigUint64Array` select byte and packed-token queries. IDs are `bigint`, never lossy JavaScript numbers.",
        "bindings/javascript",
        "bindings/javascript/test/facades.test.mjs",
        "npm test --prefix bindings/javascript",
    ),
    "dotnet": Guide(
        ".NET",
        ".NET 8+ and current C#",
        "Tier 2",
        "NuGet package `Liblevenshtein`",
        "Source-generated P/Invoke reaches the stable C ABI; `VinaryTree.Interop` carries retained resource handles between packages.",
        "Use `using`/`await using`-style lexical ownership for disposable handles. `SafeHandle` protects exceptional paths but does not replace disposal.",
        "Non-OK statuses become typed .NET exceptions with status and native diagnostic properties.",
        "Independent handles are safe across tasks. One enumerator is single-consumer; cancellation or early exit must dispose it to release the batch lease.",
        "`string`, `ReadOnlySpan<byte>`, and `ReadOnlySpan<ulong>` select Unicode, byte, and token domains.",
        "bindings/dotnet/src/VinaryTree.Liblevenshtein",
        "bindings/dotnet/tests/VinaryTree.Liblevenshtein.Tests/Program.cs",
        "dotnet run --project bindings/dotnet/tests/VinaryTree.Liblevenshtein.Tests",
    ),
    "go": Guide(
        "Go",
        "Go 1.25+ with cgo",
        "Tier 2",
        "Go module `github.com/vinary-tree/liblevenshtein-rust/bindings/go/v4`",
        "cgo calls the stable C ABI and uses the interop module for two-word retained dictionary resources.",
        "Call `Close` with `defer` immediately after successful construction. Finalizers report leaks but are not a prompt release mechanism.",
        "Native statuses become Go errors that preserve the symbolic status and native diagnostic; callers may inspect them without parsing strings.",
        "Different handles may be used by different goroutines. Serialize `Next` and `Close` for one iterator and do not retain borrowed native memory.",
        "Strings use Unicode queries; byte and `[]uint64` entry points preserve raw domain identity.",
        "bindings/go/liblevenshtein.go",
        "bindings/go/liblevenshtein_test.go",
        "go test ./bindings/go",
    ),
    "swift": Guide(
        "Swift",
        "Swift 6+ through Swift Package Manager",
        "Tier 2",
        "SwiftPM product `Liblevenshtein`",
        "A Swift system-library target imports the stable C ABI and shares `DictionaryResource` with the interop and libdictenstein packages.",
        "Close long-lived cursors explicitly and use lexical `defer`; `deinit` is a safety net whose timing is not an API guarantee.",
        "C statuses are converted to throwing Swift errors with the native diagnostic retained.",
        "Independent values are reentrant. Iterator mutation is exclusive under Swift value-access rules; never escape a batch callback buffer.",
        "`String`, contiguous bytes, and `[UInt64]` retain their distinct native domains.",
        "bindings/swift/liblevenshtein/Sources/Liblevenshtein",
        "bindings/swift/Integration/Sources/SwiftBindingIntegration/main.swift",
        "swift run --package-path bindings/swift/Integration SwiftBindingIntegration",
    ),
    "ruby": Guide(
        "Ruby",
        "Ruby 3.3+",
        "Tier 2",
        "RubyGems package `liblevenshtein`",
        "Ruby Fiddle calls the stable C ABI; modular producers yield the two-word resource without serialization.",
        "Prefer block forms and `ensure { cursor.close }`; finalizers only prevent permanent leaks.",
        "Non-OK statuses become typed Ruby exceptions exposing the status symbol and native diagnostic.",
        "Separate native handles are reentrant. Enumerate one cursor on one fiber/thread and never retain callback-scoped buffers.",
        "Ruby strings use explicit encoding rules; byte strings and integer arrays select non-Unicode domains.",
        "bindings/ruby/lib/vinary_tree/liblevenshtein",
        "bindings/ruby/test/test_liblevenshtein.rb",
        "ruby -Ibindings/ruby/lib bindings/ruby/test/test_liblevenshtein.rb",
    ),
    "fortran": Guide(
        "Fortran",
        "Fortran 2018 through fpm",
        "Tier 2",
        "fpm package `liblevenshtein`",
        "`iso_c_binding` declarations call the stable C ABI and share the interop resource derived type.",
        "Call each derived handle's `close` method; final procedures are exceptional-path protection rather than deterministic scheduling.",
        "Procedures return or raise the module's status abstraction while preserving the native diagnostic for reporting.",
        "Independent derived handles are reentrant. Do not access one mutable iterator from multiple images or retain a leased C pointer.",
        "Character data is UTF-8 marshalled explicitly; `integer(c_int8_t)` and `integer(c_int64_t)` arrays preserve byte/token domains.",
        "bindings/fortran/src/vinary_tree_liblevenshtein.f90",
        "bindings/fortran/integration/test/test_cross_project.f90",
        "fpm test -C bindings/fortran/integration --profile release",
    ),
    "ocaml": Guide(
        "OCaml",
        "OCaml 5 through dune/opam",
        "Tier 3",
        "opam package `liblevenshtein`",
        "C stubs call the stable ABI and consume `Vinary_tree_interop.resource` values from independent producers.",
        "Use the explicit `close` functions or `Fun.protect`; GC finalizers are only a last-resort retain release.",
        "C statuses become typed OCaml exceptions carrying the native diagnostic.",
        "Independent handles are domain-safe according to the documented capability flags. A cursor remains single-consumer and borrowed batches cannot escape folds.",
        "Strings carry UTF-8; bytes and int64 arrays select raw byte and packed-token domains.",
        "bindings/ocaml/vinary_tree_liblevenshtein.mli",
        "bindings/ocaml/test/snapshot.ml",
        "opam exec -- dune runtest --root bindings/ocaml",
    ),
    "haskell": Guide(
        "Haskell",
        "GHC through Cabal",
        "Tier 3",
        "Hackage package `liblevenshtein`",
        "The Haskell FFI calls the stable C ABI; `DictionaryResource` is the retained cross-package capability.",
        "Use `bracket`, `withTransducer`, and `withQueryCursor`; `ForeignPtr` finalizers protect abandoned exceptional paths.",
        "Native failures become typed Haskell exceptions/status values with the diagnostic copied before another FFI call.",
        "Independent handles are reentrant. Do not call cursor advance concurrently; masking around acquire/release prevents asynchronous-exception leaks.",
        "`Text`, `ByteString`, and vectors of `Word64` preserve the three unit domains.",
        "bindings/haskell/src/VinaryTree/Liblevenshtein.hsc",
        "bindings/haskell/test/Main.hs",
        "cabal test --project-file=bindings/haskell/cabal.project snapshot",
    ),
    "lua": Guide(
        "Lua",
        "Lua 5.4+",
        "Tier 3",
        "LuaRocks package `liblevenshtein`",
        "A C module calls the stable ABI and consumes `vinary-tree.dictionary.v1` userdata from modular producers.",
        "Use Lua 5.4 to-be-closed variables or call `:close()`; `__gc` is a leak fallback.",
        "Native failures become Lua errors containing the symbolic status and copied diagnostic.",
        "Separate userdata values are reentrant. One cursor is single-consumer and its native batch is never exposed beyond one iterator step.",
        "Lua strings are byte sequences; Unicode entry points validate UTF-8, while explicit byte/token constructors preserve their domains.",
        "bindings/lua/src/liblevenshtein_lua.c",
        "bindings/lua/tests/snapshot.lua",
        "lua bindings/lua/tests/snapshot.lua",
    ),
}


INTEROP_PACKAGES = {
    "python": "PyPI package `vinary-tree-interop`",
    "jvm": "Maven coordinate `io.vinarytree:vinary-tree-interop`",
    "javascript": "npm package `@vinary-tree/vinary-tree-interop`",
    "go": "Go module `github.com/vinary-tree/vinary-tree-interop/bindings/go/v4`",
    "swift": "SwiftPM product `VinaryTreeInterop`",
    "fortran": "fpm package `vinary-tree-interop`",
    "ocaml": "opam package `vinary-tree-interop`",
    "haskell": "Hackage package `vinary-tree-interop`",
    "lua": "C adapter header bundled by dependent LuaRocks packages",
}


INTEROP_GUIDES: dict[str, Guide] = {
    key: Guide(
        guide.language,
        guide.languages,
        guide.tier,
        INTEROP_PACKAGES[key],
        "This adapter represents the two-word `VtResource` capability and its versioned interfaces; it does not implement a dictionary or automaton.",
        guide.cleanup.replace("Transducer", "resource")
        .replace("QueryCursor", "resource handle")
        .replace("cursor", "resource handle")
        + " Close every entries cursor, and release its current generation before advancing or closing it.",
        "Interop validation failures preserve `VtStatus`; project facades map that status into their own public error currency.",
        "A retained resource may cross threads only when its advertised interface flags permit it. Retain and release remain balanced under every failure path. One entries cursor and its live batch are single-consumer; reducer callbacks must not reenter that cursor.",
        "Unit and value domains are explicit enum fields on the discovered interface; adapters must never infer them from host container types.",
        f"vinary-tree-interop/bindings/{key}",
        guide.evidence,
        guide.command,
    )
    for key, guide in GUIDES.items()
    if key
    in {
        "python",
        "jvm",
        "javascript",
        "go",
        "swift",
        "fortran",
        "ocaml",
        "haskell",
        "lua",
    }
}


def markdown_cell(value: str) -> str:
    """Escape data inserted into a GitHub-flavored Markdown table cell."""

    return value.replace("|", "\\|").replace("\n", " ")


def operation_role(name: str) -> str:
    """Return the user-facing capability represented by one ABI operation."""

    if name in {"llev_abi_version", "llev_api_revision", "llev_build_features"}:
        return "ABI compatibility and feature discovery"
    if name == "llev_last_error_message":
        return "typed failure diagnostics"
    if name.startswith("llev_true_damerau_distance"):
        return "standalone true-Damerau distance"
    if name.startswith(("llev_distance", "llev_damerau_distance")):
        return "standalone exact or thresholded distance"
    if name in {"llev_string_free", "llev_string_array_free", "llev_string_dup"}:
        return "legacy owned-string plumbing"
    if name == "llev_owned_string_free":
        return "owned result-string release"
    if name.startswith("llev_transducer_query_pattern"):
        return "phonetic-pattern dictionary query"
    if name.startswith("llev_transducer_query"):
        return "domain-preserving dictionary query"
    if name.startswith("llev_transducer"):
        return "transducer lifecycle, snapshot, or domain metadata"
    if name.startswith("llev_query_cursor"):
        return "streaming result traversal and batch leases"
    if name.startswith("llev_phonetic_pattern"):
        return "compiled phonetic-pattern lifecycle and matching"
    if name.startswith("llev_phonetic_rules"):
        return "phonetic rule-set lifecycle and rewriting"
    return "project ABI operation"


def facade_surface(key: str) -> str:
    """Render the exhaustive idiomatic public-symbol index for one facade."""

    facade = SURFACE_MODEL["languages"][key]
    grouped: dict[str, list[str]] = {}
    for operation, mapping in facade["functions"].items():
        symbols = mapping.get("symbol")
        if symbols is None:
            continue
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            grouped.setdefault(symbol, []).append(operation)

    symbol_rows: list[str] = []
    for symbol in sorted(grouped, key=str.casefold):
        operations = grouped[symbol]
        rendered_operations = ", ".join(f"`{name}`" for name in operations)
        roles = "; ".join(dict.fromkeys(operation_role(name) for name in operations))
        symbol_rows.append(
            f"| `{markdown_cell(symbol)}` | {rendered_operations} | {markdown_cell(roles)} |"
        )

    type_rows: list[str] = []
    type_omission_rows: list[str] = []
    type_roles = {
        "status": "Typed native status or error carrier",
        "algorithm": "Edit-distance algorithm selection",
        "queryOrder": "Result traversal ordering",
        "phoneticRuleSetKind": "Built-in phonetic rule-set selection",
    }
    for name, mapping in facade["enums"].items():
        symbol = mapping.get("symbol")
        if symbol is None:
            type_omission_rows.append(
                f"| `{name}` | {markdown_cell(mapping['_reason'])} |"
            )
        else:
            type_rows.append(
                f"| `{markdown_cell(symbol)}` | {type_roles[name]} | "
                f"{markdown_cell(mapping.get('note', 'Public facade type'))} |"
            )
    for name, role in (
        ("iterator", "One-shot owned-result iteration"),
        ("reducer", "Bounded batch/reducer traversal"),
    ):
        mapping = facade[name]
        symbol = mapping.get("symbol")
        if symbol is None:
            type_omission_rows.append(
                f"| `{name}` | {markdown_cell(mapping['_reason'])} |"
            )
        else:
            type_rows.append(
                f"| `{markdown_cell(symbol)}` | {role} | "
                f"{markdown_cell(mapping.get('note', 'Public facade protocol'))} |"
            )

    rendered = [
        "### Facade symbol index",
        "",
        "This table is generated from the same exhaustive model as the binding",
        "conformance gate. A public symbol may implement several ABI operations when",
        "the host language expresses domain or lifecycle choices with overloads,",
        "variants, protocols, or methods.",
        "",
        "| Public symbol | Backing native operation(s) | Capability |",
        "|---|---|---|",
        *symbol_rows,
        "",
        "### Public types and traversal protocols",
        "",
        "| Facade type or protocol | Purpose | Exposure note |",
        "|---|---|---|",
        *type_rows,
    ]
    if type_omission_rows:
        rendered.extend(
            [
                "",
                "### Facade-encapsulated model values",
                "",
                "| Model value | Idiomatic treatment |",
                "|---|---|",
                *type_omission_rows,
            ]
        )
    rendered.extend(
        [
            "",
            "Native operations omitted from the public-symbol table are deliberately",
            "encapsulated by the facade. The generated completeness matrix records every",
            "such operation with its reviewed rationale; an unreasoned absence fails CI.",
        ]
    )
    return "\n".join(rendered)


def generated_block(
    guide: Guide, *, key: str | None = None, interop: bool = False
) -> str:
    if interop:
        architecture = "../../docs/abi-reference.md"
        security = "../../docs/security-model.md"
        evolution = "../../docs/abi-evolution.md"
        family = "../../../docs/bindings/README.md"
        source = f"../../../{guide.source}"
        evidence = f"../../../{guide.evidence}"
        diagram = "../../../docs/diagrams/bindings/interface-negotiation-activity.svg"
        api_reference = "../../docs/abi-reference.md"
        api_label = "family resource ABI reference"
        concepts = """| Concept | Semantics |
|---|---|
| `VtResource` | Two pointer-sized words: an opaque context and a base vtable. A borrowed value transfers no ownership. |
| Base vtable | `struct_size`, ABI version, retain, release, and `query_interface`; it is the only mandatory interface. |
| Dictionary interface | Snapshot capture, node paging, finality, optional values, unit/value domains, and capability flags. |
| Dictionary entries interface | Optional finite lexicographic stream over one captured revision, with bounded arena batches, exact generation leases, cancellation, and a reducer path. |
| Scalar-WFST interface | Snapshot capture, start state, final weights, paged arcs, label/weight domains, and capability flags. |"""
        ownership_detail = """A borrowed resource becomes owned only after a successful `retain`. Interface
discovery does not transfer ownership, and a failed validation must release any
retain already acquired. A captured snapshot owns an independent revision and
may outlive the producing project handle. Release exactly once for every
successful retain; never release an unretained borrowed pair."""
        ownership_detail += """ An entries cursor is move-only and owns its
captured revision until `close`. Exactly one generation may be live: release
that exact generation before advancing, reducing, or closing; reducer batch
views expire when their callback returns."""
        performance = """- Pass the two-word resource by value; do not serialize or copy the graph.
- Capture one immutable snapshot and page nodes/arcs through bounded buffers.
- Negotiate entries-v1 when exact lexicographic enumeration is needed; honor all entry/unit/value limits on every batch.
- Cache a validated optional interface only while the owning resource remains retained.
- Respect capability flags before enabling parallel callback entry.
- Prefer a compact immutable graph interface when advertised; retain the paged callback fallback for compatibility."""
        failure_scope = "null resource words, truncated vtables, incompatible interface identities or versions, invalid domains, forged node/state identifiers, malformed page counts or entry arenas, stale or mismatched batch generations, live-batch conflicts, provider faults, and contained panics"
        maintainer_first = "Update the interop generator model before changing a layout, identifier, flag, or enum."
        surface_contract = "[`bindings/api.json`](../../../bindings/api.json) and the generated interop constants"
    else:
        architecture = "../../docs/language-bindings.md"
        security = "../../docs/security/binding-trust-model.md"
        evolution = "https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md"
        family = "../../docs/bindings/README.md"
        source = f"../../{guide.source}"
        evidence = f"../../{guide.evidence}"
        diagram = "../../docs/diagrams/bindings/three-layer-architecture.svg"
        api_reference = "../../docs/bindings/c-abi-reference.md"
        api_label = "`llev_*` C ABI reference"
        concepts = """| Concept | Semantics |
|---|---|
| Dictionary resource | A retained `vt.dictionary.v1` capability. Construction and mutation belong to a producer such as libdictenstein. |
| Transducer | Immutable query configuration plus a retained dictionary provider; construction is constant-time with respect to dictionary size. |
| Query cursor | A one-shot traversal over the immutable dictionary revision captured at query start. |
| Match/batch | Owned matches are stable host values; a borrowed batch is valid only inside its documented callback or lease interval. |"""
        ownership_detail = """A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles."""
        reducer_exposed = (
            SURFACE_MODEL["languages"][key or ""]["reducer"].get("symbol") is not None
        )
        traversal_performance = (
            "- Prefer batch/reducer APIs when per-match boundary crossings dominate."
            if reducer_exposed
            else "- Drain each cursor once; the iterator already fetches bounded native batches before materializing host-owned matches."
        )
        performance = f"""- Reuse transducers for repeated queries against the same resource.
- Prefer streaming cursors to whole-result materialization.
{traversal_performance}
- Keep Unicode, byte, and token domains explicit to avoid transcoding.
- Measure native, WASM, and WASI paths independently; they have different
  startup and marshalling costs but identical query semantics.

No host wrapper should cache unbounded query results. Applications that add a
memo use a revision key and a hard entry/weight bound; eviction may be
approximate because all values remain derivable from the retained snapshot."""
        failure_scope = "malformed UTF-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained Rust panics"
        maintainer_first = (
            "Update the machine-readable binding model before changing a public symbol."
        )
        surface_contract = "[`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv)"

    surface = "" if interop else facade_surface(key or "")
    if interop or reducer_exposed:
        maximum_throughput_use = "The facade batch/reducer protocol"
    else:
        maximum_throughput_use = (
            "Drain the facade iterator; no public reducer is exposed"
        )

    if not interop and key == "javascript":
        batch_rationale = (
            "It amortizes the foreign boundary while returning bounded, "
            "host-owned arrays."
        )
        result_lifetime = """Matches and batches are copied into host-owned values before returning to
JavaScript. They remain valid independently of the cursor, and no native lease
or raw pointer is exposed to user code."""
        error_guidance = """Malformed UTF-8, unsupported unit domains, incompatible resource versions,
closed handles, invalid bounds, allocation failures, provider faults, and
contained Rust panics remain distinct native causes. The facade preserves their
diagnostics but intentionally does not promise a public numeric status; branch
on the JavaScript error class and failing operation, never diagnostic prose."""
    elif interop or reducer_exposed:
        batch_rationale = (
            "It amortizes the foreign boundary and keeps borrowed views inside "
            "one lexical lease."
        )
        result_lifetime = """Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena."""
        error_guidance = f"""{failure_scope.capitalize()} are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread."""
    else:
        batch_rationale = (
            "The iterator still amortizes native calls with bounded internal "
            "batches, then releases each lease before exposing host-owned matches."
        )
        result_lifetime = """Iterator results are copied into host-owned values before their native batch
lease is released. They remain valid after iteration advances or the cursor is
closed; no raw pointer or borrowed native view reaches user code."""
        error_guidance = f"""{failure_scope.capitalize()} are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread."""

    return f"""{MARKER}

## Support and package contract

| Property | Contract |
|---|---|
| Binding | {guide.language} |
| Languages/runtime | {guide.languages} |
| Support tier | {guide.tier} |
| Distribution | {guide.package} |
| Native boundary | {guide.boundary} |
| Canonical facade source | [`{guide.source}`]({source}) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture]({architecture}) before implementing a custom provider
and the [family hub]({family}) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.]({diagram})

## Executable example and verification

The repository's canonical executable example is
[`{guide.evidence}`]({evidence}). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
{guide.command}
```

Examples deliberately construct or receive resources through public project
packages. They never import private Rust modules, depend on object layout, or
reach behind the stable C/resource ABIs.

## Public API and data model

The idiomatic facade groups the stable surface into these concepts:

{concepts}

{guide.units} Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

{surface}

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | The facade iterator protocol | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | {maximum_throughput_use} | {batch_rationale} |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[{api_label}]({api_reference}). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by {surface_contract}.

## Ownership, snapshots, and resource handoff

{guide.cleanup}

{ownership_detail}

{result_lifetime}

## Errors and failure containment

{guide.errors}

{error_guidance}

## Concurrency and reentrancy

{guide.concurrency}

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

{performance}

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model]({security}).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy]({evolution}); never infer compatibility from a
package version alone.

When loading fails, check—in order—the documented runtime/toolchain version,
CPU/OS artifact, native-access permission, loader search path, dependent
interop package pin, and process-wide JavaScript runtime identity. When a query
fails after construction, report the typed status and copied diagnostic before
reducing the case to the smallest dictionary/query pair.

## Maintainer checklist

1. {maintainer_first}
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

{END_MARKER}
"""


def render(
    path: Path, guide: Guide, *, key: str | None = None, interop: bool = False
) -> str:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        prefix = existing.split(MARKER, 1)[0].rstrip()
    else:
        prefix = (
            f"# Vinary Tree {guide.language} interop binding\n\n"
            "This package exposes the language-native representation of the stable "
            "Vinary Tree resource ABI. It is the neutral handoff layer used by "
            "dictionary, automaton, and WFST packages; it owns no algorithm-specific "
            "policy."
        )
    return f"{prefix}\n\n{generated_block(guide, key=key, interop=interop)}"


def render_interop_root() -> tuple[Path, str]:
    path = ROOT / "vinary-tree-interop/README.md"
    existing = path.read_text(encoding="utf-8")
    marker = "<!-- BEGIN GENERATED INTEROP LANGUAGE INDEX; DO NOT EDIT -->"
    end = "<!-- END GENERATED INTEROP LANGUAGE INDEX -->"
    prefix = existing.split(marker, 1)[0].rstrip()
    rows = [
        "| C/C++ | Native header and CMake/pkg-config package | This README and `docs/abi-reference.md` |",
    ]
    for key, guide in INTEROP_GUIDES.items():
        rows.append(
            f"| {guide.languages} | `{guide.package}` | [`bindings/{key}/README.md`](bindings/{key}/README.md) |"
        )
    block = "\n".join(
        [
            marker,
            "",
            "## Language adapter documentation",
            "",
            "Every published adapter uses the same retained-resource laws while mapping ownership and failures into its host language:",
            "",
            "The generated [`dictionary_entries_v1.tsv`](conformance/dictionary_entries_v1.tsv) fixture pins entries-v1 identifiers, statuses, flags, operation order, and LP64/ARM32 layouts for adapter conformance.",
            "",
            "| Language/runtime | Distribution | Guide |",
            "|---|---|---|",
            *rows,
            "",
            "The adapters are intentionally policy-free. Concrete dictionary and automaton APIs live in their project packages; these guides explain only resource representation, discovery, ownership, and safe handoff.",
            "",
            end,
            "",
        ]
    )
    return path, f"{prefix}\n\n{block}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if generated sections are stale"
    )
    args = parser.parse_args()
    outputs: list[tuple[Path, str]] = []
    for key, guide in GUIDES.items():
        path = ROOT / f"bindings/{key}/README.md"
        outputs.append((path, render(path, guide, key=key)))
    stale: list[Path] = []
    for path, rendered in outputs:
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
    if stale:
        joined = "\n".join(f"  - {path}" for path in stale)
        raise SystemExit(
            f"generated binding guides are stale:\n{joined}\nrun scripts/generate-binding-guides.py"
        )


if __name__ == "__main__":
    main()
