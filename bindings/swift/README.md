# liblevenshtein Swift bindings

The SwiftPM product `Liblevenshtein` consumes any `DictionaryResource` from
`VinaryTreeInterop`; concrete dictionaries remain in libdictenstein's Swift
package. `QueryCursor` is a `Sequence` and `IteratorProtocol`, advances through
leased native batches, and copies only the current batch into Swift ownership.
The native CMake package may be linked dynamically or statically by the parent
application; the Swift system-library target uses the installed shared library
by default.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Swift |
| Languages/runtime | Swift 6+ through Swift Package Manager |
| Support tier | Tier 2 |
| Distribution | SwiftPM product `Liblevenshtein` |
| Native boundary | A Swift system-library target imports the stable C ABI and shares `DictionaryResource` with the interop and libdictenstein packages. |
| Canonical facade source | [`bindings/swift/liblevenshtein/Sources/Liblevenshtein`](../../bindings/swift/liblevenshtein/Sources/Liblevenshtein) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/swift/Integration/Sources/SwiftBindingIntegration/main.swift`](../../bindings/swift/Integration/Sources/SwiftBindingIntegration/main.swift). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
swift run --package-path bindings/swift/Integration SwiftBindingIntegration
```

Examples deliberately construct or receive resources through public project
packages. They never import private Rust modules, depend on object layout, or
reach behind the stable C/resource ABIs.

## Public API and data model

The idiomatic facade groups the stable surface into these concepts:

| Concept | Semantics |
|---|---|
| Dictionary resource | A retained `vt.dictionary.v1` capability. Construction and mutation belong to a producer such as libdictenstein. |
| Transducer | Immutable query configuration plus a retained dictionary provider; construction is constant-time with respect to dictionary size. |
| Query cursor | A one-shot traversal over the immutable dictionary revision captured at query start. |
| Match/batch | Owned matches are stable host values; a borrowed batch is valid only inside its documented callback or lease interval. |

### Automaton selection

| Algorithm | Edit semantics | Metric? | Typical use |
|---|---|---:|---|
| Standard | Insert, delete, and substitute | yes | General spelling correction |
| Transposition | Optimal string alignment with adjacent swaps | no | Typographical swaps when metric-tree laws are unnecessary |
| Merge and split | Standard edits plus symmetric two-to-one and one-to-two edits | yes | Optical character recognition and segmentation errors |
| Damerau-Levenshtein | Unrestricted, history-composable adjacent transpositions | yes | True Damerau matching and metric indexes |

The transposition and unrestricted Damerau variants are deliberately distinct:
for example, optimal string alignment assigns distance 3 from `CA` to `ABC`,
while unrestricted Damerau-Levenshtein assigns distance 2. Select the algorithm
when constructing the transducer; all query domains and snapshot laws remain the
same.

`String`, contiguous bytes, and `[UInt64]` retain their distinct native domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `EditDistance.damerauLevenshtein` | `llev_true_damerau_distance`, `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |
| `EditDistance.damerauOSA` | `llev_damerau_distance`, `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `EditDistance.levenshtein` | `llev_distance`, `llev_distance_threshold` | standalone exact or thresholded distance |
| `LiblevenshteinError` | `llev_last_error_message` | typed failure diagnostics |
| `LiblevenshteinError.status` | `llev_last_error_message` | typed failure diagnostics |
| `PhoneticPattern.close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.llre` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.regex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet.apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.builtin` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.count` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.parse` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `QueryCache.clear` | `llev_query_cache_clear` | project ABI operation |
| `QueryCache.close` | `llev_query_cache_free` | project ABI operation |
| `QueryCache.init` | `llev_query_cache_new` | project ABI operation |
| `QueryCache.query` | `llev_query_cache_query_utf8`, `llev_query_cache_query_bytes`, `llev_query_cache_query_u64` | project ABI operation |
| `QueryCache.resetStats` | `llev_query_cache_reset_stats` | project ABI operation |
| `QueryCache.stats` | `llev_query_cache_stats` | project ABI operation |
| `QueryCursor.close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `QueryCursor.next` | `llev_query_cursor_next_batch` | streaming result traversal and batch leases |
| `QueryCursor.nextBatch` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `Transducer.close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.init` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.query` | `llev_transducer_query_utf8`, `llev_transducer_query_bytes`, `llev_transducer_query_u64`, `llev_transducer_query_pattern` | domain-preserving dictionary query; phonetic-pattern dictionary query |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `Status` | Typed native status or error carrier | open RawRepresentable value preserves statuses introduced by newer compatible API revisions |
| `Algorithm` | Edit-distance algorithm selection | Public facade type |
| `QueryOrder` | Result traversal ordering | Public facade type |
| `PhoneticRuleSetKind` | Built-in phonetic rule-set selection | Public facade type |
| `OperationApplicability` | Generalized-operation applicability selection | Public facade type |
| `UniversalVariant` | Universal edit-automaton variant selection | Public facade type |
| `QueryCursor.next` | One-shot owned-result iteration | Public facade protocol |
| `QueryCursor.reduceBatches` | Bounded batch/reducer traversal | Public facade protocol |

Native operations omitted from the public-symbol table are deliberately
encapsulated by the facade. The generated completeness matrix records every
such operation with its reviewed rationale; an unreasoned absence fails CI.

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | The facade iterator protocol | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | The facade batch/reducer protocol | It amortizes the foreign boundary and keeps borrowed views inside one lexical lease. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Close long-lived cursors explicitly and use lexical `defer`; `deinit` is a safety net whose timing is not an API guarantee.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

C statuses are converted to throwing Swift errors with the native diagnostic retained.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Independent values are reentrant. Iterator mutation is exclusive under Swift value-access rules; never escape a batch callback buffer.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Reuse transducers for repeated queries against the same resource.
- Prefer streaming cursors to whole-result materialization.
- Prefer batch/reducer APIs when per-match boundary crossings dominate.
- Keep Unicode, byte, and token domains explicit to avoid transcoding.
- Measure native, WASM, and WASI paths independently; they have different
  startup and marshalling costs but identical query semantics.

No host wrapper should cache unbounded query results. Applications that add a
memo use a revision key and a hard entry/weight bound; eviction may be
approximate because all values remain derivable from the retained snapshot.

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model](../../docs/security/binding-trust-model.md).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md); never infer compatibility from a
package version alone.

When loading fails, check—in order—the documented runtime/toolchain version,
CPU/OS artifact, native-access permission, loader search path, dependent
interop package pin, and process-wide JavaScript runtime identity. When a query
fails after construction, report the typed status and copied diagnostic before
reducing the case to the smallest dictionary/query pair.

## Maintainer checklist

1. Update the machine-readable binding model before changing a public symbol.
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

<!-- END GENERATED BINDING OPERATIONS -->
