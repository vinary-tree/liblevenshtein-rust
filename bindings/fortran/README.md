# Vinary Tree liblevenshtein for Fortran

The Fortran 2018 module exposes all distance families, Unicode/raw-byte/u64
streaming queries, retained dictionary construction, and phonetic automata. A
`query_iterator` leases one bounded native batch and returns one owned
`levenshtein_match` at a time. Finalizers are a safety net; call `close` when
deterministic release matters.

The fpm package is named `liblevenshtein` and depends on the shared
`vinary-tree-interop` module. Link with `-lliblevenshtein`; CMake installations
let applications choose shared or static linkage.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Fortran |
| Languages/runtime | Fortran 2018 through fpm |
| Support tier | Tier 2 |
| Distribution | fpm package `liblevenshtein` |
| Native boundary | `iso_c_binding` declarations call the stable C ABI and share the interop resource derived type. |
| Canonical facade source | [`bindings/fortran/src/vinary_tree_liblevenshtein.f90`](../../bindings/fortran/src/vinary_tree_liblevenshtein.f90) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/fortran/test/test_distance.f90`](../../bindings/fortran/test/test_distance.f90). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
fpm test --profile release --package liblevenshtein
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

Character data is UTF-8 marshalled explicitly; `integer(c_int8_t)` and `integer(c_int64_t)` arrays preserve byte/token domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `builtin_phonetic_rules` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `compile_phonetic_llre` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `compile_phonetic_regex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `damerau_distance` | `llev_damerau_distance` | standalone exact or thresholded distance |
| `damerau_distance_threshold` | `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `levenshtein_distance` | `llev_distance` | standalone exact or thresholded distance |
| `levenshtein_distance_threshold` | `llev_distance_threshold` | standalone exact or thresholded distance |
| `new_transducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `parse_phonetic_rules` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `phonetic_pattern%close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `phonetic_pattern%matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `phonetic_pattern%size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `phonetic_rule_set%apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `phonetic_rule_set%close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `phonetic_rule_set%length` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `query_iterator%close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `query_iterator%next` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `transducer%close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `transducer%query_bytes` | `llev_transducer_query_bytes` | domain-preserving dictionary query |
| `transducer%query_pattern` | `llev_transducer_query_pattern` | phonetic-pattern dictionary query |
| `transducer%query_text` | `llev_transducer_query_utf8` | domain-preserving dictionary query |
| `transducer%query_u64` | `llev_transducer_query_u64` | domain-preserving dictionary query |
| `true_damerau_distance` | `llev_true_damerau_distance` | standalone true-Damerau distance |
| `true_damerau_distance_threshold` | `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `llev_ok` | Typed native status or error carrier | only OK and END carry named parameters; other codes stay numeric |
| `llev_standard` | Edit-distance algorithm selection | llev_standard/llev_transposition/llev_merge_and_split/llev_damerau_levenshtein |
| `llev_traversal` | Result traversal ordering | llev_traversal/llev_distance_then_term |
| `query_iterator%next` | One-shot owned-result iteration | Public facade protocol |

### Facade-encapsulated model values

| Model value | Idiomatic treatment |
|---|---|
| `phoneticRuleSetKind` | builtin_phonetic_rules takes a raw integer kind; named constants are not provided |
| `reducer` | no public batch-reduction entry point; the safe iterator leases and materializes one bounded native batch at a time internally |

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

Call each derived handle's `close` method; final procedures are exceptional-path protection rather than deterministic scheduling.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Procedures return or raise the module's status abstraction while preserving the native diagnostic for reporting.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Independent derived handles are reentrant. Do not access one mutable iterator from multiple images or retain a leased C pointer.

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
