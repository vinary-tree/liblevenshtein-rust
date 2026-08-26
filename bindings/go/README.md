# Vinary Tree Go bindings

The Go package is an idiomatic streaming wrapper around liblevenshtein's stable
C ABI. It supports Go 1.25 and newer. `Iterator.Next` owns only the current Go
result while the iterator leases one bounded native batch; it never collects an
entire query into a slice. `Close` is deterministic and finalizers are a safety
net.

Dictionary producers implement the small `interop.DictionaryResource`
interface from
`github.com/vinary-tree/vinary-tree-interop/bindings/go/v4`.
Native Vinary
Tree producers lend two pointer-sized words for the constructor and are retained
in O(1) by the transducer.

During source-tree development, set `CGO_LDFLAGS=-L../../target/debug` and put
that directory on the platform loader path. Published archives place the shared
library in the platform package or may use the separately installed CMake
package. C/C++ consumers may select static or shared linkage.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Go |
| Languages/runtime | Go 1.25+ with cgo |
| Support tier | Tier 2 |
| Distribution | Go module `github.com/vinary-tree/liblevenshtein-rust/bindings/go/v4` |
| Native boundary | cgo calls the stable C ABI and uses the interop module for two-word retained dictionary resources. |
| Canonical facade source | [`bindings/go/liblevenshtein.go`](../../bindings/go/liblevenshtein.go) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/go/liblevenshtein_test.go`](../../bindings/go/liblevenshtein_test.go). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
go test ./bindings/go
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

Strings use Unicode queries; byte and `[]uint64` entry points preserve raw domain identity. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `BuiltinPhoneticRules` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `CompilePhoneticLLRE` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `CompilePhoneticRegex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `DamerauDistance` | `llev_damerau_distance` | standalone exact or thresholded distance |
| `DamerauDistanceThreshold` | `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `Distance` | `llev_distance` | standalone exact or thresholded distance |
| `DistanceThreshold` | `llev_distance_threshold` | standalone exact or thresholded distance |
| `Error` | `llev_last_error_message` | typed failure diagnostics |
| `Iterator.Close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `Iterator.Next` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `NewTransducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `ParsePhoneticRules` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `PhoneticPattern.Close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.Matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.Size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet.Apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.Close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.Len` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `Transducer.Close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.Query` | `llev_transducer_query_utf8` | domain-preserving dictionary query |
| `Transducer.QueryBytes` | `llev_transducer_query_bytes` | domain-preserving dictionary query |
| `Transducer.QueryPattern` | `llev_transducer_query_pattern` | phonetic-pattern dictionary query |
| `Transducer.QueryU64` | `llev_transducer_query_u64` | domain-preserving dictionary query |
| `TrueDamerauDistance` | `llev_true_damerau_distance` | standalone true-Damerau distance |
| `TrueDamerauDistanceThreshold` | `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `Error.Status` | Typed native status or error carrier | Public facade type |
| `Algorithm` | Edit-distance algorithm selection | Public facade type |
| `QueryOrder` | Result traversal ordering | Public facade type |
| `PhoneticRuleSetKind` | Built-in phonetic rule-set selection | Public facade type |
| `Iterator.Next` | One-shot owned-result iteration | Public facade protocol |

### Facade-encapsulated model values

| Model value | Idiomatic treatment |
|---|---|
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

Call `Close` with `defer` immediately after successful construction. Finalizers report leaks but are not a prompt release mechanism.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Native statuses become Go errors that preserve the symbolic status and native diagnostic; callers may inspect them without parsing strings.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Different handles may be used by different goroutines. Serialize `Next` and `Close` for one iterator and do not retain borrowed native memory.

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
