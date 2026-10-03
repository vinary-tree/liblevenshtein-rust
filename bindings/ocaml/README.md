# OCaml binding (Tier 3)

The OCaml 5 binding consumes retained `Vinary_tree_interop.resource` values
published by the separate `vinary_tree_libdictenstein` package. Native query
cursors are lazy and snapshot-stable; `to_seq` pulls one match at a time and
`fold_batches` crosses the FFI once per bounded batch.

The opam package is `liblevenshtein`; its public module is
`Vinary_tree_liblevenshtein`. The package includes an [odoc guide](doc/index.mld)
and an API reference generated from the documented
[`vinary_tree_liblevenshtein.mli`](vinary_tree_liblevenshtein.mli).

## Build and verify package documentation

With the exact `vinary-tree-interop` dependency installed in an opam switch,
run `opam install odoc` and then
`opam exec -- dune build --root bindings/ocaml @doc`. Dune writes the guide
and module reference beneath `bindings/ocaml/_build/default/_doc/_html/`.
The binding CI checks both generated entry points; the opam source-archive
contract checks that the guide, Dune stanza, interface comments, and this
package README survive deterministic staging.

Only after the source release is immutable and the opam-repository pull
request has merged, dispatch `OCaml package documentation readback` from the
exact source tag. That read-only workflow checks the version-specific
`ocaml.org` package page, odoc guide, and module API; a candidate-only source
tree is not evidence of public documentation. The [OCaml documentation
guide](https://ocaml.org/docs/generating-documentation) explains odoc's
generation and package-page conventions.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | OCaml |
| Languages/runtime | OCaml 5 through dune/opam |
| Support tier | Tier 3 |
| Distribution | opam package `liblevenshtein` |
| Native boundary | C stubs call the stable ABI and consume `Vinary_tree_interop.resource` values from independent producers. |
| Canonical facade source | [`bindings/ocaml/vinary_tree_liblevenshtein.mli`](vinary_tree_liblevenshtein.mli) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/language-bindings.md) before implementing a custom provider
and the [family hub](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](https://raw.githubusercontent.com/vinary-tree/liblevenshtein-rust/master/docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/ocaml/test/snapshot.ml`](test/snapshot.ml). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
opam exec -- dune runtest --root bindings/ocaml
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
| Match/batch | Matches and batches are copied into OCaml-owned values before their native lease is released. |

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

Strings carry UTF-8; bytes and int64 arrays select raw byte and packed-token domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `apply_rules` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `cached_query` | `llev_query_cache_query_utf8` | project ABI operation |
| `cached_query_bytes` | `llev_query_cache_query_bytes` | project ABI operation |
| `cached_query_u64` | `llev_query_cache_query_u64` | project ABI operation |
| `clear_query_cache` | `llev_query_cache_clear` | project ABI operation |
| `close_pattern` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `close_query_cache` | `llev_query_cache_free` | project ABI operation |
| `close_rules` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `close_transducer` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `cursor_close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `damerau_distance` | `llev_damerau_distance` | standalone exact or thresholded distance |
| `damerau_distance_threshold` | `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `distance` | `llev_distance` | standalone exact or thresholded distance |
| `distance_threshold` | `llev_distance_threshold` | standalone exact or thresholded distance |
| `llre_pattern` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `next` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `next_batch` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `pattern_matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `pattern_size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `phonetic_rules` | `llev_phonetic_rules_parse`, `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `query` | `llev_transducer_query_utf8` | domain-preserving dictionary query |
| `query_bytes` | `llev_transducer_query_bytes` | domain-preserving dictionary query |
| `query_cache` | `llev_query_cache_new` | project ABI operation |
| `query_cache_stats` | `llev_query_cache_stats` | project ABI operation |
| `query_pattern` | `llev_transducer_query_pattern` | phonetic-pattern dictionary query |
| `query_u64` | `llev_transducer_query_u64` | domain-preserving dictionary query |
| `regex_pattern` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `reset_query_cache_stats` | `llev_query_cache_reset_stats` | project ABI operation |
| `rules_length` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `to_seq` | `llev_query_cursor_next_batch` | streaming result traversal and batch leases |
| `transducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `true_damerau_distance` | `llev_true_damerau_distance` | standalone true-Damerau distance |
| `true_damerau_distance_threshold` | `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `algorithm` | Edit-distance algorithm selection | Public facade type |
| `query_order` | Result traversal ordering | Public facade type |
| `phonetic_rules` | Built-in phonetic rule-set selection | string selectors "english-orthography"/"english-phonetic" |
| `next` | One-shot owned-result iteration | Public facade protocol |
| `fold_batches` | Bounded batch/reducer traversal | Public facade protocol |

### Facade-encapsulated model values

| Model value | Idiomatic treatment |
|---|---|
| `status` | failures raise Stdlib Failure with the native message; the numeric status is not re-exposed |
| `operationApplicability` | the API-revision-5 selection is not yet projected into this idiomatic facade; its language-family automata parity task owns that adapter |
| `universalVariant` | the API-revision-5 selection is not yet projected into this idiomatic facade; its language-family automata parity task owns that adapter |

Native operations omitted from the public-symbol table are deliberately
encapsulated by the facade. The generated completeness matrix records every
such operation with its reviewed rationale; an unreasoned absence fails CI.

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | `to_seq` within `Fun.protect` | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | `fold_batches` with bounded native pages | It amortizes the foreign boundary while returning bounded, host-owned arrays. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/bindings/api-surface-map.json) and the [generated completeness matrix](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Use the explicit `close` functions or `Fun.protect`; GC finalizers are only a last-resort retain release.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Every match and batch is copied into OCaml-owned values before return.
The values remain valid after iteration advances or the cursor closes. The
lazy sequence does not close its cursor; scope it with `Fun.protect`.

## Errors and failure containment

C status failures raise OCaml `Failure` with a copied native diagnostic; this facade does not expose a typed status exception.

Malformed UTF-8, unsupported domains, incompatible resources,
closed handles, invalid bounds, allocation failures, provider faults, and
contained Rust panics remain distinct native causes. This facade raises
`Failure` with a copied diagnostic but does not expose the status code; do not
parse the message as a stable protocol.

## Concurrency and reentrancy

Independent handles are domain-safe according to the documented capability flags. A cursor remains single-consumer; returned matches and batches are owned OCaml values.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Reuse transducers for repeated queries against the same resource.
- Use `to_seq` for lazy results and close its cursor on all paths.
- Use `fold_batches` when per-match foreign-boundary crossings dominate.
- Keep Unicode, byte, and token domains explicit to avoid transcoding.
- Use the built-in `query_cache` with hard entry and weight limits for repeated
  queries; eviction changes performance, never snapshot-consistent results.

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/security/binding-trust-model.md).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md); never infer compatibility from a
package version alone.

When loading fails, check the OCaml/opam
version, native library and its dependent interop pin, C-stub linkage, and
loader search path. When a query fails after construction, report the
host-language error and copied diagnostic before reducing the case to the
smallest dictionary/query pair.

## Maintainer checklist

1. Update the machine-readable binding model before changing a public symbol.
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

<!-- END GENERATED BINDING OPERATIONS -->
