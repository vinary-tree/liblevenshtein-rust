# Vinary Tree liblevenshtein for Ruby

This gem supports maintained Ruby 3.3 through the latest Ruby release. Queries
are one-shot `Enumerable` objects backed by a native cursor; only a bounded
batch is leased and each yielded `Match` owns its term. A query captures its
dictionary revision at construction and can outlive the source dictionary.

Any modular dictionary gem can implement `with_resource { |context, vtable| }
to participate in O(1) retained-resource handoff. The libdictenstein gem does
so without serialization or an object-format conversion.

Set `LIBLEVENSHTEIN_LIBRARY` for a source-tree build. Release gems contain the
platform shared library under
`lib/vinary_tree/liblevenshtein/native/<platform>/`; a system installation
remains a supported loader fallback.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Ruby |
| Languages/runtime | Ruby 3.3+ |
| Support tier | Tier 2 |
| Distribution | RubyGems package `liblevenshtein` |
| Native boundary | Ruby Fiddle calls the stable C ABI; modular producers yield the two-word resource without serialization. |
| Canonical facade source | [`bindings/ruby/lib/vinary_tree/liblevenshtein`](../../bindings/ruby/lib/vinary_tree/liblevenshtein) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/ruby/test/test_cross_project.rb`](../../bindings/ruby/test/test_cross_project.rb). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
ruby -Ibindings/ruby/lib -I../libdictenstein/bindings/ruby/lib bindings/ruby/test/test_cross_project.rb
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

Ruby strings use explicit encoding rules; byte strings and integer arrays select non-Unicode domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `Error` | `llev_last_error_message` | typed failure diagnostics |
| `Error#status` | `llev_last_error_message` | typed failure diagnostics |
| `Liblevenshtein.damerau_distance` | `llev_damerau_distance` | standalone exact or thresholded distance |
| `Liblevenshtein.damerau_distance_threshold` | `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `Liblevenshtein.distance` | `llev_distance` | standalone exact or thresholded distance |
| `Liblevenshtein.distance_threshold` | `llev_distance_threshold` | standalone exact or thresholded distance |
| `Liblevenshtein.true_damerau_distance` | `llev_true_damerau_distance` | standalone true-Damerau distance |
| `Liblevenshtein.true_damerau_distance_threshold` | `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |
| `PhoneticPattern#close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern#matches?` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern#size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.compile_llre` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.compile_regex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet#apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet#close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet#length` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.builtin` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.parse` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `Query#close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `Query#each` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `Transducer#close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer#query` | `llev_transducer_query_utf8` | domain-preserving dictionary query |
| `Transducer#query_bytes` | `llev_transducer_query_bytes` | domain-preserving dictionary query |
| `Transducer#query_pattern` | `llev_transducer_query_pattern` | phonetic-pattern dictionary query |
| `Transducer#query_u64` | `llev_transducer_query_u64` | domain-preserving dictionary query |
| `Transducer.new` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `Status` | Typed native status or error carrier | all stable statuses are named constants while unknown future integer values remain representable on Error#status |
| `Algorithm` | Edit-distance algorithm selection | STANDARD/TRANSPOSITION/MERGE_AND_SPLIT/DAMERAU_LEVENSHTEIN constants; Transducer aliases remain compatible |
| `QueryOrder` | Result traversal ordering | TRAVERSAL/DISTANCE_THEN_TERM constants |
| `PhoneticRuleSetKind` | Built-in phonetic rule-set selection | ENGLISH_ORTHOGRAPHY/ENGLISH_PHONETIC constants; PhoneticRuleSet aliases remain compatible |
| `Query#each` | One-shot owned-result iteration | Public facade protocol |

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
| Maximum result throughput | Drain the facade iterator; no public reducer is exposed | The iterator still amortizes native calls with bounded internal batches, then releases each lease before exposing host-owned matches. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Prefer block forms and `ensure { cursor.close }`; finalizers only prevent permanent leaks.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Iterator results are copied into host-owned values before their native batch
lease is released. They remain valid after iteration advances or the cursor is
closed; no raw pointer or borrowed native view reaches user code.

## Errors and failure containment

Non-OK statuses become typed Ruby exceptions exposing the status symbol and native diagnostic.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Separate native handles are reentrant. Enumerate one cursor on one fiber/thread and never retain callback-scoped buffers.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Reuse transducers for repeated queries against the same resource.
- Prefer streaming cursors to whole-result materialization.
- Drain each cursor once; the iterator already fetches bounded native batches before materializing host-owned matches.
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
