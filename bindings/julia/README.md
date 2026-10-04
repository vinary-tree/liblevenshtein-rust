# Julia binding

`Liblevenshtein` is the natural Julia package for fast edit distances and
snapshot-consistent fuzzy search over Vinary Tree dictionary resources. It
calls the stable versioned C ABI directly and shares provider handles through
`VinaryTreeInterop`; dictionary data is never serialized or copied merely to
cross package boundaries.

The package version is `4.0.0-rc.6`. This feature branch prepares the package
for the Julia General registry but does not publish it.

See [`Liblevenshtein/README.md`](Liblevenshtein/README.md) for installation,
examples, the complete public surface, ownership, concurrency, error,
performance, security, and release contracts.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Julia |
| Languages/runtime | Julia 1.10+ |
| Support tier | Tier 3 |
| Distribution | General-registry package `Liblevenshtein` |
| Native boundary | `ccall` reaches the stable C ABI and `VinaryTreeInterop` carries retained dictionary resources between independent packages. |
| Canonical facade source | [`bindings/julia/Liblevenshtein/src/Liblevenshtein.jl`](../../bindings/julia/Liblevenshtein/src/Liblevenshtein.jl) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/julia/Liblevenshtein/test/runtests.jl`](../../bindings/julia/Liblevenshtein/test/runtests.jl). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
julia --project=bindings/julia/Liblevenshtein -e 'using Pkg; Pkg.test()'
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

`String`, `AbstractVector{UInt8}`, and integer vectors preserve Unicode-scalar, byte, and u64-token domains through multiple dispatch. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `abi_version` | `llev_abi_version` | ABI compatibility and feature discovery |
| `advance!` | `llev_generalized_online_advance`, `llev_universal_online_advance` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `api_revision` | `llev_api_revision` | ABI compatibility and feature discovery |
| `are_phonetically_similar` | `llev_phonetic_feature_relation` | IPA feature classification and relations |
| `articulatory_distance` | `llev_phonetic_articulatory_distance` | articulatory phonetic distance |
| `articulatory_edit_distance` | `llev_phonetic_articulatory_edit_distance` | articulatory phonetic distance |
| `build_features` | `llev_build_features` | ABI compatibility and feature discovery |
| `cache_stats` | `llev_query_cache_stats` | project ABI operation |
| `cancel!` | `llev_wallbreaker_cursor_cancel` | project ABI operation |
| `characters_with_features` | `llev_phonetic_chars_with_features` | IPA feature classification and relations |
| `clear!` | `llev_query_cache_clear` | project ABI operation |
| `close!` | `llev_transducer_free`, `llev_query_cache_free`, `llev_query_cursor_free`, `llev_phonetic_pattern_free`, `llev_phonetic_rules_free`, `llev_phonetic_grep_free`, `llev_phonetic_dictionary_free`, `llev_phonetic_online_free`, `llev_phonetic_online_stream_free`, `llev_phonetic_token_free`, `llev_phonetic_transducer_free`, `llev_generalized_automaton_free`, `llev_generalized_online_free`, `llev_universal_automaton_free`, `llev_universal_online_free`, `llev_wallbreaker_free`, `llev_wallbreaker_cursor_free` | transducer lifecycle, snapshot, or domain metadata; project ABI operation; streaming result traversal and batch leases; compiled phonetic-pattern lifecycle and matching; phonetic rule-set lifecycle and rewriting; word-boundary phonetic search and configuration; normalized phonetic dictionary construction, query, and updates; character-level phonetic search and scanner lifecycle; token-sequence phonetic matching and detail ownership; incremental phonetic rewriting and lifecycle; runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `compiled_phonetic_bytes` | `llev_owned_bytes_free`, `llev_phonetic_rules_to_bytes`, `llev_phonetic_pattern_to_bytes` | versioned compiled-byte ownership; versioned compiled phonetic-rule bytes; versioned compiled phonetic-pattern bytes |
| `distance` | `llev_distance`, `llev_distance_threshold`, `llev_distance_bytes`, `llev_distance_bytes_threshold`, `llev_distance_u64`, `llev_distance_u64_threshold` | standalone exact or thresholded distance |
| `distance_config` | `llev_phonetic_grep_distance_config` | word-boundary phonetic search and configuration |
| `evaluate` | `llev_generalized_automaton_evaluate_utf8`, `llev_universal_automaton_evaluate` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `expand_feature_based` | `llev_phonetic_expand_feature_based` | IPA feature-driven expansion |
| `expand_phonetic_alternatives` | `llev_phonetic_expand` | bounded reverse phonetic expansion |
| `expand_phonetic_with_costs` | `llev_phonetic_expand_with_costs` | bounded reverse phonetic expansion |
| `feature_set_distance` | `llev_phonetic_feature_set_distance` | IPA feature classification and relations |
| `feed!` | `llev_phonetic_online_stream_feed`, `llev_phonetic_transducer_feed` | character-level phonetic search and scanner lifecycle; incremental phonetic rewriting and lifecycle |
| `finish!` | `llev_phonetic_online_stream_finish`, `llev_phonetic_transducer_finish` | character-level phonetic search and scanner lifecycle; incremental phonetic rewriting and lifecycle |
| `GeneralizedAutomaton` | `llev_generalized_automaton_new` | runtime generalized-automaton lifecycle and prefix evaluation |
| `insert!` | `llev_phonetic_dictionary_update` | normalized phonetic dictionary construction, query, and updates |
| `is_free_phonetic_substitution` | `llev_phonetic_feature_relation` | IPA feature classification and relations |
| `load_compiled_phonetic_pattern` | `llev_phonetic_pattern_from_bytes` | versioned compiled phonetic-pattern bytes |
| `load_compiled_phonetic_rules` | `llev_phonetic_rules_from_bytes` | versioned compiled phonetic-rule bytes |
| `load_phonetic_pattern` | `llev_phonetic_pattern_load_llre_file` | trusted .llre loading with native imports |
| `load_phonetic_rules` | `llev_phonetic_rules_load_file` | trusted .llev loading with native includes |
| `match_distance` | `llev_phonetic_grep_matches` | word-boundary phonetic search and configuration |
| `merge_and_split_distance` | `llev_merge_and_split_distance`, `llev_merge_and_split_distance_threshold`, `llev_merge_and_split_distance_bytes`, `llev_merge_and_split_distance_bytes_threshold`, `llev_merge_and_split_distance_u64`, `llev_merge_and_split_distance_u64_threshold` | standalone merge-and-split distance |
| `NativeError` | `llev_last_error_message` | typed failure diagnostics |
| `next_batch!` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch`, `llev_wallbreaker_cursor_next_batch`, `llev_wallbreaker_cursor_release_batch` | streaming result traversal and batch leases; project ABI operation |
| `normalize` | `llev_phonetic_transducer_normalize` | incremental phonetic rewriting and lifecycle |
| `normalized_query` | `llev_phonetic_online_normalized_query` | character-level phonetic search and scanner lifecycle |
| `observation` | `llev_generalized_online_observation`, `llev_universal_online_observation` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `online` | `llev_generalized_online_new_utf8`, `llev_universal_online_new` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `optimal_string_alignment_distance` | `llev_damerau_distance`, `llev_damerau_distance_threshold`, `llev_damerau_distance_bytes`, `llev_damerau_distance_bytes_threshold`, `llev_damerau_distance_u64`, `llev_damerau_distance_u64_threshold` | standalone exact or thresholded distance |
| `pattern_pieces` | `llev_wallbreaker_split_utf8` | project ABI operation |
| `phonetic_features` | `llev_phonetic_features` | IPA feature classification and relations |
| `PhoneticGrep` | `llev_phonetic_grep_new` | word-boundary phonetic search and configuration |
| `PhoneticNormalizedDictionary` | `llev_phonetic_dictionary_new` | normalized phonetic dictionary construction, query, and updates |
| `PhoneticOnlineGrep` | `llev_phonetic_online_new` | character-level phonetic search and scanner lifecycle |
| `PhoneticPattern` | `llev_phonetic_pattern_compile_regex`, `llev_phonetic_pattern_compile_llre`, `llev_phonetic_pattern_size`, `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet` | `llev_owned_string_free`, `llev_phonetic_rules_parse`, `llev_phonetic_rules_builtin`, `llev_phonetic_rules_len`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticTokenGrep` | `llev_phonetic_token_new` | token-sequence phonetic matching and detail ownership |
| `PhoneticTransducer` | `llev_phonetic_transducer_new` | incremental phonetic rewriting and lifecycle |
| `query` | `llev_transducer_query_utf8`, `llev_transducer_query_bytes`, `llev_transducer_query_u64`, `llev_query_cache_query_utf8`, `llev_query_cache_query_bytes`, `llev_query_cache_query_u64`, `llev_transducer_query_pattern`, `llev_phonetic_dictionary_query`, `llev_phonetic_candidates_free`, `llev_wallbreaker_query_utf8` | domain-preserving dictionary query; project ABI operation; phonetic-pattern dictionary query; normalized phonetic dictionary construction, query, and updates; owned normalized-dictionary candidate release |
| `QueryCache` | `llev_query_cache_new` | project ABI operation |
| `reduce_batches!` | `llev_query_cursor_reduce` | streaming result traversal and batch leases |
| `remove!` | `llev_phonetic_dictionary_update` | normalized phonetic dictionary construction, query, and updates |
| `reset!` | `llev_phonetic_transducer_reset` | incremental phonetic rewriting and lifecycle |
| `reset_stats!` | `llev_query_cache_reset_stats` | project ABI operation |
| `scan` | `llev_phonetic_online_scan`, `llev_phonetic_online_matches_free`, `llev_phonetic_token_scan`, `llev_phonetic_token_matches_free` | character-level phonetic search and scanner lifecycle; token-sequence phonetic matching and detail ownership |
| `scan_line` | `llev_phonetic_grep_scan_line` | word-boundary phonetic search and configuration |
| `scan_text` | `llev_phonetic_grep_scan_text` | word-boundary phonetic search and configuration |
| `similar_phonetic_chars` | `llev_phonetic_similar_chars` | IPA feature classification and relations |
| `snapshot` | `llev_transducer_snapshot` | transducer lifecycle, snapshot, or domain metadata |
| `streaming` | `llev_phonetic_online_stream_new` | character-level phonetic search and scanner lifecycle |
| `syllable_boundaries` | `llev_phonetic_syllable_boundaries` | syllable count and boundary heuristics |
| `syllable_count` | `llev_phonetic_syllable_count` | syllable count and boundary heuristics |
| `Transducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `true_damerau_distance` | `llev_true_damerau_distance`, `llev_true_damerau_distance_threshold`, `llev_true_damerau_distance_bytes`, `llev_true_damerau_distance_bytes_threshold`, `llev_true_damerau_distance_u64`, `llev_true_damerau_distance_u64_threshold` | standalone true-Damerau distance |
| `unit_domain` | `llev_transducer_unit_domain` | transducer lifecycle, snapshot, or domain metadata |
| `UniversalAutomaton` | `llev_universal_automaton_new` | universal-automaton lifecycle, policies, and prefix evaluation |
| `voicing_pair` | `llev_phonetic_voicing_pair` | IPA feature classification and relations |
| `WallBreakerMatcher` | `llev_wallbreaker_new_utf8` | project ABI operation |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `Status` | Typed native status or error carrier | Public facade type |
| `Algorithm` | Edit-distance algorithm selection | Public facade type |
| `QueryOrder` | Result traversal ordering | Public facade type |
| `PhoneticRuleSetKind` | Built-in phonetic rule-set selection | Public facade type |
| `OperationApplicability` | Generalized-operation applicability selection | Public facade type |
| `UniversalVariant` | Universal edit-automaton variant selection | Public facade type |
| `QueryCursor` | One-shot owned-result iteration | Public facade protocol |
| `reduce_batches!` | Bounded batch/reducer traversal | Public facade protocol |

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

Use `close` in `finally` for transducers, cursors, patterns, and rule sets; finalizers are leak containment rather than deterministic scheduling.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Non-OK statuses become `NativeError` values carrying the exact numeric status, operation, and copied diagnostic.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Immutable transducers and independent cursors may run in separate tasks. A cursor and its live lexical batch remain exclusive and single-consumer.

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
