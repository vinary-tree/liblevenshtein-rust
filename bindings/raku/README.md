# Liblevenshtein for Raku

Fast edit distances and snapshot-consistent fuzzy search from Raku, backed by
liblevenshtein-rust. `NativeCall` reaches the stable C ABI while
`Vinary::Tree::Interop` carries retained dictionaries from libdictenstein or
customer-defined providers without serialization.

For repeated complete queries, construct `QueryCache.new(:$transducer)` and
call `.query` with the same `Str`, `Blob`, or u64-token inputs accepted by a
transducer. The native implementation combines TinyLFU approximate-frequency
admission with SIEVE eviction under hard entry and logical-weight bounds.
Approximation changes residency only: every miss returns the exact result.
The cache is exclusive and deliberately contains no lock, so parallel callers
shard one cache per worker. See the package Pod and the
[query-cache design](../../docs/bindings/query-cache.md) for lifecycle,
revision, and workload-selection guidance.

## Standalone distance families

Each function below accepts two values of the **same** domain: `Str` compares
Unicode scalar values, `Blob` compares raw bytes (including invalid UTF-8), and
`Positional` compares unsigned 64-bit tokens, including `2**64 - 1`. Mixed
domains are rejected; no implicit encoding or truncation changes a score.

| Function | Edit rule | Return when no `:threshold` is supplied |
|---|---|---|
| `distance` | Insert, delete, substitute | Exact nonnegative integer |
| `damerau-distance` | Optimal-string-alignment adjacent swap | Exact nonnegative integer |
| `true-damerau-distance` | Unrestricted Damerau-Levenshtein | Exact nonnegative integer |
| `merge-and-split-distance` | Standard edits plus symmetric one/two-unit operations | Exact nonnegative integer |
| `hamming-distance` | Substitution only; equal lengths required | Integer, or `Nil` for unequal lengths |
| `indel-distance` | Insert and delete only | Exact nonnegative integer |
| `affine-gap-distance` | Configurable substitution and gap-run costs | Integer, or `Nil` when no representable path exists |

The optional `:threshold($bound)` is inclusive: a score above it returns
`Nil`. Hamming's unequal-length case and affine's unrepresentable/overflow
case also return `Nil` even without a threshold. At the C ABI these are
distinguished as `SIZE_MAX - 2` (undefined) versus `SIZE_MAX - 1`
(above bound); invalid native inputs are `SIZE_MAX` and become an exception
in Raku. Inputs outside the unsigned token/cost range also throw before the
native call. Affine costs are nonnegative integers; a gap of length `k` costs
`gap-open + k * gap-extend`. The thresholded affine entry point currently
computes the exact Gotoh result before comparing it to the bound, whereas
thresholded indel uses a bounded native kernel.

```raku
use Liblevenshtein;

say distance('kitten', 'sitting'); # 3
say indel-distance(Buf.new(0xff), Buf.new(0x00)); # 2
say hamming-distance([0, 2**64 - 1], [0, 0]); # 1

my $costs = AffineGapCosts.new(
    gap-open => 3, gap-extend => 2, substitution => 10,
);
say affine-gap-distance('a', 'abcd', $costs); # 9
say affine-gap-distance('a', 'abcd', $costs, :threshold(8)).defined; # False
```

Standard `distance` already selects its native Myers/ASCII or runtime SIMD
path when applicable. These kernels are implementation details, not distinct
scoring rules or independently selectable Raku APIs. Performance and
unsupported-path evidence are in
[the distance-family benchmark](benchmark/README.md).

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | Raku |
| Languages/runtime | Raku through Rakudo and NativeCall |
| Support tier | Tier 3 |
| Distribution | zef/fez distribution `Liblevenshtein` |
| Native boundary | `NativeCall` reaches the stable C ABI and `Vinary::Tree::Interop` carries retained dictionary resources between independent distributions. |
| Canonical facade source | [`bindings/raku/lib/Liblevenshtein.rakumod`](../../bindings/raku/lib/Liblevenshtein.rakumod) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/raku/t/01-conformance.rakutest`](../../bindings/raku/t/01-conformance.rakutest). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
raku -Ibindings/raku/lib -I../vinary-tree-interop/bindings/raku/lib -I../libdictenstein/bindings/raku/lib bindings/raku/t/01-conformance.rakutest
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

`Str`, `Blob`, and `Positional` token values preserve Unicode-scalar, byte, and u64-token domains through multi methods. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `abi-version` | `llev_abi_version` | ABI compatibility and feature discovery |
| `affine-gap-distance` | `llev_affine_gap_distance`, `llev_affine_gap_distance_threshold`, `llev_affine_gap_distance_bytes`, `llev_affine_gap_distance_bytes_threshold`, `llev_affine_gap_distance_u64`, `llev_affine_gap_distance_u64_threshold` | project ABI operation |
| `api-revision` | `llev_api_revision` | ABI compatibility and feature discovery |
| `build-features` | `llev_build_features` | ABI compatibility and feature discovery |
| `damerau-distance` | `llev_damerau_distance`, `llev_damerau_distance_threshold`, `llev_damerau_distance_bytes`, `llev_damerau_distance_bytes_threshold`, `llev_damerau_distance_u64`, `llev_damerau_distance_u64_threshold` | standalone exact or thresholded distance |
| `distance` | `llev_distance`, `llev_distance_threshold`, `llev_distance_bytes`, `llev_distance_bytes_threshold`, `llev_distance_u64`, `llev_distance_u64_threshold` | standalone exact or thresholded distance |
| `hamming-distance` | `llev_hamming_distance`, `llev_hamming_distance_threshold`, `llev_hamming_distance_bytes`, `llev_hamming_distance_bytes_threshold`, `llev_hamming_distance_u64`, `llev_hamming_distance_u64_threshold` | project ABI operation |
| `indel-distance` | `llev_indel_distance`, `llev_indel_distance_threshold`, `llev_indel_distance_bytes`, `llev_indel_distance_bytes_threshold`, `llev_indel_distance_u64`, `llev_indel_distance_u64_threshold` | project ABI operation |
| `merge-and-split-distance` | `llev_merge_and_split_distance`, `llev_merge_and_split_distance_threshold`, `llev_merge_and_split_distance_bytes`, `llev_merge_and_split_distance_bytes_threshold`, `llev_merge_and_split_distance_u64`, `llev_merge_and_split_distance_u64_threshold` | standalone merge-and-split distance |
| `PhoneticPattern.accepts` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.new` | `llev_phonetic_pattern_compile_regex`, `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet.apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.elems` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.new` | `llev_phonetic_rules_parse`, `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `QueryCache.clear` | `llev_query_cache_clear` | project ABI operation |
| `QueryCache.close` | `llev_query_cache_free` | project ABI operation |
| `QueryCache.new` | `llev_query_cache_new` | project ABI operation |
| `QueryCache.query` | `llev_query_cache_query_utf8`, `llev_query_cache_query_bytes`, `llev_query_cache_query_u64` | project ABI operation |
| `QueryCache.reset-stats` | `llev_query_cache_reset_stats` | project ABI operation |
| `QueryCache.stats` | `llev_query_cache_stats` | project ABI operation |
| `QueryCursor.close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `QueryCursor.next-batch` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `Transducer.close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.new` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.query` | `llev_transducer_query_utf8`, `llev_transducer_query_bytes`, `llev_transducer_query_u64`, `llev_transducer_query_pattern` | domain-preserving dictionary query; phonetic-pattern dictionary query |
| `Transducer.snapshot` | `llev_transducer_snapshot` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.unit-domain` | `llev_transducer_unit_domain` | transducer lifecycle, snapshot, or domain metadata |
| `true-damerau-distance` | `llev_true_damerau_distance`, `llev_true_damerau_distance_threshold`, `llev_true_damerau_distance_bytes`, `llev_true_damerau_distance_bytes_threshold`, `llev_true_damerau_distance_u64`, `llev_true_damerau_distance_u64_threshold` | standalone true-Damerau distance |
| `X::Liblevenshtein` | `llev_last_error_message` | typed failure diagnostics |

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
| `reduce-batches` | Bounded batch/reducer traversal | Public facade protocol |

Native operations omitted from the public-symbol table are deliberately
encapsulated by the facade. The generated completeness matrix records every
such operation with its reviewed rationale; an unreasoned absence fails CI.

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | The facade iterator protocol | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | The facade batch/reducer protocol | It amortizes NativeCall while copying each bounded batch and settling its exact generation before host values escape. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Call `.close` or use `LEAVE` for transducers, cursors, patterns, and rule sets; `DESTROY` is only leak containment.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Every Raku match and bounded batch is copied into host-owned values before
the native generation is released. Values may outlive the cursor; no raw native
pointer or lexical lease is exposed to application code.

## Errors and failure containment

Non-OK statuses become `X::Liblevenshtein` values carrying the exact numeric status, operation, and copied diagnostic.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Inspect
`X::Liblevenshtein.status` and `operation`, never diagnostic prose; the copied
message is human context.

## Concurrency and reentrancy

Immutable transducers and independent cursors may run on separate threads. A cursor remains exclusive and single-consumer.

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
