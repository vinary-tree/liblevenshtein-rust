# Liblevenshtein.jl

Fast Unicode edit distances and snapshot-consistent fuzzy search from Julia,
backed by liblevenshtein-rust.

## Install and verify

This RC package is tested from its source tree. It requires Julia 1.10 or newer,
the `VinaryTreeInterop` Julia package, and the matching native libraries. Point
the loader at the exact liblevenshtein build:

```sh
export LIBLEVENSHTEIN_LIBRARY="$PWD/target/release/libliblevenshtein.so"
julia --project=bindings/julia/Liblevenshtein \
  -e 'using Pkg; Pkg.test()'
```

macOS uses `libliblevenshtein.dylib`; Windows uses
`liblevenshtein.dll`. A future registered artifact will select the platform
binary without this development override.

## Quick start

```julia
using Liblevenshtein

distance("kitten", "sitting")                         # 3
distance("kitten", "sitting"; threshold=2)            # nothing
optimal_string_alignment_distance("ab", "ba")         # 1
true_damerau_distance("CA", "ABC")                    # 2
merge_and_split_distance("m", "rn")                   # 1

# The same functions preserve binary and token domains through dispatch.
distance(UInt8[0xff, 0x00], UInt8[0xff, 0x01])         # 1
distance(UInt64[10, 20], UInt64[20, 10])               # 2
```

For dictionary search, create or receive a `VinaryTreeInterop.Dictionary`, then
retain it in a transducer. Every query captures one immutable dictionary
revision:

```julia
transducer = Transducer(dictionary, ALGORITHM_STANDARD)
try
    matches = collect(query(transducer, "speling", 2;
        order=ORDER_DISTANCE_THEN_TERM))
finally
    close(transducer)
end
```

`dictionary` may come from libdictenstein or a customer-defined provider that
implements `vt.dictionary.v1`; liblevenshtein does not depend on a concrete
dictionary backend.

## Choose an automaton

| Julia value | Semantics | Metric? |
|---|---|---:|
| `ALGORITHM_STANDARD` | Insert, delete, and substitute | yes |
| `ALGORITHM_TRANSPOSITION` | Optimal string alignment with adjacent swaps | no |
| `ALGORITHM_MERGE_AND_SPLIT` | Standard edits plus symmetric merge/split | yes |
| `ALGORITHM_DAMERAU_LEVENSHTEIN` | Unrestricted, history-composable adjacent swaps | yes |

Optimal string alignment and unrestricted Damerau-Levenshtein differ on edit
histories: the former assigns distance 3 from `CA` to `ABC`, while the latter
assigns distance 2. Choose the algorithm when constructing `Transducer`; query
domains, batching, and snapshot behavior are otherwise identical.

## Standalone generalized and universal automata

Use `GeneralizedAutomaton` when the edit grammar itself is runtime data. The
operation set below charges one half for substitution and one for insertion or
deletion. Native observations return the exact Julia `Rational`, so the result
does not depend on floating-point comparison in application code:

```julia
operations = GeneralizedOperationSet(
    GeneralizedOperation(0, 1, 1, :insert),
    GeneralizedOperation(1, 0, 1, :delete),
    GeneralizedOperation(1, 1, 0, :equal;
        applicability=APPLICABILITY_EQUAL),
    GeneralizedOperation(1, 1, 0.5, :substitute),
)

automaton = GeneralizedAutomaton(2, operations)
try
    result = evaluate(automaton, "cat", "cut")
    @assert result.accepting && result.distance == 1 // 2
finally
    close(automaton)
end
```

Listed restrictions are directional. This operation permits source `"ph"` to
target `"f"`, but it does not implicitly permit the reverse rewrite:

```julia
GeneralizedOperation(2, 1, 0.5, :phoneme;
    restrictions=("ph" => "f",))
```

Use `UniversalAutomaton` for the native unit-cost universal specializations.
It supports byte, Unicode-scalar, and u64-token inputs through dispatch. An
explicit `UniversalPolicy` is a typed, non-empty set of directional zero-cost
equivalences; omit it to select the allocation-free unrestricted policy:

```julia
policy = UniversalPolicy('p' => 'f')
automaton = UniversalAutomaton(0, policy;
    variant=UNIVERSAL_TRANSPOSITION)
try
    @assert accepts(automaton, "p", "f")
    @assert !accepts(automaton, "f", "p")
finally
    close(automaton)
end
```

For incremental input, `online(automaton, source)` owns an exclusive native
prefix state. `advance!` commits exactly one `Char`, `UInt8`, or u64 integer,
as selected by the source type. `prefix_observations` supplies a finite stream
and closes it automatically on exhaustion, failure, or return from its do
block:

```julia
automaton = UniversalAutomaton(1;
    variant=UNIVERSAL_TRANSPOSITION)
try
    prefix_observations(automaton, "ab", "ba") do prefixes
        @assert last(collect(prefixes)).accepting
    end
finally
    close(automaton)
end
```

`AutomatonLimits` places explicit hard ceilings on source units, committed
target units, generalized retained cells, and generalized work per step.
Failed advancement is transactional: `observation(online)` still describes
the last committed prefix. A generalized observation's
`current_row_nonempty == false` is not permanent death because an operation
that consumes several target units can reconnect an older retained row. A
universal observation with `alive == false` is permanently dead.

The online contract is exact at every committed prefix:

- `observation(online)` initially describes the empty target prefix (length
  zero), including acceptance of an empty complete target when applicable.
- Each successful `advance!` consumes exactly one unit in the source's domain,
  increments `consumed_target_length` by one, and returns the same observation
  that a following non-mutating `observation` returns. For strings, a unit is a
  Unicode scalar, not a UTF-8 byte or grapheme cluster. A failed domain,
  scalar, resource-limit, or length check does not consume the unit.
- `prefix_observations` yields the observation *after* each target unit. It
  does not yield the initial observation; an empty target therefore yields an
  empty stream. Its final value agrees with `evaluate` for a nonempty target;
  for an empty target, compare `evaluate` with the initial `observation`.
- Generalized `scaled_distance` is present only when the entire source is
  reachable within the inclusive budget. Its exact cost is the numerator
  divided by `scale_denominator`; `nothing` is not an infinite numeric cost.
  `active_positions` counts in-budget positions in the *current* target row,
  and `current_row_nonempty` reports whether this count is positive. A later
  multi-target rule may resurrect an empty row.
- Universal `source_length` is the fixed source's unit count. `alive` means
  its canonical frontier is nonempty; once false it stays false even though
  later successful advances still increment `consumed_target_length`.

An online handle owns its bound native state independently of the parent
configuration: closing the parent does not invalidate the online handle.
Online states and prefix streams are exclusive, single-consumer values. Close
them explicitly when stopping early; the do-block form closes on every exit,
and iteration closes a stream on exhaustion or an advancement failure. No
online operation is safe to race with `close` on the same wrapper.

These standalone calls compare one source with one target. They do not walk a
dictionary or materialize and filter dictionary entries. The
[standalone automata design](../../../docs/bindings/standalone-automata.md)
defines the exact operation validation, scaling, liveness, ownership, and
complexity contracts.

### Semantic regression coverage

The [package tests](test/runtests.jl) compare universal batch acceptance and
completed online acceptance with the independent dynamic-programming distance
kernels. They cover every pair of binary-alphabet words through length four,
all three variants, and inclusive thresholds zero through three. Targeted
Unicode, byte, and u64-token cases additionally check empty input, NUL, literal
dollar signs, full-width tokens, transposition plus a second edit, and unequal-
symbol merges and splits. Online observations must report the exact number of
consumed target units. These are bounded differential tests, not a proof for
arbitrary input lengths or custom equivalence policies.
The adjacent-transposition variant is checked against optimal string alignment
(OSA), not unrestricted Damerau distance; `CA` versus `ABC` distinguishes them.

### Qualification and performance budgets

`test/automata_qualification.jl` compares every initial, intermediate, and
final observation with a separate executable built from the public Rust
automata APIs. Its fixed-seed corpus covers 1,092 generalized cases (including
randomized operation subsets and exhaustive short Unicode pairs) and 3,456
universal cases (three variants, three domains, two policy modes, budgets zero
through two, and boundary values). The Julia CI job builds that executable and
requires the comparison. Invalid configurations, transactional rollback,
ownership, and stream cleanup have additional direct tests.

`benchmark/automata.jl` samples seven groups of 200 iterations after 100
warmups and compares median nanoseconds per operation with the same public
Rust control's benchmark mode. For each scenario the CI regression ceiling is
$`T_{\mathrm{Julia}} \le 10 T_{\mathrm{Rust}} + A`$, where $`A`$ is 50,000 ns
for a single construction, match, or 32-unit traversal, and 500,000 ns for a
host-side batch of 32 complete matches. These allowances absorb FFI and CI
scheduling noise but fail substantial algorithmic or marshalling regressions.
The batch is deliberately bounded host repetition, not dictionary-product
traversal; the latter is owned by a separate Julia query-capability task.

On the 2026-09-30 local debug build (Julia 1.13.1), representative medians in
nanoseconds were:

| Scenario | Public Rust | Julia facade |
|---|---:|---:|
| Generalized construction | 2,179 | 8,921 |
| Universal construction | 10 | 415 |
| Generalized complete / early / late rejection | 156,066 / 120,237 / 154,274 | 159,951 / 121,675 / 156,790 |
| Universal complete / early / late rejection | 26,053 / 5,642 / 26,400 | 31,259 / 7,431 / 31,334 |
| Generalized / universal 32-unit traversal | 156,011 / 26,172 | 174,672 / 53,089 |
| Generalized / universal batch of 32 | 4,463,982 / 508,826 | 4,509,264 / 617,118 |

These figures are diagnostic, not portable speed guarantees; CI evaluates the
budget against a native control measured on the same runner.

## Common and intended usage

- Use `distance`, `optimal_string_alignment_distance`,
  `true_damerau_distance`, and `merge_and_split_distance` for pairwise work.
  Each accepts `AbstractString`, `AbstractVector{UInt8}`, or
  `AbstractVector{UInt64}` pairs. A thresholded call returns `nothing` when the
  exact value exceeds the inclusive threshold. `damerau_distance` remains a
  compatibility spelling for optimal string alignment.
- Reuse a `Transducer` for repeated queries with the same dictionary and
  algorithm. Use `snapshot(transducer)` when several queries must observe the
  same revision even while the live dictionary changes.
- Wrap a transducer in `QueryCache` when complete queries repeat. The native
  cache applies hard entry and logical-weight bounds, TinyLFU admission, and
  SIEVE eviction; approximation changes residency, never match correctness.
- Iterate a `QueryCursor` for independently owned `Match` values. Use
  `reduce_batches!` for high-volume processing where callback-scoped native
  batches amortize the FFI boundary.
- Compile reusable `PhoneticPattern` and `PhoneticRuleSet` values only when the
  native `BUILD_FEATURE_PHONETIC` bit is present.

## API reference

The [live development API guide](https://vinary-tree.github.io/liblevenshtein-rust/julia/dev/)
is generated from the current `master` source. It is not a published RC.6
General-registry package or a versioned release reference.

| API | Contract |
|---|---|
| `distance(a, b; threshold=nothing)` | Standard Levenshtein distance over matching string, byte-vector, or u64-token-vector domains. |
| `optimal_string_alignment_distance(a, b; threshold=nothing)` | Restricted Damerau distance with adjacent transposition. |
| `damerau_distance(a, b; threshold=nothing)` | Compatibility spelling for `optimal_string_alignment_distance`. |
| `true_damerau_distance(a, b; threshold=nothing)` | Unrestricted Damerau-Levenshtein distance. |
| `merge_and_split_distance(a, b; threshold=nothing)` | Standard edits plus one-to-two split and two-to-one merge at unit cost. |
| `GeneralizedOperation`, `GeneralizedOperationSet` | Immutable runtime edit grammar with typed applicability and directional listed restrictions. |
| `UniversalPolicy`, `UNRESTRICTED_POLICY` | Typed directional zero-cost equivalences or the allocation-free unrestricted policy. |
| `GeneralizedAutomaton(k, operations)` | Owned immutable Unicode generalized automaton with inclusive integral budget `k`. |
| `UniversalAutomaton(k, policy; variant=...)` | Owned immutable universal specialization for strings, bytes, or u64 tokens. |
| `evaluate(automaton, source, target; limits=nothing)` | Complete native evaluation returning an exact typed observation. |
| `accepts(automaton, source, target; limits=nothing)` | Boolean convenience over complete evaluation. |
| `online(automaton, source; limits=nothing)` | Exclusive source-bound native prefix state. |
| `advance!(online, unit)`, `observation(online)` | Transactional one-unit advancement and non-mutating observation. |
| `prefix_observations(automaton, source, target; limits=nothing)` | Finite closeable prefix stream; its do-block form guarantees early-return cleanup. |
| `Transducer(resource, algorithm)` | Retains a `VinaryTreeInterop.Resource` or `.Dictionary`. |
| `snapshot(transducer)` | Owned immutable transducer revision. |
| `unit_domain(transducer)` | `BYTE`, `UNICODE_SCALAR`, or `U64`. |
| `query(transducer, input, k; order=...)` | Lazy query over `String`, `Vector{UInt8}`, integer tokens, or a pattern. |
| `QueryCache(transducer; max_entries=1024, max_weight=64 * 1024 * 1024)` | Retains the transducer in an exclusive, synchronization-free bounded result cache; limits apply per result-order shard. |
| `query(cache, input, k; order=...)` | Materialize exactly on a miss or return an independent cursor over a resident immutable result. |
| `cache_stats(cache)` | Copy requests, hits, misses, admissions, rejections, evictions, entries, and logical weight. |
| `clear!(cache)`, `reset_stats!(cache)` | Drop residency or reset counters without conflating the two operations. |
| `next_batch!(cursor, maximum)` | Copy one bounded leased batch into owned matches. |
| `reduce_batches!(f, initial, cursor; batch_size=256)` | Invoke `f(accumulator, BorrowedBatch)` on lexical zero-copy batches and consume the cursor. |
| `PhoneticPattern(source; llre=false)` | Compile regex or import-free LLRE source. |
| `input in pattern`, `size(pattern)` | Membership and structural size. |
| `PhoneticRuleSet(source_or_kind)` | Parse rules or select a built-in rule set. |
| `rules(input)`, `length(rules)` | Rewrite text and count enabled rules. |
| `close`, `isopen` | Deterministic lifecycle for every native owner. |

`Match.term` is a `String`, `Vector{UInt8}`, or `Vector{UInt64}` according to
`Match.unit_domain`; `Match.id` is `nothing` or `UInt64`. Enum values and ABI
constants are generated from `bindings/api.json`.

The [domain-preserving distance design](../../../docs/bindings/distance-domains.md)
defines the shared recurrence, native naming convention, threshold sentinels,
allocation behavior, and cross-domain differential tests.

## Ownership, snapshots, and batching

Constructors retain provider resources; callers keep ownership of their own
handles. A cursor owns its query-start snapshot and remains valid after its
source transducer or dictionary closes. Ordinary iteration copies terms before
releasing each generation. A `BorrowedMatch` and every view derived from it are
valid only during the current `reduce_batches!` callback; access afterward
throws. The reducer callback is contained at the C boundary, and a Julia
exception is rethrown only after native traversal has returned and the cursor
has closed.

Close long-lived values deterministically with `close` in `finally`. Finalizers
are leak containment, not a scheduling guarantee. Query cursors and online
automata are exclusive and single-consumer. Immutable transducers and
standalone configuration handles may be used by independently synchronized
tasks; do not race `close` against another operation on the same wrapper.

`QueryCache` is also exclusive and intentionally contains no lock. For
parallel workloads, allocate one cache per task or worker; each shard retains
its own bounded hot set without imposing coordination on every hit. A cached
miss captures one immutable revision before materialization. Providers must
publish `vt.snapshot.id.1`; a missing identity fails with `STATUS_UNSUPPORTED`
instead of risking stale matches. Traversal and distance-then-term results have
independent policy shards because their sequences are observably different.

## Errors, compatibility, and security

Fallible native statuses become `NativeError` with the exact numeric status,
operation, and a copied thread-local diagnostic. ABI generation 1 is checked at
module initialization; newer additive API revisions are accepted. Unit domains
remain distinct and invalid UTF-8, domain mismatches, stale leases, closed
providers, and malformed provider output fail explicitly.

Treat custom dictionary providers as untrusted code. Native negotiation
validates their versioned vtables and converts provider failures into statuses.
Set application-specific traversal and result limits, avoid retaining lexical
batch pointers, and report vulnerabilities through the repository security
policy.

## Performance and release

Transducer construction is constant-time resource negotiation. Query work is
lazy over a captured revision. Iteration allocates owned host terms by design;
`reduce_batches!` keeps descriptors and term storage native for the callback and
uses the ABI default of 256 matches to amortize calls. Benchmark native,
pairwise-FFI, iterator, and reducer paths separately on an idle host before
making performance claims.

`QueryCache` uses the production policy described by Einziger, Friedman, and
Manes, [TinyLFU](https://doi.org/10.1145/3149371), for approximate-frequency
admission and Zhang et al., [SIEVE](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo),
for low-overhead victim selection. A cold miss necessarily materializes the
complete exact result, so use ordinary lazy `query(transducer, ...)` for
one-shot or early-terminating workloads. A hit clones shared immutable native
storage and exposes it through the same `QueryCursor` contract.

The Julia package name is `Liblevenshtein`, without an organization prefix.
Release publication is intentionally disabled for this RC6 candidate; a
signed source tag and registry review remain required before General-registry
registration and versioned Documenter deployment. The development guide above
can be read independently of that future package release.
