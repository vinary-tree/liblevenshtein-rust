# Liblevenshtein.jl

`Liblevenshtein` provides Unicode edit-distance functions and lazy,
snapshot-consistent fuzzy search over any negotiated Vinary Tree dictionary
provider. Its Julia API preserves byte, Unicode-scalar, and unsigned-token
domains through multiple dispatch.

## Common usage

```julia
using Liblevenshtein

distance("kitten", "sitting")
distance("kitten", "sitting"; threshold=2)
optimal_string_alignment_distance("ab", "ba")
true_damerau_distance("CA", "ABC")
merge_and_split_distance("m", "rn")
distance(UInt8[0xff, 0x00], UInt8[0xff, 0x01])
distance(UInt64[10, 20], UInt64[20, 10])
```

All four distance families use the same multiple-dispatch contract. Strings
count Unicode scalar values, `AbstractVector{UInt8}` values are arbitrary
binary data, and `AbstractVector{UInt64}` values are application tokens.
Supplying `threshold=k` returns the exact result when it is at most `k` and
`nothing` otherwise. Dense vectors cross the native boundary without a copy;
non-dense abstract vectors are materialized temporarily to satisfy C's
contiguous-buffer contract.

The repository's
[domain-preserving distance design](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/docs/bindings/distance-domains.md)
explains the shared recurrences, ABI names, threshold sentinels, and generated
differential tests.

## Native source filtering

`jaro_similarity` uses the native Unicode Jaro scorer. A positive
`prefix_scale` up to 0.25 selects scaled Jaro-Winkler;
`jaro_winkler_similarity` uses the conventional 0.1 scale. Each call checks
`max_input_bytes` for both strings and `max_comparisons` for the worst-case
scalar pair count before scoring.

`query_jaro` intersects a fuzzy traversal with a minimum Jaro-Winkler score.
It evaluates each final borrowed source prefix as the native walk reaches it,
keeps the query-start dictionary snapshot, and streams `SpecializedMatch`
values. This filter applies to Unicode keys. Close or cancel an unfinished
cursor to release its native traversal promptly.

`ngram_candidate` and `hybrid_candidate` call native single-source n-gram and
hybrid filters. `query_ngram` and `query_hybrid` apply the same predicates at
each final prefix of a fuzzy traversal. N-gram size zero follows the native
index rule and means unigrams. The hybrid mode applies its native adaptive
Jaro-Winkler threshold after n-gram admission. The per-input byte cap defaults
to 4096, and hybrid Jaro work also has a comparison cap. Each source is
indexed only for its call.

`NativeSourceFilterIndex` copies a Unicode source into a frozen native n-gram
postings index. Hybrid mode adds the native adaptive Jaro-Winkler stage.
`query_ngram` and `query_hybrid` on this index compute a complete bounded
candidate-ID snapshot, then lazily yield terms in source order. The iterator
retains the copied IDs and terms after the index handle closes. A source
cardinality, result, or hybrid comparison ceiling raises
`SourceFilterIncomplete` before any candidate is exposed.

```julia
cursor = query_jaro(transducer, "martha", 2;
    minimum_similarity=0.85, max_comparisons=100_000)
try
    for match in cursor
        println(match.term, ": ", match.cost)
    end
finally
    close(cursor)
end
```

## Bounded scalar time series

The six scalar temporal functions use the same native kernels as Rust. The
Julia wrappers take finite real vectors and an explicit `TemporalLimits`
budget, return `TemporalDistanceOutcome`, and borrow input memory only during
the call. MSM uses `split_merge_cost`; ERP uses `gap`; unit-grid TWED uses
`stiffness` and `gap_penalty`; DTW requires a Sakoe–Chiba `band`; discrete
Fréchet has no parameters; Soft-DTW uses positive finite `gamma` and returns
a loss rather than a metric distance.

```julia
limits = TemporalLimits(max_series_len=2048, max_dp_cells=250_000,
    max_work_units=250_000, max_scratch_bytes=1 << 20)
outcome = msm_distance([1.0, 2.0], [1.0, 2.5];
    split_merge_cost=1.0, cutoff=0.5, limits)
@assert outcome.kind == :finite && outcome.value == 0.5
```

`:above_cutoff` means an alignment exists but its exact score exceeds the
inclusive cutoff. `:no_alignment` means the selected kernel has no finite
path for the operand shapes or DTW band. `:incomplete` carries a checked
resource or overflow reason and never carries a partial score. The native
revision-11 symbol is checked before resolution so older libraries continue
to serve the existing Julia surface. Invalid configuration or nonfinite
samples raise a copied `NativeError`.

### Soft-DTW gradients

`soft_dtw_gradient` evaluates the full differentiable Soft-DTW loss and
derivatives with respect to each input sample. Both operands must be nonempty
and finite, and `gamma` must be positive and finite. A finite
`SoftDtwGradientOutcome` owns `left_gradient` and `right_gradient` vectors;
each vector has the length of its corresponding input. An incomplete result
has no value or gradients and records its exact resource or numeric reason.

```julia
analysis = soft_dtw_gradient([0.0, 1.0], [0.5, 1.5]; gamma=0.75,
    limits=TemporalLimits(max_series_len=2, max_dp_cells=4,
        max_work_units=8, max_scratch_bytes=512))
@assert analysis.kind === :finite
@assert length(analysis.left_gradient) == 2
```

The reverse sweep needs complete forward and adjoint matrices, so scratch
storage grows with both operand lengths. `soft_dtw_loss` evaluates only the
loss with two retained rows when gradients are unnecessary. Soft-DTW values
and gradients are loss quantities; they do not provide metric lower bounds.

### Online temporal prefixes

`TemporalOnlineAutomaton` copies a fixed query and advances over finite
target samples for MSM, ERP, unit-grid TWED, banded DTW, and discrete
Fréchet. `observation` reports the empty target before the first advance.
`advance!` returns an exact prefix observation after a committed sample; a
resource-incomplete step leaves the prior observation unchanged. Online DTW
cutoffs and scores are squared, while scalar `dtw_distance` returns root
distance. Soft-DTW has no online automaton.

```julia
machine = TemporalOnlineAutomaton(:erp, [1.0, 2.0]; cutoff=5.0)
try
    @assert observation(machine).consumed_target_len == 0
    @assert advance!(machine, 1.0).observation.consumed_target_len == 1
finally
    close(machine)
end
```

`TemporalOnlineLimits` bounds query length, live frontier positions, work per
target sample, and scratch storage. `online_observations` presents a one-shot
lazy stream; `reduce_observations!`, normal exhaustion, cancellation, and
errors close its native machine.

### Canonical metric temporal domains

ERP identifies any sequence obtained by inserting or deleting its fixed gap
value. Discrete Fréchet identifies paths that differ only by consecutive
repetition. `MetricErpConfig` and `representative` remove gap samples;
`FrechetStutterClass` collapses repetition and requires a nonempty path.
`canonical_samples` returns a copy, preserving the representative after the
caller changes its input or returned vector.

```julia
config = MetricErpConfig(0.0)
left = representative(config, [1.0, 0.0, 2.0])
right = representative(config, [1.0, 2.0])
@assert metric_erp_distance(config, left, right).value == 0.0
```

`MetricErpIndex` and `MetricFrechetIndex` canonicalize before native
quantization, then use the frozen temporal range cursor. After `freeze!`, a
`query_metric_range` cursor retains its snapshot even if the index closes.
The usual page and cumulative limits apply. ERP representatives must use the
same gap bit pattern, including the sign of zero.

For the other two metric parameter families, `MetricMsmConfig` validates a
strictly positive split/merge cost and requires nonempty series;
`MetricTwedConfig` validates positive stiffness and nonnegative gap penalty
for unit-grid TWED. `metric_msm_distance`, `metric_twed_distance`,
`MetricMsmIndex`, and `MetricTwedIndex` retain the native bounded score and
lazy range cursor behavior under those validated configurations.

### Physical-time TWED

`TimestampedSeries` copies each nonempty finite value series and its strictly
increasing finite physical timestamps. Its unit is one of `:seconds`,
`:milliseconds`, `:microseconds`, or `:nanoseconds`; its finite origin is no
later than the first timestamp. Returned value and timestamp arrays are
copies. `MetricTimestampedTwedConfig` validates strictly positive stiffness
and nonnegative gap penalty. `metric_timestamped_twed_distance` requires
matching units and origins and returns a bounded native exact score,
above-cutoff result, or explicit incomplete reason. This physical-time
recurrence is distinct from the unit-grid TWED kernel.

`TimestampedTwedIndex` copies full-precision episodes into a typed value/time
quantized dictionary. Freeze it before search. `insert_episode!` returns a
stable insertion ID, so repeated caller IDs remain distinguishable.
`query_metric_range` yields exact native matches lazily from one captured
revision; its cursor survives closing the index and reports cumulative
resource exhaustion through `TemporalQueryIncomplete`. The index uses
quantization for pruning only and verifies every surviving collision against
the retained physical timestamps.
`query_metric_knn` computes an exact nearest-neighbor vector over the same
frozen revision. Its bounded full scan must complete before it returns; any
resource exhaustion raises `TemporalQueryIncomplete` with no partial list.
Results use distance then stable episode ID order, including when metadata
IDs repeat.

`ApproxMsmIndex` uses piecewise aggregate approximation (PAA) to select a
candidate pool, then computes exact MSM for those candidates. Its strict
`query_approx_msm_knn` result reports the number indexed, selected, and
exactly reranked. Only `proves_recall(result)` means all indexed entries were
decided. Advisory and incomplete results retain exact distances for emitted
neighbors without claiming recall. A zero-neighbor advisory result is not
evidence that no neighbor exists.

### Replayable alignment witnesses

`temporal_alignment` extracts bounded native paths for MSM, ERP, unit-grid
TWED, banded DTW, and discrete Fréchet; `timestamped_twed_alignment` covers
physical-time TWED. A finite result owns a closeable witness whose immutable
operations are copied to Julia in bounded pages. `replay_alignment` validates
the path and recomputes its score from caller-supplied operands and the
configuration captured at extraction. Julia endpoints are one-based. A
resource-incomplete result carries no witness and identifies DP, work,
scratch, witness-byte, or overflow exhaustion. See the
[usage guide](../../../README.md#replayable-temporal-alignment-witnesses).

### Lazy temporal range queries

`TemporalSeriesSource` copies a finite iterable of `(UInt64 ID, finite real
vector)` pairs. The constructor enforces entry count, per-series length, and
total sample-storage limits. A `query_temporal_range` cursor borrows that
snapshot and runs the chosen native scalar kernel only when the next result
is requested. Results follow source order and carry exact native scores.

```julia
source = TemporalSeriesSource([10 => [1.0, 2.5], 20 => [2.0, 3.0]])
budget = TemporalQueryLimits(max_candidates=100,
    max_results=20, max_total_dp_cells=10_000)
cursor = query_temporal_range(source, :dtw, [1.0, 2.0];
    band=1, cutoff=1.0, query_limits=budget)
try
    reduce_batches!((ids, batch) ->
        append!(ids, (match.id for match in batch)), UInt64[], cursor)
finally
    close(cursor)
end
```

Per-comparison `TemporalLimits` and cumulative `TemporalQueryLimits` are both
enforced. A limit or native incomplete outcome raises
`TemporalQueryIncomplete` and closes the cursor. This is an exact bounded
scan over a finite source; native trie indexes and online automata have
different pruning and continuation behavior.

For IPA feature scores, syllable heuristics, compiled rewrite rules, and
dictionary-free word search, see [Phonetic matching and analysis](phonetic.md).

## Resource-backed search

A `Transducer` accepts a `VinaryTreeInterop.Resource` or
`VinaryTreeInterop.Dictionary`. Reuse it across queries and close it
deterministically:

```julia
transducer = Transducer(dictionary, ALGORITHM_STANDARD)
try
    for match in query(transducer, "speling", 2)
        println(match.term, " ", match.distance)
    end
finally
    close(transducer)
end
```

Use `reduce_batches!` when matches do not need to escape the callback. Each
`BorrowedMatch` expires when its callback returns; call `materialize` inside
the callback when an independently owned value is required.

### Affine and weighted edit costs

The native cost configuration selects the recurrence and its number domain.
`query_cost(transducer, input, maximum, costs)` dispatches on a typed cost
configuration:

| Configuration | Native carrier | Edit kernel |
|---|---|---|
| `UNIT_EDIT_COSTS` | bounded integer addition | transducer's unit-cost algorithm |
| `AffineGapCosts(...)` | exact scaled integer addition | affine gap |
| `WeightedOperationCosts(...)` | floating addition | standard, adjacent transposition, or merge/split |
| `ContextualCosts(...)` | floating callback costs | Unicode contextual traversal |

The native `BottleneckCost` monoid belongs to time-series Fréchet kernels and
does not define a Levenshtein edit recurrence. Julia's cost dispatch selects
the concrete native automata above rather than emulating a Rust monoid trait.

`AffineGapCosts` uses exact scaled integers for a three-layer gap automaton.
For a nonempty gap of `k` units, its cost is $`g(k)=g_{\mathrm{open}}+k g_{\mathrm{extend}}`$.
The constructor derives the least exact decimal denominator unless
`scale_denominator` supplies a positive exact denominator. Both the configured
weights and the query budget must be representable at that scale. Native
results carry `scaled_cost` and `scale_denominator`; their ratio is the exact
cost. `cost` is the corresponding `Float64` presentation value.

`WeightedOperationCosts` uses native floating point operation weights. Its
`:standard`, `:typo`, and `:ocr` presets come from the native library. The
six-argument constructor configures substitution, insertion, deletion,
transposition, split, and merge costs; `match_cost` must remain zero. All costs
and the inclusive budget must be finite and nonnegative. The transducer's
standard, adjacent-transposition, or merge-and-split algorithm selects the
available edits. The unrestricted Damerau-Levenshtein algorithm has no native
float-weighted kernel and returns a configuration error.

```julia
affine = AffineGapCosts(0.5, 0.25, 1.0)
cursor = query_affine(transducer, "cat", 1.0, affine)
try
    for match in cursor
        exact_cost = match.scaled_cost // match.scale_denominator
        println(match.term, " ", exact_cost)
    end
finally
    close(cursor)
end

weighted = WeightedOperationCosts(2.0, 1.0, 1.0, 0.5, 1.5, 1.5)
cursor = query_weighted(transducer, "cat", 1.0, weighted)
try
    for match in cursor
        println(match.term, " ", match.cost)
    end
finally
    close(cursor)
end
```

Both functions also accept `AbstractVector{UInt8}` for arbitrary bytes and
`AbstractVector{UInt64}` for token identifiers. Results keep those unit types
without text conversion. Cost results contain terms and costs, not provider
value IDs, matching the native affine and weighted iterators. `CostCursor`
captures one dictionary revision and
can outlive its source transducer. `next_batch!` returns copied results;
`reduce_batches!` supplies borrowed `BorrowedCostMatch` values that expire
when the callback returns. Use `materialize` inside that callback to retain a
result. The generalized operation-set automaton below remains the choice when
the edit grammar itself must be configured as runtime data.

The `benchmark/cost_boundary.jl` script checks six
Unicode, byte, and token scenarios. It compares a direct native cursor count
with Julia's owned result materialization and enforces the package's
established tenfold native-work budget plus a 50 microsecond dispatch allowance.

### Ranked Unicode queries

`query_ranked(transducer, text, distance)` returns the native distance-then-term
cursor. `query_mode` retains that order while selecting an inclusive distance
interval from completed matches. The maximum bound is passed to the native
automaton; the minimum is applied as each match arrives.

```julia
cursor = query_mode(transducer, "speling";
    minimum_distance=1, maximum_distance=2)
try
    for match in cursor
        println(match.term, " ", match.distance)
    end
finally
    close(cursor)
end
```

`query_suggestions(transducer, text, distance, scorer)` scores one native
distance layer at a time. The scorer receives `(term, distance, id)` and returns
a number. Results sort by increasing distance, decreasing finite score, then
ascending term; non-finite scores rank last. Both adapters provide
`next_batch!`, `reduce_batches!`, `close`, and `cancel!`. Adapter reducers
receive owned vectors, while the base `QueryCursor` reducer exposes borrowed
native batches. Closing a transducer does not invalidate a cursor created from
it, because each cursor retains its query-start dictionary snapshot.

### Contextual costs and prefix pruning

`query_contextual` runs the native contextual dynamic-programming traversal.
Provide three Julia cost functions and a strictly positive lower bound for
every nonzero allowed edit. A cost function may return `nothing` to forbid an
operation. The query is fixed during traversal, so callback context contains
the full query, its one-based edit position, the already visited dictionary
prefix, and the current edge scalar. Right context of the dictionary edge is
not available during trie descent.

```julia
costs = ContextualCosts(
    (context, query_unit, dictionary_unit) ->
        query_unit == dictionary_unit ? 0.0 : 1.0,
    (context, dictionary_unit) -> 1.0,
    (context, query_unit) -> 1.0;
    minimum_nonzero_cost=1.0,
)
cursor = query_contextual(transducer, "speling", 2.0, costs)
try
    for match in cursor
        println(match.term, " ", match.cost)
    end
finally
    close(cursor)
end
```

The declared lower bound is checked for each positive finite callback cost.
An undersized bound may traverse extra prefixes; a cost below the declared
bound raises an error instead of silently losing a match. Negative and
non-finite costs are rejected as forbidden edits by the native iterator.
The `ContextualEditContext` scalar views expire when the callback returns;
copy data inside the callback if it must be retained.

`query_pruned` runs a native depth-first fuzzy traversal with a balanced
`PrefixVisitor`. `enter(unit, depth)` may reject a subtree, and its matching
`leave(unit, depth)` is called even when it rejects. `permits(prefix)` decides
whether an accepted final prefix is returned; `score(prefix)` may attach a
numeric score. `matches(candidate, query)` customizes the structural unit
comparison. Depth is one-based and result order follows dictionary DFS.

`query_filtered(transducer, text, distance, predicate)` calls the Julia
predicate with each final node's optional `UInt64` ID before the native
traversal constructs its term. It returns the same `Match` type as `query`, in
traversal order. This is useful when a scope or tenant ID rejects most fuzzy
matches, because rejected term strings are never built.
`query_by_value(transducer, text, distance, id)` applies one ID, while
`query_by_value_set(transducer, text, distance, ids)` copies the allowed IDs
when the query starts. Both use the same native filtering path and cursor
lifecycle as `query_filtered`.

```julia
visitor = PrefixVisitor(
    (unit, depth) -> depth > 1 || unit == 's',
    (unit, depth) -> nothing;
    permits=prefix -> last(prefix) == 'g',
)
cursor = query_pruned(transducer, "speling", 2, visitor)
try
    reduce_batches!((count, batch) -> count + length(batch), 0, cursor)
finally
    close(cursor)
end
```

Both specialized cursors capture one dictionary revision, lend bounded native
batch leases to `reduce_batches!`, and release them before callback return.
`materialize` copies a borrowed specialized match for use afterward. Callback
functions run on the thread advancing or closing the cursor. Keep their state
valid until `close` or `cancel!`; a prefix visitor may receive final `leave`
calls during close.

## Runtime edit grammars and standalone automata

`GeneralizedAutomaton` executes an immutable runtime operation set. The native
engine converts accepted decimal weights to one exact scale, and Julia exposes
an accepted cost as `Rational{Int}`:

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
    @assert evaluate(automaton, "cat", "cut").distance == 1 // 2
finally
    close(automaton)
end
```

`UniversalAutomaton` selects the standard, adjacent-transposition, or
merge-and-split specialization. Input types select byte, Unicode-scalar, or
u64-token semantics. A `UniversalPolicy` owns typed directional zero-cost
equivalences; the default `UNRESTRICTED_POLICY` avoids policy allocation and
lookup.

```julia
policy = UniversalPolicy('p' => 'f')
automaton = UniversalAutomaton(0, policy)
try
    @assert accepts(automaton, "p", "f")
    @assert !accepts(automaton, "f", "p")
finally
    close(automaton)
end
```

Call `online` plus `advance!` for a long-lived prefix state, or use the
closeable `prefix_observations` iterator for a finite target. Its do-block form
closes on early return as well as ordinary exhaustion. `AutomatonLimits`
supplies explicit native source, target, retained-cell, and per-step work
ceilings; a failed advance leaves the prior observation committed.

The initial `observation` describes the empty target. Each successful
`advance!` commits one domain unit and yields the same value as a subsequent
`observation`; `prefix_observations` yields only post-advance values, so an
empty target yields no iterator items. A complete `evaluate` agrees with the
last prefix value, or with the initial value for an empty target. Generalized
`scaled_distance` is a within-budget numerator over `scale_denominator`, not
an approximate float; absence means the full source is not within budget at
that prefix. Closing a parent configuration does not invalidate its bound
online state. Online states are exclusive and must be explicitly closed when
stopping early.

Generalized current-row emptiness is not a pruning certificate: a multi-target
operation can revive from an older retained row. Universal `alive == false`,
by contrast, is permanent. Standalone automata compare one source/target pair;
dictionary-product traversal remains a distinct bounded native capability.
The [Julia package guide](https://github.com/vinary-tree/liblevenshtein-rust/blob/master/bindings/julia/Liblevenshtein/README.md)
records the independent public-Rust differential corpus, measured scenarios,
and CI regression budgets. Dictionary-product traversal is owned by a
separate Julia query-capability task, not by this standalone qualification.

## Bounded repeated-query caching

`QueryCache` is an opt-in complete-result memo for repeated workloads. It uses
the native TinyLFU-admission/SIEVE-eviction implementation rather than a Julia
dictionary, preserves hard per-order entry and logical-weight bounds, and
invalidates residency when the provider revision changes:

```julia
cache = QueryCache(transducer; max_entries=512, max_weight=32 * 1024 * 1024)
try
    matches = collect(query(cache, "speling", 2))
    @show cache_stats(cache)
finally
    close(cache)
end
```

The cache is mutable, exclusive, and lock-free by ownership convention. Shard
one cache per task or worker for parallel use. A miss returns the exact result
even when policy rejects it; approximation affects only which reusable entries
remain resident. Providers without stable snapshot identity are rejected
because correctness takes precedence over hit rate.

## Automata

`Transducer` accepts `ALGORITHM_STANDARD`, `ALGORITHM_TRANSPOSITION`,
`ALGORITHM_MERGE_AND_SPLIT`, or `ALGORITHM_DAMERAU_LEVENSHTEIN`. Standard,
merge-and-split, and unrestricted Damerau-Levenshtein are metrics. The
optimal-string-alignment transposition variant is intentionally non-metric and
does not compose repeated edits through the same substring.

## API

```@autodocs
Modules = [Liblevenshtein]
Private = false
```
