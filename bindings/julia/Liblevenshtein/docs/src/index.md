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
