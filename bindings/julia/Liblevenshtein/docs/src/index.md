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

Generalized current-row emptiness is not a pruning certificate: a multi-target
operation can revive from an older retained row. Universal `alive == false`,
by contrast, is permanent. Standalone automata compare one source/target pair;
dictionary-product traversal remains a distinct bounded native capability.

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
