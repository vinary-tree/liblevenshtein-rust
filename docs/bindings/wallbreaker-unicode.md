# Finite Unicode WallBreaker binding (API revision 6)

The revision-6 `llev_wallbreaker_*` surface exposes native `PatternSplitter`
and qualified `WallBreakerQuery` result semantics to Julia through an owned,
immutable Unicode SCDAWG. It does **not** accept `vt.dictionary.v1`: that
resource has neither exact-substring lookup nor bidirectional node callbacks.
The standalone `BidirectionalExtension` helper is experimental and is not used
to produce results. No byte-key, u64-key, arbitrary foreign-provider, or
independent extension facade is claimed. FZF scoring belongs to duallity, not
this package.

```julia
matcher = WallBreakerMatcher(["café", "cafe", "αβγ"]; max_distance=1)
try
    collect(query(matcher, "café")) # exact term and one-edit neighbors
    pattern_pieces("éa", 1)        # byte and scalar offsets, zero-based
finally
    close!(matcher)
end
```

`WallBreakerMatcher` copies all borrowed UTF-8 input terms during construction
and owns one immutable SCDAWG. Duplicate terms have set semantics. A query
captures that immutable revision, verifies the **complete original dictionary
term** with the selected edit algorithm, deduplicates by term, and materializes
all accepted results before publishing a cursor. Short, empty, and unrestricted
Damerau queries borrow complete terms one at a time without cloning every
substring candidate. Queries with a provably surviving nonempty piece still
materialize one piece's substring candidates at a time. Cursor iteration only
pages already verified results; it is not a lazy search. Result order is the
SCDAWG's deterministic first-seen candidate order, not lexical or distance
order. A completed cursor outlives its matcher. `cancel!` stops future
advances; it does not invalidate an already published C batch lease. Freeing
a cursor invalidates its lease. Julia's `next_batch!` copies each result before
releasing the lease.

All seven `LlevWallBreakerLimits` fields must be positive. Hard implementation
ceilings are 4096 input descriptors, 1 MiB aggregate input UTF-8 bytes, 256
Unicode scalars per term and per query, 16 MiB conservative candidate-clone
bound, 4096 results, and 1 MiB aggregate result UTF-8 bytes. The maximum edit
distance is 8. A configured ceiling of zero or above its hard maximum is
`INVALID_ARGUMENT`. Input or query data exceeding a valid configured ceiling
returns `LIMIT_EXCEEDED`, with no matcher/cursor or partial result published.
For selective queries, the candidate preflight sums the following estimate
for each input term with Unicode-scalar length $`n`$ and UTF-8 byte length
$`b`$:

```math
(n+1)\left(b+\operatorname{sizeof}(\mathrm{SubstringMatch})+\operatorname{sizeof}((\mathrm{String},\mathrm{usize}))\right).
```

The $`n+1`$ factor bounds possible substring start positions (including an
empty pattern); native SCDAWG exact-substring results clone one full term per
occurrence. Construction validates the configured ceiling, but enforces the
computed estimate only when a query actually takes the selective path. A
complete-term streaming query is not rejected by this irrelevant estimate.
The estimate is deliberately conservative for selective queries: a feasible
query may be rejected. It bounds logical candidate storage, **not** process
RSS or the internal SCDAWG graph. The fixed corpus and scalar maxima also
bound all native iteration and distance work, but no strict work-unit budget
is claimed.

The C splitter has a two-phase contract: with insufficient capacity, it writes
the required descriptor count and returns `LIMIT_EXCEEDED` without writing
pieces. With enough capacity it writes `LlevPatternPiece` values with
zero-based Unicode-scalar and UTF-8-byte coordinates. `PatternSplitter` is a
projection only; for short patterns and unrestricted Damerau matching, the
qualified result path may use a complete empty-substring scan instead of
relying on an unproven surviving nonempty piece.

Failure classification: malformed enum/limits return `INVALID_ARGUMENT`,
required null pointers return `NULL_POINTER`, invalid input bytes return `INVALID_UTF8`, exceeded
limits return `LIMIT_EXCEEDED`, a second advance during a lease returns
`BATCH_IN_USE`, and advance after cancellation returns `CLOSED`. A batch too
small for its next single result returns `LIMIT_EXCEEDED` without advancing.
`END` publishes an empty batch. No query silently truncates results. Handles
are exclusive for mutation/advance and may not be concurrently freed while a
call is active.

Qualification: native differential tests compare complete Unicode result sets
against independent exact-distance oracles over empty, short, multibyte,
seeded random, and exhaustive corpora. Revision-6 C tests exercise budget,
two-phase split, source lifetime, lease, cancellation, and nonpartial failure;
Julia tests exercise the public facade and layout.
