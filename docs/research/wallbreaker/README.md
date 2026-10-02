# WallBreaker: exact-seed search with full-term verification

WallBreaker searches a dictionary for complete terms within an edit-distance
bound. It can avoid examining every term when a query has a nonempty substring
that must survive the allowed edits. The production query path is a
**seed-and-verify** implementation inspired by [Gerdjikov, Mihov, Mitankin,
and Schulz's WallBreaker paper](https://doi.org/10.1145/2457317.2457385).
It does not claim to implement the paper's entire bidirectional extension
algorithm: the separate `BidirectionalExtension` helper is not used to emit
`WallBreakerQuery` results.

Here, a *term* is one complete dictionary member, a *query* is the text to
match, and the *bound* is the greatest accepted edit distance. A *piece* is a
contiguous nonempty part of the query. A *snapshot* is the dictionary root
captured when an iterator is created. SCDAWG means Symmetric Compact Directed
Acyclic Word Graph; it supplies exact-substring lookup and complete-term
enumeration.

## Rust use

```rust
use libdictenstein::scdawg::ScdawgChar;
use liblevenshtein::transducer::Algorithm;
use liblevenshtein::wallbreaker::WallBreaker;

let dictionary = ScdawgChar::<()>::from_terms(["café", "cafe", "αβγ"]);
let search = WallBreaker::with_algorithm(&dictionary, 1, Algorithm::Standard);
let results: Vec<_> = search.query("café").collect();
assert!(results.iter().any(|result| result.term == "café" && result.distance == 0));
assert!(results.iter().any(|result| result.term == "cafe" && result.distance == 1));
```

`WallBreaker` and its iterator accept any backend implementing
`SubstringDictionary`, not only byte and Unicode in-memory SCDAWGs. The
current implementations are byte and Unicode in-memory SCDAWGs, byte and
Unicode persistent SCDAWGs, and byte and Unicode persistent suffix trees.
The persistent variants require the `persistent-artrie` feature. All six
share the same query and complete-term cursor contracts; their storage and
snapshot implementations differ.

## Why there are two candidate paths

The splitter uses $`k+1`$ pieces for Standard distance and $`2k+1`$ for
optimal-string-alignment Transposition and MergeAndSplit, where $`k`$ is the
bound. Under the corresponding pigeonhole proof, a matching term contains at
least one exact piece. The implementation retrieves substring occurrences for
each piece, verifies the **whole term** with the selected distance, and emits
each accepted term once in first-seen order. A piece's occurrences are
materialized before that piece is processed; accepted results are yielded one
at a time. See the [formal pigeonhole development](../../verification/wallbreaker/)
and [candidate-flow diagram](../../diagrams/automata/wallbreaker-scdawg-walk.svg).

When there are more required pieces than query characters, no nonempty piece
is guaranteed to survive. Unrestricted Damerau–Levenshtein also has no
qualified surviving-piece theorem in this implementation. A direct caller may
also supply a splitter built for a smaller bound than the query accepts.
Those cases take the complete-term path: one borrowed term from the captured
snapshot at a time is length-pruned and distance-verified, then copied into
result and deduplication storage only if accepted. Individual distance
implementations may still allocate while verifying a rejected term; in
particular, MergeAndSplit constructs a memoization key. Neither path uses an
unproven seed filter.

The executable algorithm, expressed without storage-specific traversal, is:

```text
capture one immutable dictionary root
if a nonempty surviving piece is not proved:
    for each complete term borrowed from that root:
        reject impossible Unicode-scalar lengths
        verify exact selected distance against the whole term
        copy and emit accepted terms
else:
    split the query under the proved piece-count rule
    for each piece, in splitter order:
        obtain exact-substring occurrences from that same root
        verify the whole term and emit it only on first acceptance
```

The complete-term cursor walks an insertion-order term inventory. Persistent
suffix trees skip deleted text IDs; a retained root therefore presents the
pre-mutation revision. The cursor stores a position and borrows one `&str` at
a time, rather than building an empty-substring result vector. This is a
shared trait seam, so language facades use the same native algorithm instead
of reimplementing it.

### Candidate-design decision

We considered generic node-edge depth-first search, backend-native term-ID
streams, and reuse of fuzzy-transducer iterators. A node-edge walk would need
to reconstruct complete terms and to reconcile each backend's deletion and
revision semantics. Fuzzy transducers differ in supported distance families
and ordering, so they cannot replace this query's verification contract.
The existing backend term inventories already encode unique complete members
and stable roots. A borrowed cursor over those inventories therefore gives
the smallest common interface while retaining backend-specific traversal
internals. The selective substring path remains separate because it can
avoid scanning the entire dictionary when its seed proof applies.

## Native and foreign-language limits

The pure Rust iterator is lazy over accepted results. The revision-6 C ABI,
and its Julia wrapper, intentionally collect all accepted results before
publishing a cursor so result-count and byte limits never publish partial
output. The C matcher owns an immutable Unicode SCDAWG; it does not accept an
arbitrary foreign dictionary provider. Its seven configured limits must be
positive and within the hard maxima documented in
[`liblevenshtein.h`](../../../include/liblevenshtein.h).

The candidate-clone estimate is computed from the input corpus, but checked
at **query time only for the selective path**, which materializes substring
occurrences. A complete-term streaming query does not allocate those
candidates and therefore is not rejected by that estimate. Input-term,
query-length, result-count, and result-byte limits continue to apply to both
paths. The estimate bounds a logical candidate allocation, not process RSS.
Cancellation stops future advances of an already published C result cursor;
it cannot interrupt the eager query computation. See the
[Unicode binding contract](../../bindings/wallbreaker-unicode.md) for exact
error, lease, and lifetime semantics.

## Measured short-query effect

The prospectively registered pgmcp experiment `#388` compared source commit
`0367f6e2` with the initial streaming implementation `7d86c2d6`, both against
libdictenstein `46ba112a`. It used 32 matched seeds, 16 ABBA/BAAB blocks,
192 isolated process measurements, exact result-set checks, and whole-block
bootstrap intervals. The recorded decision is `#408`; raw and analysis
artifacts are `#962` and `#963` in pgmcp.

| Workload | Treatment/control mean latency | 90% interval | Gross allocated bytes/query |
| --- | ---: | ---: | ---: |
| Short fallback | 0.0899 | 0.0816–0.1007 | 314 vs 54,833.5 |
| Selective Unicode | 0.9766 | Upper bound 1.0392 | See raw artifact |

Thus the measured short fallback was about 11.1 times faster and allocated
about 175 times fewer gross bytes on that declared fixture. The selective
path passed the predeclared 5% no-regression gate. These are measurements of
the named commits and workload, not a claim about all corpora or later source
revisions. The benchmark driver and analysis are
[`bench-wallbreaker-short-streaming.sh`](../../../scripts/bench-wallbreaker-short-streaming.sh)
and [`analyze-wallbreaker-short-streaming.R`](../../../scripts/analyze-wallbreaker-short-streaming.R).

### Integrated-source confirmation and timing-window sensitivity

The subsequent integrated revision `f6c9561e` adds the conservative-splitter
fallback and the FFI query-time limit correction. Its first matched-seed
experiment, pgmcp `#389`, retained the same 32 seeds and 192 processes but
timed only 500 operations per process. Short-query latency and allocation
again improved, **but the selective no-regression gate failed**: the selective
mean ratio was 1.0395 and its 90% whole-block interval extended to 1.0904,
above the predeclared 1.05 ceiling. The complete failed record remains in
pgmcp artifacts `#968` and `#969`; no outlying seed was removed. This result
is evidence of timing-window sensitivity, not proof that scheduler noise
alone caused the failure.

Before further measurement, pgmcp experiment `#391` froze a longer paired
protocol: 50,000 timed operations per process, the same 32 seeds and 16
ABBA/BAAB blocks, CPU-2 pinning, exact ordered-result comparison in all 192
processes, and no sample exclusions. The control was the unchanged
`0367f6e2` binary; the treatment was the clean `f6c9561e` binary. The raw
records, binary/source hashes, protocol metadata, and analysis are pgmcp
artifacts `#971` and `#972`, with accepted primary decision `#409`.

| Workload | Treatment/control mean latency | 90% whole-block interval | Gross allocated bytes/query |
| --- | ---: | ---: | ---: |
| Short fallback | 0.0890 | 0.0875–0.0907 | 314 vs 54,833.5 |
| Selective Unicode | 1.0144 | 1.0043–1.0238 | 825.5 vs 825.5 |
| Empty-query fallback | 0.0974 | 0.0951–0.0993 | 248 vs 54,719.5 |

All frozen `#391` gates passed: the selective interval's upper bound is below
1.05; short and empty means are below 0.15 with upper bounds below 0.18;
short gross allocation is under 1% of control. The longer timing window
supports no material selective regression on this fixture while leaving the
failed short-window result visible. Neither experiment establishes a
distribution-free guarantee for other inputs or host conditions.

## Verification and historical notes

Native differential tests compare all four distance families against
independent whole-term oracles for empty, short, multibyte, seeded, and
exhaustive corpora. The complete-term cursor is checked across all six
backends, including persistent deletion and revision pinning. C ABI tests
cover UTF-8, limits, batch leases, cancellation, and nonpartial failure;
the Julia facade checks its public layout and behavior. Full CI receipts must
refer to the exact source revision before any release claim.

The older [technical analysis](./technical-analysis.md),
[architectural sketches](./architectural-sketches.md), and
[implementation plan](./implementation-plan.md) are historical design notes.
In particular, their proposed bidirectional extension and old file paths
must not be read as a description of the current production query path.
