# Candidate identity, result order, and witness contract

This source-mapped contract serves pgmcp task `cbc-01-results`. It instantiates
the result component of the [CBC obligation ledger](../theory/metric-automata-cbc-obligations.md)
and the ordered-result premises of [LOCPA BF-1](../theory/lazy-ordered-cost-product-automata.md#6-lazy-synchronized-products).
It is a specification and source audit, not a proof that the Rust traversals
already satisfy every clause. The [domain matrix](CBC_DOMAIN_CAPABILITIES.md)
and [numeric authority](CBC_NUMERIC_AUTHORITY.md) determine which inputs and
scores are eligible. No production optimization or benchmark is performed here.

## 1. Original identity is operation-specific

Fix a captured index revision $`\sigma`$. Let $`U_\sigma`$ be the finite set of
**live originals** for one operation. A result identifies an element of
$`U_\sigma`$, not a quantization label, collision bucket, shared dictionary
node, automaton state, or caller metadata value. A presentation value may
repeat while the original identity differs. Conversely, an API may choose a
value as its key and replace its prior original on reinsertion.

| Operation | Live original identity and source | Consequence of equal representation or payload |
|---|---|---|
| Elastic index range and kNN | The key `V` of the [originals map](../../src/time_series/elastic/walker.rs#L690). [Insertion](../../src/time_series/elastic/walker.rs#L2638) replaces an existing `V`; each live key has one full-precision series and one bucket location. | Different `V` keys in one quantized bucket remain distinct and require separate exact scoring. Reinserting the *same* `V` replaces it; it does not add a second result. Bucket/slot is a traversal coordinate, not a stable public key across mutation. |
| Physical-time TWED range and kNN | The monotonically assigned [episode ID](../../src/time_series/timestamped_twed_index.rs#L320) stored alongside every insertion. The public [match](../../src/time_series/timestamped_twed_index.rs#L212) includes that ID and borrows the original series. | Two episodes may have equal tokenized paths, scores, and caller `value`; they remain distinct because their IDs differ. Collision bucket storage retains both. |
| Unit or weighted dictionary query iterator | The accepted [unit path](../../src/transducer/query.rs#L1165) or [float path](../../src/transducer/query_f64.rs#L434), materialized before `QueryResult::from_match`. [UnitCandidate](../../src/transducer/query_result.rs#L104) retains raw units; `Candidate` presents text. | Distinct root-to-final paths remain distinct even when a DAWG shares their final physical node. These iterators return terms, not one row per equal caller payload. For lossy `CharUnit::to_string`, a text presentation is insufficient as an identity; use the units-native result profile. |

The scope of equality is therefore a named part of the API. In particular,
deduplicating elastic results by `V` is consistent with its replacement
semantics, while deduplicating TWED results by `value`, quantized tokens, or
node address would discard live episodes. A result theorem must use the
selected $`U_\sigma`$ throughout its coverage, rank, and witness clauses.

**Collision example.** Let elastic keys `A` and `B` hold different original
series that quantize to the same byte key. If both exact scores are within
cutoff, range returns both keys once. A terminal-bucket test may reject the
entire bucket only with a certified lower bound covering *both* originals.
If key `A` is reinserted with a new series, the live population still contains
one `A` and one `B`. For TWED, inserting two equal episodes with the same
metadata value instead creates two IDs and therefore two results.

**Shared-suffix example.** Suppose dictionary paths `ab` and `cb` reach one
shared DAWG node. For query `xb`, both may be eligible at equal edit cost.
The iterator's pending items retain path context and materialize `ab` and
`cb` separately. A visited set keyed only by `(node, automaton state)` would
lose one term. A quotient may erase path identity only after proving that it
reconstructs all final paths and their exact multiplicity.

## 2. Complete-result specification

For a query $`q`$, let $`d_\sigma(q,x)`$ be the operation's authoritative exact
score of original $`x`$, and let $`E_\sigma(q)`$ be originals for which that
score is an admitted result under the selected operation. The strict bounded
kNN profiles admit finite scores below cost `TOP`; a legacy convenience range
with positive-infinite cutoff may instead pass a computed `TOP` through its
inclusive `within` comparison, as the [range fallback](../../src/time_series/elastic/walker.rs#L3291)
illustrates for ERP. Such a profile needs its own top/absence contract before
it can claim finite-result CBC. Let $`t_{\sigma,q}(x)`$ be the operation's
injective tie key, if a total ranked result is promised. The rank is

```math
R_{\sigma,q}(x)=\bigl(d_\sigma(q,x),t_{\sigma,q}(x)\bigr).
```

The cost comparator in this pair is the one declared by the
[numeric authority](CBC_NUMERIC_AUTHORITY.md); a tolerance or an ideal-real
order cannot silently replace it. Injectivity of the tie key applies to
distinct *live identities*. An exact range result at inclusive cutoff
$`\tau`$ is a permutation, in its declared order, of exactly

```math
\{x\in E_\sigma(q)\mid d_\sigma(q,x)\le\tau\}.
```

An exact kNN result is the first $`\min(k,|E_\sigma(q)|)`$ distinct originals
under its declared rank. Before $`k`$ eligible scores are verified, a kth
threshold does not exist. At $`k=0`$, complete output is empty after the API's
required input checks; at $`k>|E_\sigma(q)|`$, complete output includes every
eligible original. On **bounded outcome APIs**, a resource failure returns
`Incomplete`, whose partial set is not an exhaustive answer. Convenience
`Vec` and iterator APIs do not expose that tagged guarantee. Zero-cost or
equal-score ties never merge distinct identities.

For resumable elastic range, a [paused page](../../src/time_series/elastic/walker.rs#L1258)
returns `partial: None` and keeps verified matches in the continuation;
[`exact_partial`](../../src/time_series/elastic/walker.rs#L1193) borrows that
subset. [Cancellation](../../src/time_series/elastic/walker.rs#L1204) can
return a sorted `partial: Some(...)` because no continuation remains. The
analogous TWED range [resume/cancel paths](../../src/time_series/timestamped_twed_index.rs#L1196)
also distinguish retained continuation state from a final incomplete subset.
None of these subsets proves range exhaustion.

The source contracts differ in ordering:

| Surface | Complete output order and source | Proof boundary |
|---|---|---|
| Elastic bounded range | Exact score, then [position in the accumulated originals vector](../../src/time_series/elastic/walker.rs#L3732). Each live `V` appears once by the index invariant. | Prove cursor coverage, pending-match preservation across pages, and that accumulation position is assigned once per original. This position is query-session order, not a stable key across index mutations. |
| Elastic range with certificate (`V = u64`) | Exact score, then [stable `u64` ID](../../src/time_series/elastic/walker.rs#L4600). | The certificate verifier must reconstruct the same ID order and original membership. This is a different tie policy from bounded range. |
| Elastic convenience range | [Best score per `V`, then first encounter *of that minimum score*](../../src/time_series/elastic/walker.rs#L3693). On a strictly better duplicate, the finisher updates both score and encounter sequence. Defensive duplicate coalescing is compatible with one-live-`V` identity. | No claim about multiplicity of repeated *paths* can be inferred merely from this final coalescing. |
| Elastic bounded kNN | Full-precision scan in [bucket order](../../src/time_series/elastic/walker.rs#L3186), max heap of size at most $`k`$, [score then exact-verification encounter](../../src/time_series/elastic/walker.rs#L3791). | For a fixed snapshot, prove each live `V` is considered once and the heap retains the first $`k`$ under that order. The tie key is induced by this scan; bucket slots can change after mutation. |
| Elastic convenience kNN | [Best-first traversal](../../src/time_series/elastic/walker.rs#L3473) assigns its own exact-verification encounter sequence; [heap finalization](../../src/time_series/elastic/walker.rs#L3781) uses score then sequence. | A new scheduler can change equal-score output order. BF-1 may be instantiated only with a proved tie key and region floor matching this public order; bucket/slot floors cannot be substituted automatically. |
| Physical-time TWED range and bounded kNN | [Score with `total_cmp`, then episode ID](../../src/time_series/timestamped_twed_index.rs#L604). Range [sorts at completion](../../src/time_series/timestamped_twed_index.rs#L1240); kNN uses a [size-`k` max heap](../../src/time_series/timestamped_twed_index.rs#L493). | Prove each episode is scanned or owned by one pending region and that ID order breaks every equal-score tie. Equal caller values do not merge IDs. |
| Base `QueryIterator` and `QueryIteratorF64` | [Queued path traversal](../../src/transducer/query.rs#L1119) emits each accepted terminal path; the float variant follows the same [path materialization](../../src/transducer/query_f64.rs#L434). | These base iterators do not promise globally score-sorted kNN results. Their output profile is path/term enumeration within the cutoff; a rank theorem needs a separate scheduling contract. |
| `OrderedQueryIterator`, `query_ordered`, and `query_ranked` | The [ordered iterator](../../src/transducer/ordered_query.rs#L48) promises increasing integer edit distance, then ascending term text; [`query_ranked`](../../src/transducer/mod.rs#L908) aliases `query_ordered`. | Prove distance-layer exhaustion before emitting the next layer and lexicographic order within each layer. A text tie key is injective only on the admitted term identity/presentation domain. |
| `RankedValueQueryIterator` suggestions | [Current-layer sorting](../../src/transducer/ranked_value_query.rs#L419) orders increasing edit distance, then decreasing normalized scorer confidence, then term text ascending. | The rank depends on a caller scorer and stored value; a distance-only BF-1 key does not express it. Prove confidence normalization and complete layer materialization before final emission. Equal confidence and text may require a further identity key if distinct originals are admitted. |
| `PriorityQueryIterator` | Its [priority](../../src/transducer/priority_query.rs#L88) computes `f` with saturating `g + h`; the [queue comparator](../../src/transducer/priority_query.rs#L96) orders by `f`, then `g` and path fields before [terminal emission](../../src/transducer/priority_query.rs#L430). | This is a distinct scheduling and output profile; the source describes approximate lexicographic order. Do not treat `.take(k)` as the ordered iterator's exact distance/text top-$`k`$ without a separate admissibility, tie, and emission proof. |

For bounded elastic kNN, [candidate bounds and exact verification](../../src/time_series/elastic/walker.rs#L3233)
use the current kth cost inclusively. The [heap insertion](../../src/time_series/elastic/walker.rs#L3658)
does not replace an existing equal-cost result: a later exact encounter has a
larger sequence and therefore cannot precede the worst equal-cost member.
The current best-first convenience traversal also admits equality-bound
regions; it must not adopt cost-only equality pruning. The [BF-1 theorem](../theory/lazy-ordered-cost-product-automata.md#6-lazy-synchronized-products)
permits equality pruning with an admissible tie floor for the *same* rank
contract. The stronger conditioned floor in BF-2 requires its additional
cost-specific premise.

**Provisional kNN example.** With $`k=1`$, an exact candidate of rank
$`(5,2)`$ cannot be published as the final answer while a pending region may
contain rank $`(4,9)`$ or $`(5,1)`$. A verified heap is provisional until
every pending region is exhausted or certified unable to precede its worst
rank. [TWED incomplete outcomes](../../src/time_series/timestamped_twed_index.rs#L660)
label their sorted heap as `partial`; bounded elastic kNN returns no partial
heap on failure. Neither partial is a complete top-$`k`$ certificate.

## 3. Witness observation is separate from score order

The search results above identify candidates and scores. They do not expose a
per-result alignment trace. The [tagged score-only outcome](../../src/time_series/bounded.rs#L360)
uses `NoWitness`. ERP, TWED, Fréchet, and banded DTW [alignment scorers](../../src/time_series/alignment.rs#L1536)
return a versioned [TemporalAlignmentWitness](../../src/time_series/alignment.rs#L119);
MSM has its own [MsmAlignmentWitness](../../src/time_series/alignment.rs#L267).
Each witness must pass its kernel-specific replay before it is authoritative.

The alignment builder resolves equal *machine* transition costs by its
[local trial order](../../src/time_series/alignment.rs#L1179): `Align`, then
`AdvanceQuery`, then `AdvanceCandidate`; [replacement occurs only for a
strictly smaller candidate](../../src/time_series/alignment.rs#L1053).
This specifies a deterministic local predecessor choice for those scorers.
MSM traceback [tries Move, then Merge, then Split](../../src/time_series/alignment.rs#L575)
using bitwise score equality and has a separate proof obligation.
It does not establish a globally lexicographically least alignment among all
optimal witnesses, nor does it show that the score-only search and witness
builder use identical floating expression graphs. Replay alone checks a
feasible trace and its accumulated score: for ERP query `[1]`, candidate
`[1]`, and gap zero, an `AdvanceQuery` then `AdvanceCandidate` trace scores two
although `Align` scores zero. A canonical-witness CBC certificate therefore
needs both equality to the authoritative optimum and the canonical tie proof.
An extractor run after result selection is sufficient only when that extractor
has these properties and its arithmetic agrees with the declared authority.

For example, a path with several zero-cost ERP alignments can have one
minimum score and multiple valid operation traces. A score-only rewrite may
retain any representative that preserves the score; it cannot assert that a
specified canonical trace is unchanged. Witness bytes, replay work, and
resource failure are observations in any witness-producing contract.

## 4. Obligations for certified optimizations

1. **Ownership:** partition live originals into pending regions, private
   scoring, verified results, and soundly excluded originals. A shared node
   may own several path occurrences. A collision bucket may own several
   originals. [CertifiedStructuralSplit.v](temporal_automata/theories/CertifiedStructuralSplit.v)
   proves an abstract exact-multiplicity partition for supplied originals;
   binding that partition to Rust snapshot enumeration remains open.
2. **Unique rank:** choose the identity and tie key of the *surface being
   optimized*. Prove the tie key is injective on live originals and that its
   comparison is the public one. Changing a schedule may require preserving
   its encounter order or changing and documenting that API contract.
3. **Heap:** let $`H`$ be the set of distinct verified eligible originals.
   After each exact verification, retain precisely the best
   $`\min(k,|H|)`$ verified ranks. A score-only threshold cannot exclude an
   earlier equal-score tie. Before the heap is full, no kth bound is valid.
4. **Emission:** range completion covers all eligible originals in the
   promised order; kNN completion additionally excludes every unresolved
   improving rank. A `partial` result remains tagged `Incomplete`.
5. **Witness:** declare `score-only`, `any replayable optimum`, or a specific
   canonical optimum. Prove a rewrite against that exact observation profile;
   do not infer witness preservation from score preservation.

Two negative controls are decisive. Node-only product deduplication fails the
`ab`/`cb` shared-suffix example. Treating the first verified kNN heap entry as
final fails the pending-rank example. Neither is repaired by a mathematically
admissible cost bound alone; both require identity and session coverage.
