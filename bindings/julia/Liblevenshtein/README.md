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

For bounded ranking, use `query_ranked` or `query_mode`. For scores that order
only one distance layer at a time, use `query_suggestions`. The native
`query_contextual` cursor accepts Julia edit-cost callbacks, while
`query_pruned` accepts a balanced `PrefixVisitor` for dictionary DFS, and
`query_filtered` tests optional IDs before term construction. Its
`query_by_value` and `query_by_value_set` adapters capture one ID or a copied
set of IDs at query creation. All these
cursors capture one dictionary revision, support explicit close/cancel, and
stream batches without collecting a full result set. See the
[Julia package guide](docs/src/index.md#contextual-costs-and-prefix-pruning)
for callback lifetimes and examples.

`jaro_similarity` and `jaro_winkler_similarity` call the native Unicode
scorers with caller-chosen byte and comparison limits. `ngram_candidate` and
`hybrid_candidate` test one source against the corresponding native filter.
`query_jaro`, `query_ngram`, and `query_hybrid` apply those tests at final
source prefixes in a fuzzy traversal. They stream `SpecializedMatch` values
from one native snapshot; the
[source-filter guide](docs/src/index.md#native-source-filtering)
shows usage and limits.
`NativeSourceFilterIndex` stores a frozen native n-gram or hybrid postings
index over copied source terms. Its `query_ngram` and `query_hybrid` methods
produce closeable source-order candidate iterators. Candidate computation is
bounded and complete before iteration; the cursor can outlive the index
handle because it retains copied IDs and terms.

For exact decimal affine gaps, construct `AffineGapCosts(gap_open, gap_extend,
substitution)` and call `query_affine(transducer, query, maximum_cost, costs)`.
For floating per-operation weights, use `WeightedOperationCosts(:standard)`,
`:typo`, `:ocr`, or six custom weights with `query_weighted`. Both queries
preserve Unicode, byte, and u64 key domains and stream `CostMatch` values.
The [cost-domain guide](docs/src/index.md#affine-and-weighted-edit-costs)
explains exact scaled costs, supported algorithms, and cursor lifetimes.

## Scalar time series

`msm_distance`, `erp_distance`, `twed_distance`, `dtw_distance`,
`frechet_distance`, and `soft_dtw_loss` call their native scalar kernels through
one revision-11 ABI. Inputs are finite real vectors. A dense `Vector{Float64}`
is borrowed for the call; other vectors are converted once to contiguous
double arrays. No native handle survives the comparison.

```julia
limits = TemporalLimits(max_series_len=4096, max_dp_cells=1_000_000,
    max_work_units=1_000_000, max_scratch_bytes=1 << 20)
result = dtw_distance([1.0, 2.0, 3.0], [1.0, 2.5, 3.0];
    band=1, cutoff=2.0, limits)
@assert result.kind == :finite && result.value == 0.5
```

All kernels reserve their complete dynamic-programming, work, and scratch
budgets before computation. `kind` distinguishes `:finite`,
`:above_cutoff`, `:no_alignment`, and `:incomplete`. Only a finite result has a
`value`; an incomplete result carries a resource `reason`. Invalid samples or
configuration raise `NativeError`. TWED's `stiffness` and `gap_penalty` are
nonnegative. DTW requires an explicit band. Soft-DTW is a loss that may be
negative, so it must not be used as a metric index distance. Its cutoff may
likewise be negative. The
[temporal guide](docs/src/index.md#bounded-scalar-time-series) gives the native
parameter mapping and outcome contract.

`soft_dtw_gradient(left, right; gamma, limits)` evaluates the complete
Soft-DTW loss and both sample gradients. It requires nonempty finite series
and a positive finite smoothing parameter. A `SoftDtwGradientOutcome` carries
owned gradient vectors only when `kind === :finite`; an incomplete outcome
identifies the exhausted resource and carries no partial derivative. This
batch operation retains full forward and reverse matrices, so set explicit
DP, work, and scratch limits for the intended operand sizes. Use
`soft_dtw_loss` when only the score is needed.

For a finite collection of series, `TemporalSeriesSource` copies IDs and
samples into a bounded snapshot. `query_temporal_range` then compares one
candidate at a time with the selected native kernel and streams `TemporalMatch`
results through `next_batch!`, iteration, or `reduce_batches!`. Set
`TemporalQueryLimits` to cap total candidates, results, and DP cells across
the whole scan. Exhaustion raises `TemporalQueryIncomplete` with a reason,
preserving the distinction between a complete empty result and a stopped
query. This scan is intended for finite sources that fit its declared storage
limit; the [temporal guide](docs/src/index.md#lazy-temporal-range-queries)
shows its lifecycle.

`TemporalOnlineAutomaton` scores successive prefixes of a target stream
against a copied fixed query for MSM, ERP, TWED, DTW, and discrete Fréchet.
`observation(machine)` describes the empty target before the first sample;
`advance!(machine, sample)` commits one finite sample and returns its exact
prefix observation. A result outside the inclusive cutoff has
`distance_within_cutoff === nothing`. Online DTW uses squared distance units
for both scores and its cutoff, whereas `dtw_distance` returns root-distance
units. Soft-DTW has no online automaton.

```julia
machine = TemporalOnlineAutomaton(:erp, [1.0, 2.0, 3.0]; cutoff=5.0)
try
    @assert observation(machine).consumed_target_len == 0
    @assert advance!(machine, 1.0).kind === :advanced
finally
    close(machine)
end
```

`TemporalOnlineLimits` caps copied query length, live frontier positions,
per-sample work, and scratch bytes. A resource-incomplete step does not consume
its sample and carries a `reason`. `online_observations` is a one-shot lazy
stream over a caller-owned source iterator. `reduce_observations!`, normal
exhaustion, and errors close it; `cancel!` closes it when stopping early. The
[temporal binding guide](../../README.md#online-temporal-automata) documents
the resource and lifecycle contract.

`MetricErpConfig` and `ErpQuotientSeries` remove gap samples before computing
ERP scores or building an index. `FrechetStutterClass` collapses adjacent
repetitions before discrete Fréchet scores or index insertion. These canonical
representatives make the documented quotient spaces metric domains; raw ERP
and Fréchet series can have zero distance while differing as arrays.
`metric_erp_distance` and `metric_frechet_distance` retain the native bounded
score outcome. `MetricErpIndex` and `MetricFrechetIndex` wrap frozen native
range indexes; `query_metric_range` returns a lazy cursor that can outlive the
index handle. `query_metric_knn(index, query, k)` returns an exact bounded
result cursor over the same canonical domain; the full native scan completes
before results become visible, and the cursor survives closing the index.
For ERP, `query_metric_erp_automaton_range` instead traverses reachable
canonical antichain states in bounded pages and checks every full-precision
candidate. It accepts the same quotient representatives and gap identity.
See the [metric-domain guide](../../README.md#metric-temporal-domains)
for construction and ownership examples.

`MetricMsmConfig` validates a positive split/merge cost and a nonempty MSM
domain. `MetricTwedConfig` validates positive stiffness and nonnegative gap
penalty on the unit time grid. Their `metric_msm_distance` and
`metric_twed_distance` methods, plus `MetricMsmIndex` and `MetricTwedIndex`,
use the same native bounded distance, frozen range cursor, and exact kNN
cursor contracts.

`TimestampedSeries` owns finite values and strictly increasing physical
timestamps in a canonical unit and shared origin. `MetricTimestampedTwedConfig`
and `metric_timestamped_twed_distance` expose bounded native TWED on this
explicit-time metric domain. The [physical-time guide](../../README.md#physical-time-twed)
explains units, ownership, and result tags.
`TimestampedTwedIndex` inserts owned episodes, freezes one native revision,
and exposes lazy exact `query_metric_range` cursors with full-precision
collision verification and cumulative product limits. Duplicate caller IDs
receive distinct stable episode IDs.
`query_metric_knn` returns exact nearest episodes in distance and episode
order after a bounded complete native scan. It raises
`TemporalQueryIncomplete` on resource exhaustion without exposing partial
neighbors.

`ApproxMsmIndex` copies finite episodes into a frozen PAA-ranked source and
exactly reranks its bounded candidate pool with MSM. `query_approx_msm_knn`
returns tagged `:exhaustive`, `:advisory`, or `:incomplete` evidence with
coverage counts and exact neighbor distances. Only `proves_recall(result)`
authorizes an exact top-k or absence claim. See the
[approximate MSM guide](../../README.md#approximate-msm-nearest-neighbors).

`temporal_alignment` and `timestamped_twed_alignment` return bounded,
replayable native operation witnesses. A finite witness supports lazy pages,
iteration, `reduce_batches!`, `replay_alignment`, and `close`. Nonfinite
outcomes carry no handle. See the
[alignment witness guide](../../README.md#replayable-temporal-alignment-witnesses).

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

`benchmark/temporal_filtering.jl` measures scalar DTW, Soft-DTW gradients, reusable Keogh,
quantization, ERP canonicalization, SAX, rolling windows, online ERP prefixes,
a copied temporal scan, the frozen temporal index, physical-time TWED range
scan and index, exact physical-time nearest-neighbor scan and index,
scalar, advisory, and exhaustive MSM nearest-neighbor searches,
source-filter scans, persistent native n-gram and hybrid queries, and their
construction. Its fixed workload has 32 series of 16
samples and 32 eight-byte terms. It first checks that the indexed and scanned
result IDs, nearest-neighbor distances, and native/source filter candidates
agree, then samples five groups of 30 observable complete operations
after 20 warmups.
The workload and query budgets are finite. Run it after building the native
library and preparing the Julia binding environment:

```sh
LIBLEVENSHTEIN_LIBRARY="$PWD/target/debug/libliblevenshtein.so" \
  julia --startup-file=no --project=target/julia-test \
  bindings/julia/Liblevenshtein/benchmark/temporal_filtering.jl
```

On the 2026-10-09 local debug build (Julia 1.13.1, Threadripper PRO 5975WX,
one CPU quota, 2 GiB memory ceiling), median nanoseconds per complete call
were:

| Scenario | Median ns/op |
|---|---:|
| Scalar DTW, 16 × 16 samples | 4,940 |
| Soft-DTW gradients, 16 × 16 samples | 47,592 |
| Reusable Keogh bound, 16 samples | 1,403 |
| Quantization, 16 samples | 100 |
| ERP canonicalization, 16 samples | 54 |
| SAX, 16 samples to 4 symbols | 291 |
| Rolling windows, 16 samples to width 4 | 4,802 |
| Online ERP, 16 prefixes | 77,993 |
| 32-entry temporal scan | 161,913 |
| 32-entry temporal index | 3,082,520 |
| 32-entry scalar MSM kNN | 600,637 |
| 32-entry advisory MSM kNN | 57,004 |
| 32-entry exhaustive MSM kNN | 272,512 |
| 32-entry physical-time TWED range scan | 329,765 |
| 32-entry physical-time TWED range index | 709,150 |
| 32-entry physical-time TWED kNN scalar scan | 694,641 |
| 32-entry physical-time TWED kNN native index | 314,458 |
| 32-term n-gram source scan | 316,058 |
| 32-term hybrid source scan | 381,111 |
| 32-term native n-gram index query | 34,589 |
| 32-term native hybrid index query | 72,392 |
| 32-term native n-gram index build | 150,524 |
| 32-term native hybrid index build | 151,320 |

The range indexes are slower than their scans for this small, low-selectivity
source, while the native kNN scan is faster than repeated Julia scalar calls.
Advisory MSM is the fastest MSM search here because it reranks four of 32
episodes; its result does not prove top-k recall. The exhaustive MSM run does
prove recall and is faster than repeated scalar calls on this workload.
Persistent native source-filter queries are faster than the source scans on
this workload; the two build rows show their setup cost.
Measure each operation on the intended collection and cutoff before choosing
an index for speed. These local debug figures are diagnostic rather than a
portable performance guarantee.

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

- Use `encode_f32_series`, `decode_f32_series`,
  `encode_f64_series_as_u32_pairs`, and `decode_u32_pairs_to_f64` for exact
  Float32/Float64 trie words. They copy bounded input snapshots and produce
  lazy output; use `collect` only when storage of the full encoding is needed.
- Use `QuantizationConfig` with `encode_u8` for byte-trie words or
  `encode_u32` for wider alphabets. `bin_bounds` supplies the admissible
  interval needed by temporal pruning after outlier clamping.
- Use `encode_deltas_u8` when local differences have a useful bounded range.
  Use `sax_encode` for a symbolic word, and `sax_mindist` for its admissible
  lower bound. Both families use bounded input snapshots and lazy output.
- Use `rolling_windows` to feed unknown-length finite-sample streams into
  fixed-width snapshots, then pass a snapshot to `query_index_range` for
  exact bounded lookup against a frozen temporal index.
- Use `query_index_knn(index, query, k)` for exact nearest neighbors on a
  frozen MSM, ERP, TWED, DTW, or Fréchet index. Native search scans the whole
  snapshot under `TemporalSearchLimits` before returning a closeable result
  cursor. Iterate it or use `reduce_batches!` for bounded result copies. Limit
  exhaustion raises `TemporalQueryIncomplete` and yields no partial top-k.
  DTW neighbors carry public root-distance scores.
- Use `query_index_erp_automaton_range` on a frozen ERP index when the
  canonical automaton product is desired. It retains the same snapshot and
  cumulative limits as `query_index_range`, with bounded lazy result pages.
- Use `query_index_certified` when an exact indexed range result needs
  replayable K1–K4 evidence. Set `TemporalCertificateLimits` for cumulative
  traversal, witness, path, record, work, and result ceilings. Iterate
  `certificate_matches` and `certificate_evidence` without copying the whole
  certificate into Julia; `certificate_match_page` copies a bounded batch.
  The certificate retains the frozen snapshot after the index closes.

```julia
index = TemporalIndex(:dtw; quant_min=0, quant_max=10,
    band=2, max_entries=2, max_total_samples=6, max_series_len=3)
insert!(index, 7, [1.0, 2.0, 3.0])
freeze!(index)
certificate = query_index_certified(index, [1.0, 2.0, 3.0]; cutoff=0.0)
close(index)
try
    @assert only(collect(certificate_matches(certificate))).id == 7
    projection = read_certificate(certificate)
    @assert verify_certificate(certificate, projection)
finally
    close(certificate)
end
```

`read_certificate` explicitly materializes a complete caller-owned projection
within the configured ceilings. `verify_certificate` compares every supplied
query word, result, evidence field, path, and accounting value before native
replay against the retained snapshot; a changed well-formed projection returns
`false`. DTW accepts public root-distance cutoffs and exposes native squared
cutoffs in `certificate_info`.

For equal-dimensional vector paths in comparable coordinate units, use
`vector_frechet_ground_distance(ground, query, target)` with `ground` set to
`:l1`, `:l2`, or `:linf` to select the
audited Manhattan, Euclidean, or Chebyshev point metric. The native path
quotient removes consecutive equal points. Use
`VectorFrechetOnlineAutomaton(:l2, query; cutoff=...)` or
`vector_frechet_online_observations(:l2, query, points; cutoff=...)` to score
successive whole-point prefixes with fixed retained storage. A
`FixedChannelMetric` carries channel identity, physical units, fold-local
scales, and weights for typed vector comparisons.

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
  native `BUILD_FEATURE_PHONETIC` bit is present. Use
  `articulatory_distance` and `articulatory_edit_distance` for IPA feature
  scores, `syllable_count` and `syllable_boundaries` for orthographic or IPA
  heuristics, and `PhoneticGrep` for reusable word-boundary search over text.
  For normalized dictionaries, character-level and token-level grep,
  incremental rewriting, expansion, IPA feature queries, trusted file
  loaders, and versioned AOT bytes, see the [phonetic guide](docs/src/phonetic.md).

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
| `PhoneticFeatureWeights(; ...)` | Optional finite, nonnegative IPA feature costs. |
| `articulatory_distance(a::Char, b::Char; weights=nothing)` | Compare native IPA feature sets for two Unicode scalars. |
| `articulatory_edit_distance(a, b; weights=nothing, max_cells=4_000_000)` | Bounded native feature-weighted edit distance for strings. |
| `syllable_count(text; ipa=false, max_input_scalars=1_000_000)` | English orthographic or IPA syllable-count heuristic. |
| `syllable_boundaries(text; ipa=false, max_input_scalars=1_000_000)` | Zero-based Unicode-scalar starts, not Julia byte indices. |
| `PhoneticGrep(pattern; rules=nothing, max_distance=0, algorithm=ALGORITHM_STANDARD, case_insensitive=false)` | Reusable dictionary-free phonetic word matcher. |
| `match_distance(grep, candidate)`, `candidate in grep` | Bounded native candidate match, returning an edit cost or `nothing`. |
| `scan_line(grep, line)`, `scan_text(grep, document)` | Owned word matches with one-based line and zero-based byte offsets. |
| `distance_config(grep)` | Effective distance and optional pattern-local override. |
| `PhoneticNormalizedDictionary(terms; rules=nothing, compact=false)` | Mutable or immutable compact native normalized index; copied, relevance-ordered candidate query results. |
| `query(dictionary, text; max_distance=0)`, `insert!`, `remove!` | Normalized-space fuzzy candidates and mutation of the mutable index. |
| `PhoneticOnlineGrep(pattern; rules=nothing)`, `scan(grep, document)` | Character-level substring matches with copied source byte/scalar spans. |
| `streaming(grep)`, `feed!`, `finish!` | Bounded chunk-fed character-level scanning with single-use finish. |
| `PhoneticTokenGrep(query; rules=nothing)`, `scan(grep, document)` | Token-query grammar and copied per-token distance evidence. |
| `PhoneticTransducer(; rules=nothing)`, `feed!`, `finish!`, `reset!`, `normalize` | Context-aware incremental native rewriting and independent whole-input normalization. |
| `expand_phonetic_alternatives`, `expand_phonetic_with_costs` | Distinct exhaustive and greedy reverse-rule expansions with `PhoneticExpansionLimits`. |
| `phonetic_features`, `characters_with_features`, `feature_set_distance` | Stable IPA feature-set projection, native-table search, and weighted comparison. |
| `similar_phonetic_chars`, `voicing_pair`, `are_phonetically_similar`, `is_free_phonetic_substitution`, `expand_feature_based` | Distinct native IPA relations and expansions. |
| `load_phonetic_rules`, `load_phonetic_pattern` | Trusted local `.llev` include and `.llre` import resolution. |
| `compiled_phonetic_bytes`, `load_compiled_phonetic_rules`, `load_compiled_phonetic_pattern` | Optional API-revision-8 binary AOT roundtrip, gated by `BUILD_FEATURE_PHONETIC_AOT`. |
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
