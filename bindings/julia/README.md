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

## Indexed temporal range queries

`TemporalIndex` stores full precision series under unsigned IDs and uses byte
quantization to share search prefixes. It supports native MSM, ERP, unit-grid
TWED, banded DTW, and discrete Fréchet range search. `freeze!` makes the index
immutable; each `query_index_range` cursor retains that snapshot, so the
index handle may be closed while a cursor remains active. Soft-DTW is available
through `query_temporal_range`, which scans a copied bounded source because
there is no native elastic index for its loss.

```julia
using Liblevenshtein

index = TemporalIndex(:dtw; quant_min=0.0, quant_max=10.0,
    quant_bins=64, band=2, max_entries=100,
    max_total_samples=1_000, max_series_len=32)
Liblevenshtein.insert!(index, 7, [1.0, 2.0, 3.0])
Liblevenshtein.insert!(index, 8, [1.0, 2.2, 3.0])
freeze!(index)

cursor = query_index_range(index, [1.0, 2.0, 3.0];
    cutoff=0.5, page_work_units=10_000, page_results=16,
    limits=TemporalSearchLimits(max_series_len=32, max_results=100))
try
    for match in cursor
        println(match.id, ": ", match.distance)
    end
finally
    close(cursor)
    close(index)
end
```

The cutoff is inclusive and DTW returns root-distance units. `next_batch!`
performs at most one native page and returns owned matches. An empty batch
means the page paused before finding a match; call it again. `reduce_batches!`
closes the cursor on every exit path. A completed empty cursor proves no match.
If a cumulative resource limit
is reached, iteration raises `TemporalQueryIncomplete` with a `detail`
identifying the exhausted resource; earlier matches are
only an exact subset and do not prove that other matches are absent. Set finite
source and query limits for the intended workload, and close a cursor when
stopping early.

## Soft-DTW gradient analysis

`soft_dtw_gradient(left, right; gamma, limits)` computes the complete
Soft-DTW loss and derivatives with respect to both nonempty finite scalar
series. It returns `SoftDtwGradientOutcome`. A finite outcome owns two Julia
gradient vectors, one entry for each sample of the corresponding input. An
incomplete outcome has no loss or gradients and identifies a DP, work,
scratch, or numeric limit in `reason`. The native implementation bounds the
full forward and reverse matrices before allocation; this batch analysis
uses storage proportional to the product of the two series lengths.

```julia
analysis = soft_dtw_gradient([0.0, 1.0], [0.5, 1.5]; gamma=0.75,
    limits=TemporalLimits(max_series_len=2, max_dp_cells=4,
        max_work_units=8, max_scratch_bytes=256))
@assert analysis.kind === :finite
@assert length(analysis.left_gradient) == 2
@assert length(analysis.right_gradient) == 2
```

Soft-DTW is a differentiable loss and may be negative. Its gradients are
derivatives of that loss, not metric distances or search lower bounds.
`soft_dtw_loss` evaluates the score alone with constant row storage when
gradients are unnecessary.

## Online temporal automata

`TemporalOnlineAutomaton` holds a copied fixed query and scores successive
target prefixes with native MSM, ERP, TWED, banded DTW, or discrete Fréchet.
The initial `observation` describes the empty target. Each `advance!` returns
an exact observation after one committed finite sample. The returned
`distance_within_cutoff` is `nothing` if the complete prefix score exceeds
the inclusive cutoff; it does not imply that a later prefix cannot match.
DTW online observations and their cutoff use **squared** distance units;
`dtw_distance` uses root-distance units. Soft-DTW has no online machine.

```julia
machine = TemporalOnlineAutomaton(:erp, [1.0, 2.0, 3.0]; cutoff=5.0)
try
    @assert observation(machine).consumed_target_len == 0
    step = advance!(machine, 1.0)
    @assert step.kind === :advanced
    @assert step.observation.consumed_target_len == 1
finally
    close(machine)
end
```

`TemporalOnlineLimits` caps copied query length, live frontier positions,
work per target sample, and retained scratch bytes. A resource-incomplete
step has `kind === :incomplete`, names its limit in `reason`, and leaves the
previous observation intact. The Julia handle serializes native calls and
must be closed when no longer needed. `online_observations` wraps a source
iterator as a one-shot lazy prefix stream; `reduce_observations!`, exhaustion,
`cancel!`, and errors close it. An incomplete stream raises
`TemporalOnlineIncomplete`.

## Rolling temporal windows

`BoundedRollingWindow` consumes one finite sample at a time with fixed
retained storage. Its first `RollingWindowSnapshot` appears after
`window_len` samples; later snapshots appear every `stride` samples. Each
snapshot owns chronological values and zero-based stream offsets. `advance!`
returns a tagged `RollingWindowStep` with the snapshot, if emitted, and
per-step usage. Invalid samples leave the machine unchanged.

`rolling_windows` wraps any Julia input iterator as a one-shot lazy snapshot
stream. `reduce_windows!` and `cancel!` close it; a thrown input or budget
error closes it as well. Each snapshot can open a native
`query_index_range` cursor against a frozen temporal index.

```julia
stream = rolling_windows(0.0:8.0, 3, 2)
windows = collect(stream)
@assert [(w.start_offset, w.values) for w in windows] ==
    [(0, [0.0, 1.0, 2.0]), (2, [2.0, 3.0, 4.0]),
     (4, [4.0, 5.0, 6.0]), (6, [6.0, 7.0, 8.0])]
```

Construction checks `max_series_len`, `max_scratch_bytes`, and
`max_snapshot_bytes` before allocating the ring. The values property returns
a copy, so a caller cannot change a previously emitted window by mutating
that returned vector. The input iterator remains owned by its caller.

## Copied source filtering

`SourceFilterSource` copies and deduplicates Unicode terms under explicit
entry, term-byte, and source-byte limits. `query_ngram` and `query_hybrid`
return `SourceFilterCursor` iterators over this snapshot. Each candidate is
checked with the native n-gram or n-gram/Jaro-Winkler predicate as the cursor
advances. `SourceFilterLimits` caps cumulative candidate inspections,
accepted results, and hybrid Jaro comparisons. `next_batch!` examines at
most `page_candidates` source terms, so an empty batch means the search
paused before finding a candidate.

```julia
source = SourceFilterSource(["hello", "help", "world"];
    max_terms=3, max_term_bytes=16, max_source_bytes=48)
cursor = query_hybrid(source, "helo", 1; ngram_size=2,
    jaro_threshold=0.7, page_candidates=2,
    limits=SourceFilterLimits(max_candidates=3, max_results=3))
try
    for term in cursor
        println(term)
    end
finally
    close(cursor)
end
```

Iteration returns exact native-filter candidates in source order. It does not
claim that every candidate lies within the edit-distance cutoff; the n-gram
filter is a conservative prefilter. A completed empty cursor proves no source
term passed that predicate. Exhausted limits raise
`SourceFilterIncomplete`, leaving only an exact subset of filter candidates.

`NativeSourceFilterIndex` builds a persistent native postings index over the
same copied, deduplicated source. Choose `mode=:ngram` or `mode=:hybrid` at
construction. Its source size, term bytes, query bytes, n-gram width, and
hybrid threshold are fixed when built; n-gram mode requires a zero threshold.
`query_ngram` or `query_hybrid` first
computes the complete native candidate-ID set under `SourceFilterLimits`,
then returns a `NativeSourceFilterCursor` that yields terms in source order in
bounded `page_results` batches. A limit error exposes no partial candidate
set. The cursor keeps its copied ID snapshot and source terms after the index
handle closes; close or cancel it when no longer needed.

```julia
index = NativeSourceFilterIndex(source; mode=:hybrid,
    ngram_size=2, jaro_threshold=0.7, max_terms=3,
    max_term_bytes=16, max_source_bytes=48)
cursor = query_hybrid(index, "helo", 1;
    limits=SourceFilterLimits(max_candidates=3, max_results=3),
    page_results=2)
close(index)
@assert collect(cursor) == ["hello", "help"]
```

The native index speeds repeated candidate generation only when its postings
construction cost is amortized. Its candidate set is still a prefilter;
perform an exact edit-distance comparison when the application needs a final
distance claim.

## Metric temporal domains

Raw ERP gives zero cost to inserting or deleting its gap value, and raw
discrete Fréchet gives zero cost to repeated adjacent samples. Those raw
sequences form pseudometric domains. `MetricErpConfig` defines one finite gap
quotient; `representative(config, series)` removes every gap occurrence.
`FrechetStutterClass` collapses each run of equal samples and requires a
nonempty path. `canonical_samples` returns an owned copy. Construction checks
`max_series_len` before reading input and rejects nonfinite samples.

`metric_erp_distance` and `metric_frechet_distance` call the exact native
scalar kernels on those canonical representatives. A mismatched ERP gap,
including a different signed-zero bit pattern, is rejected. The result is a
bounded `TemporalDistanceOutcome`, so exhausted DP, work, or scratch limits
remain explicit.

`MetricMsmConfig` requires a finite positive split/merge cost and restricts
`metric_msm_distance` to nonempty series. `MetricTwedConfig` requires positive
finite stiffness and a finite nonnegative gap penalty for unit-grid TWED.
`metric_twed_distance` uses that validated configuration. These are the
metric subdomains of the corresponding full native parameter families.

`MetricErpIndex` and `MetricFrechetIndex` canonicalize before inserting into
the native temporal index. Freeze them before calling `query_metric_range`;
the result is a lazy `TemporalIndexCursor` with the same paging, cumulative
limits, close, and snapshot retention as `query_index_range`.
`MetricMsmIndex` and `MetricTwedIndex` use the corresponding validated
configurations and the same cursor contract; the MSM wrapper rejects empty
stored and query series. All four constructors seal the selected native
kernel, so the public wrapper cannot be built around a different algorithm.

```julia
config = MetricErpConfig(0.0)
index = MetricErpIndex(config; quant_min=-10.0, quant_max=10.0,
    max_entries=2, max_total_samples=8, max_series_len=4)
try
    insert!(index, 7, [1.0, 0.0, 2.0])
    insert!(index, 11, [1.0, 2.0])
    freeze!(index)
    cursor = query_metric_range(index, [1.0, 0.0, 2.0]; cutoff=0.0)
    @assert sort([match.id for match in cursor]) == UInt64[7, 11]
finally
    close(index)
end
```

### Physical-time TWED

`TimestampedSeries` copies finite scalar values and finite, strictly
increasing timestamps in one canonical unit: `:seconds`, `:milliseconds`,
`:microseconds`, or `:nanoseconds`. It requires a nonempty series and a finite
physical origin no later than its first timestamp. `values` and `timestamps`
properties return copies. A `MetricTimestampedTwedConfig` requires finite
positive stiffness and finite nonnegative gap penalty. Both operands must
have the same unit and origin.

`metric_timestamped_twed_distance` uses the native physical-time recurrence,
an inclusive cutoff, and `TemporalLimits`. It returns the same finite,
above-cutoff, and incomplete result tags as the scalar temporal metrics.
The unit-grid `metric_twed_distance` is a distinct metric domain.

```julia
config = MetricTimestampedTwedConfig(0.5, 1.0)
left = TimestampedSeries([1.0, 2.0], [10.0, 13.0];
    unit=:milliseconds, origin=10.0)
right = TimestampedSeries([1.0, 2.5], [10.0, 14.0];
    unit=:milliseconds, origin=10.0)
score = metric_timestamped_twed_distance(config, left, right)
@assert score.kind == :finite
```

`TimestampedTwedIndex` retains full-precision episodes behind typed value/time
quantization. Give it finite increasing value and time domains, bin counts,
and explicit entry, total-sample, and per-series limits. Insertion copies an
episode and returns a stable episode ID through `insert_episode!`; duplicate
caller IDs remain distinct episodes. Freeze before querying. Each
`query_metric_range` cursor lazily traverses the captured native revision,
verifies quantization collisions at full precision, and keeps its snapshot
after the index handle closes. `TimestampedTwedSearchLimits` combines common
cumulative search ceilings with product-state limits. A page may return no
matches while paused; exhaustion raises `TemporalQueryIncomplete` when the
cursor advances. `next_batch!`, iteration, `reduce_batches!`, close, and
`cancel!` follow the other temporal cursor contracts.

`query_metric_knn(index, query, k; limits=TemporalSearchLimits())` returns
the exact nearest episodes ordered by distance and stable episode ID. An
exact top-`k` claim requires examining every stored episode, so this bounded
operation completes before returning its vector. Candidate, recurrence-cell,
work, scratch, and result limits fail closed with `TemporalQueryIncomplete`;
no partial neighbor list is exposed. The query copy counts against the
scratch ceiling. The frozen index can serve independent concurrent queries.

```julia
index = TimestampedTwedIndex(config;
    unit=:milliseconds, origin=10.0,
    value_min=0.0, value_max=5.0,
    time_min=10.0, time_max=20.0,
    max_entries=2, max_total_samples=4, max_series_len=2)
try
    insert_episode!(index, 7, left)
    insert_episode!(index, 7, left)
    freeze!(index)
    cursor = query_metric_range(index, left; cutoff=0.0)
    @assert sort([match.episode_id for match in cursor]) == UInt64[0, 1]
    @assert [match.episode_id for match in query_metric_knn(index, left, 2)] ==
        UInt64[0, 1]
finally
    close(index)
end
```

## Approximate MSM nearest neighbors

`ApproxMsmIndex` ranks stored finite series by piecewise aggregate
approximation (PAA): it divides each series into `segments` regions and uses
their means to choose a bounded candidate pool. Native MSM then computes an
exact full-precision distance for every admitted candidate. `candidate_limit`
controls that pool; the effective count is at least the requested `k` and at
most the number of indexed episodes. A zero segment count ranks by length.

```julia
index = ApproxMsmIndex([7 => [1.0, 2.0], 7 => [1.1, 2.1],
    11 => [9.0, 9.0]];
    segments=2, candidate_limit=2,
    max_entries=3, max_total_samples=6,
    max_series_len=2, max_total_features=6)
try
    result = query_approx_msm_knn(index, [1.0, 2.0], 1;
        limits=TemporalSearchLimits(max_dp_cells=1000))
    for neighbor in result
        println(neighbor.id, ": ", neighbor.distance)
    end
    @assert result.kind in (:advisory, :exhaustive)
finally
    close(index)
end
```

Each emitted `ApproxMsmNeighbor` has an exact MSM distance and stable
zero-based insertion index, even when caller IDs repeat. `result.kind` is
`:exhaustive` only when every indexed episode was decided by exact MSM;
`proves_recall(result)` checks that coverage. An `:advisory` result, including
an empty one from `k=0`, cannot prove recall or absence. An `:incomplete`
result carries its stop `reason` and may retain exact partial neighbors, but
also cannot prove recall. `indexed_entries`, `candidate_entries`, and
`exact_reranked` expose the coverage accounting. Construction copies all
series under entry, sample, length, and feature-storage ceilings; query
budgets use `TemporalSearchLimits`. Results own their neighbor vectors and
remain usable after the frozen index closes. `reduce_batches!` folds bounded
batches of that result.

## Replayable temporal alignment witnesses

`temporal_alignment` extracts a deterministic native witness for MSM, ERP,
unit-grid TWED, banded DTW, or discrete Fréchet. `timestamped_twed_alignment`
does the same for metric physical-time TWED. Extraction respects DP, work,
scratch, series-length, and witness-byte ceilings. `:finite` owns a witness;
`:above_cutoff`, `:no_alignment`, and `:incomplete` do not. Incompletion
reports the exact stop class, including `:witness_bytes`.

```julia
result = temporal_alignment(:erp, [1.0, 2.0], [1.0, 2.5];
    parameter0=0.0, max_witness_bytes=1024)
@assert result.kind === :finite
try
    @assert replay_alignment(result.witness, [1.0, 2.0],
        [1.0, 2.5]) == result.distance
    @assert !isempty(alignment_page(result.witness, 1; page_size=2))
finally
    close(result.witness)
end
```

The native handle owns a bounded snapshot of the operation path and borrows
no input arrays after extraction. Iteration and `reduce_batches!` copy one
bounded page at a time. Endpoints in Julia are one-based; MSM operation tags
are `:move`, `:merge`, and `:split`, while the other kernels use `:align`,
`:advance_query`, and `:advance_candidate` with exact local-cost bits.
`replay_alignment` checks the path against the supplied operands and the
configuration captured at extraction. Close the witness when finished.

## Typed vector temporal metrics

`FixedChannelMetric` retains an ordered channel/unit schema, positive scales
and weights, and the training-fold identity used to estimate those scales.
Construct it once and reuse it for many comparisons. A
`VectorTemporalSeries` stores one vector point per matrix column, with
channels in rows. Native calls copy bounded points into the existing Rust
vector kernels; they never flatten a point into unrelated scalar time steps.
`VectorTemporalSeries` checks sample count, dimension, and input byte limits
before making its one owned Julia copy.

```julia
metric = FixedChannelMetric([
    VectorChannel("x", "metre"; scale=2.0, weight=3.0),
    VectorChannel("y", "metre"; scale=4.0, weight=5.0),
]; training_fold="train-1", estimator_revision="scales-v1")
query = VectorTemporalSeries([0.0 2.0 4.0; 1.0 3.0 1.0])
candidate = VectorTemporalSeries([0.0 2.0 3.0; 1.0 2.0 1.0])
try
    score = vector_erp_distance(metric, query, candidate; gap=[0.0, 0.0])
    @assert score.kind === :finite
    @assert score.value >= 0
finally
    close(metric)
end
```

`vector_erp_distance` removes exact gap points to select a metric quotient
representative. `vector_frechet_distance` collapses consecutive identical
points for its path quotient. `vector_dtw_distance` requires a band and is a
nonmetric diagnostic score. `vector_timestamped_twed_distance` requires
strictly increasing physical timestamps on both series, a shared unit and
origin, and a fixed vector sentinel. Vector MSM has no admitted metric
because its scalar betweenness rule has no proven vector equivalent.
`VectorTemporalLimits` bounds input copies, dimension, band width, dynamic
programming cells, work, and scratch storage. The result uses the same
`:finite`, `:above_cutoff`, `:no_alignment`, and `:incomplete` kinds as
scalar temporal scores.

For interval pruning, `VectorIntervalBox(metric, bounds)` copies one finite
closed interval per channel and retains the metric's exact channel/unit
layout. `vector_point_box_lower_bound` and
`vector_box_box_lower_bound` call the native fixed-channel K1 bounds.
`vector_erp_interval_match_lower_bound`,
`vector_erp_interval_gap_lower_bound`, and
`vector_frechet_interval_link_lower_bound` use the same native point-to-box
bound; `vector_dtw_interval_local_lower_bound_squared` uses its squared local
cost. `vector_candidate_lower_bound` invokes each native K4 family method:
ERP gap mass, Fréchet endpoints, and the currently coherent zero bound for
DTW and timestamped TWED. `TimestampedVectorIntervalBox` adds an exact
physical-time unit and interval for native TWED deletion and match bounds.
All native bound calls accept `VectorTemporalLimits` and reject a mismatched
channel layout.

```julia
metric = FixedChannelMetric([
    VectorChannel("x", "metre"), VectorChannel("y", "metre"),
]; training_fold="train-1", estimator_revision="scales-v1")
box = VectorIntervalBox(metric, [(-1.0, 1.0), (2.0, 3.0)])
try
    @assert vector_point_box_lower_bound(metric, [0.0, 0.0], box) == 2.0
finally
    close(metric)
end
```

`VectorFrechetOnlineAutomaton` compares one fixed vector query with a target
stream one point at a time. Each `advance!` yields an exact committed-prefix
observation or an incomplete step that leaves the previous prefix intact.
Retained native storage stays fixed as the target grows. The machine owns its
query and metric copies, so both Julia inputs and the metric handle may close
after construction. `vector_frechet_online_observations` wraps any iterator
of vector points in a one-shot lazy stream; `reduce_observations!` consumes
and closes it.

```julia
metric = FixedChannelMetric([
    VectorChannel("x", "metre"), VectorChannel("y", "metre"),
]; training_fold="train-1", estimator_revision="scales-v1")
query = VectorTemporalSeries([0.0 1.0; 0.0 1.0])
target = VectorTemporalSeries([0.0 1.0 2.0; 0.0 1.0 2.0])
stream = vector_frechet_online_observations(
    metric, query, eachcol(target.samples); cutoff=4.0)
try
    prefixes = reduce_observations!(
        (out, observation) -> (push!(out, observation); out),
        TemporalOnlineObservation[], stream)
    @assert length(prefixes) == 3
finally
    close(stream)
    close(metric)
end
```

## Temporal lower bounds

`temporal_lower_bound` selects the native ERP gap-mass, Fréchet endpoint,
one-sided Hausdorff, combined Fréchet candidate, root-distance Keogh, or
explicit MSM prefilter score. The named Julia methods
`erp_gap_mass_lower_bound`, `frechet_endpoint_lower_bound`,
`frechet_one_sided_hausdorff_lower_bound`,
`frechet_candidate_lower_bound`, and `lb_keogh` expose the same
operations directly. `twed_length_lower_bound` uses only two lengths and a
nonnegative gap penalty. These ERP, Fréchet, Keogh, and TWED values are
admissible pruning bounds; they are not exact distances.

`msm_length_lower_bound(query, candidate, cost)` is also safe for exact MSM
pruning: each extra sample requires a split or merge with at least the
nonnegative `cost`. The native `msm_prefix_euclidean_heuristic`,
`msm_prefix_l1_heuristic`, and `msm_combined_heuristic` preserve Rust's
prefix-score definitions, but can exceed the true MSM distance. For example,
MSM at cost 1 between `[0, 100]` and `[0, 0, 100]` is 1, while all three
heuristics score 100. Use those heuristics only when approximate filtering
and possible false negatives are intended. All four modes use the same
bounded, tagged native result contract as the other temporal bounds.

`query_msm_with_safe_bound(source, query; split_merge_cost, cutoff)` composes
the native length bound with the lazy exact MSM source scan. It skips a
candidate before dynamic programming only when its admissible bound exceeds
the cutoff, keeps source order, and retains the source snapshot and cumulative
query ceilings. `query_temporal_range` also accepts `prefilter=:msm_length`.
Heuristic prefilters require `allow_false_negatives=true`; they can omit true
matches and are unsuitable for an exact range query.

`filter_msm_source(source, query; mode=:length, threshold)` exposes the
filter-only native operation as a one-shot Julia cursor. It evaluates one
candidate on demand and returns a copied `MsmPrefilterCandidate` containing
its source ID, samples, and selector score. `MsmPrefilterLimits` bound
cumulative candidates, emitted results, and native work; `TemporalLimits`
bound each native score. Use `next_batch!`, `reduce_batches!`, `close`, or
`cancel!` to control iteration. The three heuristic modes (`:euclidean`,
`:l1`, `:combined`) require the same explicit false-negative opt-in.

```julia
query = [1.0, 2.0, 3.0]
candidate = [1.0, 4.0, 3.0]
plan = keogh_envelopes(query, 1)
try
    interval = bounds_at(plan, 2)          # native centered envelope
    root = lb_keogh(candidate, plan)       # root-distance units
    squared = lb_keogh_squared(candidate, plan)
    @assert interval == (1.0, 3.0)
    @assert root.kind === :finite
    @assert squared.value ≈ root.value^2
finally
    close(plan)
end
```

`KeoghPlan` owns the finite query envelope and can score independent
candidates concurrently. `bounds_at` uses Julia's 1-based target positions.
Every scoring call takes explicit `TemporalLimits` for sample count, work,
and scratch storage. The result distinguishes finite bounds, impossible
alignment, and incomplete arithmetic or budgets. Empty and nonfinite inputs
retain each native bound's domain rules; a reusable Keogh plan requires a
nonempty finite query.

## Lossless temporal float encoding

`encode_f32`, `decode_f32`, `encode_f64`, and `decode_f64` preserve exact IEEE
bits, including signed zero and NaN payloads. `encode_f32_total_order` and
`decode_f32_total_order` use the same reversible unsigned ordering transform
as native Rust; `encode_f32_ordered` accepts nonnegative Float32 values only.
Series methods copy no more than their declared `max_samples` or `max_words`
before returning lazy encoded or decoded iterators. This snapshot stays stable
if the caller later changes its input vector.

```julia
bits = collect(encode_f32_series(Float32[-1, 0, 1]; max_samples=3))
@assert reinterpret.(UInt32, collect(decode_f32_series(bits))) == bits
words = collect(encode_f64_series_as_u32_pairs([1.0, -0.0]))
@assert reinterpret.(UInt64, collect(decode_u32_pairs_to_f64(words))) ==
    reinterpret.(UInt64, [1.0, -0.0])
```

The pair decoder ignores an unmatched final word, matching the native Rust
encoding contract. Rust and Julia tests use the same signed-zero, infinity,
and NaN-payload bit fixtures.

`QuantizationConfig` exposes Rust's uniform binning rules through
`quantize`, `dequantize`, `bin_bounds`, and `value_diff_to_bins`.
`try_uniform_quantizer` and `quantizer_from_data` return `nothing` when a
valid bin width cannot be formed. The first and last `bin_bounds` intervals
extend to negative and positive infinity because their bins absorb
outliers. NaN maps to the first bin and positive infinity to the last.
`encode_u8`, `encode_u32`, `decode_u8`, and `decode_u32` copy at most
`max_samples` input values and produce lazy output.

```julia
config = quantizer_u8(0.0, 100.0)
bins = collect(encode_u8(config, [0.0, 50.0, 100.0]))
@assert bins == UInt8[0, 128, 255]
@assert bin_bounds(config, 0)[1] == -Inf
```

`compute_deltas` and `reconstruct_from_deltas` return lazy iterators over
bounded snapshots. `encode_deltas_u8` returns the first sample and a lazy
byte-bin iterator; `decode_deltas_u8` reconstructs from those bins using their
quantized centers. The decoded series is approximate because quantization is
lossy. Empty source encoding returns `(0.0, empty_iterator)`, matching Rust.

`sax_breakpoints` provides the native breakpoint tables for alphabets of two
through ten symbols. `sax_normalize`, `sax_paa`, and `sax_encode` preserve the
native behavior for constant and short series, including repeated samples
when the word has more segments than the source. Each operation copies at
most `max_samples` values. Normalization emits at most `max_samples` values;
PAA and encoding emit at most `max_segments` values.
`sax_mindist` checks `max_word_len` before comparing two copied words and
returns the native lower bound for equal, nonempty words.

```julia
word = collect(sax_encode([1.0, 2.0, 3.0], 8, 4;
    max_samples=3, max_segments=8))
@assert word == UInt8[0, 0, 0, 2, 2, 2, 3, 3]
@assert sax_mindist(word, word, 3, 4) == 0.0
```

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
| `advance!` | `llev_generalized_online_advance`, `llev_universal_online_advance`, `llev_temporal_online_advance`, `llev_vector_frechet_online_advance` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation; project ABI operation |
| `AffineGapCosts` | `llev_affine_costs_validate` | project ABI operation |
| `alignment_page` | `llev_temporal_alignment_page` | project ABI operation |
| `api_revision` | `llev_api_revision` | ABI compatibility and feature discovery |
| `ApproxMsmIndex` | `llev_approx_msm_index_new`, `llev_approx_msm_index_insert`, `llev_approx_msm_index_freeze` | project ABI operation |
| `are_phonetically_similar` | `llev_phonetic_feature_relation` | IPA feature classification and relations |
| `articulatory_distance` | `llev_phonetic_articulatory_distance` | articulatory phonetic distance |
| `articulatory_edit_distance` | `llev_phonetic_articulatory_edit_distance` | articulatory phonetic distance |
| `bounds_at` | `llev_keogh_plan_bounds_at` | project ABI operation |
| `build_features` | `llev_build_features` | ABI compatibility and feature discovery |
| `cache_stats` | `llev_query_cache_stats` | project ABI operation |
| `cancel!` | `llev_wallbreaker_cursor_cancel` | project ABI operation |
| `certificate_evidence_at` | `llev_temporal_certificate_evidence_at` | project ABI operation |
| `certificate_info` | `llev_temporal_certificate_info` | project ABI operation |
| `certificate_match_page` | `llev_temporal_certificate_matches` | project ABI operation |
| `certificate_query_bits` | `llev_temporal_certificate_query_bits` | project ABI operation |
| `characters_with_features` | `llev_phonetic_chars_with_features` | IPA feature classification and relations |
| `clear!` | `llev_query_cache_clear` | project ABI operation |
| `close!` | `llev_transducer_free`, `llev_query_cache_free`, `llev_cost_cursor_free`, `llev_specialized_cursor_free`, `llev_query_cursor_free`, `llev_phonetic_pattern_free`, `llev_phonetic_rules_free`, `llev_phonetic_grep_free`, `llev_phonetic_dictionary_free`, `llev_phonetic_online_free`, `llev_phonetic_online_stream_free`, `llev_phonetic_token_free`, `llev_phonetic_transducer_free`, `llev_generalized_automaton_free`, `llev_generalized_online_free`, `llev_universal_automaton_free`, `llev_universal_online_free`, `llev_wallbreaker_free`, `llev_wallbreaker_cursor_free`, `llev_timestamped_twed_index_free`, `llev_timestamped_twed_cursor_free`, `llev_keogh_plan_free`, `llev_temporal_index_free`, `llev_quantized_index_free`, `llev_quantized_cursor_free`, `llev_hybrid_index_free`, `llev_hybrid_cursor_free`, `llev_temporal_index_cursor_free`, `llev_temporal_knn_cursor_free`, `llev_temporal_certificate_free`, `llev_temporal_online_free`, `llev_source_filter_index_free`, `llev_approx_msm_index_free`, `llev_temporal_alignment_free`, `llev_vector_metric_free`, `llev_vector_frechet_online_free` | transducer lifecycle, snapshot, or domain metadata; project ABI operation; streaming result traversal and batch leases; compiled phonetic-pattern lifecycle and matching; phonetic rule-set lifecycle and rewriting; word-boundary phonetic search and configuration; normalized phonetic dictionary construction, query, and updates; character-level phonetic search and scanner lifecycle; token-sequence phonetic matching and detail ownership; incremental phonetic rewriting and lifecycle; runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `compiled_phonetic_bytes` | `llev_owned_bytes_free`, `llev_phonetic_rules_to_bytes`, `llev_phonetic_pattern_to_bytes` | versioned compiled-byte ownership; versioned compiled phonetic-rule bytes; versioned compiled phonetic-pattern bytes |
| `distance` | `llev_distance`, `llev_distance_threshold`, `llev_distance_bytes`, `llev_distance_bytes_threshold`, `llev_distance_u64`, `llev_distance_u64_threshold` | standalone exact or thresholded distance |
| `distance_config` | `llev_phonetic_grep_distance_config` | word-boundary phonetic search and configuration |
| `erp_gap_mass_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `evaluate` | `llev_generalized_automaton_evaluate_utf8`, `llev_universal_automaton_evaluate` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `expand_feature_based` | `llev_phonetic_expand_feature_based` | IPA feature-driven expansion |
| `expand_phonetic_alternatives` | `llev_phonetic_expand` | bounded reverse phonetic expansion |
| `expand_phonetic_with_costs` | `llev_phonetic_expand_with_costs` | bounded reverse phonetic expansion |
| `feature_set_distance` | `llev_phonetic_feature_set_distance` | IPA feature classification and relations |
| `feed!` | `llev_phonetic_online_stream_feed`, `llev_phonetic_transducer_feed` | character-level phonetic search and scanner lifecycle; incremental phonetic rewriting and lifecycle |
| `filter_msm_source` | `llev_temporal_lower_bound` | project ABI operation |
| `finish!` | `llev_phonetic_online_stream_finish`, `llev_phonetic_transducer_finish` | character-level phonetic search and scanner lifecycle; incremental phonetic rewriting and lifecycle |
| `FixedChannelMetric` | `llev_vector_metric_new` | project ABI operation |
| `frechet_candidate_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `frechet_endpoint_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `frechet_one_sided_hausdorff_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `freeze!` | `llev_timestamped_twed_index_freeze`, `llev_temporal_index_freeze`, `llev_quantized_index_freeze`, `llev_hybrid_index_freeze` | project ABI operation |
| `GeneralizedAutomaton` | `llev_generalized_automaton_new` | runtime generalized-automaton lifecycle and prefix evaluation |
| `hybrid_candidate` | `llev_source_filter_utf8` | project ABI operation |
| `insert!` | `llev_phonetic_dictionary_update`, `llev_temporal_index_insert`, `llev_quantized_index_insert`, `llev_hybrid_index_insert` | normalized phonetic dictionary construction, query, and updates; project ABI operation |
| `insert_episode!` | `llev_timestamped_twed_index_insert` | project ABI operation |
| `is_free_phonetic_substitution` | `llev_phonetic_feature_relation` | IPA feature classification and relations |
| `jaro_similarity` | `llev_jaro_similarity_utf8` | project ABI operation |
| `keogh_envelopes` | `llev_keogh_plan_new` | project ABI operation |
| `lb_keogh` | `llev_temporal_lower_bound`, `llev_keogh_plan_score` | project ABI operation |
| `lb_keogh_squared` | `llev_keogh_plan_score` | project ABI operation |
| `load_compiled_phonetic_pattern` | `llev_phonetic_pattern_from_bytes` | versioned compiled phonetic-pattern bytes |
| `load_compiled_phonetic_rules` | `llev_phonetic_rules_from_bytes` | versioned compiled phonetic-rule bytes |
| `load_phonetic_pattern` | `llev_phonetic_pattern_load_llre_file` | trusted .llre loading with native imports |
| `load_phonetic_rules` | `llev_phonetic_rules_load_file` | trusted .llev loading with native includes |
| `match_distance` | `llev_phonetic_grep_matches` | word-boundary phonetic search and configuration |
| `merge_and_split_distance` | `llev_merge_and_split_distance`, `llev_merge_and_split_distance_threshold`, `llev_merge_and_split_distance_bytes`, `llev_merge_and_split_distance_bytes_threshold`, `llev_merge_and_split_distance_u64`, `llev_merge_and_split_distance_u64_threshold` | standalone merge-and-split distance |
| `metric_timestamped_twed_distance` | `llev_timestamped_twed_distance` | project ABI operation |
| `msm_combined_heuristic` | `llev_temporal_lower_bound` | project ABI operation |
| `msm_length_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `msm_prefix_euclidean_heuristic` | `llev_temporal_lower_bound` | project ABI operation |
| `msm_prefix_l1_heuristic` | `llev_temporal_lower_bound` | project ABI operation |
| `NativeError` | `llev_last_error_message` | typed failure diagnostics |
| `NativeHybridMsmIndex` | `llev_hybrid_index_new` | project ABI operation |
| `NativeQuantizedIndex` | `llev_quantized_index_new` | project ABI operation |
| `NativeSourceFilterIndex` | `llev_source_filter_index_new`, `llev_source_filter_index_insert`, `llev_source_filter_index_freeze` | project ABI operation |
| `next_batch!` | `llev_cost_cursor_next_batch`, `llev_cost_cursor_release_batch`, `llev_specialized_cursor_next_batch`, `llev_specialized_cursor_release_batch`, `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch`, `llev_wallbreaker_cursor_next_batch`, `llev_wallbreaker_cursor_release_batch`, `llev_timestamped_twed_cursor_next_batch`, `llev_quantized_cursor_next_batch`, `llev_hybrid_cursor_next_batch`, `llev_temporal_index_cursor_next_batch`, `llev_temporal_knn_cursor_next_batch` | project ABI operation; streaming result traversal and batch leases |
| `ngram_candidate` | `llev_source_filter_utf8` | project ABI operation |
| `normalize` | `llev_phonetic_transducer_normalize` | incremental phonetic rewriting and lifecycle |
| `normalized_query` | `llev_phonetic_online_normalized_query` | character-level phonetic search and scanner lifecycle |
| `observation` | `llev_generalized_online_observation`, `llev_universal_online_observation`, `llev_temporal_online_observation`, `llev_vector_frechet_online_observation` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation; project ABI operation |
| `online` | `llev_generalized_online_new_utf8`, `llev_universal_online_new` | runtime generalized-automaton lifecycle and prefix evaluation; universal-automaton lifecycle, policies, and prefix evaluation |
| `optimal_string_alignment_distance` | `llev_damerau_distance`, `llev_damerau_distance_threshold`, `llev_damerau_distance_bytes`, `llev_damerau_distance_bytes_threshold`, `llev_damerau_distance_u64`, `llev_damerau_distance_u64_threshold` | standalone exact or thresholded distance |
| `original_samples` | `llev_quantized_cursor_original` | project ABI operation |
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
| `query_affine` | `llev_transducer_query_affine` | domain-preserving dictionary query |
| `query_approx_msm_knn` | `llev_approx_msm_index_query_knn` | project ABI operation |
| `query_contextual` | `llev_transducer_query_contextual_utf8` | domain-preserving dictionary query |
| `query_filter_index` | `llev_source_filter_index_query` | project ABI operation |
| `query_filtered` | `llev_transducer_query_filtered_utf8` | domain-preserving dictionary query |
| `query_hybrid` | `llev_source_filter_utf8` | project ABI operation |
| `query_hybrid_msm_knn` | `llev_hybrid_index_query_knn` | project ABI operation |
| `query_hybrid_msm_range` | `llev_hybrid_index_query_range` | project ABI operation |
| `query_index_certified` | `llev_temporal_index_query_certified` | project ABI operation |
| `query_index_erp_automaton_range` | `llev_temporal_index_query_erp_automaton_range` | project ABI operation |
| `query_index_knn` | `llev_temporal_index_query_knn` | project ABI operation |
| `query_index_range` | `llev_temporal_index_query_range` | project ABI operation |
| `query_metric_knn` | `llev_timestamped_twed_index_query_knn` | project ABI operation |
| `query_metric_range` | `llev_timestamped_twed_index_query_range` | project ABI operation |
| `query_msm_with_safe_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `query_ngram` | `llev_source_filter_utf8` | project ABI operation |
| `query_pruned` | `llev_transducer_query_pruned_utf8` | domain-preserving dictionary query |
| `query_quantized` | `llev_quantized_index_query` | project ABI operation |
| `query_weighted` | `llev_transducer_query_weighted` | domain-preserving dictionary query |
| `QueryCache` | `llev_query_cache_new` | project ABI operation |
| `reduce_batches!` | `llev_cost_cursor_reduce`, `llev_specialized_cursor_reduce`, `llev_query_cursor_reduce` | project ABI operation; streaming result traversal and batch leases |
| `remove!` | `llev_phonetic_dictionary_update` | normalized phonetic dictionary construction, query, and updates |
| `replay_alignment` | `llev_temporal_alignment_replay`, `llev_timestamped_twed_alignment_replay` | project ABI operation |
| `reset!` | `llev_phonetic_transducer_reset` | incremental phonetic rewriting and lifecycle |
| `reset_stats!` | `llev_query_cache_reset_stats` | project ABI operation |
| `scan` | `llev_phonetic_online_scan`, `llev_phonetic_online_matches_free`, `llev_phonetic_token_scan`, `llev_phonetic_token_matches_free` | character-level phonetic search and scanner lifecycle; token-sequence phonetic matching and detail ownership |
| `scan_line` | `llev_phonetic_grep_scan_line` | word-boundary phonetic search and configuration |
| `scan_text` | `llev_phonetic_grep_scan_text` | word-boundary phonetic search and configuration |
| `scratch_bytes` | `llev_temporal_online_scratch_bytes`, `llev_vector_frechet_online_scratch_bytes` | project ABI operation |
| `similar_phonetic_chars` | `llev_phonetic_similar_chars` | IPA feature classification and relations |
| `snapshot` | `llev_transducer_snapshot` | transducer lifecycle, snapshot, or domain metadata |
| `soft_dtw_gradient` | `llev_soft_dtw_gradient` | project ABI operation |
| `streaming` | `llev_phonetic_online_stream_new` | character-level phonetic search and scanner lifecycle |
| `syllable_boundaries` | `llev_phonetic_syllable_boundaries` | syllable count and boundary heuristics |
| `syllable_count` | `llev_phonetic_syllable_count` | syllable count and boundary heuristics |
| `temporal_alignment` | `llev_temporal_alignment_new` | project ABI operation |
| `temporal_distance` | `llev_temporal_distance` | project ABI operation |
| `temporal_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `TemporalIndex` | `llev_temporal_index_new` | project ABI operation |
| `TemporalOnlineAutomaton` | `llev_temporal_online_new` | project ABI operation |
| `timestamped_twed_alignment` | `llev_timestamped_twed_alignment_new` | project ABI operation |
| `TimestampedTwedIndex` | `llev_timestamped_twed_index_new` | project ABI operation |
| `Transducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `true_damerau_distance` | `llev_true_damerau_distance`, `llev_true_damerau_distance_threshold`, `llev_true_damerau_distance_bytes`, `llev_true_damerau_distance_bytes_threshold`, `llev_true_damerau_distance_u64`, `llev_true_damerau_distance_u64_threshold` | standalone true-Damerau distance |
| `twed_length_lower_bound` | `llev_twed_length_lower_bound` | project ABI operation |
| `unit_domain` | `llev_transducer_unit_domain` | transducer lifecycle, snapshot, or domain metadata |
| `UniversalAutomaton` | `llev_universal_automaton_new` | universal-automaton lifecycle, policies, and prefix evaluation |
| `vector_box_box_lower_bound` | `llev_vector_box_box_lower_bound` | project ABI operation |
| `vector_candidate_lower_bound` | `llev_vector_temporal_candidate_lower_bound` | project ABI operation |
| `vector_frechet_ground_distance` | `llev_vector_frechet_ground_distance` | project ABI operation |
| `vector_point_box_lower_bound` | `llev_vector_point_box_lower_bound` | project ABI operation |
| `vector_temporal_distance` | `llev_vector_temporal_distance` | project ABI operation |
| `vector_twed_interval_delete_lower_bound` | `llev_vector_twed_interval_lower_bound` | project ABI operation |
| `VectorFrechetOnlineAutomaton` | `llev_vector_frechet_online_new`, `llev_vector_frechet_ground_online_new` | project ABI operation |
| `verify_certificate` | `llev_temporal_certificate_verify` | project ABI operation |
| `voicing_pair` | `llev_phonetic_voicing_pair` | IPA feature classification and relations |
| `WallBreakerMatcher` | `llev_wallbreaker_new_utf8` | project ABI operation |
| `WeightedOperationCosts` | `llev_operation_costs_preset`, `llev_operation_costs_validate` | project ABI operation |

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
