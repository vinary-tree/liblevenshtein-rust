using Liblevenshtein
using Statistics

const LL = Liblevenshtein
const BENCH_SINK = Ref{Any}(nothing)

function sample(label, operation; warmup=20, iterations=30, rounds=5)
    for _ in 1:warmup
        BENCH_SINK[] = operation()
    end
    GC.gc()
    elapsed = Float64[]
    for _ in 1:rounds
        started = time_ns()
        for _ in 1:iterations
            BENCH_SINK[] = operation()
        end
        push!(elapsed, (time_ns() - started) / iterations)
    end
    println(rpad(label, 32), " median=", round(Int, median(elapsed)),
        " ns/op min=", round(Int, minimum(elapsed)),
        " max=", round(Int, maximum(elapsed)))
    BENCH_SINK[] = nothing
end

query = [sin(i / 4) for i in 1:16]
candidate = query .+ 0.05
pairs = [UInt64(id) => query .+ (id - 16) / 100 for id in 1:32]
source = LL.TemporalSeriesSource(pairs;
    max_entries=32, max_series_len=16, max_source_bytes=4096)
index = LL.TemporalIndex(:dtw; quant_min=-4, quant_max=4,
    band=2, max_entries=32, max_total_samples=512, max_series_len=16)
for (id, samples) in pairs
    LL.insert!(index, id, samples)
end
LL.freeze!(index)
terms = LL.SourceFilterSource(["term-" * lpad(string(i), 3, '0')
    for i in 1:32]; max_terms=32, max_term_bytes=8,
    max_source_bytes=256)
plan = LL.keogh_envelopes(query, 2)
quantizer = LL.quantizer_u8(-4.0, 4.0)
metric_config = LL.MetricErpConfig(0.0)
timestamped_config = LL.MetricTimestampedTwedConfig(0.5, 1.0)
timestamps = collect(1.0:16.0)
timestamped_query = LL.TimestampedSeries(query, timestamps)
timestamped_pairs = [(id, LL.TimestampedSeries(samples, timestamps))
    for (id, samples) in pairs]
timestamped_index = LL.TimestampedTwedIndex(timestamped_config;
    value_min=-4.0, value_max=4.0, time_min=0.0, time_max=17.0,
    value_bins=16, time_bins=16,
    max_entries=32, max_total_samples=512, max_series_len=16)
for (id, series) in timestamped_pairs
    LL.insert!(timestamped_index, id, series)
end
LL.freeze!(timestamped_index)

scan() = collect(LL.query_temporal_range(source, :dtw, query;
    band=2, cutoff=0.5))
indexed() = collect(LL.query_index_range(index, query;
    cutoff=0.5, page_work_units=100_000, page_results=32))
online() = collect(LL.online_observations(:erp, query, candidate;
    cutoff=100.0))
ngram() = collect(LL.query_ngram(terms, "term-007", 1;
    page_candidates=32))
hybrid() = collect(LL.query_hybrid(terms, "term-007", 1;
    page_candidates=32))
timestamped() = collect(LL.query_metric_range(timestamped_index,
    timestamped_query; cutoff=0.5, page_work_units=100_000,
    page_results=32))
timestamped_scan() = [id for (id, series) in timestamped_pairs if
    LL.metric_timestamped_twed_distance(timestamped_config,
        timestamped_query, series; cutoff=0.5).kind === :finite]

try
    expected = sort([match.id for match in scan()])
    actual = sort([match.id for match in indexed()])
    expected == actual || error("indexed and scanned result IDs differ")
    timestamped_expected = sort(timestamped_scan())
    timestamped_actual = sort([match.id for match in timestamped()])
    timestamped_expected == timestamped_actual ||
        error("timestamped indexed and scalar result IDs differ")
    online_result = online()
    length(online_result) == length(candidate) ||
        error("online result length differs from candidate length")
    online_result[end].distance_within_cutoff ≈
        LL.erp_distance(query, candidate).value ||
        error("online ERP final score differs from scalar ERP")
    println("Julia ", VERSION, "; temporal entries=32 x 16 samples; ",
        "filter terms=32 x 8 bytes; result IDs=", length(actual))
    sample("scalar DTW 16 x 16", () ->
        LL.dtw_distance(query, candidate; band=2))
    sample("Soft-DTW gradients 16 x 16", () ->
        LL.soft_dtw_gradient(query, candidate; gamma=1.0))
    sample("reusable Keogh 16", () -> LL.lb_keogh(candidate, plan))
    sample("quantize 16 samples", () ->
        collect(LL.encode_u8(quantizer, query)))
    sample("canonical ERP 16 samples", () ->
        LL.representative(metric_config, query))
    sample("SAX 16 samples to 4", () ->
        collect(LL.sax_encode(query, 4, 4)))
    sample("rolling 16 to width 4", () ->
        collect(LL.rolling_windows(query, 4, 2)))
    sample("online ERP 16 prefixes", online)
    sample("32-entry temporal scan", scan)
    sample("32-entry temporal index", indexed)
    sample("32-entry timestamped scan", timestamped_scan)
    sample("32-entry timestamped index", timestamped)
    sample("32-term ngram filter", ngram)
    sample("32-term hybrid filter", hybrid)
finally
    close(plan)
    close(index)
    close(timestamped_index)
end
