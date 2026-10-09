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
approx_advisory_index = LL.ApproxMsmIndex(pairs;
    segments=4, candidate_limit=4,
    max_entries=32, max_total_samples=512,
    max_series_len=16, max_total_features=128)
approx_exhaustive_index = LL.ApproxMsmIndex(pairs;
    segments=4, candidate_limit=32,
    max_entries=32, max_total_samples=512,
    max_series_len=16, max_total_features=128)
terms = LL.SourceFilterSource(["term-" * lpad(string(i), 3, '0')
    for i in 1:32]; max_terms=32, max_term_bytes=8,
    max_source_bytes=256)
ngram_index = LL.NativeSourceFilterIndex(terms;
    mode=:ngram, ngram_size=2, max_terms=32,
    max_term_bytes=8, max_source_bytes=256)
hybrid_index = LL.NativeSourceFilterIndex(terms;
    mode=:hybrid, ngram_size=2, jaro_threshold=0.7,
    max_terms=32, max_term_bytes=8, max_source_bytes=256)
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
vector_metric = LL.FixedChannelMetric([
    LL.VectorChannel("value", "unit"),
    LL.VectorChannel("trend", "unit"),
]; training_fold="benchmark-fold", estimator_revision="scale-v1")
vector_query_points = Matrix{Float64}(undef, 2, length(query))
vector_candidate_points = Matrix{Float64}(undef, 2, length(candidate))
for position in eachindex(query)
    vector_query_points[:, position] = [query[position], query[position] / 2]
    vector_candidate_points[:, position] =
        [candidate[position], candidate[position] / 2]
end
vector_query = LL.VectorTemporalSeries(vector_query_points)
vector_candidate = LL.VectorTemporalSeries(vector_candidate_points)

scan() = collect(LL.query_temporal_range(source, :dtw, query;
    band=2, cutoff=0.5))
indexed() = collect(LL.query_index_range(index, query;
    cutoff=0.5, page_work_units=100_000, page_results=32))
approx_advice() = LL.query_approx_msm_knn(approx_advisory_index,
    query, 3)
approx_full() = LL.query_approx_msm_knn(approx_exhaustive_index,
    query, 3)
msm_scan() = sort([(UInt(i - 1), id,
    LL.msm_distance(query, samples).value)
    for (i, (id, samples)) in enumerate(pairs)];
    by=match -> (match[3], match[1]))[1:3]
msm_prefilter() = collect(LL.filter_msm_source(source, query;
    threshold=1.0))
function msm_prefilter_native_c()
    kept = LL.MsmPrefilterCandidate[]
    limits = Ref(LL.TemporalLimits())
    for entry in source.entries
        output = Ref(LL.RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
        status = GC.@preserve query entry ccall(
            LL.native(:llev_temporal_lower_bound), Cint,
            (Ptr{Float64}, Csize_t, Ptr{Float64}, Csize_t,
                UInt32, Float64, Csize_t, Ref{LL.TemporalLimits},
                Ref{LL.RawTemporalDistanceResult}),
            pointer(query), length(query), pointer(entry.samples),
            length(entry.samples), UInt32(6), 1.0, 0, limits, output)
        status == 0 || error("native MSM prefilter control failed")
        output[].kind == 0 && output[].value <= 1.0 &&
            push!(kept, LL.MsmPrefilterCandidate(entry.id,
                copy(entry.samples), output[].value))
    end
    kept
end
online() = collect(LL.online_observations(:erp, query, candidate;
    cutoff=100.0))
ngram() = collect(LL.query_ngram(terms, "term-007", 1;
    page_candidates=32))
hybrid() = collect(LL.query_hybrid(terms, "term-007", 1;
    page_candidates=32))
indexed_ngram() = collect(LL.query_ngram(ngram_index, "term-007", 1;
    page_results=32))
indexed_hybrid() = collect(LL.query_hybrid(hybrid_index, "term-007", 1;
    page_results=32))
build_ngram() = close(LL.NativeSourceFilterIndex(terms;
    mode=:ngram, ngram_size=2, max_terms=32,
    max_term_bytes=8, max_source_bytes=256))
build_hybrid() = close(LL.NativeSourceFilterIndex(terms;
    mode=:hybrid, ngram_size=2, jaro_threshold=0.7,
    max_terms=32, max_term_bytes=8, max_source_bytes=256))
timestamped() = collect(LL.query_metric_range(timestamped_index,
    timestamped_query; cutoff=0.5, page_work_units=100_000,
    page_results=32))
timestamped_scan() = [id for (id, series) in timestamped_pairs if
    LL.metric_timestamped_twed_distance(timestamped_config,
        timestamped_query, series; cutoff=0.5).kind === :finite]
timestamped_knn() = LL.query_metric_knn(timestamped_index,
    timestamped_query, 3)
timestamped_knn_scan() = sort([(UInt64(i - 1), id,
    LL.metric_timestamped_twed_distance(timestamped_config,
        timestamped_query, series).value)
    for (i, (id, series)) in enumerate(timestamped_pairs)];
    by=match -> (match[3], match[1]))[1:3]
vector_erp() = LL.vector_erp_distance(vector_metric,
    vector_query, vector_candidate; gap=[0.0, 0.0])
vector_frechet() = LL.vector_frechet_distance(vector_metric,
    vector_query, vector_candidate)
vector_frechet_config = Ref(LL.RawVectorTemporalConfig(5, 0, C_NULL,
    0.0, 0.0, 0, Inf))
vector_frechet_limits = Ref(LL.VectorTemporalLimits())
vector_query_view = Ref(LL.raw_vector_series(vector_query))
vector_candidate_view = Ref(LL.raw_vector_series(vector_candidate))
function vector_frechet_native_c()
    output = Ref(LL.RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve vector_metric vector_query vector_candidate ccall(
        LL.native(:llev_vector_temporal_distance), Cint,
        (Ptr{Cvoid}, Ref{LL.RawVectorSeriesView},
            Ref{LL.RawVectorSeriesView}, Ref{LL.RawVectorTemporalConfig},
            Ref{LL.VectorTemporalLimits},
            Ref{LL.RawTemporalDistanceResult}),
        vector_metric.handle, vector_query_view, vector_candidate_view,
        vector_frechet_config, vector_frechet_limits, output)
    status == 0 || error("native vector Fréchet control failed")
    output[].value
end
vector_online() = collect(LL.vector_frechet_online_observations(
    vector_metric, vector_query, eachcol(vector_candidate.samples);
    cutoff=100.0))

try
    expected = sort([match.id for match in scan()])
    actual = sort([match.id for match in indexed()])
    expected == actual || error("indexed and scanned result IDs differ")
    LL.proves_recall(approx_full()) ||
        error("exhaustive approximate MSM search did not prove recall")
    !LL.proves_recall(approx_advice()) ||
        error("advisory approximate MSM search claimed recall")
    [(neighbor.insertion_index, neighbor.id, neighbor.distance)
        for neighbor in approx_full()] == msm_scan() ||
        error("exhaustive approximate MSM neighbors differ from scalar scan")
    timestamped_expected = sort(timestamped_scan())
    timestamped_actual = sort([match.id for match in timestamped()])
    timestamped_expected == timestamped_actual ||
        error("timestamped indexed and scalar result IDs differ")
    indexed_ngram() == ngram() ||
        error("persistent n-gram candidates differ from source filter")
    indexed_hybrid() == hybrid() ||
        error("persistent hybrid candidates differ from source filter")
    [(match.episode_id, match.id, match.distance)
        for match in timestamped_knn()] == timestamped_knn_scan() ||
        error("timestamped nearest neighbors differ from exact scalar scan")
    online_result = online()
    length(online_result) == length(candidate) ||
        error("online result length differs from candidate length")
    online_result[end].distance_within_cutoff ≈
        LL.erp_distance(query, candidate).value ||
        error("online ERP final score differs from scalar ERP")
    vector_online_result = vector_online()
    length(vector_online_result) == size(vector_candidate.samples, 2) ||
        error("vector online prefix count differs from candidate length")
    vector_online_result[end].distance_within_cutoff ≈
        vector_frechet().value ||
        error("online vector Fréchet differs from its native exact score")
    vector_frechet_native_c() == vector_frechet().value ||
        error("Julia vector Fréchet differs from the direct C control")
    [(entry.id, entry.score) for entry in msm_prefilter()] ==
        [(entry.id, entry.score) for entry in msm_prefilter_native_c()] ||
        error("Julia MSM prefilter differs from the direct C control")
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
    sample("vector ERP 16 x 16", vector_erp)
    sample("vector Frechet 16 x 16", vector_frechet)
    sample("vector Frechet direct C", vector_frechet_native_c)
    sample("vector online 16 prefixes", vector_online)
    sample("32-entry temporal scan", scan)
    sample("32-entry temporal index", indexed)
    sample("32-entry scalar MSM kNN", msm_scan)
    sample("32-entry MSM prefilter", msm_prefilter)
    sample("32-entry MSM direct C", msm_prefilter_native_c)
    sample("32-entry advisory MSM kNN", approx_advice)
    sample("32-entry exhaustive MSM kNN", approx_full)
    sample("32-entry timestamped scan", timestamped_scan)
    sample("32-entry timestamped index", timestamped)
    sample("32-entry timestamped kNN scan", timestamped_knn_scan)
    sample("32-entry timestamped kNN index", timestamped_knn)
    sample("32-term ngram filter", ngram)
    sample("32-term hybrid filter", hybrid)
    sample("32-term native ngram index", indexed_ngram)
    sample("32-term native hybrid index", indexed_hybrid)
    sample("32-term native ngram build", build_ngram)
    sample("32-term native hybrid build", build_hybrid)
finally
    close(plan)
    close(index)
    close(approx_advisory_index)
    close(approx_exhaustive_index)
    close(timestamped_index)
    close(ngram_index)
    close(hybrid_index)
    close(vector_metric)
end
