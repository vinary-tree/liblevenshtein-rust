@testset "lazy bounded temporal range queries" begin
    first = [1.0, 2.5]
    source = LL.TemporalSeriesSource([10 => first, 20 => [2.0, 3.0],
        30 => Float64[]])
    first[2] = 100.0
    @test length(source) == 3
    @test !isempty(source)

    for (kind, parameter0, parameter1, band) in (
        (:msm, 1.0, 0.0, 0),
        (:erp, 0.0, 0.0, 0),
        (:twed, 1.0, 0.0, 0),
        (:dtw, 0.0, 0.0, 1),
        (:frechet, 0.0, 0.0, 0),
        (:soft_dtw, 1.0, 0.0, 0),
    )
        query = [1.0, 2.0]
        expected = [(entry.id, outcome.value)
            for entry in source.entries
            for outcome in (LL.temporal_distance(kind, query, entry.samples;
                parameter0, parameter1, band, cutoff=Inf),)
            if outcome.kind == :finite]
        cursor = LL.query_temporal_range(source, kind, query;
            parameter0, parameter1, band)
        @test [(match.id, match.distance) for match in cursor] == expected
        @test !isopen(cursor)
    end

    soft_source = LL.TemporalSeriesSource([99 => [0.0, 0.0]])
    soft_cursor = LL.query_temporal_range(soft_source, :soft_dtw,
        [0.0, 0.0]; parameter0=1.0, cutoff=-0.5)
    @test only(collect(soft_cursor)).distance < -0.5

    cursor = LL.query_temporal_range(source, :erp, [1.0, 2.0];
        cutoff=Inf)
    @test LL.reduce_batches!((ids, batch) ->
        append!(ids, (match.id for match in batch)), UInt64[], cursor;
        batch_size=1) == UInt64[10, 20, 30]
    @test !isopen(cursor)

    left_cursor = LL.query_temporal_range(source, :erp, [1.0, 2.0])
    right_cursor = LL.query_temporal_range(source, :erp, [1.0, 2.0])
    left_task = Threads.@spawn collect(left_cursor)
    right_task = Threads.@spawn collect(right_cursor)
    @test fetch(left_task) == fetch(right_task)

    candidates = LL.query_temporal_range(source, :erp, [1.0, 2.0];
        query_limits=LL.TemporalQueryLimits(max_candidates=1))
    @test LL.next_batch!(candidates, 1)[1].id == 10
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(candidates, 1)
    @test !isopen(candidates)

    results = LL.query_temporal_range(source, :erp, [1.0, 2.0];
        query_limits=LL.TemporalQueryLimits(max_results=1))
    @test LL.next_batch!(results, 1)[1].id == 10
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(results, 1)

    cells = LL.query_temporal_range(source, :erp, [1.0, 2.0];
        query_limits=LL.TemporalQueryLimits(max_total_dp_cells=3))
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(cells)

    native = LL.query_temporal_range(source, :erp, [1.0, 2.0];
        limits=LL.TemporalLimits(max_dp_cells=1))
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(native)

    stopped = LL.query_temporal_range(source, :msm, [1.0, 2.0];
        parameter0=1.0)
    @test iterate(stopped) !== nothing
    LL.cancel!(stopped)
    @test !isopen(stopped)

    @test_throws ArgumentError LL.TemporalSeriesSource([1 => [NaN]])
    @test_throws ArgumentError LL.TemporalSeriesSource([1 => [1.0], 1 => [2.0]])
    @test_throws ArgumentError LL.TemporalSeriesSource([1 => [1.0]];
        max_source_bytes=0)
    @test_throws ArgumentError LL.query_temporal_range(source, :erp, [NaN])
    @test_throws ArgumentError LL.query_temporal_range(source, :erp,
        UntouchableLargeVector(4);
        limits=LL.TemporalLimits(max_series_len=3))
end
