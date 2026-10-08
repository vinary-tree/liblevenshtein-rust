@testset "native temporal index range cursors" begin
    query = [1.0, 2.0, 3.0]
    candidates = [(UInt64(10), [1.0, 2.0, 3.0]),
        (UInt64(20), [1.0, 2.5, 3.0])]
    limits = LL.TemporalSearchLimits(; max_series_len=3,
        max_scratch_bytes=4 * 1024 * 1024,
        max_continuation_bytes=4 * 1024 * 1024)
    for (kind, options) in ((:msm, (; parameter0=1.0)),
        (:erp, (; parameter0=0.0)), (:twed, (; parameter0=0.0)),
        (:dtw, (; band=2)), (:frechet, (;)))
        index = LL.TemporalIndex(kind; quant_min=-10.0,
            quant_max=10.0, quant_bins=32, max_entries=3,
            max_total_samples=9, max_series_len=3, options...)
        for (id, values) in candidates
            LL.insert!(index, id, values)
        end
        LL.freeze!(index)
        cursor = LL.query_index_range(index, query; cutoff=100.0,
            limits, page_work_units=1_000, page_results=1)
        close(index)
        actual = collect(cursor)
        @test sort([match.id for match in actual]) == UInt64[10, 20]
        for (id, values) in candidates
            expected = LL.temporal_distance(kind, query, values; options...)
            match = only(filter(value -> value.id == id, actual))
            @test expected.kind === :finite
            @test isapprox(match.distance, expected.value; atol=1e-10)
        end
        @test !isopen(cursor)
    end

    index = LL.TemporalIndex(:msm; quant_min=0.0, quant_max=10.0,
        parameter0=1.0, max_entries=1, max_total_samples=3,
        max_series_len=3)
    LL.insert!(index, 1, query)
    @test_throws LL.NativeError LL.insert!(index, 2, query)
    LL.freeze!(index)
    @test_throws ArgumentError LL.insert!(index, 1, query)
    cursor = LL.query_index_range(index, query; cutoff=0.0,
        limits, page_work_units=1_000, page_results=1)
    @test LL.reduce_batches!((sum, batch) -> sum + length(batch),
        0, cursor; batch_size=1) == 1
    @test !isopen(cursor)

    restricted = LL.TemporalSearchLimits(; max_series_len=3,
        max_work_units=0, max_scratch_bytes=4 * 1024 * 1024)
    stopped = LL.query_index_range(index, query; cutoff=10.0,
        limits=restricted, page_work_units=1_000)
    @test_throws LL.TemporalQueryIncomplete collect(stopped)
    @test !isopen(stopped)
    close(index)
end
