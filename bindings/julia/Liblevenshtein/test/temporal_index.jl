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
    @test_throws ArgumentError LL.insert!(index, 2, UntouchableLargeVector(4))
    @test_throws LL.NativeError LL.insert!(index, 2, query)
    LL.freeze!(index)
    @test_throws ArgumentError LL.query_index_range(index,
        UntouchableLargeVector(4); limits)
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
    error = try
        collect(stopped)
        nothing
    catch caught
        caught
    end
    @test error isa LL.TemporalQueryIncomplete
    @test error.reason === :native_index
    @test error.detail === :work_units
    @test !isopen(stopped)
    close(index)
end

@testset "exact bounded temporal index kNN" begin
    query = [1.0, 2.0, 3.0]
    candidates = [(UInt64(7), [1.0, 2.0, 3.0]),
        (UInt64(8), [1.0, 2.5, 3.0])]
    for (kind, options) in ((:msm, (; parameter0=1.0)),
        (:erp, (; parameter0=0.0)), (:twed, (; parameter0=0.0)),
        (:dtw, (; band=2)), (:frechet, (;)))
        index = LL.TemporalIndex(kind; quant_min=-10.0, quant_max=10.0,
            quant_bins=32, max_entries=2, max_total_samples=6,
            max_series_len=3, options...)
        try
            for (id, values) in candidates
                LL.insert!(index, id, values)
            end
            LL.freeze!(index)
            @test isempty(collect(LL.query_index_knn(index, query, 0)))
            restricted = LL.TemporalSearchLimits(max_candidates=0)
            failure = try
                LL.query_index_knn(index, query, 2; limits=restricted)
                nothing
            catch error
                error
            end
            @test failure isa LL.TemporalQueryIncomplete
            @test failure.detail === :candidates
            cursor = LL.query_index_knn(index, query, 2; page_results=1)
            close(index)
            neighbors = collect(cursor)
            @test [match.id for match in neighbors] == UInt64[7, 8]
            for (match, (_, values)) in zip(neighbors, candidates)
                expected = LL.temporal_distance(kind, query, values;
                    options...)
                @test expected.kind === :finite
                @test match.distance ≈ expected.value
            end
            @test !isopen(cursor)
        finally
            close(index)
        end
    end
    index = LL.TemporalIndex(:erp; quant_min=-10.0,
        quant_max=10.0, max_entries=1, max_total_samples=3,
        max_series_len=3)
    try
        LL.insert!(index, 7, query)
        LL.freeze!(index)
        @test LL.reduce_batches!((count, batch) -> count + length(batch),
            0, LL.query_index_knn(index, query, 1); batch_size=1) == 1
        @test_throws ArgumentError LL.query_index_knn(index,
            UntouchableLargeVector(4), 1;
            limits=LL.TemporalSearchLimits(max_series_len=3))
    finally
        close(index)
    end
end

@testset "canonical ERP automaton index range" begin
    query = [1.0, 2.0, 3.0]
    index = LL.TemporalIndex(:erp; quant_min=-10.0,
        quant_max=10.0, max_entries=2, max_total_samples=6,
        max_series_len=3)
    try
        LL.insert!(index, 7, query)
        LL.insert!(index, 8, [1.0, 2.5, 3.0])
        @test_throws ArgumentError LL.query_index_erp_automaton_range(
            index, query; cutoff=1.0)
        LL.freeze!(index)
        @test_throws LL.TemporalQueryIncomplete LL.query_index_erp_automaton_range(
            index, query; cutoff=1.0,
            limits=LL.TemporalSearchLimits(max_scratch_bytes=0))
        cursor = LL.query_index_erp_automaton_range(index, query;
            cutoff=1.0, page_work_units=1_000, page_results=1)
        ordinary = LL.query_index_range(index, query; cutoff=1.0)
        close(index)
        automaton_matches = sort([(match.id, match.distance)
            for match in cursor])
        @test automaton_matches == sort([(match.id, match.distance)
            for match in ordinary])
        @test automaton_matches == [(UInt64(7), 0.0),
            (UInt64(8), LL.erp_distance(query,
                [1.0, 2.5, 3.0]; gap=0.0).value)]
        @test !isopen(cursor)
    finally
        close(index)
    end
    other = LL.TemporalIndex(:msm; quant_min=0.0,
        quant_max=5.0, parameter0=1.0, max_entries=1)
    try
        LL.freeze!(other)
        @test_throws LL.NativeError LL.query_index_erp_automaton_range(
            other, query; cutoff=1.0)
    finally
        close(other)
    end
end

@testset "native temporal index concurrent lifecycle" begin
    query = [1.0, 2.0, 3.0]
    for _ in 1:24
        index = LL.TemporalIndex(:msm; quant_min=0.0,
            quant_max=10.0, parameter0=1.0, max_entries=1,
            max_total_samples=3, max_series_len=3)
        LL.insert!(index, 1, query)
        LL.freeze!(index)
        gate = Channel{Nothing}(2)
        search = Base.Threads.@spawn begin
            take!(gate)
            try
                collect(LL.query_index_range(index, query; cutoff=0.0,
                    page_work_units=10_000, page_results=1))
            catch error
                error
            end
        end
        closing = Base.Threads.@spawn begin
            take!(gate)
            close(index)
        end
        put!(gate, nothing)
        put!(gate, nothing)
        result = fetch(search)
        fetch(closing)
        @test result isa ArgumentError ||
            (result isa Vector{LL.TemporalMatch} &&
                [entry.id for entry in result] == UInt64[1])
        @test !isopen(index)
    end

    index = LL.TemporalIndex(:msm; quant_min=0.0,
        quant_max=10.0, parameter0=1.0, max_entries=1,
        max_total_samples=3, max_series_len=3)
    LL.insert!(index, 1, query)
    LL.freeze!(index)
    cursor = LL.query_index_range(index, query; cutoff=0.0,
        page_work_units=10_000, page_results=1)
    close(index)
    gate = Channel{Nothing}(2)
    reading = Base.Threads.@spawn begin
        take!(gate)
        LL.next_batch!(cursor, 1)
    end
    closing = Base.Threads.@spawn begin
        take!(gate)
        close(cursor)
    end
    put!(gate, nothing)
    put!(gate, nothing)
    batch = fetch(reading)
    fetch(closing)
    @test batch === nothing || batch isa Vector{LL.TemporalMatch}
    @test !isopen(cursor)
end
