@testset "bounded native hybrid MSM range" begin
    query = [1.0, 2.0, 3.0]
    source = [(UInt64(7), [1.0, 2.0, 3.0]),
        (UInt64(8), [1.1, 2.1, 3.1]),
        (UInt64(9), [3.0, 2.0, 1.0]),
        (UInt64(10), [9.0, 9.0, 9.0])]
    index = LL.NativeHybridMsmIndex(quant_min=0.0,
        quant_max=10.0, quant_bins=5, msm_cost=1.0,
        max_entries=4, max_total_samples=12, max_series_len=3)
    try
        for (id, samples) in source
            LL.insert!(index, id, samples)
        end
        @test_throws LL.NativeError LL.insert!(index, 11, [1.0])
        @test_throws ArgumentError LL.insert!(index, 11, [NaN])
        @test_throws ArgumentError LL.insert!(index, 11,
            UntouchableLargeVector(4))
        @test_throws ArgumentError LL.query_hybrid_msm_range(index,
            query; cutoff=3.0)
        LL.freeze!(index)
        @test_throws ArgumentError LL.insert!(index, 11, [1.0])
        cursor = LL.query_hybrid_msm_range(index, query; cutoff=3.0,
            page_work_units=1_000, page_results=1)
        knn_cursor = LL.query_hybrid_msm_knn(index, query, 2;
            initial_threshold=0.0, page_results=1)
        LL.close!(index)
        expected = Tuple{UInt64, Float64}[]
        for (id, samples) in source
            outcome = LL.msm_distance(query, samples; cutoff=3.0)
            outcome.kind === :finite &&
                push!(expected, (id, outcome.value))
        end
        observed = sort([(match.id, match.distance) for match in cursor])
        @test observed == sort(expected)
        @test !isopen(cursor)
        nearest = [(match.id, match.distance) for match in knn_cursor]
        @test nearest == sort(expected; by=last)[1:2]
        @test !isopen(knn_cursor)
    finally
        close(index)
    end
end

@testset "hybrid candidate limits, lifecycle, and advisory bounds" begin
    index = LL.NativeHybridMsmIndex(quant_min=0.0,
        quant_max=10.0, quant_bins=5, msm_cost=1.0,
        max_entries=2, max_total_samples=4, max_series_len=2)
    try
        LL.insert!(index, 7, [1.0, 2.0])
        LL.insert!(index, 8, [1.0, 2.0])
        LL.freeze!(index)
        cursor = LL.query_hybrid_msm_range(index, [1.0, 2.0];
            cutoff=0.0, limits=LL.TemporalSearchLimits(max_results=1),
            page_results=1)
        @test length(LL.next_batch!(cursor, 1)) == 1
        failure = try
            LL.next_batch!(cursor, 1)
            nothing
        catch error
            error
        end
        @test failure isa LL.TemporalQueryIncomplete
        @test failure.detail === :results
        @test !isopen(cursor)
        @test_throws LL.TemporalQueryIncomplete LL.query_hybrid_msm_knn(
            index, [1.0, 2.0], 2;
            limits=LL.TemporalSearchLimits(max_results=1))
        @test isempty(collect(LL.query_hybrid_msm_knn(
            index, [1.0, 2.0], 0)))
        @test_throws LL.TemporalQueryIncomplete LL.query_hybrid_msm_range(
            index, [1.0, 2.0];
                cutoff=0.0,
                limits=LL.TemporalSearchLimits(max_continuation_bytes=0))
        @test LL.reduce_batches!((count, batch) -> count + length(batch),
            0, LL.query_hybrid_msm_range(index, [1.0, 2.0];
                cutoff=0.0, page_results=1); batch_size=1) == 2
        tasks = [Threads.@spawn sort([match.id for match in
            LL.query_hybrid_msm_range(index, [1.0, 2.0]; cutoff=0.0)])
            for _ in 1:2]
        @test all(task -> fetch(task) == UInt64[7, 8], tasks)
    finally
        close(index)
    end
    safe = LL.NativeHybridMsmIndex(quant_min=0.0,
        quant_max=100.0, quant_bins=2, msm_cost=1.0,
        lower_bound=:length, max_entries=1,
        max_total_samples=3, max_series_len=3)
    heuristic = LL.NativeHybridMsmIndex(quant_min=0.0,
        quant_max=100.0, quant_bins=2, msm_cost=1.0,
        lower_bound=:euclidean, max_entries=1,
        max_total_samples=3, max_series_len=3)
    try
        LL.insert!(safe, 7, [0.0, 0.0, 100.0])
        LL.insert!(heuristic, 7, [0.0, 0.0, 100.0])
        LL.freeze!(safe)
        LL.freeze!(heuristic)
        @test [(match.id, match.distance) for match in
            LL.query_hybrid_msm_range(safe, [0.0, 100.0];
                cutoff=1.0)] == [(UInt64(7), 1.0)]
        @test isempty(collect(LL.query_hybrid_msm_range(heuristic,
            [0.0, 100.0]; cutoff=1.0)))
    finally
        close(safe)
        close(heuristic)
    end
    @test_throws ArgumentError LL.NativeHybridMsmIndex(
        quant_min=0.0, quant_max=10.0, lower_bound=:bogus)
    @test_throws LL.NativeError LL.NativeHybridMsmIndex(
        quant_min=0.0, quant_max=10.0, msm_cost=-1.0)
end
