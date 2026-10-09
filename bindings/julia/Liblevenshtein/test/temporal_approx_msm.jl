@testset "bounded approximate MSM index" begin
    entries = [UInt64(7) => [1.0], UInt64(7) => [2.0],
        UInt64(11) => [3.0]]
    exhaustive = LL.ApproxMsmIndex(entries;
        segments=1, candidate_limit=3,
        max_entries=3, max_total_samples=3,
        max_series_len=1, max_total_features=3)
    try
        entries[1].second[1] = 99.0
        result = LL.query_approx_msm_knn(exhaustive, [1.0], 1)
        @test result.kind === :exhaustive
        @test LL.proves_recall(result)
        @test (result.indexed_entries, result.candidate_entries,
            result.exact_reranked) == (3, 3, 3)
        @test [(neighbor.id, neighbor.insertion_index,
            neighbor.distance) for neighbor in result] ==
            [(UInt64(7), UInt(0), 0.0)]
        @test result.neighbors[1].distance ==
            LL.msm_distance([1.0], [1.0]).value
        @test result.usage.work_units > 0
        @test LL.reduce_batches!((ids, batch) ->
            append!(ids, [neighbor.id for neighbor in batch]),
            UInt64[], result; batch_size=1) == UInt64[7]
        tasks = [Threads.@spawn LL.query_approx_msm_knn(
            exhaustive, [1.0], 1) for _ in 1:2]
        @test all(task -> LL.proves_recall(fetch(task)), tasks)
        close(exhaustive)
        @test !isopen(exhaustive)
        @test collect(result)[1].distance == 0.0
        @test_throws ArgumentError LL.query_approx_msm_knn(
            exhaustive, [1.0], 1)
    finally
        close(exhaustive)
    end

    advisory = LL.ApproxMsmIndex([7 => [1.0], 7 => [2.0],
        11 => [3.0]]; segments=1, candidate_limit=1,
        max_entries=3, max_total_samples=3,
        max_series_len=1, max_total_features=3)
    try
        result = LL.query_approx_msm_knn(advisory, [1.0], 1)
        @test result.kind === :advisory
        @test !LL.proves_recall(result)
        @test result.neighbors[1].distance == 0.0
        @test (result.candidate_entries, result.exact_reranked) == (1, 1)
        empty_advice = LL.query_approx_msm_knn(advisory, [1.0], 0)
        @test empty_advice.kind === :advisory
        @test isempty(empty_advice)
        @test !LL.proves_recall(empty_advice)
        incomplete = LL.query_approx_msm_knn(advisory, [1.0], 1;
            limits=LL.TemporalSearchLimits(max_dp_cells=0))
        @test incomplete.kind === :incomplete
        @test incomplete.reason === :dp_cells
        @test incomplete.indexed_entries == 3
        @test !LL.proves_recall(incomplete)
        @test isempty(incomplete)
        @test_throws ArgumentError LL.query_approx_msm_knn(
            advisory, [NaN], 1)
        @test_throws ArgumentError LL.query_approx_msm_knn(
            advisory, [1.0, 2.0], 1;
            limits=LL.TemporalSearchLimits(max_series_len=1))
    finally
        close(advisory)
    end

    empty_index = LL.ApproxMsmIndex(Pair{UInt64,Vector{Float64}}[];
        max_entries=0, max_total_samples=0, max_total_features=0)
    try
        empty_result = LL.query_approx_msm_knn(empty_index, Float64[], 1)
        @test empty_result.kind === :exhaustive
        @test LL.proves_recall(empty_result)
        @test isempty(empty_result)
    finally
        close(empty_index)
    end
    @test_throws ArgumentError LL.ApproxMsmIndex([7 => [1.0]];
        split_merge_cost=-1.0)
    @test_throws ArgumentError LL.ApproxMsmIndex([7 => [1.0]];
        segments=2, max_total_features=1)
end
