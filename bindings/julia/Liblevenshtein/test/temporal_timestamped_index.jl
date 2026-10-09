@testset "native physical-time TWED range index" begin
    config = LL.MetricTimestampedTwedConfig(0.5, 1.0)
    exact = LL.TimestampedSeries([1.0, 2.0], [10.0, 13.0];
        unit=:milliseconds, origin=10.0)
    distant = LL.TimestampedSeries([3.0, 4.0], [10.0, 14.0];
        unit=:milliseconds, origin=10.0)
    index = LL.TimestampedTwedIndex(config;
        unit=:milliseconds, origin=10.0,
        value_min=0.0, value_max=5.0,
        time_min=10.0, time_max=20.0,
        value_bins=1, time_bins=1,
        max_entries=3, max_total_samples=6, max_series_len=2)
    try
        @test LL.insert_episode!(index, 7, exact) == 0
        @test LL.insert_episode!(index, 7, exact) == 1
        @test LL.insert_episode!(index, 11, distant) == 2
        @test_throws LL.NativeError LL.insert_episode!(index, 13, exact)
        @test_throws ArgumentError LL.insert_episode!(index, 7,
            LL.TimestampedSeries([1.0], [10.0];
                unit=:seconds, origin=10.0))
        LL.freeze!(index)
        @test_throws ArgumentError LL.insert_episode!(index, 13, exact)
        @test LL.freeze!(index) === index
        first = LL.query_metric_range(index, exact;
            cutoff=0.0, page_work_units=100_000, page_results=1)
        second = LL.query_metric_range(index, exact;
            cutoff=0.0, page_work_units=100_000, page_results=1)
        tasks = [Threads.@spawn begin
            independent = LL.query_metric_range(index, exact;
                cutoff=0.0, page_work_units=100_000, page_results=1)
            sort([(match.episode_id, match.id, match.distance)
                for match in independent])
        end for _ in 1:2]
        expected = [(UInt64(0), UInt64(7), 0.0),
            (UInt64(1), UInt64(7), 0.0)]
        @test all(task -> fetch(task) == expected, tasks)
        @test_throws ArgumentError LL.query_metric_range(index,
            LL.TimestampedSeries([1.0], [11.0];
                unit=:milliseconds, origin=11.0);
            cutoff=0.0)
        @test_throws ArgumentError LL.query_metric_range(index, exact;
            limits=LL.TimestampedTwedSearchLimits(
                common=LL.TemporalSearchLimits(max_series_len=1)))
        close(index)
        @test !isopen(index)
        @test sort([(match.episode_id, match.id, match.distance)
            for match in first]) == expected
        @test LL.reduce_batches!((all, batch) ->
            append!(all, [match.episode_id for match in batch]),
            UInt64[], second; batch_size=1) == UInt64[0, 1]
        @test !isopen(first) && !isopen(second)
    finally
        close(index)
    end

    bounded = LL.TimestampedTwedIndex(config;
        unit=:milliseconds, origin=10.0,
        value_min=0.0, value_max=5.0,
        time_min=10.0, time_max=20.0,
        max_entries=1, max_total_samples=2, max_series_len=2)
    try
        LL.insert!(bounded, 9, exact)
        LL.freeze!(bounded)
        limited = LL.query_metric_range(bounded, exact; cutoff=0.0,
            limits=LL.TimestampedTwedSearchLimits(
                common=LL.TemporalSearchLimits(max_results=0)))
        @test_throws LL.TemporalQueryIncomplete collect(limited)
        @test !isopen(limited)
        paused = LL.query_metric_range(bounded, exact; cutoff=0.0,
            limits=LL.TimestampedTwedSearchLimits(max_product_states=0))
        @test_throws LL.TemporalQueryIncomplete collect(paused)
        @test !isopen(paused)
        stopped = LL.query_metric_range(bounded, exact; cutoff=0.0)
        LL.cancel!(stopped)
        @test !isopen(stopped)
    finally
        close(bounded)
    end

    @test_throws ArgumentError LL.TimestampedTwedIndex(config;
        value_min=0.0, value_max=5.0, time_min=0.0, time_max=5.0,
        value_bins=0)
    @test_throws LL.NativeError LL.TimestampedTwedIndex(config;
        value_min=0.0, value_max=0.0, time_min=0.0, time_max=5.0)
end
