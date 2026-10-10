@testset "physical-time timestamped TWED" begin
    config = LL.MetricTimestampedTwedConfig(0.5, 1.0)
    raw_values = [1.0, 2.0]
    raw_times = [10.0, 13.0]
    left = LL.TimestampedSeries(raw_values, raw_times;
        unit=:milliseconds, origin=10.0)
    same = LL.TimestampedSeries([1.0, 2.0], [10.0, 13.0];
        unit=:milliseconds, origin=10.0)
    moved = LL.TimestampedSeries([1.0, 2.5], [10.0, 14.0];
        unit=:milliseconds, origin=10.0)
    @test LL.metric_timestamped_twed_distance(config, left, same).value == 0.0
    comparison = LL.metric_timestamped_twed_distance(config, left, moved)
    @test comparison.kind === :finite
    @test comparison.value > 0.0
    @test comparison.work_units > 0
    @test LL.metric_timestamped_twed_distance(config, left, moved;
        cutoff=0.0).kind === :above_cutoff
    @test LL.metric_timestamped_twed_distance(config, left, moved;
        limits=LL.TemporalLimits(max_dp_cells=1)).reason === :dp_cells
    raw_values[1] = 99.0
    raw_times[2] = 99.0
    visible = left.values
    visible[1] = 88.0
    @test left.values == [1.0, 2.0]
    @test left.timestamps == [10.0, 13.0]
    @test left.unit === :milliseconds
    @test left.origin == 10.0
    @test_throws ArgumentError left._values
    @test_throws ArgumentError LL.TimestampedSeries([1.0], [0.0, 1.0])
    @test_throws ArgumentError LL.TimestampedSeries(Float64[], Float64[])
    @test_throws ArgumentError LL.TimestampedSeries([1.0], [0.0]; unit=:fortnights)
    @test_throws ArgumentError LL.TimestampedSeries([NaN], [0.0])
    @test_throws ArgumentError LL.TimestampedSeries([1.0], [Inf])
    @test_throws ArgumentError LL.TimestampedSeries([1.0], [-1.0])
    @test_throws ArgumentError LL.TimestampedSeries([1.0, 2.0], [0.0, 0.0])
    @test_throws ArgumentError LL.TimestampedSeries(
        UntouchableLargeVector(4), UntouchableLargeVector(4);
        max_series_len=3)
    @test_throws ArgumentError LL.MetricTimestampedTwedConfig(0.0, 1.0)
    @test_throws ArgumentError LL.MetricTimestampedTwedConfig(1.0, -1.0)
    @test_throws ArgumentError LL.metric_timestamped_twed_distance(config,
        left, LL.TimestampedSeries([1.0], [10.0];
            unit=:seconds, origin=10.0))
    @test_throws ArgumentError LL.metric_timestamped_twed_distance(config,
        left, LL.TimestampedSeries([1.0], [11.0];
            unit=:milliseconds, origin=11.0))
    @test_throws ArgumentError LL.metric_timestamped_twed_distance(config,
        left, moved; cutoff=-1.0)
end
