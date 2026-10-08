@testset "bounded rolling temporal windows" begin
    machine = LL.BoundedRollingWindow(3, 2)
    retained = machine.scratch_bytes
    observed = Tuple[]
    for sample in 0:8
        step = LL.advance!(machine, sample)
        @test step.kind === :advanced
        @test step.usage.scratch_bytes == retained
        @test step.usage.work_units == 1
        if step.snapshot !== nothing
            window = step.snapshot
            push!(observed, (window.window_id, window.start_offset,
                window.end_offset, window.values))
            @test step.usage.snapshot_bytes == retained
        else
            @test step.usage.snapshot_bytes == 0
        end
    end
    @test observed == [
        (0, 0, 3, [0.0, 1.0, 2.0]),
        (1, 2, 5, [2.0, 3.0, 4.0]),
        (2, 4, 7, [4.0, 5.0, 6.0]),
        (3, 6, 9, [6.0, 7.0, 8.0]),
    ]
    close(machine)
    @test !isopen(machine)
    @test_throws ArgumentError LL.advance!(machine, 1.0)

    machine = LL.BoundedRollingWindow(4, 10_000)
    @test_throws ArgumentError LL.advance!(machine, NaN)
    @test machine.consumed == 0
    for index in 0:9_999
        LL.advance!(machine, index % 7)
    end
    @test machine.consumed == 10_000
    @test machine.scratch_bytes == 4 * sizeof(Float64)
    close(machine)

    stream = LL.rolling_windows(0.0:8.0, 3, 2)
    @test [(window.start_offset, window.values) for window in stream] ==
        [(0, [0.0, 1.0, 2.0]), (2, [2.0, 3.0, 4.0]),
            (4, [4.0, 5.0, 6.0]), (6, [6.0, 7.0, 8.0])]
    @test !isopen(stream)
    reduced = LL.rolling_windows(0.0:8.0, 3, 2)
    @test LL.reduce_windows!((sum, window) ->
        sum + window.window_id, UInt64(0), reduced) == 6
    @test !isopen(reduced)
    stopped = LL.rolling_windows(0.0:8.0, 3, 2)
    @test iterate(stopped) !== nothing
    LL.cancel!(stopped)
    @test iterate(stopped) === nothing

    index = LL.TemporalIndex(:msm; quant_min=-10.0,
        quant_max=10.0, parameter0=1.0, max_entries=2,
        max_total_samples=6, max_series_len=3)
    LL.insert!(index, 0, [0.0, 1.0, 2.0])
    LL.insert!(index, 1, [5.0, 5.0, 5.0])
    LL.freeze!(index)
    source = LL.rolling_windows([0.0, 1.0, 2.0], 3, 1)
    snapshot = only(collect(source))
    values = snapshot.values
    values[1] = 100.0
    @test snapshot.values == [0.0, 1.0, 2.0]
    cursor = LL.query_index_range(index, snapshot; cutoff=0.0,
        page_work_units=1_000, page_results=1)
    close(index)
    @test [(match.id, match.distance) for match in cursor] ==
        [(UInt64(0), 0.0)]

    @test_throws ArgumentError LL.BoundedRollingWindow(0, 1)
    @test_throws ArgumentError LL.BoundedRollingWindow(1, 0)
    @test_throws ArgumentError LL.BoundedRollingWindow(1, 1;
        max_snapshot_bytes=7)
    @test_throws ArgumentError LL.BoundedRollingWindow(2, 1;
        max_series_len=1)
    @test_throws ArgumentError collect(LL.rolling_windows(
        [1.0, NaN, 2.0], 2, 1))
end
