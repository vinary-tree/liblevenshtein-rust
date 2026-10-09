@testset "lazy native MSM source prefilter" begin
    original = [0.0, 100.0]
    source = LL.TemporalSeriesSource([
        3 => [0.0, 0.0, 0.0, 0.0, 0.0],
        1 => original,
        2 => [0.0, 0.0, 100.0],
    ])
    original[1] = 50.0
    query = [0.0, 100.0]
    safe = LL.filter_msm_source(source, query; threshold=1.0)
    kept = collect(safe)
    @test [(entry.id, entry.score) for entry in kept] ==
        [(UInt64(1), 0.0), (UInt64(2), 1.0)]
    @test kept[1].samples == query
    @test !isopen(safe)
    kept[1].samples[1] = 99.0
    replay = LL.filter_msm_source(source, query; threshold=0.0)
    @test only(LL.next_batch!(replay, 1)).samples == query
    close(replay)

    @test_throws ArgumentError LL.filter_msm_source(source, query;
        mode=:euclidean, threshold=1.0)
    approximate = LL.filter_msm_source(source, query;
        mode=:euclidean, threshold=1.0,
        allow_false_negatives=true)
    @test [entry.id for entry in approximate] == UInt64[1]
    @test_throws ArgumentError LL.filter_msm_source(source, query;
        threshold=-1.0)
    @test_throws ArgumentError LL.filter_msm_source(source, query;
        split_merge_cost=-1.0)

    limited_work = LL.filter_msm_source(source, query;
        threshold=1.0, query_limits=LL.MsmPrefilterLimits(
            max_total_work_units=2))
    @test only(LL.next_batch!(limited_work, 1)).id == UInt64(1)
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(limited_work, 1)
    @test !isopen(limited_work)

    limited_results = LL.filter_msm_source(source, query;
        threshold=1.0, query_limits=LL.MsmPrefilterLimits(max_results=1))
    @test only(LL.next_batch!(limited_results, 1)).id == UInt64(1)
    @test_throws LL.TemporalQueryIncomplete LL.next_batch!(limited_results, 1)
    @test !isopen(limited_results)

    left = LL.filter_msm_source(source, query; threshold=1.0)
    right = LL.filter_msm_source(source, query; threshold=1.0)
    tasks = [Base.Threads.@spawn [entry.id for entry in cursor]
        for cursor in (left, right)]
    @test fetch(tasks[1]) == fetch(tasks[2]) == UInt64[1, 2]
    reduced = LL.reduce_batches!((ids, batch) ->
        append!(ids, (entry.id for entry in batch)), UInt64[],
        LL.filter_msm_source(source, query; threshold=1.0);
        batch_size=1)
    @test reduced == UInt64[1, 2]

    racing = LL.filter_msm_source(source, query; threshold=1.0)
    gate = Channel{Nothing}(2)
    reading = Base.Threads.@spawn begin
        take!(gate)
        LL.next_batch!(racing, 1)
    end
    closing = Base.Threads.@spawn begin
        take!(gate)
        close(racing)
    end
    put!(gate, nothing)
    put!(gate, nothing)
    raced = fetch(reading)
    fetch(closing)
    @test raced === nothing ||
        (raced isa Vector{LL.MsmPrefilterCandidate} &&
            length(raced) == 1 && raced[1].id == UInt64(1))
    @test !isopen(racing)
end
