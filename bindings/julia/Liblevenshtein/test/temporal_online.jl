@testset "native bounded online temporal automata" begin
    @test sizeof(LL.RawTemporalOnlineObservation) == 40
    @test sizeof(LL.RawTemporalOnlineStep) == 80
    query = [1.0, 2.0, 3.0]
    target = [1.0, 2.5, 3.0]
    for (kind, parameter0, parameter1, band) in (
        (:msm, 1.0, 0.0, 0),
        (:erp, 0.0, 0.0, 0),
        (:twed, 1.0, 0.0, 0),
        (:dtw, 0.0, 0.0, 2),
        (:frechet, 0.0, 0.0, 0),
    )
        machine = LL.TemporalOnlineAutomaton(kind, query;
            parameter0, parameter1, band, cutoff=100.0)
        try
            @test LL.observation(machine).consumed_target_len == 0
            retained = LL.scratch_bytes(machine)
            @test retained > 0
            for end_index in eachindex(target)
                step = LL.advance!(machine, target[end_index])
                @test step.kind === :advanced
                @test step.reason === nothing
                @test step.observation.consumed_target_len == end_index
                @test LL.scratch_bytes(machine) == retained
                expected = LL.temporal_distance(kind, query,
                    target[1:end_index]; parameter0, parameter1,
                    band, cutoff=100.0)
                if expected.kind === :finite
                    value = kind === :dtw ? expected.value^2 :
                        expected.value
                    @test step.observation.distance_within_cutoff ≈ value
                else
                    @test step.observation.distance_within_cutoff === nothing
                end
            end
        finally
            close(machine)
        end
        @test !isopen(machine)
        @test_throws ArgumentError LL.observation(machine)
    end

    no_work = LL.TemporalOnlineLimits(max_step_work_units=0)
    stopped = LL.TemporalOnlineAutomaton(:msm, query;
        parameter0=1.0, cutoff=10.0, limits=no_work)
    step = LL.advance!(stopped, 1.0)
    @test step.kind === :incomplete
    @test step.reason === :work_units
    @test step.observation === nothing
    @test LL.observation(stopped).consumed_target_len == 0
    @test_throws LL.NativeError LL.advance!(stopped, NaN)
    @test LL.observation(stopped).consumed_target_len == 0
    close(stopped)

    @test_throws ArgumentError LL.TemporalOnlineAutomaton(:msm,
        UntouchableLargeVector(4); cutoff=10.0,
        limits=LL.TemporalOnlineLimits(max_query_len=3))
    @test_throws LL.NativeError LL.TemporalOnlineAutomaton(:msm,
        query; parameter0=1.0, cutoff=Inf)
    @test_throws LL.NativeError LL.TemporalOnlineAutomaton(:soft_dtw,
        query; parameter0=1.0, cutoff=10.0)
    @test_throws LL.NativeError LL.TemporalOnlineAutomaton(:msm,
        query; parameter0=1.0, cutoff=10.0,
        limits=LL.TemporalOnlineLimits(max_scratch_bytes=0))
    erp_unbounded = LL.TemporalOnlineAutomaton(:erp,
        query; cutoff=Inf)
    close(erp_unbounded)

    stream = LL.online_observations(:erp, query, target;
        cutoff=100.0)
    @test [observed.consumed_target_len for observed in stream] ==
        [1, 2, 3]
    @test !isopen(stream)
    reduced = LL.online_observations(:msm, query, target;
        parameter0=1.0, cutoff=100.0)
    @test LL.reduce_observations!((sum, observed) ->
        sum + observed.consumed_target_len, 0, reduced) == 6
    @test !isopen(reduced)
    failing = LL.online_observations(:msm, query, target;
        parameter0=1.0, cutoff=10.0, limits=no_work)
    @test_throws LL.TemporalOnlineIncomplete collect(failing)
    @test !isopen(failing)

    independent = [LL.TemporalOnlineAutomaton(:erp, query;
        cutoff=100.0) for _ in 1:2]
    tasks = [@async [LL.advance!(machine, sample).observation
        for sample in target] for machine in independent]
    @test fetch(tasks[1]) == fetch(tasks[2])
    foreach(close, independent)
end
