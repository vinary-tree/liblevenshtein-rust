@testset "native online vector Fréchet" begin
    metric = FixedChannelMetric([
        VectorChannel("x", "metre"; scale=2.0, weight=3.0),
        VectorChannel("y", "metre"; scale=4.0, weight=5.0),
    ]; training_fold="fold-1", estimator_revision="scale-v1")
    query = VectorTemporalSeries([0.0 1.0; 0.0 1.0])
    target = VectorTemporalSeries([0.0 1.0 1.0; 0.0 1.0 2.0])
    exact_prefixes = [vector_frechet_distance(metric, query,
        VectorTemporalSeries(target.samples[:, 1:count]); cutoff=10.0).value
        for count in 1:size(target.samples, 2)]
    machine = VectorFrechetOnlineAutomaton(metric, query; cutoff=10.0)
    limited = VectorFrechetOnlineAutomaton(metric, query; cutoff=10.0,
        limits=TemporalOnlineLimits(max_step_work_units=0))
    retained = Liblevenshtein.scratch_bytes(machine)
    @test retained > 0
    @test Liblevenshtein.observation(machine).consumed_target_len == 0
    close(metric)
    for (count, point) in enumerate(eachcol(target.samples))
        step = Liblevenshtein.advance!(machine, point)
        @test step.kind == :advanced
        @test step.observation.consumed_target_len == count
        @test step.observation.distance_within_cutoff == exact_prefixes[count]
        @test Liblevenshtein.scratch_bytes(machine) == retained
    end
    @test_throws ArgumentError Liblevenshtein.advance!(machine, [1.0])
    @test Liblevenshtein.observation(machine).consumed_target_len == 3
    incomplete = Liblevenshtein.advance!(limited, [0.0, 0.0])
    @test incomplete.kind == :incomplete
    @test incomplete.reason == :work_units
    @test Liblevenshtein.observation(limited).consumed_target_len == 0
    close(machine)
    close(limited)
    @test !isopen(machine)
    @test_throws ArgumentError Liblevenshtein.observation(machine)

    stream_metric = FixedChannelMetric([
        VectorChannel("x", "metre"), VectorChannel("y", "metre"),
    ]; training_fold="fold-1", estimator_revision="scale-v1")
    source = eachcol(target.samples)
    stream = vector_frechet_online_observations(
        stream_metric, query, source; cutoff=10.0)
    prefixes = Liblevenshtein.reduce_observations!(
        (values, observed) -> (push!(values, observed); values),
        TemporalOnlineObservation[], stream)
    @test length(prefixes) == size(target.samples, 2)
    @test !isopen(stream)
    close(stream_metric)
end

@testset "audited ground-metric online vector Fréchet" begin
    query = VectorTemporalSeries([0.0 1.0; 0.0 1.0])
    target = VectorTemporalSeries([0.0 2.0; 0.0 4.0])
    for ground in (:l1, :l2, :linf)
        expected = [vector_frechet_ground_distance(ground, query,
            VectorTemporalSeries(target.samples[:, 1:count]);
            cutoff=10.0).value for count in 1:2]
        machine = VectorFrechetOnlineAutomaton(ground, query; cutoff=10.0)
        retained = Liblevenshtein.scratch_bytes(machine)
        try
            for (count, point) in enumerate(eachcol(target.samples))
                step = Liblevenshtein.advance!(machine, point)
                @test step.kind == :advanced
                @test step.observation.distance_within_cutoff == expected[count]
                @test Liblevenshtein.scratch_bytes(machine) == retained
            end
        finally
            close(machine)
        end
        stream = vector_frechet_online_observations(ground, query,
            eachcol(target.samples); cutoff=10.0)
        @test [observation.distance_within_cutoff for observation in stream] ==
            expected
        @test !isopen(stream)
    end
    @test_throws ArgumentError VectorFrechetOnlineAutomaton(:other,
        query; cutoff=10.0)
    limited = VectorFrechetOnlineAutomaton(:l1, query; cutoff=10.0,
        limits=TemporalOnlineLimits(max_step_work_units=0))
    try
        @test Liblevenshtein.advance!(limited, [0.0, 0.0]).kind ==
            :incomplete
        @test Liblevenshtein.observation(limited).consumed_target_len == 0
    finally
        close(limited)
    end
end
