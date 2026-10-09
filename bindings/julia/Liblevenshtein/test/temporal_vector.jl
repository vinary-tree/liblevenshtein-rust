@testset "typed native vector temporal metrics" begin
    channels = [
        VectorChannel("position-x", "metre"; scale=2.0, weight=3.0),
        VectorChannel("position-y", "metre"; scale=4.0, weight=5.0),
    ]
    metric = FixedChannelMetric(channels;
        training_fold="fold-1", estimator_revision="scale-v1")
    left = VectorTemporalSeries([0.0 2.0 4.0; 1.0 3.0 1.0])
    same = VectorTemporalSeries(left.samples)
    right = VectorTemporalSeries([0.0 2.0 3.0; 1.0 2.0 1.0])
    timed_left = VectorTemporalSeries(left.samples;
        timestamps=[1.0, 2.0, 3.0], unit=:seconds)
    timed_right = VectorTemporalSeries(right.samples;
        timestamps=[1.0, 2.0, 3.0], unit=:seconds)
    timed_same = VectorTemporalSeries(left.samples;
        timestamps=[1.0, 2.0, 3.0], unit=:seconds)

    scorers = (
        (a, b) -> vector_erp_distance(metric, a, b; gap=[0.0, 0.0]),
        (a, b) -> vector_dtw_distance(metric, a, b; band=1),
        (a, b) -> vector_frechet_distance(metric, a, b),
    )
    for score in scorers
        identical = score(left, same)
        changed = score(left, right)
        @test identical.kind == :finite
        @test identical.value == 0.0
        @test changed.kind == :finite
        @test changed.value > 0
        @test changed.work_units > 0
    end
    identical_twed = vector_timestamped_twed_distance(metric,
        timed_left, timed_same; sentinel=[0.0, 0.0], stiffness=0.5,
        gap_penalty=1.0)
    changed_twed = vector_timestamped_twed_distance(metric,
        timed_left, timed_right; sentinel=[0.0, 0.0], stiffness=0.5,
        gap_penalty=1.0)
    @test identical_twed.kind == :finite
    @test identical_twed.value == 0.0
    @test changed_twed.kind == :finite
    @test changed_twed.value > 0

    limited = VectorTemporalLimits(scalar=TemporalLimits(max_work_units=0))
    @test vector_frechet_distance(metric, left, right;
        limits=limited).kind == :incomplete
    @test_throws ArgumentError vector_erp_distance(metric, left, right;
        gap=[0.0])
    @test_throws ArgumentError vector_timestamped_twed_distance(metric,
        left, right; sentinel=[0.0, 0.0], stiffness=1.0)
    @test_throws ArgumentError vector_frechet_distance(metric, left,
        VectorTemporalSeries(reshape([1.0, 2.0, 3.0], 1, 3)))
    @test_throws ArgumentError vector_temporal_distance(:msm, metric,
        left, right)
    @test_throws ArgumentError VectorTemporalSeries(left.samples;
        max_input_bytes=8)
    @test_throws Liblevenshtein.NativeError FixedChannelMetric([
        VectorChannel("x", "m"), VectorChannel("x", "m")];
        training_fold="fold", estimator_revision="v1")
    close(metric)
    @test !isopen(metric)
    @test_throws ArgumentError vector_frechet_distance(metric, left, right)
end

@testset "audited L1 L2 Linf vector Fréchet" begin
    query = VectorTemporalSeries([0.0 1.0; 0.0 1.0])
    target = VectorTemporalSeries([0.0 2.0; 0.0 4.0])
    for (ground, expected) in ((:l1, 4.0), (:l2, hypot(1.0, 3.0)),
        (:linf, 3.0))
        outcome = vector_frechet_ground_distance(ground, query, target;
            cutoff=10.0)
        @test outcome.kind == :finite
        @test outcome.value == expected
        @test outcome.work_units > 0
        @test vector_frechet_ground_distance(ground, query, query).value == 0
    end
    @test vector_frechet_ground_distance(:l1, query, target;
        limits=VectorTemporalLimits(
            scalar=TemporalLimits(max_work_units=0))).kind == :incomplete
    @test_throws ArgumentError vector_frechet_ground_distance(:other,
        query, target)
    @test_throws ArgumentError vector_frechet_ground_distance(:l1,
        query, VectorTemporalSeries(reshape([1.0, 2.0], 1, 2)))
end
