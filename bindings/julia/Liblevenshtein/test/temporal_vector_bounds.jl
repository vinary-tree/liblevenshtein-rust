@testset "typed native vector temporal bounds" begin
    metric = FixedChannelMetric([
        VectorChannel("x", "metre"; scale=2.0, weight=3.0),
        VectorChannel("y", "metre"; scale=4.0, weight=5.0),
    ]; training_fold="fold-1", estimator_revision="scale-v1")
    left = VectorIntervalBox(metric, [(-1.0, 1.0), (2.0, 3.0)])
    right = VectorIntervalBox(metric, [(3.0, 4.0), (1.0, 2.0)])
    wide = VectorIntervalBox(metric, [(-2.0, 5.0), (0.0, 5.0)])
    @test vector_box_refines(left, wide)
    @test !vector_box_refines(wide, left)
    mutated = VectorIntervalBox(metric, [(-1.0, 1.0), (2.0, 3.0)])
    pop!(mutated.intervals)
    @test_throws ArgumentError vector_box_refines(mutated, wide)
    @test vector_point_box_lower_bound(metric, [0.0, 0.0], left) == 2.5
    @test vector_box_box_lower_bound(metric, left, right) == 3.0
    @test vector_erp_interval_match_lower_bound(metric,
        [0.0, 0.0], left) == 2.5
    @test vector_erp_interval_gap_lower_bound(metric,
        [0.0, 0.0], left) == 2.5
    @test vector_dtw_interval_local_lower_bound_squared(metric,
        [0.0, 0.0], left) == 6.25
    @test vector_frechet_interval_link_lower_bound(metric,
        [0.0, 0.0], left) == 2.5
    @test_throws ArgumentError VectorIntervalBox(metric,
        [(1.0, 0.0), (0.0, 1.0)])

    x = VectorTemporalSeries([1.0 2.0; 2.0 3.0])
    y = VectorTemporalSeries([1.0 3.0; 1.0 3.0])
    @test vector_erp_candidate_lower_bound(metric, x, y;
        gap=[0.0, 0.0]) ≈ 0.25
    @test vector_frechet_candidate_lower_bound(metric, x, y) == 1.5
    @test vector_dtw_candidate_lower_bound(metric, x, y; band=1) == 0.0
    tx = VectorTemporalSeries(x.samples; timestamps=[1.0, 2.0])
    ty = VectorTemporalSeries(y.samples; timestamps=[1.0, 2.0])
    @test vector_timestamped_twed_candidate_lower_bound(metric, tx, ty;
        sentinel=[0.0, 0.0], stiffness=0.5, gap_penalty=1.0) == 0.0

    previous = TimestampedVectorIntervalBox(VectorIntervalBox(metric,
        [(0.0, 1.0), (1.0, 2.0)]), 0.5, 1.0)
    current = TimestampedVectorIntervalBox(VectorIntervalBox(metric,
        [(2.0, 3.0), (3.0, 4.0)]), 2.0, 2.5)
    broad = TimestampedVectorIntervalBox(wide, 0.0, 3.0)
    @test timestamped_vector_box_refines(current, broad)
    @test vector_twed_interval_delete_lower_bound(metric,
        current, previous; sentinel=[0.0, 0.0], stiffness=0.5,
        gap_penalty=1.0) > 0
    @test vector_twed_interval_match_lower_bound(metric,
        [2.0, 3.0], [0.0, 1.0], 2.0, 1.0,
        current, previous; sentinel=[0.0, 0.0],
        stiffness=0.5, gap_penalty=1.0) >= 0
    wrong_unit = TimestampedVectorIntervalBox(wide, 0.0, 3.0;
        unit=:milliseconds)
    @test_throws ArgumentError timestamped_vector_box_refines(
        current, wrong_unit)
    low_storage = VectorTemporalLimits(
        scalar=TemporalLimits(max_scratch_bytes=1))
    @test_throws Liblevenshtein.NativeError vector_box_box_lower_bound(
        metric, left, right; limits=low_storage)
    close(metric)
    @test_throws ArgumentError vector_point_box_lower_bound(
        metric, [0.0, 0.0], left)
end
