@testset "native bounded temporal lower bounds" begin
    left = [1.0, 2.0, 3.0]
    right = [1.0, 2.5, 3.0]
    @test LL.erp_gap_mass_lower_bound(left, right, 0.0).value ≈ 0.5
    @test LL.frechet_endpoint_lower_bound(left, right).value == 0.0
    @test LL.frechet_one_sided_hausdorff_lower_bound(left, right).value ≈ 0.5
    @test LL.frechet_candidate_lower_bound(left, right).value ≈ 0.5
    @test LL.lb_keogh(left, right, 1).value == 0.0
    @test LL.twed_length_lower_bound(3, 5, 0.7).value ≈ 1.4

    plan = LL.keogh_envelopes(left, 1)
    @test length(plan) == 3
    @test LL.bounds_at(plan, 1) == (1.0, 2.0)
    @test LL.bounds_at(plan, 4) == (3.0, 3.0)
    @test LL.bounds_at(plan, 10) === nothing
    @test LL.lb_keogh([1.0, 4.0, 3.0], plan).value ≈ 1.0
    @test LL.lb_keogh_squared([1.0, 4.0, 3.0], plan).value ≈ 1.0
    tasks = [@async LL.lb_keogh(right, plan).value for _ in 1:2]
    @test fetch(tasks[1]) == fetch(tasks[2])
    close(plan)
    @test !isopen(plan)
    @test_throws ArgumentError LL.lb_keogh(right, plan)

    @test LL.frechet_one_sided_hausdorff_lower_bound(
        Float64[], [1.0]).kind === :finite
    @test LL.frechet_endpoint_lower_bound(
        Float64[], [1.0]).kind === :no_alignment
    @test LL.frechet_endpoint_lower_bound(
        [floatmax(Float64)], [-floatmax(Float64)]).kind === :incomplete

    no_work = LL.TemporalLimits(max_work_units=0)
    work = LL.erp_gap_mass_lower_bound([1.0], [1.0], 0.0;
        limits=no_work)
    @test work.kind === :incomplete
    @test work.reason === :work_units
    no_scratch = LL.TemporalLimits(max_scratch_bytes=0)
    scratch = LL.lb_keogh([1.0, 2.0], [1.0, 2.0], 1;
        limits=no_scratch)
    @test scratch.kind === :incomplete
    @test scratch.reason === :scratch_bytes
    @test_throws LL.NativeError LL.keogh_envelopes(Float64[], 1)
    @test_throws ArgumentError LL.temporal_lower_bound(
        :unknown, left, right)
    short = LL.TemporalLimits(max_series_len=1)
    @test_throws ArgumentError LL.temporal_lower_bound(
        :keogh, left, right; band=1, limits=short)
    @test_throws ArgumentError LL.keogh_envelopes(left, 1; limits=short)
end

@testset "native MSM bounds and explicitly unsafe heuristics" begin
    left = [0.0, 100.0]
    right = [0.0, 0.0, 100.0]
    exact = LL.msm_distance(left, right; split_merge_cost=1.0).value
    @test exact == 1.0
    @test LL.msm_length_lower_bound(left, right, 1.0).value == 1.0
    @test LL.msm_prefix_euclidean_heuristic(left, right).value == 100.0
    @test LL.msm_prefix_l1_heuristic(left, right).value == 100.0
    @test LL.msm_combined_heuristic(left, right, 1.0).value == 100.0
    @test LL.msm_length_lower_bound(left, right, 1.0).value <= exact
    @test LL.msm_combined_heuristic(left, right, 1.0).value > exact
    @test LL.msm_length_lower_bound(Float64[], right, 1.0).kind ===
        :no_alignment
    @test LL.msm_length_lower_bound(Float64[], Float64[], 1.0).value == 0.0
    @test LL.msm_length_lower_bound(left, right, 1.0;
        limits=LL.TemporalLimits(max_work_units=0)).kind === :incomplete
    @test_throws LL.NativeError LL.msm_length_lower_bound(left, right, -1.0)
end
