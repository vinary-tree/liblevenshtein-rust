@testset "bounded native temporal scores" begin
    @test sizeof(LL.RawTemporalConfig) == 40
    @test sizeof(LL.TemporalLimits) == 32
    @test sizeof(LL.RawTemporalDistanceResult) == 40

    cases = [
        LL.msm_distance([1.0, 2.0], [1.0, 2.5]),
        LL.erp_distance([1.0, 2.0], [1.0, 2.5]),
        LL.twed_distance([1.0, 2.0], [1.0, 2.5]),
        LL.dtw_distance([1.0, 2.0], [1.0, 2.5]; band=1),
        LL.frechet_distance([1.0, 2.0], [1.0, 2.5]),
    ]
    @test all(result -> result.kind == :finite, cases)
    @test all(result -> result.value isa Float64, cases)
    @test LL.msm_distance([1.0, 2.0], [1.0, 2.5]).value ≈ 0.5
    @test LL.erp_distance([1.0], [2.0]).value ≈ 1.0
    @test LL.dtw_distance([1.0, 2.0], [1.0, 2.5]; band=1).value ≈ 0.5
    @test LL.frechet_distance([1.0, 2.0], [1.0, 2.5]).value ≈ 0.5
    @test LL.soft_dtw_loss([1.0], [1.0]).value ≈ 0.0
    @test LL.soft_dtw_loss([0.0, 0.0], [0.0, 0.0]; cutoff=-0.5).value < -0.5

    @test LL.erp_distance([0.0], [3.0]; cutoff=0.0).kind == :above_cutoff
    @test LL.frechet_distance(Float64[], [1.0]).kind == :no_alignment
    @test LL.dtw_distance([1.0], [1.0, 2.0]; band=0).kind == :no_alignment

    small = LL.TemporalLimits(max_dp_cells=1)
    incomplete = LL.msm_distance([1.0, 2.0], [1.0, 2.0]; limits=small)
    @test incomplete.kind == :incomplete
    @test incomplete.reason == :dp_cells
    @test incomplete.value === nothing

    @test_throws ArgumentError LL.temporal_distance(:unknown, [1.0], [1.0])
    @test_throws ArgumentError LL.TemporalLimits(max_series_len=-1)
    @test_throws LL.NativeError LL.twed_distance([1.0], [2.0]; stiffness=-1.0)
    @test_throws LL.NativeError LL.msm_distance([NaN], [1.0])
end

@testset "bounded Soft-DTW loss and gradients" begin
    x = [0.0, 1.0, 2.0]
    y = [0.5, 1.5]
    gamma = 0.75
    analysis = LL.soft_dtw_gradient(x, y; gamma)
    @test analysis.kind === :finite
    @test analysis.reason === nothing
    @test analysis.value ≈ LL.soft_dtw_loss(x, y; gamma).value
    @test length(analysis.left_gradient) == length(x)
    @test length(analysis.right_gradient) == length(y)
    @test analysis.dp_cells == length(x) * length(y)
    @test analysis.work_units == 2 * length(x) * length(y)
    @test analysis.scratch_bytes > 0

    epsilon = 1e-5
    for index in eachindex(x)
        plus = copy(x)
        minus = copy(x)
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (LL.soft_dtw_loss(plus, y; gamma).value -
            LL.soft_dtw_loss(minus, y; gamma).value) / (2epsilon)
        @test analysis.left_gradient[index] ≈ numerical atol=1e-5
    end
    for index in eachindex(y)
        plus = copy(y)
        minus = copy(y)
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (LL.soft_dtw_loss(x, plus; gamma).value -
            LL.soft_dtw_loss(x, minus; gamma).value) / (2epsilon)
        @test analysis.right_gradient[index] ≈ numerical atol=1e-5
    end
    left_copy = copy(analysis.left_gradient)
    right_copy = copy(analysis.right_gradient)
    x[1] = 100.0
    y[1] = 100.0
    @test analysis.left_gradient == left_copy
    @test analysis.right_gradient == right_copy

    incomplete = LL.soft_dtw_gradient([0.0, 1.0], [0.0, 1.0];
        limits=LL.TemporalLimits(max_dp_cells=1))
    @test incomplete.kind === :incomplete
    @test incomplete.reason === :dp_cells
    @test incomplete.value === nothing
    @test incomplete.left_gradient === nothing
    @test incomplete.right_gradient === nothing
    @test_throws ArgumentError LL.soft_dtw_gradient(Float64[], [1.0])
    @test_throws ArgumentError LL.soft_dtw_gradient(
        UntouchableLargeVector(4), [1.0];
        limits=LL.TemporalLimits(max_series_len=3))
    @test_throws LL.NativeError LL.soft_dtw_gradient([NaN], [1.0])
    @test_throws LL.NativeError LL.soft_dtw_gradient([1.0], [1.0];
        gamma=0.0)
end
