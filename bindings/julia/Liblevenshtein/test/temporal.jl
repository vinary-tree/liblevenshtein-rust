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
