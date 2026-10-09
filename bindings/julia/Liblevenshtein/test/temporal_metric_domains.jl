@testset "canonical ERP and Fréchet metric domains" begin
    config = LL.MetricErpConfig(0.0)
    raw = [1.0, 0.0, 2.0]
    left = LL.representative(config, raw)
    right = LL.ErpQuotientSeries([1.0, 2.0], 0.0)
    @test LL.canonical_samples(left) == [1.0, 2.0]
    @test left.samples == [1.0, 2.0]
    @test left.gap == 0.0
    @test LL.metric_erp_distance(config, left, right).value == 0.0
    @test LL.metric_erp_distance(config, left, right).value ==
        LL.erp_distance(LL.canonical_samples(left),
            LL.canonical_samples(right); gap=0.0).value
    @test LL.metric_erp_distance(config, left, right;
        limits=LL.TemporalLimits(max_dp_cells=0)).kind === :incomplete
    raw[1] = 99.0
    visible = left.samples
    visible[1] = 99.0
    @test LL.canonical_samples(left) == [1.0, 2.0]
    @test_throws ArgumentError left._samples
    @test_throws ArgumentError LL.metric_erp_distance(
        LL.MetricErpConfig(-0.0), left, right)
    @test_throws ArgumentError LL.MetricErpConfig(Inf)
    @test_throws ArgumentError LL.ErpQuotientSeries([NaN], 0.0)
    @test_throws ArgumentError LL.ErpQuotientSeries(
        UntouchableLargeVector(4), 0.0; max_series_len=3)
    @test LL.canonical_samples(LL.ErpQuotientSeries([-0.0], 0.0)) ==
        Float64[]

    erp_index = LL.MetricErpIndex(config; quant_min=-10.0,
        quant_max=10.0, max_entries=2, max_total_samples=8,
        max_series_len=4)
    @test_throws ArgumentError erp_index._index
    try
        LL.insert!(erp_index, 7, [1.0, 0.0, 2.0])
        LL.insert!(erp_index, 11, right)
        LL.freeze!(erp_index)
        @test_throws ArgumentError LL.insert!(erp_index, 13,
            UntouchableLargeVector(4))
        cursor = LL.query_metric_range(erp_index,
            [1.0, 0.0, 0.0, 2.0]; cutoff=0.0)
        second = LL.query_metric_range(erp_index,
            LL.representative(config, [1.0, 2.0]); cutoff=0.0)
        tasks = [Threads.@spawn begin
            independent = LL.query_metric_range(erp_index,
                [1.0, 2.0]; cutoff=0.0)
            sort([match.id for match in independent])
        end for _ in 1:2]
        @test all(task -> fetch(task) == UInt64[7, 11], tasks)
        @test_throws ArgumentError LL.query_metric_range(erp_index,
            LL.ErpQuotientSeries([1.0, 2.0], -0.0); cutoff=0.0)
        nearest = LL.query_metric_knn(erp_index,
            [1.0, 0.0, 2.0], 2; page_results=1)
        @test [match.id for match in LL.query_metric_knn(erp_index,
            right, 1)] == UInt64[7]
        @test_throws ArgumentError LL.query_metric_knn(erp_index,
            LL.ErpQuotientSeries([1.0, 2.0], -0.0), 1)
        @test_throws LL.TemporalQueryIncomplete LL.query_metric_knn(
            erp_index, right, 1;
            limits=LL.TemporalSearchLimits(max_candidates=0))
        close(erp_index)
        @test !isopen(erp_index)
        @test [(match.id, match.distance) for match in nearest] ==
            [(UInt64(7), 0.0), (UInt64(11), 0.0)]
        @test sort([(match.id, match.distance) for match in cursor]) ==
            [(UInt64(7), 0.0), (UInt64(11), 0.0)]
        @test sort([(match.id, match.distance) for match in second]) ==
            [(UInt64(7), 0.0), (UInt64(11), 0.0)]
    finally
        close(erp_index)
    end

    path = LL.FrechetStutterClass([1.0, 1.0, 2.0, 2.0, 1.0])
    other = LL.FrechetStutterClass([1.0, 2.0, 1.0])
    @test LL.canonical_samples(path) == [1.0, 2.0, 1.0]
    @test LL.metric_frechet_distance(path, other).value == 0.0
    @test LL.metric_frechet_distance(path, other).value ==
        LL.frechet_distance(LL.canonical_samples(path),
            LL.canonical_samples(other)).value
    @test LL.metric_frechet_distance(path, other;
        limits=LL.TemporalLimits(max_dp_cells=0)).kind === :incomplete
    visible = path.samples
    visible[1] = 99.0
    @test LL.canonical_samples(path) == [1.0, 2.0, 1.0]
    @test_throws ArgumentError path._samples
    @test_throws ArgumentError LL.FrechetStutterClass(Float64[])
    @test_throws ArgumentError LL.FrechetStutterClass([Inf])
    @test_throws ArgumentError LL.FrechetStutterClass(
        UntouchableLargeVector(4); max_series_len=3)
    @test LL.canonical_samples(LL.FrechetStutterClass([-0.0, 0.0])) ==
        [-0.0]

    frechet_index = LL.MetricFrechetIndex(quant_min=-10.0,
        quant_max=10.0, max_entries=2, max_total_samples=8,
        max_series_len=5)
    @test_throws ArgumentError frechet_index._index
    try
        LL.insert!(frechet_index, 7, [1.0, 1.0, 2.0, 2.0])
        LL.insert!(frechet_index, 11,
            LL.FrechetStutterClass([1.0, 2.0]))
        LL.freeze!(frechet_index)
        cursor = LL.query_metric_range(frechet_index,
            [1.0, 1.0, 1.0, 2.0]; cutoff=0.0)
        @test sort([(match.id, match.distance) for match in cursor]) ==
            [(UInt64(7), 0.0), (UInt64(11), 0.0)]
        @test_throws ArgumentError LL.query_metric_range(frechet_index,
            UntouchableLargeVector(4);
            limits=LL.TemporalSearchLimits(max_series_len=3))
        @test [match.id for match in LL.query_metric_knn(frechet_index,
            [1.0, 1.0, 2.0], 2; page_results=1)] == UInt64[7, 11]
        @test_throws ArgumentError LL.query_metric_knn(frechet_index,
            Float64[], 1)
        @test_throws ArgumentError LL.query_metric_knn(frechet_index,
            UntouchableLargeVector(4), 1;
            limits=LL.TemporalSearchLimits(max_series_len=3))
    finally
        close(frechet_index)
    end

    unrelated = LL.TemporalIndex(:dtw; quant_min=-10.0,
        quant_max=10.0, band=1, max_entries=1)
    try
        @test_throws MethodError LL.MetricErpIndex(unrelated, config)
        @test_throws MethodError LL.MetricFrechetIndex(unrelated)
    finally
        close(unrelated)
    end
end

@testset "validated MSM and unit-grid TWED metric witnesses" begin
    msm = LL.MetricMsmConfig(1.0)
    twed = LL.MetricTwedConfig(1.0, 0.0)
    query = [1.0, 2.0]
    candidate = [1.0, 2.5]
    @test LL.metric_msm_distance(msm, query, candidate).value ==
        LL.msm_distance(query, candidate; split_merge_cost=1.0).value
    @test LL.metric_twed_distance(twed, query, candidate).value ==
        LL.twed_distance(query, candidate; stiffness=1.0,
            gap_penalty=0.0).value
    @test LL.metric_msm_distance(msm, query, candidate;
        limits=LL.TemporalLimits(max_dp_cells=0)).kind === :incomplete
    @test LL.metric_twed_distance(twed, query, candidate;
        limits=LL.TemporalLimits(max_dp_cells=0)).kind === :incomplete
    @test_throws ArgumentError LL.metric_msm_distance(msm, Float64[], query)
    @test_throws ArgumentError LL.metric_msm_distance(msm, query, Float64[])
    @test_throws ArgumentError LL.MetricMsmConfig(0.0)
    @test_throws ArgumentError LL.MetricMsmConfig(Inf)
    @test_throws ArgumentError LL.MetricTwedConfig(0.0, 0.0)
    @test_throws ArgumentError LL.MetricTwedConfig(1.0, -1.0)
    @test_throws ArgumentError LL.MetricTwedConfig(1.0, NaN)

    msm_index = LL.MetricMsmIndex(msm; quant_min=0.0,
        quant_max=5.0, max_entries=2, max_total_samples=4,
        max_series_len=2)
    try
        @test_throws ArgumentError msm_index._index
        @test_throws ArgumentError LL.insert!(msm_index, 1, Float64[])
        LL.insert!(msm_index, 7, query)
        LL.insert!(msm_index, 11, [3.0, 4.0])
        LL.freeze!(msm_index)
        @test_throws ArgumentError LL.query_metric_range(msm_index,
            Float64[]; cutoff=0.0)
        @test [(match.id, match.distance) for match in
            LL.query_metric_range(msm_index, query; cutoff=0.0)] ==
            [(UInt64(7), 0.0)]
        @test [match.id for match in LL.query_metric_knn(msm_index,
            query, 2)] == UInt64[7, 11]
        @test_throws ArgumentError LL.query_metric_knn(msm_index,
            Float64[], 1)
    finally
        close(msm_index)
    end

    twed_index = LL.MetricTwedIndex(twed; quant_min=0.0,
        quant_max=5.0, max_entries=2, max_total_samples=4,
        max_series_len=2)
    try
        @test_throws ArgumentError twed_index._index
        LL.insert!(twed_index, 7, query)
        LL.insert!(twed_index, 11, [3.0, 4.0])
        LL.freeze!(twed_index)
        @test [(match.id, match.distance) for match in
            LL.query_metric_range(twed_index, query; cutoff=0.0)] ==
            [(UInt64(7), 0.0)]
        @test [match.id for match in LL.query_metric_knn(twed_index,
            query, 2)] == UInt64[7, 11]
    finally
        close(twed_index)
    end
end
