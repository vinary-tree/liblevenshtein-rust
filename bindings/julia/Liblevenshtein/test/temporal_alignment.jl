@testset "paged temporal alignment witnesses" begin
    query = [1.0, 2.0, 3.0]
    candidate = [1.0, 2.5, 3.0]
    cases = [(:msm, 1.0, 0.0, 0), (:erp, 0.0, 0.0, 0),
        (:twed, 0.5, 1.0, 0), (:dtw, 0.0, 0.0, 1),
        (:frechet, 0.0, 0.0, 0)]
    for (kind, parameter0, parameter1, band) in cases
        outcome = LL.temporal_alignment(kind, query, candidate;
            parameter0, parameter1, band)
        @test outcome.kind === :finite
        @test outcome.reason === nothing
        @test outcome.witness isa LL.TemporalAlignmentWitness
        witness = outcome.witness
        try
            native = LL.temporal_distance(kind, query, candidate;
                parameter0, parameter1, band)
            @test outcome.distance == native.value
            @test LL.replay_alignment(witness, query, candidate) ==
                outcome.distance
            steps = collect(witness)
            @test length(steps) == length(witness)
            @test !isempty(steps)
            @test LL.alignment_page(witness, 1; page_size=2) == steps[1:2]
            @test LL.reduce_batches!((count, page) -> count + length(page),
                0, witness; batch_size=2) == length(steps)
            @test all(step -> step.operation in
                (:move, :merge, :split, :align, :advance_query,
                    :advance_candidate), steps)
            @test all(step -> step.query_endpoint === nothing ||
                step.query_endpoint >= 1, steps)
            tasks = [Threads.@spawn LL.alignment_page(witness, 1;
                page_size=2) for _ in 1:2]
            @test all(task -> fetch(task) == steps[1:2], tasks)
            close(witness)
            @test !isopen(witness)
            @test_throws ArgumentError LL.alignment_page(witness)
        finally
            close(witness)
        end
    end

    incomplete = LL.temporal_alignment(:erp, query, candidate;
        max_witness_bytes=0)
    @test incomplete.kind === :incomplete
    @test incomplete.reason === :witness_bytes
    @test incomplete.witness === nothing
    above = LL.temporal_alignment(:erp, query, candidate; cutoff=0.0)
    @test above.kind === :above_cutoff
    @test above.witness === nothing
    @test_throws ArgumentError LL.temporal_alignment(:soft_dtw,
        query, candidate)

    config = LL.MetricTimestampedTwedConfig(0.5, 1.0)
    first = LL.TimestampedSeries([1.0, 2.0], [10.0, 13.0];
        unit=:milliseconds, origin=10.0)
    second = LL.TimestampedSeries([1.0, 2.5], [10.0, 14.0];
        unit=:milliseconds, origin=10.0)
    physical = LL.timestamped_twed_alignment(config, first, second)
    @test physical.kind === :finite
    witness = physical.witness
    try
        @test physical.distance ==
            LL.metric_timestamped_twed_distance(config, first, second).value
        @test LL.replay_alignment(witness, first, second) ==
            physical.distance
        @test all(step -> step.local_cost !== nothing, collect(witness))
    finally
        close(witness)
    end
end
