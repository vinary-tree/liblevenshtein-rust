@testset "bounded native quantized temporal candidates" begin
    config = LL.QuantizationConfig(0.0, 10.0, 5)
    source = [(UInt64(7), [1.0, 3.0]),
        (UInt64(8), [1.1, 3.1]),
        (UInt64(9), [3.0, 1.0])]
    index = LL.NativeQuantizedIndex(quant_min=0.0, quant_max=10.0,
        quant_bins=5, max_entries=3, max_total_samples=6,
        max_series_len=2)
    try
        for (id, samples) in source
            LL.insert!(index, id, samples)
        end
        @test_throws LL.NativeError LL.insert!(index, 10, [1.0])
        @test_throws ArgumentError LL.insert!(index, 10,
            UntouchableLargeVector(3))
        @test_throws ArgumentError LL.query_quantized(index,
            [1.0, 3.0], 0)
        LL.freeze!(index)
        @test_throws ArgumentError LL.insert!(index, 10, [1.0])
        query = [1.0, 3.0]
        encoded_query = collect(LL.encode_u8(config, query))
        expected = Dict{Symbol, Vector{Tuple{UInt64, Int}}}()
        for (algorithm, score) in ((:standard, LL.distance),
            (:transposition, LL.optimal_string_alignment_distance),
            (:merge_split, LL.merge_and_split_distance))
            matches = Tuple{UInt64, Int}[]
            for (id, samples) in source
                distance = score(encoded_query,
                    collect(LL.encode_u8(config, samples)))
                distance <= 1 && push!(matches, (id, distance))
            end
            expected[algorithm] = sort!(matches)
        end
        cursors = Dict(algorithm => LL.query_quantized(index, query, 1;
            algorithm, page_work_units=20, page_results=1)
            for algorithm in keys(expected))
        @test_throws ArgumentError LL.query_quantized(index, query, 1;
            algorithm=:true_damerau)
        @test_throws ArgumentError LL.query_quantized(index,
            UntouchableLargeVector(3), 1;
            limits=LL.TemporalSearchLimits(max_series_len=2))
        LL.close!(index)
        @test !isopen(index)
        for (algorithm, cursor) in cursors
            first_batch = LL.next_batch!(cursor, 1)
            @test first_batch !== nothing
            @test LL.original_samples(cursor, first_batch[1].id) ==
                only(samples for (id, samples) in source
                    if id == first_batch[1].id)
            observed = sort(vcat([(match.id, match.edit_distance)
                for match in first_batch], [(match.id, match.edit_distance)
                for match in cursor]))
            @test observed == expected[algorithm]
            @test !isopen(cursor)
            @test_throws ArgumentError LL.original_samples(cursor, 7)
        end
    finally
        close(index)
    end
end

@testset "quantized candidate limits, reducer, and concurrency" begin
    index = LL.NativeQuantizedIndex(quant_min=0.0, quant_max=10.0,
        quant_bins=5, max_entries=3, max_total_samples=6,
        max_series_len=2)
    try
        LL.insert!(index, 7, [1.0, 3.0])
        LL.insert!(index, 8, [1.1, 3.1])
        LL.insert!(index, 9, [1.2, 3.2])
        LL.freeze!(index)
        restricted = LL.TemporalSearchLimits(max_results=1)
        cursor = LL.query_quantized(index, [1.0, 3.0], 0;
            limits=restricted, page_results=1)
        @test length(LL.next_batch!(cursor, 1)) == 1
        failure = try
            LL.next_batch!(cursor, 1)
            nothing
        catch error
            error
        end
        @test failure isa LL.TemporalQueryIncomplete
        @test failure.detail === :results
        @test !isopen(cursor)
        @test_throws LL.TemporalQueryIncomplete LL.query_quantized(
            index, [1.0, 3.0], 0;
            limits=LL.TemporalSearchLimits(max_continuation_bytes=0))
        @test LL.reduce_batches!((count, batch) -> count + length(batch),
            0, LL.query_quantized(index, [1.0, 3.0], 0;
                page_results=1); batch_size=1) == 3
        tasks = [Threads.@spawn begin
            cursor = LL.query_quantized(index, [1.0, 3.0], 0;
                page_results=1)
            sort([match.id for match in cursor])
        end for _ in 1:2]
        @test all(task -> fetch(task) == UInt64[7, 8, 9], tasks)
    finally
        close(index)
    end
    nonfinite = LL.NativeQuantizedIndex(quant_min=0.0,
        quant_max=10.0, quant_bins=5, max_entries=1,
        max_total_samples=1, max_series_len=1)
    try
        LL.insert!(nonfinite, 1, [NaN])
        LL.freeze!(nonfinite)
        @test [(match.id, match.edit_distance) for match in
            LL.query_quantized(nonfinite, [-Inf], 0)] == [(UInt64(1), 0)]
    finally
        close(nonfinite)
    end
    replaced = LL.NativeQuantizedIndex(quant_min=0.0,
        quant_max=10.0, quant_bins=10, max_entries=1,
        max_total_samples=1, max_series_len=1)
    try
        for sample in 0:9
            LL.insert!(replaced, 1, [Float64(sample)])
        end
        LL.freeze!(replaced)
        cursor = LL.query_quantized(replaced, [9.0], 0)
        @test [(match.id, match.edit_distance) for match in cursor] ==
            [(UInt64(1), 0)]
        close(cursor)
        cursor = LL.query_quantized(replaced, [9.0], 0)
        @test LL.original_samples(cursor, 1) == [9.0]
        close(cursor)
    finally
        close(replaced)
    end
end
