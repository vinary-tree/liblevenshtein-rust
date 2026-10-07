using Test

@testset "native cost-domain automata" begin
    @test sizeof(LL.RawAffineCosts) == 32
    @test sizeof(LL.RawOperationCostsF64) == 56
    @test sizeof(LL.RawCostMatch) == 56
    @test sizeof(LL.RawCostBatch) == 24

    affine = LL.AffineGapCosts(0.5, 0.25, 1.0)
    @test affine.scale_denominator == 4
    @test LL.AffineGapCosts(0.5, 0.25, 1.0;
        scale_denominator=8).scale_denominator == 8
    @test_throws LL.NativeError LL.AffineGapCosts(0.5, 0.25, 1.0;
        scale_denominator=2)
    @test_throws LL.NativeError LL.AffineGapCosts(NaN, 1.0, 1.0)
    @test_throws ArgumentError LL.AffineGapCosts(1.0, 1.0, 1.0;
        scale_denominator=0)

    standard = LL.WeightedOperationCosts(:standard)
    @test (standard.match_cost, standard.substitution, standard.insertion) ==
        (0.0, 1.0, 1.0)
    @test LL.WeightedOperationCosts(:typo).transposition == 0.5
    @test LL.WeightedOperationCosts(:ocr).substitution == 0.8
    @test_throws ArgumentError LL.WeightedOperationCosts(:unknown)
    @test_throws LL.NativeError LL.WeightedOperationCosts(-1, 1, 1, 1, 1, 1)
    @test_throws LL.NativeError LL.WeightedOperationCosts(1, 1, 1, 1, 1, 1;
        match_cost=1)
    @test_throws LL.NativeError LL.WeightedOperationCosts(Inf, 1, 1, 1, 1, 1)

    for (domain, query_term, alternate, query_value) in (
        (Libdictenstein.UNIT_UNICODE_SCALAR, "cat", "cut", "cat"),
        (Libdictenstein.UNIT_BYTE, UInt8[0, 255], UInt8[0, 254], UInt8[0, 255]),
        (Libdictenstein.UNIT_U64, UInt64[10, typemax(UInt64)],
            UInt64[10, 0], UInt64[10, typemax(UInt64)]),
    )
        dictionary = Libdictenstein.DynamicDawg(domain)
        dictionary[query_term] = UInt64(1)
        dictionary[alternate] = UInt64(2)
        source = Libdictenstein.snapshot(dictionary)
        transducer = LL.Transducer(source)
        try
            @test LL.UNIT_EDIT_COSTS isa LL.AbstractEditCostDomain
            @test Set(m.term for m in LL.query_cost(transducer,
                query_value, 1, LL.UNIT_EDIT_COSTS)) ==
                Set((query_term, alternate))
            cursor = LL.query_affine(transducer, query_value, 1.0, affine)
            close(source)
            close(dictionary)
            close(transducer)
            results = collect(cursor)
            @test length(results) == 2
            @test all(m -> m.unit_domain == domain, results)
            @test Set(m.term for m in results) == Set((query_term, alternate))
            @test Set(m.scaled_cost for m in results) == Set((UInt(0), UInt(4)))
            @test all(m -> m.scale_denominator == 4, results)
            @test all(m -> m.cost == Float64(m.scaled_cost) / 4, results)
            @test_throws LL.NativeError LL.next_batch!(cursor)
        finally
            close(transducer)
            close(source)
            close(dictionary)
        end

        dictionary = Libdictenstein.DynamicDawg(domain)
        dictionary[query_term] = UInt64(1)
        dictionary[alternate] = UInt64(2)
        source = Libdictenstein.snapshot(dictionary)
        transducer = LL.Transducer(source)
        try
            @test Set(m.term for m in LL.query_cost(transducer,
                query_value, 1.0, standard)) ==
                Set((query_term, alternate))
            cursor = LL.query_weighted(transducer, query_value, 1.0, standard)
            results = collect(cursor)
            @test length(results) == 2
            @test Set(m.cost for m in results) == Set((0.0, 1.0))
            @test all(m -> m.scaled_cost === nothing, results)
            @test all(m -> m.unit_domain == domain, results)
            @test_throws LL.NativeError LL.query_weighted(transducer,
                query_value, -1.0, standard)
            @test_throws LL.NativeError LL.query_affine(transducer,
                query_value, 0.3, affine)
        finally
            close(transducer)
            close(source)
            close(dictionary)
        end
    end
end

@testset "cost cursor leases and borrowed reducers" begin
    dictionary = Libdictenstein.DynamicDawg()
    dictionary["cat"] = UInt64(1)
    dictionary["cut"] = UInt64(2)
    source = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(source)
    costs = LL.WeightedOperationCosts(:standard)
    try
        contextual = LL.ContextualCosts(
            (_, query_unit, dictionary_unit) -> query_unit == dictionary_unit ? 0.0 : 1.0,
            (_, _) -> 1.0,
            (_, _) -> 1.0;
            minimum_nonzero_cost=1.0)
        @test Set(m.term for m in LL.query_cost(transducer,
            "cat", 1.0, contextual)) == Set(("cat", "cut"))
        cursor = LL.query_weighted(transducer, "cat", 1.0, costs)
        try
            view = Ref(LL.RawCostBatch(Ptr{LL.RawCostMatch}(C_NULL), 0, 0))
            @test ccall(LL.native(:llev_cost_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Csize_t, Ref{LL.RawCostBatch}),
                cursor.handle, 1, view) == Cint(LL.STATUS_OK)
            @test view[].len == 1
            generation = view[].generation
            @test ccall(LL.native(:llev_cost_cursor_next_batch), Cint,
                (Ptr{Cvoid}, Csize_t, Ref{LL.RawCostBatch}),
                cursor.handle, 1, view) == Cint(LL.STATUS_BATCH_IN_USE)
            @test ccall(LL.native(:llev_cost_cursor_free), Cint,
                (Ptr{Cvoid},), cursor.handle) == Cint(LL.STATUS_BATCH_IN_USE)
            @test ccall(LL.native(:llev_cost_cursor_release_batch), Cint,
                (Ptr{Cvoid}, UInt64), cursor.handle,
                generation + 1) == Cint(LL.STATUS_INVALID_ARGUMENT)
            @test ccall(LL.native(:llev_cost_cursor_release_batch), Cint,
                (Ptr{Cvoid}, UInt64), cursor.handle,
                generation) == Cint(LL.STATUS_OK)
            @test length(LL.next_batch!(cursor, 1)) == 1
            @test LL.next_batch!(cursor, 1) === nothing
        finally
            close(cursor)
        end

        borrowed = Ref{Any}(nothing)
        result = LL.reduce_batches!(0, LL.query_weighted(transducer,
            "cat", 1.0, costs); batch_size=1) do count, batch
            borrowed[] = batch[1]
            count + length(batch)
        end
        @test result == 2
        @test_throws ArgumentError LL.materialize(borrowed[])

        unsupported = LL.Transducer(source, LL.ALGORITHM_DAMERAU_LEVENSHTEIN)
        try
            @test_throws LL.NativeError LL.query_weighted(unsupported,
                "cat", 1.0, costs)
        finally
            close(unsupported)
        end
    finally
        close(transducer)
        close(source)
        close(dictionary)
    end
end
