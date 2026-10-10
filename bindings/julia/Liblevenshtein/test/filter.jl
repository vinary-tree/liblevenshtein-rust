@testset "native Jaro source filtering" begin
    @test LL.jaro_similarity("", "") == 1.0
    @test LL.jaro_similarity("", "abc") == 0.0
    @test LL.jaro_winkler_similarity("MARTHA", "MARHTA") >
        LL.jaro_similarity("MARTHA", "MARHTA")
    @test LL.jaro_winkler_similarity_scaled("MARTHA", "MARHTA", 0.0) ==
        LL.jaro_similarity("MARTHA", "MARHTA")
    @test LL.is_similar("café", "cafe", 0.75)
    @test_throws LL.NativeError LL.jaro_similarity("abcdef", "abc";
        max_input_bytes=2)
    @test_throws LL.NativeError LL.jaro_similarity("abcdef", "abc";
        max_comparisons=2)
    @test_throws LL.NativeError LL.jaro_similarity("a", "a";
        prefix_scale=0.5)
    @test LL.ngram_candidate("hello", "help", 1)
    @test !LL.ngram_candidate("hello", "world", 0)
    @test LL.hybrid_candidate("martha", "marhta", 2)
    @test_throws LL.NativeError LL.ngram_candidate("hello", "help", 1;
        max_input_bytes=2)
    @test_throws LL.NativeError LL.hybrid_candidate("hello", "help", 1;
        max_comparisons=1)

    dictionary = Libdictenstein.DynamicDawg()
    for (term, id) in (("martha", 1), ("marhta", 2),
        ("marina", 3), ("apple", 4))
        dictionary[term] = id
    end
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    try
        expected = sort([(m.term, Float64(m.distance))
            for m in LL.query(transducer, "martha", 2)
            if LL.jaro_winkler_similarity(m.term, "martha") >= 0.8])
        cursor = LL.query_jaro(transducer, "martha", 2;
            minimum_similarity=0.8)
        observed = sort([(m.term, m.cost) for m in cursor])
        @test observed == expected
        ngram_expected = sort([(m.term, Float64(m.distance))
            for m in LL.query(transducer, "martha", 2)
            if LL.ngram_candidate("martha", m.term, 2)])
        @test sort([(m.term, m.cost) for m in
            LL.query_ngram(transducer, "martha", 2)]) == ngram_expected
        hybrid_expected = sort([(m.term, Float64(m.distance))
            for m in LL.query(transducer, "martha", 2)
            if LL.hybrid_candidate("martha", m.term, 2)])
        @test sort([(m.term, m.cost) for m in
            LL.query_hybrid(transducer, "martha", 2)]) == hybrid_expected
        @test_throws ArgumentError LL.query_jaro(transducer, "martha", 2;
            minimum_similarity=NaN)
        stopped = LL.query_jaro(transducer, "martha", 2)
        @test iterate(stopped) !== nothing
        LL.cancel!(stopped)
        @test !isopen(stopped)
    finally
        isopen(transducer) && LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
end
