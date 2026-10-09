@testset "persistent native source-filter indexes" begin
    terms = ["hello", "help", "world", "héllo", "hello"]
    source = LL.SourceFilterSource(terms;
        max_terms=4, max_term_bytes=10, max_source_bytes=32)
    ngram = LL.NativeSourceFilterIndex(source; mode=:ngram,
        ngram_size=2, max_terms=4, max_term_bytes=10,
        max_source_bytes=32, max_query_bytes=10)
    try
        @test length(ngram) == 4
        expected = collect(LL.query_ngram(source, "helo", 1;
            ngram_size=2, page_candidates=1))
        first = LL.query_ngram(ngram, "helo", 1; page_results=1)
        second = LL.query_ngram(ngram, "helo", 1; page_results=2)
        empty!(ngram.source.terms)
        close(ngram)
        @test !isopen(ngram)
        @test collect(first) == expected
        @test LL.reduce_batches!((all, batch) -> append!(all, batch),
            String[], second; batch_size=1) == expected
        @test !isopen(first) && !isopen(second)
        @test_throws ArgumentError LL.query_ngram(ngram, "helo", 1)
    finally
        close(ngram)
    end

    hybrid = LL.NativeSourceFilterIndex(source; mode=:hybrid,
        ngram_size=2, jaro_threshold=0.7, max_terms=4,
        max_term_bytes=10, max_source_bytes=32, max_query_bytes=10)
    try
        expected = collect(LL.query_hybrid(source, "helo", 1;
            ngram_size=2, jaro_threshold=0.7, page_candidates=1))
        @test collect(LL.query_hybrid(hybrid, "helo", 1;
            page_results=1)) == expected
        @test_throws ArgumentError LL.query_ngram(hybrid, "helo", 1)
        @test_throws LL.SourceFilterIncomplete LL.query_hybrid(
            hybrid, "helo", 1;
            limits=LL.SourceFilterLimits(max_candidates=3))
        @test_throws LL.SourceFilterIncomplete LL.query_hybrid(
            hybrid, "helo", 1;
            limits=LL.SourceFilterLimits(max_results=0))
        @test_throws LL.SourceFilterIncomplete LL.query_hybrid(
            hybrid, "helo", 1;
            limits=LL.SourceFilterLimits(max_comparisons=0))
        @test_throws LL.SourceFilterIncomplete LL.query_hybrid(
            hybrid, "this query is too long", 1)
        @test_throws ArgumentError LL.query_hybrid(hybrid, "helo", -1)
        stopped = LL.query_hybrid(hybrid, "helo", 1)
        LL.cancel!(stopped)
        @test !isopen(stopped)
    finally
        close(hybrid)
    end

    empty_index = LL.NativeSourceFilterIndex(String[];
        max_terms=0, max_source_bytes=0)
    try
        @test isempty(empty_index)
        @test isempty(collect(LL.query_ngram(empty_index, "", 0)))
    finally
        close(empty_index)
    end
    @test_throws ArgumentError LL.NativeSourceFilterIndex(source;
        ngram_size=0)
    @test_throws ArgumentError LL.NativeSourceFilterIndex(source;
        mode=:ngram, jaro_threshold=0.7)
    @test_throws ArgumentError LL.NativeSourceFilterIndex(source;
        mode=:hybrid, jaro_threshold=NaN)
end
