@testset "copied lazy native source filters" begin
    caller_terms = ["hello", "help", "world", "café", "hello"]
    source = LL.SourceFilterSource(caller_terms; max_terms=5,
        max_term_bytes=16, max_source_bytes=80)
    @test length(source) == 4
    caller_terms[1] = "changed"
    @test source.terms[1] == "hello"
    @test_throws ArgumentError LL.SourceFilterSource(["long"];
        max_term_bytes=2)
    @test_throws ArgumentError LL.SourceFilterSource(["a", "b"];
        max_terms=1)

    for (query, distance) in (("helo", 1), ("cafe", 1), ("z", 0))
        expected = [term for term in source.terms if
            LL.ngram_candidate(query, term, distance; ngram_size=2)]
        actual = collect(LL.query_ngram(source, query, distance;
            ngram_size=2, page_candidates=1))
        @test actual == expected
    end
    expected_hybrid = [term for term in source.terms if
        LL.hybrid_candidate("helo", term, 1; ngram_size=2,
            jaro_threshold=0.7)]
    actual_hybrid = collect(LL.query_hybrid(source, "helo", 1;
        ngram_size=2, jaro_threshold=0.7, page_candidates=1))
    @test actual_hybrid == expected_hybrid

    reduced = LL.query_ngram(source, "helo", 1; page_candidates=1)
    @test LL.reduce_batches!((all, batch) -> append!(all, batch),
        String[], reduced; batch_size=2) ==
        [term for term in source.terms if
            LL.ngram_candidate("helo", term, 1)]
    @test !isopen(reduced)

    candidates = LL.query_ngram(source, "helo", 1;
        page_candidates=1,
        limits=LL.SourceFilterLimits(max_candidates=1))
    @test_throws LL.SourceFilterIncomplete collect(candidates)
    @test !isopen(candidates)

    results = LL.query_ngram(source, "hello", 0;
        limits=LL.SourceFilterLimits(max_results=0))
    @test_throws LL.SourceFilterIncomplete collect(results)
    @test !isopen(results)

    comparisons = LL.query_hybrid(source, "hello", 1;
        limits=LL.SourceFilterLimits(max_comparisons=0))
    @test_throws LL.SourceFilterIncomplete collect(comparisons)
    @test !isopen(comparisons)

    first = LL.query_ngram(source, "helo", 1; page_candidates=1)
    second = LL.query_ngram(source, "helo", 1; page_candidates=1)
    tasks = [@async collect(cursor) for cursor in (first, second)]
    @test fetch(tasks[1]) == fetch(tasks[2])
    @test !isopen(first) && !isopen(second)

    stopped = LL.query_ngram(source, "helo", 1)
    LL.cancel!(stopped)
    @test !isopen(stopped)
end
