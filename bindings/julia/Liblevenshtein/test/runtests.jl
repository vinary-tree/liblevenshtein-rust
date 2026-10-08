using Test
using Liblevenshtein
using Libdictenstein
import VinaryTreeInterop

const LL = Liblevenshtein

struct UntouchableLargeVector <: AbstractVector{Float64}
    count::Int
end
Base.size(values::UntouchableLargeVector) = (values.count,)
Base.getindex(::UntouchableLargeVector, ::Int) =
    error("oversized input was read before its length was checked")

include("temporal.jl")
include("temporal_bounds.jl")
include("temporal_encoding.jl")
include("temporal_quantization.jl")
include("temporal_delta_encoding.jl")
include("temporal_sax_encoding.jl")
include("temporal_queries.jl")
include("temporal_index.jl")
include("temporal_rolling.jl")
include("temporal_online.jl")
include("filter.jl")
include("source_filter_queries.jl")

@testset "canonical source and indexed temporal examples" begin
    source = LL.SourceFilterSource(["hello", "help", "world"];
        max_terms=3, max_term_bytes=16, max_source_bytes=48)
    @test LL.ngram_candidate("helo", "hello", 1)
    @test LL.hybrid_candidate("helo", "hello", 1)
    @test "hello" in collect(LL.query_ngram(source, "helo", 1;
        page_candidates=2))
    @test "hello" in collect(LL.query_hybrid(source, "helo", 1;
        page_candidates=2))

    index = LL.TemporalIndex(:dtw; quant_min=0.0, quant_max=10.0,
        band=2, max_entries=2, max_total_samples=6, max_series_len=3)
    try
        LL.insert!(index, 7, [1.0, 2.0, 3.0])
        LL.freeze!(index)
        cursor = LL.query_index_range(index, [1.0, 2.0, 3.0];
            cutoff=0.0, page_work_units=1_000, page_results=1)
        @test only(collect(cursor)).id == 7
    finally
        close(index)
    end

    query = [1.0, 2.0, 3.0]
    candidate = [1.0, 4.0, 3.0]
    @test LL.temporal_lower_bound(:keogh, query, candidate;
        band=1).kind === :finite
    @test LL.erp_gap_mass_lower_bound(query, candidate, 0.0).kind === :finite
    @test LL.frechet_endpoint_lower_bound(query, candidate).kind === :finite
    @test LL.frechet_one_sided_hausdorff_lower_bound(
        query, candidate).kind === :finite
    @test LL.frechet_candidate_lower_bound(query, candidate).kind === :finite
    @test LL.twed_length_lower_bound(3, 4, 0.5).value ≈ 0.5
    plan = LL.keogh_envelopes(query, 1)
    try
        @test LL.bounds_at(plan, 2) == (1.0, 3.0)
        @test LL.lb_keogh(candidate, plan).kind === :finite
        @test LL.lb_keogh_squared(candidate, plan).kind === :finite
    finally
        close(plan)
    end
end

@testset "ABI identity and layouts" begin
    @test LL.abi_version() == LL.ABI_VERSION == 1
    @test LL.api_revision() >= LL.API_REVISION
    @test LL.build_features() & LL.BUILD_FEATURE_CORE != 0
    @test LL.STATUS_OK isa LL.Status
    @test LL.ALGORITHM_STANDARD isa LL.Algorithm
    @test LL.ORDER_TRAVERSAL isa LL.QueryOrder
    @test LL.RULES_ENGLISH_ORTHOGRAPHY isa LL.PhoneticRuleSetKind
    @test LL.APPLICABILITY_ANY isa LL.OperationApplicability
    @test LL.UNIVERSAL_STANDARD isa LL.UniversalVariant
    @test sizeof(LL.RawMatch) == 48
    @test sizeof(LL.RawBatch) == 24
    @test sizeof(LL.RawQueryCacheStats) == 64
    @test sizeof(LL.OwnedString) == 16
    @test sizeof(LL.RawAutomatonLimits) == 32
    @test sizeof(LL.RawGeneralizedRestriction) == 32
    @test sizeof(LL.RawGeneralizedOperation) == 64
    @test sizeof(LL.RawGeneralizedObservation) == 32
    @test sizeof(LL.RawUniversalEquivalence) == 16
    @test sizeof(LL.RawUniversalObservation) == 24
    @test sizeof(LL.RawWallBreakerTerm) == 16
    @test sizeof(LL.RawWallBreakerLimits) == 56
    @test sizeof(LL.RawWallBreakerResult) == 24
    @test sizeof(LL.RawWallBreakerBatch) == 24
    @test sizeof(LL.RawPatternPiece) == 40
    @test sizeof(LL.PhoneticFeatureWeights) == 56
    @test sizeof(LL.RawPhoneticGrepMatch) == 32
    @test sizeof(LL.RawUtf8View) == 16
    @test sizeof(LL.RawPhoneticCandidate) == 40
    @test sizeof(LL.RawPhoneticOnlineMatch) == 72
    @test sizeof(LL.RawPhoneticTokenDetail) == 64
    @test sizeof(LL.RawPhoneticTokenMatch) == 56
    @test sizeof(LL.PhoneticExpansionLimits) == 40
end

@testset "finite Unicode WallBreaker" begin
    terms = ["", "a", "café", "cafe", "αβγ", "αXγ", "café"]
    matcher = LL.WallBreakerMatcher(terms; max_distance=1)
    try
        expected = Set([LL.WallBreakerMatch("café", 0),
            LL.WallBreakerMatch("cafe", 1)])
        @test Set(collect(LL.query(matcher, "café"))) == expected
        @test Set(collect(LL.query(matcher, ""))) ==
            Set([LL.WallBreakerMatch("", 0), LL.WallBreakerMatch("a", 1)])
        @test collect(LL.query(matcher, "absent")) == LL.WallBreakerMatch[]
        cursor = LL.query(matcher, "café")
        LL.close!(matcher)
        try
            @test Set(collect(cursor)) == expected
        finally
            LL.close!(cursor)
        end
    finally
        LL.close!(matcher)
    end
    pieces = LL.pattern_pieces("éa", 1)
    @test [piece.content for piece in pieces] == ["é", "a"]
    @test [(piece.byte_offset, piece.byte_len, piece.start_scalar,
        piece.end_scalar) for piece in pieces] == [(0, 2, 0, 1), (2, 1, 1, 2)]
    @test_throws ArgumentError LL.WallBreakerLimits(max_terms=0)
    @test_throws ArgumentError LL.WallBreakerMatcher(["x"]; max_distance=9)
    matcher = LL.WallBreakerMatcher(["café"]; max_distance=0)
    try
        cursor = LL.query(matcher, "café")
        try
            @test_throws LL.NativeError LL.next_batch!(cursor, 1; max_bytes=1)
            @test LL.next_batch!(cursor, 1; max_bytes=16) ==
                [LL.WallBreakerMatch("café", 0)]
            LL.cancel!(cursor)
            @test_throws LL.NativeError LL.next_batch!(cursor)
        finally
            LL.close!(cursor)
        end
    finally
        LL.close!(matcher)
    end
    matcher = LL.WallBreakerMatcher(["a", "b", "c"]; max_distance=1)
    try
        cursor = LL.query(matcher, "")
        try
            @test iterate(cursor) !== nothing
            @test cursor.offset <= length(cursor.pending)
            LL.cancel!(cursor)
            @test iterate(cursor) === nothing
            @test iterate(cursor) === nothing
            @test_throws LL.NativeError LL.next_batch!(cursor)
            LL.cancel!(cursor)
        finally
            LL.close!(cursor)
        end
    finally
        LL.close!(matcher)
    end
end

@testset "bounded TinyLFU/SIEVE query cache" begin
    dictionary = Libdictenstein.DynamicDawg()
    dictionary["cat"] = 7
    dictionary["cot"] = nothing
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    cache = LL.QueryCache(transducer; max_entries=8, max_weight=1 << 20)
    try
        cold = collect(LL.query(cache, "cut", 1))
        hit = collect(LL.query(cache, "cut", 1))
        @test hit == cold
        stats = LL.cache_stats(cache)
        @test (stats.requests, stats.hits, stats.misses) == (2, 1, 1)
        @test stats.resident_entries == length(cache) == 1
        @test stats.resident_weight > 0

        LL.reset_stats!(cache)
        @test LL.cache_stats(cache).requests == 0
        @test length(cache) == 1
        LL.clear!(cache)
        @test isempty(cache)
        @test_throws ArgumentError LL.QueryCache(transducer; max_entries=-1)
        @test_throws ArgumentError LL.query(cache, "cut", -1)
    finally
        LL.close!(cache)
        LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
    @test !isopen(cache)
end

@testset "standalone generalized automata" begin
    operations = LL.GeneralizedOperationSet(
        LL.GeneralizedOperation(0, 1, 1, :insert),
        LL.GeneralizedOperation(1, 0, 1, :delete),
        LL.GeneralizedOperation(1, 1, 0, :equal;
            applicability=LL.APPLICABILITY_EQUAL),
        LL.GeneralizedOperation(1, 1, 0.5, :substitute),
    )
    automaton = LL.GeneralizedAutomaton(2, operations)
    try
        result = @inferred LL.evaluate(automaton, "cat", "cut")
        @test result.accepting
        @test result.distance == 1 // 2
        @test result.scaled_distance == 1
        @test result.scale_denominator == 2
        @test LL.accepts(automaton, "cat", "cut")

        online = LL.online(automaton, "a";
            limits=LL.AutomatonLimits(max_target_units=1))
        try
            @test LL.advance!(online, 'a').accepting
            failure = try
                LL.advance!(online, 'b')
                nothing
            catch error
                error
            end
            @test failure isa LL.NativeError
            @test failure.status == Int32(LL.STATUS_LIMIT_EXCEEDED)
            @test LL.observation(online).consumed_target_length == 1
        finally
            LL.close!(online)
        end
    finally
        LL.close!(automaton)
    end

    listed = LL.GeneralizedAutomaton(1, (
        LL.GeneralizedOperation(2, 1, 0.5, :phoneme;
            restrictions=("ph" => "f",)),
    ))
    try
        @test LL.evaluate(listed, "ph", "f").distance == 1 // 2
        @test !LL.accepts(listed, "f", "ph")
    finally
        close(listed)
    end

    bridge = LL.GeneralizedAutomaton(0, (
        LL.GeneralizedOperation(2, 2, 0, :pair;
            applicability=LL.APPLICABILITY_EQUAL),
    ))
    try
        prefixes = LL.prefix_observations(bridge, "ab", "ab")
        values = collect(prefixes)
        @test !values[1].current_row_nonempty
        @test !values[1].accepting
        @test values[2].accepting
        @test values[2].distance == 0 // 1
        @test !isopen(prefixes)
    finally
        close(bridge)
    end

    invalid = LL.GeneralizedOperation(0, 0, 1, :invalid)
    error = try
        LL.GeneralizedAutomaton(1, (invalid,))
        nothing
    catch value
        value
    end
    @test error isa LL.NativeError
    @test error.status == Int32(LL.STATUS_INVALID_ARGUMENT)
end

@testset "standalone universal automata" begin
    standard = LL.UniversalAutomaton(1; variant=LL.UNIVERSAL_STANDARD)
    transposition = LL.UniversalAutomaton(1; variant=LL.UNIVERSAL_TRANSPOSITION)
    merge_split = LL.UniversalAutomaton(1; variant=LL.UNIVERSAL_MERGE_AND_SPLIT)
    try
        @test !LL.accepts(standard, "ab", "ba")
        @test LL.accepts(transposition, "ab", "ba")
        @test LL.accepts(merge_split, "a", "ab")
        @test LL.evaluate(standard, UInt8[0xff], UInt8[0xff]).accepting
        @test LL.evaluate(standard, UInt64[typemax(UInt64)],
            UInt64[typemax(UInt64)]).accepting

        prefixes = LL.prefix_observations(transposition, "ab", "ba")
        values = collect(prefixes)
        @test length(values) == 2
        @test values[end].accepting
        @test !isopen(prefixes)
    finally
        close(standard)
        close(transposition)
        close(merge_split)
    end

    policy = LL.UniversalPolicy('p' => 'f')
    directional = LL.UniversalAutomaton(0, policy)
    try
        @test LL.accepts(directional, "p", "f")
        @test !LL.accepts(directional, "f", "p")
        mismatch = try
            LL.evaluate(directional, UInt8['p'], UInt8['f'])
            nothing
        catch error
            error
        end
        @test mismatch isa LL.NativeError
        @test mismatch.status == Int32(LL.STATUS_DOMAIN_MISMATCH)

        state = LL.online(directional, "p")
        close(directional)
        try
            @test LL.advance!(state, 'f').accepting
        finally
            close(state)
        end
    finally
        isopen(directional) && close(directional)
    end

    byte_directional = LL.UniversalAutomaton(0,
        LL.UniversalPolicy(0x70 => 0x66))
    token_directional = LL.UniversalAutomaton(0,
        LL.UniversalPolicy(UInt64(7) => UInt64(9)))
    try
        @test LL.accepts(byte_directional, UInt8[0x70], UInt8[0x66])
        @test !LL.accepts(byte_directional, UInt8[0x66], UInt8[0x70])
        @test LL.accepts(token_directional, UInt64[7], UInt64[9])
        @test !LL.accepts(token_directional, UInt64[9], UInt64[7])
    finally
        close(byte_directional)
        close(token_directional)
    end

    @test_throws ArgumentError LL.UniversalPolicy(())
    @test_throws ArgumentError LL.UniversalPolicy('a' => 'b', 1 => 2)
    @test_throws ArgumentError LL.AutomatonLimits(max_target_units=-1)

    bounded = LL.UniversalAutomaton(2)
    try
        state = LL.online(bounded, "a";
            limits=LL.AutomatonLimits(max_target_units=1))
        try
            @test LL.advance!(state, 'a').accepting
            @test_throws LL.NativeError LL.advance!(state, 'b')
            @test LL.observation(state).consumed_target_length == 1
        finally
            close(state)
        end
    finally
        close(bounded)
    end

    scoped_open = Ref(false)
    scoped = LL.UniversalAutomaton(0)
    try
        LL.prefix_observations(scoped, "x", "x") do prefixes
            scoped_open[] = isopen(prefixes)
            @test only(prefixes).accepting
        end
    finally
        close(scoped)
    end
    @test scoped_open[]
end

@testset "universal variants agree with independent distance kernels" begin
    families = (
        (LL.UNIVERSAL_STANDARD, LL.distance),
        (LL.UNIVERSAL_TRANSPOSITION, LL.optimal_string_alignment_distance),
        (LL.UNIVERSAL_MERGE_AND_SPLIT, LL.merge_and_split_distance),
    )
    words = [String(Char[iszero(bits & (1 << index)) ? 'a' : 'b'
        for index in 0:(width - 1)])
        for width in 0:4 for bits in 0:((1 << width) - 1)]
    @test length(words) == 31
    @test LL.optimal_string_alignment_distance("CA", "ABC") == 3
    @test LL.true_damerau_distance("CA", "ABC") == 2

    function check_batch_and_online(automaton, family, source, target, threshold)
        expected = family(source, target) <= threshold
        @test LL.accepts(automaton, source, target) == expected
        state = LL.online(automaton, source)
        try
            for unit in target
                LL.advance!(state, unit)
            end
            observed = LL.observation(state)
            @test observed.accepting == expected
            @test observed.consumed_target_length == length(target)
        finally
            close(state)
        end
    end

    # The native distance kernels use dynamic programming, independently of the
    # universal position-state machine. Sweep bounds to catch refunded edit
    # costs and unfinished operations, not only successful one-edit examples.
    for (variant, family) in families, threshold in 0:3
        @testset "variant=$variant threshold=$threshold" begin
            automaton = LL.UniversalAutomaton(threshold; variant=variant)
            try
                for source in words, target in words
                    check_batch_and_online(automaton, family, source, target, threshold)
                end
                for (source, target) in (
                    ("ab", "ba"), ("abc", "bad"), ("CA", "ABC"),
                    ("ab", "x"), ("x", "ab"),
                    ("", ""), ("", "\0"), ("\0", ""), ("\0a", "a\0"),
                    ("\$a", "a\$"), ("éλ", "λé"),
                )
                    check_batch_and_online(automaton, family, source, target, threshold)
                    check_batch_and_online(automaton, family,
                        collect(codeunits(source)), collect(codeunits(target)), threshold)
                    check_batch_and_online(automaton, family,
                        UInt64.(collect(source)), UInt64.(collect(target)), threshold)
                end
                # Full-width tokens and non-UTF-8 bytes are data, not padding.
                for (source, target) in (
                    (UInt8[0, 0xff], UInt8[0xff, 0]),
                    (UInt64[0, typemax(UInt64)], UInt64[typemax(UInt64), 0]),
                )
                    check_batch_and_online(automaton, family, source, target, threshold)
                end
            finally
                close(automaton)
            end
        end
    end
end

include("automata_qualification.jl")
include("cost_automata.jl")

@testset "resource-backed snapshots, iteration, and reduction" begin
    dictionary = Libdictenstein.DynamicDawg()
    dictionary["cat"] = 7
    dictionary["cot"] = nothing
    dictionary["dog"] = 9
    dictionary["ba"] = 10
    dictionary["m"] = 11
    dictionary["ABC"] = 12
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    try
        @test_throws ArgumentError LL.query(transducer, "cut", -1)
        cursor = LL.query(transducer, "cut", 1;
            order=LL.ORDER_DISTANCE_THEN_TERM)
        @test cursor isa LL.QueryCursor
        matches = collect(cursor)
        @test [(match.term, match.distance, match.id) for match in matches] ==
            [("cat", 1, UInt64(7)), ("cot", 1, nothing)]

        fixed = LL.snapshot(transducer)
        LL.close!(transducer)
        try
            batch_cursor = LL.query(fixed, "cut", 1)
            batch = LL.next_batch!(batch_cursor, 1)
            @test length(batch) == 1
            LL.close!(batch_cursor)
            @test_throws LL.NativeError LL.next_batch!(batch_cursor)

            count = LL.reduce_batches!((total, batch) -> begin
                @test all(match -> match.unit_domain ==
                    VinaryTreeInterop.UNIT_UNICODE_SCALAR, batch)
                total + length(batch)
            end, 0, LL.query(fixed, "cut", 1); batch_size=1)
            @test count == 2
        finally
            LL.close!(fixed)
        end
    finally
        isopen(transducer) && LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
end

@testset "ranked Unicode traversal adapters" begin
    dictionary = Libdictenstein.DynamicDawg()
    for (term, id) in (("cat", 1), ("bat", 2), ("cot", 20),
        ("cut", 3), ("coat", 10), ("dog", 100))
        dictionary[term] = id
    end
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    try
        ranked = collect(LL.query_ranked(transducer, "cat", 1))
        @test [(match.term, match.distance) for match in ranked] ==
            [("cat", 0), ("bat", 1), ("coat", 1), ("cot", 1), ("cut", 1)]
        @test [match.term for match in LL.query_mode(transducer, "cat";
            minimum_distance=1, maximum_distance=1)] ==
            ["bat", "coat", "cot", "cut"]
        @test_throws ArgumentError LL.query_mode(transducer, "cat";
            minimum_distance=2, maximum_distance=1)
        @test_throws ArgumentError LL.query_mode(transducer, "cat";
            maximum_distance=-1)

        scored = LL.query_suggestions(transducer, "cat", 1,
            (_, _, id) -> Float64(id))
        @test [(m.term, m.distance, m.id, m.confidence) for m in
            LL.next_batch!(scored, 2)] ==
            [("cat", 0, UInt64(1), 1.0), ("cot", 1, UInt64(20), 20.0)]
        @test [match.term for match in collect(scored)] ==
            ["coat", "cut", "bat"]
        @test !isopen(scored)
        @test LL.next_batch!(scored) === nothing
        @test [m.term for m in LL.query_suggestions(transducer, "cat", 1,
            (_, _, _) -> 0.0)] == [m.term for m in ranked]
        @test [m.term for m in LL.query_suggestions(transducer, "cat", 1,
            (term, _, _) -> term == "bat" ? NaN : 0.0)] ==
            ["cat", "coat", "cot", "cut", "bat"]

        @test LL.reduce_batches!((n, batch) -> n + length(batch), 0,
            LL.query_mode(transducer, "cat";
                minimum_distance=1, maximum_distance=1); batch_size=2) == 4
        @test LL.reduce_batches!((n, batch) -> n + length(batch), 0,
            LL.query_suggestions(transducer, "cat", 1,
                (_, _, _) -> 0.0); batch_size=2) == 5

        cancelled = LL.query_suggestions(transducer, "cat", 1,
            (_, _, _) -> 0.0)
        @test iterate(cancelled) !== nothing
        LL.cancel!(cancelled)
        @test !isopen(cancelled)
        @test iterate(cancelled) === nothing
        failed = LL.query_suggestions(transducer, "cat", 1,
            (_, _, _) -> error("scorer failure"))
        @test_throws ErrorException iterate(failed)
        @test !isopen(failed)

        pending = LL.query_suggestions(transducer, "cat", 1,
            (_, _, id) -> Float64(id))
        LL.close!(transducer)
        @test [match.term for match in pending] ==
            ["cat", "cot", "coat", "cut", "bat"]
    finally
        isopen(transducer) && LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
end

@testset "contextual and prefix-pruned native traversals" begin
    @test sizeof(LL.RawEditContext) == 48
    @test sizeof(LL.RawSpecializedMatch) == 40
    @test sizeof(LL.RawSpecializedBatch) == 24
    dictionary = Libdictenstein.DynamicDawg()
    for (term, id) in (("ce", 1), ("se", 2), ("ci", 3),
        ("cat", 4), ("sea", 5))
        dictionary[term] = id
    end
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    captured = Ref{Any}(nothing)
    costs = LL.ContextualCosts(
        (context, query_unit, candidate_unit) -> begin
            captured[] = context.query
            if query_unit == candidate_unit
                0.0
            elseif query_unit == 'c' && candidate_unit == 's' &&
                context.query_index < length(context.query) &&
                context.query[context.query_index + 1] == 'e'
                0.25
            else
                1.0
            end
        end,
        (_, _) -> 1.0,
        (_, _) -> 1.0;
        minimum_nonzero_cost=0.25)
    try
        @test_throws ArgumentError LL.ContextualCosts(
            (_, _, _) -> 0.0, (_, _) -> 1.0, (_, _) -> 1.0;
            minimum_nonzero_cost=0.0)
        @test_throws ArgumentError LL.query_contextual(transducer, "ce", NaN, costs)
        contextual = LL.query_contextual(transducer, "ce", 0.25, costs)
        @test Set((m.term, m.cost) for m in contextual) ==
            Set([("ce", 0.0), ("se", 0.25)])
        @test_throws ArgumentError captured[][1]
        @test Set(m.term for m in LL.query(transducer, "ce", 0)) == Set(["ce"])

        @test LL.reduce_batches!((n, batch) -> begin
            @test all(m -> LL.materialize(m).cost <= 0.25, batch)
            n + length(batch)
        end, 0, LL.query_contextual(transducer, "ce", 0.25, costs);
            batch_size=1) == 2
        escaped = Ref{Any}(nothing)
        LL.reduce_batches!((n, batch) -> begin
            escaped[] = batch[1]
            n + length(batch)
        end, 0, LL.query_contextual(transducer, "ce", 0.25, costs);
            batch_size=1)
        @test_throws ArgumentError LL.materialize(escaped[])

        stack = Char[]
        visitor = LL.PrefixVisitor(
            (unit, depth) -> begin
                @test depth == length(stack) + 1
                push!(stack, unit)
                depth > 1 || unit == 'c'
            end,
            (unit, depth) -> begin
                @test depth == length(stack)
                @test pop!(stack) == unit
            end;
            permits=prefix -> last(prefix) == 'e',
            score=prefix -> Float64(length(prefix)))
        pruned = LL.query_pruned(transducer, "ce", 2, visitor)
        @test [(m.term, m.cost, m.score) for m in pruned] ==
            [("ce", 0.0, 2.0)]
        @test isempty(stack)

        early = LL.query_pruned(transducer, "ce", 2,
            LL.PrefixVisitor((unit, _) -> (push!(stack, unit); true),
                (unit, _) -> (@test pop!(stack) == unit)))
        @test LL.next_batch!(early, 1) !== nothing
        LL.cancel!(early)
        @test isempty(stack)
        @test_throws LL.NativeError LL.next_batch!(early)

        failure = LL.query_contextual(transducer, "ce", 1.0,
            LL.ContextualCosts((_, _, _) -> error("callback failure"),
                (_, _) -> 1.0, (_, _) -> 1.0;
                minimum_nonzero_cost=1.0))
        @test_throws ErrorException LL.next_batch!(failure)
        @test !isopen(failure)

        violated = LL.query_contextual(transducer, "ce", 1.0,
            LL.ContextualCosts((_, q, x) -> q == x ? 0.0 : 0.1,
                (_, _) -> 1.0, (_, _) -> 1.0;
                minimum_nonzero_cost=1.0))
        @test_throws LL.NativeError LL.next_batch!(violated)
        LL.close!(violated)

        failure_stack = Char[]
        bad_visitor = LL.PrefixVisitor(
            (unit, _) -> (push!(failure_stack, unit); true),
            (unit, _) -> (@test pop!(failure_stack) == unit);
            permits=_ -> error("prefix callback failure"))
        failed_prefix = LL.query_pruned(transducer, "ce", 2, bad_visitor)
        @test_throws ErrorException LL.next_batch!(failed_prefix)
        @test !isopen(failed_prefix)
        @test isempty(failure_stack)
    finally
        LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
end

@testset "pre-materialization value filtering" begin
    dictionary = Libdictenstein.DynamicDawg()
    for (term, id) in (("ce", 1), ("se", 2), ("ci", 3),
        ("cat", 4), ("sea", 5))
        dictionary[term] = id
    end
    provider = Libdictenstein.snapshot(dictionary)
    transducer = LL.Transducer(provider)
    try
        expected = [m for m in LL.query(transducer, "ce", 2)
            if m.id !== nothing && m.id >= 3]
        observed = collect(LL.query_filtered(transducer, "ce", 2,
            id -> id !== nothing && id >= 3))
        @test [(m.term, m.distance, m.id) for m in observed] ==
            [(m.term, m.distance, m.id) for m in expected]
        @test LL.reduce_batches!((n, batch) -> n + length(batch), 0,
            LL.query_filtered(transducer, "ce", 2,
                id -> id !== nothing && id >= 3); batch_size=1) ==
            length(expected)

        by_value = collect(LL.query_by_value(transducer, "ce", 2, 3))
        @test all(m -> m.id == 3, by_value)
        @test [(m.term, m.distance, m.id) for m in by_value] ==
            [(m.term, m.distance, m.id) for m in LL.query(transducer, "ce", 2)
                if m.id == 3]
        ids = Set([2, 4])
        set_cursor = LL.query_by_value_set(transducer, "ce", 2, ids)
        push!(ids, 3)
        by_set = collect(set_cursor)
        @test [(m.term, m.distance, m.id) for m in by_set] ==
            [(m.term, m.distance, m.id) for m in LL.query(transducer, "ce", 2)
                if m.id in (2, 4)]
        @test_throws ArgumentError LL.query_by_value(transducer, "ce", 2, -1)

        stopped = LL.query_filtered(transducer, "ce", 2, _ -> true)
        @test iterate(stopped) !== nothing
        LL.cancel!(stopped)
        @test !isopen(stopped)
        @test_throws LL.NativeError LL.next_batch!(stopped)

        failed = LL.query_filtered(transducer, "ce", 2,
            _ -> error("value filter failure"))
        @test_throws ErrorException LL.next_batch!(failed)
        @test !isopen(failed)

        pending = LL.query_filtered(transducer, "ce", 2,
            id -> id !== nothing && id >= 3)
        LL.close!(transducer)
        @test [(m.term, m.id) for m in pending] ==
            [(m.term, m.id) for m in expected]
    finally
        isopen(transducer) && LL.close!(transducer)
        close(provider)
        close(dictionary)
    end
end

@testset "all unit-cost automata" begin
    dictionary = Libdictenstein.DynamicDawg()
    for (term, id) in (("ba", 1), ("m", 2), ("ABC", 3))
        dictionary[term] = id
    end
    provider = Libdictenstein.snapshot(dictionary)
    function terms(algorithm, input, maximum)
        transducer = LL.Transducer(provider, algorithm)
        try
            [match.term for match in LL.query(transducer, input, maximum)]
        finally
            LL.close!(transducer)
        end
    end
    try
        @test !("ba" in terms(LL.ALGORITHM_STANDARD, "ab", 1))
        @test "ba" in terms(LL.ALGORITHM_TRANSPOSITION, "ab", 1)
        @test "m" in terms(LL.ALGORITHM_MERGE_AND_SPLIT, "rn", 1)
        @test "ABC" in terms(LL.ALGORITHM_DAMERAU_LEVENSHTEIN, "CA", 2)
        @test !("ABC" in terms(LL.ALGORITHM_TRANSPOSITION, "CA", 2))
    finally
        close(provider)
        close(dictionary)
    end
end

@testset "distance families" begin
    @test LL.distance("kitten", "sitting") == 3
    @test LL.distance("kitten", "sitting"; threshold=2) === nothing
    @test LL.distance("kitten", "sitting"; threshold=3) == 3
    @test LL.damerau_distance("ab", "ba") == 1
    @test LL.optimal_string_alignment_distance("ab", "ba") == 1
    @test LL.true_damerau_distance("CA", "ABC") == 2
    @test LL.merge_and_split_distance("m", "rn") == 1
    @test_throws ArgumentError LL.distance("a", "b"; threshold=-1)
    @test_throws OverflowError LL.distance("a", "b"; threshold=big(typemax(UInt128)))
    malformed = String(UInt8[0xff])
    malformed_error = try
        LL.distance(malformed, "")
        nothing
    catch error
        error
    end
    @test malformed_error isa LL.NativeError
    @test malformed_error.status == Int32(LL.STATUS_INVALID_UTF8)
    @test occursin("malformed UTF-8", malformed_error.message)
    @test @inferred(LL.distance("abc", "axc")) == 1
    @test @inferred(LL.distance(UInt8[1, 2], UInt8[1, 3])) == 1
    @test @inferred(LL.distance(UInt64[1, 2], UInt64[2, 1])) == 2

    sequences = [Int[]]
    for _ in 1:3
        prefixes = copy(sequences)
        append!(sequences, [vcat(prefix, unit) for prefix in prefixes for unit in 0:2])
        unique!(sequences)
    end
    families = (
        LL.distance,
        LL.optimal_string_alignment_distance,
        LL.true_damerau_distance,
        LL.merge_and_split_distance,
    )
    for source in sequences, target in sequences, family in families
        text_source = String(Char.('a' .+ source))
        text_target = String(Char.('a' .+ target))
        byte_source = UInt8.(source)
        byte_target = UInt8.(target)
        token_source = UInt64.(source)
        token_target = UInt64.(target)
        exact = family(text_source, text_target)
        @test family(byte_source, byte_target) == exact
        @test family(token_source, token_target) == exact
        for threshold in 0:3
            expected = exact <= threshold ? exact : nothing
            @test family(text_source, text_target; threshold=threshold) === expected
            @test family(byte_source, byte_target; threshold=threshold) === expected
            @test family(token_source, token_target; threshold=threshold) === expected
        end
    end

    binary = UInt8[0xff, 0x00, 0x80]
    @test LL.distance(binary, reverse(binary)) == 2
    view_source = @view binary[1:2]
    @test LL.distance(view_source, UInt8[0xff, 0x01]) == 1
    @test_throws MethodError LL.distance([1, 2], [1, 3])
end

if LL.build_features() & LL.BUILD_FEATURE_PHONETIC != 0
    @testset "phonetic objects" begin
        pattern = LL.PhoneticPattern("cat")
        try
            @test "cat" in pattern
            @test !("cot" in pattern)
            @test all(>(0), size(pattern))
        finally
            LL.close!(pattern)
        end
        @test !isopen(pattern)

        rules = LL.PhoneticRuleSet(LL.RULES_ENGLISH_ORTHOGRAPHY)
        try
            @test length(rules) > 0
            @test rules("KNIGHT") isa String
        finally
            LL.close!(rules)
        end
    end

    @testset "native phonetic analysis" begin
        @test LL.articulatory_distance('p', 'p') == 0.0
        @test 0.0 < LL.articulatory_distance('p', 'b') <
            LL.articulatory_distance('p', 'h')
        weights = LL.PhoneticFeatureWeights(voicing=0.35)
        @test LL.articulatory_distance('p', 'b'; weights=weights) >
            LL.articulatory_distance('p', 'b')
        for source in ("", "phone", "café", "🦀a"),
            target in ("", "fone", "cafe", "🦀b")
            @test LL.articulatory_edit_distance(source, target) >= 0.0
            @test LL.articulatory_edit_distance(source, target) ==
                LL.articulatory_edit_distance(target, source)
            @test LL.articulatory_edit_distance(source, target;
                weights=LL.PhoneticFeatureWeights()) ==
                LL.articulatory_edit_distance(source, target)
        end
        @test LL.articulatory_edit_distance("p", "b") ==
            LL.articulatory_distance('p', 'b')
        @test LL.articulatory_edit_distance("p", "b"; weights=weights) ==
            LL.articulatory_distance('p', 'b'; weights=weights)
        @test LL.syllable_count("happy") == 2
        @test LL.syllable_boundaries("happy") == [0, 3]
        @test LL.syllable_count("ˈhæp.i"; ipa=true) == 2
        @test LL.syllable_boundaries("ˈhæp.i"; ipa=true) == [0, 5]
        @test isempty(LL.syllable_boundaries(""))
        @test LL.syllable_count("") == 0
        @test_throws ArgumentError LL.PhoneticFeatureWeights(voicing=-1)
        @test_throws ArgumentError LL.PhoneticFeatureWeights(voicing=NaN)
        @test_throws ArgumentError LL.articulatory_edit_distance("a", "b";
            max_cells=0)
        failure = try
            LL.articulatory_edit_distance("abc", "def"; max_cells=8)
            nothing
        catch error
            error
        end
        @test failure isa LL.NativeError
        @test failure.status == Int32(LL.STATUS_LIMIT_EXCEEDED)
        @test_throws LL.NativeError LL.syllable_count("happy";
            max_input_scalars=4)
    end

    @testset "native phonetic grep" begin
        grep = LL.PhoneticGrep("phone"; max_distance=1)
        try
            @test LL.distance_config(grep) == (effective=1, local_override=nothing)
            @test LL.match_distance(grep, "phone") == 0
            @test LL.match_distance(grep, "phon") == 1
            @test LL.match_distance(grep, "tablet") === nothing
            @test "phon" in grep
            @test ! ("tablet" in grep)
            line_matches = LL.scan_line(grep, "phone phon")
            @test [(m.text, m.line_number, m.start_byte, m.end_byte, m.distance)
                for m in line_matches] ==
                [("phone", 1, 0, 5, 0), ("phon", 1, 6, 10, 1)]
            text_matches = LL.scan_text(grep,
                "exact phone\nnear phon\nunrelated tablet")
            @test [(m.text, m.line_number, m.start_byte, m.end_byte, m.distance)
                for m in text_matches] ==
                [("phone", 1, 6, 11, 0), ("phon", 2, 5, 9, 1)]
            @test_throws LL.NativeError LL.scan_line(grep, "phone phon";
                max_matches=1)
            @test_throws LL.NativeError LL.match_distance(grep, "phone";
                max_candidate_bytes=4)
        finally
            LL.close!(grep)
        end
        @test !isopen(grep)
        @test_throws LL.NativeError LL.match_distance(grep, "phone")

        rules = LL.PhoneticRuleSet("ph -> f;")
        normalized = LL.PhoneticGrep("fone"; rules, max_distance=0)
        LL.close!(rules)
        try
            @test LL.match_distance(normalized, "phone") == 0
            @test LL.match_distance(normalized, "fone") == 0
            @test [m.text for m in LL.scan_line(normalized, "phone fone")] ==
                ["phone", "fone"]
        finally
            LL.close!(normalized)
        end

        folded = LL.PhoneticGrep("hello"; case_insensitive=true)
        try
            @test LL.match_distance(folded, "HELLO") == 0
        finally
            close(folded)
        end
        transposed = LL.PhoneticGrep("phone"; max_distance=1,
            algorithm=LL.ALGORITHM_TRANSPOSITION)
        try
            @test LL.match_distance(transposed, "phoen") == 1
        finally
            close(transposed)
        end
        @test_throws ArgumentError LL.PhoneticGrep("x"; max_distance=256)
        @test_throws LL.NativeError LL.PhoneticGrep("(")
    end

    @testset "native phonetic-normalized dictionaries" begin
        terms = ["phone", "fone", "bone", "café", "écho", "phone"]
        for compact in (false, true)
            dictionary = LL.PhoneticNormalizedDictionary(terms; compact)
            try
                candidates = LL.query(dictionary, "fone"; max_distance=0)
                @test Set(candidate.term for candidate in candidates) ==
                    Set(["phone", "fone"])
                @test all(candidate -> candidate.distance == 0, candidates)
                @test all(candidate -> candidate.normalized_form isa String, candidates)
                @test all(candidate -> candidate.term isa String, candidates)
                @test LL.query(dictionary, "🦀"; max_distance=0) == LL.PhoneticCandidate[]
                @test_throws LL.NativeError LL.query(dictionary, "fone";
                    max_results=1)
                @test_throws LL.NativeError LL.query(dictionary, "fone";
                    max_query_scalars=3)
                if compact
                    @test_throws LL.NativeError LL.insert!(dictionary, "phoen")
                else
                    @test LL.insert!(dictionary, "phoen")
                    @test !LL.insert!(dictionary, "phoen")
                    @test LL.remove!(dictionary, "phoen")
                    @test !LL.remove!(dictionary, "phoen")
                    @test push!(dictionary, "phoen") === dictionary
                    @test delete!(dictionary, "phoen") === dictionary
                end
            finally
                close(dictionary)
            end
            @test !isopen(dictionary)
            @test_throws LL.NativeError LL.query(dictionary, "fone")
        end
        @test_throws ArgumentError LL.PhoneticNormalizedDictionary("phone")
        @test_throws LL.NativeError LL.PhoneticNormalizedDictionary(["phone"];
            max_total_bytes=4)
        rules = LL.PhoneticRuleSet("ph -> f;")
        dictionary = LL.PhoneticNormalizedDictionary(["phone", "fone"]; rules)
        close(rules)
        try
            @test Set(candidate.term for candidate in LL.query(dictionary, "fone")) ==
                Set(["phone", "fone"])
        finally
            close(dictionary)
        end
    end

    @testset "native character-level phonetic grep and streaming" begin
        grep = LL.PhoneticOnlineGrep("café")
        try
            @test LL.normalized_query(grep) == "café"
            expected = LL.scan(grep, "🦀 café café")
            @test [item.original_text for item in expected] == ["café", "café"]
            @test [item.byte_range for item in expected] == [(5, 10), (11, 16)]
            @test [item.char_range for item in expected] == [(2, 6), (7, 11)]
            @test isempty(LL.scan(grep, "unrelated"))
            stream = LL.streaming(grep; max_total_bytes=64)
            close(grep)
            try
                @test LL.feed!(stream, "🦀 ca") === stream
                LL.feed!(stream, "fé ca")
                LL.feed!(stream, "fé")
                @test [(item.original_text, item.byte_range, item.char_range)
                    for item in LL.finish!(stream)] ==
                    [(item.original_text, item.byte_range, item.char_range)
                        for item in expected]
                @test_throws ArgumentError LL.finish!(stream)
                @test_throws ArgumentError LL.feed!(stream, "x")
            finally
                close(stream)
            end
        finally
            close(grep)
        end
        rules = LL.PhoneticRuleSet("ph -> f;")
        online = LL.PhoneticOnlineGrep("phone"; rules)
        close(rules)
        try
            @test LL.normalized_query(online) == "fone"
            @test Set(item.original_text for item in LL.scan(online, "fone phone")) ==
                Set(["fone", "phone"])
            @test_throws LL.NativeError LL.scan(online, "fone phone";
                max_input_bytes=9)
            @test_throws LL.NativeError LL.scan(online, "fone phone";
                max_matches=1)
        finally
            close(online)
        end
    end
    @testset "native token-query phonetic grep" begin
        grep = LL.PhoneticTokenGrep("hello world"; default_distance=1)
        try
            matches = LL.scan(grep, "helo wrld and hello world")
            @test length(matches) == 2
            @test matches[1].matched_text == "helo wrld"
            @test matches[1].total_distance == 2
            @test matches[1].byte_range == (0, 9)
            @test length(matches[1].details) == 2
            @test [detail.original_text for detail in matches[1].details] ==
                ["helo", "wrld"]
            @test [detail.distance for detail in matches[1].details] == [1, 1]
            @test matches[2].matched_text == "hello world"
            @test_throws LL.NativeError LL.scan(grep, "helo wrld";
                max_input_bytes=8)
            @test_throws LL.NativeError LL.scan(grep, "helo wrld";
                max_details=1)
        finally
            close(grep)
        end
        rules = LL.PhoneticRuleSet("ph -> f;")
        normalized = LL.PhoneticTokenGrep("fone"; rules)
        close(rules)
        try
            @test [match.matched_text for match in LL.scan(normalized, "phone")] ==
                ["phone"]
        finally
            close(normalized)
        end
        @test_throws LL.NativeError LL.PhoneticTokenGrep("(")
        @test_throws ArgumentError LL.PhoneticTokenGrep("a";
            max_query_bytes=0)
    end
    @testset "native incremental phonetic rewriting" begin
        rules = LL.PhoneticRuleSet("ph -> f;")
        transducer = LL.PhoneticTransducer(; rules)
        close(rules)
        try
            @test LL.normalize(transducer, "🦀 phone") == "🦀 fone"
            parts = [LL.feed!(transducer, "🦀 p"),
                LL.feed!(transducer, "hone"), LL.finish!(transducer)]
            @test join(parts) == "🦀 fone"
            @test_throws ArgumentError LL.feed!(transducer, "x")
            @test LL.reset!(transducer) === transducer
            @test LL.feed!(transducer, "phone") * LL.finish!(transducer) == "fone"
            LL.reset!(transducer)
            @test_throws LL.NativeError LL.feed!(transducer, "abcdefgh";
                max_output_bytes=1)
            @test LL.normalize(transducer, "phone") == "fone"
        finally
            close(transducer)
        end
        @test_throws LL.NativeError LL.normalize(transducer, "phone")
    end
    @testset "bounded reverse phonetic expansion" begin
        rules = LL.PhoneticRuleSet("ph -> f;")
        try
            pattern = LL.expand_phonetic_alternatives("fone"; rules)
            @test occursin("ph", pattern)
            @test occursin("f", pattern)
            @test LL.expand_phonetic_with_costs("fone"; rules).pattern isa String
            @test LL.expand_phonetic_with_costs("fone"; rules).max_cost >= 0
            @test LL.expand_phonetic_alternatives(""; rules) == ""
            @test_throws LL.NativeError LL.expand_phonetic_alternatives("fone";
                rules, limits=LL.PhoneticExpansionLimits(max_nodes=1))
            @test_throws LL.NativeError LL.expand_phonetic_with_costs("fone";
                rules, limits=LL.PhoneticExpansionLimits(max_output_bytes=2))
        finally
            close(rules)
        end
        @test_throws ArgumentError LL.PhoneticExpansionLimits(max_nodes=0)
    end
    @testset "native IPA feature classification" begin
        @test :Voiced in LL.phonetic_features('b')
        @test :Stop in LL.phonetic_features('p')
        @test isempty(LL.phonetic_features('🦀'))
        @test 'b' in LL.characters_with_features([:Voiced, :Stop])
        @test 'p' in LL.characters_with_features([:Voiceless, :Stop])
        @test 'p' in LL.similar_phonetic_chars('b')
        @test 'p' ∉ LL.similar_phonetic_chars('p')
        @test LL.voicing_pair('p') == 'b'
        @test LL.are_phonetically_similar('p', 'b')
        @test LL.is_free_phonetic_substitution('p', 'b')
        @test !LL.is_free_phonetic_substitution('p', 'h')
        @test LL.expand_feature_based('p') isa Vector{Char}
        @test LL.feature_set_distance([:Voiced], [:Voiceless]) >= 0
        @test_throws ArgumentError LL.characters_with_features([:Unknown])
    end
    @testset "native phonetic file loaders and versioned AOT" begin
        fixtures = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
            "tests", "fixtures", "phonetic_binding"))
        rules = LL.load_phonetic_rules(joinpath(fixtures, "rules.llev"))
        pattern = LL.load_phonetic_pattern(joinpath(fixtures, "pattern.llre"))
        try
            @test rules("phone") == "fone"
            @test "phone" in pattern
            @test !("café" in pattern)
            @test_throws LL.NativeError LL.load_phonetic_rules(
                joinpath(fixtures, "missing.llev"))
            @test_throws LL.NativeError LL.load_phonetic_rules(
                joinpath(fixtures, "rules.llev"); max_total_path_bytes=2)
            @test_throws ArgumentError LL.load_phonetic_rules(
                joinpath(fixtures, "rules.llev"); search_paths=fill("x", 65))
            if LL.build_features() & LL.BUILD_FEATURE_PHONETIC_AOT != 0
                rule_bytes = LL.compiled_phonetic_bytes(rules)
                pattern_bytes = LL.compiled_phonetic_bytes(pattern)
                @test !isempty(rule_bytes)
                @test !isempty(pattern_bytes)
                restored_rules = LL.load_compiled_phonetic_rules(rule_bytes)
                restored_pattern = LL.load_compiled_phonetic_pattern(pattern_bytes)
                try
                    @test restored_rules("phone") == rules("phone")
                    @test ("phone" in restored_pattern) == ("phone" in pattern)
                    @test !("café" in restored_pattern)
                finally
                    close(restored_rules)
                    close(restored_pattern)
                end
                @test_throws LL.NativeError LL.compiled_phonetic_bytes(rules;
                    max_output_bytes=1)
                @test_throws LL.NativeError LL.load_compiled_phonetic_rules(
                    rule_bytes; max_input_bytes=length(rule_bytes) - 1)
                corrupted = copy(rule_bytes)
                corrupted[1] = xor(corrupted[1], UInt8(0xff))
                @test_throws LL.NativeError LL.load_compiled_phonetic_rules(corrupted)
                wrong_version = copy(rule_bytes)
                wrong_version[5] = xor(wrong_version[5], UInt8(0xff))
                @test_throws LL.NativeError LL.load_compiled_phonetic_rules(wrong_version)
                @test_throws LL.NativeError LL.load_compiled_phonetic_pattern(
                    UInt8[0x00, 0x01])
            else
                @test_throws LL.NativeError LL.compiled_phonetic_bytes(rules)
                @test_throws LL.NativeError LL.load_compiled_phonetic_rules(UInt8[])
            end
        finally
            close(rules)
            close(pattern)
        end
    end
    @testset "finite phonetic cross-surface properties" begin
        # Exhaust the 4^3 short spellings rather than relying on a random seed.
        cases = [join(parts) for parts in Iterators.product(
            ("", "p", "h", "f"), ("", "p", "h", "f"), ("", "p", "h", "f"))]
        rules = LL.PhoneticRuleSet("ph -> f;")
        mutable = LL.PhoneticNormalizedDictionary(cases; rules)
        compact = LL.PhoneticNormalizedDictionary(cases; rules, compact=true)
        rewrite = LL.PhoneticTransducer(; rules)
        grep = LL.PhoneticOnlineGrep("f"; rules)
        try
            for word in cases
                left = [(c.term, c.distance, c.normalized_form)
                    for c in LL.query(mutable, word; max_distance=1)]
                right = [(c.term, c.distance, c.normalized_form)
                    for c in LL.query(compact, word; max_distance=1)]
                @test left == right

                expected = LL.normalize(rewrite, word)
                LL.reset!(rewrite)
                prefix = isempty(word) ? "" : string(first(word))
                suffix = isempty(word) ? "" : word[nextind(word, firstindex(word)):end]
                emitted = LL.feed!(rewrite, prefix) *
                    LL.feed!(rewrite, suffix) * LL.finish!(rewrite)
                @test emitted == expected
                LL.reset!(rewrite)

                baseline = [(m.original_text, m.byte_range, m.char_range, m.distance)
                    for m in LL.scan(grep, word)]
                stream = LL.streaming(grep; max_total_bytes=64)
                try
                    LL.feed!(stream, word)
                    streamed = [(m.original_text, m.byte_range, m.char_range, m.distance)
                        for m in LL.finish!(stream)]
                    @test streamed == baseline
                finally
                    close(stream)
                end
            end
        finally
            close(grep)
            close(rewrite)
            close(compact)
            close(mutable)
            close(rules)
        end
    end
end

@testset "borrow expiration" begin
    storage = [LL.RawMatch(C_NULL, 0, 0, 0, 0, UInt32(2), 0, (0x00, 0x00, 0x00))]
    GC.@preserve storage begin
        batch = LL.BorrowedBatch(pointer(storage), 1, true)
        match = batch[1]
        @test match.distance == 0
        batch.active = false
        @test_throws ArgumentError match.distance
    end
end
