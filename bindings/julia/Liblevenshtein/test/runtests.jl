using Test
using Liblevenshtein
using Libdictenstein
import VinaryTreeInterop

const LL = Liblevenshtein

@testset "ABI identity and layouts" begin
    @test LL.abi_version() == LL.ABI_VERSION == 1
    @test LL.api_revision() >= LL.API_REVISION == 5
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
