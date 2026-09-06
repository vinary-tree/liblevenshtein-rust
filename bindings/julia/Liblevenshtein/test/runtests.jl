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
    @test sizeof(LL.RawMatch) == 48
    @test sizeof(LL.RawBatch) == 24
    @test sizeof(LL.RawQueryCacheStats) == 64
    @test sizeof(LL.OwnedString) == 16
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
