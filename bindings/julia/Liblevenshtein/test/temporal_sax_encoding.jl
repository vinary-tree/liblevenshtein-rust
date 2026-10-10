@testset "bounded native-equivalent SAX encoding" begin
    for size in 2:10
        @test length(LL.sax_breakpoints(size)) == size - 1
    end
    @test LL.sax_breakpoints(1) === nothing
    @test LL.sax_breakpoints(11) === nothing
    first_breakpoints = LL.sax_breakpoints(4)
    first_breakpoints[1] = 100.0
    @test LL.sax_breakpoints(4) == [-0.67, 0.0, 0.67]

    normalized = collect(LL.sax_normalize([1.0, 2.0, 3.0,
        4.0, 5.0]))
    @test abs(sum(normalized) / length(normalized)) < 1e-9
    @test abs(sum(abs2, normalized) / length(normalized) - 1.0) < 0.1
    @test collect(LL.sax_normalize([7.0, 7.0])) == [0.0, 0.0]
    @test collect(LL.sax_normalize(Float64[])) == Float64[]
    @test all(isnan, collect(LL.sax_normalize([NaN])))

    @test collect(LL.sax_paa([1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0], 4)) == [1.5, 3.5, 5.5, 7.5]
    @test collect(LL.sax_paa([1.0, 2.0, 3.0], 8)) ==
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    @test isempty(collect(LL.sax_paa([1.0], 0)))
    @test isempty(collect(LL.sax_paa(Float64[], 4)))

    word = collect(LL.sax_encode([1.0, 2.0, 3.0, 4.0,
        5.0, 4.0, 3.0, 2.0], 4, 4))
    @test word == UInt8[0, 2, 3, 1]
    @test collect(LL.sax_encode([1.0, 2.0, 3.0], 8, 4)) ==
        UInt8[0, 0, 0, 2, 2, 2, 3, 3]
    @test collect(LL.sax_encode([7.0, 7.0], 2, 4)) ==
        UInt8[2, 2]
    @test collect(LL.sax_encode([NaN], 1, 4)) == UInt8[0]
    @test isempty(collect(LL.sax_encode([1.0], 1, 11)))

    @test LL.sax_mindist(UInt8[0, 1, 2, 3],
        UInt8[0, 1, 2, 3], 100, 4) == 0.0
    @test LL.sax_mindist(UInt8[0, 1, 2, 3],
        UInt8[1, 2, 3, 3], 100, 4) == 0.0
    @test LL.sax_mindist(UInt8[0, 3], UInt8[3, 0],
        8, 4) ≈ 2 * sqrt(2) * 1.34
    @test LL.sax_mindist(UInt8[], UInt8[], 1, 4) == Inf
    @test LL.sax_mindist(UInt8[0], UInt8[0, 1], 1, 4) == Inf
    @test LL.sax_mindist(UInt8[0], UInt8[0], 1, 11) == Inf

    series = [1.0, 2.0, 3.0]
    pending = LL.sax_paa(series, 3; max_samples=3,
        max_segments=3)
    series[1] = 100.0
    @test collect(pending) == [1.0, 2.0, 3.0]
    @test_throws ArgumentError LL.sax_paa(
        UntouchableLargeVector(4), 2; max_samples=3)
    @test_throws ArgumentError LL.sax_encode([1.0], 4, 4;
        max_segments=3)
    @test_throws ArgumentError LL.sax_mindist(
        UInt8[0, 1], UInt8[0, 1], 2, 4; max_word_len=1)
end
