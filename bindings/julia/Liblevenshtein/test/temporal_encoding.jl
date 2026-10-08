@testset "lossless temporal float encoding" begin
    patterns32 = UInt32[0x0000_0000, 0x8000_0000, 0x3f80_0000,
        0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234]
    values32 = reinterpret.(Float32, patterns32)
    @test LL.encode_f32.(values32) == patterns32
    @test reinterpret.(UInt32, LL.decode_f32.(patterns32)) == patterns32
    @test collect(LL.encode_f32_series(values32)) == patterns32
    @test reinterpret.(UInt32,
        collect(LL.decode_f32_series(patterns32))) == patterns32

    patterns64 = UInt64[0x0000_0000_0000_0000,
        0x8000_0000_0000_0000, 0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000, 0x7ff8_0000_0000_1234]
    values64 = reinterpret.(Float64, patterns64)
    @test LL.encode_f64.(values64) == patterns64
    @test reinterpret.(UInt64, LL.decode_f64.(patterns64)) == patterns64
    words = collect(LL.encode_f64_series_as_u32_pairs(values64))
    @test words == UInt32[0, 0, 0x8000_0000, 0,
        0x3ff0_0000, 0, 0xbff0_0000, 0,
        0x7ff8_0000, 0x1234]
    @test reinterpret.(UInt64,
        collect(LL.decode_u32_pairs_to_f64(words))) == patterns64
    @test length(collect(LL.decode_u32_pairs_to_f64(
        UInt32[0x3ff0_0000, 0, 0x1234]))) == 1

    ordered = Float32[-Inf, -1, -0.0, 0.0, 1, Inf]
    keys = LL.encode_f32_total_order.(ordered)
    @test issorted(keys)
    @test reinterpret.(UInt32,
        LL.decode_f32_total_order.(keys)) ==
        reinterpret.(UInt32, ordered)
    @test LL.encode_f32_ordered(1.0f0) == 0x3f80_0000
    @test_throws ArgumentError LL.encode_f32_ordered(-1.0f0)
    @test_throws ArgumentError LL.encode_f32_ordered(Float32(NaN))

    input = Float32[1, 2]
    lazy = LL.encode_f32_series(input; max_samples=2)
    input[1] = 10
    @test collect(lazy) == UInt32[0x3f80_0000, 0x4000_0000]
    @test_throws ArgumentError LL.encode_f32_series(
        UntouchableLargeVector(4); max_samples=3)
    @test_throws ArgumentError LL.decode_u32_pairs_to_f64(
        words; max_words=length(words) - 1)
end
