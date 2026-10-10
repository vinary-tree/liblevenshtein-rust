@testset "bounded native-equivalent temporal delta encoding" begin
    series = [1.0, 3.0, 6.0, 10.0]
    @test collect(LL.compute_deltas(series)) == [2.0, 3.0, 4.0]
    @test collect(LL.reconstruct_from_deltas(1.0,
        [2.0, 3.0, 4.0])) == series
    @test collect(LL.compute_deltas(Float64[])) == Float64[]
    @test collect(LL.compute_deltas([1.0])) == Float64[]
    @test collect(LL.reconstruct_from_deltas(1.0,
        Float64[])) == [1.0]

    config = LL.QuantizationConfig(-10.0, 10.0, 256)
    initial, encoded = LL.encode_deltas_u8(series, config)
    words = collect(encoded)
    @test initial == 1.0
    @test words == collect(LL.encode_u8(config, [2.0, 3.0, 4.0]))
    decoded = collect(LL.decode_deltas_u8(initial, words, config))
    @test length(decoded) == length(series)
    @test all(abs.(decoded .- series) .< 0.5)
    @test LL.encode_deltas_u8(Float64[], config)[1] == 0.0
    @test isempty(collect(LL.encode_deltas_u8(Float64[], config)[2]))
    @test collect(LL.decode_deltas_u8(1.0, UInt8[], config)) == [1.0]

    pending = LL.compute_deltas(series; max_samples=4)
    series[2] = 100.0
    @test collect(pending) == [2.0, 3.0, 4.0]
    @test_throws ArgumentError LL.compute_deltas(
        UntouchableLargeVector(4); max_samples=3)
    @test_throws ArgumentError LL.encode_deltas_u8(
        UntouchableLargeVector(4), config; max_samples=3)
    @test_throws ArgumentError LL.reconstruct_from_deltas(0.0,
        UntouchableLargeVector(4); max_deltas=3)
    @test_throws ArgumentError LL.encode_deltas_u8([1.0],
        LL.quantizer_u16(-1.0, 1.0))
end
