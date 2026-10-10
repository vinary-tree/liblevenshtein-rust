@testset "bounded native-equivalent temporal quantization" begin
    config = LL.QuantizationConfig(0.0, 100.0, 100)
    @test config.num_bins == 100
    @test LL.bin_width(config) == 1.0
    @test LL.max_error(config) == 0.5
    @test LL.quantize(config, 0.0) == 0
    @test LL.quantize(config, 50.0) == 50
    @test LL.quantize(config, 99.9) == 99
    @test LL.quantize(config, 100.0) == 99
    @test LL.quantize(config, NaN) == 0
    @test LL.quantize(config, -Inf) == 0
    @test LL.quantize(config, Inf) == 99
    @test LL.quantize(config, -floatmax(Float64)) == 0
    @test LL.quantize(config, floatmax(Float64)) == 99
    @test LL.dequantize(config, 50) == 50.5
    @test LL.bin_bounds(config, 50) == (50.0, 51.0)
    @test LL.bin_bounds(config, 0) == (-Inf, 1.0)
    @test LL.bin_bounds(config, 99) == (99.0, Inf)
    @test LL.bin_bounds(config, 1_000) == (99.0, Inf)
    @test LL.bin_bounds(LL.QuantizationConfig(0.0, 100.0, 1), 0) ==
        (-Inf, Inf)
    @test LL.value_diff_to_bins(config, 10.0) == 10
    @test LL.value_diff_to_bins(config, 0.5) == 1
    @test LL.value_diff_to_bins(config, NaN) == typemax(UInt32)
    @test LL.value_diff_to_bins(config, Inf) == typemax(UInt32)
    @test LL.try_uniform_quantizer(NaN, 1.0, 10) === nothing
    @test LL.try_uniform_quantizer(0.0, Inf, 10) === nothing
    @test LL.try_uniform_quantizer(1.0, 1.0, 10) === nothing
    @test LL.try_uniform_quantizer(0.0, 1.0, 0) === nothing
    @test LL.try_uniform_quantizer(-floatmax(Float64),
        floatmax(Float64), 10) === nothing
    @test_throws ArgumentError LL.QuantizationConfig(0.0, 1.0, 0)

    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    fitted = LL.quantizer_from_data(data, 256, 0.1)
    @test fitted !== nothing
    @test fitted.min_value == 6.0
    @test fitted.max_value == 54.0
    @test LL.quantizer_from_data([NaN, Inf], 256, 0.1) === nothing
    @test LL.quantizer_from_data([10.0, 20.0, 30.0],
        10, -0.25).min_value == 15.0

    byte_config = LL.quantizer_u8(0.0, 100.0)
    @test LL.quantize_u8(byte_config, 100.0) == typemax(UInt8)
    @test LL.quantize_u16(LL.quantizer_u16(0.0, 100.0),
        100.0) == typemax(UInt16)
    @test_throws ArgumentError LL.quantize_u8(
        LL.quantizer_u16(0.0, 100.0), 0.0)

    series = [0.0, 25.0, 50.0, 75.0, 100.0]
    encoded = collect(LL.encode_u8(byte_config, series))
    @test encoded == UInt8[0, 64, 128, 192, 255]
    @test length(collect(LL.decode_u8(byte_config, encoded))) == 5
    @test collect(LL.encode_u32(config, series)) ==
        UInt32[0, 25, 50, 75, 99]
    @test collect(LL.decode_u32(config, UInt32[0, 50, 99])) ==
        [0.5, 50.5, 99.5]
    snapshot = LL.encode_u8(byte_config, series; max_samples=5)
    series[1] = 100.0
    @test first(snapshot) == 0
    @test_throws ArgumentError LL.encode_u32(config,
        UntouchableLargeVector(4); max_samples=3)
    @test_throws ArgumentError LL.quantizer_from_data(
        UntouchableLargeVector(4), 10, 0.0; max_samples=3)
end
