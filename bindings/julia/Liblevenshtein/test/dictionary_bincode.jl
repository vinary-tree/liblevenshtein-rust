@testset "bounded native dictionary bincode" begin
    @test LL.api_revision() >= 32
    @test (LL.build_features() & LL.BUILD_FEATURE_SERIALIZATION) != 0

    source = Dict("cab" => 1, "ab" => 2, "café" => 3)
    bytes = LL.bincode_dictionary_bytes(source;
        max_terms=3, max_term_bytes=8, max_total_term_bytes=16,
        max_payload_bytes=128)
    @test !isempty(bytes)
    decoded = LL.bincode_dictionary_terms(bytes;
        max_terms=3, max_term_bytes=8, max_total_term_bytes=16,
        max_payload_bytes=128)
    try
        @test length(decoded) == 3
        @test collect(decoded) == ["ab", "cab", "café"]
        @test collect(decoded) == ["ab", "cab", "café"]
    finally
        close(decoded)
    end
    @test_throws ArgumentError iterate(decoded)
    close(decoded)

    @test_throws LL.NativeError LL.bincode_dictionary_bytes(keys(source);
        format_version=2)
    @test_throws LL.NativeError LL.bincode_dictionary_terms(bytes;
        format_version=2)
    @test_throws ArgumentError LL.bincode_dictionary_bytes(keys(source);
        max_terms=2)
    @test_throws ArgumentError LL.bincode_dictionary_bytes(keys(source);
        max_term_bytes=3)
    @test_throws LL.NativeError LL.bincode_dictionary_bytes(keys(source);
        max_payload_bytes=8)
    @test_throws ArgumentError LL.bincode_dictionary_terms(bytes;
        max_payload_bytes=8)
    @test_throws LL.NativeError LL.bincode_dictionary_terms(bytes[1:end-1])
    @test_throws LL.NativeError LL.bincode_dictionary_terms(vcat(bytes, 0xff))
    @test_throws LL.NativeError LL.bincode_dictionary_terms(UInt8[0x01, 0x02])
end
