@testset "native suffix source persistence" begin
    texts = ["cab", "café", "cab"]
    for format in (:bincode_v1, :protobuf_v1)
        encoded = LL.suffix_source_bytes(texts; format,
            max_terms=3, max_term_bytes=8,
            max_total_term_bytes=16, max_payload_bytes=512)
        snapshot = LL.suffix_source_texts(encoded; format,
            max_terms=3, max_term_bytes=8,
            max_total_term_bytes=16, max_payload_bytes=512)
        try
            @test collect(snapshot) == texts
        finally
            close(snapshot)
        end
        @test_throws ArgumentError iterate(snapshot)
        @test_throws LL.NativeError LL.suffix_source_texts(encoded[1:end-1]; format)
    end
    @test_throws ArgumentError LL.suffix_source_bytes(texts; format=:unknown)
    @test_throws ArgumentError LL.suffix_source_bytes(texts; max_terms=2)
    @test_throws LL.NativeError LL.suffix_source_bytes(texts; max_payload_bytes=8)
end
