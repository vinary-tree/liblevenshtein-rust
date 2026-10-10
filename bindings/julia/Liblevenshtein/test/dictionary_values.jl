@testset "native value-preserving Bincode" begin
    for domain in (:byte, :unicode)
        for (kind, source) in ((:u64, ["cab" => UInt64(7), "café" => UInt64(9)]),
            (:bytes, ["cab" => UInt8[0, 255], "café" => UInt8[]]))
            encoded = valued_dictionary_bytes(source;
                value_kind=kind, unit_domain=domain,
                max_entries=4, max_term_bytes=16,
                max_total_term_bytes=32, max_value_bytes=16,
                max_total_value_bytes=32, max_payload_bytes=256)
            @test !isempty(encoded)
            snapshot = valued_dictionary_entries(encoded;
                value_kind=kind, unit_domain=domain,
                max_entries=4, max_term_bytes=16,
                max_total_term_bytes=32, max_value_bytes=16,
                max_total_value_bytes=32, max_payload_bytes=256)
            try
                @test length(snapshot) == 2
                @test collect(snapshot) == source
            finally
                close(snapshot)
            end
            @test_throws ArgumentError collect(snapshot)
            @test_throws Liblevenshtein.NativeError valued_dictionary_entries(
                encoded[1:end-1]; value_kind=kind, unit_domain=domain,
                max_entries=4, max_term_bytes=16,
                max_total_term_bytes=32, max_value_bytes=16,
                max_total_value_bytes=32, max_payload_bytes=256)
            @test_throws ArgumentError valued_dictionary_entries(encoded;
                value_kind=kind, unit_domain=domain, max_payload_bytes=8)
        end
    end
    @test_throws ArgumentError valued_dictionary_bytes(["a" => -1]; value_kind=:u64)
    @test_throws ArgumentError valued_dictionary_bytes(["a" => "text"];
        value_kind=:bytes)
    @test_throws ArgumentError valued_dictionary_bytes(["a" => UInt64(1)];
        value_kind=:u64, unit_domain=:unknown)
    @test_throws Liblevenshtein.NativeError valued_dictionary_bytes(
        ["a" => UInt64(1)]; value_kind=:u64, format_version=2)
end
