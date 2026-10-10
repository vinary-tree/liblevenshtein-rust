@testset "RC.5 native serialization compatibility" begin
    root = joinpath(@__DIR__, "fixtures", "serialization_rc5")
    dictionary_formats = (
        (:bincode_v1, "dictionary-bincode-v1.bin"),
        (:protobuf_v1, "dictionary-protobuf-v1.bin"),
        (:protobuf_v2, "dictionary-protobuf-v2.bin"),
        (:gzip_bincode_v1, "dictionary-gzip-bincode-v1.bin"),
        (:gzip_protobuf_v1, "dictionary-gzip-protobuf-v1.bin"),
        (:gzip_protobuf_v2, "dictionary-gzip-protobuf-v2.bin"),
        (:protobuf_dat_v1, "dictionary-dat-v1.bin"),
    )
    for (format, file) in dictionary_formats
        snapshot = dictionary_terms(read(joinpath(root, file)); format,
            max_terms=4, max_term_bytes=16, max_total_term_bytes=32,
            max_payload_bytes=256)
        try
            @test collect(snapshot) == ["", "cab", "café"]
        finally
            close(snapshot)
        end
    end
    for (format, file) in ((:bincode_v1, "suffix-bincode-v1.bin"),
        (:protobuf_v1, "suffix-protobuf-v1.bin"))
        snapshot = suffix_source_texts(read(joinpath(root, file)); format,
            max_terms=4, max_term_bytes=16, max_total_term_bytes=32,
            max_payload_bytes=256)
        try
            @test collect(snapshot) == ["banana", "bandana"]
        finally
            close(snapshot)
        end
    end
    for (format, file) in ((:binary_v1, "operation-binary-v1.bin"),
        (:protobuf_v1, "operation-protobuf-v1.bin"),
        (:gzip_binary_v1, "operation-gzip-binary-v1.bin"),
        (:gzip_protobuf_v1, "operation-gzip-protobuf-v1.bin"))
        bytes = read(joinpath(root, file))
        snapshot = operation_set_snapshot(bytes; format, max_payload_bytes=4096,
            max_operations=8, max_operation_name_bytes=64,
            max_restriction_pairs_per_operation=8,
            max_total_restriction_pairs=8, max_restriction_text_bytes=64)
        try
            operations = collect(snapshot)
            @test length(operations) == 3
            @test operations[1].name == "match"
            @test only(operations[2].restrictions) == SerializedByteRestriction(0xff, 0x80)
            @test only(operations[3].restrictions).source == "ph"
            @test only(operations[3].restrictions).target == "f"
            @test operation_set_bytes(snapshot; format, max_payload_bytes=4096,
                max_operations=8, max_operation_name_bytes=64,
                max_restriction_pairs_per_operation=8,
                max_total_restriction_pairs=8,
                max_restriction_text_bytes=64) == bytes
            @test_throws ArgumentError GeneralizedOperationSet(snapshot)
        finally
            close(snapshot)
        end
    end
end
