@testset "native generalized operation-set persistence" begin
    source = GeneralizedOperationSet([
        GeneralizedOperation(1, 1, 0, "match";
            applicability=APPLICABILITY_EQUAL),
        GeneralizedOperation(2, 1, 0.25, "digraph";
            applicability=APPLICABILITY_LISTED, restrictions=["ph" => "f"]),
    ])
    for format in (:binary_v1, :protobuf_v1, :gzip_binary_v1,
        :gzip_protobuf_v1)
        bytes = operation_set_bytes(source; format)
        @test !isempty(bytes)
        snapshot = operation_set_snapshot(bytes; format)
        try
            @test length(snapshot) == 2
            operations = collect(snapshot)
            @test operations[1].name == "match"
            @test operations[1].applicability == APPLICABILITY_EQUAL
            @test operations[2].consume_source == 2
            @test operations[2].consume_target == 1
            @test operations[2].weight == 0.25
            @test length(operations[2].restrictions) == 1
            @test only(operations[2].restrictions).source == "ph"
            @test only(operations[2].restrictions).target == "f"
            @test length(GeneralizedOperationSet(snapshot)) == 2
            @test operation_set_bytes(snapshot; format) == bytes
            @test_throws Liblevenshtein.NativeError operation_set_bytes(snapshot; format,
                max_payload_bytes=1)
        finally
            close(snapshot)
        end
        @test_throws ArgumentError collect(snapshot)
        @test_throws Liblevenshtein.NativeError operation_set_snapshot(bytes[1:end-1]; format)
        @test_throws ArgumentError operation_set_snapshot(bytes; format,
            max_payload_bytes=1)
    end
    @test_throws ArgumentError operation_set_bytes(source; format=:unknown)
    @test_throws Liblevenshtein.NativeError operation_set_bytes(source; max_operations=1)
    @test_throws Liblevenshtein.NativeError operation_set_snapshot(UInt8[0x00]; format=:binary_v1)
end
