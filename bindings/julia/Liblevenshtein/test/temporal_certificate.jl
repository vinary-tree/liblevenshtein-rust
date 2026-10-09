@testset "bounded temporal range certificates" begin
    for kind in (:msm, :erp, :twed, :dtw, :frechet)
        index = LL.TemporalIndex(kind; quant_min=-10.0, quant_max=10.0,
            quant_bins=32, parameter0=kind === :msm ? 1.0 : 0.0,
            band=kind === :dtw ? 2 : 0, max_entries=3,
            max_total_samples=9, max_series_len=3)
        certificate = nothing
        try
            LL.insert!(index, 7, [1.0, 2.0, 3.0])
            LL.insert!(index, 8, [1.0, 2.5, 3.0])
            LL.freeze!(index)
            certificate = LL.query_index_certified(index, [1.0, 2.0, 3.0];
                cutoff=2.0)
            close(index)
            info = LL.certificate_info(certificate)
            @test info.query_len == 3
            @test info.evidence_len > 0
            @test info.result_len == 2
            @test info.cutoff_native == (kind === :dtw ? 4.0 : 2.0)
            @test LL.certificate_query_bits(certificate) ==
                reinterpret(UInt64, [1.0, 2.0, 3.0])
            @test [match.id for match in LL.certificate_matches(certificate;
                page_size=1)] == [7, 8]
            @test only(LL.certificate_match_page(certificate, 0, 1)).id == 7
            @test length(collect(LL.certificate_evidence(certificate))) ==
                info.evidence_len
            @test LL.reduce_certificate_evidence((n, _) -> n + 1, 0,
                certificate) == info.evidence_len
            proposal = LL.read_certificate(certificate)
            @test LL.verify_certificate(certificate, proposal)
            original = proposal.query_bits[1]
            proposal.query_bits[1] = 0
            @test !LL.verify_certificate(certificate, proposal)
            proposal.query_bits[1] = original
            original_match = proposal.matches[1]
            proposal.matches[1] = typeof(original_match)(99,
                original_match.distance)
            @test !LL.verify_certificate(certificate, proposal)
            proposal.matches[1] = original_match
            @test LL.verify_certificate(certificate, proposal)
            if kind === :erp
                readers = [Threads.@spawn begin
                    all(1:20) do _
                        LL.verify_certificate(certificate, proposal) &&
                            only(LL.certificate_match_page(certificate, 0, 1)).id == 7
                    end
                end for _ in 1:4]
                @test all(fetch, readers)
            end
        finally
            certificate === nothing || close(certificate)
            close(index)
        end
    end

    index = LL.TemporalIndex(:erp; quant_min=-10.0, quant_max=10.0,
        max_entries=1, max_total_samples=3, max_series_len=3)
    try
        LL.insert!(index, 7, [1.0, 2.0, 3.0])
        LL.freeze!(index)
        @test_throws LL.NativeError LL.query_index_certified(index,
            [1.0, 2.0, 3.0]; limits=LL.TemporalCertificateLimits(
                max_records=0))
    finally
        close(index)
    end
end
