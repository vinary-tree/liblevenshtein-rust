#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_temporal_certificate_evidence_at, llev_temporal_certificate_free,
    llev_temporal_certificate_info, llev_temporal_certificate_matches,
    llev_temporal_certificate_query_bits, llev_temporal_certificate_verify,
    llev_temporal_index_free, llev_temporal_index_freeze, llev_temporal_index_insert,
    llev_temporal_index_new, llev_temporal_index_query_certified, LlevStatus,
    LlevTemporalAlgorithm, LlevTemporalCertificateEvidenceHeader,
    LlevTemporalCertificateEvidenceView, LlevTemporalCertificateInfo,
    LlevTemporalCertificateLimits, LlevTemporalCertificateView, LlevTemporalConfig,
    LlevTemporalIndexConfig, LlevTemporalIndexMatch, LlevTemporalRangeCertificate,
    LlevTemporalSearchLimits,
};
use std::ptr;

fn config(algorithm: LlevTemporalAlgorithm) -> LlevTemporalIndexConfig {
    LlevTemporalIndexConfig {
        temporal: LlevTemporalConfig {
            algorithm: algorithm as u32,
            reserved: 0,
            parameter0: if matches!(algorithm, LlevTemporalAlgorithm::Msm) {
                1.0
            } else {
                0.0
            },
            parameter1: 0.0,
            band: if matches!(algorithm, LlevTemporalAlgorithm::Dtw) {
                2
            } else {
                0
            },
            cutoff: f64::INFINITY,
        },
        quant_min: -10.0,
        quant_max: 10.0,
        quant_bins: 32,
        reserved: 0,
        max_entries: 3,
        max_total_samples: 9,
        max_series_len: 3,
    }
}

fn limits() -> LlevTemporalCertificateLimits {
    LlevTemporalCertificateLimits {
        search: LlevTemporalSearchLimits {
            max_series_len: 3,
            max_dp_cells: 100_000,
            max_work_units: 100_000,
            max_scratch_bytes: 4 * 1024 * 1024,
            max_trie_nodes: 100_000,
            max_trie_edges: 100_000,
            max_candidates: 100_000,
            max_results: 3,
            max_queue_entries: 100_000,
            max_continuation_bytes: 4 * 1024 * 1024,
        },
        max_witness_bytes: 4 * 1024 * 1024,
        max_records: 100_000,
        max_path_bytes: 4 * 1024 * 1024,
        max_work_units: 100_000,
    }
}

fn query_certified(
    index: *const liblevenshtein::ffi::LlevTemporalIndex,
    limits: &LlevTemporalCertificateLimits,
) -> (LlevStatus, *mut LlevTemporalRangeCertificate) {
    let mut certificate = ptr::null_mut();
    let values = [1.0, 2.0, 3.0];
    let status = unsafe {
        llev_temporal_index_query_certified(
            index,
            values.as_ptr(),
            values.len(),
            2.0,
            limits,
            &mut certificate,
        )
    };
    (status, certificate)
}

#[test]
fn five_kernels_expose_bounded_replayable_evidence_after_index_close() {
    for algorithm in [
        LlevTemporalAlgorithm::Msm,
        LlevTemporalAlgorithm::Erp,
        LlevTemporalAlgorithm::Twed,
        LlevTemporalAlgorithm::Dtw,
        LlevTemporalAlgorithm::Frechet,
    ] {
        let mut index = ptr::null_mut();
        assert_eq!(
            unsafe { llev_temporal_index_new(&config(algorithm), &mut index) },
            LlevStatus::Ok
        );
        for (id, values) in [(7, [1.0, 2.0, 3.0]), (8, [1.0, 2.5, 3.0])] {
            assert_eq!(
                unsafe { llev_temporal_index_insert(index, id, values.as_ptr(), values.len()) },
                LlevStatus::Ok
            );
        }
        assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
        let (status, certificate) = query_certified(index, &limits());
        assert_eq!(status, LlevStatus::Ok, "{algorithm:?}");
        assert!(!certificate.is_null());
        unsafe { llev_temporal_index_free(index) };

        let mut info = LlevTemporalCertificateInfo::default();
        assert_eq!(
            unsafe { llev_temporal_certificate_info(certificate, &mut info) },
            LlevStatus::Ok
        );
        assert_eq!(info.query_len, 3);
        assert!(info.evidence_len > 0);
        assert_eq!(info.result_len, 2);
        assert_eq!(info.snapshot_present, 0);
        assert_eq!(
            info.cutoff_native,
            if matches!(algorithm, LlevTemporalAlgorithm::Dtw) {
                4.0
            } else {
                2.0
            }
        );

        let mut query_bits = vec![0_u64; info.query_len];
        for start in 0..info.query_len {
            let mut written = usize::MAX;
            assert_eq!(
                unsafe {
                    llev_temporal_certificate_query_bits(
                        certificate,
                        start,
                        query_bits[start..].as_mut_ptr(),
                        1,
                        &mut written,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(written, 1);
        }
        assert_eq!(
            query_bits,
            [1.0_f64.to_bits(), 2.0_f64.to_bits(), 3.0_f64.to_bits()]
        );

        let mut results = vec![LlevTemporalIndexMatch::default(); info.result_len];
        let mut written = usize::MAX;
        assert_eq!(
            unsafe {
                llev_temporal_certificate_matches(
                    certificate,
                    0,
                    results.as_mut_ptr(),
                    results.len(),
                    &mut written,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(written, 2);
        assert_eq!(results[0].id, 7);
        assert_eq!(results[0].distance, 0.0);
        assert_eq!(results[1].id, 8);
        assert!(results[1].distance > 0.0);

        let mut headers = Vec::new();
        let mut paths = Vec::new();
        for offset in 0..info.evidence_len {
            let mut header = LlevTemporalCertificateEvidenceHeader::default();
            let mut path_len = usize::MAX;
            assert_eq!(
                unsafe {
                    llev_temporal_certificate_evidence_at(
                        certificate,
                        offset,
                        &mut header,
                        ptr::null_mut(),
                        0,
                        &mut path_len,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(header.path_len, path_len);
            let mut path = vec![0_u8; path_len];
            assert_eq!(
                unsafe {
                    llev_temporal_certificate_evidence_at(
                        certificate,
                        offset,
                        &mut header,
                        path.as_mut_ptr(),
                        path.len(),
                        &mut path_len,
                    )
                },
                LlevStatus::Ok
            );
            assert!((1..=5).contains(&header.kind));
            headers.push(header);
            paths.push(path);
        }
        let mut evidence: Vec<_> = headers
            .iter()
            .zip(&paths)
            .map(|(header, path)| LlevTemporalCertificateEvidenceView {
                header: *header,
                path: path.as_ptr(),
            })
            .collect();
        let mut view = LlevTemporalCertificateView {
            info,
            query_bits: query_bits.as_ptr(),
            evidence: evidence.as_ptr(),
            results: results.as_ptr(),
        };
        let mut valid = 0_u8;
        assert_eq!(
            unsafe { llev_temporal_certificate_verify(certificate, &view, &mut valid) },
            LlevStatus::Ok
        );
        assert_eq!(valid, 1);

        evidence[0].header.lower_bound =
            f64::from_bits(evidence[0].header.lower_bound.to_bits() ^ 1);
        view.evidence = evidence.as_ptr();
        assert_eq!(
            unsafe { llev_temporal_certificate_verify(certificate, &view, &mut valid) },
            LlevStatus::Ok
        );
        assert_eq!(valid, 0);
        evidence[0].header = headers[0];
        results[0].id += 1;
        assert_eq!(
            unsafe { llev_temporal_certificate_verify(certificate, &view, &mut valid) },
            LlevStatus::Ok
        );
        assert_eq!(valid, 0);
        unsafe { llev_temporal_certificate_free(certificate) };
    }
}

#[test]
fn certificate_limits_fail_without_partial_handle() {
    let mut index = ptr::null_mut();
    assert_eq!(
        unsafe { llev_temporal_index_new(&config(LlevTemporalAlgorithm::Erp), &mut index) },
        LlevStatus::Ok
    );
    let values = [1.0, 2.0, 3.0];
    assert_eq!(
        unsafe { llev_temporal_index_insert(index, 7, values.as_ptr(), values.len()) },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
    let mut restricted = limits();
    restricted.max_records = 0;
    let (status, certificate) = query_certified(index, &restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert!(certificate.is_null());
    unsafe { llev_temporal_index_free(index) };
}
