use liblevenshtein::time_series::elastic::ElasticTransducer;
use liblevenshtein::time_series::{
    DtwConfig, ElasticCertificateError, ElasticCertificateLimits, ElasticRangeEvidence, ErpConfig,
    ErpTransducer, Operand, QuantizationConfig, ResourceKind, ResourceLimits,
    TemporalValidationError,
};
use proptest::prelude::*;

fn fixture(order: &[u64]) -> ErpTransducer<u64> {
    // One deliberately broad bin forces collisions to reach K4/K3 rather than
    // allowing the relaxed trie column to decide the full-precision result.
    let mut index: ErpTransducer<u64> = ErpTransducer::new(
        QuantizationConfig::uniform(-100.0, 100.0, 1),
        ErpConfig::new(0.0),
    );
    for stable_id in order {
        let series: &[f64] = match stable_id {
            7 => &[0.0, 1.0, 2.0],
            11 => &[0.0, 1.5, 2.0],
            19 => &[50.0, 50.0],
            _ => panic!("unknown fixture id"),
        };
        assert!(index.insert(*stable_id, series));
    }
    index
}

#[test]
fn certificate_is_deterministic_replayable_and_insertion_order_independent() {
    let query = [0.0, 1.0, 2.0];
    let limits = ElasticCertificateLimits::default();
    let first = fixture(&[19, 7, 11]);
    let second = fixture(&[11, 19, 7]);

    let (first_results, first_certificate) = first
        .search_range_with_certificate(&query, 0.75, limits)
        .expect("bounded certificate");
    let (second_results, second_certificate) = second
        .search_range_with_certificate(&query, 0.75, limits)
        .expect("bounded certificate after permuted insertion");

    assert_eq!(first_results, second_results);
    assert_eq!(first_certificate, second_certificate);
    assert_eq!(first_certificate.query_bits.len(), query.len());
    assert!(first_certificate.witness_bytes >= first_certificate.path_bytes);
    assert!(first
        .verify_range_certificate(&query, 0.75, &first_certificate, limits)
        .expect("replay certificate"));

    let mut changed_query = first_certificate.clone();
    changed_query.query_bits[0] ^= 1;
    assert!(!first
        .verify_range_certificate(&query, 0.75, &changed_query, limits)
        .expect("reject changed query bits"));

    let mut changed_cutoff = first_certificate.clone();
    changed_cutoff.cutoff = 0.5;
    assert!(!first
        .verify_range_certificate(&query, 0.75, &changed_cutoff, limits)
        .expect("reject changed cutoff"));

    let mut changed_accounting = first_certificate.clone();
    changed_accounting.work_units += 1;
    assert!(!first
        .verify_range_certificate(&query, 0.75, &changed_accounting, limits)
        .expect("reject changed work accounting"));

    let mut changed_evidence = first_certificate.clone();
    changed_evidence
        .evidence
        .push(ElasticRangeEvidence::PrefixPruned {
            quantized_path: vec![0],
            lower_bound: 1.0,
        });
    assert!(!first
        .verify_range_certificate(&query, 0.75, &changed_evidence, limits)
        .expect("reject an injected K1 decision"));
}

#[test]
fn generated_evidence_exercises_and_replays_every_k1_through_k4_decision() {
    let limits = ElasticCertificateLimits::default();

    let mut prefix_index: ElasticTransducer<DtwConfig, u64> =
        ElasticTransducer::new(QuantizationConfig::for_u8(-1.0, 101.0), DtwConfig::new(0));
    assert!(prefix_index.insert(1, &[100.0]));
    let (_, prefix_certificate) = prefix_index
        .search_range_with_certificate(&[0.0], 1.0, limits)
        .expect("DTW K1 certificate");
    assert!(prefix_certificate
        .evidence
        .iter()
        .any(|record| matches!(record, ElasticRangeEvidence::PrefixPruned { .. })));
    assert!(prefix_index
        .verify_range_certificate(&[0.0], 1.0, &prefix_certificate, limits)
        .expect("replay K1"));
    let mut changed_prefix = prefix_certificate.clone();
    if let Some(ElasticRangeEvidence::PrefixPruned { quantized_path, .. }) = changed_prefix
        .evidence
        .iter_mut()
        .find(|record| matches!(record, ElasticRangeEvidence::PrefixPruned { .. }))
    {
        let mut changed = quantized_path.to_vec();
        changed.push(255);
        *quantized_path = changed;
    }
    assert!(!prefix_index
        .verify_range_certificate(&[0.0], 1.0, &changed_prefix, limits)
        .expect("reject mutated K1 path"));

    let mut subtree_index: ErpTransducer<u64> =
        ErpTransducer::new(QuantizationConfig::for_u8(-1.0, 101.0), ErpConfig::new(0.0));
    assert!(subtree_index.insert(2, &[100.0]));
    let (_, subtree_certificate) = subtree_index
        .search_range_with_certificate(&[0.0], 1.0, limits)
        .expect("ERP K2 certificate");
    assert!(subtree_certificate
        .evidence
        .iter()
        .any(|record| matches!(record, ElasticRangeEvidence::SubtreePruned { .. })));
    assert!(subtree_index
        .verify_range_certificate(&[0.0], 1.0, &subtree_certificate, limits)
        .expect("replay K2"));
    let mut changed_subtree = subtree_certificate.clone();
    if let Some(ElasticRangeEvidence::SubtreePruned { lower_bound, .. }) = changed_subtree
        .evidence
        .iter_mut()
        .find(|record| matches!(record, ElasticRangeEvidence::SubtreePruned { .. }))
    {
        *lower_bound = f64::from_bits(lower_bound.to_bits() ^ 1);
    }
    assert!(!subtree_index
        .verify_range_certificate(&[0.0], 1.0, &changed_subtree, limits)
        .expect("reject mutated K2 bound"));

    let mut terminal_index: ErpTransducer<u64> =
        ErpTransducer::new(QuantizationConfig::for_u8(-1.0, 11.0), ErpConfig::new(0.0));
    assert!(terminal_index.insert(3, &[0.0]));
    let (_, terminal_certificate) = terminal_index
        .search_range_with_certificate(&[0.0, 10.0], 1.0, limits)
        .expect("terminal-row certificate");
    assert!(terminal_certificate
        .evidence
        .iter()
        .any(|record| matches!(record, ElasticRangeEvidence::TerminalPruned { .. })));
    assert!(terminal_index
        .verify_range_certificate(&[0.0, 10.0], 1.0, &terminal_certificate, limits)
        .expect("replay terminal bound"));
    let mut changed_terminal = terminal_certificate.clone();
    if let Some(ElasticRangeEvidence::TerminalPruned { lower_bound, .. }) = changed_terminal
        .evidence
        .iter_mut()
        .find(|record| matches!(record, ElasticRangeEvidence::TerminalPruned { .. }))
    {
        *lower_bound = f64::from_bits(lower_bound.to_bits() ^ 1);
    }
    assert!(!terminal_index
        .verify_range_certificate(&[0.0, 10.0], 1.0, &changed_terminal, limits)
        .expect("reject mutated terminal bound"));

    let candidate_index = fixture(&[7, 19]);
    let (_, candidate_certificate) = candidate_index
        .search_range_with_certificate(&[0.0, 1.0, 2.0], 0.75, limits)
        .expect("K4 and K3 collision certificate");
    assert!(candidate_certificate
        .evidence
        .iter()
        .any(|record| matches!(record, ElasticRangeEvidence::CandidatePruned { .. })));
    assert!(candidate_certificate
        .evidence
        .iter()
        .any(|record| matches!(record, ElasticRangeEvidence::ExactCandidate { .. })));
    assert!(candidate_index
        .verify_range_certificate(&[0.0, 1.0, 2.0], 0.75, &candidate_certificate, limits,)
        .expect("replay K4/K3"));
    let mut changed_candidate = candidate_certificate.clone();
    if let Some(ElasticRangeEvidence::CandidatePruned { stable_id, .. }) = changed_candidate
        .evidence
        .iter_mut()
        .find(|record| matches!(record, ElasticRangeEvidence::CandidatePruned { .. }))
    {
        *stable_id ^= 1;
    }
    assert!(!candidate_index
        .verify_range_certificate(&[0.0, 1.0, 2.0], 0.75, &changed_candidate, limits,)
        .expect("reject mutated K4 identity"));
    let mut changed_exact = candidate_certificate.clone();
    if let Some(ElasticRangeEvidence::ExactCandidate { survived, .. }) = changed_exact
        .evidence
        .iter_mut()
        .find(|record| matches!(record, ElasticRangeEvidence::ExactCandidate { .. }))
    {
        *survived = !*survived;
    }
    assert!(!candidate_index
        .verify_range_certificate(&[0.0, 1.0, 2.0], 0.75, &changed_exact, limits,)
        .expect("reject mutated K3 decision"));
}

#[test]
fn certificate_rejects_invalid_requests_instead_of_producing_empty_evidence() {
    let index = fixture(&[7]);
    let limits = ElasticCertificateLimits::default();
    assert_eq!(
        index.search_range_with_certificate(&[0.0, f64::NAN], 1.0, limits),
        Err(ElasticCertificateError::Validation(
            TemporalValidationError::NonFiniteSample {
                operand: Operand::Query,
                index: 1,
            }
        ))
    );
    assert_eq!(
        index.search_range_with_certificate(&[0.0], -1.0, limits),
        Err(ElasticCertificateError::Validation(
            TemporalValidationError::InvalidCutoff
        ))
    );
    assert_eq!(
        index.search_range_with_certificate(&[], 1.0, limits),
        Err(ElasticCertificateError::Unsupported)
    );
}

#[test]
fn every_certificate_resource_class_fails_closed_at_its_boundary() {
    let index = fixture(&[7, 11]);
    let query = [0.0, 1.0, 2.0];
    let defaults = ResourceLimits::default();

    let cases = [
        (
            ElasticCertificateLimits {
                max_work_units: 0,
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::WorkUnits,
        ),
        (
            ElasticCertificateLimits {
                max_records: 0,
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::Results,
        ),
        (
            ElasticCertificateLimits {
                max_path_bytes: 0,
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::WitnessBytes,
        ),
        (
            ElasticCertificateLimits {
                resources: ResourceLimits {
                    max_scratch_bytes: 0,
                    ..defaults
                },
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::ScratchBytes,
        ),
        (
            ElasticCertificateLimits {
                resources: ResourceLimits {
                    max_witness_bytes: query.len() * std::mem::size_of::<u64>() - 1,
                    ..defaults
                },
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::WitnessBytes,
        ),
        (
            ElasticCertificateLimits {
                resources: ResourceLimits {
                    max_queue_entries: 1,
                    ..defaults
                },
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::QueueEntries,
        ),
        (
            ElasticCertificateLimits {
                resources: ResourceLimits {
                    max_continuation_bytes: 0,
                    ..defaults
                },
                ..ElasticCertificateLimits::default()
            },
            ResourceKind::ContinuationBytes,
        ),
    ];

    for (limits, expected_resource) in cases {
        let error = index
            .search_range_with_certificate(&query, 1.0, limits)
            .expect_err("the selected zero/one-under resource ceiling must fail closed");
        assert!(
            matches!(
                error,
                ElasticCertificateError::BudgetExceeded { resource, .. }
                    if resource == expected_resource
            ),
            "unexpected error for {expected_resource:?}: {error:?}"
        );
    }
}

#[test]
fn certificate_walk_is_stack_safe_for_a_deep_single_path() {
    const DEPTH: usize = 100_000;
    std::thread::Builder::new()
        .name("elastic-certificate-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            let mut index: ErpTransducer<u64> =
                ErpTransducer::new(QuantizationConfig::for_u8(-1.0, 1.0), ErpConfig::new(0.0));
            let candidate = vec![0.0; DEPTH];
            assert!(index.insert(7, &candidate));
            let limits = ElasticCertificateLimits {
                resources: ResourceLimits {
                    max_series_len: DEPTH,
                    max_dp_cells: DEPTH * 8,
                    max_work_units: DEPTH * 8,
                    max_scratch_bytes: 32 * 1024 * 1024,
                    max_continuation_bytes: 64 * 1024 * 1024,
                    max_witness_bytes: 2 * 1024 * 1024,
                    ..ResourceLimits::default()
                },
                max_records: 4,
                max_path_bytes: DEPTH,
                max_work_units: DEPTH * 8,
            };
            let (result, certificate) = index
                .search_range_with_certificate(&[0.0], 0.0, limits)
                .expect("deep iterative certificate walk");
            assert_eq!(result, vec![(7, 0.0)]);
            assert_eq!(certificate.evidence.len(), 1);
            assert!(index
                .verify_range_certificate(&[0.0], 0.0, &certificate, limits)
                .expect("deep certificate replay"));
        })
        .expect("spawn constrained-stack certificate test")
        .join()
        .expect("certificate traversal must not overflow the stack");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn certified_range_equals_independent_full_scan(
        query in prop::collection::vec(-8_i16..=8, 1..7),
        candidates in prop::collection::vec(prop::collection::vec(-8_i16..=8, 1..7), 0..10),
        cutoff in 0_u16..=64,
    ) {
        let query: Vec<f64> = query.into_iter().map(f64::from).collect();
        let candidates: Vec<Vec<f64>> = candidates
            .into_iter()
            .map(|series| series.into_iter().map(f64::from).collect())
            .collect();
        let cutoff = f64::from(cutoff) / 4.0;
        let kernel = ErpConfig::new(0.0);
        let mut index: ErpTransducer<u64> = ErpTransducer::new(
            QuantizationConfig::uniform(-8.0, 8.0, 4),
            kernel,
        );
        for (stable_id, series) in candidates.iter().enumerate() {
            prop_assert!(index.insert(stable_id as u64, series));
        }

        let (mut actual, certificate) = index
            .search_range_with_certificate(
                &query,
                cutoff,
                ElasticCertificateLimits::default(),
            )
            .expect("generated request lies in the certificate domain");
        let mut expected: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(stable_id, series)| {
                let distance = kernel.distance(&query, series);
                (distance <= cutoff).then_some((stable_id as u64, distance))
            })
            .collect();
        actual.sort_by_key(|(stable_id, _)| *stable_id);
        expected.sort_by_key(|(stable_id, _)| *stable_id);

        prop_assert_eq!(actual, expected);
        prop_assert!(index
            .verify_range_certificate(
                &query,
                cutoff,
                &certificate,
                ElasticCertificateLimits::default(),
            )
            .expect("replay a generated complete certificate"));
    }
}
