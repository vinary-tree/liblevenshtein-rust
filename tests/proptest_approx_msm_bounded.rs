use liblevenshtein::time_series::{
    ApproxMsmConfig, ApproxMsmIndex, ApproxMsmSearchOutcome, IncompleteReason, MsmConfig, Operand,
    ResourceKind, ResourceLimits, TemporalValidationError,
};
use proptest::prelude::*;

fn small_series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-20i16..=20, 0..8)
        .prop_map(|series| series.into_iter().map(f64::from).collect())
}

fn brute_knn(
    database: &[Vec<f64>],
    query: &[f64],
    config: MsmConfig,
    k: usize,
) -> Vec<(usize, f64)> {
    let mut neighbors: Vec<_> = database
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let distance = config.distance(query, candidate);
            distance.is_finite().then_some((index, distance))
        })
        .collect();
    neighbors.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    neighbors.truncate(k);
    neighbors
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn every_emitted_distance_is_exact_and_only_full_reranking_proves_recall(
        database in prop::collection::vec(small_series(), 0..12),
        query in small_series(),
        segments in 0usize..10,
        candidate_limit in 0usize..14,
        k in 0usize..14,
        cost in 0u8..5,
    ) {
        let msm = MsmConfig::try_new(f64::from(cost)).unwrap();
        let config = ApproxMsmConfig::try_new(segments, candidate_limit, msm).unwrap();
        let index = ApproxMsmIndex::from_series(config, &database);
        let outcome = index
            .search_knn_bounded(&query, k, ResourceLimits::default())
            .unwrap();

        let (result, exhaustive) = match &outcome {
            ApproxMsmSearchOutcome::Exhaustive { result, .. } => (result, true),
            ApproxMsmSearchOutcome::Advisory { result, .. } => (result, false),
            ApproxMsmSearchOutcome::Incomplete { reason, .. } => {
                prop_assert!(false, "small finite case was incomplete: {reason:?}");
                unreachable!()
            }
        };
        prop_assert_eq!(outcome.proves_recall(), exhaustive);
        prop_assert_eq!(result.coverage.proves_recall(), exhaustive);
        prop_assert_eq!(
            exhaustive,
            result.coverage.exact_reranked == database.len()
                && result.coverage.candidate_entries == database.len()
        );

        let mut last: Option<(f64, usize)> = None;
        for neighbor in &result.neighbors {
            let expected = msm.distance(&query, &database[neighbor.index]);
            prop_assert!(expected.is_finite());
            prop_assert_eq!(neighbor.distance.to_bits(), expected.to_bits());
            prop_assert_eq!(*neighbor.value, neighbor.index);
            if let Some((distance, index)) = last {
                prop_assert!(
                    distance.total_cmp(&neighbor.distance).is_lt()
                        || (distance.to_bits() == neighbor.distance.to_bits()
                            && index < neighbor.index)
                );
            }
            last = Some((neighbor.distance, neighbor.index));
        }

        if exhaustive {
            let got: Vec<_> = result
                .neighbors
                .iter()
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect();
            prop_assert_eq!(got, brute_knn(&database, &query, msm, k));
        }

        let repeated = index
            .search_knn_bounded(&query, k, ResourceLimits::default())
            .unwrap();
        prop_assert_eq!(outcome, repeated);
    }
}

#[test]
fn empty_advisory_result_cannot_be_read_as_complete_absence() {
    let database = vec![vec![1.0], vec![2.0], vec![3.0]];
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &database,
    );

    let outcome = index
        .search_knn_bounded(&[1.0], 0, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        ApproxMsmSearchOutcome::Advisory { ref result, .. } if result.neighbors.is_empty()
    ));
    assert!(!outcome.proves_recall());
}

#[test]
fn invalid_request_and_invalid_stored_data_are_distinct_from_empty() {
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(2, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &[vec![1.0, 2.0]],
    );
    assert_eq!(
        index.search_knn_bounded(&[f64::NAN], 1, ResourceLimits::default()),
        Err(TemporalValidationError::NonFiniteSample {
            operand: Operand::Query,
            index: 0,
        })
    );

    let invalid = ApproxMsmIndex::from_series(
        ApproxMsmConfig::new(2, 1, MsmConfig::new(1.0)),
        &[vec![1.0, f64::INFINITY]],
    );
    assert!(matches!(
        invalid
            .search_knn_bounded(&[1.0], 1, ResourceLimits::default())
            .unwrap(),
        ApproxMsmSearchOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::InvalidStoredData,
            ..
        }
    ));
}

#[test]
fn budget_failure_is_tagged_and_transactional_before_exact_reranking() {
    let database = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(2, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &database,
    );
    let limits = ResourceLimits {
        max_dp_cells: 0,
        ..ResourceLimits::default()
    };
    let outcome = index.search_knn_bounded(&[1.0, 2.0], 1, limits).unwrap();
    assert!(matches!(
        outcome,
        ApproxMsmSearchOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::DpCells,
                limit: 0,
                ..
            },
            ..
        }
    ));
}

#[test]
fn scratch_ceiling_fails_closed_before_scoring() {
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &[vec![1.0, 2.0]],
    );
    let limits = ResourceLimits {
        max_scratch_bytes: 0,
        ..ResourceLimits::default()
    };
    assert!(matches!(
        index.search_knn_bounded(&[1.0], 1, limits).unwrap(),
        ApproxMsmSearchOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: 0,
                ..
            },
            ..
        }
    ));
}

#[test]
fn query_validation_uses_ingestion_invariant_not_full_database_rescan() {
    let database: Vec<_> = (0..64)
        .map(|offset| vec![f64::from(offset); 2_048])
        .collect();
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &database,
    );
    let limits = ResourceLimits {
        max_work_units: 3_000,
        ..ResourceLimits::default()
    };
    let outcome = index.search_knn_bounded(&[0.0], 1, limits).unwrap();
    assert!(matches!(outcome, ApproxMsmSearchOutcome::Advisory { .. }));
    assert!(outcome.usage().work_units < 3_000);
}

#[test]
fn numeric_overflow_is_not_converted_to_exhaustive_empty() {
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &[vec![f64::MAX]],
    );
    let outcome = index
        .search_knn_bounded(&[-f64::MAX], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        ApproxMsmSearchOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::NumericOverflow,
            ..
        }
    ));
}

#[test]
fn malformed_public_configuration_is_rejected_before_search() {
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig {
            segments: 1,
            candidate_limit: 1,
            msm: MsmConfig { c: f64::NAN },
        },
        &[vec![1.0]],
    );
    assert_eq!(
        index.search_knn_bounded(&[1.0], 1, ResourceLimits::default()),
        Err(TemporalValidationError::InvalidConfiguration(
            "approximate MSM split/merge cost must be finite and nonnegative",
        ))
    );
}

#[test]
fn only_an_empty_index_can_return_exhaustive_empty_without_reranking() {
    let database: Vec<Vec<f64>> = Vec::new();
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &database,
    );
    let outcome = index
        .search_knn_bounded(&[1.0], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        ApproxMsmSearchOutcome::Exhaustive { ref result, .. }
            if result.neighbors.is_empty() && result.coverage.proves_recall()
    ));
    assert!(outcome.proves_recall());
}

#[test]
fn exact_ties_use_stable_insertion_position() {
    let database = vec![vec![0.0], vec![2.0], vec![2.0], vec![4.0]];
    let index = ApproxMsmIndex::from_series(
        ApproxMsmConfig::try_new(1, database.len(), MsmConfig::try_new(1.0).unwrap()).unwrap(),
        &database,
    );
    let ApproxMsmSearchOutcome::Exhaustive { result, .. } = index
        .search_knn_bounded(&[1.0], 3, ResourceLimits::default())
        .unwrap()
    else {
        panic!("full candidate pool must be exhaustive");
    };
    let positions: Vec<_> = result
        .neighbors
        .iter()
        .map(|neighbor| neighbor.index)
        .collect();
    assert_eq!(positions, vec![0, 1, 2]);
}

#[test]
fn bounded_search_does_not_require_or_invoke_value_clone() {
    #[derive(Debug, PartialEq)]
    struct NonClone(&'static str);

    let index = ApproxMsmIndex::from_entries(
        ApproxMsmConfig::try_new(1, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
        [(NonClone("nearest"), [1.0]), (NonClone("far"), [9.0])],
    );
    let ApproxMsmSearchOutcome::Advisory { result, .. } = index
        .search_knn_bounded(&[1.0], 1, ResourceLimits::default())
        .unwrap()
    else {
        panic!("one of two candidates is an advisory pool");
    };
    assert_eq!(result.neighbors[0].value, &NonClone("nearest"));
}

#[test]
fn strict_search_is_stack_safe_on_a_constrained_thread() {
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            let series: Vec<_> = (0..768).map(f64::from).collect();
            let index = ApproxMsmIndex::from_series(
                ApproxMsmConfig::try_new(16, 1, MsmConfig::try_new(1.0).unwrap()).unwrap(),
                std::slice::from_ref(&series),
            );
            assert!(index
                .search_knn_bounded(&series, 1, ResourceLimits::default())
                .unwrap()
                .proves_recall());
        })
        .unwrap();
    handle.join().unwrap();
}
