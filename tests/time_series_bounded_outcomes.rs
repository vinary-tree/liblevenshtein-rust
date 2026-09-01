use liblevenshtein::time_series::{
    ErpConfig, ErpTransducer, ExactDecision, IncompleteReason, MetricMsmConfig,
    MetricMsmConfigError, MsmConfig, MsmTransducer, OperationOutcome, PageBudget,
    QuantizationConfig, ResourceKind, ResourceLimits, TemporalValidationError,
};

#[test]
fn strict_msm_distinguishes_exact_above_and_incomplete() {
    let config = MsmConfig::try_new(1.0).unwrap();
    let limits = ResourceLimits::default();

    let exact = config
        .distance_bounded(&[0.0, 1.0], &[0.0, 2.0], 1.0, limits)
        .unwrap();
    assert!(matches!(
        exact,
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance: 1.0, .. },
            ..
        }
    ));

    let above = config
        .distance_bounded(&[0.0, 1.0], &[0.0, 2.0], 0.5, limits)
        .unwrap();
    assert!(matches!(
        above,
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        }
    ));

    let bounded = ResourceLimits {
        max_dp_cells: 3,
        ..limits
    };
    let incomplete = config
        .distance_bounded(&[0.0, 1.0], &[0.0, 2.0], 1.0, bounded)
        .unwrap();
    assert!(matches!(
        incomplete,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::DpCells,
                limit: 3,
                requested: 4,
            },
            ..
        }
    ));
}

#[test]
fn invalid_input_never_becomes_an_empty_or_over_cutoff_result() {
    let config = MsmConfig::try_new(1.0).unwrap();
    assert_eq!(
        config.distance_bounded(&[0.0, f64::NAN], &[0.0], 1.0, ResourceLimits::default(),),
        Err(TemporalValidationError::NonFiniteSample {
            operand: liblevenshtein::time_series::Operand::Query,
            index: 1,
        })
    );
    assert_eq!(
        config.distance_bounded(&[0.0], &[0.0], f64::NAN, ResourceLimits::default(),),
        Err(TemporalValidationError::InvalidCutoff)
    );
}

#[test]
fn numeric_overflow_is_incomplete() {
    let outcome = MsmConfig::try_new(1.0)
        .unwrap()
        .distance_bounded(
            &[-f64::MAX],
            &[f64::MAX],
            f64::INFINITY,
            ResourceLimits::default(),
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            ..
        }
    ));
}

#[test]
fn metric_msm_requires_positive_cost_and_nonempty_series() {
    assert_eq!(
        MetricMsmConfig::try_new(0.0),
        Err(MetricMsmConfigError::NonPositiveSplitMergeCost)
    );
    assert_eq!(
        MsmConfig::try_new(0.0)
            .unwrap()
            .distance(&[1.0], &[1.0, 1.0]),
        0.0
    );

    let metric = MetricMsmConfig::try_new(1.0).unwrap();
    assert_eq!(
        metric.distance_bounded(&[], &[], 0.0, ResourceLimits::default()),
        Err(TemporalValidationError::EmptyMetricSeries)
    );
}

#[test]
fn paged_exact_range_equals_uninterrupted_search() {
    let series = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0, 2.5],
        vec![5.0, 6.0, 7.0],
        vec![0.0, 1.0, 2.0],
    ];
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).unwrap(),
        &series,
    );
    let query = [0.0, 1.0, 2.0];
    let expected = index.search_range(&query, 1.0);
    let mut outcome = index
        .search_range_bounded(
            &query,
            1.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 16,
                max_results: 1,
            },
        )
        .unwrap();
    let mut pauses = 0;
    loop {
        match outcome {
            OperationOutcome::Complete { value, usage } => {
                assert_eq!(value, expected);
                assert!(usage.work_units > 0);
                break;
            }
            OperationOutcome::Incomplete {
                partial,
                continuation: Some(continuation),
                ..
            } => {
                pauses += 1;
                assert!(partial.is_none());
                assert!(continuation
                    .exact_partial()
                    .iter()
                    .all(|entry| expected.contains(entry)));
                outcome = continuation.resume(PageBudget {
                    max_work_units: 16,
                    max_results: 1,
                });
            }
            other => panic!("unexpected terminal outcome: {other:?}"),
        }
    }
    assert!(pauses > 0);
}

#[test]
fn bounded_range_never_labels_result_pressure_complete() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).unwrap(),
        &[vec![0.0, 1.0]],
    );
    let limits = ResourceLimits {
        max_results: 0,
        ..ResourceLimits::default()
    };
    let outcome = index
        .search_range_bounded(
            &[0.0, 1.0],
            0.0,
            limits,
            PageBudget {
                max_work_units: 1_000,
                max_results: 1,
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            partial: Some(ref partial),
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::Results,
                limit: 0,
                requested: 1,
            },
            continuation: None,
            ..
        } if partial.is_empty()
    ));
}

#[test]
fn bounded_complete_empty_is_exhaustive() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).unwrap(),
        &[vec![5.0, 6.0]],
    );
    let outcome = index
        .search_range_bounded(
            &[0.0, 1.0],
            0.5,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 10_000,
                max_results: 10,
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Complete { value, .. } if value.is_empty()
    ));
}

#[test]
fn strict_bounded_knn_equals_exact_legacy_results() {
    let series = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0, 2.5],
        vec![5.0, 6.0, 7.0],
        vec![0.0, 1.0, 2.0],
    ];
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).unwrap(),
        &series,
    );
    let expected = index.search_knn(&[0.0, 1.0, 2.2], 3, f64::INFINITY);
    let outcome = index
        .search_knn_bounded(&[0.0, 1.0, 2.2], 3, ResourceLimits::default())
        .expect("finite kNN query is valid");
    assert!(matches!(
        outcome,
        OperationOutcome::Complete { value, usage }
            if value == expected && usage.candidates == series.len()
    ));
}

#[test]
fn strict_bounded_knn_never_turns_candidate_exhaustion_into_empty_completion() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).unwrap(),
        &[vec![0.0, 1.0], vec![2.0, 3.0]],
    );
    let limits = ResourceLimits {
        max_candidates: 0,
        ..ResourceLimits::default()
    };
    let outcome = index
        .search_knn_bounded(&[0.0, 1.0], 1, limits)
        .expect("finite kNN query is valid");
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::Candidates,
                limit: 0,
                requested: 1,
            },
            continuation: None,
            ..
        }
    ));
}

#[test]
fn bounded_product_root_frame_uses_one_compact_state_id() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).expect("positive MSM operation cost is valid"),
        &[vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 3.0]],
    );
    let outcome = index
        .search_range_bounded(
            &[0.0, 1.0, 2.0],
            10.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 0,
                max_results: 0,
            },
        )
        .expect("finite query and cutoff are valid");
    let OperationOutcome::Incomplete {
        continuation: Some(continuation),
        reason:
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                ..
            },
        ..
    } = outcome
    else {
        panic!("a zero-work page must retain a resumable product");
    };

    let stats = continuation.retained_product_state_stats();
    assert_eq!(stats.frames, 1);
    assert_eq!(stats.states, 1);
    assert_eq!(
        stats.column_cells, 0,
        "the root compact state has no reachable recurrence positions"
    );
}

#[test]
fn bounded_product_retains_a_canonical_position_antichain_not_a_dense_column() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).expect("positive MSM operation cost is valid"),
        &[vec![0.0, 1.0, 2.0]],
    );
    let outcome = index
        .search_range_bounded(
            &[0.0, 1.0, 2.0],
            10.0,
            ResourceLimits::default(),
            PageBudget {
                // One prefix-bound unit plus the four-row root transition.
                max_work_units: 5,
                max_results: 0,
            },
        )
        .expect("finite query and cutoff are valid");
    let OperationOutcome::Incomplete {
        continuation: Some(continuation),
        reason:
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                ..
            },
        ..
    } = outcome
    else {
        panic!("the page must pause before the second dictionary edge");
    };

    let stats = continuation.retained_product_state_stats();
    assert_eq!(stats.frames, 2);
    assert!(stats.states >= stats.frames);
    assert_eq!(
        stats.column_cells, 1,
        "the child stores one non-subsumed position; its vertical closure is lazy"
    );
}

#[test]
fn product_scratch_ceiling_rejects_the_root_arena_before_traversal() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).expect("positive MSM operation cost is valid"),
        &[vec![0.0]],
    );
    // A two-sample MSM query has a three-cell column. The product owns exactly
    // two shared cost columns and two shared active-row arrays; no DFS frame
    // owns a dense column. Exact state interning additionally requires a root
    // state header and collision-checked fingerprint bucket, so a ceiling that
    // fits only the two recurrence generations must fail before traversal.
    let fixed_scratch_bytes = 3 * 2 * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>());
    let limits = ResourceLimits {
        max_scratch_bytes: fixed_scratch_bytes,
        ..ResourceLimits::default()
    };
    let outcome = index
        .search_range_bounded(
            &[0.0, 1.0],
            10.0,
            limits,
            PageBudget {
                max_work_units: 1_000,
                max_results: 10,
            },
        )
        .expect("finite query and cutoff are valid");

    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit,
                requested,
            },
            continuation: None,
            ..
        } if limit == fixed_scratch_bytes && requested > fixed_scratch_bytes
    ));
}

#[test]
fn cancelling_a_generic_product_preserves_exact_membership_without_completeness() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).expect("positive MSM operation cost is valid"),
        &[vec![0.0], vec![0.0], vec![5.0]],
    );
    let outcome = index
        .search_range_bounded(
            &[0.0],
            0.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 10_000,
                max_results: 1,
            },
        )
        .expect("finite query and cutoff are valid");
    let OperationOutcome::Incomplete {
        partial: None,
        continuation: Some(continuation),
        ..
    } = outcome
    else {
        panic!("one-result page must pause with more exact work pending");
    };
    let before = continuation.exact_partial().to_vec();

    match continuation.cancel() {
        OperationOutcome::Incomplete {
            partial: Some(after),
            reason: IncompleteReason::Cancelled,
            continuation: None,
            ..
        } => {
            assert_eq!(after, before);
            assert!(after
                .iter()
                .all(|entry| index.search_range(&[0.0], 0.0).contains(entry)));
        }
        other => panic!("cancellation must remain explicitly incomplete: {other:?}"),
    }
}

#[test]
fn cancelling_the_specialized_erp_product_is_deterministic_and_terminal() {
    let index = ErpTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        ErpConfig::new(0.0),
        &[vec![0.0], vec![0.0], vec![5.0]],
    );
    let outcome = index
        .search_range_automaton_bounded(
            &[0.0],
            0.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 10_000,
                max_results: 1,
            },
        )
        .expect("finite ERP query and cutoff are valid");
    let OperationOutcome::Incomplete {
        partial: None,
        continuation: Some(continuation),
        ..
    } = outcome
    else {
        panic!("one-result ERP page must pause with exact work pending");
    };
    let before = continuation.exact_partial().to_vec();

    assert!(matches!(
        continuation.cancel(),
        OperationOutcome::Incomplete {
            partial: Some(ref after),
            reason: IncompleteReason::Cancelled,
            continuation: None,
            ..
        } if *after == before
    ));
}
