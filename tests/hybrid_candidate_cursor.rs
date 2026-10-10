use liblevenshtein::time_series::{
    HybridSearchIndex, HybridStartError, IncompleteReason, LowerBoundType, MsmConfig, PageBudget,
    QuantizationConfig, ResourceKind, ResourceLimits,
};

fn collect(index: &HybridSearchIndex<u64>, query: &[f64], cutoff: f64) -> Vec<(u64, f64)> {
    let mut cursor = index
        .search_hybrid_bounded(query, cutoff, ResourceLimits::default())
        .unwrap();
    let mut matches = Vec::new();
    for _ in 0..100 {
        let page = cursor
            .next_page(PageBudget {
                max_work_units: 1_000,
                max_results: 1,
            })
            .unwrap();
        matches.extend(page.matches);
        if page.done {
            return matches;
        }
    }
    panic!("hybrid candidate cursor failed to complete");
}

#[test]
fn bounded_hybrid_knn_matches_legacy_expanding_threshold_search() {
    let mut index = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    for (id, samples) in [
        (7, [1.0, 2.0, 3.0]),
        (8, [1.1, 2.1, 3.1]),
        (9, [3.0, 2.0, 1.0]),
        (10, [8.0, 8.0, 8.0]),
    ] {
        index.insert(id, &samples);
    }
    for k in 0..=5 {
        for initial in [0.0, 0.5, 2.0, -1.0, f64::NAN] {
            let expected = index.search_knn(&[1.0, 2.0, 3.0], k, initial);
            let observed = index
                .search_hybrid_knn_bounded(&[1.0, 2.0, 3.0], k, initial, ResourceLimits::default())
                .unwrap()
                .matches;
            assert_eq!(observed, expected, "k={k} initial={initial}");
        }
    }
}

#[test]
fn bounded_hybrid_knn_fails_without_partial_neighbors() {
    let mut index = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    index.insert(7, &[1.0, 2.0]);
    index.insert(8, &[1.1, 2.1]);
    let error = index
        .search_hybrid_knn_bounded(
            &[1.0, 2.0],
            2,
            0.0,
            ResourceLimits {
                max_results: 1,
                ..ResourceLimits::default()
            },
        )
        .err()
        .expect("k greater than result ceiling must fail");
    assert!(matches!(
        error,
        HybridStartError::Resource(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::Results,
            ..
        })
    ));
    let error = index
        .search_hybrid_knn_bounded(
            &[1.0, 2.0],
            2,
            0.0,
            ResourceLimits {
                max_queue_entries: 1,
                ..ResourceLimits::default()
            },
        )
        .err()
        .expect("top-k heap ceiling must fail");
    assert!(matches!(
        error,
        HybridStartError::Resource(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::QueueEntries,
            ..
        })
    ));
    let error = index
        .search_hybrid_knn_bounded(
            &[1.0, 2.0],
            2,
            0.0,
            ResourceLimits {
                max_candidates: 1,
                ..ResourceLimits::default()
            },
        )
        .err()
        .expect("cumulative candidate ceiling must fail");
    assert!(matches!(
        error,
        HybridStartError::Resource(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::Candidates,
            ..
        })
    ));
}

#[test]
fn bounded_hybrid_matches_legacy_candidate_and_msm_pipeline() {
    let mut index = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    for (id, samples) in [
        (7, vec![1.0, 2.0, 3.0]),
        (8, vec![1.1, 2.1, 3.1]),
        (9, vec![3.0, 2.0, 1.0]),
        (10, vec![1.0, 2.0]),
        (11, vec![9.0, 9.0, 9.0]),
    ] {
        index.insert(id, &samples);
    }
    index.insert(11, &[1.0, 2.0, 4.0]);
    index.remove(10);
    for lower_bound in [
        LowerBoundType::LengthOnly,
        LowerBoundType::EuclideanOnly,
        LowerBoundType::L1Only,
        LowerBoundType::Combined,
    ] {
        index.set_lower_bound_type(lower_bound);
        for use_bounds in [false, true] {
            index.set_use_lower_bounds(use_bounds);
            for cutoff in [0.0, 0.5, 1.0, 3.0, f64::INFINITY] {
                let mut expected = index.search_exact(&[1.0, 2.0, 3.0], cutoff);
                let mut observed = collect(&index, &[1.0, 2.0, 3.0], cutoff);
                expected.sort_by_key(|&(id, _)| id);
                observed.sort_by_key(|&(id, _)| id);
                assert_eq!(
                    observed, expected,
                    "bound={lower_bound:?} enabled={use_bounds} cutoff={cutoff}"
                );
            }
        }
    }
}

#[test]
fn hybrid_cursor_pages_and_limits_fail_closed() {
    let mut index = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    index.insert(7, &[1.0, 2.0]);
    index.insert(8, &[1.0, 2.0]);
    let mut cursor = index
        .search_hybrid_bounded(
            &[1.0, 2.0],
            1.0,
            ResourceLimits {
                max_results: 1,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 1,
            max_results: 1,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::WorkUnits,
            ..
        })
    ));
    let first = cursor
        .next_page(PageBudget {
            max_work_units: 1_000,
            max_results: 1,
        })
        .unwrap();
    assert_eq!(first.matches.len(), 1);
    assert!(!first.done);
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 1_000,
            max_results: 1,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::Results,
            ..
        })
    ));
}
