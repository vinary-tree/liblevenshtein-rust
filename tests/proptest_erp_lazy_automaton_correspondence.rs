//! Correspondence properties for the production lazy ERP automata.

use std::collections::HashMap;

use liblevenshtein::time_series::{
    ErpConfig, ErpOnlineAutomaton, ErpTransducer, OnlineAutomatonLimits, OnlineStepOutcome,
    OperationOutcome, PageBudget, QuantizationConfig, ResourceKind, ResourceLimits,
};
use proptest::prelude::*;

fn result_map(results: &[(usize, f64)]) -> HashMap<usize, f64> {
    results.iter().copied().collect()
}

fn exhaust_automaton_range(
    mut outcome: OperationOutcome<
        Vec<(usize, f64)>,
        liblevenshtein::time_series::ErpAutomatonRangeContinuation<'_, usize>,
    >,
    page: PageBudget,
) -> Vec<(usize, f64)> {
    loop {
        match outcome {
            OperationOutcome::Complete { value, .. } => return value,
            OperationOutcome::Incomplete {
                continuation: Some(next),
                reason,
                ..
            } => {
                assert!(matches!(
                    reason,
                    liblevenshtein::time_series::IncompleteReason::BudgetExceeded { .. }
                ));
                outcome = next.resume(page);
            }
            OperationOutcome::Incomplete {
                continuation: None,
                reason,
                ..
            } => panic!("valid bounded ERP query terminated: {reason:?}"),
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn every_online_prefix_matches_the_independent_scalar_dp(
        raw_query in prop::collection::vec(-20i16..=20, 0..24),
        raw_target in prop::collection::vec(-20i16..=20, 0..40),
        raw_gap in -10i16..=10,
        cutoff in 0u16..=200,
    ) {
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let target: Vec<f64> = raw_target.into_iter().map(f64::from).collect();
        let config = ErpConfig::new(f64::from(raw_gap));
        let cutoff = f64::from(cutoff);
        let mut automaton = ErpOnlineAutomaton::new(
            &query,
            config,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("generated finite query fits default online limits");

        prop_assert_eq!(
            automaton.observation().distance_within_cutoff,
            config.distance_with_cutoff(&query, &[], cutoff),
        );
        for (prefix_index, sample) in target.iter().copied().enumerate() {
            let outcome = automaton.advance(sample).expect("generated target sample is finite");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small generated transition");
                return Ok(());
            };
            prop_assert_eq!(value.consumed_target_len, prefix_index + 1);
            prop_assert!(value.active_positions <= query.len() + 1);
            prop_assert!(usage.queue_entries <= query.len() + 1);
            prop_assert_eq!(
                value.distance_within_cutoff,
                config.distance_with_cutoff(&query, &target[..=prefix_index], cutoff),
            );
        }
    }

    #[test]
    fn lazy_dictionary_product_equals_dense_walker_and_brute_force(
        raw_series in prop::collection::vec(
            prop::collection::vec(-16i16..=16, 0..10),
            0..24,
        ),
        raw_query in prop::collection::vec(-16i16..=16, 0..10),
        raw_gap in -8i16..=8,
        cutoff in 0u16..=120,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let config = ErpConfig::new(f64::from(raw_gap));
        let cutoff = f64::from(cutoff);
        let index = ErpTransducer::from_series(
            QuantizationConfig::for_u8(-16.0, 16.0),
            config,
            &series,
        );
        let page = PageBudget {
            max_work_units: 1_000_000,
            max_results: 1,
        };
        let outcome = index
            .search_range_automaton_bounded(&query, cutoff, ResourceLimits::default(), page)
            .expect("generated query and cutoff are valid");
        let lazy = exhaust_automaton_range(outcome, page);
        let dense = index.search_range(&query, cutoff);
        let brute: Vec<_> = series
            .iter()
            .enumerate()
            .filter_map(|(id, candidate)| {
                config
                    .distance_with_cutoff(&query, candidate, cutoff)
                    .map(|distance| (id, distance))
            })
            .collect();

        prop_assert_eq!(result_map(&lazy), result_map(&dense));
        prop_assert_eq!(result_map(&lazy), result_map(&brute));
        prop_assert!(lazy.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    }
}

#[test]
fn every_full_precision_member_of_a_quantization_collision_is_verified() {
    let config = ErpConfig::new(0.0);
    let mut index: ErpTransducer<usize> =
        ErpTransducer::new(QuantizationConfig::for_u8(0.0, 1_000_000.0), config);
    let candidates = [vec![10.0], vec![20.0], vec![30.0], vec![40.0]];
    for (id, candidate) in candidates.iter().enumerate() {
        assert!(index.insert(id, candidate));
    }
    let page = PageBudget {
        max_work_units: 10_000,
        max_results: 1,
    };
    let outcome = index
        .search_range_automaton_bounded(&[20.0], 10.0, ResourceLimits::default(), page)
        .expect("query is finite and bounded");
    let results = exhaust_automaton_range(outcome, page);
    assert_eq!(
        result_map(&results),
        HashMap::from([(0, 10.0), (1, 0.0), (2, 10.0)])
    );
}

#[test]
fn an_unstarted_page_is_incomplete_not_complete_empty() {
    let index = ErpTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        ErpConfig::default(),
        &[vec![1.0, 2.0]],
    );
    let outcome = index
        .search_range_automaton_bounded(
            &[1.0, 2.0],
            0.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 0,
                max_results: 0,
            },
        )
        .expect("request is valid even when its first page has no work budget");
    match outcome {
        OperationOutcome::Incomplete {
            partial,
            continuation: Some(continuation),
            reason:
                liblevenshtein::time_series::IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::WorkUnits,
                    ..
                },
            ..
        } => {
            assert_eq!(partial, None);
            assert!(continuation.exact_partial().is_empty());
            assert_eq!(continuation.retained_counts().0, 1);
        }
        other => panic!("zero-work page must pause with a continuation: {other:?}"),
    }
}

#[test]
fn numeric_overflow_is_rejected_at_construction_not_reported_above_cutoff() {
    let error = ErpOnlineAutomaton::new(
        &[f64::MAX, -f64::MAX],
        ErpConfig::default(),
        f64::INFINITY,
        OnlineAutomatonLimits::default(),
    )
    .expect_err("unrepresentable finite ERP suffix must fail closed");
    assert!(matches!(
        error,
        liblevenshtein::time_series::TemporalAutomatonError::Resource(
            liblevenshtein::time_series::IncompleteReason::NumericOverflow
        )
    ));
}
