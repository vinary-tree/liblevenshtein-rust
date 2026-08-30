//! Point-interval correspondence for every production online elastic automaton.

use liblevenshtein::time_series::{
    ElasticOnlineAutomaton, FrechetConfig, IncompleteReason, MsmConfig, MsmKernel,
    OnlineAutomatonLimits, OnlineStepOutcome, ResourceKind, TwedConfig,
};
use proptest::prelude::*;

fn close(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs() <= 1.0e-9,
        (None, None) => true,
        _ => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn msm_point_stream_matches_scalar_dp_after_every_prefix(
        raw_query in prop::collection::vec(-12i16..=12, 1..12),
        raw_target in prop::collection::vec(-12i16..=12, 0..18),
        raw_c in 0u8..=8,
        cutoff in 0u16..=160,
    ) {
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let target: Vec<f64> = raw_target.into_iter().map(f64::from).collect();
        let config = MsmConfig::new(f64::from(raw_c));
        let cutoff = f64::from(cutoff);
        let mut automaton = ElasticOnlineAutomaton::new(
            &query,
            MsmKernel::new(config),
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("small finite MSM query has an online state");
        prop_assert!(close(
            automaton.observation().distance_within_cutoff,
            config.distance_with_cutoff(&query, &[], cutoff),
        ));
        for (index, sample) in target.iter().copied().enumerate() {
            let outcome = automaton.advance(sample).expect("generated sample is finite");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small MSM transition");
                return Ok(());
            };
            prop_assert_eq!(value.consumed_target_len, index + 1);
            prop_assert!(usage.queue_entries <= query.len() + 1);
            prop_assert!(usage.work_units <= query.len() + 1);
            prop_assert!(close(
                value.distance_within_cutoff,
                config.distance_with_cutoff(&query, &target[..=index], cutoff),
            ));
        }
    }

    #[test]
    fn unit_grid_twed_point_stream_matches_scalar_dp_after_every_prefix(
        raw_query in prop::collection::vec(-10i16..=10, 0..12),
        raw_target in prop::collection::vec(-10i16..=10, 0..18),
        cutoff in 0u16..=200,
    ) {
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let target: Vec<f64> = raw_target.into_iter().map(f64::from).collect();
        let config = TwedConfig::new(0.5, 1.0);
        let cutoff = f64::from(cutoff);
        let mut automaton = ElasticOnlineAutomaton::new(
            &query,
            config,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("small finite unit-grid TWED query has an online state");
        prop_assert!(close(
            automaton.observation().distance_within_cutoff,
            config.distance_with_cutoff(&query, &[], cutoff),
        ));
        for (index, sample) in target.iter().copied().enumerate() {
            let outcome = automaton.advance(sample).expect("generated sample is finite");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small TWED transition");
                return Ok(());
            };
            prop_assert_eq!(value.consumed_target_len, index + 1);
            prop_assert!(usage.queue_entries <= query.len() + 1);
            prop_assert!(usage.work_units <= query.len() + 1);
            prop_assert!(close(
                value.distance_within_cutoff,
                config.distance_with_cutoff(&query, &target[..=index], cutoff),
            ));
        }
    }

    #[test]
    fn frechet_point_stream_matches_scalar_dp_after_every_prefix(
        raw_query in prop::collection::vec(-10i16..=10, 1..12),
        raw_target in prop::collection::vec(-10i16..=10, 0..18),
        cutoff in 0u16..=80,
    ) {
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let target: Vec<f64> = raw_target.into_iter().map(f64::from).collect();
        let config = FrechetConfig::new();
        let cutoff = f64::from(cutoff);
        let mut automaton = ElasticOnlineAutomaton::new(
            &query,
            config,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("small finite Frechet query has an online state");
        prop_assert!(close(
            automaton.observation().distance_within_cutoff,
            config.distance_with_cutoff(&query, &[], cutoff),
        ));
        for (index, sample) in target.iter().copied().enumerate() {
            let outcome = automaton.advance(sample).expect("generated sample is finite");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small Frechet transition");
                return Ok(());
            };
            prop_assert_eq!(value.consumed_target_len, index + 1);
            prop_assert!(usage.queue_entries <= query.len());
            prop_assert!(usage.work_units <= query.len());
            prop_assert!(close(
                value.distance_within_cutoff,
                config.distance_with_cutoff(&query, &target[..=index], cutoff),
            ));
        }
    }
}

#[test]
fn sparse_frontier_can_advance_below_the_dense_column_work_requirement() {
    let query = vec![0.0; 64];
    let mut automaton = ElasticOnlineAutomaton::new(
        &query,
        FrechetConfig::new(),
        0.0,
        OnlineAutomatonLimits {
            max_step_work_units: 1,
            ..OnlineAutomatonLimits::default()
        },
    )
    .expect("the fixed columns fit even though transition work is narrow");
    let outcome = automaton
        .advance(100.0)
        .expect("the target sample is finite");
    let OnlineStepOutcome::Advanced { value, usage } = outcome else {
        panic!("one dead-frontier row must fit the one-unit work ceiling");
    };
    assert_eq!(value.consumed_target_len, 1);
    assert_eq!(value.active_positions, 0);
    assert_eq!(usage.work_units, 1);
}

#[test]
fn sparse_work_exhaustion_is_pre_evaluation_and_transactional() {
    let query = vec![0.0; 64];
    let mut automaton = ElasticOnlineAutomaton::new(
        &query,
        FrechetConfig::new(),
        0.0,
        OnlineAutomatonLimits {
            max_step_work_units: 3,
            ..OnlineAutomatonLimits::default()
        },
    )
    .expect("the fixed machine state fits");
    let before = automaton.observation();

    for _ in 0..2 {
        let outcome = automaton.advance(0.0).expect("the target sample is finite");
        let OnlineStepOutcome::Incomplete { reason, usage } = outcome else {
            panic!("the fourth reachable row must not be evaluated");
        };
        assert_eq!(
            reason,
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: 3,
                requested: 4,
            }
        );
        assert_eq!(usage.work_units, 3);
        assert_eq!(automaton.observation(), before);
    }
}

#[test]
fn unknown_length_stream_does_not_grow_retained_storage() {
    let mut automaton = ElasticOnlineAutomaton::new(
        &[0.0, 1.0, 2.0],
        FrechetConfig::new(),
        100.0,
        OnlineAutomatonLimits::default(),
    )
    .expect("small finite Frechet query has an online state");
    let retained = automaton.scratch_bytes();
    for index in 0..100_000usize {
        assert!(automaton
            .advance((index % 3) as f64)
            .expect("stream sample is finite")
            .advanced());
        assert_eq!(automaton.scratch_bytes(), retained);
    }
    assert_eq!(automaton.observation().consumed_target_len, 100_000);
}

#[test]
fn unbounded_cutoff_is_rejected_until_kernel_arithmetic_is_tagged() {
    let error = ElasticOnlineAutomaton::new(
        &[1.0],
        FrechetConfig::new(),
        f64::INFINITY,
        OnlineAutomatonLimits::default(),
    )
    .expect_err("legacy positive infinity cannot distinguish overflow from unreachable state");
    assert!(matches!(
        error,
        liblevenshtein::time_series::TemporalAutomatonError::Validation(
            liblevenshtein::time_series::TemporalValidationError::InvalidCutoff
        )
    ));
}
