//! Prefix correspondence and stability properties for explicit-time TWED.

use liblevenshtein::time_series::{
    ExactDecision, MetricTimestampedTwedConfig, MetricTwedConfig, OnlineAutomatonLimits,
    OnlineStepOutcome, OperationOutcome, ResourceLimits, TimestampUnit, TimestampedSeries,
    TimestampedTwedError, TimestampedTwedOnlineAutomaton,
};
use proptest::prelude::*;

fn points(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<(i16, u8)>> {
    prop::collection::vec((-20i16..=20, 1u8..=5), min_len..max_len)
}

fn split_points(points: &[(i16, u8)], origin: f64) -> (Vec<f64>, Vec<f64>) {
    let mut time = origin;
    let mut values = Vec::with_capacity(points.len());
    let mut timestamps = Vec::with_capacity(points.len());
    for &(value, delta) in points {
        time += f64::from(delta);
        values.push(f64::from(value));
        timestamps.push(time);
    }
    (values, timestamps)
}

fn bounded_distance(
    config: MetricTimestampedTwedConfig,
    query: &TimestampedSeries,
    values: &[f64],
    timestamps: &[f64],
    cutoff: f64,
) -> Option<f64> {
    let target = TimestampedSeries::try_new_with_origin(
        values,
        timestamps,
        query.unit(),
        query.origin(),
        ResourceLimits::default(),
    )
    .expect("generated target has finite, strictly increasing timestamps");
    match config
        .distance_bounded(query, &target, cutoff, ResourceLimits::default())
        .expect("generated operands share a physical frame")
    {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => Some(distance),
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        } => None,
        other => panic!("small generated exact TWED comparison did not complete: {other:?}"),
    }
}

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
    fn every_explicit_time_prefix_matches_the_independent_scalar_dp(
        query_points in points(1, 11),
        target_points in points(0, 17),
        cutoff in 0u16..=240,
    ) {
        let origin = 0.0;
        let (query_values, query_times) = split_points(&query_points, origin);
        let (target_values, target_times) = split_points(&target_points, origin);
        let query = TimestampedSeries::try_new_with_origin(
            &query_values,
            &query_times,
            TimestampUnit::Seconds,
            origin,
            ResourceLimits::default(),
        )
        .expect("generated query has finite, strictly increasing timestamps");
        let config = MetricTimestampedTwedConfig::try_new(0.5, 1.0)
            .expect("positive stiffness and nonnegative gap penalty are metric");
        let cutoff = f64::from(cutoff);
        let mut automaton = TimestampedTwedOnlineAutomaton::new(
            query.clone(),
            TimestampUnit::Seconds,
            origin,
            config,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("small generated query fits default online limits");
        prop_assert_eq!(automaton.observation().distance_within_cutoff, None);
        let retained = automaton.scratch_bytes();

        for index in 0..target_values.len() {
            let outcome = automaton
                .advance(target_values[index], target_times[index])
                .expect("generated target timestamp is strictly increasing");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small timestamped transition");
                return Ok(());
            };
            let exact = bounded_distance(
                config,
                &query,
                &target_values[..=index],
                &target_times[..=index],
                cutoff,
            );
            prop_assert!(close(value.distance_within_cutoff, exact));
            prop_assert_eq!(value.consumed_target_len, index + 1);
            prop_assert!(value.active_positions <= query_values.len() + 1);
            prop_assert!(usage.queue_entries <= query_values.len() + 1);
            prop_assert!(usage.work_units <= query_values.len() + 1);
            prop_assert_eq!(automaton.scratch_bytes(), retained);
        }
    }

    #[test]
    fn unit_elapsed_online_twed_matches_unit_grid_twed(
        query_values in prop::collection::vec(-12i16..=12, 1..11),
        target_values in prop::collection::vec(-12i16..=12, 1..17),
        cutoff in 0u16..=180,
    ) {
        let query_values: Vec<f64> = query_values.into_iter().map(f64::from).collect();
        let target_values: Vec<f64> = target_values.into_iter().map(f64::from).collect();
        let query_times: Vec<f64> = (1..=query_values.len()).map(|time| time as f64).collect();
        let query = TimestampedSeries::try_new(
            &query_values,
            &query_times,
            TimestampUnit::Seconds,
            ResourceLimits::default(),
        )
        .expect("unit-grid query timestamps are strictly increasing");
        let physical = MetricTimestampedTwedConfig::try_new(0.5, 1.0)
            .expect("physical metric parameters are valid");
        let unit_grid = MetricTwedConfig::try_new(0.5, 1.0)
            .expect("unit-grid metric parameters are valid");
        let cutoff = f64::from(cutoff);
        let mut automaton = TimestampedTwedOnlineAutomaton::new(
            query,
            TimestampUnit::Seconds,
            0.0,
            physical,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("unit-grid query fits default online limits");

        for (index, value) in target_values.iter().copied().enumerate() {
            let outcome = automaton
                .advance(value, (index + 1) as f64)
                .expect("unit-grid target timestamps are strictly increasing");
            let OnlineStepOutcome::Advanced { value, .. } = outcome else {
                prop_assert!(false, "default limits rejected a unit-grid transition");
                return Ok(());
            };
            prop_assert!(close(
                value.distance_within_cutoff,
                unit_grid.distance_with_cutoff(&query_values, &target_values[..=index], cutoff),
            ));
        }
    }
}

#[test]
fn timestamped_sparse_work_limit_is_pre_evaluation_and_transactional() {
    let query = TimestampedSeries::try_new(
        &[0.0, 0.0, 0.0, 0.0],
        &[1.0, 2.0, 3.0, 4.0],
        TimestampUnit::Seconds,
        ResourceLimits::default(),
    )
    .expect("query is valid");
    let config =
        MetricTimestampedTwedConfig::try_new(0.5, 0.0).expect("metric parameters are valid");
    let mut automaton = TimestampedTwedOnlineAutomaton::new(
        query,
        TimestampUnit::Seconds,
        0.0,
        config,
        100.0,
        OnlineAutomatonLimits {
            max_step_work_units: 2,
            ..OnlineAutomatonLimits::default()
        },
    )
    .expect("fixed state fits independently of the step-work ceiling");
    let before = automaton.observation();
    let outcome = automaton
        .advance(0.0, 1.0)
        .expect("target timestamp is valid");
    let OnlineStepOutcome::Incomplete { reason, usage } = outcome else {
        panic!("the third reachable row must not be evaluated");
    };
    assert_eq!(usage.work_units, 2);
    assert!(matches!(
        reason,
        liblevenshtein::time_series::IncompleteReason::BudgetExceeded {
            resource: liblevenshtein::time_series::ResourceKind::WorkUnits,
            limit: 2,
            requested: 3,
        }
    ));
    assert_eq!(automaton.observation(), before);
}

#[test]
fn invalid_timestamp_is_transactional_and_physical_frames_are_typed() {
    let query = TimestampedSeries::try_new(
        &[1.0, 2.0],
        &[1.0, 2.0],
        TimestampUnit::Seconds,
        ResourceLimits::default(),
    )
    .expect("query is valid");
    let config =
        MetricTimestampedTwedConfig::try_new(1.0, 0.0).expect("metric parameters are valid");
    assert!(matches!(
        TimestampedTwedOnlineAutomaton::new(
            query.clone(),
            TimestampUnit::Milliseconds,
            0.0,
            config,
            10.0,
            OnlineAutomatonLimits::default(),
        ),
        Err(TimestampedTwedError::MixedUnits)
    ));
    assert!(matches!(
        TimestampedTwedOnlineAutomaton::new(
            query.clone(),
            TimestampUnit::Seconds,
            1.0,
            config,
            10.0,
            OnlineAutomatonLimits::default(),
        ),
        Err(TimestampedTwedError::MixedOrigins)
    ));

    let mut automaton = TimestampedTwedOnlineAutomaton::new(
        query,
        TimestampUnit::Seconds,
        0.0,
        config,
        10.0,
        OnlineAutomatonLimits::default(),
    )
    .expect("matching physical frame is valid");
    assert!(automaton
        .advance(1.0, 1.0)
        .expect("first target point is valid")
        .advanced());
    let before = automaton.observation();
    assert_eq!(
        automaton.advance(2.0, 1.0),
        Err(TimestampedTwedError::NonMonotoneTimestamp { index: 1 })
    );
    assert_eq!(automaton.observation(), before);
}

#[test]
fn unknown_length_timestamped_stream_has_constant_retained_storage() {
    let query = TimestampedSeries::try_new(
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0, 3.0],
        TimestampUnit::Seconds,
        ResourceLimits::default(),
    )
    .expect("query is valid");
    let config =
        MetricTimestampedTwedConfig::try_new(0.5, 1.0).expect("metric parameters are valid");
    let mut automaton = TimestampedTwedOnlineAutomaton::new(
        query,
        TimestampUnit::Seconds,
        0.0,
        config,
        1_000.0,
        OnlineAutomatonLimits::default(),
    )
    .expect("query fits default online limits");
    let retained = automaton.scratch_bytes();
    for index in 0..100_000usize {
        assert!(automaton
            .advance((index % 3) as f64, (index + 1) as f64)
            .expect("timestamps increase by one canonical unit")
            .advanced());
        assert_eq!(automaton.scratch_bytes(), retained);
    }
    assert_eq!(automaton.observation().consumed_target_len, 100_000);
}
