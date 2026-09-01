//! Whole-point online vector Fréchet correspondence and stream stability.

use liblevenshtein::time_series::{
    ExactDecision, L1GroundMetric, OnlineAutomatonLimits, OnlineStepOutcome, OperationOutcome,
    ResourceLimits, VectorFrechetMetric, VectorFrechetOnlineAutomaton, VectorFrechetPath,
    VectorMetricError, VectorSample,
};
use proptest::prelude::*;

fn sample(row: &[i8]) -> VectorSample {
    let coordinates: Vec<_> = row.iter().copied().map(f64::from).collect();
    VectorSample::try_new(&coordinates, ResourceLimits::default())
        .expect("generated finite positive-dimensional sample is valid")
}

fn path(rows: &[Vec<i8>]) -> VectorFrechetPath {
    VectorFrechetPath::try_new(
        rows.iter().map(|row| sample(row)).collect(),
        ResourceLimits::default(),
    )
    .expect("generated equal-dimensional nonempty path is valid")
}

fn exact_prefix(query: &VectorFrechetPath, target: &[Vec<i8>], cutoff: f64) -> Option<f64> {
    let target = path(target);
    match VectorFrechetMetric::new(L1GroundMetric)
        .distance_bounded(query, &target, cutoff, ResourceLimits::default())
        .expect("generated vector paths share one dimension")
    {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => Some(distance),
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        } => None,
        other => panic!("small generated vector comparison did not complete: {other:?}"),
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
    fn every_vector_prefix_matches_the_independent_full_path_scorer(
        dimension in 1usize..5,
        query_flat in prop::collection::vec(-8i8..=8, 1..40),
        target_flat in prop::collection::vec(-8i8..=8, 1..64),
        cutoff in 0u16..=160,
    ) {
        let query_rows: Vec<Vec<i8>> = query_flat
            .chunks(dimension)
            .filter(|row| row.len() == dimension)
            .take(10)
            .map(<[i8]>::to_vec)
            .collect();
        let target_rows: Vec<Vec<i8>> = target_flat
            .chunks(dimension)
            .filter(|row| row.len() == dimension)
            .take(16)
            .map(<[i8]>::to_vec)
            .collect();
        prop_assume!(!query_rows.is_empty() && !target_rows.is_empty());
        let query = path(&query_rows);
        let cutoff = f64::from(cutoff);
        let query_len = query.samples().len();
        let retained_dimension = query.dimension();
        let mut automaton = VectorFrechetOnlineAutomaton::new(
            query.clone(),
            L1GroundMetric,
            cutoff,
            OnlineAutomatonLimits::default(),
        )
        .expect("small generated fixed query fits default online limits");
        let retained = automaton.scratch_bytes();
        for (index, row) in target_rows.iter().enumerate() {
            let target = sample(row);
            let outcome = automaton
                .advance(&target)
                .expect("generated target has the fixed dimension");
            let OnlineStepOutcome::Advanced { value, usage } = outcome else {
                prop_assert!(false, "default limits rejected a small vector transition");
                return Ok(());
            };
            prop_assert!(close(
                value.distance_within_cutoff,
                exact_prefix(&query, &target_rows[..=index], cutoff),
            ));
            prop_assert_eq!(value.consumed_target_len, index + 1);
            prop_assert!(value.active_positions <= query_len);
            prop_assert!(usage.work_units <= query_len * retained_dimension);
            prop_assert_eq!(automaton.scratch_bytes(), retained);
        }
    }
}

#[test]
fn vector_dimension_error_is_transactional_and_stream_memory_is_constant() {
    let query = VectorFrechetPath::try_from_rows(
        &[&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0]],
        ResourceLimits::default(),
    )
    .expect("query is valid");
    let mut automaton = VectorFrechetOnlineAutomaton::new(
        query,
        L1GroundMetric,
        100.0,
        OnlineAutomatonLimits::default(),
    )
    .expect("query fits default limits");
    let wrong = VectorSample::try_new(&[0.0], ResourceLimits::default())
        .expect("one-dimensional point is independently valid");
    let before = automaton.observation();
    assert_eq!(
        automaton.advance(&wrong),
        Err(VectorMetricError::DimensionMismatch {
            expected: 2,
            observed: 1,
        })
    );
    assert_eq!(automaton.observation(), before);

    let retained = automaton.scratch_bytes();
    for index in 0..100_000usize {
        let point = VectorSample::try_new(
            &[(index % 3) as f64, (index % 5) as f64],
            ResourceLimits::default(),
        )
        .expect("stream point is finite");
        assert!(automaton
            .advance(&point)
            .expect("stream dimension is fixed")
            .advanced());
        assert_eq!(automaton.scratch_bytes(), retained);
    }
    assert_eq!(automaton.observation().consumed_target_len, 100_000);
}
