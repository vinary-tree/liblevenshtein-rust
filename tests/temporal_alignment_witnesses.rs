use std::thread;

use liblevenshtein::time_series::{
    DtwConfig, ErpConfig, ExactDecision, FrechetConfig, IncompleteReason,
    MetricTimestampedTwedConfig, OperationOutcome, ResourceKind, ResourceLimits,
    TemporalAlignmentKind, TemporalAlignmentOperation, TemporalAlignmentStep,
    TemporalAlignmentWitness, TemporalWitnessReplayError, TimestampUnit, TimestampedSeries,
    TwedConfig, TEMPORAL_ALIGNMENT_WITNESS_VERSION,
};
use proptest::prelude::*;

fn complete_witness(
    outcome: OperationOutcome<ExactDecision<TemporalAlignmentWitness>>,
) -> (f64, TemporalAlignmentWitness) {
    match outcome {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, witness },
            ..
        } => (distance, witness),
        other => panic!("expected a complete witnessed alignment, got {other:?}"),
    }
}

fn small_series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-8_i8..=8, 0..8)
        .prop_map(|values| values.into_iter().map(f64::from).collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn scalar_witnesses_replay_to_independent_score_surfaces(
        query in small_series(),
        candidate in small_series(),
        gap in -4_i8..=4,
        stiffness_quarters in 0_u8..=8,
        lambda_halves in 0_u8..=8,
        band in 0_usize..8,
    ) {
        let limits = ResourceLimits::default();

        let erp = ErpConfig::new(f64::from(gap));
        let (distance, witness) = complete_witness(
            erp.distance_with_alignment_bounded(
                &query,
                &candidate,
                f64::INFINITY,
                limits,
            )
            .expect("finite ERP inputs are valid"),
        );
        prop_assert_eq!(distance.to_bits(), erp.distance(&query, &candidate).to_bits());
        prop_assert_eq!(
            witness.replay_erp(&query, &candidate, &erp).unwrap().to_bits(),
            distance.to_bits(),
        );

        let twed = TwedConfig::new(
            f64::from(stiffness_quarters) / 4.0,
            f64::from(lambda_halves) / 2.0,
        );
        let (distance, witness) = complete_witness(
            twed.distance_with_alignment_bounded(
                &query,
                &candidate,
                f64::INFINITY,
                limits,
            )
            .expect("finite unit-grid TWED inputs are valid"),
        );
        prop_assert_eq!(distance.to_bits(), twed.distance(&query, &candidate).to_bits());
        prop_assert_eq!(
            witness
                .replay_unit_grid_twed(&query, &candidate, &twed)
                .unwrap()
                .to_bits(),
            distance.to_bits(),
        );

        let frechet = FrechetConfig::new();
        let frechet_outcome = frechet
            .distance_with_alignment_bounded(&query, &candidate, f64::INFINITY, limits)
            .expect("finite Frechet inputs are valid");
        if query.is_empty() != candidate.is_empty() {
            let no_alignment = matches!(
                frechet_outcome,
                OperationOutcome::Complete {
                    value: ExactDecision::NoFiniteAlignment,
                    ..
                }
            );
            prop_assert!(no_alignment);
        } else {
            let (distance, witness) = complete_witness(frechet_outcome);
            prop_assert_eq!(distance.to_bits(), frechet.distance(&query, &candidate).to_bits());
            prop_assert_eq!(
                witness
                    .replay_discrete_frechet(&query, &candidate, &frechet)
                    .unwrap()
                    .to_bits(),
                distance.to_bits(),
            );
        }

        let dtw = DtwConfig::new(band);
        let dtw_outcome = dtw
            .distance_with_alignment_bounded(&query, &candidate, f64::INFINITY, limits)
            .expect("finite DTW inputs are valid");
        let connected = (query.is_empty() && candidate.is_empty())
            || (!query.is_empty()
                && !candidate.is_empty()
                && query.len().abs_diff(candidate.len()) <= band);
        if connected {
            let (distance, witness) = complete_witness(dtw_outcome);
            prop_assert_eq!(distance.to_bits(), dtw.distance(&query, &candidate).to_bits());
            prop_assert_eq!(
                witness
                    .replay_banded_dtw(&query, &candidate, &dtw)
                    .unwrap()
                    .to_bits(),
                distance.to_bits(),
            );
        } else {
            let no_alignment = matches!(
                dtw_outcome,
                OperationOutcome::Complete {
                    value: ExactDecision::NoFiniteAlignment,
                    ..
                }
            );
            prop_assert!(no_alignment);
        }
    }
}

#[test]
fn timestamped_twed_witness_replays_to_score_surface() {
    let limits = ResourceLimits::default();
    let query = TimestampedSeries::try_new(
        &[0.0, 2.0, 1.0],
        &[1.0, 3.0, 8.0],
        TimestampUnit::Seconds,
        limits,
    )
    .unwrap();
    let candidate = TimestampedSeries::try_new(
        &[0.0, 1.0, 1.0, 3.0],
        &[1.0, 2.0, 6.0, 9.0],
        TimestampUnit::Seconds,
        limits,
    )
    .unwrap();
    let config = MetricTimestampedTwedConfig::try_new(0.5, 1.0).unwrap();
    let score = match config
        .distance_bounded(&query, &candidate, f64::INFINITY, limits)
        .unwrap()
    {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => distance,
        other => panic!("timestamped score must complete: {other:?}"),
    };
    let (distance, witness) = complete_witness(
        config
            .distance_with_alignment_bounded(&query, &candidate, f64::INFINITY, limits)
            .unwrap(),
    );
    assert_eq!(distance.to_bits(), score.to_bits());
    assert_eq!(witness.kind(), TemporalAlignmentKind::TimestampedTwed);
    assert_eq!(
        witness
            .replay_timestamped_twed(&query, &candidate, &config)
            .unwrap()
            .to_bits(),
        distance.to_bits()
    );
}

#[test]
fn untrusted_version_kernel_operation_endpoint_and_local_cost_fail_closed() {
    let config = ErpConfig::new(0.0);
    let (_, valid) = complete_witness(
        config
            .distance_with_alignment_bounded(
                &[1.0],
                &[2.0],
                f64::INFINITY,
                ResourceLimits::default(),
            )
            .unwrap(),
    );
    let step = valid.steps()[0];

    let wrong_version = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION + 1,
        valid.kind(),
        valid.steps().to_vec(),
    );
    assert_eq!(
        wrong_version.replay_erp(&[1.0], &[2.0], &config),
        Err(TemporalWitnessReplayError::UnsupportedVersion {
            found: TEMPORAL_ALIGNMENT_WITNESS_VERSION + 1,
        })
    );

    let wrong_kernel = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION,
        TemporalAlignmentKind::UnitGridTwed,
        valid.steps().to_vec(),
    );
    assert!(matches!(
        wrong_kernel.replay_erp(&[1.0], &[2.0], &config),
        Err(TemporalWitnessReplayError::KernelMismatch { .. })
    ));

    let bad_endpoint = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION,
        TemporalAlignmentKind::Erp,
        vec![TemporalAlignmentStep::from_raw_parts(
            step.operation(),
            Some(1),
            step.candidate_endpoint(),
            step.local_cost_bits(),
        )],
    );
    assert_eq!(
        bad_endpoint.replay_erp(&[1.0], &[2.0], &config),
        Err(TemporalWitnessReplayError::MalformedEndpoint { step_index: 0 })
    );

    let bad_local = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION,
        TemporalAlignmentKind::Erp,
        vec![TemporalAlignmentStep::from_raw_parts(
            step.operation(),
            step.query_endpoint(),
            step.candidate_endpoint(),
            step.local_cost_bits() ^ 1,
        )],
    );
    assert_eq!(
        bad_local.replay_erp(&[1.0], &[2.0], &config),
        Err(TemporalWitnessReplayError::LocalCostMismatch { step_index: 0 })
    );

    let dtw = DtwConfig::new(1);
    let bad_operation = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION,
        TemporalAlignmentKind::BandedDtw,
        vec![TemporalAlignmentStep::from_raw_parts(
            TemporalAlignmentOperation::AdvanceQuery,
            Some(0),
            None,
            0.0_f64.to_bits(),
        )],
    );
    assert_eq!(
        bad_operation.replay_banded_dtw(&[1.0], &[1.0], &dtw),
        Err(TemporalWitnessReplayError::MalformedOperation { step_index: 0 })
    );
}

#[test]
fn tie_breaking_is_align_then_query_then_candidate_and_repeatable() {
    let limits = ResourceLimits::default();
    let erp = ErpConfig::new(0.0);
    let first = complete_witness(
        erp.distance_with_alignment_bounded(&[0.0, 0.0], &[0.0, 0.0], f64::INFINITY, limits)
            .unwrap(),
    )
    .1;
    let second = complete_witness(
        erp.distance_with_alignment_bounded(&[0.0, 0.0], &[0.0, 0.0], f64::INFINITY, limits)
            .unwrap(),
    )
    .1;
    assert_eq!(first, second);
    assert!(first
        .steps()
        .iter()
        .all(|step| step.operation() == TemporalAlignmentOperation::Align));
}

#[test]
fn witness_and_trace_resource_failures_are_explicit_and_transactional() {
    let witness_limited = ResourceLimits {
        max_witness_bytes: 0,
        ..ResourceLimits::default()
    };
    let outcome = ErpConfig::new(0.0)
        .distance_with_alignment_bounded(&[0.0], &[0.0], f64::INFINITY, witness_limited)
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WitnessBytes,
                limit: 0,
                ..
            },
            continuation: None,
            ..
        }
    ));

    let scratch_limited = ResourceLimits {
        max_scratch_bytes: 1,
        ..ResourceLimits::default()
    };
    let outcome = FrechetConfig::new()
        .distance_with_alignment_bounded(&[0.0, 1.0], &[0.0, 1.0], f64::INFINITY, scratch_limited)
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: 1,
                ..
            },
            continuation: None,
            ..
        }
    ));

    let no_witness_needed = ErpConfig::new(0.0)
        .distance_with_alignment_bounded(&[0.0], &[10.0], 1.0, witness_limited)
        .unwrap();
    assert!(matches!(
        no_witness_needed,
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        }
    ));

    let band_limited = ResourceLimits {
        max_band_width: 1,
        ..ResourceLimits::default()
    };
    let outcome = DtwConfig::new(2)
        .distance_with_alignment_bounded(&[0.0], &[0.0], f64::INFINITY, band_limited)
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::BandWidth,
                limit: 1,
                requested: 2,
            },
            ..
        }
    ));
}

#[test]
fn long_extraction_and_replay_fit_a_constrained_call_stack() {
    thread::Builder::new()
        .name("temporal-witness-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            let len = 50_000;
            let zeros = vec![0.0; len];
            let limits = ResourceLimits::default();

            let erp = ErpConfig::new(0.0);
            let (_, witness) = complete_witness(
                erp.distance_with_alignment_bounded(&[0.0], &zeros, f64::INFINITY, limits)
                    .unwrap(),
            );
            assert_eq!(witness.len(), len);
            assert_eq!(witness.replay_erp(&[0.0], &zeros, &erp).unwrap(), 0.0);

            let twed = TwedConfig::new(0.25, 0.5);
            let (distance, witness) = complete_witness(
                twed.distance_with_alignment_bounded(&[0.0], &zeros, f64::INFINITY, limits)
                    .unwrap(),
            );
            assert_eq!(
                witness
                    .replay_unit_grid_twed(&[0.0], &zeros, &twed)
                    .unwrap(),
                distance
            );

            let frechet = FrechetConfig::new();
            let (_, witness) = complete_witness(
                frechet
                    .distance_with_alignment_bounded(&[0.0], &zeros, f64::INFINITY, limits)
                    .unwrap(),
            );
            assert_eq!(
                witness
                    .replay_discrete_frechet(&[0.0], &zeros, &frechet)
                    .unwrap(),
                0.0
            );

            let dtw = DtwConfig::new(0);
            let (_, witness) = complete_witness(
                dtw.distance_with_alignment_bounded(&zeros, &zeros, f64::INFINITY, limits)
                    .unwrap(),
            );
            assert_eq!(witness.len(), len);
            assert_eq!(
                witness.replay_banded_dtw(&zeros, &zeros, &dtw).unwrap(),
                0.0
            );

            let timestamps: Vec<_> = (1..=len).map(|index| index as f64).collect();
            let query =
                TimestampedSeries::try_new(&[0.0], &[1.0], TimestampUnit::Seconds, limits).unwrap();
            let candidate =
                TimestampedSeries::try_new(&zeros, &timestamps, TimestampUnit::Seconds, limits)
                    .unwrap();
            let timestamped = MetricTimestampedTwedConfig::try_new(0.25, 0.5).unwrap();
            let (distance, witness) = complete_witness(
                timestamped
                    .distance_with_alignment_bounded(&query, &candidate, f64::INFINITY, limits)
                    .unwrap(),
            );
            assert_eq!(
                witness
                    .replay_timestamped_twed(&query, &candidate, &timestamped)
                    .unwrap(),
                distance
            );
        })
        .expect("small-stack witness thread must spawn")
        .join()
        .expect("iterative witness extraction must not overflow the stack");
}
