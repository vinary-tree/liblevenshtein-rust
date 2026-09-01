use liblevenshtein::time_series::{
    ExactDecision, IncompleteReason, MsmAlignmentStep, MsmAlignmentWitness, MsmConfig,
    MsmWitnessReplayError, OperationOutcome, ResourceKind, ResourceLimits,
    MSM_ALIGNMENT_WITNESS_VERSION,
};
use proptest::prelude::*;

fn small_series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-12_i16..=12, 1..9)
        .prop_map(|values| values.into_iter().map(f64::from).collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn every_emitted_msm_witness_replays_to_the_independent_exact_score(
        query in small_series(),
        candidate in small_series(),
        half_cost in 0_u16..=16,
    ) {
        let config = MsmConfig::new(f64::from(half_cost) * 0.5);
        let expected = config.distance(&query, &candidate);
        let outcome = config
            .distance_with_alignment_bounded(
                &query,
                &candidate,
                f64::INFINITY,
                ResourceLimits::default(),
            )
            .expect("generated inputs satisfy the MSM domain");
        let OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, witness },
            usage,
        } = outcome else {
            prop_assert!(false, "finite generated inputs must have a witness");
            return Ok(());
        };
        prop_assert_eq!(distance.to_bits(), expected.to_bits());
        prop_assert_eq!(
            witness
                .replay(&query, &candidate, &config)
                .expect("emitted witness must replay")
                .to_bits(),
            distance.to_bits()
        );
        prop_assert!(witness.len() < query.len() + candidate.len());
        prop_assert_eq!(witness.steps().first(), Some(&MsmAlignmentStep::Move));
        prop_assert!(usage.witness_bytes >= std::mem::size_of::<MsmAlignmentWitness>());
    }
}

#[test]
fn malformed_or_incomplete_witnesses_fail_closed() {
    let error = MsmAlignmentWitness::default()
        .replay(&[1.0], &[1.0], &MsmConfig::new(1.0))
        .expect_err("empty path cannot certify nonempty operands");
    assert_eq!(
        error,
        MsmWitnessReplayError::MalformedPath { step_index: 0 }
    );
}

#[test]
fn unsupported_msm_witness_versions_fail_closed() {
    let witness = MsmAlignmentWitness::from_parts(
        MSM_ALIGNMENT_WITNESS_VERSION + 1,
        vec![MsmAlignmentStep::Move],
    );
    assert_eq!(
        witness.replay(&[1.0], &[1.0], &MsmConfig::new(1.0)),
        Err(MsmWitnessReplayError::UnsupportedVersion {
            found: MSM_ALIGNMENT_WITNESS_VERSION + 1,
        })
    );
}

#[test]
fn witness_limit_failure_is_tagged_and_retains_no_certificate() {
    let limits = ResourceLimits {
        max_witness_bytes: 0,
        ..ResourceLimits::default()
    };
    let outcome = MsmConfig::new(1.0)
        .distance_with_alignment_bounded(&[0.0, 1.0], &[0.0, 1.0], f64::INFINITY, limits)
        .expect("operands are valid");
    assert!(matches!(
        &outcome,
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
    assert_eq!(outcome.usage().witness_bytes, 0);
}

#[test]
fn cutoff_rejection_does_not_require_witness_storage() {
    let limits = ResourceLimits {
        max_witness_bytes: 0,
        ..ResourceLimits::default()
    };
    let outcome = MsmConfig::new(1.0)
        .distance_with_alignment_bounded(&[0.0], &[10.0], 1.0, limits)
        .expect("operands are valid");
    assert!(matches!(
        outcome,
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        }
    ));
}

#[test]
fn long_traceback_is_iterative_and_deterministic() {
    let query = [0.0];
    let candidate = vec![0.0; 100_000];
    let config = MsmConfig::new(1.0);
    let first = config
        .distance_with_alignment_bounded(
            &query,
            &candidate,
            f64::INFINITY,
            ResourceLimits::default(),
        )
        .expect("long bounded operands are valid");
    let OperationOutcome::Complete {
        value: ExactDecision::WithinCutoff { distance, witness },
        ..
    } = first
    else {
        panic!("long bounded witness must complete");
    };
    assert_eq!(witness.len(), candidate.len());
    assert_eq!(
        witness
            .replay(&query, &candidate, &config)
            .expect("long iterative witness must replay"),
        distance
    );
    assert_eq!(witness.steps()[0], MsmAlignmentStep::Move);
    assert!(witness.steps()[1..]
        .iter()
        .all(|step| *step == MsmAlignmentStep::Split));
}
