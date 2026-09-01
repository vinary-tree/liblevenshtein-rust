//! Strict scalar scorers against their independent legacy recurrences.

use liblevenshtein::time_series::{
    DtwConfig, ErpConfig, ExactDecision, FrechetConfig, IncompleteReason, OperationOutcome,
    ResourceKind, ResourceLimits, TwedConfig,
};
use proptest::prelude::*;

fn exact_distance(outcome: OperationOutcome<ExactDecision>) -> Option<f64> {
    match outcome {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => Some(distance),
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        } => None,
        other => panic!("small generated bounded score did not complete: {other:?}"),
    }
}

fn finite_series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-12_i16..=12, 0..12)
        .prop_map(|values| values.into_iter().map(f64::from).collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn bounded_erp_equals_scalar_recurrence(
        query in finite_series(),
        candidate in finite_series(),
        gap in -4_i8..=4,
        cutoff in 0_u16..=240,
    ) {
        let config = ErpConfig::new(f64::from(gap));
        let cutoff = f64::from(cutoff);
        let observed = exact_distance(config.distance_bounded(
            &query, &candidate, cutoff, ResourceLimits::default(),
        ).unwrap());
        prop_assert_eq!(observed, config.distance_with_cutoff(&query, &candidate, cutoff));
    }

    #[test]
    fn bounded_unit_grid_twed_equals_scalar_recurrence(
        query in finite_series(),
        candidate in finite_series(),
        cutoff in 0_u16..=320,
    ) {
        let config = TwedConfig::new(0.5, 1.0);
        let cutoff = f64::from(cutoff);
        let observed = exact_distance(config.distance_bounded(
            &query, &candidate, cutoff, ResourceLimits::default(),
        ).unwrap());
        prop_assert_eq!(observed, config.distance_with_cutoff(&query, &candidate, cutoff));
    }

    #[test]
    fn bounded_scalar_frechet_equals_scalar_recurrence(
        query in finite_series(),
        candidate in finite_series(),
        cutoff in 0_u16..=80,
    ) {
        let config = FrechetConfig::new();
        let cutoff = f64::from(cutoff);
        let outcome = config.distance_bounded(
            &query, &candidate, cutoff, ResourceLimits::default(),
        ).unwrap();
        let expected = config.distance_with_cutoff(&query, &candidate, cutoff);
        if query.is_empty() ^ candidate.is_empty() {
            let tagged_no_alignment = matches!(
                &outcome,
                OperationOutcome::Complete {
                    value: ExactDecision::NoFiniteAlignment,
                    ..
                }
            );
            prop_assert!(tagged_no_alignment);
            prop_assert_eq!(expected, None);
        } else {
            prop_assert_eq!(exact_distance(outcome), expected);
        }
    }

    #[test]
    fn bounded_banded_dtw_equals_scalar_recurrence(
        query in finite_series(),
        candidate in finite_series(),
        band in 0_usize..=8,
        cutoff in 0_u16..=120,
    ) {
        let config = DtwConfig::new(band);
        let cutoff = f64::from(cutoff);
        let outcome = config.distance_bounded(
            &query, &candidate, cutoff, ResourceLimits::default(),
        ).unwrap();
        let expected = config.distance_with_cutoff(&query, &candidate, cutoff);
        let structurally_impossible = (query.is_empty() ^ candidate.is_empty())
            || (!query.is_empty()
                && !candidate.is_empty()
                && query.len().abs_diff(candidate.len()) > band);
        if structurally_impossible {
            let tagged_no_alignment = matches!(
                &outcome,
                OperationOutcome::Complete {
                    value: ExactDecision::NoFiniteAlignment,
                    ..
                }
            );
            prop_assert!(tagged_no_alignment);
            prop_assert_eq!(expected, None);
        } else {
            prop_assert_eq!(exact_distance(outcome), expected);
        }
    }
}

#[test]
fn structural_and_numeric_top_classification_truth_table_is_fail_closed() {
    let dtw = DtwConfig::new(0);
    for cutoff in [0.0, f64::INFINITY] {
        assert!(matches!(
            dtw.distance_bounded(&[0.0], &[0.0, 0.0], cutoff, ResourceLimits::default())
                .expect("finite inputs and lawful cutoff"),
            OperationOutcome::Complete {
                value: ExactDecision::NoFiniteAlignment,
                ..
            }
        ));
    }

    assert!(matches!(
        dtw.distance_bounded(&[0.0], &[1.0], 0.5, ResourceLimits::default())
            .expect("finite inputs and cutoff"),
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        }
    ));
    assert!(matches!(
        dtw.distance_bounded(&[0.0], &[1.0], f64::INFINITY, ResourceLimits::default())
            .expect("finite inputs and TOP cutoff"),
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance: 1.0, .. },
            ..
        }
    ));
    assert!(matches!(
        dtw.distance_bounded(
            &[-f64::MAX],
            &[f64::MAX],
            f64::INFINITY,
            ResourceLimits::default(),
        )
        .expect("finite inputs and TOP cutoff"),
        OperationOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            ..
        }
    ));
}

#[test]
fn scalar_resource_exhaustion_is_never_an_above_cutoff_decision() {
    let limits = ResourceLimits {
        max_dp_cells: 8,
        ..ResourceLimits::default()
    };
    let outcome = ErpConfig::default_gap()
        .distance_bounded(&[0.0, 1.0], &[0.0, 2.0], 10.0, limits)
        .expect("finite inputs are valid");
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::DpCells,
                limit: 8,
                requested: 9,
            },
            continuation: None,
            ..
        }
    ));
}

#[test]
fn scalar_kernel_cutoffs_reject_the_adjacent_binary64_cost() {
    let exact_sample = 1.0_f64.next_up();
    let query = [0.0];
    let candidate = [exact_sample];

    let erp = ErpConfig::new(0.0);
    let erp_exact = erp.distance(&query, &candidate);
    assert_eq!(erp_exact, exact_sample);
    assert_eq!(erp.distance_with_cutoff(&query, &candidate, 1.0), None);
    assert_eq!(
        erp.distance_with_cutoff(&query, &candidate, erp_exact),
        Some(erp_exact)
    );

    let twed = TwedConfig::new(0.5, 1.0);
    let twed_exact = twed.distance(&query, &candidate);
    assert!(twed_exact.is_finite() && twed_exact > 0.0);
    assert_eq!(
        twed.distance_with_cutoff(&query, &candidate, twed_exact.next_down()),
        None
    );
    assert_eq!(
        twed.distance_with_cutoff(&query, &candidate, twed_exact),
        Some(twed_exact)
    );

    let frechet = FrechetConfig::new();
    let frechet_exact = frechet.distance(&query, &candidate);
    assert_eq!(frechet_exact, exact_sample);
    assert_eq!(frechet.distance_with_cutoff(&query, &candidate, 1.0), None);
    assert_eq!(
        frechet.distance_with_cutoff(&query, &candidate, frechet_exact),
        Some(frechet_exact)
    );

    let dtw = DtwConfig::new(0);
    let dtw_exact = dtw.distance(&query, &candidate);
    assert_eq!(dtw_exact, exact_sample);
    assert_eq!(dtw.distance_with_cutoff(&query, &candidate, 1.0), None);
    assert_eq!(
        dtw.distance_with_cutoff(&query, &candidate, dtw_exact),
        Some(dtw_exact)
    );
}
