//! Analysis-only Soft-DTW against an independent full-matrix oracle.

use liblevenshtein::time_series::{
    IncompleteReason, OperationOutcome, ResourceKind, ResourceLimits, SoftDtwAnalysis,
    SoftDtwConfig, SoftDtwConfigError,
};
use proptest::prelude::*;

fn soft_min(left: f64, diagonal: f64, above: f64, gamma: f64) -> f64 {
    let minimum = left.min(diagonal).min(above);
    if !minimum.is_finite() {
        return f64::INFINITY;
    }
    let mass = ((minimum - left) / gamma).exp()
        + ((minimum - diagonal) / gamma).exp()
        + ((minimum - above) / gamma).exp();
    minimum - gamma * mass.ln()
}

fn full_matrix_oracle(left: &[f64], right: &[f64], gamma: f64) -> Option<f64> {
    if left.is_empty() || right.is_empty() {
        return (left.is_empty() && right.is_empty()).then_some(0.0);
    }
    let columns = right.len() + 1;
    let mut matrix = vec![f64::INFINITY; (left.len() + 1) * columns];
    matrix[0] = 0.0;
    for row in 1..=left.len() {
        for column in 1..=right.len() {
            let delta = left[row - 1] - right[column - 1];
            matrix[row * columns + column] = delta * delta
                + soft_min(
                    matrix[(row - 1) * columns + column - 1],
                    matrix[(row - 1) * columns + column],
                    matrix[row * columns + column - 1],
                    gamma,
                );
        }
    }
    Some(matrix[left.len() * columns + right.len()])
}

fn analyze(config: SoftDtwConfig, left: &[f64], right: &[f64]) -> Option<f64> {
    match config
        .analyze_bounded(left, right, ResourceLimits::default())
        .expect("generated samples are finite")
    {
        OperationOutcome::Complete {
            value: SoftDtwAnalysis::Finite { value },
            ..
        } => Some(value),
        OperationOutcome::Complete {
            value: SoftDtwAnalysis::NoFiniteAlignment,
            ..
        } => None,
        other => panic!("small generated Soft-DTW analysis did not complete: {other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn two_row_soft_dtw_equals_the_independent_full_matrix(
        left in prop::collection::vec(-8i8..=8, 0..9),
        right in prop::collection::vec(-8i8..=8, 0..9),
        gamma_quarters in 1u8..=16,
    ) {
        let left: Vec<_> = left.into_iter().map(f64::from).collect();
        let right: Vec<_> = right.into_iter().map(f64::from).collect();
        let gamma = f64::from(gamma_quarters) / 4.0;
        let config = SoftDtwConfig::try_new(gamma).expect("generated gamma is positive");
        let observed = analyze(config, &left, &right);
        let expected = full_matrix_oracle(&left, &right, gamma);
        match (observed, expected) {
            (Some(observed), Some(expected)) => {
                prop_assert!((observed - expected).abs() <= 1.0e-10);
            }
            (None, None) => {}
            mismatch => prop_assert!(false, "alignment-domain mismatch: {mismatch:?}"),
        }
        let reverse = analyze(config, &right, &left);
        match (observed, reverse) {
            (Some(observed), Some(reverse)) => {
                prop_assert!((observed - reverse).abs() <= 1.0e-10);
            }
            (None, None) => {}
            mismatch => prop_assert!(false, "symmetry mismatch: {mismatch:?}"),
        }
    }
}

#[test]
fn soft_dtw_is_explicitly_bounded_iterative_and_analysis_only() {
    assert_eq!(
        SoftDtwConfig::try_new(0.0),
        Err(SoftDtwConfigError::InvalidGamma)
    );
    let config = SoftDtwConfig::try_new(1.0).expect("positive gamma is valid");
    let limits = ResourceLimits {
        max_work_units: 3,
        ..ResourceLimits::default()
    };
    let outcome = config
        .analyze_bounded(&[0.0, 1.0], &[0.0, 1.0], limits)
        .expect("inputs are finite");
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: 3,
                requested: 4,
            },
            continuation: None,
            ..
        }
    ));

    let long = vec![0.0; 100_000];
    assert!(matches!(
        config
            .analyze_bounded(&[0.0], &long, ResourceLimits::default())
            .expect("long finite input is valid"),
        OperationOutcome::Complete {
            value: SoftDtwAnalysis::Finite { .. },
            ..
        }
    ));
}
