//! Property-based tests for Move-Split-Merge (MSM) time series distance metric.
//!
//! These tests verify:
//! - MSM metric properties (symmetry, identity, triangle inequality)
//! - Optimized and cutoff scorer consistency
//! - Proven lower-bound validity plus heuristic counterexamples
//! - Quantization/encoding correctness

#[path = "common/time_series_strategies.rs"]
mod time_series_strategies;

use liblevenshtein::time_series::{
    combined_lb, euclidean_lb, l1_lb, length_lb, MsmConfig, QuantizationConfig,
};
use proptest::prelude::*;
use time_series_strategies::short_time_series_strategy;

// ============================================================================
// MSM Configuration
// ============================================================================

/// Default C constant for tests
const TEST_C_CONST: f64 = 1.0;

fn test_config() -> MsmConfig {
    MsmConfig::new(TEST_C_CONST)
}

/// Independent full-matrix MSM oracle used only by the correspondence tests.
///
/// This deliberately shares neither the production two-row scorer nor its
/// cutoff/final-admission helpers.  Keeping the recurrence here makes a
/// deleted terminal cutoff check observable instead of comparing one
/// production function to itself.
fn full_matrix_msm_oracle(left: &[f64], right: &[f64], split_merge: f64) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    if left.is_empty() || right.is_empty() {
        return f64::INFINITY;
    }

    fn operation_cost(value: f64, left_context: f64, right_context: f64, base: f64) -> f64 {
        if (left_context <= value && value <= right_context)
            || (left_context >= value && value >= right_context)
        {
            base
        } else {
            base + (value - left_context)
                .abs()
                .min((value - right_context).abs())
        }
    }

    let rows = left.len();
    let columns = right.len();
    let mut matrix = vec![vec![f64::INFINITY; columns]; rows];
    matrix[0][0] = (left[0] - right[0]).abs();
    for column in 1..columns {
        matrix[0][column] = matrix[0][column - 1]
            + operation_cost(right[column], left[0], right[column - 1], split_merge);
    }
    for row in 1..rows {
        matrix[row][0] =
            matrix[row - 1][0] + operation_cost(left[row], left[row - 1], right[0], split_merge);
        for column in 1..columns {
            let moved = matrix[row - 1][column - 1] + (left[row] - right[column]).abs();
            let merged = matrix[row - 1][column]
                + operation_cost(left[row], left[row - 1], right[column], split_merge);
            let split = matrix[row][column - 1]
                + operation_cost(right[column], left[row], right[column - 1], split_merge);
            matrix[row][column] = moved.min(merged).min(split);
        }
    }
    matrix[rows - 1][columns - 1]
}

// ============================================================================
// MSM Metric Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// MSM Identity: d(x, x) == 0
    #[test]
    fn prop_msm_identity(
        series in short_time_series_strategy(),
    ) {
        let config = test_config();
        let distance = config.distance(&series, &series);
        prop_assert!(
            distance.abs() < 1e-9,
            "MSM identity violated: d(x, x) = {} != 0",
            distance
        );
    }

    /// The Rust-facing extended-cost semantics use zero for two empty series
    /// and positive infinity for exactly one empty series. Every finite cutoff
    /// must therefore reject the latter in both argument orders.
    #[test]
    fn prop_msm_one_empty_is_positive_infinity(
        nonempty in prop::collection::vec(-1_000.0f64..1_000.0, 1..=8),
        split_merge_cost in 0.0f64..10.0,
        finite_cutoff in 0.0f64..10_000.0,
    ) {
        let config = MsmConfig::new(split_merge_cost);

        prop_assert_eq!(config.distance(&[], &[]), 0.0);
        for (left, right) in [(&[][..], nonempty.as_slice()), (nonempty.as_slice(), &[][..])] {
            prop_assert!(config.distance(left, right).is_infinite());
            prop_assert!(config.distance_optimized(left, right).is_infinite());
            prop_assert!(config.distance_with_matrix(left, right).distance.is_infinite());
            prop_assert_eq!(config.distance_with_cutoff(left, right, finite_cutoff), None);
        }
    }

    /// MSM Symmetry: d(x, y) == d(y, x)
    #[test]
    fn prop_msm_symmetry(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        // Skip if either is empty (MSM requires non-empty series for meaningful comparison)
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let d_xy = config.distance(&x, &y);
        let d_yx = config.distance(&y, &x);

        prop_assert!(
            (d_xy - d_yx).abs() < 1e-9,
            "MSM symmetry violated: d(x, y) = {} != d(y, x) = {}",
            d_xy, d_yx
        );
    }

    /// MSM Triangle Inequality: d(x, z) <= d(x, y) + d(y, z)
    #[test]
    fn prop_msm_triangle_inequality(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
        z in short_time_series_strategy(),
    ) {
        // Skip if any is empty
        if x.is_empty() || y.is_empty() || z.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let d_xz = config.distance(&x, &z);
        let d_xy = config.distance(&x, &y);
        let d_yz = config.distance(&y, &z);

        prop_assert!(
            d_xz <= d_xy + d_yz + 1e-9,
            "MSM triangle inequality violated: d(x,z) = {} > d(x,y) + d(y,z) = {} + {} = {}",
            d_xz, d_xy, d_yz, d_xy + d_yz
        );
    }

    /// MSM Non-negativity: d(x, y) >= 0
    #[test]
    fn prop_msm_non_negative(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let distance = config.distance(&x, &y);

        prop_assert!(
            distance >= -1e-9,
            "MSM non-negativity violated: d(x, y) = {}",
            distance
        );
    }
}

// ============================================================================
// Optimized and Cutoff Scorer Consistency Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Baseline and two-row implementations produce the same exact score.
    #[test]
    fn prop_two_row_scorer_matches_independent_full_matrix_oracle(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let oracle = full_matrix_msm_oracle(&x, &y, TEST_C_CONST);
        prop_assert!((oracle - config.distance_optimized(&x, &y)).abs() < 1e-9);
    }

    /// Wavefront with threshold should return same as without when distance is below threshold
    #[test]
    fn prop_cutoff_accepts_scores_above_the_exact_distance(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();

        // First compute without threshold
        let full_result = config.distance_with_cutoff(&x, &y, f64::INFINITY);

        if let Some(full_dist) = full_result {
            // Compute with a threshold above the actual distance
            let threshold = full_dist + 10.0;
            let threshold_result = config.distance_with_cutoff(&x, &y, threshold);

            prop_assert!(
                threshold_result.is_some(),
                "Threshold result should not be None when threshold > actual distance"
            );

            if let Some(thresh_dist) = threshold_result {
                prop_assert!(
                    (full_dist - thresh_dist).abs() < 1e-9,
                    "Distance should be same with threshold above actual"
                );
            }
        }
    }

    /// Every finite cutoff agrees with the exact scalar DP scorer in both
    /// membership and returned distance. This covers thresholds below,
    /// equal to, and above the exact distance rather than testing only the
    /// permissive side of the contract.
    #[test]
    fn prop_cutoff_matches_independent_full_matrix_oracle(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
        split_merge_cost in 0.0f64..10.0,
        cutoff in 0.0f64..100.0,
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = MsmConfig::new(split_merge_cost);
        let exact = full_matrix_msm_oracle(&x, &y, config.split_merge_cost());
        let expected = (exact <= cutoff).then_some(exact);
        let actual = config.distance_with_cutoff(&x, &y, cutoff);

        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                prop_assert!((actual - expected).abs() < 1e-9);
            }
            (None, None) => {}
            (actual, expected) => {
                prop_assert!(false, "cutoff disagreement: scorer={actual:?}, oracle={expected:?}, cutoff={cutoff}");
            }
        }
    }
}

#[test]
fn cutoff_scorer_rejects_finite_distance_above_cutoff() {
    let config = MsmConfig::new(100.0);
    let query = [0.0, 10.0];
    let target = [0.0, 20.0];

    assert_eq!(config.distance(&query, &target), 10.0);
    assert_eq!(config.distance_with_cutoff(&query, &target, 1.0), None);
}

#[test]
fn cutoff_final_guard_omission_source_mutant_is_killed() {
    let config = MsmConfig::new(0.0);

    // The first cell keeps the only initialized row below the cutoff, while
    // the terminal cell lies above it.  Early row-minimum abandonment cannot
    // mask a deleted terminal admission guard for this shape.
    let query = [0.0, 0.0];
    let target = [0.0, 10.0];
    assert_eq!(config.distance(&query, &target), 10.0);
    assert_eq!(config.distance_with_cutoff(&query, &target, 1.0), None);
}

#[test]
fn cutoff_scorer_rejects_the_immediately_adjacent_binary64_cost() {
    let config = MsmConfig::new(100.0);
    let exact = 1.0_f64.next_up();

    assert_eq!(config.distance(&[0.0], &[exact]), exact);
    assert_eq!(config.distance_with_cutoff(&[0.0], &[exact], 1.0), None);
    assert_eq!(
        config.distance_with_cutoff(&[0.0], &[exact], exact),
        Some(exact)
    );
}

#[test]
fn cutoff_scorer_preserves_the_empty_empty_boundary() {
    let config = test_config();
    assert_eq!(config.distance_with_cutoff(&[], &[], 0.0), Some(0.0));
}

// ============================================================================
// Lower Bound Validity Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Length-based lower bound: LB_length(x, y) <= MSM(x, y)
    #[test]
    fn prop_length_lb_valid(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let msm_dist = config.distance(&x, &y);
        let lb = length_lb(&x, &y, TEST_C_CONST);

        prop_assert!(
            lb <= msm_dist + 1e-9,
            "Length LB invalid: {} > MSM {}",
            lb, msm_dist
        );
    }

    /// Bound and heuristic scores should be non-negative
    #[test]
    fn prop_lb_non_negative(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        prop_assert!(euclidean_lb(&x, &y) >= 0.0, "Euclidean LB negative");
        prop_assert!(length_lb(&x, &y, TEST_C_CONST) >= 0.0, "Length LB negative");
        prop_assert!(combined_lb(&x, &y, TEST_C_CONST) >= 0.0, "Combined LB negative");
    }
}

#[test]
fn euclidean_l1_and_combined_are_heuristics_not_general_lower_bounds() {
    let config = test_config();
    let x = vec![0.0, 100.0];
    let y = vec![0.0, 0.0, 100.0];
    let msm_dist = config.distance(&x, &y);

    assert!((msm_dist - 1.0).abs() < 1e-9);
    assert!(euclidean_lb(&x, &y) > msm_dist);
    assert!(l1_lb(&x, &y) > msm_dist);
    assert!(combined_lb(&x, &y, TEST_C_CONST) > msm_dist);
}

// ============================================================================
// Quantization Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Quantization round-trip error should be bounded
    #[test]
    fn prop_quantization_bounded_error(
        values in prop::collection::vec(-100.0f64..100.0f64, 1..20),
    ) {
        let config = QuantizationConfig::for_u8(-100.0, 100.0);

        for &value in &values {
            let bin = config.quantize(value);
            let recovered = config.dequantize(bin);

            // Error should be at most half a bin width
            let max_error = config.max_error();
            let error = (value - recovered).abs();

            prop_assert!(
                error <= max_error + 1e-9,
                "Quantization error {} > max {} for value {}",
                error, max_error, value
            );
        }
    }

    /// Quantization should be monotonic
    #[test]
    fn prop_quantization_monotonic(
        v1 in -100.0f64..100.0f64,
        v2 in -100.0f64..100.0f64,
    ) {
        let config = QuantizationConfig::for_u8(-100.0, 100.0);

        let bin1 = config.quantize(v1);
        let bin2 = config.quantize(v2);

        if v1 < v2 {
            prop_assert!(bin1 <= bin2, "Quantization not monotonic: {} < {} but bin {} > bin {}", v1, v2, bin1, bin2);
        }
    }

    /// Encoded/decoded series should be close to original
    #[test]
    fn prop_encode_decode_roundtrip(
        values in prop::collection::vec(-50.0f64..50.0f64, 1..10),
    ) {
        let config = QuantizationConfig::for_u8(-100.0, 100.0);

        let encoded = config.encode_u8(&values);
        let decoded = config.decode_u8(&encoded);

        prop_assert_eq!(values.len(), decoded.len(), "Length mismatch");

        let max_error = config.max_error();
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            let error = (original - recovered).abs();
            prop_assert!(
                error <= max_error + 1e-9,
                "Roundtrip error {} > max {} for value {}",
                error, max_error, original
            );
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_msm_identical_series() {
    let config = test_config();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let distance = config.distance(&x, &x);
    assert!(
        distance.abs() < 1e-9,
        "Identical series should have distance 0"
    );
}

#[test]
fn test_msm_single_element() {
    let config = test_config();
    let x = vec![1.0];
    let y = vec![2.0];
    let distance = config.distance(&x, &y);
    assert!(
        (distance - 1.0).abs() < 1e-9,
        "Distance should be |1-2| = 1"
    );
}

#[test]
fn test_msm_length_difference() {
    let config = test_config();
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];

    // Distance should account for length difference
    let distance = config.distance(&x, &y);
    assert!(
        distance > 0.0,
        "Different length series should have non-zero distance"
    );
}

#[test]
fn test_msm_constant_series() {
    let config = test_config();
    let x = vec![5.0, 5.0, 5.0, 5.0];
    let y = vec![5.0, 5.0, 5.0, 5.0];
    let distance = config.distance(&x, &y);
    assert!(
        distance.abs() < 1e-9,
        "Identical constant series should have distance 0"
    );
}

#[test]
fn test_msm_shifted_series() {
    let config = test_config();
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 3.0, 4.0]; // Shifted up by 1

    let distance = config.distance(&x, &y);
    // Each element differs by 1, so with 3 elements the distance depends on optimal alignment
    assert!(
        distance > 0.0,
        "Shifted series should have non-zero distance"
    );
    assert!(distance < 10.0, "Distance should be reasonable");
}

#[test]
fn test_scalar_cutoff_threshold_pruning() {
    let config = test_config();
    let x = vec![0.0, 0.0, 0.0];
    let y = vec![100.0, 100.0, 100.0];

    // With a very low threshold, should return None
    let result = config.distance_with_cutoff(&x, &y, 1.0);
    assert!(result.is_none(), "Should be pruned with low threshold");

    // With a high threshold, should return Some
    let result = config.distance_with_cutoff(&x, &y, 1000.0);
    assert!(result.is_some(), "Should not be pruned with high threshold");
}

// ============================================================================
// Lower Bound Specificity Tests
// ============================================================================

#[test]
fn test_lb_identical_series() {
    let x = vec![1.0, 2.0, 3.0];

    assert!(
        euclidean_lb(&x, &x).abs() < 1e-9,
        "Euclidean LB for identical should be 0"
    );
    assert!(
        length_lb(&x, &x, 1.0).abs() < 1e-9,
        "Length LB for identical should be 0"
    );
    assert!(
        combined_lb(&x, &x, 1.0).abs() < 1e-9,
        "Combined LB for identical should be 0"
    );
}

#[test]
fn test_lb_length_difference() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0];

    let lb = length_lb(&x, &y, TEST_C_CONST);
    // Length difference is 3, so LB should be at least 3 * c_const
    assert!(
        lb >= 3.0 * TEST_C_CONST - 1e-9,
        "Length LB should account for length difference"
    );
}

// ============================================================================
// C() Function Tests
// ============================================================================

#[test]
fn test_c_function_between_values() {
    // Test C(a, b, c) when b <= a <= c
    let config = MsmConfig::new(1.0);

    // When a is between b and c, cost should be just c_const
    // C(2, 1, 3) = 1.0 because 1 <= 2 <= 3
    let cost = config.c_func(2.0, 1.0, 3.0);
    assert!(
        (cost - 1.0).abs() < 1e-9,
        "C(2,1,3) should be c_const=1.0, got {}",
        cost
    );
}

#[test]
fn test_c_function_outside_values() {
    // Test C(a, b, c) when a is outside [b, c]
    let config = MsmConfig::new(1.0);

    // When a is outside, cost should be c_const + min(|a-b|, |a-c|)
    // C(0, 2, 4) = 1.0 + min(|0-2|, |0-4|) = 1.0 + 2.0 = 3.0
    let cost = config.c_func(0.0, 2.0, 4.0);
    assert!(
        (cost - 3.0).abs() < 1e-9,
        "C(0,2,4) should be 3.0, got {}",
        cost
    );
}

// ============================================================================
// Quantization Configuration Tests
// ============================================================================

#[test]
fn test_quantization_config_u8() {
    let config = QuantizationConfig::for_u8(0.0, 100.0);

    // Should have 256 bins (0-255)
    assert!(config.quantize(0.0) == 0 || config.quantize(0.0) == 1);
    assert!(config.quantize(100.0) == 255 || config.quantize(100.0) == 254);

    // Should be monotonic
    assert!(config.quantize(25.0) < config.quantize(75.0));
}

#[test]
fn test_quantization_config_u16() {
    let config = QuantizationConfig::for_u16(0.0, 100.0);

    // Should have higher precision than u8
    let bin_a = config.quantize(50.0);
    let bin_b = config.quantize(50.1);

    // With 65536 bins over 100 units, 0.1 difference should span multiple bins
    // unless bins are very wide
    assert!(bin_b >= bin_a);
}

#[test]
fn test_quantization_from_data() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let config = QuantizationConfig::from_data(&data, 10, 0.1);

    assert!(config.is_some(), "Should create config from data");

    let config = config.unwrap();
    // All data points should be quantizable
    for &v in &data {
        let _ = config.quantize(v);
    }
}
