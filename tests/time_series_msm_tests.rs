//! Property-based tests for Move-Split-Merge (MSM) time series distance metric.
//!
//! These tests verify:
//! - MSM metric properties (symmetry, identity, triangle inequality)
//! - Wavefront vs automaton implementation consistency
//! - Proven lower-bound validity plus heuristic counterexamples
//! - Quantization/encoding correctness

#[path = "common/time_series_strategies.rs"]
mod time_series_strategies;

use liblevenshtein::time_series::{
    combined_lb, euclidean_lb, l1_lb, length_lb, msm_distance_automaton, msm_distance_wavefront,
    MsmConfig, QuantizationConfig,
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
// Wavefront vs Automaton Consistency Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Wavefront and automaton implementations should produce same results
    #[test]
    fn prop_wavefront_automaton_consistency(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();
        let max_cost = f64::INFINITY;

        let wavefront_result = msm_distance_wavefront(&x, &y, &config, max_cost);
        let automaton_result = msm_distance_automaton(&x, &y, &config, max_cost);

        match (wavefront_result, automaton_result) {
            (Some(w), Some(a)) => {
                prop_assert!(
                    (w - a).abs() < 1e-6,
                    "Wavefront ({}) and automaton ({}) disagree",
                    w, a
                );
            }
            (None, None) => {
                // Both returned None, consistent
            }
            (w, a) => {
                prop_assert!(
                    false,
                    "Inconsistent None results: wavefront={:?}, automaton={:?}",
                    w, a
                );
            }
        }
    }

    /// Wavefront with threshold should return same as without when distance is below threshold
    #[test]
    fn prop_wavefront_threshold_consistency(
        x in short_time_series_strategy(),
        y in short_time_series_strategy(),
    ) {
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }

        let config = test_config();

        // First compute without threshold
        let full_result = msm_distance_wavefront(&x, &y, &config, f64::INFINITY);

        if let Some(full_dist) = full_result {
            // Compute with a threshold above the actual distance
            let threshold = full_dist + 10.0;
            let threshold_result = msm_distance_wavefront(&x, &y, &config, threshold);

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
fn test_wavefront_threshold_pruning() {
    let config = test_config();
    let x = vec![0.0, 0.0, 0.0];
    let y = vec![100.0, 100.0, 100.0];

    // With a very low threshold, should return None
    let result = msm_distance_wavefront(&x, &y, &config, 1.0);
    assert!(result.is_none(), "Should be pruned with low threshold");

    // With a high threshold, should return Some
    let result = msm_distance_wavefront(&x, &y, &config, 1000.0);
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
