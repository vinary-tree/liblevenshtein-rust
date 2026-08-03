//! Interval primitives shared by elastic-kernel relaxations.
//!
//! Quantization represents a target sample by a closed interval. These helpers
//! compute exact minima over one-dimensional interval boxes and therefore form
//! the geometric base of each kernel's K1 proof.

/// Distance from scalar `value` to closed interval `[low, high]`.
///
/// This equals `$`\min_{x\in[\ell,h]} |v-x|`$` and remains well-defined for
/// semi-infinite quantization boundary bins.
#[inline]
pub fn interval_dist(value: f64, low: f64, high: f64) -> f64 {
    (low - value).max(0.0).max(value - high)
}

/// Minimum absolute distance between two closed intervals.
///
/// For `$`A=[\ell_A,h_A]`$` and `$`B=[\ell_B,h_B]`$`, this computes
/// `$`\min_{a\in A,b\in B}|a-b| = \max(0,\ell_A-h_B,\ell_B-h_A)`$`.
#[inline]
pub fn interval_gap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.1).max(0.0).max(b.0 - a.1)
}

/// Whether two closed intervals intersect.
#[inline]
pub fn intervals_intersect(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Smallest closed interval containing both inputs.
#[inline]
pub fn interval_envelope(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0.min(b.0), a.1.max(b.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn separated_touching_and_overlapping_intervals() {
        assert_eq!(interval_gap((0.0, 1.0), (3.0, 5.0)), 2.0);
        assert_eq!(interval_gap((0.0, 1.0), (1.0, 2.0)), 0.0);
        assert_eq!(interval_gap((-2.0, 4.0), (1.0, 3.0)), 0.0);
        assert!(intervals_intersect((0.0, 1.0), (1.0, 2.0)));
        assert_eq!(interval_envelope((0.0, 1.0), (-2.0, 3.0)), (-2.0, 3.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn interval_gap_is_symmetric_nonnegative_and_exact_on_degenerate_bins(
            a0 in -1.0e3f64..1.0e3,
            aw in 0.0f64..1.0e3,
            b0 in -1.0e3f64..1.0e3,
            bw in 0.0f64..1.0e3,
        ) {
            let a = (a0, a0 + aw);
            let b = (b0, b0 + bw);
            let gap = interval_gap(a, b);
            prop_assert!(gap >= 0.0);
            prop_assert_eq!(gap, interval_gap(b, a));
            prop_assert_eq!(gap == 0.0, intervals_intersect(a, b));

            let point_a = (a0, a0);
            let point_b = (b0, b0);
            prop_assert_eq!(interval_gap(point_a, point_b), (a0 - b0).abs());
            prop_assert_eq!(interval_dist(a0, b0, b0), (a0 - b0).abs());
        }
    }
}
