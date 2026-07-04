//! Shared checked/saturating numeric conversions.
//!
//! This crate-internal module is the single home for the *pure* numeric
//! conversion helpers that were previously copy-pasted across the `transducer`,
//! `time_series`, and `cache` modules. Every helper here is total over its input
//! domain (no panics in release builds) with an explicitly documented
//! saturation, rounding, and `NaN` contract.
//!
//! Scope: only pure conversions live here — `f64 → usize` (floor/ceil),
//! `usize → i64` differences, and `Duration → u64` milliseconds. Domain-specific
//! arithmetic (position/offset/state-id checks, window sizing, quantization)
//! stays in its owning module.
//!
//! # `f64 → usize` conversion strategy
//!
//! A finite non-negative `f64` is decomposed directly from its IEEE-754
//! binary64 bit pattern (sign/exponent/mantissa) rather than via the lossy
//! `value as usize` cast. This yields an exact `⌊value⌋` (and, from the dropped
//! fraction bits, `⌈value⌉`) as a wide `u128`, which is then narrowed to `usize`
//! with `usize::try_from` so that oversized magnitudes saturate deterministically
//! instead of wrapping or triggering UB.

// IEEE-754 binary64 field widths, used to floor/ceil a finite non-negative f64
// into an exact integer without going through the lossy `as` cast.
const F64_EXPONENT_MASK: u64 = 0x7ff;
const F64_MANTISSA_BITS: i32 = 52;
const F64_MANTISSA_MASK: u64 = (1u64 << F64_MANTISSA_BITS) - 1;
const F64_HIDDEN_BIT: u64 = 1u64 << F64_MANTISSA_BITS;
const F64_EXPONENT_BIAS: i32 = 1023;

/// Floor a non-negative `f64` to `usize`, returning `None` for inputs outside
/// the representable non-negative domain.
///
/// Contract:
/// - `NaN`, or any strictly-negative value (including `-∞`) → `None`.
/// - `+∞`, or any finite value whose floor exceeds `usize::MAX` → `Some(usize::MAX)`.
/// - Otherwise → `Some(⌊value⌋)`. `-0.0` is treated as `0.0` → `Some(0)`.
#[inline(always)]
pub(crate) fn nonnegative_floor_to_usize_checked(value: f64) -> Option<usize> {
    if value.is_nan() || value < 0.0 {
        None
    } else if !value.is_finite() {
        Some(usize::MAX)
    } else {
        finite_nonnegative_floor_to_usize(value).or(Some(usize::MAX))
    }
}

/// Floor a non-negative `f64` to `usize`, saturating out-of-domain inputs to `0`
/// or `usize::MAX` instead of signalling with `None`.
///
/// Contract:
/// - `NaN`, `-∞`, or any value `≤ 0.0` (including `-0.0`) → `0`.
/// - `+∞`, or any finite value whose floor exceeds `usize::MAX` → `usize::MAX`.
/// - Otherwise → `⌊value⌋`.
#[inline(always)]
pub(crate) fn nonnegative_floor_to_usize_saturating(value: f64) -> usize {
    if value.is_nan() || value <= 0.0 {
        0
    } else if !value.is_finite() {
        usize::MAX
    } else {
        finite_nonnegative_floor_to_usize(value).unwrap_or(usize::MAX)
    }
}

/// Ceil a non-negative `f64` to `usize`, saturating out-of-domain inputs.
///
/// Contract:
/// - `NaN`, `-∞`, or any value `≤ 0.0` (including `-0.0`) → `0`.
/// - `+∞`, or any finite value whose ceiling exceeds `usize::MAX` → `usize::MAX`.
/// - Otherwise → `⌈value⌉`.
#[inline(always)]
pub(crate) fn nonnegative_ceil_to_usize(value: f64) -> usize {
    if value.is_nan() || value <= 0.0 {
        0
    } else if !value.is_finite() {
        usize::MAX
    } else {
        finite_nonnegative_ceil_to_usize(value).unwrap_or(usize::MAX)
    }
}

/// Compute `left - right` as an `i64`, saturating on overflow in either direction.
///
/// Contract:
/// - `left ≥ right` → `+(left - right)`, saturating to `i64::MAX` when the gap
///   exceeds `i64::MAX`.
/// - `left < right` → `-(right - left)`, saturating to `i64::MIN` when the gap
///   exceeds `i64::MAX`.
#[inline]
pub(crate) fn saturating_usize_difference_i64(left: usize, right: usize) -> i64 {
    if left >= right {
        i64::try_from(left - right).unwrap_or(i64::MAX)
    } else {
        match i64::try_from(right - left) {
            Ok(diff) => -diff,
            Err(_) => i64::MIN,
        }
    }
}

/// Convert a [`Duration`](std::time::Duration) to whole milliseconds as `u64`,
/// saturating to `u64::MAX` when the millisecond count exceeds `u64::MAX`.
///
/// Gated to the same feature set as its sole consumer (the `cache` module), so
/// this shared copy is compiled exactly when it is used.
#[cfg(any(feature = "pathmap-backend", feature = "phonetic-rules"))]
#[inline]
pub(crate) fn duration_millis_u64_saturating(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Floor a finite, non-negative `f64` to `usize`, or `None` if the exact floor
/// overflows `usize`.
#[inline(always)]
fn finite_nonnegative_floor_to_usize(value: f64) -> Option<usize> {
    let (integer, _) = finite_nonnegative_floor_parts(value)?;
    usize::try_from(integer).ok()
}

/// Ceil a finite, non-negative `f64` to `usize`, or `None` if the exact ceiling
/// overflows `usize`.
#[inline(always)]
fn finite_nonnegative_ceil_to_usize(value: f64) -> Option<usize> {
    let (integer, has_fraction) = finite_nonnegative_floor_parts(value)?;
    let ceiling = if has_fraction {
        integer.checked_add(1)?
    } else {
        integer
    };
    usize::try_from(ceiling).ok()
}

/// Decompose a finite, non-negative `f64` into its exact integer floor (as a
/// wide `u128`) and a flag indicating whether any fractional bits were dropped.
///
/// Returns `None` only if the exact floor overflows `u128` (unreachable for the
/// binary64 domain, but handled defensively). The caller (`_ceil_to_usize`) uses
/// the fraction flag to round up.
#[inline(always)]
fn finite_nonnegative_floor_parts(value: f64) -> Option<(u128, bool)> {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);

    let bits = value.to_bits();
    let exponent_bits = u16::try_from((bits >> F64_MANTISSA_BITS) & F64_EXPONENT_MASK).ok()?;
    let mantissa = bits & F64_MANTISSA_MASK;

    if exponent_bits == 0 {
        return Some((0, mantissa != 0));
    }

    let exponent = i32::from(exponent_bits) - F64_EXPONENT_BIAS;
    if exponent < 0 {
        return Some((0, true));
    }

    let significand = F64_HIDDEN_BIT | mantissa;
    if exponent >= F64_MANTISSA_BITS {
        let shift = u32::try_from(exponent - F64_MANTISSA_BITS).ok()?;
        let integer = u128::from(significand).checked_shl(shift)?;
        return Some((integer, false));
    }

    let shift = u32::try_from(F64_MANTISSA_BITS - exponent).ok()?;
    let integer = u128::from(significand >> shift);
    let fraction_mask = (1u64 << shift) - 1;
    Some((integer, (significand & fraction_mask) != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2^52 — the largest f64 magnitude whose ULP is still 1, so its floor and
    // ceiling are exact and equal.
    const EXACT_LARGE: f64 = 4_503_599_627_370_496.0;

    fn expected_large() -> usize {
        usize::try_from(4_503_599_627_370_496_u128).unwrap_or(usize::MAX)
    }

    #[test]
    fn nonnegative_floor_to_usize_checked_contract() {
        assert_eq!(nonnegative_floor_to_usize_checked(f64::NAN), None);
        assert_eq!(nonnegative_floor_to_usize_checked(f64::NEG_INFINITY), None);
        assert_eq!(nonnegative_floor_to_usize_checked(-1.0), None);
        assert_eq!(nonnegative_floor_to_usize_checked(-0.0), Some(0));
        assert_eq!(nonnegative_floor_to_usize_checked(0.0), Some(0));
        assert_eq!(nonnegative_floor_to_usize_checked(0.9), Some(0));
        assert_eq!(nonnegative_floor_to_usize_checked(2.0), Some(2));
        assert_eq!(nonnegative_floor_to_usize_checked(2.9), Some(2));
        assert_eq!(
            nonnegative_floor_to_usize_checked(EXACT_LARGE),
            Some(expected_large())
        );
        assert_eq!(
            nonnegative_floor_to_usize_checked(f64::INFINITY),
            Some(usize::MAX)
        );
        assert_eq!(
            nonnegative_floor_to_usize_checked(f64::MAX),
            Some(usize::MAX)
        );
    }

    #[test]
    fn nonnegative_floor_to_usize_saturating_contract() {
        assert_eq!(nonnegative_floor_to_usize_saturating(f64::NAN), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(f64::NEG_INFINITY), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(-1.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(-0.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(0.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(0.1), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(2.0), 2);
        assert_eq!(nonnegative_floor_to_usize_saturating(2.9), 2);
        assert_eq!(
            nonnegative_floor_to_usize_saturating(EXACT_LARGE),
            expected_large()
        );
        assert_eq!(
            nonnegative_floor_to_usize_saturating(f64::INFINITY),
            usize::MAX
        );
        assert_eq!(nonnegative_floor_to_usize_saturating(f64::MAX), usize::MAX);
    }

    #[test]
    fn nonnegative_ceil_to_usize_contract() {
        assert_eq!(nonnegative_ceil_to_usize(f64::NAN), 0);
        assert_eq!(nonnegative_ceil_to_usize(f64::NEG_INFINITY), 0);
        assert_eq!(nonnegative_ceil_to_usize(-1.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(-0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.1), 1);
        assert_eq!(nonnegative_ceil_to_usize(2.0), 2);
        assert_eq!(nonnegative_ceil_to_usize(2.1), 3);
        assert_eq!(nonnegative_ceil_to_usize(EXACT_LARGE), expected_large());
        assert_eq!(nonnegative_ceil_to_usize(f64::INFINITY), usize::MAX);
        assert_eq!(nonnegative_ceil_to_usize(f64::MAX), usize::MAX);
    }

    #[test]
    fn floor_and_ceil_agree_on_integers_and_differ_on_fractions() {
        assert_eq!(nonnegative_floor_to_usize_saturating(7.0), 7);
        assert_eq!(nonnegative_ceil_to_usize(7.0), 7);
        assert_eq!(nonnegative_floor_to_usize_saturating(7.001), 7);
        assert_eq!(nonnegative_ceil_to_usize(7.001), 8);
    }

    #[test]
    fn saturating_usize_difference_i64_contract() {
        assert_eq!(saturating_usize_difference_i64(0, 0), 0);
        assert_eq!(saturating_usize_difference_i64(5, 3), 2);
        assert_eq!(saturating_usize_difference_i64(3, 5), -2);

        let max_i64_usize = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
        assert_eq!(saturating_usize_difference_i64(max_i64_usize, 0), i64::MAX);
        assert_eq!(saturating_usize_difference_i64(usize::MAX, 0), i64::MAX);
        assert_eq!(
            saturating_usize_difference_i64(0, usize::MAX),
            if usize::MAX == max_i64_usize {
                -i64::MAX
            } else {
                i64::MIN
            }
        );
        if let Some(one_past_i64_max) = max_i64_usize.checked_add(1) {
            assert_eq!(
                saturating_usize_difference_i64(0, one_past_i64_max),
                i64::MIN
            );
        }
    }

    #[cfg(any(feature = "pathmap-backend", feature = "phonetic-rules"))]
    #[test]
    fn duration_millis_u64_saturating_contract() {
        use std::time::Duration;
        assert_eq!(duration_millis_u64_saturating(Duration::ZERO), 0);
        assert_eq!(
            duration_millis_u64_saturating(Duration::from_millis(1234)),
            1234
        );
        assert_eq!(
            duration_millis_u64_saturating(Duration::from_secs(u64::MAX)),
            u64::MAX
        );
    }
}
