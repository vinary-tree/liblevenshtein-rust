//! Subsumption Relation for Universal Positions
//!
//! Implements the subsumption relation ≤^χ_s from Definition 11 (thesis pages 18-21).
//!
//! # Theory Background
//!
//! The subsumption relation is a partial order on positions that enables state minimization.
//! Position π₁ subsumes π₂ (written π₁ <^χ_s π₂) if π₂ represents a "better" state
//! (more errors available, close enough in position).
//!
//! ## Definition 11: Subsumption for Standard Levenshtein (χ = ε)
//!
//! ```text
//! i#e ≤^ε_s j#f  ⇔  f > e ∧ |j - i| ≤ f - e
//! ```
//!
//! **Intuition**: Position j#f subsumes i#e if:
//! 1. f > e (more errors available at j)
//! 2. |j - i| ≤ f - e (positions close enough given error difference)
//!
//! ## Subsumption for Transposition (χ = t)
//!
//! ```text
//! i#e   ≤^t_s j#f     ⇔  f > e ∧ |j - i| ≤ f - e
//! i#e_t ≤^t_s j#f     ⇔  f > e ∧ |j + 1 - i| ≤ f - e
//! i#e   ≤^t_s j#f_t   ⇔  false  (different types)
//! i#e_t ≤^t_s j#f_t   ⇔  false  (different types)
//! ```
//!
//! ## Subsumption for Merge/Split (χ = ms)
//!
//! ```text
//! i#e   ≤^ms_s j#f    ⇔  f > e ∧ |j - i| ≤ f - e
//! i#e_s ≤^ms_s j#f    ⇔  f > e ∧ |j - i| ≤ f - e
//! i#e   ≤^ms_s j#f_s  ⇔  false  (different types)
//! i#e_s ≤^ms_s j#f_s  ⇔  false  (different types)
//! ```
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::transducer::universal::{UniversalPosition, Standard, subsumes};
//!
//! let pos1 = UniversalPosition::<Standard>::new_i(3, 3, 3)
//!     .expect("3#3 satisfies the I-position invariant");
//! let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 3)
//!     .expect("2#2 satisfies the I-position invariant");
//!
//! // The second position has fewer errors, so strict subsumption fails.
//! assert!(!subsumes(&pos1, &pos2, 3));
//!
//! let pos3 = UniversalPosition::<Standard>::new_i(1, 1, 3)
//!     .expect("1#1 satisfies the I-position invariant");
//! let pos4 = UniversalPosition::<Standard>::new_i(2, 2, 3)
//!     .expect("2#2 satisfies the I-position invariant");
//!
//! // Check: 1#1 ≤^ε_s 2#2
//! // f > e: 2 > 1 ✓
//! // |j - i| ≤ f - e: |2 - 1| = 1 ≤ 2 - 1 = 1 ✓
//! assert!(subsumes(&pos3, &pos4, 3));
//! ```

use crate::transducer::universal::position::{PositionVariant, UniversalPosition};

#[cfg(test)]
use crate::transducer::universal::position::{
    MergeAndSplit, MergeSplitState, Standard, Transposition, TranspositionState,
};

/// Check if pos1 strictly subsumes pos2 (pos1 <^χ_s pos2)
///
/// # Arguments
///
/// - `pos1`: First position (potential subsumer)
/// - `pos2`: Second position (potentially subsumed)
/// - `max_distance`: Maximum edit distance n
///
/// # Returns
///
/// `true` if pos1 <^χ_s pos2 (pos2 is subsumed by pos1), `false` otherwise
///
/// # Theory
///
/// From Definition 11 (page 18):
/// - Both positions must have same parameter type (I or M)
/// - pos2 must have more errors available (f > e)
/// - Positions must be close enough (|j - i| ≤ f - e)
/// - For transposition/merge-split: special rules for type subscripts
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::universal::{subsumes, Standard, UniversalPosition};
///
/// let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 3)
///     .expect("1#1 satisfies the I-position invariant");
/// let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 3)
///     .expect("2#2 satisfies the I-position invariant");
/// assert!(subsumes(&pos1, &pos2, 3)); // 1#1 <^ε_s 2#2
/// ```
#[inline(always)]
pub fn subsumes<V: PositionVariant>(
    pos1: &UniversalPosition<V>,
    pos2: &UniversalPosition<V>,
    max_distance: u8,
) -> bool {
    // Dispatch to variant-specific implementation
    subsumes_impl(pos1, pos2, max_distance)
}

/// Generic subsumption implementation (works for Standard variant)
///
/// For Standard variant, subsumption is straightforward:
/// i#e ≤^ε_s j#f ⇔ f > e ∧ |j - i| ≤ f - e
#[inline(always)]
fn subsumes_impl<V: PositionVariant>(
    pos1: &UniversalPosition<V>,
    pos2: &UniversalPosition<V>,
    _max_distance: u8,
) -> bool {
    use UniversalPosition::*;

    match (pos1, pos2) {
        // I-type subsumption: both must be I-type
        (
            INonFinal {
                offset: i,
                errors: e,
                variant_state: v1,
            },
            INonFinal {
                offset: j,
                errors: f,
                variant_state: v2,
            },
        ) => {
            // Definition 11: f > e ∧ |j - i| ≤ f - e
            // Early exit on error check
            if *f <= *e {
                return false;
            }

            let error_diff = u32::from(f - e);
            V::subsumption_distance(v1, v2, *i, *j).is_some_and(|distance| distance <= error_diff)
        }

        // M-type subsumption: both must be M-type
        (
            MFinal {
                offset: i,
                errors: e,
                variant_state: v1,
            },
            MFinal {
                offset: j,
                errors: f,
                variant_state: v2,
            },
        ) => {
            // Same formula as I-type: f > e ∧ |j - i| ≤ f - e
            if *f <= *e {
                return false;
            }

            let error_diff = u32::from(f - e);
            V::subsumption_distance(v1, v2, *i, *j).is_some_and(|distance| distance <= error_diff)
        }

        // Different parameter types: no subsumption
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsumption_basic_i_type() {
        // Test case: 1#1 ≤^ε_s 2#2
        // Invariant check: |1| = 1 ≤ 1 ✓, |2| = 2 ≤ 2 ✓
        let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        // Check: f > e (2 > 1) ✓
        //        |j - i| ≤ f - e (|2 - 1| = 1 ≤ 2 - 1 = 1) ✓
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_fails_distance_too_large() {
        // Test: 0#1 does NOT subsume 2#2
        // Invariant check: |0| = 0 ≤ 1 ✓, |2| = 2 ≤ 2 ✓
        let pos1 = UniversalPosition::<Standard>::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        // Check: f > e (2 > 1) ✓
        //        |j - i| ≤ f - e (|2 - 0| = 2 ≤ 2 - 1 = 1) ✗
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_fails_equal_errors() {
        // Positions with equal errors: no subsumption
        // Invariant check: |1| = 1 ≤ 1 ✓, |1| = 1 ≤ 1 ✓
        let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        // f = e, so f > e fails
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_m_type() {
        // Test M-type positions
        let pos1 = UniversalPosition::<Standard>::new_m(-2, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        // Check: f > e (1 > 0) ✓
        //        |j - i| ≤ f - e (|-1 - (-2)| = 1 ≤ 1 - 0 = 1) ✓
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_no_subsumption_across_types() {
        // I-type and M-type: no subsumption
        let i_pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        assert!(!subsumes(&i_pos, &m_pos, 2));
        assert!(!subsumes(&m_pos, &i_pos, 2));
    }

    // =========================================================================
    // Additional I-type Subsumption Tests
    // =========================================================================

    #[test]
    fn test_subsumption_i_exact_boundary() {
        // Test: 0#1 ≤^ε_s 1#2
        // Check: f > e (2 > 1) ✓
        //        |j - i| ≤ f - e (|1 - 0| = 1 ≤ 2 - 1 = 1) ✓ (exact boundary)
        let pos1 = UniversalPosition::<Standard>::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(1, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_i_negative_offsets() {
        // Test: -1#1 ≤^ε_s -2#2
        // Check: f > e (2 > 1) ✓
        //        |j - i| ≤ f - e (|-2 - (-1)| = |-1| = 1 ≤ 2 - 1 = 1) ✓
        let pos1 = UniversalPosition::<Standard>::new_i(-1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_i_zero_offset() {
        // Test: 0#0 ≤^ε_s 0#1
        // Check: f > e (1 > 0) ✓
        //        |j - i| ≤ f - e (|0 - 0| = 0 ≤ 1 - 0 = 1) ✓
        let pos1 = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(0, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_subsumption_i_large_error_diff() {
        // Test: 0#1 ≤^ε_s 2#3
        // Check: f > e (3 > 1) ✓
        //        |j - i| ≤ f - e (|2 - 0| = 2 ≤ 3 - 1 = 2) ✓ (exact boundary)
        let pos1 = UniversalPosition::<Standard>::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(2, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_no_subsumption_i_reverse_order() {
        // Test: 2#2 does NOT subsume 1#1 (reversed order from basic test)
        // Check: f > e (1 > 2) ✗
        let pos1 = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    // =========================================================================
    // Additional M-type Subsumption Tests
    // =========================================================================

    #[test]
    fn test_subsumption_m_exact_boundary() {
        // Test: M + (-2)#0 ≤^ε_s M + (-1)#1
        // Check: f > e (1 > 0) ✓
        //        |j - i| ≤ f - e (|-1 - (-2)| = 1 ≤ 1 - 0 = 1) ✓
        let pos1 = UniversalPosition::<Standard>::new_m(-2, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_subsumption_m_zero_offset_to_negative() {
        // Test: M + 0#0 ≤^ε_s M + (-1)#1
        // Check: f > e (1 > 0) ✓
        //        |j - i| ≤ f - e (|-1 - 0| = 1 ≤ 1 - 0 = 1) ✓
        let pos1 = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_no_subsumption_m_distance_exceeds() {
        // Test: M + (-4)#2 does NOT subsume M + (-2)#2 (with n=3)
        // Check: f > e (2 > 1) ✓
        //        |j - i| ≤ f - e (|-2 - (-4)| = 2 ≤ 2 - 1 = 1) ✗
        let pos1 = UniversalPosition::<Standard>::new_m(-4, 1, 3)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_no_subsumption_m_equal_errors() {
        // M-type positions with equal errors: no subsumption
        let pos1 = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-2, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(!subsumes(&pos1, &pos2, 2));
    }

    // =========================================================================
    // Reflexivity and Symmetry Tests
    // =========================================================================

    #[test]
    fn test_no_reflexive_subsumption() {
        // Position cannot subsume itself (requires f > e)
        let pos = UniversalPosition::<Standard>::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(!subsumes(&pos, &pos, 2));
    }

    #[test]
    fn test_subsumption_not_symmetric() {
        // If A subsumes B, B does not subsume A
        let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        assert!(subsumes(&pos1, &pos2, 3)); // pos1 subsumes pos2
        assert!(!subsumes(&pos2, &pos1, 3)); // but not vice versa
    }

    // =========================================================================
    // Variant Tests
    // =========================================================================

    #[test]
    fn test_subsumption_with_transposition_variant() {
        // Subsumption works the same for Transposition variant
        let pos1 = UniversalPosition::<Transposition>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Transposition>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_transposition_subsumption_respects_variant_state() {
        let usual = UniversalPosition::<Transposition>::new_i_with_state(
            0,
            1,
            3,
            TranspositionState::Usual,
        )
        .expect("test fixture: usual transposition position");
        let transposing = UniversalPosition::<Transposition>::new_i_with_state(
            1,
            1,
            3,
            TranspositionState::Transposing,
        )
        .expect("test fixture: transposing position");
        let higher_error_transposing = UniversalPosition::<Transposition>::new_i_with_state(
            0,
            2,
            3,
            TranspositionState::Transposing,
        )
        .expect("test fixture: higher-error transposing position");
        let later_usual = UniversalPosition::<Transposition>::new_i_with_state(
            1,
            2,
            3,
            TranspositionState::Usual,
        )
        .expect("test fixture: later usual transposition position");

        assert!(
            !subsumes(&usual, &higher_error_transposing, 3),
            "usual positions must not subsume transposition-in-progress states"
        );
        assert!(
            subsumes(&transposing, &later_usual, 3),
            "transposition-in-progress states use the adjusted |j + 1 - i| distance"
        );
        assert!(
            !subsumes(&transposing, &higher_error_transposing, 3),
            "transposition-in-progress states must not subsume each other"
        );
    }

    #[test]
    fn test_subsumption_with_merge_split_variant() {
        // Subsumption works the same for MergeAndSplit variant
        let pos1 = UniversalPosition::<MergeAndSplit>::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<MergeAndSplit>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_merge_split_subsumption_respects_variant_state() {
        let usual =
            UniversalPosition::<MergeAndSplit>::new_i_with_state(0, 1, 3, MergeSplitState::Usual)
                .expect("test fixture: usual merge-split position");
        let splitting = UniversalPosition::<MergeAndSplit>::new_i_with_state(
            0,
            1,
            3,
            MergeSplitState::Splitting,
        )
        .expect("test fixture: splitting position");
        let higher_error_usual =
            UniversalPosition::<MergeAndSplit>::new_i_with_state(0, 2, 3, MergeSplitState::Usual)
                .expect("test fixture: higher-error usual merge-split position");
        let higher_error_splitting = UniversalPosition::<MergeAndSplit>::new_i_with_state(
            0,
            2,
            3,
            MergeSplitState::Splitting,
        )
        .expect("test fixture: higher-error splitting position");

        assert!(
            !subsumes(&usual, &higher_error_splitting, 3),
            "usual positions must not subsume split-in-progress states"
        );
        assert!(
            subsumes(&splitting, &higher_error_usual, 3),
            "split-in-progress positions may subsume usual positions under the merge/split rule"
        );
        assert!(
            !subsumes(&splitting, &higher_error_splitting, 3),
            "split-in-progress positions must not subsume each other"
        );
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_subsumption_max_distance() {
        // Test with positions at maximum distance n=3
        let pos1 = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(3, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        // Check: f > e (3 > 2) ✓
        //        |j - i| ≤ f - e (|3 - 2| = 1 ≤ 3 - 2 = 1) ✓
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumption_i_type_distance_does_not_wrap_at_u8_boundary() {
        let pos1 = UniversalPosition::<Standard>::new_i(-200, 200, 255)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(56, 255, 255)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        // Actual distance is 256 and error difference is 55, so there is no
        // subsumption. Truncating distance to u8 would incorrectly compare 0.
        assert!(!subsumes(&pos1, &pos2, 255));
    }

    #[test]
    fn test_subsumption_m_type_distance_does_not_wrap_at_u8_boundary() {
        let pos1 = UniversalPosition::<Standard>::new_m(-300, 45, 255)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(0, 255, 255)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        // Actual distance is 300 and error difference is 210, so there is no
        // subsumption. Truncating distance to u8 would incorrectly compare 44.
        assert!(!subsumes(&pos1, &pos2, 255));
    }

    #[test]
    fn test_subsumption_min_distance() {
        // Test with n=1 (minimum useful distance)
        let pos1 = UniversalPosition::<Standard>::new_i(0, 0, 1)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(0, 1, 1)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        // Check: f > e (1 > 0) ✓
        //        |j - i| ≤ f - e (|0 - 0| = 0 ≤ 1 - 0 = 1) ✓
        assert!(subsumes(&pos1, &pos2, 1));
    }

    #[test]
    fn test_subsumption_mixed_sign_offsets() {
        // Test with pos1 negative, pos2 positive
        let pos1 = UniversalPosition::<Standard>::new_i(-1, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(1, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        // Check: f > e (3 > 2) ✓
        //        |j - i| ≤ f - e (|1 - (-1)| = 2 ≤ 3 - 2 = 1) ✗
        assert!(!subsumes(&pos1, &pos2, 3));
    }
}
