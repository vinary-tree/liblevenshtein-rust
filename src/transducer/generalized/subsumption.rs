//! Subsumption Relation for Generalized Positions
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
//! # Generalized Implementation
//!
//! Generalized positions include usual states and intermediate transposition or
//! splitting states. The standard offset/error subsumption rule applies only
//! within the same position variant; cross-variant subsumption is intentionally
//! rejected because intermediate states have different futures.

use super::position::GeneralizedPosition;

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
///
/// # Generalized Variant Rule
///
/// The standard offset/error subsumption rule is applied only to positions of
/// the same variant. This keeps transposition and split intermediates from
/// being collapsed into usual states with incompatible continuation semantics.
///
/// # Example
///
/// ```ignore
/// let pos1 = GeneralizedPosition::new_i(4, 1, 3)?;
/// let pos2 = GeneralizedPosition::new_i(5, 2, 3)?;
/// assert!(subsumes(&pos1, &pos2, 3));  // 4#1 <^ε_s 5#2
/// ```
#[inline(always)]
pub fn subsumes(pos1: &GeneralizedPosition, pos2: &GeneralizedPosition, max_distance: u8) -> bool {
    subsumes_standard(pos1, pos2, max_distance)
}

/// Standard subsumption rule implementation
///
/// From Definition 11 (page 18): i#e ≤^ε_s j#f ⇔ f > e ∧ |j - i| ≤ f - e
///
/// # Phase 2d Extension
///
/// With multi-character operations, positions can be in intermediate states (transposing, splitting).
/// **Key Rule**: Only positions of the same variant can subsume each other.
///
/// Rationale:
/// - Transposing/splitting positions represent intermediate states with different constraints
/// - They have different futures in the automaton
/// - Cross-variant subsumption would be incorrect
///
/// # Arguments
///
/// - `pos1`: Position i#e (potential subsumer)
/// - `pos2`: Position j#f (potentially subsumed)
/// - `_max_distance`: Maximum distance n (unused for standard, kept for consistency)
///
/// # Returns
///
/// `true` if pos1 <^ε_s pos2, `false` otherwise
///
/// # Conditions
///
/// 1. Both must be same variant (INonFinal, MFinal, ITransposing, etc.)
/// 2. pos2 has more errors: f > e
/// 3. Positions close enough: |j - i| ≤ f - e
fn subsumes_standard(
    pos1: &GeneralizedPosition,
    pos2: &GeneralizedPosition,
    _max_distance: u8,
) -> bool {
    use GeneralizedPosition::*;

    // Helper function for the actual subsumption check (same for all variants)
    fn check_subsumption(i: i32, e: u8, j: i32, f: u8) -> bool {
        // f > e (pos2 has more errors available)
        if f <= e {
            return false;
        }

        let error_diff = (f - e) as i32;
        let offset_diff = (j - i).abs();

        // |j - i| ≤ f - e
        offset_diff <= error_diff
    }

    match (pos1, pos2) {
        // I-type subsumes I-type (usual state)
        (
            INonFinal {
                offset: i,
                errors: e,
            },
            INonFinal {
                offset: j,
                errors: f,
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // M-type subsumes M-type (usual state)
        (
            MFinal {
                offset: i,
                errors: e,
            },
            MFinal {
                offset: j,
                errors: f,
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // I-type transposing subsumes I-type transposing
        (
            ITransposing {
                offset: i,
                errors: e,
            },
            ITransposing {
                offset: j,
                errors: f,
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // M-type transposing subsumes M-type transposing
        (
            MTransposing {
                offset: i,
                errors: e,
            },
            MTransposing {
                offset: j,
                errors: f,
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // I-type splitting subsumes I-type splitting
        (
            ISplitting {
                offset: i,
                errors: e,
                ..
            },
            ISplitting {
                offset: j,
                errors: f,
                ..
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // M-type splitting subsumes M-type splitting
        (
            MSplitting {
                offset: i,
                errors: e,
                ..
            },
            MSplitting {
                offset: j,
                errors: f,
                ..
            },
        ) => check_subsumption(*i, *e, *j, *f),

        // Different variants never subsume each other
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsumes_i_type_standard() {
        // Test case: 1#2 <^ε_s 2#3
        // Valid positions: |1| ≤ 2 ✓ and |2| ≤ 3 ✓
        // f > e: 3 > 2 ✓
        // |j - i| ≤ f - e: |2 - 1| = 1 ≤ 3 - 2 = 1 ✓
        let pos1 = GeneralizedPosition::new_i(1, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(2, 3, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_not_subsumes_too_far() {
        // 0#2 should not subsume -2#3 (offset difference too large)
        // Valid positions: |0| ≤ 2 ✓ and |-2| ≤ 3 ✓
        // f > e: 3 > 2 ✓
        // |j - i| ≤ f - e: |-2 - 0| = 2 ≤ 3 - 2 = 1? NO
        let pos1 = GeneralizedPosition::new_i(0, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(-2, 3, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_not_subsumes_same_errors() {
        // Cannot subsume if same error count
        let pos1 = GeneralizedPosition::new_i(0, 1, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(1, 1, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert!(!subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumes_m_type_standard() {
        // M-type subsumption works the same way
        let pos1 = GeneralizedPosition::new_m(-1, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        let pos2 = GeneralizedPosition::new_m(-2, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_not_subsumes_different_types() {
        // I-type cannot subsume M-type and vice versa
        let i_pos = GeneralizedPosition::new_i(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let m_pos = GeneralizedPosition::new_m(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert!(!subsumes(&i_pos, &m_pos, 2));
        assert!(!subsumes(&m_pos, &i_pos, 2));
    }

    #[test]
    fn test_subsumes_reflexive_false() {
        // A position cannot subsume itself (requires f > e)
        let pos = GeneralizedPosition::new_i(0, 1, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert!(!subsumes(&pos, &pos, 3));
    }

    #[test]
    fn test_subsumes_boundary_case() {
        // Boundary case: |j - i| = f - e exactly
        let pos1 = GeneralizedPosition::new_i(0, 0, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(2, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        // f > e: 2 > 0 ✓
        // |j - i| ≤ f - e: |2 - 0| = 2 ≤ 2 - 0 = 2 ✓
        assert!(subsumes(&pos1, &pos2, 3));
    }

    #[test]
    fn test_subsumes_negative_offsets() {
        // Test with negative offsets
        let pos1 = GeneralizedPosition::new_i(-1, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(0, 3, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        // f > e: 3 > 2 ✓
        // |j - i| ≤ f - e: |0 - (-1)| = 1 ≤ 3 - 2 = 1 ✓
        assert!(subsumes(&pos1, &pos2, 3));
    }

    // Tests for Phase 2d.2: Variant subsumption

    #[test]
    fn test_same_variant_subsumption_transposing() {
        // I+(-1)#1_t subsumes I+0#2_t (same variant)
        let pos1 = GeneralizedPosition::new_i_transposing(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        let pos2 = GeneralizedPosition::new_i_transposing(0, 2, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        // f > e: 2 > 1 ✓
        // |j - i| ≤ f - e: |0 - (-1)| = 1 ≤ 2 - 1 = 1 ✓
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_same_variant_subsumption_splitting() {
        // I+(-1)#1_s subsumes I+0#2_s (same variant)
        let pos1 = GeneralizedPosition::new_i_splitting(-1, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        let pos2 = GeneralizedPosition::new_i_splitting(0, 2, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_same_variant_subsumption_m_transposing() {
        // M+(-1)#1_t subsumes M+(-2)#2_t (same variant)
        let pos1 = GeneralizedPosition::new_m_transposing(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        let pos2 = GeneralizedPosition::new_m_transposing(-2, 2, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_same_variant_subsumption_m_splitting() {
        // M+(-1)#1_s subsumes M+(-2)#2_s (same variant)
        let pos1 = GeneralizedPosition::new_m_splitting(-1, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");
        let pos2 = GeneralizedPosition::new_m_splitting(-2, 2, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");
        assert!(subsumes(&pos1, &pos2, 2));
    }

    #[test]
    fn test_different_variant_no_subsumption_usual_vs_transposing() {
        // I+0#1 (usual) does NOT subsume I+0#2_t (transposing)
        let pos1 = GeneralizedPosition::new_i(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i_transposing(0, 2, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        assert!(!subsumes(&pos1, &pos2, 2));

        // And vice versa
        assert!(!subsumes(&pos2, &pos1, 2));
    }

    #[test]
    fn test_different_variant_no_subsumption_transposing_vs_splitting() {
        // I+0#1_t (transposing) does NOT subsume I+0#2_s (splitting)
        let pos1 = GeneralizedPosition::new_i_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        let pos2 = GeneralizedPosition::new_i_splitting(0, 2, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert!(!subsumes(&pos1, &pos2, 2));

        // And vice versa
        assert!(!subsumes(&pos2, &pos1, 2));
    }

    #[test]
    fn test_different_variant_no_subsumption_usual_vs_splitting() {
        // I+0#1 (usual) does NOT subsume I+0#2_s (splitting)
        let pos1 = GeneralizedPosition::new_i(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i_splitting(0, 2, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert!(!subsumes(&pos1, &pos2, 2));

        // And vice versa
        assert!(!subsumes(&pos2, &pos1, 2));
    }

    #[test]
    fn test_different_variant_same_offset_errors() {
        // Even with same offset and errors, different variants don't subsume
        let i_usual = GeneralizedPosition::new_i(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let i_trans = GeneralizedPosition::new_i_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        let i_split = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");

        // No cross-variant subsumption
        assert!(!subsumes(&i_usual, &i_trans, 2));
        assert!(!subsumes(&i_usual, &i_split, 2));
        assert!(!subsumes(&i_trans, &i_usual, 2));
        assert!(!subsumes(&i_trans, &i_split, 2));
        assert!(!subsumes(&i_split, &i_usual, 2));
        assert!(!subsumes(&i_split, &i_trans, 2));
    }

    #[test]
    fn test_m_type_variants_no_cross_subsumption() {
        // M-type variants also don't subsume across types
        let m_usual = GeneralizedPosition::new_m(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        let m_trans = GeneralizedPosition::new_m_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        let m_split = GeneralizedPosition::new_m_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");

        assert!(!subsumes(&m_usual, &m_trans, 2));
        assert!(!subsumes(&m_usual, &m_split, 2));
        assert!(!subsumes(&m_trans, &m_usual, 2));
        assert!(!subsumes(&m_trans, &m_split, 2));
        assert!(!subsumes(&m_split, &m_usual, 2));
        assert!(!subsumes(&m_split, &m_trans, 2));
    }
}
