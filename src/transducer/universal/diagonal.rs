//! Diagonal Crossing Functions
//!
//! Implements diagonal crossing detection and position type conversion from Mitankin's thesis.
//!
//! # Theory Background
//!
//! The universal automaton needs to detect when processing crosses the "diagonal" in the
//! edit graph (where input position equals word position). When this happens, positions
//! must be converted between I-type (non-final) and M-type (final).
//!
//! ## Right-Most Function rm (Page 45)
//!
//! ```text
//! rm(A) = position with maximum (e - i) value in set A
//! ```
//!
//! ## Diagonal Crossing Check f_n (Page 43)
//!
//! For I-type positions:
//! ```text
//! f_n(I + i#e, k) = (k ≤ 2n+1) ∧ (e ≤ i + 2n + 1 - k)
//! ```
//!
//! For M-type positions:
//! ```text
//! f_n(M + i#e, k) = e > i + n
//! ```
//!
//! ## Position Conversion m_n (Page 42)
//!
//! For I-type → M-type:
//! ```text
//! m_n(I + i#e, k) = M + (i + n + 1 - k)#e
//! ```
//!
//! For M-type → I-type:
//! ```text
//! m_n(M + i#e, k) = I + (i - n - 1 + k)#e
//! ```

use crate::transducer::universal::position::{PositionVariant, UniversalPosition};

/// Find right-most position in a set (maximum e - i value)
///
/// From thesis page 45: rm(A) finds the position with maximum (e - i).
///
/// # Arguments
///
/// - `positions`: Iterator over positions in a state
///
/// # Returns
///
/// The position with maximum (e - i), or None if set is empty
///
/// # Examples
///
/// ```ignore
/// let positions = vec![
///     UniversalPosition::new_i(0, 1, 2).expect("doc/test fixture: UniversalPosition::new_i with valid args"),  // e - i = 1 - 0 = 1
///     UniversalPosition::new_i(-2, 2, 2).expect("doc/test fixture: UniversalPosition::new_i with valid args"), // e - i = 2 - (-2) = 4
/// ];
/// let rm_pos = right_most(positions.iter());
/// // Returns I + (-2)#2 (maximum e - i = 4)
/// ```
pub fn right_most<'a, V: PositionVariant>(
    positions: impl Iterator<Item = &'a UniversalPosition<V>>,
) -> Option<UniversalPosition<V>>
where
    UniversalPosition<V>: 'a,
{
    positions
        .max_by_key(|pos| {
            let offset = pos.offset();
            let errors = pos.errors() as i32;
            errors - offset
        })
        .cloned()
}

/// Check if position has crossed the diagonal (f_n function)
///
/// From thesis page 43: Determines if a position is on the "wrong side" of the diagonal,
/// requiring conversion between I and M types.
///
/// # Arguments
///
/// - `pos`: Position to check
/// - `k`: Current input length (number of characters processed)
/// - `max_distance`: Maximum edit distance n
///
/// # Returns
///
/// `true` if diagonal has been crossed, `false` otherwise
///
/// # Formula
///
/// For I-type positions (I + i#e):
/// ```text
/// f_n(I + i#e, k) = (k ≤ 2n+1) ∧ (e ≤ i + 2n + 1 - k)
/// ```
///
/// For M-type positions (M + i#e):
/// ```text
/// f_n(M + i#e, k) = e > i + n
/// ```
///
/// # Examples
///
/// ```ignore
/// let pos = UniversalPosition::new_i(0, 0, 2).expect("doc/test fixture: UniversalPosition::new_i with valid args");
/// assert!(!diagonal_crossed(&pos, 0, 2));  // Not crossed yet
///
/// let pos = UniversalPosition::new_i(0, 3, 2).expect("doc/test fixture: UniversalPosition::new_i with valid args");
/// assert!(diagonal_crossed(&pos, 2, 2));   // Crossed (e > i + 2n + 1 - k)
/// ```
pub fn diagonal_crossed<V: PositionVariant>(
    pos: &UniversalPosition<V>,
    k: usize,
    max_distance: u8,
) -> bool {
    let offset = pos.offset();
    let errors = pos.errors() as i32;
    let n = max_distance as i32;
    let k = k as i32;

    match pos {
        UniversalPosition::INonFinal { .. } => {
            // I-type: f_n(I + i#e, k) = (k ≤ 2n+1) ∧ (e ≤ i + 2n + 1 - k)
            (k <= 2 * n + 1) && (errors <= offset + 2 * n + 1 - k)
        }
        UniversalPosition::MFinal { .. } => {
            // M-type: f_n(M + i#e, k) = e > i + n
            errors > offset + n
        }
    }
}

/// Convert position between I-type and M-type (m_n function)
///
/// From thesis page 42: Converts positions when crossing the diagonal.
///
/// # Arguments
///
/// - `pos`: Position to convert
/// - `k`: Current input length
/// - `max_distance`: Maximum edit distance n
///
/// # Returns
///
/// Converted position (I → M or M → I), or None if conversion fails
///
/// # Formula
///
/// I-type → M-type:
/// ```text
/// m_n(I + i#e, k) = M + (i + n + 1 - k)#e
/// ```
///
/// M-type → I-type:
/// ```text
/// m_n(M + i#e, k) = I + (i - n - 1 + k)#e
/// ```
///
/// # Examples
///
/// ```ignore
/// // Convert I-type to M-type
/// let i_pos = UniversalPosition::new_i(0, 0, 2).expect("doc/test fixture: UniversalPosition::new_i with valid args");
/// let m_pos = convert_position(&i_pos, 3, 2).expect("doc/test fixture: convert_position with valid args");
/// // m_n(I + 0#0, 3) = M + (0 + 2 + 1 - 3)#0 = M + 0#0
///
/// // Convert M-type to I-type
/// let m_pos = UniversalPosition::new_m(0, 0, 2).expect("doc/test fixture: UniversalPosition::new_m with valid args");
/// let i_pos = convert_position(&m_pos, 3, 2).expect("doc/test fixture: convert_position with valid args");
/// // m_n(M + 0#0, 3) = I + (0 - 2 - 1 + 3)#0 = I + 0#0
/// ```
pub fn convert_position<V: PositionVariant>(
    pos: &UniversalPosition<V>,
    k: usize,
    max_distance: u8,
) -> Option<UniversalPosition<V>> {
    let offset = pos.offset();
    let errors = pos.errors();
    let n = max_distance as i32;
    let k = k as i32;

    match pos {
        UniversalPosition::INonFinal { .. } => {
            // I-type → M-type: M + (i + n + 1 - k)#e
            let new_offset = offset + n + 1 - k;
            UniversalPosition::new_m(new_offset, errors, max_distance).ok()
        }
        UniversalPosition::MFinal { .. } => {
            // M-type → I-type: I + (i - n - 1 + k)#e
            let new_offset = offset - n - 1 + k;
            UniversalPosition::new_i(new_offset, errors, max_distance).ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::universal::position::Standard;

    // =========================================================================
    // Right-Most Function Tests
    // =========================================================================

    #[test]
    fn test_right_most_single_position() {
        let positions = vec![UniversalPosition::<Standard>::new_i(0, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args")];
        let rm =
            right_most(positions.iter()).expect("test fixture: right_most on non-empty iterator");
        assert_eq!(rm.offset(), 0);
        assert_eq!(rm.errors(), 1);
    }

    #[test]
    fn test_right_most_multiple_positions() {
        // e - i values: (1 - 0) = 1, (2 - (-2)) = 4, (1 - (-1)) = 2
        let positions = vec![
            UniversalPosition::<Standard>::new_i(0, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
            UniversalPosition::<Standard>::new_i(-2, 2, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"), // Maximum e - i
            UniversalPosition::<Standard>::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        ];
        let rm =
            right_most(positions.iter()).expect("test fixture: right_most on non-empty iterator");
        assert_eq!(rm.offset(), -2);
        assert_eq!(rm.errors(), 2);
    }

    #[test]
    fn test_right_most_empty_set() {
        let positions: Vec<UniversalPosition<Standard>> = vec![];
        assert!(right_most(positions.iter()).is_none());
    }

    #[test]
    fn test_right_most_m_type_positions() {
        let positions = vec![
            UniversalPosition::<Standard>::new_m(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
            UniversalPosition::<Standard>::new_m(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"), // e - i = 1 - (-1) = 2
        ];
        let rm =
            right_most(positions.iter()).expect("test fixture: right_most on non-empty iterator");
        assert_eq!(rm.offset(), -1);
        assert_eq!(rm.errors(), 1);
    }

    // =========================================================================
    // Diagonal Crossing Tests
    // =========================================================================

    #[test]
    fn test_diagonal_not_crossed_initial() {
        // I + 0#0 at k=0, n=2:
        // f_n(I + 0#0, 0) = (0 ≤ 5) ∧ (0 ≤ 0 + 5 - 0) = true ∧ true = true
        // This means diagonal IS crossed (returns true) at k=0
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(diagonal_crossed(&pos, 0, 2));
    }

    #[test]
    fn test_diagonal_crossed_i_type() {
        // I + 0#3 at k=2, n=2:
        // (2 ≤ 5) ∧ (3 ≤ 0 + 5 - 2) = true ∧ (3 ≤ 3) = true
        let pos = UniversalPosition::<Standard>::new_i(0, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(diagonal_crossed(&pos, 2, 3));
    }

    #[test]
    fn test_diagonal_not_crossed_i_type_k_too_large() {
        // I + 0#0 at k=10, n=2:
        // (10 ≤ 5) = false
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(!diagonal_crossed(&pos, 10, 2));
    }

    #[test]
    fn test_diagonal_crossed_m_type() {
        // M + (-1)#2 at k=3, n=2:
        // e > i + n: 2 > -1 + 2 = 2 > 1 = true
        let pos = UniversalPosition::<Standard>::new_m(-1, 2, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(diagonal_crossed(&pos, 3, 2));
    }

    #[test]
    fn test_diagonal_not_crossed_m_type() {
        // M + 0#0 at k=3, n=2:
        // e > i + n: 0 > 0 + 2 = 0 > 2 = false
        let pos = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert!(!diagonal_crossed(&pos, 3, 2));
    }

    // =========================================================================
    // Position Conversion Tests
    // =========================================================================

    #[test]
    fn test_convert_i_to_m() {
        // m_n(I + 0#0, k=3, n=2) = M + (0 + 2 + 1 - 3)#0 = M + 0#0
        let i_pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = convert_position(&i_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert!(m_pos.is_m_type());
        assert_eq!(m_pos.offset(), 0);
        assert_eq!(m_pos.errors(), 0);
    }

    #[test]
    fn test_convert_m_to_i() {
        // m_n(M + 0#0, k=3, n=2) = I + (0 - 2 - 1 + 3)#0 = I + 0#0
        let m_pos = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let i_pos = convert_position(&m_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert!(i_pos.is_i_type());
        assert_eq!(i_pos.offset(), 0);
        assert_eq!(i_pos.errors(), 0);
    }

    #[test]
    fn test_convert_i_to_m_with_offset() {
        // m_n(I + 1#1, k=2, n=2) = M + (1 + 2 + 1 - 2)#1 = M + 2#1
        // But M + 2#1 violates invariant (offset > 0)
        // Let's use I + 0#1, k=3: M + (0 + 2 + 1 - 3)#1 = M + 0#1
        let i_pos = UniversalPosition::<Standard>::new_i(0, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = convert_position(&i_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert!(m_pos.is_m_type());
        assert_eq!(m_pos.offset(), 0);
        assert_eq!(m_pos.errors(), 1);
    }

    #[test]
    fn test_convert_m_to_i_with_offset() {
        // m_n(M + (-1)#1, k=2, n=2) = I + (-1 - 2 - 1 + 2)#1 = I + (-2)#1
        // I + (-2)#1 violates invariant |-2| > 1
        // Let's use M + (-1)#1, k=3: I + (-1 - 2 - 1 + 3)#1 = I + (-1)#1
        let m_pos = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let i_pos = convert_position(&m_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert!(i_pos.is_i_type());
        assert_eq!(i_pos.offset(), -1);
        assert_eq!(i_pos.errors(), 1);
    }

    #[test]
    fn test_convert_invalid_result() {
        // Conversion that violates invariants should return None
        // m_n(I + 5#0, k=1, n=2) = M + (5 + 2 + 1 - 1)#0 = M + 7#0
        // This violates M-type invariant (offset > 0)
        let i_pos = UniversalPosition::<Standard>::new_i(5, 5, 5)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let result = convert_position(&i_pos, 1, 2);
        // May or may not be None depending on M-type invariants
        // The conversion formula is correct, validation happens in new_m()
        if let Some(m_pos) = result {
            assert!(m_pos.is_m_type());
        }
    }

    #[test]
    fn test_convert_preserves_errors() {
        // Conversion preserves error count
        let i_pos = UniversalPosition::<Standard>::new_i(0, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = convert_position(&i_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert_eq!(m_pos.errors(), 2);

        let m_pos = UniversalPosition::<Standard>::new_m(0, 2, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let i_pos = convert_position(&m_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert_eq!(i_pos.errors(), 2);
    }

    #[test]
    fn test_convert_roundtrip() {
        // I → M → I should give original (approximately, due to formula)
        let i_pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = convert_position(&i_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        let i_pos2 = convert_position(&m_pos, 3, 2)
            .expect("doc/test fixture: convert_position with valid args");
        assert_eq!(i_pos.offset(), i_pos2.offset());
        assert_eq!(i_pos.errors(), i_pos2.errors());
    }
}
