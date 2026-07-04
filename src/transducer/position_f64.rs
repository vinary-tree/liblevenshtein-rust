//! Float-weighted position in the Levenshtein automaton.
//!
//! This module provides `PositionF64`, an extension of `Position` that uses
//! float-valued accumulated costs instead of integer error counts.
//!
//! # Overview
//!
//! While `Position` tracks integer `num_errors` (0, 1, 2, ...), `PositionF64`
//! tracks float `accumulated_cost` (0.0, 0.5, 1.3, ...). This enables:
//!
//! - **Weighted edit distance**: Different costs for different operations
//! - **Fine-grained ranking**: Better differentiation between matches
//! - **Domain-specific matching**: OCR, phonetic, keyboard proximity
//!
//! # Subsumption with Floats
//!
//! The subsumption relation is adapted for float costs with epsilon handling and
//! generalized to weighted insertion/deletion costs:
//!
//! Position `p1` at (i, e) subsumes `p2` at (j, f) if:
//! - `e <= f` (accumulated cost comparison)
//! - `|i - j| * max(insertion, deletion) <= (f - e)` (weighted bounded-diagonal
//!   property; reduces to `|i - j| <= f - e` for unit insertion/deletion costs)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::{PositionF64, Algorithm};
//!
//! let p1 = PositionF64::new(3, 1.5);  // At index 3 with cost 1.5
//! let p2 = PositionF64::new(4, 2.5);  // At index 4 with cost 2.5
//!
//! // With unit ins/del costs: |3 - 4| * 1 = 1 <= (2.5 - 1.5) = 1.0 ✓
//! assert!(p1.subsumes(&p2, Algorithm::Standard, 5, 1.0));
//! ```
//!
//! # Memory Layout
//!
//! `PositionF64` is 25 bytes (2 usizes + f64 + bool) and is `Copy` for efficient
//! state manipulation during automaton traversal.

use super::algorithm::Algorithm;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Epsilon for float comparisons in subsumption.
const EPSILON: f64 = 1e-9;

#[inline(always)]
fn nonnegative_floor_to_usize(value: f64) -> Option<usize> {
    if value.is_nan() || value < 0.0 {
        None
    } else if !value.is_finite() {
        Some(usize::MAX)
    } else {
        finite_nonnegative_floor_to_usize(value).or(Some(usize::MAX))
    }
}

fn finite_nonnegative_floor_to_usize(value: f64) -> Option<usize> {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);

    const MANTISSA_BITS: i32 = 52;
    const EXPONENT_BIAS: i32 = 1023;
    const EXPONENT_MASK: u64 = 0x7ff;
    const MANTISSA_MASK: u64 = (1_u64 << MANTISSA_BITS) - 1;
    const HIDDEN_BIT: u64 = 1_u64 << MANTISSA_BITS;

    let bits = value.to_bits();
    let exponent_bits = u16::try_from((bits >> MANTISSA_BITS) & EXPONENT_MASK).ok()?;

    if exponent_bits == 0 {
        return Some(0);
    }

    let exponent = i32::from(exponent_bits) - EXPONENT_BIAS;
    if exponent < 0 {
        return Some(0);
    }

    let significand = HIDDEN_BIT | (bits & MANTISSA_MASK);
    let integer = if exponent >= MANTISSA_BITS {
        let shift = u32::try_from(exponent - MANTISSA_BITS).ok()?;
        u128::from(significand).checked_shl(shift)?
    } else {
        let shift = u32::try_from(MANTISSA_BITS - exponent).ok()?;
        u128::from(significand >> shift)
    };

    usize::try_from(integer).ok()
}

/// A position in the float-weighted Levenshtein automaton state.
///
/// Analogous to [`Position`](super::Position), but with float-valued
/// `accumulated_cost` instead of integer `num_errors`.
///
/// # Fields
///
/// - `term_index`: Characters consumed from the query term
/// - `accumulated_cost`: Total cost of operations to reach this position
/// - `is_special`: Flag for extended algorithm states (transposition/merge-split)
///
/// # Copy Semantics
///
/// `PositionF64` is `Copy` (25 bytes) for efficient state transitions.
/// This eliminates allocation overhead when copying positions during traversal.
#[derive(Debug, Clone, Copy)]
pub struct PositionF64 {
    /// Index into the query term (characters consumed).
    pub term_index: usize,

    /// Accumulated cost of edit operations to reach this position.
    /// Unlike `Position.num_errors`, this is a float allowing weighted operations.
    pub accumulated_cost: f64,

    /// Special flag for extended algorithm states.
    ///
    /// - For Transposition: indicates a transposition is in progress
    /// - For MergeAndSplit: indicates a merge/split operation state
    pub is_special: bool,
}

impl PositionF64 {
    /// Create a new position with the given index and cost.
    ///
    /// # Arguments
    ///
    /// * `term_index` - Characters consumed from query
    /// * `accumulated_cost` - Total cost to reach this position (≥ 0.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::PositionF64;
    ///
    /// let pos = PositionF64::new(5, 2.5);
    /// assert_eq!(pos.term_index, 5);
    /// assert!((pos.accumulated_cost - 2.5).abs() < 1e-9);
    /// assert!(!pos.is_special);
    /// ```
    #[inline(always)]
    pub fn new(term_index: usize, accumulated_cost: f64) -> Self {
        debug_assert!(
            accumulated_cost >= 0.0,
            "Accumulated cost must be non-negative"
        );
        Self {
            term_index,
            accumulated_cost,
            is_special: false,
        }
    }

    /// Create a new special position (for extended algorithms).
    ///
    /// Special positions track intermediate states for transposition
    /// and merge/split operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::PositionF64;
    ///
    /// let pos = PositionF64::new_special(3, 1.0);
    /// assert!(pos.is_special);
    /// ```
    #[inline(always)]
    pub fn new_special(term_index: usize, accumulated_cost: f64) -> Self {
        debug_assert!(
            accumulated_cost >= 0.0,
            "Accumulated cost must be non-negative"
        );
        Self {
            term_index,
            accumulated_cost,
            is_special: true,
        }
    }

    /// Create the initial position at index 0 with cost 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::PositionF64;
    ///
    /// let pos = PositionF64::initial();
    /// assert_eq!(pos.term_index, 0);
    /// assert_eq!(pos.accumulated_cost, 0.0);
    /// ```
    #[inline(always)]
    pub fn initial() -> Self {
        Self::new(0, 0.0)
    }

    /// Check if this position subsumes another position.
    ///
    /// Position `p1` subsumes `p2` if all candidates reachable from `p2`
    /// are also reachable from `p1`. This allows pruning redundant states.
    ///
    /// # Float Subsumption
    ///
    /// The subsumption formula adapts the integer version for floats and
    /// generalizes it to weighted insertion/deletion costs:
    /// - `self.accumulated_cost <= other.accumulated_cost` (with epsilon)
    /// - `|i - j| * max(insertion, deletion) <= (f - e)` (worst-case cost of
    ///   realigning the term index by `|i - j|` steps must fit within the cost
    ///   slack; reduces to the classic `|i - j| <= f - e` when insertion and
    ///   deletion both cost `1`)
    ///
    /// Using `max(insertion, deletion)` is conservative and sound: it only
    /// prunes `other` when even the most expensive realignment is affordable,
    /// so a position that leads to the sole in-budget match is never dropped
    /// (which the naive unit-cost bound could do when `insertion`/`deletion > 1`).
    ///
    /// # Algorithm-Specific Logic
    ///
    /// | Algorithm | Special Handling |
    /// |-----------|-----------------|
    /// | Standard | Basic formula |
    /// | Transposition | Special position compatibility |
    /// | MergeAndSplit | Only same variant-state positions subsume each other |
    ///
    /// # Parameters
    ///
    /// - `other`: The position to check subsumption against
    /// - `algorithm`: The algorithm variant determining subsumption rules
    /// - `query_length`: Length of the query term
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{PositionF64, Algorithm};
    ///
    /// let p1 = PositionF64::new(3, 1.0);
    /// let p2 = PositionF64::new(4, 2.5);
    ///
    /// // |3 - 4| * max(1, 1) = 1 <= (2.5 - 1.0) = 1.5 ✓  (unit ins/del costs)
    /// assert!(p1.subsumes(&p2, Algorithm::Standard, 10, 1.0));
    ///
    /// // Cannot subsume position with lower cost
    /// assert!(!p2.subsumes(&p1, Algorithm::Standard, 10, 1.0));
    /// ```
    pub fn subsumes(
        &self,
        other: &PositionF64,
        algorithm: Algorithm,
        query_length: usize,
        max_index_op_cost: f64,
    ) -> bool {
        let i = self.term_index;
        let e = self.accumulated_cost;
        let s = self.is_special;

        let j = other.term_index;
        let f = other.accumulated_cost;
        let t = other.is_special;

        // Must have lower or equal cost to subsume (with epsilon tolerance)
        if e > f + EPSILON {
            return false;
        }

        let cost_slack = f - e;

        match algorithm {
            Algorithm::Standard => {
                // Weighted standard subsumption: realigning the term index by
                // |i - j| steps costs at most |i - j| * max(insertion, deletion),
                // which must fit within the cost slack (f - e). Reduces to the
                // classic |i - j| <= f - e when ins/del both cost 1.
                let index_diff = i.abs_diff(j) as f64;
                index_diff * max_index_op_cost <= cost_slack + EPSILON
            }

            Algorithm::Transposition => {
                if s {
                    if t {
                        // Both special: must be at same position
                        return i == j;
                    }
                    // lhs special, rhs not: requires rhs at query length and same position
                    let Some(f_as_usize) = nonnegative_floor_to_usize(f) else {
                        return false;
                    };
                    return (f_as_usize == query_length) && (i == j);
                }

                if t {
                    // A non-special (normal) position cannot subsume a special
                    // transposition-in-progress position: the special position still
                    // owes the second half of a transposition that a normal position
                    // cannot reproduce. Reachable only with `s == false` — every
                    // `s == true` case returns in the block above — so the previous
                    // "both special" arm here was dead code and has been removed.
                    return false;
                }

                // Neither special: standard formula (weighted by max(ins, del)).
                let index_diff = i.abs_diff(j) as f64;
                index_diff * max_index_op_cost <= cost_slack + EPSILON
            }

            Algorithm::MergeAndSplit => {
                // MergeAndSplit normal and split-in-progress positions have
                // different continuations, so they cannot subsume each other.
                if s != t {
                    return false;
                }

                // The automaton never generates representatives past the query.
                if i > query_length {
                    return false;
                }

                // A final pending split cannot consume the second split
                // character required by a non-final pending split.
                if s && i >= query_length && j < query_length {
                    return false;
                }

                // Must have strictly lower cost for MergeAndSplit
                // This allows (i,e,false) and (i,e,true) to coexist
                if e >= f - EPSILON {
                    return false;
                }

                // Keep pruning same-index only. Cross-index pruning can erase
                // delete-closure witnesses needed by split/merge completions.
                i == j
            }
        }
    }

    /// Compare positions for sorting (lexicographic order).
    ///
    /// Order: term_index (asc), then accumulated_cost (asc), then is_special (false < true)
    ///
    /// # Note
    ///
    /// Float comparison uses total_cmp for consistent ordering that handles
    /// NaN and -0.0 correctly.
    pub fn compare(&self, other: &PositionF64) -> Ordering {
        self.term_index
            .cmp(&other.term_index)
            .then_with(|| self.accumulated_cost.total_cmp(&other.accumulated_cost))
            .then_with(|| self.is_special.cmp(&other.is_special))
    }

    /// Check if this position is approximately equal to another.
    ///
    /// Uses epsilon tolerance for float comparison.
    pub fn approx_eq(&self, other: &PositionF64) -> bool {
        self.term_index == other.term_index
            && (self.accumulated_cost - other.accumulated_cost).abs() < EPSILON
            && self.is_special == other.is_special
    }
}

impl PartialEq for PositionF64 {
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

impl PartialOrd for PositionF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for PositionF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

impl Eq for PositionF64 {}

impl Hash for PositionF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term_index.hash(state);
        self.accumulated_cost.to_bits().hash(state);
        self.is_special.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const TEST_EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TEST_EPSILON
    }

    fn position_hash(position: &PositionF64) -> u64 {
        let mut hasher = DefaultHasher::new();
        position.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn nonnegative_floor_to_usize_handles_float_boundaries() {
        assert_eq!(nonnegative_floor_to_usize(f64::NAN), None);
        assert_eq!(nonnegative_floor_to_usize(-1.0), None);
        assert_eq!(nonnegative_floor_to_usize(-0.0), Some(0));
        assert_eq!(nonnegative_floor_to_usize(0.0), Some(0));
        assert_eq!(nonnegative_floor_to_usize(2.9), Some(2));
        assert_eq!(nonnegative_floor_to_usize(f64::INFINITY), Some(usize::MAX));
        assert_eq!(nonnegative_floor_to_usize(f64::MAX), Some(usize::MAX));
    }

    #[test]
    fn test_position_creation() {
        let pos = PositionF64::new(5, 2.5);
        assert_eq!(pos.term_index, 5);
        assert!(approx_eq(pos.accumulated_cost, 2.5));
        assert!(!pos.is_special);
    }

    #[test]
    fn test_position_special() {
        let pos = PositionF64::new_special(3, 1.5);
        assert!(pos.is_special);
        assert_eq!(pos.term_index, 3);
        assert!(approx_eq(pos.accumulated_cost, 1.5));
    }

    #[test]
    fn test_position_initial() {
        let pos = PositionF64::initial();
        assert_eq!(pos.term_index, 0);
        assert!(approx_eq(pos.accumulated_cost, 0.0));
        assert!(!pos.is_special);
    }

    #[test]
    fn test_subsumption_standard_basic() {
        let max_distance = 5;

        // (5, 2.0) should subsume (5, 3.0) - same position, lower cost
        let p1 = PositionF64::new(5, 2.0);
        let p2 = PositionF64::new(5, 3.0);
        assert!(
            p1.subsumes(&p2, Algorithm::Standard, max_distance, 1.0),
            "p1(5, 2.0) should subsume p2(5, 3.0)"
        );

        // (5, 2.0) should subsume (4, 3.0) - |5-4| = 1 <= (3.0-2.0) = 1.0
        let p3 = PositionF64::new(5, 2.0);
        let p4 = PositionF64::new(4, 3.0);
        assert!(
            p3.subsumes(&p4, Algorithm::Standard, max_distance, 1.0),
            "p3(5, 2.0) should subsume p4(4, 3.0)"
        );
    }

    #[test]
    fn test_subsumption_standard_float_costs() {
        let max_distance = 5;

        // (3, 1.5) should subsume (4, 2.6) - |3-4| = 1 <= (2.6-1.5) = 1.1
        let p1 = PositionF64::new(3, 1.5);
        let p2 = PositionF64::new(4, 2.6);
        assert!(
            p1.subsumes(&p2, Algorithm::Standard, max_distance, 1.0),
            "p1(3, 1.5) should subsume p2(4, 2.6)"
        );

        // (3, 1.5) should NOT subsume (5, 2.3) - |3-5| = 2 > (2.3-1.5) = 0.8
        let p3 = PositionF64::new(3, 1.5);
        let p4 = PositionF64::new(5, 2.3);
        assert!(
            !p3.subsumes(&p4, Algorithm::Standard, max_distance, 1.0),
            "p3(3, 1.5) should NOT subsume p4(5, 2.3)"
        );
    }

    #[test]
    fn test_subsumption_cannot_subsume_lower_cost() {
        let max_distance = 5;

        // (5, 3.0) should NOT subsume (5, 2.0) - higher cost
        let p1 = PositionF64::new(5, 3.0);
        let p2 = PositionF64::new(5, 2.0);
        assert!(
            !p1.subsumes(&p2, Algorithm::Standard, max_distance, 1.0),
            "Higher cost position cannot subsume lower cost position"
        );
    }

    #[test]
    fn test_subsumption_transposition_special() {
        let max_distance = 5;

        // Both special: must be at same position
        let p1 = PositionF64::new_special(5, 2.0);
        let p2 = PositionF64::new_special(5, 3.0);
        assert!(
            p1.subsumes(&p2, Algorithm::Transposition, max_distance, 1.0),
            "special(5, 2.0) should subsume special(5, 3.0)"
        );

        let p3 = PositionF64::new_special(5, 2.0);
        let p4 = PositionF64::new_special(6, 3.0);
        assert!(
            !p3.subsumes(&p4, Algorithm::Transposition, max_distance, 1.0),
            "special(5, 2.0) should NOT subsume special(6, 3.0)"
        );

        // Normal cannot subsume special
        let p5 = PositionF64::new(5, 2.0);
        let p6 = PositionF64::new_special(4, 3.0);
        assert!(
            !p5.subsumes(&p6, Algorithm::Transposition, max_distance, 1.0),
            "normal cannot subsume special in transposition"
        );
    }

    #[test]
    fn test_subsumption_transposition_special_floor_conversion_boundaries() {
        let lhs = PositionF64::new_special(5, 1.0);
        let rhs = PositionF64::new(5, 5.9);
        assert!(lhs.subsumes(&rhs, Algorithm::Transposition, 5, 1.0));
        assert!(!lhs.subsumes(&rhs, Algorithm::Transposition, 6, 1.0));

        let nan_rhs = PositionF64 {
            term_index: 5,
            accumulated_cost: f64::NAN,
            is_special: false,
        };
        assert!(!lhs.subsumes(&nan_rhs, Algorithm::Transposition, 5, 1.0));

        let negative_lhs = PositionF64 {
            term_index: 0,
            accumulated_cost: -2.0,
            is_special: true,
        };
        let negative_rhs = PositionF64 {
            term_index: 0,
            accumulated_cost: -1.0,
            is_special: false,
        };
        assert!(!negative_lhs.subsumes(&negative_rhs, Algorithm::Transposition, 0, 1.0));

        let max_lhs = PositionF64::new_special(usize::MAX, 1.0);
        let infinite_rhs = PositionF64::new(usize::MAX, f64::INFINITY);
        assert!(max_lhs.subsumes(&infinite_rhs, Algorithm::Transposition, usize::MAX, 1.0));
    }

    #[test]
    fn test_subsumption_merge_split() {
        let query_length = 5;

        // Different variant states cannot subsume each other.
        let p1 = PositionF64::new_special(5, 2.0);
        let p2 = PositionF64::new(5, 3.0);
        assert!(
            !p1.subsumes(&p2, Algorithm::MergeAndSplit, query_length, 1.0),
            "special cannot subsume non-special in merge-split"
        );

        let p2a = PositionF64::new(5, 2.0);
        let p2b = PositionF64::new_special(4, 3.0);
        assert!(
            !p2a.subsumes(&p2b, Algorithm::MergeAndSplit, query_length, 1.0),
            "normal cannot subsume special in merge-split"
        );

        let p2c = PositionF64::new_special(5, 1.0);
        let p2d = PositionF64::new_special(4, 2.0);
        assert!(
            !p2c.subsumes(&p2d, Algorithm::MergeAndSplit, query_length, 1.0),
            "final special cannot subsume non-final special in merge-split"
        );

        // Cross-index pruning is intentionally disabled for MergeAndSplit.
        let p3 = PositionF64::new(5, 2.0);
        let p4 = PositionF64::new(4, 3.0);
        assert!(
            !p3.subsumes(&p4, Algorithm::MergeAndSplit, query_length, 1.0),
            "normal(5, 2.0) should NOT subsume normal(4, 3.0)"
        );

        let p5 = PositionF64::new_special(5, 2.0);
        let p6 = PositionF64::new_special(5, 3.0);
        assert!(
            p5.subsumes(&p6, Algorithm::MergeAndSplit, query_length, 1.0),
            "special(5, 2.0) should subsume special(5, 3.0)"
        );
    }

    #[test]
    fn test_position_ordering() {
        let p1 = PositionF64::new(3, 1.0);
        let p2 = PositionF64::new(3, 2.0);
        let p3 = PositionF64::new(4, 1.0);

        assert!(p1 < p2); // Same index, lower cost
        assert!(p1 < p3); // Lower index
        assert!(p2 < p3); // Lower index
    }

    #[test]
    fn test_approx_eq() {
        let p1 = PositionF64::new(3, 1.5);
        let p2 = PositionF64::new(3, 1.5 + 1e-12); // Within epsilon
        let p3 = PositionF64::new(3, 1.6);

        assert!(p1.approx_eq(&p2));
        assert_ne!(p1, p2);
        assert!(!p1.approx_eq(&p3));
    }

    #[test]
    fn test_eq_ord_hash_share_total_float_identity() {
        let positive_zero = PositionF64::new(0, 0.0);
        let negative_zero = PositionF64::new(0, -0.0);

        assert!(positive_zero.approx_eq(&negative_zero));
        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let nan_position = PositionF64 {
            term_index: 0,
            accumulated_cost: f64::NAN,
            is_special: false,
        };
        let same_nan_position = nan_position;

        assert_eq!(nan_position, same_nan_position);
        assert_eq!(nan_position.cmp(&same_nan_position), Ordering::Equal);
        assert_eq!(
            position_hash(&nan_position),
            position_hash(&same_nan_position)
        );
    }
}
