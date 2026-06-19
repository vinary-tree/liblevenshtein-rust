//! MSM automaton position with value tracking for data-dependent costs.
//!
//! This module provides `MsmPosition`, which tracks not only the position and
//! accumulated cost in the automaton, but also the last query value needed for
//! computing the MSM C() function during transitions.
//!
//! # Difference from PositionF64
//!
//! | Aspect | PositionF64 | MsmPosition |
//! |--------|-------------|-------------|
//! | Cost model | Operation-level (static) | Transition-level (data-dependent) |
//! | Extra state | None | `last_query_value: f64` |
//! | Use case | Weighted Levenshtein | MSM time series metric |
//!
//! The `last_query_value` field is essential for computing the C(a, b, c) function
//! in MSM, where the cost depends on three values: the current value, the previous
//! query value, and the target value.

use std::cmp::Ordering;
use std::fmt;

/// A position in the MSM automaton with value tracking.
///
/// Unlike `PositionF64` which only tracks position and cost, `MsmPosition` also
/// tracks the last query value consumed, which is needed for the C() function:
///
/// ```text
/// C(a, b, c) = c_const           if b ≤ a ≤ c OR b ≥ a ≥ c
///            = c_const + min(|a-b|, |a-c|)  otherwise
/// ```
///
/// Where:
/// - `a` is the value being inserted/removed
/// - `b` is the previous query value (`last_query_value`)
/// - `c` is the target series value
#[derive(Clone, Copy)]
pub struct MsmPosition {
    /// Index in the query series (x).
    /// Represents how many query elements have been consumed.
    pub query_index: usize,

    /// Index in the target series (y).
    /// Represents how many target elements have been matched.
    pub target_index: usize,

    /// Accumulated cost so far.
    pub accumulated_cost: f64,

    /// The last query value consumed (x_{query_index-1}).
    /// Used for computing C() function in merge-like transitions.
    /// For the initial position, this is typically set to the first query value
    /// or a sentinel value.
    pub last_query_value: f64,

    /// The last target value consumed (y_{target_index-1}).
    /// Used for computing C() function in split-like transitions.
    pub last_target_value: f64,

    /// Whether this position represents a "special" state.
    /// In MSM, special positions may track pending merge/split operations.
    pub is_special: bool,
}

impl MsmPosition {
    /// Create a new MSM position.
    #[inline]
    pub fn new(
        query_index: usize,
        target_index: usize,
        accumulated_cost: f64,
        last_query_value: f64,
        last_target_value: f64,
    ) -> Self {
        Self {
            query_index,
            target_index,
            accumulated_cost,
            last_query_value,
            last_target_value,
            is_special: false,
        }
    }

    /// Create an initial position at the start of both series.
    #[inline]
    pub fn initial(first_query_value: f64, first_target_value: f64) -> Self {
        Self {
            query_index: 0,
            target_index: 0,
            accumulated_cost: 0.0,
            last_query_value: first_query_value,
            last_target_value: first_target_value,
            is_special: false,
        }
    }

    /// Create a position with explicit special flag.
    #[inline]
    pub fn with_special(
        query_index: usize,
        target_index: usize,
        accumulated_cost: f64,
        last_query_value: f64,
        last_target_value: f64,
        is_special: bool,
    ) -> Self {
        Self {
            query_index,
            target_index,
            accumulated_cost,
            last_query_value,
            last_target_value,
            is_special,
        }
    }

    /// Check if this position can reach acceptance within the given cost threshold.
    ///
    /// For MSM, acceptance means we've consumed all of both series.
    #[inline]
    pub fn can_reach_acceptance(
        &self,
        query_length: usize,
        target_length: usize,
        max_cost: f64,
        c_const: f64,
    ) -> bool {
        // Remaining elements in each series
        let remaining_query = query_length.saturating_sub(self.query_index);
        let remaining_target = target_length.saturating_sub(self.target_index);

        // Lower bound on remaining cost:
        // - If we have extra query elements, we need at least |diff| * c operations
        // - If we have extra target elements, we need at least |diff| * c operations
        let length_diff = remaining_query.abs_diff(remaining_target);
        let min_remaining_cost = length_diff as f64 * c_const;

        self.accumulated_cost + min_remaining_cost <= max_cost + 1e-9
    }

    /// Check if this position has reached final state (both series consumed).
    #[inline]
    pub fn is_final(&self, query_length: usize, target_length: usize) -> bool {
        self.query_index >= query_length && self.target_index >= target_length
    }

    /// Compute distance to the "diagonal" (equal progress in both series).
    ///
    /// This is useful for subsumption checks and pruning.
    #[inline]
    pub fn diagonal_distance(&self) -> i64 {
        self.query_index as i64 - self.target_index as i64
    }
}

impl fmt::Debug for MsmPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MsmPos({},{},cost={:.3},qv={:.2},tv={:.2}{})",
            self.query_index,
            self.target_index,
            self.accumulated_cost,
            self.last_query_value,
            self.last_target_value,
            if self.is_special { ",special" } else { "" }
        )
    }
}

impl fmt::Display for MsmPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({},{})→{:.3}",
            self.query_index, self.target_index, self.accumulated_cost
        )
    }
}

impl PartialEq for MsmPosition {
    fn eq(&self, other: &Self) -> bool {
        self.query_index == other.query_index
            && self.target_index == other.target_index
            && self.is_special == other.is_special
            && (self.accumulated_cost - other.accumulated_cost).abs() < 1e-9
            && (self.last_query_value - other.last_query_value).abs() < 1e-9
            && (self.last_target_value - other.last_target_value).abs() < 1e-9
    }
}

impl Eq for MsmPosition {}

/// Ordering for positions: primarily by accumulated cost, then by indices.
///
/// This ordering is useful for priority queue-based search algorithms.
impl PartialOrd for MsmPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MsmPosition {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: lower cost is better (comes first)
        match self
            .accumulated_cost
            .partial_cmp(&other.accumulated_cost)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Secondary: higher index progress is better (more consumed)
        match (self.query_index + self.target_index)
            .cmp(&(other.query_index + other.target_index))
            .reverse()
        {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Tertiary: prefer non-special positions
        match self.is_special.cmp(&other.is_special) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Final: arbitrary but consistent ordering by indices
        (self.query_index, self.target_index).cmp(&(other.query_index, other.target_index))
    }
}

/// Check if position `a` subsumes position `b` (makes `b` redundant).
///
/// In MSM, position A subsumes position B if:
/// 1. A is at the same or further progress (query_index, target_index)
/// 2. A has lower or equal accumulated cost
/// 3. A can reach any state that B can reach with equal or lower cost
///
/// This is more complex than Levenshtein subsumption because the C() function
/// depends on the last values, so positions with different last values may
/// not be comparable.
///
/// # Simplification
///
/// For efficiency, we use a conservative subsumption that only considers
/// positions at the exact same (query_index, target_index) location with
/// the same special flag and similar last values:
#[inline]
pub fn msm_subsumes(a: &MsmPosition, b: &MsmPosition, epsilon: f64) -> bool {
    // Same indices and special flag
    if a.query_index != b.query_index
        || a.target_index != b.target_index
        || a.is_special != b.is_special
    {
        return false;
    }

    // A must have lower or equal cost
    if a.accumulated_cost > b.accumulated_cost + epsilon {
        return false;
    }

    // Last values should be close (otherwise C() function behavior differs)
    let query_value_close = (a.last_query_value - b.last_query_value).abs() < epsilon;
    let target_value_close = (a.last_target_value - b.last_target_value).abs() < epsilon;

    query_value_close && target_value_close
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_position_creation() {
        let pos = MsmPosition::new(2, 3, 1.5, 2.0, 3.0);
        assert_eq!(pos.query_index, 2);
        assert_eq!(pos.target_index, 3);
        assert!((pos.accumulated_cost - 1.5).abs() < EPSILON);
        assert!((pos.last_query_value - 2.0).abs() < EPSILON);
        assert!((pos.last_target_value - 3.0).abs() < EPSILON);
        assert!(!pos.is_special);
    }

    #[test]
    fn test_initial_position() {
        let pos = MsmPosition::initial(1.0, 2.0);
        assert_eq!(pos.query_index, 0);
        assert_eq!(pos.target_index, 0);
        assert!((pos.accumulated_cost - 0.0).abs() < EPSILON);
        assert!((pos.last_query_value - 1.0).abs() < EPSILON);
        assert!((pos.last_target_value - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_is_final() {
        let pos = MsmPosition::new(3, 4, 2.0, 1.0, 2.0);
        assert!(pos.is_final(3, 4));
        assert!(pos.is_final(2, 3)); // past the end
        assert!(!pos.is_final(4, 4));
        assert!(!pos.is_final(3, 5));
    }

    #[test]
    fn test_can_reach_acceptance() {
        let pos = MsmPosition::new(2, 2, 1.0, 1.0, 2.0);

        // At both sequence ends, cost 1.0 <= 2.0.
        assert!(pos.can_reach_acceptance(2, 2, 2.0, 1.0));

        // 1 extra query element, needs at least 1.0 more cost
        // Total would be 2.0, which is <= 2.0
        assert!(pos.can_reach_acceptance(3, 2, 2.0, 1.0));

        // 2 extra query elements, needs at least 2.0 more cost
        // Total would be 3.0, which exceeds 2.0
        assert!(!pos.can_reach_acceptance(4, 2, 2.0, 1.0));
    }

    #[test]
    fn test_diagonal_distance() {
        let pos1 = MsmPosition::new(3, 3, 0.0, 0.0, 0.0);
        assert_eq!(pos1.diagonal_distance(), 0);

        let pos2 = MsmPosition::new(5, 3, 0.0, 0.0, 0.0);
        assert_eq!(pos2.diagonal_distance(), 2);

        let pos3 = MsmPosition::new(2, 5, 0.0, 0.0, 0.0);
        assert_eq!(pos3.diagonal_distance(), -3);
    }

    #[test]
    fn test_subsumption() {
        // Same position, A has lower cost
        let a = MsmPosition::new(2, 2, 1.0, 3.0, 4.0);
        let b = MsmPosition::new(2, 2, 2.0, 3.0, 4.0);
        assert!(msm_subsumes(&a, &b, EPSILON));
        assert!(!msm_subsumes(&b, &a, EPSILON));

        // Same cost
        let c = MsmPosition::new(2, 2, 1.0, 3.0, 4.0);
        assert!(msm_subsumes(&a, &c, EPSILON));
        assert!(msm_subsumes(&c, &a, EPSILON));

        // Different indices - no subsumption
        let d = MsmPosition::new(3, 2, 0.5, 3.0, 4.0);
        assert!(!msm_subsumes(&a, &d, EPSILON));
        assert!(!msm_subsumes(&d, &a, EPSILON));

        // Different last values - no subsumption (C() function would behave differently)
        let e = MsmPosition::new(2, 2, 1.0, 5.0, 4.0);
        assert!(!msm_subsumes(&a, &e, EPSILON));
    }

    #[test]
    fn test_ordering() {
        let pos1 = MsmPosition::new(2, 2, 1.0, 0.0, 0.0);
        let pos2 = MsmPosition::new(2, 2, 2.0, 0.0, 0.0);
        let pos3 = MsmPosition::new(3, 3, 1.0, 0.0, 0.0);

        // Lower cost comes first
        assert!(pos1 < pos2);

        // Same cost, more progress comes first
        assert!(pos3 < pos1);
    }

    #[test]
    fn test_debug_format() {
        let pos = MsmPosition::with_special(2, 3, 1.5, 2.0, 3.0, true);
        let debug = format!("{:?}", pos);
        assert!(debug.contains("MsmPos"));
        assert!(debug.contains("2,3"));
        assert!(debug.contains("1.5"));
        assert!(debug.contains("special"));
    }

    #[test]
    fn test_display_format() {
        let pos = MsmPosition::new(2, 3, 1.5, 0.0, 0.0);
        let display = format!("{}", pos);
        assert_eq!(display, "(2,3)→1.500");
    }
}
