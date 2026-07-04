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
use std::hash::{Hash, Hasher};

#[inline]
fn saturating_usize_difference_i64(left: usize, right: usize) -> i64 {
    if left >= right {
        i64::try_from(left - right).unwrap_or(i64::MAX)
    } else {
        match i64::try_from(right - left) {
            Ok(diff) => -diff,
            Err(_) => i64::MIN,
        }
    }
}

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
        saturating_usize_difference_i64(self.query_index, self.target_index)
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
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for MsmPosition {}

impl Hash for MsmPosition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query_index.hash(state);
        self.target_index.hash(state);
        self.accumulated_cost.to_bits().hash(state);
        self.last_query_value.to_bits().hash(state);
        self.last_target_value.to_bits().hash(state);
        self.is_special.hash(state);
    }
}

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
        let self_progress = self.query_index.saturating_add(self.target_index);
        let other_progress = other.query_index.saturating_add(other.target_index);

        // Lower cost comes first; higher progress breaks ties.
        self.accumulated_cost
            .total_cmp(&other.accumulated_cost)
            .then_with(|| other_progress.cmp(&self_progress))
            .then_with(|| self.is_special.cmp(&other.is_special))
            .then_with(|| self.query_index.cmp(&other.query_index))
            .then_with(|| self.target_index.cmp(&other.target_index))
            .then_with(|| self.last_query_value.total_cmp(&other.last_query_value))
            .then_with(|| self.last_target_value.total_cmp(&other.last_target_value))
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const EPSILON: f64 = 1e-9;

    fn position_hash(position: &MsmPosition) -> u64 {
        let mut hasher = DefaultHasher::new();
        position.hash(&mut hasher);
        hasher.finish()
    }

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
    fn test_diagonal_distance_saturates_extreme_indices() {
        let max_i64_usize = usize::try_from(i64::MAX).unwrap_or(usize::MAX);

        let max_positive = MsmPosition::new(max_i64_usize, 0, 0.0, 0.0, 0.0);
        assert_eq!(max_positive.diagonal_distance(), i64::MAX);
        assert_eq!(saturating_usize_difference_i64(usize::MAX, 0), i64::MAX);

        let max_negative = MsmPosition::new(0, max_i64_usize, 0.0, 0.0, 0.0);
        assert_eq!(max_negative.diagonal_distance(), -i64::MAX);

        if let Some(one_past_i64_max) = max_i64_usize.checked_add(1) {
            let min_negative = MsmPosition::new(0, one_past_i64_max, 0.0, 0.0, 0.0);
            assert_eq!(min_negative.diagonal_distance(), i64::MIN);
        }

        assert_eq!(
            saturating_usize_difference_i64(0, usize::MAX),
            if usize::MAX == max_i64_usize {
                -i64::MAX
            } else {
                i64::MIN
            }
        );
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
    fn test_eq_ord_hash_share_total_float_identity() {
        let base = MsmPosition::new(2, 2, 1.0, 3.0, 4.0);
        let near_cost = MsmPosition::new(2, 2, 1.0 + EPSILON / 2.0, 3.0, 4.0);

        assert!(msm_subsumes(&base, &near_cost, EPSILON));
        assert_ne!(base, near_cost);
        assert_ne!(base.cmp(&near_cost), Ordering::Equal);

        let positive_zero = MsmPosition::new(0, 0, 0.0, 1.0, 2.0);
        let negative_zero = MsmPosition::new(0, 0, -0.0, 1.0, 2.0);

        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let different_last_value = MsmPosition::new(2, 2, 1.0, 3.0, 5.0);
        assert_ne!(base, different_last_value);
        assert_ne!(base.cmp(&different_last_value), Ordering::Equal);

        let nan_position = MsmPosition::new(0, 0, f64::NAN, 1.0, 2.0);
        let same_nan_position = nan_position;
        assert_eq!(nan_position, same_nan_position);
        assert_eq!(nan_position.cmp(&same_nan_position), Ordering::Equal);
        assert_eq!(
            position_hash(&nan_position),
            position_hash(&same_nan_position)
        );
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
