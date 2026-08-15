//! Float-weighted automaton state (collection of PositionF64).
//!
//! This module provides `StateF64`, an extension of `State` that works with
//! float-valued accumulated costs instead of integer error counts.
//!
//! # Overview
//!
//! A `StateF64` represents a set of positions in a float-weighted Levenshtein
//! automaton. It maintains positions in sorted order and removes subsumed
//! positions to minimize state space.
//!
//! # SmallVec Optimization
//!
//! Uses SmallVec with inline size of 8 to avoid heap allocations for typical
//! states. This is justified by the **bounded diagonal property** (Theorem 8.2,
//! Mitankin et al., TCS 2011), which mathematically bounds state size.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
//!
//! let mut state = StateF64::new();
//! state.insert(PositionF64::new(0, 0.0), Algorithm::Standard, 5, 1.0);
//! state.insert(PositionF64::new(1, 1.0), Algorithm::Standard, 5, 1.0);
//!
//! assert!(!state.is_empty());
//! assert!(state.min_distance().expect("test fixture: min_distance on non-empty state") < 0.0001); // ~0.0
//! ```

use super::algorithm::Algorithm;
use super::position_f64::PositionF64;
use smallvec::SmallVec;

/// Epsilon for float comparisons in state operations.
const EPSILON: f64 = 1e-9;

/// A state in the float-weighted Levenshtein automaton.
///
/// A state is a collection of `PositionF64` values, maintained in sorted order.
/// Duplicate and subsumed positions are automatically removed to minimize
/// state space.
///
/// # SmallVec Optimization
///
/// Uses SmallVec with inline size of 8 to avoid heap allocations for typical
/// states. The bounded diagonal property (Theorem 8.2) guarantees most states
/// have 2-5 positions, with 8 being a safe upper bound.
///
/// # Thread Safety
///
/// `StateF64` is `Send` and `Sync` when the underlying `PositionF64` is,
/// allowing use in parallel algorithms.
#[derive(Debug, Clone, PartialEq)]
pub struct StateF64 {
    /// Positions in this state, maintained in sorted order.
    /// SmallVec avoids heap allocation for states with ≤ 8 positions.
    positions: SmallVec<[PositionF64; 8]>,
}

impl StateF64 {
    /// Create a new empty state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::StateF64;
    ///
    /// let state = StateF64::new();
    /// assert!(state.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            positions: SmallVec::new(),
        }
    }

    /// Create a state with a single position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64};
    ///
    /// let pos = PositionF64::new(0, 0.0);
    /// let state = StateF64::single(pos);
    /// assert_eq!(state.len(), 1);
    /// ```
    pub fn single(position: PositionF64) -> Self {
        let mut positions = SmallVec::new();
        positions.push(position);
        Self { positions }
    }

    /// Create the initial state for a query.
    ///
    /// The initial state contains a single position at index 0 with cost 0.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::StateF64;
    ///
    /// let state = StateF64::initial();
    /// assert_eq!(state.len(), 1);
    /// ```
    pub fn initial() -> Self {
        Self::single(PositionF64::initial())
    }

    /// Create a state from a vector of positions.
    ///
    /// Positions will be sorted and deduplicated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64};
    ///
    /// let positions = vec![
    ///     PositionF64::new(1, 1.0),
    ///     PositionF64::new(0, 0.0),
    /// ];
    /// let state = StateF64::from_positions(positions);
    /// assert_eq!(state.len(), 2);
    /// ```
    pub fn from_positions(mut positions: Vec<PositionF64>) -> Self {
        positions.sort();
        positions.dedup_by(|a, b| a.approx_eq(b));
        Self {
            positions: SmallVec::from_vec(positions),
        }
    }

    /// Add a position to this state with online subsumption checking.
    ///
    /// This uses an "online" approach that checks subsumption during insertion,
    /// providing O(1) best case when the position is already subsumed.
    ///
    /// # Subsumption
    ///
    /// - If `position` is subsumed by an existing position, it's not added
    /// - Existing positions subsumed by `position` are removed
    /// - Maintains sorted order for efficient iteration
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state = StateF64::new();
    /// state.insert(PositionF64::new(5, 2.0), Algorithm::Standard, 10, 1.0);
    ///
    /// // This position is subsumed by (5, 2.0) because
    /// // |5-5| = 0 <= (3.0-2.0) = 1.0
    /// state.insert(PositionF64::new(5, 3.0), Algorithm::Standard, 10, 1.0);
    ///
    /// assert_eq!(state.len(), 1); // Only (5, 2.0) remains
    /// ```
    pub fn insert(
        &mut self,
        position: PositionF64,
        algorithm: Algorithm,
        query_length: usize,
        max_index_op_cost: f64,
    ) -> bool {
        // The return value is a membership-change witness, not a length-change
        // witness: a retained representative can remove several subsumed ones.
        // Check if this position is subsumed by an existing one
        for existing in &self.positions {
            if existing.approx_eq(&position)
                || existing.subsumes(&position, algorithm, query_length, max_index_op_cost)
            {
                return false; // Already covered by existing position
            }
        }

        // Remove any positions that this new position subsumes
        self.positions
            .retain(|p| !position.subsumes(p, algorithm, query_length, max_index_op_cost));

        // Insert in sorted position
        let insert_pos = self
            .positions
            .binary_search(&position)
            .unwrap_or_else(|pos| pos);
        self.positions.insert(insert_pos, position);
        true
    }

    /// Merge another state into this one.
    ///
    /// All positions from `other` are inserted with subsumption checking.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state1 = StateF64::single(PositionF64::new(0, 0.0));
    /// // Note: (0,0) would subsume (1,1) since |0-1|=1 <= 1.0-0.0=1
    /// // Use a position that won't be subsumed
    /// let state2 = StateF64::single(PositionF64::new(3, 0.5));
    ///
    /// state1.merge(&state2, Algorithm::Standard, 10, 1.0);
    /// assert_eq!(state1.len(), 2);
    /// ```
    pub fn merge(
        &mut self,
        other: &StateF64,
        algorithm: Algorithm,
        query_length: usize,
        max_index_op_cost: f64,
    ) {
        for position in &other.positions {
            self.insert(*position, algorithm, query_length, max_index_op_cost);
        }
    }

    /// Get the head (first) position.
    ///
    /// Since positions are sorted, this is the position with the lowest
    /// term_index and accumulated_cost.
    pub fn head(&self) -> Option<&PositionF64> {
        self.positions.first()
    }

    /// Get all positions.
    #[inline(always)]
    pub fn positions(&self) -> &[PositionF64] {
        &self.positions
    }

    /// Check if this state is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get the number of positions.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Iterate over positions.
    pub fn iter(&self) -> impl Iterator<Item = &PositionF64> {
        self.positions.iter()
    }

    /// Clear all positions from this state.
    ///
    /// Keeps the underlying allocation for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Copy all positions from another state.
    ///
    /// Clears this state and copies all positions from the source.
    #[inline]
    pub fn copy_from(&mut self, other: &StateF64) {
        self.positions.clear();
        self.positions.reserve(other.positions.len());
        self.positions.extend_from_slice(&other.positions);
    }

    /// Get the minimum accumulated cost in this state.
    ///
    /// Returns the smallest `accumulated_cost` among all positions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state = StateF64::new();
    /// state.insert(PositionF64::new(3, 2.5), Algorithm::Standard, 10, 1.0);
    /// state.insert(PositionF64::new(4, 1.5), Algorithm::Standard, 10, 1.0);
    ///
    /// let min = state.min_distance().expect("test fixture: min_distance on non-empty state");
    /// assert!((min - 1.5).abs() < 1e-9);
    /// ```
    #[inline]
    pub fn min_distance(&self) -> Option<f64> {
        self.positions.first().map(|first| {
            // Fast path: single position
            if self.positions.len() == 1 {
                return first.accumulated_cost;
            }

            // Find minimum across all positions
            self.positions
                .iter()
                .map(|p| p.accumulated_cost)
                .fold(f64::INFINITY, f64::min)
        })
    }

    /// Infer the edit distance for a final state.
    ///
    /// For a final state (at end of dictionary term), computes the distance
    /// based on remaining characters in query term.
    ///
    /// # Formula
    ///
    /// For each non-special position:
    /// `distance = accumulated_cost + (query_length - term_index) * deletion_cost`
    ///
    /// The `query_length - term_index` characters left in the query term once a
    /// dictionary term ends must each be deleted to reach acceptance, so they are
    /// charged at the configured `deletion_cost` rather than a hard-coded `1.0`.
    /// Passing `deletion_cost = 1.0` recovers the classic unit-cost edit distance.
    ///
    /// Returns the minimum such distance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state = StateF64::new();
    /// state.insert(PositionF64::new(3, 1.0), Algorithm::Standard, 7, 1.0);
    ///
    /// // Position at index 3 with cost 1.0, query length 7, unit deletion cost:
    /// // Distance = 1.0 + (7 - 3) * 1.0 = 5.0
    /// let dist = state.infer_distance(7, 1.0);
    /// assert!((dist.expect("doc example: distance available") - 5.0).abs() < 1e-9);
    /// ```
    #[inline]
    pub fn infer_distance(&self, query_length: usize, deletion_cost: f64) -> Option<f64> {
        // A valid accepting position has consumed at most `query_length` query
        // characters. Positions with `term_index > query_length` are transition
        // artifacts — a match/substitution generated one past the query end when the
        // look-ahead window is padded beyond the query term — and must NOT be treated
        // as acceptances: doing so previously let their spurious (often cheaper)
        // substitution cost be reported instead of the true trailing-insertion
        // distance. Remaining query characters are then deletions, each charged at
        // `deletion_cost` (unit cost recovers the classic edit distance).
        self.positions
            .iter()
            .filter(|p| !p.is_special && p.term_index <= query_length)
            .map(|p| {
                let remaining = (query_length - p.term_index) as f64;
                p.accumulated_cost + remaining * deletion_cost
            })
            .fold(None, |acc, dist| match acc {
                None => Some(dist),
                Some(min) => Some(min.min(dist)),
            })
    }

    /// Infer the edit distance for prefix matching.
    ///
    /// For prefix matching, we only care if we've consumed the entire query.
    /// Returns the minimum cost among positions that have consumed >= query_length
    /// characters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state = StateF64::new();
    /// state.insert(PositionF64::new(5, 1.5), Algorithm::Standard, 5, 1.0);
    /// state.insert(PositionF64::new(3, 0.5), Algorithm::Standard, 5, 1.0);
    ///
    /// // Only position at index 5 (>= query_length 5) qualifies
    /// let dist = state.infer_prefix_distance(5);
    /// assert!((dist.expect("doc example: distance available") - 1.5).abs() < 1e-9);
    /// ```
    #[inline]
    pub fn infer_prefix_distance(&self, query_length: usize) -> Option<f64> {
        // Fast path: single position
        if self.positions.len() == 1 {
            let p = &self.positions[0];
            return if p.term_index >= query_length {
                Some(p.accumulated_cost)
            } else {
                None
            };
        }

        // General case: find minimum among positions that consumed the full query
        self.positions
            .iter()
            .filter(|p| p.term_index >= query_length)
            .map(|p| p.accumulated_cost)
            .fold(None, |acc, cost| match acc {
                None => Some(cost),
                Some(min) => Some(min.min(cost)),
            })
    }

    /// Check if all positions have cost exceeding the threshold.
    ///
    /// Useful for early pruning during traversal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::{StateF64, PositionF64, Algorithm};
    ///
    /// let mut state = StateF64::new();
    /// state.insert(PositionF64::new(0, 2.5), Algorithm::Standard, 5, 1.0);
    /// state.insert(PositionF64::new(1, 3.0), Algorithm::Standard, 5, 1.0);
    ///
    /// assert!(state.all_exceed_threshold(2.4));
    /// assert!(!state.all_exceed_threshold(2.5));
    /// ```
    #[inline]
    pub fn all_exceed_threshold(&self, threshold: f64) -> bool {
        self.positions
            .iter()
            .all(|p| p.accumulated_cost > threshold + EPSILON)
    }
}

impl Default for StateF64 {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<PositionF64> for StateF64 {
    fn from_iter<T: IntoIterator<Item = PositionF64>>(iter: T) -> Self {
        Self::from_positions(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TEST_EPSILON
    }

    #[test]
    fn test_state_creation() {
        let state = StateF64::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_state_single() {
        let pos = PositionF64::new(3, 1.5);
        let state = StateF64::single(pos);
        assert_eq!(state.len(), 1);
        assert!(state
            .head()
            .expect("test fixture: head on non-empty state")
            .approx_eq(&pos));
    }

    #[test]
    fn test_state_initial() {
        let state = StateF64::initial();
        assert_eq!(state.len(), 1);
        let head = state.head().expect("test fixture: head on non-empty state");
        assert_eq!(head.term_index, 0);
        assert!(approx_eq(head.accumulated_cost, 0.0));
    }

    #[test]
    fn insertion_reports_retained_position_when_state_shrinks() {
        let mut state =
            StateF64::from_positions(vec![PositionF64::new(0, 1.0), PositionF64::new(2, 1.0)]);

        assert!(state.insert(PositionF64::new(1, 0.0), Algorithm::Standard, 3, 1.0,));
        assert_eq!(state.positions(), &[PositionF64::new(1, 0.0)]);

        assert!(!state.insert(PositionF64::new(1, 1.0), Algorithm::Standard, 3, 1.0,));
    }

    #[test]
    fn merge_split_insertion_rejects_approximate_duplicates() {
        let mut state = StateF64::new();
        let position = PositionF64::new(1, 1.0);
        assert!(state.insert(position, Algorithm::MergeAndSplit, 3, 1.0));
        assert!(!state.insert(
            PositionF64::new(1, 1.0 + 1e-12),
            Algorithm::MergeAndSplit,
            3,
            1.0,
        ));
        assert_eq!(state.len(), 1);
        assert!(state.positions()[0].approx_eq(&position));
    }

    #[test]
    fn test_state_from_iter_sorts_and_deduplicates() {
        let state: StateF64 = vec![
            PositionF64::new(3, 2.0),
            PositionF64::new(1, 0.5),
            PositionF64::new(3, 2.0),
        ]
        .into_iter()
        .collect();

        assert_eq!(state.len(), 2);
        assert_eq!(state.positions()[0].term_index, 1);
        assert!(approx_eq(state.positions()[0].accumulated_cost, 0.5));
        assert_eq!(state.positions()[1].term_index, 3);
        assert!(approx_eq(state.positions()[1].accumulated_cost, 2.0));
    }

    #[test]
    fn test_insert_maintains_order() {
        let mut state = StateF64::new();
        let query_length = 10;

        state.insert(
            PositionF64::new(3, 2.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(1, 1.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(2, 1.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        let positions: Vec<_> = state.positions().to_vec();
        assert_eq!(positions[0].term_index, 1);
        assert_eq!(positions[1].term_index, 2);
        assert_eq!(positions[2].term_index, 3);
    }

    #[test]
    fn test_subsumption_removes_positions() {
        let mut state = StateF64::new();
        let query_length = 10;

        // Insert (5, 3.0) first
        state.insert(
            PositionF64::new(5, 3.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        assert_eq!(state.len(), 1);

        // Insert (5, 2.0) which subsumes (5, 3.0)
        // |5-5| = 0 <= (3.0-2.0) = 1.0 ✓
        state.insert(
            PositionF64::new(5, 2.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        assert_eq!(state.len(), 1);

        // Verify (5, 2.0) is in the state
        let pos = state.head().expect("test fixture: head on non-empty state");
        assert_eq!(pos.term_index, 5);
        assert!(approx_eq(pos.accumulated_cost, 2.0));
    }

    #[test]
    fn test_position_subsumed_on_insert() {
        let mut state = StateF64::new();
        let query_length = 10;

        // Insert (5, 2.0) first
        state.insert(
            PositionF64::new(5, 2.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        // Try to insert (5, 3.0) which is subsumed by (5, 2.0)
        state.insert(
            PositionF64::new(5, 3.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        assert_eq!(state.len(), 1);
        assert!(approx_eq(
            state
                .head()
                .expect("test fixture: head on non-empty state")
                .accumulated_cost,
            2.0
        ));
    }

    #[test]
    fn test_min_distance() {
        let mut state = StateF64::new();
        let query_length = 10;

        state.insert(
            PositionF64::new(3, 2.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(4, 1.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(5, 3.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        assert!(approx_eq(
            state
                .min_distance()
                .expect("test fixture: min_distance on non-empty state"),
            1.5
        ));
    }

    #[test]
    fn test_infer_distance() {
        let mut state = StateF64::new();
        let query_length = 7;

        state.insert(
            PositionF64::new(3, 1.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(5, 2.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        // Position (3, 1.0): 1.0 + (7-3) * 1.0 = 5.0
        // Position (5, 2.0): 2.0 + (7-5) * 1.0 = 4.0
        let dist = state
            .infer_distance(query_length, 1.0)
            .expect("test fixture: infer_distance on non-empty state");
        assert!(approx_eq(dist, 4.0));
    }

    #[test]
    fn test_infer_prefix_distance() {
        let mut state = StateF64::new();
        let query_length = 5;

        state.insert(
            PositionF64::new(5, 1.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(3, 0.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        // Only (5, 1.5) qualifies (term_index >= query_length)
        let dist = state
            .infer_prefix_distance(query_length)
            .expect("test fixture: infer_prefix_distance on qualifying state");
        assert!(approx_eq(dist, 1.5));
    }

    #[test]
    fn test_all_exceed_threshold() {
        let mut state = StateF64::new();
        let query_length = 10;

        state.insert(
            PositionF64::new(0, 2.5),
            Algorithm::Standard,
            query_length,
            1.0,
        );
        state.insert(
            PositionF64::new(1, 3.0),
            Algorithm::Standard,
            query_length,
            1.0,
        );

        assert!(state.all_exceed_threshold(2.4));
        assert!(!state.all_exceed_threshold(2.5));
        assert!(!state.all_exceed_threshold(3.0));
    }

    #[test]
    fn test_merge() {
        let mut state1 = StateF64::single(PositionF64::new(0, 0.0));
        // Use positions that won't be subsumed: (0,0) subsumes (1,1) because |0-1|=1 <= 1.0-0.0=1
        // But (0,0) does NOT subsume (3, 0.5) because |0-3|=3 > 0.5-0.0=0.5
        let state2 = StateF64::single(PositionF64::new(3, 0.5));

        state1.merge(&state2, Algorithm::Standard, 10, 1.0);
        assert_eq!(state1.len(), 2);
    }

    #[test]
    fn test_clear_and_copy() {
        let mut state1 = StateF64::single(PositionF64::new(0, 0.0));
        let state2 = StateF64::single(PositionF64::new(5, 2.5));

        state1.copy_from(&state2);
        assert_eq!(state1.len(), 1);
        assert_eq!(
            state1
                .head()
                .expect("test fixture: head on non-empty state")
                .term_index,
            5
        );
    }
}
