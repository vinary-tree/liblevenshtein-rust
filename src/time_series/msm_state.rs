//! MSM automaton state containing multiple positions.
//!
//! This module provides `MsmState`, which maintains a collection of `MsmPosition`s
//! representing all reachable states in the MSM automaton at a given point in
//! the traversal.
//!
//! # State Management
//!
//! Unlike the string Levenshtein automaton which has bounded state size (Theorem 8.2),
//! MSM's data-dependent costs can lead to more positions. However, subsumption and
//! cost-based pruning still limit practical state sizes.

use super::msm_position::{msm_subsumes, MsmPosition};
use smallvec::SmallVec;
use std::fmt;

/// Default inline capacity for positions in MsmState.
/// MSM may have more active positions than Levenshtein due to data-dependent costs,
/// but empirically 16 is sufficient for most practical cases.
const MSM_STATE_INLINE_CAPACITY: usize = 16;

/// MSM automaton state: a collection of active positions.
///
/// Each position tracks:
/// - Progress in both query and target series
/// - Accumulated cost
/// - Last consumed values (for C() function)
///
/// The state maintains sorted, non-subsumed positions for efficient transitions.
#[derive(Clone)]
pub struct MsmState {
    /// Active positions, sorted by (query_index, target_index, cost).
    positions: SmallVec<[MsmPosition; MSM_STATE_INLINE_CAPACITY]>,
}

impl MsmState {
    /// Create an empty MSM state.
    #[inline]
    pub fn new() -> Self {
        Self {
            positions: SmallVec::new(),
        }
    }

    /// Create a state with a single initial position.
    #[inline]
    pub fn initial(first_query_value: f64, first_target_value: f64) -> Self {
        let pos = MsmPosition::initial(first_query_value, first_target_value);
        let mut positions = SmallVec::new();
        positions.push(pos);
        Self { positions }
    }

    /// Create a state from a single position.
    #[inline]
    pub fn single(position: MsmPosition) -> Self {
        let mut positions = SmallVec::new();
        positions.push(position);
        Self { positions }
    }

    /// Create a state with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: SmallVec::with_capacity(capacity),
        }
    }

    /// Get the number of active positions.
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if the state has no positions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Clear all positions from the state.
    #[inline]
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Get an iterator over positions.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &MsmPosition> {
        self.positions.iter()
    }

    /// Get the minimum accumulated cost among all positions.
    ///
    /// Returns `None` if the state is empty.
    #[inline]
    pub fn min_cost(&self) -> Option<f64> {
        self.positions
            .iter()
            .map(|p| p.accumulated_cost)
            .min_by(|a, b| a.total_cmp(b))
    }

    /// Get the minimum distance at final positions.
    ///
    /// Only considers positions that have consumed all of both series.
    /// Returns `None` if no position is final.
    #[inline]
    pub fn min_final_distance(&self, query_length: usize, target_length: usize) -> Option<f64> {
        self.positions
            .iter()
            .filter(|p| p.is_final(query_length, target_length))
            .map(|p| p.accumulated_cost)
            .min_by(|a, b| a.total_cmp(b))
    }

    /// Check if any position has reached the final state.
    #[inline]
    pub fn has_final(&self, query_length: usize, target_length: usize) -> bool {
        self.positions
            .iter()
            .any(|p| p.is_final(query_length, target_length))
    }

    /// Insert a position, maintaining subsumption invariant.
    ///
    /// The position is only inserted if:
    /// 1. It's not subsumed by any existing position
    /// 2. It doesn't exceed the cost threshold
    ///
    /// If inserted, any existing positions subsumed by the new one are removed.
    pub fn insert(&mut self, position: MsmPosition, max_cost: f64, epsilon: f64) {
        // Early rejection if cost exceeds threshold
        if position.accumulated_cost > max_cost + epsilon {
            return;
        }

        // Check if new position is subsumed by existing
        for existing in &self.positions {
            if msm_subsumes(existing, &position, epsilon) {
                return;
            }
        }

        // Remove positions subsumed by the new one
        self.positions
            .retain(|existing| !msm_subsumes(&position, existing, epsilon));

        // Insert the new position
        self.positions.push(position);
    }

    /// Insert a position without subsumption checking (for performance).
    ///
    /// Use this only when you know the position won't create duplicates.
    #[inline]
    pub fn insert_unchecked(&mut self, position: MsmPosition) {
        self.positions.push(position);
    }

    /// Remove positions that exceed the cost threshold.
    pub fn prune_by_cost(&mut self, max_cost: f64, epsilon: f64) {
        self.positions
            .retain(|p| p.accumulated_cost <= max_cost + epsilon);
    }

    /// Apply subsumption to remove redundant positions.
    ///
    /// This is a quadratic operation and should be called sparingly,
    /// typically after batch insertions.
    pub fn apply_subsumption(&mut self, epsilon: f64) {
        if self.positions.len() <= 1 {
            return;
        }

        let mut keep = vec![true; self.positions.len()];

        for i in 0..self.positions.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..self.positions.len() {
                if !keep[j] {
                    continue;
                }

                if msm_subsumes(&self.positions[i], &self.positions[j], epsilon) {
                    keep[j] = false;
                } else if msm_subsumes(&self.positions[j], &self.positions[i], epsilon) {
                    keep[i] = false;
                    break;
                }
            }
        }

        let mut write_idx = 0;
        for (read_idx, keep_position) in keep.iter().enumerate().take(self.positions.len()) {
            if *keep_position {
                if write_idx != read_idx {
                    self.positions[write_idx] = self.positions[read_idx];
                }
                write_idx += 1;
            }
        }
        self.positions.truncate(write_idx);
    }

    /// Sort positions by (query_index, target_index, cost).
    ///
    /// Useful for consistent iteration order and debugging.
    pub fn sort(&mut self) {
        self.positions.sort_by(|a, b| {
            (a.query_index, a.target_index)
                .cmp(&(b.query_index, b.target_index))
                .then_with(|| a.accumulated_cost.total_cmp(&b.accumulated_cost))
        });
    }

    /// Get positions as a slice (for testing and debugging).
    #[inline]
    pub fn positions(&self) -> &[MsmPosition] {
        &self.positions
    }

    /// Convert to a vector of positions.
    #[inline]
    pub fn into_positions(self) -> SmallVec<[MsmPosition; MSM_STATE_INLINE_CAPACITY]> {
        self.positions
    }
}

impl Default for MsmState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MsmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MsmState[")?;
        for (i, pos) in self.positions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", pos)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for MsmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, pos) in self.positions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", pos)?;
        }
        write!(f, "]")
    }
}

impl<'a> IntoIterator for &'a MsmState {
    type Item = &'a MsmPosition;
    type IntoIter = std::slice::Iter<'a, MsmPosition>;

    fn into_iter(self) -> Self::IntoIter {
        self.positions.iter()
    }
}

impl IntoIterator for MsmState {
    type Item = MsmPosition;
    type IntoIter = smallvec::IntoIter<[MsmPosition; MSM_STATE_INLINE_CAPACITY]>;

    fn into_iter(self) -> Self::IntoIter {
        self.positions.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_new_state_is_empty() {
        let state = MsmState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
        assert!(state.min_cost().is_none());
    }

    #[test]
    fn test_initial_state() {
        let state = MsmState::initial(1.0, 2.0);
        assert_eq!(state.len(), 1);
        assert!(!state.is_empty());

        let pos = &state.positions()[0];
        assert_eq!(pos.query_index, 0);
        assert_eq!(pos.target_index, 0);
        assert!((pos.accumulated_cost - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_single() {
        let pos = MsmPosition::new(2, 3, 1.5, 2.0, 3.0);
        let state = MsmState::single(pos);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_insert_with_cost_threshold() {
        let mut state = MsmState::new();

        // Insert within threshold
        let pos1 = MsmPosition::new(1, 1, 1.0, 0.0, 0.0);
        state.insert(pos1, 2.0, EPSILON);
        assert_eq!(state.len(), 1);

        // Insert exceeding threshold - should be rejected
        let pos2 = MsmPosition::new(2, 2, 3.0, 0.0, 0.0);
        state.insert(pos2, 2.0, EPSILON);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_insert_with_subsumption() {
        let mut state = MsmState::new();

        // Insert first position
        let pos1 = MsmPosition::new(1, 1, 2.0, 1.0, 2.0);
        state.insert(pos1, 10.0, EPSILON);
        assert_eq!(state.len(), 1);

        // Insert better position at same location - should subsume the first
        let pos2 = MsmPosition::new(1, 1, 1.0, 1.0, 2.0);
        state.insert(pos2, 10.0, EPSILON);
        assert_eq!(state.len(), 1);
        assert!((state.positions()[0].accumulated_cost - 1.0).abs() < EPSILON);

        // Insert worse position - should be rejected
        let pos3 = MsmPosition::new(1, 1, 3.0, 1.0, 2.0);
        state.insert(pos3, 10.0, EPSILON);
        assert_eq!(state.len(), 1);

        // Insert position at different location - should be added
        let pos4 = MsmPosition::new(2, 2, 1.5, 1.0, 2.0);
        state.insert(pos4, 10.0, EPSILON);
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn test_min_cost() {
        let mut state = MsmState::new();

        state.insert_unchecked(MsmPosition::new(1, 1, 3.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(2, 2, 1.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(3, 3, 2.0, 0.0, 0.0));

        assert!((state.min_cost().expect("expected Some min_cost in test") - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cost_ordering_is_total_for_nan() {
        let mut state = MsmState::new();

        state.insert_unchecked(MsmPosition::new(1, 1, f64::NAN, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(1, 1, 1.0, 0.0, 0.0));

        assert_eq!(state.min_cost(), Some(1.0));

        state.sort();
        assert_eq!(state.positions()[0].accumulated_cost, 1.0);
        assert!(state.positions()[1].accumulated_cost.is_nan());
    }

    #[test]
    fn test_min_final_distance() {
        let mut state = MsmState::new();

        // Non-final position
        state.insert_unchecked(MsmPosition::new(2, 3, 1.0, 0.0, 0.0));
        // Final position (at 3, 4)
        state.insert_unchecked(MsmPosition::new(3, 4, 2.5, 0.0, 0.0));
        // Another final position with lower cost
        state.insert_unchecked(MsmPosition::new(3, 4, 2.0, 0.0, 0.0));

        let min_final = state.min_final_distance(3, 4);
        assert!(min_final.is_some());
        assert!((min_final.expect("expected Some min_final in test") - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_has_final() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(2, 3, 1.0, 0.0, 0.0));

        assert!(!state.has_final(3, 4));
        assert!(state.has_final(2, 3));
        assert!(state.has_final(1, 2)); // past the end counts as final
    }

    #[test]
    fn test_prune_by_cost() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(1, 1, 1.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(2, 2, 2.5, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(3, 3, 3.0, 0.0, 0.0));

        assert_eq!(state.len(), 3);

        state.prune_by_cost(2.5, EPSILON);
        assert_eq!(state.len(), 2);

        state.prune_by_cost(1.5, EPSILON);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_apply_subsumption() {
        let mut state = MsmState::new();

        // Add positions with same location but different costs
        state.insert_unchecked(MsmPosition::new(1, 1, 2.0, 1.0, 2.0));
        state.insert_unchecked(MsmPosition::new(1, 1, 1.0, 1.0, 2.0)); // should survive
        state.insert_unchecked(MsmPosition::new(1, 1, 3.0, 1.0, 2.0));

        assert_eq!(state.len(), 3);

        state.apply_subsumption(EPSILON);

        // Only the best position should remain
        assert_eq!(state.len(), 1);
        assert!((state.positions()[0].accumulated_cost - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_sort() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(3, 3, 1.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(1, 1, 2.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(2, 2, 1.5, 0.0, 0.0));

        state.sort();

        assert_eq!(state.positions()[0].query_index, 1);
        assert_eq!(state.positions()[1].query_index, 2);
        assert_eq!(state.positions()[2].query_index, 3);
    }

    #[test]
    fn test_clear() {
        let mut state = MsmState::initial(1.0, 2.0);
        assert!(!state.is_empty());

        state.clear();
        assert!(state.is_empty());
    }

    #[test]
    fn test_iteration() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(1, 1, 1.0, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(2, 2, 2.0, 0.0, 0.0));

        let count = state.iter().count();
        assert_eq!(count, 2);

        let count_ref = (&state).into_iter().count();
        assert_eq!(count_ref, 2);

        let count_owned = state.into_iter().count();
        assert_eq!(count_owned, 2);
    }

    #[test]
    fn test_debug_format() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(1, 2, 1.5, 0.0, 0.0));

        let debug = format!("{:?}", state);
        assert!(debug.contains("MsmState"));
        assert!(debug.contains("MsmPos"));
    }

    #[test]
    fn test_display_format() {
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(1, 2, 1.5, 0.0, 0.0));
        state.insert_unchecked(MsmPosition::new(2, 3, 2.0, 0.0, 0.0));

        let display = format!("{}", state);
        assert!(display.contains("(1,2)"));
        assert!(display.contains("(2,3)"));
    }
}
