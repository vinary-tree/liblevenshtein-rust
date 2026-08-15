//! Automaton state (collection of positions).

use super::algorithm::Algorithm;
use super::position::Position;
use super::variant::{with_variant, AutomatonVariant, TransitionCtx, VariantSpec};
use super::variants::StandardV;
use smallvec::SmallVec;

/// A state in the Levenshtein automaton.
///
/// A state is a collection of positions, maintained in sorted order.
/// Duplicate and subsumed positions are automatically removed to
/// minimize state space.
///
/// # SmallVec Optimization
///
/// Uses SmallVec with inline size of 8 to avoid heap allocations for typical states.
/// This optimization is theoretically justified by the **bounded diagonal property**
/// (Theorem 8.2, Mitankin et al., TCS 2011).
///
/// For Standard Levenshtein with error bound n=2:
/// - Diagonal bound c = 2
/// - Band width = 2c + 1 = 5 diagonals
/// - Typical state size ≤ 8 positions (with subsumption)
///
/// This is not empirical tuning — it's a mathematical guarantee. Profiling confirms
/// most states have 2-5 positions, with the inline capacity rarely exceeded.
///
/// # Theoretical Foundation
///
/// The bounded diagonal property states that for bounded length difference
/// operations, positions in a state cluster around the main diagonal in the
/// dynamic programming matrix. This creates a bounded "band" of active positions,
/// mathematically limiting state size independent of word length.
///
/// ## References
///
/// - Mitankin, P., Mihov, S., Schulz, K.U. (2011). "Deciding Word Neighborhood
///   with Universal Neighborhood Automata". *Theoretical Computer Science*,
///   410(37-39):2339-2358.
/// - See: `docs/research/universal-levenshtein/TCS_2011_LAZY_APPLICABILITY.md`
///   Section 1.1 for detailed analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Positions in this state, maintained in sorted order.
    /// SmallVec avoids heap allocation for states with ≤ 8 positions (the typical case).
    /// Profiling shows most states have 2-5 positions, making this optimization effective.
    positions: SmallVec<[Position; 8]>,
}

impl State {
    /// Create a new empty state
    pub fn new() -> Self {
        Self {
            positions: SmallVec::new(),
        }
    }

    /// Create a state with a single position
    pub fn single(position: Position) -> Self {
        let mut positions = SmallVec::new();
        positions.push(position);
        Self { positions }
    }

    /// Create a state from a vector of positions
    ///
    /// Positions will be sorted and deduplicated
    pub fn from_positions(mut positions: Vec<Position>) -> Self {
        positions.sort();
        positions.dedup();
        Self {
            positions: SmallVec::from_vec(positions),
        }
    }

    /// Add a position to this state with online subsumption checking.
    ///
    /// ## Design: Online vs Batch Subsumption
    ///
    /// This uses an "online" approach that checks subsumption during insertion,
    /// rather than C++'s "batch" approach of inserting all positions then removing
    /// subsumed ones in a separate pass.
    ///
    /// ### Why Online is Superior:
    ///
    /// - **3.3x faster** on average (benchmarked across all algorithms)
    /// - **O(1) best case** with early exit when position is already subsumed
    /// - **O(kn) typical complexity** where k << n due to subsumption pruning
    /// - **Better cache locality** - checks recently inserted positions first
    /// - **Lower memory overhead** - never allocates space for positions that will be discarded
    ///
    /// ### Performance Data:
    ///
    /// | Positions | Online | Batch  | Speedup |
    /// |-----------|--------|--------|---------|
    /// | n=50      | 1.7µs  | 5.6µs  | 3.3x    |
    /// | n=100     | 2.6µs  | 9.2µs  | 3.5x    |
    /// | n=200     | 4.3µs  | 16.5µs | 3.8x    |
    ///
    /// The speedup increases with state size, confirming the O(kn) vs O(n²) advantage.
    ///
    /// See `SUBSUMPTION_OPTIMIZATION_REPORT.md` for detailed analysis.
    ///
    /// ## Implementation
    ///
    /// Maintains sorted order and removes subsumed positions incrementally.
    /// Returns `true` exactly when `position` is retained in the state.
    ///
    /// The return value is intentionally independent of the state's length: inserting one
    /// representative may remove several positions that it subsumes. Fixpoint callers must
    /// use this result instead of comparing lengths before and after insertion.
    pub fn insert(
        &mut self,
        position: Position,
        algorithm: Algorithm,
        query_length: usize,
    ) -> bool {
        let ctx = TransitionCtx::unit(query_length, 0, false);
        with_variant!(VariantSpec::from(algorithm), |V| {
            self.insert_with::<V>(position, &ctx)
        })
    }

    /// Insert using a compile-time automaton variant.
    ///
    /// The runtime algorithm dispatch happens in the caller; the hot
    /// subsumption loop therefore contains only the selected variant's rule.
    #[inline]
    pub(crate) fn insert_with<V: AutomatonVariant>(
        &mut self,
        position: Position,
        ctx: &TransitionCtx<V::Params>,
    ) -> bool {
        crate::causal_perf::record_state_insert_attempts(1);
        // Check if this position is subsumed by an existing one
        for existing in &self.positions {
            // Exact identity is independent of algorithmic dominance. In
            // particular, MergeAndSplit dominance is intentionally strict and
            // therefore irreflexive, but a canonical state must still reject
            // duplicate representatives.
            if existing == &position {
                return false; // Already covered by existing position
            }
            crate::causal_perf::record_subsumption_checks(1);
            if V::subsumes(existing, &position, ctx) {
                return false;
            }
        }

        // Remove any positions that this new position subsumes
        self.positions.retain(|p| {
            crate::causal_perf::record_subsumption_checks(1);
            !V::subsumes(&position, p, ctx)
        });

        // Insert in sorted position
        let insert_pos = self
            .positions
            .binary_search(&position)
            .unwrap_or_else(|pos| pos);
        self.positions.insert(insert_pos, position);
        crate::causal_perf::record_state_insert_retained(1);
        true
    }

    /// Merge another state into this one
    pub fn merge(&mut self, other: &State, algorithm: Algorithm, query_length: usize) {
        let ctx = TransitionCtx::unit(query_length, 0, false);
        with_variant!(VariantSpec::from(algorithm), |V| {
            self.merge_with::<V>(other, &ctx)
        });
    }

    #[inline]
    pub(crate) fn merge_with<V: AutomatonVariant>(
        &mut self,
        other: &State,
        ctx: &TransitionCtx<V::Params>,
    ) {
        for position in &other.positions {
            self.insert_with::<V>(*position, ctx);
        }
    }

    /// Get the head (first) position
    pub fn head(&self) -> Option<&Position> {
        self.positions.first()
    }

    /// Get all positions
    #[inline(always)]
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Check if this state is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get the number of positions
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Iterate over positions
    pub fn iter(&self) -> impl Iterator<Item = &Position> {
        self.positions.iter()
    }

    /// Clear all positions from this state.
    ///
    /// This keeps the underlying Vec allocation, making it efficient for reuse
    /// in a StatePool. After clearing, the state will be empty but retain its
    /// capacity.
    ///
    /// # Performance
    ///
    /// - Time: O(1) - just sets Vec length to 0
    /// - Memory: Vec capacity is preserved for reuse
    #[inline]
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Copy all positions from another state into this one.
    ///
    /// This clears the current state and then copies all positions from the
    /// source state. The source state is unchanged.
    ///
    /// # Performance
    ///
    /// - Time: O(n) where n is the number of positions in source
    /// - Memory: Reuses this state's Vec allocation if capacity is sufficient
    /// - Position is Copy, so this is a fast memcpy of the positions
    #[inline]
    pub fn copy_from(&mut self, other: &State) {
        crate::causal_perf::record_state_copy_calls(1);
        crate::causal_perf::record_state_positions_copied(other.positions.len() as u64);
        crate::causal_perf::record_state_bytes_copied(
            other
                .positions
                .len()
                .saturating_mul(std::mem::size_of::<Position>()) as u64,
        );
        self.positions.clear();
        self.positions.reserve(other.positions.len());
        self.positions.extend_from_slice(&other.positions);
    }

    /// Get the minimum edit distance in this state
    ///
    /// Returns the smallest `num_errors` among all positions
    #[inline]
    pub fn min_distance(&self) -> Option<usize> {
        // Optimization: positions are sorted, and since we maintain subsumption,
        // the first position often has the minimum errors. Check it first.
        self.positions.first().map(|first| {
            // Fast path: if we only have one position, return it immediately
            if self.positions.len() == 1 {
                return first.num_errors;
            }

            // SIMD path: use vectorized horizontal minimum for 4-8 positions
            #[cfg(target_arch = "x86_64")]
            {
                let len = self.positions.len();
                if (4..=8).contains(&len) {
                    let errors: smallvec::SmallVec<[usize; 8]> =
                        self.positions.iter().map(|p| p.num_errors).collect();
                    return super::simd::find_minimum_simd(&errors, len);
                }
            }

            // Scalar fallback for len > 8 or when SIMD unavailable.
            self.positions[1..]
                .iter()
                .fold(first.num_errors, |min_errors, p| {
                    min_errors.min(p.num_errors)
                })
        })
    }

    /// Infer the edit distance for a final state
    ///
    /// For a final state (at end of dictionary term), infer the
    /// distance based on remaining characters in query term
    #[inline]
    pub fn infer_distance(&self, query_length: usize) -> Option<usize> {
        self.infer_distance_with::<StandardV>(query_length, ())
    }

    /// Infer a final distance through a compile-time variant policy.
    ///
    /// The public unit-cost algorithms currently share the same finishing
    /// rule. Keeping that rule behind the variant seam lets later weighted
    /// variants specialize it without reintroducing a branch in the scan.
    #[inline]
    pub(crate) fn infer_distance_with<V: AutomatonVariant>(
        &self,
        query_length: usize,
        params: V::Params,
    ) -> Option<usize> {
        // Fast path: single position (common case)
        if self.positions.len() == 1 {
            return V::finish_cost(&self.positions[0], query_length, params);
        }

        // General case: find minimum across all NON-SPECIAL positions
        // Special positions are intermediate states for transposition/merge/split
        // and should not contribute to the final distance calculation
        self.positions
            .iter()
            .filter_map(|p| V::finish_cost(p, query_length, params))
            .min()
    }

    /// Infer the edit distance for prefix matching
    ///
    /// For prefix matching, we only care if we've consumed the entire query
    /// (allowing the dictionary term to be longer). Returns the minimum number
    /// of errors among positions that have consumed >= query_length characters.
    ///
    /// Returns None if no position has consumed the full query yet.
    #[inline]
    pub fn infer_prefix_distance(&self, query_length: usize) -> Option<usize> {
        // Fast path: single position
        if self.positions.len() == 1 {
            let p = &self.positions[0];
            return if p.term_index >= query_length {
                Some(p.num_errors)
            } else {
                None
            };
        }

        // General case: find minimum among positions that consumed the full query
        self.positions
            .iter()
            .filter(|p| p.term_index >= query_length)
            .map(|p| p.num_errors)
            .min()
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<Position> for State {
    fn from_iter<T: IntoIterator<Item = Position>>(iter: T) -> Self {
        Self::from_positions(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = State::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_state_single_position() {
        let pos = Position::new(3, 1);
        let state = State::single(pos);
        assert_eq!(state.len(), 1);
        assert_eq!(state.head(), Some(&pos));
    }

    #[test]
    fn test_state_from_iter_sorts_and_deduplicates() {
        let state: State = vec![
            Position::new(3, 2),
            Position::new(1, 0),
            Position::new(3, 2),
        ]
        .into_iter()
        .collect();

        assert_eq!(
            state.positions(),
            &[Position::new(1, 0), Position::new(3, 2)]
        );
    }

    #[test]
    fn test_state_insert_maintains_order() {
        let mut state = State::new();
        let max_distance = 3;
        // Insert positions - (3,1) will subsume (2,2) and (4,2) with Standard subsumption
        // (3,1) subsumes (2,2): |3-2|=1 <= (2-1)=1 ✓
        // (3,1) subsumes (4,2): |3-4|=1 <= (2-1)=1 ✓
        state.insert(Position::new(2, 2), Algorithm::Standard, max_distance);
        state.insert(Position::new(3, 1), Algorithm::Standard, max_distance); // This subsumes (2,2)
        state.insert(Position::new(4, 2), Algorithm::Standard, max_distance); // This is subsumed by (3,1)

        let positions: Vec<_> = state.positions().to_vec();
        // Only (3,1) should remain
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], Position::new(3, 1));
    }

    #[test]
    fn insertion_reports_retained_position_when_state_shrinks() {
        let mut state = State::from_positions(vec![Position::new(0, 1), Position::new(2, 1)]);

        // (1, 0) subsumes both existing representatives. The state shrinks from two
        // positions to one, but the new representative was still retained.
        assert!(state.insert(Position::new(1, 0), Algorithm::Standard, 3));
        assert_eq!(state.positions(), &[Position::new(1, 0)]);

        assert!(!state.insert(Position::new(1, 1), Algorithm::Standard, 3));
    }

    #[test]
    fn merge_split_insertion_rejects_exact_duplicates() {
        let mut state = State::new();
        let position = Position::new(1, 1);
        assert!(state.insert(position, Algorithm::MergeAndSplit, 3));
        assert!(!state.insert(position, Algorithm::MergeAndSplit, 3));
        assert_eq!(state.positions(), &[position]);
    }

    #[test]
    fn test_state_subsumption() {
        let mut state = State::new();
        let max_distance = 3;
        state.insert(Position::new(5, 2), Algorithm::Standard, max_distance);
        assert_eq!(state.len(), 1);

        // Try to insert a position that IS subsumed: (5,2) subsumes (4,3) because |5-4|=1 <= (3-2)=1
        state.insert(Position::new(4, 3), Algorithm::Standard, max_distance); // Subsumed by (5,2)
        assert_eq!(state.len(), 1, "(4,3) should be subsumed by (5,2)");

        // Insert a position at SAME index with fewer errors - should subsume
        state.insert(Position::new(5, 1), Algorithm::Standard, max_distance); // Subsumes (5,2) at same position
        assert_eq!(state.len(), 1, "(5,1) should replace (5,2)");

        // Verify (5,1) is in the state
        let pos_at_5 = state
            .positions()
            .iter()
            .find(|p| p.term_index == 5)
            .expect("test fixture: position with term_index 5 was just inserted");
        assert_eq!(pos_at_5.num_errors, 1);
    }

    #[test]
    fn test_state_min_distance() {
        let mut state = State::new();
        let max_distance = 3;
        state.insert(Position::new(3, 2), Algorithm::Standard, max_distance);
        state.insert(Position::new(4, 1), Algorithm::Standard, max_distance);
        state.insert(Position::new(5, 3), Algorithm::Standard, max_distance);

        assert_eq!(state.min_distance(), Some(1));
    }

    #[test]
    fn test_state_infer_distance() {
        let mut state = State::new();
        let max_distance = 3;
        state.insert(Position::new(3, 1), Algorithm::Standard, max_distance); // At position 3 with 1 error
        state.insert(Position::new(4, 2), Algorithm::Standard, max_distance); // At position 4 with 2 errors

        let query_length = 7;
        // Position (3,1): needs 4 more chars = 1+4=5 distance
        // Position (4,2): needs 3 more chars = 2+3=5 distance
        assert_eq!(state.infer_distance(query_length), Some(5));
    }
}
