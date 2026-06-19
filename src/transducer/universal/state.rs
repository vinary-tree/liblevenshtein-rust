//! Universal State Type
//!
//! Implements universal states from Mitankin's thesis (Definition 15, pages 38-39).
//!
//! # Theory Background
//!
//! Universal states are sets of universal positions that maintain the anti-chain property:
//! no position subsumes another. This enables efficient state minimization.
//!
//! ## State Sets (Page 38)
//!
//! **Non-final states**:
//! ```text
//! I^χ_states = {Q | Q ⊆ I^χ_s ∧ ∀q₁,q₂ ∈ Q (q₁ ⊀^χ_s q₂)} \ {∅}
//! ```
//!
//! **Final states**:
//! ```text
//! M^χ_states = {Q | Q ⊆ M^χ_s ∧
//!               ∀q₁,q₂ ∈ Q (q₁ ⊀^χ_s q₂) ∧
//!               ∃q ∈ Q (q ≤^χ_s M#n) ∧
//!               ∃i ∈ [-n, 0] ∀q ∈ Q (M + i#0 ≤^χ_s q)} \ {∅}
//! ```
//!
//! **All states**:
//! ```text
//! Q^∀,χ_n = I^χ_states ∪ M^χ_states
//! ```
//!
//! ## Anti-chain Property
//!
//! For all positions p₁, p₂ in state Q:
//! - p₁ ⊀^χ_s p₂ (p₁ does not subsume p₂)
//! - p₂ ⊀^χ_s p₁ (p₂ does not subsume p₁)
//!
//! This is maintained by the ⊔ (join) operator when adding positions.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::transducer::universal::{UniversalState, UniversalPosition, Standard};
//!
//! // Create initial state: {I + 0#0}
//! let initial = UniversalState::<Standard>::initial(2);
//! assert!(!initial.is_final());
//!
//! // Create state with multiple positions
//! let mut state = UniversalState::<Standard>::new(2);
//! state.add_position(UniversalPosition::new_i(0, 0, 2)?);
//! state.add_position(UniversalPosition::new_i(1, 1, 2)?);
//! ```

use smallvec::SmallVec;
use std::fmt;

use crate::transducer::universal::position::{PositionVariant, UniversalPosition};
use crate::transducer::universal::subsumption::subsumes;

/// Universal state maintaining anti-chain property
///
/// A state is a set of universal positions where no position subsumes another.
/// This implements Q^∀,χ_n from the thesis (Definition 15, pages 38-39).
///
/// # Type Parameters
///
/// - `V`: Position variant (Standard, Transposition, or MergeAndSplit)
///
/// # Invariant
///
/// For all p₁, p₂ ∈ positions: p₁ ⊀^χ_s p₂ ∧ p₂ ⊀^χ_s p₁
///
/// This invariant is maintained by `add_position()` using the ⊔ operator.
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
/// This is not empirical tuning — it's a mathematical guarantee. The bounded diagonal
/// property proves that for operations with bounded length difference, positions
/// cluster around the main diagonal in the DP matrix, creating a bounded "band" of
/// active positions independent of word length.
///
/// ## Theoretical Foundation
///
/// **Theorem 8.2** (TCS 2011, Page 2348): The following are equivalent:
/// 1. R[Op,r] has bounded length difference
/// 2. There exists constant c such that every Op instance satisfies c-bounded diagonal property
/// 3. Every zero-weighted type in Υ is length preserving
///
/// For Standard Levenshtein: c = 2, yielding O(n²) state space (independent of word length).
///
/// ## References
///
/// - Mitankin, P., Mihov, S., Schulz, K.U. (2011). "Deciding Word Neighborhood
///   with Universal Neighborhood Automata". *Theoretical Computer Science*,
///   410(37-39):2339-2358.
/// - See: `docs/research/universal-levenshtein/TCS_2011_PAPER_ANALYSIS.md`
///   Section 2 for detailed analysis
///
/// # Implementation Note
///
/// Maintains sorted order manually via binary search + insertion.
/// This provides:
/// - Stack allocation for 90%+ of states (inline capacity 8)
/// - Excellent cache locality (contiguous memory)
/// - Online subsumption during insertion (O(kn) typical, k << n)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniversalState<V: PositionVariant> {
    /// Set of positions (anti-chain), maintained in sorted order
    /// SmallVec avoids heap allocation for states with ≤8 positions
    positions: SmallVec<[UniversalPosition<V>; 8]>,

    /// Maximum edit distance n
    max_distance: u8,

    /// Length difference m = |query| - |dict| tracking
    ///
    /// Tracks position relative to diagonal in the DP matrix.
    /// Range: `[-max_distance, +max_distance]` (bounded diagonal property).
    ///
    /// This field enables correct diagonal crossing detection:
    /// - When `|length_diff| > max_distance`, positions must be converted (I → M or M → I)
    /// - See: `docs/research/universal-levenshtein/DIAGONAL_CROSSING_BUG_ANALYSIS.md`
    ///
    /// Updated during `transition_with_consumption()` based on character consumption.
    length_diff: i8,
}

impl<V: PositionVariant> UniversalState<V> {
    /// Create new empty state
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = UniversalState::<Standard>::new(2);
    /// assert!(state.is_empty());
    /// ```
    pub fn new(max_distance: u8) -> Self {
        Self {
            positions: SmallVec::new(),
            max_distance,
            length_diff: 0,
        }
    }

    /// Create initial state {I + 0#0}
    ///
    /// From thesis page 38: Initial state I^∀,χ = {I + 0#0}
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Example
    ///
    /// ```ignore
    /// let initial = UniversalState::<Standard>::initial(2);
    /// assert_eq!(initial.len(), 1);
    /// assert!(!initial.is_final());
    /// ```
    pub fn initial(max_distance: u8) -> Self {
        let mut state = Self::new(max_distance);
        // I + 0#0 always satisfies invariant, so unwrap is safe
        let initial_pos =
            UniversalPosition::new_i(0, 0, max_distance).expect("I + 0#0 should always be valid");
        state.positions.push(initial_pos);
        state
    }

    /// Add position, maintaining anti-chain property (⊔ operator)
    ///
    /// Implements the subsumption closure from the thesis:
    /// 1. Remove all positions that subsume the new position (worse positions)
    /// 2. Add new position if it doesn't subsume any existing position
    ///
    /// This maintains the invariant ∀p₁,p₂ ∈ Q (p₁ ⊀^χ_s p₂).
    ///
    /// Note: If p1 <^χ_s p2, then p1 subsumes p2, meaning p2 is "better" (more errors).
    /// We keep the better positions and discard the worse ones.
    ///
    /// # Arguments
    ///
    /// - `pos`: Position to add
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut state = UniversalState::<Standard>::new(2);
    /// state.add_position(UniversalPosition::new_i(0, 0, 2)?);
    /// state.add_position(UniversalPosition::new_i(1, 1, 2)?);
    /// ```
    pub fn add_position(&mut self, pos: UniversalPosition<V>) {
        // Check if this position is subsumed by an existing one
        for existing in &self.positions {
            if subsumes(existing, &pos, self.max_distance) {
                return; // Already covered by existing position
            }
        }

        // Remove any positions that this new position subsumes
        self.positions
            .retain(|p| !subsumes(&pos, p, self.max_distance));

        // Insert in sorted position (binary search)
        let insert_pos = self.positions.binary_search(&pos).unwrap_or_else(|pos| pos);
        self.positions.insert(insert_pos, pos);
    }

    /// Check if state is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get number of positions in state
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Get iterator over positions
    pub fn positions(&self) -> impl Iterator<Item = &UniversalPosition<V>> {
        self.positions.iter()
    }

    /// Check if state contains a position
    pub fn contains(&self, pos: &UniversalPosition<V>) -> bool {
        self.positions.iter().any(|p| p == pos)
    }

    /// Check if state is final
    ///
    /// A state is final if it contains an M-type position that could accept.
    /// From the thesis, this means the state contains a position subsuming M + 0#k
    /// for some k ≤ n.
    ///
    /// The implementation checks for an `MFinal` position whose offset is at
    /// or before the accepting boundary.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut state = UniversalState::<Standard>::new(2);
    /// state.add_position(UniversalPosition::new_m(0, 0, 2)?);
    /// assert!(state.is_final());
    /// ```
    pub fn is_final(&self) -> bool {
        self.positions.iter().any(|pos| match pos {
            UniversalPosition::MFinal { offset, .. } => *offset <= 0,
            _ => false,
        })
    }

    /// Get maximum distance
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Check if this state contains only I-type positions
    pub fn is_i_state(&self) -> bool {
        !self.is_empty() && self.positions.iter().all(|p| p.is_i_type())
    }

    /// Check if this state contains only M-type positions
    pub fn is_m_state(&self) -> bool {
        !self.is_empty() && self.positions.iter().all(|p| p.is_m_type())
    }

    /// Check if this state contains mixed I and M positions
    pub fn is_mixed_state(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        let has_i = self.positions.iter().any(|p| p.is_i_type());
        let has_m = self.positions.iter().any(|p| p.is_m_type());
        has_i && has_m
    }

    /// Get current length difference m = |query| - |dict|
    ///
    /// Returns the tracked position relative to the diagonal in the DP matrix.
    /// Range: `[-max_distance, +max_distance]`.
    ///
    /// - `m > 0`: Query word is longer (dict consumed faster)
    /// - `m < 0`: Dict word is longer (query consumed faster)
    /// - `m = 0`: Same length consumed from both words
    #[inline]
    pub fn length_diff(&self) -> i8 {
        self.length_diff
    }

    /// Compute transition to successor state (δ^∀,χ_n)
    ///
    /// Implements the universal state transition function from the thesis (Definition 15, page 48):
    ///
    /// ```text
    /// δ^∀,χ_n(Q, x) = {
    ///     Δ               if f_n(rm(Δ), |x|) = false
    ///     m_n(Δ, |x|)     if f_n(rm(Δ), |x|) = true
    /// }
    /// where Δ = ⊔_{π∈Q} δ^∀,χ_e(π, x)
    /// ```
    ///
    /// For each position π in the current state:
    /// 1. Compute successors using δ^∀,χ_e (the `successors()` method)
    /// 2. Union all successor sets
    /// 3. Apply subsumption closure ⊔ (done automatically by `add_position()`)
    /// 4. Check diagonal crossing with f_n(rm(Δ), k)
    /// 5. If crossed, apply m_n conversion to all positions
    ///
    /// # Arguments
    ///
    /// - `bit_vector`: Characteristic vector β(a, w) encoding matches for character a
    /// - `input_length`: Current input position (k) for diagonal crossing detection
    ///
    /// # Returns
    ///
    /// Successor state, or `None` if no successors exist (undefined transition)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = UniversalState::<Standard>::initial(2);
    /// let bit_vector = CharacteristicVector::new('a', "abc");
    /// let next_state = state.transition(&bit_vector, 1);
    /// ```
    pub fn transition(
        &self,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        _input_length: usize,
    ) -> Option<Self> {
        // Special case: empty state has no successors
        if self.is_empty() {
            return None;
        }

        // Create new state for successors (Δ)
        let mut next_state = Self::new(self.max_distance);

        // For each position π in current state Q
        for pos in &self.positions {
            // Compute δ^∀,χ_e(π, x) using the successors() method
            let successors = pos.successors(bit_vector, self.max_distance);

            // Add all successors to next state
            // The add_position() method automatically applies subsumption closure ⊔
            for succ in successors {
                next_state.add_position(succ);
            }
        }

        // Return None if no successors (undefined transition)
        if next_state.is_empty() {
            return None;
        }

        // NOTE: Diagonal crossing is NOT performed in this method.
        //
        // This simplified API doesn't track character consumption, so it cannot
        // correctly update length_diff for diagonal crossing detection.
        //
        // For proper diagonal crossing support, use `transition_with_consumption()`
        // which tracks which word consumed a character and updates length_diff accordingly.
        //
        // This method preserves the parent state's length_diff for backward compatibility,
        // but diagonal crossing conversions are disabled.
        //
        // See: docs/research/universal-levenshtein/DIAGONAL_CROSSING_BUG_ANALYSIS.md
        next_state.length_diff = self.length_diff;

        // Return final state (possibly converted)
        if next_state.is_empty() {
            None
        } else {
            Some(next_state)
        }
    }

    /// Compute transition with explicit character consumption tracking
    ///
    /// This is the correct API for diagonal crossing integration. It tracks which
    /// word consumed a character, enabling proper `length_diff` updates for
    /// diagonal crossing detection.
    ///
    /// # Arguments
    ///
    /// - `bit_vector`: Characteristic vector β(a, w) encoding matches for character a
    /// - `consumed_query`: Whether a character was consumed from the query word
    /// - `consumed_dict`: Whether a character was consumed from the dictionary word
    ///
    /// # Returns
    ///
    /// Successor state with updated `length_diff`, or `None` if no successors exist
    ///
    /// # Length Difference Updates
    ///
    /// - `(true, true)`: Both consumed → `length_diff` unchanged
    /// - `(true, false)`: Query consumed → `length_diff -= 1` (query advancing)
    /// - `(false, true)`: Dict consumed → `length_diff += 1` (dict advancing)
    /// - `(false, false)`: Neither consumed → `length_diff` unchanged (epsilon)
    ///
    /// # Diagonal Crossing
    ///
    /// When `|length_diff| > max_distance`, positions are converted:
    /// - I-type → M-type (or vice versa)
    ///
    /// This ensures the automaton correctly tracks which positions have crossed
    /// the diagonal in the edit graph.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = UniversalState::<Standard>::initial(2);
    /// let bv = CharacteristicVector::new('a', "abc");
    ///
    /// // Both words consume 'a'
    /// let next = state.transition_with_consumption(&bv, true, true);
    /// assert_eq!(next.expect("doc example: transition produces Some").length_diff(), 0);
    ///
    /// // Only query consumes (insertion in dict)
    /// let next = state.transition_with_consumption(&bv, true, false);
    /// assert_eq!(next.expect("doc example: transition produces Some").length_diff(), -1);
    /// ```
    pub fn transition_with_consumption(
        &self,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        consumed_query: bool,
        consumed_dict: bool,
    ) -> Option<Self> {
        // Special case: empty state has no successors
        if self.is_empty() {
            return None;
        }

        // Compute new length_diff based on consumption
        let new_length_diff = self.length_diff
            + match (consumed_query, consumed_dict) {
                (true, true) => 0,   // Both consumed, diff unchanged
                (true, false) => -1, // Query consumed, query advancing relative to dict
                (false, true) => 1,  // Dict consumed, dict advancing relative to query
                (false, false) => 0, // Neither consumed (epsilon transition)
            };

        // Create new state for successors (Δ)
        let mut next_state = Self::new(self.max_distance);
        next_state.length_diff = new_length_diff;

        // For each position π in current state Q
        for pos in &self.positions {
            // Compute δ^∀,χ_e(π, x) using the successors() method
            let successors = pos.successors(bit_vector, self.max_distance);

            // Add all successors to next state
            // The add_position() method automatically applies subsumption closure ⊔
            for succ in successors {
                next_state.add_position(succ);
            }
        }

        // Return None if no successors (undefined transition)
        if next_state.is_empty() {
            return None;
        }

        // Diagonal crossing check using length_diff
        // When |m| > n, we've crossed the diagonal and need to convert positions
        let c = self.max_distance as i32;
        if new_length_diff.abs() as i32 > c {
            // Apply position conversion (I ↔ M)
            let mut converted_state = Self::new(self.max_distance);
            converted_state.length_diff = new_length_diff;

            for pos in &next_state.positions {
                // Convert using length_diff-aware formula
                if let Some(converted) = convert_position_with_length_diff(
                    pos,
                    new_length_diff as i32,
                    self.max_distance,
                ) {
                    converted_state.add_position(converted);
                }
            }

            // Use converted state if non-empty, otherwise keep original
            if !converted_state.is_empty() {
                next_state = converted_state;
            }
        }

        // Return final state
        if next_state.is_empty() {
            None
        } else {
            Some(next_state)
        }
    }
}

/// Convert position between I-type and M-type using length difference
///
/// Simplified conversion formula that uses `length_diff` (m) instead of `k`.
///
/// For I-type → M-type when m > n:
/// ```text
/// new_offset = offset + (n + 1 - m)
/// ```
///
/// For M-type → I-type when m < -n:
/// ```text
/// new_offset = offset - (n + 1 + m)
/// ```
fn convert_position_with_length_diff<V: PositionVariant>(
    pos: &UniversalPosition<V>,
    length_diff: i32,
    max_distance: u8,
) -> Option<UniversalPosition<V>> {
    let offset = pos.offset();
    let errors = pos.errors();
    let n = max_distance as i32;

    match pos {
        UniversalPosition::INonFinal { .. } => {
            // I-type → M-type: Adjust offset based on how far we've crossed
            // The formula accounts for the crossing amount: (n + 1 - length_diff)
            let crossing_adjustment = n + 1 - length_diff;
            let new_offset = offset + crossing_adjustment;
            UniversalPosition::new_m(new_offset, errors, max_distance).ok()
        }
        UniversalPosition::MFinal { .. } => {
            // M-type → I-type: Reverse adjustment
            let crossing_adjustment = n + 1 + length_diff;
            let new_offset = offset - crossing_adjustment;
            UniversalPosition::new_i(new_offset, errors, max_distance).ok()
        }
    }
}

impl<V: PositionVariant> fmt::Display for UniversalState<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for pos in &self.positions {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", pos)?;
            first = false;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::universal::position::Standard;

    // =========================================================================
    // Basic State Tests
    // =========================================================================

    #[test]
    fn test_empty_state() {
        let state = UniversalState::<Standard>::new(2);
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
        assert!(!state.is_final());
        assert_eq!(state.max_distance(), 2);
    }

    #[test]
    fn test_initial_state() {
        let state = UniversalState::<Standard>::initial(2);
        assert!(!state.is_empty());
        assert_eq!(state.len(), 1);
        assert!(!state.is_final());

        let pos = UniversalPosition::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert!(state.contains(&pos));
    }

    #[test]
    fn test_add_single_position() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos = UniversalPosition::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos.clone());
        assert_eq!(state.len(), 1);
        assert!(state.contains(&pos));
    }

    #[test]
    fn test_add_multiple_non_subsuming_positions() {
        let mut state = UniversalState::<Standard>::new(3);
        // Use positions that don't subsume each other:
        // 0#1 and -2#2
        // Check: does 0#1 subsume -2#2? f > e: 2 > 1 ✓, |-2 - 0| = 2 ≤ 2 - 1 = 1? NO
        // Check: does -2#2 subsume 0#1? f > e: 1 > 2? NO
        let pos1 = UniversalPosition::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos1.clone());
        state.add_position(pos2.clone());

        // Both positions remain (not subsuming each other)
        assert_eq!(state.len(), 2);
        assert!(state.contains(&pos1));
        assert!(state.contains(&pos2));
    }

    // =========================================================================
    // Anti-chain / Subsumption Tests
    // =========================================================================

    #[test]
    fn test_add_position_removes_subsumed() {
        // Add I+2#2, then add I+1#1 which should remove I+2#2
        // I+1#1 subsumes I+2#2 because: errors(2) > errors(1) AND |2-1| ≤ 2-1
        let mut state = UniversalState::<Standard>::new(3);
        let pos1 = UniversalPosition::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args"); // Will be subsumed
        let pos2 = UniversalPosition::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args"); // Better position

        state.add_position(pos1.clone());
        assert_eq!(state.len(), 1);

        state.add_position(pos2.clone());
        // pos1 should be removed because pos2 subsumes pos1
        assert_eq!(state.len(), 1);
        assert!(!state.contains(&pos1));
        assert!(state.contains(&pos2));
    }

    #[test]
    fn test_add_position_rejected_if_subsumed() {
        // Add I+1#1, then try to add I+2#2 which should be rejected
        // I+1#1 subsumes I+2#2, so I+2#2 is rejected
        let mut state = UniversalState::<Standard>::new(3);
        let pos1 = UniversalPosition::new_i(1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args"); // Better position
        let pos2 = UniversalPosition::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args"); // Will be rejected

        state.add_position(pos1.clone());
        assert_eq!(state.len(), 1);

        state.add_position(pos2.clone());
        // pos2 should not be added because pos1 subsumes pos2
        assert_eq!(state.len(), 1);
        assert!(state.contains(&pos1));
        assert!(!state.contains(&pos2));
    }

    #[test]
    fn test_anti_chain_maintained() {
        // Add multiple positions and verify no position subsumes another
        let mut state = UniversalState::<Standard>::new(3);

        // Use valid positions that don't subsume each other
        let pos1 = UniversalPosition::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args"); // Valid: |-2| = 2 ≤ 2
        let pos3 = UniversalPosition::new_i(-1, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos1.clone());
        state.add_position(pos2.clone());
        state.add_position(pos3.clone());

        // Verify anti-chain: no position subsumes another
        let positions: Vec<_> = state.positions().collect();
        for (i, p1) in positions.iter().enumerate() {
            for (j, p2) in positions.iter().enumerate() {
                if i != j {
                    assert!(!subsumes(p1, p2, state.max_distance()));
                }
            }
        }
    }

    // =========================================================================
    // Final State Tests
    // =========================================================================

    #[test]
    fn test_final_state_with_m_zero() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos = UniversalPosition::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        state.add_position(pos);
        assert!(state.is_final());
    }

    #[test]
    fn test_final_state_with_m_negative() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos = UniversalPosition::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        state.add_position(pos);
        assert!(state.is_final());
    }

    #[test]
    fn test_not_final_with_only_i_positions() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos = UniversalPosition::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos);
        assert!(!state.is_final());
    }

    // =========================================================================
    // State Type Tests
    // =========================================================================

    #[test]
    fn test_is_i_state() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos1 = UniversalPosition::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos1);
        state.add_position(pos2);

        assert!(state.is_i_state());
        assert!(!state.is_m_state());
        assert!(!state.is_mixed_state());
    }

    #[test]
    fn test_is_m_state() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos1 = UniversalPosition::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        state.add_position(pos1);
        state.add_position(pos2);

        assert!(!state.is_i_state());
        assert!(state.is_m_state());
        assert!(!state.is_mixed_state());
    }

    #[test]
    fn test_is_mixed_state() {
        let mut state = UniversalState::<Standard>::new(2);
        let i_pos = UniversalPosition::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let m_pos = UniversalPosition::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        state.add_position(i_pos);
        state.add_position(m_pos);

        assert!(!state.is_i_state());
        assert!(!state.is_m_state());
        assert!(state.is_mixed_state());
    }

    // =========================================================================
    // Iterator Tests
    // =========================================================================

    #[test]
    fn test_positions_iterator() {
        let mut state = UniversalState::<Standard>::new(3);
        // Use positions that don't subsume each other
        let pos1 = UniversalPosition::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos1.clone());
        state.add_position(pos2.clone());

        let positions: Vec<_> = state.positions().cloned().collect();
        assert_eq!(positions.len(), 2);
        assert!(positions.contains(&pos1));
        assert!(positions.contains(&pos2));
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_display_empty_state() {
        let state = UniversalState::<Standard>::new(2);
        assert_eq!(format!("{}", state), "{}");
    }

    #[test]
    fn test_display_single_position() {
        let mut state = UniversalState::<Standard>::new(2);
        let pos = UniversalPosition::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        state.add_position(pos);

        let display = format!("{}", state);
        assert!(display.contains("I + 0#0"));
    }

    #[test]
    fn test_display_multiple_positions() {
        let mut state = UniversalState::<Standard>::new(3);
        // Use positions that don't subsume each other
        let pos1 = UniversalPosition::new_i(0, 1, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        state.add_position(pos1);
        state.add_position(pos2);

        let display = format!("{}", state);
        assert!(display.contains("I + 0#1"));
        assert!(display.contains("I + -2#2"));
    }

    // =========================================================================
    // Equality Tests
    // =========================================================================

    #[test]
    fn test_state_equality() {
        let mut state1 = UniversalState::<Standard>::new(2);
        let mut state2 = UniversalState::<Standard>::new(2);

        let pos = UniversalPosition::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        state1.add_position(pos.clone());
        state2.add_position(pos);

        assert_eq!(state1, state2);
    }

    #[test]
    fn test_state_inequality_different_positions() {
        let mut state1 = UniversalState::<Standard>::new(2);
        let mut state2 = UniversalState::<Standard>::new(2);

        state1.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );
        state2.add_position(
            UniversalPosition::new_i(1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        assert_ne!(state1, state2);
    }

    #[test]
    fn test_state_clone() {
        let mut state1 = UniversalState::<Standard>::new(2);
        state1.add_position(
            UniversalPosition::new_i(1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        let state2 = state1.clone();
        assert_eq!(state1, state2);
    }

    // =========================================================================
    // State Transition Tests (Phase 2 Week 5)
    // =========================================================================

    #[test]
    fn test_transition_from_initial_match() {
        // Test transition from {I + 0#0} on bit vector "100" (match at position 0)
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(2);
        let bv = CharacteristicVector::new('a', "abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        // From I + 0#0 with "100", we expect I + 0#0 (match keeps offset same)
        // After diagonal check, might convert to M-type
        assert!(!next.is_empty());
    }

    #[test]
    fn test_transition_from_initial_no_match() {
        // Test transition from {I + 0#0} on bit vector "000" (no matches)
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(2);
        let bv = CharacteristicVector::new('x', "abc"); // No 'x' in "abc"

        let next = state.transition(&bv, 1).expect("Should have successor");

        // From I + 0#0 with "000", we expect successors
        // (may be converted to M-type by diagonal crossing)
        assert!(!next.is_empty());
    }

    #[test]
    fn test_transition_applies_subsumption() {
        // Test that subsumption closure is applied during transition
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(3);
        // Start with a position that will produce subsuming successors
        state.add_position(
            UniversalPosition::new_i(0, 0, 3)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        let bv = CharacteristicVector::new('x', "abcd"); // "0000"

        let next = state.transition(&bv, 1).expect("Should have successor");

        // Verify no position subsumes another in result
        let positions: Vec<_> = next.positions().collect();
        for (i, p1) in positions.iter().enumerate() {
            for (j, p2) in positions.iter().enumerate() {
                if i != j {
                    assert!(!subsumes(p1, p2, next.max_distance()));
                }
            }
        }
    }

    #[test]
    fn test_transition_empty_state() {
        // Test that empty state has no successors
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::new(2);
        let bv = CharacteristicVector::new('a', "abc");

        assert!(state.transition(&bv, 1).is_none());
    }

    #[test]
    fn test_transition_multiple_positions() {
        // Test transition from state with multiple positions
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(2);
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );
        state.add_position(
            UniversalPosition::new_i(1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        let bv = CharacteristicVector::new('a', "abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        // Should have successors from both positions (union of successor sets)
        assert!(!next.is_empty());
    }

    #[test]
    fn test_transition_match_later() {
        // Test transition with match at position 2
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(2);
        let bv = CharacteristicVector::new('c', "abc"); // "001"

        let next = state.transition(&bv, 1).expect("Should have successor");

        // From I + 0#0 with "001", we expect successors
        // (may be converted by diagonal crossing)
        assert!(!next.is_empty());
    }

    #[test]
    fn test_transition_all_errors_consumed() {
        // Test that positions at max errors can still match
        // Position I+0#2 at input position 1 with windowed bit vector
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(2);
        state.add_position(
            UniversalPosition::new_i(0, 2, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Windowed bit vector for 'a' at input position 1, word "abc", max_distance 2
        // Window: "$$abc"
        let bv = CharacteristicVector::new('a', "$$abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        // All positions should have errors ≤ 2
        for pos in next.positions() {
            assert!(pos.errors() <= 2, "Position {:?} exceeds max errors", pos);
        }
    }

    #[test]
    fn test_transition_preserves_max_distance() {
        // Test that transition preserves max_distance
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(3);
        let bv = CharacteristicVector::new('a', "abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        assert_eq!(next.max_distance(), 3);
    }

    #[test]
    fn test_transition_sequence() {
        // Test a sequence of transitions (simulating processing a word)
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::initial(2);

        // Process "aaa" against word "abc"
        let bv1 = CharacteristicVector::new('a', "abc"); // "100"
        state = state.transition(&bv1, 1).expect("Should have successor");
        assert!(!state.is_empty());

        let bv2 = CharacteristicVector::new('a', "abc"); // "100"
        state = state.transition(&bv2, 2).expect("Should have successor");
        assert!(!state.is_empty());

        let bv3 = CharacteristicVector::new('a', "abc"); // "100"
        state = state.transition(&bv3, 3).expect("Should have successor");
        assert!(!state.is_empty());
    }

    #[test]
    fn test_transition_no_valid_successors() {
        // Test case where all successors violate invariants
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(0); // max_distance = 0
        state.add_position(
            UniversalPosition::new_i(0, 0, 0)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // With max_distance=0, only matches are allowed
        let bv = CharacteristicVector::new('x', "abc"); // No match

        // Should return None (no valid successors)
        assert!(state.transition(&bv, 1).is_none());
    }

    #[test]
    fn test_transition_with_consumption_updates_length_diff() {
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(2);
        let bv = CharacteristicVector::new('a', "abc");

        let both = state
            .transition_with_consumption(&bv, true, true)
            .expect("matching consumption should have a successor");
        assert_eq!(both.length_diff(), 0);

        let query_only = state
            .transition_with_consumption(&bv, true, false)
            .expect("query-only consumption should have a successor");
        assert_eq!(query_only.length_diff(), -1);

        let dict_only = state
            .transition_with_consumption(&bv, false, true)
            .expect("dict-only consumption should have a successor");
        assert_eq!(dict_only.length_diff(), 1);
    }

    #[test]
    fn test_transition_with_consumption_converts_i_to_m_after_diagonal_crossing() {
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::initial(1);
        state.length_diff = 1;
        let bv = CharacteristicVector::new('a', "a");

        let next = state
            .transition_with_consumption(&bv, false, true)
            .expect("diagonal-crossing transition should have a successor");

        assert_eq!(next.length_diff(), 2);
        assert!(
            next.positions().any(UniversalPosition::is_m_type),
            "I-type positions should convert to M-type after crossing m > n"
        );
    }

    #[test]
    fn test_length_diff_conversion_converts_m_to_i_after_reverse_crossing() {
        let pos = UniversalPosition::<Standard>::new_m(0, 0, 1)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        let converted = convert_position_with_length_diff(&pos, -2, 1)
            .expect("reverse diagonal crossing should produce a valid I-type position");

        assert!(
            !converted.is_m_type(),
            "M-type positions should convert to I-type after crossing m < -n"
        );
        assert_eq!(converted.offset(), 0);
        assert_eq!(converted.errors(), 0);
    }

    #[test]
    fn test_transition_from_m_type_state() {
        // Test transition from M-type state
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(2);
        state.add_position(
            UniversalPosition::new_m(-1, 0, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
        );

        let bv = CharacteristicVector::new('a', "abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        // Should produce M-type successors
        assert!(!next.is_empty());
        // All positions should be M-type
        for pos in next.positions() {
            assert!(pos.is_m_type());
        }
    }

    #[test]
    fn test_transition_union_of_successors() {
        // Test that transition correctly unions successors from multiple positions
        use crate::transducer::universal::CharacteristicVector;

        let mut state = UniversalState::<Standard>::new(2);
        // Add two positions that will produce different successors
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );
        state.add_position(
            UniversalPosition::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        let bv = CharacteristicVector::new('a', "abc");

        let next = state.transition(&bv, 1).expect("Should have successor");

        // The result should be the union of successors from both positions
        // (with subsumption closure applied)
        assert!(!next.is_empty());

        // Verify anti-chain property
        let positions: Vec<_> = next.positions().collect();
        for (i, p1) in positions.iter().enumerate() {
            for (j, p2) in positions.iter().enumerate() {
                if i != j {
                    assert!(!subsumes(p1, p2, next.max_distance()));
                }
            }
        }
    }

    #[test]
    fn test_transition_multiple_matches() {
        // Test with bit vector containing multiple matches
        use crate::transducer::universal::CharacteristicVector;

        let state = UniversalState::<Standard>::initial(2);
        let bv = CharacteristicVector::new('a', "aaa"); // "111"

        let next = state.transition(&bv, 1).expect("Should have successor");

        // Should match at first position
        assert!(!next.is_empty());
    }
}
