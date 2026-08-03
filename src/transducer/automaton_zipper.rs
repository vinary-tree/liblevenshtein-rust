//! Zipper for Levenshtein automaton state.
//!
//! This module provides a zipper abstraction for navigating through Levenshtein
//! automaton states. Unlike dictionary zippers which navigate graph structures,
//! the automaton zipper tracks the current state of the Levenshtein automaton
//! as it processes input characters.

use crate::transducer::transition::{transition_state_pooled_ref, TransitionSettings};
use crate::transducer::{Algorithm, Position, State, StatePool, Unrestricted};
use std::sync::Arc;

/// Zipper for tracking Levenshtein automaton state during traversal.
///
/// An `AutomatonZipper` represents a position in the Levenshtein automaton,
/// maintaining the current state (collection of positions), the query being
/// matched, and the algorithm parameters.
///
/// # Design
///
/// Unlike traditional zippers that navigate tree structures, AutomatonZipper
/// represents a "cursor" in the automaton's state space. Each transition
/// (consuming a character) moves to a new state, which may have multiple
/// positions representing different edit distances and offsets.
///
/// # Memory Efficiency
///
/// - `State`: Uses `SmallVec<[Position; 8]>` to avoid heap allocation for typical states
/// - `query`: Shared via `Arc` to enable cheap cloning across multiple zippers
/// - Total overhead: ~32 bytes + query length (shared)
///
/// # Thread Safety
///
/// AutomatonZipper is `Clone` and can be safely shared across threads.
/// State transitions create new zippers, leaving the original unchanged.
///
/// # Examples
///
/// ```
/// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
///
/// let query = "hello";
/// let max_distance = 2;
/// let algorithm = Algorithm::Standard;
///
/// // Create root zipper
/// let zipper = AutomatonZipper::new(
///     query.as_bytes(),
///     max_distance,
///     algorithm
/// );
///
/// // Check if zipper is viable (has any positions)
/// assert!(zipper.is_viable());
///
/// // Transition on character 'h'
/// let mut pool = StatePool::new();
/// if let Some(next) = zipper.transition(b'h', &mut pool) {
///     println!("Transitioned successfully");
/// }
/// ```
#[derive(Clone)]
pub struct AutomatonZipper {
    /// Current automaton state (collection of positions)
    state: State,

    /// Query string being matched (shared across zippers)
    query: Arc<Vec<u8>>,

    /// Maximum allowed edit distance
    max_distance: usize,

    /// Levenshtein algorithm variant
    algorithm: Algorithm,
}

impl AutomatonZipper {
    /// Create a new automaton zipper at the initial state.
    ///
    /// The initial state contains positions for each possible distance from 0 to max_distance,
    /// all starting at term_index 0. This allows the automaton to match terms with any
    /// number of initial insertions up to max_distance.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to match against (as byte slice)
    /// * `max_distance` - Maximum edit distance threshold
    /// * `algorithm` - Levenshtein algorithm variant to use
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new(
    ///     "test".as_bytes(),
    ///     1,  // max distance
    ///     Algorithm::Standard
    /// );
    ///
    /// // Initial state has positions for distances 0 and 1
    /// assert!(zipper.is_viable());
    /// ```
    pub fn new(query: &[u8], max_distance: usize, algorithm: Algorithm) -> Self {
        let mut state = State::new();

        // Initialize with positions for each possible starting distance
        // This handles the case where the dictionary term starts with insertions
        for distance in 0..=max_distance {
            state.insert(Position::new(0, distance), algorithm, query.len());
        }

        AutomatonZipper {
            state,
            query: Arc::new(query.to_vec()),
            max_distance,
            algorithm,
        }
    }

    /// Create an automaton zipper with a specific state.
    ///
    /// This is used internally when transitioning to create child zippers.
    ///
    /// # Arguments
    ///
    /// * `state` - The automaton state
    /// * `query` - Shared query reference
    /// * `max_distance` - Maximum edit distance
    /// * `algorithm` - Algorithm variant
    fn with_state(
        state: State,
        query: Arc<Vec<u8>>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        AutomatonZipper {
            state,
            query,
            max_distance,
            algorithm,
        }
    }

    /// Transition the automaton by consuming a dictionary character.
    ///
    /// Computes the next state by transitioning all positions in the current
    /// state given the input character. Uses the provided StatePool for
    /// efficient state allocation.
    ///
    /// # Arguments
    ///
    /// * `dict_char` - The character from the dictionary to consume
    /// * `pool` - StatePool for efficient state allocation/reuse
    ///
    /// # Returns
    ///
    /// `Some(next_zipper)` if the transition produces a viable state,
    /// `None` if no valid positions remain (dead end).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    ///
    /// let zipper = AutomatonZipper::new("cat".as_bytes(), 1, Algorithm::Standard);
    /// let mut pool = StatePool::new();
    ///
    /// // Transition on 'c'
    /// let zipper = zipper.transition(b'c', &mut pool).expect("should transition");
    ///
    /// // Transition on 'a'
    /// let zipper = zipper.transition(b'a', &mut pool).expect("should transition");
    ///
    /// // Transition on 't'
    /// let zipper = zipper.transition(b't', &mut pool).expect("should transition");
    ///
    /// // Should be accepting now (exact match)
    /// assert!(zipper.min_distance().is_some());
    /// ```
    #[inline]
    pub fn transition(&self, dict_char: u8, pool: &mut StatePool) -> Option<Self> {
        transition_state_pooled_ref(
            &self.state,
            pool,
            &Unrestricted, // Default policy: allow all substitutions
            dict_char,
            &self.query,
            TransitionSettings::new(
                self.max_distance,
                self.algorithm,
                false, // substring_mode
            ),
        )
        .map(|next_state| {
            AutomatonZipper::with_state(
                next_state,
                Arc::clone(&self.query),
                self.max_distance,
                self.algorithm,
            )
        })
    }

    /// Get the minimum edit distance if the automaton is in an accepting state.
    ///
    /// An accepting state is one where at least one position has consumed all
    /// query characters (term_index == query.len()). The minimum distance is
    /// the smallest num_errors among all such positions.
    ///
    /// This is primarily used for substring matching mode. For whole-term
    /// matching, use [`infer_distance`](Self::infer_distance), which charges the
    /// query suffix left by every normal position.
    ///
    /// # Returns
    ///
    /// `Some(min_distance)` if at least one position is accepting, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    ///
    /// let zipper = AutomatonZipper::new("cat".as_bytes(), 1, Algorithm::Standard);
    /// let mut pool = StatePool::new();
    ///
    /// // Not accepting at root (for non-empty query)
    /// assert_eq!(zipper.min_distance_accepting(), None);
    ///
    /// // Transition through "cat"
    /// let zipper = zipper.transition(b'c', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    /// let zipper = zipper.transition(b'a', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    /// let zipper = zipper.transition(b't', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    ///
    /// // Now accepting with distance 0 (exact match)
    /// assert_eq!(zipper.min_distance_accepting(), Some(0));
    /// ```
    pub fn min_distance_accepting(&self) -> Option<usize> {
        let query_len = self.query.len();

        self.state
            .positions()
            .iter()
            .filter(|p| p.term_index == query_len && !p.is_special())
            .map(|p| p.num_errors)
            .min()
    }

    /// Get the minimum error count from any position (for substring mode).
    ///
    /// This returns the minimum num_errors regardless of whether positions
    /// have consumed all query characters. Used in substring matching.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new("test".as_bytes(), 1, Algorithm::Standard);
    ///
    /// // Initial state has a position with 0 errors
    /// assert_eq!(zipper.min_distance(), Some(0));
    /// ```
    pub fn min_distance(&self) -> Option<usize> {
        self.state.min_distance()
    }

    /// Infer the edit distance when the current dictionary path ends.
    ///
    /// This is used when the dictionary signals that a term ends at the current
    /// position (is_final). The distance is inferred from the current automaton
    /// state and the stored query length.
    ///
    /// # Arguments
    ///
    /// * `_term_length` - Length of the dictionary term. Retained for API
    ///   compatibility; the finishing cost is defined by the stored query
    ///   length and each position's query index.
    ///
    /// # Returns
    ///
    /// `Some(distance)` if the term can be matched within max_distance,
    /// `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    ///
    /// let zipper = AutomatonZipper::new("test".as_bytes(), 2, Algorithm::Standard);
    /// let mut pool = StatePool::new();
    ///
    /// // Transition through "tes" (3 characters)
    /// let zipper = zipper.transition(b't', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    /// let zipper = zipper.transition(b'e', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    /// let zipper = zipper.transition(b's', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    ///
    /// // If dictionary term "tes" ends here, the unmatched final 't' in the
    /// // query costs one deletion.
    /// assert_eq!(zipper.infer_distance(3), Some(1));
    /// ```
    pub fn infer_distance(&self, _term_length: usize) -> Option<usize> {
        self.state.infer_distance(self.query.len())
    }

    /// Check if the zipper is in a viable state.
    ///
    /// A viable state has at least one position, meaning there are still
    /// possible matches. A non-viable state (empty) is a dead end.
    ///
    /// # Returns
    ///
    /// `true` if the state has positions, `false` if empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    ///
    /// let zipper = AutomatonZipper::new("a".as_bytes(), 1, Algorithm::Standard);
    /// let mut pool = StatePool::new();
    ///
    /// // Initial state is viable
    /// assert!(zipper.is_viable());
    ///
    /// // Transition on 'a' should still be viable
    /// let zipper = zipper.transition(b'a', &mut pool).expect("doc/test fixture: transition on valid dictionary path");
    /// assert!(zipper.is_viable());
    /// ```
    pub fn is_viable(&self) -> bool {
        !self.state.is_empty()
    }

    /// Get the query string being matched.
    ///
    /// Returns a reference to the query bytes. This is shared across all
    /// zippers created from the same root.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new("hello".as_bytes(), 1, Algorithm::Standard);
    /// assert_eq!(zipper.query(), "hello".as_bytes());
    /// ```
    pub fn query(&self) -> &[u8] {
        &self.query
    }

    /// Get the maximum distance threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new("test".as_bytes(), 2, Algorithm::Standard);
    /// assert_eq!(zipper.max_distance(), 2);
    /// ```
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Get the algorithm variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new("test".as_bytes(), 1, Algorithm::Transposition);
    /// assert_eq!(zipper.algorithm(), Algorithm::Transposition);
    /// ```
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get a reference to the current state (for debugging/testing).
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    ///
    /// let zipper = AutomatonZipper::new("test".as_bytes(), 1, Algorithm::Standard);
    /// let state = zipper.state();
    /// // State contains positions for distances 0 and 1
    /// assert!(!state.is_empty());
    /// ```
    pub fn state(&self) -> &State {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_initial_state() {
        let zipper = AutomatonZipper::new(b"test", 2, Algorithm::Standard);

        assert!(zipper.is_viable());
        assert_eq!(zipper.query(), b"test");
        assert_eq!(zipper.max_distance(), 2);
        assert_eq!(zipper.algorithm(), Algorithm::Standard);
    }

    #[test]
    fn test_root_not_accepting() {
        let zipper = AutomatonZipper::new(b"test", 1, Algorithm::Standard);
        assert_eq!(zipper.min_distance_accepting(), None);
    }

    #[test]
    fn test_min_distance_accepting_ignores_non_accepting_and_special_positions() {
        let state = State::from_positions(vec![
            Position::new(1, 0),
            Position::new(3, 2),
            Position::new_osa_transposing(3, 0),
            Position::new(3, 1),
        ]);
        let zipper =
            AutomatonZipper::with_state(state, Arc::new(b"abc".to_vec()), 2, Algorithm::Standard);

        assert_eq!(zipper.min_distance_accepting(), Some(1));
    }

    #[test]
    fn test_exact_match_transition() {
        let zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Transition through "cat"
        let z1 = zipper.transition(b'c', &mut pool).expect("c should work");
        assert!(z1.is_viable());
        assert_eq!(z1.min_distance_accepting(), None);

        let z2 = z1.transition(b'a', &mut pool).expect("a should work");
        assert!(z2.is_viable());
        assert_eq!(z2.min_distance_accepting(), None);

        let z3 = z2.transition(b't', &mut pool).expect("t should work");
        assert!(z3.is_viable());

        // Should be accepting with distance 0
        assert_eq!(z3.min_distance_accepting(), Some(0));
    }

    #[test]
    fn test_single_substitution() {
        let zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Transition through "bat" (one substitution: c->b)
        let z1 = zipper.transition(b'b', &mut pool).expect("b should work");
        let z2 = z1.transition(b'a', &mut pool).expect("a should work");
        let z3 = z2.transition(b't', &mut pool).expect("t should work");

        // Should be accepting with distance 1
        assert_eq!(z3.min_distance_accepting(), Some(1));
    }

    #[test]
    fn test_insertion() {
        let zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Transition through "cart" (one insertion: r)
        let z1 = zipper.transition(b'c', &mut pool).expect("c should work");
        let z2 = z1.transition(b'a', &mut pool).expect("a should work");
        let z3 = z2.transition(b'r', &mut pool).expect("r should work");
        let z4 = z3.transition(b't', &mut pool).expect("t should work");

        // Should be accepting with distance 1
        assert_eq!(z4.min_distance_accepting(), Some(1));
    }

    #[test]
    fn test_deletion() {
        let zipper = AutomatonZipper::new(b"cart", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Transition through "cat" (one deletion: r)
        let z1 = zipper.transition(b'c', &mut pool).expect("c should work");
        let z2 = z1.transition(b'a', &mut pool).expect("a should work");
        let z3 = z2.transition(b't', &mut pool).expect("t should work");

        // Should be accepting with distance 1 (r was deleted)
        assert_eq!(z3.min_distance_accepting(), Some(1));
    }

    #[test]
    fn test_exceeds_max_distance() {
        let zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Try to transition through "dog" (3 substitutions, exceeds max_distance=1)
        let z1_opt = zipper.transition(b'd', &mut pool);

        // Should still have a viable state (can still recover with 1 more error)
        if let Some(z1) = z1_opt {
            let z2_opt = z1.transition(b'o', &mut pool);

            if let Some(z2) = z2_opt {
                // After 2 substitutions, should not be viable anymore
                let z3_opt = z2.transition(b'g', &mut pool);

                // Might transition but won't be accepting
                if let Some(z3) = z3_opt {
                    // Distance would be 3, which exceeds max_distance=1
                    // So min_distance_accepting should be None or > 1
                    if let Some(dist) = z3.min_distance_accepting() {
                        assert!(dist > 1, "Distance should exceed max_distance");
                    }
                }
            }
        }
    }

    #[test]
    fn test_infer_distance_short_term() {
        let zipper = AutomatonZipper::new(b"test", 2, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Transition through "te" (2 characters)
        let z1 = zipper
            .transition(b't', &mut pool)
            .expect("doc/test fixture: transition on valid dictionary path");
        let z2 = z1
            .transition(b'e', &mut pool)
            .expect("test fixture: transition on valid dictionary path");

        // The dictionary path ends after "te", while two query units remain.
        // A position at query index 2 therefore finishes at cost 2.
        assert_eq!(z2.infer_distance(2), Some(2));
    }

    #[test]
    fn test_clone_creates_independent_zipper() {
        let zipper = AutomatonZipper::new(b"test", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        let clone1 = zipper.clone();
        let clone2 = zipper.clone();

        // Transition clone1
        let _z1 = clone1.transition(b't', &mut pool);

        // clone2 should still be at root
        assert_eq!(clone2.min_distance_accepting(), None);
    }

    #[test]
    fn test_transposition_algorithm() {
        let zipper = AutomatonZipper::new(b"ab", 1, Algorithm::Transposition);
        let mut pool = StatePool::new();

        // Transition through "ba" (transposition)
        let z1 = zipper.transition(b'b', &mut pool).expect("b should work");
        let z2 = z1.transition(b'a', &mut pool).expect("a should work");

        // Should be accepting with distance 1 (one transposition)
        assert_eq!(z2.min_distance_accepting(), Some(1));
    }

    #[test]
    fn test_empty_query() {
        let zipper = AutomatonZipper::new(b"", 1, Algorithm::Standard);
        let mut pool = StatePool::new();

        // Empty query should be accepting immediately
        assert_eq!(zipper.min_distance_accepting(), Some(0));

        // Any character should still allow accepting (insertion)
        let z1 = zipper.transition(b'a', &mut pool).expect("should work");
        assert_eq!(z1.min_distance_accepting(), Some(1));
    }
}
