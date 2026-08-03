//! Universal Levenshtein Automaton
//!
//! Implements the complete Universal Levenshtein Automaton A^∀,χ_n from Mitankin's thesis.
//!
//! # Theory Background
//!
//! ## Definition 15: Universal Levenshtein Automaton (Page 30)
//!
//! ```text
//! A^∀,χ_n = ⟨Σ^∀_n, Q^∀,χ_n, I^∀,χ, F^∀,χ_n, δ^∀,χ_n⟩
//! ```
//!
//! Where:
//! - **Σ^∀_n**: Bit vectors of length ≤ 2n + 2
//! - **Q^∀,χ_n**: State space (I^χ_states ∪ M^χ_states)
//! - **I^∀,χ**: Initial state {I#0}
//! - **F^∀,χ_n**: Final states M^χ_states (states with M-type positions)
//! - **δ^∀,χ_n**: Transition function
//!
//! ## Acceptance Condition (Page 48)
//!
//! Given a word w and input x, the automaton accepts if:
//! 1. Encode the pair as h_n(w, x) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
//! 2. Process the bit vector sequence: δ^∀,χ_n*(I^∀,χ, h_n(w, x))
//! 3. Check if the resulting state is in F^∀,χ_n (contains M-type positions)
//!
//! ## Key Properties
//!
//! - **Parameter-free**: Same automaton works for any word w
//! - **Deterministic**: δ^∀,χ_n is a function (not a relation)
//! - **Finite**: State space is finite for fixed n
//! - **Universal**: Simulates A^D,χ_n(w) for any word w

use crate::transducer::universal::{
    CharacteristicVector, PositionVariant, UniversalPosition, UniversalState,
};
use crate::transducer::{SubstitutionPolicy, Unrestricted};

/// Universal Levenshtein Automaton A^∀,χ_n
///
/// A parameter-free automaton that recognizes the Levenshtein neighborhood
/// L^χ_{Lev}(n, w) for any word w without modification.
///
/// # Type Parameters
///
/// - `V`: Position variant (Standard, Transposition, or MergeAndSplit)
/// - `P`: Substitution policy (defaults to [`Unrestricted`])
///
/// The default [`Unrestricted`] policy is a zero-sized type, so there is
/// zero memory or performance overhead for the default case.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::transducer::universal::{UniversalAutomaton, Standard};
///
/// // Create automaton for maximum distance n=2
/// let automaton = UniversalAutomaton::<Standard>::new(2);
///
/// // Check if "test" accepts "text" (distance 1)
/// assert!(automaton.accepts("test", "text"));
///
/// // Check if "test" accepts "hello" (distance > 2)
/// assert!(!automaton.accepts("test", "hello"));
/// ```
#[derive(Debug, Clone)]
pub struct UniversalAutomaton<V: PositionVariant, P: SubstitutionPolicy = Unrestricted> {
    /// Maximum edit distance n
    max_distance: u8,
    /// Phantom data for position variant and substitution policy
    _phantom: std::marker::PhantomData<(V, P)>,
}

// Backward-compatible constructors for Unrestricted policy
impl<V: PositionVariant> UniversalAutomaton<V, Unrestricted> {
    /// Create a new Universal Levenshtein Automaton for maximum distance n
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    ///
    /// # Returns
    ///
    /// A new `UniversalAutomaton` instance with unrestricted substitutions
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = UniversalAutomaton::<Standard>::new(2);
    /// ```
    #[must_use]
    pub fn new(max_distance: u8) -> Self {
        Self {
            max_distance,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Generic methods (work with any policy)
impl<V: PositionVariant, P: SubstitutionPolicy> UniversalAutomaton<V, P> {
    /// Create a new Universal Levenshtein Automaton with custom substitution policy
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    /// - `policy`: Substitution policy to use
    ///
    /// # Returns
    ///
    /// A new `UniversalAutomaton` instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let policy_set = SubstitutionSet::phonetic_basic();
    /// let policy = Restricted::new(&policy_set);
    /// let automaton = UniversalAutomaton::<Standard>::with_policy(2, policy);
    /// ```
    #[must_use]
    pub fn with_policy(max_distance: u8, _policy: P) -> Self {
        Self {
            max_distance,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the maximum edit distance n
    #[must_use]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Create the initial state I^∀,χ = {I#0}
    ///
    /// From thesis page 38: The initial state contains a single I-type position
    /// with offset 0 and 0 errors.
    ///
    /// # Returns
    ///
    /// Initial state {I#0}
    fn initial_state(&self) -> UniversalState<V> {
        let mut state = UniversalState::new(self.max_distance);
        // I#0: I-type position with offset 0, errors 0
        if let Ok(pos) = UniversalPosition::new_i(0, 0, self.max_distance) {
            state.add_position(pos);
        }
        state
    }

    /// Check if a state is accepting according to Levenshtein distance criterion
    ///
    /// From thesis page 24 (Proposition 11): A position i#e is accepting if:
    /// ```text
    /// p - i ≤ n - e
    /// ```
    /// Where p is word length, i is position, n is max distance, e is errors.
    ///
    /// This translates to: remaining characters ≤ remaining error budget.
    ///
    /// # Arguments
    ///
    /// - `state`: State to check after processing all input
    /// - `word_len`: Length of dictionary word
    /// - `input_len`: Length of input string processed
    ///
    /// # Returns
    ///
    /// `true` if state is accepting (within distance n), `false` otherwise
    #[must_use]
    fn is_accepting(&self, state: &UniversalState<V>, word_len: usize, input_len: usize) -> bool {
        let n = i128::from(self.max_distance);
        let word_len = word_len as i128;
        let input_len = input_len as i128;

        state.positions().any(|pos| {
            if pos.is_m_type() {
                // M-type positions are past the word end
                // They're accepting if offset ≤ 0 and errors ≤ n
                pos.offset() <= 0 && pos.errors() <= self.max_distance
            } else {
                // I-type positions: check if we can reach word end with remaining errors
                // Current word position = input_len + offset
                let current_word_pos = input_len + i128::from(pos.offset());

                if current_word_pos < 0 {
                    return false; // Before word start
                }

                // Remaining characters to match = word_len - current_word_pos
                let remaining_chars = word_len - current_word_pos;

                // Remaining error budget = max_distance - errors_used
                let remaining_errors = n - i128::from(pos.errors());

                // Accept if we can delete remaining characters with available errors
                // Proposition 11: p - i ≤ n - e
                remaining_chars >= 0 && remaining_chars <= remaining_errors
            }
        })
    }

    /// Check if word w accepts input x within the maximum distance
    ///
    /// From thesis page 51-52: Encodes the pair (w, x) as h_n(w, x) and
    /// processes it through the automaton.
    ///
    /// # Arguments
    ///
    /// - `word`: Dictionary word w
    /// - `input`: Input string x to match against
    ///
    /// # Returns
    ///
    /// `true` if Lev(w, x) ≤ n, `false` otherwise
    ///
    /// # Algorithm
    ///
    /// 1. Start with initial state I^∀,χ = {I#0}
    /// 2. For each character x_i in input:
    ///    - Compute relevant subword s_n(w, i)
    ///    - Compute characteristic vector β(x_i, s_n(w, i))
    ///    - Apply transition: state := δ^∀,χ_n(state, β)
    /// 3. Check if final state is in F^∀,χ_n (contains M-type positions)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = UniversalAutomaton::<Standard>::new(2);
    ///
    /// // Distance 1: one substitution
    /// assert!(automaton.accepts("test", "text"));
    ///
    /// // Distance 0: identical
    /// assert!(automaton.accepts("test", "test"));
    ///
    /// // Distance 3: too far
    /// assert!(!automaton.accepts("test", "hello"));
    /// ```
    pub fn accepts(&self, word: &str, input: &str) -> bool {
        let word_chars: Vec<char> = word.chars().collect();
        let word_len = word_chars.len();
        let input_len = input.chars().count();

        // Special case 1: Empty input (outside domain of h_n from thesis page 51)
        // From Levenshtein definition: d(w, ε) = |w|
        // Accept if |w| ≤ n
        if input.is_empty() {
            return word_len <= self.max_distance as usize;
        }

        // Special case 2: Input too long (encoding h_n undefined)
        // From thesis page 51: h_n(w, x) defined only if |x| ≤ |w| + n
        if input_len > word_len.saturating_add(self.max_distance as usize) {
            return false;
        }

        // Start with initial state {I#0}
        let mut state = self.initial_state();

        // Process each character of input
        // This generates the bit vector sequence h_n(w, x) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
        for (i, input_char) in input.chars().enumerate() {
            // Compute relevant subword s_n(w, i+1)
            // From thesis page 51: s_n(w, i) = w_{i-n}...w_{min(|w|, i+n+1)}
            let position = i.saturating_add(1);
            let subword = self.relevant_subword_from_chars(&word_chars, position);

            // Compute characteristic vector β(x_i, s_n(w, i))
            let bit_vector = CharacteristicVector::new(input_char, &subword);

            // Apply transition: state := δ^∀,χ_n(state, β)
            // Pass input position (1-indexed) as k parameter for diagonal crossing
            if let Some(next_state) = state.transition(&bit_vector, position) {
                state = next_state;
            } else {
                // Transition failed (¬!), reject
                return false;
            }
        }

        // Check acceptance using Proposition 11 criterion (thesis page 24)
        // A position i#e is accepting if: p - i ≤ n - e
        // (remaining characters ≤ remaining error budget)
        self.is_accepting(&state, word_len, input_len)
    }

    /// Compute relevant subword s_n(w, i)
    ///
    /// From thesis page 51:
    /// ```text
    /// s_n(w, i) = w_{i-n}w_{i-n+1}...w_v
    /// where v = min(|w|, i + n + 1)
    /// ```
    ///
    /// Pad with '$' for positions before start of word.
    ///
    /// # Arguments
    ///
    /// - `word`: Dictionary word w
    /// - `position`: Position i (1-indexed)
    ///
    /// # Returns
    ///
    /// Relevant subword around position i
    #[cfg(test)]
    fn relevant_subword(&self, word: &str, position: usize) -> String {
        let word_chars: Vec<char> = word.chars().collect();
        self.relevant_subword_from_chars(&word_chars, position)
    }

    fn relevant_subword_from_chars(&self, word_chars: &[char], position: usize) -> String {
        let n = self.max_distance as usize;
        let word_len = word_chars.len();

        // From thesis page 51: s_n(w, i) = w_{i-n}...w_v where v = min(|w|, i + n + 1).
        // Positions are 1-indexed in the thesis, while slices are 0-indexed.
        let pad_count = n.saturating_add(1).saturating_sub(position);
        let start = position.saturating_sub(n).max(1);
        let end = position.saturating_add(n).saturating_add(1).min(word_len);
        let word_char_count = if start <= end { end - start + 1 } else { 0 };

        let mut result = String::with_capacity(pad_count.saturating_add(word_char_count));
        result.extend(std::iter::repeat_n('$', pad_count));

        if start <= end {
            result.extend(word_chars[start - 1..end].iter().copied());
        }

        result
    }

    /// Process a bit vector sequence and return the final state
    ///
    /// This is useful for debugging and testing intermediate states.
    ///
    /// # Arguments
    ///
    /// - `bit_vectors`: Sequence of characteristic vectors
    ///
    /// # Returns
    ///
    /// Final state after processing all bit vectors, or None if any transition fails
    pub fn process(&self, bit_vectors: &[CharacteristicVector]) -> Option<UniversalState<V>> {
        let mut state = self.initial_state();

        for (i, bv) in bit_vectors.iter().enumerate() {
            state = state.transition(bv, i.saturating_add(1))?;
        }

        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::universal::Standard;

    // =========================================================================
    // Basic Automaton Creation Tests
    // =========================================================================

    #[test]
    fn test_new_automaton() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        assert_eq!(automaton.max_distance(), 2);
    }

    #[test]
    fn test_initial_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let state = automaton.initial_state();

        // Initial state should have exactly one position: I#0
        let positions: Vec<_> = state.positions().collect();
        assert_eq!(positions.len(), 1);
        assert!(positions[0].is_i_type());
        assert_eq!(positions[0].offset(), 0);
        assert_eq!(positions[0].errors(), 0);
    }

    #[test]
    fn test_is_accepting_i_type_at_word_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 4 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should be accepting: word_len=4, input_len=4, offset=0, errors=0
        // current_word_pos = 4 + 0 = 4, remaining = 4 - 4 = 0 ≤ (2 - 0) = 2 ✓
        assert!(automaton.is_accepting(&state, 4, 4));
    }

    #[test]
    fn test_is_accepting_i_type_before_word_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 2 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should be accepting: word_len=4, input_len=2, offset=0, errors=0
        // current_word_pos = 2 + 0 = 2, remaining = 4 - 2 = 2 ≤ (2 - 0) = 2 ✓
        assert!(automaton.is_accepting(&state, 4, 2));
    }

    #[test]
    fn test_is_accepting_m_type_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // M + 0#0 (past word end with 0 errors)
        state.add_position(
            UniversalPosition::new_m(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
        );

        // M-type with offset ≤ 0 and errors ≤ n is accepting
        assert!(automaton.is_accepting(&state, 4, 5));
    }

    #[test]
    fn test_is_accepting_mixed_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );
        state.add_position(
            UniversalPosition::new_m(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
        );

        // State with at least one accepting position (M-type) is accepting
        assert!(automaton.is_accepting(&state, 4, 5));
    }

    #[test]
    fn test_not_accepting_too_many_remaining_chars() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 0 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should NOT be accepting: remaining = 4 - 0 = 4 > (2 - 0) = 2
        assert!(!automaton.is_accepting(&state, 4, 0));
    }

    // =========================================================================
    // Relevant Subword Tests
    // =========================================================================

    #[test]
    fn test_relevant_subword_start() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 1, n=2: window is [i-n, v] = [-1, 4] inclusive
        // v = min(|w|, i+n+1) = min(4, 1+2+1) = 4
        // Should be: $, $, w_1, w_2, w_3, w_4 = $$test (6 chars, 2n+2)
        let subword = automaton.relevant_subword("test", 1);
        assert_eq!(subword, "$$test");
    }

    #[test]
    fn test_relevant_subword_middle() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 3: window is [1, 6)
        // Should be: w[0], w[1], w[2], w[3]
        let subword = automaton.relevant_subword("test", 3);
        assert_eq!(subword, "test");
    }

    #[test]
    fn test_relevant_subword_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 4: window is [2, 7) but word ends at 4
        // Should be: w[1], w[2], w[3]
        let subword = automaton.relevant_subword("test", 4);
        assert_eq!(subword, "est");
    }

    #[test]
    fn test_relevant_subword_n1() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        // Position 2, n=1: window is [i-n, v] = [1, 4] inclusive
        // v = min(|w|, i+n+1) = min(4, 2+1+1) = 4
        // Should be: w_1, w_2, w_3, w_4 = test (4 chars, 2n+2 = 4)
        let subword = automaton.relevant_subword("test", 2);
        assert_eq!(subword, "test");
    }

    #[test]
    fn test_relevant_subword_unicode_character_positions() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        let subword = automaton.relevant_subword("éaß", 2);
        assert_eq!(subword, "éaß");
    }

    #[test]
    fn relevant_subword_at_saturated_position_is_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(u8::MAX);
        let subword = automaton.relevant_subword("abc", usize::MAX);
        assert_eq!(subword, "");
    }

    // =========================================================================
    // Acceptance Tests
    // =========================================================================

    #[test]
    fn test_accepts_identical() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 0
        assert!(automaton.accepts("test", "test"));
    }

    #[test]
    fn test_accepts_substitution() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one substitution (s → x)
        assert!(automaton.accepts("test", "text"));
    }

    #[test]
    fn test_accepts_insertion() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one insertion (added 'a')
        assert!(automaton.accepts("test", "teast"));
    }

    #[test]
    fn test_accepts_deletion() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one deletion (removed 's')
        assert!(automaton.accepts("test", "tet"));
    }

    #[test]
    fn test_rejects_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 5: too many differences
        assert!(!automaton.accepts("test", "hello"));
    }

    #[test]
    fn test_accepts_empty_to_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 0
        assert!(automaton.accepts("", ""));
    }

    #[test]
    fn test_accepts_empty_word() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: insert two characters
        assert!(automaton.accepts("", "ab"));
    }

    #[test]
    fn test_rejects_empty_word_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 3: too many insertions
        assert!(!automaton.accepts("", "abc"));
    }

    #[test]
    fn test_accepts_to_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: delete two characters
        assert!(automaton.accepts("ab", ""));
    }

    #[test]
    fn test_rejects_to_empty_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 3: too many deletions
        assert!(!automaton.accepts("abc", ""));
    }

    #[test]
    fn test_accepts_multiple_edits() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: two substitutions
        assert!(automaton.accepts("test", "best"));
        assert!(automaton.accepts("test", "tent"));
    }

    #[test]
    fn test_accepts_n1() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        // Distance 1: one substitution (t→x at position 2)
        assert!(automaton.accepts("test", "text"));
        // Distance 1: one substitution (t→b at position 0)
        assert!(automaton.accepts("test", "best"));
        // Distance 2: should reject (two substitutions needed)
        assert!(!automaton.accepts("test", "bear"));
    }

    #[test]
    fn test_accepts_unicode_by_character_distance() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        assert!(automaton.accepts("é", ""));
        assert!(automaton.accepts("", "é"));
        assert!(automaton.accepts("café", "cafe"));
        assert!(!automaton.accepts("éø", ""));
    }

    #[test]
    fn test_accepts_longer_words() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1
        assert!(automaton.accepts("algorithm", "algorythm"));
        // Distance 2
        assert!(automaton.accepts("algorithm", "algarithm"));
    }

    // =========================================================================
    // Transposition Variant Tests
    // =========================================================================

    #[test]
    fn test_transposition_adjacent_swap_start() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap first two characters: "test" → "etst"
        assert!(automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_adjacent_swap_middle() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap middle characters: "test" → "tset"
        assert!(automaton.accepts("test", "tset"));
    }

    #[test]
    fn test_transposition_adjacent_swap_end() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap last two characters: "test" → "tets"
        assert!(automaton.accepts("test", "tets"));
    }

    #[test]
    fn test_transposition_with_standard_operations() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(2);

        // Transposition + deletion: "test" → "tset" → "set" (distance 2)
        assert!(automaton.accepts("test", "set"));

        // Transposition + insertion: "test" → "tset" → "taset" (distance 2)
        assert!(automaton.accepts("test", "taset"));
    }

    #[test]
    fn test_transposition_longer_words() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // "algorithm" → "lagorithm" (swap 'a' and 'l')
        assert!(automaton.accepts("algorithm", "lagorithm"));

        // "algorithm" → "aglorithm" (swap 'l' and 'g')
        assert!(automaton.accepts("algorithm", "aglorithm"));
    }

    #[test]
    fn test_transposition_rejects_non_adjacent() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Cannot swap non-adjacent chars with distance 1
        // "test" → "stet" requires swapping 't' and 's' (positions 0 and 2)
        // This needs 2 operations, so should reject
        assert!(!automaton.accepts("test", "stet"));
    }

    #[test]
    fn test_transposition_empty_and_single_char() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Empty word
        assert!(automaton.accepts("", ""));

        // Single character - transposition mode still supports standard operations
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // Accepts via substitution (transposition includes standard ops)
    }

    #[test]
    fn test_transposition_two_chars() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Two characters - single transposition
        assert!(automaton.accepts("ab", "ba"));
        assert!(automaton.accepts("xy", "yx"));
    }

    #[test]
    fn test_transposition_distance_zero() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(0);

        // Distance 0 - only exact matches
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("test", "etst")); // Would need transposition
    }

    #[test]
    fn test_transposition_vs_standard() {
        use crate::transducer::universal::Transposition;

        // With transposition: "test" → "etst" = distance 1
        let trans_automaton = UniversalAutomaton::<Transposition>::new(1);
        assert!(trans_automaton.accepts("test", "etst"));

        // With standard: "test" → "etst" requires 2 operations
        // (delete 't', insert 'e' OR substitute twice)
        let std_automaton = UniversalAutomaton::<Standard>::new(1);
        assert!(!std_automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_multiple_swaps() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(2);

        // Two transpositions: "abcd" → "bacd" → "badc"
        assert!(automaton.accepts("abcd", "badc"));
    }

    #[test]
    fn test_transposition_with_repeated_chars() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // "abcd" → "bacd" (swap first two adjacent chars)
        assert!(automaton.accepts("abcd", "bacd"));

        // "aabb" → "abab" (swap middle two adjacent chars)
        assert!(automaton.accepts("aabb", "abab"));

        // "aabc" → "aacb" (swap last two adjacent chars)
        assert!(automaton.accepts("aabc", "aacb"));
    }

    // ============================================================================
    // MERGE AND SPLIT TESTS
    // ============================================================================

    #[test]
    fn test_merge_and_split_distance_zero() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(0);

        // Distance 0 should only accept identical strings
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("hello", "hello"));
        assert!(!automaton.accepts("hello", "helo"));
        assert!(!automaton.accepts("hello", "helllo"));
    }

    #[test]
    fn test_merge_simple() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge: "ab" → "a" (merge two input chars 'ab' into one word char 'a')
        // Input has "ab", word has "a" - merge consumes 2 input chars for 1 word char
        assert!(automaton.accepts("ab", "a"));

        // Merge at different positions
        assert!(automaton.accepts("abc", "ac")); // merge 'ab' → 'a'
        assert!(automaton.accepts("xab", "xa")); // merge 'ab' → 'a' at end
        assert!(automaton.accepts("xaby", "xay")); // merge 'ab' → 'a' in middle
    }

    #[test]
    fn test_split_simple() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split: "a" → "ab" (split one input char 'a' into two word chars 'ab')
        // Input has "a", word has "ab" - split expands 1 input char to 2 word chars
        assert!(automaton.accepts("a", "ab"));

        // Split at different positions
        assert!(automaton.accepts("ac", "abc")); // split 'a' → 'ab'
        assert!(automaton.accepts("xa", "xab")); // split 'a' → 'ab' at end
        assert!(automaton.accepts("xay", "xaby")); // split 'a' → 'ab' in middle

        // Additional split tests
        assert!(automaton.accepts("b", "bc")); // split 'b' → 'bc'
        assert!(automaton.accepts("t", "te")); // split 't' → 'te'
    }

    #[test]
    fn test_merge_and_split_longer_words() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge in longer words
        assert!(automaton.accepts("algorithm", "algorihm")); // merge 'it' → 'i'
        assert!(automaton.accepts("banana", "banna")); // merge 'an' → 'n'

        // Split in longer words
        assert!(automaton.accepts("algorithim", "algorithm")); // split 'i' → 'it'
        assert!(automaton.accepts("banna", "banana")); // split 'n' → 'an'
    }

    #[test]
    fn test_merge_and_split_with_standard_operations() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge/split mode should include ALL standard operations

        // Standard insertion
        assert!(automaton.accepts("test", "teest"));

        // Standard deletion
        assert!(automaton.accepts("test", "tst"));

        // Standard substitution
        assert!(automaton.accepts("test", "best"));
    }

    #[test]
    fn test_merge_and_split_empty_and_single_char() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Empty word
        assert!(automaton.accepts("", ""));

        // Single character - merge/split mode still supports standard operations
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // substitution
        assert!(automaton.accepts("a", "")); // deletion
        assert!(automaton.accepts("", "a")); // insertion

        // Single char to two chars (split)
        assert!(automaton.accepts("a", "ab"));

        // Two chars to one char (merge)
        assert!(automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_merge_at_start() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge at the very start of the word
        assert!(automaton.accepts("abcd", "acd")); // merge 'ab' → 'a'
        assert!(automaton.accepts("test", "est")); // merge 'te' → 'e'
    }

    #[test]
    fn test_merge_at_end() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge at the very end of the word
        assert!(automaton.accepts("test", "tes")); // merge 'st' → 's'
        assert!(automaton.accepts("abcd", "abc")); // merge 'cd' → 'c'
    }

    #[test]
    fn test_split_at_start() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split at the very start of the word
        assert!(automaton.accepts("acd", "abcd")); // split 'a' → 'ab'
        assert!(automaton.accepts("est", "test")); // split 'e' → 'te'
    }

    #[test]
    fn test_split_at_end() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split at the very end of the word
        assert!(automaton.accepts("tes", "test")); // split 's' → 'st'
        assert!(automaton.accepts("abc", "abcd")); // split 'c' → 'cd'
    }

    #[test]
    fn test_merge_and_split_multiple_operations() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(2);

        // Multiple merge operations
        assert!(automaton.accepts("abcd", "ac")); // merge 'ab' → 'a', merge 'cd' → 'c'

        // Multiple split operations
        assert!(automaton.accepts("ac", "abcd")); // split 'a' → 'ab', split 'c' → 'cd'

        // Mix of operations
        assert!(automaton.accepts("abc", "abbc")); // split 'b' → 'bb'
        assert!(automaton.accepts("abbc", "abc")); // merge 'bb' → 'b'
    }

    #[test]
    fn test_merge_and_split_vs_standard() {
        use crate::transducer::universal::{MergeAndSplit, Standard};
        let merge_split_automaton = UniversalAutomaton::<MergeAndSplit>::new(1);
        let standard_automaton = UniversalAutomaton::<Standard>::new(1);

        // Standard operations should work in both
        assert_eq!(
            standard_automaton.accepts("test", "best"),
            merge_split_automaton.accepts("test", "best")
        );
        assert_eq!(
            standard_automaton.accepts("test", "tst"),
            merge_split_automaton.accepts("test", "tst")
        );

        // Verify that merge/split automaton does support these operations
        // Note: We can't easily demonstrate that standard DOESN'T support merge/split
        // because "ab" → "a" can be achieved with deletion, and "a" → "ab" with insertion.
        // The key difference is efficiency: merge/split does it in 1 operation,
        // while standard needs 2 operations.

        // With distance 1, merge/split can do:
        assert!(merge_split_automaton.accepts("ab", "a")); // merge in 1 op
        assert!(merge_split_automaton.accepts("a", "ab")); // split in 1 op
        assert!(merge_split_automaton.accepts("abc", "ac")); // merge 'ab' → 'a'
        assert!(merge_split_automaton.accepts("ac", "abc")); // split 'a' → 'ab'

        // Standard automaton can also accept these, but via different paths
        // (e.g., deletion + substitution for "ab" → "a", insertion + substitution for "a" → "ab")
        // So these tests just verify merge/split works correctly
    }

    #[test]
    fn test_merge_and_split_with_repeated_chars() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge with repeated characters
        assert!(automaton.accepts("aab", "ab")); // merge 'aa' → 'a'
        assert!(automaton.accepts("aabb", "abb")); // merge 'aa' → 'a'
        assert!(automaton.accepts("abbb", "abb")); // merge 'bb' → 'b'

        // Split with repeated characters
        assert!(automaton.accepts("ab", "aab")); // split 'a' → 'aa'
        assert!(automaton.accepts("abb", "aabb")); // split 'a' → 'aa'
        assert!(automaton.accepts("abb", "abbb")); // split 'b' → 'bb'
    }
}
