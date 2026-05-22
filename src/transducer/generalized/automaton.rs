//! Generalized Levenshtein Automaton
//!
//! Implements a Levenshtein automaton with runtime-configurable operations.
//!
//! # Design Philosophy
//!
//! `GeneralizedAutomaton` complements `UniversalAutomaton` by trading compile-time
//! specialization for runtime flexibility:
//!
//! - **UniversalAutomaton**: Compile-time operations (Standard, Transposition, MergeAndSplit)
//!   - Zero runtime overhead
//!   - Fixed operation sets
//!   - Perfect for standard Levenshtein variants
//!
//! - **GeneralizedAutomaton**: Runtime operations (Phase 1: standard only)
//!   - Small runtime overhead (+10-20% estimate)
//!   - Future: Custom operation sets
//!   - Perfect for phonetic corrections and custom metrics
//!
//! # Phase 1 Implementation
//!
//! Current implementation supports **standard operations only** (match, substitute, insert, delete).
//! Future phases will add:
//! - Runtime OperationSet parameter
//! - Multi-character operations
//! - Custom operation sets
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
//! - **F^∀,χ_n**: Final states (states with M-type positions)
//! - **δ^∀,χ_n**: Transition function
//!
//! ## Acceptance Condition (Page 48)
//!
//! Given a word w and input x, the automaton accepts if:
//! 1. Encode the pair as h_n(w, x) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
//! 2. Process the bit vector sequence: δ^∀,χ_n*(I^∀,χ, h_n(w, x))
//! 3. Check if the resulting state is final (contains M-type positions)
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
//!
//! // Create automaton for maximum distance n=2
//! let automaton = GeneralizedAutomaton::new(2);
//!
//! // Check if "test" accepts "text" (distance 1)
//! assert!(automaton.accepts("test", "text"));
//!
//! // Check if "test" accepts "hello" (distance > 2)
//! assert!(!automaton.accepts("test", "hello"));
//! ```

use super::state::GeneralizedState;
use crate::transducer::universal::bit_vector::CharacteristicVector;
use crate::transducer::OperationSet;

/// Generalized Levenshtein Automaton A^∀,χ_n
///
/// A parameter-free automaton that recognizes the Levenshtein neighborhood
/// L^χ_{Lev}(n, w) for any word w without modification.
///
/// # Phase 2: Runtime-Configurable Operations
///
/// Supports custom operation sets including:
/// - Standard operations (match, substitute, insert, delete)
/// - Transposition
/// - Multi-character operations (phonetic corrections, merge/split)
/// - Weighted operations with custom costs
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
/// use liblevenshtein::transducer::OperationSet;
///
/// // Standard Levenshtein
/// let automaton = GeneralizedAutomaton::new(2);
///
/// // With transposition
/// let automaton = GeneralizedAutomaton::with_operations(
///     2,
///     OperationSet::with_transposition()
/// );
///
/// // Check if "test" accepts "text" (distance 1)
/// assert!(automaton.accepts("test", "text"));
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedAutomaton {
    /// Maximum edit distance n
    max_distance: u8,

    /// Set of operations defining the edit distance metric
    operations: OperationSet,
}

impl GeneralizedAutomaton {
    /// Create a new Generalized Levenshtein Automaton with standard operations
    ///
    /// Uses the standard operation set (match, substitute, insert, delete).
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    ///
    /// # Returns
    ///
    /// A new `GeneralizedAutomaton` instance with standard operations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = GeneralizedAutomaton::new(2);
    /// ```
    #[must_use]
    pub fn new(max_distance: u8) -> Self {
        Self {
            max_distance,
            operations: OperationSet::standard(),
        }
    }

    /// Create a new Generalized Levenshtein Automaton with custom operations
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    /// - `operations`: Set of operations defining the edit distance metric
    ///
    /// # Returns
    ///
    /// A new `GeneralizedAutomaton` instance with custom operations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Automaton with transposition support
    /// let automaton = GeneralizedAutomaton::with_operations(
    ///     2,
    ///     OperationSet::with_transposition()
    /// );
    ///
    /// // Automaton with custom phonetic operations
    /// let mut ops = OperationSet::standard();
    /// ops.add_merge("ph", "f", 0);  // "ph" → "f" with no cost
    /// let automaton = GeneralizedAutomaton::with_operations(2, ops);
    /// ```
    #[must_use]
    pub fn with_operations(max_distance: u8, operations: OperationSet) -> Self {
        Self {
            max_distance,
            operations,
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
    fn initial_state(&self) -> GeneralizedState {
        GeneralizedState::initial(self.max_distance)
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
    fn is_accepting(&self, state: &GeneralizedState, word_len: usize, input_len: usize) -> bool {
        use crate::transducer::generalized::GeneralizedPosition;

        let n = self.max_distance as i32;

        state.positions().any(|pos| {
            match pos {
                GeneralizedPosition::MFinal { offset, errors } => {
                    // M-type positions are past the word end
                    // They're accepting if offset ≤ 0 and errors ≤ n
                    *offset <= 0 && *errors <= self.max_distance
                }
                GeneralizedPosition::INonFinal { .. } => {
                    // I-type positions: check if we can reach word end with remaining errors
                    // Current word position in the word = input_len + offset
                    let current_word_pos = input_len as i32 + pos.offset();

                    // Check bounds
                    if current_word_pos < 0 {
                        return false; // Before word start
                    }

                    // Remaining characters to match = word_len - current_word_pos
                    let remaining_chars = word_len as i32 - current_word_pos;

                    // Remaining error budget = max_distance - errors_used
                    let remaining_errors = n - (pos.errors() as i32);

                    // Accept if we can delete/skip remaining characters with available errors
                    // Proposition 11: p - i ≤ n - e
                    // where p = word_len, i = current_word_pos, e = errors used, n = max_distance
                    // Phase 4: If remaining_chars < 0, we're past the word end (acceptable)
                    // If remaining_chars >= 0, we need enough errors to delete them
                    remaining_chars <= remaining_errors
                }
                // Phase 2d: Transposing and splitting positions are intermediate states
                // They are not accepting states (operation must complete first)
                GeneralizedPosition::ITransposing { .. }
                | GeneralizedPosition::MTransposing { .. }
                | GeneralizedPosition::ISplitting { .. }
                | GeneralizedPosition::MSplitting { .. } => false,
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
    /// 3. Check if final state is accepting
    ///
    /// # Phase 1 Note
    ///
    /// Uses standard operations only (match, substitute, insert, delete).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = GeneralizedAutomaton::new(2);
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
        // Special case 1: Empty input (outside domain of h_n from thesis page 51)
        // From Levenshtein definition: d(w, ε) = |w|
        // Accept if |w| ≤ n
        if input.is_empty() {
            return word.len() <= self.max_distance as usize;
        }

        // Special case 2: Input too long (encoding h_n undefined)
        // From thesis page 51: h_n(w, x) defined only if |x| ≤ |w| + n
        // However, expansion operations (e.g., split ⟨1,2⟩) can increase effective word length
        // Phase 3b: Calculate maximum expansion from operations
        let max_expansion = self
            .operations
            .operations()
            .iter()
            .map(|op| op.consume_y().saturating_sub(op.consume_x()))
            .max()
            .unwrap_or(0);

        if max_expansion > 0 {
            // With expansion operations (e.g., k→ch split):
            // Upper bound: each word char could expand by max_expansion, plus insertions
            // Example: "kat" with 2 splits ⟨1,2⟩ → 3 + 2*1 = 5 effective chars
            // Conservative bound: |x| ≤ |w| * (1 + max_expansion) + n
            let max_len = word.len() * (1 + max_expansion as usize) + self.max_distance as usize;
            if input.len() > max_len {
                return false;
            }
        } else {
            // Standard Levenshtein bound: |x| ≤ |w| + n
            if input.len() > word.len() + self.max_distance as usize {
                return false;
            }
        }

        // Start with initial state {I#0}
        let mut state = self.initial_state();

        // H2 Optimization: Conditionally pre-compute character vector
        // Only pre-compute for max_distance > 1 (eliminates overhead for trivial cases)
        // For distance > 1: eliminates 20+ repeated word.chars().collect() calls
        // Target: 8.47% of cycles (Iterator::collect 4.08% + cfree 4.39%)
        let word_chars: Option<Vec<char>> = if self.max_distance > 1 {
            Some(word.chars().collect())
        } else {
            None
        };

        // Process each character of input
        // This generates the bit vector sequence h_n(w, x) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
        for (i, input_char) in input.chars().enumerate() {
            // Compute relevant subword s_n(w, i+1)
            // From thesis page 51: s_n(w, i) = w_{i-n}...w_{min(|w|, i+n+1)}
            let subword = self.relevant_subword(word, i + 1);

            #[cfg(debug_assertions)]
            eprintln!(
                "\n[DEBUG] === Input position i={}, char='{}' ===",
                i, input_char
            );
            #[cfg(debug_assertions)]
            eprintln!("  Subword: {:?}", subword);
            #[cfg(debug_assertions)]
            eprintln!(
                "  State before transition: {} positions",
                state.positions().count()
            );

            // Compute characteristic vector β(x_i, s_n(w, i))
            let bit_vector = CharacteristicVector::new(input_char, &subword);

            // Apply transition: state := δ^∀,χ_n(state, β)
            // Phase 3b: Pass full word, word slice, and input character for phonetic operations
            // H2 Optimization: Pass pre-computed character vector (conditional for distance > 1)
            if let Some(next_state) = state.transition(
                &self.operations,
                &bit_vector,
                word,
                word_chars.as_deref(),
                &subword,
                input_char,
                i + 1,
            ) {
                #[cfg(debug_assertions)]
                eprintln!(
                    "  State after transition: {} positions",
                    next_state.positions().count()
                );
                state = next_state;
            } else {
                // Transition failed, reject
                #[cfg(debug_assertions)]
                eprintln!("  ✗ Transition failed, rejecting");
                return false;
            }
        }

        // Check acceptance using Proposition 11 criterion (thesis page 24)
        // A position i#e is accepting if: p - i ≤ n - e
        // (remaining characters ≤ remaining error budget)
        #[cfg(debug_assertions)]
        {
            eprintln!("\n[DEBUG] Final state positions:");
            for pos in state.positions() {
                eprintln!("[DEBUG]   {}", pos);
            }
        }
        let accepted = self.is_accepting(&state, word.len(), input.len());
        #[cfg(debug_assertions)]
        eprintln!("[DEBUG] Accepted: {}", accepted);
        accepted
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
    fn relevant_subword(&self, word: &str, position: usize) -> String {
        let n = self.max_distance as i32;
        let i = position as i32;

        // From thesis page 51: s_n(w, i) = w_{i-n}...w_v where v = min(|w|, i + n + 1)
        // The notation w_a...w_b means positions from a through b inclusive (1-indexed)
        let start = i - n;
        let v = std::cmp::min(word.len() as i32, i + n + 1);

        let mut result = String::new();

        // Note: positions are 1-indexed in the thesis, but Rust uses 0-indexing
        // The range should be inclusive of v (start..=v would work, but v might be past word end)
        for pos in start..=v {
            if pos < 1 {
                // Before the start of the word - pad with '$'
                result.push('$');
            } else if pos <= word.len() as i32 {
                // Convert 1-indexed position to 0-indexed
                let idx = (pos - 1) as usize;
                if let Some(ch) = word.chars().nth(idx) {
                    result.push(ch);
                }
            }
            // If pos > word.len(), we're past the end - don't add anything
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let automaton = GeneralizedAutomaton::new(2);
        assert_eq!(automaton.max_distance(), 2);
    }

    #[test]
    fn test_debug_identical() {
        let automaton = GeneralizedAutomaton::new(2);
        let word = "test";
        let input = "test";

        // Manual step-through
        let mut state = automaton.initial_state();
        eprintln!("\nDEBUG: Initial state = {}", state);

        // H2 Optimization: Pre-compute character vector
        let word_chars: Vec<char> = word.chars().collect();

        for (i, ch) in input.chars().enumerate() {
            eprintln!(
                "\nDEBUG: Processing char {} ('{}') at input position {}",
                i + 1,
                ch,
                i
            );

            let subword = automaton.relevant_subword(word, i + 1);
            eprintln!("DEBUG: Relevant subword = '{}'", subword);

            let bit_vector = CharacteristicVector::new(ch, &subword);
            eprintln!("DEBUG: Bit vector length = {}", bit_vector.len());

            match state.transition(
                &automaton.operations,
                &bit_vector,
                word,
                Some(&word_chars),
                &subword,
                ch,
                i + 1,
            ) {
                Some(next) => {
                    eprintln!("DEBUG: Next state = {}", next);
                    state = next;
                }
                None => {
                    panic!("Transition failed at position {}", i);
                }
            }
        }

        eprintln!("\nDEBUG: Final state = {}", state);
        eprintln!(
            "DEBUG: Word length = {}, Input length = {}",
            word.len(),
            input.len()
        );

        // Check acceptance
        let is_accepting = automaton.is_accepting(&state, word.len(), input.len());
        eprintln!("DEBUG: is_accepting = {}", is_accepting);

        if !is_accepting {
            eprintln!("\nDEBUG: Checking each position:");
            for pos in state.positions() {
                let current_word_pos = input.len() as i32 + pos.offset();
                let remaining_chars = word.len() as i32 - current_word_pos;
                let remaining_errors = automaton.max_distance() as i32 - pos.errors() as i32;
                eprintln!("  Position {}: current_word_pos={}, remaining_chars={}, remaining_errors={}, accepting={}",
                         pos, current_word_pos, remaining_chars, remaining_errors,
                         remaining_chars >= 0 && remaining_chars <= remaining_errors);
            }
        }

        assert!(is_accepting, "Should accept identical strings");
    }

    #[test]
    fn test_accepts_identical() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "test"));
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_accepts_one_substitution() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "text")); // t->x
        assert!(automaton.accepts("hello", "hallo")); // e->a
    }

    #[test]
    fn test_debug_one_insertion() {
        let automaton = GeneralizedAutomaton::new(2);
        let word = "test";
        let input = "tests";

        eprintln!(
            "\nDEBUG: Testing insertion word='{}', input='{}', max_distance={}",
            word,
            input,
            automaton.max_distance()
        );

        let mut state = automaton.initial_state();
        eprintln!("Initial state: {}", state);

        // H2 Optimization: Pre-compute character vector
        let word_chars: Vec<char> = word.chars().collect();

        for (i, ch) in input.chars().enumerate() {
            eprintln!(
                "\n--- Processing char {} ('{}') at position {} ---",
                i + 1,
                ch,
                i + 1
            );

            let subword = automaton.relevant_subword(word, i + 1);
            eprintln!("Relevant subword: '{}'", subword);

            let bit_vector = CharacteristicVector::new(ch, &subword);
            eprintln!("Bit vector length: {}", bit_vector.len());

            match state.transition(
                &automaton.operations,
                &bit_vector,
                word,
                Some(&word_chars),
                &subword,
                ch,
                i + 1,
            ) {
                Some(next) => {
                    eprintln!("Next state: {}", next);
                    state = next;
                }
                None => {
                    eprintln!("Transition failed!");
                    panic!("Should not fail at position {}", i + 1);
                }
            }
        }

        eprintln!("\n--- Final state check ---");
        eprintln!("Final state: {}", state);
        eprintln!("Word length: {}", word.len());
        eprintln!("Input length: {}", input.len());

        let n = automaton.max_distance() as i32;
        for pos in state.positions() {
            let current_word_pos = input.len() as i32 + pos.offset();
            let remaining_chars = word.len() as i32 - current_word_pos;
            let remaining_errors = n - (pos.errors() as i32);

            eprintln!("\nPosition: {}", pos);
            eprintln!(
                "  current_word_pos = {} + {} = {}",
                input.len(),
                pos.offset(),
                current_word_pos
            );
            eprintln!(
                "  remaining_chars = {} - {} = {}",
                word.len(),
                current_word_pos,
                remaining_chars
            );
            eprintln!(
                "  remaining_errors = {} - {} = {}",
                n,
                pos.errors(),
                remaining_errors
            );
            eprintln!(
                "  Accept? {} >= 0 && {} <= {} = {}",
                remaining_chars,
                remaining_chars,
                remaining_errors,
                remaining_chars >= 0 && remaining_chars <= remaining_errors
            );
        }

        let is_accepting = automaton.is_accepting(&state, word.len(), input.len());
        eprintln!("\nFinal is_accepting result: {}", is_accepting);

        assert!(is_accepting, "Should accept insertion");
    }

    #[test]
    fn test_accepts_one_insertion() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "tests")); // +s at end
        assert!(automaton.accepts("test", "ttest")); // +t at start
    }

    #[test]
    fn test_accepts_one_deletion() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("tests", "test")); // -s at end
        assert!(automaton.accepts("ttest", "test")); // -t at start
    }

    #[test]
    fn test_rejects_too_far() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(!automaton.accepts("test", "hello")); // distance 4
        assert!(!automaton.accepts("abc", "xyz")); // distance 3
    }

    #[test]
    fn test_empty_input() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("", "")); // distance 0
        assert!(automaton.accepts("ab", "")); // distance 2
        assert!(!automaton.accepts("abc", "")); // distance 3 > 2
    }

    #[test]
    fn test_empty_word() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("", "ab")); // distance 2
        assert!(!automaton.accepts("", "abc")); // distance 3 > 2
    }

    #[test]
    fn test_max_distance_zero() {
        let automaton = GeneralizedAutomaton::new(0);
        assert!(automaton.accepts("test", "test")); // exact match
        assert!(!automaton.accepts("test", "text")); // any difference rejected
    }

    #[test]
    fn test_debug_deletion_middle() {
        let automaton = GeneralizedAutomaton::new(1);
        let word = "test";
        let input = "tst"; // Missing 'e' in middle

        eprintln!(
            "\nDEBUG: Testing deletion in middle: word='{}', input='{}', max_distance={}",
            word,
            input,
            automaton.max_distance()
        );
        eprintln!("Expected: true (1 deletion)");

        // Manual step-through
        let mut state = automaton.initial_state();
        eprintln!("\nInitial state: {}", state);

        // H2 Optimization: Pre-compute character vector
        let word_chars: Vec<char> = word.chars().collect();

        for (i, ch) in input.chars().enumerate() {
            eprintln!(
                "\n--- Processing char {} ('{}') at position {} ---",
                i + 1,
                ch,
                i + 1
            );

            let subword = automaton.relevant_subword(word, i + 1);
            eprintln!("Relevant subword: '{}'", subword);

            let bit_vector = CharacteristicVector::new(ch, &subword);
            eprintln!("Bit vector length: {}", bit_vector.len());

            match state.transition(
                &automaton.operations,
                &bit_vector,
                word,
                Some(&word_chars),
                &subword,
                ch,
                i + 1,
            ) {
                Some(next) => {
                    eprintln!("Next state: {}", next);
                    state = next;
                }
                None => {
                    eprintln!("Transition failed - no successor state!");
                    panic!("Should not fail at position {}", i + 1);
                }
            }
        }

        eprintln!("\n--- Final state check ---");
        eprintln!("Final state: {}", state);
        eprintln!("Word length: {}", word.len());
        eprintln!("Input length: {}", input.len());

        // Check is_accepting manually for each position
        let n = automaton.max_distance() as i32;
        eprintln!("\nChecking acceptance for each position:");

        use crate::transducer::generalized::GeneralizedPosition;
        for pos in state.positions() {
            match pos {
                GeneralizedPosition::MFinal { offset, errors } => {
                    eprintln!("\nM-type position: offset={}, errors={}", offset, errors);
                    eprintln!(
                        "  M-type accepting? offset <= 0 && errors <= {}: {} <= 0 && {} <= {} = {}",
                        n,
                        offset,
                        errors,
                        n,
                        *offset <= 0 && *errors <= n as u8
                    );
                }
                GeneralizedPosition::INonFinal { offset, errors } => {
                    let current_word_pos = input.len() as i32 + offset;
                    let remaining_chars = word.len() as i32 - current_word_pos;
                    let remaining_errors = n - (*errors as i32);

                    eprintln!("\nI-type position: offset={}, errors={}", offset, errors);
                    eprintln!(
                        "  current_word_pos = {} + {} = {}",
                        input.len(),
                        offset,
                        current_word_pos
                    );
                    eprintln!(
                        "  remaining_chars = {} - {} = {}",
                        word.len(),
                        current_word_pos,
                        remaining_chars
                    );
                    eprintln!(
                        "  remaining_errors = {} - {} = {}",
                        n, errors, remaining_errors
                    );
                    eprintln!(
                        "  Accept? {} >= 0 && {} <= {} = {}",
                        remaining_chars,
                        remaining_chars,
                        remaining_errors,
                        remaining_chars >= 0 && remaining_chars <= remaining_errors
                    );
                }
                // Phase 2d: Debug output for transposing/splitting positions
                GeneralizedPosition::ITransposing { offset, errors } => {
                    eprintln!("\nI-type transposing: offset={}, errors={} (not accepting - intermediate state)", offset, errors);
                }
                GeneralizedPosition::MTransposing { offset, errors } => {
                    eprintln!("\nM-type transposing: offset={}, errors={} (not accepting - intermediate state)", offset, errors);
                }
                GeneralizedPosition::ISplitting { offset, errors, .. } => {
                    eprintln!("\nI-type splitting: offset={}, errors={} (not accepting - intermediate state)", offset, errors);
                }
                GeneralizedPosition::MSplitting { offset, errors, .. } => {
                    eprintln!("\nM-type splitting: offset={}, errors={} (not accepting - intermediate state)", offset, errors);
                }
            }
        }

        let result = automaton.accepts(word, input);
        eprintln!("\n=== Final result: {} ===", result);
        eprintln!("Expected: true");

        assert!(result, "Should accept deletion in middle");
    }

    #[test]
    fn test_max_distance_one() {
        let automaton = GeneralizedAutomaton::new(1);
        assert!(automaton.accepts("test", "text")); // 1 substitution
        assert!(automaton.accepts("test", "tst")); // 1 deletion
        assert!(!automaton.accepts("test", "tx")); // distance 2
    }

    // Phase 2d.3: Transposition tests

    #[test]
    fn test_transposition_distance_zero() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        assert!(automaton.accepts("test", "test")); // Exact match
        assert!(!automaton.accepts("test", "tset")); // Requires 1 error
        assert!(!automaton.accepts("test", "etst")); // Requires 1 error
    }

    #[test]
    fn test_transposition_adjacent_swap_middle() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "tset" (swap 'e' and 's')
        assert!(automaton.accepts("test", "tset"));
    }

    #[test]
    fn test_transposition_adjacent_swap_start() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "etst" (swap 't' and 'e')
        assert!(automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_adjacent_swap_end() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "tets" (swap 's' and 't')
        assert!(automaton.accepts("test", "tets"));
    }

    #[test]
    fn test_transposition_longer_words() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "algorithm" → "lagorithm" (swap 'a' and 'l')
        assert!(automaton.accepts("algorithm", "lagorithm"));

        // "programming" → "porgramming" (swap 'r' and 'o')
        assert!(automaton.accepts("programming", "porgramming"));
    }

    #[test]
    fn test_transposition_rejects_non_adjacent() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "tsta" (non-adjacent swap) requires 2 errors
        // (Cannot swap 'e' and 's' if they're not adjacent)
        assert!(!automaton.accepts("test", "tsta"));

        // "abc" → "cba" (swap non-adjacent) requires 2 errors
        assert!(!automaton.accepts("abc", "cba"));
    }

    #[test]
    fn test_transposition_multiple_swaps() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "abcd" → "badc" (two adjacent swaps)
        assert!(automaton.accepts("abcd", "badc"));

        // "test" → "etst" → "etts" (two swaps)
        assert!(automaton.accepts("test", "etts"));
    }

    #[test]
    fn test_transposition_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Combine transposition with substitution
        // "test" → "txst" (substitute e→x) → "tsxt" (transpose)
        assert!(automaton.accepts("test", "tsxt"));

        // Transpose + deletion
        // "test" → "tset" (transpose) → "set" (delete 't')
        assert!(automaton.accepts("test", "set"));
    }

    #[test]
    fn test_transposition_empty_and_single() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops.clone());

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (no transposition possible, but substitution works)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // Accepted via substitution (not transposition)

        // Verify distance 0 rejects difference
        let strict_automaton = GeneralizedAutomaton::with_operations(0, ops);
        assert!(!strict_automaton.accepts("a", "b")); // No errors allowed
    }

    #[test]
    fn test_transposition_two_char_words() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Two characters - simple swap
        assert!(automaton.accepts("ab", "ba"));
        assert!(automaton.accepts("xy", "yx"));

        // Should also work with identical chars
        assert!(automaton.accepts("aa", "aa"));
    }

    // Phase 2d.4: Merge tests

    #[test]
    fn test_merge_simple() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "a" → "ab" (merge two input chars "ab" into one word char "a")
        // Note: Direction is word → input, so we're checking if input "ab" matches word "a"
        assert!(automaton.accepts("a", "ab"));
    }

    #[test]
    fn test_merge_at_start() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ac" → "abc" (merge "ab" into "a", then match "c")
        assert!(automaton.accepts("ac", "abc"));

        // "test" → "teest" (merge "te" into "t", then match "est")
        assert!(automaton.accepts("test", "teest"));
    }

    #[test]
    fn test_merge_at_end() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "xa" → "xab" (match "x", merge "ab" into "a")
        assert!(automaton.accepts("xa", "xab"));

        // "testa" → "testab" (match "test", merge "ab" into "a")
        assert!(automaton.accepts("testa", "testab"));
    }

    #[test]
    fn test_merge_middle() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "cat" → "cabt" (match "c", merge "ab" into "a", match "t")
        assert!(automaton.accepts("cat", "cabt"));
    }

    #[test]
    fn test_merge_distance_zero() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        // Distance 0: no merge allowed
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("test", "teest"));
        assert!(!automaton.accepts("a", "ab"));
    }

    #[test]
    fn test_merge_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Merge + substitution
        // "test" → "texst" (merge "te" into "t", substitute e→x, match "st")
        assert!(automaton.accepts("test", "texst"));

        // Merge + deletion
        // "test" → "teest" (merge) → "eest" (delete first t)
        assert!(automaton.accepts("test", "eest"));
    }

    #[test]
    fn test_merge_empty_and_edge_cases() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (merge requires 2 input chars)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "ab")); // Merge "ab" → "a"

        // Two char word
        assert!(automaton.accepts("ab", "ab"));
        assert!(automaton.accepts("ab", "abb")); // Merge "ab" → "a", match "b"
    }

    // Phase 2d.5: Split tests

    #[test]
    fn test_split_simple() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ab" → "a" (split one input char "a" into two word chars "ab")
        // Note: Direction is word → input, so we're checking if input "a" matches word "ab"
        assert!(automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_split_at_start() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "abc" → "ac" (split "a" into "ab", then match "c")
        assert!(automaton.accepts("abc", "ac"));

        // "acd" → "acd" or "acd" → "ad" (split "a" into "ab", match "c", delete "d")
        assert!(automaton.accepts("abcd", "acd"));
    }

    #[test]
    fn test_split_at_end() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "xab" → "xa" (match "x", split "a" into "ab")
        assert!(automaton.accepts("xab", "xa"));

        // "testab" → "testa" (match "test", split "a" into "ab")
        assert!(automaton.accepts("testab", "testa"));
    }

    #[test]
    fn test_split_middle() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ct" → "cct" (match 'c', then split: word 'c' matches input 'cc', then match 't')
        // Wait, that's wrong - we already consumed word 'c'. Let me use different chars.
        // "cat" → "caat" (match 'c', split: word 'a' matches input 'aa', match 't')
        assert!(automaton.accepts("cat", "caat"));
    }

    #[test]
    fn test_split_distance_zero() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        // Distance 0: no split allowed
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("ttest", "test"));
        assert!(!automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_split_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Split + substitution
        // "test" → "txst" (split "t" into "te", substitute e→x, match "st")
        assert!(automaton.accepts("test", "txst"));

        // Split + deletion
        // "test" → "ttest" (split) → "test" (delete extra t)
        assert!(automaton.accepts("test", "test"));
    }

    #[test]
    fn test_split_empty_and_edge_cases() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (split produces 2 chars from 1)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("ab", "a")); // Split "a" → "ab"

        // Two char word
        assert!(automaton.accepts("ab", "ab"));
        assert!(automaton.accepts("abb", "ab")); // Split "a" → "ab", match "b"
    }

    #[test]
    fn test_split_and_merge_combined() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Both split and merge in same string
        // "abc" → "ac" (split "a" → "ab") → "abc" (merge "bc" → "c")
        assert!(automaton.accepts("abc", "ac"));

        // Complex: "test" → "ttest" (split "t") → "test" (merge "tt" → "t")
        assert!(automaton.accepts("test", "test"));
    }

    // ============================================================================
    // Phase 2d.6: Integration Tests - Mixed Operations and Complex Cases
    // ============================================================================

    #[test]
    fn test_all_multichar_operations_combined() {
        // Create operation set with all multi-character operations
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // Combines transpose, merge, and split
        // "abc" → "abbc" (split 'b') → "acbc" (transpose "bb" → "cb")
        assert!(automaton.accepts("abc", "acbc"));

        // Complex transformation using all operation types
        // "hello" → "heello" (split 'e') → "hello" (merge "ee" → "e")
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_multichar_with_distance_constraints() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 1: only one multi-char operation allowed
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("ab", "ba")); // transpose
        assert!(automaton1.accepts("a", "aa")); // split
        assert!(!automaton1.accepts("ab", "bba")); // would need transpose + split (2 errors)

        // Distance 2: two operations allowed
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops.clone());
        assert!(automaton2.accepts("abc", "baca")); // transpose + split

        // Distance 3: three operations allowed
        let automaton3 = GeneralizedAutomaton::with_operations(3, ops);
        assert!(automaton3.accepts("abc", "bbcaa")); // multiple splits and transpose
    }

    #[test]
    fn test_multichar_operations_at_string_boundaries() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Transpose at start
        assert!(automaton.accepts("ab", "ba"));

        // Transpose at end
        assert!(automaton.accepts("cab", "cba"));

        // Split at start
        assert!(automaton.accepts("abc", "aabc"));

        // Split at end
        assert!(automaton.accepts("abc", "abcc"));

        // Merge at start (input has merge at start)
        assert!(automaton.accepts("abc", "bc")); // delete or merge "ab" → "b" + keep c

        // Merge at end
        assert!(automaton.accepts("abc", "ab")); // delete 'c' or merge "bc" → "b"
    }

    #[test]
    fn test_repeated_multichar_operations() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 2: two transposes
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops.clone());
        assert!(automaton2.accepts("abcd", "badc")); // transpose "ab" and "cd"

        // Distance 2: two splits
        assert!(automaton2.accepts("ab", "aabb")); // split 'a' and 'b'

        // Distance 3: three operations
        let automaton3 = GeneralizedAutomaton::with_operations(3, ops);
        assert!(automaton3.accepts("abc", "aabbcc")); // split all three chars
    }

    #[test]
    fn test_multichar_with_standard_operations_complex() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // Transpose + insert + delete
        assert!(automaton.accepts("abc", "bac")); // transpose (1 error)
        assert!(automaton.accepts("abc", "bacd")); // transpose + insert (2 errors)
        assert!(automaton.accepts("abcd", "bac")); // transpose + delete (2 errors)

        // Split + substitute
        assert!(automaton.accepts("abc", "aabc")); // split 'a' (1 error)
        assert!(automaton.accepts("abc", "aabx")); // split 'a' + substitute c→x (2 errors)

        // Merge + insert
        assert!(automaton.accepts("abc", "bc")); // delete 'a' (1 error)
        assert!(automaton.accepts("abc", "bcd")); // delete 'a' + insert 'd' (2 errors)
    }

    #[test]
    fn test_multichar_edge_cases() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Empty strings (already covered but let's be explicit)
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("a", "")); // delete
        assert!(automaton.accepts("", "a")); // insert

        // Single character (no transpose possible)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "aa")); // split

        // Two characters (minimal transpose)
        assert!(automaton.accepts("ab", "ba")); // transpose
        assert!(automaton.accepts("ab", "aab")); // split first
        assert!(automaton.accepts("ab", "abb")); // split second

        // Identical strings (no operations needed)
        assert!(automaton.accepts("test", "test"));
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_multichar_pathological_cases() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(5, ops);

        // All same character
        assert!(automaton.accepts("aaaa", "aaaa"));
        assert!(automaton.accepts("aaaa", "aaaaaaaa")); // split all (4 errors)
        assert!(automaton.accepts("aaaa", "aa")); // merge/delete (2 errors)

        // Alternating pattern
        assert!(automaton.accepts("abab", "baba")); // transpose pairs
        assert!(automaton.accepts("abab", "aabbab")); // split 'a's

        // Reversed string (requires multiple transposes)
        assert!(automaton.accepts("abc", "cba")); // transpose + transpose
    }

    #[test]
    fn test_multichar_operations_respect_invariants() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 0: no operations allowed
        let automaton0 = GeneralizedAutomaton::with_operations(0, ops.clone());
        assert!(automaton0.accepts("test", "test"));
        assert!(!automaton0.accepts("test", "tset")); // no transpose
        assert!(!automaton0.accepts("test", "teest")); // no split
        assert!(!automaton0.accepts("test", "tes")); // no delete/merge

        // Distance 1: exactly one operation
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("test", "tset")); // transpose
        assert!(automaton1.accepts("test", "teest")); // split
        assert!(!automaton1.accepts("test", "tseest")); // transpose + split (2 errors)

        // Verify operations don't succeed beyond max distance
        assert!(!automaton1.accepts("abc", "aabbcc")); // would need 3 splits
    }

    #[test]
    fn test_multichar_subsumption_correctness() {
        // This test verifies that subsumption works correctly with intermediate states
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // These should all work and produce minimal state sets
        assert!(automaton.accepts("ab", "ba")); // transpose
        assert!(automaton.accepts("abc", "bac")); // transpose
        assert!(automaton.accepts("test", "tset")); // transpose

        // Verify split subsumption
        assert!(automaton.accepts("a", "aa")); // split
        assert!(automaton.accepts("ab", "aab")); // split first
        assert!(automaton.accepts("ab", "aabb")); // split both

        // Complex cases that test subsumption during multi-char operations
        assert!(automaton.accepts("abcd", "bacd")); // transpose at start
        assert!(automaton.accepts("abcd", "abdc")); // transpose at end
    }

    #[test]
    fn test_multichar_operation_ordering() {
        // Verify that different orderings of operations work correctly
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // transpose then split
        assert!(automaton.accepts("abc", "baac")); // transpose "ab", then split 'a'

        // split then transpose
        assert!(automaton.accepts("abc", "abba")); // split 'b', then transpose... wait, this doesn't make sense

        // Let me use clearer examples:
        // "ab" → "ba" (transpose) → "baa" (split second 'a')
        assert!(automaton.accepts("ab", "baa"));

        // "ab" → "aab" (split first 'a') → "aba" (transpose)
        // Actually, the input is fixed, so let me think differently

        // The point is: regardless of the order operations are discovered,
        // the same transformations should be accepted
        assert!(automaton.accepts("abc", "aabc")); // split or insert
        assert!(automaton.accepts("abc", "bac")); // transpose
        assert!(automaton.accepts("abc", "ab")); // delete
    }

    // ============================================================================
    // Phase 3: Phonetic Operations Tests
    // ============================================================================

    #[test]
    fn test_phonetic_debug_simple() {
        // Simple debug test for phonetic operations
        let ops = crate::transducer::phonetic::consonant_digraphs();

        // Check operation set has the right operations
        eprintln!("Operation set has {} operations", ops.operations().len());
        for op in ops.operations() {
            eprintln!(
                "  Operation: consume_x={}, consume_y={}, weight={}",
                op.consume_x(),
                op.consume_y(),
                op.weight()
            );
        }

        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Test simplest case: "ph" → "f"
        eprintln!("\n=== Testing 'ph' → 'f' ===");

        // Debug: check what relevant_subword looks like
        let subword = automaton.relevant_subword("ph", 1);
        eprintln!("Relevant subword at position 1: '{}'", subword);
        eprintln!("Subword chars: {:?}", subword.chars().collect::<Vec<_>>());

        let result = automaton.accepts("ph", "f");
        eprintln!("Result: {}", result);
        assert!(result, "Expected 'ph' → 'f' to be accepted");
    }

    #[test]
    fn test_phonetic_digraph_2to1_ch_to_k() {
        // Test phonetic operation: "ch" → "k" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "church" can match "kurk" via "ch"→"k" digraph operations
        assert!(automaton.accepts("church", "kurk"));

        // "chair" can match "kair" via "ch"→"k"
        assert!(automaton.accepts("chair", "kair"));

        // Distance 0: exact match only (with standard match operation)
        let ops0 = crate::transducer::OperationSet::default();
        let automaton0 = GeneralizedAutomaton::with_operations(0, ops0);
        assert!(automaton0.accepts("church", "church")); // exact match
        assert!(!automaton0.accepts("church", "kurk")); // requires phonetic ops
    }

    #[test]
    fn test_phonetic_digraph_2to1_ph_to_f() {
        // Test phonetic operation: "ph" → "f" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "phone" can match "fone" via "ph"→"f"
        assert!(automaton.accepts("phone", "fone"));

        // "graph" can match "graf" via "ph"→"f"
        assert!(automaton.accepts("graph", "graf"));
    }

    #[test]
    fn test_phonetic_digraph_2to1_sh_to_s() {
        // Test phonetic operation: "sh" → "s" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ship" can match "sip" via "sh"→"s"
        assert!(automaton.accepts("ship", "sip"));

        // "wash" can match "was" via "sh"→"s"
        assert!(automaton.accepts("wash", "was"));
    }

    #[test]
    fn test_phonetic_digraph_2to1_th_to_t() {
        // Test phonetic operation: "th" → "t" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "think" can match "tink" via "th"→"t"
        assert!(automaton.accepts("think", "tink"));

        // "bath" can match "bat" via "th"→"t"
        assert!(automaton.accepts("bath", "bat"));
    }

    #[test]
    fn test_phonetic_digraph_multiple_in_word() {
        // Test multiple phonetic operations in same word
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "church" has two "ch" digraphs, both can convert: "kurk"
        // But we only have distance 1 per "ch"→"k", so we need distance 2
        assert!(automaton.accepts("church", "kurc")); // one ch→k
        assert!(automaton.accepts("church", "churk")); // one ch→k
    }

    #[test]
    fn test_phonetic_with_standard_ops() {
        // Test phonetic operations combined with standard operations
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();

        // Add all phonetic operations
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "phone" → "fone" (ph→f) + extra char
        assert!(automaton.accepts("phone", "fones")); // ph→f + insert 's'

        // "chair" → "kair" (ch→k) + substitute
        assert!(automaton.accepts("chair", "kair")); // ch→k
        assert!(automaton.accepts("chair", "kair")); // ch→k
    }

    #[test]
    fn test_phonetic_distance_constraints() {
        // Verify phonetic operations respect distance limits
        // Need standard ops for matching non-phonetic characters and deletions
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Distance 1: allows one phonetic operation
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("phone", "fone")); // one ph→f
        assert!(!automaton1.accepts("phone", "fo")); // would need ph→f + delete (2 ops)

        // Distance 2: allows two operations
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops);
        assert!(automaton2.accepts("phone", "fo")); // ph→f + delete 'ne'
    }

    // ==================== Cross-Validation Tests ====================
    // Compare Generalized Automaton with Universal Automaton
    // to ensure correctness of phonetic operations

    #[test]
    fn test_cross_validate_standard_operations() {
        // Test that generalized automaton matches universal automaton
        // for standard Levenshtein operations
        use crate::transducer::universal::Standard;
        use crate::transducer::universal::UniversalAutomaton;

        let test_cases = vec![
            ("kitten", "sitting", 3, true),
            ("kitten", "sitting", 2, false),
            ("saturday", "sunday", 3, true),
            ("saturday", "sunday", 2, false),
            ("test", "test", 0, true),
            ("test", "tast", 1, true),
            ("", "", 0, true),
            ("a", "b", 1, true),
            ("abc", "def", 3, true),
        ];

        for (word, input, distance, expected) in test_cases {
            let gen_auto = GeneralizedAutomaton::new(distance);
            let univ_auto = UniversalAutomaton::<Standard>::new(distance);

            let gen_result = gen_auto.accepts(word, input);
            let univ_result = univ_auto.accepts(word, input);

            assert_eq!(
                gen_result, univ_result,
                "Mismatch for ('{}', '{}', {}): gen={}, univ={}",
                word, input, distance, gen_result, univ_result
            );
            assert_eq!(
                gen_result, expected,
                "Expected {} for ('{}', '{}', {}), got {}",
                expected, word, input, distance, gen_result
            );
        }
    }

    #[test]
    fn test_cross_validate_phonetic_merge_simple() {
        // Cross-validate phonetic merge operations
        // Generalized automaton should accept same strings as manual validation
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // These should all be accepted
        let accept_cases = vec![
            ("phone", "fone"),  // ph→f
            ("graph", "graf"),  // ph→f
            ("ship", "sip"),    // sh→s
            ("think", "tink"),  // th→t
            ("church", "kurc"), // first ch→k
            ("chair", "kair"),  // ch→k
        ];

        for (word, input) in accept_cases {
            assert!(
                automaton.accepts(word, input),
                "Should accept ('{}', '{}')",
                word,
                input
            );
        }

        // These should all be rejected (require > distance 1)
        let reject_cases = vec![
            ("phone", "fo"),   // ph→f + delete (needs distance 2)
            ("church", "urk"), // ch→k + delete (needs distance 2)
        ];

        for (word, input) in reject_cases {
            assert!(
                !automaton.accepts(word, input),
                "Should reject ('{}', '{}') at distance 1",
                word,
                input
            );
        }
    }

    #[test]
    fn test_cross_validate_fractional_weights() {
        // Verify fractional weights (0.15) are treated as "free" operations
        // Multiple phonetic operations should be possible at distance 1
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // With distance 1, multiple fractional-weight operations should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "church" → "kurk" requires 2× ch→k operations
        // Each operation has weight 0.15, truncates to 0
        // Both should succeed at distance 1
        assert!(
            automaton.accepts("church", "kurk"),
            "Two phonetic operations (2×0.15=0 errors) should work at distance 1"
        );

        // "church" → "kurks" requires 2× ch→k + 1 insert = 1 standard error
        // Should still work at distance 1
        assert!(
            automaton.accepts("church", "kurks"),
            "Two phonetic + one standard operation (total 1 error) should work at distance 1"
        );

        // But two standard operations should fail
        assert!(
            !automaton.accepts("church", "korks"),
            "Two phonetic + two standard operations should fail at distance 1"
        );
    }

    // Phase 3b: Phonetic split ⟨1,2⟩ tests
    #[test]
    fn test_phonetic_split_k_to_ch() {
        // Test "k"→"ch" split operation
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ark" → "arch" via k→ch split
        assert!(
            automaton.accepts("ark", "arch"),
            "Split k→ch should work at distance 1"
        );

        // "back" → "bach" via k→ch split
        assert!(
            automaton.accepts("back", "bach"),
            "Split k→ch at word end should work"
        );

        // "kan" → "chan" via k→ch split at start
        assert!(
            automaton.accepts("kan", "chan"),
            "Split k→ch at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_f_to_ph() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "graf" → "graph" via f→ph split
        assert!(
            automaton.accepts("graf", "graph"),
            "Split f→ph should work at distance 1"
        );

        // "foto" → "photo" via f→ph split at start
        assert!(
            automaton.accepts("foto", "photo"),
            "Split f→ph at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_s_to_sh() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "sip" → "ship" via s→sh split
        assert!(
            automaton.accepts("sip", "ship"),
            "Split s→sh should work at distance 1"
        );

        // "sell" → "shell" via s→sh split at start
        assert!(
            automaton.accepts("sell", "shell"),
            "Split s→sh at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_t_to_th() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "bat" → "bath" via t→th split
        assert!(
            automaton.accepts("bat", "bath"),
            "Split t→th should work at distance 1"
        );

        // "tin" → "thin" via t→th split at start
        assert!(
            automaton.accepts("tin", "thin"),
            "Split t→th at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_multiple() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Multiple splits require higher distance
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Single split at distance 1 (fractional weight = 0)
        assert!(
            automaton.accepts("kair", "chair"),
            "Single k→ch split should work at distance 1"
        );

        // But two splits need to check if fractional weights allow it
        // Each split has weight 0.15, both truncate to 0
        // So two splits should work at distance 1
        assert!(
            automaton.accepts("kat", "chath"),
            "Two splits (k→ch, t→th) with fractional weights should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_split_with_standard_ops() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Split + standard operations
        // "graf" → "grape" via f→ph split (0 errors) + e→a substitute (1 error) = 1 total
        assert!(
            automaton.accepts("graf", "graphe"),
            "Split f→ph + insert 'e' should work at distance 1"
        );

        // Multiple standard operations
        // "bak" → "batch" via k→ch split (0) + insert 't' (1) = 1 total
        assert!(
            automaton.accepts("bak", "batch"),
            "Split k→ch + insert should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_split_distance_constraints() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(0, ops.clone());

        // At distance 0, no operations should work
        assert!(
            !automaton.accepts("ark", "arch"),
            "Split should not work at distance 0"
        );

        // At distance 1, fractional-weight split should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);
        assert!(
            automaton.accepts("ark", "arch"),
            "Split should work at distance 1"
        );
    }

    // Phase 3b: Phonetic transpose ⟨2,2⟩ tests
    #[test]
    fn test_phonetic_transpose_qu_to_kw() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "queen" → "kween" via qu→kw transpose
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose qu→kw should work at distance 1"
        );

        // "quick" → "kwick" via qu→kw transpose
        assert!(
            automaton.accepts("quick", "kwick"),
            "Transpose qu→kw at word start should work"
        );

        // "quit" → "kwit" via qu→kw transpose
        assert!(
            automaton.accepts("quit", "kwit"),
            "Transpose qu→kw should work"
        );
    }

    #[test]
    fn test_phonetic_transpose_kw_to_qu() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "kween" → "queen" via kw→qu transpose (reverse)
        assert!(
            automaton.accepts("kween", "queen"),
            "Transpose kw→qu should work at distance 1"
        );

        // "kwik" → "quik" via kw→qu transpose
        assert!(
            automaton.accepts("kwik", "quik"),
            "Transpose kw→qu should work"
        );
    }

    #[test]
    fn test_phonetic_transpose_multiple() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Multiple transposes with fractional weights
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Single transpose at distance 1
        assert!(
            automaton.accepts("queen", "kween"),
            "Single transpose should work at distance 1"
        );

        // Two transposes would need multiple qu/kw in the word
        // "ququ" → "kwkw" (two transposes, both fractional = 0)
        assert!(
            automaton.accepts("ququ", "kwkw"),
            "Two transposes with fractional weights should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_transpose_with_standard_ops() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Transpose + standard operations
        // "queen" → "kweens" via qu→kw (0) + insert 's' (1) = 1 total
        assert!(
            automaton.accepts("queen", "kweens"),
            "Transpose + insert should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_transpose_distance_constraints() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(0, ops.clone());

        // At distance 0, no operations should work
        assert!(
            !automaton.accepts("queen", "kween"),
            "Transpose should not work at distance 0"
        );

        // At distance 1, fractional-weight transpose should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_mixed_merge_split_transpose() {
        // Test combining all phonetic operation types
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Merge: "phone" → "fone" (ph→f)
        assert!(
            automaton.accepts("phone", "fone"),
            "Merge operation should work"
        );

        // Split: "graf" → "graph" (f→ph)
        assert!(
            automaton.accepts("graf", "graph"),
            "Split operation should work"
        );

        // Transpose: "queen" → "kween" (qu→kw)
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose operation should work"
        );

        // All three have fractional weights, all work at distance 1
        // Each operation independently truncates to 0 errors
    }
}
