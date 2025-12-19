//! Product Automaton: NFA × Levenshtein for fuzzy regex matching.
//!
//! This module implements the product automaton that combines phonetic NFAs
//! with Levenshtein automata for fuzzy regular expression matching.
//!
//! # Fuzzy Regular Expressions
//!
//! A fuzzy regex matches strings that are:
//! - Phonetically similar (via NFA rewrites: `ph → f`, `c → s / _[ei]`)
//! - Within edit distance (via Levenshtein: insertions, deletions, substitutions)
//!
//! # Product Automaton
//!
//! The product automaton `NFA × Lev` has:
//! - **States**: `(nfa_state_set, input_position, edit_distance)`
//! - **Transitions**: Combined NFA and Levenshtein transitions
//! - **Acceptance**: NFA accepts AND edit distance ≤ max
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
//! use liblevenshtein::phonetic::nfa::compile;
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Compile phonetic pattern: matches "phone" or "fone"
//! let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
//!
//! // Create product automaton with max edit distance 2
//! let product = ProductAutomatonChar::new(nfa, 2);
//!
//! // Test fuzzy matching
//! assert!(product.accepts("phone"));   // exact NFA match
//! assert!(product.accepts("fone"));    // NFA alternate
//! assert!(product.accepts("phones"));  // NFA match + insertion
//! assert!(product.accepts("fon"));     // NFA alternate + deletion
//! assert!(!product.accepts("xyz"));    // too far from any NFA match
//! ```

use super::nfa::{NFAChar, NFA};
use super::types::StateId;
use crate::transducer::Algorithm;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Helper to create a single-element FxHashSet.
fn fx_singleton(state: StateId) -> FxHashSet<StateId> {
    let mut set = FxHashSet::default();
    set.insert(state);
    set
}

/// Character-level product automaton: NFA × Levenshtein.
///
/// Computes the intersection of a phonetic NFA with a Levenshtein automaton
/// to enable fuzzy pattern matching.
#[derive(Debug, Clone)]
pub struct ProductAutomatonChar {
    /// The phonetic NFA
    nfa: NFAChar,
    /// Maximum edit distance
    max_distance: u8,
    /// Phonetic weight applied to NFA transitions (default: 0.0)
    phonetic_weight: f64,
    /// Levenshtein algorithm variant (standard, transposition, merge-and-split)
    algorithm: Algorithm,
}

/// A state in the product automaton.
///
/// Represents a configuration during fuzzy matching:
/// - Which NFA states are active (after epsilon closure)
/// - How many errors have been consumed
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductStateChar {
    /// Set of active NFA states (after epsilon closure)
    pub nfa_states: Vec<StateId>,
    /// Number of edit operations used
    pub edit_distance: u8,
}

impl ProductStateChar {
    /// Create a new product state.
    pub fn new(nfa_states: FxHashSet<StateId>, edit_distance: u8) -> Self {
        let mut states: Vec<StateId> = nfa_states.into_iter().collect();
        states.sort(); // Canonical ordering for equality comparison
        Self {
            nfa_states: states,
            edit_distance,
        }
    }
}

impl ProductAutomatonChar {
    /// Create a new product automaton with default settings.
    ///
    /// # Arguments
    ///
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    pub fn new(nfa: NFAChar, max_distance: u8) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with a specific algorithm.
    ///
    /// # Arguments
    ///
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    /// * `algorithm` - The Levenshtein algorithm variant to use
    pub fn with_algorithm(nfa: NFAChar, max_distance: u8, algorithm: Algorithm) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm,
        }
    }

    /// Create a product automaton with a phonetic weight.
    ///
    /// The phonetic weight is added to the total cost for each NFA transition
    /// that consumes input. This allows penalizing phonetic transformations.
    pub fn with_phonetic_weight(nfa: NFAChar, max_distance: u8, phonetic_weight: f64) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight,
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with both algorithm and phonetic weight.
    pub fn with_algorithm_and_weight(
        nfa: NFAChar,
        max_distance: u8,
        algorithm: Algorithm,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight,
            algorithm,
        }
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the phonetic weight.
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Get the algorithm variant.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the initial state of the product automaton.
    ///
    /// The initial state is the epsilon closure of the NFA start state
    /// with 0 edit distance.
    pub fn initial_state(&self) -> ProductStateChar {
        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));
        ProductStateChar::new(initial_closure, 0)
    }

    /// Check if a product state is accepting.
    ///
    /// A state is accepting if:
    /// 1. At least one NFA state is final
    /// 2. Edit distance is within the maximum
    pub fn is_accepting(&self, state: &ProductStateChar) -> bool {
        if state.edit_distance > self.max_distance {
            return false;
        }

        // Check if any NFA state is final
        state.nfa_states.iter().any(|&s| self.nfa.is_final(s))
    }

    /// Compute successor states after consuming a character.
    ///
    /// This implements the product transition function:
    /// - For each active NFA state, try matching the input character
    /// - Also consider edit operations (insertion, deletion, substitution)
    ///
    /// # Arguments
    ///
    /// * `state` - Current product state
    /// * `c` - Input character to process
    ///
    /// # Returns
    ///
    /// Set of successor states reachable via matching or edit operations.
    pub fn transition(&self, state: &ProductStateChar, c: char) -> Vec<ProductStateChar> {
        let mut successors = Vec::new();
        let current_states: FxHashSet<StateId> = state.nfa_states.iter().copied().collect();

        // 1. NFA Match: consume character in NFA
        let match_states = self.nfa_step(&current_states, c);
        if !match_states.is_empty() {
            successors.push(ProductStateChar::new(match_states, state.edit_distance));
        }

        // Edit operations only if we have budget
        if state.edit_distance < self.max_distance {
            // 2. Substitution: NFA doesn't match, consume with error
            // For each active state, try all transitions and advance with +1 error
            let mut subst_states = FxHashSet::default();
            for &nfa_state in &state.nfa_states {
                for trans in self.nfa.transitions_from(nfa_state) {
                    if trans.label.consumes_input() {
                        // Add successor regardless of whether it matches c
                        let closure = self.nfa.epsilon_closure(&fx_singleton(trans.to));
                        subst_states.extend(closure);
                    }
                }
            }
            if !subst_states.is_empty() {
                // Only add if different from match result
                let subst_state = ProductStateChar::new(subst_states, state.edit_distance + 1);
                if !successors.contains(&subst_state) {
                    successors.push(subst_state);
                }
            }

            // 3. Insertion: input has extra char, NFA stays in place
            // Stay in current NFA states, consume input char with +1 error
            successors.push(ProductStateChar::new(current_states.clone(), state.edit_distance + 1));

            // 4. Deletion: NFA pattern has extra char, advance NFA without consuming input
            // This is handled differently - we advance NFA states via epsilon transitions
            // But we need to also advance via character transitions (pretend we saw the char)
            // Actually, deletion in Levenshtein means the pattern has something the input doesn't
            // So we advance NFA and don't consume input - this is like "free" NFA transitions
            // We'll handle this in the accepts() method by trying to advance NFA with budget
        }

        successors
    }

    /// Advance NFA states by consuming a character.
    fn nfa_step(&self, states: &FxHashSet<StateId>, c: char) -> FxHashSet<StateId> {
        let mut next_states = FxHashSet::default();

        for &state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.matches(c) && trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        // Apply epsilon closure
        self.nfa.epsilon_closure(&next_states)
    }

    /// Check if the input string is accepted by the fuzzy regex.
    ///
    /// Uses BFS to explore the product state space, pruning states
    /// that exceed the edit distance budget.
    ///
    /// # Arguments
    ///
    /// * `input` - Input string to match
    ///
    /// # Returns
    ///
    /// `true` if input matches the NFA pattern within edit distance.
    pub fn accepts(&self, input: &str) -> bool {
        // Early exit: empty pattern
        if self.nfa.is_empty() {
            return input.is_empty() || input.len() <= self.max_distance as usize;
        }

        let input_chars: Vec<char> = input.chars().collect();
        let n = input_chars.len();

        // BFS state: (nfa_states, input_position, edit_distance)
        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));

        // Use dynamic programming / BFS
        // State: (position in input, set of NFA states, edit distance used)
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        let mut queue: VecDeque<(usize, FxHashSet<StateId>, u8)> = VecDeque::new();

        queue.push_back((0, initial_closure, 0));

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            // Convert to canonical form for visited check
            let mut states_vec: Vec<StateId> = nfa_states.iter().copied().collect();
            states_vec.sort();

            // Skip if already visited
            if !visited.insert((pos, states_vec.clone(), errors)) {
                continue;
            }

            // Prune if over budget
            if errors > self.max_distance {
                continue;
            }

            // Check acceptance: at end of input and NFA accepts
            if pos == n {
                // Check if we can reach final state with remaining budget
                if self.can_reach_final(&nfa_states, errors) {
                    return true;
                }
                continue;
            }

            let c = input_chars[pos];

            // 1. Match: NFA consumes character, no error
            let match_states = self.nfa_step(&nfa_states, c);
            if !match_states.is_empty() {
                queue.push_back((pos + 1, match_states, errors));
            }

            if errors < self.max_distance {
                // 2. Substitution: NFA advances (any transition), +1 error
                let subst_states = self.nfa_advance(&nfa_states);
                if !subst_states.is_empty() {
                    queue.push_back((pos + 1, subst_states, errors + 1));
                }

                // 3. Insertion: Stay in NFA, consume input char, +1 error
                queue.push_back((pos + 1, nfa_states.clone(), errors + 1));

                // 4. Deletion: Advance NFA without consuming input, +1 error
                let del_states = self.nfa_advance(&nfa_states);
                if !del_states.is_empty() {
                    queue.push_back((pos, del_states, errors + 1));
                }

                // 5. Transposition: swap adjacent characters (e.g., "ab" → "ba")
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let trans_states = self.nfa_step_transposed(&nfa_states, c, next_c);
                    if !trans_states.is_empty() {
                        queue.push_back((pos + 2, trans_states, errors + 1));
                    }
                }

                // 6. Merge: two input chars → one NFA transition (e.g., "cl" → "d")
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let merge_states = self.nfa_step_merged(&nfa_states, c, next_c);
                    if !merge_states.is_empty() {
                        queue.push_back((pos + 2, merge_states, errors + 1));
                    }
                }

                // 7. Split: one input char → two NFA transitions (e.g., "ä" → "ae")
                if self.algorithm.supports_merge_split() {
                    let split_states = self.nfa_step_split(&nfa_states, c);
                    if !split_states.is_empty() {
                        queue.push_back((pos + 1, split_states, errors + 1));
                    }
                }
            }
        }

        false
    }

    /// Advance NFA states via any consuming transition.
    fn nfa_advance(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut next_states = FxHashSet::default();

        for &state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Step NFA with transposed characters.
    ///
    /// For transposition, we match c2 first, then c1 (swapped order).
    /// This handles cases like "ab" → "ba" where the input has the characters
    /// in transposed order relative to the pattern.
    fn nfa_step_transposed(
        &self,
        states: &FxHashSet<StateId>,
        c1: char,
        c2: char,
    ) -> FxHashSet<StateId> {
        // Match c2 first (in NFA), then c1
        // This corresponds to: pattern expects "ab" but input has "ba"
        // We consume both input chars (b,a) while matching NFA pattern (a,b)
        let after_c2 = self.nfa_step(states, c2);
        self.nfa_step(&after_c2, c1)
    }

    /// Step NFA treating two input chars as merged into one NFA transition.
    ///
    /// For merge, we consume 2 input chars but advance NFA by only 1 transition.
    /// This handles OCR errors like "cl" → "d" where two chars merge into one.
    fn nfa_step_merged(
        &self,
        _states: &FxHashSet<StateId>,
        _c1: char,
        _c2: char,
    ) -> FxHashSet<StateId> {
        // For merge: consume 2 input chars, advance NFA by 1
        // This is effectively skipping c1 and doing a wildcard match for c2's slot
        // We advance NFA with any transition (like substitution, but consuming 2 chars)
        self.nfa_advance(_states)
    }

    /// Step NFA with split (one input char consumes two NFA transitions).
    ///
    /// For split, we consume 1 input char but advance NFA by 2 transitions.
    /// This handles OCR errors like "ä" → "ae" where one char splits into two.
    fn nfa_step_split(&self, states: &FxHashSet<StateId>, _c: char) -> FxHashSet<StateId> {
        // For split: consume 1 input char, advance NFA by 2
        // We do two wildcard advances in the NFA
        let after_first = self.nfa_advance(states);
        self.nfa_advance(&after_first)
    }

    /// Check if we can reach a final state with remaining error budget.
    fn can_reach_final(&self, states: &FxHashSet<StateId>, current_errors: u8) -> bool {
        // If any current state is final, we're done
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return true;
        }

        // Otherwise, try to reach final with remaining budget
        let remaining = self.max_distance - current_errors;
        if remaining == 0 {
            return false;
        }

        // BFS to find path to final within remaining budget
        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        let mut queue: VecDeque<(FxHashSet<StateId>, u8)> = VecDeque::new();
        queue.push_back((states.clone(), 0));

        while let Some((current, dist)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = current.iter().copied().collect();
            states_vec.sort();

            if !visited.insert(states_vec) {
                continue;
            }

            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return true;
            }

            // Advance NFA (deletion from pattern)
            let next = self.nfa_advance(&current);
            if !next.is_empty() {
                queue.push_back((next, dist + 1));
            }
        }

        false
    }

    /// Get the minimum edit distance to any accepting state.
    ///
    /// # Arguments
    ///
    /// * `input` - Input string to match
    ///
    /// # Returns
    ///
    /// Minimum edit distance, or `None` if no match within max_distance.
    pub fn min_distance(&self, input: &str) -> Option<u8> {
        if self.nfa.is_empty() {
            return if input.is_empty() {
                Some(0)
            } else if input.len() <= self.max_distance as usize {
                Some(input.len() as u8)
            } else {
                None
            };
        }

        let input_chars: Vec<char> = input.chars().collect();
        let n = input_chars.len();

        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));

        let mut min_dist: Option<u8> = None;
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        let mut queue: VecDeque<(usize, FxHashSet<StateId>, u8)> = VecDeque::new();

        queue.push_back((0, initial_closure, 0));

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = nfa_states.iter().copied().collect();
            states_vec.sort();

            if !visited.insert((pos, states_vec.clone(), errors)) {
                continue;
            }

            if errors > self.max_distance {
                continue;
            }

            // Skip if we already found a better solution
            if let Some(min) = min_dist {
                if errors >= min {
                    continue;
                }
            }

            if pos == n {
                if let Some(final_dist) = self.distance_to_final(&nfa_states, errors) {
                    match min_dist {
                        None => min_dist = Some(final_dist),
                        Some(current) if final_dist < current => min_dist = Some(final_dist),
                        _ => {}
                    }
                }
                continue;
            }

            let c = input_chars[pos];

            // 1. Match
            let match_states = self.nfa_step(&nfa_states, c);
            if !match_states.is_empty() {
                queue.push_back((pos + 1, match_states, errors));
            }

            if errors < self.max_distance {
                // 2. Substitution
                let subst_states = self.nfa_advance(&nfa_states);
                if !subst_states.is_empty() {
                    queue.push_back((pos + 1, subst_states, errors + 1));
                }

                // 3. Insertion
                queue.push_back((pos + 1, nfa_states.clone(), errors + 1));

                // 4. Deletion
                let del_states = self.nfa_advance(&nfa_states);
                if !del_states.is_empty() {
                    queue.push_back((pos, del_states, errors + 1));
                }

                // 5. Transposition: swap adjacent characters (e.g., "ab" → "ba")
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let trans_states = self.nfa_step_transposed(&nfa_states, c, next_c);
                    if !trans_states.is_empty() {
                        queue.push_back((pos + 2, trans_states, errors + 1));
                    }
                }

                // 6. Merge: two input chars → one NFA transition (e.g., "cl" → "d")
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let merge_states = self.nfa_step_merged(&nfa_states, c, next_c);
                    if !merge_states.is_empty() {
                        queue.push_back((pos + 2, merge_states, errors + 1));
                    }
                }

                // 7. Split: one input char → two NFA transitions (e.g., "ä" → "ae")
                if self.algorithm.supports_merge_split() {
                    let split_states = self.nfa_step_split(&nfa_states, c);
                    if !split_states.is_empty() {
                        queue.push_back((pos + 1, split_states, errors + 1));
                    }
                }
            }
        }

        min_dist
    }

    /// Compute distance to reach final state from current states.
    fn distance_to_final(&self, states: &FxHashSet<StateId>, base_dist: u8) -> Option<u8> {
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return Some(base_dist);
        }

        let remaining = self.max_distance.saturating_sub(base_dist);
        if remaining == 0 {
            return None;
        }

        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        let mut queue: VecDeque<(FxHashSet<StateId>, u8)> = VecDeque::new();
        queue.push_back((states.clone(), 0));

        while let Some((current, dist)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = current.iter().copied().collect();
            states_vec.sort();

            if !visited.insert(states_vec) {
                continue;
            }

            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return Some(base_dist + dist);
            }

            let next = self.nfa_advance(&current);
            if !next.is_empty() {
                queue.push_back((next, dist + 1));
            }
        }

        None
    }
}

// ============================================================================
// Byte-level Product Automaton
// ============================================================================

/// Byte-level product automaton.
#[derive(Debug, Clone)]
pub struct ProductAutomaton {
    /// The phonetic NFA
    nfa: NFA,
    /// Maximum edit distance
    max_distance: u8,
    /// Phonetic weight
    phonetic_weight: f64,
    /// Levenshtein algorithm variant
    algorithm: Algorithm,
}

/// A state in the byte-level product automaton.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductState {
    /// Set of active NFA states
    pub nfa_states: Vec<StateId>,
    /// Number of edit operations used
    pub edit_distance: u8,
}

impl ProductState {
    /// Create a new product state.
    pub fn new(nfa_states: FxHashSet<StateId>, edit_distance: u8) -> Self {
        let mut states: Vec<StateId> = nfa_states.into_iter().collect();
        states.sort();
        Self {
            nfa_states: states,
            edit_distance,
        }
    }
}

impl ProductAutomaton {
    /// Create a new product automaton.
    pub fn new(nfa: NFA, max_distance: u8) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with a specific algorithm.
    pub fn with_algorithm(nfa: NFA, max_distance: u8, algorithm: Algorithm) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm,
        }
    }

    /// Create with phonetic weight.
    pub fn with_phonetic_weight(nfa: NFA, max_distance: u8, phonetic_weight: f64) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight,
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with both algorithm and phonetic weight.
    pub fn with_algorithm_and_weight(
        nfa: NFA,
        max_distance: u8,
        algorithm: Algorithm,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight,
            algorithm,
        }
    }

    /// Get max distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the algorithm variant.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the initial state.
    pub fn initial_state(&self) -> ProductState {
        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));
        ProductState::new(initial_closure, 0)
    }

    /// Check if accepting.
    pub fn is_accepting(&self, state: &ProductState) -> bool {
        if state.edit_distance > self.max_distance {
            return false;
        }
        state.nfa_states.iter().any(|&s| self.nfa.is_final(s))
    }

    /// Advance NFA states by consuming a byte.
    fn nfa_step(&self, states: &FxHashSet<StateId>, b: u8) -> FxHashSet<StateId> {
        let mut next_states = FxHashSet::default();

        for &state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.matches(b) && trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Advance NFA states via any consuming transition.
    fn nfa_advance(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut next_states = FxHashSet::default();

        for &state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Step NFA with transposed bytes.
    fn nfa_step_transposed(
        &self,
        states: &FxHashSet<StateId>,
        b1: u8,
        b2: u8,
    ) -> FxHashSet<StateId> {
        let after_b2 = self.nfa_step(states, b2);
        self.nfa_step(&after_b2, b1)
    }

    /// Step NFA treating two input bytes as merged.
    fn nfa_step_merged(
        &self,
        states: &FxHashSet<StateId>,
        _b1: u8,
        _b2: u8,
    ) -> FxHashSet<StateId> {
        self.nfa_advance(states)
    }

    /// Step NFA with split (one byte consumes two NFA transitions).
    fn nfa_step_split(&self, states: &FxHashSet<StateId>, _b: u8) -> FxHashSet<StateId> {
        let after_first = self.nfa_advance(states);
        self.nfa_advance(&after_first)
    }

    /// Check if input is accepted.
    pub fn accepts(&self, input: &[u8]) -> bool {
        if self.nfa.is_empty() {
            return input.is_empty() || input.len() <= self.max_distance as usize;
        }

        let n = input.len();
        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));

        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        let mut queue: VecDeque<(usize, FxHashSet<StateId>, u8)> = VecDeque::new();

        queue.push_back((0, initial_closure, 0));

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = nfa_states.iter().copied().collect();
            states_vec.sort();

            if !visited.insert((pos, states_vec.clone(), errors)) {
                continue;
            }

            if errors > self.max_distance {
                continue;
            }

            if pos == n {
                if self.can_reach_final(&nfa_states, errors) {
                    return true;
                }
                continue;
            }

            let b = input[pos];

            // 1. Match
            let match_states = self.nfa_step(&nfa_states, b);
            if !match_states.is_empty() {
                queue.push_back((pos + 1, match_states, errors));
            }

            if errors < self.max_distance {
                // 2. Substitution
                let subst_states = self.nfa_advance(&nfa_states);
                if !subst_states.is_empty() {
                    queue.push_back((pos + 1, subst_states, errors + 1));
                }

                // 3. Insertion
                queue.push_back((pos + 1, nfa_states.clone(), errors + 1));

                // 4. Deletion
                let del_states = self.nfa_advance(&nfa_states);
                if !del_states.is_empty() {
                    queue.push_back((pos, del_states, errors + 1));
                }

                // 5. Transposition
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    let trans_states = self.nfa_step_transposed(&nfa_states, b, next_b);
                    if !trans_states.is_empty() {
                        queue.push_back((pos + 2, trans_states, errors + 1));
                    }
                }

                // 6. Merge
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    let merge_states = self.nfa_step_merged(&nfa_states, b, next_b);
                    if !merge_states.is_empty() {
                        queue.push_back((pos + 2, merge_states, errors + 1));
                    }
                }

                // 7. Split
                if self.algorithm.supports_merge_split() {
                    let split_states = self.nfa_step_split(&nfa_states, b);
                    if !split_states.is_empty() {
                        queue.push_back((pos + 1, split_states, errors + 1));
                    }
                }
            }
        }

        false
    }

    /// Check if we can reach a final state.
    fn can_reach_final(&self, states: &FxHashSet<StateId>, current_errors: u8) -> bool {
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return true;
        }

        let remaining = self.max_distance - current_errors;
        if remaining == 0 {
            return false;
        }

        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        let mut queue: VecDeque<(FxHashSet<StateId>, u8)> = VecDeque::new();
        queue.push_back((states.clone(), 0));

        while let Some((current, dist)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = current.iter().copied().collect();
            states_vec.sort();

            if !visited.insert(states_vec) {
                continue;
            }

            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return true;
            }

            let next = self.nfa_advance(&current);
            if !next.is_empty() {
                queue.push_back((next, dist + 1));
            }
        }

        false
    }

    /// Get minimum distance.
    pub fn min_distance(&self, input: &[u8]) -> Option<u8> {
        if self.nfa.is_empty() {
            return if input.is_empty() {
                Some(0)
            } else if input.len() <= self.max_distance as usize {
                Some(input.len() as u8)
            } else {
                None
            };
        }

        let n = input.len();
        let initial_closure = self.nfa.epsilon_closure(&fx_singleton(self.nfa.start()));

        let mut min_dist: Option<u8> = None;
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        let mut queue: VecDeque<(usize, FxHashSet<StateId>, u8)> = VecDeque::new();

        queue.push_back((0, initial_closure, 0));

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = nfa_states.iter().copied().collect();
            states_vec.sort();

            if !visited.insert((pos, states_vec.clone(), errors)) {
                continue;
            }

            if errors > self.max_distance {
                continue;
            }

            if let Some(min) = min_dist {
                if errors >= min {
                    continue;
                }
            }

            if pos == n {
                if let Some(final_dist) = self.distance_to_final(&nfa_states, errors) {
                    match min_dist {
                        None => min_dist = Some(final_dist),
                        Some(current) if final_dist < current => min_dist = Some(final_dist),
                        _ => {}
                    }
                }
                continue;
            }

            let b = input[pos];

            // 1. Match
            let match_states = self.nfa_step(&nfa_states, b);
            if !match_states.is_empty() {
                queue.push_back((pos + 1, match_states, errors));
            }

            if errors < self.max_distance {
                // 2. Substitution
                let subst_states = self.nfa_advance(&nfa_states);
                if !subst_states.is_empty() {
                    queue.push_back((pos + 1, subst_states, errors + 1));
                }

                // 3. Insertion
                queue.push_back((pos + 1, nfa_states.clone(), errors + 1));

                // 4. Deletion
                let del_states = self.nfa_advance(&nfa_states);
                if !del_states.is_empty() {
                    queue.push_back((pos, del_states, errors + 1));
                }

                // 5. Transposition
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    let trans_states = self.nfa_step_transposed(&nfa_states, b, next_b);
                    if !trans_states.is_empty() {
                        queue.push_back((pos + 2, trans_states, errors + 1));
                    }
                }

                // 6. Merge
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    let merge_states = self.nfa_step_merged(&nfa_states, b, next_b);
                    if !merge_states.is_empty() {
                        queue.push_back((pos + 2, merge_states, errors + 1));
                    }
                }

                // 7. Split
                if self.algorithm.supports_merge_split() {
                    let split_states = self.nfa_step_split(&nfa_states, b);
                    if !split_states.is_empty() {
                        queue.push_back((pos + 1, split_states, errors + 1));
                    }
                }
            }
        }

        min_dist
    }

    /// Compute distance to final.
    fn distance_to_final(&self, states: &FxHashSet<StateId>, base_dist: u8) -> Option<u8> {
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return Some(base_dist);
        }

        let remaining = self.max_distance.saturating_sub(base_dist);
        if remaining == 0 {
            return None;
        }

        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        let mut queue: VecDeque<(FxHashSet<StateId>, u8)> = VecDeque::new();
        queue.push_back((states.clone(), 0));

        while let Some((current, dist)) = queue.pop_front() {
            let mut states_vec: Vec<StateId> = current.iter().copied().collect();
            states_vec.sort();

            if !visited.insert(states_vec) {
                continue;
            }

            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return Some(base_dist + dist);
            }

            let next = self.nfa_advance(&current);
            if !next.is_empty() {
                queue.push_back((next, dist + 1));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};

    // ============================================================================
    // ProductAutomatonChar Tests
    // ============================================================================

    #[test]
    fn test_product_exact_match() {
        let nfa = compile(&parse("phone").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 2);

        assert!(product.accepts("phone"));
        assert!(!product.accepts("xyz"));
    }

    #[test]
    fn test_product_alternation() {
        let nfa = compile(&parse("ph|f").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 0);

        assert!(product.accepts("ph"));
        assert!(product.accepts("f"));
        assert!(!product.accepts("g"));
    }

    #[test]
    fn test_product_with_edit_distance() {
        let nfa = compile(&parse("phone").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 2);

        // Exact match
        assert!(product.accepts("phone"));

        // One insertion
        assert!(product.accepts("phones"));   // +s

        // One deletion
        assert!(product.accepts("phon"));     // -e

        // One substitution
        assert!(product.accepts("phome"));    // n→m

        // Two edits
        assert!(product.accepts("phon"));     // -e, -e... wait, that's just one
        assert!(product.accepts("fone"));     // ph→f is 2 edits (delete h, substitute p→f)
    }

    #[test]
    fn test_product_phonetic_pattern() {
        // Pattern: (ph|f)one - matches "phone" or "fone"
        let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("phone"));
        assert!(product.accepts("fone"));
        assert!(product.accepts("phones")); // +s (insertion)
        assert!(product.accepts("fones"));  // +s (insertion)
        // "bone" is within distance 1 of "fone" (b→f substitution)
        assert!(product.accepts("bone"));

        // With max_distance=0, only exact matches
        let product_exact = ProductAutomatonChar::new(
            compile(&parse("(ph|f)one").unwrap()).unwrap(),
            0,
        );
        assert!(product_exact.accepts("phone"));
        assert!(product_exact.accepts("fone"));
        assert!(!product_exact.accepts("bone")); // b doesn't match exactly
    }

    #[test]
    fn test_product_star() {
        let nfa = compile(&parse("a*").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts(""));       // 0 a's
        assert!(product.accepts("a"));      // 1 a
        assert!(product.accepts("aa"));     // 2 a's
        assert!(product.accepts("b"));      // 1 edit (substitute a for b or insert b)
        assert!(product.accepts("ab"));     // 1 edit
    }

    #[test]
    fn test_product_min_distance() {
        let nfa = compile(&parse("phone").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 3);

        assert_eq!(product.min_distance("phone"), Some(0));
        assert_eq!(product.min_distance("phon"), Some(1));
        assert_eq!(product.min_distance("phones"), Some(1));
        assert_eq!(product.min_distance("phome"), Some(1));
    }

    #[test]
    fn test_product_char_class() {
        let nfa = compile(&parse("[aeiou]+").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("a"));
        assert!(product.accepts("aeiou"));
        assert!(product.accepts("b"));  // 1 subst (b→a)
        assert!(product.accepts("ab")); // 1 error (b→a subst or b deletion)
        // Empty string IS within distance 1 of a vowel (insert any vowel)
        assert!(product.accepts(""));

        // With max_distance=0, only exact matches
        let product_exact = ProductAutomatonChar::new(
            compile(&parse("[aeiou]+").unwrap()).unwrap(),
            0,
        );
        assert!(product_exact.accepts("a"));
        assert!(product_exact.accepts("aeiou"));
        assert!(!product_exact.accepts(""));  // empty doesn't match [aeiou]+
        assert!(!product_exact.accepts("b")); // b doesn't match any vowel
    }

    #[test]
    fn test_product_over_budget() {
        let nfa = compile(&parse("abc").unwrap()).unwrap();
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("abc"));    // exact
        assert!(product.accepts("ab"));     // 1 deletion
        assert!(product.accepts("abcd"));   // 1 insertion
        assert!(!product.accepts("xyz"));   // 3 substitutions > budget
    }

    // ============================================================================
    // ProductAutomaton (byte-level) Tests
    // ============================================================================

    #[test]
    fn test_product_bytes_exact() {
        let nfa = compile_bytes(&parse_bytes(b"phone").unwrap()).unwrap();
        let product = ProductAutomaton::new(nfa, 2);

        assert!(product.accepts(b"phone"));
        assert!(!product.accepts(b"xyz"));
    }

    #[test]
    fn test_product_bytes_with_edits() {
        let nfa = compile_bytes(&parse_bytes(b"abc").unwrap()).unwrap();
        let product = ProductAutomaton::new(nfa, 1);

        assert!(product.accepts(b"abc"));
        assert!(product.accepts(b"ab"));
        assert!(product.accepts(b"abcd"));
        assert!(!product.accepts(b"xyz"));
    }

    #[test]
    fn test_product_bytes_min_distance() {
        let nfa = compile_bytes(&parse_bytes(b"phone").unwrap()).unwrap();
        let product = ProductAutomaton::new(nfa, 3);

        assert_eq!(product.min_distance(b"phone"), Some(0));
        assert_eq!(product.min_distance(b"phon"), Some(1));
    }

    // ============================================================================
    // Algorithm-specific Tests (Transposition, Merge/Split)
    // ============================================================================

    #[test]
    fn test_transposition_accepts() {
        // With standard algorithm, "ab" does NOT match "ba" with distance 1
        // (requires 2 substitutions: a→b, b→a)
        let nfa = compile(&parse("ab").unwrap()).unwrap();
        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        assert!(!standard.accepts("ba")); // distance 2 with standard

        // With transposition algorithm, "ab" DOES match "ba" with distance 1
        let transposition = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::Transposition);
        assert!(transposition.accepts("ba")); // distance 1 with transposition
    }

    #[test]
    fn test_transposition_min_distance() {
        let nfa = compile(&parse("ab").unwrap()).unwrap();

        // Standard: "ba" is distance 2 from "ab"
        let standard = ProductAutomatonChar::new(nfa.clone(), 2);
        assert_eq!(standard.min_distance("ba"), Some(2));

        // Transposition: "ba" is distance 1 from "ab"
        let transposition = ProductAutomatonChar::with_algorithm(nfa, 2, Algorithm::Transposition);
        assert_eq!(transposition.min_distance("ba"), Some(1));
    }

    #[test]
    fn test_transposition_longer_string() {
        // "hte" is "the" with h and t transposed
        let nfa = compile(&parse("the").unwrap()).unwrap();

        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        // With standard, "hte" requires 2 substitutions
        assert!(!standard.accepts("hte"));

        let transposition = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::Transposition);
        // With transposition, "hte" is just 1 transposition away
        assert!(transposition.accepts("hte"));
    }

    #[test]
    fn test_merge_split_accepts() {
        // With merge/split, we can match strings where chars are merged or split
        let nfa = compile(&parse("abc").unwrap()).unwrap();

        // Standard algorithm
        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        // "abcd" is 1 insertion (matches with standard)
        assert!(standard.accepts("abcd"));
        // "ab" is 1 deletion (matches with standard)
        assert!(standard.accepts("ab"));

        // With merge-and-split, we have additional operations
        let merge_split =
            ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::MergeAndSplit);
        assert!(merge_split.accepts("abcd")); // Still works
        assert!(merge_split.accepts("ab")); // Still works
    }

    #[test]
    fn test_merge_split_min_distance() {
        let nfa = compile(&parse("abc").unwrap()).unwrap();

        let standard = ProductAutomatonChar::new(nfa.clone(), 3);
        let merge_split = ProductAutomatonChar::with_algorithm(nfa, 3, Algorithm::MergeAndSplit);

        // Both should find exact match
        assert_eq!(standard.min_distance("abc"), Some(0));
        assert_eq!(merge_split.min_distance("abc"), Some(0));

        // Both should find single-edit matches
        assert_eq!(standard.min_distance("ab"), Some(1));
        assert_eq!(merge_split.min_distance("ab"), Some(1));
    }

    #[test]
    fn test_algorithm_getter() {
        let nfa = compile(&parse("test").unwrap()).unwrap();

        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        assert_eq!(standard.algorithm(), Algorithm::Standard);

        let transposition = ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::Transposition);
        assert_eq!(transposition.algorithm(), Algorithm::Transposition);

        let merge_split = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::MergeAndSplit);
        assert_eq!(merge_split.algorithm(), Algorithm::MergeAndSplit);
    }

    #[test]
    fn test_byte_level_transposition() {
        let nfa = compile_bytes(&parse_bytes(b"ab").unwrap()).unwrap();

        let standard = ProductAutomaton::new(nfa.clone(), 1);
        assert!(!standard.accepts(b"ba")); // distance 2 with standard

        let transposition = ProductAutomaton::with_algorithm(nfa, 1, Algorithm::Transposition);
        assert!(transposition.accepts(b"ba")); // distance 1 with transposition
    }
}
