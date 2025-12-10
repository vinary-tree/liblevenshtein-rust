//! Lazy DFA construction for NFA simulation.
//!
//! This module provides lazy (on-demand) DFA construction from NFAs.
//! Instead of constructing the full DFA upfront (which can have exponentially
//! many states), we construct DFA states lazily as they are encountered during
//! matching.
//!
//! # Advantages
//!
//! - **Memory efficient**: Only constructs states that are actually visited
//! - **Fast startup**: No upfront powerset construction
//! - **Cache-friendly**: Frequently used transitions are cached
//!
//! # Design
//!
//! A DFA state corresponds to a set of NFA states (the powerset construction).
//! We cache the mapping from (DFA_state, character) -> DFA_state for fast lookup.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::{LazyDFAChar, NFAChar};
//!
//! let nfa = /* build NFA for pattern */;
//! let mut dfa = LazyDFAChar::new(nfa);
//!
//! // Check if string matches
//! assert!(dfa.accepts("phone"));
//! assert!(!dfa.accepts("xyz"));
//!
//! // Subsequent queries benefit from cached transitions
//! assert!(dfa.accepts("phone")); // Uses cached transitions
//! ```

use super::nfa::{NFAChar, NFA};
use super::types::StateId;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

// ============================================================================
// Character-level Lazy DFA
// ============================================================================

/// A DFA state represented as a sorted vector of NFA states.
///
/// We use a sorted `Vec` instead of `HashSet` for:
/// - Deterministic hashing (order-independent)
/// - Cache-friendly iteration
/// - Efficient equality comparison
pub type DFAStateChar = Vec<StateId>;

/// Lazy DFA for character-level NFA simulation.
///
/// Constructs DFA states on-demand during matching, caching transitions
/// for efficient repeated queries.
#[derive(Debug, Clone)]
pub struct LazyDFAChar {
    /// The underlying NFA
    nfa: NFAChar,
    /// Cached transitions: (dfa_state, char) -> dfa_state
    cache: FxHashMap<(DFAStateChar, char), DFAStateChar>,
    /// The initial DFA state (epsilon closure of NFA start state)
    initial_state: DFAStateChar,
    /// Cache of which DFA states are accepting
    accepting_cache: FxHashMap<DFAStateChar, bool>,
}

impl LazyDFAChar {
    /// Create a new lazy DFA from an NFA.
    pub fn new(nfa: NFAChar) -> Self {
        // Compute initial state as epsilon closure of NFA start
        let mut initial_set = FxHashSet::default();
        initial_set.insert(nfa.start());
        let initial_closure = nfa.epsilon_closure(&initial_set);
        let initial_state = Self::set_to_state(&initial_closure);

        Self {
            nfa,
            cache: FxHashMap::default(),
            initial_state,
            accepting_cache: FxHashMap::default(),
        }
    }

    /// Convert a set of NFA states to a canonical DFA state representation.
    fn set_to_state(states: &FxHashSet<StateId>) -> DFAStateChar {
        let mut vec: Vec<StateId> = states.iter().copied().collect();
        vec.sort_unstable();
        vec
    }

    /// Convert a DFA state back to a set for NFA operations.
    fn state_to_set(state: &DFAStateChar) -> FxHashSet<StateId> {
        state.iter().copied().collect()
    }

    /// Get the initial DFA state.
    #[inline]
    pub fn initial_state(&self) -> &DFAStateChar {
        &self.initial_state
    }

    /// Check if a DFA state is accepting.
    ///
    /// A DFA state is accepting if any of its constituent NFA states is final.
    pub fn is_accepting(&mut self, state: &DFAStateChar) -> bool {
        if let Some(&accepting) = self.accepting_cache.get(state) {
            return accepting;
        }

        let accepting = state.iter().any(|&s| self.nfa.is_final(s));
        self.accepting_cache.insert(state.clone(), accepting);
        accepting
    }

    /// Compute the transition from a DFA state on a character.
    ///
    /// This is the core lazy construction: we compute DFA states on-demand
    /// and cache the results.
    pub fn transition(&mut self, state: &DFAStateChar, c: char) -> DFAStateChar {
        // Check cache first
        let cache_key = (state.clone(), c);
        if let Some(next) = self.cache.get(&cache_key) {
            return next.clone();
        }

        // Compute transition: for each NFA state, collect states reachable on c
        let current_set = Self::state_to_set(state);
        let mut next_set = FxHashSet::default();

        for &nfa_state in &current_set {
            for trans in self.nfa.transitions_from(nfa_state) {
                if trans.label.matches(c) && trans.label.consumes_input() {
                    next_set.insert(trans.to);
                }
            }
        }

        // Apply epsilon closure
        let next_closure = self.nfa.epsilon_closure(&next_set);
        let next_state = Self::set_to_state(&next_closure);

        // Cache the result
        self.cache.insert(cache_key, next_state.clone());
        next_state
    }

    /// Check if an input string is accepted by the NFA.
    ///
    /// Uses lazy DFA construction with caching for efficient matching.
    pub fn accepts(&mut self, input: &str) -> bool {
        let mut current = self.initial_state.clone();

        for c in input.chars() {
            current = self.transition(&current, c);
            if current.is_empty() {
                return false; // Dead state
            }
        }

        self.is_accepting(&current)
    }

    /// Get the number of cached transitions.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the transition cache.
    ///
    /// Useful for memory management in long-running applications.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.accepting_cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            transition_cache_size: self.cache.len(),
            accepting_cache_size: self.accepting_cache.len(),
        }
    }
}

// ============================================================================
// Byte-level Lazy DFA
// ============================================================================

/// A DFA state for byte-level NFA.
pub type DFAState = Vec<StateId>;

/// Lazy DFA for byte-level NFA simulation.
#[derive(Debug, Clone)]
pub struct LazyDFA {
    /// The underlying NFA
    nfa: NFA,
    /// Cached transitions: (dfa_state, byte) -> dfa_state
    cache: FxHashMap<(DFAState, u8), DFAState>,
    /// The initial DFA state
    initial_state: DFAState,
    /// Cache of which DFA states are accepting
    accepting_cache: FxHashMap<DFAState, bool>,
}

impl LazyDFA {
    /// Create a new lazy DFA from an NFA.
    pub fn new(nfa: NFA) -> Self {
        let mut initial_set = FxHashSet::default();
        initial_set.insert(nfa.start());
        let initial_closure = nfa.epsilon_closure(&initial_set);
        let initial_state = Self::set_to_state(&initial_closure);

        Self {
            nfa,
            cache: FxHashMap::default(),
            initial_state,
            accepting_cache: FxHashMap::default(),
        }
    }

    /// Convert a set of NFA states to a canonical DFA state.
    fn set_to_state(states: &FxHashSet<StateId>) -> DFAState {
        let mut vec: Vec<StateId> = states.iter().copied().collect();
        vec.sort_unstable();
        vec
    }

    /// Convert a DFA state back to a set.
    fn state_to_set(state: &DFAState) -> FxHashSet<StateId> {
        state.iter().copied().collect()
    }

    /// Get the initial DFA state.
    #[inline]
    pub fn initial_state(&self) -> &DFAState {
        &self.initial_state
    }

    /// Check if a DFA state is accepting.
    pub fn is_accepting(&mut self, state: &DFAState) -> bool {
        if let Some(&accepting) = self.accepting_cache.get(state) {
            return accepting;
        }

        let accepting = state.iter().any(|&s| self.nfa.is_final(s));
        self.accepting_cache.insert(state.clone(), accepting);
        accepting
    }

    /// Compute the transition from a DFA state on a byte.
    pub fn transition(&mut self, state: &DFAState, b: u8) -> DFAState {
        let cache_key = (state.clone(), b);
        if let Some(next) = self.cache.get(&cache_key) {
            return next.clone();
        }

        let current_set = Self::state_to_set(state);
        let mut next_set = FxHashSet::default();

        for &nfa_state in &current_set {
            for trans in self.nfa.transitions_from(nfa_state) {
                if trans.label.matches(b) && trans.label.consumes_input() {
                    next_set.insert(trans.to);
                }
            }
        }

        let next_closure = self.nfa.epsilon_closure(&next_set);
        let next_state = Self::set_to_state(&next_closure);

        self.cache.insert(cache_key, next_state.clone());
        next_state
    }

    /// Check if input is accepted.
    pub fn accepts(&mut self, input: &[u8]) -> bool {
        let mut current = self.initial_state.clone();

        for &b in input {
            current = self.transition(&current, b);
            if current.is_empty() {
                return false;
            }
        }

        self.is_accepting(&current)
    }

    /// Get the number of cached transitions.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the transition cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.accepting_cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            transition_cache_size: self.cache.len(),
            accepting_cache_size: self.accepting_cache.len(),
        }
    }
}

// ============================================================================
// Cache Statistics
// ============================================================================

/// Statistics about the lazy DFA cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cached transition entries
    pub transition_cache_size: usize,
    /// Number of cached accepting state entries
    pub accepting_cache_size: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};

    #[test]
    fn test_lazy_dfa_simple() {
        let nfa = compile(&parse("abc").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("abc"));
        assert!(!dfa.accepts("ab"));
        assert!(!dfa.accepts("abcd"));
        assert!(!dfa.accepts("xyz"));
    }

    #[test]
    fn test_lazy_dfa_alternation() {
        let nfa = compile(&parse("cat|dog").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("cat"));
        assert!(dfa.accepts("dog"));
        assert!(!dfa.accepts("ca"));
        assert!(!dfa.accepts("do"));
        assert!(!dfa.accepts("catdog"));
    }

    #[test]
    fn test_lazy_dfa_star() {
        let nfa = compile(&parse("a*").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aa"));
        assert!(dfa.accepts("aaa"));
        assert!(!dfa.accepts("b"));
        assert!(!dfa.accepts("ab"));
    }

    #[test]
    fn test_lazy_dfa_plus() {
        let nfa = compile(&parse("a+").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(!dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aa"));
        assert!(dfa.accepts("aaa"));
        assert!(!dfa.accepts("b"));
    }

    #[test]
    fn test_lazy_dfa_char_class() {
        let nfa = compile(&parse("[aeiou]+").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aeiou"));
        assert!(dfa.accepts("oui"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("xyz"));
    }

    #[test]
    fn test_lazy_dfa_complex() {
        let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("phone"));
        assert!(dfa.accepts("fone"));
        assert!(!dfa.accepts("bone"));
        assert!(!dfa.accepts("phon"));
    }

    #[test]
    fn test_lazy_dfa_caching() {
        let nfa = compile(&parse("test").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        // First query builds cache
        assert!(dfa.accepts("test"));
        let stats1 = dfa.cache_stats();
        assert!(stats1.transition_cache_size > 0);

        // Second query uses cache (size shouldn't change)
        assert!(dfa.accepts("test"));
        let stats2 = dfa.cache_stats();
        assert_eq!(stats1.transition_cache_size, stats2.transition_cache_size);
    }

    #[test]
    fn test_lazy_dfa_cache_clear() {
        let nfa = compile(&parse("test").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("test"));
        assert!(dfa.cache_size() > 0);

        dfa.clear_cache();
        assert_eq!(dfa.cache_size(), 0);

        // Should still work after clearing
        assert!(dfa.accepts("test"));
    }

    #[test]
    fn test_lazy_dfa_bytes() {
        let nfa = compile_bytes(&parse_bytes(b"hello").unwrap()).unwrap();
        let mut dfa = LazyDFA::new(nfa);

        assert!(dfa.accepts(b"hello"));
        assert!(!dfa.accepts(b"world"));
        assert!(!dfa.accepts(b"hell"));
    }

    #[test]
    fn test_lazy_dfa_bytes_alternation() {
        let nfa = compile_bytes(&parse_bytes(b"yes|no").unwrap()).unwrap();
        let mut dfa = LazyDFA::new(nfa);

        assert!(dfa.accepts(b"yes"));
        assert!(dfa.accepts(b"no"));
        assert!(!dfa.accepts(b"maybe"));
    }

    #[test]
    fn test_lazy_dfa_epsilon_pattern() {
        // Use a pattern that accepts empty string (empty alternation or a*)
        let nfa = compile(&parse("a*").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        // a* accepts empty string
        assert!(dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aa"));
    }

    #[test]
    fn test_lazy_dfa_optional() {
        let nfa = compile(&parse("colou?r").unwrap()).unwrap();
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("color"));
        assert!(dfa.accepts("colour"));
        assert!(!dfa.accepts("colr"));
        assert!(!dfa.accepts("colouur"));
    }
}
