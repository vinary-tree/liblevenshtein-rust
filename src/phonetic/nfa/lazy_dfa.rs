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
//! ```rust
//! use liblevenshtein::phonetic::nfa::{compile, LazyDFAChar};
//! use liblevenshtein::phonetic::regex::parse;
//!
//! let regex = parse("(ph|f)one").expect("doc: regex parse must succeed");
//! let nfa = compile(&regex).expect("doc: NFA compilation must succeed");
//! let mut dfa = LazyDFAChar::new(nfa);
//!
//! // Check if string matches
//! assert!(dfa.accepts("phone"));
//! assert!(!dfa.accepts("xyz"));
//!
//! // Subsequent queries benefit from cached transitions
//! let cached_transitions = dfa.cache_size();
//! assert!(cached_transitions > 0);
//! assert!(dfa.accepts("phone"));
//! assert_eq!(dfa.cache_size(), cached_transitions);
//! ```

use super::state_set::StateSet;
use super::types::StateId;
use super::{NFAChar, NFA};
use rustc_hash::FxHashMap;

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

/// Compact ID for a DFA state (H8 optimization).
/// Using u32 for cache-friendly key size while supporting millions of states.
type DFAStateId = u32;

#[inline]
fn dfa_state_id_from_len(len: usize) -> Option<DFAStateId> {
    DFAStateId::try_from(len).ok()
}

#[inline]
fn dfa_state_index(id: DFAStateId) -> Option<usize> {
    usize::try_from(id).ok()
}

/// Lazy DFA for character-level NFA simulation.
///
/// Constructs DFA states on-demand during matching, caching transitions
/// for efficient repeated queries.
///
/// # H8 Optimization: State ID Caching
///
/// Instead of using `Vec<StateId>` as cache keys (which requires O(n) hash
/// and comparison), we assign each unique DFA state a compact numeric ID.
/// Cache lookups become O(1) instead of O(n).
#[derive(Debug, Clone)]
pub struct LazyDFAChar {
    /// The underlying NFA
    nfa: NFAChar,
    /// Cached transitions: (state_id, char) -> state_id (H8: O(1) keys)
    cache: FxHashMap<(DFAStateId, char), DFAStateId>,
    /// Registry mapping DFA states to compact IDs
    state_to_id: FxHashMap<DFAStateChar, DFAStateId>,
    /// Reverse mapping: ID -> DFA state
    id_to_state: Vec<DFAStateChar>,
    /// The initial DFA state ID
    initial_state_id: DFAStateId,
    /// Cache of which DFA state IDs are accepting
    accepting_cache: FxHashMap<DFAStateId, bool>,
}

impl LazyDFAChar {
    /// Create a new lazy DFA from an NFA.
    pub fn new(nfa: NFAChar) -> Self {
        // Compute initial state as epsilon closure of NFA start
        let initial_closure = nfa.epsilon_closure_single(nfa.start());
        let initial_state = Self::set_to_state(&initial_closure);

        // Register the initial state with ID 0
        let mut state_to_id = FxHashMap::default();
        state_to_id.insert(initial_state.clone(), 0);
        let id_to_state = vec![initial_state];

        Self {
            nfa,
            cache: FxHashMap::default(),
            state_to_id,
            id_to_state,
            initial_state_id: 0,
            accepting_cache: FxHashMap::default(),
        }
    }

    /// Convert a set of NFA states to a canonical DFA state representation.
    fn set_to_state(states: &StateSet) -> DFAStateChar {
        let mut vec: Vec<StateId> = states.iter().collect();
        vec.sort_unstable();
        vec
    }

    /// Convert a DFA state back to a StateSet for NFA operations.
    fn state_to_set(state: &DFAStateChar) -> StateSet {
        state.iter().copied().collect()
    }

    /// Get or create a state ID for a DFA state (H8 optimization).
    #[inline]
    fn get_or_create_id(&mut self, state: DFAStateChar) -> Option<DFAStateId> {
        if let Some(&id) = self.state_to_id.get(&state) {
            return Some(id);
        }
        let id = dfa_state_id_from_len(self.id_to_state.len())?;
        self.state_to_id.insert(state.clone(), id);
        self.id_to_state.push(state);
        Some(id)
    }

    /// Get the DFA state for an ID.
    #[inline]
    fn get_state(&self, id: DFAStateId) -> Option<&DFAStateChar> {
        dfa_state_index(id).and_then(|index| self.id_to_state.get(index))
    }

    /// Get the initial DFA state.
    #[inline]
    pub fn initial_state(&self) -> &DFAStateChar {
        self.id_to_state
            .first()
            .expect("lazy DFA state registry is never empty")
    }

    /// Check if a DFA state is accepting (by ID).
    #[inline]
    fn is_accepting_id(&mut self, state_id: DFAStateId) -> bool {
        if let Some(&accepting) = self.accepting_cache.get(&state_id) {
            return accepting;
        }

        let Some(state) = self.get_state(state_id) else {
            return false;
        };
        let accepting = state.iter().any(|&s| self.nfa.is_final(s));
        self.accepting_cache.insert(state_id, accepting);
        accepting
    }

    /// Check if a DFA state is accepting.
    ///
    /// A DFA state is accepting if any of its constituent NFA states is final.
    pub fn is_accepting(&mut self, state: &DFAStateChar) -> bool {
        // Look up the state ID (should exist if we're checking acceptance)
        if let Some(&state_id) = self.state_to_id.get(state) {
            self.is_accepting_id(state_id)
        } else {
            // State not registered - compute directly
            state.iter().any(|&s| self.nfa.is_final(s))
        }
    }

    /// Compute the transition from a DFA state ID on a character (H8 core).
    #[inline]
    fn transition_id(&mut self, state_id: DFAStateId, c: char) -> Option<DFAStateId> {
        // Check cache first - O(1) lookup with compact key
        let cache_key = (state_id, c);
        if let Some(&next_id) = self.cache.get(&cache_key) {
            return Some(next_id);
        }

        // Compute transition: for each NFA state, collect states reachable on c
        let state = self.get_state(state_id)?.clone(); // Clone needed for borrow checker
        let current_set = Self::state_to_set(&state);
        let mut next_set = StateSet::new();

        for nfa_state in current_set.iter() {
            for trans in self.nfa.transitions_from(nfa_state) {
                if trans.label.matches(c) && trans.label.consumes_input() {
                    next_set.insert(trans.to);
                }
            }
        }

        // Apply epsilon closure
        let next_closure = self.nfa.epsilon_closure(&next_set);
        let next_state = Self::set_to_state(&next_closure);

        // Get or create ID for next state
        let next_id = self.get_or_create_id(next_state)?;

        // Cache the result with compact key
        self.cache.insert(cache_key, next_id);
        Some(next_id)
    }

    /// Compute the transition from a DFA state on a character.
    ///
    /// This is the core lazy construction: we compute DFA states on-demand
    /// and cache the results.
    pub fn transition(&mut self, state: &DFAStateChar, c: char) -> DFAStateChar {
        // Get or create state ID
        let state_id = if let Some(&id) = self.state_to_id.get(state) {
            id
        } else {
            let Some(id) = self.get_or_create_id(state.clone()) else {
                return DFAStateChar::new();
            };
            id
        };

        let Some(next_id) = self.transition_id(state_id, c) else {
            return DFAStateChar::new();
        };
        self.get_state(next_id).cloned().unwrap_or_default()
    }

    /// Check if an input string is accepted by the NFA.
    ///
    /// Uses lazy DFA construction with caching for efficient matching.
    /// H8 optimization: internally uses state IDs for O(1) cache lookups.
    pub fn accepts(&mut self, input: &str) -> bool {
        let mut current_id = self.initial_state_id;

        for c in input.chars() {
            let Some(next_id) = self.transition_id(current_id, c) else {
                return false;
            };
            current_id = next_id;
            // Check for dead state (empty state has a dedicated ID or is checked)
            let Some(current_state) = self.get_state(current_id) else {
                return false;
            };
            if current_state.is_empty() {
                return false;
            }
        }

        self.is_accepting_id(current_id)
    }

    /// Get the number of cached transitions.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the number of unique DFA states discovered.
    #[inline]
    pub fn state_count(&self) -> usize {
        self.id_to_state.len()
    }

    /// Clear the transition cache.
    ///
    /// Useful for memory management in long-running applications.
    /// Note: This also clears the state registry, resetting to initial state only.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.accepting_cache.clear();
        // Reset state registry to only contain the initial state
        let Some(initial) = self.id_to_state.first().cloned() else {
            return;
        };
        self.state_to_id.clear();
        self.state_to_id.insert(initial.clone(), 0);
        self.id_to_state.clear();
        self.id_to_state.push(initial);
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
///
/// # H8 Optimization: State ID Caching
///
/// Uses compact numeric IDs for cache keys instead of `Vec<StateId>`.
#[derive(Debug, Clone)]
pub struct LazyDFA {
    /// The underlying NFA
    nfa: NFA,
    /// Cached transitions: (state_id, byte) -> state_id (H8: O(1) keys)
    cache: FxHashMap<(DFAStateId, u8), DFAStateId>,
    /// Registry mapping DFA states to compact IDs
    state_to_id: FxHashMap<DFAState, DFAStateId>,
    /// Reverse mapping: ID -> DFA state
    id_to_state: Vec<DFAState>,
    /// The initial DFA state ID
    initial_state_id: DFAStateId,
    /// Cache of which DFA state IDs are accepting
    accepting_cache: FxHashMap<DFAStateId, bool>,
}

impl LazyDFA {
    /// Create a new lazy DFA from an NFA.
    pub fn new(nfa: NFA) -> Self {
        let initial_closure = nfa.epsilon_closure_single(nfa.start());
        let initial_state = Self::set_to_state(&initial_closure);

        // Register the initial state with ID 0
        let mut state_to_id = FxHashMap::default();
        state_to_id.insert(initial_state.clone(), 0);
        let id_to_state = vec![initial_state];

        Self {
            nfa,
            cache: FxHashMap::default(),
            state_to_id,
            id_to_state,
            initial_state_id: 0,
            accepting_cache: FxHashMap::default(),
        }
    }

    /// Convert a set of NFA states to a canonical DFA state.
    fn set_to_state(states: &StateSet) -> DFAState {
        let mut vec: Vec<StateId> = states.iter().collect();
        vec.sort_unstable();
        vec
    }

    /// Convert a DFA state back to a StateSet.
    fn state_to_set(state: &DFAState) -> StateSet {
        state.iter().copied().collect()
    }

    /// Get or create a state ID for a DFA state (H8 optimization).
    #[inline]
    fn get_or_create_id(&mut self, state: DFAState) -> Option<DFAStateId> {
        if let Some(&id) = self.state_to_id.get(&state) {
            return Some(id);
        }
        let id = dfa_state_id_from_len(self.id_to_state.len())?;
        self.state_to_id.insert(state.clone(), id);
        self.id_to_state.push(state);
        Some(id)
    }

    /// Get the DFA state for an ID.
    #[inline]
    fn get_state(&self, id: DFAStateId) -> Option<&DFAState> {
        dfa_state_index(id).and_then(|index| self.id_to_state.get(index))
    }

    /// Get the initial DFA state.
    #[inline]
    pub fn initial_state(&self) -> &DFAState {
        self.id_to_state
            .first()
            .expect("lazy DFA state registry is never empty")
    }

    /// Check if a DFA state is accepting (by ID).
    #[inline]
    fn is_accepting_id(&mut self, state_id: DFAStateId) -> bool {
        if let Some(&accepting) = self.accepting_cache.get(&state_id) {
            return accepting;
        }

        let Some(state) = self.get_state(state_id) else {
            return false;
        };
        let accepting = state.iter().any(|&s| self.nfa.is_final(s));
        self.accepting_cache.insert(state_id, accepting);
        accepting
    }

    /// Check if a DFA state is accepting.
    pub fn is_accepting(&mut self, state: &DFAState) -> bool {
        if let Some(&state_id) = self.state_to_id.get(state) {
            self.is_accepting_id(state_id)
        } else {
            state.iter().any(|&s| self.nfa.is_final(s))
        }
    }

    /// Compute the transition from a DFA state ID on a byte (H8 core).
    #[inline]
    fn transition_id(&mut self, state_id: DFAStateId, b: u8) -> Option<DFAStateId> {
        let cache_key = (state_id, b);
        if let Some(&next_id) = self.cache.get(&cache_key) {
            return Some(next_id);
        }

        let state = self.get_state(state_id)?.clone();
        let current_set = Self::state_to_set(&state);
        let mut next_set = StateSet::new();

        for nfa_state in current_set.iter() {
            for trans in self.nfa.transitions_from(nfa_state) {
                if trans.label.matches(b) && trans.label.consumes_input() {
                    next_set.insert(trans.to);
                }
            }
        }

        let next_closure = self.nfa.epsilon_closure(&next_set);
        let next_state = Self::set_to_state(&next_closure);

        let next_id = self.get_or_create_id(next_state)?;
        self.cache.insert(cache_key, next_id);
        Some(next_id)
    }

    /// Compute the transition from a DFA state on a byte.
    pub fn transition(&mut self, state: &DFAState, b: u8) -> DFAState {
        let state_id = if let Some(&id) = self.state_to_id.get(state) {
            id
        } else {
            let Some(id) = self.get_or_create_id(state.clone()) else {
                return DFAState::new();
            };
            id
        };

        let Some(next_id) = self.transition_id(state_id, b) else {
            return DFAState::new();
        };
        self.get_state(next_id).cloned().unwrap_or_default()
    }

    /// Check if input is accepted.
    pub fn accepts(&mut self, input: &[u8]) -> bool {
        let mut current_id = self.initial_state_id;

        for &b in input {
            let Some(next_id) = self.transition_id(current_id, b) else {
                return false;
            };
            current_id = next_id;
            let Some(current_state) = self.get_state(current_id) else {
                return false;
            };
            if current_state.is_empty() {
                return false;
            }
        }

        self.is_accepting_id(current_id)
    }

    /// Get the number of cached transitions.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the number of unique DFA states discovered.
    #[inline]
    pub fn state_count(&self) -> usize {
        self.id_to_state.len()
    }

    /// Clear the transition cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.accepting_cache.clear();
        let Some(initial) = self.id_to_state.first().cloned() else {
            return;
        };
        self.state_to_id.clear();
        self.state_to_id.insert(initial.clone(), 0);
        self.id_to_state.clear();
        self.id_to_state.push(initial);
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
    fn test_dfa_state_id_conversion_accepts_largest_u32_id() {
        assert_eq!(dfa_state_id_from_len(u32::MAX as usize), Some(u32::MAX));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_dfa_state_id_conversion_rejects_out_of_range_ids() {
        let out_of_range = (u32::MAX as usize) + 1;
        assert_eq!(dfa_state_id_from_len(out_of_range), None);
    }

    #[test]
    fn test_lazy_dfa_char_invalid_state_id_is_dead() {
        let nfa = compile(&parse("abc").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.get_state(u32::MAX).is_none());
        assert_eq!(dfa.transition_id(u32::MAX, 'a'), None);
        assert!(!dfa.is_accepting_id(u32::MAX));
    }

    #[test]
    fn test_lazy_dfa_byte_invalid_state_id_is_dead() {
        let nfa = compile_bytes(&parse_bytes(b"abc").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFA::new(nfa);

        assert!(dfa.get_state(u32::MAX).is_none());
        assert_eq!(dfa.transition_id(u32::MAX, b'a'), None);
        assert!(!dfa.is_accepting_id(u32::MAX));
    }

    #[test]
    fn test_lazy_dfa_simple() {
        let nfa = compile(&parse("abc").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("abc"));
        assert!(!dfa.accepts("ab"));
        assert!(!dfa.accepts("abcd"));
        assert!(!dfa.accepts("xyz"));
    }

    #[test]
    fn test_lazy_dfa_alternation() {
        let nfa = compile(&parse("cat|dog").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("cat"));
        assert!(dfa.accepts("dog"));
        assert!(!dfa.accepts("ca"));
        assert!(!dfa.accepts("do"));
        assert!(!dfa.accepts("catdog"));
    }

    #[test]
    fn test_lazy_dfa_star() {
        let nfa = compile(&parse("a*").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
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
        let nfa = compile(&parse("a+").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(!dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aa"));
        assert!(dfa.accepts("aaa"));
        assert!(!dfa.accepts("b"));
    }

    #[test]
    fn test_lazy_dfa_char_class() {
        let nfa = compile(&parse("[aeiou]+").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aeiou"));
        assert!(dfa.accepts("oui"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("xyz"));
    }

    #[test]
    fn test_lazy_dfa_complex() {
        let nfa = compile(&parse("(ph|f)one").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("phone"));
        assert!(dfa.accepts("fone"));
        assert!(!dfa.accepts("bone"));
        assert!(!dfa.accepts("phon"));
    }

    #[test]
    fn test_lazy_dfa_caching() {
        let nfa = compile(&parse("test").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
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
        let nfa = compile(&parse("test").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
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
        let nfa = compile_bytes(&parse_bytes(b"hello").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFA::new(nfa);

        assert!(dfa.accepts(b"hello"));
        assert!(!dfa.accepts(b"world"));
        assert!(!dfa.accepts(b"hell"));
    }

    #[test]
    fn test_lazy_dfa_bytes_alternation() {
        let nfa = compile_bytes(&parse_bytes(b"yes|no").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFA::new(nfa);

        assert!(dfa.accepts(b"yes"));
        assert!(dfa.accepts(b"no"));
        assert!(!dfa.accepts(b"maybe"));
    }

    #[test]
    fn test_lazy_dfa_epsilon_pattern() {
        // Use a pattern that accepts empty string (empty alternation or a*)
        let nfa = compile(&parse("a*").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        // a* accepts empty string
        assert!(dfa.accepts(""));
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("aa"));
    }

    #[test]
    fn test_lazy_dfa_optional() {
        let nfa = compile(&parse("colou?r").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let mut dfa = LazyDFAChar::new(nfa);

        assert!(dfa.accepts("color"));
        assert!(dfa.accepts("colour"));
        assert!(!dfa.accepts("colr"));
        assert!(!dfa.accepts("colouur"));
    }
}
