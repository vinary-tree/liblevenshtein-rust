//! LevenshteinWfst wrapper for lling-llang Wfst trait.
//!
//! This module provides [`LevenshteinWfst`], a wrapper that exposes a
//! Levenshtein transducer as a lling-llang `Wfst<char, TropicalWeight>`.
//!
//! # UTF-8 Support
//!
//! This implementation properly handles UTF-8 characters. Each Unicode
//! character (not byte) counts as one unit for edit distance calculation.

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, Wfst,
    WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};
use crate::transducer::Algorithm;

use super::state_encoding;
use super::state_source::LevenshteinStateSource;

/// A Levenshtein transducer exposed as a lling-llang WFST.
///
/// This wrapper presents the product of a dictionary and Levenshtein automaton
/// as a weighted finite state transducer with:
/// - **Input labels**: Query characters (the misspelled input)
/// - **Output labels**: Dictionary characters (the corrections)
/// - **Weights**: Edit distances as `TropicalWeight` (lower is better)
///
/// The product state space (dictionary_node × automaton_state) is encoded
/// into a single `StateId` for compatibility with lling-llang's interface.
///
/// # Type Parameters
///
/// - `D`: Dictionary type implementing [`Dictionary`] with `char` units
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::LevenshteinWfst;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use lling_llang::prelude::*;
///
/// let dict = DynamicDawgChar::from_terms(vec!["hello", "help", "world"]);
/// let lev_wfst = LevenshteinWfst::new(&dict, "helo", 2);
///
/// // Use with lling-llang's composition
/// // let composed = compose(lev_wfst, other_wfst);
/// ```
#[derive(Clone)]
pub struct LevenshteinWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The state source for computing transitions
    state_source: LevenshteinStateSource<D>,
    /// Cached states (state_id -> computed state info)
    cache: FxHashMap<StateId, CachedState>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm variant
    algorithm: Algorithm,
    /// Maximum automaton states for state encoding
    max_automaton_states: u32,
    /// Cache policy
    cache_policy: lling_llang::wfst::CachePolicy,
    /// Maximum cache size for LRU policy
    max_cache_size: usize,
}

/// Cached state information for a single WFST state.
#[derive(Clone)]
struct CachedState {
    is_final: bool,
    final_weight: TropicalWeight,
    transitions: SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
}

/// Default maximum cache size for LRU policy (100,000 states)
const DEFAULT_MAX_CACHE_SIZE: usize = 100_000;

impl<D> LevenshteinWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new Levenshtein WFST for the given query and max distance.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `query`: The query string to find corrections for
    /// - `max_distance`: Maximum edit distance for matches
    ///
    /// # Returns
    ///
    /// A new `LevenshteinWfst` ready for composition or traversal.
    ///
    /// # UTF-8 Support
    ///
    /// The query string is properly handled as UTF-8 characters.
    pub fn new(dictionary: &D, query: &str, max_distance: usize) -> Self {
        Self::with_algorithm(dictionary, query, max_distance, Algorithm::Standard)
    }

    /// Create a new Levenshtein WFST with a specific algorithm.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `query`: The query string to find corrections for
    /// - `max_distance`: Maximum edit distance for matches
    /// - `algorithm`: The Levenshtein algorithm variant to use
    pub fn with_algorithm(
        dictionary: &D,
        query: &str,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        let state_source = LevenshteinStateSource::with_algorithm(
            dictionary,
            query,
            max_distance,
            algorithm,
        );

        let query_len = query.chars().count();
        let max_automaton_states =
            state_encoding::estimate_automaton_states(query_len, max_distance);

        Self {
            state_source,
            cache: FxHashMap::default(),
            max_distance,
            algorithm,
            max_automaton_states,
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
        }
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Get the algorithm being used.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the query string.
    pub fn query(&self) -> String {
        self.state_source.query()
    }

    /// Set the maximum cache size for LRU eviction.
    ///
    /// Only takes effect when cache policy is set to LRU.
    pub fn set_max_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state: StateId) {
        if self.cache.contains_key(&state) {
            return;
        }

        // Use the state source to compute the state
        let lazy_state = self.state_source.compute_state(state);

        // Convert LazyState to CachedState via pattern matching
        let cached = match lazy_state {
            LazyState::Computed {
                is_final,
                final_weight,
                transitions,
            } => CachedState {
                is_final,
                final_weight,
                transitions,
            },
            LazyState::Pending => {
                // Should not happen since compute_state always returns Computed
                CachedState {
                    is_final: false,
                    final_weight: TropicalWeight::zero(),
                    transitions: SmallVec::new(),
                }
            }
        };

        // Apply cache eviction if using LRU and over limit
        if let lling_llang::wfst::CachePolicy::Lru { max_states } = self.cache_policy {
            // Use max_states from policy, falling back to max_cache_size if 0
            let limit = if max_states > 0 { max_states } else { self.max_cache_size };
            if self.cache.len() >= limit {
                // Simple eviction: remove ~10% of entries
                // A more sophisticated implementation would track access order
                let to_remove = (self.cache.len() / 10).max(1);
                let keys: Vec<_> = self.cache.keys().take(to_remove).copied().collect();
                for key in keys {
                    self.cache.remove(&key);
                }
            }
        }

        self.cache.insert(state, cached);
    }
}

// We implement Wfst via LazyWfstWrapper + StateSource pattern
// This provides proper lazy evaluation with caching

impl<D> Wfst<char, TropicalWeight> for LevenshteinWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn start(&self) -> StateId {
        self.state_source.start()
    }

    fn is_final(&self, state: StateId) -> bool {
        self.cache
            .get(&state)
            .map(|s| s.is_final)
            .unwrap_or(false)
    }

    fn final_weight(&self, state: StateId) -> TropicalWeight {
        self.cache
            .get(&state)
            .map(|s| s.final_weight)
            .unwrap_or_else(TropicalWeight::zero)
    }

    fn transitions(&self, state: StateId) -> &[WeightedTransition<char, TropicalWeight>] {
        static EMPTY: &[WeightedTransition<char, TropicalWeight>] = &[];
        self.cache
            .get(&state)
            .map(|s| s.transitions.as_slice())
            .unwrap_or(EMPTY)
    }

    fn num_states(&self) -> usize {
        // Return the number of computed (cached) states
        self.cache.len()
    }

    /// For lazy WFSTs, is_empty() returns false if there's a valid start state.
    #[inline]
    fn is_empty(&self) -> bool {
        // A Levenshtein WFST always has at least a start state
        false
    }

    /// For lazy WFSTs, any encoded state is potentially valid.
    /// The actual validity is determined when the state is expanded.
    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        // Decode and check if both components are within reasonable bounds
        let (dict_node, automaton_state) =
            state_encoding::decode(state, self.max_automaton_states);

        // The automaton state is always valid if < max_automaton_states
        // The dict_node is valid if it's in our registry or is the root (0)
        // For lazy evaluation, we assume any decoded state is potentially valid
        automaton_state < self.max_automaton_states || dict_node == 0
    }
}

impl<D> LazyWfst<char, TropicalWeight> for LevenshteinWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn is_expanded(&self, state: StateId) -> bool {
        self.cache.contains_key(&state)
    }

    fn expand(&mut self, state: StateId) {
        self.ensure_state(state);
    }

    fn transitions_lazy(&mut self, state: StateId) -> &[WeightedTransition<char, TropicalWeight>] {
        self.ensure_state(state);
        self.transitions(state)
    }

    fn cache_policy(&self) -> lling_llang::wfst::CachePolicy {
        self.cache_policy
    }

    fn set_cache_policy(&mut self, policy: lling_llang::wfst::CachePolicy) {
        self.cache_policy = policy;
    }

    fn computed_states(&self) -> usize {
        self.cache.len()
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_levenshtein_wfst_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = LevenshteinWfst::new(&dict, "helo", 2);

        assert_eq!(wfst.max_distance(), 2);
        assert_eq!(wfst.algorithm(), Algorithm::Standard);
    }

    #[test]
    fn test_levenshtein_wfst_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = LevenshteinWfst::new(&dict, "helo", 2);

        // Start state should be (0, 0) encoded
        let start = wfst.start();
        let (dict_node, auto_state) =
            state_encoding::decode(start, wfst.max_automaton_states);
        assert_eq!(dict_node, 0);
        assert_eq!(auto_state, 0);
    }

    #[test]
    fn test_levenshtein_wfst_with_algorithm() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let wfst =
            LevenshteinWfst::with_algorithm(&dict, "tset", 2, Algorithm::Transposition);

        assert_eq!(wfst.algorithm(), Algorithm::Transposition);
    }

    #[test]
    fn test_levenshtein_wfst_expand_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = LevenshteinWfst::new(&dict, "helo", 2);

        // Initially no states are expanded
        let start = wfst.start();
        assert!(!wfst.is_expanded(start));

        // Expand the start state
        wfst.expand(start);
        assert!(wfst.is_expanded(start));

        // Should have some transitions from start state
        let transitions = wfst.transitions(start);
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_levenshtein_wfst_utf8_support() {
        // Test with non-ASCII characters
        let dict = DynamicDawgChar::<()>::from_terms(vec!["café", "naïve", "北京"]);
        let mut wfst = LevenshteinWfst::new(&dict, "cafe", 1);

        let start = wfst.start();
        wfst.expand(start);

        // Should handle UTF-8 properly
        assert!(wfst.computed_states() > 0);
    }

    #[test]
    fn test_levenshtein_wfst_cache_policy() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let mut wfst = LevenshteinWfst::new(&dict, "test", 1);

        // Default should be CacheAll
        assert!(matches!(wfst.cache_policy(), lling_llang::wfst::CachePolicy::CacheAll));

        // Can change to LRU
        wfst.set_cache_policy(lling_llang::wfst::CachePolicy::Lru { max_states: 1000 });
        assert!(matches!(wfst.cache_policy(), lling_llang::wfst::CachePolicy::Lru { .. }));
    }
}
