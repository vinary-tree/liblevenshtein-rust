//! UniversalLevenshteinWfst wrapper for lling-llang Wfst trait.
//!
//! This module provides [`UniversalLevenshteinWfst`], a wrapper that exposes a
//! Universal Levenshtein transducer as a lling-llang `Wfst<char, TropicalWeight>`.
//!
//! # Key Benefits over Parameterized WFST
//!
//! - **Precomputation**: The automaton structure is query-agnostic and can be
//!   precomputed once for a given max_distance
//! - **State Deduplication**: Uses a registry to deduplicate universal states
//! - **Variant Support**: Supports Standard, Transposition, and MergeAndSplit variants

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, Wfst,
    WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};
use crate::transducer::universal::PositionVariant;

use super::state_encoding;
use super::universal_state_source::UniversalLevenshteinStateSource;

/// A Universal Levenshtein transducer exposed as a lling-llang WFST.
///
/// This wrapper presents the product of a dictionary and Universal Levenshtein
/// automaton as a weighted finite state transducer with:
/// - **Input labels**: Query characters (the misspelled input)
/// - **Output labels**: Dictionary characters (the corrections)
/// - **Weights**: Edit distances as `TropicalWeight` (lower is better)
///
/// # Type Parameters
///
/// - `V`: Position variant (Standard, Transposition, or MergeAndSplit)
/// - `D`: Dictionary type implementing [`Dictionary`] with `char` units
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::UniversalLevenshteinWfst;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::transducer::universal::Standard;
/// use lling_llang::prelude::*;
///
/// let dict = DynamicDawgChar::from_terms(vec!["hello", "help", "world"]);
/// let lev_wfst = UniversalLevenshteinWfst::<Standard, _>::new(&dict, "helo", 2);
///
/// // Use with lling-llang's composition
/// // let composed = compose(lev_wfst, other_wfst);
/// ```
#[derive(Clone)]
pub struct UniversalLevenshteinWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The state source for computing transitions
    state_source: UniversalLevenshteinStateSource<V, D>,
    /// Cached states (state_id -> computed state info)
    cache: FxHashMap<StateId, CachedState>,
    /// Maximum edit distance
    max_distance: u8,
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

impl<V, D> UniversalLevenshteinWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new Universal Levenshtein WFST for the given query and max distance.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `query`: The query string to find corrections for
    /// - `max_distance`: Maximum edit distance for matches
    ///
    /// # Returns
    ///
    /// A new `UniversalLevenshteinWfst` ready for composition or traversal.
    pub fn new(dictionary: &D, query: &str, max_distance: u8) -> Self {
        let state_source = UniversalLevenshteinStateSource::new(dictionary, query, max_distance);

        let query_len = query.chars().count();
        let max_automaton_states =
            state_encoding::estimate_automaton_states(query_len, max_distance as usize);

        Self {
            state_source,
            cache: FxHashMap::default(),
            max_distance,
            max_automaton_states,
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
        }
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the query string.
    pub fn query(&self) -> String {
        self.state_source.query()
    }

    /// Set the maximum cache size for LRU eviction.
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
            LazyState::Pending => CachedState {
                is_final: false,
                final_weight: TropicalWeight::zero(),
                transitions: SmallVec::new(),
            },
        };

        // Apply cache eviction if using LRU and over limit
        if let lling_llang::wfst::CachePolicy::Lru { max_states } = self.cache_policy {
            let limit = if max_states > 0 { max_states } else { self.max_cache_size };
            if self.cache.len() >= limit {
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

impl<V, D> Wfst<char, TropicalWeight> for UniversalLevenshteinWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
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
        self.cache.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        let (dict_node, automaton_state) =
            state_encoding::decode(state, self.max_automaton_states);
        automaton_state < self.max_automaton_states || dict_node == 0
    }
}

impl<V, D> LazyWfst<char, TropicalWeight> for UniversalLevenshteinWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
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

/// A pre-bound Universal WFST that can be cloned efficiently.
///
/// This is useful when you want to create multiple queries against the same
/// dictionary with the same automaton variant.
pub struct BoundUniversalWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    dictionary: D,
    max_distance: u8,
    _phantom: std::marker::PhantomData<V>,
}

impl<V, D> BoundUniversalWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new bound universal WFST builder.
    pub fn new(dictionary: D, max_distance: u8) -> Self {
        Self {
            dictionary,
            max_distance,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a WFST for a specific query.
    pub fn with_query(&self, query: &str) -> UniversalLevenshteinWfst<V, D> {
        UniversalLevenshteinWfst::new(&self.dictionary, query, self.max_distance)
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }
}

impl<V, D> Clone for BoundUniversalWfst<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            dictionary: self.dictionary.clone(),
            max_distance: self.max_distance,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;
    use crate::transducer::universal::Standard;

    #[test]
    fn test_universal_levenshtein_wfst_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = UniversalLevenshteinWfst::<Standard, _>::new(&dict, "helo", 2);

        assert_eq!(wfst.max_distance(), 2);
        assert_eq!(wfst.query(), "helo");
    }

    #[test]
    fn test_universal_levenshtein_wfst_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = UniversalLevenshteinWfst::<Standard, _>::new(&dict, "helo", 2);

        let start = wfst.start();
        let (dict_node, auto_state) =
            state_encoding::decode(start, wfst.max_automaton_states);
        assert_eq!(dict_node, 0);
        assert_eq!(auto_state, 0);
    }

    #[test]
    fn test_universal_levenshtein_wfst_expand_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = UniversalLevenshteinWfst::<Standard, _>::new(&dict, "helo", 2);

        let start = wfst.start();
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
        assert!(wfst.computed_states() >= 1);
    }

    #[test]
    fn test_bound_universal_wfst() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let bound = BoundUniversalWfst::<Standard, _>::new(dict, 2);

        let wfst1 = bound.with_query("helo");
        let wfst2 = bound.with_query("wrld");

        assert_eq!(wfst1.query(), "helo");
        assert_eq!(wfst2.query(), "wrld");
        assert_eq!(wfst1.max_distance(), wfst2.max_distance());
    }

    #[test]
    fn test_universal_levenshtein_wfst_cache_policy() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let mut wfst = UniversalLevenshteinWfst::<Standard, _>::new(&dict, "test", 1);

        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::CacheAll
        ));

        wfst.set_cache_policy(lling_llang::wfst::CachePolicy::Lru { max_states: 1000 });
        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::Lru { .. }
        ));
    }

    #[test]
    fn test_universal_wfst_transposition_variant() {
        use crate::transducer::universal::Transposition;

        let dict = DynamicDawgChar::<()>::from_terms(vec!["test", "tset"]);
        let wfst = UniversalLevenshteinWfst::<Transposition, _>::new(&dict, "tset", 1);

        assert_eq!(wfst.max_distance(), 1);
        assert_eq!(wfst.query(), "tset");
    }

    #[test]
    fn test_universal_wfst_merge_and_split_variant() {
        use crate::transducer::universal::MergeAndSplit;

        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "helo"]);
        let wfst = UniversalLevenshteinWfst::<MergeAndSplit, _>::new(&dict, "helo", 1);

        assert_eq!(wfst.max_distance(), 1);
        assert_eq!(wfst.query(), "helo");
    }
}
