//! WallBreaker WFST wrapper for large error bound matching.
//!
//! This module provides [`WallBreakerWfst`], a WFST wrapper for the WallBreaker
//! algorithm that overcomes the "wall effect" in similarity search with large
//! error bounds.
//!
//! # Overview
//!
//! Traditional Levenshtein automata traverse dictionaries left-to-right. With
//! error bound `k`, the first `k` steps must explore ALL prefixes up to length
//! `k` before any filtering occurs. This creates a "wall" that limits performance.
//!
//! WallBreaker overcomes this by:
//! 1. Splitting the query into pieces (pigeonhole principle)
//! 2. Finding exact substring matches using SCDAWG
//! 3. Extending bidirectionally from matches
//! 4. Verifying total distance
//!
//! # WFST Representation
//!
//! The WFST representation models WallBreaker as a lazy transducer where:
//! - **States**: Represent (piece_index, match_position, extension_state)
//! - **Transitions**: Character consumption with edit costs
//! - **Weights**: Edit distances as TropicalWeight
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::wallbreaker_wfst::WallBreakerWfst;
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//!
//! let dict = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);
//! let wfst = WallBreakerWfst::new(&dict, "cathedrel", 2);
//!
//! // Use in WFST pipeline...
//! ```
//!
//! # Algorithm Selection
//!
//! Piece counts are algorithm-dependent (formally verified):
//! - **Standard**: k+1 pieces
//! - **Transposition**: 2k+1 pieces
//! - **MergeAndSplit**: 2k+1 pieces

use std::marker::PhantomData;

use lling_llang::prelude::{
    LazyState, LazyWfst, StateId, StateSource, TropicalWeight, WeightedTransition, Wfst,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::transducer::Algorithm;
use crate::wallbreaker::{WallBreaker, WallBreakerResult};
use libdictenstein::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use libdictenstein::{Dictionary, DictionaryNode};

/// Cached state for WallBreaker WFST.
#[derive(Clone)]
struct CachedWallBreakerState {
    /// Whether this is a final (accepting) state.
    is_final: bool,
    /// Final weight if final.
    final_weight: TropicalWeight,
    /// Outgoing transitions.
    transitions: SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
}

/// Composite state key for WallBreaker.
///
/// States encode:
/// - `result_index`: Index into the result set
/// - `char_position`: Position within the result term
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct WallBreakerStateKey {
    /// Index into the results list.
    result_index: u32,
    /// Character position within the result term.
    char_position: u32,
}

/// WallBreaker WFST wrapper.
///
/// Exposes the WallBreaker algorithm as a WFST compatible with lling-llang
/// composition pipelines. This is particularly effective for large error bounds
/// where traditional automata hit the "wall effect".
///
/// # Architecture
///
/// The WFST lazily computes states by:
/// 1. Running WallBreaker query to get candidate matches
/// 2. Creating states for each (match, position) pair
/// 3. Generating transitions that trace through match terms
///
/// # Type Parameters
///
/// * `D` - Dictionary type implementing `SubstringDictionary` (typically SCDAWG)
#[derive(Clone)]
pub struct WallBreakerWfst<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    /// Phantom marker preserving the dictionary lifetime parameter.
    _dictionary: PhantomData<&'a D>,

    /// The query string.
    query: String,

    /// Maximum edit distance.
    max_distance: usize,

    /// Algorithm type.
    algorithm: Algorithm,

    /// Cached query results.
    results: Vec<WallBreakerResult>,

    /// State ID to state key mapping.
    state_map: FxHashMap<StateId, WallBreakerStateKey>,

    /// State key to state ID mapping (reverse).
    reverse_map: FxHashMap<WallBreakerStateKey, StateId>,

    /// Cached state computations.
    cache: FxHashMap<StateId, CachedWallBreakerState>,

    /// Next available state ID.
    next_state_id: StateId,

    /// Cache policy.
    cache_policy: lling_llang::wfst::CachePolicy,
}

impl<'a, D> WallBreakerWfst<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new WallBreaker WFST.
    ///
    /// Uses the Standard algorithm by default.
    pub fn new(dictionary: &'a D, query: &str, max_distance: usize) -> Self {
        Self::with_algorithm(dictionary, query, max_distance, Algorithm::Standard)
    }

    /// Create a new WallBreaker WFST with specific algorithm.
    pub fn with_algorithm(
        dictionary: &'a D,
        query: &str,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        // Run WallBreaker query to get results
        let wb = WallBreaker::with_algorithm(dictionary, max_distance, algorithm);
        let results: Vec<_> = wb.query(query).collect();

        let mut wfst = Self {
            _dictionary: PhantomData,
            query: query.to_string(),
            max_distance,
            algorithm,
            results,
            state_map: FxHashMap::default(),
            reverse_map: FxHashMap::default(),
            cache: FxHashMap::default(),
            next_state_id: 0,
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
        };

        // Register start state
        let start_key = WallBreakerStateKey {
            result_index: u32::MAX, // Special "super-start" state
            char_position: 0,
        };
        wfst.register_state(start_key);

        wfst
    }

    /// Register a state and return its ID.
    fn register_state(&mut self, key: WallBreakerStateKey) -> StateId {
        if let Some(&id) = self.reverse_map.get(&key) {
            return id;
        }

        let id = self.next_state_id;
        self.next_state_id += 1;
        self.state_map.insert(id, key);
        self.reverse_map.insert(key, id);
        id
    }

    /// Get the query string.
    #[inline]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Get the algorithm.
    #[inline]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the number of results found.
    #[inline]
    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state_id: StateId) {
        if self.cache.contains_key(&state_id) {
            return;
        }

        let key = match self.state_map.get(&state_id) {
            Some(k) => *k,
            None => return,
        };

        let (is_final, final_weight, transitions) = if key.result_index == u32::MAX {
            // Super-start state: transitions to start of each result
            self.compute_super_start_transitions()
        } else {
            self.compute_result_state(&key)
        };

        self.cache.insert(
            state_id,
            CachedWallBreakerState {
                is_final,
                final_weight,
                transitions,
            },
        );
    }

    /// Compute transitions from the super-start state.
    fn compute_super_start_transitions(
        &mut self,
    ) -> (
        bool,
        TropicalWeight,
        SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
    ) {
        let mut transitions = SmallVec::new();

        // Collect result info upfront to avoid borrow conflicts
        let result_info: Vec<_> = self
            .results
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.term.is_empty())
            .map(|(idx, r)| {
                let first_char = r
                    .term
                    .chars()
                    .next()
                    .expect("filtered out empty terms above");
                let term_len = r.term.len();
                let distance = r.distance;
                (idx, first_char, term_len, distance)
            })
            .collect();

        // Create transitions to the start of each result
        for (result_idx, first_char, term_len, distance) in result_info {
            let new_key = WallBreakerStateKey {
                result_index: result_idx as u32,
                char_position: 1,
            };
            let new_id = self.register_state(new_key);

            // Transition consumes first character, weight is distance contribution
            let weight = if term_len == 1 {
                distance as f64
            } else {
                0.0 // Distance applied at final state
            };

            transitions.push(WeightedTransition::new(
                0,
                Some(first_char),
                Some(first_char),
                new_id,
                TropicalWeight::new(weight),
            ));
        }

        // Super-start is final only if we have empty results
        let has_empty_result = self
            .results
            .iter()
            .any(|r| r.term.is_empty() && r.distance <= self.max_distance);
        let final_weight = if has_empty_result {
            self.results
                .iter()
                .filter(|r| r.term.is_empty())
                .map(|r| r.distance as f64)
                .fold(f64::INFINITY, f64::min)
        } else {
            f64::INFINITY
        };

        (
            has_empty_result,
            TropicalWeight::new(final_weight),
            transitions,
        )
    }

    /// Compute state for a result position.
    fn compute_result_state(
        &mut self,
        key: &WallBreakerStateKey,
    ) -> (
        bool,
        TropicalWeight,
        SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
    ) {
        let mut transitions = SmallVec::new();

        // Extract info upfront to avoid borrow conflicts
        let result_info = match self.results.get(key.result_index as usize) {
            Some(r) => {
                let term_chars: Vec<char> = r.term.chars().collect();
                let distance = r.distance;
                Some((term_chars, distance))
            }
            None => None,
        };

        let (term_chars, distance) = match result_info {
            Some(info) => info,
            None => return (false, TropicalWeight::infinity(), transitions),
        };

        let pos = key.char_position as usize;

        // Check if we're at the end of the term
        let is_final = pos >= term_chars.len();
        let final_weight = if is_final {
            TropicalWeight::new(distance as f64)
        } else {
            TropicalWeight::infinity()
        };

        // Add transition to next character if not at end
        if pos < term_chars.len() {
            let next_char = term_chars[pos];
            let new_key = WallBreakerStateKey {
                result_index: key.result_index,
                char_position: (pos + 1) as u32,
            };
            let new_id = self.register_state(new_key);

            // Weight is 0 for intermediate transitions, distance at final
            let weight = if pos + 1 >= term_chars.len() {
                distance as f64
            } else {
                0.0
            };

            transitions.push(WeightedTransition::new(
                0,
                Some(next_char),
                Some(next_char),
                new_id,
                TropicalWeight::new(weight),
            ));
        }

        (is_final, final_weight, transitions)
    }
}

impl<'a, D> Wfst<char, TropicalWeight> for WallBreakerWfst<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    fn start(&self) -> StateId {
        0
    }

    fn is_final(&self, state: StateId) -> bool {
        self.cache.get(&state).map(|s| s.is_final).unwrap_or(false)
    }

    fn final_weight(&self, state: StateId) -> TropicalWeight {
        self.cache
            .get(&state)
            .map(|s| s.final_weight)
            .unwrap_or_else(TropicalWeight::infinity)
    }

    fn transitions(&self, state: StateId) -> &[WeightedTransition<char, TropicalWeight>] {
        static EMPTY: &[WeightedTransition<char, TropicalWeight>] = &[];
        self.cache
            .get(&state)
            .map(|s| s.transitions.as_slice())
            .unwrap_or(EMPTY)
    }

    fn num_states(&self) -> usize {
        self.state_map.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        self.state_map.contains_key(&state)
    }
}

impl<'a, D> LazyWfst<char, TropicalWeight> for WallBreakerWfst<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
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

impl<'a, D> StateSource<char, TropicalWeight> for WallBreakerWfst<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    fn compute_state(&self, _state: StateId) -> LazyState<char, TropicalWeight> {
        // Requires mutable access for state registration
        LazyState::Pending
    }

    fn start(&self) -> StateId {
        0
    }

    fn num_states_hint(&self) -> Option<usize> {
        // Estimate: each result contributes (term_len + 1) states
        let total_chars: usize = self.results.iter().map(|r| r.term.len()).sum();
        Some(1 + total_chars + self.results.len())
    }
}

/// Builder for WallBreaker WFST.
pub struct WallBreakerWfstBuilder<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    query: Option<String>,
    max_distance: usize,
    algorithm: Algorithm,
}

impl<'a, D> WallBreakerWfstBuilder<'a, D>
where
    D: Dictionary + SubstringDictionary + Clone + Send + Sync,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new builder.
    pub fn new(dictionary: &'a D) -> Self {
        Self {
            dictionary,
            query: None,
            max_distance: 2,
            algorithm: Algorithm::Standard,
        }
    }

    /// Set the query string.
    pub fn query(mut self, query: &str) -> Self {
        self.query = Some(query.to_string());
        self
    }

    /// Set the maximum distance.
    pub fn max_distance(mut self, distance: usize) -> Self {
        self.max_distance = distance;
        self
    }

    /// Set the algorithm.
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Use standard Levenshtein algorithm.
    pub fn standard(mut self) -> Self {
        self.algorithm = Algorithm::Standard;
        self
    }

    /// Use transposition (Damerau-Levenshtein) algorithm.
    pub fn transposition(mut self) -> Self {
        self.algorithm = Algorithm::Transposition;
        self
    }

    /// Use merge-and-split algorithm.
    pub fn merge_and_split(mut self) -> Self {
        self.algorithm = Algorithm::MergeAndSplit;
        self
    }

    /// Build the WFST.
    pub fn build(self) -> Result<WallBreakerWfst<'a, D>, String> {
        let query = self.query.ok_or_else(|| "Query not set".to_string())?;
        Ok(WallBreakerWfst::with_algorithm(
            self.dictionary,
            &query,
            self.max_distance,
            self.algorithm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::scdawg::Scdawg;

    #[test]
    fn test_wallbreaker_wfst_creation() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        assert!(!wfst.is_empty());
        assert_eq!(wfst.query(), "helo");
        assert_eq!(wfst.max_distance(), 2);
    }

    #[test]
    fn test_wallbreaker_wfst_results() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        // Should have found matches
        assert!(wfst.num_results() > 0);
    }

    #[test]
    fn test_wallbreaker_wfst_start_state() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let wfst = WallBreakerWfst::new(&dict, "tset", 2);

        let start = Wfst::start(&wfst);
        assert!(wfst.is_valid_state(start));
    }

    #[test]
    fn test_wallbreaker_wfst_lazy_expansion() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = WallBreakerWfst::new(&dict, "helo", 2);

        let start = Wfst::start(&wfst);
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_wallbreaker_wfst_transitions() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = WallBreakerWfst::new(&dict, "helo", 2);

        let start = Wfst::start(&wfst);
        wfst.expand(start);

        let transitions = wfst.transitions(start);
        // Should have transitions to results
        assert!(!transitions.is_empty() || wfst.num_results() == 0);
    }

    #[test]
    fn test_wallbreaker_wfst_with_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["test", "tset"]);

        let wfst_std = WallBreakerWfst::new(&dict, "tset", 1);
        let wfst_trans =
            WallBreakerWfst::with_algorithm(&dict, "tset", 1, Algorithm::Transposition);

        assert!(matches!(wfst_std.algorithm(), Algorithm::Standard));
        assert!(matches!(wfst_trans.algorithm(), Algorithm::Transposition));
    }

    #[test]
    fn test_builder_creation() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let result = WallBreakerWfstBuilder::new(&dict)
            .query("tset")
            .max_distance(2)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_no_query() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let result = WallBreakerWfstBuilder::new(&dict).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_transposition() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let result = WallBreakerWfstBuilder::new(&dict)
            .query("tset")
            .transposition()
            .build();

        assert!(result.is_ok());
        assert!(matches!(
            result.expect("test fixture: build must be Ok").algorithm(),
            Algorithm::Transposition
        ));
    }

    #[test]
    fn test_builder_with_merge_and_split() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let result = WallBreakerWfstBuilder::new(&dict)
            .query("test")
            .merge_and_split()
            .build();

        assert!(result.is_ok());
        assert!(matches!(
            result.expect("test fixture: build must be Ok").algorithm(),
            Algorithm::MergeAndSplit
        ));
    }

    #[test]
    fn test_wfst_cache_operations() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wfst = WallBreakerWfst::new(&dict, "test", 1);

        wfst.expand(0);
        let before = wfst.computed_states();

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
        assert!(before > 0);
    }

    #[test]
    fn test_wfst_num_states_hint() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        let hint = StateSource::<char, TropicalWeight>::num_states_hint(&wfst);
        assert!(hint.is_some());
        assert!(hint.expect("expected Some hint in test") > 0);
    }

    #[test]
    fn test_wallbreaker_wfst_empty_results() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "zzzzz", 1);

        // With max distance 1, "zzzzz" shouldn't match anything
        assert!(wfst.is_empty() || wfst.num_results() == 0);
    }

    #[test]
    fn test_wallbreaker_wfst_exact_match() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "hello", 0);

        // Should find exact match
        assert_eq!(wfst.num_results(), 1);
    }

    #[test]
    fn test_wallbreaker_wfst_trace_path() {
        let dict = Scdawg::<()>::from_terms(vec!["cat"]);
        let mut wfst = WallBreakerWfst::new(&dict, "cat", 0);

        // Expand and trace through the WFST
        let start = Wfst::start(&wfst);
        wfst.expand(start);

        // Start should have transitions
        let trans = wfst.transitions(start);
        if !trans.is_empty() {
            // Follow first transition
            let next = trans[0].to;
            wfst.expand(next);

            // Continue following until final
            let mut current = next;
            for _ in 0..10 {
                let t = wfst.transitions(current);
                if t.is_empty() || wfst.is_final(current) {
                    break;
                }
                current = t[0].to;
                wfst.expand(current);
            }
        }
    }
}
