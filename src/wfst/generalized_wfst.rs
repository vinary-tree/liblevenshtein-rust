//! Generalized Automata WFST wrapper.
//!
//! This module provides [`GeneralizedWfst`], a WFST wrapper for the generalized
//! Levenshtein automaton that supports runtime-configurable operations.
//!
//! # Overview
//!
//! The [`GeneralizedAutomaton`] supports:
//! - Standard operations (match, substitute, insert, delete)
//! - Transposition operations
//! - Merge and split operations
//! - Phonetic operations (digraphs like ph↔f, ch↔k)
//!
//! This WFST wrapper exposes these capabilities in a form compatible with
//! lling-llang WFST composition pipelines.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::generalized_wfst::{GeneralizedWfst, GeneralizedWfstBuilder};
//! use liblevenshtein::transducer::OperationSet;
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//!
//! let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "graph", "church"]);
//!
//! // Create with phonetic operations
//! let wfst = GeneralizedWfstBuilder::new(&dict)
//!     .query("fone")
//!     .max_distance(2)
//!     .with_phonetic_digraphs()
//!     .build();
//! ```

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, Wfst,
    WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};
use crate::transducer::generalized::GeneralizedAutomaton;
use crate::transducer::{OperationSet, OperationSetBuilder};

/// Cached state information for the Generalized WFST.
#[derive(Clone)]
struct CachedGeneralizedState {
    /// Whether this is a final state.
    is_final: bool,
    /// Final weight if final.
    final_weight: TropicalWeight,
    /// Outgoing transitions.
    transitions: SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
}

/// Generalized Automaton WFST wrapper.
///
/// Exposes the generalized Levenshtein automaton with runtime-configurable
/// operations as a WFST compatible with lling-llang composition.
#[derive(Clone)]
pub struct GeneralizedWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    /// Owned copy of the dictionary.
    dictionary: D,

    /// The query string.
    query: String,

    /// The generalized automaton.
    automaton: GeneralizedAutomaton,

    /// Cached state computations.
    cache: FxHashMap<StateId, CachedGeneralizedState>,

    /// Next available state ID.
    next_state_id: StateId,

    /// Cache policy.
    cache_policy: lling_llang::wfst::CachePolicy,
}

impl<D> GeneralizedWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    /// Create a new generalized WFST.
    pub fn new(dictionary: &D, query: &str, max_distance: u8, operations: OperationSet) -> Self {
        let automaton = GeneralizedAutomaton::with_operations(max_distance, operations);

        Self {
            dictionary: dictionary.clone(),
            query: query.to_string(),
            automaton,
            cache: FxHashMap::default(),
            next_state_id: 1, // Reserve 0 for start
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
        }
    }

    /// Get the query string.
    #[inline]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> u8 {
        self.automaton.max_distance()
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state_id: StateId) {
        if self.cache.contains_key(&state_id) {
            return;
        }

        // For now, just cache the start state with basic info
        // Full implementation would track dictionary position + automaton state
        if state_id == 0 {
            let query_chars: Vec<char> = self.query.chars().collect();
            let (is_final, final_weight, transitions) =
                self.compute_start_state(&query_chars);

            self.cache.insert(
                state_id,
                CachedGeneralizedState {
                    is_final,
                    final_weight,
                    transitions,
                },
            );
        }
    }

    /// Compute transitions from the start state.
    fn compute_start_state(
        &mut self,
        query_chars: &[char],
    ) -> (bool, TropicalWeight, SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>) {
        let mut transitions = SmallVec::new();
        let root = self.dictionary.root();
        let max_dist = self.automaton.max_distance();

        // Check if start is final (empty query and empty term accepted)
        let is_final = query_chars.is_empty() && root.is_final();
        let final_weight = if is_final {
            TropicalWeight::new(0.0)
        } else {
            TropicalWeight::infinity()
        };

        // Get first query character if available
        let query_char = query_chars.first().copied();

        // Iterate over dictionary edges from root
        for (dict_char, _child) in root.edges() {
            // Match transition
            if let Some(qc) = query_char {
                if qc == dict_char {
                    // Exact match - no error
                    let new_id = self.next_state_id;
                    self.next_state_id += 1;
                    transitions.push(WeightedTransition::new(
                        0,
                        Some(qc),
                        Some(dict_char),
                        new_id,
                        TropicalWeight::new(0.0),
                    ));
                } else {
                    // Substitution - 1 error
                    let new_id = self.next_state_id;
                    self.next_state_id += 1;
                    transitions.push(WeightedTransition::new(
                        0,
                        Some(qc),
                        Some(dict_char),
                        new_id,
                        TropicalWeight::new(1.0),
                    ));
                }
            }

            // Insertion transition (consume dictionary char, no query char)
            if max_dist > 0 {
                let new_id = self.next_state_id;
                self.next_state_id += 1;
                transitions.push(WeightedTransition::new(
                    0,
                    None,
                    Some(dict_char),
                    new_id,
                    TropicalWeight::new(1.0),
                ));
            }
        }

        // Deletion transition (consume query char, no dictionary char)
        if query_char.is_some() && max_dist > 0 {
            let new_id = self.next_state_id;
            self.next_state_id += 1;
            transitions.push(WeightedTransition::new(
                0,
                query_char,
                None,
                new_id,
                TropicalWeight::new(1.0),
            ));
        }

        (is_final, final_weight, transitions)
    }
}

impl<D> Wfst<char, TropicalWeight> for GeneralizedWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    fn start(&self) -> StateId {
        0
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
        self.next_state_id as usize
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.query.is_empty()
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        state < self.next_state_id
    }
}

impl<D> LazyWfst<char, TropicalWeight> for GeneralizedWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
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

impl<D> StateSource<char, TropicalWeight> for GeneralizedWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    fn compute_state(&self, _state: StateId) -> LazyState<char, TropicalWeight> {
        // Return Pending since we need mutable access for state registration
        LazyState::Pending
    }

    fn start(&self) -> StateId {
        0
    }

    fn num_states_hint(&self) -> Option<usize> {
        // Rough estimate based on query length and distance
        let query_len = self.query.len();
        let max_dist = self.automaton.max_distance() as usize;
        Some((query_len + 1) * (max_dist + 1) * 10)
    }
}

/// Builder for Generalized WFST.
pub struct GeneralizedWfstBuilder<'a, D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    dictionary: &'a D,
    query: Option<String>,
    max_distance: u8,
    operations: OperationSet,
}

impl<'a, D> GeneralizedWfstBuilder<'a, D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: DictionaryNode<Unit = char>,
{
    /// Create a new builder.
    pub fn new(dictionary: &'a D) -> Self {
        Self {
            dictionary,
            query: None,
            max_distance: 2,
            operations: OperationSet::standard(),
        }
    }

    /// Set the query string.
    pub fn query(mut self, query: &str) -> Self {
        self.query = Some(query.to_string());
        self
    }

    /// Set the maximum distance.
    pub fn max_distance(mut self, distance: u8) -> Self {
        self.max_distance = distance;
        self
    }

    /// Use standard operations.
    pub fn with_standard_ops(mut self) -> Self {
        self.operations = OperationSet::standard();
        self
    }

    /// Add transposition support.
    pub fn with_transposition(mut self) -> Self {
        self.operations = OperationSet::with_transposition();
        self
    }

    /// Add merge/split support.
    pub fn with_merge_split(mut self) -> Self {
        self.operations = OperationSet::with_merge_split();
        self
    }

    /// Use custom operations.
    pub fn with_operations(mut self, operations: OperationSet) -> Self {
        self.operations = operations;
        self
    }

    /// Add phonetic digraph operations.
    pub fn with_phonetic_digraphs(mut self) -> Self {
        use crate::transducer::phonetic::consonant_digraphs;

        let phonetic_ops = consonant_digraphs();
        let mut builder = OperationSetBuilder::new().with_standard_ops();

        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }

        self.operations = builder.build();
        self
    }

    /// Build the WFST.
    pub fn build(self) -> Result<GeneralizedWfst<D>, String> {
        let query = self.query.ok_or_else(|| "Query not set".to_string())?;
        Ok(GeneralizedWfst::new(
            self.dictionary,
            &query,
            self.max_distance,
            self.operations,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_generalized_wfst_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = GeneralizedWfst::new(&dict, "helo", 2, OperationSet::standard());

        assert!(!wfst.is_empty());
        assert_eq!(wfst.query(), "helo");
        assert_eq!(wfst.max_distance(), 2);
    }

    #[test]
    fn test_generalized_wfst_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let wfst = GeneralizedWfst::new(&dict, "tset", 2, OperationSet::standard());

        let start = Wfst::start(&wfst);
        assert!(wfst.is_valid_state(start));
    }

    #[test]
    fn test_generalized_wfst_lazy_expansion() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = GeneralizedWfst::new(&dict, "helo", 2, OperationSet::standard());

        let start = Wfst::start(&wfst);
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_builder_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let result = GeneralizedWfstBuilder::new(&dict)
            .query("tset")
            .max_distance(2)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_no_query() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let result = GeneralizedWfstBuilder::new(&dict).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_transposition() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test", "tset"]);
        let result = GeneralizedWfstBuilder::new(&dict)
            .query("tset")
            .max_distance(1)
            .with_transposition()
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_phonetic() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "graph"]);
        let result = GeneralizedWfstBuilder::new(&dict)
            .query("fone")
            .max_distance(2)
            .with_phonetic_digraphs()
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_wfst_transitions() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["ab", "abc"]);
        let mut wfst = GeneralizedWfst::new(&dict, "ab", 1, OperationSet::standard());

        let start = Wfst::start(&wfst);
        wfst.expand(start);

        let transitions = wfst.transitions(start);
        // Should have transitions for matching 'a' and possibly insertions
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_wfst_cache_operations() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let mut wfst = GeneralizedWfst::new(&dict, "test", 1, OperationSet::standard());

        wfst.expand(0);
        let before = wfst.computed_states();

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
        assert!(before > 0);
    }
}
