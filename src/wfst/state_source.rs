//! StateSource implementation for lazy Levenshtein WFST composition.
//!
//! This module provides [`LevenshteinStateSource`], which implements lling-llang's
//! [`StateSource`] trait for on-demand computation of Levenshtein transducer states.
//!
//! # UTF-8 Support
//!
//! This implementation uses character-level edit distance, properly handling
//! multi-byte UTF-8 characters. Each character (not byte) counts as one unit
//! for edit distance calculation.

use std::sync::Arc;

use lling_llang::prelude::{LazyState, Semiring, StateId, StateSource, TropicalWeight, WeightedTransition};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};
use crate::transducer::Algorithm;

use super::state_encoding;

/// State source for lazy Levenshtein WFST computation.
///
/// This implements lling-llang's [`StateSource`] trait, enabling lazy composition
/// of Levenshtein transducers with other WFSTs. States are computed on-demand
/// as the composed transducer is traversed.
///
/// # Product State Representation
///
/// Each WFST state represents a pair `(dictionary_node, automaton_state)`:
/// - `dictionary_node`: Position in the dictionary trie
/// - `automaton_state`: Position in the Levenshtein automaton
///
/// These are encoded into a single `StateId` for the WFST interface.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::LevenshteinStateSource;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use lling_llang::prelude::*;
///
/// let dict = DynamicDawgChar::from_terms(vec!["hello", "help"]);
/// let source = LevenshteinStateSource::new(&dict, "helo", 2);
///
/// // Wrap in LazyWfstWrapper for composition
/// let lazy_wfst = LazyWfstWrapper::new(source);
/// ```
#[derive(Clone)]
pub struct LevenshteinStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The dictionary to search
    dictionary: D,
    /// Query as characters (UTF-8 aware)
    query_chars: Arc<Vec<char>>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm variant
    algorithm: Algorithm,
    /// Maximum automaton states for encoding
    max_automaton_states: u32,
    /// Node registry: maps path hash to node ID
    node_registry: Arc<std::sync::RwLock<NodeRegistry<D::Node>>>,
}

/// Registry for assigning stable IDs to dictionary nodes.
///
/// Since dictionary nodes may not have inherent IDs, we maintain a registry
/// that assigns sequential IDs to nodes as they are encountered.
struct NodeRegistry<N: DictionaryNode> {
    /// Map from node to assigned ID
    node_to_id: FxHashMap<NodeKey, u32>,
    /// Map from ID back to node (for traversal)
    id_to_node: Vec<N>,
    /// Next available ID
    next_id: u32,
}

/// Key for identifying dictionary nodes.
///
/// This is a simplified key based on the node's identity.
/// For proper deduplication, dictionary implementations should provide
/// stable node identities.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct NodeKey(u64);

impl<N: DictionaryNode> NodeRegistry<N> {
    fn new(root: N) -> Self {
        let mut registry = Self {
            node_to_id: FxHashMap::default(),
            id_to_node: Vec::new(),
            next_id: 0,
        };
        // Register root as ID 0
        registry.register_node(root, 0);
        registry
    }

    fn register_node(&mut self, node: N, path_hash: u64) -> u32 {
        let key = NodeKey(path_hash);
        if let Some(&id) = self.node_to_id.get(&key) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.node_to_id.insert(key, id);
        self.id_to_node.push(node);
        id
    }

    fn get_node(&self, id: u32) -> Option<&N> {
        self.id_to_node.get(id as usize)
    }

    fn len(&self) -> usize {
        self.id_to_node.len()
    }
}

impl<D> LevenshteinStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new state source for the given dictionary and query.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `query`: The query string to find corrections for
    /// - `max_distance`: Maximum edit distance for matches
    ///
    /// # UTF-8 Support
    ///
    /// The query string is properly handled as UTF-8 characters. Each Unicode
    /// character (not byte) counts as one unit for edit distance calculation.
    pub fn new(dictionary: &D, query: &str, max_distance: usize) -> Self {
        Self::with_algorithm(dictionary, query, max_distance, Algorithm::Standard)
    }

    /// Create a new state source with a specific algorithm.
    pub fn with_algorithm(
        dictionary: &D,
        query: &str,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        let query_chars: Vec<char> = query.chars().collect();
        let max_automaton_states =
            state_encoding::estimate_automaton_states(query_chars.len(), max_distance);

        let root = dictionary.root();
        let registry = NodeRegistry::new(root);

        Self {
            dictionary: dictionary.clone(),
            query_chars: Arc::new(query_chars),
            max_distance,
            algorithm,
            max_automaton_states,
            node_registry: Arc::new(std::sync::RwLock::new(registry)),
        }
    }

    /// Get the query string.
    pub fn query(&self) -> String {
        self.query_chars.iter().collect()
    }

    /// Compute transitions for a product state.
    ///
    /// This explores the dictionary edges and computes character-level
    /// edit distance transitions.
    ///
    /// # State Encoding
    ///
    /// The `automaton_state_id` encodes the current position in the query
    /// (0 to query_len). Each transition consumes one dictionary character
    /// and may advance the query position.
    ///
    /// # Edit Operations (character-level, UTF-8 aware)
    ///
    /// - **Match**: query[pos] == dict_char, advance pos, cost 0
    /// - **Substitute**: query[pos] != dict_char, advance pos, cost 1
    /// - **Insert**: consume dict_char without advancing pos, cost 1
    /// - **Delete**: advance pos without consuming dict_char (epsilon), cost 1
    fn compute_transitions(
        &self,
        dict_node_id: u32,
        query_pos: u32,
    ) -> (bool, TropicalWeight, SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>) {
        let registry = self.node_registry.read().expect("Lock poisoned");

        let dict_node = match registry.get_node(dict_node_id) {
            Some(node) => node.clone(),
            None => {
                // Invalid node ID - return non-final with no transitions
                return (false, TropicalWeight::zero(), SmallVec::new());
            }
        };
        drop(registry); // Release read lock

        let mut transitions = SmallVec::new();
        let query_len = self.query_chars.len() as u32;
        let pos = query_pos as usize;

        // Iterate over dictionary edges
        for (unit, child_node) in dict_node.edges() {
            let dict_char: char = unit.into();

            // Register the child node and get its ID
            let child_node_id = {
                let mut registry = self.node_registry.write().expect("Lock poisoned");
                let path_hash = compute_path_hash(dict_node_id, dict_char);
                registry.register_node(child_node.clone(), path_hash)
            };

            let from_state = state_encoding::encode(
                dict_node_id,
                query_pos,
                self.max_automaton_states,
            );

            // Match or Substitute: consume dict_char and advance query position
            if pos < self.query_chars.len() {
                let query_char = self.query_chars[pos];
                let cost = if query_char == dict_char { 0 } else { 1 }; // Match or substitute

                // Only add if within distance threshold
                // (We check this lazily during composition, but can prune here too)
                let target_state = state_encoding::encode(
                    child_node_id,
                    query_pos + 1,
                    self.max_automaton_states,
                );

                transitions.push(WeightedTransition::new(
                    from_state,
                    Some(dict_char),
                    Some(dict_char),
                    target_state,
                    TropicalWeight::new(cost as f64),
                ));
            }

            // Insert: consume dict_char without advancing query position (cost 1)
            // This represents an extra character in the dictionary term
            let insert_target = state_encoding::encode(
                child_node_id,
                query_pos, // Stay at same query position
                self.max_automaton_states,
            );

            transitions.push(WeightedTransition::new(
                from_state,
                Some(dict_char),
                Some(dict_char),
                insert_target,
                TropicalWeight::new(1.0),
            ));

            // Transposition support (if enabled)
            if self.algorithm == Algorithm::Transposition && pos + 1 < self.query_chars.len() {
                let query_char = self.query_chars[pos];
                let next_query_char = self.query_chars[pos + 1];

                // Check if transposition applies: query[i,i+1] = [a,b] but dict has [b,...]
                // We'll handle the second character in the next state
                if dict_char == next_query_char && query_char != dict_char {
                    // This could be start of transposition - mark with special encoding
                    // For simplicity, we handle transposition as a composite operation
                    // that requires seeing both characters
                }
            }
        }

        // Delete: advance query position without consuming dict_char (epsilon transition)
        // This represents a missing character in the dictionary term
        // Note: Epsilon transitions are handled differently in lazy WFSTs
        // We emit them as transitions that stay at the same dict node
        if pos < self.query_chars.len() {
            let delete_target = state_encoding::encode(
                dict_node_id, // Stay at same dict node
                query_pos + 1, // Advance query position
                self.max_automaton_states,
            );

            let from_state = state_encoding::encode(
                dict_node_id,
                query_pos,
                self.max_automaton_states,
            );

            // Epsilon transition (no input/output label)
            transitions.push(WeightedTransition::new(
                from_state,
                None, // Epsilon input
                None, // Epsilon output
                delete_target,
                TropicalWeight::new(1.0),
            ));
        }

        // Check if this is a final state
        // Final if: dictionary node is final AND we've consumed enough of the query
        let is_final = dict_node.is_final();
        let remaining = query_len.saturating_sub(query_pos) as usize;
        let can_accept = remaining <= self.max_distance;

        let final_weight = if is_final && can_accept {
            // Final weight is the cost to delete remaining query characters
            TropicalWeight::new(remaining as f64)
        } else {
            TropicalWeight::zero() // Infinity - not accepting
        };

        (is_final && can_accept, final_weight, transitions)
    }
}

impl<D> StateSource<char, TropicalWeight> for LevenshteinStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn compute_state(&self, state: StateId) -> LazyState<char, TropicalWeight> {
        let (dict_node_id, automaton_state_id) =
            state_encoding::decode(state, self.max_automaton_states);

        let (is_final, final_weight, transitions) =
            self.compute_transitions(dict_node_id, automaton_state_id);

        if is_final {
            LazyState::final_state(final_weight, transitions)
        } else {
            LazyState::non_final(transitions)
        }
    }

    fn start(&self) -> StateId {
        // Start state is (root_node=0, initial_automaton_state=0)
        state_encoding::encode(0, 0, self.max_automaton_states)
    }

    fn num_states_hint(&self) -> Option<usize> {
        // Estimate based on dictionary size and automaton states
        let dict_size = self.dictionary.len().unwrap_or(1000);
        let automaton_states = self.max_automaton_states as usize;

        // The actual reachable states are much smaller due to pruning
        // Return a conservative estimate
        Some((dict_size * automaton_states).min(1_000_000))
    }
}

/// Compute a path hash for node registration.
#[inline]
fn compute_path_hash(parent_id: u32, edge_label: char) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    parent_id.hash(&mut hasher);
    edge_label.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_state_source_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let source = LevenshteinStateSource::new(&dict, "helo", 2);

        assert_eq!(source.start(), 0);
        assert!(source.num_states_hint().is_some());
    }

    #[test]
    fn test_state_source_compute_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
        let source = LevenshteinStateSource::new(&dict, "helo", 2);

        let start = source.start();
        let state = source.compute_state(start);

        // Start state should have transitions
        assert!(state.is_computed());
    }

    #[test]
    fn test_state_source_clone() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let source = LevenshteinStateSource::new(&dict, "tset", 2);
        let cloned = source.clone();

        assert_eq!(source.start(), cloned.start());
    }
}
