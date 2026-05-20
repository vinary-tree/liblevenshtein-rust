//! StateSource implementation for Universal Levenshtein WFST.
//!
//! This module provides [`UniversalLevenshteinStateSource`], which implements
//! lling-llang's [`StateSource`] trait using the Universal Levenshtein Automaton.
//!
//! # Key Differences from Parameterized Automaton
//!
//! The Universal Automaton is query-agnostic and can be precomputed once for a
//! given maximum edit distance. When bound to a query via [`BoundUniversalWfst`],
//! it encodes the dictionary terms as bit vectors and processes them through
//! the automaton.

use std::sync::Arc;

use lling_llang::prelude::{LazyState, Semiring, StateId, StateSource, TropicalWeight, WeightedTransition};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};
use crate::transducer::universal::{
    CharacteristicVector, PositionVariant, UniversalAutomaton, UniversalState,
};

use super::state_encoding;

/// State registry for deduplicating Universal Levenshtein states.
///
/// Since `UniversalState<V>` is a complex set of positions, we assign each
/// unique state a sequential ID for efficient WFST state encoding.
pub struct UniversalStateRegistry<V: PositionVariant> {
    /// Map from state to assigned ID
    state_to_id: FxHashMap<UniversalStateKey, u32>,
    /// Map from ID back to state
    id_to_state: Vec<UniversalState<V>>,
}

/// Key for hashing UniversalState.
#[derive(Clone, PartialEq, Eq, Hash)]
struct UniversalStateKey(Vec<u8>);

impl<V: PositionVariant> UniversalStateRegistry<V> {
    /// Create a new registry with the initial state.
    pub fn new(max_distance: u8) -> Self {
        let mut registry = Self {
            state_to_id: FxHashMap::default(),
            id_to_state: Vec::new(),
        };

        // Register initial state as ID 0
        let initial = UniversalState::initial(max_distance);
        registry.register_state(initial);

        registry
    }

    /// Register a state and return its ID.
    pub fn register_state(&mut self, state: UniversalState<V>) -> u32 {
        let key = self.state_to_key(&state);
        if let Some(&id) = self.state_to_id.get(&key) {
            return id;
        }

        let id = self.id_to_state.len() as u32;
        self.state_to_id.insert(key, id);
        self.id_to_state.push(state);
        id
    }

    /// Get a state by ID.
    pub fn get_state(&self, id: u32) -> Option<&UniversalState<V>> {
        self.id_to_state.get(id as usize)
    }

    /// Convert a state to a hashable key.
    fn state_to_key(&self, state: &UniversalState<V>) -> UniversalStateKey {
        // Serialize positions to bytes for hashing
        let mut bytes = Vec::new();
        for pos in state.positions() {
            // Encode each position as type + offset + errors
            let (pos_type, offset, errors) = if pos.is_i_type() {
                (0u8, pos.offset(), pos.errors())
            } else {
                (1u8, pos.offset(), pos.errors())
            };
            bytes.push(pos_type);
            bytes.extend_from_slice(&(offset as i16).to_le_bytes());
            bytes.push(errors);
        }
        UniversalStateKey(bytes)
    }

    /// Number of registered states.
    pub fn len(&self) -> usize {
        self.id_to_state.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_state.is_empty()
    }
}

/// State source for Universal Levenshtein WFST computation.
///
/// This implements lling-llang's [`StateSource`] trait using the Universal
/// Levenshtein Automaton. States are computed on-demand as the WFST is traversed.
///
/// # Product State Representation
///
/// Each WFST state represents a pair `(dictionary_node_id, automaton_state_id)`:
/// - `dictionary_node_id`: Position in the dictionary trie
/// - `automaton_state_id`: ID of the universal automaton state in the registry
#[derive(Clone)]
pub struct UniversalLevenshteinStateSource<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The dictionary to search
    dictionary: D,
    /// Query as characters (for computing bit vectors)
    query_chars: Arc<Vec<char>>,
    /// Universal automaton
    automaton: UniversalAutomaton<V>,
    /// State registry for deduplication
    state_registry: Arc<std::sync::RwLock<UniversalStateRegistry<V>>>,
    /// Node registry for dictionary nodes
    node_registry: Arc<std::sync::RwLock<NodeRegistry<D::Node>>>,
    /// Maximum automaton states for encoding
    max_automaton_states: u32,
}

/// Registry for assigning stable IDs to dictionary nodes.
struct NodeRegistry<N: DictionaryNode> {
    /// Map from path hash to node ID
    node_to_id: FxHashMap<u64, u32>,
    /// Map from ID back to node
    id_to_node: Vec<N>,
}

impl<N: DictionaryNode> NodeRegistry<N> {
    fn new(root: N) -> Self {
        let mut registry = Self {
            node_to_id: FxHashMap::default(),
            id_to_node: Vec::new(),
        };
        // Register root as ID 0
        registry.register_node(root, 0);
        registry
    }

    fn register_node(&mut self, node: N, path_hash: u64) -> u32 {
        if let Some(&id) = self.node_to_id.get(&path_hash) {
            return id;
        }

        let id = self.id_to_node.len() as u32;
        self.node_to_id.insert(path_hash, id);
        self.id_to_node.push(node);
        id
    }

    fn get_node(&self, id: u32) -> Option<&N> {
        self.id_to_node.get(id as usize)
    }
}

impl<V, D> UniversalLevenshteinStateSource<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
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
    pub fn new(dictionary: &D, query: &str, max_distance: u8) -> Self {
        let query_chars: Vec<char> = query.chars().collect();
        let max_automaton_states =
            state_encoding::estimate_automaton_states(query_chars.len(), max_distance as usize);

        let root = dictionary.root();
        let node_registry = NodeRegistry::new(root);
        let state_registry = UniversalStateRegistry::new(max_distance);

        Self {
            dictionary: dictionary.clone(),
            query_chars: Arc::new(query_chars),
            automaton: UniversalAutomaton::new(max_distance),
            state_registry: Arc::new(std::sync::RwLock::new(state_registry)),
            node_registry: Arc::new(std::sync::RwLock::new(node_registry)),
            max_automaton_states,
        }
    }

    /// Get the query string.
    pub fn query(&self) -> String {
        self.query_chars.iter().collect()
    }

    /// Compute the relevant subword for a dictionary term at a given position.
    ///
    /// From thesis page 51: s_n(w, i) = w_{i-n}...w_v where v = min(|w|, i + n + 1)
    fn relevant_subword(&self, word: &[char], position: usize) -> String {
        let n = self.automaton.max_distance() as i32;
        let i = position as i32;

        let start = i - n;
        let v = std::cmp::min(word.len() as i32, i + n + 1);

        let mut result = String::new();
        for pos in start..=v {
            if pos < 1 {
                result.push('$');
            } else if pos <= word.len() as i32 {
                let idx = (pos - 1) as usize;
                result.push(word[idx]);
            }
        }
        result
    }

    /// Compute transitions for a product state.
    fn compute_transitions(
        &self,
        dict_node_id: u32,
        automaton_state_id: u32,
    ) -> (bool, TropicalWeight, SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>) {
        let node_registry = self.node_registry.read().expect("Lock poisoned");
        let state_registry = self.state_registry.read().expect("Lock poisoned");

        let dict_node = match node_registry.get_node(dict_node_id) {
            Some(node) => node.clone(),
            None => {
                return (false, TropicalWeight::zero(), SmallVec::new());
            }
        };

        let automaton_state = match state_registry.get_state(automaton_state_id) {
            Some(state) => state.clone(),
            None => {
                return (false, TropicalWeight::zero(), SmallVec::new());
            }
        };

        drop(node_registry);
        drop(state_registry);

        let mut transitions = SmallVec::new();
        let query_len = self.query_chars.len();

        // For each dictionary edge, compute the transition
        for (unit, child_node) in dict_node.edges() {
            let dict_char: char = unit.into();

            // Register the child node
            let child_node_id = {
                let mut registry = self.node_registry.write().expect("Lock poisoned");
                let path_hash = compute_path_hash(dict_node_id, dict_char);
                registry.register_node(child_node.clone(), path_hash)
            };

            // For Universal Automaton, we need to compute the bit vector
            // based on the current position in the query
            // The automaton state tracks the query position implicitly

            // Compute the current query position from the automaton state
            // For simplicity, we track position based on transitions made
            let current_pos = self.estimate_query_position(&automaton_state);

            if current_pos <= query_len {
                // Compute characteristic vector for this character
                let subword = self.relevant_subword(&self.query_chars, current_pos);
                let bit_vector = CharacteristicVector::new(dict_char, &subword);

                // Compute next automaton state
                if let Some(next_auto_state) = automaton_state.transition(&bit_vector, current_pos) {
                    // Register the new automaton state
                    let next_auto_id = {
                        let mut registry = self.state_registry.write().expect("Lock poisoned");
                        registry.register_state(next_auto_state.clone())
                    };

                    // Compute the edit cost based on match
                    let cost = if current_pos > 0
                        && current_pos <= query_len
                        && self.query_chars[current_pos - 1] == dict_char
                    {
                        0.0 // Match
                    } else {
                        1.0 // Edit operation
                    };

                    let from_state = state_encoding::encode(
                        dict_node_id,
                        automaton_state_id,
                        self.max_automaton_states,
                    );

                    let target_state = state_encoding::encode(
                        child_node_id,
                        next_auto_id,
                        self.max_automaton_states,
                    );

                    transitions.push(WeightedTransition::new(
                        from_state,
                        Some(dict_char),
                        Some(dict_char),
                        target_state,
                        TropicalWeight::new(cost),
                    ));
                }
            }
        }

        // Check if this is a final state
        let is_final = dict_node.is_final() && automaton_state.is_final();
        let final_weight = if is_final {
            // Compute minimum errors from accepting positions
            let min_errors = automaton_state
                .positions()
                .filter(|p| p.is_m_type() && p.offset() <= 0)
                .map(|p| p.errors())
                .min()
                .unwrap_or(self.automaton.max_distance() + 1);
            TropicalWeight::new(min_errors as f64)
        } else {
            TropicalWeight::zero()
        };

        (is_final, final_weight, transitions)
    }

    /// Estimate the current query position from the automaton state.
    ///
    /// This is an approximation based on the positions in the state.
    fn estimate_query_position(&self, state: &UniversalState<V>) -> usize {
        // Find the minimum query position from I-type positions
        state
            .positions()
            .filter(|p| p.is_i_type())
            .map(|p| p.offset())
            .min()
            .map(|offset| {
                // Offset represents deviation from diagonal
                // Position = errors + offset for I-type
                if offset >= 0 {
                    offset as usize
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }
}

impl<V, D> StateSource<char, TropicalWeight> for UniversalLevenshteinStateSource<V, D>
where
    V: PositionVariant + Clone + Send + Sync,
    V::State: Send + Sync,
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
        let dict_size = self.dictionary.len().unwrap_or(1000);
        let state_registry = self.state_registry.read().expect("Lock poisoned");
        let automaton_states = state_registry.len();
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
    use crate::transducer::universal::Standard;

    #[test]
    fn test_universal_state_registry_creation() {
        let registry = UniversalStateRegistry::<Standard>::new(2);
        assert_eq!(registry.len(), 1); // Initial state
    }

    #[test]
    fn test_universal_state_registry_register() {
        let mut registry = UniversalStateRegistry::<Standard>::new(2);
        let state = UniversalState::initial(2);
        let id = registry.register_state(state.clone());
        assert_eq!(id, 0); // Should be same as initial

        // Same state should get same ID
        let id2 = registry.register_state(state);
        assert_eq!(id, id2);
    }

    #[test]
    fn test_universal_state_source_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let source = UniversalLevenshteinStateSource::<Standard, _>::new(&dict, "helo", 2);

        assert_eq!(source.start(), 0);
        assert!(source.num_states_hint().is_some());
    }

    #[test]
    fn test_universal_state_source_query() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello"]);
        let source = UniversalLevenshteinStateSource::<Standard, _>::new(&dict, "helo", 2);

        assert_eq!(source.query(), "helo");
    }
}
