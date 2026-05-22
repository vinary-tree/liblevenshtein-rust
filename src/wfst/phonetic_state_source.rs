//! StateSource implementation for Phonetic WFST composition.
//!
//! This module provides [`PhoneticStateSource`], which implements lling-llang's
//! [`StateSource`] trait for phonetic fuzzy matching. It composes a phonetic
//! NFA with a Levenshtein automaton for combined phonetic + edit distance matching.
//!
//! # Architecture
//!
//! The phonetic state source computes product states:
//! - **NFA states**: Active states in the phonetic NFA (after epsilon closure)
//! - **Edit distance**: Current edit distance consumed
//! - **Dictionary node**: Position in the dictionary trie
//!
//! The product state space is encoded into a single `StateId` for WFST composition.

use std::sync::Arc;

use lling_llang::prelude::{
    LazyState, Semiring, StateId, StateSource, TropicalWeight, WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::{NFAChar, ProductAutomatonChar, ProductStateChar};
use libdictenstein::{Dictionary, DictionaryNode};

use super::state_encoding;

/// State source for phonetic WFST composition.
///
/// This implements lling-llang's [`StateSource`] trait using the phonetic
/// product automaton (NFA × Levenshtein). States are computed on-demand as
/// the composed transducer is traversed.
///
/// # Product State Representation
///
/// Each WFST state represents a tuple:
/// - `dictionary_node_id`: Position in the dictionary trie
/// - `product_state_id`: ID of the phonetic product state (NFA × Levenshtein)
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::PhoneticStateSource;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::phonetic::nfa::compile;
/// use liblevenshtein::phonetic::regex::parse;
/// use lling_llang::prelude::*;
///
/// let dict = DynamicDawgChar::from_terms(vec!["phone", "fone", "phon"]);
/// let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
/// let source = PhoneticStateSource::new(&dict, nfa, 2);
///
/// // Use with LazyWfstWrapper for composition
/// let lazy_wfst = LazyWfstWrapper::new(source);
/// ```
#[cfg(feature = "phonetic-rules")]
#[derive(Clone)]
pub struct PhoneticStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The dictionary to search
    dictionary: D,
    /// Product automaton (NFA × Levenshtein)
    product: Arc<ProductAutomatonChar>,
    /// Maximum edit distance
    max_distance: u8,
    /// Phonetic weight (cost for phonetic transformations)
    phonetic_weight: f64,
    /// Maximum product states for encoding
    max_product_states: u32,
    /// Node registry: maps path hash to node ID
    node_registry: Arc<std::sync::RwLock<NodeRegistry<D::Node>>>,
    /// Product state registry: maps product state to ID
    product_state_registry: Arc<std::sync::RwLock<ProductStateRegistry>>,
}

/// Registry for assigning stable IDs to dictionary nodes.
struct NodeRegistry<N: DictionaryNode> {
    /// Map from path hash to assigned ID
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

/// Registry for assigning stable IDs to product states.
#[cfg(feature = "phonetic-rules")]
struct ProductStateRegistry {
    /// Map from product state to assigned ID
    state_to_id: FxHashMap<ProductStateKey, u32>,
    /// Map from ID back to state
    id_to_state: Vec<ProductStateChar>,
}

/// Key for hashing ProductStateChar.
#[cfg(feature = "phonetic-rules")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct ProductStateKey(Vec<u8>);

#[cfg(feature = "phonetic-rules")]
impl ProductStateRegistry {
    fn new(initial_state: ProductStateChar) -> Self {
        let mut registry = Self {
            state_to_id: FxHashMap::default(),
            id_to_state: Vec::new(),
        };
        registry.register_state(initial_state);
        registry
    }

    fn register_state(&mut self, state: ProductStateChar) -> u32 {
        let key = Self::state_to_key(&state);
        if let Some(&id) = self.state_to_id.get(&key) {
            return id;
        }

        let id = self.id_to_state.len() as u32;
        self.state_to_id.insert(key, id);
        self.id_to_state.push(state);
        id
    }

    fn get_state(&self, id: u32) -> Option<&ProductStateChar> {
        self.id_to_state.get(id as usize)
    }

    fn state_to_key(state: &ProductStateChar) -> ProductStateKey {
        let mut bytes = Vec::with_capacity(state.nfa_states.len() * 4 + 1);
        for &s in &state.nfa_states {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes.push(state.edit_distance());
        ProductStateKey(bytes)
    }

    fn len(&self) -> usize {
        self.id_to_state.len()
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new phonetic state source.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `nfa`: The phonetic NFA pattern
    /// - `max_distance`: Maximum edit distance for matches
    pub fn new(dictionary: &D, nfa: NFAChar, max_distance: u8) -> Self {
        Self::with_phonetic_weight(dictionary, nfa, max_distance, 0.0)
    }

    /// Create a new phonetic state source with custom phonetic weight.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `nfa`: The phonetic NFA pattern
    /// - `max_distance`: Maximum edit distance for matches
    /// - `phonetic_weight`: Cost for phonetic transformations
    pub fn with_phonetic_weight(
        dictionary: &D,
        nfa: NFAChar,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        let product =
            ProductAutomatonChar::with_phonetic_weight(nfa, max_distance, phonetic_weight);
        let initial_state = product.initial_state();

        // Estimate max product states based on NFA size and max distance
        let max_product_states = ((max_distance as u32 + 1) * 1000).max(10_000);

        let root = dictionary.root();
        let node_registry = NodeRegistry::new(root);
        let product_state_registry = ProductStateRegistry::new(initial_state);

        Self {
            dictionary: dictionary.clone(),
            product: Arc::new(product),
            max_distance,
            phonetic_weight,
            max_product_states,
            node_registry: Arc::new(std::sync::RwLock::new(node_registry)),
            product_state_registry: Arc::new(std::sync::RwLock::new(product_state_registry)),
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

    /// Compute transitions for a product state.
    fn compute_transitions(
        &self,
        dict_node_id: u32,
        product_state_id: u32,
    ) -> (
        bool,
        TropicalWeight,
        SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
    ) {
        let node_registry = self.node_registry.read().expect("Lock poisoned");
        let product_registry = self.product_state_registry.read().expect("Lock poisoned");

        let dict_node = match node_registry.get_node(dict_node_id) {
            Some(node) => node.clone(),
            None => {
                return (false, TropicalWeight::zero(), SmallVec::new());
            }
        };

        let product_state = match product_registry.get_state(product_state_id) {
            Some(state) => state.clone(),
            None => {
                return (false, TropicalWeight::zero(), SmallVec::new());
            }
        };

        drop(node_registry);
        drop(product_registry);

        let mut transitions = SmallVec::new();

        // For each dictionary edge, compute product transitions
        for (unit, child_node) in dict_node.edges() {
            let dict_char: char = unit.into();

            // Register the child node
            let child_node_id = {
                let mut registry = self.node_registry.write().expect("Lock poisoned");
                let path_hash = compute_path_hash(dict_node_id, dict_char);
                registry.register_node(child_node.clone(), path_hash)
            };

            // Compute product automaton successors for this character
            let successors = self.product.transition(&product_state, dict_char);

            for successor in successors {
                // Register the successor product state
                let successor_id = {
                    let mut registry = self.product_state_registry.write().expect("Lock poisoned");
                    registry.register_state(successor.clone())
                };

                // Compute transition cost
                let cost = if successor.edit_distance() > product_state.edit_distance() {
                    // Edit operation was used
                    1.0
                } else {
                    // Exact match (may include phonetic transformation)
                    self.phonetic_weight
                };

                let from_state =
                    state_encoding::encode(dict_node_id, product_state_id, self.max_product_states);

                let target_state =
                    state_encoding::encode(child_node_id, successor_id, self.max_product_states);

                transitions.push(WeightedTransition::new(
                    from_state,
                    Some(dict_char),
                    Some(dict_char),
                    target_state,
                    TropicalWeight::new(cost),
                ));
            }
        }

        // Check if this is a final state
        let is_final = dict_node.is_final() && self.product.is_accepting(&product_state);
        let final_weight = if is_final {
            // Final weight is the edit distance consumed
            TropicalWeight::new(product_state.edit_distance() as f64)
        } else {
            TropicalWeight::zero()
        };

        (is_final, final_weight, transitions)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D> StateSource<char, TropicalWeight> for PhoneticStateSource<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn compute_state(&self, state: StateId) -> LazyState<char, TropicalWeight> {
        let (dict_node_id, product_state_id) =
            state_encoding::decode(state, self.max_product_states);

        let (is_final, final_weight, transitions) =
            self.compute_transitions(dict_node_id, product_state_id);

        if is_final {
            LazyState::final_state(final_weight, transitions)
        } else {
            LazyState::non_final(transitions)
        }
    }

    fn start(&self) -> StateId {
        // Start state is (root_node=0, initial_product_state=0)
        state_encoding::encode(0, 0, self.max_product_states)
    }

    fn num_states_hint(&self) -> Option<usize> {
        let dict_size = self.dictionary.len().unwrap_or(1000);
        let product_registry = self.product_state_registry.read().expect("Lock poisoned");
        let product_states = product_registry.len();
        Some((dict_size * product_states).min(1_000_000))
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
#[cfg(feature = "phonetic-rules")]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::compile;
    use crate::phonetic::regex::parse;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_phonetic_state_source_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "fone", "help"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let source = PhoneticStateSource::new(&dict, nfa, 2);

        assert_eq!(source.max_distance(), 2);
        assert_eq!(source.phonetic_weight(), 0.0);
    }

    #[test]
    fn test_phonetic_state_source_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let source = PhoneticStateSource::new(&dict, nfa, 1);

        let start = source.start();
        let (dict_node, product_state) = state_encoding::decode(start, source.max_product_states);
        assert_eq!(dict_node, 0);
        assert_eq!(product_state, 0);
    }

    #[test]
    fn test_phonetic_state_source_with_weight() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let source = PhoneticStateSource::with_phonetic_weight(&dict, nfa, 2, 0.5);

        assert_eq!(source.phonetic_weight(), 0.5);
    }

    #[test]
    fn test_phonetic_state_source_compute_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "help"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");
        let source = PhoneticStateSource::new(&dict, nfa, 1);

        let start = source.start();
        let state = source.compute_state(start);

        // Start state should have transitions
        assert!(state.is_computed());
    }

    #[test]
    fn test_phonetic_state_source_num_states_hint() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test", "rest", "best"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let source = PhoneticStateSource::new(&dict, nfa, 1);

        let hint = source.num_states_hint();
        assert!(hint.is_some());
        assert!(hint.expect("expected Some hint in test") > 0);
    }
}
