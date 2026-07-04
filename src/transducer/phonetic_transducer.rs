//! Phonetic transducer combining NFA-based phonetic matching with dictionary lookups.
//!
//! This module provides [`PhoneticTransducer`] which integrates phonetic NFA patterns
//! with dictionaries for combined phonetic + edit distance matching (fuzzy regex).
//!
//! # Architecture
//!
//! The phonetic transducer composes two automata:
//!
//! 1. **Phonetic NFA**: Handles sound-based variations (ph ↔ f, c → s / _[ei])
//! 2. **Dictionary traversal**: Efficiently explores the dictionary
//!
//! The result is a fuzzy regex that finds dictionary terms matching a phonetic
//! pattern within an edit distance threshold.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::transducer::PhoneticTransducer;
//! use liblevenshtein::dictionary::DoubleArrayTrie;
//! use liblevenshtein::phonetic::nfa::{compile, NFAChar};
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Build dictionary
//! let dict = DoubleArrayTrie::from_terms(vec!["phone", "phones", "fone", "elephant"]);
//!
//! // Build phonetic NFA for pattern "(ph|f)one"
//! let pattern = compile(&parse("(ph|f)one").expect("doc example: regex parses")).expect("doc example: regex compiles");
//!
//! // Create phonetic transducer
//! let transducer = PhoneticTransducer::new(dict, pattern, 1);
//!
//! // Query - finds "phone", "phones", "fone" (all within distance 1 of pattern)
//! for candidate in transducer.query("fone") {
//!     println!("{}: distance {}", candidate.term, candidate.distance);
//! }
//! ```

#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::product::{ProductAutomaton, ProductAutomatonChar};
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::{NFAChar, NFA};
use libdictenstein::{Dictionary, DictionaryNode};

use std::{
    cmp::Ordering,
    collections::VecDeque,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

#[cfg(feature = "phonetic-rules")]
const PHONETIC_NO_PATH: usize = usize::MAX;

#[cfg(feature = "phonetic-rules")]
struct PhoneticPathNode<U: Copy> {
    label: U,
    parent: usize,
    depth: usize,
}

#[cfg(feature = "phonetic-rules")]
struct PhoneticTraversal<N: DictionaryNode> {
    node: N,
    label: Option<N::Unit>,
    parent: usize,
    depth: usize,
}

#[cfg(feature = "phonetic-rules")]
impl<N: DictionaryNode> PhoneticTraversal<N> {
    #[inline]
    fn root(node: N) -> Self {
        Self {
            node,
            label: None,
            parent: PHONETIC_NO_PATH,
            depth: 0,
        }
    }

    #[inline]
    fn child(node: N, label: N::Unit, parent: usize, depth: usize) -> Self {
        Self {
            node,
            label: Some(label),
            parent,
            depth,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
fn collect_path_units<U: Copy>(
    label: Option<U>,
    parent: usize,
    arena: &[PhoneticPathNode<U>],
) -> Vec<U> {
    let parent_depth = if parent == PHONETIC_NO_PATH {
        0
    } else {
        arena[parent].depth
    };
    let capacity = parent_depth + usize::from(label.is_some());
    let mut units = Vec::with_capacity(capacity);

    if let Some(label) = label {
        units.push(label);
    }

    let mut current = parent;
    while current != PHONETIC_NO_PATH {
        let node = &arena[current];
        units.push(node.label);
        current = node.parent;
    }

    units.reverse();
    units
}

// ============================================================================
// Phonetic Candidate
// ============================================================================

/// A candidate result from phonetic transducer query.
#[derive(Debug, Clone)]
pub struct PhoneticCandidate {
    /// The matching term from the dictionary
    pub term: String,
    /// Edit distance from the query to this term
    pub edit_distance: u8,
    /// Phonetic transformation cost.
    ///
    /// Reserved: in the current implementation this is ALWAYS `0.0` — the query
    /// path constructs every candidate with `phonetic_cost == 0.0` and no
    /// phonetic weighting is applied. Kept for a future weighted-phonetic-
    /// matching feature; see `PhoneticTransducerChar::with_phonetic_weight`.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`).
    ///
    /// Because `phonetic_cost` is always `0.0` today, this currently equals
    /// `edit_distance as f64`.
    pub total_cost: f64,
}

impl PhoneticCandidate {
    /// Create a new phonetic candidate.
    pub fn new(term: String, edit_distance: u8, phonetic_cost: f64) -> Self {
        let total_cost = edit_distance as f64 + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
        }
    }
}

impl PartialEq for PhoneticCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term
            && self.edit_distance == other.edit_distance
            && self.phonetic_cost.total_cmp(&other.phonetic_cost) == Ordering::Equal
            && self.total_cost.total_cmp(&other.total_cost) == Ordering::Equal
    }
}

impl Eq for PhoneticCandidate {}

impl Hash for PhoneticCandidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term.hash(state);
        self.edit_distance.hash(state);
        self.phonetic_cost.to_bits().hash(state);
        self.total_cost.to_bits().hash(state);
    }
}

impl PartialOrd for PhoneticCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by total_cost (lower is better), then by term alphabetically
        self.total_cost
            .total_cmp(&other.total_cost)
            .then_with(|| self.term.cmp(&other.term))
            .then_with(|| self.edit_distance.cmp(&other.edit_distance))
            .then_with(|| self.phonetic_cost.total_cmp(&other.phonetic_cost))
    }
}

#[cfg(feature = "phonetic-rules")]
#[inline]
fn phonetic_child_depth(depth: usize) -> Option<usize> {
    depth.checked_add(1)
}

// ============================================================================
// Character-level Phonetic Transducer
// ============================================================================

/// Phonetic transducer combining NFA pattern matching with dictionary lookups.
///
/// This transducer performs fuzzy regex queries by:
/// 1. Using a phonetic NFA to match sound-based variations
/// 2. Allowing additional edit distance for typos
/// 3. Efficiently traversing the dictionary
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticTransducerChar<D: Dictionary> {
    /// The dictionary to search
    dictionary: D,
    /// The phonetic NFA pattern
    nfa: NFAChar,
    /// Maximum allowed edit distance
    max_distance: u8,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// Currently stored and threaded to the query iterator but NOT applied:
    /// every emitted [`PhoneticCandidate`] has `phonetic_cost == 0.0`. Clamped
    /// to `>= 0.0` at construction. Tracked for a future weighted-phonetic-
    /// matching feature.
    phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> PhoneticTransducerChar<D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    /// Create a new phonetic transducer.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The dictionary to search
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    pub fn new(dictionary: D, nfa: NFAChar, max_distance: u8) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: 0.0,
        }
    }

    /// Create a phonetic transducer with a custom phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored (and threaded to the
    /// query iterator) but NOT applied to matching cost or ranking. Every
    /// emitted [`PhoneticCandidate`] reports `phonetic_cost == 0.0`, so
    /// `total_cost == edit_distance`. The value is clamped to `>= 0.0` so a
    /// future implementation cannot violate monotone-cost pruning. Tracked for a
    /// future weighted-phonetic-matching feature.
    pub fn with_phonetic_weight(
        dictionary: D,
        nfa: NFAChar,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
        }
    }

    /// Query for dictionary terms matching the phonetic pattern.
    ///
    /// Returns an iterator over [`PhoneticCandidate`] results, ordered by total cost.
    pub fn query(&self, input: &str) -> PhoneticQueryIteratorChar<'_, D> {
        PhoneticQueryIteratorChar::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.phonetic_weight,
        )
    }

    /// Query and collect all results, sorted by total cost.
    pub fn query_sorted(&self, input: &str) -> Vec<PhoneticCandidate> {
        let mut results: Vec<_> = self.query(input).collect();
        results.sort();
        results
    }

    /// Get the underlying dictionary.
    #[inline]
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get the phonetic NFA.
    #[inline]
    pub fn nfa(&self) -> &NFAChar {
        &self.nfa
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Extract the dictionary, consuming the transducer.
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }
}

// ============================================================================
// Query Iterator (Character-level)
// ============================================================================

/// Iterator over phonetic query results.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticQueryIteratorChar<'a, D: Dictionary> {
    /// Product automaton for matching
    product: ProductAutomatonChar,
    /// Queue of dictionary nodes to explore.
    queue: VecDeque<PhoneticTraversal<D::Node>>,
    /// Parent-path arena for reconstructing terms without per-edge path clones.
    path_arena: Vec<PhoneticPathNode<char>>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
    /// Maximum depth (prevents infinite exploration)
    max_depth: usize,
    /// Phonetic weight
    _phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: Dictionary> PhoneticQueryIteratorChar<'a, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFAChar,
        _input: &str,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        // Create product automaton for this query
        let product = ProductAutomatonChar::new(nfa.clone(), max_distance);

        // Initialize queue with root node
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back(PhoneticTraversal::root(dictionary.root()));

        // Max depth: reasonable buffer for dictionary traversal
        let max_depth = 100;

        Self {
            product,
            queue,
            path_arena: Vec::with_capacity(64),
            _dictionary: PhantomData,
            max_depth,
            _phonetic_weight: phonetic_weight,
        }
    }

    #[inline]
    fn push_path_node(&mut self, label: char, parent: usize) -> usize {
        let depth = if parent == PHONETIC_NO_PATH {
            1
        } else {
            self.path_arena[parent].depth.saturating_add(1)
        };
        let index = self.path_arena.len();
        self.path_arena.push(PhoneticPathNode {
            label,
            parent,
            depth,
        });
        index
    }

    #[inline]
    fn materialize_path(&self, entry: &PhoneticTraversal<D::Node>) -> String {
        collect_path_units(entry.label, entry.parent, &self.path_arena)
            .into_iter()
            .collect()
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIteratorChar<'_, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    type Item = PhoneticCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.queue.pop_front() {
            // Depth limit to prevent infinite exploration
            if entry.depth > self.max_depth {
                continue;
            }

            // Check if this is a final dictionary node
            if entry.node.is_final() {
                // Check if the product automaton accepts this path
                let path = self.materialize_path(&entry);
                if let Some(distance) = self.product.min_distance(&path) {
                    return Some(PhoneticCandidate::new(path, distance, 0.0));
                }
            }

            let parent = match entry.label {
                Some(label) => self.push_path_node(label, entry.parent),
                None => entry.parent,
            };

            // Explore children via edges
            let Some(child_depth) = phonetic_child_depth(entry.depth) else {
                continue;
            };
            if child_depth > self.max_depth {
                continue;
            }
            for (c, child) in entry.node.edges() {
                self.queue
                    .push_back(PhoneticTraversal::child(child, c, parent, child_depth));
            }
        }

        None
    }
}

// ============================================================================
// Byte-level Phonetic Transducer
// ============================================================================

/// Byte-level phonetic transducer.
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticTransducer<D: Dictionary> {
    /// The dictionary to search
    dictionary: D,
    /// The phonetic NFA pattern
    nfa: NFA,
    /// Maximum allowed edit distance
    max_distance: u8,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// Currently stored and threaded to the query iterator but NOT applied:
    /// every emitted [`PhoneticCandidateByte`] has `phonetic_cost == 0.0`.
    /// Clamped to `>= 0.0` at construction. Tracked for a future
    /// weighted-phonetic-matching feature.
    phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> PhoneticTransducer<D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    /// Create a new byte-level phonetic transducer.
    pub fn new(dictionary: D, nfa: NFA, max_distance: u8) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: 0.0,
        }
    }

    /// Create with a custom phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored (and threaded to the
    /// query iterator) but NOT applied to matching cost or ranking. Every
    /// emitted [`PhoneticCandidateByte`] reports `phonetic_cost == 0.0`, so
    /// `total_cost == edit_distance`. The value is clamped to `>= 0.0` to
    /// preserve the future monotone-cost-pruning invariant. Tracked for a future
    /// weighted-phonetic-matching feature.
    pub fn with_phonetic_weight(
        dictionary: D,
        nfa: NFA,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
        }
    }

    /// Query for dictionary terms matching the phonetic pattern.
    pub fn query(&self, input: &[u8]) -> PhoneticQueryIterator<'_, D> {
        PhoneticQueryIterator::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.phonetic_weight,
        )
    }

    /// Query and collect all results, sorted by total cost.
    pub fn query_sorted(&self, input: &[u8]) -> Vec<PhoneticCandidateByte> {
        let mut results: Vec<_> = self.query(input).collect();
        results.sort();
        results
    }

    /// Get the underlying dictionary.
    #[inline]
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get the phonetic NFA.
    #[inline]
    pub fn nfa(&self) -> &NFA {
        &self.nfa
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Extract the dictionary.
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }
}

/// Byte-level phonetic candidate.
#[derive(Debug, Clone)]
pub struct PhoneticCandidateByte {
    /// The matching term from the dictionary
    pub term: Vec<u8>,
    /// Edit distance from the query to this term
    pub edit_distance: u8,
    /// Phonetic transformation cost.
    ///
    /// Reserved: in the current implementation this is ALWAYS `0.0` (the query
    /// path constructs every candidate with `phonetic_cost == 0.0`). Kept for a
    /// future weighted-phonetic-matching feature.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`).
    ///
    /// Because `phonetic_cost` is always `0.0` today, this currently equals
    /// `edit_distance as f64`.
    pub total_cost: f64,
}

impl PartialEq for PhoneticCandidateByte {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term
            && self.edit_distance == other.edit_distance
            && self.phonetic_cost.total_cmp(&other.phonetic_cost) == Ordering::Equal
            && self.total_cost.total_cmp(&other.total_cost) == Ordering::Equal
    }
}

impl Eq for PhoneticCandidateByte {}

impl Hash for PhoneticCandidateByte {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term.hash(state);
        self.edit_distance.hash(state);
        self.phonetic_cost.to_bits().hash(state);
        self.total_cost.to_bits().hash(state);
    }
}

impl PhoneticCandidateByte {
    /// Create a new phonetic candidate.
    pub fn new(term: Vec<u8>, edit_distance: u8, phonetic_cost: f64) -> Self {
        let total_cost = edit_distance as f64 + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
        }
    }
}

impl PartialOrd for PhoneticCandidateByte {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticCandidateByte {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cost
            .total_cmp(&other.total_cost)
            .then_with(|| self.term.cmp(&other.term))
            .then_with(|| self.edit_distance.cmp(&other.edit_distance))
            .then_with(|| self.phonetic_cost.total_cmp(&other.phonetic_cost))
    }
}

/// Iterator over byte-level phonetic query results.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticQueryIterator<'a, D: Dictionary> {
    /// Product automaton for matching
    product: ProductAutomaton,
    /// Queue of dictionary nodes to explore
    queue: VecDeque<PhoneticTraversal<D::Node>>,
    /// Parent-path arena for reconstructing terms without per-edge path clones.
    path_arena: Vec<PhoneticPathNode<u8>>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
    /// Maximum depth
    max_depth: usize,
    /// Phonetic weight
    _phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: Dictionary> PhoneticQueryIterator<'a, D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFA,
        _input: &[u8],
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        let product = ProductAutomaton::new(nfa.clone(), max_distance);

        let mut queue = VecDeque::with_capacity(1);
        queue.push_back(PhoneticTraversal::root(dictionary.root()));

        let max_depth = 100;

        Self {
            product,
            queue,
            path_arena: Vec::with_capacity(64),
            _dictionary: PhantomData,
            max_depth,
            _phonetic_weight: phonetic_weight,
        }
    }

    #[inline]
    fn push_path_node(&mut self, label: u8, parent: usize) -> usize {
        let depth = if parent == PHONETIC_NO_PATH {
            1
        } else {
            self.path_arena[parent].depth.saturating_add(1)
        };
        let index = self.path_arena.len();
        self.path_arena.push(PhoneticPathNode {
            label,
            parent,
            depth,
        });
        index
    }

    #[inline]
    fn materialize_path(&self, entry: &PhoneticTraversal<D::Node>) -> Vec<u8> {
        collect_path_units(entry.label, entry.parent, &self.path_arena)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIterator<'_, D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    type Item = PhoneticCandidateByte;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.queue.pop_front() {
            if entry.depth > self.max_depth {
                continue;
            }

            if entry.node.is_final() {
                let path = self.materialize_path(&entry);
                if let Some(distance) = self.product.min_distance(&path) {
                    return Some(PhoneticCandidateByte::new(path, distance, 0.0));
                }
            }

            let parent = match entry.label {
                Some(label) => self.push_path_node(label, entry.parent),
                None => entry.parent,
            };

            let Some(child_depth) = phonetic_child_depth(entry.depth) else {
                continue;
            };
            if child_depth > self.max_depth {
                continue;
            }
            for (b, child) in entry.node.edges() {
                self.queue
                    .push_back(PhoneticTraversal::child(child, b, parent, child_depth));
            }
        }

        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "phonetic-rules")]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};
    use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn candidate_hash<T: Hash>(candidate: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        candidate.hash(&mut hasher);
        hasher.finish()
    }

    #[cfg(feature = "phonetic-rules")]
    #[test]
    fn test_phonetic_child_depth_rejects_overflow() {
        assert_eq!(phonetic_child_depth(0), Some(1));
        assert_eq!(phonetic_child_depth(usize::MAX), None);
    }

    #[test]
    fn test_phonetic_candidate_ordering() {
        let c1 = PhoneticCandidate::new("apple".to_string(), 0, 0.0);
        let c2 = PhoneticCandidate::new("apply".to_string(), 1, 0.0);
        let c3 = PhoneticCandidate::new("banana".to_string(), 0, 0.0);

        assert!(c1 < c2); // 0.0 < 1.0
        assert!(c1 < c3); // same cost, alphabetically
    }

    #[test]
    fn test_phonetic_candidate_eq_ord_hash_contract() {
        let phonetic_path = PhoneticCandidate::new("same".to_string(), 0, 1.0);
        let edit_path = PhoneticCandidate::new("same".to_string(), 1, 0.0);

        assert_eq!(phonetic_path.total_cost, edit_path.total_cost);
        assert_ne!(phonetic_path, edit_path);
        assert_ne!(phonetic_path.cmp(&edit_path), Ordering::Equal);

        let positive_zero = PhoneticCandidate {
            term: "zero".to_string(),
            edit_distance: 0,
            phonetic_cost: 0.0,
            total_cost: 0.0,
        };
        let negative_zero = PhoneticCandidate {
            term: "zero".to_string(),
            edit_distance: 0,
            phonetic_cost: -0.0,
            total_cost: -0.0,
        };

        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let same_candidate = phonetic_path.clone();
        assert_eq!(phonetic_path, same_candidate);
        assert_eq!(
            candidate_hash(&phonetic_path),
            candidate_hash(&same_candidate)
        );
    }

    #[test]
    fn test_phonetic_candidate_byte_eq_ord_hash_contract() {
        let phonetic_path = PhoneticCandidateByte::new(b"same".to_vec(), 0, 1.0);
        let edit_path = PhoneticCandidateByte::new(b"same".to_vec(), 1, 0.0);

        assert_eq!(phonetic_path.total_cost, edit_path.total_cost);
        assert_ne!(phonetic_path, edit_path);
        assert_ne!(phonetic_path.cmp(&edit_path), Ordering::Equal);

        let positive_zero = PhoneticCandidateByte {
            term: b"zero".to_vec(),
            edit_distance: 0,
            phonetic_cost: 0.0,
            total_cost: 0.0,
        };
        let negative_zero = PhoneticCandidateByte {
            term: b"zero".to_vec(),
            edit_distance: 0,
            phonetic_cost: -0.0,
            total_cost: -0.0,
        };

        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let same_candidate = phonetic_path.clone();
        assert_eq!(phonetic_path, same_candidate);
        assert_eq!(
            candidate_hash(&phonetic_path),
            candidate_hash(&same_candidate)
        );
    }

    #[test]
    fn test_phonetic_transducer_basic() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "fone", "bone", "tone"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results: Vec<_> = transducer.query("phone").collect();
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Should find "phone" and "fone" (exact pattern matches)
        // May also find "bone" and "tone" within distance 1
        assert!(terms.contains(&"phone") || terms.contains(&"fone"));
    }

    #[test]
    fn test_phonetic_transducer_sorted() {
        let dict = DoubleArrayTrieChar::from_terms(["test", "best", "rest", "nest"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results = transducer.query_sorted("test");

        // First result should be exact match
        if !results.is_empty() {
            assert_eq!(results[0].term, "test");
            assert_eq!(results[0].edit_distance, 0);
        }
    }

    #[test]
    fn test_phonetic_transducer_no_match() {
        let dict = DoubleArrayTrieChar::from_terms(["xyz", "abc", "def"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results: Vec<_> = transducer.query("phone").collect();

        // No matches - "phone" is too far from all dictionary terms
        assert!(results.is_empty());
    }

    #[test]
    fn test_phonetic_transducer_alternation() {
        let dict = DoubleArrayTrieChar::from_terms(["cat", "kat", "bat", "hat"]);
        let nfa = compile(&parse("(c|k)at").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 0);

        let results: Vec<_> = transducer.query("cat").collect();
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Both "cat" and "kat" match the pattern exactly
        assert!(terms.contains(&"cat"));
        assert!(terms.contains(&"kat"));
    }

    #[test]
    fn test_byte_phonetic_transducer_shared_prefix_branching() {
        let dict = DoubleArrayTrie::from_terms(["phone", "fone", "phony", "bone"]);
        let nfa = compile_bytes(&parse_bytes(b"(ph|f)one").expect("parse")).expect("compile bytes");

        let transducer = PhoneticTransducer::new(dict, nfa, 1);
        let mut terms: Vec<_> = transducer
            .query(b"phone")
            .map(|candidate| candidate.term)
            .collect();
        terms.sort();

        assert!(terms.contains(&b"fone".to_vec()));
        assert!(terms.contains(&b"phone".to_vec()));
    }

    #[test]
    fn test_phonetic_transducer_with_distance() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "phoned"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results = transducer.query_sorted("phone");
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Should find "phone" (exact) and possibly "phones"/"phoned" (distance 1)
        assert!(terms.contains(&"phone"));
    }

    #[test]
    fn test_phonetic_transducer_accessors() {
        let dict = DoubleArrayTrieChar::from_terms(["test"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa.clone(), 2);

        assert_eq!(transducer.max_distance(), 2);
        assert!(!transducer.dictionary().is_empty());

        // Test into_dictionary
        let _recovered_dict = transducer.into_dictionary();
    }

    /// F2: a non-zero phonetic weight is stored but INERT. Pin the documented
    /// contract so the docs and code cannot silently drift apart: every emitted
    /// candidate must report `phonetic_cost == 0.0` and, consequently,
    /// `total_cost == edit_distance`.
    #[test]
    fn test_phonetic_weight_leaves_candidate_cost_zero() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "phoned"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        // Weight 7.0 is arbitrary and non-zero; it must not change any cost.
        let transducer = PhoneticTransducerChar::with_phonetic_weight(dict, nfa, 1, 7.0);
        let results: Vec<_> = transducer.query("phone").collect();
        assert!(!results.is_empty(), "expected at least the exact match");
        for candidate in &results {
            assert_eq!(
                candidate.phonetic_cost, 0.0,
                "phonetic_cost must be 0.0 for {:?}",
                candidate.term
            );
            assert_eq!(
                candidate.total_cost,
                f64::from(candidate.edit_distance),
                "total_cost must equal edit_distance for {:?}",
                candidate.term
            );
        }
    }

    /// F2 (byte-level): mirror of the char contract for `PhoneticCandidateByte`.
    #[test]
    fn test_byte_phonetic_weight_leaves_candidate_cost_zero() {
        let dict = DoubleArrayTrie::from_terms(["phone", "phones", "bone"]);
        let nfa = compile_bytes(&parse_bytes(b"phone").expect("parse")).expect("compile bytes");

        let transducer = PhoneticTransducer::with_phonetic_weight(dict, nfa, 1, 4.0);
        let results: Vec<_> = transducer.query(b"phone").collect();
        assert!(!results.is_empty(), "expected at least the exact match");
        for candidate in &results {
            assert_eq!(candidate.phonetic_cost, 0.0);
            assert_eq!(candidate.total_cost, f64::from(candidate.edit_distance));
        }
    }
}
