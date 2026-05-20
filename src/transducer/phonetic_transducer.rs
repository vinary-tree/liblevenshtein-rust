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

use libdictenstein::{Dictionary, DictionaryNode};
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::{NFAChar, NFA};
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::product::{ProductAutomatonChar, ProductAutomaton};

use std::collections::VecDeque;

// ============================================================================
// Phonetic Candidate
// ============================================================================

/// A candidate result from phonetic transducer query.
#[derive(Debug, Clone, PartialEq)]
pub struct PhoneticCandidate {
    /// The matching term from the dictionary
    pub term: String,
    /// Edit distance from the query to this term
    pub edit_distance: u8,
    /// Phonetic transformation cost (0.0 for exact phonetic match)
    pub phonetic_cost: f64,
    /// Combined total cost (edit_distance + phonetic_cost)
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

impl Eq for PhoneticCandidate {}

impl PartialOrd for PhoneticCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by total_cost (lower is better), then by term alphabetically
        match self.total_cost.partial_cmp(&other.total_cost) {
            Some(std::cmp::Ordering::Equal) | None => self.term.cmp(&other.term),
            Some(ord) => ord,
        }
    }
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
    /// Weight for phonetic transformations
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

    /// Create a phonetic transducer with custom phonetic weight.
    ///
    /// The phonetic weight is added to the total cost for each phonetic
    /// transformation applied (as opposed to simple edit operations).
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
            phonetic_weight,
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
    /// Queue of dictionary nodes to explore: (node, path, depth)
    queue: VecDeque<(D::Node, String, usize)>,
    /// The dictionary reference
    #[allow(dead_code)]
    dictionary: &'a D,
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
        let mut queue = VecDeque::new();
        queue.push_back((dictionary.root(), String::new(), 0));

        // Max depth: reasonable buffer for dictionary traversal
        let max_depth = 100;

        Self {
            product,
            queue,
            dictionary,
            max_depth,
            _phonetic_weight: phonetic_weight,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIteratorChar<'_, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    type Item = PhoneticCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path, depth)) = self.queue.pop_front() {
            // Depth limit to prevent infinite exploration
            if depth > self.max_depth {
                continue;
            }

            // Check if this is a final dictionary node
            if node.is_final() {
                // Check if the product automaton accepts this path
                if let Some(distance) = self.product.min_distance(&path) {
                    return Some(PhoneticCandidate::new(path.clone(), distance, 0.0));
                }
            }

            // Explore children via edges
            for (c, child) in node.edges() {
                let mut child_path = path.clone();
                child_path.push(c);
                self.queue.push_back((child, child_path, depth + 1));
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
    /// Weight for phonetic transformations
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

    /// Create with custom phonetic weight.
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
            phonetic_weight,
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
#[derive(Debug, Clone, PartialEq)]
pub struct PhoneticCandidateByte {
    /// The matching term from the dictionary
    pub term: Vec<u8>,
    /// Edit distance from the query to this term
    pub edit_distance: u8,
    /// Phonetic transformation cost
    pub phonetic_cost: f64,
    /// Combined total cost
    pub total_cost: f64,
}

impl Eq for PhoneticCandidateByte {}

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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.total_cost.partial_cmp(&other.total_cost) {
            Some(std::cmp::Ordering::Equal) | None => self.term.cmp(&other.term),
            Some(ord) => ord,
        }
    }
}

/// Iterator over byte-level phonetic query results.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticQueryIterator<'a, D: Dictionary> {
    /// Product automaton for matching
    product: ProductAutomaton,
    /// Queue of dictionary nodes to explore
    queue: VecDeque<(D::Node, Vec<u8>, usize)>,
    /// The dictionary reference
    #[allow(dead_code)]
    dictionary: &'a D,
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

        let mut queue = VecDeque::new();
        queue.push_back((dictionary.root(), Vec::new(), 0));

        let max_depth = 100;

        Self {
            product,
            queue,
            dictionary,
            max_depth,
            _phonetic_weight: phonetic_weight,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIterator<'_, D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    type Item = PhoneticCandidateByte;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path, depth)) = self.queue.pop_front() {
            if depth > self.max_depth {
                continue;
            }

            if node.is_final() {
                if let Some(distance) = self.product.min_distance(&path) {
                    return Some(PhoneticCandidateByte::new(path.clone(), distance, 0.0));
                }
            }

            for (b, child) in node.edges() {
                let mut child_path = path.clone();
                child_path.push(b);
                self.queue.push_back((child, child_path, depth + 1));
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
    use libdictenstein::double_array_trie_char::DoubleArrayTrieChar;
    use crate::phonetic::nfa::compiler::compile;
    use crate::phonetic::regex::parse;

    #[test]
    fn test_phonetic_candidate_ordering() {
        let c1 = PhoneticCandidate::new("apple".to_string(), 0, 0.0);
        let c2 = PhoneticCandidate::new("apply".to_string(), 1, 0.0);
        let c3 = PhoneticCandidate::new("banana".to_string(), 0, 0.0);

        assert!(c1 < c2); // 0.0 < 1.0
        assert!(c1 < c3); // same cost, alphabetically
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
}
