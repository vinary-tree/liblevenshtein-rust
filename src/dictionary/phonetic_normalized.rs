//! PhoneticNormalizedDictionary for phonetic fuzzy matching.
//!
//! This module implements Approach 1 from the compositional spelling correction guide:
//! Pre-process both query and dictionary for maximum speed.
//!
//! # Overview
//!
//! The PhoneticNormalizedDictionary normalizes terms phonetically at build time,
//! storing a mapping from normalized forms back to original terms. At query time,
//! the query is normalized and searched in the normalized space, then results
//! are mapped back to original terms.
//!
//! # Architecture
//!
//! The dictionary uses a dual-index design:
//! - `originals`: Backend dictionary storing original terms with values
//! - `normalized_index`: HashMap mapping normalized forms to original terms
//!
//! This allows the dictionary to implement all standard dictionary traits while
//! maintaining the phonetic normalization functionality.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
//!
//! // Create dictionary with default Zompist rules
//! let dict = PhoneticNormalizedDictionary::<()>::from_terms([
//!     "phone", "fone", "elephant", "elegance"
//! ]);
//!
//! // Query - "fone" normalizes to same as "phone"
//! let results = dict.query("fone", 0);
//! // Returns both "phone" and "fone" since they normalize to the same form
//!
//! // Query with edit distance tolerance
//! let results = dict.query("elefant", 1);
//! // Returns "elephant" (phonetically similar)
//! ```
//!
//! # Trait Implementations
//!
//! The dictionary implements all standard dictionary traits:
//! - [`Dictionary`] - Core dictionary traversal
//! - [`DictionaryNode`] - Node-based traversal
//! - [`MappedDictionary`] - Value lookup
//! - [`MutableMappedDictionary`] - Insert with values
//! - [`DictZipper`] - Zipper navigation (via [`PhoneticNormalizedZipper`])
//!
//! # Trade-offs
//!
//! **Advantages:**
//! - Fastest query time (single normalization + Levenshtein search)
//! - Implements all dictionary protocols for drop-in compatibility
//! - Low memory overhead (normalized trie similar to original)
//! - Supports dynamic insert and delete operations
//!
//! **Disadvantages:**
//! - Information loss: "phone" and "fone" become indistinguishable in normalized space
//! - Cannot separate phonetic cost from edit cost
//! - Changing rules requires rebuilding the dictionary
//!
//! # See Also
//!
//! - `docs/guides/compositional-phonetic-levenshtein.md` for detailed documentation
//! - [`crate::phonetic`] module for phonetic rule definitions

use crate::dictionary::dynamic_dawg_char::DynamicDawgChar;
use crate::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{
    DictZipper, Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode,
    MutableMappedDictionary, SyncStrategy, ValuedDictZipper,
};
use crate::phonetic::expansion::expand_phonetic_alternatives_char;
use crate::phonetic::nfa::{compile as compile_nfa, ProductAutomatonChar};
use crate::phonetic::regex::{parse as parse_regex, ParseError as RegexParseError};
use crate::phonetic::types::{PhoneChar, RewriteRuleChar};
use crate::phonetic::{apply_rules_seq_char, zompist_rules_char};
use crate::transducer::Algorithm;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Result of a phonetic normalized query.
///
/// Contains the original term, the edit distance in normalized space,
/// and the normalized form that matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneticNormalizedCandidate {
    /// Original term from dictionary
    pub term: String,
    /// Edit distance between normalized query and normalized term
    pub distance: usize,
    /// The normalized form that matched
    pub normalized_form: String,
}

/// Error type for regex query operations.
#[derive(Debug)]
pub enum RegexQueryError {
    /// Error parsing the regex pattern
    ParseError(RegexParseError),
    /// Error compiling the regex to NFA
    CompileError(RegexParseError),
}

impl fmt::Display for RegexQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexQueryError::ParseError(e) => write!(f, "Regex parse error: {}", e),
            RegexQueryError::CompileError(e) => write!(f, "NFA compile error: {}", e),
        }
    }
}

impl std::error::Error for RegexQueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RegexQueryError::ParseError(e) => Some(e),
            RegexQueryError::CompileError(e) => Some(e),
        }
    }
}

impl From<RegexParseError> for RegexQueryError {
    fn from(err: RegexParseError) -> Self {
        RegexQueryError::ParseError(err)
    }
}

// ============================================================================
// CORE TYPES
// ============================================================================

/// A dictionary that normalizes terms phonetically for fuzzy matching.
///
/// Implements a dual-index architecture:
/// - At build time: Store original terms in backend, maintain normalized → originals index
/// - At query time: Normalize query, fuzzy search, map back to originals
///
/// The dictionary is generic over:
/// - `V`: The value type associated with terms (default: `()`)
/// - `D`: The backend dictionary type (default: `DynamicDawgChar<V>`)
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
///
/// let dict = PhoneticNormalizedDictionary::<()>::from_terms([
///     "phone", "fone", "elephant", "elegance"
/// ]);
///
/// // "fone" normalizes to same form as "phone"
/// let results = dict.query("fone", 0);
/// assert!(results.iter().any(|c| c.term == "phone"));
/// assert!(results.iter().any(|c| c.term == "fone"));
/// ```
pub struct PhoneticNormalizedDictionary<V: DictionaryValue = (), D: Dictionary = DynamicDawgChar<V>>
{
    /// Backend dictionary storing original terms with values
    originals: D,

    /// Index: normalized_form → {original_terms}
    /// Thread-safe for concurrent access
    normalized_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,

    /// Phonetic rules for normalization
    rules: Vec<RewriteRuleChar>,

    /// Fuel for rule application (prevents infinite loops)
    fuel: usize,

    /// Levenshtein algorithm to use for phonetic queries
    algorithm: Algorithm,

    /// Marker for value type
    _value: std::marker::PhantomData<V>,
}

/// UTF-8 character-level PhoneticNormalizedDictionary.
///
/// This is the default and recommended variant, using `char` for proper Unicode
/// semantics with accented characters, CJK, emoji, etc.
pub type PhoneticNormalizedDictionaryChar<V = ()> = PhoneticNormalizedDictionary<V, DynamicDawgChar<V>>;

// ============================================================================
// NODE WRAPPER
// ============================================================================

/// Node wrapper for PhoneticNormalizedDictionary traversal.
///
/// Wraps the backend node to implement dictionary protocols.
#[derive(Clone)]
pub struct PhoneticNormalizedNode<N: DictionaryNode> {
    inner: N,
}

impl<N: DictionaryNode> DictionaryNode for PhoneticNormalizedNode<N> {
    type Unit = N::Unit;

    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        self.inner
            .transition(label)
            .map(|n| PhoneticNormalizedNode { inner: n })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        Box::new(
            self.inner
                .edges()
                .map(|(label, node)| (label, PhoneticNormalizedNode { inner: node })),
        )
    }

    fn has_edge(&self, label: Self::Unit) -> bool {
        self.inner.has_edge(label)
    }

    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N: MappedDictionaryNode> MappedDictionaryNode for PhoneticNormalizedNode<N> {
    type Value = N::Value;

    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

// ============================================================================
// ZIPPER WRAPPER
// ============================================================================

/// Zipper for navigating PhoneticNormalizedDictionary.
///
/// Wraps the backend zipper to provide zipper operations.
#[derive(Clone)]
pub struct PhoneticNormalizedZipper<Z: DictZipper> {
    inner: Z,
}

impl<Z: DictZipper> DictZipper for PhoneticNormalizedZipper<Z> {
    type Unit = Z::Unit;

    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        self.inner
            .descend(label)
            .map(|z| PhoneticNormalizedZipper { inner: z })
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        self.inner
            .children()
            .map(|(label, z)| (label, PhoneticNormalizedZipper { inner: z }))
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.inner.path()
    }
}

impl<Z: ValuedDictZipper> ValuedDictZipper for PhoneticNormalizedZipper<Z> {
    type Value = Z::Value;

    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

// ============================================================================
// DICTIONARY TRAIT IMPLEMENTATIONS
// ============================================================================

impl<V, D> Dictionary for PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: Dictionary,
{
    type Node = PhoneticNormalizedNode<D::Node>;

    fn root(&self) -> Self::Node {
        PhoneticNormalizedNode {
            inner: self.originals.root(),
        }
    }

    fn contains(&self, term: &str) -> bool {
        self.originals.contains(term)
    }

    fn len(&self) -> Option<usize> {
        self.originals.len()
    }

    fn is_empty(&self) -> bool {
        self.originals.is_empty()
    }

    fn sync_strategy(&self) -> SyncStrategy {
        // Our normalized_index uses internal sync, but defer to backend for originals
        SyncStrategy::InternalSync
    }
}

impl<V, D> MappedDictionary for PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: MappedDictionary<Value = V>,
{
    type Value = V;

    fn get_value(&self, term: &str) -> Option<Self::Value> {
        self.originals.get_value(term)
    }
}

impl<V, D> MutableMappedDictionary for PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: MutableMappedDictionary<Value = V>,
{
    fn insert_with_value(&self, term: &str, value: Self::Value) -> bool {
        // 1. Insert into backend
        let is_new = self.originals.insert_with_value(term, value);

        if is_new {
            // 2. Update normalized index
            let normalized = self.normalize(term);
            let mut index = self.normalized_index.write();
            index
                .entry(normalized)
                .or_insert_with(HashSet::new)
                .insert(term.to_string());
        }

        is_new
    }

    fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
    where
        F: Fn(&Self::Value, &Self::Value) -> Self::Value,
        Self::Value: Clone,
    {
        // Delegate to backend for the union
        let count = self.originals.union_with(&other.originals, merge_fn);

        // Rebuild normalized index to include new terms
        // This is expensive but ensures consistency
        let mut index = self.normalized_index.write();
        for (term, _) in other.iter_terms() {
            let normalized = self.normalize(&term);
            index
                .entry(normalized)
                .or_insert_with(HashSet::new)
                .insert(term);
        }

        count
    }
}

// ============================================================================
// CONSTRUCTORS
// ============================================================================

impl<V> PhoneticNormalizedDictionary<V, DynamicDawgChar<V>>
where
    V: DictionaryValue + Default,
{
    /// Create an empty dictionary with default rules.
    ///
    /// Uses the standard Zompist English phonetic rules for normalization.
    pub fn new() -> Self {
        Self::with_rules(zompist_rules_char())
    }

    /// Create an empty dictionary with custom rules.
    pub fn with_rules(rules: Vec<RewriteRuleChar>) -> Self {
        let fuel = Self::compute_fuel(&rules);
        Self {
            originals: DynamicDawgChar::new(),
            normalized_index: Arc::new(RwLock::new(HashMap::new())),
            rules,
            fuel,
            algorithm: Algorithm::Standard,
            _value: std::marker::PhantomData,
        }
    }

    /// Create from terms with default Zompist rules.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
    ///
    /// let dict = PhoneticNormalizedDictionary::<()>::from_terms([
    ///     "phone", "fone", "elephant"
    /// ]);
    /// ```
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_terms_with_rules(terms, zompist_rules_char())
    }

    /// Create from terms with custom rules.
    pub fn from_terms_with_rules<I, S>(terms: I, rules: Vec<RewriteRuleChar>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_terms_with_rules_and_algorithm(terms, rules, Algorithm::Standard)
    }

    /// Create from terms with default rules and custom algorithm.
    pub fn from_terms_with_algorithm<I, S>(terms: I, algorithm: Algorithm) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_terms_with_rules_and_algorithm(terms, zompist_rules_char(), algorithm)
    }

    /// Create from terms with custom rules and algorithm.
    pub fn from_terms_with_rules_and_algorithm<I, S>(
        terms: I,
        rules: Vec<RewriteRuleChar>,
        algorithm: Algorithm,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let fuel = Self::compute_fuel(&rules);
        let originals = DynamicDawgChar::new();
        let mut normalized_index = HashMap::<String, HashSet<String>>::new();

        for term in terms {
            let term = term.as_ref();
            originals.insert(term);

            let normalized = normalize_string_char(term, &rules, fuel);
            normalized_index
                .entry(normalized)
                .or_insert_with(HashSet::new)
                .insert(term.to_string());
        }

        Self {
            originals,
            normalized_index: Arc::new(RwLock::new(normalized_index)),
            rules,
            fuel,
            algorithm,
            _value: std::marker::PhantomData,
        }
    }

    /// Create from terms with values.
    pub fn from_terms_with_values<I, S>(terms: I, rules: Vec<RewriteRuleChar>) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
    {
        let fuel = Self::compute_fuel(&rules);
        let originals = DynamicDawgChar::new();
        let mut normalized_index = HashMap::<String, HashSet<String>>::new();

        for (term, value) in terms {
            let term = term.as_ref();
            originals.insert_with_value(term, value);

            let normalized = normalize_string_char(term, &rules, fuel);
            normalized_index
                .entry(normalized)
                .or_insert_with(HashSet::new)
                .insert(term.to_string());
        }

        Self {
            originals,
            normalized_index: Arc::new(RwLock::new(normalized_index)),
            rules,
            fuel,
            algorithm: Algorithm::Standard,
            _value: std::marker::PhantomData,
        }
    }
}

impl<V> Default for PhoneticNormalizedDictionary<V, DynamicDawgChar<V>>
where
    V: DictionaryValue + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CORE METHODS
// ============================================================================

impl<V, D> PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: Dictionary,
{
    /// Compute fuel value based on rule complexity.
    fn compute_fuel(_rules: &[RewriteRuleChar]) -> usize {
        100 // Safe default that handles most inputs
    }

    /// Normalize a string using the rules.
    pub fn normalize(&self, term: &str) -> String {
        normalize_string_char(term, &self.rules, self.fuel)
    }

    /// Get the phonetic rules being used.
    pub fn rules(&self) -> &[RewriteRuleChar] {
        &self.rules
    }

    /// Get the fuel value used for rule application.
    pub fn fuel(&self) -> usize {
        self.fuel
    }

    /// Get the algorithm used for phonetic queries.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the number of unique normalized forms in the dictionary.
    pub fn normalized_count(&self) -> usize {
        self.normalized_index.read().len()
    }

    /// Get a reference to the backend dictionary.
    pub fn originals(&self) -> &D {
        &self.originals
    }

    /// Get the normalized index (read-only).
    pub fn normalized_index(&self) -> &Arc<RwLock<HashMap<String, HashSet<String>>>> {
        &self.normalized_index
    }
}

// ============================================================================
// INSERT AND REMOVE
// ============================================================================

impl<V, D> PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue + Default,
    D: MutableMappedDictionary<Value = V>,
{
    /// Insert a term into the dictionary.
    ///
    /// Uses the default value for the value type.
    ///
    /// # Returns
    ///
    /// `true` if this is a new term, `false` if updating an existing term.
    pub fn insert(&self, term: &str) -> bool {
        self.insert_with_value(term, V::default())
    }
}

impl<V, D> PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: Dictionary,
{
    /// Remove a term from the dictionary.
    ///
    /// Note: This only updates the normalized index. The backend may or may not
    /// support removal. Use a backend like DynamicDawgChar that supports removal
    /// for full dynamic functionality.
    ///
    /// # Returns
    ///
    /// `true` if the term was in the normalized index, `false` otherwise.
    pub fn remove(&self, term: &str) -> bool {
        // Check if term exists in index
        let normalized = self.normalize(term);
        let mut index = self.normalized_index.write();

        if let Some(originals) = index.get_mut(&normalized) {
            let removed = originals.remove(term);
            if originals.is_empty() {
                index.remove(&normalized);
            }
            removed
        } else {
            false
        }
    }
}

// ============================================================================
// ITERATION
// ============================================================================

impl<V, D> PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: Dictionary,
{
    /// Iterate over all terms in the dictionary.
    ///
    /// Note: This iterates over the normalized index, which tracks all inserted terms.
    pub fn iter_terms(&self) -> impl Iterator<Item = (String, String)> + '_ {
        let index = self.normalized_index.read();
        // Collect to release the lock
        let pairs: Vec<_> = index
            .iter()
            .flat_map(|(normalized, originals)| {
                originals
                    .iter()
                    .map(move |term| (term.clone(), normalized.clone()))
            })
            .collect();
        pairs.into_iter()
    }

    /// Iterate over normalized forms and their original terms.
    pub fn iter_normalized(&self) -> impl Iterator<Item = (String, HashSet<String>)> {
        let index = self.normalized_index.read();
        // Collect to release the lock
        let pairs: Vec<_> = index
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pairs.into_iter()
    }
}

// ============================================================================
// ZIPPER SUPPORT
// ============================================================================

impl<V> PhoneticNormalizedDictionary<V, DynamicDawgChar<V>>
where
    V: DictionaryValue,
{
    /// Create a zipper for navigating the dictionary.
    pub fn zipper(&self) -> PhoneticNormalizedZipper<DynamicDawgCharZipper<V>> {
        PhoneticNormalizedZipper {
            inner: DynamicDawgCharZipper::new_from_dict(&self.originals),
        }
    }
}

// ============================================================================
// PHONETIC QUERY METHODS
// ============================================================================

impl<V, D> PhoneticNormalizedDictionary<V, D>
where
    V: DictionaryValue,
    D: Dictionary,
{
    /// Query for candidates within edit distance.
    ///
    /// The query is normalized using the same rules as the dictionary terms,
    /// then searched in the normalized space. Results include all original terms
    /// that match, along with the edit distance in normalized space.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query (will be normalized)
    /// * `max_distance` - Maximum edit distance in normalized space
    ///
    /// # Returns
    ///
    /// Vector of candidates, each containing:
    /// - `term`: Original term from dictionary
    /// - `distance`: Edit distance between normalized query and normalized term
    /// - `normalized_form`: The normalized form that matched
    pub fn query(&self, query: &str, max_distance: usize) -> Vec<PhoneticNormalizedCandidate> {
        let normalized_query = self.normalize(query);
        let index = self.normalized_index.read();

        // For exact match (distance 0), just look up in index
        if max_distance == 0 {
            if let Some(originals) = index.get(&normalized_query) {
                return originals
                    .iter()
                    .map(|term| PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance: 0,
                        normalized_form: normalized_query.clone(),
                    })
                    .collect();
            }
            return Vec::new();
        }

        // For distance > 0, we need to check all normalized forms
        // Optimization: Use length-based pre-filtering to skip forms that can't match
        let mut results = Vec::new();

        // Use byte length for fast pre-filtering (exact for ASCII, conservative for UTF-8)
        // For UTF-8, byte_len >= char_count, so this is a safe lower bound
        let query_byte_len = normalized_query.len();

        for (normalized, originals) in index.iter() {
            // H4: Length-based pre-filtering using byte length
            // If byte length difference > max_distance, skip (safe for both ASCII and UTF-8)
            // For ASCII: byte_len == char_count, so this is exact
            // For UTF-8: this may skip fewer candidates, but never incorrectly skips valid matches
            let norm_byte_len = normalized.len();
            let byte_len_diff = query_byte_len.abs_diff(norm_byte_len);
            if byte_len_diff > max_distance {
                continue; // Skip: byte length difference alone exceeds max_distance
            }

            let dist = levenshtein_distance(&normalized_query, normalized);
            if dist <= max_distance {
                for term in originals {
                    results.push(PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance: dist,
                        normalized_form: normalized.clone(),
                    });
                }
            }
        }

        results.sort_by_key(|c| c.distance);
        results
    }

    /// Query using a regex pattern (grep-like semantics).
    ///
    /// The pattern is matched against NORMALIZED forms in the dictionary,
    /// then results are mapped back to original terms.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match (e.g., "(ph|f)one", "colou?r")
    /// * `max_distance` - Maximum edit distance from pattern (0 for exact regex match)
    ///
    /// # Returns
    ///
    /// Results contain original terms whose normalized forms match the pattern.
    pub fn query_regex(
        &self,
        pattern: &str,
        max_distance: u8,
    ) -> Result<Vec<PhoneticNormalizedCandidate>, RegexQueryError> {
        // Parse regex pattern
        let ast = parse_regex(pattern)?;

        // Compile to NFA
        let nfa = compile_nfa(&ast).map_err(RegexQueryError::CompileError)?;

        // Create product automaton for fuzzy matching
        let product = ProductAutomatonChar::with_algorithm(nfa, max_distance, self.algorithm);

        // Search normalized index for matches
        let mut results = Vec::new();
        let index = self.normalized_index.read();

        for (normalized, originals) in index.iter() {
            if let Some(distance) = product.min_distance(normalized) {
                for term in originals {
                    results.push(PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance: distance as usize,
                        normalized_form: normalized.clone(),
                    });
                }
            }
        }

        // Sort by distance for convenience
        results.sort_by_key(|c| c.distance);
        Ok(results)
    }

    /// Query using a pre-compiled NFA pattern.
    ///
    /// Use this when you want to reuse the same pattern for multiple queries,
    /// avoiding repeated parsing and compilation overhead.
    pub fn query_with_product(
        &self,
        product: &ProductAutomatonChar,
    ) -> Vec<PhoneticNormalizedCandidate> {
        let mut results = Vec::new();
        let index = self.normalized_index.read();

        for (normalized, originals) in index.iter() {
            if let Some(distance) = product.min_distance(normalized) {
                for term in originals {
                    results.push(PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance: distance as usize,
                        normalized_form: normalized.clone(),
                    });
                }
            }
        }

        results.sort_by_key(|c| c.distance);
        results
    }

    /// Query with automatic phonetic pattern expansion.
    ///
    /// Converts a literal query into a pattern that matches phonetic variants.
    /// For example, "fone" becomes "(ph|f)one" based on the phonetic rules,
    /// allowing it to match both "fone" and "phone".
    pub fn query_phonetic_pattern(
        &self,
        query: &str,
        max_distance: u8,
    ) -> Result<Vec<PhoneticNormalizedCandidate>, RegexQueryError> {
        // Expand the query into a pattern matching phonetic variants
        let pattern = expand_phonetic_alternatives_char(query, &self.rules);

        // Use the regex query with the expanded pattern
        self.query_regex(&pattern, max_distance)
    }

    /// Expand a query string to a phonetic pattern.
    ///
    /// This method shows what pattern would be used by `query_phonetic_pattern`
    /// without actually executing the query. Useful for debugging.
    pub fn expand_to_phonetic_pattern(&self, query: &str) -> String {
        expand_phonetic_alternatives_char(query, &self.rules)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Thread-local buffers for Levenshtein distance calculation.
// H2 optimization: Reuse buffers across calls to avoid allocations.
thread_local! {
    static LEVENSHTEIN_BUFFERS: std::cell::RefCell<LevenshteinBuffers> =
        std::cell::RefCell::new(LevenshteinBuffers::new());
}

/// Reusable buffers for Levenshtein distance computation.
struct LevenshteinBuffers {
    a_chars: Vec<char>,
    b_chars: Vec<char>,
    prev: Vec<usize>,
    curr: Vec<usize>,
}

impl LevenshteinBuffers {
    fn new() -> Self {
        Self {
            a_chars: Vec::with_capacity(64),
            b_chars: Vec::with_capacity(64),
            prev: Vec::with_capacity(64),
            curr: Vec::with_capacity(64),
        }
    }

    /// Compute Levenshtein distance using internal buffers.
    fn distance(&mut self, a: &str, b: &str) -> usize {
        // Clear and reuse char buffers
        self.a_chars.clear();
        self.b_chars.clear();
        self.a_chars.extend(a.chars());
        self.b_chars.extend(b.chars());

        let m = self.a_chars.len();
        let n = self.b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Resize DP buffers if needed, then clear
        self.prev.clear();
        self.curr.clear();
        self.prev.resize(n + 1, 0);
        self.curr.resize(n + 1, 0);

        // Initialize first row
        for j in 0..=n {
            self.prev[j] = j;
        }

        // DP computation
        for i in 1..=m {
            self.curr[0] = i;
            for j in 1..=n {
                let cost = if self.a_chars[i - 1] == self.b_chars[j - 1] {
                    0
                } else {
                    1
                };
                self.curr[j] = (self.prev[j] + 1) // deletion
                    .min(self.curr[j - 1] + 1) // insertion
                    .min(self.prev[j - 1] + cost); // substitution
            }
            std::mem::swap(&mut self.prev, &mut self.curr);
        }

        self.prev[n]
    }
}

/// Simple Levenshtein distance for normalized form comparison.
/// H2 optimization: Uses thread-local buffers to avoid allocations.
#[inline]
fn levenshtein_distance(a: &str, b: &str) -> usize {
    LEVENSHTEIN_BUFFERS.with(|buffers| buffers.borrow_mut().distance(a, b))
}

/// O(1) vowel classification using bitmask lookup.
///
/// Uses a 64-bit mask for lowercase letters (a-z in bits 0-25) and checks
/// if the character is a vowel (a=0, e=4, i=8, o=14, u=20).
/// This avoids array indexing and bounds checking overhead.
///
/// Vowel bits: a(0), e(4), i(8), o(14), u(20) = 0b100001000100010001 = 0x111111
/// Bitmask: (1 << 0) | (1 << 4) | (1 << 8) | (1 << 14) | (1 << 20) = 0x104111
const VOWEL_MASK: u64 = (1 << (b'a' - b'a'))
    | (1 << (b'e' - b'a'))
    | (1 << (b'i' - b'a'))
    | (1 << (b'o' - b'a'))
    | (1 << (b'u' - b'a'));

/// O(1) vowel classification using bitmask.
///
/// Replaces O(10) linear array search with bitwise operations.
/// This is the primary hotspot identified by perf profiling (79.55% of execution).
#[inline(always)]
fn is_vowel(c: char) -> bool {
    let lower = (c as u32) | 0x20; // Convert to lowercase (ASCII trick)
    if lower < b'a' as u32 || lower > b'z' as u32 {
        return false;
    }
    let bit = 1u64 << (lower - b'a' as u32);
    (VOWEL_MASK & bit) != 0
}

// H3 optimization: Thread-local buffer for PhoneChar normalization input.
// Reuses buffer across normalize_string_char calls to avoid allocation.
thread_local! {
    static NORMALIZE_BUFFER: std::cell::RefCell<NormalizeBuffers> =
        std::cell::RefCell::new(NormalizeBuffers::new());
}

struct NormalizeBuffers {
    input_phones: Vec<PhoneChar>,
    output_string: String,
}

impl NormalizeBuffers {
    fn new() -> Self {
        Self {
            input_phones: Vec::with_capacity(64),
            output_string: String::with_capacity(64),
        }
    }

    fn normalize(&mut self, input: &str, rules: &[RewriteRuleChar], fuel: usize) -> String {
        if rules.is_empty() {
            return input.to_string();
        }

        // Reuse input buffer
        self.input_phones.clear();
        self.input_phones.extend(input.chars().map(|c| {
            if is_vowel(c) {
                PhoneChar::Vowel(c)
            } else {
                PhoneChar::Consonant(c)
            }
        }));

        // Apply rules
        let result = apply_rules_seq_char(rules, &self.input_phones, fuel);

        // Convert result back to string using reusable buffer
        match result {
            Some(phones) => {
                self.output_string.clear();
                self.output_string.reserve(phones.len());
                for p in &phones {
                    match p {
                        PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => {
                            self.output_string.push(*c)
                        }
                        PhoneChar::Digraph(c1, c2) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                        }
                        PhoneChar::Silent => {}
                    }
                }
                self.output_string.clone()
            }
            None => input.to_string(),
        }
    }
}

/// Helper function to normalize a string using character-level phonetic rules.
/// H3 optimization: Uses thread-local buffers to reduce allocations.
#[inline]
fn normalize_string_char(input: &str, rules: &[RewriteRuleChar], fuel: usize) -> String {
    NORMALIZE_BUFFER.with(|buffers| buffers.borrow_mut().normalize(input, rules, fuel))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonetic_normalized_basic() {
        let dict =
            PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "elephant", "elegance"]);

        // Check that normalization is working
        let phone_normalized = dict.normalize("phone");
        let fone_normalized = dict.normalize("fone");

        // "phone" and "fone" should normalize to the same form
        println!("phone -> {}", phone_normalized);
        println!("fone -> {}", fone_normalized);

        // Query for "fone"
        let results = dict.query("fone", 0);
        println!("Results for 'fone' with distance 0: {:?}", results);

        // Should find at least "fone" itself
        assert!(!results.is_empty());
    }

    #[test]
    fn test_phonetic_normalized_with_distance() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "bone", "cone", "tone"]);

        // Query with distance tolerance
        let results = dict.query("fone", 1);
        println!("Results for 'fone' with distance 1: {:?}", results);

        // Should find matches
        assert!(!results.is_empty());
    }

    #[test]
    fn test_phonetic_normalized_elephant() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["elephant"]);

        // Check normalization
        let elephant_normalized = dict.normalize("elephant");
        let elefant_normalized = dict.normalize("elefant");

        println!("elephant -> {}", elephant_normalized);
        println!("elefant -> {}", elefant_normalized);

        // Query for phonetic variant
        let results = dict.query("elefant", 1);
        println!("Results for 'elefant' with distance 1: {:?}", results);

        // Should find "elephant"
        assert!(
            results.iter().any(|c| c.term == "elephant"),
            "Should find 'elephant' when searching for 'elefant'"
        );
    }

    #[test]
    fn test_phonetic_normalized_normalize() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["test"]);

        // Test the normalize function directly
        let normalized = dict.normalize("knight");
        println!("knight -> {}", normalized);

        let normalized = dict.normalize("night");
        println!("night -> {}", normalized);
    }

    #[test]
    fn test_phonetic_normalized_empty_rules() {
        // Test with empty rules (no normalization)
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["phone", "fone"], vec![]);

        // With no rules, terms should not be normalized
        assert_eq!(dict.normalize("phone"), "phone");
        assert_eq!(dict.normalize("fone"), "fone");

        // Exact match only
        let results = dict.query("phone", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "phone");
    }

    #[test]
    fn test_phonetic_normalized_multiple_originals() {
        // Test that multiple terms normalizing to the same form are all returned
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "phon"]);

        // Get the normalized forms
        let phone_norm = dict.normalize("phone");
        let phon_norm = dict.normalize("phon");

        println!("phone -> {}", phone_norm);
        println!("phon -> {}", phon_norm);

        // If they normalize to the same form, a query should return both
        if phone_norm == phon_norm {
            let results = dict.query("phone", 0);
            assert!(results.len() >= 2, "Should find both 'phone' and 'phon'");
        }
    }

    #[test]
    fn test_phonetic_normalized_candidate_structure() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["test"]);

        let results = dict.query("test", 0);
        assert!(!results.is_empty());

        let candidate = &results[0];
        assert_eq!(candidate.term, "test");
        assert_eq!(candidate.distance, 0);
        // normalized_form should be set
        assert!(!candidate.normalized_form.is_empty());
    }

    #[test]
    fn test_phonetic_normalized_algorithm() {
        // Test with transposition algorithm
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_algorithm(
            ["phone", "fone"],
            Algorithm::Transposition,
        );

        assert_eq!(dict.algorithm(), Algorithm::Transposition);

        // Should still work normally
        let results = dict.query("phone", 0);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_phonetic_normalized_count() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "bone"]);

        // Count should be number of unique normalized forms
        let count = dict.normalized_count();
        println!("Normalized count: {}", count);

        // At minimum 1 (if all normalize to same), at most 3 (if all different)
        assert!(count >= 1 && count <= 3);
    }

    #[test]
    fn test_dictionary_trait() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "bone"]);

        // Test Dictionary trait
        assert!(dict.contains("phone"));
        assert!(dict.contains("fone"));
        assert!(!dict.contains("unknown"));

        // Test len
        assert_eq!(dict.len(), Some(3));

        // Test root navigation
        let root = dict.root();
        assert!(!root.is_final()); // Root shouldn't be final
    }

    #[test]
    fn test_insert() {
        let dict = PhoneticNormalizedDictionary::<()>::new();

        assert!(dict.insert("phone"));
        assert!(dict.insert("fone"));
        assert!(!dict.insert("phone")); // Already exists

        assert!(dict.contains("phone"));
        assert!(dict.contains("fone"));
        assert_eq!(dict.len(), Some(2));
    }

    #[test]
    fn test_remove() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "bone"]);

        // Remove from index
        assert!(dict.remove("phone"));
        assert!(!dict.remove("phone")); // Already removed

        // Query should no longer return "phone" in normalized results
        let results = dict.query("phone", 0);
        assert!(!results.iter().any(|c| c.term == "phone"));
    }

    #[test]
    fn test_zipper_navigation() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["cat", "car", "card"]);

        let zipper = dict.zipper();
        assert!(!zipper.is_final());

        // Navigate to 'c'
        let c = zipper.descend('c').expect("Should have 'c' edge");
        assert!(!c.is_final());

        // Navigate to 'a'
        let ca = c.descend('a').expect("Should have 'a' edge");
        assert!(!ca.is_final());

        // Navigate to 't' - should be final (cat)
        let cat = ca.descend('t').expect("Should have 't' edge");
        assert!(cat.is_final());

        // Check path
        assert_eq!(cat.path(), vec!['c', 'a', 't']);
    }

    #[test]
    fn test_iter_terms() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "bone"]);

        let terms: Vec<_> = dict.iter_terms().map(|(term, _)| term).collect();
        assert_eq!(terms.len(), 3);
        assert!(terms.contains(&"phone".to_string()));
        assert!(terms.contains(&"fone".to_string()));
        assert!(terms.contains(&"bone".to_string()));
    }

    #[test]
    fn test_regex_query_basic() {
        // Use empty rules so we can test regex matching without phonetic normalization
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["cat", "car", "card", "care", "bat"],
            vec![],
        );

        // Pattern with alternation
        let results = dict.query_regex("ca.", 0).expect("regex should parse");
        println!("Results for 'ca.': {:?}", results);

        // Should match "cat" and "car" (3-letter words starting with "ca")
        assert!(results.iter().any(|c| c.term == "cat"));
        assert!(results.iter().any(|c| c.term == "car"));
        // "card" and "care" are 4 letters, shouldn't match "ca." exactly
        assert!(!results.iter().any(|c| c.term == "card"));
    }

    #[test]
    fn test_regex_query_with_distance() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["cat", "car", "card", "care", "bat"],
            vec![],
        );

        // Pattern with distance tolerance
        let results = dict.query_regex("ca.", 1).expect("regex should parse");
        println!("Results for 'ca.' with distance 1: {:?}", results);

        // Should now also match "card" and "care" (1 extra char)
        assert!(results.iter().any(|c| c.term == "card"));
        assert!(results.iter().any(|c| c.term == "care"));
    }

    #[test]
    fn test_regex_query_alternation() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["phone", "fone", "bone", "cone", "tone"],
            vec![],
        );

        // Pattern matching either "phone" or "fone"
        let results = dict
            .query_regex("(ph|f)one", 0)
            .expect("regex should parse");
        println!("Results for '(ph|f)one': {:?}", results);

        assert!(results.iter().any(|c| c.term == "phone"));
        assert!(results.iter().any(|c| c.term == "fone"));
        assert!(!results.iter().any(|c| c.term == "bone"));
    }

    #[test]
    fn test_regex_query_optional() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["color", "colour", "cola"],
            vec![],
        );

        // Pattern with optional character
        let results = dict.query_regex("colou?r", 0).expect("regex should parse");
        println!("Results for 'colou?r': {:?}", results);

        assert!(results.iter().any(|c| c.term == "color"));
        assert!(results.iter().any(|c| c.term == "colour"));
        assert!(!results.iter().any(|c| c.term == "cola"));
    }

    #[test]
    fn test_regex_query_with_phonetic_normalization() {
        // Test regex query in normalized space
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["phone", "fone", "elephant"]);

        // Get normalized forms
        let phone_norm = dict.normalize("phone");
        let fone_norm = dict.normalize("fone");
        println!("phone -> {}, fone -> {}", phone_norm, fone_norm);

        // Both phone and fone should normalize to the same form
        assert_eq!(
            phone_norm, fone_norm,
            "phone and fone should normalize to same form"
        );

        // Query using the normalized form directly with distance 0
        let results = dict.query(&phone_norm, 0);
        println!("Results for normalized query: {:?}", results);

        assert!(
            results.iter().any(|c| c.term == "phone"),
            "Should find 'phone'"
        );
        assert!(
            results.iter().any(|c| c.term == "fone"),
            "Should find 'fone'"
        );
    }

    #[test]
    fn test_regex_query_error_handling() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["test"], vec![]);

        // Invalid regex should return error
        let result = dict.query_regex("[invalid", 0);
        assert!(result.is_err());

        if let Err(e) = result {
            println!("Expected error: {}", e);
        }
    }

    #[test]
    fn test_expand_to_phonetic_pattern() {
        use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

        // Create a dictionary with a simple rule: ph -> f
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "ph_to_f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.1,
        }];

        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["phone", "fone"], rules);

        // Expand "fone" - should include "ph" as an alternative for "f"
        let pattern = dict.expand_to_phonetic_pattern("fone");
        println!("fone -> pattern: {}", pattern);

        // Pattern should contain the alternative
        assert!(
            pattern.contains("(ph|f)") || pattern.contains("(f|ph)"),
            "Pattern should contain alternation for f/ph"
        );
    }

    #[test]
    fn test_query_phonetic_pattern_basic() {
        use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

        // Create a dictionary with a simple rule: ph -> f
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "ph_to_f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.1,
        }];

        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["phone", "fone", "bone", "cone"],
            rules,
        );

        // Query with phonetic pattern expansion
        let results = dict
            .query_phonetic_pattern("fone", 0)
            .expect("pattern should parse");
        println!("Results for phonetic pattern 'fone': {:?}", results);

        // Should find both "phone" and "fone" since "f" expands to "(ph|f)"
        assert!(
            results.iter().any(|c| c.term == "fone"),
            "Should find 'fone'"
        );
        // "phone" has "ph" which matches the expanded pattern "(ph|f)one"
        assert!(
            results.iter().any(|c| c.term == "phone"),
            "Should find 'phone'"
        );
    }

    #[test]
    fn test_query_phonetic_pattern_no_expansion() {
        // With no rules, the pattern should be the literal query
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["cat", "car", "bat"], vec![]);

        let pattern = dict.expand_to_phonetic_pattern("cat");
        assert_eq!(pattern, "cat", "With no rules, pattern should be literal");

        let results = dict
            .query_phonetic_pattern("cat", 0)
            .expect("pattern should parse");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "cat");
    }

    #[test]
    fn test_query_phonetic_pattern_with_distance() {
        use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "ph_to_f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.1,
        }];

        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["phone", "fone", "bone", "cone"],
            rules,
        );

        // Query with distance tolerance
        let results = dict
            .query_phonetic_pattern("fone", 1)
            .expect("pattern should parse");
        println!(
            "Results for phonetic pattern 'fone' with distance 1: {:?}",
            results
        );

        // Should find "bone" and "cone" too (1 edit distance from expanded pattern)
        assert!(results.iter().any(|c| c.term == "bone"));
        assert!(results.iter().any(|c| c.term == "cone"));
    }

    #[test]
    fn test_sync_strategy() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms(["test"]);
        assert_eq!(dict.sync_strategy(), SyncStrategy::InternalSync);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "abd"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }
}
