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
//! The dictionary uses a `FuzzyMultiMap` internally which provides:
//! - `originals`: Backend dictionary storing original terms with values
//! - `normalized_multimap`: FuzzyMultiMap mapping normalized forms to original terms
//!
//! The `FuzzyMultiMap` provides O(k log n) fuzzy queries using Levenshtein automaton
//! pruning on a trie, which is much more efficient than the previous BK-tree approach.
//!
//! # Example
//!
//! ```rust
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
//! - Fastest query time (single normalization + Levenshtein automaton search)
//! - Implements all dictionary protocols for drop-in compatibility
//! - Low memory overhead (normalized trie similar to original)
//! - Supports dynamic insert and delete operations
//! - Uses Levenshtein automaton pruning for efficient fuzzy queries
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

use crate::cache::multimap::FuzzyMultiMap;
use crate::dictionary::node_adapter::{
    impl_transparent_dictionary_node, impl_transparent_mapped_dictionary_node,
    map_transparent_traversal_root,
};
use crate::distance::standard_distance;
use crate::phonetic::expansion::expand_phonetic_alternatives_char;
use crate::phonetic::nfa::{compile as compile_nfa, ProductAutomatonChar};
use crate::phonetic::regex::{parse as parse_regex, ParseError as RegexParseError};
use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};
use crate::phonetic::{apply_rules_seq_char, zompist_rules_char};
use crate::transducer::language::{LanguageProduct, LanguageQueryIterator};
use crate::transducer::Algorithm;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::char_zipper::DynamicDawgCharZipper;
use libdictenstein::{
    DictZipper, Dictionary, DictionaryNode, DictionaryTraversalRoot, DictionaryValue,
    MappedDictionary, MappedDictionaryNode, MutableMappedDictionary, SyncStrategy,
    ValuedDictZipper,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

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

struct CandidateBestMatch {
    distance: usize,
    normalized_form: String,
}

impl CandidateBestMatch {
    fn new(distance: usize, normalized_form: &str) -> Self {
        Self {
            distance,
            normalized_form: normalized_form.to_owned(),
        }
    }

    fn replace_if_better(&mut self, distance: usize, normalized_form: &str) {
        if distance < self.distance {
            self.distance = distance;
            self.normalized_form.clear();
            self.normalized_form.push_str(normalized_form);
        }
    }
}

fn upsert_candidate_best_match<T>(
    by_term: &mut HashMap<String, CandidateBestMatch>,
    term: T,
    distance: usize,
    normalized_form: &str,
) where
    T: AsRef<str> + Into<String>,
{
    let term_ref = term.as_ref();
    if term_ref.is_empty() {
        return;
    }

    if let Some(existing) = by_term.get_mut(term_ref) {
        existing.replace_if_better(distance, normalized_form);
        return;
    }

    by_term.insert(
        term.into(),
        CandidateBestMatch::new(distance, normalized_form),
    );
}

fn best_matches_into_candidates(
    by_term: HashMap<String, CandidateBestMatch>,
) -> Vec<PhoneticNormalizedCandidate> {
    let mut candidates = Vec::with_capacity(by_term.len());
    for (term, best_match) in by_term {
        candidates.push(PhoneticNormalizedCandidate {
            term,
            distance: best_match.distance,
            normalized_form: best_match.normalized_form,
        });
    }
    candidates
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

/// Error type for compact term-id dictionary construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermIdDictionaryBuildError {
    /// The compact dictionary stores term identifiers as `u32`; the next
    /// distinct term would require a wider identifier.
    TermIdOverflow {
        /// The first identifier that cannot be represented by the compact `u32` payload.
        next_id: usize,
    },
}

impl fmt::Display for TermIdDictionaryBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermIdDictionaryBuildError::TermIdOverflow { next_id } => write!(
                f,
                "term-id dictionary cannot assign term id {next_id}; compact ids are stored as u32"
            ),
        }
    }
}

impl std::error::Error for TermIdDictionaryBuildError {}

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
/// ```rust
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

    /// FuzzyMultiMap: normalized_form → {original_terms}
    /// Uses Levenshtein automaton pruning for efficient fuzzy queries
    /// O(k log n) complexity vs O(n) linear scan
    normalized_multimap: FuzzyMultiMap<HashSet<String>, DynamicDawgChar<HashSet<String>>>,

    /// Complete phonetic rules for introspection and pattern expansion.
    rules: Vec<RewriteRuleChar>,

    /// Rules that still need sequential rewrite application.
    sequential_rules: Vec<RewriteRuleChar>,

    /// Exact whole-word lexical normalizations keyed by original spelling.
    whole_word_normalizations: HashMap<String, Vec<String>>,

    /// Fuel for rule application (prevents infinite loops)
    fuel: usize,

    /// Marker for value type
    _value: std::marker::PhantomData<V>,
}

/// UTF-8 character-level PhoneticNormalizedDictionary.
///
/// This is the default and recommended variant, using `char` for proper Unicode
/// semantics with accented characters, CJK, emoji, etc.
pub type PhoneticNormalizedDictionaryChar<V = ()> =
    PhoneticNormalizedDictionary<V, DynamicDawgChar<V>>;

fn sort_candidates_by_relevance(
    query: &str,
    mut results: Vec<PhoneticNormalizedCandidate>,
) -> Vec<PhoneticNormalizedCandidate> {
    if results.len() < 2 {
        return results;
    }

    results.sort_unstable_by_key(|candidate| candidate.distance);

    let mut sorted = Vec::with_capacity(results.len());
    let mut candidates = results.into_iter().peekable();

    while let Some(candidate) = candidates.next() {
        let normalized_distance = candidate.distance;

        if !matches!(
            candidates.peek(),
            Some(next) if next.distance == normalized_distance
        ) {
            sorted.push(candidate);
            continue;
        }

        let mut group = Vec::with_capacity(2);
        group.push(candidate);

        while let Some(next) = candidates.peek() {
            if next.distance != normalized_distance {
                break;
            }
            if let Some(next) = candidates.next() {
                group.push(next);
            }
        }

        let mut ranked = Vec::with_capacity(group.len());
        for candidate in group.drain(..) {
            let original_distance = standard_distance(query, &candidate.term);
            ranked.push((original_distance, candidate));
        }

        ranked.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.term.cmp(&right.1.term))
        });
        sorted.extend(ranked.into_iter().map(|(_, candidate)| candidate));
    }

    sorted
}

fn is_exact_whole_word_context(context: &ContextChar) -> bool {
    match context {
        ContextChar::And(left, right) => {
            matches!(
                (left.as_ref(), right.as_ref()),
                (ContextChar::Initial, ContextChar::Final)
                    | (ContextChar::Final, ContextChar::Initial)
            )
        }
        _ => false,
    }
}

#[cfg(test)]
fn phone_chars_from_input(input: &str) -> Vec<PhoneChar> {
    input
        .chars()
        .map(|c| {
            if is_vowel(c) {
                PhoneChar::Vowel(c)
            } else {
                PhoneChar::Consonant(c)
            }
        })
        .collect()
}

fn phone_chars_to_string(phones: &[PhoneChar]) -> String {
    let mut output = String::with_capacity(phones.len());
    for phone in phones {
        match phone {
            PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => output.push(*c),
            PhoneChar::Digraph(c1, c2) => {
                output.push(*c1);
                output.push(*c2);
            }
            PhoneChar::Trigraph(c1, c2, c3) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
            }
            PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
            }
            PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
            }
            PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
                output.push(*c6);
            }
            PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
                output.push(*c6);
                output.push(*c7);
            }
            PhoneChar::Sequence(chars) => {
                for c in chars {
                    output.push(*c);
                }
            }
            PhoneChar::Silent => {}
        }
    }
    output
}

fn split_normalization_rules(
    rules: &[RewriteRuleChar],
) -> (Vec<RewriteRuleChar>, HashMap<String, Vec<String>>) {
    let mut sequential_rules = Vec::with_capacity(rules.len());
    let mut map: HashMap<String, Vec<String>> = HashMap::with_capacity(rules.len());
    for rule in rules {
        if is_exact_whole_word_context(&rule.context) {
            let pattern = phone_chars_to_string(&rule.pattern);
            let replacement = phone_chars_to_string(&rule.replacement);
            let replacements = map.entry(pattern).or_default();
            if !replacements.contains(&replacement) {
                replacements.push(replacement);
            }
        } else {
            sequential_rules.push(rule.clone());
        }
    }
    (sequential_rules, map)
}

fn normalized_forms_char_with_map(
    input: &str,
    sequential_rules: &[RewriteRuleChar],
    fuel: usize,
    whole_word_normalizations: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    if let Some(replacements) = whole_word_normalizations.get(input) {
        return replacements.clone();
    }

    vec![normalize_string_char(input, sequential_rules, fuel)]
}

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

impl_transparent_dictionary_node!(PhoneticNormalizedNode, |_owner, child| {
    PhoneticNormalizedNode { inner: child }
});
impl_transparent_mapped_dictionary_node!(PhoneticNormalizedNode);

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

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        map_transparent_traversal_root(self.originals.traversal_root(), |inner| {
            PhoneticNormalizedNode { inner }
        })
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

    #[inline]
    fn is_suffix_based(&self) -> bool {
        self.originals.is_suffix_based()
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
            // 2. Update normalized multimap
            let term_string = term.to_string();
            for normalized in self.normalized_forms(term) {
                self.normalized_multimap.update_or_insert(
                    &normalized,
                    HashSet::from([term_string.clone()]),
                    |set| {
                        set.insert(term_string.clone());
                    },
                );
            }
        }

        is_new
    }

    fn update_or_insert<F>(&self, term: &str, default_value: Self::Value, update_fn: F) -> bool
    where
        F: Fn(&mut Self::Value),
    {
        // Check if this is a new term
        let existed = self.originals.get_value(term).is_some();

        // Update or insert in backend
        let is_new = self
            .originals
            .update_or_insert(term, default_value, update_fn);

        // If newly inserted, also update normalized multimap
        if is_new && !existed {
            let term_string = term.to_string();
            for normalized in self.normalized_forms(term) {
                self.normalized_multimap.update_or_insert(
                    &normalized,
                    HashSet::from([term_string.clone()]),
                    |set| {
                        set.insert(term_string.clone());
                    },
                );
            }
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

        // Update normalized multimap with new terms
        for (term, _) in other.iter_terms() {
            for normalized in self.normalized_forms(&term) {
                self.normalized_multimap.update_or_insert(
                    &normalized,
                    HashSet::from([term.clone()]),
                    |set| {
                        set.insert(term.clone());
                    },
                );
            }
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
        Self::with_rules_and_algorithm(rules, Algorithm::Standard)
    }

    /// Create an empty dictionary with custom rules and algorithm.
    pub fn with_rules_and_algorithm(rules: Vec<RewriteRuleChar>, algorithm: Algorithm) -> Self {
        let fuel = Self::compute_fuel(&rules);
        let (sequential_rules, whole_word_normalizations) = split_normalization_rules(&rules);
        let normalized_dict = DynamicDawgChar::<HashSet<String>>::new();
        Self {
            originals: DynamicDawgChar::new(),
            normalized_multimap: FuzzyMultiMap::new(normalized_dict, algorithm),
            rules,
            sequential_rules,
            whole_word_normalizations,
            fuel,
            _value: std::marker::PhantomData,
        }
    }

    /// Create from terms with default Zompist rules.
    ///
    /// # Example
    ///
    /// ```rust
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
        let (sequential_rules, whole_word_normalizations) = split_normalization_rules(&rules);
        let originals = DynamicDawgChar::new();
        let normalized_dict = DynamicDawgChar::<HashSet<String>>::new();

        for term in terms {
            let term = term.as_ref();
            originals.insert_with_value(term, V::default());

            let term_string = term.to_string();

            // Use update_or_insert to add term to the HashSet for this normalized form
            for normalized in normalized_forms_char_with_map(
                term,
                &sequential_rules,
                fuel,
                &whole_word_normalizations,
            ) {
                normalized_dict.update_or_insert(
                    &normalized,
                    HashSet::from([term_string.clone()]),
                    |set| {
                        set.insert(term_string.clone());
                    },
                );
            }
        }

        Self {
            originals,
            normalized_multimap: FuzzyMultiMap::new(normalized_dict, algorithm),
            rules,
            sequential_rules,
            whole_word_normalizations,
            fuel,
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
        let (sequential_rules, whole_word_normalizations) = split_normalization_rules(&rules);
        let originals = DynamicDawgChar::new();
        let normalized_dict = DynamicDawgChar::<HashSet<String>>::new();

        for (term, value) in terms {
            let term = term.as_ref();
            originals.insert_with_value(term, value);

            let term_string = term.to_string();

            for normalized in normalized_forms_char_with_map(
                term,
                &sequential_rules,
                fuel,
                &whole_word_normalizations,
            ) {
                normalized_dict.update_or_insert(
                    &normalized,
                    HashSet::from([term_string.clone()]),
                    |set| {
                        set.insert(term_string.clone());
                    },
                );
            }
        }

        Self {
            originals,
            normalized_multimap: FuzzyMultiMap::new(normalized_dict, Algorithm::Standard),
            rules,
            sequential_rules,
            whole_word_normalizations,
            fuel,
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

/// Opt-in compact phonetic normalized dictionary using term-id payloads.
///
/// This variant preserves the normalized-query semantics of
/// [`PhoneticNormalizedDictionary`] while replacing normalized-form payloads from
/// `HashSet<String>` to sorted `Vec<u32>` term identifiers. Original spellings
/// are stored once in `terms`, and normalized trie leaves carry compact ids.
///
/// The default dictionary remains unchanged for API compatibility; this type is
/// intended for memory-sensitive workloads and scientific comparison.
pub struct PhoneticNormalizedTermIdDictionary {
    terms: Vec<String>,
    normalized_multimap: FuzzyMultiMap<Vec<u32>, DynamicDawgChar<Vec<u32>>>,
    rules: Vec<RewriteRuleChar>,
    sequential_rules: Vec<RewriteRuleChar>,
    whole_word_normalizations: HashMap<String, Vec<String>>,
    fuel: usize,
}

fn compact_term_id_from_len(term_count: usize) -> Result<u32, TermIdDictionaryBuildError> {
    u32::try_from(term_count).map_err(|_| TermIdDictionaryBuildError::TermIdOverflow {
        next_id: term_count,
    })
}

fn compact_term_id_capacity_hint(size_hint: (usize, Option<usize>)) -> usize {
    const INITIAL_CAPACITY_LIMIT: usize = 1 << 20;

    let (lower_bound, _) = size_hint;
    lower_bound.min(INITIAL_CAPACITY_LIMIT)
}

impl PhoneticNormalizedTermIdDictionary {
    /// Create from terms with default Zompist rules.
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::try_from_terms(terms)
            .expect("term-id dictionary exceeded the compact u32 identifier capacity")
    }

    /// Try to create from terms with default Zompist rules.
    pub fn try_from_terms<I, S>(terms: I) -> Result<Self, TermIdDictionaryBuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::try_from_terms_with_rules(terms, zompist_rules_char())
    }

    /// Create from terms with custom rules.
    pub fn from_terms_with_rules<I, S>(terms: I, rules: Vec<RewriteRuleChar>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::try_from_terms_with_rules(terms, rules)
            .expect("term-id dictionary exceeded the compact u32 identifier capacity")
    }

    /// Try to create from terms with custom rules.
    pub fn try_from_terms_with_rules<I, S>(
        terms: I,
        rules: Vec<RewriteRuleChar>,
    ) -> Result<Self, TermIdDictionaryBuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::try_from_terms_with_rules_and_algorithm(terms, rules, Algorithm::Standard)
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
        Self::try_from_terms_with_rules_and_algorithm(terms, rules, algorithm)
            .expect("term-id dictionary exceeded the compact u32 identifier capacity")
    }

    /// Try to create from terms with custom rules and algorithm.
    pub fn try_from_terms_with_rules_and_algorithm<I, S>(
        terms: I,
        rules: Vec<RewriteRuleChar>,
        algorithm: Algorithm,
    ) -> Result<Self, TermIdDictionaryBuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let fuel = PhoneticNormalizedDictionary::<()>::compute_fuel(&rules);
        let (sequential_rules, whole_word_normalizations) = split_normalization_rules(&rules);
        let normalized_dict = DynamicDawgChar::<Vec<u32>>::new();
        let terms = terms.into_iter();
        let capacity_hint = compact_term_id_capacity_hint(terms.size_hint());
        let mut term_table = Vec::with_capacity(capacity_hint);
        let mut term_ids = HashMap::with_capacity(capacity_hint);

        for term in terms {
            let term = term.as_ref();
            let term_id = match term_ids.get(term).copied() {
                Some(term_id) => term_id,
                None => {
                    let term_id = compact_term_id_from_len(term_table.len())?;
                    term_table.push(term.to_string());
                    term_ids.insert(term.to_string(), term_id);
                    term_id
                }
            };

            for normalized in normalized_forms_char_with_map(
                term,
                &sequential_rules,
                fuel,
                &whole_word_normalizations,
            ) {
                normalized_dict.update_or_insert(&normalized, vec![term_id], |ids| {
                    match ids.binary_search(&term_id) {
                        Ok(_) => {}
                        Err(pos) => ids.insert(pos, term_id),
                    }
                });
            }
        }

        Ok(Self {
            terms: term_table,
            normalized_multimap: FuzzyMultiMap::new(normalized_dict, algorithm),
            rules,
            sequential_rules,
            whole_word_normalizations,
            fuel,
        })
    }

    /// Normalize a string using this dictionary's rules.
    pub fn normalize(&self, term: &str) -> String {
        self.normalized_forms(term)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Return all normalized forms for a term.
    pub fn normalized_forms(&self, term: &str) -> Vec<String> {
        normalized_forms_char_with_map(
            term,
            &self.sequential_rules,
            self.fuel,
            &self.whole_word_normalizations,
        )
    }

    /// Number of original terms stored in the compact term table.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Number of unique normalized forms.
    pub fn normalized_count(&self) -> usize {
        self.normalized_multimap.dictionary().len().unwrap_or(0)
    }

    /// Get the phonetic rules being used.
    pub fn rules(&self) -> &[RewriteRuleChar] {
        &self.rules
    }

    /// Get the algorithm used for phonetic queries.
    pub fn algorithm(&self) -> Algorithm {
        self.normalized_multimap.algorithm()
    }

    /// Query for candidates within edit distance in normalized space.
    pub fn query(&self, query: &str, max_distance: usize) -> Vec<PhoneticNormalizedCandidate> {
        let normalized_queries = self.normalized_forms(query);
        let mut by_term_id: HashMap<u32, PhoneticNormalizedCandidate> =
            HashMap::with_capacity(self.term_count());

        if max_distance == 0 {
            for normalized_query in &normalized_queries {
                if let Some(term_ids) = self
                    .normalized_multimap
                    .dictionary()
                    .get_value(normalized_query)
                {
                    for term_id in term_ids {
                        self.upsert_candidate(
                            &mut by_term_id,
                            term_id,
                            0,
                            normalized_query.clone(),
                        );
                    }
                }
            }
        } else {
            for normalized_query in &normalized_queries {
                for (normalized_form, distance, term_ids) in self
                    .normalized_multimap
                    .query_with_distance(normalized_query, max_distance)
                {
                    for term_id in term_ids {
                        self.upsert_candidate(
                            &mut by_term_id,
                            term_id,
                            distance,
                            normalized_form.clone(),
                        );
                    }
                }
            }
        }

        sort_candidates_by_relevance(query, by_term_id.into_values().collect())
    }

    fn upsert_candidate(
        &self,
        by_term_id: &mut HashMap<u32, PhoneticNormalizedCandidate>,
        term_id: u32,
        distance: usize,
        normalized_form: String,
    ) {
        let Some(term_index) = usize::try_from(term_id).ok() else {
            return;
        };
        let Some(term) = self.terms.get(term_index) else {
            return;
        };

        match by_term_id.get_mut(&term_id) {
            Some(existing) if distance < existing.distance => {
                existing.distance = distance;
                existing.normalized_form = normalized_form;
            }
            Some(_) => {}
            None => {
                by_term_id.insert(
                    term_id,
                    PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance,
                        normalized_form,
                    },
                );
            }
        }
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
        self.normalized_forms(term)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    fn normalized_forms(&self, term: &str) -> Vec<String> {
        normalized_forms_char_with_map(
            term,
            &self.sequential_rules,
            self.fuel,
            &self.whole_word_normalizations,
        )
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
        self.normalized_multimap.algorithm()
    }

    /// Get the number of unique normalized forms in the dictionary.
    pub fn normalized_count(&self) -> usize {
        self.normalized_multimap.dictionary().len().unwrap_or(0)
    }

    /// Get a reference to the backend dictionary.
    pub fn originals(&self) -> &D {
        &self.originals
    }

    /// Get a reference to the normalized multimap.
    pub fn normalized_multimap(
        &self,
    ) -> &FuzzyMultiMap<HashSet<String>, DynamicDawgChar<HashSet<String>>> {
        &self.normalized_multimap
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
    /// Note: This updates the normalized multimap. The backend may or may not
    /// support removal. Use a backend like DynamicDawgChar that supports removal
    /// for full dynamic functionality.
    ///
    /// # Returns
    ///
    /// `true` if the term was in the normalized index, `false` otherwise.
    pub fn remove(&self, term: &str) -> bool {
        let mut removed = false;
        for normalized in self.normalized_forms(term) {
            if let Some(originals) = self.normalized_multimap.dictionary().get_value(&normalized) {
                if originals.contains(term) {
                    let mut new_originals = originals.clone();
                    new_originals.remove(term);

                    if new_originals.is_empty() {
                        self.normalized_multimap.insert(&normalized, HashSet::new());
                    } else {
                        self.normalized_multimap.insert(&normalized, new_originals);
                    }
                    removed = true;
                }
            }
        }
        removed
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
    /// Note: This iterates over the normalized multimap, which tracks all inserted terms.
    pub fn iter_terms(&self) -> impl Iterator<Item = (String, String)> + '_ {
        let capacity = self.len().unwrap_or(0);
        let mut seen = HashSet::with_capacity(capacity);
        let mut pairs = Vec::with_capacity(capacity);

        for (normalized, originals) in self.normalized_multimap.dictionary().iter() {
            if originals.is_empty() {
                continue;
            }

            for term in originals.iter().filter(|term| !term.is_empty()) {
                if seen.contains(term.as_str()) {
                    continue;
                }

                seen.insert(term.clone());
                pairs.push((term.clone(), normalized.clone()));
            }
        }

        pairs.into_iter()
    }

    /// Iterate over normalized forms and their original terms.
    pub fn iter_normalized(&self) -> impl Iterator<Item = (String, HashSet<String>)> {
        let capacity = self.normalized_multimap.dictionary().len().unwrap_or(0);
        let mut pairs = Vec::with_capacity(capacity);

        for (normalized, originals) in self.normalized_multimap.dictionary().iter() {
            if !originals.is_empty() {
                pairs.push((normalized.clone(), originals.clone()));
            }
        }

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

    /// Query using a regex pattern against ORIGINAL (non-normalized) terms.
    ///
    /// Unlike `query_regex` which searches normalized forms, this method
    /// searches the original terms directly. This is useful with patterns
    /// from `expand_to_phonetic_pattern` which are designed to match
    /// original spellings like "phone", "fone", etc.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Regex pattern to match (e.g., "(ph|f)on", "kn.*t")
    /// * `max_distance` - Maximum edit distance from pattern (0 for exact regex match)
    ///
    /// # Returns
    ///
    /// Results contain original terms that match the pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
    /// use liblevenshtein::phonetic::{ContextChar, PhoneChar, RewriteRuleChar};
    ///
    /// let ph_to_f = RewriteRuleChar {
    ///     rule_id: 1,
    ///     rule_name: "ph to f".into(),
    ///     pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
    ///     replacement: vec![PhoneChar::Consonant('f')],
    ///     context: ContextChar::Anywhere,
    ///     weight: 0.1,
    ///     syllable_condition: None,
    /// };
    /// let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
    ///     ["phone", "fone", "bone"],
    ///     vec![ph_to_f],
    /// );
    ///
    /// // Expand "fone" to phonetic pattern
    /// let pattern = dict.expand_to_phonetic_pattern("fone");
    /// // Search original terms with the pattern
    /// let results = dict
    ///     .query_original_regex(&pattern, 0)
    ///     .expect("doc: generated pattern must compile");
    /// let terms: Vec<_> = results.iter().map(|candidate| candidate.term.as_str()).collect();
    /// assert!(terms.contains(&"phone"));
    /// assert!(terms.contains(&"fone"));
    /// assert!(!terms.contains(&"bone"));
    /// ```
    pub fn query_original_regex(
        &self,
        pattern: &str,
        max_distance: u8,
    ) -> Result<Vec<PhoneticNormalizedCandidate>, RegexQueryError> {
        // Parse regex pattern
        let ast = parse_regex(pattern)?;

        // Compile to NFA
        let nfa = compile_nfa(&ast).map_err(RegexQueryError::CompileError)?;

        // Create product automaton for fuzzy matching
        let product = ProductAutomatonChar::with_algorithm(nfa, max_distance, self.algorithm());

        // Search ORIGINAL terms (not normalized) for matches
        let mut results = Vec::with_capacity(self.originals.len().unwrap_or(0));

        for (original, _) in self.originals.iter() {
            if let Some(distance) = product.min_distance(&original) {
                let normalized_form = self.normalize(&original);
                results.push(PhoneticNormalizedCandidate {
                    term: original,
                    distance: distance as usize,
                    normalized_form,
                });
            }
        }

        // Sort by distance for convenience
        results.sort_by_key(|c| c.distance);
        Ok(results)
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
    /// # Performance Optimization
    ///
    /// For exact matches (d=0), this uses a direct trie lookup which is
    /// **100-300× faster** than the automaton-based search. This optimization
    /// is based on benchmark results showing HashMap/trie lookup at ~2µs vs
    /// automaton traversal at ~600µs for 100 queries.
    ///
    /// For fuzzy matches (d≥1), uses FuzzyMultiMap's Levenshtein automaton
    /// for efficient O(k log n) fuzzy queries with trie pruning.
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
        let normalized_queries = self.normalized_forms(query);

        // Fast path for exact match (d=0): Direct trie lookup is 100-300× faster
        // than automaton traversal (benchmark: 2µs vs 600µs for 100 queries)
        if max_distance == 0 {
            let mut by_term = HashMap::with_capacity(self.len().unwrap_or(0));
            for normalized_query in &normalized_queries {
                if let Some(originals) = self
                    .normalized_multimap
                    .dictionary()
                    .get_value(normalized_query)
                {
                    for term in originals {
                        upsert_candidate_best_match(
                            &mut by_term,
                            term.as_str(),
                            0,
                            normalized_query,
                        );
                    }
                }
            }
            return sort_candidates_by_relevance(query, best_matches_into_candidates(by_term));
        }

        // Fuzzy path (d≥1): Use Levenshtein automaton for efficient trie pruning
        let mut by_term: HashMap<String, CandidateBestMatch> =
            HashMap::with_capacity(self.len().unwrap_or(0));
        for normalized_query in &normalized_queries {
            let fuzzy_results = self
                .normalized_multimap
                .query_with_distance(normalized_query, max_distance);

            for (normalized_form, distance, originals) in fuzzy_results {
                for term in originals {
                    upsert_candidate_best_match(
                        &mut by_term,
                        term,
                        distance,
                        normalized_form.as_str(),
                    );
                }
            }
        }

        sort_candidates_by_relevance(query, best_matches_into_candidates(by_term))
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
        let product = ProductAutomatonChar::with_algorithm(nfa, max_distance, self.algorithm());

        Ok(self.query_with_product(&product))
    }

    /// Query using a pre-compiled NFA pattern.
    ///
    /// Use this when you want to reuse the same pattern for multiple queries,
    /// avoiding repeated parsing and compilation overhead.
    pub fn query_with_product(
        &self,
        product: &ProductAutomatonChar,
    ) -> Vec<PhoneticNormalizedCandidate> {
        if product.supports_incremental_trie_intersection() {
            return self.query_with_product_trie(product);
        }

        self.query_with_product_scan(product)
    }

    fn query_with_product_scan(
        &self,
        product: &ProductAutomatonChar,
    ) -> Vec<PhoneticNormalizedCandidate> {
        let mut results = Vec::with_capacity(self.len().unwrap_or(0));

        for (normalized, originals) in self.normalized_multimap.dictionary().iter() {
            if originals.is_empty() {
                continue; // Skip empty sets from removals
            }
            if let Some(distance) = product.min_distance(&normalized) {
                for term in originals.iter() {
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

    fn query_with_product_trie(
        &self,
        product: &ProductAutomatonChar,
    ) -> Vec<PhoneticNormalizedCandidate> {
        let mut results = Vec::with_capacity(self.len().unwrap_or(0));
        let language_product = LanguageProduct::new(product.nfa().clone(), product.max_distance());
        let query = LanguageQueryIterator::from_dictionary(
            self.normalized_multimap.dictionary(),
            language_product,
        );

        for matched in query {
            let normalized_form: String = matched.units.into_iter().collect();
            if let Some(originals) = matched.node.value() {
                for term in originals.iter().filter(|term| !term.is_empty()) {
                    results.push(PhoneticNormalizedCandidate {
                        term: term.clone(),
                        distance: usize::from(matched.distance),
                        normalized_form: normalized_form.clone(),
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
    /// The query is first normalized, then expanded to a regex pattern that
    /// matches all possible original spellings. For example:
    /// - "fone" → normalize → "fon" → expand → "(ph|f)on"
    ///
    /// The resulting pattern is designed to match **original spellings** in
    /// the dictionary. Use with `query_original_regex` for dictionary search,
    /// or use directly for searching external text.
    pub fn expand_to_phonetic_pattern(&self, query: &str) -> String {
        let normalized = self.normalize(query);
        expand_phonetic_alternatives_char(&normalized, &self.rules)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// O(1) vowel classification using bitmask lookup.
///
/// Uses a 64-bit mask for lowercase letters (a-z in bits 0-25) and checks
/// if the character is a vowel (a=0, e=4, i=8, o=14, u=20).
/// This avoids array indexing and bounds checking overhead.
///
/// Vowel bits: a(0), e(4), i(8), o(14), u(20) = 0b100001000100010001 = 0x111111
/// Bitmask: (1 << 0) | (1 << 4) | (1 << 8) | (1 << 14) | (1 << 20) = 0x104111
const VOWEL_MASK: u64 =
    1 | (1 << (b'e' - b'a')) | (1 << (b'i' - b'a')) | (1 << (b'o' - b'a')) | (1 << (b'u' - b'a'));

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
                for p in phones.iter() {
                    match p {
                        PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => {
                            self.output_string.push(*c)
                        }
                        PhoneChar::Digraph(c1, c2) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                        }
                        PhoneChar::Trigraph(c1, c2, c3) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                            self.output_string.push(*c3);
                        }
                        PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                            self.output_string.push(*c3);
                            self.output_string.push(*c4);
                        }
                        PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                            self.output_string.push(*c3);
                            self.output_string.push(*c4);
                            self.output_string.push(*c5);
                        }
                        PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                            self.output_string.push(*c3);
                            self.output_string.push(*c4);
                            self.output_string.push(*c5);
                            self.output_string.push(*c6);
                        }
                        PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                            self.output_string.push(*c1);
                            self.output_string.push(*c2);
                            self.output_string.push(*c3);
                            self.output_string.push(*c4);
                            self.output_string.push(*c5);
                            self.output_string.push(*c6);
                            self.output_string.push(*c7);
                        }
                        PhoneChar::Sequence(chars) => {
                            for c in chars {
                                self.output_string.push(*c);
                            }
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
        let dict = PhoneticNormalizedDictionary::<()>::from_terms([
            "phone", "fone", "elephant", "elegance",
        ]);

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
        let dict =
            PhoneticNormalizedDictionary::<()>::from_terms(["phone", "bone", "cone", "tone"]);

        // Query with distance tolerance
        let results = dict.query("fone", 1);
        println!("Results for 'fone' with distance 1: {:?}", results);

        // Should find matches
        assert!(!results.is_empty());
    }

    #[test]
    fn test_sort_candidates_by_relevance_preserves_ordering_contract() {
        let results = sort_candidates_by_relevance(
            "cat",
            vec![
                PhoneticNormalizedCandidate {
                    term: "bat".to_string(),
                    distance: 1,
                    normalized_form: "bat".to_string(),
                },
                PhoneticNormalizedCandidate {
                    term: "far".to_string(),
                    distance: 0,
                    normalized_form: "far".to_string(),
                },
                PhoneticNormalizedCandidate {
                    term: "cart".to_string(),
                    distance: 1,
                    normalized_form: "cart".to_string(),
                },
                PhoneticNormalizedCandidate {
                    term: "cat".to_string(),
                    distance: 1,
                    normalized_form: "cat".to_string(),
                },
            ],
        );

        let terms: Vec<_> = results
            .iter()
            .map(|candidate| candidate.term.as_str())
            .collect();
        assert_eq!(terms, ["far", "cat", "bat", "cart"]);
    }

    #[test]
    fn test_candidate_best_match_keeps_single_best_entry_per_term() {
        let mut by_term = HashMap::new();

        upsert_candidate_best_match(&mut by_term, "phone".to_string(), 2, "fone");
        upsert_candidate_best_match(&mut by_term, "phone", 1, "phone");
        upsert_candidate_best_match(&mut by_term, "phone", 3, "phane");
        upsert_candidate_best_match(&mut by_term, "", 0, "");

        let candidates = best_matches_into_candidates(by_term);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].term, "phone");
        assert_eq!(candidates[0].distance, 1);
        assert_eq!(candidates[0].normalized_form, "phone");
    }

    #[test]
    fn test_term_id_dictionary_matches_string_payload_exact_query() {
        let terms = ["phone", "fone", "bone", "cone", "tone", "elephant"];
        let string_payload = PhoneticNormalizedDictionary::<()>::from_terms(terms);
        let term_id_payload = PhoneticNormalizedTermIdDictionary::from_terms(terms);
        let fallible_term_id_payload = PhoneticNormalizedTermIdDictionary::try_from_terms(terms)
            .expect("terms fit in u32 ids");

        assert_eq!(term_id_payload.term_count(), terms.len());
        assert_eq!(fallible_term_id_payload.term_count(), terms.len());
        assert_eq!(
            term_id_payload.normalized_count(),
            string_payload.normalized_count()
        );
        assert_eq!(
            term_id_payload.query("fone", 0),
            string_payload.query("fone", 0)
        );
        assert_eq!(
            fallible_term_id_payload.query("fone", 0),
            string_payload.query("fone", 0)
        );
    }

    #[test]
    fn test_term_id_conversion_accepts_largest_u32_id() {
        assert_eq!(compact_term_id_from_len(u32::MAX as usize), Ok(u32::MAX));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_term_id_conversion_rejects_ids_beyond_u32_range() {
        let next_id = u32::MAX as usize + 1;

        let error = compact_term_id_from_len(next_id).expect_err("id must exceed u32");

        assert_eq!(
            error,
            TermIdDictionaryBuildError::TermIdOverflow { next_id }
        );
        assert_eq!(
            error.to_string(),
            "term-id dictionary cannot assign term id 4294967296; compact ids are stored as u32"
        );
    }

    #[test]
    fn test_term_id_dictionary_matches_string_payload_fuzzy_query() {
        let terms = [
            "phone", "fone", "bone", "cone", "tone", "elephant", "elefant",
        ];
        let string_payload = PhoneticNormalizedDictionary::<()>::from_terms(terms);
        let term_id_payload = PhoneticNormalizedTermIdDictionary::from_terms(terms);

        assert_eq!(
            term_id_payload.query("elefant", 1),
            string_payload.query("elefant", 1)
        );
        assert_eq!(
            term_id_payload.query("fone", 1),
            string_payload.query("fone", 1)
        );
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
        let dict =
            PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["phone", "fone"], vec![]);

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
        assert!((1..=3).contains(&count));
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

    fn exact_word_rule(id: usize, pattern: &str, replacement: &str) -> RewriteRuleChar {
        RewriteRuleChar {
            rule_id: id,
            rule_name: format!("{pattern}_to_{replacement}"),
            pattern: phone_chars_from_input(pattern),
            replacement: replacement.chars().map(PhoneChar::Consonant).collect(),
            context: ContextChar::And(Box::new(ContextChar::Initial), Box::new(ContextChar::Final)),
            weight: 0.0,
            syllable_condition: None,
        }
    }

    #[test]
    fn test_whole_word_alternate_normalized_forms() {
        let rules = vec![
            exact_word_rule(1, "read", "RED"),
            exact_word_rule(2, "read", "REED"),
            exact_word_rule(3, "red", "RED"),
            exact_word_rule(4, "reed", "REED"),
        ];
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["read", "red", "reed"],
            rules,
        );

        let results = dict.query("read", 0);
        let terms: HashSet<_> = results
            .iter()
            .map(|candidate| candidate.term.as_str())
            .collect();

        assert!(terms.contains("read"));
        assert!(terms.contains("red"));
        assert!(terms.contains("reed"));

        assert!(dict.remove("read"));
        assert!(!dict
            .query("red", 0)
            .iter()
            .any(|candidate| candidate.term == "read"));
        assert!(!dict
            .query("reed", 0)
            .iter()
            .any(|candidate| candidate.term == "read"));
    }

    #[test]
    fn test_duplicate_whole_word_rules_are_deduplicated_once() {
        let rules = vec![
            exact_word_rule(1, "read", "RED"),
            exact_word_rule(2, "read", "RED"),
            exact_word_rule(3, "read", "REED"),
        ];
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["read"], rules);

        assert_eq!(
            dict.normalized_forms("read"),
            vec!["RED".to_string(), "REED".to_string()]
        );
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
    fn test_iter_terms_deduplicates_multiple_normalized_forms() {
        let rules = vec![
            exact_word_rule(1, "read", "RED"),
            exact_word_rule(2, "read", "REED"),
        ];
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["read"], rules);

        let terms: Vec<_> = dict.iter_terms().map(|(term, _)| term).collect();
        assert_eq!(terms, ["read".to_string()]);
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
    fn test_query_with_product_trie_matches_scan() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["cat", "car", "card", "care", "cart", "bat", "bar"],
            vec![],
        );
        let ast = parse_regex("ca.").expect("regex should parse");
        let nfa = compile_nfa(&ast).expect("nfa should compile");
        let product = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::Standard);

        let mut trie_results: Vec<_> = dict
            .query_with_product(&product)
            .into_iter()
            .map(|c| (c.term, c.distance, c.normalized_form))
            .collect();
        let mut scan_results: Vec<_> = dict
            .query_with_product_scan(&product)
            .into_iter()
            .map(|c| (c.term, c.distance, c.normalized_form))
            .collect();
        trie_results.sort();
        scan_results.sort();

        assert_eq!(trie_results, scan_results);
    }

    #[test]
    fn test_query_regex_matches_precompiled_product_query() {
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["cat", "car", "card", "care", "cart", "bat", "bar"],
            vec![],
        );
        let pattern = "ca.";
        let ast = parse_regex(pattern).expect("regex should parse");
        let nfa = compile_nfa(&ast).expect("nfa should compile");
        let product = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::Standard);

        let mut regex_results: Vec<_> = dict
            .query_regex(pattern, 1)
            .expect("regex query should succeed")
            .into_iter()
            .map(|c| (c.term, c.distance, c.normalized_form))
            .collect();
        let mut product_results: Vec<_> = dict
            .query_with_product(&product)
            .into_iter()
            .map(|c| (c.term, c.distance, c.normalized_form))
            .collect();
        regex_results.sort();
        product_results.sort();

        assert_eq!(regex_results, product_results);
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
            syllable_condition: None,
        }];

        let dict =
            PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["phone", "fone"], rules);

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
            syllable_condition: None,
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
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["cat", "car", "bat"],
            vec![],
        );

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
            syllable_condition: None,
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
    fn test_query_original_regex() {
        use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

        // Create a dictionary with a simple rule: ph -> f
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "ph_to_f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.1,
            syllable_condition: None,
        }];

        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            ["phone", "fone", "bone", "cone"],
            rules,
        );

        // Get the expanded pattern for "fone"
        let pattern = dict.expand_to_phonetic_pattern("fone");

        // Query ORIGINAL terms with the expanded pattern
        let results = dict
            .query_original_regex(&pattern, 0)
            .expect("pattern should parse");

        // Should find both "phone" (matches "ph" + "on") and "fone" (matches "f" + "on")
        assert!(
            results.iter().any(|c| c.term == "phone"),
            "Should find 'phone' via original regex"
        );
        assert!(
            results.iter().any(|c| c.term == "fone"),
            "Should find 'fone' via original regex"
        );
        // "bone" and "cone" don't match the pattern
        assert!(
            !results.iter().any(|c| c.term == "bone"),
            "Should NOT find 'bone'"
        );
        assert!(
            !results.iter().any(|c| c.term == "cone"),
            "Should NOT find 'cone'"
        );
    }

    #[test]
    fn test_expand_normalizes_input() {
        use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

        // Create a dictionary with rules: ph -> f, ne -> n (to show normalization effect)
        let rules = vec![RewriteRuleChar {
            rule_id: 1,
            rule_name: "ph_to_f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.1,
            syllable_condition: None,
        }];

        let dict =
            PhoneticNormalizedDictionary::<()>::from_terms_with_rules(["phone", "fone"], rules);

        // Both "phone" and "fone" normalize to "fon"
        let phone_norm = dict.normalize("phone");
        let fone_norm = dict.normalize("fone");
        assert_eq!(phone_norm, fone_norm, "Both should normalize to same form");

        // Expanding either should give the same pattern (since both normalize first)
        let pattern_from_phone = dict.expand_to_phonetic_pattern("phone");
        let pattern_from_fone = dict.expand_to_phonetic_pattern("fone");
        assert_eq!(
            pattern_from_phone, pattern_from_fone,
            "Expanding either query should give same pattern"
        );
        println!("Pattern: {}", pattern_from_fone);
    }
}
