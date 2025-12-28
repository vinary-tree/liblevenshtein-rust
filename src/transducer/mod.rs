//! Lazy Levenshtein automata for approximate string matching.
//!
//! This module implements lazy (on-demand) construction of Levenshtein automata
//! for efficient fuzzy string matching against dictionaries.
//!
//! Also known as **Parameterized Levenshtein Automata** in academic literature
//! (Schulz & Mihov, 2002).
//!
//! # Terminology
//!
//! - **Lazy**: States constructed on-demand during queries
//! - **Parameterized**: Academic term emphasizing query-word-specific construction
//!
//! See [`universal`] module for eager (precomputed) automata.

mod algorithm;
mod automaton_zipper;
pub mod builder;
mod builder_api;
pub mod costs_f64;
pub mod helpers;
mod intersection;
mod intersection_f64;
pub mod intersection_zipper;
pub mod operation_set;
pub mod operation_type;
mod ordered_query;
pub mod phonetic;
#[cfg(feature = "phonetic-rules")]
pub mod phonetic_transducer;
mod pool;
mod pool_f64;
mod position;
mod position_f64;
mod query;
mod query_f64;
mod query_result;
mod state;
mod state_f64;
pub mod substitution_policy;
pub mod substitution_set;
pub mod substitution_set_char;
pub mod transition;
pub mod transition_f64;
pub mod universal;
pub mod generalized;
mod value_filtered_query;
mod zipper_query_iterator;

#[cfg(feature = "simd")]
pub mod simd;

pub use algorithm::Algorithm;
pub use automaton_zipper::AutomatonZipper;
pub use builder::{BuilderError, TransducerBuilder};
pub use builder_api::QueryBuilder;
pub use costs_f64::OperationCostsF64;
pub use intersection::{Intersection, PathNode};
pub use intersection_f64::IntersectionF64;
pub use intersection_zipper::IntersectionZipper;
pub use operation_set::{OperationSet, OperationSetBuilder};
pub use operation_type::OperationType;
pub use ordered_query::{OrderedCandidate, OrderedQueryIterator};
pub use pool::StatePool;
pub use pool_f64::StatePoolF64;
pub use position::Position;
pub use position_f64::PositionF64;
pub use query::{Candidate, CandidateIterator, QueryIterator, StringQueryIterator};
pub use query_f64::{
    CandidateF64, CandidateIteratorF64, QueryIteratorF64, QueryResultF64, StringQueryIteratorF64,
};
pub use query_result::QueryResult;
pub use state::State;
pub use state_f64::StateF64;
pub use substitution_policy::{
    Restricted, RestrictedChar, SubstitutionPolicy, SubstitutionPolicyChar, SubstitutionPolicyFor, Unrestricted,
};
pub use substitution_set::SubstitutionSet;
pub use substitution_set_char::SubstitutionSetChar;
pub use transition_f64::{
    initial_state_f64, transition_position_f64, transition_state_f64, transition_state_pooled_f64,
};
pub use value_filtered_query::{ValueFilteredQueryIterator, ValueSetFilteredQueryIterator};
pub use zipper_query_iterator::ZipperQueryIterator;

#[cfg(feature = "phonetic-rules")]
pub use phonetic_transducer::{
    PhoneticCandidate, PhoneticCandidateByte, PhoneticQueryIterator, PhoneticQueryIteratorChar,
    PhoneticTransducer, PhoneticTransducerChar,
};

use crate::dictionary::{Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode};
use std::collections::HashSet;

/// Main transducer for approximate string matching.
///
/// The transducer combines a dictionary with a Levenshtein automaton
/// to efficiently find all terms within a given edit distance of a query.
///
/// # Type Parameters
///
/// - `D`: Dictionary type implementing [`Dictionary`]
/// - `P`: Substitution policy (defaults to [`Unrestricted`])
///
/// The default [`Unrestricted`] policy is a zero-sized type, so there is
/// zero memory or performance overhead for the default case.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// for term in transducer.query("tset", 2) {
///     println!("Found: {}", term);
/// }
/// ```
///
/// # Custom Substitution Policy
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
///
/// // Allow phonetic substitutions like 'f' ↔ 'ph', 'c' ↔ 'k'
/// let policy_set = SubstitutionSet::phonetic_basic();
/// let policy = Restricted::new(&policy_set);
///
/// let dict = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
/// let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);
///
/// // "fone" matches "phone" with restricted substitutions
/// for term in transducer.query("fone", 1) {
///     println!("Found: {}", term);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: D,
    algorithm: Algorithm,
    policy: P,
}

// Constructors for Unrestricted policy (backward compatible)
impl<D: Dictionary> Transducer<D, Unrestricted>
where
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    /// Create a new transducer with the given dictionary and algorithm
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self {
        Self {
            dictionary,
            algorithm,
            policy: Unrestricted,
        }
    }

    /// Create a transducer with the Standard algorithm.
    ///
    /// This is a convenience constructor for the most common use case.
    /// The Standard algorithm supports insert, delete, and substitute operations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
    /// let transducer = Transducer::standard(dict);
    /// // Equivalent to: Transducer::new(dict, Algorithm::Standard)
    /// ```
    pub fn standard(dictionary: D) -> Self {
        Self::new(dictionary, Algorithm::Standard)
    }

    /// Create a transducer with the Transposition algorithm.
    ///
    /// The Transposition algorithm adds support for swapping adjacent characters,
    /// useful for catching common typos like "teh" → "the".
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["the", "quick"]);
    /// let transducer = Transducer::with_transposition(dict);
    /// // Will match "teh" to "the" with distance 1
    /// ```
    pub fn with_transposition(dictionary: D) -> Self {
        Self::new(dictionary, Algorithm::Transposition)
    }

    /// Create a transducer with the MergeAndSplit algorithm.
    ///
    /// The MergeAndSplit algorithm adds support for merge and split operations,
    /// useful for catching spacing errors like "every one" ↔ "everyone".
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["everyone", "someone"]);
    /// let transducer = Transducer::with_merge_split(dict);
    /// // Will match "every one" to "everyone" with distance 1
    /// ```
    pub fn with_merge_split(dictionary: D) -> Self {
        Self::new(dictionary, Algorithm::MergeAndSplit)
    }

    /// Create a transducer with custom substitutions.
    ///
    /// This is a convenience method for creating a transducer with the Standard
    /// algorithm and a restricted substitution policy. It's useful when you want
    /// to allow specific character equivalences (like phonetic matching).
    ///
    /// For more control, use [`with_policy`](Self::with_policy) directly.
    ///
    /// # Parameters
    ///
    /// - `dictionary`: The dictionary to search
    /// - `algorithm`: The Levenshtein algorithm variant
    /// - `substitution_set`: Set of allowed character substitutions
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::transducer::{SubstitutionSet, Algorithm};
    ///
    /// // Create phonetic substitution set
    /// let substitutions = SubstitutionSet::phonetic_basic();
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
    /// let transducer = Transducer::with_substitutions(
    ///     dict,
    ///     Algorithm::Standard,
    ///     substitutions
    /// );
    ///
    /// // "fone" matches "phone" via f↔p phonetic equivalence
    /// let results: Vec<_> = transducer.query("fone", 1).collect();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SubstitutionSet::phonetic_basic()`] - Common phonetic equivalences
    /// - [`SubstitutionSet::keyboard_qwerty()`] - Keyboard proximity typos
    /// - [`SubstitutionSet::leet_speak()`] - Leetspeak substitutions
    /// - [`SubstitutionSet::ocr_friendly()`] - OCR confusion pairs
    pub fn with_substitutions(
        dictionary: D,
        algorithm: Algorithm,
        substitution_set: SubstitutionSet,
    ) -> Transducer<D, Restricted<'static>>
    where
        <D::Node as DictionaryNode>::Unit: From<u8>,
        Restricted<'static>: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    {
        let set: &'static SubstitutionSet = Box::leak(Box::new(substitution_set));
        let policy = Restricted::new(set);
        Transducer::with_policy(dictionary, algorithm, policy)
    }
}

// Generic methods (work with any policy)
impl<D: Dictionary, P: SubstitutionPolicy + SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>> Transducer<D, P> {
    /// Create a transducer with a custom substitution policy.
    ///
    /// This allows you to restrict which character substitutions are allowed
    /// during matching. For example, you can enable phonetic matching where
    /// 'f' and 'ph' are considered equivalent.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// // Create a phonetic substitution set
    /// let mut policy_set = SubstitutionSet::new();
    /// policy_set.allow('f', 'p');  // Allow f ↔ p
    /// policy_set.allow('c', 'k');  // Allow c ↔ k
    ///
    /// let policy = Restricted::new(&policy_set);
    /// let dict = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
    /// let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);
    ///
    /// // "fone" will match "phone" via f↔p substitution
    /// for term in transducer.query("fone", 1) {
    ///     println!("Found: {}", term);
    /// }
    /// ```
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self {
        Self {
            dictionary,
            algorithm,
            policy,
        }
    }

    /// Get the algorithm used by this transducer
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get a reference to the underlying dictionary
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Extract the underlying dictionary, consuming the transducer.
    ///
    /// This is useful when you need to:
    /// - Serialize the dictionary independently
    /// - Perform maintenance operations outside the transducer context
    /// - Reuse the dictionary in another transducer or engine
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "pathmap-backend")]
    /// # {
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let dict: PathMapDictionary = PathMapDictionary::from_terms(["test", "testing"]);
    /// let transducer = Transducer::new(dict, Algorithm::Standard);
    ///
    /// // Extract the dictionary
    /// let dict = transducer.into_inner();
    /// assert_eq!(dict.len(), Some(2));
    /// # }
    /// ```
    #[inline]
    pub fn into_inner(self) -> D {
        self.dictionary
    }

    /// Alias for `into_inner()` - extracts the underlying dictionary.
    ///
    /// Provided for semantic clarity when specifically working with dictionaries.
    #[inline]
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }

    /// Create a fluent query builder
    ///
    /// Provides a more ergonomic, self-documenting API for querying.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
    /// let transducer = Transducer::new(dict, Algorithm::Standard);
    ///
    /// // Fluent API
    /// let results: Vec<_> = transducer
    ///     .query_builder("tset")
    ///     .max_distance(2)
    ///     .limit(10)
    ///     .collect();
    ///
    /// // Ordered results
    /// let top_matches: Vec<_> = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)
    ///     .ordered()
    ///     .take(5)
    ///     .map(|c| c.term)
    ///     .collect();
    /// ```
    pub fn query_builder(&self, term: impl Into<String>) -> QueryBuilder<'_, D> {
        QueryBuilder::new(&self.dictionary, term, 2, self.algorithm)
    }

    /// Query for terms within `max_distance` edits of `term`
    ///
    /// Returns an iterator over matching terms (strings only)
    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<D::Node, String, P> {
        QueryIterator::with_policy_and_substring(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy,
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query for terms with their edit distances
    ///
    /// Returns an iterator over `Candidate` structs containing both
    /// the matching term and its edit distance computed from automaton states
    pub fn query_with_distance(
        &self,
        term: &str,
        max_distance: usize,
    ) -> QueryIterator<D::Node, Candidate, P> {
        QueryIterator::with_policy_and_substring(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy,
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query for terms in distance-first, lexicographic order
    ///
    /// Returns an iterator that yields results ordered by:
    /// 1. Primary: Ascending edit distance (0, 1, 2, ...)
    /// 2. Secondary: Lexicographic (alphabetical)
    ///
    /// This ordering enables efficient "top-k" queries and take-while patterns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest"]);
    /// let transducer = Transducer::new(dict, Algorithm::Standard);
    ///
    /// // Get first 3 closest matches
    /// for candidate in transducer.query_ordered("tset", 2).take(3) {
    ///     println!("{}: {}", candidate.term, candidate.distance);
    /// }
    ///
    /// // Get all matches within distance 1
    /// for candidate in transducer.query_ordered("tset", 2)
    ///     .take_while(|c| c.distance <= 1)
    /// {
    ///     println!("{}", candidate.term);
    /// }
    /// ```
    pub fn query_ordered(&self, term: &str, max_distance: usize) -> OrderedQueryIterator<D::Node, P> {
        OrderedQueryIterator::with_policy_and_substring(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy,
            self.dictionary.is_suffix_based(),
        )
    }

    /// Alias for [`query`](Self::query) - returns matching term strings.
    ///
    /// This is a more descriptive name that makes it clear the method returns
    /// just the term strings without distance information.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
    /// let transducer = Transducer::standard(dict);
    ///
    /// for term in transducer.query_terms("tset", 2) {
    ///     println!("Match: {}", term);
    /// }
    /// ```
    pub fn query_terms(&self, term: &str, max_distance: usize) -> QueryIterator<D::Node, String, P> {
        self.query(term, max_distance)
    }

    /// Alias for [`query_with_distance`](Self::query_with_distance) - returns candidates with distances.
    ///
    /// This is a more concise name for getting both terms and their edit distances.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest"]);
    /// let transducer = Transducer::standard(dict);
    ///
    /// for candidate in transducer.query_candidates("test", 1) {
    ///     println!("{}: distance {}", candidate.term, candidate.distance);
    /// }
    /// ```
    pub fn query_candidates(
        &self,
        term: &str,
        max_distance: usize,
    ) -> QueryIterator<D::Node, Candidate, P> {
        self.query_with_distance(term, max_distance)
    }

    /// Alias for [`query_ordered`](Self::query_ordered) - returns ranked results by distance.
    ///
    /// This name emphasizes that results are returned in ranked order
    /// (closest matches first).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest"]);
    /// let transducer = Transducer::standard(dict);
    ///
    /// // Get top 5 closest matches
    /// for candidate in transducer.query_ranked("test", 2).take(5) {
    ///     println!("{}: distance {}", candidate.term, candidate.distance);
    /// }
    /// ```
    pub fn query_ranked(&self, term: &str, max_distance: usize) -> OrderedQueryIterator<D::Node, P> {
        self.query_ordered(term, max_distance)
    }
}

// Value-aware query methods (only available for MappedDictionary)
impl<D, P> Transducer<D, P>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    /// Query with value-based filtering during traversal.
    ///
    /// This method filters candidates based on their associated values **during**
    /// graph traversal, providing 10-100x speedup compared to post-filtering for
    /// selective predicates.
    ///
    /// # Performance
    ///
    /// - **Post-filtering**: Explores 100% of matches, filters afterwards
    /// - **Value-filtered**: Only explores branches matching the predicate
    ///
    /// For a query matching 1000 terms where only 10 are in the target scope:
    /// - Post-filtering: Explores 1000 terms, returns 10 (slow)
    /// - Value-filtered: Explores ~10-50 terms, returns 10 (fast!)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    ///
    /// // Dictionary with scope IDs
    /// let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
    ///     ("println", 1),    // std scope
    ///     ("my_func", 2),    // local scope
    ///     ("other_func", 3), // other scope
    /// ]);
    ///
    /// let transducer = Transducer::new(dict, Algorithm::Standard);
    ///
    /// // Query for "func" in local scope only
    /// let matches: Vec<_> = transducer
    ///     .query_filtered("func", 2, |scope_id| *scope_id == 2)
    ///     .collect();
    ///
    /// // Returns: [("my_func", 2)] - other_func is never explored!
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Code completion**: Filter by lexical scope ID
    /// - **Tagged search**: Filter by category, tag, or metadata
    /// - **Access control**: Filter by permission level
    /// - **Multi-tenancy**: Filter by tenant ID
    pub fn query_filtered<F>(
        &self,
        term: &str,
        max_distance: usize,
        filter: F,
    ) -> ValueFilteredQueryIterator<D::Node, F>
    where
        F: Fn(&D::Value) -> bool,
    {
        ValueFilteredQueryIterator::new(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            filter,
        )
    }

    /// Query with value set membership filtering.
    ///
    /// Optimized for the common case of checking if a value is in a set.
    /// This is particularly useful for hierarchical scope queries where you
    /// want to match terms visible from multiple nested scopes.
    ///
    /// # Example: Hierarchical Lexical Scope
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    ///
    /// let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
    ///     ("println", 1),    // global scope
    ///     ("format", 1),     // global scope
    ///     ("my_func", 2),    // module scope
    ///     ("helper", 3),     // function scope
    ///     ("local_var", 4),  // block scope
    /// ]);
    ///
    /// let transducer = Transducer::new(dict, Algorithm::Standard);
    ///
    /// // In block scope 4, can see: global(1), module(2), function(3), block(4)
    /// let visible_scopes: HashSet<u32> = [1, 2, 3, 4].iter().cloned().collect();
    ///
    /// let matches: Vec<_> = transducer
    ///     .query_by_value_set("func", 2, &visible_scopes)
    ///     .map(|c| c.term)
    ///     .collect();
    ///
    /// // Returns: ["my_func", "helper"] - only from visible scopes
    /// // Does NOT return items from other modules/files
    /// ```
    ///
    /// # Performance
    ///
    /// For a 10,000-term dictionary where 100 terms are in visible scopes:
    /// - Post-filtering: ~10ms (explores all 10,000 terms)
    /// - Value-filtered: ~0.1ms (explores ~100-500 terms)
    /// - **100x speedup!**
    pub fn query_by_value_set(
        &self,
        term: &str,
        max_distance: usize,
        value_set: &HashSet<D::Value>,
    ) -> ValueSetFilteredQueryIterator<D::Node, D::Value>
    where
        D::Value: Eq + std::hash::Hash + Clone,
    {
        ValueSetFilteredQueryIterator::new(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            value_set.clone(),
        )
    }
}
