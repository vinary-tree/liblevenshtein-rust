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
//! # Distance semantics and indexing
//!
//! A **metric** is a non-negative, symmetric distance that is zero only for
//! identical inputs and satisfies the triangle inequality. Metric trees such as
//! BK-trees and VP-trees use that last law for pruning. A trie walker based on an
//! admissible dynamic-programming lower bound has a different proof obligation
//! and does not inherently require a metric.
//!
//! | [`Algorithm`] | Distance semantics | Metric? | Safe for metric-tree pruning? |
//! |---|---|---:|---:|
//! | [`Algorithm::Standard`] | Levenshtein | yes | yes |
//! | [`Algorithm::Transposition`] | optimal string alignment (restricted Damerau) | no | no |
//! | [`Algorithm::MergeAndSplit`] | generic symmetric merge/split metric | yes | yes |
//! | [`Algorithm::DamerauLevenshtein`] | unrestricted Damerau–Levenshtein | yes | yes |
//!
//! Call [`Algorithm::is_metric`] when selecting an index whose correctness
//! contract explicitly requires the triangle inequality.
//!
//! See [`universal`] module for eager (precomputed) automata.

mod algorithm;
pub mod articulatory_costs;
mod automaton_zipper;
pub mod builder;
mod builder_api;
mod contextual_costs;
mod contextual_query;
pub mod costs_f64;
mod dictionary_traversal;
pub mod generalized;
pub mod helpers;
mod intersection;
mod intersection_f64;
pub mod intersection_zipper;
pub mod language;
mod match_mode;
pub mod operation_set;
#[cfg(feature = "serialization")]
mod operation_set_binary;
#[cfg(feature = "compression")]
mod operation_set_gzip;
#[cfg(feature = "protobuf")]
mod operation_set_protobuf;
pub mod operation_type;
mod ordered_query;
mod packed_lanes;
mod packed_standard;
pub mod phonetic;
#[cfg(feature = "phonetic-rules")]
pub mod phonetic_transducer;
mod pool;
mod pool_f64;
mod position;
mod position_f64;
mod prefix_pruner;
mod prefix_query;
mod presets;
mod priority_query;
mod query;
pub mod query_cache;
mod query_f64;
mod query_result;
mod ranked_value_query;
mod state;
mod state_f64;
mod subsequence_query;
pub mod substitution_policy;
pub mod substitution_set;
pub mod substitution_set_char;
mod suggestion;
pub mod transition;
pub mod transition_f64;
pub mod universal;
mod value_filtered_query;
mod variant;
mod variants;
mod zipper_query_iterator;

#[cfg(target_arch = "x86_64")]
pub mod simd;

pub use algorithm::Algorithm;
pub use articulatory_costs::ArticulatoryCosts;
pub use automaton_zipper::AutomatonZipper;
pub use builder::{BuilderError, TransducerBuilder};
pub use builder_api::QueryBuilder;
pub use costs_f64::OperationCostsF64;
pub use intersection::{Intersection, PathNode};
pub use intersection_f64::IntersectionF64;
pub use intersection_zipper::IntersectionZipper;
pub use match_mode::{MatchMode, MatchModeError, MatchModeQueryIterator};
pub use operation_set::{
    EmptySideRate, OperationSet, OperationSetBuilder, OperationSetValidationError,
    MAX_OPERATION_SET_TOTAL_CONSUMPTION,
};
#[cfg(feature = "serialization")]
pub use operation_set_binary::{
    OperationSetBinaryError, OperationSetBinaryLimits, MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
    OPERATION_SET_BINARY_MAGIC, OPERATION_SET_BINARY_VERSION,
};
#[cfg(feature = "compression")]
pub use operation_set_gzip::{OperationSetGzipError, MAX_OPERATION_SET_GZIP_INPUT_BYTES};
#[cfg(feature = "protobuf")]
pub use operation_set_protobuf::OperationSetProtobufError;
pub use operation_type::{OperationApplicability, OperationType};
pub use ordered_query::{OrderedCandidate, OrderedQueryIterator};
pub use pool::StatePool;
pub use pool_f64::StatePoolF64;
pub use position::Position;
pub use position_f64::PositionF64;
pub use prefix_pruner::{AllowedPrefixes, NoPruning, PrefixPruner};
pub use prefix_query::{PrefixQueryIterator, PrefixQueryMatch, PrefixQueryStats};
pub use priority_query::{
    priority_query, priority_query_with_policy, PriorityCandidate, PriorityQueryIterator,
};
pub use query::{
    AffineCandidate, AffineQueryIterator, Candidate, CandidateIterator, QueryIterator,
    StringQueryIterator, UnitCandidate, UnitCandidateIterator, UnitQueryIterator,
};
pub use query_cache::VersionedQueryCache;
pub use query_f64::{
    CandidateF64, CandidateIteratorF64, QueryIteratorF64, QueryResultF64, StringQueryIteratorF64,
    UnitCandidateF64, UnitCandidateIteratorF64, UnitQueryIteratorF64,
};
pub use query_result::QueryResult;
pub use ranked_value_query::RankedValueQueryIterator;
pub use state::State;
pub use state_f64::StateF64;
pub use subsequence_query::{SubsequenceMatch, SubsequenceQueryIterator, SubsequenceQueryStats};
pub use substitution_policy::{
    OwnedRestricted, Restricted, RestrictedChar, SubstitutionPolicy, SubstitutionPolicyChar,
    SubstitutionPolicyFor, Unrestricted,
};
pub use substitution_set::{
    SubstitutionPair, SubstitutionSet, MAX_SUBSTITUTION_PAIRS, MAX_SUBSTITUTION_TEXT_BYTES,
};
pub use substitution_set_char::SubstitutionSetChar;
pub use suggestion::{FrequencyValue, LogFrequencyScorer, Suggestion, SuggestionScorer};
pub use transition_f64::{
    initial_state_f64, transition_position_f64, transition_state_f64, transition_state_pooled_f64,
    TransitionSettingsF64,
};
pub use value_filtered_query::{
    ValueFilteredQueryIterator, ValueSetFilteredQueryIterator, ValueTerm,
    ValueYieldingQueryIterator,
};
pub use variant::PositionKind;
pub use variants::AffineGapParams;
pub use zipper_query_iterator::ZipperQueryIterator;

#[cfg(feature = "phonetic-rules")]
pub use phonetic_transducer::{
    PhoneticCandidate, PhoneticCandidateByte, PhoneticQueryIterator, PhoneticQueryIteratorChar,
    PhoneticTransducer, PhoneticTransducerChar, PhoneticValueCandidate, PhoneticValueCandidateByte,
    PhoneticValueQueryIterator, PhoneticValueQueryIteratorChar,
};

use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode,
};
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

    /// Create a transducer using unrestricted Damerau–Levenshtein distance.
    pub fn with_damerau_levenshtein(dictionary: D) -> Self {
        Self::new(dictionary, Algorithm::DamerauLevenshtein)
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
    ) -> Transducer<D, OwnedRestricted>
    where
        <D::Node as DictionaryNode>::Unit: From<u8>,
        OwnedRestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    {
        let policy = OwnedRestricted::new(substitution_set);
        Transducer::with_policy(dictionary, algorithm, policy)
    }
}

// Generic methods (work with any policy)
impl<
        D: Dictionary,
        P: SubstitutionPolicy + SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    > Transducer<D, P>
{
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
    pub fn query_builder(&self, term: impl Into<String>) -> QueryBuilder<'_, D, P> {
        QueryBuilder::new(
            &self.dictionary,
            term,
            2,
            self.algorithm,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query dictionary terms by standard edit distance to a regular language.
    ///
    /// Unlike [`query`](Self::query), which compares against one query sequence,
    /// this intersects the dictionary with all words recognized by `language`.
    /// The product is unit-generic and does not inspect this transducer's
    /// [`Algorithm`] or substitution policy; its semantics are unit-cost match,
    /// insertion, deletion, and substitution.
    pub fn query_language<L>(
        &self,
        language: L,
        max_distance: u8,
    ) -> language::LanguageQueryIterator<D::Node, L>
    where
        L: language::LanguageAutomaton<<D::Node as DictionaryNode>::Unit>,
    {
        let product = language::LanguageProduct::new(language, max_distance);
        language::LanguageQueryIterator::from_dictionary(&self.dictionary, product)
    }

    /// Compile a phonetic regular expression and query by distance to its
    /// language.
    #[cfg(feature = "phonetic-rules")]
    pub fn query_regex(
        &self,
        pattern: &str,
        max_distance: u8,
    ) -> crate::phonetic::regex::error::ParseResult<
        language::LanguageQueryIterator<D::Node, crate::phonetic::nfa::NFAChar>,
    >
    where
        D::Node: DictionaryNode<Unit = char>,
    {
        // Reject pathologically long source text before the parser builds its
        // (potentially left-deep) AST. The two-states-per-source-scalar policy
        // is intentionally conservative for escaped/syntax-heavy patterns;
        // the expansion-aware estimate below catches compact repetitions.
        let source_scalars = pattern.chars().count();
        let source_state_bound = source_scalars.saturating_mul(2);
        if source_state_bound > language::LANGUAGE_PRODUCT_MAX_STATES {
            return Err(crate::phonetic::regex::error::ParseError::new(
                crate::phonetic::regex::error::ParseErrorKind::PatternTooComplex {
                    size: source_state_bound,
                    max: language::LANGUAGE_PRODUCT_MAX_STATES,
                },
                crate::phonetic::common::Position::start(),
            ));
        }
        let regex = crate::phonetic::regex::parse(pattern)?;
        let estimated_states = crate::phonetic::nfa::estimate_thompson_states(&regex)?;
        if estimated_states > language::LANGUAGE_PRODUCT_MAX_STATES {
            return Err(crate::phonetic::regex::error::ParseError::new(
                crate::phonetic::regex::error::ParseErrorKind::PatternTooComplex {
                    size: estimated_states,
                    max: language::LANGUAGE_PRODUCT_MAX_STATES,
                },
                crate::phonetic::common::Position::start(),
            ));
        }
        let nfa = crate::phonetic::nfa::compile(&regex)?;
        if nfa.num_states() > language::LANGUAGE_PRODUCT_MAX_STATES {
            return Err(crate::phonetic::regex::error::ParseError::new(
                crate::phonetic::regex::error::ParseErrorKind::PatternTooComplex {
                    size: nfa.num_states(),
                    max: language::LANGUAGE_PRODUCT_MAX_STATES,
                },
                crate::phonetic::common::Position::start(),
            ));
        }
        Ok(self.query_language(nfa, max_distance))
    }

    /// Query for terms within `max_distance` edits of `term`
    ///
    /// Returns an iterator over matching terms (strings only)
    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<D::Node, String, P> {
        QueryIterator::with_traversal_root_and_policy(
            self.dictionary.traversal_root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
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
        QueryIterator::with_traversal_root_and_policy(
            self.dictionary.traversal_root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query with affine gap costs using exact fixed-point arithmetic.
    ///
    /// A gap of length `k` costs `gap_open + k * gap_extend`. Decimal
    /// parameters and `max_cost` are converted exactly; inexact budgets return
    /// [`ScaleError`](crate::cost::ScaleError) instead of being rounded.
    pub fn query_affine(
        &self,
        term: &str,
        max_cost: f64,
        params: AffineGapParams,
    ) -> Result<AffineQueryIterator<D::Node, P>, crate::cost::ScaleError> {
        let max_cost = params.scale_cost(max_cost)?;
        let inner = self.query_affine_scaled(term, max_cost, params);
        Ok(AffineQueryIterator::new(inner, params))
    }

    /// Query with affine costs and an already-scaled exact budget.
    ///
    /// Distances in the returned [`Candidate`] values use the same scale as
    /// `params`; call [`AffineGapParams::unscale_cost`] for presentation.
    pub fn query_affine_scaled(
        &self,
        term: &str,
        max_cost: usize,
        params: AffineGapParams,
    ) -> QueryIterator<D::Node, Candidate, P> {
        QueryIterator::with_affine_policy_and_substring(
            self.dictionary.root(),
            term.to_string(),
            max_cost,
            params,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query for terms within `max_distance` edits of a raw **unit sequence**.
    ///
    /// The units-native analogue of [`query`](Self::query): it takes the query as a
    /// `&[Unit]` slice — bypassing the `&str` → units conversion — and yields each
    /// match as its raw `Vec<Unit>` sequence. This is the entry point for token-id
    /// (`u64`) dictionaries built via `insert_sequence(&[u64])`: the string path
    /// ([`query`](Self::query)) would byte-pack the query via `CharUnit::from_str`
    /// and never match sequence-built data. For `u8`/`char` dictionaries it is
    /// equivalent to `query`, just returning `Vec<u8>`/`Vec<char>` instead of `String`.
    ///
    /// The automaton engine is fully unit-generic (equality-based characteristic
    /// vector), so all three [`Algorithm`]s work over any `Unit` under the default
    /// `Unrestricted` policy.
    pub fn query_units(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_distance: usize,
    ) -> QueryIterator<D::Node, Vec<<D::Node as DictionaryNode>::Unit>, P> {
        QueryIterator::with_traversal_root_and_units(
            self.dictionary.traversal_root(),
            units.to_vec(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query for unit-sequence terms with their edit distances.
    ///
    /// The units-native analogue of [`query_with_distance`](Self::query_with_distance):
    /// yields [`UnitCandidate`] (`{ term: Vec<Unit>, distance }`). See
    /// [`query_units`](Self::query_units) for why the `&[Unit]` surface is required
    /// for `u64` token dictionaries.
    pub fn query_units_with_distance(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_distance: usize,
    ) -> QueryIterator<D::Node, UnitCandidate<<D::Node as DictionaryNode>::Unit>, P> {
        QueryIterator::with_traversal_root_and_units(
            self.dictionary.traversal_root(),
            units.to_vec(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query native unit sequences with exact scaled affine-gap costs.
    pub fn query_units_affine_scaled(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_cost: usize,
        params: AffineGapParams,
    ) -> QueryIterator<D::Node, UnitCandidate<<D::Node as DictionaryNode>::Unit>, P> {
        QueryIterator::with_affine_units(
            self.dictionary.root(),
            units.to_vec(),
            max_cost,
            params,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query for unit-sequence terms within a **weighted** cost budget, using
    /// per-operation float costs ([`OperationCostsF64`]).
    ///
    /// The units-native, float-weighted analogue of [`query_units`](Self::query_units):
    /// it takes the query as a `&[Unit]` slice and yields [`UnitCandidateF64`]
    /// (`{ term: Vec<Unit>, distance: f64 }`), where `distance` is the minimal
    /// **weighted** edit cost `<= max_cost`. This lets a single `u64` token search
    /// prune on a combined cost (e.g. per-word substitution weights, or an n-gram
    /// `−log P` folded into `costs`) rather than plain unit-cost edit distance.
    pub fn query_units_weighted(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_cost: f64,
        costs: OperationCostsF64,
    ) -> QueryIteratorF64<D::Node, UnitCandidateF64<<D::Node as DictionaryNode>::Unit>, P> {
        QueryIteratorF64::with_traversal_root_and_units(
            self.dictionary.traversal_root(),
            units.to_vec(),
            max_cost,
            self.algorithm,
            costs,
            self.policy.clone(),
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
    pub fn query_ordered(
        &self,
        term: &str,
        max_distance: usize,
    ) -> OrderedQueryIterator<D::Node, P> {
        OrderedQueryIterator::with_policy_and_substring(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            self.dictionary.is_suffix_based(),
        )
    }

    /// Query using an ergonomic completed-candidate distance mode.
    ///
    /// This is sugar over [`query_ordered`](Self::query_ordered). `Exact` and
    /// the lower end of `Range` filter completed candidates; they cannot prune
    /// dictionary prefixes and therefore do not provide an algorithmic speedup.
    /// The underlying ordered iterator still uses the selected maximum as its
    /// automaton budget and remains lazy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::double_array_trie::DoubleArrayTrie;
    /// use liblevenshtein::transducer::{Algorithm, MatchMode, Transducer};
    ///
    /// let transducer = Transducer::new(
    ///     DoubleArrayTrie::from_terms(["cat", "bat", "cot", "coat"]),
    ///     Algorithm::Standard,
    /// );
    /// let terms: Vec<_> = transducer
    ///     .query_mode("cat", MatchMode::Exact(1))?
    ///     .map(|candidate| candidate.term)
    ///     .collect();
    /// assert_eq!(terms, ["bat", "coat", "cot"]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn query_mode(
        &self,
        term: &str,
        mode: MatchMode,
    ) -> Result<MatchModeQueryIterator<OrderedQueryIterator<D::Node, P>>, MatchModeError> {
        let (_, max_distance) = mode.bounds()?;
        MatchModeQueryIterator::try_new(self.query_ordered(term, max_distance), mode)
    }

    /// Query with a stateful prefix pruner during an explicit fuzzy DFS.
    ///
    /// This is a DFS surface because [`PrefixPruner`] has balanced enter/leave
    /// stack semantics. It returns the same match set and distances as
    /// [`query_units_with_distance`](Self::query_units_with_distance), after
    /// intersecting that set with the pruner's accepted terminal set. Result
    /// order is dictionary DFS order, not distance order.
    pub fn query_with_pruner<R>(
        &self,
        term: &str,
        max_distance: usize,
        pruner: R,
    ) -> PrefixQueryIterator<D::Node, P, R>
    where
        R: PrefixPruner<<D::Node as DictionaryNode>::Unit>,
    {
        PrefixQueryIterator::with_policy_and_pruner(
            self.dictionary.root(),
            <D::Node as DictionaryNode>::Unit::from_str(term),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            pruner,
            self.dictionary.is_suffix_based(),
        )
    }

    /// Unit-native counterpart of [`query_with_pruner`](Self::query_with_pruner).
    pub fn query_units_with_pruner<R>(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_distance: usize,
        pruner: R,
    ) -> PrefixQueryIterator<D::Node, P, R>
    where
        R: PrefixPruner<<D::Node as DictionaryNode>::Unit>,
    {
        PrefixQueryIterator::with_policy_and_pruner(
            self.dictionary.root(),
            units.to_vec(),
            max_distance,
            self.algorithm,
            self.policy.clone(),
            pruner,
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
    pub fn query_terms(
        &self,
        term: &str,
        max_distance: usize,
    ) -> QueryIterator<D::Node, String, P> {
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
    pub fn query_ranked(
        &self,
        term: &str,
        max_distance: usize,
    ) -> OrderedQueryIterator<D::Node, P> {
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
    /// Query mapped values lazily, ordered first by edit distance and then by
    /// decreasing scorer confidence within the current distance layer.
    pub fn query_suggestions<S>(
        &self,
        term: &str,
        max_distance: usize,
        scorer: S,
    ) -> RankedValueQueryIterator<D::Node, S, P>
    where
        S: SuggestionScorer<D::Value>,
        P: Clone,
    {
        RankedValueQueryIterator::with_policy(
            self.dictionary.root(),
            term.to_owned(),
            max_distance,
            self.algorithm,
            scorer,
            self.policy.clone(),
        )
    }

    /// Units-native suggestion query for token alphabets such as `u64`.
    pub fn query_unit_suggestions<S>(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_distance: usize,
        scorer: S,
    ) -> RankedValueQueryIterator<D::Node, S, P>
    where
        S: SuggestionScorer<D::Value>,
        P: Clone,
    {
        RankedValueQueryIterator::with_units(
            self.dictionary.root(),
            units.to_vec(),
            max_distance,
            self.algorithm,
            scorer,
            self.policy.clone(),
        )
    }

    /// Query with value-based filtering during result collection.
    ///
    /// This method checks each final node's associated value before materializing
    /// its term string, which reduces allocation work for predicates that reject
    /// many in-range candidates.
    ///
    /// # Performance
    ///
    /// - **Post-filtering**: Materializes fuzzy matches, then filters afterwards
    /// - **Value-filtered**: Traverses the same automaton graph, but filters before term construction
    ///
    /// For a query matching many terms where only a subset has the target value,
    /// value-filtered queries avoid allocating returned strings for rejected
    /// final nodes while still descending through those nodes' children.
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
    /// // Returns: [("my_func", 2)] - other_func fails the value predicate
    /// // before its term string is returned.
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

    /// Query yielding `(term, distance, value)` for each match within
    /// `max_distance`, reading each match's value during traversal so the
    /// caller avoids a second dictionary lookup per result.
    pub fn query_values(
        &self,
        term: &str,
        max_distance: usize,
    ) -> ValueYieldingQueryIterator<D::Node> {
        ValueYieldingQueryIterator::new(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
        )
    }

    /// Query yielding `(term: Vec<Unit>, distance, value)` for each match within
    /// `max_distance` of a raw **unit sequence**.
    ///
    /// The units-native analogue of [`query_values`](Self::query_values): it takes the
    /// query as a `&[Unit]` slice (bypassing the lossy `&str`→units byte-packing) and
    /// yields the matched key as its raw `Vec<Unit>` together with its stored value —
    /// so `T_lex`/`T_gram` recover both the corrected token-id sequence **and** its
    /// per-sequence value (e.g. an n-gram frequency or a term-id) in one pass, with no
    /// string round-trip. See [`query_units`](Self::query_units).
    pub fn query_units_values(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        max_distance: usize,
    ) -> ValueYieldingQueryIterator<D::Node, Vec<<D::Node as DictionaryNode>::Unit>> {
        ValueYieldingQueryIterator::with_unit_query(
            self.dictionary.root(),
            units.to_vec(),
            max_distance,
            self.algorithm,
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
    /// For a large dictionary with many in-range terms outside the visible set,
    /// this avoids allocating result strings for rejected terms while preserving
    /// the same fuzzy traversal semantics as a normal query.
    pub fn query_by_value_set<'a>(
        &self,
        term: &str,
        max_distance: usize,
        value_set: &'a HashSet<D::Value>,
    ) -> ValueSetFilteredQueryIterator<'a, D::Node, D::Value>
    where
        D::Value: Eq + std::hash::Hash,
    {
        ValueSetFilteredQueryIterator::new_borrowed(
            self.dictionary.root(),
            term.to_string(),
            max_distance,
            self.algorithm,
            value_set,
        )
    }
}
pub use contextual_costs::{ContextualCost, EditContext, EnglishSoftC, PositionalSilentE};
pub use contextual_query::{
    ContextualCandidate, ContextualQueryError, ContextualQueryIterator, ContextualQueryStats,
};
