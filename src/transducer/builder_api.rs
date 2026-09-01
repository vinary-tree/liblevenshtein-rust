//! Fluent builder API for constructing queries
//!
//! This module provides a more ergonomic, self-documenting API for querying
//! dictionaries with various options.

use super::{
    AffineGapParams, AffineQueryIterator, Algorithm, OrderedQueryIterator, QueryIterator,
    SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted,
};
use libdictenstein::Dictionary;

/// Fluent builder for constructing Levenshtein queries
///
/// # Examples
///
/// ```rust
/// use liblevenshtein::prelude::*;
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Simple query
/// let results: Vec<_> = transducer
///     .query_builder("tset")
///     .max_distance(2)
///     .execute()
///     .collect();
///
/// // Ordered query with prefix matching
/// let results: Vec<_> = transducer
///     .query_builder("te")
///     .max_distance(1)
///     .ordered()
///     .prefix()
///     .take(10)
///     .collect();
/// ```
pub struct QueryBuilder<'a, D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: &'a D,
    term: String,
    max_distance: usize,
    algorithm: Algorithm,
    policy: P,
    suffix_based: bool,
}

impl<'a, D: Dictionary, P: SubstitutionPolicy> QueryBuilder<'a, D, P> {
    /// Create a new query builder
    pub(crate) fn new(
        dictionary: &'a D,
        term: impl Into<String>,
        default_distance: usize,
        algorithm: Algorithm,
        policy: P,
        suffix_based: bool,
    ) -> Self {
        Self {
            dictionary,
            term: term.into(),
            max_distance: default_distance,
            algorithm,
            policy,
            suffix_based,
        }
    }

    /// Set the maximum edit distance
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test", "testing"]);
    /// let transducer = Transducer::standard(dict);
    /// let results: Vec<_> = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)  // Allow up to 2 edits
    ///     .execute()
    ///     .collect();
    /// assert!(results.contains(&"test".to_owned()));
    /// ```
    pub fn max_distance(mut self, distance: usize) -> Self {
        self.max_distance = distance;
        self
    }

    /// Set the Levenshtein algorithm
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test"]);
    /// let transducer = Transducer::standard(dict);
    /// let results: Vec<_> = transducer
    ///     .query_builder("tset")
    ///     .max_distance(1)
    ///     .algorithm(Algorithm::Transposition)
    ///     .execute()
    ///     .collect();
    /// assert_eq!(results, ["test"]);
    /// ```
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Execute the query and return an iterator over matching terms
    ///
    /// Returns terms in arbitrary order as they are found during traversal.
    ///
    /// # Note
    ///
    /// For prefix matching, use `.ordered().prefix()`.
    pub fn execute(self) -> QueryIterator<D::Node, String, P>
    where
        P: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        QueryIterator::with_traversal_root_and_policy(
            self.dictionary.traversal_root(),
            self.term,
            self.max_distance,
            self.algorithm,
            self.policy,
            self.suffix_based,
        )
    }

    /// Execute this builder's term as an affine-gap query.
    ///
    /// The affine budget is independent of [`max_distance`](Self::max_distance)
    /// because it is expressed in the parameter set's cost domain.
    pub fn affine_gap(
        self,
        max_cost: f64,
        params: AffineGapParams,
    ) -> Result<AffineQueryIterator<D::Node, P>, crate::cost::ScaleError>
    where
        P: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        let max_cost = params.scale_cost(max_cost)?;
        let inner = QueryIterator::with_affine_traversal_root_and_substring(
            self.dictionary.traversal_root(),
            self.term,
            max_cost,
            params,
            self.policy,
            self.suffix_based,
        );
        Ok(AffineQueryIterator::new(inner, params))
    }

    /// Execute the query with ordered results
    ///
    /// Returns an ordered iterator that yields results sorted by:
    /// 1. Edit distance (ascending)
    /// 2. Lexicographic order (for ties)
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test", "best", "rest"]);
    /// let transducer = Transducer::standard(dict);
    /// let results: Vec<_> = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)
    ///     .ordered()
    ///     .take(5)  // Get top 5 closest matches
    ///     .map(|candidate| candidate.term)
    ///     .collect();
    /// assert_eq!(results.first().map(String::as_str), Some("test"));
    /// ```
    ///
    /// # Prefix Matching
    ///
    /// For prefix matching, chain `.prefix()` after this method:
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test", "tested", "best"]);
    /// let transducer = Transducer::standard(dict);
    /// let results: Vec<_> = transducer
    ///     .query_builder("te")
    ///     .max_distance(0)
    ///     .ordered()
    ///     .prefix()  // Match terms starting with query
    ///     .map(|candidate| candidate.term)
    ///     .collect();
    /// assert_eq!(results, ["test", "tested"]);
    /// ```
    pub fn ordered(self) -> OrderedQueryIterator<D::Node, P>
    where
        P: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        OrderedQueryIterator::with_traversal_root_and_policy_and_substring(
            self.dictionary.traversal_root(),
            self.term,
            self.max_distance,
            self.algorithm,
            self.policy,
            self.suffix_based,
        )
    }

    /// Execute and collect results into a vector
    ///
    /// Convenience method for common use case of collecting all results.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test", "best"]);
    /// let transducer = Transducer::standard(dict);
    /// let results = transducer
    ///     .query_builder("test")
    ///     .max_distance(1)
    ///     .collect_vec();
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn collect_vec(self) -> Vec<String>
    where
        P: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        self.execute().collect()
    }

    /// Execute with a limit on the number of results
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DoubleArrayTrie::from_terms(["test", "best", "rest"]);
    /// let transducer = Transducer::standard(dict);
    /// let results: Vec<_> = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)
    ///     .limit(2)
    ///     .collect();
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn limit(self, n: usize) -> impl Iterator<Item = String>
    where
        P: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        self.execute().take(n)
    }
}

#[cfg(test)]
mod tests {
    use crate::transducer::{Algorithm, SubstitutionSet, Transducer};
    use libdictenstein::double_array_trie::DoubleArrayTrie;

    #[test]
    fn test_query_builder_basic() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer
            .query_builder("test")
            .max_distance(0)
            .execute()
            .collect();

        assert_eq!(results, vec!["test"]);
    }

    #[test]
    fn test_query_builder_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer
            .query_builder("test")
            .max_distance(1)
            .execute()
            .collect();

        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"best".to_string()));
        assert!(results.contains(&"rest".to_string()));
    }

    #[test]
    fn test_query_builder_ordered_prefix() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested", "best"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Use .ordered().prefix() for prefix matching
        let results: Vec<_> = transducer
            .query_ordered("tes", 0)
            .prefix()
            .map(|c| c.term)
            .collect();

        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"testing".to_string()));
        assert!(results.contains(&"tested".to_string()));
        assert!(!results.contains(&"best".to_string()));
    }

    #[test]
    fn test_query_builder_ordered() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer
            .query_builder("test")
            .max_distance(2)
            .ordered()
            .take(3)
            .map(|c| c.term)
            .collect();

        // Exact match should come first
        assert_eq!(results[0], "test");
    }

    #[test]
    fn test_query_builder_collect_vec() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results = transducer
            .query_builder("test")
            .max_distance(1)
            .collect_vec();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_builder_limit() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer
            .query_builder("test")
            .max_distance(1)
            .limit(2)
            .collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_builder_preserves_transducer_policy() {
        let dict = DoubleArrayTrie::from_terms(vec!["cat"]);
        let mut substitutions = SubstitutionSet::new();
        substitutions.allow('c', 'k');

        let transducer = Transducer::with_substitutions(dict, Algorithm::Standard, substitutions);

        let results: Vec<_> = transducer
            .query_builder("kat")
            .max_distance(0)
            .execute()
            .collect();

        assert_eq!(results, vec!["cat"]);
    }
}
