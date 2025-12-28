//! Fluent builder API for constructing queries
//!
//! This module provides a more ergonomic, self-documenting API for querying
//! dictionaries with various options.

use super::{Algorithm, OrderedQueryIterator, QueryIterator, SubstitutionPolicyFor, Unrestricted};
use crate::dictionary::Dictionary;

/// Fluent builder for constructing Levenshtein queries
///
/// # Examples
///
/// ```rust,ignore
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
pub struct QueryBuilder<'a, D: Dictionary> {
    dictionary: &'a D,
    term: String,
    max_distance: usize,
    algorithm: Algorithm,
}

impl<'a, D: Dictionary> QueryBuilder<'a, D> {
    /// Create a new query builder
    pub(crate) fn new(
        dictionary: &'a D,
        term: impl Into<String>,
        default_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        Self {
            dictionary,
            term: term.into(),
            max_distance: default_distance,
            algorithm,
        }
    }

    /// Set the maximum edit distance
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)  // Allow up to 2 edits
    ///     .execute();
    /// ```
    pub fn max_distance(mut self, distance: usize) -> Self {
        self.max_distance = distance;
        self
    }

    /// Set the Levenshtein algorithm
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = transducer
    ///     .query_builder("test")
    ///     .algorithm(Algorithm::Transposition)
    ///     .execute();
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
    pub fn execute(self) -> QueryIterator<D::Node>
    where
        Unrestricted: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        QueryIterator::new(
            self.dictionary.root(),
            self.term,
            self.max_distance,
            self.algorithm,
        )
    }

    /// Execute the query with ordered results
    ///
    /// Returns an ordered iterator that yields results sorted by:
    /// 1. Edit distance (ascending)
    /// 2. Lexicographic order (for ties)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results: Vec<_> = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)
    ///     .ordered()
    ///     .take(5)  // Get top 5 closest matches
    ///     .collect();
    /// ```
    ///
    /// # Prefix Matching
    ///
    /// For prefix matching, chain `.prefix()` after this method:
    ///
    /// ```rust,ignore
    /// let results: Vec<_> = transducer
    ///     .query_builder("te")
    ///     .ordered()
    ///     .prefix()  // Match terms starting with query
    ///     .collect();
    /// ```
    pub fn ordered(self) -> OrderedQueryIterator<D::Node>
    where
        Unrestricted: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        OrderedQueryIterator::new(
            self.dictionary.root(),
            self.term,
            self.max_distance,
            self.algorithm,
        )
    }

    /// Execute and collect results into a vector
    ///
    /// Convenience method for common use case of collecting all results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = transducer
    ///     .query_builder("test")
    ///     .max_distance(1)
    ///     .collect_vec();
    /// ```
    pub fn collect_vec(self) -> Vec<String>
    where
        Unrestricted: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        self.execute().collect()
    }

    /// Execute with a limit on the number of results
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = transducer
    ///     .query_builder("test")
    ///     .max_distance(2)
    ///     .limit(10);
    /// ```
    pub fn limit(self, n: usize) -> impl Iterator<Item = String>
    where
        Unrestricted: SubstitutionPolicyFor<<D::Node as crate::dictionary::DictionaryNode>::Unit>,
    {
        self.execute().take(n)
    }
}

#[cfg(test)]
mod tests {
    use crate::dictionary::double_array_trie::DoubleArrayTrie;
    use crate::transducer::{Algorithm, Transducer};

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
}
