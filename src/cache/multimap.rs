//! FuzzyMultiMap - Value aggregation from fuzzy-matched keys.
//!
//! This module implements the value aggregation layer originally requested by the user.
//! Given a fuzzy query, it finds all matching keys and aggregates their values using
//! collection-specific logic.
//!
//! # Overview
//!
//! FuzzyMultiMap is the primary feature of this refactoring. It:
//!
//! 1. Queries a fuzzy dictionary for matches within a distance threshold
//! 2. Retrieves the value for each matched key
//! 3. Aggregates all values using collection-type-specific logic
//!
//! # Example Use Case
//!
//! ```rust,ignore
//! use std::collections::HashSet;
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//! use liblevenshtein::cache::multimap::FuzzyMultiMap;
//!
//! // Map: foo -> {1,2}, bar -> {3}, baz -> {4,5}
//! let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
//! dict.insert_with_value("foo", HashSet::from([1, 2]));
//! dict.insert_with_value("bar", HashSet::from([3]));
//! dict.insert_with_value("baz", HashSet::from([4, 5]));
//!
//! let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
//!
//! // Query "bat" with distance 1
//! // Matches: "bar" (distance 1), "baz" (distance 1)
//! // Values: {3}, {4,5}
//! // Result: {3,4,5} (union)
//! let result = fuzzy.query("bat", 1);
//! assert_eq!(result, Some(HashSet::from([3, 4, 5])));
//! ```

use crate::transducer::{Algorithm, Transducer};
use libdictenstein::{DictionaryValue, MappedDictionary, MutableMappedDictionary};
use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

/// Trait for types that can aggregate multiple values into one.
///
/// This trait defines how to combine multiple values of the same type into a single
/// aggregated result. Different collection types implement this differently:
/// - `HashSet<T>`: Union of all sets
/// - `BTreeSet<T>`: Union of all sets
/// - `Vec<T>`: Concatenation of all vectors
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::cache::multimap::CollectionAggregate;
/// use std::collections::HashSet;
///
/// let sets = vec![
///     HashSet::from([1, 2]),
///     HashSet::from([2, 3]),
///     HashSet::from([3, 4]),
/// ];
///
/// let result = HashSet::aggregate(sets.into_iter());
/// assert_eq!(result, HashSet::from([1, 2, 3, 4]));
/// ```
pub trait CollectionAggregate: Sized {
    /// Aggregates multiple values into one.
    ///
    /// # Arguments
    ///
    /// - `values`: Iterator of values to aggregate
    ///
    /// # Returns
    ///
    /// Single aggregated value
    fn aggregate<I>(values: I) -> Self
    where
        I: Iterator<Item = Self>;
}

// HashSet: union of all sets
impl<T> CollectionAggregate for HashSet<T>
where
    T: Eq + Hash + Clone,
{
    fn aggregate<I>(values: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        // Optimization: Pre-allocate with size hint to reduce rehashing
        let mut values = values.peekable();

        // Estimate capacity from first set's size
        let initial_capacity = values
            .peek()
            .map(|first_set| first_set.len() * 2) // Heuristic: 2x first set size
            .unwrap_or(0);

        let mut acc = HashSet::with_capacity(initial_capacity);

        for set in values {
            // Reserve additional capacity if needed
            if acc.len() + set.len() > acc.capacity() {
                acc.reserve(set.len());
            }
            acc.extend(set);
        }

        acc
    }
}

// BTreeSet: union of all sets
impl<T> CollectionAggregate for BTreeSet<T>
where
    T: Ord + Clone,
{
    fn aggregate<I>(values: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        values.fold(BTreeSet::new(), |mut acc, set| {
            acc.extend(set);
            acc
        })
    }
}

// Vec: concatenation
impl<T> CollectionAggregate for Vec<T>
where
    T: Clone,
{
    fn aggregate<I>(values: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        // Optimization: Pre-allocate capacity based on size hints
        let mut values = values.peekable();

        // Estimate capacity from first vec's size
        let initial_capacity = values
            .peek()
            .map(|first_vec| first_vec.len() * 2) // Heuristic: 2x first vec size
            .unwrap_or(0);

        let mut acc = Vec::with_capacity(initial_capacity);

        for vec in values {
            // Reserve additional capacity if needed
            acc.reserve(vec.len());
            acc.extend(vec);
        }

        acc
    }
}

/// FuzzyMultiMap provides value aggregation from fuzzy-matched keys.
///
/// This is the core feature requested by the user - it finds all keys within
/// a given edit distance and aggregates their values using collection-specific logic.
///
/// # Type Parameters
///
/// - `C`: Collection type for values (must implement `CollectionAggregate`)
/// - `D`: Dictionary type (must implement `MappedDictionary<Value = C>`)
///
/// # Examples
///
/// ## HashSet Union
///
/// ```rust,ignore
/// use std::collections::HashSet;
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::cache::multimap::FuzzyMultiMap;
///
/// let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
/// dict.insert_with_value("hello", HashSet::from([1, 2]));
/// dict.insert_with_value("hallo", HashSet::from([3]));
/// dict.insert_with_value("hullo", HashSet::from([4, 5]));
///
/// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
///
/// // Find all values for terms within distance 1 of "hllo"
/// let result = fuzzy.query("hllo", 1).unwrap();
/// // Matches: "hello", "hallo", "hullo" all within distance 1
/// // Result: Union of {1,2}, {3}, {4,5}
/// ```
///
/// ## Vec Concatenation
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::cache::multimap::FuzzyMultiMap;
///
/// let dict: DynamicDawgChar<Vec<i32>> = DynamicDawgChar::new();
/// dict.insert_with_value("foo", vec![1, 2]);
/// dict.insert_with_value("fob", vec![3]);
/// dict.insert_with_value("fog", vec![4, 5]);
///
/// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
///
/// // Concatenates vectors from all matches
/// let result = fuzzy.query("foe", 1).unwrap();
/// ```
pub struct FuzzyMultiMap<C, D>
where
    D: crate::dictionary::Dictionary,
{
    dictionary: D,
    transducer: Transducer<D>,
    _phantom: std::marker::PhantomData<C>,
}

impl<C, D> FuzzyMultiMap<C, D>
where
    C: CollectionAggregate + DictionaryValue,
    D: MappedDictionary<Value = C> + crate::dictionary::Dictionary + Clone,
{
    /// Creates a new FuzzyMultiMap.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The fuzzy dictionary to query
    /// - `algorithm`: Levenshtein algorithm variant (Standard, Transposition, or MergeAndSplit)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
    /// dict.insert_with_value("foo", HashSet::from([1]));
    /// dict.insert_with_value("bar", HashSet::from([2]));
    ///
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    /// ```
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self {
        let transducer = Transducer::new(dictionary.clone(), algorithm);
        Self {
            dictionary,
            transducer,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Queries for fuzzy matches and aggregates their values.
    ///
    /// This is the core operation: find all keys within `max_distance` of `query_term`
    /// and aggregate their values using the collection's aggregation logic.
    ///
    /// # Arguments
    ///
    /// - `query_term`: The term to search for
    /// - `max_distance`: Maximum Levenshtein distance for matches
    ///
    /// # Returns
    ///
    /// - `Some(C)`: Aggregated values from all matches
    /// - `None`: No matches found
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
    /// dict.insert_with_value("foo", HashSet::from([1, 2]));
    /// dict.insert_with_value("bar", HashSet::from([3]));
    /// dict.insert_with_value("baz", HashSet::from([4, 5]));
    ///
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    ///
    /// // Query "bat" - matches "bar" and "baz" at distance 1
    /// let result = fuzzy.query("bat", 1).unwrap();
    /// assert_eq!(result, HashSet::from([3, 4, 5]));
    /// ```
    pub fn query(&self, query_term: &str, max_distance: usize) -> Option<C> {
        if max_distance == 0 {
            return self.dictionary.get_value(query_term);
        }

        // Optimization: Single-pass collection - avoid double Vec allocation
        // Instead of collecting candidates then mapping to values, we fuse the operations
        let mut values = self
            .transducer
            .query(query_term, max_distance)
            .filter_map(|term| self.dictionary.get_value(&term))
            .peekable();

        // Check if we have any results before aggregating
        values.peek()?;

        // Aggregate using collection-specific logic
        Some(C::aggregate(values))
    }

    /// Gets the inner dictionary reference.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
    /// dict.insert_with_value("foo", HashSet::from([1]));
    ///
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    /// assert_eq!(fuzzy.dictionary().len(), Some(1));
    /// ```
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Gets the algorithm being used.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict: DynamicDawgChar<HashSet<i32>> = DynamicDawgChar::new();
    /// dict.insert_with_value("foo", HashSet::from([1]));
    ///
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Transposition);
    /// assert_eq!(fuzzy.algorithm(), Algorithm::Transposition);
    /// ```
    pub fn algorithm(&self) -> Algorithm {
        self.transducer.algorithm()
    }

    /// Queries for fuzzy matches and returns (matched_key, distance, values) tuples.
    ///
    /// Unlike [`Self::query`], this method preserves the matched key and its edit distance,
    /// enabling applications that need to distinguish between different matches
    /// (e.g., PhoneticNormalizedDictionary for mapping back to original terms).
    ///
    /// # Arguments
    ///
    /// - `query_term`: The term to search for
    /// - `max_distance`: Maximum Levenshtein distance for matches
    ///
    /// # Returns
    ///
    /// A vector of `(matched_key, distance, values)` tuples, where:
    /// - `matched_key`: The dictionary key that matched
    /// - `distance`: The edit distance between query and matched key
    /// - `values`: The values associated with the matched key
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict: DynamicDawgChar<Vec<String>> = DynamicDawgChar::new();
    /// dict.insert_with_value("foo", vec!["original_foo".to_string()]);
    /// dict.insert_with_value("bar", vec!["original_bar".to_string()]);
    /// dict.insert_with_value("baz", vec!["original_baz".to_string()]);
    ///
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    ///
    /// // Query "bat" - matches "bar" and "baz" at distance 1
    /// let results = fuzzy.query_with_distance("bat", 1);
    /// for (key, distance, values) in results {
    ///     println!("Matched '{}' (distance {}): {:?}", key, distance, values);
    /// }
    /// ```
    pub fn query_with_distance(
        &self,
        query_term: &str,
        max_distance: usize,
    ) -> Vec<(String, usize, C)> {
        if max_distance == 0 {
            return self
                .dictionary
                .get_value(query_term)
                .map(|value| vec![(query_term.to_string(), 0, value)])
                .unwrap_or_default();
        }

        self.transducer
            .query_with_distance(query_term, max_distance)
            .filter_map(|candidate| {
                self.dictionary
                    .get_value(&candidate.term)
                    .map(|value| (candidate.term, candidate.distance, value))
            })
            .collect()
    }
}

// ============================================================================
// Mutation Methods (for MutableMappedDictionary types)
// ============================================================================

impl<C, D> FuzzyMultiMap<C, D>
where
    C: CollectionAggregate + DictionaryValue,
    D: MutableMappedDictionary<Value = C> + crate::dictionary::Dictionary + Clone,
{
    /// Insert a term with the given value, or replace if it exists.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    ///
    /// # Arguments
    ///
    /// - `term`: The term to insert
    /// - `value`: The value to associate with the term
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict = DynamicDawgChar::<HashSet<u32>>::new();
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    ///
    /// let was_new = fuzzy.insert("foo", HashSet::from([1, 2]));
    /// assert!(was_new);
    ///
    /// let was_new = fuzzy.insert("foo", HashSet::from([3]));
    /// assert!(!was_new); // Already existed
    /// ```
    pub fn insert(&self, term: &str, value: C) -> bool {
        self.dictionary.insert_with_value(term, value)
    }

    /// Update an existing term's value or insert with a default value.
    ///
    /// This is useful for accumulation patterns where you want to add to an existing
    /// collection (e.g., add an element to a `HashSet`) rather than replace it.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    ///
    /// # Arguments
    ///
    /// - `term`: The term to update or insert
    /// - `default_value`: The value to use if the term doesn't exist
    /// - `update_fn`: Function to apply to the existing value if the term exists
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::multimap::FuzzyMultiMap;
    ///
    /// let dict = DynamicDawgChar::<HashSet<String>>::new();
    /// let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);
    ///
    /// // First insert: creates new entry
    /// let was_new = fuzzy.update_or_insert(
    ///     "normalized_form",
    ///     HashSet::from(["original1".to_string()]),
    ///     |set| { set.insert("original1".to_string()); }
    /// );
    /// assert!(was_new);
    ///
    /// // Second insert: adds to existing set
    /// let was_new = fuzzy.update_or_insert(
    ///     "normalized_form",
    ///     HashSet::new(),
    ///     |set| { set.insert("original2".to_string()); }
    /// );
    /// assert!(!was_new);
    ///
    /// // Now "normalized_form" maps to {"original1", "original2"}
    /// ```
    pub fn update_or_insert<F>(&self, term: &str, default_value: C, update_fn: F) -> bool
    where
        F: Fn(&mut C),
    {
        self.dictionary
            .update_or_insert(term, default_value, update_fn)
    }
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_hashset_union() {
        // User's original example
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", HashSet::from([1, 2])),
            ("bar", HashSet::from([3])),
            ("baz", HashSet::from([4, 5])),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // Query "bat" with distance 1
        // Should match "bar" and "baz"
        let result = fuzzy.query("bat", 1).expect("expected Some result in test");
        assert_eq!(result, HashSet::from([3, 4, 5]));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_vec_concatenation() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", vec![1, 2]),
            ("fob", vec![3]),
            ("fog", vec![4, 5]),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // Query "foo" with distance 0 - exact match only
        let result = fuzzy.query("foo", 0).expect("expected Some result in test");
        assert_eq!(result, vec![1, 2]);

        // Query "fox" with distance 1
        // Should match "fob" and "fog"
        let result = fuzzy.query("fox", 1).expect("expected Some result in test");
        // Vec concatenation maintains order of matches
        assert!(result.contains(&3));
        assert!(result.contains(&4));
        assert!(result.contains(&5));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_no_matches() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", HashSet::from([1])),
            ("bar", HashSet::from([2])),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // Query with no matches
        let result = fuzzy.query("xyz", 1);
        assert!(result.is_none());
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_exact_match() {
        let dict = PathMapDictionary::from_terms_with_values([("hello", HashSet::from([1, 2, 3]))]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        let result = fuzzy
            .query("hello", 0)
            .expect("expected Some result in test");
        assert_eq!(result, HashSet::from([1, 2, 3]));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_exact_distance_zero_miss() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("hello", HashSet::from([1, 2, 3])),
            ("hallo", HashSet::from([4])),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        assert_eq!(fuzzy.query("hullo", 0), None);
        assert!(fuzzy.query_with_distance("hullo", 0).is_empty());
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_overlapping_values() {
        // Test that HashSet properly handles duplicate values
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", HashSet::from([1, 2])),
            ("foe", HashSet::from([2, 3])),
            ("fog", HashSet::from([3, 4])),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // All three match "fox" at distance 1
        let result = fuzzy.query("fox", 1).expect("expected Some result in test");
        // Should be union: {1, 2, 3, 4}
        assert_eq!(result, HashSet::from([1, 2, 3, 4]));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_with_transposition() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("hello", HashSet::from([1])),
            ("ehllo", HashSet::from([2])), // Adjacent transposition of "hello"
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Transposition);

        // Query with transposition distance
        let result = fuzzy
            .query("hello", 2)
            .expect("expected Some result in test");
        // Should at least match "hello" (exact)
        assert!(result.contains(&1));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_query_with_distance() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("foo", vec!["original_foo".to_string()]),
            ("bar", vec!["original_bar".to_string()]),
            ("baz", vec!["original_baz".to_string()]),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // Query "bat" with distance 1
        // Should match "bar" and "baz"
        let results = fuzzy.query_with_distance("bat", 1);

        // Should have 2 results
        assert_eq!(results.len(), 2);

        // Check that we get both matches with their distances
        let bar_result = results.iter().find(|(key, _, _)| key == "bar");
        assert!(bar_result.is_some());
        let (_, distance, values) = bar_result.expect("expected Some bar_result in test");
        assert_eq!(*distance, 1);
        assert_eq!(values, &vec!["original_bar".to_string()]);

        let baz_result = results.iter().find(|(key, _, _)| key == "baz");
        assert!(baz_result.is_some());
        let (_, distance, values) = baz_result.expect("expected Some baz_result in test");
        assert_eq!(*distance, 1);
        assert_eq!(values, &vec!["original_baz".to_string()]);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_query_with_distance_exact() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("test", vec!["exact_match".to_string()]),
            ("tost", vec!["near_match".to_string()]),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        // Exact match should have distance 0
        let results = fuzzy.query_with_distance("test", 1);

        let exact = results.iter().find(|(key, _, _)| key == "test");
        assert!(exact.is_some());
        let (_, distance, _) = exact.expect("expected Some exact in test");
        assert_eq!(*distance, 0);

        let near = results.iter().find(|(key, _, _)| key == "tost");
        assert!(near.is_some());
        let (_, distance, _) = near.expect("expected Some near in test");
        assert_eq!(*distance, 1);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_fuzzy_multimap_query_with_distance_exact_distance_zero() {
        let dict = PathMapDictionary::from_terms_with_values([
            ("test", vec!["exact_match".to_string()]),
            ("tost", vec!["near_match".to_string()]),
        ]);

        let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

        let results = fuzzy.query_with_distance("test", 0);
        assert_eq!(
            results,
            vec![("test".to_string(), 0, vec!["exact_match".to_string()])]
        );
    }
}
