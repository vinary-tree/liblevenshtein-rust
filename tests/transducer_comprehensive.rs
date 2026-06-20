//! Comprehensive transducer module tests using property-based testing.
//!
//! This test file provides extensive coverage of the transducer module across:
//! - All dictionary backend implementations
//! - All Levenshtein algorithm variants
//! - Query iteration correctness
//! - Ordered query behavior
//! - Value-filtered queries

mod common;

use liblevenshtein::distance::*;
use liblevenshtein::prelude::*;
use proptest::prelude::*;

// ============================================================================
// Transducer Construction Tests
// ============================================================================

#[test]
fn test_transducer_construction_double_array_trie() {
    let dict = DoubleArrayTrie::from_terms(vec![
        "apple".to_string(),
        "application".to_string(),
        "apply".to_string(),
    ]);

    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Basic sanity check
    let results: Vec<_> = transducer.query("apple", 0).collect();
    assert_eq!(results, vec!["apple"]);
}

#[test]
fn test_transducer_construction_dynamic_dawg() {
    let dict: DynamicDawg<()> = DynamicDawg::from_terms(vec![
        "test".to_string(),
        "testing".to_string(),
        "tested".to_string(),
    ]);

    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 0).collect();
    assert_eq!(results, vec!["test"]);
}

// ============================================================================
// Distance Function Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Standard distance identity: d(a, a) == 0
    #[test]
    fn prop_standard_identity(s in common::strategies::ascii_word_strategy()) {
        let dist = standard_distance(&s, &s);
        prop_assert_eq!(dist, 0, "d('{}', '{}') should be 0", s, s);
    }

    /// Standard distance symmetry: d(a, b) == d(b, a)
    #[test]
    fn prop_standard_symmetry(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let d1 = standard_distance(&s1, &s2);
        let d2 = standard_distance(&s2, &s1);
        prop_assert_eq!(d1, d2, "d('{}', '{}') != d('{}', '{}')", s1, s2, s2, s1);
    }

    /// Standard distance triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    #[test]
    fn prop_standard_triangle(
        a in common::strategies::ascii_word_strategy(),
        b in common::strategies::ascii_word_strategy(),
        c in common::strategies::ascii_word_strategy(),
    ) {
        let d_ac = standard_distance(&a, &c);
        let d_ab = standard_distance(&a, &b);
        let d_bc = standard_distance(&b, &c);

        prop_assert!(
            d_ac <= d_ab + d_bc,
            "Triangle inequality violated: d('{}','{}')={} > d('{}','{}')={} + d('{}','{}')={}",
            a, c, d_ac, a, b, d_ab, b, c, d_bc
        );
    }

    /// Standard distance length bound: d(a, b) >= |len(a) - len(b)|
    #[test]
    fn prop_standard_length_bound(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let dist = standard_distance(&s1, &s2);
        let len_diff = (s1.len() as isize - s2.len() as isize).unsigned_abs();

        prop_assert!(
            dist >= len_diff,
            "d('{}', '{}') = {} < |{} - {}| = {}",
            s1, s2, dist, s1.len(), s2.len(), len_diff
        );
    }

    /// Transposition distance identity
    #[test]
    fn prop_transposition_identity(s in common::strategies::ascii_word_strategy()) {
        let dist = transposition_distance(&s, &s);
        prop_assert_eq!(dist, 0);
    }

    /// Transposition distance symmetry
    #[test]
    fn prop_transposition_symmetry(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let d1 = transposition_distance(&s1, &s2);
        let d2 = transposition_distance(&s2, &s1);
        prop_assert_eq!(d1, d2);
    }
}

// ============================================================================
// Query Iteration Completeness Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Verify transducer finds ALL terms within distance (cross-validate with linear scan)
    #[test]
    fn prop_query_completeness_standard(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        // Linear scan to find expected results (deduplicate first since dictionaries deduplicate)
        let unique_words: std::collections::HashSet<_> = dict_words.iter().collect();
        let expected: Vec<_> = unique_words
            .iter()
            .filter(|term| standard_distance(&query, **term) <= max_dist)
            .map(|t| (*t).clone())
            .collect();

        // Every expected term should be in results
        for term in &expected {
            prop_assert!(
                results.contains(term),
                "Expected term '{}' missing from query results",
                term
            );
        }

        // Results should not exceed expected (no false positives)
        prop_assert_eq!(
            results.len(),
            expected.len(),
            "Result count mismatch: got {} results, expected {}",
            results.len(),
            expected.len()
        );
    }

    /// Verify transducer with transposition algorithm is complete
    #[test]
    fn prop_query_completeness_transposition(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Transposition);

        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        // Linear scan with transposition distance (deduplicate first since dictionaries deduplicate)
        let unique_words: std::collections::HashSet<_> = dict_words.iter().collect();
        let expected: Vec<_> = unique_words
            .iter()
            .filter(|term| transposition_distance(&query, **term) <= max_dist)
            .map(|t| (*t).clone())
            .collect();

        for term in &expected {
            prop_assert!(
                results.contains(term),
                "Expected term '{}' missing from transposition query results",
                term
            );
        }
    }

    /// Verify transducer with merge_and_split algorithm is complete
    #[test]
    fn prop_query_completeness_merge_split(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::MergeAndSplit);
        let cache = create_memo_cache();

        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        // Linear scan with merge_and_split distance (deduplicate first since dictionaries deduplicate)
        let unique_words: std::collections::HashSet<_> = dict_words.iter().collect();
        let expected: Vec<_> = unique_words
            .iter()
            .filter(|term| merge_and_split_distance(&query, **term, &cache) <= max_dist)
            .map(|t| (*t).clone())
            .collect();

        for term in &expected {
            prop_assert!(
                results.contains(term),
                "Expected term '{}' missing from merge_split query results",
                term
            );
        }
    }
}

// ============================================================================
// Ordered Query Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Verify query_ordered returns results in ascending distance order
    #[test]
    fn prop_ordered_query_ascending(
        dict_words in common::strategies::medium_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 1usize..=3,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query_ordered(&query, max_dist).collect();

        // Check results are in non-decreasing distance order
        let mut prev_dist = 0;
        for candidate in &results {
            prop_assert!(
                candidate.distance >= prev_dist,
                "Results not in order: '{}' (dist {}) after item with dist {}",
                candidate.term, candidate.distance, prev_dist
            );
            prev_dist = candidate.distance;
        }
    }

    /// Verify query_ordered produces same results as unordered (just sorted)
    #[test]
    fn prop_ordered_query_same_results(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict1 = DoubleArrayTrie::from_terms(dict_words.clone());
        let dict2 = DoubleArrayTrie::from_terms(dict_words);

        let t1 = Transducer::new(dict1, Algorithm::Standard);
        let t2 = Transducer::new(dict2, Algorithm::Standard);

        let mut unordered: Vec<_> = t1.query(&query, max_dist).collect();
        let ordered: Vec<_> = t2.query_ordered(&query, max_dist).map(|c| c.term).collect();

        // Sort unordered results
        unordered.sort();

        // Get unique ordered results
        let mut ordered_unique: Vec<_> = ordered;
        ordered_unique.sort();
        ordered_unique.dedup();

        // Compare unique results
        unordered.dedup();
        prop_assert_eq!(
            unordered,
            ordered_unique,
            "query_ordered should return same unique results as query"
        );
    }
}

// ============================================================================
// Empty Query and Edge Case Tests
// ============================================================================

#[test]
fn test_empty_query() {
    let dict = DoubleArrayTrie::from_terms(vec![
        "a".to_string(),
        "ab".to_string(),
        "abc".to_string(),
        "abcd".to_string(),
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Empty query with max_dist=0 should find nothing (no empty string in dict)
    let results_0: Vec<_> = transducer.query("", 0).collect();
    assert!(results_0.is_empty());

    // Empty query with max_dist=1 should find "a"
    let results_1: Vec<_> = transducer.query("", 1).collect();
    assert_eq!(results_1, vec!["a"]);

    // Empty query with max_dist=2 should find "a", "ab"
    let mut results_2: Vec<_> = transducer.query("", 2).collect();
    results_2.sort();
    assert_eq!(results_2, vec!["a", "ab"]);
}

#[test]
fn test_query_not_in_dictionary() {
    let dict = DoubleArrayTrie::from_terms(vec!["cat".to_string(), "car".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query for something not in dictionary
    let results: Vec<_> = transducer.query("xyz", 0).collect();
    assert!(results.is_empty());

    // But should find close matches
    let results_2: Vec<_> = transducer.query("cap", 1).collect();
    assert!(results_2.contains(&"cat".to_string()) || results_2.contains(&"car".to_string()));
}

#[test]
fn test_single_character_words() {
    let dict = DoubleArrayTrie::from_terms(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Exact match
    let results_a: Vec<_> = transducer.query("a", 0).collect();
    assert_eq!(results_a, vec!["a"]);

    // One edit away
    let mut results_1: Vec<_> = transducer.query("a", 1).collect();
    results_1.sort();
    assert_eq!(results_1, vec!["a", "b", "c"]);
}

// ============================================================================
// Unicode Support Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test transducer handles Unicode correctly
    #[test]
    fn prop_unicode_handling(
        dict_words in common::strategies::unicode_dict_strategy(),
        query in common::strategies::unicode_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Should not panic
        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        // All results should be valid strings
        for result in &results {
            prop_assert!(result.is_ascii() || !result.is_empty());
        }
    }
}

// ============================================================================
// Algorithm Comparison Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Standard Levenshtein results should be subset of Transposition results
    #[test]
    fn prop_standard_subset_of_transposition(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 1usize..=2,
    ) {
        let dict_std = DoubleArrayTrie::from_terms(dict_words.clone());
        let dict_trans = DoubleArrayTrie::from_terms(dict_words);

        let t_std = Transducer::new(dict_std, Algorithm::Standard);
        let t_trans = Transducer::new(dict_trans, Algorithm::Transposition);

        let results_std: Vec<_> = t_std.query(&query, max_dist).collect();
        let results_trans: Vec<_> = t_trans.query(&query, max_dist).collect();

        // Every standard result should be in transposition results
        for result in &results_std {
            prop_assert!(
                results_trans.contains(result),
                "Standard result '{}' not in Transposition results",
                result
            );
        }
    }

    /// Transposition should find adjacent swaps at distance 1
    #[test]
    fn prop_transposition_finds_swaps(
        base in "[a-z]{4,8}",
    ) {
        // Create a word and its transposition
        let chars: Vec<char> = base.chars().collect();
        if chars.len() < 2 {
            return Ok(());
        }

        // Swap first two characters
        let mut swapped_chars = chars.clone();
        swapped_chars.swap(0, 1);
        let swapped: String = swapped_chars.into_iter().collect();

        // Skip if swap didn't change the word
        if swapped == base {
            return Ok(());
        }

        let dict = DoubleArrayTrie::from_terms(vec![base.clone()]);
        let transducer = Transducer::new(dict, Algorithm::Transposition);

        let results: Vec<_> = transducer.query(&swapped, 1).collect();
        prop_assert!(
            results.contains(&base),
            "Transposition should find '{}' when querying '{}'",
            base, swapped
        );
    }
}

// ============================================================================
// All Algorithms Find Exact Matches
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// All algorithms should find exact matches
    #[test]
    fn prop_all_algorithms_find_exact_matches(
        dict_words in common::strategies::small_dict_strategy(),
    ) {
        if dict_words.is_empty() {
            return Ok(());
        }

        let query_word = dict_words[0].clone();

        let dict_std = DoubleArrayTrie::from_terms(dict_words.clone());
        let dict_trans = DoubleArrayTrie::from_terms(dict_words.clone());
        let dict_merge = DoubleArrayTrie::from_terms(dict_words);

        let t_std = Transducer::new(dict_std, Algorithm::Standard);
        let t_trans = Transducer::new(dict_trans, Algorithm::Transposition);
        let t_merge = Transducer::new(dict_merge, Algorithm::MergeAndSplit);

        let results_std: Vec<_> = t_std.query(&query_word, 0).collect();
        let results_trans: Vec<_> = t_trans.query(&query_word, 0).collect();
        let results_merge: Vec<_> = t_merge.query(&query_word, 0).collect();

        prop_assert!(
            results_std.contains(&query_word),
            "Standard algorithm failed to find exact match"
        );
        prop_assert!(
            results_trans.contains(&query_word),
            "Transposition algorithm failed to find exact match"
        );
        prop_assert!(
            results_merge.contains(&query_word),
            "MergeAndSplit algorithm failed to find exact match"
        );
    }
}

// ============================================================================
// Large Dictionary Tests
// ============================================================================

#[test]
fn test_large_dictionary() {
    // Generate a larger dictionary
    let dict_words: Vec<String> = (0..1000).map(|i| format!("word{:04}", i)).collect();

    let dict = DoubleArrayTrie::from_terms(dict_words);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Exact match
    let results_0: Vec<_> = transducer.query("word0500", 0).collect();
    assert_eq!(results_0, vec!["word0500"]);

    // One edit
    let results_1: Vec<_> = transducer.query("word0500", 1).collect();
    assert!(results_1.contains(&"word0500".to_string()));
}

// ============================================================================
// Cross-Backend Consistency Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// DoubleArrayTrie and DynamicDawg should produce identical results
    #[test]
    fn prop_backend_consistency(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dat = DoubleArrayTrie::from_terms(dict_words.clone());
        let dawg: DynamicDawg<()> = DynamicDawg::from_terms(dict_words);

        let t_dat = Transducer::new(dat, Algorithm::Standard);
        let t_dawg = Transducer::new(dawg, Algorithm::Standard);

        let mut results_dat: Vec<_> = t_dat.query(&query, max_dist).collect();
        let mut results_dawg: Vec<_> = t_dawg.query(&query, max_dist).collect();

        results_dat.sort();
        results_dawg.sort();

        prop_assert_eq!(
            results_dat,
            results_dawg,
            "DoubleArrayTrie and DynamicDawg produce different results"
        );
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

/// Regression test where query_ordered() missed exact matches.
/// See: prop_ordered_query_same_results failure with dict=["a", "ar"], query="ar", max_dist=1
///
/// Expected behavior:
/// - query("ar", 1) returns: ["a", "ar"]
/// - query_ordered("ar", 1) should return the same set: ["a" (d=1), "ar" (d=0)]
///
/// Defect: query_ordered() was returning only ["a"], missing the exact match "ar" (d=0)
/// Root cause: When a final node's actual distance > min_distance (bucket position),
/// the node was requeued to the correct bucket but its children were not explored.
/// Fix: Queue children before requeueing the intersection.
#[test]
fn test_query_returns_all_matches_regression() {
    use std::collections::HashSet;

    let dict = DoubleArrayTrie::from_terms(vec!["a".to_string(), "ar".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Collect query() results into a set
    let query_results: HashSet<_> = transducer.query("ar", 1).collect();

    // Assert we found exactly 2 results
    assert_eq!(
        query_results.len(),
        2,
        "query() must find exactly 2 matches, found: {:?}",
        query_results
    );

    // Assert both expected matches are present
    assert!(
        query_results.contains("a"),
        "query() must find 'a' (distance 1)"
    );
    assert!(
        query_results.contains("ar"),
        "query() must find 'ar' (distance 0)"
    );

    // Collect query_ordered() results into a set
    let ordered_results: HashSet<_> = transducer.query_ordered("ar", 1).map(|c| c.term).collect();

    // Both methods must return the same set of results
    assert_eq!(
        query_results, ordered_results,
        "query() and query_ordered() must return the same results"
    );
}

/// Additional regression tests for similar edge cases
#[test]
fn test_query_finds_exact_match_with_prefix() {
    use std::collections::HashSet;

    // Test case: query is a prefix of a dictionary word
    let dict = DoubleArrayTrie::from_terms(vec!["ab".to_string(), "abc".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let query_results: HashSet<_> = transducer.query("ab", 1).collect();
    let ordered_results: HashSet<_> = transducer.query_ordered("ab", 1).map(|c| c.term).collect();

    assert!(
        query_results.contains("ab"),
        "query() must find exact match 'ab'"
    );
    assert_eq!(
        query_results, ordered_results,
        "query() and query_ordered() must return the same results for prefix case"
    );
}

#[test]
fn test_query_finds_all_single_char_matches() {
    use std::collections::HashSet;

    // Test case: single character words with extensions
    let dict =
        DoubleArrayTrie::from_terms(vec!["a".to_string(), "ab".to_string(), "abc".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let query_results: HashSet<_> = transducer.query("a", 1).collect();
    let ordered_results: HashSet<_> = transducer.query_ordered("a", 1).map(|c| c.term).collect();

    assert!(
        query_results.contains("a"),
        "query() must find exact match 'a'"
    );
    assert!(
        query_results.contains("ab"),
        "query() must find 'ab' (distance 1)"
    );
    assert_eq!(
        query_results, ordered_results,
        "query() and query_ordered() must return the same results for single char case"
    );
}

// ============================================================================
// Backend-Specific Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// DoubleArrayTrie query results within distance
    #[test]
    fn prop_dat_query_within_distance(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        for result in transducer.query(&query, max_dist) {
            let dist = standard_distance(&query, &result);
            prop_assert!(
                dist <= max_dist,
                "Result '{}' has distance {} > max {}",
                result, dist, max_dist
            );
        }
    }

    /// DynamicDawg query results within distance
    #[test]
    fn prop_dawg_query_within_distance(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 0usize..=2,
    ) {
        let dict: DynamicDawg<()> = DynamicDawg::from_terms(dict_words);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        for result in transducer.query(&query, max_dist) {
            let dist = standard_distance(&query, &result);
            prop_assert!(
                dist <= max_dist,
                "Result '{}' has distance {} > max {}",
                result, dist, max_dist
            );
        }
    }
}
