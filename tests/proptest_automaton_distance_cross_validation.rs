//! Cross-validation property-based tests between Levenshtein automaton and distance functions.
//!
//! These tests verify that the Levenshtein automaton returns exactly the same results
//! as a linear scan over the dictionary using the corresponding distance function.
//! This is the ultimate correctness test - both implementations must agree on all results.
//!
//! For each algorithm (Standard, Transposition, MergeAndSplit), we test that:
//! 1. Automaton results match linear scan results (no false positives, no false negatives)
//! 2. Distance computations by automaton match distance function results
//! 3. Both implementations handle edge cases consistently

use liblevenshtein::distance::*;
use liblevenshtein::prelude::*;
use proptest::prelude::*;
use std::collections::HashSet;

// ============================================================================
// Test Data Generators
// ============================================================================

/// Strategy for generating simple ASCII words (easier to debug)
fn ascii_word_strategy() -> impl Strategy<Value = String> {
    "[a-z]{0,15}"
}

/// Strategy for generating Unicode words (comprehensive testing)
fn unicode_word_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        any::<char>().prop_filter("Valid unicode", |c| !c.is_control()),
        0..15,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

/// Strategy for generating a small dictionary (for quick tests)
fn small_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 1..=20)
}

/// Strategy for generating a medium dictionary (for realistic tests)
fn medium_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 20..=100)
}

/// Strategy for edit distance values
fn distance_strategy() -> impl Strategy<Value = usize> {
    0usize..=3
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Perform linear scan using the appropriate distance function.
/// Returns all dictionary words within max_distance of query.
fn linear_scan_standard(
    dict_words: &[String],
    query: &str,
    max_distance: usize,
) -> HashSet<String> {
    dict_words
        .iter()
        .filter(|word| standard_distance(query, word) <= max_distance)
        .cloned()
        .collect()
}

/// Linear scan for Transposition algorithm
fn linear_scan_transposition(
    dict_words: &[String],
    query: &str,
    max_distance: usize,
) -> HashSet<String> {
    dict_words
        .iter()
        .filter(|word| transposition_distance(query, word) <= max_distance)
        .cloned()
        .collect()
}

/// Linear scan for MergeAndSplit algorithm
fn linear_scan_merge_split(
    dict_words: &[String],
    query: &str,
    max_distance: usize,
) -> HashSet<String> {
    let cache = create_memo_cache();
    dict_words
        .iter()
        .filter(|word| merge_and_split_distance(query, word, &cache) <= max_distance)
        .cloned()
        .collect()
}

/// Query automaton and collect results into HashSet
fn automaton_query(
    dict_words: &[String],
    query: &str,
    max_distance: usize,
    algorithm: Algorithm,
) -> HashSet<String> {
    let dict = DoubleArrayTrie::from_terms(dict_words);
    let transducer = Transducer::new(dict, algorithm);
    transducer.query(query, max_distance).collect()
}

// ============================================================================
// Standard Algorithm Cross-Validation Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// **Critical Test**: Automaton results must exactly match linear scan (Standard algorithm)
    ///
    /// This test verifies that the Levenshtein automaton produces identical results
    /// to a brute-force linear scan using the standard_distance function.
    #[test]
    fn prop_standard_automaton_matches_linear_scan(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        // Get results from both methods
        let linear_results = linear_scan_standard(&dict_words, &query, max_dist);
        let automaton_results = automaton_query(&dict_words, &query, max_dist, Algorithm::Standard);

        // Check for false positives (automaton returned word that shouldn't match)
        for word in &automaton_results {
            let actual_distance = standard_distance(&query, word);
            prop_assert!(
                linear_results.contains(word),
                "FALSE POSITIVE: Automaton returned '{}' for query '{}' with max_distance {}, \
                 but linear scan didn't find it. Actual distance: {}. \
                 Dictionary: {:?}",
                word, query, max_dist, actual_distance, dict_words
            );
        }

        // Check for false negatives (automaton missed word that should match)
        for word in &linear_results {
            let actual_distance = standard_distance(&query, word);
            prop_assert!(
                automaton_results.contains(word),
                "FALSE NEGATIVE: Linear scan found '{}' for query '{}' with max_distance {}, \
                 but automaton didn't return it. Actual distance: {}. \
                 Dictionary: {:?}, Automaton results: {:?}",
                word, query, max_dist, actual_distance, dict_words, automaton_results
            );
        }

        // Both should have same number of results
        prop_assert_eq!(
            automaton_results.len(),
            linear_results.len(),
            "Result count mismatch for query '{}' with max_distance {}. \
             Automaton: {} results, Linear: {} results. \
             Dictionary: {:?}",
            query, max_dist, automaton_results.len(), linear_results.len(), dict_words
        );
    }

    /// Verify that automaton distance values match distance function (Standard)
    #[test]
    fn prop_standard_automaton_distance_matches_function(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Query with distances
        for candidate in transducer.query_with_distance(&query, max_dist) {
            let computed_distance = standard_distance(&query, &candidate.term);
            prop_assert_eq!(
                candidate.distance, computed_distance,
                "Distance mismatch for query '{}' -> '{}'. \
                 Automaton reported: {}, distance function computed: {}. \
                 Dictionary: {:?}",
                query, candidate.term, candidate.distance, computed_distance, dict_words
            );
        }
    }

    /// Test with larger dictionaries (Standard algorithm)
    #[test]
    fn prop_standard_large_dict_matches(
        dict_words in medium_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in 0usize..=2  // Lower max distance for larger dicts
    ) {
        let linear_results = linear_scan_standard(&dict_words, &query, max_dist);
        let automaton_results = automaton_query(&dict_words, &query, max_dist, Algorithm::Standard);

        prop_assert_eq!(
            automaton_results,
            linear_results,
            "Large dictionary mismatch for query '{}' with max_distance {}. \
             Dictionary size: {}",
            query, max_dist, dict_words.len()
        );
    }

    /// Test with Unicode strings (Standard algorithm)
    ///
    /// Note: Uses DoubleArrayTrieChar for character-level matching to match
    /// the character-level distance function behavior.
    #[test]
    fn prop_standard_unicode_matches(
        dict_words in prop::collection::vec(unicode_word_strategy(), 1..=10),
        query in unicode_word_strategy(),
        max_dist in 0usize..=2
    ) {
        use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

        let linear_results = linear_scan_standard(&dict_words, &query, max_dist);

        // Use character-level dictionary for Unicode to match character-level distance function
        let dict = DoubleArrayTrieChar::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let automaton_results: HashSet<String> = transducer.query(&query, max_dist).collect();

        prop_assert_eq!(
            automaton_results,
            linear_results,
            "Unicode string mismatch for query '{}' with max_distance {}",
            query, max_dist
        );
    }
}

// ============================================================================
// Transposition Algorithm Cross-Validation Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// **Critical Test**: Automaton results must exactly match linear scan (Transposition algorithm)
    #[test]
    fn prop_transposition_automaton_matches_linear_scan(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let linear_results = linear_scan_transposition(&dict_words, &query, max_dist);
        let automaton_results = automaton_query(&dict_words, &query, max_dist, Algorithm::Transposition);

        // Check for false positives
        for word in &automaton_results {
            let actual_distance = transposition_distance(&query, word);
            prop_assert!(
                linear_results.contains(word),
                "FALSE POSITIVE (Transposition): Automaton returned '{}' for query '{}' \
                 with max_distance {}, but linear scan didn't find it. Actual distance: {}. \
                 Dictionary: {:?}",
                word, query, max_dist, actual_distance, dict_words
            );
        }

        // Check for false negatives
        for word in &linear_results {
            let actual_distance = transposition_distance(&query, word);
            prop_assert!(
                automaton_results.contains(word),
                "FALSE NEGATIVE (Transposition): Linear scan found '{}' for query '{}' \
                 with max_distance {}, but automaton didn't return it. Actual distance: {}. \
                 Dictionary: {:?}, Automaton results: {:?}",
                word, query, max_dist, actual_distance, dict_words, automaton_results
            );
        }

        prop_assert_eq!(
            automaton_results.len(),
            linear_results.len(),
            "Result count mismatch (Transposition) for query '{}' with max_distance {}",
            query, max_dist
        );
    }

    /// Verify automaton distance values match distance function (Transposition)
    #[test]
    fn prop_transposition_automaton_distance_matches_function(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Transposition);

        for candidate in transducer.query_with_distance(&query, max_dist) {
            let computed_distance = transposition_distance(&query, &candidate.term);
            prop_assert_eq!(
                candidate.distance, computed_distance,
                "Distance mismatch (Transposition) for query '{}' -> '{}'. \
                 Automaton: {}, Function: {}",
                query, candidate.term, candidate.distance, computed_distance
            );
        }
    }

    /// Test transposition-specific cases (adjacent character swaps)
    #[test]
    fn prop_transposition_handles_swaps_correctly(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let linear_results = linear_scan_transposition(&dict_words, &query, max_dist);
        let automaton_results = automaton_query(&dict_words, &query, max_dist, Algorithm::Transposition);

        prop_assert_eq!(
            automaton_results,
            linear_results,
            "Transposition swap handling mismatch for query '{}'",
            query
        );
    }
}

// ============================================================================
// MergeAndSplit Algorithm Cross-Validation Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// **Critical Test**: Automaton results must exactly match linear scan (MergeAndSplit algorithm)
    #[test]
    fn prop_merge_split_automaton_matches_linear_scan(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let linear_results = linear_scan_merge_split(&dict_words, &query, max_dist);
        let automaton_results = automaton_query(&dict_words, &query, max_dist, Algorithm::MergeAndSplit);

        // Check for false positives
        for word in &automaton_results {
            let cache = create_memo_cache();
            let actual_distance = merge_and_split_distance(&query, word, &cache);
            prop_assert!(
                linear_results.contains(word),
                "FALSE POSITIVE (MergeAndSplit): Automaton returned '{}' for query '{}' \
                 with max_distance {}, but linear scan didn't find it. Actual distance: {}. \
                 Dictionary: {:?}",
                word, query, max_dist, actual_distance, dict_words
            );
        }

        // Check for false negatives
        for word in &linear_results {
            let cache = create_memo_cache();
            let actual_distance = merge_and_split_distance(&query, word, &cache);
            prop_assert!(
                automaton_results.contains(word),
                "FALSE NEGATIVE (MergeAndSplit): Linear scan found '{}' for query '{}' \
                 with max_distance {}, but automaton didn't return it. Actual distance: {}. \
                 Dictionary: {:?}, Automaton results: {:?}",
                word, query, max_dist, actual_distance, dict_words, automaton_results
            );
        }

        prop_assert_eq!(
            automaton_results.len(),
            linear_results.len(),
            "Result count mismatch (MergeAndSplit) for query '{}' with max_distance {}",
            query, max_dist
        );
    }

    /// Verify automaton distance values match distance function (MergeAndSplit)
    #[test]
    fn prop_merge_split_automaton_distance_matches_function(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::MergeAndSplit);
        let cache = create_memo_cache();

        for candidate in transducer.query_with_distance(&query, max_dist) {
            let computed_distance = merge_and_split_distance(&query, &candidate.term, &cache);
            prop_assert_eq!(
                candidate.distance, computed_distance,
                "Distance mismatch (MergeAndSplit) for query '{}' -> '{}'. \
                 Automaton: {}, Function: {}",
                query, candidate.term, candidate.distance, computed_distance
            );
        }
    }
}

// ============================================================================
// Edge Case Tests (All Algorithms)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Test with empty query string
    #[test]
    fn prop_empty_query_all_algorithms(
        dict_words in small_dict_strategy(),
        max_dist in distance_strategy()
    ) {
        let query = "";

        // Standard
        let linear_std = linear_scan_standard(&dict_words, query, max_dist);
        let auto_std = automaton_query(&dict_words, query, max_dist, Algorithm::Standard);
        prop_assert_eq!(auto_std, linear_std, "Empty query mismatch (Standard)");

        // Transposition
        let linear_trans = linear_scan_transposition(&dict_words, query, max_dist);
        let auto_trans = automaton_query(&dict_words, query, max_dist, Algorithm::Transposition);
        prop_assert_eq!(auto_trans, linear_trans, "Empty query mismatch (Transposition)");

        // MergeAndSplit
        let linear_ms = linear_scan_merge_split(&dict_words, query, max_dist);
        let auto_ms = automaton_query(&dict_words, query, max_dist, Algorithm::MergeAndSplit);
        prop_assert_eq!(auto_ms, linear_ms, "Empty query mismatch (MergeAndSplit)");
    }

    /// Test with empty dictionary
    #[test]
    fn prop_empty_dictionary_all_algorithms(
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        let dict_words: Vec<String> = vec![];

        // All algorithms should return empty results for empty dictionary
        let auto_std = automaton_query(&dict_words, &query, max_dist, Algorithm::Standard);
        prop_assert!(auto_std.is_empty(), "Empty dict should return no results (Standard)");

        let auto_trans = automaton_query(&dict_words, &query, max_dist, Algorithm::Transposition);
        prop_assert!(auto_trans.is_empty(), "Empty dict should return no results (Transposition)");

        let auto_ms = automaton_query(&dict_words, &query, max_dist, Algorithm::MergeAndSplit);
        prop_assert!(auto_ms.is_empty(), "Empty dict should return no results (MergeAndSplit)");
    }

    /// Test with duplicate words in dictionary
    #[test]
    fn prop_duplicate_words_all_algorithms(
        word in ascii_word_strategy(),
        query in ascii_word_strategy(),
        max_dist in distance_strategy()
    ) {
        // Dictionary with duplicates
        let dict_words = vec![word.clone(), word.clone(), word];

        // Standard
        let linear_std = linear_scan_standard(&dict_words, &query, max_dist);
        let auto_std = automaton_query(&dict_words, &query, max_dist, Algorithm::Standard);
        prop_assert_eq!(auto_std, linear_std, "Duplicate words mismatch (Standard)");

        // Transposition
        let linear_trans = linear_scan_transposition(&dict_words, &query, max_dist);
        let auto_trans = automaton_query(&dict_words, &query, max_dist, Algorithm::Transposition);
        prop_assert_eq!(auto_trans, linear_trans, "Duplicate words mismatch (Transposition)");

        // MergeAndSplit
        let linear_ms = linear_scan_merge_split(&dict_words, &query, max_dist);
        let auto_ms = automaton_query(&dict_words, &query, max_dist, Algorithm::MergeAndSplit);
        prop_assert_eq!(auto_ms, linear_ms, "Duplicate words mismatch (MergeAndSplit)");
    }

    /// Test with distance = 0 (exact match only)
    #[test]
    fn prop_exact_match_only_all_algorithms(
        dict_words in small_dict_strategy(),
        query in ascii_word_strategy()
    ) {
        let max_dist = 0;

        // Standard
        let linear_std = linear_scan_standard(&dict_words, &query, max_dist);
        let auto_std = automaton_query(&dict_words, &query, max_dist, Algorithm::Standard);
        prop_assert_eq!(auto_std, linear_std, "Exact match mismatch (Standard)");

        // Transposition
        let linear_trans = linear_scan_transposition(&dict_words, &query, max_dist);
        let auto_trans = automaton_query(&dict_words, &query, max_dist, Algorithm::Transposition);
        prop_assert_eq!(auto_trans, linear_trans, "Exact match mismatch (Transposition)");

        // MergeAndSplit
        let linear_ms = linear_scan_merge_split(&dict_words, &query, max_dist);
        let auto_ms = automaton_query(&dict_words, &query, max_dist, Algorithm::MergeAndSplit);
        prop_assert_eq!(auto_ms, linear_ms, "Exact match mismatch (MergeAndSplit)");
    }
}

// ============================================================================
// Regression Tests (Known Defect Cases)
// ============================================================================

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test for deletion defect (from proptest_levenshtein.rs)
    #[test]
    fn test_deletion_bug_cross_validation() {
        let dict_words = vec!["test".to_string(), "apple".to_string(), "world".to_string()];
        let query = "testt";
        let max_dist = 1;

        let linear_results = linear_scan_standard(&dict_words, query, max_dist);
        let automaton_results = automaton_query(&dict_words, query, max_dist, Algorithm::Standard);

        assert_eq!(
            automaton_results, linear_results,
            "Deletion regression: automaton should match linear scan"
        );
        assert!(
            automaton_results.iter().any(|term| term == "test"),
            "Should find 'test' for query 'testt' with distance 1"
        );
    }

    /// Regression for a leading delete followed by a match.
    ///
    /// This is the small executable case behind the Standard completeness
    /// antichain proof obligation: query "ab" should accept dictionary "b"
    /// at distance 1 by deleting the leading 'a' before matching 'b'.
    #[test]
    fn test_standard_leading_delete_then_match() {
        let dict_words = vec!["b".to_string()];
        let query = "ab";
        let max_dist = 1;

        let linear_results = linear_scan_standard(&dict_words, query, max_dist);
        let automaton_results = automaton_query(&dict_words, query, max_dist, Algorithm::Standard);

        assert_eq!(
            automaton_results, linear_results,
            "Leading-delete case: Standard automaton should match linear scan"
        );
        assert!(automaton_results.contains("b"));
    }

    /// Test specific transposition case
    #[test]
    fn test_transposition_specific_case() {
        let dict_words = vec!["ab".to_string(), "ba".to_string(), "abc".to_string()];
        let query = "ab";
        let max_dist = 1;

        let linear_results = linear_scan_transposition(&dict_words, query, max_dist);
        let automaton_results =
            automaton_query(&dict_words, query, max_dist, Algorithm::Transposition);

        assert_eq!(
            automaton_results, linear_results,
            "Transposition case: automaton should match linear scan"
        );
    }

    /// Test merge and split specific case
    #[test]
    fn test_merge_split_specific_case() {
        let dict_words = vec!["aa".to_string(), "a".to_string(), "aaa".to_string()];
        let query = "aa";
        let max_dist = 1;

        let linear_results = linear_scan_merge_split(&dict_words, query, max_dist);
        let automaton_results =
            automaton_query(&dict_words, query, max_dist, Algorithm::MergeAndSplit);

        assert_eq!(
            automaton_results, linear_results,
            "Merge/split case: automaton should match linear scan"
        );
    }

    /// Generic MergeAndSplit permits any two query chars to merge into one dict char.
    #[test]
    fn test_merge_split_generic_two_to_one_acceptance() {
        let dict_words = vec!["c".to_string()];
        let query = "ab";
        let max_dist = 1;
        let cache = create_memo_cache();

        assert_eq!(merge_and_split_distance(query, "c", &cache), 1);

        let linear_results = linear_scan_merge_split(&dict_words, query, max_dist);
        let automaton_results =
            automaton_query(&dict_words, query, max_dist, Algorithm::MergeAndSplit);

        assert_eq!(automaton_results, linear_results);
        assert!(automaton_results.contains("c"));
    }

    /// Generic MergeAndSplit permits any one query char to split into two dict chars.
    #[test]
    fn test_merge_split_generic_one_to_two_acceptance() {
        let dict_words = vec!["bc".to_string()];
        let query = "a";
        let max_dist = 1;
        let cache = create_memo_cache();

        assert_eq!(merge_and_split_distance(query, "bc", &cache), 1);

        let linear_results = linear_scan_merge_split(&dict_words, query, max_dist);
        let automaton_results =
            automaton_query(&dict_words, query, max_dist, Algorithm::MergeAndSplit);

        assert_eq!(automaton_results, linear_results);
        assert!(automaton_results.contains("bc"));
    }
}
