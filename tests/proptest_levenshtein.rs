//! Property-based tests for Levenshtein automaton using proptest
//!
//! These tests help discover edge cases and minimal failing examples
//! for the Levenshtein distance algorithm.

use liblevenshtein::prelude::*;
use proptest::prelude::*;

// Strategy for generating simple ASCII words
fn word_strategy() -> impl Strategy<Value = String> {
    "[a-z]{1,10}"
}

// Strategy for generating a small dictionary
fn small_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(word_strategy(), 1..=10)
}

// Helper: compute naive Levenshtein distance for verification
fn naive_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
        *cell = j;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len1][len2]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: For any dictionary and query, all returned results should have
    /// Levenshtein distance <= max_distance
    #[test]
    fn prop_results_within_distance(
        dict_words in small_dict_strategy(),
        query in word_strategy(),
        max_dist in 0usize..=5  // Increased from 2 to catch high-distance regressions
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        for result in results {
            let actual_dist = naive_levenshtein_distance(&query, &result);
            prop_assert!(
                actual_dist <= max_dist,
                "Result '{}' has distance {} from query '{}', but max_distance was {}",
                result, actual_dist, query, max_dist
            );
        }
    }

    /// Property: If a word is in the dictionary and distance=0, it should be found
    #[test]
    fn prop_exact_match_found(
        dict_words in small_dict_strategy(),
    ) {
        if dict_words.is_empty() {
            return Ok(());
        }

        let query_word = dict_words[0].clone();
        let dict = DoubleArrayTrie::from_terms(dict_words);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query(&query_word, 0).collect();

        prop_assert!(
            results.contains(&query_word),
            "Dictionary should contain exact match for '{}'",
            query_word
        );
    }

    /// Property: All words in dictionary within max_distance should be found
    ///
    /// This is the key property that can help find deletion regressions.
    #[test]
    fn prop_all_close_words_found(
        dict_words in small_dict_strategy(),
        query in word_strategy(),
        max_dist in 1usize..=5  // Increased from 2 to catch high-distance regressions
    ) {
        let dict = DoubleArrayTrie::from_terms(dict_words.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query(&query, max_dist).collect();

        // Check that all words in dictionary within max_distance are found
        for dict_word in &dict_words {
            let actual_dist = naive_levenshtein_distance(&query, dict_word);
            if actual_dist <= max_dist {
                prop_assert!(
                    results.contains(dict_word),
                    "POTENTIAL REGRESSION: Dictionary {:?}, query '{}', distance {}: \
                     word '{}' has actual distance {} but was not found in results {:?}",
                    dict_words, query, max_dist, dict_word, actual_dist, results
                );
            }
        }
    }
}

#[cfg(test)]
mod manual_shrink_tests {
    use super::*;

    /// Regression test: This was a minimal failing case discovered by proptest
    /// that helped identify the deletion defect. Now fixed and serves as a regression test.
    #[test]
    fn minimal_failing_case_from_proptest() {
        // Original failing case: Dict ["test", "apple", "world"], query "testt", distance 1
        // Expected: "test" should be found (1 deletion from query)
        // This defect was fixed in the Levenshtein query iterator

        let dict = DoubleArrayTrie::from_terms(vec!["test", "apple", "world"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let results: Vec<_> = transducer.query("testt", 1).collect();

        assert!(
            results.contains(&"test".to_string()),
            "Regression: 'testt' should find 'test' with distance 1"
        );
    }
}
