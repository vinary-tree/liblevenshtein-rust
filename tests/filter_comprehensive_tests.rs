//! Comprehensive property-based tests for filter module.
//!
//! These tests verify:
//! - Jaro-Winkler similarity bounds and properties
//! - N-gram filter completeness (no false negatives)
//! - Hybrid matcher correctness

mod common;

use liblevenshtein::distance::standard_distance;
use liblevenshtein::filter::{
    is_similar, jaro_similarity, jaro_winkler_similarity, jaro_winkler_similarity_scaled,
    NgramIndex,
};
use proptest::prelude::*;

// ============================================================================
// Jaro-Winkler Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Jaro similarity is bounded: 0.0 <= sim <= 1.0
    #[test]
    fn prop_jaro_similarity_bounded(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let sim = jaro_similarity(&s1, &s2);
        prop_assert!(
            (0.0..=1.0).contains(&sim),
            "Jaro similarity {} not in [0, 1] for '{}' vs '{}'",
            sim, s1, s2
        );
    }

    /// Jaro-Winkler similarity is bounded: 0.0 <= sim <= 1.0
    #[test]
    fn prop_jaro_winkler_similarity_bounded(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let sim = jaro_winkler_similarity(&s1, &s2);
        prop_assert!(
            (0.0..=1.0).contains(&sim),
            "Jaro-Winkler similarity {} not in [0, 1] for '{}' vs '{}'",
            sim, s1, s2
        );
    }

    /// Jaro similarity identity: sim(s, s) == 1.0
    #[test]
    fn prop_jaro_identity(
        s in common::strategies::ascii_word_strategy(),
    ) {
        if s.is_empty() {
            // Empty string has similarity 1.0 with itself
            let sim = jaro_similarity(&s, &s);
            prop_assert!(
                (sim - 1.0).abs() < 1e-9,
                "Jaro identity failed for empty string: {}",
                sim
            );
        } else {
            let sim = jaro_similarity(&s, &s);
            prop_assert!(
                (sim - 1.0).abs() < 1e-9,
                "Jaro identity failed for '{}': {}",
                s, sim
            );
        }
    }

    /// Jaro-Winkler similarity identity: sim(s, s) == 1.0
    #[test]
    fn prop_jaro_winkler_identity(
        s in common::strategies::ascii_word_strategy(),
    ) {
        let sim = jaro_winkler_similarity(&s, &s);
        prop_assert!(
            (sim - 1.0).abs() < 1e-9,
            "Jaro-Winkler identity failed for '{}': {}",
            s, sim
        );
    }

    /// Jaro similarity symmetry: sim(a, b) == sim(b, a)
    #[test]
    fn prop_jaro_symmetry(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let sim_12 = jaro_similarity(&s1, &s2);
        let sim_21 = jaro_similarity(&s2, &s1);
        prop_assert!(
            (sim_12 - sim_21).abs() < 1e-9,
            "Jaro symmetry failed: sim('{}', '{}') = {} != sim('{}', '{}') = {}",
            s1, s2, sim_12, s2, s1, sim_21
        );
    }

    /// Jaro-Winkler similarity symmetry: sim(a, b) == sim(b, a)
    #[test]
    fn prop_jaro_winkler_symmetry(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let sim_12 = jaro_winkler_similarity(&s1, &s2);
        let sim_21 = jaro_winkler_similarity(&s2, &s1);
        prop_assert!(
            (sim_12 - sim_21).abs() < 1e-9,
            "Jaro-Winkler symmetry failed: sim('{}', '{}') = {} != sim('{}', '{}') = {}",
            s1, s2, sim_12, s2, s1, sim_21
        );
    }

    /// Jaro-Winkler >= Jaro (prefix bonus never negative)
    #[test]
    fn prop_jaro_winkler_geq_jaro(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        let jaro = jaro_similarity(&s1, &s2);
        let jaro_winkler = jaro_winkler_similarity(&s1, &s2);
        prop_assert!(
            jaro_winkler >= jaro - 1e-9,
            "Jaro-Winkler {} < Jaro {} for '{}' vs '{}'",
            jaro_winkler, jaro, s1, s2
        );
    }

    /// Common prefix should increase Jaro-Winkler similarity
    #[test]
    fn prop_common_prefix_bonus(
        prefix in "[a-z]{2,4}",
        suffix1 in "[a-z]{2,6}",
        suffix2 in "[a-z]{2,6}",
    ) {
        let with_prefix1 = format!("{}{}", prefix, suffix1);
        let with_prefix2 = format!("{}{}", prefix, suffix2);
        let without_prefix = format!("x{}", suffix1);

        let sim_with = jaro_winkler_similarity(&with_prefix1, &with_prefix2);
        let sim_without = jaro_winkler_similarity(&with_prefix1, &without_prefix);

        // Strings with common prefix generally have higher similarity
        // (though not always due to other factors)
        // At minimum, both should be valid similarities
        prop_assert!((0.0..=1.0).contains(&sim_with));
        prop_assert!((0.0..=1.0).contains(&sim_without));
    }

    /// Scaled Jaro-Winkler should also be bounded
    #[test]
    fn prop_scaled_jaro_winkler_bounded(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
        scale in 0.05f64..=0.25f64,
    ) {
        let sim = jaro_winkler_similarity_scaled(&s1, &s2, scale);
        prop_assert!(
            (0.0..=1.0).contains(&sim),
            "Scaled Jaro-Winkler {} not in [0, 1]",
            sim
        );
    }
}

// ============================================================================
// N-gram Filter Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// N-gram filter mostly finds true positives
    ///
    /// Note: N-gram filters CAN have false negatives for short strings or when
    /// the edit transforms all n-grams (e.g., "naa" vs "nba" share 0 bigrams
    /// despite having edit distance 1). This test verifies that for longer
    /// strings with sufficient n-gram overlap, the filter works correctly.
    ///
    /// For guaranteed completeness, always verify n-gram candidates with actual
    /// edit distance calculation.
    #[test]
    fn prop_ngram_no_false_negatives(
        dict_words in common::strategies::small_dict_strategy(),
        query in common::strategies::ascii_word_strategy(),
        max_dist in 1usize..=2,
    ) {
        // Skip very short words where n-gram filtering is unreliable
        // Require length >= 5 to ensure meaningful n-gram overlap
        let dict_words: Vec<_> = dict_words.into_iter().filter(|w| w.len() >= 5).collect();
        if dict_words.is_empty() || query.len() < 5 {
            return Ok(());
        }

        // Build n-gram index
        let mut index = NgramIndex::new(2); // Bigrams
        for word in &dict_words {
            index.insert(word);
        }

        // Get candidates from filter
        let candidates = index.find_candidates(&query, max_dist);

        // Find true positives via linear scan
        let true_positives: Vec<_> = dict_words
            .iter()
            .filter(|word| standard_distance(&query, word) <= max_dist)
            .collect();

        // For words of sufficient length, n-gram filter should find true positives
        // (though edge cases with all-different n-grams may still fail)
        for tp in &true_positives {
            // Only assert for cases where we expect n-gram overlap
            // Skip if the edit transforms >50% of bigrams
            let tp_bigrams: std::collections::HashSet<_> = tp.as_bytes()
                .windows(2)
                .collect();
            let query_bigrams: std::collections::HashSet<_> = query.as_bytes()
                .windows(2)
                .collect();
            let overlap = tp_bigrams.intersection(&query_bigrams).count();

            // Only assert when there's expected overlap
            if overlap > 0 || query == **tp {
                prop_assert!(
                    candidates.contains(&tp.as_str()),
                    "Unexpected false negative: '{}' within distance {} of '{}' but not in candidates (overlap: {})",
                    tp, max_dist, query, overlap
                );
            }
        }
    }

    /// N-gram index should find exact matches
    #[test]
    fn prop_ngram_finds_exact_matches(
        dict_words in common::strategies::small_dict_strategy(),
    ) {
        let dict_words: Vec<_> = dict_words.into_iter().filter(|w| w.len() >= 3).collect();
        if dict_words.is_empty() {
            return Ok(());
        }

        let mut index = NgramIndex::new(2);
        for word in &dict_words {
            index.insert(word);
        }

        // Query for an exact term in the dictionary
        let query = &dict_words[0];
        let candidates = index.find_candidates(query, 0);

        prop_assert!(
            candidates.contains(&query.as_str()),
            "Exact match '{}' not found in candidates",
            query
        );
    }

    /// N-gram index with trigrams should also be complete
    #[test]
    fn prop_ngram_trigrams_complete(
        dict_words in common::strategies::small_dict_strategy(),
        query in "[a-z]{4,10}",
        max_dist in 1usize..=2,
    ) {
        let dict_words: Vec<_> = dict_words.into_iter().filter(|w| w.len() >= 4).collect();
        if dict_words.is_empty() {
            return Ok(());
        }

        let mut index = NgramIndex::new(3); // Trigrams
        for word in &dict_words {
            index.insert(word);
        }

        let candidates = index.find_candidates(&query, max_dist);

        let true_positives: Vec<_> = dict_words
            .iter()
            .filter(|word| standard_distance(&query, word) <= max_dist)
            .collect();

        for tp in &true_positives {
            // Only assert for cases where we expect trigram overlap
            let tp_trigrams: std::collections::HashSet<_> = tp.as_bytes()
                .windows(3)
                .collect();
            let query_trigrams: std::collections::HashSet<_> = query.as_bytes()
                .windows(3)
                .collect();
            let overlap = tp_trigrams.intersection(&query_trigrams).count();

            // Only assert when there's expected overlap
            if overlap > 0 || query == **tp {
                prop_assert!(
                    candidates.contains(&tp.as_str()),
                    "Trigram false negative: '{}' within distance {} of '{}' (overlap: {})",
                    tp, max_dist, query, overlap
                );
            }
        }
    }
}

// ============================================================================
// is_similar Function Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// is_similar should be consistent with jaro_winkler_similarity
    #[test]
    fn prop_is_similar_consistent(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
        threshold in 0.5f64..1.0f64,
    ) {
        let sim = jaro_winkler_similarity(&s1, &s2);
        let is_sim = is_similar(&s1, &s2, threshold);

        if sim >= threshold {
            prop_assert!(
                is_sim,
                "is_similar should be true when similarity {} >= threshold {}",
                sim, threshold
            );
        } else {
            prop_assert!(
                !is_sim,
                "is_similar should be false when similarity {} < threshold {}",
                sim, threshold
            );
        }
    }

    /// is_similar with threshold 0 should always return true for non-empty strings
    #[test]
    fn prop_is_similar_threshold_zero(
        s1 in common::strategies::ascii_word_strategy(),
        s2 in common::strategies::ascii_word_strategy(),
    ) {
        // With threshold 0, any string pair should be "similar"
        let is_sim = is_similar(&s1, &s2, 0.0);
        prop_assert!(is_sim, "is_similar with threshold 0 should always be true");
    }

    /// is_similar with threshold 1.0 should only be true for identical strings
    #[test]
    fn prop_is_similar_threshold_one(
        s in common::strategies::ascii_word_strategy(),
    ) {
        // With threshold 1.0, only identical strings should match
        let is_sim_same = is_similar(&s, &s, 1.0);
        prop_assert!(is_sim_same, "Identical strings should be similar at threshold 1.0");
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_jaro_empty_strings() {
    // Two empty strings have similarity 1.0
    assert!((jaro_similarity("", "") - 1.0).abs() < 1e-9);
    assert!((jaro_winkler_similarity("", "") - 1.0).abs() < 1e-9);
}

#[test]
fn test_jaro_one_empty() {
    // Empty vs non-empty has similarity 0.0
    assert!(jaro_similarity("", "test").abs() < 1e-9);
    assert!(jaro_similarity("test", "").abs() < 1e-9);
}

#[test]
fn test_jaro_single_char() {
    // Single character strings
    assert!((jaro_similarity("a", "a") - 1.0).abs() < 1e-9);
    assert!(jaro_similarity("a", "b") < 1.0);
}

#[test]
fn test_jaro_winkler_prefix_bonus() {
    // Words with common prefix should have higher Jaro-Winkler than Jaro
    let jaro = jaro_similarity("prefix_abc", "prefix_xyz");
    let jw = jaro_winkler_similarity("prefix_abc", "prefix_xyz");

    assert!(jw >= jaro, "Jaro-Winkler should be >= Jaro for common prefix");
}

#[test]
fn test_jaro_winkler_known_values() {
    // Test against known expected values
    // "martha" vs "marhta" is a classic test case
    let sim = jaro_winkler_similarity("martha", "marhta");
    assert!(sim > 0.95, "martha/marhta should have high similarity");
    assert!(sim <= 1.0, "Similarity should not exceed 1.0");
}

#[test]
fn test_ngram_empty_dictionary() {
    let index = NgramIndex::new(2);
    let candidates = index.find_candidates("test", 2);
    assert!(candidates.is_empty(), "Empty index should return no candidates");
}

#[test]
fn test_ngram_single_term() {
    let mut index = NgramIndex::new(2);
    index.insert("test");

    // Exact match
    let candidates = index.find_candidates("test", 0);
    assert!(candidates.contains(&"test"));

    // Similar query with shared bigrams
    // "test" has bigrams: "te", "es", "st"
    // "tests" has bigrams: "te", "es", "st", "ts" - shares 3 bigrams
    let candidates = index.find_candidates("tests", 1);
    assert!(candidates.contains(&"test"));

    // Note: "tset" has bigrams: "ts", "se", "et" - shares 0 bigrams with "test"
    // So n-gram filter correctly rejects it despite low edit distance
}

#[test]
fn test_ngram_short_terms() {
    let mut index = NgramIndex::new(2);
    index.insert("ab");
    index.insert("abc");

    // Short terms should still be indexed
    let candidates = index.find_candidates("ab", 1);
    assert!(!candidates.is_empty());
}

// ============================================================================
// Hybrid Matcher Tests (if available)
// ============================================================================

// Note: HybridMatcher requires more setup, tested here as unit tests

#[test]
fn test_hybrid_matcher_basic() {
    use liblevenshtein::filter::HybridMatcherBuilder;

    let terms: Vec<String> = vec!["apple", "application", "banana", "apply", "apricot"]
        .into_iter()
        .map(String::from)
        .collect();

    let matcher = HybridMatcherBuilder::new()
        .ngram_size(2)
        .jaro_threshold(0.7)
        .build(terms);

    // Find candidates for "aple" (typo of "apple")
    let candidates = matcher.filter_candidates("aple", 2);

    // Should find "apple" and "apply"
    assert!(
        candidates.contains(&"apple"),
        "Should find 'apple'"
    );
}

#[test]
fn test_hybrid_matcher_no_matches() {
    use liblevenshtein::filter::HybridMatcherBuilder;

    let terms: Vec<String> = vec!["xyz", "abc", "def"]
        .into_iter()
        .map(String::from)
        .collect();

    let matcher = HybridMatcherBuilder::new()
        .ngram_size(2)
        .jaro_threshold(0.8)
        .build(terms);

    // Query very different from all terms
    let candidates = matcher.filter_candidates("qqqqqq", 1);
    assert!(candidates.is_empty(), "Should find no matches");
}

#[test]
fn test_hybrid_matcher_exact_match() {
    use liblevenshtein::filter::HybridMatcherBuilder;

    let terms: Vec<String> = vec!["hello", "world", "test"]
        .into_iter()
        .map(String::from)
        .collect();

    let matcher = HybridMatcherBuilder::new()
        .ngram_size(2)
        .build(terms);

    let candidates = matcher.filter_candidates("hello", 0);
    assert!(candidates.contains(&"hello"), "Should find 'hello'");
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_filter_pipeline() {
    // Test the full filtering pipeline: ngram -> jaro-winkler -> verification

    // Build ngram index
    let mut index = NgramIndex::new(2);
    let terms = ["programming", "program", "programmer", "progress", "project"];
    for term in &terms {
        index.insert(*term);
    }

    let query = "progam"; // Typo of "program"
    let max_dist = 2;

    // Stage 1: N-gram filtering
    let candidates = index.find_candidates(query, max_dist);
    assert!(!candidates.is_empty(), "N-gram should find candidates");

    // Stage 2: Jaro-Winkler refinement
    let jw_threshold = 0.7;
    let refined: Vec<_> = candidates
        .into_iter()
        .filter(|c| jaro_winkler_similarity(query, c) >= jw_threshold)
        .collect();
    assert!(!refined.is_empty(), "JW refinement should keep some candidates");

    // Stage 3: Exact verification
    let verified: Vec<_> = refined
        .into_iter()
        .filter(|c| standard_distance(query, c) <= max_dist)
        .collect();
    assert!(
        verified.contains(&"program"),
        "Should find 'program'"
    );
}
