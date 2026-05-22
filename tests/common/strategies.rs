//! Shared proptest strategies for liblevenshtein test suite.
//!
//! These strategies provide consistent test data generation across all tests.
//! Each test binary uses a subset of strategies, so per-binary compilation
//! flags the unused ones — silence them at file scope.

#![allow(dead_code)]

use proptest::prelude::*;
use std::ops::Range;

/// Generate ASCII lowercase words (simple case).
///
/// Produces strings of 1-10 lowercase ASCII letters.
pub fn ascii_word_strategy() -> impl Strategy<Value = String> {
    "[a-z]{1,10}"
}

/// Generate ASCII words with mixed case.
///
/// Produces strings of 1-10 ASCII letters (upper and lower case).
pub fn ascii_mixed_case_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z]{1,10}"
}

/// Generate Unicode words including Greek letters.
///
/// Produces strings of 1-8 characters from Latin, Greek, and numeric sets.
pub fn unicode_word_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9αβγδεζηθικλμνξοπρστυφχψω]{1,8}"
}

/// Generate Unicode words with extended character sets.
///
/// Includes Cyrillic characters in addition to Latin and Greek.
pub fn extended_unicode_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Zαβγδεабвгдеёж]{1,8}"
}

/// Generate small dictionaries of ASCII words.
///
/// Produces vectors of 1-10 lowercase ASCII words.
pub fn small_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 1..=10)
}

/// Generate medium dictionaries of ASCII words.
///
/// Produces vectors of 10-50 lowercase ASCII words.
pub fn medium_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 10..=50)
}

/// Generate dictionaries with Unicode words.
///
/// Produces vectors of 1-10 Unicode words.
pub fn unicode_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(unicode_word_strategy(), 1..=10)
}

/// Generate time series data with random walk pattern.
///
/// Produces realistic time series data that resembles financial or sensor data.
///
/// # Arguments
///
/// * `len` - Range of lengths to generate
///
/// # Example
///
/// ```ignore
/// use proptest::prelude::*;
/// use crate::common::strategies::time_series_strategy;
///
/// proptest! {
///     #[test]
///     fn test_time_series(data in time_series_strategy(5..20)) {
///         assert!(data.len() >= 5 && data.len() < 20);
///     }
/// }
/// ```
pub fn time_series_strategy(len: Range<usize>) -> impl Strategy<Value = Vec<f64>> {
    // Generate a length within range, then generate that many f64 values
    (len.clone()).prop_flat_map(move |length| {
        prop::collection::vec(-10.0f64..10.0f64, length..=length).prop_map(|deltas| {
            // Convert deltas to a random walk
            let mut series = Vec::with_capacity(deltas.len());
            let mut current = 0.0;
            for delta in deltas {
                current += delta * 0.1; // Scale down deltas
                series.push(current);
            }
            series
        })
    })
}

/// Generate short time series data.
///
/// Produces time series of 2-10 values, useful for quick tests.
pub fn short_time_series_strategy() -> impl Strategy<Value = Vec<f64>> {
    time_series_strategy(2..10)
}

/// Generate medium time series data.
///
/// Produces time series of 10-50 values.
pub fn medium_time_series_strategy() -> impl Strategy<Value = Vec<f64>> {
    time_series_strategy(10..50)
}

/// Generate IPA phonetic patterns.
///
/// Produces strings containing common IPA symbols for phonetic testing.
///
/// Note: This is a simplified strategy using ASCII approximations.
/// For full IPA testing, use unicode ranges directly.
pub fn ipa_pattern_strategy() -> impl Strategy<Value = String> {
    // Common IPA-like characters (simplified to ASCII-compatible for testing)
    "[aeioupbdtfvszmnlrkgwj]{1,8}"
}

/// Generate edit distance values.
///
/// Produces small non-negative integers suitable for max_distance parameters.
pub fn distance_strategy() -> impl Strategy<Value = usize> {
    0usize..=3
}

/// Generate single lowercase ASCII characters.
pub fn ascii_char_strategy() -> impl Strategy<Value = char> {
    prop::char::range('a', 'z')
}

/// Generate query-dictionary pairs where the query is likely to have matches.
///
/// This strategy generates a dictionary and a query that is a modified version
/// of one dictionary word, ensuring interesting test cases.
pub fn query_with_matches_strategy() -> impl Strategy<Value = (Vec<String>, String, usize)> {
    small_dict_strategy().prop_flat_map(|dict| {
        if dict.is_empty() {
            // Return an arbitrary empty case
            Just((vec!["test".to_string()], "test".to_string(), 0)).boxed()
        } else {
            // Pick a word from the dict and optionally modify it
            let dict_clone = dict.clone();
            (
                Just(dict),
                prop::sample::select(dict_clone),
                distance_strategy(),
            )
                .prop_map(|(d, word, max_dist)| (d, word, max_dist))
                .boxed()
        }
    })
}

/// Generate pairs of strings with known edit distance.
///
/// Useful for testing distance calculations with ground truth.
pub fn string_pair_with_edits_strategy(
    max_edits: usize,
) -> impl Strategy<Value = (String, String, usize)> {
    ascii_word_strategy().prop_flat_map(move |base| {
        let edits = 0..=max_edits.min(base.len());
        (Just(base.clone()), Just(base), edits).prop_map(|(original, mut modified, num_edits)| {
            // Apply random edits (simplified: just change first N chars)
            let chars: Vec<char> = modified.chars().collect();
            let mut edited_chars = chars.clone();
            for i in 0..num_edits.min(edited_chars.len()) {
                // Change character to something different
                edited_chars[i] = if edited_chars[i] == 'a' { 'z' } else { 'a' };
            }
            modified = edited_chars.into_iter().collect();
            (original, modified, num_edits)
        })
    })
}

/// Generate context IDs for hierarchical testing.
pub fn context_id_strategy() -> impl Strategy<Value = usize> {
    0usize..1000
}

/// Generate parent-child context ID pairs.
pub fn parent_child_strategy() -> impl Strategy<Value = (usize, usize)> {
    (0usize..500).prop_flat_map(|parent| (Just(parent), (parent + 1)..1000))
}

// ============================================================================
// Transducer Testing Strategies
// ============================================================================

/// Generate operation costs for f64-based transducers.
///
/// Produces (insert_cost, delete_cost, substitute_cost) tuples with valid ranges.
pub fn operation_costs_f64_strategy() -> impl Strategy<Value = (f64, f64, f64)> {
    (
        0.1f64..=10.0, // insert cost
        0.1f64..=10.0, // delete cost
        0.1f64..=10.0, // substitute cost
    )
}

/// Generate asymmetric operation costs for f64-based transducers.
///
/// Produces costs where insert != delete, useful for testing asymmetric distance.
pub fn asymmetric_costs_f64_strategy() -> impl Strategy<Value = (f64, f64, f64)> {
    (
        0.5f64..=5.0,  // insert cost
        1.0f64..=10.0, // delete cost (different range)
        0.1f64..=2.0,  // substitute cost
    )
}

/// Generate bit vectors for characteristic vector testing.
///
/// Produces vectors of booleans of specified length.
pub fn bit_vector_strategy(len: usize) -> impl Strategy<Value = Vec<bool>> {
    prop::collection::vec(prop::bool::ANY, len)
}

/// Generate small bit vectors (1-16 bits).
pub fn small_bit_vector_strategy() -> impl Strategy<Value = Vec<bool>> {
    (1usize..=16).prop_flat_map(|len| prop::collection::vec(prop::bool::ANY, len))
}

/// Generate position indices for transducer states.
///
/// # Arguments
///
/// * `max_pos` - Maximum position value (exclusive)
pub fn position_strategy(max_pos: usize) -> impl Strategy<Value = usize> {
    0usize..max_pos
}

/// Generate (position, errors) pairs for state testing.
///
/// # Arguments
///
/// * `max_pos` - Maximum position value
/// * `max_errors` - Maximum error count
pub fn position_error_strategy(
    max_pos: usize,
    max_errors: usize,
) -> impl Strategy<Value = (usize, usize)> {
    (0usize..max_pos, 0usize..=max_errors)
}

/// Generate word length for query testing.
pub fn word_length_strategy() -> impl Strategy<Value = usize> {
    1usize..=20
}

/// Generate max distance values for transducer configuration.
pub fn max_distance_strategy() -> impl Strategy<Value = usize> {
    1usize..=5
}

// ============================================================================
// Phonetic/NFA Testing Strategies
// ============================================================================

/// Generate simple regex patterns for NFA testing.
///
/// Produces patterns like "ab", "a|b", "a*", "(ab)+".
pub fn simple_regex_pattern_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Literal strings
        "[a-z]{1,4}",
        // Alternations
        "[a-z]{1,2}".prop_map(|s| format!("{}|{}", s, s.chars().rev().collect::<String>())),
        // Kleene star
        "[a-z]{1,2}".prop_map(|s| format!("{}*", s)),
        // Plus quantifier
        "[a-z]{1,2}".prop_map(|s| format!("{}+", s)),
        // Optional
        "[a-z]{1,2}".prop_map(|s| format!("{}?", s)),
        // Groups
        "[a-z]{1,3}".prop_map(|s| format!("({})", s)),
    ]
}

/// Generate character class patterns for NFA testing.
///
/// Produces patterns like "[abc]", "[a-z]", "[^xy]".
pub fn char_class_pattern_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Simple character class
        "[a-z]{2,5}".prop_map(|s| format!("[{}]", s)),
        // Range
        Just("[a-z]".to_string()),
        Just("[A-Z]".to_string()),
        Just("[0-9]".to_string()),
        // Negated class
        "[a-z]{2,4}".prop_map(|s| format!("[^{}]", s)),
    ]
}

/// Generate character ranges for CharClass testing.
///
/// Produces (start_char, end_char) pairs where start <= end.
pub fn char_range_strategy() -> impl Strategy<Value = (char, char)> {
    prop::char::range('a', 'z')
        .prop_flat_map(|start| prop::char::range(start, 'z').prop_map(move |end| (start, end)))
}

/// Generate NFA state IDs.
pub fn state_id_strategy() -> impl Strategy<Value = usize> {
    0usize..1000
}

/// Generate small NFA state ID sets.
pub fn state_set_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(state_id_strategy(), 0..=10)
}

// ============================================================================
// Filter Testing Strategies
// ============================================================================

/// Generate n-gram sizes for filter testing.
pub fn ngram_size_strategy() -> impl Strategy<Value = usize> {
    2usize..=4
}

/// Generate Jaro-Winkler similarity thresholds.
pub fn jaro_threshold_strategy() -> impl Strategy<Value = f64> {
    0.5f64..=0.95
}

/// Generate pairs of similar strings for Jaro-Winkler testing.
///
/// Produces (original, modified) pairs where modified has small edits.
pub fn similar_string_pair_strategy() -> impl Strategy<Value = (String, String)> {
    ascii_word_strategy().prop_flat_map(|original| {
        let len = original.len();
        if len < 2 {
            Just((original.clone(), original)).boxed()
        } else {
            // Randomly modify one character
            (Just(original.clone()), 0..len, prop::char::range('a', 'z'))
                .prop_map(|(orig, pos, new_char)| {
                    let mut chars: Vec<char> = orig.chars().collect();
                    if chars[pos] != new_char {
                        chars[pos] = new_char;
                    }
                    (orig, chars.into_iter().collect())
                })
                .boxed()
        }
    })
}

/// Generate filter candidate lists for hybrid matcher testing.
pub fn filter_candidates_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(ascii_word_strategy(), 5..=50)
}

// ============================================================================
// Time Series Testing Strategies (Extended)
// ============================================================================

/// Generate MSM cost parameter (c value).
pub fn msm_cost_strategy() -> impl Strategy<Value = f64> {
    0.1f64..=10.0
}

/// Generate quantization bin counts.
pub fn quantization_bins_strategy() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(16u32),
        Just(32u32),
        Just(64u32),
        Just(128u32),
        Just(256u32),
    ]
}

/// Generate quantization range parameters.
///
/// Produces (min, max) pairs where min < max.
pub fn quantization_range_strategy() -> impl Strategy<Value = (f64, f64)> {
    (-100.0f64..0.0).prop_flat_map(|min| (0.0f64..100.0).prop_map(move |max| (min, max)))
}

/// Generate pairs of time series for distance testing.
pub fn time_series_pair_strategy(len: Range<usize>) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    (time_series_strategy(len.clone()), time_series_strategy(len))
}

/// Generate time series with similar patterns.
///
/// Produces (original, noisy_copy) pairs where noisy_copy has small perturbations.
pub fn similar_time_series_strategy(
    len: Range<usize>,
) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    time_series_strategy(len).prop_flat_map(|original| {
        let noise_range = -0.5f64..0.5f64;
        prop::collection::vec(noise_range, original.len()).prop_map(move |noise| {
            let noisy: Vec<f64> = original
                .iter()
                .zip(noise.iter())
                .map(|(v, n)| v + n)
                .collect();
            (original.clone(), noisy)
        })
    })
}

// ============================================================================
// Contextual/Hierarchical Testing Strategies
// ============================================================================

/// Generate context hierarchy depths.
pub fn context_depth_strategy() -> impl Strategy<Value = usize> {
    1usize..=5
}

/// Generate context weight values.
pub fn context_weight_strategy() -> impl Strategy<Value = f64> {
    0.0f64..=1.0
}

/// Generate context path (sequence of context IDs from root to leaf).
pub fn context_path_strategy(max_depth: usize) -> impl Strategy<Value = Vec<usize>> {
    (1usize..=max_depth).prop_flat_map(|depth| prop::collection::vec(context_id_strategy(), depth))
}
