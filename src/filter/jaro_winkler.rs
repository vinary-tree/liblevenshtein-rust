//! Jaro and Jaro-Winkler string similarity.
//!
//! This module provides fast string similarity metrics for pre-filtering
//! approximate matches before expensive Levenshtein computation.
//!
//! # Jaro Similarity
//!
//! The Jaro similarity measures the similarity between two strings based on:
//! - Number of matching characters (within a distance window)
//! - Number of transpositions (matched chars in different order)
//!
//! Formula:
//! ```text
//! jaro = (m/|s1| + m/|s2| + (m-t)/m) / 3
//! ```
//! Where m = matches, t = transpositions/2
//!
//! # Jaro-Winkler Similarity
//!
//! Jaro-Winkler extends Jaro by giving a bonus for common prefixes:
//! ```text
//! jaro_winkler = jaro + (prefix_len * 0.1 * (1 - jaro))
//! ```
//!
//! This makes it especially good for matching names and correcting typos,
//! as people tend to get the beginning of words correct.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::filter::{jaro_similarity, jaro_winkler_similarity};
//!
//! // High similarity for similar strings
//! let sim = jaro_winkler_similarity("martha", "marhta");
//! assert!(sim > 0.95);
//!
//! // Low similarity for different strings
//! let sim = jaro_winkler_similarity("hello", "world");
//! assert!(sim < 0.5);
//! ```
//!
//! # Complexity
//!
//! - Time: O(|s1| * |s2|) worst case, often better in practice
//! - Space: O(|s1| + |s2|) for match tracking

/// Maximum prefix length for Jaro-Winkler bonus.
const MAX_PREFIX_LENGTH: usize = 4;

/// Scaling factor for prefix bonus in Jaro-Winkler.
const PREFIX_SCALE: f64 = 0.1;

/// Compute Jaro similarity between two strings.
///
/// Returns a value in [0.0, 1.0] where 1.0 is an exact match.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// Jaro similarity score in [0.0, 1.0].
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::filter::jaro_similarity;
///
/// assert_eq!(jaro_similarity("", ""), 1.0);
/// assert_eq!(jaro_similarity("abc", "abc"), 1.0);
/// assert!(jaro_similarity("martha", "marhta") > 0.9);
/// ```
pub fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    jaro_similarity_chars(&s1_chars, &s2_chars)
}

/// Compute Jaro similarity from character slices.
///
/// This is the core implementation that works on pre-converted character vectors.
fn jaro_similarity_chars(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    // Handle empty strings
    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    // Match distance: characters must be within (max(len1, len2) / 2) - 1
    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Find matching characters
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;

    ((m / len1 as f64) + (m / len2 as f64) + ((m - t) / m)) / 3.0
}

/// Compute Jaro-Winkler similarity between two strings.
///
/// Extends Jaro similarity with a prefix bonus for strings that match
/// from the beginning. This is especially useful for name matching
/// and typo correction.
///
/// Returns a value in [0.0, 1.0] where 1.0 is an exact match.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// Jaro-Winkler similarity score in [0.0, 1.0].
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::filter::jaro_winkler_similarity;
///
/// assert_eq!(jaro_winkler_similarity("", ""), 1.0);
/// assert_eq!(jaro_winkler_similarity("abc", "abc"), 1.0);
///
/// // Prefix bonus makes similar-prefix strings score higher
/// let jw = jaro_winkler_similarity("MARTHA", "MARHTA");
/// let j = liblevenshtein::filter::jaro_similarity("MARTHA", "MARHTA");
/// assert!(jw > j);
/// ```
pub fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    jaro_winkler_similarity_chars(&s1_chars, &s2_chars)
}

/// Compute Jaro-Winkler similarity from character slices.
fn jaro_winkler_similarity_chars(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let jaro = jaro_similarity_chars(s1_chars, s2_chars);

    // Common prefix length (up to MAX_PREFIX_LENGTH)
    let prefix_len = s1_chars
        .iter()
        .zip(s2_chars.iter())
        .take(MAX_PREFIX_LENGTH)
        .take_while(|(a, b)| a == b)
        .count();

    // Winkler modification: boost for common prefix
    jaro + (prefix_len as f64 * PREFIX_SCALE * (1.0 - jaro))
}

/// Compute Jaro-Winkler similarity with a custom prefix scale.
///
/// Allows tuning the prefix bonus for specific use cases.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
/// * `prefix_scale` - Scaling factor for prefix bonus (typically 0.1)
///
/// # Panics
///
/// Panics if prefix_scale is not in [0.0, 0.25] (to ensure similarity stays in [0, 1]).
pub fn jaro_winkler_similarity_scaled(s1: &str, s2: &str, prefix_scale: f64) -> f64 {
    assert!(
        (0.0..=0.25).contains(&prefix_scale),
        "prefix_scale must be in [0.0, 0.25] to ensure result in [0, 1], got {}",
        prefix_scale
    );

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let jaro = jaro_similarity_chars(&s1_chars, &s2_chars);

    let prefix_len = s1_chars
        .iter()
        .zip(s2_chars.iter())
        .take(MAX_PREFIX_LENGTH)
        .take_while(|(a, b)| a == b)
        .count();

    jaro + (prefix_len as f64 * prefix_scale * (1.0 - jaro))
}

/// Check if two strings are similar according to Jaro-Winkler.
///
/// Convenience function for threshold-based filtering.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
/// * `threshold` - Minimum similarity to be considered "similar"
///
/// # Returns
///
/// `true` if Jaro-Winkler similarity >= threshold.
#[inline]
pub fn is_similar(s1: &str, s2: &str, threshold: f64) -> bool {
    jaro_winkler_similarity(s1, s2) >= threshold
}

/// Convert Jaro-Winkler similarity to approximate edit distance.
///
/// This is a rough approximation for filtering purposes.
/// The relationship is not linear, but this provides a useful heuristic.
///
/// # Arguments
///
/// * `similarity` - Jaro-Winkler similarity in [0.0, 1.0]
/// * `avg_len` - Average length of the strings being compared
///
/// # Returns
///
/// Approximate edit distance (not guaranteed to be exact).
pub fn similarity_to_distance_approx(similarity: f64, avg_len: f64) -> f64 {
    // Rough approximation: distance ≈ (1 - similarity) * avg_len
    (1.0 - similarity) * avg_len
}

/// Convert edit distance to approximate Jaro-Winkler similarity.
///
/// Inverse of `similarity_to_distance_approx`.
///
/// # Arguments
///
/// * `distance` - Edit distance
/// * `avg_len` - Average length of the strings being compared
///
/// # Returns
///
/// Approximate Jaro-Winkler similarity in [0.0, 1.0].
pub fn distance_to_similarity_approx(distance: f64, avg_len: f64) -> f64 {
    if avg_len <= 0.0 {
        return 1.0;
    }
    (1.0 - distance / avg_len).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_jaro_empty_strings() {
        assert!((jaro_similarity("", "") - 1.0).abs() < EPSILON);
        assert!((jaro_similarity("a", "") - 0.0).abs() < EPSILON);
        assert!((jaro_similarity("", "b") - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_jaro_identical_strings() {
        assert!((jaro_similarity("abc", "abc") - 1.0).abs() < EPSILON);
        assert!((jaro_similarity("hello", "hello") - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_jaro_classic_examples() {
        // Classic Jaro examples from literature
        // MARTHA vs MARHTA: should be around 0.944
        let sim = jaro_similarity("MARTHA", "MARHTA");
        assert!(sim > 0.94 && sim < 0.95, "MARTHA/MARHTA = {}", sim);

        // DWAYNE vs DUANE: should be around 0.822
        let sim = jaro_similarity("DWAYNE", "DUANE");
        assert!(sim > 0.82 && sim < 0.84, "DWAYNE/DUANE = {}", sim);

        // DIXON vs DICKSONX: should be around 0.767
        let sim = jaro_similarity("DIXON", "DICKSONX");
        assert!(sim > 0.76 && sim < 0.78, "DIXON/DICKSONX = {}", sim);
    }

    #[test]
    fn test_jaro_symmetry() {
        let pairs = [("hello", "world"), ("abc", "xyz"), ("test", "tset")];

        for (a, b) in pairs {
            let sim1 = jaro_similarity(a, b);
            let sim2 = jaro_similarity(b, a);
            assert!(
                (sim1 - sim2).abs() < EPSILON,
                "jaro({}, {}) = {} != {} = jaro({}, {})",
                a,
                b,
                sim1,
                sim2,
                b,
                a
            );
        }
    }

    #[test]
    fn test_jaro_winkler_empty_strings() {
        assert!((jaro_winkler_similarity("", "") - 1.0).abs() < EPSILON);
        assert!((jaro_winkler_similarity("a", "") - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_jaro_winkler_identical_strings() {
        assert!((jaro_winkler_similarity("abc", "abc") - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_jaro_winkler_prefix_bonus() {
        // Jaro-Winkler should be >= Jaro for strings with common prefix
        let pairs = [
            ("MARTHA", "MARHTA"), // Common prefix "MAR"
            ("hello", "helo"),    // Common prefix "hel"
            ("test", "tset"),     // Common prefix "t"
        ];

        for (a, b) in pairs {
            let jaro = jaro_similarity(a, b);
            let jw = jaro_winkler_similarity(a, b);
            assert!(
                jw >= jaro - EPSILON,
                "JW({}, {}) = {} should be >= Jaro = {}",
                a,
                b,
                jw,
                jaro
            );
        }
    }

    #[test]
    fn test_jaro_winkler_classic_examples() {
        // MARTHA vs MARHTA with Winkler bonus: should be around 0.961
        let sim = jaro_winkler_similarity("MARTHA", "MARHTA");
        assert!(sim > 0.96 && sim < 0.97, "MARTHA/MARHTA JW = {}", sim);
    }

    #[test]
    fn test_jaro_winkler_symmetry() {
        let pairs = [("hello", "world"), ("abc", "xyz"), ("test", "tset")];

        for (a, b) in pairs {
            let sim1 = jaro_winkler_similarity(a, b);
            let sim2 = jaro_winkler_similarity(b, a);
            assert!(
                (sim1 - sim2).abs() < EPSILON,
                "jw({}, {}) = {} != {} = jw({}, {})",
                a,
                b,
                sim1,
                sim2,
                b,
                a
            );
        }
    }

    #[test]
    fn test_jaro_winkler_scaled() {
        let sim_default = jaro_winkler_similarity("hello", "helo");
        let sim_scaled = jaro_winkler_similarity_scaled("hello", "helo", 0.2);

        // Higher scale = bigger prefix bonus
        assert!(
            sim_scaled > sim_default,
            "scaled {} should be > default {}",
            sim_scaled,
            sim_default
        );
    }

    #[test]
    #[should_panic(expected = "prefix_scale must be in [0.0, 0.25]")]
    fn test_jaro_winkler_scaled_invalid() {
        jaro_winkler_similarity_scaled("a", "b", 0.3);
    }

    #[test]
    fn test_similarity_to_distance() {
        // Perfect match = 0 distance
        let dist = similarity_to_distance_approx(1.0, 5.0);
        assert!((dist - 0.0).abs() < EPSILON);

        // Complete mismatch ≈ avg_len distance
        let dist = similarity_to_distance_approx(0.0, 5.0);
        assert!((dist - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_similarity() {
        // 0 distance = perfect match
        let sim = distance_to_similarity_approx(0.0, 5.0);
        assert!((sim - 1.0).abs() < EPSILON);

        // Distance = avg_len ≈ no similarity
        let sim = distance_to_similarity_approx(5.0, 5.0);
        assert!((sim - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_unicode() {
        // Should handle Unicode correctly
        let sim = jaro_winkler_similarity("café", "cafe");
        assert!(sim > 0.8); // Similar but not identical

        let sim = jaro_winkler_similarity("日本語", "日本語");
        assert!((sim - 1.0).abs() < EPSILON);
    }
}
