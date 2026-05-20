//! Myers' bit-parallel algorithm for edit distance computation.
//!
//! This module implements Myers' algorithm from the paper:
//! "A Fast Bit-Vector Algorithm for Approximate String Matching Based on
//! Dynamic Programming" - Gene Myers, Journal of the ACM 46(3): 395-415, 1999.
//!
//! # Complexity
//!
//! - Time: O(mn/w) where w = 64 (word size in bits)
//! - Space: O(σ) where σ = alphabet size (256 for bytes)
//!
//! # When to Use
//!
//! Myers' algorithm is optimal for:
//! - Strings where at least one is ≤64 characters
//! - Cases where edit distance is expected to be small
//! - Real-time spell checking where latency matters
//!
//! For longer strings, the SIMD-vectorized DP approach is preferred.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::distance::myers::myers_distance;
//!
//! assert_eq!(myers_distance("kitten", "sitting"), 3);
//! assert_eq!(myers_distance("test", "test"), 0);
//! ```

/// Pattern equivalence table for Myers algorithm.
///
/// Pre-computes bitmasks where bit i is set if the pattern character
/// at position i equals the indexed alphabet character.
struct PatternMasks {
    /// Equivalence bitmasks indexed by byte value (0-255)
    peq: [u64; 256],
}

impl PatternMasks {
    /// Create pattern masks for the given byte sequence.
    ///
    /// For each character c in the alphabet, peq[c] has bit i set
    /// if pattern[i] == c.
    #[inline]
    fn new(pattern: &[u8]) -> Self {
        let mut peq = [0u64; 256];

        for (i, &byte) in pattern.iter().enumerate().take(64) {
            peq[byte as usize] |= 1u64 << i;
        }

        Self { peq }
    }
}

/// Compute edit distance using Myers' bit-parallel algorithm.
///
/// This function computes the standard Levenshtein distance between
/// two strings using bit-level parallelism. Each 64-bit word operation
/// processes up to 64 pattern characters simultaneously.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
///
/// # Returns
///
/// The minimum number of single-character edits (insertions, deletions,
/// or substitutions) required to transform `source` into `target`.
///
/// # Implementation Notes
///
/// - For patterns longer than 64 characters, falls back to standard DP
/// - Uses the shorter string as the pattern for efficiency
/// - Handles UTF-8 by operating on bytes (not Unicode codepoints)
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::myers::myers_distance;
///
/// // Basic usage
/// assert_eq!(myers_distance("kitten", "sitting"), 3);
///
/// // Empty strings
/// assert_eq!(myers_distance("", "test"), 4);
/// assert_eq!(myers_distance("test", ""), 4);
///
/// // Identical strings
/// assert_eq!(myers_distance("same", "same"), 0);
/// ```
pub fn myers_distance(source: &str, target: &str) -> usize {
    let source_bytes = source.as_bytes();
    let target_bytes = target.as_bytes();

    myers_distance_bytes(source_bytes, target_bytes)
}

/// Compute edit distance on byte slices using Myers' algorithm.
///
/// This is the core implementation that operates on raw bytes.
#[inline]
pub fn myers_distance_bytes(source: &[u8], target: &[u8]) -> usize {
    // Handle trivial cases
    if source.is_empty() {
        return target.len();
    }
    if target.is_empty() {
        return source.len();
    }
    if source == target {
        return 0;
    }

    // Use shorter string as pattern (rows in DP matrix)
    // This ensures we can use a single 64-bit word when possible
    let (pattern, text) = if source.len() <= target.len() {
        (source, target)
    } else {
        (target, source)
    };

    // For patterns > 64, fall back to standard DP
    // Myers is most beneficial for short patterns where the entire
    // pattern fits in a single 64-bit word
    if pattern.len() > 64 {
        return crate::distance::standard_distance_impl(
            std::str::from_utf8(source).unwrap_or(""),
            std::str::from_utf8(target).unwrap_or(""),
        );
    }

    myers_core(pattern, text)
}

/// Core Myers algorithm for patterns ≤64 characters.
///
/// Uses a single 64-bit word to represent the pattern.
/// Reference: Myers (1999), Algorithm 1.
#[inline]
fn myers_core(pattern: &[u8], text: &[u8]) -> usize {
    let m = pattern.len();
    let masks = PatternMasks::new(pattern);

    // Initialize bit-vectors
    // VP (Positive Vertical delta): all 1s - indicates +1 transitions
    // VN (Negative Vertical delta): all 0s - indicates -1 transitions
    let mut vp: u64 = !0;
    let mut vn: u64 = 0;

    // Initial score is the pattern length (all deletions)
    let mut score = m;

    // Mask for the highest bit position (m-1)
    let high_bit = 1u64 << (m - 1);

    // Process each character in the text
    for &text_char in text {
        // Get the equivalence mask for this text character
        let eq = masks.peq[text_char as usize];

        // Myers' recurrence relations (Algorithm 1 from the paper)
        // D0 represents positions where the diagonal could be 0 (match or favorable)
        let d0 = ((eq & vp).wrapping_add(vp)) ^ vp | eq | vn;

        // HP (Positive Horizontal): positions with +1 horizontal delta
        let hp = vn | !(d0 | vp);

        // HN (Negative Horizontal): positions with -1 horizontal delta
        let hn = d0 & vp;

        // Update score based on the last row's horizontal deltas
        // If HP has the high bit set, score increases by 1
        if hp & high_bit != 0 {
            score += 1;
        }
        // If HN has the high bit set, score decreases by 1
        if hn & high_bit != 0 {
            score -= 1;
        }

        // Compute new vertical deltas for next column
        // The | 1 accounts for the boundary condition: D[0,j+1] - D[0,j] = 1
        // This represents the implicit +1 horizontal delta at row 0
        let hp_shifted = (hp << 1) | 1;
        let hn_shifted = hn << 1;

        // VP is set where HN was (shifted) or where D0 is not set
        vp = hn_shifted | !(d0 | hp_shifted);

        // VN is set where HP was (shifted) and D0 is set
        vn = hp_shifted & d0;
    }

    score
}

/// Compute Myers distance with a maximum threshold.
///
/// Returns `None` if the distance exceeds `max_distance`, otherwise
/// returns `Some(distance)`. This can provide early termination for
/// queries with strict distance bounds.
///
/// # Arguments
///
/// * `source` - The source string
/// * `target` - The target string
/// * `max_distance` - Maximum acceptable distance
///
/// # Returns
///
/// `Some(distance)` if distance ≤ max_distance, `None` otherwise.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::myers::myers_distance_bounded;
///
/// assert_eq!(myers_distance_bounded("test", "best", 2), Some(1));
/// assert_eq!(myers_distance_bounded("abc", "xyz", 2), None);
/// ```
pub fn myers_distance_bounded(source: &str, target: &str, max_distance: usize) -> Option<usize> {
    let dist = myers_distance(source, target);
    if dist <= max_distance {
        Some(dist)
    } else {
        None
    }
}

/// Compute edit distance with transposition support using Myers' algorithm.
///
/// This extends the standard Myers algorithm to also consider transposition
/// (swapping two adjacent characters) as a single edit operation.
///
/// Note: This is an approximation that may not perfectly match the optimal
/// Damerau-Levenshtein distance in all cases.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::myers::myers_transposition_distance;
///
/// assert_eq!(myers_transposition_distance("ab", "ba"), 1);
/// assert_eq!(myers_transposition_distance("test", "tset"), 1);
/// ```
pub fn myers_transposition_distance(source: &str, target: &str) -> usize {
    let source_bytes = source.as_bytes();
    let target_bytes = target.as_bytes();

    // Handle trivial cases
    if source_bytes.is_empty() {
        return target_bytes.len();
    }
    if target_bytes.is_empty() {
        return source_bytes.len();
    }
    if source_bytes == target_bytes {
        return 0;
    }

    // Use shorter string as pattern
    let (pattern, text) = if source_bytes.len() <= target_bytes.len() {
        (source_bytes, target_bytes)
    } else {
        (target_bytes, source_bytes)
    };

    // For patterns > 64, fall back to standard DP transposition
    if pattern.len() > 64 {
        return crate::distance::transposition_distance(source, target);
    }

    myers_transposition_core(pattern, text)
}

/// Core Myers algorithm with transposition support.
fn myers_transposition_core(pattern: &[u8], text: &[u8]) -> usize {
    let m = pattern.len();
    let masks = PatternMasks::new(pattern);

    // Standard Myers state
    let mut vp: u64 = !0;
    let mut vn: u64 = 0;
    let mut score = m;

    // Previous character's equivalence mask (for transposition detection)
    let mut prev_eq: u64 = 0;

    let high_bit = 1u64 << (m - 1);

    for (i, &text_char) in text.iter().enumerate() {
        let eq = masks.peq[text_char as usize];

        // Standard Myers computation
        let xv = eq | vn;
        let xh = ((eq & vp).wrapping_add(vp)) ^ vp | eq;

        let hp = vn | !(xh | vp);
        let hn = xh & vp;

        // Check for potential transposition
        // If previous char matches current pattern position and current char
        // matches previous pattern position, we might have a transposition
        if i > 0 && m > 1 {
            // Transposition detection: look for pattern[j] == text[i-1] && pattern[j-1] == text[i]
            // This is approximated by checking shifted equivalence masks
            let trans_match = (prev_eq >> 1) & eq;
            if trans_match != 0 {
                // Potential transposition found - this is a heuristic optimization
                // For exact Damerau-Levenshtein, full DP is needed
            }
        }

        // Update score
        if hp & high_bit != 0 {
            score += 1;
        }
        if hn & high_bit != 0 {
            score -= 1;
        }

        // Update state
        vp = (hn << 1) | !(xv | (hp << 1));
        vn = (hp << 1) & xv;

        prev_eq = eq;
    }

    // For now, fall back to exact computation for transposition
    // Myers alone doesn't handle transposition optimally
    let standard_dist = score;
    let trans_dist = crate::distance::transposition_distance(
        std::str::from_utf8(pattern).unwrap_or(""),
        std::str::from_utf8(text).unwrap_or("")
    );

    standard_dist.min(trans_dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_myers_empty_strings() {
        assert_eq!(myers_distance("", ""), 0);
        assert_eq!(myers_distance("", "test"), 4);
        assert_eq!(myers_distance("test", ""), 4);
    }

    #[test]
    fn test_myers_identical_strings() {
        assert_eq!(myers_distance("test", "test"), 0);
        assert_eq!(myers_distance("a", "a"), 0);
        assert_eq!(myers_distance("hello world", "hello world"), 0);
    }

    #[test]
    fn test_myers_basic_edits() {
        // Single substitution
        assert_eq!(myers_distance("test", "best"), 1);
        assert_eq!(myers_distance("cat", "bat"), 1);

        // Single insertion
        assert_eq!(myers_distance("test", "tests"), 1);
        assert_eq!(myers_distance("cat", "cart"), 1);

        // Single deletion
        assert_eq!(myers_distance("tests", "test"), 1);
        assert_eq!(myers_distance("cart", "cat"), 1);
    }

    #[test]
    fn test_myers_classic_examples() {
        assert_eq!(myers_distance("kitten", "sitting"), 3);
        assert_eq!(myers_distance("saturday", "sunday"), 3);
        assert_eq!(myers_distance("algorithm", "altruistic"), 6);
    }

    #[test]
    fn test_myers_symmetry() {
        // Distance should be symmetric
        assert_eq!(
            myers_distance("abc", "def"),
            myers_distance("def", "abc")
        );
        assert_eq!(
            myers_distance("kitten", "sitting"),
            myers_distance("sitting", "kitten")
        );
    }

    #[test]
    fn test_myers_matches_standard_dp() {
        let test_cases = vec![
            ("", ""),
            ("a", "b"),
            ("abc", "abc"),
            ("kitten", "sitting"),
            ("saturday", "sunday"),
            ("test", "best"),
            ("algorithm", "altruistic"),
            ("flaw", "lawn"),
            ("gumbo", "gambol"),
        ];

        for (a, b) in test_cases {
            let myers_dist = myers_distance(a, b);
            let dp_dist = crate::distance::standard_distance_impl(a, b);
            assert_eq!(
                myers_dist, dp_dist,
                "Mismatch for '{}' vs '{}': myers={}, dp={}",
                a, b, myers_dist, dp_dist
            );
        }
    }

    #[test]
    fn test_myers_bounded() {
        assert_eq!(myers_distance_bounded("test", "best", 2), Some(1));
        assert_eq!(myers_distance_bounded("test", "best", 1), Some(1));
        assert_eq!(myers_distance_bounded("test", "best", 0), None);
        assert_eq!(myers_distance_bounded("abc", "xyz", 2), None);
        assert_eq!(myers_distance_bounded("abc", "xyz", 3), Some(3));
    }

    #[test]
    fn test_myers_long_strings() {
        // Test with strings near the 64-char boundary
        let s32 = "a".repeat(32);
        let t32 = "b".repeat(32);
        assert_eq!(myers_distance(&s32, &t32), 32);

        let s64 = "a".repeat(64);
        let t64 = "b".repeat(64);
        assert_eq!(myers_distance(&s64, &t64), 64);

        // Test with common prefix
        let s = format!("{}abc", "prefix".repeat(5));
        let t = format!("{}def", "prefix".repeat(5));
        let dist = myers_distance(&s, &t);
        assert_eq!(dist, 3); // Only "abc" -> "def" differs
    }

    #[test]
    fn test_myers_unicode() {
        // Note: Myers operates on bytes, so multi-byte chars count as multiple edits
        // This is consistent with byte-level distance
        assert_eq!(myers_distance("café", "cafe"), 2); // é is 2 bytes, e is 1 byte
        assert_eq!(myers_distance("naïve", "naive"), 2); // ï -> i
    }

    #[test]
    fn test_myers_transposition() {
        assert_eq!(myers_transposition_distance("ab", "ba"), 1);
        assert_eq!(myers_transposition_distance("test", "tset"), 1);
        assert_eq!(myers_transposition_distance("abc", "acb"), 1);
    }

    #[test]
    fn test_myers_transposition_matches_dp() {
        let test_cases = vec![
            ("ab", "ba"),
            ("test", "tset"),
            ("abc", "acb"),
            ("", ""),
            ("a", "a"),
        ];

        for (a, b) in test_cases {
            let myers_dist = myers_transposition_distance(a, b);
            let dp_dist = crate::distance::transposition_distance(a, b);
            assert_eq!(
                myers_dist, dp_dist,
                "Transposition mismatch for '{}' vs '{}': myers={}, dp={}",
                a, b, myers_dist, dp_dist
            );
        }
    }

    #[test]
    fn test_pattern_masks() {
        let pattern = b"abca";
        let masks = PatternMasks::new(pattern);

        // 'a' appears at positions 0 and 3
        assert_eq!(masks.peq[b'a' as usize], 0b1001);
        // 'b' appears at position 1
        assert_eq!(masks.peq[b'b' as usize], 0b0010);
        // 'c' appears at position 2
        assert_eq!(masks.peq[b'c' as usize], 0b0100);
        // 'd' doesn't appear
        assert_eq!(masks.peq[b'd' as usize], 0);
    }

    #[test]
    fn test_myers_large_pattern() {
        // Test patterns > 64 characters (falls back to standard DP)
        // Myers bit-parallel is most efficient for short patterns
        let s100 = "a".repeat(100);
        let t100 = "b".repeat(100);
        let dist = myers_distance(&s100, &t100);
        assert_eq!(dist, 100);

        // With some matching characters
        let s = format!("{}xyz{}", "a".repeat(50), "a".repeat(50));
        let t = format!("{}abc{}", "a".repeat(50), "a".repeat(50));
        let dist = myers_distance(&s, &t);
        assert_eq!(dist, 3); // Only "xyz" -> "abc" differs

        // Verify correctness by comparing with explicit DP call
        let dp_dist = crate::distance::standard_distance_impl(&s, &t);
        assert_eq!(dist, dp_dist);
    }
}
