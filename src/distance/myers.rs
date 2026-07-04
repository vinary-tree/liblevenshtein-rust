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

use smallvec::SmallVec;

const SHORT_TRANSPOSITION_DP_BYTE_LIMIT: usize = 8;
const STACK_TRANSPOSITION_ROW_LIMIT: usize = 32;

/// Pattern equivalence table for Myers algorithm.
///
/// Pre-computes bitmasks where bit i is set if the pattern character
/// at position i equals the indexed alphabet character.
struct PatternMasks {
    /// Equivalence bitmasks indexed by byte value (0-255)
    peq: [u64; 256],
}

#[inline]
fn byte_mask_index(byte: u8) -> usize {
    usize::from(byte)
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
            peq[byte_mask_index(byte)] |= 1u64 << i;
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

    // For patterns > 64, fall back to byte-level DP. Do not route through
    // `str::from_utf8(...).unwrap_or("")`: this API is explicitly byte-based,
    // and invalid UTF-8 must still produce a real byte edit distance.
    if pattern.len() > 64 {
        return byte_distance_impl(source, target);
    }

    myers_core(pattern, text)
}

/// Compute byte-level Myers distance with a maximum threshold.
///
/// Returns `None` when the byte edit distance is greater than `max_distance`.
/// For short patterns this uses the regular bit-parallel Myers core and checks
/// the result. For longer patterns it switches to a banded dynamic program, so
/// tight thresholds avoid the full `O(mn)` matrix scan.
#[inline]
pub fn myers_distance_bytes_bounded(
    source: &[u8],
    target: &[u8],
    max_distance: usize,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }

    let length_delta = source.len().abs_diff(target.len());
    if length_delta > max_distance {
        return None;
    }

    if source.is_empty() {
        return (target.len() <= max_distance).then_some(target.len());
    }
    if target.is_empty() {
        return (source.len() <= max_distance).then_some(source.len());
    }
    if max_distance == 0 {
        return None;
    }

    let (pattern, text) = if source.len() <= target.len() {
        (source, target)
    } else {
        (target, source)
    };

    if pattern.len() <= 64 {
        let distance = myers_core(pattern, text);
        return (distance <= max_distance).then_some(distance);
    }

    byte_distance_bounded_impl(source, target, max_distance)
}

/// Scalar Levenshtein distance over raw byte slices.
///
/// This is the correctness fallback for [`myers_distance_bytes`] when the
/// shorter input does not fit in a single 64-bit Myers word. It deliberately
/// preserves byte semantics for arbitrary binary data.
fn byte_distance_impl(source: &[u8], target: &[u8]) -> usize {
    let (rows, cols) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let m = rows.len();
    let n = cols.len();

    if n == 0 {
        return m;
    }

    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row = vec![0; n + 1];

    for (i, &row_byte) in rows.iter().enumerate() {
        curr_row[0] = i + 1;

        for (j, &col_byte) in cols.iter().enumerate() {
            let cost = usize::from(row_byte != col_byte);
            let col = j + 1;
            curr_row[col] = (prev_row[col] + 1)
                .min(curr_row[col - 1] + 1)
                .min(prev_row[col - 1] + cost);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

fn byte_distance_bounded_impl(source: &[u8], target: &[u8], max_distance: usize) -> Option<usize> {
    let (rows, cols) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let m = rows.len();
    let n = cols.len();

    if n == 0 {
        return (m <= max_distance).then_some(m);
    }
    if max_distance == usize::MAX {
        return Some(byte_distance_impl(rows, cols));
    }

    let cap = max_distance + 1;
    let mut prev_row = vec![cap; n + 1];
    let mut curr_row = vec![cap; n + 1];

    for (j, cell) in prev_row
        .iter_mut()
        .take(n.min(max_distance) + 1)
        .enumerate()
    {
        *cell = j;
    }

    for i in 1..=m {
        let start = i.saturating_sub(max_distance).max(1);
        let end = n.min(i.saturating_add(max_distance));

        if start > end {
            return None;
        }

        let clear_start = start.saturating_sub(1);
        let clear_end = n.min(end.saturating_add(1));
        for cell in &mut curr_row[clear_start..=clear_end] {
            *cell = cap;
        }

        curr_row[0] = if i <= max_distance { i } else { cap };

        let mut row_min = curr_row[0];
        for j in start..=end {
            let cost = usize::from(rows[i - 1] != cols[j - 1]);
            let best = prev_row[j]
                .saturating_add(1)
                .min(cap)
                .min(curr_row[j - 1].saturating_add(1).min(cap))
                .min(prev_row[j - 1].saturating_add(cost).min(cap));

            curr_row[j] = best;
            row_min = row_min.min(best);
        }

        if row_min > max_distance {
            return None;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    (prev_row[n] <= max_distance).then_some(prev_row[n])
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
        let eq = masks.peq[byte_mask_index(text_char)];

        // Myers' recurrence relations (Algorithm 1 from the paper)
        // D0 represents positions where the diagonal could be 0 (match or favorable).
        // `wrapping_add` is intentional and load-bearing: the bit-parallel recurrence
        // relies on modular (two's-complement) carry propagation across the word, so
        // overflow at the most-significant bit must wrap rather than panic or saturate.
        // This is the crate's only intentional wrapping arithmetic (Phase 1 residual-cast
        // audit): every other integer operation is checked, saturating, or provably bounded.
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
    myers_distance_bytes_bounded(source.as_bytes(), target.as_bytes(), max_distance)
}

/// Compute edit distance with adjacent transposition support.
///
/// The ordinary Myers recurrence does not carry the extra predecessor state
/// needed for exact adjacent transpositions. This function therefore uses the
/// same optimal-string-alignment recurrence as [`crate::distance::transposition_distance`],
/// with fixed-size stack rows for the common short-string case that Myers users
/// typically care about.
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
    if source.is_empty() {
        return target.chars().count();
    }
    if target.is_empty() {
        return source.chars().count();
    }
    if source == target {
        return 0;
    }

    // Criterion measurements show the shared DP is faster for tiny inputs; the
    // optimized row recurrence pulls ahead once strings are larger than this.
    if source.len().max(target.len()) <= SHORT_TRANSPOSITION_DP_BYTE_LIMIT {
        return crate::distance::transposition_distance(source, target);
    }

    let source_chars: SmallVec<[char; STACK_TRANSPOSITION_ROW_LIMIT]> = source.chars().collect();
    let target_chars: SmallVec<[char; STACK_TRANSPOSITION_ROW_LIMIT]> = target.chars().collect();

    transposition_distance_chars(&source_chars, &target_chars)
}

fn transposition_distance_chars(source: &[char], target: &[char]) -> usize {
    if source.is_empty() {
        return target.len();
    }
    if target.is_empty() {
        return source.len();
    }
    if source == target {
        return 0;
    }

    let (rows, cols) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };

    if cols.len() <= STACK_TRANSPOSITION_ROW_LIMIT {
        transposition_distance_chars_stack(rows, cols)
    } else {
        transposition_distance_chars_heap(rows, cols)
    }
}

fn transposition_distance_chars_stack(rows: &[char], cols: &[char]) -> usize {
    let m = rows.len();
    let n = cols.len();
    debug_assert!(n <= STACK_TRANSPOSITION_ROW_LIMIT);

    let mut two_ago = [0usize; STACK_TRANSPOSITION_ROW_LIMIT + 1];
    let mut prev_row = [0usize; STACK_TRANSPOSITION_ROW_LIMIT + 1];
    let mut curr_row = [0usize; STACK_TRANSPOSITION_ROW_LIMIT + 1];

    for (j, cell) in prev_row.iter_mut().take(n + 1).enumerate() {
        *cell = j;
    }

    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let substitution_cost = usize::from(rows[i - 1] != cols[j - 1]);

            let mut best = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + substitution_cost);

            if i > 1 && j > 1 && rows[i - 1] == cols[j - 2] && rows[i - 2] == cols[j - 1] {
                best = best.min(two_ago[j - 2] + 1);
            }

            curr_row[j] = best;
        }

        std::mem::swap(&mut two_ago, &mut prev_row);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

fn transposition_distance_chars_heap(rows: &[char], cols: &[char]) -> usize {
    let m = rows.len();
    let n = cols.len();

    let mut two_ago = vec![0usize; n + 1];
    let mut prev_row = vec![0usize; n + 1];
    let mut curr_row = vec![0usize; n + 1];

    for (j, cell) in prev_row.iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let substitution_cost = usize::from(rows[i - 1] != cols[j - 1]);

            let mut best = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + substitution_cost);

            if i > 1 && j > 1 && rows[i - 1] == cols[j - 2] && rows[i - 2] == cols[j - 1] {
                best = best.min(two_ago[j - 2] + 1);
            }

            curr_row[j] = best;
        }

        std::mem::swap(&mut two_ago, &mut prev_row);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
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
        assert_eq!(myers_distance("abc", "def"), myers_distance("def", "abc"));
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
    fn test_myers_bounded_long_strings_use_thresholded_byte_dp() {
        let source = format!("{}{}", "x".repeat(80), "shared".repeat(20));
        let target = format!("{}{}", "y".repeat(80), "shared".repeat(20));

        assert_eq!(myers_distance_bounded(&source, &target, 79), None);
        assert_eq!(myers_distance_bounded(&source, &target, 80), Some(80));
    }

    #[test]
    fn test_myers_bounded_rejects_by_length_delta() {
        let source = "a".repeat(128);
        let target = "a".repeat(64);

        assert_eq!(myers_distance_bounded(&source, &target, 63), None);
        assert_eq!(myers_distance_bounded(&source, &target, 64), Some(64));
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
            ("日本", "本日"),
            ("café", "cafe"),
            ("naïve", "naïve"),
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
    fn test_myers_transposition_heap_path_matches_dp() {
        let source = format!("{}ab{}", "x".repeat(70), "y".repeat(5));
        let target = format!("{}ba{}", "x".repeat(70), "y".repeat(5));

        assert_eq!(myers_transposition_distance(&source, &target), 1);
        assert_eq!(
            myers_transposition_distance(&source, &target),
            crate::distance::transposition_distance(&source, &target)
        );
    }

    #[test]
    fn test_pattern_masks() {
        let pattern = b"abca";
        let masks = PatternMasks::new(pattern);

        // 'a' appears at positions 0 and 3
        assert_eq!(masks.peq[byte_mask_index(b'a')], 0b1001);
        // 'b' appears at position 1
        assert_eq!(masks.peq[byte_mask_index(b'b')], 0b0010);
        // 'c' appears at position 2
        assert_eq!(masks.peq[byte_mask_index(b'c')], 0b0100);
        // 'd' doesn't appear
        assert_eq!(masks.peq[byte_mask_index(b'd')], 0);
    }

    #[test]
    fn test_pattern_masks_cover_byte_index_boundaries() {
        let pattern = [0x00, 0xff, 0x00, 0xff];
        let masks = PatternMasks::new(&pattern);

        assert_eq!(byte_mask_index(0x00), 0);
        assert_eq!(byte_mask_index(0xff), 255);
        assert_eq!(masks.peq[byte_mask_index(0x00)], 0b0101);
        assert_eq!(masks.peq[byte_mask_index(0xff)], 0b1010);
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

    #[test]
    fn test_myers_long_invalid_utf8_bytes_use_byte_distance() {
        let mut source = vec![0xff; 65];
        let mut target = vec![0xff; 65];
        target[32] = 0xfe;

        assert_eq!(myers_distance_bytes(&source, &target), 1);

        source.push(0x00);
        assert_eq!(myers_distance_bytes(&source, &target), 2);
    }

    #[test]
    fn test_myers_bounded_long_invalid_utf8_bytes_use_byte_distance() {
        let mut source = vec![0xff; 65];
        let mut target = vec![0xff; 65];
        target[32] = 0xfe;

        assert_eq!(myers_distance_bytes_bounded(&source, &target, 0), None);
        assert_eq!(myers_distance_bytes_bounded(&source, &target, 1), Some(1));

        source.push(0x00);
        assert_eq!(myers_distance_bytes_bounded(&source, &target, 1), None);
        assert_eq!(myers_distance_bytes_bounded(&source, &target, 2), Some(2));
    }

    #[test]
    fn test_myers_bounded_long_byte_path_matches_unbounded() {
        let insertion_source = vec![b'a'; 96];
        let mut insertion_target = insertion_source.clone();
        insertion_target.splice(48..48, [b'b', b'c', b'd']);

        let cases = vec![
            (vec![b'a'; 80], vec![b'a'; 80]),
            (
                {
                    let mut v = vec![b'a'; 80];
                    v[40] = b'b';
                    v
                },
                vec![b'a'; 80],
            ),
            (vec![b'x'; 96], vec![b'y'; 88]),
            (insertion_source, insertion_target),
            (
                [vec![0xff; 70], vec![0x00, 0x01, 0x02]].concat(),
                [vec![0xff; 70], vec![0x00, 0x02]].concat(),
            ),
        ];

        for (source, target) in cases {
            let distance = myers_distance_bytes(&source, &target);
            for max_distance in 0..=distance + 2 {
                let expected = (distance <= max_distance).then_some(distance);
                assert_eq!(
                    myers_distance_bytes_bounded(&source, &target, max_distance),
                    expected,
                    "bounded mismatch for lengths {} and {} at threshold {}",
                    source.len(),
                    target.len(),
                    max_distance
                );
            }
        }
    }
}
