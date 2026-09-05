//! Distance metric implementations.
//!
//! This module provides various Levenshtein distance implementations
//! for direct distance computation between two strings.
//!
//! Two implementation styles are available:
//! - **Iterative DP**: Space-optimized dynamic programming (2-3 rows)
//! - **Recursive + Memoization**: C++-style recursive approach with caching
//! - **SIMD-accelerated**: AVX2/SSE4.1 vectorized implementations (optional)

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(not(feature = "eviction-dashmap"))]
use crate::sync_compat::RwLock;

#[cfg(feature = "eviction-dashmap")]
use dashmap::DashMap;

#[cfg(not(feature = "eviction-dashmap"))]
use rustc_hash::FxHashMap;

use smallvec::SmallVec;

#[cfg(target_arch = "x86_64")]
pub mod simd;

mod affine;
mod hamming;
mod indel;
pub mod myers;
mod units;

pub use affine::{affine_gap_distance, affine_gap_distance_units};
pub use hamming::{hamming_distance, hamming_distance_units};
pub use indel::{indel_distance, indel_distance_bounded};
pub use units::{
    damerau_levenshtein_distance_units, damerau_levenshtein_distance_units_bounded,
    merge_and_split_distance_units, merge_and_split_distance_units_bounded,
    standard_distance_units, standard_distance_units_bounded, transposition_distance_units,
    transposition_distance_units_bounded,
};

/// A symmetric pair of strings for use as cache keys.
///
/// Ensures that `(a, b)` and `(b, a)` are treated as identical keys,
/// leveraging the symmetric property of distance functions: `d(a,b) == d(b,a)`.
///
/// Strings are ordered lexicographically and stored as `Arc<str>` for
/// efficient cloning and memory sharing.
#[derive(Clone, Debug)]
struct SymmetricPair {
    first: Arc<str>,
    second: Arc<str>,
}

impl SymmetricPair {
    /// Create a new SymmetricPair, ordering strings lexicographically.
    #[inline(always)]
    fn new(a: &str, b: &str) -> Self {
        match a.cmp(b) {
            Ordering::Less | Ordering::Equal => Self {
                first: Arc::from(a),
                second: Arc::from(b),
            },
            Ordering::Greater => Self {
                first: Arc::from(b),
                second: Arc::from(a),
            },
        }
    }
}

impl PartialEq for SymmetricPair {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first && self.second == other.second
    }
}

impl Eq for SymmetricPair {}

impl Hash for SymmetricPair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.first.hash(state);
        self.second.hash(state);
    }
}

#[inline(always)]
fn split_first_char(s: &str) -> Option<(char, &str)> {
    let mut char_indices = s.char_indices();
    let (_, first) = char_indices.next()?;
    let tail = char_indices
        .next()
        .map_or("", |(byte_idx, _)| &s[byte_idx..]);
    Some((first, tail))
}

#[derive(Debug, Clone, Copy)]
struct CommonAffixSlices<'a> {
    prefix_len: usize,
    source_len: usize,
    target_len: usize,
    source_core: &'a str,
    target_core: &'a str,
}

#[inline(always)]
fn strip_common_affix_slices<'a>(a: &'a str, b: &'a str) -> CommonAffixSlices<'a> {
    let a_chars: SmallVec<[(usize, char); 32]> = a.char_indices().collect();
    let b_chars: SmallVec<[(usize, char); 32]> = b.char_indices().collect();

    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 || len_b == 0 {
        return CommonAffixSlices {
            prefix_len: 0,
            source_len: len_a,
            target_len: len_b,
            source_core: a,
            target_core: b,
        };
    }

    let mut prefix_len = 0;
    let min_len = len_a.min(len_b);
    while prefix_len < min_len && a_chars[prefix_len].1 == b_chars[prefix_len].1 {
        prefix_len += 1;
    }

    if prefix_len == min_len {
        return CommonAffixSlices {
            prefix_len,
            source_len: len_a - prefix_len,
            target_len: len_b - prefix_len,
            source_core: slice_char_range(a, &a_chars, prefix_len, len_a),
            target_core: slice_char_range(b, &b_chars, prefix_len, len_b),
        };
    }

    let mut suffix_len = 0;
    while suffix_len < (min_len - prefix_len)
        && a_chars[len_a - 1 - suffix_len].1 == b_chars[len_b - 1 - suffix_len].1
    {
        suffix_len += 1;
    }

    let source_end = len_a - suffix_len;
    let target_end = len_b - suffix_len;

    CommonAffixSlices {
        prefix_len,
        source_len: source_end - prefix_len,
        target_len: target_end - prefix_len,
        source_core: slice_char_range(a, &a_chars, prefix_len, source_end),
        target_core: slice_char_range(b, &b_chars, prefix_len, target_end),
    }
}

#[inline(always)]
fn slice_char_range<'a>(
    source: &'a str,
    char_indices: &[(usize, char)],
    start: usize,
    end: usize,
) -> &'a str {
    let start_byte = char_indices
        .get(start)
        .map(|(byte_idx, _)| *byte_idx)
        .unwrap_or(source.len());
    let end_byte = char_indices
        .get(end)
        .map(|(byte_idx, _)| *byte_idx)
        .unwrap_or(source.len());
    &source[start_byte..end_byte]
}

/// Strip common prefix and suffix from two strings.
///
/// Returns `(start_offset, adjusted_len_a, adjusted_len_b)` where:
/// - `start_offset`: Number of common prefix characters
/// - `adjusted_len_a`: Length of first string minus common prefix/suffix
/// - `adjusted_len_b`: Length of second string minus common prefix/suffix
///
/// This optimization significantly speeds up distance computation for
/// strings with substantial overlap.
#[inline(always)]
pub fn strip_common_affixes(a: &str, b: &str) -> (usize, usize, usize) {
    let slices = strip_common_affix_slices(a, b);
    (slices.prefix_len, slices.source_len, slices.target_len)
}

/// Thread-safe memoization cache for distance functions.
///
/// Uses either `DashMap` (lock-free, feature "eviction-dashmap") or
/// `RwLock<FxHashMap>` (fast hash) for concurrent access to cached distance results.
pub struct MemoCache {
    #[cfg(feature = "eviction-dashmap")]
    cache: DashMap<SymmetricPair, usize>,

    #[cfg(not(feature = "eviction-dashmap"))]
    cache: RwLock<FxHashMap<SymmetricPair, usize>>,
}

impl MemoCache {
    fn new() -> Self {
        Self {
            #[cfg(feature = "eviction-dashmap")]
            cache: DashMap::new(),

            #[cfg(not(feature = "eviction-dashmap"))]
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    fn get(&self, key: &SymmetricPair) -> Option<usize> {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.cache.get(key).map(|entry| *entry)
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            self.cache.read().get(key).copied()
        }
    }

    fn insert(&self, key: SymmetricPair, value: usize) {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.cache.insert(key, value);
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            self.cache.write().insert(key, value);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.cache.len()
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            self.cache.read().len()
        }
    }
}

/// Compute standard Levenshtein distance between two strings.
///
/// Uses dynamic programming to compute the minimum number of
/// single-character edits (insertions, deletions, substitutions)
/// required to transform `source` into `target`.
///
/// This function automatically selects the optimal algorithm:
/// - **Myers' bit-parallel**: For strings where both are ≤64 bytes (O(mn/64) time)
/// - **SIMD-vectorized DP**: For longer strings when `simd` feature is enabled
/// - **Scalar DP**: Fallback for longer strings without SIMD
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::standard_distance;
///
/// assert_eq!(standard_distance("kitten", "sitting"), 3);
/// assert_eq!(standard_distance("test", "test"), 0);
/// ```
pub fn standard_distance(source: &str, target: &str) -> usize {
    let source_len = source.len();
    let target_len = target.len();

    // Myers' bit-parallel is optimal for short ASCII strings (≤64 bytes)
    // It processes 64 positions per 64-bit word operation
    // Note: Myers operates on bytes, so we only use it for ASCII to maintain
    // character-based semantics for Unicode strings
    if source_len <= 64 && target_len <= 64 && source.is_ascii() && target.is_ascii() {
        return myers::myers_distance(source, target);
    }

    #[cfg(target_arch = "x86_64")]
    {
        simd::standard_distance_simd(source, target)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        standard_distance_impl(source, target)
    }
}

/// Scalar implementation of standard Levenshtein distance.
///
/// This is the non-SIMD version that always uses scalar operations.
/// Public for benchmarking and testing purposes.
pub fn standard_distance_impl(source: &str, target: &str) -> usize {
    let source_chars: SmallVec<[char; 32]> = source.chars().collect();
    let target_chars: SmallVec<[char; 32]> = target.chars().collect();
    standard_distance_units(&source_chars, &target_chars)
}

/// Compute standard Levenshtein distance up to a maximum threshold.
///
/// Returns `None` as soon as the distance is proven to exceed `max_distance`.
/// Unlike the byte-oriented Myers bounded helper, this function preserves the
/// crate's public Unicode character semantics.
pub fn standard_distance_bounded(source: &str, target: &str, max_distance: usize) -> Option<usize> {
    if source == target {
        return Some(0);
    }

    let affixes = strip_common_affix_slices(source, target);
    if affixes.source_len.abs_diff(affixes.target_len) > max_distance {
        return None;
    }
    if affixes.source_len == 0 {
        return (affixes.target_len <= max_distance).then_some(affixes.target_len);
    }
    if affixes.target_len == 0 {
        return (affixes.source_len <= max_distance).then_some(affixes.source_len);
    }
    if max_distance == 0 {
        return None;
    }
    if max_distance == usize::MAX {
        return Some(standard_distance(affixes.source_core, affixes.target_core));
    }

    let source_chars: SmallVec<[char; 32]> = affixes.source_core.chars().collect();
    let target_chars: SmallVec<[char; 32]> = affixes.target_core.chars().collect();
    standard_distance_units_bounded(&source_chars, &target_chars, max_distance)
}

/// Compute optimal string alignment (OSA) distance.
///
/// Extends standard Levenshtein distance to also consider transposition
/// (swapping two adjacent characters) as a single edit operation.
/// OSA is also called *restricted Damerau distance*: a substring may be edited
/// at most once. It differs from unrestricted Damerau–Levenshtein distance and
/// does not satisfy the triangle inequality.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::transposition_distance;
///
/// assert_eq!(transposition_distance("ab", "ba"), 1); // One transposition
/// assert_eq!(transposition_distance("test", "tset"), 1); // One transposition
/// ```
pub fn transposition_distance(source: &str, target: &str) -> usize {
    let source_chars: SmallVec<[char; 32]> = source.chars().collect();
    let target_chars: SmallVec<[char; 32]> = target.chars().collect();
    transposition_distance_units(&source_chars, &target_chars)
}

/// Compute unrestricted Damerau–Levenshtein distance.
///
/// This is the full Lowrance–Wagner dynamic program with a last-occurrence
/// table over the union alphabet. Unlike [`transposition_distance`], an edit
/// may act on a substring changed by an earlier edit. The implementation keeps
/// the complete matrix deliberately: it is the obvious reference oracle for
/// the streaming automaton, not a windowed production shortcut.
///
/// # Examples
///
/// ```rust
/// use liblevenshtein::distance::damerau_levenshtein_distance;
///
/// assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
/// assert_eq!(damerau_levenshtein_distance("CA", "ABC"), 2);
/// ```
pub fn damerau_levenshtein_distance(source: &str, target: &str) -> usize {
    let source: SmallVec<[char; 32]> = source.chars().collect();
    let target: SmallVec<[char; 32]> = target.chars().collect();
    damerau_levenshtein_distance_units(&source, &target)
}

/// Compute unrestricted Damerau–Levenshtein distance up to a threshold.
///
/// The length difference is an admissible constant-time rejection. Remaining
/// cases use the full reference recurrence and retain the result only when it
/// is within `max_distance`.
pub fn damerau_levenshtein_distance_bounded(
    source: &str,
    target: &str,
    max_distance: usize,
) -> Option<usize> {
    let source: SmallVec<[char; 32]> = source.chars().collect();
    let target: SmallVec<[char; 32]> = target.chars().collect();
    damerau_levenshtein_distance_units_bounded(&source, &target, max_distance)
}

/// Compute optimal string alignment distance up to a maximum threshold.
///
/// This uses the same optimal-string-alignment recurrence as
/// [`transposition_distance`] and returns `None` when the distance is proven to
/// exceed `max_distance`.
pub fn transposition_distance_bounded(
    source: &str,
    target: &str,
    max_distance: usize,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }

    let affixes = strip_common_affix_slices(source, target);
    if affixes.source_len.abs_diff(affixes.target_len) > max_distance {
        return None;
    }
    if affixes.source_len == 0 {
        return (affixes.target_len <= max_distance).then_some(affixes.target_len);
    }
    if affixes.target_len == 0 {
        return (affixes.source_len <= max_distance).then_some(affixes.source_len);
    }
    if max_distance == 0 {
        return None;
    }
    if max_distance == usize::MAX {
        return Some(transposition_distance(
            affixes.source_core,
            affixes.target_core,
        ));
    }

    let source_chars: SmallVec<[char; 32]> = affixes.source_core.chars().collect();
    let target_chars: SmallVec<[char; 32]> = affixes.target_core.chars().collect();
    transposition_distance_units_bounded(&source_chars, &target_chars, max_distance)
}

// ============================================================================
// Recursive Memoized Implementations (C++-style)
// ============================================================================

/// Recursive standard Levenshtein distance with memoization and optimizations.
///
/// This implementation mirrors the C++ recursive approach with:
/// - Thread-safe memoization cache
/// - Common prefix/suffix stripping
/// - Early exit optimizations
///
/// Best for scenarios with many repeated distance queries on similar strings.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::standard_distance_recursive;
///
/// let cache = liblevenshtein::distance::create_memo_cache();
/// assert_eq!(standard_distance_recursive("kitten", "sitting", &cache), 3);
/// assert_eq!(standard_distance_recursive("test", "test", &cache), 0);
/// ```
pub fn standard_distance_recursive(source: &str, target: &str, cache: &MemoCache) -> usize {
    // Check cache first
    let cache_key = SymmetricPair::new(source, target);
    if let Some(distance) = cache.get(&cache_key) {
        return distance;
    }

    // Handle base cases
    if source.is_empty() {
        return target.chars().count();
    }
    if target.is_empty() {
        return source.chars().count();
    }

    // Strip common prefix and suffix (major optimization)
    let affixes = strip_common_affix_slices(source, target);

    // If strings are identical after stripping, distance is 0
    if affixes.source_len == 0 && affixes.target_len == 0 {
        cache.insert(cache_key, 0);
        return 0;
    }

    // If one string is fully consumed, distance is remaining chars in other
    if affixes.source_len == 0 {
        let result = affixes.target_len;
        cache.insert(cache_key, result);
        return result;
    }
    if affixes.target_len == 0 {
        let result = affixes.source_len;
        cache.insert(cache_key, result);
        return result;
    }

    let s_remaining = affixes.source_core;
    let t_remaining = affixes.target_core;
    let Some((a, s)) = split_first_char(s_remaining) else {
        let result = affixes.target_len;
        cache.insert(cache_key, result);
        return result;
    };
    let Some((b, t)) = split_first_char(t_remaining) else {
        let result = affixes.source_len;
        cache.insert(cache_key, result);
        return result;
    };

    let mut distance;

    if a == b {
        // Characters match - no cost
        distance = standard_distance_recursive(s, t, cache);

        // Early exit optimization
        if distance == 0 {
            cache.insert(cache_key, distance);
            return distance;
        }
    } else {
        // Characters differ - try all three operations

        // Deletion: advance source
        distance = standard_distance_recursive(s, t_remaining, cache);

        // Early exit
        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        // Insertion: advance target
        let ins_dist = standard_distance_recursive(s_remaining, t, cache);
        distance = distance.min(ins_dist);

        // Early exit
        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        // Substitution: advance both
        let sub_dist = standard_distance_recursive(s, t, cache);
        distance = distance.min(sub_dist);

        distance += 1; // Cost of operation
    }

    cache.insert(cache_key, distance);
    distance
}

/// Recursive transposition distance with memoization.
///
/// Extends standard Levenshtein to support transposition (swapping adjacent chars)
/// as a single operation. Uses the same recursive + memoization approach.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::transposition_distance_recursive;
///
/// let cache = liblevenshtein::distance::create_memo_cache();
/// assert_eq!(transposition_distance_recursive("ab", "ba", &cache), 1);
/// assert_eq!(transposition_distance_recursive("test", "tset", &cache), 1);
/// ```
pub fn transposition_distance_recursive(source: &str, target: &str, cache: &MemoCache) -> usize {
    // Check cache first
    let cache_key = SymmetricPair::new(source, target);
    if let Some(distance) = cache.get(&cache_key) {
        return distance;
    }

    // Handle base cases
    if source.is_empty() {
        return target.chars().count();
    }
    if target.is_empty() {
        return source.chars().count();
    }

    // Strip common prefix and suffix (major optimization)
    let affixes = strip_common_affix_slices(source, target);

    // If strings are identical after stripping, distance is 0
    if affixes.source_len == 0 && affixes.target_len == 0 {
        cache.insert(cache_key, 0);
        return 0;
    }

    // If one string is fully consumed, distance is remaining chars in other
    if affixes.source_len == 0 {
        let result = affixes.target_len;
        cache.insert(cache_key, result);
        return result;
    }
    if affixes.target_len == 0 {
        let result = affixes.source_len;
        cache.insert(cache_key, result);
        return result;
    }

    let s_remaining = affixes.source_core;
    let t_remaining = affixes.target_core;
    let Some((a, s)) = split_first_char(s_remaining) else {
        let result = affixes.target_len;
        cache.insert(cache_key, result);
        return result;
    };
    let Some((b, t)) = split_first_char(t_remaining) else {
        let result = affixes.source_len;
        cache.insert(cache_key, result);
        return result;
    };

    let mut distance;

    if a == b {
        distance = transposition_distance_recursive(s, t, cache);

        if distance == 0 {
            cache.insert(cache_key, distance);
            return distance;
        }
    } else {
        // Standard operations: deletion, insertion, substitution
        distance = transposition_distance_recursive(s, t_remaining, cache);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let ins_dist = transposition_distance_recursive(s_remaining, t, cache);
        distance = distance.min(ins_dist);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let sub_dist = transposition_distance_recursive(s, t, cache);
        distance = distance.min(sub_dist);

        // Check for transposition
        // Requires at least 2 chars remaining in both strings
        if let (Some((a1, ss)), Some((b1, tt))) = (split_first_char(s), split_first_char(t)) {
            // Transposition: source[0] == target[1] && source[1] == target[0]
            if a == b1 && a1 == b {
                let trans_dist = transposition_distance_recursive(ss, tt, cache);
                distance = distance.min(trans_dist);
            }
        }

        distance += 1;
    }

    cache.insert(cache_key, distance);
    distance
}

/// Iterative, stack-safe merge-and-split distance with memoization.
///
/// Supports merge (two query chars → one dict char) and split
/// (one query char → two dict chars) operations, in addition to
/// standard Levenshtein operations.
///
/// This is useful for OCR errors, phonetic matching, and other
/// specialized scenarios.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::merge_and_split_distance;
///
/// let cache = liblevenshtein::distance::create_memo_cache();
/// // "m" → "rn" (split) is one operation
/// assert_eq!(merge_and_split_distance("m", "rn", &cache), 1);
/// // "rn" → "m" (merge) is one operation
/// assert_eq!(merge_and_split_distance("rn", "m", &cache), 1);
/// ```
pub fn merge_and_split_distance(source: &str, target: &str, cache: &MemoCache) -> usize {
    let cache_key = SymmetricPair::new(source, target);
    if let Some(distance) = cache.get(&cache_key) {
        return distance;
    }
    let source_units: SmallVec<[char; 32]> = source.chars().collect();
    let target_units: SmallVec<[char; 32]> = target.chars().collect();
    let distance = merge_and_split_distance_units(&source_units, &target_units);
    cache.insert(cache_key, distance);
    distance
}

/// Compute Unicode-scalar merge-and-split distance within an inclusive bound.
///
/// This stack-safe, banded implementation does not consult a memo cache. It
/// returns `None` when the exact distance is greater than `max_distance`.
pub fn merge_and_split_distance_bounded(
    source: &str,
    target: &str,
    max_distance: usize,
) -> Option<usize> {
    let source_units: SmallVec<[char; 32]> = source.chars().collect();
    let target_units: SmallVec<[char; 32]> = target.chars().collect();
    merge_and_split_distance_units_bounded(&source_units, &target_units, max_distance)
}

/// Create a new memoization cache for recursive distance functions.
///
/// The cache is thread-safe and can be shared across multiple distance
/// computations. Reusing a cache can significantly improve performance
/// when computing distances for many string pairs.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::distance::{create_memo_cache, standard_distance_recursive};
///
/// let cache = create_memo_cache();
/// let d1 = standard_distance_recursive("test", "best", &cache);
/// let d2 = standard_distance_recursive("test", "rest", &cache);
/// // Subsequent calls benefit from cached results
/// ```
pub fn create_memo_cache() -> MemoCache {
    MemoCache::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_distance_identical() {
        assert_eq!(standard_distance("test", "test"), 0);
        assert_eq!(standard_distance("", ""), 0);
    }

    #[test]
    fn test_standard_distance_empty() {
        assert_eq!(standard_distance("", "test"), 4);
        assert_eq!(standard_distance("test", ""), 4);
    }

    #[test]
    fn test_standard_distance_basic() {
        assert_eq!(standard_distance("kitten", "sitting"), 3);
        assert_eq!(standard_distance("saturday", "sunday"), 3);
        assert_eq!(standard_distance("test", "best"), 1);
    }

    #[test]
    fn test_standard_distance_bounded_matches_exact() {
        let cases = [
            ("", ""),
            ("", "test"),
            ("kitten", "sitting"),
            ("saturday", "sunday"),
            ("café", "cafe"),
            ("préABCΩ", "préXYZΩ"),
        ];

        for (source, target) in cases {
            let exact = standard_distance(source, target);
            for threshold in 0..=exact + 1 {
                assert_eq!(
                    standard_distance_bounded(source, target, threshold),
                    (exact <= threshold).then_some(exact),
                    "bounded standard mismatch for '{source}' vs '{target}' at {threshold}"
                );
            }
        }
    }

    #[test]
    fn test_transposition_distance() {
        assert_eq!(transposition_distance("ab", "ba"), 1);
        assert_eq!(transposition_distance("test", "tset"), 1);
        assert_eq!(transposition_distance("abc", "acb"), 1);
    }

    #[test]
    fn test_transposition_distance_bounded_matches_exact() {
        let cases = [
            ("", ""),
            ("", "test"),
            ("ab", "ba"),
            ("test", "tset"),
            ("kitten", "sitting"),
            ("préabΩ", "prébaΩ"),
        ];

        for (source, target) in cases {
            let exact = transposition_distance(source, target);
            for threshold in 0..=exact + 1 {
                assert_eq!(
                    transposition_distance_bounded(source, target, threshold),
                    (exact <= threshold).then_some(exact),
                    "bounded transposition mismatch for '{source}' vs '{target}' at {threshold}"
                );
            }
        }
    }

    #[test]
    fn test_transposition_vs_standard() {
        // Transposition should be cheaper than standard for swaps
        let trans_dist = transposition_distance("test", "tset");
        let std_dist = standard_distance("test", "tset");

        assert_eq!(trans_dist, 1);
        assert_eq!(std_dist, 2); // Standard requires 2 substitutions
    }

    // Tests for recursive memoized implementations

    #[test]
    fn test_standard_distance_recursive_basic() {
        let cache = create_memo_cache();
        assert_eq!(standard_distance_recursive("kitten", "sitting", &cache), 3);
        assert_eq!(standard_distance_recursive("saturday", "sunday", &cache), 3);
        assert_eq!(standard_distance_recursive("test", "best", &cache), 1);
    }

    #[test]
    fn test_standard_distance_recursive_identical() {
        let cache = create_memo_cache();
        assert_eq!(standard_distance_recursive("test", "test", &cache), 0);
        assert_eq!(standard_distance_recursive("", "", &cache), 0);
    }

    #[test]
    fn test_standard_distance_recursive_empty() {
        let cache = create_memo_cache();
        assert_eq!(standard_distance_recursive("", "test", &cache), 4);
        assert_eq!(standard_distance_recursive("test", "", &cache), 4);
    }

    #[test]
    fn test_standard_recursive_matches_iterative() {
        let cache = create_memo_cache();
        let test_cases = vec![
            ("", ""),
            ("a", "b"),
            ("abc", "abc"),
            ("kitten", "sitting"),
            ("saturday", "sunday"),
            ("test", "best"),
            ("algorithm", "altruistic"),
        ];

        for (a, b) in test_cases {
            assert_eq!(
                standard_distance_recursive(a, b, &cache),
                standard_distance(a, b),
                "Mismatch for '{}' vs '{}'",
                a,
                b
            );
        }
    }

    #[test]
    fn test_transposition_distance_recursive_basic() {
        let cache = create_memo_cache();
        assert_eq!(transposition_distance_recursive("ab", "ba", &cache), 1);
        assert_eq!(transposition_distance_recursive("test", "tset", &cache), 1);
        assert_eq!(transposition_distance_recursive("abc", "acb", &cache), 1);
    }

    #[test]
    fn test_transposition_recursive_matches_iterative() {
        let cache = create_memo_cache();
        let test_cases = vec![
            ("", ""),
            ("a", "b"),
            ("ab", "ba"),
            ("test", "tset"),
            ("abc", "acb"),
            ("kitten", "sitting"),
        ];

        for (a, b) in test_cases {
            assert_eq!(
                transposition_distance_recursive(a, b, &cache),
                transposition_distance(a, b),
                "Mismatch for '{}' vs '{}'",
                a,
                b
            );
        }
    }

    #[test]
    fn test_merge_and_split_distance_basic() {
        let cache = create_memo_cache();

        // Basic operations
        assert_eq!(merge_and_split_distance("", "", &cache), 0);
        assert_eq!(merge_and_split_distance("a", "a", &cache), 0);
        assert_eq!(merge_and_split_distance("", "test", &cache), 4);
        assert_eq!(merge_and_split_distance("test", "", &cache), 4);

        // Standard operations should work
        assert_eq!(merge_and_split_distance("test", "best", &cache), 1); // substitution

        // Merge and split should be cheaper than standard operations
        // But we need to verify the actual behavior matches C++
    }

    #[test]
    fn test_merge_and_split_symmetry() {
        let cache = create_memo_cache();
        // Distance should be symmetric
        assert_eq!(
            merge_and_split_distance("abc", "def", &cache),
            merge_and_split_distance("def", "abc", &cache)
        );
    }

    #[test]
    fn test_cache_reuse() {
        let cache = create_memo_cache();

        // First call should populate cache
        let d1 = standard_distance_recursive("test", "best", &cache);
        assert_eq!(d1, 1);

        // Second call should use cache (same result)
        let d2 = standard_distance_recursive("test", "best", &cache);
        assert_eq!(d2, 1);

        // Symmetric call should also use cache
        let d3 = standard_distance_recursive("best", "test", &cache);
        assert_eq!(d3, 1);

        // Cache should have at least one entry
        assert!(cache.len() >= 1);
    }

    #[test]
    fn test_common_prefix_optimization() {
        let cache = create_memo_cache();

        // Strings with long common prefix
        let s1 = "commonprefix_abc";
        let s2 = "commonprefix_def";

        let distance = standard_distance_recursive(s1, s2, &cache);
        // Should only need to compute distance on differing part
        assert_eq!(distance, 3); // "abc" -> "def"
    }

    #[test]
    fn test_common_affix_slices_unicode_core() {
        let slices = strip_common_affix_slices("préABCΩ", "préXYZΩ");

        assert_eq!(slices.prefix_len, 3);
        assert_eq!(slices.source_len, 3);
        assert_eq!(slices.target_len, 3);
        assert_eq!(slices.source_core, "ABC");
        assert_eq!(slices.target_core, "XYZ");
    }

    #[test]
    fn test_recursive_split_helper_preserves_utf8_boundaries() {
        assert_eq!(split_first_char("éclair"), Some(('é', "clair")));
        assert_eq!(split_first_char("Ω"), Some(('Ω', "")));
        assert_eq!(split_first_char(""), None);
    }

    #[test]
    fn test_unicode_support() {
        let cache = create_memo_cache();

        // Unicode characters
        assert_eq!(standard_distance_recursive("café", "cafe", &cache), 1);
        assert_eq!(standard_distance_recursive("日本", "日本", &cache), 0);
        assert_eq!(transposition_distance_recursive("日本", "本日", &cache), 1);
    }

    #[test]
    fn test_recursive_distances_use_unicode_affix_boundaries() {
        let standard_cache = create_memo_cache();
        assert_eq!(
            standard_distance_recursive("préABCΩ", "préXYZΩ", &standard_cache),
            3
        );

        let transposition_cache = create_memo_cache();
        assert_eq!(
            transposition_distance_recursive("préabΩ", "prébaΩ", &transposition_cache),
            1
        );

        let merge_split_cache = create_memo_cache();
        assert_eq!(
            merge_and_split_distance("prézΩ", "préxyΩ", &merge_split_cache),
            1
        );
    }

    #[test]
    fn test_unicode_empty_string() {
        let cache = create_memo_cache();

        let a = "";
        let b = "¡";

        let iterative = standard_distance(a, b);
        let recursive = standard_distance_recursive(a, b, &cache);

        eprintln!(
            "String b: '{}', bytes: {}, chars: {}",
            b,
            b.len(),
            b.chars().count()
        );
        eprintln!("Iterative: {}, Recursive: {}", iterative, recursive);

        assert_eq!(
            recursive, iterative,
            "Unicode distance mismatch for empty string"
        );
    }
}
