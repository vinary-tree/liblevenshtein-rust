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
use std::sync::RwLock;

#[cfg(feature = "eviction-dashmap")]
use dashmap::DashMap;

#[cfg(not(feature = "eviction-dashmap"))]
use rustc_hash::FxHashMap;

use smallvec::SmallVec;

#[cfg(feature = "simd")]
pub mod simd;

pub mod myers;

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

/// Helper function to extract substring from position t+1 onwards (by characters, not bytes).
/// Mirrors the C++ `f(u, t)` helper function.
///
/// # Arguments
/// * `s` - The string to extract from
/// * `char_offset` - Number of characters to skip from the beginning
#[inline(always)]
fn substring_from(s: &str, char_offset: usize) -> &str {
    // Find the byte index of the character at char_offset
    let mut char_indices = s.char_indices();

    // Iterate to find the byte position after skipping char_offset characters
    for _ in 0..char_offset {
        if char_indices.next().is_none() {
            // Not enough characters, return empty string
            return "";
        }
    }

    // The next char_indices.next() would give us the character at char_offset
    // But we want everything from char_offset onwards
    match char_indices.next() {
        Some((byte_idx, _)) => &s[byte_idx..],
        None => "", // Reached end of string
    }
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
    let a_chars: SmallVec<[char; 32]> = a.chars().collect();
    let b_chars: SmallVec<[char; 32]> = b.chars().collect();

    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 || len_b == 0 {
        return (0, len_a, len_b);
    }

    // Find common prefix
    let mut prefix_len = 0;
    let min_len = len_a.min(len_b);
    while prefix_len < min_len && a_chars[prefix_len] == b_chars[prefix_len] {
        prefix_len += 1;
    }

    if prefix_len == min_len {
        // One string is a prefix of the other
        return (prefix_len, len_a - prefix_len, len_b - prefix_len);
    }

    // Find common suffix (but don't overlap with prefix)
    let mut suffix_len = 0;
    while suffix_len < (min_len - prefix_len)
        && a_chars[len_a - 1 - suffix_len] == b_chars[len_b - 1 - suffix_len]
    {
        suffix_len += 1;
    }

    (
        prefix_len,
        len_a - prefix_len - suffix_len,
        len_b - prefix_len - suffix_len,
    )
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
            self.cache.read().expect("poisoned RwLock; only fatal if writer panicked").get(key).copied()
        }
    }

    fn insert(&self, key: SymmetricPair, value: usize) {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.cache.insert(key, value);
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            self.cache.write().expect("poisoned RwLock; only fatal if writer panicked").insert(key, value);
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
            self.cache.read().expect("poisoned RwLock; only fatal if writer panicked").len()
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

    #[cfg(feature = "simd")]
    {
        simd::standard_distance_simd(source, target)
    }

    #[cfg(not(feature = "simd"))]
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

    let m = source_chars.len();
    let n = target_chars.len();

    // Handle edge cases
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use space-optimized version (two rows instead of full matrix)
    let mut prev_row = vec![0; n + 1];
    let mut curr_row = vec![0; n + 1];

    // Initialize first row
    for (j, item) in prev_row.iter_mut().enumerate().take(n + 1) {
        *item = j;
    }

    // Fill the matrix
    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let cost = if source_chars[i - 1] == target_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

/// Compute Levenshtein distance with transposition support.
///
/// Extends standard Levenshtein distance to also consider transposition
/// (swapping two adjacent characters) as a single edit operation.
/// This is also known as Damerau-Levenshtein distance.
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

    let m = source_chars.len();
    let n = target_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Need three rows for transposition
    let mut two_ago = vec![0; n + 1];
    let mut prev_row = vec![0; n + 1];
    let mut curr_row = vec![0; n + 1];

    // Initialize first row
    for (j, item) in prev_row.iter_mut().enumerate().take(n + 1) {
        *item = j;
    }

    // Fill the matrix
    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let cost = if source_chars[i - 1] == target_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution

            // Check for transposition
            if i > 1
                && j > 1
                && source_chars[i - 1] == target_chars[j - 2]
                && source_chars[i - 2] == target_chars[j - 1]
            {
                curr_row[j] = curr_row[j].min(two_ago[j - 2] + 1);
            }
        }

        // Rotate rows
        std::mem::swap(&mut two_ago, &mut prev_row);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
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
    let (prefix_len, adjusted_source_len, adjusted_target_len) =
        strip_common_affixes(source, target);

    // If strings are identical after stripping, distance is 0
    if adjusted_source_len == 0 && adjusted_target_len == 0 {
        cache.insert(cache_key, 0);
        return 0;
    }

    // If one string is fully consumed, distance is remaining chars in other
    if adjusted_source_len == 0 {
        let result = adjusted_target_len;
        cache.insert(cache_key, result);
        return result;
    }
    if adjusted_target_len == 0 {
        let result = adjusted_source_len;
        cache.insert(cache_key, result);
        return result;
    }

    // Extract the core substrings (after prefix, before suffix)
    let source_chars: SmallVec<[char; 32]> = source.chars().collect();
    let target_chars: SmallVec<[char; 32]> = target.chars().collect();

    let s_remaining: String = source_chars[prefix_len..prefix_len + adjusted_source_len]
        .iter()
        .collect();
    let t_remaining: String = target_chars[prefix_len..prefix_len + adjusted_target_len]
        .iter()
        .collect();

    let a = source_chars[prefix_len];
    let b = target_chars[prefix_len];

    // Compute substrings for recursion
    let s = substring_from(&s_remaining, 1); // source without first char
    let t = substring_from(&t_remaining, 1); // target without first char

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
        distance = standard_distance_recursive(s, &t_remaining, cache);

        // Early exit
        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        // Insertion: advance target
        let ins_dist = standard_distance_recursive(&s_remaining, t, cache);
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
    let (prefix_len, adjusted_source_len, adjusted_target_len) =
        strip_common_affixes(source, target);

    // If strings are identical after stripping, distance is 0
    if adjusted_source_len == 0 && adjusted_target_len == 0 {
        cache.insert(cache_key, 0);
        return 0;
    }

    // If one string is fully consumed, distance is remaining chars in other
    if adjusted_source_len == 0 {
        let result = adjusted_target_len;
        cache.insert(cache_key, result);
        return result;
    }
    if adjusted_target_len == 0 {
        let result = adjusted_source_len;
        cache.insert(cache_key, result);
        return result;
    }

    // Extract the core substrings (after prefix, before suffix)
    let source_chars: SmallVec<[char; 32]> = source.chars().collect();
    let target_chars: SmallVec<[char; 32]> = target.chars().collect();

    let s_remaining: String = source_chars[prefix_len..prefix_len + adjusted_source_len]
        .iter()
        .collect();
    let t_remaining: String = target_chars[prefix_len..prefix_len + adjusted_target_len]
        .iter()
        .collect();

    let a = source_chars[prefix_len];
    let b = target_chars[prefix_len];

    let s = substring_from(&s_remaining, 1);
    let t = substring_from(&t_remaining, 1);

    let mut distance;

    if a == b {
        distance = transposition_distance_recursive(s, t, cache);

        if distance == 0 {
            cache.insert(cache_key, distance);
            return distance;
        }
    } else {
        // Standard operations: deletion, insertion, substitution
        distance = transposition_distance_recursive(s, &t_remaining, cache);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let ins_dist = transposition_distance_recursive(&s_remaining, t, cache);
        distance = distance.min(ins_dist);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let sub_dist = transposition_distance_recursive(s, t, cache);
        distance = distance.min(sub_dist);

        // Check for transposition
        // Requires at least 2 chars remaining in both strings
        if !s.is_empty() && !t.is_empty() {
            let s_chars: SmallVec<[char; 32]> = s.chars().collect();
            let t_chars: SmallVec<[char; 32]> = t.chars().collect();

            let a1 = s_chars[0];
            let b1 = t_chars[0];

            // Transposition: source[0] == target[1] && source[1] == target[0]
            if a == b1 && a1 == b {
                let ss = substring_from(s, 1);
                let tt = substring_from(t, 1);
                let trans_dist = transposition_distance_recursive(ss, tt, cache);
                distance = distance.min(trans_dist);
            }
        }

        distance += 1;
    }

    cache.insert(cache_key, distance);
    distance
}

/// Recursive merge-and-split distance with memoization.
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
    let (prefix_len, adjusted_source_len, adjusted_target_len) =
        strip_common_affixes(source, target);

    // If strings are identical after stripping, distance is 0
    if adjusted_source_len == 0 && adjusted_target_len == 0 {
        cache.insert(cache_key, 0);
        return 0;
    }

    // If one string is fully consumed, distance is remaining chars in other
    if adjusted_source_len == 0 {
        let result = adjusted_target_len;
        cache.insert(cache_key, result);
        return result;
    }
    if adjusted_target_len == 0 {
        let result = adjusted_source_len;
        cache.insert(cache_key, result);
        return result;
    }

    // Extract the core substrings (after prefix, before suffix)
    let source_chars: SmallVec<[char; 32]> = source.chars().collect();
    let target_chars: SmallVec<[char; 32]> = target.chars().collect();

    let s_remaining: String = source_chars[prefix_len..prefix_len + adjusted_source_len]
        .iter()
        .collect();
    let t_remaining: String = target_chars[prefix_len..prefix_len + adjusted_target_len]
        .iter()
        .collect();

    let a = source_chars[prefix_len];
    let b = target_chars[prefix_len];

    let s = substring_from(&s_remaining, 1);
    let t = substring_from(&t_remaining, 1);

    let mut distance;

    if a == b {
        distance = merge_and_split_distance(s, t, cache);

        if distance == 0 {
            cache.insert(cache_key, distance);
            return distance;
        }
    } else {
        // Standard operations
        distance = merge_and_split_distance(s, &t_remaining, cache);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let ins_dist = merge_and_split_distance(&s_remaining, t, cache);
        distance = distance.min(ins_dist);

        if distance == 0 {
            cache.insert(cache_key, 1);
            return 1;
        }

        let sub_dist = merge_and_split_distance(s, t, cache);
        distance = distance.min(sub_dist);

        // Split operation: one source char → two target chars
        // Skip 2 chars in target (f(w, 1) in C++)
        // Check against t_remaining (w in C++), not t
        if t_remaining.chars().count() > 1 {
            let tt = substring_from(&t_remaining, 2); // Skip 2 chars from t_remaining
            let split_dist = merge_and_split_distance(s, tt, cache);
            distance = distance.min(split_dist);
        }

        // Merge operation: two source chars → one target char
        // Skip 2 chars in source (f(v, 1) in C++)
        // Check against s_remaining (v in C++), not s
        if s_remaining.chars().count() > 1 {
            let ss = substring_from(&s_remaining, 2); // Skip 2 chars from s_remaining
            let merge_dist = merge_and_split_distance(ss, t, cache);
            distance = distance.min(merge_dist);
        }

        distance += 1;
    }

    cache.insert(cache_key, distance);
    distance
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
    fn test_transposition_distance() {
        assert_eq!(transposition_distance("ab", "ba"), 1);
        assert_eq!(transposition_distance("test", "tset"), 1);
        assert_eq!(transposition_distance("abc", "acb"), 1);
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
    fn test_unicode_support() {
        let cache = create_memo_cache();

        // Unicode characters
        assert_eq!(standard_distance_recursive("café", "cafe", &cache), 1);
        assert_eq!(standard_distance_recursive("日本", "日本", &cache), 0);
        assert_eq!(transposition_distance_recursive("日本", "本日", &cache), 1);
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
