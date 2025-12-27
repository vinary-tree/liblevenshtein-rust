//! Path Compression Operations for Persistent ART
//!
//! This module provides higher-level path compression operations used during
//! ART traversal, insertion, and deletion. Path compression is a key optimization
//! that reduces tree height by storing common prefixes inline in nodes.
//!
//! # Path Compression Overview
//!
//! In a standard trie, each edge represents a single character. With path compression,
//! consecutive single-child edges are collapsed into a prefix stored in the child node:
//!
//! ```text
//! Without path compression:       With path compression:
//!
//!       [root]                         [root]
//!         │ 'h'                           │ 'h'
//!        [a]                            [a]
//!         │ 'e'                  prefix: "ello"
//!        [b]                             │
//!         │ 'l'                        [final]
//!        [c]
//!         │ 'l'
//!        [d]
//!         │ 'o'
//!      [final]
//! ```
//!
//! This dramatically reduces tree height and memory usage for strings with common prefixes.
//!
//! # Operations
//!
//! - [`prefix_mismatch`]: Find where a key diverges from a node's prefix
//! - [`split_prefix`]: Split a prefix at a given position
//! - [`extend_prefix`]: Add bytes to an existing prefix
//! - [`truncate_prefix`]: Remove bytes from the beginning of a prefix

use super::nodes::{CompressedPrefix, MAX_PREFIX_LEN};

/// Result of comparing a search key against a node's compressed prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixMatchResult {
    /// The key fully matches the prefix (key matches for prefix_len bytes)
    FullMatch {
        /// Number of bytes that matched
        matched: usize,
    },
    /// The key partially matches (mismatch at position `mismatch_pos`)
    PartialMatch {
        /// Position where the mismatch occurred
        mismatch_pos: usize,
        /// The prefix byte at the mismatch position
        prefix_byte: u8,
        /// The key byte at the mismatch position
        key_byte: u8,
    },
    /// The key is shorter than the prefix
    KeyTooShort {
        /// Number of bytes that matched before key ended
        matched: usize,
    },
}

/// Find where a search key diverges from a node's compressed prefix.
///
/// This is the core operation during ART traversal. When descending the tree,
/// we need to check if the search key matches each node's prefix before
/// following child pointers.
///
/// # Arguments
///
/// * `prefix` - The node's compressed prefix
/// * `prefix_len` - The actual length of the prefix (may be less than MAX_PREFIX_LEN)
/// * `key` - The search key to compare
/// * `key_offset` - The current position in the key (how many bytes already consumed)
///
/// # Returns
///
/// A `PrefixMatchResult` indicating whether and where the key matches/diverges.
///
/// # Example
///
/// ```ignore
/// let prefix = CompressedPrefix::from_bytes(b"hello");
/// let result = prefix_mismatch(&prefix, 5, b"help me", 0);
/// assert!(matches!(result, PrefixMatchResult::PartialMatch {
///     mismatch_pos: 3,
///     prefix_byte: b'l',
///     key_byte: b'p',
/// }));
/// ```
pub fn prefix_mismatch(
    prefix: &CompressedPrefix,
    prefix_len: usize,
    key: &[u8],
    key_offset: usize,
) -> PrefixMatchResult {
    let remaining_key = &key[key_offset..];
    let check_len = prefix_len.min(MAX_PREFIX_LEN);

    // Check if key is too short
    if remaining_key.len() < check_len {
        // Key might still partially match
        for i in 0..remaining_key.len() {
            if prefix.bytes[i] != remaining_key[i] {
                return PrefixMatchResult::PartialMatch {
                    mismatch_pos: i,
                    prefix_byte: prefix.bytes[i],
                    key_byte: remaining_key[i],
                };
            }
        }
        return PrefixMatchResult::KeyTooShort {
            matched: remaining_key.len(),
        };
    }

    // Compare prefix bytes
    for i in 0..check_len {
        if prefix.bytes[i] != remaining_key[i] {
            return PrefixMatchResult::PartialMatch {
                mismatch_pos: i,
                prefix_byte: prefix.bytes[i],
                key_byte: remaining_key[i],
            };
        }
    }

    PrefixMatchResult::FullMatch { matched: check_len }
}

/// Result of splitting a prefix at a given position
#[derive(Debug, Clone)]
pub struct SplitPrefix {
    /// The part before the split point (goes to new inner node)
    pub before: CompressedPrefix,
    /// Length of the "before" prefix
    pub before_len: usize,
    /// The byte at the split point (becomes the edge label)
    pub split_byte: u8,
    /// The part after the split point (remains on original node)
    pub after: CompressedPrefix,
    /// Length of the "after" prefix
    pub after_len: usize,
}

/// Split a prefix at a given position.
///
/// When inserting a key that diverges from an existing prefix, we need to
/// split the node. This function calculates the three parts:
/// 1. The common prefix (before the divergence)
/// 2. The diverging byte (becomes an edge label)
/// 3. The remaining suffix (stays with the original node)
///
/// # Arguments
///
/// * `prefix` - The original compressed prefix
/// * `prefix_len` - The actual length of the prefix
/// * `split_pos` - The position at which to split (0 <= split_pos < prefix_len)
///
/// # Returns
///
/// A `SplitPrefix` containing all three parts.
///
/// # Panics
///
/// Panics if `split_pos >= prefix_len` or `split_pos >= MAX_PREFIX_LEN`.
pub fn split_prefix(prefix: &CompressedPrefix, prefix_len: usize, split_pos: usize) -> SplitPrefix {
    assert!(
        split_pos < prefix_len,
        "split position {} must be less than prefix length {}",
        split_pos,
        prefix_len
    );
    assert!(
        split_pos < MAX_PREFIX_LEN,
        "split position {} must be less than MAX_PREFIX_LEN {}",
        split_pos,
        MAX_PREFIX_LEN
    );

    // Build the "before" prefix (bytes [0, split_pos))
    let mut before = CompressedPrefix::empty();
    if split_pos > 0 {
        before.bytes[..split_pos].copy_from_slice(&prefix.bytes[..split_pos]);
    }

    // The split byte
    let split_byte = prefix.bytes[split_pos];

    // Build the "after" prefix (bytes [split_pos + 1, prefix_len))
    let mut after = CompressedPrefix::empty();
    let after_len = prefix_len.saturating_sub(split_pos + 1);
    if after_len > 0 {
        let after_start = split_pos + 1;
        let copy_len = after_len.min(MAX_PREFIX_LEN);
        after.bytes[..copy_len].copy_from_slice(&prefix.bytes[after_start..after_start + copy_len]);
    }

    SplitPrefix {
        before,
        before_len: split_pos,
        split_byte,
        after,
        after_len,
    }
}

/// Extend a prefix by prepending bytes.
///
/// When a node is promoted (e.g., during deletion), we may need to prepend
/// the parent's prefix to the child's prefix.
///
/// # Arguments
///
/// * `base_prefix` - The original prefix
/// * `base_len` - Length of the original prefix
/// * `prepend_bytes` - Bytes to prepend
/// * `prepend_edge` - The edge label byte (inserted between prepend_bytes and base_prefix)
///
/// # Returns
///
/// A tuple of (new_prefix, new_len). If the combined length exceeds MAX_PREFIX_LEN,
/// the prefix is truncated to MAX_PREFIX_LEN.
///
/// # Example
///
/// ```ignore
/// // base_prefix = "world", prepend = "hel", edge = 'l', result = "hello"+"world" truncated
/// let (new_prefix, new_len) = extend_prefix(&base, 5, b"hel", b'l');
/// ```
pub fn extend_prefix(
    base_prefix: &CompressedPrefix,
    base_len: usize,
    prepend_bytes: &[u8],
    prepend_edge: u8,
) -> (CompressedPrefix, usize) {
    let mut result = CompressedPrefix::empty();

    // Calculate total needed length
    let total_len = prepend_bytes.len() + 1 + base_len; // prepend + edge + base
    let actual_len = total_len.min(MAX_PREFIX_LEN);

    let mut write_pos = 0;

    // Copy prepend bytes
    let prepend_copy = prepend_bytes.len().min(MAX_PREFIX_LEN);
    if prepend_copy > 0 && write_pos < MAX_PREFIX_LEN {
        let copy_len = prepend_copy.min(MAX_PREFIX_LEN - write_pos);
        result.bytes[write_pos..write_pos + copy_len]
            .copy_from_slice(&prepend_bytes[..copy_len]);
        write_pos += copy_len;
    }

    // Copy edge byte
    if write_pos < MAX_PREFIX_LEN {
        result.bytes[write_pos] = prepend_edge;
        write_pos += 1;
    }

    // Copy base prefix bytes
    if write_pos < MAX_PREFIX_LEN && base_len > 0 {
        let copy_len = base_len.min(MAX_PREFIX_LEN - write_pos);
        result.bytes[write_pos..write_pos + copy_len]
            .copy_from_slice(&base_prefix.bytes[..copy_len]);
    }

    (result, actual_len)
}

/// Truncate a prefix by removing bytes from the beginning.
///
/// After consuming part of a prefix during traversal, the remaining prefix
/// can be calculated with this function.
///
/// # Arguments
///
/// * `prefix` - The original prefix
/// * `prefix_len` - Length of the original prefix
/// * `remove_count` - Number of bytes to remove from the beginning
///
/// # Returns
///
/// A tuple of (new_prefix, new_len). If remove_count >= prefix_len, returns an empty prefix.
pub fn truncate_prefix(
    prefix: &CompressedPrefix,
    prefix_len: usize,
    remove_count: usize,
) -> (CompressedPrefix, usize) {
    if remove_count >= prefix_len {
        return (CompressedPrefix::empty(), 0);
    }

    let new_len = prefix_len - remove_count;
    let mut result = CompressedPrefix::empty();

    let copy_len = new_len.min(MAX_PREFIX_LEN);
    result.bytes[..copy_len].copy_from_slice(&prefix.bytes[remove_count..remove_count + copy_len]);

    (result, new_len)
}

/// Calculate the common prefix length between two byte slices.
///
/// # Arguments
///
/// * `a` - First byte slice
/// * `b` - Second byte slice
///
/// # Returns
///
/// The number of bytes that match at the beginning of both slices.
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let max_len = a.len().min(b.len());
    for i in 0..max_len {
        if a[i] != b[i] {
            return i;
        }
    }
    max_len
}

/// Create a CompressedPrefix from two byte slices that share a common prefix.
///
/// This is useful when inserting a new key that shares a common prefix with
/// an existing node's key.
///
/// # Arguments
///
/// * `a` - First byte slice
/// * `b` - Second byte slice
///
/// # Returns
///
/// A tuple of (prefix, common_len) representing the common prefix.
pub fn make_common_prefix(a: &[u8], b: &[u8]) -> (CompressedPrefix, usize) {
    let common_len = common_prefix_len(a, b);
    let prefix_len = common_len.min(MAX_PREFIX_LEN);
    let prefix = CompressedPrefix::from_bytes(&a[..prefix_len]);
    (prefix, common_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_mismatch_full_match() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let result = prefix_mismatch(&prefix, 5, b"hello world", 0);
        assert_eq!(result, PrefixMatchResult::FullMatch { matched: 5 });
    }

    #[test]
    fn test_prefix_mismatch_partial() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let result = prefix_mismatch(&prefix, 5, b"help me", 0);
        assert_eq!(
            result,
            PrefixMatchResult::PartialMatch {
                mismatch_pos: 3,
                prefix_byte: b'l',
                key_byte: b'p',
            }
        );
    }

    #[test]
    fn test_prefix_mismatch_key_too_short() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let result = prefix_mismatch(&prefix, 5, b"hel", 0);
        assert_eq!(result, PrefixMatchResult::KeyTooShort { matched: 3 });
    }

    #[test]
    fn test_prefix_mismatch_with_offset() {
        let prefix = CompressedPrefix::from_bytes(b"world");
        let result = prefix_mismatch(&prefix, 5, b"hello world", 6);
        assert_eq!(result, PrefixMatchResult::FullMatch { matched: 5 });
    }

    #[test]
    fn test_prefix_mismatch_at_start() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let result = prefix_mismatch(&prefix, 5, b"world", 0);
        assert_eq!(
            result,
            PrefixMatchResult::PartialMatch {
                mismatch_pos: 0,
                prefix_byte: b'h',
                key_byte: b'w',
            }
        );
    }

    #[test]
    fn test_split_prefix_middle() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let split = split_prefix(&prefix, 5, 2);

        assert_eq!(split.before.as_slice(split.before_len), b"he");
        assert_eq!(split.before_len, 2);
        assert_eq!(split.split_byte, b'l');
        assert_eq!(split.after.as_slice(split.after_len), b"lo");
        assert_eq!(split.after_len, 2);
    }

    #[test]
    fn test_split_prefix_at_start() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let split = split_prefix(&prefix, 5, 0);

        assert_eq!(split.before_len, 0);
        assert_eq!(split.split_byte, b'h');
        assert_eq!(split.after.as_slice(split.after_len), b"ello");
        assert_eq!(split.after_len, 4);
    }

    #[test]
    fn test_split_prefix_at_end() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let split = split_prefix(&prefix, 5, 4);

        assert_eq!(split.before.as_slice(split.before_len), b"hell");
        assert_eq!(split.before_len, 4);
        assert_eq!(split.split_byte, b'o');
        assert_eq!(split.after_len, 0);
    }

    #[test]
    fn test_extend_prefix() {
        let base = CompressedPrefix::from_bytes(b"world");
        let (result, len) = extend_prefix(&base, 5, b"hel", b'l');

        // "hel" + 'l' + "world" = "hellworld" = 9 bytes
        assert_eq!(len, 9);
        assert_eq!(&result.bytes[..9], b"hellworld");
    }

    #[test]
    fn test_extend_prefix_truncation() {
        let base = CompressedPrefix::from_bytes(b"world");
        // Create a long prepend that will cause truncation
        let prepend = b"this is a very long prefix that will be truncated";
        let (result, len) = extend_prefix(&base, 5, prepend, b'!');

        // Should be capped at MAX_PREFIX_LEN
        assert_eq!(len, MAX_PREFIX_LEN);
        assert_eq!(result.bytes.len(), MAX_PREFIX_LEN);
    }

    #[test]
    fn test_truncate_prefix() {
        let prefix = CompressedPrefix::from_bytes(b"hello world");
        let (result, len) = truncate_prefix(&prefix, 11, 6);

        assert_eq!(len, 5);
        assert_eq!(result.as_slice(len), b"world");
    }

    #[test]
    fn test_truncate_prefix_all() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let (result, len) = truncate_prefix(&prefix, 5, 5);

        assert_eq!(len, 0);
        assert_eq!(result.as_slice(len), b"");
    }

    #[test]
    fn test_truncate_prefix_overflow() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        let (result, len) = truncate_prefix(&prefix, 5, 10);

        assert_eq!(len, 0);
        assert_eq!(result.as_slice(len), b"");
    }

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len(b"hello", b"help"), 3);
        assert_eq!(common_prefix_len(b"hello", b"hello"), 5);
        assert_eq!(common_prefix_len(b"hello", b"world"), 0);
        assert_eq!(common_prefix_len(b"", b"hello"), 0);
        assert_eq!(common_prefix_len(b"h", b"hello"), 1);
    }

    #[test]
    fn test_make_common_prefix() {
        let (prefix, len) = make_common_prefix(b"hello", b"help");
        assert_eq!(len, 3);
        assert_eq!(prefix.as_slice(len), b"hel");
    }

    #[test]
    fn test_make_common_prefix_none() {
        let (prefix, len) = make_common_prefix(b"hello", b"world");
        assert_eq!(len, 0);
        assert_eq!(prefix.as_slice(len), b"");
    }

    #[test]
    fn test_make_common_prefix_exact() {
        let (prefix, len) = make_common_prefix(b"hello", b"hello");
        assert_eq!(len, 5);
        assert_eq!(prefix.as_slice(len), b"hello");
    }
}
