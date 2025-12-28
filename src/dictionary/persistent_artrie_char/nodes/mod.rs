//! ART Node Types for Persistent Adaptive Radix Trie (Character-Level)
//!
//! This module implements adaptive node types for 32-bit (u32/char) keys:
//!
//! | Node Type  | Children | Search Method      | Size   | When Used       |
//! |------------|----------|-------------------|--------|-----------------|
//! | CharNode4  | 1-4      | Linear scan       | ~96B   | Sparse nodes    |
//! | CharNode16 | 5-16     | AVX2 SIMD (8×u32) | ~240B  | Moderate fanout |
//! | CharNode48 | 17-48    | Binary search     | ~624B  | High fanout     |
//! | CharBucket | >48      | HashMap-like      | Varies | Dense nodes     |
//!
//! Note: CharNode256 is impossible for u32 keys (would require 4GB array).
//!
//! # Key Differences from Byte-Level ART
//!
//! - Keys are `u32` (4 bytes) instead of `u8` (1 byte)
//! - Node256 is skipped entirely - use CharBucket instead
//! - Node48 cannot use 256-byte index; uses sorted keys + binary search
//! - Node16 uses AVX2 (256-bit) for 8×u32 comparison instead of SSE4.1 (128-bit) for 16×u8
//! - Path compression: 24 bytes (6 chars) instead of 12 bytes (12 u8)
//!
//! # SIMD Optimization
//!
//! CharNode16 uses AVX2 SIMD instructions (`_mm256_cmpeq_epi32`) for parallel
//! key comparison, comparing 8 u32 values simultaneously (two 256-bit registers).

pub mod node4_char;
pub mod node16_char;
pub mod node48_char;
pub mod bucket_char;

pub use node4_char::CharNode4;
pub use node16_char::CharNode16;
pub use node48_char::CharNode48;
pub use bucket_char::CharBucket;

use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum path compression prefix length for char nodes (6 chars = 24 bytes)
pub const CHAR_MAX_PREFIX_LEN: usize = 6;

/// Node flags (reuse from byte version)
pub mod flags {
    /// Node represents a valid dictionary entry (is_final)
    pub const IS_FINAL: u8 = 0b0000_0001;
    /// Node has been modified (dirty)
    pub const IS_DIRTY: u8 = 0b0000_0010;
    /// Node is a leaf (bucket pointer)
    pub const IS_LEAF: u8 = 0b0000_0100;
}

/// Common header for all char ART node types
///
/// This header is shared by CharNode4, CharNode16, CharNode48, and CharBucket.
/// Total size: 16 bytes (same as byte version for consistency).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CharNodeHeader {
    /// Node type discriminant (4, 16, 48, 49 for bucket)
    pub node_type: u8,
    /// Length of the compressed prefix (0-6 chars)
    pub prefix_len: u8,
    /// Node flags (IS_FINAL, IS_DIRTY, IS_LEAF)
    pub flags: u8,
    /// Padding for alignment
    pub _padding: u8,
    /// Number of children currently stored
    pub num_children: u16,
    /// More padding for 8-byte alignment
    pub _padding2: [u8; 2],
    /// Version counter for optimistic locking
    pub version: u64,
}

impl CharNodeHeader {
    /// Create a new char node header
    pub fn new(node_type: u8) -> Self {
        Self {
            node_type,
            prefix_len: 0,
            flags: 0,
            _padding: 0,
            num_children: 0,
            _padding2: [0; 2],
            version: 0,
        }
    }

    /// Check if this node represents a final (accepting) state
    #[inline]
    pub fn is_final(&self) -> bool {
        self.flags & flags::IS_FINAL != 0
    }

    /// Set the final flag
    #[inline]
    pub fn set_final(&mut self, final_state: bool) {
        if final_state {
            self.flags |= flags::IS_FINAL;
        } else {
            self.flags &= !flags::IS_FINAL;
        }
    }

    /// Check if this node is dirty (modified since last write-back)
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.flags & flags::IS_DIRTY != 0
    }

    /// Set the dirty flag
    #[inline]
    pub fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.flags |= flags::IS_DIRTY;
        } else {
            self.flags &= !flags::IS_DIRTY;
        }
    }

    /// Check if this is a leaf node (points to a bucket)
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.flags & flags::IS_LEAF != 0
    }

    /// Set the leaf flag
    #[inline]
    pub fn set_leaf(&mut self, leaf: bool) {
        if leaf {
            self.flags |= flags::IS_LEAF;
        } else {
            self.flags &= !flags::IS_LEAF;
        }
    }

    /// Increment the version counter (for optimistic locking)
    #[inline]
    pub fn increment_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

impl Default for CharNodeHeader {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Compressed prefix for path compression (character-level)
///
/// Stores up to 6 Unicode characters (24 bytes) of shared prefix.
/// This trades off prefix capacity (6 chars vs 12 bytes) for proper
/// Unicode handling where edit distance counts characters, not bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CharCompressedPrefix {
    /// The prefix characters (u32 for full Unicode support)
    pub chars: [u32; CHAR_MAX_PREFIX_LEN],
}

impl CharCompressedPrefix {
    /// Create an empty prefix
    pub const fn empty() -> Self {
        Self {
            chars: [0; CHAR_MAX_PREFIX_LEN],
        }
    }

    /// Create a prefix from a character slice
    ///
    /// # Panics
    /// Panics if the slice is longer than CHAR_MAX_PREFIX_LEN
    pub fn from_chars(chars: &[u32]) -> Self {
        assert!(
            chars.len() <= CHAR_MAX_PREFIX_LEN,
            "prefix too long: {} > {}",
            chars.len(),
            CHAR_MAX_PREFIX_LEN
        );
        let mut prefix = Self::empty();
        prefix.chars[..chars.len()].copy_from_slice(chars);
        prefix
    }

    /// Create a prefix from Rust chars
    pub fn from_char_iter<I: IntoIterator<Item = char>>(chars: I) -> Self {
        let mut prefix = Self::empty();
        for (i, c) in chars.into_iter().take(CHAR_MAX_PREFIX_LEN).enumerate() {
            prefix.chars[i] = c as u32;
        }
        prefix
    }

    /// Check if a key matches this prefix
    ///
    /// Returns the number of matching characters (up to prefix_len)
    pub fn match_key(&self, key: &[u32], prefix_len: usize) -> usize {
        let check_len = prefix_len.min(key.len()).min(CHAR_MAX_PREFIX_LEN);
        for i in 0..check_len {
            if self.chars[i] != key[i] {
                return i;
            }
        }
        check_len
    }

    /// Get the prefix as a slice
    pub fn as_slice(&self, len: usize) -> &[u32] {
        &self.chars[..len.min(CHAR_MAX_PREFIX_LEN)]
    }

    /// Convert prefix to Rust chars
    pub fn to_chars(&self, len: usize) -> Vec<char> {
        self.chars[..len.min(CHAR_MAX_PREFIX_LEN)]
            .iter()
            .filter_map(|&c| char::from_u32(c))
            .collect()
    }
}

impl Default for CharCompressedPrefix {
    fn default() -> Self {
        Self::empty()
    }
}

/// Unified char ART node enum
///
/// This enum wraps all char node types for type-safe dispatch.
/// Note: There is no CharNode256 - CharBucket handles >48 children.
#[derive(Debug, Clone)]
pub enum CharNode {
    /// Node with 1-4 children
    N4(Box<CharNode4>),
    /// Node with 5-16 children
    N16(Box<CharNode16>),
    /// Node with 17-48 children
    N48(Box<CharNode48>),
    /// Node with >48 children (HashMap-like)
    Bucket(Box<CharBucket>),
}

impl CharNode {
    /// Get the node header
    pub fn header(&self) -> &CharNodeHeader {
        match self {
            CharNode::N4(n) => &n.header,
            CharNode::N16(n) => &n.header,
            CharNode::N48(n) => &n.header,
            CharNode::Bucket(n) => &n.header,
        }
    }

    /// Get the node header mutably
    pub fn header_mut(&mut self) -> &mut CharNodeHeader {
        match self {
            CharNode::N4(n) => &mut n.header,
            CharNode::N16(n) => &mut n.header,
            CharNode::N48(n) => &mut n.header,
            CharNode::Bucket(n) => &mut n.header,
        }
    }

    /// Get the compressed prefix
    pub fn prefix(&self) -> &CharCompressedPrefix {
        match self {
            CharNode::N4(n) => &n.prefix,
            CharNode::N16(n) => &n.prefix,
            CharNode::N48(n) => &n.prefix,
            CharNode::Bucket(n) => &n.prefix,
        }
    }

    /// Get the compressed prefix mutably
    pub fn prefix_mut(&mut self) -> &mut CharCompressedPrefix {
        match self {
            CharNode::N4(n) => &mut n.prefix,
            CharNode::N16(n) => &mut n.prefix,
            CharNode::N48(n) => &mut n.prefix,
            CharNode::Bucket(n) => &mut n.prefix,
        }
    }

    /// Check if this node is final (accepting state)
    #[inline]
    pub fn is_final(&self) -> bool {
        self.header().is_final()
    }

    /// Look up a child by key character
    pub fn find_child(&self, key: u32) -> Option<&SwizzledPtr> {
        match self {
            CharNode::N4(n) => n.find_child(key),
            CharNode::N16(n) => n.find_child(key),
            CharNode::N48(n) => n.find_child(key),
            CharNode::Bucket(n) => n.find_child(key),
        }
    }

    /// Look up a child by key character (mutable)
    pub fn find_child_mut(&mut self, key: u32) -> Option<&mut SwizzledPtr> {
        match self {
            CharNode::N4(n) => n.find_child_mut(key),
            CharNode::N16(n) => n.find_child_mut(key),
            CharNode::N48(n) => n.find_child_mut(key),
            CharNode::Bucket(n) => n.find_child_mut(key),
        }
    }

    /// Get an iterator over all (key, child) pairs
    pub fn iter_children(&self) -> Box<dyn Iterator<Item = (u32, &SwizzledPtr)> + '_> {
        match self {
            CharNode::N4(n) => Box::new(n.iter_children()),
            CharNode::N16(n) => Box::new(n.iter_children()),
            CharNode::N48(n) => Box::new(n.iter_children()),
            CharNode::Bucket(n) => Box::new(n.iter_children()),
        }
    }

    /// Get the number of children
    #[inline]
    pub fn num_children(&self) -> usize {
        self.header().num_children as usize
    }

    /// Check if the node is full
    pub fn is_full(&self) -> bool {
        match self {
            CharNode::N4(n) => n.is_full(),
            CharNode::N16(n) => n.is_full(),
            CharNode::N48(n) => n.is_full(),
            CharNode::Bucket(_) => false, // Bucket can always grow
        }
    }

    /// Check if the node should grow to a larger type
    pub fn should_grow(&self) -> bool {
        self.is_full()
    }

    /// Check if the node should shrink to a smaller type
    pub fn should_shrink(&self) -> bool {
        match self {
            CharNode::N4(_) => false,
            CharNode::N16(n) => n.header.num_children <= 4,
            CharNode::N48(n) => n.header.num_children <= 16,
            CharNode::Bucket(n) => n.header.num_children <= 48,
        }
    }

    /// Create an empty CharNode4
    pub fn new_node4() -> Self {
        CharNode::N4(Box::new(CharNode4::new()))
    }

    /// Create an empty CharNode16
    pub fn new_node16() -> Self {
        CharNode::N16(Box::new(CharNode16::new()))
    }

    /// Create an empty CharNode48
    pub fn new_node48() -> Self {
        CharNode::N48(Box::new(CharNode48::new()))
    }

    /// Create an empty CharBucket
    pub fn new_bucket() -> Self {
        CharNode::Bucket(Box::new(CharBucket::new()))
    }

    /// Grow this node to the next larger type.
    ///
    /// Growth path: CharNode4 → CharNode16 → CharNode48 → CharBucket
    pub fn grow(&self) -> Option<CharNode> {
        match self {
            CharNode::N4(n) => Some(CharNode::N16(Box::new(n.grow()))),
            CharNode::N16(n) => Some(CharNode::N48(Box::new(n.grow()))),
            CharNode::N48(n) => Some(CharNode::Bucket(Box::new(n.grow()))),
            CharNode::Bucket(_) => None, // Bucket grows internally
        }
    }

    /// Shrink this node to the next smaller type.
    ///
    /// Shrink path: CharBucket → CharNode48 → CharNode16 → CharNode4
    pub fn shrink(&self) -> Option<CharNode> {
        match self {
            CharNode::N4(_) => None,
            CharNode::N16(n) => Some(CharNode::N4(Box::new(n.shrink()))),
            CharNode::N48(n) => Some(CharNode::N16(Box::new(n.shrink()))),
            CharNode::Bucket(n) => Some(CharNode::N48(Box::new(n.shrink()))),
        }
    }

    /// Add a child to this node, automatically growing if necessary.
    pub fn add_child_growing(&mut self, key: u32, child: SwizzledPtr) -> Result<Option<CharNode>, AddChildError> {
        match self {
            CharNode::N4(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(CharNode::N16(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            CharNode::N16(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(CharNode::N48(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            CharNode::N48(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(CharNode::Bucket(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            CharNode::Bucket(n) => {
                n.add_child(key, child)?;
                Ok(None)
            }
        }
    }

    /// Remove a child from this node, automatically shrinking if appropriate.
    pub fn remove_child_shrinking(&mut self, key: u32) -> Option<(SwizzledPtr, Option<CharNode>)> {
        match self {
            CharNode::N4(n) => {
                n.remove_child(key).map(|removed| (removed, None))
            }
            CharNode::N16(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 4 {
                        Some((removed, Some(CharNode::N4(Box::new(n.shrink())))))
                    } else {
                        Some((removed, None))
                    }
                } else {
                    None
                }
            }
            CharNode::N48(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 16 {
                        Some((removed, Some(CharNode::N16(Box::new(n.shrink())))))
                    } else {
                        Some((removed, None))
                    }
                } else {
                    None
                }
            }
            CharNode::Bucket(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 48 {
                        Some((removed, Some(CharNode::N48(Box::new(n.shrink())))))
                    } else {
                        Some((removed, None))
                    }
                } else {
                    None
                }
            }
        }
    }
}

/// Trait for common char ART node operations
///
/// Similar to byte-level ArtNode but uses u32 keys for Unicode support.
pub trait CharArtNode {
    /// Find a child by key character (u32)
    fn find_child(&self, key: u32) -> Option<&SwizzledPtr>;

    /// Find a child mutably by key character (u32)
    fn find_child_mut(&mut self, key: u32) -> Option<&mut SwizzledPtr>;

    /// Add a child with the given key
    fn add_child(&mut self, key: u32, child: SwizzledPtr) -> Result<(), AddChildError>;

    /// Remove a child by key
    fn remove_child(&mut self, key: u32) -> Option<SwizzledPtr>;

    /// Check if the node is full
    fn is_full(&self) -> bool;

    /// Get an iterator over (key, child) pairs
    fn iter_children(&self) -> impl Iterator<Item = (u32, &SwizzledPtr)>;
}

/// Error when adding a child fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddChildError {
    /// Node is full, needs to grow to a larger type
    NodeFull,
    /// Key already exists
    KeyExists,
}

impl std::fmt::Display for AddChildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddChildError::NodeFull => write!(f, "node is full"),
            AddChildError::KeyExists => write!(f, "key already exists"),
        }
    }
}

impl std::error::Error for AddChildError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_node_header_flags() {
        let mut header = CharNodeHeader::new(4);
        assert!(!header.is_final());
        assert!(!header.is_dirty());
        assert!(!header.is_leaf());

        header.set_final(true);
        assert!(header.is_final());

        header.set_dirty(true);
        assert!(header.is_dirty());

        header.set_leaf(true);
        assert!(header.is_leaf());

        header.set_final(false);
        assert!(!header.is_final());
        assert!(header.is_dirty()); // Other flags unchanged
    }

    #[test]
    fn test_char_node_header_version() {
        let mut header = CharNodeHeader::new(4);
        assert_eq!(header.version, 0);

        header.increment_version();
        assert_eq!(header.version, 1);

        header.increment_version();
        assert_eq!(header.version, 2);
    }

    #[test]
    fn test_char_compressed_prefix() {
        let chars: Vec<u32> = "hello".chars().map(|c| c as u32).collect();
        let prefix = CharCompressedPrefix::from_chars(&chars);
        assert_eq!(prefix.as_slice(5), &chars[..]);
        assert_eq!(prefix.as_slice(3), &chars[..3]);
    }

    #[test]
    fn test_char_prefix_from_rust_chars() {
        let prefix = CharCompressedPrefix::from_char_iter("hello".chars());
        assert_eq!(prefix.to_chars(5), vec!['h', 'e', 'l', 'l', 'o']);
    }

    #[test]
    fn test_char_prefix_unicode() {
        // Test with Unicode characters (emoji, CJK)
        let prefix = CharCompressedPrefix::from_char_iter("日本🎉".chars());
        assert_eq!(prefix.to_chars(3), vec!['日', '本', '🎉']);
    }

    #[test]
    fn test_char_prefix_matching() {
        let chars: Vec<u32> = "hello".chars().map(|c| c as u32).collect();
        let prefix = CharCompressedPrefix::from_chars(&chars);

        // Full match
        let key: Vec<u32> = "hello world".chars().map(|c| c as u32).collect();
        assert_eq!(prefix.match_key(&key, 5), 5);

        // Partial match
        let key: Vec<u32> = "help".chars().map(|c| c as u32).collect();
        assert_eq!(prefix.match_key(&key, 5), 3);

        // No match
        let key: Vec<u32> = "world".chars().map(|c| c as u32).collect();
        assert_eq!(prefix.match_key(&key, 5), 0);
    }

    #[test]
    #[should_panic(expected = "prefix too long")]
    fn test_char_prefix_too_long() {
        // 7 chars exceeds CHAR_MAX_PREFIX_LEN (6)
        let chars: Vec<u32> = "1234567".chars().map(|c| c as u32).collect();
        CharCompressedPrefix::from_chars(&chars);
    }
}
