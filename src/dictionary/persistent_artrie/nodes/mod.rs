//! ART Node Types for Persistent Adaptive Radix Trie
//!
//! This module implements the four adaptive node types used in ART:
//!
//! | Node Type | Children | Search Method | Size   | When Used |
//! |-----------|----------|---------------|--------|-----------|
//! | Node4     | 1-4      | Linear scan   | ~48B   | Sparse nodes |
//! | Node16    | 5-16     | SIMD (SSE4.1) | ~160B  | Moderate fanout |
//! | Node48    | 17-48    | Index array   | ~656B  | High fanout |
//! | Node256   | 49-256   | Direct array  | ~2KB   | Dense nodes |
//!
//! # Node Layout
//!
//! All nodes share a common header followed by type-specific data:
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │                     NodeHeader (16 bytes)                  │
//! ├───────────────────────────────────────────────────────────┤
//! │ node_type: u8        │ Node type discriminant             │
//! │ num_children: u8     │ Current number of children         │
//! │ prefix_len: u8       │ Compressed prefix length (0-12)    │
//! │ flags: u8            │ Flags (is_final, dirty, etc.)      │
//! │ _padding: [u8; 4]    │ Alignment padding                  │
//! │ version: u64         │ Version for optimistic locking     │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Path Compression
//!
//! Each node can store up to 12 bytes of compressed prefix. When traversing
//! the trie, if the prefix matches, we skip those bytes in the search key.
//! This significantly reduces tree height for common patterns.
//!
//! # SIMD Optimization
//!
//! Node16 uses SSE4.1 SIMD instructions (`_mm_cmpeq_epi8`) for parallel
//! key comparison, finding matches in ~3 cycles vs ~16 for linear scan.

pub mod node4;
pub mod node16;
pub mod node48;
pub mod node256;

pub use node4::Node4;
pub use node16::Node16;
pub use node48::Node48;
pub use node256::Node256;

use super::swizzled_ptr::SwizzledPtr;

/// Maximum path compression prefix length (12 bytes)
pub const MAX_PREFIX_LEN: usize = 12;

/// Node flags
pub mod flags {
    /// Node represents a valid dictionary entry (is_final)
    pub const IS_FINAL: u8 = 0b0000_0001;
    /// Node has been modified (dirty)
    pub const IS_DIRTY: u8 = 0b0000_0010;
    /// Node is a leaf (bucket pointer)
    pub const IS_LEAF: u8 = 0b0000_0100;
}

/// Common header for all ART node types
///
/// This header is shared by Node4, Node16, Node48, and Node256.
/// Total size: 16 bytes (2 cache lines would hold header + small node).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NodeHeader {
    /// Node type discriminant (4, 16, 48, or 0 for Node256)
    pub node_type: u8,
    /// Length of the compressed prefix (0-12)
    pub prefix_len: u8,
    /// Node flags (IS_FINAL, IS_DIRTY, IS_LEAF)
    pub flags: u8,
    /// Padding for alignment
    pub _padding: u8,
    /// Number of children currently stored (u16 to support Node256 with 256 children)
    pub num_children: u16,
    /// More padding for 8-byte alignment
    pub _padding2: [u8; 2],
    /// Version counter for optimistic locking
    pub version: u64,
}

impl NodeHeader {
    /// Create a new node header
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

impl Default for NodeHeader {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Compressed prefix for path compression
///
/// Stores up to 12 bytes of shared prefix that can be skipped during traversal.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompressedPrefix {
    /// The prefix bytes
    pub bytes: [u8; MAX_PREFIX_LEN],
}

impl CompressedPrefix {
    /// Create an empty prefix
    pub const fn empty() -> Self {
        Self {
            bytes: [0; MAX_PREFIX_LEN],
        }
    }

    /// Create a prefix from a byte slice
    ///
    /// # Panics
    /// Panics if the slice is longer than MAX_PREFIX_LEN
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() <= MAX_PREFIX_LEN,
            "prefix too long: {} > {}",
            bytes.len(),
            MAX_PREFIX_LEN
        );
        let mut prefix = Self::empty();
        prefix.bytes[..bytes.len()].copy_from_slice(bytes);
        prefix
    }

    /// Check if a key matches this prefix
    ///
    /// Returns the number of matching bytes (up to prefix_len)
    pub fn match_key(&self, key: &[u8], prefix_len: usize) -> usize {
        let check_len = prefix_len.min(key.len()).min(MAX_PREFIX_LEN);
        for i in 0..check_len {
            if self.bytes[i] != key[i] {
                return i;
            }
        }
        check_len
    }

    /// Get the prefix as a slice
    pub fn as_slice(&self, len: usize) -> &[u8] {
        &self.bytes[..len.min(MAX_PREFIX_LEN)]
    }
}

impl Default for CompressedPrefix {
    fn default() -> Self {
        Self::empty()
    }
}

/// Unified ART node enum
///
/// This enum wraps all four node types for type-safe dispatch.
#[derive(Debug, Clone)]
pub enum Node {
    /// Node with 1-4 children
    N4(Box<Node4>),
    /// Node with 5-16 children
    N16(Box<Node16>),
    /// Node with 17-48 children
    N48(Box<Node48>),
    /// Node with 49-256 children
    N256(Box<Node256>),
}

impl Node {
    /// Get the node header
    pub fn header(&self) -> &NodeHeader {
        match self {
            Node::N4(n) => &n.header,
            Node::N16(n) => &n.header,
            Node::N48(n) => &n.header,
            Node::N256(n) => &n.header,
        }
    }

    /// Get the node header mutably
    pub fn header_mut(&mut self) -> &mut NodeHeader {
        match self {
            Node::N4(n) => &mut n.header,
            Node::N16(n) => &mut n.header,
            Node::N48(n) => &mut n.header,
            Node::N256(n) => &mut n.header,
        }
    }

    /// Get the compressed prefix
    pub fn prefix(&self) -> &CompressedPrefix {
        match self {
            Node::N4(n) => &n.prefix,
            Node::N16(n) => &n.prefix,
            Node::N48(n) => &n.prefix,
            Node::N256(n) => &n.prefix,
        }
    }

    /// Get the compressed prefix mutably
    pub fn prefix_mut(&mut self) -> &mut CompressedPrefix {
        match self {
            Node::N4(n) => &mut n.prefix,
            Node::N16(n) => &mut n.prefix,
            Node::N48(n) => &mut n.prefix,
            Node::N256(n) => &mut n.prefix,
        }
    }

    /// Check if this node is final (accepting state)
    #[inline]
    pub fn is_final(&self) -> bool {
        self.header().is_final()
    }

    /// Look up a child by key byte
    pub fn find_child(&self, key: u8) -> Option<&SwizzledPtr> {
        match self {
            Node::N4(n) => n.find_child(key),
            Node::N16(n) => n.find_child(key),
            Node::N48(n) => n.find_child(key),
            Node::N256(n) => n.find_child(key),
        }
    }

    /// Look up a child by key byte (mutable)
    pub fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr> {
        match self {
            Node::N4(n) => n.find_child_mut(key),
            Node::N16(n) => n.find_child_mut(key),
            Node::N48(n) => n.find_child_mut(key),
            Node::N256(n) => n.find_child_mut(key),
        }
    }

    /// Get an iterator over all (key, child) pairs
    pub fn iter_children(&self) -> Box<dyn Iterator<Item = (u8, &SwizzledPtr)> + '_> {
        match self {
            Node::N4(n) => Box::new(n.iter_children()),
            Node::N16(n) => Box::new(n.iter_children()),
            Node::N48(n) => Box::new(n.iter_children()),
            Node::N256(n) => Box::new(n.iter_children()),
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
            Node::N4(n) => n.is_full(),
            Node::N16(n) => n.is_full(),
            Node::N48(n) => n.is_full(),
            Node::N256(n) => n.is_full(),
        }
    }

    /// Check if the node should grow to a larger type
    pub fn should_grow(&self) -> bool {
        self.is_full()
    }

    /// Check if the node should shrink to a smaller type
    pub fn should_shrink(&self) -> bool {
        match self {
            Node::N4(_) => false,
            Node::N16(n) => n.header.num_children <= 4,
            Node::N48(n) => n.header.num_children <= 16,
            Node::N256(n) => n.header.num_children <= 48,
        }
    }

    /// Create an empty Node4
    pub fn new_node4() -> Self {
        Node::N4(Box::new(Node4::new()))
    }

    /// Create an empty Node16
    pub fn new_node16() -> Self {
        Node::N16(Box::new(Node16::new()))
    }

    /// Create an empty Node48
    pub fn new_node48() -> Self {
        Node::N48(Box::new(Node48::new()))
    }

    /// Create an empty Node256
    pub fn new_node256() -> Self {
        Node::N256(Box::new(Node256::new()))
    }

    /// Grow this node to the next larger type.
    ///
    /// This is called when a node becomes full and needs more capacity.
    ///
    /// - Node4 → Node16
    /// - Node16 → Node48
    /// - Node48 → Node256
    /// - Node256 → None (already at max size)
    ///
    /// Returns `None` if the node is already Node256 (cannot grow further).
    pub fn grow(&self) -> Option<Node> {
        match self {
            Node::N4(n) => Some(Node::N16(Box::new(n.grow()))),
            Node::N16(n) => Some(Node::N48(Box::new(n.grow()))),
            Node::N48(n) => Some(Node::N256(Box::new(n.grow()))),
            Node::N256(_) => None, // Cannot grow further
        }
    }

    /// Shrink this node to the next smaller type.
    ///
    /// This is called when a node has too few children for its type.
    ///
    /// - Node16 → Node4 (when ≤4 children)
    /// - Node48 → Node16 (when ≤16 children)
    /// - Node256 → Node48 (when ≤48 children)
    /// - Node4 → None (already at min size)
    ///
    /// Returns `None` if the node is already Node4 (cannot shrink further).
    pub fn shrink(&self) -> Option<Node> {
        match self {
            Node::N4(_) => None, // Cannot shrink further
            Node::N16(n) => Some(Node::N4(Box::new(n.shrink()))),
            Node::N48(n) => Some(Node::N16(Box::new(n.shrink()))),
            Node::N256(n) => Some(Node::N48(Box::new(n.shrink()))),
        }
    }

    /// Add a child to this node, automatically growing if necessary.
    ///
    /// If the node is full, this method will grow the node to the next
    /// larger type and return the new node. Otherwise, it adds the child
    /// to the existing node and returns None.
    ///
    /// # Returns
    ///
    /// - `Ok(None)` - Child added successfully to existing node
    /// - `Ok(Some(new_node))` - Node was grown and child added to new node
    /// - `Err(AddChildError::KeyExists)` - Key already exists
    pub fn add_child_growing(&mut self, key: u8, child: SwizzledPtr) -> Result<Option<Node>, AddChildError> {
        match self {
            Node::N4(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(Node::N16(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            Node::N16(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(Node::N48(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            Node::N48(n) => {
                if n.is_full() {
                    let mut grown = n.grow();
                    grown.add_child(key, child)?;
                    Ok(Some(Node::N256(Box::new(grown))))
                } else {
                    n.add_child(key, child)?;
                    Ok(None)
                }
            }
            Node::N256(n) => {
                n.add_child(key, child)?;
                Ok(None)
            }
        }
    }

    /// Remove a child from this node, automatically shrinking if appropriate.
    ///
    /// If the node has too few children after removal, this method will
    /// shrink the node to the next smaller type and return the new node.
    /// Otherwise, it removes the child and returns None.
    ///
    /// # Returns
    ///
    /// - `Some((removed_child, None))` - Child removed, node unchanged
    /// - `Some((removed_child, Some(new_node)))` - Child removed and node shrunk
    /// - `None` - Key not found
    pub fn remove_child_shrinking(&mut self, key: u8) -> Option<(SwizzledPtr, Option<Node>)> {
        match self {
            Node::N4(n) => {
                n.remove_child(key).map(|removed| (removed, None))
            }
            Node::N16(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 4 {
                        Some((removed, Some(Node::N4(Box::new(n.shrink())))))
                    } else {
                        Some((removed, None))
                    }
                } else {
                    None
                }
            }
            Node::N48(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 16 {
                        Some((removed, Some(Node::N16(Box::new(n.shrink())))))
                    } else {
                        Some((removed, None))
                    }
                } else {
                    None
                }
            }
            Node::N256(n) => {
                if let Some(removed) = n.remove_child(key) {
                    if n.header.num_children <= 48 {
                        Some((removed, Some(Node::N48(Box::new(n.shrink())))))
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

/// Trait for common ART node operations
pub trait ArtNode {
    /// Find a child by key byte
    fn find_child(&self, key: u8) -> Option<&SwizzledPtr>;

    /// Find a child mutably by key byte
    fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr>;

    /// Add a child with the given key
    fn add_child(&mut self, key: u8, child: SwizzledPtr) -> Result<(), AddChildError>;

    /// Remove a child by key
    fn remove_child(&mut self, key: u8) -> Option<SwizzledPtr>;

    /// Check if the node is full
    fn is_full(&self) -> bool;

    /// Get an iterator over (key, child) pairs
    fn iter_children(&self) -> impl Iterator<Item = (u8, &SwizzledPtr)>;
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
    fn test_node_header_flags() {
        let mut header = NodeHeader::new(4);
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
    fn test_node_header_version() {
        let mut header = NodeHeader::new(4);
        assert_eq!(header.version, 0);

        header.increment_version();
        assert_eq!(header.version, 1);

        header.increment_version();
        assert_eq!(header.version, 2);
    }

    #[test]
    fn test_compressed_prefix() {
        let prefix = CompressedPrefix::from_bytes(b"hello");
        assert_eq!(prefix.as_slice(5), b"hello");
        assert_eq!(prefix.as_slice(3), b"hel");
    }

    #[test]
    fn test_prefix_matching() {
        let prefix = CompressedPrefix::from_bytes(b"hello");

        // Full match
        assert_eq!(prefix.match_key(b"hello world", 5), 5);

        // Partial match
        assert_eq!(prefix.match_key(b"help", 5), 3);

        // No match
        assert_eq!(prefix.match_key(b"world", 5), 0);

        // Key shorter than prefix
        assert_eq!(prefix.match_key(b"hel", 5), 3);
    }

    #[test]
    #[should_panic(expected = "prefix too long")]
    fn test_prefix_too_long() {
        CompressedPrefix::from_bytes(b"this is way too long for the prefix");
    }
}
