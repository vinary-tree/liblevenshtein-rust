//! Node256: ART node for 49-256 children with direct array lookup.
//!
//! Uses a direct 256-element array where each position corresponds to a key byte.
//! This provides O(1) lookup with no bounds checking needed.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ NodeHeader (16 bytes)                                     │
//! ├───────────────────────────────────────────────────────────┤
//! │ CompressedPrefix (12 bytes)                               │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 256] │ Direct array (2048 bytes)  │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~2076 bytes
//! ```
//!
//! # Tradeoffs
//!
//! - **Pros**: O(1) lookup with no search, best for dense nodes
//! - **Cons**: Large memory footprint even for sparse occupancy
//!
//! Use Node256 when a node has more than 48 children.

use super::{AddChildError, ArtNode, CompressedPrefix, NodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a Node256 (all 256 possible byte values)
pub const NODE256_MAX_CHILDREN: usize = 256;

/// ART node with 49-256 children
///
/// Uses a direct 256-element array for O(1) key lookup.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Node256 {
    /// Common node header (node_type = 0 for Node256)
    pub header: NodeHeader,
    /// Compressed prefix for path compression
    pub prefix: CompressedPrefix,
    /// Direct child array: children[key] is the child for that key byte
    pub children: [SwizzledPtr; NODE256_MAX_CHILDREN],
}

impl Node256 {
    /// Create a new empty Node256
    pub fn new() -> Self {
        Self {
            header: NodeHeader::new(0), // 0 indicates Node256
            prefix: CompressedPrefix::empty(),
            children: std::array::from_fn(|_| SwizzledPtr::null()),
        }
    }

    /// Create a Node256 with a prefix
    pub fn with_prefix(prefix: &[u8]) -> Self {
        let mut node = Self::new();
        node.prefix = CompressedPrefix::from_bytes(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }
}

impl Default for Node256 {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtNode for Node256 {
    fn find_child(&self, key: u8) -> Option<&SwizzledPtr> {
        let child = &self.children[key as usize];
        if child.is_null() {
            None
        } else {
            Some(child)
        }
    }

    fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr> {
        let child = &mut self.children[key as usize];
        if child.is_null() {
            None
        } else {
            Some(child)
        }
    }

    fn add_child(&mut self, key: u8, child: SwizzledPtr) -> Result<(), AddChildError> {
        // Check for duplicate
        if !self.children[key as usize].is_null() {
            return Err(AddChildError::KeyExists);
        }

        self.children[key as usize] = child;
        self.header.num_children += 1;

        Ok(())
    }

    fn remove_child(&mut self, key: u8) -> Option<SwizzledPtr> {
        let slot = &mut self.children[key as usize];
        if slot.is_null() {
            return None;
        }

        let removed = slot.clone();
        *slot = SwizzledPtr::null();
        self.header.num_children -= 1;

        Some(removed)
    }

    fn is_full(&self) -> bool {
        // Node256 can never be full for practical purposes
        // (would need all 256 children populated)
        self.header.num_children as usize >= NODE256_MAX_CHILDREN
    }

    fn iter_children(&self) -> impl Iterator<Item = (u8, &SwizzledPtr)> {
        self.children
            .iter()
            .enumerate()
            .filter_map(|(key, child)| {
                if child.is_null() {
                    None
                } else {
                    Some((key as u8, child))
                }
            })
    }
}

impl Node256 {
    /// Shrink this node to a Node48
    pub fn shrink(&self) -> super::Node48 {
        debug_assert!(
            self.header.num_children <= 48,
            "cannot shrink Node256 with {} children",
            self.header.num_children
        );

        let mut node48 = super::Node48::new();
        node48.header = self.header.clone();
        node48.header.node_type = 48;
        node48.prefix = self.prefix;

        // Copy non-null children
        let mut slot = 0;
        for key in 0..=255u8 {
            if !self.children[key as usize].is_null() {
                node48.index[key as usize] = slot;
                node48.children[slot as usize] = self.children[key as usize].clone();
                slot += 1;
            }
        }

        node48
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node256() {
        let node = Node256::new();
        assert_eq!(node.header.node_type, 0); // Node256 uses 0
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = Node256::new();

        // Add children at various positions
        for key in [0, 64, 128, 192, 255u8] {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 5);

        // Find all children
        for key in [0, 64, 128, 192, 255u8] {
            assert!(
                node.find_child(key).is_some(),
                "should find key {}",
                key
            );
        }

        // Should not find non-existent keys
        assert!(node.find_child(1).is_none());
        assert!(node.find_child(127).is_none());
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node256::new();

        for i in [0, 50, 100, 150, 200, 250u8] {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i, child).expect("add should succeed");
        }

        // Remove middle element
        let removed = node.remove_child(100);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 5);
        assert!(node.find_child(100).is_none());

        // Other children should still be present
        for key in [0, 50, 150, 200, 250u8] {
            assert!(node.find_child(key).is_some());
        }
    }

    #[test]
    fn test_iter_children() {
        let mut node = Node256::new();

        let keys = [10, 20, 30, 40, 50u8];
        for &key in &keys {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(key, child).expect("add should succeed");
        }

        let found_keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(found_keys, keys.to_vec());
    }

    #[test]
    fn test_shrink_to_node48() {
        let mut node = Node256::new();

        for i in 0..48 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i as u8, child).expect("add should succeed");
        }

        let node48 = node.shrink();
        assert_eq!(node48.header.node_type, 48);
        assert_eq!(node48.header.num_children, 48);

        // Verify children transferred
        for i in 0..48 {
            assert!(node48.find_child(i as u8).is_some());
        }
    }

    #[test]
    fn test_duplicate_key() {
        let mut node = Node256::new();

        let child = SwizzledPtr::on_disk(1, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert!(node.add_child(100, child).is_ok());

        let child = SwizzledPtr::on_disk(2, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(100, child), Err(AddChildError::KeyExists));
    }

    #[test]
    fn test_dense_node256() {
        let mut node = Node256::new();

        // Add all 256 children
        for key in 0..=255u8 {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(key, child).is_ok());
        }

        assert!(node.is_full());
        // num_children wraps around at 256 (u8 max)
        // For a full Node256, we check is_full() instead
        assert_eq!(node.iter_children().count(), 256);

        // All keys should be findable
        for key in 0..=255u8 {
            assert!(node.find_child(key).is_some());
        }
    }
}
