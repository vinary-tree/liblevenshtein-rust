//! Node48: ART node for 17-48 children with index array lookup.
//!
//! Uses a 256-byte index array to map key bytes to child positions.
//! This provides O(1) lookup while still being more space-efficient than Node256.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ NodeHeader (16 bytes)                                     │
//! ├───────────────────────────────────────────────────────────┤
//! │ CompressedPrefix (12 bytes)                               │
//! ├───────────────────────────────────────────────────────────┤
//! │ index: [u8; 256]  │ Maps key byte -> child slot (or 255) │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 48] │ Child pointers (384 bytes)  │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~668 bytes
//! ```
//!
//! # Index Array
//!
//! - `index[key] == 255` means the key is not present
//! - `index[key] < 48` means the child is at `children[index[key]]`
//!
//! This provides O(1) lookup with a single array access plus bounds check.

use super::{AddChildError, ArtNode, CompressedPrefix, NodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a Node48
pub const NODE48_MAX_CHILDREN: usize = 48;

/// Sentinel value indicating no child at this key
pub const NO_CHILD: u8 = 255;

/// ART node with 17-48 children
///
/// Uses a 256-byte index array for O(1) key lookup.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Node48 {
    /// Common node header
    pub header: NodeHeader,
    /// Compressed prefix for path compression
    pub prefix: CompressedPrefix,
    /// Index array: maps key byte to child slot (255 = no child)
    pub index: [u8; 256],
    /// Child pointers (only first num_children are valid)
    pub children: [SwizzledPtr; NODE48_MAX_CHILDREN],
}

impl Node48 {
    /// Create a new empty Node48
    pub fn new() -> Self {
        Self {
            header: NodeHeader::new(48),
            prefix: CompressedPrefix::empty(),
            index: [NO_CHILD; 256],
            children: std::array::from_fn(|_| SwizzledPtr::null()),
        }
    }

    /// Create a Node48 with a prefix
    pub fn with_prefix(prefix: &[u8]) -> Self {
        let mut node = Self::new();
        node.prefix = CompressedPrefix::from_bytes(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Find the first free slot in the children array
    fn find_free_slot(&self) -> Option<usize> {
        let count = self.header.num_children as usize;
        if count < NODE48_MAX_CHILDREN {
            Some(count)
        } else {
            None
        }
    }
}

impl Default for Node48 {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtNode for Node48 {
    fn find_child(&self, key: u8) -> Option<&SwizzledPtr> {
        let slot = self.index[key as usize];
        if slot == NO_CHILD {
            None
        } else {
            Some(&self.children[slot as usize])
        }
    }

    fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr> {
        let slot = self.index[key as usize];
        if slot == NO_CHILD {
            None
        } else {
            Some(&mut self.children[slot as usize])
        }
    }

    fn add_child(&mut self, key: u8, child: SwizzledPtr) -> Result<(), AddChildError> {
        // Check for duplicate
        if self.index[key as usize] != NO_CHILD {
            return Err(AddChildError::KeyExists);
        }

        // Find a free slot
        let slot = self.find_free_slot().ok_or(AddChildError::NodeFull)?;

        // Add the child
        self.index[key as usize] = slot as u8;
        self.children[slot] = child;
        self.header.num_children += 1;

        Ok(())
    }

    fn remove_child(&mut self, key: u8) -> Option<SwizzledPtr> {
        let slot = self.index[key as usize];
        if slot == NO_CHILD {
            return None;
        }

        let removed = self.children[slot as usize].clone();
        self.index[key as usize] = NO_CHILD;

        // Move the last child to fill the gap (if not already the last)
        let last = self.header.num_children as usize - 1;
        if (slot as usize) != last {
            // Find the key that points to the last slot
            for i in 0..256 {
                if self.index[i] == last as u8 {
                    self.index[i] = slot;
                    break;
                }
            }
            self.children[slot as usize] = self.children[last].clone();
        }

        self.children[last] = SwizzledPtr::null();
        self.header.num_children -= 1;

        Some(removed)
    }

    fn is_full(&self) -> bool {
        self.header.num_children as usize >= NODE48_MAX_CHILDREN
    }

    fn iter_children(&self) -> impl Iterator<Item = (u8, &SwizzledPtr)> {
        self.index
            .iter()
            .enumerate()
            .filter_map(|(key, &slot)| {
                if slot != NO_CHILD {
                    Some((key as u8, &self.children[slot as usize]))
                } else {
                    None
                }
            })
    }
}

impl Node48 {
    /// Shrink this node to a Node16
    pub fn shrink(&self) -> super::Node16 {
        debug_assert!(
            self.header.num_children <= 16,
            "cannot shrink Node48 with {} children",
            self.header.num_children
        );

        let mut node16 = super::Node16::new();
        node16.header = self.header.clone();
        node16.header.node_type = 16;
        node16.prefix = self.prefix;

        // Collect keys in sorted order
        let mut idx = 0;
        for key in 0..=255u8 {
            let slot = self.index[key as usize];
            if slot != NO_CHILD {
                node16.keys[idx] = key;
                node16.children[idx] = self.children[slot as usize].clone();
                idx += 1;
            }
        }

        node16
    }

    /// Grow this node to a Node256
    pub fn grow(&self) -> super::Node256 {
        let mut node256 = super::Node256::new();
        node256.header = self.header.clone();
        node256.header.node_type = 0; // Node256 uses 0
        node256.prefix = self.prefix;

        // Copy children to direct array
        for key in 0..=255u8 {
            let slot = self.index[key as usize];
            if slot != NO_CHILD {
                node256.children[key as usize] = self.children[slot as usize].clone();
            }
        }

        node256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node48() {
        let node = Node48::new();
        assert_eq!(node.header.node_type, 48);
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = Node48::new();

        // Add children with various keys
        for key in [0, 50, 100, 150, 200, 255u8] {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 6);

        // Find all children
        for key in [0, 50, 100, 150, 200, 255u8] {
            assert!(
                node.find_child(key).is_some(),
                "should find key {}",
                key
            );
        }

        // Should not find non-existent keys
        assert!(node.find_child(1).is_none());
        assert!(node.find_child(51).is_none());
    }

    #[test]
    fn test_node48_full() {
        let mut node = Node48::new();

        for i in 0..48 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(i as u8, child).is_ok());
        }

        assert!(node.is_full());

        let child = SwizzledPtr::on_disk(48, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(48, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node48::new();

        for i in 0..20 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i as u8, child).expect("add should succeed");
        }

        // Remove middle element
        let removed = node.remove_child(10);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 19);
        assert!(node.find_child(10).is_none());

        // Other children should still be present
        for i in 0..20 {
            if i != 10 {
                assert!(node.find_child(i).is_some(), "should find key {}", i);
            }
        }
    }

    #[test]
    fn test_iter_children() {
        let mut node = Node48::new();

        let keys = [5, 10, 15, 20, 25u8];
        for &key in &keys {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(key, child).expect("add should succeed");
        }

        let found_keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(found_keys, keys.to_vec()); // iter_children returns sorted
    }

    #[test]
    fn test_shrink_to_node16() {
        let mut node = Node48::new();

        for i in 0..16 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i as u8, child).expect("add should succeed");
        }

        let node16 = node.shrink();
        assert_eq!(node16.header.node_type, 16);
        assert_eq!(node16.header.num_children, 16);

        // Verify children transferred
        for i in 0..16 {
            assert!(node16.find_child(i as u8).is_some());
        }
    }

    #[test]
    fn test_duplicate_key() {
        let mut node = Node48::new();

        let child = SwizzledPtr::on_disk(1, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert!(node.add_child(42, child).is_ok());

        let child = SwizzledPtr::on_disk(2, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(42, child), Err(AddChildError::KeyExists));
    }
}
