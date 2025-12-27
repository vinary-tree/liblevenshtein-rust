//! Node4: ART node for 1-4 children with linear scan lookup.
//!
//! This is the smallest node type, optimized for sparse nodes with few children.
//! Uses simple linear scan which is efficient for <= 4 elements.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ NodeHeader (16 bytes)                                     │
//! ├───────────────────────────────────────────────────────────┤
//! │ CompressedPrefix (12 bytes)                               │
//! ├───────────────────────────────────────────────────────────┤
//! │ keys: [u8; 4]     │ Key bytes for each child             │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 4]  │ Child pointers (32 bytes)  │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~64 bytes (fits in one cache line)
//! ```

use super::{AddChildError, ArtNode, CompressedPrefix, NodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a Node4
pub const NODE4_MAX_CHILDREN: usize = 4;

/// ART node with 1-4 children
///
/// Uses linear scan for lookup, which is optimal for small arrays.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Node4 {
    /// Common node header
    pub header: NodeHeader,
    /// Compressed prefix for path compression
    pub prefix: CompressedPrefix,
    /// Key bytes (sorted for predictable iteration order)
    pub keys: [u8; NODE4_MAX_CHILDREN],
    /// Child pointers corresponding to keys
    pub children: [SwizzledPtr; NODE4_MAX_CHILDREN],
}

impl Node4 {
    /// Create a new empty Node4
    pub fn new() -> Self {
        Self {
            header: NodeHeader::new(4),
            prefix: CompressedPrefix::empty(),
            keys: [0; NODE4_MAX_CHILDREN],
            children: [
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
            ],
        }
    }

    /// Create a Node4 with a prefix
    pub fn with_prefix(prefix: &[u8]) -> Self {
        let mut node = Self::new();
        node.prefix = CompressedPrefix::from_bytes(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Find the index of a key, or where it should be inserted
    fn find_key_index(&self, key: u8) -> Result<usize, usize> {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] == key {
                return Ok(i);
            }
            if self.keys[i] > key {
                return Err(i);
            }
        }
        Err(count)
    }
}

impl Default for Node4 {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtNode for Node4 {
    fn find_child(&self, key: u8) -> Option<&SwizzledPtr> {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] == key {
                return Some(&self.children[i]);
            }
        }
        None
    }

    fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr> {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] == key {
                return Some(&mut self.children[i]);
            }
        }
        None
    }

    fn add_child(&mut self, key: u8, child: SwizzledPtr) -> Result<(), AddChildError> {
        let count = self.header.num_children as usize;

        if count >= NODE4_MAX_CHILDREN {
            return Err(AddChildError::NodeFull);
        }

        // Find insertion point (keep sorted)
        match self.find_key_index(key) {
            Ok(_) => return Err(AddChildError::KeyExists),
            Err(insert_pos) => {
                // Shift elements to make room
                for i in (insert_pos..count).rev() {
                    self.keys[i + 1] = self.keys[i];
                    // Clone the swizzled pointer (atomic load/store)
                    self.children[i + 1] = self.children[i].clone();
                }

                self.keys[insert_pos] = key;
                self.children[insert_pos] = child;
                self.header.num_children += 1;
                Ok(())
            }
        }
    }

    fn remove_child(&mut self, key: u8) -> Option<SwizzledPtr> {
        let count = self.header.num_children as usize;

        if let Ok(index) = self.find_key_index(key) {
            let removed = self.children[index].clone();

            // Shift elements down
            for i in index..(count - 1) {
                self.keys[i] = self.keys[i + 1];
                self.children[i] = self.children[i + 1].clone();
            }

            // Clear the last slot
            self.keys[count - 1] = 0;
            self.children[count - 1] = SwizzledPtr::null();
            self.header.num_children -= 1;

            Some(removed)
        } else {
            None
        }
    }

    fn is_full(&self) -> bool {
        self.header.num_children as usize >= NODE4_MAX_CHILDREN
    }

    fn iter_children(&self) -> impl Iterator<Item = (u8, &SwizzledPtr)> {
        let count = self.header.num_children as usize;
        self.keys[..count]
            .iter()
            .zip(self.children[..count].iter())
            .map(|(&k, c)| (k, c))
    }
}

impl Node4 {
    /// Grow this node into a Node16
    pub fn grow(&self) -> super::Node16 {
        let mut node16 = super::Node16::new();
        node16.header = self.header.clone();
        node16.header.node_type = 16;
        node16.prefix = self.prefix;

        // Copy children
        let count = self.header.num_children as usize;
        for i in 0..count {
            node16.keys[i] = self.keys[i];
            node16.children[i] = self.children[i].clone();
        }

        node16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node4() {
        let node = Node4::new();
        assert_eq!(node.header.node_type, 4);
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = Node4::new();

        // Add children in random order
        let child_c = SwizzledPtr::on_disk(1, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        let child_a = SwizzledPtr::on_disk(2, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        let child_b = SwizzledPtr::on_disk(3, 0, crate::dictionary::persistent_artrie::NodeType::Node4);

        assert!(node.add_child(b'c', child_c).is_ok());
        assert!(node.add_child(b'a', child_a).is_ok());
        assert!(node.add_child(b'b', child_b).is_ok());

        assert_eq!(node.header.num_children, 3);

        // Keys should be sorted
        assert_eq!(node.keys[0], b'a');
        assert_eq!(node.keys[1], b'b');
        assert_eq!(node.keys[2], b'c');

        // Find children
        assert!(node.find_child(b'a').is_some());
        assert!(node.find_child(b'b').is_some());
        assert!(node.find_child(b'c').is_some());
        assert!(node.find_child(b'd').is_none());
    }

    #[test]
    fn test_node4_full() {
        let mut node = Node4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i as u32, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
            assert!(node.add_child(i as u8, child).is_ok());
        }

        assert!(node.is_full());

        // Adding one more should fail
        let child = SwizzledPtr::on_disk(4, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(4, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i as u32, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
            node.add_child(i as u8, child).expect("add should succeed");
        }

        assert_eq!(node.header.num_children, 4);

        // Remove middle element
        let removed = node.remove_child(2);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 3);
        assert!(node.find_child(2).is_none());

        // Other children should still be present
        assert!(node.find_child(0).is_some());
        assert!(node.find_child(1).is_some());
        assert!(node.find_child(3).is_some());
    }

    #[test]
    fn test_duplicate_key() {
        let mut node = Node4::new();

        let child = SwizzledPtr::on_disk(1, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert!(node.add_child(b'a', child).is_ok());

        let child = SwizzledPtr::on_disk(2, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(b'a', child), Err(AddChildError::KeyExists));
    }

    #[test]
    fn test_iter_children() {
        let mut node = Node4::new();

        for i in 0..3 {
            let child = SwizzledPtr::on_disk(i as u32, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
            node.add_child(b'a' + i, child).expect("add should succeed");
        }

        let children: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(children, vec![b'a', b'b', b'c']);
    }

    #[test]
    fn test_with_prefix() {
        let node = Node4::with_prefix(b"hello");
        assert_eq!(node.header.prefix_len, 5);
        assert_eq!(node.prefix.as_slice(5), b"hello");
    }
}
