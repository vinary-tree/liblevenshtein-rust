//! CharNode4: ART node for 1-4 children with linear scan lookup.
//!
//! This is the smallest node type, used when a node has very few children.
//! Lookup is performed via simple linear scan, which is efficient for ≤4 elements.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ CharNodeHeader (16 bytes)                                 │
//! ├───────────────────────────────────────────────────────────┤
//! │ CharCompressedPrefix (24 bytes) - 6 u32 chars             │
//! ├───────────────────────────────────────────────────────────┤
//! │ keys: [u32; 4]         │ Key characters (16 bytes)        │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 4] │ Child pointers (32 bytes)    │
//! ├───────────────────────────────────────────────────────────┤
//! │ value_ptr: SwizzledPtr │ Value pointer for final nodes    │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~96 bytes
//! ```

use super::{AddChildError, CharArtNode, CharCompressedPrefix, CharNodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a CharNode4
pub const CHARNODE4_MAX_CHILDREN: usize = 4;

/// ART node with 1-4 children (character-level)
///
/// Uses simple linear scan for lookup, which is cache-friendly and fast
/// for small numbers of children.
#[repr(C, align(8))]
#[derive(Debug, Clone)]
pub struct CharNode4 {
    /// Common node header
    pub header: CharNodeHeader,
    /// Compressed prefix for path compression (6 chars max)
    pub prefix: CharCompressedPrefix,
    /// Key characters (u32 for full Unicode support)
    pub keys: [u32; CHARNODE4_MAX_CHILDREN],
    /// Child pointers corresponding to keys
    pub children: [SwizzledPtr; CHARNODE4_MAX_CHILDREN],
    /// Optional value pointer (for nodes that are final)
    pub value_ptr: SwizzledPtr,
}

impl CharNode4 {
    /// Create a new empty CharNode4
    pub fn new() -> Self {
        Self {
            header: CharNodeHeader::new(104), // CHARNODE4 type
            prefix: CharCompressedPrefix::empty(),
            keys: [0; CHARNODE4_MAX_CHILDREN],
            children: [
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
            ],
            value_ptr: SwizzledPtr::null(),
        }
    }

    /// Create a CharNode4 with a prefix
    pub fn with_prefix(prefix: &[u32]) -> Self {
        let mut node = Self::new();
        node.prefix = CharCompressedPrefix::from_chars(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Create a CharNode4 from a character iterator (convenience method)
    pub fn with_prefix_chars<I: IntoIterator<Item = char>>(chars: I) -> Self {
        let mut node = Self::new();
        node.prefix = CharCompressedPrefix::from_char_iter(chars);
        // Note: caller should set prefix_len appropriately
        node
    }

    /// Find a key using linear scan
    fn find_key_index(&self, key: u32) -> Option<usize> {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] == key {
                return Some(i);
            }
        }
        None
    }

    /// Find the insertion point for a key (maintains sorted order)
    fn find_insert_point(&self, key: u32) -> usize {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] >= key {
                return i;
            }
        }
        count
    }

    /// Grow this node to a CharNode16
    pub fn grow(&self) -> super::CharNode16 {
        let mut node16 = super::CharNode16::new();
        node16.header = self.header.clone();
        node16.header.node_type = 16;
        node16.prefix = self.prefix;
        node16.value_ptr = self.value_ptr.clone();

        let count = self.header.num_children as usize;
        for i in 0..count {
            node16.keys[i] = self.keys[i];
            node16.children[i] = self.children[i].clone();
        }

        node16
    }
}

impl Default for CharNode4 {
    fn default() -> Self {
        Self::new()
    }
}

impl CharArtNode for CharNode4 {
    fn find_child(&self, key: u32) -> Option<&SwizzledPtr> {
        self.find_key_index(key).map(|i| &self.children[i])
    }

    fn find_child_mut(&mut self, key: u32) -> Option<&mut SwizzledPtr> {
        if let Some(i) = self.find_key_index(key) {
            Some(&mut self.children[i])
        } else {
            None
        }
    }

    fn add_child(&mut self, key: u32, child: SwizzledPtr) -> Result<(), AddChildError> {
        let count = self.header.num_children as usize;

        if count >= CHARNODE4_MAX_CHILDREN {
            return Err(AddChildError::NodeFull);
        }

        // Check for duplicate
        if self.find_key_index(key).is_some() {
            return Err(AddChildError::KeyExists);
        }

        // Find insertion point (keep sorted)
        let insert_pos = self.find_insert_point(key);

        // Shift elements to make room
        for i in (insert_pos..count).rev() {
            self.keys[i + 1] = self.keys[i];
            self.children[i + 1] = self.children[i].clone();
        }

        self.keys[insert_pos] = key;
        self.children[insert_pos] = child;
        self.header.num_children += 1;
        Ok(())
    }

    fn remove_child(&mut self, key: u32) -> Option<SwizzledPtr> {
        let count = self.header.num_children as usize;

        if let Some(index) = self.find_key_index(key) {
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
        self.header.num_children as usize >= CHARNODE4_MAX_CHILDREN
    }

    fn iter_children(&self) -> impl Iterator<Item = (u32, &SwizzledPtr)> {
        let count = self.header.num_children as usize;
        self.keys[..count]
            .iter()
            .zip(self.children[..count].iter())
            .map(|(&k, c)| (k, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::persistent_artrie::NodeType;

    #[test]
    fn test_new_charnode4() {
        let node = CharNode4::new();
        assert_eq!(node.header.node_type, 104); // CHARNODE4
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = CharNode4::new();

        // Add children (Unicode code points)
        let keys = ['a' as u32, 'z' as u32, 'm' as u32, 'f' as u32];
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 4);

        // Keys should be sorted: a, f, m, z
        let sorted_keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(sorted_keys, vec!['a' as u32, 'f' as u32, 'm' as u32, 'z' as u32]);

        // Find all children
        for &key in &keys {
            assert!(
                node.find_child(key).is_some(),
                "should find key '{}'",
                char::from_u32(key).unwrap_or('?')
            );
        }

        // Should not find non-existent keys
        assert!(node.find_child('x' as u32).is_none());
    }

    #[test]
    fn test_add_unicode_children() {
        let mut node = CharNode4::new();

        // Add Unicode characters (emoji, CJK)
        let keys = ['🎉' as u32, '日' as u32, '本' as u32, 'α' as u32];
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 4);

        // All should be findable
        for &key in &keys {
            assert!(node.find_child(key).is_some());
        }
    }

    #[test]
    fn test_charnode4_full() {
        let mut node = CharNode4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            assert!(node.add_child(i, child).is_ok());
        }

        assert!(node.is_full());

        let child = SwizzledPtr::on_disk(4, 0, NodeType::Node4);
        assert_eq!(node.add_child(4, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_duplicate_key() {
        let mut node = CharNode4::new();

        let child = SwizzledPtr::on_disk(42, 0, NodeType::Node4);
        assert!(node.add_child(42, child.clone()).is_ok());
        assert_eq!(node.add_child(42, child), Err(AddChildError::KeyExists));
    }

    #[test]
    fn test_remove_child() {
        let mut node = CharNode4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        // Remove middle element
        let removed = node.remove_child(2);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 3);
        assert!(node.find_child(2).is_none());

        // Other children should still be present
        assert!(node.find_child(0).is_some());
        assert!(node.find_child(1).is_some());
        assert!(node.find_child(3).is_some());

        // Keys should still be sorted: 0, 1, 3
        let keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![0, 1, 3]);
    }

    #[test]
    fn test_iter_children() {
        let mut node = CharNode4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(b'a' as u32 + i, child).expect("add should succeed");
        }

        let keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec!['a' as u32, 'b' as u32, 'c' as u32, 'd' as u32]
        );
    }

    #[test]
    fn test_with_prefix() {
        let prefix: Vec<u32> = "test".chars().map(|c| c as u32).collect();
        let node = CharNode4::with_prefix(&prefix);

        assert_eq!(node.header.prefix_len, 4);
        assert_eq!(node.prefix.to_chars(4), vec!['t', 'e', 's', 't']);
    }

    #[test]
    fn test_grow_to_node16() {
        let mut node = CharNode4::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        node.header.set_final(true);
        let prefix: Vec<u32> = "abc".chars().map(|c| c as u32).collect();
        node.prefix = CharCompressedPrefix::from_chars(&prefix);
        node.header.prefix_len = 3;

        let node16 = node.grow();

        // Verify type changed
        assert_eq!(node16.header.node_type, 16);
        assert_eq!(node16.header.num_children, 4);

        // Verify all children transferred
        for i in 0..4 {
            assert!(node16.find_child(i).is_some());
        }

        // Verify prefix and flags preserved
        assert!(node16.header.is_final());
        assert_eq!(node16.prefix.to_chars(3), vec!['a', 'b', 'c']);
    }
}
