//! CharNode48: ART node for 17-48 children with binary search lookup.
//!
//! Unlike byte-level Node48 which uses a 256-byte index array for O(1) lookup,
//! CharNode48 cannot use that approach (would require a 4GB index for u32 keys).
//! Instead, we maintain sorted keys and use binary search for O(log 48) lookup.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ CharNodeHeader (16 bytes)                                 │
//! ├───────────────────────────────────────────────────────────┤
//! │ CharCompressedPrefix (24 bytes) - 6 u32 chars             │
//! ├───────────────────────────────────────────────────────────┤
//! │ keys: [u32; 48]           │ Sorted key chars (192 bytes)  │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 48] │ Child pointers (384 bytes)  │
//! ├───────────────────────────────────────────────────────────┤
//! │ value_ptr: SwizzledPtr    │ Value pointer for final nodes │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~624 bytes
//! ```
//!
//! # Performance
//!
//! Binary search on 48 sorted u32 keys requires at most 6 comparisons (log₂(48) ≈ 5.58).
//! This is still very fast and significantly better than linear scan for 48 elements.

use super::{AddChildError, CharArtNode, CharCompressedPrefix, CharNodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a CharNode48
pub const CHARNODE48_MAX_CHILDREN: usize = 48;

/// ART node with 17-48 children (character-level)
///
/// Uses sorted keys with binary search for O(log 48) lookup.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CharNode48 {
    /// Common node header
    pub header: CharNodeHeader,
    /// Compressed prefix for path compression (6 chars max)
    pub prefix: CharCompressedPrefix,
    /// Sorted key characters
    pub keys: [u32; CHARNODE48_MAX_CHILDREN],
    /// Child pointers corresponding to keys (same order as keys)
    pub children: [SwizzledPtr; CHARNODE48_MAX_CHILDREN],
    /// Optional value pointer (for nodes that are final)
    pub value_ptr: SwizzledPtr,
}

impl CharNode48 {
    /// Create a new empty CharNode48
    pub fn new() -> Self {
        Self {
            header: CharNodeHeader::new(148), // CHARNODE48 type
            prefix: CharCompressedPrefix::empty(),
            keys: [0; CHARNODE48_MAX_CHILDREN],
            children: std::array::from_fn(|_| SwizzledPtr::null()),
            value_ptr: SwizzledPtr::null(),
        }
    }

    /// Create a CharNode48 with a prefix
    pub fn with_prefix(prefix: &[u32]) -> Self {
        let mut node = Self::new();
        node.prefix = CharCompressedPrefix::from_chars(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Find a key using binary search
    ///
    /// Returns the index if found, None otherwise.
    fn find_key_index(&self, key: u32) -> Option<usize> {
        let count = self.header.num_children as usize;
        if count == 0 {
            return None;
        }

        match self.keys[..count].binary_search(&key) {
            Ok(index) => Some(index),
            Err(_) => None,
        }
    }

    /// Find the insertion point for a key using binary search
    fn find_insert_point(&self, key: u32) -> usize {
        let count = self.header.num_children as usize;
        match self.keys[..count].binary_search(&key) {
            Ok(index) => index,  // Key exists (shouldn't happen in add_child)
            Err(index) => index, // Where key should be inserted
        }
    }

    /// Shrink this node to a CharNode16
    pub fn shrink(&self) -> super::CharNode16 {
        debug_assert!(
            self.header.num_children <= 16,
            "cannot shrink CharNode48 with {} children",
            self.header.num_children
        );

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

    /// Grow this node to a CharBucket
    pub fn grow(&self) -> super::CharBucket {
        let mut bucket = super::CharBucket::new();
        bucket.header = self.header.clone();
        bucket.header.node_type = 49; // Use 49 to distinguish from Node48
        bucket.prefix = self.prefix;
        bucket.value_ptr = self.value_ptr.clone();

        let count = self.header.num_children as usize;
        for i in 0..count {
            bucket.entries.insert(self.keys[i], self.children[i].clone());
        }

        bucket
    }
}

impl Default for CharNode48 {
    fn default() -> Self {
        Self::new()
    }
}

impl CharArtNode for CharNode48 {
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

        if count >= CHARNODE48_MAX_CHILDREN {
            return Err(AddChildError::NodeFull);
        }

        // Check for duplicate using binary search
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
        self.header.num_children as usize >= CHARNODE48_MAX_CHILDREN
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
    fn test_new_charnode48() {
        let node = CharNode48::new();
        assert_eq!(node.header.node_type, 148); // CHARNODE48
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = CharNode48::new();

        // Add children in random order
        let keys: Vec<u32> = vec![50, 10, 30, 70, 20, 40, 60, 80];
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 8);

        // Keys should be sorted
        let sorted: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(sorted, vec![10, 20, 30, 40, 50, 60, 70, 80]);

        // Find all children
        for &key in &keys {
            assert!(node.find_child(key).is_some(), "should find key {}", key);
        }

        // Should not find non-existent keys
        assert!(node.find_child(100).is_none());
    }

    #[test]
    fn test_binary_search_correctness() {
        let mut node = CharNode48::new();

        // Add 30 keys to exercise binary search properly
        for i in (0..30).map(|x| x * 1000) {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        // Verify all are findable
        for i in (0..30).map(|x| x * 1000) {
            assert!(node.find_child(i).is_some(), "should find key {}", i);
        }

        // Verify gaps are not found
        for i in (0..30).map(|x| x * 1000 + 1) {
            assert!(node.find_child(i).is_none(), "should not find key {}", i);
        }
    }

    #[test]
    fn test_charnode48_full() {
        let mut node = CharNode48::new();

        for i in 0..48 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            assert!(node.add_child(i, child).is_ok());
        }

        assert!(node.is_full());

        let child = SwizzledPtr::on_disk(48, 0, NodeType::Node4);
        assert_eq!(node.add_child(48, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_remove_child() {
        let mut node = CharNode48::new();

        for i in 0..30 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        // Remove middle element
        let removed = node.remove_child(15);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 29);
        assert!(node.find_child(15).is_none());

        // Other children should still be present
        for i in 0..30 {
            if i != 15 {
                assert!(node.find_child(i).is_some());
            }
        }

        // Verify still sorted
        let keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        for window in keys.windows(2) {
            assert!(window[0] < window[1], "keys should be sorted");
        }
    }

    #[test]
    fn test_shrink_to_node16() {
        let mut node = CharNode48::new();

        for i in 0..16 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        node.header.set_final(true);
        let node16 = node.shrink();

        assert_eq!(node16.header.node_type, 16);
        assert_eq!(node16.header.num_children, 16);
        assert!(node16.header.is_final());

        // Verify children transferred
        for i in 0..16 {
            assert!(node16.find_child(i).is_some());
        }
    }

    #[test]
    fn test_unicode_keys() {
        let mut node = CharNode48::new();

        // Add Unicode code points
        let keys: Vec<u32> = "αβγδεζηθικλμνξοπρστυφχψω".chars().map(|c| c as u32).collect();
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 24);

        // All should be findable via binary search
        for &key in &keys {
            assert!(
                node.find_child(key).is_some(),
                "should find key {}",
                char::from_u32(key).unwrap_or('?')
            );
        }
    }
}
