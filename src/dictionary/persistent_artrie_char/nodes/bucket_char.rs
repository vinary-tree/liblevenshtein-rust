//! CharBucket: HashMap-like storage for >48 children.
//!
//! Since CharNode256 is impossible for u32 keys (would require 4GB array),
//! CharBucket handles nodes with more than 48 children using a HashMap.
//!
//! This is the final node type in the char variant growth chain:
//! CharNode4 → CharNode16 → CharNode48 → CharBucket
//!
//! # Performance
//!
//! - Lookup: O(1) average (HashMap)
//! - Insert: O(1) average (HashMap)
//! - Iteration: O(n) where n is number of children
//!
//! # Memory Usage
//!
//! Memory usage is proportional to the number of children. For typical use cases
//! with 49-256 children, this is comparable to a byte-level Node256.

use std::collections::HashMap;

use super::{AddChildError, CharArtNode, CharCompressedPrefix, CharNodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Minimum children before shrinking to CharNode48
pub const CHARBUCKET_SHRINK_THRESHOLD: usize = 48;

/// ART node with >48 children (character-level)
///
/// Uses HashMap for O(1) average-case lookup and insertion.
/// This replaces Node256 which is impossible for u32 keys.
#[derive(Debug, Clone)]
pub struct CharBucket {
    /// Common node header
    pub header: CharNodeHeader,
    /// Compressed prefix for path compression (6 chars max)
    pub prefix: CharCompressedPrefix,
    /// Children stored in HashMap for O(1) lookup
    pub entries: HashMap<u32, SwizzledPtr>,
    /// Optional value pointer (for nodes that are final)
    pub value_ptr: SwizzledPtr,
}

impl CharBucket {
    /// Create a new empty CharBucket
    pub fn new() -> Self {
        Self {
            header: CharNodeHeader::new(101), // CHARBUCKET type
            prefix: CharCompressedPrefix::empty(),
            entries: HashMap::with_capacity(64),
            value_ptr: SwizzledPtr::null(),
        }
    }

    /// Create a CharBucket with a prefix
    pub fn with_prefix(prefix: &[u32]) -> Self {
        let mut node = Self::new();
        node.prefix = CharCompressedPrefix::from_chars(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Shrink this bucket to a CharNode48
    ///
    /// Should only be called when num_children <= 48.
    pub fn shrink(&self) -> super::CharNode48 {
        debug_assert!(
            self.header.num_children as usize <= CHARBUCKET_SHRINK_THRESHOLD,
            "cannot shrink CharBucket with {} children",
            self.header.num_children
        );

        let mut node48 = super::CharNode48::new();
        node48.header = self.header.clone();
        node48.header.node_type = 48;
        node48.prefix = self.prefix;
        node48.value_ptr = self.value_ptr.clone();

        // Collect entries and sort by key
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|&(k, _)| *k);

        for (i, (key, child)) in entries.iter().enumerate() {
            node48.keys[i] = **key;
            node48.children[i] = (*child).clone();
        }

        node48
    }

    /// Get the number of children
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the bucket is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CharBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl CharArtNode for CharBucket {
    fn find_child(&self, key: u32) -> Option<&SwizzledPtr> {
        self.entries.get(&key)
    }

    fn find_child_mut(&mut self, key: u32) -> Option<&mut SwizzledPtr> {
        self.entries.get_mut(&key)
    }

    fn add_child(&mut self, key: u32, child: SwizzledPtr) -> Result<(), AddChildError> {
        // Check for duplicate
        if self.entries.contains_key(&key) {
            return Err(AddChildError::KeyExists);
        }

        self.entries.insert(key, child);
        self.header.num_children += 1;
        Ok(())
    }

    fn remove_child(&mut self, key: u32) -> Option<SwizzledPtr> {
        if let Some(removed) = self.entries.remove(&key) {
            self.header.num_children -= 1;
            Some(removed)
        } else {
            None
        }
    }

    fn is_full(&self) -> bool {
        // CharBucket can grow indefinitely
        false
    }

    fn iter_children(&self) -> impl Iterator<Item = (u32, &SwizzledPtr)> {
        self.entries.iter().map(|(&k, c)| (k, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::persistent_artrie::NodeType;

    #[test]
    fn test_new_charbucket() {
        let node = CharBucket::new();
        assert_eq!(node.header.node_type, 101); // CHARBUCKET
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full()); // Never full
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = CharBucket::new();

        // Add many children
        for i in 0..100 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            assert!(node.add_child(i, child).is_ok());
        }

        assert_eq!(node.header.num_children, 100);
        assert_eq!(node.len(), 100);

        // Find all children
        for i in 0..100 {
            assert!(node.find_child(i).is_some(), "should find key {}", i);
        }

        // Should not find non-existent keys
        assert!(node.find_child(200).is_none());
    }

    #[test]
    fn test_charbucket_never_full() {
        let mut node = CharBucket::new();

        // Add hundreds of children
        for i in 0..500 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            assert!(node.add_child(i, child).is_ok());
            assert!(!node.is_full()); // Should never be full
        }

        assert_eq!(node.header.num_children, 500);
    }

    #[test]
    fn test_duplicate_key() {
        let mut node = CharBucket::new();

        let child = SwizzledPtr::on_disk(42, 0, NodeType::Node4);
        assert!(node.add_child(42, child.clone()).is_ok());
        assert_eq!(node.add_child(42, child), Err(AddChildError::KeyExists));
    }

    #[test]
    fn test_remove_child() {
        let mut node = CharBucket::new();

        for i in 0..60 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        // Remove some elements
        for i in (0..60).step_by(2) {
            let removed = node.remove_child(i);
            assert!(removed.is_some());
        }

        assert_eq!(node.header.num_children, 30);

        // Odd keys should still be present
        for i in (1..60).step_by(2) {
            assert!(node.find_child(i).is_some());
        }

        // Even keys should be gone
        for i in (0..60).step_by(2) {
            assert!(node.find_child(i).is_none());
        }
    }

    #[test]
    fn test_shrink_to_node48() {
        let mut node = CharBucket::new();

        for i in 0..48 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        node.header.set_final(true);
        let node48 = node.shrink();

        assert_eq!(node48.header.node_type, 48);
        assert_eq!(node48.header.num_children, 48);
        assert!(node48.header.is_final());

        // Verify children transferred and sorted
        let keys: Vec<_> = node48.iter_children().map(|(k, _)| k).collect();
        for i in 0..48u32 {
            assert_eq!(keys[i as usize], i);
        }
    }

    #[test]
    fn test_unicode_keys() {
        let mut node = CharBucket::new();

        // Add many Unicode code points
        let chars: Vec<u32> = "αβγδεζηθικλμνξοπρστυφχψω日本語中文한글🎉🎊🎋🎌🎍🎎🎏🎐🎑🎒🎓"
            .chars()
            .map(|c| c as u32)
            .collect();

        for &key in &chars {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        // All should be findable
        for &key in &chars {
            assert!(
                node.find_child(key).is_some(),
                "should find key {}",
                char::from_u32(key).unwrap_or('?')
            );
        }
    }

    #[test]
    fn test_iter_children() {
        let mut node = CharBucket::new();

        for i in 0..20 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        let keys: std::collections::HashSet<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 20);
        for i in 0..20 {
            assert!(keys.contains(&i));
        }
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut node = CharBucket::new();

        assert!(node.is_empty());
        assert_eq!(node.len(), 0);

        for i in 0..10 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        assert!(!node.is_empty());
        assert_eq!(node.len(), 10);
    }
}
