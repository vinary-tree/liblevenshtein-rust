//! Node16: ART node for 5-16 children with SIMD lookup.
//!
//! Uses SSE4.1 SIMD instructions for parallel key comparison when available,
//! falling back to linear scan on other platforms.
//!
//! # SIMD Optimization
//!
//! The key array is 16 bytes, which fits perfectly in an SSE register.
//! We use `_mm_cmpeq_epi8` to compare all 16 keys simultaneously:
//!
//! ```text
//! keys:   [a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p]
//! search: [h][h][h][h][h][h][h][h][h][h][h][h][h][h][h][h]
//! result: [0][0][0][0][0][0][0][FF][0][0][0][0][0][0][0][0]
//!                              ^^ match found at index 7
//! ```
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ NodeHeader (16 bytes)                                     │
//! ├───────────────────────────────────────────────────────────┤
//! │ CompressedPrefix (12 bytes)                               │
//! ├───────────────────────────────────────────────────────────┤
//! │ keys: [u8; 16]    │ Key bytes (SIMD aligned)             │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 16] │ Child pointers (128 bytes) │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~168 bytes
//! ```

use super::{AddChildError, ArtNode, CompressedPrefix, NodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a Node16
pub const NODE16_MAX_CHILDREN: usize = 16;

/// ART node with 5-16 children
///
/// Uses SIMD (SSE4.1) for parallel key comparison when available.
#[repr(C, align(16))] // Align for SIMD
#[derive(Debug, Clone)]
pub struct Node16 {
    /// Common node header
    pub header: NodeHeader,
    /// Compressed prefix for path compression
    pub prefix: CompressedPrefix,
    /// Key bytes (16-byte aligned for SIMD)
    pub keys: [u8; NODE16_MAX_CHILDREN],
    /// Child pointers corresponding to keys
    pub children: [SwizzledPtr; NODE16_MAX_CHILDREN],
}

impl Node16 {
    /// Create a new empty Node16
    pub fn new() -> Self {
        Self {
            header: NodeHeader::new(16),
            prefix: CompressedPrefix::empty(),
            keys: [0; NODE16_MAX_CHILDREN],
            children: [
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
                SwizzledPtr::null(),
            ],
        }
    }

    /// Create a Node16 with a prefix
    pub fn with_prefix(prefix: &[u8]) -> Self {
        let mut node = Self::new();
        node.prefix = CompressedPrefix::from_bytes(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Find a key using SIMD when available
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    fn find_key_index_simd(&self, key: u8) -> Option<usize> {
        use std::arch::x86_64::*;

        unsafe {
            // Load all 16 keys into an SSE register
            let keys = _mm_loadu_si128(self.keys.as_ptr() as *const __m128i);

            // Broadcast the search key to all 16 positions
            let search = _mm_set1_epi8(key as i8);

            // Compare all keys simultaneously
            let cmp = _mm_cmpeq_epi8(keys, search);

            // Convert comparison result to a bitmask
            let mask = _mm_movemask_epi8(cmp) as u32;

            if mask != 0 {
                let index = mask.trailing_zeros() as usize;
                if index < self.header.num_children as usize {
                    return Some(index);
                }
            }
            None
        }
    }

    /// Find a key using linear scan (fallback)
    fn find_key_index_linear(&self, key: u8) -> Option<usize> {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] == key {
                return Some(i);
            }
        }
        None
    }

    /// Find the insertion point for a key (maintains sorted order)
    fn find_insert_point(&self, key: u8) -> usize {
        let count = self.header.num_children as usize;
        for i in 0..count {
            if self.keys[i] >= key {
                return i;
            }
        }
        count
    }
}

impl Default for Node16 {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtNode for Node16 {
    fn find_child(&self, key: u8) -> Option<&SwizzledPtr> {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
        {
            self.find_key_index_simd(key).map(|i| &self.children[i])
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1")))]
        {
            self.find_key_index_linear(key).map(|i| &self.children[i])
        }
    }

    fn find_child_mut(&mut self, key: u8) -> Option<&mut SwizzledPtr> {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
        let index = self.find_key_index_simd(key);

        #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1")))]
        let index = self.find_key_index_linear(key);

        index.map(move |i| &mut self.children[i])
    }

    fn add_child(&mut self, key: u8, child: SwizzledPtr) -> Result<(), AddChildError> {
        let count = self.header.num_children as usize;

        if count >= NODE16_MAX_CHILDREN {
            return Err(AddChildError::NodeFull);
        }

        // Check for duplicate
        if self.find_key_index_linear(key).is_some() {
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

    fn remove_child(&mut self, key: u8) -> Option<SwizzledPtr> {
        let count = self.header.num_children as usize;

        if let Some(index) = self.find_key_index_linear(key) {
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
        self.header.num_children as usize >= NODE16_MAX_CHILDREN
    }

    fn iter_children(&self) -> impl Iterator<Item = (u8, &SwizzledPtr)> {
        let count = self.header.num_children as usize;
        self.keys[..count]
            .iter()
            .zip(self.children[..count].iter())
            .map(|(&k, c)| (k, c))
    }
}

impl Node16 {
    /// Shrink this node to a Node4
    pub fn shrink(&self) -> super::Node4 {
        debug_assert!(
            self.header.num_children <= 4,
            "cannot shrink Node16 with {} children",
            self.header.num_children
        );

        let mut node4 = super::Node4::new();
        node4.header = self.header.clone();
        node4.header.node_type = 4;
        node4.prefix = self.prefix;

        let count = self.header.num_children as usize;
        for i in 0..count {
            node4.keys[i] = self.keys[i];
            node4.children[i] = self.children[i].clone();
        }

        node4
    }

    /// Grow this node to a Node48
    pub fn grow(&self) -> super::Node48 {
        let mut node48 = super::Node48::new();
        node48.header = self.header.clone();
        node48.header.node_type = 48;
        node48.prefix = self.prefix;

        let count = self.header.num_children as usize;
        for i in 0..count {
            let key = self.keys[i];
            node48.index[key as usize] = i as u8;
            node48.children[i] = self.children[i].clone();
        }

        node48
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node16() {
        let node = Node16::new();
        assert_eq!(node.header.node_type, 16);
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = Node16::new();

        // Add children in random order
        for &key in &[b'h', b'a', b'd', b'f', b'c', b'e', b'g', b'b'] {
            let child = SwizzledPtr::on_disk(
                key as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 8);

        // Keys should be sorted
        assert_eq!(&node.keys[..8], b"abcdefgh");

        // Find all children
        for key in b'a'..=b'h' {
            assert!(
                node.find_child(key).is_some(),
                "should find key '{}'",
                key as char
            );
        }

        // Should not find non-existent keys
        assert!(node.find_child(b'z').is_none());
    }

    #[test]
    fn test_node16_full() {
        let mut node = Node16::new();

        for i in 0..16 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            assert!(node.add_child(i as u8, child).is_ok());
        }

        assert!(node.is_full());

        let child = SwizzledPtr::on_disk(16, 0, crate::dictionary::persistent_artrie::NodeType::Node4);
        assert_eq!(node.add_child(16, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node16::new();

        for i in 0..10 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i as u8, child).expect("add should succeed");
        }

        // Remove middle element
        let removed = node.remove_child(5);
        assert!(removed.is_some());
        assert_eq!(node.header.num_children, 9);
        assert!(node.find_child(5).is_none());

        // Other children should still be present
        for i in 0..10 {
            if i != 5 {
                assert!(node.find_child(i).is_some());
            }
        }
    }

    #[test]
    fn test_iter_children() {
        let mut node = Node16::new();

        for i in 0..8 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(b'a' + i, child).expect("add should succeed");
        }

        let keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        assert_eq!(keys, (b'a'..=b'h').collect::<Vec<_>>());
    }

    #[test]
    fn test_shrink_to_node4() {
        let mut node = Node16::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(
                i as u32,
                0,
                crate::dictionary::persistent_artrie::NodeType::Node4,
            );
            node.add_child(i as u8, child).expect("add should succeed");
        }

        let node4 = node.shrink();
        assert_eq!(node4.header.node_type, 4);
        assert_eq!(node4.header.num_children, 4);

        // Verify children transferred
        for i in 0..4 {
            assert!(node4.find_child(i as u8).is_some());
        }
    }
}
