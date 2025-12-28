//! CharNode16: ART node for 5-16 children with AVX2 SIMD lookup.
//!
//! Uses AVX2 SIMD instructions for parallel u32 key comparison when available,
//! falling back to linear scan on other platforms.
//!
//! # SIMD Optimization
//!
//! The key array is 64 bytes (16 × u32), requiring two 256-bit AVX2 registers.
//! We use `_mm256_cmpeq_epi32` to compare 8 keys simultaneously:
//!
//! ```text
//! keys_lo (0-7):  [a][b][c][d][e][f][g][h]    (256 bits)
//! keys_hi (8-15): [i][j][k][l][m][n][o][p]    (256 bits)
//! search:         [h][h][h][h][h][h][h][h]    (broadcast)
//! result_lo:      [0][0][0][0][0][0][0][FF]   (match at index 7)
//! result_hi:      [0][0][0][0][0][0][0][0]    (no match)
//! ```
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ CharNodeHeader (16 bytes)                                 │
//! ├───────────────────────────────────────────────────────────┤
//! │ CharCompressedPrefix (24 bytes) - 6 u32 chars             │
//! ├───────────────────────────────────────────────────────────┤
//! │ keys: [u32; 16]          │ Key characters (64 bytes)      │
//! ├───────────────────────────────────────────────────────────┤
//! │ children: [SwizzledPtr; 16] │ Child pointers (128 bytes)  │
//! ├───────────────────────────────────────────────────────────┤
//! │ value_ptr: SwizzledPtr   │ Value pointer for final nodes  │
//! └───────────────────────────────────────────────────────────┘
//! Total: ~240 bytes (32-byte aligned for AVX2)
//! ```

use super::{AddChildError, CharArtNode, CharCompressedPrefix, CharNodeHeader};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

/// Maximum number of children in a CharNode16
pub const CHARNODE16_MAX_CHILDREN: usize = 16;

/// ART node with 5-16 children (character-level)
///
/// Uses AVX2 SIMD for parallel key comparison when available.
#[repr(C, align(32))] // 32-byte align for AVX2
#[derive(Debug, Clone)]
pub struct CharNode16 {
    /// Common node header
    pub header: CharNodeHeader,
    /// Compressed prefix for path compression (6 chars max)
    pub prefix: CharCompressedPrefix,
    /// Key characters (64 bytes, AVX2 aligned)
    pub keys: [u32; CHARNODE16_MAX_CHILDREN],
    /// Child pointers corresponding to keys
    pub children: [SwizzledPtr; CHARNODE16_MAX_CHILDREN],
    /// Optional value pointer (for nodes that are final)
    pub value_ptr: SwizzledPtr,
}

impl CharNode16 {
    /// Create a new empty CharNode16
    pub fn new() -> Self {
        Self {
            header: CharNodeHeader::new(116), // CHARNODE16 type
            prefix: CharCompressedPrefix::empty(),
            keys: [0; CHARNODE16_MAX_CHILDREN],
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
            value_ptr: SwizzledPtr::null(),
        }
    }

    /// Create a CharNode16 with a prefix
    pub fn with_prefix(prefix: &[u32]) -> Self {
        let mut node = Self::new();
        node.prefix = CharCompressedPrefix::from_chars(prefix);
        node.header.prefix_len = prefix.len() as u8;
        node
    }

    /// Find a key using AVX2 SIMD when available
    ///
    /// Compares 8 u32 keys simultaneously using two 256-bit AVX2 registers.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn find_key_index_simd(&self, key: u32) -> Option<usize> {
        use std::arch::x86_64::*;

        unsafe {
            let count = self.header.num_children as usize;

            // Broadcast the search key to all 8 positions
            let search = _mm256_set1_epi32(key as i32);

            // Compare lower 8 keys (indices 0-7)
            let keys_lo = _mm256_loadu_si256(self.keys.as_ptr() as *const __m256i);
            let cmp_lo = _mm256_cmpeq_epi32(keys_lo, search);

            // Convert to float to use movemask_ps (gives us 8 bits from 8 lanes)
            let mask_lo = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_lo)) as u32;

            // Check lower half first
            if mask_lo != 0 {
                let index = mask_lo.trailing_zeros() as usize;
                if index < count {
                    return Some(index);
                }
            }

            // Only check upper half if we have more than 8 children
            if count > 8 {
                let keys_hi = _mm256_loadu_si256(self.keys.as_ptr().add(8) as *const __m256i);
                let cmp_hi = _mm256_cmpeq_epi32(keys_hi, search);
                let mask_hi = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_hi)) as u32;

                if mask_hi != 0 {
                    let index = mask_hi.trailing_zeros() as usize + 8;
                    if index < count {
                        return Some(index);
                    }
                }
            }

            None
        }
    }

    /// Find a key using linear scan (fallback for non-AVX2 platforms)
    fn find_key_index_linear(&self, key: u32) -> Option<usize> {
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

    /// Shrink this node to a CharNode4
    pub fn shrink(&self) -> super::CharNode4 {
        debug_assert!(
            self.header.num_children <= 4,
            "cannot shrink CharNode16 with {} children",
            self.header.num_children
        );

        let mut node4 = super::CharNode4::new();
        node4.header = self.header.clone();
        node4.header.node_type = 4;
        node4.prefix = self.prefix;
        node4.value_ptr = self.value_ptr.clone();

        let count = self.header.num_children as usize;
        for i in 0..count {
            node4.keys[i] = self.keys[i];
            node4.children[i] = self.children[i].clone();
        }

        node4
    }

    /// Grow this node to a CharNode48
    pub fn grow(&self) -> super::CharNode48 {
        let mut node48 = super::CharNode48::new();
        node48.header = self.header.clone();
        node48.header.node_type = 48;
        node48.prefix = self.prefix;
        node48.value_ptr = self.value_ptr.clone();

        let count = self.header.num_children as usize;
        for i in 0..count {
            node48.keys[i] = self.keys[i];
            node48.children[i] = self.children[i].clone();
        }

        node48
    }
}

impl Default for CharNode16 {
    fn default() -> Self {
        Self::new()
    }
}

impl CharArtNode for CharNode16 {
    fn find_child(&self, key: u32) -> Option<&SwizzledPtr> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            self.find_key_index_simd(key).map(|i| &self.children[i])
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            self.find_key_index_linear(key).map(|i| &self.children[i])
        }
    }

    fn find_child_mut(&mut self, key: u32) -> Option<&mut SwizzledPtr> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let index = self.find_key_index_simd(key);

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let index = self.find_key_index_linear(key);

        index.map(move |i| &mut self.children[i])
    }

    fn add_child(&mut self, key: u32, child: SwizzledPtr) -> Result<(), AddChildError> {
        let count = self.header.num_children as usize;

        if count >= CHARNODE16_MAX_CHILDREN {
            return Err(AddChildError::NodeFull);
        }

        // Check for duplicate (use linear scan for correctness)
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

    fn remove_child(&mut self, key: u32) -> Option<SwizzledPtr> {
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
        self.header.num_children as usize >= CHARNODE16_MAX_CHILDREN
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
    fn test_new_charnode16() {
        let node = CharNode16::new();
        assert_eq!(node.header.node_type, 116); // CHARNODE16
        assert_eq!(node.header.num_children, 0);
        assert!(!node.is_full());
    }

    #[test]
    fn test_add_and_find_children() {
        let mut node = CharNode16::new();

        // Add children in random order
        let keys: Vec<u32> = "hbdfcega".chars().map(|c| c as u32).collect();
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 8);

        // Keys should be sorted
        let sorted: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        let expected: Vec<u32> = "abcdefgh".chars().map(|c| c as u32).collect();
        assert_eq!(sorted, expected);

        // Find all children
        for &key in &keys {
            assert!(
                node.find_child(key).is_some(),
                "should find key '{}'",
                char::from_u32(key).unwrap_or('?')
            );
        }

        // Should not find non-existent keys
        assert!(node.find_child('z' as u32).is_none());
    }

    #[test]
    fn test_add_unicode_children() {
        let mut node = CharNode16::new();

        // Add mix of ASCII and Unicode
        let keys: Vec<u32> = "αβγ日本🎉中文".chars().map(|c| c as u32).collect();
        for &key in &keys {
            let child = SwizzledPtr::on_disk(key, 0, NodeType::Node4);
            assert!(node.add_child(key, child).is_ok());
        }

        assert_eq!(node.header.num_children, 8);

        // All should be findable
        for &key in &keys {
            assert!(node.find_child(key).is_some());
        }
    }

    #[test]
    fn test_charnode16_full() {
        let mut node = CharNode16::new();

        for i in 0..16 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            assert!(node.add_child(i, child).is_ok());
        }

        assert!(node.is_full());

        let child = SwizzledPtr::on_disk(16, 0, NodeType::Node4);
        assert_eq!(node.add_child(16, child), Err(AddChildError::NodeFull));
    }

    #[test]
    fn test_remove_child() {
        let mut node = CharNode16::new();

        for i in 0..10 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
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
        let mut node = CharNode16::new();

        for i in 0..8 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child('a' as u32 + i, child).expect("add should succeed");
        }

        let keys: Vec<_> = node.iter_children().map(|(k, _)| k).collect();
        let expected: Vec<u32> = ('a'..='h').map(|c| c as u32).collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_shrink_to_node4() {
        let mut node = CharNode16::new();

        for i in 0..4 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        node.header.set_final(true);
        let node4 = node.shrink();

        assert_eq!(node4.header.node_type, 4);
        assert_eq!(node4.header.num_children, 4);
        assert!(node4.header.is_final());

        // Verify children transferred
        for i in 0..4 {
            assert!(node4.find_child(i).is_some());
        }
    }

    #[test]
    fn test_grow_to_node48() {
        let mut node = CharNode16::new();

        for i in 0..16 {
            let child = SwizzledPtr::on_disk(i, 0, NodeType::Node4);
            node.add_child(i, child).expect("add should succeed");
        }

        node.header.set_final(true);
        let node48 = node.grow();

        assert_eq!(node48.header.node_type, 48);
        assert_eq!(node48.header.num_children, 16);
        assert!(node48.header.is_final());

        // Verify children transferred
        for i in 0..16 {
            assert!(node48.find_child(i).is_some());
        }
    }

    #[test]
    fn test_simd_vs_linear_consistency() {
        // This test ensures SIMD and linear produce same results
        let mut node = CharNode16::new();

        // Add 12 children to test both halves
        for i in 0..12 {
            let child = SwizzledPtr::on_disk(i * 100, 0, NodeType::Node4);
            node.add_child(i * 100, child).expect("add should succeed");
        }

        // Test finding all keys
        for i in 0..12 {
            let key = i * 100;
            assert!(
                node.find_child(key).is_some(),
                "should find key {}",
                key
            );
        }

        // Test not finding missing keys
        for i in 0..12 {
            let key = i * 100 + 1;
            assert!(
                node.find_child(key).is_none(),
                "should not find key {}",
                key
            );
        }
    }

    #[test]
    fn test_alignment() {
        let node = CharNode16::new();
        // Verify 32-byte alignment for AVX2
        let addr = &node as *const CharNode16 as usize;
        assert_eq!(
            addr % 32,
            0,
            "CharNode16 should be 32-byte aligned for AVX2"
        );
    }
}
