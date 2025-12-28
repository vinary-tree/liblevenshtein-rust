//! Bucket ↔ ART Node Transitions
//!
//! This module handles the transitions between bucket leaf nodes and ART internal nodes.
//! These transitions occur when:
//!
//! 1. **Bucket → ART**: A bucket becomes full and needs to be converted to an ART node
//!    with child buckets (one per first-byte of entries)
//!
//! 2. **ART → Bucket**: An ART node's children all become small enough to be merged
//!    back into a single bucket
//!
//! # Architecture
//!
//! ```text
//! Before (single bucket):
//! ┌─────────────────────────────────────────┐
//! │ Bucket: ["apple", "apricot", "banana",  │
//! │          "berry", "cherry"]             │
//! └─────────────────────────────────────────┘
//!
//! After (ART node with child buckets):
//! ┌─────────────┐
//! │  Node4      │
//! │ a→ b→ c→    │
//! └──┬──┬──┬────┘
//!    │  │  │
//!    │  │  └─► Bucket: ["herry"]
//!    │  │
//!    │  └────► Bucket: ["anana", "erry"]
//!    │
//!    └───────► Bucket: ["pple", "pricot"]
//! ```

use super::bucket::{BucketError, StringBucket};
use super::nodes::{ArtNode, Node, Node4};
use super::swizzled_ptr::SwizzledPtr;

/// Threshold for converting a bucket to an ART node
/// (when bucket has this many unique first-bytes)
pub const BUCKET_TO_ART_THRESHOLD: usize = 4;

/// Threshold for merging ART children back to a bucket
/// (when total entries across all children is below this)
pub const ART_TO_BUCKET_THRESHOLD: usize = 32;

/// Result of a bucket-to-ART transition
#[derive(Debug)]
pub struct BucketToArtResult {
    /// The new ART node
    pub node: Node,
    /// Child buckets keyed by their edge byte
    pub children: Vec<(u8, StringBucket)>,
    /// Whether this node is final (had an empty suffix in the bucket)
    pub is_final: bool,
    /// Value associated with the final state (if any)
    pub final_value: Option<Vec<u8>>,
}

/// Result of an ART-to-bucket transition
#[derive(Debug)]
pub struct ArtToBucketResult {
    /// The merged bucket
    pub bucket: StringBucket,
}

/// Error during transition operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    /// Bucket doesn't meet criteria for conversion
    BucketNotReady(String),
    /// ART node doesn't meet criteria for merging
    ArtNotReady(String),
    /// Bucket operation failed
    BucketError(BucketError),
    /// Resulting bucket would be too large
    MergedBucketTooLarge,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionError::BucketNotReady(msg) => {
                write!(f, "bucket not ready for conversion: {}", msg)
            }
            TransitionError::ArtNotReady(msg) => {
                write!(f, "ART node not ready for merging: {}", msg)
            }
            TransitionError::BucketError(e) => write!(f, "bucket error: {}", e),
            TransitionError::MergedBucketTooLarge => {
                write!(f, "merged bucket would exceed size limit")
            }
        }
    }
}

impl std::error::Error for TransitionError {}

impl From<BucketError> for TransitionError {
    fn from(e: BucketError) -> Self {
        TransitionError::BucketError(e)
    }
}

/// Check if a bucket should be converted to an ART node
///
/// A bucket should be converted when:
/// 1. It's full or near the split threshold
/// 2. It has entries with multiple distinct first-bytes
pub fn should_convert_bucket_to_art(bucket: &StringBucket) -> bool {
    if !bucket.header().should_split() {
        return false;
    }

    // Count distinct first bytes
    let result = bucket.split_by_first_byte();
    result.buckets.len() >= BUCKET_TO_ART_THRESHOLD
}

/// Convert a bucket to an ART node with child buckets
///
/// This splits the bucket by first byte and creates an ART node (Node4 initially)
/// with edges to child buckets.
pub fn bucket_to_art_node(bucket: &StringBucket) -> Result<BucketToArtResult, TransitionError> {
    let split_result = bucket.split_by_first_byte();

    if split_result.buckets.is_empty() && split_result.finals.is_empty() {
        return Err(TransitionError::BucketNotReady("bucket is empty".to_string()));
    }

    // Create a new Node4 (will grow as needed when children are added)
    let mut node = Node4::new();
    let has_values = bucket.header().has_values();

    // Determine if this node is final
    let is_final = !split_result.finals.is_empty();
    let final_value = if is_final {
        split_result.finals.first().and_then(|(_, v)| v.clone())
    } else {
        None
    };

    node.header.set_final(is_final);

    // Collect children
    let mut children: Vec<(u8, StringBucket)> = Vec::new();

    // Collect children first
    for (byte, child_bucket) in split_result.buckets {
        children.push((byte, child_bucket));
    }

    // Now build the appropriate node type based on child count
    let node = if children.len() <= 4 {
        // Node4 can hold all children
        for (byte, _) in &children {
            let ptr = SwizzledPtr::null();
            let _ = node.add_child(*byte, ptr);
        }
        Node::N4(Box::new(node))
    } else if children.len() <= 16 {
        // Need Node16
        let mut node16 = node.grow();
        for (byte, _) in &children {
            let ptr = SwizzledPtr::null();
            let _ = node16.add_child(*byte, ptr);
        }
        Node::N16(Box::new(node16))
    } else if children.len() <= 48 {
        // Need Node48
        let node16 = node.grow();
        let mut node48 = node16.grow();
        for (byte, _) in &children {
            let ptr = SwizzledPtr::null();
            let _ = node48.add_child(*byte, ptr);
        }
        Node::N48(Box::new(node48))
    } else {
        // Need Node256
        let node16 = node.grow();
        let node48 = node16.grow();
        let mut node256 = node48.grow();
        for (byte, _) in &children {
            let ptr = SwizzledPtr::null();
            let _ = node256.add_child(*byte, ptr);
        }
        Node::N256(Box::new(node256))
    };

    Ok(BucketToArtResult {
        node,
        children,
        is_final,
        final_value,
    })
}

/// Check if an ART node's children should be merged back to a bucket
///
/// This checks if the total size of all child buckets is small enough
/// to fit in a single bucket.
pub fn should_merge_art_to_bucket(children: &[(u8, &StringBucket)]) -> bool {
    let total_entries: usize = children.iter().map(|(_, b)| b.len()).sum();
    total_entries <= ART_TO_BUCKET_THRESHOLD
}

/// Merge ART node children back into a single bucket
///
/// This collects all entries from child buckets and creates a single
/// bucket with the edge byte prepended to each suffix.
pub fn art_node_to_bucket(
    children: &[(u8, &StringBucket)],
    is_final: bool,
    final_value: Option<&[u8]>,
) -> Result<ArtToBucketResult, TransitionError> {
    let has_values = children.iter().any(|(_, b)| b.header().has_values())
        || final_value.is_some();

    let mut bucket = if has_values {
        StringBucket::with_values()
    } else {
        StringBucket::new()
    };

    // Add final entry if this node was final
    if is_final {
        if let Some(value) = final_value {
            bucket.insert(b"", value)?;
        } else {
            bucket.insert_key(b"")?;
        }
    }

    // Collect entries from all children
    for (edge_byte, child) in children {
        for i in 0..child.len() {
            let entry = child.get_entry(i).expect("valid index");
            let suffix = child.get_suffix(&entry);
            let value = child.get_value(&entry);

            // Prepend edge byte to suffix
            let mut full_suffix = vec![*edge_byte];
            full_suffix.extend_from_slice(suffix);

            if let Some(v) = value {
                bucket.insert(&full_suffix, v)?;
            } else {
                bucket.insert_key(&full_suffix)?;
            }
        }
    }

    Ok(ArtToBucketResult { bucket })
}

/// Represents a child pointer that can be either a bucket or an ART node
#[derive(Debug, Clone)]
pub enum ChildNode {
    /// A bucket leaf node
    Bucket(StringBucket),
    /// An ART internal node with its own children
    ArtNode {
        /// The node itself
        node: Node,
        /// Whether this node represents a final state
        is_final: bool,
        /// Value if this is a final state with a value
        value: Option<Vec<u8>>,
        /// Child nodes (for nested ART)
        children: Vec<(u8, ChildNode)>,
    },
    /// A disk-backed reference (not yet loaded)
    ///
    /// This variant is used for lazy loading. When accessed, the SwizzledPtr
    /// is resolved by loading the node from disk via the BufferManager.
    DiskRef {
        /// The swizzled pointer containing disk location
        ptr: SwizzledPtr,
    },
}

impl ChildNode {
    /// Create a new bucket child
    pub fn bucket(b: StringBucket) -> Self {
        ChildNode::Bucket(b)
    }

    /// Create a new ART node child
    pub fn art_node(node: Node, is_final: bool, value: Option<Vec<u8>>) -> Self {
        ChildNode::ArtNode {
            node,
            is_final,
            value,
            children: Vec::new(),
        }
    }

    /// Create a new ART node child with children
    pub fn art_node_with_children(
        node: Node,
        is_final: bool,
        value: Option<Vec<u8>>,
        children: Vec<(u8, ChildNode)>,
    ) -> Self {
        ChildNode::ArtNode {
            node,
            is_final,
            value,
            children,
        }
    }

    /// Create a new disk reference child
    pub fn disk_ref(ptr: SwizzledPtr) -> Self {
        ChildNode::DiskRef { ptr }
    }

    /// Check if this is a bucket
    pub fn is_bucket(&self) -> bool {
        matches!(self, ChildNode::Bucket(_))
    }

    /// Check if this is a disk reference (not yet loaded)
    pub fn is_disk_ref(&self) -> bool {
        matches!(self, ChildNode::DiskRef { .. })
    }

    /// Get the SwizzledPtr if this is a disk reference
    pub fn as_disk_ref(&self) -> Option<&SwizzledPtr> {
        match self {
            ChildNode::DiskRef { ptr } => Some(ptr),
            _ => None,
        }
    }

    /// Get as bucket reference
    pub fn as_bucket(&self) -> Option<&StringBucket> {
        match self {
            ChildNode::Bucket(b) => Some(b),
            _ => None,
        }
    }

    /// Get as mutable bucket reference
    pub fn as_bucket_mut(&mut self) -> Option<&mut StringBucket> {
        match self {
            ChildNode::Bucket(b) => Some(b),
            _ => None,
        }
    }

    /// Get as ART node reference
    pub fn as_art_node(&self) -> Option<(&Node, bool, &Option<Vec<u8>>, &Vec<(u8, ChildNode)>)> {
        match self {
            ChildNode::ArtNode {
                node,
                is_final,
                value,
                children,
            } => Some((node, *is_final, value, children)),
            _ => None,
        }
    }

    /// Get as mutable ART node reference
    pub fn as_art_node_mut(
        &mut self,
    ) -> Option<(&mut Node, &mut bool, &mut Option<Vec<u8>>, &mut Vec<(u8, ChildNode)>)> {
        match self {
            ChildNode::ArtNode {
                node,
                is_final,
                value,
                children,
            } => Some((node, is_final, value, children)),
            _ => None,
        }
    }

    /// Recursively insert a key into this child node.
    ///
    /// This handles all child types:
    /// - `Bucket`: directly insert into the bucket
    /// - `ArtNode`: recursively descend through nested ART structure
    /// - `DiskRef`: not supported for mutation (returns false)
    ///
    /// Returns `true` if the key was newly inserted, `false` if it already existed
    /// or if insertion failed.
    pub fn insert_key(&mut self, remaining: &[u8]) -> bool {
        match self {
            ChildNode::Bucket(bucket) => {
                bucket.insert_key(remaining).unwrap_or(false)
            }
            ChildNode::ArtNode {
                is_final,
                value: _,
                children,
                ..
            } => {
                if remaining.is_empty() {
                    // Insert at this node (make it final)
                    if *is_final {
                        false // Already exists
                    } else {
                        *is_final = true;
                        true
                    }
                } else {
                    let first = remaining[0];
                    let rest = &remaining[1..];

                    // Find child with matching byte
                    for (b, child) in children.iter_mut() {
                        if *b == first {
                            return child.insert_key(rest);
                        }
                    }

                    // No matching child, create new bucket
                    let mut new_bucket = StringBucket::with_values();
                    let _ = new_bucket.insert_key(rest);
                    children.push((first, ChildNode::Bucket(new_bucket)));
                    true
                }
            }
            ChildNode::DiskRef { .. } => {
                // Cannot insert into disk ref without loading first
                false
            }
        }
    }

    /// Recursively remove a key from this child node.
    ///
    /// This handles all child types:
    /// - `Bucket`: directly remove from the bucket
    /// - `ArtNode`: recursively descend through nested ART structure
    /// - `DiskRef`: not supported for mutation (returns false)
    ///
    /// Returns `true` if the key was removed, `false` if it didn't exist
    /// or if removal failed.
    pub fn remove_key(&mut self, remaining: &[u8]) -> bool {
        match self {
            ChildNode::Bucket(bucket) => {
                bucket.remove(remaining).is_some()
            }
            ChildNode::ArtNode {
                is_final,
                value,
                children,
                ..
            } => {
                if remaining.is_empty() {
                    // Remove at this node
                    if *is_final {
                        *is_final = false;
                        *value = None;
                        true
                    } else {
                        false // Didn't exist
                    }
                } else {
                    let first = remaining[0];
                    let rest = &remaining[1..];

                    // Find child with matching byte
                    for (b, child) in children.iter_mut() {
                        if *b == first {
                            return child.remove_key(rest);
                        }
                    }
                    false // Child not found
                }
            }
            ChildNode::DiskRef { .. } => {
                // Cannot remove from disk ref without loading first
                false
            }
        }
    }

    /// Check if this child node contains a key.
    ///
    /// This handles all child types:
    /// - `Bucket`: directly check in the bucket
    /// - `ArtNode`: recursively descend through nested ART structure
    /// - `DiskRef`: not supported (returns false)
    pub fn contains_key(&self, remaining: &[u8]) -> bool {
        match self {
            ChildNode::Bucket(bucket) => {
                bucket.contains(remaining)
            }
            ChildNode::ArtNode {
                is_final,
                children,
                ..
            } => {
                if remaining.is_empty() {
                    *is_final
                } else {
                    let first = remaining[0];
                    let rest = &remaining[1..];

                    // Find child with matching byte
                    for (b, child) in children.iter() {
                        if *b == first {
                            return child.contains_key(rest);
                        }
                    }
                    false // Child not found
                }
            }
            ChildNode::DiskRef { .. } => {
                // Cannot check disk ref without loading
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_convert_empty_bucket() {
        let bucket = StringBucket::new();
        assert!(!should_convert_bucket_to_art(&bucket));
    }

    #[test]
    fn test_should_convert_small_bucket() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();
        assert!(!should_convert_bucket_to_art(&bucket));
    }

    #[test]
    fn test_bucket_to_art_basic() {
        let mut bucket = StringBucket::new();

        // Insert entries with different first bytes
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();
        bucket.insert_key(b"cherry").unwrap();

        let result = bucket_to_art_node(&bucket).unwrap();

        // Should have 3 children
        assert_eq!(result.children.len(), 3);

        // Should not be final
        assert!(!result.is_final);

        // Check children
        let a_child = result.children.iter().find(|(b, _)| *b == b'a');
        assert!(a_child.is_some());
        let (_, a_bucket) = a_child.unwrap();
        assert!(a_bucket.contains(b"pple"));

        let b_child = result.children.iter().find(|(b, _)| *b == b'b');
        assert!(b_child.is_some());
        let (_, b_bucket) = b_child.unwrap();
        assert!(b_bucket.contains(b"anana"));
    }

    #[test]
    fn test_bucket_to_art_with_final() {
        let mut bucket = StringBucket::new();

        bucket.insert_key(b"").unwrap(); // Final marker
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();

        let result = bucket_to_art_node(&bucket).unwrap();

        // Should be final
        assert!(result.is_final);
        assert!(result.final_value.is_none());
    }

    #[test]
    fn test_bucket_to_art_with_value() {
        let mut bucket = StringBucket::with_values();

        bucket.insert(b"", b"root_value").unwrap();
        bucket.insert(b"apple", b"apple_value").unwrap();

        let result = bucket_to_art_node(&bucket).unwrap();

        assert!(result.is_final);
        assert_eq!(result.final_value, Some(b"root_value".to_vec()));
    }

    #[test]
    fn test_art_to_bucket_basic() {
        // Create child buckets
        let mut a_bucket = StringBucket::new();
        a_bucket.insert_key(b"pple").unwrap();
        a_bucket.insert_key(b"pricot").unwrap();

        let mut b_bucket = StringBucket::new();
        b_bucket.insert_key(b"anana").unwrap();

        let children: Vec<(u8, &StringBucket)> = vec![(b'a', &a_bucket), (b'b', &b_bucket)];

        let result = art_node_to_bucket(&children, false, None).unwrap();

        // Should have all entries with edge bytes prepended
        assert_eq!(result.bucket.len(), 3);
        assert!(result.bucket.contains(b"apple"));
        assert!(result.bucket.contains(b"apricot"));
        assert!(result.bucket.contains(b"banana"));
    }

    #[test]
    fn test_art_to_bucket_with_final() {
        let mut a_bucket = StringBucket::new();
        a_bucket.insert_key(b"pple").unwrap();

        let children: Vec<(u8, &StringBucket)> = vec![(b'a', &a_bucket)];

        let result = art_node_to_bucket(&children, true, None).unwrap();

        // Should include the empty suffix for final state
        assert!(result.bucket.contains(b""));
        assert!(result.bucket.contains(b"apple"));
    }

    #[test]
    fn test_roundtrip_bucket_art_bucket() {
        let mut original = StringBucket::new();

        original.insert_key(b"apple").unwrap();
        original.insert_key(b"apricot").unwrap();
        original.insert_key(b"banana").unwrap();
        original.insert_key(b"berry").unwrap();
        original.insert_key(b"cherry").unwrap();

        // Collect original entries
        let original_entries: Vec<_> = original.iter().map(|(_, s)| s.to_vec()).collect();

        // Convert to ART
        let art_result = bucket_to_art_node(&original).unwrap();

        // Convert back to bucket
        let children: Vec<(u8, &StringBucket)> = art_result
            .children
            .iter()
            .map(|(b, bucket)| (*b, bucket))
            .collect();

        let bucket_result =
            art_node_to_bucket(&children, art_result.is_final, art_result.final_value.as_deref())
                .unwrap();

        // Should have same entries
        let restored_entries: Vec<_> = bucket_result.bucket.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(original_entries, restored_entries);
    }

    #[test]
    fn test_should_merge_art_to_bucket() {
        let mut small_bucket = StringBucket::new();
        small_bucket.insert_key(b"test").unwrap();

        let children: Vec<(u8, &StringBucket)> = vec![(b'a', &small_bucket)];
        assert!(should_merge_art_to_bucket(&children));

        // With many entries, should not merge
        let mut large_bucket = StringBucket::new();
        for i in 0..50 {
            let key = format!("{:03}", i);
            large_bucket.insert_key(key.as_bytes()).unwrap();
        }

        let children: Vec<(u8, &StringBucket)> = vec![(b'a', &large_bucket)];
        assert!(!should_merge_art_to_bucket(&children));
    }

    #[test]
    fn test_child_node_enum() {
        let bucket = StringBucket::new();
        let child = ChildNode::bucket(bucket);
        assert!(child.is_bucket());
        assert!(child.as_bucket().is_some());

        let node = Node::N4(Box::new(Node4::new()));
        let child = ChildNode::art_node(node, false, None);
        assert!(!child.is_bucket());
        assert!(child.as_bucket().is_none());
    }

    #[test]
    fn test_child_node_insert_key_bucket() {
        let bucket = StringBucket::new();
        let mut child = ChildNode::bucket(bucket);

        // Insert into bucket child
        assert!(child.insert_key(b"apple"));
        assert!(child.insert_key(b"banana"));

        // Duplicate returns false
        assert!(!child.insert_key(b"apple"));

        // Verify keys exist
        assert!(child.contains_key(b"apple"));
        assert!(child.contains_key(b"banana"));
        assert!(!child.contains_key(b"cherry"));
    }

    #[test]
    fn test_child_node_insert_key_art() {
        let node = Node::N4(Box::new(Node4::new()));
        let mut child = ChildNode::art_node_with_children(node, false, None, Vec::new());

        // Insert creates new bucket children
        assert!(child.insert_key(b"apple"));
        assert!(child.insert_key(b"apricot"));

        // Verify keys exist
        assert!(child.contains_key(b"apple"));
        assert!(child.contains_key(b"apricot"));
        assert!(!child.contains_key(b"banana"));

        // Insert at empty path (marks node as final)
        assert!(child.insert_key(b""));
        assert!(!child.insert_key(b"")); // Already final
    }

    #[test]
    fn test_child_node_remove_key_bucket() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();

        let mut child = ChildNode::bucket(bucket);

        // Remove existing key
        assert!(child.remove_key(b"apple"));
        assert!(!child.contains_key(b"apple"));

        // Remove non-existing key
        assert!(!child.remove_key(b"apple"));

        // Verify other key still exists
        assert!(child.contains_key(b"banana"));
    }

    #[test]
    fn test_child_node_remove_key_art() {
        let node = Node::N4(Box::new(Node4::new()));
        let mut child = ChildNode::art_node_with_children(node, true, None, Vec::new());

        // Insert some keys
        assert!(child.insert_key(b"apple"));

        // Remove from final state
        assert!(child.remove_key(b""));

        // Remove existing key
        assert!(child.remove_key(b"apple"));
        assert!(!child.contains_key(b"apple"));

        // Remove non-existing key
        assert!(!child.remove_key(b"apple"));
    }

    #[test]
    fn test_child_node_nested_art_operations() {
        // Create a nested ART structure: root ART -> child ART -> bucket
        let inner_node = Node::N4(Box::new(Node4::new()));
        let inner_bucket = StringBucket::new();
        let inner_child = ChildNode::art_node_with_children(
            inner_node,
            false,
            None,
            vec![(b'p', ChildNode::Bucket(inner_bucket))],
        );

        let outer_node = Node::N4(Box::new(Node4::new()));
        let mut outer_child = ChildNode::art_node_with_children(
            outer_node,
            false,
            None,
            vec![(b'a', inner_child)],
        );

        // Insert through nested structure
        assert!(outer_child.insert_key(b"apple")); // a -> p -> ple

        // Verify the key exists
        assert!(outer_child.contains_key(b"apple"));
        assert!(!outer_child.contains_key(b"apricot"));

        // Remove through nested structure
        assert!(outer_child.remove_key(b"apple"));
        assert!(!outer_child.contains_key(b"apple"));
    }

    #[test]
    fn test_child_node_disk_ref_operations() {
        let ptr = SwizzledPtr::null();
        let mut child = ChildNode::disk_ref(ptr);

        // All operations on disk ref return false (not supported without loading)
        assert!(!child.insert_key(b"test"));
        assert!(!child.remove_key(b"test"));
        assert!(!child.contains_key(b"test"));
    }
}
