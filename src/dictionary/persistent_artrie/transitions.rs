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
#[derive(Debug)]
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
        }
    }

    /// Check if this is a bucket
    pub fn is_bucket(&self) -> bool {
        matches!(self, ChildNode::Bucket(_))
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
}
