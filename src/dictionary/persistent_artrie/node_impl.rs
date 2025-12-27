//! Dictionary Node Implementation for Persistent ART
//!
//! This module provides the `PersistentARTrieNode` type that implements the
//! `DictionaryNode` trait, enabling the Persistent Adaptive Radix Trie to work
//! with the Levenshtein automata transducer.
//!
//! # Architecture
//!
//! The node implementation handles two types of children:
//! 1. **ART Nodes**: Internal nodes with up to 256 children (Node4/16/48/256)
//! 2. **Buckets**: Leaf nodes containing multiple suffixes
//!
//! Traversal follows a hybrid approach:
//! - ART nodes use pointer-based child lookup
//! - Buckets use binary search for suffix lookup
//!
//! # Thread Safety
//!
//! All operations are thread-safe through immutable access patterns. The nodes
//! are designed to be cloned cheaply using `Arc` for shared ownership.

use std::sync::Arc;

use crate::dictionary::{DictionaryNode, value::DictionaryValue};
use super::bucket::StringBucket;
use super::nodes::{ArtNode, Node, Node4, Node16, Node48, Node256};

/// A node in the Persistent ART that can be either an ART internal node or a bucket leaf.
///
/// This type implements `DictionaryNode` for integration with the Levenshtein transducer.
#[derive(Clone, Debug)]
pub struct PersistentARTrieNode<V: DictionaryValue = ()> {
    /// The inner node representation
    inner: Arc<NodeInner<V>>,
    /// Current position within a bucket (for bucket iteration)
    bucket_position: Option<BucketPosition>,
}

/// Position within a bucket for iterating through its suffixes
#[derive(Clone, Debug)]
struct BucketPosition {
    /// Current suffix being traversed
    suffix: Vec<u8>,
    /// Index of current entry in the bucket
    entry_index: usize,
    /// Current position within the suffix
    suffix_offset: usize,
}

/// Inner representation of a node
#[derive(Debug)]
enum NodeInner<V: DictionaryValue> {
    /// An ART internal node
    ArtNode {
        /// The node variant (Node4/16/48/256)
        node: Node,
        /// Whether this is a final state
        is_final: bool,
        /// Value if this is a final state (for mapped dictionaries)
        value: Option<V>,
    },
    /// A bucket leaf node
    Bucket {
        /// The bucket containing multiple entries
        bucket: StringBucket,
    },
    /// The root node of the trie
    Root {
        /// Root node (always an ART node initially)
        node: Node,
        /// Whether root is final (empty string is in dictionary)
        is_final: bool,
        /// Value for empty string
        value: Option<V>,
    },
    /// An empty node (used for non-existent transitions)
    Empty,
}

impl<V: DictionaryValue> PersistentARTrieNode<V> {
    /// Create a new root node
    pub fn new_root() -> Self {
        Self {
            inner: Arc::new(NodeInner::Root {
                node: Node::N4(Box::new(Node4::new())),
                is_final: false,
                value: None,
            }),
            bucket_position: None,
        }
    }

    /// Create a new ART node
    pub fn new_art_node(node: Node, is_final: bool, value: Option<V>) -> Self {
        Self {
            inner: Arc::new(NodeInner::ArtNode {
                node,
                is_final,
                value,
            }),
            bucket_position: None,
        }
    }

    /// Create a new bucket node
    pub fn new_bucket(bucket: StringBucket) -> Self {
        Self {
            inner: Arc::new(NodeInner::Bucket { bucket }),
            bucket_position: None,
        }
    }

    /// Create a bucket node at a specific position
    fn new_bucket_at_position(bucket: Arc<NodeInner<V>>, position: BucketPosition) -> Self {
        Self {
            inner: bucket,
            bucket_position: Some(position),
        }
    }

    /// Create an empty node (for non-existent transitions)
    fn empty() -> Self {
        Self {
            inner: Arc::new(NodeInner::Empty),
            bucket_position: None,
        }
    }

    /// Check if this is an empty node
    pub fn is_empty(&self) -> bool {
        matches!(&*self.inner, NodeInner::Empty)
    }

    /// Get the associated value (for mapped dictionaries)
    pub fn value(&self) -> Option<&V> {
        match &*self.inner {
            NodeInner::ArtNode { value, .. } => value.as_ref(),
            NodeInner::Root { value, .. } => value.as_ref(),
            NodeInner::Bucket { bucket } => {
                if let Some(ref pos) = self.bucket_position {
                    if pos.suffix_offset == pos.suffix.len() {
                        // At the end of suffix, check for value
                        if let Some(entry) = bucket.get_entry(pos.entry_index) {
                            if entry.has_value() {
                                // We can't return a reference to the value here
                                // since it would require parsing the bytes
                                return None;
                            }
                        }
                    }
                }
                None
            }
            NodeInner::Empty => None,
        }
    }

    /// Get child edges from an ART node
    fn art_edges(&self, node: &Node) -> Vec<(u8, Self)> {
        let mut edges = Vec::new();

        match node {
            Node::N4(n) => {
                for (key, child_ptr) in n.iter_children() {
                    // For now, we create placeholder nodes
                    // In full implementation, we'd resolve the SwizzledPtr
                    let child = Self::new_art_node(
                        Node::N4(Box::new(Node4::new())),
                        false,
                        None,
                    );
                    edges.push((key, child));
                }
            }
            Node::N16(n) => {
                for (key, _child_ptr) in n.iter_children() {
                    let child = Self::new_art_node(
                        Node::N4(Box::new(Node4::new())),
                        false,
                        None,
                    );
                    edges.push((key, child));
                }
            }
            Node::N48(n) => {
                for (key, _child_ptr) in n.iter_children() {
                    let child = Self::new_art_node(
                        Node::N4(Box::new(Node4::new())),
                        false,
                        None,
                    );
                    edges.push((key, child));
                }
            }
            Node::N256(n) => {
                for (key, _child_ptr) in n.iter_children() {
                    let child = Self::new_art_node(
                        Node::N4(Box::new(Node4::new())),
                        false,
                        None,
                    );
                    edges.push((key, child));
                }
            }
        }

        edges
    }

    /// Get edges from a bucket (first bytes of all suffixes)
    fn bucket_edges(&self, bucket: &StringBucket) -> Vec<(u8, Self)> {
        let mut edges = Vec::new();
        let mut seen_bytes = [false; 256];

        for i in 0..bucket.len() {
            if let Some(entry) = bucket.get_entry(i) {
                let suffix = bucket.get_suffix(&entry);
                if !suffix.is_empty() {
                    let first_byte = suffix[0];
                    if !seen_bytes[first_byte as usize] {
                        seen_bytes[first_byte as usize] = true;

                        // Create a child node positioned at this suffix
                        let position = BucketPosition {
                            suffix: suffix.to_vec(),
                            entry_index: i,
                            suffix_offset: 1, // Skip the first byte (edge label)
                        };
                        let child = Self::new_bucket_at_position(self.inner.clone(), position);
                        edges.push((first_byte, child));
                    }
                }
            }
        }

        edges
    }
}

impl<V: DictionaryValue> DictionaryNode for PersistentARTrieNode<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        match &*self.inner {
            NodeInner::ArtNode { is_final, .. } => *is_final,
            NodeInner::Root { is_final, .. } => *is_final,
            NodeInner::Bucket { bucket } => {
                if let Some(ref pos) = self.bucket_position {
                    // Final if we've consumed the entire suffix
                    pos.suffix_offset == pos.suffix.len()
                } else {
                    // At bucket root, check for empty suffix
                    bucket.contains(b"")
                }
            }
            NodeInner::Empty => false,
        }
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        match &*self.inner {
            NodeInner::ArtNode { node, .. } | NodeInner::Root { node, .. } => {
                // Transition via ART node
                let child_ptr = match node {
                    Node::N4(n) => n.find_child(label),
                    Node::N16(n) => n.find_child(label),
                    Node::N48(n) => n.find_child(label),
                    Node::N256(n) => n.find_child(label),
                };

                if child_ptr.is_some() {
                    // In full implementation, resolve SwizzledPtr
                    // For now, return a placeholder
                    Some(Self::new_art_node(
                        Node::N4(Box::new(Node4::new())),
                        false,
                        None,
                    ))
                } else {
                    None
                }
            }
            NodeInner::Bucket { bucket } => {
                if let Some(ref pos) = self.bucket_position {
                    // Continue traversing the current suffix
                    if pos.suffix_offset < pos.suffix.len() {
                        if pos.suffix[pos.suffix_offset] == label {
                            let new_pos = BucketPosition {
                                suffix: pos.suffix.clone(),
                                entry_index: pos.entry_index,
                                suffix_offset: pos.suffix_offset + 1,
                            };
                            return Some(Self::new_bucket_at_position(self.inner.clone(), new_pos));
                        }
                    }
                    None
                } else {
                    // Search for matching suffix in bucket
                    for i in 0..bucket.len() {
                        if let Some(entry) = bucket.get_entry(i) {
                            let suffix = bucket.get_suffix(&entry);
                            if !suffix.is_empty() && suffix[0] == label {
                                let position = BucketPosition {
                                    suffix: suffix.to_vec(),
                                    entry_index: i,
                                    suffix_offset: 1,
                                };
                                return Some(Self::new_bucket_at_position(
                                    self.inner.clone(),
                                    position,
                                ));
                            }
                        }
                    }
                    None
                }
            }
            NodeInner::Empty => None,
        }
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        match &*self.inner {
            NodeInner::ArtNode { node, .. } | NodeInner::Root { node, .. } => {
                Box::new(self.art_edges(node).into_iter())
            }
            NodeInner::Bucket { bucket } => {
                if let Some(ref pos) = self.bucket_position {
                    // Within a suffix, there's at most one edge
                    if pos.suffix_offset < pos.suffix.len() {
                        let next_byte = pos.suffix[pos.suffix_offset];
                        let new_pos = BucketPosition {
                            suffix: pos.suffix.clone(),
                            entry_index: pos.entry_index,
                            suffix_offset: pos.suffix_offset + 1,
                        };
                        let child = Self::new_bucket_at_position(self.inner.clone(), new_pos);
                        Box::new(std::iter::once((next_byte, child)))
                    } else {
                        Box::new(std::iter::empty())
                    }
                } else {
                    Box::new(self.bucket_edges(bucket).into_iter())
                }
            }
            NodeInner::Empty => Box::new(std::iter::empty()),
        }
    }

    fn has_edge(&self, label: Self::Unit) -> bool {
        self.transition(label).is_some()
    }

    fn edge_count(&self) -> Option<usize> {
        match &*self.inner {
            NodeInner::ArtNode { node, .. } | NodeInner::Root { node, .. } => {
                Some(node.header().num_children as usize)
            }
            NodeInner::Bucket { bucket } => {
                if self.bucket_position.is_some() {
                    // Within a suffix, there's at most one edge
                    Some(1)
                } else {
                    // Count unique first bytes
                    let mut count = 0;
                    let mut seen = [false; 256];
                    for i in 0..bucket.len() {
                        if let Some(entry) = bucket.get_entry(i) {
                            let suffix = bucket.get_suffix(&entry);
                            if !suffix.is_empty() && !seen[suffix[0] as usize] {
                                seen[suffix[0] as usize] = true;
                                count += 1;
                            }
                        }
                    }
                    Some(count)
                }
            }
            NodeInner::Empty => Some(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_node() {
        let node: PersistentARTrieNode<()> = PersistentARTrieNode::empty();
        assert!(node.is_empty());
        assert!(!node.is_final());
        assert!(node.transition(b'a').is_none());
        assert_eq!(node.edges().count(), 0);
    }

    #[test]
    fn test_root_node() {
        let node: PersistentARTrieNode<()> = PersistentARTrieNode::new_root();
        assert!(!node.is_empty());
        assert!(!node.is_final());
    }

    #[test]
    fn test_bucket_node() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();
        bucket.insert_key(b"cherry").unwrap();

        let node: PersistentARTrieNode<()> = PersistentARTrieNode::new_bucket(bucket);

        // Should have edges for 'a', 'b', 'c'
        let edges: Vec<_> = node.edges().collect();
        assert_eq!(edges.len(), 3);

        // Transition through 'a' -> 'p' -> 'p' -> 'l' -> 'e'
        let a_node = node.transition(b'a');
        assert!(a_node.is_some());

        let p1_node = a_node.unwrap().transition(b'p');
        assert!(p1_node.is_some());

        let p2_node = p1_node.unwrap().transition(b'p');
        assert!(p2_node.is_some());

        let l_node = p2_node.unwrap().transition(b'l');
        assert!(l_node.is_some());

        let e_node = l_node.unwrap().transition(b'e');
        assert!(e_node.is_some());
        assert!(e_node.unwrap().is_final());
    }

    #[test]
    fn test_bucket_with_empty_suffix() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"").unwrap();
        bucket.insert_key(b"test").unwrap();

        let node: PersistentARTrieNode<()> = PersistentARTrieNode::new_bucket(bucket);

        // Should be final at root (empty suffix exists)
        assert!(node.is_final());
    }

    #[test]
    fn test_art_node_final() {
        let mut art = Node4::new();
        art.header.set_final(true);

        let node: PersistentARTrieNode<()> =
            PersistentARTrieNode::new_art_node(Node::N4(Box::new(art)), true, None);

        assert!(node.is_final());
    }

    #[test]
    fn test_clone() {
        let node: PersistentARTrieNode<()> = PersistentARTrieNode::new_root();
        let cloned = node.clone();

        assert!(!cloned.is_empty());
        assert!(!cloned.is_final());
    }
}
