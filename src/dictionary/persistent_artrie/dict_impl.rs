//! Dictionary Implementation for Persistent ART
//!
//! This module provides the `PersistentARTrie` dictionary type that implements
//! the `Dictionary` and `MappedDictionary` traits for integration with the
//! Levenshtein automata transducer.
//!
//! # In-Memory vs Disk-Backed
//!
//! This implementation currently provides an in-memory version suitable for
//! development and testing. The disk-backed version with memory-mapped I/O
//! will be added in a future phase.
//!
//! # Thread Safety
//!
//! The dictionary uses `Arc<RwLock>` for thread-safe concurrent access.
//! Read operations can proceed in parallel, while writes are serialized.

use std::sync::{Arc, RwLock};

use crate::dictionary::{Dictionary, MappedDictionary, SyncStrategy};
use crate::dictionary::value::DictionaryValue;
use super::bucket::StringBucket;
use super::node_impl::PersistentARTrieNode;
use super::nodes::{ArtNode, Node, Node4, AddChildError};
use super::swizzled_ptr::SwizzledPtr;
use super::transitions::{bucket_to_art_node, ChildNode};

/// A Persistent Adaptive Radix Trie dictionary.
///
/// This dictionary stores terms in a hybrid structure combining:
/// - **ART nodes** for efficient internal node traversal (Node4/16/48/256)
/// - **String buckets** for efficient leaf storage (multiple terms per bucket)
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
///
/// let mut dict = PersistentARTrie::new();
/// dict.insert("hello");
/// dict.insert("world");
///
/// assert!(dict.contains("hello"));
/// assert!(!dict.contains("hi"));
/// ```
#[derive(Clone)]
pub struct PersistentARTrie<V: DictionaryValue = ()> {
    /// Inner state protected by read-write lock
    pub(crate) inner: Arc<RwLock<PersistentARTrieInner<V>>>,
}

/// Inner state of the Persistent ART
pub(crate) struct PersistentARTrieInner<V: DictionaryValue> {
    /// Root node of the trie (starts as a bucket, grows to ART)
    pub(crate) root: TrieRoot<V>,
    /// Number of terms in the dictionary
    pub(crate) term_count: usize,
    /// Whether the dictionary has been modified
    pub(crate) dirty: bool,
}

/// The root of the trie can be either a bucket or an ART node
pub(crate) enum TrieRoot<V: DictionaryValue> {
    /// Root is a single bucket (for small dictionaries)
    Bucket(StringBucket),
    /// Root is an ART node (for larger dictionaries)
    ArtNode {
        /// The root ART node
        node: Node,
        /// Child nodes (bucket or sub-ART)
        children: Vec<(u8, ChildNode)>,
        /// Whether empty string is in dictionary
        is_final: bool,
        /// Value for empty string
        value: Option<V>,
    },
}

impl<V: DictionaryValue> PersistentARTrie<V> {
    /// Create a new empty dictionary
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(PersistentARTrieInner {
                root: TrieRoot::Bucket(StringBucket::with_values()),
                term_count: 0,
                dirty: false,
            })),
        }
    }

    /// Insert a term into the dictionary (without value)
    pub fn insert(&mut self, term: &str) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.insert_impl(term.as_bytes(), None)
    }

    /// Insert a term with an associated value
    pub fn insert_with_value(&mut self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.insert_impl(term.as_bytes(), Some(value))
    }

    /// Remove a term from the dictionary
    pub fn remove(&mut self, term: &str) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.remove_impl(term.as_bytes())
    }

    /// Check if the dictionary is dirty (has uncommitted changes)
    pub fn is_dirty(&self) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        inner.dirty
    }

    /// Mark the dictionary as clean (after flushing to disk)
    pub fn mark_clean(&mut self) {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.dirty = false;
    }

    /// Get a snapshot node for traversal
    fn get_root_node(&self) -> PersistentARTrieNode<V> {
        let inner = self.inner.read().expect("lock poisoned");
        match &inner.root {
            TrieRoot::Bucket(bucket) => PersistentARTrieNode::new_bucket(bucket.clone()),
            TrieRoot::ArtNode {
                node,
                is_final,
                value,
                ..
            } => PersistentARTrieNode::new_art_node(
                node.clone(),
                *is_final,
                value.clone(),
            ),
        }
    }
}

impl<V: DictionaryValue> Default for PersistentARTrie<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> PersistentARTrieInner<V> {
    /// Insert implementation
    fn insert_impl(&mut self, term: &[u8], value: Option<V>) -> bool {
        let inserted = match &mut self.root {
            TrieRoot::Bucket(bucket) => {
                // Clone value here in case we need to retry after bucket conversion
                let value_for_retry = value.clone();

                let result = bucket.insert_key(term);

                match result {
                    Ok(inserted) => {
                        // Check if bucket needs to be converted to ART
                        if bucket.header().should_split() {
                            self.convert_bucket_to_art();
                        }
                        inserted
                    }
                    Err(_) => {
                        // Bucket is full, convert to ART and retry
                        self.convert_bucket_to_art();
                        // Retry insert in the new ART structure
                        self.insert_impl(term, value_for_retry);
                        true
                    }
                }
            }
            TrieRoot::ArtNode {
                node,
                children,
                is_final,
                value: root_value,
            } => {
                if term.is_empty() {
                    // Inserting empty string
                    if *is_final {
                        if value.is_some() {
                            *root_value = value;
                        }
                        false
                    } else {
                        *is_final = true;
                        *root_value = value;
                        true
                    }
                } else {
                    // Find or create child for first byte
                    let first_byte = term[0];
                    let remaining = &term[1..];

                    // Find existing child
                    let child_idx = children.iter().position(|(b, _)| *b == first_byte);

                    if let Some(idx) = child_idx {
                        // Insert into existing child
                        match &mut children[idx].1 {
                            ChildNode::Bucket(bucket) => {
                                match bucket.insert_key(remaining) {
                                    Ok(inserted) => inserted,
                                    Err(_) => {
                                        // TODO: Handle bucket split/conversion
                                        false
                                    }
                                }
                            }
                            ChildNode::ArtNode { .. } => {
                                // TODO: Recursive insert into child ART
                                false
                            }
                        }
                    } else {
                        // Create new child bucket
                        let mut bucket = StringBucket::with_values();
                        let _ = bucket.insert_key(remaining);

                        // Add child to ART node
                        let ptr = SwizzledPtr::null();
                        let _ = match node {
                            Node::N4(n) => n.add_child(first_byte, ptr),
                            Node::N16(n) => n.add_child(first_byte, ptr),
                            Node::N48(n) => n.add_child(first_byte, ptr),
                            Node::N256(n) => n.add_child(first_byte, ptr),
                        };

                        children.push((first_byte, ChildNode::Bucket(bucket)));
                        true
                    }
                }
            }
        };

        if inserted {
            self.term_count += 1;
            self.dirty = true;
        }

        inserted
    }

    /// Remove implementation
    fn remove_impl(&mut self, term: &[u8]) -> bool {
        let removed = match &mut self.root {
            TrieRoot::Bucket(bucket) => bucket.remove(term).is_some(),
            TrieRoot::ArtNode {
                node: _,
                children,
                is_final,
                value,
            } => {
                if term.is_empty() {
                    if *is_final {
                        *is_final = false;
                        *value = None;
                        true
                    } else {
                        false
                    }
                } else {
                    let first_byte = term[0];
                    let remaining = &term[1..];

                    let child_idx = children.iter().position(|(b, _)| *b == first_byte);

                    if let Some(idx) = child_idx {
                        match &mut children[idx].1 {
                            ChildNode::Bucket(bucket) => bucket.remove(remaining).is_some(),
                            ChildNode::ArtNode { .. } => {
                                // TODO: Recursive remove
                                false
                            }
                        }
                    } else {
                        false
                    }
                }
            }
        };

        if removed {
            self.term_count -= 1;
            self.dirty = true;
        }

        removed
    }

    /// Convert root bucket to ART node structure
    fn convert_bucket_to_art(&mut self) {
        if let TrieRoot::Bucket(bucket) = &self.root {
            if let Some(result) = bucket_to_art_node(bucket).ok() {
                let children: Vec<(u8, ChildNode)> = result
                    .children
                    .into_iter()
                    .map(|(b, bucket)| (b, ChildNode::Bucket(bucket)))
                    .collect();

                self.root = TrieRoot::ArtNode {
                    node: result.node,
                    children,
                    is_final: result.is_final,
                    value: None, // TODO: preserve value
                };
            }
        }
    }
}

impl<V: DictionaryValue> Dictionary for PersistentARTrie<V> {
    type Node = PersistentARTrieNode<V>;

    fn root(&self) -> Self::Node {
        self.get_root_node()
    }

    fn contains(&self, term: &str) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        match &inner.root {
            TrieRoot::Bucket(bucket) => bucket.contains(term.as_bytes()),
            TrieRoot::ArtNode {
                node: _,
                children,
                is_final,
                ..
            } => {
                if term.is_empty() {
                    return *is_final;
                }

                let bytes = term.as_bytes();
                let first_byte = bytes[0];
                let remaining = &bytes[1..];

                for (b, child) in children {
                    if *b == first_byte {
                        return match child {
                            ChildNode::Bucket(bucket) => bucket.contains(remaining),
                            ChildNode::ArtNode { .. } => {
                                // TODO: Recursive contains
                                false
                            }
                        };
                    }
                }
                false
            }
        }
    }

    fn len(&self) -> Option<usize> {
        let inner = self.inner.read().expect("lock poisoned");
        Some(inner.term_count)
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync
    }
}

impl<V: DictionaryValue> MappedDictionary for PersistentARTrie<V> {
    type Value = V;

    fn get_value(&self, _term: &str) -> Option<Self::Value> {
        // TODO: Implement value retrieval
        None
    }
}

impl<V: DictionaryValue> std::fmt::Debug for PersistentARTrie<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().expect("lock poisoned");
        f.debug_struct("PersistentARTrie")
            .field("term_count", &inner.term_count)
            .field("dirty", &inner.dirty)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dictionary() {
        let dict: PersistentARTrie = PersistentARTrie::new();
        assert_eq!(dict.len(), Some(0));
        assert!(!dict.is_dirty());
    }

    #[test]
    fn test_insert_and_contains() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert("apple"));
        assert!(dict.insert("banana"));
        assert!(dict.insert("cherry"));

        assert!(dict.contains("apple"));
        assert!(dict.contains("banana"));
        assert!(dict.contains("cherry"));
        assert!(!dict.contains("date"));

        assert_eq!(dict.len(), Some(3));
        assert!(dict.is_dirty());
    }

    #[test]
    fn test_duplicate_insert() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert("test"));
        assert!(!dict.insert("test")); // Duplicate

        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_remove() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("apple");
        dict.insert("banana");

        assert!(dict.remove("apple"));
        assert!(!dict.contains("apple"));
        assert!(dict.contains("banana"));

        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_remove_not_found() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("apple");

        assert!(!dict.remove("banana"));
        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_empty_string() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert(""));
        assert!(dict.contains(""));

        dict.insert("test");
        assert!(dict.contains(""));
        assert!(dict.contains("test"));
    }

    #[test]
    fn test_dictionary_trait() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("hello");
        dict.insert("world");

        // Test through Dictionary trait
        let dict_ref: &dyn Dictionary<Node = _> = &dict;
        assert!(dict_ref.contains("hello"));
        assert!(!dict_ref.contains("hi"));
    }

    #[test]
    fn test_mark_clean() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("test");
        assert!(dict.is_dirty());

        dict.mark_clean();
        assert!(!dict.is_dirty());
    }

    #[test]
    fn test_many_insertions() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        // Insert many terms to trigger bucket conversion
        for i in 0..100 {
            dict.insert(&format!("word{:03}", i));
        }

        assert_eq!(dict.len(), Some(100));

        // Verify all terms exist
        for i in 0..100 {
            assert!(dict.contains(&format!("word{:03}", i)));
        }
    }

    #[test]
    fn test_sync_strategy() {
        let dict: PersistentARTrie = PersistentARTrie::new();
        assert_eq!(dict.sync_strategy(), SyncStrategy::InternalSync);
    }

    #[test]
    fn test_clone() {
        let mut dict1: PersistentARTrie = PersistentARTrie::new();
        dict1.insert("test");

        let dict2 = dict1.clone();

        // Both should see the same data (Arc sharing)
        assert!(dict2.contains("test"));
        assert_eq!(dict2.len(), Some(1));
    }
}
