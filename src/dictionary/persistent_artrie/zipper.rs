//! Persistent ART zipper implementation.
//!
//! This module provides a zipper implementation for PersistentARTrie that uses
//! node-based navigation with lock-per-operation pattern for thread safety.

use std::sync::{Arc, RwLock};

use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use super::bucket::StringBucket;
use super::dict_impl::{PersistentARTrie, PersistentARTrieInner, TrieRoot};
use super::transitions::ChildNode;

/// Zipper for Persistent ART dictionaries.
///
/// `PersistentARTrieZipper` provides efficient navigation through Persistent ART
/// structures using a path-based approach with thread-safe concurrent access.
///
/// # Design
///
/// The zipper stores:
/// - `inner`: Shared reference to the trie inner structure (Arc<RwLock>)
/// - `path`: Current path from root to focus
/// - `state`: Current navigation state (root, bucket, or ART node)
///
/// Operations use a lock-per-operation pattern, acquiring a read lock only for
/// the duration of each operation to maximize concurrency.
///
/// # Thread Safety
///
/// Each operation acquires a read lock, performs the operation, and releases it.
/// This allows:
/// - Multiple concurrent readers (navigating different zippers)
/// - Exclusive write access for modifications (insert/remove)
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
/// use liblevenshtein::dictionary::persistent_artrie::zipper::PersistentARTrieZipper;
///
/// let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
/// dict.insert("cat");
/// dict.insert("catch");
///
/// let zipper = PersistentARTrieZipper::new_from_dict(&dict);
///
/// // Navigate through "cat"
/// if let Some(c) = zipper.descend(b'c') {
///     if let Some(a) = c.descend(b'a') {
///         if let Some(t) = a.descend(b't') {
///             if t.is_final() {
///                 println!("Found 'cat'");
///             }
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct PersistentARTrieZipper<V: DictionaryValue = ()> {
    /// Shared reference to trie inner structure
    inner: Arc<RwLock<PersistentARTrieInner<V>>>,

    /// Path from root to current position
    path: Vec<u8>,
}

impl<V: DictionaryValue> PersistentARTrieZipper<V> {
    /// Create a new zipper at the root of the Persistent ART.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the PersistentARTrie dictionary
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    /// use liblevenshtein::dictionary::persistent_artrie::zipper::PersistentARTrieZipper;
    ///
    /// let dict: PersistentARTrie<()> = PersistentARTrie::new();
    /// let zipper = PersistentARTrieZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &PersistentARTrie<V>) -> Self {
        PersistentARTrieZipper {
            inner: dict.inner.clone(),
            path: Vec::new(),
        }
    }

    /// Get the current path from root.
    ///
    /// Returns the sequence of edge labels from root to current position.
    pub fn current_path(&self) -> &[u8] {
        &self.path
    }
}

impl<V: DictionaryValue> DictZipper for PersistentARTrieZipper<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        self.is_final_at_path(&inner, &self.path)
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        let inner = self.inner.read().expect("lock poisoned");

        // Check if the path + label leads to a valid position
        let mut new_path = self.path.clone();
        new_path.push(label);

        if self.has_path(&inner, &new_path) {
            Some(PersistentARTrieZipper {
                inner: self.inner.clone(),
                path: new_path,
            })
        } else {
            None
        }
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.path.clone()
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        // Collect children to avoid holding lock during iteration
        let children: Vec<u8> = {
            let inner = self.inner.read().expect("lock poisoned");
            self.get_children_at_path(&inner, &self.path)
        };

        // Create iterator from collected children
        let inner = self.inner.clone();
        let base_path = self.path.clone();
        children.into_iter().map(move |label| {
            let mut new_path = base_path.clone();
            new_path.push(label);
            (
                label,
                PersistentARTrieZipper {
                    inner: inner.clone(),
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> PersistentARTrieZipper<V> {
    /// Check if a path exists in the trie
    fn has_path(&self, inner: &PersistentARTrieInner<V>, path: &[u8]) -> bool {
        match &inner.root {
            TrieRoot::Bucket(bucket) => {
                // For bucket root, check if any entry starts with or equals this path
                self.bucket_has_path(bucket, path)
            }
            TrieRoot::ArtNode { children, .. } => {
                if path.is_empty() {
                    true // Root always exists
                } else {
                    let first_byte = path[0];
                    let remaining = &path[1..];

                    for (b, child) in children {
                        if *b == first_byte {
                            return match child {
                                ChildNode::Bucket(bucket) => {
                                    self.bucket_has_path(bucket, remaining)
                                }
                                ChildNode::ArtNode { .. } => {
                                    // TODO: Recursive check for nested ART nodes
                                    !remaining.is_empty()
                                }
                            };
                        }
                    }
                    false
                }
            }
        }
    }

    /// Check if bucket has entries starting with path
    fn bucket_has_path(&self, bucket: &StringBucket, path: &[u8]) -> bool {
        // Empty path always matches (root of bucket)
        if path.is_empty() {
            return true;
        }

        // Check if any entry starts with this path
        for i in 0..bucket.len() {
            if let Some(entry) = bucket.get_entry(i) {
                let suffix = bucket.get_suffix(&entry);
                if suffix.starts_with(path) || suffix == path {
                    return true;
                }
            }
        }
        false
    }

    /// Check if position is final
    fn is_final_at_path(&self, inner: &PersistentARTrieInner<V>, path: &[u8]) -> bool {
        match &inner.root {
            TrieRoot::Bucket(bucket) => {
                bucket.contains(path)
            }
            TrieRoot::ArtNode { children, is_final, .. } => {
                if path.is_empty() {
                    return *is_final;
                }

                let first_byte = path[0];
                let remaining = &path[1..];

                for (b, child) in children {
                    if *b == first_byte {
                        return match child {
                            ChildNode::Bucket(bucket) => {
                                bucket.contains(remaining)
                            }
                            ChildNode::ArtNode { is_final: child_final, .. } => {
                                if remaining.is_empty() {
                                    *child_final
                                } else {
                                    // TODO: Recursive check for nested ART nodes
                                    false
                                }
                            }
                        };
                    }
                }
                false
            }
        }
    }

    /// Get all children (edge labels) at current path
    fn get_children_at_path(&self, inner: &PersistentARTrieInner<V>, path: &[u8]) -> Vec<u8> {
        match &inner.root {
            TrieRoot::Bucket(bucket) => {
                self.get_bucket_children(bucket, path)
            }
            TrieRoot::ArtNode { children, .. } => {
                if path.is_empty() {
                    // At root, return all first-level children
                    children.iter().map(|(b, _)| *b).collect()
                } else {
                    let first_byte = path[0];
                    let remaining = &path[1..];

                    for (b, child) in children {
                        if *b == first_byte {
                            return match child {
                                ChildNode::Bucket(bucket) => {
                                    self.get_bucket_children(bucket, remaining)
                                }
                                ChildNode::ArtNode { .. } => {
                                    // TODO: Handle nested ART nodes
                                    Vec::new()
                                }
                            };
                        }
                    }
                    Vec::new()
                }
            }
        }
    }

    /// Get children from bucket at given prefix
    fn get_bucket_children(&self, bucket: &StringBucket, prefix: &[u8]) -> Vec<u8> {
        let mut seen = [false; 256];
        let mut children = Vec::new();

        for i in 0..bucket.len() {
            if let Some(entry) = bucket.get_entry(i) {
                let suffix = bucket.get_suffix(&entry);
                if suffix.starts_with(prefix) && suffix.len() > prefix.len() {
                    let next_byte = suffix[prefix.len()];
                    if !seen[next_byte as usize] {
                        seen[next_byte as usize] = true;
                        children.push(next_byte);
                    }
                }
            }
        }

        children
    }
}

impl<V: DictionaryValue> ValuedDictZipper for PersistentARTrieZipper<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        // TODO: Implement value retrieval from buckets
        // For now, return None as values aren't fully supported yet
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_zipper_not_final() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("test");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
    }

    #[test]
    fn test_descend_single_term() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("cat");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);

        let c = zipper.descend(b'c').expect("should have 'c'");
        let a = c.descend(b'a').expect("should have 'a'");
        let t = a.descend(b't').expect("should have 't'");

        assert!(t.is_final());
    }

    #[test]
    fn test_descend_nonexistent() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("cat");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);

        assert!(zipper.descend(b'x').is_none());
    }

    #[test]
    fn test_children() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("can");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);

        // Root should have 'c' as child
        let children: Vec<_> = zipper.children().collect();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].0, b'c');
    }

    #[test]
    fn test_path() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("cat");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);
        assert!(zipper.path().is_empty());

        let c = zipper.descend(b'c').unwrap();
        assert_eq!(c.path(), vec![b'c']);

        let a = c.descend(b'a').unwrap();
        assert_eq!(a.path(), vec![b'c', b'a']);
    }

    #[test]
    fn test_empty_string() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("");
        dict.insert("test");

        let zipper = PersistentARTrieZipper::new_from_dict(&dict);
        assert!(zipper.is_final()); // Empty string makes root final
    }

    #[test]
    fn test_clone() {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        dict.insert("test");

        let zipper1 = PersistentARTrieZipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        assert_eq!(zipper1.path(), zipper2.path());
        assert_eq!(zipper1.is_final(), zipper2.is_final());
    }
}
