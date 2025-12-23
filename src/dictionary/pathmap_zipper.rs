//! PathMap-backed zipper implementation.
//!
//! This module provides a zipper implementation for PathMapDictionary that uses
//! path-based navigation with lock-per-operation pattern for thread safety.

use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use pathmap::utils::BitMask;
use pathmap::zipper::{ReadZipperUntracked, Zipper, ZipperMoving, ZipperValues};
use pathmap::PathMap;
use std::sync::Arc;

use crate::sync_compat::RwLock;

/// Zipper for PathMap-backed dictionaries.
///
/// `PathMapZipper` provides efficient navigation through PathMap trie structures
/// using a path-based approach with structural sharing via `Arc`.
///
/// # Design
///
/// Instead of storing a persistent zipper (which has lifetime issues with RwLock),
/// we store:
/// - `map`: Shared reference to the PathMap (Arc<RwLock>)
/// - `path`: Current path from root (Arc<Vec<u8>> for cheap cloning)
///
/// Zippers are recreated on-demand for each operation using a lock-per-operation
/// pattern, which maximizes concurrency by only holding locks briefly.
///
/// # Thread Safety
///
/// Each operation acquires a read lock, performs the operation, and releases it.
/// This allows:
/// - Multiple concurrent readers (navigating different zippers)
/// - Exclusive write access for modifications (insert/remove)
///
/// # Performance
///
/// - `Arc<Vec<u8>>` for paths: Cheap cloning via reference counting
/// - Lock-per-operation: Minimal lock contention
/// - No heap allocation for short paths (up to pointer size)
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
///
/// let dict = PathMapDictionary::<()>::new();
/// // ... insert terms ...
///
/// let zipper = PathMapZipper::new_from_dict(&dict);
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
pub struct PathMapZipper<V: DictionaryValue> {
    /// Shared reference to PathMap (wrapped in RwLock for thread safety)
    map: Arc<RwLock<PathMap<V>>>,

    /// Current path from root (Arc for cheap cloning)
    /// Changed from Arc<Vec<u8>> to Arc<[u8]> for true COW semantics
    path: Arc<[u8]>,
}

impl<V: DictionaryValue> PathMapZipper<V> {
    /// Create a new zipper at the root of the PathMap.
    ///
    /// # Arguments
    ///
    /// * `map` - Shared reference to the PathMap
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// let zipper = PathMapZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &crate::dictionary::pathmap::PathMapDictionary<V>) -> Self {
        PathMapZipper {
            map: dict.map.clone(),
            path: Arc::from(vec![]),
        }
    }

    /// Create a zipper from an Arc<RwLock<PathMap<V>>> directly.
    ///
    /// This is used internally when the PathMap reference is already available.
    pub fn new(map: Arc<RwLock<PathMap<V>>>) -> Self {
        PathMapZipper {
            map,
            path: Arc::from(vec![]),
        }
    }

    /// Create a zipper at a specific path.
    ///
    /// # Arguments
    ///
    /// * `map` - Shared reference to the PathMap
    /// * `path` - Path from root to desired position
    ///
    /// # Note
    ///
    /// This does not verify that the path exists. Use `descend()` from root
    /// if you need validation.
    pub fn at_path(map: Arc<RwLock<PathMap<V>>>, path: Vec<u8>) -> Self {
        PathMapZipper {
            map,
            path: Arc::from(path),
        }
    }

    /// Execute a function with a temporary zipper at the current path.
    ///
    /// This is the core operation that manages lock acquisition and zipper creation.
    /// The function `f` is called with a borrowed zipper positioned at the current path.
    ///
    /// # Thread Safety
    ///
    /// Acquires a read lock on the PathMap for the duration of `f`.
    ///
    /// # Panics
    ///
    /// Panics if the RwLock is poisoned (another thread panicked while holding the lock).
    #[inline(always)]
    fn with_zipper<F, R>(&self, f: F) -> R
    where
        F: FnOnce(ReadZipperUntracked<'_, 'static, V>) -> R,
    {
        let map = self.map.read();
        let zipper = if self.path.is_empty() {
            map.read_zipper()
        } else {
            map.read_zipper_at_path(&self.path)
        };
        f(zipper)
    }
}

impl<V: DictionaryValue> DictZipper for PathMapZipper<V> {
    type Unit = u8;

    #[inline]
    fn is_final(&self) -> bool {
        self.with_zipper(|z| z.is_val())
    }

    #[inline]
    fn descend(&self, label: Self::Unit) -> Option<Self> {
        // Build new path (COW: only allocate when extending)
        let mut new_path = Vec::with_capacity(self.path.len() + 1);
        new_path.extend_from_slice(&self.path);
        new_path.push(label);

        // Check if path exists in PathMap
        let exists = {
            let map = self.map.read();
            let mut zipper = map.read_zipper();
            zipper.descend_to(&new_path);
            zipper.path_exists()
        };

        if exists {
            Some(PathMapZipper {
                map: Arc::clone(&self.map),
                path: Arc::from(new_path), // Convert Vec<u8> to Arc<[u8]>
            })
        } else {
            None
        }
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        // OPTIMIZATION: Lock batching - acquire lock once and validate all children
        // This reduces lock acquisitions from O(n) to O(1) where n is the number of candidates

        let map = Arc::clone(&self.map);
        let path = Arc::clone(&self.path);

        // Acquire lock once and extract all child information
        let valid_children: Vec<(u8, Arc<[u8]>)> = {
            let map_guard = self.map.read();
            let zipper = if self.path.is_empty() {
                map_guard.read_zipper()
            } else {
                map_guard.read_zipper_at_path(&self.path)
            };

            // Get child mask to filter candidates
            let mask = zipper.child_mask();

            // Validate all potential children in one critical section
            (0u8..=255)
                .filter(|&byte| mask.test_bit(byte))
                .filter_map(|byte| {
                    // Build candidate path
                    let mut new_path = Vec::with_capacity(path.len() + 1);
                    new_path.extend_from_slice(&path);
                    new_path.push(byte);

                    // Validate path exists (within same lock)
                    let mut test_zipper = map_guard.read_zipper();
                    test_zipper.descend_to(&new_path);

                    if test_zipper.path_exists() {
                        Some((byte, Arc::from(new_path)))
                    } else {
                        None
                    }
                })
                .collect()
        }; // Lock released here

        // Convert to iterator outside the critical section
        valid_children.into_iter().map(move |(byte, child_path)| {
            (
                byte,
                PathMapZipper {
                    map: Arc::clone(&map),
                    path: child_path,
                },
            )
        })
    }

    #[inline]
    fn path(&self) -> Vec<Self::Unit> {
        self.path.to_vec()
    }
}

impl<V: DictionaryValue> ValuedDictZipper for PathMapZipper<V> {
    type Value = V;

    #[inline]
    fn value(&self) -> Option<Self::Value> {
        self.with_zipper(|z| z.val().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::pathmap::PathMapDictionary;
    use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};

    #[test]
    fn test_root_zipper_not_final() {
        let dict = PathMapDictionary::<()>::new();
        let zipper = PathMapZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.path(), Vec::<u8>::new());
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict = PathMapDictionary::<()>::new();
        let zipper = PathMapZipper::new_from_dict(&dict);

        assert!(zipper.descend(b'a').is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("catch");

        let zipper = PathMapZipper::new_from_dict(&dict);

        // Navigate to 'c'
        let c = zipper.descend(b'c').expect("'c' should exist");
        assert!(!c.is_final());
        assert_eq!(c.path(), vec![b'c']);

        // Navigate to 'ca'
        let a = c.descend(b'a').expect("'a' should exist");
        assert!(!a.is_final());
        assert_eq!(a.path(), vec![b'c', b'a']);

        // Navigate to 'cat'
        let t = a.descend(b't').expect("'t' should exist");
        assert!(t.is_final()); // "cat" is a complete term
        assert_eq!(t.path(), vec![b'c', b'a', b't']);

        // Navigate to 'catc'
        let c2 = t.descend(b'c').expect("'c' should exist");
        assert!(!c2.is_final());

        // Navigate to 'catch'
        let h = c2.descend(b'h').expect("'h' should exist");
        assert!(h.is_final()); // "catch" is a complete term
        assert_eq!(h.path(), vec![b'c', b'a', b't', b'c', b'h']);
    }

    #[test]
    fn test_children_iteration() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("a");
        dict.insert("b");
        dict.insert("c");

        let zipper = PathMapZipper::new_from_dict(&dict);

        let children: Vec<_> = zipper.children().collect();

        assert_eq!(children.len(), 3);

        let labels: Vec<u8> = children.iter().map(|(label, _)| *label).collect();
        assert!(labels.contains(&b'a'));
        assert!(labels.contains(&b'b'));
        assert!(labels.contains(&b'c'));

        // Verify each child is final (single-character terms)
        for (_, child) in children {
            assert!(child.is_final());
        }
    }

    #[test]
    fn test_children_with_prefix() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let zipper = PathMapZipper::new_from_dict(&dict);

        // Navigate to 'c'
        let c = zipper.descend(b'c').expect("'c' should exist");

        // 'c' should have one child: 'a'
        let children: Vec<_> = c.children().collect();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].0, b'a');

        // Navigate to 'ca'
        let a = c.descend(b'a').expect("'a' should exist");

        // 'ca' should have two children: 't' and 'r'
        let children: Vec<_> = a.children().collect();
        assert_eq!(children.len(), 2);

        let labels: Vec<u8> = children.iter().map(|(label, _)| *label).collect();
        assert!(labels.contains(&b't'));
        assert!(labels.contains(&b'r'));
    }

    #[test]
    fn test_valued_zipper() {
        let dict = PathMapDictionary::<u32>::new();
        dict.insert_with_value("print", 42);
        dict.insert_with_value("parse", 100);

        let zipper = PathMapZipper::new_from_dict(&dict);

        // Navigate to "print"
        let p = zipper.descend(b'p').unwrap();
        let r = p.descend(b'r').unwrap();
        let i = r.descend(b'i').unwrap();
        let n = i.descend(b'n').unwrap();
        let t = n.descend(b't').unwrap();

        assert!(t.is_final());
        assert_eq!(t.value(), Some(42));

        // Navigate to "parse" - restart from 'p'
        let a = p.descend(b'a').unwrap();
        let r = a.descend(b'r').unwrap();
        let s = r.descend(b's').unwrap();
        let e = s.descend(b'e').unwrap();

        assert!(e.is_final());
        assert_eq!(e.value(), Some(100));
    }

    #[test]
    fn test_valued_zipper_with_vec() {
        let dict = PathMapDictionary::<Vec<u32>>::new();
        dict.insert_with_value("global", vec![0]);
        dict.insert_with_value("local", vec![1, 2, 3]);

        let zipper = PathMapZipper::new_from_dict(&dict);

        // Navigate to "global"
        let mut z = zipper.clone();
        for &byte in b"global" {
            z = z.descend(byte).unwrap();
        }

        assert!(z.is_final());
        assert_eq!(z.value(), Some(vec![0]));

        // Navigate to "local"
        let mut z = zipper;
        for &byte in b"local" {
            z = z.descend(byte).unwrap();
        }

        assert!(z.is_final());
        assert_eq!(z.value(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_path_reconstruction() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("hello");

        let zipper = PathMapZipper::new_from_dict(&dict);

        let mut z = zipper;
        let mut expected_path = Vec::new();

        for &byte in b"hello" {
            z = z.descend(byte).unwrap();
            expected_path.push(byte);
            assert_eq!(z.path(), expected_path);
        }

        assert_eq!(z.path(), b"hello".to_vec());
        assert_eq!(String::from_utf8(z.path()).unwrap(), "hello");
    }

    #[test]
    fn test_clone_independence() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("abc");

        let zipper = PathMapZipper::new_from_dict(&dict);

        let z1 = zipper.clone();
        let z2 = zipper.clone();

        // Navigate z1
        let z1_a = z1.descend(b'a').unwrap();

        // z2 should still be at root
        assert_eq!(z2.path(), Vec::<u8>::new());

        // z1_a should be at 'a'
        assert_eq!(z1_a.path(), vec![b'a']);
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = PathMapDictionary::<()>::new();
        let zipper = PathMapZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.path(), Vec::<u8>::new());

        // No children in empty dictionary
        let children: Vec<_> = zipper.children().collect();
        assert_eq!(children.len(), 0);

        // Can't descend anywhere
        assert!(zipper.descend(b'a').is_none());
    }
}
