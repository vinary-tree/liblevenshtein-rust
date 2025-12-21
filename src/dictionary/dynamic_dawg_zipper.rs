//! Dynamic DAWG zipper implementation.
//!
//! This module provides a zipper implementation for DynamicDawg that uses
//! node-index-based navigation with lock-per-operation pattern for thread safety.

use crate::dictionary::dynamic_dawg::{DynamicDawg, DynamicDawgInner};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use crate::sync_compat::RwLock;
use std::sync::Arc;

/// Zipper for Dynamic DAWG dictionaries.
///
/// `DynamicDawgZipper` provides efficient navigation through Dynamic DAWG structures
/// using a node-index-based approach with thread-safe concurrent access.
///
/// # Design
///
/// The zipper stores:
/// - `inner`: Shared reference to the DAWG inner structure (Arc<RwLock>)
/// - `node`: Current node index in the DAWG
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
/// # Performance
///
/// - Node-index-based: No path storage overhead
/// - Lock-per-operation: Minimal lock contention
/// - Lightweight Clone: Just Arc clone + usize copy
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::dynamic_dawg::DynamicDawg;
/// use liblevenshtein::dictionary::dynamic_dawg_zipper::DynamicDawgZipper;
///
/// let dict: DynamicDawg<()> = DynamicDawg::new();
/// dict.insert("cat");
/// dict.insert("catch");
///
/// let zipper = DynamicDawgZipper::new_from_dict(&dict);
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
pub struct DynamicDawgZipper<V: DictionaryValue = ()> {
    /// Shared reference to DAWG inner structure
    inner: Arc<RwLock<DynamicDawgInner<V>>>,

    /// Current node index (0 is root)
    node: usize,

    /// Path from root to current position
    path: Vec<u8>,
}

impl<V: DictionaryValue> DynamicDawgZipper<V> {
    /// Create a new zipper at the root of the Dynamic DAWG.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the DynamicDawg dictionary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::dynamic_dawg::DynamicDawg;
    /// use liblevenshtein::dictionary::dynamic_dawg_zipper::DynamicDawgZipper;
    ///
    /// let dict: DynamicDawg<()> = DynamicDawg::new();
    /// let zipper = DynamicDawgZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &DynamicDawg<V>) -> Self {
        DynamicDawgZipper {
            inner: dict.inner.clone(),
            node: 0, // Root is always node 0 in DynamicDawg
            path: Vec::new(),
        }
    }

    /// Get the current node index.
    ///
    /// Useful for debugging or advanced use cases.
    pub fn node(&self) -> usize {
        self.node
    }
}

impl<V: DictionaryValue> DictZipper for DynamicDawgZipper<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        if self.node < inner.nodes.len() {
            inner.nodes[self.node].is_final
        } else {
            false
        }
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        let inner = self.inner.read();
        if self.node >= inner.nodes.len() {
            return None;
        }

        // Find the edge with the given label
        for (edge_label, target_node) in &inner.nodes[self.node].edges {
            if *edge_label == label {
                let mut new_path = self.path.clone();
                new_path.push(label);
                return Some(DynamicDawgZipper {
                    inner: self.inner.clone(),
                    node: *target_node,
                    path: new_path,
                });
            }
        }

        None
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.path.clone()
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        // Collect edges to avoid holding lock during iteration
        let edges: Vec<(u8, usize)> = {
            let inner = self.inner.read();
            if self.node < inner.nodes.len() {
                inner.nodes[self.node].edges.iter().copied().collect()
            } else {
                Vec::new()
            }
        };

        // Create iterator from collected edges
        let inner = self.inner.clone();
        let base_path = self.path.clone();
        edges.into_iter().map(move |(label, target)| {
            let mut new_path = base_path.clone();
            new_path.push(label);
            (
                label,
                DynamicDawgZipper {
                    inner: inner.clone(),
                    node: target,
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> ValuedDictZipper for DynamicDawgZipper<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        let inner = self.inner.read();
        if self.node < inner.nodes.len() && inner.nodes[self.node].is_final {
            inner.nodes[self.node].value.clone()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_zipper_not_final() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        dict.insert("test");

        let zipper = DynamicDawgZipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
        assert_eq!(zipper.node(), 0);
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        dict.insert("test");

        let zipper = DynamicDawgZipper::new_from_dict(&dict);
        assert!(zipper.descend(b'x').is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        dict.insert("cat");
        dict.insert("catch");

        let zipper = DynamicDawgZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let c = zipper.descend(b'c').expect("Should descend to 'c'");
        assert!(!c.is_final());

        let a = c.descend(b'a').expect("Should descend to 'a'");
        assert!(!a.is_final());

        let t = a.descend(b't').expect("Should descend to 't'");
        assert!(t.is_final(), "'cat' should be a final state");

        // Continue to "catch"
        let c2 = t.descend(b'c').expect("Should descend to 'c' from 'cat'");
        let h = c2.descend(b'h').expect("Should descend to 'h'");
        assert!(h.is_final(), "'catch' should be a final state");
    }

    #[test]
    fn test_children_iteration() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let zipper = DynamicDawgZipper::new_from_dict(&dict);

        // Root should have children 'c' and 'd'
        let children: Vec<u8> = zipper.children().map(|(label, _)| label).collect();
        assert!(children.contains(&b'c'));
        assert!(children.contains(&b'd'));
    }

    #[test]
    fn test_valued_zipper() {
        let dict: DynamicDawg<u32> = DynamicDawg::new();
        dict.insert_with_value("cat", 1);
        dict.insert_with_value("catch", 2);

        let zipper = DynamicDawgZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let cat_zipper = zipper
            .descend(b'c')
            .and_then(|z| z.descend(b'a'))
            .and_then(|z| z.descend(b't'))
            .expect("Should navigate to 'cat'");

        assert_eq!(cat_zipper.value(), Some(1));

        // Navigate to "catch"
        let catch_zipper = cat_zipper
            .descend(b'c')
            .and_then(|z| z.descend(b'h'))
            .expect("Should navigate to 'catch'");

        assert_eq!(catch_zipper.value(), Some(2));
    }

    #[test]
    fn test_clone_independence() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        dict.insert("test");

        let zipper1 = DynamicDawgZipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        // Both zippers should navigate independently
        let z1_t = zipper1.descend(b't');
        let z2_t = zipper2.descend(b't');

        assert!(z1_t.is_some());
        assert!(z2_t.is_some());
        assert_eq!(z1_t.unwrap().node(), z2_t.unwrap().node());
    }

    #[test]
    fn test_empty_dictionary() {
        let dict: DynamicDawg<()> = DynamicDawg::new();
        let zipper = DynamicDawgZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_value_none_for_non_final() {
        let dict: DynamicDawg<u32> = DynamicDawg::new();
        dict.insert_with_value("cat", 42);

        let zipper = DynamicDawgZipper::new_from_dict(&dict);

        // Navigate to 'c' (non-final)
        let c_zipper = zipper.descend(b'c').expect("Should descend to 'c'");
        assert_eq!(
            c_zipper.value(),
            None,
            "Non-final node should have no value"
        );
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dict = StdArc::new({
            let d: DynamicDawg<()> = DynamicDawg::new();
            d.insert("test");
            d.insert("testing");
            d
        });

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let dict_clone = dict.clone();
                thread::spawn(move || {
                    let zipper = DynamicDawgZipper::new_from_dict(&dict_clone);
                    zipper.descend(b't').is_some()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }
}
