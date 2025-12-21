//! Dynamic DAWG character-level zipper implementation.
//!
//! This module provides a zipper implementation for DynamicDawgChar that uses
//! node-index-based navigation with lock-per-operation pattern for thread safety.
//! Unlike DynamicDawgZipper which operates on bytes, this operates on Unicode
//! characters for correct multi-byte UTF-8 handling.

use crate::dictionary::dynamic_dawg_char::{DynamicDawgChar, DynamicDawgCharInner};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use crate::sync_compat::RwLock;
use std::sync::Arc;

/// Zipper for Dynamic DAWG character-level dictionaries.
///
/// `DynamicDawgCharZipper` provides efficient navigation through Dynamic DAWG structures
/// using a node-index-based approach with thread-safe concurrent access. This variant
/// operates on `char` units instead of `u8` bytes, providing correct Unicode semantics.
///
/// # Design
///
/// The zipper stores:
/// - `inner`: Shared reference to the DAWG inner structure (Arc<RwLock>)
/// - `node`: Current node index in the DAWG
/// - `path`: Path from root to current position (Vec<char>)
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
/// - Lightweight Clone: Just Arc clone + usize copy + path clone
///
/// # Unicode Support
///
/// This zipper correctly handles multi-byte UTF-8 sequences by operating on
/// `char` values (Unicode scalar values) instead of raw bytes:
///
/// ```ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
///
/// let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
/// dict.insert("café");  // 4 characters, 5 bytes
/// dict.insert("naïve"); // 5 characters, 6 bytes
///
/// let zipper = DynamicDawgCharZipper::new_from_dict(&dict);
///
/// // Navigate through "café" (character-by-character)
/// if let Some(c) = zipper.descend('c') {
///     if let Some(a) = c.descend('a') {
///         if let Some(f) = a.descend('f') {
///             if let Some(e) = f.descend('é') {  // Single char 'é'
///                 if e.is_final() {
///                     println!("Found 'café'");
///                 }
///             }
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct DynamicDawgCharZipper<V: DictionaryValue = ()> {
    /// Shared reference to DAWG inner structure
    inner: Arc<RwLock<DynamicDawgCharInner<V>>>,

    /// Current node index (0 is root)
    node: usize,

    /// Path from root to current position
    path: Vec<char>,
}

impl<V: DictionaryValue> DynamicDawgCharZipper<V> {
    /// Create a new zipper at the root of the Dynamic DAWG.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the DynamicDawgChar dictionary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    /// use liblevenshtein::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
    ///
    /// let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
    /// let zipper = DynamicDawgCharZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &DynamicDawgChar<V>) -> Self {
        DynamicDawgCharZipper {
            inner: dict.inner.clone(),
            node: 0, // Root is always node 0 in DynamicDawgChar
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

impl<V: DictionaryValue> DictZipper for DynamicDawgCharZipper<V> {
    type Unit = char;

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
                return Some(DynamicDawgCharZipper {
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
        let edges: Vec<(char, usize)> = {
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
                DynamicDawgCharZipper {
                    inner: inner.clone(),
                    node: target,
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> ValuedDictZipper for DynamicDawgCharZipper<V> {
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
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("test");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
        assert_eq!(zipper.node(), 0);
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("test");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        assert!(zipper.descend('x').is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("cat");
        dict.insert("catch");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let c = zipper.descend('c').expect("Should descend to 'c'");
        assert!(!c.is_final());

        let a = c.descend('a').expect("Should descend to 'a'");
        assert!(!a.is_final());

        let t = a.descend('t').expect("Should descend to 't'");
        assert!(t.is_final(), "'cat' should be a final state");

        // Continue to "catch"
        let c2 = t.descend('c').expect("Should descend to 'c' from 'cat'");
        let h = c2.descend('h').expect("Should descend to 'h'");
        assert!(h.is_final(), "'catch' should be a final state");
    }

    #[test]
    fn test_unicode_navigation() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("café");
        dict.insert("naïve");
        dict.insert("🎉");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate to "café" (4 characters, 5 bytes)
        let cafe_zipper = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('f'))
            .and_then(|z| z.descend('é'))
            .expect("Should navigate to 'café'");

        assert!(cafe_zipper.is_final());

        // Navigate to emoji
        let emoji_zipper = zipper.descend('🎉').expect("Should descend to emoji");
        assert!(emoji_zipper.is_final());
    }

    #[test]
    fn test_children_iteration() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Root should have children 'c' and 'd'
        let children: Vec<char> = zipper.children().map(|(label, _)| label).collect();
        assert!(children.contains(&'c'));
        assert!(children.contains(&'d'));
    }

    #[test]
    fn test_valued_zipper() {
        let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
        dict.insert_with_value("cat", 1);
        dict.insert_with_value("catch", 2);

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let cat_zipper = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('t'))
            .expect("Should navigate to 'cat'");

        assert_eq!(cat_zipper.value(), Some(1));

        // Navigate to "catch"
        let catch_zipper = cat_zipper
            .descend('c')
            .and_then(|z| z.descend('h'))
            .expect("Should navigate to 'catch'");

        assert_eq!(catch_zipper.value(), Some(2));
    }

    #[test]
    fn test_unicode_valued_zipper() {
        let dict: DynamicDawgChar<String> = DynamicDawgChar::new();
        dict.insert_with_value("café", "coffee".to_string());
        dict.insert_with_value("naïve", "innocent".to_string());

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate to "café"
        let cafe_zipper = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('f'))
            .and_then(|z| z.descend('é'))
            .expect("Should navigate to 'café'");

        assert_eq!(cafe_zipper.value(), Some("coffee".to_string()));
    }

    #[test]
    fn test_clone_independence() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("test");

        let zipper1 = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        // Both zippers should navigate independently
        let z1_t = zipper1.descend('t');
        let z2_t = zipper2.descend('t');

        assert!(z1_t.is_some());
        assert!(z2_t.is_some());
        assert_eq!(z1_t.unwrap().node(), z2_t.unwrap().node());
    }

    #[test]
    fn test_empty_dictionary() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_value_none_for_non_final() {
        let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
        dict.insert_with_value("cat", 42);

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate to 'c' (non-final)
        let c_zipper = zipper.descend('c').expect("Should descend to 'c'");
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
            let d: DynamicDawgChar<()> = DynamicDawgChar::new();
            d.insert("test");
            d.insert("testing");
            d
        });

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let dict_clone = dict.clone();
                thread::spawn(move || {
                    let zipper = DynamicDawgCharZipper::new_from_dict(&dict_clone);
                    zipper.descend('t').is_some()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_path_tracking() {
        let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
        dict.insert("café");

        let zipper = DynamicDawgCharZipper::new_from_dict(&dict);

        // Navigate through "café"
        let c = zipper.descend('c').unwrap();
        assert_eq!(c.path(), vec!['c']);

        let a = c.descend('a').unwrap();
        assert_eq!(a.path(), vec!['c', 'a']);

        let f = a.descend('f').unwrap();
        assert_eq!(f.path(), vec!['c', 'a', 'f']);

        let e = f.descend('é').unwrap();
        assert_eq!(e.path(), vec!['c', 'a', 'f', 'é']);
    }
}
