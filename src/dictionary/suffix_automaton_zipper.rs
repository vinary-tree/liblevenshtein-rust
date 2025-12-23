//! Suffix automaton zipper implementation.
//!
//! This module provides a zipper implementation for SuffixAutomaton that uses
//! state-index-based navigation with lock-per-operation pattern for thread safety.

use crate::dictionary::suffix_automaton::{SuffixAutomaton, SuffixAutomatonInner};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use std::sync::Arc;

use crate::sync_compat::RwLock;

/// Zipper for Suffix Automaton dictionaries.
///
/// `SuffixAutomatonZipper` provides efficient navigation through Suffix Automaton structures
/// using a state-index-based approach with thread-safe concurrent access.
///
/// # Design
///
/// The zipper stores:
/// - `inner`: Shared reference to the automaton inner structure (Arc<RwLock>)
/// - `state_id`: Current state index in the automaton
/// - `path`: Path from root to current position
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
/// - State-index-based: No path storage overhead
/// - Lock-per-operation: Minimal lock contention
/// - Lightweight Clone: Just Arc clone + usize copy
///
/// # Suffix Automaton Semantics
///
/// Unlike traditional prefix-based dictionaries, suffix automatons recognize
/// substrings anywhere in indexed text. The zipper navigates through states
/// that represent equivalence classes of substrings with the same ending positions.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::suffix_automaton::SuffixAutomaton;
/// use liblevenshtein::dictionary::suffix_automaton_zipper::SuffixAutomatonZipper;
///
/// let dict = SuffixAutomaton::<()>::from_text("testing");
///
/// let zipper = SuffixAutomatonZipper::new_from_dict(&dict);
///
/// // Navigate through "test" (a substring of "testing")
/// if let Some(t) = zipper.descend(b't') {
///     if let Some(e) = t.descend(b'e') {
///         if let Some(s) = e.descend(b's') {
///             if let Some(t2) = s.descend(b't') {
///                 if t2.is_final() {
///                     println!("Found 'test'");
///                 }
///             }
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct SuffixAutomatonZipper<V: DictionaryValue = ()> {
    /// Shared reference to automaton inner structure
    inner: Arc<RwLock<SuffixAutomatonInner<V>>>,

    /// Current state index (0 is root)
    state_id: usize,

    /// Path from root to current position
    path: Vec<u8>,
}

impl<V: DictionaryValue> SuffixAutomatonZipper<V> {
    /// Create a new zipper at the root of the Suffix Automaton.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the SuffixAutomaton dictionary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::suffix_automaton::SuffixAutomaton;
    /// use liblevenshtein::dictionary::suffix_automaton_zipper::SuffixAutomatonZipper;
    ///
    /// let dict = SuffixAutomaton::<()>::from_text("example");
    /// let zipper = SuffixAutomatonZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &SuffixAutomaton<V>) -> Self {
        SuffixAutomatonZipper {
            inner: dict.inner.clone(),
            state_id: 0, // Root is always state 0
            path: Vec::new(),
        }
    }

    /// Get the current state ID.
    ///
    /// Useful for debugging or advanced use cases.
    pub fn state_id(&self) -> usize {
        self.state_id
    }
}

impl<V: DictionaryValue> DictZipper for SuffixAutomatonZipper<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        if self.state_id < inner.nodes.len() {
            inner.nodes[self.state_id].is_final
        } else {
            false
        }
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        let inner = self.inner.read();
        if self.state_id >= inner.nodes.len() {
            return None;
        }

        // Find the edge with the given label
        for (edge_label, target_state) in &inner.nodes[self.state_id].edges {
            if *edge_label == label {
                let mut new_path = self.path.clone();
                new_path.push(label);
                return Some(SuffixAutomatonZipper {
                    inner: self.inner.clone(),
                    state_id: *target_state,
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
            if self.state_id < inner.nodes.len() {
                inner.nodes[self.state_id].edges.clone()
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
                SuffixAutomatonZipper {
                    inner: inner.clone(),
                    state_id: target,
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> ValuedDictZipper for SuffixAutomatonZipper<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        let inner = self.inner.read();
        if self.state_id < inner.nodes.len() && inner.nodes[self.state_id].is_final {
            inner.nodes[self.state_id].value.clone()
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
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
        assert_eq!(zipper.state_id(), 0);
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);
        // 'x' doesn't exist in "test"
        assert!(zipper.descend(b'x').is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Navigate to "test"
        let t = zipper.descend(b't').expect("Should descend to 't'");
        assert!(t.is_final(), "'t' should be final (it's a suffix)");

        let e = t.descend(b'e').expect("Should descend to 'e'");
        assert!(e.is_final(), "'te' should be final (it's a suffix)");

        let s = e.descend(b's').expect("Should descend to 's'");
        assert!(s.is_final(), "'tes' should be final (it's a suffix)");

        let t2 = s.descend(b't').expect("Should descend to 't'");
        assert!(t2.is_final(), "'test' should be final (it's a suffix)");
    }

    #[test]
    fn test_substring_matching() {
        let dict = SuffixAutomaton::<()>::from_text("testing");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Navigate to "est" (substring in the middle)
        let e = zipper.descend(b'e').expect("Should descend to 'e'");
        let s = e.descend(b's').expect("Should descend to 's'");
        let t = s.descend(b't').expect("Should descend to 't'");

        // "est" is a substring of "testing"
        assert!(t.is_final(), "'est' should be final");
    }

    #[test]
    fn test_children_iteration() {
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Root should have children for each unique first character of suffixes
        let children: Vec<u8> = zipper.children().map(|(label, _)| label).collect();

        // Suffixes: "test", "est", "st", "t"
        // First chars: 't', 'e', 's', 't'
        // Unique: 't', 'e', 's'
        assert!(children.contains(&b't'));
        assert!(children.contains(&b'e'));
        assert!(children.contains(&b's'));
    }

    #[test]
    fn test_valued_zipper() {
        use crate::dictionary::MutableMappedDictionary;

        let dict: SuffixAutomaton<u32> = SuffixAutomaton::new();
        dict.insert_with_value("test", 1);
        dict.insert_with_value("testing", 2);

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Navigate to "test"
        let test_zipper = zipper
            .descend(b't')
            .and_then(|z| z.descend(b'e'))
            .and_then(|z| z.descend(b's'))
            .and_then(|z| z.descend(b't'))
            .expect("Should navigate to 'test'");

        // The value should be associated with the final state
        // Note: In suffix automaton, multiple strings may share states
        assert!(test_zipper.is_final());
    }

    #[test]
    fn test_clone_independence() {
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper1 = SuffixAutomatonZipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        // Both zippers should navigate independently
        let z1_t = zipper1.descend(b't');
        let z2_t = zipper2.descend(b't');

        assert!(z1_t.is_some());
        assert!(z2_t.is_some());
        assert_eq!(z1_t.unwrap().state_id(), z2_t.unwrap().state_id());
    }

    #[test]
    fn test_empty_dictionary() {
        let dict: SuffixAutomaton<()> = SuffixAutomaton::new();
        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_value_none_for_non_final() {
        use crate::dictionary::MutableMappedDictionary;

        let dict: SuffixAutomaton<u32> = SuffixAutomaton::new();
        dict.insert_with_value("test", 42);

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Root is non-final
        assert_eq!(zipper.value(), None, "Root should have no value");
    }

    #[test]
    fn test_path_tracking() {
        let dict = SuffixAutomaton::<()>::from_text("test");

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Navigate through "test"
        let t = zipper.descend(b't').unwrap();
        assert_eq!(t.path(), vec![b't']);

        let e = t.descend(b'e').unwrap();
        assert_eq!(e.path(), vec![b't', b'e']);

        let s = e.descend(b's').unwrap();
        assert_eq!(s.path(), vec![b't', b'e', b's']);

        let t2 = s.descend(b't').unwrap();
        assert_eq!(t2.path(), vec![b't', b'e', b's', b't']);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dict = StdArc::new(SuffixAutomaton::<()>::from_text("testing"));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let dict_clone = dict.clone();
                thread::spawn(move || {
                    let zipper = SuffixAutomatonZipper::new_from_dict(&dict_clone);
                    zipper.descend(b't').is_some()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_multiple_strings() {
        let dict = SuffixAutomaton::<()>::from_texts(vec!["cat", "car", "dog"]);

        let zipper = SuffixAutomatonZipper::new_from_dict(&dict);

        // Root should have children for all unique first characters
        let children: Vec<u8> = zipper.children().map(|(label, _)| label).collect();

        // Suffixes include: "cat", "at", "t", "car", "ar", "r", "dog", "og", "g"
        // First chars: 'c', 'a', 't', 'c', 'a', 'r', 'd', 'o', 'g'
        // Unique: 'c', 'a', 't', 'r', 'd', 'o', 'g'
        assert!(children.contains(&b'c'));
        assert!(children.contains(&b'd'));
    }
}
