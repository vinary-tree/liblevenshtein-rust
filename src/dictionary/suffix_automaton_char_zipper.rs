//! Suffix automaton character-level zipper implementation.
//!
//! This module provides a zipper implementation for SuffixAutomatonChar that uses
//! state-index-based navigation with lock-per-operation pattern for thread safety.
//! Unlike SuffixAutomatonZipper which operates on bytes, this operates on Unicode
//! characters for correct multi-byte UTF-8 handling.

use crate::dictionary::suffix_automaton_char::{SuffixAutomatonChar, SuffixAutomatonCharInner};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use std::sync::Arc;

use crate::sync_compat::RwLock;

/// Zipper for Suffix Automaton dictionaries.
///
/// `SuffixAutomatonCharZipper` provides efficient navigation through Suffix Automaton structures
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
/// use liblevenshtein::dictionary::suffix_automaton_char::SuffixAutomatonChar;
/// use liblevenshtein::dictionary::suffix_automaton_char_zipper::SuffixAutomatonCharZipper;
///
/// let dict = SuffixAutomatonChar::<()>::from_text("testing");
///
/// let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);
///
/// // Navigate through "test" (a substring of "testing")
/// if let Some(t) = zipper.descend('t') {
///     if let Some(e) = t.descend('e') {
///         if let Some(s) = e.descend('s') {
///             if let Some(t2) = s.descend('t') {
///                 if t2.is_final() {
///                     println!("Found 'test'");
///                 }
///             }
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct SuffixAutomatonCharZipper<V: DictionaryValue = ()> {
    /// Shared reference to automaton inner structure
    inner: Arc<RwLock<SuffixAutomatonCharInner<V>>>,

    /// Current state index (0 is root)
    state_id: usize,

    /// Path from root to current position
    path: Vec<char>,
}

impl<V: DictionaryValue> SuffixAutomatonCharZipper<V> {
    /// Create a new zipper at the root of the Suffix Automaton.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the SuffixAutomatonChar dictionary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::suffix_automaton_char::SuffixAutomatonChar;
    /// use liblevenshtein::dictionary::suffix_automaton_char_zipper::SuffixAutomatonCharZipper;
    ///
    /// let dict = SuffixAutomatonChar::<()>::from_text("example");
    /// let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &SuffixAutomatonChar<V>) -> Self {
        SuffixAutomatonCharZipper {
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

impl<V: DictionaryValue> DictZipper for SuffixAutomatonCharZipper<V> {
    type Unit = char;

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
                return Some(SuffixAutomatonCharZipper {
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
        let edges: Vec<(char, usize)> = {
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
                SuffixAutomatonCharZipper {
                    inner: inner.clone(),
                    state_id: target,
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> ValuedDictZipper for SuffixAutomatonCharZipper<V> {
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
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
        assert_eq!(zipper.state_id(), 0);
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);
        // 'x' doesn't exist in "test"
        assert!(zipper.descend('x').is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate to "test"
        let t = zipper.descend('t').expect("Should descend to 't'");
        assert!(t.is_final(), "'t' should be final (it's a suffix)");

        let e = t.descend('e').expect("Should descend to 'e'");
        assert!(e.is_final(), "'te' should be final (it's a suffix)");

        let s = e.descend('s').expect("Should descend to 's'");
        assert!(s.is_final(), "'tes' should be final (it's a suffix)");

        let t2 = s.descend('t').expect("Should descend to 't'");
        assert!(t2.is_final(), "'test' should be final (it's a suffix)");
    }

    #[test]
    fn test_substring_matching() {
        let dict = SuffixAutomatonChar::<()>::from_text("testing");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate to "est" (substring in the middle)
        let e = zipper.descend('e').expect("Should descend to 'e'");
        let s = e.descend('s').expect("Should descend to 's'");
        let t = s.descend('t').expect("Should descend to 't'");

        // "est" is a substring of "testing"
        assert!(t.is_final(), "'est' should be final");
    }

    #[test]
    fn test_children_iteration() {
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Root should have children for each unique first character of suffixes
        let children: Vec<char> = zipper.children().map(|(label, _)| label).collect();

        // Suffixes: "test", "est", "st", "t"
        // First chars: 't', 'e', 's', 't'
        // Unique: 't', 'e', 's'
        assert!(children.contains(&'t'));
        assert!(children.contains(&'e'));
        assert!(children.contains(&'s'));
    }

    #[test]
    fn test_valued_zipper() {
        use crate::dictionary::MutableMappedDictionary;

        let dict: SuffixAutomatonChar<u32> = SuffixAutomatonChar::new();
        dict.insert_with_value("test", 1);
        dict.insert_with_value("testing", 2);

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate to "test"
        let test_zipper = zipper
            .descend('t')
            .and_then(|z| z.descend('e'))
            .and_then(|z| z.descend('s'))
            .and_then(|z| z.descend('t'))
            .expect("Should navigate to 'test'");

        // The value should be associated with the final state
        // Note: In suffix automaton, multiple strings may share states
        assert!(test_zipper.is_final());
    }

    #[test]
    fn test_clone_independence() {
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper1 = SuffixAutomatonCharZipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        // Both zippers should navigate independently
        let z1_t = zipper1.descend('t');
        let z2_t = zipper2.descend('t');

        assert!(z1_t.is_some());
        assert!(z2_t.is_some());
        assert_eq!(z1_t.unwrap().state_id(), z2_t.unwrap().state_id());
    }

    #[test]
    fn test_empty_dictionary() {
        let dict: SuffixAutomatonChar<()> = SuffixAutomatonChar::new();
        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_value_none_for_non_final() {
        use crate::dictionary::MutableMappedDictionary;

        let dict: SuffixAutomatonChar<u32> = SuffixAutomatonChar::new();
        dict.insert_with_value("test", 42);

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Root is non-final
        assert_eq!(zipper.value(), None, "Root should have no value");
    }

    #[test]
    fn test_path_tracking() {
        let dict = SuffixAutomatonChar::<()>::from_text("test");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate through "test"
        let t = zipper.descend('t').unwrap();
        assert_eq!(t.path(), vec!['t']);

        let e = t.descend('e').unwrap();
        assert_eq!(e.path(), vec!['t', 'e']);

        let s = e.descend('s').unwrap();
        assert_eq!(s.path(), vec!['t', 'e', 's']);

        let t2 = s.descend('t').unwrap();
        assert_eq!(t2.path(), vec!['t', 'e', 's', 't']);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dict = StdArc::new(SuffixAutomatonChar::<()>::from_text("testing"));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let dict_clone = dict.clone();
                thread::spawn(move || {
                    let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict_clone);
                    zipper.descend('t').is_some()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_unicode_navigation() {
        let dict = SuffixAutomatonChar::<()>::from_text("café");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate to "café" (4 characters, 5 bytes)
        let c = zipper.descend('c').expect("Should descend to 'c'");
        assert!(c.is_final());

        let a = c.descend('a').expect("Should descend to 'a'");
        assert!(a.is_final());

        let f = a.descend('f').expect("Should descend to 'f'");
        assert!(f.is_final());

        let e = f.descend('é').expect("Should descend to 'é'");
        assert!(e.is_final(), "'café' should be final");

        // Verify path
        assert_eq!(e.path(), vec!['c', 'a', 'f', 'é']);
    }

    #[test]
    fn test_emoji_navigation() {
        let dict = SuffixAutomatonChar::<()>::from_text("test🎉ing");

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Navigate to emoji
        let t = zipper.descend('t').expect("Should descend to 't'");
        let e = t.descend('e').expect("Should descend to 'e'");
        let s = e.descend('s').expect("Should descend to 's'");
        let t2 = s.descend('t').expect("Should descend to 't'");
        let emoji = t2.descend('🎉').expect("Should descend to emoji");

        assert!(emoji.is_final());
        assert_eq!(emoji.path(), vec!['t', 'e', 's', 't', '🎉']);
    }

    #[test]
    fn test_multiple_strings() {
        let dict = SuffixAutomatonChar::<()>::from_texts(vec!["cat", "car", "dog"]);

        let zipper = SuffixAutomatonCharZipper::new_from_dict(&dict);

        // Root should have children for all unique first characters
        let children: Vec<char> = zipper.children().map(|(label, _)| label).collect();

        // Suffixes include: "cat", "at", "t", "car", "ar", "r", "dog", "og", "g"
        // First chars: 'c', 'a', 't', 'c', 'a', 'r', 'd', 'o', 'g'
        // Unique: 'c', 'a', 't', 'r', 'd', 'o', 'g'
        assert!(children.contains(&'c'));
        assert!(children.contains(&'d'));
    }
}
