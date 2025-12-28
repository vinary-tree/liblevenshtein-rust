//! Zipper for navigating DoubleArrayTrie structures.
//!
//! Provides efficient navigation through DoubleArrayTrie with support for
//! accessing values at final states.

use crate::dictionary::double_array_trie::{DATShared, DoubleArrayTrie};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};

/// Zipper for navigating DoubleArrayTrie structures.
///
/// Provides a functional interface for exploring the trie structure while
/// accessing values associated with final states.
///
/// # Type Parameters
///
/// * `V` - The type of values associated with dictionary terms (default: `()`)
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::DoubleArrayTrie;
/// use liblevenshtein::dictionary::double_array_trie_zipper::DoubleArrayTrieZipper;
/// use liblevenshtein::dictionary::zipper::{DictZipper, ValuedDictZipper};
///
/// let dict = DoubleArrayTrie::from_terms_with_values(vec![
///     ("cat", 1),
///     ("catch", 2),
/// ]);
///
/// let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
///
/// // Navigate to "cat"
/// let z = zipper.descend(b'c')
///     .and_then(|z| z.descend(b'a'))
///     .and_then(|z| z.descend(b't'))
///     .unwrap();
///
/// assert!(z.is_final());
/// assert_eq!(z.value(), Some(1));
/// ```
#[derive(Clone, Debug)]
pub struct DoubleArrayTrieZipper<V: DictionaryValue = ()> {
    /// Current state index
    state: usize,

    /// Path from root to current position
    path: Vec<u8>,

    /// Shared DAT data
    shared: DATShared<V>,
}

impl<V: DictionaryValue> DoubleArrayTrieZipper<V> {
    /// Create a new zipper at the root of the dictionary.
    ///
    /// # Arguments
    ///
    /// * `dict` - The DoubleArrayTrie to navigate
    ///
    /// # Returns
    ///
    /// A zipper positioned at the root of the trie.
    pub fn new_from_dict(dict: &DoubleArrayTrie<V>) -> Self {
        Self {
            state: 1, // Root is state 1 in DoubleArrayTrie
            path: Vec::new(),
            shared: dict.shared.clone(),
        }
    }

    /// Get the current state index.
    ///
    /// Useful for debugging or advanced use cases.
    pub fn state(&self) -> usize {
        self.state
    }
}

impl<V: DictionaryValue> DictZipper for DoubleArrayTrieZipper<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        self.state < self.shared.is_final.len() && self.shared.is_final[self.state]
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        if self.state >= self.shared.base.len() {
            return None;
        }

        let base = self.shared.base[self.state];
        if base < 0 {
            return None;
        }

        let next = (base as usize) + (label as usize);
        if next >= self.shared.check.len() || self.shared.check[next] != self.state as i32 {
            return None;
        }

        let mut new_path = self.path.clone();
        new_path.push(label);

        Some(Self {
            state: next,
            path: new_path,
            shared: self.shared.clone(),
        })
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        // Use precomputed edge list for efficiency
        let edges: &[u8] = if self.state < self.shared.edges.len() {
            &self.shared.edges[self.state]
        } else {
            &[]
        };

        edges
            .iter()
            .filter_map(move |&byte| self.descend(byte).map(|child| (byte, child)))
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.path.clone()
    }
}

impl<V: DictionaryValue> ValuedDictZipper for DoubleArrayTrieZipper<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        if self.is_final() && self.state < self.shared.values.len() {
            self.shared.values[self.state].clone()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::double_array_trie::DoubleArrayTrie;

    #[test]
    fn test_zipper_root_not_final() {
        let dict: DoubleArrayTrie<()> = DoubleArrayTrie::from_terms(vec!["test"]);
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.state(), 1); // Root is state 1
    }

    #[test]
    fn test_zipper_navigation() {
        let dict = DoubleArrayTrie::from_terms(vec!["cat", "catch"]);
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let z = zipper
            .descend(b'c')
            .and_then(|z| z.descend(b'a'))
            .and_then(|z| z.descend(b't'))
            .unwrap();

        assert!(z.is_final());

        // Continue to "catch"
        let z = z.descend(b'c').and_then(|z| z.descend(b'h')).unwrap();

        assert!(z.is_final());
    }

    #[test]
    fn test_zipper_nonexistent_path() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        let result = zipper.descend(b'x');
        assert!(result.is_none());
    }

    #[test]
    fn test_zipper_with_values() {
        let dict = DoubleArrayTrie::from_terms_with_values(vec![("cat", 1), ("catch", 2)]);

        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        // Navigate to "cat"
        let z = zipper
            .descend(b'c')
            .and_then(|z| z.descend(b'a'))
            .and_then(|z| z.descend(b't'))
            .unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some(1));

        // Continue to "catch"
        let z = z.descend(b'c').and_then(|z| z.descend(b'h')).unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some(2));
    }

    #[test]
    fn test_zipper_children_iteration() {
        let dict = DoubleArrayTrie::from_terms(vec!["ab", "ac", "ad"]);
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        // Navigate to 'a'
        let a_zipper = zipper.descend(b'a').unwrap();

        let children: Vec<u8> = a_zipper.children().map(|(label, _)| label).collect();

        // Should have edges for 'b', 'c', 'd'
        assert!(children.contains(&b'b'));
        assert!(children.contains(&b'c'));
        assert!(children.contains(&b'd'));
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_zipper_value_at_non_final() {
        let dict = DoubleArrayTrie::from_terms_with_values(vec![("test", 42)]);

        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        // Navigate to "te" (not final)
        let z = zipper.descend(b't').and_then(|z| z.descend(b'e')).unwrap();

        assert!(!z.is_final());
        assert_eq!(z.value(), None); // No value at non-final state
    }

    #[test]
    fn test_zipper_empty_dict() {
        let dict: DoubleArrayTrie<i32> = DoubleArrayTrie::new();
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_zipper_string_values() {
        let dict = DoubleArrayTrie::from_terms_with_values(vec![
            ("hello", "greeting".to_string()),
            ("world", "noun".to_string()),
        ]);

        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        // Navigate to "hello"
        let z = zipper
            .descend(b'h')
            .and_then(|z| z.descend(b'e'))
            .and_then(|z| z.descend(b'l'))
            .and_then(|z| z.descend(b'l'))
            .and_then(|z| z.descend(b'o'))
            .unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some("greeting".to_string()));
    }
}
