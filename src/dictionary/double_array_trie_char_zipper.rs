//! Zipper for navigating DoubleArrayTrieChar structures.
//!
//! Provides efficient navigation through DoubleArrayTrieChar with support for
//! accessing values at final states and proper Unicode character handling.

use crate::dictionary::double_array_trie_char::{DATSharedChar, DoubleArrayTrieChar};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};

/// Zipper for navigating DoubleArrayTrieChar structures.
///
/// Provides a functional interface for exploring the trie structure at the
/// Unicode character level while accessing values associated with final states.
///
/// # Type Parameters
///
/// * `V` - The type of values associated with dictionary terms (default: `()`)
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::DoubleArrayTrieChar;
/// use liblevenshtein::dictionary::double_array_trie_char_zipper::DoubleArrayTrieCharZipper;
/// use liblevenshtein::dictionary::zipper::{DictZipper, ValuedDictZipper};
///
/// let dict = DoubleArrayTrieChar::from_terms_with_values(vec![
///     ("café", 1),
///     ("中文", 2),
/// ]);
///
/// let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);
///
/// // Navigate to "café"
/// let z = zipper.descend('c')
///     .and_then(|z| z.descend('a'))
///     .and_then(|z| z.descend('f'))
///     .and_then(|z| z.descend('é'))
///     .unwrap();
///
/// assert!(z.is_final());
/// assert_eq!(z.value(), Some(1));
/// ```
#[derive(Clone, Debug)]
pub struct DoubleArrayTrieCharZipper<V: DictionaryValue = ()> {
    /// Current state index
    state: usize,

    /// Path from root to current position
    path: Vec<char>,

    /// Shared DAT data
    shared: DATSharedChar<V>,
}

impl<V: DictionaryValue> DoubleArrayTrieCharZipper<V> {
    /// Create a new zipper at the root of the dictionary.
    ///
    /// # Arguments
    ///
    /// * `dict` - The DoubleArrayTrieChar to navigate
    ///
    /// # Returns
    ///
    /// A zipper positioned at the root of the trie.
    pub fn new_from_dict(dict: &DoubleArrayTrieChar<V>) -> Self {
        Self {
            state: 0, // Root is state 0 in DoubleArrayTrieChar
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

impl<V: DictionaryValue> DictZipper for DoubleArrayTrieCharZipper<V> {
    type Unit = char;

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

        let char_code = label as u32;
        let next = (base as u32).wrapping_add(char_code) as usize;

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
        let edges: &[char] = if self.state < self.shared.edges.len() {
            &self.shared.edges[self.state]
        } else {
            &[]
        };

        edges
            .iter()
            .filter_map(move |&c| self.descend(c).map(|child| (c, child)))
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.path.clone()
    }
}

impl<V: DictionaryValue> ValuedDictZipper for DoubleArrayTrieCharZipper<V> {
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
    use crate::dictionary::double_array_trie_char::DoubleArrayTrieChar;

    #[test]
    fn test_zipper_root_not_final() {
        let dict: DoubleArrayTrieChar<()> = DoubleArrayTrieChar::from_terms(vec!["test"]);
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.state(), 0); // Root is state 0
    }

    #[test]
    fn test_zipper_unicode_navigation() {
        let dict = DoubleArrayTrieChar::from_terms(vec!["café", "中文"]);
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to "café"
        let z = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('f'))
            .and_then(|z| z.descend('é'))
            .unwrap();

        assert!(z.is_final());

        // Navigate to "中文"
        let z = zipper.descend('中').and_then(|z| z.descend('文')).unwrap();

        assert!(z.is_final());
    }

    #[test]
    fn test_zipper_nonexistent_path() {
        let dict = DoubleArrayTrieChar::from_terms(vec!["test"]);
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        let result = zipper.descend('x');
        assert!(result.is_none());
    }

    #[test]
    fn test_zipper_with_unicode_values() {
        let dict = DoubleArrayTrieChar::from_terms_with_values(vec![("café", 1), ("中文", 2)]);

        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to "café"
        let z = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('f'))
            .and_then(|z| z.descend('é'))
            .unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some(1));

        // Navigate to "中文"
        let z = zipper.descend('中').and_then(|z| z.descend('文')).unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some(2));
    }

    #[test]
    fn test_zipper_children_iteration() {
        let dict = DoubleArrayTrieChar::from_terms(vec!["ab", "ac", "ad"]);
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to 'a'
        let a_zipper = zipper.descend('a').unwrap();

        let children: Vec<char> = a_zipper.children().map(|(label, _)| label).collect();

        // Should have edges for 'b', 'c', 'd'
        assert!(children.contains(&'b'));
        assert!(children.contains(&'c'));
        assert!(children.contains(&'d'));
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_zipper_value_at_non_final() {
        let dict = DoubleArrayTrieChar::from_terms_with_values(vec![("test", 42)]);

        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to "te" (not final)
        let z = zipper.descend('t').and_then(|z| z.descend('e')).unwrap();

        assert!(!z.is_final());
        assert_eq!(z.value(), None); // No value at non-final state
    }

    #[test]
    fn test_zipper_empty_dict() {
        let dict = DoubleArrayTrieChar::<i32>::from_terms_with_values(Vec::<(&str, i32)>::new());
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_zipper_string_values() {
        let dict = DoubleArrayTrieChar::from_terms_with_values(vec![
            ("hello", "greeting".to_string()),
            ("世界", "world".to_string()),
        ]);

        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to "hello"
        let z = zipper
            .descend('h')
            .and_then(|z| z.descend('e'))
            .and_then(|z| z.descend('l'))
            .and_then(|z| z.descend('l'))
            .and_then(|z| z.descend('o'))
            .unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some("greeting".to_string()));
    }

    #[test]
    fn test_zipper_emoji() {
        let dict = DoubleArrayTrieChar::from_terms_with_values(vec![
            ("🎉", "party".to_string()),
            ("🌍", "earth".to_string()),
        ]);

        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        // Navigate to "🎉"
        let z = zipper.descend('🎉').unwrap();

        assert!(z.is_final());
        assert_eq!(z.value(), Some("party".to_string()));
    }

    #[test]
    fn test_zipper_path_tracking() {
        let dict = DoubleArrayTrieChar::from_terms(vec!["café"]);
        let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

        let z = zipper
            .descend('c')
            .and_then(|z| z.descend('a'))
            .and_then(|z| z.descend('f'))
            .and_then(|z| z.descend('é'))
            .unwrap();

        let path = z.path();
        assert_eq!(path, vec!['c', 'a', 'f', 'é']);
    }
}
