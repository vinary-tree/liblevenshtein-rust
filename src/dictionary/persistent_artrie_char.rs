//! Character-level Persistent Adaptive Radix Trie for proper Unicode support.
//!
//! This module provides a character-based variant of PersistentARTrie that operates
//! at the Unicode character level rather than byte level. This ensures correct edit
//! distance semantics for multi-byte UTF-8 sequences.
//!
//! ## Differences from PersistentARTrie
//!
//! - Edge labels are `char` (4 bytes) instead of `u8` (1 byte)
//! - Distance calculations count characters, not bytes
//! - Correct semantics: "" → "¡" is distance 1, not 2
//!
//! ## Performance Trade-offs
//!
//! - **Memory**: Uses char-indexed edges (larger fanout space)
//! - **Speed**: Slightly slower due to UTF-8 encoding/decoding
//! - **Correctness**: Proper Unicode semantics
//!
//! ## Use Cases
//!
//! Use `PersistentARTrieChar` when:
//! - Dictionary contains non-ASCII Unicode characters
//! - Edit distance must be measured in characters, not bytes
//! - Correctness is more important than maximum performance

use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use crate::dictionary::{Dictionary, DictionaryNode, MappedDictionary};
use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "parking_lot")]
use parking_lot::RwLock;
#[cfg(not(feature = "parking_lot"))]
use std::sync::RwLock;

/// Shared inner state for PersistentARTrieChar
#[derive(Debug)]
struct PersistentARTrieCharInner<V: DictionaryValue> {
    /// Root node of the trie
    root: Arc<CharTrieNode<V>>,
    /// Number of terms in the dictionary
    len: usize,
}

/// A character-indexed trie node for Unicode support
#[derive(Debug, Clone)]
struct CharTrieNode<V: DictionaryValue> {
    /// Is this node the end of a complete term?
    is_final: bool,
    /// Children indexed by character
    children: BTreeMap<char, Arc<CharTrieNode<V>>>,
    /// Optional value associated with this node (if final)
    value: Option<V>,
}

impl<V: DictionaryValue> Default for CharTrieNode<V> {
    fn default() -> Self {
        Self {
            is_final: false,
            children: BTreeMap::new(),
            value: None,
        }
    }
}

impl<V: DictionaryValue> CharTrieNode<V> {
    /// Create a new empty node
    fn new() -> Self {
        Self::default()
    }

    /// Create a final node with a value
    fn new_final(value: V) -> Self {
        Self {
            is_final: true,
            children: BTreeMap::new(),
            value: Some(value),
        }
    }

    /// Get a child by character
    fn get_child(&self, c: char) -> Option<&Arc<CharTrieNode<V>>> {
        self.children.get(&c)
    }

    /// Iterate over children
    fn iter_children(&self) -> impl Iterator<Item = (char, &Arc<CharTrieNode<V>>)> {
        self.children.iter().map(|(&c, node)| (c, node))
    }
}

/// Character-level Persistent Adaptive Radix Trie for Unicode support.
///
/// This dictionary provides proper Unicode character-level edit distance
/// calculations, ensuring that multi-byte UTF-8 characters are counted
/// as single edit operations.
#[derive(Debug)]
pub struct PersistentARTrieChar<V: DictionaryValue = ()> {
    inner: Arc<RwLock<PersistentARTrieCharInner<V>>>,
}

impl<V: DictionaryValue> Default for PersistentARTrieChar<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> Clone for PersistentARTrieChar<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V: DictionaryValue> PersistentARTrieChar<V> {
    /// Create a new empty character-level trie
    pub fn new() -> Self {
        let inner = PersistentARTrieCharInner {
            root: Arc::new(CharTrieNode::new()),
            len: 0,
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Insert a term into the trie
    pub fn insert(&self, term: &str) -> bool
    where
        V: Default,
    {
        self.insert_with_value(term, V::default())
    }

    /// Insert a term with an associated value
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        #[cfg(feature = "parking_lot")]
        let mut guard = self.inner.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut guard = self.inner.write().expect("lock poisoned");

        // Navigate to the insertion point, creating nodes as needed
        let chars: Vec<char> = term.chars().collect();
        let mut current = Arc::clone(&guard.root);
        let mut path: Vec<(char, Arc<CharTrieNode<V>>)> = Vec::new();

        for &c in &chars {
            let next = current.get_child(c).cloned();
            path.push((c, Arc::clone(&current)));
            current = match next {
                Some(node) => node,
                None => Arc::new(CharTrieNode::new()),
            };
        }

        // Check if already exists
        if current.is_final {
            return false;
        }

        // Build the new path from bottom up
        let mut new_node = CharTrieNode {
            is_final: true,
            children: current.children.clone(),
            value: Some(value),
        };

        for (c, parent) in path.into_iter().rev() {
            let mut new_parent = CharTrieNode {
                is_final: parent.is_final,
                children: parent.children.clone(),
                value: parent.value.clone(),
            };
            new_parent.children.insert(c, Arc::new(new_node));
            new_node = new_parent;
        }

        guard.root = Arc::new(new_node);
        guard.len += 1;
        true
    }

    /// Check if a term exists in the trie
    pub fn contains(&self, term: &str) -> bool {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");

        let mut current = Arc::clone(&guard.root);
        for c in term.chars() {
            match current.get_child(c) {
                Some(child) => current = Arc::clone(child),
                None => return false,
            }
        }
        current.is_final
    }

    /// Get the number of terms in the dictionary
    pub fn len(&self) -> usize {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");
        guard.len
    }

    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the root node
    pub fn root(&self) -> PersistentARTrieCharNode<V> {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");

        PersistentARTrieCharNode {
            node: Arc::clone(&guard.root),
        }
    }
}

/// Build from an iterator of terms
impl<V: DictionaryValue + Default> FromIterator<String> for PersistentARTrieChar<V> {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let trie = Self::new();
        for term in iter {
            trie.insert(&term);
        }
        trie
    }
}

impl<'a, V: DictionaryValue + Default> FromIterator<&'a str> for PersistentARTrieChar<V> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let trie = Self::new();
        for term in iter {
            trie.insert(term);
        }
        trie
    }
}

/// Node in the character-level trie for DictionaryNode trait
#[derive(Debug, Clone)]
pub struct PersistentARTrieCharNode<V: DictionaryValue = ()> {
    node: Arc<CharTrieNode<V>>,
}

impl<V: DictionaryValue> DictionaryNode for PersistentARTrieCharNode<V> {
    type Unit = char;

    fn is_final(&self) -> bool {
        self.node.is_final
    }

    fn transition(&self, label: char) -> Option<Self> {
        self.node.get_child(label).map(|child| Self {
            node: Arc::clone(child),
        })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
        let edges: Vec<_> = self
            .node
            .iter_children()
            .map(|(c, child)| {
                (
                    c,
                    Self {
                        node: Arc::clone(child),
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }
}

impl<V: DictionaryValue> Dictionary for PersistentARTrieChar<V> {
    type Node = PersistentARTrieCharNode<V>;

    fn root(&self) -> Self::Node {
        PersistentARTrieChar::root(self)
    }

    fn contains(&self, term: &str) -> bool {
        PersistentARTrieChar::contains(self, term)
    }

    fn len(&self) -> Option<usize> {
        Some(PersistentARTrieChar::len(self))
    }
}

impl<V: DictionaryValue> MappedDictionary for PersistentARTrieChar<V> {
    type Value = V;

    fn get_value(&self, term: &str) -> Option<V> {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");

        let mut current = Arc::clone(&guard.root);
        for c in term.chars() {
            match current.get_child(c) {
                Some(child) => current = Arc::clone(child),
                None => return None,
            }
        }
        if current.is_final {
            current.value.clone()
        } else {
            None
        }
    }
}

/// Iterator over terms in the trie
pub struct PersistentARTrieCharIterator<V: DictionaryValue> {
    stack: Vec<(String, Arc<CharTrieNode<V>>)>,
}

impl<V: DictionaryValue> Iterator for PersistentARTrieCharIterator<V> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((prefix, node)) = self.stack.pop() {
            // Push children in reverse order for correct alphabetical order
            let children: Vec<_> = node.iter_children().collect();
            for (c, child) in children.into_iter().rev() {
                let mut new_prefix = prefix.clone();
                new_prefix.push(c);
                self.stack.push((new_prefix, Arc::clone(child)));
            }

            if node.is_final {
                return Some(prefix);
            }
        }
        None
    }
}

impl<V: DictionaryValue> IntoIterator for &PersistentARTrieChar<V> {
    type Item = String;
    type IntoIter = PersistentARTrieCharIterator<V>;

    fn into_iter(self) -> Self::IntoIter {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");

        PersistentARTrieCharIterator {
            stack: vec![(String::new(), Arc::clone(&guard.root))],
        }
    }
}

impl<V: DictionaryValue> PersistentARTrieChar<V> {
    /// Iterate over all terms in the dictionary
    pub fn iter(&self) -> PersistentARTrieCharIterator<V> {
        self.into_iter()
    }
}

/// Zipper for navigating the character-level trie
#[derive(Debug, Clone)]
pub struct PersistentARTrieCharZipper<V: DictionaryValue = ()> {
    node: PersistentARTrieCharNode<V>,
    path_vec: Vec<char>,
}

impl<V: DictionaryValue> PersistentARTrieCharZipper<V> {
    /// Create a new zipper at the root
    pub fn new(dict: &PersistentARTrieChar<V>) -> Self {
        Self {
            node: dict.root(),
            path_vec: Vec::new(),
        }
    }

    /// Get the current path as a string
    pub fn path_string(&self) -> String {
        self.path_vec.iter().collect()
    }
}

impl<V: DictionaryValue> DictZipper for PersistentARTrieCharZipper<V> {
    type Unit = char;

    fn is_final(&self) -> bool {
        self.node.is_final()
    }

    fn descend(&self, label: char) -> Option<Self> {
        self.node.transition(label).map(|child| {
            let mut new_path = self.path_vec.clone();
            new_path.push(label);
            Self {
                node: child,
                path_vec: new_path,
            }
        })
    }

    fn children(&self) -> impl Iterator<Item = (char, Self)> {
        let path = self.path_vec.clone();
        self.node.node.children.iter().map(move |(&c, child)| {
            let mut new_path = path.clone();
            new_path.push(c);
            (
                c,
                Self {
                    node: PersistentARTrieCharNode {
                        node: Arc::clone(child),
                    },
                    path_vec: new_path,
                },
            )
        })
    }

    fn path(&self) -> Vec<char> {
        self.path_vec.clone()
    }
}

impl<V: DictionaryValue> ValuedDictZipper for PersistentARTrieCharZipper<V> {
    type Value = V;

    fn value(&self) -> Option<V> {
        if self.node.is_final() {
            self.node.node.value.clone()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_insert_ascii() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        assert!(trie.insert("hello"));
        assert!(trie.insert("world"));
        assert!(!trie.insert("hello")); // Duplicate
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_insert_unicode() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        assert!(trie.insert("héllo")); // é is one character
        assert!(trie.insert("日本語")); // Japanese characters
        assert!(trie.insert("emoji😀")); // Emoji
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_contains() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("hello");
        trie.insert("héllo");

        assert!(trie.contains("hello"));
        assert!(trie.contains("héllo"));
        assert!(!trie.contains("helo"));
        assert!(!trie.contains("hello ")); // Trailing space
    }

    #[test]
    fn test_edges_unicode() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("日本");
        trie.insert("日曜日");

        let root = trie.root();
        let edges: Vec<_> = root.edges().collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, '日');
    }

    #[test]
    fn test_transition() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("café");

        let mut node = trie.root();
        assert!(node.transition('c').is_some());
        node = node.transition('c').unwrap();
        assert!(node.transition('a').is_some());
        node = node.transition('a').unwrap();
        assert!(node.transition('f').is_some());
        node = node.transition('f').unwrap();
        assert!(node.transition('é').is_some());
        node = node.transition('é').unwrap();
        assert!(node.is_final());
    }

    #[test]
    fn test_iterator() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("a");
        trie.insert("ab");
        trie.insert("abc");

        let terms: Vec<_> = trie.iter().collect();
        assert_eq!(terms.len(), 3);
        assert!(terms.contains(&"a".to_string()));
        assert!(terms.contains(&"ab".to_string()));
        assert!(terms.contains(&"abc".to_string()));
    }

    #[test]
    fn test_zipper() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("hello");
        trie.insert("help");

        let zipper = PersistentARTrieCharZipper::new(&trie);
        let zipper = zipper.descend('h').expect("should have 'h'");
        let zipper = zipper.descend('e').expect("should have 'e'");
        let zipper = zipper.descend('l').expect("should have 'l'");

        let edges: Vec<_> = zipper.children().map(|(c, _)| c).collect();
        assert_eq!(edges.len(), 2); // 'l' and 'p'
    }

    #[test]
    fn test_from_iter() {
        let terms = vec!["alpha", "beta", "gamma"];
        let trie: PersistentARTrieChar<()> = terms.into_iter().collect();
        assert_eq!(trie.len(), 3);
        assert!(trie.contains("alpha"));
        assert!(trie.contains("beta"));
        assert!(trie.contains("gamma"));
    }

    #[test]
    fn test_value_storage() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
        trie.insert_with_value("one", 1);
        trie.insert_with_value("two", 2);
        trie.insert_with_value("three", 3);

        assert_eq!(trie.get_value("one"), Some(1));
        assert_eq!(trie.get_value("two"), Some(2));
        assert_eq!(trie.get_value("three"), Some(3));
        assert_eq!(trie.get_value("four"), None);
    }

    #[test]
    fn test_unicode_correctness() {
        // This test verifies that multi-byte characters are treated as single units
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("¡");

        let root = trie.root();
        // Should have exactly one edge (for '¡'), not two edges (for the bytes)
        let edges: Vec<_> = root.edges().collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, '¡');
    }
}
