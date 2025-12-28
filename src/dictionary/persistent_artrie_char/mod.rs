//! Character-level Persistent Adaptive Radix Trie for proper Unicode support.
//!
//! This module provides a character-based variant of PersistentARTrie that operates
//! at the Unicode character level rather than byte level. This ensures correct edit
//! distance semantics for multi-byte UTF-8 sequences.
//!
//! ## Module Structure
//!
//! - `nodes`: ART node types adapted for u32/char keys (CharNode4, CharNode16, CharNode48, CharBucket)
//!
//! ## Differences from PersistentARTrie
//!
//! - Edge labels are `char` (4 bytes) instead of `u8` (1 byte)
//! - Distance calculations count characters, not bytes
//! - Correct semantics: "" → "¡" is distance 1, not 2
//! - No Node256: Would require 4GB array for u32 keys
//! - CharBucket handles >48 children using HashMap
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

// ART node types for char keys
pub mod nodes;

// Serialization for char nodes
pub mod serialization_char;

// Disk-backed implementation
#[cfg(feature = "persistent-artrie")]
pub mod dict_impl_char;

// Re-export disk-backed types
#[cfg(feature = "persistent-artrie")]
pub use dict_impl_char::DiskBackedCharTrieInner;

// Re-export node types
pub use nodes::{
    AddChildError, CharArtNode, CharBucket, CharCompressedPrefix, CharNode, CharNode16, CharNode4,
    CharNode48, CharNodeHeader, CHAR_MAX_PREFIX_LEN,
};

// Re-export serialization
pub use serialization_char::{
    char_from_bytes, char_serialized_size, char_to_bytes, deserialize_char_node,
    serialize_char_node, CHAR_FORMAT_VERSION, CHAR_NODE_MAGIC, CHAR_SERIALIZED_HEADER_SIZE,
    SerializedCharNodeHeader,
};

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
    #[allow(dead_code)]
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

/// Iterator over terms and values in the trie
pub struct PersistentARTrieCharValueIterator<V: DictionaryValue> {
    stack: Vec<(String, Arc<CharTrieNode<V>>)>,
}

impl<V: DictionaryValue> Iterator for PersistentARTrieCharValueIterator<V> {
    type Item = (String, V);

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
                if let Some(value) = node.value.clone() {
                    return Some((prefix, value));
                }
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

    /// Iterate over all terms and their values in the dictionary
    ///
    /// Returns an iterator that yields `(String, V)` pairs for each term
    /// in the dictionary.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
    /// trie.insert_with_value("hello", 1);
    /// trie.insert_with_value("world", 2);
    ///
    /// for (term, value) in trie.iter_with_values() {
    ///     println!("{}: {}", term, value);
    /// }
    /// ```
    pub fn iter_with_values(&self) -> PersistentARTrieCharValueIterator<V> {
        #[cfg(feature = "parking_lot")]
        let guard = self.inner.read();
        #[cfg(not(feature = "parking_lot"))]
        let guard = self.inner.read().expect("lock poisoned");

        PersistentARTrieCharValueIterator {
            stack: vec![(String::new(), Arc::clone(&guard.root))],
        }
    }

    /// Iterate over all terms as character vectors
    ///
    /// This is useful when you need character-level access to terms
    /// rather than string representations.
    pub fn iter_chars(&self) -> impl Iterator<Item = Vec<char>> + '_ {
        self.iter().map(|s| s.chars().collect())
    }

    /// Iterate over all terms and values as character vectors
    ///
    /// Returns `(Vec<char>, V)` pairs for character-level processing.
    pub fn iter_chars_with_values(&self) -> impl Iterator<Item = (Vec<char>, V)> + '_ {
        self.iter_with_values()
            .map(|(s, v)| (s.chars().collect(), v))
    }
}

// === Atomic Operations (for serde-capable values) ===

impl<V: DictionaryValue + serde::Serialize + serde::de::DeserializeOwned> PersistentARTrieChar<V> {
    /// Atomically increment a numeric value.
    ///
    /// If the term doesn't exist, it's created with the delta as its initial value.
    /// The value must be interpretable as i64.
    ///
    /// # Arguments
    ///
    /// * `term` - The term whose value to increment
    /// * `delta` - The amount to add (can be negative for decrement)
    ///
    /// # Returns
    ///
    /// The new value after incrementing.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be serialized/deserialized as i64.
    pub fn increment(&self, term: &str, delta: i64) -> Result<i64, String> {
        #[cfg(feature = "parking_lot")]
        let mut inner = self.inner.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut inner = self.inner.write().expect("lock poisoned");

        // Get current value
        let current: i64 = if let Some(v) = Self::get_value_from_node(&inner.root, term.chars()) {
            let bytes = bincode::serialize(&v).map_err(|e| e.to_string())?;
            if bytes.len() == 8 {
                i64::from_le_bytes(bytes.try_into().unwrap())
            } else {
                bincode::deserialize::<i64>(&bytes).map_err(|e| e.to_string())?
            }
        } else {
            0
        };

        let new_value = current + delta;

        // Create value from i64
        let value_bytes = bincode::serialize(&new_value).map_err(|e| e.to_string())?;
        let v: V = bincode::deserialize(&value_bytes).map_err(|e| e.to_string())?;

        // Update the trie
        let chars: Vec<char> = term.chars().collect();
        // Check existence BEFORE getting mutable reference
        let existed = Self::contains_chars_impl(&inner.root, chars.iter().copied());
        let root = Arc::make_mut(&mut inner.root);
        Self::insert_chars_impl(root, chars.iter().copied(), Some(v));
        if !existed {
            inner.len += 1;
        }

        Ok(new_value)
    }

    /// Atomically update or insert a value.
    ///
    /// # Returns
    ///
    /// `true` if a new term was inserted, `false` if an existing term was updated.
    pub fn upsert(&self, term: &str, value: V) -> bool {
        #[cfg(feature = "parking_lot")]
        let mut inner = self.inner.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut inner = self.inner.write().expect("lock poisoned");

        let chars: Vec<char> = term.chars().collect();
        // Check existence BEFORE getting mutable reference
        let existed = Self::contains_chars_impl(&inner.root, chars.iter().copied());
        let root = Arc::make_mut(&mut inner.root);

        // Remove existing and insert new
        if existed {
            Self::remove_chars_impl(root, chars.iter().copied());
        }
        Self::insert_chars_impl(root, chars.iter().copied(), Some(value));

        if !existed {
            inner.len += 1;
        }

        !existed
    }

    /// Atomically compare and swap a value.
    ///
    /// Updates the value only if the current value matches `expected`.
    ///
    /// # Returns
    ///
    /// `true` if the swap succeeded, `false` if the current value didn't match expected.
    pub fn compare_and_swap(&self, term: &str, expected: Option<V>, new_value: V) -> bool {
        #[cfg(feature = "parking_lot")]
        let mut inner = self.inner.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut inner = self.inner.write().expect("lock poisoned");

        let chars: Vec<char> = term.chars().collect();
        let current = Self::get_value_from_node(&inner.root, chars.iter().copied());

        // Check if current matches expected
        let matches = match (&current, &expected) {
            (None, None) => true,
            (Some(c), Some(e)) => {
                let c_bytes = bincode::serialize(c).ok();
                let e_bytes = bincode::serialize(e).ok();
                c_bytes == e_bytes
            }
            _ => false,
        };

        if matches {
            let root = Arc::make_mut(&mut inner.root);
            let existed = current.is_some();
            if existed {
                Self::remove_chars_impl(root, chars.iter().copied());
            }
            Self::insert_chars_impl(root, chars.iter().copied(), Some(new_value));
            if !existed {
                inner.len += 1;
            }
        }

        matches
    }

    /// Get the current value and increment atomically (fetch-and-add).
    ///
    /// Returns the value *before* the increment.
    pub fn fetch_add(&self, term: &str, delta: i64) -> Result<i64, String> {
        let new_value = self.increment(term, delta)?;
        Ok(new_value - delta)
    }

    /// Get or insert a default value atomically.
    ///
    /// If the term exists, returns its current value.
    /// If not, inserts the default value and returns it.
    pub fn get_or_insert(&self, term: &str, default: V) -> V {
        #[cfg(feature = "parking_lot")]
        let mut inner = self.inner.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut inner = self.inner.write().expect("lock poisoned");

        let chars: Vec<char> = term.chars().collect();

        if let Some(v) = Self::get_value_from_node(&inner.root, chars.iter().copied()) {
            return v;
        }

        // Insert default value
        let root = Arc::make_mut(&mut inner.root);
        Self::insert_chars_impl(root, chars.iter().copied(), Some(default.clone()));
        inner.len += 1;

        default
    }

    /// Helper to get value from a node by character path
    fn get_value_from_node(node: &CharTrieNode<V>, mut chars: impl Iterator<Item = char>) -> Option<V> {
        match chars.next() {
            None => {
                if node.is_final {
                    node.value.clone()
                } else {
                    None
                }
            }
            Some(c) => {
                node.children
                    .get(&c)
                    .and_then(|child| Self::get_value_from_node(child, chars))
            }
        }
    }

    /// Helper to check if term exists
    fn contains_chars_impl(node: &CharTrieNode<V>, mut chars: impl Iterator<Item = char>) -> bool {
        match chars.next() {
            None => node.is_final,
            Some(c) => node
                .children
                .get(&c)
                .is_some_and(|child| Self::contains_chars_impl(child, chars)),
        }
    }

    /// Helper to insert with value
    fn insert_chars_impl(node: &mut CharTrieNode<V>, mut chars: impl Iterator<Item = char>, value: Option<V>) {
        match chars.next() {
            None => {
                node.is_final = true;
                node.value = value;
            }
            Some(c) => {
                let child = node.children.entry(c).or_insert_with(|| Arc::new(CharTrieNode::default()));
                let child = Arc::make_mut(child);
                Self::insert_chars_impl(child, chars, value);
            }
        }
    }

    /// Helper to remove a term
    fn remove_chars_impl(node: &mut CharTrieNode<V>, mut chars: impl Iterator<Item = char>) -> bool {
        match chars.next() {
            None => {
                let was_final = node.is_final;
                node.is_final = false;
                node.value = None;
                was_final
            }
            Some(c) => {
                if let Some(child) = node.children.get_mut(&c) {
                    let child = Arc::make_mut(child);
                    Self::remove_chars_impl(child, chars)
                } else {
                    false
                }
            }
        }
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

    #[test]
    fn test_iter_with_values() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
        trie.insert_with_value("alpha", 1);
        trie.insert_with_value("beta", 2);
        trie.insert_with_value("gamma", 3);

        let results: Vec<_> = trie.iter_with_values().collect();
        assert_eq!(results.len(), 3);

        // Check that we got all expected pairs
        assert!(results.contains(&("alpha".to_string(), 1)));
        assert!(results.contains(&("beta".to_string(), 2)));
        assert!(results.contains(&("gamma".to_string(), 3)));
    }

    #[test]
    fn test_iter_with_values_unicode() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
        trie.insert_with_value("日本語", 100);
        trie.insert_with_value("café", 200);
        trie.insert_with_value("emoji😀", 300);

        let results: Vec<_> = trie.iter_with_values().collect();
        assert_eq!(results.len(), 3);

        assert!(results.contains(&("日本語".to_string(), 100)));
        assert!(results.contains(&("café".to_string(), 200)));
        assert!(results.contains(&("emoji😀".to_string(), 300)));
    }

    #[test]
    fn test_iter_chars() {
        let trie: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        trie.insert("abc");
        trie.insert("日本");

        let results: Vec<_> = trie.iter_chars().collect();
        assert_eq!(results.len(), 2);

        // Check character vectors
        assert!(results.contains(&vec!['a', 'b', 'c']));
        assert!(results.contains(&vec!['日', '本']));
    }

    #[test]
    fn test_iter_chars_with_values() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
        trie.insert_with_value("hello", 1);
        trie.insert_with_value("世界", 2);

        let results: Vec<_> = trie.iter_chars_with_values().collect();
        assert_eq!(results.len(), 2);

        assert!(results.contains(&(vec!['h', 'e', 'l', 'l', 'o'], 1)));
        assert!(results.contains(&(vec!['世', '界'], 2)));
    }

    #[test]
    fn test_iter_empty_trie() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

        assert_eq!(trie.iter().count(), 0);
        assert_eq!(trie.iter_with_values().count(), 0);
        assert_eq!(trie.iter_chars().count(), 0);
        assert_eq!(trie.iter_chars_with_values().count(), 0);
    }

    #[test]
    fn test_iter_with_values_ordered() {
        // Test that iteration produces terms in a deterministic order
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();
        trie.insert_with_value("cat", 1);
        trie.insert_with_value("car", 2);
        trie.insert_with_value("card", 3);
        trie.insert_with_value("care", 4);

        let results: Vec<_> = trie.iter_with_values().collect();
        assert_eq!(results.len(), 4);

        // All terms should be present
        let terms: Vec<_> = results.iter().map(|(t, _)| t.as_str()).collect();
        assert!(terms.contains(&"cat"));
        assert!(terms.contains(&"car"));
        assert!(terms.contains(&"card"));
        assert!(terms.contains(&"care"));
    }

    // === Atomic Operations Tests ===

    #[test]
    fn test_atomic_increment_new() {
        let trie: PersistentARTrieChar<i64> = PersistentARTrieChar::new();

        let result = trie.increment("counter", 5).expect("increment");
        assert_eq!(result, 5);
        assert!(trie.contains("counter"));
    }

    #[test]
    fn test_atomic_increment_existing() {
        let trie: PersistentARTrieChar<i64> = PersistentARTrieChar::new();

        trie.upsert("counter", 10i64);
        let result = trie.increment("counter", 5).expect("increment");
        assert_eq!(result, 15);

        let result = trie.increment("counter", -3).expect("decrement");
        assert_eq!(result, 12);
    }

    #[test]
    fn test_atomic_upsert_new() {
        let trie: PersistentARTrieChar<String> = PersistentARTrieChar::new();

        let is_new = trie.upsert("greeting", "hello".to_string());
        assert!(is_new);
        assert_eq!(trie.get_value("greeting"), Some("hello".to_string()));
    }

    #[test]
    fn test_atomic_upsert_existing() {
        let trie: PersistentARTrieChar<String> = PersistentARTrieChar::new();

        trie.upsert("greeting", "hello".to_string());
        let is_new = trie.upsert("greeting", "hi".to_string());
        assert!(!is_new);
        assert_eq!(trie.get_value("greeting"), Some("hi".to_string()));
    }

    #[test]
    fn test_atomic_compare_and_swap_success() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

        trie.upsert("counter", 0i32);
        let success = trie.compare_and_swap("counter", Some(0), 1);
        assert!(success);
        assert_eq!(trie.get_value("counter"), Some(1));
    }

    #[test]
    fn test_atomic_compare_and_swap_failure() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

        trie.upsert("counter", 5i32);
        let success = trie.compare_and_swap("counter", Some(0), 10);
        assert!(!success);
        assert_eq!(trie.get_value("counter"), Some(5));
    }

    #[test]
    fn test_atomic_fetch_add() {
        let trie: PersistentARTrieChar<i64> = PersistentARTrieChar::new();

        trie.upsert("counter", 10i64);
        let old = trie.fetch_add("counter", 5).expect("fetch_add");
        assert_eq!(old, 10);
    }

    #[test]
    fn test_atomic_get_or_insert_new() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

        let value = trie.get_or_insert("key", 42);
        assert_eq!(value, 42);
        assert!(trie.contains("key"));
    }

    #[test]
    fn test_atomic_get_or_insert_existing() {
        let trie: PersistentARTrieChar<i32> = PersistentARTrieChar::new();

        trie.upsert("key", 100i32);
        let value = trie.get_or_insert("key", 42);
        assert_eq!(value, 100);
    }

    #[test]
    fn test_atomic_unicode_terms() {
        let trie: PersistentARTrieChar<i64> = PersistentARTrieChar::new();

        // Test with Unicode terms
        trie.increment("日本語カウンター", 1).expect("increment");
        trie.increment("日本語カウンター", 1).expect("increment");

        let result = trie.increment("日本語カウンター", 0).expect("read");
        assert_eq!(result, 2);

        trie.upsert("café", 100i64);
        assert_eq!(trie.get_value("café"), Some(100));
    }
}
