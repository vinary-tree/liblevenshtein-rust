//! Character-level SCDAWG for WallBreaker algorithm with Unicode support.
//!
//! This module provides `ScdawgChar`, a Unicode-aware variant of [`Scdawg`](super::scdawg::Scdawg)
//! that operates on Unicode scalar values (`char`) instead of bytes (`u8`).
//!
//! # When to Use ScdawgChar
//!
//! Use `ScdawgChar` when:
//! - Working with non-ASCII text (accented characters, CJK, emoji, etc.)
//! - You need correct character-level Levenshtein distances
//! - Pattern pieces for WallBreaker should be character-aligned
//!
//! # Performance Trade-offs
//!
//! Compared to byte-level `Scdawg`:
//! - **Memory**: ~4x edge label storage (4 bytes per `char` vs 1 byte per `u8`)
//! - **Speed**: ~5-10% slower due to UTF-8 encoding/decoding
//! - **Correctness**: Proper Unicode semantics (e.g., "café" has 4 characters, not 5 bytes)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::scdawg_char::ScdawgChar;
//! use liblevenshtein::dictionary::SubstringDictionary;
//!
//! // Create a Unicode-aware SCDAWG
//! let scdawg = ScdawgChar::<()>::from_terms(vec!["café", "naïve", "中文"]);
//!
//! // Find substring matches (character-level)
//! let matches = scdawg.find_exact_substring("afé");
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].term, "café");
//! assert_eq!(matches[0].position, 1);  // Position 1 in characters, not bytes
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringDictionary, SubstringMatch};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
use crate::sync_compat::RwLock;

/// SCDAWG node with character-level (Unicode) edge labels.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct ScdawgNodeChar<V: DictionaryValue = ()> {
    /// Forward edges: (char label, target node index).
    pub(crate) forward_edges: SmallVec<[(char, usize); 4]>,

    /// Backward edges: for each label, list of parent node indices.
    pub(crate) backward_edges: SmallVec<[(char, SmallVec<[usize; 2]>); 2]>,

    /// Suffix link for construction and navigation.
    pub(crate) suffix_link: Option<usize>,

    /// Parent node index (usize::MAX = no parent/root).
    pub(crate) parent: usize,

    /// Edge label from parent to this node.
    pub(crate) parent_label: char,

    /// Maximum length of strings reaching this state.
    pub(crate) max_length: usize,

    /// Minimum length of strings reaching this state.
    pub(crate) min_length: usize,

    /// Depth from root.
    pub(crate) depth: usize,

    /// True if this state represents end-of-word.
    pub(crate) is_final: bool,

    /// Reference count for deletion support.
    pub(crate) ref_count: usize,

    /// Optional value for final nodes.
    pub(crate) value: Option<V>,
}

/// Sentinel value for "no parent" (root node).
const NO_PARENT: usize = usize::MAX;

impl<V: DictionaryValue> ScdawgNodeChar<V> {
    fn root() -> Self {
        Self {
            forward_edges: SmallVec::new(),
            backward_edges: SmallVec::new(),
            suffix_link: None,
            parent: NO_PARENT,
            parent_label: '\0',
            max_length: 0,
            min_length: 0,
            depth: 0,
            is_final: false,
            ref_count: 1,
            value: None,
        }
    }

    fn new_with_parent(
        max_length: usize,
        min_length: usize,
        parent: usize,
        parent_label: char,
        depth: usize,
    ) -> Self {
        Self {
            forward_edges: SmallVec::new(),
            backward_edges: SmallVec::new(),
            suffix_link: None,
            parent,
            parent_label,
            max_length,
            min_length,
            depth,
            is_final: false,
            ref_count: 0,
            value: None,
        }
    }

    #[inline]
    fn find_forward_edge(&self, label: char) -> Option<usize> {
        if self.forward_edges.len() < 16 {
            self.forward_edges
                .iter()
                .find(|(c, _)| *c == label)
                .map(|(_, t)| *t)
        } else {
            self.forward_edges
                .binary_search_by_key(&label, |(c, _)| *c)
                .ok()
                .map(|idx| self.forward_edges[idx].1)
        }
    }

    fn add_forward_edge(&mut self, label: char, target: usize) {
        match self.forward_edges.binary_search_by_key(&label, |(c, _)| *c) {
            Ok(idx) => self.forward_edges[idx].1 = target,
            Err(idx) => self.forward_edges.insert(idx, (label, target)),
        }
    }

    fn add_backward_edge(&mut self, label: char, parent: usize) {
        for (l, parents) in &mut self.backward_edges {
            if *l == label {
                if !parents.contains(&parent) {
                    parents.push(parent);
                }
                return;
            }
        }
        let mut parents = SmallVec::new();
        parents.push(parent);
        self.backward_edges.push((label, parents));
        self.backward_edges.sort_by_key(|(l, _)| *l);
    }

    fn backward_edge_iter(&self) -> impl Iterator<Item = (char, usize)> + '_ {
        self.backward_edges
            .iter()
            .flat_map(|(label, parents)| parents.iter().map(move |&p| (*label, p)))
    }

    fn find_backward_edges(&self, label: char) -> Vec<usize> {
        for (l, parents) in &self.backward_edges {
            if *l == label {
                return parents.to_vec();
            }
        }
        Vec::new()
    }
}

/// Inner mutable state of the character-level SCDAWG.
#[derive(Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct ScdawgCharInner<V: DictionaryValue> {
    pub(crate) nodes: Vec<ScdawgNodeChar<V>>,
    term_count: usize,
    last_node: usize,
    needs_compaction: bool,
    #[cfg_attr(feature = "serialization", serde(skip))]
    suffix_cache: FxHashMap<u64, usize>,
    #[cfg_attr(feature = "serialization", serde(skip))]
    term_index: HashMap<String, Vec<usize>>,
}

impl<V: DictionaryValue> ScdawgCharInner<V> {
    fn new() -> Self {
        Self {
            nodes: vec![ScdawgNodeChar::root()],
            term_count: 0,
            last_node: 0,
            needs_compaction: false,
            suffix_cache: FxHashMap::default(),
            term_index: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: usize, label: char, to: usize) {
        self.nodes[from].add_forward_edge(label, to);
        self.nodes[to].add_backward_edge(label, from);
        self.nodes[to].ref_count += 1;
    }

    fn insert(&mut self, term: &str) -> bool {
        if term.is_empty() {
            if self.nodes[0].is_final {
                return false;
            }
            self.nodes[0].is_final = true;
            self.term_count += 1;
            return true;
        }

        let chars: Vec<char> = term.chars().collect();
        let mut current = 0usize;
        let mut depth = 0usize;
        let mut path = vec![0usize];

        for &ch in &chars {
            depth += 1;

            if let Some(next) = self.nodes[current].find_forward_edge(ch) {
                current = next;
                self.nodes[current].ref_count += 1;
            } else {
                let parent_max_length = self.nodes[current].max_length;
                let new_idx = self.nodes.len();
                self.nodes.push(ScdawgNodeChar::new_with_parent(
                    parent_max_length + 1,
                    parent_max_length + 1,
                    current,
                    ch,
                    depth,
                ));

                self.add_edge(current, ch, new_idx);
                self.setup_suffix_link(current, ch, new_idx);

                current = new_idx;
            }

            path.push(current);
        }

        if self.nodes[current].is_final {
            return false;
        }

        self.nodes[current].is_final = true;
        self.term_count += 1;
        self.last_node = current;
        self.term_index.insert(term.to_string(), path);

        true
    }

    fn setup_suffix_link(&mut self, parent: usize, label: char, new_node: usize) {
        if parent == 0 {
            self.nodes[new_node].suffix_link = Some(0);
            return;
        }

        if let Some(parent_suffix) = self.nodes[parent].suffix_link {
            if let Some(target) = self.nodes[parent_suffix].find_forward_edge(label) {
                self.nodes[new_node].suffix_link = Some(target);
            } else {
                self.nodes[new_node].suffix_link = Some(0);
            }
        } else {
            self.nodes[new_node].suffix_link = Some(0);
        }
    }

    fn insert_with_value(&mut self, term: &str, value: V) -> bool {
        let inserted = self.insert(term);
        if inserted {
            self.nodes[self.last_node].value = Some(value);
        }
        inserted
    }

    fn remove(&mut self, term: &str) -> bool {
        if term.is_empty() {
            if !self.nodes[0].is_final {
                return false;
            }
            self.nodes[0].is_final = false;
            self.nodes[0].value = None;
            self.term_count -= 1;
            return true;
        }

        let chars: Vec<char> = term.chars().collect();
        let mut current = 0;

        for &ch in &chars {
            let next = match self.nodes[current].find_forward_edge(ch) {
                Some(n) => n,
                None => return false,
            };
            if self.nodes[next].ref_count > 0 {
                self.nodes[next].ref_count -= 1;
            }
            current = next;
        }

        if !self.nodes[current].is_final {
            return false;
        }

        self.nodes[current].is_final = false;
        self.nodes[current].value = None;
        self.term_count -= 1;
        self.needs_compaction = true;
        self.term_index.remove(term);

        true
    }

    fn contains(&self, term: &str) -> bool {
        let mut current = 0;
        for ch in term.chars() {
            match self.nodes[current].find_forward_edge(ch) {
                Some(next) => current = next,
                None => return false,
            }
        }
        self.nodes[current].is_final
    }

    /// Find all occurrences of a substring pattern.
    ///
    /// Note: This implementation uses O(total_chars * pattern_len) complexity.
    /// See scdawg.rs for detailed explanation of why O(|pattern| + occurrences)
    /// optimization requires a true suffix automaton rather than a DAWG.
    fn find_exact_substring(&self, pattern: &str) -> Vec<(String, usize)> {
        if pattern.is_empty() {
            return self.collect_all_terms().into_iter().map(|t| (t, 0)).collect();
        }

        let pattern_chars: Vec<char> = pattern.chars().collect();
        let mut results = Vec::new();

        for term in self.collect_all_terms() {
            let term_chars: Vec<char> = term.chars().collect();

            for pos in 0..=term_chars.len().saturating_sub(pattern_chars.len()) {
                if term_chars[pos..].starts_with(&pattern_chars) {
                    results.push((term.clone(), pos));
                }
            }
        }

        results
    }

    fn path_to_node(&self, node: usize) -> Vec<char> {
        let mut path = Vec::new();
        let mut current = node;

        while current != 0 && self.nodes[current].parent != NO_PARENT {
            path.push(self.nodes[current].parent_label);
            current = self.nodes[current].parent;
        }

        path.reverse();
        path
    }

    fn collect_all_terms(&self) -> Vec<String> {
        let mut terms = Vec::new();
        let mut stack: Vec<(usize, Vec<char>)> = vec![(0, Vec::new())];

        while let Some((current, path)) = stack.pop() {
            if self.nodes[current].is_final {
                terms.push(path.iter().collect());
            }

            for &(label, child) in &self.nodes[current].forward_edges {
                let mut new_path = path.clone();
                new_path.push(label);
                stack.push((child, new_path));
            }
        }

        terms
    }

    fn compact(&mut self) {
        if !self.needs_compaction {
            return;
        }

        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![0usize];
        reachable[0] = true;

        while let Some(node) = stack.pop() {
            for &(_, child) in &self.nodes[node].forward_edges {
                if !reachable[child] {
                    reachable[child] = true;
                    stack.push(child);
                }
            }
        }

        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut new_idx = 0usize;
        for (old_idx, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                remap[old_idx] = new_idx;
                new_idx += 1;
            }
        }

        let mut new_nodes = Vec::with_capacity(new_idx);
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if !reachable[old_idx] {
                continue;
            }

            let mut new_node = node.clone();

            for (_, target) in &mut new_node.forward_edges {
                *target = remap[*target];
            }

            for (_, parents) in &mut new_node.backward_edges {
                for parent in parents {
                    if *parent != NO_PARENT && *parent < remap.len() {
                        *parent = remap[*parent];
                    }
                }
            }

            if new_node.parent != NO_PARENT && new_node.parent < remap.len() {
                new_node.parent = remap[new_node.parent];
            }

            if let Some(ref mut suffix) = new_node.suffix_link {
                if *suffix < remap.len() {
                    *suffix = remap[*suffix];
                }
            }

            new_nodes.push(new_node);
        }

        self.nodes = new_nodes;
        self.needs_compaction = false;

        self.term_index.clear();
        for term in self.collect_all_terms() {
            let mut path = vec![0usize];
            let mut current = 0;
            for ch in term.chars() {
                if let Some(next) = self.nodes[current].find_forward_edge(ch) {
                    current = next;
                    path.push(current);
                }
            }
            self.term_index.insert(term, path);
        }
    }
}

/// A character-level SCDAWG for WallBreaker algorithm with Unicode support.
///
/// This is the UTF-8 aware variant of [`Scdawg`](super::scdawg::Scdawg).
/// It provides correct character-level operations for multi-byte UTF-8 text.
#[derive(Clone, Debug)]
pub struct ScdawgChar<V: DictionaryValue = ()> {
    pub(crate) inner: Arc<RwLock<ScdawgCharInner<V>>>,
}

impl<V: DictionaryValue> ScdawgChar<V> {
    /// Create a new empty character-level SCDAWG.
    pub fn new() -> Self {
        ScdawgChar {
            inner: Arc::new(RwLock::new(ScdawgCharInner::new())),
        }
    }

    /// Create from an iterator of terms.
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let scdawg = ScdawgChar::new();
        for term in terms {
            scdawg.insert(term.as_ref());
        }
        scdawg
    }

    /// Create from an iterator of (term, value) pairs.
    pub fn from_terms_with_values<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
    {
        let scdawg = ScdawgChar::new();
        for (term, value) in terms {
            scdawg.insert_with_value(term.as_ref(), value);
        }
        scdawg
    }

    /// Insert a term.
    pub fn insert(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        inner.insert(term)
    }

    /// Insert a term with an associated value.
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write();
        inner.insert_with_value(term, value)
    }

    /// Remove a term.
    pub fn remove(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        inner.remove(term)
    }

    /// Check if compaction is needed.
    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read();
        inner.needs_compaction
    }

    /// Compact the SCDAWG.
    pub fn compact(&self) {
        let mut inner = self.inner.write();
        inner.compact();
    }

    /// Get the number of terms.
    pub fn term_count(&self) -> usize {
        let inner = self.inner.read();
        inner.term_count
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes.len()
    }

    /// Get an iterator over all terms.
    pub fn iter(&self) -> impl Iterator<Item = String> {
        let inner = self.inner.read();
        inner.collect_all_terms().into_iter()
    }

    /// Get the value associated with a term.
    pub fn get_value(&self, term: &str) -> Option<V>
    where
        V: Clone,
    {
        let inner = self.inner.read();
        let mut current = 0;
        for ch in term.chars() {
            match inner.nodes[current].find_forward_edge(ch) {
                Some(next) => current = next,
                None => return None,
            }
        }
        if inner.nodes[current].is_final {
            inner.nodes[current].value.clone()
        } else {
            None
        }
    }
}

impl<V: DictionaryValue> Default for ScdawgChar<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dictionary trait implementations
// ============================================================================

/// Node wrapper for character-level SCDAWG.
#[derive(Clone)]
pub struct ScdawgCharNode<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgCharInner<V>>>,
    node_idx: usize,
}

impl<V: DictionaryValue> std::fmt::Debug for ScdawgCharNode<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("ScdawgCharNode")
            .field("node_idx", &self.node_idx)
            .field("is_final", &inner.nodes[self.node_idx].is_final)
            .field("depth", &inner.nodes[self.node_idx].depth)
            .finish()
    }
}

impl<V: DictionaryValue> Dictionary for ScdawgChar<V> {
    type Node = ScdawgCharNode<V>;

    fn root(&self) -> Self::Node {
        ScdawgCharNode {
            inner: Arc::clone(&self.inner),
            node_idx: 0,
        }
    }

    fn contains(&self, term: &str) -> bool {
        let inner = self.inner.read();
        inner.contains(term)
    }

    fn len(&self) -> Option<usize> {
        Some(self.term_count())
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync
    }
}

impl<V: DictionaryValue> DictionaryNode for ScdawgCharNode<V> {
    type Unit = char;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].is_final
    }

    fn transition(&self, label: char) -> Option<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .find_forward_edge(label)
            .map(|idx| ScdawgCharNode {
                inner: Arc::clone(&self.inner),
                node_idx: idx,
            })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .forward_edges
            .iter()
            .map(|&(label, idx)| {
                (
                    label,
                    ScdawgCharNode {
                        inner: Arc::clone(&self.inner),
                        node_idx: idx,
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }

    fn edge_count(&self) -> Option<usize> {
        let inner = self.inner.read();
        Some(inner.nodes[self.node_idx].forward_edges.len())
    }
}

unsafe impl<V: DictionaryValue> Send for ScdawgCharNode<V> {}
unsafe impl<V: DictionaryValue> Sync for ScdawgCharNode<V> {}

// ============================================================================
// BidirectionalDictionaryNode implementation
// ============================================================================

impl<V: DictionaryValue> BidirectionalDictionaryNode for ScdawgCharNode<V> {
    fn parent(&self) -> Option<Self> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NO_PARENT {
            None
        } else {
            Some(ScdawgCharNode {
                inner: Arc::clone(&self.inner),
                node_idx: node.parent,
            })
        }
    }

    fn parent_label(&self) -> Option<char> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NO_PARENT {
            None
        } else {
            Some(node.parent_label)
        }
    }

    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .backward_edge_iter()
            .map(|(label, parent_idx)| {
                (
                    label,
                    ScdawgCharNode {
                        inner: Arc::clone(&self.inner),
                        node_idx: parent_idx,
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }

    fn reverse_transition(&self, label: char) -> Vec<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .find_backward_edges(label)
            .into_iter()
            .map(|idx| ScdawgCharNode {
                inner: Arc::clone(&self.inner),
                node_idx: idx,
            })
            .collect()
    }

    fn depth(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].depth
    }
}

// ============================================================================
// SubstringDictionary implementation
// ============================================================================

impl<V: DictionaryValue> SubstringDictionary for ScdawgChar<V> {
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>> {
        let inner = self.inner.read();
        let occurrences = inner.find_exact_substring(pattern);
        let pattern_len = pattern.chars().count();

        occurrences
            .into_iter()
            .map(|(term, position)| {
                let mut node_idx = 0;
                for ch in term.chars().take(position + pattern_len) {
                    if let Some(next) = inner.nodes[node_idx].find_forward_edge(ch) {
                        node_idx = next;
                    }
                }

                SubstringMatch::new(
                    ScdawgCharNode {
                        inner: Arc::clone(&self.inner),
                        node_idx,
                    },
                    term,
                    position,
                    pattern_len,
                )
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scdawg_char_unicode() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["café", "naïve", "中文"]);
        assert_eq!(scdawg.term_count(), 3);
        assert!(scdawg.contains("café"));
        assert!(scdawg.contains("naïve"));
        assert!(scdawg.contains("中文"));
        assert!(!scdawg.contains("cafe")); // Without accent
    }

    #[test]
    fn test_scdawg_char_substring_search() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["café"]);
        let matches = scdawg.find_exact_substring("afé");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].term, "café");
        assert_eq!(matches[0].position, 1); // Character position, not byte
        assert_eq!(matches[0].length, 3);   // 3 characters
    }

    #[test]
    fn test_scdawg_char_bidirectional() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["中文"]);

        let root = scdawg.root();
        let zhong = root.transition('中').unwrap();
        let wen = zhong.transition('文').unwrap();

        assert!(wen.is_final());
        assert_eq!(wen.depth(), 2);

        // Walk back
        let back = wen.parent().unwrap();
        assert_eq!(wen.parent_label(), Some('文'));
        assert_eq!(back.depth(), 1);

        let back_root = back.parent().unwrap();
        assert!(back_root.parent().is_none());
    }

    #[test]
    fn test_scdawg_char_path_string() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["café"]);

        let root = scdawg.root();
        let c = root.transition('c').unwrap();
        let a = c.transition('a').unwrap();
        let f = a.transition('f').unwrap();
        let e = f.transition('é').unwrap();

        assert_eq!(e.path_string(), "café");
        assert_eq!(a.path_string(), "ca");
    }

    #[test]
    fn test_scdawg_char_with_values() {
        let scdawg = ScdawgChar::<u32>::new();
        scdawg.insert_with_value("日本語", 42);

        assert_eq!(scdawg.get_value("日本語"), Some(42));
        assert_eq!(scdawg.get_value("日本"), None);
    }

    #[test]
    fn test_scdawg_char_emoji() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["hello🎉world"]);

        assert!(scdawg.contains("hello🎉world"));
        assert_eq!(scdawg.term_count(), 1);

        // Emoji is 1 character
        let matches = scdawg.find_exact_substring("🎉");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].position, 5); // After "hello"
    }
}
