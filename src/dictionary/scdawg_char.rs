//! Character-level SCDAWG implementation with Unicode support.
//!
//! This module implements an SCDAWG (Symmetric Compact Directed Acyclic Word Graph)
//! that operates on Unicode scalar values (`char`) instead of bytes (`u8`).
//!
//! # When to Use ScdawgChar
//!
//! Use `ScdawgChar` when:
//! - Working with non-ASCII text (accented characters, CJK, emoji, etc.)
//! - You need correct character-level Levenshtein distances
//! - Pattern pieces for WallBreaker should be character-aligned
//!
//! # Features
//!
//! - **O(|pattern|) substring search**: True suffix automaton indexing ALL substrings
//! - **Left extension edges**: Bidirectional traversal via sext links
//! - **IS features**: freq(), locations() operations from Blumer et al. (1987)
//! - **Unicode support**: Proper character-level semantics
//!
//! # Performance Trade-offs
//!
//! Compared to byte-level `Scdawg`:
//! - **Memory**: ~4x edge label storage (4 bytes per `char` vs 1 byte per `u8`)
//! - **Speed**: Slightly slower due to larger edge labels
//! - **Correctness**: Proper Unicode semantics (e.g., "café" has 4 characters, not 5 bytes)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::dictionary::scdawg_char::ScdawgChar;
//! use liblevenshtein::dictionary::SubstringDictionary;
//!
//! // Create a Unicode-aware SCDAWG
//! let scdawg = ScdawgChar::<()>::from_terms(["café", "naïve", "中文"]);
//!
//! // O(|pattern|) substring search
//! assert!(scdawg.contains_substring("afé"));
//! assert!(scdawg.contains_substring("中"));
//!
//! // Find all occurrences
//! let matches = scdawg.find_exact_substring("afé");
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].position, 1);  // Position 1 in characters, not bytes
//! ```

use std::sync::Arc;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringDictionary, SubstringMatch};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{Dictionary, DictionaryNode};
use crate::sync_compat::RwLock;

/// Sentinel value for "no suffix link" or "no parent".
const NIL: usize = usize::MAX;

// ============================================================================
// True SCDAWG Char Node
// ============================================================================

/// A node in the Unicode-aware SCDAWG.
///
/// Represents an equivalence class of substrings that share the same end-position set.
#[derive(Clone, Debug)]
struct ScdawgCharNode<V: DictionaryValue = ()> {
    /// Forward edges (right extensions): label -> target node.
    /// These allow appending characters to the represented string.
    forward_edges: SmallVec<[(char, usize); 4]>,

    /// Suffix link: points to the state representing the longest proper suffix
    /// in a different equivalence class.
    suffix_link: usize,

    /// Left extension edges: label -> target node.
    /// These allow prepending characters to the represented string.
    /// Derived from suffix links after construction.
    left_edges: SmallVec<[(char, usize); 2]>,

    /// Maximum length of strings in this equivalence class (in characters).
    length: usize,

    /// Whether this state is a final state (end of some term).
    is_final: bool,

    /// Which terms end at this state (for multi-string support).
    /// Stores (term_index, position_in_term) pairs.
    term_ends: SmallVec<[(usize, usize); 2]>,

    /// Optional value associated with final states.
    value: Option<V>,

    /// Parent node in the canonical path (for bidirectional traversal).
    parent: usize,

    /// Edge label from parent to this node (last character of canonical path).
    parent_label: char,

    /// First character of the canonical (longest) string represented by this node.
    /// Used for computing left extension edges (sext links).
    first_char: char,

    /// Depth from root (number of edges in canonical path).
    depth: usize,
}

impl<V: DictionaryValue> ScdawgCharNode<V> {
    /// Create a new root node.
    fn root() -> Self {
        Self {
            forward_edges: SmallVec::new(),
            suffix_link: NIL,
            left_edges: SmallVec::new(),
            length: 0,
            is_final: false,
            term_ends: SmallVec::new(),
            value: None,
            parent: NIL,
            parent_label: '\0',
            first_char: '\0', // Root has no first character
            depth: 0,
        }
    }

    /// Create a new non-root node.
    fn new(length: usize, suffix_link: usize, first_char: char) -> Self {
        Self {
            forward_edges: SmallVec::new(),
            suffix_link,
            left_edges: SmallVec::new(),
            length,
            is_final: false,
            term_ends: SmallVec::new(),
            value: None,
            parent: NIL,
            parent_label: '\0',
            first_char,
            depth: 0,
        }
    }

    /// Find a forward edge by label.
    /// Uses binary search for sorted edges.
    #[inline(always)]
    fn get_edge(&self, label: char) -> Option<usize> {
        match self.forward_edges.binary_search_by_key(&label, |(l, _)| *l) {
            Ok(idx) => Some(self.forward_edges[idx].1),
            Err(_) => None,
        }
    }

    /// Add or update a forward edge.
    /// Maintains sorted order for binary search.
    #[inline(always)]
    fn set_edge(&mut self, label: char, target: usize) {
        match self.forward_edges.binary_search_by_key(&label, |(l, _)| *l) {
            Ok(idx) => self.forward_edges[idx].1 = target,
            Err(idx) => self.forward_edges.insert(idx, (label, target)),
        }
    }
}

// ============================================================================
// True SCDAWG Char Inner State
// ============================================================================

/// Inner mutable state of the Unicode-aware SCDAWG.
#[derive(Debug)]
struct ScdawgCharInner<V: DictionaryValue> {
    /// All nodes. Index 0 is always root.
    nodes: Vec<ScdawgCharNode<V>>,

    /// Last created node (for online construction).
    last: usize,

    /// Number of terms inserted.
    term_count: usize,

    /// Stored terms for enumeration.
    terms: Vec<String>,

    /// Fast duplicate detection using hash set.
    term_set: FxHashSet<String>,

    /// Whether left edges have been computed.
    left_edges_computed: bool,
}

impl<V: DictionaryValue> ScdawgCharInner<V> {
    /// Create a new empty Unicode-aware SCDAWG.
    fn new() -> Self {
        Self {
            nodes: vec![ScdawgCharNode::root()],
            last: 0,
            term_count: 0,
            terms: Vec::new(),
            term_set: FxHashSet::default(),
            left_edges_computed: false,
        }
    }

    /// Create with pre-allocated capacity.
    fn with_capacity(term_count: usize, total_chars: usize) -> Self {
        // Suffix automaton has at most 2*n nodes for n characters
        let estimated_nodes = total_chars.saturating_mul(2);
        let mut nodes = Vec::with_capacity(estimated_nodes);
        nodes.push(ScdawgCharNode::root());
        Self {
            nodes,
            last: 0,
            term_count: 0,
            terms: Vec::with_capacity(term_count),
            term_set: FxHashSet::with_capacity_and_hasher(term_count, Default::default()),
            left_edges_computed: false,
        }
    }

    /// Allocate a new node and return its index.
    fn alloc_node(&mut self, length: usize, suffix_link: usize, first_char: char) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ScdawgCharNode::new(length, suffix_link, first_char));
        idx
    }

    /// Clone a node (for split operations).
    fn clone_node(&mut self, src: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(self.nodes[src].clone());
        idx
    }

    /// Insert a single character, extending the suffix automaton.
    ///
    /// This is the core of Blumer et al.'s online suffix automaton construction.
    fn sa_extend(&mut self, c: char, term_idx: usize, pos: usize) {
        // Compute first_char for the new node:
        // - If extending from root, first_char is c (this char is the start of the string)
        // - Otherwise, inherit first_char from the current last node
        let first_char = if self.nodes[self.last].length == 0 {
            c
        } else {
            self.nodes[self.last].first_char
        };

        // Create new state for the new longest suffix
        let cur = self.alloc_node(self.nodes[self.last].length + 1, 0, first_char);

        // Set parent info for the new node
        self.nodes[cur].parent = self.last;
        self.nodes[cur].parent_label = c;
        self.nodes[cur].depth = self.nodes[self.last].depth + 1;

        // Walk up suffix links, adding edges to the new state
        let mut p = self.last;

        // Phase 1: Add edges from states that don't have edge labeled c
        while p != NIL && self.nodes[p].get_edge(c).is_none() {
            self.nodes[p].set_edge(c, cur);
            p = self.nodes[p].suffix_link;
        }

        if p == NIL {
            // Case 1: We reached the root without finding edge c
            // New state's suffix link goes to root
            self.nodes[cur].suffix_link = 0;
        } else {
            // Found a state p that has edge c
            let q = self.nodes[p].get_edge(c).unwrap();

            if self.nodes[p].length + 1 == self.nodes[q].length {
                // Case 2: Edge p->q is a "solid" edge (no need to split)
                self.nodes[cur].suffix_link = q;
            } else {
                // Case 3: Need to split state q
                // Create clone of q with shorter length
                let clone = self.clone_node(q);
                self.nodes[clone].length = self.nodes[p].length + 1;

                // Compute first_char for clone:
                // Clone represents the string from root to p, then c
                // If p is root, first_char is c; otherwise inherit from p
                self.nodes[clone].first_char = if self.nodes[p].length == 0 {
                    c
                } else {
                    self.nodes[p].first_char
                };

                // Update suffix links
                self.nodes[cur].suffix_link = clone;
                self.nodes[q].suffix_link = clone;

                // Update parent info for clone
                self.nodes[clone].parent = p;
                self.nodes[clone].parent_label = c;
                self.nodes[clone].depth = self.nodes[p].depth + 1;

                // Clear term_ends from clone (it's not a real final state)
                self.nodes[clone].term_ends.clear();
                self.nodes[clone].is_final = false;
                self.nodes[clone].value = None;

                // Redirect edges from p and its suffix chain that point to q
                while p != NIL && self.nodes[p].get_edge(c) == Some(q) {
                    self.nodes[p].set_edge(c, clone);
                    p = self.nodes[p].suffix_link;
                }
            }
        }

        // Record position in term
        self.nodes[cur].term_ends.push((term_idx, pos));

        self.last = cur;
        self.left_edges_computed = false; // Invalidate left edges
    }

    /// Insert a term into the SCDAWG.
    fn insert(&mut self, term: &str) -> bool {
        // Check for duplicate using O(1) hash lookup
        if self.term_set.contains(term) {
            return false;
        }

        let term_idx = self.term_count;

        // Reset to root for new term
        self.last = 0;

        // Insert each character
        for (pos, ch) in term.chars().enumerate() {
            self.sa_extend(ch, term_idx, pos);
        }

        // Mark the final state
        self.nodes[self.last].is_final = true;

        let term_string = term.to_string();
        self.term_set.insert(term_string.clone());
        self.terms.push(term_string);
        self.term_count += 1;

        true
    }

    /// Insert a term with an associated value.
    fn insert_with_value(&mut self, term: &str, value: V) -> bool {
        if self.insert(term) {
            self.nodes[self.last].value = Some(value);
            true
        } else {
            false
        }
    }

    /// Compute left extension edges from suffix links.
    ///
    /// For each suffix link from node A to node B, we add a left extension edge
    /// from B to A with the label being A's first_char.
    fn compute_left_edges(&mut self) {
        if self.left_edges_computed {
            return;
        }

        // Clear existing left edges
        for node in &mut self.nodes {
            node.left_edges.clear();
        }

        // For each node with a suffix link, add left edge to the suffix target
        for node_idx in 1..self.nodes.len() {
            let suffix_target = self.nodes[node_idx].suffix_link;
            if suffix_target != NIL {
                let label = self.nodes[node_idx].first_char;
                self.nodes[suffix_target].left_edges.push((label, node_idx));
            }
        }

        self.left_edges_computed = true;
    }

    /// Find exact substring matches using O(|pattern|) traversal.
    fn find_substring_fast(&self, pattern: &str) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0); // Empty pattern matches at root
        }

        let mut current = 0; // Start at root
        for ch in pattern.chars() {
            match self.nodes[current].get_edge(ch) {
                Some(next) => current = next,
                None => return None, // Pattern not found
            }
        }

        Some(current) // Return the node where pattern ends
    }

    /// Check if pattern is a substring of any term.
    fn contains_substring(&self, pattern: &str) -> bool {
        self.find_substring_fast(pattern).is_some()
    }

    /// Find all occurrences of a substring pattern.
    ///
    /// Returns (term, position) pairs where position is in characters.
    fn find_exact_substring(&self, pattern: &str) -> Vec<(String, usize)> {
        if pattern.is_empty() {
            // Empty pattern matches at position 0 of every term
            return self.terms.iter().map(|t| (t.clone(), 0)).collect();
        }

        // First, find the node where pattern ends (O(|pattern|))
        let end_node = match self.find_substring_fast(pattern) {
            Some(node) => node,
            None => return Vec::new(),
        };

        // Pattern length in characters
        let pattern_len = pattern.chars().count();
        let mut results = Vec::new();

        // DFS to find all final states reachable from end_node
        self.collect_term_positions(end_node, pattern_len, &mut results);

        results
    }

    /// Collect all term positions reachable from a node.
    fn collect_term_positions(
        &self,
        node: usize,
        pattern_len: usize,
        results: &mut Vec<(String, usize)>,
    ) {
        // Check if this node has term endings
        for &(term_idx, end_pos) in &self.nodes[node].term_ends {
            if end_pos + 1 >= pattern_len {
                let start_pos = end_pos + 1 - pattern_len;
                if term_idx < self.terms.len() {
                    results.push((self.terms[term_idx].clone(), start_pos));
                }
            }
        }

        // Traverse via left edges to find all nodes where this pattern occurs
        for &(_, target) in &self.nodes[node].left_edges {
            self.collect_term_positions(target, pattern_len, results);
        }
    }

    /// Check if the SCDAWG contains a complete term.
    fn contains(&self, term: &str) -> bool {
        self.term_set.contains(term)
    }

    /// Get the number of terms.
    fn term_count(&self) -> usize {
        self.term_count
    }

    // ========================================================================
    // IS Features Helper Methods
    // ========================================================================

    /// Get the frequency (occurrence count) of a substring pattern.
    fn frequency(&self, pattern: &str) -> usize {
        if pattern.is_empty() {
            // Empty pattern matches at every position in every term
            return self.terms.iter().map(|t| t.chars().count() + 1).sum();
        }

        match self.find_substring_fast(pattern) {
            Some(node) => {
                let mut count = 0;
                self.count_occurrences(node, &mut count);
                count
            }
            None => 0,
        }
    }

    /// Count all occurrences reachable from a node.
    fn count_occurrences(&self, node: usize, count: &mut usize) {
        *count += self.nodes[node].term_ends.len();
        for &(_, target) in &self.nodes[node].left_edges {
            self.count_occurrences(target, count);
        }
    }
}

// ============================================================================
// Public ScdawgChar Type
// ============================================================================

/// Unicode-aware Symmetric Compact DAWG with O(|pattern|) substring search.
///
/// This is a proper suffix automaton implementation that indexes ALL substrings
/// of all terms, enabling efficient substring search and bidirectional extension.
/// Uses `char` for edge labels to support Unicode text.
#[derive(Clone, Debug)]
pub struct ScdawgChar<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgCharInner<V>>>,
}

impl<V: DictionaryValue> Default for ScdawgChar<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> ScdawgChar<V> {
    /// Create a new empty Unicode-aware SCDAWG.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ScdawgCharInner::new())),
        }
    }

    /// Create from an iterator of terms.
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Collect terms to enable pre-allocation
        let terms_vec: Vec<S> = terms.into_iter().collect();
        let term_count = terms_vec.len();
        let total_chars: usize = terms_vec.iter().map(|s| s.as_ref().chars().count()).sum();

        let inner = ScdawgCharInner::with_capacity(term_count, total_chars);
        let scdawg = Self {
            inner: Arc::new(RwLock::new(inner)),
        };
        {
            let mut inner = scdawg.inner.write();
            for term in terms_vec {
                inner.insert(term.as_ref());
            }
            inner.compute_left_edges();
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
        let result = inner.insert(term);
        if result {
            inner.compute_left_edges();
        }
        result
    }

    /// Insert a term with a value.
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write();
        let result = inner.insert_with_value(term, value);
        if result {
            inner.compute_left_edges();
        }
        result
    }

    /// Check if a substring exists in any term.
    pub fn contains_substring(&self, pattern: &str) -> bool {
        let inner = self.inner.read();
        inner.contains_substring(pattern)
    }

    /// Iterate over all terms.
    pub fn iter(&self) -> impl Iterator<Item = String> {
        let inner = self.inner.read();
        inner.terms.clone().into_iter()
    }

    /// Get the number of terms in the SCDAWG.
    pub fn term_count(&self) -> usize {
        self.inner.read().term_count()
    }

    /// Get the number of nodes in the SCDAWG.
    pub fn node_count(&self) -> usize {
        self.inner.read().nodes.len()
    }

    /// Get the value associated with a term.
    pub fn get_value(&self, term: &str) -> Option<V>
    where
        V: Clone,
    {
        let inner = self.inner.read();
        let mut current = 0;
        for ch in term.chars() {
            match inner.nodes[current].get_edge(ch) {
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

    // ========================================================================
    // IS Features (Blumer et al. 1987)
    // ========================================================================

    /// Find a substring and return a handle to its SCDAWG state.
    ///
    /// This is the `find(x)` operation from Blumer et al. (1987).
    pub fn find(&self, pattern: &str) -> Option<ScdawgCharNodeHandle<V>> {
        let inner = self.inner.read();
        inner.find_substring_fast(pattern).map(|node_idx| ScdawgCharNodeHandle {
            inner: Arc::clone(&self.inner),
            node_idx,
        })
    }

    /// Get the frequency (occurrence count) of a substring pattern.
    ///
    /// This is the `freq(x)` operation from Blumer et al. (1987).
    pub fn freq(&self, pattern: &str) -> usize {
        let inner = self.inner.read();
        inner.frequency(pattern)
    }

    /// Get the frequency at a specific SCDAWG node handle.
    pub fn freq_at(&self, handle: &ScdawgCharNodeHandle<V>) -> usize {
        let inner = self.inner.read();
        let mut count = 0;
        inner.count_occurrences(handle.node_idx, &mut count);
        count
    }

    /// Get all occurrence locations of a substring pattern.
    ///
    /// This is the `locations(x)` operation from Blumer et al. (1987).
    /// Returns (term, start_position) pairs where position is in characters.
    pub fn locations(&self, pattern: &str) -> Vec<(String, usize)> {
        let inner = self.inner.read();
        inner.find_exact_substring(pattern)
    }

    /// Get all occurrence locations from a specific SCDAWG node handle.
    pub fn locations_at(&self, handle: &ScdawgCharNodeHandle<V>, pattern_len: usize) -> Vec<(String, usize)> {
        let inner = self.inner.read();
        let mut results = Vec::new();
        inner.collect_term_positions(handle.node_idx, pattern_len, &mut results);
        results
    }
}

// ============================================================================
// Dictionary Trait Implementation
// ============================================================================

impl<V: DictionaryValue> Dictionary for ScdawgChar<V> {
    type Node = ScdawgCharNodeHandle<V>;

    fn len(&self) -> Option<usize> {
        Some(self.inner.read().term_count())
    }

    fn contains(&self, term: &str) -> bool {
        self.inner.read().contains(term)
    }

    fn root(&self) -> Self::Node {
        ScdawgCharNodeHandle {
            inner: Arc::clone(&self.inner),
            node_idx: 0,
        }
    }

    fn sync_strategy(&self) -> crate::dictionary::SyncStrategy {
        crate::dictionary::SyncStrategy::ExternalSync
    }
}

// ============================================================================
// Node Handle
// ============================================================================

/// Handle to a node in the Unicode-aware SCDAWG.
#[derive(Clone)]
pub struct ScdawgCharNodeHandle<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgCharInner<V>>>,
    node_idx: usize,
}

impl<V: DictionaryValue> std::fmt::Debug for ScdawgCharNodeHandle<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScdawgCharNodeHandle")
            .field("node_idx", &self.node_idx)
            .finish()
    }
}

impl<V: DictionaryValue> DictionaryNode for ScdawgCharNodeHandle<V> {
    type Unit = char;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].is_final
    }

    fn transition(&self, label: char) -> Option<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .get_edge(label)
            .map(|idx| ScdawgCharNodeHandle {
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
                    ScdawgCharNodeHandle {
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

unsafe impl<V: DictionaryValue> Send for ScdawgCharNodeHandle<V> {}
unsafe impl<V: DictionaryValue> Sync for ScdawgCharNodeHandle<V> {}

// ============================================================================
// BidirectionalDictionaryNode Implementation
// ============================================================================

impl<V: DictionaryValue> BidirectionalDictionaryNode for ScdawgCharNodeHandle<V> {
    fn parent(&self) -> Option<Self> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NIL {
            None
        } else {
            Some(ScdawgCharNodeHandle {
                inner: Arc::clone(&self.inner),
                node_idx: node.parent,
            })
        }
    }

    fn parent_label(&self) -> Option<char> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NIL {
            None
        } else {
            Some(node.parent_label)
        }
    }

    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .left_edges
            .iter()
            .map(|&(label, idx)| {
                (
                    label,
                    ScdawgCharNodeHandle {
                        inner: Arc::clone(&self.inner),
                        node_idx: idx,
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }

    fn reverse_transition(&self, label: char) -> Vec<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .left_edges
            .iter()
            .filter(|(l, _)| *l == label)
            .map(|(_, idx)| ScdawgCharNodeHandle {
                inner: Arc::clone(&self.inner),
                node_idx: *idx,
            })
            .collect()
    }

    fn depth(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].depth
    }
}

// ============================================================================
// SubstringDictionary Implementation
// ============================================================================

impl<V: DictionaryValue> SubstringDictionary for ScdawgChar<V> {
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>> {
        let inner = self.inner.read();
        let occurrences = inner.find_exact_substring(pattern);
        let pattern_len = pattern.chars().count();

        occurrences
            .into_iter()
            .map(|(term, position)| {
                // Find the node at the end of the pattern match
                let mut node_idx = 0;
                for ch in term.chars().take(position + pattern_len) {
                    if let Some(next) = inner.nodes[node_idx].get_edge(ch) {
                        node_idx = next;
                    }
                }

                SubstringMatch::new(
                    ScdawgCharNodeHandle {
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
    fn test_scdawg_char_empty() {
        let scdawg = ScdawgChar::<()>::new();
        assert_eq!(scdawg.term_count(), 0);
        assert!(!scdawg.contains("anything"));
    }

    #[test]
    fn test_scdawg_char_insert_single() {
        let scdawg = ScdawgChar::<()>::new();
        assert!(scdawg.insert("hello"));
        assert!(!scdawg.insert("hello")); // Duplicate
        assert_eq!(scdawg.term_count(), 1);
        assert!(scdawg.contains("hello"));
    }

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

        // Test O(|pattern|) substring search
        assert!(scdawg.contains_substring("afé"));
        assert!(scdawg.contains_substring("ca"));
        assert!(scdawg.contains_substring("fé"));
        assert!(!scdawg.contains_substring("xyz"));
    }

    #[test]
    fn test_scdawg_char_find_exact_substring() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["café"]);
        let matches = scdawg.find_exact_substring("afé");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].term, "café");
        assert_eq!(matches[0].position, 1); // Character position, not byte
        assert_eq!(matches[0].length, 3);   // 3 characters
    }

    #[test]
    fn test_scdawg_char_cjk() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["中文字"]);

        assert!(scdawg.contains_substring("中"));
        assert!(scdawg.contains_substring("中文"));
        assert!(scdawg.contains_substring("文字"));
        assert!(scdawg.contains_substring("中文字"));

        let matches = scdawg.find_exact_substring("文");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].position, 1); // Position 1 in characters
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

    #[test]
    fn test_scdawg_char_multiple_terms() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["abc", "bcd", "cde"]);

        // Each term should be found
        assert!(scdawg.contains("abc"));
        assert!(scdawg.contains("bcd"));
        assert!(scdawg.contains("cde"));

        // Common substrings
        assert!(scdawg.contains_substring("bc")); // In abc and bcd
        assert!(scdawg.contains_substring("cd")); // In bcd and cde
    }

    #[test]
    fn test_scdawg_char_is_freq() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["abab"]);

        // "ab" appears twice in "abab": at positions 0 and 2
        assert_eq!(scdawg.freq("ab"), 2);

        // "a" appears twice
        assert_eq!(scdawg.freq("a"), 2);

        // Non-existent pattern
        assert_eq!(scdawg.freq("xyz"), 0);
    }

    #[test]
    fn test_scdawg_char_is_locations() {
        let scdawg = ScdawgChar::<()>::from_terms(vec!["abab"]);

        let locs = scdawg.locations("ab");

        // Should find "ab" at positions 0 and 2 in "abab"
        assert_eq!(locs.len(), 2);

        let positions: std::collections::HashSet<_> = locs.iter().map(|(_, pos)| *pos).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&2));
    }

    #[test]
    fn test_scdawg_char_left_extension_edges() {
        // "abc" and "dbc" share suffix "bc"
        let scdawg = ScdawgChar::<()>::from_terms(vec!["abc", "dbc"]);

        // Navigate to "bc" node
        let root = scdawg.root();
        let node_b = root.transition('b').expect("Should have edge 'b'");
        let node_bc = node_b.transition('c').expect("Should have edge 'c'");

        // "bc" should have left extensions for both 'a' and 'd'
        let left_edges: Vec<_> = node_bc.reverse_edges().collect();
        let labels: std::collections::HashSet<_> = left_edges.iter().map(|(l, _)| *l).collect();

        assert!(labels.contains(&'a'), "Should have left extension 'a'");
        assert!(labels.contains(&'d'), "Should have left extension 'd'");
    }
}
