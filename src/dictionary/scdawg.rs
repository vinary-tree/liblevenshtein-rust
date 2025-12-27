//! Symmetric Compact DAWG (SCDAWG) implementation.
//!
//! This module implements an SCDAWG (Symmetric Compact Directed Acyclic Word Graph)
//! following the algorithms described in:
//! - Blumer et al. (1987): "Complete Inverted Files for Efficient Text Retrieval and Analysis"
//! - Inenaga et al. (2005): "On-line construction of compact directed acyclic word graphs"
//!
//! # Features
//!
//! - **O(|pattern|) substring search**: True suffix automaton indexing ALL substrings
//! - **Left extension edges**: Bidirectional traversal via sext links
//! - **IS features**: freq(), locations() operations from Blumer et al.
//! - **WallBreaker compatible**: Supports requirements (1a), (1b), (1c)
//!
//! # Algorithm Overview
//!
//! For each term, we build a suffix automaton that indexes all substrings.
//! For multi-string support, each term is processed independently with shared structure.
//!
//! # Data Structure
//!
//! Each node represents an equivalence class of substrings with the same end-position set.
//! - `forward_edges`: Standard CDAWG edges (appending characters)
//! - `suffix_link`: Points to the longest proper suffix in a different equivalence class
//! - `left_edges`: Left extension edges (prepending characters) - derived from suffix links
//! - `length`: Maximum length of strings in this equivalence class
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//! use liblevenshtein::dictionary::SubstringDictionary;
//!
//! // Create an SCDAWG from terms
//! let scdawg = Scdawg::<()>::from_terms(["cathedral", "category", "catering"]);
//!
//! // O(|pattern|) substring search
//! assert!(scdawg.contains_substring("cat"));
//! assert!(scdawg.contains_substring("thedr"));
//!
//! // Find all occurrences
//! let matches = scdawg.find_exact_substring("cat");
//! assert_eq!(matches.len(), 3);  // Found in all three terms
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

/// End marker base for multi-string support.
/// Each term gets a unique end marker: END_MARKER_BASE + term_index.
const END_MARKER_BASE: u8 = 0x01; // Use low bytes as end markers

// ============================================================================
// True SCDAWG Node
// ============================================================================

/// A node in the true SCDAWG.
///
/// Represents an equivalence class of substrings that share the same end-position set.
#[derive(Clone, Debug)]
struct ScdawgNode<V: DictionaryValue = ()> {
    /// Forward edges (right extensions): label -> target node.
    /// These allow appending characters to the represented string.
    forward_edges: SmallVec<[(u8, usize); 4]>,

    /// Suffix link: points to the state representing the longest proper suffix
    /// in a different equivalence class.
    suffix_link: usize,

    /// Left extension edges: label -> target node.
    /// These allow prepending characters to the represented string.
    /// Derived from suffix links after construction.
    left_edges: SmallVec<[(u8, usize); 2]>,

    /// Maximum length of strings in this equivalence class.
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
    parent_label: u8,

    /// First character of the canonical (longest) string represented by this node.
    /// Used for computing left extension edges (sext links).
    first_char: u8,

    /// Depth from root (number of edges in canonical path).
    depth: usize,
}

impl<V: DictionaryValue> ScdawgNode<V> {
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
            parent_label: 0,
            first_char: 0, // Root has no first character
            depth: 0,
        }
    }

    /// Create a new non-root node.
    fn new(length: usize, suffix_link: usize, first_char: u8) -> Self {
        Self {
            forward_edges: SmallVec::new(),
            suffix_link,
            left_edges: SmallVec::new(),
            length,
            is_final: false,
            term_ends: SmallVec::new(),
            value: None,
            parent: NIL,
            parent_label: 0,
            first_char,
            depth: 0,
        }
    }

    /// Find a forward edge by label.
    /// Uses binary search for sorted edges (typically small, but helps at scale).
    #[inline(always)]
    fn get_edge(&self, label: u8) -> Option<usize> {
        // For small edge counts, linear is fine; binary search helps at scale
        match self.forward_edges.binary_search_by_key(&label, |(l, _)| *l) {
            Ok(idx) => Some(self.forward_edges[idx].1),
            Err(_) => None,
        }
    }

    /// Add or update a forward edge.
    /// Maintains sorted order for binary search.
    #[inline(always)]
    fn set_edge(&mut self, label: u8, target: usize) {
        match self.forward_edges.binary_search_by_key(&label, |(l, _)| *l) {
            Ok(idx) => self.forward_edges[idx].1 = target,
            Err(idx) => self.forward_edges.insert(idx, (label, target)),
        }
    }

    /// Check if this is the root node.
    #[inline]
    fn is_root(&self) -> bool {
        self.parent == NIL && self.length == 0
    }
}

// ============================================================================
// True SCDAWG Inner State
// ============================================================================

/// Inner mutable state of the true SCDAWG.
#[derive(Debug)]
struct ScdawgInner<V: DictionaryValue> {
    /// All nodes. Index 0 is always root.
    nodes: Vec<ScdawgNode<V>>,

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

impl<V: DictionaryValue> ScdawgInner<V> {
    /// Create a new empty true SCDAWG.
    fn new() -> Self {
        Self {
            nodes: vec![ScdawgNode::root()],
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
        nodes.push(ScdawgNode::root());
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
    fn alloc_node(&mut self, length: usize, suffix_link: usize, first_char: u8) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ScdawgNode::new(length, suffix_link, first_char));
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
    fn sa_extend(&mut self, c: u8, term_idx: usize, pos: usize) {
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
        // For multi-string SA, we need to handle this carefully
        // Option 1: Reset last to root (separate suffix trees)
        // Option 2: Use unique end markers (generalized suffix automaton)

        // We use Option 1 for simplicity - each term builds its own suffix structure
        // connected to the shared root
        self.last = 0;

        // Insert each character
        for (pos, &byte) in term.as_bytes().iter().enumerate() {
            self.sa_extend(byte, term_idx, pos);
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
    /// For each suffix link from node A to node B (representing that B's string
    /// is a suffix of A's string), we add a left extension edge from B to A
    /// that allows prepending the distinguishing character.
    ///
    /// The key insight is that if A represents string "xyz" and B represents "yz",
    /// then prepending 'x' to B's string gives A's string. So the left extension
    /// edge label is 'x' - the FIRST character of A's canonical string.
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
                // The label is the FIRST character of the canonical string
                // This allows prepending that character to extend leftward
                let label = self.nodes[node_idx].first_char;
                self.nodes[suffix_target].left_edges.push((label, node_idx));
            }
        }

        self.left_edges_computed = true;
    }

    /// Find exact substring matches using O(|pattern|) traversal.
    ///
    /// This is the KEY improvement over the naive implementation.
    fn find_substring_fast(&self, pattern: &str) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0); // Empty pattern matches at root
        }

        let mut current = 0; // Start at root
        for &byte in pattern.as_bytes() {
            match self.nodes[current].get_edge(byte) {
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
    /// Returns (term, position) pairs.
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

        // Now enumerate all final states reachable from this node
        // and collect the terms/positions
        let pattern_len = pattern.len();
        let mut results = Vec::new();

        // DFS to find all final states reachable from end_node
        self.collect_term_positions(end_node, pattern_len, &mut results);

        results
    }

    /// Collect all term positions reachable from a node.
    ///
    /// This traverses all nodes that have the pattern as a suffix. In the suffix
    /// automaton, if node Q has suffix_link to node P, then strings at Q have
    /// strings at P as suffixes. So if P represents pattern "ab", then any node
    /// whose suffix_link chain leads to P also contains "ab" as a suffix.
    ///
    /// We use left_edges (inverse of suffix links) to traverse from P to all
    /// nodes Q where the pattern occurs.
    fn collect_term_positions(
        &self,
        node: usize,
        pattern_len: usize,
        results: &mut Vec<(String, usize)>,
    ) {
        // Check if this node has term endings
        // Each term_ends entry (term_idx, end_pos) means a string of this equivalence
        // class ends at position end_pos in term term_idx.
        // The pattern (of length pattern_len) that we searched for starts at:
        // start_pos = end_pos + 1 - pattern_len
        for &(term_idx, end_pos) in &self.nodes[node].term_ends {
            if end_pos + 1 >= pattern_len {
                let start_pos = end_pos + 1 - pattern_len;
                if term_idx < self.terms.len() {
                    results.push((self.terms[term_idx].clone(), start_pos));
                }
            }
        }

        // Traverse via left edges (inverse suffix links) to find all nodes
        // that have this pattern as a suffix. Those nodes' term_ends also
        // contain occurrences of the pattern.
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

    /// Iterate over all terms.
    fn iter_terms(&self) -> impl Iterator<Item = &String> {
        self.terms.iter()
    }

    // ========================================================================
    // IS Features Helper Methods
    // ========================================================================

    /// Get the frequency (occurrence count) of a substring pattern.
    fn frequency(&self, pattern: &str) -> usize {
        if pattern.is_empty() {
            // Empty pattern matches at every position in every term
            return self.terms.iter().map(|t| t.len() + 1).sum();
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
    ///
    /// This traverses via left edges (inverse suffix links) to find all nodes
    /// that have the pattern at this node as a suffix, and counts all term_ends
    /// entries, giving the total occurrence count.
    fn count_occurrences(&self, node: usize, count: &mut usize) {
        // Count direct occurrences at this node
        *count += self.nodes[node].term_ends.len();

        // Traverse via left edges to find all nodes where this pattern occurs
        for &(_, target) in &self.nodes[node].left_edges {
            self.count_occurrences(target, count);
        }
    }
}

// ============================================================================
// Public True SCDAWG Type
// ============================================================================

/// True Symmetric Compact DAWG with O(|pattern|) substring search.
///
/// This is a proper suffix automaton implementation that indexes ALL substrings
/// of all terms, enabling efficient substring search and bidirectional extension.
#[derive(Clone, Debug)]
pub struct Scdawg<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgInner<V>>>,
}

impl<V: DictionaryValue> Default for Scdawg<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> Scdawg<V> {
    /// Create a new empty true SCDAWG.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ScdawgInner::new())),
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
        let total_chars: usize = terms_vec.iter().map(|s| s.as_ref().len()).sum();

        let inner = ScdawgInner::with_capacity(term_count, total_chars);
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

    // ========================================================================
    // IS Features (Blumer et al. 1987)
    // ========================================================================

    /// Find a substring and return a handle to its SCDAWG state.
    ///
    /// This is the `find(x)` operation from Blumer et al. (1987).
    /// Returns `None` if the pattern is not a substring of any term.
    ///
    /// # Time Complexity
    ///
    /// O(|pattern|) - linear in pattern length.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scdawg = Scdawg::<()>::from_terms(["cathedral", "category"]);
    /// if let Some(handle) = scdawg.find("cat") {
    ///     println!("Pattern 'cat' found, frequency: {}", scdawg.freq_at(&handle));
    /// }
    /// ```
    pub fn find(&self, pattern: &str) -> Option<ScdawgNodeHandle<V>> {
        let inner = self.inner.read();
        inner.find_substring_fast(pattern).map(|node_idx| ScdawgNodeHandle {
            inner: Arc::clone(&self.inner),
            node_idx,
        })
    }

    /// Get the frequency (occurrence count) of a substring pattern.
    ///
    /// This is the `freq(x)` operation from Blumer et al. (1987).
    /// Returns the total number of occurrences across all terms.
    ///
    /// # Time Complexity
    ///
    /// O(|pattern| + k) where k is the number of occurrences.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scdawg = Scdawg::<()>::from_terms(["abab", "bab"]);
    /// assert_eq!(scdawg.freq("ab"), 3); // 2 in "abab" + 1 in "bab"
    /// ```
    pub fn freq(&self, pattern: &str) -> usize {
        let inner = self.inner.read();
        inner.frequency(pattern)
    }

    /// Get the frequency at a specific SCDAWG node handle.
    ///
    /// Use this with `find()` for efficient repeated frequency queries.
    pub fn freq_at(&self, handle: &ScdawgNodeHandle<V>) -> usize {
        let inner = self.inner.read();
        let mut count = 0;
        inner.count_occurrences(handle.node_idx, &mut count);
        count
    }

    /// Get all occurrence locations of a substring pattern.
    ///
    /// This is the `locations(x)` operation from Blumer et al. (1987).
    /// Returns (term, start_position) pairs for every occurrence.
    ///
    /// # Time Complexity
    ///
    /// O(|pattern| + k) where k is the number of occurrences.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scdawg = Scdawg::<()>::from_terms(["abab"]);
    /// let locs = scdawg.locations("ab");
    /// // Returns: [("abab", 0), ("abab", 2)]
    /// ```
    pub fn locations(&self, pattern: &str) -> Vec<(String, usize)> {
        let inner = self.inner.read();
        inner.find_exact_substring(pattern)
    }

    /// Get all occurrence locations from a specific SCDAWG node handle.
    ///
    /// Use this with `find()` for efficient repeated location queries.
    pub fn locations_at(&self, handle: &ScdawgNodeHandle<V>, pattern_len: usize) -> Vec<(String, usize)> {
        let inner = self.inner.read();
        let mut results = Vec::new();
        inner.collect_term_positions(handle.node_idx, pattern_len, &mut results);
        results
    }
}

// ============================================================================
// Dictionary Trait Implementation
// ============================================================================

impl<V: DictionaryValue> Dictionary for Scdawg<V> {
    type Node = ScdawgNodeHandle<V>;

    fn len(&self) -> Option<usize> {
        Some(self.inner.read().term_count())
    }

    fn contains(&self, term: &str) -> bool {
        self.inner.read().contains(term)
    }

    fn root(&self) -> Self::Node {
        ScdawgNodeHandle {
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

/// Handle to a node in the true SCDAWG.
#[derive(Clone)]
pub struct ScdawgNodeHandle<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgInner<V>>>,
    node_idx: usize,
}

impl<V: DictionaryValue> std::fmt::Debug for ScdawgNodeHandle<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScdawgNodeHandle")
            .field("node_idx", &self.node_idx)
            .finish()
    }
}

impl<V: DictionaryValue> DictionaryNode for ScdawgNodeHandle<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].is_final
    }

    fn transition(&self, label: u8) -> Option<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .get_edge(label)
            .map(|idx| ScdawgNodeHandle {
                inner: Arc::clone(&self.inner),
                node_idx: idx,
            })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .forward_edges
            .iter()
            .map(|&(label, idx)| {
                (
                    label,
                    ScdawgNodeHandle {
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

unsafe impl<V: DictionaryValue> Send for ScdawgNodeHandle<V> {}
unsafe impl<V: DictionaryValue> Sync for ScdawgNodeHandle<V> {}

// ============================================================================
// BidirectionalDictionaryNode Implementation
// ============================================================================

impl<V: DictionaryValue> BidirectionalDictionaryNode for ScdawgNodeHandle<V> {
    fn parent(&self) -> Option<Self> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NIL {
            None
        } else {
            Some(ScdawgNodeHandle {
                inner: Arc::clone(&self.inner),
                node_idx: node.parent,
            })
        }
    }

    fn parent_label(&self) -> Option<u8> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NIL {
            None
        } else {
            Some(node.parent_label)
        }
    }

    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .left_edges
            .iter()
            .map(|&(label, idx)| {
                (
                    label,
                    ScdawgNodeHandle {
                        inner: Arc::clone(&self.inner),
                        node_idx: idx,
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }

    fn reverse_transition(&self, label: u8) -> Vec<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .left_edges
            .iter()
            .filter(|(l, _)| *l == label)
            .map(|(_, idx)| ScdawgNodeHandle {
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

impl<V: DictionaryValue> SubstringDictionary for Scdawg<V> {
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>> {
        let inner = self.inner.read();
        let occurrences = inner.find_exact_substring(pattern);

        occurrences
            .into_iter()
            .map(|(term, position)| {
                // Find the node at the end of the pattern match
                let mut node_idx = 0;
                for &byte in term.as_bytes().iter().take(position + pattern.len()) {
                    if let Some(next) = inner.nodes[node_idx].get_edge(byte) {
                        node_idx = next;
                    }
                }

                SubstringMatch::new(
                    ScdawgNodeHandle {
                        inner: Arc::clone(&self.inner),
                        node_idx,
                    },
                    term,
                    position,
                    pattern.len(),
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
    fn test_scdawg_empty() {
        let scdawg = Scdawg::<()>::new();
        assert_eq!(scdawg.term_count(), 0);
        assert!(!scdawg.contains("anything"));
    }

    #[test]
    fn test_scdawg_insert_single() {
        let scdawg = Scdawg::<()>::new();
        assert!(scdawg.insert("hello"));
        assert!(!scdawg.insert("hello")); // Duplicate
        assert_eq!(scdawg.term_count(), 1);
        assert!(scdawg.contains("hello"));
    }

    #[test]
    fn test_scdawg_substring_search() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);

        // Test substring existence
        assert!(scdawg.contains_substring("cat"));
        assert!(scdawg.contains_substring("the"));
        assert!(scdawg.contains_substring("edral"));
        assert!(scdawg.contains_substring("gory"));
        assert!(!scdawg.contains_substring("xyz"));
    }

    #[test]
    fn test_scdawg_find_exact_substring() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello", "world"]);

        let matches = scdawg.find_exact_substring("hello");
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.term == "hello" && m.position == 0));
    }

    #[test]
    fn test_scdawg_internal_substring() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cathedral"]);

        // Test internal substrings
        assert!(scdawg.contains_substring("thedr"));
        assert!(scdawg.contains_substring("hedr"));
        assert!(scdawg.contains_substring("edra"));
    }

    #[test]
    fn test_scdawg_multiple_terms() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abc", "bcd", "cde"]);

        // Each term should be found
        assert!(scdawg.contains("abc"));
        assert!(scdawg.contains("bcd"));
        assert!(scdawg.contains("cde"));

        // Common substrings
        assert!(scdawg.contains_substring("bc")); // In abc and bcd
        assert!(scdawg.contains_substring("cd")); // In bcd and cde
    }

    #[test]
    fn test_scdawg_iter() {
        let terms = vec!["apple", "banana", "cherry"];
        let scdawg = Scdawg::<()>::from_terms(terms.clone());

        let collected: Vec<_> = scdawg.iter().collect();
        assert_eq!(collected.len(), 3);
        for term in terms {
            assert!(collected.contains(&term.to_string()));
        }
    }

    /// Test that left extension edges are computed from suffix links.
    ///
    /// Left extension edges are derived from suffix links: if node A has a suffix link
    /// to node B, then B gets a left extension edge pointing to A with label = A's first_char.
    ///
    /// For a single term "abc", all suffix states collapse into equivalence classes,
    /// so no intermediate nodes have suffix links pointing to them. Left extension
    /// edges only appear when multiple terms share suffixes.
    #[test]
    fn test_left_extension_edges() {
        use crate::dictionary::substring::BidirectionalDictionaryNode;
        use crate::dictionary::Dictionary;

        // For left extension edges to exist, we need multiple terms sharing suffixes.
        // "abc" and "dbc" both end in "bc", so the node representing "bc" should have
        // left extension edges for both 'a' (to "abc") and 'd' (to "dbc").
        let scdawg = Scdawg::<()>::from_terms(vec!["abc", "dbc"]);

        // Navigate to the node representing "bc" via root -> 'b' -> 'c'
        let root = scdawg.root();
        let node_b = root.transition(b'b').expect("Should have edge 'b' from root");
        let node_bc = node_b.transition(b'c').expect("Should have edge 'c' from 'b'");

        // The left extension edges from "bc" should have labels 'a' and 'd'
        let left_edges: Vec<_> = node_bc.reverse_edges().collect();
        let labels: std::collections::HashSet<_> = left_edges.iter().map(|(l, _)| *l).collect();

        // Check for left extension edge with label 'a' (from "abc" suffix linking to "bc")
        assert!(
            labels.contains(&b'a'),
            "Node 'bc' should have left extension edge with label 'a'. \
             Found edges: {:?}",
            left_edges.iter().map(|(l, _)| *l as char).collect::<Vec<_>>()
        );

        // Check for left extension edge with label 'd' (from "dbc" suffix linking to "bc")
        assert!(
            labels.contains(&b'd'),
            "Node 'bc' should have left extension edge with label 'd'. \
             Found edges: {:?}",
            left_edges.iter().map(|(l, _)| *l as char).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // IS Features Tests (Blumer et al. 1987)
    // =========================================================================

    #[test]
    fn debug_abab_structure() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abab"]);
        let inner = scdawg.inner.read();

        // Print all nodes with term_ends
        eprintln!("\nNode structure for 'abab':");
        for (i, node) in inner.nodes.iter().enumerate() {
            eprintln!(
                "Node {}: length={}, term_ends={:?}, edges={:?}",
                i,
                node.length,
                node.term_ends,
                node.forward_edges
                    .iter()
                    .map(|(l, t)| (*l as char, *t))
                    .collect::<Vec<_>>()
            );
        }

        // Navigate to "ab" and check what we find
        let ab_node = inner.find_substring_fast("ab").unwrap();
        eprintln!("\nNode for 'ab': {}", ab_node);
        eprintln!("term_ends at 'ab': {:?}", inner.nodes[ab_node].term_ends);
        eprintln!("children of 'ab': {:?}", inner.nodes[ab_node].forward_edges);

        // Try counting manually
        let mut results = Vec::new();
        inner.collect_term_positions(ab_node, 2, &mut results);
        eprintln!("\nCollected positions: {:?}", results);
    }

    #[test]
    fn test_is_find() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cathedral", "category"]);

        // Should find common prefix
        assert!(scdawg.find("cat").is_some());

        // Should find internal substring
        assert!(scdawg.find("the").is_some());

        // Should not find non-existent pattern
        assert!(scdawg.find("xyz").is_none());
    }

    #[test]
    fn test_is_freq_single_term() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abab"]);

        // "ab" appears twice in "abab": at positions 0 and 2
        assert_eq!(scdawg.freq("ab"), 2, "Pattern 'ab' should appear twice in 'abab'");

        // "a" appears twice in "abab": at positions 0 and 2
        assert_eq!(scdawg.freq("a"), 2, "Pattern 'a' should appear twice in 'abab'");

        // "b" appears twice in "abab": at positions 1 and 3
        assert_eq!(scdawg.freq("b"), 2, "Pattern 'b' should appear twice in 'abab'");

        // "abab" appears once
        assert_eq!(scdawg.freq("abab"), 1, "Pattern 'abab' should appear once");

        // Non-existent pattern
        assert_eq!(scdawg.freq("xyz"), 0, "Non-existent pattern should have freq 0");
    }

    #[test]
    fn test_is_freq_multiple_terms() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abc", "bcd", "cde"]);

        // "bc" appears in "abc" (pos 1) and "bcd" (pos 0) = 2 occurrences
        assert_eq!(scdawg.freq("bc"), 2, "Pattern 'bc' should appear twice");

        // "cd" appears in "bcd" (pos 1) and "cde" (pos 0) = 2 occurrences
        assert_eq!(scdawg.freq("cd"), 2, "Pattern 'cd' should appear twice");

        // "c" appears in all three terms
        assert_eq!(scdawg.freq("c"), 3, "Pattern 'c' should appear three times");
    }

    #[test]
    fn test_is_locations() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abab"]);

        let locs = scdawg.locations("ab");

        // Should find "ab" at positions 0 and 2 in "abab"
        assert_eq!(locs.len(), 2, "Should find 2 occurrences of 'ab'");

        let positions: std::collections::HashSet<_> = locs.iter().map(|(_, pos)| *pos).collect();
        assert!(positions.contains(&0), "Should find 'ab' at position 0");
        assert!(positions.contains(&2), "Should find 'ab' at position 2");
    }

    #[test]
    fn test_is_locations_multiple_terms() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cat", "cathedral", "scatter"]);

        let locs = scdawg.locations("cat");

        // Debug: print what we found
        eprintln!("\nLocations of 'cat': {:?}", locs);

        // "cat" appears at:
        // - "cat" position 0
        // - "cathedral" position 0
        // - "scatter" position 2
        let term_positions: std::collections::HashSet<_> =
            locs.iter().map(|(term, pos)| (term.as_str(), *pos)).collect();

        assert!(term_positions.contains(&("cat", 0)), "Should find 'cat' at position 0 in 'cat'");
        assert!(term_positions.contains(&("cathedral", 0)), "Should find 'cat' at position 0 in 'cathedral'");

        // Note: "scatter" contains "cat" starting at position 2 (s-c-a-t-t-e-r, indices 2,3,4)
        // Wait, let me verify: "scatter" = s(0) c(1) a(2) t(3) t(4) e(5) r(6)
        // So "cat" would be at positions... c(1) a(2) t(3), starting at index 1, not 2!
        // Let me fix the test
        assert!(term_positions.contains(&("scatter", 1)),
            "Should find 'cat' at position 1 in 'scatter'. Found: {:?}", term_positions);
    }

    #[test]
    fn test_is_freq_at_and_locations_at() {
        let scdawg = Scdawg::<()>::from_terms(vec!["abab", "bab"]);

        // First find the pattern
        let handle = scdawg.find("ab").expect("Should find 'ab'");

        // Then get frequency at that handle
        let freq = scdawg.freq_at(&handle);
        assert!(freq >= 2, "Should have at least 2 occurrences of 'ab'");

        // And locations at that handle
        let locs = scdawg.locations_at(&handle, 2);
        assert!(!locs.is_empty(), "Should have locations for 'ab'");
    }

    /// Test left extensions with multiple terms sharing suffixes
    #[test]
    fn test_left_extension_multiple_terms() {
        use crate::dictionary::substring::BidirectionalDictionaryNode;
        use crate::dictionary::Dictionary;

        // "abc" and "xbc" share suffix "bc"
        let scdawg = Scdawg::<()>::from_terms(vec!["abc", "xbc"]);

        // Navigate to "bc" node
        let root = scdawg.root();
        let node_b = root.transition(b'b').expect("Should have edge 'b'");
        let node_bc = node_b.transition(b'c').expect("Should have edge 'c'");

        // "bc" should have left extensions for both 'a' (-> "abc") and 'x' (-> "xbc")
        let left_edges: Vec<_> = node_bc.reverse_edges().collect();
        let labels: std::collections::HashSet<_> = left_edges.iter().map(|(l, _)| *l).collect();

        assert!(
            labels.contains(&b'a'),
            "Node 'bc' should have left extension 'a' -> 'abc'"
        );
        assert!(
            labels.contains(&b'x'),
            "Node 'bc' should have left extension 'x' -> 'xbc'"
        );
    }

}
