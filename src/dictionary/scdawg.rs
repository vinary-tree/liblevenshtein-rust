//! Symmetric Compact DAWG (SCDAWG) for WallBreaker algorithm.
//!
//! This module implements an SCDAWG (Symmetric Compact Directed Acyclic Word Graph),
//! which is the foundational data structure for the WallBreaker approximate string
//! matching algorithm.
//!
//! # Overview
//!
//! An SCDAWG is a compact DAWG extended with:
//! - **Suffix links**: For traversing to shorter suffixes (construction/substring search)
//! - **Parent links**: For traversing backward toward root (bidirectional extension)
//! - **Symmetric structure**: Enabling efficient bidirectional pattern matching
//!
//! # Key Features
//!
//! - **Substring Search**: O(|pattern|) time to find pattern location
//! - **Bidirectional Traversal**: Forward (root→leaves) and backward (node→root)
//! - **Linear Construction**: O(n) time and space via Inenaga et al. algorithm
//! - **WallBreaker Support**: Full support for left/right extension phases
//!
//! # Use Cases
//!
//! The SCDAWG is specifically designed for:
//!
//! 1. **WallBreaker Algorithm**: Finding approximate matches with large error bounds
//! 2. **Substring Search**: Finding all occurrences of a pattern in the dictionary
//! 3. **Bidirectional Extension**: Extending matches left and right from anchor points
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//! use liblevenshtein::dictionary::SubstringDictionary;
//!
//! // Create an SCDAWG from terms
//! let scdawg = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);
//!
//! // Find substring matches
//! let matches = scdawg.find_exact_substring("ther");
//! for m in &matches {
//!     println!("Found '{}' at position {} in '{}'",
//!         m.matched_substring(), m.position, m.term);
//! }
//! ```
//!
//! # Algorithm Reference
//!
//! The SCDAWG construction follows the online algorithm from:
//! - Inenaga et al. (2005): "On-Line Construction of Compact Directed Acyclic Word Graphs"
//!
//! The WallBreaker algorithm that uses SCDAWG is from:
//! - Gerdjikov et al. (2013): "WallBreaker - overcoming the wall effect in similarity search"
//!
//! # Thread Safety
//!
//! Uses `Arc<RwLock<...>>` for interior mutability, following the same pattern as
//! `DynamicDawg`. Safe for concurrent reads, exclusive writes.

use std::collections::HashMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// DictionaryIterator is used with zipper-based dictionaries
// SCDAWG provides its own iter() method directly
use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringDictionary, SubstringMatch};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
use crate::sync_compat::RwLock;

/// SCDAWG node with bidirectional links.
///
/// Each node represents an equivalence class of substrings (like suffix automaton),
/// but with additional parent links for backward traversal needed by WallBreaker.
///
/// # Memory Layout (64-bit)
///
/// | Field | Size (bytes) | Notes |
/// |-------|--------------|-------|
/// | forward_edges | 24 | SmallVec<[(u8, usize); 4]> |
/// | backward_edges | 24 | SmallVec<[(u8, SmallVec<[usize; 2]>); 2]> |
/// | suffix_link | 16 | Option<usize> |
/// | parent | 8 | usize (u32::MAX = none) |
/// | parent_label | 1 | u8 (0 if no parent) |
/// | max_length | 8 | usize |
/// | min_length | 8 | usize |
/// | depth | 8 | usize |
/// | is_final | 1 | bool |
/// | ref_count | 8 | usize |
/// | value | 8+ | Option<V> |
///
/// Total: ~115 bytes per node (actual may vary due to alignment)
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct ScdawgNode<V: DictionaryValue = ()> {
    /// Forward edges: (byte label, target node index).
    ///
    /// Kept sorted by label for efficient binary search.
    /// SmallVec avoids heap allocation for nodes with ≤4 edges (common case).
    pub(crate) forward_edges: SmallVec<[(u8, usize); 4]>,

    /// Backward edges: for each label, list of parent node indices.
    ///
    /// This is the inverse of forward_edges: if parent has edge (label, self),
    /// then self.backward_edges contains (label, [parent, ...]).
    ///
    /// Multiple parents possible for the same label in DAWG structures.
    pub(crate) backward_edges: SmallVec<[(u8, SmallVec<[usize; 2]>); 2]>,

    /// Suffix link: points to state representing the longest proper suffix
    /// in a different endpos equivalence class.
    ///
    /// Used for construction and substring search enumeration.
    pub(crate) suffix_link: Option<usize>,

    /// Parent node index in the spanning tree (toward root).
    ///
    /// Uses usize::MAX as sentinel for "no parent" (root node).
    /// This is the canonical parent from construction order.
    pub(crate) parent: usize,

    /// Edge label from parent to this node.
    ///
    /// Only meaningful when parent != usize::MAX.
    pub(crate) parent_label: u8,

    /// Maximum length of strings reaching this state.
    ///
    /// All strings in this equivalence class have length ≤ max_length.
    pub(crate) max_length: usize,

    /// Minimum length of strings reaching this state.
    ///
    /// All strings in this equivalence class have length ≥ min_length.
    /// For most nodes: min_length = suffix_link.max_length + 1
    pub(crate) min_length: usize,

    /// Depth from root (number of edges in canonical path).
    ///
    /// Root has depth 0, its children have depth 1, etc.
    pub(crate) depth: usize,

    /// True if this state represents an end-of-word position.
    pub(crate) is_final: bool,

    /// Reference count for dynamic deletion support.
    pub(crate) ref_count: usize,

    /// Optional value associated with this node (only for final nodes).
    pub(crate) value: Option<V>,
}

/// Sentinel value for "no parent" (root node).
const NO_PARENT: usize = usize::MAX;

impl<V: DictionaryValue> ScdawgNode<V> {
    /// Create a new root node.
    fn root() -> Self {
        Self {
            forward_edges: SmallVec::new(),
            backward_edges: SmallVec::new(),
            suffix_link: None,
            parent: NO_PARENT,
            parent_label: 0,
            max_length: 0,
            min_length: 0,
            depth: 0,
            is_final: false,
            ref_count: 1, // Root always has ref_count 1
            value: None,
        }
    }

    /// Create a new non-root node.
    fn new(max_length: usize, min_length: usize) -> Self {
        Self {
            forward_edges: SmallVec::new(),
            backward_edges: SmallVec::new(),
            suffix_link: None,
            parent: NO_PARENT,
            parent_label: 0,
            max_length,
            min_length,
            depth: 0,
            is_final: false,
            ref_count: 0,
            value: None,
        }
    }

    /// Create a new node with parent information.
    fn new_with_parent(
        max_length: usize,
        min_length: usize,
        parent: usize,
        parent_label: u8,
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

    /// Find a forward edge by label.
    ///
    /// Uses linear search for small edge counts, binary search for larger.
    #[inline]
    fn find_forward_edge(&self, label: u8) -> Option<usize> {
        if self.forward_edges.len() < 16 {
            self.forward_edges
                .iter()
                .find(|(b, _)| *b == label)
                .map(|(_, t)| *t)
        } else {
            self.forward_edges
                .binary_search_by_key(&label, |(b, _)| *b)
                .ok()
                .map(|idx| self.forward_edges[idx].1)
        }
    }

    /// Add a forward edge, maintaining sorted order.
    fn add_forward_edge(&mut self, label: u8, target: usize) {
        match self.forward_edges.binary_search_by_key(&label, |(b, _)| *b) {
            Ok(idx) => self.forward_edges[idx].1 = target,
            Err(idx) => self.forward_edges.insert(idx, (label, target)),
        }
    }

    /// Remove a forward edge.
    fn remove_forward_edge(&mut self, label: u8) -> Option<usize> {
        match self.forward_edges.binary_search_by_key(&label, |(b, _)| *b) {
            Ok(idx) => Some(self.forward_edges.remove(idx).1),
            Err(_) => None,
        }
    }

    /// Add a backward edge from a parent node.
    fn add_backward_edge(&mut self, label: u8, parent: usize) {
        // Find existing entry for this label
        for (l, parents) in &mut self.backward_edges {
            if *l == label {
                if !parents.contains(&parent) {
                    parents.push(parent);
                }
                return;
            }
        }
        // No entry for this label, create new
        let mut parents = SmallVec::new();
        parents.push(parent);
        self.backward_edges.push((label, parents));
        // Keep sorted by label
        self.backward_edges.sort_by_key(|(l, _)| *l);
    }

    /// Remove a backward edge from a parent node.
    fn remove_backward_edge(&mut self, label: u8, parent: usize) {
        for (l, parents) in &mut self.backward_edges {
            if *l == label {
                parents.retain(|p| *p != parent);
                return;
            }
        }
    }

    /// Get all backward edges as (label, parent_index) pairs.
    fn backward_edge_iter(&self) -> impl Iterator<Item = (u8, usize)> + '_ {
        self.backward_edges
            .iter()
            .flat_map(|(label, parents)| parents.iter().map(move |&p| (*label, p)))
    }

    /// Find backward edges by label.
    fn find_backward_edges(&self, label: u8) -> Vec<usize> {
        for (l, parents) in &self.backward_edges {
            if *l == label {
                return parents.to_vec();
            }
        }
        Vec::new()
    }

    /// Check if this is the root node.
    #[inline]
    fn is_root(&self) -> bool {
        self.parent == NO_PARENT && self.depth == 0
    }
}

/// Inner mutable state of the SCDAWG.
#[derive(Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct ScdawgInner<V: DictionaryValue> {
    /// All nodes in the SCDAWG. Index 0 is always the root.
    pub(crate) nodes: Vec<ScdawgNode<V>>,

    /// Number of distinct terms in the dictionary.
    term_count: usize,

    /// Last node created during construction (for online algorithm).
    last_node: usize,

    /// Whether the structure needs compaction after deletions.
    needs_compaction: bool,

    /// Suffix cache for efficient suffix sharing during construction.
    #[cfg_attr(feature = "serialization", serde(skip))]
    suffix_cache: FxHashMap<u64, usize>,

    /// Term index: maps terms to their node paths for deletion support.
    #[cfg_attr(feature = "serialization", serde(skip))]
    term_index: HashMap<String, Vec<usize>>,
}

impl<V: DictionaryValue> ScdawgInner<V> {
    /// Create a new empty SCDAWG inner.
    fn new() -> Self {
        Self {
            nodes: vec![ScdawgNode::root()],
            term_count: 0,
            last_node: 0,
            needs_compaction: false,
            suffix_cache: FxHashMap::default(),
            term_index: HashMap::new(),
        }
    }

    /// Get the root node index (always 0).
    #[inline]
    fn root(&self) -> usize {
        0
    }

    /// Allocate a new node and return its index.
    fn alloc_node(&mut self, max_length: usize, min_length: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ScdawgNode::new(max_length, min_length));
        idx
    }

    /// Allocate a new node with parent info and return its index.
    fn alloc_node_with_parent(
        &mut self,
        max_length: usize,
        min_length: usize,
        parent: usize,
        parent_label: u8,
        depth: usize,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ScdawgNode::new_with_parent(
            max_length,
            min_length,
            parent,
            parent_label,
            depth,
        ));
        idx
    }

    /// Add a forward edge and corresponding backward edge.
    fn add_edge(&mut self, from: usize, label: u8, to: usize) {
        self.nodes[from].add_forward_edge(label, to);
        self.nodes[to].add_backward_edge(label, from);
        self.nodes[to].ref_count += 1;
    }

    /// Remove a forward edge and corresponding backward edge.
    fn remove_edge(&mut self, from: usize, label: u8, to: usize) {
        self.nodes[from].remove_forward_edge(label);
        self.nodes[to].remove_backward_edge(label, from);
        if self.nodes[to].ref_count > 0 {
            self.nodes[to].ref_count -= 1;
        }
    }

    /// Insert a term using the Inenaga et al. online algorithm.
    ///
    /// This is based on the suffix automaton construction but maintains
    /// parent links for bidirectional traversal.
    fn insert(&mut self, term: &str) -> bool {
        if term.is_empty() {
            // Handle empty string: just mark root as final
            if self.nodes[0].is_final {
                return false;
            }
            self.nodes[0].is_final = true;
            self.term_count += 1;
            return true;
        }

        let bytes = term.as_bytes();
        let mut current = 0usize; // Start at root
        let mut depth = 0usize;
        let mut path = vec![0usize]; // Track path for term_index

        for &byte in bytes {
            depth += 1;

            if let Some(next) = self.nodes[current].find_forward_edge(byte) {
                // Edge exists, follow it
                current = next;
                self.nodes[current].ref_count += 1;
            } else {
                // Create new node
                let parent_max_length = self.nodes[current].max_length;
                let new_node = self.alloc_node_with_parent(
                    parent_max_length + 1,
                    parent_max_length + 1,
                    current,
                    byte,
                    depth,
                );

                // Add edge (updates backward edge too)
                self.add_edge(current, byte, new_node);

                // Set up suffix link for the new node
                self.setup_suffix_link(current, byte, new_node);

                current = new_node;
            }

            path.push(current);
        }

        // Mark final and set value
        if self.nodes[current].is_final {
            return false; // Term already exists
        }

        self.nodes[current].is_final = true;
        self.term_count += 1;
        self.last_node = current;

        // Store term path for deletion support
        self.term_index.insert(term.to_string(), path);

        true
    }

    /// Set up suffix link for a newly created node.
    ///
    /// Follows suffix links from parent to find the appropriate target.
    fn setup_suffix_link(&mut self, parent: usize, label: u8, new_node: usize) {
        // If parent is root, suffix link goes to root
        if parent == 0 {
            self.nodes[new_node].suffix_link = Some(0);
            return;
        }

        // Follow suffix link of parent
        if let Some(parent_suffix) = self.nodes[parent].suffix_link {
            // Try to find edge with same label from parent's suffix
            if let Some(target) = self.nodes[parent_suffix].find_forward_edge(label) {
                self.nodes[new_node].suffix_link = Some(target);
            } else {
                // No edge, suffix link goes to root
                self.nodes[new_node].suffix_link = Some(0);
            }
        } else {
            // Parent has no suffix link (shouldn't happen after construction)
            self.nodes[new_node].suffix_link = Some(0);
        }
    }

    /// Insert a term with an associated value.
    fn insert_with_value(&mut self, term: &str, value: V) -> bool {
        let inserted = self.insert(term);
        if inserted {
            self.nodes[self.last_node].value = Some(value);
        }
        inserted
    }

    /// Remove a term from the SCDAWG.
    ///
    /// This marks nodes for potential removal but doesn't immediately compact.
    /// Call `compact()` to reclaim space.
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

        // Check if term exists
        if !self.term_index.contains_key(term) {
            // Term not in index, check by traversal
            let mut current = 0;
            for &byte in term.as_bytes() {
                match self.nodes[current].find_forward_edge(byte) {
                    Some(next) => current = next,
                    None => return false,
                }
            }
            if !self.nodes[current].is_final {
                return false;
            }
        }

        // Traverse and decrement ref counts
        let mut current = 0;
        for &byte in term.as_bytes() {
            let next = match self.nodes[current].find_forward_edge(byte) {
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

        // Remove from term index
        self.term_index.remove(term);

        true
    }

    /// Check if the SCDAWG contains a term.
    fn contains(&self, term: &str) -> bool {
        let mut current = 0;
        for &byte in term.as_bytes() {
            match self.nodes[current].find_forward_edge(byte) {
                Some(next) => current = next,
                None => return false,
            }
        }
        self.nodes[current].is_final
    }

    /// Find all occurrences of a substring pattern.
    ///
    /// Returns a list of (term, position) pairs where the pattern was found.
    ///
    /// Note: This is a naive O(n*m) implementation where n is total characters
    /// and m is pattern length. A full SCDAWG implementation would use suffix
    /// links for O(|pattern| + occurrences) complexity.
    fn find_exact_substring(&self, pattern: &str) -> Vec<(String, usize)> {
        if pattern.is_empty() {
            // Empty pattern matches at position 0 of every term
            return self.collect_all_terms().into_iter().map(|t| (t, 0)).collect();
        }

        let pattern_bytes = pattern.as_bytes();
        let mut results = Vec::new();

        // Collect all terms and search each one
        // This is the naive approach - a proper SCDAWG would do this more efficiently
        for term in self.collect_all_terms() {
            let term_bytes = term.as_bytes();

            // Find all occurrences of pattern in term
            for pos in 0..=term_bytes.len().saturating_sub(pattern_bytes.len()) {
                if term_bytes[pos..].starts_with(pattern_bytes) {
                    results.push((term.clone(), pos));
                }
            }
        }

        results
    }

    /// Enumerate all term occurrences that contain the matched pattern.
    fn enumerate_pattern_occurrences(&self, pattern_end_node: usize, pattern_len: usize) -> Vec<(String, usize)> {
        let mut results = Vec::new();

        // Collect all complete paths through pattern_end_node
        self.collect_paths_through_node(pattern_end_node, pattern_len, &mut results);

        results
    }

    /// Collect all complete term paths that go through a specific node.
    fn collect_paths_through_node(
        &self,
        node: usize,
        pattern_len: usize,
        results: &mut Vec<(String, usize)>,
    ) {
        // Get prefix (path from root to this node)
        let prefix = self.path_to_node(node);
        let position = prefix.len().saturating_sub(pattern_len);

        // Enumerate all suffixes from this node (paths to final nodes)
        let mut stack: Vec<(usize, Vec<u8>)> = vec![(node, Vec::new())];

        while let Some((current, suffix)) = stack.pop() {
            if self.nodes[current].is_final {
                // Found a complete term
                let mut term = prefix.clone();
                term.extend(&suffix);
                let term_string = String::from_utf8_lossy(&term).to_string();
                results.push((term_string, position));
            }

            // Continue to children
            for &(label, child) in &self.nodes[current].forward_edges {
                let mut new_suffix = suffix.clone();
                new_suffix.push(label);
                stack.push((child, new_suffix));
            }
        }
    }

    /// Get the path (bytes) from root to a node.
    fn path_to_node(&self, node: usize) -> Vec<u8> {
        let mut path = Vec::new();
        let mut current = node;

        while current != 0 && self.nodes[current].parent != NO_PARENT {
            path.push(self.nodes[current].parent_label);
            current = self.nodes[current].parent;
        }

        path.reverse();
        path
    }

    /// Collect all terms in the SCDAWG.
    fn collect_all_terms(&self) -> Vec<String> {
        let mut terms = Vec::new();
        let mut stack: Vec<(usize, Vec<u8>)> = vec![(0, Vec::new())];

        while let Some((current, path)) = stack.pop() {
            if self.nodes[current].is_final {
                terms.push(String::from_utf8_lossy(&path).to_string());
            }

            for &(label, child) in &self.nodes[current].forward_edges {
                let mut new_path = path.clone();
                new_path.push(label);
                stack.push((child, new_path));
            }
        }

        terms
    }

    /// Compact the SCDAWG by removing unreachable nodes.
    fn compact(&mut self) {
        if !self.needs_compaction {
            return;
        }

        // Mark reachable nodes
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![0usize]; // Start from root
        reachable[0] = true;

        while let Some(node) = stack.pop() {
            for &(_, child) in &self.nodes[node].forward_edges {
                if !reachable[child] {
                    reachable[child] = true;
                    stack.push(child);
                }
            }
        }

        // Build remapping
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut new_idx = 0usize;
        for (old_idx, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                remap[old_idx] = new_idx;
                new_idx += 1;
            }
        }

        // Create new node vector with remapped indices
        let mut new_nodes = Vec::with_capacity(new_idx);
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if !reachable[old_idx] {
                continue;
            }

            let mut new_node = node.clone();

            // Remap forward edges
            for (_, target) in &mut new_node.forward_edges {
                *target = remap[*target];
            }

            // Remap backward edges
            for (_, parents) in &mut new_node.backward_edges {
                for parent in parents {
                    if *parent != NO_PARENT && *parent < remap.len() {
                        *parent = remap[*parent];
                    }
                }
            }

            // Remap parent
            if new_node.parent != NO_PARENT && new_node.parent < remap.len() {
                new_node.parent = remap[new_node.parent];
            }

            // Remap suffix link
            if let Some(ref mut suffix) = new_node.suffix_link {
                if *suffix < remap.len() {
                    *suffix = remap[*suffix];
                }
            }

            new_nodes.push(new_node);
        }

        self.nodes = new_nodes;
        self.needs_compaction = false;

        // Rebuild term index (expensive but necessary)
        self.term_index.clear();
        for term in self.collect_all_terms() {
            let mut path = vec![0usize];
            let mut current = 0;
            for &byte in term.as_bytes() {
                if let Some(next) = self.nodes[current].find_forward_edge(byte) {
                    current = next;
                    path.push(current);
                }
            }
            self.term_index.insert(term, path);
        }
    }
}

/// A Symmetric Compact DAWG (SCDAWG) for WallBreaker algorithm.
///
/// This is the main public type for SCDAWG-based dictionaries. It supports:
/// - Standard dictionary operations (insert, remove, contains)
/// - Substring search (find patterns anywhere in terms)
/// - Bidirectional traversal (for WallBreaker extension)
///
/// # Thread Safety
///
/// Uses `Arc<RwLock<...>>` for interior mutability. Safe for concurrent reads,
/// exclusive writes.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::scdawg::Scdawg;
///
/// // Create and populate
/// let scdawg = Scdawg::<()>::new();
/// scdawg.insert("hello");
/// scdawg.insert("world");
///
/// // Check membership
/// assert!(scdawg.contains("hello"));
/// assert!(!scdawg.contains("missing"));
/// ```
#[derive(Clone, Debug)]
pub struct Scdawg<V: DictionaryValue = ()> {
    pub(crate) inner: Arc<RwLock<ScdawgInner<V>>>,
}

impl<V: DictionaryValue> Scdawg<V> {
    /// Create a new empty SCDAWG.
    pub fn new() -> Self {
        Scdawg {
            inner: Arc::new(RwLock::new(ScdawgInner::new())),
        }
    }

    /// Create an SCDAWG from an iterator of terms.
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let scdawg = Scdawg::new();
        for term in terms {
            scdawg.insert(term.as_ref());
        }
        scdawg
    }

    /// Create an SCDAWG from an iterator of (term, value) pairs.
    pub fn from_terms_with_values<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        S: AsRef<str>,
    {
        let scdawg = Scdawg::new();
        for (term, value) in terms {
            scdawg.insert_with_value(term.as_ref(), value);
        }
        scdawg
    }

    /// Insert a term into the SCDAWG.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    pub fn insert(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        inner.insert(term)
    }

    /// Insert a term with an associated value.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write();
        inner.insert_with_value(term, value)
    }

    /// Remove a term from the SCDAWG.
    ///
    /// Returns `true` if the term was removed, `false` if it didn't exist.
    pub fn remove(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        inner.remove(term)
    }

    /// Check if the SCDAWG needs compaction.
    ///
    /// Returns `true` if `remove()` has been called and `compact()` hasn't.
    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read();
        inner.needs_compaction
    }

    /// Compact the SCDAWG by removing unreachable nodes.
    ///
    /// Call this periodically after removals to reclaim memory.
    pub fn compact(&self) {
        let mut inner = self.inner.write();
        inner.compact();
    }

    /// Get the number of terms in the SCDAWG.
    pub fn term_count(&self) -> usize {
        let inner = self.inner.read();
        inner.term_count
    }

    /// Get the number of nodes in the SCDAWG.
    pub fn node_count(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes.len()
    }

    /// Get an iterator over all terms in the SCDAWG.
    pub fn iter(&self) -> impl Iterator<Item = String> {
        let inner = self.inner.read();
        inner.collect_all_terms().into_iter()
    }

    /// Get an iterator over all (term, value) pairs.
    pub fn iter_with_values(&self) -> impl Iterator<Item = (String, V)>
    where
        V: Clone,
    {
        let inner = self.inner.read();
        let mut results = Vec::new();

        let mut stack: Vec<(usize, Vec<u8>)> = vec![(0, Vec::new())];
        while let Some((current, path)) = stack.pop() {
            if inner.nodes[current].is_final {
                let term = String::from_utf8_lossy(&path).to_string();
                if let Some(ref value) = inner.nodes[current].value {
                    results.push((term, value.clone()));
                }
            }

            for &(label, child) in &inner.nodes[current].forward_edges {
                let mut new_path = path.clone();
                new_path.push(label);
                stack.push((child, new_path));
            }
        }

        results.into_iter()
    }

    /// Get the value associated with a term.
    pub fn get_value(&self, term: &str) -> Option<V>
    where
        V: Clone,
    {
        let inner = self.inner.read();
        let mut current = 0;
        for &byte in term.as_bytes() {
            match inner.nodes[current].find_forward_edge(byte) {
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

impl<V: DictionaryValue> Default for Scdawg<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dictionary trait implementations
// ============================================================================

/// Node wrapper for SCDAWG dictionary traversal.
#[derive(Clone)]
pub struct ScdawgNode2<V: DictionaryValue = ()> {
    inner: Arc<RwLock<ScdawgInner<V>>>,
    node_idx: usize,
}

impl<V: DictionaryValue> std::fmt::Debug for ScdawgNode2<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("ScdawgNode2")
            .field("node_idx", &self.node_idx)
            .field("is_final", &inner.nodes[self.node_idx].is_final)
            .field("depth", &inner.nodes[self.node_idx].depth)
            .finish()
    }
}

impl<V: DictionaryValue> Dictionary for Scdawg<V> {
    type Node = ScdawgNode2<V>;

    fn root(&self) -> Self::Node {
        ScdawgNode2 {
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

impl<V: DictionaryValue> DictionaryNode for ScdawgNode2<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        let inner = self.inner.read();
        inner.nodes[self.node_idx].is_final
    }

    fn transition(&self, label: u8) -> Option<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .find_forward_edge(label)
            .map(|idx| ScdawgNode2 {
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
                    ScdawgNode2 {
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

unsafe impl<V: DictionaryValue> Send for ScdawgNode2<V> {}
unsafe impl<V: DictionaryValue> Sync for ScdawgNode2<V> {}

// ============================================================================
// BidirectionalDictionaryNode implementation
// ============================================================================

impl<V: DictionaryValue> BidirectionalDictionaryNode for ScdawgNode2<V> {
    fn parent(&self) -> Option<Self> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NO_PARENT {
            None
        } else {
            Some(ScdawgNode2 {
                inner: Arc::clone(&self.inner),
                node_idx: node.parent,
            })
        }
    }

    fn parent_label(&self) -> Option<u8> {
        let inner = self.inner.read();
        let node = &inner.nodes[self.node_idx];
        if node.parent == NO_PARENT {
            None
        } else {
            Some(node.parent_label)
        }
    }

    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
        let inner = self.inner.read();
        let edges: Vec<_> = inner.nodes[self.node_idx]
            .backward_edge_iter()
            .map(|(label, parent_idx)| {
                (
                    label,
                    ScdawgNode2 {
                        inner: Arc::clone(&self.inner),
                        node_idx: parent_idx,
                    },
                )
            })
            .collect();
        Box::new(edges.into_iter())
    }

    fn reverse_transition(&self, label: u8) -> Vec<Self> {
        let inner = self.inner.read();
        inner.nodes[self.node_idx]
            .find_backward_edges(label)
            .into_iter()
            .map(|idx| ScdawgNode2 {
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
                    if let Some(next) = inner.nodes[node_idx].find_forward_edge(byte) {
                        node_idx = next;
                    }
                }

                SubstringMatch::new(
                    ScdawgNode2 {
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
    fn test_scdawg_insert_multiple() {
        let scdawg = Scdawg::<()>::from_terms(vec!["apple", "application", "apply"]);
        assert_eq!(scdawg.term_count(), 3);
        assert!(scdawg.contains("apple"));
        assert!(scdawg.contains("application"));
        assert!(scdawg.contains("apply"));
        assert!(!scdawg.contains("app")); // Prefix, not a term
    }

    #[test]
    fn test_scdawg_remove() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        assert!(scdawg.remove("hello"));
        assert!(!scdawg.remove("hello")); // Already removed
        assert_eq!(scdawg.term_count(), 1);
        assert!(!scdawg.contains("hello"));
        assert!(scdawg.contains("world"));
    }

    #[test]
    fn test_scdawg_with_values() {
        let scdawg = Scdawg::<u32>::new();
        scdawg.insert_with_value("hello", 42);
        scdawg.insert_with_value("world", 99);

        assert_eq!(scdawg.get_value("hello"), Some(42));
        assert_eq!(scdawg.get_value("world"), Some(99));
        assert_eq!(scdawg.get_value("missing"), None);
    }

    #[test]
    fn test_scdawg_dictionary_trait() {
        let scdawg = Scdawg::<()>::from_terms(vec!["test", "testing", "tested"]);

        let root = scdawg.root();
        assert!(!root.is_final());

        // Traverse "test"
        let t = root.transition(b't').expect("t");
        let e = t.transition(b'e').expect("e");
        let s = e.transition(b's').expect("s");
        let t2 = s.transition(b't').expect("t");
        assert!(t2.is_final());
    }

    #[test]
    fn test_scdawg_bidirectional() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello"]);

        // Get to the 'o' node
        let root = scdawg.root();
        let h = root.transition(b'h').unwrap();
        let e = h.transition(b'e').unwrap();
        let l1 = e.transition(b'l').unwrap();
        let l2 = l1.transition(b'l').unwrap();
        let o = l2.transition(b'o').unwrap();

        assert!(o.is_final());

        // Walk back to root
        let back_l2 = o.parent().expect("parent of o");
        assert_eq!(o.parent_label(), Some(b'o'));

        let back_l1 = back_l2.parent().expect("parent of l2");
        assert_eq!(back_l2.parent_label(), Some(b'l'));

        let back_e = back_l1.parent().expect("parent of l1");
        let back_h = back_e.parent().expect("parent of e");
        let back_root = back_h.parent().expect("parent of h");

        assert!(back_root.parent().is_none()); // Root has no parent
    }

    #[test]
    fn test_scdawg_depth() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello"]);

        let root = scdawg.root();
        assert_eq!(root.depth(), 0);

        let h = root.transition(b'h').unwrap();
        assert_eq!(h.depth(), 1);

        let e = h.transition(b'e').unwrap();
        assert_eq!(e.depth(), 2);
    }

    #[test]
    fn test_scdawg_path_string() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello"]);

        let root = scdawg.root();
        let h = root.transition(b'h').unwrap();
        let e = h.transition(b'e').unwrap();
        let l1 = e.transition(b'l').unwrap();
        let l2 = l1.transition(b'l').unwrap();
        let o = l2.transition(b'o').unwrap();

        assert_eq!(o.path_string(), "hello");
        assert_eq!(e.path_string(), "he");
        assert_eq!(root.path_string(), "");
    }

    #[test]
    fn test_scdawg_substring_search_simple() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cathedral"]);
        let matches = scdawg.find_exact_substring("thedr");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].term, "cathedral");
        assert_eq!(matches[0].position, 2);
        assert_eq!(matches[0].length, 5);
    }

    #[test]
    fn test_scdawg_substring_search_multiple() {
        let scdawg = Scdawg::<()>::from_terms(vec!["cathedral", "cathedrals"]);
        let matches = scdawg.find_exact_substring("thedr");

        assert_eq!(matches.len(), 2);
        // Both matches should be at position 2
        for m in &matches {
            assert_eq!(m.position, 2);
            assert_eq!(m.length, 5);
            assert!(m.term == "cathedral" || m.term == "cathedrals");
        }
    }

    #[test]
    fn test_scdawg_substring_search_not_found() {
        let scdawg = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let matches = scdawg.find_exact_substring("xyz");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_scdawg_compact() {
        // Use terms with no shared prefixes so removal actually orphans nodes
        let scdawg = Scdawg::<()>::from_terms(vec!["aaa", "bbb", "ccc"]);

        let before_nodes = scdawg.node_count();
        assert_eq!(scdawg.term_count(), 3);

        scdawg.remove("aaa");
        scdawg.remove("bbb");
        assert!(scdawg.needs_compaction());

        scdawg.compact();
        assert!(!scdawg.needs_compaction());

        let after_nodes = scdawg.node_count();
        // After compaction, unreachable nodes should be removed
        // ccc needs 4 nodes: root + c + c + c
        assert!(after_nodes <= before_nodes, "after={} should be <= before={}", after_nodes, before_nodes);
        assert!(scdawg.contains("ccc"));
        assert!(!scdawg.contains("aaa"));
        assert!(!scdawg.contains("bbb"));
    }

    #[test]
    fn test_scdawg_iter() {
        let terms = vec!["alpha", "beta", "gamma"];
        let scdawg = Scdawg::<()>::from_terms(terms.clone());

        let mut collected: Vec<String> = scdawg.iter().collect();
        collected.sort();

        let mut expected: Vec<String> = terms.iter().map(|s| s.to_string()).collect();
        expected.sort();

        assert_eq!(collected, expected);
    }

    #[test]
    fn test_scdawg_empty_term() {
        let scdawg = Scdawg::<()>::new();
        scdawg.insert("");
        assert!(scdawg.contains(""));
        assert_eq!(scdawg.term_count(), 1);

        scdawg.remove("");
        assert!(!scdawg.contains(""));
        assert_eq!(scdawg.term_count(), 0);
    }
}
