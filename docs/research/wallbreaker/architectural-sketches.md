# WallBreaker Architectural Sketches

**Date**: 2025-11-06
**Purpose**: Detailed code designs, trait definitions, struct layouts, and pseudo-code for WallBreaker implementation.

---

## Executive Summary

This document provides concrete code sketches for implementing WallBreaker (Option B - Hybrid approach). All code shown here is **design-level pseudo-Rust** intended to guide implementation, not final production code.

**Key Components**:
1. New traits: `SubstringDictionary`, `BidirectionalDictionaryNode`
2. Extended SuffixAutomaton with parent links
3. Bidirectional state management
4. Pattern splitting logic
5. Hybrid extension algorithm
6. WallBreaker query iterator

---

## Table of Contents

1. [New Trait Definitions](#new-trait-definitions)
2. [SuffixAutomaton Extensions](#suffixautomaton-extensions)
3. [Bidirectional State Management](#bidirectional-state-management)
4. [Pattern Splitting](#pattern-splitting)
5. [Hybrid Extension Algorithm](#hybrid-extension-algorithm)
6. [WallBreaker Query Iterator](#wallbreaker-query-iterator)
7. [Public API Integration](#public-api-integration)
8. [Testing Utilities](#testing-utilities)

---

## 1. New Trait Definitions

### 1.1 SubstringDictionary Trait

**Purpose**: Enable exact substring search within dictionary terms.

**Location**: `/src/dictionary/mod.rs`

```rust
/// A dictionary that supports efficient exact substring search.
///
/// This trait extends the base `Dictionary` trait with substring search
/// capabilities required for the WallBreaker algorithm.
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
///
/// let dict = SuffixAutomaton::from_iter(vec!["test", "testing", "tester"]);
/// let matches = dict.find_exact_substring("test");
///
/// assert_eq!(matches.len(), 3); // All three terms contain "test"
/// ```
pub trait SubstringDictionary: Dictionary {
    /// Finds all dictionary terms containing the exact substring `pattern`.
    ///
    /// Returns a vector of matches, each containing:
    /// - The dictionary node at the end of the substring match
    /// - The complete term containing the substring
    /// - The position within the term where the substring starts
    ///
    /// # Arguments
    ///
    /// * `pattern` - The exact substring to search for
    ///
    /// # Returns
    ///
    /// A vector of `SubstringMatch` instances, one per occurrence.
    /// Multiple occurrences within the same term are reported separately.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(|pattern| + k) where k is the number of matches
    /// - Space complexity: O(k)
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>>;
}

/// A single match of an exact substring within a dictionary term.
#[derive(Debug, Clone)]
pub struct SubstringMatch<N> {
    /// The dictionary node at the end of the substring match.
    /// This is the starting point for bidirectional extension.
    pub node: N,

    /// The complete dictionary term containing the substring.
    pub term: String,

    /// The byte position within `term` where the substring starts.
    /// For example, if term="testing" and pattern="est", position=1.
    pub position: usize,

    /// The length of the matched substring (in bytes).
    /// Useful for computing relative positions during extension.
    pub length: usize,
}

impl<N> SubstringMatch<N> {
    /// Creates a new substring match.
    pub fn new(node: N, term: String, position: usize, length: usize) -> Self {
        Self {
            node,
            term,
            position,
            length,
        }
    }

    /// Returns the portion of the term before the match (left context).
    pub fn left_context(&self) -> &str {
        &self.term[..self.position]
    }

    /// Returns the matched substring.
    pub fn matched_substring(&self) -> &str {
        &self.term[self.position..self.position + self.length]
    }

    /// Returns the portion of the term after the match (right context).
    pub fn right_context(&self) -> &str {
        &self.term[self.position + self.length..]
    }
}
```

### 1.2 BidirectionalDictionaryNode Trait

**Purpose**: Enable reverse traversal through dictionary structure.

**Location**: `/src/dictionary/mod.rs`

```rust
/// A dictionary node supporting bidirectional traversal.
///
/// This trait extends `DictionaryNode` with reverse navigation capabilities
/// required for WallBreaker's left extension phase.
pub trait BidirectionalDictionaryNode: DictionaryNode {
    /// Returns the parent node (one step toward root).
    ///
    /// Returns `None` if this is the root node.
    fn parent(&self) -> Option<Self>;

    /// Returns all reverse edges (characters and parent nodes).
    ///
    /// This is the inverse of `edges()` - instead of returning children,
    /// it returns the possible parent transitions.
    ///
    /// # Returns
    ///
    /// An iterator over (character, parent_node) pairs.
    /// The character is the label on the edge from parent to this node.
    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_>;

    /// Performs a reverse transition by consuming `label` backward.
    ///
    /// This is the inverse of `transition()`.
    ///
    /// # Arguments
    ///
    /// * `label` - The character to consume in reverse
    ///
    /// # Returns
    ///
    /// A vector of parent nodes that lead to this node via `label`.
    /// Multiple parents are possible in suffix automata.
    fn reverse_transition(&self, label: Self::Unit) -> Vec<Self>;

    /// Returns the depth of this node (distance from root).
    ///
    /// Used for position tracking during bidirectional extension.
    fn depth(&self) -> usize;
}
```

---

## 2. SuffixAutomaton Extensions

### 2.1 Extended SuffixNode Structure

**Location**: `/src/dictionary/suffix_automaton.rs`

```rust
/// Internal node structure for SuffixAutomaton with bidirectional support.
#[derive(Debug, Clone)]
pub(crate) struct SuffixNode<V: DictionaryValue = ()> {
    /// Forward edges to child nodes.
    pub(crate) edges: Vec<(u8, usize)>,

    /// Suffix link for substring matching (existing).
    pub(crate) suffix_link: Option<usize>,

    /// NEW: Parent link for reverse traversal.
    pub(crate) parent: Option<usize>,

    /// NEW: Parent edge label (character on edge from parent to this node).
    pub(crate) parent_label: Option<u8>,

    /// Maximum length of strings ending at this node.
    pub(crate) max_length: usize,

    /// Whether this node represents a complete dictionary term.
    pub(crate) is_final: bool,

    /// Optional value associated with this term.
    pub(crate) value: Option<V>,
}

impl<V: DictionaryValue> SuffixNode<V> {
    /// Creates a new suffix node with parent link tracking.
    pub(crate) fn new(
        parent: Option<usize>,
        parent_label: Option<u8>,
        max_length: usize,
    ) -> Self {
        Self {
            edges: Vec::new(),
            suffix_link: None,
            parent,
            parent_label,
            max_length,
            is_final: false,
            value: None,
        }
    }

    /// Returns the depth of this node (distance from root).
    pub(crate) fn depth(&self) -> usize {
        self.max_length
    }
}
```

### 2.2 SubstringDictionary Implementation

**Location**: `/src/dictionary/suffix_automaton.rs`

```rust
impl<V: DictionaryValue> SubstringDictionary for SuffixAutomaton<V> {
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch<Self::Node>> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let pattern_bytes = pattern.as_bytes();
        let mut results = Vec::new();

        // Phase 1: Find the node representing the substring
        let mut current_idx = 0; // Start from root
        for &byte in pattern_bytes {
            if let Some(next_idx) = self.find_edge(current_idx, byte) {
                current_idx = next_idx;
            } else {
                // Pattern not found in dictionary
                return Vec::new();
            }
        }

        // Phase 2: Traverse suffix links to find all occurrences
        let mut visited = HashSet::new();
        let mut to_visit = vec![current_idx];

        while let Some(node_idx) = to_visit.pop() {
            if !visited.insert(node_idx) {
                continue; // Already processed
            }

            // For each occurrence, find the complete term and position
            let occurrences = self.find_complete_terms_containing(node_idx, pattern);
            results.extend(occurrences);

            // Follow suffix link to find other occurrences
            if let Some(suffix_idx) = self.nodes[node_idx].suffix_link {
                to_visit.push(suffix_idx);
            }
        }

        results
    }
}

impl<V: DictionaryValue> SuffixAutomaton<V> {
    /// Finds all complete dictionary terms that pass through the given node.
    ///
    /// For each term, computes where the pattern starts within it.
    fn find_complete_terms_containing(
        &self,
        node_idx: usize,
        pattern: &str,
    ) -> Vec<SubstringMatch<SuffixAutomatonNode<V>>> {
        let mut results = Vec::new();

        // Traverse forward from node to find all complete terms
        let mut stack = vec![(node_idx, String::new())];

        while let Some((current_idx, suffix)) = stack.pop() {
            let node = &self.nodes[current_idx];

            if node.is_final {
                // Found a complete term
                // Compute full term by traversing backward to root
                let prefix = self.reconstruct_prefix(current_idx);
                let full_term = format!("{}{}", prefix, suffix);

                // Find position of pattern within full_term
                if let Some(position) = full_term.find(pattern) {
                    results.push(SubstringMatch::new(
                        SuffixAutomatonNode {
                            automaton: self,
                            index: node_idx,
                        },
                        full_term,
                        position,
                        pattern.len(),
                    ));
                }
            }

            // Continue traversing forward
            for &(label, next_idx) in &node.edges {
                let mut new_suffix = suffix.clone();
                new_suffix.push(label as char);
                stack.push((next_idx, new_suffix));
            }
        }

        results
    }

    /// Reconstructs the prefix by traversing parent links backward to root.
    fn reconstruct_prefix(&self, mut node_idx: usize) -> String {
        let mut prefix = Vec::new();

        while let Some(parent_idx) = self.nodes[node_idx].parent {
            if let Some(label) = self.nodes[node_idx].parent_label {
                prefix.push(label);
            }
            node_idx = parent_idx;
        }

        prefix.reverse();
        String::from_utf8(prefix).unwrap()
    }

    /// Helper to find edge with given label.
    fn find_edge(&self, node_idx: usize, label: u8) -> Option<usize> {
        self.nodes[node_idx]
            .edges
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, idx)| *idx)
    }
}
```

### 2.3 BidirectionalDictionaryNode Implementation

**Location**: `/src/dictionary/suffix_automaton.rs`

```rust
impl<'a, V: DictionaryValue> BidirectionalDictionaryNode for SuffixAutomatonNode<'a, V> {
    fn parent(&self) -> Option<Self> {
        self.automaton.nodes[self.index].parent.map(|parent_idx| {
            SuffixAutomatonNode {
                automaton: self.automaton,
                index: parent_idx,
            }
        })
    }

    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        // In suffix automaton, each node has at most one parent
        // (though suffix links provide alternative paths)
        let parent_opt = self.parent();
        let label_opt = self.automaton.nodes[self.index].parent_label;

        Box::new(
            parent_opt
                .into_iter()
                .zip(label_opt.into_iter())
                .map(|(parent, label)| (label, parent)),
        )
    }

    fn reverse_transition(&self, label: Self::Unit) -> Vec<Self> {
        // Check if parent edge has the desired label
        if self.automaton.nodes[self.index].parent_label == Some(label) {
            self.parent().into_iter().collect()
        } else {
            Vec::new()
        }
    }

    fn depth(&self) -> usize {
        self.automaton.nodes[self.index].depth()
    }
}
```

---

## 3. Bidirectional State Management

### 3.1 Extended Position Representation

**Location**: `/src/transducer/position.rs`

```rust
/// Direction of extension in bidirectional WallBreaker algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionDirection {
    /// Extending left from substring match (consuming query/term backward).
    Left,
    /// Extending right from substring match (consuming query/term forward).
    Right,
}

/// Position within both query and dictionary term.
///
/// Extended to support bidirectional traversal with relative positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// Index into the query string (0-based).
    query_index: usize,

    /// Index into the dictionary term (0-based).
    term_index: usize,

    /// NEW: Extension direction (for bidirectional traversal).
    direction: ExtensionDirection,

    /// NEW: Offset from substring match position.
    /// For left extension, this is negative (conceptually).
    /// For right extension, this is positive.
    offset: isize,
}

impl Position {
    /// Creates a new position for bidirectional extension.
    pub fn new_bidirectional(
        query_index: usize,
        term_index: usize,
        direction: ExtensionDirection,
        offset: isize,
    ) -> Self {
        Self {
            query_index,
            term_index,
            direction,
            offset,
        }
    }

    /// Creates a traditional (forward-only) position.
    pub fn new(query_index: usize, term_index: usize) -> Self {
        Self {
            query_index,
            term_index,
            direction: ExtensionDirection::Right,
            offset: 0,
        }
    }

    /// Advances position in the extension direction.
    pub fn advance(&self, direction: ExtensionDirection) -> Self {
        match direction {
            ExtensionDirection::Right => Self {
                query_index: self.query_index + 1,
                term_index: self.term_index + 1,
                direction,
                offset: self.offset + 1,
            },
            ExtensionDirection::Left => Self {
                query_index: self.query_index.saturating_sub(1),
                term_index: self.term_index.saturating_sub(1),
                direction,
                offset: self.offset - 1,
            },
        }
    }

    /// Returns the extension direction.
    pub fn direction(&self) -> ExtensionDirection {
        self.direction
    }

    /// Returns the offset from substring match position.
    pub fn offset(&self) -> isize {
        self.offset
    }

    // ... existing methods (query_index, term_index, etc.)
}
```

### 3.2 Bidirectional State

**Location**: `/src/transducer/state.rs`

```rust
/// State for bidirectional Levenshtein automaton.
///
/// Tracks both left and right extension states separately,
/// allowing for independent exploration in each direction.
#[derive(Debug, Clone)]
pub struct BidirectionalState {
    /// Positions for left extension.
    left_positions: Vec<Position>,

    /// Positions for right extension.
    right_positions: Vec<Position>,

    /// Total error budget consumed in left direction.
    left_distance: usize,

    /// Total error budget consumed in right direction.
    right_distance: usize,
}

impl BidirectionalState {
    /// Creates a new bidirectional state centered at substring match.
    pub fn new_centered(match_position: usize, query_length: usize) -> Self {
        Self {
            left_positions: vec![Position::new_bidirectional(
                match_position,
                0,
                ExtensionDirection::Left,
                0,
            )],
            right_positions: vec![Position::new_bidirectional(
                match_position,
                0,
                ExtensionDirection::Right,
                0,
            )],
            left_distance: 0,
            right_distance: 0,
        }
    }

    /// Returns the total edit distance (left + right).
    pub fn total_distance(&self) -> usize {
        self.left_distance + self.right_distance
    }

    /// Checks if the total distance exceeds the maximum allowed.
    pub fn exceeds_max_distance(&self, max_distance: usize) -> bool {
        self.total_distance() > max_distance
    }

    /// Merges left and right states after complete extension.
    pub fn merge(left: Self, right: Self) -> Option<Self> {
        let total = left.left_distance + right.right_distance;

        Some(Self {
            left_positions: left.left_positions,
            right_positions: right.right_positions,
            left_distance: left.left_distance,
            right_distance: right.right_distance,
        })
    }

    /// Returns an iterator over all positions (left + right).
    pub fn all_positions(&self) -> impl Iterator<Item = &Position> {
        self.left_positions.iter().chain(self.right_positions.iter())
    }
}
```

---

## 4. Pattern Splitting

### 4.1 PatternSplitter

**Location**: `/src/wallbreaker/pattern_splitter.rs` (new file)

```rust
/// Splits a pattern into b+1 pieces for WallBreaker algorithm.
///
/// The splitting guarantees that at least one piece will match exactly
/// if the total edit distance is ≤ b.
#[derive(Debug)]
pub struct PatternSplitter {
    /// Maximum edit distance (b).
    max_distance: usize,
}

impl PatternSplitter {
    /// Creates a new pattern splitter.
    pub fn new(max_distance: usize) -> Self {
        Self { max_distance }
    }

    /// Splits the pattern into b+1 approximately equal pieces.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The query pattern to split
    ///
    /// # Returns
    ///
    /// A vector of pattern pieces. Each piece is a substring of the original.
    ///
    /// # Algorithm
    ///
    /// - If pattern.len() >= b+1: Split into b+1 equal-length pieces
    /// - If pattern.len() < b+1: Use overlapping pieces or single piece
    ///
    /// # Examples
    ///
    /// ```
    /// let splitter = PatternSplitter::new(2); // b = 2, need 3 pieces
    /// let pieces = splitter.split("algorithm");
    /// assert_eq!(pieces.len(), 3);
    /// // pieces: ["alg", "ori", "thm"]
    /// ```
    pub fn split(&self, pattern: &str) -> Vec<PatternPiece> {
        let num_pieces = self.max_distance + 1;
        let pattern_len = pattern.len();

        if pattern_len == 0 {
            return Vec::new();
        }

        if pattern_len < num_pieces {
            // Pattern too short: use overlapping pieces or single piece
            return self.split_short_pattern(pattern);
        }

        // Normal case: split into approximately equal pieces
        let piece_size = pattern_len / num_pieces;
        let mut pieces = Vec::new();

        for i in 0..num_pieces {
            let start = i * piece_size;
            let end = if i == num_pieces - 1 {
                pattern_len
            } else {
                (i + 1) * piece_size
            };

            pieces.push(PatternPiece {
                content: pattern[start..end].to_string(),
                start_offset: start,
                piece_index: i,
            });
        }

        pieces
    }

    /// Handles short patterns (length < b+1) with overlapping pieces.
    fn split_short_pattern(&self, pattern: &str) -> Vec<PatternPiece> {
        // For very short patterns, use single piece or character-by-character
        if pattern.len() <= 2 {
            return vec![PatternPiece {
                content: pattern.to_string(),
                start_offset: 0,
                piece_index: 0,
            }];
        }

        // Use overlapping 2-character pieces
        let mut pieces = Vec::new();
        for (i, window) in pattern.as_bytes().windows(2).enumerate() {
            pieces.push(PatternPiece {
                content: String::from_utf8_lossy(window).to_string(),
                start_offset: i,
                piece_index: i,
            });
        }

        pieces
    }
}

/// A piece of a split pattern.
#[derive(Debug, Clone)]
pub struct PatternPiece {
    /// The substring content.
    pub content: String,

    /// Byte offset from start of original pattern.
    pub start_offset: usize,

    /// Index of this piece (0 to b).
    pub piece_index: usize,
}

impl PatternPiece {
    /// Returns the length of this piece (in bytes).
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Checks if this piece is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}
```

---

## 5. Hybrid Extension Algorithm

### 5.1 HybridExtension

**Location**: `/src/wallbreaker/hybrid_extension.rs` (new file)

```rust
use crate::dictionary::{BidirectionalDictionaryNode, SubstringMatch};
use crate::transducer::{BidirectionalState, ExtensionDirection, Position};

/// Hybrid left/right extension for WallBreaker algorithm.
///
/// Uses parent links for left extension and forward edges for right extension.
pub struct HybridExtension<'a, N> {
    /// The substring match to extend from.
    substring_match: &'a SubstringMatch<N>,

    /// Maximum total edit distance allowed.
    max_distance: usize,

    /// Query pattern (full).
    query: &'a str,

    /// State pool for memory efficiency.
    state_pool: &'a mut StatePool,
}

impl<'a, N: BidirectionalDictionaryNode> HybridExtension<'a, N> {
    /// Creates a new hybrid extension.
    pub fn new(
        substring_match: &'a SubstringMatch<N>,
        query: &'a str,
        max_distance: usize,
        state_pool: &'a mut StatePool,
    ) -> Self {
        Self {
            substring_match,
            max_distance,
            query,
            state_pool,
        }
    }

    /// Performs bidirectional extension and returns candidates.
    ///
    /// Returns a vector of (term, distance) pairs that satisfy the distance bound.
    pub fn extend(&mut self) -> Vec<(String, usize)> {
        let mut results = Vec::new();

        // Phase 1: Extend left from substring match
        let left_states = self.extend_left();

        // Phase 2: For each left state, extend right
        for left_state in left_states {
            let right_states = self.extend_right(&left_state);

            // Phase 3: Merge and verify total distance
            for right_state in right_states {
                if let Some(merged) = BidirectionalState::merge(left_state.clone(), right_state) {
                    if !merged.exceeds_max_distance(self.max_distance) {
                        // Valid candidate found
                        let term = self.reconstruct_term(&merged);
                        let distance = merged.total_distance();
                        results.push((term, distance));
                    }
                }
            }
        }

        results
    }

    /// Extends left from substring match using parent links.
    fn extend_left(&mut self) -> Vec<BidirectionalState> {
        let mut results = Vec::new();
        let mut pending: VecDeque<BidirectionalState> = VecDeque::new();

        // Initialize with substring match position
        let piece_start = self.substring_match.position;
        let initial_state = BidirectionalState::new_centered(piece_start, self.query.len());
        pending.push_back(initial_state);

        while let Some(current_state) = pending.pop_front() {
            // Check if we've reached the start of the term or query
            if self.reached_left_boundary(&current_state) {
                results.push(current_state);
                continue;
            }

            // Check if we've exceeded distance budget
            if current_state.left_distance >= self.max_distance {
                continue; // Prune this path
            }

            // Try reverse transitions (parent links)
            for (label, parent_node) in self.substring_match.node.reverse_edges() {
                // Apply left transition filter
                if let Some(next_state) = self.transition_left(&current_state, label) {
                    pending.push_back(next_state);
                }
            }

            // Also try error transitions (insertion, deletion from dictionary perspective)
            // These don't consume dictionary characters
            for error_state in self.error_transitions_left(&current_state) {
                pending.push_back(error_state);
            }
        }

        results
    }

    /// Extends right from current position using forward edges.
    fn extend_right(&mut self, left_state: &BidirectionalState) -> Vec<BidirectionalState> {
        let mut results = Vec::new();
        let mut pending: VecDeque<(N, BidirectionalState)> = VecDeque::new();

        // Start from substring match node, extend right
        let initial_state = left_state.clone(); // Continue from left extension
        pending.push_back((self.substring_match.node.clone(), initial_state));

        while let Some((current_node, current_state)) = pending.pop_front() {
            // Check if we've reached the end of the term or query
            if self.reached_right_boundary(&current_state) {
                results.push(current_state);
                continue;
            }

            // Check if we've exceeded distance budget
            if current_state.total_distance() >= self.max_distance {
                continue; // Prune this path
            }

            // Try forward transitions
            for (label, child_node) in current_node.edges() {
                // Apply right transition filter
                if let Some(next_state) = self.transition_right(&current_state, label) {
                    pending.push_back((child_node, next_state));
                }
            }

            // Error transitions
            for error_state in self.error_transitions_right(&current_state) {
                pending.push_back((current_node.clone(), error_state));
            }
        }

        results
    }

    /// Applies left transition filter (consumes character backward).
    ///
    /// Algorithm:
    /// 1. Read the next active left-side query index from `state`.
    /// 2. Compare `label` with the corresponding query byte.
    /// 3. Apply the reverse Levenshtein transition recurrence to each live
    ///    left position.
    /// 4. Drop positions whose edit cost exceeds the configured threshold.
    /// 5. Return `None` only when the resulting left frontier is empty.
    fn transition_left(&self, state: &BidirectionalState, label: u8) -> Option<BidirectionalState> {
        let left_positions = state
            .left_positions
            .iter()
            .filter_map(|position| position.transition_left(label, self.query, self.max_distance))
            .collect::<Vec<_>>();

        (!left_positions.is_empty()).then(|| BidirectionalState {
            left_positions,
            right_positions: state.right_positions.clone(),
        })
    }

    /// Applies right transition filter (consumes character forward).
    ///
    /// Algorithm:
    /// 1. Read the next active right-side query index from `state`.
    /// 2. Compare `label` with the corresponding query byte.
    /// 3. Apply the forward Levenshtein transition recurrence to each live
    ///    right position.
    /// 4. Drop positions whose edit cost exceeds the configured threshold.
    /// 5. Return `None` only when the resulting right frontier is empty.
    fn transition_right(&self, state: &BidirectionalState, label: u8) -> Option<BidirectionalState> {
        let right_positions = state
            .right_positions
            .iter()
            .filter_map(|position| position.transition_right(label, self.query, self.max_distance))
            .collect::<Vec<_>>();

        (!right_positions.is_empty()).then(|| BidirectionalState {
            left_positions: state.left_positions.clone(),
            right_positions,
        })
    }

    /// Checks if left extension has reached dictionary or query start.
    fn reached_left_boundary(&self, state: &BidirectionalState) -> bool {
        // Check if all positions are at index 0 (start of query/term)
        state.left_positions.iter().all(|p| p.query_index() == 0)
    }

    /// Checks if right extension has reached dictionary or query end.
    fn reached_right_boundary(&self, state: &BidirectionalState) -> bool {
        // Check if all positions are at end of query
        let query_len = self.query.len();
        state
            .right_positions
            .iter()
            .all(|p| p.query_index() >= query_len)
    }

    /// Generates error transitions for left extension.
    fn error_transitions_left(&self, state: &BidirectionalState) -> Vec<BidirectionalState> {
        // Insertion, deletion, substitution in reverse
        // ... (detailed implementation omitted)
        Vec::new()
    }

    /// Generates error transitions for right extension.
    fn error_transitions_right(&self, state: &BidirectionalState) -> Vec<BidirectionalState> {
        // Standard insertion, deletion, substitution
        // ... (detailed implementation omitted)
        Vec::new()
    }

    /// Reconstructs the complete term from bidirectional state.
    fn reconstruct_term(&self, state: &BidirectionalState) -> String {
        // Use parent links to reconstruct left context
        // Use forward edges to reconstruct right context
        // Combine: left_context + matched_substring + right_context
        self.substring_match.term.clone() // Simplified
    }
}
```

---

## 6. WallBreaker Query Iterator

### 6.1 Main Query Iterator

**Location**: `/src/wallbreaker/query_iterator.rs` (new file)

```rust
use std::collections::HashSet;

/// Iterator for WallBreaker fuzzy search queries.
///
/// Implements the complete WallBreaker algorithm:
/// 1. Split pattern into b+1 pieces
/// 2. Find exact matches for each piece
/// 3. Extend bidirectionally from each match
/// 4. Verify total distance and deduplicate results
pub struct WallBreakerQueryIterator<'a, D>
where
    D: SubstringDictionary + BidirectionalDictionary,
{
    /// The dictionary to search.
    dictionary: &'a D,

    /// The query pattern.
    query: &'a str,

    /// Maximum edit distance.
    max_distance: usize,

    /// Pattern pieces (b+1 pieces).
    pattern_pieces: Vec<PatternPiece>,

    /// Current piece index being explored.
    current_piece: usize,

    /// Substring matches for current piece.
    current_matches: Vec<SubstringMatch<D::Node>>,

    /// Current match index within current_matches.
    current_match_idx: usize,

    /// Pending results from hybrid extension.
    pending_results: VecDeque<(String, usize)>,

    /// Set of already-yielded results (for deduplication).
    seen_results: HashSet<String>,

    /// State pool for memory efficiency.
    state_pool: StatePool,
}

impl<'a, D> WallBreakerQueryIterator<'a, D>
where
    D: SubstringDictionary + BidirectionalDictionary,
{
    /// Creates a new WallBreaker query iterator.
    pub fn new(dictionary: &'a D, query: &'a str, max_distance: usize) -> Self {
        let splitter = PatternSplitter::new(max_distance);
        let pattern_pieces = splitter.split(query);

        Self {
            dictionary,
            query,
            max_distance,
            pattern_pieces,
            current_piece: 0,
            current_matches: Vec::new(),
            current_match_idx: 0,
            pending_results: VecDeque::new(),
            seen_results: HashSet::new(),
            state_pool: StatePool::new(),
        }
    }

    /// Advances to next pattern piece.
    fn advance_to_next_piece(&mut self) -> bool {
        self.current_piece += 1;

        if self.current_piece >= self.pattern_pieces.len() {
            return false; // No more pieces
        }

        // Find substring matches for new piece
        let piece = &self.pattern_pieces[self.current_piece];
        self.current_matches = self.dictionary.find_exact_substring(&piece.content);
        self.current_match_idx = 0;

        true
    }

    /// Processes next substring match.
    fn process_next_match(&mut self) {
        if self.current_match_idx >= self.current_matches.len() {
            // No more matches for this piece, try next piece
            if !self.advance_to_next_piece() {
                return; // No more pieces, iteration complete
            }
        }

        let substring_match = &self.current_matches[self.current_match_idx];
        self.current_match_idx += 1;

        // Perform hybrid extension
        let mut extension = HybridExtension::new(
            substring_match,
            self.query,
            self.max_distance,
            &mut self.state_pool,
        );

        let results = extension.extend();

        // Add results to pending queue (will be deduplicated)
        for (term, distance) in results {
            if !self.seen_results.contains(&term) {
                self.pending_results.push_back((term.clone(), distance));
                self.seen_results.insert(term);
            }
        }
    }
}

impl<'a, D> Iterator for WallBreakerQueryIterator<'a, D>
where
    D: SubstringDictionary + BidirectionalDictionary,
{
    type Item = (String, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have pending results, yield them first
            if let Some(result) = self.pending_results.pop_front() {
                return Some(result);
            }

            // Otherwise, process next substring match
            if self.current_matches.is_empty() && self.current_piece == 0 {
                // First call: initialize with first piece
                if !self.pattern_pieces.is_empty() {
                    let piece = &self.pattern_pieces[0];
                    self.current_matches = self.dictionary.find_exact_substring(&piece.content);
                }
            }

            self.process_next_match();

            // If still no pending results, we're done
            if self.pending_results.is_empty()
                && self.current_match_idx >= self.current_matches.len()
                && self.current_piece >= self.pattern_pieces.len()
            {
                return None;
            }
        }
    }
}
```

---

## 7. Public API Integration

### 7.1 New Public API Functions

**Location**: `/src/lib.rs`

```rust
/// Performs fuzzy search using WallBreaker algorithm.
///
/// This is optimized for large error bounds (≥4) and long patterns (≥50 chars).
/// For small error bounds, consider using `fuzzy_search()` instead.
///
/// # Requirements
///
/// - Dictionary must implement `SubstringDictionary` trait
/// - Currently only `SuffixAutomaton` supports this
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
///
/// let dict = SuffixAutomaton::from_iter(vec!["test", "testing", "tester"]);
/// let results: Vec<_> = fuzzy_search_wallbreaker(&dict, "texting", 2).collect();
/// assert!(results.contains(&"testing".to_string()));
/// ```
pub fn fuzzy_search_wallbreaker<'a, D>(
    dictionary: &'a D,
    query: &'a str,
    max_distance: usize,
) -> impl Iterator<Item = String> + 'a
where
    D: SubstringDictionary + BidirectionalDictionary,
{
    WallBreakerQueryIterator::new(dictionary, query, max_distance).map(|(term, _)| term)
}

/// Automatically selects the best fuzzy search algorithm.
///
/// Uses WallBreaker if beneficial (large error bound, long pattern),
/// otherwise falls back to traditional approach.
///
/// # Decision Logic
///
/// - If max_distance ≥ 4 AND pattern.len() ≥ 50: Use WallBreaker
/// - Otherwise: Use traditional approach
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
///
/// let dict = SuffixAutomaton::from_iter(vec!["test", "testing"]);
///
/// // Short pattern, small distance: uses traditional
/// let results: Vec<_> = fuzzy_search_auto(&dict, "test", 2).collect();
///
/// // Long pattern, large distance: uses WallBreaker
/// let results: Vec<_> = fuzzy_search_auto(&dict, "extraordinary", 8).collect();
/// ```
pub fn fuzzy_search_auto<'a, D>(
    dictionary: &'a D,
    query: &'a str,
    max_distance: usize,
) -> Box<dyn Iterator<Item = String> + 'a>
where
    D: Dictionary + SubstringDictionary + BidirectionalDictionary,
{
    // Decision thresholds (can be tuned based on benchmarks)
    const MIN_DISTANCE_FOR_WALLBREAKER: usize = 4;
    const MIN_PATTERN_LENGTH_FOR_WALLBREAKER: usize = 50;

    if max_distance >= MIN_DISTANCE_FOR_WALLBREAKER
        && query.len() >= MIN_PATTERN_LENGTH_FOR_WALLBREAKER
    {
        // Use WallBreaker
        Box::new(fuzzy_search_wallbreaker(dictionary, query, max_distance))
    } else {
        // Use traditional
        Box::new(fuzzy_search(dictionary, query, max_distance))
    }
}
```

---

## 8. Testing Utilities

### 8.1 Test Helper Functions

**Location**: `/tests/wallbreaker/helpers.rs`

```rust
use liblevenshtein::prelude::*;

/// Computes the true Levenshtein distance between two strings.
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        dp[i][0] = i;
    }
    for j in 0..=len2 {
        dp[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            dp[i + 1][j + 1] = std::cmp::min(
                std::cmp::min(dp[i][j + 1] + 1, dp[i + 1][j] + 1),
                dp[i][j] + cost,
            );
        }
    }

    dp[len1][len2]
}

/// Verifies that WallBreaker results match traditional approach.
pub fn verify_correctness<D>(dictionary: &D, query: &str, max_distance: usize)
where
    D: Dictionary + SubstringDictionary + BidirectionalDictionary,
{
    let trad_results: HashSet<_> = fuzzy_search(dictionary, query, max_distance).collect();
    let wb_results: HashSet<_> = fuzzy_search_wallbreaker(dictionary, query, max_distance).collect();

    assert_eq!(
        trad_results, wb_results,
        "WallBreaker results must match traditional approach\n\
         Traditional: {:?}\n\
         WallBreaker: {:?}",
        trad_results, wb_results
    );
}

/// Verifies that all results are within the distance bound.
pub fn verify_distances<D>(dictionary: &D, query: &str, max_distance: usize)
where
    D: SubstringDictionary + BidirectionalDictionary,
{
    for result in fuzzy_search_wallbreaker(dictionary, query, max_distance) {
        let actual_distance = levenshtein_distance(query, &result);
        assert!(
            actual_distance <= max_distance,
            "Result '{}' has distance {} > max_distance {}",
            result,
            actual_distance,
            max_distance
        );
    }
}
```

---

## 9. Summary

This document provides concrete architectural sketches for implementing WallBreaker (Option B - Hybrid approach). Key components include:

1. **New Traits**: `SubstringDictionary` and `BidirectionalDictionaryNode` extend existing dictionary capabilities
2. **SuffixAutomaton Extensions**: Add parent links and implement new traits
3. **Bidirectional State**: Track left/right extensions separately with relative positioning
4. **Pattern Splitting**: Divide pattern into b+1 pieces for exact substring search
5. **Hybrid Extension**: Use parent links for left extension, forward edges for right
6. **Query Iterator**: Orchestrate the complete WallBreaker algorithm with deduplication
7. **Public API**: Clean integration with automatic selection logic

**Next Steps**:
1. Review these designs for feasibility and completeness
2. Begin implementation following [progress-tracker.md](./progress-tracker.md)
3. Validate with tests from [benchmarking-plan.md](./benchmarking-plan.md)
4. Iterate based on benchmark results

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-06
**Related Documents**:
- [implementation-plan.md](./implementation-plan.md) - Detailed implementation phases
- [technical-analysis.md](./technical-analysis.md) - Current architecture analysis
- [progress-tracker.md](./progress-tracker.md) - Task breakdown and tracking
