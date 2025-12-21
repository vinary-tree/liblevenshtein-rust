# Phase B: Lattice Integration Guide

**Last Updated**: 2025-12-21
**Version**: v0.8.0
**Status**: PROPOSED

> ⚠️ **PROPOSAL NOTICE**: This document describes a **proposed** `src/lattice/` module for structured multi-candidate output. This structure is a design specification for future implementation.
>
> **Current Implementation**: liblevenshtein v0.8.0 returns flat iterators from transducer queries. See [Current v0.8.0 Capabilities](#current-v080-capabilities) for available features.

This document provides detailed implementation guidance for Phase B of the MORK integration: creating lattice data structures for ranked, multi-candidate approximate matching.

## Overview

**Goal**: Extend liblevenshtein to return structured lattice DAGs instead of flat iterators, enabling ranked results with weighted scores.

**Result**: Queries return `Lattice` structures that MORK can traverse to extract n-best paths with combined edit distance and phonetic costs.

---

## Current v0.8.0 Capabilities

Before implementing the proposed lattice module, liblevenshtein v0.8.0 provides:

### What's Available Now

| Feature | API | Description |
|---------|-----|-------------|
| Flat candidate iteration | `transducer.query(term, max_dist)` | Returns `Iterator<Candidate>` |
| Distance-based results | `Candidate { term, distance }` | Edit distance per match |
| ProductAutomaton | `ProductAutomatonChar::new(nfa, max_dist)` | NFA × Levenshtein composition |

### Current Query Pattern

```rust
use liblevenshtein::transducer::{Algorithm, Transducer};
use liblevenshtein::dictionary::DynamicDawgChar;

let dict = DynamicDawgChar::from_iter(["the", "ten", "tea", "cat", "bat", "car"]);
let transducer = Transducer::new(&dict, Algorithm::Standard);

// Current: flat iterator of candidates
for candidate in transducer.query("teh", 2) {
    println!("{}: distance {}", candidate.term, candidate.distance);
}
// Output (unordered):
// the: distance 1
// ten: distance 2
// tea: distance 2
```

### Workaround: Manual Ranking

Until lattice support is added, you can manually collect and sort:

```rust
let mut candidates: Vec<_> = transducer.query("teh", 2).collect();
candidates.sort_by_key(|c| c.distance);
let top_5 = &candidates[..5.min(candidates.len())];
```

---

## Architecture

### Lattice Data Flow

```
Query Term
    |
    v
Transducer::query_lattice()
    |
    | Builds DAG of candidates with weighted edges
    v
Lattice { nodes, edges, vocab }
    |
    v
LatticeZipper (MORK adapter)
    |
    | Iterates paths by total weight
    v
AFactor::LatticeSource
    |
    v
ProductZipper → Unification → Ranked Results
```

### Component Locations (PROPOSED)

> **Note**: This is the **proposed** module structure. It does not currently exist.

```
liblevenshtein-rust/src/
├── lattice/                  # PROPOSED - To be implemented
│   ├── mod.rs                # Core Lattice struct
│   ├── node.rs               # Node representation
│   ├── edge.rs               # Edge with weights and metadata
│   ├── builder.rs            # LatticeBuilder for construction
│   └── path_iterator.rs      # N-best path extraction

MORK/kernel/src/
├── lattice_zipper.rs         # Zipper adapter for lattice paths
└── sources.rs                # LatticeSource variant in AFactor
```

---

## Data Structures

### Core Lattice Type

**File**: `liblevenshtein-rust/src/lattice/mod.rs`

```rust
//! Lattice data structures for multi-candidate approximate matching.
//!
//! A lattice represents the set of all correction candidates as a weighted DAG.
//! Each edge represents a word choice with associated costs (edit distance,
//! phonetic similarity, etc.).

pub mod node;
pub mod edge;
pub mod builder;
pub mod path_iterator;

use std::sync::Arc;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unique identifier for a node in the lattice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Unique identifier for an edge in the lattice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub usize);

/// Identifier for vocabulary deduplication.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VocabId(pub usize);

/// A lattice representing multiple candidate corrections.
///
/// The lattice is a weighted DAG where:
/// - Nodes represent positions in the input sequence
/// - Edges represent word choices at each position
/// - Edge weights combine edit distance and other costs
///
/// # Example
///
/// For input "teh cat", with candidates:
/// - "teh" → ["the", "ten", "tea"]
/// - "cat" → ["cat", "bat", "car"]
///
/// The lattice has structure:
/// ```text
/// Start ─┬─ "the" (0.0) ─┬─ "cat" (0.0) ─┬─ End
///        ├─ "ten" (1.0) ─┼─ "bat" (1.0) ─┤
///        └─ "tea" (1.0) ─┴─ "car" (1.0) ─┘
/// ```
#[derive(Clone, Debug)]
pub struct Lattice {
    /// All nodes in the lattice
    pub(crate) nodes: Vec<Node>,

    /// All edges in the lattice
    pub(crate) edges: Vec<Edge>,

    /// Start node (position 0)
    pub(crate) start: NodeId,

    /// End node (final position)
    pub(crate) end: NodeId,

    /// Deduplicated vocabulary (label strings)
    pub(crate) vocab: IndexMap<Arc<str>, VocabId>,

    /// Metadata about the lattice
    pub(crate) metadata: LatticeMetadata,
}

/// Metadata about lattice construction.
#[derive(Clone, Debug, Default)]
pub struct LatticeMetadata {
    /// Original input that generated this lattice
    pub input: String,

    /// Maximum edit distance used during construction
    pub max_distance: usize,

    /// Size of dictionary used
    pub dictionary_size: usize,

    /// Cached path count (computed on demand)
    pub path_count: Option<usize>,
}

impl Lattice {
    /// Create an empty lattice.
    pub fn new() -> Self {
        let start_node = Node::new(NodeId(0));
        let end_node = Node::new(NodeId(1));

        Self {
            nodes: vec![start_node, end_node],
            start: NodeId(0),
            end: NodeId(1),
            edges: Vec::new(),
            vocab: IndexMap::new(),
            metadata: LatticeMetadata::default(),
        }
    }

    /// Get the start node ID.
    pub fn start(&self) -> NodeId {
        self.start
    }

    /// Get the end node ID.
    pub fn end(&self) -> NodeId {
        self.end
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Get an edge by ID.
    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.0)
    }

    /// Get the label string for a vocabulary ID.
    pub fn label(&self, vocab_id: VocabId) -> Option<&str> {
        self.vocab
            .get_index(vocab_id.0)
            .map(|(s, _)| s.as_ref())
    }

    /// Get all outgoing edges from a node.
    pub fn outgoing_edges(&self, node_id: NodeId) -> impl Iterator<Item = &Edge> {
        self.nodes
            .get(node_id.0)
            .into_iter()
            .flat_map(|n| n.outgoing.iter())
            .filter_map(|&eid| self.edges.get(eid.0))
    }

    /// Count total number of paths from start to end.
    pub fn path_count(&self) -> usize {
        if let Some(count) = self.metadata.path_count {
            return count;
        }

        // Dynamic programming: count[node] = number of paths from node to end
        let mut count = vec![0usize; self.nodes.len()];
        count[self.end.0] = 1;

        // Process nodes in reverse topological order
        for node_id in (0..self.nodes.len()).rev() {
            if node_id == self.end.0 {
                continue;
            }
            for &edge_id in &self.nodes[node_id].outgoing {
                let target = self.edges[edge_id.0].target;
                count[node_id] += count[target.0];
            }
        }

        count[self.start.0]
    }

    /// Get top-k paths by total weight (lowest weight first).
    pub fn k_best_paths(&self, k: usize) -> Vec<LatticePath> {
        path_iterator::k_best(self, k)
    }

    /// Iterate over all paths (may be exponential; use k_best for large lattices).
    pub fn paths(&self) -> PathIterator<'_> {
        PathIterator::new(self)
    }
}

impl Default for Lattice {
    fn default() -> Self {
        Self::new()
    }
}
```

### Node Type

**File**: `liblevenshtein-rust/src/lattice/node.rs`

```rust
//! Node representation in the lattice.

use super::{NodeId, EdgeId};
use smallvec::SmallVec;

/// A node in the lattice DAG.
///
/// Nodes represent positions in the input sequence. Each node tracks
/// its outgoing and incoming edges for efficient traversal.
#[derive(Clone, Debug)]
pub struct Node {
    /// Unique identifier for this node
    pub id: NodeId,

    /// Outgoing edges (to later positions)
    pub outgoing: SmallVec<[EdgeId; 8]>,

    /// Incoming edges (from earlier positions)
    pub incoming: SmallVec<[EdgeId; 8]>,

    /// Position in input sequence (if applicable)
    pub position: Option<usize>,
}

impl Node {
    /// Create a new node with the given ID.
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            outgoing: SmallVec::new(),
            incoming: SmallVec::new(),
            position: None,
        }
    }

    /// Create a node at a specific position.
    pub fn at_position(id: NodeId, position: usize) -> Self {
        Self {
            id,
            outgoing: SmallVec::new(),
            incoming: SmallVec::new(),
            position: Some(position),
        }
    }

    /// Add an outgoing edge.
    pub fn add_outgoing(&mut self, edge_id: EdgeId) {
        self.outgoing.push(edge_id);
    }

    /// Add an incoming edge.
    pub fn add_incoming(&mut self, edge_id: EdgeId) {
        self.incoming.push(edge_id);
    }
}
```

### Edge Type

**File**: `liblevenshtein-rust/src/lattice/edge.rs`

```rust
//! Edge representation in the lattice.

use super::{NodeId, EdgeId, VocabId};

/// An edge in the lattice DAG.
///
/// Edges represent word choices with associated weights and metadata.
/// The weight combines multiple signals (edit distance, phonetic cost, etc.).
#[derive(Clone, Debug)]
pub struct Edge {
    /// Unique identifier for this edge
    pub id: EdgeId,

    /// Source node (earlier position)
    pub source: NodeId,

    /// Target node (later position)
    pub target: NodeId,

    /// Label (word) for this edge
    pub label: VocabId,

    /// Combined weight (lower is better)
    pub weight: f32,

    /// Additional metadata about this edge
    pub metadata: EdgeMetadata,
}

/// Metadata about an edge's origin and properties.
#[derive(Clone, Debug, Default)]
pub struct EdgeMetadata {
    /// Edit distance component (if from fuzzy matching)
    pub distance: Option<u8>,

    /// Whether this edge came from phonetic matching
    pub is_phonetic: bool,

    /// Phonetic rule ID (if applicable)
    pub rule_id: Option<usize>,

    /// Original query term this edge corrects
    pub original_term: Option<String>,
}

impl Edge {
    /// Create a new edge with the given properties.
    pub fn new(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: VocabId,
        weight: f32,
    ) -> Self {
        Self {
            id,
            source,
            target,
            label,
            weight,
            metadata: EdgeMetadata::default(),
        }
    }

    /// Create an edge with full metadata.
    pub fn with_metadata(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: VocabId,
        weight: f32,
        metadata: EdgeMetadata,
    ) -> Self {
        Self {
            id,
            source,
            target,
            label,
            weight,
            metadata,
        }
    }
}
```

### LatticeBuilder

**File**: `liblevenshtein-rust/src/lattice/builder.rs`

```rust
//! Builder for constructing lattices incrementally.

use super::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Builder for constructing lattices incrementally.
///
/// # Example
///
/// ```rust
/// let mut builder = LatticeBuilder::new();
/// builder.add_correction(0, 1, "the", 0.0, 0, false);
/// builder.add_correction(0, 1, "ten", 1.0, 1, false);
/// builder.add_correction(1, 2, "cat", 0.0, 0, false);
/// let lattice = builder.build(2);
/// ```
pub struct LatticeBuilder {
    /// Nodes being built
    nodes: Vec<Node>,

    /// Edges being built
    edges: Vec<Edge>,

    /// Map from position to node ID
    node_map: HashMap<usize, NodeId>,

    /// Vocabulary deduplication
    vocab: IndexMap<Arc<str>, VocabId>,

    /// Next node ID to assign
    next_node_id: usize,

    /// Next edge ID to assign
    next_edge_id: usize,

    /// Input string being corrected
    input: String,
}

impl LatticeBuilder {
    /// Create a new lattice builder.
    pub fn new() -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_map: HashMap::new(),
            vocab: IndexMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            input: String::new(),
        };

        // Create start node at position 0
        let start_id = builder.get_or_create_node(0);
        assert_eq!(start_id, NodeId(0));

        builder
    }

    /// Set the input string for metadata.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input = input.into();
        self
    }

    /// Get or create a node at the given position.
    fn get_or_create_node(&mut self, position: usize) -> NodeId {
        if let Some(&id) = self.node_map.get(&position) {
            return id;
        }

        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(Node::at_position(id, position));
        self.node_map.insert(position, id);
        id
    }

    /// Get or create a vocabulary entry for a word.
    fn get_or_intern_vocab(&mut self, word: impl Into<String>) -> VocabId {
        let word: String = word.into();
        let arc: Arc<str> = Arc::from(word.as_str());

        if let Some(&id) = self.vocab.get(&arc) {
            return id;
        }

        let id = VocabId(self.vocab.len());
        self.vocab.insert(arc, id);
        id
    }

    /// Add a correction edge to the lattice.
    ///
    /// # Arguments
    /// * `start_pos` - Starting position in input
    /// * `end_pos` - Ending position in input
    /// * `word` - The correction word
    /// * `weight` - Edge weight (lower is better)
    /// * `distance` - Edit distance (if from fuzzy matching)
    /// * `is_phonetic` - Whether this is a phonetic match
    pub fn add_correction(
        &mut self,
        start_pos: usize,
        end_pos: usize,
        word: impl Into<String>,
        weight: f32,
        distance: u8,
        is_phonetic: bool,
    ) -> &mut Self {
        let source = self.get_or_create_node(start_pos);
        let target = self.get_or_create_node(end_pos);
        let label = self.get_or_intern_vocab(word);

        let edge_id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        let metadata = EdgeMetadata {
            distance: Some(distance),
            is_phonetic,
            rule_id: None,
            original_term: None,
        };

        let edge = Edge::with_metadata(edge_id, source, target, label, weight, metadata);

        // Update node connectivity
        self.nodes[source.0].add_outgoing(edge_id);
        self.nodes[target.0].add_incoming(edge_id);

        self.edges.push(edge);
        self
    }

    /// Build the final lattice.
    ///
    /// # Arguments
    /// * `end_pos` - Position of the end node
    pub fn build(mut self, end_pos: usize) -> Lattice {
        let end = self.get_or_create_node(end_pos);

        Lattice {
            nodes: self.nodes,
            edges: self.edges,
            start: NodeId(0),
            end,
            vocab: self.vocab,
            metadata: LatticeMetadata {
                input: self.input,
                max_distance: 0,  // Set by caller
                dictionary_size: 0,  // Set by caller
                path_count: None,
            },
        }
    }
}

impl Default for LatticeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
```

### Path Iterator and N-Best Extraction

**File**: `liblevenshtein-rust/src/lattice/path_iterator.rs`

```rust
//! Path extraction from lattices.

use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A path through the lattice.
#[derive(Clone, Debug)]
pub struct LatticePath {
    /// Sequence of edge IDs forming the path
    pub edges: Vec<EdgeId>,

    /// Total weight of the path
    pub total_weight: f32,

    /// Reconstructed sentence
    pub sentence: String,
}

impl LatticePath {
    /// Create a new path.
    pub fn new(edges: Vec<EdgeId>, total_weight: f32, sentence: String) -> Self {
        Self {
            edges,
            total_weight,
            sentence,
        }
    }
}

/// Iterator over all paths in a lattice.
pub struct PathIterator<'a> {
    lattice: &'a Lattice,
    stack: Vec<PathState>,
}

struct PathState {
    node: NodeId,
    edges: Vec<EdgeId>,
    weight: f32,
}

impl<'a> PathIterator<'a> {
    pub fn new(lattice: &'a Lattice) -> Self {
        Self {
            lattice,
            stack: vec![PathState {
                node: lattice.start,
                edges: Vec::new(),
                weight: 0.0,
            }],
        }
    }
}

impl<'a> Iterator for PathIterator<'a> {
    type Item = LatticePath;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state) = self.stack.pop() {
            if state.node == self.lattice.end {
                // Reached end; reconstruct sentence
                let sentence = state
                    .edges
                    .iter()
                    .filter_map(|&eid| {
                        let edge = self.lattice.edge(eid)?;
                        self.lattice.label(edge.label)
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                return Some(LatticePath::new(state.edges, state.weight, sentence));
            }

            // Explore outgoing edges (in reverse order for DFS)
            let node = self.lattice.node(state.node)?;
            for &edge_id in node.outgoing.iter().rev() {
                let edge = self.lattice.edge(edge_id)?;
                let mut new_edges = state.edges.clone();
                new_edges.push(edge_id);

                self.stack.push(PathState {
                    node: edge.target,
                    edges: new_edges,
                    weight: state.weight + edge.weight,
                });
            }
        }

        None
    }
}

/// Extract top-k paths by weight using Dijkstra-like algorithm.
pub fn k_best(lattice: &Lattice, k: usize) -> Vec<LatticePath> {
    #[derive(Clone)]
    struct HeapEntry {
        weight: f32,
        node: NodeId,
        edges: Vec<EdgeId>,
    }

    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.weight == other.weight
        }
    }

    impl Eq for HeapEntry {}

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse order for min-heap (lower weight = higher priority)
            other
                .weight
                .partial_cmp(&self.weight)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::new();
    let mut results = Vec::with_capacity(k);

    heap.push(HeapEntry {
        weight: 0.0,
        node: lattice.start,
        edges: Vec::new(),
    });

    while let Some(entry) = heap.pop() {
        if entry.node == lattice.end {
            let sentence = entry
                .edges
                .iter()
                .filter_map(|&eid| {
                    let edge = lattice.edge(eid)?;
                    lattice.label(edge.label)
                })
                .collect::<Vec<_>>()
                .join(" ");

            results.push(LatticePath::new(entry.edges, entry.weight, sentence));

            if results.len() >= k {
                break;
            }
            continue;
        }

        if let Some(node) = lattice.node(entry.node) {
            for &edge_id in &node.outgoing {
                if let Some(edge) = lattice.edge(edge_id) {
                    let mut new_edges = entry.edges.clone();
                    new_edges.push(edge_id);

                    heap.push(HeapEntry {
                        weight: entry.weight + edge.weight,
                        node: edge.target,
                        edges: new_edges,
                    });
                }
            }
        }
    }

    results
}
```

---

## Transducer Extension

**File**: `liblevenshtein-rust/src/transducer/mod.rs` (additions)

```rust
use crate::lattice::{Lattice, LatticeBuilder, LatticeMetadata};

/// Scored candidate with detailed metadata.
#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    /// The matched term
    pub text: String,

    /// Edit distance from query
    pub edit_distance: usize,

    /// Phonetic cost component
    pub phonetic_cost: f64,

    /// Total combined score
    pub total_score: f64,
}

impl<D: Dictionary, P: SubstitutionPolicy + ...> Transducer<D, P> {
    /// Query returning a lattice of all corrections for multiple tokens.
    ///
    /// # Arguments
    /// * `tokens` - Sequence of query tokens
    /// * `max_distance` - Maximum edit distance per token
    ///
    /// # Returns
    /// A `Lattice` containing all correction candidates with weighted edges.
    pub fn query_lattice(&self, tokens: &[&str], max_distance: usize) -> Lattice {
        let mut builder = LatticeBuilder::new()
            .with_input(tokens.join(" "));

        let mut position = 0;
        for token in tokens {
            let next_position = position + 1;

            // Add exact match if exists
            if self.dictionary.contains(token) {
                builder.add_correction(
                    position,
                    next_position,
                    *token,
                    0.0,  // Weight 0 for exact match
                    0,    // Distance 0
                    false,
                );
            }

            // Add fuzzy matches
            for candidate in self.query_candidates(token, max_distance) {
                if candidate.distance == 0 {
                    continue;  // Skip exact match (already added)
                }

                // Weight: inverse of distance (lower weight = better)
                let weight = candidate.distance as f32;

                builder.add_correction(
                    position,
                    next_position,
                    &candidate.term,
                    weight,
                    candidate.distance as u8,
                    false,
                );
            }

            position = next_position;
        }

        let mut lattice = builder.build(position);
        lattice.metadata.max_distance = max_distance;
        lattice.metadata.dictionary_size = self.dictionary.len().unwrap_or(0);
        lattice
    }

    /// Query returning top-k candidates with scores.
    ///
    /// # Arguments
    /// * `term` - Single query term
    /// * `n` - Number of results to return
    /// * `max_distance` - Maximum edit distance
    pub fn query_nbest(
        &self,
        term: &str,
        n: usize,
        max_distance: usize,
    ) -> Vec<ScoredCandidate> {
        let mut candidates: Vec<_> = self
            .query_candidates(term, max_distance)
            .map(|c| ScoredCandidate {
                text: c.term.clone(),
                edit_distance: c.distance,
                phonetic_cost: 0.0,  // Phase C adds phonetic scoring
                total_score: c.distance as f64,
            })
            .collect();

        // Sort by total_score (ascending)
        candidates.sort_by(|a, b| {
            a.total_score
                .partial_cmp(&b.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(n);
        candidates
    }
}
```

---

## MORK Integration

### LatticeZipper

**File**: `MORK/kernel/src/lattice_zipper.rs`

```rust
//! Zipper adapter that presents lattice paths as navigable trie paths.

use liblevenshtein::lattice::{Lattice, LatticePath, PathIterator};
use pathmap::zipper::{Zipper, ZipperIteration, ZipperAbsolutePath};

/// Zipper over lattice paths for integration with MORK's ProductZipper.
///
/// Iterates over paths from the lattice, presenting each as a virtual
/// trie path for unification.
pub struct LatticeZipper<'a> {
    /// Reference to the lattice
    lattice: &'a Lattice,

    /// Current path (if any)
    current_path: Option<LatticePath>,

    /// Path iterator
    path_iterator: PathIterator<'a>,

    /// Buffer for MORK expression path encoding
    path_buffer: Vec<u8>,

    /// Prefix for all result paths
    prefix: Vec<u8>,
}

impl<'a> LatticeZipper<'a> {
    /// Create a new LatticeZipper from a lattice.
    pub fn new(lattice: &'a Lattice, prefix: &[u8]) -> Self {
        let mut path_iterator = lattice.paths();
        let current_path = path_iterator.next();

        let mut zipper = Self {
            lattice,
            current_path,
            path_iterator,
            path_buffer: Vec::with_capacity(1024),
            prefix: prefix.to_vec(),
        };

        zipper.rebuild_path_buffer();
        zipper
    }

    /// Create a LatticeZipper that yields only top-k paths.
    pub fn k_best(lattice: &'a Lattice, k: usize, prefix: &[u8]) -> KBestLatticeZipper<'a> {
        let paths = lattice.k_best_paths(k);
        KBestLatticeZipper::new(paths, prefix)
    }

    /// Rebuild the path buffer from current path.
    fn rebuild_path_buffer(&mut self) {
        self.path_buffer.clear();
        self.path_buffer.extend_from_slice(&self.prefix);

        if let Some(ref path) = self.current_path {
            // Encode sentence as MORK expression
            // Format: SymbolSize(n) + bytes
            let sentence_bytes = path.sentence.as_bytes();
            if sentence_bytes.len() <= 63 {
                // Single symbol encoding
                self.path_buffer.push(0xC0 | sentence_bytes.len() as u8);
                self.path_buffer.extend_from_slice(sentence_bytes);
            } else {
                // Long symbol handling (implementation-specific)
                self.path_buffer.extend_from_slice(sentence_bytes);
            }
        }
    }

    /// Get the weight of the current path.
    pub fn current_weight(&self) -> Option<f32> {
        self.current_path.as_ref().map(|p| p.total_weight)
    }

    /// Get the sentence of the current path.
    pub fn current_sentence(&self) -> Option<&str> {
        self.current_path.as_ref().map(|p| p.sentence.as_str())
    }
}

impl<'a> ZipperIteration for LatticeZipper<'a> {
    fn to_next_val(&mut self) -> bool {
        self.current_path = self.path_iterator.next();
        if self.current_path.is_some() {
            self.rebuild_path_buffer();
            true
        } else {
            false
        }
    }

    fn is_val(&self) -> bool {
        self.current_path.is_some()
    }

    fn to_next_step(&mut self) -> bool {
        self.to_next_val()
    }
}

impl<'a> ZipperAbsolutePath for LatticeZipper<'a> {
    fn path(&self) -> &[u8] {
        &self.path_buffer
    }

    fn origin_path(&self) -> &[u8] {
        &self.path_buffer
    }
}

impl<'a> Zipper for LatticeZipper<'a> {
    type Value = f32;  // Weight as value

    fn val(&self) -> Option<&Self::Value> {
        // Return weight as the "value" for filtering/ranking
        self.current_path.as_ref().map(|p| &p.total_weight)
    }
}

/// K-best lattice zipper for efficient top-k iteration.
pub struct KBestLatticeZipper<'a> {
    paths: Vec<LatticePath>,
    index: usize,
    path_buffer: Vec<u8>,
    prefix: Vec<u8>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> KBestLatticeZipper<'a> {
    pub fn new(paths: Vec<LatticePath>, prefix: &[u8]) -> Self {
        let mut zipper = Self {
            paths,
            index: 0,
            path_buffer: Vec::with_capacity(1024),
            prefix: prefix.to_vec(),
            _marker: std::marker::PhantomData,
        };
        zipper.rebuild_path_buffer();
        zipper
    }

    fn rebuild_path_buffer(&mut self) {
        self.path_buffer.clear();
        self.path_buffer.extend_from_slice(&self.prefix);

        if let Some(path) = self.paths.get(self.index) {
            let sentence_bytes = path.sentence.as_bytes();
            if sentence_bytes.len() <= 63 {
                self.path_buffer.push(0xC0 | sentence_bytes.len() as u8);
                self.path_buffer.extend_from_slice(sentence_bytes);
            } else {
                self.path_buffer.extend_from_slice(sentence_bytes);
            }
        }
    }

    pub fn current_weight(&self) -> Option<f32> {
        self.paths.get(self.index).map(|p| p.total_weight)
    }
}

impl<'a> ZipperIteration for KBestLatticeZipper<'a> {
    fn to_next_val(&mut self) -> bool {
        if self.index + 1 < self.paths.len() {
            self.index += 1;
            self.rebuild_path_buffer();
            true
        } else {
            false
        }
    }

    fn is_val(&self) -> bool {
        self.index < self.paths.len()
    }

    fn to_next_step(&mut self) -> bool {
        self.to_next_val()
    }
}
```

---

## Usage Examples

### Building a Lattice

```rust
use liblevenshtein::lattice::LatticeBuilder;

let mut builder = LatticeBuilder::new()
    .with_input("teh cat");

// Token 0: "teh" → corrections
builder.add_correction(0, 1, "the", 0.0, 0, false);
builder.add_correction(0, 1, "ten", 1.0, 1, false);
builder.add_correction(0, 1, "tea", 1.0, 1, false);

// Token 1: "cat" → corrections
builder.add_correction(1, 2, "cat", 0.0, 0, false);
builder.add_correction(1, 2, "bat", 1.0, 1, false);
builder.add_correction(1, 2, "car", 1.0, 1, false);

let lattice = builder.build(2);

// Get top-5 paths
for path in lattice.k_best_paths(5) {
    println!("{}: weight {:.2}", path.sentence, path.total_weight);
}
// Output:
// the cat: weight 0.00
// the bat: weight 1.00
// the car: weight 1.00
// ten cat: weight 1.00
// tea cat: weight 1.00
```

### Transducer Query with Lattice

```rust
use liblevenshtein::transducer::Transducer;

let transducer = Transducer::new(dictionary, Algorithm::Standard);
let lattice = transducer.query_lattice(&["teh", "quik", "brwn", "fox"], 2);

println!("Path count: {}", lattice.path_count());

// Get top corrections
for path in lattice.k_best_paths(10) {
    println!("{}: weight {:.2}", path.sentence, path.total_weight);
}
```

### MORK Query with Ranked Results

```metta
; Query with lattice-based ranked fuzzy matching
!(match &space (fuzzy-ranked "phone" 3 5) $results)

; Returns top 5 matches:
; [(phone 0.0) (fone 1.0) (phon 1.0) (phones 1.0) (foam 2.0)]
```

---

## Performance Considerations

### Lattice Size

- **Nodes**: O(n+1) where n = number of tokens
- **Edges**: O(n × k) where k = average candidates per token
- **Paths**: O(k^n) worst case (exponential)

### Memory Optimization

1. **Vocabulary deduplication**: Labels stored once via `IndexMap`
2. **SmallVec for edges**: Inline storage for small edge lists
3. **Lazy path iteration**: Paths generated on demand

### K-Best Extraction

- **Dijkstra-based**: O(k × log(E)) where E = number of edges
- **Avoids path explosion**: Only materializes top-k paths

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_builder() {
        let mut builder = LatticeBuilder::new();
        builder.add_correction(0, 1, "a", 0.0, 0, false);
        builder.add_correction(0, 1, "b", 1.0, 1, false);
        builder.add_correction(1, 2, "c", 0.0, 0, false);

        let lattice = builder.build(2);

        assert_eq!(lattice.nodes.len(), 3);  // Start, pos 1, end
        assert_eq!(lattice.edges.len(), 3);
        assert_eq!(lattice.path_count(), 2);  // "a c" and "b c"
    }

    #[test]
    fn test_k_best_paths() {
        let mut builder = LatticeBuilder::new();
        builder.add_correction(0, 1, "a", 0.0, 0, false);
        builder.add_correction(0, 1, "b", 1.0, 1, false);
        builder.add_correction(0, 1, "c", 2.0, 2, false);

        let lattice = builder.build(1);
        let paths = lattice.k_best_paths(2);

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].sentence, "a");
        assert_eq!(paths[0].total_weight, 0.0);
        assert_eq!(paths[1].sentence, "b");
        assert_eq!(paths[1].total_weight, 1.0);
    }
}
```

---

## References

- [Lattice Data Structures Specification](../wfst/lattice_data_structures.md)
- [Phase A: FuzzySource](./fuzzy_source.md)
- [Phase C: WFST Composition](./wfst_composition.md)
