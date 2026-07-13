# Hierarchical Zipper Architecture for Contextual Code Completion

[← Documentation Index](../README.md)

## Document Status
- **Status**: Design Approved
- **Created**: 2025-10-31
- **Last Updated**: 2025-10-31
- **Implementation Status**: Not Started

![Context scope tree: nested lexical scopes (global → module → class → function) and how a child context inherits visibility of its ancestors' draft and finalized terms.](../diagrams/contextual/context-scope-tree.svg)

## Table of Contents
1. [Overview](#overview)
2. [Use Case](#use-case)
3. [Design Requirements](#design-requirements)
4. [Architecture](#architecture)
5. [Zipper Hierarchy](#zipper-hierarchy)
6. [Implementation Plan](#implementation-plan)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [References](#references)

## Overview

This document describes the design for a hierarchical zipper system that enables contextual code completion with concurrent read-write operations. The system allows character-by-character insertion of terms while simultaneously querying for spelling corrections, with full support for hierarchical lexical scopes and undo/redo operations.

### Key Features
- **Concurrent read-write**: Insert characters while querying for completions
- **Character-level rollback**: Undo individual character insertions (backspace support)
- **Checkpoint-based undo/redo**: Save and restore insertion states
- **Hierarchical scope visibility**: Child contexts see parent drafts (lexical scoping)
- **Multi-backend support**: Works with PathMapDictionary and DoubleArrayTrie
- **Zero-cost abstractions**: Trait-based design with monomorphization

## Use Case

### Contextual Code Completion Scenario

As a user types an identifier in their code editor:

1. **Character insertion**: Each keystroke inserts a character into a draft term
2. **Query for completions**: After each character, query for fuzzy matches
3. **Exact match by construction**: The draft term itself appears as exact match
4. **Correction candidates**: Other terms appear as spelling corrections
5. **Scope filtering**: Only terms visible in current lexical scope are shown
6. **Hierarchical visibility**: Terms from parent scopes (global, module, class) are visible
7. **Finalization**: When user accepts a completion, mark it as final with context ID
8. **Rollback support**: Backspace removes characters, undo/redo restore states

### Example Flow

```rust
let engine = ContextualCompletionEngine::new();

// User is typing in function scope (context 42)
// Parent contexts: global (0), module (1), class (10)
let context_hierarchy = vec![0, 1, 10, 42];

// User types 'p'
engine.insert_char(42, 'p')?;
let completions = engine.complete("p", 1, &context_hierarchy);
// Results: ["print", "parse", "push", "p" (draft)]

// User types 'r'
engine.insert_char(42, 'r')?;
let completions = engine.complete("pr", 1, &context_hierarchy);
// Results: ["print", "printf", "process", "pr" (draft)]

// User saves checkpoint for undo
let checkpoint = engine.checkpoint(42)?;

// User types 'i'
engine.insert_char(42, 'i')?;
let completions = engine.complete("pri", 1, &context_hierarchy);
// Results: ["print", "printf", "private", "pri" (draft)]

// User hits backspace
engine.rollback_char(42)?;
// Back to "pr"

// User hits Ctrl+Z (undo to checkpoint)
engine.restore(42, checkpoint)?;
// Back to "p"

// User accepts "print" completion
engine.finalize(42, vec![42])?;  // Mark visible in context 42
// "print" is now a finalized term
```

## Design Requirements

### Functional Requirements

1. **FR1**: Insert characters one at a time into draft terms
2. **FR2**: Query for fuzzy matches including draft and finalized terms
3. **FR3**: Rollback individual character insertions (backspace)
4. **FR4**: Create checkpoints and restore to them (undo/redo)
5. **FR5**: Finalize draft terms with context ID associations
6. **FR6**: Discard draft terms without finalizing
7. **FR7**: Filter query results by hierarchical context visibility
8. **FR8**: Support all three Levenshtein algorithm variants
9. **FR9**: Work with multiple dictionary backends

### Non-Functional Requirements

1. **NFR1**: Thread-safe concurrent reads and writes
2. **NFR2**: Performance within 10% of existing query operations
3. **NFR3**: Memory overhead < 1KB per active draft context
4. **NFR4**: Lock-free or fine-grained locking for concurrency
5. **NFR5**: Backward compatible with existing Transducer API
6. **NFR6**: Zero-cost abstractions (monomorphization, no trait objects)

### Design Constraints

1. **DC1**: Must extend existing Transducer API, not replace it
2. **DC2**: Must work with PathMapDictionary (primary) and DoubleArrayTrie
3. **DC3**: Cannot modify existing Dictionary/DictionaryNode traits
4. **DC4**: Must reuse existing StatePool and Position types
5. **DC5**: Should leverage existing PathMap zipper infrastructure

## Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│         Contextual Completion Engine (Layer 4)          │
│  - Draft management (insert/rollback/checkpoint)        │
│  - Hierarchical context tree                            │
│  - Query fusion (finalized + drafts)                    │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Finalized Terms │     │  Draft Buffers  │
│  (PathMap/DAT)  │     │  (per-context)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Intersection Zipper   │
         │     (Layer 3)         │
         └─────┬───────────┬─────┘
               │           │
       ┌───────▼─────┐ ┌──▼──────────────┐
       │  Dictionary │ │   Automaton     │
       │   Zipper    │ │    Zipper       │
       │  (Layer 1)  │ │   (Layer 2)     │
       └─────────────┘ └─────────────────┘
```

### Component Responsibilities

#### Layer 1: Dictionary Zipper
- **Purpose**: Navigate dictionary structure (DAWG, PathMap, DAT)
- **Operations**: descend, children, is_final, value
- **Implementations**: PathMapZipper, DoubleArrayTrieZipper
- **Abstraction**: DictionaryZipper trait

#### Layer 2: Automaton Zipper
- **Purpose**: Track Levenshtein automaton state during traversal
- **Operations**: transition, min_distance, is_accepting
- **State**: State (positions), query, algorithm, max_distance
- **Integration**: Uses StatePool for allocation reuse

#### Layer 3: Intersection Zipper
- **Purpose**: Compose dictionary and automaton zippers
- **Operations**: children (combined navigation), is_match, distance, term
- **Parent tracking**: Lightweight PathNode chain for path reconstruction
- **Generic**: Works with any DictionaryZipper implementation

#### Layer 4: Contextual Completion Engine
- **Purpose**: Manage drafts, finalization, and hierarchical queries
- **Draft tracking**: Per-context buffers with character history
- **Checkpoints**: Save/restore for undo/redo
- **Query fusion**: Union of finalized terms + visible draft terms
- **Context tree**: Hierarchical structure for visibility resolution

## Zipper Hierarchy

### Layer 1: Dictionary Zipper Trait

```rust
/// Core trait for dictionary navigation
pub trait DictionaryZipper: Clone {
    type Unit: CharUnit;

    /// Check if current position marks end of term
    fn is_final(&self) -> bool;

    /// Navigate to child node with given label
    /// Returns None if no such child exists
    fn descend(&self, label: Self::Unit) -> Option<Self>;

    /// Iterator over all children (label, child zipper pairs)
    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)>;

    /// Get current path from root (for debugging)
    fn path(&self) -> Vec<Self::Unit>;
}

/// Extension for dictionaries with associated values
pub trait ValuedDictionaryZipper: DictionaryZipper {
    type Value: DictionaryValue;

    /// Get value at current position if final
    fn value(&self) -> Option<Self::Value>;
}
```

#### PathMapZipper Implementation

```rust
/// Zipper for PathMapDictionary backend (lock-free, TrieRef-based)
pub struct PathMapZipper<V: DictionaryValue> {
    /// Lock-free focus handle over a copy-on-write `TrieRef` snapshot of the PathMap
    root: TrieRefLike,

    /// Current path from root
    path: Arc<[u8]>,
}

impl<V: DictionaryValue> DictionaryZipper for PathMapZipper<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        self.with_zipper(|z| z.is_val())
    }

    fn descend(&self, label: u8) -> Option<Self> {
        let mut new_path = (*self.path).clone();
        new_path.push(label);

        // Check if path exists
        let exists = {
            let map = self.map.read().unwrap();
            let mut zipper = map.read_zipper();
            zipper.descend_to(&new_path);
            zipper.path_exists()
        };

        if exists {
            Some(PathMapZipper {
                map: Arc::clone(&self.map),
                path: Arc::new(new_path),
            })
        } else {
            None
        }
    }

    fn children(&self) -> impl Iterator<Item = (u8, Self)> {
        let mask = self.with_zipper(|z| z.child_mask());
        let map = Arc::clone(&self.map);
        let path = Arc::clone(&self.path);

        (0u8..=255)
            .filter(move |&byte| mask.contains(byte))
            .filter_map(move |byte| {
                PathMapZipper {
                    map: Arc::clone(&map),
                    path: Arc::clone(&path)
                }.descend(byte).map(|z| (byte, z))
            })
    }

    fn path(&self) -> Vec<u8> {
        (*self.path).clone()
    }

    // Helper for lock-per-operation pattern
    fn with_zipper<F, R>(&self, f: F) -> R
    where
        F: FnOnce(ReadZipperUntracked<'_, 'static, V>) -> R,
    {
        let map = self.map.read().unwrap();
        let mut zipper = map.read_zipper();
        zipper.descend_to(&*self.path);
        f(zipper)
    }
}

impl<V: DictionaryValue> ValuedDictionaryZipper for PathMapZipper<V> {
    type Value = V;

    fn value(&self) -> Option<V> {
        self.with_zipper(|z| z.val().cloned())
    }
}
```

#### DoubleArrayTrieZipper Implementation

```rust
/// Zipper for DoubleArrayTrie backend (read-only)
#[derive(Clone)]
pub struct DoubleArrayTrieZipper {
    /// Reference to trie structure
    trie: Arc<DoubleArrayTrie>,

    /// Current node index (Copy type for efficiency)
    node_index: usize,

    /// Accumulated path for debugging
    path: Arc<Vec<u8>>,
}

impl DictionaryZipper for DoubleArrayTrieZipper {
    type Unit = u8;

    fn is_final(&self) -> bool {
        self.trie.is_final(self.node_index)
    }

    fn descend(&self, label: u8) -> Option<Self> {
        self.trie.transition(self.node_index, label).map(|next_index| {
            let mut new_path = (*self.path).clone();
            new_path.push(label);

            DoubleArrayTrieZipper {
                trie: Arc::clone(&self.trie),
                node_index: next_index,
                path: Arc::new(new_path),
            }
        })
    }

    fn children(&self) -> impl Iterator<Item = (u8, Self)> {
        let trie = Arc::clone(&self.trie);
        let node_index = self.node_index;
        let path = Arc::clone(&self.path);

        (0u8..=255)
            .filter_map(move |byte| {
                trie.transition(node_index, byte).map(|next_index| {
                    let mut new_path = (*path).clone();
                    new_path.push(byte);

                    let zipper = DoubleArrayTrieZipper {
                        trie: Arc::clone(&trie),
                        node_index: next_index,
                        path: Arc::new(new_path),
                    };

                    (byte, zipper)
                })
            })
    }

    fn path(&self) -> Vec<u8> {
        (*self.path).clone()
    }
}
```

### Layer 2: Automaton Zipper

```rust
/// Zipper for Levenshtein automaton state
#[derive(Clone)]
pub struct AutomatonZipper {
    /// Current automaton state (positions)
    state: State,

    /// Query string (shared across all zippers in a query)
    query: Arc<Vec<u8>>,

    /// Maximum allowed edit distance
    max_distance: usize,

    /// Algorithm variant (Standard, Transposition, MergeAndSplit)
    algorithm: Algorithm,
}

impl AutomatonZipper {
    /// Create root automaton zipper for query
    pub fn new(query: &[u8], max_distance: usize, algorithm: Algorithm) -> Self {
        let mut state = State::new();

        // Initialize with starting positions for each distance threshold
        for distance in 0..=max_distance {
            state.insert(Position {
                term_index: 0,
                num_errors: distance,
                is_special: false,
            });
        }

        AutomatonZipper {
            state,
            query: Arc::new(query.to_vec()),
            max_distance,
            algorithm,
        }
    }

    /// Transition automaton state by consuming a character
    pub fn transition(&self, label: u8, pool: &mut StatePool) -> Option<Self> {
        transition_state_pooled(
            &self.state,
            pool,
            label,
            &self.query,
            self.max_distance,
            self.algorithm,
            false, // substring_mode
        ).map(|next_state| AutomatonZipper {
            state: next_state,
            query: Arc::clone(&self.query),
            max_distance: self.max_distance,
            algorithm: self.algorithm,
        })
    }

    /// Get minimum distance if state is accepting
    pub fn min_distance(&self) -> Option<usize> {
        self.state.min_distance()
    }

    /// Infer distance for a term of given length
    pub fn infer_distance(&self, term_length: usize) -> Option<usize> {
        self.state.infer_distance(term_length)
    }

    /// Check if state could lead to valid matches
    pub fn is_viable(&self) -> bool {
        !self.state.is_empty()
    }
}
```

### Layer 3: Intersection Zipper

```rust
/// Zipper representing intersection of dictionary and automaton
pub struct IntersectionZipper<D: DictionaryZipper> {
    /// Dictionary zipper at current position
    dict: D,

    /// Automaton zipper at current state
    automaton: AutomatonZipper,

    /// Parent chain for path reconstruction (lightweight)
    parent: Option<Box<PathNode<D::Unit>>>,
}

impl<D: DictionaryZipper> IntersectionZipper<D> {
    /// Create root intersection zipper
    pub fn new(dict: D, automaton: AutomatonZipper) -> Self {
        IntersectionZipper {
            dict,
            automaton,
            parent: None,
        }
    }

    /// Check if current position represents a match
    pub fn is_match(&self) -> bool {
        self.dict.is_final() && self.distance().is_some()
    }

    /// Get distance for current match (if it is a match)
    pub fn distance(&self) -> Option<usize> {
        if self.dict.is_final() {
            // Reconstruct term length from parent chain
            let term_length = self.depth();
            self.automaton.infer_distance(term_length)
        } else {
            None
        }
    }

    /// Get depth (term length) from parent chain
    pub fn depth(&self) -> usize {
        self.parent.as_ref().map(|p| p.depth()).unwrap_or(0) + 1
    }

    /// Reconstruct full term from parent chain
    pub fn term(&self) -> String {
        let labels = self.collect_labels();
        String::from_utf8_lossy(&labels).into_owned()
    }

    /// Navigate to children (combined dictionary + automaton transitions)
    pub fn children(&self, pool: &mut StatePool)
        -> impl Iterator<Item = IntersectionZipper<D>> + '_
    {
        let parent_for_children = self.parent.clone();

        self.dict.children()
            .filter_map(move |(label, dict_child)| {
                // Try to transition automaton with this label
                self.automaton.transition(label, pool).map(|auto_child| {
                    // Create parent node for child
                    let new_parent = Some(Box::new(PathNode::new(
                        label,
                        parent_for_children.clone(),
                    )));

                    IntersectionZipper {
                        dict: dict_child,
                        automaton: auto_child,
                        parent: new_parent,
                    }
                })
            })
    }

    // Helper to collect labels from parent chain
    fn collect_labels(&self) -> Vec<D::Unit> {
        let mut labels = Vec::new();
        let mut current = &self.parent;

        while let Some(node) = current {
            labels.push(node.label());
            current = node.parent();
        }

        labels.reverse();
        labels
    }
}
```

### Layer 4: Contextual Completion Engine

See [contextual-completion-api.md](./contextual-completion-api.md) for complete API design.

## Implementation Plan

See [contextual-completion-roadmap.md](./contextual-completion-roadmap.md) for detailed implementation roadmap.

### Phase Overview

1. **Phase 1**: Core zipper traits and basic implementations
2. **Phase 2**: Automaton and intersection zipper composition
3. **Phase 3**: Draft buffer with character-level rollback and checkpoints
4. **Phase 4**: Contextual completion engine with hierarchical visibility
5. **Phase 5**: Transducer integration and backward compatibility
6. **Phase 6**: Optimization, benchmarking, and documentation

## Testing Strategy

### Unit Tests

Each layer should have comprehensive unit tests:

#### Layer 1: Dictionary Zipper Tests
- Basic navigation (descend, children)
- Boundary conditions (empty path, invalid transitions)
- Value access (finality, value retrieval)
- Backend-specific tests (PathMap, DoubleArrayTrie)

#### Layer 2: Automaton Zipper Tests
- State initialization with all distance thresholds
- Transition correctness for each algorithm variant
- Distance computation accuracy
- StatePool integration (allocation reuse)

#### Layer 3: Intersection Zipper Tests
- Combined navigation (dictionary + automaton sync)
- Match detection (is_match, distance)
- Path reconstruction (term accuracy)
- Edge cases (no matches, max distance exceeded)

#### Layer 4: Engine Tests
- Character insertion and rollback
- Checkpoint creation and restoration
- Hierarchical visibility resolution
- Finalization and discard operations
- Concurrent access patterns

### Integration Tests

Test complete workflows:

```rust
#[test]
fn test_contextual_completion_workflow() {
    let engine = ContextualCompletionEngine::new();

    // Setup hierarchy: global(0) -> module(1) -> function(42)
    engine.create_context(0, None);  // root
    engine.create_context(1, Some(0));
    engine.create_context(42, Some(1));

    // Finalize some terms in parent contexts
    engine.finalize_direct(0, "print", vec![0]);  // global
    engine.finalize_direct(1, "parse", vec![1]);  // module

    // User types in function context
    engine.insert_char(42, 'p');
    engine.insert_char(42, 'r');

    // Query with hierarchy
    let results = engine.complete("pr", 1, &[0, 1, 42]);

    // Should see: parent terms + draft
    assert!(results.iter().any(|c| c.term == "print"));  // from global
    assert!(results.iter().any(|c| c.term == "parse"));  // from module
    assert!(results.iter().any(|c| c.term == "pr" && c.is_draft));

    // Finalize draft
    engine.finalize(42, vec![42]);

    // Query again - now "pr" is finalized
    let results = engine.complete("pr", 1, &[0, 1, 42]);
    assert!(results.iter().any(|c| c.term == "pr" && !c.is_draft));
}
```

### Concurrency Tests

Test thread-safety:

```rust
#[test]
fn test_concurrent_insertions_and_queries() {
    let engine = Arc::new(ContextualCompletionEngine::new());
    let mut handles = vec![];

    // Spawn multiple threads inserting in different contexts
    for context_id in 0..8 {
        let engine = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            for ch in "hello".chars() {
                engine.insert_char(context_id, ch).unwrap();
            }
            engine.finalize(context_id, vec![context_id]).unwrap();
        }));
    }

    // Spawn threads querying concurrently
    for _ in 0..8 {
        let engine = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            for i in 1..=5 {
                let query = "hello"[..i].to_string();
                let _ = engine.complete(&query, 1, &[0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all terms were finalized
    let results = engine.complete("hello", 0, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(results.len(), 8);
}
```

### Benchmark Tests

Performance regression tests:

```rust
#[bench]
fn bench_query_with_drafts(b: &mut Bencher) {
    let engine = setup_large_dictionary(10000);

    // Add 100 draft terms
    for i in 0..100 {
        for ch in format!("draft{}", i).chars() {
            engine.insert_char(i, ch).unwrap();
        }
    }

    b.iter(|| {
        engine.complete("draf", 1, &(0..100).collect::<Vec<_>>())
    });
}
```

## Performance Considerations

### Memory Overhead

**Per-context draft tracking:**
- `VecDeque<char>`: ~32 bytes + characters (grows incrementally)
- Checkpoints: ~16 bytes per checkpoint (position + metadata)
- Context tree node: ~48 bytes (id, parent, children)

**Estimated overhead:**
- 10 active contexts: ~500 bytes baseline
- 10 characters per draft: ~40 bytes × 10 = 400 bytes
- 5 checkpoints per context: ~80 bytes × 10 = 800 bytes
- **Total: ~1.7KB for typical usage** ✅ (within NFR3: < 1KB per context)

### Query Performance

**Fusion overhead:**
- Finalized query: Uses existing Transducer (baseline)
- Draft scan: Linear in number of visible drafts (typically < 100)
- Union + dedup: `$\mathcal{O}(n \log  n)$` where `n` = total results (typically < 1000)

**Expected impact:**
- Best case (no drafts): 0% overhead
- Typical case (10-20 drafts): ~5-10% overhead
- Worst case (100 drafts): ~20% overhead

**Target: < 10% average overhead** (NFR2)

### Lock Contention

**PathMap lock strategy:**
- Fine-grained: Lock-per-operation on zipper methods
- Read locks: Multiple concurrent queries
- Write locks: Finalization serializes briefly
- Draft locks: DashMap for per-context parallelism

**Expected contention:**
- Read-heavy workload (99% queries, 1% insertions): Minimal
- Write-heavy workload (50% insertions): Moderate
- Mixed workload: Low to moderate

### Optimization Opportunities

1. **Cache compiled queries**: Reuse automaton zipper for repeated queries
2. **Bloom filter for drafts**: Fast negative lookup before scanning
3. **SIMD distance computation**: Vectorize state transitions
4. **Background compaction**: Periodic PathMap compaction in separate thread
5. **Query result caching**: LRU cache for recent (query, contexts) → results

## References

### Related Documents
- [API Design](./contextual-completion-api.md) - Complete API specification
- [Implementation Roadmap](./contextual-completion-roadmap.md) - Detailed implementation plan
- [Progress Tracking](./contextual-completion-progress.md) - Implementation status

### External References
- [Huet's Zipper (1997)](https://www.st.cs.uni-saarland.de/edu/seminare/2005/advanced-fp/docs/huet-zipper.pdf) - Original zipper paper
- [PathMap crate](https://docs.rs/pathmap/) - Persistent trie with zippers
- [Levenshtein Automata](https://julesjacobs.com/2015/06/17/disqus-levenshtein-simple-and-fast.html) - Fast fuzzy matching
- [liblevenshtein architecture](../developer-guide/architecture.md) - Existing system design

### Code Locations
- Existing Dictionary trait: `src/dictionary/mod.rs:111-211`
- Existing Intersection: `src/transducer/intersection.rs`
- Existing StatePool: `src/transducer/pool.rs`
- PathMap integration: `src/dictionary/pathmap.rs`
