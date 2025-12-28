# liblevenshtein-rust Architecture

**Version:** 0.4.0
**Last Updated:** 2025-10-26

This document describes the architecture and design principles of liblevenshtein-rust.

---

## Table of Contents

- [Overview](#overview)
- [Module Organization](#module-organization)
- [Core Components](#core-components)
- [Design Principles](#design-principles)
- [Performance Architecture](#performance-architecture)
- [Thread Safety](#thread-safety)
- [Future Directions](#future-directions)

---

## Overview

liblevenshtein-rust is a high-performance library for approximate string matching using Levenshtein automata. The architecture is designed around three core pillars:

1. **Pluggable Dictionary Backends** - Multiple trie implementations (PathMap, DoubleArrayTrie, DynamicDawg)
2. **Efficient Automata** - Optimized state machines for Levenshtein distance computation
3. **Flexible Serialization** - Multiple formats with optional compression

### Key Characteristics

- **Type-safe** - Extensive use of Rust's type system for correctness
- **Zero-cost abstractions** - Trait-based design with no runtime overhead
- **Memory-efficient** - Structural sharing via Arc, object pooling, stack allocation
- **Concurrent-safe** - RwLock-based interior mutability for shared dictionaries
- **Feature-gated** - Modular compilation with optional features

---

## Module Organization

```
src/
├── lib.rs                      # Public API, prelude, feature gates
│
├── dictionary/                 # Dictionary backends (trait-based)
│   ├── mod.rs                  # Dictionary trait, DictionaryNode trait
│   ├── factory.rs              # Unified factory for creating dictionaries
│   ├── pathmap.rs              # PathMap backend (default, thread-safe)
│   ├── double_array_trie.rs    # DoubleArrayTrie (O(1) transitions, read-optimized)
│   ├── dynamic_dawg.rs         # Mutable DAWG (insert/delete/minimize)
│   └── suffix_automaton.rs     # Suffix automaton (substring matching)
│
├── transducer/                 # Levenshtein automata
│   ├── mod.rs                  # Transducer struct, public API
│   ├── algorithm.rs            # Algorithm enum (Standard, Transposition, MergeAndSplit)
│   ├── state.rs                # Automaton state representation
│   ├── state_pool.rs           # Object pool for state reuse (major optimization)
│   ├── transition.rs           # State transition logic
│   ├── position.rs             # Position in automaton
│   ├── intersection.rs         # Dictionary-automaton intersection
│   ├── query.rs                # Unordered query iterator
│   ├── ordered_query.rs        # Distance-first, lexicographic query iterator
│   └── builder_api.rs          # Fluent builder pattern (experimental)
│
├── distance/                   # Distance calculations
│   └── mod.rs                  # Levenshtein, Transposition distance functions
│
├── serialization/              # Dictionary persistence (feature: serialization)
│   ├── mod.rs                  # Traits, format implementations
│   ├── bincode_impl.rs         # Bincode format
│   ├── json_impl.rs            # JSON format
│   ├── protobuf_impl.rs        # Protobuf format (feature: protobuf)
│   ├── gzip.rs                 # Gzip compression wrapper (feature: compression)
│   └── proto.rs                # Generated protobuf code
│
├── cli/                        # Shared CLI/REPL infrastructure (feature: cli)
│   ├── mod.rs                  # CLI module root, public exports
│   ├── args.rs                 # Clap argument parsing (CLI-specific)
│   ├── commands.rs             # Shared command execution (load, save, query)
│   ├── detect.rs               # Format auto-detection (shared)
│   └── paths.rs                # Path utilities, persistent config (shared)
│
└── repl/                       # Interactive REPL (feature: cli)
    ├── mod.rs                  # REPL module root
    ├── state.rs                # REPL state management (uses cli::commands)
    ├── command.rs              # REPL command parsing and execution
    ├── helper.rs               # Rustyline integration
    └── highlighter.rs          # Syntax highlighting
```

---

## Core Components

### 1. Dictionary Abstraction

**Trait:** `Dictionary`

```rust
pub trait Dictionary: Send + Sync {
    type Node: DictionaryNode;

    fn root(&self) -> Self::Node;
    fn len(&self) -> Option<usize>;
    fn sync_strategy(&self) -> SyncStrategy;
}
```

**Purpose:** Provides a unified interface for different trie implementations.

**Implementations:**
- **PathMapDictionary** - High-performance trie with structural sharing (default)
  - Uses `Arc<RwLock<PathMap>>` for thread-safe mutations
  - Excellent query performance
  - Lazy edge iteration (15-50% faster)

- **DawgDictionary** - Static DAWG (Directed Acyclic Word Graph)
  - Space-efficient via node sharing
  - Read-only after construction
  - Best for static dictionaries

- **DynamicDawg** - Mutable DAWG
  - Online insert/delete operations
  - Minimize operation for compaction
  - Reference-counted nodes (Arc)

**Dictionary Factory Pattern:**

```rust
pub enum DictContainer {
    PathMap(PathMapDictionary),
    Dawg(DawgDictionary),
    DynamicDawg(DynamicDawg),
}

impl DictContainer {
    pub fn from_backend(backend: DictionaryBackend, terms: Vec<String>) -> Self {
        match backend {
            DictionaryBackend::PathMap => Self::PathMap(PathMapDictionary::from_terms(terms)),
            DictionaryBackend::Dawg => Self::Dawg(DawgDictionary::from_terms(terms)),
            DictionaryBackend::DynamicDawg => Self::DynamicDawg(DynamicDawg::from_terms(terms)),
        }
    }
}
```

### 2. Transducer (Levenshtein Automaton)

**Struct:** `Transducer<D: Dictionary>`

```rust
pub struct Transducer<D: Dictionary> {
    dictionary: D,
    algorithm: Algorithm,
}
```

**Purpose:** Coordinates dictionary traversal with Levenshtein automaton.

**Key Methods:**
- `query(term, max_distance)` - Unordered results
- `query_ordered(term, max_distance)` - Distance-first, then lexicographic
- `query_with_distance(term, max_distance)` - Include edit distances

**Query Iterator Architecture:**

```
User Query
    ↓
Transducer::query()
    ↓
QueryIterator
    ↓
┌─────────────────┬──────────────────┐
│  Dictionary     │   Automaton      │
│  Traversal      │   State Machine  │
└─────────────────┴──────────────────┘
    ↓                      ↓
 Trie Edges          State Transitions
    ↓                      ↓
        Intersection
             ↓
         Results
```

### 3. Ordered Query Iterator

**Problem:** Results from basic query are unordered (depth-first traversal)

**Solution:** Heap-based priority queue for ordered results

```rust
pub struct OrderedQueryIterator<N: DictionaryNode> {
    heap: BinaryHeap<Reverse<Intersection<N>>>,
    seen: HashSet<Vec<u8>>,
    max_distance: usize,
}
```

**Ordering:** Primary by distance (ascending), secondary by term (lexicographic)

**Optimizations:**
- Deduplication via `HashSet`
- Prefix mode support
- Custom predicate filtering

### 4. State Pool (Object Pool Pattern)

**Problem:** Repeated allocation/deallocation of State objects in hot paths

**Solution:** `StatePool` - reuse State instances

```rust
pub struct StatePool {
    pool: RefCell<Vec<State>>,
}

pub struct PooledState<'pool> {
    state: Option<State>,
    pool: &'pool StatePool,
}

impl Drop for PooledState<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.pool.pool.borrow_mut().push(state);
        }
    }
}
```

**Impact:** Exceptional performance gains (see `docs/PERFORMANCE.md`)

### 5. Serialization System

**Trait-based design:**

```rust
pub trait DictionarySerializer {
    fn serialize<D, W>(&self, dictionary: &D, writer: W) -> Result<()>
    where
        D: Dictionary,
        W: Write;

    fn deserialize<D, R>(&self, reader: R) -> Result<D>
    where
        D: DictionaryFromTerms,
        R: Read;
}
```

**Implementations:**
- `BincodeSerializer` - Binary format (fast, compact)
- `JsonSerializer` - JSON format (human-readable)
- `ProtobufSerializer` - Protocol Buffers (cross-language)
- `GzipSerializer<S>` - Wrapper for compression (85% size reduction)

**Format Auto-Detection:**
- Magic bytes (exact)
- File extension (heuristic)
- Content analysis (fallback)

### 6. CLI/REPL Architecture

**Design:** Shared infrastructure with separate interfaces

```
┌─────────────────────────────────────┐
│         User Interface              │
├──────────────────┬──────────────────┤
│   CLI (Clap)     │   REPL (Rustyline)│
│   - args.rs      │   - command.rs   │
│   - One-shot     │   - Interactive  │
└──────────────────┴──────────────────┘
           ↓                ↓
┌─────────────────────────────────────┐
│      Shared Infrastructure          │
│   (src/cli/)                        │
│   - commands.rs                     │
│     • load_dictionary()             │
│     • save_dictionary()             │
│     • query operations              │
│   - detect.rs                       │
│     • detect_format()               │
│   - paths.rs                        │
│     • config management             │
│     • path validation               │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│      Core Library                   │
│   - Dictionary                      │
│   - Transducer                      │
│   - Serialization                   │
└─────────────────────────────────────┘
```

**Shared Components:**
- `cli::commands::load_dictionary()` - Used by both CLI and REPL
- `cli::commands::save_dictionary()` - Used by both CLI and REPL
- `cli::detect::detect_format()` - Format auto-detection
- `cli::paths::PersistentConfig` - Configuration management
- `cli::args::SerializationFormat` - Format enum

**CLI-Specific:**
- `cli::args` - Clap command-line parsing (struct-based)
- One-shot execution model

**REPL-Specific:**
- `repl::command` - Text-based command parsing
- `repl::state` - Persistent REPL session state
- `repl::helper` - Rustyline integration (completion, history)
- `repl::highlighter` - Syntax highlighting
- Interactive session model

**Benefits:**
- Code reuse between CLI and REPL
- Consistent behavior (same load/save logic)
- Single source of truth for format detection
- Easier maintenance (fix once, works in both)

---

## Design Principles

### 1. Trait-Based Polymorphism

**Philosophy:** Use traits for abstraction, concrete types for performance

```rust
// Abstraction via traits
pub trait Dictionary { ... }

// Concrete implementations optimized for specific use cases
pub struct PathMapDictionary { ... }
pub struct DawgDictionary { ... }
```

**Benefits:**
- Zero-cost abstraction
- Easy to add new backends
- Type-safe API

### 2. Interior Mutability with RwLock

**Pattern:** Shared ownership + interior mutability

```rust
pub struct PathMapDictionary {
    map: Arc<RwLock<PathMap<()>>>,
    term_count: Arc<RwLock<usize>>,
}
```

**Benefits:**
- Thread-safe concurrent queries
- Modifications don't invalidate existing references
- `Clone` is cheap (Arc increment)

**Trade-offs:**
- Lock acquisition overhead
- Potential lock contention

### 3. Arc Path Sharing

**Optimization:** Share paths via Arc instead of cloning

```rust
pub struct PathMapNode {
    map: Arc<RwLock<PathMap<()>>>,
    path: Arc<Vec<u8>>,  // Shared, not cloned
}
```

**Before:**
```rust
let mut new_path = self.path.clone();  // Deep copy
new_path.push(label);
```

**After:**
```rust
let new_path = {
    let mut path = Vec::with_capacity(self.path.len() + 1);
    path.extend_from_slice(&self.path);
    path.push(label);
    Arc::new(path)  // Cheap reference counting
};
```

### 4. SmallVec for Stack Allocation

**Pattern:** Stack-allocate small collections

```rust
use smallvec::SmallVec;

// Stack-allocated for ≤8 elements, heap for larger
type EdgeList = SmallVec<[(u8, Node); 8]>;
```

**Benefits:**
- Reduced heap allocations
- Better cache locality
- Zero overhead for small cases

### 5. Lazy Evaluation

**Pattern:** Generate results on-demand via iterators

```rust
pub fn query<'a>(
    &'a self,
    term: &'a str,
    max_distance: usize,
) -> impl Iterator<Item = String> + 'a {
    QueryIterator::new(...)
}
```

**Benefits:**
- Memory-efficient (no intermediate collections)
- Early termination support
- Composable with adapters (`.filter()`, `.take()`, etc.)

### 6. Feature Gates

**Pattern:** Modular compilation with cargo features

```toml
[features]
default = []
serialization = ["serde", "bincode", "serde_json"]
protobuf = ["prost", "bytes", "prost-build", "serialization"]
compression = ["flate2", "serialization"]
cli = ["clap", "anyhow", "rustyline", "colored", "dirs"]
```

**Benefits:**
- Minimal dependencies for library-only use
- Faster compilation
- Smaller binary size

---

## Performance Architecture

### Optimization Stack

```
┌─────────────────────────────────────┐
│  Application Level                  │
│  - Lazy evaluation                  │
│  - Early termination                │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Algorithm Level                    │
│  - Ordered heap-based iteration     │
│  - Prefix mode optimization         │
│  - Filter predicate evaluation      │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Data Structure Level               │
│  - Arc path sharing                 │
│  - SmallVec stack allocation        │
│  - Lazy edge iteration              │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Memory Management Level            │
│  - StatePool object reuse           │
│  - Reference counting (Arc)         │
│  - Interior mutability (RwLock)     │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Compiler Level                     │
│  - Aggressive inlining (#[inline])  │
│  - Target-specific features         │
│  - LTO (Link-Time Optimization)     │
└─────────────────────────────────────┘
```

### Hot Paths

**Identified via profiling (flamegraphs):**

1. **State transitions** - `transition.rs`
   - Inlined transition functions
   - Optimized epsilon closure

2. **Dictionary traversal** - `pathmap.rs`, `dawg_query.rs`
   - Lazy edge iteration
   - Index-based queries for DAWG

3. **Result collection** - `ordered_query.rs`
   - Heap-based priority queue
   - Efficient deduplication

### Benchmarking

**Tools:**
- Criterion.rs - Statistical benchmarking
- Flamegraph - CPU profiling
- cargo-llvm-cov - Code coverage

**Key Metrics:**
- Query throughput (queries/second)
- Result latency (ms to first result)
- Memory usage (bytes per term)
- Serialization speed (MB/s)

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for detailed performance analysis.

---

## Thread Safety

### Concurrent Access Patterns

**Scenario 1: Multiple Readers**
```rust
// Multiple threads can query simultaneously
let dict = PathMapDictionary::from_terms(terms);
let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

thread::spawn(move || {
    for result in transducer.query("test", 2) {
        println!("{}", result);
    }
});

// Another thread queries in parallel
for result in transducer.query("hello", 1) {
    println!("{}", result);
}
```

**Scenario 2: Concurrent Modification**
```rust
let dict = PathMapDictionary::from_terms(terms);
let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

// Background thread modifies dictionary
thread::spawn(move || {
    dict.insert("new_term");
});

// Main thread queries (sees updates after write lock released)
for result in transducer.query("new", 1) {
    println!("{}", result);
}
```

### SyncStrategy Enum

```rust
pub enum SyncStrategy {
    Persistent,    // Immutable, always safe (Dawg)
    InternalSync,  // Internal synchronization (future: lock-free structures)
    ExternalSync,  // Requires RwLock (PathMap, DynamicDawg)
}
```

**Purpose:** Communicates thread-safety requirements to library users

---

## Future Directions

### Planned Enhancements

1. **SIMD Optimizations** (v0.4.0)
   - Vectorize distance calculations
   - Parallel edge traversal
   - Target: 2-3x speedup for large queries

2. **Async Support** (v0.5.0)
   - `async fn query()` returning `Stream<Item = String>`
   - Non-blocking I/O for serialization
   - Tokio/async-std compatibility

3. **Custom Allocators** (v0.6.0)
   - Arena allocators for query sessions
   - Reduce fragmentation
   - Predictable memory usage

4. **GPU Acceleration** (Research)
   - Large-scale parallel queries
   - CUDA/OpenCL backends
   - Target: 10-100x for very large dictionaries

### Potential Refactorings

1. **Error Types** - Replace `anyhow::Error` with `thiserror` types
2. **Modular Serialization** - Split `serialization/mod.rs` into `formats/`
3. **Query Builder** - Promote `builder_api.rs` to stable API
4. **Plugin System** - Dynamic loading of dictionary backends

See [`docs/FUTURE_ENHANCEMENTS.md`](docs/FUTURE_ENHANCEMENTS.md) for details.

---

## References

- **Performance:** [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- **Features:** [`docs/FEATURES.md`](docs/FEATURES.md)
- **Contributing:** [`contributing.md`](CONTRIBUTING.md)
- **Build Guide:** [`building.md`](BUILD.md)

---

## Summary

liblevenshtein-rust achieves high performance through:

1. **Efficient data structures** - Tries with structural sharing
2. **Smart memory management** - Object pooling, Arc sharing, stack allocation
3. **Zero-cost abstractions** - Trait-based design with no runtime overhead
4. **Profiling-driven optimization** - Targeted improvements to hot paths
5. **Concurrent-safe design** - RwLock-based interior mutability

The architecture is designed to be **extensible** (easy to add backends, formats, algorithms) while maintaining **performance** (zero-cost abstractions, profiling-guided optimizations) and **safety** (Rust's type system, thread-safety guarantees).
