# Zipper-Based vs Node-Based Query Performance Analysis

[← Documentation Index](../README.md)

**Date:** 2025-11-03
**Author:** Performance Analysis
**Status:** Completed

---

## Executive Summary

This document analyzes the performance trade-offs between zipper-based and node-based query iteration for fuzzy string matching in liblevenshtein-rust.

**Key Finding:** Zipper-based queries are **1.66-1.97× slower** than node-based queries across all edit distances, but provide essential architectural benefits for contextual completion use cases.

**Recommendation:** Use zipper-based approach for contextual code completion (primary use case) and node-based for simple, high-throughput fuzzy matching.

---

## Performance Benchmark Results

### Single Query Performance

Benchmarked on 115-word English dictionary with query "time":

| Edit Distance | Node-Based | Zipper-Based | Overhead | Speedup (Node) |
|--------------|------------|--------------|----------|----------------|
| **0 (exact)** | 9.56 µs | 15.94 µs | +66.7% | **1.67× faster** |
| **1** | 53.25 µs | 100.14 µs | +88.1% | **1.88× faster** |
| **2** | 156.46 µs | 309.00 µs | +97.5% | **1.97× faster** |

**Observation:** Performance gap widens with edit distance. At distance 2, node-based is nearly 2× faster.

### Throughput Comparison

| Distance | Node-Based | Zipper-Based | Ratio |
|----------|-----------|--------------|-------|
| 0 | 104.60 K queries/sec | 62.75 K queries/sec | 1.67× |
| 1 | 18.78 K queries/sec | 9.99 K queries/sec | 1.88× |
| 2 | 6.39 K queries/sec | 3.24 K queries/sec | 1.97× |

### Batch Query Performance (5 queries)

| Approach | Total Time | Per-Query | Throughput |
|----------|-----------|-----------|------------|
| **Node-based** | 286.15 µs | 57.23 µs | 17.47 K/sec |
| **Zipper-based** | 535.70 µs | 107.14 µs | 9.33 K/sec |
| **Overhead** | +87.2% | +87.2% | -46.7% |

### Iterator Creation Overhead

| Approach | Creation Time | Difference |
|----------|---------------|------------|
| **Node-based** | 216.95 ns | Baseline |
| **Zipper-based** | 220.08 ns | +1.44% |

**Key Finding:** Iterator creation cost is nearly identical. Performance difference comes from traversal, not setup.

---

## Root Cause Analysis: Why Zippers Are Slower

### 1. Indirection Overhead

**Node-based (direct access):**
```rust
// QueryIterator directly accesses nodes
let node = dictionary.get_node(node_id);
let edges = node.edges();
```

**Zipper-based (layered abstraction):**
```rust
// Multiple layers of abstraction
PathMapZipper {
    map: Arc<RwLock<PathMap<V>>>,  // Lock overhead
    path: Arc<[u8]>,                // Path tracking
}

IntersectionZipper {
    dict: PathMapZipper,            // Dictionary zipper
    automaton: AutomatonZipper,     // Automaton zipper
    parent: Option<Box<PathNode>>,  // Term reconstruction
}
```

**Estimated cost:** 30-50 ns per node due to layering.

### 2. Lock Acquisition Overhead

Every zipper operation acquires a read lock:
```rust
fn with_zipper<F, R>(&self, f: F) -> R {
    let map = self.map.read().unwrap();  // Lock acquired
    // ... operation ...
}  // Lock released
```

Operations that acquire locks:
- `descend()`: 1 lock
- `is_final()`: 1 lock
- `children()`: 1 lock (after optimization)
- `value()`: 1 lock

**Cost:** ~10-20 ns per lock (uncontended, single-threaded)
**Frequency:** Multiple times per node during BFS traversal

### 3. Memory Allocation Overhead

**Node-based allocations:**
- Minimal (reuses StatePool, small Vec for queue)

**Zipper-based allocations:**
- `Arc<[u8]>` for each path extension (mitigated by COW optimization)
- `Box<PathNode>` for each child during traversal
- `Vec<(u8, Arc<[u8]>)>` for children collection (lock batching)

**Estimated overhead:** 50-100 ns per node visited

### 4. Struct Size & Cache Efficiency

**Node-based queue item:**
```rust
struct Candidate {
    node_id: usize,        // 8 bytes
    state: State,          // ~32 bytes (SmallVec)
    depth: usize,          // 8 bytes
}
// Total: ~48 bytes
```

**Zipper-based queue item:**
```rust
struct IntersectionZipper {
    dict: PathMapZipper {
        map: Arc<RwLock>,       // 8 bytes (pointer)
        path: Arc<[u8]>,        // 8 bytes (pointer)
    },
    automaton: AutomatonZipper {
        state: State,           // ~32 bytes
        query: Arc<Vec<u8>>,    // 8 bytes (pointer)
        max_distance: usize,    // 8 bytes
        algorithm: Algorithm,   // 1 byte
    },
    parent: Option<Box<PathNode>>,  // 8 bytes (pointer)
}
// Total: ~73+ bytes
```

**Impact:** Larger structs → more memory traffic → cache misses → slower performance

---

## What Zippers Provide (Despite Performance Cost)

### 1. Compositional Architecture ✅

Clean separation of concerns via trait composition:

```rust
// DictZipper trait - any dictionary can be zipped
pub trait DictZipper: Clone {
    type Unit;
    fn is_final(&self) -> bool;
    fn descend(&self, label: Self::Unit) -> Option<Self>;
    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)>;
}

// Compose dictionary + automaton cleanly
let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);
```

**Node-based:** Tightly coupled to specific dictionary type, monolithic implementation.

### 2. Thread Safety by Design ✅

```rust
// Zippers use Arc + RwLock internally
let engine = Arc::new(ContextualCompletionEngine::new());

// Share safely across threads
thread::spawn(move || {
    let results = engine.complete(ctx, "query", 2);  // Thread-safe
});
```

**Node-based:** Requires manual synchronization (Mutex/RwLock wrapper).

### 3. Immutable, Functional Traversal ✅

```rust
let parent = zipper.clone();  // Cheap Arc clone
let child = zipper.descend(b'a').unwrap();

// parent is unchanged - can reuse for other children
let other_child = parent.descend(b'b').unwrap();
```

**Node-based:** Mutable state requires careful management, easy to introduce bugs.

### 4. Hierarchical Context Support ✅

The killer feature for contextual code completion:

```rust
let engine = ContextualCompletionEngine::new();

// Create hierarchy
let global = engine.create_root_context(0);
let function = engine.create_child_context(1, global)?;
let block = engine.create_child_context(2, function)?;

// Add terms at different scopes
engine.finalize_direct(global, "global_var")?;
engine.finalize_direct(function, "param")?;
engine.finalize_direct(block, "local_var")?;

// Query from block sees all ancestor scopes
let results = engine.complete(block, "glo", 2);
// Returns: ["global_var"] (visible from parent)
```

**Node-based:** No concept of context hierarchy - impossible to implement.

### 5. Path Tracking & Term Reconstruction ✅

```rust
// Zipper maintains path automatically
impl PathMapZipper {
    path: Arc<[u8]>,  // Full path from root
}

// Term reconstruction is trivial
let term = intersection.term();  // Builds from PathNode chain
```

**Node-based:** Requires separate term accumulation buffer, manual tracking.

### 6. Draft + Finalized Term Mixing ✅

```rust
// Add finalized term
engine.finalize_direct(ctx, "completed_term")?;

// Add draft (not finalized)
engine.insert_str(ctx, "draft_text")?;

// Query returns both
let results = engine.complete(ctx, "com", 2);
// Returns: ["completed_term"] (finalized) + ["draft_text"] (if matches)
```

**Node-based:** Only supports static dictionary, no draft state.

---

## Performance in Context: Is This Acceptable?

### For Simple Fuzzy Matching: ⚠️ No

If you only need fuzzy string matching against a single static dictionary, the 2× overhead is unnecessary:

```rust
// Use node-based for this
let transducer = Transducer::new(dict, Algorithm::Standard);
let results: Vec<_> = transducer.query("test", 2).collect();
```

**When to use node-based:**
- Batch indexing (millions of queries)
- Single dictionary, no context
- Performance-critical path
- No need for thread safety
- Simple CLI tools

### For Contextual Code Completion: ✅ Yes

The primary use case requires features only zippers provide:

```rust
// Realistic code completion scenario
let engine = Arc::new(ContextualCompletionEngine::new());

// Global scope
let global = engine.create_root_context(0);
engine.finalize_direct(global, "std::vector")?;
engine.finalize_direct(global, "std::string")?;

// Function scope
let function = engine.create_child_context(1, global)?;
engine.finalize_direct(function, "parameter")?;

// Block scope with draft
let block = engine.create_child_context(2, function)?;
engine.insert_str(block, "local_variable")?;

// Query sees all visible contexts + draft
let completions = engine.complete(block, "std", 2);
// Returns: ["std::vector", "std::string"] from global scope
```

**Absolute performance is still excellent:**
- Query at distance 1: **100 µs = 0.1 ms** ✅
- Query at distance 2: **309 µs = 0.3 ms** ✅
- Sub-millisecond latency acceptable for interactive UI

**When to use zipper-based:**
- Code completion engines (LSP servers)
- Hierarchical scoping (global/class/function/block)
- Multi-context queries
- Thread-safe concurrent access
- Draft + finalized mixing
- Interactive applications

---

## Future Optimization Opportunities

Several optimizations could reduce the gap from 1.88-1.97× to ~1.2-1.3×:

### 1. Arena Allocation for PathNode (5-8% improvement)

**Current:**
```rust
// Box allocation per node
let parent = Some(Box::new(PathNode::new(label, parent)));
```

**Proposed:**
```rust
// Arena allocation (all nodes in contiguous memory)
let arena = Arena::new();
let parent = Some(arena.alloc(PathNode::new(label, parent)));
```

**Benefit:** Reduces allocator overhead, improves cache locality.

### 2. Lazy Zipper Initialization (10-15% improvement)

**Current:**
```rust
// Zipper created eagerly
let zipper = PathMapZipper::new_from_dict(&dict);
```

**Proposed:**
```rust
// Only create when first accessed
enum LazyZipper {
    Uninitialized { map: Arc, path: Arc },
    Initialized(PathMapZipper),
}
```

**Benefit:** Defers lock acquisition until needed.

### 3. Lock-Free PathMap Access (20-30% improvement)

**Current:**
```rust
map: Arc<RwLock<PathMap<V>>>  // Lock on every access
```

**Proposed:**
```rust
map: Arc<LockFreePathMap<V>>  // Atomic operations
```

**Benefit:** Eliminates lock overhead entirely (requires PathMap changes).

### 4. Aggressive Inlining (2-5% improvement)

**Current:** Some methods have `#[inline]`

**Proposed:**
```rust
#[inline(always)]  // Force inlining
fn descend(&self, label: u8) -> Option<Self>
```

**Benefit:** Eliminates function call overhead in hot loops.

### 5. SmallVec for Short Paths (3-5% improvement)

**Current:**
```rust
path: Arc<[u8]>  // Heap allocation even for short paths
```

**Proposed:**
```rust
path: SmallVec<[u8; 16]>  // Stack allocation for paths ≤ 16 bytes
```

**Benefit:** Most dictionary terms are `<16` bytes → no heap allocation.

**Estimated combined impact:** Reduce overhead from 88-97% to 20-30% (1.2-1.3× vs node-based).

---

## Recommendations

### Decision Matrix

| Use Case | Recommended Approach | Why |
|----------|---------------------|-----|
| **Code completion** | ✅ Zipper-based | Requires hierarchical contexts, drafts, thread safety |
| **Spell checker** | Node-based | Single dictionary, batch queries, no context needed |
| **Search indexing** | Node-based | High throughput critical, no hierarchical features |
| **LSP server** | ✅ Zipper-based | Multi-file contexts, concurrent requests, draft buffers |
| **CLI tool** | Node-based | Simple queries, no thread safety needed |
| **Multi-dictionary merge** | ✅ Zipper-based | Compositional architecture shines here |

### Hybrid Approach

For applications needing both, use the right tool for each task:

```rust
// Batch indexing: use node-based for speed
fn index_corpus(dict: &PathMapDictionary) {
    let transducer = Transducer::new(dict, Algorithm::Standard);
    for term in corpus {
        let matches = transducer.query(term, 2);
        index.add(matches);
    }
}

// Interactive completion: use zipper-based for features
fn provide_completions(engine: &ContextualCompletionEngine, ctx: ContextId) {
    let results = engine.complete(ctx, user_input, 2);
    display_to_user(results);
}
```

---

## Benchmark Methodology

### Test Environment
- **Hardware:** x86_64 with AVX2 support
- **Rust version:** 1.70+
- **Compiler flags:** `RUSTFLAGS="-C target-cpu=native"`
- **Dictionary:** 115 common English words
- **Queries:** "time", "test", "work", "people", "algorithm"
- **Iterations:** 100 samples per benchmark

### Benchmark Code
Located in: `benches/zipper_vs_node_benchmark.rs`

Run with:
```bash
cargo bench --bench zipper_vs_node_benchmark --features pathmap-backend
```

### Measurements
- **Single query:** `query("time", distance).collect()`
- **Batch query:** 5 queries in sequence
- **Iterator creation:** Just initialization, no traversal
- **All measurements:** Median of 100 samples

---

## Related Documentation

- **Optimization Report:** `docs/design/contextual-completion-progress.md` (Phase 6 complete)
- **Zipper Optimizations:** `/tmp/final_optimization_report.md` (3-9% improvements)
- **Contextual Engine Design:** `docs/design/contextual-completion-progress.md`
- **Performance Baselines:** `/tmp/zipper_optimization_baseline.md`

---

## Conclusion

**Zipper-based queries are 1.66-1.97× slower than node-based**, but this is an **intentional architectural trade-off**:

| Factor | Node-Based | Zipper-Based | Winner |
|--------|-----------|--------------|--------|
| **Raw Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Node |
| **Composability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Zipper |
| **Thread Safety** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Zipper |
| **Hierarchical Context** | ⭐ | ⭐⭐⭐⭐⭐ | Zipper |
| **Maintainability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Zipper |
| **Use Case Fit** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Zipper |

**For contextual code completion** (the primary use case), **zipper-based is the correct architectural choice** despite the performance overhead.

The absolute performance is still excellent (<1ms queries), and the architectural benefits (composability, thread safety, hierarchical contexts) are essential for the intended use case.

Future optimizations can reduce the overhead to ~1.2-1.3× with arena allocation, lazy initialization, and lock-free data structures.

**Status:** ✅ Zipper-based approach validated for contextual completion use case.
