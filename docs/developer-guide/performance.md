# Performance Optimizations

**Version:** 0.9.1
**Last Updated:** 2026-06-19

This document summarizes the key performance optimizations implemented in liblevenshtein-rust.

---

## Overview

The library has undergone extensive performance optimization, achieving:
- **40-60% overall performance improvements**
- **3.3x speedup for DAWG operations**
- **15-50% faster PathMap edge iteration**
- **5-18% improvements for filtering/prefix operations**
- **85% file size reduction with compression**

---

## Key Optimizations

### 1. Arc Path Sharing

**Problem:** Expensive cloning of path vectors during traversal
**Solution:** Use `Arc<Vec<u8>>` for shared ownership of paths

**Impact:**
- Eliminated deep copies of path data
- Cheap reference counting instead of allocation
- Significant reduction in memory allocations

**Implementation:** `src/transducer/query.rs`, `src/dictionary/dawg_query.rs`

```rust
// Before: Deep copy on every transition
let mut new_path = self.path.clone();
new_path.push(label);

// After: Shared ownership with Arc
let new_path = {
    let mut path = Vec::with_capacity(self.path.len() + 1);
    path.extend_from_slice(&self.path);
    path.push(label);
    Arc::new(path)
};
```

### 2. StatePool - Object Pool Pattern

**Problem:** Repeated allocation/deallocation of State objects in hot paths
**Solution:** Object pool that reuses State instances

**Impact:**
- **Exceptional performance gains**
- Eliminated allocation overhead in query loops
- Reduced garbage collection pressure

**Implementation:** `src/transducer/state_pool.rs`

```rust
pub struct StatePool {
    pool: RefCell<Vec<State>>,
}

impl StatePool {
    pub fn acquire(&self, position: StatePosition) -> PooledState {
        let mut pool = self.pool.borrow_mut();
        let mut state = pool.pop().unwrap_or_else(State::default);
        state.reset(position);
        PooledState { state: Some(state), pool: self }
    }
}
```

### 3. SmallVec Integration

**Problem:** Frequent small vector allocations
**Solution:** Stack-allocated vectors for small collections

**Impact:**
- Reduced heap allocation pressure
- Faster operations for small collections (< 8 elements)
- Zero-cost abstraction for common cases

**Implementation:** Used throughout for edge lists, state collections

```rust
use smallvec::SmallVec;

// Stack-allocated for ≤8 elements, heap for larger
type EdgeList = SmallVec<[(u8, Node); 8]>;
```

### 4. Lazy Edge Iteration

**Problem:** Eager collection of all edges into vectors
**Solution:** Zero-copy iterator over PathMap edges

**Impact:**
- **15-50% faster** edge iteration
- No intermediate allocations
- Pay-as-you-go evaluation

**Implementation:** `src/dictionary/pathmap.rs`

```rust
pub fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
    // Get child mask (cheap - just bit tests)
    let edge_bytes: SmallVec<[u8; 8]> = self.with_zipper(|zipper| {
        let mask = zipper.child_mask();
        (0..=255u8).filter(|byte| mask.test_bit(*byte)).collect()
    });

    // Lazy iterator - creates nodes on-demand
    Box::new(edge_bytes.into_iter().filter_map(move |byte| {
        // ... create node only when iterated
    }))
}
```

### 5. Aggressive Inlining

**Problem:** Function call overhead in hot paths
**Solution:** Strategic use of `#[inline]` and `#[inline(always)]`

**Impact:**
- 5-18% improvements in filtering/prefix scenarios
- Better compiler optimization opportunities
- Reduced function call overhead

**Functions Inlined:**
- `is_final()`, `transition()` - called millions of times
- Epsilon closure handling
- Filter predicate evaluation

### 6. Index-Based DAWG Queries

**Problem:** Cloning DAWG nodes during traversal
**Solution:** Use node indices instead of cloning

**Impact:**
- Memory-efficient traversal
- Eliminated node cloning overhead
- Faster DAWG queries

**Implementation:** `src/dictionary/dawg_query.rs`

```rust
pub struct DawgNode {
    index: usize,  // Reference by index, not clone
    // ...
}
```

---

## Compression & Serialization

### Gzip Compression

**Feature:** Optional gzip compression for serialized dictionaries

**Results:**
- **85% file size reduction**
- ~1.6x deserialization overhead
- Transparent compression/decompression

**Usage:**
```rust
use liblevenshtein::serialization::{BincodeSerializer, GzipSerializer};

// Save compressed
GzipSerializer::<BincodeSerializer>::serialize(&dict, file)?;

// Load compressed
let dict = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
```

**Formats Supported:**
- `bincode-gz` - Binary format with gzip
- `json-gz` - JSON with gzip
- `protobuf-gz` - Protocol Buffers with gzip

---

## Benchmarking Methodology

### Tools Used

- **Criterion.rs** - Statistical benchmarking
- **Flamegraph** - CPU profiling and hotspot identification
- **cargo-llvm-cov** - Code coverage analysis
- **PGO** - Profile-Guided Optimization (tested, not currently used)

### Benchmark Types

1. **Micro-benchmarks** - Individual operation performance
2. **Real-world benchmarks** - System dictionary (/usr/share/dict/words)
3. **Profiling benchmarks** - Flamegraph generation for analysis
4. **Filtering benchmarks** - Code completion scenarios

### Running Benchmarks

```bash
# All benchmarks
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo bench

# Specific benchmark suite
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo bench --bench dawg_benchmarks

# With profiling
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo flamegraph --bench profiling_benchmark
```

---

## Performance Tips

### For Library Users

1. **Use ordered queries sparingly** - `query_ordered()` has overhead for sorting
2. **Enable features selectively** - Only enable features you need
3. **Reuse transducers** - Create once, query many times
4. **Consider DAWG for read-only** - More space-efficient than PathMap
5. **Use compression for storage** - 85% smaller files with minimal overhead

### For Contributors

1. **Profile before optimizing** - Use flamegraph to find hotspots
2. **Benchmark changes** - Verify improvements with Criterion
3. **Consider allocation patterns** - Use SmallVec for small collections
4. **Inline hot paths** - But measure impact, don't over-inline
5. **Test with real data** - Synthetic benchmarks can mislead

---

## Archived Performance Documentation

Detailed historical performance analysis is available in the archived performance documentation:

### Optimization Phases
- **OPTIMIZATION_SUMMARY.md** - Complete optimization overview
- **optimization/** - Detailed phase-by-phase reports
  - Phase 4: SmallVec investigation
  - Phase 5: StatePool implementation (exceptional results)
  - Phase 6: Arc path sharing (highly successful)

### Specific Optimizations
- **ARC_OPTIMIZATION_RESULTS.md**
- **DAWG_OPTIMIZATION_RESULTS.md**
- **PATHNODE_OPTIMIZATION_RESULTS.md**
- **SERIALIZATION_OPTIMIZATION_RESULTS.md**

### Comparisons & Validation
- **DAWG_COMPARISON.md** - DAWG vs PathMap analysis
- **JAVA_COMPARISON.md** - vs original Java implementation
- **REAL_WORLD_VALIDATION.md** - System dictionary tests

### Code Completion
- **CODE_COMPLETION_PERFORMANCE.md** - Filtering optimization strategies
- **CONTEXTUAL_FILTERING_OPTIMIZATION.md** - Bitmap masking

---

## Future Optimization Opportunities

See `FUTURE_ENHANCEMENTS.md` for planned improvements.

### Potential Areas

1. **SIMD Operations** - Vectorize distance calculations
2. **Parallel Queries** - Rayon-based parallelization for large result sets
3. **Custom Allocators** - Specialized allocators for specific workloads
4. **Better Heuristics** - Adaptive thresholds based on dictionary characteristics
5. **GPU Acceleration** - For very large dictionaries (research)

---

## References

- **Profiling Guide:** See [`building.md`](building.md#profiling) for profiling instructions
- **Benchmarks:** Run `cargo bench` to see current performance
- **Examples:** See [`examples/`](../examples/README.md) for real-world usage patterns
- **Contributing:** See [`contributing.md`](contributing.md) for optimization guidelines
