# DAWG Optimization Opportunities

## Executive Summary

Analysis of the DAWG implementations reveals **significant optimization opportunities** with potential for **10-30% performance improvements** through targeted optimizations. The highest-impact improvements focus on edge lookup, lock contention, and allocation patterns.

## Three DAWG Implementations

1. **DawgDictionary** - Static, immutable DAWG (fastest, most memory-efficient)
2. **DynamicDawg** - Mutable DAWG with online operations (moderate overhead)
3. **PathMapDictionary** - Persistent trie via PathMap library (highest overhead)

## High-Impact Optimizations (10-30% improvement potential)

### 1. Binary Insertion for Edges (DynamicDawg)

**Current Issue:**
```rust
// Line 148, 162 in dynamic_dawg.rs
inner.nodes[node_idx].edges.sort_by_key(|(b, _)| *b);
```

**Problem:** Sorts entire edge list on every insertion - O(n log n)

**Solution:** Binary search insertion to maintain sorted order
```rust
fn insert_edge_sorted(edges: &mut Vec<(u8, usize)>, label: u8, target: usize) {
    let pos = edges.binary_search_by_key(&label, |(l, _)| *l)
        .unwrap_or_else(|e| e);
    edges.insert(pos, (label, target));
}
```

**Expected Impact:** 5-15% faster insertion

---

### 2. Binary Search Edge Lookup (All Implementations)

**Current Issue:**
```rust
// Line 281 in dawg.rs
fn transition(&self, label: u8) -> Option<Self> {
    self.nodes[self.node_idx]
        .edges
        .iter()
        .find(|(l, _)| *l == label)  // O(n) linear search
}
```

**Problem:** Linear search through edges on every transition

**Solution:** Use binary_search since edges are sorted
```rust
fn transition(&self, label: u8) -> Option<Self> {
    self.nodes[self.node_idx]
        .edges
        .binary_search_by_key(&label, |(l, _)| *l)
        .ok()
        .map(|idx| Self {
            nodes: Arc::clone(&self.nodes),
            node_idx: self.nodes[self.node_idx].edges[idx].1,
        })
}
```

**Alternative:** 256-entry lookup table for ASCII alphabets
```rust
struct DawgNode {
    edges: [Option<usize>; 256],  // Direct lookup
    is_final: bool,
}
```

**Expected Impact:** 3-8% faster queries (hot path optimization)

---

### 3. Reduce RwLock Contention (DynamicDawg)

**Current Issue:**
```rust
// Lines 628-663 in dynamic_dawg.rs
fn is_final(&self) -> bool {
    let inner = self.dawg.read().unwrap();  // Lock acquired
    inner.nodes[self.node_idx].is_final
}

fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
    let inner = self.dawg.read().unwrap();  // Lock acquired again
    // ... iteration
}
```

**Problem:** Every node operation acquires lock, causing contention during queries

**Solution 1:** Eager caching in DynamicDawgNode
```rust
pub struct DynamicDawgNode {
    dawg: Arc<RwLock<DynamicDawgInner>>,
    node_idx: usize,
    cached_is_final: bool,           // Cached at construction
    cached_edges: Vec<(u8, usize)>,  // Cached at construction
}
```

**Solution 2:** Atomic operations for read-only fields
```rust
use std::sync::atomic::{AtomicBool, Ordering};

struct DawgNode {
    edges: Vec<(u8, usize)>,
    is_final: AtomicBool,  // Lock-free reads
    ref_count: AtomicUsize,
}
```

**Expected Impact:** 5-15% faster queries with DynamicDawg

---

### 4. Implement Suffix Sharing in DynamicDawg

**Current Issue:**
```rust
// Line 380 in dynamic_dawg.rs
fn find_or_create_suffix(&mut self, _suffix: &[u8], _is_final: bool, _last: bool) -> Option<usize> {
    None  // NOT IMPLEMENTED!
}
```

**Problem:** Insertions create duplicate nodes for identical suffixes

**Solution:** Implement suffix sharing using HashMap
```rust
fn find_or_create_suffix(&mut self, suffix: &[u8], is_final: bool) -> Option<usize> {
    if suffix.is_empty() {
        return None;
    }

    // Compute suffix signature
    let sig = self.compute_suffix_signature(suffix, is_final);

    // Check if already exists
    if let Some(&node_idx) = self.suffix_map.get(&sig) {
        return Some(node_idx);
    }

    // Create new suffix chain
    let node_idx = self.create_suffix_chain(suffix, is_final);
    self.suffix_map.insert(sig, node_idx);
    Some(node_idx)
}
```

**Expected Impact:**
- 20-40% smaller DAWG for natural language
- 5-10% faster insertion (reduced node creation)

---

## Medium-Impact Optimizations (5-10% improvement)

### 5. Remove Unused suffix_map Field

**Issue:** Lines 39-40 in dynamic_dawg.rs
```rust
#[allow(dead_code)]
suffix_map: HashMap<NodeSignature, usize>,
```

**Problem:** Field exists but never populated - wasted memory

**Solution:** Remove or implement properly (see #4)

**Impact:** 2-5% memory savings

---

### 6. Optimize PathMapDictionary Zipper Usage

**Issue:** Lines 195-210 in pathmap.rs
```rust
fn with_zipper<F, R>(&self, f: F) -> R
where F: FnOnce(ReadZipperUntracked<'static, ()>) -> R {
    let map = self.map.read().unwrap();
    let zipper = map.read_zipper_at_path(&**self.path);  // Recreation overhead
    f(zipper)
}
```

**Problem:** Zipper recreated on every edges() call

**Solution:** Cache child mask or use PathMap's batch operations

**Impact:** 3-7% faster queries with PathMapDictionary

---

### 7. Resolve minimize() vs compact() Discrepancy

**Issue:** Line 824 in dynamic_dawg.rs
```rust
// Investigation item: minimize() and compact() produce different node counts
```

**Problem:**
- Two minimization strategies with different results
- Unclear which is correct
- Risk of false positives

**Solution:**
1. Add property-based testing
2. Determine root cause of difference
3. Choose optimal strategy

**Impact:** Correctness improvement, may enable better compression

---

## Low-Impact Optimizations (<5% improvement)

### 8. Inline Hot Path Methods
- `DictionaryNode::is_final()`, `transition()`, `edges()`
- Add `#[inline]` attributes

### 9. SmallVec Capacity Tuning
- PathMapDictionary uses `SmallVec<[u8; 8]>`
- Profile to determine optimal size

### 10. StatePool Size Tuning
- Current max: 32 states
- Could be adaptive based on workload

---

## Performance Profiling Data

**From existing benchmarks:**

**Hot Path:** `query.rs:90` - `queue_children()`
- Called thousands of times per query
- Each call: `intersection.node.edges()` iteration

**State Cloning:** 21.73% of runtime
- Vec allocation: 6.00%
- Position copies: 7.44%
- StatePool helps but doesn't eliminate

**Lock Overhead:** Estimated 5-10% for DynamicDawg queries
- RwLock acquisition on every node operation
- Critical section for edge iteration

---

## Memory Layout Analysis

### Current Memory per Node

**DawgDictionary (Static):**
- DawgNode: 24 bytes (Vec header) + edges
- Per edge: 9 bytes (u8 + usize)
- Total: ~33 bytes/node + edge storage

**DynamicDawg (Mutable):**
- DawgNode: 32 bytes (Vec + bool + ref_count)
- Per edge: 9 bytes
- RwLock overhead: ~64 bytes per dictionary
- Total: ~41 bytes/node + overhead

**PathMapDictionary:**
- PathMapNode wrapper: 32 bytes (Arc + Arc<Vec<u8>>)
- PathMap internal: Variable
- Path vector: Shared via Arc (efficient)

### Optimization: Compressed Edge Representation

**Current:**
```rust
edges: Vec<(u8, usize)>  // 9 bytes per edge on 64-bit
```

**Optimized (for small alphabets):**
```rust
edges: SmallVec<[(u8, u32); 4]>  // 5 bytes per edge, stack-allocated for ≤4 edges
```

**Impact:**
- 44% less memory per edge
- Stack allocation for nodes with ≤4 children (most nodes)
- Requires limiting to 2^32 nodes (acceptable)

---

## Recommended Optimization Priority

### Phase 1: High-Impact, Low-Risk (Recommended)
1. ✅ Binary search edge lookup (#2) - 3-8% improvement
2. ✅ Binary insertion for edges (#1) - 5-15% improvement
3. ✅ Remove unused suffix_map (#5) - Memory cleanup

**Estimated combined impact:** 10-20% faster queries, 5-15% faster insertions

### Phase 2: Medium-Impact, Moderate-Risk
4. ✅ Implement suffix sharing (#4) - 20-40% smaller, 5-10% faster inserts
5. ✅ Reduce lock contention (#3) - 5-15% faster DynamicDawg queries
6. ✅ Investigate minimize() discrepancy (#7) - Correctness

**Estimated combined impact:** 30-50% smaller DAWGs, 10-20% faster overall

### Phase 3: Advanced Optimizations
7. Compressed edge representation - 40% memory savings
8. Lock-free data structures - 10-20% faster concurrent access
9. Custom allocator for nodes - 5-10% allocation speedup

---

## Testing Strategy

**For each optimization:**
1. **Correctness Tests:**
   - Verify no false positives/negatives
   - Test edge cases (empty, single term, duplicates)
   - Run `test_minimize_no_false_positives()`

2. **Performance Benchmarks:**
   - Query performance (primary metric)
   - Insertion performance
   - Memory usage
   - Compare: Before vs After

3. **Regression Tests:**
   - All existing tests must pass
   - Serialization roundtrip tests
   - Cross-format compatibility

---

## Code Locations

**Key files for optimization:**
- `src/dictionary/dawg.rs` - Static DAWG (edge lookup)
- `src/dictionary/dynamic_dawg.rs` - Mutable DAWG (insertion, locks)
- `src/dictionary/pathmap.rs` - PathMap wrapper (zipper overhead)
- `src/transducer/query.rs` - Hot path (traversal pattern)
- `src/transducer/pool.rs` - State pooling (allocation reuse)

**Benchmark files:**
- `benches/benchmarks.rs` - Existing query benchmarks
- `benches/micro_benchmarks.rs` - (if exists) Micro-benchmarks

---

## Conclusion

The DAWG implementations have **significant optimization potential**:

✅ **Quick wins:** Binary search edge lookup, binary insertion (Phase 1)
✅ **Major improvements:** Suffix sharing, lock-free reads (Phase 2)
✅ **Advanced:** Compressed edges, custom allocators (Phase 3)

**Recommended next step:** Implement Phase 1 optimizations for immediate 10-20% performance gain with minimal risk.
