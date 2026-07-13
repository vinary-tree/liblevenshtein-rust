# MSM Time Series Benchmark Analysis

## Overview

This document summarizes benchmark results for the MSM (Move-Split-Merge) metric implementation,
including the direct DP algorithm, lower-bound pruning, and hybrid trie-based search.

**Test Environment:**
- Series length: 50 elements (unless otherwise noted)
- Values: Random floats in [0, 100]
- MSM cost parameter: c = 1.0
- Threshold: 50.0 (where applicable)

## 1. MSM DP Algorithm Performance

| Series Length | Standard DP | Optimized DP | Speedup |
|--------------|-------------|--------------|---------|
| 10           | ~565 ns     | ~565 ns      | 1.0×    |
| 50           | ~19 µs      | ~14 µs       | 1.4×    |
| 100          | ~77 µs      | ~56 µs       | 1.4×    |
| 500          | ~2.94 ms    | ~2.12 ms     | 1.4×    |

**Key Findings:**
- The optimized DP (using O(n) space instead of O(mn)) provides ~40% speedup
- Complexity: O(mn) time, O(min(m,n)) space for optimized variant
- Throughput: ~168-238 Kelem/s for 500-element series

## 2. Lower Bound Performance

| Series Length | Length LB | Euclidean LB | L1 LB | Combined LB |
|--------------|-----------|--------------|-------|-------------|
| 10           | 2.0 ns    | 11 ns        | 10 ns | 12 ns       |
| 50           | 2.0 ns    | 40 ns        | 35 ns | 42 ns       |
| 100          | 2.0 ns    | 80 ns        | 75 ns | 107 ns      |
| 500          | 2.0 ns    | 424 ns       | 400 ns| 446 ns      |

**Key Findings:**
- Length LB is O(1) at ~2 ns regardless of series length
- Euclidean/L1 LBs are O(n) with excellent cache locality
- Combined LB computes max of all bounds for tightest pruning
- **Speedup vs Full MSM**: 4,700-5,000× faster for 500-element series

## 3. Lower Bound Effectiveness in Hybrid Search

### LB Type Comparison (500 series database, 50-element series)

| LB Type     | Search Time | Relative |
|-------------|-------------|----------|
| Length Only | 143.66 ms   | 1.00×    |
| Euclidean   | 136.69 ms   | 0.95×    |
| L1          | 135.23 ms   | 0.94×    |
| Combined    | 136.12 ms   | 0.95×    |

**Key Findings:**
- Euclidean and L1 bounds provide tighter pruning than length-only
- Combined LB adds slight overhead but provides most aggressive pruning
- For random data with threshold=50.0, ~5-6% improvement from tighter LBs

### With/Without LB Pruning

| Database Size | With LB    | Without LB | Speedup |
|---------------|------------|------------|---------|
| 100 series    | 25.7 ms    | 31.2 ms    | 1.21×   |
| 500 series    | 145.9 ms   | 154.1 ms   | 1.06×   |
| 1000 series   | 287.9 ms   | 312.6 ms   | 1.09×   |

**Key Findings:**
- LB pruning provides 6-21% speedup depending on database size
- Greater relative benefit for smaller databases (lower trie overhead ratio)
- LB pruning becomes more valuable as MSM threshold tightens

## 4. Brute Force vs Indexed Search

| Database Size | Brute Force + LB | Hybrid Index | Ratio |
|---------------|------------------|--------------|-------|
| 100 series    | 6.7 µs           | 26.6 ms      | 1:4000 |
| 500 series    | 35.6 µs          | 139.1 ms     | 1:3900 |

**Analysis:**

This surprising result requires careful interpretation:

1. **Brute Force with LB** only computes lower bounds in the fast path. When the LB exceeds the threshold (most candidates for tight thresholds), no full MSM is computed.

2. **Hybrid Index** incurs overhead from:
   - Trie traversal using Levenshtein automaton
   - Quantization encoding/decoding
   - More candidates passing trie filter (approximate)
   - Full MSM verification for each candidate

3. **When Brute Force Wins**:
   - Small databases (< 1,000 series)
   - Tight thresholds (LB prunes most candidates)
   - No prefix sharing in data (random series)

4. **When Hybrid Index Wins**:
   - Large databases with shared prefixes
   - Loose thresholds where many candidates need verification
   - Repeated queries amortize index construction cost
   - Approximate search sufficient (skip MSM verification)

**Recommendation:** Use brute-force with LB pruning for databases under 10,000 series. Consider hybrid indexing for larger databases or when prefix sharing is expected.

## 5. Quantization Level Impact

| Bins (K) | Search Time | Relative |
|----------|-------------|----------|
| 16       | 10.8 ms     | 1.0×     |
| 64       | 84.5 ms     | 7.8×     |
| 256      | 85.1 ms     | 7.9×     |

**Key Findings:**
- Coarser quantization (16 bins) is ~8× faster due to smaller alphabet
- 64 and 256 bins have similar performance (automaton overhead dominates)
- Trade-off: Fewer bins = faster search but more false positives

**Recommendation:** Start with K=64 bins for balanced precision/performance. Use K=16 for approximate search where speed is critical.

## 6. Scaling Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| MSM DP    | O(mn)      | m, n = series lengths |
| Length LB | O(1)       | Just length comparison |
| Euclidean LB | O(min(m,n)) | Single pass, SIMD-friendly |
| Trie Search | O(m × k^d) | m = query length, k = alphabet, d = max distance |
| Hybrid Search | O(m × k^d + c × mn) | c = candidate count after pruning |

### Space Complexity

| Structure | Space | Notes |
|-----------|-------|-------|
| MSM DP    | O(min(m,n)) | Optimized variant |
| Trie Index | O(N × L) | N = series count, L = avg length, with prefix sharing |
| Hybrid Index | O(N × L + N × L') | Additional storage for original floats |

## 7. Recommendations

### Small Databases (< 1,000 series)
```rust
// Use brute force with lower bounds
use liblevenshtein::time_series::{search_with_lb, MsmConfig};

let results = search_with_lb(&query, &database, threshold, &msm_config);
```

### Medium Databases (1,000 - 100,000 series)
```rust
// Use hybrid search with euclidean lower bounds
use liblevenshtein::time_series::{HybridSearchIndex, QuantizationConfig, MsmConfig, LowerBoundType};

let mut index = HybridSearchIndex::new(quant_config, msm_config);
index.set_lower_bound_type(LowerBoundType::EuclideanOnly);
// ... insert series ...
let results = index.search_exact(&query, threshold);
```

### Large Databases (> 100,000 series)
```rust
// Use approximate search with verification on top-k
let candidates = index.search_approximate(&query, max_distance);
// Verify top-k candidates with exact MSM
```

## 8. Future Optimizations

1. **SIMD-accelerated Euclidean LB** - Use AVX2/SSE4 for parallel distance computation
2. **Parallel LB pruning** - Use Rayon for multi-threaded candidate filtering
3. **Adaptive LB selection** - Choose LB type based on series characteristics
4. **Tighter lower bounds** - Implement LB_Keogh or envelope-based bounds
5. **Batch queries** - Amortize index traversal across multiple queries
