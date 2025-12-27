# WallBreaker Optimization Scientific Ledger

**Created**: 2025-12-27
**Purpose**: Track empirical results for WallBreaker optimizations with statistical rigor

## Methodology

### Statistical Requirements
- **Sample size**: Minimum 30-50 benchmark iterations per configuration
- **Significance level**: α = 0.05 (95% confidence)
- **Effect size**: Report Cohen's d or percentage improvement
- **Tool**: Criterion.rs built-in statistical analysis (t-test, confidence intervals)

### Decision Criteria
- **ACCEPT**: p < 0.05, no regressions, all tests pass
- **REJECT**: p ≥ 0.05 or introduces regressions

---

## Experiment 1: Baseline Benchmarks

**Date**: 2025-12-27
**Branch**: `feat/wallbreaker-benchmarks`
**Purpose**: Establish performance baseline for WallBreaker algorithm before optimizations
**Status**: ✅ COMPLETE

### Test Configuration
- **Dictionary source**: `/usr/share/dict/words` (88,996 words loaded)
- **Sample sizes**: 50 iterations per configuration (30 for construction)
- **Benchmark framework**: Criterion.rs 0.5

### WallBreaker Query Performance

| Dict Size | Distance | Query Len | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|----------|-----------|-----------|--------------|--------------|------------|
| 1,000     | 2        | 10        | 10.107 ms | 10.032 ms    | 10.187 ms    | 989 elem/s |
| 1,000     | 2        | 20        | 6.567 ms  | 6.482 ms     | 6.660 ms     | 1,523 elem/s |
| 1,000     | 4        | 10        | **6.799 s** | 6.722 s    | 6.875 s      | 1.47 elem/s |
| 1,000     | 4        | 20        | 52.26 ms  | 51.77 ms     | 52.74 ms     | 191 elem/s |
| 10,000    | 2        | 20        | 62.99 ms  | 62.69 ms     | 63.30 ms     | 159 elem/s |
| 10,000    | 4        | 20        | 159.67 ms | 158.20 ms    | 161.29 ms    | 62.6 elem/s |
| 10,000    | 4        | 50        | 99.68 ms  | 99.39 ms     | 99.98 ms     | 100 elem/s |
| 10,000    | 8        | 50        | 253.13 ms | 250.82 ms    | 255.56 ms    | 39.5 elem/s |
| 88,996    | 2        | 20        | 817.89 ms | 814.13 ms    | 821.84 ms    | 12.2 elem/s |
| 88,996    | 4        | 50        | 1.301 s   | 1.297 s      | 1.306 s      | 7.68 elem/s |
| 88,996    | 8        | 50        | 2.517 s   | 2.484 s      | 2.546 s      | 3.97 elem/s |
| 88,996    | 8        | 100       | 2.292 s   | 2.241 s      | 2.341 s      | 4.36 elem/s |
| 88,996    | 16       | 100       | 6.438 s   | 6.390 s      | 6.485 s      | 1.55 elem/s |

### Traditional Transducer Performance (Comparison)

| Dict Size | Distance | Query Len | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|----------|-----------|-----------|--------------|--------------|------------|
| 1,000     | 2        | 10        | 1.015 ms  | 1.010 ms     | 1.020 ms     | 9.85 Kelem/s |
| 1,000     | 2        | 20        | 1.022 ms  | 1.014 ms     | 1.031 ms     | 9.78 Kelem/s |
| 1,000     | 4        | 10        | 5.119 ms  | 5.090 ms     | 5.146 ms     | 1.95 Kelem/s |
| 1,000     | 4        | 20        | 5.034 ms  | 5.005 ms     | 5.067 ms     | 1.99 Kelem/s |
| 10,000    | 2        | 20        | 5.509 ms  | 5.484 ms     | 5.534 ms     | 1.82 Kelem/s |
| 10,000    | 4        | 20        | 43.90 ms  | 43.69 ms     | 44.12 ms     | 228 elem/s |
| 10,000    | 4        | 50        | 44.57 ms  | 44.31 ms     | 44.84 ms     | 224 elem/s |
| 10,000    | 8        | 50        | 191.74 ms | 190.74 ms    | 192.77 ms    | 52.2 elem/s |
| 88,996    | 2        | 20        | 22.67 ms  | 22.56 ms     | 22.78 ms     | 441 elem/s |
| 88,996    | 4        | 50        | 305.28 ms | 302.69 ms    | 307.79 ms    | 32.8 elem/s |

### Performance Comparison: Traditional vs WallBreaker

| Configuration | WallBreaker | Traditional | **Speedup (Trad/WB)** | Notes |
|---------------|-------------|-------------|----------------------|-------|
| d1000_k2_q10  | 10.107 ms   | 1.015 ms    | **10.0× faster**     | |
| d1000_k2_q20  | 6.567 ms    | 1.022 ms    | **6.4× faster**      | |
| d1000_k4_q10  | 6,799 ms    | 5.119 ms    | **1,328× faster**    | ⚠️ Extreme |
| d1000_k4_q20  | 52.26 ms    | 5.034 ms    | **10.4× faster**     | |
| d10000_k2_q20 | 62.99 ms    | 5.509 ms    | **11.4× faster**     | |
| d10000_k4_q20 | 159.67 ms   | 43.90 ms    | **3.6× faster**      | |
| d10000_k4_q50 | 99.68 ms    | 44.57 ms    | **2.2× faster**      | |
| d10000_k8_q50 | 253.13 ms   | 191.74 ms   | **1.32× faster**     | |
| d88996_k2_q20 | 817.89 ms   | 22.67 ms    | **36× faster**       | |
| d88996_k4_q50 | 1,301 ms    | 305.28 ms   | **4.3× faster**      | |

### Direct Comparison Group (wallbreaker_vs_traditional)

| Configuration | WallBreaker | Traditional | **Speedup (Trad/WB)** |
|---------------|-------------|-------------|----------------------|
| medium_d4_q50 (10K) | 112.09 ms | 52.72 ms | **2.1× faster** |
| medium_d8_q50 (10K) | 263.92 ms | 193.76 ms | **1.36× faster** |
| large_d4_q50 (50K)  | 691.77 ms | 176.11 ms | **3.9× faster** |
| large_d8_q100 (50K) | 1,377.8 ms | 1,040.2 ms | **1.32× faster** |

### SCDAWG Construction Time

| Dict Size | SCDAWG | DynamicDawg | Ratio |
|-----------|--------|-------------|-------|
| 1,000     | 702.91 µs | 314.66 µs | 2.2× slower |
| 10,000    | 6.957 ms  | 2.845 ms  | 2.4× slower |
| 88,996    | 127.75 ms | 27.62 ms  | **4.6× slower** |

### Substring Search Performance

| Pattern Length | Mean Time | 95% CI Lower | 95% CI Upper |
|----------------|-----------|--------------|--------------|
| 5 chars        | 225.29 ms | 222.96 ms    | 227.94 ms    |
| 10 chars       | 218.54 ms | 215.82 ms    | 221.46 ms    |
| 15 chars       | 222.42 ms | 215.66 ms    | 229.64 ms    |
| 20 chars       | 288.39 ms | 280.79 ms    | 295.69 ms    |

*Note: These are for 20 pattern searches against a 50K dictionary*

### Pattern Splitting Performance

| Distance | Query Len | Mean Time |
|----------|-----------|-----------|
| 2        | 20        | 29.96 µs  |
| 2        | 50        | 42.40 µs  |
| 2        | 100       | 61.94 µs  |
| 4        | 20        | 34.19 µs  |
| 4        | 50        | 45.41 µs  |
| 4        | 100       | 65.29 µs  |
| 8        | 20        | 47.71 µs  |
| 8        | 50        | 72.50 µs  |
| 8        | 100       | 91.03 µs  |
| 16       | 20        | 80.85 µs  |
| 16       | 50        | 110.60 µs |
| 16       | 100       | 132.73 µs |

*Pattern splitting is fast (<135µs) - NOT a bottleneck*

---

## Critical Observations

### 1. ⚠️ WallBreaker is SLOWER than Traditional in ALL configurations

**Finding**: The traditional Levenshtein transducer outperforms WallBreaker in every tested configuration. This is the **opposite** of expected behavior based on the WallBreaker paper.

**Speedup ratios**:
- Best case for WallBreaker: 1.32× slower (d10000_k8_q50)
- Worst case for WallBreaker: **1,328× slower** (d1000_k4_q10)
- Typical case: 3-10× slower

### 2. Root Cause: Naive O(n*m) Substring Search

**Evidence**: Substring search takes 215-288ms for 20 patterns against 50K dictionary
- This is ~10-15ms per substring search
- For WallBreaker with k=4, we need 5 pattern pieces × substring searches
- This dominates the runtime

**Implication**: Phase 2 (suffix link optimization) is **CRITICAL** and may provide 10-100× improvement.

### 3. Short Queries are Pathological

**Finding**: d1000_k4_q10 (query length 10, distance 4) takes 6.8 SECONDS
- Query splits into 5 pieces of length 2 each
- 2-char patterns match almost everything in dictionary
- Creates massive false-positive explosion

**Implication**: WallBreaker requires query_length >> max_distance to be effective.

### 4. Pattern Splitting is NOT a Bottleneck

**Finding**: Pattern splitting takes 30-133µs (microseconds)
- This is <0.01% of total runtime
- Phase 3 (frequency-based splitting) may help with false positives, not raw speed

### 5. SCDAWG Construction is Expensive

**Finding**: SCDAWG construction is 2.2-4.6× slower than DynamicDawg
- 89K words: 128ms vs 28ms
- This is a one-time cost but worth noting

---

## Conclusion

The baseline benchmarks reveal that the current WallBreaker implementation is **not yet competitive** with the traditional transducer. The primary bottleneck is the naive O(n*m) substring search, which must be optimized using SCDAWG suffix links before WallBreaker can demonstrate its theoretical advantages.

**Priority Order** (revised based on data):
1. **Phase 2 (Suffix Links)**: CRITICAL - expected 10-100× improvement potential
2. **Phase 3 (Frequency Splitting)**: May reduce false positives
3. **Phase 4 (SIMD)**: Optimization on top of already-fixed algorithm

---

## Experiment 2: Suffix Link Substring Search Optimization

**Date**: 2025-12-27
**Branch**: `feat/wallbreaker-substring-opt`
**Baseline**: Experiment 1 results
**Status**: ❌ REJECTED (Architectural Incompatibility)

### Hypothesis
- **H₀**: Suffix link-based substring search provides no performance improvement over naive O(n*m) search
- **H₁**: Suffix link-based search reduces substring search time by >30% for patterns >10 chars

### Acceptance Criteria
- p < 0.05 improvement over baseline
- >30% reduction in substring search time for patterns >10 chars
- All existing tests pass

### Investigation Results

**Finding**: The optimization is **architecturally incompatible** with the current SCDAWG implementation.

The SCDAWG as implemented is a **DAWG (Directed Acyclic Word Graph)** for dictionary terms, NOT a **true suffix automaton**. The critical difference:

| Property | DAWG (Current) | Suffix Automaton (Required) |
|----------|----------------|----------------------------|
| Forward edges from root | Only dictionary term prefixes | All substrings of all terms |
| Pattern "thedr" in "cathedral" | No path from root starting with 't' | Path exists: root → t → h → e → d → r |
| Substring search complexity | O(total_chars × pattern_len) | O(\|pattern\| + occurrences) |

**Attempted Implementation**:
```rust
// Walk from root following forward edges to find pattern end node
let mut current = 0; // Start at root
for &byte in pattern.as_bytes() {
    match self.nodes[current].find_forward_edge(byte) {
        Some(next) => current = next,
        None => return Vec::new(), // Pattern not found
    }
}
```

**Why It Failed**:
- When searching for substring "thedr" in dictionary containing "cathedral"
- Root node only has edge 'c' (from "cathedral"), not 't'
- The pattern "thedr" is an internal substring, not accessible via forward edges from root

**Root Cause**:
The suffix links in the current implementation are simplified - they point to root when no matching edge exists. A true suffix automaton (Blumer et al.'s algorithm) maintains suffix links that connect all substring equivalence classes.

### Alternative Approaches Considered

1. **True Suffix Automaton**: Requires fundamental reconstruction of the SCDAWG
   - Would need to store all suffixes of all terms (significant memory overhead)
   - Construction algorithm would need complete rewrite

2. **Suffix Array/Tree**: Alternative data structure for substring search
   - Separate from DAWG, additional memory cost
   - Would require maintaining two structures

3. **Current Approach**: O(n*m) enumeration search
   - Simple, correct, no additional memory
   - Performance acceptable for most use cases (215-288ms for 20 patterns in 50K dict)

### Conclusion

**Decision**: ❌ REJECTED

**Rationale**: The hypothesis assumed the SCDAWG was a true suffix automaton with forward edges for all substrings. Investigation revealed it is a DAWG that only stores dictionary term prefixes. Implementing suffix link-based substring search would require converting to a true suffix automaton, which is a major architectural change beyond the scope of this optimization.

**Impact on WallBreaker**: The substring search bottleneck identified in Experiment 1 remains. Alternative optimization strategies should be explored:
1. Reduce the number of substring searches needed (Phase 3: smarter pattern splitting)
2. Parallel substring searches using SIMD (Phase 4)
3. Bloom filter pre-filtering to skip impossible patterns
4. Consider building a separate suffix array for substring search

---

## Experiment 3: Frequency-Based Pattern Splitting

**Date**: TBD
**Branch**: `feat/wallbreaker-freq-split`
**Baseline**: Last ACCEPTED optimization

### Hypothesis
- **H₀**: Frequency-based pattern splitting provides no performance improvement over uniform splitting
- **H₁**: Splitting at rare-character positions reduces query time by >10%

### Acceptance Criteria
- p < 0.05 improvement over baseline
- >10% reduction in query time
- All existing tests pass

### Results

| Configuration | Baseline (ns) | Optimized (ns) | Δ% | p-value | Cohen's d | Significant? |
|---------------|---------------|----------------|-----|---------|-----------|--------------||
| TBD           | TBD           | TBD            | TBD | TBD     | TBD       | TBD          |

### Conclusion
**Decision**: TBD (ACCEPTED/REJECTED)
**Rationale**: TBD

---

## Experiment 4: SIMD Acceleration for Extension Operations

**Date**: TBD
**Branch**: `feat/wallbreaker-simd`
**Baseline**: Last ACCEPTED optimization

### Hypothesis
- **H₀**: SIMD vectorization provides no performance improvement for extension operations
- **H₁**: SIMD batch processing provides >50% improvement in extension phase

### Acceptance Criteria
- p < 0.05 improvement over baseline
- >50% improvement in extension phase
- Identical results between scalar and SIMD implementations
- All existing tests pass

### Results

| Configuration | Baseline (ns) | SIMD (ns) | Δ% | p-value | Cohen's d | Significant? |
|---------------|---------------|-----------|-----|---------|-----------|--------------||
| TBD           | TBD           | TBD       | TBD | TBD     | TBD       | TBD          |

### Conclusion
**Decision**: TBD (ACCEPTED/REJECTED)
**Rationale**: TBD

---

## Summary of Decisions

| Experiment | Branch | Decision | Key Metric | Notes |
|------------|--------|----------|------------|-------|
| Baseline   | feat/wallbreaker-benchmarks | ✅ COMPLETE | WallBreaker 1.3-1328× slower than traditional | Substring search is critical bottleneck |
| Suffix Links | feat/wallbreaker-substring-opt | ❌ REJECTED | Architectural incompatibility | SCDAWG is DAWG, not suffix automaton |
| Freq Split | feat/wallbreaker-freq-split | TBD | TBD | Will base off benchmarks branch |
| SIMD | feat/wallbreaker-simd | TBD | TBD | |
