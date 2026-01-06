# Batch Query Processing Experiment Log

## Overview

**Goal**: Achieve MLP (Memory-Level Parallelism) by interleaving multiple independent fuzzy queries to overlap DRAM latencies.

**Hypothesis**: Interleaving N independent queries will achieve 2-4x throughput improvement for batch workloads.

**Dictionary**: `data/english_words.txt` (123,985 English words)

---

## Experiment 1: Basic Batch Query Processing

### Date
2026-01-05

### Hypothesis
Round-robin processing of multiple queries overlaps DRAM wait times, improving throughput.

### Design
- `BatchQueryProcessor` struct with interleaved BFS loop
- Process ONE intersection per query per round
- Share StatePool across all queries

### Implementation
- New file: `src/transducer/batch.rs`
- Export via `src/transducer/mod.rs`

### Baseline Measurement
Dictionary: `data/english_words.txt` (123,985 words)
Algorithm: Standard Levenshtein
Max distance: 2
Queries: Evenly sampled from dictionary

### Results

| Queries | Sequential | Batch | Throughput (seq) | Throughput (batch) | **Regression** |
|---------|-----------|-------|------------------|-------------------|----------------|
| 10 | 17.2ms | 18.1ms | 581 elem/s | 553 elem/s | **+5%** |
| 50 | 94.3ms | 100.8ms | 530 elem/s | 496 elem/s | **+7%** |
| 100 | 186.6ms | 229.9ms | 536 elem/s | 435 elem/s | **+23%** |
| 500 | 922ms | 1.38s | 542 elem/s | 362 elem/s | **+50%** |
| 1000 | 1.82s | ~2.9s | 550 elem/s | ~350 elem/s | **+60%** |

### Analysis

**The hypothesis is REJECTED.** Batch processing is consistently **slower** than sequential processing, with the overhead increasing as batch size grows.

**Root cause analysis:**

1. **Dictionary fits in L3 cache**: The 124K word dictionary likely fits entirely in L3 cache (~32-64MB on modern CPUs). This means DRAM latency is NOT the bottleneck - cache hits dominate.

2. **Overhead of interleaving**: The round-robin scheduling introduces significant overhead:
   - VecDeque operations for each query's pending queue
   - State pool contention/coordination across queries
   - Cache pollution from switching between query states
   - Loss of CPU prefetcher benefits (sequential access patterns disrupted)

3. **Already optimized baseline**: The existing sequential `QueryIterator` is highly optimized with:
   - StatePool for allocation reuse
   - SmallVec for stack allocation
   - Efficient BFS traversal
   - SIMD-accelerated operations (when applicable)

4. **MLP hypothesis invalid for this workload**: Cimple-style MLP works when:
   - Operations have long DRAM latencies (100+ ns)
   - Workload is I/O or memory bound
   - Cache hit rates are low

   None of these conditions apply here - the workload is CPU-bound with high cache hit rates.

### Decision
- [ ] Accept: Commit changes
- [x] **Reject: Revert changes**

### Lessons Learned

1. **Profile before optimizing**: Should have profiled to confirm DRAM latency was a bottleneck before implementing MLP.

2. **Cache effects dominate**: For in-memory data structures that fit in cache, interleaving doesn't help - it actually hurts by polluting the cache and disrupting prefetching.

3. **MLP requires memory-bound workloads**: Cimple-style interleaving only helps when waiting for slow I/O or uncached memory accesses.

4. **Simple is often faster**: The straightforward sequential approach benefits from:
   - Better cache locality (one query at a time)
   - CPU prefetcher effectiveness
   - No coordination overhead

5. **Batch APIs can still be useful** for ergonomics (single function call for multiple queries) even if not for performance. However, the implementation should just loop sequentially internally.

---

## Experiment 2: Varying Batch Size

### Hypothesis
Larger batch sizes provide more MLP opportunity but increase memory pressure.

### Design
Test batch sizes: 4, 8, 16, 32

### Results
(To be filled)

---

## Notes

- Dictionary terms are short (average ~8 chars in English)
- SIMD optimizations already achieve 20-64% gains within single queries
- This experiment targets query-level parallelism, not intra-query SIMD
