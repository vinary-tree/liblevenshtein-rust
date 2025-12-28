# ARTrie Optimization Scientific Ledger

**Created**: 2025-12-28
**Purpose**: Track empirical results for Persistent ARTrie (u8 and u32 variants) optimizations with statistical rigor

## Methodology

### Statistical Requirements
- **Sample size**: 50 benchmark iterations (Criterion default)
- **Significance level**: α = 0.05 (95% confidence)
- **Effect size**: Report percentage improvement and Cohen's d where applicable
- **Tool**: Criterion.rs built-in statistical analysis (t-test, confidence intervals)

### Decision Criteria
- **ACCEPT**: p < 0.05, no regressions > 2%, all tests pass
- **REJECT**: p ≥ 0.05 or introduces regressions > 2%

### Hardware Configuration
- **CPU**: Intel Xeon E5-2699 v3 @ 2.30GHz (Turbo: 3.57 GHz)
- **Cores**: 36 physical cores (72 threads with HT)
- **Architecture**: Haswell-EP, x86_64
- **SIMD Support**: AVX2, AVX, FMA, SSE4.2, SSE4.1
- **L1 Cache**: 1.1 MiB each (Data + Instruction)
- **L2 Cache**: ~9 MB
- **L3 Cache**: ~45 MB
- **RAM**: 252 GB DDR4-2133 ECC
- **Storage**: Samsung 990 PRO 4TB NVMe
- **Benchmark Environment**: `taskset -c 0` (pinned to core 0)

---

## Experiment 1: Baseline Benchmarks

**Date**: 2025-12-28
**Branch**: `master`
**Purpose**: Establish performance baseline before optimizations
**Status**: COMPLETE

### Test Configuration
- **Benchmark framework**: Criterion.rs 0.5
- **Sample size**: 50 iterations (default), 10-20 for I/O benchmarks
- **CPU affinity**: `taskset -c 0`

### PersistentARTrie (u8 variant) Baseline

**Status**: COMPLETE

#### Construction Performance

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 4.4586 µs | 4.4213 µs    | 4.5044 µs    | 22.429 Melem/s |
| 500       | 44.657 µs | 44.217 µs    | 45.079 µs    | 11.196 Melem/s |
| 1000      | 63.775 µs | 63.351 µs    | 64.246 µs    | 15.680 Melem/s |
| 5000      | 279.14 µs | 276.97 µs    | 281.34 µs    | 17.912 Melem/s |

#### Lookup Performance (100 queries)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 4.6794 µs | 4.6413 µs    | 4.7157 µs    | 21.370 Melem/s |
| 1000      | 6.4569 µs | 6.3422 µs    | 6.5698 µs    | 15.487 Melem/s |
| 5000      | 7.0879 µs | 7.0289 µs    | 7.1456 µs    | 14.109 Melem/s |

#### Edge Traversal Performance (DFS)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 5.7938 µs | 5.7431 µs    | 5.8487 µs    | 17.260 Melem/s |
| 1000      | 2.1033 µs | 2.0857 µs    | 2.1211 µs    | 475.44 Melem/s |
| 5000      | 464.83 ns | 460.75 ns    | 468.53 ns    | 10.757 Gelem/s |

#### Transition Performance (100 transitions)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 24.731 µs | 24.583 µs    | 24.889 µs    | 4.0435 Melem/s |
| 1000      | 11.617 µs | 11.541 µs    | 11.699 µs    | 8.6083 Melem/s |
| 5000      | 10.189 µs | 10.127 µs    | 10.252 µs    | 9.8146 Melem/s |

#### Disk I/O Performance

| Dict Size | Operation | Mean Time | 95% CI Lower | 95% CI Upper |
|-----------|-----------|-----------|--------------|--------------|
| 100       | create_insert_sync | 81.735 µs | 80.403 µs | 83.760 µs |
| 100       | recovery  | 118.39 µs | 116.63 µs | 119.53 µs |
| 100       | checkpoint | 643.14 µs | 636.25 µs | 652.41 µs |
| 500       | create_insert_sync | 243.60 µs | 240.37 µs | 246.86 µs |
| 500       | recovery  | 443.53 µs | 439.58 µs | 448.18 µs |
| 500       | checkpoint | 5.2551 ms | 5.2002 ms | 5.3398 ms |
| 1000      | create_insert_sync | 342.68 µs | 339.80 µs | 346.11 µs |
| 1000      | recovery  | 657.30 µs | 651.13 µs | 667.03 µs |
| 1000      | checkpoint | 5.2327 ms | 5.1856 ms | 5.3337 ms |

### PersistentARTrieChar (u32 variant) Baseline

**Status**: COMPLETE

#### Construction Performance (Unicode)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 153.54 µs | 151.54 µs    | 155.59 µs    | 651.29 Kelem/s |
| 500       | 948.09 µs | 935.84 µs    | 955.72 µs    | 527.38 Kelem/s |
| 1000      | 2.0792 ms | 2.0467 ms    | 2.1179 ms    | 480.96 Kelem/s |
| 5000      | 10.281 ms | 10.121 ms    | 10.446 ms    | 486.35 Kelem/s |

#### Construction Performance (ASCII)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 206.78 µs | 204.15 µs    | 209.11 µs    | 483.61 Kelem/s |
| 500       | 1.0480 ms | 1.0383 ms    | 1.0574 ms    | 477.09 Kelem/s |
| 1000      | 2.1316 ms | 2.1165 ms    | 2.1497 ms    | 469.13 Kelem/s |
| 5000      | 11.387 ms | 11.342 ms    | 11.422 ms    | 439.11 Kelem/s |

#### Lookup Performance (100 queries)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 15.269 µs | 15.144 µs    | 15.417 µs    | 6.5494 Melem/s |
| 1000      | 18.397 µs | 18.286 µs    | 18.503 µs    | 5.4356 Melem/s |
| 5000      | 20.109 µs | 19.948 µs    | 20.285 µs    | 4.9730 Melem/s |

#### CJK Lookup Performance (100 queries, 1000 terms)

| Benchmark | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| cjk_lookup | 11.089 µs | 10.989 µs   | 11.185 µs    | 9.0176 Melem/s |

#### Edge Traversal Performance (DFS)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 12.654 µs | 12.531 µs    | 12.794 µs    | 7.9024 Melem/s |
| 1000      | 88.141 µs | 87.497 µs    | 88.809 µs    | 11.345 Melem/s |
| 5000      | 336.50 µs | 334.13 µs    | 338.81 µs    | 14.859 Melem/s |

#### Transition Performance (100 transitions)

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 13.659 µs | 13.565 µs    | 13.766 µs    | 7.3210 Melem/s |
| 1000      | 17.310 µs | 17.150 µs    | 17.458 µs    | 5.7772 Melem/s |
| 5000      | 19.361 µs | 19.217 µs    | 19.519 µs    | 5.1650 Melem/s |

#### Emoji Transition Performance

| Benchmark | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| emoji_transitions | 4.6050 µs | 4.5610 µs | 4.6534 µs | 10.858 Melem/s |

#### Iteration Performance

| Dict Size | Mean Time | 95% CI Lower | 95% CI Upper | Throughput |
|-----------|-----------|--------------|--------------|------------|
| 100       | 21.936 µs | 21.763 µs    | 22.134 µs    | 4.5587 Melem/s |
| 500       | 78.178 µs | 77.273 µs    | 79.396 µs    | 6.3957 Melem/s |
| 1000      | 161.03 µs | 159.54 µs    | 162.69 µs    | 6.2101 Melem/s |

#### Disk I/O Performance

| Dict Size | Operation | Mean Time | 95% CI Lower | 95% CI Upper |
|-----------|-----------|-----------|--------------|--------------|
| 100       | create_insert_sync | 124.74 µs | 123.43 µs | 125.81 µs |
| 100       | recovery  | 148.75 µs | 146.89 µs | 150.74 µs |
| 100       | checkpoint | 83.323 ms | 82.389 ms | 85.285 ms |
| 500       | create_insert_sync | 416.93 µs | 410.59 µs | 424.91 µs |
| 500       | recovery  | 589.32 µs | 583.16 µs | 595.38 µs |
| 500       | checkpoint | 322.93 ms | 320.80 ms | 325.18 ms |
| 1000      | create_insert_sync | 791.22 µs | 786.82 µs | 796.66 µs |
| 1000      | recovery  | 1.0952 ms | 1.0876 ms | 1.1099 ms |
| 1000      | checkpoint | 582.45 ms | 578.52 ms | 586.04 ms |

---

## Experiment 2: Perf Profile Analysis

**Date**: 2025-12-28
**Branch**: `master`
**Purpose**: Identify performance bottlenecks using perf profiling
**Status**: IN PROGRESS (hardware counters complete, hot path analysis pending)

### u8 Variant Hot Path Analysis

#### Top Symbols by CPU Time

*Pending detailed perf record analysis*

#### Hardware Performance Counters

| Metric | Value | Rate |
|--------|-------|------|
| Cycles | 343.94 B | - |
| Instructions | 453.47 B | - |
| IPC    | 1.32 | - |
| Cache References | 3.83 B | - |
| Cache Misses | 301.36 M | - |
| Cache Miss Rate | - | 7.87% |
| Branch Instructions | 87.51 B | - |
| Branch Misses | 2.71 B | - |
| Branch Miss Rate | - | 3.09% |

### u32 Variant Hot Path Analysis

#### Top Symbols by CPU Time

*Pending detailed perf record analysis*

#### Hardware Performance Counters

| Metric | Value | Rate |
|--------|-------|------|
| Cycles | 410.71 B | - |
| Instructions | 434.86 B | - |
| IPC    | 1.06 | - |
| Cache References | 3.73 B | - |
| Cache Misses | 258.19 M | - |
| Cache Miss Rate | - | 6.92% |
| Branch Instructions | 82.14 B | - |
| Branch Misses | 2.60 B | - |
| Branch Miss Rate | - | 3.16% |

### Identified Bottlenecks

**Preliminary Analysis from Hardware Counters:**

1. **u32 Variant Low IPC (1.06)**: The u32 variant shows lower instruction-level parallelism compared to u8 (1.32 IPC). This suggests potential memory stalls or inefficient instruction scheduling. The AVX2 SIMD code in CharNode16 may have dependency chains that limit pipelining.

2. **Both Variants - High Branch Misses (~3%)**: Both variants show ~3% branch misprediction rates. The `find_child()` implementations have conditional branches that the predictor struggles with, especially for not-found cases.

3. **Cache Miss Rate Moderate (7-8%)**: Both variants show manageable cache miss rates. The u32 variant is slightly better (6.92% vs 7.87%), likely due to fewer memory accesses per lookup despite larger key sizes.

**Optimization Opportunities:**
- S4 (Inline Expansion): Reducing call overhead may improve IPC
- U32-1 (Early-exit AVX2): Could improve u32 IPC by avoiding unnecessary register loads
- S3 (Branch Hints): `unlikely()` hints may reduce branch misprediction penalty

---

## Hypothesis Testing Log

### Tier 1: High Impact, Low Risk

#### Hypothesis S4: Inline Expansion of Hot Functions
- **Status**: PENDING
- **H0**: `#[inline(always)]` provides no improvement
- **H1**: Forced inlining reduces call overhead by >3%
- **Files**: All node*.rs files
- **Acceptance**: p < 0.05, >3% improvement

#### Hypothesis U8-2: SIMD Prefix Matching (u8 variant)
- **Status**: PENDING
- **H0**: SIMD prefix matching provides no improvement
- **H1**: SIMD prefix matching improves `match_key()` by >10%
- **File**: `src/dictionary/persistent_artrie/nodes/mod.rs`
- **Acceptance**: p < 0.05, >10% improvement

#### Hypothesis U32-1: Early-exit AVX2 (u32 variant)
- **Status**: PENDING
- **H0**: Current dual-register AVX2 is optimal
- **H1**: Early-exit on first register match improves average case by >5%
- **File**: `src/dictionary/persistent_artrie_char/nodes/node16_char.rs`
- **Acceptance**: p < 0.05, >5% improvement

### Tier 2: Medium Impact

#### Hypothesis U32-3: SIMD Char Prefix Matching
- **Status**: PENDING
- **H0**: SIMD provides no improvement for 6-char prefix matching
- **H1**: AVX2 u32 comparison improves `match_key()` by >10%
- **File**: `src/dictionary/persistent_artrie_char/nodes/mod.rs`
- **Acceptance**: p < 0.05, >10% improvement

#### Hypothesis S2: Cache Line Alignment
- **Status**: PENDING
- **H0**: Cache line alignment provides no improvement
- **H1**: 64-byte alignment reduces cache line splits by >3%
- **Files**: All node*.rs files
- **Acceptance**: p < 0.05, >3% improvement, no regression

### Tier 3: Speculative

#### Hypothesis U8-1: SSE4.1 to AVX2 Upgrade
- **Status**: PENDING
- **H0**: AVX2 provides no improvement over SSE4.1 for 16-byte comparison
- **H1**: AVX2's wider registers reduce instruction count by >10%
- **File**: `src/dictionary/persistent_artrie/nodes/node16.rs`
- **Acceptance**: p < 0.05, >10% improvement

#### Hypothesis U32-2: Binary Search for CharBucket
- **Status**: PENDING
- **H0**: HashMap lookup is optimal for >48 children
- **H1**: Sorted array + binary search outperforms HashMap for <256 children
- **File**: `src/dictionary/persistent_artrie_char/nodes/bucket_char.rs`
- **Acceptance**: p < 0.05, >10% improvement

#### Hypothesis U8-3: SIMD Scan for Sparse Node48
- **Status**: PENDING
- **H0**: Index array lookup is optimal for Node48
- **H1**: SIMD scan reduces lookup time for sparse Node48 by >5%
- **File**: `src/dictionary/persistent_artrie/nodes/node48.rs`
- **Acceptance**: p < 0.05, >5% improvement

#### Hypothesis S1: Software Prefetching
- **Status**: PENDING
- **H0**: Software prefetching provides no improvement
- **H1**: Prefetching next node reduces cache miss latency by >5%
- **Files**: All dict_impl.rs files
- **Acceptance**: p < 0.05, >5% improvement

#### Hypothesis S3: Branch Prediction Hints
- **Status**: PENDING
- **H0**: Branch hints provide no improvement
- **H1**: `unlikely()` hints reduce misprediction by >3%
- **Files**: All node*.rs find_child() implementations
- **Acceptance**: p < 0.05, >3% improvement

#### Hypothesis U8-4: Loop Unrolling in add_child()
- **Status**: PENDING
- **H0**: Loop unrolling provides no improvement for shift operations
- **H1**: Unrolled shift reduces insertion time by >3%
- **File**: `src/dictionary/persistent_artrie/nodes/node16.rs`
- **Acceptance**: p < 0.05, >3% improvement

#### Hypothesis U32-4: SIMD Binary Search for CharNode48
- **Status**: PENDING
- **H0**: Current binary search is optimal
- **H1**: SIMD-assisted binary search reduces lookup time by >10%
- **File**: `src/dictionary/persistent_artrie_char/nodes/node48_char.rs`
- **Acceptance**: p < 0.05, >10% improvement

---

## Summary of Results

| Hypothesis | Status | Change | p-value | Decision |
|------------|--------|--------|---------|----------|
| S4         | PENDING |       |         |          |
| U8-2       | PENDING |       |         |          |
| U32-1      | PENDING |       |         |          |
| U32-3      | PENDING |       |         |          |
| S2         | PENDING |       |         |          |
| U8-1       | PENDING |       |         |          |
| U32-2      | PENDING |       |         |          |
| U8-3       | PENDING |       |         |          |
| S1         | PENDING |       |         |          |
| S3         | PENDING |       |         |          |
| U8-4       | PENDING |       |         |          |
| U32-4      | PENDING |       |         |          |
