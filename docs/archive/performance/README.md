# Archived Performance Documentation

This directory contains historical performance optimization documentation for liblevenshtein-rust.

**Status:** Archived - For historical reference only
**Current Documentation:** See [`docs/PERFORMANCE.md`](../../developer-guide/performance.md) for current performance information

---

## Overview

These documents track the optimization journey from v0.1.0 through v0.2.0, showing the iterative process of improving performance through profiling, benchmarking, and targeted optimizations.

**Key Results:**
- 40-60% overall performance improvements
- 3.3x speedup for DAWG operations
- 15-50% faster PathMap edge iteration
- 5-18% improvements for filtering/prefix operations

---

## Recommended Reading Order

### 1. Start Here: High-Level Overview

- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - Complete optimization overview
  - All optimization phases summarized
  - Key takeaways and lessons learned
  - Profiling-guided approach

### 2. Detailed Phase Reports

See [`../optimization/`](../optimization/README.md) for phase-by-phase optimization journey:

- **[Phase 4: SmallVec Investigation](../optimization/PHASE4_SMALLVEC_INVESTIGATION.md)**
  - Stack-allocated vectors for small collections
  - Reduced heap allocation pressure

- **[Phase 5: StatePool Results](../optimization/PHASE5_STATEPOOL_RESULTS.md)** - **⭐ Exceptional Results**
  - Object pool pattern for state reuse
  - Eliminated allocation overhead in hot paths

- **[Phase 6: Arc Path Results](../optimization/PHASE6_ARC_PATH_RESULTS.md)** - **⭐ Highly Successful**
  - Shared ownership for path tracking
  - Eliminated expensive cloning

- **[Profiling Comparison](../optimization/PROFILING_COMPARISON.md)**
  - Before/after flamegraphs
  - Hotspot identification

### 3. Specific Optimizations

#### Core Engine
- **[ARC_OPTIMIZATION_RESULTS.md](ARC_OPTIMIZATION_RESULTS.md)** - Arc-based path sharing
- **[PATHNODE_OPTIMIZATION_RESULTS.md](PATHNODE_OPTIMIZATION_RESULTS.md)** - Lightweight node optimization
- **[QUERY_ARC_ANALYSIS.md](QUERY_ARC_ANALYSIS.md)** - Arc usage patterns in queries

#### DAWG Backend
- **[DAWG_OPTIMIZATION_RESULTS.md](DAWG_OPTIMIZATION_RESULTS.md)** - 3.3x speedup
- **[DAWG_OPTIMIZATIONS_APPLIED.md](DAWG_OPTIMIZATIONS_APPLIED.md)** - Implementation details
- **[DAWG_OPTIMIZATION_OPPORTUNITIES.md](DAWG_OPTIMIZATION_OPPORTUNITIES.md)** - Future improvements
- **[INDEX_BASED_QUERY_RESULTS.md](INDEX_BASED_QUERY_RESULTS.md)** - Memory-efficient queries

#### Serialization & Storage
- **[SERIALIZATION_OPTIMIZATION_RESULTS.md](SERIALIZATION_OPTIMIZATION_RESULTS.md)** - Compression benchmarks
- **[SERIALIZATION_OPTIMIZATIONS_APPLIED.md](SERIALIZATION_OPTIMIZATIONS_APPLIED.md)** - Implementation details
- **[SERIALIZATION_OPTIMIZATION_PLAN.md](SERIALIZATION_OPTIMIZATION_PLAN.md)** - Planning document
- **[SERIALIZATION_BENCHMARK_BASELINE.md](SERIALIZATION_BENCHMARK_BASELINE.md)** - Baseline measurements

#### Filtering & Code Completion
- **[CODE_COMPLETION_PERFORMANCE.md](CODE_COMPLETION_PERFORMANCE.md)** - Optimization strategies
- **[CONTEXTUAL_FILTERING_OPTIMIZATION.md](CONTEXTUAL_FILTERING_OPTIMIZATION.md)** - Bitmap masking

#### Other Optimizations
- **[THRESHOLD_TUNING_RESULTS.md](THRESHOLD_TUNING_RESULTS.md)** - Adaptive thresholds
- **[PGO_IMPACT_ANALYSIS.md](PGO_IMPACT_ANALYSIS.md)** - Profile-Guided Optimization analysis

### 4. Comparisons & Validation

- **[DAWG_COMPARISON.md](DAWG_COMPARISON.md)** - DAWG vs PathMap performance
- **[JAVA_COMPARISON.md](JAVA_COMPARISON.md)** - vs original Java implementation
- **[REAL_WORLD_VALIDATION.md](REAL_WORLD_VALIDATION.md)** - System dictionary tests

### 5. Early Phase Results

- **[PHASE2_RESULTS.md](PHASE2_RESULTS.md)** - Early optimization phase
- **[PHASE3_RESULTS.md](PHASE3_RESULTS.md)** - Intermediate results
- **[PROFILING_AND_PGO_RESULTS.md](PROFILING_AND_PGO_RESULTS.md)** - Profiling + PGO experiments
- **[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)** - General performance analysis
- **[OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)** - Additional optimization results

---

## Document Organization

```
archive/performance/
├── README.md (this file)
│
├── High-Level Overviews
│   └── OPTIMIZATION_SUMMARY.md          # ⭐ Start here
│
├── Core Engine Optimizations
│   ├── ARC_OPTIMIZATION_RESULTS.md
│   ├── PATHNODE_OPTIMIZATION_RESULTS.md
│   └── QUERY_ARC_ANALYSIS.md
│
├── DAWG Backend
│   ├── DAWG_OPTIMIZATION_RESULTS.md
│   ├── DAWG_OPTIMIZATIONS_APPLIED.md
│   ├── DAWG_OPTIMIZATION_OPPORTUNITIES.md
│   └── INDEX_BASED_QUERY_RESULTS.md
│
├── Serialization
│   ├── SERIALIZATION_OPTIMIZATION_RESULTS.md
│   ├── SERIALIZATION_OPTIMIZATIONS_APPLIED.md
│   ├── SERIALIZATION_OPTIMIZATION_PLAN.md
│   └── SERIALIZATION_BENCHMARK_BASELINE.md
│
├── Filtering & Code Completion
│   ├── CODE_COMPLETION_PERFORMANCE.md
│   └── CONTEXTUAL_FILTERING_OPTIMIZATION.md
│
├── Comparisons & Validation
│   ├── DAWG_COMPARISON.md
│   ├── JAVA_COMPARISON.md
│   └── REAL_WORLD_VALIDATION.md
│
├── Tuning & Analysis
│   ├── THRESHOLD_TUNING_RESULTS.md
│   ├── PGO_IMPACT_ANALYSIS.md
│   └── PERFORMANCE_ANALYSIS.md
│
└── Early Phases
    ├── PHASE2_RESULTS.md
    ├── PHASE3_RESULTS.md
    ├── PROFILING_AND_PGO_RESULTS.md
    └── OPTIMIZATION_RESULTS.md
```

---

## Key Lessons Learned

### 1. Profile Before Optimizing
- Flamegraphs revealed true hotspots (often surprising)
- Intuition about "slow" code was frequently wrong
- Measure, don't guess

### 2. Object Pools Are Powerful
- StatePool elimination of allocation overhead was exceptional
- Significant impact in tight loops
- Low implementation complexity for high reward

### 3. Arc Sharing Beats Cloning
- Cheap reference counting vs expensive deep copies
- Especially effective for immutable data
- Path sharing was highly successful

### 4. Small Optimizations Add Up
- SmallVec, inlining, lazy iteration individually modest
- Combined effect: 5-18% improvements
- Incremental approach works

### 5. Real-World Benchmarks Matter
- Synthetic benchmarks can mislead
- System dictionaries (/usr/share/dict/words) provided realistic validation
- Performance varies with dictionary characteristics

---

## Methodology

### Tools Used
- **Criterion.rs** - Statistical benchmarking
- **Flamegraph** - CPU profiling and hotspot identification
- **cargo-llvm-cov** - Code coverage analysis
- **PGO** - Profile-Guided Optimization (tested, not currently used)

### Process
1. **Baseline** - Establish current performance
2. **Profile** - Identify hotspots with flamegraphs
3. **Hypothesize** - Target specific optimization
4. **Implement** - Make targeted changes
5. **Benchmark** - Measure impact with Criterion
6. **Validate** - Test with real-world dictionaries
7. **Iterate** - Repeat for next optimization

---

## Current Status

This archived documentation reflects the state of optimizations through v0.2.0. For current performance information, see:

- **[docs/PERFORMANCE.md](../../developer-guide/performance.md)** - Current performance documentation
- **[docs/FUTURE_ENHANCEMENTS.md](../../research/future-enhancements.md)** - Planned future optimizations

---

## Navigation

- **[← Back to Documentation Index](../../README.md)**
- **[→ Current Performance Docs](../../developer-guide/performance.md)**
- **[→ Optimization Phases](../optimization/README.md)**
