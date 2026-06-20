# WallBreaker Benchmarking Plan

**Date**: 2025-11-06
**Purpose**: Define performance validation strategy, success criteria, and benchmarking methodology for WallBreaker implementation.

---

## Executive Summary

This document outlines the comprehensive benchmarking plan to validate WallBreaker performance improvements. The primary goal is to confirm that Option B (Hybrid) achieves **3300x speedup** over traditional approach for large error bounds and long patterns.

**Key Success Criteria**:
- Query time < 0.2ms for 100-char pattern, max_distance=16, 100K dictionary
- Speedup ≥ 1000x for max_distance ≥ 4, pattern length ≥ 50
- Memory overhead < 50% during query execution
- Construction time increase < 20%

---

## Table of Contents

1. [Benchmarking Objectives](#benchmarking-objectives)
2. [Test Scenarios](#test-scenarios)
3. [Performance Metrics](#performance-metrics)
4. [Success Criteria](#success-criteria)
5. [Benchmark Implementation](#benchmark-implementation)
6. [Validation Methodology](#validation-methodology)
7. [Hardware Configuration](#hardware-configuration)
8. [Reporting Format](#reporting-format)

---

## 1. Benchmarking Objectives

### Primary Objectives

1. **Validate WallBreaker Performance Gain**
   - Confirm speedup targets for various error bounds
   - Identify sweet spot (where WallBreaker excels)
   - Identify weak spots (where traditional is better)

2. **Ensure Correctness**
   - Verify WallBreaker results match traditional approach
   - Validate distance calculations are accurate
   - Confirm no false positives or false negatives

3. **Measure Resource Usage**
   - Memory overhead during query execution
   - Memory overhead during construction
   - CPU utilization patterns

4. **Establish Baseline**
   - Document current performance (traditional approach)
   - Provide comparison point for future optimizations
   - Track performance regressions

### Secondary Objectives

- Identify optimization opportunities
- Compare Hybrid (Option B) with projected Full SCDAWG (Option A)
- Validate automatic selection logic (when to use WallBreaker)
- Document performance characteristics for users

---

## 2. Test Scenarios

### Scenario Matrix

We'll test across three dimensions:
1. **Error Bound** (max_distance): 1, 2, 4, 8, 12, 16
2. **Pattern Length**: 4, 10, 20, 50, 100, 200 characters
3. **Dictionary Size**: 1K, 10K, 100K, 500K, 1M terms

**Total Scenarios**: 6 × 6 × 5 = **180 benchmark combinations**

### Scenario Categories

#### Category 1: Small Error Bound (Traditional Expected to Win)
**Purpose**: Confirm WallBreaker doesn't hurt performance when not beneficial

| max_distance | Pattern Length | Dictionary Size | Expected Winner |
|--------------|----------------|-----------------|-----------------|
| 1 | 10 | 10K | Traditional |
| 2 | 20 | 100K | Traditional |
| 2 | 50 | 100K | WallBreaker (marginal) |

**Success Criteria**: WallBreaker no more than 20% slower than traditional

---

#### Category 2: Medium Error Bound (WallBreaker Should Excel)
**Purpose**: Validate WallBreaker's core value proposition

| max_distance | Pattern Length | Dictionary Size | Expected Speedup |
|--------------|----------------|-----------------|------------------|
| 4 | 20 | 10K | 5-10x |
| 4 | 50 | 100K | 50-100x |
| 8 | 50 | 100K | 500-1000x |
| 8 | 100 | 500K | 1000-2000x |

**Success Criteria**: Speedup ≥ 10x for all scenarios

---

#### Category 3: Large Error Bound (Maximum Performance Gain)
**Purpose**: Demonstrate extreme wall effect mitigation

| max_distance | Pattern Length | Dictionary Size | Expected Speedup |
|--------------|----------------|-----------------|------------------|
| 12 | 100 | 100K | 2000-3000x |
| 16 | 100 | 100K | 3000-5000x |
| 16 | 200 | 500K | 5000-10000x |

**Success Criteria**: Speedup ≥ 1000x for all scenarios, query time < 1ms

---

#### Category 4: Edge Cases
**Purpose**: Ensure robustness

| Scenario | Pattern | max_distance | Expected Behavior |
|----------|---------|--------------|-------------------|
| Very short pattern | "ab" | 4 | Handle gracefully (may fall back to traditional) |
| Pattern = max_distance | "test" | 4 | Pattern split into 5 pieces, each very short |
| Empty dictionary | Any | Any | Return empty results quickly |
| Single-term dictionary | Any | Any | Correct result, low overhead |
| Very large dictionary | Any | 16 | 5M terms, still fast |

**Success Criteria**: No crashes, correct results, reasonable performance

---

### Benchmark Datasets

#### Dictionary 1: English Words (100K terms)
**Source**: `/usr/share/dict/words` (or similar)
**Characteristics**:
- Realistic word distribution
- Lengths 2-20 characters
- Common prefixes/suffixes

**Test Patterns**:
- Existing words: "dictionary", "performance", "algorithm"
- Typos: "dictionery" (1 error), "perfomance" (1 error)
- Fuzzy matches: "algoritm" (1 error), "aproximate" (2 errors)

---

#### Dictionary 2: Synthetic (1M terms)
**Generation**: Random strings from alphabet
**Characteristics**:
- Lengths 5-50 characters
- Uniform distribution
- No natural language structure

**Test Patterns**:
- Random substrings
- Random with errors injected

---

#### Dictionary 3: DNA Sequences (500K terms)
**Source**: Short genomic sequences (k-mers)
**Characteristics**:
- 4-letter alphabet (A, C, G, T)
- Lengths 20-100 characters
- High similarity (challenging for distance algorithms)

**Test Patterns**:
- Real sequences with sequencing errors
- Synthetic sequences with substitutions

---

## 3. Performance Metrics

### 3.1 Primary Metrics

#### Query Time
**Definition**: Time from query start to completion (all results yielded)

**Measurement**:
```rust
let start = Instant::now();
let results: Vec<_> = fuzzy_search(&dict, pattern, max_distance).collect();
let duration = start.elapsed();
```

**Reporting**: Median, mean, min, max, stddev over 100 iterations

**Target**: < 0.2ms for 100-char pattern, max_distance=16, 100K dictionary

---

#### Speedup Ratio
**Definition**: Traditional query time / WallBreaker query time

**Calculation**:
```
speedup = median_time_traditional / median_time_wallbreaker
```

**Reporting**: Speedup factor (e.g., "3300x faster")

**Target**: ≥ 1000x for max_distance ≥ 4, pattern length ≥ 50

---

#### Memory Usage (Query)
**Definition**: Peak memory used during query execution

**Measurement**:
```rust
// Track allocations during query
let mem_before = get_memory_usage();
let results: Vec<_> = fuzzy_search(&dict, pattern, max_distance).collect();
let mem_after = get_memory_usage();
let mem_delta = mem_after - mem_before;
```

**Reporting**: MB used, percentage overhead vs baseline

**Target**: < 50% overhead vs traditional

---

### 3.2 Secondary Metrics

#### Construction Time
**Definition**: Time to build dictionary structure (WallBreaker-enabled)

**Measurement**:
```rust
let start = Instant::now();
let dict = SuffixAutomaton::from_iter(terms);
let duration = start.elapsed();
```

**Reporting**: Median time over 10 iterations

**Target**: < 20% increase vs current SuffixAutomaton construction

---

#### Construction Memory
**Definition**: Memory used by dictionary structure

**Measurement**: Size of SuffixAutomaton with parent links vs without

**Reporting**: MB used, percentage overhead

**Target**: < 30% increase (parent links are small)

---

#### Substring Search Time
**Definition**: Time to find all exact substring matches (phase 1 of WallBreaker)

**Measurement**:
```rust
let start = Instant::now();
let matches = dict.find_exact_substring(pattern_piece);
let duration = start.elapsed();
```

**Reporting**: Microseconds per search

**Target**: < 10% of total query time

---

#### Extension Time
**Definition**: Time for left + right extension from each substring match

**Measurement**: Profile left/right extension separately

**Reporting**: Microseconds per extension

**Target**: < 90% of total query time (bulk of work)

---

### 3.3 Correctness Metrics

#### Result Accuracy
**Definition**: Percentage of results that match traditional approach

**Measurement**:
```rust
let trad_results: HashSet<_> = traditional_search(...).collect();
let wb_results: HashSet<_> = wallbreaker_search(...).collect();
let accuracy = (trad_results == wb_results) as f64 * 100.0;
```

**Reporting**: Percentage (should be 100%)

**Target**: 100% accuracy (no false positives, no false negatives)

---

#### Distance Accuracy
**Definition**: Verify computed distances are correct

**Measurement**: For each result, compute true Levenshtein distance and compare

**Reporting**: Percentage of correct distances

**Target**: 100% accuracy

---

## 4. Success Criteria

### 4.1 Tier 1: MUST ACHIEVE (Blockers for Release)

| Criterion | Target | Measurement | Priority |
|-----------|--------|-------------|----------|
| **Correctness** | 100% | Results match traditional | CRITICAL |
| **Distance Accuracy** | 100% | Distances correct | CRITICAL |
| **No Crashes** | 0 | All scenarios run | CRITICAL |
| **Performance (b≥4)** | ≥ 100x speedup | Median query time | CRITICAL |

**If any Tier 1 criterion fails**: Do not proceed to release. Fix issues first.

---

### 4.2 Tier 2: SHOULD ACHIEVE (Performance Targets)

| Criterion | Target | Measurement | Priority |
|-----------|--------|-------------|----------|
| **Query Time (b=16)** | < 0.2ms | 100-char, 100K dict | HIGH |
| **Speedup (b≥4)** | ≥ 1000x | Pattern ≥50 chars | HIGH |
| **Memory Overhead** | < 50% | Query memory | HIGH |
| **Construction Time** | < 20% increase | Build time | HIGH |
| **Small b Performance** | ≤ 20% slower | b ≤ 2 | MEDIUM |

**If Tier 2 criteria fail**: Investigate optimization opportunities. May still release with caveats.

---

### 4.3 Tier 3: NICE TO HAVE (Stretch Goals)

| Criterion | Target | Measurement | Priority |
|-----------|--------|-------------|----------|
| **Query Time (b=16)** | < 0.1ms | 100-char, 100K dict | LOW |
| **Speedup (b≥8)** | ≥ 3000x | Pattern ≥50 chars | LOW |
| **Memory Overhead** | < 30% | Query memory | LOW |
| **Construction Time** | < 10% increase | Build time | LOW |

**If Tier 3 criteria met**: Excellent! Document and celebrate. Not required for release.

---

### 4.4 Performance Targets by Scenario

**Small Error Bound (b ≤ 2)**:
- ✅ WallBreaker no more than 20% slower than traditional
- ✅ Automatic selection chooses traditional (correct decision)

**Medium Error Bound (b = 4-8)**:
- ✅ Speedup ≥ 100x for pattern ≥ 20 chars
- ✅ Speedup ≥ 1000x for pattern ≥ 50 chars
- ✅ Query time < 5ms

**Large Error Bound (b ≥ 12)**:
- ✅ Speedup ≥ 2000x
- ✅ Query time < 1ms
- ✅ Memory overhead < 50%

---

## 5. Benchmark Implementation

### 5.1 Benchmark Structure

**File**: `/benches/wallbreaker_comparison.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use liblevenshtein::prelude::*;

fn benchmark_wallbreaker_vs_traditional(c: &mut Criterion) {
    let mut group = c.benchmark_group("wallbreaker_comparison");

    // Load dictionaries
    let dict_10k = load_dictionary("data/dict_10k.txt");
    let dict_100k = load_dictionary("data/dict_100k.txt");
    let dict_1m = load_dictionary("data/dict_1m.txt");

    // Test patterns
    let patterns = vec![
        ("short", "test", 4),
        ("medium", "algorithm", 8),
        ("long", "extraordinarily", 16),
    ];

    for (name, pattern, max_distance) in patterns {
        // Traditional approach
        group.bench_with_input(
            BenchmarkId::new("traditional", name),
            &pattern,
            |b, pattern| {
                b.iter(|| {
                    fuzzy_search(&dict_100k, black_box(pattern), max_distance)
                        .collect::<Vec<_>>()
                });
            },
        );

        // WallBreaker approach
        group.bench_with_input(
            BenchmarkId::new("wallbreaker", name),
            &pattern,
            |b, pattern| {
                b.iter(|| {
                    fuzzy_search_wallbreaker(&dict_100k, black_box(pattern), max_distance)
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_wallbreaker_vs_traditional);
criterion_main!(benches);
```

### 5.2 Benchmark Datasets

**Location**: `/benches/data/`

**Files to Generate**:
- `dict_1k.txt` - 1,000 terms
- `dict_10k.txt` - 10,000 terms
- `dict_100k.txt` - 100,000 terms
- `dict_500k.txt` - 500,000 terms
- `dict_1m.txt` - 1,000,000 terms

**Generation Script** (`/benches/generate_datasets.sh`):
```bash
#!/bin/bash

# English words (100K)
head -n 100000 /usr/share/dict/words > benches/data/dict_100k.txt
head -n 10000 /usr/share/dict/words > benches/data/dict_10k.txt
head -n 1000 /usr/share/dict/words > benches/data/dict_1k.txt

# Synthetic (1M)
python3 benches/generate_synthetic.py 1000000 > benches/data/dict_1m.txt
python3 benches/generate_synthetic.py 500000 > benches/data/dict_500k.txt
```

### 5.3 Test Pattern Generation

**File**: `/benches/generate_test_patterns.py`

```python
import random
import string

def generate_pattern(length: int, alphabet: str = string.ascii_lowercase) -> str:
    return ''.join(random.choice(alphabet) for _ in range(length))

def inject_errors(pattern: str, num_errors: int) -> str:
    """Inject substitutions, insertions, deletions"""
    pattern = list(pattern)
    for _ in range(num_errors):
        error_type = random.choice(['sub', 'ins', 'del'])
        pos = random.randint(0, len(pattern) - 1)

        if error_type == 'sub':
            pattern[pos] = random.choice(string.ascii_lowercase)
        elif error_type == 'ins':
            pattern.insert(pos, random.choice(string.ascii_lowercase))
        elif error_type == 'del' and len(pattern) > 1:
            del pattern[pos]

    return ''.join(pattern)

# Generate test patterns
for length in [10, 20, 50, 100, 200]:
    pattern = generate_pattern(length)
    for errors in [1, 2, 4, 8, 16]:
        if errors < length // 2:  # Reasonable error bound
            fuzzy = inject_errors(pattern, errors)
            print(f"{length},{errors},{pattern},{fuzzy}")
```

### 5.4 Running Benchmarks

**Full Benchmark Suite**:
```bash
# Run all benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench --bench wallbreaker_comparison

# Run specific scenario
RUSTFLAGS="-C target-cpu=native" cargo bench --bench wallbreaker_comparison -- "b=16"

# Generate flamegraph
RUSTFLAGS="-C target-cpu=native" cargo flamegraph --bench wallbreaker_comparison
```

**Quick Smoke Test**:
```bash
# Run subset of scenarios for quick validation
RUSTFLAGS="-C target-cpu=native" cargo bench --bench wallbreaker_comparison -- "quick"
```

---

## 6. Validation Methodology

### 6.1 Correctness Validation

**Step 1: Exact Match Test**
```rust
#[test]
fn test_wallbreaker_correctness() {
    let dict = load_test_dictionary();
    let pattern = "algorithm";
    let max_distance = 4;

    let trad_results: HashSet<_> =
        fuzzy_search(&dict, pattern, max_distance).collect();
    let wb_results: HashSet<_> =
        fuzzy_search_wallbreaker(&dict, pattern, max_distance).collect();

    assert_eq!(trad_results, wb_results, "Results must match exactly");
}
```

**Step 2: Distance Verification**
```rust
#[test]
fn test_distance_accuracy() {
    let dict = load_test_dictionary();
    let pattern = "test";
    let max_distance = 2;

    for result in fuzzy_search_wallbreaker(&dict, pattern, max_distance) {
        let actual_distance = levenshtein_distance(pattern, &result);
        assert!(actual_distance <= max_distance, "Distance must be within bound");
    }
}
```

**Step 3: Edge Case Testing**
```rust
#[test]
fn test_edge_cases() {
    let dict = load_test_dictionary();

    // Empty pattern
    let results: Vec<_> = fuzzy_search_wallbreaker(&dict, "", 0).collect();
    assert!(results.is_empty() || results == vec![""]);

    // Pattern longer than any term
    let results: Vec<_> = fuzzy_search_wallbreaker(&dict, "x".repeat(1000), 5).collect();
    // Should not crash, results depend on dictionary

    // max_distance = 0 (exact match)
    let results: Vec<_> = fuzzy_search_wallbreaker(&dict, "test", 0).collect();
    assert!(results.is_empty() || results == vec!["test"]);
}
```

### 6.2 Performance Validation

**Step 1: Baseline Measurement**
```bash
# Measure traditional approach performance
RUSTFLAGS="-C target-cpu=native" cargo bench --bench traditional_baseline

# Save results
cargo bench --bench traditional_baseline -- --save-baseline traditional
```

**Step 2: WallBreaker Measurement**
```bash
# Measure WallBreaker performance
RUSTFLAGS="-C target-cpu=native" cargo bench --bench wallbreaker_comparison

# Save results
cargo bench --bench wallbreaker_comparison -- --save-baseline wallbreaker
```

**Step 3: Comparison**
```bash
# Compare baselines
cargo bench -- --baseline traditional --baseline wallbreaker
```

### 6.3 Memory Validation

**Using Valgrind (massif)**:
```bash
# Profile memory usage
valgrind --tool=massif ./target/release/deps/wallbreaker_comparison-*

# Visualize
ms_print massif.out.* > memory_profile.txt
```

**Using Custom Memory Tracking**:
```rust
use std::alloc::{GlobalAlloc, Layout, System};

struct TrackingAllocator;

static mut ALLOCATED: usize = 0;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED += layout.size();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED -= layout.size();
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;
```

---

## 7. Hardware Configuration

### 7.1 Reference Hardware

**Primary Benchmark Machine**:
- **CPU**: AMD Ryzen 9 3950X (16 cores, 32 threads, 3.5-4.7 GHz)
- **RAM**: 64 GB DDR4-3200
- **Storage**: NVMe SSD
- **OS**: Linux 6.17.5-arch1-1

**Reason for Selection**: Representative of modern development machines, matches hardware mentioned in context.

### 7.2 Additional Test Platforms

**Low-End** (verify performance on constrained hardware):
- CPU: Intel Core i5 (4 cores)
- RAM: 8 GB
- Storage: SATA SSD

**High-End** (verify scalability):
- CPU: AMD Ryzen Threadripper (32+ cores)
- RAM: 128 GB
- Storage: NVMe RAID

### 7.3 Benchmark Configuration

**Rust Configuration**:
```toml
[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1
```

**Environment**:
```bash
export RUSTFLAGS="-C target-cpu=native"
export CARGO_PROFILE_BENCH_DEBUG=false
```

**System Preparation**:
- Disable CPU frequency scaling (set to performance mode)
- Disable hyperthreading (for consistency)
- Close unnecessary background processes
- Run benchmarks multiple times (≥10 iterations)
- Use `nice -n -20` for high priority

---

## 8. Reporting Format

### 8.1 Summary Report Template

```markdown
# WallBreaker Benchmark Results

**Date**: YYYY-MM-DD
**Hardware**: [CPU model, RAM, OS]
**Rust Version**: [version]
**Build**: RUSTFLAGS="-C target-cpu=native" cargo bench

---

## Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Query Time (b=16, 100 chars) | < 0.2ms | [measured latency] | pass/fail |
| Speedup (b>=4, pattern>=50) | >= 1000x | [measured speedup] | pass/fail |
| Memory Overhead | < 50% | [measured overhead] | pass/fail |
| Construction Time Increase | < 20% | [measured increase] | pass/fail |
| Correctness | 100% | 100% | ✅ |

---

## Detailed Results

### Small Error Bound (b ≤ 2)

| Pattern | max_distance | Traditional | WallBreaker | Speedup | Winner |
|---------|--------------|-------------|-------------|---------|--------|
| "test" (4 chars) | 2 | [measure] | [measure] | [measure] | Traditional |
| "algorithm" (9 chars) | 2 | [measure] | [measure] | [measure] | Traditional |

**Analysis**: Fill in the measured small-error-bound delta; slower WallBreaker behavior is expected here.

---

### Medium Error Bound (b = 4-8)

| Pattern | max_distance | Traditional | WallBreaker | Speedup |
|---------|--------------|-------------|-------------|---------|
| "algorithm" | 4 | [measure] | [measure] | [measure] |
| "extraordinary" (50+ chars) | 8 | [measure] | [measure] | [measure] |

**Analysis**: Fill in the measured speedup and compare it against the 100x target.

---

### Large Error Bound (b ≥ 12)

| Pattern | max_distance | Traditional | WallBreaker | Speedup |
|---------|--------------|-------------|-------------|---------|
| "extraordinarily" (100 chars) | 16 | [measure] | [measure] | [measure] |

**Analysis**: Fill in the measured speedup and compare it against the 1000x target.

---

## Memory Profile

| Phase | Traditional | WallBreaker | Overhead |
|-------|-------------|-------------|----------|
| Construction | [measure] | [measure] | [measure] |
| Query | [measure] | [measure] | [measure] |

---

## Recommendations

[Based on results, any optimization suggestions or notes]
```

### 8.2 Visualization

**Generate Performance Charts**:
```python
import matplotlib.pyplot as plt

# Speedup vs Error Bound
error_bounds = [1, 2, 4, 8, 12, 16]
speedups = [0.8, 0.9, 100, 500, 2000, 3300]  # Example data

plt.plot(error_bounds, speedups, marker='o')
plt.axhline(y=1, color='r', linestyle='--', label='Break-even')
plt.xlabel('Error Bound (max_distance)')
plt.ylabel('Speedup (x)')
plt.title('WallBreaker Speedup vs Error Bound')
plt.yscale('log')
plt.legend()
plt.savefig('speedup_vs_error_bound.png')
```

**Generate Comparison Table**:
```bash
# Generate markdown table from benchmark results
cargo bench -- --output-format bencher | python3 scripts/parse_benchmarks.py
```

---

## 9. Continuous Benchmarking

### 9.1 CI/CD Integration

**GitHub Actions** (`.github/workflows/benchmark.yml`):
```yaml
name: Benchmark

on:
  push:
    branches: [master, feature/wallbreaker]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: |
          RUSTFLAGS="-C target-cpu=native" cargo bench --bench wallbreaker_comparison
      - name: Compare with baseline
        run: |
          cargo bench -- --baseline master
```

### 9.2 Performance Regression Detection

**Threshold**: Fail if performance degrades by >10%

```bash
# Compare current with baseline
cargo bench -- --baseline master > results.txt

# Parse and check for regressions
python3 scripts/check_regression.py results.txt
```

---

## 10. Appendices

### Appendix A: Benchmark Checklist

Before running benchmarks:
- [ ] Hardware configuration documented
- [ ] System in performance mode (CPU scaling disabled)
- [ ] Background processes minimized
- [ ] Benchmark datasets generated
- [ ] Test patterns prepared
- [ ] Baseline measurements taken

During benchmarking:
- [ ] Run multiple iterations (≥10)
- [ ] Monitor for thermal throttling
- [ ] Check for outliers (discard if necessary)
- [ ] Verify correctness in parallel

After benchmarking:
- [ ] Results saved to file
- [ ] Comparison with targets
- [ ] Visualizations generated
- [ ] Report written
- [ ] Results archived

### Appendix B: Useful Commands

```bash
# Full benchmark suite
RUSTFLAGS="-C target-cpu=native" cargo bench

# Specific benchmark
cargo bench --bench wallbreaker_comparison -- "b=16"

# With flamegraph
cargo flamegraph --bench wallbreaker_comparison

# Save baseline
cargo bench -- --save-baseline my-baseline

# Compare baselines
cargo bench -- --baseline baseline1 --baseline baseline2

# Generate report
cargo bench | tee benchmark_results.txt
```

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-06
**Next Steps**: Implement benchmarks during Phase 3 (Testing & Integration)
