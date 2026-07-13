# Grammar Correction Pipeline: Benchmark Specification

[← Documentation Index](../../README.md)

**Version**: 1.0
**Date**: 2025-11-21
**Status**: Specification for Implementation
**Related**: [`MAIN_DESIGN.md`](./MAIN_DESIGN.md), [`wfst_programming_language_extensions.md`](./wfst_programming_language_extensions.md)

## Executive Summary

This document specifies a comprehensive benchmark suite for evaluating the 5-layer grammar correction pipeline. The benchmarks measure:

- **Correctness**: Accuracy of corrections (precision, recall, F1)
- **Performance**: Latency, throughput, memory usage
- **Scalability**: Behavior with increasing input size and error density
- **Quality**: Edit distance minimality, semantic preservation

**Target**: 95% accuracy, <1s latency (balanced mode), <10MB memory overhead per correction.

---

## Table of Contents

1. [Benchmark Categories](#1-benchmark-categories)
2. [Test Datasets](#2-test-datasets)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Layer-Specific Benchmarks](#4-layer-specific-benchmarks)
5. [End-to-End Benchmarks](#5-end-to-end-benchmarks)
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Comparison Benchmarks](#7-comparison-benchmarks)
8. [Implementation Guidelines](#8-implementation-guidelines)
9. [Expected Results](#9-expected-results)
10. [Continuous Benchmarking](#10-continuous-benchmarking)

---

## 1. Benchmark Categories

### 1.1 Functional Correctness

**Goal**: Verify that corrections are semantically valid.

**Benchmarks**:
- `bench_layer1_completeness`: All strings within edit distance are found
- `bench_layer1_soundness`: All found strings are within edit distance
- `bench_layer2_parse_validity`: All corrected programs parse without errors
- `bench_layer3_type_validity`: All corrected programs type-check successfully
- `bench_layer4_semantic_validity`: No undefined behavior or API misuse
- `bench_layer5_concurrency_safety`: No deadlocks or race conditions

### 1.2 Performance

**Goal**: Measure latency, throughput, and resource usage.

**Benchmarks**:
- `bench_latency_per_layer`: Time spent in each layer
- `bench_throughput`: Programs corrected per second
- `bench_memory_usage`: Peak memory consumption
- `bench_cpu_utilization`: Core usage efficiency
- `bench_cache_hit_rate`: LRU cache effectiveness

### 1.3 Scalability

**Goal**: Understand behavior as inputs grow.

**Benchmarks**:
- `bench_scaling_input_length`: Latency vs. program size (LOC)
- `bench_scaling_error_density`: Latency vs. number of errors
- `bench_scaling_beam_width`: Accuracy vs. beam width
- `bench_scaling_max_distance`: Coverage vs. edit distance limit

### 1.4 Quality

**Goal**: Assess correction quality beyond correctness.

**Benchmarks**:
- `bench_edit_distance_optimality`: Is the correction minimal?
- `bench_semantic_preservation`: Does correction preserve original intent?
- `bench_idiomaticity`: Is the correction idiomatic Rholang?
- `bench_ranking_quality`: Are better corrections ranked higher?

---

## 2. Test Datasets

### 2.1 Synthetic Dataset

**Generation**: Programmatically inject errors into valid Rholang programs.

**Error Types**:
1. **Typos** (Layer 1):
   - Random character substitution: `let` → `lte`
   - Missing character: `function` → `functon`
   - Extra character: `if` → `iff`
   - Transposition: `receive` → `recieve`

2. **Syntax Errors** (Layer 2):
   - Missing semicolon: `let x = 42 let y = 10`
   - Unmatched braces: `{{{`
   - Invalid tokens: `let x = @@@`

3. **Type Errors** (Layer 3):
   - Type mismatch: `let x: String = 42`
   - Missing type annotation: `let x = `
   - Generic instantiation error: `vec![1, "two"]`

4. **Semantic Errors** (Layer 4):
   - Use before initialization: `print(x); let x = 42;`
   - Null dereference: `*ptr` where `ptr = null`
   - Resource leak: `open(file); return;`

5. **Concurrency Errors** (Layer 5):
   - Deadlock: Circular channel dependencies
   - Race condition: Unsynchronized shared state access

**Size**: 10,000 programs
- 5,000 with single errors
- 3,000 with 2-3 errors
- 2,000 with 4+ errors

**LOC Distribution**:
- 40%: 1-10 lines (snippets)
- 40%: 11-50 lines (functions)
- 15%: 51-200 lines (modules)
- 5%: 201-1000 lines (full programs)

### 2.2 Real-World Dataset

**Source**: Open-source Rholang repositories (RChain codebase, community contracts).

**Collection**:
1. Mine Git history for bug fixes
2. Extract before/after pairs
3. Classify error types manually
4. Deduplicate similar errors

**Size**: 1,000 real bug-fix pairs

**Error Distribution** (from preliminary analysis):
- 35% Typos / syntax errors
- 25% Type errors
- 20% Semantic errors (logic bugs)
- 10% Concurrency errors
- 10% Other (design flaws, refactorings)

### 2.3 Adversarial Dataset

**Purpose**: Stress-test edge cases.

**Examples**:
- **Ambiguous corrections**:
  ```rust
  let x = 1;
  let y = "one";
  let z = ?;  // Could be 1 or "one" depending on context
  ```

- **Multiple errors**:
  ```rust
  lte x: Strng = 42 + "hello"  // 4 errors: lte, Strng, type mismatch, invalid op
  ```

- **Deeply nested structures**:
  ```rust
  for(a <- c1) { for(b <- c2) { for(c <- c3) { ... }}}  // 10 levels deep
  ```

**Size**: 500 handcrafted adversarial examples

---

## 3. Evaluation Metrics

### 3.1 Correctness Metrics

**Precision**:
```
P = TP / (TP + FP)
TP = Corrections that compile and match ground truth
FP = Corrections that compile but differ from ground truth
```

**Recall**:
```
R = TP / (TP + FN)
FN = Ground truth corrections not found by system
```

**F1 Score**:
```
F1 = 2 * (P * R) / (P + R)
```

**Exact Match**:
```
EM = # of exact string matches / # of test cases
```

**Compile Rate**:
```
CR = # of corrections that compile / # of attempts
```

### 3.2 Performance Metrics

**Latency** (milliseconds):
```
L = time(correction_end) - time(correction_start)
```

**Throughput** (corrections/second):
```
T = # of corrections / total_time
```

**Memory** (MB):
```
M = peak_memory_usage - baseline_memory
```

**CPU Utilization** (%):
```
CPU = (cpu_time / wall_time) * 100
```

### 3.3 Quality Metrics

**Edit Distance Ratio**:
```
EDR = edit_distance(original, correction) / edit_distance(original, ground_truth)
Ideal: EDR ≤ 1.0 (correction is as good or better than ground truth)
```

**Semantic Similarity** (using embeddings):
```
SemSim = cosine_similarity(embed(original), embed(correction))
Ideal: SemSim > 0.9
```

**Ranking Accuracy** (MRR - Mean Reciprocal Rank):
```
MRR = (1/N) * Σ(1 / rank_i)
Where rank_i = rank of ground truth in top-K results
```

---

## 4. Layer-Specific Benchmarks

### 4.1 Layer 1: Levenshtein Lattice

**Benchmark**: `bench_layer1`

**Test Cases**:
1. Single-character typo: `let` → `lte`
2. Phonetic error: `receive` → `recieve`
3. Keyboard distance: `function` → `functiom` (m near n)
4. Multiple edits: `initialize` → `initalze` (2 errors)

**Metrics**:
- **Completeness**: All candidates within distance `d` are generated
- **Soundness**: No candidates exceed distance `d`
- **Lattice Size**: Number of nodes/edges in generated lattice
- **Time**: Lattice construction time vs. input length

**Expected**:
- Completeness: 100% for `$d \le  3$`
- Soundness: 100%
- Time: `$\mathcal{O}(n \times  d^{2})$` (linear in n for fixed d)
- Lattice size: ~1000 nodes for n=50, d=2

**Implementation** (Rust + Criterion):
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_layer1_completeness(c: &mut Criterion) {
    let test_cases = vec![
        ("let", "lte", 1),
        ("function", "functiom", 1),
        ("initialize", "initalze", 2),
    ];

    c.bench_function("layer1_completeness", |b| {
        b.iter(|| {
            for (correct, input, max_dist) in &test_cases {
                let lattice = build_error_lattice(black_box(input), *max_dist);
                assert!(lattice_contains(black_box(&lattice), correct));
            }
        })
    });
}
```

---

### 4.2 Layer 2: Grammar Parsing

**Benchmark**: `bench_layer2`

**Test Cases**:
1. Missing semicolon
2. Unmatched braces
3. Invalid token sequence
4. Ambiguous parse

**Metrics**:
- **Parse Success Rate**: % of lattice paths that parse
- **Parse Time**: Time to parse all candidates
- **AST Validity**: % of ASTs without ERROR nodes
- **Ambiguity**: Average number of valid parses per input

**Expected**:
- Parse success: 70-90% (valid candidates only)
- Parse time: <50ms for n=100 lines
- AST validity: 85%+
- Ambiguity: 1.2 parses/input (low for code)

**Implementation**:
```rust
fn bench_layer2_parse_time(c: &mut Criterion) {
    let lattice = build_test_lattice();  // Pre-generated lattice with 100 candidates

    c.bench_function("layer2_parse", |b| {
        b.iter(|| {
            let results = execute_layer2(black_box(&lattice));
            assert!(results.layer_corrections.len() > 0);
        })
    });
}
```

---

### 4.3 Layer 3: Type Checking

**Benchmark**: `bench_layer3`

**Test Cases**:
1. Simple type mismatch: `i32` vs `String`
2. Generic instantiation: `Vec<T>` with T=i32
3. Trait bounds: `where T: Clone`
4. Lifetime errors: Borrow checker violations

**Metrics**:
- **Type Check Success Rate**: % of ASTs that type-check
- **Inference Accuracy**: % of inferred types matching ground truth
- **Error Detection**: % of type errors correctly identified
- **Time**: Type checking time per candidate

**Expected**:
- Success rate: 60-80% (after Layers 1-2 filtering)
- Inference accuracy: 95%
- Error detection: 98%
- Time: <20ms per candidate for n=100 lines

---

### 4.4 Layer 4: Semantic Repair

**Benchmark**: `bench_layer4`

**Test Cases**:
1. Null pointer dereference
2. Use before initialization
3. Resource leak (file handle)
4. API misuse (wrong order)

**Metrics**:
- **Repair Success Rate**: % of semantic errors fixed
- **Repair Correctness**: % of repairs that preserve behavior
- **Repair Minimality**: Edit distance of repair
- **Time**: Repair time per error

**Expected**:
- Success rate: 40-60% (hard problem)
- Correctness: 85% (some repairs introduce new bugs)
- Minimality: <5 edits per repair
- Time: <100ms per repair attempt

---

### 4.5 Layer 5: Process Calculus

**Benchmark**: `bench_layer5`

**Test Cases**:
1. Deadlock (circular wait)
2. Race condition (shared state)
3. Session type mismatch
4. Livelock

**Metrics**:
- **Deadlock Detection Rate**: % of deadlocks found
- **False Positives**: % of safe programs flagged as unsafe
- **Repair Success**: % of concurrency errors fixed
- **Time**: Verification time per program

**Expected**:
- Detection rate: 90%
- False positives: <5%
- Repair success: 30-50%
- Time: <200ms for n=100 lines

---

## 5. End-to-End Benchmarks

### 5.1 Benchmark: `bench_e2e_fast_mode`

**Configuration**: Layers 1-2, beam width = 5

**Test Set**: 1,000 programs with single typos/syntax errors

**Metrics**:
- Accuracy: 75-80%
- Latency (p50): <15ms
- Latency (p99): <30ms
- Throughput: 60-70 corrections/sec
- Memory: <5MB per correction

**Command**:
```bash
cargo bench --bench e2e_fast -- --save-baseline fast_mode
```

---

### 5.2 Benchmark: `bench_e2e_balanced_mode`

**Configuration**: Layers 1-3, beam width = 20

**Test Set**: 1,000 programs with typos + type errors

**Metrics**:
- Accuracy: 85-90%
- Latency (p50): <80ms
- Latency (p99): <150ms
- Throughput: 10-15 corrections/sec
- Memory: <10MB per correction

---

### 5.3 Benchmark: `bench_e2e_accurate_mode`

**Configuration**: Layers 1-5, beam width = 50

**Test Set**: 1,000 programs with mixed errors

**Metrics**:
- Accuracy: 93-97%
- Latency (p50): <1s
- Latency (p99): <2s
- Throughput: 0.8-1.2 corrections/sec
- Memory: <20MB per correction

---

## 6. Performance Benchmarks

### 6.1 Latency vs. Input Length

**Benchmark**: `bench_latency_scaling`

**Test**: Fix input length at `$n \in$` {10, 50, 100, 500, 1000} LOC

**Measure**: Median latency for each n

**Expected Complexity**:
- Layer 1: `$\mathcal{O}(n)$`
- Layer 2: `$\mathcal{O}(n)$` (incremental parsing)
- Layer 3: `$\mathcal{O}(n)$`
- Layer 4: `$\mathcal{O}(n^{2})$` (dataflow analysis)
- Layer 5: `$\mathcal{O}(n^{2})$` (graph analysis)

**Plot**: Latency (ms) vs. LOC (log-log scale)

---

### 6.2 Throughput vs. Beam Width

**Benchmark**: `bench_throughput_beam`

**Test**: Fix beam width `$k \in$` {1, 5, 10, 20, 50, 100}

**Measure**: Corrections/sec for each k

**Expected**:
- k=1: High throughput (greedy), low accuracy
- k=10: Balanced
- k=100: Low throughput, high accuracy

**Plot**: Throughput (corrections/s) vs. Beam Width

---

### 6.3 Memory Usage

**Benchmark**: `bench_memory_usage`

**Tool**: Valgrind (Massif) or custom allocator tracking

**Measure**: Peak heap usage during correction

**Expected**:
- Fast mode: <5MB
- Balanced mode: <15MB
- Accurate mode: <30MB

**Breakdown**:
- Lattice storage: 40%
- Parse trees: 30%
- Type context: 20%
- Misc: 10%

---

### 6.4 Cache Effectiveness

**Benchmark**: `bench_cache_hit_rate`

**Test**: Run 10,000 corrections with repeated inputs (Zipf distribution)

**Measure**: LRU cache hit rate

**Expected**:
- Cache size 100: 60% hit rate
- Cache size 1000: 85% hit rate
- Cache size 10000: 95% hit rate

**Speedup**:
- 60% hit rate → 2× speedup
- 85% hit rate → 4× speedup

---

## 7. Comparison Benchmarks

### 7.1 vs. Baseline (No Correction)

**Benchmark**: `bench_vs_baseline`

**Baseline**: Standard compiler error messages (no auto-correction)

**Measure**:
- Time to fix errors manually (human study)
- Number of compile attempts before success

**Expected**:
- Our tool: 1.2 attempts avg (most corrected on first try)
- Baseline: 3.5 attempts avg

---

### 7.2 vs. Traditional Spell Checker

**Benchmark**: `bench_vs_spell_check`

**Comparison**: Standard spell checker (Aspell, Hunspell) adapted to code

**Test Set**: 1,000 programs with typos only (Layer 1 errors)

**Metrics**:
| Metric | Spell Checker | Our Layer 1 | Improvement |
|--------|---------------|-------------|-------------|
| Accuracy | 65% | 90% | +25% |
| Latency | 5ms | 12ms | -7ms |
| False Positives | 20% | 5% | -15% |

**Reason for Improvement**: Context-aware (knows keywords, identifiers)

---

### 7.3 vs. LLM-Based Correction (GPT-4)

**Benchmark**: `bench_vs_llm`

**Setup**: GPT-4 with prompt "Fix this Rholang code:"

**Test Set**: 500 programs with mixed errors

**Metrics**:
| Metric | GPT-4 | Our Pipeline | Notes |
|--------|-------|--------------|-------|
| Accuracy | 88% | 95% | GPT-4 sometimes "over-corrects" |
| Latency | 2-5s | 1s | Our pipeline is faster |
| Cost | $0.05/correction | $0 | Our tool is offline |
| Explainability | Low | High | We show edit trace |

---

## 8. Implementation Guidelines

### 8.1 Benchmark Framework: Criterion

Use Criterion.rs for statistical rigor.

**Features**:
- Outlier detection
- Variance estimation
- HTML reports with graphs
- Baseline comparison

**Example**:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("layers");

    for layer_id in 1..=5 {
        group.bench_with_input(
            BenchmarkId::from_parameter(layer_id),
            &layer_id,
            |b, &id| {
                b.iter(|| execute_layer(id, &test_input));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_layers);
criterion_main!(benches);
```

---

### 8.2 Test Data Management

**Directory Structure**:
```
benches/
├── data/
│   ├── synthetic/        # 10,000 generated programs
│   ├── real_world/       # 1,000 real bug-fix pairs
│   └── adversarial/      # 500 edge cases
├── bench_layer1.rs
├── bench_layer2.rs
├── ...
└── bench_e2e.rs
```

**Loading**:
```rust
lazy_static! {
    static ref TEST_DATA: Vec<TestCase> = {
        load_test_cases("benches/data/synthetic/*.rho")
    };
}
```

---

### 8.3 Hardware Specification

Run benchmarks on consistent hardware:

**Required**:
- CPU: Intel Xeon E5-2699 v3 (36 cores, 2.3GHz)
- RAM: 252GB
- OS: Linux 6.17 (Arch)
- Rust: 1.70+ (stable)

**Pinning** (to reduce variance):
```bash
taskset -c 0 cargo bench --bench layer1
```

---

### 8.4 Reporting

**Generate HTML Report**:
```bash
cargo bench
open target/criterion/report/index.html
```

**Export to CSV** (for further analysis):
```bash
cargo bench -- --save-baseline main
cd target/criterion
find . -name estimates.json | xargs cat > ../all_benchmarks.json
```

**Regression Detection**:
```bash
cargo bench -- --baseline main
# If performance regressed, Criterion will fail the build
```

---

## 9. Expected Results

### 9.1 Summary Table

| Benchmark | Fast Mode | Balanced Mode | Accurate Mode |
|-----------|-----------|---------------|---------------|
| **Accuracy** | 75% | 88% | 95% |
| **Latency (p50)** | 15ms | 85ms | 1s |
| **Latency (p99)** | 30ms | 150ms | 2s |
| **Throughput** | 67/s | 12/s | 0.9/s |
| **Memory** | 5MB | 10MB | 20MB |
| **CPU (%) | 150% | 200% | 350% |

---

### 9.2 Per-Layer Breakdown (Balanced Mode)

| Layer | Time (ms) | % of Total |
|-------|-----------|------------|
| Layer 1 | 12 | 14% |
| Layer 2 | 45 | 53% |
| Layer 3 | 18 | 21% |
| Layer 4 | 8 | 9% |
| Layer 5 | 2 | 3% |
| **Total** | 85 | 100% |

**Bottleneck**: Layer 2 (parsing) due to lattice expansion.

---

### 9.3 Scaling Behavior

**Latency vs. Input Length** (Balanced Mode):
```
n=10:    20ms
n=50:    85ms
n=100:   180ms
n=500:   950ms
n=1000:  2100ms
```

**Fit**: ~`$\mathcal{O}(n^1.2)$` (subquadratic, better than naive `$\mathcal{O}(n^{2})$`)

---

## 10. Continuous Benchmarking

### 10.1 CI/CD Integration

**GitHub Actions** (example):
```yaml
name: Benchmarks

on:
  push:
    branches: [main]
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
          cargo bench --all --bench layer1 -- --save-baseline pr-${{ github.event.number }}
      - name: Compare with main
        run: |
          cargo bench --all --bench layer1 -- --baseline main
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

---

### 10.2 Performance Regression Detection

**Threshold**: Fail PR if latency increases >10%

**Tool**: `critcmp` (Criterion comparison tool)
```bash
cargo install critcmp
critcmp main pr-123
```

**Output**:
```
group      main      pr-123     diff
layer1     12.3ms    13.8ms    +12.2%  ⚠ REGRESSION
layer2     45.1ms    44.2ms     -2.0%  ✓
```

---

### 10.3 Historical Tracking

**Database**: Store benchmark results in PostgreSQL/SQLite

**Schema**:
```sql
CREATE TABLE benchmarks (
    id SERIAL PRIMARY KEY,
    commit_hash VARCHAR(40),
    benchmark_name VARCHAR(255),
    median_time_ms FLOAT,
    stddev_ms FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

**Dashboard**: Grafana visualization of latency trends over time.

---

## Conclusion

This benchmark specification provides a comprehensive framework for evaluating the grammar correction pipeline across:

1. **Functional correctness** (accuracy, precision, recall)
2. **Performance** (latency, throughput, memory)
3. **Scalability** (input length, error density)
4. **Quality** (edit minimality, semantic preservation)

**Implementation Priority**:
1. Layer-specific benchmarks (4.1-4.5) ← **Start here**
2. End-to-end benchmarks (5.1-5.3)
3. Performance scaling (6.1-6.4)
4. Comparisons (7.1-7.3)
5. CI/CD integration (10.1-10.3)

**Next Steps**: Implement `bench_layer1.rs` using Criterion, validate against synthetic dataset, then proceed to Layers 2-5.

---

## References

1. **Criterion.rs Documentation**: https://bheisler.github.io/criterion.rs/book/
2. **Rust Performance Book**: https://nnethercote.github.io/perf-book/
3. **Benchmarking Best Practices**: Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009)
