# Performance Targets

Benchmarks and optimization goals for the simplification transpiler.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

This document defines performance targets, benchmarking methodology, and optimization strategies for the simplification transpiler.

---

## Performance Targets

### Primary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Latency** | <50ms for typical programs | Interactive use in IDE |
| **Latency P99** | <200ms | Acceptable for batch processing |
| **AST Size Reduction** | 15-30% | Meaningful simplification |
| **Memory Overhead** | <10% of input size | Minimal resource usage |
| **Throughput** | >1000 terms/sec | Batch processing capability |

### Secondary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Cold Start** | <100ms | First invocation |
| **Cache Hit Rate** | >60% | Effectiveness of memoization |
| **Rule Application Rate** | >10000 rules/sec | Core rewriting speed |
| **Verification Overhead** | <20% of total time | Validation cost |

---

## Benchmark Suite

### Benchmark Categories

1. **Micro-benchmarks**: Individual rule application speed
2. **Synthetic benchmarks**: Generated terms of varying size/depth
3. **Real-world benchmarks**: Actual Rholang programs
4. **Stress tests**: Pathological cases

### Synthetic Benchmark Specifications

```rust
/// Generate synthetic benchmark terms
pub struct BenchmarkGenerator {
    /// Random seed for reproducibility
    seed: u64,
}

impl BenchmarkGenerator {
    /// Generate term with specified characteristics
    pub fn generate(&self, spec: BenchSpec) -> Proc {
        match spec {
            BenchSpec::LinearChain(n) => self.linear_chain(n),
            BenchSpec::BinaryTree(depth) => self.binary_tree(depth),
            BenchSpec::ParallelFlat(width) => self.parallel_flat(width),
            BenchSpec::NestedScopes(depth) => self.nested_scopes(depth),
            BenchSpec::MixedWorkload(n) => self.mixed_workload(n),
        }
    }

    /// Linear sequence of operations: op₁; op₂; ...; opₙ
    fn linear_chain(&self, n: usize) -> Proc {
        // Size: O(n), Depth: O(n)
        (0..n).fold(Proc::Nil, |acc, i| {
            Proc::Seq(Box::new(self.random_op(i)), Box::new(acc))
        })
    }

    /// Balanced binary tree of Par nodes
    fn binary_tree(&self, depth: usize) -> Proc {
        // Size: O(2^depth), Depth: O(depth)
        if depth == 0 {
            self.random_leaf()
        } else {
            Proc::Par(
                Box::new(self.binary_tree(depth - 1)),
                Box::new(self.binary_tree(depth - 1)),
            )
        }
    }

    /// Flat parallel composition: P₁ | P₂ | ... | Pₙ
    fn parallel_flat(&self, width: usize) -> Proc {
        // Size: O(width), Depth: O(1)
        let procs: Vec<_> = (0..width).map(|i| self.random_leaf()).collect();
        procs.into_iter().reduce(|a, b| Proc::Par(Box::new(a), Box::new(b))).unwrap_or(Proc::Nil)
    }

    /// Deeply nested scopes: new x₁ in (new x₂ in (... new xₙ in P))
    fn nested_scopes(&self, depth: usize) -> Proc {
        // Size: O(depth), Depth: O(depth)
        (0..depth).fold(self.random_leaf(), |acc, i| {
            Proc::New(Var::new(format!("x{}", i)), Box::new(acc))
        })
    }
}
```

### Benchmark Matrix

| Benchmark | Size (n) | Depth | Expected Time | Expected Reduction |
|-----------|----------|-------|---------------|-------------------|
| linear-100 | 100 | 100 | <5ms | 10-20% |
| linear-1000 | 1000 | 1000 | <20ms | 10-20% |
| linear-10000 | 10000 | 10000 | <100ms | 10-20% |
| tree-10 | 1024 | 10 | <10ms | 20-30% |
| tree-15 | 32768 | 15 | <50ms | 20-30% |
| tree-20 | 1M | 20 | <500ms | 20-30% |
| parallel-100 | 100 | 1 | <2ms | 5-15% |
| parallel-1000 | 1000 | 1 | <10ms | 5-15% |
| scopes-50 | 50 | 50 | <5ms | 30-50% |
| scopes-100 | 100 | 100 | <10ms | 30-50% |
| mixed-1000 | ~1000 | ~20 | <20ms | 15-25% |

---

## Optimization Strategies

### Strategy 1: Rule Indexing

Build index from pattern head to applicable rules:

```rust
pub struct RuleIndex {
    /// Index by constructor at pattern root
    by_constructor: HashMap<Constructor, Vec<RuleId>>,

    /// Index by pattern size (for quick filtering)
    by_size: BTreeMap<usize, Vec<RuleId>>,
}

impl RuleIndex {
    /// Find potentially applicable rules in O(1)
    pub fn lookup(&self, term: &Proc) -> impl Iterator<Item = RuleId> {
        let constructor = term.constructor();
        self.by_constructor.get(&constructor)
            .into_iter()
            .flatten()
            .filter(|id| self.size_compatible(*id, term.size()))
    }
}
```

**Expected speedup**: 5-10x for rule matching

### Strategy 2: Incremental Rewriting

Only re-check rules at positions that changed:

```rust
pub struct IncrementalRewriter {
    /// Positions that need re-checking
    dirty: HashSet<Position>,

    /// Cache of previous matches
    match_cache: HashMap<Position, Vec<RuleId>>,
}

impl IncrementalRewriter {
    pub fn apply_and_mark_dirty(&mut self, rule: &Rule, pos: Position, term: &mut Proc) {
        // Apply rule at position
        self.apply_at(rule, pos, term);

        // Mark position and ancestors as dirty
        let mut p = pos;
        while let Some(parent) = p.parent() {
            self.dirty.insert(p);
            p = parent;
        }

        // Invalidate match cache for dirty positions
        for dirty_pos in &self.dirty {
            self.match_cache.remove(dirty_pos);
        }
    }
}
```

**Expected speedup**: 2-5x for iterative simplification

### Strategy 3: PathMap Memoization

Cache simplified terms for reuse:

```rust
pub struct SimplificationCache {
    /// PathMap-based cache
    cache: PathMap<CachedResult>,

    /// Statistics
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl SimplificationCache {
    pub fn get_or_simplify<F>(&self, term: &Proc, simplify: F) -> Proc
    where
        F: FnOnce(&Proc) -> Proc,
    {
        let key = term.content_hash();

        if let Some(cached) = self.cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return cached.result.clone();
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = simplify(term);
        self.cache.insert(key, CachedResult { result: result.clone() });
        result
    }
}
```

**Expected speedup**: 1.5-3x for programs with repeated subterms

### Strategy 4: Parallel Phase Execution

Run independent analyses in parallel:

```rust
pub async fn parallel_analysis(proc: &Proc) -> AnalysisFacts {
    let (reachable, live, constants) = tokio::join!(
        async { ReachabilityAnalysis::analyze(proc) },
        async { LivenessAnalysis::analyze(proc) },
        async { ConstantPropagation::analyze(proc) },
    );

    AnalysisFacts {
        reachable,
        live,
        constant_values: constants,
        ..Default::default()
    }
}
```

**Expected speedup**: 2-3x for analysis phase on multi-core systems

### Strategy 5: SIMD Pattern Matching

Use SIMD for comparing term hashes:

```rust
#[cfg(target_arch = "x86_64")]
pub fn batch_hash_compare(pattern_hashes: &[u64], term_hash: u64) -> Vec<bool> {
    use std::arch::x86_64::*;

    unsafe {
        let target = _mm256_set1_epi64x(term_hash as i64);
        let mut results = Vec::with_capacity(pattern_hashes.len());

        for chunk in pattern_hashes.chunks(4) {
            let patterns = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi64(patterns, target);
            let mask = _mm256_movemask_epi8(cmp);
            // Extract per-element results from mask
            for i in 0..chunk.len() {
                results.push((mask >> (i * 8)) & 0xFF == 0xFF);
            }
        }

        results
    }
}
```

**Expected speedup**: 2-4x for batch pattern matching

---

## Profiling Methodology

### Tools

- **perf**: CPU profiling on Linux
- **flamegraph**: Visualization of call stacks
- **criterion**: Rust benchmarking framework
- **heaptrack**: Memory allocation profiling

### Profiling Commands

```bash
# CPU profile with perf
perf record -g --call-graph dwarf ./target/release/simplify bench_input.rho
perf report

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Memory profile
heaptrack ./target/release/simplify bench_input.rho
heaptrack_gui heaptrack.simplify.*.gz

# Benchmark with criterion
cargo bench --bench simplification
```

### Key Metrics to Profile

1. **Rule matching time**: How long to find applicable rules
2. **Pattern match time**: MORK transform_multi_multi_ overhead
3. **Substitution time**: Template instantiation cost
4. **Hash computation time**: Content hash for cycle detection
5. **Memory allocation**: Temporary term construction
6. **Cache effectiveness**: PathMap hit/miss ratio

---

## Benchmark Implementation

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplification");

    // Synthetic benchmarks
    let generator = BenchmarkGenerator::new(42);

    for size in [100, 1000, 10000] {
        let term = generator.generate(BenchSpec::LinearChain(size));

        group.bench_with_input(
            BenchmarkId::new("linear", size),
            &term,
            |b, term| {
                let engine = SimplificationEngine::new();
                b.iter(|| {
                    black_box(engine.simplify(term.clone()))
                });
            },
        );
    }

    for depth in [10, 15, 18] {
        let term = generator.generate(BenchSpec::BinaryTree(depth));
        let size = 1 << depth;

        group.bench_with_input(
            BenchmarkId::new("tree", size),
            &term,
            |b, term| {
                let engine = SimplificationEngine::new();
                b.iter(|| {
                    black_box(engine.simplify(term.clone()))
                });
            },
        );
    }

    group.finish();
}

fn bench_rule_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_matching");

    let registry = RuleRegistry::new();
    let term = generate_test_term(1000);

    group.bench_function("find_applicable", |b| {
        b.iter(|| {
            black_box(registry.find_applicable(&term))
        });
    });

    group.bench_function("indexed_lookup", |b| {
        let index = RuleIndex::from_registry(&registry);
        b.iter(|| {
            black_box(index.lookup(&term).collect::<Vec<_>>())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_simplification, bench_rule_matching);
criterion_main!(benches);
```

---

## Performance Regression Testing

```rust
/// Performance regression test
#[test]
fn perf_regression_linear_1000() {
    let generator = BenchmarkGenerator::new(42);
    let term = generator.generate(BenchSpec::LinearChain(1000));

    let engine = SimplificationEngine::new();

    let start = Instant::now();
    let result = engine.simplify(term.clone());
    let elapsed = start.elapsed();

    // Latency target: <20ms
    assert!(
        elapsed < Duration::from_millis(20),
        "Performance regression: took {:?}",
        elapsed
    );

    // Size reduction target: >10%
    let reduction = (term.size() - result.size()) as f64 / term.size() as f64;
    assert!(
        reduction > 0.10,
        "Size reduction too low: {:.1}%",
        reduction * 100.0
    );
}

#[test]
fn perf_regression_tree_15() {
    let generator = BenchmarkGenerator::new(42);
    let term = generator.generate(BenchSpec::BinaryTree(15));

    let engine = SimplificationEngine::new();

    let start = Instant::now();
    let result = engine.simplify(term.clone());
    let elapsed = start.elapsed();

    // Latency target: <50ms for 32K nodes
    assert!(
        elapsed < Duration::from_millis(50),
        "Performance regression: took {:?}",
        elapsed
    );
}
```

---

## Continuous Monitoring

Track performance metrics over time:

```rust
pub struct PerformanceMonitor {
    /// Time series of latency measurements
    latency_history: VecDeque<(Instant, Duration)>,

    /// Moving average window
    window_size: usize,

    /// Alert thresholds
    p50_threshold: Duration,
    p99_threshold: Duration,
}

impl PerformanceMonitor {
    pub fn record(&mut self, latency: Duration) {
        self.latency_history.push_back((Instant::now(), latency));

        // Maintain window
        while self.latency_history.len() > self.window_size {
            self.latency_history.pop_front();
        }

        // Check for regression
        if let Some(p99) = self.percentile(99) {
            if p99 > self.p99_threshold {
                eprintln!("Performance regression detected: P99 = {:?}", p99);
            }
        }
    }

    fn percentile(&self, p: usize) -> Option<Duration> {
        let mut latencies: Vec<_> = self.latency_history.iter()
            .map(|(_, d)| *d)
            .collect();
        latencies.sort();

        let idx = (latencies.len() * p) / 100;
        latencies.get(idx).copied()
    }
}
```

---

## Summary

| Target | Value | Measurement |
|--------|-------|-------------|
| Latency (typical) | <50ms | Criterion benchmark |
| Latency (P99) | <200ms | Percentile tracking |
| AST Size Reduction | 15-30% | Size before/after |
| Memory Overhead | <10% | heaptrack |
| Cache Hit Rate | >60% | PathMap statistics |
| Throughput | >1000 terms/sec | Batch benchmark |

---

## Next Steps

- Implement benchmark suite
- Run baseline measurements
- Apply optimization strategies iteratively
- Establish CI/CD performance gates

---

## Changelog

- **2025-12-06**: Initial performance targets documentation
