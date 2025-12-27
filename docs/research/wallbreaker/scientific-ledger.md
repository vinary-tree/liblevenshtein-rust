# WallBreaker Optimization Scientific Ledger

**Created**: 2025-12-27
**Purpose**: Track empirical results for WallBreaker optimizations with statistical rigor

## Methodology

### Statistical Requirements
- **Sample size**: Minimum 30 benchmark iterations per configuration
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

### Test Matrix

| Dict Size | Max Distance | Query Length | Notes |
|-----------|--------------|--------------|-------|
| 1,000     | 2, 4         | 10, 20       | Small dictionary baseline |
| 10,000    | 2, 4, 8      | 20, 50       | Medium dictionary |
| 100,000   | 2, 4, 8, 16  | 20, 50, 100  | Large dictionary, wall effect regime |

### Command
```bash
cargo bench --bench wallbreaker_benchmarks -- --save-baseline wallbreaker-baseline
```

### Results

#### WallBreaker Query Performance
| Dict Size | Distance | Query Len | Mean (ns) | Std Dev (ns) | 95% CI Lower | 95% CI Upper |
|-----------|----------|-----------|-----------|--------------|--------------|--------------|
| TBD       | TBD      | TBD       | TBD       | TBD          | TBD          | TBD          |

#### Traditional Transducer Performance (Comparison)
| Dict Size | Distance | Query Len | Mean (ns) | Std Dev (ns) | 95% CI Lower | 95% CI Upper |
|-----------|----------|-----------|-----------|--------------|--------------|--------------|
| TBD       | TBD      | TBD       | TBD       | TBD          | TBD          | TBD          |

#### SCDAWG Construction Time
| Dict Size | Mean (ms) | Std Dev (ms) |
|-----------|-----------|--------------|
| TBD       | TBD       | TBD          |

### Observations
- TBD

---

## Experiment 2: Suffix Link Substring Search Optimization

**Date**: TBD
**Branch**: `feat/wallbreaker-substring-opt`
**Baseline**: Experiment 1 results

### Hypothesis
- **H₀**: Suffix link-based substring search provides no performance improvement over naive O(n*m) search
- **H₁**: Suffix link-based search reduces substring search time by >30% for patterns >10 chars

### Acceptance Criteria
- p < 0.05 improvement over baseline
- >30% reduction in substring search time for patterns >10 chars
- All existing tests pass

### Command
```bash
cargo bench --bench wallbreaker_benchmarks -- --baseline wallbreaker-baseline
```

### Results

| Configuration | Baseline (ns) | Optimized (ns) | Δ% | p-value | Cohen's d | Significant? |
|---------------|---------------|----------------|-----|---------|-----------|--------------|
| TBD           | TBD           | TBD            | TBD | TBD     | TBD       | TBD          |

### Conclusion
**Decision**: TBD (ACCEPTED/REJECTED)
**Rationale**: TBD

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
|---------------|---------------|----------------|-----|---------|-----------|--------------|
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
|---------------|---------------|-----------|-----|---------|-----------|--------------|
| TBD           | TBD           | TBD       | TBD | TBD     | TBD       | TBD          |

### Conclusion
**Decision**: TBD (ACCEPTED/REJECTED)
**Rationale**: TBD

---

## Summary of Decisions

| Experiment | Branch | Decision | Key Metric | Notes |
|------------|--------|----------|------------|-------|
| Baseline   | feat/wallbreaker-benchmarks | N/A | Baseline established | |
| Suffix Links | feat/wallbreaker-substring-opt | TBD | TBD | |
| Freq Split | feat/wallbreaker-freq-split | TBD | TBD | |
| SIMD | feat/wallbreaker-simd | TBD | TBD | |
