# LLev and Fuzzy Regex Optimization Journal

Scientific journal tracking optimization experiments for LLev (phonetic rewrite rules) and fuzzy regex/NFA components.

## Experiment Tracking

| ID | Hypothesis | Status | Branch | Effect Size | p-value |
|----|------------|--------|--------|-------------|---------|
| H1 | Intern phonetic class names | PENDING | opt/llev-h1-intern-class-names | - | - |
| H2 | Named class lookup optimization | PENDING | opt/llev-h2-named-class-lookup | - | - |
| H3 | Symbol table FxHashMap | PENDING | opt/llev-h3-symbol-table | - | - |
| H4 | SmallVec for character classes | PENDING | opt/llev-h4-smallvec-charclass | - | - |
| H5 | Precomputed epsilon closure | PENDING | opt/nfa-h5-precomputed-epsilon | - | - |
| H6 | CharClass bitmap acceleration | PENDING | opt/nfa-h6-charclass-bitmap | - | - |
| H7 | Bitset state representation | PENDING | opt/nfa-h7-bitset-states | - | - |
| H8 | Lazy DFA cache key optimization | PENDING | opt/nfa-h8-lazydfa-cache | - | - |
| H9 | Transition table restructuring | PENDING | opt/nfa-h9-transition-table | - | - |

---

## Baseline Measurements

**Date**: TBD
**Branch**: `opt/baseline`
**Commit**: TBD

### LLev Parsing

| Rule File | Mean | Std Dev | 95% CI | p95 |
|-----------|------|---------|--------|-----|
| zompist.llev | TBD | TBD | TBD | TBD |
| homophones.llev | TBD | TBD | TBD | TBD |
| text_speak.llev | TBD | TBD | TBD | TBD |

### NFA Construction

| Pattern | Mean | Std Dev | 95% CI | States |
|---------|------|---------|--------|--------|
| TBD | TBD | TBD | TBD | TBD |

### Pattern Matching Throughput

| Document Size | Mean (MB/s) | Std Dev | 95% CI |
|---------------|-------------|---------|--------|
| TBD | TBD | TBD | TBD |

### Perf Top Symbols

```
TBD - Run perf record and populate
```

---

## Experiment Template

```markdown
## Experiment: H[N] - [Name]

### Date: YYYY-MM-DD
### Branch: opt/[component]-h[N]-[name]

### Hypothesis
**H0 (Null)**: [Null hypothesis]
**H1 (Alternative)**: [Alternative hypothesis]

### Implementation Details
- Files modified: [list]
- Lines changed: +X/-Y
- Key changes: [description]

### Baseline Results
- Benchmark: [name]
- Mean: X.XX ms +/- Y.YY ms (95% CI)
- Median: X.XX ms
- p95: X.XX ms

### Post-Optimization Results
- Mean: X.XX ms +/- Y.YY ms (95% CI)
- Median: X.XX ms
- p95: X.XX ms

### Statistical Analysis
- Improvement: XX.X%
- t-statistic: X.XX
- p-value: 0.XXXX
- Effect size (Cohen's d): X.XX ([interpretation])

### Decision
- [x] ACCEPTED - p < 0.05, improvement > 5%
- [ ] REJECTED - p >= 0.05 or regression
- [ ] DEFERRED - marginal improvement

### Flamegraph Comparison
- Before: `artifacts/flamegraph_baseline_[name].svg`
- After: `artifacts/flamegraph_h[N]_[name].svg`

### Notes
[Observations, lessons learned, follow-up ideas]
```

---

## Detailed Experiment Records

(Individual experiments will be added below as they are conducted)

