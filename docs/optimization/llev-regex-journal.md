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

**Date**: 2025-12-18
**Branch**: `opt/baseline`
**Commit**: `5a470df9c341ed89b5fd32b8bbbbb4b06fcd089f`

### LLev Parsing (load_file)

| Rule File | Mean | 95% CI | Throughput |
|-----------|------|--------|------------|
| zompist.llev (9.3KB, 62 rules) | 143.59 µs | [143.59, 144.79] µs | 67.4 MiB/s |
| homophones.llev (4.8KB, 43 rules) | 135.61 µs | [135.61, 136.79] µs | 36.7 MiB/s |
| text_speak.llev (5.9KB, 60 rules) | 198.53 µs | [198.53, 200.85] µs | 29.6 MiB/s |

### RuleSet Construction (from_llev)

| Rule File | Mean | 95% CI | Throughput |
|-----------|------|--------|------------|
| zompist.llev | 16.92 µs | [16.92, 17.12] µs | 3.66 Melem/s |
| homophones.llev | 20.20 µs | [20.20, 20.66] µs | 2.11 Melem/s |
| text_speak.llev | 30.51 µs | [30.51, 30.96] µs | 1.94 Melem/s |

### Cold Start (end-to-end parse + compile)

| Rule File | Mean | 95% CI |
|-----------|------|--------|
| zompist.llev | 173.83 µs | [173.83, 176.14] µs |
| homophones.llev | 164.29 µs | [164.29, 168.67] µs |
| text_speak.llev | 250.66 µs | [250.66, 253.92] µs |

### Small Parse Throughput

| Benchmark | Mean | Throughput |
|-----------|------|------------|
| 100 simple rules | 173.49 µs | 20.2 MiB/s |

### Lexer Tokenization

| Rule File | Mean | Throughput |
|-----------|------|------------|
| zompist.llev | 42.89 µs | 218.1 MiB/s |
| homophones.llev | 30.50 µs | 158.1 MiB/s |
| text_speak.llev | 53.26 µs | 111.7 MiB/s |

### Perf Top Symbols

```
TBD - Run perf record to identify hot paths
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

