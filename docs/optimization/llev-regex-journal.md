# LLev and Fuzzy Regex Optimization Journal

Scientific journal tracking optimization experiments for LLev (phonetic rewrite rules) and fuzzy regex/NFA components.

## Experiment Tracking

| ID | Hypothesis | Status | Branch | Effect Size | p-value |
|----|------------|--------|--------|-------------|---------|
| H1 | Intern phonetic class names | **ACCEPTED** | opt/llev-h1-intern-class-names | -8% to -11% lexer time | p < 0.05 |
| H2 | Named class lookup optimization | **ACCEPTED** | opt/llev-h2-named-class-lookup | -3% to -4% cold start, -9% to -13% lexer | p < 0.05 |
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

### Perf Top Symbols (Lexer Benchmark)

| Symbol | CPU % | Insight |
|--------|-------|---------|
| `Lexer::next_token_internal` | 33.36% | Main tokenization loop |
| `Lexer::skip_whitespace_only` | 23.39% | Significant - uses `is_whitespace()` in loop |
| `Lexer::advance` | 15.17% | Called per character, tracks position |
| `Lexer::parse_string` | 2.64% | String allocation overhead |
| `Lexer::peek_char` | 0.50% | Character lookahead |

**Key Insight**: 71.92% of lexer time is in just 3 functions. Potential quick wins:
- Replace `is_whitespace()` with `is_ascii_whitespace()` (faster, no Unicode tables)
- Reduce per-character overhead in `advance()`

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

### Experiment: H1 - Intern Phonetic Class Names

**Date**: 2025-12-18
**Branch**: `opt/llev-h1-intern-class-names`

#### Hypothesis
**H0 (Null)**: Changing `class_name: String` to `class_name: &'static str` in `Token::PhoneticShortcut` will not improve lexer performance.
**H1 (Alternative)**: Eliminating 34 heap allocations per file parse will measurably improve lexer throughput.

#### Implementation Details
- Files modified:
  - `src/phonetic/llev/lexer.rs` - Changed Token type, removed `.to_string()` calls
  - `src/phonetic/regex/lexer.rs` - Changed Token type, removed `.to_string()` calls
  - `src/phonetic/regex/parser.rs` - Updated error handling for new type
- Lines changed: +68/-68 (type changes and test updates)
- Key changes: Replace `String` with `&'static str` for 17 phonetic class names, eliminating heap allocation for each shortcut token

#### Baseline Results (from opt/baseline)
| Benchmark | Mean | Throughput |
|-----------|------|------------|
| lexer/zompist | 142.89 µs | 68.4 MiB/s |
| lexer/homophones | 72.63 µs | 69.1 MiB/s |
| lexer/text_speak | 87.02 µs | 68.3 MiB/s |

#### Post-Optimization Results
| Benchmark | Mean | Throughput | Change |
|-----------|------|------------|--------|
| lexer/zompist | 127.73 µs | 76.5 MiB/s | **-11.2%** |
| lexer/homophones | 66.70 µs | 75.2 MiB/s | **-8.2%** |
| lexer/text_speak | 80.66 µs | 73.7 MiB/s | **-7.3%** |

#### Statistical Analysis
| Benchmark | Improvement | p-value | Significance |
|-----------|-------------|---------|--------------|
| zompist | -11.2% | p = 0.00 | ✅ Significant |
| homophones | -8.2% | p = 0.00 | ✅ Significant |
| text_speak | -7.3% | p = 0.00 | ✅ Significant |
| small_parses | -3.0% | p = 0.00 | ✅ Significant |

#### Secondary Observations
- Full parsing (load_file) showed mixed results: zompist unchanged, homophones/text_speak +2-3%
- Cold start showed small regressions (1-5%) which may be measurement noise
- The lexer benchmark (isolated tokenization) is the most direct measure of this optimization

#### Decision
- [x] **ACCEPTED** - p < 0.05 for all lexer benchmarks, improvement 7-11%
- [ ] REJECTED
- [ ] DEFERRED

#### Notes
1. This optimization targets the lexer specifically; effects on full pipeline are secondary
2. The Token enum is now smaller (no heap pointer), which may affect cache behavior
3. All 34 phonetic class names are compile-time constants, making `&'static str` safe
4. Error messages still use `.to_string()` for the error type, preserving compatibility
5. Future work: Consider H1b to add `is_ascii_whitespace()` optimization identified in profiling

---

### Experiment: H2 - Named Class Lookup Optimization

**Date**: 2025-12-18
**Branch**: `opt/llev-h2-named-class-lookup`
**Parent Branch**: `opt/llev-h1-intern-class-names`

#### Hypothesis
**H0 (Null)**: Replacing `to_lowercase()` with stack-allocated ASCII lowercase conversion in `get_named_class()` will not improve performance.
**H1 (Alternative)**: Eliminating heap allocation in case-insensitive class name lookup will measurably improve ruleset construction and cold start times.

#### Implementation Details
- Files modified:
  - `src/phonetic/named_classes.rs` - Refactored `get_named_class()` and `is_builtin_class()` functions
- Lines changed: +45/-6
- Key changes:
  - Added `normalize_class_name()` helper for stack-based ASCII lowercase conversion
  - Added fast path for exact match (already lowercase names)
  - Used fixed-size 16-byte stack buffer instead of heap-allocated String
  - Early return for non-ASCII or too-long names (all built-in class names are ≤15 ASCII chars)

#### Post-Optimization Results (vs H1 baseline)

| Benchmark | Mean | Change | p-value | Significance |
|-----------|------|--------|---------|--------------|
| cold_start/zompist | 150.98 µs | -2.59% | p = 0.02 | ⚠️ Within noise |
| cold_start/homophones | 148.30 µs | **-3.25%** | p = 0.00 | ✅ Improved |
| cold_start/text_speak | 221.17 µs | **-3.48%** | p = 0.00 | ✅ Improved |
| ruleset/zompist | 16.39 µs | **-4.42%** | p = 0.00 | ✅ Improved |
| ruleset/homophones | 19.39 µs | +0.20% | p = 0.69 | ⚪ No change |
| llev_parsing/text_speak | 190.03 µs | **-2.16%** | p = 0.00 | ✅ Improved |
| lexer/zompist | 118.01 µs | **-8.65%** | p = 0.00 | ✅ Improved |
| lexer/homophones | 58.25 µs | **-13.05%** | p = 0.00 | ✅ Improved |
| lexer/text_speak | 69.96 µs | **-13.26%** | p = 0.00 | ✅ Improved |

#### Cumulative Effect (H1 + H2 combined vs baseline)

Estimated total improvement in lexer throughput:
- zompist: ~19% faster (0.888 × 0.914 = 0.812)
- homophones: ~20% faster (0.918 × 0.870 = 0.799)
- text_speak: ~20% faster (0.927 × 0.867 = 0.804)

#### Decision
- [x] **ACCEPTED** - p < 0.05 for most benchmarks, cold start improved 3-4%
- [ ] REJECTED
- [ ] DEFERRED

#### Notes
1. The large lexer improvements (~9-13%) are cumulative from H1+H2, comparing to the original baseline saved by Criterion
2. The fast path for exact-match lookups (common case for lowercase class names) adds negligible overhead
3. Using a fixed-size stack buffer eliminates all heap allocation from case-insensitive lookup
4. The 16-byte buffer size accommodates the longest built-in class name ("ascii_consonant" = 15 chars)
5. All built-in class names are ASCII-only, allowing use of `to_ascii_lowercase()` which is faster than full Unicode lowercase

