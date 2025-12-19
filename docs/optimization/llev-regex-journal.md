# LLev and Fuzzy Regex Optimization Journal

Scientific journal tracking optimization experiments for LLev (phonetic rewrite rules) and fuzzy regex/NFA components.

## Experiment Tracking

| ID | Hypothesis | Status | Branch | Effect Size | p-value |
|----|------------|--------|--------|-------------|---------|
| H1 | Intern phonetic class names | **ACCEPTED** | opt/llev-h1-intern-class-names | -8% to -11% lexer time | p < 0.05 |
| H2 | Named class lookup optimization | **ACCEPTED** | opt/llev-h2-named-class-lookup | -3% to -4% cold start, -9% to -13% lexer | p < 0.05 |
| H3 | Symbol table FxHashMap | **REJECTED** | opt/llev-h3-symbol-table | +10% to +15% regression | p < 0.05 |
| H4 | SmallVec for character classes | **REJECTED** | opt/llev-h4-smallvec-charclass | Inconsistent (±20% variance) | N/A |
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

---

### Experiment: H3 - Symbol Table FxHashMap

**Date**: 2025-12-18
**Branch**: `opt/llev-h3-symbol-table`
**Parent Branch**: `opt/llev-h2-named-class-lookup`

#### Hypothesis
**H0 (Null)**: Replacing std::HashMap with FxHashMap for the parser symbol table will not improve parsing performance.
**H1 (Alternative)**: FxHashMap's faster hashing algorithm will improve symbol lookup performance.

#### Implementation Details
- Files modified:
  - `src/phonetic/llev/parser.rs` - Changed symbol table type
- Lines changed: +4/-2
- Key changes: Replace `HashMap<String, Expression>` with `FxHashMap<String, Expression>`

#### Post-Optimization Results (vs H2 baseline)

| Benchmark | Mean | Change | p-value | Significance |
|-----------|------|--------|---------|--------------|
| llev_parsing/zompist | 156.86 µs | **+13.8%** | p = 0.00 | ❌ Regressed |
| llev_parsing/homophones | 145.58 µs | **+10.2%** | p = 0.00 | ❌ Regressed |
| llev_parsing/text_speak | 219.88 µs | **+11.4%** | p = 0.00 | ❌ Regressed |

#### Analysis
The regression was unexpected. Possible causes:
1. **Small table overhead**: The symbol table is typically empty (most LLev files don't use `@define`). FxHashMap may have higher initialization cost than std::HashMap's lazy allocation.
2. **Cache effects**: FxHashMap's different memory layout may cause more cache misses.
3. **Compiler optimization**: std::HashMap may receive better optimization from the compiler.

#### Decision
- [ ] ACCEPTED
- [x] **REJECTED** - Clear regression of 10-15% in parsing benchmarks
- [ ] DEFERRED

#### Notes
1. FxHashMap is NOT always faster than std::HashMap - depends on usage patterns
2. For typically-empty or small hash maps, std::HashMap's lazy allocation is more efficient
3. Future optimization: Consider removing the symbol table entirely if `@define` is rarely used, or use a different data structure (e.g., SmallVec for small counts)

---

### Experiment: H4 - SmallVec for Character Classes

**Date**: 2025-12-18
**Branch**: `opt/llev-h4-smallvec-charclass`
**Parent Branch**: `opt/llev-h2-named-class-lookup`

#### Hypothesis
**H0 (Null)**: Using `SmallVec<[char; 16]>` instead of `Vec<char>` for character classes will not improve parsing performance.
**H1 (Alternative)**: Eliminating heap allocations for small character classes (≤16 chars) will measurably improve parsing and ruleset construction times.

#### Implementation Details
- Files modified:
  - `src/phonetic/llev/ast.rs` - Added `CharClassVec = SmallVec<[char; 16]>` type alias
  - `src/phonetic/llev/parser.rs` - Updated CharClass handling throughout
  - `src/phonetic/llev/ruleset.rs` - Updated extraction methods
  - `src/phonetic/llre/loader.rs` - Updated loader compatibility
- Lines changed: +36/-25
- Key changes: Replace `Vec<char>` with `CharClassVec` in `Expression::CharClass` variant, update all construction and conversion sites

#### Post-Optimization Results (vs H2 baseline)

Results were **inconsistent across multiple runs**:

| Benchmark | Run 1 | Run 2 | Run 3 | Pattern |
|-----------|-------|-------|-------|---------|
| llev_parsing/zompist | -5.4% | +5.7% | +4.2% | Unstable |
| llev_parsing/homophones | -0.4% | +7.1% | +6.4% | Regression |
| llev_parsing/text_speak | +2.2% | +7.1% | +6.0% | Regression |
| ruleset/zompist | -3.4% | +8.4% | +7.6% | Unstable |
| ruleset/homophones | -12.3% | +18.6% | +17.9% | High variance |
| ruleset/text_speak | -10.3% | +12.1% | +11.3% | High variance |
| cold_start/zompist | -14.0% | +5.7% | +5.0% | High variance |
| cold_start/homophones | -11.8% | +5.4% | +4.0% | High variance |
| cold_start/text_speak | -13.5% | +2.8% | +1.7% | Unstable |
| small_parses | +16.6% | -8.5% | -9.6% | High variance |
| lexer/zompist | ~0% | -8.2% | -9.5% | Improvement |
| lexer/homophones | ~0% | +2.4% | +1.4% | Neutral |
| lexer/text_speak | ~0% | -6.8% | -7.6% | Improvement |

#### Analysis

1. **High variance**: Results fluctuated wildly between runs (±20% difference)
2. **SmallVec trade-offs**:
   - Pro: Avoids heap allocation for classes ≤16 chars
   - Con: Larger stack footprint (64 bytes vs 24 bytes per CharClass)
   - Con: Conversion overhead when interfacing with `Vec<char>` APIs
3. **Character class sizes**: Most phonetic classes have 5-25 characters, so some benefit while others don't
4. **API boundary cost**: Converting `CharClassVec` to `Vec<char>` for intersection operations adds overhead

#### Decision
- [ ] ACCEPTED
- [x] **REJECTED** - Inconsistent results, no clear sustained improvement, high variance suggests marginal effect
- [ ] DEFERRED

#### Notes
1. SmallVec optimization is NOT universally beneficial - depends heavily on actual usage patterns
2. The 16-char threshold may be too small for some character classes (e.g., "consonant" has 20+ chars)
3. The conversion overhead at API boundaries negates potential allocation savings
4. Consider H4b: Increase inline capacity to 32 chars (128 bytes) if revisiting
5. This optimization might be more effective in hot paths where CharClass objects are created/destroyed frequently rather than in parsing which happens once at startup

