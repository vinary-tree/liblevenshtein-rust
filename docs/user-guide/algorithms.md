# Levenshtein Algorithm Guide

**Version**: 0.9.1
**Last Updated**: 2026-08-01

This guide explains the Levenshtein distance algorithms supported by liblevenshtein-rust and when to use each one.

> **Forward compatibility.** `Algorithm` is non-exhaustive. Applications that
> pattern-match it must include a wildcard arm so later exact algorithms can be
> added without forcing an immediate source-breaking release. The existing
> `Standard`, `Transposition` (OSA), and `MergeAndSplit` query behavior is
> unchanged; internal selection is monomorphized once per dictionary edge.

The parameter-free `Algorithm` selectors and the separately parameterized
affine-gap query are summarized below. The operation-set diagram covers the
unit-cost selectors; affine gaps additionally remember which gap layer is open.

![Levenshtein operation sets: Standard (insert, delete, substitute), Transposition (adds adjacent swap), and Merge-and-Split (adds character merge and split)](../diagrams/automata/operation-sets.svg)

## What is Levenshtein Distance?

The Levenshtein distance is a metric for measuring the difference between two strings. It counts the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into another.

**Example:**
- "kitten" → "sitting" requires 3 edits (distance = 3):
  1. kitten → sitten (substitute 'k' with 's')
  2. sitten → sittin (substitute 'e' with 'i')
  3. sittin → sitting (insert 'g')

## Supported Algorithms

liblevenshtein-rust supports four parameter-free `Algorithm` selectors plus an
exact, parameterized affine-gap query:

### 1. Standard Levenshtein

**Operations**: Insert, Delete, Substitute

**Use Cases:**
- General string matching
- Spell checking
- Fuzzy search in databases
- Data deduplication

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec!["hello", "world", "test"]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Find terms within distance 2 of "helo"
for candidate in transducer.query_with_distance("helo", 2) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
hello: 1  (insert 'l')
```

**When to use:** Default choice for most fuzzy matching use cases.

### 2. Transposition (Optimal String Alignment)

**Operations**: Insert, Delete, Substitute, Transposition

**Transposition**: Swap two adjacent characters (counted as 1 edit instead of 2)

**Use Cases:**
- Typo correction (common typo pattern)
- Keyboard input errors
- OCR with character swapping
- User input validation

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "best", "rest"
]);
let transducer = Transducer::new(dict, Algorithm::Transposition);

// "tset" is a transposition of "test" (swapped 's' and 'e')
for candidate in transducer.query_with_distance("tset", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
test: 1  (transposition)
```

**With Standard algorithm**, "tset" → "test" requires 2 edits (substitute twice), so it wouldn't be found with distance 1.

**Semantics:** This is optimal string alignment (OSA), also called restricted
Damerau distance. It is not unrestricted Damerau–Levenshtein: a substring
cannot be edited twice. OSA is not a metric, so do not use it with metric-tree
pruning. See [`Algorithm::is_metric`](../../src/transducer/algorithm.rs).

**When to use:** When you expect users to make transposition errors (very common with typing) and the index does not require the triangle inequality.

### 3. Unrestricted Damerau–Levenshtein

**Operations:** Insert, Delete, Substitute, and adjacent Transposition, with
later operations allowed to edit an earlier operation's output.

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(["AC", "ABC", "CA"]);
let transducer = Transducer::with_damerau_levenshtein(dict);
let results: Vec<_> = transducer
    .query_with_distance("CA", 2)
    .map(|candidate| (candidate.term, candidate.distance))
    .collect();

assert!(results.contains(&("ABC".to_owned(), 2)));
```

**Semantics:** This is the true metric distance. The same query under OSA costs
3. Use it when edits can overlap or compose, or when a metric-dependent index
needs a transposition-aware distance.

**Limits:** The unit-cost automaton stores its pending endpoint delta in one
byte. The exact API ceiling is 255, while the measured and recommended fuzzy
search range is 1–3. Weighted `_f64` and phonetic NFA-product APIs reject this
selector because they do not carry the required history.

See the [literate algorithm chapter](../algorithms/11-true-damerau/README.md)
for the recurrence, proof/test mapping, and resource measurements.

### 4. Affine gap

**Operations:** Match, Substitute, Query Gap, Dictionary Gap

An affine gap charges one opening cost $`g_o`$ and one extension cost
$`g_e`$ per symbol. A length-$`r`$ run therefore costs
$`G(r)=g_o+r g_e`$. This makes one contiguous insertion or deletion region
cheaper than repeatedly opening short gaps.

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::transducer::{AffineGapParams, Algorithm, Transducer};

let dictionary = DoubleArrayTrie::from_terms(["a", "abcd"]);
let transducer = Transducer::new(dictionary, Algorithm::Standard);
let costs = AffineGapParams::new(3.0, 2.0, 10.0).expect("exact costs");
let candidate = transducer
    .query_affine("a", 9.0, costs)
    .expect("exact budget")
    .find(|candidate| candidate.term == "abcd")
    .expect("length-three dictionary gap is affordable");

assert_eq!(candidate.distance, 9.0); // 3 + 3 * 2
assert_eq!(candidate.scaled_distance, 9);
```

**Semantics:** Decimal costs are converted to exact fixed-point integers or
rejected. The stored `Algorithm` does not change an affine query; the typed
method supplies its own three-layer Gotoh variant. `query_affine_scaled` avoids
decimal conversion when a caller already has an exact scaled budget.

**When to use:** Sequence alignment, OCR, or any model where a contiguous
missing/extra region should pay one opening penalty.

See the [literate affine-gap chapter](../algorithms/10-affine-gap/README.md) for
the recurrence, cost convention, proof mapping, and resource controls.

### 5. Merge and Split

**Operations**: Insert, Delete, Substitute, Merge, Split

- **Merge**: Combine two characters into one (e.g., "ab" → "a")
- **Split**: Split one character into two (e.g., "a" → "ab")

**Use Cases:**
- OCR (Optical Character Recognition) errors
- Concatenation/separation issues
- Word segmentation errors
- Character recognition with merge/split ambiguity

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "hello", "world"
]);
let transducer = Transducer::new(dict, Algorithm::MergeAndSplit);

// "te st" can be merged to "test"
// "hel lo" can be merged to "hello"
for candidate in transducer.query_with_distance("te st", 2) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**When to use:** When dealing with OCR output or text where words may be incorrectly split or merged.

## Performance Characteristics

### Time Complexity

All algorithms share the same asymptotics, differing only by small constant factors:
- **Worst-case query time**: $`\mathcal{O}(n \times m \times k)`$, where:
  - $`n`$ = query term length
  - $`m`$ = dictionary size (number of edges)
  - $`k`$ = maximum distance
- **In practice**: far faster — the automaton is walked in lock-step with the
  dictionary, so work tracks the explored near-match frontier (early termination
  and subsumption pruning), not the full dictionary size $`m`$.

### Space Complexity

- **Standard**: $`\mathcal{O}(n \times k)`$ state space
- **Transposition**: $`\mathcal{O}(n \times k)`$ state space with transposition tracking
- **Unrestricted Damerau**: $`\mathcal{O}(n \times k^2)`$ worst-case pending-history frontier
- **Affine gap**: three layer-tagged $`\mathcal{O}(n \times k)`$ frontiers; zero extension can expose the full remaining query
- **Merge and Split**: $`\mathcal{O}(n \times k)`$ state space with merge/split tracking

### Benchmark Comparison

Relative query performance (DoubleArrayTrie, max distance = 2):

| Algorithm | Relative Speed | Use Case |
|-----------|----------------|----------|
| Standard | 1.0× (baseline) | General fuzzy matching |
| Transposition | 0.9× | Typo correction |
| Merge and Split | 0.7× | OCR error handling |

**Note**: With SIMD enabled, all algorithms see 20-64% performance improvements.

## Algorithm Selection Guide

### Decision Tree

```
Do you need to handle...
│
├─ General typos and spelling errors?
│  └─> Use Algorithm::Standard
│
├─ Typing mistakes with swapped characters?
│  └─> Use Algorithm::Transposition
│
├─ Long contiguous insertion/deletion regions with one opening penalty?
│  └─> Use query_affine / query_affine_scaled
│
└─ OCR character-confusion merges or splits?
   └─> Use Algorithm::MergeAndSplit
```

### Real-World Examples

**Code Completion (IDE):**
```rust
// Users make transposition errors when typing fast
Algorithm::Transposition
```

**Spell Checker:**
```rust
// General spelling errors
Algorithm::Standard
```

**OCR Post-Processing:**
```rust
// Characters may be merged or split incorrectly
Algorithm::MergeAndSplit
```

**Database Fuzzy Search:**
```rust
// General matching with typos
Algorithm::Standard
```

**Form Input Validation:**
```rust
// Catch typing errors including transpositions
Algorithm::Transposition
```

## Distance Threshold Selection

Choosing the right maximum distance (`max_distance`) is crucial:

### Guidelines

- **Distance 1**: Strict matching, catches single-character errors
  - Best for: Short words (3-5 chars), real-time autocomplete
  - Example: "test" matches "test", "fest", "tess", "tes"

- **Distance 2**: Moderate matching, most common choice
  - Best for: Medium words (6-10 chars), spell checking
  - Example: "test" matches above + "tent", "best", "tests"

- **Distance 3+**: Loose matching, more false positives
  - Best for: Long words (10+ chars), fuzzy name matching
  - Use with caution: can return many irrelevant results

### Rule of Thumb

$`\text{max\_distance} \approx \text{word\_length} / 4`$

Examples:
- 4-letter word: distance 1
- 8-letter word: distance 2
- 12-letter word: distance 3

### Testing Your Distance Threshold

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester",
    "best", "rest", "west", "nest"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Test different distances
for max_dist in 1..=3 {
    println!("\nDistance {}:", max_dist);
    for candidate in transducer.query_with_distance("test", max_dist).sorted() {
        println!("  {}: {}", candidate.term, candidate.distance);
    }
}
```

## Advanced Usage

### Alignment-expressible Class-A distances

Use the direct references when comparing two strings, and use the same preset
with `GeneralizedAutomaton` when operation-driven acceptance is useful:

```rust
use liblevenshtein::distance::{hamming_distance, indel_distance_bounded};
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::OperationSet;

assert_eq!(hamming_distance("abc", "bca"), Some(3));
assert_eq!(hamming_distance("abc", "ab"), None);
assert_eq!(indel_distance_bounded("abc", "bca", 2), Some(2));

let skip = GeneralizedAutomaton::try_with_operations(2, OperationSet::bounded_skip())?;
assert_eq!(skip.scaled_distance("crate", "cat")?, Some(2));
assert_eq!(skip.scaled_distance("cat", "crate")?, None);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Hamming requires equal Unicode-scalar lengths. Indel permits only insertion and
deletion, so substitution costs two. Bounded skip is directional: the second
argument must be a subsequence of the first. These presets do not connect to
the dictionary `QueryIterator`; the preregistered specialized-walker gate was
rejected. See the [Class-A design](../design/class-a-presets.md) and
[literate references](../algorithms/15-class-a-presets/README.md).

### Distance to a regular language

Use a language product when the query denotes several legal strings rather
than one literal. With the `phonetic-rules` feature, `query_regex` compiles a
bounded regular expression and returns the exact distance to its language:

```rust
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use liblevenshtein::transducer::{Algorithm, Transducer};

let dictionary = DoubleArrayTrieChar::from_terms([
    "phone", "fone", "phones", "stone",
]);
let transducer = Transducer::new(dictionary, Algorithm::Standard);
let matches: Vec<_> = transducer
    .query_regex("(ph|f)one", 1)
    .expect("valid bounded regex")
    .collect();
assert_eq!(matches.len(), 3);
```

The returned distance for term $`w`$ is
$`\min_{v\in L}d_{\mathrm{Lev}}(w,v)`$, where $`L`$ is the regex language.
`query_regex` uses Standard insertion, deletion, and substitution costs; the
`Algorithm` stored in the transducer does not change these semantics.

For byte, Unicode-scalar, or token-ID languages constructed in code, use
`query_language` with `SmallDfa<U>` or another `LanguageAutomaton<U>`. Untrusted
regexes are capped before Thompson construction; custom automata passed to
`query_language` are trusted and should be checked through `state_count()` at
the application boundary.

See the [language-product algorithm](../algorithms/13-language-products/README.md),
[design](../design/language-product.md), and
[resource guidance](../security/resource-exhaustion.md).

### Exact ERP time-series search

Use Edit distance with Real Penalty (ERP) when samples are real-valued, local
time shifts matter, and unmatched samples should be charged relative to one
fixed baseline $`g`$. `ErpTransducer` supports exact threshold and k-nearest
search over a quantized prefix trie:

```rust
use liblevenshtein::time_series::{ErpConfig, ErpTransducer, QuantizationConfig};

let series = vec![
    vec![1.0, 2.0, 3.0],
    vec![1.0, 0.0, 2.0, 3.0],
    vec![7.0, 8.0],
];
let index = ErpTransducer::from_series(
    QuantizationConfig::for_u8(-10.0, 10.0),
    ErpConfig::new(0.0),
    &series,
);

let within_one = index.search_range(&[1.0, 2.0, 3.0], 1.0);
let nearest_two = index.search_knn(&[1.0, 2.0, 3.0], 2, f64::INFINITY);
assert_eq!(within_one.len(), 2);
assert_eq!(nearest_two.len(), 2);
```

Quantization affects prefix sharing and pruning strength, not returned scores:
every survivor is re-evaluated against its stored full-precision original.
Choose $`g`$ once for the index. Inserting or deleting a sample equal to
$`g`$ costs zero, so ERP is a pseudometric on raw sequences and may produce
zero-distance ties between different vectors. Reject NaN and infinities at an
external API boundary and cap both query and reference lengths; exact ERP is
quadratic in the worst case.

See the [ERP paper analysis](../research/erp/PAPER_SUMMARY.md),
[elastic-kernel design](../design/elastic-kernels.md), and
[literate elastic algorithm](../algorithms/12-elastic-measures/README.md).

### Exact TWED time-series search

Use Time Warp Edit Distance (TWED) when adjacent-sample shape and temporal
displacement should both contribute to an additive edit cost. The crate uses
unit-spaced timestamps. $`\nu`$ controls temporal stiffness and
$`\lambda`$ is the constant part of every insertion or deletion.

For a metric-safe index configuration, validate finite $`\nu>0`$ and finite
$`\lambda\ge0`$:

```rust
use liblevenshtein::time_series::{
    MetricTwedConfig, MetricTwedTransducer, QuantizationConfig,
};

let series = vec![
    vec![0.0, 1.0, 2.0, 3.0],
    vec![0.0, 1.2, 2.1, 3.0],
    vec![8.0, 9.0],
];
let kernel = MetricTwedConfig::try_new(0.5, 1.0)
    .expect("positive stiffness and non-negative penalty");
let index = MetricTwedTransducer::from_series(
    QuantizationConfig::for_u8(0.0, 10.0),
    kernel,
    &series,
);

let exact = index.search_range(&[0.0, 1.0, 2.0, 3.0], 0.0);
let nearest_two = index.search_knn(&[0.0, 1.0, 2.0, 3.0], 2, f64::INFINITY);
assert_eq!(exact, vec![(0, 0.0)]);
assert_eq!(nearest_two.len(), 2);
```

`TwedConfig` exposes the complete non-negative family when the application
intentionally needs $`\nu=0`$. It does not implement the metric marker:

```rust
use liblevenshtein::time_series::TwedConfig;

let degenerate = TwedConfig::new(0.0, 0.0);
assert_eq!(degenerate.distance(&[0.0, 1.0], &[1.0]), 0.0);
```

That unequal-series zero-distance witness is why callers must not infer
metricity from parameter non-negativity alone. `MetricTwedConfig` makes the
strict primary-source premise a type invariant; `TwedConfig` remains suitable
for exact lower-bound trie search because that traversal does not use the
triangle inequality.

Quantization affects prefix sharing and bound tightness, never returned
scores. Empty/nonempty distance is finite and accumulates segment deletions
from the zero sentinel. Non-finite samples are outside the exact search domain.
Worst-case time is $`\mathcal{O}(mn)`$, so cap both sequence lengths and total
candidate work for untrusted requests. Setting $`\lambda=0`$ also disables
the length lower bound and can reduce pruning without changing correctness.

See the [Marteau source analysis](../research/twed/PAPER_SUMMARY.md),
[kernel design](../design/elastic-kernels.md),
[formal proof map](../verification/README.md), and
[resource controls](../security/resource-exhaustion.md).

### Exact discrete Fréchet time-series search

Use discrete Fréchet when the application cares about the worst separation
along an order-preserving coupling rather than the sum of local deviations.
`FrechetTransducer` provides exact threshold and k-nearest search with the same
quantized prefix trie, but its DP accumulates with bottleneck `max`:

```rust
use liblevenshtein::time_series::{
    FrechetConfig, FrechetTransducer, QuantizationConfig,
};

let series = vec![
    vec![1.0, 2.0, 3.0],
    vec![1.0, 1.0, 2.0, 3.0],
    vec![7.0, 8.0],
];
let index = FrechetTransducer::from_series(
    QuantizationConfig::for_u8(-10.0, 10.0),
    FrechetConfig::new(),
    &series,
);

let exact_stutters = index.search_range(&[1.0, 2.0, 3.0], 0.0);
let nearest_two = index.search_knn(&[1.0, 2.0, 3.0], 2, f64::INFINITY);
assert_eq!(exact_stutters.len(), 2);
assert_eq!(nearest_two.len(), 2);
```

Quantization changes only prefix sharing and pruning strength; emitted scores
come from the full-precision two-row DP. Consecutive duplicates collapse at
zero cost, so distinct raw vectors can tie at zero. Both-empty distance is
zero, exactly one empty side has infinite distance, and kNN emits finite scores
only. Reject non-finite samples and cap both sides before accepting untrusted
work: worst-case exact time remains $`\mathcal{O}(mn)`$.

See the [Eiter–Mannila analysis](../research/frechet/PAPER_SUMMARY.md),
[formal proof map](../verification/README.md), and
[literate recurrence](../algorithms/12-elastic-measures/README.md).

### Exact banded-DTW time-series search

Use banded dynamic time warping (DTW) when the application wants an additive
alignment cost, accepts a caller-selected temporal window, and does not require
metric-tree semantics. The **band** is the inclusive Sakoe–Chiba half-width
$`w`$: only alignments with $`\lvert i-j\rvert\le w`$ exist. It is required
because it changes both the distance and the resource bound.

```rust
use liblevenshtein::time_series::{
    DtwConfig, DtwTransducer, QuantizationConfig,
};

let series = vec![
    vec![0.0, 1.0, 2.0, 3.0],
    vec![0.0, 1.0, 1.0, 2.0, 3.0],
    vec![8.0, 9.0],
];
let index = DtwTransducer::from_series(
    QuantizationConfig::for_u8(0.0, 10.0),
    DtwConfig::new(1), // required; there is no implicit unbanded mode
    &series,
);

// Thresholds and returned scores are conventional square-root DTW units.
let near = index.search_range(&[0.0, 1.0, 2.0, 3.0], 0.5);
let nearest_two = index.search_knn(&[0.0, 1.0, 2.0, 3.0], 2, 0.0);
assert!(!near.is_empty());
assert!(near.iter().all(|(_, distance)| *distance <= 0.5));
assert_eq!(nearest_two.len(), 2);
```

The kernel accumulates squared deviations internally, including LB_Keogh and
trie-column bounds, and converts units only at the public boundary.
Quantization affects sharing and pruning but never emitted scores. A length
difference larger than $`w`$, exactly one empty side, or a non-finite sample
has no finite result.

DTW is **not a metric**. It is symmetric and non-negative, but it can violate
the triangle inequality; `DtwConfig::IS_METRIC` is therefore `false`, and the
type cannot be supplied where `MetricElasticKernel` is required. Do not put it
in a BK-tree, VP-tree, cover tree, or any index whose pruning proof uses metric
balls. `DtwTransducer` is exact because it uses interval and LB_Keogh lower
bounds followed by full-precision re-scoring instead.

For untrusted workloads, cap query length, reference length, band, result
count, and wall-clock work. A band as wide as the series recovers quadratic
work, and an infinite cutoff can still visit the full trie. See the
[DTW source analysis](../research/dtw/PAPER_SUMMARY.md),
[kernel design](../design/elastic-kernels.md),
[formal proof map](../verification/README.md), and
[resource controls](../security/resource-exhaustion.md).

### Observing elastic-search pruning

Every generic elastic transducer, including the MSM, ERP, TWED, and discrete
Fréchet aliases, can return observational kNN counters without changing its
results. `DtwTransducer` exposes the same API while retaining root-distance
results:

```rust
use liblevenshtein::time_series::{
    DtwConfig, DtwTransducer, QuantizationConfig,
};

let series = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
let index = DtwTransducer::from_series(
    QuantizationConfig::for_u8(0.0, 5.0),
    DtwConfig::new(1),
    &series,
);
let query = [0.0, 1.0, 2.0];
let ordinary = index.search_knn(&query, 1, f64::INFINITY);
let (observed, stats) =
    index.search_knn_with_stats(&query, 1, f64::INFINITY);

assert_eq!(observed, ordinary);
assert!(stats.accounting_is_consistent());
assert_eq!(stats.prefix_pruned + stats.columns_built, stats.visited_edges);
```

Use raw counts when comparing configurations. In particular, record prefix
prunes, built columns, column prunes, candidate-bound prunes, exact evaluations,
and cutoff abandonments separately; one favorable ratio can otherwise conceal
work shifted into another stage. These counters are diagnostic observations,
not admission-control limits. The reproducible five-measure command and fixed
analysis are documented in the
[shared UCR ledger](../scientific-ledger/elastic-ucr-harness-2026-08-01.md).

### Combining with Filters

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester", "best"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Only words with distance ≤ 1 and starting with 't'
for candidate in transducer
    .query_with_distance("test", 2)
    .filter(|c| c.distance <= 1 && c.term.starts_with('t'))
{
    println!("{}: {}", candidate.term, candidate.distance);
}
```

### Prefix Matching

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "apple", "application"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Only match terms with prefix "tes"
for candidate in transducer
    .query_with_distance("test", 1)
    .with_prefix("tes")
{
    println!("{}", candidate.term);
}
```

## Unicode Considerations

For correct character-level distances with Unicode text, use character-level dictionaries:

```rust
use liblevenshtein::prelude::*;

// ❌ Byte-level (incorrect for Unicode)
let dict_byte = DoubleArrayTrie::from_terms(vec!["café", "naïve"]);

// ✅ Character-level (correct for Unicode)
let dict_char = DoubleArrayTrieChar::from_terms(vec!["café", "naïve"]);

let transducer = Transducer::new(dict_char, Algorithm::Standard);

// Now distance is calculated correctly for multi-byte characters
for candidate in transducer.query_with_distance("cafe", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Trade-offs:**
- ~5% performance overhead
- 4× memory for edge labels
- Correct Unicode Levenshtein distances

## Related Documentation

- [Getting Started](getting-started.md) - Basic usage guide
- [Features](features.md) - Complete feature list
- [Code Completion Guide](code-completion.md) - Building code completion
- [Glossary](../GLOSSARY.md) - Definitions of terms used throughout the docs
- [Architecture Overview](../architecture/overview.md) - How the crates fit together
- [Benchmarks](../benchmarks/README.md) - Performance measurements
- [Language Products](../algorithms/13-language-products/README.md) - Fuzzy distance to a regular language
- [Elastic Measures](../algorithms/12-elastic-measures/README.md) - Exact MSM, ERP, discrete Fréchet, and banded-DTW trie search

## References

- [Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Damerau-Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
- Schulz, K. U., & Mihov, S. (2002). "Fast string correction with Levenshtein automata." *International Journal on Document Analysis and Recognition (IJDAR)*, 5(1), 67–85. DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
- Chen, L., & Ng, R. T. (2004). "On the Marriage of Lp-norms and Edit Distance." *VLDB 2004*, 792–803. DOI: [10.1016/B978-012088469-8.50070-X](https://doi.org/10.1016/B978-012088469-8.50070-X)

---

[← Documentation Index](../README.md)
