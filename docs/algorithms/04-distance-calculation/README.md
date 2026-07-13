# Layer 4: Distance Calculation

## Overview

The Distance Calculation layer provides direct computation of edit distances between two strings. Unlike fuzzy search algorithms that find candidates within a dictionary, these functions compute exact distances for any pair of strings.

This layer implements multiple algorithmic approaches and distance variants, each optimized for different use cases.

Throughout this layer, `$m$` and `$n$` denote the lengths (in characters) of the two strings being compared.

![Distance dispatch: how a requested algorithm (standard, transposition, merge-and-split) selects the matching edit-distance routine](../../diagrams/distance/distance-dispatch.svg)

*Distance dispatch: the requested `Algorithm` selects the corresponding edit-distance routine.*

## Distance Metrics

### Standard Levenshtein Distance

The minimum number of single-character edits (insertions, deletions, substitutions) to transform one string into another.

```rust
use liblevenshtein::distance::standard_distance;

assert_eq!(standard_distance("kitten", "sitting"), 3);
// Edits: k→s, e→i, insert g
```

**Complexity**: `$\mathcal{O}(mn)$` time, `$\mathcal{O}(\min(m,n))$` space

### Damerau-Levenshtein Distance (Transposition)

Extends standard Levenshtein to include transposition (swapping adjacent characters) as a single operation.

```rust
use liblevenshtein::distance::transposition_distance;

assert_eq!(transposition_distance("test", "tset"), 1);
// One transposition: 'es' → 'se'

// Compare with standard distance:
assert_eq!(standard_distance("test", "tset"), 2);
// Two substitutions required
```

**Complexity**: `$\mathcal{O}(mn)$` time, `$\mathcal{O}(\min(m,n))$` space (3 rows)

### Merge-and-Split Distance

Supports merge (two chars → one char) and split (one char → two chars) operations, useful for OCR errors and phonetic matching.

```rust
use liblevenshtein::distance::merge_and_split_distance;

let cache = liblevenshtein::distance::create_memo_cache();

// Split: 'm' → 'rn'
assert_eq!(merge_and_split_distance("m", "rn", &cache), 1);

// Merge: 'rn' → 'm'
assert_eq!(merge_and_split_distance("rn", "m", &cache), 1);
```

**Complexity**: `$\mathcal{O}(mn)$` time with memoization

## Implementation Approaches

### 1. Iterative Dynamic Programming (Default)

Space-optimized DP using 2-3 row vectors instead of full matrix.

**Pros**:
- Predictable performance: `$\mathcal{O}(mn)$` always
- Low memory footprint: `$\mathcal{O}(\min(m,n))$`
- No recursion stack overhead
- Cache-friendly sequential access

**Cons**:
- Recomputes subproblems for repeated queries
- No early exit optimizations

**Best for**:
- One-off distance queries
- Memory-constrained environments
- Predictable latency requirements

### 2. Recursive + Memoization (C++-style)

Recursive descent with thread-safe caching and aggressive optimizations.

**Pros**:
- Cache reuse across queries
- Common prefix/suffix stripping (major speedup)
- Early exit on distance=0
- Symmetric pair deduplication

**Cons**:
- Recursion stack depth (mitigated by prefix stripping)
- Cache overhead (~24 bytes per unique pair)

**Best for**:
- Repeated queries on similar strings
- Batch distance computations
- Strings with common prefixes/suffixes

### 3. SIMD-Accelerated (AVX2/SSE4.1)

Vectorized implementations for parallel processing of DP operations.

**Pros**:
- 2-4× speedup on long strings (>16 chars)
- Runtime CPU feature detection
- Automatic fallback to scalar

**Cons**:
- Overhead dominates for short strings
- Architecture-specific (x86_64 only currently)

**Best for**:
- Long strings (>16 characters)
- High-throughput distance computation
- x86_64 CPUs with AVX2/SSE4.1

## Architecture

### Module Structure

```
src/distance/
├── mod.rs          - Iterative DP + recursive implementations
└── simd.rs         - SIMD-accelerated versions (AVX2, SSE4.1)
```

### Algorithm Selection Flowchart

```
                    ┌─────────────────────┐
                    │ Distance Query      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Need transposition? │
                    └──────┬─────────┬────┘
                    No     │         │ Yes
                           │         │
          ┌────────────────▼─┐   ┌──▼────────────────────┐
          │ Need merge/split?│   │ transposition_distance│
          └────┬─────────┬───┘   │ (iterative or         │
           No  │         │ Yes   │  recursive)           │
               │         │       └───────────────────────┘
    ┌──────────▼──┐ ┌───▼──────────────────┐
    │ Repeated    │ │ merge_and_split_dist │
    │ queries?    │ │ (recursive + cache)  │
    └──┬─────┬────┘ └──────────────────────┘
   No  │     │ Yes
       │     │
  ┌────▼──┐  └──────┬────────────────────────────┐
  │ Short │         │ Long common prefix/suffix? │
  │ or    │         └─────┬─────────────┬────────┘
  │ long? │            No │             │ Yes
  └──┬──┬─┘               │             │
  Short│ Long        ┌────▼────┐   ┌────▼────────────────┐
       │             │ standard│   │ recursive + cache   │
   ┌───▼──────────┐  │ _distance│  │ (prefix stripping)  │
   │ standard_    │  │ (scalar)│   └─────────────────────┘
   │ distance_impl│  └─────────┘
   │ (scalar)     │
   └──────────────┘
       │
       │ x86_64 + len>16?
       ▼
  ┌───────────────────┐
  │ standard_distance │
  │ (SIMD auto-select)│
  │ AVX2 > SSE > scal │
  └───────────────────┘
```

## Optimizations

### 1. Common Prefix/Suffix Stripping

Recursively strip identical characters from start and end before computing distance.

```rust
// "prefix_abc_suffix" vs "prefix_def_suffix"
// Strip → compute distance("abc", "def") → add back prefix/suffix length

pub fn strip_common_affixes(a: &str, b: &str) -> (usize, usize, usize) {
    // Returns (prefix_len, adjusted_len_a, adjusted_len_b)
}
```

**Impact**: 5-10× speedup for strings with 50%+ overlap

### 2. Space Optimization (Row-Based DP)

Use 2-3 row vectors instead of full `$m \times n$` matrix.

```
Traditional:  O(m×n) space → Full matrix
Optimized:    O(n) space   → Only previous row + current row
With trans:   O(3n) space  → Need two-rows-ago for transposition
```

**Impact**: 100× memory reduction for long strings (m=1000, n=1000)

### 3. Symmetric Pair Caching

Exploit d(a,b) = d(b,a) to deduplicate cache entries.

```rust
struct SymmetricPair {
    first: Arc<str>,   // Lexicographically smaller
    second: Arc<str>,  // Lexicographically larger
}
```

**Impact**: 50% cache size reduction for bidirectional queries

### 4. Early Exit Optimization

Return immediately when distance=0 found in recursion.

```rust
if a == b {
    distance = recursive(s, t, cache);
    if distance == 0 {
        cache.insert(key, 0);
        return 0;  // Early exit!
    }
}
```

**Impact**: 2-3× speedup for near-identical strings

### 5. SIMD Vectorization

Process 8 (AVX2) or 4 (SSE4.1) DP cells in parallel.

```rust
// Scalar: for j in 1..=n { compute cell[j] }
// SIMD:   for j in (1..=n).step_by(8) { compute 8 cells in parallel }
```

**Impact**: 2-4× throughput for strings >16 chars

## Usage Examples

### Example 1: Basic Distance Computation

```rust
use liblevenshtein::distance::standard_distance;

fn main() {
    // Identical strings
    assert_eq!(standard_distance("test", "test"), 0);

    // Empty strings
    assert_eq!(standard_distance("", "hello"), 5);
    assert_eq!(standard_distance("world", ""), 5);

    // Classic example: "kitten" → "sitting"
    assert_eq!(standard_distance("kitten", "sitting"), 3);
    // k→s, e→i, +g

    // Single character difference
    assert_eq!(standard_distance("test", "best"), 1);
}
```

### Example 2: Transposition Distance

```rust
use liblevenshtein::distance::{standard_distance, transposition_distance};

fn main() {
    let s1 = "test";
    let s2 = "tset";

    // Standard distance: requires 2 substitutions
    // test → tset
    //   ^  ^
    // e→s, s→e (2 ops)
    assert_eq!(standard_distance(s1, s2), 2);

    // Transposition distance: 1 swap
    // test → tset (swap 'e' and 's')
    assert_eq!(transposition_distance(s1, s2), 1);

    // Common typos where transposition is better:
    assert_eq!(transposition_distance("form", "from"), 1);  // 1 swap
    assert_eq!(standard_distance("form", "from"), 2);       // 2 substitutions

    assert_eq!(transposition_distance("teh", "the"), 1);
    assert_eq!(transposition_distance("recieve", "receive"), 1);
}
```

### Example 3: Recursive with Memoization

```rust
use liblevenshtein::distance::{create_memo_cache, standard_distance_recursive};

fn main() {
    let cache = create_memo_cache();

    // First query: cache miss, computes distance
    let d1 = standard_distance_recursive("kitten", "sitting", &cache);
    assert_eq!(d1, 3);

    // Second query: cache hit, instant return
    let d2 = standard_distance_recursive("kitten", "sitting", &cache);
    assert_eq!(d2, 3);

    // Symmetric query: also uses cache (due to SymmetricPair)
    let d3 = standard_distance_recursive("sitting", "kitten", &cache);
    assert_eq!(d3, 3);

    // Process batch of string pairs
    let pairs = vec![
        ("test", "best"),
        ("hello", "hallo"),
        ("algorithm", "logarithm"),
        ("test", "best"),  // Duplicate: uses cache
    ];

    for (a, b) in pairs {
        let distance = standard_distance_recursive(a, b, &cache);
        println!("{} ↔ {}: {}", a, b, distance);
    }
}
```

### Example 4: Common Prefix Optimization

```rust
use liblevenshtein::distance::{create_memo_cache, standard_distance_recursive};

fn main() {
    let cache = create_memo_cache();

    // Strings with long common prefix
    let s1 = "http://example.com/path/to/resource1";
    let s2 = "http://example.com/path/to/resource2";

    // Only computes distance on differing part: "resource1" vs "resource2"
    let distance = standard_distance_recursive(s1, s2, &cache);
    assert_eq!(distance, 1);  // Just the '1' vs '2'

    // Without prefix stripping, would compute full O(m×n) = O(38×38) = 1,444 ops
    // With prefix stripping: O(9×9) = 81 ops → 18× faster!
}
```

### Example 5: SIMD Acceleration (Automatic on x86_64)

```rust
use liblevenshtein::distance::standard_distance;

fn main() {
    // On x86_64 targets, automatically uses AVX2 if available, else SSE4.1,
    // else scalar. CPU feature selection happens at runtime via
    // is_x86_feature_detected!. On other architectures the scalar path runs.
    let long_string_a = "a".repeat(100);
    let long_string_b = "b".repeat(100);

    let distance = standard_distance(&long_string_a, &long_string_b);
    assert_eq!(distance, 100);

    // For short strings, SIMD overhead is skipped
    let distance_short = standard_distance("test", "best");
    assert_eq!(distance_short, 1);  // Uses scalar implementation
}
```

### Example 6: Merge-and-Split Distance

```rust
use liblevenshtein::distance::{create_memo_cache, merge_and_split_distance};

fn main() {
    let cache = create_memo_cache();

    // OCR errors: 'm' often misread as 'rn'
    assert_eq!(merge_and_split_distance("m", "rn", &cache), 1);  // Split
    assert_eq!(merge_and_split_distance("rn", "m", &cache), 1);  // Merge

    // 'cl' → 'd' (ligature)
    assert_eq!(merge_and_split_distance("cl", "d", &cache), 1);

    // Combined operations
    let ocr_original = "modern";
    let ocr_scanned = "rnodern";  // 'm' → 'rn' at start
    let distance = merge_and_split_distance(ocr_original, ocr_scanned, &cache);
    // Should be less than standard Levenshtein
    println!("{} ↔ {}: {}", ocr_original, ocr_scanned, distance);
}
```

### Example 7: Unicode Support

```rust
use liblevenshtein::distance::standard_distance;

fn main() {
    // CJK characters (multi-byte UTF-8)
    assert_eq!(standard_distance("日本", "日本"), 0);
    assert_eq!(standard_distance("日本", "日人"), 1);  // Last char differs

    // Emoji (4-byte UTF-8)
    assert_eq!(standard_distance("😀😁😂", "😀😁😃"), 1);

    // Mixed scripts
    assert_eq!(standard_distance("café", "cafe"), 1);  // é → e

    // Character-level (not byte-level) distance
    assert_eq!(standard_distance("é", "e"), 1);  // 1 char substitution
    // NOT 2 (é is 2 bytes in UTF-8 but 1 character)
}
```

### Example 8: Batch Processing with Caching

```rust
use liblevenshtein::distance::{create_memo_cache, standard_distance_recursive};
use std::time::Instant;

fn main() {
    let cache = create_memo_cache();

    let dictionary = vec![
        "apple", "application", "apply", "banana", "band", "can", "candy",
    ];

    let query = "app";

    let start = Instant::now();

    // Compute distances to all dictionary terms
    let mut results: Vec<_> = dictionary
        .iter()
        .map(|term| {
            let distance = standard_distance_recursive(query, term, &cache);
            (term, distance)
        })
        .collect();

    results.sort_by_key(|(_, dist)| *dist);

    let elapsed = start.elapsed();

    println!("Query: '{}' ({}ms)", query, elapsed.as_millis());
    for (term, distance) in results.iter().take(5) {
        println!("  {} (distance={})", term, distance);
    }

    // Subsequent queries benefit from cache
    let query2 = "apl";
    let results2: Vec<_> = dictionary
        .iter()
        .map(|term| {
            let distance = standard_distance_recursive(query2, term, &cache);
            (term, distance)
        })
        .collect();

    // This query is faster due to cached subproblems
}
```

## Performance Benchmarking

### Standard Distance (Scalar)

| String Length | Time (µs) | Throughput |
|---------------|-----------|------------|
| 10 × 10 | 0.8 | 1.25M ops/sec |
| 50 × 50 | 18.5 | 54K ops/sec |
| 100 × 100 | 72.3 | 13.8K ops/sec |
| 500 × 500 | 1,820 | 549 ops/sec |

### Standard Distance (SIMD AVX2)

| String Length | Time (µs) | Speedup vs Scalar |
|---------------|-----------|-------------------|
| 10 × 10 | 1.2 | 0.67× (overhead) |
| 50 × 50 | 8.3 | 2.23× |
| 100 × 100 | 28.4 | 2.54× |
| 500 × 500 | 512 | 3.55× |

**Key Insight**: SIMD wins for strings >16 chars, scalar wins for short strings.

### Recursive with Memoization

| Operation | Time | Cache Benefit |
|-----------|------|---------------|
| Cold cache (first query) | 1.2× scalar | - |
| Warm cache (cache hit) | 150ns | ~500× faster |
| Common prefix (80% overlap) | 0.15× scalar | 6.7× faster |

### Memory Usage

| Implementation | Space Complexity | 1000×1000 Strings |
|----------------|------------------|-------------------|
| Full matrix DP | `$\mathcal{O}(mn)$` | ~4 MB |
| 2-row optimized | `$\mathcal{O}(\min(m,n))$` | ~4 KB |
| 3-row (transposition) | `$\mathcal{O}(3\min(m,n))$` | ~12 KB |
| Recursive + cache | `$\mathcal{O}(\text{depth} + \text{cache})$` | ~24 KB (1000 cached pairs) |

## Integration with Other Layers

### Used By

- **Layer 6 (Fuzzy Search)**: Candidate distance verification
- **Layer 7 (Contextual Completion)**: Query-to-term matching
- **Layer 2 (Levenshtein Automata)**: Reference distance for validation

### Uses

- None (bottom layer for direct distance computation)

## Testing Coverage

The distance module includes comprehensive tests:

1. **Correctness**: Verified against known examples
2. **Edge Cases**: Empty strings, identical strings, single-char differences
3. **Unicode**: CJK, emoji, multi-byte characters
4. **Consistency**: Iterative vs recursive implementations match
5. **Symmetry**: d(a,b) = d(b,a)
6. **Cache**: Memoization reduces redundant computation
7. **SIMD**: Runtime CPU detection, fallback to scalar

See `src/distance/mod.rs:752-964` for test suite.

## Related Documentation

- **[Iterative DP Algorithm](./algorithms/iterative-dp.md)** - Space-optimized dynamic programming
- **[Recursive Memoization](./algorithms/recursive-memoization.md)** - C++-style recursive approach
- **[Optimizations](./algorithms/optimizations.md)** - Prefix stripping, SIMD, caching strategies
- **[Layer 2: Levenshtein Automata](../02-levenshtein-automata/README.md)** - Uses distance for validation
- **[Fuzzy Search (Intersection & Traversal)](../03-intersection-traversal/README.md)** - Uses distance for ranking

## References

1. Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals". *Soviet Physics Doklady*, 10(8), 707-710.
2. Damerau, F. J. (1964). "A technique for computer detection and correction of spelling errors". *Communications of the ACM*, 7(3), 171-176. DOI: [10.1145/363958.363994](https://doi.org/10.1145/363958.363994)
3. Wagner, R. A., & Fischer, M. J. (1974). "The String-to-String Correction Problem". *Journal of the ACM*, 21(1), 168-173. DOI: [10.1145/321796.321811](https://doi.org/10.1145/321796.321811)
4. Hyyrö, H. (2001). "A Bit-Vector Algorithm for Computing Levenshtein and Damerau Edit Distances". *Technical Report A-2001-10, University of Tampere*.
5. Benchmark data from `benches/real_world_benchmark.rs`

## Quick Reference

```rust
// Iterative DP (default, predictable)
use liblevenshtein::distance::standard_distance;
let d = standard_distance("test", "best");

// With transposition
use liblevenshtein::distance::transposition_distance;
let d = transposition_distance("test", "tset");

// Recursive + memoization (for repeated queries)
use liblevenshtein::distance::{create_memo_cache, standard_distance_recursive};
let cache = create_memo_cache();
let d = standard_distance_recursive("test", "best", &cache);

// Merge/split operations
use liblevenshtein::distance::merge_and_split_distance;
let d = merge_and_split_distance("m", "rn", &cache);

// SIMD-accelerated (automatic on x86_64; runtime selects AVX2/SSE4.1/scalar)
let d = standard_distance("long_string_a", "long_string_b");
```
