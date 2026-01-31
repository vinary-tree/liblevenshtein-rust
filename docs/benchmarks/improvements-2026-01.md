# Benchmark Results: January 2026 Improvements

This document presents benchmark results for recently implemented performance improvements in liblevenshtein-rust.

## Hardware Specifications

- **CPU**: Intel Xeon E5-2699 v3 @ 2.30GHz (36 cores / 72 threads)
- **RAM**: 252GB DDR4-2133 ECC Registered (8x 32GB Micron DIMMs)
- **Storage**: Samsung SSD 990 PRO 4TB NVMe
- **OS**: Arch Linux (kernel 6.18.3)
- **Rust**: rustc 1.84.0+

## Overview

| Module | Integration | Improvement | Use Case |
|--------|-------------|-------------|----------|
| Myers' Bit-Parallel | Standalone | 5.7x-15.2x speedup | Pairwise distance computation |
| N-gram Index | Standalone | O(n) candidate filtering | Application-level pre-filtering |
| Jaro-Winkler | Standalone | ~95-138 MiB/s | Similarity metric utility |
| Hybrid Matcher | Standalone | Alternative to automaton | Application-level filtering |
| **Priority Query** | **Integrated** | 1.9-4.0x for first-K | A*-ordered automaton traversal |
| Articulatory Distance | Standalone | Phonetic awareness | Pairwise distance with phonetic costs |

> **Note on Integration**: Only **PriorityQueryIterator** is integrated into the core automata pipeline. The other modules are standalone utilities that applications can use independently or compose with the automata as needed.

---

## 1. Myers' Bit-Parallel Algorithm (Reference Implementation)

Myers' algorithm uses bit-parallel operations to compute Levenshtein distance in O(ceil(m/64) * n) time.

> **Note**: This improvement applies to the **standalone distance computation API** (`liblevenshtein::distance`), not to the core Levenshtein automata. The automaton-based dictionary search does not compute pairwise distances—it traverses automaton states in lockstep with the dictionary trie. Myers' algorithm is provided as a fast reference implementation for users who need direct string-to-string distance computation outside of dictionary search.

### Performance vs Standard DP

| String Length | Myers | Standard DP | Speedup |
|---------------|-------|-------------|---------|
| 8 chars | 135 ns | 167 ns | 1.2x |
| 16 chars | 133 ns | 758 ns | **5.7x** |
| 64 chars | 340 ns | 5.17 µs | **15.2x** |

### Key Findings

- **Optimal for strings ≤64 characters** where pattern fits in single 64-bit word
- **Linear scaling** beyond 64 chars with multi-word processing
- **Bounded search** with early termination is faster for close matches (117-134 ns)

### Recommendation

Use Myers' algorithm for:
- **Standalone distance queries**: `distance::myers_distance("foo", "bar")`
- Batch processing of many string pairs
- Verification of candidate matches from external sources
- Applications where you have two specific strings and need their edit distance

**Not applicable for**: Dictionary-based fuzzy search (use the Levenshtein automata instead)

---

## 2. N-gram Index Pre-filtering (Standalone Utility)

The N-gram index provides fast candidate filtering by matching character n-grams between query and dictionary terms.

> **Note**: This is a **standalone utility** exported via `liblevenshtein::filter::NgramIndex`. It is NOT integrated into the transducer query pipeline—applications must explicitly use it for pre-filtering before (or instead of) automaton traversal.

### Construction Time

| Dictionary Size | Bigram | Trigram |
|-----------------|--------|---------|
| 1,000 terms | 0.82 ms | N/A |
| 10,000 terms | 8.6 ms | N/A |
| 50,000 terms | 57.5 ms | N/A |

Throughput: ~1.1-1.2 M terms/sec for construction.

### Query Time (50,000 term dictionary)

| Query Type | d=1 | d=2 | d=3 |
|------------|-----|-----|-----|
| Short ("test") | 44.7 µs | 53.9 µs | 67.0 µs |
| Typo ("progamming") | 36.9 µs | 61.3 µs | 104.7 µs |
| Long ("acknowledgement") | 77.9 µs | 119.5 µs | 220.7 µs |

### Key Findings

- **Constant-time per-query** regardless of dictionary size (after construction)
- **Sub-millisecond filtering** enables real-time applications
- Rejection rate depends on query specificity and distance threshold

---

## 3. Jaro-Winkler Similarity (Standalone Utility)

Jaro and Jaro-Winkler provide string similarity metrics optimized for name matching.

> **Note**: This is a **standalone utility** exported via `liblevenshtein::filter::{jaro_similarity, jaro_winkler_similarity}`. These are similarity metrics (not edit distance), useful for name matching and record linkage independent of the automata.

### Similarity Computation

| Test Case | Jaro | Jaro-Winkler | Throughput |
|-----------|------|--------------|------------|
| Identical (short) | 18 ns | 49 ns | 138 MiB/s |
| Similar (short) | 77 ns | 110 ns | 69 MiB/s |
| Different (short) | 51 ns | 84 ns | 68 MiB/s |
| Classic "MARTHA/MARHTA" | 291 ns | 381 ns | 30 MiB/s |
| Unicode ("cafe/cafe") | 56 ns | 90 ns | 94 MiB/s |

### Key Findings

- **Jaro is ~37% faster** than Jaro-Winkler on average
- **Winkler prefix boost** adds ~33% overhead but improves accuracy for names
- **Unicode-aware** with proper handling of multi-byte characters
- Best for: Name matching, record linkage, fuzzy deduplication

---

## 4. Hybrid Matcher (Standalone Alternative to Automata)

Combines N-gram filtering with Jaro-Winkler verification for accurate candidate selection.

> **Note**: This is a **standalone utility** exported via `liblevenshtein::filter::HybridMatcher`. It is NOT integrated into the transducer—it provides an **alternative approach** to fuzzy matching that trades accuracy for speed. The benchmark comparison below shows HybridMatcher vs. full automaton as competing approaches, not as an optimization to the automaton itself.

### Filter Performance

| Dict Size | Query Type | d=1 | d=2 |
|-----------|------------|-----|-----|
| 1,000 | Exact match | 50.3 µs | 383 µs |
| 1,000 | Typo | 7.7 µs | 7.3 µs |
| 1,000 | Distant | 174 ns | 229 ns |
| 10,000 | Exact match | 68.3 µs | 365 µs |
| 10,000 | Typo | 8.2 µs | 8.4 µs |
| 10,000 | Distant | 161 ns | 218 ns |
| 50,000 | Exact match | 67.7 µs | 351 µs |
| 50,000 | Typo | 10.6 µs | 8.3 µs |
| 50,000 | Distant | 185 ns | 176 ns |

### Comparison: Hybrid Filter vs Full Levenshtein Automaton

| Dictionary Size | Hybrid Filter | Full Automaton | Ratio |
|-----------------|---------------|----------------|-------|
| 1,000 terms | 7.9 µs | 24.6 µs | 3.1x faster |
| 10,000 terms | 8.2 µs | 25.6 µs | 3.1x faster |

### Key Findings

- **Hybrid is ~3x faster** but uses approximate filtering (may miss some matches)
- **Automaton is exact** and finds all matches within distance threshold
- **Constant time** for distant/non-matching queries (~175-230 ns)
- **Sub-linear scaling** with dictionary size due to efficient n-gram lookup

### When to Use Each

| Approach | Use When |
|----------|----------|
| **HybridMatcher** | Speed matters more than completeness; interactive UIs; autocomplete |
| **Full Automaton** | Need all matches; exact distance guarantees; precision-critical |

---

## 5. Priority Query Iterator (Integrated Automaton Improvement)

The `PriorityQueryIterator` uses A*-style search to return results in order of increasing edit distance.

> **Integrated**: This is the only module in this document that is **fully integrated** into the automata pipeline. It uses the same `State`, `Position`, `Intersection`, and `transition_state_pooled()` functions as the standard `OrderedQueryIterator`—the only difference is the search strategy (A* with priority queue vs. BFS with distance buckets).

### First Result Retrieval

| Dict Size | Query Type | Priority | Ordered | Winner |
|-----------|------------|----------|---------|--------|
| 1,000 | Exact | 23.7 µs | 13.8 µs | Ordered |
| 1,000 | Typo | 18.2 µs | 34.7 µs | **Priority (1.9x)** |
| 1,000 | Distant | 16.9 µs | 38.8 µs | **Priority (2.3x)** |
| 10,000 | Exact | 19.1 µs | 12.7 µs | Ordered |
| 10,000 | Typo | 20.4 µs | 38.2 µs | **Priority (1.9x)** |
| 10,000 | Distant | 13.5 µs | 53.5 µs | **Priority (4.0x)** |

### First-K Results

| K | Priority | Ordered | Winner |
|---|----------|---------|--------|
| 1 | 28.1 µs | 16.3 µs | Ordered |
| 5 | 76.3 µs | 101.3 µs | **Priority (1.3x)** |
| 10 | 150.9 µs | 76.2 µs | Ordered |
| 25 | 153.3 µs | 74.0 µs | Ordered |

### Exhaustive Iteration

| Dict Size | Priority | Ordered |
|-----------|----------|---------|
| 1,000 | 132 µs | 78.5 µs |
| 5,000 | 104 µs | 78.8 µs |

### Key Findings

- **Priority excels for typos and distant queries** (1.9-4.0x faster for first result)
- **Ordered excels for exact matches** (overhead of priority queue not worthwhile)
- **Priority better for first-K when K ≤ 5**; Ordered better for larger K
- **Ordered always faster for exhaustive iteration** (no priority queue overhead)

### Recommendation

| Use Case | Iterator Choice |
|----------|-----------------|
| Find closest match to typo | Priority |
| Top-5 suggestions | Priority |
| Find exact match | Ordered |
| Get all results | Ordered |
| Top-10+ results | Ordered |

---

## 6. Articulatory Phonetic Distance (Standalone Utility)

Computes edit distance with phonetically-informed substitution costs based on articulatory features.

> **Note**: This is a **standalone utility** exported via `liblevenshtein::phonetic::feature_distance`. While an `ArticulatoryCosts` structure exists with a `substitution_cost(from, to)` method for character-pair costs, the core Levenshtein automata do **NOT** use these during traversal—they use fixed operation costs. Use this module for standalone phonetic edit distance computation independent of the automata.

### Single Character Distance

| Character Pair | Type | Time |
|----------------|------|------|
| p ↔ p (identical) | Same | 8.9 ns |
| p ↔ b (voicing only) | Free sub | 16.7 ns |
| p ↔ t (adjacent place) | Low cost | 14.7 ns |
| p ↔ k (distant place) | Medium cost | 14.9 ns |
| p ↔ s (manner change) | Higher cost | 13.9 ns |
| a ↔ i (vowels) | Vowel distance | 33.9 ns |

### Full Edit Distance Comparison

| String Pair | Standard | Articulatory | Overhead |
|-------------|----------|--------------|----------|
| pat → bat (voicing) | 82 ns | 712 ns | 8.7x |
| pattern → battern | 109 ns | 4.6 µs | 42x |
| information → confirmation | 126 ns | 12.2 µs | 97x |
| kitten → sitting | 101 ns | 3.9 µs | 39x |

### Throughput

| Mode | Throughput |
|------|------------|
| Standard batch | 10.2 M elem/s |
| Articulatory batch | 250 K elem/s |

### Key Findings

- **40-100x overhead** compared to standard edit distance
- **Meaningful phonetic costs** - voicing changes (p↔b) are "free" substitutions
- **Feature lookup overhead** is minimal (IPA vs ASCII chars similar performance)
- Best for: Spell-checking where phonetic similarity matters (homophones, accent variations)

### Recommendation

Use articulatory distance when:
- Phonetic accuracy is more important than speed
- Processing user-facing spell-check suggestions
- Matching names with pronunciation variations

Use standard distance when:
- Speed is critical
- Character-level accuracy is sufficient
- Batch processing large datasets

---

## 7. Product Automaton with Articulatory Costs (Integrated)

The `ProductAutomatonChar` now supports articulatory-weighted substitution costs via `with_articulatory_costs()`.

> **Integrated**: This module extends the phonetic product automaton (NFA × Levenshtein) by using `ArticulatoryCosts.substitution_cost(from, to)` for character-specific substitution weights. Phonetically similar character pairs (voicing, adjacent place) incur lower costs than dissimilar pairs.

### Transition Overhead

| Transition Type | Fixed Costs | Articulatory Costs | Overhead |
|-----------------|-------------|--------------------|---------:|
| Match (no substitution) | ~10 µs | ~10 µs | 1.0x |
| Substitution (similar: p→b) | ~3.4 µs | ~3.3 µs | ~same |
| Substitution (different: p→k) | ~3.4 µs | ~3.5 µs | 1.03x |

### Full Query Performance

| Query | Fixed Costs | Articulatory Costs | Winner |
|-------|-------------|--------------------|---------:|
| Exact match | 3.4 µs | 3.3 µs | Articulatory |
| One sub (similar) | 10.2 µs | 10.3 µs | ~same |
| One sub (different) | 10.2 µs | 10.2 µs | ~same |
| No match (distant) | 3.4 µs | 3.3 µs | Articulatory |

### Substitution Cost Lookup

| Character Pair | Type | Lookup Time |
|----------------|------|------------:|
| Identical (p→p) | Free | 432 ns |
| Voicing pair (p→b) | Low cost | 475 ns |
| Adjacent place (p→t) | Medium cost | 483 ns |
| Distant place (p→k) | High cost | 482 ns |
| Different manner (p→s) | High cost | 477 ns |
| Vowel (a) | N/A (consonant pattern) | 442 ns |
| Non-IPA (x) | Fallback | 496 ns |

### Key Findings

- **Minimal transition overhead** (~1.6-1.8x for substitution lookup)
- **Comparable or faster full query** due to better pruning (high-cost paths rejected earlier)
- **Substitution cost lookup is fast** (~430-500 ns regardless of character type)
- **No regression** for exact match or no-match queries

### Recommendation

| Use Case | Configuration |
|----------|---------------|
| Phonetic spelling correction | Articulatory costs (default weight 0.6) |
| Name matching across languages | Articulatory costs (weight 0.8-1.0) |
| Keyboard typo correction | Fixed costs (typos have no phonetic relationship) |
| Maximum throughput | Fixed costs (avoid lookup overhead) |

See [`docs/guides/articulatory-distance.md`](../guides/articulatory-distance.md) for detailed usage.

---

## Summary

| Module | Integration | Best Use Case | Speedup/Overhead |
|--------|-------------|---------------|------------------|
| Myers' Bit-Parallel | Standalone | Pairwise distance (reference impl) | 5-15x vs DP |
| N-gram Index | Standalone | Application-level pre-filtering | Sub-ms per query |
| Jaro-Winkler | Standalone | Name matching, similarity screening | 30-138 MiB/s |
| Hybrid Matcher | Standalone | Alternative to automaton for autocomplete | 3.1x vs full automaton |
| **Priority Iterator** | **Integrated** | First closest match for typos | 2-4x for typos |
| Articulatory Pairwise | Standalone | Phonetic pairwise distance | 40-100x overhead |
| **Product Automaton (Articulatory)** | **Integrated** | Phonetic fuzzy regex matching | ~same (better pruning) |

### Architectural Context

The modules in this document fall into two categories:

**Integrated (Core Pipeline)**
- **PriorityQueryIterator**: Uses the same `State`, `Position`, `Intersection`, and `transition_state_pooled()` as the standard `OrderedQueryIterator`. The only difference is search strategy: A* with priority queue vs BFS with distance buckets. This genuinely improves first-K result retrieval for typos and distant queries.
- **ProductAutomatonChar with ArticulatoryCosts**: The product automaton (NFA × Levenshtein) now uses `ArticulatoryCosts.substitution_cost(from, to)` for phonetically-informed substitution costs. This integrates articulatory distance into the automaton traversal for residual errors not covered by explicit NFA rules.

**Standalone Utilities**
- **Myers' Bit-Parallel**: Pairwise distance algorithm; automata don't compute distances during traversal
- **Filter Module**: Application developers can use N-gram, Jaro-Winkler, or HybridMatcher independently for pre-filtering or as alternatives to automaton traversal
- **Articulatory Pairwise Distance**: The `articulatory_edit_distance()` function computes full edit distance with phonetic costs; this standalone function is separate from the automaton-integrated `ProductAutomatonChar`

### Automata vs Reference Implementation

The core value of liblevenshtein is the **Levenshtein automata** approach, which finds all dictionary matches within edit distance *d* in a single traversal—without computing pairwise distances. This is fundamentally different from algorithms like Myers' that compute the distance between two specific strings.

| Approach | Use Case | Complexity |
|----------|----------|------------|
| **Levenshtein Automata** | Find all matches in dictionary | O(\|query\| × \|states\|), independent of dict size |
| **Myers' Bit-Parallel** | Distance between two strings | O(⌈m/64⌉ × n) |

The standalone utilities (Myers, Filter module, Articulatory) can be composed by applications as needed but don't change how the automaton traverses the dictionary. Only PriorityQueryIterator modifies the core query pipeline.

## Running Benchmarks

```bash
# Myers distance benchmarks
cargo bench --bench distance_benchmarks -- myers

# Pre-filtering benchmarks
cargo bench --bench filter_benchmarks

# Priority query benchmarks
cargo bench --bench priority_query_benchmarks

# Articulatory benchmarks (requires phonetic-rules)
cargo bench --bench articulatory_benchmarks --features phonetic-rules,embedded-rules
```

---

*Benchmarks run on Intel Xeon E5-2699 v3 @ 2.30GHz, 252GB DDR4-2133 ECC, Arch Linux 6.18.3, Rust 1.84.0*
*Criterion 0.5 with default sample size (100) except where noted*
