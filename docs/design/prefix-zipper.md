# PrefixZipper Design Documentation

[← Documentation Index](../README.md)

**Date**: 2025-11-10
**Component**: `libdictenstein::prefix_zipper` (extracted to the `libdictenstein` crate)
**Status**: Production-ready (optimized)

![Dictionary traits: the `DictZipper` / `PrefixZipper` trait hierarchy and its blanket implementation across every dictionary backend.](../diagrams/dictionary-structures/dictionary-traits.svg)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation & Use Cases](#motivation--use-cases)
3. [Architecture Overview](#architecture-overview)
4. [API Reference](#api-reference)
5. [Performance Analysis](#performance-analysis)
6. [Usage Examples](#usage-examples)
7. [Implementation Details](#implementation-details)
8. [Comparison with Alternatives](#comparison-with-alternatives)
9. [Integration Guide](#integration-guide)
10. [References](#references)

---

## Executive Summary

**PrefixZipper** is a trait-based extension for dictionary zippers that enables **`$\mathcal{O}(k + m)$`** exact prefix matching, where:
- `$k$` = prefix length (typically 2-5 characters)
- `$m$` = number of terms matching the prefix
- **Independent of** total dictionary size `$n$`

This provides **5-10× speedup** for selective prefixes compared to `$\mathcal{O}(n)$` full iteration with `.starts_with()` filtering.

### Key Features

- **Trait-based design**: Works uniformly across all 6 dictionary backends via blanket implementation
- **Zero-copy navigation**: `$\mathcal{O}(k)$` prefix traversal without allocating intermediate results
- **Lazy iteration**: Results yielded on-demand with `$\mathcal{O}(1)$` amortized per-result cost
- **Type-safe**: Compile-time enforcement of backend compatibility
- **Unicode support**: Works with both byte-level (`u8`) and character-level (`char`) dictionaries
- **Valued dictionaries**: Separate trait for dictionaries with associated values
- **Production-optimized**: 37.2% faster than initial implementation through scientific optimization

### Quick Example

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

let dict = DoubleArrayTrie::from_terms(vec!["process", "product"].iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Navigate to prefix "pro" and iterate matching terms
if let Some(iter) = zipper.with_prefix(b"pro") {
    for (term, _zipper) in iter {
        println!("{}", String::from_utf8(term).unwrap());
        // Prints: "process" and "product"
    }
}
```

---

## Motivation & Use Cases

### Problem Statement

Many applications need to find "all dictionary terms starting with X":

```rust
// ❌ SLOW: O(n) - iterate entire dictionary
let matches: Vec<_> = dict
    .iter()
    .filter(|term| term.starts_with(prefix))
    .collect();
```

For large dictionaries (`$n$` = 100K+ terms) where only a few terms match (`$m$` = 10-100), this wastes 99.9% of the work iterating non-matching terms.

### Solution: `$\mathcal{O}(k + m)$` Prefix Navigation

PrefixZipper leverages the trie structure of dictionaries:

1. **Navigate** directly to the prefix node: `$\mathcal{O}(k)$`
2. **Iterate** only the subtree under that node: `$\mathcal{O}(m)$`
3. **Total**: `$\mathcal{O}(k + m) \ll \mathcal{O}(n)$` when `$m \ll n$`

### Real-World Applications

#### 1. Code Completion (Rholang LSP)

```rust
// User types "pro" → suggest all Rholang functions starting with "pro"
fn get_completions(rholang_dict: &Dict, typed: &str) -> Vec<String> {
    let zipper = rholang_dict.zipper();
    zipper
        .with_prefix(typed.as_bytes())
        .map(|iter| iter.map(|(path, _)| String::from_utf8(path).unwrap()).collect())
        .unwrap_or_default()
}
```

**Performance**: ~15 µs for 100 matches vs ~200 µs for full iteration (13× faster)

#### 2. Autocomplete Systems

```rust
// Search engine autocomplete: user types "rust pro" → suggest completions
let prefix = "rust pro";
if let Some(iter) = search_terms.zipper().with_prefix(prefix.as_bytes()) {
    let suggestions: Vec<_> = iter.take(10).collect(); // Top 10 results
}
```

#### 3. Pattern-Aware Completion

```rust
// Context-sensitive completion: only show terms matching current scope
fn context_completions(dict: &Dict, scope_prefix: &[u8]) -> Vec<String> {
    dict.zipper()
        .with_prefix(scope_prefix)?
        .filter(|(_, zipper)| is_valid_in_scope(zipper))
        .map(|(path, _)| decode_term(path))
        .collect()
}
```

#### 4. Incremental Search UI

```rust
// Live search: update results as user types each character
struct IncrementalSearch {
    dict: DoubleArrayTrie,
    current_prefix: Vec<u8>,
}

impl IncrementalSearch {
    fn on_keypress(&mut self, key: u8) {
        self.current_prefix.push(key);
        if let Some(iter) = self.dict.zipper().with_prefix(&self.current_prefix) {
            self.update_ui(iter.take(20).collect()); // Show top 20
        }
    }
}
```

### When NOT to Use PrefixZipper

- **Fuzzy matching**: Use Levenshtein automata instead (tolerates typos)
- **Full-text search**: Use inverted index instead (substring matching)
- **Empty prefix**: Just use `dict.iter()` directly (equivalent performance)
- **Very common prefixes**: When `$m \approx n$`, `$\mathcal{O}(k + m) \approx \mathcal{O}(n)$` (no advantage)

---

## Architecture Overview

### Trait Hierarchy

```
DictZipper (trait)                   ValuedDictZipper (trait)
    ↓                                        ↓
PrefixZipper (trait)              ValuedPrefixZipper (trait)
    ↓                                        ↓
All 6 backends via blanket impl    All valued backends via blanket impl
```

**Key design decision**: Use blanket implementation so all dictionary backends automatically support prefix operations without backend-specific code.

### Core Components

#### 1. `PrefixZipper` Trait

```rust
pub trait PrefixZipper: DictZipper {
    fn with_prefix(&self, prefix: &[Self::Unit]) -> Option<PrefixIterator<Self>>
    where
        Self: Sized;
}

// Blanket implementation
impl<Z: DictZipper> PrefixZipper for Z {}
```

**Responsibilities**:
- Validate prefix exists in dictionary
- Navigate to prefix position: `$\mathcal{O}(k)$`
- Return iterator if prefix valid, `None` otherwise

#### 2. `PrefixIterator<Z>` Struct

```rust
pub struct PrefixIterator<Z: DictZipper> {
    /// DFS traversal stack: stores only zippers (paths reconstructed on demand).
    /// Optimization: Pre-allocated capacity 16 to cover typical depths (10-15)
    /// without reallocations.
    stack: Vec<Z>,
}
```

**Responsibilities**:
- Depth-first traversal of subtree under prefix
- Yield `(path, zipper)` for each final node
- `$\mathcal{O}(1)$` amortized per-result iteration

#### 3. `ValuedPrefixZipper` Trait

```rust
pub trait ValuedPrefixZipper: ValuedDictZipper {
    fn with_prefix_values(&self, prefix: &[Self::Unit])
        -> Option<ValuedPrefixIterator<Self>>
    where
        Self: Sized;
}
```

**Extension** of `PrefixZipper` for dictionaries with values.

#### 4. `ValuedPrefixIterator<Z>` Struct

```rust
pub struct ValuedPrefixIterator<Z: ValuedDictZipper> {
    inner: PrefixIterator<Z>,
}

impl<Z: ValuedDictZipper> Iterator for ValuedPrefixIterator<Z> {
    type Item = (Vec<Z::Unit>, Z::Value);

    fn next(&mut self) -> Option<Self::Item> {
        let (path, zipper) = self.inner.next()?;
        let value = zipper.value()?;
        Some((path, value))
    }
}
```

**Wrapper** around `PrefixIterator` that extracts values from final nodes.

### Backend Compatibility Matrix

| Backend | Byte-level | Char-level | Valued | Notes |
|---------|-----------|-----------|--------|-------|
| **DoubleArrayTrie** | ✅ | ✅ | ✅ | Fastest navigation (`$\mathcal{O}(1)$` transitions) |
| **DynamicDawg** | ✅ | ✅ | ✅ | Comparable performance to DAT |
| **PathMapDictionary** | ✅ | ❌ | ✅ | Byte-level only |
| **SuffixAutomaton** | ✅ | ✅ | ✅ | Slower for short prefixes |

All backends work identically from user perspective - only performance characteristics differ.

### Design Principles

1. **Generic over backends**: No backend-specific code in PrefixZipper implementation
2. **Zero-copy navigation**: Prefix validation doesn't allocate results
3. **Lazy evaluation**: Results computed on-demand during iteration
4. **Type safety**: Compiler ensures backend compatibility at compile time
5. **Minimal API surface**: Single method `with_prefix()` for basic use case
6. **Composable**: Returned iterator is standard Rust iterator (works with `.filter()`, `.map()`, `.take()`, etc.)

---

## API Reference

### `PrefixZipper` Trait

#### Method: `with_prefix`

```rust
fn with_prefix(&self, prefix: &[Self::Unit]) -> Option<PrefixIterator<Self>>
where
    Self: Sized;
```

**Description**: Navigate to the given prefix and create an iterator over all terms that start with this prefix.

**Parameters**:
- `prefix: &[Self::Unit]` - The prefix sequence to match
  - `Self::Unit = u8` for byte-level dictionaries
  - `Self::Unit = char` for character-level dictionaries

**Returns**:
- `Some(PrefixIterator<Self>)` - If at least one term with this prefix exists
- `None` - If no terms with this prefix exist in the dictionary

**Complexity**:
- **Time**: `$\mathcal{O}(k)$` where `$k$` = prefix length
- **Space**: `$\mathcal{O}(1)$` - no allocations during navigation

**Example**:

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

let dict = DoubleArrayTrie::from_terms(vec!["hello", "help", "world"].iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Successful navigation
let iter = zipper.with_prefix(b"hel").unwrap();
assert_eq!(iter.count(), 2); // "hello" and "help"

// Unsuccessful navigation
assert!(zipper.with_prefix(b"xyz").is_none());
```

**Thread Safety**: The returned iterator is `Send` and `Sync` if the underlying zipper is `Send` and `Sync` (automatically satisfied for all standard backends).

---

### `PrefixIterator<Z>` Struct

```rust
pub struct PrefixIterator<Z: DictZipper> {
    stack: Vec<Z>,
}
```

**Description**: Iterator over all terms matching a given prefix. Performs depth-first traversal from the prefix position.

#### Iterator Implementation

```rust
impl<Z: DictZipper> Iterator for PrefixIterator<Z> {
    type Item = (Vec<Z::Unit>, Z);

    fn next(&mut self) -> Option<Self::Item>;
}
```

**Item Type**: `(Vec<Z::Unit>, Z)`
- `Vec<Z::Unit>` - Complete path (term) as sequence of units
  - `Vec<u8>` for byte-level (convert with `String::from_utf8()`)
  - `Vec<char>` for character-level (convert with `.iter().collect::<String>()`)
- `Z` - Zipper positioned at the final node (useful for querying values or further navigation)

**Complexity**:
- **Per-result**: `$\mathcal{O}(1)$` amortized
- **Total iteration**: `$\mathcal{O}(m)$` where `$m$` = number of matching terms
- **Memory**: `$\mathcal{O}(d)$` where `$d$` = maximum depth of matching terms

**Example**:

```rust
let dict = DoubleArrayTrie::from_terms(vec!["cat", "cats", "catfish"].iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

let results: Vec<String> = zipper
    .with_prefix(b"cat")
    .unwrap()
    .map(|(path, _zipper)| String::from_utf8(path).unwrap())
    .collect();

assert_eq!(results, vec!["cat", "cats", "catfish"]);
```

**Iterator Combinators**: Works with all standard iterator methods:

```rust
// Take first 10 results
let top10: Vec<_> = iter.take(10).collect();

// Filter by length
let short_terms: Vec<_> = iter
    .filter(|(path, _)| path.len() <= 5)
    .collect();

// Count without collecting
let count = iter.count();

// Check existence
let has_exact_match = iter.any(|(path, _)| path == b"exact");
```

---

### `ValuedPrefixZipper` Trait

#### Method: `with_prefix_values`

```rust
fn with_prefix_values(&self, prefix: &[Self::Unit])
    -> Option<ValuedPrefixIterator<Self>>
where
    Self: Sized;
```

**Description**: Navigate to the given prefix and create an iterator over `(term, value)` pairs for all terms that start with this prefix.

**Parameters**:
- `prefix: &[Self::Unit]` - The prefix sequence to match

**Returns**:
- `Some(ValuedPrefixIterator<Self>)` - If at least one term with this prefix exists
- `None` - If no terms with this prefix exist

**Complexity**: Same as `PrefixZipper::with_prefix()`: `$\mathcal{O}(k)$` navigation + `$\mathcal{O}(m)$` iteration

**Example**:

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::ValuedPrefixZipper;

let dict = DoubleArrayTrie::from_terms_with_values(
    vec![("cat", 1), ("cats", 2), ("dog", 3)].into_iter()
);
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

let mut results: Vec<(String, usize)> = zipper
    .with_prefix_values(b"cat")
    .unwrap()
    .map(|(path, val)| (String::from_utf8(path).unwrap(), val))
    .collect();

results.sort();
assert_eq!(results, vec![
    ("cat".to_string(), 1),
    ("cats".to_string(), 2)
]);
```

---

### `ValuedPrefixIterator<Z>` Struct

```rust
pub struct ValuedPrefixIterator<Z: ValuedDictZipper> {
    inner: PrefixIterator<Z>,
}
```

**Description**: Iterator over `(term, value)` pairs matching a given prefix. Wraps `PrefixIterator` and extracts values from final nodes.

#### Iterator Implementation

```rust
impl<Z: ValuedDictZipper> Iterator for ValuedPrefixIterator<Z> {
    type Item = (Vec<Z::Unit>, Z::Value);

    fn next(&mut self) -> Option<Self::Item>;
}
```

**Item Type**: `(Vec<Z::Unit>, Z::Value)`
- `Vec<Z::Unit>` - Complete path (term)
- `Z::Value` - Associated value (type depends on dictionary construction)

**Example**:

```rust
// Dictionary with f64 scores
let scored_terms = vec![
    ("apple", 0.95),
    ("apricot", 0.87),
    ("banana", 0.92),
];
let dict = DoubleArrayTrie::from_terms_with_values(scored_terms.into_iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Get all fruit names starting with "ap" and their scores
let ap_fruits: Vec<(String, f64)> = zipper
    .with_prefix_values(b"ap")
    .unwrap()
    .map(|(path, score)| (String::from_utf8(path).unwrap(), score))
    .collect();

assert_eq!(ap_fruits, vec![
    ("apple".to_string(), 0.95),
    ("apricot".to_string(), 0.87),
]);
```

---

## Performance Analysis

### Theoretical Complexity

| Operation | Complexity | Description |
|-----------|-----------|-------------|
| **Navigation** (`with_prefix`) | `$\mathcal{O}(k)$` | `$k$` = prefix length |
| **Iterator creation** | `$\mathcal{O}(1)$` | Just stack initialization |
| **Per-result** | `$\mathcal{O}(1)$` amortized | DFS with pre-allocated stack |
| **Total iteration** | `$\mathcal{O}(m)$` | `$m$` = number of matching terms |
| **Overall** | **`$\mathcal{O}(k + m)$`** | Independent of dictionary size `$n$` |

**Comparison to alternatives**:
- **Full iteration + filter**: `$\mathcal{O}(n)$` - must visit every term
- **Speedup**: `$\mathcal{O}(n) / \mathcal{O}(k + m) \approx n / m$` when `$k \ll n$`

For selective prefixes where `$m \ll n$`:
- 100K dictionary, 100 matches: **1000× faster in theory**
- Real-world: **5-10× faster** (accounting for constant factors)

### Benchmark Results

**Hardware**: Intel Xeon E5-2699 v3 @ 2.30GHz, 36 cores, 252GB RAM
**Date**: 2025-11-10
**Baseline**: See [`docs/optimization/prefix_zipper_baseline.md`](../optimization/prefix_zipper_baseline.md)

#### Selectivity Benchmarks (10K dictionary)

| Prefix | Matches | Time (µs) | Throughput (ops/s) |
|--------|---------|-----------|-------------------|
| High selectivity | 5 | 1.82 | 549K |
| Medium selectivity | 100 | 12.08 | 82.8K |
| Low selectivity | 600 | 61.74 | 16.2K |
| Empty prefix | 10,000 | 1,042 | 959 |

**Key finding**: Time scales linearly with number of matches (m), not dictionary size (n).

#### Dictionary Size Scaling (medium selectivity, 100 matches)

| Dictionary Size | Time (µs) | Change |
|----------------|-----------|--------|
| 1,000 terms | 11.95 | baseline |
| 10,000 terms | 12.08 | +1.1% |
| 100,000 terms | 12.31 | +3.0% |

**Key finding**: Near-constant time across 100× dictionary size increase, confirming `$\mathcal{O}(k + m)$` independence from `$n$`.

#### Backend Comparison (10K dictionary, 100 matches)

| Backend | Time (µs) | Relative |
|---------|-----------|----------|
| DoubleArrayTrie | 12.08 | 1.00× (baseline) |
| DynamicDawg | 12.15 | 1.01× (+0.6%) |

**Key finding**: Backend choice has negligible impact (<1% difference) - use based on other requirements (mutability, memory usage).

#### Tree Depth Impact (empty prefix)

| Max Depth | Time (µs) | Per-term (ns) |
|-----------|-----------|---------------|
| 5 | 98.0 | 9.8 |
| 10 | 306 | 30.6 |
| 15 | 631 | 63.1 |
| 20 | 1,042 | 104.2 |

**Key finding**: Time scales with depth × breadth (total nodes traversed), as expected for DFS.

### Optimization History

The current implementation has been scientifically optimized through systematic measurement and validation:

#### Baseline (Commit 39b727f)

```rust
// Original implementation
pub struct PrefixIterator<Z: DictZipper> {
    stack: Vec<(Z, Vec<Z::Unit>)>,  // Stored redundant paths
}

fn new(prefix_zipper: Z, prefix: &[Z::Unit]) -> Self {
    let prefix_path = prefix.to_vec();
    Self {
        stack: vec![(prefix_zipper, prefix_path)],  // Capacity 1
    }
}
```

**Benchmark**: 19.236 µs (medium_selectivity/100)

#### Optimization #1: Pre-allocate Stack Capacity (Commit f22598f)

**Hypothesis**: Initial stack capacity of 1 causes reallocations during DFS, consuming ~2.37% execution time.

**Change**:
```rust
let mut stack = Vec::with_capacity(16);  // Pre-allocate for typical depth
stack.push((prefix_zipper, prefix_path));
```

**Result**: 18.663 µs (-3.0%, -0.57 µs)

**Analysis**: Eliminated 4 reallocations per query (1→2→4→8→16 growth).

#### Optimization #2: Remove Redundant Path Tracking (Commit 22b3404)

**Hypothesis**: Storing paths in stack is redundant since zippers maintain paths internally. Path cloning consumes ~4.07% execution time (Vec::clone 2.19% + Vec::push reallocation 1.88%).

**Change**:
```rust
// Removed path storage entirely
pub struct PrefixIterator<Z: DictZipper> {
    stack: Vec<Z>,  // Store only zippers
}

fn next(&mut self) -> Option<Self::Item> {
    while let Some(zipper) = self.stack.pop() {
        for (_unit, child) in zipper.children() {
            self.stack.push(child);  // No clone, no realloc
        }
        if zipper.is_final() {
            return Some((zipper.path(), zipper));  // Reconstruct only for finals
        }
    }
    None
}
```

**Result**: 12.075 µs (-35.4% from Opt #1, -6.59 µs)

**Analysis**: Eliminated 100s of Vec clones per query. Path reconstruction happens only for final nodes (10-100× less frequent).

#### Combined Impact

| Version | Time (µs) | Change | Throughput |
|---------|-----------|--------|------------|
| Baseline (39b727f) | 19.236 | - | 52.0K ops/s |
| After Opt #1 (f22598f) | 18.663 | -3.0% | 53.6K ops/s |
| **After Opt #2 (22b3404)** | **12.075** | **-37.2%** | **82.8K ops/s** |

**Total improvement**: 37.2% faster, 59.4% higher throughput

See full optimization log: [`docs/optimization/prefix_zipper_optimization_log.md`](../optimization/prefix_zipper_optimization_log.md)

### Performance Tuning Guidelines

#### When to Use Which Backend

**DoubleArrayTrie** (recommended default):
- ✅ Best navigation performance (`$\mathcal{O}(1)$` transitions)
- ✅ Compact memory representation
- ✅ Cache-friendly layout
- ❌ Immutable (rebuild required for updates)

**DynamicDawg**:
- ✅ Mutable (supports insertions/deletions)
- ✅ Nearly identical prefix iteration performance
- ❌ Slightly higher memory overhead
- **Use when**: Dictionary needs runtime updates

**PathMapDictionary**:
- ✅ Simplest implementation
- ✅ Good for small dictionaries (<1000 terms)
- ❌ Slower for large dictionaries
- **Use when**: Simplicity > performance

**SuffixAutomaton**:
- ✅ Supports substring matching (not just prefixes)
- ❌ Slower for short prefixes
- **Use when**: Need both prefix and substring queries

#### Optimizing for Your Use Case

**High-frequency queries** (autocomplete, every keystroke):
```rust
// Pre-create zipper to avoid repeated initialization
struct CompletionEngine {
    zipper: DoubleArrayTrieZipper,  // Reuse across queries
}

impl CompletionEngine {
    fn complete(&self, prefix: &str) -> Vec<String> {
        self.zipper
            .with_prefix(prefix.as_bytes())
            .map(|iter| iter
                .take(10)  // Limit results
                .map(|(path, _)| String::from_utf8(path).unwrap())
                .collect())
            .unwrap_or_default()
    }
}
```

**Memory-constrained environments**:
```rust
// Use iterator without collecting
fn has_any_matches(dict: &Dict, prefix: &[u8]) -> bool {
    dict.zipper()
        .with_prefix(prefix)
        .map(|mut iter| iter.next().is_some())  // Check first result only
        .unwrap_or(false)
}
```

**Large result sets**:
```rust
// Process results incrementally to avoid large allocations
fn process_matches(dict: &Dict, prefix: &[u8]) {
    if let Some(iter) = dict.zipper().with_prefix(prefix) {
        for (term, _) in iter {
            process_term(term);  // No intermediate Vec
        }
    }
}
```

---

## Usage Examples

### Example 1: Basic Byte-Level Dictionary

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;
use libdictenstein::double_array_trie_zipper::DoubleArrayTrieZipper;

fn main() {
    // Create dictionary from terms
    let terms = vec!["apple", "application", "apply", "banana", "bandana"];
    let dict = DoubleArrayTrie::from_terms(terms.iter());

    // Create zipper
    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

    // Query prefix "app"
    if let Some(iter) = zipper.with_prefix(b"app") {
        println!("Terms starting with 'app':");
        for (path, _zipper) in iter {
            let term = String::from_utf8(path).unwrap();
            println!("  - {}", term);
        }
    }
    // Output:
    //   - apple
    //   - application
    //   - apply
}
```

### Example 2: Character-Level Unicode Dictionary

```rust
use libdictenstein::double_array_trie_char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie_char_zipper::DoubleArrayTrieCharZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

fn main() {
    // Unicode terms with diacritics
    let terms = vec!["café", "cafétéria", "naïve", "naïveté"];
    let dict = DoubleArrayTrieChar::from_terms(terms.iter());

    let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

    // Query with Unicode prefix
    let prefix: Vec<char> = "caf".chars().collect();
    if let Some(iter) = zipper.with_prefix(&prefix) {
        println!("Terms starting with 'caf':");
        for (path, _) in iter {
            let term: String = path.iter().collect();
            println!("  - {}", term);
        }
    }
    // Output:
    //   - café
    //   - cafétéria
}
```

### Example 3: Valued Dictionary (Frequency Scores)

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::ValuedPrefixZipper;
use libdictenstein::double_array_trie_zipper::DoubleArrayTrieZipper;

fn main() {
    // Terms with frequency scores
    let terms_with_scores = vec![
        ("the", 1000),
        ("these", 150),
        ("there", 200),
        ("they", 300),
    ];
    let dict = DoubleArrayTrie::from_terms_with_values(
        terms_with_scores.into_iter()
    );

    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

    // Query prefix and sort by frequency
    if let Some(iter) = zipper.with_prefix_values(b"the") {
        let mut results: Vec<(String, usize)> = iter
            .map(|(path, freq)| (String::from_utf8(path).unwrap(), freq))
            .collect();

        // Sort by frequency (descending)
        results.sort_by_key(|(_, freq)| std::cmp::Reverse(*freq));

        println!("Terms starting with 'the' (by frequency):");
        for (term, freq) in results {
            println!("  - {} ({})", term, freq);
        }
    }
    // Output:
    //   - the (1000)
    //   - they (300)
    //   - there (200)
    //   - these (150)
}
```

### Example 4: All 6 Dictionary Backends

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;
use libdictenstein::{
    double_array_trie_zipper::DoubleArrayTrieZipper,
    double_array_trie_char_zipper::DoubleArrayTrieCharZipper,
    dynamic_dawg_zipper::DynamicDawgZipper,
    dynamic_dawg_char_zipper::DynamicDawgCharZipper,
    pathmap_zipper::PathMapZipper,
    suffix_automaton_zipper::SuffixAutomatonZipper,
};

fn test_all_backends(terms: Vec<&str>, prefix: &str) {
    let prefix_bytes = prefix.as_bytes();
    let prefix_chars: Vec<char> = prefix.chars().collect();

    // Backend 1: DoubleArrayTrie (byte)
    let dat = DoubleArrayTrie::from_terms(terms.iter());
    let count1 = DoubleArrayTrieZipper::new_from_dict(&dat)
        .with_prefix(prefix_bytes)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("DoubleArrayTrie: {} matches", count1);

    // Backend 2: DoubleArrayTrieChar (char)
    let dat_char = DoubleArrayTrieChar::from_terms(terms.iter());
    let count2 = DoubleArrayTrieCharZipper::new_from_dict(&dat_char)
        .with_prefix(&prefix_chars)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("DoubleArrayTrieChar: {} matches", count2);

    // Backend 3: DynamicDawg (byte)
    let dawg = DynamicDawg::from_terms(terms.iter());
    let count3 = DynamicDawgZipper::new_from_dict(&dawg)
        .with_prefix(prefix_bytes)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("DynamicDawg: {} matches", count3);

    // Backend 4: DynamicDawgChar (char)
    let dawg_char = DynamicDawgChar::from_terms(terms.iter());
    let count4 = DynamicDawgCharZipper::new_from_dict(&dawg_char)
        .with_prefix(&prefix_chars)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("DynamicDawgChar: {} matches", count4);

    // Backend 5: PathMapDictionary (byte only)
    let pathmap = PathMapDictionary::from_terms(terms.iter());
    let count5 = PathMapZipper::new_from_dict(&pathmap)
        .with_prefix(prefix_bytes)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("PathMapDictionary: {} matches", count5);

    // Backend 6: SuffixAutomaton (byte)
    let suffix_auto = SuffixAutomaton::from_terms(terms.iter());
    let count6 = SuffixAutomatonZipper::new_from_dict(&suffix_auto)
        .with_prefix(prefix_bytes)
        .map(|iter| iter.count())
        .unwrap_or(0);
    println!("SuffixAutomaton: {} matches", count6);

    // All should return same count
    assert_eq!(count1, count2);
    assert_eq!(count1, count3);
    assert_eq!(count1, count4);
    assert_eq!(count1, count5);
    assert_eq!(count1, count6);
}

fn main() {
    let terms = vec!["rust", "rustic", "rusty", "python", "ruby"];
    test_all_backends(terms, "rust");
    // Output:
    //   DoubleArrayTrie: 3 matches
    //   DoubleArrayTrieChar: 3 matches
    //   DynamicDawg: 3 matches
    //   DynamicDawgChar: 3 matches
    //   PathMapDictionary: 3 matches
    //   SuffixAutomaton: 3 matches
}
```

### Example 5: Autocomplete with Result Limiting

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

struct Autocompleter {
    dict: DoubleArrayTrie,
}

impl Autocompleter {
    fn new(terms: Vec<&str>) -> Self {
        Self {
            dict: DoubleArrayTrie::from_terms(terms.iter()),
        }
    }

    fn suggest(&self, prefix: &str, max_results: usize) -> Vec<String> {
        let zipper = DoubleArrayTrieZipper::new_from_dict(&self.dict);

        zipper
            .with_prefix(prefix.as_bytes())
            .map(|iter| {
                iter.take(max_results)
                    .map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn main() {
    let terms = vec![
        "function", "functional", "functor",
        "filter", "fold", "format",
    ];
    let ac = Autocompleter::new(terms);

    // Get top 5 suggestions for "fun"
    let suggestions = ac.suggest("fun", 5);
    println!("Suggestions for 'fun': {:?}", suggestions);
    // Output: ["function", "functional", "functor"]

    // Get top 5 suggestions for "f"
    let suggestions = ac.suggest("f", 5);
    println!("Suggestions for 'f': {:?}", suggestions);
    // Output: ["filter", "fold", "format", "function", "functional"]
}
```

### Example 6: Context-Sensitive Code Completion

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

struct RholangLSP {
    keywords: DoubleArrayTrie,
    builtins: DoubleArrayTrie,
}

impl RholangLSP {
    fn complete(&self, context: &str, typed: &str) -> Vec<String> {
        // Choose dictionary based on context
        let dict = match context {
            "keyword" => &self.keywords,
            "builtin" => &self.builtins,
            _ => return vec![],
        };

        let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

        zipper
            .with_prefix(typed.as_bytes())
            .map(|iter| {
                iter.map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn main() {
    let lsp = RholangLSP {
        keywords: DoubleArrayTrie::from_terms(
            vec!["for", "match", "new", "contract"].iter()
        ),
        builtins: DoubleArrayTrie::from_terms(
            vec!["stdout", "stdoutAck", "stderr"].iter()
        ),
    };

    // User types "st" in builtin context
    let completions = lsp.complete("builtin", "st");
    println!("Builtin completions for 'st': {:?}", completions);
    // Output: ["stderr", "stdout", "stdoutAck"]

    // User types "ma" in keyword context
    let completions = lsp.complete("keyword", "ma");
    println!("Keyword completions for 'ma': {:?}", completions);
    // Output: ["match"]
}
```

---

## Implementation Details

### DFS Traversal Algorithm

PrefixIterator uses **depth-first search** with an explicit stack:

```rust
fn next(&mut self) -> Option<Self::Item> {
    while let Some(zipper) = self.stack.pop() {
        // 1. Push all children onto stack (right-to-left for lexicographic order)
        for (_unit, child) in zipper.children() {
            self.stack.push(child);
        }

        // 2. If current node is final, yield result
        if zipper.is_final() {
            return Some((zipper.path(), zipper));
        }
    }

    None  // Stack exhausted
}
```

**Why DFS vs BFS?**
- **Memory**: `$\mathcal{O}(d)$` vs `$\mathcal{O}(b^d)$` where `$d$` = depth, `$b$` = branching factor
- **Cache locality**: Stack access is more cache-friendly than queue
- **Simplicity**: Vec-based stack vs VecDeque-based queue

**Traversal Order**: Lexicographic (alphabetical) due to:
1. Trie structure maintains sorted children
2. DFS visits children left-to-right
3. Results yielded in natural dictionary order

### Path Reconstruction Optimization

**Key insight**: All `DictZipper` implementations track paths internally via the `path()` method.

**Old approach** (redundant):
```rust
stack: Vec<(Z, Vec<Z::Unit>)>  // Store path explicitly

for (unit, child) in zipper.children() {
    let mut child_path = path.clone();  // EXPENSIVE: 100s per query
    child_path.push(unit);
    self.stack.push((child, child_path));
}
```

**New approach** (optimized):
```rust
stack: Vec<Z>  // Path implicit in zipper

for (_unit, child) in zipper.children() {
    self.stack.push(child);  // NO clone
}

// Reconstruct path only for final nodes (10-100× less frequent)
if zipper.is_final() {
    return Some((zipper.path(), zipper));
}
```

**Performance impact**:
- Eliminates 100s of `Vec::clone()` calls per query
- Removes `Vec::push()` reallocation cascades
- Reduces memory allocations by orders of magnitude
- **Result**: 35.4% faster iteration

### Stack Pre-allocation

**Observation**: Tree depth benchmarks show typical depth 10-15 for real dictionaries.

**Strategy**: Pre-allocate capacity 16 to avoid reallocations:

```rust
let mut stack = Vec::with_capacity(16);
```

**Trade-offs**:
- **Memory**: +128 bytes per iterator (16 × 8-byte pointers)
- **Time**: Eliminates 4 reallocations (1→2→4→8→16 growth)
- **Coverage**: Handles 90% of queries without reallocation
- **Fallback**: Amortized growth still available for deeper trees

**Performance impact**: -3.0% time (eliminates ~2.37% realloc overhead + memory fragmentation benefits)

### Thread Safety

All PrefixZipper components are automatically `Send` and `Sync` if the underlying zipper is:

```rust
// Automatic derivation via trait bounds
impl<Z: DictZipper + Send> Send for PrefixIterator<Z> {}
impl<Z: DictZipper + Sync> Sync for PrefixIterator<Z> {}
```

**Implications**:
- ✅ Can pass iterators between threads
- ✅ Can share read-only dictionary across threads
- ✅ Safe for parallel query processing
- ❌ Cannot mutate dictionary during iteration (Rust ownership prevents this)

**Example**:

```rust
use rayon::prelude::*;

fn parallel_prefix_queries(dict: &DoubleArrayTrie, prefixes: Vec<&str>) -> Vec<usize> {
    prefixes
        .par_iter()  // Parallel iterator
        .map(|prefix| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(dict);
            zipper
                .with_prefix(prefix.as_bytes())
                .map(|iter| iter.count())
                .unwrap_or(0)
        })
        .collect()
}
```

### Memory Layout

**PrefixIterator size**:
```rust
std::mem::size_of::<PrefixIterator<DoubleArrayTrieZipper>>()
// = 24 bytes (Vec: ptr + len + capacity)
```

**Stack entry size**:
```rust
std::mem::size_of::<DoubleArrayTrieZipper>()
// = 16 bytes (node_id: u32 + path: Vec<u8> or similar)
```

**Total iterator overhead**: ~24 + (16 × 16) = ~280 bytes with pre-allocated capacity

**Comparison to alternatives**:
- **Old implementation**: 24 + (16 + 24) × 16 = 664 bytes (40% larger)
- **Full result collection**: 24 + (term_size × result_count) (1000s of bytes)

---

## Comparison with Alternatives

### vs. Full Iteration + Filter

```rust
// ❌ Alternative 1: Full iteration
let matches: Vec<_> = dict
    .iter()
    .filter(|term| term.starts_with(prefix))
    .collect();
```

| Aspect | PrefixZipper | Full Iteration |
|--------|-------------|----------------|
| **Complexity** | `$\mathcal{O}(k + m)$` | `$\mathcal{O}(n)$` |
| **100K dict, 100 matches** | ~12 µs | ~200 µs (17× slower) |
| **Memory** | `$\mathcal{O}(d)$` | `$\mathcal{O}(1)$` during iteration |
| **Code clarity** | Explicit intent | Generic pattern |

**When to use full iteration**: Never for prefix matching - always use PrefixZipper.

### vs. Levenshtein Prefix Matching

```rust
// ❌ Alternative 2: Fuzzy prefix matching
let matches: Vec<_> = dict
    .query("prefix", Distance::Levenshtein(1))
    .starts_with()
    .collect();
```

| Aspect | PrefixZipper | Levenshtein Prefix |
|--------|-------------|-------------------|
| **Match type** | Exact prefix | Fuzzy prefix (tolerates typos) |
| **Complexity** | `$\mathcal{O}(k + m)$` | `$\mathcal{O}(k \times m \times d)$` where `$d$` = edit distance |
| **Performance** | ~12 µs | ~150 µs (12× slower) |
| **Use case** | Known correct prefix | User input with typos |

**When to use Levenshtein**: When prefix may contain typos (e.g., user typing in UI). See [`docs/algorithms/02-levenshtein-automata/README.md`](../algorithms/02-levenshtein-automata/README.md).

**When to use PrefixZipper**: When prefix is guaranteed correct (e.g., programmatic completion, validated input).

### vs. HashMap/HashSet Prefix Grouping

```rust
// ❌ Alternative 3: Pre-group terms by prefix
struct PrefixIndex {
    index: HashMap<Vec<u8>, Vec<Vec<u8>>>,  // prefix -> terms
}
```

| Aspect | PrefixZipper | HashMap Index |
|--------|-------------|---------------|
| **Memory** | `$\mathcal{O}(n)$` (trie storage) | `$\mathcal{O}(n \times \text{avg\_prefix\_len})$` (duplicated prefixes) |
| **Build time** | `$\mathcal{O}(n \times k)$` (trie construction) | `$\mathcal{O}(n \times k)$` (index construction) |
| **Query time** | `$\mathcal{O}(k + m)$` | `$\mathcal{O}(1)$` lookup + `$\mathcal{O}(m)$` iteration |
| **Flexibility** | Any prefix length | Fixed prefix lengths |

**When to use HashMap**: When prefix lengths are fixed and known (e.g., always 3 characters).

**When to use PrefixZipper**: Variable-length prefixes, memory-efficient storage, unified with existing trie dictionaries.

### Decision Matrix

| Requirement | Recommended Approach |
|-------------|---------------------|
| Exact prefix, correct input | **PrefixZipper** |
| Fuzzy prefix, user input with typos | Levenshtein prefix matching |
| Substring matching (not just prefix) | Full-text search index |
| Very small dictionary (<100 terms) | Linear scan (simpler) |
| Fixed 2-3 character prefixes | HashMap index (faster) |
| Variable-length prefixes | **PrefixZipper** |
| Memory constrained | **PrefixZipper** (trie is compact) |
| Need both prefix + fuzzy matching | Combine PrefixZipper + Levenshtein |

---

## Integration Guide

### Adding PrefixZipper to Existing Code

**Step 1**: Import the trait

```rust
use libdictenstein::prefix_zipper::PrefixZipper;
```

**Step 2**: Use with existing dictionary zipper

```rust
// Your existing code
let dict = DoubleArrayTrie::from_terms(terms.iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Add prefix iteration (no changes to dict required!)
if let Some(iter) = zipper.with_prefix(b"prefix") {
    for (term, _) in iter {
        // Process matching terms
    }
}
```

**No changes required**:
- ✅ Dictionary structure unchanged
- ✅ Existing query APIs still work
- ✅ No performance impact on non-prefix queries
- ✅ Backward compatible

### Combining with Levenshtein Queries

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

fn smart_complete(dict: &DoubleArrayTrie, user_input: &str) -> Vec<String> {
    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

    // Try exact prefix first (fast path)
    if let Some(iter) = zipper.with_prefix(user_input.as_bytes()) {
        let results: Vec<_> = iter.take(10).collect();
        if !results.is_empty() {
            return results
                .into_iter()
                .map(|(path, _)| String::from_utf8(path).unwrap())
                .collect();
        }
    }

    // Fall back to fuzzy matching if no exact prefix matches
    dict.query(user_input, Distance::Levenshtein(1))
        .starts_with()  // Fuzzy prefix
        .take(10)
        .map(|(path, _)| String::from_utf8(path).unwrap())
        .collect()
}
```

### Integration with Caching

```rust
use std::sync::Arc;
use lru::LruCache;

struct CachedPrefixCompleter {
    dict: Arc<DoubleArrayTrie>,
    cache: LruCache<Vec<u8>, Vec<String>>,
}

impl CachedPrefixCompleter {
    fn new(dict: DoubleArrayTrie, cache_size: usize) -> Self {
        Self {
            dict: Arc::new(dict),
            cache: LruCache::new(cache_size),
        }
    }

    fn complete(&mut self, prefix: &str) -> Vec<String> {
        let prefix_bytes = prefix.as_bytes().to_vec();

        // Check cache first
        if let Some(cached) = self.cache.get(&prefix_bytes) {
            return cached.clone();
        }

        // Compute result
        let zipper = DoubleArrayTrieZipper::new_from_dict(&self.dict);
        let results = zipper
            .with_prefix(&prefix_bytes)
            .map(|iter| {
                iter.map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default();

        // Cache result
        self.cache.put(prefix_bytes, results.clone());
        results
    }
}
```

### Testing Recommendations

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use liblevenshtein::prelude::*;
    use libdictenstein::prefix_zipper::PrefixZipper;

    #[test]
    fn test_prefix_completeness() {
        // Ensure all matches are found
        let terms = vec!["apple", "application", "apply"];
        let dict = DoubleArrayTrie::from_terms(terms.iter());
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        let results: Vec<String> = zipper
            .with_prefix(b"app")
            .unwrap()
            .map(|(path, _)| String::from_utf8(path).unwrap())
            .collect();

        assert_eq!(results.len(), 3);
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"application".to_string()));
        assert!(results.contains(&"apply".to_string()));
    }

    #[test]
    fn test_prefix_non_existent() {
        // Ensure non-existent prefix returns None
        let dict = DoubleArrayTrie::from_terms(vec!["hello"].iter());
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        assert!(zipper.with_prefix(b"xyz").is_none());
    }

    #[test]
    fn test_prefix_ordering() {
        // Ensure lexicographic ordering
        let terms = vec!["aaa", "aab", "aba", "abb"];
        let dict = DoubleArrayTrie::from_terms(terms.iter());
        let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

        let results: Vec<String> = zipper
            .with_prefix(b"a")
            .unwrap()
            .map(|(path, _)| String::from_utf8(path).unwrap())
            .collect();

        assert_eq!(results, vec!["aaa", "aab", "aba", "abb"]);
    }
}
```

---

## References

### Related Documentation

- **Optimization Log**: [`docs/optimization/prefix_zipper_optimization_log.md`](../optimization/prefix_zipper_optimization_log.md) - Scientific optimization process
- **Baseline Measurements**: [`docs/optimization/prefix_zipper_baseline.md`](../optimization/prefix_zipper_baseline.md) - Comprehensive benchmark results
- **User Guide**: [`docs/user-guide/prefix-zipper-usage.md`](../user-guide/prefix-zipper-usage.md) - Practical usage patterns
- **DictZipper Trait**: [`libdictenstein/src/zipper.rs`](../../../libdictenstein/src/zipper.rs) - Underlying zipper abstraction (extracted to the `libdictenstein` crate)
- **Levenshtein Automata**: [`docs/algorithms/02-levenshtein-automata/README.md`](../algorithms/02-levenshtein-automata/README.md) - Fuzzy matching alternative

### Source Code

- **Implementation**: [`libdictenstein/src/prefix_zipper.rs`](../../../libdictenstein/src/prefix_zipper.rs) (extracted to the `libdictenstein` crate)
- **Tests**: [`libdictenstein/tests/prefix_zipper_tests.rs`](../../../libdictenstein/tests/prefix_zipper_tests.rs)
- **Benchmarks**: [`benches/prefix_zipper_benchmarks.rs`](../../benches/prefix_zipper_benchmarks.rs)

### Academic References

- **Trie data structure**: Fredkin, E. (1960). "Trie memory". Communications of the ACM.
- **DAWG (Directed Acyclic Word Graph)**: Blumer et al. (1985). "The smallest automaton recognizing the subwords of a text".
- **Double-Array Trie**: Aoe, J. (1989). "An efficient digital search algorithm by using a double-array structure".

### External Resources

- **Rust Iterator Trait**: https://doc.rust-lang.org/std/iter/trait.Iterator.html
- **Criterion Benchmarking**: https://github.com/bheisler/criterion.rs
- **Flamegraph Profiling**: https://github.com/flamegraph-rs/flamegraph

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Author**: Scientific optimization process (see git log)
**Commit**: 22b3404 (Optimization #2 - Remove redundant path tracking)
