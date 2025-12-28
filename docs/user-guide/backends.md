# Dictionary Backend Guide

**Version**: 0.8.0
**Last Updated**: 2025-12-28

This guide explains the different dictionary backends available in liblevenshtein-rust and how to choose the right one for your use case.

## Overview

liblevenshtein-rust uses a trait-based design that allows multiple dictionary implementations with the same fuzzy matching interface. Each backend has different trade-offs in terms of:

- **Construction time**: How long it takes to build the dictionary
- **Query performance**: How fast fuzzy searches are
- **Memory usage**: RAM footprint
- **Update support**: Whether the dictionary can be modified after construction
- **Use case fit**: What scenarios each backend excels at

## Available Backends

### 1. DoubleArrayTrie (Recommended Default)

**Type**: Double-Array Trie with conflict resolution

**Characteristics:**
- **Construction**: Medium (conflict resolution)
- **Query**: Excellent (O(1) transitions, excellent cache locality)
- **Memory**: Minimal
- **Updates**: No (immutable after construction)
- **Unicode**: Use `DoubleArrayTrieChar` variant

**When to use:**
- Default choice for most static dictionaries
- Best overall query performance
- Memory-efficient
- Large dictionaries (100K+ terms)

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester"
]);

let transducer = Transducer::new(dict, Algorithm::Standard);
for term in transducer.query("tset", 2) {
    println!("{}", term);
}
```

**Feature flag:** `dat-backend` (enabled by default)

### 2. DoubleArrayTrieChar (Unicode Support)

**Type**: Character-level Double-Array Trie

**Characteristics:**
- **Construction**: Medium
- **Query**: Very Good (~5% slower than byte-level)
- **Memory**: Moderate (4× edge label memory)
- **Updates**: No
- **Unicode**: ✅ Correct character-level distances

**When to use:**
- Unicode text with multi-byte characters (accented, CJK, emoji)
- Need correct character-level Levenshtein distances
- Internationalization requirements

**Example:**

```rust
use liblevenshtein::prelude::*;

// Multi-byte UTF-8 characters handled correctly
let dict = DoubleArrayTrieChar::from_terms(vec![
    "café", "naïve", "日本語", "emoji😀"
]);

let transducer = Transducer::new(dict, Algorithm::Standard);
for candidate in transducer.query_with_distance("cafe", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Trade-offs:**
- ~5% performance overhead
- 4× memory for edge labels
- Correct Unicode Levenshtein distances

**Feature flag:** `dat-backend` (enabled by default)

### 3. PathMapDictionary (Dynamic Updates)

**Type**: Trie with structural sharing and interior mutability

**Characteristics:**
- **Construction**: Fast
- **Query**: Very Good
- **Memory**: Moderate
- **Updates**: ✅ Yes (thread-safe with `RwLock`)
- **Unicode**: Use `PathMapDictionaryChar` variant

**When to use:**
- Need runtime dictionary updates
- Insert/remove terms dynamically
- Concurrent updates and queries
- Medium-sized dictionaries (10K-100K terms)

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms(vec![
    "test", "testing"
]);

// Insert new terms at runtime
dict.insert("tested");
dict.insert("tester");

// Remove terms
dict.remove("testing");

let transducer = Transducer::new(dict, Algorithm::Standard);
for term in transducer.query("test", 1) {
    println!("{}", term);
}
```

**Thread safety:**
- Multiple concurrent readers
- Exclusive writer access via `RwLock`
- Queries see updates immediately

**Feature flag:** `pathmap-backend` (optional)

### 4. PathMapDictionaryChar (Dynamic Unicode)

**Type**: Character-level PathMap with dynamic updates

**Characteristics:**
- **Construction**: Fast
- **Query**: Good (~10% slower than byte-level)
- **Memory**: High (4× edge labels + structural overhead)
- **Updates**: ✅ Yes (thread-safe)
- **Unicode**: ✅ Correct character-level distances

**When to use:**
- Dynamic Unicode dictionaries
- Need both updates and correct Unicode distances
- Internationalized applications with runtime changes

**Feature flag:** `pathmap-backend` (optional)

### 5. DynamicDawg (Updates + Space Efficiency)

**Type**: DAWG with online insert/delete/minimize operations

**Characteristics:**
- **Construction**: Fast (incremental)
- **Query**: Good
- **Memory**: Low (maintains minimization)
- **Updates**: ✅ Yes (thread-safe with `RwLock`)

**When to use:**
- Need both updates and space efficiency
- Incremental dictionary construction
- Memory-constrained dynamic dictionaries

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DynamicDawg::from_terms(vec!["test"]);

// Online insertion with automatic minimization
dict.insert("testing");
dict.insert("tested");

// Online deletion
dict.remove("test");

println!("Nodes after minimization: {}", dict.node_count());
```

**Feature flag:** `dawg-backend` (optional)

### 6. SuffixAutomaton (Substring Matching)

**Type**: Suffix automaton for infix matching

**Characteristics:**
- **Construction**: Fast
- **Query**: Good (supports substring matching)
- **Memory**: Moderate
- **Updates**: No
- **Special**: Supports substring/infix matching

**When to use:**
- Need substring matching (not just prefix)
- Searching for patterns within words
- Text indexing applications

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = SuffixAutomaton::from_terms(vec![
    "testing", "fastest", "contest"
]);

// Can match substring "test" in any position
let transducer = Transducer::new(dict, Algorithm::Standard);
for term in transducer.query("test", 1) {
    println!("{}", term);
}
```

**Feature flag:** `suffix-automaton-backend` (optional)

### 7. SCDAWG (Symmetric Compact DAWG)

**Type**: Symmetric Compact Directed Acyclic Word Graph with bidirectional traversal

**Characteristics:**
- **Construction**: Medium (builds suffix automaton per term)
- **Query**: Excellent for substring (O(|pattern|))
- **Memory**: Moderate
- **Updates**: No (immutable after construction)
- **Special**: True suffix automaton indexing ALL substrings with bidirectional edges

**When to use:**
- Need O(|pattern|) substring search
- Bidirectional pattern traversal (left/right extensions)
- Text indexing with substring frequency queries
- WallBreaker pattern splitting algorithm

**Example:**

```rust
use liblevenshtein::prelude::*;

let scdawg = Scdawg::<()>::from_terms(["cathedral", "category", "catering"]);

// O(|pattern|) substring search
assert!(scdawg.contains_substring("cat"));
assert!(scdawg.contains_substring("thedr"));

// Find all occurrences
let matches = scdawg.find_exact_substring("cat");
assert_eq!(matches.len(), 3);  // Found in all three terms
```

**Feature flag:** `scdawg-backend` (optional)

### 8. PersistentARTrie (Disk-Based)

**Type**: Persistent Adaptive Radix Trie with memory-mapped storage

**Characteristics:**
- **Construction**: Fast (incremental inserts)
- **Query**: Excellent (adaptive node sizes, SIMD acceleration)
- **Memory**: Disk-based (configurable buffer cache)
- **Updates**: ✅ Yes (with WAL for crash recovery)
- **Special**: Handles dictionaries larger than RAM

**When to use:**
- Dictionary too large to fit in memory
- Need persistence across application restarts
- Crash recovery required
- Memory-constrained environments with large dictionaries

**Example:**

```rust
use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;

// Create a new persistent dictionary
let dict = PersistentARTrie::create("words.part")?;

// Insert terms (persisted to disk)
dict.insert("hello", ())?;
dict.insert("world", ())?;

// Query with transducer
let transducer = Transducer::new(&dict, Algorithm::Standard);
for result in transducer.query("helo", 1) {
    println!("{}: distance {}", result.term, result.distance);
}
```

**Architecture:**
- Adaptive node sizes: Node4, Node16 (SIMD), Node48, Node256
- B-trie buckets for efficient leaf storage
- Pointer swizzling for lazy loading
- Write-ahead logging (WAL) for crash recovery

**Feature flag:** `persistent-artrie` (optional)

## Backend Comparison

### Performance Summary

| Backend | Construction | Query | Memory | Updates | Unicode Variant |
|---------|-------------|-------|--------|---------|----------------|
| DoubleArrayTrie | ●●●○○ Medium | ●●●●● Excellent | ●●●●● Minimal | ✗ No | DoubleArrayTrieChar |
| PathMap | ●●●●○ Fast | ●●●●○ Very Good | ●●●○○ Moderate | ✅ Yes | PathMapDictionaryChar |
| DynamicDawg | ●●●●○ Fast | ●●●○○ Good | ●●●●○ Low | ✅ Yes | DynamicDawgChar |
| SuffixAutomaton | ●●●●○ Fast | ●●●○○ Good | ●●●○○ Moderate | ✗ No | SuffixAutomatonChar |
| SCDAWG | ●●●○○ Medium | ●●●●● Excellent (substring) | ●●●○○ Moderate | ✗ No | ScdawgChar |
| PersistentARTrie | ●●●●○ Fast | ●●●●● Excellent | Disk-based | ✅ Yes | PersistentARTrieChar |

### Benchmark Results

Query performance relative to DoubleArrayTrie (100K terms, distance 2):

| Backend | Relative Speed | Memory (MB) |
|---------|----------------|-------------|
| DoubleArrayTrie | 1.0× (baseline) | 8.5 |
| DoubleArrayTrieChar | 0.95× | 11.2 |
| PathMapDictionary | 0.92× | 12.3 |
| PathMapDictionaryChar | 0.87× | 16.8 |
| DynamicDawg | 0.85× | 7.8 |
| SuffixAutomaton | 0.82× | 10.5 |
| SCDAWG | 0.90× (substring: 1.2×) | 14.2 |
| PersistentARTrie | 0.88× | Disk + cache |

**Note**: All backends benefit from SIMD acceleration (20-64% faster with `simd` feature).

## Decision Guide

### Choose DoubleArrayTrie when:
- ✅ You need the best query performance
- ✅ Dictionary is static (no updates needed)
- ✅ Memory efficiency matters
- ✅ Default choice for most use cases

### Choose DoubleArrayTrieChar when:
- ✅ Working with Unicode text
- ✅ Need correct character-level distances
- ✅ Internationalization is required
- ✅ Can accept ~5% performance overhead

### Choose PathMapDictionary when:
- ✅ Need runtime dictionary updates
- ✅ Insert/remove operations required
- ✅ Thread-safe concurrent access needed
- ✅ Dictionary changes frequently

### Choose PathMapDictionaryChar when:
- ✅ Need both Unicode and dynamic updates
- ✅ Internationalized app with runtime changes
- ✅ Can accept higher memory usage

### Choose DynamicDawg when:
- ✅ Need both updates and space efficiency
- ✅ Memory constrained but need updates
- ✅ Can accept slightly slower queries

### Choose SuffixAutomaton when:
- ✅ Need substring/infix matching
- ✅ Pattern matching within words
- ✅ Text indexing applications

### Choose SCDAWG when:
- ✅ Need O(|pattern|) substring search
- ✅ Bidirectional pattern traversal required
- ✅ Pattern splitting (WallBreaker algorithm)
- ✅ Substring frequency queries

### Choose PersistentARTrie when:
- ✅ Dictionary larger than available RAM
- ✅ Need persistence across restarts
- ✅ Crash recovery is required
- ✅ Building dictionary incrementally over time

## Feature Flags

Enable backends via Cargo features:

```toml
[dependencies]
liblevenshtein = {
    git = "https://github.com/universal-automata/liblevenshtein-rust",
    tag = "v0.8.0",
    features = [
        "dat-backend",              # DoubleArrayTrie (default)
        "pathmap-backend",          # PathMapDictionary
        "dawg-backend",             # DynamicDawg
        "suffix-automaton-backend", # SuffixAutomaton
        "scdawg-backend",           # SCDAWG
        "persistent-artrie"         # PersistentARTrie
    ]
}
```

## Custom Backends

You can implement your own dictionary backend by implementing the `Dictionary` trait:

```rust
use liblevenshtein::dictionary::{Dictionary, DictionaryNode};

pub struct MyCustomDictionary {
    // Your implementation
}

impl Dictionary for MyCustomDictionary {
    type Node = MyNode;

    fn root(&self) -> Self::Node {
        // Return root node
    }

    fn len(&self) -> Option<usize> {
        // Return number of terms
    }

    fn contains(&self, term: &str) -> bool {
        // Check if term exists
    }

    // ... other required methods
}
```

See [Developer Guide](../developer-guide/architecture.md) for more details on custom backends.

## Related Documentation

- [Getting Started](getting-started.md) - Basic usage
- [Algorithms](algorithms.md) - Levenshtein algorithm variants
- [Thread Safety](thread-safety.md) - Concurrent access patterns
- [Serialization](serialization.md) - Save and load dictionaries
- [Benchmarks](../benchmarks/) - Detailed performance measurements

## References

- [Trie (Wikipedia)](https://en.wikipedia.org/wiki/Trie)
- [Directed Acyclic Word Graph (Wikipedia)](https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton)
- [Double-Array Trie](https://linux.thai.net/~thep/datrie/datrie.html)
- [Suffix Automaton](https://cp-algorithms.com/string/suffix-automaton.html)
