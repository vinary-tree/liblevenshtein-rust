# liblevenshtein-rust

[![Crates.io](https://img.shields.io/crates/v/liblevenshtein.svg)](https://crates.io/crates/liblevenshtein)
[![CI](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/ci.yml)
[![Nightly Tests](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/nightly.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/nightly.yml)
[![Release](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/release.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Rust implementation of **Levenshtein Automata** for fast approximate string matching with O(|W|) automaton construction and O(|D|) dictionary traversal.

Based on "Fast String Correction with Levenshtein-Automata" (Schulz & Mihov, 2002).

## Table of Contents

- [Quick Start](#quick-start)
- [Common Use Cases](#common-use-cases)
- [Thread Safety & Parallelism](#thread-safety--parallelism)
- [Dictionary Types](#dictionary-types)
- [Levenshtein Automata](#levenshtein-automata)
- [Universal Automata (Restricted Substitutions)](#universal-automata-restricted-substitutions)
- [MSM Automata (Time Series)](#msm-automata-time-series)
- [Weighted Automata](#weighted-automata)
- [WallBreaker Algorithm (Large Error Bounds)](#wallbreaker-algorithm-large-error-bounds)
- [LLev Phonetic Rules](#llev-phonetic-rules)
- [LLRE Fuzzy Regular Expressions](#llre-fuzzy-regular-expressions)
- [Phonetic NFA + Levenshtein Composition](#phonetic-nfa--levenshtein-composition)
- [WFST Integration with lling-llang](#wfst-integration-with-lling-llang)
- [Contextual Completion Engine](#contextual-completion-engine)
- [FuzzyCache (Eviction Policies)](#fuzzycache-eviction-policies)
- [Additional Features](#additional-features)
- [Performance](#performance)
- [Feature Flags Reference](#feature-flags-reference)
- [References](#references)

---

## Quick Start

```rust
use liblevenshtein::prelude::*;

// Create a dictionary
let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

// Create a transducer with Standard Levenshtein distance
let transducer = Transducer::new(dict, Algorithm::Standard);

// Query for terms within edit distance 2
for candidate in transducer.query_with_distance("tset", 2) {
    println!("{}: distance {}", candidate.term, candidate.distance);
}
// Output: test: distance 1
```

### Installation

```toml
[dependencies]
liblevenshtein = "0.8"

# SIMD (AVX2/SSE4.1) is automatic on x86_64 targets with runtime CPU detection.

# With phonetic rules:
liblevenshtein = { version = "0.8", features = ["phonetic-rules"] }
```

**Thread-safe by design**: All dictionary types implement `Send + Sync`. Queries are lock-free, and dynamic dictionaries support atomic insert/remove operations for concurrent access.

---

## Common Use Cases

Find the right starting point for your application:

| Task | Solution | Section |
|------|----------|---------|
| **Spell checking** | Standard Levenshtein with static dictionary | [Levenshtein Automata](#levenshtein-automata) |
| **Autocomplete / prefix search** | Dictionary prefix iteration | [Prefix Search](#prefix-search-command-completion) |
| **IDE code completion** | Hierarchical scopes with draft management | [Contextual Completion Engine](#contextual-completion-engine) |
| **Fuzzy search with metadata** | Aggregate values from fuzzy matches | [FuzzyMultiMap](#fuzzymultimap-value-aggregation) |
| **Phonetic matching** | Pattern NFAs composed with Levenshtein | [Phonetic NFA Composition](#phonetic-nfa--levenshtein-composition) |
| **Time series similarity** | Move-Split-Merge metric | [MSM Automata](#msm-automata-time-series) |
| **Keyboard typo correction** | Transposition algorithm | [Algorithm Variants](#algorithm-variants) |
| **OCR error correction** | MergeAndSplit + restricted substitutions | [Universal Automata](#universal-automata-restricted-substitutions) |
| **Large error bounds (k > 5)** | WallBreaker with SCDAWG | [WallBreaker Algorithm](#wallbreaker-algorithm-large-error-bounds) |
| **Substring fuzzy search** | SCDAWG + bidirectional extension | [SCDAWG](#scdawg-symmetric-compact-dawg) |
| **Persistent LM training** | Memory-mapped ARTrie for incremental updates | [Persistent ARTrie](#persistentmemory-mapped-dictionaries) |
| **WFST composition** | Compose with language models via lling-llang | [WFST Integration](#wfst-integration-with-lling-llang) |
| **Caching with eviction** | Composable TTL, LRU, LFU, cost-aware policies | [FuzzyCache](#fuzzycache-eviction-policies) |

---

## Thread Safety & Parallelism

liblevenshtein is designed for **concurrent, parallel workloads** from the ground up.

### Concurrency Guarantees

| Operation | Semantics |
|-----------|-----------|
| **Queries** | Lock-free, fully parallel |
| **Insert** | Atomic, non-blocking readers |
| **Remove** | Atomic, non-blocking readers |
| **Contains** | Lock-free |

### Sharing Across Threads

All dictionary types implement `Send + Sync`:

```rust
use std::sync::Arc;
use std::thread;
use liblevenshtein::prelude::*;

let dict = Arc::new(DynamicDawg::from_terms(vec!["hello", "world"]));

// Spawn parallel query threads
let handles: Vec<_> = (0..4).map(|_| {
    let dict = Arc::clone(&dict);
    thread::spawn(move || {
        let transducer = Transducer::new(&*dict, Algorithm::Standard);
        transducer.query("helo", 1).collect::<Vec<_>>()
    })
}).collect();

// All threads query concurrently without blocking
for handle in handles {
    let results = handle.join().expect("thread panicked");
}
```

### Atomic Updates

Dynamic dictionaries use fine-grained interior mutability:

- **Readers never block**: Queries proceed during concurrent inserts/removes
- **Writers are atomic**: Insert and remove complete as single operations
- **No external locking required**: Safe to share `Arc<DynamicDawg>` directly

---

## Dictionary Types

### Label Types and Interpretation

Dictionaries store labels of different sizes. While named for character types, they can store arbitrary values:

| Label Size | Types | Character Use | Arbitrary Use |
|------------|-------|---------------|---------------|
| **1-byte (u8)** | DoubleArrayTrie, DynamicDawg, SuffixAutomaton | ASCII characters | Byte sequences, small integers (0-255), bit flags |
| **4-byte (char/u32)** | DoubleArrayTrieChar, DynamicDawgChar, SuffixAutomatonChar | Unicode codepoints | 32-bit integers, small floats (via bit-casting) |
| **8-byte (u64)** | DynamicDawgU64 | N/A | 64-bit integers, double-precision floats (via bit-casting), compound keys |

```rust
use liblevenshtein::prelude::*;

// Store 32-bit integers as "characters"
let dict = DynamicDawgChar::new();
let values: Vec<char> = vec![0x1234, 0x5678, 0x9ABC]
    .into_iter()
    .map(|n| char::from_u32(n).unwrap())
    .collect();
dict.insert_seq(&values);
```

#### UTF-8 Compliance (*Char Variants)

The `*Char` variants (DoubleArrayTrieChar, DynamicDawgChar, etc.) provide **correct character-level Levenshtein distance** for UTF-8 encoded text. This matters because:

- **Byte-level is wrong for multi-byte characters**: "café" is 5 bytes but 4 characters. Using byte-level dictionaries, "café" → "cafe" would compute as 2 edits (replacing the 2-byte `é`), not 1.
- **Character-level is correct**: `*Char` variants use `char` (Unicode scalar values), so "café" → "cafe" = 1 substitution (é → e).

| Text | Byte Length | Char Length | Edit to ASCII |
|------|-------------|-------------|---------------|
| café | 5 | 4 | 1 (é→e) |
| 中文 | 6 | 2 | 2 |
| 🎉 | 4 | 1 | 1 |

**Use `*Char` variants for**: Non-ASCII text, internationalized content, CJK, Cyrillic, Arabic, emoji, or any text with accented characters.

### When to Use Each Dictionary

| Dictionary | Best For | Characteristics |
|------------|----------|-----------------|
| **DoubleArrayTrie** | Static ASCII dictionaries | O(1) transitions, 8 bytes/state, fastest queries |
| **DoubleArrayTrieChar** | Static Unicode dictionaries | Correct character-level distances |
| **DynamicDawg** | Dynamic ASCII dictionaries | Thread-safe insert/remove, SIMD + Bloom filter |
| **DynamicDawgChar** | Dynamic Unicode dictionaries | Character-level with runtime modifications |
| **DynamicDawgU64** | 64-bit label spaces | Large identifier spaces |
| **SuffixAutomaton** | Substring/infix search (ASCII) | Find patterns anywhere in terms |
| **SuffixAutomatonChar** | Substring/infix search (UTF-8) | Unicode-aware substring matching |
| **Scdawg** | Large error bound substring (ASCII) | O(\|pattern\|) search, WallBreaker support |
| **ScdawgChar** | Large error bound substring (UTF-8) | Unicode WallBreaker support |
| **PersistentARTrie** | Memory-mapped ASCII dictionaries | Zero-copy disk access |
| **PersistentARTrieChar** | Memory-mapped UTF-8 dictionaries | Zero-copy, `persistent-artrie` feature |
| **PathMapDictionaryChar** | Dynamic UTF-8 with PathMap backend | `pathmap-backend` feature required |

**All types are `Send + Sync`**: Safe to share across threads. Static types (DoubleArrayTrie, PersistentARTrie) are immutable after construction. Dynamic types (DynamicDawg, SuffixAutomaton, Scdawg) support atomic concurrent modifications.

### High-Performance Read-Only Dictionaries

Best for static dictionaries that don't change after construction:

```rust
use liblevenshtein::prelude::*;

// ASCII dictionary (1 byte per label)
let ascii_dict = DoubleArrayTrie::from_terms(vec!["hello", "world"]);

// Unicode dictionary (4 bytes per label)
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
let unicode_dict = DoubleArrayTrieChar::from_terms(vec!["café", "naïve", "中文"]);
```

### Dynamic Dictionaries (Insert & Remove)

Support runtime modifications with thread-safe operations:

```rust
use liblevenshtein::prelude::*;

let dict = DynamicDawg::new();
dict.insert("initial");

// Thread-safe modifications
dict.insert("added");
dict.remove("initial");

// Query immediately sees changes
assert!(dict.contains("added"));
assert!(!dict.contains("initial"));
```

**Thread Safety & Non-Blocking Semantics**:
- **Queries are lock-free**: Readers never block, even during concurrent writes
- **Atomic modifications**: Insert and remove are single atomic operations
- **No reader starvation**: Writers don't block pending reads
- **Zero external synchronization**: Share via `Arc<DynamicDawg>` without mutexes

#### Optimizations for DynamicDawg Variants

**Bloom Filter**: Pre-filter transitions to skip impossible branches:

```rust
use liblevenshtein::prelude::*;

// Enable Bloom filter with expected capacity
let dict = DynamicDawg::with_config(
    f32::INFINITY,  // Auto-minimize threshold (disabled)
    Some(10000),    // Bloom filter capacity
);
// Result: 88-93% faster contains() operations
```

**SIMD Acceleration**: Vectorized transition lookups with AVX2/SSE4.1:

```rust
// Automatic on x86_64 targets (runtime-detected via is_x86_feature_detected!)
// Provides 20-64% faster queries; no feature flag required
```

#### PathMap Backend (Alternative Dynamic Dictionary)

For workloads with frequent updates, the PathMap backend offers an alternative:

```rust
// Requires: features = ["pathmap-backend"]
use liblevenshtein::dictionary::pathmap_char::PathMapDictionaryChar;

let dict = PathMapDictionaryChar::new();
dict.insert("日本語");  // UTF-8 compliant
dict.insert("한국어");
```

### Prefix Search (Command Completion)

Navigate to a prefix and iterate only matching terms:

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::dictionary::prefix_zipper::PrefixZipper;
use liblevenshtein::dictionary::double_array_trie_zipper::DoubleArrayTrieZipper;

let dict = DoubleArrayTrie::from_terms(vec!["getValue", "getVariable", "setValue"]);
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

if let Some(iter) = zipper.with_prefix(b"get") {
    for (path, _) in iter {
        let term = String::from_utf8(path).expect("valid UTF-8");
        println!("Found: {}", term);
        // Output: getValue, getVariable
    }
}
```

### Substring/Suffix Search

Find patterns anywhere within terms:

```rust
use liblevenshtein::dictionary::suffix_automaton::SuffixAutomaton;

let sa = SuffixAutomaton::from_text("hello world");

// Check if substring exists
assert!(sa.contains_substring("llo wo"));
assert!(!sa.contains_substring("xyz"));
```

### SCDAWG (Symmetric Compact DAWG)

A **Symmetric Compact DAWG** (Blumer et al. 1987, Inenaga et al. 2005) indexes *all* substrings of a text and supports bidirectional traversal via left extension edges.

**Key properties:**
- O(|pattern|) substring search
- Bidirectional traversal for extending matches in both directions
- Required backend for the [WallBreaker algorithm](#wallbreaker-algorithm-large-error-bounds)
- Space-efficient: O(n) nodes for text of length n

```rust
use liblevenshtein::dictionary::scdawg::Scdawg;           // ASCII
use liblevenshtein::dictionary::scdawg_char::ScdawgChar;  // UTF-8

// ASCII version (byte-level)
let scdawg = Scdawg::from_text("mississippi");
assert!(scdawg.contains_substring("issi"));

// UTF-8 version (character-level) - use for non-ASCII text
let scdawg_utf8 = ScdawgChar::from_text("北京欢迎你");
assert!(scdawg_utf8.contains_substring("欢迎"));  // 2 characters, not 6 bytes

// Bidirectional navigation (for WallBreaker)
if let Some(state) = scdawg.follow_substring("iss") {
    // Extend left: what characters precede "iss"?
    let left_extensions = scdawg.left_extensions(state);
    // Returns: ['m', 's'] (from "miss" and "siss")

    // Extend right: what characters follow "iss"?
    let right_extensions = scdawg.right_extensions(state);
    // Returns: ['i'] (from "issi")
}
```

**When to use SCDAWG over SuffixAutomaton:**
| Feature | SuffixAutomaton | SCDAWG |
|---------|-----------------|--------|
| Substring search | O(\|pattern\|) | O(\|pattern\|) |
| Bidirectional extension | No | Yes |
| WallBreaker support | No | Yes |
| Memory | Lower | Slightly higher (sext links) |

### Persistent/Memory-Mapped Dictionaries

Zero-copy disk access for large dictionaries (requires `persistent-artrie` feature):

```rust
use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;           // ASCII
use liblevenshtein::dictionary::persistent_artrie_char::PersistentARTrieChar;  // UTF-8

// Memory-map from file (zero-copy) - ASCII version
let dict = PersistentARTrie::open("dictionary.dat").expect("failed to open");

// UTF-8 version for Unicode dictionaries
let dict_utf8 = PersistentARTrieChar::open("unicode_dict.dat").expect("failed to open");

// Query without loading entire file into memory
let transducer = Transducer::new(dict, Algorithm::Standard);
```

### Value-Mapped Dictionaries

Store metadata with each term:

```rust
use liblevenshtein::prelude::*;

// Store word frequencies
let dict: DynamicDawg<u32> = DynamicDawg::new();
dict.insert_with_value("apple", 1500);
dict.insert_with_value("apply", 850);

// Retrieve values
assert_eq!(dict.get_value("apple"), Some(1500));

// Fuzzy lookup with values
let map = FuzzyMap::new(dict, Algorithm::Standard);
let results = map.get_with_distance("aple", 1);
for (term, value, distance) in results {
    println!("{}: {} (distance {})", term, value, distance);
}
```

### FuzzyMultiMap (Value Aggregation)

Aggregate values from multiple fuzzy-matched keys. Useful when multiple dictionary entries may match and you need all associated data.

| Collection Type | Aggregation |
|-----------------|-------------|
| `HashSet<T>` | Union of all sets |
| `BTreeSet<T>` | Union of all sets |
| `Vec<T>` | Concatenation |

```rust
use std::collections::HashSet;
use liblevenshtein::prelude::*;
use liblevenshtein::cache::multimap::FuzzyMultiMap;

// Map terms to sets of document IDs
let dict: DynamicDawgChar<HashSet<u32>> = DynamicDawgChar::new();
dict.insert_with_value("color", HashSet::from([1, 2, 5]));
dict.insert_with_value("colour", HashSet::from([3, 4]));
dict.insert_with_value("colr", HashSet::from([6]));

let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

// Query "colur" (distance 1) - matches "color" and "colour"
// Result: union of {1,2,5} and {3,4} = {1,2,3,4,5}
let doc_ids = fuzzy.query("colur", 1).expect("no matches");
```

Preserve match details with `query_with_distance`:

```rust
// Get (matched_key, distance, values) tuples
for (key, distance, doc_ids) in fuzzy.query_with_distance("colur", 1) {
    println!("'{}' (distance {}): {:?}", key, distance, doc_ids);
}
// Output:
//   'color' (distance 1): {1, 2, 5}
//   'colour' (distance 1): {3, 4}
```

---

## Levenshtein Automata

### Algorithm Variants

| Algorithm | Operations | Use Case |
|-----------|-----------|----------|
| **Standard** | Insert, delete, substitute | General fuzzy matching |
| **Transposition** | + adjacent swaps | Typing errors (teh→the) |
| **MergeAndSplit** | + two↔one character | OCR errors (rn→m, vv→w) |

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec!["the", "them", "then"]);

// Standard: "teh" → "the" costs 2 (delete h, insert h)
let standard = Transducer::new(dict.clone(), Algorithm::Standard);

// Transposition: "teh" → "the" costs 1 (swap e↔h)
let transposition = Transducer::new(dict.clone(), Algorithm::Transposition);

// MergeAndSplit: "rn" ↔ "m" costs 1
let merge_split = Transducer::new(dict, Algorithm::MergeAndSplit);
```

### Query Types

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec!["apple", "apply", "ape"]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Basic query - returns matching terms
for term in transducer.query("aple", 1) {
    println!("{}", term);
}

// With distances
for c in transducer.query_with_distance("aple", 1) {
    println!("{}: {}", c.term, c.distance);
}

// Ordered by distance, then alphabetically
for c in transducer.query_ordered("aple", 1) {
    println!("{}: {}", c.term, c.distance);
}

// Filtered by predicate
for c in transducer.query_filtered("aple", 2, |v| *v > 100) {
    println!("{}", c.term);
}

// Filtered by value set (optimized for scope visibility)
use std::collections::HashSet;
let visible_scopes: HashSet<u32> = [1, 2, 3].iter().cloned().collect();
for c in transducer.query_by_value_set("func", 2, &visible_scopes) {
    println!("{}", c.term);  // Only terms in scopes 1, 2, or 3
}
```

---

## Universal Automata (Restricted Substitutions)

Universal Automata allow **restricted substitutions** where only specific character pairs can be substituted at zero cost.

### SubstitutionPolicy

```rust
use liblevenshtein::transducer::substitution_policy::{
    SubstitutionPolicy, Unrestricted, Restricted
};
use liblevenshtein::transducer::SubstitutionSet;

// Unrestricted (default) - standard Levenshtein
let unrestricted = Unrestricted;

// Restricted - only explicit pairs allowed
let mut set = SubstitutionSet::new();
set.allow('c', 'k');  // c ↔ k costs 0
set.allow('f', 'p');  // f ↔ p costs 0 (single char pairs only)
let restricted = Restricted::new(&set);
```

### Pre-built Substitution Sets

```rust
use liblevenshtein::transducer::SubstitutionSet;

// Phonetic confusions (f↔ph, c↔k, s↔z)
let phonetic = SubstitutionSet::phonetic_basic();

// QWERTY keyboard adjacency
let keyboard = SubstitutionSet::keyboard_qwerty();

// OCR visual similarity (0↔O, 1↔l↔I)
let ocr = SubstitutionSet::ocr_friendly();

// Leet speak (3↔e, 4↔a, 0↔o)
let leet = SubstitutionSet::leet_speak();
```

### Unicode Substitutions

```rust
use liblevenshtein::transducer::substitution_policy::RestrictedChar;
use liblevenshtein::transducer::SubstitutionSetChar;

let mut set = SubstitutionSetChar::new();
set.allow('é', 'e');  // Accent-insensitive
set.allow('ñ', 'n');
let policy = RestrictedChar::new(&set);
```

---

## MSM Automata (Time Series)

The **Move-Split-Merge (MSM)** metric provides similarity measurement for time series data.

### Move-Split-Merge Operations

| Operation | Description | Cost |
|-----------|-------------|------|
| **Move** | Change a value | \|change\| |
| **Split** | Duplicate into two elements | c (configurable) |
| **Merge** | Combine two equal elements | c (configurable) |

```rust
use liblevenshtein::wfst::msm::{MsmWfst, MsmWfstBuilder};
use liblevenshtein::time_series::MsmConfig;

let query = vec![1.0, 2.0, 3.0];
let target = vec![1.0, 2.0, 3.0];

let wfst = MsmWfstBuilder::new()
    .query(&query)
    .msm_config(MsmConfig::new(1.0))  // Split/merge cost
    .max_cost(10.0)
    .add_target(0, &target)
    .build()
    .expect("build failed");
```

### Adaptive MSM (Online Learning)

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

let config = AdaptiveMsmConfig::default()
    .with_learning_rate(0.1)
    .with_initial_cost(1.0);

let mut adaptive = AdaptiveMsm::new(config);

// Update parameters based on feedback
adaptive.update_from_match(query_series, matched_series);
```

---

## Weighted Automata

Weighted automata support **variable operation costs** for context-sensitive matching.

### Custom Operation Costs

```rust
use liblevenshtein::transducer::OperationCosts;

let costs = OperationCosts::new()
    .with_insert_cost(1.0)
    .with_delete_cost(1.0)
    .with_substitute_cost(1.5)     // Substitution costs more
    .with_transpose_cost(0.5);     // Transposition costs less
```

### Generalized Automata

```rust
use liblevenshtein::wfst::GeneralizedWfst;

let wfst = GeneralizedWfst::builder()
    .operation_costs(costs)
    .dictionary(&dict)
    .query("example")
    .max_cost(3.0)
    .build();
```

---

## WallBreaker Algorithm (Large Error Bounds)

Traditional Levenshtein automata suffer from the **wall effect**: the first *b* steps must explore all prefixes of length ≤ *b*, regardless of the actual matches. For large error bounds (e.g., *k* = 16), this becomes prohibitively expensive.

**WallBreaker** overcomes this by using the **pigeonhole principle**: if a pattern of length *m* matches with at most *k* errors, at least one piece of the pattern must match exactly when split into *p* pieces.

### How It Works

1. **Split** the query into *p* pieces (number depends on algorithm)
2. **Find exact substring matches** using SCDAWG in O(|piece|) time
3. **Extend bidirectionally** from each match to verify within error bound
4. **Merge results** across all piece matches

### Piece Counts by Algorithm

The minimum number of pieces required (formally verified in Coq):

| Algorithm | Pieces Required | Reason |
|-----------|-----------------|--------|
| **Standard** | k + 1 | Each error affects at most one piece |
| **Transposition** | 2k + 1 | Swaps can affect adjacent pieces |
| **MergeAndSplit** | 2k + 1 | Merge/split can span piece boundaries |

### Performance

For a 750,000-word dictionary with 100-character patterns and 16 errors:

| Approach | Time |
|----------|------|
| Traditional automaton | ~500ms |
| WallBreaker | ~0.088ms |
| **Speedup** | **5,600x** |

### Usage

```rust
use liblevenshtein::wallbreaker::{WallBreaker, WallBreakerConfig};
use liblevenshtein::dictionary::scdawg::Scdawg;
use liblevenshtein::transducer::Algorithm;

// Build SCDAWG from dictionary (required for WallBreaker)
let scdawg = Scdawg::from_terms(dictionary_terms);

let config = WallBreakerConfig::new()
    .algorithm(Algorithm::Standard)
    .max_distance(16);

let wallbreaker = WallBreaker::new(scdawg, config);

// Query with large error bound - fast!
for candidate in wallbreaker.query("misspelled_query_term") {
    println!("{}: distance {}", candidate.term, candidate.distance);
}
```

### WallBreakerWfst (WFST Integration)

For composition with language models:

```rust
use liblevenshtein::wfst::{WallBreakerWfst, WallBreakerWfstBuilder};

let wfst = WallBreakerWfstBuilder::new()
    .scdawg(&scdawg)
    .query("example_query")
    .max_distance(10)
    .build()
    .expect("build failed");

// Compose with n-gram language model
// let composed = compose(wfst, language_model);
```

### When to Use WallBreaker

| Scenario | Recommendation |
|----------|----------------|
| Short queries, small *k* (≤ 3) | Standard transducer (simpler) |
| Long queries, large *k* (≥ 5) | WallBreaker (much faster) |
| Pattern length > 50, *k* > 10 | WallBreaker (essential) |

---

## LLev Phonetic Rules

The `.llev` format defines phonetic rewrite rules with metadata and context conditions.

### Rule File Format

```text
@name "English Phonetic Rules"
@version "1.0"

# Simple substitution
ph -> f;                          # phone → fone

# Context-dependent (after vowel)
gh -> / [:vowel:]_;               # night → nit

# Before specific characters
c -> s / _[:front_vowel:];        # city → sity
```

### Complete Syntax Specification

**Case Sensitivity:** LLev is **case-insensitive** by default (phonetic rules match sounds, not spelling). Use `(?c:pattern)` or `(?-i:pattern)` for case-sensitive matching.

**File Metadata Directives:**

```text
@name "English Phonetic Rules"    # Rule file name
@version "1.0"                    # Version string
@author "John Doe"                # Author attribution
@description "Base phonetic..."   # Description text
@include "common.llev"            # Include another rule file
@define VOWEL = [aeiouAEIOU]      # Define reusable symbol (UPPERCASE)
```

**Rule Metadata Blocks:**

```text
[id: 1, name: "ph to f", weight: 0.0, group: orthography, enabled: true, ipa: "/f/"]
ph -> f;

[id: 20, name: "soft c", weight: 0.15, group: consonants]
c -> s / _$FRONT_VOWEL;
```

| Metadata Key | Description |
|--------------|-------------|
| `id` | Unique integer identifier |
| `name` | Human-readable rule name |
| `weight` | Cost/priority (0.0 = exact, higher = less likely) |
| `group` | Organizational category |
| `enabled` | Boolean to enable/disable rule |
| `ipa` | IPA transcription: `/f/` phonemic, `[f]` phonetic |

**Rule Syntax:**

| Form | Description |
|------|-------------|
| `pattern -> replacement;` | Simple replacement |
| `pattern -> replacement / context;` | Conditional replacement |
| `pattern -> / context;` | Deletion (empty replacement) |

**Pattern Elements:**

| Syntax | Description |
|--------|-------------|
| `.` | Any single character (wildcard) |
| `[aeiou]` | Character class |
| `[^aeiou]` | Negated character class |
| `[a-z]` | Character range |
| `[:VOWEL:]` | Named character class |
| `[:VOWEL sound:]` | Feature bundle (intersection) |
| `[:!nasal voiced:]` | Negated feature in bundle |
| `$SYMBOL` | Reference defined symbol |
| `(ph\|f)` | Alternation |
| `*`, `+`, `?` | Quantifiers (zero+, one+, optional) |
| `{n}`, `{n,m}` | Repetition |
| `(?c:...)` | Case-sensitive group |

**Phonetic Shortcuts:**

| Shortcut | Matches | Negated |
|----------|---------|---------|
| `\v` | Vowel | `\V` |
| `\c` | Consonant | `\C` |
| `\f` | Front vowel | `\F` |
| `\k` | Back vowel | `\K` |
| `\h` | High vowel | `\H` |
| `\l` | Low vowel | `\L` |
| `\m` | Mid vowel | `\M` |
| `\p` | Stop/plosive | `\P` |
| `\e` | Fricative | `\E` |
| `\a` | Affricate | `\A` |
| `\z` | Nasal | `\Z` |
| `\q` | Liquid | `\Q` |
| `\g` | Glide | `\G` |
| `\o` | Voiced | `\O` |

**Context Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `_` | Match position marker | `[:vowel:]_` = after vowel |
| `#` | Word boundary | `#_` = word start, `_#` = word end |
| `&` | AND (both must match) | `[:voiced:]_&_[:vowel:]` |
| `\|` | OR (either matches) | `#_\|[:vowel:]_` |
| `!` | NOT (negation) | `![:nasal:]_` = not after nasal |

**Syllable Conditions:**

```text
gh -> / [:vowel:]_ if not initial_syllable;   # Silent gh except word-initial
e -> / _# if final_syllable;                   # Drop final silent e
a -> æ / _ if monosyllable;                    # Short a in monosyllables
y -> i / _# if polysyllable;                   # Y to I in polysyllables
```

| Condition | Description |
|-----------|-------------|
| `if monosyllable` | Single-syllable word |
| `if polysyllable` | Multi-syllable word |
| `if open_syllable` | Syllable ends with vowel |
| `if closed_syllable` | Syllable ends with consonant |
| `if initial_syllable` | In first syllable |
| `if final_syllable` | In last syllable |

### Built-in Language Support

**50 languages** with pre-compiled phonetic rules (Rust modules):

| Language Family | Languages |
|-----------------|-----------|
| **Romance** | Spanish, Italian, French, Portuguese, Romanian, Catalan |
| **Germanic** | English, German, Dutch, Swedish, Norwegian, Danish, Icelandic |
| **Slavic** | Russian, Polish, Ukrainian, Czech, Slovak, Croatian, Serbian, Bulgarian, Belarusian |
| **Celtic** | Welsh, Irish |
| **Indic** | Hindi, Urdu, Marathi, Bengali, Gujarati, Telugu, Tamil, Punjabi |
| **East Asian** | Chinese, Japanese, Korean |
| **Southeast Asian** | Vietnamese, Thai, Indonesian, Tagalog |
| **Semitic** | Arabic, Hebrew, Maltese |
| **Other** | Turkish, Hungarian, Finnish, Basque, Greek, Persian, Georgian, Armenian |

**122 total languages** have `.llev` rule data files loadable at runtime, including: Afrikaans, Albanian, Amharic, Azerbaijani, Estonian, Hausa, Hawaiian, Kazakh, Khmer, Kurdish, Latvian, Lithuanian, Malay, Mongolian, Nepali, Pashto, Swahili, Uzbek, Yoruba, Zulu, and many more.

```rust
use liblevenshtein::phonetic::rules::{english, spanish, german};

let english_rules = english::base();      // 62 rules
let homophones = english::homophones();   // too/two→to
let text_speak = english::text_speak();   // u→you, thx→thanks

let spanish_rules = spanish::base();
let german_rules = german::base();
```

### Loading and Applying Rules

```rust
use liblevenshtein::phonetic::llev::{parse_str, apply_rules_seq_char};

let rules = parse_str(r#"
    ph -> f;
    gh -> / [:vowel:]_;
"#).expect("parse failed");

let normalized = apply_rules_seq_char(&rules.rules, "phone");
// Result: "fone"
```

### Custom Rule Files

```rust
use liblevenshtein::phonetic::llev::parse_file;
use std::path::Path;

let rules = parse_file(Path::new("custom.llev")).expect("parse failed");
```

---

## LLRE Fuzzy Regular Expressions

The `.llre` format provides regex-style patterns with phonetic extensions.

**Case Sensitivity:** LLRE is **case-sensitive** by default (standard regex behavior). Use `(?i)` or `(?i:pattern)` for case-insensitive matching.

### Pattern Syntax

**Wildcards and Quantifiers:**

| Syntax | Description |
|--------|-------------|
| `.` | Any single character (wildcard) |
| `*` | Zero or more |
| `+` | One or more |
| `?` | Optional |
| `{n}` | Exactly n occurrences |
| `{n,}` | At least n occurrences |
| `{,m}` | No more than m occurrences |
| `{n,m}` | n to m occurrences |

**Anchors:**

| Syntax | Description |
|--------|-------------|
| `^` | Start of line/input |
| `$` | End of line/input |
| `\A` | Absolute start of input |
| `\Z` | End of input (allows trailing newline) |
| `\z` | Strict end of input |
| `#` | Word boundary |

**Grouping and Alternation:**

| Syntax | Description | Example |
|--------|-------------|---------|
| `(...)` | Capture group | `(ph\|f)one` |
| `(?:...)` | Non-capturing group | `(?:un)?do` |
| `(?<name>...)` | Named capture group | `(?<vowel>[aeiou])` |
| `(?&name)` | Subroutine call/reference | `(?&vowel)` |
| `a\|b` | Alternation | `cat\|dog` |

**Feature Flags:**

| Syntax | Description |
|--------|-------------|
| `(?i)` | Case insensitive (rest of pattern) |
| `(?-i)` | Explicitly case sensitive |
| `(?m)` | Multiline (^ $ match line boundaries) |
| `(?-m)` | Disable multiline |
| `(?s)` | Dotall (. matches newlines) |
| `(?-s)` | Disable dotall |
| `(?a)` | Accent insensitive |
| `(?f)` | Feature-based phonetic matching |

**Scoped Flags:**

| Syntax | Description |
|--------|-------------|
| `(?i:pattern)` | Case insensitive group |
| `(?a:pattern)` | Accent insensitive group |
| `(?s:pattern)` | Dotall group |
| `(?m:pattern)` | Multiline group |
| `(?f:pattern)` | Feature-based group |
| `(?ia:pattern)` | Combined flags |
| `(?u:NFC)` | Unicode NFC normalization |
| `(?u:NFD)` | Unicode NFD normalization |
| `(?u:NFKC)` | Unicode NFKC normalization |
| `(?u:NFKD)` | Unicode NFKD normalization |
| `(?;N)` | Local Levenshtein distance limit (N edits) |

**File-Level Flags (.llre files):**

```text
@flags multiline              # or @flags m
@flags dotall                 # or @flags s
@flags case_insensitive       # or @flags i or @flags ignorecase
@flags unicode                # or @flags u
@flags multiline, dotall, case_insensitive   # Multiple
```

**Phonetic Shortcuts:**

| Shortcut | Matches | Negated |
|----------|---------|---------|
| `\v` | Vowel | `\V` |
| `\c` | Consonant | `\C` |
| `\f` | Front vowel | `\F` |
| `\k` | Back vowel | `\K` |
| `\h` | High vowel | `\H` |
| `\l` | Low vowel | `\L` |
| `\m` | Mid vowel | `\M` |
| `\p` | Stop/plosive | `\P` |
| `\e` | Fricative | `\E` |
| `\a` | Affricate | `\A` (inside char class) |
| `\z` | Nasal | `\Z` (inside char class) |
| `\q` | Liquid | `\Q` |
| `\g` | Glide | `\G` |
| `\o` | Voiced | `\O` |

**Character Classes:**

| Syntax | Description | Example |
|--------|-------------|---------|
| `[abc]` | Character set | One of a, b, or c |
| `[^abc]` | Negated set | Any char except a, b, c |
| `[a-z]` | Range | a through z |
| `[a-zA-Z0-9]` | Multiple ranges | Alphanumeric |
| `[[:name:]]` | Named class in set | `[[:vowel:]]` |
| `[:name:]` | Named class | See tables below |

### Named Character Classes

**Vowel Classes:**

| Class | Description |
|-------|-------------|
| `[:vowel:]` | All vowels (a, e, i, o, u + IPA: ə, ɪ, ʊ, ɛ, ɔ, æ, ɑ, etc.) |
| `[:front_vowel:]` | Front vowels (i, e, æ, ɪ, ɛ) |
| `[:back_vowel:]` | Back vowels (o, u, ɔ, ʊ, ɑ) |
| `[:high_vowel:]` | High/close vowels (i, u, ɪ, ʊ) |
| `[:mid_vowel:]` | Mid vowels (e, o, ə, ɛ, ɔ) |
| `[:low_vowel:]` | Low/open vowels (a, æ, ɑ, ɐ) |
| `[:central_vowel:]` | Central vowels (ə, ɐ, ɨ, ʉ) |
| `[:schwa:]` | Schwa (ə) |
| `[:rounded:]` | Rounded vowels (o, u, ɔ, ʊ, y, ø) |
| `[:unrounded:]` | Unrounded vowels (a, e, i, æ, ɛ, ɪ) |
| `[:ascii_vowel:]` | ASCII vowels only (a, e, i, o, u) |
| `[:ipa_vowel:]` | IPA vowel symbols only |

**Consonant Classes:**

| Class | Description |
|-------|-------------|
| `[:consonant:]` | All consonants |
| `[:stop:]` / `[:plosive:]` | Stops (p, b, t, d, k, g, ʔ) |
| `[:fricative:]` | Fricatives (f, v, s, z, ʃ, ʒ, θ, ð, h, x, ɣ) |
| `[:affricate:]` | Affricates (tʃ, dʒ, ts, dz) |
| `[:nasal:]` | Nasals (m, n, ŋ, ɲ, ɴ) |
| `[:liquid:]` | Liquids (l, r, ɹ, ɾ, ʁ) |
| `[:glide:]` / `[:semivowel:]` | Glides (w, j, ʍ) |
| `[:approximant:]` | Approximants (w, j, l, ɹ) |
| `[:lateral:]` | Laterals (l, ɫ, ʎ, ɬ) |
| `[:trill:]` | Trills (r, ʀ, ʙ) |
| `[:tap:]` / `[:flap:]` | Taps/flaps (ɾ, ɽ) |
| `[:ascii_consonant:]` | ASCII consonants only |
| `[:ipa_consonant:]` | IPA consonant symbols only |

**Voice Classes:**

| Class | Description |
|-------|-------------|
| `[:voiced:]` | Voiced sounds (b, d, g, v, z, ʒ, m, n, l, r, w, j) |
| `[:voiceless:]` | Voiceless sounds (p, t, k, f, s, ʃ, θ, h, x) |
| `[:voiced_fricative:]` | Voiced fricatives (v, z, ʒ, ð, ɣ) |
| `[:voiceless_fricative:]` | Voiceless fricatives (f, s, ʃ, θ, h, x) |
| `[:sibilant:]` | Sibilants (s, z, ʃ, ʒ, ts, dz, tʃ, dʒ) |

**Place of Articulation:**

| Class | Description |
|-------|-------------|
| `[:bilabial:]` | Bilabials (p, b, m, ɸ, β, ʙ) |
| `[:labiodental:]` | Labiodentals (f, v, ɱ) |
| `[:dental:]` | Dentals (θ, ð, t̪, d̪) |
| `[:alveolar:]` | Alveolars (t, d, n, s, z, l, r, ɾ) |
| `[:postalveolar:]` | Postalveolars (ʃ, ʒ, tʃ, dʒ) |
| `[:retroflex:]` | Retroflexes (ʈ, ɖ, ɳ, ʂ, ʐ, ɻ, ɽ) |
| `[:palatal:]` | Palatals (c, ɟ, ɲ, ç, ʝ, j, ʎ) |
| `[:velar:]` | Velars (k, g, ŋ, x, ɣ, w) |
| `[:uvular:]` | Uvulars (q, ɢ, ɴ, χ, ʁ, ʀ) |
| `[:pharyngeal:]` | Pharyngeals (ħ, ʕ) |
| `[:glottal:]` | Glottals (ʔ, h, ɦ) |

**Manner/Phonological Features:**

| Class | Description |
|-------|-------------|
| `[:obstruent:]` | Obstruents (stops + fricatives + affricates) |
| `[:sonorant:]` | Sonorants (nasals + liquids + glides + vowels) |
| `[:continuant:]` | Continuants (fricatives + approximants + vowels) |
| `[:aspirated_affricate:]` | Aspirated affricates (t͡sʰ, t͡ʃʰ, etc.) |

**Special Classes:**

| Class | Description |
|-------|-------------|
| `[:click:]` | Click consonants (ʘ, ǀ, ǃ, ǂ, ǁ) |
| `[:implosive:]` | Implosives (ɓ, ɗ, ɠ, ʄ, ʛ) |
| `[:ejective:]` | Ejectives (pʼ, tʼ, kʼ, sʼ, etc.) |

**POSIX Classes:**

| Class | Description |
|-------|-------------|
| `[:alpha:]` | All alphabetic characters |
| `[:lower:]` | Lowercase letters (a-z) |
| `[:upper:]` | Uppercase letters (A-Z) |
| `[:digit:]` | Digits 0-9 |
| `[:alnum:]` | Alphanumeric |
| `[:word:]` | Word characters (a-z, A-Z, 0-9, _) |
| `[:space:]` | Whitespace |
| `[:punct:]` | Punctuation |

### Compiling Patterns

```rust
use liblevenshtein::phonetic::llre;

let pattern = llre::compile_pattern("[:fricative:]one").expect("compile failed");
assert!(pattern.matches("fone"));   // f ∈ fricative
assert!(pattern.matches("shone"));  // sh ∈ fricative
assert!(!pattern.matches("bone"));  // b ∉ fricative
```

---

## Phonetic NFA + Levenshtein Composition

Combine phonetic pattern NFAs with Levenshtein automata for fuzzy phonetic matching.

### How Composition Works

1. **Phonetic NFA** recognizes multiple phonetic spellings (ph|f)one
2. **Levenshtein automaton** allows edit operations (insert, delete, substitute)
3. **Product automaton** accepts inputs matching the pattern within edit distance

### ProductAutomaton Usage

```rust
use liblevenshtein::phonetic::nfa::{compile, ProductAutomatonChar};
use liblevenshtein::phonetic::regex::parse;

// Parse and compile phonetic pattern
let regex = parse("(ph|f)one").expect("parse failed");
let nfa = compile(&regex).expect("compile failed");

// Compose with Levenshtein (max distance 2)
let product = ProductAutomatonChar::new(nfa, 2);

// Fuzzy phonetic matching
assert!(product.accepts("phone"));   // Exact (distance 0)
assert!(product.accepts("fone"));    // Exact (distance 0)
assert!(product.accepts("phones"));  // Insert 's' (distance 1)
assert!(product.accepts("phon"));    // Delete 'e' (distance 1)
assert!(product.accepts("phome"));   // Substitute n→m (distance 1)

// Get minimum distance
assert_eq!(product.min_distance("phone"), Some(0));
assert_eq!(product.min_distance("fon"), Some(1));
assert_eq!(product.min_distance("xyz"), None);  // Outside budget
```

### PhoneticGrep Convenience API

```rust
use liblevenshtein::phonetic::grep::PhoneticGrep;

// Quick on-the-fly matching
let grep = PhoneticGrep::from_pattern("phone", 1).expect("build failed");
assert!(grep.matches("phone").is_some());
assert!(grep.matches("fone").is_some());

// With case + accent insensitivity
let grep = PhoneticGrep::from_pattern("(?ia:cafe)", 1).expect("build failed");
assert!(grep.matches("CAFÉ").is_some());
```

---

## WFST Integration with lling-llang

The `wfst` feature enables integration with the [lling-llang](https://github.com/f1r3fly-io/lling-llang) weighted finite-state transducer library.

### LevenshteinWfst

Lazy WFST wrapper for Levenshtein × dictionary product:

```rust
use liblevenshtein::wfst::{LevenshteinWfst, DictionaryBackend};
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

let dict = DynamicDawgChar::from_terms(vec!["hello", "help", "world"]);

// Create WFST for query "helo" with max distance 2
let lev_wfst = LevenshteinWfst::new(&dict, "helo", 2);

// Compose with language model
// let composed = compose(lev_wfst, language_model);
```

### PhoneticWfst Pipeline

```rust
use liblevenshtein::wfst::{PhoneticWfst, PhoneticWfstBuilder};

let wfst = PhoneticWfstBuilder::new()
    .rules_file("english.llev")
    .dictionary(&dict)
    .max_distance(2)
    .build()
    .expect("build failed");
```

### MsmWfst for Time Series

```rust
use liblevenshtein::wfst::msm::MsmWfst;

// See MSM Automata section for full example
```

### WallBreakerWfst

Optimized for large error bounds:

```rust
use liblevenshtein::wfst::{WallBreakerWfst, WallBreakerWfstBuilder};

let wfst = WallBreakerWfstBuilder::new()
    .dictionary(&dict)
    .query("example")
    .max_distance(5)  // Large error bound
    .build()
    .expect("build failed");
```

---

## Contextual Completion Engine

IDE-like code completion with hierarchical scopes and draft management.

### Hierarchical Contexts

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Standard);

// Create scope hierarchy: global → function → block
let global = engine.create_root_context(0);
let function = engine.create_child_context(1, global).expect("create failed");
let block = engine.create_child_context(2, function).expect("create failed");

// Add terms to scopes
engine.finalize_direct(global, "std::vector").expect("insert failed");
engine.finalize_direct(function, "parameter").expect("insert failed");
```

### Draft Management

```rust
// Incremental typing (draft state)
engine.insert_str(block, "local_var").expect("insert failed");

// Query sees draft + finalized terms from visible scopes
let completions = engine.complete(block, "loc", 1);
for comp in completions {
    println!("{} (draft: {}, distance: {})",
             comp.term, comp.is_draft, comp.distance);
}

// Checkpoint/undo for editor integration
engine.checkpoint(block).expect("checkpoint failed");
engine.insert_str(block, "iable").expect("insert failed");  // "local_variable"
engine.undo(block).expect("undo failed");  // Restore to "local_var"

// Finalize draft to add to dictionary
let term = engine.finalize(block).expect("finalize failed");
```

### Complete Example

See [`examples/contextual_completion.rs`](examples/contextual_completion.rs) for a full IDE simulation.

---

## FuzzyCache (Eviction Policies)

Composable cache eviction wrappers using the **decorator pattern**. Stack policies to combine multiple eviction behaviors.

### Available Policies

| Policy | Eviction Criterion | Use Case |
|--------|-------------------|----------|
| **Noop** | None (pass-through) | Benchmarking, testing |
| **LazyInit** | N/A (deferred initialization) | Sparse dictionaries, memoization |
| **TTL** | Entry age > duration | Session caching |
| **LRU** | Least recently accessed | General-purpose |
| **Age** | Oldest insertion time (FIFO) | Fair eviction |
| **LFU** | Lowest access count | Long-lived caches |
| **CostAware** | `(age × size) / (hits + 1)` | Balance regeneration cost vs. space |
| **MemoryPressure** | `size / (hit_rate + 0.1)` | Memory-constrained environments |

### Basic Usage

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

let dict = DynamicDawg::from_terms(vec!["test", "testing", "tested"]);
let lru_cache = Lru::new(dict);

// Use with transducer - access patterns are tracked
let transducer = Transducer::new(&lru_cache, Algorithm::Standard);
for result in transducer.query("tset", 2) {
    println!("{}", result.term);
}

// Query recency for eviction decisions
if let Some(age) = lru_cache.recency("test") {
    println!("Last accessed {:?} ago", age);
}

// Evict least recently used from a candidate set
let candidates = vec!["test", "testing", "tested"];
if let Some(evicted) = lru_cache.evict_lru(&candidates) {
    println!("Evicted: {}", evicted);
}
```

### Composing Policies

Stack wrappers to combine behaviors (innermost applied first):

```rust
use liblevenshtein::cache::eviction::{Lru, Ttl, MemoryPressure};
use std::time::Duration;

let dict = DynamicDawg::from_terms(vec!["alpha", "beta", "gamma"]);

// Stack: MemoryPressure → TTL → LRU
let memory = MemoryPressure::new(dict);           // Track size/hits
let ttl = Ttl::new(memory, Duration::from_secs(300));  // 5-minute expiration
let cache = Lru::new(ttl);                        // Track recency

// Entries now subject to ALL policies:
// - Expire after 5 minutes (TTL)
// - Evict least-recently-used when needed (LRU)
// - Consider memory pressure for large entries (MemoryPressure)
```

### Thread Safety

All eviction wrappers are **thread-safe** via `Arc<RwLock<HashMap>>`:
- Multiple concurrent readers
- Exclusive writers with atomic metadata updates
- Safe to share via `Arc<Lru<D>>` across threads

---

## Additional Features

### Serialization

```rust
use liblevenshtein::prelude::*;
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

// Save with compression (85% size reduction)
let file = File::create("dict.bin.gz").expect("create failed");
GzipSerializer::<BincodeSerializer>::serialize(&dict, file).expect("serialize failed");

// Load compressed
let file = File::open("dict.bin.gz").expect("open failed");
let dict: DoubleArrayTrie = GzipSerializer::<BincodeSerializer>::deserialize(file)
    .expect("deserialize failed");
```

Requires `serialization` and `compression` features.

### Performance Optimizations

| Optimization | Effect | Feature |
|--------------|--------|---------|
| **SIMD** | 20-64% faster queries | automatic on x86_64 |
| **Bloom Filter** | 88-93% faster contains() | Built-in |
| **StatePool** | Reduced allocations | Built-in |
| **Arc Path Sharing** | Eliminated cloning | Built-in |

### CLI Tool

```bash
# Install
cargo install liblevenshtein --features cli,compression,protobuf

# Query
liblevenshtein query "test" --dict words.txt -m 2

# Convert formats
liblevenshtein convert words.txt words.bin.gz --to-format bincode-gz

# Interactive REPL
liblevenshtein repl --dict words.bin.gz
```

### WASM Support

```toml
[dependencies]
liblevenshtein = { version = "0.8", features = ["wasm"] }
```

### Grep Support (Document Extraction)

```toml
[dependencies]
liblevenshtein = { version = "0.8", features = ["grep-support"] }
```

Supports PDF, DOCX, and archive formats.

---

## Performance

### Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Automaton construction | O(\|W\|) - linear in query length |
| Dictionary traversal | O(\|D\|) - linear in dictionary edges |
| Space | O(\|W\|) states for fixed error bound |

### Benchmark Comparison

Construction and query times for 10,000 words:

| Backend | Construction | Exact Match | Distance 1 | Distance 2 |
|---------|--------------|-------------|------------|------------|
| DoubleArrayTrie | 3.2ms | 6.6µs | 12.9µs | 16.3µs |
| PathMap | 3.5ms | 71.1µs | 888µs | 5,919µs |
| DAWG | 7.2ms | 19.8µs | 319µs | 2,150µs |

---

## Feature Flags Reference

| Feature | Description |
|---------|-------------|
| `phonetic-rules` | LLev/LLRE phonetic pattern languages |
| `pathmap-backend` | PathMap dictionary backend |
| `wfst` | lling-llang WFST integration |
| `serialization` | Save/load dictionaries |
| `compression` | Gzip compression for serialization |
| `protobuf` | Protocol Buffers format |
| `cli` | Command-line interface |
| `wasm` | WebAssembly support |
| `grep-support` | Document extraction (PDF, DOCX) |
| `persistent-artrie` | Memory-mapped ARTrie |
| `bloom-filter` | Probabilistic filtering |

---

## References

- **Core Algorithm**: Schulz, Klaus U., and Stoyan Mihov. "Fast string correction with Levenshtein automata." *International Journal on Document Analysis and Recognition* 5.1 (2002): 67-85.

- **Universal Automata**: Mitankin, Petar, Stoyan Mihov, and Klaus Schulz. "Universal Levenshtein Automata. Building and Properties." *Information Processing & Management* 41.4 (2005): 687-702.

- [Algorithm Documentation](docs/research/levenshtein-automata/README.md)
- [Implementation Mapping](docs/research/levenshtein-automata/implementation-mapping.md)
- [GitHub Repository](https://github.com/universal-automata/liblevenshtein-rust)
- [Original C++ Implementation](https://github.com/universal-automata/liblevenshtein-cpp)

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
