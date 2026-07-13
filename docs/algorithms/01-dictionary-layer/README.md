# Dictionary Layer

**Navigation**: [← Back to Algorithms](../README.md) | [Next Layer: Automata →](../02-levenshtein-automata/README.md)

> **Note**: The dictionary implementations have been extracted to the
> [libdictenstein](https://github.com/f1r3fly-io/libdictenstein) crate for
> independent use. For comprehensive dictionary documentation, see:
> - [libdictenstein/docs/algorithms/](../../../../libdictenstein/docs/algorithms/README.md) - Implementation guides
> - [libdictenstein/docs/theory/](../../../../libdictenstein/docs/theory/) - SCDAWG and disk-trie theory
>
> This document focuses on how liblevenshtein integrates dictionaries with Levenshtein transducers.

## Overview

The Dictionary Layer forms the foundation of liblevenshtein's fuzzy matching capabilities. It provides pluggable backend implementations for storing and traversing collections of terms, optimized for efficient character-by-character navigation required by Levenshtein automata.

This layer abstracts over different data structures (tries, DAWGs, double-array tries) through common traits, allowing you to choose the best backend for your specific use case while maintaining a consistent API.

![Taxonomy of dictionary backends, grouping trie, DAWG, and suffix-automaton families with their byte-level and Unicode (char) variants](../../diagrams/dictionary-structures/backend-taxonomy.svg)

*Backend taxonomy: how the available dictionary implementations relate to one another.*

![The Dictionary, MappedDictionary, and DictionaryNode trait hierarchy and the methods each exposes](../../diagrams/dictionary-structures/dictionary-traits.svg)

*Dictionary trait hierarchy: the common interface every backend implements.*

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dictionary Layer API                         │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Dictionary    │  │ MappedDictionary │  │ DictionaryNode  │ │
│  │   (Trait)      │  │     (Trait)      │  │    (Trait)      │ │
│  └────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐      ┌──────────────┐    ┌────────────┐
    │   Trie   │      │    DAWG      │    │   Suffix   │
    │ Backends │      │   Backends   │    │  Automaton │
    └──────────┘      └──────────────┘    └────────────┘
         │                   │                   │
    ┌────┴────┐         ┌────┴────┐             │
    │   DAT   │         │ Dynamic │             │
    │  (rec)  │         │  DAWG   │             │
    └─────────┘         └─────────┘             │
         │                                       │
    ┌────┴────┐                                  │
    │ DAT-Char│                                  │
    │ (UTF-8) │                                  │
    └─────────┘                                  │
```

**Legend**: (rec) = recommended default

## Core Concepts

### 1. Dictionary Trait

The `Dictionary` trait defines the minimal interface for any dictionary backend:

```rust
pub trait Dictionary {
    type Node: DictionaryNode;

    fn root(&self) -> Self::Node;
    fn contains(&self, term: &str) -> bool;
    fn len(&self) -> Option<usize>;
    fn is_empty(&self) -> bool;
}
```

**Key Features**:
- **Graph-based traversal**: Navigate character-by-character through nodes
- **Backend agnostic**: Works with any underlying data structure
- **Lazy evaluation**: Only explores paths needed for fuzzy matching

### 2. DictionaryNode Trait

Nodes represent positions in the dictionary graph:

```rust
pub trait DictionaryNode: Clone + Send + Sync {
    type Unit: CharUnit;  // u8 or char

    fn is_final(&self) -> bool;
    fn transition(&self, label: Self::Unit) -> Option<Self>;
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_>;
}
```

**Key Features**:
- **Unit abstraction**: Supports both byte-level (u8) and character-level (char)
- **Lazy edge iteration**: Only compute edges when needed
- **Thread-safe**: Clone + Send + Sync for concurrent queries

### 3. MappedDictionary Trait

Extensions for dictionaries that associate values with terms:

```rust
pub trait MappedDictionary: Dictionary {
    type Value: DictionaryValue;

    fn get_value(&self, term: &str) -> Option<Self::Value>;
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where F: Fn(&Self::Value) -> bool;
}
```

**Performance Impact**: Filtering during traversal provides **10-100x speedup** compared to post-filtering.

See [Value Storage](../09-value-storage/README.md) for detailed documentation.

### 4. Character Units

The library supports two modes for handling text:

| Mode | Type | Best For | Correctness |
|------|------|----------|-------------|
| **Byte-level** | `u8` | ASCII/Latin-1, Speed | Edit distances on byte sequences |
| **Character-level** | `char` | Unicode text | Proper Unicode code point distances |

**Example**:
```rust
// Byte-level: "café" = ['c', 'a', 'f', 0xC3, 0xA9] (5 bytes)
let dict_bytes = DoubleArrayTrie::from_terms(vec!["café"]);

// Character-level: "café" = ['c', 'a', 'f', 'é'] (4 chars)
let dict_chars = DoubleArrayTrieChar::from_terms(vec!["café"]);

// Different Levenshtein distances:
// "cafe" → "café": distance 1 (char-level), distance 2 (byte-level)
```

## Available Implementations

### Production Ready (Recommended)

#### 1. DoubleArrayTrie (⭐ Default Choice)

**Best for**: General-purpose applications

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

let mut dict = DoubleArrayTrie::from_terms(vec![
    "algorithm", "approximate", "automaton"
]);
dict.insert("analysis");  // Supports runtime insertions
```

**Characteristics**:
- ⚡ **3x faster** queries than DAWG
- 💾 **8 bytes/state** memory footprint
- 🔧 **Append-only** dynamic updates
- 🎯 **Cache-efficient** BASE/CHECK arrays

[→ Detailed Guide](implementations/double-array-trie.md)

#### 2. DoubleArrayTrieChar (Unicode)

**Best for**: Multi-language applications with proper Unicode handling

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;

let mut dict = DoubleArrayTrieChar::from_terms(vec![
    "café", "naïve", "中文", "🎉"
]);
dict.insert("新しい");
```

**Characteristics**:
- ✅ **Character-level** distances
- 🌍 **Full Unicode** support (CJK, emoji, accents)
- 📊 **~5% overhead** vs byte-level
- 💾 **4x memory** for edge labels (char vs u8)

[→ Detailed Guide](implementations/double-array-trie-char.md)

#### 3. DynamicDawg

**Best for**: Applications requiring both insert and remove operations

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict = DynamicDawg::from_terms(vec!["initial", "terms"]);
dict.insert("new_term");  // ✅ Thread-safe
dict.remove("old_term");  // ✅ Supports removal
```

**Characteristics**:
- 🔒 **Thread-safe** insert AND remove
- 🔄 **Active queries** see updates immediately
- 📉 **Good performance** for fully dynamic use
- 💾 **Moderate memory** overhead

[→ Detailed Guide](implementations/dynamic-dawg.md)

#### 4. DynamicDawgChar (Unicode + Dynamic)

**Best for**: Unicode applications with full dynamic updates

```rust
use libdictenstein::dynamic_dawg::DynamicDawgChar;

let dict = DynamicDawgChar::from_terms(vec!["café", "中文"]);
dict.insert("新しい");  // ✅ Unicode + thread-safe
dict.remove("café");    // ✅ Full removal support
```

**Characteristics**:
- ✅ **Character-level** Unicode distances
- 🔒 **Thread-safe** insert and remove
- 📊 **~5% overhead** vs byte-level
- 🌍 **Full Unicode** support

[→ Detailed Guide](implementations/dynamic-dawg-char.md)

### Specialized Use Cases

#### 5. SuffixAutomaton

**Best for**: Substring/infix search within text

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let dict = SuffixAutomaton::from_source_text("the quick brown fox");
// Finds "quick" even though it's not a prefix
```

**Characteristics**:
- 🔍 **Substring matching** (not just prefixes)
- 📝 **Text indexing** use cases
- 💾 **2x memory** vs standard tries

[→ Detailed Guide](implementations/suffix-automaton.md)

#### 6. PathMapDictionary (Feature: `pathmap-backend`)

**Best for**: Frequent updates with simpler structure

```rust
#[cfg(feature = "pathmap-backend")]
use libdictenstein::pathmap::PathMapDictionary;

let dict = PathMapDictionary::from_terms(vec!["test"]);
dict.insert("new");  // Simpler internal structure
```

**Characteristics**:
- 📦 **Simple structure** for updates
- 🔒 **Thread-safe**
- 📉 **2-3x slower** than DoubleArrayTrie
- 💾 **Higher memory** usage

### Removed backends (historical note)

The classic static **`DawgDictionary`** and the arena-optimized **`OptimizedDawg`** were
**removed in the 0.9.x line**, when the dictionary backends were extracted into the
[`libdictenstein`](https://crates.io/crates/libdictenstein) crate. Their roles are now
served by current backends:

- **`DynamicDawg`** / **`DynamicDawgChar`** — a minimized DAWG that *additionally* supports
  runtime insertions and removals (with SIMD and Bloom-filter acceleration) — for the
  directed-acyclic-word-graph structure itself; and
- **`DoubleArrayTrie`** / **`DoubleArrayTrieChar`** — for static, read-optimized
  dictionaries where the term set is fixed after construction.

Code that imported either removed type should migrate to one of these; the
[decision guide](#decision-guide) below selects the right one for each workload.

## Decision Guide

### Quick Selection Flowchart

```
Start: What do you need?
│
├─ Need to remove terms? ──Yes──> Unicode? ─Yes─> DynamicDawgChar
│   │                                │
│   No                               └─No──> DynamicDawg
│   │
├─ Unicode text? ──Yes──> DoubleArrayTrieChar
│   │
│   No
│   │
└─> DoubleArrayTrie (recommended default)
```

### Detailed Comparison Table

| Feature | DAT | DAT-Char | DynDAWG | DynDAWG-Char | PathMap | SuffixAuto |
|---------|-----|----------|---------|--------------|---------|------------|
| **Query Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Construction** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Insert** | ✅ Append | ✅ Append | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Remove** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Union** | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Clone Cost** | `$\mathcal{O}(n)$` | `$\mathcal{O}(n)$` | `$\mathcal{O}(1)$` | `$\mathcal{O}(1)$` | `$\mathcal{O}(1)$` | N/A |
| **Clone Sharing** | ❌ Deep | ❌ Deep | ✅ Arc | ✅ Arc | ✅ Arc×2 | N/A |
| **Unicode** | Byte | ✅ Char | Byte | ✅ Char | Byte | Byte |
| **Thread-Safe** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Use Case** | General | Unicode | Dynamic | Dyn+Unicode | Simple | Substring |

## Performance Benchmarks

Based on 10,000-word dictionary:

### Construction Time

```
DoubleArrayTrie:     3.2ms
DoubleArrayTrieChar: 3.4ms  (+6%)
PathMapDictionary:   3.5ms  (+9%)
DynamicDawg:         4.1ms  (+28%)
```

### Exact Match (single term)

```
DoubleArrayTrie:     6.6µs
DoubleArrayTrieChar: 6.9µs  (+5%)
PathMapDictionary:   71.1µs (+977%)
```

### Contains Check (100 terms)

```
DoubleArrayTrie:     0.22µs per check
DoubleArrayTrieChar: 0.23µs (+5%)
PathMapDictionary:   132µs  (+59900%)
```

### Fuzzy Search (max distance 2)

```
DoubleArrayTrie:     16.3µs
DoubleArrayTrieChar: 17.1µs  (+5%)
PathMapDictionary:   5,919µs (+36200%)
```

**Key Takeaway**: DoubleArrayTrie variants are consistently 3-30x faster than alternatives for fuzzy matching workloads.

## Memory Characteristics

### Per-State Memory (approximate)

```
DoubleArrayTrie:     8 bytes/state
DoubleArrayTrieChar: 12 bytes/state (char labels = 4x u8)
DynamicDawg:         24 bytes/state (Arc overhead)
PathMapDictionary:   32 bytes/state (HashMap overhead)
SuffixAutomaton:     48 bytes/state (suffix links)
```

### Example: 50,000 terms

```
DoubleArrayTrie:     ~800 KB
DoubleArrayTrieChar: ~1.2 MB
DynamicDawg:         ~2.4 MB
PathMapDictionary:   ~3.2 MB
```

## Common Use Cases

### 1. Web Application Autocomplete

**Recommendation**: `DoubleArrayTrie` or `DoubleArrayTrieChar`

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

// Initialize once at startup
let dict = DoubleArrayTrie::from_terms(load_product_names());

// Per-request fuzzy search
fn autocomplete(query: &str, max_distance: usize) -> Vec<String> {
    let automaton = LevenshteinAutomaton::new(query, max_distance, Algorithm::Standard);
    automaton.query(&dict).collect()
}
```

**Why**: Fast queries (microseconds), low memory, append-only updates for new products.

### 2. Multi-Language Spell Checker

**Recommendation**: `DoubleArrayTrieChar`

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;

let dict = DoubleArrayTrieChar::from_terms(vec![
    // English
    "color", "colour",
    // Spanish
    "niño", "año",
    // Chinese
    "你好", "世界",
    // Emoji
    "😀", "🎉"
]);

// Correct Levenshtein distances for all languages
```

**Why**: Character-level distances handle accents, CJK, emoji correctly.

### 3. Real-Time Collaborative Editor

**Recommendation**: `DynamicDawg` or `DynamicDawgChar`

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict = DynamicDawg::new();

// User adds word to personal dictionary
dict.insert("refactoring");

// User removes word
dict.remove("typo");

// Active autocomplete queries see changes immediately
```

**Why**: Thread-safe insert/remove, queries reflect updates instantly.

### 4. Code Completion with Scope Filtering

**Recommendation**: `DoubleArrayTrie<u32>` with values

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("println", 1),   // Global scope
    ("format", 1),    // Global scope
    ("my_var", 42),   // Local scope 42
    ("temp", 42),     // Local scope 42
]);

// Query only local scope (10-100x faster than post-filtering)
let results = query_with_filter(&dict, "temp", 2, |scope| *scope == 42);
```

**Why**: Value filtering during traversal is dramatically faster. See [Value Storage](../09-value-storage/README.md).

### 5. Document Search (Substring Matching)

**Recommendation**: `SuffixAutomaton`

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let doc = "The quick brown fox jumps over the lazy dog";
let dict = SuffixAutomaton::from_source_text(doc);

// Find "quick" even though it's not at the beginning
let results = fuzzy_search(&dict, "quik", 1);  // Finds "quick"
```

**Why**: Matches substrings anywhere in text, not just prefixes.

### 6. Merging User and System Dictionaries

**Recommendation**: `DynamicDawg` or `PathMapDictionary` with values

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

// System-wide default frequencies
let system_dict: DynamicDawg<u32> = DynamicDawg::new();
system_dict.insert_with_value("algorithm", 1000);
system_dict.insert_with_value("database", 800);

// User-specific word frequencies
let user_dict: DynamicDawg<u32> = DynamicDawg::new();
user_dict.insert_with_value("algorithm", 50);  // User types this often
user_dict.insert_with_value("refactoring", 30); // User-specific term

// Merge: prioritize user frequencies but include system terms
system_dict.union_with(&user_dict, |system_freq, user_freq| {
    // Boost user terms by 10x for better autocomplete ranking
    user_freq * 10 + system_freq
});

// Result: "algorithm" = 1500 (50*10 + 1000)
//         "refactoring" = 300 (30*10 + 0)
//         "database" = 800 (unchanged)
```

**Why**: Union operations enable personalized autocomplete by combining user patterns with system defaults, custom merge logic for ranking.

**Alternative with Configuration Layers**:
```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MutableMappedDictionary;

// Default application settings
let defaults: PathMapDictionary<String> = PathMapDictionary::new();
defaults.insert_with_value("theme", "light".to_string());
defaults.insert_with_value("language", "en".to_string());

// User preferences
let user_prefs: PathMapDictionary<String> = PathMapDictionary::new();
user_prefs.insert_with_value("theme", "dark".to_string()); // Override

// Merge: user preferences override defaults (last-writer-wins)
defaults.union_replace(&user_prefs);

// Effective config: theme=dark, language=en
```

**Why**: PathMapDictionary's structural sharing makes it ideal for configuration layers with frequent snapshots.

## Integration with Levenshtein Automata

The Dictionary Layer is designed to work seamlessly with Layer 2 (Automata):

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

// Step 1: Create dictionary
let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

// Step 2: Create automaton for query
let automaton = LevenshteinAutomaton::new("tset", 1, Algorithm::Standard);

// Step 3: Query dictionary with automaton
let results: Vec<String> = automaton.query(&dict).collect();
// Results: ["test"] (distance 1: swap 's' and 'e')
```

The automaton traverses the dictionary graph using `DictionaryNode::transition()` to explore only paths within the distance threshold.

See [Automata Layer](../02-levenshtein-automata/README.md) for details.

## Thread Safety

All dictionary implementations in this library are **thread-safe for concurrent reads**:

```rust
use std::sync::Arc;
use std::thread;

let dict = Arc::new(DoubleArrayTrie::from_terms(vec!["test"]));

// Multiple threads can query simultaneously
let handles: Vec<_> = (0..4).map(|_| {
    let dict = Arc::clone(&dict);
    thread::spawn(move || {
        dict.contains("test")  // ✅ Safe
    })
}).collect();
```

For concurrent writes, dictionaries have different strategies:

| Dictionary | Strategy | Writes | Notes |
|-----------|----------|--------|-------|
| DoubleArrayTrie | `Persistent` | Rebuild + atomic swap | Append-only via builder |
| DynamicDawg | `InternalSync` | Direct mutation | Lock-free (`ArcSwap` reads; CAS writes) |
| PathMapDictionary | `InternalSync` | Direct mutation | Lock-free (`ArcSwap<PathMapState>` swap) |

## Advanced Topics

### Custom Dictionary Implementation

To implement a custom backend:

```rust
use libdictenstein::{Dictionary, DictionaryNode, CharUnit};

#[derive(Clone)]
struct MyNode {
    // Your node structure
}

impl DictionaryNode for MyNode {
    type Unit = u8;

    fn is_final(&self) -> bool {
        // Check if this node marks end of term
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        // Follow edge labeled with 'label'
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        // Return all outgoing edges
    }
}

struct MyDictionary {
    // Your dictionary structure
}

impl Dictionary for MyDictionary {
    type Node = MyNode;

    fn root(&self) -> Self::Node {
        // Return root node
    }

    fn len(&self) -> Option<usize> {
        Some(/* term count */)
    }
}
```

### Serialization

Dictionaries can be serialized for persistence:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};

let dict = DoubleArrayTrie::from_terms(vec!["test"]);

// Serialize (`Vec<u8>` implements `Write`)
let mut bytes = Vec::new();
BincodeSerializer::serialize(&dict, &mut bytes)?;
std::fs::write("dict.bin", &bytes)?;

// Deserialize (`&[u8]` implements `Read`)
let bytes = std::fs::read("dict.bin")?;
let dict: DoubleArrayTrie = BincodeSerializer::deserialize(&bytes[..])?;
```

See [Serialization Guide](../../user-guide/serialization.md) for details.

## Related Documentation

- [Value Storage](../09-value-storage/README.md) - Term-to-value mappings
- [Automata Layer](../02-levenshtein-automata/README.md) - Levenshtein automata that query dictionaries
- [Zipper Navigation](../06-zipper-navigation/README.md) - Hierarchical navigation pattern
- Performance Guide - Detailed benchmarks and optimization tips

## Academic References

### Foundational Papers

1. **Aoe, J. (1989)**. "An Efficient Digital Search Algorithm by Using a Double-Array Structure"
   - *IEEE Transactions on Software Engineering*, 15(9), 1066-1077
   - DOI: [10.1109/32.31365](https://doi.org/10.1109/32.31365)
   - 📄 Original double-array trie algorithm

2. **Yata, S., Oono, M., Morita, K., Fuketa, M., Sumitomo, T., & Aoe, J. (2007)**. "A compact static double-array keeping character codes"
   - *Information Processing & Management*, 43(1), 237-247
   - DOI: [10.1016/j.ipm.2006.06.001](https://doi.org/10.1016/j.ipm.2006.06.001)
   - 📄 Optimization techniques for DATs

3. **Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987)**. "Complete inverted files for efficient text retrieval and analysis"
   - *Journal of the ACM*, 34(3), 578-595
   - DOI: [10.1145/28869.28873](https://doi.org/10.1145/28869.28873)
   - 📄 DAWG construction algorithms

### Textbooks

4. **Gusfield, D. (1997)**. *Algorithms on Strings, Trees, and Sequences: Computer Science and Computational Biology*
   - Cambridge University Press
   - ISBN: 978-0521585194
   - 📚 Comprehensive coverage of string algorithms and suffix structures

### Open Access Resources

5. **Schulz, K. U., & Mihov, S. (2002)**. "Fast String Correction with Levenshtein Automata"
   - *International Journal on Document Analysis and Recognition*, 5(1), 67-85
   - 📄 [Available via ResearchGate](https://www.researchgate.net/)
   - Core algorithm for fuzzy matching with tries

## Next Steps

- **Deep Dive**: Read the [DoubleArrayTrie Implementation Guide](implementations/double-array-trie.md)
- **Unicode**: Learn about [DoubleArrayTrieChar](implementations/double-array-trie-char.md)
- **Values**: Explore [Value Storage](../09-value-storage/README.md)
- **Query**: Understand [Levenshtein Automata](../02-levenshtein-automata/README.md)

---

**Navigation**: [← Back to Algorithms](../README.md) | [Next Layer: Automata →](../02-levenshtein-automata/README.md)
