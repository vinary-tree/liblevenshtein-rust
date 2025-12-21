# liblevenshtein-rust

[![Crates.io](https://img.shields.io/crates/v/liblevenshtein.svg)](https://crates.io/crates/liblevenshtein)
[![CI](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/ci.yml)
[![Nightly Tests](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/nightly.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/nightly.yml)
[![Release](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/release.yml/badge.svg)](https://github.com/universal-automata/liblevenshtein-rust/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Rust implementation of [liblevenshtein](https://github.com/universal-automata/liblevenshtein-cpp) for fast approximate string matching using **Levenshtein Automata**—deterministic finite automata that enable O(|W|) construction complexity for error-tolerant dictionary queries.

Based on "Fast String Correction with Levenshtein-Automata" (Schulz & Mihov, 2002).

## Overview

This library provides efficient fuzzy string matching against large dictionaries using finite state automata. It supports multiple Levenshtein distance algorithms and pluggable dictionary backends.

## What Makes This Library Fast

Unlike naive approaches that compute Levenshtein distance for every dictionary word (O(|D| × |W| × |V|)), this library uses **Levenshtein Automata**—a deterministic finite automaton that accepts exactly all words within distance n from a query word.

### Key Algorithmic Advantages

1. **O(|W|) Automaton Construction** - Linear in query word length for fixed error bound n
2. **O(|D|) Dictionary Traversal** - Single pass through dictionary (measured as total edges)
3. **No Distance Recomputation** - Automaton guides search, avoiding per-word calculations
4. **Proven Correctness** - Accepts exactly L_Lev(n,W) = {V | d_L(W,V) ≤ n}

**Result:** Query time is effectively independent of dictionary size for well-distributed tries!

**Theory:** Based on "Fast String Correction with Levenshtein-Automata" (Schulz & Mihov, 2002). See [complete algorithm documentation](docs/research/levenshtein-automata/) for algorithmic details including Position structures, Subsumption relations, and Characteristic vectors.

## Theoretical Background

### How Levenshtein Automata Work

Traditional approaches compute edit distance for every dictionary word, requiring O(|D| × |W| × |V|) operations. This library uses **Levenshtein Automata**—deterministic finite automata that accept exactly all words within distance n from query word W:

```
LEV_n(W) accepts L_Lev(n,W) = {V | d_L(W,V) ≤ n}
```

### Key Concepts

**Position (i#e)**: Represents "matched i characters of W with e errors"
- Example: 3#1 means "matched 3 characters, used 1 error"
- Range: 0 ≤ i ≤ |W|, 0 ≤ e ≤ n

**Subsumption (⊑)**: Eliminates redundant positions for minimal state sets
- Position i#e subsumes j#f if: (e < f) ∧ (|j-i| ≤ f-e)
- Any word accepted from j#f is also accepted from i#e

**Characteristic Vectors (χ)**: Encode where characters appear in query word
- χ(x,W) = ⟨b₁,...,b_w⟩ where b_i = 1 if W[i] = x
- Used to determine valid transitions without recomputing distances

**Elementary Transitions**: Move between positions based on dictionary character
- Defined in Tables 4.1, 7.1, 8.1 of the paper for each algorithm variant
- Enable O(1) state updates during traversal

**Result:** Automaton construction is O(|W|), dictionary traversal is O(|D|), regardless of dictionary size!

### Foundational Papers

**Core Algorithm:**
- Schulz, Klaus U., and Stoyan Mihov. "Fast string correction with Levenshtein automata." *International Journal on Document Analysis and Recognition* 5.1 (2002): 67-85.

**Extensions:**
- Mitankin, Petar, Stoyan Mihov, and Klaus Schulz. "Universal Levenshtein Automata. Building and Properties." *Information Processing & Management* 41.4 (2005): 687-702.

### Documentation

- [Complete Algorithm Documentation](docs/research/levenshtein-automata/README.md) - Detailed paper summary
- [Implementation Mapping](docs/research/levenshtein-automata/implementation-mapping.md) - Code-to-paper correspondence
- [Glossary](docs/research/levenshtein-automata/glossary.md) - Terms and notation reference
- [Universal Levenshtein Automata Research](docs/research/universal-levenshtein/) - Restricted substitutions
- [Weighted Distance Research](docs/research/weighted-levenshtein-automata/) - Variable operation costs

### Features

- **Fast approximate string matching** using Levenshtein Automata with **O(|W|) construction** complexity
- **Multiple algorithms** (all maintaining O(|W|) construction complexity):
  - `Standard`: Insert, delete, substitute operations (Chapters 4-6 of foundational paper)
    - Example: "hello" → "helo" (delete), "helo" → "hello" (insert), "hello" → "hallo" (substitute)
    - Use for: General fuzzy matching, spell checking
  - `Transposition`: Adds adjacent character swaps (Chapter 7) - Damerau-Levenshtein distance
    - Example: "teh" → "the" costs 1 (not 2), "recieve" → "receive" costs 1
    - Use for: Typing errors, keyboard input corrections
  - `MergeAndSplit`: Adds two-character ↔ one-character operations (Chapter 8)
    - Example: "rn" ↔ "m", "vv" ↔ "w" (common in OCR)
    - Use for: Scanned documents, character recognition systems
- **Multiple dictionary backends**:
  - **DoubleArrayTrie** (recommended, default) - O(1) transitions with excellent cache locality and minimal memory footprint
  - **DoubleArrayTrieChar** - Character-level variant for correct Unicode Levenshtein distances
  - **PathMap** (optional `pathmap-backend` feature) - high-performance trie with structural sharing, supports dynamic updates
  - **PathMapChar** (optional `pathmap-backend` feature) - Character-level PathMap for Unicode fuzzy matching
  - **DAWG** - Directed Acyclic Word Graph for space-efficient storage
  - **OptimizedDawg** - Arena-based DAWG with 20-25% faster construction
  - **DynamicDawg** - DAWG with online insert/delete/minimize operations
  - **DynamicDawgChar** - Character-level DAWG with Unicode support and online updates
  - **SuffixAutomaton** - For substring/infix matching
  - **SuffixAutomatonChar** - Character-level suffix automaton for Unicode substring matching
  - Extensible trait-based design for custom backends
- **Ordered results**: `OrderedQueryIterator` returns results sorted by distance first, then lexicographically
- **Filtering and prefix matching**: Filter results with custom predicates, enable prefix mode for code completion
- **Serialization support** (optional `serialization` feature):
  - **Multiple formats**: Bincode (binary), JSON, Plain Text (newline-delimited UTF-8), Protobuf
  - **Optimized Protobuf formats**: V2 (62% smaller), DAT-specific, SuffixAutomaton-specific
  - **Gzip compression** (optional `compression` feature) - 85% file size reduction
  - Save and load dictionaries to/from disk
- **Full-featured CLI tool** (optional `cli` feature):
  - Interactive REPL for exploration
  - Query, insert, delete, convert operations
  - Support for all serialization formats including compressed variants
- **Runtime dictionary updates**:
  - Thread-safe insert, remove, and clear operations (PathMap, DynamicDawg)
  - Queries automatically see updates via `RwLock`-based interior mutability
  - Concurrent queries during modifications
- **Lazy evaluation** - results generated on-demand
- **Efficient memory usage** - shared dictionary state across transducers
- **SIMD acceleration** (optional `simd` feature, x86_64 only):
  - AVX2/SSE4.1 optimized operations with runtime CPU feature detection
  - **20-64% faster** query performance across all workloads
  - Optimized characteristic vectors, position subsumption, state operations
  - Dictionary edge lookup with data-driven threshold tuning
  - Distance matrix computation with vectorized operations
  - Automatic fallback to scalar implementation when SIMD unavailable
- **WASM/WASI support** (optional `wasm` feature):
  - WebAssembly compilation for browser and Node.js targets
  - WASI support for server-side WebAssembly runtimes
  - `wasm-phonetic` feature combines WASM with phonetic rules
  - C FFI via `ffi` feature for native integration
- **LLev/LLRE - Phonetic Pattern Languages** (requires `phonetic-rules` feature):
  - `.llev` format for phonetic rewrite rules with metadata and includes
  - `.llre` format for regex-style patterns with NFA compilation
  - Regex-like syntax: quantifiers (`*`, `+`, `?`, `{n,m}`), alternation, grouping
  - Built-in named character classes (`[:alpha:]`, `[:vowel:]`, etc.)
  - NFA engine with automatic optimization (epsilon elimination, dead state removal)
  - AOT compilation to binary format for instant loading
  - See [LLev Grammar](docs/grammar/llev.ebnf) and [LLRE Grammar](docs/grammar/llre.ebnf)

## Installation

### From crates.io (Recommended)

Add to your `Cargo.toml`:

```toml
[dependencies]
liblevenshtein = "0.8"

# Or with SIMD acceleration (x86_64 only, requires SSE4.1/AVX2):
liblevenshtein = { version = "0.8", features = ["simd"] }
```

Or install the CLI tool:

```bash
cargo install liblevenshtein --features cli,compression,protobuf
```

### Pre-built Packages

Download pre-built packages from the [GitHub Releases](https://github.com/universal-automata/liblevenshtein-rust/releases) page:

- **Debian/Ubuntu**: `.deb` packages
- **Fedora/RHEL/CentOS**: `.rpm` packages
- **Arch Linux**: `.pkg.tar.zst` packages
- **macOS**: `.dmg` disk images for easy installation (x86_64 and ARM64)
- **Binaries**: `.tar.gz` and `.zip` archives for Linux and macOS (x86_64 and ARM64)

## Usage

### Basic Querying

```rust
use liblevenshtein::prelude::*;

// Create a dictionary from terms (using DoubleArrayTrie for best performance)
let terms = vec!["test", "testing", "tested", "tester"];
let dict = DoubleArrayTrie::from_terms(terms);

// Create a transducer with Standard algorithm
let transducer = Transducer::new(dict, Algorithm::Standard);

// Query for terms within edit distance 2
for term in transducer.query("tset", 2) {
    println!("Match: {}", term);
}

// Query with distances
for candidate in transducer.query_with_distance("tset", 2) {
    println!("Match: {} (distance: {})", candidate.term, candidate.distance);
}
```

**Output:**
```
Match: test
Match: tester
Match: test (distance: 1)
Match: tester (distance: 2)
```

### Unicode Support

For correct character-level Levenshtein distances with Unicode text, use the character-level dictionary variants:

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
use liblevenshtein::prelude::*;

// Create a character-level dictionary for Unicode content
let terms = vec!["café", "naïve", "中文", "🎉"];
let dict = DoubleArrayTrieChar::from_terms(terms);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Character-level distance: "" → "¡" is distance 1 (one character)
// Byte-level would incorrectly report distance 2 (two UTF-8 bytes)
let results: Vec<_> = transducer.query("", 1).collect();
// Results include single-character Unicode terms

// Works with accented characters
for candidate in transducer.query_with_distance("cafe", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
    // Output: café: 1 (one character substitution: e→é)
}
```

**Character-level backends**:
- `DoubleArrayTrieChar` - Character-level Double-Array Trie
- `PathMapDictionaryChar` (requires `pathmap-backend` feature) - Character-level PathMap with dynamic updates

**When to use**:
- ✅ Dictionaries containing non-ASCII Unicode (accented characters, CJK, emoji)
- ✅ When edit distance must count characters, not bytes
- ✅ Multi-language applications requiring correct Unicode semantics

**Performance**: ~5% overhead for UTF-8 decoding, 4x memory for edge labels (char vs u8).

### Runtime Dictionary Updates

The **DynamicDawg** backend supports **thread-safe runtime updates**:

```rust
use liblevenshtein::prelude::*;

// Create dictionary with runtime update support
let dict = DynamicDawg::from_terms(vec!["cat", "dog"]);
let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

// Insert new terms at runtime
dict.insert("cot");
dict.insert("bat");

// Queries immediately see the new terms
let results: Vec<_> = transducer.query("cot", 1).collect();
// Results: ["cat", "cot"]

// Remove terms
dict.remove("dog");
```

**Thread Safety**: The dictionary uses `RwLock` for interior mutability:
- Multiple concurrent queries are allowed (read locks)
- Modifications acquire exclusive write locks
- Active `Transducer` instances automatically see updates

**Performance Optimizations**: Configure Bloom filter and auto-minimization for better performance:

```rust
use liblevenshtein::prelude::*;

// Enable Bloom filter for 10x faster contains() operations
let dict = DynamicDawg::with_config(
    f32::INFINITY,  // Auto-minimize threshold (disabled)
    Some(10000),    // Bloom filter capacity (enabled)
);

// Or enable auto-minimization for bulk insertions (30% faster for large datasets)
let dict = DynamicDawg::with_config(
    1.5,            // Auto-minimize at 50% growth
    None,           // Bloom filter (disabled)
);

// Or enable both optimizations
let dict = DynamicDawg::with_config(
    1.5,            // Auto-minimize threshold
    Some(10000),    // Bloom filter capacity
);
```

**Optimization Guide**:
- **Bloom Filter**: Use when frequently checking if terms exist (88-93% faster `contains()`)
- **Auto-Minimization**: Use for bulk insertions of 1000+ terms (30% faster, prevents memory bloat)
- Default: Both disabled for maximum flexibility and minimal overhead on small datasets

**Note**: PathMapDictionary also supports runtime updates but requires the optional `pathmap-backend` feature.

See [`examples/dynamic_dictionary.rs`](examples/dynamic_dictionary.rs) for a complete demonstration.

### Value-Mapped Dictionaries

Store metadata with dictionary terms using **value-mapped dictionaries**:

```rust
use liblevenshtein::prelude::*;
use std::collections::HashSet;

// DynamicDawg with integer values (e.g., word frequencies)
let dict: DynamicDawg<u32> = DynamicDawg::new();
dict.insert_with_value("apple", 42);
dict.insert_with_value("apply", 17);

// Retrieve values
assert_eq!(dict.get_value("apple"), Some(42));
assert_eq!(dict.get_value("banana"), None);

// Use with FuzzyMap for fuzzy lookups with values
let map = FuzzyMap::new(dict, Algorithm::Standard);
let results = map.get_with_distance("aple", 1);  // fuzzy lookup
// Results include both terms and their values

// Or use FuzzyMultiMap to aggregate multiple values
let dict: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
dict.insert_with_value("test", HashSet::from([1, 2, 3]));
```

**Supported backends**:
- `DynamicDawg<V>` - Dynamic dictionary with values of type `V`
- `DynamicDawgChar<V>` - Character-level dynamic dictionary with Unicode support
- `PathMapDictionary<V>` - PathMap with values (requires `pathmap-backend`)
- `PathMapDictionaryChar<V>` - Character-level PathMap with values

**Common value types**:
- `u32` / `u64` - Frequencies, scores, IDs
- `HashSet<T>` - Multiple associations per term
- `Vec<T>` - Ordered collections
- Any type implementing `Clone + Send + Sync + 'static`

**Integration with contextual completion**:
```rust
use liblevenshtein::prelude::*;

// Use DynamicDawg backend for contextual completion
let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
let engine = ContextualCompletionEngine::with_dictionary(
    dict,
    Algorithm::Standard
);

// Insert terms with context IDs
let ctx = engine.create_root_context();
engine.insert_finalized(ctx, "variable", vec![ctx]);
```

### Ordered Results

Get results sorted by edit distance first, then alphabetically:

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec!["apple", "apply", "ape", "app"]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Results ordered by distance, then alphabetically
for candidate in transducer.query_ordered("aple", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
ape: 1
apple: 1
apply: 1
```

### Filtering and Prefix Matching

Filter results and enable prefix matching for code completion:

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "getValue", "getVariable", "setValue", "setVariable"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Prefix matching with filtering
for candidate in transducer
    .query_ordered("getVal", 1)
    .prefix()  // Match terms starting with query ± edits
    .filter(|c| c.term.starts_with("get"))  // Only getter methods
{
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
getValue: 0
getVariable: 1
```

See [`examples/code_completion_demo.rs`](examples/code_completion_demo.rs) and [`examples/contextual_filtering_optimization.rs`](examples/contextual_filtering_optimization.rs) for more examples.

### Dictionary Backend Comparison

The library provides multiple dictionary backends optimized for different use cases:

| Backend | Best For | Performance Highlights |
|---------|----------|----------------------|
| **DoubleArrayTrie** | **General use** (recommended) | 3x faster queries, 30x faster contains, 8 bytes/state |
| **DoubleArrayTrieChar** | Unicode text, character-level distances | Correct Unicode semantics, ~5% overhead |
| **PathMap** | Dynamic updates, runtime modifications | Thread-safe insert/delete, fastest dynamic backend |
| **PathMapChar** | Unicode + dynamic updates | Character-level distances with runtime modifications |
| **DAWG** | Static dictionaries, moderate size | Good balance of speed and memory |
| **OptimizedDawg** | Static dictionaries, construction speed | 20-25% faster construction than DAWG |
| **DynamicDawg** | Occasional modifications | Best fuzzy matching for dynamic use |
| **DynamicDawgChar** | Unicode + occasional modifications | Character-level with insert/remove, ~5% overhead |
| **SuffixAutomaton** | Substring/infix matching | Find patterns anywhere in text |

**Performance Comparison** (10,000 words):

```
Construction:     DAT: 3.2ms   PathMap: 3.5ms   DAWG: 7.2ms
Exact Match:      DAT: 6.6µs   DAWG: 19.8µs     PathMap: 71.1µs
Contains (100):   DAT: 0.22µs  DAWG: 6.7µs      PathMap: 132µs
Distance 1:       DAT: 12.9µs  DAWG: 319µs      PathMap: 888µs
Distance 2:       DAT: 16.3µs  DAWG: 2,150µs    PathMap: 5,919µs
Memory/state:     DAT: ~8B     OptDawg: ~13B    DAWG: ~32B
```

**Why DoubleArrayTrie is Fastest:**
- **O(1) transitions** - Direct array indexing vs pointer chasing
- **Excellent cache locality** - Sequential memory layout minimizes cache misses
- **Minimal memory footprint** - ~8 bytes per state (vs 32 bytes for DAWG)
- **Perfect for parallel traversal** - Optimal implementation of dictionary automaton A^D from paper

All backends implement the dictionary automaton A^D that is traversed in parallel with the Levenshtein automaton LEV_n(W) during queries. The choice of backend affects the constant factors in the O(|D|) query complexity, with DoubleArrayTrie providing the smallest constants due to hardware-friendly memory access patterns.

**Prefix Matching Support**: All backends except `SuffixAutomaton` support efficient prefix matching through the `.prefix()` method, making them ideal for code completion and autocomplete applications.

**When to use each backend**:
- **Static dictionaries (ASCII/Latin-1)** → `DoubleArrayTrie` (best overall performance, default choice)
- **Static dictionaries (Unicode)** → `DoubleArrayTrieChar` (correct character-level distances)
- **Dynamic updates (ASCII/Latin-1)** → `DynamicDawg` (thread-safe runtime modifications)
- **Dynamic updates (Unicode)** → `DynamicDawgChar` (character-level with insert/remove, best fuzzy matching)
- **Dynamic updates (Unicode, frequent)** → `PathMapChar` (alternative, requires `pathmap-backend`)
- **Substring search** → `SuffixAutomaton` (finds patterns anywhere in text)
- **Memory-constrained** → `DoubleArrayTrie` (8 bytes/state, most efficient)
- **Multi-language apps** → Character-level variants (`*Char`) for correct Unicode semantics

### Serialization and Compression

Save and load dictionaries with optional compression:

```rust
use liblevenshtein::prelude::*;
use std::fs::File;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

// Save with compression (85% file size reduction)
let file = File::create("dict.bin.gz")?;
GzipSerializer::<BincodeSerializer>::serialize(&dict, file)?;

// Load compressed dictionary
let file = File::open("dict.bin.gz")?;
let dict: DoubleArrayTrie = GzipSerializer::<BincodeSerializer>::deserialize(file)?;

// Or use optimized Protobuf formats
let file = File::create("dict.pb")?;
OptimizedProtobufSerializer::serialize(&dict, file)?;  // 62% smaller than standard
```

Requires `serialization` and `compression` features:

```toml
[dependencies]
liblevenshtein = { git = "https://github.com/universal-automata/liblevenshtein-rust", tag = "v0.8.0", features = ["serialization", "compression"] }
```

### CLI Tool

The library includes a full-featured command-line tool with interactive REPL:

```bash
# Install with CLI support (from GitHub)
cargo install --git https://github.com/universal-automata/liblevenshtein-rust --tag v0.8.0 \
  --features cli,compression,protobuf liblevenshtein

# Or download pre-built binaries from GitHub Releases

# Query a dictionary
liblevenshtein query "test" --dict /usr/share/dict/words -m 2 -s

# Convert between formats with compression
liblevenshtein convert words.txt words.bin.gz \
  --to-format bincode-gz --to-backend path-map

# Launch interactive REPL
liblevenshtein repl --dict words.bin.gz
```

The CLI auto-detects formats from file extensions and supports:
- **Formats**: text, bincode, json, protobuf (plus `-gz` compressed variants)
- **Backends**: path-map, dawg, dynamic-dawg
- **Operations**: query, insert, delete, convert, save, info

See [`docs/developer-guide/building.md`](docs/developer-guide/building.md) for comprehensive CLI documentation.

### FuzzyMultiMap - Aggregating Values

The `FuzzyMultiMap` enables fuzzy lookup of associated values, aggregating results from all matching terms:

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use std::collections::HashSet;

// Create a dictionary with values (e.g., user IDs for each name)
let dict = PathMapDictionary::from_terms_with_values([
    ("alice", HashSet::from([101, 102])),
    ("alicia", HashSet::from([103])),
    ("bob", HashSet::from([201])),
    ("robert", HashSet::from([202, 203])),
]);

// Create fuzzy multimap
let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

// Query "alise" - matches both "alice" and "alicia" at distance 1
let user_ids = fuzzy_map.query("alise", 1).unwrap();
// Returns: HashSet {101, 102, 103} - union of all matching values

// Practical use case: find all IDs for fuzzy name search
let ids = fuzzy_map.query("rob", 2).unwrap();
// Returns IDs for both "bob" (distance 1) and "robert" (distance 2)
```

**Supported collection types**:
- `HashSet<T>` - Union of all matching sets
- `BTreeSet<T>` - Union of all matching sets
- `Vec<T>` - Concatenation of all matching vectors

**Use cases**:
- User ID lookup with fuzzy name matching
- Tag aggregation across similar terms
- Multi-valued dictionary queries with error tolerance

### Contextual Code Completion (Zipper-Based)

The `ContextualCompletionEngine` provides hierarchical scope-aware code completion with draft state management:

```rust
use liblevenshtein::contextual::ContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

// Create engine (thread-safe, shareable with Arc)
let engine = ContextualCompletionEngine::with_algorithm(Algorithm::Standard);

// Create hierarchical scopes (global → function → block)
let global = engine.create_root_context(0);
let function = engine.create_child_context(1, global).unwrap();
let block = engine.create_child_context(2, function).unwrap();

// Add finalized terms to each scope
engine.finalize_direct(global, "std::vector").unwrap();
engine.finalize_direct(global, "std::string").unwrap();
engine.finalize_direct(function, "parameter").unwrap();
engine.finalize_direct(function, "result").unwrap();

// Incremental typing in block scope (draft state)
engine.insert_str(block, "local_var").unwrap();

// Query completions - sees all visible scopes (block + function + global)
let completions = engine.complete(block, "par", 1);
for comp in completions {
    println!("{} (distance: {}, draft: {}, from context: {:?})",
             comp.term, comp.distance, comp.is_draft, comp.contexts);
}
// Output: parameter (distance: 0, draft: false, from context: [1])

// Checkpoint/undo support for editor integration
engine.checkpoint(block).unwrap();
engine.insert_str(block, "iable").unwrap();  // Now: "local_variable"
engine.undo(block).unwrap();  // Restore to: "local_var"

// Finalize draft to add to dictionary
let term = engine.finalize(block).unwrap();  // Returns "local_var"
assert!(engine.has_term("local_var"));

// Query now sees finalized term
let results = engine.complete(block, "loc", 1);
// Returns: local_var (now finalized, visible in this context)
```

**Features**:
- **Hierarchical visibility**: Child contexts see parent terms (global → function → block)
- **Draft state**: Incremental typing without polluting finalized dictionary
- **Checkpoint/undo**: Editor-friendly state management
- **Thread-safe**: Share engine across threads with `Arc`
- **Mixed queries**: Search both drafts and finalized terms simultaneously

**Performance** (sub-millisecond for interactive use):
- Insert character: ~4 µs
- Checkpoint: ~116 ns
- Query (500 terms, distance 1): ~11.5 µs
- Query (distance 2): ~309 µs

**Use cases**:
- LSP servers with multi-file scope awareness
- Code editors with context-sensitive completion
- REPL environments with session-scoped symbols
- Any application requiring hierarchical fuzzy matching

See [`examples/contextual_completion.rs`](examples/contextual_completion.rs) for a complete example.

### Prefix-Based Iteration (PrefixZipper)

For efficient autocomplete and code completion scenarios, use the `PrefixZipper` trait to navigate directly to a prefix and iterate only matching terms. This is **5-10× faster** than full dictionary iteration with filtering when the prefix matches a small subset of terms.

**Performance**: O(k) navigation + O(m) iteration, where k = prefix length, m = matching terms.

```rust
use liblevenshtein::prelude::*;
use liblevenshtein::dictionary::prefix_zipper::PrefixZipper;
use liblevenshtein::dictionary::double_array_trie_zipper::DoubleArrayTrieZipper;

let terms = vec!["process", "processUser", "produce", "product", "apple"];
let dict = DoubleArrayTrie::from_terms(terms.iter());

// Create zipper and navigate to prefix
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
if let Some(iter) = zipper.with_prefix(b"proc") {
    for (path, _zipper) in iter {
        let term = String::from_utf8(path).unwrap();
        println!("Found: {}", term);
        // Prints only: "process" and "processUser" (not all 5 terms!)
    }
}
```

**Unicode support** (character-level):

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
use liblevenshtein::dictionary::double_array_trie_char_zipper::DoubleArrayTrieCharZipper;
use liblevenshtein::dictionary::prefix_zipper::PrefixZipper;

let terms = vec!["café", "cafétéria", "naïve"];
let dict = DoubleArrayTrieChar::from_terms(terms.iter());

let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);
let prefix: Vec<char> = "caf".chars().collect();

if let Some(iter) = zipper.with_prefix(&prefix) {
    for (path, _) in iter {
        let term: String = path.iter().collect();
        println!("Found: {}", term);
    }
}
```

**Valued dictionaries** (for metadata like scope IDs, frequencies, etc.):

```rust
use liblevenshtein::dictionary::prefix_zipper::ValuedPrefixZipper;

let dict = DoubleArrayTrie::from_terms_with_values(
    vec![("cat", 1), ("cats", 2), ("dog", 3)].into_iter()
);

let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
if let Some(iter) = zipper.with_prefix_values(b"cat") {
    for (path, value) in iter {
        let term = String::from_utf8(path).unwrap();
        println!("{} -> {}", term, value);
        // Prints: "cat -> 1", "cats -> 2"
    }
}
```

**Use cases**:
- Code completion / autocomplete
- Prefix search in large dictionaries
- Pattern-aware completion (LSP servers)
- Any scenario requiring "terms starting with X"

**Backend support**: All dictionary backends support `PrefixZipper` (DoubleArrayTrie, DynamicDawg, PathMap, etc.) through blanket trait implementation.

### Advanced: Direct Zipper API

For fine-grained control over traversal, use the zipper API directly:

```rust
use liblevenshtein::dictionary::pathmap::PathMapDictionary;
use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
use liblevenshtein::transducer::intersection_zipper::IntersectionZipper;

// Create dictionary
let dict = PathMapDictionary::<()>::new();
dict.insert("cat");
dict.insert("cats");
dict.insert("dog");

// Create zippers
let dict_zipper = PathMapZipper::new_from_dict(&dict);
let auto_zipper = AutomatonZipper::new("cot".as_bytes(), 1, Algorithm::Standard);

// Intersect dictionary and automaton
let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

// Manual traversal
let mut pool = StatePool::new();
for (label, child) in intersection.children(&mut pool) {
    if child.is_match() {
        println!("Match: {} (distance: {})",
                 child.term(),
                 child.distance().unwrap());
    }

    // Recurse into child for custom traversal algorithms
    for (_, grandchild) in child.children(&mut pool) {
        // Custom logic here...
    }
}
```

**When to use**:
- Custom traversal algorithms (DFS, A*, beam search)
- Early termination with custom heuristics
- Integration with external data structures
- Research and experimentation

**Note**: Most users should use `Transducer::query()` or `ContextualCompletionEngine` instead. The zipper API is lower-level and requires manual state management.

### Value-Filtered Queries

For performance optimization when querying dictionaries with scoped values:

```rust
use liblevenshtein::prelude::*;
use std::collections::HashSet;

// Dictionary with scope IDs
let dict = PathMapDictionary::from_terms_with_values([
    ("global_var", 0),      // Scope 0 (global)
    ("function_param", 1),  // Scope 1 (function)
    ("local_var", 2),       // Scope 2 (block)
]);

let transducer = Transducer::new(dict, Algorithm::Standard);

// Query only specific scopes (e.g., visible from scope 1)
let visible_scopes = HashSet::from([0, 1]);  // global + function
for term in transducer.query_by_value_set("param", 1, &visible_scopes) {
    println!("{}", term);
}
// Output: function_param (scope 1 is visible)
// Does NOT return: local_var (scope 2 not in visible set)

// Or use custom predicate
for term in transducer.query_filtered("var", 1, |scope_id| *scope_id <= 1) {
    println!("{}", term);
}
// Returns: global_var, function_param (scopes 0, 1)
```

**Performance benefit**: Filtering by value during traversal is **significantly faster** than post-filtering results, especially for large dictionaries with many out-of-scope matches.

**Use cases**:
- Scope-based code completion (only show visible symbols)
- Access control (filter by user permissions)
- Multi-tenancy (filter by tenant ID)

### Phonetic Rewrite Rules (Formally Verified)

The `phonetic-rules` feature provides **mathematically verified** phonetic transformation rules for fuzzy matching. Use `PhoneticGrep` for on-the-fly matching with automatic Levenshtein composition:

```rust
use liblevenshtein::phonetic::grep::PhoneticGrep;

// PhoneticGrep composes NFA patterns with Levenshtein automata
// No preprocessing or normalization required!
let grep = PhoneticGrep::from_pattern("phone", 1)?;

// Matches within edit distance 1
assert!(grep.matches("phone").is_some());  // exact match (distance 0)
assert!(grep.matches("fone").is_some());   // distance 1 (p→f)
assert!(grep.matches("phon").is_some());   // distance 1 (missing e)
assert!(grep.matches("phones").is_some()); // distance 1 (extra s)
```

**Named Character Classes** (phonetic-aware):
```rust
// [:vowel:] includes ASCII vowels + IPA vowels (ə, ɪ, ʊ, ɛ, etc.)
// [:consonant:] includes ASCII consonants + IPA symbols
// [:fricative:] matches f,v,s,z,h + digraphs (sh,th,zh)
// [:nasal:] matches m,n + ng digraph + IPA ŋ
// [:stop:] / [:plosive:] matches p,b,t,d,k,g
// [:voiced:] / [:voiceless:] by voicing

// Match words with nasal-vowel-stop pattern within distance 1
let grep = PhoneticGrep::from_pattern("[:nasal:][:vowel:][:stop:]", 1)?;
// Matches: "nap", "map", "nod", "mob", "nip", "mat" ...

// Front vowels for palatalization contexts
let grep = PhoneticGrep::from_pattern("c[:front_vowel:]", 1)?;
// Matches: "ce", "ci" (soft c contexts)
```

**Phonetic Flags** (compile-time transformations):
```rust
// Case-insensitive matching
let grep = PhoneticGrep::from_pattern("(?i:hello)", 1)?;
assert!(grep.matches("HELLO").is_some());

// Accent-insensitive matching
let grep = PhoneticGrep::from_pattern("(?a:cafe)", 0)?;
assert!(grep.matches("café").is_some());  // é → e

// Combined case + accent insensitive
let grep = PhoneticGrep::from_pattern("(?ia:cafe)", 1)?;
assert!(grep.matches("CAFÉ").is_some());

// Local distance override: (?;N:pattern)
let grep = PhoneticGrep::from_pattern("(?;2:difficult)", 0)?;
// Pattern-level distance of 2, ignoring the constructor's 0
```

**Formal Verification**: All algorithms are proven correct in Coq/Rocq:

*Phonetic Rules (5 theorems):*
1. **Well-formedness** - All rules satisfy structural constraints
2. **Bounded Expansion** - Output length ≤ input length + 20 (guaranteed)
3. **Non-Confluence** - Rule application order matters (proven)
4. **Termination** - Sequential application always terminates (guaranteed)
5. **Idempotence** - Fixed points remain unchanged (stable semantics)

*Levenshtein Automata:*
- Complete axiom-free proofs for distance properties (triangle inequality, lower bounds)
- Position skipping optimization: 50/50 proofs complete (100% verified)
- Modular proof decomposition with 0 Admitted lemmas

**Rule Sets**:
- `orthography_rules()` - 8 orthographic transformations (ch→ç, ph→f, silent letters)
- `phonetic_rules()` - 3 phonetic approximations for fuzzy matching (th→t, qu↔kw)
- `zompist_rules()` - Complete 13-rule set from Zompist English spelling system

**Dual u8/char Support** (following existing codebase patterns):
```rust
// Byte-level (ASCII, ~5% faster, 4× less memory)
let result = apply_rules_seq(&orthography_rules(), &phones_u8, fuel);

// Character-level (Unicode, correct for accented chars, CJK, emoji)
let result = apply_rules_seq_char(&orthography_rules_char(), &phones_char, fuel);
```

**With Phonetic Rules** (for spelling normalization):
```rust
use liblevenshtein::phonetic::grep::PhoneticGrep;
use std::path::Path;

// Load rules from .llev file and compose with Levenshtein automaton
let grep = PhoneticGrep::with_rules(
    "fone",                    // Query (phonetic spelling)
    Path::new("english.llev"), // Rules: ph→f, gh→silent, etc.
    1                          // Max Levenshtein distance
)?;

// "phone" matches "fone" after rule application + distance tolerance
assert!(grep.matches("phone").is_some());
assert!(grep.matches("fone").is_some());   // exact after normalization
assert!(grep.matches("phones").is_some()); // distance 1
```

**Verification Artifacts**:
- Coq proofs: [`docs/verification/phonetic/`](docs/verification/phonetic/)
- Rust implementation: [`src/phonetic/`](src/phonetic/)
- Property tests: [`src/phonetic/properties.rs`](src/phonetic/properties.rs) (14 tests, 3,584 test cases)
- Benchmarks: [`benches/phonetic_rules.rs`](benches/phonetic_rules.rs) (7 benchmark groups)
- Example: [`examples/phonetic_rewrite.rs`](examples/phonetic_rewrite.rs)

Enable with:
```toml
[dependencies]
liblevenshtein = { git = "https://github.com/universal-automata/liblevenshtein-rust", features = ["phonetic-rules"] }
```

Run the example:
```bash
cargo run --example phonetic_rewrite --features phonetic-rules
```

**Source**: Based on [Zompist's English spelling rules](https://zompist.com/spell.html) with complete formal verification in Coq/Rocq (630+ lines of proofs, 100% proven, zero Admitted).

### LLev/LLRE - Phonetic Pattern Languages

For complex phonetic matching, use `.llev` files for rewrite rules or `.llre` files for regex-style patterns with phonetic extensions:

```rust
use liblevenshtein::phonetic::llev::parse_str;
use liblevenshtein::phonetic::llre;

// LLev: Rewrite rules with phonetic context
let rules = parse_str(r#"
    @name "English Phonetic Rules"

    [id: 1, name: "ph to f"]
    ph -> f;  // phone → fone

    [id: 2, name: "soft c"]
    c -> s / _[:front_vowel:];  // city → sity (before e,i)

    [id: 3, name: "silent gh"]
    gh -> / [:vowel:]_;  // night → nit (after vowel)
"#)?;

// LLRE: Regex patterns with named character classes
let pattern = llre::compile_pattern("[:fricative:]one")?;
assert!(pattern.matches("fone"));    // f ∈ [:fricative:]
assert!(pattern.matches("shone"));   // sh ∈ [:fricative:]
assert!(pattern.matches("zone"));    // z ∈ [:fricative:]

// LLRE with flags
let pattern = llre::compile_pattern("(?ia:[:nasal:][:vowel:][:stop:])")?;
// Case+accent insensitive nasal-vowel-stop pattern

// AOT compilation for instant loading
#[cfg(feature = "serialization")]
{
    llre::save(&pattern, "pattern.llre.bin")?;
    let loaded = llre::load("pattern.llre.bin")?;
}
```

See [`examples/phonetic_spellcheck`](examples/phonetic_spellcheck/) for a complete demo, and the [LLev Grammar](docs/grammar/llev.ebnf) / [LLRE Grammar](docs/grammar/llre.ebnf) for full syntax reference.

## Documentation

- **[Technical Glossary](docs/GLOSSARY.md)** - Comprehensive reference for all technical terms (70+ entries)
- **[User Guide](docs/user-guide/)** - Getting started, algorithms, backends, and features
- **[Developer Guide](docs/developer-guide/)** - Architecture, building, contributing, and performance
- **[Building and Testing](docs/developer-guide/building.md)** - Comprehensive build instructions and CLI usage
- **[Contributing Guidelines](docs/developer-guide/contributing.md)** - How to contribute to the project
- **[Features Overview](docs/user-guide/features.md)** - Detailed feature documentation
- **[Publishing Guide](docs/developer-guide/publishing.md)** - Requirements for publishing to crates.io
- **[Changelog](CHANGELOG.md)** - Version history and release notes

## How Code Maps to Theory

For developers wanting to understand the implementation or extend the algorithms:

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| **Position (i#e)** | [`src/transducer/position.rs:11-35`](src/transducer/position.rs#L11-L35) | Index + error count structure |
| **Subsumption (⊑)** | [`src/transducer/position.rs:231-269`](src/transducer/position.rs#L231-L269) | Position redundancy elimination |
| **Characteristic Vector (χ)** | [`src/transducer/position.rs`](src/transducer/position.rs) | Character occurrence encoding |
| **Elementary Transitions (δ)** | [`src/transducer/transition.rs:119-438`](src/transducer/transition.rs#L119-L438) | Table 4.1, 7.1, 8.1 from paper |
| **State Transitions (Δ)** | [`src/transducer/query.rs`](src/transducer/query.rs) | Parallel traversal implementation |
| **Algorithm Variants** | [`src/transducer/algorithm.rs`](src/transducer/algorithm.rs) | Standard/Transposition/MergeAndSplit |
| **Imitation Method** | [`src/transducer/query.rs:86-188`](src/transducer/query.rs#L86-L188) | Chapter 6 - on-demand state generation |
| **Dictionary Automaton (A^D)** | [`src/dictionary/`](src/dictionary/) | Multiple backend implementations |

**Key Implementation Patterns:**

- **Position Structure**: Tracks (term_index, num_errors, is_special) corresponding to i#e_t or i#e_s in the paper
- **Subsumption Checking**: Ensures minimal state sets by eliminating redundant positions
- **Transition Functions**: Implement the case analysis from Tables 4.1, 7.1, and 8.1
- **Parallel Traversal**: Simultaneously navigate dictionary automaton A^D and Levenshtein automaton LEV_n(W)

See [Implementation Mapping](docs/research/levenshtein-automata/implementation-mapping.md) for detailed code-to-paper correspondence with examples.

## Performance

The library maintains the theoretical guarantees from the foundational paper while adding practical optimizations:

### Theoretical Complexity

- **Construction:** O(|W|) time for automaton construction (Theorem 5.2.1)
- **Query:** O(|D|) time for dictionary traversal where |D| is total edges (Chapter 3)
- **Space:** O(|W|) states for fixed error bound n (Theorem 4.0.32)

**vs Naive Approach:** Computing Levenshtein distance for every dictionary entry requires O(|D| × |W| × |V|) operations. This library achieves **100-1000× speedup** on large dictionaries by avoiding per-word distance calculations through automata-guided search.

### Core Optimizations

Beyond the algorithmic foundations, the implementation includes:

- **Arc Path Sharing**: Eliminated expensive cloning operations during traversal
- **StatePool**: Object pool pattern for state reuse with exceptional performance gains
- **SmallVec Integration**: Stack-allocated vectors reduce heap allocation pressure
- **Lazy Edge Iteration**: 15-50% faster PathMap edge iteration with zero-copy implementation
- **Aggressive Inlining**: Hot path functions inlined for optimal performance

Benchmarks show 3.3x speedup for DAWG operations and 5-18% improvements across filtering/prefix scenarios.

### SIMD Acceleration (optional `simd` feature)

When compiled with the `simd` feature on x86_64 platforms, the library achieves **20-64% performance gains** through:

- **8 SIMD-optimized components** across critical performance paths
- **AVX2/SSE4.1 implementations** with runtime CPU feature detection
- **Data-driven threshold tuning** based on empirical benchmarking
- **Automatic fallback** to scalar implementation when SIMD unavailable

Key optimizations:
- Characteristic vector operations (vectorized bit manipulation)
- Position subsumption checking (parallel state comparisons)
- State operations (vectorized distance computations)
- Dictionary edge lookups (batched queries with adaptive thresholds)
- Distance matrix computation (vectorized row/column operations)

Performance improvements vary by workload:
- Small dictionaries (< 1,000 terms): 20-30% faster
- Medium dictionaries (1,000-10,000 terms): 30-45% faster
- Large dictionaries (> 10,000 terms): 45-64% faster

See `docs/analysis/` for detailed SIMD performance analysis (950+ lines of documentation).

### Unicode Performance

Character-level dictionary variants (`*Char`) for Unicode support:
- **~5% overhead** for UTF-8 decoding during traversal
- **4x memory** for edge labels (4 bytes per `char` vs 1 byte per `u8`)
- **Zero-cost abstraction** via monomorphization (no runtime polymorphism)
- Same query performance characteristics as byte-level variants

When to use:
- ✅ Use `*Char` variants for multi-language dictionaries with non-ASCII Unicode
- ✅ Use byte-level variants (`DoubleArrayTrie`, `PathMapDictionary`) for ASCII/Latin-1 content

### Phonetic Pattern Matching

When using the `phonetic-rules` feature:
- **NFA optimizations**: 2-7× speedup through epsilon elimination, dead state removal, and transition deduplication
- **Position skipping**: Up to 26.6× speedup for pattern matching with skip-ahead optimization
- **Lexer optimizations**: 7-11% speedup via string interning and stack-based operations

## Research & Planned Features

This library includes extensive research documentation on potential enhancements:

### Universal Levenshtein Automata (Planned)

Support for **restricted substitutions** where only specific character pairs can be substituted:

**Use Cases:**
- **Keyboard proximity** - Only adjacent keys allowed (QWERTY/AZERTY/Dvorak layouts)
  - Example: 'a' ↔ 's' ↔ 'd' (adjacent on QWERTY)
- **OCR error modeling** - Visual similarity constraints
  - Example: 'l' ↔ 'I' ↔ '1', 'O' ↔ '0', 'rn' ↔ 'm'
- **Phonetic matching** - Sound-alike rules
  - Example: 'f' ↔ 'ph', 'k' ↔ 'c', 's' ↔ 'z'

**Theory:** Based on "Universal Levenshtein Automata. Building and Properties" (Mitankin, Mihov, Schulz, 2005). Maintains O(|W|) construction complexity with restricted substitution set S ⊆ Σ × Σ.

**Status:** Research complete, implementation planned. See [Universal Levenshtein Automata Documentation](docs/research/universal-levenshtein/) for complete details.

### Weighted Operations (Research Phase)

Support for variable operation costs through discretization:

**Use Cases:**
- **Keyboard distance costs** - Closer keys cost less (adjacent: 0.5, same row: 1.0, different rows: 1.5)
- **Context-dependent costs** - Position-aware error penalties (beginning/end of word)
- **Frequency-based costs** - Common errors cost less than rare ones

**Complexity:** O(|W| × max_cost/precision) through cost discretization—still linear in query length when cost range and precision are fixed.

**Status:** Methodology documented, implementation research phase. See [Weighted Levenshtein Automata Research](docs/research/weighted-levenshtein-automata/) for feasibility analysis and implementation approach.

### Contributing Research

Interested in implementing these features or researching new extensions? See:
- [Contributing Guidelines](docs/developer-guide/contributing.md)
- [Research Documentation Index](docs/research/)
- [Implementation Mapping](docs/research/levenshtein-automata/implementation-mapping.md)

The research documentation provides complete theoretical foundations and implementation roadmaps for extending the library.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## References

- [Original C++ implementation](https://github.com/universal-automata/liblevenshtein-cpp)
- [PathMap backend](https://github.com/Adam-Vandervorst/PathMap)
- [GitHub Repository](https://github.com/universal-automata/liblevenshtein-rust)
- [Release Page](https://github.com/universal-automata/liblevenshtein-rust/releases)
