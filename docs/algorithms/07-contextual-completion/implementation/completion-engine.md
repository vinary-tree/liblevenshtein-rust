# Completion Engine Implementation

**Deep dive into DynamicContextualCompletionEngine and StaticContextualCompletionEngine architectures, APIs, and implementation details.**

---

[← Back to Layer 7](../README.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Engine Types](#engine-types)
3. [Architecture](#architecture)
4. [DynamicContextualCompletionEngine API](#dynamiccontextualcompletionengine-api)
5. [StaticContextualCompletionEngine API](#staticcontextualcompletionengine-api)
6. [Backend Selection Guide](#backend-selection-guide)
7. [Thread Safety & Concurrency](#thread-safety--concurrency)
8. [Usage Examples](#usage-examples)
9. [Performance Characteristics](#performance-characteristics)
10. [Implementation Notes](#implementation-notes)

---

## Overview

The Contextual Completion Engine comes in two variants, each optimized for different use cases:

| Engine Type | Mutability | Best For | Performance |
|-------------|------------|----------|-------------|
| **DynamicContextualCompletionEngine** | Mutable dictionary | REPL, prototyping, dynamic vocabularies | Good (PathMap baseline) to Excellent (DynamicDawg 2.8×) |
| **StaticContextualCompletionEngine** | Immutable dictionary | LSP servers, pre-built libraries | Excellent (DoubleArrayTrie 12×) |

Both engines share the same core concepts:
- **Hierarchical contexts** for lexical scope visibility
- **Draft buffers** for character-level editing
- **Checkpoint stacks** for undo/redo
- **Query fusion** (finalized terms + drafts)

The key difference is how finalized terms are stored:
- **Dynamic**: Inserts directly into mutable dictionary via `insert_with_value()`
- **Static**: Stores in separate `HashMap<String, Vec<ContextId>>` (dictionary is read-only)

---

## Engine Types

### DynamicContextualCompletionEngine

```rust
pub struct DynamicContextualCompletionEngine<D = PathMapDictionary<Vec<ContextId>>>
where
    D: MutableMappedDictionary<Value = Vec<ContextId>> + Clone,
{
    drafts: Arc<Mutex<HashMap<ContextId, DraftBuffer>>>,
    checkpoints: Arc<Mutex<HashMap<ContextId, CheckpointStack>>>,
    context_tree: Arc<RwLock<ContextTree>>,
    transducer: Arc<RwLock<Transducer<D>>>,
}
```

**Supported Backends**:
- `PathMapDictionary<Vec<ContextId>>` (default)
- `PathMapDictionaryChar<Vec<ContextId>>` (Unicode)
- `DynamicDawg<Vec<ContextId>>` (~2.8× faster)
- `DynamicDawgChar<Vec<ContextId>>` (Unicode + fast)

**Key Feature**: `finalize()` directly mutates the dictionary.

### StaticContextualCompletionEngine

```rust
pub struct StaticContextualCompletionEngine<D = DoubleArrayTrie<Vec<ContextId>>>
where
    D: MappedDictionary<Value = Vec<ContextId>> + Clone,
{
    drafts: Arc<Mutex<HashMap<ContextId, DraftBuffer>>>,
    checkpoints: Arc<Mutex<HashMap<ContextId, CheckpointStack>>>,
    context_tree: Arc<RwLock<ContextTree>>,
    transducer: Arc<RwLock<Transducer<D>>>,
    finalized_terms: Arc<RwLock<HashMap<String, Vec<ContextId>>>>,  // ← NEW!
}
```

**Supported Backends**:
- `DoubleArrayTrie<Vec<ContextId>>` (default, ~12× faster)
- `DoubleArrayTrieChar<Vec<ContextId>>` (Unicode + fastest)

**Key Feature**: `finalize()` stores in separate HashMap (dictionary untouched).

---

## Architecture

### Component Interactions

```
┌─────────────────────────────────────────────────────────┐
│         Contextual Completion Engine                    │
│                                                         │
│  User Operations:                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ insert_char() / rollback_char()                 │  │
│  │        ↓                                        │  │
│  │   DraftBuffer (Vec<char>)                       │  │
│  │        ↓                                        │  │
│  │   checkpoint() → CheckpointStack                │  │
│  │        ↓                                        │  │
│  │   finalize() → Dictionary OR HashMap            │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Query Operations:                                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │ complete(query, max_distance)                   │  │
│  │        ↓                                        │  │
│  │   ContextTree.visible_contexts(context)         │  │
│  │        ↓                                        │  │
│  │   Transducer.query_with_distance()              │  │
│  │        ↓                                        │  │
│  │   Filter by visible contexts                    │  │
│  │        ↓                                        │  │
│  │   Merge with draft results                      │  │
│  │        ↓                                        │  │
│  │   Sort and return                               │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### State Management

**Drafts**: Stored in `HashMap<ContextId, DraftBuffer>` protected by `Mutex`.

- Each context has its own `DraftBuffer`
- Character-level operations (`insert`, `delete`)
- Cleared on `finalize()` or `discard()`

**Checkpoints**: Stored in `HashMap<ContextId, CheckpointStack>` protected by `Mutex`.

- Each context has its own stack of checkpoints
- Checkpoint saves current buffer position
- `undo()` pops and restores to previous checkpoint
- Cleared on `finalize()` or explicit `clear_checkpoints()`

**Context Tree**: Single tree protected by `RwLock`.

- Hierarchical parent-child relationships
- Visibility computed by traversing parent chain
- Read-heavy workload (queries frequent, modifications rare)

**Dictionary/Transducer**: Protected by `RwLock`.

- Read-heavy for queries
- Write lock only for `finalize()` (dynamic engine) or never (static engine)

---

## DynamicContextualCompletionEngine API

### Constructors

#### Default Backend (PathMapDictionary)

```rust
// Simplest: default algorithm
let engine = DynamicContextualCompletionEngine::new();

// Custom algorithm
use liblevenshtein::transducer::Algorithm;
let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Transposition);
```

#### DynamicDawg Backend (~2.8× faster queries)

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

// Byte-level (ASCII/Latin-1)
let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);

// Character-level (full Unicode)
let engine = DynamicContextualCompletionEngine::with_dynamic_dawg_char(Algorithm::Standard);
```

#### Custom Backend

```rust
use libdictenstein::pathmap::PathMapDictionary;

let dict = PathMapDictionary::new();
let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
```

#### Pre-Built Dictionary Injection

**Use Case**: Inject pre-built dictionaries from parallel workspace indexing, bulk loading, or external sources.

```rust
use liblevenshtein::contextual::{DynamicContextualCompletionEngine, ContextId};
use libdictenstein::dynamic_dawg::DynamicDawg;
use liblevenshtein::transducer::Algorithm;
use rayon::prelude::*;

// Example 1: Simple pre-built dictionary
let dict = DynamicDawg::new();
dict.insert_with_value("function", vec![0u32]);
dict.insert_with_value("variable", vec![0u32]);

let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
// Engine ready with pre-populated dictionary
```

**Example 2: Parallel Workspace Construction**

```rust
// Build dictionaries in parallel (one per document)
let documents = vec![
    (1u32, "fn main() { let x = 10; }"),
    (2u32, "fn helper() { let y = 20; }"),
    (3u32, "struct Foo { field: i32 }"),
];

let dicts: Vec<DynamicDawg<Vec<ContextId>>> = documents
    .par_iter()
    .map(|(ctx_id, source)| {
        let terms = extract_identifiers(source);  // Your tokenizer
        let dict = DynamicDawg::new();
        for term in terms {
            dict.insert_with_value(term, vec![*ctx_id]);
        }
        dict
    })
    .collect();

// Merge using binary tree reduction (~150× faster than sequential)
fn merge_deduplicated(left: &Vec<u32>, right: &Vec<u32>) -> Vec<u32> {
    let mut merged = left.clone();
    merged.extend(right);
    merged.sort_unstable();
    merged.dedup();
    merged
}

let merged = merge_tree_parallel(dicts, merge_deduplicated);

// Inject merged dictionary into engine
let engine = DynamicContextualCompletionEngine::with_dictionary(merged, Algorithm::Standard);

// Query with context filtering
let results = engine.complete(1, "ma", 1);
// Returns results visible to context 1 (document 1 + ancestors)
```

**Example 3: Bulk Load from External Source**

```rust
// Load pre-existing terms (e.g., from database, file, cache)
let stdlib_terms = vec![
    ("std::vec::Vec", vec![0]),
    ("std::io::println", vec![0]),
    ("std::collections::HashMap", vec![0]),
];

let dict = DynamicDawg::from_iter(
    stdlib_terms.into_iter().map(|(term, contexts)| (term, contexts))
);

let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
```

**Value Type Requirements**:

The dictionary **must** use `Vec<ContextId>` (i.e., `Vec<u32>`) as the value type:

```rust
// ✓ Correct
let dict: DynamicDawg<Vec<ContextId>> = DynamicDawg::new();
let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);

// ✗ Wrong - type mismatch
let dict: DynamicDawg<String> = DynamicDawg::new();
let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
//                                                              ^^^^ expected Vec<u32>, found String
```

**Thread Safety**:

- `with_dictionary()` takes ownership of the dictionary
- Dictionary is wrapped in `Arc<RwLock<...>>` internally
- Safe to clone engine and share across threads
- Parallel construction is safe (no shared state between builder instances)

**Performance Notes**:

| Pattern | Time (100 docs, 1K terms/doc) | Notes |
|---------|-------------------------------|-------|
| Sequential merge | ~50s | `$\mathcal{O}(N^{2}\cdot n\cdot m)$` - avoid! |
| Binary tree parallel merge | ~0.3s | `$\mathcal{O}(n\cdot m\cdot \log  N)$` with parallelism |
| Direct injection | <1ms | Zero overhead constructor |

**→ See**: [Parallel Workspace Indexing Pattern](../patterns/parallel-workspace-indexing.md) for complete production-ready implementation with benchmarks and best practices.

### Context Management

```rust
// Create root context (e.g., global scope)
let global: ContextId = 0;
engine.create_root_context(global);

// Create child context (e.g., function scope inside global)
let func: ContextId = 10;
engine.create_child_context(func, global)?;

// Check if context exists
if engine.context_exists(func) {
    // ...
}

// Get visible contexts (self + ancestors)
let visible = engine.get_visible_contexts(func);
// Returns: [10, 0] (func + global)

// Remove context (and all descendants)
engine.remove_context(func);
```

### Draft Operations

```rust
let ctx = engine.create_root_context(0);

// Insert single character
engine.insert_char(ctx, 'h')?;
engine.insert_char(ctx, 'i')?;

// Insert string (calls insert_char() for each character)
engine.insert_str(ctx, "hello")?;

// Get current draft
let draft = engine.get_draft(ctx); // Some("hello")

// Check if draft exists
if engine.has_draft(ctx) {
    // ...
}

// Delete last character (backspace)
let deleted = engine.delete_char(ctx); // Some('o')
// Draft is now "hell"

// Clear entire draft
engine.clear_draft(ctx)?;
```

### Checkpoint Operations

```rust
let ctx = engine.create_root_context(0);

// Type some text
engine.insert_str(ctx, "hello")?;

// Save checkpoint
engine.checkpoint(ctx)?;

// Continue typing
engine.insert_str(ctx, " world")?;

// Save another checkpoint
engine.checkpoint(ctx)?;

// Type more
engine.insert_char(ctx, '!')?;
assert_eq!(engine.get_draft(ctx), Some("hello world!".to_string()));

// Undo to previous checkpoint
engine.undo(ctx)?;
assert_eq!(engine.get_draft(ctx), Some("hello world".to_string()));

// Undo again
engine.undo(ctx)?;
assert_eq!(engine.get_draft(ctx), Some("hello".to_string()));

// Check number of checkpoints
let count = engine.checkpoint_count(ctx); // 0 (all consumed by undo)

// Clear all checkpoints
engine.clear_checkpoints(ctx)?;
```

### Finalization

```rust
let ctx = engine.create_root_context(0);

// Method 1: Finalize current draft
engine.insert_str(ctx, "function")?;
let term = engine.finalize(ctx)?;  // Returns "function"
assert!(!engine.has_draft(ctx));   // Draft cleared

// Method 2: Finalize directly (bypass draft)
engine.finalize_direct(ctx, "variable")?;

// Query if term exists
assert!(engine.has_term("function"));
assert!(engine.has_term("variable"));

// Get contexts where term is defined
let contexts = engine.term_contexts("function");
assert_eq!(contexts, vec![ctx]);
```

### Completion Queries

```rust
let ctx = engine.create_root_context(0);

// Add finalized terms
engine.finalize_direct(ctx, "hello")?;
engine.finalize_direct(ctx, "help")?;

// Add draft
engine.insert_str(ctx, "hero")?;

// Query with max distance 2
let results = engine.complete(ctx, "hel", 2);
// Returns:
// - "hel" (draft, distance 0)
// - "hello" (finalized, distance 2)
// - "help" (finalized, distance 1)
// - "hero" (draft, distance 2)

// Query only drafts
let draft_results = engine.complete_drafts(ctx, "her", 1);
// Returns only "hero" (distance 1)

// Query only finalized terms
let finalized_results = engine.complete_finalized(ctx, "hel", 1);
// Returns "help" (distance 1), excludes "hello" (distance 2)
```

### Discard Draft

```rust
let ctx = engine.create_root_context(0);

// Type something
engine.insert_str(ctx, "mistake")?;
engine.checkpoint(ctx)?;

// Discard without finalizing
engine.discard(ctx)?;

// Draft and checkpoints cleared
assert!(!engine.has_draft(ctx));
assert_eq!(engine.checkpoint_count(ctx), 0);
```

---

## StaticContextualCompletionEngine API

### Constructors

```rust
use liblevenshtein::contextual::StaticContextualCompletionEngine;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::transducer::Algorithm;

// Build static dictionary
let mut builder = DoubleArrayTrie::builder();
builder.insert_with_value("std::vec::Vec", Some(vec![0]));
builder.insert_with_value("std::io::println", Some(vec![0]));
let dict = builder.build();

// Create engine with DoubleArrayTrie
let engine = StaticContextualCompletionEngine::with_double_array_trie(
    dict,
    Algorithm::Standard
);

// Or with Unicode support
let dict_char = DoubleArrayTrieChar::builder()
    .insert_with_value("世界", Some(vec![0]))
    .build();
let engine_char = StaticContextualCompletionEngine::with_double_array_trie_char(
    dict_char,
    Algorithm::Standard
);
```

### API Parity

**Note**: Most APIs are **identical** to `DynamicContextualCompletionEngine`:

- Context management: `create_root_context()`, `create_child_context()`, etc.
- Draft operations: `insert_char()`, `delete_char()`, `clear_draft()`, etc.
- Checkpoint operations: `checkpoint()`, `undo()`, `clear_checkpoints()`, etc.
- Query operations: `complete()`, `complete_drafts()`, etc.

### Finalization Differences

**Key Difference**: `finalize()` stores in `finalized_terms` HashMap, NOT in dictionary.

```rust
let ctx = engine.create_root_context(0);

// Draft finalization
engine.insert_str(ctx, "my_function")?;
engine.finalize(ctx)?;
// ✓ Stored in finalized_terms HashMap
// ✗ NOT added to DoubleArrayTrie (immutable)

// Direct finalization
engine.finalize_direct(ctx, "my_variable")?;
// ✓ Also stored in finalized_terms
```

### Query Fusion (3 Sources)

**Static engine queries 3 separate sources**:

```rust
let results = engine.complete(ctx, "std", 1);

// Queries:
// 1. Static dictionary (fast DoubleArrayTrie)
// 2. finalized_terms HashMap (small, user-defined)
// 3. Draft buffers (always fresh)

// All results merged, deduplicated, and sorted
```

**Performance**: Dictionary queries remain extremely fast (~12-16µs), finalized_terms adds negligible overhead (<1µs for typical hash lookups).

---

## Backend Selection Guide

### Decision Matrix

| Criteria | Recommended Backend | Engine Type |
|----------|---------------------|-------------|
| **Need runtime insertion?** | Yes → DynamicDawg / PathMap | Dynamic |
| **Pre-built vocabulary?** | Yes → DoubleArrayTrie | Static |
| **Unicode required?** | Yes → *Char variants | Either |
| **Maximum performance?** | DoubleArrayTrie | Static |
| **Prototyping/flexibility?** | PathMapDictionary | Dynamic |
| **LSP server (large stdlib)?** | DoubleArrayTrie | Static |
| **REPL (dynamic symbols)?** | DynamicDawg | Dynamic |

### Performance Comparison

**Query @ Distance 1 (500 terms)**:

| Backend | Engine | Time (µs) | vs PathMap |
|---------|--------|-----------|------------|
| PathMapDictionary | Dynamic | 11.5 | 1.0× |
| DynamicDawg | Dynamic | 4.1 | 2.8× |
| DoubleArrayTrie | Static | 0.96 | 12.0× |

**Finalization**:

| Backend | Engine | Time (µs) | Notes |
|---------|--------|-----------|-------|
| PathMapDictionary | Dynamic | ~8 | Mutation + rebuild |
| DynamicDawg | Dynamic | ~6 | Incremental insert |
| DoubleArrayTrie | Static | ~2 | HashMap insert only |

### Memory Overhead

| Backend | Per-Term Overhead | Notes |
|---------|-------------------|-------|
| PathMapDictionary | ~48 bytes | Arc pointers + PathMap nodes |
| DynamicDawg | ~24 bytes | Compact node representation |
| DoubleArrayTrie | ~16 bytes | Densest structure |
| finalized_terms (Static) | ~56 bytes | HashMap entry + String + Vec |

**Static Engine Note**: Most terms in static dictionary (minimal overhead), only user-defined terms in `finalized_terms` HashMap.

---

## Thread Safety & Concurrency

### Lock Granularity

| Component | Lock Type | Contention | Access Pattern |
|-----------|-----------|------------|----------------|
| **Drafts** | `Mutex` | Low | Short critical sections (char ops) |
| **Checkpoints** | `Mutex` | Low | Infrequent (user-initiated undo) |
| **ContextTree** | `RwLock` | Very Low | Read-heavy (many queries, few mutations) |
| **Transducer** | `RwLock` | Very Low (Dynamic) / None (Static) | Reads during queries, writes only on finalize (Dynamic) |

### Concurrent Access Patterns

**Safe Patterns**:

```rust
// Multiple threads querying different contexts
let engine = Arc::new(engine);
let handles: Vec<_> = (0..10)
    .map(|i| {
        let engine = Arc::clone(&engine);
        thread::spawn(move || {
            engine.complete(i as ContextId, "query", 1)
        })
    })
    .collect();
```

```rust
// Concurrent insertions to different contexts
let engine = Arc::new(engine);
thread::spawn({
    let engine = Arc::clone(&engine);
    move || engine.insert_str(1, "hello")
});
thread::spawn({
    let engine = Arc::clone(&engine);
    move || engine.insert_str(2, "world")
});
```

**Potential Contention**:

- **Same context, high frequency**: Multiple threads rapidly inserting to same context
  - Mitigation: Partition work by context ID
- **Frequent finalization (Dynamic)**: Write locks on dictionary
  - Mitigation: Batch finalizations or use Static engine

**Lock-Free Future**:

Future optimization could use `DashMap` for drafts/checkpoints to eliminate locks for disjoint contexts.

---

## Usage Examples

### Example 1: LSP Server with Static Engine

```rust
use liblevenshtein::contextual::StaticContextualCompletionEngine;
use libdictenstein::double_array_trie::DoubleArrayTrie;

// Pre-build standard library dictionary
let mut builder = DoubleArrayTrie::builder();
for symbol in load_stdlib_symbols() {
    builder.insert_with_value(&symbol, Some(vec![0]));
}
let dict = builder.build();

// Create static engine
let engine = StaticContextualCompletionEngine::with_double_array_trie(
    dict,
    Algorithm::Standard
);

// Track file/scope contexts
let file_ctx = 1;
let class_ctx = 10;
let method_ctx = 100;

engine.create_root_context(file_ctx);
engine.create_child_context(class_ctx, file_ctx)?;
engine.create_child_context(method_ctx, class_ctx)?;

// User types "std::"
engine.insert_str(method_ctx, "std::")?;

// LSP completion request
let results = engine.complete(method_ctx, "std::", 0);
// Fast query of pre-built dictionary (no finalization needed)
```

### Example 2: REPL with Dynamic Engine

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);

let session_ctx = 0;
engine.create_root_context(session_ctx);

// User defines variable
engine.insert_str(session_ctx, "my_var")?;
engine.finalize(session_ctx)?;

// User defines function
engine.insert_str(session_ctx, "my_function")?;
engine.finalize(session_ctx)?;

// Later, user types "my"
let results = engine.complete(session_ctx, "my", 0);
// Returns both "my_var" and "my_function" (exact matches)
```

### Example 3: Multi-Context Isolation

```rust
let engine = DynamicContextualCompletionEngine::new();

// File 1
let file1_ctx = 1;
engine.create_root_context(file1_ctx);
engine.finalize_direct(file1_ctx, "file1_var")?;

// File 2
let file2_ctx = 2;
engine.create_root_context(file2_ctx);
engine.finalize_direct(file2_ctx, "file2_var")?;

// Query from file1 - only sees file1_var
let results1 = engine.complete(file1_ctx, "file", 1);
assert!(results1.iter().any(|c| c.term == "file1_var"));
assert!(!results1.iter().any(|c| c.term == "file2_var"));

// Query from file2 - only sees file2_var
let results2 = engine.complete(file2_ctx, "file", 1);
assert!(results2.iter().any(|c| c.term == "file2_var"));
assert!(!results2.iter().any(|c| c.term == "file1_var"));
```

### Example 4: Hierarchical Scopes with Visibility

```rust
let engine = DynamicContextualCompletionEngine::new();

// global → module → class → method
let global = 0;
let module = 1;
let class = 10;
let method = 100;

engine.create_root_context(global);
engine.create_child_context(module, global)?;
engine.create_child_context(class, module)?;
engine.create_child_context(method, class)?;

// Define at each level
engine.finalize_direct(global, "global_var")?;
engine.finalize_direct(module, "module_var")?;
engine.finalize_direct(class, "class_var")?;
engine.finalize_direct(method, "method_var")?;

// Query from method - sees all ancestors
let results = engine.complete(method, "", 0);
assert_eq!(results.len(), 4); // All four visible
assert!(results.iter().any(|c| c.term == "global_var"));
assert!(results.iter().any(|c| c.term == "module_var"));
assert!(results.iter().any(|c| c.term == "class_var"));
assert!(results.iter().any(|c| c.term == "method_var"));

// Query from module - only sees module + global
let results_module = engine.complete(module, "", 0);
assert_eq!(results_module.len(), 2);
assert!(results_module.iter().any(|c| c.term == "global_var"));
assert!(results_module.iter().any(|c| c.term == "module_var"));
```

### Example 5: Checkpoint-Based Undo for Editor

```rust
let engine = DynamicContextualCompletionEngine::new();
let ctx = engine.create_root_context(0);

// User types word
engine.insert_str(ctx, "function")?;

// Save checkpoint after word
engine.checkpoint(ctx)?;

// User adds space and starts new word
engine.insert_char(ctx, ' ')?;
engine.insert_str(ctx, "foo")?;

// Save checkpoint
engine.checkpoint(ctx)?;

// User makes typo
engine.insert_str(ctx, "xqz")?;

// User hits Ctrl+Z (undo)
engine.undo(ctx)?;
assert_eq!(engine.get_draft(ctx), Some("function foo".to_string()));

// User hits Ctrl+Z again (undo)
engine.undo(ctx)?;
assert_eq!(engine.get_draft(ctx), Some("function".to_string()));
```

---

## Performance Characteristics

### Operation Latencies

| Operation | Time (µs) | Throughput | Notes |
|-----------|-----------|------------|-------|
| `insert_char()` | ~4 | 12M chars/sec | Append to Vec<char> |
| `delete_char()` | ~3 | 333K ops/sec | Pop from Vec<char> |
| `checkpoint()` | ~0.116 | 8.6M checkpoints/sec | Copy buffer length |
| `undo()` | ~0.2 | 5M restores/sec | Truncate Vec<char> |
| `finalize()` (Dynamic) | ~8 | 125K finalizations/sec | Dictionary insert |
| `finalize()` (Static) | ~2 | 500K finalizations/sec | HashMap insert |
| `complete()` (dist=1) | ~11.5 | 87K queries/sec | PathMap baseline |
| `complete()` (dist=1) | ~4.1 | 244K queries/sec | DynamicDawg |
| `complete()` (dist=1) | ~0.96 | 1.04M queries/sec | DoubleArrayTrie |

**Key Observations**:
- Character operations sub-10µs (imperceptible latency)
- Checkpoints extremely cheap (~116ns)
- Query performance dominated by dictionary backend choice
- Static engine provides 12× speedup over PathMap baseline

### Memory Usage

**Per-Context Overhead**:

- DraftBuffer: ~48 bytes + (4 bytes × chars)
- CheckpointStack: ~24 bytes + (8 bytes × checkpoints)
- ContextTree node: ~16 bytes
- **Total**: ~88 bytes base + data

**Scaling**:
- 100 contexts: ~8.8 KB base
- 1,000 contexts: ~88 KB base
- 10,000 contexts: ~880 KB base

**Conclusion**: Memory overhead negligible for realistic LSP server workloads.

---

## Accessor Methods

Both engine types provide accessor methods to introspect internal state and access underlying components.

### Accessing Underlying Components

#### transducer() - Get Transducer Reference

Access the underlying `Transducer` that powers fuzzy matching.

**Signature**:
```rust
pub fn transducer(&self) -> &Arc<RwLock<Transducer<D>>>
```

**Returns**: Shared reference to the transducer (wrapped in Arc<RwLock> for thread safety)

**Use Cases**:
- Access the dictionary via `transducer.read().dictionary()`
- Query algorithm settings: `transducer.read().algorithm()`
- Perform low-level transducer operations
- Clone the dictionary for serialization or external processing

**Example**:
```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();

// Access the transducer
let transducer_ref = engine.transducer();

// Read lock to access transducer methods
{
    let transducer = transducer_ref.read();

    // Get the algorithm
    let algo = transducer.algorithm();
    println!("Using algorithm: {:?}", algo);

    // Access the dictionary
    let dict = transducer.dictionary();
    println!("Dictionary size: {:?}", dict.len());
}

// Clone the dictionary for external use
let dict_clone = {
    let transducer = transducer_ref.read();
    transducer.dictionary().clone()
};

// Now you can work with the dictionary independently
assert_eq!(dict_clone.len(), Some(0));
```

**Dictionary Access Pattern**:
```rust
// Pattern 1: Read-only dictionary access
let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);
engine.finalize_direct(0, "hello")?;

{
    let transducer = engine.transducer().read();
    let dict = transducer.dictionary();
    assert!(dict.contains("hello"));
    assert_eq!(dict.term_count(), 1);
}

// Pattern 2: Extract dictionary for maintenance
let dict_clone = engine.transducer().read().dictionary().clone();

// Perform compaction if needed (for DynamicDawg)
if dict_clone.needs_compaction() {
    let removed = dict_clone.compact();
    println!("Compaction freed {} nodes", removed);
}
```

**Thread Safety**:
- `Arc<RwLock<...>>` allows multiple concurrent readers OR one writer
- Read lock: `engine.transducer().read()`
- Write lock (rare): `engine.transducer().write()`
- Lock held only during block scope - release promptly

**Lock Contention Example**:
```rust
use std::thread;
use std::sync::Arc;

let engine = Arc::new(DynamicContextualCompletionEngine::new());

// Concurrent transducer/dictionary access (safe)
let handles: Vec<_> = (0..10)
    .map(|_| {
        let e = Arc::clone(&engine);
        thread::spawn(move || {
            let transducer = e.transducer().read();
            transducer.dictionary().len()
        })
    })
    .collect();

for h in handles {
    h.join().unwrap();
}
```

---

###  Context State Accessors

**Already documented** in [Context Management](#context-management) section:
- `context_exists(id)` - Check if context exists
- `get_visible_contexts(id)` - Get context + ancestor chain

**Already documented** in [Draft Operations](#draft-operations) section:
- `get_draft(id)` - Retrieve current draft text
- `has_draft(id)` - Check if draft exists

**Already documented** in [Checkpoint Operations](#checkpoint-operations) section:
- `checkpoint_count(id)` - Number of undo points available

**Already documented** in [Finalization](#finalization) section:
- `has_term(term)` - Check if term finalized
- `term_contexts(term)` - Get contexts defining term

---

### Performance Characteristics

**Accessor Latencies**:

| Method | Latency | Notes |
|--------|---------|-------|
| `transducer()` | ~3ns | Returns Arc reference |
| `transducer().read()` | ~50ns | Acquires read lock |
| `.dictionary()` | ~2ns | Field access |
| `.algorithm()` | ~1ns | Copy enum |
| `context_exists()` | ~100ns | HashMap lookup |
| `get_visible_contexts()` | ~500ns | Tree traversal (depth-dependent) |
| `has_draft()` | ~100ns | HashMap lookup |
| `checkpoint_count()` | ~100ns | Vec len() |

**Best Practices**:
- Minimize lock hold duration - acquire lock, read data, release
- Clone data if you need to work with it outside the lock scope
- Use `{}` blocks to explicitly control lock lifetime

```rust
// ✓ Good: Lock released immediately
let dict_len = {
    let transducer = engine.transducer().read();
    transducer.dictionary().len()
}; // Lock released here

// ✗ Bad: Lock held for entire scope
let transducer = engine.transducer().read();
// ... lots of code ...
let dict_len = transducer.dictionary().len();
// Lock still held!
```

---

## Implementation Notes

### Why Two Engine Types?

**Design Rationale**:

1. **Static engine cannot use MutableMappedDictionary** - DoubleArrayTrie doesn't support insertion
2. **Query fusion architecture** - Static engine needs separate storage for user-defined terms
3. **Zero-cost abstraction** - Generic over backend, monomorphization eliminates dynamic dispatch

**Alternative Considered**: Single engine with `Option<HashMap>` for finalized terms. **Rejected** because:
- Always pays HashMap cost even when not needed (dynamic engine)
- Confusing API (when to use HashMap vs dictionary?)
- Type system better expresses the distinction

### Error Handling

**Common Errors**:

```rust
pub enum ContextError {
    ContextNotFound(ContextId),      // Context doesn't exist
    NoDraftBuffer(ContextId),        // No draft buffer (internal error)
    NoCheckpointStack(ContextId),    // No checkpoint stack (internal error)
    NoCheckpoints(ContextId),        // Cannot undo - no checkpoints available
    EmptyDraft(ContextId),           // Cannot finalize empty draft
    EmptyTerm,                       // Cannot finalize_direct with empty string
    ParentNotFound(ContextId),       // Parent context doesn't exist
    ContextAlreadyExists(ContextId), // Context ID already in use
}
```

**Error Recovery**:

- `ContextNotFound`: Check `context_exists()` before operations
- `NoCheckpoints`: Check `checkpoint_count()` before undo
- `EmptyDraft`: Check `has_draft()` before finalize

### Future Optimizations

**Planned Improvements**:

1. **DashMap for drafts/checkpoints**: Eliminate lock contention for disjoint contexts
2. **Per-context locks**: Fine-grained locking instead of global Mutex
3. **Async API**: Non-blocking operations for high-concurrency servers
4. **Snapshot isolation**: MVCC for concurrent reads during finalization

---

[← Back to Layer 7](../README.md)
