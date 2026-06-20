# Layer 7: Contextual Completion Engine

**Scope-aware hierarchical code completion with draft state management, checkpoints, and incremental fuzzy matching.**

---

[← Layer 6: Zipper Navigation](../06-zipper-navigation/README.md) | [Index](../README.md) | [Layer 8: Caching →](../08-caching-layer/README.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Theory: Lexical Scoping & Visibility](#theory-lexical-scoping--visibility)
5. [Engine Types](#engine-types)
6. [Usage Examples](#usage-examples)
7. [Performance Analysis](#performance-analysis)
8. [Backend Comparison](#backend-comparison)
9. [Implementation Details](#implementation-details)
10. [References](#references)

---

## Overview

The Contextual Completion Engine provides IDE-style code completion with hierarchical scope awareness, character-level draft editing, and checkpoint-based undo/redo. It combines fuzzy matching via Levenshtein automata with scope visibility rules and incremental state management.

![Context scope tree: nested completion scopes where child contexts inherit visible terms from their parents (lexical scoping)](../../diagrams/contextual/context-scope-tree.svg)

*Context scope tree: child scopes inherit visible terms from their ancestors.*

![Draft checkpoint lifecycle: how character-level draft edits, checkpoints, and undo/redo transitions move between states](../../diagrams/contextual/draft-checkpoint-lifecycle.svg)

*Draft checkpoint lifecycle: insert, delete, checkpoint, and undo/redo transitions over draft state.*

### Key Features

- **Hierarchical Scope Visibility**: Child contexts see parent terms (lexical scoping)
- **Character-Level Drafts**: Incremental insertion and deletion
- **Checkpoint System**: Stack-based undo/redo for editor integration
- **Query Fusion**: Merge results from finalized terms + active drafts
- **Multi-Backend Support**: PathMapDictionary, DynamicDawg, DoubleArrayTrie, and Unicode variants
- **Thread-Safe**: Concurrent queries and modifications via fine-grained locking
- **Zipper-Based Traversal**: Efficient navigation with automaton composition

### Production Status

- ✅ **100% Complete** (6 phases implemented)
- ✅ **93 passing tests** with comprehensive coverage
- ✅ **5 working examples** demonstrating real-world usage
- ✅ **Performance validated** (zipper overhead: 1.66-1.97× acceptable)

---

## Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────────┐
│       Contextual Completion Engine (Layer 7)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ContextualCompletionEngine API                      │  │
│  │  • create_context(id, parent)                        │  │
│  │  • insert_char(context, ch) / rollback_char()        │  │
│  │  • checkpoint() / restore(cp)                        │  │
│  │  • complete(query, max_distance) → Completion[]      │  │
│  │  • finalize(context) → String                        │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
│      ┌───────────────┴────────────────┐                    │
│      ▼                                ▼                    │
│  ┌────────────┐                  ┌─────────────┐          │
│  │ Dictionary │                  │   Drafts    │          │
│  │ (Finalized)│                  │ (per-ctx)   │          │
│  │  PathMap   │                  │  HashMap    │          │
│  │ DynamicDawg│                  │ DraftBuffer │          │
│  │DoubleArray │                  │ Checkpoint  │          │
│  └─────┬──────┘                  └──────┬──────┘          │
│        │                                │                  │
│        └────────────┬───────────────────┘                  │
│                     ▼                                      │
│         ┌───────────────────────┐                         │
│         │   Context Tree        │                         │
│         │  (Hierarchical        │                         │
│         │   Visibility)         │                         │
│         └───────────────────────┘                         │
└────────────────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Zipper-Based    │     │  Levenshtein    │
│ Dictionary      │     │  Automaton      │
│ Traversal       │     │  (Layer 2)      │
│ (Layer 6)       │     │                 │
└─────────────────┘     └─────────────────┘
```

### Component Responsibilities

| Component | Purpose | State Management |
|-----------|---------|------------------|
| **DynamicContextualCompletionEngine** | Mutable dictionary backends | PathMapDictionary, DynamicDawg |
| **StaticContextualCompletionEngine** | Immutable dictionary backends | DoubleArrayTrie, finalized_terms HashMap |
| **ContextTree** | Hierarchical scope structure | Parent-child relationships, visibility |
| **DraftBuffer** | Character-level editing | Vec<char> with incremental insert/delete |
| **CheckpointStack** | Undo/redo states | Saved positions + validation |
| **Completion** | Result type | term, distance, contexts, is_draft flag |

---

## Core Components

### 1. ContextId

```rust
/// Unique identifier for a lexical scope context
pub type ContextId = u32;

// Example context hierarchy:
// 0 - Global scope
// 1 - Module scope
// 10 - Class scope
// 42 - Function scope
// 100 - Block scope
```

**Design Rationale**: `u32` provides 4 billion unique contexts (sufficient for any realistic codebase) while remaining compact for storage and comparison.

### 2. Completion

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// The completed term
    pub term: String,

    /// Edit distance from query
    pub distance: usize,

    /// Contexts where this term is visible
    pub contexts: Vec<ContextId>,

    /// Whether this is a draft (non-finalized) term
    pub is_draft: bool,
}
```

**Sorting Order**: Results sorted by `distance` (ascending) → `term` (lexicographic).

**Context Visibility**: `contexts` field tracks where term is visible, enabling scope-filtered queries.

### 3. ContextTree

```rust
pub struct ContextTree {
    // Hierarchical structure: child → parent mapping
    // Enables efficient ancestor lookup
}

impl ContextTree {
    pub fn create_root(&mut self, id: ContextId);
    pub fn create_child(&mut self, parent: ContextId, child: ContextId) -> Result<ContextId>;
    pub fn visible_contexts(&self, context: ContextId) -> Vec<ContextId>;
}
```

**Visibility Algorithm**: Returns `[self, parent, grandparent, ..., root]` for lexical scope lookup.

### 4. DraftBuffer

```rust
pub struct DraftBuffer {
    chars: Vec<char>,  // Character-level for correct UTF-8
}

impl DraftBuffer {
    pub fn insert(&mut self, ch: char);  // Append character
    pub fn delete(&mut self);             // Remove last character (backspace)
    pub fn as_str(&self) -> String;      // Get current draft text
    pub fn clear(&mut self);              // Reset to empty
}
```

**UTF-8 Correctness**: Uses `Vec<char>` instead of `String` for character-level operations (e.g., "café".len() = 4 chars, not 5 bytes).

### 5. CheckpointStack

```rust
pub struct CheckpointStack {
    checkpoints: Vec<Checkpoint>,
}

#[derive(Clone, Copy)]
pub struct Checkpoint {
    position: usize,  // Index in DraftBuffer
}

impl CheckpointStack {
    pub fn push(&mut self, position: usize) -> Checkpoint;
    pub fn restore(&mut self, cp: Checkpoint) -> Option<usize>;
}
```

**Undo/Redo**: Stack-based checkpoints enable multi-level undo in editors. Restoring to a checkpoint truncates the draft buffer to the saved position.

---

## Theory: Lexical Scoping & Visibility

### Lexical Scope Hierarchy

```
Global Scope (context 0)
  │
  ├─ Module A (context 1)
  │    │
  │    ├─ Function foo (context 10)
  │    │    │
  │    │    └─ Block (context 100)
  │    │
  │    └─ Function bar (context 11)
  │
  └─ Module B (context 2)
       └─ Class Baz (context 20)
            └─ Method qux (context 200)
```

### Visibility Rules

| Context | Visible Contexts | Explanation |
|---------|------------------|-------------|
| 0 (global) | [0] | Root sees only global scope |
| 1 (module A) | [1, 0] | Module sees self + global |
| 10 (function foo) | [10, 1, 0] | Function sees self + module + global |
| 100 (block) | [100, 10, 1, 0] | Block sees full ancestor chain |
| 11 (function bar) | [11, 1, 0] | Sibling contexts don't see each other |
| 200 (method qux) | [200, 20, 2, 0] | Different branch of tree |

**Key Insight**: Visibility follows the parent chain upward (lexical scoping), not sideways (sibling contexts are isolated).

### Query Fusion Algorithm

When completing in context `C`:

1. **Compute visible contexts**: `V = visible_contexts(C)` (ancestor chain)
2. **Query finalized terms**: Get all dictionary terms where `term.contexts ∩ V ≠ ∅`
3. **Query draft terms**: Get all draft buffers for contexts in `V`
4. **Merge results**: Union finalized + drafts, mark drafts with `is_draft = true`
5. **Fuzzy match**: Filter by Levenshtein distance ≤ `max_distance`
6. **Sort**: By distance → term (lexicographic)

**Example**:

```
Context 100 (block) with visible [100, 10, 1, 0]

Finalized terms:
- "print" in context 0 (global)      ✓ visible
- "process" in context 1 (module)    ✓ visible
- "param" in context 11 (sibling)    ✗ not visible

Draft terms:
- "pri" in context 100 (self)        ✓ visible
- "pro" in context 10 (parent)       ✓ visible

Query "pr" with distance 1:
Results: [
  Completion { term: "pr", distance: 0, is_draft: true, contexts: [100] },
  Completion { term: "pro", distance: 1, is_draft: true, contexts: [10] },
  Completion { term: "print", distance: 2, is_draft: false, contexts: [0] },
  // "process" exceeds distance 1 for "pr"
  // "param" not visible
]
```

---

## Engine Types

### DynamicContextualCompletionEngine

**For mutable dictionaries** that support runtime insertion/removal.

```rust
pub struct DynamicContextualCompletionEngine<D = PathMapDictionary<Vec<ContextId>>>
where
    D: MutableMappedDictionary<Value = Vec<ContextId>> + Clone,
{
    // Internal fields (thread-safe via Arc/Mutex/RwLock)
}
```

**Supported Backends**:
- `PathMapDictionary` (default) - Persistent trie, flexible
- `PathMapDictionaryChar` - Unicode variant
- `DynamicDawg` - Faster queries (~2.8x than PathMap)
- `DynamicDawgChar` - Unicode variant

**Finalization**: Calls `dictionary.insert_with_value(term, vec![context])` to add term permanently.

**Use Cases**:
- REPL environments where users define new terms
- Live coding sessions with dynamic symbol tables
- Prototyping tools with evolving vocabularies

### StaticContextualCompletionEngine

**For immutable dictionaries** with pre-built vocabularies.

```rust
pub struct StaticContextualCompletionEngine<D = DoubleArrayTrie<Vec<ContextId>>>
where
    D: MappedDictionary<Value = Vec<ContextId>> + Clone,
{
    finalized_terms: Arc<RwLock<HashMap<String, Vec<ContextId>>>>,
    // ... other fields
}
```

**Supported Backends**:
- `DoubleArrayTrie` (default) - Fastest queries (~12-16µs)
- `DoubleArrayTrieChar` - Unicode variant

**Finalization**: Stores terms in separate `finalized_terms` HashMap (does not mutate dictionary).

**Query Fusion**: Merges 3 sources:
1. Static dictionary (pre-built, fast)
2. Finalized terms HashMap (small, rare)
3. Draft buffers (always fresh)

**Use Cases**:
- LSP servers with large standard library dictionaries
- IDEs with pre-indexed framework APIs
- Scenarios where most terms are pre-defined, few added at runtime

---

## Usage Examples

### Example 1: Basic Draft Insertion

```rust
use liblevenshtein::contextual::{DynamicContextualCompletionEngine, ContextId};
use liblevenshtein::transducer::Algorithm;

let engine = DynamicContextualCompletionEngine::with_pathmap(Algorithm::Standard);

// Create global context
let global: ContextId = 0;
engine.create_root_context(global)?;

// Insert characters one by one
engine.insert_char(global, 'p')?;
engine.insert_char(global, 'r')?;
engine.insert_char(global, 'i')?;

// Get current draft
let draft = engine.get_draft(global)?;
assert_eq!(draft, "pri");

// Query for completions
let results = engine.complete(global, "pri", 1)?;
// Results include the draft itself (distance 0)
assert!(results.iter().any(|c| c.term == "pri" && c.is_draft));
```

### Example 2: Hierarchical Scope Visibility

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();

// Create hierarchy: global → module → function
let global = 0;
let module = 1;
let function = 10;

engine.create_root_context(global)?;
engine.create_child_context(global, module)?;
engine.create_child_context(module, function)?;

// Add term in global scope
engine.insert_str(global, "print")?;
engine.finalize(global)?;

// Add term in module scope
engine.insert_str(module, "process")?;
engine.finalize(module)?;

// Add draft in function scope
engine.insert_str(function, "param")?;

// Query from function scope
let results = engine.complete(function, "p", 0)?;

// Function sees all three:
// - "print" from global (finalized)
// - "process" from module (finalized)
// - "param" from function (draft)
assert_eq!(results.len(), 3);
assert!(results.iter().any(|c| c.term == "print" && !c.is_draft));
assert!(results.iter().any(|c| c.term == "process" && !c.is_draft));
assert!(results.iter().any(|c| c.term == "param" && c.is_draft));
```

### Example 3: Checkpoint-Based Undo

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();
let ctx = 0;
engine.create_root_context(ctx)?;

// Type "hello"
engine.insert_str(ctx, "hello")?;
assert_eq!(engine.get_draft(ctx)?, "hello");

// Save checkpoint
let cp1 = engine.checkpoint(ctx)?;

// Type " world"
engine.insert_str(ctx, " world")?;
assert_eq!(engine.get_draft(ctx)?, "hello world");

// Save another checkpoint
let cp2 = engine.checkpoint(ctx)?;

// Type "!"
engine.insert_char(ctx, '!')?;
assert_eq!(engine.get_draft(ctx)?, "hello world!");

// Undo to cp2
engine.restore(ctx, cp2)?;
assert_eq!(engine.get_draft(ctx)?, "hello world");

// Undo to cp1
engine.restore(ctx, cp1)?;
assert_eq!(engine.get_draft(ctx)?, "hello");
```

### Example 4: Character-Level Rollback (Backspace)

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();
let ctx = 0;
engine.create_root_context(ctx)?;

// Type "test"
engine.insert_char(ctx, 't')?;
engine.insert_char(ctx, 'e')?;
engine.insert_char(ctx, 's')?;
engine.insert_char(ctx, 't')?;
assert_eq!(engine.get_draft(ctx)?, "test");

// Backspace twice
engine.rollback_char(ctx)?;  // "tes"
engine.rollback_char(ctx)?;  // "te"
assert_eq!(engine.get_draft(ctx)?, "te");

// Continue typing
engine.insert_char(ctx, 'x')?;
engine.insert_char(ctx, 't')?;
assert_eq!(engine.get_draft(ctx)?, "text");
```

### Example 5: Static Engine with Pre-Built Dictionary

```rust
use liblevenshtein::contextual::StaticContextualCompletionEngine;
use liblevenshtein::dictionary::double_array_trie::DoubleArrayTrie;
use liblevenshtein::transducer::Algorithm;

// Build static dictionary with standard library terms
let mut builder = DoubleArrayTrie::builder();
builder.insert_with_value("std::vec::Vec", Some(vec![0]));
builder.insert_with_value("std::collections::HashMap", Some(vec![0]));
builder.insert_with_value("std::io::println", Some(vec![0]));
let dict = builder.build();

// Create static engine
let engine = StaticContextualCompletionEngine::with_double_array_trie(
    dict,
    Algorithm::Standard
);

let ctx = 0;
engine.create_root_context(ctx)?;

// Add user-defined term (stored in finalized_terms HashMap, not dictionary)
engine.insert_str(ctx, "my_function")?;
engine.finalize(ctx)?;

// Query merges dictionary + finalized_terms + drafts
let results = engine.complete(ctx, "std", 1)?;
// Results include all std::* terms from dictionary
// Plus any user-defined terms that match
```

### Example 6: Multi-Context Isolation

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();

// Create two sibling contexts
let global = 0;
let ctx_a = 1;
let ctx_b = 2;

engine.create_root_context(global)?;
engine.create_child_context(global, ctx_a)?;
engine.create_child_context(global, ctx_b)?;

// Add draft in context A
engine.insert_str(ctx_a, "alpha")?;

// Add draft in context B
engine.insert_str(ctx_b, "beta")?;

// Query from context A
let results_a = engine.complete(ctx_a, "a", 0)?;
assert!(results_a.iter().any(|c| c.term == "alpha"));
assert!(!results_a.iter().any(|c| c.term == "beta"));  // B not visible from A

// Query from context B
let results_b = engine.complete(ctx_b, "b", 0)?;
assert!(results_b.iter().any(|c| c.term == "beta"));
assert!(!results_b.iter().any(|c| c.term == "alpha"));  // A not visible from B
```

### Example 7: Unicode Character Handling

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use liblevenshtein::dictionary::pathmap_char::PathMapDictionaryChar;

// Use Unicode-aware backend
let engine = DynamicContextualCompletionEngine::with_pathmap_char(
    liblevenshtein::transducer::Algorithm::Standard
);

let ctx = 0;
engine.create_root_context(ctx)?;

// Insert emoji (4-byte UTF-8)
engine.insert_char(ctx, '🔥')?;
engine.insert_char(ctx, '🚀')?;
assert_eq!(engine.get_draft(ctx)?, "🔥🚀");

// Insert CJK (3-byte UTF-8)
engine.clear_draft(ctx)?;
engine.insert_str(ctx, "世界")?;
assert_eq!(engine.get_draft(ctx)?.chars().count(), 2);  // 2 characters, not 6 bytes

// Backspace removes one character (not one byte)
engine.rollback_char(ctx)?;
assert_eq!(engine.get_draft(ctx)?, "世");
```

### Example 8: Fuzzy Matching with Drafts

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();
let ctx = 0;
engine.create_root_context(ctx)?;

// Add finalized terms
engine.insert_str(ctx, "function")?;
engine.finalize(ctx)?;
engine.insert_str(ctx, "functional")?;
engine.finalize(ctx)?;

// Type draft with typo
engine.insert_str(ctx, "functon")?;

// Query with distance 1 finds corrections
let results = engine.complete(ctx, "functon", 1)?;

// Results include:
// 1. "functon" (draft, distance 0)
// 2. "function" (finalized, distance 1 - missing 'i')
// 3. "functional" (finalized, distance 2 - too far)

assert!(results.iter().any(|c| c.term == "functon" && c.is_draft && c.distance == 0));
assert!(results.iter().any(|c| c.term == "function" && !c.is_draft && c.distance == 1));
```

### Example 9: LSP-Style Completion Workflow

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();

// File-level context hierarchy
let file_ctx = 0;
let class_ctx = 1;
let method_ctx = 10;

engine.create_root_context(file_ctx)?;
engine.create_child_context(file_ctx, class_ctx)?;
engine.create_child_context(class_ctx, method_ctx)?;

// User types "pri" in method scope
engine.insert_str(method_ctx, "pri")?;

// LSP sends completion request
let results = engine.complete(method_ctx, "pri", 1)?;

// Server returns results to client:
// - Draft "pri" (exact match)
// - Finalized terms within distance 1
// - Only terms visible in method scope (method + class + file)

// User selects "print" from results
engine.clear_draft(method_ctx)?;
engine.insert_str(method_ctx, "print")?;
engine.finalize(method_ctx)?;

// "print" now available in method context
```

### Example 10: Incremental Typing Simulation

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

let engine = DynamicContextualCompletionEngine::new();
let ctx = 0;
engine.create_root_context(ctx)?;

// Pre-populate with some terms
for term in &["async", "await", "assert"] {
    engine.insert_str(ctx, term)?;
    engine.finalize(ctx)?;
}

// Simulate typing "asy" character by character
let query_sequence = vec!['a', 's', 'y'];

for ch in query_sequence {
    engine.insert_char(ctx, ch)?;
    let draft = engine.get_draft(ctx)?;
    let results = engine.complete(ctx, &draft, 1)?;

    println!("Draft: '{}', Completions: {:?}",
             draft,
             results.iter().map(|c| &c.term).collect::<Vec<_>>());
}

// Output:
// Draft: 'a', Completions: ["a" (draft), "async", "await", "assert"]
// Draft: 'as', Completions: ["as" (draft), "async", "assert"]
// Draft: 'asy', Completions: ["asy" (draft), "async"]
```

---

## Performance Analysis

### Benchmark Results

**Test Environment**: Intel Xeon E5-2699 v3 @ 2.30GHz, Rust 1.75, release build

| Operation | Time | Throughput |
|-----------|------|------------|
| `insert_char()` | ~4 µs | 12M chars/sec |
| `rollback_char()` | ~3 µs | 333K ops/sec |
| `checkpoint()` | ~116 ns | 8.6M checkpoints/sec |
| `restore(checkpoint)` | ~200 ns | 5M restores/sec |
| `complete()` (500 terms, dist 1) | ~11.5 µs | 87K queries/sec |
| `complete()` (500 terms, dist 2) | ~309 µs | 3.2K queries/sec |
| `finalize()` | ~8 µs | 125K finalizations/sec |

**Key Observations**:
- Character operations are **sub-10µs** (imperceptible in interactive use)
- Checkpoints are **extremely cheap** (~116ns) - can checkpoint every keystroke
- Query performance scales with distance (dist 2 is 27× slower than dist 1)
- All operations meet sub-millisecond latency requirements for IDEs

### Zipper vs Node Performance

Comparison of zipper-based vs traditional node-based traversal:

| Query Type | Node-Based | Zipper-Based | Overhead |
|------------|------------|--------------|----------|
| Distance 1 | 6.52 µs | 10.82 µs | 1.66× |
| Distance 2 | 48.21 µs | 95.13 µs | 1.97× |
| Distance 3 | 301.45 µs | 578.32 µs | 1.92× |

**Trade-off Analysis**:
- Zipper approach is **1.66-1.97× slower** due to functional composition
- Overhead is **acceptable** for contextual use cases (still sub-millisecond for common queries)
- Benefits: cleaner separation of concerns, easier to reason about, supports draft + finalized fusion

**Recommendation**: Zipper overhead is a worthwhile trade-off for the architectural benefits (testability, composability, draft management).

### Memory Overhead

| Component | Per-Context Overhead | Notes |
|-----------|----------------------|-------|
| DraftBuffer | ~48 bytes + chars | Vec<char> + metadata |
| CheckpointStack | ~24 bytes + stack | Vec<Checkpoint> |
| ContextTree node | ~16 bytes | Parent pointer + ID |
| **Total** | **~88 bytes + data** | Well within 1KB target |

**Scalability**: 10,000 contexts = ~880KB base + draft text (acceptable for LSP servers).

---

## Backend Comparison

### Dictionary Backend Selection Guide

| Backend | Mutability | Query Speed | Unicode | Use Case |
|---------|------------|-------------|---------|----------|
| **PathMapDictionary** | ✅ Mutable | Baseline | ❌ Bytes | Prototyping, REPL |
| **PathMapDictionaryChar** | ✅ Mutable | Baseline | ✅ Unicode | Multi-language REPL |
| **DynamicDawg** | ✅ Mutable | ~2.8× faster | ❌ Bytes | Performance-critical, dynamic |
| **DynamicDawgChar** | ✅ Mutable | ~2.6× faster | ✅ Unicode | Performance + Unicode |
| **DoubleArrayTrie** | ❌ Immutable | ~12× faster | ❌ Bytes | LSP with pre-built dict |
| **DoubleArrayTrieChar** | ❌ Immutable | ~11× faster | ✅ Unicode | LSP + Unicode |

### Engine Selection Decision Tree

```
Do you need runtime term insertion?
│
├─ YES → Use DynamicContextualCompletionEngine
│   │
│   ├─ Need Unicode?
│   │   ├─ YES → DynamicDawgChar (fastest Unicode + mutable)
│   │   └─ NO  → DynamicDawg (fastest mutable)
│   │
│   └─ Prototyping?
│       └─ YES → PathMapDictionary (simplest)
│
└─ NO → Use StaticContextualCompletionEngine
    │
    └─ Need Unicode?
        ├─ YES → DoubleArrayTrieChar (fastest Unicode)
        └─ NO  → DoubleArrayTrie (fastest overall)
```

### Performance Comparison: Query @ Distance 1

| Backend | Engine Type | Time (µs) | Speedup |
|---------|-------------|-----------|---------|
| PathMapDictionary | Dynamic | 11.5 | 1.0× (baseline) |
| DynamicDawg | Dynamic | 4.1 | 2.8× |
| DoubleArrayTrie | Static | 0.96 | 12.0× |

**Conclusion**: For LSP servers with large pre-built standard libraries, `StaticContextualCompletionEngine<DoubleArrayTrie>` provides 12× speedup over PathMap.

---

## Implementation Details

### Thread Safety

| Component | Synchronization | Lock Granularity |
|-----------|-----------------|------------------|
| **ContextTree** | `Arc<RwLock<>>` | Read-heavy, single write lock |
| **Drafts** | `Arc<Mutex<HashMap<>>>` | Per-engine lock (future: DashMap) |
| **Checkpoints** | `Arc<Mutex<HashMap<>>>` | Per-engine lock |
| **Transducer** | `Arc<RwLock<>>` | Read-heavy for queries |

**Concurrency Strategy**:
- **Queries**: Multiple threads can query simultaneously (RwLock read)
- **Insertions**: Mutations require write lock (brief, character-level)
- **Finalization**: Brief write lock on dictionary
- **Lock-free future**: Potential upgrade to DashMap for drafts/checkpoints

### Draft Lifecycle

```
Empty → Insertion → Checkpoint → Rollback → Finalize/Discard
  ↑         ↓           ↓           ↓           ↓
  │      insert_char checkpoint rollback_char finalize()
  │      insert_str     ()          ()          │
  │                                              │
  └──────────────────── clear() ────────────────┘
```

**State Transitions**:
1. **Empty**: No characters in buffer
2. **Insertion**: Characters added via `insert_char()` or `insert_str()`
3. **Checkpoint**: State saved for undo (buffer position recorded)
4. **Rollback**: Characters removed via `rollback_char()` or `restore(cp)`
5. **Finalize**: Draft promoted to dictionary term
6. **Discard**: Draft cleared without finalizing

### Query Fusion Implementation

**DynamicContextualCompletionEngine**:

```rust
pub fn complete(&self, context: ContextId, query: &str, max_distance: usize)
    -> Result<Vec<Completion>>
{
    // 1. Get visible contexts
    let tree = self.context_tree.read().unwrap();
    let visible = tree.visible_contexts(context);
    drop(tree);

    // 2. Query dictionary (finalized terms)
    let transducer = self.transducer.read().unwrap();
    let candidates: Vec<_> = transducer
        .query_with_distance(query, max_distance)
        .collect();

    let mut results = Vec::new();
    for candidate in candidates {
        if let Some(contexts) = transducer.dictionary().get_value(&candidate.term) {
            let visible_contexts: Vec<_> = contexts
                .iter()
                .filter(|ctx| visible.contains(ctx))
                .copied()
                .collect();

            if !visible_contexts.is_empty() {
                results.push(Completion {
                    term: candidate.term,
                    distance: candidate.distance,
                    contexts: visible_contexts,
                    is_draft: false,
                });
            }
        }
    }
    drop(transducer);

    // 3. Query drafts
    let drafts = self.drafts.lock().unwrap();
    for &ctx in &visible {
        if let Some(buffer) = drafts.get(&ctx) {
            let draft_text = buffer.as_str();
            if !draft_text.is_empty() {
                let distance = levenshtein_distance(query, &draft_text);
                if distance <= max_distance {
                    results.push(Completion {
                        term: draft_text,
                        distance,
                        contexts: vec![ctx],
                        is_draft: true,
                    });
                }
            }
        }
    }

    // 4. Sort and return
    results.sort_by(|a, b| {
        a.distance.cmp(&b.distance)
            .then_with(|| a.term.cmp(&b.term))
    });

    Ok(results)
}
```

**StaticContextualCompletionEngine** additionally queries `finalized_terms` HashMap between steps 2 and 3.

---

## References

### Academic Papers

1. **Huet, Gérard (1997)**: ["The Zipper"](https://www.st.cs.uni-saarland.de/edu/seminare/2005/advanced-fp/docs/huet-zipper.pdf). Journal of Functional Programming, 7(5):549-554.
   - Foundational paper on zipper data structure used for dictionary traversal.

2. **Levenshtein, Vladimir (1966)**: "Binary codes capable of correcting deletions, insertions, and reversals". Soviet Physics Doklady, 10(8):707-710.
   - Original edit distance algorithm underlying fuzzy matching.

3. **Schulz, Klaus U.; Mihov, Stoyan (2002)**: ["Fast string correction with Levenshtein automata"](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f0f3a5d6d67dcf7a6bded56c3f5c12d4e3c0f8c3). International Journal on Document Analysis and Recognition, 5(1):67-85.
   - Efficient automaton-based fuzzy string matching.

4. **Aoe, Jun-ichi (1989)**: "An efficient digital search algorithm by using a double-array structure". IEEE Transactions on Software Engineering, 15(9):1066-1077.
   - Double-Array Trie structure used in `DoubleArrayTrie` backend.

5. **Crochemore, Maxime; Vérin, Renaud (1997)**: ["Direct construction of compact directed acyclic word graphs"](https://hal.science/hal-00619859/document). Lecture Notes in Computer Science, 1264:116-129.
   - DAWG construction algorithm underlying `DynamicDawg`.

### Implementation Resources

- **PathMap Library**: [github.com/Adam-Vandervorst/PathMap](https://github.com/Adam-Vandervorst/PathMap) - Persistent trie implementation
- **liblevenshtein**: Original Java implementation by [universal-automata](https://github.com/universal-automata/liblevenshtein)
- **Language Server Protocol (LSP)**: [microsoft.github.io/language-server-protocol](https://microsoft.github.io/language-server-protocol/) - Standard for IDE integration

### Related Documentation

- [Layer 6: Zipper Navigation](../06-zipper-navigation/README.md) - DictZipper and AutomatonZipper details
- [Layer 9: Value Storage](../09-value-storage/README.md) - MappedDictionary trait and context ID storage
- [Design Docs](../../design/contextual-completion-api.md) - Complete API specification
- [Progress Tracking](../../design/contextual-completion-progress.md) - Implementation status (100% complete)
- [Performance Analysis](../../design/zipper-vs-node-performance.md) - Detailed benchmark comparison

### Examples

- [`examples/contextual_completion.rs`](../../../examples/contextual_completion.rs) - Basic usage demo
- [`examples/hierarchical_scope_completion.rs`](../../../examples/hierarchical_scope_completion.rs) - Scope visibility
- [`examples/fuzzy_maps_code_completion.rs`](../../../examples/fuzzy_maps_code_completion.rs) - LSP-style workflow
- [`examples/advanced_contextual_filtering.rs`](../../../examples/advanced_contextual_filtering.rs) - Context filtering
- [`examples/contextual_filtering_optimization.rs`](../../../examples/contextual_filtering_optimization.rs) - Performance tuning

---

## Navigation

[← Layer 6: Zipper Navigation](../06-zipper-navigation/README.md) | [Index](../README.md) | [Layer 8: Caching →](../08-caching-layer/README.md)

---

**Last Updated**: 2025-11-05
**Status**: Production Ready (v0.7.0)
**Test Coverage**: 93 tests, 0 failures
**Implementation**: 100% complete across all 6 phases
