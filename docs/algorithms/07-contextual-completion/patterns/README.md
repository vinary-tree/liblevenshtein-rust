# Contextual Completion Patterns

**Decision guide for selecting the right implementation pattern for your contextual completion use case.**

---

[← Back to Layer 7](../README.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Pattern Selection Guide](#pattern-selection-guide)
3. [Available Patterns](#available-patterns)
4. [Decision Matrix](#decision-matrix)
5. [Common Use Cases](#common-use-cases)
6. [Performance Comparison](#performance-comparison)

---

## Overview

This directory contains production-ready patterns for implementing contextual code completion at scale. Each pattern addresses specific architectural challenges:

| Pattern | Primary Use Case | Key Benefit |
|---------|------------------|-------------|
| **[Parallel Workspace Indexing](parallel-workspace-indexing.md)** | Multi-document projects (LSP, IDE) | ~150× speedup via parallel construction + binary tree merge |

**Pattern Structure**: Each pattern guide includes:
- Problem statement and motivation
- Complete working implementation
- Performance analysis and benchmarks
- Common pitfalls and troubleshooting
- Production-ready code examples

---

## Pattern Selection Guide

### Quick Decision Tree

```
Are you building an LSP server or IDE?
  ↓ YES → Parallel Workspace Indexing
  ↓ NO
  ↓
Are you working with multiple files/documents?
  ↓ YES → Parallel Workspace Indexing
  ↓ NO
  ↓
Single file with incremental updates?
  ↓ YES → Use DynamicContextualCompletionEngine directly
  ↓         (see ../implementation/completion-engine.md)
  ↓
Static pre-built dictionary (e.g., stdlib)?
  ↓ YES → Use StaticContextualCompletionEngine
            (see ../implementation/completion-engine.md)
```

---

## Available Patterns

### 1. Parallel Workspace Indexing

**→ [Full Guide](parallel-workspace-indexing.md)**

**Problem**: Sequential construction and merging of dictionaries for multi-document workspaces is `$\mathcal{O}(N^{2}\cdot n\cdot m)$`, becoming prohibitively slow beyond ~10 documents.

**Solution**: Parallel construction using Rayon + binary tree reduction achieves `$\mathcal{O}(n\cdot m\cdot \log  N)$` with ~150× speedup for 100 documents.

**When to Use**:
- LSP servers indexing multiple source files
- IDE workspace-wide completion
- Multi-module projects
- Code search across repositories

**Key Metrics**:
- 100 documents × 1K terms: **0.3s** (vs 50s sequential)
- Memory: Linear scaling (~30KB per 1K terms)
- Parallelism: Scales to available CPU cores

**Quick Example**:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use rayon::prelude::*;

// Build per-document dictionaries in parallel
let dicts: Vec<DynamicDawg<Vec<u32>>> = documents
    .par_iter()
    .map(|(ctx_id, source)| {
        let dict = DynamicDawg::new();
        for term in extract_terms(source) {
            dict.insert_with_value(term, vec![*ctx_id]);
        }
        dict
    })
    .collect();

// Merge using binary tree reduction
let merged = merge_tree_parallel(dicts, merge_deduplicated);

// Inject into engine
let engine = DynamicContextualCompletionEngine::with_dictionary(
    merged,
    Algorithm::Standard
);
```

**→ See**: [Complete Implementation](parallel-workspace-indexing.md)

---

## Decision Matrix

### By Characteristics

| Characteristic | Pattern | Notes |
|----------------|---------|-------|
| **Multiple documents** | Parallel Workspace Indexing | Essential for LSP/IDE |
| **Single document** | Direct Engine Usage | No pattern needed |
| **Pre-built dictionary** | Static Engine + DI | Use `StaticContextualCompletionEngine::with_double_array_trie()` |
| **Incremental updates** | Dynamic Engine | Use `DynamicContextualCompletionEngine` with `finalize()` |
| **100+ documents** | Parallel Workspace Indexing | Sequential becomes ~50× slower |
| **Real-time requirements** | Parallel Workspace Indexing | Sub-second indexing critical |

### By Application Type

| Application | Recommended Pattern | Backend | Rationale |
|-------------|-------------------|---------|-----------|
| **LSP Server** | Parallel Workspace Indexing | DynamicDawg | Fast parallel build, runtime updates supported |
| **IDE Plugin** | Parallel Workspace Indexing | DynamicDawg | Same as LSP |
| **REPL** | Direct Engine Usage | DynamicDawg | Single context, incremental symbols |
| **Static Analysis Tool** | Static Engine | DoubleArrayTrie | Pre-built stdlib, no runtime changes |
| **Prototyping** | Direct Engine Usage | PathMapDictionary | Simplest API |

---

## Common Use Cases

### Use Case 1: LSP Server (Rust, TypeScript, Python, etc.)

**Requirements**:
- Index 50-500 source files
- Sub-second initial indexing
- Incremental updates on file changes
- Context-aware completion (scoped visibility)

**Recommended Pattern**: **Parallel Workspace Indexing**

**Implementation**:

```rust
// Initial workspace indexing
let workspace_files = discover_source_files("./src");

let dicts: Vec<DynamicDawg<Vec<ContextId>>> = workspace_files
    .par_iter()
    .enumerate()
    .map(|(idx, path)| {
        let ctx_id = idx as u32 + 1;
        let source = std::fs::read_to_string(path).unwrap();
        let terms = parse_identifiers(&source);  // Language-specific tokenizer

        let dict = DynamicDawg::new();
        for term in terms {
            dict.insert_with_value(term, vec![ctx_id]);
        }
        dict
    })
    .collect();

let merged = merge_tree_parallel(dicts, merge_deduplicated);
let engine = DynamicContextualCompletionEngine::with_dictionary(merged, Algorithm::Standard);

// Create context hierarchy
for (file_id, deps) in dependency_graph {
    if deps.is_empty() {
        engine.create_root_context(file_id);
    } else {
        for dep_id in deps {
            engine.create_child_context(file_id, dep_id)?;
        }
    }
}

// Handle file changes incrementally
fn on_file_change(engine: &Engine, file_id: ContextId, new_source: &str) {
    // Extract new terms
    let new_terms = parse_identifiers(new_source);

    // Finalize new symbols
    for term in new_terms {
        engine.finalize_direct(file_id, &term)?;
    }
}
```

**→ See**: [Parallel Workspace Indexing](parallel-workspace-indexing.md)

---

### Use Case 2: IDE Plugin (VS Code, IntelliJ, etc.)

**Requirements**:
- Same as LSP (many IDEs use LSP protocol internally)
- Responsive UI (sub-100ms query latency)
- Memory efficiency for large projects

**Recommended Pattern**: **Parallel Workspace Indexing** + **DynamicDawg backend**

**Key Optimizations**:

```rust
// Use DynamicDawg for 2.8× faster queries than PathMapDictionary
let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);

// Build workspace index asynchronously
tokio::spawn(async move {
    let dicts = build_workspace_dicts_parallel(&workspace);
    let merged = merge_tree_parallel(dicts, merge_deduplicated);

    // Inject into engine
    engine_handle.replace_dictionary(merged);  // Hypothetical API
});

// Query with low latency (~4µs @ distance 1)
let results = engine.complete(current_file_id, "std::", 1);
```

---

### Use Case 3: REPL or Notebook Environment

**Requirements**:
- Single interactive session
- Incremental symbol definitions
- No multi-document complexity

**Recommended Pattern**: **Direct Engine Usage** (no pattern needed)

**Implementation**:

```rust
let engine = DynamicContextualCompletionEngine::with_dynamic_dawg(Algorithm::Standard);
let session_ctx = 0;
engine.create_root_context(session_ctx);

// User defines variable
engine.finalize_direct(session_ctx, "my_var")?;

// User defines function
engine.finalize_direct(session_ctx, "my_function")?;

// Later, user types "my"
let results = engine.complete(session_ctx, "my", 0);
// Returns: ["my_var", "my_function"]
```

**→ See**: [Completion Engine Documentation](../implementation/completion-engine.md)

---

### Use Case 4: Static Library Documentation Browser

**Requirements**:
- Pre-built dictionary (stdlib, framework APIs)
- No runtime modifications
- Maximum query performance

**Recommended Pattern**: **Static Engine** with **DoubleArrayTrie**

**Implementation**:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

// Pre-build static dictionary (offline or at startup)
let mut builder = DoubleArrayTrie::builder();
for symbol in load_stdlib_symbols() {
    builder.insert_with_value(&symbol, Some(vec![0]));
}
let dict = builder.build();

// Create static engine (12× faster queries than PathMapDictionary)
let engine = StaticContextualCompletionEngine::with_double_array_trie(
    dict,
    Algorithm::Standard
);

// User can still define local terms (stored in separate HashMap)
let user_ctx = 1;
engine.create_root_context(user_ctx);
engine.finalize_direct(user_ctx, "my_local_var")?;

// Queries combine static dictionary + user terms
let results = engine.complete(user_ctx, "std", 1);
// Fast queries over large stdlib dictionary
```

**→ See**: [Completion Engine Documentation](../implementation/completion-engine.md#staticcontextualcompletionengine-api)

---

## Performance Comparison

### Pattern Performance (100 Documents, 1K Terms Each)

| Approach | Construction Time | Query Time (dist=1) | Memory Overhead | Scalability |
|----------|-------------------|---------------------|-----------------|-------------|
| **Sequential + PathMap** | ~50s | 11.5µs | Low | Poor (`$\mathcal{O}(N^{2})$`) |
| **Parallel Workspace + DynamicDawg** | **0.3s** | **4.1µs** | Low | Excellent (`$\mathcal{O}(\log  N)$`) |
| **Static + DoubleArrayTrie** | ~1s (build) + 0.1s (load) | **0.96µs** | Lowest | N/A (pre-built) |

**Key Takeaways**:
- Parallel workspace indexing is **~167× faster** than sequential for construction
- DynamicDawg provides **2.8× faster** queries than PathMapDictionary
- DoubleArrayTrie provides **12× faster** queries than PathMapDictionary (static only)
- Memory scales linearly with vocabulary size (~30KB per 1K terms)

---

## Next Steps

1. **Read pattern guides**: Start with [Parallel Workspace Indexing](parallel-workspace-indexing.md)
2. **Review engine APIs**: See [Completion Engine Documentation](../implementation/completion-engine.md)
3. **Choose backend**: See [Dictionary Layer Documentation](../../01-dictionary-layer/README.md)
4. **Implement**: Use production-ready code examples from pattern guides

---

[← Back to Layer 7](../README.md)
