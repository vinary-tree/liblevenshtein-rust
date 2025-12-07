# PathMap Integration Infrastructure

**Status**: Active Integration
**Last Updated**: 2024-12-06

---

## Table of Contents

1. [Overview](#overview)
2. [PathMap as the Primary Integration Point](#pathmap-as-the-primary-integration-point)
3. [Architecture Alignment](#architecture-alignment)
4. [PathMap Feature in liblevenshtein](#pathmap-feature-in-liblevenshtein)
5. [Shared Zipper Pattern](#shared-zipper-pattern)
6. [PathMapDictionary Implementation](#pathmapdictionary-implementation)
7. [Integration with MORK](#integration-with-mork)
8. [Extended PathMap Schemas](#extended-pathmap-schemas)
9. [Performance Characteristics](#performance-characteristics)
10. [Configuration Guide](#configuration-guide)

---

## Overview

PathMap is a trie-based prefix-compressed key-value store that serves as the shared storage layer for three integrated projects:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│   │ liblevenshtein│  │     MORK      │  │   MeTTa App   │  │
│   │   (fuzzy)     │  │  (patterns)   │  │   (queries)   │  │
│   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  │
└───────────┼──────────────────┼──────────────────┼───────────┘
            │                  │                  │
            v                  v                  v
┌─────────────────────────────────────────────────────────────┐
│                     PathMap (Shared)                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Trie-based key-value store with zipper navigation  │   │
│   │  - Prefix compression                               │   │
│   │  - Memory-mapped I/O                                │   │
│   │  - Concurrent read access                           │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Project Locations

| Project | Location | Purpose |
|---------|----------|---------|
| PathMap | `/home/dylon/Workspace/f1r3fly.io/PathMap/` | Shared trie storage |
| liblevenshtein | `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/` | Fuzzy matching automata |
| MORK | `/home/dylon/Workspace/f1r3fly.io/MORK/` | MeTTa pattern matching |

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Shared vocabulary** | Single dictionary used by fuzzy matching and pattern queries |
| **Memory efficiency** | Prefix compression reduces memory footprint |
| **Cache coherence** | All projects read from same memory-mapped trie |
| **Zipper compatibility** | Unified navigation abstraction across all projects |
| **Dialogue persistence** | Conversation history and entity state preserved across sessions |
| **Learning storage** | User preferences and learned patterns stored efficiently |

---

## PathMap as the Primary Integration Point

PathMap is **the primary shared layer** between liblevenshtein and MORK. It is important to understand that:

1. **liblevenshtein is NOT embedded into MORK** - liblevenshtein remains an external library
2. **MORK has an adapter** (FuzzySource) that calls liblevenshtein - it does not contain fuzzy matching code
3. **PathMap is the shared storage** - both projects read from the same memory-mapped trie

### Three Integration Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: MeTTa Query Syntax                                    │
│  !(match &space (fuzzy "colr" 2 $result) $result)              │
│  User-facing query language                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: MORK FuzzySource Adapter                              │
│  Implements Source trait → calls liblevenshtein                 │
│  Location: MORK/kernel/src/fuzzy_source.rs                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: liblevenshtein + PathMap (THIS LAYER)                 │
│  PathMapDictionary backend for liblevenshtein transducers       │
│  Location: liblevenshtein-rust/src/dictionary/pathmap.rs       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PathMap Storage (Shared)                                       │
│  Memory-mapped trie shared by liblevenshtein and MORK          │
│  Same data used by BTMSource, ACTSource, and PathMapDictionary │
└─────────────────────────────────────────────────────────────────┘
```

### Why PathMap is Primary

| Aspect | Explanation |
|--------|-------------|
| **Single source of truth** | Both liblevenshtein and MORK read the same vocabulary |
| **No duplication** | Dictionary words stored once, not copied between libraries |
| **Cache coherence** | Memory-mapped access means consistent data across components |
| **Independent operation** | liblevenshtein can function without MORK (using PathMap directly) |

### Data Flow

```
MeTTa Query → MORK Space
                 │
                 ▼
     FuzzySource.query()  ←── MORK adapter (Layer 2)
                 │
                 ▼
 liblevenshtein::Transducer  ←── External library
                 │
                 ▼
    PathMapDictionary.get()  ←── Layer 1 (this doc)
                 │
                 ▼
    PathMap (memory-mapped)  ←── Shared storage
```

---

## Architecture Alignment

All three projects share fundamental architectural patterns that enable clean integration.

### Common Patterns

#### 1. Trie-Based Storage

```
         root
        /    \
       c      d
      / \      \
     a   o      o
    /   / \      \
   t   l   w      g
       o   ?
       r
```

- **PathMap**: Trie nodes with values at terminals
- **liblevenshtein**: Dictionary tries (DoubleArrayTrie, DAWG, PathMapDictionary)
- **MORK**: BTMSource and ACTSource use trie-like structures

#### 2. Zipper Navigation

All three projects use zippers for efficient traversal:

```rust
// PathMap zipper
pub trait Zipper {
    fn down(&mut self, key: &[u8]) -> bool;
    fn up(&mut self) -> bool;
    fn value(&self) -> Option<&Value>;
}

// liblevenshtein dictionary interface
pub trait Dictionary {
    fn contains(&self, key: &[u8]) -> bool;
    fn get(&self, key: &[u8]) -> Option<&Value>;
    fn iter(&self) -> impl Iterator<Item = (&[u8], &Value)>;
}

// MORK Source trait
pub trait Source {
    type Zipper: Clone;
    fn zipper(&self) -> Self::Zipper;
    fn down(&self, z: &mut Self::Zipper, key: &[u8]) -> bool;
    fn up(&self, z: &mut Self::Zipper);
}
```

#### 3. Byte-Oriented Keys

All projects use byte sequences as keys:

```rust
// PathMap
pathmap.insert(b"hello", value);

// liblevenshtein
dictionary.contains(b"hello");

// MORK (symbols encoded as bytes)
space.query(symbol_bytes);
```

---

## PathMap Feature in liblevenshtein

liblevenshtein provides a `pathmap-backend` feature for direct PathMap integration.

### Feature Configuration

```toml
# Cargo.toml
[dependencies]
liblevenshtein = {
    path = "../liblevenshtein-rust",
    features = ["pathmap-backend"]
}
```

### Conditional Compilation

```rust
// src/dictionary/mod.rs

#[cfg(feature = "pathmap-backend")]
pub mod pathmap;

#[cfg(feature = "pathmap-backend")]
pub use pathmap::PathMapDictionary;
```

### Available Dictionary Backends

| Backend | Feature Flag | Use Case |
|---------|--------------|----------|
| `DoubleArrayTrie` | default | Fast static dictionary (ASCII) |
| `DoubleArrayTrieChar` | default | Fast static dictionary (UTF-8) |
| `DynamicDawg` | default | Updatable dictionary (ASCII) |
| `DynamicDawgChar` | default | Updatable dictionary (UTF-8) |
| `SuffixAutomaton` | default | Substring search |
| **`PathMapDictionary`** | `pathmap-backend` | **Shared storage with MORK** |

---

## Shared Zipper Pattern

The zipper pattern enables efficient traversal without full tree reconstruction.

### Conceptual Model

A zipper is a "focus" on a tree node plus the "context" needed to reconstruct the whole tree:

```
         [root]
        /      \
      [a]      [b]
      /          \
   (focus)       [c]

Zipper at "a":
  - focus: node "a"
  - context: [path from root, siblings]

Operations:
  - down(): Move focus to child
  - up(): Move focus to parent (using context)
  - value(): Get value at focus
```

### PathMap Zipper Implementation

```rust
// PathMap provides ReadZipperUntracked for read-only traversal
pub struct ReadZipperUntracked<'a> {
    map: &'a PathMap,
    path: Vec<u8>,
    node: NodeRef,
}

impl<'a> ReadZipperUntracked<'a> {
    pub fn down(&mut self, key: &[u8]) -> bool {
        // Follow edge labeled `key` to child node
        if let Some(child) = self.node.child(key) {
            self.path.extend_from_slice(key);
            self.node = child;
            true
        } else {
            false
        }
    }

    pub fn up(&mut self) -> bool {
        // Return to parent node
        if self.path.is_empty() {
            false
        } else {
            // Reconstruct parent from root (or use cached parent ref)
            self.path.pop();
            self.node = self.map.node_at(&self.path);
            true
        }
    }

    pub fn value(&self) -> Option<&Value> {
        self.node.value()
    }
}
```

### liblevenshtein Zipper Adapter

```rust
// src/dictionary/pathmap.rs

use pathmap::ReadZipperUntracked;

/// Adapter to use PathMap zipper with liblevenshtein transducers
pub struct PathMapZipper<'a> {
    inner: ReadZipperUntracked<'a>,
}

impl<'a> PathMapZipper<'a> {
    /// Create zipper at root of PathMap
    pub fn new(pathmap: &'a PathMap) -> Self {
        Self {
            inner: pathmap.zipper(),
        }
    }

    /// Descend to child by label
    pub fn down(&mut self, label: u8) -> bool {
        self.inner.down(&[label])
    }

    /// Return to parent
    pub fn up(&mut self) -> bool {
        self.inner.up()
    }

    /// Check if current node is a terminal (has value)
    pub fn is_terminal(&self) -> bool {
        self.inner.value().is_some()
    }

    /// Get current path as key
    pub fn key(&self) -> &[u8] {
        self.inner.path()
    }
}
```

---

## PathMapDictionary Implementation

The `PathMapDictionary` struct provides the `Dictionary` trait implementation for PathMap.

### Core Implementation

```rust
// src/dictionary/pathmap.rs

use pathmap::PathMap;
use crate::dictionary::Dictionary;

/// Dictionary backed by PathMap trie
pub struct PathMapDictionary {
    /// Underlying PathMap storage
    map: PathMap,
}

impl PathMapDictionary {
    /// Create from existing PathMap
    pub fn from_pathmap(map: PathMap) -> Self {
        Self { map }
    }

    /// Create from word list file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let map = PathMap::new();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let word = line?;
            map.insert(word.as_bytes(), ());  // Unit value for simple dictionary
        }

        Ok(Self { map })
    }

    /// Create memory-mapped from file
    pub fn mmap(path: impl AsRef<Path>) -> Result<Self, Error> {
        let map = PathMap::mmap(path)?;
        Ok(Self { map })
    }

    /// Get underlying PathMap (for sharing with MORK)
    pub fn inner(&self) -> &PathMap {
        &self.map
    }
}

impl Dictionary for PathMapDictionary {
    type Zipper<'a> = PathMapZipper<'a> where Self: 'a;

    fn contains(&self, key: &[u8]) -> bool {
        self.map.get(key).is_some()
    }

    fn get(&self, key: &[u8]) -> Option<&Value> {
        self.map.get(key)
    }

    fn zipper(&self) -> Self::Zipper<'_> {
        PathMapZipper::new(&self.map)
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl DictionaryIter for PathMapDictionary {
    type Iter<'a> = PathMapIter<'a> where Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        PathMapIter::new(&self.map)
    }
}
```

### Iterator Implementation

```rust
/// Iterator over PathMap entries
pub struct PathMapIter<'a> {
    map: &'a PathMap,
    stack: Vec<(PathMapZipper<'a>, usize)>,
    current_key: Vec<u8>,
}

impl<'a> PathMapIter<'a> {
    fn new(map: &'a PathMap) -> Self {
        let mut stack = Vec::new();
        let zipper = PathMapZipper::new(map);
        stack.push((zipper, 0));

        Self {
            map,
            stack,
            current_key: Vec::new(),
        }
    }
}

impl<'a> Iterator for PathMapIter<'a> {
    type Item = (&'a [u8], &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        // DFS traversal of trie
        while let Some((mut zipper, child_idx)) = self.stack.pop() {
            // Try to descend to next child
            if let Some(label) = zipper.child_label(child_idx) {
                // Push current state back (for next child)
                self.stack.push((zipper.clone(), child_idx + 1));

                // Descend to this child
                zipper.down(label);
                self.current_key.push(label);

                // Push child state
                self.stack.push((zipper.clone(), 0));

                // If this is a terminal, yield it
                if zipper.is_terminal() {
                    if let Some(value) = self.map.get(&self.current_key) {
                        return Some((&self.current_key, value));
                    }
                }
            } else {
                // No more children, backtrack
                self.current_key.pop();
            }
        }

        None
    }
}
```

---

## Integration with MORK

PathMap serves as the shared storage between liblevenshtein and MORK.

### Shared Dictionary Scenario

```rust
use pathmap::PathMap;
use liblevenshtein::{PathMapDictionary, Transducer, Algorithm};
use mork_kernel::{Space, FuzzySource};

fn setup_shared_dictionary() -> Result<(), Error> {
    // Create shared PathMap
    let pathmap = PathMap::mmap("dictionary.pathmap")?;

    // Use in liblevenshtein for fuzzy matching
    let dictionary = PathMapDictionary::from_pathmap(pathmap.clone());
    let transducer = Transducer::builder()
        .dictionary(dictionary)
        .algorithm(Algorithm::Transposition)
        .max_distance(2)
        .build();

    // Use same PathMap in MORK for pattern queries
    let mork_space = Space::new();
    mork_space.add_source(FuzzySource::from_transducer(transducer));

    // Now both systems share the same underlying trie
    Ok(())
}
```

### FuzzySource Integration

```rust
// MORK kernel integration

use liblevenshtein::{PathMapDictionary, Transducer};
use mork_kernel::Source;

/// MORK Source backed by liblevenshtein transducer
pub struct FuzzySource {
    transducer: Transducer<PathMapDictionary>,
    max_distance: usize,
}

impl FuzzySource {
    pub fn new(pathmap: PathMap, max_distance: usize) -> Self {
        let dictionary = PathMapDictionary::from_pathmap(pathmap);
        let transducer = Transducer::builder()
            .dictionary(dictionary)
            .algorithm(Algorithm::Transposition)
            .max_distance(max_distance)
            .build();

        Self { transducer, max_distance }
    }
}

impl Source for FuzzySource {
    type Zipper = TransducerZipper;

    fn zipper(&self) -> Self::Zipper {
        TransducerZipper::new(&self.transducer)
    }

    fn query(&self, pattern: &[u8]) -> Vec<Match> {
        self.transducer
            .query(pattern, self.max_distance)
            .map(|candidate| Match {
                key: candidate.term.clone(),
                distance: candidate.distance,
            })
            .collect()
    }
}
```

---

## Extended PathMap Schemas

Beyond dictionary storage, PathMap supports the extended correction architecture with schemas for dialogue state, agent configuration, and knowledge storage.

### Dialogue State Schema

PathMap stores conversation context for dialogue-aware correction:

```
/dialogue/{dialogue_id}/
    /meta/
        created_at -> timestamp
        participants -> [participant_id, ...]
        status -> active|archived
    /turn/{turn_id}/
        raw -> raw text bytes
        corrected -> corrected text bytes
        speaker -> participant_id
        timestamp -> unix timestamp
        speech_act -> encoded speech act
        entities/ -> entity mentions
        topics/ -> topic references
    /entity/{entity_id}/
        name -> canonical name
        type -> entity type
        attributes/ -> key-value attributes
        introduced_at -> turn_id
    /coref/{entity_id}/
        {mention_idx} -> (turn_id, span_start, span_end)
    /topic/{topic_id}/
        label -> topic label
        parent -> parent topic_id (optional)
        keywords/ -> {keyword} -> count
        active_turns/ -> [turn_id, ...]
```

### Agent Configuration Schema

PathMap stores LLM agent configuration and learned patterns:

```
/agent/{agent_id}/
    /config/
        endpoint -> LLM endpoint configuration
        max_tokens -> token limit
        correction_level -> 0.0-1.0
    /feedback/
        /pattern/{pattern_id}/
            error_pattern -> pattern specification
            correction -> correction template
            confidence -> float
            support_count -> int
    /user/{user_id}/
        formality_level -> 0.0-1.0
        vocabulary_level -> 0.0-1.0
        personal_dictionary/ -> {word} -> true
        ignored_words/ -> {word} -> true
        error_patterns/ -> [pattern_id, ...]
```

### Knowledge Base Schema

PathMap stores facts for hallucination detection and fact checking:

```
/knowledge/
    /entity/{entity_id}/
        canonical_name -> string
        type -> entity type
        aliases/ -> [alias, ...]
    /fact/{fact_id}/
        subject -> entity_id
        predicate -> relation name
        object -> entity_id or value
        confidence -> 0.0-1.0
```

### Usage with Extended Layers

```rust
use pathmap::PathMap;

// Store dialogue turn
fn store_turn(pathmap: &PathMap, dialogue_id: &str, turn: &Turn) {
    let base = format!("/dialogue/{}/turn/{}", dialogue_id, turn.id);
    pathmap.insert(format!("{}/raw", base).as_bytes(), turn.raw.as_bytes());
    pathmap.insert(format!("{}/speaker", base).as_bytes(), turn.speaker.as_bytes());
    pathmap.insert(format!("{}/timestamp", base).as_bytes(), &turn.timestamp.to_le_bytes());
}

// Store learned error pattern
fn store_pattern(pathmap: &PathMap, agent_id: &str, pattern: &ErrorPattern) {
    let base = format!("/agent/{}/feedback/pattern/{}", agent_id, pattern.id);
    pathmap.insert(format!("{}/error_pattern", base).as_bytes(), pattern.error.as_bytes());
    pathmap.insert(format!("{}/correction", base).as_bytes(), pattern.correction.as_bytes());
}
```

**See**: [Dialogue Context Layer](../../mettail/dialogue/README.md) for usage details.

---

## Performance Characteristics

### Memory Efficiency

PathMap uses prefix compression to minimize memory:

```
Words: ["cat", "car", "card", "care", "careful"]

Uncompressed trie:    Prefix-compressed PathMap:
      root                    root
     /                       /
    c                       c
    |                       |
    a                       a
   / \                     /|\
  t   r                   t r  [shared prefix "ca"]
      |                     |
      d                    d,e [edges]
      |                     |
      ?                  ful [suffix compression]
```

**Memory comparison** (100K English words):

| Structure | Memory |
|-----------|--------|
| `Vec<String>` | ~8 MB |
| `HashSet<String>` | ~12 MB |
| `DoubleArrayTrie` | ~4 MB |
| `PathMap` | ~3 MB |

### Lookup Performance

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| Exact lookup | O(k) | <1 μs |
| Prefix scan | O(k + m) | <10 μs |
| Fuzzy query (d=2) | O(k × 3^d) | <100 μs |

Where:
- k = key length
- m = number of matches
- d = edit distance

### Concurrent Access

PathMap supports concurrent reads via memory mapping:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let pathmap = Arc::new(PathMap::mmap("dictionary.pathmap")?);

// Multiple threads can read concurrently
let results: Vec<_> = queries.par_iter()
    .map(|query| {
        let dict = PathMapDictionary::from_pathmap(pathmap.clone());
        transducer_for(dict).query(query, 2).collect()
    })
    .collect();
```

---

## Configuration Guide

### Building liblevenshtein with PathMap

```bash
# Enable PathMap backend
cargo build --features pathmap-backend

# Enable PathMap + all optimizations
cargo build --release --features "pathmap-backend simd bloom"
```

### Creating a Shared Dictionary

```rust
use pathmap::PathMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn create_shared_dictionary(
    word_list: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<(), Error> {
    let map = PathMap::new();

    // Load words
    let file = File::open(word_list)?;
    for line in BufReader::new(file).lines() {
        let word = line?;
        map.insert(word.as_bytes(), ());
    }

    // Persist to disk
    map.save(output)?;

    println!("Created dictionary with {} entries", map.len());
    Ok(())
}
```

### Loading in Applications

```rust
// In liblevenshtein
let dict = PathMapDictionary::mmap("shared.pathmap")?;

// In MORK
let source = BTMSource::from_pathmap("shared.pathmap")?;

// Both now share the same memory-mapped trie
```

### Cargo Dependencies

```toml
# In liblevenshtein-rust/Cargo.toml
[features]
pathmap-backend = ["dep:pathmap"]

[dependencies]
pathmap = { path = "../../PathMap", optional = true }

# In MORK/kernel/Cargo.toml
[dependencies]
liblevenshtein = { path = "../../liblevenshtein-rust", features = ["pathmap-backend"] }
pathmap = { path = "../../PathMap" }
```

---

## Related Documentation

### Core Integration

- [MORK Integration Overview](../mork/README.md) - Full MORK integration architecture
- [FuzzySource Implementation](../mork/fuzzy_source.md) - Phase A details
- [PathMap Book](https://github.com/your-org/PathMap/pathmap-book/) - PathMap documentation
- [Zipper Pattern](https://en.wikipedia.org/wiki/Zipper_(data_structure)) - Theoretical background

### Extended Architecture

- [Dialogue Context Layer](../../mettail/dialogue/README.md) - Turn history and coreference resolution
- [Agent Learning Layer](../../mettail/agent-learning/README.md) - Feedback patterns and user preferences
- [LLM Integration](../../mettail/llm-integration/README.md) - Context injection using PathMap
- [Correction WFST Architecture](../../mettail/correction-wfst/01-architecture-overview.md) - Full three-tier architecture overview

---

## Troubleshooting

### Common Issues

**Issue**: `feature "pathmap-backend" is not enabled`

```
Solution: Add --features pathmap-backend to cargo command
```

**Issue**: `PathMap file not found`

```
Solution: Ensure dictionary file exists and path is correct.
Use absolute paths or paths relative to working directory.
```

**Issue**: Memory mapping fails on large dictionary

```
Solution: Ensure sufficient virtual address space.
On 32-bit systems, use file-based PathMap instead of mmap.
```

**Issue**: Concurrent write conflicts

```
Solution: PathMap supports concurrent reads but single writer.
Use write locks or process-level coordination for updates.
```

---

## Future Enhancements

1. **Incremental updates**: Efficient dictionary updates without full rebuild
2. **Distributed PathMap**: Network-accessible shared dictionary
3. **Compression**: Additional compression for very large dictionaries
4. **Bloom filter integration**: Fast negative lookups before trie traversal
