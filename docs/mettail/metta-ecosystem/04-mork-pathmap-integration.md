# MORK and PathMap Integration

This document describes the MORK (MeTTa Optimal Reduction Kernel) and PathMap
storage layer integration that underpins MeTTaTron and enables efficient pattern
matching for semantic type checking.

**Sources**:
- MORK: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- PathMap: `/home/dylon/Workspace/f1r3fly.io/PathMap/`

---

## Table of Contents

1. [Overview](#overview)
2. [PathMap Architecture](#pathmap-architecture)
3. [MORK Space Implementation](#mork-space-implementation)
4. [Integration Flow](#integration-flow)
5. [Performance Optimizations](#performance-optimizations)
6. [Cross-Project Integration](#cross-project-integration)
7. [Relevance to Semantic Type Checking](#relevance-to-semantic-type-checking)

---

## Overview

MORK and PathMap form the storage and pattern matching backbone for MeTTaTron:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MORK/PathMap Stack                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    MeTTaTron Compiler                      │ │
│  │  (Parser → Compiler → Evaluator)                          │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      MORK Kernel                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Space     │  │  Reduction  │  │  Pattern    │        │ │
│  │  │   Manager   │  │  Engine     │  │  Matching   │        │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │ │
│  │         │                │                │                │ │
│  │         └────────────────┼────────────────┘                │ │
│  │                          │                                  │ │
│  └──────────────────────────┼──────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      PathMap                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │    Trie     │  │   Zipper    │  │   Ring      │        │ │
│  │  │   Storage   │  │  Navigation │  │   Module    │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │  ┌─────────────┐  ┌─────────────┐                         │ │
│  │  │ Morphisms   │  │ Merkleize   │                         │ │
│  │  └─────────────┘  └─────────────┘                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **MORK** | Pattern matching, reduction, space management |
| **PathMap** | Trie-based storage, algebraic operations, merkleization |

---

## PathMap Architecture

PathMap is a trie-based storage system with algebraic structure, designed for
efficient storage and retrieval of hierarchical data.

### Module Organization

```
PathMap/src/
├── lib.rs           # Core PathMap type and operations
├── ring.rs          # Ring module for algebraic operations
├── zipper.rs        # Zipper pattern for navigation
├── morphisms.rs     # Structure-preserving transformations
└── merkleize.rs     # Content-addressed hashing
```

### Core Data Structure

PathMap implements a trie (prefix tree) with these properties:

```rust
/// PathMap: A trie structure for hierarchical key-value storage
pub struct PathMap<K, V> {
    /// Root node of the trie
    root: Node<K, V>,
    /// Configuration options
    config: PathMapConfig,
}

/// Node in the trie
struct Node<K, V> {
    /// Value stored at this node (if any)
    value: Option<V>,
    /// Children indexed by key segment
    children: HashMap<K, Box<Node<K, V>>>,
}
```

### Ring Module

The ring module provides algebraic operations on PathMap, enabling:

- **Addition**: Merge two PathMaps (union of keys)
- **Multiplication**: Compose PathMaps (path concatenation)
- **Identity elements**: Empty PathMap (additive), root-only (multiplicative)

```rust
/// PathMap forms a ring over values
impl<K, V> Ring for PathMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Ring,
{
    fn zero() -> Self {
        PathMap::empty()
    }

    fn one() -> Self {
        PathMap::singleton(vec![], V::one())
    }

    fn add(&self, other: &Self) -> Self {
        self.merge(other, |a, b| a.add(b))
    }

    fn mul(&self, other: &Self) -> Self {
        self.compose(other)
    }
}
```

### Zipper Pattern

The zipper module enables efficient navigation and modification:

```rust
/// Zipper for PathMap navigation
pub struct PathMapZipper<K, V> {
    /// Current focus node
    focus: Node<K, V>,
    /// Path back to root (breadcrumbs)
    context: Vec<(K, HashMap<K, Box<Node<K, V>>>)>,
}

impl<K, V> PathMapZipper<K, V> {
    /// Move focus to child
    fn down(&mut self, key: &K) -> Option<()>;

    /// Move focus to parent
    fn up(&mut self) -> Option<()>;

    /// Modify value at focus
    fn modify<F>(&mut self, f: F) where F: FnOnce(&mut Option<V>);

    /// Reconstruct PathMap from zipper
    fn unzip(self) -> PathMap<K, V>;
}
```

### Merkleization

Content-addressed hashing for distributed storage:

```rust
/// Merkle hash of PathMap subtree
pub fn merkleize<K, V, H>(pathmap: &PathMap<K, V>) -> Hash
where
    K: AsRef<[u8]>,
    V: AsRef<[u8]>,
    H: Hasher,
{
    // Recursively hash children
    let child_hashes: Vec<Hash> = pathmap.children()
        .map(|(k, child)| {
            let child_hash = merkleize::<K, V, H>(child);
            H::hash(&[k.as_ref(), &child_hash])
        })
        .collect();

    // Combine with value hash
    let value_hash = pathmap.value()
        .map(|v| H::hash(v.as_ref()))
        .unwrap_or(H::hash(&[]));

    H::hash_combine(&child_hashes, &value_hash)
}
```

---

## MORK Space Implementation

MORK's Space type wraps PathMap with pattern matching capabilities.

### Space Structure

From `MORK/kernel/src/space.rs`:

```rust
/// Space: Pattern matching over PathMap storage
pub struct Space {
    /// Underlying PathMap storage
    map: PathMap<Vec<u8>, Vec<u8>>,
    /// Shared mapping handle for distributed access
    handle: SharedMappingHandle,
    /// Pattern matching index
    pattern_index: PatternIndex,
}
```

### Pattern Matching

MORK implements efficient pattern matching via the `coreferential_transition`
function, which handles De Bruijn indexed variables:

```rust
/// Coreferential transition for pattern matching
///
/// Handles the case where the same variable appears multiple times
/// in a pattern, ensuring all occurrences unify to the same value.
pub fn coreferential_transition(
    space: &Space,
    pattern: &[u8],
    bindings: &mut Bindings,
) -> Result<Vec<Match>, MatchError> {
    // Parse De Bruijn indices from pattern
    let indices = parse_debruijn_indices(pattern)?;

    // Track variable occurrences
    let mut occurrences: HashMap<u16, Vec<Position>> = HashMap::new();
    for (pos, idx) in indices {
        occurrences.entry(idx).or_default().push(pos);
    }

    // Ensure coreferential consistency
    for (idx, positions) in &occurrences {
        if positions.len() > 1 {
            // All positions must unify to same value
            let first_value = bindings.get(idx)?;
            for pos in &positions[1..] {
                let value = extract_at_position(pattern, *pos)?;
                if value != first_value {
                    return Err(MatchError::CoreferentialMismatch);
                }
            }
        }
    }

    // Perform matching
    space.match_pattern(pattern, bindings)
}
```

### De Bruijn Index Encoding

Variables are encoded using De Bruijn indices for efficient handling:

```rust
/// Encode variable with De Bruijn index
fn encode_variable(index: u16) -> Vec<u8> {
    let mut bytes = vec![VARIABLE_TAG];
    bytes.extend_from_slice(&index.to_le_bytes());
    bytes
}

/// Variable tag in byte encoding
const VARIABLE_TAG: u8 = 0xFF;

/// Parse De Bruijn index from bytes
fn parse_debruijn_index(bytes: &[u8]) -> Option<u16> {
    if bytes.first() == Some(&VARIABLE_TAG) && bytes.len() >= 3 {
        Some(u16::from_le_bytes([bytes[1], bytes[2]]))
    } else {
        None
    }
}
```

---

## Integration Flow

### MeTTa to PathMap Conversion

The conversion pipeline from MeTTa values to PathMap storage:

```
┌─────────────────────────────────────────────────────────────────┐
│                 MeTTa → PathMap Conversion                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MettaValue                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ConversionContext                             │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ variable_indices: HashMap<String, u16>              │  │ │
│  │  │ next_index: u16                                     │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              metta_to_mork_bytes()                         │ │
│  │                                                            │ │
│  │  Symbol → [TAG_SYMBOL, ...bytes...]                       │ │
│  │  Variable → [TAG_VAR, de_bruijn_index...]                 │ │
│  │  Expression → [TAG_EXPR, length, ...children...]         │ │
│  │  Grounded → [TAG_GROUNDED, type_id, ...data...]          │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│                       Vec<u8>                                   │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    MORK Space                              │ │
│  │  space.insert(path, bytes)?                               │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    PathMap                                 │ │
│  │  Stored at trie path for efficient retrieval              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### PathMap to Rholang Par Conversion

For Rholang integration, PathMap converts to/from Rholang Par:

```rust
/// Convert MeTTa state to Rholang Par via PathMap
pub fn metta_state_to_pathmap_par(state: &MettaState) -> Result<Par, Error> {
    // Convert MeTTa state to PathMap representation
    let pathmap = state_to_pathmap(state)?;

    // Convert PathMap to Rholang Par
    pathmap_to_par(&pathmap)
}

/// Convert PathMap to Rholang Par
fn pathmap_to_par(pathmap: &PathMap) -> Result<Par, Error> {
    // Build Par from PathMap structure
    let mut par = Par::default();

    for (path, value) in pathmap.iter() {
        let channel = path_to_channel(&path)?;
        let data = bytes_to_par_data(&value)?;
        par.sends.push(Send {
            chan: channel,
            data: vec![data],
            persistent: false,
        });
    }

    Ok(par)
}
```

### Magic Numbers for Serialization

MeTTaTron uses magic numbers to identify data types in serialized form:

```rust
/// Magic number for MeTTa multiplicities field
pub const METTA_MULTIPLICITIES_MAGIC: u64 = 0x4D455454_41_4D554C; // "METTAMUL"

/// Magic number for MeTTa expression
pub const METTA_EXPR_MAGIC: u64 = 0x4D455454_41_45585052; // "METTAEXPR"

/// Magic number for MeTTa symbol
pub const METTA_SYMBOL_MAGIC: u64 = 0x4D455454_41_53594D; // "METTASYM"
```

---

## Performance Optimizations

### Bloom Filter for Negative Lookups

MORK uses Bloom filters for fast negative lookups:

```rust
/// Space with Bloom filter optimization
pub struct OptimizedSpace {
    space: Space,
    bloom: BloomFilter<Vec<u8>>,
}

impl OptimizedSpace {
    /// Check if pattern might match (fast negative)
    pub fn might_match(&self, pattern: &[u8]) -> bool {
        self.bloom.might_contain(pattern)
    }

    /// Match pattern with Bloom pre-filter
    pub fn match_pattern(&self, pattern: &[u8]) -> Option<Vec<Match>> {
        // Fast negative check
        if !self.might_match(pattern) {
            return None;
        }

        // Full match only if Bloom positive
        self.space.match_pattern(pattern)
    }
}
```

### LRU Cache for Hot Patterns

Frequently accessed patterns are cached:

```rust
/// Cache for hot pattern results
pub struct PatternCache {
    cache: LruCache<Vec<u8>, Vec<Match>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl PatternCache {
    pub fn get_or_compute<F>(&mut self, pattern: &[u8], compute: F) -> Vec<Match>
    where
        F: FnOnce() -> Vec<Match>,
    {
        if let Some(cached) = self.cache.get(pattern) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return cached.clone();
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = compute();
        self.cache.put(pattern.to_vec(), result.clone());
        result
    }
}
```

### Concurrent Read Support

PathMap supports concurrent reads via shared handles:

```rust
/// Shared handle for concurrent access
pub struct SharedMappingHandle {
    inner: Arc<RwLock<PathMap<Vec<u8>, Vec<u8>>>>,
}

impl SharedMappingHandle {
    /// Read-only access (concurrent)
    pub fn read(&self) -> RwLockReadGuard<PathMap<Vec<u8>, Vec<u8>>> {
        self.inner.read().unwrap()
    }

    /// Write access (exclusive)
    pub fn write(&self) -> RwLockWriteGuard<PathMap<Vec<u8>, Vec<u8>>> {
        self.inner.write().unwrap()
    }
}
```

---

## Cross-Project Integration

### Integration Points

| Project | Integration | Purpose |
|---------|------------|---------|
| **MeTTaTron** | MORK backend | Pattern matching for evaluation |
| **Rholang** | PathMap Par | Cross-language data exchange |
| **liblevenshtein** | PhoneticNormalizedDictionary | Phonetic-aware fuzzy dictionary lookups |
| **MeTTaIL** | Predicate storage | Type predicate evaluation |

### FuzzySource Integration [CURRENT]

liblevenshtein integrates via `PhoneticNormalizedDictionary` for phonetic-aware fuzzy matching:

```rust
use liblevenshtein::dictionary::phonetic_normalized::{
    PhoneticNormalizedDictionary, PhoneticNormalizedCandidate
};
use liblevenshtein::phonetic::rules::english;

/// FuzzySource trait for phonetic-aware fuzzy dictionary lookup.
pub trait FuzzySource {
    /// Fuzzy lookup with phonetic normalization and maximum edit distance.
    fn fuzzy_lookup(&self, query: &str, max_distance: usize)
        -> Vec<PhoneticNormalizedCandidate>;
}

/// PathMap implementation using PhoneticNormalizedDictionary.
impl FuzzySource for PathMap<Vec<u8>, Vec<u8>> {
    fn fuzzy_lookup(&self, query: &str, max_distance: usize)
        -> Vec<PhoneticNormalizedCandidate>
    {
        // Extract terms from PathMap
        let terms: Vec<String> = self.iter()
            .filter_map(|(k, _)| String::from_utf8(k.clone()).ok())
            .collect();

        // Build PhoneticNormalizedDictionary with English phonetic rules
        let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
            &terms,
            english::combined()
        );

        // Query with phonetic normalization
        // - d=0: Direct trie lookup (100-300× faster)
        // - d≥1: FuzzyMultiMap with Levenshtein automaton pruning O(k log n)
        dict.query(query, max_distance)
    }
}
```

### PhoneticNormalizedDictionary Features

- **Automatic normalization**: Terms normalized at build time using phonetic rules
- **FuzzyMultiMap acceleration**: O(k log n) fuzzy queries via Levenshtein automaton pruning
- **Thread-local buffers**: Optimized for concurrent queries (H3 optimization)
- **Pre-compiled rules**: `english::zompist()`, `english::homophones()`, `english::text_speak()`, `english::combined()`

### MORK Space with Phonetic Matching

```rust
use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedCandidate;

/// MORK Space implementation using PhoneticNormalizedDictionary.
impl FuzzySource for Space {
    fn fuzzy_lookup(&self, query: &str, max_distance: usize)
        -> Vec<PhoneticNormalizedCandidate>
    {
        self.map.fuzzy_lookup(query, max_distance)
    }
}

/// Usage in MeTTa queries:
/// - (fuzzy "colr" 2 $result)          - Standard Levenshtein
/// - (fuzzy-phonetic "fone" 2 $result) - Phonetic-aware matching
```

---

## Relevance to Semantic Type Checking

### Type Predicate Storage

MORK/PathMap can store type predicates for semantic checking:

```
PathMap Structure for Types:
├── types/
│   ├── Int/           → type definition
│   ├── String/        → type definition
│   └── (-> $a $b)/    → function type pattern
├── predicates/
│   ├── well-typed/    → well-typed predicate
│   ├── terminates/    → termination predicate
│   └── namespace-safe/ → namespace isolation
└── rules/
    ├── subtype/       → subtyping rules
    └── inference/     → type inference rules
```

### Pattern-Based Type Checking

Type checking as pattern matching in MORK:

```rust
/// Check if term has type via pattern matching
fn check_type(space: &Space, term: &[u8], expected_type: &[u8]) -> bool {
    // Build type query pattern
    let pattern = build_type_pattern(term, expected_type);

    // Match against type knowledge base
    let matches = space.match_pattern(&pattern);

    !matches.is_empty()
}

/// Build pattern for type query
fn build_type_pattern(term: &[u8], expected_type: &[u8]) -> Vec<u8> {
    // (: term expected_type)
    let mut pattern = vec![TAG_EXPR, 3]; // Expression of 3 elements
    pattern.push(TAG_SYMBOL);
    pattern.extend_from_slice(b":");
    pattern.extend_from_slice(term);
    pattern.extend_from_slice(expected_type);
    pattern
}
```

### Behavioral Type Patterns

OSLF behavioral types as MORK patterns:

```rust
/// Store behavioral type predicate
fn store_behavioral_type(space: &mut Space, predicate: &BehavioralPredicate) {
    let path = predicate_to_path(predicate);
    let encoding = encode_predicate(predicate);
    space.insert(&path, &encoding);
}

/// Query behavioral property
fn query_behavioral(space: &Space, process: &[u8], property: Property) -> bool {
    let pattern = build_behavioral_pattern(process, property);
    !space.match_pattern(&pattern).is_empty()
}
```

---

## Summary

MORK and PathMap provide:

1. **Efficient Storage**: Trie-based PathMap with zipper navigation
2. **Pattern Matching**: MORK's De Bruijn indexed matching
3. **Algebraic Structure**: Ring module for compositional operations
4. **Content Addressing**: Merkleization for distributed storage
5. **Performance**: Bloom filters, LRU caching, concurrent reads
6. **Integration**: Cross-project compatibility via shared storage

These capabilities directly support MeTTaIL semantic type checking by:
- Storing type predicates as patterns
- Enabling efficient type queries via pattern matching
- Supporting behavioral type reasoning through reduction tracking
- Providing Rholang interop via PathMap Par conversion

---

## References

- MORK: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- PathMap: `/home/dylon/Workspace/f1r3fly.io/PathMap/`
- See [03-mettatron.md](./03-mettatron.md) for MeTTaTron integration
- See [bibliography.md](../reference/bibliography.md) for complete references
