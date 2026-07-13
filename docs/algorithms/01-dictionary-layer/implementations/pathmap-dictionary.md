# PathMapDictionary Implementation

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory: Persistent Data Structures](#theory-persistent-data-structures)
3. [PathMap Library](#pathmap-library)
4. [Data Structure](#data-structure)
5. [Construction Methods](#construction-methods)
6. [Accessor Methods](#accessor-methods)
7. [Union Operations](#union-operations)
8. [Usage Examples](#usage-examples)
9. [Performance Analysis](#performance-analysis)
10. [When to Use](#when-to-use)
11. [References](#references)

## Overview

`PathMapDictionary` is a dictionary backend built on the **PathMap** library, which provides persistent (immutable) trie structures with structural sharing. It's the simplest dynamic dictionary option but trades performance for simplicity and immutability guarantees.

### Key Advantages

- 🔄 **Full dynamic updates**: Insert AND remove at runtime
- 🔒 **Thread-safe**: Lock-free concurrent reads, atomic-swap writes
- 📦 **Simple implementation**: Thin wrapper around PathMap
- 💎 **Persistent semantics**: Structural sharing between versions
- 🎯 **Easy to use**: Straightforward API

### Key Trade-offs

- ⚠️ **Slower queries**: 2-3x slower than DoubleArrayTrie
- ⚠️ **Higher memory**: More overhead than specialized tries
- ⚠️ **Feature-gated**: Requires `pathmap-backend` feature

### When to Use

✅ **Use PathMapDictionary when:**
- Simplicity is more important than maximum performance
- Need full insert/remove capabilities
- Prefer well-tested external library
- Experimenting or prototyping

⚠️ **Consider alternatives when:**
- Performance is critical → Use `DoubleArrayTrie` (3x faster)
- Need maximum efficiency → Use `DynamicDawg`
- Unicode required → Use `PathMapDictionaryChar`

## Theory: Persistent Data Structures

### What are Persistent Data Structures?

**Persistent** data structures preserve previous versions after modifications through **structural sharing**.

**Example**: Adding "test" to dictionary containing ["best", "rest"]

**Mutable approach** (traditional):
```
Before:  root → 'b'/'r' → 'est'
After:   root → 'b'/'r'/'t' → 'est'  (modifies in-place)
Old version lost!
```

**Persistent approach** (PathMap):
```
Version 1:  root₁ → 'b'/'r' → 'est'
Version 2:  root₂ → 'b'/'r'/'t' → 'est'
                     ↑   ↑    ↑
                     └───┴────┘
                   Shared nodes (not copied)

Both versions coexist!
```

### Structural Sharing

Only changed path from root is copied; rest is shared:

```
Insert "team" into {"test", "testing"}:

Old tree:
  root → 't' → 'e' → 's' → 't' (final)
                      ↓
                     'i' → 'n' → 'g' (final)

New tree (after adding "team"):
  root' → 't' → 'e' → 's' → 't' (final)  ← Shared
                ↓       ↓
               'a'     'i' → 'n' → 'g' (final)  ← Shared
                ↓
               'm' (final)  ← New

Nodes marked "Shared" are reused, not copied
```

**Memory**: Only `$\mathcal{O}(m)$` new nodes for m-character insert

## PathMap Library

### External Dependency

PathMapDictionary wraps the `pathmap` crate:
- **Repository**: [https://github.com/Adam-Vandervorst/PathMap](https://github.com/Adam-Vandervorst/PathMap)
- **Purpose**: Persistent trie data structure
- **License**: MIT

### Enabling PathMapDictionary

Add to `Cargo.toml`:

```toml
[dependencies]
liblevenshtein = { version = "0.9", features = ["pathmap-backend"] }
```

Or use CLI:

```bash
cargo add liblevenshtein --features pathmap-backend
```

### PathMap Features

- **Persistent**: Old versions preserved
- **Structural sharing**: Efficient memory use
- **Thread-safe**: Immutable data structures
- **Generic values**: Map terms to arbitrary types

## Data Structure

### Core Components

```rust
pub struct PathMapDictionary<V: DictionaryValue = ()> {
    // Single lock-free snapshot: the PathMap handle and the term count are
    // published together inside one immutable `PathMapState<V>`.
    state: Arc<ArcSwap<PathMapState<V>>>,
}
```

### Wrapper Design

PathMapDictionary is a thin wrapper that:
1. Manages PathMap lifecycle
2. Tracks term count
3. Provides liblevenshtein Dictionary trait
4. Handles thread safety via a lock-free atomic snapshot (`ArcSwap`)

### Memory Layout

```
┌─────────────────┬─────────────────┐
│ Component       │ Overhead        │
├─────────────────┼─────────────────┤
│ Arc pointer     │ 8 bytes         │
│ ArcSwap cell    │ 8 bytes (atomic)│
│ PathMap         │ ~32 bytes/node  │
│ term_count      │ 8 bytes         │
└─────────────────┴─────────────────┘
```

**Per-node overhead**: ~32 bytes (HashMap-based)

**Example**: 10,000-term dictionary `$\approx$` 320 KB

### Clone Behavior & Memory Semantics

`PathMapDictionary` holds a **single** `Arc<ArcSwap<PathMapState<V>>>` internally, making `.clone()` a **shallow copy** that shares all underlying data. The clone behavior is like `DynamicDawg`: one lock-free, Arc-wrapped component:

```rust
use libdictenstein::pathmap::PathMapDictionary;

let dict1: PathMapDictionary = PathMapDictionary::from_terms(vec!["test", "testing"]);
let dict2 = dict1.clone();  // O(1) - increments ONE Arc refcount

// Both dict1 and dict2 share the SAME underlying PathMap and term count
dict1.insert("new_term");
assert!(dict2.contains("new_term"));  // ✅ Mutations visible through dict2!

// Term count is also shared
assert_eq!(dict1.len(), Some(3));
assert_eq!(dict2.len(), Some(3));  // Same count
```

#### Characteristics

| Property | Behavior | Impact |
|----------|----------|--------|
| **Time Complexity** | `$\mathcal{O}(1)$` | One atomic increment |
| **Space Complexity** | `$\mathcal{O}(1)$` | ~8 bytes (one Arc pointer) |
| **Data Sharing** | ✅ Complete | All clones share PathMap + term count |
| **Mutation Visibility** | ✅ Global | Changes via any clone affect all |
| **Thread Safety** | ✅ Lock-free | Readers never block; writers swap atomically |
| **Independence** | ❌ None | No isolation between clones |

#### How Clone Works

The clone operation increments a **single** atomic reference counter:

```rust
pub struct PathMapDictionary<V> {
    state: Arc<ArcSwap<PathMapState<V>>>,  // ← single lock-free Arc
}

// Cloning increments the single Arc refcount
let dict2 = dict1.clone();
// Equivalent to:
// Arc::clone(&dict1.state)
// Cost: ~1-2 CPU cycles (one atomic increment)
```

**What gets cloned:**
- ✅ Arc smart pointer for the shared state (~8 bytes on stack)
- ❌ NOT the ArcSwap cell
- ❌ NOT the PathMap trie structure
- ❌ NOT the term count value itself

**Memory allocation:**
- Zero heap allocation
- Only stack space for one Arc pointer (~8 bytes)
- All data remains shared

#### Single-Arc Lock-Free Design

PathMapDictionary publishes one immutable snapshot behind a single atomic pointer,
so readers never take a lock and never block a writer:

```rust
// Readers load the current immutable snapshot with no lock (lock-free)
let state = self.state.load_full();   // Arc<PathMapState<V>>: PathMap + term count
// `state.map` is the shared PathMap; `state.len` is the term count.

// Writers build a new PathMapState and publish it with an atomic compare-and-swap.
```

**Why a single Arc?**
- **Consistency**: The PathMap handle and term count are published together, so a reader never observes a torn map/count pair
- **Lock-free reads**: A reader loads a snapshot without blocking a concurrent writer
- **Cheaper clone**: One atomic increment instead of two

#### Structural Sharing vs Arc Sharing

**Important distinction** - PathMapDictionary has TWO types of sharing:

1. **Arc-based sharing (clone behavior):**
   ```rust
   let dict2 = dict1.clone();
   // dict1 and dict2 share the SAME PathMap instance
   dict1.insert("new");
   assert!(dict2.contains("new"));  // ✅ Visible
   ```

2. **PathMap structural sharing (persistent data structure):**
   ```rust
   let mut map1 = PathMap::new();
   map1.insert(b"test", 1);

   let mut map2 = map1.clone();  // PathMap's clone creates new version
   map2.insert(b"new", 2);

   // map1 and map2 share internal trie nodes where possible
   // But are independent: map1 doesn't see "new"
   ```

**For PathMapDictionary:**
- `.clone()` creates Arc-based sharing (visible mutations)
- PathMap's internal structural sharing is orthogonal (optimization)

#### When to Use Cloning

✅ **Good use cases:**

1. **Multi-threaded access:**
   ```rust
   use std::thread;

   let dict: PathMapDictionary = PathMapDictionary::from_terms(vec!["hello", "world"]);

   let handles: Vec<_> = (0..4).map(|_| {
       let dict_clone = dict.clone();
       thread::spawn(move || {
           dict_clone.contains("hello")
       })
   }).collect();
   ```

2. **Configuration management:**
   ```rust
   let config_dict: PathMapDictionary<String> = load_config();

   // Share across services
   let service1_dict = config_dict.clone();
   let service2_dict = config_dict.clone();

   // All see updates when config reloads
   reload_config_into(&config_dict);
   ```

3. **Caching and lookup tables:**
   ```rust
   let cache: PathMapDictionary<CachedValue> = build_cache();

   // Share cache across request handlers
   for _ in 0..10 {
       let handler_cache = cache.clone();
       spawn_handler(handler_cache);
   }
   ```

❌ **Bad use cases (common mistakes):**

1. **Expecting independent copies:**
   ```rust
   let dict1: PathMapDictionary = PathMapDictionary::from_terms(vec!["original"]);
   let dict2 = dict1.clone();

   dict1.insert("modified");
   // ❌ WRONG: Expecting dict2 unchanged
   // ✅ REALITY: dict2 also contains "modified"
   ```

2. **Creating versioned snapshots:**
   ```rust
   let dict: PathMapDictionary<u32> = load_data();
   let v1 = dict.clone();  // ❌ NOT a snapshot!

   dict.insert("v2_data");
   // v1 now also contains v2_data - not versioned
   ```

3. **Isolating test fixtures:**
   ```rust
   let base_fixture: PathMapDictionary = create_test_data();
   let test1_dict = base_fixture.clone();  // ❌ Shared!
   let test2_dict = base_fixture.clone();  // ❌ Shared!

   // Modifications in test1 affect test2!
   ```

#### Alternative: True Independence

For **independent copies** where mutations don't affect other instances:

**Option 1: Serialize/Deserialize**
```rust
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};

// Create a deep copy via serialization. `PathMapDictionary` does not implement serde's
// `Serialize`/`Deserialize`; it round-trips through `BincodeSerializer`, which encodes the
// dictionary's *terms* via the `Dictionary` trait and rebuilds it with `DictionaryFromTerms`.
let mut bytes = Vec::new();
BincodeSerializer::serialize(&dict1, &mut bytes)?;
let dict2: PathMapDictionary = BincodeSerializer::deserialize(&bytes[..])?;

// Now independent
dict1.insert("new");
assert!(!dict2.contains("new"));  // ✅ Independent
```

**Option 2: Rebuild from terms**
```rust
// Extract all terms
let terms: Vec<String> = dict1.iter().collect();

// Build new independent dictionary
let dict2: PathMapDictionary = PathMapDictionary::from_terms(terms);
```

**Option 3: Extract with values**
```rust
// For dictionaries with values
let entries: Vec<(String, V)> = dict1
    .iter()
    .filter_map(|term| dict1.get_value(term).map(|v| (term.clone(), v)))
    .collect();

let dict2: PathMapDictionary<V> = PathMapDictionary::from_terms_with_values(entries);
```

**Cost comparison:**

| Method | Time | Space | Independence |
|--------|------|-------|--------------|
| `.clone()` | `$\mathcal{O}(1)$` | `$\mathcal{O}(1)$` | ❌ Shared |
| Serialize/Deserialize | `$\mathcal{O}(n)$` | `$\mathcal{O}(n)$` | ✅ Full |
| Rebuild from terms | `$\mathcal{O}(n \cdot \log m)$` | `$\mathcal{O}(n)$` | ✅ Full |
| Rebuild with values | `$\mathcal{O}(n \cdot \log m)$` | `$\mathcal{O}(n)$` | ✅ Full |

#### Comparison with Other Dictionaries

| Dictionary | Arc Count | Clone Cost | Shared Data? |
|------------|-----------|------------|--------------|
| **PathMapDictionary** | 1 (state) | `$\mathcal{O}(1)$` | ✅ Yes |
| **DynamicDawg** | 1 (inner) | `$\mathcal{O}(1)$` | ✅ Yes |
| **DynamicDawgChar** | 1 (inner) | `$\mathcal{O}(1)$` | ✅ Yes |
| **DoubleArrayTrie** | 0 (no Arc) | `$\mathcal{O}(n)$` | ❌ No |
| **DoubleArrayTrieChar** | 0 (no Arc) | `$\mathcal{O}(n)$` | ❌ No |

**Key differences:**
- PathMapDictionary: One Arc increment (single lock-free state snapshot)
- DynamicDawg variants: One Arc increment (inner struct contains count)
- DoubleArrayTrie: Full deep copy (immutable, no Arc needed)

#### Thread Safety Considerations

PathMapDictionary's single lock-free snapshot supports unlimited concurrent readers:

```rust
use std::thread;

let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
    ("key1", 100),
    ("key2", 200),
]);

// Multiple concurrent readers
let readers: Vec<_> = (0..10).map(|i| {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.get_value(&format!("key{}", i))
    })
}).collect();

// Concurrent writer (does NOT block readers; publishes via atomic swap)
let writer = {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.insert_with_value("key3", 300)
    })
};
```

**Lock-free semantics:**
- **Read operations** (lock-free snapshot load): `contains()`, `get_value()`, `len()`, iteration
- **Write operations** (atomic snapshot publish): `insert()`, `insert_with_value()`, `remove()`, `union_with()`
- **Atomicity**: The PathMap and term count are published together in one `PathMapState`, so a reader never observes a torn map/count pair

**Performance implications:**
- Reads are lock-free: a reader atomically loads a snapshot and never blocks
- Writes publish a new snapshot via compare-and-swap (retried under contention)
- Historical note (pre-lock-free `RwLock` model): read-lock overhead was ~10-20ns and write-lock overhead ~50-100ns plus contention; the lock-free model removes the read-lock cost entirely

#### Summary

**Key Takeaways:**
1. 🔗 `.clone()` creates **shallow copy** with one Arc increment (state snapshot)
2. 🚀 **`$\mathcal{O}(1)$`** time and space - just atomic reference counting
3. 🔄 **Mutations visible** across all clones (Arc-based sharing)
4. 🌳 **Structural sharing** is separate (PathMap's persistent trie optimization)
5. 🔒 **Thread-safe** via a single lock-free atomic snapshot (readers never block)
6. 📊 For **independence**, use serialization or rebuild from terms (`$\mathcal{O}(n)$` cost)

## Construction Methods

PathMapDictionary provides constructors optimized for simple use cases and rapid prototyping.

### Overview

| Constructor | Complexity | Use Case | Thread-Safe |
|-------------|-----------|----------|-------------|
| `new()` | `$\mathcal{O}(1)$` | Empty start | ✅ |
| `from_terms()` | `$\mathcal{O}(n \cdot \log m)$` | Simple list | ✅ |
| `from_terms_with_values()` | `$\mathcal{O}(n \cdot \log m)$` | With metadata | ✅ |

Where n = number of terms, m = dictionary size (grows with insertions)

**Note**: PathMapDictionary uses `insert()` internally which is `$\mathcal{O}(\log m)$`, making bulk construction `$\mathcal{O}(n \cdot \log m)$` vs `$\mathcal{O}(n \cdot m)$` for DAWG variants.

### Empty Dictionary

Create an empty dictionary for incremental updates:

```rust
use libdictenstein::pathmap::PathMapDictionary;

// Create empty dictionary
let dict: PathMapDictionary = PathMapDictionary::new();

// Add terms incrementally
dict.insert("hello");
dict.insert("world");

// With values
let valued_dict: PathMapDictionary<u32> = PathMapDictionary::new();
valued_dict.insert_with_value("apple", 100);
valued_dict.insert_with_value("banana", 200);
```

**Characteristics:**
- **Time**: `$\mathcal{O}(1)$` - Minimal initialization
- **Memory**: ~80 bytes (one Arc pointer + empty PathMap + term count)
- **Simplicity**: Easiest to use, minimal boilerplate

**When to use:**
- ✅ Prototyping and quick experiments
- ✅ Small dictionaries (< 1,000 terms)
- ✅ When simplicity matters more than performance

### From Terms

Build from iterator of terms:

```rust
use libdictenstein::pathmap::PathMapDictionary;

// From Vec
let terms = vec!["test", "testing", "tester"];
let dict = PathMapDictionary::from_terms(terms);

// From any iterator
use std::collections::HashSet;
let term_set: HashSet<&str> = ["dog", "cat", "bird"].iter().copied().collect();
let dict = PathMapDictionary::from_terms(term_set);
```

**Characteristics:**
- **Time**: `$\mathcal{O}(n \cdot \log m)$` where m grows from 0 to n
- **Memory**: ~32 bytes per node (HashMap-based)
- **Structural sharing**: Minimal (PathMap not optimized for bulk insert)

### From Terms with Values

Build with associated values (frequencies, IDs, etc.):

```rust
use libdictenstein::pathmap::PathMapDictionary;

type ContextId = u32;

// Term frequencies
let freq_dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
    ("the", 1000000),
    ("hello", 50000),
    ("rare", 10),
]);

// Context IDs for code completion
let completion_dict: PathMapDictionary<Vec<ContextId>> =
    PathMapDictionary::from_terms_with_values(vec![
        ("println", vec![1, 2, 3]),  // Global contexts
        ("my_var", vec![42]),         // Local context
    ]);

// Configuration values
let config_dict: PathMapDictionary<String> = PathMapDictionary::from_terms_with_values(vec![
    ("app.name", "MyApp".to_string()),
    ("app.version", "1.0.0".to_string()),
    ("app.debug", "false".to_string()),
]);
```

**Value type requirements:**
- Must implement `DictionaryValue` trait
- Bounds: `Clone + Send + Sync + 'static`
- **Recommended**: Use `PathMapDictionary` for simple value types; `DynamicDawg` for complex structures

### Constructor Comparison

**Performance** (10,000 terms, Intel Xeon E5-2699 v3 @ 2.30GHz):

| Method | Time | Memory | vs DynamicDawg |
|--------|------|--------|----------------|
| `new()` + inserts | ~12ms | ~320KB | ~3× slower |
| `from_terms()` | ~12ms | ~320KB | ~3× slower |
| `from_terms_with_values()` | ~13ms | ~320KB | ~3× slower |

**Memory usage**:

```
Small (1K terms):     ~40KB  (vs ~30KB DynamicDawg)
Medium (10K terms):   ~320KB (vs ~250KB DynamicDawg)
Large (100K terms):   ~3.2MB (vs ~2.5MB DynamicDawg)
```

**Trade-offs**:
- **Simpler API**: Easier to use, less boilerplate
- **Slower**: 2-3× slower than DynamicDawg for bulk operations
- **More memory**: ~30% higher memory footprint
- **Good enough**: For < 10K terms, difference is negligible

### Best Practices

**1. Choose PathMapDictionary for simplicity:**
```rust
// ✅ Good: Prototyping, small dictionaries
let dict = PathMapDictionary::from_terms(vec!["test", "demo"]);

// ⚠️ Consider DynamicDawg: Large dictionaries, performance-critical
let dict = DynamicDawg::from_iter(large_term_list);  // Faster
```

**2. Use with contextual completion engine:**
```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;

// PathMapDictionary is the DEFAULT backend
let engine = DynamicContextualCompletionEngine::new();  // Uses PathMapDictionary

// Or explicit construction
let dict: PathMapDictionary<Vec<u32>> = PathMapDictionary::from_terms_with_values(terms);
let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);
```

**3. Pre-build for workspace indexing:**
```rust
use rayon::prelude::*;

// Build per-document dictionaries in parallel
let dicts: Vec<PathMapDictionary<Vec<u32>>> = documents
    .par_iter()
    .map(|(ctx_id, doc)| {
        let terms: Vec<(String, Vec<u32>)> = extract_terms(doc)
            .into_iter()
            .map(|term| (term, vec![*ctx_id]))
            .collect();

        PathMapDictionary::from_terms_with_values(terms)
    })
    .collect();

// Merge using union_with (see Union Operations section)
```

→ See [Parallel Workspace Indexing](../../07-contextual-completion/patterns/parallel-workspace-indexing.md) for complete pattern.

### Comparison with Other Dictionaries

**When to choose PathMapDictionary:**

| Factor | PathMapDictionary | DynamicDawg | DoubleArrayTrie |
|--------|------------------|-------------|-----------------|
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Speed** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dynamic updates** | ✅ Full | ✅ Full | ⚠️ Append-only |
| **Learning curve** | ✅ Minimal | Medium | High |
| **Use case** | Prototyping | Production | Performance |

**Decision guide:**

```
Start with: PathMapDictionary
  ↓
  If performance matters → Switch to DynamicDawg (~3× faster)
  ↓
  If static dictionary → Switch to DoubleArrayTrie (~12× faster)
```

### Parallel Construction

PathMapDictionary supports the same parallel construction pattern as DynamicDawg:

```rust
use rayon::prelude::*;

// Build dictionaries in parallel
let dicts: Vec<PathMapDictionary<Vec<u32>>> = documents
    .par_iter()
    .map(|(ctx_id, doc)| {
        let terms_with_contexts: Vec<_> = extract_terms(doc)
            .into_iter()
            .map(|term| (term, vec![*ctx_id]))
            .collect();

        PathMapDictionary::from_terms_with_values(terms_with_contexts)
    })
    .collect();

// Binary tree merge (see Parallel Workspace Indexing guide)
let merged = merge_tree_parallel(dicts);

// Create engine
let engine = DynamicContextualCompletionEngine::with_dictionary(
    merged,
    Algorithm::Standard
);
```

**Performance note**: Parallel construction still beneficial despite slower per-dictionary speed - wall-clock time scales with available CPU cores.

## Accessor Methods

PathMapDictionary provides the same core accessor methods as other dictionary backends, with simplicity as the primary design goal.

**→ See**: [DynamicDawg Accessor Methods](dynamic-dawg.md#accessor-methods) for comprehensive documentation.

### Key Differences from DynamicDawg

PathMapDictionary accessor methods have **simpler** implementations but **slower** performance:

| Method | PathMapDictionary | DynamicDawg | Performance Impact |
|--------|-------------------|-------------|---------------------|
| `contains(term)` | `$\mathcal{O}(m \cdot \log k)$` | `$\mathcal{O}(m)$` | ~2-3× slower |
| `get_value(term)` | `$\mathcal{O}(m \cdot \log k)$` | `$\mathcal{O}(m)$` | ~2-3× slower |
| `term_count()` | `$\mathcal{O}(1)$` | `$\mathcal{O}(1)$` | Similar |
| `len()` / `is_empty()` | `$\mathcal{O}(1)$` | `$\mathcal{O}(1)$` | Similar |

*Where*: `m` = term length, `k` = average branching factor (~26 for English)

### Quick Reference

```rust
use libdictenstein::pathmap::PathMapDictionary;

let dict = PathMapDictionary::from_terms(vec!["test", "testing", "tested"]);

// Term existence (slower than DynamicDawg, simpler code)
assert!(dict.contains("test"));
assert!(dict.contains("testing"));
assert!(!dict.contains("unknown"));

// Value retrieval
let dict_valued: PathMapDictionary<u32> = PathMapDictionary::new();
dict_valued.insert_with_value("key", 42);
assert_eq!(dict_valued.get_value("key"), Some(42));

// Size queries (O(1), same as Dynamic Dawg)
assert_eq!(dict.term_count(), 3);
assert_eq!(dict.len(), Some(3));
assert!(!dict.is_empty());

// No compaction needed (persistent structure doesn't fragment)
// No node_count() method (implementation detail differs)
// No needs_compaction() (not applicable to PathMap)

// Traversal (via Dictionary trait)
use libdictenstein::{Dictionary, DictionaryNode};
let root = dict.root();
// ... navigate via transition() as with other backends
```

### Performance Characteristics

**Accessor Latencies** (10K term dictionary):

| Method | PathMapDictionary | DynamicDawg | PathMap/DynamicDawg Ratio |
|--------|-------------------|-------------|---------------------------|
| `contains()` | ~700ns | ~250ns | 2.8× slower |
| `get_value()` | ~750ns | ~260ns | 2.9× slower |
| `term_count()` | ~5ns | ~5ns | Same |
| `len()` / `is_empty()` | ~5ns | ~5ns | Same |

**Why slower?**:
- PathMap uses **tree traversal** with log(k) comparisons per level
- DynamicDawg uses **direct indexing** via edge lookup

**Trade-off**: Simplicity and persistent semantics vs performance.

### Persistent Semantics

PathMapDictionary accessor methods benefit from **structural sharing**:

```rust
let dict1 = PathMapDictionary::from_terms(vec!["test", "testing"]);
let dict2 = dict1.clone(); // Shallow clone (Arc increment)

// Both share same underlying structure
assert!(dict1.contains("test"));
assert!(dict2.contains("test"));

// Modifications create new structure (copy-on-write)
dict2.insert("new_term");
assert!(!dict1.contains("new_term")); // Original unchanged
assert!(dict2.contains("new_term"));  // New version has it

// Accessor methods see correct version
assert_eq!(dict1.term_count(), 2);
assert_eq!(dict2.term_count(), 3);
```

### Thread Safety

PathMapDictionary accessors are thread-safe via Arc-based sharing:

```rust
use std::sync::Arc;
use std::thread;

let dict = Arc::new(PathMapDictionary::from_terms(vec!["hello", "world"]));

// Concurrent reads safe
let handles: Vec<_> = (0..10)
    .map(|_| {
        let d = Arc::clone(&dict);
        thread::spawn(move || d.contains("hello"))
    })
    .collect();

for h in handles {
    assert!(h.join().unwrap());
}

// Mutations create new versions (no locks needed)
let dict2 = Arc::new((*dict).clone());
dict2.insert("new");
// Original dict unchanged, dict2 has new term
```

---

## Union Operations

### Overview

The `union_with()` and `union_replace()` methods enable **merging two PathMapDictionary instances** with custom value combination logic, while preserving **structural sharing** properties of the persistent trie. Essential for:

- 🔄 Merging configuration layers (defaults + user overrides)
- 📊 Combining statistics from independent data sources
- 🗂️ Building composite lookup tables
- 💾 Creating snapshots with incremental updates

**Key Characteristics**:
- 🔒 **Thread-safe**: Writers publish a new snapshot via atomic swap; reads stay lock-free
- 🌳 **Structural sharing**: Leverages PathMap's persistent data structure benefits
- ⚡ **Iterator-based**: Uses PathMap's efficient iteration over key-value pairs
- 🎯 **Flexible**: Custom merge functions for value conflicts
- 🔧 **Simple**: Straightforward implementation via iteration + insertion

### union_with() - Merge with Custom Logic

Combines two dictionaries by iterating all terms from the source dictionary and inserting into the target, applying a custom merge function when values conflict.

**Signature**:
```rust
fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
where
    F: Fn(&Self::Value, &Self::Value) -> Self::Value,
    Self::Value: Clone
```

**Parameters**:
- `other`: Source dictionary to merge from
- `merge_fn`: Function `(existing_value, new_value) -> merged_value` for conflicts
- **Returns**: Number of terms processed from `other`

**Algorithm**: Iteration-based insertion
1. Load an immutable snapshot of `other`'s state (lock-free)
2. Clone `self`'s current `PathMapState` to build the next revision off to the side
3. Iterate all `(key, value)` pairs in `other`
4. For each pair:
   - If key exists in the working map: Apply `merge_fn` and update
   - If key is new: Insert with cloned value
5. Publish the updated `PathMapState` (map + term count) with an atomic compare-and-swap (retry on contention)

**Complexity**:
- **Time**: `$\mathcal{O}(n \cdot \log m)$` where n = terms in `other`, m = terms in `self`
  - `$\mathcal{O}(n)$` for iteration over `other`
  - `$\mathcal{O}(\log m)$` per PathMap insertion/lookup
- **Space**: `$\mathcal{O}(\log m)$` for PathMap tree height (structural sharing reduces actual allocation)

### Why Iteration Instead of PathMap's join()?

PathMap provides native `join_into()` and `pjoin()` methods, but they require `V: Lattice`:

```rust
// PathMap native (requires Lattice trait)
pub fn join_into<V: Lattice>(&mut self, other: &PathMap<V>) { ... }
```

**Limitation**: The `Lattice` trait requires specific algebraic properties:
- Commutative: `$a \sqcup b = b \sqcup a$`
- Associative: `$(a \sqcup b) \sqcup c = a \sqcup (b \sqcup c)$`
- Idempotent: `$a \sqcup a = a$`

**Our approach**: Uses **arbitrary merge functions** without algebraic constraints:
- ✅ Supports non-commutative merges: `(old, new) → new` (last-writer-wins)
- ✅ Supports non-idempotent merges: `(a, b) → a + b` (sum aggregation)
- ✅ Flexible merge logic: Any `Fn(&V, &V) -> V`

**Trade-off**: Slightly slower (~15-20% overhead) but far more flexible.

### Example 1: Sum Aggregation

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MutableMappedDictionary;

// First dataset: term frequencies
let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
dict1.insert_with_value("algorithm", 10);
dict1.insert_with_value("database", 5);

// Second dataset: more frequencies
let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
dict2.insert_with_value("algorithm", 7);    // Overlap
dict2.insert_with_value("distributed", 3);  // New

// Merge by summing counts
let processed = dict1.union_with(&dict2, |left, right| left + right);

// Results:
// - algorithm: 17 (10 + 7)
// - database: 5 (unchanged)
// - distributed: 3 (new)
assert_eq!(dict1.get_value("algorithm"), Some(17));
assert_eq!(dict1.get_value("distributed"), Some(3));
assert_eq!(processed, 2);
```

### Example 2: Configuration Merging

Demonstrates typical use case of layering configurations:

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MutableMappedDictionary;

// System defaults
let defaults: PathMapDictionary<String> = PathMapDictionary::new();
defaults.insert_with_value("theme", "light".to_string());
defaults.insert_with_value("font_size", "12".to_string());
defaults.insert_with_value("autosave", "true".to_string());

// User preferences
let user_prefs: PathMapDictionary<String> = PathMapDictionary::new();
user_prefs.insert_with_value("theme", "dark".to_string());  // Override
user_prefs.insert_with_value("language", "en".to_string()); // New

// Merge: user preferences override defaults
defaults.union_with(&user_prefs, |_default, user| user.clone());

// Results:
// - theme: "dark" (user override)
// - font_size: "12" (default preserved)
// - autosave: "true" (default preserved)
// - language: "en" (new from user)
assert_eq!(defaults.get_value("theme"), Some("dark".to_string()));
assert_eq!(defaults.get_value("font_size"), Some("12".to_string()));
```

### Example 3: Set Union with Lists

Merge lists of associated data:

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MutableMappedDictionary;

let dict1: PathMapDictionary<Vec<u32>> = PathMapDictionary::new();
dict1.insert_with_value("rust", vec![1, 2, 3]);
dict1.insert_with_value("python", vec![4]);

let dict2: PathMapDictionary<Vec<u32>> = PathMapDictionary::new();
dict2.insert_with_value("rust", vec![2, 3, 5]);  // Overlapping values
dict2.insert_with_value("golang", vec![6, 7]);

// Merge by concatenating and deduplicating
dict1.union_with(&dict2, |left, right| {
    let mut merged = left.clone();
    merged.extend(right.clone());
    merged.sort_unstable();
    merged.dedup();
    merged
});

// rust: [1,2,3,5] (merged and deduplicated)
// python: [4] (unchanged)
// golang: [6,7] (new)
assert_eq!(dict1.get_value("rust"), Some(vec![1, 2, 3, 5]));
```

### union_replace() - Keep Right Values

Convenience method for last-writer-wins semantics.

**Signature**:
```rust
fn union_replace(&self, other: &Self) -> usize
where
    Self::Value: Clone
```

**Example**:
```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MutableMappedDictionary;

let dict1: PathMapDictionary<&str> = PathMapDictionary::new();
dict1.insert_with_value("status", "draft");
dict1.insert_with_value("version", "1.0");

let dict2: PathMapDictionary<&str> = PathMapDictionary::new();
dict2.insert_with_value("status", "published");  // Override
dict2.insert_with_value("author", "alice");      // New

// Simple replacement
dict1.union_replace(&dict2);

assert_eq!(dict1.get_value("status"), Some("published"));
assert_eq!(dict1.get_value("version"), Some("1.0"));
assert_eq!(dict1.get_value("author"), Some("alice"));
```

### Implementation Details

The union operation uses **PathMap's iterator** with lock-free snapshot publication:

```rust
// Simplified implementation (lock-free publish via compare-and-swap)
fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize {
    let other_state = other.load_state();   // immutable snapshot of `other`
    let mut backoff = CasBackoff::new();

    loop {
        let current = self.load_state();
        let mut next_map = current.map.clone();  // persistent clone (structural sharing)
        let mut next_len = current.len;
        let mut processed = 0;

        // Iterate over all entries in other's snapshot
        for (key_bytes, other_value) in other_state.map.iter() {
            processed += 1;

            if let Some(self_value) = next_map.get(&key_bytes) {
                // Key exists: merge the values
                let merged = merge_fn(self_value, other_value);
                next_map.insert(&key_bytes, merged);
            } else {
                // Key doesn't exist: insert from other
                next_map.insert(&key_bytes, other_value.clone());
                next_len += 1;
            }
        }

        // Publish atomically; retry if another writer won the race
        if self.compare_store_state(&current, PathMapState::new(next_map, next_len)) {
            return processed;
        }
        backoff.snooze();
    }
}
```

**Why This Approach?**

1. **Simplicity**: Leverages PathMap's well-tested iterator
2. **Flexibility**: No trait constraints on value types
3. **Correctness**: The compare-and-swap publish makes each union atomic to readers
4. **Structural sharing**: PathMap automatically shares structure between old and new versions

**Publish Semantics**:
- Snapshot read of `other`: Lock-free; never blocks
- Atomic publish to `self`: A new `PathMapState` is installed with compare-and-swap
- Single transaction: Readers see either the pre-union or post-union snapshot, never a partial state

### Performance Characteristics

| Operation | Time Complexity | Space Complexity | Typical Performance (10K terms) |
|-----------|----------------|------------------|--------------------------------|
| `union_with()` | `$\mathcal{O}(n \cdot \log m)$` | `$\mathcal{O}(\log m)$` | ~80ms |
| `union_replace()` | `$\mathcal{O}(n \cdot \log m)$` | `$\mathcal{O}(\log m)$` | ~80ms |
| Iteration | `$\mathcal{O}(n)$` | `$\mathcal{O}(1)$` | ~15ms |
| Per-term insertion | `$\mathcal{O}(\log m)$` | `$\mathcal{O}(\log m)$` | ~5-8µs |

**Variables**:
- n = number of terms in source dictionary
- m = number of terms in target dictionary
- log m = PathMap tree height (typically 5-10 levels)

**Comparison with DynamicDawg**:
```
PathMapDictionary: ~80ms for 10K terms (O(n·log m))
DynamicDawg:       ~50ms for 10K terms (O(n·m))

Reason: PathMap insertion is O(log m) vs DAWG's O(m)
Trade-off: PathMap offers structural sharing and immutability
```

**Benchmark Results** (Intel Xeon E5-2699 v3 @ 2.30GHz):

| Dictionary Size | union_with() | Throughput |
|----------------|-------------|------------|
| 1,000 terms    | 6.8ms       | 147K terms/s |
| 10,000 terms   | 80ms        | 125K terms/s |
| 100,000 terms  | 950ms       | 105K terms/s |

*Note*: Performance includes merge function execution and structural sharing overhead.

### When to Use Union Operations

✅ **Use `union_with()` when:**
- **Parallel workspace indexing**: Merging per-document dictionaries built in parallel (→ [Parallel Workspace Pattern](../../07-contextual-completion/patterns/parallel-workspace-indexing.md))
- Merging configuration layers with override semantics
- Combining statistics where structural sharing is beneficial
- Building composite lookup tables from multiple sources
- Aggregating data where immutability is valuable

✅ **Use `union_replace()` when:**
- Applying updates with last-writer-wins semantics
- Synchronizing dictionaries where newer data always wins
- Implementing configuration hot-reloading

⚠️ **Consider DynamicDawg when:**
- Union performance is critical (40% faster)
- Structural sharing not needed
- Frequent mutations expected

⚠️ **Consider alternatives when:**
- **Very large dictionaries**: Pre-merge offline or use batch processing
- **Frequent unions**: Consider maintaining separate indices
- **Simple addition**: If only adding new terms (no conflicts), use simple iteration

### Structural Sharing Considerations

PathMapDictionary's persistent nature means union operations benefit from structural sharing:

```rust
let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
// Insert 100,000 terms...

let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
// Insert 100 terms (mostly new)...

// Union creates new version sharing structure with dict1
dict1.union_with(&dict2, |a, b| a + b);

// Memory overhead: Only ~100 new nodes created
// Most of dict1's structure is reused via structural sharing
```

**Benefits**:
- 💾 **Memory efficient**: Only delta nodes allocated
- 🔒 **Safe snapshots**: Old version still accessible
- 🚀 **Fast clones**: `$\mathcal{O}(1)$` shallow copy of Arc

**Caveats**:
- Lock contention on write during union
- No direct zipper-based traversal (unlike DynamicDawg)
- Iterator overhead vs direct node manipulation

## Usage Examples

### Example 1: Basic Usage

```rust
use libdictenstein::pathmap::PathMapDictionary;

// Create empty dictionary
let dict: PathMapDictionary<()> = PathMapDictionary::new();

// Insert terms
dict.insert("test");
dict.insert("testing");
dict.insert("tested");

assert!(dict.contains("test"));
assert_eq!(dict.len(), Some(3));

// Remove term
dict.remove("tested");
assert!(!dict.contains("tested"));
assert_eq!(dict.len(), Some(2));
```

### Example 2: From Existing Terms

```rust
use libdictenstein::pathmap::PathMapDictionary;

let dict = PathMapDictionary::from_terms(vec![
    "algorithm",
    "approximate",
    "automaton",
]);

assert!(dict.contains("algorithm"));
assert_eq!(dict.len(), Some(3));

// Add more terms
dict.insert("analysis");
assert_eq!(dict.len(), Some(4));
```

### Example 3: With Values

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MappedDictionary;

// Map terms to category IDs
let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
    ("test", 1),
    ("testing", 1),
    ("production", 2),
]);

// Query values
assert_eq!(dict.get_value("test"), Some(1));
assert_eq!(dict.get_value("production"), Some(2));

// Update value
dict.insert_with_value("test", 99);
assert_eq!(dict.get_value("test"), Some(99));
```

### Example 4: Fuzzy Search

```rust
use libdictenstein::pathmap::PathMapDictionary;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = PathMapDictionary::from_terms(vec![
    "test", "testing", "tested", "best", "rest"
]);

// Fuzzy search
let automaton = LevenshteinAutomaton::new("tset", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["test"] (distance 1: transposition)
```

### Example 5: Thread-Safe Updates

```rust
use libdictenstein::pathmap::PathMapDictionary;
use std::sync::Arc;
use std::thread;

let dict = Arc::new(PathMapDictionary::from_terms(vec!["initial"]));

// Spawn writer thread
let dict_writer = Arc::clone(&dict);
let writer = thread::spawn(move || {
    dict_writer.insert("new_term");
});

// Spawn reader threads
let handles: Vec<_> = (0..4).map(|_| {
    let dict_reader = Arc::clone(&dict);
    thread::spawn(move || {
        dict_reader.contains("initial")
    })
}).collect();

writer.join().unwrap();
for handle in handles {
    assert!(handle.join().unwrap());
}
```

### Example 6: Dynamic User Dictionary

```rust
use libdictenstein::pathmap::PathMapDictionary;

// User's personal dictionary
let user_dict = PathMapDictionary::new();

// User adds custom words
user_dict.insert("refactoring");
user_dict.insert("debugging");
user_dict.insert("profiling");

assert_eq!(user_dict.len(), Some(3));

// User removes a word
user_dict.remove("debugging");
assert_eq!(user_dict.len(), Some(2));

// Check existence
assert!(user_dict.contains("refactoring"));
assert!(!user_dict.contains("debugging"));
```

### Example 7: Metadata Storage

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MappedDictionary;

#[derive(Clone, Debug)]
struct TermMetadata {
    frequency: u32,
    last_used: u64,
}

impl libdictenstein::DictionaryValue for TermMetadata {}

let dict: PathMapDictionary<TermMetadata> = PathMapDictionary::new();

// Add terms with metadata
dict.insert_with_value("test", TermMetadata {
    frequency: 100,
    last_used: 1234567890,
});

dict.insert_with_value("testing", TermMetadata {
    frequency: 50,
    last_used: 1234567891,
});

// Query metadata
if let Some(meta) = dict.get_value("test") {
    println!("Frequency: {}", meta.frequency);
}
```

### Example 8: Prototyping

```rust
use libdictenstein::pathmap::PathMapDictionary;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

// Quick prototype for fuzzy matching
fn prototype_fuzzy_matcher(words: Vec<&str>, query: &str) {
    let dict = PathMapDictionary::from_terms(words);

    let automaton = LevenshteinAutomaton::new(query, 2, Algorithm::Standard);
    let results: Vec<String> = automaton.query(&dict).collect();

    println!("Matches for '{}': {:?}", query, results);
}

prototype_fuzzy_matcher(
    vec!["hello", "world", "test"],
    "helo"  // Typo
);
// Output: Matches for 'helo': ["hello"]
```

## Performance Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Insert** | `$\mathcal{O}(m \log n)$` | m = term length, n = dict size |
| **Remove** | `$\mathcal{O}(m \log n)$` | HashMap operations |
| **Contains** | `$\mathcal{O}(m \log n)$` | Tree traversal + lookups |
| **Fuzzy search** | `$\mathcal{O}(m \times d^{2} \times b \times \log n)$` | Additional log factor |

### Benchmark Results

#### Construction

```
Build from 10,000 terms:
  PathMapDictionary:  3.5ms
  DoubleArrayTrie:    3.2ms   (8% faster)
  DynamicDawg:        4.1ms   (15% slower)
```

#### Runtime Operations

```
Single insertion:
  PathMapDictionary:  ~2.1µs
  DynamicDawg:        ~800ns  (2.6x faster)
  DoubleArrayTrie:    N/A (append-only)

Single deletion:
  PathMapDictionary:  ~2.5µs
  DynamicDawg:        ~1.2µs  (2x faster)

Contains check:
  PathMapDictionary:  ~350ns
  DoubleArrayTrie:    ~120ns  (2.9x faster)
  DynamicDawg:        ~450ns  (slower)
```

#### Fuzzy Search

```
Query "test" (distance 1) in 10K-term dict:
  PathMapDictionary:  38.7µs
  DoubleArrayTrie:    12.9µs  (3x faster)
  DynamicDawg:        42.3µs  (similar)

Query "test" (distance 2):
  PathMapDictionary:  91.2µs
  DoubleArrayTrie:    16.3µs  (5.6x faster)
  DynamicDawg:        68.9µs  (1.3x faster)
```

### Memory Usage

```
10,000-term dictionary:
  PathMapDictionary:  ~320 KB
  DoubleArrayTrie:    ~100 KB  (3.2x smaller)
  DynamicDawg:        ~294 KB  (similar)

Memory overhead:
  PathMapDictionary:  ~32 bytes/node (HashMap)
  DoubleArrayTrie:    ~10 bytes/state
  DynamicDawg:        ~25 bytes/node
```

### Comparison Summary

```
                    Construction  Memory   Contains  Fuzzy(d=2)  Insert  Remove
─────────────────────────────────────────────────────────────────────────────────
PathMapDictionary   3.5ms        320KB    350ns     91.2µs      2.1µs   2.5µs
DoubleArrayTrie     3.2ms        100KB    120ns     16.3µs      N/A     N/A
DynamicDawg         4.1ms        294KB    450ns     68.9µs      800ns   1.2µs
```

**Verdict**: PathMapDictionary is 2-3x slower than optimized alternatives, but provides simplicity and full dynamic updates.

## When to Use

### Decision Matrix

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| **Prototyping** | ✅ PathMapDictionary | Quick to use |
| **Simple applications** | ✅ PathMapDictionary | Easy API |
| **Maximum performance** | ⚠️ DoubleArrayTrie | 3x faster |
| **Memory-constrained** | ⚠️ DoubleArrayTrie | 3x smaller |
| **Dynamic + fast** | ⚠️ DynamicDawg | 2x faster updates |

### Ideal Use Cases

1. **Prototyping**
   - Quick experiments
   - Proof of concept
   - Algorithm validation

2. **Small Dictionaries**
   - <1000 terms
   - Performance not critical
   - Simplicity valued

3. **Educational/Learning**
   - Understanding fuzzy matching
   - Teaching examples
   - Simple demonstrations

4. **Low-Traffic Applications**
   - Infrequent queries
   - Small user base
   - Development/testing

### When to Migrate Away

Consider switching to specialized dictionaries when:

✅ **DoubleArrayTrie** if:
- Query performance becomes bottleneck
- Dictionary becomes mostly static
- Memory usage is concern

✅ **DynamicDawg** if:
- Frequent updates needed
- Better update performance required
- Still need full dynamic capabilities

## Related Documentation

- [Dictionary Layer](../README.md) - Overview of all dictionary types
- [DoubleArrayTrie](double-array-trie.md) - Faster alternative
- [DynamicDawg](dynamic-dawg.md) - Faster dynamic alternative
- PathMapDictionaryChar - Unicode variant
- [Value Storage](../../09-value-storage/README.md) - Using values

## References

### PathMap Library

1. **PathMap Repository**
   - 📦 [https://github.com/Adam-Vandervorst/PathMap](https://github.com/Adam-Vandervorst/PathMap)
   - Underlying persistent trie implementation

### Persistent Data Structures

2. **Okasaki, C. (1999)**. *Purely Functional Data Structures*
   - Cambridge University Press
   - ISBN: 978-0521663502
   - 📚 Comprehensive coverage of persistent structures

3. **Driscoll, J. R., Sarnak, N., Sleator, D. D., & Tarjan, R. E. (1989)**. "Making data structures persistent"
   - *Journal of Computer and System Sciences*, 38(1), 86-124
   - DOI: [10.1016/0022-0000(89)90034-2](https://doi.org/10.1016/0022-0000(89)90034-2)
   - 📄 Foundational paper on persistence

### Trie Structures

4. **Fredkin, E. (1960)**. "Trie memory"
   - *Communications of the ACM*, 3(9), 490-499
   - DOI: [10.1145/367390.367400](https://doi.org/10.1145/367390.367400)
   - 📄 Original trie paper

## Next Steps

- **Performance**: Compare with [DoubleArrayTrie](double-array-trie.md)
- **Dynamic**: Explore [DynamicDawg](dynamic-dawg.md)
- **Unicode**: Check PathMapDictionaryChar
- **Values**: Learn about [Value Storage](../../09-value-storage/README.md)

---

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)
