# DynamicDawg Implementation

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory: DAWG Structure](#theory-dawg-structure)
3. [Dynamic Modifications](#dynamic-modifications)
4. [Data Structure](#data-structure)
5. [Construction Methods](#construction-methods)
6. [Accessor Methods](#accessor-methods)
7. [Key Algorithms](#key-algorithms)
8. [Union Operations](#union-operations)
9. [Usage Examples](#usage-examples)
10. [Performance Analysis](#performance-analysis)
11. [When to Use](#when-to-use)
12. [References](#references)

## Overview

`DynamicDawg` is a **Directed Acyclic Word Graph** that supports **runtime insertions and deletions** while maintaining thread-safe access. Unlike static DAWG implementations, DynamicDawg allows the dictionary to evolve during application lifetime.

### Key Advantages

- 🔄 **Full dynamic updates**: Insert AND remove terms at runtime
- 🔒 **Thread-safe (lock-free)**: Concurrent reads never block; writers publish via CAS
- 💾 **Space-efficient**: Shares common suffixes (20-40% reduction)
- ⚡ **Good performance**: Suitable for dictionaries with frequent updates
- 📊 **Reference counting**: Safe deletion without orphaning nodes

### When to Use

✅ **Use DynamicDawg when:**
- Dictionary changes frequently (adds and removes)
- Need thread-safe concurrent access
- Building dynamic word lists (user dictionaries, session-specific terms)
- Real-time collaborative applications

⚠️ **Consider alternatives when:**
- Dictionary is static or append-only → Use `DoubleArrayTrie` (3x faster)
- Need maximum query performance → Use `DoubleArrayTrie`
- Working with Unicode → Use `DynamicDawgChar`

## Theory: DAWG Structure

### What is a DAWG?

A **Directed Acyclic Word Graph** is a compressed trie that shares common suffixes, not just prefixes.

**Example**: Terms ["car", "card", "cart", "star", "start"]

```
Regular Trie (prefix sharing only):
       (root)
       /    \
      c      s
      |      |
      a      t
      |      |
      r      a
     / \     |
    d   t    r
            / \
           t   (nothing - "star")

DAWG (prefix AND suffix sharing):
       (root)
       /    \
      c      s
      |      |
      a      t
      |      |
      r ─────┘  ← Shares "ar" suffix
     / \
    d   t
```

**Space savings**: DAWG nodes = ~50-70% of trie nodes for natural language.

### Suffix Sharing

Multiple prefixes can point to the same suffix:
```
"card" = c→a→r→d(final)
"cart" = c→a→r→t(final)
"hard" = h→a→r→d(final)  ← Shares "r→d" with "card"
"hart" = h→a→r→t(final)  ← Shares "r→t" with "cart"
```

This is achieved by **hashing node signatures** and reusing nodes with identical right languages.

## Dynamic Modifications

### Insertion Algorithm

Adding a term while maintaining minimality:

```rust
fn insert(&self, term: &str) {
    let mut lock = self.inner.write();  // simplified pseudocode — the real backend is
                                        // lock-free: it mutates a cloned snapshot and
                                        // publishes it via CAS (see "Thread Safety" below)

    // Traverse existing path
    let mut node_idx = 0;  // Root
    let mut path = Vec::new();

    for byte in term.bytes() {
        path.push(node_idx);

        // Find or create edge
        node_idx = match lock.find_edge(node_idx, byte) {
            Some(child_idx) => child_idx,
            None => {
                // Create new suffix
                let new_suffix = lock.create_suffix(&term[pos..]);
                lock.add_edge(node_idx, byte, new_suffix);
                return;
            }
        };
    }

    // Mark final
    lock.nodes[node_idx].is_final = true;
}
```

**Complexity**: $`\mathcal{O}(m)`$ where m = term length

### Deletion Algorithm

Removing a term requires reference counting:

```rust
fn remove(&self, term: &str) -> bool {
    let mut lock = self.inner.write();

    // Traverse to term
    let mut node_idx = 0;
    let mut path = Vec::new();

    for byte in term.bytes() {
        path.push(node_idx);
        node_idx = lock.find_edge(node_idx, byte)?;
    }

    if !lock.nodes[node_idx].is_final {
        return false;  // Term not in dictionary
    }

    // Mark as non-final
    lock.nodes[node_idx].is_final = false;

    // Decrement reference counts along path
    for &idx in path.iter().rev() {
        lock.nodes[idx].ref_count -= 1;

        // Delete node if no longer referenced
        if lock.nodes[idx].ref_count == 0 && !lock.nodes[idx].is_final {
            lock.delete_node(idx);
        } else {
            break;  // Still in use
        }
    }

    lock.needs_compaction = true;
    true
}
```

**Complexity**: $`\mathcal{O}(m)`$

### Compaction

Over time, deletions create orphaned branches. Compaction restores minimality:

```rust
pub fn compact(&self) {
    let mut lock = self.inner.write();

    if !lock.needs_compaction {
        return;
    }

    // Rebuild suffix cache
    lock.suffix_cache.clear();
    lock.rebuild_suffix_cache();

    // Merge equivalent nodes
    lock.merge_equivalent_nodes();

    lock.needs_compaction = false;
}
```

**Complexity**: $`\mathcal{O}(n)`$ where n = total nodes

**When to compact**:
- After many deletions (10%+ of dictionary removed)
- When query performance degrades
- During maintenance windows

## Data Structure

### Core Components

```rust
pub struct DynamicDawg<V: DictionaryValue = ()> {
    inner: Arc<DynamicDawgInner<V>>,
}

// `DynamicDawgInner` is the unit-generic lock-free DAWG core,
// `LockFreeDawg<u8, V>`. Published nodes are immutable. Readers retain one
// root revision; writers path-copy an affected route and publish the new
// GraphVersion through one ArcSwap compare_exchange (CAS) loop.
type DynamicDawgInner<V = ()> = LockFreeDawg<u8, V>;
```

### Memory Layout

```
┌─────────────────┬─────────────┬────────────────┐
│ Component       │ Size        │ Per Node       │
├─────────────────┼─────────────┼────────────────┤
│ SmallVec edges  │ Inline ≤4   │ ~16 bytes      │
│ is_final        │ 1 byte      │ 1 byte         │
│ ref_count       │ 8 bytes     │ 8 bytes        │
│ value (Option)  │ V or 1 byte │ Varies         │
├─────────────────┼─────────────┼────────────────┤
│ Total per node  │ ~25+ bytes  │ ~25 bytes      │
│ Overhead        │ Arc         │ 8 bytes total  │
└─────────────────┴─────────────┴────────────────┘
```

**Example**: 10,000-term dictionary $`\approx`$ 250KB (nodes) + 32KB (suffix cache)

### Clone Behavior & Memory Semantics

`DynamicDawg` uses `Arc<...>` internally (the lock-free `LockFreeDawg` core), making `.clone()` a **shallow copy** that shares all underlying data structures between clones:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict1 = DynamicDawg::from_iter(vec!["test", "testing"]);
let dict2 = dict1.clone();  // O(1) - only increments Arc refcount

// Both dict1 and dict2 point to the SAME underlying data
dict1.insert("new_term");
assert!(dict2.contains("new_term"));  // ✅ Mutations visible through dict2!

// Term count reflects changes made via either clone
assert_eq!(dict1.len(), Some(3));
assert_eq!(dict2.len(), Some(3));  // Same count
```

#### Characteristics

| Property | Behavior | Impact |
|----------|----------|--------|
| **Time Complexity** | $`\mathcal{O}(1)`$ | Single atomic increment |
| **Space Complexity** | $`\mathcal{O}(1)`$ | ~16 bytes (Arc pointer only) |
| **Data Sharing** | ✅ Complete | All clones share same node graph |
| **Mutation Visibility** | ✅ Global | Changes via any clone affect all |
| **Thread Safety** | ✅ Lock-free | Readers never block; writers publish via CAS |
| **Independence** | ❌ None | No isolation between clones |

#### How Clone Works

The clone operation only increments an atomic reference counter:

```rust
pub struct DynamicDawg<V> {
    inner: Arc<DynamicDawgInner<V>>,  // ← Single Arc
}

// Cloning increments Arc's atomic refcount
let dict2 = dict1.clone();
// Equivalent to: Arc::clone(&dict1.inner)
// Cost: ~1-2 CPU cycles (atomic increment)
```

**What gets cloned:**
- ✅ Arc smart pointer (~16 bytes on stack)
- ❌ NOT the lock-free core (shared, never copied)
- ❌ NOT the node graph
- ❌ NOT the suffix cache or bloom filter
- ❌ NOT any internal structures

**Memory allocation:**
- Zero heap allocation
- Only stack space for new Arc pointer
- All data remains shared

#### When to Use Cloning

✅ **Good use cases:**

1. **Multi-threaded access** - Share across threads:
   ```rust
   use std::thread;

   let dict = DynamicDawg::from_iter(vec!["hello", "world"]);

   let handles: Vec<_> = (0..4).map(|_| {
       let dict_clone = dict.clone();  // Cheap clone for each thread
       thread::spawn(move || {
           // Each thread can read concurrently
           dict_clone.contains("hello")
       })
   }).collect();
   ```

2. **Storing in multiple data structures:**
   ```rust
   let mut map1 = HashMap::new();
   let mut map2 = HashMap::new();

   let dict = DynamicDawg::from_iter(vec!["term1", "term2"]);
   map1.insert("key1", dict.clone());
   map2.insert("key2", dict.clone());  // Same underlying data
   ```

3. **Convenience aliases:**
   ```rust
   let system_dict = DynamicDawg::from_iter(vec!["system"]);
   let dict = system_dict.clone();  // Short alias
   ```

❌ **Bad use cases (common mistakes):**

1. **Expecting independent copies:**
   ```rust
   let dict1 = DynamicDawg::from_iter(vec!["original"]);
   let dict2 = dict1.clone();

   dict1.insert("modified");
   // ❌ WRONG: Expecting dict2 to still have only "original"
   // ✅ REALITY: dict2 also contains "modified"
   ```

2. **Avoiding mutation visibility:**
   ```rust
   let dict1 = build_dictionary();
   let dict2 = dict1.clone();  // ❌ Won't create independent copy

   modify_dictionary(&dict1);
   // dict2 sees all modifications - they share data!
   ```

3. **Creating snapshots:**
   ```rust
   let dict = DynamicDawg::from_iter(vec!["v1"]);
   let snapshot = dict.clone();  // ❌ NOT a snapshot!

   dict.insert("v2");
   // "snapshot" now also contains "v2" - not a true snapshot
   ```

#### Alternative: True Independence

If you need **independent copies** where modifications don't affect other instances, `clone()` is insufficient. Options include:

**Option 1: Serialize/Deserialize**
```rust
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};

// Create deep copy via serialization (`Vec<u8>: Write`, `&[u8]: Read`)
let mut bytes = Vec::new();
BincodeSerializer::serialize(&dict1, &mut bytes)?;
let dict2: DynamicDawg = BincodeSerializer::deserialize(&bytes[..])?;

// Now dict1 and dict2 are truly independent
dict1.insert("new");
assert!(!dict2.contains("new"));  // ✅ Independent
```

**Option 2: Rebuild from terms**
```rust
// Extract all terms
let terms: Vec<String> = dict1.iter().collect();

// Build new independent dictionary
let dict2 = DynamicDawg::from_iter(terms);

// dict2 is now completely independent
```

**Cost comparison:**

| Method | Time | Space | Independence |
|--------|------|-------|--------------|
| `.clone()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | ❌ Shared |
| Serialize/Deserialize | $`\mathcal{O}(n)`$ | $`\mathcal{O}(n)`$ | ✅ Full |
| Rebuild from terms | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(n)`$ | ✅ Full |

#### Comparison with Other Dictionaries

Different dictionary implementations have different clone semantics:

| Dictionary | Clone Type | Cost | Shared Data? |
|------------|------------|------|--------------|
| **DynamicDawg** | Shallow (Arc) | $`\mathcal{O}(1)`$ | ✅ Yes |
| **DynamicDawgChar** | Shallow (Arc) | $`\mathcal{O}(1)`$ | ✅ Yes |
| **PathMapDictionary** | Shallow (Arc) | $`\mathcal{O}(1)`$ | ✅ Yes |
| **DoubleArrayTrie** | Deep copy | $`\mathcal{O}(n)`$ | ❌ No |
| **DoubleArrayTrieChar** | Deep copy | $`\mathcal{O}(n)`$ | ❌ No |

**Why the difference?**
- **Mutable dictionaries** (DynamicDawg, PathMap) use Arc for shared ownership with interior mutability
- **Immutable dictionaries** (DoubleArrayTrie) don't use Arc, so clone creates full independent copies

#### Thread Safety Considerations

The Arc-based clone enables safe concurrent access patterns:

```rust
use std::sync::Arc;
use std::thread;

let dict = DynamicDawg::from_iter(vec!["concurrent", "access"]);

// Multiple concurrent readers (fast - no blocking)
let readers: Vec<_> = (0..10).map(|i| {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.contains(&format!("term{}", i))  // Many readers OK
    })
}).collect();

// Concurrent writer (lock-free; does not block readers)
let writer = {
    let dict = dict.clone();
    thread::spawn(move || {
        dict.insert("new_term")  // Lock-free insertion (CAS publication)
    })
};
```

**Concurrency semantics:**
- **Readers are wait-free** — each retains one immutable root revision and never blocks
- **Writers path-copy and publish a new root** via `compare_exchange` (CAS) loops
- Write operations: `insert()`, `remove()`, `union_with()`, `compact()`
- Read operations: `contains()`, `get_value()`, `len()`, iteration

**Performance impact (historical):**
- Earlier `RwLock`-based releases measured ~10-20ns read-lock and ~50-100ns write-lock overhead
- The backend is now lock-free: reads incur only an atomic `ArcSwap` load, and writes a CAS publication
- Concurrent readers therefore no longer serialize behind writers

#### Summary

**Key Takeaways:**
1. 🔗 `.clone()` creates a **shallow copy** - all clones share the same data
2. 🚀 **$`\mathcal{O}(1)`$** time and space - just increments atomic reference count
3. 🔄 **Mutations are visible** across all clones (by design)
4. 🔒 **Thread-safe and lock-free** — readers never block; writers publish via `compare_exchange` (CAS)
5. 📊 For **independence**, use serialization or rebuild from terms ($`\mathcal{O}(n)`$ cost)

### Optimizations

#### 1. SmallVec for Edges

Most nodes have $`\le 4`$ edges. `SmallVec` avoids heap allocation:

```rust
// Inline storage for ≤4 edges (stack allocated)
edges: SmallVec<[(u8, usize); 4]>

// Typical case: 2 edges → no heap allocation
// Rare case: >4 edges → heap allocation
```

**Impact**: 30-40% faster node access

#### 2. Suffix Cache

Hash node signatures to detect identical suffixes:

```rust
fn compute_signature(node: &DawgNode) -> u64 {
    let mut hasher = FxHasher::default();

    node.is_final.hash(&mut hasher);

    for (label, child_idx) in &node.edges {
        label.hash(&mut hasher);
        child_signature(child_idx).hash(&mut hasher);
    }

    hasher.finish()
}

// Check cache before creating new nodes
if let Some(&existing_idx) = suffix_cache.get(&signature) {
    return existing_idx;  // Reuse existing
}
```

**Impact**: 20-40% space reduction

#### 3. Bloom Filter

Fast negative lookup rejection:

```rust
fn contains(&self, term: &str) -> bool {
    let lock = self.inner.read();

    // Fast rejection (no DAWG traversal needed)
    if let Some(ref bloom) = lock.bloom_filter {
        if !bloom.might_contain(term) {
            return false;  // Definitely not present
        }
    }

    // Full DAWG traversal
    lock.traverse(term)
}
```

**Impact**: 5-10x faster negative lookups

#### 4. Lazy Minimization

Defer expensive minimization until threshold reached:

```rust
// Minimize only when node count grows significantly
if nodes.len() > last_minimized * auto_minimize_threshold {
    self.minimize();
    last_minimized = nodes.len();
}
```

**Impact**: Amortizes $`\mathcal{O}(n)`$ cost over many insertions

## Construction Methods

DynamicDawg provides multiple constructors for different initialization patterns, enabling both incremental construction and bulk loading scenarios.

### Overview

| Constructor | Complexity | Use Case | Thread-Safe |
|-------------|-----------|----------|-------------|
| `new()` | $`\mathcal{O}(1)`$ | Empty start, incremental | ✅ |
| `from_iter()` | $`\mathcal{O}(n \cdot m)`$ | Bulk load from iterator | ✅ |
| `from_terms()` | $`\mathcal{O}(n \cdot m)`$ | Simple term list | ✅ |
| `insert_with_value()` | $`\mathcal{O}(m)`$ amortized | Per-term values | ✅ |

Where n = number of terms, m = average term length

### Empty Dictionary

Create an empty dictionary for incremental population:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// Create empty dictionary
let dict: DynamicDawg = DynamicDawg::new();

// Incrementally add terms
dict.insert("hello");
dict.insert("world");

// Or with values
let valued_dict: DynamicDawg<u32> = DynamicDawg::new();
valued_dict.insert_with_value("hello", 100);
valued_dict.insert_with_value("world", 200);
```

**Characteristics:**
- **Time**: $`\mathcal{O}(1)`$ - Allocates minimal structure
- **Memory**: small fixed allocation (Arc + empty lock-free DynamicDawgInner)
- **Use case**: Real-time incremental updates, streaming input

**When to use:**
- ✅ Building dictionary gradually (e.g., parsing documents one-by-one)
- ✅ Interactive applications where terms arrive over time
- ✅ Need to start querying before all data available

### From Iterator

Build dictionary from any iterator over string-like items:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// From Vec
let terms = vec!["apple", "banana", "cherry"];
let dict = DynamicDawg::from_iter(terms);

// From HashSet
use std::collections::HashSet;
let term_set: HashSet<&str> = ["dog", "cat", "bird"].iter().copied().collect();
let dict = DynamicDawg::from_iter(term_set);

// From file lines
use std::fs::File;
use std::io::{BufRead, BufReader};

let file = File::open("dictionary.txt")?;
let lines = BufReader::new(file).lines().filter_map(|l| l.ok());
let dict = DynamicDawg::from_iter(lines);
```

**Characteristics:**
- **Time**: $`\mathcal{O}(n \cdot m)`$ where n=terms, m=avg length
- **Memory**: Linear with term count (~250KB for 10K terms)
- **Optimization**: Pre-sorting terms improves cache locality

**Performance tip:**
```rust
// Sort terms first for better performance
let mut terms = vec!["zebra", "apple", "mango"];
terms.sort_unstable();  // ~10-15% faster construction
let dict = DynamicDawg::from_iter(terms);
```

### From Simple Term List

Convenience wrapper for common case of Vec/slice of terms:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// Direct from slice
let dict = DynamicDawg::from_terms(&["test", "testing", "tester"]);

// From Vec
let terms = vec!["hello".to_string(), "world".to_string()];
let dict = DynamicDawg::from_terms(terms);
```

**Equivalent to `from_iter()`** but more concise for simple cases.

### With Associated Values

Insert terms with associated metadata (frequencies, IDs, etc.):

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// Example: Term frequencies
let dict: DynamicDawg<u32> = DynamicDawg::new();
dict.insert_with_value("the", 1000000);    // Very common
dict.insert_with_value("hello", 50000);    // Common
dict.insert_with_value("xylophone", 100);  // Rare

// Example: Context IDs (for code completion)
type ContextId = u32;
let dict: DynamicDawg<Vec<ContextId>> = DynamicDawg::new();
dict.insert_with_value("println", vec![1, 2, 3]);  // Visible in contexts 1,2,3
dict.insert_with_value("my_func", vec![42]);       // Only in context 42

// Retrieve values
if let Some(freq) = dict.get_value("the") {
    println!("Frequency: {}", freq);  // 1000000
}
```

**Value type requirements:**
- Must implement `DictionaryValue` trait
- Bounds: `Clone + Send + Sync + 'static`
- Common types: `u32`, `String`, `Vec<T>`, custom structs

### Constructor Comparison

**Performance** (10,000 terms, Intel Xeon E5-2699 v3 @ 2.30GHz):

| Method | Time | Memory Peak | Notes |
|--------|------|-------------|-------|
| `new()` + inserts | ~8.2ms | ~250KB | Sequential, more lock overhead |
| `from_iter()` | ~4.1ms | ~250KB | Bulk construction, less overhead |
| `from_terms()` | ~4.1ms | ~250KB | Same as from_iter |
| Pre-sorted input | ~3.5ms | ~250KB | 15% faster due to cache locality |

**Memory usage** (varies with term count and length):

```
Small (1K terms):     ~30KB
Medium (10K terms):   ~250KB
Large (100K terms):   ~2.5MB
Very large (1M terms): ~25MB
```

### Best Practices

**1. Choose the right constructor:**
```rust
// ✅ Good: Bulk load with from_iter()
let dict = DynamicDawg::from_iter(large_term_list);

// ❌ Avoid: Many individual inserts when you have all data
let dict = DynamicDawg::new();
for term in large_term_list {
    dict.insert(term);  // Slower due to per-insert overhead
}
```

**2. Pre-sort for performance:**
```rust
let mut terms = load_terms();
terms.sort_unstable();  // 10-15% speedup
let dict = DynamicDawg::from_iter(terms);
```

**3. Choose appropriate value types:**
```rust
// ✅ Good: Use u32 for IDs (4 bytes)
let dict: DynamicDawg<u32> = DynamicDawg::new();

// ⚠️ Acceptable but larger: Use String for metadata
let dict: DynamicDawg<String> = DynamicDawg::new();  // Higher memory

// ✅ Best for code completion: Vec<ContextId>
let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
```

**4. Error handling with file input:**
```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_dictionary(path: &Path) -> Result<DynamicDawg, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let terms: Vec<String> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().is_empty())  // Skip empty
        .collect();

    Ok(DynamicDawg::from_iter(terms))
}
```

### Parallel Construction

For workspace-scale dictionaries (100+ documents):

```rust
use rayon::prelude::*;

// Build per-document dictionaries in parallel
let dicts: Vec<DynamicDawg<Vec<u32>>> = documents
    .par_iter()
    .map(|(ctx_id, doc)| {
        let terms = extract_terms(doc);
        let dict = DynamicDawg::new();
        for term in terms {
            dict.insert_with_value(term, vec![*ctx_id]);
        }
        dict
    })
    .collect();

// Merge using union_with (see Union Operations section)
// Full pattern documented in Contextual Completion guide
```

→ See [Parallel Workspace Indexing](../../07-contextual-completion/patterns/parallel-workspace-indexing.md) for complete pattern with ~150× speedup.

## Accessor Methods

DynamicDawg provides comprehensive methods for querying dictionary contents and metadata.

### Overview Table

| Method | Returns | Complexity | Thread-Safe | Description |
|--------|---------|------------|-------------|-------------|
| `contains(term)` | `bool` | $`\mathcal{O}(m)`$ | ✅ Yes | Check if term exists |
| `get_value(term)` | `Option<V>` | $`\mathcal{O}(m)`$ | ✅ Yes | Retrieve associated value |
| `len()` | `Option<usize>` | $`\mathcal{O}(1)`$ | ✅ Yes | Get term count (Dictionary trait) |
| `is_empty()` | `bool` | $`\mathcal{O}(1)`$ | ✅ Yes | Check if empty (Dictionary trait) |
| `term_count()` | `usize` | $`\mathcal{O}(1)`$ | ✅ Yes | Get exact term count |
| `node_count()` | `usize` | $`\mathcal{O}(1)`$ | ✅ Yes | Get internal node count |
| `needs_compaction()` | `bool` | $`\mathcal{O}(1)`$ | ✅ Yes | Check if compaction recommended |
| `root()` | `DynamicDawgNode` | $`\mathcal{O}(1)`$ | ✅ Yes | Get root node for traversal |

*Note*: `m` = term length (in bytes).

---

### contains() - Term Existence Check

Check if a term exists in the dictionary.

**Signature**:
```rust
pub fn contains(&self, term: &str) -> bool
```

**Performance**:
- **Complexity**: $`\mathcal{O}(m)`$ where m is term length
- **Optimizations**: Bloom filter for fast negative lookups (~100× faster rejection)
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict = DynamicDawg::from_terms(vec!["cat", "dog"]);

assert!(dict.contains("cat"));
assert!(dict.contains("dog"));
assert!(!dict.contains("bird"));
assert!(!dict.contains("ca")); // Prefix doesn't count
```

**Bloom Filter Optimization** (enabled by default):

```rust
// With Bloom filter (default)
let dict = DynamicDawg::new(); // Bloom filter auto-enabled
dict.insert("term1");
dict.insert("term2");

// Fast negative lookup (~100× faster than full traversal)
assert!(!dict.contains("nonexistent")); // Bloom filter rejects immediately

// Custom Bloom filter capacity
let dict = DynamicDawg::with_config(2.0, Some(10_000));
// Optimized for ~10,000 terms
```

**Thread Safety**:
```rust
use std::sync::Arc;
use std::thread;

let dict = Arc::new(DynamicDawg::from_terms(vec!["hello", "world"]));

// Concurrent reads are safe
let handles: Vec<_> = (0..10)
    .map(|_| {
        let d = Arc::clone(&dict);
        thread::spawn(move || d.contains("hello"))
    })
    .collect();

for h in handles {
    assert!(h.join().unwrap());
}
```

---

### get_value() - Retrieve Associated Value

Get the value associated with a term (for value-storing dictionaries).

**Signature**:
```rust
pub fn get_value(&self, term: &str) -> Option<V>
where
    V: Clone + Send + Sync + 'static
```

**Returns**:
- `Some(value)` if term exists and has associated value
- `None` if term doesn't exist or has no value

**Performance**:
- **Complexity**: $`\mathcal{O}(m)`$ where m is term length
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// Dictionary with integer values
let dict: DynamicDawg<u32> = DynamicDawg::new();
dict.insert_with_value("apple", 42);
dict.insert_with_value("banana", 100);

assert_eq!(dict.get_value("apple"), Some(42));
assert_eq!(dict.get_value("banana"), Some(100));
assert_eq!(dict.get_value("cherry"), None); // Doesn't exist

// Dictionary with vector values (contextual completion)
let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
dict.insert_with_value("function", vec![1, 2, 3]); // Context IDs
dict.insert_with_value("variable", vec![2, 4]);

assert_eq!(dict.get_value("function"), Some(vec![1, 2, 3]));
assert_eq!(dict.get_value("variable"), Some(vec![2, 4]));
```

**Value Type Requirements**:
```rust
// ✓ Valid value types
DynamicDawg<()>           // Unit type (no values)
DynamicDawg<u32>          // Primitive
DynamicDawg<String>       // Owned string
DynamicDawg<Vec<u32>>     // Vector (for contextual completion)
DynamicDawg<Arc<Data>>    // Shared data

// ✗ Invalid (doesn't implement DictionaryValue trait)
// DynamicDawg<&str>      // References not allowed
// DynamicDawg<Rc<Data>>  // !Send
```

---

### len() and is_empty() - Dictionary Trait Methods

Standard collection size queries via `Dictionary` trait.

**Signatures**:
```rust
fn len(&self) -> Option<usize>  // Dictionary trait
fn is_empty(&self) -> bool      // Dictionary trait
```

**Returns**:
- `len()`: `Some(count)` for DynamicDawg (exact count always available)
- `is_empty()`: `true` if no terms, `false` otherwise

**Performance**:
- **Complexity**: $`\mathcal{O}(1)`$ - stored counter
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
use libdictenstein::{Dictionary, dynamic_dawg::DynamicDawg};

let dict = DynamicDawg::new();
assert_eq!(dict.len(), Some(0));
assert!(dict.is_empty());

dict.insert("test");
assert_eq!(dict.len(), Some(1));
assert!(!dict.is_empty());

dict.insert("another");
assert_eq!(dict.len(), Some(2));

dict.remove("test");
assert_eq!(dict.len(), Some(1));
```

---

### term_count() - Direct Term Count

Get exact number of terms (DynamicDawg-specific method, bypasses Option wrapper).

**Signature**:
```rust
pub fn term_count(&self) -> usize
```

**Performance**:
- **Complexity**: $`\mathcal{O}(1)`$ - stored counter
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
let dict = DynamicDawg::from_terms(vec!["apple", "banana", "cherry"]);
assert_eq!(dict.term_count(), 3);

dict.remove("banana");
assert_eq!(dict.term_count(), 2);

// Compare with Dictionary::len()
assert_eq!(dict.len(), Some(2)); // Wrapped in Option
assert_eq!(dict.term_count(), 2); // Direct usize
```

**Use Cases**:
- Progress tracking during bulk operations
- Capacity planning for data structures
- Debugging and logging

---

### node_count() - Internal Structure Size

Get number of internal nodes (useful for performance analysis).

**Signature**:
```rust
pub fn node_count(&self) -> usize
```

**Returns**: Total number of DAWG nodes (including non-final nodes)

**Performance**:
- **Complexity**: $`\mathcal{O}(1)`$ - stored counter
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
let dict = DynamicDawg::new();
assert_eq!(dict.node_count(), 1); // Just root node

dict.insert("cat");
dict.insert("car");
dict.insert("card");

// Nodes: root, 'c', 'a', 'r'/'t', 'd'
// Note: Exact count depends on suffix sharing
let nodes = dict.node_count();
assert!(nodes >= 4); // At least one node per unique character position

// After compaction, node count may decrease
let removed = dict.compact();
assert_eq!(removed, 0); // Already minimal
```

**Interpretation**:
- Higher node count → More memory usage
- `node_count()` $`\approx`$ `term_count()` → Good compression (lots of sharing)
- `node_count()` >> `term_count()` → Poor compression (deletions without compaction)

**Monitoring Example**:
```rust
let dict = DynamicDawg::new();

for term in generate_terms(10_000) {
    dict.insert(term);
}

// Check compression ratio
let ratio = dict.node_count() as f64 / dict.term_count() as f64;
println!("Nodes per term: {:.2}", ratio);
// Typical: 0.6-0.8 for natural language
// Lower is better (more suffix sharing)
```

---

### needs_compaction() - Compaction Recommendation

Check if deletion has left orphaned nodes requiring compaction.

**Signature**:
```rust
pub fn needs_compaction(&self) -> bool
```

**Returns**:
- `true` if deletions have occurred and compaction recommended
- `false` if structure is minimal or only insertions occurred

**Performance**:
- **Complexity**: $`\mathcal{O}(1)`$ - flag check
- **Concurrency**: Lock-free read (no reader lock)

**Example**:
```rust
let dict = DynamicDawg::from_terms(vec!["test", "testing", "tested"]);
assert!(!dict.needs_compaction()); // Freshly built

dict.remove("tested");
assert!(dict.needs_compaction()); // Deletion creates orphaned nodes

let removed = dict.compact();
assert!(!dict.needs_compaction()); // Compacted
assert!(removed > 0); // Some nodes were removed
```

**Best Practices**:
```rust
// Pattern: Batch deletions + single compaction
let dict = DynamicDawg::from_terms(generate_terms(10_000));

// Delete many terms
for term in terms_to_delete {
    dict.remove(&term);
}

// Check before compacting
if dict.needs_compaction() {
    let removed_nodes = dict.compact();
    println!("Compaction freed {} nodes", removed_nodes);
}
```

**Performance Guidance**:
- Compaction is $`\mathcal{O}(n)`$ where n = total characters
- Compact periodically, not after every deletion
- Typical trigger: After removing >10% of terms

---

### root() - Root Node for Traversal

Get the root node for manual graph traversal (Dictionary trait method).

**Signature**:
```rust
fn root(&self) -> DynamicDawgNode // From Dictionary trait
```

**Returns**: Node at root of DAWG (entry point for traversal)

**Performance**:
- **Complexity**: $`\mathcal{O}(1)`$
- **Concurrency**: Wait-free traversal over one immutable root revision

**Example**:
```rust
use libdictenstein::{Dictionary, DictionaryNode};

let dict = DynamicDawg::from_terms(vec!["cat", "car", "card"]);

// Manual traversal
let root = dict.root();
assert!(!root.is_final()); // Root typically not final

// Navigate to "car"
if let Some(c_node) = root.transition(b'c') {
    if let Some(a_node) = c_node.transition(b'a') {
        if let Some(r_node) = a_node.transition(b'r') {
            assert!(r_node.is_final()); // "car" exists

            // Check if "card" exists
            if let Some(d_node) = r_node.transition(b'd') {
                assert!(d_node.is_final()); // "card" exists
            }
        }
    }
}
```

**Zipper-Based Traversal** (preferred for complex navigation):

```rust
use libdictenstein::zipper::DictZipper;
use libdictenstein::dynamic_dawg::DynamicDawgZipper;

let dict: DynamicDawg<u32> = DynamicDawg::new();
dict.insert_with_value("hello", 42);

let zipper = DynamicDawgZipper::new_from_dict(&dict);

// Traverse with zipper
let result = zipper
    .descend(b'h')
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b'l'))
    .and_then(|z| z.descend(b'l'))
    .and_then(|z| z.descend(b'o'));

if let Some(final_zipper) = result {
    assert!(final_zipper.is_final());
    assert_eq!(final_zipper.value(), Some(42));
}
```

---

### Performance Summary

**Accessor Method Latencies** (10K term dictionary):

| Method | Latency | Throughput | Notes |
|--------|---------|------------|-------|
| `contains()` (hit) | ~250ns | 4M ops/sec | Full traversal |
| `contains()` (miss, Bloom) | ~50ns | 20M ops/sec | Bloom rejection |
| `get_value()` | ~260ns | 3.8M ops/sec | Traversal + clone |
| `len()` / `term_count()` | ~5ns | 200M ops/sec | Counter read |
| `is_empty()` | ~5ns | 200M ops/sec | Counter comparison |
| `node_count()` | proportional to live graph | — | Unique-node traversal |
| `needs_compaction()` | ~2ns | 500M ops/sec | Flag read |
| `root()` | ~3ns | 333M ops/sec | Return node 0 |

**Concurrency**:
- All accessors are lock-free reads (no reader lock)
- Multiple threads can query concurrently
- Readers never block on other readers or on writers
- Writers publish new state via `compare_exchange` (CAS)

**Memory Overhead**:
- Term count: 8 bytes (usize)
- Node count: 8 bytes (usize)
- Bloom filter: ~100KB for 10K terms (optional)
- Compaction flag: 1 byte (bool)

---

## Key Algorithms

### Insert with Suffix Sharing

```rust
fn insert_with_sharing(&mut self, term: &[u8], value: Option<V>) {
    let mut node_idx = 0;

    for (i, &byte) in term.iter().enumerate() {
        // Try to follow existing edge
        if let Some(child_idx) = self.find_edge(node_idx, byte) {
            node_idx = child_idx;
            continue;
        }

        // Need to create new branch
        // Check if remainder matches existing suffix
        let remainder = &term[i..];
        let signature = self.compute_suffix_signature(remainder, value.clone());

        if let Some(&cached_idx) = self.suffix_cache.get(&signature) {
            // Reuse existing suffix!
            self.add_edge(node_idx, byte, cached_idx);
            self.nodes[cached_idx].ref_count += 1;
            return;
        }

        // Create new suffix
        let new_idx = self.create_suffix(remainder, value);
        self.add_edge(node_idx, byte, new_idx);
        self.suffix_cache.insert(signature, new_idx);
        return;
    }

    // Mark final
    self.nodes[node_idx].is_final = true;
    self.nodes[node_idx].value = value;
}
```

### Reference-Counted Deletion

```rust
fn remove_with_ref_counting(&mut self, term: &[u8]) -> bool {
    // Traverse and record path
    let mut path = Vec::new();
    let mut node_idx = 0;

    for &byte in term {
        path.push((node_idx, byte));
        node_idx = self.find_edge(node_idx, byte)?;
    }

    if !self.nodes[node_idx].is_final {
        return false;
    }

    // Unmark final
    self.nodes[node_idx].is_final = false;
    self.nodes[node_idx].value = None;

    // Decrement reference counts
    for (parent_idx, label) in path.iter().rev() {
        let child_idx = self.find_edge(*parent_idx, *label).unwrap();
        self.nodes[child_idx].ref_count -= 1;

        // Delete if unreferenced
        if self.nodes[child_idx].ref_count == 0 &&
           !self.nodes[child_idx].is_final &&
           self.nodes[child_idx].edges.is_empty() {
            self.remove_edge(*parent_idx, *label);
        } else {
            break;  // Still in use
        }
    }

    self.needs_compaction = true;
    true
}
```

## Union Operations

### Overview

The `union_with()` and `union_replace()` methods enable **merging two DynamicDawg dictionaries** with custom value combination logic. This is essential for scenarios like:

- 📊 Aggregating statistics across multiple data sources
- 🔄 Merging user-specific and global dictionaries
- 🗂️ Combining category hierarchies
- 🔢 Building composite symbol tables

**Key Characteristics**:
- 🔒 **Thread-safe**: Lock-free reads; writers publish via CAS
- 💾 **DAWG-preserving**: Maintains minimization through `insert_with_value()`
- ⚡ **Efficient**: $`\mathcal{O}(n \cdot m)`$ traversal with minimal memory overhead
- 🎯 **Flexible**: Custom merge functions for value conflicts

### union_with() - Merge with Custom Logic

Combines two dictionaries by inserting all terms from the source dictionary, applying a custom merge function when values conflict.

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

**Algorithm**: Depth-First Search (DFS) traversal
1. Initialize stack with root node `(node_idx=0, path=Vec::new())`
2. Pop `(node_idx, path)` from stack
3. If node is final:
   - Convert path bytes to UTF-8 string
   - Check if term exists in `self`
   - If exists: Apply `merge_fn` and update
   - If new: Insert with original value
4. Push all children onto stack (reversed for consistent ordering)
5. Repeat until stack empty

**Complexity**:
- **Time**: $`\mathcal{O}(n \cdot m)`$ where n = terms in `other`, m = average term length
  - $`\mathcal{O}(n \cdot m)`$ for DFS traversal
  - $`\mathcal{O}(m)`$ per term for `insert_with_value()`
- **Space**: $`\mathcal{O}(d)`$ where d = maximum trie depth (typically < 50)
  - DFS stack size proportional to deepest path
  - Constant additional memory

### Example 1: Sum Aggregation

Merge term counts by summing conflicting values:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

// First dataset: word frequencies
let dict1: DynamicDawg<u32> = DynamicDawg::new();
dict1.insert_with_value("apple", 10);
dict1.insert_with_value("banana", 5);
dict1.insert_with_value("cherry", 3);

// Second dataset: more frequencies
let dict2: DynamicDawg<u32> = DynamicDawg::new();
dict2.insert_with_value("apple", 7);   // Overlap - will sum
dict2.insert_with_value("banana", 2);  // Overlap - will sum
dict2.insert_with_value("date", 4);    // New entry

// Merge by summing counts
let processed = dict1.union_with(&dict2, |left, right| left + right);

// Results:
// - apple: 17 (10 + 7)
// - banana: 7 (5 + 2)
// - cherry: 3 (unchanged)
// - date: 4 (new)
assert_eq!(dict1.get_value("apple"), Some(17));
assert_eq!(dict1.get_value("date"), Some(4));
assert_eq!(processed, 3); // Processed 3 terms from dict2
```

### Example 2: Set Union with Deduplication

Merge lists of associated IDs, eliminating duplicates:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

// First dictionary: terms with associated document IDs
let dict1: DynamicDawg<Vec<u32>> = DynamicDawg::new();
dict1.insert_with_value("algorithm", vec![1, 2, 5]);
dict1.insert_with_value("database", vec![3, 7]);

// Second dictionary: more document associations
let dict2: DynamicDawg<Vec<u32>> = DynamicDawg::new();
dict2.insert_with_value("algorithm", vec![2, 4, 5]); // Overlap: [2,5]
dict2.insert_with_value("distributed", vec![6, 8]);

// Merge by concatenating and deduplicating
dict1.union_with(&dict2, |left, right| {
    let mut merged = left.clone();
    merged.extend(right.clone());
    merged.sort_unstable();
    merged.dedup();
    merged
});

// Results:
// - algorithm: [1, 2, 4, 5] (merged and deduplicated)
// - database: [3, 7] (unchanged)
// - distributed: [6, 8] (new)
assert_eq!(dict1.get_value("algorithm"), Some(vec![1, 2, 4, 5]));
```

### Example 3: Maximum Value Selection

Keep the highest value when terms conflict:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

// Dictionary 1: initial scores
let dict1: DynamicDawg<i32> = DynamicDawg::new();
dict1.insert_with_value("performance", 85);
dict1.insert_with_value("reliability", 92);

// Dictionary 2: updated scores
let dict2: DynamicDawg<i32> = DynamicDawg::new();
dict2.insert_with_value("performance", 90); // Higher score
dict2.insert_with_value("reliability", 88); // Lower score
dict2.insert_with_value("security", 95);    // New metric

// Keep maximum value for conflicts
dict1.union_with(&dict2, |left, right| (*left).max(*right));

// Results:
// - performance: 90 (max of 85, 90)
// - reliability: 92 (max of 92, 88)
// - security: 95 (new)
assert_eq!(dict1.get_value("performance"), Some(90));
assert_eq!(dict1.get_value("reliability"), Some(92));
```

### Example 4: Shared Prefix Handling

Demonstrates correct behavior with terms sharing common prefixes:

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

// Dictionary with "test" prefix family
let dict1: DynamicDawg<u32> = DynamicDawg::new();
dict1.insert_with_value("test", 1);
dict1.insert_with_value("testing", 2);
dict1.insert_with_value("tester", 3);

// More "test" variants
let dict2: DynamicDawg<u32> = DynamicDawg::new();
dict2.insert_with_value("test", 10);      // Conflict
dict2.insert_with_value("tested", 4);     // New, shares "test" prefix
dict2.insert_with_value("testimony", 5);  // New, shares "test" prefix

dict1.union_with(&dict2, |left, right| left + right);

// All terms preserved correctly despite shared prefixes
assert_eq!(dict1.len().unwrap(), 5);
assert_eq!(dict1.get_value("test"), Some(11));       // 1 + 10
assert_eq!(dict1.get_value("tested"), Some(4));      // New
assert_eq!(dict1.get_value("testimony"), Some(5));   // New
```

### union_replace() - Keep Right Values

Convenience method equivalent to `union_with(other, |_, right| right.clone())`. Keeps values from `other` when terms conflict.

**Signature**:
```rust
fn union_replace(&self, other: &Self) -> usize
where
    Self::Value: Clone
```

**Example**:
```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MutableMappedDictionary;

let dict1: DynamicDawg<&str> = DynamicDawg::new();
dict1.insert_with_value("version", "1.0");
dict1.insert_with_value("status", "beta");

let dict2: DynamicDawg<&str> = DynamicDawg::new();
dict2.insert_with_value("version", "2.0");    // Override
dict2.insert_with_value("author", "alice");   // New

// Replace conflicting values with those from dict2
dict1.union_replace(&dict2);

// Results:
// - version: "2.0" (replaced)
// - status: "beta" (unchanged)
// - author: "alice" (new)
assert_eq!(dict1.get_value("version"), Some("2.0"));
assert_eq!(dict1.get_value("status"), Some("beta"));
```

### Implementation Details

The union operation uses **iterative depth-first search** to traverse all terms in the source dictionary:

```rust
// Simplified pseudocode
fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize {
    let other_inner = other.inner.read();
    let mut processed = 0;

    // Initialize DFS with root: (node_index, accumulated_path)
    let mut stack: Vec<(usize, Vec<u8>)> = vec![(0, Vec::new())];

    while let Some((node_idx, path)) = stack.pop() {
        let node = &other_inner.nodes[node_idx];

        // Process final nodes (complete terms)
        if node.is_final {
            if let Ok(term) = std::str::from_utf8(&path) {
                processed += 1;

                if let Some(other_value) = &node.value {
                    if let Some(self_value) = self.get_value(term) {
                        // Term exists - merge values
                        let merged = merge_fn(&self_value, other_value);
                        self.insert_with_value(term, merged);
                    } else {
                        // New term - insert directly
                        self.insert_with_value(term, other_value.clone());
                    }
                }
            }
        }

        // Push children onto stack (reversed for consistent order)
        for &(label, target_idx) in node.edges.iter().rev() {
            let mut child_path = path.clone();
            child_path.push(label);
            stack.push((target_idx, child_path));
        }
    }

    processed
}
```

**Why Iterative DFS?**
- ✅ **No stack overflow**: Handles very deep tries (e.g., long terms)
- ✅ **Memory efficient**: $`\mathcal{O}(d)`$ space vs $`\mathcal{O}(n)`$ for recursion
- ✅ **Consistent ordering**: Reversed edges ensure predictable traversal
- ✅ **Debuggable**: Explicit stack state visible at each step

**Why Use `insert_with_value()`?**

The implementation delegates to `insert_with_value()` rather than manipulating nodes directly. This design choice:

1. **Preserves DAWG minimization**: Insertion logic handles suffix sharing and node deduplication
2. **Maintains reference counts**: Proper accounting for shared nodes
3. **Simpler and safer**: Avoids complex graph manipulation bugs
4. **Future-proof**: Benefits from optimizations to insertion algorithm

**Trade-off**: Slightly slower than direct node manipulation, but correctness > speed for complex structures.

### Performance Characteristics

| Operation | Time Complexity | Space Complexity | Typical Performance (10K terms) |
|-----------|----------------|------------------|--------------------------------|
| `union_with()` | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(d)`$ | ~50ms |
| `union_replace()` | $`\mathcal{O}(n \cdot m)`$ | $`\mathcal{O}(d)`$ | ~50ms |
| DFS traversal | $`\mathcal{O}(n)`$ | $`\mathcal{O}(d)`$ | ~20ms |
| Per-term insertion | $`\mathcal{O}(m)`$ | $`\mathcal{O}(1)`$ amortized | ~2-5µs |

**Variables**:
- n = number of terms in source dictionary
- m = average term length (typically 5-15 bytes)
- d = maximum trie depth (typically 20-50)

**Memory Profile**:
```
Stack size: ~200-2000 bytes (depth × 40 bytes per frame)
Peak allocation: O(m) for path accumulation
No heap allocations during traversal (Vec reused)
```

**Benchmark Results** (Intel Xeon E5-2699 v3 @ 2.30GHz):

| Dictionary Size | union_with() | Throughput |
|----------------|-------------|------------|
| 1,000 terms    | 4.2ms       | 238K terms/s |
| 10,000 terms   | 48ms        | 208K terms/s |
| 100,000 terms  | 520ms       | 192K terms/s |

*Note*: Performance includes merge function execution. Simple operations (e.g., sum) add minimal overhead.

### When to Use Union Operations

✅ **Use `union_with()` when:**
- **Parallel workspace indexing**: Merging per-document dictionaries built in parallel (→ [Parallel Workspace Pattern](../../07-contextual-completion/patterns/parallel-workspace-indexing.md))
- Merging user-specific and system dictionaries
- Aggregating statistics from multiple sources (word counts, frequencies)
- Combining hierarchical categories or tags
- Building composite symbol tables in compilers/interpreters
- Synchronizing dictionaries across distributed systems
- Implementing set operations on labeled data

✅ **Use `union_replace()` when:**
- Updating dictionaries with newer data (last-writer-wins semantics)
- Applying configuration overrides (defaults + user settings)
- Merging dictionaries where conflicts indicate stale data

⚠️ **Consider alternatives when:**
- **Dictionaries are static**: Pre-merge at build time with [`from_terms_with_values()`](dynamic-dawg.md#example-2-with-values)
- **One dictionary much larger**: Iterate the smaller dictionary and insert into larger (avoids traversing large dict)
- **No value merging needed**: Use simple iteration: `for (term, value) in dict2.iter() { dict1.insert_with_value(term, value); }`
- **Frequent unions on same dictionaries**: Cache union result or use different data structure (e.g., separate indices)

### Thread Safety Considerations

Union operations are **fully thread-safe** (lock-free reads; CAS-published writes):

```rust
use std::sync::Arc;
use std::thread;

let dict1 = Arc::new(DynamicDawg::new());
let dict2 = Arc::new(DynamicDawg::new());

// Populate dictionaries from multiple threads
let handles: Vec<_> = (0..4).map(|i| {
    let d1 = Arc::clone(&dict1);
    let d2 = Arc::clone(&dict2);

    thread::spawn(move || {
        if i % 2 == 0 {
            d1.insert_with_value(&format!("term_{}", i), i);
        } else {
            d2.insert_with_value(&format!("term_{}", i), i);
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }

// Merge from any thread
dict1.union_with(&dict2, |a, b| a + b);
```

**Concurrency**: Union traverses an immutable snapshot of `other` while publishing lock-free `compare_exchange` (CAS) insertions into `self`:
- ✅ Concurrent reads from `self` never block (readers load an atomic snapshot)
- ✅ Concurrent reads from `other` are always safe
- ⚠️ Concurrent writers to `self` contend only at the CAS boundary (retry on conflict)

For high-concurrency scenarios, consider:
1. Performing union on a clone
2. Batching multiple unions
3. Using snapshot-and-merge patterns

## Usage Examples

### Example 1: Basic Usage

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

// Create empty DAWG
let dict = DynamicDawg::new();

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

### Example 2: With Values

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MappedDictionary;

let dict: DynamicDawg<u32> = DynamicDawg::new();

// Insert with values
dict.insert_with_value("test", 1);
dict.insert_with_value("testing", 2);

// Query values
assert_eq!(dict.get_value("test"), Some(1));
assert_eq!(dict.get_value("testing"), Some(2));

// Remove preserves other terms
dict.remove("test");
assert_eq!(dict.get_value("testing"), Some(2));
```

### Example 3: From Existing Terms

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict = DynamicDawg::from_terms(vec![
    "algorithm", "approximate", "automaton"
]);

// Add new terms at runtime
dict.insert("analysis");

assert!(dict.contains("algorithm"));
assert!(dict.contains("analysis"));
```

### Example 4: Thread-Safe Updates

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use std::sync::Arc;
use std::thread;

let dict = Arc::new(DynamicDawg::from_terms(vec!["initial"]));

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

### Example 5: Compaction

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;

let dict = DynamicDawg::from_terms(vec![
    "test1", "test2", "test3", "test4", "test5"
]);

println!("Before deletion: {} nodes", dict.node_count());

// Remove many terms
for i in 1..=4 {
    dict.remove(&format!("test{}", i));
}

println!("After deletion: {} nodes (may have orphans)", dict.node_count());

// Compact to restore minimality
dict.compact();

println!("After compaction: {} nodes", dict.node_count());
```

### Example 6: Fuzzy Search with Dynamic Updates

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DynamicDawg::from_terms(vec!["test", "testing"]);

// Fuzzy search
let automaton = LevenshteinAutomaton::new("tset", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();
println!("{:?}", results);  // ["test"]

// Add term dynamically
dict.insert("tester");

// Search again (sees new term)
let results: Vec<String> = automaton.query(&dict).collect();
println!("{:?}", results);  // ["test", "tester"]
```

## Performance Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Insert** | $`\mathcal{O}(m)`$ | m = term length |
| **Remove** | $`\mathcal{O}(m)`$ | Plus ref count updates |
| **Contains** | $`\mathcal{O}(m)`$ | With Bloom filter: $`\mathcal{O}(1)`$ rejection |
| **Compact** | $`\mathcal{O}(n)`$ | n = total nodes |
| **Query (fuzzy)** | $`\mathcal{O}(m \times d^{2} \times b)`$ | d = distance, b = branching |

### Benchmark Results

#### Construction

```
Build from 10,000 terms:
  DynamicDawg:      4.1ms
  DoubleArrayTrie:  3.2ms  (22% faster)
```

#### Runtime Operations

```
Single insertion (amortized):
  DynamicDawg:      ~800ns

Single deletion:
  DynamicDawg:      ~1.2µs

Contains check:
  With Bloom filter:    ~150ns (negative)
  Without Bloom filter: ~350ns (negative)
  Positive lookup:      ~450ns
```

#### Fuzzy Search

```
Query "test" (distance 2) in 10K-term dict:
  DynamicDawg:      42.3µs
  DoubleArrayTrie:  16.3µs  (2.6x faster)
```

### Memory Usage

```
10,000-term dictionary:
  Nodes:          ~250KB
  Suffix cache:   ~32KB
  Bloom filter:   ~12KB
  Total:          ~294KB

vs DoubleArrayTrie: ~100KB (3x smaller)
```

**Trade-off**: DynamicDawg uses more memory for update flexibility

### Compaction Impact

```
After removing 30% of terms:
  Before compaction:  350KB (orphaned nodes)
  After compaction:   210KB (40% reduction)

Compaction time:      ~8ms for 10K terms
```

## When to Use

### Decision Matrix

| Scenario | Recommended | Alternative |
|----------|-------------|-------------|
| **Frequent adds + removes** | ✅ DynamicDawg | - |
| **Append-only** | ⚠️ DoubleArrayTrie | 3x faster |
| **Static dictionary** | ⚠️ DoubleArrayTrie | 3x faster, 3x smaller |
| **Unicode text** | ⚠️ DynamicDawgChar | Correct distances |
| **Maximum performance** | ⚠️ DoubleArrayTrie | Faster queries |
| **Real-time collaboration** | ✅ DynamicDawg | Thread-safe |

### Ideal Use Cases

1. **User Dictionaries**
   - Add custom words during session
   - Remove typos or unwanted entries
   - Personal vocabulary evolves

2. **Session-Specific Terms**
   - Add terms from current document
   - Clear when document closes
   - Dynamic scope-based dictionaries

3. **Collaborative Editing**
   - Multiple users add/remove terms
   - Thread-safe concurrent access
   - Real-time updates

4. **Adaptive Systems**
   - Learn new terms from user input
   - Remove deprecated entries
   - Evolving vocabulary

## Related Documentation

- [Dictionary Layer](../README.md) - Overview of all dictionary types
- [DynamicDawgChar](dynamic-dawg-char.md) - Unicode variant
- [DoubleArrayTrie](double-array-trie.md) - Faster alternative for static/append-only
- [Value Storage](../../09-value-storage/README.md) - Using values with DynamicDawg

## References

### Academic Papers

1. **Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987)**. "Complete inverted files for efficient text retrieval and analysis"
   - *Journal of the ACM*, 34(3), 578-595
   - DOI: [10.1145/28869.28873](https://doi.org/10.1145/28869.28873)
   - 📄 DAWG construction algorithms

2. **Crochemore, M., & Vérin, R. (1997)**. "Direct construction of compact directed acyclic word graphs"
   - *Annual Symposium on Combinatorial Pattern Matching*, 116-129
   - DOI: [10.1007/3-540-63220-4_55](https://doi.org/10.1007/3-540-63220-4_55)
   - 📄 Incremental DAWG construction

3. **Inenaga, S., Hoshino, H., Shinohara, A., Takeda, M., & Arikawa, S. (2001)**. "On-line construction of compact directed acyclic word graphs"
   - *Annual Symposium on Combinatorial Pattern Matching*, 83-97
   - DOI: [10.1007/3-540-48194-X_8](https://doi.org/10.1007/3-540-48194-X_8)
   - 📄 Online DAWG modifications

### Textbooks

4. **Gusfield, D. (1997)**. *Algorithms on Strings, Trees, and Sequences*
   - Cambridge University Press, Chapter 6
   - ISBN: 978-0521585194
   - 📚 Suffix structures and DAWGs

## Next Steps

- **Unicode**: Learn about [DynamicDawgChar](dynamic-dawg-char.md)
- **Performance**: Compare with [DoubleArrayTrie](double-array-trie.md)
- **Values**: Explore [Value Storage](../../09-value-storage/README.md)

---

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)
