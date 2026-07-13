# Layer 8: Caching Layer

## Overview

The Caching Layer provides composable eviction strategy wrappers that add caching behavior to any dictionary implementation. Using the decorator pattern, these wrappers maintain separate metadata (access times, hit counts, sizes) without modifying the underlying dictionary.

![Eviction decorator stack: composable eviction wrappers (LRU, TTL, Noop) layered over an inner dictionary via the decorator pattern](../../diagrams/cache/eviction-decorator-stack.svg)

*Eviction decorator stack: eviction strategies compose as decorators over any inner dictionary.*

**Key Features**:
- **Composable**: Stack multiple eviction strategies
- **Non-Intrusive**: No changes to dictionary implementations
- **Thread-Safe**: Arc<RwLock<HashMap>> for concurrent access
- **Zero-Cost Abstractions**: Noop wrapper for production disabling

## Architecture

### Decorator Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│              Eviction Wrapper (e.g., Lru<D>)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  inner: D                                                 │   │
│  │  metadata: Arc<RwLock<HashMap<String, Metadata>>>        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           Inner Dictionary (PathMapDictionary, etc.)             │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Separation of concerns: caching logic separate from dictionary logic
- Flexibility: Change eviction strategy without modifying dictionary
- Composition: Combine multiple strategies (TTL + LRU, etc.)

### Module Structure

```
src/cache/
├── mod.rs                  - Cache module entry point
├── multimap.rs             - Multi-value caching
└── eviction/
    ├── mod.rs              - Eviction module documentation
    ├── noop.rs             - Zero-cost passthrough
    ├── lazy_init.rs        - Deferred initialization
    ├── ttl.rs              - Time-to-live expiration
    ├── age.rs              - FIFO (First In, First Out)
    ├── lru.rs              - Least Recently Used
    ├── lru_optimized.rs    - Optimized LRU implementation
    ├── lfu.rs              - Least Frequently Used
    ├── cost_aware.rs       - Cost-to-value ratio
    └── memory_pressure.rs  - Memory-aware eviction
```

## Available Eviction Policies

### Core Wrappers

#### Noop (Zero-Cost Passthrough)

Identity wrapper that forwards all calls directly to inner dictionary.

```rust
use liblevenshtein::cache::eviction::Noop;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms(["hello", "world"]);
let noop = Noop::new(dict);

// Zero overhead - direct passthrough
assert!(noop.contains("hello"));
```

**Use Case**: Production builds where caching metadata is not needed.

#### LazyInit (Deferred Initialization)

Defers dictionary construction until first access.

**Variants**:
- `LazyInitDefault<D>`: Uses `Default::default()`
- `LazyInitFn<D, F>`: Uses provided closure
- `LazyInitFull<D>`: Uses `Fn() -> D` closure

```rust
use liblevenshtein::cache::eviction::LazyInitDefault;
use liblevenshtein::prelude::*;

// Dictionary not constructed yet
let lazy = LazyInitDefault::<PathMapDictionary<()>>::new();

// Constructed on first access
assert!(!lazy.contains("test"));
```

**Use Case**: Large dictionaries that may not be needed in every code path.

### Time-Based Eviction

#### TTL (Time-to-Live)

Filters expired entries based on fixed duration.

```rust
use liblevenshtein::cache::eviction::Ttl;
use liblevenshtein::prelude::*;
use std::time::Duration;

let dict = PathMapDictionary::from_terms_with_values([
    ("session_token", "abc123"),
    ("user_id", "12345"),
]);

let ttl = Ttl::new(dict, Duration::from_secs(300));  // 5 minutes

// Accesses before expiration work
assert_eq!(ttl.get_value("session_token"), Some("abc123"));

// After 300 seconds, entries return None
std::thread::sleep(Duration::from_secs(301));
assert_eq!(ttl.get_value("session_token"), None);
```

**Metadata**: `inserted_at: Instant`

**Eviction Criterion**: `inserted_at.elapsed() > duration`

**Use Cases**:
- Session tokens
- Cache invalidation
- Temporary data

#### Age (FIFO)

Evicts oldest entries first (First In, First Out).

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms_with_values([
    ("old", 1),
    ("new", 2),
]);

let age = Age::new(dict);

// Find oldest entry
let oldest = age.find_oldest(&["old", "new"]);
assert_eq!(oldest, Some("old".to_string()));
```

**Metadata**: `inserted_at: Instant`

**Eviction Criterion**: Oldest `inserted_at`

**Use Cases**:
- Log rotation
- Simple eviction without access tracking
- FIFO queues

#### LRU (Least Recently Used)

Evicts entries not accessed recently.

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms_with_values([
    ("active", 1),
    ("cold", 2),
]);

let lru = Lru::new(dict);

// Access "active" repeatedly
lru.get_value("active");
lru.get_value("active");

// Access "cold" once
lru.get_value("cold");

// Find least recently used
let lru_entry = lru.find_lru(&["active", "cold"]);
assert_eq!(lru_entry, Some("cold".to_string()));
```

**Metadata**: `last_accessed: Instant`

**Eviction Criterion**: Longest `last_accessed.elapsed()`

**Use Cases**:
- Code completion (favor recent identifiers)
- General-purpose caching
- Hot/cold data separation

### Frequency-Based Eviction

#### LFU (Least Frequently Used)

Evicts entries with lowest access count.

```rust
use liblevenshtein::cache::eviction::Lfu;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms_with_values([
    ("popular", 1),
    ("rare", 2),
]);

let lfu = Lfu::new(dict);

// Access "popular" 10 times
for _ in 0..10 {
    lfu.get_value("popular");
}

// Access "rare" once
lfu.get_value("rare");

// Find least frequently used
let lfu_entry = lfu.find_lfu(&["popular", "rare"]);
assert_eq!(lfu_entry, Some("rare".to_string()));
```

**Metadata**: `access_count: usize`

**Eviction Criterion**: Lowest `access_count`

**Use Cases**:
- Popular content caching
- Frequency-based prioritization
- Statistical analysis

### Cost-Based Eviction

#### CostAware

Balances age, size, and hit count using formula: `$(\text{age} \times \text{size}) / (\text{hits} + 1)$`

```rust
use liblevenshtein::cache::eviction::CostAware;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms_with_values([
    ("small_hot", vec![1, 2, 3]),          // Small, frequently accessed
    ("large_cold", vec![1; 1000]),         // Large, rarely accessed
]);

let cost = CostAware::new(dict);

// Access "small_hot" repeatedly
for _ in 0..100 {
    cost.get_value("small_hot");
}

// Access "large_cold" once
cost.get_value("large_cold");

// Find highest cost (worst value)
let high_cost = cost.find_highest_cost(&["small_hot", "large_cold"]);
assert_eq!(high_cost, Some("large_cold".to_string()));
```

**Metadata**:
- `inserted_at: Instant`
- `access_count: usize`
- `size: usize`

**Eviction Criterion**: Highest `$(\text{age} \times \text{size}) / (\text{hits} + 1)$`

**Use Cases**:
- Memory-constrained systems
- Balancing recency vs frequency
- Cost-benefit optimization

#### MemoryPressure

Memory-aware eviction using formula: `$\text{size} / (\text{hit\_rate} + 0.1)$`

```rust
use liblevenshtein::cache::eviction::MemoryPressure;
use liblevenshtein::prelude::*;

let dict = PathMapDictionary::from_terms_with_values([
    ("efficient", vec![1, 2]),     // Small, high hit rate
    ("wasteful", vec![1; 500]),    // Large, low hit rate
]);

let memory = MemoryPressure::new(dict);

// Access "efficient" 100 times
for _ in 0..100 {
    memory.get_value("efficient");
}

// Access "wasteful" once
memory.get_value("wasteful");

// Find highest memory pressure
let high_pressure = memory.find_highest_pressure(&["efficient", "wasteful"]);
assert_eq!(high_pressure, Some("wasteful".to_string()));
```

**Metadata**:
- `size: usize`
- `total_accesses: usize`
- `hits: usize`

**Eviction Criterion**: Highest `$\text{size} / (\text{hit\_rate} + 0.1)$`

**Use Cases**:
- Low-memory environments
- Cache efficiency optimization
- Adaptive memory management

## Composition Strategies

### Sequential Composition

Stack wrappers to combine eviction strategies:

```rust
use liblevenshtein::cache::eviction::{Lru, Ttl};
use liblevenshtein::prelude::*;
use std::time::Duration;

let dict = PathMapDictionary::from_terms_with_values([
    ("foo", 42),
    ("bar", 99),
]);

// Compose TTL + LRU
let ttl = Ttl::new(dict, Duration::from_secs(300));
let lru = Lru::new(ttl);

// Entries expire after 5 minutes AND track recency
assert_eq!(lru.get_value("foo"), Some(42));
```

**Order Matters**:
- Inner wrapper executes first
- Outer wrapper sees filtered results

### Example Compositions

#### TTL + LRU (Time + Recency)

```rust
Lru::new(Ttl::new(dict, duration))
```

**Effect**: Expire old entries, evict least recent among remaining

**Use Case**: Session management with recency tracking

#### LFU + CostAware (Frequency + Size)

```rust
CostAware::new(Lfu::new(dict))
```

**Effect**: Track frequency, then balance cost/benefit

**Use Case**: High-performance caching with memory constraints

#### LazyInit + LRU (Deferred + Recency)

```rust
Lru::new(LazyInitDefault::new())
```

**Effect**: Defer loading until first access, then track recency

**Use Case**: Large dictionaries with uncertain usage patterns

## Performance Characteristics

### Metadata Overhead

| Wrapper | Per-Entry Overhead | Thread-Safety |
|---------|-------------------|---------------|
| Noop | 0 bytes | N/A (passthrough) |
| LazyInit | 0 bytes (until init) | Arc (initialization) |
| TTL | 16 bytes (Instant) | RwLock |
| Age | 16 bytes (Instant) | RwLock |
| LRU | 16 bytes (Instant) | RwLock |
| LFU | 8 bytes (usize) | RwLock |
| CostAware | 32 bytes (Instant + 2×usize) | RwLock |
| MemoryPressure | 24 bytes (3×usize) | RwLock |

### Lock Contention

**RwLock Performance**:
- **Read-heavy**: Multiple concurrent reads (shared lock)
- **Write-heavy**: Serialized writes (exclusive lock)

**Mitigation**:
- Use Noop in production if metadata not needed
- Shard cache by key prefix to reduce contention
- Profile lock wait times under load

### Time Complexity

| Operation | Noop | Metadata Wrappers |
|-----------|------|-------------------|
| get_value | `$\mathcal{O}(d)$` | `$\mathcal{O}(d)$` + `$\mathcal{O}(1)$` metadata update |
| contains | `$\mathcal{O}(d)$` | `$\mathcal{O}(d)$` + `$\mathcal{O}(1)$` metadata update |
| find_lru/lfu | N/A | `$\mathcal{O}(n)$` scan |

Where:
- `d` = dictionary operation complexity
- `n` = number of candidates

## Usage Guidelines

### When to Use Each Policy

| Use Case | Recommended Policy | Rationale |
|----------|-------------------|-----------|
| Web sessions | TTL | Fixed expiration time |
| Code completion | LRU | Favor recent identifiers |
| Popular content | LFU | Frequency-based ranking |
| Memory-constrained | MemoryPressure | Size-aware eviction |
| Balanced | CostAware | Multi-factor optimization |
| Log rotation | Age | Simple FIFO ordering |

### Composition Patterns

#### Hot/Cold Separation

```rust
// Hot tier: LRU for active data
// Cold tier: Age for archival

let hot_cache = Lru::new(small_dict);
let cold_cache = Age::new(large_dict);
```

#### Multi-Tier Caching

```rust
// L1: TTL for short-lived data
// L2: LRU for medium-term data
// L3: Age for long-term data

let l1 = Ttl::new(dict, Duration::from_secs(60));
let l2 = Lru::new(Ttl::new(dict, Duration::from_secs(3600)));
let l3 = Age::new(dict);
```

## Accessor Methods

Eviction wrappers and FuzzyMultiMap provide accessor methods to retrieve underlying components.

### dictionary() - Access Underlying Dictionary

**Available on**: All eviction wrappers and `FuzzyMultiMap`

**Signature**:
```rust
pub fn dictionary(&self) -> &D
```

**Returns**: Reference to the wrapped dictionary

**Use Cases**:
- Access dictionary-specific methods
- Perform bulk operations on the dictionary
- Query dictionary metadata (size, structure)
- Clone the dictionary for external processing

**Example with Eviction Wrapper**:
```rust
use liblevenshtein::cache::eviction::Lru;
use libdictenstein::pathmap::PathMapDictionary;

let dict = PathMapDictionary::from_terms(vec!["test", "testing", "tested"]);
let cached = Lru::new(dict); // LRU eviction wrapper

// Access the underlying dictionary
let inner_dict = cached.dictionary();
assert_eq!(inner_dict.term_count(), 3);
assert!(inner_dict.contains("test"));

// Clone for external use
let dict_clone = cached.dictionary().clone();
// Now you can work with the dictionary independently
```

**Example with FuzzyMultiMap**:
```rust
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use libdictenstein::dynamic_dawg::DynamicDawg;
use liblevenshtein::transducer::Algorithm;
use std::collections::HashSet;

let dict: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
dict.insert_with_value("hello", HashSet::from([1, 2]));

let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

// Access the dictionary
let inner_dict = fuzzy_map.dictionary();
assert!(inner_dict.contains("hello"));
assert_eq!(inner_dict.term_count(), 1);

// Perform dictionary maintenance (e.g., compaction for DynamicDawg)
let dict_clone = fuzzy_map.dictionary().clone();
if dict_clone.needs_compaction() {
    dict_clone.compact();
}
```

### algorithm() - Get Algorithm (FuzzyMultiMap only)

**Available on**: `FuzzyMultiMap`

**Signature**:
```rust
pub fn algorithm(&self) -> Algorithm
```

**Returns**: The Levenshtein algorithm being used

**Example**:
```rust
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use liblevenshtein::transducer::Algorithm;

let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Transposition);
assert_eq!(fuzzy.algorithm(), Algorithm::Transposition);
```

### Composed Wrapper Access

When composing multiple eviction strategies, you can access each layer:

```rust
use liblevenshtein::cache::eviction::{Lru, Ttl};

let dict = PathMapDictionary::from_terms(vec!["test"]);
let with_ttl = Ttl::new(dict, Duration::from_secs(60));
let with_lru = Lru::new(with_ttl);

// Access outermost layer
let ttl_wrapper = with_lru.dictionary();

// Access inner dictionary
let inner_dict = ttl_wrapper.dictionary();
assert!(inner_dict.contains("test"));

// Or chain in one go
let dict_ref = with_lru.dictionary().dictionary();
```

---

## Testing Eviction Policies

### Basic Eviction Test

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

#[test]
fn test_lru_eviction() {
    let dict = PathMapDictionary::from_terms_with_values([
        ("a", 1),
        ("b", 2),
        ("c", 3),
    ]);

    let lru = Lru::new(dict);

    // Access order: a, b, c
    lru.get_value("a");
    std::thread::sleep(std::time::Duration::from_millis(10));

    lru.get_value("b");
    std::thread::sleep(std::time::Duration::from_millis(10));

    lru.get_value("c");

    // LRU should be "a" (oldest access)
    let lru_entry = lru.find_lru(&["a", "b", "c"]);
    assert_eq!(lru_entry, Some("a".to_string()));
}
```

### Composition Test

```rust
use liblevenshtein::cache::eviction::{Lru, Ttl};
use liblevenshtein::prelude::*;
use std::time::Duration;

#[test]
fn test_ttl_lru_composition() {
    let dict = PathMapDictionary::from_terms_with_values([
        ("foo", 42),
    ]);

    let ttl = Ttl::new(dict, Duration::from_millis(100));
    let lru = Lru::new(ttl);

    // Access before expiration
    assert_eq!(lru.get_value("foo"), Some(42));

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(150));

    // Entry expired by TTL layer
    assert_eq!(lru.get_value("foo"), None);
}
```

## Related Documentation

- **[LRU Policy](./eviction-policies/lru.md)** - Least Recently Used implementation
- **[LFU Policy](./eviction-policies/lfu.md)** - Least Frequently Used implementation
- **[TTL Policy](./eviction-policies/ttl.md)** - Time-to-live expiration
- **[CostAware Policy](./eviction-policies/cost-aware.md)** - Cost-benefit balancing
- **[MemoryPressure Policy](./eviction-policies/memory-pressure.md)** - Memory-aware eviction

## References

1. Source code:
   - `src/cache/mod.rs` - Cache module entry point
   - `src/cache/eviction/mod.rs` - Eviction wrappers overview
   - `src/cache/eviction/*.rs` - Individual policy implementations

2. Decorator Pattern: Gang of Four Design Patterns

3. Caching Strategies:
   - Belady, L. A. (1966). "A Study of Replacement Algorithms for Virtual-Storage Computer"
   - O'Neil, E. J., et al. (1993). "The LRU-K Page Replacement Algorithm"

## Quick Reference

```rust
// Core Wrappers
use liblevenshtein::cache::eviction::{Noop, LazyInitDefault};

// Time-Based
use liblevenshtein::cache::eviction::{Ttl, Age, Lru};

// Frequency-Based
use liblevenshtein::cache::eviction::Lfu;

// Cost-Based
use liblevenshtein::cache::eviction::{CostAware, MemoryPressure};

// Basic Usage
let dict = PathMapDictionary::from_terms_with_values([("key", "value")]);
let cached = Lru::new(dict);
cached.get_value("key");

// Composition
let composed = Lru::new(Ttl::new(dict, Duration::from_secs(300)));
```
