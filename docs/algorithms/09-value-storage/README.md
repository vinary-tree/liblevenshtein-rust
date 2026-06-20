# Layer 9: Value Storage Architecture

**Associate arbitrary data with dictionary terms for fuzzy maps, scope-aware completion, and filtered queries.**

---

## Overview

Value storage enables dictionaries to map terms to associated data, transforming them into "fuzzy maps" - approximate key-value stores with Levenshtein distance tolerance. This is essential for:

- **Code completion** with scope IDs
- **Categorization** and metadata
- **Filtered queries** with 10-100x speedup
- **Contextual search** (scope-aware, type-aware)

**Key Insight:** Values are stored per state, not per term. This allows efficient access during graph traversal, enabling filtering **during** search rather than after (10-100x faster for selective filters).

![Value-filtered pruning: a value predicate is evaluated at each state during traversal, so non-matching subtrees are pruned before their terms are enumerated](../../diagrams/traversal/value-filtered-pruning.svg)

*Value-filtered pruning: state-indexed values let the traversal skip non-matching subtrees mid-walk.*

---

## Architecture

### Core Concept: State-Indexed Storage

```
┌─────────────────────────────────────────────────────┐
│  Terms                                               │
│  ↓ (via graph transitions)                          │
│  States                                              │
│  ↓ (via state index)                                │
│  Values                                              │
└─────────────────────────────────────────────────────┘

Example for DoubleArrayTrie:

State:  0     1     2     3     4     5     6
Term:   -   "a"  "ap" "app"    -  "apple" "apply"
Final:  F     F     F     T     F     T      T
Value:  N     N     N     N     N   Some(1) Some(3)
        ↑                             ↑
        |                             |
    root (not final)          final state with value
```

**Key Properties:**
- Values stored in `Arc<Vec<Option<V>>>`
- Indexed by state number (parallel to `is_final`)
- Only final states can have `Some(value)`
- Non-final states have `None`
- Cloned on access (Rust ownership)

---

## Memory Layout

### DoubleArrayTrie Value Storage

```rust
pub(crate) struct DATShared<V: DictionaryValue = ()> {
    pub(crate) base: Arc<Vec<i32>>,       // Transition base offsets
    pub(crate) check: Arc<Vec<i32>>,      // Parent verification
    pub(crate) is_final: Arc<Vec<bool>>,  // Final state markers
    pub(crate) edges: Arc<Vec<Vec<u8>>>,  // Outgoing edges
    pub(crate) values: Arc<Vec<Option<V>>>,  // ← NEW: Value storage
}
```

**Memory Overhead:**
- +8 bytes per state on 64-bit systems (Arc pointer size)
- `Option<V>` overhead: 1 byte discriminant + sizeof(V)
- Total: ~9-16 bytes per state depending on V

**Example with Concrete Type:**

```rust
// V = i32 (4 bytes)
values: Arc<Vec<Option<i32>>>
// Memory per state: 1 (discriminant) + 4 (i32) = 5 bytes
// + padding = 8 bytes

// V = String (24 bytes on 64-bit)
values: Arc<Vec<Option<String>>>
// Memory per state: 1 + 24 = 25 bytes
// + padding = 32 bytes
```

---

## Traits

### MappedDictionary Trait

```rust
pub trait MappedDictionary: Dictionary {
    type Value: DictionaryValue;

    /// Get the value associated with a term.
    ///
    /// Returns `None` if term doesn't exist.
    fn get_value(&self, term: &str) -> Option<Self::Value>;

    /// Check if term exists with value matching predicate.
    ///
    /// More efficient than get_value() + predicate test.
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool;
}
```

### MappedDictionaryNode Trait

```rust
pub trait MappedDictionaryNode: DictionaryNode {
    type Value: DictionaryValue;

    /// Get the value at this node if it's final.
    ///
    /// Returns `None` if not final or no value.
    fn value(&self) -> Option<Self::Value>;
}
```

### ValuedDictZipper Trait

```rust
pub trait ValuedDictZipper: DictZipper {
    type Value: DictionaryValue;

    /// Get value at current position if final.
    fn value(&self) -> Option<Self::Value>;
}
```

### DictionaryValue Constraint

```rust
pub trait DictionaryValue: Clone + Send + Sync + 'static {}

// Auto-implemented for all qualifying types
impl<T: Clone + Send + Sync + 'static> DictionaryValue for T {}
```

**Common Value Types:**
- `i32`, `u32`, `usize` - Scope IDs, categories
- `String`, `&'static str` - Labels, descriptions
- `f32`, `f64` - Scores, weights
- Custom structs - Complex metadata

---

## Usage Examples

### Basic Value Storage

```rust
use liblevenshtein::prelude::*;

// Create dictionary with values
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("apple", 1),
    ("banana", 2),
    ("cherry", 3),
]);

// Get value for exact term
assert_eq!(dict.get_value("apple"), Some(1));
assert_eq!(dict.get_value("unknown"), None);

// Check with predicate
assert!(dict.contains_with_value("apple", |v| *v > 0));
assert!(!dict.contains_with_value("apple", |v| *v > 5));
```

### Unicode Values

```rust
// Character-level dictionary with values
let dict = DoubleArrayTrieChar::from_terms_with_values(vec![
    ("café", "coffee"),
    ("中文", "Chinese"),
    ("🎉", "party"),
]);

assert_eq!(dict.get_value("café"), Some("coffee"));
assert_eq!(dict.get_value("中文"), Some("Chinese"));
assert_eq!(dict.get_value("🎉"), Some("party"));
```

### Builder Pattern

```rust
let mut builder = DoubleArrayTrieBuilder::new();

// Insert terms with values
builder.insert_with_value("apple", Some(1));
builder.insert_with_value("banana", Some(2));

// Insert term without value (for backward compatibility)
builder.insert("cherry");  // value will be None

let dict = builder.build();
```

### Fuzzy Search with Values

```rust
// Basic fuzzy search returns (term, value) pairs
let results: Vec<(String, i32)> = dict
    .fuzzy_search_with_values("aple", 2)
    .collect();

// Filter results after search (slower)
let filtered: Vec<(String, i32)> = results
    .into_iter()
    .filter(|(_, v)| *v < 3)
    .collect();
```

### Value-Filtered Query (10-100x Faster!)

```rust
// Filter DURING traversal - prunes search space early
let results: Vec<(String, i32)> = dict
    .fuzzy_search_filtered("aple", 2, |v| *v < 3)
    .collect();

// Benchmark comparison:
// Post-filter:    100ms  (explores entire search space)
// During-filter:   10ms  (10% selectivity = 10x speedup)
// During-filter:    1ms  (1% selectivity = 100x speedup)
```

### Zipper Navigation with Values

```rust
use liblevenshtein::dictionary::zipper::*;

let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Navigate to a term
let z = zipper
    .descend(b'a')
    .and_then(|z| z.descend(b'p'))
    .and_then(|z| z.descend(b'p'))
    .and_then(|z| z.descend(b'l'))
    .and_then(|z| z.descend(b'e'))
    .unwrap();

// Access value at current position
if z.is_final() {
    println!("Found term: {:?}", String::from_utf8_lossy(&z.path()));
    println!("Value: {:?}", z.value());  // Some(1)
}

// Iterate children with values
for (label, child) in z.children() {
    if child.is_final() {
        println!("{}: {:?}", label as char, child.value());
    }
}
```

---

## Use Cases

### 1. Code Completion with Scope IDs

**Problem:** IDE needs to suggest only symbols visible in current scope.

**Solution:** Store scope IDs as values, filter during query.

```rust
// Symbol table with scope IDs
let symbols = DoubleArrayTrie::from_terms_with_values(vec![
    // Current scope (id=1)
    ("myFunction", 1),
    ("myVariable", 1),
    ("localHelper", 1),

    // Parent scope (id=2)
    ("parentFunction", 2),
    ("globalVar", 2),

    // Sibling scope (id=3)
    ("siblingFunc", 3),
]);

// Get completions for current scope only
fn get_completions(
    symbols: &DoubleArrayTrie<usize>,
    query: &str,
    max_distance: usize,
    current_scope: usize,
) -> Vec<String> {
    symbols
        .fuzzy_search_filtered(query, max_distance, |scope_id| {
            *scope_id == current_scope || *scope_id == 2  // Current + parent
        })
        .map(|(term, _)| term)
        .collect()
}

// Usage
let completions = get_completions(&symbols, "myFun", 2, 1);
// Returns: ["myFunction", "myVariable"]
// Filters out: ["siblingFunc"] (wrong scope)

// Performance: 10-100x faster than querying all symbols
// then filtering by scope!
```

### 2. Categorized Dictionary

**Problem:** Search only within specific categories (e.g., fruits vs vegetables).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Fruit = 1,
    Vegetable = 2,
    Grain = 3,
}

let foods = DoubleArrayTrie::from_terms_with_values(vec![
    // Fruits
    ("apple", Category::Fruit),
    ("banana", Category::Fruit),
    ("cherry", Category::Fruit),

    // Vegetables
    ("carrot", Category::Vegetable),
    ("celery", Category::Vegetable),

    // Grains
    ("wheat", Category::Grain),
    ("rice", Category::Grain),
]);

// Search only fruits
let fruit_matches: Vec<String> = foods
    .fuzzy_search_filtered("aple", 2, |cat| *cat == Category::Fruit)
    .map(|(term, _)| term)
    .collect();

// Returns: ["apple"]
// Does NOT return: ["carrot"] (wrong category, even though "aple" → "carrot" has distance 3)
```

### 3. Fuzzy Map (Approximate Key-Value Lookup)

**Problem:** Standard HashMap requires exact keys. Need approximate matching.

```rust
// Product ID → Price mapping with typo tolerance
let products = DoubleArrayTrie::from_terms_with_values(vec![
    ("PROD-12345", 29.99),
    ("PROD-12346", 39.99),
    ("PROD-54321", 49.99),
]);

// Lookup with typo tolerance
fn fuzzy_lookup(
    map: &DoubleArrayTrie<f64>,
    key: &str,
    max_distance: usize,
) -> Option<(String, f64)> {
    map.fuzzy_search_with_values(key, max_distance)
        .min_by_key(|(term, _)| {
            // Prefer exact match, then closest
            levenshtein_distance(key, term)
        })
}

// Usage
let result = fuzzy_lookup(&products, "PROD-12346", 2);
assert_eq!(result, Some(("PROD-12346".to_string(), 39.99)));

// With typo
let result = fuzzy_lookup(&products, "PROD-12346", 2);  // Missing '6'
// Still finds: ("PROD-12346", 39.99)
```

### 4. Metadata Storage

**Problem:** Need to store complex metadata with each term.

```rust
#[derive(Debug, Clone)]
struct SymbolMetadata {
    scope_id: usize,
    type_name: String,
    doc_string: String,
    deprecated: bool,
}

let symbols = DoubleArrayTrie::from_terms_with_values(vec![
    ("myFunction", SymbolMetadata {
        scope_id: 1,
        type_name: "fn() -> i32".to_string(),
        doc_string: "Calculates the answer".to_string(),
        deprecated: false,
    }),
    ("oldFunction", SymbolMetadata {
        scope_id: 1,
        type_name: "fn() -> i32".to_string(),
        doc_string: "Use myFunction instead".to_string(),
        deprecated: true,
    }),
]);

// Filter out deprecated symbols during query
let active_symbols: Vec<(String, SymbolMetadata)> = symbols
    .fuzzy_search_filtered("myFun", 2, |meta| !meta.deprecated)
    .collect();

// Only returns: [("myFunction", ...)]
```

### 5. Hierarchical Scopes (Multi-level Filtering)

```rust
#[derive(Debug, Clone)]
struct ScopeInfo {
    scope_id: usize,
    parent_scopes: Vec<usize>,  // Visibility chain
    visibility: Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Visibility {
    Public,
    Private,
    Protected,
}

let symbols = DoubleArrayTrie::from_terms_with_values(vec![
    ("publicFunc", ScopeInfo {
        scope_id: 1,
        parent_scopes: vec![0],
        visibility: Visibility::Public,
    }),
    ("privateHelper", ScopeInfo {
        scope_id: 1,
        parent_scopes: vec![0],
        visibility: Visibility::Private,
    }),
]);

// Complex visibility check
fn is_visible(info: &ScopeInfo, current_scope: usize) -> bool {
    // Public always visible
    if info.visibility == Visibility::Public {
        return true;
    }

    // Private only visible in same scope
    if info.visibility == Visibility::Private {
        return info.scope_id == current_scope;
    }

    // Protected visible in scope + children
    info.scope_id == current_scope ||
        info.parent_scopes.contains(&current_scope)
}

// Query with complex filter
let visible: Vec<String> = symbols
    .fuzzy_search_filtered("Func", 2, |info| is_visible(info, 1))
    .map(|(term, _)| term)
    .collect();
```

---

## Performance Analysis

### Memory Impact

**Per-State Overhead:**
- `Option<V>`: 1 byte (discriminant) + sizeof(V)
- Typical values:
  - `V = ()`: 1 byte (optimized away by compiler)
  - `V = i32`: 5 bytes → 8 bytes (padding)
  - `V = String`: 25 bytes → 32 bytes (padding)
  - `V = Arc<String>`: 9 bytes → 16 bytes

**Total Memory:**
```
Memory = states × (base_size + value_overhead)

Example (10,000 terms, ~50,000 states):
- Without values: 50,000 × 8 = 400 KB
- With i32 values: 50,000 × 16 = 800 KB  (+400 KB)
- With String values: 50,000 × 40 = 2 MB  (+1.6 MB)
```

### Query Performance

**No Value Access:**
- Zero overhead (values not accessed during traversal)
- Monomorphization optimizes away unused code

**With Value Filtering:**

| Selectivity | Search Space | Post-Filter | During-Filter | Speedup |
|-------------|--------------|-------------|---------------|---------|
| 100% (no filter) | 100% | 100ms | 100ms | 1x |
| 50% | 50% | 100ms | 50ms | **2x** |
| 10% | 10% | 100ms | 10ms | **10x** |
| 1% | 1% | 100ms | 1ms | **100x** |

**Key Insight:** Filtering during traversal prunes the search space early, avoiding exploration of irrelevant subtrees.

**Example Benchmark (10,000 terms, distance 2):**
```rust
// Post-filtering (explores all matches)
let results: Vec<_> = dict
    .fuzzy_search("query", 2)  // 1000 matches
    .filter(|term| matches_scope(term))  // Only 10 relevant
    .collect();
// Time: 100ms

// During-filtering (only explores relevant subtrees)
let results: Vec<_> = dict
    .fuzzy_search_filtered("query", 2, |scope_id| is_relevant(scope_id))
    .collect();  // 10 matches
// Time: 1ms (100x faster!)
```

### Clone Cost

Values are cloned on access. For expensive types, consider:

1. **Arc wrapping:**
```rust
// Instead of String
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("term", Arc::new("expensive data".to_string())),
]);
// Clone cost: 8 bytes (pointer) instead of full string
```

2. **Reference counted metadata:**
```rust
#[derive(Clone)]
struct Metadata {
    scope_id: usize,           // 8 bytes (cheap to clone)
    details: Arc<String>,      // 8 bytes (cheap to clone)
}
```

---

## Implementation Details

### DoubleArrayTrie Value Storage

**Construction:**
```rust
// from_terms_with_values sorts and deduplicates
pub fn from_terms_with_values<I, S>(terms: I) -> Self
where
    I: IntoIterator<Item = (S, V)>,
    S: AsRef<str>,
{
    let mut term_value_pairs: Vec<(String, V)> = terms
        .into_iter()
        .map(|(s, v)| (s.as_ref().to_string(), v))
        .collect();

    term_value_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Duplicates: keep last value
    term_value_pairs.dedup_by(|a, b| {
        if a.0 == b.0 {
            b.1 = a.1.clone();
            true
        } else {
            false
        }
    });

    let mut builder = DoubleArrayTrieBuilder::new();
    for (term, value) in term_value_pairs {
        builder.insert_with_value(&term, Some(value));
    }
    builder.build()
}
```

**Value Retrieval:**
```rust
pub fn get_value(&self, term: &str) -> Option<V> {
    // Navigate to final state
    let mut state = 1; // Root
    for &byte in term.as_bytes() {
        let base = self.shared.base[state];
        if base < 0 {
            return None;
        }
        let next = (base as usize) + (byte as usize);
        if next >= self.shared.check.len()
            || self.shared.check[next] != state as i32 {
            return None;
        }
        state = next;
    }

    // Check if final and return value
    if state < self.shared.is_final.len()
        && self.shared.is_final[state] {
        self.shared.values.get(state).and_then(|v| v.clone())
    } else {
        None
    }
}
```

### DynamicDawg Value Storage

Stores values directly in nodes:

```rust
pub struct DawgNode<V: DictionaryValue = ()> {
    edges: SmallVec<[(u8, usize); 4]>,
    ref_count: usize,
    is_final: bool,
    value: Option<V>,  // ← Stored per node
}
```

**Trade-offs:**
- More flexible (per-node values)
- Slightly higher memory (not arena-allocated)
- Better for dynamic updates

---

## Comparison: Dictionary Value Storage

| Dictionary | Storage Location | Memory Overhead | Access Time | Notes |
|------------|------------------|-----------------|-------------|-------|
| DoubleArrayTrie | State-indexed array | 8-16 bytes/state | `O(1)` | Arena-allocated, cache-friendly |
| DoubleArrayTrieChar | State-indexed array | 8-16 bytes/state | `O(1)` | Same as byte variant |
| DynamicDawg | Per-node field | 8-16 bytes/node | `O(1)` | More flexible |
| DynamicDawgChar | Per-node field | 8-16 bytes/node | `O(1)` | Unicode support |
| PathMap | Persistent structure | Variable | `O(1)` | Structural sharing |

---

## Best Practices

### 1. Choose Value Type Carefully

**Good:**
```rust
// Small, cheap to clone
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("term", 42usize),  // 8 bytes
]);
```

**Better:**
```rust
// Use Arc for expensive data
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("term", Arc::new(ExpensiveData { /* ... */ })),
]);
```

### 2. Use Value Filtering for Selective Queries

**Slow:**
```rust
// Post-filtering
let results: Vec<_> = dict
    .fuzzy_search("query", 2)
    .filter(|term| expensive_check(term))
    .collect();
```

**Fast:**
```rust
// During-traversal filtering
let results: Vec<_> = dict
    .fuzzy_search_filtered("query", 2, |value| cheap_check(value))
    .collect();
```

### 3. Batch Insertions

**Slow:**
```rust
for (term, value) in terms {
    builder.insert_with_value(&term, Some(value));
}
```

**Fast:**
```rust
// from_terms_with_values sorts once
let dict = DoubleArrayTrie::from_terms_with_values(terms);
```

### 4. Consider Backward Compatibility

```rust
// Default type parameter for zero-cost abstraction
type MyDict = DoubleArrayTrie;  // V = ()
type ValuedDict = DoubleArrayTrie<usize>;  // V = usize
```

---

## Related Documentation

- `MappedDictionary` Trait
- [DoubleArrayTrie Implementation](../01-dictionary-layer/implementations/double-array-trie.md)
- Value-Filtered Queries
- Code Completion Use Case
- Performance Benchmarks

---

## Examples

- Basic Value Storage
- Scope-Aware Completion
- Fuzzy Map
- Filtered Query Performance

---

## References

- **Phase 6 Implementation**: `docs/implementation-status/phase-6-dictionary-layer-completeness.md`
- **MappedDictionary Trait**: `src/dictionary/mod.rs:302-326`
- **ValuedDictZipper Trait**: `src/dictionary/zipper.rs`
- **DoubleArrayTrie Values**: `src/dictionary/double_array_trie.rs`
