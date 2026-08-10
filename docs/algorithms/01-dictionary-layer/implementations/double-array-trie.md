# DoubleArrayTrie Implementation

**Navigation**: [← Dictionary Layer](../README.md) | [Algorithms Home](../../README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory](#theory)
3. [Data Structure](#data-structure)
4. [Construction Algorithm](#construction-algorithm)
5. [Query Operations](#query-operations)
6. [Value Storage](#value-storage)
7. [Usage Examples](#usage-examples)
8. [Performance Analysis](#performance-analysis)
9. [Advanced Topics](#advanced-topics)
10. [References](#references)

## Overview

`DoubleArrayTrie` (DAT) is the **recommended default dictionary** for most applications in liblevenshtein. It provides exceptional performance for fuzzy matching queries through a cache-efficient array-based representation of trie structures.

### Key Advantages

- ⚡ **3x faster** queries than DAWG-based dictionaries
- 💾 **~8 bytes per state** - minimal memory footprint
- 🎯 **Cache-efficient** - sequential array access patterns
- 🔧 **Append-only updates** - can add terms at runtime
- 🔒 **Thread-safe** - safe for concurrent queries

### When to Use

✅ **Use DoubleArrayTrie when:**
- You need the best overall performance
- Memory efficiency is important
- Dictionary is mostly static or append-only
- Working primarily with ASCII/Latin-1 text
- You want the simplest, most reliable choice

⚠️ **Consider alternatives when:**
- You need to remove terms frequently → Use `DynamicDawg`
- Working with Unicode text → Use `DoubleArrayTrieChar`
- Need substring matching → Use `SuffixAutomaton`

## Theory

### The Double-Array Algorithm

The double-array trie algorithm, invented by Jun-ichi Aoe in 1989, represents a trie using two parallel arrays: **BASE** and **CHECK**.

#### Traditional Trie Problems

Standard trie implementations use pointer-based nodes:

```
Traditional Node (32-64 bytes):
┌─────────────┬──────────────────────────┐
│  is_final   │  children: HashMap/Vec   │
└─────────────┴──────────────────────────┘
      1 byte           24-56 bytes
```

**Problems**:
1. High memory overhead per node
2. Poor cache locality (pointer chasing)
3. Unpredictable memory access patterns

#### Double-Array Solution

Instead of pointers, represent the trie using two integer arrays:

```
BASE[s] + c = t    (transition from state s via character c to state t)
CHECK[t] = s       (verify that state t came from state s)
```

**Advantages**:
1. Constant-time transitions: $`\mathcal{O}(1)`$
2. Sequential memory layout: cache-friendly
3. Predictable access patterns: CPU prefetcher-friendly
4. Compact representation: ~8 bytes per state

### How It Works

Consider a trie with these terms: `["cat", "car", "card"]`

```
Traditional Trie:
       (root)
         |
         c
         |
         a
        / \
       t   r
           |
           d
```

#### BASE Array

`BASE[s]` stores an offset for state `s`. To transition via character `c`:

```
next_state = BASE[current_state] + char_code(c)
```

#### CHECK Array

`CHECK[t]` validates the transition. If `CHECK[next_state] == current_state`, the transition is valid.

#### Example Walkthrough

For the term `"car"`:

```
State 0 (root):
  BASE[0] = 100

Transition 'c' (99):
  next = BASE[0] + 99 = 100 + 99 = 199
  CHECK[199] = 0 ✓ (valid)
  current = 199

State 199:
  BASE[199] = 200

Transition 'a' (97):
  next = BASE[199] + 97 = 200 + 97 = 297
  CHECK[297] = 199 ✓ (valid)
  current = 297

State 297:
  BASE[297] = 300

Transition 'r' (114):
  next = BASE[297] + 114 = 300 + 114 = 414
  CHECK[414] = 297 ✓ (valid)
  is_final[414] = true ✓ ("car" is in dictionary)
```

### Conflict Resolution

When inserting edges, we must find BASE values that don't conflict with existing states. This is similar to open addressing in hash tables.

**Collision Example**:
```
Inserting 'a' and 'b' from root:
  BASE[0] = 100

  Insert 'a' (97): state 197 = BASE[0] + 97
  Insert 'b' (98): state 198 = BASE[0] + 98

  Both work! No collision.
```

**If collision occurs**:
```
BASE[0] = 100
State 197 already used by another transition

Solution: Try BASE[0] = 101, 102, ... until no conflicts
```

The construction algorithm finds BASE values that minimize conflicts and array size.

## Data Structure

### Core Components

```rust
pub struct DoubleArrayTrie<V: DictionaryValue = ()> {
    shared: DATShared<V>,
}

pub(crate) struct DATShared<V: DictionaryValue = ()> {
    pub(crate) base: Arc<Vec<i32>>,      // BASE array
    pub(crate) check: Arc<Vec<i32>>,     // CHECK array
    pub(crate) is_final: Arc<Vec<bool>>, // Final state markers
    pub(crate) edges: Arc<Vec<Vec<u8>>>, // Precomputed edge labels
    pub(crate) values: Arc<Vec<Option<V>>>, // Associated values
}
```

### Memory Layout

For a dictionary with N states:

```
┌────────────────┬────────┬─────────────┐
│ Component      │ Size   │ Per State   │
├────────────────┼────────┼─────────────┤
│ BASE array     │ 4N     │ 4 bytes     │
│ CHECK array    │ 4N     │ 4 bytes     │
│ is_final       │ N      │ 1 byte      │
│ edges (avg)    │ ~2N    │ ~2 bytes    │
│ values (none)  │ N      │ 1 byte*     │
├────────────────┼────────┼─────────────┤
│ Total          │ ~10N   │ ~10 bytes   │
└────────────────┴────────┴─────────────┘
```

*When V=(), `Option<()>` is zero-sized

**Example**: 50,000-term dictionary $`\approx`$ 500KB

### Cache Efficiency

The sequential array layout provides excellent cache performance:

```
Query "test" - Memory Access Pattern:
┌────────────────────────────────────┐
│ BASE[0]                            │ ← Cache line 1
│ CHECK[t_state]                     │ ← Cache line 2 (prefetched)
│ BASE[t_state]                      │ ← Cache line 2
│ CHECK[te_state]                    │ ← Cache line 3 (prefetched)
│ ...                                │
└────────────────────────────────────┘

Traditional trie pointer chasing:
Node* root → Node* t → Node* te → Node* tes → Node* test
  ↑           ↑          ↑           ↑           ↑
Random       Random     Random      Random     Random
address      address    address     address    address
(cache miss) (cache miss) (cache miss) (cache miss) (cache miss)
```

## Construction Algorithm

### Overview

Building a DoubleArrayTrie involves:

1. **Collect and sort terms** - lexicographic order
2. **Build suffix trie** - group terms by common prefixes
3. **Allocate states** - find conflict-free BASE values
4. **Populate arrays** - fill BASE, CHECK, is_final

### Algorithm Steps

```rust
pub fn from_terms<I, S>(terms: I) -> Self
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // Step 1: Collect and sort terms
    let mut terms: Vec<Vec<u8>> = terms
        .into_iter()
        .map(|s| s.as_ref().bytes().collect())
        .collect();

    terms.sort_unstable();
    terms.dedup();  // Remove duplicates

    // Step 2: Build via incremental construction
    let mut builder = DoubleArrayTrieBuilder::new();
    for term in terms {
        builder.insert(&term, ());
    }

    builder.build()
}
```

### Incremental Construction

The builder maintains arrays and grows them as needed:

```rust
impl<V: DictionaryValue> DoubleArrayTrieBuilder<V> {
    fn insert(&mut self, term: &[u8], value: V) {
        let mut state = 0;  // Start at root

        for &byte in term {
            // Find or create transition
            state = match self.get_transition(state, byte) {
                Some(next) => next,
                None => self.add_transition(state, byte),
            };
        }

        // Mark as final and store value
        self.is_final[state] = true;
        self.values[state] = Some(value);
    }

    fn add_transition(&mut self, from: usize, label: u8) -> usize {
        // Find a BASE value that avoids conflicts, growing storage until
        // the local transition set can be represented.
        let base = loop {
            if let Some(base) = self.find_base(from, label) {
                break base;
            }
            let next_len = self
                .check
                .len()
                .saturating_mul(2)
                .max(self.check.len().saturating_add(256))
                .max(256);
            self.grow_arrays(next_len);
        };

        let to = base + usize::from(label);
        if to >= self.base.len() {
            self.grow_arrays(to + 1);
        }

        self.base[from] = i32::try_from(base).expect("BASE index fits i32 storage");
        self.check[to] = i32::try_from(from).expect("CHECK parent fits i32 storage");
        self.edges[from].push(label);

        to
    }

    fn find_base(&self, state: usize, new_label: u8) -> Option<usize> {
        // Get existing labels from this state
        let existing_labels = &self.edges[state];

        // Try base values that fit in the currently allocated arrays.
        for base in state..self.check.len() {
            // Check if this base works for all labels
            let works = existing_labels
                .iter()
                .copied()
                .chain(std::iter::once(new_label))
                .all(|label| {
                    let Some(target) = base.checked_add(usize::from(label)) else {
                        return false;
                    };
                    target < self.check.len() && self.check[target] < 0
                });

            if works {
                return Some(base);
            }
        }

        None
    }
}
```

### Complexity Analysis

- **Time**: $`\mathcal{O}(N \times L \times M)`$ where:
  - N = number of terms
  - L = average term length
  - M = average branching factor (~2-3 for natural language)

- **Space**: $`\mathcal{O}(S)`$ where S = number of states
  - Typically $`S \approx 0.5N`$ to $`2N`$ depending on prefix sharing

### Optimization: Sorted Insertion

Inserting terms in lexicographic order improves locality:

```rust
// Good: Sequential state allocation
["abc", "abd", "abe"]  → states 0→1→2→3, 0→1→2→4, 0→1→2→5

// Bad: Scattered allocation
["abe", "abc", "abd"]  → may require relocation/growth
```

## Query Operations

### Exact Match (contains)

```rust
fn contains(&self, term: &str) -> bool {
    let mut state = 0;  // Start at root

    for byte in term.bytes() {
        // Attempt transition
        let base = self.shared.base[state];
        if base < 0 {
            return false;  // No outgoing edges
        }

        let next = (base as usize) + (byte as usize);

        // Validate transition
        if next >= self.shared.check.len() ||
           self.shared.check[next] != state as i32 {
            return false;  // Invalid transition
        }

        state = next;
    }

    // Check if final state
    state < self.shared.is_final.len() && self.shared.is_final[state]
}
```

**Complexity**: $`\mathcal{O}(L)`$ where L = term length

**Performance**: ~6.6µs for 10,000-term dictionary

### Fuzzy Matching

Fuzzy matching uses Levenshtein automata to traverse the trie:

```rust
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);

let automaton = LevenshteinAutomaton::new("tset", 1, Algorithm::Standard);

let results: Vec<String> = automaton.query(&dict).collect();
// Returns: ["test"] (transposition distance = 1)
```

**Complexity**: $`\mathcal{O}(L \times D \times B)`$ where:
- L = query length
- D = max distance
- B = average branching factor

**Performance**: ~16.3µs for distance 2, 10,000-term dictionary

See [Levenshtein Automata](../../02-levenshtein-automata/README.md) for details.

### Edge Iteration

Pre-computed edge lists enable efficient iteration:

```rust
impl DictionaryNode for DATNode {
    fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
        let edges = if self.state < self.shared.edges.len() {
            &self.shared.edges[self.state][..]
        } else {
            &[]
        };

        Box::new(edges.iter().filter_map(move |&label| {
            self.transition(label).map(|node| (label, node))
        }))
    }
}
```

**Why pre-compute**: Computing edges on-demand requires scanning all 256 possible bytes, which is ~30x slower.

## Value Storage

### Adding Values to Terms

DoubleArrayTrie supports associating arbitrary values with terms:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

// Create dictionary with scope IDs
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("println", 1),    // Global scope
    ("format", 1),     // Global scope
    ("my_var", 42),    // Local scope
    ("scratch", 42),   // Local scope
]);

// Query specific value
assert_eq!(dict.get_value("my_var"), Some(42));

// Check with predicate
assert!(dict.contains_with_value("scratch", |&scope| scope == 42));
```

### Value Storage Architecture

Values are stored in a parallel array indexed by state:

```
State 0 (root):         value = None
State 197 ('c'):        value = None
State 297 ('ca'):       value = None
State 414 ('car'):      value = Some(42) ← Final state
State 415 ('cart'):     value = Some(99) ← Final state
```

**Memory**: `values: Arc<Vec<Option<V>>>`
- Final states: `Some(value)`
- Non-final states: `None`

### Filtered Queries

Filter by value during traversal for dramatic speedups:

```rust
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("test", 1),
    ("testing", 2),
    ("scratch", 1),
    ("scratchpad", 2),
]);

// Only return results with scope 1
let automaton = LevenshteinAutomaton::new("tst", 2, Algorithm::Standard)
    .with_value_filter(|&scope| scope == 1);

let results: Vec<String> = automaton.query(&dict).collect();
// Returns: ["test", "scratch"] (scope 1 only)
```

**Performance**: 10-100x faster than post-filtering when filters are selective.

See [Value Storage Guide](../../09-value-storage/README.md) for comprehensive documentation.

## Usage Examples

### Example 1: Basic Dictionary

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

// Create from terms
let dict = DoubleArrayTrie::from_terms(vec![
    "algorithm",
    "approximate",
    "automaton",
    "analysis",
]);

// Check membership
assert!(dict.contains("algorithm"));
assert!(!dict.contains("algo"));

// Get size
assert_eq!(dict.len(), Some(4));
```

### Example 2: Append-Only Updates

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;

// Start with initial terms
let mut dict = DoubleArrayTrie::from_terms(vec![
    "initial",
    "terms",
]);

// Add new term at runtime
dict.insert("runtime");

assert!(dict.contains("runtime"));
```

**Note**: `insert()` is append-only. It cannot modify or remove existing terms.

### Example 3: Fuzzy Search

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec![
    "kitten", "sitting", "saturday", "sunday",
]);

// Find terms within distance 2 of "sittin"
let automaton = LevenshteinAutomaton::new("sittin", 2, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["sitting", "kitten"]
```

### Example 4: Value-Based Filtering

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

// Code completion: map identifiers to scope IDs
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("println", 0),      // Built-in
    ("print", 0),        // Built-in
    ("format", 0),       // Built-in
    ("my_function", 1),  // User-defined
    ("my_variable", 1),  // User-defined
    ("temp_var", 2),     // Local scope
]);

// Fuzzy search only in local scope (ID = 2)
let automaton = LevenshteinAutomaton::new("tmpvar", 2, Algorithm::Standard)
    .with_value_filter(|&scope| scope == 2);

let results: Vec<String> = automaton.query(&dict).collect();
// Returns: ["temp_var"] (only local scope)
```

### Example 5: Builder Pattern

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieBuilder;

let mut builder = DoubleArrayTrieBuilder::new();

// Add terms incrementally
builder.insert(b"first", 1);
builder.insert(b"second", 2);
builder.insert(b"third", 3);

// Build final dictionary
let dict = builder.build();

assert_eq!(dict.get_value("second"), Some(2));
```

### Example 6: Thread-Safe Concurrent Queries

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use std::sync::Arc;
use std::thread;

let dict = Arc::new(DoubleArrayTrie::from_terms(vec![
    "concurrent", "thread", "safe", "query",
]));

// Spawn multiple query threads
let handles: Vec<_> = (0..4).map(|i| {
    let dict = Arc::clone(&dict);
    thread::spawn(move || {
        // Each thread can query independently
        dict.contains("thread")
    })
}).collect();

// All threads succeed
for handle in handles {
    assert!(handle.join().unwrap());
}
```

### Example 7: Serialization

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};

let dict = DoubleArrayTrie::from_terms(vec!["save", "load"]);

// Serialize to bytes (`Vec<u8>` implements `Write`)
let mut bytes = Vec::new();
BincodeSerializer::serialize(&dict, &mut bytes).unwrap();
std::fs::write("dict.bin", &bytes).unwrap();

// Deserialize (`&[u8]` implements `Read`)
let bytes = std::fs::read("dict.bin").unwrap();
let loaded: DoubleArrayTrie = BincodeSerializer::deserialize(&bytes[..]).unwrap();

assert!(loaded.contains("save"));
```

### Example 8: Large Dictionary

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use std::fs;

// Load dictionary from file (e.g., /usr/share/dict/words)
let words: Vec<String> = fs::read_to_string("/usr/share/dict/words")
    .unwrap()
    .lines()
    .map(|s| s.to_lowercase())
    .collect();

println!("Loading {} words...", words.len());

let start = std::time::Instant::now();
let dict = DoubleArrayTrie::from_terms(words);
println!("Built in {:?}", start.elapsed());
// Typical output: "Built in 150ms" for ~100K words

// Fast queries
let start = std::time::Instant::now();
assert!(dict.contains("algorithm"));
println!("Query took {:?}", start.elapsed());
// Typical output: "Query took 2µs"
```

## Performance Analysis

### Benchmark Results

#### Construction (10,000 terms)

```
DoubleArrayTrie:     3.2ms
DynamicDawg:         4.1ms  (+28%)
PathMapDictionary:   3.5ms  (+9%)
```

**Insight**: DAT has fast construction, especially for sorted inputs.

#### Exact Match (single query)

```
DoubleArrayTrie:     6.6µs
PathMapDictionary:   71.1µs (+977%)
```

**Insight**: Array-based access is 3-10x faster than pointer-based.

#### Contains Check (100 sequential queries)

```
DoubleArrayTrie:     0.22µs per check
PathMapDictionary:   132µs  (+59900%)
```

**Insight**: Cache locality matters enormously for repeated queries.

#### Fuzzy Search (max distance 1)

```
DoubleArrayTrie:     12.9µs
PathMapDictionary:   888µs  (+6800%)
```

#### Fuzzy Search (max distance 2)

```
DoubleArrayTrie:     16.3µs
PathMapDictionary:   5,919µs (+36200%)
```

**Insight**: Performance advantage grows with search complexity.

### Memory Usage

#### Per-State Memory (measured)

```
DoubleArrayTrie:     ~8 bytes
DoubleArrayTrieChar: ~12 bytes (char labels)
DynamicDawg:         ~24 bytes
PathMapDictionary:   ~32 bytes
```

#### Real Dictionary Examples

**100K words (e.g., English dictionary)**:
- DoubleArrayTrie: ~800 KB
- PathMapDictionary: ~3.2 MB

**1M entries (e.g., product database)**:
- DoubleArrayTrie: ~8 MB
- PathMapDictionary: ~32 MB

### Scaling Characteristics

```
Dictionary Size  │  Construction  │  Query Time  │  Memory
─────────────────┼────────────────┼──────────────┼──────────
1,000 terms      │  0.3ms         │  5.1µs       │  80 KB
10,000 terms     │  3.2ms         │  6.6µs       │  800 KB
100,000 terms    │  35ms          │  7.8µs       │  8 MB
1,000,000 terms  │  420ms         │  9.2µs       │  80 MB
```

**Observations**:
- Construction: $`\mathcal{O}(N \log N)`$ due to sorting
- Query: $`\mathcal{O}(L)`$ - independent of dictionary size!
- Memory: Linear with term count

### CPU Cache Impact

Measured on typical modern CPU (32KB L1, 256KB L2, 8MB L3):

```
Working Set Size  │  Cache Level  │  Query Time
──────────────────┼───────────────┼──────────────
< 32 KB           │  L1           │  5.2µs
< 256 KB          │  L2           │  6.8µs
< 8 MB            │  L3           │  8.1µs
> 8 MB            │  RAM          │  12.3µs
```

**Takeaway**: DAT benefits massively from cache locality.

### Comparison: DAT vs a classic pointer-based DAWG

The right column characterizes the **classic static minimized DAWG** (the pre-0.9.x
`DawgDictionary`, since **removed** — see the [layer overview](../README.md#removed-backends-historical-note)).
It is retained here only to illustrate the array-based vs pointer-based access tradeoff;
for a *current* pointer-based backend that additionally supports updates, see
[`DynamicDawg`](dynamic-dawg.md).

| Aspect | DoubleArrayTrie | Classic static DAWG (removed) |
|--------|-----------------|-------------------------------|
| **Access Pattern** | Sequential arrays | Pointer chasing |
| **Cache Locality** | Excellent | Poor |
| **Query Time** | 6.6µs | 19.8µs |
| **Memory/State** | 8 bytes | 16 bytes |
| **Construction** | 3.2ms | 7.2ms |
| **Updates** | Append-only | Static |

**Verdict**: DAT wins on all metrics for fuzzy matching workloads against the classic
static DAWG; against `DynamicDawg` it trades update support for raw read speed and memory.

## Advanced Topics

### Custom Value Types

Any type implementing `DictionaryValue` can be stored:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Metadata {
    frequency: u32,
    category: String,
    timestamp: u64,
}

impl libdictenstein::DictionaryValue for Metadata {}

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("term1", Metadata {
        frequency: 100,
        category: "common".into(),
        timestamp: 1234567890,
    }),
]);
```

**Constraint**: `V: Clone + Send + Sync + 'static`

### Incremental Updates

For append-only use cases, use the builder:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieBuilder;
use std::sync::{Arc, RwLock};

struct AppendOnlyDict {
    dict: Arc<RwLock<DoubleArrayTrie>>,
}

impl AppendOnlyDict {
    fn new(initial: Vec<&str>) -> Self {
        let dict = DoubleArrayTrie::from_terms(initial);
        Self {
            dict: Arc::new(RwLock::new(dict)),
        }
    }

    fn add_term(&self, term: &str) {
        // Rebuild with new term (copy-on-write pattern)
        let mut dict = self.dict.write().unwrap();

        // Extract existing terms + new term
        // (In practice, maintain a separate term list)
        let mut all_terms = vec![term.to_string()];
        // ... add existing terms

        *dict = DoubleArrayTrie::from_terms(all_terms);
    }
}
```

**Note**: For frequent updates, consider `DynamicDawg` instead.

### Zipper Navigation

Use zippers for hierarchical navigation with value access:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("test", 1),
    ("testing", 2),
]);

let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Navigate step by step
let z = zipper.descend(b't')
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b's'))
    .and_then(|z| z.descend(b't'))
    .unwrap();

assert!(z.is_final());
assert_eq!(z.value(), Some(1));

// Continue navigation
let z2 = z.descend(b'i')
    .and_then(|z| z.descend(b'n'))
    .and_then(|z| z.descend(b'g'))
    .unwrap();

assert_eq!(z2.value(), Some(2));

// Get path
let path = z2.path();
assert_eq!(path, b"testing");
```

See [Zipper Navigation](../../06-zipper-navigation/README.md) for details.

### Integration with External Systems

#### Redis-backed Dictionary

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use redis::Commands;

fn load_from_redis() -> DoubleArrayTrie {
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();

    let terms: Vec<String> = con.smembers("dictionary:terms").unwrap();
    DoubleArrayTrie::from_terms(terms)
}

fn save_to_redis(dict: &DoubleArrayTrie) {
    // `use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};`
    let mut bytes = Vec::new();
    BincodeSerializer::serialize(dict, &mut bytes).unwrap();
    let client = redis::Client::open("redis://127.0.0.1/").unwrap();
    let mut con = client.get_connection().unwrap();
    let _: () = con.set("dictionary:dat", bytes).unwrap();
}
```

#### Database-backed Dictionary

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use sqlx::PgPool;

async fn load_from_postgres(pool: &PgPool) -> DoubleArrayTrie<u32> {
    let rows: Vec<(String, i32)> = sqlx::query_as(
        "SELECT term, category_id FROM dictionary ORDER BY term"
    )
    .fetch_all(pool)
    .await
    .unwrap();

    let terms: Vec<(&str, u32)> = rows.iter()
        .map(|(term, id)| (term.as_str(), *id as u32))
        .collect();

    DoubleArrayTrie::from_terms_with_values(terms)
}
```

### Memory-Mapped Files

For very large dictionaries, use memory mapping:

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::serialization::{BincodeSerializer, DictionarySerializer};
use memmap2::Mmap;
use std::fs::File;

// Save dictionary
let dict = DoubleArrayTrie::from_terms(load_huge_wordlist());
let mut bytes = Vec::new();
BincodeSerializer::serialize(&dict, &mut bytes).unwrap();
std::fs::write("huge_dict.bin", &bytes).unwrap();

// Memory-map the file, then decode straight from the mapping (`&[u8]: Read`),
// so the encoded image is never additionally buffered on the heap.
let file = File::open("huge_dict.bin").unwrap();
let mmap = unsafe { Mmap::map(&file).unwrap() };
let dict: DoubleArrayTrie = BincodeSerializer::deserialize(&mmap[..]).unwrap();
// NOTE: `dict` is an owned dictionary rebuilt from the mapping — it does not
// borrow the mapped pages. Memory-mapping avoids buffering the *file* in RAM;
// it is not a zero-copy view of the dictionary itself.
```

**Benefits**:
- The encoded image is never buffered on the heap — bincode decodes straight from the mapping
- OS manages paging of the mapped file
- Multiple processes can share the mapped pages of the *encoded file*

> The decoded `DoubleArrayTrie` is still an owned, heap-resident structure. For a dictionary
> that genuinely reads from disk without rebuilding it in memory, use the disk-persisted
> [`PersistentARTrie`](../../../user-guide/backends.md) family instead.

## References

### Academic Papers

1. **Aoe, J. (1989)**. "An Efficient Digital Search Algorithm by Using a Double-Array Structure"
   - *IEEE Transactions on Software Engineering*, 15(9), 1066-1077
   - DOI: [10.1109/32.31365](https://doi.org/10.1109/32.31365)
   - 📄 **Original double-array algorithm**

2. **Yata, S., Oono, M., Morita, K., Fuketa, M., Sumitomo, T., & Aoe, J. (2007)**. "A compact static double-array keeping character codes"
   - *Information Processing & Management*, 43(1), 237-247
   - DOI: [10.1016/j.ipm.2006.06.001](https://doi.org/10.1016/j.ipm.2006.06.001)
   - 📄 **Optimization techniques**

3. **Yata, S., Morita, K., Fuketa, M., & Aoe, J. (2008)**. "Fast String Matching with Space-Efficient Word Graphs"
   - *Innovations in Information Technology*, 79-83
   - DOI: [10.1109/innovations.2008.4781726](https://doi.org/10.1109/innovations.2008.4781726)
   - 📄 **Space optimizations**

### Open Access Resources

4. **Linux-Thailand Double Array Trie**
   - 📄 [http://linux.thai.net/~thep/datrie/](http://linux.thai.net/~thep/datrie/)
   - Excellent tutorial with diagrams

5. **CP-Algorithms: Aho-Corasick Algorithm**
   - 📄 [https://cp-algorithms.com/string/aho_corasick.html](https://cp-algorithms.com/string/aho_corasick.html)
   - Related trie-based algorithms

### Implementation References

6. **libdatrie** (C implementation)
   - 📦 [https://github.com/tlwg/libdatrie](https://github.com/tlwg/libdatrie)
   - Production-quality C library

7. **Darts (Double-ARray Trie System)**
   - 📦 [https://github.com/s-yata/darts-clone](https://github.com/s-yata/darts-clone)
   - High-performance C++ implementation

### Textbooks

8. **Gusfield, D. (1997)**. *Algorithms on Strings, Trees, and Sequences*
   - Cambridge University Press, Chapter 8
   - ISBN: 978-0521585194

## Next Steps

- **Unicode Support**: Learn about [DoubleArrayTrieChar](double-array-trie-char.md)
- **Dynamic Updates**: Explore [DynamicDawg](dynamic-dawg.md)
- **Value Storage**: Read the [Value Storage Guide](../../09-value-storage/README.md)
- **Fuzzy Matching**: Understand [Levenshtein Automata](../../02-levenshtein-automata/README.md)
- **Navigation**: Try [Zipper Pattern](../../06-zipper-navigation/README.md)

---

**Navigation**: [← Dictionary Layer](../README.md) | [Algorithms Home](../../README.md)
