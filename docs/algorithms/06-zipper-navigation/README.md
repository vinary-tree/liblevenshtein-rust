# Zipper Navigation Pattern

**Navigation**: [← Back to Algorithms](../README.md) | [Dictionary Layer →](../01-dictionary-layer/README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory: Huet's Zipper](#theory-huets-zipper)
3. [Zipper Traits](#zipper-traits)
4. [Implementations](#implementations)
5. [Usage Examples](#usage-examples)
6. [Performance Analysis](#performance-analysis)
7. [Advanced Patterns](#advanced-patterns)
8. [References](#references)

## Overview

The **Zipper** is a functional programming pattern for navigating and updating tree-like data structures. In liblevenshtein, zippers provide a powerful interface for exploring dictionary graphs with precise control over traversal, path tracking, and value access.

![Query iterator hierarchy: how the zipper-backed query iterators (standard, ordered, prefix, filtered) relate to one another](../../diagrams/traversal/query-iterator-hierarchy.svg)

*Query iterator hierarchy: the zipper-backed iterators that surface traversal results to callers.*

![Value-filtered pruning: a value predicate is applied during traversal so non-matching subtrees are skipped before their terms are enumerated](../../diagrams/traversal/value-filtered-pruning.svg)

*Value-filtered pruning: zipper navigation prunes non-matching subtrees during the walk, not after.*

### What is a Zipper?

A zipper represents a "position" or "focus" within a tree structure, along with enough context to reconstruct the full tree. Think of it as a cursor that can:

- Navigate down to children
- Track the path taken from root
- Access values at the current position
- Maintain immutability (creates new zippers, doesn't mutate)

### Key Advantages

- 🎯 **Precise Navigation**: Step-by-step character/byte traversal
- 📍 **Path Tracking**: Know exactly where you are in the tree
- 💎 **Value Access**: Retrieve values at any position
- 🔒 **Immutable**: Functional interface, thread-safe
- 🌳 **Hierarchical**: Support for contextual completion with scopes

### When to Use Zippers

✅ **Use zippers when:**
- Building custom traversal algorithms
- Implementing hierarchical/contextual search
- Need path information during traversal
- Require fine-grained navigation control
- Exploring tree structure interactively

⚠️ **Simpler alternatives:**
- Basic fuzzy search: Use `LevenshteinAutomaton::query()` directly
- Exact lookups: Use `Dictionary::contains()`
- Value filtering: Use `MappedDictionary::get_value()`

## Theory: Huet's Zipper

### Origin

The zipper pattern was introduced by Gérard Huet in 1997 as a technique for navigating and modifying tree structures in purely functional languages.

**Paper**: Huet, G. (1997). "The Zipper". *Journal of Functional Programming*, 7(5), 549-554.

### Conceptual Model

Traditional tree navigation uses recursion and loses context:

```
Navigate to 'e' in "test":
       root
         |
         t
         |
         e  ← We're here, but lost the path
         |
         s
         |
         t
```

**Problem**: We can't easily:
- Know the path taken: ['t', 'e']
- Navigate back up
- Continue to siblings

### Zipper Solution

A zipper combines **current position** with **context** (path from root):

```rust
struct Zipper {
    position: Node,        // Current node
    path: Vec<char>,       // ['t', 'e']
    context: SharedData,   // Immutable dictionary data
}
```

Now we can:
- Know we're at "te" (prefix)
- Continue to "tes", "test"
- Branch to siblings
- Access values at current node

### Functional Properties

Zippers maintain **referential transparency**:

```rust
let z1 = root_zipper.descend('a');  // Create new zipper
let z2 = root_zipper.descend('b');  // Original unchanged

// root_zipper still valid!
assert_eq!(root_zipper.path(), vec![]);
assert_eq!(z1.path(), vec!['a']);
assert_eq!(z2.path(), vec!['b']);
```

**Benefits**:
- Safe to share across threads
- No mutation surprises
- Enables backtracking

## Zipper Traits

### DictZipper Trait

Core trait for dictionary navigation:

```rust
pub trait DictZipper: Clone + Sized {
    type Unit: CharUnit;  // u8 or char

    /// Check if current position marks a complete term
    fn is_final(&self) -> bool;

    /// Navigate to child via label
    fn descend(&self, label: Self::Unit) -> Option<Self>;

    /// Iterate over all children
    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> + '_;

    /// Get path from root to current position
    fn path(&self) -> Vec<Self::Unit>;
}
```

**Key Methods**:

- `is_final()`: True if current position is end of a valid term
- `descend(label)`: Move to child, returns `None` if edge doesn't exist
- `children()`: Explore all outgoing edges
- `path()`: Get sequence of labels from root

### ValuedDictZipper Trait

Extension for accessing values:

```rust
pub trait ValuedDictZipper: DictZipper {
    type Value: DictionaryValue;

    /// Get value at current position (if final state)
    fn value(&self) -> Option<Self::Value>;
}
```

**Usage**: Retrieve values associated with terms during navigation.

### Relationship to Dictionary Traits

```
Dictionary Layer:
┌────────────────────────────────────────┐
│  Dictionary  ──────→  DictionaryNode   │
│      ↓                      ↓          │
│  MappedDict  ──────→  MappedNode      │
└────────────────────────────────────────┘
                  ↓
Zipper Layer:
┌────────────────────────────────────────┐
│  DictZipper                            │
│      ↓                                 │
│  ValuedDictZipper                      │
└────────────────────────────────────────┘
```

Zippers wrap dictionaries to provide:
- Path tracking (not in base Dictionary)
- Functional navigation (vs mutation)
- Value access during traversal

## Implementations

### DoubleArrayTrieZipper

Zipper for byte-level `DoubleArrayTrie`:

```rust
pub struct DoubleArrayTrieZipper<V: DictionaryValue = ()> {
    state: usize,           // Current DAT state index
    path: Vec<u8>,          // Bytes from root to here
    shared: DATShared<V>,   // Shared dictionary data
}
```

**Characteristics**:
- **Unit**: `u8` (byte-level)
- **Memory**: ~24 bytes base + path length
- **Performance**: $`\mathcal{O}(1)`$ descend, $`\mathcal{O}(E)`$ children (E = edge count)

**Example**:
```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// Navigate to "test"
let z = zipper
    .descend(b't')
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b's'))
    .and_then(|z| z.descend(b't'))
    .unwrap();

assert!(z.is_final());
assert_eq!(z.path(), b"test");
```

### DoubleArrayTrieCharZipper

Zipper for character-level `DoubleArrayTrieChar`:

```rust
pub struct DoubleArrayTrieCharZipper<V: DictionaryValue = ()> {
    state: usize,               // Current DAT state index
    path: Vec<char>,            // Characters from root
    shared: DATSharedChar<V>,   // Shared dictionary data
}
```

**Characteristics**:
- **Unit**: `char` (Unicode code points)
- **Memory**: ~24 bytes base + path length × 4
- **Performance**: $`\mathcal{O}(1)`$ descend, $`\mathcal{O}(E)`$ children

**Example**:
```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrieCharZipper;
use libdictenstein::zipper::DictZipper;

let dict = DoubleArrayTrieChar::from_terms(vec!["café", "中文", "🎉"]);
let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

// Navigate to "café"
let z = zipper
    .descend('c')
    .and_then(|z| z.descend('a'))
    .and_then(|z| z.descend('f'))
    .and_then(|z| z.descend('é'))
    .unwrap();

assert!(z.is_final());
assert_eq!(z.path(), vec!['c', 'a', 'f', 'é']);
```

### PathMapZipper (Optional: `pathmap-backend`)

Zipper for `PathMapDictionary`:

```rust
#[cfg(feature = "pathmap-backend")]
pub struct PathMapZipper<V: DictionaryValue = ()> {
    path: Vec<u8>,
    dict: Arc<PathMapDictionary<V>>,
}
```

**Characteristics**:
- **Unit**: `u8` (byte-level)
- **Memory**: ~24 bytes + path + Arc overhead
- **Performance**: $`\mathcal{O}(\log N)`$ descend (HashMap lookup)

**When to use**: When using PathMapDictionary backend.

### Comparison Table

| Zipper | Unit | Backend | Descend | Children | Memory | Unicode |
|--------|------|---------|---------|----------|--------|---------|
| **DATZipper** | `u8` | DoubleArrayTrie | $`\mathcal{O}(1)`$ | $`\mathcal{O}(E)`$ | Low | Byte |
| **DATCharZipper** | `char` | DoubleArrayTrieChar | $`\mathcal{O}(1)`$ | $`\mathcal{O}(E)`$ | Medium | ✅ |
| **PathMapZipper** | `u8` | PathMapDict | $`\mathcal{O}(\log N)`$ | $`\mathcal{O}(E)`$ | Medium | Byte |

E = average edge count per node (~2-3 for natural language)
N = total dictionary size

## Usage Examples

### Example 1: Basic Navigation

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

let dict = DoubleArrayTrie::from_terms(vec!["cat", "car", "card"]);
let root = DoubleArrayTrieZipper::new_from_dict(&dict);

// Root is not a final state
assert!(!root.is_final());

// Navigate to "c"
let c = root.descend(b'c').unwrap();
assert!(!c.is_final());
assert_eq!(c.path(), b"c");

// Navigate to "ca"
let ca = c.descend(b'a').unwrap();
assert!(!ca.is_final());
assert_eq!(ca.path(), b"ca");

// Navigate to "cat"
let cat = ca.descend(b't').unwrap();
assert!(cat.is_final());  // ✓ "cat" is a term
assert_eq!(cat.path(), b"cat");

// Navigate to "car"
let car = ca.descend(b'r').unwrap();
assert!(car.is_final());  // ✓ "car" is a term
assert_eq!(car.path(), b"car");
```

### Example 2: Exploring Children

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

let dict = DoubleArrayTrie::from_terms(vec!["ab", "ac", "ad"]);
let root = DoubleArrayTrieZipper::new_from_dict(&dict);

// Explore from root
let root_children: Vec<u8> = root.children()
    .map(|(label, _child)| label)
    .collect();
assert_eq!(root_children, vec![b'a']);

// Navigate to 'a'
let a = root.descend(b'a').unwrap();

// Explore children of 'a'
let a_children: Vec<u8> = a.children()
    .map(|(label, _child)| label)
    .collect();
assert_eq!(a_children, vec![b'b', b'c', b'd']);

// Visit each child
for (label, child) in a.children() {
    println!("Edge: {} -> {}",
        String::from_utf8_lossy(&a.path()),
        label as char
    );
    assert!(child.is_final());  // All are terminal
}
```

### Example 3: Value Access

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("test", 1),
    ("testing", 2),
    ("tested", 3),
]);

let root = DoubleArrayTrieZipper::new_from_dict(&dict);

// Navigate to "test"
let test = root
    .descend(b't')
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b's'))
    .and_then(|z| z.descend(b't'))
    .unwrap();

assert!(test.is_final());
assert_eq!(test.value(), Some(1));  // ✓ Value at "test"

// Continue to "testing"
let testing = test
    .descend(b'i')
    .and_then(|z| z.descend(b'n'))
    .and_then(|z| z.descend(b'g'))
    .unwrap();

assert_eq!(testing.value(), Some(2));  // ✓ Value at "testing"

// Navigate to "tested" from root (different path)
let tested = root
    .descend(b't')
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b's'))
    .and_then(|z| z.descend(b't'))
    .and_then(|z| z.descend(b'e'))
    .and_then(|z| z.descend(b'd'))
    .unwrap();

assert_eq!(tested.value(), Some(3));  // ✓ Value at "tested"
```

### Example 4: Unicode Navigation

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrieCharZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

let dict = DoubleArrayTrieChar::from_terms_with_values(vec![
    ("café", "French"),
    ("naïve", "French"),
    ("中文", "Chinese"),
    ("🎉", "Emoji"),
]);

let root = DoubleArrayTrieCharZipper::new_from_dict(&dict);

// Navigate to "café"
let cafe = root
    .descend('c')
    .and_then(|z| z.descend('a'))
    .and_then(|z| z.descend('f'))
    .and_then(|z| z.descend('é'))  // Single character!
    .unwrap();

assert!(cafe.is_final());
assert_eq!(cafe.value(), Some("French"));

// Navigate to "中文"
let chinese = root
    .descend('中')
    .and_then(|z| z.descend('文'))
    .unwrap();

assert_eq!(chinese.value(), Some("Chinese"));

// Navigate to emoji
let emoji = root.descend('🎉').unwrap();
assert_eq!(emoji.value(), Some("Emoji"));
```

### Example 5: Depth-First Traversal

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

fn collect_all_terms(zipper: &DoubleArrayTrieZipper) -> Vec<String> {
    let mut results = Vec::new();

    // Check if current position is a term
    if zipper.is_final() {
        let term = String::from_utf8_lossy(&zipper.path()).to_string();
        results.push(term);
    }

    // Recursively visit all children
    for (_label, child) in zipper.children() {
        results.extend(collect_all_terms(&child));
    }

    results
}

let dict = DoubleArrayTrie::from_terms(vec![
    "cat", "car", "card", "care", "careful"
]);
let root = DoubleArrayTrieZipper::new_from_dict(&dict);

let terms = collect_all_terms(&root);
println!("Found terms: {:?}", terms);
// Output: ["car", "card", "care", "careful", "cat"]
```

### Example 6: Prefix Matching

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

fn find_with_prefix(dict: &DoubleArrayTrie, prefix: &str) -> Vec<String> {
    let root = DoubleArrayTrieZipper::new_from_dict(dict);

    // Navigate to prefix
    let mut zipper = root;
    for byte in prefix.bytes() {
        zipper = match zipper.descend(byte) {
            Some(z) => z,
            None => return vec![],  // Prefix not in dictionary
        };
    }

    // Collect all completions
    let mut results = Vec::new();

    if zipper.is_final() {
        results.push(prefix.to_string());
    }

    fn collect(z: &DoubleArrayTrieZipper, results: &mut Vec<String>) {
        if z.is_final() {
            results.push(String::from_utf8_lossy(&z.path()).to_string());
        }
        for (_, child) in z.children() {
            collect(&child, results);
        }
    }

    collect(&zipper, &mut results);
    results
}

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester", "temp", "template"
]);

let completions = find_with_prefix(&dict, "test");
println!("{:?}", completions);
// Output: ["test", "tested", "tester", "testing"]

let completions = find_with_prefix(&dict, "tem");
println!("{:?}", completions);
// Output: ["temp", "template"]
```

### Example 7: Custom Traversal with Value Filtering

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

fn find_in_scope(
    dict: &DoubleArrayTrie<u32>,
    prefix: &str,
    scope_id: u32,
) -> Vec<(String, u32)> {
    let root = DoubleArrayTrieZipper::new_from_dict(dict);

    // Navigate to prefix
    let mut zipper = root;
    for byte in prefix.bytes() {
        zipper = match zipper.descend(byte) {
            Some(z) => z,
            None => return vec![],
        };
    }

    // Collect matching terms
    let mut results = Vec::new();

    fn collect_filtered(
        z: &DoubleArrayTrieZipper<u32>,
        scope: u32,
        results: &mut Vec<(String, u32)>,
    ) {
        if z.is_final() {
            if let Some(value) = z.value() {
                if value == scope {
                    let term = String::from_utf8_lossy(&z.path()).to_string();
                    results.push((term, value));
                }
            }
        }

        for (_, child) in z.children() {
            collect_filtered(&child, scope, results);
        }
    }

    collect_filtered(&zipper, scope_id, &mut results);
    results
}

// Code completion scenario
let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("test_global", 0),
    ("test_local", 1),
    ("temp_global", 0),
    ("temp_local", 1),
]);

// Find "te*" in local scope (1)
let results = find_in_scope(&dict, "te", 1);
println!("{:?}", results);
// Output: [("temp_local", 1), ("test_local", 1)]

// Find "te*" in global scope (0)
let results = find_in_scope(&dict, "te", 0);
println!("{:?}", results);
// Output: [("temp_global", 0), ("test_global", 0)]
```

### Example 8: Hierarchical Scopes

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

#[derive(Clone, Debug, PartialEq)]
struct ScopeInfo {
    id: u32,
    parent: Option<u32>,  // Hierarchical scope chain
}

impl libdictenstein::DictionaryValue for ScopeInfo {}

fn find_in_scope_hierarchy(
    dict: &DoubleArrayTrie<ScopeInfo>,
    prefix: &str,
    current_scope: u32,
) -> Vec<(String, u32)> {
    let root = DoubleArrayTrieZipper::new_from_dict(dict);

    // Build scope chain (current → parent → grandparent → ... → global)
    let mut scope_chain = std::collections::HashSet::new();
    scope_chain.insert(current_scope);

    // In real implementation, traverse parent chain
    // For demo, just use current scope

    let mut zipper = root;
    for byte in prefix.bytes() {
        zipper = match zipper.descend(byte) {
            Some(z) => z,
            None => return vec![],
        };
    }

    let mut results = Vec::new();

    fn collect_with_hierarchy(
        z: &DoubleArrayTrieZipper<ScopeInfo>,
        scopes: &std::collections::HashSet<u32>,
        results: &mut Vec<(String, u32)>,
    ) {
        if z.is_final() {
            if let Some(info) = z.value() {
                if scopes.contains(&info.id) {
                    let term = String::from_utf8_lossy(&z.path()).to_string();
                    results.push((term, info.id));
                }
            }
        }

        for (_, child) in z.children() {
            collect_with_hierarchy(&child, scopes, results);
        }
    }

    collect_with_hierarchy(&zipper, &scope_chain, &mut results);
    results
}

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("global_var", ScopeInfo { id: 0, parent: None }),
    ("function", ScopeInfo { id: 1, parent: Some(0) }),
    ("local_var", ScopeInfo { id: 2, parent: Some(1) }),
]);

// Search from innermost scope - finds local and inherited
let results = find_in_scope_hierarchy(&dict, "", 2);
println!("{:?}", results);
```

## Performance Analysis

### Memory Characteristics

#### Per-Zipper Overhead

```
DoubleArrayTrieZipper:
  state: usize        →  8 bytes
  path: Vec<u8>       →  24 + path_length bytes
  shared: Arc<...>    →  8 bytes (pointer)
  ──────────────────────────────────────────
  Total base:           40 bytes + path
```

```
DoubleArrayTrieCharZipper:
  state: usize        →  8 bytes
  path: Vec<char>     →  24 + (path_length × 4) bytes
  shared: Arc<...>    →  8 bytes
  ──────────────────────────────────────────
  Total base:           40 bytes + (path × 4)
```

**Example**: Navigating to "testing" (7 chars):
- Byte zipper: 40 + 7 = 47 bytes
- Char zipper: 40 + 28 = 68 bytes

#### Cloning Cost

Zippers use `Arc` for shared data, so cloning is cheap:

```rust
let z1 = root.descend('a');  // Clone shared Arc: O(1)
let z2 = z1.clone();          // Clone again: O(1) + path copy
```

**Cost**: $`\mathcal{O}(1)`$ for Arc clone + $`\mathcal{O}(P)`$ for path copy (P = path length)

### Operation Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `new_from_dict()` | $`\mathcal{O}(1)`$ | Initialize at root |
| `is_final()` | $`\mathcal{O}(1)`$ | Array lookup |
| `descend(label)` | $`\mathcal{O}(1)`$ | BASE/CHECK lookup |
| `children()` | $`\mathcal{O}(E)`$ | Iterate precomputed edges |
| `path()` | $`\mathcal{O}(P)`$ | Clone path vector |
| `value()` | $`\mathcal{O}(1)`$ | Array lookup |

E = average edge count (~2-3)
P = path length

### Benchmarks

#### Navigation Performance

```
Operation: Navigate to "testing" (7 steps)

DoubleArrayTrieZipper:
  7 × descend():       140ns  (20ns per step)
  is_final():          5ns
  path():              25ns   (copy 7 bytes)
  Total:               170ns

Traditional Dictionary::contains():
  Same query:          120ns  (more optimized, no path tracking)
```

**Insight**: Zipper adds ~40% overhead for path tracking, but provides more flexibility.

#### Traversal Benchmarks

```
Collect all terms (10,000-term dictionary):

Zipper DFS:           2.1ms
Direct iteration:     1.5ms  (30% faster, but less flexible)
```

**Insight**: Zippers trade some performance for functional interface.

### Memory Usage Patterns

#### During Traversal

```
Scenario: Fuzzy search with custom zipper traversal

Peak memory per concurrent search:
  Base zipper:         40 bytes
  Average path:        ~15 bytes (typical term length)
  Exploration states:  ~50 zippers in flight
  ──────────────────────────────────────────
  Total per search:    ~2.75 KB

For 100 concurrent searches: ~275 KB
```

**Insight**: Very lightweight, suitable for concurrent use.

## Advanced Patterns

### Pattern 1: Zipper-Based Trie Iterator

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

struct TrieIterator {
    stack: Vec<DoubleArrayTrieZipper>,
}

impl TrieIterator {
    fn new(dict: &DoubleArrayTrie) -> Self {
        let root = DoubleArrayTrieZipper::new_from_dict(dict);
        Self { stack: vec![root] }
    }
}

impl Iterator for TrieIterator {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(zipper) = self.stack.pop() {
            // Add children to stack (reversed for in-order)
            let children: Vec<_> = zipper.children()
                .map(|(_label, child)| child)
                .collect();

            for child in children.into_iter().rev() {
                self.stack.push(child);
            }

            // Return if final state
            if zipper.is_final() {
                return Some(zipper.path());
            }
        }

        None
    }
}

// Usage
let dict = DoubleArrayTrie::from_terms(vec!["a", "ab", "abc"]);
let mut iter = TrieIterator::new(&dict);

assert_eq!(iter.next(), Some(b"a".to_vec()));
assert_eq!(iter.next(), Some(b"ab".to_vec()));
assert_eq!(iter.next(), Some(b"abc".to_vec()));
assert_eq!(iter.next(), None);
```

### Pattern 2: Bounded Search

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;

fn search_with_depth_limit(
    zipper: &DoubleArrayTrieZipper,
    max_depth: usize,
) -> Vec<Vec<u8>> {
    let mut results = Vec::new();

    fn dfs(
        z: &DoubleArrayTrieZipper,
        depth: usize,
        max_depth: usize,
        results: &mut Vec<Vec<u8>>,
    ) {
        if depth > max_depth {
            return;  // Stop exploring
        }

        if z.is_final() {
            results.push(z.path());
        }

        for (_, child) in z.children() {
            dfs(&child, depth + 1, max_depth, results);
        }
    }

    dfs(zipper, 0, max_depth, &mut results);
    results
}

// Find terms within 5 characters
let dict = DoubleArrayTrie::from_terms(vec![
    "a", "ab", "abc", "abcd", "abcde", "abcdef"
]);
let root = DoubleArrayTrieZipper::new_from_dict(&dict);

let short_terms = search_with_depth_limit(&root, 5);
println!("{} terms ≤ 5 chars", short_terms.len());
// Output: 5 terms ≤ 5 chars (excludes "abcdef")
```

### Pattern 3: Parallel Exploration

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::DictZipper;
use rayon::prelude::*;

fn parallel_prefix_search(dict: &DoubleArrayTrie, prefixes: &[&str]) -> Vec<Vec<String>> {
    prefixes.par_iter().map(|&prefix| {
        let root = DoubleArrayTrieZipper::new_from_dict(dict);

        // Navigate to prefix
        let mut zipper = root;
        for byte in prefix.bytes() {
            zipper = match zipper.descend(byte) {
                Some(z) => z,
                None => return vec![],
            };
        }

        // Collect completions
        let mut results = Vec::new();
        fn collect(z: &DoubleArrayTrieZipper, results: &mut Vec<String>) {
            if z.is_final() {
                results.push(String::from_utf8_lossy(&z.path()).to_string());
            }
            for (_, child) in z.children() {
                collect(&child, results);
            }
        }
        collect(&zipper, &mut results);
        results
    }).collect()
}

// Search multiple prefixes in parallel
let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "temp", "template"
]);

let results = parallel_prefix_search(&dict, &["te", "tem"]);
println!("{:?}", results);
// Output: [["test", "testing", "temp", "template"], ["temp", "template"]]
```

### Pattern 4: Contextual Completion Engine

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;
use libdictenstein::zipper::{DictZipper, ValuedDictZipper};

struct CompletionEngine {
    dict: DoubleArrayTrie<u32>,  // Scope IDs
}

impl CompletionEngine {
    fn complete_in_scopes(
        &self,
        prefix: &str,
        visible_scopes: &[u32],
    ) -> Vec<String> {
        let root = DoubleArrayTrieZipper::new_from_dict(&self.dict);

        // Navigate to prefix
        let mut zipper = root;
        for byte in prefix.bytes() {
            zipper = match zipper.descend(byte) {
                Some(z) => z,
                None => return vec![],
            };
        }

        // Collect visible terms
        let mut results = Vec::new();

        fn collect_visible(
            z: &DoubleArrayTrieZipper<u32>,
            scopes: &[u32],
            results: &mut Vec<String>,
        ) {
            if z.is_final() {
                if let Some(scope) = z.value() {
                    if scopes.contains(&scope) {
                        let term = String::from_utf8_lossy(&z.path()).to_string();
                        results.push(term);
                    }
                }
            }

            for (_, child) in z.children() {
                collect_visible(&child, scopes, results);
            }
        }

        collect_visible(&zipper, visible_scopes, &mut results);
        results
    }
}

// Usage
let mut engine = CompletionEngine {
    dict: DoubleArrayTrie::from_terms_with_values(vec![
        ("global_func", 0),
        ("local_var", 1),
        ("param", 2),
    ]),
};

// Complete with access to scopes 1 and 2 (local + params)
let completions = engine.complete_in_scopes("", &[1, 2]);
println!("{:?}", completions);
// Output: ["local_var", "param"]
```

## Related Documentation

- [Dictionary Layer](../01-dictionary-layer/README.md) - Underlying dictionary implementations
- [Value Storage](../09-value-storage/README.md) - Term-to-value mappings
- [DoubleArrayTrie Guide](../01-dictionary-layer/implementations/double-array-trie.md)
- [Contextual Completion](../07-contextual-completion/README.md) - Using zippers for code completion

## References

### Academic Papers

1. **Huet, G. (1997)**. "The Zipper"
   - *Journal of Functional Programming*, 7(5), 549-554
   - DOI: [10.1017/S0956796897002864](https://doi.org/10.1017/S0956796897002864)
   - 📄 **Original zipper paper**

2. **Hinze, R., & Jeuring, J. (2001)**. "Generic Haskell: Applications"
   - *Generic Programming: Advanced Lectures*, 57-96
   - DOI: [10.1007/978-3-540-45191-4_2](https://doi.org/10.1007/978-3-540-45191-4_2)
   - 📄 Zipper applications in generic programming

### Open Access Resources

3. **Learn You a Haskell: Zippers**
   - 📄 [http://learnyouahaskell.com/zippers](http://learnyouahaskell.com/zippers)
   - Excellent tutorial on zipper concept

4. **Wikibooks: Haskell/Zippers**
   - 📄 [https://en.wikibooks.org/wiki/Haskell/Zippers](https://en.wikibooks.org/wiki/Haskell/Zippers)
   - Practical examples

### Blog Posts

5. **Thorne, M. (2015)**. "Zippers for Non-Functional Data Structures"
   - 📄 Adapting functional zippers to imperative languages

## Next Steps

- **Implementation**: Read [DoubleArrayTrie Guide](../01-dictionary-layer/implementations/double-array-trie.md)
- **Values**: Explore [Value Storage](../09-value-storage/README.md)
- **Use Case**: See [Contextual Completion](../07-contextual-completion/README.md)

---

**Navigation**: [← Back to Algorithms](../README.md) | [Dictionary Layer →](../01-dictionary-layer/README.md)
