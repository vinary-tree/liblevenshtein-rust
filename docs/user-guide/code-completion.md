# Code Completion Guide

A comprehensive guide to using liblevenshtein-rust for building intelligent code completion systems.

Code completion is most useful when suggestions respect lexical scope: an identifier
defined in an inner block should be visible there and in its descendants, but not in
sibling or parent scopes. The contextual-completion engine models this as a tree of
scopes, illustrated below.

![Context scope tree: nested completion scopes where child contexts inherit the visible terms of their ancestors (lexical scoping)](../diagrams/contextual/context-scope-tree.svg)

*Context scope tree: child scopes inherit visible terms from their ancestors.*

## Quick Start

```rust
use liblevenshtein::prelude::*;

// Build dictionary from code identifiers
let identifiers = vec!["getValue", "getVariable", "setValue", "calculate"];
let dict = PathMapDictionary::from_iter(identifiers);

// Get autocomplete suggestions with typo tolerance
for candidate in Transducer::new(dict, Algorithm::Standard)
    .query_ordered("getVal", 1)    // Max edit distance: 1
    .prefix()                       // Enable prefix matching
    .take(5)                        // Top 5 results
{
    println!("{}: distance {}", candidate.term, candidate.distance);
}

// Output:
//   getValue: distance 0
//   getVariable: distance 1
//   setValue: distance 1
```

## Core Features

### 1. Prefix Matching

Enable autocomplete where dictionary terms can be longer than the user's input.

```rust
// User types: "get"
// Matches: "getValue", "getVariable", "getResult", etc.
query.prefix()
```

**How it works:**
- Standard matching requires exact length: query="test" matches only "test"
- Prefix matching allows longer terms: query="test" matches "test", "testing", "tester"
- Characters beyond the query length are treated as "free" (no edit distance penalty)

**Use cases:**
- Autocomplete as user types
- Search-as-you-type interfaces
- Command completion in shells

### 2. Filtering

Apply context-aware filters to narrow down suggestions.

```rust
// Filter to only public functions
query.filter(|candidate| is_public_function(&candidate.term))
```

**Common filters:**
- **Scope**: Current function, class, or module
- **Visibility**: Public vs private identifiers
- **Type**: Functions vs variables vs classes
- **Naming conventions**: camelCase vs snake_case vs PascalCase

**Example: Multi-criteria filtering**
```rust
transducer
    .query_ordered("user", 1)
    .prefix()
    .filter(|c| {
        // Only public methods in current scope
        is_public(&c.term) && in_current_scope(&c.term)
    })
    .take(10)
```

### 3. Typo Tolerance

Set maximum edit distance to handle user typos.

```rust
// Allow up to 2 character differences
query_ordered("getUserNme", 2)  // Matches "getUserName"
```

**Recommended distances:**
- `distance=0`: Exact prefix matching (fast, restrictive)
- `distance=1`: Single typo tolerance (balanced)
- `distance=2`: Multiple typo tolerance (slower, permissive)
- `distance=3+`: Use sparingly (very slow for large dictionaries)

### 4. Ordering Guarantees

Results are ordered by:
1. **Primary**: Edit distance (0, 1, 2, ...)
2. **Secondary**: Lexicographic order (within each distance)

This enables efficient top-K queries and distance-bounded searches.

```rust
// Get exactly 5 best matches
query.prefix().take(5)

// Get all matches within distance 1
query.prefix().take_while(|c| c.distance <= 1)
```

## Performance Optimization

### Three Strategies

| Strategy | Best For | Setup Cost | Query Speed | Memory |
|----------|----------|------------|-------------|---------|
| **Post-Filtering** | Small dicts (<1K) | None | Baseline | Low |
| **Bitmap Masking** | Medium dicts, moderate filtering | O(n) | 2-5x faster | Medium |
| **Sub-Trie** | Large dicts, restrictive filtering | O(n log n) | 10-200x faster | High |

See [Contextual Filtering Optimization](../archive/performance/CONTEXTUAL_FILTERING_OPTIMIZATION.md) for detailed analysis.

### Decision Tree

```
Is dictionary < 1,000 terms?
├─ YES → Use post-filtering
└─ NO ↓

Does context change every query?
├─ YES → Use bitmap masking
└─ NO ↓

Does filter remove > 90% of terms?
├─ YES → Use sub-trie construction
└─ NO → Use bitmap masking
```

## Real-World Examples

### Example 1: IDE Code Completion

**Requirements:**
- 100,000+ identifiers across project
- User typing in function body
- Need results in < 5ms
- Context: current scope + imports

**Solution: Bitmap Masking**

```rust
struct IDECompletion {
    full_dict: PathMapDictionary,
    active_mask: Vec<bool>,
    symbols: Vec<Symbol>,
}

impl IDECompletion {
    fn on_scope_change(&mut self, scope: &Scope) {
        // Update bitmap when scope changes (infrequent)
        for (i, symbol) in self.symbols.iter().enumerate() {
            self.active_mask[i] = scope.contains(&symbol.name)
                || imports.contains(&symbol.name);
        }
    }

    fn complete(&self, input: &str) -> Vec<Candidate> {
        // Fast queries within same scope (frequent)
        Transducer::new(self.full_dict.clone(), Algorithm::Standard)
            .query_ordered(input, 1)
            .prefix()
            .filter(|c| self.is_active(&c.term))
            .take(10)
            .collect()
    }
}
```

**Performance:**
- Context switch: ~10ms (once per scope change)
- Query: ~0.5ms (every keystroke)
- Total for 100 keystrokes in same scope: 10ms + (100 × 0.5ms) = 60ms

### Example 2: API Documentation Search

**Requirements:**
- 10,000 API methods
- User searches for methods in specific module
- Filter specified upfront
- Many searches within same filter

**Solution: Sub-Trie Construction**

```rust
struct APISearch {
    all_methods: Vec<APIMethod>,
    module_tries: HashMap<String, PathMapDictionary>,
}

impl APISearch {
    fn search_module(&mut self, module: &str, query: &str) -> Vec<Candidate> {
        // Build or retrieve cached sub-trie
        let dict = self.module_tries.entry(module.to_string())
            .or_insert_with(|| {
                let methods: Vec<_> = self.all_methods
                    .iter()
                    .filter(|m| m.module == module)
                    .map(|m| m.name.as_str())
                    .collect();
                PathMapDictionary::from_iter(methods)
            });

        Transducer::new(dict.clone(), Algorithm::Standard)
            .query_ordered(query, 2)
            .prefix()
            .take(20)
            .collect()
    }
}
```

**Performance:**
- First query per module: ~50ms (builds sub-trie)
- Subsequent queries: ~0.05ms (cached sub-trie)
- 1000 queries across 10 modules: 500ms + (990 × 0.05ms) ≈ 550ms

### Example 3: Command-Line Autocomplete

**Requirements:**
- ~500 commands/options
- Context changes every command
- Simple filters (e.g., "starts with --")

**Solution: Post-Filtering**

```rust
fn complete_command(input: &str, dict: &PathMapDictionary) -> Vec<String> {
    Transducer::new(dict.clone(), Algorithm::Standard)
        .query_ordered(input, 1)
        .prefix()
        .filter(|c| {
            // Simple filter: flags start with --
            if input.starts_with("--") {
                c.term.starts_with("--")
            } else {
                !c.term.starts_with("--")
            }
        })
        .take(5)
        .map(|c| c.term)
        .collect()
}
```

**Performance:**
- Query: ~0.2ms (small dictionary, simple filter)
- Good enough for interactive CLI

## Advanced Techniques

### Combining Filters and Prefix Matching

```rust
// Get top 5 public functions starting with "get"
transducer
    .query_ordered("get", 0)
    .prefix()
    .filter(|c| is_public_function(c.term))
    .take(5)
```

### Distance-Bounded Search

```rust
// All matches within distance 1
transducer
    .query_ordered("user", 2)
    .prefix()
    .take_while(|c| c.distance <= 1)
```

### Lazy Evaluation

The iterator is lazy - it only computes results as needed:

```rust
// Only explores dictionary until 3 results found
transducer.query_ordered("get", 1).prefix().take(3)
```

### Chaining Transformations

```rust
// Complex pipeline
transducer
    .query_ordered("getVal", 1)
    .prefix()                                    // Prefix matching
    .filter(|c| c.term.len() > 5)               // Length filter
    .filter(|c| is_in_scope(c.term))            // Scope filter
    .take_while(|c| c.distance <= 1)            // Distance bound
    .take(10)                                    // Top 10
```

## Testing Your Implementation

```rust
#[test]
fn test_code_completion() {
    let dict = PathMapDictionary::from_iter(vec![
        "getValue", "getVariable", "setValue", "calculate"
    ]);

    // Test prefix matching
    let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
        .query_ordered("getV", 0)
        .prefix()
        .collect();

    assert_eq!(results.len(), 2);  // getValue, getVariable
    assert_eq!(results[0].distance, 0);

    // Test typo tolerance
    let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
        .query_ordered("getValu", 1)
        .prefix()
        .collect();

    assert!(results.iter().any(|c| c.term == "getValue"));
}
```

## Common Pitfalls

### ❌ Pitfall 1: Using high edit distances with large dictionaries

```rust
// Very slow for 100K dictionary!
transducer.query_ordered("x", 5).prefix().take(10)
```

**Solution**: Keep distance ≤ 2 for large dictionaries

### ❌ Pitfall 2: Expensive filters in hot path

```rust
// Calls expensive_check() for every candidate!
query.prefix().filter(|c| expensive_check(c.term))
```

**Solution**: Pre-filter dictionary or use bitmap masking

### ❌ Pitfall 3: Not using prefix mode for autocomplete

```rust
// Won't match "testing" when user types "test"
query_ordered("test", 0)  // Missing .prefix()
```

**Solution**: Always use `.prefix()` for autocomplete

### ❌ Pitfall 4: Rebuilding dictionary on every query

```rust
// Rebuilds dictionary every time!
for query in user_queries {
    let dict = PathMapDictionary::from_iter(terms.clone());
    Transducer::new(dict, Algorithm::Standard).query_ordered(query, 1);
}
```

**Solution**: Build dictionary once, reuse for all queries

## API Reference

### OrderedQueryIterator

**Methods:**
- `prefix() -> PrefixOrderedQueryIterator` - Enable prefix matching
- `filter<F>(predicate: F) -> FilteredOrderedQueryIterator` - Add filter predicate

**Inherited from Iterator:**
- `take(n)` - Take first n results
- `take_while(predicate)` - Take while condition holds
- `collect()` - Collect all results into Vec
- `map(f)` - Transform results
- `filter(predicate)` - Additional filtering

### OrderedCandidate

**Fields:**
- `term: String` - The matched dictionary term
- `distance: usize` - Edit distance from query

### Algorithm

**Variants:**
- `Standard` - Insert, delete, substitute (fastest)
- `Transposition` - Adds character swap support
- `MergeAndSplit` - Adds merge/split operations (most flexible)

## Further Reading

- [Contextual Filtering Optimization](../archive/performance/CONTEXTUAL_FILTERING_OPTIMIZATION.md) - Detailed performance guide
- [examples/code_completion_demo.rs](../../examples/code_completion_demo.rs) - Basic usage
- [examples/advanced_contextual_filtering.rs](../../examples/advanced_contextual_filtering.rs) - Bitmap masking
- [examples/contextual_filtering_optimization.rs](../../examples/contextual_filtering_optimization.rs) - Strategy comparison

## Summary

liblevenshtein-rust provides powerful building blocks for code completion:

✅ **Prefix matching** - Essential for autocomplete
✅ **Typo tolerance** - Handle user mistakes gracefully
✅ **Filtering** - Context-aware suggestions
✅ **Ordering guarantees** - Distance-first, then lexicographic
✅ **Lazy evaluation** - Efficient top-K queries
✅ **Flexible optimization** - Post-filter, bitmap mask, or sub-tries

Choose the right combination of features and optimizations for your use case, and you'll have a fast, robust code completion system.

---

[← Documentation Index](../README.md)
