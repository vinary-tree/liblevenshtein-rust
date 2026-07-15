# PrefixZipper Usage Guide

**Quick practical guide for using PrefixZipper in your applications**

`PrefixZipper` is a trait implemented by the per-backend zipper types (provided by the
**[libdictenstein](https://crates.io/crates/libdictenstein)** crate and re-exported here).
The trait hierarchy below shows how the dictionary and zipper traits relate.

![The Dictionary, MappedDictionary, and DictionaryNode trait hierarchy and the methods each exposes](../diagrams/dictionary-structures/dictionary-traits.svg)

*Dictionary trait hierarchy: the zipper types navigate the same backends shown here.*

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Patterns](#common-patterns)
3. [Backend Selection](#backend-selection)
4. [Performance Tuning](#performance-tuning)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
liblevenshtein = "0.9.1"
```

### Basic Usage (5 lines of code)

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

// 1. Create dictionary
let dict = DoubleArrayTrie::from_terms(vec!["apple", "apply"].iter());

// 2. Create zipper
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

// 3. Query prefix
if let Some(iter) = zipper.with_prefix(b"app") {
    // 4. Iterate results
    for (term_bytes, _zipper) in iter {
        // 5. Convert to String
        let term = String::from_utf8(term_bytes).unwrap();
        println!("{}", term);
    }
}
```

### Copy-Paste Template

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;
use libdictenstein::double_array_trie::DoubleArrayTrieZipper;

fn prefix_search(dictionary_terms: Vec<&str>, user_prefix: &str) -> Vec<String> {
    // Build dictionary
    let dict = DoubleArrayTrie::from_terms(dictionary_terms.iter());

    // Create zipper
    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

    // Query and collect results
    zipper
        .with_prefix(user_prefix.as_bytes())
        .map(|iter| {
            iter.map(|(path, _)| String::from_utf8(path).unwrap())
                .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let results = prefix_search(
        vec!["hello", "help", "world"],
        "hel"
    );
    println!("{:?}", results); // ["hello", "help"]
}
```

---

## Common Patterns

### Pattern 1: Autocomplete (Top N Results)

**Use case**: User types, show first 10 suggestions

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

struct Autocompleter {
    dict: DoubleArrayTrie,
}

impl Autocompleter {
    fn suggest(&self, typed: &str, limit: usize) -> Vec<String> {
        let zipper = DoubleArrayTrieZipper::new_from_dict(&self.dict);

        zipper
            .with_prefix(typed.as_bytes())
            .map(|iter| {
                iter.take(limit)  // Only take first N
                    .map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn main() {
    let ac = Autocompleter {
        dict: DoubleArrayTrie::from_terms(
            vec!["apple", "application", "apply", "banana"].iter()
        ),
    };

    let suggestions = ac.suggest("app", 2);
    println!("{:?}", suggestions); // First 2 matches
}
```

### Pattern 2: Ranked Results (with Scores)

**Use case**: Sort completions by frequency/popularity

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::ValuedPrefixZipper;

fn ranked_completions(
    terms_with_scores: Vec<(&str, usize)>,
    prefix: &str,
) -> Vec<(String, usize)> {
    // Create valued dictionary
    let dict = DoubleArrayTrie::from_terms_with_values(
        terms_with_scores.into_iter()
    );
    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

    // Query and sort by score
    zipper
        .with_prefix_values(prefix.as_bytes())
        .map(|iter| {
            let mut results: Vec<_> = iter
                .map(|(path, score)| {
                    (String::from_utf8(path).unwrap(), score)
                })
                .collect();

            // Sort by score descending
            results.sort_by_key(|(_, score)| std::cmp::Reverse(*score));
            results
        })
        .unwrap_or_default()
}

fn main() {
    let terms = vec![
        ("the", 1000),    // Most common
        ("these", 150),
        ("there", 200),
        ("they", 300),
    ];

    let ranked = ranked_completions(terms, "the");
    println!("{:?}", ranked);
    // Output: [("the", 1000), ("they", 300), ("there", 200), ("these", 150)]
}
```

### Pattern 3: Code Completion (LSP)

**Use case**: Language server protocol autocomplete

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

struct LanguageServer {
    keywords: DoubleArrayTrie,
    functions: DoubleArrayTrie,
}

impl LanguageServer {
    fn complete(&self, context: CompletionContext) -> Vec<Completion> {
        let dict = match context.kind {
            CompletionKind::Keyword => &self.keywords,
            CompletionKind::Function => &self.functions,
        };

        let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

        zipper
            .with_prefix(context.prefix.as_bytes())
            .map(|iter| {
                iter.map(|(path, _)| Completion {
                    label: String::from_utf8(path).unwrap(),
                    kind: context.kind,
                })
                .collect()
            })
            .unwrap_or_default()
    }
}

#[derive(Copy, Clone)]
enum CompletionKind {
    Keyword,
    Function,
}

struct CompletionContext {
    prefix: String,
    kind: CompletionKind,
}

struct Completion {
    label: String,
    kind: CompletionKind,
}

fn main() {
    let lsp = LanguageServer {
        keywords: DoubleArrayTrie::from_terms(
            vec!["for", "while", "if", "match"].iter()
        ),
        functions: DoubleArrayTrie::from_terms(
            vec!["println", "print", "format"].iter()
        ),
    };

    let completions = lsp.complete(CompletionContext {
        prefix: "pr".to_string(),
        kind: CompletionKind::Function,
    });

    for c in completions {
        println!("{}", c.label);  // "print", "println"
    }
}
```

### Pattern 4: Incremental Search

**Use case**: Update results as user types each character

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

struct IncrementalSearch {
    dict: DoubleArrayTrie,
    current_prefix: String,
    current_results: Vec<String>,
}

impl IncrementalSearch {
    fn new(terms: Vec<&str>) -> Self {
        Self {
            dict: DoubleArrayTrie::from_terms(terms.iter()),
            current_prefix: String::new(),
            current_results: Vec::new(),
        }
    }

    fn on_keypress(&mut self, key: char) {
        // Update prefix
        self.current_prefix.push(key);

        // Recompute results
        let zipper = DoubleArrayTrieZipper::new_from_dict(&self.dict);
        self.current_results = zipper
            .with_prefix(self.current_prefix.as_bytes())
            .map(|iter| {
                iter.take(20)  // Top 20
                    .map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default();
    }

    fn on_backspace(&mut self) {
        self.current_prefix.pop();

        // Recompute results
        let zipper = DoubleArrayTrieZipper::new_from_dict(&self.dict);
        self.current_results = zipper
            .with_prefix(self.current_prefix.as_bytes())
            .map(|iter| {
                iter.take(20)
                    .map(|(path, _)| String::from_utf8(path).unwrap())
                    .collect()
            })
            .unwrap_or_default();
    }

    fn results(&self) -> &[String] {
        &self.current_results
    }
}

fn main() {
    let mut search = IncrementalSearch::new(vec![
        "apple", "application", "apply", "banana",
    ]);

    // Simulate user typing "app"
    search.on_keypress('a');
    println!("After 'a': {:?}", search.results());

    search.on_keypress('p');
    println!("After 'ap': {:?}", search.results());

    search.on_keypress('p');
    println!("After 'app': {:?}", search.results());
}
```

### Pattern 5: Unicode Support

**Use case**: Non-ASCII text (international languages)

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrieCharZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

fn unicode_completion(terms: Vec<&str>, prefix: &str) -> Vec<String> {
    // Use char-based dictionary for Unicode
    let dict = DoubleArrayTrieChar::from_terms(terms.iter());
    let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);

    // Convert prefix to Vec<char>
    let prefix_chars: Vec<char> = prefix.chars().collect();

    zipper
        .with_prefix(&prefix_chars)
        .map(|iter| {
            iter.map(|(path_chars, _)| {
                // Convert Vec<char> back to String
                path_chars.iter().collect()
            })
            .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let terms = vec![
        "café",
        "cafétéria",
        "naïve",
        "naïveté",
        "résumé",
    ];

    let results = unicode_completion(terms, "caf");
    println!("{:?}", results); // ["café", "cafétéria"]

    let results = unicode_completion(terms, "naï");
    println!("{:?}", results); // ["naïve", "naïveté"]
}
```

### Pattern 6: Existence Check (No Collection)

**Use case**: Just check if any matches exist (fastest)

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

fn has_prefix(dict: &DoubleArrayTrie, prefix: &str) -> bool {
    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

    zipper
        .with_prefix(prefix.as_bytes())
        .map(|mut iter| iter.next().is_some())  // Check just first result
        .unwrap_or(false)
}

fn main() {
    let dict = DoubleArrayTrie::from_terms(
        vec!["hello", "world"].iter()
    );

    assert!(has_prefix(&dict, "hel"));   // true
    assert!(!has_prefix(&dict, "xyz"));  // false
}
```

---

## Backend Selection

### Quick Decision Tree

```
Do you need Unicode (non-ASCII)?
├─ YES → Use *Char variants (DoubleArrayTrieChar, DynamicDawgChar)
└─ NO  → Use byte variants (continue below)

Do you need to modify the dictionary after creation?
├─ YES → Use DynamicDawg (supports insertion/deletion)
└─ NO  → Continue below

Is your dictionary small (<1000 terms)?
├─ YES → Use PathMapDictionary (simplest)
└─ NO  → Use DoubleArrayTrie (fastest for large dictionaries)

Do you also need substring matching (not just prefix)?
├─ YES → Use SuffixAutomaton (supports both)
└─ NO  → Use DoubleArrayTrie (recommended default)
```

### Backend Comparison Table

| Backend | Mutability | Unicode | Performance | Memory | Use Case |
|---------|-----------|---------|-------------|--------|----------|
| **DoubleArrayTrie** | ❌ Immutable | ✅ Byte + Char | ⭐⭐⭐ Fastest | ⭐⭐⭐ Compact | **Default choice** |
| **DynamicDawg** | ✅ Mutable | ✅ Byte + Char | ⭐⭐⭐ Fast | ⭐⭐ Medium | Need updates |
| **PathMapDictionary** | ❌ Immutable | ⚠️ Byte only | ⭐⭐ Good | ⭐ Larger | Small dicts |
| **SuffixAutomaton** | ❌ Immutable | ✅ Byte + Char | ⭐ Slower | ⭐⭐ Medium | Need substrings |

### Code Examples for Each Backend

#### DoubleArrayTrie (Default)

```rust
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

let dict = DoubleArrayTrie::from_terms(vec!["term1", "term2"].iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
let results = zipper.with_prefix(b"term");
```

#### DoubleArrayTrieChar (Unicode)

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrieCharZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

let dict = DoubleArrayTrieChar::from_terms(vec!["café", "naïve"].iter());
let zipper = DoubleArrayTrieCharZipper::new_from_dict(&dict);
let prefix: Vec<char> = "caf".chars().collect();
let results = zipper.with_prefix(&prefix);
```

#### DynamicDawg (Mutable)

```rust
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::dynamic_dawg::DynamicDawgZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

let mut dict = DynamicDawg::from_terms(vec!["term1"].iter());

// Add term after creation
dict.insert("term2");

let zipper = DynamicDawgZipper::new_from_dict(&dict);
let results = zipper.with_prefix(b"term");
```

#### PathMapDictionary (Simple)

```rust
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::pathmap::PathMapZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

let dict = PathMapDictionary::from_terms(vec!["term1", "term2"].iter());
let zipper = PathMapZipper::new_from_dict(&dict);
let results = zipper.with_prefix(b"term");
```

---

## Performance Tuning

### Optimization 1: Reuse Zippers

**❌ Slow** (recreate zipper every query):
```rust
fn slow_autocomplete(dict: &DoubleArrayTrie, prefix: &str) -> Vec<String> {
    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);  // Repeated
    zipper.with_prefix(prefix.as_bytes())
        .map(|iter| iter.map(|(p, _)| String::from_utf8(p).unwrap()).collect())
        .unwrap_or_default()
}
```

**✅ Fast** (cache zipper):
```rust
struct FastAutocompleter {
    zipper: DoubleArrayTrieZipper,  // Reused
}

impl FastAutocompleter {
    fn complete(&self, prefix: &str) -> Vec<String> {
        self.zipper.with_prefix(prefix.as_bytes())
            .map(|iter| iter.map(|(p, _)| String::from_utf8(p).unwrap()).collect())
            .unwrap_or_default()
    }
}
```

**Speedup**: ~2-3× faster (avoids repeated zipper initialization)

### Optimization 2: Limit Results Early

**❌ Slow** (collect all, then limit):
```rust
let all_results: Vec<_> = iter.collect();
let top10 = &all_results[..10.min(all_results.len())];
```

**✅ Fast** (limit during iteration):
```rust
let top10: Vec<_> = iter.take(10).collect();
```

**Speedup**: 10-100× faster for selective prefixes (avoids iterating unused results)

### Optimization 3: Avoid Unnecessary String Conversion

**❌ Slow** (convert all to String):
```rust
let results: Vec<String> = iter
    .map(|(path, _)| String::from_utf8(path).unwrap())
    .collect();
```

**✅ Fast** (keep as bytes if possible):
```rust
let results: Vec<Vec<u8>> = iter
    .map(|(path, _)| path)
    .collect();

// Convert only when displaying
for term_bytes in results {
    println!("{}", String::from_utf8_lossy(&term_bytes));
}
```

**Speedup**: ~20% faster (avoids UTF-8 validation overhead)

### Optimization 4: Parallel Queries

**Use case**: Process multiple prefixes simultaneously

```rust
use rayon::prelude::*;
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

fn parallel_completions(
    dict: &DoubleArrayTrie,
    prefixes: Vec<&str>,
) -> Vec<Vec<String>> {
    prefixes
        .par_iter()  // Parallel iterator
        .map(|prefix| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(dict);
            zipper
                .with_prefix(prefix.as_bytes())
                .map(|iter| {
                    iter.map(|(p, _)| String::from_utf8(p).unwrap())
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

fn main() {
    let dict = DoubleArrayTrie::from_terms(
        vec!["hello", "help", "world", "work"].iter()
    );

    let results = parallel_completions(
        &dict,
        vec!["hel", "wor"],
    );

    println!("{:?}", results);
    // [["hello", "help"], ["work", "world"]]
}
```

**Speedup**: Near-linear with core count (e.g., 8× on 8 cores)

### Performance Checklist

- [ ] Using appropriate backend (DoubleArrayTrie for most cases)
- [ ] Reusing zippers instead of recreating per query
- [ ] Limiting results with `.take(N)` instead of collecting all
- [ ] Keeping bytes instead of converting to String unnecessarily
- [ ] Pre-building dictionary (not rebuilding per query)
- [ ] Using valued dictionaries for ranked results (avoids separate scoring)
- [ ] Considering parallel queries for batch operations

---

## Troubleshooting

### Problem 1: Empty Results

**Symptom**: `with_prefix()` returns `None` but terms exist

```rust
let dict = DoubleArrayTrie::from_terms(vec!["hello"].iter());
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
let results = zipper.with_prefix(b"Hello");  // Returns None!
```

**Cause**: Case mismatch (prefix matching is case-sensitive)

**Solution**: Normalize case before querying

```rust
let prefix_lower = "Hello".to_lowercase();
let results = zipper.with_prefix(prefix_lower.as_bytes());
```

### Problem 2: Unicode Garbled

**Symptom**: Unicode characters appear garbled

```rust
let dict = DoubleArrayTrie::from_terms(vec!["café"].iter());
let results: Vec<_> = /* ... */;
// Results look like: [99, 97, 102, 195, 169]  // UTF-8 bytes
```

**Cause**: Using byte-level dictionary with multi-byte UTF-8

**Solution**: Use char-level dictionary

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;

let dict = DoubleArrayTrieChar::from_terms(vec!["café"].iter());
// Now works correctly with Unicode
```

### Problem 3: Slow Performance

**Symptom**: Queries take 100+ µs (expected ~10-20 µs)

**Possible causes**:

1. **Creating zipper per query**
   ```rust
   // ❌ Don't do this in hot loop
   for prefix in many_prefixes {
       let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
       // ...
   }
   ```
   **Fix**: Create zipper once, reuse

2. **Collecting all results unnecessarily**
   ```rust
   // ❌ Don't collect if you only need first N
   let all: Vec<_> = iter.collect();
   ```
   **Fix**: Use `.take(N)`

3. **Wrong backend**
   ```rust
   // ❌ PathMapDictionary is slow for large dicts
   let dict = PathMapDictionary::from_terms(many_terms);
   ```
   **Fix**: Use DoubleArrayTrie for large dictionaries

### Problem 4: Compilation Errors

**Error**: "trait `PrefixZipper` is not implemented for..."

```rust
error[E0599]: no method named `with_prefix` found for type `...`
```

**Cause**: Missing import

**Solution**: Import the trait

```rust
use libdictenstein::prefix_zipper::PrefixZipper;
```

**Error**: Type mismatch with `Unit`

```rust
error[E0308]: mismatched types
expected `&[u8]`, found `&[char]`
```

**Cause**: Mixing byte and char dictionaries

**Solution**: Match dictionary type

```rust
// Byte dictionary → byte prefix
let dict = DoubleArrayTrie::from_terms(...);
zipper.with_prefix(b"prefix");  // &[u8]

// Char dictionary → char prefix
let dict = DoubleArrayTrieChar::from_terms(...);
let prefix: Vec<char> = "prefix".chars().collect();
zipper.with_prefix(&prefix);  // &[char]
```

---

## FAQ

### Q: Should I use PrefixZipper or Levenshtein prefix matching?

**A**: Depends on your input:

- **User typing** (may have typos) → Levenshtein prefix (tolerates errors)
- **Programmatic** (correct input) → PrefixZipper (faster)
- **Hybrid**: Try PrefixZipper first, fall back to Levenshtein if no results

```rust
// Smart completion: try exact first, then fuzzy
fn smart_complete(dict: &DoubleArrayTrie, user_input: &str) -> Vec<String> {
    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

    // Try exact prefix (fast)
    if let Some(iter) = zipper.with_prefix(user_input.as_bytes()) {
        let exact: Vec<_> = iter.take(10).collect();
        if !exact.is_empty() {
            return exact.into_iter()
                .map(|(p, _)| String::from_utf8(p).unwrap())
                .collect();
        }
    }

    // Fall back to fuzzy (slower but tolerant)
    dict.query(user_input, Distance::Levenshtein(1))
        .starts_with()
        .take(10)
        .map(|(p, _)| String::from_utf8(p).unwrap())
        .collect()
}
```

### Q: How do I handle case-insensitive matching?

**A**: Normalize case before building dictionary and querying:

```rust
// Build dictionary with lowercase terms
let terms: Vec<String> = original_terms
    .iter()
    .map(|t| t.to_lowercase())
    .collect();
let dict = DoubleArrayTrie::from_terms(terms.iter());

// Query with lowercase prefix
fn complete(dict: &DoubleArrayTrie, user_input: &str) -> Vec<String> {
    let prefix_lower = user_input.to_lowercase();
    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);

    zipper
        .with_prefix(prefix_lower.as_bytes())
        .map(|iter| iter.map(|(p, _)| String::from_utf8(p).unwrap()).collect())
        .unwrap_or_default()
}
```

### Q: Can I filter results beyond prefix matching?

**A**: Yes, use standard iterator methods:

```rust
zipper
    .with_prefix(b"pre")
    .map(|iter| {
        iter.filter(|(path, _)| path.len() <= 10)  // Max length
            .filter(|(path, _)| !path.contains(&b'-'))  // No hyphens
            .take(20)  // Top 20
            .collect()
    })
```

### Q: How do I get both the term and its value?

**A**: Use `ValuedPrefixZipper` trait:

```rust
use libdictenstein::prefix_zipper::ValuedPrefixZipper;

let dict = DoubleArrayTrie::from_terms_with_values(
    vec![("apple", 10), ("apply", 20)].into_iter()
);
let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

if let Some(iter) = zipper.with_prefix_values(b"app") {
    for (term, value) in iter {
        println!("{} -> {}", String::from_utf8(term).unwrap(), value);
    }
}
```

### Q: What's the maximum prefix length?

**A**: Unlimited (bounded only by dictionary structure). However:

- **Practical limit**: Longest term in dictionary
- **Performance**: Still $`\mathcal{O}(k)`$ for any $`k`$
- **Memory**: $`\mathcal{O}(1)`$ during navigation

### Q: Can I modify the dictionary while iterating?

**A**: No (Rust ownership prevents this). If you need mutations:

1. **Collect results first**:
   ```rust
   let results: Vec<_> = iter.collect();
   // Now can modify dict
   ```

2. **Use DynamicDawg** (supports modifications between queries, not during)

### Q: How do I benchmark my specific use case?

**A**: Use Criterion:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use liblevenshtein::prelude::*;
use libdictenstein::prefix_zipper::PrefixZipper;

fn benchmark_my_use_case(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(/* your terms */);
    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);

    c.bench_function("my_prefix_query", |b| {
        b.iter(|| {
            zipper
                .with_prefix(black_box(b"my_prefix"))
                .map(|iter| iter.collect::<Vec<_>>())
        });
    });
}

criterion_group!(benches, benchmark_my_use_case);
criterion_main!(benches);
```

---

## Additional Resources

- **Full API Documentation**: [`docs/design/prefix-zipper.md`](../design/prefix-zipper.md)
- **Optimization Details**: [`docs/optimization/prefix_zipper_optimization_log.md`](../optimization/prefix_zipper_optimization_log.md)
- **Benchmark Results**: [`docs/optimization/prefix_zipper_baseline.md`](../optimization/prefix_zipper_baseline.md)
- **Source Code**: [`prefix_zipper.rs`](https://github.com/f1r3fly-io/libdictenstein/blob/main/src/prefix_zipper.rs) (in the libdictenstein crate)
- **Test Examples**: [`prefix_zipper_tests.rs`](https://github.com/f1r3fly-io/libdictenstein/blob/main/tests/prefix_zipper_tests.rs) (in the libdictenstein crate)

## Related Documentation

- [Backends](backends.md) - Dictionary backend comparison
- [Code Completion Guide](code-completion.md) - Building completion systems
- [Getting Started](getting-started.md) - Basic usage

---

**Document Version**: 1.1
**Last Updated**: 2026-06-19

[← Documentation Index](../README.md)
