# SuffixAutomaton Implementation

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory: Suffix Automata](#theory-suffix-automata)
3. [Substring vs Prefix Matching](#substring-vs-prefix-matching)
4. [Data Structure](#data-structure)
5. [Construction Algorithm](#construction-algorithm)
6. [Usage Examples](#usage-examples)
7. [Performance Analysis](#performance-analysis)
8. [When to Use](#when-to-use)
9. [References](#references)

## Overview

`SuffixAutomaton` is a specialized dictionary for **substring matching** (finding patterns anywhere in text), unlike traditional tries which only match prefixes. It's the go-to choice for full-text search, code search, and document indexing where patterns can appear at any position.

### Key Advantages

- 🔍 **Substring matching**: Find patterns anywhere, not just at word boundaries
- 💾 **Space-efficient**: $`\le 2n-1`$ states for $`n`$ characters
- ⚡ **Fast construction**: $`\mathcal{O}(n)`$ online construction
- 🔄 **Dynamic updates**: Insert and remove text at runtime
- 📍 **Position tracking**: Know where matches occur in source text

### When to Use

✅ **Use SuffixAutomaton when:**
- Need to find patterns anywhere in text (not just prefixes)
- Full-text search within documents
- Code search (find "calculate" in "recalculate")
- Log analysis (find error codes anywhere)
- Bioinformatics (DNA/protein sequence search)

⚠️ **Use standard dictionaries when:**
- Only need prefix/whole-word matching → Use `DoubleArrayTrie` (3x faster)
- Spell checking / autocomplete → Use `DoubleArrayTrie` or `DynamicDawg`

## Theory: Suffix Automata

### What is a Suffix Automaton?

A **suffix automaton** is a minimal deterministic finite automaton (DFA) that recognizes all **suffixes** of indexed text.

**Example**: Text "banana"

**Suffixes**:
- "banana"
- "anana"
- "nana"
- "ana"
- "na"
- "a"
- "" (empty)

**Key Property**: Any path from root = some substring of the text

### How It Differs from Tries

**Traditional Trie** (prefix matching):
```
Dictionary: ["test", "testing", "tested"]

Trie structure:
       (root)
         |
         t
         |
         e
         |
         s
         |
         t (final "test")
        / \
       i   e
       |   |
       n   d (final "tested")
       |
       g (final "testing")

Query "tes":    ✅ Prefix match
Query "test":   ✅ Complete match
Query "est":    ❌ Not a prefix
Query "sting":  ❌ Not a prefix
```

**Suffix Automaton** (substring matching):
```
Text: "testing"

Suffix Automaton recognizes ALL substrings:
"t", "te", "tes", "test", "testi", "testin", "testing"
"e", "es", "est", "esti", "estin", "esting"
"s", "st", "sti", "stin", "sting"
"t", "ti", "tin", "ting"
"i", "in", "ing"
"n", "ng"
"g"

Query "test":   ✅ Substring match
Query "sting":  ✅ Substring match
Query "tin":    ✅ Substring match
Query "xyz":    ❌ Not in text
```

### Endpos Equivalence

States group substrings by their **ending positions** (endpos):

```
Text: "banana" (positions 0-5)

Substrings ending at position 5 (all suffixes):
  "banana" (0-5)
  "anana" (1-5)
  "nana" (2-5)
  "ana" (3-5)
  "na" (4-5)
  "a" (5-5)

States in suffix automaton ≈ equivalence classes of endpos sets
```

**Minimality**: This grouping ensures $`\le 2n-1`$ states for $`n`$ characters.

### Suffix Links

Each state has a **suffix link** pointing to the longest proper suffix in a different endpos class:

```
State representing "ana" → suffix link → state representing "na"
State representing "banana" → suffix link → state representing "anana"
```

Suffix links form a tree structure used during construction and navigation.

## Substring vs Prefix Matching

### Use Case Comparison

| Scenario | Prefix Dictionary | SuffixAutomaton |
|----------|-------------------|-----------------|
| **Autocomplete** | ✅ "test" → "testing" | ⚠️ Overkill |
| **Spell checking** | ✅ Check whole words | ⚠️ Overkill |
| **Code search** | ❌ Misses "recalculate" | ✅ Finds "calculate" |
| **Log search** | ❌ Misses "ERROR_123" mid-line | ✅ Finds "ERROR_123" |
| **Document search** | ❌ Only finds start of words | ✅ Finds anywhere |
| **DNA/protein search** | ❌ Only finds prefixes | ✅ Finds patterns anywhere |

### Example: Code Search

**Problem**: Find all occurrences of "calculate" in source code

```rust
let code = r#"
fn recalculate_total(items: &[Item]) -> f64 {
    items.iter().map(|i| i.price * calculate_tax(i)).sum()
}

fn calculate_tax(item: &Item) -> f64 {
    item.tax_rate * item.price
}
"#;
```

**With Prefix Dictionary** (DoubleArrayTrie):
```rust
let dict = DoubleArrayTrie::from_terms(vec!["calculate", "recalculate"]);

// Query "calculate"
assert!(dict.contains("calculate"));  // ✅
// But won't find "calculate" inside "recalculate"!
```

**With SuffixAutomaton**:
```rust
let dict = SuffixAutomaton::from_text(code);

// Query "calculate"
// Finds BOTH occurrences:
//   1. Inside "recalculate" at position X
//   2. In "calculate_tax" at position Y

let positions = dict.match_positions("calculate");
println!("Found at positions: {:?}", positions);
// Output: [(0, X), (0, Y)]  // (doc_id, position)
```

## Data Structure

### Core Components

```rust
pub struct SuffixAutomaton {
    // Lock-free: the whole automaton lives behind one atomic pointer.
    // (Shipped as the `LockFreeSuffixAutomaton<u8, V>` newtype, which wraps
    //  `Arc<ArcSwap<SuffixAutomatonInner>>`.)
    inner: Arc<ArcSwap<SuffixAutomatonInner>>,
}

struct SuffixAutomatonInner {
    nodes: Vec<SuffixNode>,          // State storage
    last_state: usize,               // Current state during construction
    text_count: usize,               // Number of indexed texts
    needs_compaction: bool,          // Deletion flag
}

struct SuffixNode {
    edges: Vec<(u8, usize)>,         // Label → child state
    suffix_link: Option<usize>,      // Longest proper suffix link
    max_length: usize,               // Longest string in this class
    is_final: bool,                  // End-of-string marker
    ref_count: usize,                // For garbage collection
}
```

### Memory Layout

```
┌─────────────────┬─────────────┬────────────────┐
│ Component       │ Size        │ Per State      │
├─────────────────┼─────────────┼────────────────┤
│ edges (Vec)     │ ~24 bytes   │ ~24 bytes      │
│ suffix_link     │ 16 bytes    │ 16 bytes       │
│ max_length      │ 8 bytes     │ 8 bytes        │
│ is_final        │ 1 byte      │ 1 byte         │
│ ref_count       │ 8 bytes     │ 8 bytes        │
├─────────────────┼─────────────┼────────────────┤
│ Total per state │ ~57 bytes   │ ~57 bytes      │
└─────────────────┴─────────────┴────────────────┘
```

**For text of n characters**:
- States: $`\le 2n-1`$ (typically $`\approx 1.5n`$)
- Total memory: ~85n bytes

**Example**: 10,000-character document $`\approx`$ 850 KB

## Construction Algorithm

### Online Construction

Suffix automaton is built character by character:

```rust
fn extend(&mut self, byte: u8) {
    let new_state = self.nodes.len();
    self.nodes.push(SuffixNode {
        edges: Vec::new(),
        suffix_link: None,
        max_length: self.nodes[self.last_state].max_length + 1,
        is_final: false,
        ref_count: 0,
    });

    // Add edge from previous states
    let mut current = self.last_state;
    while let Some(curr) = current {
        if self.nodes[curr].has_edge(byte) {
            break;
        }
        self.nodes[curr].add_edge(byte, new_state);
        current = self.nodes[curr].suffix_link;
    }

    // Set suffix link for new state
    if current.is_none() {
        // All states now have edge to new_state
        self.nodes[new_state].suffix_link = Some(0); // Root
    } else {
        let curr = current.unwrap();
        let target = self.nodes[curr].get_edge(byte).unwrap();

        if self.nodes[target].max_length == self.nodes[curr].max_length + 1 {
            // No split needed
            self.nodes[new_state].suffix_link = Some(target);
        } else {
            // Split state (complex case)
            let clone = self.clone_state(target);
            self.nodes[new_state].suffix_link = Some(clone);
            self.nodes[target].suffix_link = Some(clone);

            // Redirect edges
            self.redirect_edges(curr, byte, clone);
        }
    }

    self.last_state = new_state;
}
```

**Complexity**: $`\mathcal{O}(1)`$ amortized per character

### From Multiple Texts

Build generalized suffix automaton:

```rust
fn from_texts<I, S>(texts: I) -> Self
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut automaton = SuffixAutomaton::new();

    for text in texts {
        automaton.insert(text.as_ref());
    }

    automaton
}

fn insert(&self, text: &str) {
    // Lock-free: mutate a clone of the current snapshot, then publish it with
    // an atomic swap (the shipped code retries the publish with compare-and-swap).
    let mut next = (**self.inner.load()).clone();

    // Reset to root for new text
    next.last_state = 0;

    // Extend with each character
    for byte in text.bytes() {
        next.extend(byte);
    }

    // Mark final states
    let mut state = next.last_state;
    while let Some(s) = state {
        next.nodes[s].is_final = true;
        state = next.nodes[s].suffix_link;
    }

    next.text_count += 1;

    self.inner.store(Arc::new(next));
}
```

## Usage Examples

### Example 1: Basic Substring Search

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let text = "the quick brown fox jumps over the lazy dog";
let dict = SuffixAutomaton::from_text(text);

// Find substrings
assert!(dict.contains("quick"));       // ✅ Found
assert!(dict.contains("brown fox"));   // ✅ Found
assert!(dict.contains("fox jumps"));   // ✅ Found
assert!(dict.contains("lazy"));        // ✅ Found
assert!(!dict.contains("fast"));       // ❌ Not in text
```

### Example 2: Code Search

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let code = r#"
fn calculate_total(items: &[Item]) -> f64 {
    items.iter().map(|i| i.price).sum()
}

fn recalculate() {
    let total = calculate_total(&items);
}
"#;

let dict = SuffixAutomaton::from_text(code);

// Fuzzy search for "calculate" with typos
let automaton = LevenshteinAutomaton::new("calculat", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Finds: "calculate" (appears in both functions)
```

### Example 3: Multi-Document Search

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let documents = vec![
    "Levenshtein automata for approximate matching",
    "Suffix trees and suffix arrays for pattern search",
    "Double array tries for efficient dictionaries",
];

let dict = SuffixAutomaton::from_texts(documents);

// Search across all documents
assert!(dict.contains("automata"));     // Doc 0
assert!(dict.contains("suffix"));       // Doc 1
assert!(dict.contains("array"));        // Doc 2
assert!(dict.contains("for"));          // All docs (common word)
```

### Example 4: Position Tracking

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let text = "banana";
let dict = SuffixAutomaton::from_text(text);

// Find where "ana" appears
let positions = dict.match_positions("ana");

println!("'ana' appears at positions: {:?}", positions);
// Output: [(0, 1), (0, 3)]
//          ↑       ↑
//          Doc 0,  Doc 0,
//          pos 1   pos 3
//          b[ana]na ban[ana]
```

### Example 5: Dynamic Updates

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let dict = SuffixAutomaton::new();

// Add texts dynamically
dict.insert("testing the suffix automaton");
dict.insert("another test string");

assert!(dict.contains("test"));
assert!(dict.contains("suffix"));

// Remove text
dict.remove("another test string");

// Compact to reclaim space
if dict.needs_compaction() {
    dict.compact();
}
```

### Example 6: Log Analysis

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let logs = vec![
    "[2024-01-01] INFO: Server started",
    "[2024-01-01] ERROR_CODE_123: Connection failed",
    "[2024-01-01] WARN: High memory usage",
    "[2024-01-01] ERROR_CODE_456: Timeout",
];

let dict = SuffixAutomaton::from_texts(logs);

// Find all error codes
let automaton = LevenshteinAutomaton::new("ERROR_CODE", 0, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("Error codes found: {:?}", results);
// Finds: "ERROR_CODE_123", "ERROR_CODE_456"
```

### Example 7: DNA Sequence Search

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let dna_sequence = "ATCGATCGATCGATCGTAGCTAGCTAGCT";
let dict = SuffixAutomaton::from_text(dna_sequence);

// Find motif
assert!(dict.contains("ATCG"));
assert!(dict.contains("TAGC"));

// Find with mismatches (fuzzy)
let automaton = LevenshteinAutomaton::new("ATCG", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("Motifs (distance ≤1): {:?}", results);
// Finds patterns similar to "ATCG"
```

### Example 8: Incremental Indexing

```rust
use libdictenstein::suffix_automaton::SuffixAutomaton;

let dict = SuffixAutomaton::new();

// Build index incrementally as data arrives
for line in read_stream() {
    dict.insert(&line);

    // Query immediately available
    if dict.contains("ERROR") {
        alert("Error detected!");
    }
}

// Periodic maintenance
if dict.text_count() > 1000 && dict.needs_compaction() {
    dict.compact();
}
```

## Performance Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Construction** | $`\mathcal{O}(n)`$ | n = text length |
| **Insert character** | $`\mathcal{O}(1)`$ amortized | Online construction |
| **Contains (exact)** | $`\mathcal{O}(m)`$ | m = query length |
| **Fuzzy search** | $`\mathcal{O}(m \times d^{2} \times b)`$ | d = distance, b = branching |
| **Compact** | $`\mathcal{O}(s)`$ | s = number of states |

### Benchmark Results

#### Construction

```
Index 10,000-character text:
  SuffixAutomaton:     ~8ms
  DoubleArrayTrie:     ~3ms (but only prefixes)

Index 100,000-character text:
  SuffixAutomaton:     ~85ms
```

#### Query Performance

```
Exact substring search (10K-char text):
  Query "test":        ~450ns
  Query "algorithm":   ~680ns

Fuzzy substring search (distance 2):
  Query "test":        ~38µs
  Query "algorithm":   ~91µs
```

#### Space Usage

```
Text size: 10,000 characters
  States:              ~15,000 (1.5× text length)
  Memory:              ~850 KB

Text size: 100,000 characters
  States:              ~150,000
  Memory:              ~8.5 MB
```

### Comparison with Prefix Dictionaries

```
Task: Index 10,000 words (avg 8 chars = 80K chars total)

                    Construction    Memory      Contains    Fuzzy (d=2)
────────────────────────────────────────────────────────────────────────
DoubleArrayTrie     3.2ms          800 KB      6.6µs       16.3µs
SuffixAutomaton     68ms           6.8 MB      450ns       82µs

Substring matching? ❌             ✅           ❌          ✅
```

**Trade-off**: SuffixAutomaton uses more memory and construction time, but enables substring matching not possible with prefix dictionaries.

## When to Use

### Decision Matrix

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Full-text search** | ✅ SuffixAutomaton | Need substring matching |
| **Code search** | ✅ SuffixAutomaton | Find identifiers anywhere |
| **Log analysis** | ✅ SuffixAutomaton | Error codes mid-line |
| **Bioinformatics** | ✅ SuffixAutomaton | DNA/protein motifs |
| **Autocomplete** | ⚠️ DoubleArrayTrie | Only need prefixes |
| **Spell checking** | ⚠️ DoubleArrayTrie | Whole words only |
| **Dictionary lookup** | ⚠️ DoubleArrayTrie | Much faster |

### Ideal Use Cases

1. **Code Search Engines**
   - Find function/variable names anywhere
   - Handle camelCase, snake_case
   - Fuzzy matching for typos

2. **Document Search**
   - Full-text search within documents
   - Find phrases anywhere
   - Multi-document indexing

3. **Log Monitoring**
   - Search error codes/patterns
   - Real-time log analysis
   - Pattern matching mid-line

4. **Bioinformatics**
   - DNA/RNA sequence search
   - Protein motif finding
   - Genome indexing

5. **Data Mining**
   - Pattern discovery in streams
   - Substring frequency analysis
   - Text analytics

## Related Documentation

- [Dictionary Layer](../README.md) - Overview of all dictionary types
- [DoubleArrayTrie](double-array-trie.md) - Faster alternative for prefix matching
- [DynamicDawg](dynamic-dawg.md) - Dynamic prefix dictionary

## References

### Academic Papers

1. **Blumer, A., Blumer, J., Ehrenfeucht, A., Haussler, D., & McConnell, R. M. (1985)**. "The smallest automaton recognizing the subwords of a text"
   - *Theoretical Computer Science*, 40, 31-55
   - DOI: [10.1016/0304-3975(85)90157-4](https://doi.org/10.1016/0304-3975(85)90157-4)
   - 📄 **Original suffix automaton paper**

2. **Crochemore, M. (1986)**. "Transducers and repetitions"
   - *Theoretical Computer Science*, 45(1), 63-86
   - DOI: [10.1016/0304-3975(86)90041-1](https://doi.org/10.1016/0304-3975(86)90041-1)
   - 📄 Online construction algorithm

3. **Inenaga, S., Hoshino, H., Shinohara, A., Takeda, M., & Arikawa, S. (2005)**. "On-line construction of symmetric compact directed acyclic word graphs"
   - *Discrete Applied Mathematics*, 146(2), 156-179
   - DOI: [10.1016/j.dam.2004.04.012](https://doi.org/10.1016/j.dam.2004.04.012)
   - 📄 Generalized suffix automaton

### Textbooks

4. **Crochemore, M., & Rytter, W. (2002)**. *Jewels of Stringology*
   - World Scientific, Chapter 6
   - ISBN: 978-9810248970
   - 📚 Comprehensive suffix structure coverage

5. **Gusfield, D. (1997)**. *Algorithms on Strings, Trees, and Sequences*
   - Cambridge University Press
   - ISBN: 978-0521585194
   - 📚 Suffix trees and related structures

### Open Access Resources

6. **CP-Algorithms: Suffix Automaton**
   - 📄 [https://cp-algorithms.com/string/suffix-automaton.html](https://cp-algorithms.com/string/suffix-automaton.html)
   - Practical construction algorithms

## Next Steps

- **Prefix Dictionaries**: Compare with [DoubleArrayTrie](double-array-trie.md)
- **Dynamic Updates**: Explore [DynamicDawg](dynamic-dawg.md)
- **Fuzzy Matching**: Read [Levenshtein Automata](../../02-levenshtein-automata/README.md)

---

**Navigation**: [← Dictionary Layer](../README.md) | [DoubleArrayTrie](double-array-trie.md) | [Algorithms Home](../../README.md)
