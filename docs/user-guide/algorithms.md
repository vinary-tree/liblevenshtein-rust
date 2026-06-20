# Levenshtein Algorithm Guide

**Version**: 0.9.1
**Last Updated**: 2026-06-19

This guide explains the Levenshtein distance algorithms supported by liblevenshtein-rust and when to use each one.

The three operation sets supported by the transducer are summarised below. Each variant simply adds elementary edit operations to the standard set; the additions are highlighted per algorithm.

![Levenshtein operation sets: Standard (insert, delete, substitute), Transposition (adds adjacent swap), and Merge-and-Split (adds character merge and split)](../diagrams/automata/operation-sets.svg)

## What is Levenshtein Distance?

The Levenshtein distance is a metric for measuring the difference between two strings. It counts the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into another.

**Example:**
- "kitten" → "sitting" requires 3 edits (distance = 3):
  1. kitten → sitten (substitute 'k' with 's')
  2. sitten → sittin (substitute 'e' with 'i')
  3. sittin → sitting (insert 'g')

## Supported Algorithms

liblevenshtein-rust supports three variants of Levenshtein distance:

### 1. Standard Levenshtein

**Operations**: Insert, Delete, Substitute

**Use Cases:**
- General string matching
- Spell checking
- Fuzzy search in databases
- Data deduplication

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec!["hello", "world", "test"]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Find terms within distance 2 of "helo"
for candidate in transducer.query_with_distance("helo", 2) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
hello: 1  (insert 'l')
```

**When to use:** Default choice for most fuzzy matching use cases.

### 2. Transposition (Damerau-Levenshtein)

**Operations**: Insert, Delete, Substitute, Transposition

**Transposition**: Swap two adjacent characters (counted as 1 edit instead of 2)

**Use Cases:**
- Typo correction (common typo pattern)
- Keyboard input errors
- OCR with character swapping
- User input validation

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "best", "rest"
]);
let transducer = Transducer::new(dict, Algorithm::Transposition);

// "tset" is a transposition of "test" (swapped 's' and 'e')
for candidate in transducer.query_with_distance("tset", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Output:**
```
test: 1  (transposition)
```

**With Standard algorithm**, "tset" → "test" requires 2 edits (substitute twice), so it wouldn't be found with distance 1.

**When to use:** When you expect users to make transposition errors (very common with typing).

### 3. Merge and Split

**Operations**: Insert, Delete, Substitute, Merge, Split

- **Merge**: Combine two characters into one (e.g., "ab" → "a")
- **Split**: Split one character into two (e.g., "a" → "ab")

**Use Cases:**
- OCR (Optical Character Recognition) errors
- Concatenation/separation issues
- Word segmentation errors
- Character recognition with merge/split ambiguity

**Example:**

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "hello", "world"
]);
let transducer = Transducer::new(dict, Algorithm::MergeAndSplit);

// "te st" can be merged to "test"
// "hel lo" can be merged to "hello"
for candidate in transducer.query_with_distance("te st", 2) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**When to use:** When dealing with OCR output or text where words may be incorrectly split or merged.

## Performance Characteristics

### Time Complexity

All algorithms have similar time complexity:
- **Query time**: `𝒪(n × m × k)`, where:
  - `n` = query term length
  - `m` = dictionary size
  - `k` = maximum distance
- **In practice**: Much faster due to early termination and pruning

### Space Complexity

- **Standard**: `𝒪(n × k)` state space
- **Transposition**: `𝒪(n × k)` state space with transposition tracking
- **Merge and Split**: `𝒪(n × k)` state space with merge/split tracking

### Benchmark Comparison

Relative query performance (DoubleArrayTrie, max distance = 2):

| Algorithm | Relative Speed | Use Case |
|-----------|----------------|----------|
| Standard | 1.0× (baseline) | General fuzzy matching |
| Transposition | 0.9× | Typo correction |
| Merge and Split | 0.7× | OCR error handling |

**Note**: With SIMD enabled, all algorithms see 20-64% performance improvements.

## Algorithm Selection Guide

### Decision Tree

```
Do you need to handle...
│
├─ General typos and spelling errors?
│  └─> Use Algorithm::Standard
│
├─ Typing mistakes with swapped characters?
│  └─> Use Algorithm::Transposition
│
└─ OCR errors or word segmentation issues?
   └─> Use Algorithm::MergeAndSplit
```

### Real-World Examples

**Code Completion (IDE):**
```rust
// Users make transposition errors when typing fast
Algorithm::Transposition
```

**Spell Checker:**
```rust
// General spelling errors
Algorithm::Standard
```

**OCR Post-Processing:**
```rust
// Characters may be merged or split incorrectly
Algorithm::MergeAndSplit
```

**Database Fuzzy Search:**
```rust
// General matching with typos
Algorithm::Standard
```

**Form Input Validation:**
```rust
// Catch typing errors including transpositions
Algorithm::Transposition
```

## Distance Threshold Selection

Choosing the right maximum distance (`max_distance`) is crucial:

### Guidelines

- **Distance 1**: Strict matching, catches single-character errors
  - Best for: Short words (3-5 chars), real-time autocomplete
  - Example: "test" matches "test", "fest", "tess", "tes"

- **Distance 2**: Moderate matching, most common choice
  - Best for: Medium words (6-10 chars), spell checking
  - Example: "test" matches above + "tent", "best", "tests"

- **Distance 3+**: Loose matching, more false positives
  - Best for: Long words (10+ chars), fuzzy name matching
  - Use with caution: can return many irrelevant results

### Rule of Thumb

`max_distance ≈ word_length / 4`

Examples:
- 4-letter word: distance 1
- 8-letter word: distance 2
- 12-letter word: distance 3

### Testing Your Distance Threshold

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester",
    "best", "rest", "west", "nest"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Test different distances
for max_dist in 1..=3 {
    println!("\nDistance {}:", max_dist);
    for candidate in transducer.query_with_distance("test", max_dist).sorted() {
        println!("  {}: {}", candidate.term, candidate.distance);
    }
}
```

## Advanced Usage

### Combining with Filters

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester", "best"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Only words with distance ≤ 1 and starting with 't'
for candidate in transducer
    .query_with_distance("test", 2)
    .filter(|c| c.distance <= 1 && c.term.starts_with('t'))
{
    println!("{}: {}", candidate.term, candidate.distance);
}
```

### Prefix Matching

```rust
use liblevenshtein::prelude::*;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "apple", "application"
]);
let transducer = Transducer::new(dict, Algorithm::Standard);

// Only match terms with prefix "tes"
for candidate in transducer
    .query_with_distance("test", 1)
    .with_prefix("tes")
{
    println!("{}", candidate.term);
}
```

## Unicode Considerations

For correct character-level distances with Unicode text, use character-level dictionaries:

```rust
use liblevenshtein::prelude::*;

// ❌ Byte-level (incorrect for Unicode)
let dict_byte = DoubleArrayTrie::from_terms(vec!["café", "naïve"]);

// ✅ Character-level (correct for Unicode)
let dict_char = DoubleArrayTrieChar::from_terms(vec!["café", "naïve"]);

let transducer = Transducer::new(dict_char, Algorithm::Standard);

// Now distance is calculated correctly for multi-byte characters
for candidate in transducer.query_with_distance("cafe", 1) {
    println!("{}: {}", candidate.term, candidate.distance);
}
```

**Trade-offs:**
- ~5% performance overhead
- 4× memory for edge labels
- Correct Unicode Levenshtein distances

## Related Documentation

- [Getting Started](getting-started.md) - Basic usage guide
- [Features](features.md) - Complete feature list
- [Code Completion Guide](code-completion.md) - Building code completion
- [Glossary](../GLOSSARY.md) - Definitions of terms used throughout the docs
- [Architecture Overview](../architecture/overview.md) - How the crates fit together
- [Benchmarks](../benchmarks/) - Performance measurements

## References

- [Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Damerau-Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
- Schulz, K. U., & Mihov, S. (2002). "Fast string correction with Levenshtein automata." *International Journal on Document Analysis and Recognition (IJDAR)*, 5(1), 67–85. DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)

---

[← Documentation Index](../README.md)
