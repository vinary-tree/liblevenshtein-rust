# Levenshtein Automata Layer

**Navigation**: [← Dictionary Layer](../01-dictionary-layer/README.md) | [Back to Algorithms](../README.md) | [Intersection Layer →](../03-intersection-traversal/README.md)

## Table of Contents

1. [Overview](#overview)
2. [Theory: Levenshtein Distance](#theory-levenshtein-distance)
3. [Finite Automata Approach](#finite-automata-approach)
4. [Algorithm Variants](#algorithm-variants)
5. [State Representation](#state-representation)
6. [Usage Examples](#usage-examples)
7. [Performance Analysis](#performance-analysis)
8. [Algorithm Selection Guide](#algorithm-selection-guide)
9. [References](#references)

## Overview

The Levenshtein Automata Layer provides **finite state machines** for computing approximate string matches efficiently. Instead of comparing every dictionary term individually ($`\mathcal{O}(N)`$ operations, where $`N`$ is the dictionary size), these automata traverse the dictionary graph once, finding all matches within a distance threshold.

![Levenshtein NFA for a query term and error bound k, with match, insertion, deletion, and substitution transitions between positions](../../diagrams/automata/levenshtein-nfa.svg)

*Levenshtein NFA: the non-deterministic automaton whose accepting runs are exactly the edits within distance `k`.*

![Position-set state: a simulated state is the subsumption-reduced set of (boundary, error) positions reachable while reading the query](../../diagrams/automata/position-set-state.svg)

*Position-set state: each simulated state is a subsumption-reduced set of `(i, e)` positions.*

### Key Innovation

Traditional fuzzy matching:
```rust
// Naive approach: Compare query against every term
for term in dictionary {
    if levenshtein_distance(query, term) <= max_distance {
        results.push(term);
    }
}
// Complexity: O(N × M) where N = dictionary size, M = term length
```

Levenshtein automaton approach:
```rust
// Smart approach: Traverse dictionary graph with automaton
let automaton = LevenshteinAutomaton::new(query, max_distance, algorithm);
let results = automaton.query(&dictionary);
// Complexity: O(M × D × B) where M = query length, D = max distance, B = branching factor
// Independent of dictionary size!
```

### Key Advantages

- 🚀 **Sublinear time**: Query time independent of dictionary size
- 🎯 **Exact results**: Finds all matches within distance threshold
- 💾 **Memory efficient**: No need to materialize all comparisons
- 🔧 **Flexible**: Four algorithm variants for different distance semantics
- 🌍 **Unicode support**: Works with both byte-level and character-level dictionaries

## Theory: Levenshtein Distance

### Definition

The **Levenshtein distance** (edit distance) between two strings is the minimum number of single-character edits needed to transform one string into another.

**Allowed operations**:
1. **Insertion**: Add a character
2. **Deletion**: Remove a character
3. **Substitution**: Replace one character with another

### Examples

```
"kitten" → "sitting" = 3 operations:
  1. Substitute k → s:  "sitten"
  2. Substitute e → i:  "sittin"
  3. Insert g:          "sitting"

"café" → "cafe" = 1 operation:
  1. Substitute é → e

"algorithm" → "algorism" = 2 operations:
  1. Delete h:          "algoritm"
  2. Delete t:          "algorism"
```

### Dynamic Programming Algorithm

Classic DP computes distance in $`\mathcal{O}(M \times N)`$ time ($`M`$, $`N`$ the two string lengths):

```
     ""  c  a  f  e
""    0  1  2  3  4
c     1  0  1  2  3
a     2  1  0  1  2
f     3  2  1  0  1
é     4  3  2  1  1

Distance("café", "cafe") = 1
```

**Recurrence**:
```
D[i][j] = min(
    D[i-1][j] + 1,      // Deletion
    D[i][j-1] + 1,      // Insertion
    D[i-1][j-1] + cost  // Substitution (cost = 0 if match, 1 otherwise)
)
```

### Limitations of DP for Fuzzy Search

For fuzzy dictionary search with $`N`$ terms:
- **Time**: $`\mathcal{O}(N \times M \times L)`$ where $`M`$ = query length, $`L`$ = average term length
- **Space**: $`\mathcal{O}(M \times L)`$ per comparison
- **Problem**: Wasteful for large dictionaries!

## Finite Automata Approach

### Key Insight

Instead of computing distance for each term separately, build a **Levenshtein automaton** that recognizes all strings within distance D of a query string.

The automaton can then traverse the dictionary graph in a single pass.

### How It Works

1. **Build automaton** for query "test" with max distance 2
2. **Start at dictionary root** with automaton's initial state
3. **Synchronously traverse**:
   - Dictionary edge labeled 'c' → follow dictionary to child
   - Automaton transition on 'c' → compute new automaton state
4. **If automaton accepts** at a dictionary final state → match found!
5. **Continue** exploring all paths

### State Structure

Each automaton state represents a set of possible "positions" in the query string, along with accumulated edit distances:

```
Query: "test", max distance: 2

Initial state: [(0, 0)]
  Position 0, distance 0

After reading 't':
  [(1, 0)]     ← Matched 't', advance position, distance 0

After reading 'e':
  [(2, 0)]     ← Matched 'e', advance position, distance 0

After reading 'x':
  [(3, 1)]     ← Substitute 's'→'x', advance position, distance 1
  [(2, 1)]     ← Delete 'x', stay at position 2, distance 1
  [(1, 1)]     ← Insert 'x' in query, stay at position 1, distance 1
```

The automaton tracks **all possible ways** the query could align with the input, pruning paths that exceed the distance threshold.

### Advantages Over DP

| Aspect | Dynamic Programming | Levenshtein Automaton |
|--------|---------------------|----------------------|
| **Dictionary traversal** | $`\mathcal{O}(N)`$ separate DPs | $`\mathcal{O}(1)`$ shared traversal |
| **Duplicate prefixes** | Recomputed $`N`$ times | Computed once |
| **Memory** | $`\mathcal{O}(M \times L)`$ per term | $`\mathcal{O}(M \times D)`$ for all terms |
| **Dictionary size** | Linear impact | No impact |

**Example**: For dictionary with 100K terms sharing prefix "test", DP computes "test" 100K times, automaton computes once.

## Algorithm Variants

liblevenshtein provides four algorithm variants for different distance semantics:

### 1. Standard (Classic Levenshtein)

**Operations**: Insertion, Deletion, Substitution

```rust
use liblevenshtein::levenshtein::Algorithm;

let automaton = LevenshteinAutomaton::new("test", 2, Algorithm::Standard);
```

**Use when**: Standard edit distance suffices (most common case)

**Examples**:
- "test" → "text" = 1 (substitute s→x)
- "test" → "est" = 1 (delete t)
- "test" → "tests" = 1 (insert s)

### 2. Transposition (Optimal String Alignment)

**Operations**: Insertion, Deletion, Substitution, **Transposition** (swap adjacent characters)

```rust
use liblevenshtein::levenshtein::Algorithm;

let automaton = LevenshteinAutomaton::new("test", 1, Algorithm::Transposition);
```

**Use when**: Common typing errors include character swaps

**Examples**:
- "test" → "tset" = 1 (transpose e↔s) vs 2 in standard
- "algorithm" → "algorihtm" = 1 (transpose t↔h) vs 2 in standard

**Advantage**: More natural for spell checkers (humans often swap adjacent chars)

**Restriction**: The recurrence is OSA (restricted Damerau), so a substring
cannot be edited twice. It differs from unrestricted Damerau–Levenshtein and
violates the triangle inequality on `CA`, `AC`, `ABC` (`3 > 1 + 1`).

### 3. Unrestricted Damerau–Levenshtein

**Operations**: Insertion, deletion, substitution, and adjacent transposition,
with later edits allowed to revisit an earlier edit's output.

```rust
use liblevenshtein::prelude::*;

let dictionary = DoubleArrayTrie::from_terms(["AC", "ABC", "CA"]);
let transducer = Transducer::with_damerau_levenshtein(dictionary);
let matches: Vec<_> = transducer.query_with_distance("CA", 2).collect();
assert!(matches.iter().any(|candidate| {
    candidate.term == "ABC" && candidate.distance == 2
}));
```

**Use when**: A transposition may compose with a later insertion, deletion, or
substitution and metricity matters. The separating example is `CA` → `ABC`:
unrestricted Damerau distance is 2, while OSA is 3.

The bounded implementation uses a history-carrying continuation and has a
`$`\mathcal{O}(k^2)`$` state envelope. See the dedicated
[literate chapter](../11-true-damerau/README.md) for the recurrence,
subsumption proof, formal evidence, resource ceiling, and corpus results.

### 4. Merge-and-Split

**Operations**: Insertion, Deletion, Substitution, **Merge** (two chars → one), **Split** (one char → two)

```rust
use liblevenshtein::levenshtein::Algorithm;

let automaton = LevenshteinAutomaton::new("test", 2, Algorithm::MergeAndSplit);
```

**Use when**: Handling OCR errors, text normalization

**Examples**:
- "test" → "tst" = 1 (merge e,s → nothing)
- "test" → "teest" = 1 (split e → e,e)

**Advantage**: Better for OCR/handwriting recognition errors

### Comparison Table

| Feature | Standard | Transposition (OSA) | Unrestricted Damerau | Merge-and-Split |
|---------|----------|---------------------|-----------------------|-----------------|
| **Operations** | I, D, S | I, D, S, T | I, D, S, composable T | I, D, S, M, Sp |
| **Transpositions** | ❌ (cost 2) | ✅, non-overlapping | ✅, composable | ❌ (cost 2) |
| **Metric** | yes | no | yes | yes |
| **Adjacent merges** | ❌ | ❌ | ❌ | ✅ (cost 1) |
| **Character splits** | ❌ | ❌ | ❌ | ✅ (cost 1) |
| **Frontier envelope** | $`\mathcal{O}(k)`$ | $`\mathcal{O}(k)`$ | $`\mathcal{O}(k^2)`$ | operation-dependent |
| **Use case** | General fuzzy search | Ordinary swapped keys | Composable edits, metric distance | OCR errors |

**Legend**: I=Insert, D=Delete, S=Substitute, T=Transpose, M=Merge, Sp=Split

## State Representation

### Position-Distance Pairs

Each state is a set of (position, distance) pairs:

```rust
type State = Vec<(usize, usize)>;  // Vec<(position in query, accumulated distance)>
```

**Example**: Query "test", max distance 2

```
After reading "tx":

Standard algorithm:
  [(0, 0), (1, 1), (2, 2)]

Interpretation:
  - (2, 2): Could be at position 2 with distance 2
            "te" matches with substitute e→x, delete s
  - (1, 1): Could be at position 1 with distance 1
            "t" matches, delete x
  - (0, 0): Could be at position 0 with distance 0
            No characters matched yet (all deletions)
```

### State Minimization

States with duplicate positions are merged, keeping minimum distance:

```
Before minimization:
  [(1, 1), (2, 2), (1, 2), (3, 1)]

After minimization:
  [(1, 1), (2, 2), (3, 1)]

Dropped (1, 2) because (1, 1) is better (same position, lower distance)
```

### Acceptance Condition

A state is **accepting** if any position-distance pair satisfies:
```rust
position == query.len() && distance <= max_distance
```

Meaning: We've matched the entire query within the distance budget.

### Transition Function

Given current state and input character, compute next state:

```rust
fn transition(state: &State, input: char, query: &[char], max_dist: usize) -> State {
    let mut next_state = Vec::new();

    for &(pos, dist) in state {
        // Match: advance position, distance unchanged
        if pos < query.len() && query[pos] == input {
            next_state.push((pos + 1, dist));
        }

        // Substitution: advance position, increment distance
        if pos < query.len() && dist < max_dist {
            next_state.push((pos + 1, dist + 1));
        }

        // Deletion: stay at position, increment distance (delete input char)
        if dist < max_dist {
            next_state.push((pos, dist + 1));
        }

        // Insertion: advance position, increment distance (insert into query)
        if pos < query.len() && dist < max_dist {
            // This is handled by not advancing on input
            // (represented in the structure of the state)
        }
    }

    minimize(next_state)  // Remove dominated pairs
}
```

## Usage Examples

### Example 1: Basic Fuzzy Search

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "text", "best", "rest"
]);

// Find terms within distance 1 of "test"
let automaton = LevenshteinAutomaton::new("test", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["best", "rest", "test", "text"] (all within distance 1)
```

### Example 2: Transposition for Typos

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec![
    "algorithm", "logarithm", "align"
]);

// Common typo: swapped letters
let automaton = LevenshteinAutomaton::new("algorihtm", 1, Algorithm::Transposition);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["algorithm"] (distance 1 via transposition)

// Standard algorithm would require distance 2
let automaton_std = LevenshteinAutomaton::new("algorihtm", 1, Algorithm::Standard);
let results_std: Vec<String> = automaton_std.query(&dict).collect();
println!("{:?}", results_std);
// Output: [] (no matches within distance 1)
```

### Example 3: Unicode Fuzzy Matching

```rust
use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrieChar::from_terms(vec![
    "café", "naïve", "résumé"
]);

// Find "cafe" (missing accent)
let automaton = LevenshteinAutomaton::new("cafe", 1, Algorithm::Standard);
let results: Vec<String> = automaton.query(&dict).collect();

println!("{:?}", results);
// Output: ["café"] (distance 1: substitute e→é)
```

### Example 4: Value Filtering During Traversal

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms_with_values(vec![
    ("test", 1),
    ("testing", 1),
    ("text", 2),
    ("best", 2),
]);

// Fuzzy search only in category 1
let automaton = LevenshteinAutomaton::new("test", 1, Algorithm::Standard)
    .with_value_filter(|&category| category == 1);

let results: Vec<String> = automaton.query(&dict).collect();
println!("{:?}", results);
// Output: ["test", "testing"] (only category 1, despite "text" and "best" matching)
```

### Example 5: Distance Threshold Exploration

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec![
    "test", "testing", "tested", "tester", "text", "best", "rest", "fest"
]);

// Show how results grow with distance
for max_distance in 0..=3 {
    let automaton = LevenshteinAutomaton::new("test", max_distance, Algorithm::Standard);
    let results: Vec<String> = automaton.query(&dict).collect();

    println!("Distance {}: {} matches", max_distance, results.len());
    println!("  {:?}", results);
}

// Output:
// Distance 0: 1 matches
//   ["test"]
// Distance 1: 4 matches
//   ["best", "fest", "rest", "test", "text"]
// Distance 2: 7 matches
//   ["best", "fest", "rest", "test", "tested", "tester", "testing", "text"]
// Distance 3: 8 matches
//   ["best", "fest", "rest", "test", "tested", "tester", "testing", "text"]
```

### Example 6: Custom Algorithm Selection

```rust
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::levenshtein::Algorithm;
use liblevenshtein::levenshtein_automaton::LevenshteinAutomaton;

let dict = DoubleArrayTrie::from_terms(vec![
    "form", "from", "forum", "format"
]);

// Query with transposition
let query = "from";
let typo = "form";  // Common transposition error

// Standard: treats transposition as 2 operations
let auto_std = LevenshteinAutomaton::new(typo, 1, Algorithm::Standard);
let results_std: Vec<String> = auto_std.query(&dict).collect();
println!("Standard distance 1: {:?}", results_std);
// Output: ["form"] (exact match only)

// Transposition: treats transposition as 1 operation
let auto_trans = LevenshteinAutomaton::new(typo, 1, Algorithm::Transposition);
let results_trans: Vec<String> = auto_trans.query(&dict).collect();
println!("Transposition distance 1: {:?}", results_trans);
// Output: ["form", "from"] (includes transposition)
```

## Performance Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Automaton construction** | $`\mathcal{O}(M \times D)`$ | $`M`$ = query length, $`D`$ = max distance |
| **Single transition** | $`\mathcal{O}(D^{2})`$ | Process $`D^{2}`$ state components |
| **Query traversal** | $`\mathcal{O}(M \times D^{2} \times B)`$ | $`B`$ = avg branching factor |
| **Total query** | $`\mathcal{O}(M \times D^{2} \times B)`$ | **Independent of dictionary size!** |

### Space Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| **State size** | $`\mathcal{O}(M \times D)`$ | Position-distance pairs |
| **Automaton cache** | $`\mathcal{O}(1)`$ | Reused across queries |
| **Query results** | $`\mathcal{O}(K)`$ | $`K`$ = number of matches |

### Benchmark Results

#### Construction Time (varies by query length)

```
Query length 5, max distance 2:
  Standard:        ~100ns
  Transposition:   ~150ns
  Merge-and-Split: ~200ns
```

#### Query Time (10,000-term dictionary)

```
Query "test", max distance 1:
  Standard:        12.9µs
  Transposition:   18.4µs (+43%)
  Merge-and-Split: 24.7µs (+91%)

Query "test", max distance 2:
  Standard:        16.3µs
  Transposition:   28.1µs (+72%)
  Merge-and-Split: 41.9µs (+157%)
```

**Insight**: Complexity grows with distance and algorithm sophistication.

#### Comparison vs Naive Approach

```
Fuzzy search in 100,000-term dictionary:
  Naive (compare all):  ~250ms
  Automaton:           ~35µs

Speedup: 7,000x
```

### Scaling with Distance

```
Dictionary: 10,000 terms
Query: "test"

Distance │ Matches │ Standard │ Transposition │ Merge-Split
─────────┼─────────┼──────────┼───────────────┼─────────────
0        │ 1       │ 8.2µs    │ 8.5µs         │ 8.9µs
1        │ 12      │ 12.9µs   │ 18.4µs        │ 24.7µs
2        │ 89      │ 16.3µs   │ 28.1µs        │ 41.9µs
3        │ 478     │ 21.7µs   │ 42.3µs        │ 68.1µs
```

**Observations**:
- Performance degrades gracefully with distance
- Transposition ~1.5x slower than standard
- Merge-and-Split ~2.5x slower than standard

## Algorithm Selection Guide

### Decision Flowchart

```
What's your use case?

├─ General fuzzy search / spell check
│  ├─ Common typing errors (swapped letters)?
│  │  ├─ Yes, edits never overlap → Transposition (OSA)
│  │  ├─ Yes, edits may compose or metricity matters → DamerauLevenshtein
│  │  └─ No  → Standard
│
├─ OCR / handwriting recognition
│  └─ Merge-and-Split
│
├─ Maximum performance
│  └─ Standard
│
└─ Academic / linguistic research
   └─ Choose based on linguistic model
```

### Detailed Comparison

| Criterion | Standard | Transposition | Unrestricted Damerau | Merge-and-Split |
|-----------|----------|---------------|-----------------------|-----------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Accuracy (typing)** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Accuracy (OCR)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Recommendations

**Use Standard when:**
- ✅ General-purpose fuzzy matching
- ✅ Performance is critical
- ✅ Transpositions are rare or acceptable as distance 2
- ✅ Simple, well-understood metric needed

**Use Transposition when:**
- ✅ User-facing spell checkers
- ✅ Autocomplete with typo tolerance
- ✅ Keyboard input errors common
- ✅ Natural language text correction

**Use unrestricted Damerau–Levenshtein when:**
- ✅ A later edit may affect a transposed region
- ✅ Triangle-inequality-dependent indexing or reasoning is required
- ✅ The higher `$`\mathcal{O}(k^2)`$` frontier cost is acceptable
- ✅ The service can enforce a small practical budget such as 1 through 3

**Use Merge-and-Split when:**
- ✅ OCR error correction
- ✅ Handwriting recognition
- ✅ Text normalization (e.g., "cannot" vs "can not")
- ✅ Specialized domain with character merging/splitting errors

## Related Documentation

- [Dictionary Layer](../01-dictionary-layer/README.md) - Data structures traversed by automata
- [Intersection Layer](../03-intersection-traversal/README.md) - How automata traverse dictionaries
- [SIMD Optimization](../05-simd-optimization/README.md) - Vectorized acceleration
- Performance Guide - Detailed benchmarks

## References

### Foundational Papers

1. **Schulz, K. U., & Mihov, S. (2002)**. "Fast String Correction with Levenshtein Automata"
   - *International Journal on Document Analysis and Recognition*, 5(1), 67-85
   - DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
   - 📄 **Core algorithm for Levenshtein automata**

2. **Levenshtein, V. I. (1966)**. "Binary codes capable of correcting deletions, insertions, and reversals"
   - *Soviet Physics Doklady*, 10(8), 707-710
   - 📄 Original Levenshtein distance paper

3. **Damerau, F. J. (1964)**. "A technique for computer detection and correction of spelling errors"
   - *Communications of the ACM*, 7(3), 171-176
   - DOI: [10.1145/363958.363994](https://doi.org/10.1145/363958.363994)
   - 📄 Damerau-Levenshtein distance (with transpositions)

### Algorithm Variants

4. **Lowrance, R., & Wagner, R. A. (1975)**. "An extension of the string-to-string correction problem"
   - *Journal of the ACM*, 22(2), 177-183
   - DOI: [10.1145/321879.321880](https://doi.org/10.1145/321879.321880)
   - 📄 Generalized edit distance

5. **Oommen, B. J., & Loke, R. K. S. (1997)**. "Pattern recognition of strings with substitutions, insertions, deletions and generalized transpositions"
   - *Pattern Recognition*, 30(5), 789-800
   - DOI: [10.1016/S0031-3203(96)00101-X](https://doi.org/10.1016/S0031-3203(96)00101-X)
   - 📄 Extended transposition models

### Textbooks

6. **Navarro, G. (2001)**. "A guided tour to approximate string matching"
   - *ACM Computing Surveys*, 33(1), 31-88
   - DOI: [10.1145/375360.375365](https://doi.org/10.1145/375360.375365)
   - 📚 Comprehensive survey of approximate matching algorithms

7. **Gusfield, D. (1997)**. *Algorithms on Strings, Trees, and Sequences*
   - Cambridge University Press
   - ISBN: 978-0521585194
   - 📚 Chapter 11: Edit distance and alignment

### Open Access Resources

8. **CP-Algorithms: String Matching**
   - 📄 [https://cp-algorithms.com/string/string-hashing.html](https://cp-algorithms.com/string/string-hashing.html)
   - Practical algorithms and implementations

## Next Steps

- **Implementation**: Read about [Intersection Layer](../03-intersection-traversal/README.md)
- **Optimization**: Explore [SIMD Acceleration](../05-simd-optimization/README.md)
- **Dictionary**: Review [Dictionary Layer](../01-dictionary-layer/README.md)
- **Practice**: Try the [Usage Examples](../../examples/README.md)

---

**Navigation**: [← Dictionary Layer](../01-dictionary-layer/README.md) | [Back to Algorithms](../README.md) | [Intersection Layer →](../03-intersection-traversal/README.md)
