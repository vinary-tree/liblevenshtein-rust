[← Documentation Index](../../README.md)

# Implementation Mapping: Paper to Code

**How Paper Concepts Map to liblevenshtein-rust**

**Date**: 2025-11-06

**Source paper**: Schulz, K. U. & Mihov, S. (2002). *Fast string correction with Levenshtein automata*. IJDAR 5, 67–85. [doi:10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)

---

## Overview

This document provides a detailed mapping between concepts, algorithms, and structures from the paper "Fast String Correction with Levenshtein-Automata" and their implementations in the liblevenshtein-rust codebase.

**Purpose**: Enable developers to:
1. Understand code by referencing paper sections
2. Verify correctness against theoretical foundations  
3. Extend algorithms based on paper insights
4. Debug by checking against paper specifications

---

## Quick Reference Table

| Paper Concept | Code Location | Type |
|---------------|---------------|------|
| Position i#e | `/src/transducer/position.rs:11-35` | Struct |
| Subsumption $`\sqsubseteq`$ |  `/src/transducer/position.rs:231-269` | Method |
| Characteristic Vector $`\chi`$ |  `/src/transducer/position.rs` | Function |
| Elementary Transition $`\delta`$ |  `/src/transducer/transition.rs:119-438` | Function |
| State Transition $`\Delta`$ |  `/src/transducer/query.rs` | Method |
| Algorithm Variants | `/src/transducer/algorithm.rs` | Enum |
| LEV_n(W) Construction | `/src/transducer/builder.rs` | Builder |
| Imitation Method | `/src/transducer/query.rs:86-188` | Iterator |

---

## Core Data Structures

### Position (Definition 4.0.12)

**Paper Definition**: i#e where $`0 \le  i \le  |W|, 0 \le  e \le  n`$

**Code**: `/src/transducer/position.rs`

```rust
pub struct Position {
    pub term_index: usize,     // Corresponds to 'i' in paper
    pub num_errors: usize,     // Corresponds to 'e' in paper
    pub is_special: bool,      // Flag for transposition/merge/split
}
```

**Mapping**:
- `term_index` = i (index into query word)
- `num_errors` = e (error count)
- `is_special` = t or s flag (for extended operations)
  - false = regular position (i#e_0)
  - true = special position (i#e_1 for transposition or merge/split)

**Usage Example**:
```rust
let pos = Position::new(3, 1, false);  // Represents 3#1 from paper
```

### State (Definition 4.0.18)

**Paper Definition**: Set M of positions with specific properties

**Code**: `State` type alias (likely `HashSet<Position>` or similar)

**Properties Enforced**:
1. Contains base position i#0
2. No position subsumes another
3. All positions reachable from base

**Note**: State validation may be implicit in transition logic rather than explicit checks.

---

## Subsumption Relation (Definition 4.0.15)

**Paper Definition**: i#$`e \sqsubseteq  j`$#f if $`(e < f) \land  (|j-i| \le  f-e)`$

**Code**: `/src/transducer/position.rs` (subsumption logic)

**Implementation Notes**:
- Used to remove redundant positions from states
- Critical for keeping state sets minimal
- May be implemented as method on Position or State

**Example from Paper**:
```
3#0 ⊑ 4#1?  
  Check: 0 < 1 ✓ and |4-3| = 1 ≤ 1-0 = 1 ✓  
  Result: YES
```

**Corresponding Code Logic** (conceptual):
```rust
impl Position {
    pub fn subsumes(&self, other: &Position) -> bool {
        self.num_errors < other.num_errors &&
        (self.term_index.abs_diff(other.term_index) <= 
         other.num_errors - self.num_errors)
    }
}
```

---

## Characteristic Vectors (Definition 4.0.10)

**Paper Definition**: $`\chi (x,V) = \langle b_{1},...,b_v\rangle`$ where b_j = 1 if V[j] = x

**Code**: `/src/transducer/position.rs` (characteristic vector functions)

**Purpose**: Determine which transitions are possible from a position

**Implementation**:
- Likely uses bit-vectors or boolean arrays
- Computed for relevant subword $`W[\pi ]`$
- Used in elementary transition logic

**Example from Paper**:
```
χ('l', "hello") = ⟨0,0,1,1,0⟩
```

**Corresponding Code** (conceptual):
```rust
fn characteristic_vector(ch: char, word: &[char]) -> Vec<bool> {
    word.iter().map(|&c| c == ch).collect()
}
```

---

## Elementary Transitions (Table 4.1, 7.1, 8.1)

**Paper**: Defines transition from single position $`\pi`$ under character x

**Code**: `/src/transducer/transition.rs`

### Standard Operations (Table 4.1 → Lines 119-188)

**Function Signature** (conceptual):
```rust
pub(crate) fn standard_transition(
    curr_state: &State,
    dict_char: char,
    query_chars: &[char],
    max_distance: usize,
) -> State
```

**Implements**:
- **Case 1**: First character matches $`(\chi  = \langle 1,...\rangle)`$
  - Return {(i+1)#e}
  
- **Case 2**: Match later $`(\chi  = \langle 0,...,0,1,...\rangle`$ at position j)
  - Return {i#(e+1), (i+1)#(e+1), (i+j)#(e+j-1)}
  
- **Case 3**: No match $`(\chi  = \langle 0,...,0\rangle)`$
  - Return {i#(e+1), (i+1)#(e+1)}

**Code Structure**:
```rust
// Simplified from actual implementation
for position in curr_state {
    let i = position.term_index;
    let e = position.num_errors;
    
    // Match
    if query_chars[i] == dict_char {
        next_state.insert(Position::new(i + 1, e, false));
    }
    
    // Substitution (if no match and within error budget)
    if query_chars[i] != dict_char && e < max_distance {
        next_state.insert(Position::new(i + 1, e + 1, false));
    }
    
    // Deletion
    if e < max_distance {
        next_state.insert(Position::new(i + 1, e + 1, false));
    }
    
    // Insertion
    if e < max_distance {
        next_state.insert(Position::new(i, e + 1, false));
    }
}
```

### Transposition Operations (Table 7.1 → Lines 195-319)

**Function**: `transposition_transition()`

**Additional Logic**:
- Handles `is_special` flag (t in paper)
- Detects potential transpositions $`(W[i+1] \ne  x`$ but W[i+2] = x)
- Completes transposition when special position matches expected character

**Code Pattern**:
```rust
// Check for transposition opportunity
if !is_special && i + 1 < query_len {
    let next_char = query_chars[i + 1];
    if query_chars[i] != dict_char && next_char == dict_char {
        // Set special flag for transposition
        next_state.insert(Position::new(i, e + 1, true));
    }
}

// Complete transposition
if is_special && query_chars[i] == dict_char {
    next_state.insert(Position::new(i + 1, e, false));
}
```

### Merge/Split Operations (Table 8.1 → Lines 327-438)

**Function**: `merge_split_transition()`

**Additional Logic**:
- Merge: Skip two characters in query (i+2) for one dict character
- Split: Consume two input characters for one query character
- Uses `is_special` flag (s in paper)

---

## Algorithm Variants (Chapter 4, 7, 8)

**Paper**: Three families of automata

**Code**: `/src/transducer/algorithm.rs`

```rust
pub enum Algorithm {
    Standard,        // Chapters 4-6: Insertions, deletions, substitutions
    Transposition,   // Chapter 7: + adjacent character swaps
    MergeAndSplit,   // Chapter 8: + two chars ↔ one char
}
```

**Usage**:
```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)  // Choose variant
    .build_from_iter(words);
```

---

## Automaton Construction

### Builder Pattern (Chapter 5)

**Paper**: Construction algorithm using parametric tables

**Code**: `/src/transducer/builder.rs`

**Builder Methods**:
```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    // Other configuration fields
}

impl<D> TransducerBuilder<D> {
    pub fn new() -> Self { /* ... */ }
    
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
    
    pub fn build_from_iter<I>(self, terms: I) -> Transducer<D> {
        // Construct dictionary and initialize automaton
    }
}
```

**Corresponds to**: Algorithm in Theorem 5.2.1

### Imitation Method (Chapter 6)

**Paper**: Simulate LEV_n(W) without explicit construction

**Code**: `/src/transducer/query.rs:86-188`

**Implementation**:
```rust
pub struct QueryIterator<D> {
    dictionary: &D,
    query_chars: Vec<char>,
    algorithm: Algorithm,
    max_distance: usize,
    // State management for traversal
}

impl<D> Iterator for QueryIterator<D> {
    fn next(&mut self) -> Option<String> {
        // Parallel traversal of dictionary and simulated automaton
        // Uses transition functions based on algorithm variant
    }
}
```

**Corresponds to**: Algorithm in Chapter 6, Figure 6.1

**Key Aspects**:
1. On-demand state generation
2. Parallel traversal of dictionary and Levenshtein automaton
3. Uses transition functions (standard/transposition/merge-split)
4. Returns words when both dictionary and automaton reach final states

---

## Transition Function Dispatch

**Paper**: $`\Delta (M,x)`$ uses algorithm-specific elementary transitions

**Code**: `/src/transducer/query.rs` (transition dispatch logic)

**Conceptual Implementation**:
```rust
impl QueryIterator {
    fn next_state(&self, dict_char: char) -> State {
        match self.algorithm {
            Algorithm::Standard => {
                standard_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                )
            }
            Algorithm::Transposition => {
                transposition_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                )
            }
            Algorithm::MergeAndSplit => {
                merge_split_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                )
            }
        }
    }
}
```

---

## Dictionary Integration

**Paper**: Dictionary automaton $`A^D`$ traversed in parallel with LEV_n(W)

**Code**: Dictionary implementations live in the `libdictenstein` crate (the dictionary family was extracted from `liblevenshtein-rust` into its own crate; see the project's dictionary-family layout). Only `src/dictionary/mod.rs` and `src/dictionary/phonetic_normalized.rs` remain in this crate as the integration surface.

**Dictionary Types** (in `libdictenstein`):
- DAWG (Directed Acyclic Word Graph)
- Dynamic DAWG (`DynamicDawg`, `DynamicDawgChar`)
- Suffix Automaton (`SuffixAutomaton`, `SuffixAutomatonChar`)
- Double Array Trie (`DoubleArrayTrie`, `DoubleArrayTrieChar`)

**Interface** (conceptual):
```rust
pub trait Dictionary {
    fn root(&self) -> NodeRef;
    fn transition(&self, node: NodeRef, ch: char) -> Option<NodeRef>;
    fn is_final(&self, node: NodeRef) -> bool;
}
```

**Parallel Traversal**:
```rust
// Conceptual parallel traversal
let mut dict_node = dictionary.root();
let mut automaton_state = initial_state();

for ch in input_chars {
    dict_node = dictionary.transition(dict_node, ch)?;
    automaton_state = next_state(automaton_state, ch);
    
    if dictionary.is_final(dict_node) && is_accepting(automaton_state) {
        yield current_word;
    }
}
```

---

## Parametric Tables (Chapter 5)

**Paper**: Tables T_n defining state types and transitions for fixed degree n

**Code**: **Not explicitly stored as tables** in current implementation

**Reason**: The imitation method (Chapter 6) computes transitions on-demand using characteristic vectors, avoiding need to materialize full parametric tables.

**Implicit Encoding**: The transition functions (`standard_transition`, etc.) encode the table logic:
- Input: Current state, character, characteristic vectors
- Output: Next state
- Logic: Implements Cases 1-3 from Table 4.1 (and extensions for Tables 7.1, 8.1)

**Trade-off**:
- Paper's table approach: $`\mathcal{O}(1)`$ lookup, $`\mathcal{O}(4^n)`$ preprocessing
- Code's on-demand approach: $`\mathcal{O}(\text{state} \text{size})`$ computation, no preprocessing

Both have $`\mathcal{O}(\lvert W\rvert)`$ total complexity for constructing LEV_n(W).

---

## Complexity Verification

### Construction Time: $`\mathcal{O}(\lvert W\rvert)`$

**Paper**: Theorem 5.2.1

**Code**: 
- Builder creates query iterator: $`\mathcal{O}(\lvert W\rvert)`$ to store query characters
- State initialization: $`\mathcal{O}(1)`$
- Per-transition computation: $`\mathcal{O}(\text{state} \text{size})`$ = $`\mathcal{O}(1)`$ for fixed n

**Total**: $`\mathcal{O}(\lvert W\rvert)`$ ✓

### Query Time: $`\mathcal{O}(\lvert D\rvert)`$

**Paper**: Chapter 3, parallel traversal

**Code**:
- Traverse dictionary edges: $`\mathcal{O}(\lvert D\rvert)`$
- Compute transitions: $`\mathcal{O}(\lvert D\rvert \times  \text{state} \text{size})`$ = $`\mathcal{O}(\lvert D\rvert)`$ for fixed n

**Total**: $`\mathcal{O}(\lvert D\rvert)`$ ✓

---

## Error Handling

**Paper**: Implicitly assumes valid inputs

**Code**: Rust's type system and error handling

**Safety Mechanisms**:
- `usize` for indices (no negative indices)
- Bounds checking on array accesses
- Option/Result types for fallible operations

**Example**:
```rust
// Safe array access
if i < query_chars.len() {
    let ch = query_chars[i];  // Guaranteed safe
}
```

---

## Testing and Validation

**Paper**: Experimental results (Chapter 8.3)

**Code**: Test suites in `/tests/`

**Test Categories**:
1. **Unit Tests**: Individual transition functions
2. **Integration Tests**: End-to-end queries
3. **Property Tests**: Correctness properties from paper
4. **Benchmark Tests**: Performance validation

**Validation Against Paper**:
- Compare results with known Levenshtein distances
- Verify automaton accepts L_Lev(n,W)
- Check performance matches $`\mathcal{O}(\lvert W\rvert)`$ + $`\mathcal{O}(\lvert D\rvert)`$ complexity

---

## Optimization Techniques

### From Paper

1. **Subsumption**: Remove redundant positions (Definition 4.0.15)
2. **Raising Lemma**: Reuse lower-degree computations (Lemma 4.0.27)
3. **Characteristic Vectors**: Compact representation of character distribution

### In Code

1. **Lazy Evaluation**: Compute states on-demand (imitation method)
2. **State Deduplication**: HashSet/BTreeSet for state storage
3. **Early Termination**: Stop when error budget exceeded

---

## Extension Points

### Adding New Edit Operations

**Paper Approach** (Chapters 7-8):
1. Define extended positions (i#e_flag)
2. Define extended subsumption
3. Create transition table
4. Prove $`\mathcal{O}(\lvert W\rvert)`$ complexity

**Code Approach**:
1. Add flag to `Position` struct
2. Extend transition function
3. Add to `Algorithm` enum
4. Implement in `transition.rs`

**Example**: Adding context-dependent costs

```rust
// Extend Position
pub struct Position {
    pub term_index: usize,
    pub num_errors: usize,
    pub is_special: bool,
    pub context: Context,  // NEW: track context
}

// Add Algorithm variant
pub enum Algorithm {
    Standard,
    Transposition,
    MergeAndSplit,
    ContextDependent,  // NEW
}

// Implement transition
pub(crate) fn context_transition(...) -> State {
    // Use context to determine costs
}
```

### Universal Levenshtein Automata (Restricted Substitutions)

**Paper Extension**: See `/docs/research/universal-levenshtein/`

**Code Integration**: Add substitution set parameter
```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    substitution_set: Option<SubstitutionSet>,  // NEW
}
```

Modify transition logic to check substitution validity.

---

## Debugging Guide

### Problem: Incorrect matches returned

**Check**:
1. Transition functions implement Tables 4.1/7.1/8.1 correctly
2. Subsumption logic matches Definition 4.0.15
3. Characteristic vectors computed correctly

**Tool**: Add logging to transition functions, compare with paper examples

### Problem: Missing valid matches

**Check**:
1. Error budget not exceeded prematurely
2. All transitions from Table 4.1 implemented
3. Final state detection correct

**Tool**: Trace automaton state during query execution

### Problem: Performance issues

**Check**:
1. Subsumption removing redundant positions
2. State size not growing beyond $`\mathcal{O}(n)`$ positions
3. Dictionary traversal efficient

**Tool**: Profile transition function calls, measure state sizes

---

## Further Resources

### Paper Sections

- **Definitions**: Chapter 4 for all core concepts
- **Algorithms**: Chapter 5 for construction, Chapter 6 for imitation
- **Extensions**: Chapters 7-8 for transposition/merge/split
- **Proofs**: Throughout, especially Theorem 4.0.32

### Code Files

- **Core**: `position.rs`, `transition.rs`, `algorithm.rs`
- **Construction**: `builder.rs`
- **Query**: `query.rs`
- **Dictionary**: `dictionary/` module

### Documentation

- [README.md](./README.md): Overview
- [PAPER_SUMMARY.md](./PAPER_SUMMARY.md): Complete chapter summaries
- [glossary.md](./glossary.md): Term definitions

---

**Last Updated**: 2025-11-06
**Status**: Complete mapping of paper to code
**Next**: Implement new features based on paper insights
