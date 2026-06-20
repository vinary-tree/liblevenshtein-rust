[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Implementation Plan

**Date**: 2025-11-06
**Status**: Planning Phase
**Estimated Duration**: 2-4 weeks

---

## Overview

This document provides a detailed, phase-by-phase implementation plan for adding Universal Levenshtein Automata (restricted substitutions) to liblevenshtein-rust.

**Implementation Approach**: Option B (Configuration-Based) - Add substitution_set as optional field in TransducerBuilder.

---

## Implementation Phases

### Phase 1: Core Restricted Substitutions (1-2 weeks)

**Goal**: Implement basic SubstitutionSet structure and integrate with Standard algorithm.

**Deliverables**:
- ✅ SubstitutionSet structure with HashSet backend
- ✅ Modified transition functions with substitution checks
- ✅ Extended Builder API
- ✅ Basic unit tests
- ✅ End-to-end integration test

---

#### Task 1.1: Create SubstitutionSet Structure (2 days)

**File**: `/src/transducer/substitution.rs` (NEW)

**Implementation**:

```rust
use std::collections::HashSet;

/// Represents a set of allowed character substitutions.
///
/// Used to restrict which character pairs can be substituted in
/// Levenshtein distance computation. If (a, b) ∈ S, then character
/// `a` can be substituted for character `b` with cost = 1.
/// Otherwise, substitution is not allowed (cost = ∞).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstitutionSet {
    /// Set of allowed (source, target) character pairs.
    allowed: HashSet<(char, char)>,
}

impl SubstitutionSet {
    /// Creates a new empty substitution set.
    ///
    /// Note: An empty set means NO substitutions are allowed
    /// (only insert/delete operations).
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Creates an unrestricted substitution set.
    ///
    /// This represents the standard Levenshtein distance where
    /// all character substitutions are allowed.
    pub fn unrestricted() -> Self {
        // In practice, we'll use Option<SubstitutionSet> where
        // None = unrestricted, so this is mainly for documentation.
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Adds an allowed substitution from `a` to `b`.
    ///
    /// # Example
    /// ```
    /// let mut set = SubstitutionSet::new();
    /// set.add('a', 's');  // Allow 'a' → 's' substitution
    /// assert!(set.is_allowed('a', 's'));
    /// assert!(!set.is_allowed('s', 'a'));  // Not bidirectional
    /// ```
    pub fn add(&mut self, a: char, b: char) {
        self.allowed.insert((a, b));
    }

    /// Adds an allowed substitution in both directions.
    ///
    /// Equivalent to calling `add(a, b)` and `add(b, a)`.
    ///
    /// # Example
    /// ```
    /// let mut set = SubstitutionSet::new();
    /// set.add_bidirectional('a', 's');
    /// assert!(set.is_allowed('a', 's'));
    /// assert!(set.is_allowed('s', 'a'));  // Bidirectional
    /// ```
    pub fn add_bidirectional(&mut self, a: char, b: char) {
        self.allowed.insert((a, b));
        self.allowed.insert((b, a));
    }

    /// Checks if substituting `a` for `b` is allowed.
    ///
    /// # Returns
    /// `true` if (a, b) ∈ S, `false` otherwise.
    #[inline]
    pub fn is_allowed(&self, a: char, b: char) -> bool {
        self.allowed.contains(&(a, b))
    }

    /// Returns the number of allowed substitution pairs.
    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    /// Checks if the substitution set is empty.
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

impl Default for SubstitutionSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_unidirectional() {
        let mut set = SubstitutionSet::new();
        set.add('a', 'b');

        assert!(set.is_allowed('a', 'b'));
        assert!(!set.is_allowed('b', 'a'));
    }

    #[test]
    fn test_add_bidirectional() {
        let mut set = SubstitutionSet::new();
        set.add_bidirectional('x', 'y');

        assert!(set.is_allowed('x', 'y'));
        assert!(set.is_allowed('y', 'x'));
    }

    #[test]
    fn test_empty_set() {
        let set = SubstitutionSet::new();

        assert!(set.is_empty());
        assert!(!set.is_allowed('a', 'b'));
    }
}
```

**Validation**:
- [ ] All unit tests pass
- [ ] Documentation compiles correctly
- [ ] No clippy warnings

---

#### Task 1.2: Add SubstitutionSet to TransducerBuilder (1 day)

**File**: `/src/transducer/builder.rs`

**Changes**:

```rust
// Add field to TransducerBuilder struct
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    substitution_set: Option<SubstitutionSet>,  // NEW FIELD
    // ... other existing fields
}

// Add builder methods
impl<D> TransducerBuilder<D> {
    pub fn new() -> Self {
        Self {
            algorithm: Algorithm::Standard,
            substitution_set: None,  // Default: unrestricted
            // ... other default values
        }
    }

    /// Configures the transducer to use restricted substitutions.
    ///
    /// When set, only substitutions in the provided set are allowed
    /// during fuzzy search. This is useful for keyboard-proximity
    /// spell checking, OCR error correction, phonetic matching, etc.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::prelude::*;
    ///
    /// let mut qwerty = SubstitutionSet::new();
    /// qwerty.add_bidirectional('a', 's');
    /// qwerty.add_bidirectional('a', 'd');
    /// // ... add more keyboard-adjacent pairs
    ///
    /// let dict = TransducerBuilder::new()
    ///     .algorithm(Algorithm::Standard)
    ///     .with_substitution_set(qwerty)
    ///     .build_from_iter(words);
    /// ```
    pub fn with_substitution_set(mut self, set: SubstitutionSet) -> Self {
        self.substitution_set = Some(set);
        self
    }
}
```

**Validation**:
- [ ] Compiles without errors
- [ ] Builder pattern still works correctly
- [ ] Documentation example compiles

---

#### Task 1.3: Modify Transition Functions (2-3 days)

**File**: `/src/transducer/transition.rs`

**Changes to Standard algorithm transition** (lines 119-188):

```rust
// OLD signature
pub(crate) fn standard_transition(
    /* existing parameters */
) -> State {
    // ...
}

// NEW signature
pub(crate) fn standard_transition(
    /* existing parameters */
    substitution_set: Option<&SubstitutionSet>,  // NEW PARAMETER
) -> State {
    // ... existing setup ...

    for &Position { term_index: i, num_errors: e, is_special: _ } in curr_state {
        if i < query_len {
            let query_char = query_chars[i];

            // Match: unchanged
            if query_char == dict_char {
                next_state.insert(Position::new(i + 1, e, false));
            }

            // Substitution: CHECK VALIDITY
            if query_char != dict_char && e < max_distance {
                // NEW: Check if substitution is allowed
                let substitution_allowed = match substitution_set {
                    None => true,  // No restrictions = all allowed
                    Some(s) => s.is_allowed(query_char, dict_char),
                };

                if substitution_allowed {
                    next_state.insert(Position::new(i + 1, e + 1, false));
                }
            }

            // Deletion: unchanged
            if e < max_distance {
                next_state.insert(Position::new(i + 1, e + 1, false));
            }
        }

        // Insertion: unchanged
        if e < max_distance {
            next_state.insert(Position::new(i, e + 1, false));
        }
    }

    next_state
}
```

**Similar changes needed** for:
- Transposition algorithm (lines 195-319)
- MergeAndSplit algorithm (lines 327-438)

**Note**: Transposition and MergeAndSplit need careful analysis:
- Transposition operation itself is unrestricted
- But character comparisons in transposition logic may need substitution checks
- MergeAndSplit operations are unrestricted
- But adjacent substitutions need validation

**Validation**:
- [ ] All existing tests still pass (backward compatibility)
- [ ] New test with empty substitution set (no substitutions allowed)
- [ ] New test with specific substitution set

---

#### Task 1.4: Thread Substitution Set Through Query Pipeline (1 day)

**File**: `/src/transducer/query.rs`

**Changes**:

```rust
// Query iterator needs access to substitution_set from builder
impl<D> QueryIterator<D> {
    pub(crate) fn new(
        /* existing parameters */
        substitution_set: Option<SubstitutionSet>,  // NEW
    ) -> Self {
        // Store substitution_set in QueryIterator struct
        // Pass to transition functions during traversal
    }

    fn next_state(&mut self, dict_char: char) -> State {
        // Pass substitution_set reference to transition function
        match self.algorithm {
            Algorithm::Standard => {
                standard_transition(
                    /* existing args */
                    self.substitution_set.as_ref(),  // NEW
                )
            }
            Algorithm::Transposition => {
                transposition_transition(
                    /* existing args */
                    self.substitution_set.as_ref(),  // NEW
                )
            }
            Algorithm::MergeAndSplit => {
                merge_split_transition(
                    /* existing args */
                    self.substitution_set.as_ref(),  // NEW
                )
            }
        }
    }
}
```

**Validation**:
- [ ] Query still works with None (unrestricted)
- [ ] Query respects substitution restrictions

---

#### Task 1.5: Unit Tests (1-2 days)

**File**: `/src/transducer/substitution.rs` (tests module)

**Test cases**:

1. **Basic operations**:
   - Add unidirectional substitution
   - Add bidirectional substitution
   - Check allowed/disallowed substitutions
   - Empty set behavior

2. **Builder integration**:
   - Build with no substitution set (None)
   - Build with custom substitution set
   - Verify backward compatibility

3. **Transition function**:
   - Allowed substitution creates transition
   - Disallowed substitution blocks transition
   - None set allows all substitutions

**File**: `/tests/integration/universal_la.rs` (NEW)

**End-to-end test**:

```rust
#[test]
fn test_keyboard_proximity_typo() {
    // Create simple QWERTY adjacency set
    let mut qwerty = SubstitutionSet::new();
    qwerty.add_bidirectional('t', 'y');  // 't' and 'y' are adjacent
    qwerty.add_bidirectional('e', 's');  // 'e' and 's' adjacent
    qwerty.add_bidirectional('t', 'r');
    // Not adding ('x', 's') - they're far apart on keyboard

    let words = vec!["test", "rest", "best"];

    let dict = TransducerBuilder::new()
        .algorithm(Algorithm::Standard)
        .with_substitution_set(qwerty)
        .build_from_iter(words);

    // Query: "tesy" - 'y' adjacent to 't' on keyboard
    let results: Vec<_> = dict.fuzzy_search("tesy", 1).collect();
    assert!(results.contains(&"test"));  // Should find (y→t substitution allowed)

    // Query: "texs" - 'x' NOT adjacent to 's'
    let results: Vec<_> = dict.fuzzy_search("texs", 1).collect();
    assert!(!results.contains(&"test"));  // Should NOT find (x→s not allowed)
}
```

**Validation**:
- [ ] All tests pass
- [ ] Coverage > 90% for new code

---

### Phase 2: Practical Use Cases (1 week)

**Goal**: Add preset substitution sets for common use cases.

**Deliverables**:
- ✅ QWERTY keyboard preset
- ✅ AZERTY keyboard preset
- ✅ Dvorak keyboard preset
- ✅ OCR confusion preset
- ✅ Phonetic English preset
- ✅ Documentation and examples

---

#### Task 2.1: QWERTY Keyboard Preset (1 day)

**File**: `/src/transducer/substitution.rs`

**Implementation**:

```rust
impl SubstitutionSet {
    /// Creates a substitution set for QWERTY keyboard proximity.
    ///
    /// Allows substitutions between horizontally and diagonally
    /// adjacent keys on a standard QWERTY keyboard layout.
    ///
    /// Useful for spell-checking typos caused by finger slips.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = TransducerBuilder::new()
    ///     .with_substitution_set(SubstitutionSet::qwerty())
    ///     .build_from_iter(words);
    ///
    /// // "tesy" → "test" (y↔t adjacent)
    /// let results: Vec<_> = dict.fuzzy_search("tesy", 1).collect();
    /// assert!(results.contains(&"test"));
    /// ```
    pub fn qwerty() -> Self {
        let mut set = SubstitutionSet::new();

        // Row 1: q w e r t y u i o p
        let row1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'];
        for i in 0..row1.len() - 1 {
            set.add_bidirectional(row1[i], row1[i + 1]);
        }

        // Row 2: a s d f g h j k l
        let row2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'];
        for i in 0..row2.len() - 1 {
            set.add_bidirectional(row2[i], row2[i + 1]);
        }

        // Row 3: z x c v b n m
        let row3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm'];
        for i in 0..row3.len() - 1 {
            set.add_bidirectional(row3[i], row3[i + 1]);
        }

        // Diagonal adjacencies (row 1 ↔ row 2)
        let diagonals_1_2 = [
            ('q', 'a'), ('w', 'a'), ('w', 's'), ('e', 's'), ('e', 'd'),
            ('r', 'd'), ('r', 'f'), ('t', 'f'), ('t', 'g'), ('y', 'g'),
            ('y', 'h'), ('u', 'h'), ('u', 'j'), ('i', 'j'), ('i', 'k'),
            ('o', 'k'), ('o', 'l'), ('p', 'l'),
        ];
        for (a, b) in diagonals_1_2 {
            set.add_bidirectional(a, b);
        }

        // Diagonal adjacencies (row 2 ↔ row 3)
        let diagonals_2_3 = [
            ('a', 'z'), ('s', 'z'), ('s', 'x'), ('d', 'x'), ('d', 'c'),
            ('f', 'c'), ('f', 'v'), ('g', 'v'), ('g', 'b'), ('h', 'b'),
            ('h', 'n'), ('j', 'n'), ('j', 'm'), ('k', 'm'),
        ];
        for (a, b) in diagonals_2_3 {
            set.add_bidirectional(a, b);
        }

        set
    }
}
```

**Builder convenience method**:

```rust
impl<D> TransducerBuilder<D> {
    /// Configures QWERTY keyboard proximity constraints.
    ///
    /// Shorthand for `.with_substitution_set(SubstitutionSet::qwerty())`.
    pub fn with_qwerty_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::qwerty())
    }
}
```

**Validation**:
- [ ] All adjacencies tested
- [ ] Example compiles and runs
- [ ] Documentation clear

---

#### Task 2.2: OCR Confusion Preset (1 day)

**Implementation**:

```rust
impl SubstitutionSet {
    /// Creates a substitution set for OCR visual confusions.
    ///
    /// Allows substitutions between visually similar characters
    /// commonly confused in optical character recognition.
    ///
    /// # Example Confusions
    /// - `0` ↔ `O` (digit zero vs letter O)
    /// - `1` ↔ `I` ↔ `l` (digit one, capital I, lowercase L)
    /// - `5` ↔ `S`
    /// - `8` ↔ `B`
    /// - etc.
    pub fn ocr_confusions() -> Self {
        let mut set = SubstitutionSet::new();

        // Digit-letter confusions
        set.add_bidirectional('0', 'O');
        set.add_bidirectional('0', 'o');
        set.add_bidirectional('1', 'I');
        set.add_bidirectional('1', 'l');
        set.add_bidirectional('I', 'l');
        set.add_bidirectional('2', 'Z');
        set.add_bidirectional('5', 'S');
        set.add_bidirectional('8', 'B');

        // Letter-letter confusions
        set.add_bidirectional('c', 'e');
        set.add_bidirectional('n', 'm');
        set.add_bidirectional('u', 'v');
        set.add_bidirectional('r', 'n');

        set
    }
}
```

**Validation**:
- [ ] OCR test cases pass
- [ ] Documentation with examples

---

#### Task 2.3: Phonetic English Preset (1 day)

**Implementation**:

```rust
impl SubstitutionSet {
    /// Creates a substitution set for English phonetic similarities.
    ///
    /// Allows substitutions between characters/digraphs with
    /// similar pronunciations in English.
    pub fn phonetic_english() -> Self {
        let mut set = SubstitutionSet::new();

        // Consonant confusions
        set.add_bidirectional('c', 'k');  // cat/kat
        set.add_bidirectional('c', 's');  // city/sity
        set.add_bidirectional('s', 'z');  // hose/hoze
        set.add_bidirectional('f', 'v');  // leaf/leav
        set.add_bidirectional('g', 'j');  // giraffe/jiraffe

        // Vowel confusions
        set.add_bidirectional('a', 'e');
        set.add_bidirectional('i', 'y');

        set
    }
}
```

**Validation**:
- [ ] Phonetic test cases
- [ ] Documentation

---

#### Task 2.4: Documentation and Examples (1 day)

**Files**:
- `/examples/keyboard_spell_checker.rs` (NEW)
- `/examples/ocr_correction.rs` (NEW)
- Update `/README.md` with Universal LA section

**Example**:

```rust
// examples/keyboard_spell_checker.rs
use liblevenshtein::prelude::*;

fn main() {
    // Load dictionary
    let words = vec!["test", "text", "next", "best"];

    // Build with QWERTY restrictions
    let dict = TransducerBuilder::new()
        .with_qwerty_substitutions()
        .build_from_iter(words);

    // Queries
    let queries = [
        ("tesy", "Likely typo: y↔t adjacent"),
        ("texr", "Likely typo: r↔t adjacent"),
        ("texz", "Unlikely typo: z↔s far apart"),
    ];

    for (query, description) in queries {
        println!("\nQuery: '{}' - {}", query, description);
        let results: Vec<_> = dict.fuzzy_search(query, 1).collect();
        println!("Results: {:?}", results);
    }
}
```

**Validation**:
- [ ] Examples compile and run
- [ ] README updated
- [ ] cargo doc builds successfully

---

### Phase 3: Integration with Existing Algorithms (1 week)

**Goal**: Ensure Universal LA works with Transposition and MergeAndSplit algorithms.

**Deliverables**:
- ✅ Transposition + restricted substitutions
- ✅ MergeAndSplit + restricted substitutions
- ✅ Comprehensive test suite
- ✅ Performance benchmarks

---

#### Task 3.1: Transposition Algorithm Integration (2 days)

**File**: `/src/transducer/transition.rs` (Transposition section, lines 195-319)

**Analysis needed**:
- Transposition operation (ab → ba) should remain unrestricted
- But if transposition involves character comparison, check substitution set
- Ensure characteristic vector logic handles restricted substitutions

**Test cases**:

```rust
#[test]
fn test_transposition_with_restrictions() {
    let mut set = SubstitutionSet::new();
    set.add_bidirectional('a', 's');  // Allow a↔s substitution

    let words = vec!["fast", "last"];
    let dict = TransducerBuilder::new()
        .algorithm(Algorithm::Transposition)
        .with_substitution_set(set)
        .build_from_iter(words);

    // "fsat" → "fast" (transposition, no substitution needed)
    let results: Vec<_> = dict.fuzzy_search("fsat", 1).collect();
    assert!(results.contains(&"fast"));

    // "lxst" → should NOT find "last" (x↔a not allowed)
    let results: Vec<_> = dict.fuzzy_search("lxst", 1).collect();
    assert!(!results.contains(&"last"));
}
```

**Validation**:
- [ ] Transposition tests pass
- [ ] No regression in existing transposition tests

---

#### Task 3.2: MergeAndSplit Algorithm Integration (2 days)

**File**: `/src/transducer/transition.rs` (MergeAndSplit section, lines 327-438)

**Analysis needed**:
- Merge/split operations should remain unrestricted
- Substitution checks apply to non-merge/split transitions
- Verify characteristic vector handles edge cases

**Test cases**:

```rust
#[test]
fn test_merge_split_with_restrictions() {
    let mut set = SubstitutionSet::new();
    // Define allowed substitutions

    let words = vec![/* test words */];
    let dict = TransducerBuilder::new()
        .algorithm(Algorithm::MergeAndSplit)
        .with_substitution_set(set)
        .build_from_iter(words);

    // Test merge/split operations work
    // Test substitutions are restricted
}
```

**Validation**:
- [ ] MergeAndSplit tests pass
- [ ] No regression in existing tests

---

#### Task 3.3: Performance Benchmarks (2 days)

**File**: `/benches/universal_la.rs` (NEW)

**Benchmarks**:

1. **Baseline (no restrictions)**:
   ```rust
   let dict = TransducerBuilder::new()
       .algorithm(Algorithm::Standard)
       .build_from_iter(large_dictionary);
   ```

2. **With restrictions**:
   ```rust
   let dict = TransducerBuilder::new()
       .algorithm(Algorithm::Standard)
       .with_qwerty_substitutions()
       .build_from_iter(large_dictionary);
   ```

3. **Metrics**:
   - Query throughput (queries/sec)
   - Latency (median, p95, p99)
   - Memory usage

**Expected results**:
- 10-20% overhead acceptable
- Document actual overhead
- Identify optimization opportunities

**Validation**:
- [ ] Benchmarks run successfully
- [ ] Results documented
- [ ] Overhead within acceptable range

---

### Phase 4: Optimization (Optional, 1 week)

**Goal**: Reduce overhead of substitution checks to <10%.

**Deliverables**:
- ⚠️ Optimized SubstitutionSet lookup
- ⚠️ SIMD acceleration (if applicable)
- ⚠️ Cache-friendly data structures
- ⚠️ Subsumption refinement (if needed)

---

#### Task 4.1: Perfect Hashing for Static Sets (2 days)

**Approach**: Use `phf` crate for compile-time perfect hash functions.

**Implementation**:

```rust
// For preset substitution sets, use phf::Map
use phf::Map;

static QWERTY_MAP: Map<(char, char), ()> = phf_map! {
    ('q', 'w') => (),
    ('w', 'q') => (),
    // ... all pairs at compile time
};

impl SubstitutionSet {
    pub fn qwerty_optimized() -> Self {
        // Use QWERTY_MAP for O(1) lookup with zero collisions
    }
}
```

**Validation**:
- [ ] Performance improvement measured
- [ ] No loss of functionality

---

#### Task 4.2: Bit Vector for ASCII (1 day)

**Approach**: For ASCII-only alphabets, use 256×256 bit array.

**Implementation**:

```rust
pub struct AsciiSubstitutionSet {
    // 256×256 bits = 8 KB
    allowed: [u64; 1024],  // 256*256 / 64 = 1024
}

impl AsciiSubstitutionSet {
    #[inline]
    pub fn is_allowed(&self, a: char, b: char) -> bool {
        if a.is_ascii() && b.is_ascii() {
            let a = a as usize;
            let b = b as usize;
            let index = (a << 8) | b;
            let word = index / 64;
            let bit = index % 64;
            (self.allowed[word] & (1u64 << bit)) != 0
        } else {
            false
        }
    }
}
```

**Validation**:
- [ ] Benchmark shows improvement
- [ ] ASCII test cases pass

---

#### Task 4.3: Subsumption Logic Review (2 days)

**Concern**: Paper notes that d_L^S may not satisfy triangle inequality.

**Investigation**:
1. Review current subsumption predicates
2. Test with edge cases where triangle inequality might fail
3. Determine if subsumption needs refinement

**Possible outcomes**:
- ✅ Current subsumption works fine (likely)
- ⚠️ Need modified subsumption for restricted substitutions
- ❌ Disable subsumption for Universal LA (fallback)

**Validation**:
- [ ] Correctness tests pass
- [ ] No false positives/negatives
- [ ] Performance impact measured

---

## Testing Strategy

### Unit Tests
- [ ] SubstitutionSet operations
- [ ] Builder configuration
- [ ] Transition function logic
- [ ] Preset substitution sets

### Integration Tests
- [ ] End-to-end query with restrictions
- [ ] All three algorithms (Standard, Transposition, MergeAndSplit)
- [ ] Edge cases (empty set, unrestricted, etc.)

### Performance Tests
- [ ] Benchmark overhead
- [ ] Memory usage
- [ ] Scalability with dictionary size

### Regression Tests
- [ ] All existing tests pass
- [ ] Backward compatibility preserved
- [ ] No performance regression for unrestricted case

---

## Documentation Requirements

### API Documentation
- [ ] SubstitutionSet struct and methods
- [ ] Builder API extensions
- [ ] Preset constructors
- [ ] Usage examples

### User Guide
- [ ] Concept explanation
- [ ] When to use restricted substitutions
- [ ] How to define custom substitution sets
- [ ] Performance considerations

### Examples
- [ ] Keyboard spell checker
- [ ] OCR correction
- [ ] Phonetic matching
- [ ] Custom domain-specific rules

---

## Success Criteria

### Phase 1
- [ ] SubstitutionSet implemented and tested
- [ ] Standard algorithm respects restrictions
- [ ] At least one end-to-end test passes

### Phase 2
- [ ] Three preset substitution sets available
- [ ] Documentation and examples complete
- [ ] User feedback positive

### Phase 3
- [ ] All algorithms work with restrictions
- [ ] Performance overhead < 20%
- [ ] No regressions

### Phase 4 (Optional)
- [ ] Performance overhead < 10%
- [ ] Advanced optimizations documented

---

## Risk Mitigation

### Performance Risk
- **Mitigation**: Benchmark early and often
- **Fallback**: Document overhead, make feature opt-in

### Correctness Risk
- **Mitigation**: Comprehensive test suite
- **Fallback**: Compare with brute-force computation

### Complexity Risk
- **Mitigation**: Start simple, iterate
- **Fallback**: Reduce scope (Standard algorithm only)

---

## Timeline

### Week 1-2: Phase 1
- Days 1-2: SubstitutionSet structure
- Days 3-4: Builder and transition functions
- Days 5-7: Testing and bug fixes

### Week 3: Phase 2
- Days 1-3: Preset substitution sets (QWERTY, OCR, phonetic)
- Days 4-5: Documentation and examples
- Days 6-7: Buffer for polish

### Week 4: Phase 3
- Days 1-2: Transposition integration
- Days 3-4: MergeAndSplit integration
- Days 5-7: Performance benchmarks

### Week 5 (Optional): Phase 4
- Days 1-2: Perfect hashing
- Days 3-4: Subsumption review
- Days 5-7: Final optimization and documentation

---

## Next Steps

1. **Review** this plan with team/maintainers
2. **Approve** implementation approach (Option B recommended)
3. **Begin** Phase 1, Task 1.1 (Create SubstitutionSet structure)
4. **Track** progress using [progress-tracker.md](./progress-tracker.md)

---

**Last Updated**: 2025-11-06
**Status**: Planning Complete, Ready to Start Phase 1
