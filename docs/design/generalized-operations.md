# Generalized Operation Type System Design

[← Documentation Index](../README.md)

**Date**: 2025-11-12
**Status**: 📋 **DESIGN PHASE** - Not yet implemented
**Applies To**: Both lazy and universal Levenshtein automata
**Estimated Effort**: 2-3 weeks implementation + 1 week testing

---

## Executive Summary

This document proposes a generalized operation type system to replace the current hardcoded `Algorithm` enum with a flexible, data-driven framework based on the TCS 2011 paper (Section 3). This will enable:

- **Custom operation types** beyond standard Levenshtein (Unicode normalization, case folding, etc.)
- **Weighted operations** with different costs (confidence-scored OCR corrections)
- **Restricted substitutions** (keyboard proximity, phonetic matching)
- **Backward compatibility** with existing Standard/Transposition/MergeAndSplit algorithms

**Key Benefits**:
- Extensibility without modifying core automaton code
- Data-driven operation definitions (no recompilation needed)
- Supports domain-specific edit distances (NLP, bioinformatics, spell checking)
- Maintains performance through static dispatch where possible

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Proposed API Design](#3-proposed-api-design)
4. [Migration Strategy](#4-migration-strategy)
5. [Performance Considerations](#5-performance-considerations)
6. [Use Case Examples](#6-use-case-examples)
7. [Implementation Phases](#7-implementation-phases)
8. [Testing Strategy](#8-testing-strategy)
9. [Backward Compatibility](#9-backward-compatibility)
10. [Future Extensions](#10-future-extensions)

---

## 1. Problem Statement

### Current Limitations

**File**: `src/transducer/algorithm.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    Standard,        // Hardcoded: match, insert, delete, substitute
    Transposition,   // Hardcoded: + transpose
    MergeAndSplit,   // Hardcoded: + merge, split
}
```

**Problems**:

1. **Not Extensible**: Adding new operation types requires:
   - Modifying the enum
   - Updating match arms in transition logic
   - Recompiling the library

2. **No Custom Costs**: All operations have weight 1.0 (except match = 0.0)
   - Cannot model confidence scores
   - Cannot prioritize certain operations

3. **No Restricted Substitutions**: Cannot limit which character pairs can be substituted
   - Cannot model keyboard proximity (q↔w but not q↔z)
   - Cannot model phonetic similarity (f↔ph)

4. **Hardcoded Logic**: Operation definitions embedded in code, not data
   - Cannot load custom operations from config
   - Cannot experiment with new operations without recompilation

### Real-World Use Cases Blocked

#### Use Case 1: OCR Correction with Confidence Scores
```
Problem: OCR engine confuses 'O' with '0', 'I' with '1', 'l' with 'I'
Desired: Weight substitutions by confusion probability
- O↔0: weight 0.2 (common confusion, low cost)
- I↔1: weight 0.3
- arbitrary: weight 1.0 (standard cost)
```

**Blocked**: No way to specify per-operation weights.

#### Use Case 2: Unicode Normalization
```
Problem: Query "café" should match "cafe" in dictionary
Desired: Custom operation: é → e (decompose diacritic) with weight 0.1
```

**Blocked**: No way to define custom operations.

#### Use Case 3: Keyboard Proximity Spell Checking
```
Problem: QWERTY typos "nane" → "name" (m/n adjacent) more likely than "pane" → "name"
Desired: Weight substitutions by keyboard distance
- Adjacent keys: weight 0.3
- Nearby keys: weight 0.6
- Distant keys: weight 1.0
```

**Blocked**: No per-pair substitution weights.

#### Use Case 4: Phonetic Matching
```
Problem: "ph" sounds like "f", "c" sounds like "k" or "s"
Desired: ph↔f with weight 0.2, c↔k/c↔s with weight 0.3
```

**Blocked**: No multi-character operations with custom weights.

---

## 2. Theoretical Foundation

![Operation sets: how the Standard, Transposition, and Merge-and-Split presets compose edit operations from operation triples (characters consumed from the first word, characters consumed from the second word, and a weight), and how restricted substitution sets layer onto them.](../diagrams/automata/operation-sets.svg)

### From TCS 2011 Paper (Section 3, Pages 2341-2342)

**Operation Type**: A triple `$t = \langle t^x, t^y, t^w\rangle$` where:
- `t^x`: Number of characters consumed from first word
- `t^y`: Number of characters consumed from second word
- `t^w`: Operation weight/cost

**Standard Levenshtein Operations**:
```
Match:         ⟨1, 1, 0⟩  (consume both, no cost)
Substitution:  ⟨1, 1, 1⟩  (consume both, cost 1)
Insertion:     ⟨0, 1, 1⟩  (consume second only, cost 1)
Deletion:      ⟨1, 0, 1⟩  (consume first only, cost 1)
Transposition: ⟨2, 2, 1⟩  (consume 2 from each, cost 1)
```

**Restricted Operations**: `$op = \langle op^x, op^y, op^r, op^w\rangle$` where:
- `$op^r \subseteq \Sigma^{op^x} \times \Sigma^{op^y}$`: Allowed character pair replacements (`$\Sigma$` denotes the alphabet)

**Examples**:
```
Keyboard QWERTY proximity:
  op^r = {(q,w), (q,a), (w,e), (w,q), (w,s), ...}

OCR confusion sets:
  op^r = {(O,0), (0,O), (I,1), (1,I), (l,I), ...}

Phonetic similarity:
  op^r = {(f,ph), (ph,f), (c,k), (c,s), ...}
```

### Bounded Diagonal Property (Theorem 8.2)

**Constraint**: For universal automata to exist:
- All **zero-weighted** operations must be **length-preserving**
- i.e., if `t^w = 0`, then `t^x = t^y`

**Implication**: Match operation must consume same characters from both words (`$\langle 1,1,0\rangle$`).

### Operation Set Requirements

An operation set `Op` must satisfy:
1. **Contains match**: `$\langle 1, 1, 0\rangle \in Op$` (required for bounded diagonal)
2. **Finite**: `$\lvert Op\rvert < \infty$`
3. **Bounded consumption**: `$\max(t^x, t^y) \le k$` for some constant `$k$`

---

## 3. Proposed API Design

### Core Types

**File**: `src/transducer/operation.rs` (new)

```rust
/// Operation type from TCS 2011 paper
///
/// Represents a single edit operation as a triple ⟨t^x, t^y, t^w⟩
#[derive(Debug, Clone, PartialEq)]
pub struct OperationType {
    /// Characters consumed from first word (t^x)
    pub x_consumed: u8,

    /// Characters consumed from second word (t^y)
    pub y_consumed: u8,

    /// Operation weight/cost (t^w)
    ///
    /// - 0.0: No cost (match operation)
    /// - 1.0: Standard cost
    /// - Other: Custom weights
    pub weight: f32,

    /// Optional: Restricted character pairs (op^r)
    ///
    /// If None: operation applies to all character pairs
    /// If Some: operation only applies to pairs in the set
    pub restriction: Option<SubstitutionSet>,

    /// Human-readable name for debugging
    pub name: &'static str,
}

impl OperationType {
    /// Create a new operation type
    pub fn new(x_consumed: u8, y_consumed: u8, weight: f32, name: &'static str) -> Self {
        Self {
            x_consumed,
            y_consumed,
            weight,
            restriction: None,
            name,
        }
    }

    /// Create operation with restricted character pairs
    pub fn with_restriction(
        x_consumed: u8,
        y_consumed: u8,
        weight: f32,
        restriction: SubstitutionSet,
        name: &'static str,
    ) -> Self {
        Self {
            x_consumed,
            y_consumed,
            weight,
            restriction: Some(restriction),
            name,
        }
    }

    /// Check if operation is a match (zero-weighted)
    pub fn is_match(&self) -> bool {
        self.weight == 0.0
    }

    /// Check if operation is length-preserving
    pub fn is_length_preserving(&self) -> bool {
        self.x_consumed == self.y_consumed
    }

    /// Check if operation applies to character pair (a, b)
    pub fn applies_to(&self, a: char, b: char) -> bool {
        match &self.restriction {
            None => true,  // Unrestricted
            Some(set) => set.allows(a, b),
        }
    }
}
```

### Operation Set

```rust
/// Set of operation types defining custom edit distance
#[derive(Debug, Clone)]
pub struct OperationSet {
    /// All operation types
    operations: Vec<OperationType>,

    /// Cached: index of match operation (for fast lookup)
    match_index: usize,

    /// Cached: maximum character consumption
    max_consumption: u8,
}

impl OperationSet {
    /// Create operation set from types
    ///
    /// # Panics
    ///
    /// Panics if no match operation (⟨1, 1, 0⟩) is present
    pub fn new(operations: Vec<OperationType>) -> Self {
        // Validate: must contain match operation
        let match_index = operations
            .iter()
            .position(|op| op.is_match() && op.x_consumed == 1 && op.y_consumed == 1)
            .expect("Operation set must contain match operation ⟨1,1,0⟩");

        // Compute max consumption
        let max_consumption = operations
            .iter()
            .map(|op| op.x_consumed.max(op.y_consumed))
            .max()
            .unwrap_or(1);

        Self {
            operations,
            match_index,
            max_consumption,
        }
    }

    /// Get all operations
    pub fn operations(&self) -> &[OperationType] {
        &self.operations
    }

    /// Get match operation
    pub fn match_operation(&self) -> &OperationType {
        &self.operations[self.match_index]
    }

    /// Get maximum character consumption
    pub fn max_consumption(&self) -> u8 {
        self.max_consumption
    }

    /// Find operations applicable to character pair
    pub fn applicable_operations(&self, a: char, b: char) -> impl Iterator<Item = &OperationType> {
        self.operations.iter().filter(move |op| op.applies_to(a, b))
    }
}
```

### Preset Operation Sets

```rust
impl OperationSet {
    /// Standard Levenshtein operations
    pub fn standard() -> Self {
        Self::new(vec![
            OperationType::new(1, 1, 0.0, "match"),
            OperationType::new(1, 1, 1.0, "substitute"),
            OperationType::new(0, 1, 1.0, "insert"),
            OperationType::new(1, 0, 1.0, "delete"),
        ])
    }

    /// Levenshtein with transposition
    pub fn transposition() -> Self {
        Self::new(vec![
            OperationType::new(1, 1, 0.0, "match"),
            OperationType::new(1, 1, 1.0, "substitute"),
            OperationType::new(0, 1, 1.0, "insert"),
            OperationType::new(1, 0, 1.0, "delete"),
            OperationType::new(2, 2, 1.0, "transpose"),
        ])
    }

    /// Keyboard proximity substitutions (QWERTY)
    pub fn keyboard_qwerty() -> Self {
        let proximity = SubstitutionSet::keyboard_qwerty();
        Self::new(vec![
            OperationType::new(1, 1, 0.0, "match"),
            OperationType::with_restriction(1, 1, 0.3, proximity.clone(), "keyboard_subst"),
            OperationType::new(1, 1, 1.0, "substitute"),  // Fallback
            OperationType::new(0, 1, 1.0, "insert"),
            OperationType::new(1, 0, 1.0, "delete"),
        ])
    }

    /// OCR confusion-weighted substitutions
    pub fn ocr_friendly() -> Self {
        let ocr_confusion = SubstitutionSet::ocr_friendly();
        Self::new(vec![
            OperationType::new(1, 1, 0.0, "match"),
            OperationType::with_restriction(1, 1, 0.2, ocr_confusion, "ocr_subst"),
            OperationType::new(1, 1, 1.0, "substitute"),  // Fallback
            OperationType::new(0, 1, 1.0, "insert"),
            OperationType::new(1, 0, 1.0, "delete"),
        ])
    }

    /// Phonetic similarity substitutions
    pub fn phonetic() -> Self {
        let phonetic = SubstitutionSet::phonetic_basic();
        Self::new(vec![
            OperationType::new(1, 1, 0.0, "match"),
            OperationType::with_restriction(1, 1, 0.3, phonetic, "phonetic_subst"),
            OperationType::new(1, 1, 1.0, "substitute"),  // Fallback
            OperationType::new(0, 1, 1.0, "insert"),
            OperationType::new(1, 0, 1.0, "delete"),
        ])
    }
}
```

### Builder Pattern

```rust
/// Builder for custom operation sets
pub struct OperationSetBuilder {
    operations: Vec<OperationType>,
}

impl OperationSetBuilder {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add match operation (required)
    pub fn with_match(mut self) -> Self {
        self.operations.push(OperationType::new(1, 1, 0.0, "match"));
        self
    }

    /// Add standard operations (insert, delete, substitute)
    pub fn with_standard_ops(mut self) -> Self {
        self.operations.push(OperationType::new(1, 1, 1.0, "substitute"));
        self.operations.push(OperationType::new(0, 1, 1.0, "insert"));
        self.operations.push(OperationType::new(1, 0, 1.0, "delete"));
        self
    }

    /// Add custom operation
    pub fn with_operation(mut self, op: OperationType) -> Self {
        self.operations.push(op);
        self
    }

    /// Add weighted operation
    pub fn with_weighted_op(
        mut self,
        x: u8,
        y: u8,
        weight: f32,
        name: &'static str,
    ) -> Self {
        self.operations.push(OperationType::new(x, y, weight, name));
        self
    }

    /// Add restricted substitution
    pub fn with_restricted_subst(
        mut self,
        restriction: SubstitutionSet,
        weight: f32,
        name: &'static str,
    ) -> Self {
        self.operations.push(OperationType::with_restriction(
            1, 1, weight, restriction, name,
        ));
        self
    }

    /// Build operation set
    pub fn build(self) -> OperationSet {
        OperationSet::new(self.operations)
    }
}

// Usage example:
let custom_ops = OperationSetBuilder::new()
    .with_match()
    .with_standard_ops()
    .with_weighted_op(2, 2, 0.8, "transpose")
    .with_restricted_subst(SubstitutionSet::keyboard_qwerty(), 0.3, "keyboard")
    .build();
```

---

## 4. Migration Strategy

### Phase 1: Introduce New Types Alongside Old

**Goal**: Add new types without breaking existing code

**Changes**:
1. Create `src/transducer/operation.rs` with new types
2. Keep existing `Algorithm` enum
3. Add conversion: `impl From<Algorithm> for OperationSet`

```rust
impl From<Algorithm> for OperationSet {
    fn from(alg: Algorithm) -> Self {
        match alg {
            Algorithm::Standard => OperationSet::standard(),
            Algorithm::Transposition => OperationSet::transposition(),
            Algorithm::MergeAndSplit => OperationSet::merge_and_split(),
        }
    }
}
```

**Status**: No breaking changes

### Phase 2: Update Internal APIs

**Goal**: Make automaton internals use `OperationSet`

**Changes**:
1. Update lazy transition logic to use `OperationSet`
2. Update universal successor generation to use `OperationSet`
3. Keep public APIs accepting `Algorithm` (convert internally)

```rust
// Internal API
impl State {
    fn transition_internal(&self, ops: &OperationSet, ...) -> State {
        // New implementation using ops.operations()
    }

    // Public API (backward compatible)
    pub fn transition(&self, algorithm: Algorithm, ...) -> State {
        let ops = OperationSet::from(algorithm);
        self.transition_internal(&ops, ...)
    }
}
```

**Status**: Breaking changes to internal APIs only

### Phase 3: Add New Public APIs

**Goal**: Expose `OperationSet` in public APIs

**Changes**:
1. Add new methods accepting `OperationSet`
2. Deprecate old methods accepting `Algorithm`
3. Update examples and documentation

```rust
impl State {
    /// New API: custom operation sets
    pub fn transition_with_ops(&self, ops: &OperationSet, ...) -> State {
        self.transition_internal(ops, ...)
    }

    /// Old API: deprecated but still works
    #[deprecated(since = "0.7.0", note = "Use transition_with_ops instead")]
    pub fn transition(&self, algorithm: Algorithm, ...) -> State {
        let ops = OperationSet::from(algorithm);
        self.transition_with_ops(&ops, ...)
    }
}
```

**Status**: Deprecation warnings, backward compatible

### Phase 4: Remove Deprecated APIs (Future)

**Goal**: Clean up codebase

**Changes**:
1. Remove `Algorithm` enum (breaking change)
2. Remove deprecated methods
3. Require `OperationSet` everywhere

**Status**: Major version bump (0.x.0 → 1.0.0)

---

## 5. Performance Considerations

### Static vs Dynamic Dispatch

**Problem**: Operation set is now data, not types
- Current: `match algorithm { Standard => ..., Transposition => ... }` (static dispatch)
- Proposed: Iterate over `ops.operations()` (dynamic dispatch)

**Impact**: Potential performance regression

#### Mitigation 1: Inline Common Path

```rust
// Fast path for standard operations
if ops.is_standard() {
    // Hardcoded standard logic (inlined)
    return standard_transition(...);
}

// Slow path for custom operations
return generic_transition(ops, ...);
```

#### Mitigation 2: Cache Applicability

```rust
/// Precompute which operations apply to which character pairs
struct OperationCache {
    /// For each char pair (a, b), list of applicable operation indices
    cache: HashMap<(char, char), SmallVec<[usize; 4]>>,
}

impl OperationCache {
    fn applicable_ops(&self, a: char, b: char, ops: &OperationSet) -> &[usize] {
        self.cache.get(&(a, b))
            .map(|v| v.as_slice())
            .unwrap_or_else(|| {
                // Fallback: iterate operations
                ops.operations().iter().enumerate()
                    .filter(|(_, op)| op.applies_to(a, b))
                    .map(|(i, _)| i)
                    .collect()
            })
    }
}
```

#### Mitigation 3: Trait Objects with Specialization

```rust
trait OperationSetImpl {
    fn transition(&self, state: &State, ...) -> State;
}

// Specialized implementation for standard ops
struct StandardOps;
impl OperationSetImpl for StandardOps {
    fn transition(&self, state: &State, ...) -> State {
        // Inlined, optimized standard logic
    }
}

// Generic implementation for custom ops
struct GenericOps {
    ops: Vec<OperationType>,
}
impl OperationSetImpl for GenericOps {
    fn transition(&self, state: &State, ...) -> State {
        // Generic logic iterating operations
    }
}

pub struct OperationSet {
    impl_: Box<dyn OperationSetImpl>,
}
```

### Benchmarking Plan

**Test Scenarios**:
1. Standard Levenshtein (baseline)
2. Standard via `OperationSet` (measure overhead)
3. Keyboard proximity (custom weighted)
4. OCR confusion (restricted + weighted)

**Acceptance Criteria**:
- Standard via `OperationSet`: < 5% overhead
- Custom operations: < 20% overhead vs hardcoded equivalent

---

## 6. Use Case Examples

### Example 1: Unicode Normalization

```rust
// Define é → e decomposition with low cost
let unicode_norm = OperationSetBuilder::new()
    .with_match()
    .with_standard_ops()
    .with_restricted_subst(
        SubstitutionSet::from_pairs(&[
            ('é', 'e'), ('è', 'e'), ('ê', 'e'), ('ë', 'e'),
            ('à', 'a'), ('â', 'a'), ('ä', 'a'),
            // ... more diacritics
        ]),
        0.1,  // Low cost for diacritic removal
        "unicode_norm"
    )
    .build();

// Query: "café" matches "cafe" with distance 0.1
let automaton = build_automaton("café", 2, unicode_norm);
assert!(automaton.accepts("cafe"));
assert_eq!(automaton.distance("cafe"), 0.1);
```

### Example 2: OCR Correction

```rust
// Weight substitutions by OCR confusion probability
let ocr_ops = OperationSetBuilder::new()
    .with_match()
    .with_restricted_subst(
        SubstitutionSet::from_pairs(&[
            ('O', '0'), ('0', 'O'),  // Common confusion
            ('I', '1'), ('1', 'I'),
            ('l', 'I'), ('I', 'l'),
        ]),
        0.2,  // Low cost (high confidence)
        "ocr_common"
    )
    .with_restricted_subst(
        SubstitutionSet::from_pairs(&[
            ('S', '5'), ('5', 'S'),  // Less common
            ('B', '8'), ('8', 'B'),
        ]),
        0.5,  // Medium cost
        "ocr_rare"
    )
    .with_standard_ops()  // Fallback
    .build();

// "H3LL0" → "HELLO" with distance 0.2 + 0.2 = 0.4
```

### Example 3: Phonetic Matching

```rust
// Model phonetic similarity
let phonetic = OperationSetBuilder::new()
    .with_match()
    .with_restricted_subst(
        SubstitutionSet::from_pairs(&[
            ('f', "ph"),  // f ↔ ph
            ('c', 'k'),   // hard c ↔ k
            ('c', 's'),   // soft c ↔ s
        ]),
        0.3,
        "phonetic"
    )
    .with_standard_ops()
    .build();

// "telefone" matches "telephone" (f ↔ ph)
```

### Example 4: Case-Insensitive with Penalty

```rust
// Allow case changes but with small cost
let case_insensitive = OperationSetBuilder::new()
    .with_match()
    .with_weighted_op(1, 1, 0.05, "case_change")  // Low cost for case
    .with_standard_ops()
    .build();

// "Hello" matches "HELLO" with distance 0.25 (5 case changes × 0.05)
```

### Example 5: English Phonetic Corrections

**Complete analysis**: See [English Phonetic Feasibility Analysis](../research/phonetic-corrections/ENGLISH_PHONETIC_FEASIBILITY.md)

```rust
// Model phonetic spelling corrections based on English pronunciation rules
let phonetic_english = OperationSetBuilder::new()
    .with_match()

    // Consonant digraphs: ph→f, ch→ç, sh→$, th→+
    .with_weighted_op_restricted(
        2, 1, 0.15,
        SubstitutionSet::from_pairs(&[
            ("ph", "f"), ("ch", "ç"), ("sh", "$"), ("th", "+"),
            ("qu", "kw"), ("wr", "r"), ("wh", "w"),
        ]),
        "consonant_digraphs"
    )

    // Vowel digraphs: ea→ë, oa→ö, etc.
    .with_weighted_op_restricted(
        2, 1, 0.15,
        SubstitutionSet::from_pairs(&[
            ("ea", "ë"), ("ee", "ë"), ("oa", "ö"),
            ("ai", "ä"), ("ou", "ôw"), ("oi", "öy"),
        ]),
        "vowel_digraphs"
    )

    // Silent e deletion
    .with_weighted_op_restricted(
        1, 0, 0.1,
        SubstitutionSet::from_chars(&['e']),
        "silent_e"
    )

    // Complex GH patterns: igh→ï (right→rït)
    .with_weighted_op_restricted(
        3, 1, 0.2,
        SubstitutionSet::from_pairs(&[
            ("igh", "ï"), ("aught", "òt"), ("ough", "ö"),
        ]),
        "gh_patterns"
    )

    .with_standard_ops()
    .build();

// Examples:
let automaton = build_automaton("telephone", 2, phonetic_english);
assert!(automaton.accepts("tel@fön"));  // Phonetic spelling
assert!(automaton.accepts("telefone")); // Common misspelling

let automaton = build_automaton("daughter", 3, phonetic_english);
assert!(automaton.accepts("dòt@r"));    // Phonetic transformation

// Use cases:
// - Spell checking with phonetic suggestions
// - "Sounds like" search queries
// - OCR post-processing (errors often preserve pronunciation)
// - Cross-lingual name matching (transliterations)
```

**Coverage**: 60-85% of English phonetic transformations achievable with this framework

**Implementation Guide**: See [Implementation Guide](../research/phonetic-corrections/IMPLEMENTATION_GUIDE.md)

**Required Extensions for Full Coverage**:
- **Larger operations** (d=3,4): For patterns like "aught"→"òt" (4→2)
- **Position-aware operations**: For word-initial/final rules (kn→n only at start)
- **Bi-directional context** (lazy only): For c→s/k based on following vowel

**Limitations**:
- ❌ Cannot model retroactive modifications (gh lengthening previous vowel)
- ❌ Cannot detect syllable boundaries (unbounded lookahead)
- ❌ Cannot distinguish morphological context (suffix -able vs word "table")

**Performance**:
- Memory: 8-50 MB (depending on operation set size)
- Speed: 3-10× faster than dynamic programming for dictionary search
- Coverage: 75-85% of common English words

---

## 7. Implementation Phases

### Phase 1: Core Types (Week 1)

**Files to Create**:
- `src/transducer/operation.rs` - Core types
- `src/transducer/operation/builder.rs` - Builder pattern
- `src/transducer/operation/presets.rs` - Preset operation sets

**Tests**:
- Unit tests for `OperationType`
- Unit tests for `OperationSet`
- Builder pattern tests
- Preset correctness tests

**Deliverables**:
- ✅ Core types compile
- ✅ All unit tests pass
- ✅ Documentation complete

### Phase 2: Lazy Integration (Week 2)

**Files to Modify**:
- `src/transducer/lazy.rs` - Update transition logic
- `src/transducer/state.rs` - Update state operations

**Changes**:
1. Add `transition_with_ops()` method
2. Refactor existing transition logic to use operations
3. Keep `Algorithm` enum, add conversion

**Tests**:
- Regression tests (all existing tests still pass)
- New tests with custom operations
- Benchmark: measure overhead

**Deliverables**:
- ✅ Lazy automata work with `OperationSet`
- ✅ No performance regression for standard ops
- ✅ Custom operations tested

### Phase 3: Universal Integration (Week 2-3)

**Files to Modify**:
- `src/transducer/universal/position.rs` - Successor generation
- `src/transducer/universal/state.rs` - State transitions

**Changes**:
1. Update `successors()` to use `OperationSet`
2. Handle weighted operations in distance calculation
3. Test with restricted substitutions

**Tests**:
- Universal automaton with custom ops
- Word-pair distance with weights
- Characteristic vector with restrictions

**Deliverables**:
- ✅ Universal automata work with `OperationSet`
- ✅ Weighted operations compute correct distances
- ✅ Integration tests pass

### Phase 4: Public API & Documentation (Week 3)

**Changes**:
1. Add public constructors accepting `OperationSet`
2. Deprecate old `Algorithm`-based constructors
3. Update examples
4. Write migration guide

**Documentation**:
- API documentation for all new types
- Migration guide (Algorithm → OperationSet)
- Use case examples
- Performance comparison benchmarks

**Deliverables**:
- ✅ Public APIs finalized
- ✅ Documentation complete
- ✅ Examples updated
- ✅ Migration guide published

---

## 8. Testing Strategy

### Unit Tests

**File**: `tests/operation_tests.rs`

```rust
#[test]
fn test_operation_type_match() {
    let op = OperationType::new(1, 1, 0.0, "match");
    assert!(op.is_match());
    assert!(op.is_length_preserving());
}

#[test]
fn test_operation_set_standard() {
    let ops = OperationSet::standard();
    assert_eq!(ops.operations().len(), 4);
    assert_eq!(ops.match_operation().weight, 0.0);
}

#[test]
fn test_restricted_substitution() {
    let subst_set = SubstitutionSet::from_pairs(&[('a', 'b'), ('b', 'a')]);
    let op = OperationType::with_restriction(1, 1, 0.5, subst_set, "test");
    assert!(op.applies_to('a', 'b'));
    assert!(op.applies_to('b', 'a'));
    assert!(!op.applies_to('a', 'c'));
}
```

### Integration Tests

**File**: `tests/generalized_operations_integration.rs`

```rust
#[test]
fn test_lazy_automaton_with_custom_ops() {
    let ops = OperationSetBuilder::new()
        .with_match()
        .with_standard_ops()
        .with_weighted_op(2, 2, 0.8, "transpose")
        .build();

    let automaton = build_lazy_automaton("kitten", 2, ops);
    assert!(automaton.accepts("sitting"));
}

#[test]
fn test_weighted_distance_calculation() {
    let ops = OperationSet::keyboard_qwerty();  // Adjacent keys cost 0.3
    let automaton = build_lazy_automaton("name", 2, ops);

    // "nane" → "name" (m/n adjacent) should have distance 0.3
    assert_eq!(automaton.distance("nane"), 0.3);
}
```

### Property-Based Tests

**Using proptest**:

```rust
proptest! {
    #[test]
    fn test_custom_ops_satisfy_distance_properties(
        word1 in "[a-z]{3,10}",
        word2 in "[a-z]{3,10}",
    ) {
        let ops = OperationSet::standard();

        // Distance should be symmetric
        let d1 = compute_distance(&word1, &word2, &ops);
        let d2 = compute_distance(&word2, &word1, &ops);
        assert_eq!(d1, d2);

        // Distance to self should be 0
        let d_self = compute_distance(&word1, &word1, &ops);
        assert_eq!(d_self, 0.0);
    }
}
```

### Performance Benchmarks

**File**: `benches/operation_overhead_bench.rs`

```rust
fn bench_standard_hardcoded(c: &mut Criterion) {
    c.bench_function("standard/hardcoded", |b| {
        b.iter(|| {
            let automaton = build_lazy_automaton("kitten", 2, Algorithm::Standard);
            automaton.distance("sitting")
        })
    });
}

fn bench_standard_via_operation_set(c: &mut Criterion) {
    c.bench_function("standard/operation_set", |b| {
        let ops = OperationSet::standard();
        b.iter(|| {
            let automaton = build_lazy_automaton_ops("kitten", 2, &ops);
            automaton.distance("sitting")
        })
    });
}

fn bench_keyboard_weighted(c: &mut Criterion) {
    c.bench_function("keyboard/weighted", |b| {
        let ops = OperationSet::keyboard_qwerty();
        b.iter(|| {
            let automaton = build_lazy_automaton_ops("hello", 2, &ops);
            automaton.distance("hekko")  // e→k (adjacent keys)
        })
    });
}

criterion_group!(benches, bench_standard_hardcoded, bench_standard_via_operation_set, bench_keyboard_weighted);
criterion_main!(benches);
```

---

## 9. Backward Compatibility

### Compatibility Matrix

| Version | `Algorithm` enum | `OperationSet` | Breaking Changes |
|---------|------------------|----------------|------------------|
| 0.6.x (current) | ✅ Available | ❌ Not available | - |
| 0.7.0 (Phase 1-3) | ✅ Available | ✅ Available | None (additive) |
| 0.8.0 (Phase 4) | ⚠️ Deprecated | ✅ Primary API | Deprecation warnings |
| 1.0.0 (Future) | ❌ Removed | ✅ Only API | `Algorithm` removed |

### Migration Path for Users

#### Step 1: Update to 0.7.0 (No Code Changes)

Existing code continues to work:

```rust
// This still works (no changes needed)
let automaton = build_automaton("test", 2, Algorithm::Standard);
```

#### Step 2: Try New API (Optional)

Users can experiment with new features:

```rust
// New: custom operation sets
let ops = OperationSet::keyboard_qwerty();
let automaton = build_automaton_ops("test", 2, ops);
```

#### Step 3: Update to 0.8.0 (Deprecation Warnings)

Old API deprecated but functional:

```rust
// Warning: use `build_automaton_ops` instead
let automaton = build_automaton("test", 2, Algorithm::Standard);
```

Fix warnings by migrating:

```rust
// New way (no warnings)
let ops = OperationSet::standard();
let automaton = build_automaton_ops("test", 2, ops);
```

#### Step 4: Update to 1.0.0 (Breaking)

Must use new API:

```rust
// Only way
let ops = OperationSet::standard();
let automaton = build_automaton_ops("test", 2, ops);
```

### Automated Migration Tool

Provide script to convert code:

```bash
# Converts Algorithm::Standard → OperationSet::standard()
$ cargo run --bin migrate_operations src/
Migrating src/main.rs...
  - Line 42: Algorithm::Standard → OperationSet::standard()
  - Line 57: Algorithm::Transposition → OperationSet::transposition()
Done! 2 conversions.
```

---

## 10. Future Extensions

### Extension 1: Serializable Operation Sets

**Goal**: Load custom operations from config files

```toml
# operations.toml
[[operation]]
name = "match"
x_consumed = 1
y_consumed = 1
weight = 0.0

[[operation]]
name = "keyboard_subst"
x_consumed = 1
y_consumed = 1
weight = 0.3
restriction = "keyboard_qwerty"

[[operation]]
name = "substitute"
x_consumed = 1
y_consumed = 1
weight = 1.0
```

Implementation:

```rust
impl OperationSet {
    pub fn from_toml_file(path: &Path) -> Result<Self, Error> {
        // Deserialize TOML → Vec<OperationType> → OperationSet
    }

    pub fn from_json(json: &str) -> Result<Self, Error> {
        // Deserialize JSON → Vec<OperationType> → OperationSet
    }
}
```

### Extension 2: Context-Aware Operations

**Goal**: Operations that depend on surrounding context

```rust
pub struct ContextualOperationType {
    op: OperationType,
    /// Precondition: operation only applies if context matches
    context: Box<dyn Fn(&str, usize) -> bool>,
}

// Example: Allow 'c'→'s' only before 'i' or 'e' (soft c)
let soft_c = ContextualOperationType {
    op: OperationType::new(1, 1, 0.5, "soft_c"),
    context: Box::new(|word, pos| {
        word.chars().nth(pos + 1)
            .map(|c| c == 'i' || c == 'e')
            .unwrap_or(false)
    }),
};
```

### Extension 3: Machine Learning Weights

**Goal**: Learn operation weights from training data

```rust
pub struct LearnedOperationSet {
    base_ops: OperationSet,
    weight_model: Box<dyn Fn(char, char) -> f32>,
}

impl LearnedOperationSet {
    /// Train weights from word pairs with target distances
    pub fn train(
        pairs: &[(String, String, f32)],
        base_ops: OperationSet,
    ) -> Self {
        // Use gradient descent to learn weights
        // that minimize error on training pairs
    }

    /// Adjust weights for specific domain
    pub fn fine_tune(&mut self, pairs: &[(String, String, f32)]) {
        // Incremental learning
    }
}
```

### Extension 4: Multi-Character Operations

**Goal**: Support operations consuming >2 characters

Current limitation: `$t^x, t^y \le 2$` (hardcoded in successor logic, where `$k = 2$`)

Extension:

```rust
pub struct MultiCharOperation {
    x_consumed: u8,  // No limit
    y_consumed: u8,  // No limit
    weight: f32,
    /// Pattern in first word
    x_pattern: Regex,
    /// Pattern in second word
    y_pattern: Regex,
}

// Example: "ough" → "off" (English spelling)
let spelling_reform = MultiCharOperation {
    x_consumed: 4,
    y_consumed: 3,
    weight: 0.5,
    x_pattern: Regex::new("ough").unwrap(),
    y_pattern: Regex::new("off").unwrap(),
};
```

---

## Summary

This design provides a **flexible, extensible operation type system** that:

✅ Replaces hardcoded `Algorithm` enum with data-driven `OperationSet`
✅ Supports weighted operations and restricted substitutions
✅ Maintains backward compatibility through gradual migration
✅ Applies to both lazy and universal automata
✅ Provides clear implementation roadmap (3 weeks)
✅ Includes comprehensive testing strategy
✅ Plans for future extensions (serialization, ML weights, context-aware)

**Next Steps**:
1. Review and approve this design
2. Create feature branch: `feature/generalized-operations`
3. Implement Phase 1 (core types)
4. Benchmark Phase 2 (lazy integration)
5. Complete Phase 3-4 (universal + public API)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Author**: Claude Code (Anthropic AI Assistant)
**Status**: 📋 **READY FOR REVIEW** - Design complete, awaiting implementation approval
