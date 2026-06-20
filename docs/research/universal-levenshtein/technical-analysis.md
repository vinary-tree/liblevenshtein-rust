[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Technical Analysis

**Date**: 2025-11-06
**Status**: Analysis Complete

---

## Overview

This document analyzes the current liblevenshtein-rust codebase to identify:
1. Current implementation of Levenshtein automata
2. Where restricted substitutions fit into the architecture
3. Gaps that need to be filled
4. Integration points for Universal LA features

---

## Current Algorithm Support

### Supported Algorithms

The library currently supports three algorithm variants defined in `/src/transducer/algorithm.rs`:

```rust
pub enum Algorithm {
    Standard,        // Basic Levenshtein (insert, delete, substitute)
    Transposition,   // Standard + adjacent character swaps
    MergeAndSplit,   // Standard + character merge/split operations
}
```

**Key characteristics**:
- All operations have **uniform cost = 1**
- No support for restricted substitutions
- No support for variable operation costs
- Clean separation of algorithms via enum

### Operation Costs

**Current state**: All operations (insert, delete, substitute, transpose, merge, split) have uniform cost = 1.

**Implication for Universal LA**:
- Universal LA maintains uniform cost = 1 for allowed operations
- Restricted substitutions are **binary**: either allowed (cost=1) or blocked (cost=∞)
- This aligns well with current architecture - no cost model changes needed

---

## Core Data Structures

### Position Structure

Defined in `/src/transducer/position.rs:11-35`:

```rust
pub struct Position {
    pub term_index: usize,      // Index into query term
    pub num_errors: usize,      // Accumulated edit distance
    pub is_special: bool,       // Flag for Transposition/MergeAndSplit
}
```

**Analysis**:
- `term_index`: Tracks position in query string
- `num_errors`: Cumulative edit operations (all cost=1)
- `is_special`: Used by Transposition and MergeAndSplit algorithms

**Impact for Universal LA**:
- **No changes needed** to Position structure itself
- Restricted substitutions are enforced at transition time, not stored in Position
- `num_errors` continues to track count of operations (allowed operations only)

### Characteristic Vector

Current implementation in `/src/transducer/position.rs:231-269`:

```rust
// Standard algorithm characteristic vector
pub(crate) fn characteristic_vector(
    term: &str,
    query_chars: &[char],
) -> Vec<u64> {
    // Creates binary vector: 1 if character matches at position, 0 otherwise
    // Used to determine valid transitions
}
```

**Key concept**: The characteristic vector χ(a, w) represents where character `a` appears in word `w`.

**Universal LA modification needed**:
- Current: χ(a, w[i]) = 1 if w[i] == a, else 0
- **New S-characteristic vector**: χ_s(a, w[i]) = 1 if (w[i], a) ∈ S (substitution allowed), else 0

**Implementation approach**:
```rust
// Enhanced characteristic vector with substitution set
pub(crate) fn s_characteristic_vector(
    term: &str,
    query_chars: &[char],
    substitution_set: Option<&SubstitutionSet>,
) -> Vec<u64> {
    match substitution_set {
        None => characteristic_vector(term, query_chars),  // Standard behavior
        Some(s) => {
            // Check (term_char, query_char) ∈ S for each position
            // Set bit to 1 if substitution is allowed
        }
    }
}
```

---

## Transition Functions

### Location

All transition logic is in `/src/transducer/transition.rs`:

- **Lines 119-188**: Standard algorithm transitions
- **Lines 195-319**: Transposition algorithm transitions
- **Lines 327-438**: MergeAndSplit algorithm transitions

### Standard Algorithm Transitions

Current logic (simplified):

```rust
// Match: same character, no error increment
if query_char == dict_char {
    next_state.insert(Position::new(i + 1, e, false));
}

// Substitution: different characters, increment error
if query_char != dict_char {
    next_state.insert(Position::new(i + 1, e + 1, false));
}

// Insertion: consume dict char, increment error
next_state.insert(Position::new(i, e + 1, false));

// Deletion: consume query char, increment error
next_state.insert(Position::new(i + 1, e + 1, false));
```

**Critical observation**: Substitution currently allows **any** character pair (query_char, dict_char) when they differ.

### Required Modification for Universal LA

**Change needed in substitution transition**:

```rust
// OLD: Unconditional substitution
if query_char != dict_char {
    next_state.insert(Position::new(i + 1, e + 1, false));
}

// NEW: Check substitution set
if query_char != dict_char {
    // Check if substitution is allowed by the set S
    let substitution_allowed = match substitution_set {
        None => true,  // No restrictions = all substitutions allowed
        Some(s) => s.is_allowed(query_char, dict_char),
    };

    if substitution_allowed {
        next_state.insert(Position::new(i + 1, e + 1, false));
    }
}
```

**Impact**:
- Must pass `substitution_set: Option<&SubstitutionSet>` through transition functions
- Add conditional check before creating substitution transitions
- No changes to insert/delete transitions (always allowed)
- Match transition unchanged (identical characters, no substitution needed)

### Transposition Algorithm

**Current behavior**: Allows swapping adjacent characters (ab → ba)

**Universal LA extension**: Paper covers restricted substitutions + transposition

**Modification needed**:
- Transposition operation itself remains unrestricted (always allowed)
- But if transposition involves position advancement with character mismatch, substitution check applies
- Likely **minimal changes** beyond Standard algorithm modifications

### MergeAndSplit Algorithm

**Current behavior**:
- Merge: Two characters → one (e.g., "rn" → "m")
- Split: One character → two (e.g., "m" → "rn")

**Universal LA extension**: Paper discusses combining with merge/split

**Modification needed**:
- Merge/split operations themselves remain unrestricted
- Substitution checks apply to non-merge/split transitions
- Similar pattern to Transposition algorithm

---

## Builder API

### Current Interface

Located in `/src/transducer/builder.rs`:

```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    // ... other configuration fields
}

impl<D> TransducerBuilder<D> {
    pub fn new() -> Self { /* ... */ }

    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    // ... other builder methods
}
```

**Usage example**:
```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .build_from_iter(words);
```

### Required Extensions for Universal LA

**Option A: New Algorithm Variant** (Not Recommended)

```rust
pub enum Algorithm {
    Standard,
    Transposition,
    MergeAndSplit,
    RestrictedSubstitution,  // NEW
}
```

**Problems**:
- Cannot combine with Transposition or MergeAndSplit
- Requires duplicate transition logic
- Less flexible

**Option B: Configuration Field** (Recommended)

```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    substitution_set: Option<SubstitutionSet>,  // NEW field
}

impl<D> TransducerBuilder<D> {
    pub fn with_substitution_set(mut self, set: SubstitutionSet) -> Self {
        self.substitution_set = Some(set);
        self
    }

    pub fn with_qwerty_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::qwerty())
    }

    pub fn with_ocr_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::ocr_confusions())
    }
}
```

**Advantages**:
- Works with **all** algorithms (Standard, Transposition, MergeAndSplit)
- Backward compatible (None = no restrictions = current behavior)
- Can easily add preset substitution sets
- Aligns with paper's approach (restrictions orthogonal to operation type)

---

## Query Processing

### Current Query Flow

Located in `/src/transducer/query.rs:86-188`:

1. Initialize starting state (Position with term_index=0, num_errors=0)
2. Traverse dictionary structure (DAWG/Trie/etc.)
3. For each dictionary node:
   - Compute next states via transition functions
   - Filter states by error threshold
   - Apply subsumption to prune redundant states
4. Collect matching dictionary terms

**Integration point for Universal LA**:
- Pass `substitution_set` from TransducerBuilder through to transition functions
- No changes to overall query flow
- Substitution checks happen inside transition functions

---

## Subsumption Logic

### Current Implementation

Located in `/src/transducer/position.rs`:

```rust
// Position A subsumes Position B if:
// - A.term_index == B.term_index
// - A.num_errors <= B.num_errors
// - A.is_special matches appropriately
```

**Purpose**: Prune redundant states to reduce search space.

**Example**: If position (i=5, e=2) exists, position (i=5, e=3) is subsumed (same progress, more errors).

### Impact of Restricted Substitutions

**Paper warning** (Section 3): The generalized distance d_L^S may **not satisfy triangle inequality** when substitutions are restricted.

**Implication**: Subsumption logic may need adjustment.

**Potential issue**:
```
Standard LA:  d(A, C) ≤ d(A, B) + d(B, C)  (triangle inequality holds)
Universal LA: May violate triangle inequality when substitution paths are blocked
```

**Mitigation strategy**:
1. **Phase 1**: Keep current subsumption logic (likely works for most cases)
2. **Phase 4** (Optimization): Investigate if subsumption needs refinement
3. **Testing**: Use comprehensive test cases to detect incorrect pruning

**Expected impact**: Likely minimal, as subsumption compares positions with same term_index (not different paths through edit graph).

---

## Architecture Gaps

### What Exists ✅

1. **Algorithm enum**: Clean separation of algorithm types
2. **Position structure**: Tracks state efficiently
3. **Transition functions**: Well-structured, easy to modify
4. **Builder API**: Flexible configuration pattern
5. **Characteristic vectors**: Foundation for Universal LA extension

### What's Missing ❌

1. **SubstitutionSet structure**: Need to represent set S
2. **S-characteristic vectors**: Modified χ_s computation
3. **Substitution validation**: Check (a, b) ∈ S in transitions
4. **Builder configuration**: Add substitution_set field
5. **Preset substitution sets**: QWERTY, OCR, phonetic, etc.
6. **Serialization support**: Save/load SubstitutionSet with dictionaries
7. **Documentation**: API docs for restricted substitutions

---

## Integration Points

### 1. SubstitutionSet Creation

**Where**: New module `/src/transducer/substitution.rs`

**What**:
```rust
pub struct SubstitutionSet {
    allowed: HashSet<(char, char)>,
}

impl SubstitutionSet {
    pub fn new() -> Self;
    pub fn unrestricted() -> Self;
    pub fn add(&mut self, a: char, b: char);
    pub fn add_bidirectional(&mut self, a: char, b: char);
    pub fn is_allowed(&self, a: char, b: char) -> bool;

    // Preset constructors
    pub fn qwerty() -> Self;
    pub fn azerty() -> Self;
    pub fn dvorak() -> Self;
    pub fn ocr_confusions() -> Self;
    pub fn phonetic_english() -> Self;
}
```

### 2. TransducerBuilder Extension

**Where**: `/src/transducer/builder.rs`

**What**:
- Add `substitution_set: Option<SubstitutionSet>` field
- Add `with_substitution_set()` method
- Add convenience methods for presets

### 3. Transition Function Modification

**Where**: `/src/transducer/transition.rs`

**What**:
- Add `substitution_set: Option<&SubstitutionSet>` parameter
- Check substitution validity before creating transition
- Pass through all three algorithm implementations

### 4. Query Iterator Update

**Where**: `/src/transducer/query.rs`

**What**:
- Pass `substitution_set` from builder to transition functions
- No logic changes, just parameter threading

### 5. Characteristic Vector Extension

**Where**: `/src/transducer/position.rs`

**What**:
- Add `s_characteristic_vector()` function
- Check (term_char, query_char) ∈ S for substitutions
- Fall back to standard χ when substitution_set is None

---

## Backward Compatibility

### Strategy

**Key principle**: Universal LA features are **opt-in** via configuration.

**Ensuring compatibility**:

1. **Default behavior unchanged**:
   ```rust
   // This still works exactly as before
   let dict = TransducerBuilder::new()
       .algorithm(Algorithm::Standard)
       .build_from_iter(words);
   ```

2. **None means unrestricted**:
   ```rust
   substitution_set: Option<SubstitutionSet>
   // None → all substitutions allowed (current behavior)
   // Some(set) → only substitutions in set allowed (new behavior)
   ```

3. **No breaking changes**:
   - Algorithm enum unchanged (Option B approach)
   - Position structure unchanged
   - Existing APIs unchanged

4. **Feature flag** (optional):
   ```toml
   [features]
   universal-la = []
   ```
   Can gate code behind feature flag if needed.

---

## Performance Considerations

### Expected Overhead

**Optimistic**: 5-10% slowdown (if substitution checks are well-optimized)

**Realistic**: 10-20% slowdown (typical case with HashSet lookup)

**Worst-case**: 30% slowdown (large substitution sets, poor cache locality)

### Overhead Sources

1. **Substitution validity check**: `𝒪(1)` HashSet lookup per potential substitution
   - Typical cost: ~5-10 nanoseconds
   - Per-query impact: Depends on dictionary size and error bound

2. **Memory overhead**: SubstitutionSet storage
   - HashSet<(char, char)>: ~24 bytes per entry + hash table overhead
   - Typical size: 50-500 pairs → 1-12 KB

3. **Cache effects**: Additional memory accesses for substitution checks
   - May impact cache hit rate
   - Mitigated by storing SubstitutionSet in hot cache lines

### Optimization Opportunities

1. **Perfect hashing**: For static substitution sets
   - Pre-compute minimal perfect hash function
   - `𝒪(1)` lookup with zero collisions
   - Trade-off: Higher setup cost, lower runtime cost

2. **Bit vectors**: For small alphabets (ASCII, DNA)
   - 256×256 bit array = 8 KB for full ASCII
   - Single bit test instead of hash lookup
   - Very cache-friendly

3. **SIMD**: Batch substitution checks
   - Check multiple character pairs in parallel
   - Requires careful data layout

4. **Caching**: Memoize recent substitution checks
   - Small LRU cache (16-32 entries)
   - High hit rate in typical queries

---

## Test Coverage Needs

### Unit Tests Required

1. **SubstitutionSet structure**:
   - Add/remove pairs
   - Bidirectional additions
   - Unrestricted set behavior
   - Preset constructors (QWERTY, OCR, etc.)

2. **Transition functions**:
   - Substitution allowed → transition created
   - Substitution blocked → no transition
   - None substitution set → all allowed

3. **End-to-end query**:
   - QWERTY keyboard proximity
   - OCR confusion sets
   - Combination with Transposition
   - Combination with MergeAndSplit

### Integration Tests Required

1. **Real dictionary queries**:
   - English dictionary + QWERTY constraints
   - DNA sequences + transition/transversion constraints
   - OCR text + visual confusion sets

2. **Performance benchmarks**:
   - Baseline (no restrictions) vs restricted substitutions
   - Different substitution set sizes
   - Different algorithms (Standard, Transposition, MergeAndSplit)

3. **Correctness validation**:
   - Compare results with brute-force distance computation
   - Verify no false positives (disallowed substitutions counted)
   - Verify no false negatives (allowed substitutions missed)

---

## Codebase Files to Modify

### Must Modify

1. **`/src/transducer/builder.rs`**
   - Add `substitution_set` field
   - Add `with_substitution_set()` method

2. **`/src/transducer/transition.rs`**
   - Add parameter to transition functions
   - Add substitution validity checks

3. **`/src/transducer/query.rs`**
   - Pass substitution_set through query pipeline

### Must Create

4. **`/src/transducer/substitution.rs`** (NEW)
   - Define SubstitutionSet structure
   - Implement preset constructors

### Should Modify

5. **`/src/transducer/position.rs`**
   - Add s_characteristic_vector function
   - Extend characteristic vector logic

### May Modify (Optional)

6. **`/src/transducer/algorithm.rs`**
   - Only if choosing Option A (new Algorithm variant)
   - Not needed for Option B (configuration approach)

---

## Dependencies

### External Crates Needed

**None for basic implementation**. Standard library is sufficient:
- `std::collections::HashSet` for SubstitutionSet storage
- Existing serialization infrastructure (serde) for persistence

### Optional Dependencies

1. **Perfect hashing** (optimization):
   - `phf` crate for compile-time perfect hash functions
   - Use for static preset substitution sets

2. **SIMD** (optimization):
   - No new dependencies (use std::simd when stabilized)
   - Manual SIMD via platform intrinsics if needed

---

## Risk Assessment

### Low Risk ✅

1. **Backward compatibility**: Option B approach ensures no breaking changes
2. **Code complexity**: Modifications are localized and well-defined
3. **Testing**: Easy to create comprehensive test suite

### Medium Risk ⚠️

1. **Performance overhead**: 10-20% slowdown acceptable for most use cases
   - Mitigation: Benchmarking and optimization (Phase 4)
   - Opt-out: Users can avoid feature if performance-critical

2. **Subsumption logic**: May need refinement for restricted substitutions
   - Mitigation: Extensive testing to detect issues
   - Fallback: Disable subsumption for Universal LA (correctness over speed)

### High Risk ❌

**None identified**. Implementation is well-understood with clear path forward.

---

## Summary

### Current State

The liblevenshtein-rust codebase is **well-positioned** for Universal LA implementation:
- Clean architecture with clear separation of concerns
- Flexible builder pattern for configuration
- Well-structured transition functions
- Existing characteristic vector foundation

### Key Changes Needed

1. **New SubstitutionSet structure** (new file)
2. **Builder API extension** (minor addition)
3. **Transition function checks** (conditional logic)
4. **Characteristic vector enhancement** (algorithm extension)

### Effort Estimate

- **Core implementation**: 1-2 weeks
- **Preset substitution sets**: 1 week
- **Testing & documentation**: 1 week
- **Optimization** (optional): 1 week

**Total**: 2-4 weeks for complete implementation.

---

**Next Steps**:
1. Review [decision-matrix.md](./decision-matrix.md) for implementation approach
2. Read [implementation-plan.md](./implementation-plan.md) for phase breakdown
3. Check [architectural-sketches.md](./architectural-sketches.md) for detailed code designs

---

**Last Updated**: 2025-11-06
**Status**: Analysis Complete, Ready for Implementation Planning
