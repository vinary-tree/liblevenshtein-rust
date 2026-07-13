[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Decision Matrix

**Date**: 2025-11-06
**Status**: Decision Made - Option B Recommended

---

## Overview

This document compares implementation approaches for adding Universal Levenshtein Automata (restricted substitutions) to liblevenshtein-rust and provides a recommendation.

---

## Implementation Options

### Option A: New Algorithm Variant

Add `RestrictedSubstitution` as a 4th variant to the `Algorithm` enum.

### Option B: Configuration-Based

Add `substitution_set` as an optional field in `TransducerBuilder`, orthogonal to the `Algorithm` choice.

---

## Detailed Comparison

### Option A: Algorithm Enum Variant

**Implementation**:

```rust
pub enum Algorithm {
    Standard,
    Transposition,
    MergeAndSplit,
    RestrictedSubstitution,  // NEW VARIANT
}
```

**Usage**:

```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::RestrictedSubstitution)
    .with_substitution_set(set)  // Configure which substitutions
    .build_from_iter(words);
```

#### Advantages ✅

1. **Clear separation**: Restricted substitutions are a distinct "algorithm" choice
2. **Explicit in type system**: Algorithm enum documents all variants
3. **Simple mental model**: "Choose an algorithm" is familiar pattern

#### Disadvantages ❌

1. **Cannot combine algorithms**:
   - Cannot do Transposition + Restricted Substitutions
   - Cannot do MergeAndSplit + Restricted Substitutions
   - Would need: `RestrictedTransposition`, `RestrictedMergeAndSplit` variants → enum explosion

2. **Code duplication**:
   - Each "restricted" variant duplicates transition logic
   - Maintenance burden: bug fixes need multiple locations

3. **Inconsistent with paper**:
   - Paper treats restricted substitutions as orthogonal to operation type
   - Substitution set S applies to ALL algorithms (Standard, Transposition, MergeAndSplit)

4. **Future inflexibility**:
   - If adding more configuration (e.g., weighted operations), enum becomes unwieldy
   - Would need combinatorial explosion of variants

5. **API confusion**:
   - What if user sets `.algorithm(RestrictedSubstitution)` but doesn't call `.with_substitution_set()`?
   - Need to handle invalid configurations

---

### Option B: Configuration Field

**Implementation**:

```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,                      // Standard, Transposition, MergeAndSplit
    substitution_set: Option<SubstitutionSet>, // NEW: None = unrestricted
}
```

**Usage**:

```rust
// Standard + restricted substitutions
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .with_substitution_set(qwerty_set)
    .build_from_iter(words);

// Transposition + restricted substitutions
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Transposition)
    .with_substitution_set(ocr_set)
    .build_from_iter(words);

// Standard + unrestricted (backward compatible)
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .build_from_iter(words);  // No substitution_set → unrestricted
```

#### Advantages ✅

1. **Composability**: Works with ALL existing algorithms
   - Standard + restricted substitutions ✅
   - Transposition + restricted substitutions ✅
   - MergeAndSplit + restricted substitutions ✅

2. **Code reuse**:
   - Single substitution check in each algorithm's transition function
   - No duplication of logic
   - Bug fixes in one place

3. **Aligns with paper**:
   - Paper presents restricted substitutions as orthogonal to operation type
   - Set S applies uniformly across all algorithms

4. **Future-proof**:
   - Easy to add more configuration options (e.g., cost weights, custom operations)
   - Builder pattern scales well

5. **Backward compatible**:
   - `None` substitution set → unrestricted (current behavior)
   - Existing code unchanged
   - Opt-in feature

6. **Clear semantics**:
   - `None` = "all substitutions allowed"
   - `Some(set)` = "only substitutions in set allowed"
   - No ambiguous states

7. **Flexible presets**:
   - Can provide convenience methods: `.with_qwerty_substitutions()`, `.with_ocr_confusions()`, etc.
   - Users can easily switch between presets

#### Disadvantages ❌

1. **Slightly more complex builder**:
   - One more field in `TransducerBuilder`
   - One more parameter to thread through query pipeline
   - **Mitigation**: Well-established pattern, minimal complexity

2. **Less explicit in type system**:
   - Substitution restrictions not visible in `Algorithm` enum
   - **Mitigation**: Clear documentation, good API naming

---

## Comparison Table

| Criterion | Option A (Enum Variant) | Option B (Configuration) |
|-----------|------------------------|-------------------------|
| **Composability** | ❌ Cannot combine with Transposition/MergeAndSplit | ✅ Works with all algorithms |
| **Code duplication** | ❌ Duplicates transition logic per variant | ✅ Single implementation, reused |
| **Alignment with paper** | ⚠️ Treats as separate algorithm | ✅ Orthogonal to operation type |
| **Backward compatibility** | ⚠️ New variant, but existing code OK | ✅ Optional field, perfect compat |
| **Future extensibility** | ❌ Enum explosion for combinations | ✅ Builder pattern scales well |
| **API clarity** | ✅ Explicit in Algorithm enum | ⚠️ Requires reading builder docs |
| **Implementation complexity** | 🔴 High (duplicate code paths) | 🟢 Low (add checks to existing) |
| **Type safety** | ✅ Compiler enforces algorithm choice | ⚠️ Optional field (but semantics clear) |
| **Preset convenience** | ⚠️ Can provide, but per-algorithm | ✅ Easy builder methods for presets |
| **Configuration errors** | ❌ Need to handle missing SubstitutionSet | ✅ None = valid default |

---

## Use Case Analysis

### Use Case 1: QWERTY Spell Checker (Standard Algorithm)

**Option A**:
```rust
.algorithm(Algorithm::RestrictedSubstitution)
.with_substitution_set(SubstitutionSet::qwerty())
```

**Option B**:
```rust
.algorithm(Algorithm::Standard)
.with_substitution_set(SubstitutionSet::qwerty())
// OR
.algorithm(Algorithm::Standard)
.with_qwerty_substitutions()
```

**Winner**: Option B (clearer that we're using Standard + restrictions)

---

### Use Case 2: OCR Correction with Transposition

**Option A**:
```rust
// CANNOT DO THIS - would need new enum variant
// Algorithm::RestrictedTransposition doesn't exist
// Would require adding it, duplicating all transposition logic
```

**Option B**:
```rust
.algorithm(Algorithm::Transposition)
.with_substitution_set(SubstitutionSet::ocr_confusions())
```

**Winner**: Option B (only option that supports this)

---

### Use Case 3: Phonetic Matching with Merge/Split

**Option A**:
```rust
// CANNOT DO THIS - would need Algorithm::RestrictedMergeAndSplit
// More enum variants, more duplication
```

**Option B**:
```rust
.algorithm(Algorithm::MergeAndSplit)
.with_substitution_set(SubstitutionSet::phonetic_english())
```

**Winner**: Option B (only option that supports this)

---

### Use Case 4: Unrestricted Matching (Current Behavior)

**Option A**:
```rust
.algorithm(Algorithm::Standard)  // Existing variant
```

**Option B**:
```rust
.algorithm(Algorithm::Standard)  // No .with_substitution_set() call
```

**Winner**: Tie (both maintain backward compatibility)

---

## Paper Alignment

### Paper's Approach

From **Section 2** of "Universal Levenshtein Automata for a Generalization of the Levenshtein Distance":

> "We consider a generalization of the Levenshtein distance where substitutions are restricted to pairs in a set **`$S \subseteq \Sigma \times \Sigma$`**."

Key observation: The set **S** is a **parameter to the distance function**, not a fundamentally different algorithm.

> "The construction of the universal Levenshtein automaton can be extended to handle this restriction **in combination with** transposition, merge, and split operations."

**Interpretation**: Restricted substitutions are **orthogonal** to the choice of operations (transposition, merge, split).

### Alignment Score

- **Option A**: ⚠️ Treats restricted substitutions as a separate algorithm type (misalignment)
- **Option B**: ✅ Treats restricted substitutions as a parameter/configuration (aligns with paper)

---

## Implementation Effort

### Option A: New Algorithm Variant

**Effort**:
1. Add `Algorithm::RestrictedSubstitution` variant
2. Duplicate transition logic from `Standard` algorithm
3. Add substitution checks in duplicated code
4. Later: Add `Algorithm::RestrictedTransposition` (duplicate again)
5. Later: Add `Algorithm::RestrictedMergeAndSplit` (duplicate again)

**Total**: 🔴 High effort, with ongoing maintenance burden

---

### Option B: Configuration Field

**Effort**:
1. Add `substitution_set: Option<SubstitutionSet>` field to builder
2. Add `with_substitution_set()` method
3. Thread parameter through query pipeline
4. Add one `if` check in each algorithm's transition function
5. Done - works with all algorithms

**Total**: 🟢 Low effort, single implementation

---

## Maintenance Burden

### Option A

**Scenario**: Bug found in transposition logic

**Fix required**:
1. Fix `Algorithm::Transposition`
2. Fix `Algorithm::RestrictedTransposition` (duplicate code path)
3. Verify both fixes are consistent

**Burden**: 🔴 High - every bug needs multiple fixes

---

### Option B

**Scenario**: Bug found in transposition logic

**Fix required**:
1. Fix `Algorithm::Transposition` transition function
2. Done - substitution check is orthogonal

**Burden**: 🟢 Low - single fix location

---

## Extensibility

### Future: Weighted Operations

**Option A**: Would need:
```rust
pub enum Algorithm {
    Standard,
    Transposition,
    MergeAndSplit,
    RestrictedSubstitution,
    WeightedStandard,              // NEW
    WeightedTransposition,         // NEW
    WeightedMergeAndSplit,         // NEW
    RestrictedWeightedStandard,    // NEW (combination!)
    RestrictedWeightedTransposition,  // Enum explosion!
    // 🔥 This is unsustainable
}
```

**Option B**: Would add:
```rust
pub struct TransducerBuilder<D> {
    algorithm: Algorithm,                      // Still just 3 variants
    substitution_set: Option<SubstitutionSet>, // Existing
    operation_weights: Option<OperationWeights>,  // NEW field
}
```

**Winner**: Option B (scales to multiple orthogonal features)

---

## Community Expectations

### Rust Ecosystem Patterns

**Builder pattern best practices**:
- Configuration via methods, not enum proliferation
- Composability over rigid types
- Optional fields for optional features

**Examples**:
```rust
// reqwest (HTTP client)
let client = Client::builder()
    .timeout(Duration::from_secs(10))  // Optional config
    .build();

// tokio (async runtime)
let rt = Builder::new_multi_thread()
    .worker_threads(4)          // Optional config
    .enable_all()               // Feature flag
    .build();
```

**Pattern**: Option B aligns with Rust ecosystem conventions.

---

## Recommendation

### ✅ Option B: Configuration-Based (Recommended)

**Rationale**:

1. **Aligns with paper**: Treats restricted substitutions as parameter, not algorithm type
2. **Composable**: Works with all existing algorithms (Standard, Transposition, MergeAndSplit)
3. **Maintainable**: Single implementation, no code duplication
4. **Future-proof**: Scales to additional configuration options
5. **Backward compatible**: Existing code works unchanged
6. **Rust idiomatic**: Follows builder pattern best practices

**Trade-offs accepted**:
- Slightly less explicit in type system (mitigated by good documentation)
- One more field to manage (minimal complexity cost)

---

## Implementation Checklist

### Core Changes (Option B)

- [ ] Add `substitution_set: Option<SubstitutionSet>` to `TransducerBuilder`
- [ ] Add `.with_substitution_set(set)` method
- [ ] Thread parameter through query pipeline
- [ ] Add substitution check in `standard_transition()`
- [ ] Add substitution check in `transposition_transition()`
- [ ] Add substitution check in `merge_split_transition()`

### Convenience Methods

- [ ] `.with_qwerty_substitutions()`
- [ ] `.with_azerty_substitutions()`
- [ ] `.with_dvorak_substitutions()`
- [ ] `.with_ocr_confusions()`
- [ ] `.with_phonetic_english()`

### Documentation

- [ ] Builder API docs explaining substitution_set
- [ ] Examples for each preset
- [ ] Migration guide (if needed - likely not)

---

## Alternative Considered: Hybrid Approach

**Idea**: Start with Option B, later add Option A if needed.

**Analysis**:
- ✅ Doesn't preclude future enum variants
- ⚠️ Likely never needed (Option B is sufficient)
- ❌ Would create two ways to do same thing (confusing)

**Decision**: Not recommended - stick with Option B exclusively.

---

## Conclusion

**Decision**: Implement **Option B (Configuration-Based)**

**Next Steps**:
1. Proceed with implementation plan in [implementation-plan.md](./implementation-plan.md)
2. Start with Phase 1, Task 1.1: Create SubstitutionSet structure
3. Track progress in [progress-tracker.md](./progress-tracker.md)

---

**Last Updated**: 2025-11-06
**Status**: Decision Made - Option B Selected
**Implementation**: Ready to Begin
