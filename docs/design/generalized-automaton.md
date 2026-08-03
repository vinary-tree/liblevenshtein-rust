# GeneralizedAutomaton Design Document

[← Documentation Index](../README.md)

**Date**: 2025-11-12
**Status**: Historical initial design; the current implementation contract is
the [generalized-automaton repair](generalized-automaton-repair.md)

> This page preserves the original phased design context. It is not the
> current source of truth for acceptance, weighted costs, performance, or
> public APIs. The shipped engine uses an exact scaled sparse alignment graph;
> its older universal-position types remain only as a compatibility surface.

---

## Executive Summary

This document describes the design and implementation plan for `GeneralizedAutomaton`, a new automaton type that supports **runtime-configurable operations** via `OperationSet`. This enables phonetic corrections, custom edit distance metrics, and other advanced string matching features while preserving the performance of the existing `UniversalAutomaton`.

![Automaton implementations: `UniversalAutomaton` selects edit operations at compile time via `PositionVariant`, whereas `GeneralizedAutomaton` selects them at runtime via an `OperationSet`.](../diagrams/automata/automaton-implementations.svg)

---

## Motivation

### Current Limitations

The existing `UniversalAutomaton` uses compile-time specialization via `PositionVariant`:

```rust
pub struct UniversalAutomaton<V: PositionVariant, P: SubstitutionPolicy> {
    max_distance: u8,
    policy: P,  // Unused
    _phantom: PhantomData<V>,
}
```

**Problems:**
1. Operations are **hardcoded** at compile-time (Standard, Transposition, MergeAndSplit)
2. Cannot use custom operations (phonetic corrections, weighted edits, etc.)
3. `PositionVariant` and `OperationSet` create **semantic conflict** if mixed

### Why a New Type?

**Semantic Clarity:**
```rust
// Clear: Fixed operations, compile-time optimized
let fast = UniversalAutomaton::<Standard>::new(2);

// Clear: Custom operations, runtime flexibility
let custom = GeneralizedAutomaton::new(2, phonetic_english_basic());
```

**Performance Preservation:**
- `UniversalAutomaton` - Zero overhead (unchanged)
- `GeneralizedAutomaton` - Pays for runtime flexibility only when needed

**Academic Alignment:**
- Matches TCS 2011 terminology ("generalized operations")
- Clean conceptual separation

---

## Design Goals

### Primary Goals

1. **Runtime Operation Configuration**: Accept `OperationSet` parameter
2. **Single-Character Operations First**: Support $`\langle 1,1,w\rangle`$ operations initially
3. **Backward Compatibility**: Existing code unchanged, zero regressions
4. **Clear Semantics**: No confusion about compile vs runtime operations
5. **Testability**: Independent test suite, isolated from universal tests

### Non-Goals (Phase 1)

- Multi-character operations ($`\langle m,n,w\rangle`$ where $`m>1`$ or $`n>1`$)
- Weighted operations (fractional costs)
- Context-dependent operations
- Performance optimization (focus on correctness first)

---

## Architecture

### Type Hierarchy

```rust
// Existing: Compile-time operations via PositionVariant
pub struct UniversalAutomaton<V: PositionVariant, P: SubstitutionPolicy = Unrestricted> {
    max_distance: u8,
    policy: P,
    _phantom: PhantomData<V>,
}

// NEW: Runtime operations via OperationSet
pub struct GeneralizedAutomaton<P: SubstitutionPolicy = Unrestricted> {
    max_distance: u8,
    policy: P,
    operation_set: OperationSet,
}
```

**Key Difference**: No `PositionVariant` generic parameter - operations come from `operation_set` field.

### Module Structure

```
src/transducer/
├── universal/           # Existing - UNCHANGED
│   ├── mod.rs
│   ├── automaton.rs     # UniversalAutomaton<V, P>
│   ├── position.rs      # UniversalPosition<V>
│   ├── state.rs         # UniversalState<V>
│   ├── subsumption.rs
│   ├── bit_vector.rs
│   └── diagonal.rs
│
├── generalized/         # NEW
│   ├── mod.rs
│   ├── automaton.rs     # GeneralizedAutomaton<P>
│   ├── position.rs      # GeneralizedPosition (no variant)
│   ├── state.rs         # GeneralizedState
│   └── bit_vector.rs    # CharacteristicVector (same as universal)
│
├── operation_set.rs     # Already exists
├── operation_type.rs    # Already exists
└── phonetic.rs          # Already exists
```

---

## Implementation Plan

### Phase 1: Structure and Positions (Week 1)

**Goal**: Create basic GeneralizedAutomaton structure and position types.

**Tasks:**
1. Create `src/transducer/generalized/` directory
2. Copy `universal/position.rs` → `generalized/position.rs`
3. Remove `PositionVariant` generic parameter from `GeneralizedPosition`
4. Update position invariants (same as Universal)
5. Add basic tests for position creation

**Deliverable**: `GeneralizedPosition` that compiles and passes basic tests.

### Phase 2: State and Transitions (Week 1-2)

**Goal**: Implement state management and transition logic with OperationSet.

**Tasks:**
1. Copy `universal/state.rs` → `generalized/state.rs`
2. Remove `PositionVariant` from `GeneralizedState`
3. Modify `transition()` to accept `OperationSet` parameter
4. Update `successors()` to iterate over operations:
   ```rust
   pub fn successors(
       &self,
       bit_vector: &CharacteristicVector,
       max_distance: u8,
       operation_set: &OperationSet,  // NEW
   ) -> Vec<Self> {
       let mut successors = Vec::new();

       // Filter to single-character operations only (Phase 1)
       for op in operation_set.operations().iter()
           .filter(|op| op.consume_x() == 1 && op.consume_y() == 1)
       {
           // Generate successor based on operation
           match (op.consume_x(), op.consume_y()) {
               (1, 1) if op.weight() == 0.0 => {
                   // Match operation
                   if bit_vector.is_match(...) {
                       successors.push(...);
                   }
               }
               (1, 1) => {
                   // Substitution
                   if op.can_apply(...) {
                       successors.push(...);
                   }
               }
               (0, 1) => {
                   // Insertion
                   successors.push(...);
               }
               (1, 0) => {
                   // Deletion
                   successors.push(...);
               }
               _ => {}  // Skip multi-char ops in Phase 1
           }
       }

       successors
   }
   ```
5. Add state transition tests

**Deliverable**: `GeneralizedState` with working transitions.

### Phase 3: Automaton Integration (Week 2)

**Goal**: Implement GeneralizedAutomaton with accepts() method.

**Tasks:**
1. Create `src/transducer/generalized/automaton.rs`
2. Implement `GeneralizedAutomaton` structure:
   ```rust
   pub struct GeneralizedAutomaton<P: SubstitutionPolicy = Unrestricted> {
       max_distance: u8,
       policy: P,
       operation_set: OperationSet,
   }

   impl<P: SubstitutionPolicy> GeneralizedAutomaton<P> {
       pub fn new(max_distance: u8, operation_set: OperationSet) -> Self {
           Self {
               max_distance,
               policy: P::default(),
               operation_set,
           }
       }

       pub fn accepts(&self, word: &str, query: &str) -> bool {
           // Similar to UniversalAutomaton::accepts()
           // but passes operation_set to state.transition()
       }
   }
   ```
3. Copy characteristic vector logic from `universal/bit_vector.rs`
4. Add integration tests
5. Create module exports in `generalized/mod.rs`

**Deliverable**: Working `GeneralizedAutomaton` that accepts strings.

### Phase 4: Testing and Documentation (Week 2-3)

**Goal**: Comprehensive tests and documentation.

**Tasks:**
1. Add unit tests for all components
2. Add integration tests for phonetic operations
3. Add examples in doc comments
4. Create user-facing documentation
5. Update main README with GeneralizedAutomaton example

**Deliverable**: Fully tested and documented GeneralizedAutomaton.

---

## API Design

### Construction

```rust
use liblevenshtein::transducer::GeneralizedAutomaton;
use liblevenshtein::transducer::OperationSet;

// With standard operations
let ops = OperationSet::standard();
let automaton = GeneralizedAutomaton::new(2, ops);

// With phonetic operations
let ops = phonetic_english_basic();
let automaton = GeneralizedAutomaton::new(2, ops);

// With custom operations
let ops = OperationSetBuilder::new()
    .with_match()
    .with_substitution()
    .with_insertion()
    .with_deletion()
    .build();
let automaton = GeneralizedAutomaton::new(2, ops);
```

### Usage

```rust
// Check if query matches word within distance
if automaton.accepts("phone", "fone") {
    println!("Match!");
}

// Note: Phase 1 only supports accepts()
// Future: distance(), matches(), etc.
```

---

## Key Differences from UniversalAutomaton

| Aspect | UniversalAutomaton | GeneralizedAutomaton |
|--------|-------------------|---------------------|
| Operations | Compile-time (PositionVariant) | Runtime (OperationSet) |
| Performance | Zero-cost abstraction | Small runtime overhead |
| Flexibility | 3 fixed variants | Unlimited custom ops |
| Multi-char ops | Not supported | Phase 2 (future) |
| Weighted ops | Not supported | Phase 3 (future) |
| API complexity | Simple (variant only) | Medium (need OperationSet) |

---

## Testing Strategy

### Unit Tests

1. **Position Tests** (`generalized/position.rs`):
   - Position creation with valid parameters
   - Invariant enforcement
   - Successor generation

2. **State Tests** (`generalized/state.rs`):
   - State creation
   - Transition logic with various OperationSets
   - Subsumption

3. **Automaton Tests** (`generalized/automaton.rs`):
   - Construction
   - Basic accepts() functionality
   - Edge cases (empty strings, max distance

, etc.)

### Integration Tests

```rust
#[test]
fn test_standard_operations() {
    let ops = OperationSet::standard();
    let automaton = GeneralizedAutomaton::new(2, ops);

    assert!(automaton.accepts("test", "test"));  // exact match
    assert!(automaton.accepts("test", "tst"));   // deletion
    assert!(automaton.accepts("test", "teest")); // insertion
    assert!(automaton.accepts("test", "tast"));  // substitution
}

#[test]
fn test_phonetic_operations() {
    let ops = phonetic_english_basic();
    let automaton = GeneralizedAutomaton::new(1, ops);

    // Phonetic substitutions
    assert!(automaton.accepts("phone", "fone"));   // ph→f
    assert!(automaton.accepts("knight", "nite"));  // kn→n, gh→(silent)
    assert!(automaton.accepts("write", "rite"));   // wr→r
}

#[test]
fn test_custom_operations() {
    let ops = OperationSetBuilder::new()
        .with_match()
        .with_deletion()
        .build();
    let automaton = GeneralizedAutomaton::new(1, ops);

    // Only match and deletion allowed
    assert!(automaton.accepts("test", "test"));  // match
    assert!(automaton.accepts("test", "tst"));   // deletion
    assert!(!automaton.accepts("test", "teest")); // insertion not allowed
}
```

---

## Performance Considerations

### Expected Performance Characteristics

**Compared to UniversalAutomaton:**
- **Memory**: +24 bytes (OperationSet = Vec<OperationType>)
- **Construction**: +negligible (copy OperationSet)
- **Transition**: +10-20% overhead (dynamic dispatch over operations)

**For Typical Use Cases:**
- Standard ops: ~15% slower than Universal (acceptable tradeoff)
- Phonetic ops: Enables new functionality (no Universal equivalent)
- Small operation sets ($`\le 10`$ ops): Minimal overhead
- Large operation sets ($`>20`$ ops): May see more overhead

### Optimization Opportunities (Future)

1. **Operation Indexing**: Pre-index operations by `(consume_x, consume_y)` signature
2. **Static Operation Sets**: Special-case common operation sets
3. **SIMD**: Vectorize operation matching for large sets
4. **Caching**: Cache successors for common positions

---

## Future Extensions

### Phase 2: Multi-Character Operations

**Goal**: Support $`\langle m,n,w\rangle`$ where $`m>1`$ or $`n>1`$ (transposition, merge, split, phonetic digraphs).

**Required Changes:**
1. Extend `CharacteristicVector` to match patterns
2. Update position arithmetic for variable-length consumption
3. Add pattern matching to `successors()`
4. Update subsumption checks

**Estimated**: 2-3 weeks

### Phase 3: Weighted Operations

**Goal**: Support fractional operation costs (e.g., 0.15 for phonetic substitutions).

**Required Changes:**
1. Change `errors: u8` → `errors: f64` in positions
2. Update invariant checks for floating-point
3. Adjust subsumption relation for fractional differences
4. Add epsilon tolerance for comparisons

**Estimated**: 1-2 weeks

### Phase 4: Shared Core (Optional)

**Goal**: Extract common code between Universal and Generalized to eliminate duplication.

**Approach**: Create `OperationProvider` trait with compile-time and runtime implementations.

**Estimated**: 2-3 weeks

---

## Migration Guide

### For Existing Users

**No changes required** - `UniversalAutomaton` remains unchanged:

```rust
use liblevenshtein::transducer::universal::{UniversalAutomaton, Standard};

let automaton = UniversalAutomaton::<Standard>::new(2);
assert!(automaton.accepts("test", "tset"));
```

### For New Users (Phonetic Operations)

```rust
use liblevenshtein::transducer::GeneralizedAutomaton;
use liblevenshtein::transducer::phonetic::phonetic_english_basic;

let ops = phonetic_english_basic();
let automaton = GeneralizedAutomaton::new(2, ops);

// "fone" matches "phone" via ph→f phonetic rule
assert!(automaton.accepts("phone", "fone"));
```

### Choosing Between Universal and Generalized

**Use UniversalAutomaton when:**
- You need standard/transposition/merge-split operations only
- Performance is critical (every nanosecond counts)
- You prefer compile-time guarantees

**Use GeneralizedAutomaton when:**
- You need phonetic corrections
- You want custom operation sets
- You need weighted operations (future)
- Flexibility > absolute performance

---

## Open Questions

1. **Subsumption Relation**: Does the Standard subsumption relation work for weighted operations? Need mathematical verification.

2. **Diagonal Crossing**: The Universal implementation has diagonal crossing disabled (bug). Should Generalized include it from the start?

3. **Error Representation**: Should we support fractional errors in Phase 1, or defer to Phase 3?

4. **API Surface**: Should we expose `GeneralizedPosition` and `GeneralizedState` publicly, or keep them internal?

---

## References

- [TCS 2011 Paper Analysis](../research/universal-levenshtein/TCS_2011_PAPER_ANALYSIS.md)
- [Generalized Operations Design](generalized-operations.md)
- [Phonetic Operations Implementation](../research/phonetic-corrections/IMPLEMENTATION_STATUS.md)
- Universal Automata Analysis (to be created)

---

## Appendix: Code Structure

### File Sizes (Estimated)

| File | Lines | Description |
|------|-------|-------------|
| `generalized/mod.rs` | ~50 | Module exports |
| `generalized/automaton.rs` | ~300 | GeneralizedAutomaton impl |
| `generalized/position.rs` | ~400 | Position logic without variant |
| `generalized/state.rs` | ~500 | State transitions with OperationSet |
| `generalized/bit_vector.rs` | ~200 | Characteristic vectors (copy from universal) |
| **Total** | **~1450** | |

### Dependencies

```rust
// generalized/mod.rs
pub use self::automaton::GeneralizedAutomaton;
pub use self::position::GeneralizedPosition;
pub use self::state::GeneralizedState;

mod automaton;
mod position;
mod state;
mod bit_vector;

// Reuse from universal
use super::universal::{subsumption, diagonal};
```

---

**Last Updated**: 2025-11-12
**Next Review**: After Phase 1 completion
