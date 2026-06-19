# Phase 2d Implementation Plan: Multi-Character Operations

**Date**: 2025-11-13
**Status**: PLANNING
**Prerequisites**: Universal Phase 2 (Transposition) ✅, Universal Phase 3 (Merge/Split) ✅

## Executive Summary

This document provides a comprehensive implementation plan for adding multi-character operation support (transposition, merge, split) to the GeneralizedAutomaton. The implementation uses position state variants to track multi-step operations without requiring API changes, ensuring 100% backward compatibility.

**Estimated Effort**: 15-22 hours (2-3 days)
**Risk Level**: Low-Medium
**Breaking Changes**: None

---

## Table of Contents

1. [Background](#1-background)
2. [Current State Assessment](#2-current-state-assessment)
3. [Architecture Decision](#3-architecture-decision)
4. [Offset Calculation Formulas](#4-offset-calculation-formulas)
5. [State Machine Design](#5-state-machine-design)
6. [Implementation Phases](#6-implementation-phases)
7. [Testing Strategy](#7-testing-strategy)
8. [Risk Assessment](#8-risk-assessment)
9. [Success Criteria](#9-success-criteria)
10. [References](#10-references)

---

## 1. Background

### 1.1 Context

Phase 2d was previously paused (see `phase2d_analysis.md`) until Universal automaton transposition and merge/split implementations were complete. Those implementations are now complete:

- **Universal Phase 2 (Transposition)**: Complete - 168/168 tests passing
- **Universal Phase 3 (Merge/Split)**: Complete - 181/181 tests passing

The Universal implementations provide:
- Validated offset calculation formulas
- Proven state machine designs
- Cross-validation against lazy automaton (Mitankin's thesis)
- Comprehensive test suites

### 1.2 Goals

Implement multi-character operations for GeneralizedAutomaton:
- **Transposition** ⟨2,2,1⟩: Adjacent character swap
- **Merge** ⟨2,1,1⟩: Two input chars → one word char
- **Split** ⟨1,2,1⟩: One input char → two word chars

### 1.3 Requirements

**Functional**:
- Support all three multi-character operations
- Maintain backward compatibility (100%)
- Cross-validate against Universal automaton
- Pass comprehensive test suite

**Non-Functional**:
- No breaking API changes
- Performance within 20% of Universal automaton
- Clean, maintainable code
- Comprehensive documentation

---

## 2. Current State Assessment

### 2.1 What Works Well

**Strengths**:
- ✅ OperationSet infrastructure for runtime-configurable operations
- ✅ `OperationType::can_apply()` for operation validation
- ✅ SmallVec optimization in place
- ✅ Subsumption logic proven correct
- ✅ CharacteristicVector abstraction for bit matching
- ✅ Comprehensive test framework

**Phase 2c Complete**:
- Single-character operations working (match, substitute, insert, delete)
- Runtime operation iteration via OperationSet
- Dynamic operation weights
- Multiple operations of same type

### 2.2 What Needs Implementation

**Critical Gaps**:
1. **No state variants** for tracking multi-step operations
2. **Multi-character operations skipped** in successor generation:
   - `state.rs:235-237` (I-type successors)
   - `state.rs:346-348` (M-type successors)
3. **Subsumption logic** needs extension for new variants

**Code Locations**:
- `position.rs:105-130` - GeneralizedPosition enum (needs 4 new variants)
- `state.rs:215-321` - I-type successor generation
- `state.rs:323-377` - M-type successor generation
- `subsumption.rs` - Subsumption logic

### 2.3 Infrastructure Ready

**Available Components**:
- `OperationSet::with_transposition()` - creates operation set with transposition
- `OperationSet::with_merge_split()` - creates operation set with merge/split
- `OperationType::new(2, 2, 1.0, "transpose")` - transposition operation
- `OperationType::new(2, 1, 1.0, "merge")` - merge operation
- `OperationType::new(1, 2, 1.0, "split")` - split operation

---

## 3. Architecture Decision

### 3.1 Options Considered

#### Option A: Position Variants (RECOMMENDED)

**Design**: Add state-tracking variants to GeneralizedPosition enum

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GeneralizedPosition {
    // Existing
    INonFinal { offset: i32, errors: u8 },
    MFinal { offset: i32, errors: u8 },

    // NEW: Multi-step operation states
    ITransposing { offset: i32, errors: u8 },
    MTransposing { offset: i32, errors: u8 },
    ISplitting { offset: i32, errors: u8 },
    MSplitting { offset: i32, errors: u8 },
}
```

**Pros**:
- ✅ No breaking API changes
- ✅ Matches proven Universal automaton design
- ✅ State self-contained in positions
- ✅ Can validate against Universal implementation
- ✅ Clean separation of concerns

**Cons**:
- More position variants to handle
- Subsumption logic more complex
- Limited to known multi-char operations

**Verdict**: **SELECTED** - Best balance of compatibility and functionality

#### Option B: Thread Word/Input Through API

**Design**: Add word/input parameters to transition()

```rust
// BREAKING CHANGE
pub fn transition(
    &self,
    operations: &OperationSet,
    bit_vector: &CharacteristicVector,
    word: &str,           // NEW
    input: &str,          // NEW
    input_position: usize, // NEW
) -> Option<Self>
```

**Pros**:
- Generalizes to any multi-character operation
- Direct character access for validation

**Cons**:
- ❌ **BREAKING API CHANGE** - affects all call sites
- ❌ Still needs state variants for two-step operations
- ❌ Violates CharacteristicVector abstraction

**Verdict**: **REJECTED** - Breaking changes not justified

#### Option C: Context Struct

**Design**: Bundle context into optional struct

```rust
pub struct TransitionContext<'a> {
    word: &'a str,
    input: &'a str,
    input_position: usize,
}

pub fn transition(
    &self,
    operations: &OperationSet,
    bit_vector: &CharacteristicVector,
    context: Option<&TransitionContext>,  // NEW
) -> Option<Self>
```

**Pros**:
- Less parameter pollution than Option B
- Extensible

**Cons**:
- ❌ Still a breaking change
- ❌ Still needs state variants
- Extra overhead

**Verdict**: **REJECTED** - Doesn't eliminate need for variants

### 3.2 Chosen Approach

**Option A: Position Variants**

**Rationale**:
1. No breaking changes → maximum compatibility
2. Proven design from Universal automaton → lower risk
3. State self-contained → cleaner abstraction
4. Can cross-validate → higher confidence
5. Subsumption complexity manageable → acceptable trade-off

**Trade-offs Accepted**:
- Limited to transposition, merge, split (but these are the only multi-char operations in literature)
- More variants to maintain (but finite set)

---

## 4. Offset Calculation Formulas

### 4.1 Universal Automaton Offset Model

**Key Insight**: Position `I+offset#e` at input position `k` represents word position `i = offset + k`.

This relative offset model differs from lazy automaton's absolute positions, requiring offset adjustments during state transitions.

### 4.2 Transposition ⟨2,2,1⟩

#### Enter Transposition

```
Lazy:      i#e → i#(e+1)_t
Universal: I+offset#e → I+(offset-1)#(e+1)_t

At input k, position I+offset#e:
- Current word position: i = offset + k
- Target at next input k+1: still at i (same word position)
- Calculation: offset' + (k+1) = offset + k
- Result: offset' = offset - 1
```

**Implementation**:
```rust
// Enter transposition: stay at same word position
let new_offset = offset - 1;
let new_errors = errors + 1;
let new_position = GeneralizedPosition::new_i_transposing(
    new_offset,
    new_errors,
    max_distance
)?;
```

**Bit Vector Check**: `bit_vector[offset + n + 1]` - check next position for swap setup

#### Complete Transposition

```
Lazy:      i#(e+1)_t → (i+2)#e
Universal: I+offset#(e+1)_t → I+(offset+1)#e

At input k, position I+offset#(e+1)_t:
- Current word position: i = offset + k
- Target at next input k+1: i+2 (jump 2 word positions)
- Calculation: offset' + (k+1) = (offset + k) + 2
- Result: offset' = offset + 1
```

**Implementation**:
```rust
// Complete transposition: jump 2 word positions
let new_offset = offset + 1;
let new_errors = errors - 1;  // Decrement (entered with +1)
let new_position = GeneralizedPosition::new_i(
    new_offset,
    new_errors,
    max_distance
)?;
```

**Bit Vector Check**: `bit_vector[offset + n]` - verify swap at current position

**Validation**: Cross-validated with Universal (`position.rs:219-264`) and lazy (`transition.rs:287,347`)

### 4.3 Merge ⟨2,1,1⟩

#### Direct Operation (Single Step)

```
Lazy:      i#e → (i+2)#(e+1)
Universal: I+offset#e → I+(offset+1)#(e+1)

At input k, position I+offset#e:
- Current word position: i = offset + k
- Consume 2 input chars: k → k+1 (transition), then k+1 → k+2 (implicit)
- Match 1 word char at i
- Target at next input k+1: need to be at i+2
- Calculation: offset' + (k+1) = (offset + k) + 2
- Result: offset' = offset + 1
```

**Implementation**:
```rust
// Merge: consume 2 input chars, match 1 word char
let next_match_index = (offset + max_distance as i32 + 1) as usize;
if next_match_index < bit_vector.len()
    && bit_vector.is_match(next_match_index)
    && errors < max_distance
{
    let new_offset = offset + 1;
    let new_errors = errors + 1;
    if let Ok(succ) = GeneralizedPosition::new_i(
        new_offset,
        new_errors,
        max_distance
    ) {
        successors.push(succ);
    }
}
```

**Bit Vector Check**: `bit_vector[offset + n + 1]` - check next position for merge availability

**Validation**: Cross-validated with Universal (`position.rs:416`) and lazy (`transition.rs:420,454`)

### 4.4 Split ⟨1,2,1⟩

#### Enter Split

```
Lazy:      i#e → i#(e+1)_s
Universal: I+offset#e → I+(offset-1)#(e+1)_s

At input k, position I+offset#e:
- Current word position: i = offset + k
- Target at next input k+1: still at i (same word position)
- Calculation: offset' + (k+1) = offset + k
- Result: offset' = offset - 1
```

**Implementation**:
```rust
// Enter split: stay at same word position
let match_index = (offset + max_distance as i32) as usize;
if match_index < bit_vector.len()
    && bit_vector.is_match(match_index)
    && errors < max_distance
{
    let new_offset = offset - 1;
    let new_errors = errors + 1;
    if let Ok(split) = GeneralizedPosition::new_i_splitting(
        new_offset,
        new_errors,
        max_distance
    ) {
        successors.push(split);
    }
}
```

**Bit Vector Check**: `bit_vector[offset + n]` - check current position for first word char

#### Complete Split

```
Lazy:      i#(e+1)_s → (i+1)#e
Universal: I+offset#(e+1)_s → I+offset#e

At input k, position I+offset#(e+1)_s:
- Current word position: i = offset + k
- Target at next input k+1: i+1 (advance 1 word position)
- Calculation: offset' + (k+1) = (offset + k) + 1
- Result: offset' = offset + 0  (stays same!)
```

**Implementation**:
```rust
// Complete split: advance 1 word position
let match_index = (offset + max_distance as i32) as usize;
if match_index < bit_vector.len()
    && bit_vector.is_match(match_index)
{
    let new_offset = offset;  // +0
    let new_errors = errors - 1;  // Decrement
    if let Ok(succ) = GeneralizedPosition::new_i(
        new_offset,
        new_errors,
        max_distance
    ) {
        successors.push(succ);
    }
}
```

**Bit Vector Check**: `bit_vector[offset + n]` - check current position for second word char

**Validation**: Cross-validated with Universal (`position.rs:436,394`) and lazy (`transition.rs:415,459`)

### 4.5 Offset Summary Table

| Operation | Step | Offset Delta | Error Delta | State Change | Bit Vector Check |
|-----------|------|--------------|-------------|--------------|------------------|
| Match ⟨1,1,0⟩ | Direct | +0 | +0 | - | `[offset + n]` |
| Substitute ⟨1,1,1⟩ | Direct | +0 | +1 | - | Any |
| Insert ⟨0,1,1⟩ | Direct | +0 | +1 | - | None |
| Delete ⟨1,0,1⟩ | Direct | -1 | +1 | - | None |
| Transpose | Enter | -1 | +1 | → Transposing | `[offset + n + 1]` |
| Transpose | Complete | +1 | -1 | → Usual | `[offset + n]` |
| Merge ⟨2,1,1⟩ | Direct | +1 | +1 | - | `[offset + n + 1]` |
| Split | Enter | -1 | +1 | → Splitting | `[offset + n]` |
| Split | Complete | +0 | -1 | → Usual | `[offset + n]` |

**Pattern Recognition**:
- **Enter multi-step**: `offset - 1` (stay at same word position)
- **Complete transposition**: `offset + 1` (jump 2 word positions)
- **Complete split**: `offset + 0` (advance 1 word position)
- **Merge**: `offset + 1` (like transposition complete, but direct)

---

## 5. State Machine Design

### 5.1 Position Variants

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GeneralizedPosition {
    // Existing variants
    INonFinal { offset: i32, errors: u8 },
    MFinal { offset: i32, errors: u8 },

    // NEW: Transposition states
    ITransposing { offset: i32, errors: u8 },
    MTransposing { offset: i32, errors: u8 },

    // NEW: Split states
    ISplitting { offset: i32, errors: u8 },
    MSplitting { offset: i32, errors: u8 },
}
```

**Invariants**: Same as existing I-type and M-type invariants apply to transposing/splitting variants.

### 5.2 State Transition Diagram

```
┌─────────────────┐
│  I/M-NonFinal   │ (Usual State)
│   (offset#e)    │
└────────┬────────┘
         │
         ├──(transpose enter)──► ┌───────────────┐
         │  offset-1, e+1        │ I/M-Transposing│
         │                       │  (offset#(e+1)_t)│
         │                       └────────┬───────┘
         │                                │
         │                                └──(match cv[0])──► I/M-NonFinal
         │                                   offset+1, e-1     (jumped +2 word pos)
         │
         ├──(split enter)──────► ┌───────────────┐
         │  offset-1, e+1        │  I/M-Splitting │
         │                       │  (offset#(e+1)_s)│
         │                       └────────┬───────┘
         │                                │
         │                                └──(match cv[0])──► I/M-NonFinal
         │                                   offset+0, e-1     (advanced +1 word pos)
         │
         ├──(merge direct)─────► I/M-NonFinal
         │  offset+1, e+1        (consumed +2 input, +1 word)
         │
         └──(standard ops)─────► I/M-NonFinal
            various offsets
```

### 5.3 Subsumption Rules

**Key Principle**: Only positions of the same variant can subsume each other.

**Rationale**:
- Transposing/splitting positions are more specific than usual positions
- They represent intermediate states with additional constraints
- Different states represent different futures in the automaton

**Implementation**:
```rust
pub fn subsumes(
    p1: &GeneralizedPosition,
    p2: &GeneralizedPosition,
    max_distance: u8
) -> bool {
    use GeneralizedPosition::*;

    // Only same-variant positions can subsume each other
    match (p1, p2) {
        (INonFinal { .. }, INonFinal { .. }) |
        (MFinal { .. }, MFinal { .. }) => {
            // Existing subsumption logic
            subsumes_same_type(p1, p2, max_distance)
        }
        (ITransposing { .. }, ITransposing { .. }) |
        (MTransposing { .. }, MTransposing { .. }) => {
            // Same logic as INonFinal (same invariants)
            subsumes_same_type(p1, p2, max_distance)
        }
        (ISplitting { .. }, ISplitting { .. }) |
        (MSplitting { .. }, MSplitting { .. }) => {
            // Same logic as INonFinal (same invariants)
            subsumes_same_type(p1, p2, max_distance)
        }
        _ => false,  // Different variants never subsume
    }
}
```

**Example**:
- `I+0#1` (usual) does NOT subsume `I+0#1_t` (transposing)
- `I+0#1_t` (transposing) does NOT subsume `I+0#1` (usual)
- `I+(-1)#1` (usual) subsumes `I+0#1` (usual) if errors equal

---

## 6. Implementation Phases

### Phase 2d.1: Position Variants (2-3 hours)

**Objective**: Add new position variants without changing behavior

**Tasks**:

1. **Update GeneralizedPosition enum** (`position.rs`):
   ```rust
   #[derive(Clone, Debug, PartialEq, Eq, Hash)]
   pub enum GeneralizedPosition {
       INonFinal { offset: i32, errors: u8 },
       MFinal { offset: i32, errors: u8 },
       ITransposing { offset: i32, errors: u8 },   // NEW
       MTransposing { offset: i32, errors: u8 },   // NEW
       ISplitting { offset: i32, errors: u8 },     // NEW
       MSplitting { offset: i32, errors: u8 },     // NEW
   }
   ```

2. **Add constructor methods**:
   ```rust
   pub fn new_i_transposing(offset: i32, errors: u8, max_distance: u8)
       -> Result<Self, PositionError>

   pub fn new_m_transposing(offset: i32, errors: u8, max_distance: u8)
       -> Result<Self, PositionError>

   pub fn new_i_splitting(offset: i32, errors: u8, max_distance: u8)
       -> Result<Self, PositionError>

   pub fn new_m_splitting(offset: i32, errors: u8, max_distance: u8)
       -> Result<Self, PositionError>
   ```

   **Note**: Use same invariants as INonFinal/MFinal for transposing/splitting variants.

3. **Update Display implementation**:
   ```rust
   impl fmt::Display for GeneralizedPosition {
       fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
           match self {
               // ... existing ...
               ITransposing { offset, errors } =>
                   write!(f, "I + {}#{}_t", offset, errors),
               MTransposing { offset, errors } =>
                   write!(f, "M + {}#{}_t", offset, errors),
               ISplitting { offset, errors } =>
                   write!(f, "I + {}#{}_s", offset, errors),
               MSplitting { offset, errors } =>
                   write!(f, "M + {}#{}_s", offset, errors),
           }
       }
   }
   ```

4. **Update Ord implementation**:
   ```rust
   impl Ord for GeneralizedPosition {
       fn cmp(&self, other: &Self) -> std::cmp::Ordering {
           // Sort by: (variant_priority, errors, offset)
           // I-types before M-types
           // Within I-types: NonFinal < Transposing < Splitting
           // ...
       }
   }
   ```

5. **Update accessor methods** (`offset()`, `errors()`):
   ```rust
   pub fn offset(&self) -> i32 {
       match self {
           INonFinal { offset, .. } |
           MFinal { offset, .. } |
           ITransposing { offset, .. } |
           MTransposing { offset, .. } |
           ISplitting { offset, .. } |
           MSplitting { offset, .. } => *offset,
       }
   }
   ```

6. **Update pattern matches throughout codebase**:
   - Ensure all `match` statements on GeneralizedPosition are exhaustive
   - Add default cases for new variants (no behavior change yet)

**Files Modified**:
- `src/transducer/generalized/position.rs` (+80 lines)

**Validation**:
- All existing tests pass (no behavior changes)
- `cargo test` passes
- `cargo clippy` clean
- New constructors work correctly

**Test Cases**:
```rust
#[test]
fn test_new_i_transposing_valid() {
    let pos = GeneralizedPosition::new_i_transposing(0, 1, 2).unwrap();
    assert_eq!(pos.offset(), 0);
    assert_eq!(pos.errors(), 1);
}

#[test]
fn test_new_i_transposing_invalid() {
    // Same invariants as INonFinal
    assert!(GeneralizedPosition::new_i_transposing(3, 1, 2).is_err());
}

#[test]
fn test_display_transposing() {
    let pos = GeneralizedPosition::new_i_transposing(1, 2, 3).unwrap();
    assert_eq!(format!("{}", pos), "I + 1#2_t");
}
```

---

### Phase 2d.2: Subsumption Updates (1-2 hours)

**Objective**: Update subsumption logic for new variants

**Tasks**:

1. **Update subsumption function** (`subsumption.rs`):
   ```rust
   pub fn subsumes(
       p1: &GeneralizedPosition,
       p2: &GeneralizedPosition,
       max_distance: u8
   ) -> bool {
       use GeneralizedPosition::*;

       // Different variants never subsume each other
       match (p1, p2) {
           // Existing cases
           (INonFinal { offset: o1, errors: e1 },
            INonFinal { offset: o2, errors: e2 }) => {
               subsumes_i_type(*o1, *e1, *o2, *e2, max_distance)
           }
           (MFinal { offset: o1, errors: e1 },
            MFinal { offset: o2, errors: e2 }) => {
               subsumes_m_type(*o1, *e1, *o2, *e2, max_distance)
           }

           // NEW: Transposing variants (same logic as NonFinal)
           (ITransposing { offset: o1, errors: e1 },
            ITransposing { offset: o2, errors: e2 }) => {
               subsumes_i_type(*o1, *e1, *o2, *e2, max_distance)
           }
           (MTransposing { offset: o1, errors: e1 },
            MTransposing { offset: o2, errors: e2 }) => {
               subsumes_m_type(*o1, *e1, *o2, *e2, max_distance)
           }

           // NEW: Splitting variants (same logic as NonFinal)
           (ISplitting { offset: o1, errors: e1 },
            ISplitting { offset: o2, errors: e2 }) => {
               subsumes_i_type(*o1, *e1, *o2, *e2, max_distance)
           }
           (MSplitting { offset: o1, errors: e1 },
            MSplitting { offset: o2, errors: e2 }) => {
               subsumes_m_type(*o1, *e1, *o2, *e2, max_distance)
           }

           // Different variants never subsume
           _ => false,
       }
   }
   ```

2. **Add unit tests for variant subsumption**:
   ```rust
   #[test]
   fn test_same_variant_subsumption() {
       // I+(-1)#1_t subsumes I+0#1_t
       let p1 = GeneralizedPosition::new_i_transposing(-1, 1, 2).unwrap();
       let p2 = GeneralizedPosition::new_i_transposing(0, 1, 2).unwrap();
       assert!(subsumes(&p1, &p2, 2));
   }

   #[test]
   fn test_different_variant_no_subsumption() {
       // I+0#1 (usual) does NOT subsume I+0#1_t (transposing)
       let p1 = GeneralizedPosition::new_i(0, 1, 2).unwrap();
       let p2 = GeneralizedPosition::new_i_transposing(0, 1, 2).unwrap();
       assert!(!subsumes(&p1, &p2, 2));
       assert!(!subsumes(&p2, &p1, 2));
   }

   #[test]
   fn test_splitting_vs_transposing_no_subsumption() {
       // I+0#1_s (splitting) does NOT subsume I+0#1_t (transposing)
       let p1 = GeneralizedPosition::new_i_splitting(0, 1, 2).unwrap();
       let p2 = GeneralizedPosition::new_i_transposing(0, 1, 2).unwrap();
       assert!(!subsumes(&p1, &p2, 2));
       assert!(!subsumes(&p2, &p1, 2));
   }
   ```

**Files Modified**:
- `src/transducer/generalized/subsumption.rs` (+30 lines)

**Validation**:
- All subsumption tests pass
- Edge cases covered (same offset/errors, different variants)
- No regressions in existing subsumption behavior

---

### Phase 2d.3: Transposition Implementation (4-5 hours)

**Objective**: Implement transposition enter and complete logic

**Tasks**:

1. **Update `successors_i_type()` in `state.rs`**:

   ```rust
   fn successors_i_type(
       &self,
       offset: i32,
       errors: u8,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       let mut successors = Vec::new();
       let n = self.max_distance as i32;
       let match_index = (offset + n) as usize;
       let next_match_index = (offset + n + 1) as usize;

       // ... existing logic for standard operations ...

       // NEW: Transposition support
       // Check if we have a transposition operation
       let has_transpose_op = operations.operations().iter()
           .any(|op| op.consume_x() == 2 && op.consume_y() == 2);

       if has_transpose_op && errors < self.max_distance {
           // Enter transposition: check next position
           if next_match_index < bit_vector.len()
               && bit_vector.is_match(next_match_index)
           {
               // Enter transposing state: offset-1, errors+1
               if let Ok(trans) = GeneralizedPosition::new_i_transposing(
                   offset - 1,
                   errors + 1,
                   self.max_distance
               ) {
                   successors.push(trans);
               }
           }
       }

       successors
   }
   ```

2. **Update `successors()` to handle ITransposing positions**:

   ```rust
   fn successors(
       &self,
       pos: &GeneralizedPosition,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       match pos {
           GeneralizedPosition::INonFinal { offset, errors } => {
               self.successors_i_type(*offset, *errors, operations, bit_vector)
           }
           GeneralizedPosition::MFinal { offset, errors } => {
               self.successors_m_type(*offset, *errors, operations, bit_vector)
           }

           // NEW: Handle transposing positions
           GeneralizedPosition::ITransposing { offset, errors } => {
               self.successors_i_transposing(*offset, *errors, bit_vector)
           }
           GeneralizedPosition::MTransposing { offset, errors } => {
               self.successors_m_transposing(*offset, *errors, bit_vector)
           }

           // Splitting positions handled later
           _ => Vec::new(),
       }
   }
   ```

3. **Implement `successors_i_transposing()` helper**:

   ```rust
   fn successors_i_transposing(
       &self,
       offset: i32,
       errors: u8,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       let mut successors = Vec::new();
       let n = self.max_distance as i32;
       let match_index = (offset + n) as usize;

       // Complete transposition: check current position
       if match_index < bit_vector.len()
           && bit_vector.is_match(match_index)
       {
           // Complete transposition: offset+1, errors-1
           if let Ok(succ) = GeneralizedPosition::new_i(
               offset + 1,  // Jump 2 word positions
               errors - 1,   // Decrement error (was incremented on enter)
               self.max_distance
           ) {
               successors.push(succ);
           }
       }

       successors
   }
   ```

4. **Implement M-type transposition** (same logic for MTransposing):

   ```rust
   fn successors_m_type(
       &self,
       offset: i32,
       errors: u8,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       // ... existing logic ...

       // NEW: Transposition for M-type
       let has_transpose_op = operations.operations().iter()
           .any(|op| op.consume_x() == 2 && op.consume_y() == 2);

       if has_transpose_op && errors < self.max_distance {
           // Similar logic to I-type
           // ...
       }

       successors
   }
   ```

5. **Add comprehensive tests** (`automaton.rs`):

   ```rust
   #[cfg(test)]
   mod transposition_tests {
       use super::*;
       use crate::transducer::OperationSet;

       #[test]
       fn test_transposition_distance_zero() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(0, ops);

           assert!(automaton.accepts("test", "test"));
           assert!(!automaton.accepts("test", "tset"));  // Requires 1 error
       }

       #[test]
       fn test_transposition_adjacent_swap_middle() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "test" → "tset" (swap 'e' and 's')
           assert!(automaton.accepts("test", "tset"));
       }

       #[test]
       fn test_transposition_adjacent_swap_start() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "test" → "etst" (swap 't' and 'e')
           assert!(automaton.accepts("test", "etst"));
       }

       #[test]
       fn test_transposition_adjacent_swap_end() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "test" → "tets" (swap 's' and 't')
           assert!(automaton.accepts("test", "tets"));
       }

       #[test]
       fn test_transposition_longer_words() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "algorithm" → "lagorithm" (swap 'a' and 'l')
           assert!(automaton.accepts("algorithm", "lagorithm"));
       }

       #[test]
       fn test_transposition_rejects_non_adjacent() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "test" → "tsta" (non-adjacent swap) requires 2 errors
           assert!(!automaton.accepts("test", "tsta"));
       }

       #[test]
       fn test_transposition_multiple_swaps() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(2, ops);

           // "abcd" → "badc" (two adjacent swaps)
           assert!(automaton.accepts("abcd", "badc"));
       }

       #[test]
       fn test_transposition_with_standard_operations() {
           let ops = OperationSet::with_transposition();
           let automaton = GeneralizedAutomaton::with_operations(2, ops);

           // Combine transposition with substitution
           // "test" → "txst" (substitute e→x) → "tsxt" (transpose)
           assert!(automaton.accepts("test", "tsxt"));
       }
   }
   ```

**Files Modified**:
- `src/transducer/generalized/state.rs` (+120 lines)
- `src/transducer/generalized/automaton.rs` (+150 lines)

**Cross-Validation**:
```rust
#[test]
fn test_cross_validation_transposition() {
    use crate::transducer::universal::{UniversalAutomaton, Transposition};

    let generalized = GeneralizedAutomaton::with_operations(
        2,
        OperationSet::with_transposition()
    );
    let universal = UniversalAutomaton::<Transposition>::new(2);

    let test_cases = vec![
        ("test", "tset", true),
        ("test", "etst", true),
        ("test", "tets", true),
        ("test", "test", true),
        ("test", "tsta", false),  // Non-adjacent
        ("abcd", "badc", true),
        ("", "", true),
        ("a", "a", true),
    ];

    for (word, input, expected) in test_cases {
        let gen_result = generalized.accepts(word, input);
        let uni_result = universal.accepts(word, input);

        assert_eq!(
            gen_result, uni_result,
            "Mismatch for ({}, {}): generalized={}, universal={}",
            word, input, gen_result, uni_result
        );
        assert_eq!(gen_result, expected);
    }
}
```

**Validation**:
- All transposition tests pass
- Cross-validation with Universal automaton succeeds
- No regressions in existing tests

---

### Phase 2d.4: Merge Implementation (2-3 hours)

**Objective**: Implement merge operation (direct, no intermediate state)

**Tasks**:

1. **Update `successors_i_type()` for merge**:

   ```rust
   fn successors_i_type(
       &self,
       offset: i32,
       errors: u8,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       // ... existing logic ...

       // NEW: Merge operation support
       let has_merge_op = operations.operations().iter()
           .any(|op| op.consume_x() == 2 && op.consume_y() == 1);

       if has_merge_op && errors < self.max_distance {
           let next_match_index = (offset + n + 1) as usize;

           // Merge: consume 2 input chars, match 1 word char
           if next_match_index < bit_vector.len()
               && bit_vector.is_match(next_match_index)
           {
               // Direct transition: offset+1, errors+1
               if let Ok(merge) = GeneralizedPosition::new_i(
                   offset + 1,
                   errors + 1,
                   self.max_distance
               ) {
                   successors.push(merge);
               }
           }
       }

       successors
   }
   ```

2. **Update M-type** (similar logic for M-type positions)

3. **Add merge tests**:

   ```rust
   #[cfg(test)]
   mod merge_tests {
       use super::*;

       #[test]
       fn test_merge_simple() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "ab" → "a" (merge two input chars into one word char)
           assert!(automaton.accepts("ab", "a"));
           assert!(automaton.accepts("abc", "ac"));  // merge 'ab' → 'a'
           assert!(automaton.accepts("xab", "xa"));  // merge at end
       }

       #[test]
       fn test_merge_at_start() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           assert!(automaton.accepts("ab", "a"));
           assert!(automaton.accepts("abcd", "acd"));
       }

       #[test]
       fn test_merge_at_end() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           assert!(automaton.accepts("xab", "xa"));
           assert!(automaton.accepts("testab", "testa"));
       }

       #[test]
       fn test_merge_with_standard_operations() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(2, ops);

           // Combine merge with substitution
           assert!(automaton.accepts("test", "txst"));
       }
   }
   ```

**Files Modified**:
- `src/transducer/generalized/state.rs` (+60 lines)
- `src/transducer/generalized/automaton.rs` (+80 lines)

**Cross-Validation**:
```rust
#[test]
fn test_cross_validation_merge() {
    use crate::transducer::universal::{UniversalAutomaton, MergeAndSplit};

    let generalized = GeneralizedAutomaton::with_operations(
        1,
        OperationSet::with_merge_split()
    );
    let universal = UniversalAutomaton::<MergeAndSplit>::new(1);

    let test_cases = vec![
        ("ab", "a", true),
        ("abc", "ac", true),
        ("xab", "xa", true),
        ("test", "test", true),
    ];

    for (word, input, expected) in test_cases {
        assert_eq!(
            generalized.accepts(word, input),
            universal.accepts(word, input),
            "Mismatch for ({}, {})", word, input
        );
    }
}
```

---

### Phase 2d.5: Split Implementation (3-4 hours)

**Objective**: Implement split enter and complete logic

**Tasks**:

1. **Update `successors_i_type()` for split enter**:

   ```rust
   fn successors_i_type(
       &self,
       offset: i32,
       errors: u8,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       // ... existing logic ...

       // NEW: Split operation support
       let has_split_op = operations.operations().iter()
           .any(|op| op.consume_x() == 1 && op.consume_y() == 2);

       if has_split_op && errors < self.max_distance {
           let match_index = (offset + n) as usize;

           // Enter split: check current position
           if match_index < bit_vector.len()
               && bit_vector.is_match(match_index)
           {
               // Enter splitting state: offset-1, errors+1
               if let Ok(split) = GeneralizedPosition::new_i_splitting(
                   offset - 1,
                   errors + 1,
                   self.max_distance
               ) {
                   successors.push(split);
               }
           }
       }

       successors
   }
   ```

2. **Add split completion in `successors()`**:

   ```rust
   fn successors(
       &self,
       pos: &GeneralizedPosition,
       operations: &OperationSet,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       match pos {
           // ... existing cases ...

           // NEW: Handle splitting positions
           GeneralizedPosition::ISplitting { offset, errors } => {
               self.successors_i_splitting(*offset, *errors, bit_vector)
           }
           GeneralizedPosition::MSplitting { offset, errors } => {
               self.successors_m_splitting(*offset, *errors, bit_vector)
           }
       }
   }
   ```

3. **Implement `successors_i_splitting()` helper**:

   ```rust
   fn successors_i_splitting(
       &self,
       offset: i32,
       errors: u8,
       bit_vector: &CharacteristicVector,
   ) -> Vec<GeneralizedPosition> {
       let mut successors = Vec::new();
       let n = self.max_distance as i32;
       let match_index = (offset + n) as usize;

       // Complete split: check current position for second word char
       if match_index < bit_vector.len()
           && bit_vector.is_match(match_index)
       {
           // Complete split: offset+0, errors-1
           if let Ok(succ) = GeneralizedPosition::new_i(
               offset,      // +0 (stays same!)
               errors - 1,  // Decrement error
               self.max_distance
           ) {
               successors.push(succ);
           }
       }

       successors
   }
   ```

4. **Add split tests**:

   ```rust
   #[cfg(test)]
   mod split_tests {
       use super::*;

       #[test]
       fn test_split_simple() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           // "a" → "ab" (split one input char into two word chars)
           assert!(automaton.accepts("a", "ab"));
           assert!(automaton.accepts("ac", "abc"));  // split 'a' → 'ab'
           assert!(automaton.accepts("xa", "xab"));  // split at end
       }

       #[test]
       fn test_split_at_start() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           assert!(automaton.accepts("a", "ab"));
           assert!(automaton.accepts("acd", "abcd"));
       }

       #[test]
       fn test_split_at_end() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(1, ops);

           assert!(automaton.accepts("xa", "xab"));
           assert!(automaton.accepts("testa", "testab"));
       }

       #[test]
       fn test_split_with_standard_operations() {
           let ops = OperationSet::with_merge_split();
           let automaton = GeneralizedAutomaton::with_operations(2, ops);

           // Combine split with other operations
           assert!(automaton.accepts("test", "txst"));
       }
   }
   ```

**Files Modified**:
- `src/transducer/generalized/state.rs` (+120 lines)
- `src/transducer/generalized/automaton.rs` (+100 lines)

---

### Phase 2d.6: Integration Testing (2-3 hours)

**Objective**: Comprehensive testing and cross-validation

**Tasks**:

1. **Mixed operation tests**:

   ```rust
   #[test]
   fn test_all_operations_combined() {
       let ops = OperationSetBuilder::new()
           .with_standard_ops()
           .with_transposition()
           .with_merge()
           .with_split()
           .build();

       let automaton = GeneralizedAutomaton::with_operations(2, ops);

       // Complex sequences
       assert!(automaton.accepts("algorithm", "lagorithm"));  // transpose
       assert!(automaton.accepts("test", "txst"));            // substitute
       assert!(automaton.accepts("ab", "a"));                 // merge
       assert!(automaton.accepts("a", "ab"));                 // split
   }
   ```

2. **Edge case tests**:

   ```rust
   #[test]
   fn test_edge_cases() {
       let ops = OperationSet::with_merge_split();
       let automaton = GeneralizedAutomaton::with_operations(2, ops);

       // Empty strings
       assert!(automaton.accepts("", ""));

       // Single character
       assert!(automaton.accepts("a", "a"));

       // Boundary conditions
       assert!(automaton.accepts("test", "test"));
   }
   ```

3. **Full cross-validation suite**:

   ```rust
   #[test]
   fn test_comprehensive_cross_validation() {
       // Test all three operations against Universal automaton
       // ... comprehensive test matrix ...
   }
   ```

4. **Performance benchmarking** (optional, for documentation):

   ```rust
   #[bench]
   fn bench_generalized_vs_universal(b: &mut Bencher) {
       // Compare performance
   }
   ```

**Files Modified**:
- `src/transducer/generalized/automaton.rs` (+100 lines)

**Validation**:
- 100% test pass rate
- Cross-validation success
- Performance analysis

---

### Phase 2d.7: Documentation and Cleanup (1-2 hours)

**Objective**: Document implementation and create completion summary

**Tasks**:

1. **Update inline documentation** in source files
2. **Create completion document**: `docs/generalized/phase2d_complete.md`
3. **Update README** (if applicable)
4. **Add doc examples**:

   ```rust
   /// # Examples
   ///
   /// ```rust
   /// use liblevenshtein::transducer::{GeneralizedAutomaton, OperationSet};
   ///
   /// let ops = OperationSet::with_transposition();
   /// let automaton = GeneralizedAutomaton::with_operations(1, ops);
   ///
   /// assert!(automaton.accepts("test", "tset"));  // Transposition
   /// ```
   ```

5. **Final code review** and cleanup

**Deliverables**:
- `docs/generalized/phase2d_complete.md` (completion summary)
- Updated inline docs
- Clean, well-documented code

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Position Variants**:
- ✓ Constructor validation for each variant
- ✓ Display formatting
- ✓ Ordering (PartialOrd, Ord)
- ✓ Accessor methods (offset(), errors())

**Subsumption**:
- ✓ Same-variant subsumption
- ✓ Cross-variant non-subsumption
- ✓ Edge cases (same offset/errors, different variants)

**Successor Generation**:
- ✓ Each operation type in isolation
- ✓ I-type vs M-type differences
- ✓ Bit vector boundary conditions

### 7.2 Integration Tests

**Transposition**:
- ✓ Adjacent swaps at start, middle, end
- ✓ Multiple transpositions
- ✓ Combined with standard operations
- ✓ Distance 0 edge cases
- ✓ Longer words
- ✓ Rejects non-adjacent swaps

**Merge**:
- ✓ Merge at start, middle, end
- ✓ Combined with standard operations
- ✓ Distance limits

**Split**:
- ✓ Split at start, middle, end
- ✓ Combined with standard operations
- ✓ Distance limits

**Mixed Operations**:
- ✓ All operations combined
- ✓ Complex sequences
- ✓ Edge cases (empty, single char)

### 7.3 Cross-Validation

**Against Universal Automaton**:

```rust
fn cross_validate_operation<V: PositionVariant>(
    operation_name: &str,
    test_cases: Vec<(&str, &str, bool)>
) {
    let generalized = GeneralizedAutomaton::with_operations(
        2,
        get_operation_set(operation_name)
    );
    let universal = UniversalAutomaton::<V>::new(2);

    for (word, input, expected) in test_cases {
        assert_eq!(
            generalized.accepts(word, input),
            universal.accepts(word, input),
            "Mismatch for ({}, {}) in {}", word, input, operation_name
        );
        assert_eq!(generalized.accepts(word, input), expected);
    }
}
```

**Test Coverage Matrix**:

| Operation | Start | Middle | End | Empty | Single | Combined |
|-----------|-------|--------|-----|-------|--------|----------|
| Transpose | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Merge | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Split | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 7.4 Performance Tests

**Baseline Comparison**:
- GeneralizedAutomaton vs UniversalAutomaton
- Expected: 10-20% slower (due to runtime dispatch)
- Measure state count growth
- Memory usage (SmallVec still effective?)

**Regression Prevention**:
- Ensure single-character performance unchanged
- No unexpected allocations
- Subsumption efficiency maintained

---

## 8. Risk Assessment

### 8.1 Technical Risks

**Medium Risk**:

1. **Subsumption Complexity**
   - **Risk**: New variants complicate subsumption logic
   - **Mitigation**: Extensive unit tests, cross-validation
   - **Impact**: Medium (may need iteration)

2. **Bit Vector Semantics**
   - **Risk**: Multi-character operations may misinterpret bit vector indices
   - **Mitigation**: Follow Universal implementation exactly, thorough testing
   - **Impact**: Medium (debugging may be time-consuming)

3. **State Transition Edge Cases**
   - **Risk**: Missing edge cases in two-step operations
   - **Mitigation**: Comprehensive test matrix, cross-validation
   - **Impact**: Medium (could delay completion)

**Low Risk**:

1. **Position Variant Design**
   - **Risk**: Proven in Universal automaton
   - **Impact**: Low

2. **Offset Calculations**
   - **Risk**: Formulas validated against lazy automaton
   - **Impact**: Low

3. **Breaking Changes**
   - **Risk**: None (internal changes only)
   - **Impact**: None

### 8.2 Schedule Risks

**Optimistic Scenario** (11 hours):
- Smooth implementation
- No surprises
- 1.5 days calendar time

**Realistic Scenario** (15-22 hours):
- Some debugging needed
- Iteration on subsumption
- 2-3 days calendar time

**Pessimistic Scenario** (25-30 hours):
- Major debugging required
- Subsumption issues
- Edge case hunting
- 3-4 days calendar time

### 8.3 Mitigation Strategies

1. **Incremental Implementation**:
   - Complete each phase before moving to next
   - Validate at each step
   - Don't accumulate technical debt

2. **Cross-Validation Early and Often**:
   - Test against Universal after each operation
   - Catch issues immediately
   - Build confidence incrementally

3. **Reference Implementation**:
   - Keep Universal automaton code open
   - Copy patterns where appropriate
   - Don't reinvent proven solutions

---

## 9. Success Criteria

### 9.1 Functional Requirements

- [ ] All transposition tests pass
- [ ] All merge tests pass
- [ ] All split tests pass
- [ ] All existing tests still pass (backward compatibility)
- [ ] Cross-validation with Universal automaton succeeds
- [ ] Mixed operation tests pass
- [ ] Edge case tests pass

### 9.2 Performance Requirements

- [ ] Single-character operations: no regression
- [ ] Multi-character operations: within 20% of Universal automaton
- [ ] Memory usage: SmallVec optimization still effective
- [ ] No unexpected allocations in hot paths

### 9.3 Code Quality

- [ ] No clippy warnings
- [ ] All tests passing
- [ ] Comprehensive documentation
- [ ] Clean commit history
- [ ] Inline comments explain offset calculations

### 9.4 Deliverables

- [ ] Working implementation (all phases complete)
- [ ] Comprehensive test suite (100+ tests)
- [ ] Performance analysis (benchmarks)
- [ ] Completion document (`phase2d_complete.md`)
- [ ] Updated inline documentation

---

## 10. References

### 10.1 Universal Automaton Implementation

**Transposition**:
- `src/transducer/universal/position.rs:219-264` (I-type transposition)
- `src/transducer/universal/position.rs:285-332` (M-type transposition)
- `docs/universal/transposition_phase2_summary.md`

**Merge/Split**:
- `src/transducer/universal/position.rs:366-521` (I-type and M-type merge/split)
- `docs/universal/merge_split_phase3_complete.md`
- `docs/universal/merge_split_analysis.md`

### 10.2 Lazy Automaton (Validation)

- `src/transducer/transition.rs:280-495` (lazy automaton transitions)
- Lines 287, 347: Transposition
- Lines 420, 454: Merge
- Lines 415, 459: Split

### 10.3 Theoretical Foundation

- Mitankin, Petar. "Universal Levenshtein Automata - Building and Properties"
- Located: `/home/dylon/Papers/Approximate String Matching/Universal Levenshtein Automata - Building and Properties/`

### 10.4 Previous Analysis

- `docs/generalized/phase2d_analysis.md` (initial analysis, now superseded)
- `docs/universal/README.md` (Universal automaton status)

---

## Appendix A: Implementation Checklist

### Before Starting
- [ ] Review Universal automaton implementations
- [ ] Review lazy automaton
- [ ] Study offset calculation formulas
- [ ] Understand subsumption rules
- [ ] Set up cross-validation test framework

### Phase 2d.1: Position Variants
- [ ] Add ITransposing, MTransposing variants
- [ ] Add ISplitting, MSplitting variants
- [ ] Update Display impl
- [ ] Update PartialOrd/Ord impls
- [ ] Add constructors with invariant checks
- [ ] Update all pattern matches (exhaustive)
- [ ] Run tests (should all pass, no behavior change)

### Phase 2d.2: Subsumption
- [ ] Update subsumption.rs for new variants
- [ ] Implement same-variant-only rule
- [ ] Add unit tests for variant subsumption
- [ ] Test cross-variant non-subsumption
- [ ] Run tests

### Phase 2d.3: Transposition
- [ ] Detect transposition in OperationSet
- [ ] Implement enter logic (I-type)
- [ ] Implement complete logic (I-type)
- [ ] Implement enter logic (M-type)
- [ ] Implement complete logic (M-type)
- [ ] Add transposition tests
- [ ] Cross-validate against Universal
- [ ] Run all tests

### Phase 2d.4: Merge
- [ ] Detect merge in OperationSet
- [ ] Implement merge logic (I-type)
- [ ] Implement merge logic (M-type)
- [ ] Add merge tests
- [ ] Cross-validate against Universal
- [ ] Run all tests

### Phase 2d.5: Split
- [ ] Detect split in OperationSet
- [ ] Implement enter logic (I-type)
- [ ] Implement complete logic (I-type)
- [ ] Implement enter logic (M-type)
- [ ] Implement complete logic (M-type)
- [ ] Add split tests
- [ ] Cross-validate against Universal
- [ ] Run all tests

### Phase 2d.6: Integration
- [ ] Mixed operation tests
- [ ] Edge case tests
- [ ] Performance benchmarks
- [ ] Full cross-validation suite
- [ ] Final test run (should be 100%)

### Phase 2d.7: Documentation
- [ ] Update inline docs
- [ ] Create phase2d_complete.md
- [ ] Add doc examples
- [ ] Code review
- [ ] Git commit

---

## Appendix B: Quick Reference Tables

### Offset Calculation Quick Reference

| Operation | Step | Formula | Error | State |
|-----------|------|---------|-------|-------|
| Transpose | Enter | `offset - 1` | `+1` | `→ _t` |
| Transpose | Complete | `offset + 1` | `-1` | `→ usual` |
| Merge | Direct | `offset + 1` | `+1` | `usual` |
| Split | Enter | `offset - 1` | `+1` | `→ _s` |
| Split | Complete | `offset + 0` | `-1` | `→ usual` |

### Bit Vector Index Quick Reference

| Operation | Step | Index | Check |
|-----------|------|-------|-------|
| Transpose | Enter | `offset + n + 1` | Next char |
| Transpose | Complete | `offset + n` | Current char |
| Merge | Direct | `offset + n + 1` | Next char |
| Split | Enter | `offset + n` | Current char |
| Split | Complete | `offset + n` | Current char |

---

## Conclusion

This implementation plan provides a clear, incremental path to adding multi-character operation support to GeneralizedAutomaton. The approach prioritizes:

1. **No breaking changes** - 100% backward compatible
2. **Proven design** - mirrors Universal automaton's success
3. **Incremental progress** - each phase builds on previous work
4. **Continuous validation** - cross-validation at every step
5. **Manageable scope** - 15-22 hours (2-3 days)

**Key Success Factors**:
- State variants solve the problem without API changes
- Offset formulas validated against Universal and lazy automata
- Comprehensive testing strategy ensures correctness
- Cross-validation provides high confidence

**Ready to implement in a fresh session!**
