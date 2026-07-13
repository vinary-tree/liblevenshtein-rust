# Phase 2d Analysis: Multi-Character Operations

**Date**: 2025-11-13
**Status**: PLANNING

## Discovery: Universal Automaton Transposition Status

After examining the Universal automaton codebase, I discovered:

1. **Transposition variant is defined** (`src/transducer/universal/position.rs:110-129`):
   - `Transposition` enum with `Usual` and `TranspositionState` variants
   - Implements `PositionVariant` trait

2. **But transposition successor generation is NOT implemented**:
   - `UniversalPosition<V>::successors()` only handles Standard operations
   - No separate `successors_i_type_transposition()` or `successors_m_type_transposition()` methods
   - No transposition-specific logic in state transitions

3. **This means**:
   - Transposition support is **planned but not yet implemented** in Universal automaton
   - We cannot cross-validate GeneralizedAutomaton against UniversalAutomaton<Transposition>
   - We need to implement transposition from first principles using Mitankin's thesis

## Theoretical Foundation

From Mitankin's thesis, transposition is defined as operation ⟨2, 2, 1.0⟩:
- **consume_x = 2**: Consumes 2 characters from dictionary word
- **consume_y = 2**: Consumes 2 characters from query input
- **weight = 1.0**: Costs 1 error

### Transposition Semantics

Transposition swaps two adjacent characters:
- **Example**: "test" → "tset" (swap 'e' and 's')
- **Condition**: `word[i..i+2]` reversed equals `input[j..j+2]`
- **Formal**: If `word[i] = b`, `word[i+1] = a`, `input[j] = a`, `input[j+1] = b`, then transposition applies

## Implementation Strategy for GeneralizedAutomaton

### Phase 2d Goals

1. ✅ Remove `consume_x/y > 1` skip checks
2. ⬜ Implement multi-character character extraction from word/input
3. ⬜ Implement multi-character bit vector matching
4. ⬜ Add transposition operation to OperationSet
5. ⬜ Test transposition with simple cases

### Key Challenges

#### Challenge 1: Character Extraction

**Problem**: Current successor generation only looks at single characters via bit vector.

**Solution**: Need to extract substrings from word and input:
```rust
// For operation with consume_x=2, consume_y=2
let dict_chars: &str = /* extract 2 chars from word at position i */;
let query_chars: &str = /* extract 2 chars from input at position j */;

if op.is_transposition() && dict_chars.chars().rev().eq(query_chars.chars()) {
    // Transposition applies
}
```

#### Challenge 2: Bit Vector Semantics

**Problem**: CharacteristicVector encodes single-character matches. For transposition, we need to check TWO positions match (but in reverse order).

**Current bit vector**: `β(a, w)` returns 1 at position `i` if `w[i] = a`

**For transposition**: Need to check if `w[i..i+2]` reversed equals `input[j..j+2]`

**Solution Options**:

**Option A**: Extend CharacteristicVector to support multi-character lookups
```rust
impl CharacteristicVector {
    /// Check if word[index..index+len] matches pattern (possibly reversed)
    pub fn matches_at(&self, index: usize, pattern: &str, reverse: bool) -> bool;
}
```

**Option B**: Access word directly in successor generation (breaking current abstraction)
```rust
fn successors_i_type(
    &self,
    offset: i32,
    errors: u8,
    operations: &OperationSet,
    bit_vector: &CharacteristicVector,
    word: &str,  // NEW parameter
    input_char: char, // NEW parameter
) -> Vec<GeneralizedPosition>
```

**Option C**: Check transposition via operation's `can_apply()` method with extracted substrings

**Recommendation**: **Option C** - Use operation's `can_apply()` method. This is the cleanest and most aligned with the OperationSet design.

#### Challenge 3: Position Offset Calculation

For I-type positions with transposition ⟨2,2,w⟩:

**Current single-char logic**:
- Match ⟨1,1,0⟩: `offset` stays same (I^ε conversion: `(t+1)#e → I+t#e`)
- Delete ⟨1,0,1⟩: `offset - 1` (I^ε conversion: `t#(e+1) → I+(t-1)#(e+1)`)
- Insert/Substitute ⟨1,1,1⟩: `offset` stays same

**Multi-char transposition**:
- Transposition ⟨2,2,1⟩: Similar to match but consumes 2 from both sides
- I^ε conversion: `(t+2)#(e+1) → I+(t+1)#(e+1)`
- **New offset**: `offset + 1` (advance by 1, not 0)
- **New errors**: `errors + 1`

**General formula**:
```rust
let offset_delta = op.consume_x() as i32 - 1;  // -1 for I^ε conversion
let new_offset = offset + offset_delta;
let new_errors = errors + op.weight() as u8;
```

### Implementation Plan

#### Step 1: Add Word/Input Parameters (BREAKING CHANGE)

This requires refactoring the API to pass word and input context:

```rust
// state.rs
pub fn transition(
    &self,
    operations: &OperationSet,
    bit_vector: &CharacteristicVector,
    word: &str,           // NEW
    input: &str,          // NEW
    input_position: usize, // NEW
) -> Option<Self>
```

**Impact**: This is a BREAKING change that affects all call sites.

**Alternative**: Store word/input in state or pass via a context struct.

#### Step 2: Implement Multi-Character Matching

```rust
fn can_apply_operation(
    op: &OperationType,
    word: &str,
    word_position: usize,
    input: &str,
    input_position: usize,
) -> bool {
    let consume_x = op.consume_x() as usize;
    let consume_y = op.consume_y() as usize;

    // Check bounds
    if word_position + consume_x > word.len() {
        return false;
    }
    if input_position + consume_y > input.len() {
        return false;
    }

    // Extract substrings
    let word_chars = &word[word_position..word_position + consume_x];
    let input_chars = &input[input_position..input_position + consume_y];

    // Check if operation can apply
    op.can_apply(word_chars, input_chars)
}
```

#### Step 3: Update Successor Generation

Remove the `consume_x/y > 1` skip checks and handle multi-char operations:

```rust
for op in operations.operations() {
    // Compute actual word and input positions
    let word_pos = (input_position as i32 + offset) as usize;
    let input_pos = input_position;

    // Check if operation can apply
    if !can_apply_operation(op, word, word_pos, input, input_pos) {
        continue;
    }

    // Compute successor offset based on consume_x
    let offset_delta = op.consume_x() as i32 - 1;  // I^ε conversion
    let new_offset = offset + offset_delta;
    let new_errors = errors + op.weight() as u8;

    if new_errors <= self.max_distance {
        if let Ok(succ) = GeneralizedPosition::new_i(new_offset, new_errors, self.max_distance) {
            successors.push(succ);
        }
    }
}
```

### Testing Strategy

#### Test Case 1: Adjacent Swap
```rust
#[test]
fn test_transposition_adjacent() {
    let mut ops = OperationSet::standard();
    ops.add_operation(OperationType::new(2, 2, 1.0)); // Transposition

    let automaton = GeneralizedAutomaton::with_operations(2, ops);

    // "test" vs "tset" (swap 'e' and 's')
    assert!(automaton.accepts("test", "tset"));
}
```

#### Test Case 2: Transposition at Boundaries
```rust
#[test]
fn test_transposition_at_start() {
    // "test" vs "etst" (swap at position 0)
    assert!(automaton.accepts("test", "etst"));
}

#[test]
fn test_transposition_at_end() {
    // "test" vs "tets" (swap 't' and 's')
    assert!(automaton.accepts("test", "tets"));
}
```

#### Test Case 3: Transposition + Other Edits
```rust
#[test]
fn test_transposition_with_substitution() {
    // Total distance 2: transposition + substitution
    // "test" → "txst" (e→x) → "tsxt" (swap x and s)
    assert!(automaton.accepts("test", "tsxt"));
}
```

## Deferred Tasks

Due to the complexity of implementing transposition and the API changes required, I propose **deferring Phase 2d** and documenting the current state as **Phase 2c COMPLETE**.

### What's Complete

- ✅ OperationSet infrastructure
- ✅ Runtime-configurable operations
- ✅ Dynamic operation iteration
- ✅ Custom operation weights
- ✅ Multiple operations of same type
- ✅ All Phase 1 tests passing (57/57)
- ✅ 100% backward compatibility

### What's Remaining for Full Multi-Char Support

- ⬜ API refactoring to pass word/input context
- ⬜ Multi-character extraction logic
- ⬜ Transposition operation implementation
- ⬜ Multi-character bit vector semantics
- ⬜ Transposition tests
- ⬜ Cross-validation (blocked on Universal automaton transposition implementation)

## Recommendation

**Complete Phase 2 as "Partial Multi-Char Support"**:
- Phase 2a ✅ Infrastructure
- Phase 2b ✅ Parameter threading
- Phase 2c ✅ Single-char operations refactored
- Phase 2d ⬜ **DEFERRED** - Requires major API changes

**Document current status**:
- GeneralizedAutomaton supports runtime-configurable single-character operations
- Multi-character operations (transposition, phonetic digraphs) require API refactoring
- Foundation is solid and ready for future enhancement

**Benefits of deferring**:
1. Preserve current clean API
2. Avoid breaking changes
3. Maintain 100% backward compatibility
4. Allow time for Universal automaton transposition implementation
5. Can cross-validate once Universal automaton supports transposition

## Alternative: Minimal Multi-Char Implementation

If we want to make progress on transposition without major API changes:

### Approach: Add Transposition Variant to GeneralizedPosition

Similar to how Universal has `Transposition::TranspositionState`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneralizedPosition {
    INonFinal { offset: i32, errors: u8 },
    MFinal { offset: i32, errors: u8 },
    ITransposition { offset: i32, errors: u8, pending_char: char }, // NEW
}
```

This allows tracking transposition state without needing full word/input context.

**Pros**:
- No API breaking changes
- Can implement transposition incrementally
- Matches Universal automaton's design

**Cons**:
- More complex position types
- Need to update subsumption logic
- May not generalize to other multi-char operations

## Next Steps

**Option 1: Defer Phase 2d**
1. Update phase2_progress.md to mark Phase 2c as complete
2. Create phase3_planning.md for future multi-char support
3. Focus on other priorities

**Option 2: Implement Minimal Transposition**
1. Add `ITransposition` position variant
2. Implement transposition-specific successor logic
3. Add transposition tests
4. Document limitations

**Recommendation**: **Option 1 - Defer** until Universal automaton provides a reference implementation to validate against.
