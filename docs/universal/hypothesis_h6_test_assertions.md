# Hypothesis H6: Test Assertions May Be Incorrect

**Date**: 2025-11-13
**Status**: Resolved historical investigation

## Summary

After fixing the main transposition offset defect (H5), 10 out of 12 tests passed. The 2 remaining failures were later resolved as incorrect test assertions; see `transposition_phase2_summary.md`.

## Failing Test 1: `test_transposition_empty_and_single_char`

### Assertion
```rust
assert!(!automaton.accepts("a", "b")); // Would need substitution
```

### Issue
The implementation ACCEPTS "a" → "b" via substitution (distance 1), but the test expects REJECT.

### Analysis

From `src/transducer/transition.rs` (lazy automaton, lines 284-301):
- The transposition variant DOES include ALL standard operations (insertion, deletion, substitution)
- Line 288: `next.push(Position::new(i + 1, 1)); // substitution`
- This is consistent across ALL branches of the transposition transition function

The comment "Would need substitution" suggests the test author believed transposition mode should NOT allow substitution. However, both the lazy implementation and Mitankin's thesis indicate that transposition is an ADDITIONAL operation, not a REPLACEMENT for standard operations.

### Hypothesis

**The test assertion is incorrect.** The Transposition variant correctly accepts standard operations, so "a" → "b" with n=1 SHOULD accept via substitution.

### Verification Resolution

Mitankin's thesis and the lazy automaton transition logic confirm that the transposition variant includes standard operations.

---

## Failing Test 2: `test_transposition_with_repeated_chars`

###Assertion
```rust
assert!(automaton.accepts("aabb", "baab")); // swap first two
```

### Issue
The implementation REJECTS "aabb" → "baab", but test expects ACCEPT.

### Analysis

Word: "aabb" = positions [1='a', 2='a', 3='b', 4='b']
Input: "baab" = reading ['b', 'a', 'a', 'b']

#### Simulation Trace
```
Input[0]='b' at k=1:
  - Position I+0#0 at word_pos=1 (word[1]='a')
  - Bit vector for 'b': word[1]='a'(no), word[2]='a'(no)
  - No transposition entry (would need match at j=1, but word[2]='a' != 'b')

Input[1]='a' at k=2:
  - Positions from standard operations (deletion/substitution)
  - No transposition completion

Final: Ends at word_pos=3, needs word_pos=4
```

The issue is that at input k=1, reading 'b':
- word[1]='a' (no match at current position)
- word[2]='a' (no match at next position - would need 'b' here for transposition)
- word[3]='b' (match at j=2, not j=1, so no transposition)

**Question: How does "aabb" → "baab" work with only adjacent transpositions?**

Transforming "aabb" → "baab":
- Positions: a₁ a₂ b₃ b₄ → b? a? a? b?
- This would require swapping a₁ with b₃ (NOT adjacent!)

### Hypothesis

**The test may also be incorrect.** Converting "aabb" to "baab" requires NON-adjacent transposition, which is not supported by the ⟨2,2,1⟩ operation.

However, there's a possibility I'm misunderstanding how transposition works in the automaton. The trace for "aabb" vs "abab" showed transposition DOES work, so maybe there's a subtlety I'm missing.

### Alternative Interpretation

Maybe the automaton can perform transposition "lazily" across multiple input characters? Let me re-examine "abab" case:
- Input[0]='a': enters transposing state I+-1#1_t
- Input[1]='b': state persists
- Input[2]='a': completes transposition

This spans 3 input characters, but what's the actual word transformation?

At Input[0], we're at word position 1 (word[1]='a', matches 'a')
Entering transposing: we increment errors and stay at same word position
At Input[2], we complete and jump to word position 1+2=3?

Wait - I need to trace this more carefully to understand the transposition semantics.

### Verification Resolution

1. The lazy automaton rejects "aabb" vs "baab" as a non-adjacent transposition case.
2. The Universal automaton rejection is therefore correct.
3. The test assertion was wrong and was corrected in the Phase 2 summary.

---

## Hypothesis Test Plan

1. **For Test 1**: Check Mitankin's thesis definition of Transposition variant - does it include standard operations?

2. **For Test 2**:
   - Trace "aabb" vs "baab" through lazy automaton manually or with debug output
   - Compare with "aabb" vs "abab" which we know works
   - Identify the specific semantic difference

3. **Decision Matrix Resolution**:
   - Test 1 assertion was wrong because standard substitution is valid.
   - Test 2 assertion was wrong because the requested transformation is not adjacent.
   - No Universal implementation defect remained for these cases.

## Resolution

The lazy automaton comparison was completed, findings were documented, and the two incorrect assertions were corrected.
