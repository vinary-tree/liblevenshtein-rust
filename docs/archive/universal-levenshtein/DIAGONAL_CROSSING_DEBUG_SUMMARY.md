# Diagonal Crossing Debug Session - Summary

## Session Objective
Continue debugging the diagonal crossing integration for Universal Levenshtein Automata (Phase 4).

## Initial Status
- **149/156 tests passing** (95.5%)
- **7 acceptance tests failing** - all for non-empty word/input pairs
- Diagonal crossing integration disabled (commented out)

## Major Discovery: Bit Vector Position Bug

### Root Cause
The fundamental bug was in `src/transducer/universal/position.rs` in the I-type successor function.

**The Bug**: Used `bit_vector.starts_with_one()` to check for matches, which only checks index 0. For universal positions I+offset#errors, the match should be checked at index `offset + n`.

**Why This Matters**:
- For position I+offset#errors at input k, concrete word position is i = offset + k
- Bit vector s_n(w,k) starts at position k-n
- Position i in bit vector corresponds to index: (offset + k) - (k-n) = offset + n

### Fix Applied
Changed from:
```rust
if bit_vector.starts_with_one() { ... }
```

To:
```rust
let match_index = (max_distance as i32 + offset) as usize;
if match_index < bit_vector.len() && bit_vector.is_match(match_index) { ... }
```

## Progress Made

### Tests Fixed (3/7 → 6/10)
✓ test_accepts_identical ("test" → "test")
✓ test_accepts_deletion ("test" → "tet")
✓ test_accepts_insertion ("test" → "teast")
✓ test_accepts_empty_to_empty
✓ test_accepts_empty_word
✓ test_accepts_to_empty

### Tests Still Failing (4/10)
✗ test_accepts_substitution ("test" → "text")
✗ test_accepts_multiple_edits ("test" → "best")
✗ test_accepts_n1 ("test" → "text" with n=1)
✗ test_accepts_longer_words ("algorithm" → "algorythm")

## Ongoing Investigation: Substitutions

###Problem
All failing tests involve **substitutions** - cases where the input character doesn't match the word character, requiring both positions to advance with an error.

### Attempts Made

**Attempt 1**: Return early when no match at current position
```rust
if !bit_vector.is_match(match_index) {
    // Generate substitution
    successors.push(I+offset#(errors+1));
    return successors;
}
```
**Result**: Broke insertions (6/10 → 8/10 then back to 6/10)

**Attempt 2**: Add substitution alongside delete/insert
```rust
if match_index < bit_vector.len() && !is_match(match_index) {
    // DELETE
    successors.push(I+(offset-1)#(errors+1));
    // SUBSTITUTE
    successors.push(I+offset#(errors+1));
    // SKIP-TO-MATCH
    ...
}
```
**Result**: Same 4 substitution tests still failing (6/10)

### Current Hypothesis

The issue may be related to **when the substitution logic applies**. The thesis formulas check the **entire bit vector pattern** (starts_with_one, is_all_zeros), but we're checking a **specific position** (match_index).

Possible issues:
1. Substitution successors being generated but then subsumed incorrectly
2. Acceptance condition not handling substitution paths correctly
3. The bit vector window not including the right positions for substitutions
4. Need to distinguish between "substitute at current position" vs "insert into input"

## Files Modified

1. `src/transducer/universal/position.rs` (lines 387-460)
   - Added match_index calculation
   - Rewrote I-type successor logic to be position-aware
   - Added explicit substitution handling

2. `src/transducer/universal/automaton.rs`
   - Removed debug output (was added/removed during investigation)

3. `src/transducer/universal/state.rs`
   - Removed debug output
   - Diagonal crossing remains disabled

## Documentation Created

1. `PHASE4_BIT_VECTOR_BUG.md` - Analysis of the bit vector position bug
2. `PHASE4_STATUS.md` - Current status tracking
3. `PHASE4_SUBSTITUTION_ANALYSIS.md` - Deep analysis of substitution problem
4. This file - Session summary

## Next Steps

1. **Add detailed tracing** for "test" → "text" to see:
   - What successors are generated at each step
   - Whether substitution positions exist in final state
   - Why acceptance check fails

2. **Verify subsumption** isn't incorrectly removing substitution paths

3. **Check acceptance condition** for states with errors > 0

4. Consider whether the **thesis formulas need different adaptation** for universal positions

## Key Insight

The thesis defines transitions for **concrete** positions where the bit vector represents a fixed neighborhood. Universal positions use **relative offsets** which change the semantics. The formulas may need more significant adaptation than just index calculation.

## Current State

- Significant progress: 60% → 60% (6/10 passing, but different tests)
- Core infrastructure works: matches, deletions, insertions all functional
- Remaining issue is narrow: only substitutions fail
- The bug is likely subtle - probably in how substitution successors interact with subsumption or acceptance checking
