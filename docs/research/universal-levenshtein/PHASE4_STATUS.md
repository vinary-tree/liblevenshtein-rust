# Phase 4: Universal Automaton - Current Status

## Summary

Phase 4 implementation has made significant progress. The bit vector positioning bug has been fixed, and 6 out of 10 acceptance tests now pass.

## Test Status

### Passing (6/10):
1. `test_accepts_empty_to_empty` ✓
2. `test_accepts_deletion` ✓
3. `test_accepts_identical` ✓
4. `test_accepts_empty_word` ✓
5. `test_accepts_insertion` ✓
6. `test_accepts_to_empty` ✓

### Failing (4/10):
1. `test_accepts_substitution` - "test" → "text" (1 substitution)
2. `test_accepts_multiple_edits` - "test" → "best" (1 substitution)
3. `test_accepts_n1` - "test" → "text" with n=1
4. `test_accepts_longer_words` - "algorithm" → "algorythm" (1 substitution)

## Key Fix Applied

**Bit Vector Position Bug (FIXED)**

The major bug was in how I-type positions checked for matches. The code was using `starts_with_one()` which checked if the bit vector started with a 1, but this was wrong because:

- For position I+offset#errors at input k, the concrete word position is i = offset + k
- The bit vector s_n(w,k) starts at position k-n
- Position i in the bit vector corresponds to index: i - (k-n) = offset + n

**Fix**: Changed from `bit_vector.starts_with_one()` to `bit_vector.is_match(offset + max_distance)`.

This fixed identical word matching and single edit operations (insertion, deletion).

## Remaining Issue

**Substitutions are NOT working**.

All 4 failing tests involve substitutions:
- "test" → "text": substitution at position 3 (s→x)
- "test" → "best": substitution at position 1 (t→b)
- "algorithm" → "algorythm": substitution at position 5 (i→y)

### Hypothesis

The error transitions should handle substitutions correctly via the `i+1#e+1` transition (which becomes I+offset#(errors+1) after I^ε conversion). This transition is present in the code.

The issue might be:
1. The substitution transition is being subsumed by another position
2. The acceptance condition is rejecting valid final states
3. There's a subtle bug in how positions are converted or checked

### Next Steps

1. Add detailed debug tracing for "test" → "text" to see:
   - What positions are generated at each step
   - Whether the substitution position I+0#1 is created and persists
   - What the final state looks like
   - Why acceptance fails

2. Check if the substitution position is being subsumed incorrectly

3. Verify the acceptance condition for states with errors > 0

## Diagonal Crossing

Diagonal crossing integration is now available through `UniversalState::transition_with_consumption()`, which carries explicit `length_diff` metadata and performs conversion when the tracked difference crosses the configured distance bound.

The older `transition()` method remains a compatibility path and intentionally does not perform diagonal crossing because it cannot infer whether the query side, dictionary side, both sides, or neither side consumed a character.

Focused verification passed on 2026-06-19:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  env CARGO_BUILD_JOBS=1 cargo test -j1 --lib transducer::universal::state::tests -- --test-threads=1
```

Result: 36/36 universal state tests passed, including length-difference updates and I/M conversion boundary tests.

## Code Changes Made

1. `src/transducer/universal/position.rs`:
   - Line 387-402: Fixed I-type match check to use `offset + max_distance` index
   - Line 448-469: Fixed skip-to-match logic with proper bounds checking

2. `src/transducer/universal/automaton.rs`:
   - Removed debug output (was temporarily added for investigation)

3. `src/transducer/universal/state.rs`:
   - Removed debug output
   - Added `length_diff` tracking and consumption-aware diagonal crossing

## Documentation Created

1. `PHASE4_BIT_VECTOR_BUG.md`: Detailed analysis of the bit vector positioning bug and fix
2. This file: Current status tracking
