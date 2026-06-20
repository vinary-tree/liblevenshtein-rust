# Transposition Fix - Cross-Validation with Lazy Automaton

**Date**: 2025-11-13
**Status**: Defect identified through cross-validation; resolved in Phase 2 summary

## Key Finding from transition.rs

The lazy automaton implementation (src/transducer/transition.rs) reveals the correct transposition logic:

### Transposition Initiation (lines 284-289, 327-331)

**When** `index_of_match(cv, h, k) == Some(1)` (match at position 1):
```rust
next.push(Position::new(i, e + 1)); // insertion
next.push(Position::new_special(i, e + 1)); // transposition start
next.push(Position::new(i + 1, e + 1)); // substitution
next.push(Position::new(i + 2, e + 1)); // direct transposition jump
```

**Interpretation**: When the current input character matches position 1 (the NEXT word position), we can:
- Do standard operations (insertion, substitution)
- OR enter transposition state
- OR jump directly ahead by 2 (for immediate transposition detection)

### Transposition Completion (lines 344-349)

**When in transposition state** (`is_special == true`):
```rust
if cv[h] {  // Check position 0 - CURRENT position
    next.push(Position::new(i + 2, e));
}
```

**Interpretation**: When in transposing state, check if the CURRENT input matches the CURRENT word position (cv[0]) to complete the transposition.

## Universal Automaton Mapping

For UniversalAutomaton with Mitankin's notation:

### Current Understanding (WRONG)

My current implementation:
- **Enter transposition** when `b[1] = 1` (match at next position)  ← This part seems correct!
- **Complete transposition** when `b[1] = 1` (match at next position) ← This is WRONG!

### Correct Mapping

Based on lazy automaton:
- **Enter transposition** when `b[1] = 1` (match at next position) ✓
- **Complete transposition** when `b[0] = 1` (match at CURRENT position) ✓

But wait! In the Transposing state, we've already advanced the position by 1. So the "current" position in Transposing state might correspond to a different bit vector index.

## The Real Issue

I think the problem is that I'm misunderstanding what position the Transposing state represents.

When we enter Transposing state from position i#e:
- We create position (i+1)#(e+1)_t
- This position has offset = (i+1) - (e+1) = i - e (same as original!)
- BUT we're processing the NEXT input character now

So when checking the bit vector in Transposing state:
- The bit vector corresponds to the NEXT input character
- Position (i+1)#(e+1)_t is at word position i+1
- We want to check if this NEXT input matches word position i (to complete the swap)
- In the bit vector for NEXT input: word position i is at index `n + (offset from perspective of next input)`

## Hypothesis H4: State Transition Logic Error

The issue might be:
1. **Usual → Transposing**: Check `b[1] = 1`, create position with offset+1, errors+1 ✓ (seems correct)
2. **Transposing → Usual**: Currently checking `b[0] = 1` at `match_index = n+offset`

But in Transposing state with position (i+1)#(e+1):
- offset = i+1-(e+1) = i-e
- We're now at word position i+1
- We want to match against word position i (one before current)
- In the bit vector: position i corresponds to index `n + (i - (i+1)) = n - 1`

Wait, that doesn't make sense either. The bit vector is always indexed from the current word window...

## Next Steps (Resume Point)

1. **Add comprehensive debug logging** to both Usual and Transposing state branches
2. **Trace "ab" → "ba"** step by step with actual bit vector values
3. **Compare with lazy automaton behavior** on same input
4. **Fix the Transposing state logic** based on findings
5. **Document the corrected understanding**

The debugging needs to show:
- Current position (offset, errors, state)
- Current input character and position
- Bit vector contents
- Which indices are checked
- What transitions are generated

This is complex enough that it requires careful step-by-step analysis with concrete examples.
