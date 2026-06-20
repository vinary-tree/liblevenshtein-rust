# Hypothesis H5: Offset Calculation Defect in Transposition Completion

**Date**: 2025-11-13
**Status**: Resolved historical investigation

## The Defect

For "ab" → "ba" with n=1:

**Observed Behavior**:
1. Input[0]='b': Position I+0#0 enters transposition → I+-1#1_t
2. Input[1]='a': Position I+-1#1_t completes transposition → I+2#0
3. Final check: word_pos = offset + input_len = 2 + 2 = 4, expected = 2
4. **REJECT** (should accept!)

## Root Cause Analysis

### The Problem with Offset Arithmetic

When completing transposition, I was using:
```rust
offset_new = offset_transposing + 3
```

This came from the calculation:
- Lazy: i#(e+1)_t → (i+2)#e (jump from position i to position i+2)
- Universal: offset = i - e, so offset_new = (i+2) - e = i - e + 2 = offset + 2

But wait! The transposing state has offset = i - (e+1), so:
- offset_new = (i+2) - e = (i - (e+1)) + 3 = offset_transposing + 3

This seemed correct, but it's producing word_pos = 4 instead of 2!

### The Key Insight: Input Position Matters

The issue is that **offsets are relative to the input position when they're evaluated**.

In the lazy automaton:
- Position i#e is ABSOLUTE: it's always at word position i

In the Universal automaton:
- Position I+offset#e at input k is at word position `i = offset + k`
- The SAME position I+offset#e represents DIFFERENT absolute word positions at different input positions!

So when we create a successor position I+offset'#e', we need to think about:
- At what input position k' will this position be evaluated?
- What absolute word position i' do we want at that time?
- offset' = i' - k'

### The Correct Calculation

**Entering transposition** (at input k):
- Current: I+offset#e at input k, word position i = offset + k
- Lazy creates: i#(e+1)_t (same absolute position i)
- Next input: k+1
- Universal needs: offset' such that offset' + (k+1) = i = offset + k
- Therefore: offset' = (offset + k) - (k+1) = offset - 1 ✓ (this was correct!)

**Completing transposition** (at input k):
- Current: I+offset#e_t at input k, word position i = offset + k
- Lazy creates: (i+2)#(e-1) (jump to absolute position i+2)
- Next input: k+1
- Universal needs: offset' such that offset' + (k+1) = i + 2 = (offset + k) + 2
- Therefore: offset' = (offset + k + 2) - (k+1) = offset + 1

**But I was using offset + 3!** That's the defect.

## The Fix

Change transposition completion from:
```rust
offset + 3
```

To:
```rust
offset + 1
```

### Verification with "ab" → "ba"

1. Input[0] (k=1): I+0#0 enters transposition
   - offset' = 0 - 1 = -1
   - Creates: I+-1#1_t

2. Input[1] (k=2): I+-1#1_t completes transposition
   - offset' = -1 + 1 = 0
   - Creates: I+0#0

3. Final check (after k=2):
   - word_pos = 0 + 2 = 2 = word.len() ✓

This should work!

## Implementation

Apply the fix to both I-type and M-type transposition completion:
- Line ~252: Change from `offset + 3` to `offset + 1`
- Line ~317: Change from `offset + 3` to `offset + 1`
