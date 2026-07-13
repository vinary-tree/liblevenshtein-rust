# Phase 4: Bit Vector Position Bug Analysis

## Problem Statement

The universal automaton is failing to accept identical strings like `accepts("test", "test")`. The transition fails at step 3 because all positions have exhausted the error budget by step 2.

## Root Cause

The bit vector positioning and offset semantics are misaligned. Let me trace through the failing case:

### Step 1: Input 't', Position 1

**Word**: "test"
**Subword**: "$$test" (positions -1, 0, 1, 2, 3, 4 relative to input position 1)
**Bit vector**: `[false, false, true, false, false, true]`
**Current state**: {I+0#0}

From position I+0#0:
- offset=0 means we're at concrete position i = offset + input_pos = 0 + 1 = 1
- The bit vector represents s_n(w,1) = "$$test"
- Position 0 in bit_vector corresponds to word position i-n = 1-2 = -1 (padding '$')
- Position 1 in bit_vector corresponds to word position 0 (padding '$')
- **Position 2 in bit_vector corresponds to word position 1** (THIS IS THE MATCH!)

**Expected behavior**: Since we're at word position 1 and input char='t' matches word[1]='t', we should advance WITHOUT error.

**Actual behavior**: The code checks `bit_vector.starts_with_one()` which is FALSE because the bit vector starts with `[false, false, ...]`. The match is at position 2, not position 0!

## The Conceptual Issue

The problem is that **the bit vector index doesn't directly correspond to the offset**.

For position I+offset#errors at input position k:
- Concrete word position: i = offset + k
- Bit vector window: positions [k-n, min(|w|, k+n+1)]
- The word position i corresponds to bit_vector[n + offset]

In our case:
- I+0#0 at input k=1
- Concrete word position: i = 0 + 1 = 1
- Bit vector window: [1-2, min(4, 1+2+1)] = [-1, 4]
- Word position 1 corresponds to bit_vector[2 + 0] = bit_vector[2] ✓

## The Fix

The match check needs to account for the offset within the bit vector window:

```rust
// Current (WRONG):
if bit_vector.starts_with_one() {
    // Match at position 1
    if let Ok(succ) = Self::new_i(offset, errors, max_distance) {
        successors.push(succ);
    }
    return successors;
}
```

Should be:

```rust
// Check if there's a match at the current position
// For I+offset#errors at input k, word position i = offset + k
// In bit vector s_n(w,k), this corresponds to index: n + offset
let match_index = (max_distance as i32 + offset) as usize;

if match_index < bit_vector.len() && bit_vector.get(match_index) == Some(true) {
    // Match at current position - advance without error
    if let Ok(succ) = Self::new_i(offset, errors, max_distance) {
        successors.push(succ);
    }
    return successors;
}
```

Wait, but that's not quite right either because `get()` might not exist...

Actually, let me reconsider. The thesis says:

> δ^D,ε_e(i#e, b) where b is the bit vector

The position i#e is a **concrete position** (not universal). The universal position I+t#e gets converted:
- Before transition: I+t#e → concrete position i = t + k (where k is input position)
- Apply δ^D,ε_e(i#e, b)
- After transition: convert back using I^ε

But the bit vector b is computed as β(x, s_n(w, k)) where s_n starts at position k-n.

So for I+t#e at input k:
- Concrete i = t + k
- Bit vector starts at k-n
- Position i in the bit vector is at index: i - (k-n) = (t+k) - (k-n) = t + n

YES! The match index should be `offset + n`.

## Verification

Let's verify with our example:
- Position: I+0#0
- Input position k=1
- Max distance n=2
- Concrete position i = 0 + 1 = 1
- Bit vector s_n(w,1) starts at position 1-2 = -1
- Position 1 in bit vector: index = 1 - (-1) = 2 ✓
- Using formula: index = offset + n = 0 + 2 = 2 ✓

Perfect! The formula is: **match_index = offset + n**

## Implementation Fix

The `starts_with_one()` check is fundamentally wrong. We need to check the bit at position `offset + n`.

Additionally, for the "skip to match" logic, we need to calculate the correct j value relative to the current position, not relative to the start of the bit vector.
