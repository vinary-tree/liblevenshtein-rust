# Phase 4: Substitution Problem - Deep Analysis

## Current Status
- **6/10 acceptance tests passing**
- **Insertions and deletions work** ✓
- **Exact matches work** ✓
- **Substitutions DON'T work** ✗

## The Core Problem

The issue is a **semantic mismatch** between:
1. **Thesis transitions** (Definition 7, page 14): Defined for concrete positions using bit vector prefixes
2. **Universal positions**: Use relative offsets and need position-aware matching

## Thesis Transition Definition (Concrete)

```
δ^D,ε_e(i#e, b) = {
    {i+1#e}                          if 1 < b (match at START of subword)
    {i#e+1, i+1#e+1}                 if b = 0^k & e < n (all zeros)
    {i#e+1, i+1#e+1, i+j#e+j-1}     if 0 < b (has match later)
    {i#e+1}                          if b = ε (empty)
}
```

The bit vector b = β(x, s_n(w,i)) where s_n(w,i) starts at position i-n.

The formula "1 < b" checks if b **starts with 1**, meaning position i-n matches.

## Universal Position Semantics

For I+offset#errors at input position k:
- Concrete word position: i = k + offset
- Bit vector: β(x, s_n(w,k)) starting at position k-n
- **Position i in word corresponds to bit_vector[n + offset]**

## The Mismatch

The thesis checks `starts_with_one()` which is position `i-n` (the START of the subword).

But for universal positions, we care about position `k + offset` (the CURRENT position), which is at index `n + offset` in the bit vector.

## Why Insertions/Deletions Work

The delete/insert transitions in the thesis don't depend on matching at a specific position - they're always available when errors remain. That's why they work correctly.

## Why Substitutions Don't Work

Substitutions require advancing BOTH word position AND input position when there's a mismatch. This is the `(t+1)#(e+1)` transition.

In the thesis, this is part of case "b = 0^k" (all zeros) or "0 < b" (has match later). But these check the ENTIRE bit vector, not just the current position.

For universal positions, we need:
- **If match at current position** (bit_vector[n+offset] == 1): Match transition only
- **If NO match at current position** (bit_vector[n+offset] == 0): Delete + Substitute + Skip-to-match

## Attempted Fixes

### Fix 1: Return early on substitution
```rust
if !bit_vector.is_match(match_index) {
    // Substitution
    successors.push(I+offset#(errors+1));
    return successors;
}
```
**Problem**: Broke insertions because it returned early and didn't generate delete/insert/skip transitions.

### Fix 2: Add substitution in Case 4
```rust
if match_index < bit_vector.len() {
    successors.push(I+offset#(errors+1)); // Substitution
}
```
**Problem**: Case 4 only runs when bit vector "0 < b" (starts with 0, has 1 later), not for all non-matches.

## The Correct Solution

The issue is that **the thesis formulas don't directly apply to universal positions**. We need to adapt them.

For I-type universal positions, the logic should be:

```rust
let match_index = n + offset;

if match_index < bit_vector.len() {
    if bit_vector.is_match(match_index) {
        // MATCH at current position: only this successor
        return {I+offset#errors};
    } else {
        // NO MATCH at current position: generate error successors
        if errors < max_distance {
            let mut successors = vec![];

            // DELETE: skip this word character
            successors.push(I+(offset-1)#(errors+1));

            // SUBSTITUTE: advance both, add error
            successors.push(I+offset#(errors+1));

            // SKIP-TO-MATCH: if there's a match later in window
            if let Some(next_match_idx) = find_next_match_after(match_index) {
                let distance = next_match_idx - match_index;
                successors.push(I+(offset+distance)#(errors+distance-1));
            }

            return successors;
        }
    }
}

// Position beyond window - fallback to generic error logic
...
```

This handles substitutions explicitly when the current position doesn't match.

## Next Step

Implement the correct solution above.
