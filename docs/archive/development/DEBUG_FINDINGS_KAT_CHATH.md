# Debug Findings: "kat"→"chath" Double Split Failure

**Date:** 2025-11-13  
**Issue:** Phonetic double split test failing at distance 1  
**Status:** ROOT CAUSE IDENTIFIED

## Executive Summary

The test `accepts("kat", "chath")` with distance=1 fails immediately due to an **overly restrictive length check** in the `accepts()` function. The check assumes standard Levenshtein operations and doesn't account for phonetic split operations that can "expand" the word.

**Location:** `/src/transducer/generalized/automaton.rs`, lines 300-301

## Problem Details

### Test Case
```rust
// From src/transducer/generalized/automaton.rs:1612
assert!(automaton.accepts("kat", "chath"),
       "Two splits (k→ch, t→th) with fractional weights should work at distance 1");
```

### Expected Behavior
- "kat" should match "chath" with distance 1
- k→ch split: weight 0.15 → truncates to 0 errors
- t→th split: weight 0.15 → truncates to 0 errors
- Total: 0 + 0 = 0 errors, well within distance 1

### Actual Behavior
- Function returns `false` immediately at line 300-301
- Never processes any characters
- Never attempts phonetic split operations

## Root Cause

### The Problematic Code
```rust
// src/transducer/generalized/automaton.rs:300-301
if input.len() > word.len() + self.max_distance as usize {
    return false;  // ← REJECTS TOO EARLY
}
```

### The Calculation
- word = "kat" (length 3)
- input = "chath" (length 5)
- max_distance = 1
- Check: `5 > 3 + 1` → `5 > 4` → **TRUE, returns false**

## Why This Happens

The length check is based on standard Levenshtein distance theory, which assumes:

| Operation | Δ Word Length | Δ Input Length | Net Effect |
|-----------|---------------|----------------|------------|
| Match     | 0             | 0              | 0          |
| Substitute| 0             | 0              | 0          |
| Insert    | 0             | +1             | +1         |
| Delete    | +1            | 0              | -1         |

**Maximum input length = word_length + max_distance** (all insertions)

But phonetic split operations break this assumption:

| Operation      | Consume Word | Consume Input | Expansion |
|----------------|--------------|---------------|-----------|
| k→ch split ⟨1,2⟩| 1 char       | 2 chars       | +1        |
| t→th split ⟨1,2⟩| 1 char       | 2 chars       | +1        |

**With two splits:**
- "kat" (3 chars) + 2 expansions = 5 effective chars
- Can match "chath" (5 chars)
- But length check only allows up to 4 chars!

## Evidence from Working vs Failing Cases

### ✓ Working: "kair"→"chair" (Single Split)
```
word length    = 4
input length   = 5
max_distance   = 1
Check: 5 > 4+1 = 5 > 5 = FALSE → proceeds ✓
Expansions needed: 1 (within bounds)
```

### ✗ Failing: "kat"→"chath" (Double Split)
```
word length    = 3
input length   = 5
max_distance   = 1
Check: 5 > 3+1 = 5 > 4 = TRUE → rejects ✗
Expansions needed: 2 (exceeds bounds by 1)
```

## Detailed Analysis

### Phonetic Split Operation Semantics

A split ⟨1,2⟩ operation:
1. Enters splitting state on first input char (e.g., 'c')
2. Completes on second input char (e.g., 'h')
3. Matches ONE word character (e.g., 'k')
4. Consumes TWO input characters ('c' + 'h')

**Net effect:** Word effectively "expands" by +1 character

### Multiple Splits with Fractional Weights

Fractional weight operations (w < 1.0):
- Truncate to 0 errors when applied
- Can be applied multiple times within error budget
- Each application adds +1 expansion

For distance n, theoretically up to n such splits could occur (though in practice fewer).

## Proposed Solutions

### Option 1: Dynamic Bound Based on Operations (Most Precise)

```rust
// Calculate maximum expansion from split operations
fn max_expansion_per_split(&self) -> usize {
    self.operations.operations().iter()
        .filter(|op| op.consume_y() > op.consume_x())
        .map(|op| (op.consume_y() - op.consume_x()) as usize)
        .max()
        .unwrap_or(0)
}

// In accepts():
let max_expansion = self.max_expansion_per_split();
let max_splits = if max_expansion > 0 {
    // Estimate: allow up to max_distance splits with fractional weight
    self.max_distance as usize
} else {
    0
};

let max_input_len = word.len() + self.max_distance as usize + (max_splits * max_expansion);
if input.len() > max_input_len {
    return false;
}
```

**Pros:** Most accurate, optimal performance  
**Cons:** Requires operation set analysis

### Option 2: Conditional Check (Simpler)

```rust
// Only apply strict length check for standard operations
let has_expansion_ops = self.operations.operations().iter()
    .any(|op| op.consume_y() > op.consume_x());

if !has_expansion_ops && input.len() > word.len() + self.max_distance as usize {
    return false;
}
```

**Pros:** Simple, correct for current use case  
**Cons:** May allow unnecessary processing with expansion ops

### Option 3: Conservative Buffer (Simplest)

```rust
// Add buffer for potential phonetic operations
// Conservative: allow extra chars for phonetic expansion
let buffer = self.max_distance as usize;
if input.len() > word.len() + self.max_distance as usize + buffer {
    return false;
}
```

**Pros:** Simplest implementation  
**Cons:** Least precise, may process invalid inputs

## Recommendation

**Implement Option 2** as immediate fix:
- Correctly handles current phonetic operations
- Simple to implement and test
- No performance impact on standard operations

**Future enhancement:** Implement Option 1 if:
- Performance profiling shows the check is a bottleneck
- Need more precise bounds for complex operation sets

## Implementation Plan

1. **Modify** `/src/transducer/generalized/automaton.rs:300-301`
2. **Add test** for length check with expansion operations
3. **Verify** existing tests still pass
4. **Document** the change in code comments

## Testing Checklist

After fix:
- [ ] ✓ "kair"→"chair" (single split) still works
- [ ] ✓ "kat"→"chath" (double split) now works
- [ ] ✓ Standard Levenshtein cases unaffected
- [ ] ✓ Length check still rejects truly invalid inputs
- [ ] ✓ No performance regression for standard operations

## Related Files

- `/src/transducer/generalized/automaton.rs` - Contains problematic length check
- `/src/transducer/generalized/state.rs` - Split completion logic (working correctly)
- `/src/transducer/phonetic.rs` - Phonetic operation definitions
- `/src/transducer/operation_set.rs` - Operation set management

## Notes

The split completion logic in `state.rs` (lines 935-980 for I-type, 1018-1054 for M-type) is working correctly. The issue is purely in the pre-processing length check that prevents the automaton from even trying to process the input.

This is a classic case of an optimization (early rejection) being too aggressive for an extended operation set.
