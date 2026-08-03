# MergeAndSplit Algorithm Bug Analysis

## Executive Summary

Investigation into MergeAndSplit algorithm failures revealed **two critical bugs**, neither of which were in the Schulz & Mihov paper's formal definition. The paper's subsumption logic is correct as specified. The bugs were in the **implementation details**: missing validation of the paper's strict inequality requirement and generating invalid split positions.

## Background

The user reported test failures showing:
- Query `"b"` → Dictionary `"aaa"` returned distance **3** instead of **2**
- Empty query `""` → Dictionary `"aaa"` incorrectly matched with distance **2** instead of rejecting (actual distance **3**)

## The Schulz & Mihov Paper's Formal Definition

The paper specifies MergeAndSplit subsumption with **strict inequality**:

1. `(i,e,false)` subsumes `(j,f,false)` iff `e < f` and `|j-i| <= f-e`
2. `(i,e,false)` subsumes `(j,f,true)` iff `e < f` and `|j-i| <= f-e`
3. `(i,e,true)` subsumes `(j,f,true)` iff `e < f` and `|j-i| <= f-e`
4. `(i,e,true)` cannot subsume `(j,f,false)` (implied)

Note the **strict inequality** `e < f`, NOT `e <= f`.

## Bug #1: Missing Strict Inequality Enforcement

### The Issue

The C++/Java implementations were missing an explicit `e < f` check. Instead, they only checked the distance formula:

```cpp
// C++/Java approach (BUGGY)
if (s && !t) return false;
return abs(i - j) <= (f - e);
```

When `e == f`, the formula becomes `abs(i - j) <= 0`, which means `i == j`. This inadvertently allowed subsumption when errors were equal AND positions were the same.

### The Impact

Consider position `(0,1,false)` and `(0,1,true)` at the same location:
- Both have `i=0, e=1`
- Formula: `abs(0-0) <= (1-1)` → `0 <= 0` → **TRUE**
- Result: `(0,1,false)` subsumes `(0,1,true)`

This is **wrong** because:
- `(0,1,true)` is a special position (split-in-progress)
- It represents a fundamentally different computational path
- Subsuming it prematurely terminates the split operation
- The split path could lead to a better distance!

### Example: Query "b" → "aaa"

Without strict inequality:
1. Start: `(0,0,false)`
2. Process first 'a': generates `(0,1,false)` (insert) and `(0,1,true)` (split start)
3. **BUG**: `(0,1,false)` subsumes `(0,1,true)` because `e==f==1` and `i==j==0`
4. Split path blocked → only insertion path remains
5. Result: distance 3 (delete 'b', insert 'aaa')

With strict inequality (correct):
1. Start: `(0,0,false)`
2. Process first 'a': generates `(0,1,false)` and `(0,1,true)`
3. **Both survive** because strict inequality prevents `e==f` subsumption
4. Split path: `(0,1,true)` → `(1,1,false)` completes split
5. Result: distance 2 (split 'b'→'aa', insert final 'a')

### The Fix

```rust
// Enforce strict inequality from paper
if e >= f {
    return false;
}

// Then check distance formula
let index_diff = i.abs_diff(j);
let error_diff = f - e;
index_diff <= error_diff
```

This explicitly enforces the paper's `e < f` requirement **before** checking the distance formula.

### Why C++/Java Appeared to Work

The C++/Java implementations had an accidental bug in subsumes.cpp that **masked** this issue:

```cpp
// Line 24 of subsumes.cpp (BUG!)
bool t = lhs->is_special();  // Should be rhs->is_special()
```

This bug caused `s` and `t` to always have the same value, so the code never reached the problematic branch when `!s && t`. The implementations "worked" by accident, but with a different bug.

## Bug #2: Invalid Split Positions for Empty/Short Queries

### The Issue

The transition logic generated split positions without checking if query characters were available:

```rust
// BUGGY CODE
next.push(Position::new_special(i, e + 1)); // Always generated!
```

For an **empty query** (length 0), this created:
1. Start: `(0,0,false)` at position 0
2. Process 'a': generate `(0,1,true)` (split start)
3. Process next 'a': split completes → `(1,1,false)`
4. **BUG**: `term_index=1` exceeds `query_length=0` (invalid position!)

### The Impact

Invalid positions beyond the query boundary created phantom match paths:
- Empty query `""` → `"aaa"` incorrectly matched with distance 2
- Automaton explored impossible computational states
- Distance calculations became unreliable

### The Fix

Add validation before generating split positions:

```rust
// Split operation: one query char becomes two dict chars
if i + 1 <= query_length {
    next.push(Position::new_special(i, e + 1));
}
```

This matches the existing merge validation:

```rust
// Merge operation: skip 2 query chars
if i + 2 <= query_length {
    next.push(Position::new(i + 2, e + 1));
}
```

**Rationale**:
- **Split**: Requires ≥1 query character to split into 2 dictionary characters
- **Merge**: Requires ≥2 query characters to merge into 1 dictionary character
- **Empty query**: Cannot perform split or merge operations

## Root Cause Analysis

### Why These Bugs Existed

1. **Subtle Paper Interpretation**: The strict inequality `e < f` is easy to miss when implementing the distance formula, especially since the formula itself doesn't enforce it.

2. **C++/Java Precedent**: The original implementations had a compensating bug that masked the subsumption issue, leading implementers to believe the logic was correct.

3. **Missing Boundary Checks**: The merge validation was added, but the corresponding split validation was overlooked.

4. **Edge Case Testing**: Empty query tests were not comprehensive enough to catch these issues early.

## Test Results

### Before Fixes

```
Query "b" → "aaa": distance 3 (expected 2) ❌
Query "" → "aaa" with max_dist=2: matched (should reject) ❌
```

### After Fixes

```
Query "b" → "aaa": distance 2 (split operation) ✅
Query "" → "aaa" with max_dist=2: correctly rejected ✅
All 182 library tests: passing ✅
MergeAndSplit cross-validation: 15/16 passing ✅
```

The remaining Unicode test failure is unrelated (issue with `DoubleArrayTrieChar` and leading spaces).

## Key Takeaways

1. **The paper is correct** - The Schulz & Mihov formal definition with strict inequality `e < f` is sound and necessary.

2. **Strict inequality is essential** - Allowing `e == f` subsumption incorrectly prunes valid computational paths for merge/split operations.

3. **Boundary validation matters** - Split positions require the same careful validation as merge positions.

4. **Special positions are special** - Positions with `is_special=true` represent fundamentally different states and must be handled carefully in subsumption logic.

## Implementation Notes

### Modified Files

- `src/transducer/position.rs:152-178` - Added strict inequality check in MergeAndSplit subsumption
- `src/transducer/transition.rs:327-426` - Added split position validation at 4 locations

### Key Code Locations

```rust
// position.rs - Subsumption with strict inequality
if e >= f {
    return false;  // Enforce e < f from paper
}

// transition.rs - Split position validation
if i + 1 <= query_length {
    next.push(Position::new_special(i, e + 1));
}
```

## References

- Schulz, K. U., & Mihov, S. (2002). Fast string correction with Levenshtein automata.
- liblevenshtein-java: https://github.com/vinary-tree/liblevenshtein-java
- liblevenshtein-cpp: https://github.com/vinary-tree/liblevenshtein-cpp

## Conclusion

Both bugs stemmed from incomplete implementation of the paper's requirements rather than errors in the paper itself. The fixes ensure that:

1. The paper's strict inequality requirement is explicitly enforced
2. Split positions are only generated when query characters are available
3. The MergeAndSplit algorithm correctly handles all edge cases including empty queries

The Rust implementation now correctly implements the Schulz & Mihov algorithm and, with the strict inequality fix, is actually **more correct** than the original C++/Java implementations.
