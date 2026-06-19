# H1 Optimization Implementation Plan

**Target**: `successors_i_type()` method in `src/transducer/generalized/state.rs`
**Current Performance**: 2.75% of total cycles (largest single function hotspot)
**Expected Improvement**: 2-4% additional speedup after H2
**Date**: 2025-11-18

---

## Executive Summary

The `successors_i_type()` method is the largest single function hotspot identified in baseline profiling at **2.75% of total cycles**. After completing H2 optimization (character vector caching, 28-31% speedup), this method remains the primary target for further optimization.

**Key Bottlenecks Identified**:
1. **String allocations**: Multiple `.to_string()` calls for `can_apply()` checks (lines 297-298, 316, 332, 349-350, 417-418, 473)
2. **Operation filtering overhead**: Repeated iteration and Vec collection for transpose/split operations (lines 365-367, 452-454)
3. **Vector allocations**: Successors Vec could use SmallVec to avoid heap allocations
4. **Multiple operation passes**: 4 separate loops over operations (standard, transpose, merge, split)

---

## Baseline Analysis

From `PHASE1_FLAMEGRAPH_ANALYSIS.md`:

- **successors_i_type**: 2.75% of cycles (320-line method)
- **Iterator::collect**: 4.08% (H2 optimization reduced this)
- **cfree (deallocation)**: 4.39% (H2 optimization reduced this)
- **OperationType::can_apply**: 0.79% (string allocation overhead visible)

**Post-H2 Status**:
- Character vector caching (H2) eliminated major allocation overhead
- Remaining bottleneck: String allocations within `can_apply()` calls
- Secondary bottleneck: Operation filtering/iteration patterns

---

## Optimization Strategies

### Strategy 1: Reduce String Allocations (**HIGH PRIORITY**)

**Problem**: Multiple `.to_string()` calls create temporary heap allocations for `can_apply()` checks.

**Current Code Pattern**:
```rust
// Line 297-298: Match operation
let word_char_str = word_slice_chars[match_index].to_string();
let input_char_str = input_char.to_string();
if op.can_apply(word_char_str.as_bytes(), input_char_str.as_bytes()) {
    // ...
}

// Line 316: Delete operation
let word_char_str = word_slice_chars[match_index].to_string();
if op.can_apply(word_char_str.as_bytes(), &[]) {
    // ...
}

// Line 332: Insert operation
let input_char_str = input_char.to_string();
if op.can_apply(&[], input_char_str.as_bytes()) {
    // ...
}

// Line 349-350: Substitute operation
let word_char_str = word_slice_chars[match_index].to_string();
let input_char_str = input_char.to_string();
if op.can_apply(word_char_str.as_bytes(), input_char_str.as_bytes()) {
    // ...
}

// Line 417-418: Merge operation (2,1)
let word_2chars: String = word_slice_chars[match_index..match_index+2].iter().collect();
let input_1char = input_char.to_string();
if op.can_apply(word_2chars.as_bytes(), input_1char.as_bytes()) {
    // ...
}

// Line 473: Split operation (1,2)
let word_1char = word_slice_chars[match_index].to_string();
if op.can_apply_to_source(word_1char.as_bytes()) {
    // ...
}
```

**Analysis**:
- Each `.to_string()` allocates on the heap
- Immediately converted back to `&[u8]` via `.as_bytes()`
- Allocations occur in inner loops with high iteration counts
- These strings are single-use and immediately dropped

**Proposed Solution 1a: Pre-compute UTF-8 Byte Slices**

Pre-compute UTF-8 byte representations once per method call:

```rust
// At method beginning (after line 275):
let word_slice_bytes: Vec<&[u8]> = word_slice_chars.iter()
    .map(|c| {
        // Create a small stack buffer for each character's UTF-8 encoding
        // Most chars are 1-4 bytes, stored inline without heap allocation
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        s.as_bytes()  // This borrows from buf, need different approach
    })
    .collect();
```

**Issue**: Lifetime problem - need to store the encoded bytes somewhere persistent.

**Proposed Solution 1b: Use char::encode_utf8() with Stack Buffer**

```rust
// For single character operations:
let mut word_buf = [0u8; 4];
let word_bytes = word_slice_chars[match_index].encode_utf8(&mut word_buf).as_bytes();

let mut input_buf = [0u8; 4];
let input_bytes = input_char.encode_utf8(&mut input_buf).as_bytes();

if op.can_apply(word_bytes, input_bytes) {
    // ...
}
```

**Benefits**:
- Zero heap allocations (stack buffers)
- UTF-8 encoding is fast (1-4 bytes)
- Borrows are valid within block scope

**Proposed Solution 1c: Pre-encode Input Character Once**

The input character is constant throughout the method:

```rust
// At method beginning (after line 273):
let mut input_char_buf = [0u8; 4];
let input_char_bytes = input_char.encode_utf8(&mut input_char_buf).as_bytes();
```

Then reuse `input_char_bytes` everywhere instead of `input_char.to_string().as_bytes()`.

**Expected Impact**:
- Eliminate 10-15 heap allocations per method call
- Reduce string allocation overhead from ~0.79% to near zero
- Estimated improvement: **0.5-1.0% overall speedup**

---

### Strategy 2: Optimize Operation Filtering (**MEDIUM PRIORITY**)

**Problem**: Operations are filtered multiple times with intermediate Vec allocations.

**Current Code**:
```rust
// Lines 285-361: Loop over ALL operations for standard ops
for op in operations.operations() {
    if op.consume_x() > 1 || op.consume_y() > 1 {
        continue;  // Skip multi-char ops
    }
    // Handle match, delete, insert, substitute
}

// Lines 365-367: Filter for transpose operations
let transpose_ops: Vec<_> = operations.operations().iter()
    .filter(|op| op.consume_x() == 2 && op.consume_y() == 2)
    .collect();

// Lines 421-445: Loop over ALL operations for merge
for op in operations.operations() {
    if op.consume_x() == 2 && op.consume_y() == 1 {
        // Handle merge
    }
}

// Lines 452-454: Filter for split operations
let split_ops: Vec<_> = operations.operations().iter()
    .filter(|op| op.consume_x() == 1 && op.consume_y() == 2)
    .collect();
```

**Analysis**:
- 4 separate iterations over the operation set
- 2 intermediate Vec allocations (transpose_ops, split_ops)
- Operations are checked repeatedly for same properties

**Proposed Solution 2a: Pre-categorize Operations at Automaton Construction**

Modify `OperationSet` to store operations by category:

```rust
// In src/transducer/operation_set.rs:
pub struct OperationSet {
    all_operations: Vec<OperationType>,

    // Pre-categorized for fast access:
    standard_ops: Vec<OperationType>,      // (1,1) operations
    transpose_ops: Vec<OperationType>,     // (2,2) operations
    merge_ops: Vec<OperationType>,         // (2,1) operations
    split_ops: Vec<OperationType>,         // (1,2) operations
}
```

Then in `successors_i_type`:

```rust
// No filtering needed!
for op in operations.standard_ops() {
    // Handle match, delete, insert, substitute
}

if !operations.transpose_ops().is_empty() && errors < self.max_distance {
    // Handle transpose (no Vec allocation)
}

for op in operations.merge_ops() {
    // Handle merge
}

if !operations.split_ops().is_empty() && can_enter_split {
    // Handle split (no Vec allocation)
}
```

**Benefits**:
- Zero intermediate allocations
- Single pass construction at automaton creation (amortized O(1))
- Cleaner code, easier to maintain
- Direct iteration over relevant operations only

**Effort**: Medium (requires OperationSet refactoring)

**Expected Impact**: **0.2-0.5% overall speedup** (eliminate filtering overhead)

**Proposed Solution 2b: Use Iterator Chains (Lower Effort Alternative)**

If OperationSet refactoring is too invasive, use iterator chains without collecting:

```rust
// Lines 365-367: Don't collect, just check existence
let has_transpose = operations.operations().iter()
    .any(|op| op.consume_x() == 2 && op.consume_y() == 2);

if has_transpose && errors < self.max_distance {
    for op in operations.operations().iter()
        .filter(|op| op.consume_x() == 2 && op.consume_y() == 2)
    {
        // Handle transpose
    }
}
```

**Benefits**: No Vec allocation, lower refactoring effort

**Tradeoff**: Still multiple iterations, less optimal than 2a

**Expected Impact**: **0.1-0.3% overall speedup**

---

### Strategy 3: SmallVec for Successors (**LOW PRIORITY**)

**Problem**: `successors` Vec allocates on heap even for small result sets.

**Current Code**:
```rust
fn successors_i_type(...) -> Vec<GeneralizedPosition> {
    let mut successors = Vec::new();
    // ...
    successors
}
```

**Analysis**:
- Most transitions generate 1-4 successors (exact match=1, typical=2-3)
- Vec::new() always uses heap allocation after first push
- SmallVec could store 4-8 elements inline

**Proposed Solution**:

```rust
use smallvec::{SmallVec, smallvec};

fn successors_i_type(...) -> SmallVec<[GeneralizedPosition; 8]> {
    let mut successors = SmallVec::new();
    // ... rest unchanged
    successors
}
```

**Benefits**:
- Inline storage for typical cases (no heap allocation)
- Zero-cost abstraction when exceeding inline capacity
- Reduces pressure on allocator

**Effort**: Low (type change + update call sites)

**Expected Impact**: **0.2-0.4% overall speedup** (reduce allocation overhead)

**Note**: SmallVec already imported in the codebase, so dependency exists.

---

### Strategy 4: Combine Operation Loops (EXPERIMENTAL)

**Problem**: Multiple sequential passes over operations could be merged.

**Current Structure**:
1. Loop for standard ops (match, delete, insert, substitute)
2. Handle transpose ops
3. Loop for merge ops
4. Handle split ops
5. Skip-to-match optimization

**Proposed Solution**:

Single loop with operation categorization:

```rust
for op in operations.operations() {
    match (op.consume_x(), op.consume_y()) {
        (1, 1) if op.is_match() => {
            // Handle match
        }
        (1, 1) if op.is_substitution() => {
            // Handle substitute
        }
        (1, 0) => {
            // Handle delete
        }
        (0, 1) => {
            // Handle insert
        }
        (2, 2) => {
            // Handle transpose
        }
        (2, 1) => {
            // Handle merge
        }
        (1, 2) => {
            // Handle split
        }
        _ => {}
    }
}
```

**Tradeoff Analysis**:
- **Pro**: Single iteration over operations
- **Con**: More complex control flow
- **Con**: May reduce inlining opportunities
- **Con**: Match expression overhead

**Risk**: Might not improve performance due to branch prediction overhead

**Recommendation**: **DEFER** - Try Strategies 1-3 first, then profile again

---

## Implementation Phases

### Phase 1: String Allocation Elimination (Strategy 1)

**Steps**:
1. Pre-encode `input_char` once at method beginning
2. Replace all `input_char.to_string()` with `input_char_bytes`
3. Replace all `word_char.to_string()` with `encode_utf8()` stack buffer
4. Verify correctness with test suite (725 tests)
5. Benchmark and measure improvement

**Expected Time**: 2-3 hours
**Expected Improvement**: 0.5-1.0% speedup
**Risk**: Low (straightforward refactoring)

### Phase 2: Operation Filtering (Strategy 2a or 2b)

**Steps**:
1. **Option A** (High effort, high reward): Refactor OperationSet
   - Add pre-categorized operation vectors
   - Update constructor to populate categories
   - Update all call sites in successors methods
   - Benchmark improvement

2. **Option B** (Low effort, medium reward): Iterator chains
   - Replace `collect()` with iterator chains
   - Test for correctness
   - Benchmark improvement

**Expected Time**:
- Option A: 4-6 hours
- Option B: 1-2 hours

**Expected Improvement**:
- Option A: 0.2-0.5% speedup
- Option B: 0.1-0.3% speedup

**Risk**:
- Option A: Medium (broader refactoring)
- Option B: Low (minimal changes)

**Recommendation**: Start with Option B, upgrade to Option A if profiling shows filtering overhead is significant

### Phase 3: SmallVec Optimization (Strategy 3)

**Steps**:
1. Change return type to `SmallVec<[GeneralizedPosition; 8]>`
2. Update all call sites in:
   - `successors_i_type()`
   - `successors_m_type()`
   - `successors_i_transposing()`
   - `successors_m_transposing()`
   - `successors_i_splitting()`
   - `successors_m_splitting()`
3. Benchmark to find optimal inline capacity (4, 8, 16?)
4. Verify correctness (725 tests)
5. Measure improvement

**Expected Time**: 2-3 hours
**Expected Improvement**: 0.2-0.4% speedup
**Risk**: Low (type change only)

---

## Total Expected Improvement

**Conservative Estimate**: 0.8-1.5% overall speedup
**Optimistic Estimate**: 1.2-1.9% overall speedup

**Combined with H2**:
- H2: 28-31% speedup ✅ Complete
- H1: 0.8-1.9% additional speedup
- **Total**: ~30-33% cumulative speedup

---

## Success Criteria

1. **Performance**: Achieve at least 1% additional speedup over H2 baseline
2. **Correctness**: All 725 tests passing
3. **Code Quality**: No increase in complexity, maintain readability
4. **No Regressions**: Distance 0-3 benchmarks all improve or stay within noise

---

## Open Questions

1. **OperationSet Refactoring**: Is pre-categorization worth the effort?
   - **Answer via**: Implement Strategy 2b first, profile, decide if 2a needed

2. **SmallVec Capacity**: What's the optimal inline capacity?
   - **Answer via**: Benchmark with capacities 4, 8, 16 and measure

3. **Post-H2 Hotspots**: Are there new bottlenecks after H2?
   - **Answer via**: Use the retained H1/H2 result reports; a post-H2
     flamegraph is outside this report's measured artifact set.

4. **can_apply() Internals**: Can we optimize the operation matching logic?
   - **Answer via**: Profile `OperationType::can_apply` after Strategy 1 implemented

---

## Next Steps

1. **Review this plan** with stakeholder
2. **Begin Phase 1**: String allocation elimination (highest impact)
3. **Benchmark Phase 1** and validate improvement
4. **Decide on Phase 2**: Strategy 2a vs 2b based on profiling data
5. **Implement Phase 3**: SmallVec if Phases 1-2 show promise
6. **Document results** in H1_RESULTS.md

---

## References

- **Baseline Profiling**: `docs/optimization/PHASE1_FLAMEGRAPH_ANALYSIS.md`
- **H2 Optimization**: `docs/optimization/H2_RESULTS.md` (28-31% speedup achieved)
- **Target Code**: `src/transducer/generalized/state.rs:260-569` (`successors_i_type`)
- **Operation Types**: `src/transducer/operation_type.rs`
- **Operation Set**: `src/transducer/operation_set.rs`
