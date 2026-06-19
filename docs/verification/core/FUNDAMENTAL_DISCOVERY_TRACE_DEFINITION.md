# Fundamental Discovery: is_valid_trace Allows Duplicate Pairs

**Date**: 2025-11-22
**Discovery**: Deep investigation revealed is_valid_trace definition permits duplicate pairs
**Impact**: Blocks NoDup-based proofs, requires alternative approaches

## The Discovery

### What We Found

The `is_valid_trace` definition in Distance.v does NOT prevent duplicate pairs from appearing in traces.

**Root Cause**: `compatible_pairs(p, p) = true`

```coq
Definition compatible_pairs (p1 p2 : nat * nat) : bool :=
  let '(i1, j1) := p1 in
  let '(i2, j2) := p2 in
  if (i1 =? i2) && (j1 =? j2) then true  (* same pair - ALLOWS duplicates! *)
  else if (i1 =? i2) || (j1 =? j2) then false
  else if i1 <? i2 then j1 <? j2 else j2 <? j1.
```

**Consequence**: `is_valid_trace_aux` checks `forallb (compatible_pairs head) tail`, which passes even if `head` appears again in `tail`.

### Concrete Example

A trace like `T = [(1,2), (3,4), (1,2), (5,6)]` would pass `is_valid_trace`:
- First `(1,2)` checks compatibility with `[(3,4), (1,2), (5,6)]`
  - `compatible_pairs (1,2) (3,4) = true` ✓
  - `compatible_pairs (1,2) (1,2) = true` ✓ (allows duplicate!)
  - `compatible_pairs (1,2) (5,6) = true` ✓
- Recursive check continues...

### Attempted Proof

Attempted to prove:
```coq
Lemma is_valid_trace_aux_NoDup :
  forall T, is_valid_trace_aux T = true -> NoDup T.
```

**Result**: UNPROVABLE with current definition (lines 1972-2037 in Distance.v)

The proof reaches a point where we need to derive a contradiction from:
- `(i,j) ∈ tail`
- `compatible_pairs (i,j) (i,j) = true`

But this is consistent! No contradiction exists.

## Impact Analysis

### Blocked Lemmas

**Direct impact**:
1. `is_valid_trace_aux_NoDup` - UNPROVABLE (fundamental blocker)
2. `touched_in_A_NoDup` - BLOCKED (depends on #1)
3. `touched_in_B_NoDup` - BLOCKED (depends on #1)

**Indirect impact**:
4. `incl_length_correct` - Uses NoDup hypotheses
5. `trace_composition_cost_bound` Part 2 - Original approach used NoDup + incl

### Why This Matters for Triangle Inequality

The triangle inequality proof (`lev_distance_triangle_inequality`) uses:
- `distance_equals_min_trace_cost` (optimal trace exists)
- `trace_composition_cost_bound` (composition cost bound)

Part 2 of `trace_composition_cost_bound` was planned to use:
```
touched_in_A A C comp ⊆ touched_in_A A B T1  (proven)
touched_in_B A C comp ⊆ touched_in_B B C T2  (proven)
+ NoDup on touched lists
→ length bounds
→ deletion/insertion cost arithmetic with 2|B| slack
```

**Without NoDup**: The `incl` → `length` implication fails.

## Resolution Paths

### Option 1: Strengthen is_valid_trace Definition ⭐ RECOMMENDED

**Change**:
```coq
Definition is_valid_trace (A B : list Char) (T : Trace A B) : bool :=
  (forallb (valid_pair (length A) (length B)) T) &&
  is_valid_trace_aux T &&
  nodupb pair_eq_dec T.  (* ADD THIS *)
```

**Pros**:
- Makes mathematical intent explicit
- Unblocks all NoDup lemmas immediately
- Clean, principled solution

**Cons**:
- Requires proving DP algorithm generates unique pairs
- May need to update existing proofs using is_valid_trace

**Estimated effort**:
- Definition change: 5 minutes
- Update existing proofs: 1-2 hours
- Prove DP generates NoDup traces: 3-5 hours

### Option 2: Semantic Proof (DP Never Generates Duplicates)

**Approach**: Prove as lemma rather than definition:
```coq
Lemma dp_trace_has_nodup :
  forall A B T,
    extracted_from_dp_matrix A B T ->
    is_valid_trace A B T = true ->
    NoDup T.
```

**Pros**:
- No definition changes
- Captures semantic property

**Cons**:
- Requires formalizing DP extraction
- May be harder to use in proofs

**Estimated effort**: 5-10 hours (DP formalization)

### Option 3: Alternative Approach WITHOUT NoDup ⚡ PRAGMATIC

**Insight**: Part 2 arithmetic might not actually NEED NoDup!

**Alternative bound for incl without NoDup**:
```coq
Lemma incl_length_with_multiplicity :
  forall {A} (l1 l2 : list A),
    incl l1 l2 ->
    length l1 <= length l2 * length l2.
```

Or use direct counting:
```coq
Lemma touched_length_bound_by_trace :
  forall A B T,
    length (touched_in_A A B T) <= length T.
```
*(Already proven at line 1930!)*

**New Part 2 approach**:
- Use `length (touched_in_A A C comp) <= length comp` (proven)
- Use `length comp <= length T1 * length T2` (composition size bound)
- Different arithmetic without incl length bound

**Pros**:
- Works with current definitions
- No refactoring needed

**Cons**:
- Weaker bounds (quadratic vs linear)
- May not give tight enough inequality

**Estimated effort**: 2-4 hours (exploratory arithmetic)

### Option 4: Axiomatize and Move On

**Approach**: Accept current admits, document clearly, continue with other work

**Pros**:
- Immediate progress on other goals
- Scientific value in documentation

**Cons**:
- Leaves verification incomplete

## Recommendations

### Immediate (Next Session):

**For production/completeness**: Option 1 (strengthen definition)
- Most principled
- Unblocks everything
- Clear path forward

**For exploration/research**: Option 3 (alternative counting)
- Tests if NoDup is actually needed
- May discover simpler proofs
- Lower risk

### Strategic:

1. **Week 1**: Try Option 3 (2-4 hours)
   - If successful: Triangle inequality complete without definition changes!
   - If blocked: Clear evidence NoDup is truly needed

2. **Week 2**: If Option 3 fails, implement Option 1
   - Strengthen definition (30 min)
   - Update proofs (2-3 hours)
   - Prove DP generates NoDup (3-5 hours)

3. **Week 3+**: Complete DP extraction (Option 2 bonus)
   - Formalize trace extraction
   - Prove optimality
   - Full end-to-end verification

## Current Status

**Commits**: Fundamental discovery documented in code (lines 2017-2046)

**Admits remaining**:
- 3 NoDup-related (blocked by this issue)
- 2 fold_left witness (separate hard problem)
- 2 DP extraction (architectural, can defer)

**Triangle inequality**: Provable modulo 4-5 well-documented admits

## Scientific Value

This discovery exemplifies the scientific method in formal verification:

1. **Hypothesis**: "NoDup should be provable from is_valid_trace"
2. **Experiment**: Attempted proof
3. **Discovery**: Definition allows duplicates
4. **Analysis**: Identified exact cause (compatible_pairs)
5. **Solutions**: Enumerated viable paths forward

**Documentation** of failed proof attempts is as valuable as successful proofs:
it guides later proof sessions and prevents repeated dead ends.

## References

- Distance.v lines 1972-2046: Attempted proof with detailed analysis
- Distance.v lines 800-820: compatible_pairs and is_valid_trace definitions
- PHASE5_COMPLETION_ATTEMPT.md: Initial investigation
- This document: Comprehensive resolution strategies
