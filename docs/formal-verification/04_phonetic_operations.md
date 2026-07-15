# Phase 4: Phonetic Operations - Formal Specification

**Status**: ✅ Formal model and Rust split integration complete; all active phonetic operation proofs verified
**Date**: November 17, 2025 (refreshed 2026-06-19)
**Files**: `rocq/liblevenshtein/PhoneticOperations.v`

## Executive Summary

Phase 4 establishes the formal semantics of phonetic split operations through rigorous proof-driven design. Unlike Phase 3's bug fixes, Phase 4 takes a top-down approach: define the formal model first, prove invariant preservation, then derive the Rust implementation from the proven specification.

**Key Achievement**: Discovered critical preconditions through proof attempts that prevent invalid transitions, ensuring phonetic splits maintain the same mathematical rigor as standard Levenshtein operations.

## Motivation

The existing phonetic operation implementation exhibited "buggy and hackish" behavior with multiple regression tests exposing edge-case gaps. Rather than iterative bug fixes, we adopted formal verification to derive correct-by-construction semantics.

**User Directive**: "I don't want to bounce around between a bunch of one-off attempts to fix these anymore, can we skip straight to the formal verification step to derive how they should be implemented?"

## Phonetic Splits as Primitive Operations

### Why Not Decomposable?

Phonetic splits (e.g., f→ph, k→ch, t→th) are **primitive multi-character transformations**, similar to skip-to-match. They cannot be decomposed into standard operations:

- **NOT** "delete 'f', insert 'p', insert 'h'"
  - Would cost 3 operations
  - Would create 3 intermediate positions
  - Wrong semantics: deletes then inserts, doesn't recognize equivalence

- **IS** "recognize 'f' and 'ph' as phonetically equivalent"
  - Costs 0 or fractional (typically 0.15, truncates to 0)
  - Single atomic transformation
  - Preserves phonetic equivalence relationship

### Three-Phase Lifecycle

Phonetic splits are multi-step operations with distinct phases:

```
1. ENTRY: I-type → ISplitting
   - Apply cost (fractional, truncates to 0)
   - Decrement offset by 1 (prepare for multi-char match)
   - Record entry character for validation

2. PROGRESS: ISplitting → ISplitting
   - Consume additional input characters
   - Validate pattern match (e.g., 'f' + 'ph')
   - No cost, stable position

3. COMPLETION: ISplitting → I-type or M-type
   - Increment offset by 1 (net effect: same as entry)
   - Errors unchanged (cost applied at entry)
   - Transition based on word boundary
```

**Example**: "graf" → "graphe" with f→ph split

```
Start: I+0#0 (processed 'gra')

Step 1 (Entry): word[3]='f', input='p'
  Check: Can apply f→ph split
  → ISplitting+(-1)#0_f (cost 0, offset decremented)

Step 2 (Progress): ISplitting+(-1)#0_f, input='h'
  Check: 'f' + 'ph' matches pattern
  → ISplitting+(-1)#0_f (consuming 'h')

Step 3 (Completion): Complete split
  → I+0#0 (offset -1+1=0, errors same)
  → M+0#0 (past word end, converts to M-type)

Step 4 (Standard): M+0#0, input='e'
  → INSERT (cost 1)
  → M+1#1

Total cost: 0 (split) + 1 (insert) = 1 error
```

## Formal Model

### Type Signatures

```coq
(* Entry: I-type → ISplitting *)
Inductive i_split_entry : Position -> ascii -> nat -> Position -> Prop

(* Progress: ISplitting → ISplitting *)
Inductive i_split_progress : Position -> ascii -> Position -> Prop

(* Completion: ISplitting → I-type or M-type *)
Inductive i_split_completion : Position -> nat -> Position -> Prop

(* Combined: Entry → Completion *)
Inductive i_phonetic_split : Position -> ascii -> nat -> Position -> Prop
```

### Critical Preconditions Discovered Through Proofs

The proof process revealed two essential preconditions not obvious from informal specifications:

#### 1. Offset Lower Bound: `offset > -n`

**Why Needed**: Entry decrements offset by 1. Without this bound:
- offset = -n → offset - 1 = -n - 1 (OUT OF BOUNDS)
- Violates I-splitting invariant: $`-n \le  \text{offset} \le  n`$

**Discovery**: Proof attempt failed with goal:
```coq
offset - 1 >= -Z.of_nat n
```
When precondition only had `offset >= -n`, equality case violated the goal.

**Fix**: Require strict inequality `offset > -n` in entry precondition.

#### 2. Fractional Cost Budget: `split_cost = 0 → |offset| < errors`

**Why Needed**: For fractional costs (0.15 truncates to 0), need reachability:
- Require: $`|\text{offset} - 1| \le  \text{errors} + \text{cost}`$
- For cost = 0, offset = 0, errors = 0:
  - $`|-1| = 1 \le  0`$ (FAILS)

**Discovery**: Proof attempt with case analysis on cost:
```coq
destruct cost as [| cost'].
+ (* cost = 0: exposes the impossible goal |-1| ≤ 0 *)
+ (* cost ≥ 1: proof succeeds with |offset - 1| ≤ |offset| + 1 ≤ errors + cost *)
```

**Fix**: For fractional costs (cost=0), require strict inequality `|offset| < errors`.

**Insight**: This creates a "budget requirement" for fractional operations. You can only apply fractional-cost operations when you have spare reachability budget. This elegantly captures the intuition that "free" operations still require positional flexibility.

### Complete Entry Relation

```coq
Inductive i_split_entry : Position -> ascii -> nat -> Position -> Prop :=
  | ISplitEntry : forall offset errors n entry_char split_cost,
      (* Standard I-type preconditions *)
      (-Z.of_nat n <= offset <= Z.of_nat n) ->
      (Z.abs offset <= Z.of_nat errors) ->
      (errors <= n)%nat ->
      (errors + split_cost <= n)%nat ->

      (* CRITICAL: Discovered through proof attempts *)
      (offset > -Z.of_nat n) ->  (* Ensures offset - 1 ≥ -n *)
      (split_cost = 0%nat -> Z.abs offset < Z.of_nat errors) ->  (* Budget for fractional costs *)

      i_split_entry
        (mkPosition VarINonFinal offset errors n None)
        entry_char
        split_cost
        (mkPosition VarISplitting (offset - 1) (errors + split_cost) n (Some entry_char)).
```

### Completion Relation Design

The completion relation has preconditions that ensure invariants are trivially satisfied:

```coq
Inductive i_split_completion : Position -> nat -> Position -> Prop :=
  | ISplitComplete_ToI : forall offset errors n entry_char result_offset,
      result_offset = offset + 1 ->
      (* Precondition: Result must satisfy I-invariant *)
      (Z.abs result_offset <= Z.of_nat errors) ->
      (-Z.of_nat n <= result_offset <= Z.of_nat n) ->
      i_split_completion
        (mkPosition VarISplitting offset errors n (Some entry_char))
        n
        (mkPosition VarINonFinal result_offset errors n None)

  | ISplitComplete_ToM : forall offset errors n entry_char result_offset m_offset,
      result_offset = offset + 1 ->
      m_offset = result_offset - Z.of_nat n ->
      (* Precondition: Result must satisfy M-invariant *)
      (-Z.of_nat (2 * n) <= m_offset <= 0) ->
      (Z.of_nat errors >= -m_offset - Z.of_nat n) ->
      i_split_completion
        (mkPosition VarISplitting offset errors n (Some entry_char))
        n
        (mkPosition VarMFinal m_offset errors n None).
```

**Design Pattern**: By requiring the result position to satisfy the target invariant as a precondition, the completion proof becomes trivial (just `assumption`). This pushes complexity to the caller (entry + splitting state invariant).

## Proven Theorems

All three theorems proven with `Qed` (no `Admitted`):

### Theorem 1: Entry Preserves Invariant

```coq
Theorem i_split_entry_preserves_invariant : forall p entry_char cost p',
  i_invariant p ->
  i_split_entry p entry_char cost p' ->
  i_splitting_invariant p'.
```

**Proof Strategy**: Case analysis on cost (0 vs $`\ge 1)`$
- **cost = 0**: Use fractional budget precondition `|offset| < errors`
  - Need: $`|\text{offset} - 1| \le  \text{errors}`$
  - Have: `|offset| < errors`
  - Conclude: $`|\text{offset} - 1| \le  |\text{offset}| + 1 < \text{errors} + 1`$, so $`|\text{offset} - 1| \le  \text{errors}`$ ✓
- **cost $`\ge  1`$**: Standard arithmetic
  - $`|\text{offset} - 1| \le  |\text{offset}| + 1 \le  \text{errors} + 1 \le  \text{errors} + \text{cost}`$ ✓

**Line Reference**: PhoneticOperations.v:220-248

### Theorem 2: Completion Preserves Invariant

```coq
Theorem i_split_completion_preserves_invariant : forall p n p',
  i_splitting_invariant p ->
  i_split_completion p n p' ->
  (i_invariant p' \/ m_invariant p').
```

**Proof Strategy**: Trivial by preconditions
- Destruct on completion constructor (ToI or ToM)
- Preconditions directly assert result satisfies invariant
- Just `assumption` for each goal ✓

**Line Reference**: PhoneticOperations.v:251-290

### Theorem 3: Phonetic Split Preserves Invariant

```coq
Theorem i_phonetic_split_preserves_invariant : forall p entry_char cost p',
  i_invariant p ->
  i_phonetic_split p entry_char cost p' ->
  (i_invariant p' \/ m_invariant p').
```

**Proof Strategy**: Composition
- Invert phonetic split to get entry and completion steps
- Apply Theorem 1: entry preserves invariant → splitting invariant
- Apply Theorem 2: completion preserves invariant → final invariant ✓

**Line Reference**: PhoneticOperations.v:293-305

## M-Type Phonetic Splits

M-type splits are simpler because M-type is already past the word end:

```coq
Inductive m_split_entry : Position -> ascii -> nat -> Position -> Prop :=
  | MSplitEntry : forall offset errors n entry_char split_cost,
      (* M-type preconditions *)
      (-Z.of_nat (2 * n) <= offset <= 0) ->
      (Z.of_nat errors >= -offset - Z.of_nat n) ->
      (errors <= n)%nat ->
      (errors + split_cost <= n)%nat ->

      m_split_entry
        (mkPosition VarMFinal offset errors n None)
        entry_char
        split_cost
        (mkPosition VarMSplitting (offset - 1) (errors + split_cost) n (Some entry_char)).

Inductive m_split_completion : Position -> Position -> Prop :=
  | MSplitComplete : forall offset errors n entry_char result_offset,
      result_offset = offset + 1 ->
      m_split_completion
        (mkPosition VarMSplitting offset errors n (Some entry_char))
        (mkPosition VarMFinal result_offset errors n None).

Inductive m_phonetic_split : Position -> ascii -> nat -> Position -> Prop :=
  | MPhoneticSplit : forall p1 p2 p3 entry_char split_cost,
      m_split_entry p1 entry_char split_cost p2 ->
      m_split_completion p2 p3 ->
      m_phonetic_split p1 entry_char split_cost p3.
```

**Note**: M-type doesn't need the critical preconditions because:
- Offset already bounded: $`\text{offset} \le  0`$, so `offset > -2n` is always satisfied
- Reachability uses different formula: $`\text{errors} \ge  -\text{offset} - n`$ (looser than $`|\text{offset}| \le  \text{errors}`$)

## Invariant Definitions

```coq
Definition i_splitting_invariant (p : Position) : Prop :=
  variant p = VarISplitting /\
  let n := max_distance p in
  let offset := offset p in
  let errors := errors p in
  Z.abs offset <= Z.of_nat errors /\
  -Z.of_nat n <= offset <= Z.of_nat n /\
  (errors <= n)%nat.

Definition m_splitting_invariant (p : Position) : Prop :=
  variant p = VarMSplitting /\
  let n := max_distance p in
  let offset := offset p in
  let errors := errors p in
  Z.of_nat errors >= -offset - Z.of_nat n /\
  -Z.of_nat (2 * n) <= offset <= 0 /\
  (errors <= n)%nat.
```

**Pattern**: Splitting invariants match their corresponding final invariants (I-splitting matches I-type, M-splitting matches M-type), differing only in variant tag.

## Lessons Learned

### 1. Proof-Driven Precondition Discovery

**Anti-pattern**: Guess preconditions from informal specs, then try to prove.

**Best practice**: Start with minimal preconditions, let proof attempts reveal necessary constraints.

**Example**:
```coq
(* Initial attempt (WRONG) *)
offset >= -Z.of_nat n  (* Allows offset = -n *)

(* Proof fails with goal: offset - 1 >= -n *)
(* When offset = -n: -n - 1 >= -n (FALSE) *)

(* Corrected (RIGHT) *)
offset > -Z.of_nat n  (* Strict inequality *)
```

### 2. Case Analysis on Cost Reveals Edge Cases

The $`\text{cost}=0`$ case has fundamentally different requirements than $`\text{cost} \ge 1`$:
- **cost $`\ge  1`$**: Budget naturally covers offset adjustment
- **cost = 0**: Need extra positional flexibility (strict inequality)

This pattern likely applies to other fractional-cost operations.

### 3. Preconditions in Relations Make Proofs Trivial

By encoding invariant requirements as preconditions in the completion relation:
```coq
(Z.abs result_offset <= Z.of_nat errors) ->  (* Part of constructor *)
```

The invariant preservation proof reduces to:
```coq
Proof.
  intros. inversion H. assumption.  (* Trivial! *)
Qed.
```

**Trade-off**: Pushes complexity to the caller, but makes compositional reasoning easier.

### 4. Variants as State Machine States

The Position variant field acts as a state machine:
- `VarINonFinal`: Normal I-type position
- `VarISplitting`: Mid-phonetic-split I-type
- `VarMFinal`: Normal M-type position
- `VarMSplitting`: Mid-phonetic-split M-type

Transitions:
- Standard ops: Within same variant
- Phonetic splits: Cross variant boundaries (I ↔ ISplitting, M ↔ MSplitting)
- Skip-to-match: I → M (across type boundary)

### 5. Formalization Clarifies Informal Specs

The three-phase lifecycle wasn't clear from Rust code. Formal modeling revealed:
- Entry and completion are distinct operations
- Progress is a separate relation (currently identity, but extensible)
- Net effect: offset unchanged, errors increased by cost
- Intermediate state: offset temporarily decremented

This structure provides hooks for future extensions (multi-step patterns, validation).

## Proven Composition Properties

The original Phase 4 proof obligation set covered single split lifecycle invariants.
The refreshed `PhoneticOperations.v` also proves composition and cost accounting
properties needed for consecutive splits and split-plus-standard-operation paths:

```coq
Theorem i_phonetic_split_composes_with_i_successor : ...
Theorem m_phonetic_split_composes_with_m_successor : ...
Theorem consecutive_i_phonetic_splits_preserve_invariant : ...
Theorem consecutive_m_phonetic_splits_preserve_invariant : ...
Theorem i_phonetic_split_cost_correct : ...
Theorem m_phonetic_split_cost_correct : ...
Theorem i_phonetic_split_then_i_successor_cost_correct : ...
Theorem m_phonetic_split_then_m_successor_cost_correct : ...
```

**Resolved questions**:
1. A split followed by a standard successor preserves the target invariant when
   the split target satisfies the standard successor preconditions.
2. Consecutive I-type and M-type splits preserve invariants.
3. Split cost accounting is additive across split-plus-standard-successor paths.

**Remaining modeling questions**:
1. Can two splits overlap? (e.g., "fph" → split 'f'→'ph', then split 'ph'→?)
2. Does split order matter?
3. What are the tight cost bounds for arbitrary-length overlapping split chains?

### Progress Relation Extension

Current definition is identity:
```coq
Inductive i_split_progress : Position -> ascii -> Position -> Prop :=
  | ISplitProgress : forall offset errors n entry_char input_char,
      i_split_progress
        (mkPosition VarISplitting offset errors n (Some entry_char))
        input_char
        (mkPosition VarISplitting offset errors n (Some entry_char)).
```

**Future**: Add character sequence validation:
- Check that accumulated input matches target pattern
- Maintain partial match state
- Reject invalid sequences early

### Multi-Character Entry Patterns

Current model: entry_char is single character ('f').

**Extension**: Support multi-character entry patterns ('ch', 'ph', 'th'):
- Entry consumes multiple word characters
- Offset adjustment proportional to pattern length
- Preconditions must account for longer decrements

### Fractional Cost Semantics

Current: `split_cost = 0` (0.15 truncates to 0).

**Questions**:
1. Should we model fractional costs explicitly in Coq?
2. Use Q (rationals) instead of nat for costs?
3. How does truncation interact with composition?

## Implementation Status

The formal relations have corresponding Rust transition paths in
`src/transducer/generalized/{position,state}.rs`:

1. **Translated Relations**
   - `i_split_entry` and `m_split_entry` map to split-entry constructors and
     generalized successor generation.
   - `i_split_progress` and `m_split_progress` map to splitting-state successor
     handling.
   - `i_split_completion` and `m_split_completion` map to split-completion
     transitions.

2. **Precondition Checks**
   - Entry guards enforce offset bounds and distance-budget constraints.
   - Splitting constructors enforce relaxed intermediate invariants.
   - Completion restores standard I-type or M-type invariants.

3. **Regression Coverage**
   - Focused split tests cover single splits, consecutive splits, split plus
     standard operation, and distance constraints.
   - `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env CARGO_BUILD_JOBS=1 cargo test -j1 --lib test_phonetic_split -- --test-threads=1`
     passed 7/7 focused phonetic split tests on 2026-06-19.

4. **Remaining Evaluation**
   - Property-based split lifecycle tests would still be useful as randomized
     empirical checks of the Rocq invariants.
   - Performance benchmarks should measure the split-entry and split-completion
     paths against standard-operation hot paths before further specialization.

## Files Modified

- ✅ `rocq/liblevenshtein/PhoneticOperations.v`: 483 lines, all active phonetic operation proofs with `Qed`
- ✅ `rocq/liblevenshtein/_CoqProject`: Added PhoneticOperations.v to build
- ✅ `docs/formal-verification/04_phonetic_operations.md`: This document

## Verification Status

```bash
$ grep -n "Admitted\|admit" PhoneticOperations.v
✅ No admits found

$ make PhoneticOperations.vo
make: 'PhoneticOperations.vo' is up to date.
```

All active phonetic operation theorems are proven, no admitted lemmas remain in
`PhoneticOperations.v`, and compilation succeeds through the Rocq build.

## Next Evaluation Steps

1. ✅ Complete formal model (DONE)
2. ✅ Prove invariant preservation (DONE)
3. ✅ Implement Rust code from formal model (DONE)
4. ✅ Fix focused split regressions: consecutive splits and split plus standard operation (DONE)
5. Add randomized property tests for split lifecycle invariants.
6. Benchmark split paths against standard-operation hot paths.
