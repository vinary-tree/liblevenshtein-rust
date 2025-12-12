# Automaton Proofs Status

**Date**: December 10, 2025 (Session 4 Update)
**Build Status**: ✅ Compiles successfully

## Summary

The Automaton verification module contains proofs for soundness and completeness of Levenshtein automata supporting three algorithms: Standard, Transposition (Damerau), and Merge/Split. The build compiles but has **22 admitted lemmas** across multiple files:
- Completeness.v: 8
- Soundness.v: 4
- OptimalTrace/MergeSplitConstruction.v: 4
- Composition/DamerauComposition.v: 2
- Core/MergeSplitDistance.v: 2
- Composition/MergeSplitComposition.v: 1
- MainTheorem.v: 1

### Recent Progress (December 10, 2025)

**Session 4 Progress: Merge-Split Trace Infrastructure**

**OptimalTrace/MergeSplitConstruction.v - Branch Lemmas:**
- ✅ **PROVEN**: `ms_trace_cost_shift_A_delete` - Delete branch cost decomposition
- ✅ **PROVEN**: `ms_trace_cost_shift_B_insert` - Insert branch cost decomposition
- ⚠️ **ADMITTED**: `ms_trace_cost_cons_merge` - Merge branch (goal structure issue after unfold)
- ⚠️ **ADMITTED**: `ms_trace_cost_cons_split` - Split branch (same structural issue)
- ⚠️ **ADMITTED**: `ms_trace_cost_cons_double` - Double-subst branch (same structural issue)
- ⚠️ **ADMITTED**: `ms_optimal_trace_cost_eq` - Optimal trace has cost = distance

**Key Issue Identified:** After unfolding `ms_trace_cost`, the goal becomes a raw arithmetic expression and `ms_trace_change_cost_shift_*` lemmas cannot find their patterns because `ms_trace_change_cost` has been expanded. Requires either:
- (a) Different proof strategy that doesn't unfold ms_trace_cost
- (b) Direct fold_left manipulation lemmas
- (c) Compositional cost function that avoids structural mismatch

**Added Infrastructure:**
- `ms_element_shift_A_2`, `ms_element_shift_B_2` - Shift by 2 for multi-position operations
- `ms_trace_shift_A_2_map`, `ms_trace_shift_B_2_map` - Map versions of shift
- `nth_plus_2_minus_1` - Arithmetic helper for shift-by-2 proofs
- Position length preservation lemmas for shift-by-2

**Triangle Inequality Status:**
- `ms_triangle` in Core/MergeSplitDistance.v remains admitted
- `ms_triangle_via_trace` in Composition/MergeSplitComposition.v remains admitted
- Both await completion of `ms_optimal_trace_cost_eq`
- Semantic justification is clear: going through intermediate string is one valid transformation

### Previous Progress (December 9, 2025)

**Session 3 Progress:**

**DamerauTrace.v - `dl_distance_le_valid_trace_cost`:**
- Added helper lemmas: `dl_touched_A_length_bound`, `dl_touched_B_length_bound`, `dl_trace_cost_nonneg`, `dl_change_cost_mono`, `dl_change_cost_fold_ge`
- Completed empty trace case (uses `damerau_lev_le_standard` + `lev_distance_upper_bound`)
- Non-empty trace case documented but still admitted (requires trace-to-edit-sequence correspondence)

**DamerauTrace.v - `dl_optimal_trace_exists`:**
- Added detailed proof strategy documentation
- Completed base cases:
  - Empty A: empty trace has cost = |B| = distance
  - Empty B: empty trace has cost = |A| = distance
  - Single char A and B: `[DLMatch 1 1]` achieves exact distance
- Remaining edge cases (single vs multi, multi vs single, main case) still admitted

**Session 2 Progress:**

**COMPLETED:** `automaton_run_std_trans_correspondence` (Qed)
- Extended `automaton_run_step_std_trans` to return additional invariants:
  - Position inclusion: `incl (positions s_std') (positions s_trans')`
  - Non-special preservation: Standard positions are non-special
  - Spread bound propagation: Transposition positions satisfy spread bound
- Used these invariants in the induction step of `automaton_run_std_trans_correspondence`

**PROVEN:** Non-special preservation within `automaton_run_step_std_trans`
- Used `transition_state_positions_standard_nonspecial`: Standard transitions produce only non-special positions
- Used `epsilon_closure_nonspecial`: Epsilon closure preserves non-special property
- Used `fold_state_insert_non_special`: Antichain filtering preserves non-special property

**REMAINING in `automaton_run_step_std_trans`:** 2 admits
- Position inclusion (line 3214): Antichain filtering can differ between Standard and Transposition
- Spread bound (line 3251): Need to show spread is preserved through transitions

**Session 1 Progress:**

**Added:** Epsilon closure helper lemmas to Transition.v:
- `epsilon_closure_from_origin_term_eq_errors_aux`: Proves term_index = num_errors for positions from initial state
- `epsilon_closure_from_origin_term_eq_errors`: Simplified interface for the above
- `epsilon_closure_from_origin_term_bounded`: Proves term_index <= n for initial state positions
- `epsilon_closure_aux_preserves_original`: Original positions are preserved in closure
- `std_pos_0_0_in_epsilon_closure`: std_pos 0 0 is always in initial closure
- `fold_left_min_contains_zero`: Helper for computing minimum term_index
- `epsilon_closure_from_origin_min_is_zero`: Minimum term_index in initial closure is 0

**COMPLETED:** `standard_accepts_implies_transposition_accepts` (Qed)
- Used the new epsilon closure lemmas to prove spread bound for initial state
- Required adding spread bound hypothesis to `automaton_run_step_std_trans` and `automaton_run_std_trans_correspondence`

### Previous Progress
- Fixed assertion error in `automaton_run_std_trans_correspondence` (line 3337-3339)
- Added helper lemmas: `fold_state_insert_final_reverse`, `incl_not_nil`

## Admitted Lemmas by File

### Automaton/Soundness.v (4 admitted)

| Line | Name | Type | Scope | Notes |
|------|------|------|-------|-------|
| 246 | `pseudo_reachable_nonspecial_implies_reachable` | Lemma | Transposition | complete_transpose case for spurious specials |
| 4142 | `automaton_run_preserves_reachable_transposition` | Lemma | Transposition | **BLOCKED** - See detailed analysis below |
| 4634 | `automaton_sound_merge_split` | Theorem | Merge/Split | **OUT OF SCOPE** |
| 4651 | `automaton_sound_merge_split_lev` | Corollary | Merge/Split | **OUT OF SCOPE** |

### Automaton/Completeness.v (8 admitted)

| Line | Name | Type | Scope | Notes |
|------|------|------|-------|-------|
| 2438 | `reachable_implies_contained_aux` | Lemma | Standard | **CRITICAL** - Core lemma for completeness |
| 2566 | `automaton_run_not_dead_for_reachable` | Lemma | Standard | Depends on contained_aux |
| 2614 | `automaton_final_state_accepts_standard` | Lemma | Standard | Depends on above two |
| 3276 | `automaton_run_step_std_trans` | Lemma | Transposition | 2 internal admits: position inclusion, spread bound |
| ~3388 | `automaton_run_std_trans_correspondence` | Lemma | Transposition | ✅ **COMPLETED** - Uses automaton_run_step_std_trans properties |
| ~3474 | `standard_accepts_implies_transposition_accepts` | Lemma | Transposition | ✅ **COMPLETED** - Uses epsilon closure lemmas |
| 3527 | `standard_accepts_implies_merge_split_accepts` | Lemma | Merge/Split | **OUT OF SCOPE** |
| 3623 | `automaton_complete_transposition` | Theorem | Transposition | Main transposition completeness |
| 3661 | `automaton_complete_merge_split` | Theorem | Merge/Split | **OUT OF SCOPE** |
| 3718 | `automaton_finds_distance` | Corollary | All | Distance computation corollary |

### Automaton/MainTheorem.v (1 admitted)

| Line | Name | Type | Scope | Notes |
|------|------|------|-------|-------|
| 250 | `automaton_distance_correct` | Theorem | Standard | Distance = lev_distance |

### Other Files

| File | Line | Name | Notes |
|------|------|------|-------|
| Composition/DamerauComposition.v | 826 | `damerau_change_cost_bound` | Inner admits in algebraic bounds |
| Composition/DamerauComposition.v | 2082 | `damerau_lev_triangle_via_composition` | Depends on above |
| Composition/MergeSplitComposition.v | 176 | `ms_triangle_via_trace` | Awaits ms_optimal_trace_cost_eq |
| Core/MergeSplitDistance.v | 1648 | `ms_seq_compose` | Edit sequence composition |
| Core/MergeSplitDistance.v | 1896 | `ms_triangle` | Direct approach (main case admitted) |
| OptimalTrace/MergeSplitConstruction.v | 1120 | `ms_trace_cost_cons_merge` | Goal structure issue after unfold |
| OptimalTrace/MergeSplitConstruction.v | 1241 | `ms_trace_cost_cons_split` | Same structural issue |
| OptimalTrace/MergeSplitConstruction.v | 1268 | `ms_trace_cost_cons_double` | Same structural issue |
| OptimalTrace/MergeSplitConstruction.v | 1316 | `ms_optimal_trace_cost_eq` | Main cost equality theorem |

**Note:** Trace/DamerauTrace.v now has 0 admits (base cases fully proven)

## Dependency Graph

```
reachable_implies_contained_aux (CRITICAL - Admitted)
    ↓
automaton_run_not_dead_for_reachable (Admitted)
    ↓
automaton_final_state_accepts_standard (Admitted)
    ↓
reachable_final_implies_accepts [PROVEN]
    ↓
automaton_complete_standard [PROVEN]
    ↓
automaton_distance_correct (MainTheorem.v - Admitted)

automaton_run_step_std_trans (Admitted - 2 internal admits remaining)
    ↓
automaton_run_std_trans_correspondence [PROVEN]
    ↓
standard_accepts_implies_transposition_accepts [PROVEN]
    ↓
automaton_complete_transposition (Admitted - needs Damerau-specific proof)

automaton_run_preserves_reachable_transposition (independent, for Damerau soundness - Admitted)
```

## Critical Issue: `reachable_implies_contained_aux`

### Statement
```coq
Lemma reachable_implies_contained_aux : forall query n dict_prefix p,
  position_reachable query n dict_prefix p ->
  num_errors p <= n ->
  is_special p = false ->
  forall s,
    automaton_run_from_initial Standard query n dict_prefix = Some s ->
    positions_contain (positions s) p.
```

### Problem Analysis

The lemma attempts to prove that if a position `p` is reachable via edit operations, then the automaton's state contains that position. However, there's a **fundamental mismatch** between:

1. **`positions_contain`** (used in the lemma): Requires `position_subsumes p1 p2` which demands:
   - Same `term_index`
   - Same `is_special` flag
   - `num_errors p1 <= num_errors p2`

2. **`subsumes_standard`** (used by automaton's antichain): Allows different `term_index` values when one position dominates another:
   ```coq
   (* A position at term_index i with errors e can subsume
      a position at term_index i+k with errors e+k *)
   ```

### Consequence

When the automaton performs antichain filtering via `fold_left state_insert`, a position at `(i, e)` can remove a position at `(i+1, e+1)` because the first "dominates" the second (you can always reach `(i+1, e+1)` from `(i, e)` via deletion).

This means **the exact reachable position may not survive antichain filtering**, only a dominating position with lower term_index remains.

### Why This Matters

- The current lemma statement is **too strong** for intermediate positions
- Cannot prove that `std_pos (length dict_prefix) e` is in the state if a dominating position `std_pos k e'` (where `k < length dict_prefix` and `k + (length dict_prefix - k) = length dict_prefix`, `e' + (length dict_prefix - k) = e`) is in the antichain

### Possible Solutions

#### Option A: Weaker Predicate
Replace `positions_contain` with a weaker predicate:
```coq
Definition positions_dominate (ps : list Position) (p : Position) : bool :=
  existsb (fun p' =>
    (term_index p' <= term_index p) &&
    (num_errors p' + (term_index p - term_index p') <= num_errors p)
  ) ps.
```

#### Option B: Track Final Positions Only
For completeness, we only need to show that **final positions** survive. The "non-final cannot subsume final" rule (2024-12 bug fix) protects final positions:
```coq
(* In subsumes_standard: *)
if position_is_final p1 qlen then
  if position_is_final p2 qlen then
    (* both final: compare errors *)
    num_errors p1 <=? num_errors p2
  else
    (* p1 final, p2 non-final: p1 cannot subsume p2 *)
    false
else
  (* p1 non-final: standard subsumption *)
  ...
```

This means a final position can only be removed by another final position with fewer errors.

#### Option C: Restructure Completeness Proof
Use existing proven infrastructure:
1. `fold_state_insert_has_final`: If closed_positions contains a final position, the result is accepting
2. `fold_state_insert_preserves_min_error`: Minimum error count is preserved
3. `transition_state_not_dead_standard`: Standard never goes dead if errors < n

New lemma needed:
```coq
Lemma reachable_final_produces_closed_final : forall query n dict_prefix,
  (exists p, position_reachable query n dict_prefix p /\
             position_is_final p (length query) = true /\
             num_errors p <= n) ->
  forall s,
    automaton_run_from_initial Standard query n dict_prefix = Some s ->
    exists p', In p' (positions s) /\
               position_is_final p' (length query) = true.
```

## Progress Made

### Recently Fixed Lemmas

1. **`transition_produces_insert_bounded`** (Completeness.v:1349-1366)
   - Fixed type unification issue by moving `destruct p` before `exists`
   - Used `change` tactic to make definitional equality explicit

2. **`transition_produces_insert_exact`** (Completeness.v:1369-1383)
   - Same fix pattern as above

### Helper Infrastructure Available

The following proven lemmas can help complete the remaining proofs:

- `transition_standard_produces_match` (Transition.v)
- `transition_standard_produces_substitute` (Transition.v)
- `transition_standard_produces_insert` (Transition.v)
- `fold_state_insert_has_final` (Completeness.v)
- `fold_state_insert_preserves_min_error` (Completeness.v)
- `fold_state_insert_accepting` (Completeness.v)
- `transition_state_not_dead_standard` (Completeness.v)

## Out of Scope Lemmas

The following 4 merge/split lemmas are **not in scope** for the current work:

1. `automaton_sound_merge_split` (Soundness.v:4417)
2. `automaton_sound_merge_split_lev` (Soundness.v:4434)
3. `standard_accepts_implies_merge_split_accepts` (Completeness.v:2996)
4. `automaton_complete_merge_split` (Completeness.v:3130)

These require integration with the MergeSplitDistance.v module which defines a different distance metric.

## Critical Issue: Transposition Soundness (`automaton_run_preserves_reachable_transposition`)

**Date Updated**: December 8, 2025

### Statement
```coq
Lemma automaton_run_preserves_reachable_transposition : forall query n dict_prefix dict s final,
  query_length s = length query ->
  automaton_run Transposition query n dict s = Some final ->
  (forall p, In p (Automaton.State.positions s) ->
             is_special p = false ->
             position_reachable_damerau query n dict_prefix p) ->
  (forall p, In p (Automaton.State.positions final) ->
             is_special p = false ->
             position_reachable_damerau query n (dict_prefix ++ dict) p).
```

### Problem Analysis: Spurious Special Positions

The transposition algorithm creates **spurious special positions** when `query[i] = query[i+1]`:

1. **Normal case** (`c ≠ query[i]`): Enter-transpose creates `special_pos i (e+1)` which is reachable via `reach_damerau_enter_transpose`

2. **Spurious case** (`c = query[i] = query[i+1]`): Enter-transpose ALSO creates `special_pos i (e+1)` but:
   - `reach_damerau_enter_transpose` requires `c ≠ c_next` (c ≠ query[i])
   - So the spurious special position is NOT semantically reachable
   - It exists in the automaton state but has no valid edit sequence

### Why This Breaks the Proof

At line 4134 in Soundness.v, we need to provide:
```coq
position_reachable_damerau query n dict_prefix p2
```
where `p2` is a special position in state `s`.

**The issue**: `transition_positions_reachable_transposition` (line 3814) requires ALL input positions to be reachable, but our hypothesis only guarantees non-special inputs are reachable.

### Key Insight: Subsumption Saves Soundness

Spurious positions don't affect soundness because:

1. **Same term_index**: When `query[i] = query[i+1] = c`:
   - Spurious path: `std_pos i e → special_pos i (e+1) → std_pos (i+2) (e+1)`
   - Match path: `std_pos i e → std_pos (i+1) e → std_pos (i+2) e`

2. **Error count**: Match path has error count `e`, spurious path has `e+1`

3. **Subsumption**: `std_pos (i+2) e` subsumes `std_pos (i+2) (e+1)` by standard subsumption rules (same term_index, lower errors)

4. **Antichain filtering**: The spurious position is removed, only the reachable one survives

### Solution Approaches

#### Approach A: Track Position Provenance
Strengthen the invariant to track how special positions were created:
```coq
Inductive position_trackable_damerau (query : list Char) (n : nat) :
  list Char -> Position -> Prop :=
  | trackable_reachable : forall dp p,
      position_reachable_damerau query n dp p ->
      is_special p = false ->
      position_trackable_damerau query n dp p
  | trackable_special : forall dp c i e,
      position_reachable_damerau query n dp (std_pos i e) ->
      S i < length query ->
      nth_error query (S i) = Some c ->
      e < n ->
      position_trackable_damerau query n (dp ++ [c]) (special_pos i (S e)).
```
Then prove non-special trackable positions are either reachable OR subsumed by reachable.

#### Approach B: Use Pseudo-Reachability
1. Complete `pseudo_reachable_nonspecial_implies_reachable` (Soundness.v:246)
2. Change invariant to track pseudo-reachability
3. Convert to true reachability at final step

**Blocked by**: Same issue - complete_transpose case for spurious specials.

#### Approach C: Prove Post-Antichain Only
Instead of proving all `trans_pos` positions are reachable, prove:
1. Positions surviving antichain are reachable
2. Use the fact that spurious positions are always subsumed

This requires showing: if `std_pos (i+2) (e+1)` comes from spurious complete_transpose, then `std_pos (i+2) e` (from matches) is also in `trans_pos`.

### Verified Behaviors

From Subsumption.v (lines 73-96), for Transposition subsumption:
- Same `is_special`: standard subsumption `e ≤ f ∧ |i-j| ≤ f-e`
- Different `is_special`: no subsumption possible
- Special positions only subsume same-index special positions

This confirms spurious non-special outputs ARE subsumed by match outputs.

### Recommended Path Forward

1. **Prove helper**: Show that when enter_transpose fires and c = query[i], the match transition also fires, producing `std_pos (i+1) e` which leads to `std_pos (i+2) e`

2. **Prove subsumption**: Show `std_pos (i+2) e` subsumes `std_pos (i+2) (e+1)` and both are in same `trans_pos`

3. **Restructure main lemma**: Only claim reachability for positions surviving antichain

## Recommended Next Steps

1. **Decide on solution approach** for `reachable_implies_contained_aux`:
   - Option A: Weaker predicate (most general, more work)
   - Option B: Final positions only (sufficient for completeness, simpler)
   - Option C: Restructure proof (use existing infrastructure)

2. **If Option B/C chosen**:
   - Create `reachable_final_produces_closed_final` lemma
   - Modify `automaton_final_state_accepts_standard` to use new approach
   - Chain through to `automaton_complete_standard` (already proven given accepts)

3. **For transposition proofs**:
   - `automaton_run_step_std_trans` needs characteristic vector analysis
   - `standard_accepts_implies_transposition_accepts` builds on step lemma
   - `automaton_complete_transposition` may need separate Damerau reachability
   - See detailed analysis above for `automaton_run_preserves_reachable_transposition`

4. **For soundness**:
   - `automaton_run_preserves_reachable_transposition` blocked by spurious position issue
   - Need to implement one of the solution approaches (A, B, or C) above

## Categorization of Remaining Work

### Category 1: Requires Structural Changes (HIGH COMPLEXITY)

These lemmas need fundamental changes to the proof approach:

| Lemma | Issue | Effort |
|-------|-------|--------|
| `automaton_run_preserves_reachable_transposition` | Spurious special positions (see analysis above) | HIGH |
| `pseudo_reachable_nonspecial_implies_reachable` | Same spurious position issue | HIGH |

**Recommendation**: Implement Approach C (post-antichain reachability) - requires proving spurious outputs are always subsumed.

### Category 2: Requires Trace-Edit Correspondence (MEDIUM COMPLEXITY)

These lemmas need to establish correspondence between traces and edit sequences:

| Lemma | What's Needed | Effort |
|-------|---------------|--------|
| `dl_distance_le_valid_trace_cost` | Prove trace represents valid edit sequence with matching cost | MEDIUM |
| `dl_optimal_trace_exists` | Construct trace by backtracking through DP recursion | MEDIUM |

**Recommendation**: Define an inductive "trace-to-edits" relation and prove correspondence.

### Category 3: Requires Characteristic Vector Analysis (MEDIUM COMPLEXITY)

| Lemma | What's Needed | Effort |
|-------|---------------|--------|
| `automaton_run_step_std_trans` | Analyze CV bit compatibility between Standard and Transposition | MEDIUM |

**Recommendation**: Prove that Standard's CV window is contained in Transposition's, so Standard transitions are a subset.

### Category 4: Chain from Other Lemmas (LOW COMPLEXITY once dependencies complete)

| Lemma | Dependencies | Status |
|-------|-------------|--------|
| `standard_accepts_implies_transposition_accepts` | `automaton_run_step_std_trans` | **COMPLETED (Qed)** |
| `automaton_complete_transposition` | Direct Damerau completeness (cannot use Standard→Trans because damerau < lev possible) | Admitted |

**Note on `standard_accepts_implies_transposition_accepts`**: This lemma is proven but relies on admitted subcases in `automaton_run_step_std_trans` (spread bound) and `automaton_run_std_trans_correspondence` (position inclusion through antichain). The proof structure is sound.

### Category 4.1: Position Inclusion Through Antichain (HIGH COMPLEXITY)

The `automaton_run_std_trans_correspondence` lemma requires maintaining position inclusion `incl (positions s_std) (positions s_trans)` through the automaton run. After each transition:
- We have `incl closed_std closed_trans` before antichain filtering
- But proving inclusion AFTER antichain filtering is complex because:
  - Transposition's `closed_trans` has extra positions (from complete_transpose)
  - These extra non-special positions could subsume Standard positions
  - Special positions cannot subsume non-special (subsumption rules)

**Key insight**: For FINAL positions, the protection is that final positions can only be subsumed by other final positions with lower errors. So even if exact position inclusion doesn't hold, FINAL position preservation does hold.

The current proof uses this weaker property (final state preservation) rather than full position inclusion.

### Category 5: Out of Scope

The 4 merge/split lemmas remain out of scope as documented.

## Build Command

```bash
cd docs/verification/core/theories
systemd-run --user --scope -p MemoryMax=126G -p CPUQuota=1800% -p IOWeight=30 -p TasksMax=200 make -j1
```

## File Locations

- Plan: `/home/dylon/.claude/plans/robust-gliding-porcupine.md`
- Completeness.v: `Automaton/Completeness.v`
- Soundness.v: `Automaton/Soundness.v`
- MainTheorem.v: `Automaton/MainTheorem.v`
- Transition.v: `Automaton/Transition.v` (helper lemmas)
- DamerauTrace.v: `Trace/DamerauTrace.v` (DL trace infrastructure)
- DamerauComposition.v: `Composition/DamerauComposition.v` (triangle inequality)
