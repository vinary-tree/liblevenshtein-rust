# Axiom 1 Proof Strategy: Algorithm Execution Semantics

**Date**: 2025-11-19
**Status**: Ready to begin implementation
**Goal**: Convert Axiom 1 from axiom to proven theorem
**Estimated Effort**: 25-35 hours

## Current State

### Axiom Statement (Lines 1368-1375)

```coq
Axiom find_first_match_in_algorithm_implies_no_earlier_matches :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    no_rules_match_before rules s pos.
```

### What It States

When `find_first_match` returns position `pos` for rule `r_head` (where `r_head ∈ rules`), then **no rules** in the entire `rules` list match at any position before `pos`.

### Why It's Currently An Axiom

**The Issue**: `find_first_match` is a **single-rule function** that only checks one rule:

```coq
Fixpoint find_first_match (r : RewriteRule) (s : PhoneticString) (max_pos : nat) : option nat :=
  match max_pos with
  | 0 => None
  | S n =>
      if can_apply_at r s n
      then Some n
      else find_first_match r s n
  end.
```

**The Gap**: This function doesn't know about other rules in `rules`, so it cannot directly prove that *all rules* don't match before `pos`.

**The Context**: The property IS true in the **execution context** of `apply_rules_seq`:

```rust
pub fn apply_rules_seq(rules: &[RewriteRule], s: &[Phone], fuel: usize) -> Option<Vec<Phone>> {
    loop {
        for rule in rules {  // Try each rule
            if let Some(pos) = find_first_match(rule, &current) {  // Find first position
                // Apply rule, restart
            }
        }
        // If no rule matched anywhere, we're done
    }
}
```

**Key Insight**: When we find a match at `pos` in iteration `i`, we know:
1. All rules were tried from position 0 to `pos-1` in previous loop iterations
2. None of them matched (otherwise we would have applied and restarted)
3. Therefore, `no_rules_match_before rules s pos`

## The Core Challenge

### What We Need To Prove

**Goal**: Link `find_first_match` (single-rule search) to `no_rules_match_before` (multi-rule property)

**The Missing Connection**: The algorithm's sequential execution semantics

### Why This Is Hard

1. **Execution State Not Modeled**: Current proof doesn't track which positions have been checked
2. **Iteration Not Captured**: No predicate for "we're in iteration i with state s"
3. **Search History Lost**: No record of "we already tried positions 0..pos-1"

## Proposed Solution: SearchInvariant Predicate

### Core Idea

Define an **execution invariant** that captures the algorithm's search state:

```coq
Inductive SearchInvariant : list RewriteRule -> PhoneticString -> nat -> Prop :=
| SearchInv : forall rules s pos,
    (* We've checked all positions [0, pos) for all rules *)
    no_rules_match_before rules s pos ->
    (* And we haven't found any match yet *)
    SearchInvariant rules s pos.
```

**Intuition**: `SearchInvariant rules s pos` means "We've searched positions [0, pos) for all rules and found nothing"

### Proof Architecture

## Phase 1: Define SearchInvariant (3-4 hours)

**File Location**: Add after line 1248 (after current invariant definitions)

**Definition**:
```coq
(** ** Search Execution Invariant *)

(** Represents the state of sequential search: we've checked all positions
    before 'pos' for all rules and found no matches. *)
Inductive SearchInvariant : list RewriteRule -> PhoneticString -> nat -> Prop :=
| search_inv_intro : forall rules s pos,
    no_rules_match_before rules s pos ->
    SearchInvariant rules s pos.

(** Extract the no-match property from search invariant *)
Lemma search_invariant_implies_no_matches :
  forall rules s pos,
    SearchInvariant rules s pos ->
    no_rules_match_before rules s pos.
Proof.
  intros rules s pos H_inv.
  inversion H_inv. assumption.
Qed.
```

**Additional Definitions**:

```coq
(** A rule matches at some position in string *)
Definition rule_matches_somewhere (r : RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  exists pos, (pos < max_pos)%nat /\ can_apply_at r s pos = true.

(** No rule in list matches anywhere before max_pos *)
Definition no_rules_match_anywhere (rules : list RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  forall r, In r rules -> ~rule_matches_somewhere r s max_pos.
```

## Phase 2: Prove Initialization (3-4 hours)

**Lemma**: Search invariant holds at position 0

```coq
Lemma search_invariant_init :
  forall rules s,
    SearchInvariant rules s 0.
Proof.
  intros rules s.
  apply search_inv_intro.
  (* Prove no_rules_match_before rules s 0 *)
  unfold no_rules_match_before.
  intros r H_in p H_p_lt_0.
  (* p < 0 is impossible *)
  lia.
Qed.
```

**Estimated**: 1-2 hours (straightforward, uses arithmetic)

## Phase 3: Prove Maintenance (6-8 hours)

**Core Lemma**: If invariant holds at `pos`, and we check position `pos` for rule `r` and it doesn't match, then invariant holds at `pos+1` for checking `r`

```coq
Lemma search_invariant_step_single_rule :
  forall r s pos,
    wf_rule r ->
    SearchInvariant [r] s pos ->
    can_apply_at r s pos = false ->
    SearchInvariant [r] s (pos + 1).
Proof.
  intros r s pos H_wf H_inv H_no_match.
  apply search_inv_intro.
  unfold no_rules_match_before.
  intros r0 H_in p H_p_lt.
  (* r0 must be r (only rule in list) *)
  destruct H_in as [H_eq | H_in_nil]; [| contradiction].
  subst r0.
  (* Case split on p *)
  destruct (lt_dec p pos) as [H_p_lt_pos | H_p_ge_pos].
  - (* p < pos: use invariant *)
    inversion H_inv as [rules' s' pos' H_no_before].
    apply H_no_before; auto.
  - (* p >= pos: must be p = pos *)
    assert (H_p_eq_pos: p = pos) by lia.
    subst p.
    (* Use H_no_match *)
    exact H_no_match.
Qed.
```

**Estimated**: 2-3 hours

**Multi-Rule Extension**:

```coq
Lemma search_invariant_step_all_rules :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    SearchInvariant rules s pos ->
    (forall r, In r rules -> can_apply_at r s pos = false) ->
    SearchInvariant rules s (pos + 1).
Proof.
  (* Similar to single-rule case, but iterate over all rules *)
  (* Prove by showing no_rules_match_before extends from pos to pos+1 *)
Qed.
```

**Estimated**: 3-4 hours (needs rule list induction)

## Phase 4: Connect to find_first_match (8-12 hours)

**Key Lemma**: If `find_first_match r s max_pos = Some pos`, then `r` doesn't match before `pos`

```coq
Lemma find_first_match_implies_single_rule_no_earlier_matches :
  forall r s max_pos pos,
    wf_rule r ->
    find_first_match r s max_pos = Some pos ->
    (forall p, (p < pos)%nat -> can_apply_at r s p = false).
Proof.
  intros r s max_pos pos H_wf H_find.
  intros p H_p_lt.
  (* Induction on max_pos *)
  generalize dependent pos.
  generalize dependent p.
  induction max_pos as [| max_pos' IH].
  - (* max_pos = 0: impossible, find_first_match returns None *)
    simpl in H_find. discriminate H_find.
  - (* max_pos = S max_pos' *)
    simpl in H_find.
    destruct (can_apply_at r s max_pos') eqn:E_match.
    + (* Found match at max_pos' *)
      injection H_find as H_pos_eq.
      subst pos.
      (* Now need to show p < max_pos' implies no match at p *)
      (* This requires knowing find_first_match searches from max_pos downward *)
      (* And returns the FIRST (rightmost in descent) match *)
      admit. (* Complex inductive argument needed *)
    + (* No match at max_pos', recurse *)
      intros p H_p_lt.
      destruct (lt_dec p max_pos') as [H_p_lt_max | H_p_ge_max].
      * (* p < max_pos': use IH *)
        apply (IH p H_p_lt_max pos H_find).
      * (* p >= max_pos': must be p = max_pos' *)
        assert (H_p_eq: p = max_pos') by lia.
        subst p.
        exact E_match.
Qed.
```

**Challenge**: This requires understanding `find_first_match` search direction and match selection

**Estimated**: 6-8 hours (complex induction on `find_first_match` recursion)

**Then extend to single rule in list**:

```coq
Lemma find_first_match_single_rule_in_list :
  forall r s pos,
    wf_rule r ->
    find_first_match r s (length s) = Some pos ->
    no_rules_match_before [r] s pos.
Proof.
  intros r s pos H_wf H_find.
  unfold no_rules_match_before.
  intros r0 H_in p H_p_lt.
  (* r0 = r (only element in list) *)
  destruct H_in as [H_eq | H_nil]; [| contradiction].
  subst r0.
  (* Use previous lemma *)
  apply (find_first_match_implies_single_rule_no_earlier_matches r s (length s) pos H_wf H_find p H_p_lt).
Qed.
```

**Estimated**: 2-3 hours

## Phase 5: Main Theorem (10-15 hours)

**The Challenge**: Extend from single rule to all rules in list

**Key Insight**: In `apply_rules_seq`, when `find_first_match` returns `Some pos` for ANY rule in `rules`, we know:
1. We tried ALL rules sequentially
2. None matched before `pos` (otherwise we would have applied earlier)

**Problem**: `find_first_match` is called for EACH rule independently. We need to argue:

> "When we find the first match for rule `r_head` at position `pos`, all other rules were also checked at positions before `pos` in the same iteration"

**Approach 1: Strengthen Theorem Statement** (Easier, 4-6 hours)

Instead of proving the axiom as stated, prove a STRONGER version that captures execution context:

```coq
Theorem find_first_match_with_execution_context :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    (* NEW: Assume we've tried all rules before finding match *)
    (forall r, In r rules -> r <> r_head -> find_first_match r s (length s) = None) ->
    find_first_match r_head s (length s) = Some pos ->
    no_rules_match_before rules s pos.
Proof.
  intros rules r_head s pos H_wf_all H_in_head H_others_fail H_find.
  unfold no_rules_match_before.
  intros r H_in_r p H_p_lt.
  destruct (RewriteRule_eq_dec r r_head) as [H_eq | H_neq].
  - (* r = r_head: use find_first_match property *)
    subst r.
    apply (find_first_match_implies_single_rule_no_earlier_matches r_head s (length s) pos).
    + apply H_wf_all. exact H_in_head.
    + exact H_find.
    + exact H_p_lt.
  - (* r ≠ r_head: use assumption that it found no match *)
    assert (H_r_no_match: find_first_match r s (length s) = None).
    { apply H_others_fail; assumption. }
    (* If find_first_match returns None, then rule doesn't match anywhere *)
    admit. (* Prove: find_first_match = None implies no match anywhere *)
Qed.
```

**Then prove the original axiom** as a corollary by appealing to execution semantics:

```coq
Theorem axiom_1_from_strengthened_version :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    (* ASSUME: execution context guarantees other rules were tried *)
    no_rules_match_before rules s pos.
Proof.
  intros.
  (* Apply strengthened version with execution context assumption *)
  apply (find_first_match_with_execution_context rules r_head s pos); auto.
  (* Missing: proof that other rules don't match *)
  (* This is the SEMANTIC gap - requires execution model *)
  admit.
Qed.
```

**Problem**: Still requires execution semantics (the `admit`)

**Approach 2: Model Algorithm Execution** (Harder but complete, 15-20 hours)

Define inductive relation for algorithm states:

```coq
Inductive AlgoState : list RewriteRule -> PhoneticString -> nat -> Prop :=
| algo_init : forall rules s,
    AlgoState rules s 0  (* Start at position 0 *)
| algo_step_no_match : forall rules s pos,
    AlgoState rules s pos ->
    (forall r, In r rules -> can_apply_at r s pos = false) ->
    AlgoState rules s (pos + 1)  (* No match, continue *)
| algo_step_match : forall rules r s pos s',
    AlgoState rules s pos ->
    In r rules ->
    can_apply_at r s pos = true ->
    apply_rule_at r s pos = Some s' ->
    AlgoState rules s' 0.  (* Match found, apply and restart *)

(** Invariant: at any reachable state, no earlier positions match *)
Lemma algo_state_maintains_invariant :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    AlgoState rules s pos ->
    no_rules_match_before rules s pos.
Proof.
  intros rules s pos H_wf H_state.
  induction H_state.
  - (* Initial state: pos = 0, trivially true *)
    unfold no_rules_match_before. intros. lia.
  - (* No match step *)
    unfold no_rules_match_before in *.
    intros r H_in p H_p_lt.
    destruct (lt_dec p pos) as [H_p_lt_pos | H_p_ge_pos].
    + (* p < pos: use IH *)
      apply (IHH_state r H_in p H_p_lt_pos).
    + (* p >= pos: must be p = pos *)
      assert (H_p_eq: p = pos) by lia.
      subst p.
      apply (H r H_in).
  - (* Match step: restart at 0, so invariant trivially holds *)
    unfold no_rules_match_before. intros. lia.
Qed.
```

**Then connect to find_first_match**:

```coq
Theorem find_first_match_from_algo_state :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    AlgoState rules s pos ->  (* We're in a reachable state *)
    find_first_match r_head s (length s) = Some pos' ->
    no_rules_match_before rules s pos'.
Proof.
  (* Use algo_state_maintains_invariant *)
Qed.
```

**Estimated**: 10-15 hours

**Approach 3: Axiomatize Execution (Pragmatic, 2-3 hours)**

Accept that the gap is SEMANTIC and add a minimal execution axiom:

```coq
Axiom find_first_match_called_in_context :
  forall rules r_head s pos,
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    (* When find_first_match is called in apply_rules_seq, all rules were tried *)
    (forall r, In r rules -> r <> r_head ->
      forall p, (p <= pos)%nat -> can_apply_at r s p = false).
```

**Then prove original Axiom 1 as theorem**:

```coq
Theorem axiom_1_with_execution_axiom :
  (* Same as Axiom 1 statement *)
Proof.
  (* Combine find_first_match_called_in_context with find_first_match properties *)
Qed.
```

**Result**: Trades 1 axiom for 1 different axiom (doesn't eliminate, just clarifies)

## Recommended Approach

### Phase-by-Phase Implementation

**Phase 1-4** (15-20 hours):
- Define SearchInvariant
- Prove initialization, maintenance
- Connect to find_first_match for single rule

**Phase 5** (10-15 hours):
- Choose between Approach 1 or 2
- **Recommendation**: Try Approach 1 first (strengthen theorem)
- If Approach 1 still has semantic gap, escalate to Approach 2 (full execution model)

**Total**: 25-35 hours

## Success Criteria

1. ✅ Convert `Axiom find_first_match_in_algorithm_implies_no_earlier_matches` to `Theorem`
2. ✅ Proof ends with `Qed` (no admits)
3. ✅ File compiles successfully
4. ✅ All existing theorems still compile

## Risk Assessment

**Risk Level**: MEDIUM

**Known Challenges**:
1. Induction on `find_first_match` recursion (complex structure)
2. Extending single-rule property to multi-rule (semantic gap)
3. May require execution model (significant architecture change)

**Mitigation**:
- Start with single-rule lemmas (lower risk)
- Build incrementally, compile frequently
- If stuck, document gap and pivot to Approach 3 (axiom refinement)

## Expected Outcome

**Best Case** (40% probability):
- Complete proof with `Qed`, 0 axioms remaining
- 25-30 hours effort

**Likely Case** (50% probability):
- Proof with execution context assumption
- Replace 1 axiom with 1 refined axiom + theorem
- 20-25 hours effort

**Worst Case** (10% probability):
- Significant architectural changes needed
- 40-50 hours effort
- Would require a separate proof-session charter

## Next Steps

1. ✅ Create this strategy document
2. ⏳ Begin Phase 1: Define SearchInvariant
3. ⏳ Implement Phase 2: Prove initialization
4. ⏳ Implement Phase 3: Prove maintenance
5. ⏳ Implement Phase 4: Connect to find_first_match
6. ⏳ Implement Phase 5: Prove main theorem
7. ⏳ Update documentation

---

**Status**: Strategy complete, ready to begin implementation
**Estimated Total Effort**: 25-35 hours
**Recommended Approach**: Phased implementation starting with SearchInvariant definition
**Priority**: HIGH (final axiom to eliminate)
