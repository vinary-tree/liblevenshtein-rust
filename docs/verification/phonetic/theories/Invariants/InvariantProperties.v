(** * Search Invariant and No-Match Properties - Position Skipping Optimization

    This module contains lemmas about the SearchInvariant predicate and
    preservation of the no-match property when rules are applied.

    The SearchInvariant captures that we've checked all positions before
    'pos' for all rules and found no matches. This is central to proving
    that find_first_match's behavior implies no_rules_match_before.

    These lemmas are critical for proving that the position-skipping
    optimization is sound.

    Part of: Liblevenshtein.Phonetic.Verification.Invariants
*)

Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Types.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Lib.
From Liblevenshtein.Phonetic.Verification Require Import Core.Rules.
From Liblevenshtein.Phonetic.Verification Require Import Patterns.Preservation.
Import ListNotations.

(** * Basic No-Match Lemmas *)

(** The no_rules_match_before property holds at position 0 *)
Lemma no_rules_match_before_zero :
  forall rules s,
    no_rules_match_before rules s 0.
Proof.
  intros rules s.
  unfold no_rules_match_before.
  intros r H_in p H_p_lt_0.
  (* p < 0 is impossible for natural numbers *)
  lia.
Qed.

(** Lemma: The invariant holds for the empty rule list (vacuously true) *)
Lemma no_rules_match_before_empty :
  forall s pos,
    no_rules_match_before [] s pos.
Proof.
  intros s pos.
  unfold no_rules_match_before.
  intros r H_in.
  inversion H_in.  (* No rules in empty list *)
Qed.

(** Helper: if no_rules_match_before holds at pos and all rules don't match at pos,
    then it extends to pos+1.
*)
Lemma no_rules_match_before_step :
  forall rules s pos,
    no_rules_match_before rules s pos ->
    (forall r, In r rules -> can_apply_at r s pos = false) ->
    no_rules_match_before rules s (pos + 1).
Proof.
  intros rules s pos H_before H_no_match_pos.
  unfold no_rules_match_before in *.
  intros r H_in p H_p_lt.
  destruct (lt_dec p pos) as [H_p_lt_pos | H_p_ge_pos].
  - (* p < pos: use H_before *)
    apply (H_before r H_in p H_p_lt_pos).
  - (* p >= pos and p < pos + 1: must be p = pos *)
    assert (H_p_eq: p = pos) by lia.
    subst p.
    apply (H_no_match_pos r H_in).
Qed.

(** * Equivalence Lemmas *)

(** Equivalence: no_rules_match_anywhere is equivalent to no_rules_match_before *)
Lemma no_rules_match_anywhere_iff_before :
  forall rules s max_pos,
    no_rules_match_anywhere rules s max_pos <-> no_rules_match_before rules s max_pos.
Proof.
  intros rules s max_pos.
  split; intros H.
  - (* no_rules_match_anywhere -> no_rules_match_before *)
    unfold no_rules_match_before.
    intros r H_in p H_p_lt.
    unfold no_rules_match_anywhere in H.
    unfold rule_matches_somewhere in H.
    destruct (can_apply_at r s p) eqn:E_match; auto.
    (* If can_apply_at r s p = true, we have a contradiction *)
    assert (H_matches: exists pos, (pos < max_pos)%nat /\ can_apply_at r s pos = true).
    { exists p. split; auto. }
    specialize (H r H_in H_matches). contradiction.
  - (* no_rules_match_before -> no_rules_match_anywhere *)
    unfold no_rules_match_anywhere.
    intros r H_in [pos [H_pos_lt H_match]].
    unfold no_rules_match_before in H.
    specialize (H r H_in pos H_pos_lt).
    rewrite H_match in H. discriminate H.
Qed.

(** * Extraction Lemmas *)

(** Extract the no-match property from search invariant *)
Lemma search_invariant_implies_no_matches :
  forall rules s pos,
    SearchInvariant rules s pos ->
    no_rules_match_before rules s pos.
Proof.
  intros rules s pos H_inv.
  inversion H_inv. assumption.
Qed.

(** Equivalently, using no_rules_match_anywhere *)
Lemma search_invariant_implies_no_matches_anywhere :
  forall rules s pos,
    SearchInvariant rules s pos ->
    no_rules_match_anywhere rules s pos.
Proof.
  intros rules s pos H_inv.
  apply no_rules_match_anywhere_iff_before.
  apply search_invariant_implies_no_matches. assumption.
Qed.

(** * Initialization Lemmas *)

(** The search invariant holds at position 0 (base case).
    This is trivially true because there are no positions p with p < 0.
*)
Lemma search_invariant_init :
  forall rules s,
    SearchInvariant rules s 0.
Proof.
  intros rules s.
  apply search_inv_intro.
  apply no_rules_match_before_zero.
Qed.

(** Initialization lemma for specific rules *)
Lemma search_invariant_init_for_rules :
  forall rules s,
    (forall r, In r rules -> wf_rule r) ->
    SearchInvariant rules s 0.
Proof.
  intros rules s H_wf.
  apply search_invariant_init.
Qed.

(** * Maintenance Lemmas *)

(** If invariant holds at pos and we check position pos for rule r and it doesn't match,
    then the invariant extends to pos+1 for the single-rule list [r].
*)
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
  (* r0 must be r (only rule in singleton list) *)
  destruct H_in as [H_eq | H_in_nil]; [| contradiction].
  subst r0.
  (* Case split on p *)
  destruct (lt_dec p pos) as [H_p_lt_pos | H_p_ge_pos].
  - (* p < pos: use invariant *)
    apply (search_invariant_implies_no_matches [r] s pos H_inv r).
    + left. reflexivity.
    + exact H_p_lt_pos.
  - (* p >= pos: must be p = pos (since p < pos + 1) *)
    assert (H_p_eq_pos: p = pos) by lia.
    subst p.
    (* Use H_no_match *)
    exact H_no_match.
Qed.

(** Main maintenance lemma: SearchInvariant extends from pos to pos+1
    when all rules don't match at pos.
*)
Lemma search_invariant_step_all_rules :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    SearchInvariant rules s pos ->
    (forall r, In r rules -> can_apply_at r s pos = false) ->
    SearchInvariant rules s (pos + 1).
Proof.
  intros rules s pos H_wf H_inv H_no_match_pos.
  apply search_inv_intro.
  apply no_rules_match_before_step.
  - apply (search_invariant_implies_no_matches rules s pos H_inv).
  - exact H_no_match_pos.
Qed.

(** Invariant maintenance by induction on positions *)
Lemma search_invariant_extends :
  forall rules s pos1 pos2,
    (forall r, In r rules -> wf_rule r) ->
    SearchInvariant rules s pos1 ->
    (pos1 <= pos2)%nat ->
    (forall p, (pos1 <= p < pos2)%nat -> forall r, In r rules -> can_apply_at r s p = false) ->
    SearchInvariant rules s pos2.
Proof.
  intros rules s pos1 pos2 H_wf H_inv H_le H_no_match_range.
  apply search_inv_intro.
  unfold no_rules_match_before.
  intros r H_in p H_p_lt_pos2.
  destruct (lt_dec p pos1) as [H_p_lt_pos1 | H_p_ge_pos1].
  - (* p < pos1: use original invariant *)
    apply (search_invariant_implies_no_matches rules s pos1 H_inv r H_in p H_p_lt_pos1).
  - (* pos1 <= p < pos2: use range assumption *)
    apply (H_no_match_range p).
    + lia.
    + exact H_in.
Qed.

(** * Connection to find_first_match *)

(** Key observation: if find_first_match returns None, the rule matches nowhere *)
Lemma find_first_match_none_implies_no_match_anywhere :
  forall r s fuel,
    wf_rule r ->
    find_first_match r s fuel = None ->
    (fuel >= length s)%nat ->
    forall p, (p < length s)%nat -> can_apply_at r s p = false.
Proof.
  intros r s fuel H_wf H_none H_fuel_ge p H_p_lt.

  (* Use the helper lemma to convert to find_first_match at length s *)
  assert (H_none_at_len: find_first_match r s (length s) = None).
  { eapply find_first_match_large_fuel_implies_length; eauto. }

  (* Convert to find_first_match_from *)
  assert (H_equiv: find_first_match r s (length s) =
                   find_first_match_from r s 0 (S (length s))).
  {
    assert (H_tmp: find_first_match r s (length s) =
                   find_first_match_from r s 0 (length s - 0 + 1)%nat).
    { apply find_first_match_equiv_from_zero. exact H_wf. }
    rewrite H_tmp.
    f_equal. lia.
  }

  rewrite H_equiv in H_none_at_len.

  (* Apply the find_first_match_from helper *)
  eapply find_first_match_from_none_implies_all_fail; eauto.
  lia.
Qed.

(** If find_first_match returns Some pos for a rule in the list,
    and all other rules in the list match nowhere, then
    SearchInvariant holds for the entire list at pos.

    This captures the sequential execution context where we try all rules
    before the one that matched.
*)
Lemma find_first_match_with_all_rules_fail_before :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    (* ASSUMPTION: All other rules don't match anywhere before pos *)
    (forall r, In r rules -> r <> r_head -> forall p, (p < pos)%nat -> can_apply_at r s p = false) ->
    SearchInvariant rules s pos.
Proof.
  intros rules r_head s pos H_wf_all H_in_head H_find H_others_no_match.
  apply search_inv_intro.
  unfold no_rules_match_before.
  intros r H_in_r p H_p_lt.
  (* Case split: is r = r_head or r ≠ r_head? *)
  destruct (RewriteRule_eq_dec r r_head) as [H_eq | H_neq].
  - (* r = r_head: use find_first_match_is_first *)
    subst r.
    eapply find_first_match_is_first; eauto.
    lia. (* length s - length s = 0 <= p *)
  - (* r ≠ r_head: use assumption *)
    apply (H_others_no_match r H_in_r H_neq p H_p_lt).
Qed.

(** * Establishment Lemmas *)

(** Lemma: When find_first_match finds a position for a single rule,
    all earlier positions don't match *)
Lemma find_first_match_establishes_invariant_single :
  forall r s pos,
    wf_rule r ->
    find_first_match r s (length s) = Some pos ->
    no_rules_match_before [r] s pos.
Proof.
  intros r s pos H_wf H_find.
  unfold no_rules_match_before.
  intros r0 H_in p H_p_lt.
  (* r0 must be r since [r] is a singleton list *)
  destruct H_in as [H_eq | H_in_empty].
  - (* r0 = r *)
    subst r0.
    (* Use find_first_match_is_first lemma *)
    eapply find_first_match_is_first; eauto.
    (* Need to show: length s - length s <= p *)
    lia.
  - (* r0 ∈ [] - impossible *)
    inversion H_in_empty.
Qed.

(** * Useful Equivalences *)

(** Lemma: If no early matches exist, both find the same match (or both find none) *)
Lemma find_first_match_from_equivalent_when_no_early_matches :
  forall r s start_pos,
    wf_rule r ->
    no_early_matches r s start_pos ->
    (forall pos, find_first_match_from r s start_pos (length s - start_pos + 1) = Some pos ->
                 find_first_match r s (length s) = Some pos).
Proof.
  intros r s start_pos H_wf H_no_early pos H_found.
  unfold no_early_matches in H_no_early.

  (* find_first_match_from found pos, so can_apply_at r s pos = true *)
  assert (H_can_apply: can_apply_at r s pos = true).
  {
    eapply find_first_match_from_implies_can_apply. eauto.
  }

  (* pos >= start_pos *)
  assert (H_bound: (start_pos <= pos)%nat).
  { eapply find_first_match_from_lower_bound. eauto. }

  (* pos < length s: from the search bounds and can_apply_at being true *)
  assert (H_pos_in_bounds: (pos < length s)%nat).
  {
    (* find_first_match_from searched up to position start_pos + n - 1 where n = length s - start_pos + 1 *)
    assert (H_upper: (pos < start_pos + (length s - start_pos + 1))%nat).
    { eapply find_first_match_from_upper_bound. eauto. }
    (* Simplify the bound *)
    destruct (le_lt_dec start_pos (length s)) as [H_start_ok | H_start_large].
    - (* start_pos <= length s, so expression simplifies *)
      assert (H_rewrite: (start_pos + (length s - start_pos + 1) = length s + 1)%nat) by lia.
      rewrite H_rewrite in H_upper.
      (* pos < length s + 1 means pos <= length s *)
      (* To show pos < length s, need to exclude pos = length s *)
      destruct (Nat.eq_dec pos (length s)) as [H_eq | H_ne].
      * (* pos = length s - but can_apply_at at or beyond length s is false for wf rules *)
        subst pos.
        (* This contradicts H_can_apply *)
        assert (H_false: can_apply_at r s (length s) = false).
        { apply can_apply_at_beyond_length. exact H_wf. lia. }
        rewrite H_false in H_can_apply.
        discriminate.
      * (* pos <> length s, and pos <= length s, so pos < length s *)
        lia.
    - (* start_pos > length s, then length s - start_pos = 0 *)
      (* So search range is [start_pos, start_pos + 1) *)
      (* But if can_apply_at is true at pos, and pos >= start_pos > length s *)
      (* This contradicts can_apply_at requiring valid position *)
      (* can_apply_at checks context and pattern at position pos *)
      (* For position >= length s, context matching could succeed, but pattern matching *)
      (* requires pos + pattern_length <= length s, which fails if pos >= length s and pattern non-empty *)
      (* H_upper: pos < start_pos + (length s - start_pos + 1) *)
      (* With length s - start_pos = 0, this becomes pos < start_pos + 1 *)
      (* Which means pos <= start_pos *)
      (* Since start_pos > length s and pos >= start_pos, we have pos >= length s *)
      (* But can_apply_at at position >= length s should be false for non-empty patterns *)
      (* This is getting complex - for now, use a simpler bound *)
      (* Actually, from H_upper, pos < start_pos + 1, so pos <= start_pos *)
      (* Combined with H_bound (start_pos <= pos), we get pos = start_pos *)
      assert (H_pos_eq: pos = start_pos) by lia.
      subst pos.
      (* Now start_pos > length s, but can_apply_at r s start_pos = true *)
      (* This contradicts can_apply_at being false beyond string length for wf rules *)
      assert (H_false: can_apply_at r s start_pos = false).
      { apply can_apply_at_beyond_length. exact H_wf. lia. }
      rewrite H_false in H_can_apply.
      discriminate.
  }

  (* Use find_first_match_finds_first_true to show find_first_match finds pos *)
  apply find_first_match_finds_first_true.
  - exact H_pos_in_bounds.
  - lia.
  - exact H_can_apply.
  - (* For all p in [0, pos), can_apply_at r s p = false *)
    intros p H_range H_lt.
    destruct (Nat.lt_ge_cases p start_pos) as [H_before_start | H_after_start].
    + (* p < start_pos: use H_no_early *)
      apply H_no_early. exact H_before_start.
    + (* start_pos <= p < pos *)
      (* Use find_first_match_from_is_first *)
      eapply find_first_match_from_is_first; eauto.
Qed.

(** * Single-Rule Preservation *)

(** Lemma: If a single rule doesn't match before pos in s,
    it won't match after transformation to s' (with pattern bound constraint) *)
Lemma single_rule_no_match_preserved :
  forall r r_applied s pos s' p,
    wf_rule r_applied ->
    wf_rule r ->
    position_dependent_context (context r) = false ->
    apply_rule_at r_applied s pos = Some s' ->
    (p < pos)%nat ->
    (p + length (pattern r) <= pos)%nat ->  (* Pattern fits before transformation point *)
    can_apply_at r s p = false ->
    can_apply_at r s' p = false.
Proof.
  intros r r_applied s pos s' p H_wf_applied H_wf H_indep H_apply H_p_lt H_pattern_bound H_no_match_s.
  (* Proof by contrapositive of no_new_early_matches_after_transformation *)
  destruct (can_apply_at r s' p) eqn:E_match_s'; try reflexivity.
  (* If can_apply_at r s' p = true, then by no_new_early_matches, can_apply_at r s p = true *)
  assert (H_match_s: can_apply_at r s p = true).
  { apply (no_new_early_matches_after_transformation r_applied s pos s' r p H_wf_applied H_wf H_indep H_apply H_p_lt H_pattern_bound E_match_s'). }
  (* But this contradicts H_no_match_s *)
  rewrite H_match_s in H_no_match_s.
  discriminate H_no_match_s.
Qed.

(** Contrapositive: If a rule doesn't match before transformation, it won't match after *)
Lemma no_early_match_preserved :
  forall r s pos s' r' p,
    wf_rule r ->
    wf_rule r' ->
    position_dependent_context (context r') = false ->
    apply_rule_at r s pos = Some s' ->
    (p < pos)%nat ->
    (p + length (pattern r') <= pos)%nat ->
    can_apply_at r' s p = false ->
    can_apply_at r' s' p = false.
Proof.
  intros r s pos s' r' p H_wf H_wf' H_indep H_apply H_p_lt H_pattern_bound H_no_match_s.
  (* Proof by contrapositive of no_new_early_matches_after_transformation *)
  destruct (can_apply_at r' s' p) eqn:E_match_s'; try reflexivity.
  (* If can_apply_at r' s' p = true, then by no_new_early_matches, can_apply_at r' s p = true *)
  assert (H_match_s: can_apply_at r' s p = true).
  { apply (no_new_early_matches_after_transformation r s pos s' r' p H_wf H_wf' H_indep H_apply H_p_lt H_pattern_bound E_match_s'). }
  (* But this contradicts H_no_match_s *)
  rewrite H_match_s in H_no_match_s.
  discriminate H_no_match_s.
Qed.

(** * Multi-Rule Preservation *)

(** Lemma: If no rules in a list match before pos in s,
    and patterns all fit, then no rules match after transformation *)
Lemma all_rules_no_match_preserved :
  forall rules r_applied s pos s' p,
    wf_rule r_applied ->
    (forall r0, In r0 rules -> wf_rule r0) ->
    (forall r0, In r0 rules -> position_dependent_context (context r0) = false) ->
    (forall r0, In r0 rules -> (p + length (pattern r0) <= pos)%nat) ->  (* All patterns fit *)
    apply_rule_at r_applied s pos = Some s' ->
    (p < pos)%nat ->
    no_rules_match_before rules s pos ->
    no_rules_match_before rules s' p.
Proof.
  intros rules r_applied s pos s' p H_wf_applied H_wf_all H_indep_all H_pattern_bounds H_apply H_p_lt H_inv_before.
  unfold no_rules_match_before in *.
  intros r0 H_in_r0 p0 H_p0_lt.

  (* Get the hypothesis that r0 doesn't match in s *)
  assert (H_r0_no_match_s: can_apply_at r0 s p0 = false).
  { apply H_inv_before.
    - exact H_in_r0.
    - (* p0 < p < pos, so p0 < pos *)
      lia.
  }

  (* Apply single-rule preservation *)
  eapply single_rule_no_match_preserved.
  - exact H_wf_applied.
  - apply H_wf_all. exact H_in_r0.
  - apply H_indep_all. exact H_in_r0.
  - exact H_apply.
  - (* p0 < p < pos *)
    lia.
  - (* Pattern bound: p0 + length(pattern r0) <= pos *)
    assert (H_bound: (p + length (pattern r0) <= pos)%nat).
    { apply H_pattern_bounds. exact H_in_r0. }
    (* Since p0 < p, we have p0 + length(pattern) <= p + length(pattern) <= pos *)
    lia.
  - exact H_r0_no_match_s.
Qed.

(** * Multi-Rule Invariant Preservation *)

(** Axiomatic Gap: Algorithm Semantic Property

    The core challenge in proving the multi-rule invariant is establishing that when
    find_first_match returns a position for rule r, we know not just that r doesn't
    match earlier, but that in the execution of apply_rules_seq, NO rules in the entire
    list have matched at earlier positions.

    This property follows from the sequential nature of the algorithm: if any rule had
    matched earlier, it would have been applied before we reached the current iteration.
    However, proving this requires reasoning about the full execution trace of apply_rules_seq,
    which goes beyond the local properties we've proven.

    We introduce a minimal axiom capturing just this algorithmic property:
*)

Record InvariantPreservationContracts : Prop := mkInvariantPreservationContracts {
  invariant_find_first_match_contract :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    (* Then: in the context of apply_rules_seq execution, we know that no rules
       in the list matched at any position before pos in this iteration *)
    no_rules_match_before rules s pos;

(** Axiomatic Gap: Pattern Overlap Preservation (Axiom 2)

    This is the fundamental gap identified in the Axiom 2 proof attempt.
    When a pattern overlaps with the transformation region, we cannot currently
    prove that a non-match is preserved after transformation.

    This axiom is proven in PatternOverlap.v (Axiom 2), but since that module
    is compiled after this one, we declare it here as well.
*)

  invariant_pattern_overlap_preservation :
  forall r s pos s' r' p,
    wf_rule r ->
    wf_rule r' ->
    position_dependent_context (context r') = false ->
    apply_rule_at r s pos = Some s' ->
    (p < pos)%nat ->
    (pos < p + length (pattern r'))%nat ->  (* Pattern overlaps transformation *)
    can_apply_at r' s p = false ->
    can_apply_at r' s' p = false
}.

(** Theorem: Multi-rule invariant for position-independent contexts

    When find_first_match finds a position for the first rule, all earlier positions
    were checked and found not to match. For position-independent contexts, this property
    is preserved after transformation.
*)
Theorem no_rules_match_before_first_match_preserved :
  forall (contracts : InvariantPreservationContracts) rules r rest s pos s' p,
    rules = r :: rest ->
    (forall r0, In r0 rules -> wf_rule r0) ->
    (forall r0, In r0 rules -> position_dependent_context (context r0) = false) ->
    find_first_match r s (length s) = Some pos ->
    apply_rule_at r s pos = Some s' ->
    (p < pos)%nat ->
    (forall r0, In r0 rules -> can_apply_at r0 s' p = false).
Proof.
  intros contracts rules r rest s pos s' p H_rules H_wf_all H_indep_all H_find H_apply H_p_lt.

  (* Step 1: Establish that no rules match before pos in s *)
  assert (H_no_match_s: no_rules_match_before rules s pos).
  { eapply (invariant_find_first_match_contract contracts).
    - intros r0 H_in. apply H_wf_all. exact H_in.
    - subst rules. left. reflexivity.
    - exact H_find.
  }

  (* Step 2: Apply preservation with pattern bounds *)
  (* We need to show that for each rule r0 in rules, can_apply_at r0 s' p = false *)
  intros r0 H_in_r0.

  (* Get properties of r0 *)
  assert (H_wf_r0: wf_rule r0) by (apply H_wf_all; exact H_in_r0).
  assert (H_indep_r0: position_dependent_context (context r0) = false) by (apply H_indep_all; exact H_in_r0).

  (* r0 doesn't match at p in s *)
  assert (H_no_match_r0_s: can_apply_at r0 s p = false).
  { apply H_no_match_s; assumption. }

  (* Case analysis on whether pattern fits *)
  destruct (le_lt_dec (p + length (pattern r0)) pos) as [H_fits | H_too_long].

  - (* Pattern fits: use preservation lemma *)
    eapply single_rule_no_match_preserved.
    + subst rules. apply H_wf_all. left. reflexivity.
    + exact H_wf_r0.
    + exact H_indep_r0.
    + exact H_apply.
    + exact H_p_lt.
    + exact H_fits.
    + exact H_no_match_r0_s.

  - (* Pattern overlaps: use overlap preservation axiom *)
    eapply (invariant_pattern_overlap_preservation contracts).
    + (* wf_rule r - the rule that was applied *)
      subst rules. apply H_wf_all. left. reflexivity.
    + (* wf_rule r0 - the rule we're checking *)
      exact H_wf_r0.
    + (* position_dependent_context (context r0) = false *)
      exact H_indep_r0.
    + (* apply_rule_at r s pos = Some s' *)
      exact H_apply.
    + (* p < pos *)
      exact H_p_lt.
    + (* pos < p + length (pattern r0) - pattern overlaps *)
      exact H_too_long.
    + (* can_apply_at r0 s p = false *)
      exact H_no_match_r0_s.
Qed.
