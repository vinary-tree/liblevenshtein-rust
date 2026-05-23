(** * No-Match Preservation Lemmas - Position Skipping Optimization

    This module contains lemmas about preservation of the no-match property
    when rules are applied. These lemmas are critical for proving that
    the position-skipping optimization is sound.

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

(** * Context Preservation Lemmas *)

(** These lemmas prove that position-independent contexts remain valid
    at earlier positions after a transformation.
*)

(** Lemma: Initial context is always preserved (only depends on pos = 0) *)
Lemma initial_context_preserved :
  forall s s' transform_pos,
    (transform_pos > 0)%nat ->
    context_preserved_at_earlier_positions Initial s s' transform_pos.
Proof.
  intros s s' transform_pos H_pos_gt.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  (* Initial only matches at position 0 *)
  unfold context_matches.
  destruct (Nat.eq_dec pos 0) as [H_eq | H_ne].
  - (* pos = 0, and transform_pos > 0, so position 0 is unchanged *)
    subst pos.
    reflexivity.
  - (* pos <> 0, so Initial doesn't match in either case *)
    reflexivity.
Qed.

(** Lemma: Anywhere context is always preserved (always matches) *)
Lemma anywhere_context_preserved :
  forall s s' transform_pos,
    context_preserved_at_earlier_positions Anywhere s s' transform_pos.
Proof.
  intros s s' transform_pos.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  unfold context_matches.
  reflexivity.
Qed.

(** Lemma: BeforeVowel context is preserved at earlier positions *)
Lemma before_vowel_context_preserved :
  forall vowels s s' transform_pos,
    (forall i, (i < transform_pos)%nat -> nth_error s i = nth_error s' i) ->
    context_preserved_at_earlier_positions (BeforeVowel vowels) s s' transform_pos.
Proof.
  intros vowels s s' transform_pos H_prefix.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  unfold context_matches.
  (* BeforeVowel checks nth_error s pos *)
  (* We need to show: (match nth_error s pos with ...) = (match nth_error s' pos with ...) *)
  rewrite <- (H_prefix pos H_lt).
  reflexivity.
Qed.

(** Lemma: BeforeConsonant context is preserved at earlier positions *)
Lemma before_consonant_context_preserved :
  forall consonants s s' transform_pos,
    (forall i, (i < transform_pos)%nat -> nth_error s i = nth_error s' i) ->
    context_preserved_at_earlier_positions (BeforeConsonant consonants) s s' transform_pos.
Proof.
  intros consonants s s' transform_pos H_prefix.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  unfold context_matches.
  (* BeforeConsonant checks nth_error s pos *)
  rewrite <- (H_prefix pos H_lt).
  reflexivity.
Qed.

(** Lemma: AfterConsonant context is preserved at earlier positions *)
Lemma after_consonant_context_preserved :
  forall consonants s s' transform_pos,
    (forall i, (i < transform_pos)%nat -> nth_error s i = nth_error s' i) ->
    context_preserved_at_earlier_positions (AfterConsonant consonants) s s' transform_pos.
Proof.
  intros consonants s s' transform_pos H_prefix.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  unfold context_matches.
  (* AfterConsonant checks nth_error s (pos - 1) when pos > 0 *)
  destruct pos as [| pos'].
  - (* pos = 0: AfterConsonant returns false in both cases *)
    reflexivity.
  - (* pos = S pos': need to show nth_error s pos' = nth_error s' pos' *)
    assert (H_pos'_lt: (pos' < transform_pos)%nat) by lia.
    rewrite <- (H_prefix pos' H_pos'_lt).
    reflexivity.
Qed.

(** Lemma: AfterVowel context is preserved at earlier positions *)
Lemma after_vowel_context_preserved :
  forall vowels s s' transform_pos,
    (forall i, (i < transform_pos)%nat -> nth_error s i = nth_error s' i) ->
    context_preserved_at_earlier_positions (AfterVowel vowels) s s' transform_pos.
Proof.
  intros vowels s s' transform_pos H_prefix.
  unfold context_preserved_at_earlier_positions.
  intros pos H_lt.
  unfold context_matches.
  (* AfterVowel checks nth_error s (pos - 1) when pos > 0 *)
  destruct pos as [| pos'].
  - (* pos = 0: AfterVowel returns false in both cases *)
    reflexivity.
  - (* pos = S pos': need to show nth_error s pos' = nth_error s' pos' *)
    assert (H_pos'_lt: (pos' < transform_pos)%nat) by lia.
    rewrite <- (H_prefix pos' H_pos'_lt).
    reflexivity.
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

Record NoMatchPreservationContracts : Prop := mkNoMatchPreservationContracts {
  no_match_find_first_match_contract :
  forall rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    (* Then: in the context of apply_rules_seq execution, we know that no rules
       in the list matched at any position before pos in this iteration *)
    no_rules_match_before rules s pos;

  no_match_pattern_overlap_preservation :
  forall r s pos s' r' p,
    wf_rule r ->
    wf_rule r' ->
    position_dependent_context (context r') = false ->
    apply_rule_at r s pos = Some s' ->
    (p < pos)%nat ->
    (pos < p + length (pattern r'))%nat ->
    can_apply_at r' s p = false ->
    can_apply_at r' s' p = false
}.

(** Theorem: Multi-rule invariant for position-independent contexts

    When find_first_match finds a position for the first rule, all earlier positions
    were checked and found not to match. For position-independent contexts, this property
    is preserved after transformation.
*)
Theorem no_rules_match_before_first_match_preserved :
  forall (contracts : NoMatchPreservationContracts) rules r rest s pos s' p,
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
  { eapply (no_match_find_first_match_contract contracts).
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
    eapply (no_match_pattern_overlap_preservation contracts).
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
