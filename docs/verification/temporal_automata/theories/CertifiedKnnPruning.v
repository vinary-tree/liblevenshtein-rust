(** * Certified region pruning under a monotone best-k threshold

    This is an exact-natural ghost model. A region contains actual ranks only
    for semantic quantification; a runtime bound must be produced and checked
    without enumerating them. The scope and Rust correspondence obligations
    are separate from these conditional theorems. *)

From Stdlib Require Import Arith Lia List Permutation Sorting.Sorted.
From Liblevenshtein.TemporalAutomata Require Import CertifiedKnnHeap.
Import ListNotations.
Set Implicit Arguments.

Inductive region_summary :=
| Unknown
| Empty
| Known (floor : ranked_entry).

Definition region_summary_sound (region : list ranked_entry)
    (summary : region_summary) : Prop :=
  match summary with
  | Unknown => True
  | Empty => region = []
  | Known floor =>
      forall candidate, In candidate region -> rank_le floor candidate
  end.

Definition competitive (capacity : nat) (selected : list ranked_entry)
    (candidate : ranked_entry) : Prop :=
  match kth_rank capacity selected with
  | None => True
  | Some worst => rank_lt candidate worst
  end.

Definition prune_decision (capacity : nat) (selected : list ranked_entry)
    (summary : region_summary) : bool :=
  match summary with
  | Unknown => false
  | Empty => true
  | Known floor =>
      match kth_rank capacity selected with
      | None => false
      | Some worst => if rank_le_dec worst floor then true else false
      end
  end.

Record scoped_region_summary (Scope : Type) := {
  bound_scope : Scope;
  bound_summary : region_summary
}.

Definition scoped_summary_sound {Scope : Type}
    (ranked_region : Scope -> list ranked_entry)
    (claim : scoped_region_summary Scope) : Prop :=
  region_summary_sound
    (ranked_region (bound_scope claim)) (bound_summary claim).

Definition scoped_prune_decision {Scope : Type}
    (scope_eq_dec : forall left right : Scope,
      {left = right} + {left <> right})
    (active : Scope) (capacity : nat) (selected : list ranked_entry)
    (claim : scoped_region_summary Scope) : bool :=
  if scope_eq_dec active (bound_scope claim) then
    prune_decision capacity selected (bound_summary claim)
  else false.

Lemma rank_lt_le_trans : forall first middle last,
  rank_lt first middle -> rank_le middle last -> rank_lt first last.
Proof.
  intros first middle last Hstrict Hweak.
  unfold rank_lt, rank_le in *.
  destruct Hstrict as [Hcost | [Hcost Htie]];
    destruct Hweak as [Hnext | [Hnext Hnext_tie]];
    [left | left | left | right]; lia.
Qed.

Lemma rank_le_excludes_reverse_lt : forall threshold candidate,
  rank_le threshold candidate -> ~ rank_lt candidate threshold.
Proof.
  intros threshold candidate Hweak Hstrict.
  unfold rank_le, rank_lt in *.
  destruct Hweak as [Hcost | [Hcost Htie]];
    destruct Hstrict as [Hcost_back | [Hcost_back Htie_back]];
    lia.
Qed.

Theorem unknown_summary_cannot_prune : forall capacity selected,
  prune_decision capacity selected Unknown = false.
Proof. reflexivity. Qed.

Theorem known_summary_requires_full_heap :
  forall capacity selected floor,
    prune_decision capacity selected (Known floor) = true ->
    capacity > 0 /\ length selected = capacity.
Proof.
  intros capacity selected floor Hprune.
  unfold prune_decision in Hprune.
  destruct (kth_rank capacity selected) as [worst |] eqn:Hkth;
    [|discriminate].
  now destruct (kth_rank_some_full capacity selected Hkth)
    as [Hpositive [Hfull _]].
Qed.

Theorem prune_decision_sound :
  forall capacity selected region summary,
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    forall candidate, In candidate region ->
      ~ competitive capacity selected candidate.
Proof.
  intros capacity selected region summary Hsound Hprune candidate Hin.
  destruct summary as [| |floor]; simpl in Hprune, Hsound.
  - discriminate.
  - subst region; contradiction.
  - destruct (kth_rank capacity selected) as [worst |] eqn:Hkth;
      [|discriminate].
    destruct (rank_le_dec worst floor) as [Hfloor |];
      [|discriminate].
    unfold competitive; rewrite Hkth.
    eapply rank_le_excludes_reverse_lt.
    eapply rank_le_trans; [exact Hfloor | now apply Hsound].
Qed.

Theorem scoped_prune_decision_sound :
  forall (Scope : Type)
    (scope_eq_dec : forall left right : Scope,
      {left = right} + {left <> right})
    (ranked_region : Scope -> list ranked_entry)
    (active : Scope) capacity selected claim,
    scoped_summary_sound ranked_region claim ->
    scoped_prune_decision scope_eq_dec active capacity selected claim = true ->
    forall candidate, In candidate (ranked_region active) ->
      ~ competitive capacity selected candidate.
Proof.
  intros Scope scope_eq_dec ranked_region active capacity selected claim
    Hsound Hprune candidate Hin.
  unfold scoped_prune_decision in Hprune.
  destruct (scope_eq_dec active (bound_scope claim)) as [Hequal | Hneq];
    [|discriminate].
  rewrite Hequal in Hin.
  eapply prune_decision_sound; eauto.
Qed.

Theorem changed_scope_cannot_prune :
  forall (Scope : Type)
    (scope_eq_dec : forall left right : Scope,
      {left = right} + {left <> right})
    (active : Scope) capacity selected claim,
    active <> bound_scope claim ->
    scoped_prune_decision scope_eq_dec active capacity selected claim = false.
Proof.
  intros Scope scope_eq_dec active capacity selected claim Hdifferent.
  unfold scoped_prune_decision.
  destruct (scope_eq_dec active (bound_scope claim)) as [Hequal |];
    [contradiction | reflexivity].
Qed.

Theorem prune_remains_sound_at_lower_threshold :
  forall capacity old_selected new_selected region summary old_worst new_worst,
    region_summary_sound region summary ->
    prune_decision capacity old_selected summary = true ->
    kth_rank capacity old_selected = Some old_worst ->
    kth_rank capacity new_selected = Some new_worst ->
    rank_le new_worst old_worst ->
    forall candidate, In candidate region ->
      ~ competitive capacity new_selected candidate.
Proof.
  intros capacity old_selected new_selected region summary old_worst
    new_worst Hsound Hprune Hold Hnew Hmonotone candidate Hin Hcomp.
  pose proof (prune_decision_sound capacity old_selected region summary
    Hsound Hprune candidate Hin) as Hexcluded.
  unfold competitive in *.
  rewrite Hold in Hexcluded; rewrite Hnew in Hcomp.
  apply Hexcluded.
  eapply rank_lt_le_trans; eauto.
Qed.

Theorem prune_decision_persists_at_lower_threshold :
  forall capacity old_selected new_selected summary old_worst new_worst,
    prune_decision capacity old_selected summary = true ->
    kth_rank capacity old_selected = Some old_worst ->
    kth_rank capacity new_selected = Some new_worst ->
    rank_le new_worst old_worst ->
    prune_decision capacity new_selected summary = true.
Proof.
  intros capacity old_selected new_selected summary old_worst new_worst
    Hprune Hold Hnew Hmonotone.
  destruct summary as [| |floor]; simpl in *.
  - discriminate.
  - reflexivity.
  - rewrite Hold in Hprune.
    destruct (rank_le_dec old_worst floor) as [Hfloor |];
      [|discriminate].
    rewrite Hnew.
    destruct (rank_le_dec new_worst floor) as [Hnew_floor | Hnot].
    + reflexivity.
    + exfalso; apply Hnot.
      eapply rank_le_trans; eauto.
Qed.

Definition excluded_sound (capacity : nat) (selected excluded
    : list ranked_entry) : Prop :=
  forall candidate, In candidate excluded ->
    ~ competitive capacity selected candidate.

Theorem prune_region_extends_sound_exclusions :
  forall capacity selected region excluded summary,
    excluded_sound capacity selected excluded ->
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    excluded_sound capacity selected (region ++ excluded).
Proof.
  intros capacity selected region excluded summary Hprevious Hsound Hprune
    candidate Hin.
  apply in_app_or in Hin.
  destruct Hin as [Hregion | Hexcluded].
  - eapply prune_decision_sound; eauto.
  - now apply Hprevious.
Qed.

Theorem sound_exclusions_survive_lower_threshold :
  forall capacity old_selected new_selected excluded old_worst new_worst,
    excluded_sound capacity old_selected excluded ->
    kth_rank capacity old_selected = Some old_worst ->
    kth_rank capacity new_selected = Some new_worst ->
    rank_le new_worst old_worst ->
    excluded_sound capacity new_selected excluded.
Proof.
  intros capacity old_selected new_selected excluded old_worst new_worst
    Hsound Hold Hnew Hmonotone candidate Hin Hcomp.
  specialize (Hsound candidate Hin).
  unfold competitive in *.
  rewrite Hold in Hsound; rewrite Hnew in Hcomp.
  apply Hsound; eapply rank_lt_le_trans; eauto.
Qed.

Definition rank_ownership (universe verified pending excluded
    : list ranked_entry) : Prop :=
  Permutation universe (verified ++ pending ++ excluded).

Theorem prune_transfers_region_ownership :
  forall universe verified region remaining excluded,
    rank_ownership universe verified (region ++ remaining) excluded ->
    rank_ownership universe verified remaining (region ++ excluded).
Proof.
  intros universe verified region remaining excluded Hcover.
  unfold rank_ownership in *.
  eapply Permutation_trans; [exact Hcover |].
  apply Permutation_app; [reflexivity |].
  repeat rewrite app_assoc.
  apply Permutation_app; [apply Permutation_app_comm | reflexivity].
Qed.

Theorem prune_action_preserves_ownership_and_exclusions :
  forall universe verified region remaining excluded
    capacity selected summary,
    rank_ownership universe verified (region ++ remaining) excluded ->
    excluded_sound capacity selected excluded ->
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    rank_ownership universe verified remaining (region ++ excluded) /\
    excluded_sound capacity selected (region ++ excluded).
Proof.
  intros universe verified region remaining excluded capacity selected
    summary Hcover Hexcluded Hsound Hprune.
  split.
  - now apply prune_transfers_region_ownership.
  - eapply prune_region_extends_sound_exclusions; eauto.
Qed.

Theorem scoped_prune_action_preserves_ownership_and_exclusions :
  forall (Scope : Type)
    (scope_eq_dec : forall left right : Scope,
      {left = right} + {left <> right})
    (ranked_region : Scope -> list ranked_entry)
    (active : Scope) universe verified remaining excluded
    capacity selected claim,
    rank_ownership universe verified
      (ranked_region active ++ remaining) excluded ->
    excluded_sound capacity selected excluded ->
    scoped_summary_sound ranked_region claim ->
    scoped_prune_decision scope_eq_dec active capacity selected claim = true ->
    rank_ownership universe verified remaining
      (ranked_region active ++ excluded) /\
    excluded_sound capacity selected
      (ranked_region active ++ excluded).
Proof.
  intros Scope scope_eq_dec ranked_region active universe verified
    remaining excluded capacity selected claim Hcover Hexcluded
    Hbound Hprune.
  unfold scoped_prune_decision in Hprune.
  destruct (scope_eq_dec active (bound_scope claim)) as [Hequal | Hneq];
    [|discriminate].
  eapply prune_action_preserves_ownership_and_exclusions;
    [exact Hcover | exact Hexcluded | | exact Hprune].
  unfold scoped_summary_sound in Hbound.
  now rewrite Hequal.
Qed.

Theorem pruned_region_follows_best_k_step :
  forall capacity entry verified selected rejected
    next_selected next_rejected excluded old_worst new_worst,
    best_k_certificate capacity verified selected rejected ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    excluded_sound capacity selected excluded ->
    kth_rank capacity selected = Some old_worst ->
    kth_rank capacity next_selected = Some new_worst ->
    excluded_sound capacity next_selected excluded.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected excluded old_worst new_worst Hbest Hstep Hsound
    Hold Hnew.
  eapply sound_exclusions_survive_lower_threshold;
    [exact Hsound | exact Hold | exact Hnew |].
  eapply full_kth_rank_nonincreasing; eauto.
Qed.

Theorem pruned_summary_follows_best_k_step :
  forall capacity entry verified selected rejected
    next_selected next_rejected summary old_worst new_worst,
    best_k_certificate capacity verified selected rejected ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    prune_decision capacity selected summary = true ->
    kth_rank capacity selected = Some old_worst ->
    kth_rank capacity next_selected = Some new_worst ->
    prune_decision capacity next_selected summary = true.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected summary old_worst new_worst Hbest Hstep Hprune
    Hold Hnew.
  eapply prune_decision_persists_at_lower_threshold;
    [exact Hprune | exact Hold | exact Hnew |].
  eapply full_kth_rank_nonincreasing; eauto.
Qed.

Lemma best_k_step_keeps_full_threshold :
  forall capacity entry selected rejected next_selected next_rejected
    old_worst,
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    kth_rank capacity selected = Some old_worst ->
    exists new_worst, kth_rank capacity next_selected = Some new_worst.
Proof.
  intros capacity entry selected rejected next_selected next_rejected
    old_worst Hstep Hold.
  destruct Hstep.
  - subst capacity; discriminate.
  - destruct (kth_rank_some_full capacity selected Hold)
      as [_ [Hfull _]]; lia.
  - apply kth_rank_full_exists.
    + rewrite length_app in H; simpl in H; lia.
    + rewrite insert_ranked_length.
      rewrite length_app in H; simpl in H; lia.
  - now exists old_worst.
Qed.

Theorem pruned_region_survives_best_k_step :
  forall capacity entry verified selected rejected
    next_selected next_rejected excluded old_worst,
    best_k_certificate capacity verified selected rejected ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    excluded_sound capacity selected excluded ->
    kth_rank capacity selected = Some old_worst ->
    excluded_sound capacity next_selected excluded.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected excluded old_worst Hbest Hstep Hsound Hold.
  destruct (best_k_step_keeps_full_threshold Hstep Hold)
    as [new_worst Hnew].
  eapply pruned_region_follows_best_k_step; eauto.
Qed.

Theorem pruned_summary_survives_best_k_step :
  forall capacity entry verified selected rejected
    next_selected next_rejected summary old_worst,
    best_k_certificate capacity verified selected rejected ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    prune_decision capacity selected summary = true ->
    kth_rank capacity selected = Some old_worst ->
    prune_decision capacity next_selected summary = true.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected summary old_worst Hbest Hstep Hprune Hold.
  destruct (best_k_step_keeps_full_threshold Hstep Hold)
    as [new_worst Hnew].
  eapply pruned_summary_follows_best_k_step; eauto.
Qed.

Theorem pruned_region_follows_improved_threshold :
  forall capacity old_selected new_selected region excluded summary
    old_worst new_worst,
    excluded_sound capacity old_selected excluded ->
    region_summary_sound region summary ->
    prune_decision capacity old_selected summary = true ->
    kth_rank capacity old_selected = Some old_worst ->
    kth_rank capacity new_selected = Some new_worst ->
    rank_le new_worst old_worst ->
    excluded_sound capacity new_selected (region ++ excluded).
Proof.
  intros capacity old_selected new_selected region excluded summary
    old_worst new_worst Hexcluded Hbound Hprune Hold Hnew Hmonotone.
  eapply sound_exclusions_survive_lower_threshold;
    [|exact Hold | exact Hnew | exact Hmonotone].
  eapply prune_region_extends_sound_exclusions; eauto.
Qed.

Theorem prune_selected_precedes_region :
  forall capacity verified selected rejected region summary,
    best_k_certificate capacity verified selected rejected ->
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    forall retained candidate,
      In retained selected -> In candidate region ->
      rank_le retained candidate.
Proof.
  intros capacity verified selected rejected region summary Hbest Hsound
    Hprune retained candidate Hretained Hcandidate.
  destruct summary as [| |floor]; simpl in Hsound, Hprune.
  - discriminate.
  - subst region; contradiction.
  - destruct (kth_rank capacity selected) as [worst |] eqn:Hkth;
      [|discriminate].
    destruct (rank_le_dec worst floor) as [Hfloor |];
      [|discriminate].
    eapply rank_le_trans.
    + eapply kth_rank_is_selected_maximum; eauto.
    + eapply rank_le_trans; [exact Hfloor | now apply Hsound].
Qed.

(** This counterfactual certificate adds the hidden exact ranks to the
    reference universe. It proves result preservation; it does not claim that
    the runtime verifier actually evaluated a pruned region. *)
Theorem pruned_region_preserves_best_k_certificate :
  forall capacity verified selected rejected region summary,
    best_k_certificate capacity verified selected rejected ->
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    NoDup (map entry_original (region ++ verified)) ->
    NoDup (map entry_tie (region ++ verified)) ->
    best_k_certificate capacity (region ++ verified)
      selected (rejected ++ region).
Proof.
  intros capacity verified selected rejected region summary Hbest
    Hsound Hprune Hdistinct_original Hdistinct_tie.
  pose proof Hbest as Hbase.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Hprecede
    Horiginal Htie].
  constructor.
  - eapply Permutation_trans.
    + apply Permutation_app_comm.
    + eapply Permutation_trans.
      * apply Permutation_app; [exact Hcover | reflexivity].
      * now rewrite app_assoc.
  - exact Hsorted.
  - exact Hcap.
  - intro Hunder.
    specialize (Hfill Hunder); subst rejected.
    destruct summary as [| |floor]; simpl in Hsound, Hprune.
    + discriminate.
    + now subst region.
    + destruct (known_summary_requires_full_heap capacity selected floor Hprune)
        as [_ Hfull]; lia.
  - intros retained discarded Hretained Hdiscarded.
    apply in_app_or in Hdiscarded.
    destruct Hdiscarded as [Hold | Hregion].
    + eapply Hprecede; eauto.
    + eapply prune_selected_precedes_region; eauto.
  - exact Hdistinct_original.
  - exact Hdistinct_tie.
Qed.

Theorem pruned_region_cannot_change_best_k :
  forall capacity verified selected rejected region summary,
    best_k_certificate capacity verified selected rejected ->
    region_summary_sound region summary ->
    prune_decision capacity selected summary = true ->
    NoDup (map entry_original (region ++ verified)) ->
    NoDup (map entry_tie (region ++ verified)) ->
    selected = firstn capacity (sort_ranked (region ++ verified)).
Proof.
  intros capacity verified selected rejected region summary Hbest
    Hsound Hprune Horiginal Htie.
  apply certified_best_k_is_canonical with (rejected := rejected ++ region).
  eapply pruned_region_preserves_best_k_certificate; eauto.
Qed.

Example cost_only_equality_prune_is_unsound :
  entry_cost tie_zero = entry_cost tie_one /\
  rank_lt tie_zero tie_one /\
  prune_decision 1 [tie_one] (Known tie_zero) = false.
Proof.
  split; [reflexivity |].
  split; [unfold rank_lt, tie_zero, tie_one; simpl; right; lia |
    reflexivity].
Qed.

Example unknown_summary_is_not_empty :
  prune_decision 1 [tie_one] Unknown = false /\
  prune_decision 1 [tie_one] Empty = true.
Proof. split; reflexivity. Qed.

Example increased_threshold_breaks_prune_persistence :
  prune_decision 1 [tie_zero] (Known tie_one) = true /\
  prune_decision 1 [later_worse] (Known tie_one) = false.
Proof. split; reflexivity. Qed.

Definition scope_zero_claim : scoped_region_summary nat :=
  {| bound_scope := 0; bound_summary := Known tie_zero |}.

Example changed_revision_rejects_old_bound :
  scoped_prune_decision Nat.eq_dec 1 1 [tie_zero]
    scope_zero_claim = false /\
  prune_decision 1 [tie_zero] (Known tie_zero) = true.
Proof. split; reflexivity. Qed.
