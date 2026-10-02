(** * Indistinguishable evidence and incompatible exact answers

    A deterministic evidence-only decision receives the evidence object and
    requested capacity, not the hidden completion. [None] means defer; a
    returned list claims the completed ordered answer. Two feasible models
    with incompatible answers therefore force a sound decision to defer.

    The concrete obstruction below is inhabited: one verified original has
    cost five, while an unresolved original can cost four or eight within
    the same recorded interval. The exact top-one answers are proved to be
    [1] and [0], respectively, including uniqueness against [best_k_over].

    The interval reassignment lemma applies to the independent closed-bound
    language of [EvidenceStopping]. It does not add metric triangles, shared
    recurrence constraints, uncertainty about eligibility, or runtime state.
    No claim is made about all evidence languages, randomized decisions,
    active evidence acquisition, minimum work, or Rust correspondence. *)

From Stdlib Require Import Arith Lia List Sorting.Sorted.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder EvidenceStopping.
Import ListNotations.
Set Implicit Arguments.

Section Indistinguishability.
  Context {Cost : Type}.
  Variable cost_order : certificate_order Cost.

  Definition verified_indistinguishable (evidence : @stopping_evidence Cost)
      (first second : @completion Cost) : Prop :=
    forall original, In original (verified_originals evidence) ->
      order_equiv cost_order (first original) (second original).

  Theorem same_feasible_evidence_hides_unverified_scores :
    forall evidence first second,
      feasible_completion cost_order evidence first ->
      feasible_completion cost_order evidence second ->
      verified_indistinguishable evidence first second.
  Proof.
    intros evidence first second Hfirst Hsecond original Hin.
    pose proof (@feasible_completions_agree_on_verified_ranks Cost cost_order
      evidence first second original Hfirst Hsecond Hin) as Hagree.
    exact (proj1 Hagree).
  Qed.

  Definition set_original_cost (assignment : @completion Cost)
      (original : nat) (replacement : Cost) : @completion Cost :=
    fun candidate =>
      if Nat.eq_dec candidate original then replacement else assignment candidate.

  (** All other assigned costs remain fixed. This is constructive endpoint
      or interior-point attainability for one eligible unverified identity
      in this specific independent-interval evidence language. *)
  Theorem unverified_interval_reassignment_is_feasible :
    forall evidence assignment original replacement,
      feasible_completion cost_order evidence assignment ->
      In original (eligible_originals evidence) ->
      ~ In original (verified_originals evidence) ->
      order_le cost_order (lower_endpoint evidence original) replacement ->
      order_le cost_order replacement (upper_endpoint evidence original) ->
      feasible_completion cost_order evidence
        (set_original_cost assignment original replacement).
  Proof.
    intros evidence assignment original replacement
      [Hformed [Hscores Hbounds]] Heligible Hunverified Hlower Hupper.
    split; [exact Hformed |]; split.
    - intros candidate score Hobserved; unfold set_original_cost.
      destruct (Nat.eq_dec candidate original) as [Hequal | Hdifferent].
      + subst candidate; exfalso; apply Hunverified.
        unfold verified_originals; apply in_map_iff.
        exists (original, score); split; [reflexivity | exact Hobserved].
      + apply Hscores; exact Hobserved.
    - intros candidate Hin; unfold set_original_cost.
      destruct (Nat.eq_dec candidate original) as [Hequal | Hdifferent].
      + subst candidate; split; [exact Hlower | exact Hupper].
      + apply Hbounds; exact Hin.
  Qed.

  Definition evidence_only_decider : Type :=
    @stopping_evidence Cost -> nat -> option (list nat).

  Definition decider_sound_at (decider : evidence_only_decider)
      (evidence : @stopping_evidence Cost) (capacity : nat) : Prop :=
    forall selected, decider evidence capacity = Some selected ->
      universally_safe_to_stop cost_order evidence capacity selected.

  Theorem disjoint_feasible_answers_prevent_certified_output :
    forall evidence capacity first second,
      feasible_completion cost_order evidence first ->
      feasible_completion cost_order evidence second ->
      (forall selected,
        best_k_over cost_order (eligible_originals evidence) first capacity selected ->
        ~ best_k_over cost_order (eligible_originals evidence) second capacity selected) ->
      forall selected,
        ~ universally_safe_to_stop cost_order evidence capacity selected.
  Proof.
    intros evidence capacity first second Hfirst Hsecond Hdisjoint selected
      [_ Hall].
    apply (Hdisjoint selected (Hall first Hfirst)).
    exact (Hall second Hsecond).
  Qed.

  Theorem ambiguous_models_force_deferral :
    forall evidence capacity first second decider,
      feasible_completion cost_order evidence first ->
      feasible_completion cost_order evidence second ->
      (forall selected,
        best_k_over cost_order (eligible_originals evidence) first capacity selected ->
        ~ best_k_over cost_order (eligible_originals evidence) second capacity selected) ->
      decider_sound_at decider evidence capacity ->
      decider evidence capacity = None.
  Proof.
    intros evidence capacity first second decider Hfirst Hsecond Hdisjoint Hsound.
    destruct (decider evidence capacity) as [selected|] eqn:Hdecision;
      [|reflexivity].
    exfalso.
    eapply disjoint_feasible_answers_prevent_certified_output;
      [exact Hfirst | exact Hsecond | exact Hdisjoint |].
    apply Hsound; exact Hdecision.
  Qed.

  Theorem inconsistent_evidence_forces_deferral :
    forall evidence capacity decider,
      ~ evidence_consistent cost_order evidence ->
      decider_sound_at decider evidence capacity ->
      decider evidence capacity = None.
  Proof.
    intros evidence capacity decider Hbad Hsound.
    destruct (decider evidence capacity) as [selected|] eqn:Hdecision;
      [|reflexivity].
    exfalso.
    apply (@inconsistent_evidence_cannot_authorize_stopping Cost cost_order
      evidence capacity selected Hbad).
    apply Hsound; exact Hdecision.
  Qed.

  Lemma singleton_minimum_is_best_one : forall universe assignment winner,
    In winner universe ->
    (forall other, In other universe -> original_le cost_order assignment winner other) ->
    best_k_over cost_order universe assignment 1 [winner].
  Proof.
    intros universe assignment winner Hin Hminimum; constructor.
    - constructor; [simpl; tauto | constructor].
    - intros chosen [Hequal | Hfalse]; [subst chosen; exact Hin | contradiction].
    - constructor; constructor.
    - simpl; lia.
    - intros original Heligible Hnot; split; [reflexivity |].
      intros chosen [Hequal | Hfalse]; [subst chosen | contradiction].
      apply Hminimum; exact Heligible.
  Qed.

  (** The canonical answer below is derived from the independent completed
      best-k contract, rather than assumed as part of indistinguishability. *)
  Theorem strict_singleton_winner_is_unique : forall universe assignment winner,
    NoDup universe -> In winner universe ->
    (forall other, In other universe -> other <> winner ->
      original_lt cost_order assignment winner other) ->
    forall selected,
      best_k_over cost_order universe assignment 1 selected -> selected = [winner].
  Proof.
    intros universe assignment winner Hdistinct Hin Hstrict selected Hbest.
    pose proof (@best_k_has_required_cardinality Cost cost_order
      universe assignment 1 selected Hdistinct Hbest) as Hsize.
    assert (Hnonempty : 0 < length universe).
    { destruct universe; simpl in *; [contradiction | lia]. }
    rewrite Nat.min_l in Hsize by lia.
    destruct selected as [|chosen rest]; [simpl in Hsize; lia |].
    destruct rest as [|extra tail]; [|simpl in Hsize; lia].
    destruct Hbest as [_ Hselected_eligible _ _ Homitted].
    destruct (Nat.eq_dec chosen winner) as [Hequal | Hdifferent].
    - subst chosen; reflexivity.
    - exfalso.
      assert (Hchosen : In chosen universe).
      { apply Hselected_eligible; simpl; auto. }
      assert (Hnot : ~ In winner [chosen]).
      { intros [Hequal | Hfalse]; [apply Hdifferent; exact Hequal | contradiction]. }
      destruct (Homitted winner Hin Hnot) as [_ Hafter].
      eapply strict_order_conflicts_with_reverse.
      + apply Hstrict; [exact Hchosen | exact Hdifferent].
      + apply Hafter; simpl; auto.
  Qed.
End Indistinguishability.

Module InhabitedAmbiguity.
  Import NaturalEvidenceControls.

  Definition shared_evidence := permissive_evidence.
  Definition low_completion := improving_completion.
  Definition high_completion :=
    @set_original_cost nat low_completion 1 8.

  Lemma low_model_is_feasible :
    feasible_completion natural_certificate_order shared_evidence low_completion.
  Proof. exact improving_assignment_is_feasible. Qed.

  Lemma high_model_is_feasible :
    feasible_completion natural_certificate_order shared_evidence high_completion.
  Proof.
    unfold high_completion; apply unverified_interval_reassignment_is_feasible.
    - exact low_model_is_feasible.
    - simpl; auto.
    - simpl; lia.
    - change (0 <= 8); lia.
    - change (8 <= 10); lia.
  Qed.

  Lemma both_models_have_the_same_verified_evidence :
    verified_indistinguishable natural_certificate_order
      shared_evidence low_completion high_completion.
  Proof.
    apply same_feasible_evidence_hides_unverified_scores;
      [exact low_model_is_feasible | exact high_model_is_feasible].
  Qed.

  Example only_an_unverified_eligible_score_changes :
    low_completion 0 = high_completion 0 /\
    low_completion 1 = 4 /\ high_completion 1 = 8 /\
    ~ In 1 (verified_originals shared_evidence).
  Proof.
    split; [reflexivity |]; split; [reflexivity |]; split; [reflexivity |].
    simpl; lia.
  Qed.

  Lemma two_originals_are_distinct : NoDup (eligible_originals shared_evidence).
  Proof.
    constructor; [simpl; lia |].
    constructor; [simpl; tauto | constructor].
  Qed.

  Lemma low_one_is_strictly_before_zero :
    original_lt natural_certificate_order low_completion 1 0.
  Proof.
    change ((4 < 5 \/ (4 = 5 /\ 1 <= 0)) /\ ~ (4 = 5 /\ 1 = 0)); lia.
  Qed.

  Lemma high_zero_is_strictly_before_one :
    original_lt natural_certificate_order high_completion 0 1.
  Proof.
    change ((5 < 8 \/ (5 = 8 /\ 0 <= 1)) /\ ~ (5 = 8 /\ 0 = 1)); lia.
  Qed.

  Theorem low_top_one_is_attained :
    best_k_over natural_certificate_order
      (eligible_originals shared_evidence) low_completion 1 [1].
  Proof.
    apply singleton_minimum_is_best_one; [simpl; auto |].
    intros other Hin; simpl in Hin.
    destruct Hin as [Hequal | [Hequal | Hfalse]];
      [subst other | subst other | contradiction].
    - exact (proj1 low_one_is_strictly_before_zero).
    - apply original_le_reflexive.
  Qed.

  Theorem high_top_one_is_attained :
    best_k_over natural_certificate_order
      (eligible_originals shared_evidence) high_completion 1 [0].
  Proof.
    apply singleton_minimum_is_best_one; [simpl; auto |].
    intros other Hin; simpl in Hin.
    destruct Hin as [Hequal | [Hequal | Hfalse]];
      [subst other | subst other | contradiction].
    - apply original_le_reflexive.
    - exact (proj1 high_zero_is_strictly_before_one).
  Qed.

  Theorem low_top_one_is_unique : forall selected,
    best_k_over natural_certificate_order
      (eligible_originals shared_evidence) low_completion 1 selected -> selected = [1].
  Proof.
    apply strict_singleton_winner_is_unique.
    - exact two_originals_are_distinct.
    - simpl; auto.
    - intros other Hin Hdifferent; simpl in Hin.
      destruct Hin as [Hequal | [Hequal | Hfalse]];
        [subst other | subst other | contradiction].
      + exact low_one_is_strictly_before_zero.
      + exfalso; apply Hdifferent; reflexivity.
  Qed.

  Theorem high_top_one_is_unique : forall selected,
    best_k_over natural_certificate_order
      (eligible_originals shared_evidence) high_completion 1 selected -> selected = [0].
  Proof.
    apply strict_singleton_winner_is_unique.
    - exact two_originals_are_distinct.
    - simpl; auto.
    - intros other Hin Hdifferent; simpl in Hin.
      destruct Hin as [Hequal | [Hequal | Hfalse]];
        [subst other | subst other | contradiction].
      + exfalso; apply Hdifferent; reflexivity.
      + exact high_zero_is_strictly_before_one.
  Qed.

  Theorem same_evidence_has_two_different_exact_top_one_answers :
    feasible_completion natural_certificate_order shared_evidence low_completion /\
    feasible_completion natural_certificate_order shared_evidence high_completion /\
    verified_indistinguishable natural_certificate_order
      shared_evidence low_completion high_completion /\
    full_best_k natural_certificate_order
      (eligible_originals shared_evidence) low_completion 1 [1] /\
    full_best_k natural_certificate_order
      (eligible_originals shared_evidence) high_completion 1 [0] /\
    [1] <> [0].
  Proof.
    split; [exact low_model_is_feasible |].
    split; [exact high_model_is_feasible |].
    split; [exact both_models_have_the_same_verified_evidence |].
    split.
    - split; [exact low_top_one_is_attained |]; split; [reflexivity | lia].
    - split.
      + split; [exact high_top_one_is_attained |]; split; [reflexivity | lia].
      + discriminate.
  Qed.

  Lemma no_selected_list_is_correct_in_both_models : forall selected,
    best_k_over natural_certificate_order
      (eligible_originals shared_evidence) low_completion 1 selected ->
    ~ best_k_over natural_certificate_order
      (eligible_originals shared_evidence) high_completion 1 selected.
  Proof.
    intros selected Hlow Hhigh.
    pose proof (@low_top_one_is_unique selected Hlow) as Hone.
    pose proof (@high_top_one_is_unique selected Hhigh) as Hzero.
    rewrite Hone in Hzero; discriminate.
  Qed.

  Theorem ambiguous_interval_evidence_has_no_certified_exact_output :
    forall selected,
      ~ universally_safe_to_stop natural_certificate_order shared_evidence 1 selected.
  Proof.
    apply (@disjoint_feasible_answers_prevent_certified_output nat
      natural_certificate_order shared_evidence 1 low_completion high_completion).
    - exact low_model_is_feasible.
    - exact high_model_is_feasible.
    - exact no_selected_list_is_correct_in_both_models.
  Qed.

  Theorem deterministic_evidence_only_decider_must_defer :
    forall decider,
      decider_sound_at natural_certificate_order decider shared_evidence 1 ->
      decider shared_evidence 1 = None.
  Proof.
    intros decider Hsound.
    apply (@ambiguous_models_force_deferral nat natural_certificate_order
      shared_evidence 1 low_completion high_completion decider).
    - exact low_model_is_feasible.
    - exact high_model_is_feasible.
    - exact no_selected_list_is_correct_in_both_models.
    - exact Hsound.
  Qed.

  Corollary no_total_exact_decision_works_for_both_models :
    forall choose : @stopping_evidence nat -> nat -> list nat,
      ~ universally_safe_to_stop natural_certificate_order
        shared_evidence 1 (choose shared_evidence 1).
  Proof. intros; apply ambiguous_interval_evidence_has_no_certified_exact_output. Qed.

  (** A full verified heap exists here, but its existence alone does not
      reveal which of the two exact answers is required. *)
  Example ambiguous_evidence_already_has_a_full_verified_choice :
    full_verified_choice natural_certificate_order shared_evidence 1 [0] 0.
  Proof. apply singleton_verified_choice; reflexivity. Qed.

  Example separated_evidence_has_an_underfull_verified_choice :
    underfull_verified_choice natural_certificate_order separated_evidence 2 [0].
  Proof.
    split; [simpl; lia |].
    intros assignment Hmodel; change
      (best_k_over natural_certificate_order [0] assignment 2 [0]).
    constructor.
    - constructor; [simpl; tauto | constructor].
    - intros original Hin; exact Hin.
    - constructor; constructor.
    - simpl; lia.
    - intros original Hin Hnot; contradiction.
  Qed.

  Example no_better_unresolved_rank_does_not_fill_a_vacancy :
    evidence_consistent natural_certificate_order separated_evidence /\
    no_feasible_unresolved_influence natural_certificate_order separated_evidence 0 /\
    universally_safe_to_stop natural_certificate_order separated_evidence 1 [0] /\
    ~ universally_safe_to_stop natural_certificate_order separated_evidence 2 [0].
  Proof.
    split.
    - exists separated_completion; exact separated_evidence_has_a_feasible_completion.
    - split; [exact separated_evidence_has_no_unresolved_influence |].
      split; [exact inhabited_full_selection_is_safe |].
      exact inhabited_underfull_stopping_is_rejected.
  Qed.

  Example contradictory_scores_force_deferral : forall decider capacity,
    decider_sound_at natural_certificate_order decider contradictory_scores capacity ->
    decider contradictory_scores capacity = None.
  Proof.
    intros decider capacity Hsound;
      eapply (@inconsistent_evidence_forces_deferral nat natural_certificate_order);
      [exact contradictory_scores_have_no_completion | exact Hsound].
  Qed.

  Example impossible_interval_forces_deferral : forall decider capacity,
    decider_sound_at natural_certificate_order decider impossible_interval capacity ->
    decider impossible_interval capacity = None.
  Proof.
    intros decider capacity Hsound;
      eapply (@inconsistent_evidence_forces_deferral nat natural_certificate_order);
      [exact impossible_interval_has_no_completion | exact Hsound].
  Qed.
End InhabitedAmbiguity.
