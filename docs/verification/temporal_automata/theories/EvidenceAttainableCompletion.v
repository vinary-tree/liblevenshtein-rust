(** * Constructing a feasible answer-changing completion from an endpoint

    This module closes the attainable-completion obligation for the specific
    independent closed-interval evidence language of EvidenceStopping. Its
    admissibility premises are structural validity, coherent exact recorded
    scores, ordered interval endpoints, and containment of every recorded
    score. Feasibility is a conclusion, not a hidden premise of the main
    construction. The witness assigns recorded scores to verified identities
    and lower endpoints to all unverified identities.

    For a full verified best-k selection, an unresolved lower-endpoint rank
    preceding the worst selected rank supplies an actual feasible completion
    in which that selection is not the completed best-k. At equal costs,
    an earlier tie permits the construction; a later tie does not. Underfull
    selections instead fail whenever an eligible unverified identity remains,
    independently of its rank.

    Costs use a supplied lawful order and identities are natural tie keys.
    The refusal predicate is mathematical; no Boolean comparator, heap
    producer, snapshot binding, or runtime evidence validator is implemented.
    The construction does not preserve additional relational or metric
    constraints absent from the declared evidence language. No Rust,
    floating-point, or universal runtime-optimality claim is made. *)

From Stdlib Require Import Arith Lia List Sorting.Sorted.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder
  EvidenceStopping EvidenceIndistinguishability.
Import ListNotations.
Set Implicit Arguments.

Section AttainableCompletion.
  Context {Cost : Type}.
  Variable cost_order : certificate_order Cost.

  Record interval_evidence_admissible (evidence : @stopping_evidence Cost) : Prop := {
    interval_identity_structure : evidence_well_formed evidence;
    interval_observations_coherent : verified_scores_coherent cost_order evidence;
    interval_endpoints_ordered : forall original,
      In original (eligible_originals evidence) ->
      order_le cost_order (lower_endpoint evidence original)
        (upper_endpoint evidence original);
    interval_observations_enclosed : forall original score,
      In (original, score) (verified_observations evidence) ->
      order_le cost_order (lower_endpoint evidence original) score /\
      order_le cost_order score (upper_endpoint evidence original)
  }.

  Lemma admissible_evidence_constructs_endpoint_model : forall evidence,
    interval_evidence_admissible evidence ->
    feasible_completion cost_order evidence (endpoint_completion evidence).
  Proof.
    intros evidence [Hidentity Hcoherent Hordered Henclosed].
    apply coherent_bounded_evidence_has_a_completion;
      [exact Hidentity | exact Hcoherent | exact Hordered | exact Henclosed].
  Qed.

  Lemma unverified_endpoint_completion_is_lower :
    forall (evidence : @stopping_evidence Cost) original,
    ~ In original (verified_originals evidence) ->
    endpoint_completion evidence original = lower_endpoint evidence original.
  Proof.
    intros evidence original Hunverified; unfold endpoint_completion.
    destruct (lookup_verified original (verified_observations evidence))
      as [score|] eqn:Hlookup; [|reflexivity].
    exfalso; apply Hunverified.
    unfold verified_originals; apply in_map_iff.
    exists (original, score); split; [reflexivity |].
    eapply lookup_verified_returns_observation; exact Hlookup.
  Qed.

  Definition lower_endpoint_refusal (evidence : @stopping_evidence Cost)
      (worst original : nat) : Prop :=
    In original (eligible_originals evidence) /\
    ~ In original (verified_originals evidence) /\
    generic_rank_strict cost_order natural_certificate_order
      (lower_endpoint evidence original, original)
      (original_rank (endpoint_completion evidence) worst).

  Lemma improving_unverified_original_invalidates_full_selection :
    forall evidence capacity selected worst assignment original,
      full_verified_choice cost_order evidence capacity selected worst ->
      feasible_completion cost_order evidence assignment ->
      In original (eligible_originals evidence) ->
      ~ In original (verified_originals evidence) ->
      original_lt cost_order assignment original worst ->
      ~ best_k_over cost_order (eligible_originals evidence)
          assignment capacity selected.
  Proof.
    intros evidence capacity selected worst assignment original
      [_ [_ [Hworst Hverified]]] Hmodel Heligible Hunverified Hbefore Hglobal.
    destruct (Hverified assignment Hmodel) as [[_ Hselected_verified _ _ _] _].
    assert (Hnotselected : ~ In original selected).
    { intro Hin; apply Hunverified, Hselected_verified; exact Hin. }
    destruct Hglobal as [_ _ _ _ Homitted].
    destruct (Homitted original Heligible Hnotselected) as [_ Hafter].
    eapply strict_order_conflicts_with_reverse; [exact Hbefore |].
    apply Hafter; exact Hworst.
  Qed.

  (** The constructed witness is exposed, including the score that attains
      the refusing lower endpoint. No pre-existing feasible model is assumed. *)
  Theorem endpoint_refusal_constructs_answer_changing_model :
    forall evidence capacity selected worst original,
      interval_evidence_admissible evidence ->
      full_verified_choice cost_order evidence capacity selected worst ->
      lower_endpoint_refusal evidence worst original ->
      feasible_completion cost_order evidence (endpoint_completion evidence) /\
      endpoint_completion evidence original = lower_endpoint evidence original /\
      original_lt cost_order (endpoint_completion evidence) original worst /\
      ~ best_k_over cost_order (eligible_originals evidence)
          (endpoint_completion evidence) capacity selected.
  Proof.
    intros evidence capacity selected worst original Hadmissible Hchoice
      [Heligible [Hunverified Hbefore]].
    pose proof (@admissible_evidence_constructs_endpoint_model evidence Hadmissible)
      as Hmodel.
    pose proof (@unverified_endpoint_completion_is_lower evidence original Hunverified)
      as Hendpoint.
    assert (Hstrict : original_lt cost_order
      (endpoint_completion evidence) original worst).
    { change (generic_rank_strict cost_order natural_certificate_order
        (endpoint_completion evidence original, original)
        (original_rank (endpoint_completion evidence) worst)).
      rewrite Hendpoint; exact Hbefore. }
    split; [exact Hmodel |]; split; [exact Hendpoint |]; split; [exact Hstrict |].
    eapply improving_unverified_original_invalidates_full_selection;
      [exact Hchoice | exact Hmodel | exact Heligible | exact Hunverified | exact Hstrict].
  Qed.

  Corollary lower_endpoint_refusal_has_attainable_completion :
    forall evidence capacity selected worst original,
      interval_evidence_admissible evidence ->
      full_verified_choice cost_order evidence capacity selected worst ->
      lower_endpoint_refusal evidence worst original ->
      exists assignment,
        feasible_completion cost_order evidence assignment /\
        assignment original = lower_endpoint evidence original /\
        unresolved_influence cost_order evidence assignment worst /\
        ~ best_k_over cost_order (eligible_originals evidence)
            assignment capacity selected.
  Proof.
    intros evidence capacity selected worst original Hadmissible Hchoice Hrefusal.
    destruct (@endpoint_refusal_constructs_answer_changing_model
      evidence capacity selected worst original Hadmissible Hchoice Hrefusal)
      as [Hmodel [Hattains [Hbefore Hnotbest]]].
    destruct Hrefusal as [Heligible [Hunverified _]].
    exists (endpoint_completion evidence).
    split; [exact Hmodel |]; split; [exact Hattains |]; split.
    - exists original; split; [exact Heligible |].
      split; [exact Hunverified | exact Hbefore].
    - exact Hnotbest.
  Qed.

  Corollary endpoint_refusal_blocks_universal_stopping :
    forall evidence capacity selected worst original,
      interval_evidence_admissible evidence ->
      full_verified_choice cost_order evidence capacity selected worst ->
      lower_endpoint_refusal evidence worst original ->
      ~ universally_safe_to_stop cost_order evidence capacity selected.
  Proof.
    intros evidence capacity selected worst original Hadmissible Hchoice Hrefusal
      [_ Hsafe].
    destruct (@lower_endpoint_refusal_has_attainable_completion
      evidence capacity selected worst original Hadmissible Hchoice Hrefusal)
      as [assignment [Hmodel [_ [_ Hnotbest]]]].
    apply Hnotbest, Hsafe; exact Hmodel.
  Qed.

  Theorem equal_cost_refusal_iff_earlier_tie :
    forall lower worst_cost original worst,
      order_equiv cost_order lower worst_cost ->
      (generic_rank_strict cost_order natural_certificate_order
        (lower, original) (worst_cost, worst) <-> original < worst).
  Proof.
    intros lower worst_cost original worst Hequal.
    rewrite (@rank_strict_is_lexicographic Cost nat cost_order
      natural_certificate_order (lower, original) (worst_cost, worst)).
    change ((order_lt cost_order lower worst_cost \/
      (order_equiv cost_order lower worst_cost /\ original < worst)) <->
      original < worst).
    split.
    - intros [Hless | [_ Htie]]; [|exact Htie].
      destruct (proj1 (order_lt_cases cost_order lower worst_cost) Hless)
        as [_ Hnotequal]; contradiction.
    - intro Htie; constructor 2; split; [exact Hequal | exact Htie].
  Qed.

  Corollary equal_cost_later_tie_cannot_refuse :
    forall lower worst_cost original worst,
      order_equiv cost_order lower worst_cost -> worst <= original ->
      ~ generic_rank_strict cost_order natural_certificate_order
          (lower, original) (worst_cost, worst).
  Proof.
    intros lower worst_cost original worst Hequal Hlater Hstrict.
    apply (proj1 (@equal_cost_refusal_iff_earlier_tie
      lower worst_cost original worst Hequal)) in Hstrict; lia.
  Qed.

  (** A vacancy has no kth-rank condition. An eligible unverified original
      prevents completion even when its attainable cost is very large. *)
  Theorem underfull_vacancy_constructs_answer_changing_model :
    forall evidence capacity selected original,
      interval_evidence_admissible evidence ->
      underfull_verified_choice cost_order evidence capacity selected ->
      In original (eligible_originals evidence) ->
      ~ In original (verified_originals evidence) ->
      feasible_completion cost_order evidence (endpoint_completion evidence) /\
      ~ best_k_over cost_order (eligible_originals evidence)
          (endpoint_completion evidence) capacity selected.
  Proof.
    intros evidence capacity selected original Hadmissible [Hunder Hverified]
      Heligible Hunverified.
    pose proof (@admissible_evidence_constructs_endpoint_model evidence Hadmissible)
      as Hmodel.
    split; [exact Hmodel |].
    intro Hglobal.
    destruct (Hverified (endpoint_completion evidence) Hmodel)
      as [_ Hselected_verified _ _ _].
    assert (Hnotselected : ~ In original selected).
    { intro Hin; apply Hunverified, Hselected_verified; exact Hin. }
    destruct Hglobal as [_ _ _ _ Homitted].
    destruct (Homitted original Heligible Hnotselected) as [Hfull _]; lia.
  Qed.
End AttainableCompletion.

Module EqualityBoundControls.
  (** Either identity can be the sole verified original. The other has
      attainable cost five, and can also take any higher cost through ten. *)
  Definition tied_evidence (verified : nat) : @stopping_evidence nat :=
    {| eligible_originals := [0; 1];
       verified_observations := [(verified, 5)];
       lower_endpoint := fun _ => 5; upper_endpoint := fun _ => 10 |}.

  Lemma tied_evidence_is_admissible : forall verified,
    In verified [0; 1] ->
    interval_evidence_admissible natural_certificate_order (tied_evidence verified).
  Proof.
    intros verified Hin; constructor.
    - split.
      + constructor; [simpl; lia |].
        constructor; [simpl; tauto | constructor].
      + intros original Hobserved; simpl in Hobserved.
        destruct Hobserved as [Hequal | Hfalse]; [subst original; exact Hin | contradiction].
    - intros original first second Hfirst Hsecond.
      simpl in Hfirst, Hsecond.
      destruct Hfirst as [Hfirst | Hfalse]; [|contradiction].
      destruct Hsecond as [Hsecond | Hfalse]; [|contradiction].
      inversion Hfirst; inversion Hsecond; subst; reflexivity.
    - intros original Heligible; change (5 <= 10); lia.
    - intros original score Hobserved; simpl in Hobserved.
      destruct Hobserved as [Hequal | Hfalse]; [inversion Hequal; subst | contradiction].
      change (5 <= 5 /\ 5 <= 10); lia.
  Qed.

  Lemma tied_selection_is_full : forall verified,
    full_verified_choice natural_certificate_order
      (tied_evidence verified) 1 [verified] verified.
  Proof. intro; apply singleton_verified_choice; reflexivity. Qed.

  Example equality_bound_earlier_tie_refuses :
    lower_endpoint_refusal natural_certificate_order (tied_evidence 1) 1 0.
  Proof.
    split; [simpl; auto |]; split; [simpl; lia |].
    change ((5 < 5 \/ (5 = 5 /\ 0 <= 1)) /\ ~ (5 = 5 /\ 0 = 1)); lia.
  Qed.

  Theorem equality_bound_earlier_tie_constructs_counterexample :
    exists assignment,
      feasible_completion natural_certificate_order (tied_evidence 1) assignment /\
      assignment 0 = 5 /\
      unresolved_influence natural_certificate_order (tied_evidence 1) assignment 1 /\
      ~ best_k_over natural_certificate_order [0; 1] assignment 1 [1].
  Proof.
    apply (@lower_endpoint_refusal_has_attainable_completion nat
      natural_certificate_order (tied_evidence 1) 1 [1] 1 0).
    - apply tied_evidence_is_admissible; simpl; auto.
    - apply tied_selection_is_full.
    - exact equality_bound_earlier_tie_refuses.
  Qed.

  (** This control also constructs a different completed answer; it does
      not merely assert failure of the old one. *)
  Example equality_bound_earlier_tie_has_new_exact_answer :
    feasible_completion natural_certificate_order (tied_evidence 1)
      (endpoint_completion (tied_evidence 1)) /\
    best_k_over natural_certificate_order [0; 1]
      (endpoint_completion (tied_evidence 1)) 1 [0] /\
    [0] <> [1].
  Proof.
    split.
    - apply admissible_evidence_constructs_endpoint_model.
      apply tied_evidence_is_admissible; simpl; auto.
    - split; [|discriminate].
      apply singleton_minimum_is_best_one; [simpl; auto |].
      intros other Hin; simpl in Hin.
      destruct Hin as [Hequal | [Hequal | Hfalse]];
        [subst other | subst other | contradiction].
      + apply original_le_reflexive.
      + change (5 < 5 \/ (5 = 5 /\ 0 <= 1)); lia.
  Qed.

  Example equality_bound_later_tie_does_not_refuse :
    ~ lower_endpoint_refusal natural_certificate_order (tied_evidence 0) 0 1.
  Proof.
    intros [_ [_ Hbefore]].
    change ((5 < 5 \/ (5 = 5 /\ 1 <= 0)) /\ ~ (5 = 5 /\ 1 = 0)) in Hbefore.
    lia.
  Qed.

  Theorem equality_bound_later_tie_is_safe :
    universally_safe_to_stop natural_certificate_order (tied_evidence 0) 1 [0].
  Proof.
    assert (Hadmissible : interval_evidence_admissible
      natural_certificate_order (tied_evidence 0)).
    { apply tied_evidence_is_admissible; simpl; auto. }
    assert (Hconsistent : evidence_consistent natural_certificate_order (tied_evidence 0)).
    { exists (endpoint_completion (tied_evidence 0)).
      now apply admissible_evidence_constructs_endpoint_model. }
    apply (proj2 (@bf4_full_safety_characterization nat natural_certificate_order
      (tied_evidence 0) 1 [0] 0 Hconsistent (tied_selection_is_full 0))).
    intros assignment [_ [Hscores Hbounds]]
      [original [Heligible [Hunverified Hbefore]]].
    simpl in Heligible.
    destruct Heligible as [Hequal | [Hequal | Hfalse]];
      [subst original | subst original | contradiction].
    - apply Hunverified; simpl; auto.
    - pose proof (Hscores 0 5 (or_introl eq_refl)) as Hzero.
      pose proof (Hbounds 1 (or_intror (or_introl eq_refl))) as Hone.
      change (assignment 0 = 5) in Hzero.
      change (5 <= assignment 1 /\ assignment 1 <= 10) in Hone.
      change ((assignment 1 < assignment 0 \/
        (assignment 1 = assignment 0 /\ 1 <= 0)) /\
        ~ (assignment 1 = assignment 0 /\ 1 = 0)) in Hbefore; lia.
  Qed.

  Lemma later_tie_selection_is_underfull :
    underfull_verified_choice natural_certificate_order (tied_evidence 0) 2 [0].
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

  Example later_tie_still_fills_an_underfull_vacancy :
    feasible_completion natural_certificate_order (tied_evidence 0)
      (endpoint_completion (tied_evidence 0)) /\
    ~ best_k_over natural_certificate_order [0; 1]
      (endpoint_completion (tied_evidence 0)) 2 [0].
  Proof.
    apply (@underfull_vacancy_constructs_answer_changing_model nat
      natural_certificate_order (tied_evidence 0) 2 [0] 1).
    - apply tied_evidence_is_admissible; simpl; auto.
    - exact later_tie_selection_is_underfull.
    - simpl; auto.
    - simpl; lia.
  Qed.
End EqualityBoundControls.
