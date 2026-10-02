(** * A finite complete stopping test for independent natural-cost intervals

    The executable test scans the explicit eligible identity list. A full
    verified best-k selection stops exactly when no unresolved lower-endpoint
    rank precedes its verified worst member. An underfull selection stops
    exactly when all eligible identities are verified. At capacity zero only
    the empty result is accepted.

    Correctness is conditional on admissible independent closed-interval
    evidence and the explicit verified-selection producer contract. This
    function does not validate interval coherence, the heap producer, or
    captured source identities. The consistency witness is constructed from
    admissibility rather than assumed in an endpoint-attainment premise.

    Costs and injective identity/tie keys are naturals. Finite scan evaluation
    is proved by structural recursion, with one abstract visit per supplied
    eligible list entry. That count excludes membership-scan and arithmetic
    costs and is not a Rust work or latency theorem. An inhabited relational
    control shows why completeness does not transfer to a stronger evidence
    language whose feasible assignments exclude the interval endpoints.
    No extraction, binary64 correspondence, or runtime optimality is claimed. *)

From Stdlib Require Import Arith Bool Lia List Sorting.Sorted.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder
  EvidenceStopping EvidenceIndistinguishability EvidenceAttainableCompletion.
Import ListNotations.
Set Implicit Arguments.

(** Both loops and natural comparisons are executable. The explicit scan
    makes its finite recursive argument visible independently of safety. *)
Fixpoint interval_finite_all (test : nat -> bool) (originals : list nat) : bool :=
  match originals with
  | [] => true
  | original :: rest => test original && interval_finite_all test rest
  end.

Lemma interval_finite_all_true_iff : forall test originals,
  interval_finite_all test originals = true <->
  forall original, In original originals -> test original = true.
Proof.
  intros test originals; induction originals as [|head rest IH]; simpl.
  - split; [intros _ original Hin; contradiction | intros _; reflexivity].
  - rewrite Bool.andb_true_iff, IH; split.
    + intros [Hhead Hrest] original [Hequal | Hin].
      * subst original; exact Hhead.
      * apply Hrest; exact Hin.
    + intro Hall; split.
      * apply Hall; now left.
      * intros original Hin; apply Hall; now right.
Qed.

(** This relation models an eager outer scan. Counting each entry once does
    not assert unit cost for its predicate or implementation evaluation. *)
Inductive interval_scan_evaluation (test : nat -> bool)
    : list nat -> bool -> nat -> Prop :=
| IntervalScanNil : interval_scan_evaluation test [] true 0
| IntervalScanCons : forall original rest answer visits,
    interval_scan_evaluation test rest answer visits ->
    interval_scan_evaluation test (original :: rest)
      (test original && answer) (S visits).

Theorem interval_finite_scan_terminates : forall test originals,
  interval_scan_evaluation test originals
    (interval_finite_all test originals) (length originals).
Proof.
  intros test originals; induction originals as [|head rest IH]; simpl.
  - constructor.
  - constructor; exact IH.
Qed.

Theorem interval_scan_evaluation_agrees_with_function :
  forall test originals answer visits,
    interval_scan_evaluation test originals answer visits ->
    answer = interval_finite_all test originals /\ visits = length originals.
Proof.
  intros test originals answer visits Hscan; induction Hscan.
  - split; reflexivity.
  - destruct IHHscan as [Hanswer Hvisits]; simpl; subst; split; reflexivity.
Qed.

Definition interval_verifiedb (evidence : @stopping_evidence nat)
    (original : nat) : bool :=
  if in_dec Nat.eq_dec original (verified_originals evidence)
  then true else false.

Lemma interval_verifiedb_true_iff : forall evidence original,
  interval_verifiedb evidence original = true <->
  In original (verified_originals evidence).
Proof.
  intros evidence original; unfold interval_verifiedb.
  destruct (in_dec Nat.eq_dec original (verified_originals evidence))
    as [Hin | Hnot]; split; intros H; try reflexivity; try assumption;
    try discriminate; contradiction.
Qed.

Definition interval_natural_rank_leb (first second : nat * nat) : bool :=
  (fst first <? fst second) ||
  ((fst first =? fst second) && (snd first <=? snd second)).

Lemma interval_natural_rank_leb_true_iff : forall first second,
  interval_natural_rank_leb first second = true <->
  generic_rank_le natural_certificate_order natural_certificate_order first second.
Proof.
  intros [first_cost first_tie] [second_cost second_tie].
  unfold interval_natural_rank_leb; simpl.
  rewrite Bool.orb_true_iff, Bool.andb_true_iff,
    Nat.ltb_lt, Nat.eqb_eq, Nat.leb_le.
  reflexivity.
Qed.

Definition interval_full_entryb (evidence : @stopping_evidence nat)
    (worst original : nat) : bool :=
  if interval_verifiedb evidence original then true
  else interval_natural_rank_leb
    (original_rank (endpoint_completion evidence) worst)
    (lower_endpoint evidence original, original).

Definition full_interval_stopb (evidence : @stopping_evidence nat)
    (worst : nat) : bool :=
  interval_finite_all (interval_full_entryb evidence worst)
    (eligible_originals evidence).

Definition underfull_interval_stopb (evidence : @stopping_evidence nat) : bool :=
  interval_finite_all (interval_verifiedb evidence) (eligible_originals evidence).

Lemma full_interval_stopb_spec : forall evidence worst,
  full_interval_stopb evidence worst = true <->
  forall original,
    In original (eligible_originals evidence) ->
    ~ In original (verified_originals evidence) ->
    generic_rank_le natural_certificate_order natural_certificate_order
      (original_rank (endpoint_completion evidence) worst)
      (lower_endpoint evidence original, original).
Proof.
  intros evidence worst; unfold full_interval_stopb.
  rewrite interval_finite_all_true_iff; split.
  - intros Hall original Heligible Hunverified.
    specialize (Hall original Heligible).
    unfold interval_full_entryb, interval_verifiedb in Hall.
    destruct (in_dec Nat.eq_dec original (verified_originals evidence))
      as [Hverified | Hunknown]; [contradiction |].
    now apply interval_natural_rank_leb_true_iff.
  - intros Hall original Heligible.
    unfold interval_full_entryb, interval_verifiedb.
    destruct (in_dec Nat.eq_dec original (verified_originals evidence))
      as [Hverified | Hunknown]; [reflexivity |].
    apply interval_natural_rank_leb_true_iff, Hall; assumption.
Qed.

Lemma underfull_interval_stopb_spec : forall evidence,
  underfull_interval_stopb evidence = true <->
  eligible_unresolved_exhausted evidence.
Proof.
  intro evidence; unfold underfull_interval_stopb,
    eligible_unresolved_exhausted, incl.
  rewrite interval_finite_all_true_iff; split; intros Hall original Hin.
  - apply interval_verifiedb_true_iff, Hall; exact Hin.
  - apply interval_verifiedb_true_iff, Hall; exact Hin.
Qed.

Theorem full_and_underfull_interval_scans_terminate : forall evidence worst,
  interval_scan_evaluation (interval_full_entryb evidence worst)
    (eligible_originals evidence) (full_interval_stopb evidence worst)
    (length (eligible_originals evidence)) /\
  interval_scan_evaluation (interval_verifiedb evidence)
    (eligible_originals evidence) (underfull_interval_stopb evidence)
    (length (eligible_originals evidence)).
Proof. intros; split; apply interval_finite_scan_terminates. Qed.

Lemma admissible_natural_interval_evidence_is_consistent : forall evidence,
  interval_evidence_admissible natural_certificate_order evidence ->
  evidence_consistent natural_certificate_order evidence.
Proof.
  intros evidence Hadmissible; exists (endpoint_completion evidence).
  apply admissible_evidence_constructs_endpoint_model; exact Hadmissible.
Qed.

Theorem full_interval_stopb_sound : forall evidence capacity selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  full_verified_choice natural_certificate_order evidence capacity selected worst ->
  full_interval_stopb evidence worst = true ->
  universally_safe_to_stop natural_certificate_order evidence capacity selected.
Proof.
  intros evidence capacity selected worst Hadmissible Hchoice Hstop.
  pose proof (@admissible_evidence_constructs_endpoint_model nat
    natural_certificate_order evidence Hadmissible) as Hendpoint_model.
  assert (Hconsistent : evidence_consistent natural_certificate_order evidence).
  { exists (endpoint_completion evidence); exact Hendpoint_model. }
  assert (Hworst_verified : In worst (verified_originals evidence)).
  { destruct Hchoice as [_ [_ [Hworst Hproducer]]].
    destruct (Hproducer (endpoint_completion evidence) Hendpoint_model)
      as [[_ Hselected_verified _ _ _] _].
    apply Hselected_verified; exact Hworst. }
  apply (proj2 (@bf4_full_safety_characterization nat natural_certificate_order
    evidence capacity selected worst Hconsistent Hchoice)).
  intros assignment Hmodel [original [Heligible [Hunverified Hbefore]]].
  pose proof (proj1 (full_interval_stopb_spec evidence worst)
    Hstop original Heligible Hunverified) as Hfloor.
  pose proof (@feasible_completions_agree_on_verified_ranks nat
    natural_certificate_order evidence (endpoint_completion evidence)
    assignment worst Hendpoint_model Hmodel Hworst_verified) as Hagree.
  destruct Hagree as [Hworst_cost _].
  change (endpoint_completion evidence worst = assignment worst) in Hworst_cost.
  destruct Hmodel as [_ [_ Hbounds]].
  specialize (Hbounds original Heligible).
  change (lower_endpoint evidence original <= assignment original /\
    assignment original <= upper_endpoint evidence original) in Hbounds.
  change (endpoint_completion evidence worst < lower_endpoint evidence original \/
    (endpoint_completion evidence worst = lower_endpoint evidence original /\
      worst <= original)) in Hfloor.
  change ((assignment original < assignment worst \/
    (assignment original = assignment worst /\ original <= worst)) /\
    ~ (assignment original = assignment worst /\ original = worst)) in Hbefore.
  lia.
Qed.

Theorem full_interval_stopb_complete : forall evidence capacity selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  full_verified_choice natural_certificate_order evidence capacity selected worst ->
  universally_safe_to_stop natural_certificate_order evidence capacity selected ->
  full_interval_stopb evidence worst = true.
Proof.
  intros evidence capacity selected worst Hadmissible Hchoice Hsafe.
  apply full_interval_stopb_spec; intros original Heligible Hunverified.
  pose proof (@unverified_endpoint_completion_is_lower nat evidence
    original Hunverified) as Hendpoint.
  rewrite <- Hendpoint.
  apply (@not_earlier_implies_reverse_order nat natural_certificate_order
    (endpoint_completion evidence) original worst).
  intro Hbefore.
  assert (Hrefusal : lower_endpoint_refusal natural_certificate_order
      evidence worst original).
  { split; [exact Heligible |]; split; [exact Hunverified |].
    unfold original_lt, original_rank in Hbefore.
    rewrite Hendpoint in Hbefore; exact Hbefore. }
  exact (@endpoint_refusal_blocks_universal_stopping nat natural_certificate_order
    evidence capacity selected worst original Hadmissible Hchoice Hrefusal Hsafe).
Qed.

Theorem full_interval_stopb_correct : forall evidence capacity selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  full_verified_choice natural_certificate_order evidence capacity selected worst ->
  (full_interval_stopb evidence worst = true <->
    universally_safe_to_stop natural_certificate_order evidence capacity selected).
Proof.
  intros evidence capacity selected worst Hadmissible Hchoice; split.
  - intro Hstop; eapply full_interval_stopb_sound; eauto.
  - intro Hsafe; eapply full_interval_stopb_complete; eauto.
Qed.

Theorem underfull_interval_stopb_correct : forall evidence capacity selected,
  interval_evidence_admissible natural_certificate_order evidence ->
  underfull_verified_choice natural_certificate_order evidence capacity selected ->
  (underfull_interval_stopb evidence = true <->
    universally_safe_to_stop natural_certificate_order evidence capacity selected).
Proof.
  intros evidence capacity selected Hadmissible Hchoice.
  rewrite underfull_interval_stopb_spec.
  apply iff_sym, underfull_safety_requires_exhaustion; [|exact Hchoice].
  apply admissible_natural_interval_evidence_is_consistent; exact Hadmissible.
Qed.

(** Dispatch performs no search beyond one finite scan. An overfull list is
    rejected. A supplied worst ID is consulted only in the positive full
    branch; its validity belongs to [interval_selection_produced] below. *)
Definition interval_stopb (evidence : @stopping_evidence nat) (capacity : nat)
    (selected : list nat) (worst : nat) : bool :=
  match capacity with
  | 0 => match selected with [] => true | _ :: _ => false end
  | S _ =>
      if length selected <? capacity then underfull_interval_stopb evidence
      else if length selected =? capacity then full_interval_stopb evidence worst
      else false
  end.

Theorem zero_capacity_interval_stopb_correct : forall evidence selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  (interval_stopb evidence 0 selected worst = true <->
    universally_safe_to_stop natural_certificate_order evidence 0 selected).
Proof.
  intros evidence selected worst Hadmissible.
  pose proof (@admissible_evidence_constructs_endpoint_model nat
    natural_certificate_order evidence Hadmissible) as Hmodel.
  split.
  - destruct selected as [|head rest]; [|discriminate].
    intros _; split.
    + exists (endpoint_completion evidence); exact Hmodel.
    + intros assignment Hassignment; apply zero_capacity_selects_nothing.
  - intros [_ Hsafe].
    destruct (Hsafe (endpoint_completion evidence) Hmodel) as [_ _ _ Hsize _].
    destruct selected as [|head rest]; [reflexivity | simpl in Hsize; lia].
Qed.

Theorem full_branch_interval_stopb_correct : forall evidence capacity selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  full_verified_choice natural_certificate_order evidence capacity selected worst ->
  (interval_stopb evidence capacity selected worst = true <->
    universally_safe_to_stop natural_certificate_order evidence capacity selected).
Proof.
  intros evidence capacity selected worst Hadmissible Hchoice.
  pose proof (proj1 Hchoice) as Hsize.
  pose proof (proj1 (proj2 Hchoice)) as Hpositive.
  destruct capacity as [|capacity]; [lia |].
  unfold interval_stopb; rewrite Hsize, Nat.ltb_irrefl, Nat.eqb_refl.
  apply full_interval_stopb_correct; assumption.
Qed.

Theorem underfull_branch_interval_stopb_correct :
  forall evidence capacity selected worst,
    interval_evidence_admissible natural_certificate_order evidence ->
    underfull_verified_choice natural_certificate_order evidence capacity selected ->
    (interval_stopb evidence capacity selected worst = true <->
      universally_safe_to_stop natural_certificate_order evidence capacity selected).
Proof.
  intros evidence capacity selected worst Hadmissible Hchoice.
  pose proof (proj1 Hchoice) as Hsize.
  destruct capacity as [|capacity]; [lia |].
  assert (Hunder : (length selected <? S capacity) = true).
  { apply Nat.ltb_lt; exact Hsize. }
  unfold interval_stopb; rewrite Hunder.
  apply underfull_interval_stopb_correct; assumption.
Qed.

Definition interval_selection_produced (evidence : @stopping_evidence nat)
    (capacity : nat) (selected : list nat) (worst : nat) : Prop :=
  (capacity = 0 /\ selected = []) \/
  full_verified_choice natural_certificate_order evidence capacity selected worst \/
  underfull_verified_choice natural_certificate_order evidence capacity selected.

Theorem interval_stopb_sound_and_complete : forall evidence capacity selected worst,
  interval_evidence_admissible natural_certificate_order evidence ->
  interval_selection_produced evidence capacity selected worst ->
  (interval_stopb evidence capacity selected worst = true <->
    universally_safe_to_stop natural_certificate_order evidence capacity selected).
Proof.
  intros evidence capacity selected worst Hadmissible
    [[Hzero Hempty] | [Hfull | Hunder]].
  - subst capacity; apply zero_capacity_interval_stopb_correct; exact Hadmissible.
  - apply full_branch_interval_stopb_correct; assumption.
  - apply underfull_branch_interval_stopb_correct; assumption.
Qed.

Module FiniteIntervalStoppingControls.
  Import EqualityBoundControls.

  Example equality_earlier_tie_boolean_refusal :
    interval_stopb (tied_evidence 1) 1 [1] 1 = false.
  Proof. reflexivity. Qed.

  Example equality_later_tie_boolean_acceptance :
    interval_stopb (tied_evidence 0) 1 [0] 0 = true.
  Proof. reflexivity. Qed.

  Example underfull_later_tie_boolean_refusal :
    interval_stopb (tied_evidence 0) 2 [0] 0 = false.
  Proof. reflexivity. Qed.

  Example zero_capacity_boolean_acceptance :
    interval_stopb (tied_evidence 0) 0 [] 99 = true.
  Proof. reflexivity. Qed.

  Example nonempty_zero_capacity_boolean_refusal :
    interval_stopb (tied_evidence 0) 0 [0] 99 = false.
  Proof. reflexivity. Qed.

  (** This explicit control records the validation boundary. The Boolean
      kernel alone can accept malformed evidence at capacity zero; its
      theorem therefore cannot be used without admissibility. *)
  Example inconsistent_evidence_is_outside_the_test_contract :
    interval_stopb NaturalEvidenceControls.contradictory_scores 0 [] 0 = true /\
    ~ universally_safe_to_stop natural_certificate_order
        NaturalEvidenceControls.contradictory_scores 0 [].
  Proof.
    split; [reflexivity |].
    apply NaturalEvidenceControls.contradictory_scores_cannot_authorize_any_result.
  Qed.
End FiniteIntervalStoppingControls.

Module RelationalEndpointLimitation.
  (** This genuine cross-original constraint requires the unresolved cost
      to exceed the recorded original's cost. It is absent from the interval
      evidence language used by the executable test. *)
  Definition relational_model (assignment : @completion nat) : Prop :=
    feasible_completion natural_certificate_order
      InhabitedAmbiguity.shared_evidence assignment /\
    assignment 0 < assignment 1.

  Lemma relational_control_has_admissible_intervals :
    interval_evidence_admissible natural_certificate_order
      InhabitedAmbiguity.shared_evidence.
  Proof.
    destruct InhabitedAmbiguity.low_model_is_feasible
      as [Hformed [Hscores Hbounds]].
    constructor.
    - exact Hformed.
    - intros original first second Hfirst Hsecond.
      pose proof (Hscores original first Hfirst) as Hfirst_score.
      pose proof (Hscores original second Hsecond) as Hsecond_score.
      change (InhabitedAmbiguity.low_completion original = first) in Hfirst_score.
      change (InhabitedAmbiguity.low_completion original = second) in Hsecond_score.
      change (first = second); lia.
    - intros original Hin; change (0 <= 10); lia.
    - intros original score Hin; simpl in Hin.
      destruct Hin as [Hequal | Hfalse]; [inversion Hequal; subst | contradiction].
      change (0 <= 5 /\ 5 <= 10); lia.
  Qed.

  Lemma relational_control_has_full_verified_selection :
    full_verified_choice natural_certificate_order
      InhabitedAmbiguity.shared_evidence 1 [0] 0.
  Proof. apply singleton_verified_choice; reflexivity. Qed.

  Theorem relational_evidence_is_inhabited :
    relational_model InhabitedAmbiguity.high_completion.
  Proof.
    split; [exact InhabitedAmbiguity.high_model_is_feasible |].
    change (5 < 8); lia.
  Qed.

  Theorem interval_endpoint_violates_relational_evidence :
    ~ relational_model (endpoint_completion InhabitedAmbiguity.shared_evidence).
  Proof. intros [_ Hrelation]; change (5 < 0) in Hrelation; lia. Qed.

  Theorem relational_models_all_preserve_verified_answer : forall assignment,
    relational_model assignment ->
    best_k_over natural_certificate_order [0; 1] assignment 1 [0].
  Proof.
    intros assignment [_ Hrelation].
    apply singleton_minimum_is_best_one; [simpl; auto |].
    intros other Hin; simpl in Hin.
    destruct Hin as [Hequal | [Hequal | Hfalse]];
      [subst other | subst other | contradiction].
    - apply original_le_reflexive.
    - change (assignment 0 < assignment 1 \/
        (assignment 0 = assignment 1 /\ 0 <= 1)); now left.
  Qed.

  Theorem interval_test_is_not_complete_for_this_relational_language :
    interval_evidence_admissible natural_certificate_order
      InhabitedAmbiguity.shared_evidence /\
    full_verified_choice natural_certificate_order
      InhabitedAmbiguity.shared_evidence 1 [0] 0 /\
    interval_stopb InhabitedAmbiguity.shared_evidence 1 [0] 0 = false /\
    (exists assignment, relational_model assignment) /\
    (forall assignment, relational_model assignment ->
      best_k_over natural_certificate_order [0; 1] assignment 1 [0]).
  Proof.
    split; [exact relational_control_has_admissible_intervals |].
    split; [exact relational_control_has_full_verified_selection |].
    split; [reflexivity |]; split.
    - exists InhabitedAmbiguity.high_completion;
        exact relational_evidence_is_inhabited.
    - exact relational_models_all_preserve_verified_answer.
  Qed.
End RelationalEndpointLimitation.
