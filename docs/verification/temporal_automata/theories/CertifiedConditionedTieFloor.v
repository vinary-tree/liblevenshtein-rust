(** * BF-2: equality-conditioned tie floors for certified best-k pruning

    A universal cost lower bound fixes the primary rank coordinate. Its
    secondary coordinate need bound ties only among candidates whose exact
    cost equals that lower bound. The equality slice may be empty even when
    the region is nonempty; neither the cost floor nor the rank floor needs
    to be attained.

    Costs and tie keys are exact naturals. Regions contain exact entries for
    semantic quantification, not a proposed runtime enumeration. Instances
    must supply sound cost and conditional-tie bounds, bind them to the
    captured region/scope and comparator, and discharge original/tie identity
    premises. The imported heap certificate represents verified exact ranks.
    No binary64, Rust heap, bound producer, or physical-resource refinement
    is established here. *)

From Stdlib Require Import Arith Lia List.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedKnnHeap CertifiedKnnPruning.
Import ListNotations.
Set Implicit Arguments.

(** The original field is ignored by [rank_le]. This synthetic bound is not
    claimed to be an actual candidate or an owned original occurrence. *)
Definition conditioned_rank_floor (lower tie_floor : nat) : ranked_entry :=
  {| entry_original := 0;
     entry_cost := lower;
     entry_tie := tie_floor |}.

Definition universal_cost_floor (lower : nat) (region : list ranked_entry)
    : Prop :=
  forall candidate, In candidate region -> lower <= entry_cost candidate.

Definition conditioned_tie_floor (lower tie_floor : nat)
    (region : list ranked_entry) : Prop :=
  forall candidate, In candidate region ->
    entry_cost candidate = lower -> tie_floor <= entry_tie candidate.

Definition equality_cost_slice (lower : nat) (region : list ranked_entry)
    : list ranked_entry :=
  filter (fun candidate => Nat.eqb (entry_cost candidate) lower) region.

Lemma equality_cost_slice_membership : forall lower region candidate,
  In candidate (equality_cost_slice lower region) <->
  In candidate region /\ entry_cost candidate = lower.
Proof.
  intros lower region candidate.
  unfold equality_cost_slice; rewrite filter_In, Nat.eqb_eq.
  reflexivity.
Qed.

Theorem bf2_exact_characterization : forall lower tie_floor region,
  universal_cost_floor lower region ->
  (region_summary_sound region (Known (conditioned_rank_floor lower tie_floor))
   <-> conditioned_tie_floor lower tie_floor region).
Proof.
  intros lower tie_floor region Hcost.
  unfold region_summary_sound, conditioned_tie_floor; split.
  - intros Hrank candidate Hin Hequal.
    specialize (Hrank candidate Hin).
    unfold rank_le, conditioned_rank_floor in Hrank; simpl in Hrank; lia.
  - intros Htie candidate Hin.
    pose proof (Hcost candidate Hin) as Hlower.
    specialize (Htie candidate Hin).
    unfold rank_le, conditioned_rank_floor; simpl; lia.
Qed.

Corollary conditioned_floor_is_sound : forall lower tie_floor region,
  universal_cost_floor lower region ->
  conditioned_tie_floor lower tie_floor region ->
  region_summary_sound region (Known (conditioned_rank_floor lower tie_floor)).
Proof.
  intros lower tie_floor region Hcost Htie.
  apply (proj2 (@bf2_exact_characterization lower tie_floor region Hcost)).
  exact Htie.
Qed.

Theorem conditioned_floor_is_exactly_a_slice_floor :
  forall lower tie_floor region,
    conditioned_tie_floor lower tie_floor region <->
    (forall candidate, In candidate (equality_cost_slice lower region) ->
      tie_floor <= entry_tie candidate).
Proof.
  intros lower tie_floor region; split.
  - intros Hfloor candidate Hin.
    apply equality_cost_slice_membership in Hin as [Hin Hequal].
    now apply Hfloor.
  - intros Hfloor candidate Hin Hequal.
    apply Hfloor, equality_cost_slice_membership; now split.
Qed.

(** An empty equality slice imposes no tie restriction. This is weaker than
    an empty region and must not be represented by the [Empty] summary. *)
Theorem empty_equality_slice_admits_every_tie_floor : forall lower region,
  equality_cost_slice lower region = [] ->
  forall tie_floor, conditioned_tie_floor lower tie_floor region.
Proof.
  intros lower region Hempty tie_floor candidate Hin Hequal.
  assert (Hslice : In candidate (equality_cost_slice lower region)).
  { apply equality_cost_slice_membership; now split. }
  rewrite Hempty in Hslice; contradiction.
Qed.

Corollary unattained_cost_floor_needs_no_minimum_tie : forall lower region,
  universal_cost_floor lower region ->
  equality_cost_slice lower region = [] ->
  forall tie_floor,
    region_summary_sound region (Known (conditioned_rank_floor lower tie_floor)).
Proof.
  intros lower region Hcost Hempty tie_floor.
  apply conditioned_floor_is_sound; [exact Hcost |].
  now apply empty_equality_slice_admits_every_tie_floor.
Qed.

Definition conditioned_prune_decision (capacity : nat)
    (selected : list ranked_entry) (lower tie_floor : nat) : bool :=
  prune_decision capacity selected
    (Known (conditioned_rank_floor lower tie_floor)).

Theorem conditioned_prune_requires_full_heap :
  forall capacity selected lower tie_floor,
    conditioned_prune_decision capacity selected lower tie_floor = true ->
    capacity > 0 /\ length selected = capacity.
Proof.
  intros capacity selected lower tie_floor Hprune.
  eapply known_summary_requires_full_heap; exact Hprune.
Qed.

Theorem conditioned_prune_is_disabled_underfull :
  forall capacity selected lower tie_floor,
    length selected < capacity ->
    conditioned_prune_decision capacity selected lower tie_floor = false.
Proof.
  intros capacity selected lower tie_floor Hunder.
  unfold conditioned_prune_decision, prune_decision.
  now rewrite (@kth_rank_underfull capacity selected Hunder).
Qed.

Theorem conditioned_prune_is_disabled_at_zero_capacity :
  forall selected lower tie_floor,
    conditioned_prune_decision 0 selected lower tie_floor = false.
Proof. reflexivity. Qed.

(** A [Some] kth rank supplies the full-heap gate. A best-k certificate also
    establishes that this rank is the maximum of the verified selection. *)
Theorem conditioned_floor_prunes_against_verified_kth :
  forall capacity verified selected rejected region lower tie_floor worst,
    best_k_certificate capacity verified selected rejected ->
    kth_rank capacity selected = Some worst ->
    universal_cost_floor lower region ->
    conditioned_tie_floor lower tie_floor region ->
    rank_le worst (conditioned_rank_floor lower tie_floor) ->
    conditioned_prune_decision capacity selected lower tie_floor = true /\
    (forall candidate, In candidate region ->
      ~ competitive capacity selected candidate) /\
    (forall retained candidate,
      In retained selected -> In candidate region -> rank_le retained candidate).
Proof.
  intros capacity verified selected rejected region lower tie_floor worst
    Hbest Hkth Hcost Htie Hthreshold.
  assert (Hsound : region_summary_sound region
    (Known (conditioned_rank_floor lower tie_floor))).
  { now apply conditioned_floor_is_sound. }
  assert (Hprune : conditioned_prune_decision capacity selected lower tie_floor
    = true).
  { unfold conditioned_prune_decision, prune_decision; rewrite Hkth.
    destruct (rank_le_dec worst (conditioned_rank_floor lower tie_floor));
      [reflexivity | contradiction]. }
  split; [exact Hprune |]; split.
  - eapply prune_decision_sound; [exact Hsound | exact Hprune].
  - eapply prune_selected_precedes_region;
      [exact Hbest | exact Hsound | exact Hprune].
Qed.

(** Preservation of the canonical original identities additionally needs
    freshness across the hidden region and the already verified entries.
    The reference union does not assert that the region was scored at runtime. *)
Theorem conditioned_prune_preserves_canonical_best_k :
  forall capacity verified selected rejected region lower tie_floor,
    best_k_certificate capacity verified selected rejected ->
    universal_cost_floor lower region ->
    conditioned_tie_floor lower tie_floor region ->
    conditioned_prune_decision capacity selected lower tie_floor = true ->
    NoDup (map entry_original (region ++ verified)) ->
    NoDup (map entry_tie (region ++ verified)) ->
    selected = firstn capacity (sort_ranked (region ++ verified)).
Proof.
  intros capacity verified selected rejected region lower tie_floor
    Hbest Hcost Htie Hprune Horiginal Hdistinct_tie.
  eapply pruned_region_cannot_change_best_k with
    (summary := Known (conditioned_rank_floor lower tie_floor));
    [exact Hbest | | exact Hprune | exact Horiginal | exact Hdistinct_tie].
  now apply conditioned_floor_is_sound.
Qed.

Module ConditionedFloorControls.
  Definition expensive : ranked_entry :=
    {| entry_original := 1; entry_cost := 6; entry_tie := 1 |}.
  Definition equal_cost : ranked_entry :=
    {| entry_original := 9; entry_cost := 5; entry_tie := 9 |}.
  Definition threshold : ranked_entry :=
    {| entry_original := 7; entry_cost := 5; entry_tie := 7 |}.
  Definition region := [expensive; equal_cost].

  Lemma region_has_cost_floor_five : universal_cost_floor 5 region.
  Proof.
    intros candidate [Hequal | [Hequal | []]]; subst candidate;
      unfold expensive, equal_cost; simpl; lia.
  Qed.

  Lemma region_has_conditioned_tie_floor_nine :
    conditioned_tie_floor 5 9 region.
  Proof.
    intros candidate [Hequal | [Hequal | []]] Heq; subst candidate;
      unfold expensive, equal_cost in *; simpl in *; lia.
  Qed.

  Lemma singleton_threshold_is_verified_best_k : forall capacity,
    0 < capacity -> best_k_certificate capacity [threshold] [threshold] [].
  Proof.
    intros capacity Hpositive.
    change (best_k_certificate capacity (threshold :: [])
      (insert_ranked threshold []) []).
    apply underfull_admission_preserves_best_k.
    - simpl; exact Hpositive.
    - apply empty_best_k_certificate.
    - simpl; tauto.
    - simpl; tauto.
  Qed.

  Example global_floor_one_but_conditioned_floor_nine :
    (forall candidate, In candidate region -> 1 <= entry_tie candidate) /\
    ~ (forall candidate, In candidate region -> 9 <= entry_tie candidate) /\
    equality_cost_slice 5 region = [equal_cost] /\
    conditioned_tie_floor 5 9 region.
  Proof.
    split.
    - intros candidate [Hequal | [Hequal | []]]; subst candidate;
        unfold expensive, equal_cost; simpl; lia.
    - split.
      + intro Hglobal.
        assert (Hin : In expensive region) by (left; reflexivity).
        specialize (Hglobal expensive Hin).
        unfold expensive in Hglobal; simpl in Hglobal; lia.
      + split; [reflexivity | exact region_has_conditioned_tie_floor_nine].
  Qed.

  Example conditioned_floor_enables_equality_pruning :
    best_k_certificate 1 [threshold] [threshold] [] /\
    kth_rank 1 [threshold] = Some threshold /\
    conditioned_prune_decision 1 [threshold] 5 1 = false /\
    conditioned_prune_decision 1 [threshold] 5 9 = true.
  Proof.
    split; [apply singleton_threshold_is_verified_best_k; lia |].
    repeat split; reflexivity.
  Qed.

  Example conditioned_floor_cannot_fill_a_vacant_slot_by_pruning :
    best_k_certificate 2 [threshold] [threshold] [] /\
    kth_rank 2 [threshold] = None /\
    conditioned_prune_decision 2 [threshold] 5 9 = false /\
    competitive 2 [threshold] equal_cost.
  Proof.
    split; [apply singleton_threshold_is_verified_best_k; lia |].
    split; [reflexivity |].
    split; [reflexivity | exact I].
  Qed.

  Example nonempty_region_with_unattained_cost_floor :
    [expensive] <> [] /\
    equality_cost_slice 5 [expensive] = [] /\
    (forall tie_floor, region_summary_sound [expensive]
      (Known (conditioned_rank_floor 5 tie_floor))).
  Proof.
    split; [discriminate |].
    split; [reflexivity |].
    apply unattained_cost_floor_needs_no_minimum_tie.
    - intros candidate [Hequal | []]; subst candidate;
        unfold expensive; simpl; lia.
    - reflexivity.
  Qed.

  Definition below_claimed_cost : ranked_entry :=
    {| entry_original := 4; entry_cost := 4; entry_tie := 9 |}.

  Example conditional_tie_without_universal_cost_bound_is_insufficient :
    conditioned_tie_floor 5 9 [below_claimed_cost] /\
    ~ region_summary_sound [below_claimed_cost]
      (Known (conditioned_rank_floor 5 9)).
  Proof.
    split.
    - intros candidate [Hequal | []] Hcost; subst candidate;
        unfold below_claimed_cost in Hcost; simpl in Hcost; lia.
    - intro Hbad.
      specialize (Hbad below_claimed_cost (or_introl eq_refl)).
      unfold rank_le, conditioned_rank_floor, below_claimed_cost in Hbad;
        simpl in Hbad; lia.
  Qed.
End ConditionedFloorControls.
