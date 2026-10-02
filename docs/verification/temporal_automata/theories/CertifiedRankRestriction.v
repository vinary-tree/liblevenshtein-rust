(** * Scoped restriction and combination of ordered rank facts

    A region is an explicitly scoped predicate over complete (cost,tie)
    ranks. A parent certificate restricts to a child only with both the
    same scope and an inclusion proof. Two lower-rank facts may be combined
    by the lawful lexicographic maximum when the target region is covered by
    both facts under that scope. Independent global cost and tie inequalities
    also permit a coordinatewise maximum, provided each selected coordinate
    is one of its two proved bounds and is above both in its carrier order.

    Equality-conditioned tie floors do not obey the coordinatewise rule.
    These are semantic natural/rational-order facts, not a binary64 bound
    producer, a heap certificate, or a Rust correspondence theorem. *)

From Stdlib Require Import Arith Lia.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder.

Section ScopedRankRestriction.
  Context {Scope Cost Tie : Type}.
  Variable cost_order : certificate_order Cost.
  Variable tie_order : certificate_order Tie.

  Record scoped_region := {
    region_scope : Scope;
    region_contains : Cost * Tie -> Prop
  }.

  Definition scoped_subset (parent child : scoped_region) : Prop :=
    region_scope child = region_scope parent /\
    forall candidate,
      region_contains child candidate -> region_contains parent candidate.

  Definition common_scope_coverage
      (target left right : scoped_region) : Prop :=
    region_scope target = region_scope left /\
    region_scope target = region_scope right /\
    forall candidate,
      region_contains target candidate ->
      region_contains left candidate /\ region_contains right candidate.

  Definition scoped_rank_floor (region : scoped_region)
      (floor : Cost * Tie) : Prop :=
    forall candidate, region_contains region candidate ->
      generic_rank_le cost_order tie_order floor candidate.

  Theorem parent_rank_certificate_restricts_to_scoped_child :
    forall parent child certificate,
      scoped_subset parent child ->
      certificate_sound cost_order tie_order
        (region_contains parent) certificate ->
      certificate_sound cost_order tie_order
        (region_contains child) certificate.
  Proof.
    intros parent child certificate [_ Hinclusion] Hsound
      candidate Hchild.
    apply Hsound, Hinclusion; exact Hchild.
  Qed.

  Theorem parent_rank_floor_restricts_to_scoped_child :
    forall parent child floor,
      scoped_subset parent child ->
      scoped_rank_floor parent floor -> scoped_rank_floor child floor.
  Proof.
    intros parent child floor [_ Hinclusion] Hfloor candidate Hchild.
    apply Hfloor, Hinclusion; exact Hchild.
  Qed.

  Theorem compatible_rank_facts_combine_by_lexicographic_maximum :
    forall target left right first_floor second_floor
      (comparator : lawful_rank_comparator cost_order tie_order),
      common_scope_coverage target left right ->
      scoped_rank_floor left first_floor ->
      scoped_rank_floor right second_floor ->
      scoped_rank_floor target
        (rank_maximum comparator first_floor second_floor).
  Proof.
    intros target left right first_floor second_floor comparator
      [_ [_ Hcoverage]] Hfirst Hsecond candidate Htarget.
    destruct (Hcoverage candidate Htarget) as [Hinleft Hinright].
    apply (@rank_maximum_is_least_upper_bound
      Cost Tie cost_order tie_order comparator first_floor second_floor
      candidate).
    - apply Hfirst; exact Hinleft.
    - apply Hsecond; exact Hinright.
  Qed.

  (** The equivalence is the exact criterion for a global certificate:
      its cost and tie claims hold independently over the same region. *)
  Theorem global_certificate_iff_independent_coordinate_floors :
    forall region lower_cost lower_tie,
      certificate_sound cost_order tie_order (region_contains region)
        (RankGlobal lower_cost lower_tie) <->
      (forall candidate, region_contains region candidate ->
        order_le cost_order lower_cost (fst candidate)) /\
      (forall candidate, region_contains region candidate ->
        order_le tie_order lower_tie (snd candidate)).
  Proof.
    intros region lower_cost lower_tie; split.
    - intros Hglobal; split; intros candidate Hin;
        specialize (Hglobal candidate Hin); simpl in Hglobal; tauto.
    - intros [Hcost Htie] candidate Hin.
      simpl; split; [apply Hcost | apply Htie]; exact Hin.
  Qed.

  Definition maximum_input_choice {Carrier : Type}
      (ordering : certificate_order Carrier)
      (first second chosen : Carrier) : Prop :=
    (chosen = first \/ chosen = second) /\
    order_le ordering first chosen /\
    order_le ordering second chosen.

  Theorem independent_global_facts_admit_coordinatewise_maximum :
    forall target left right
      first_cost first_tie second_cost second_tie
      chosen_cost chosen_tie,
      common_scope_coverage target left right ->
      certificate_sound cost_order tie_order (region_contains left)
        (RankGlobal first_cost first_tie) ->
      certificate_sound cost_order tie_order (region_contains right)
        (RankGlobal second_cost second_tie) ->
      maximum_input_choice cost_order first_cost second_cost chosen_cost ->
      maximum_input_choice tie_order first_tie second_tie chosen_tie ->
      certificate_sound cost_order tie_order (region_contains target)
        (RankGlobal chosen_cost chosen_tie).
  Proof.
    intros target left right first_cost first_tie second_cost second_tie
      chosen_cost chosen_tie [_ [_ Hcoverage]] Hfirst Hsecond
      Hcost_choice Htie_choice candidate Htarget.
    destruct (Hcoverage candidate Htarget) as [Hinleft Hinright].
    specialize (Hfirst candidate Hinleft).
    specialize (Hsecond candidate Hinright).
    simpl in Hfirst, Hsecond |- *.
    destruct Hcost_choice as [[Hchosen_cost | Hchosen_cost] _];
      destruct Htie_choice as [[Hchosen_tie | Hchosen_tie] _];
      subst; tauto.
  Qed.
End ScopedRankRestriction.

(** A real max-selector chooses one of its inputs. Nat.max supplies a
    concrete, executable instance of the separate coordinate requirement. *)
Lemma natural_maximum_is_an_input_choice : forall first second,
  maximum_input_choice natural_certificate_order first second
    (Nat.max first second).
Proof.
  intros first second.
  unfold maximum_input_choice; simpl.
  destruct (le_dec first second) as [Hle | Hgt].
  - rewrite Nat.max_r by lia.
    split; [now right | split; lia].
  - rewrite Nat.max_l by lia.
    split; [now left | split; lia].
Qed.

Example independent_global_maximum_is_sound_for_a_candidate :
  forall candidate : nat * nat,
    certificate_denotes natural_certificate_order
      natural_certificate_order (RankGlobal 5 1) candidate ->
    certificate_denotes natural_certificate_order
      natural_certificate_order (RankGlobal 6 0) candidate ->
    certificate_denotes natural_certificate_order
      natural_certificate_order (RankGlobal 6 1) candidate.
Proof.
  intros [cost tie] Hfirst Hsecond.
  simpl in *; lia.
Qed.

Example conditioned_coordinatewise_maximum_invents_an_unsound_floor :
  certificate_denotes natural_certificate_order natural_certificate_order
    (RankConditioned 5 9) (6, 1) /\
  certificate_denotes natural_certificate_order natural_certificate_order
    (RankConditioned 6 1) (6, 1) /\
  ~ certificate_denotes natural_certificate_order natural_certificate_order
    (RankConditioned 6 9) (6, 1) /\
  rank_maximum NaturalRankInstance.natural_rank_comparator
    (5, 9) (6, 1) = (6, 1).
Proof.
  split.
  - simpl; split; [lia | intro Hfalse; lia].
  - split.
    + simpl; split; [lia | intros _; lia].
    + split.
      * simpl; intros [_ Htie]; specialize (Htie eq_refl); lia.
      * reflexivity.
Qed.
