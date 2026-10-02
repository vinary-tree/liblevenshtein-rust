(** * Ordered rank certificates, including a strict cost cut

    Certificate syntax is data, not proof of its claim. [certificate_sound]
    binds a certificate to a semantic region. [certificate_floor] interprets
    what lower ranks follow from that data, and [certificate_interpretation_sound]
    proves that those interpretations are sound for the region.

    The order interface permits quotient equality, as required by rational
    costs. No cost successor, top tie key, or numeric sentinel is assumed.
    Unknown information, an empty region, independent global floors,
    equality-conditioned floors, and a strict primary-cost cut have separate
    constructors and denotations. Regions are predicates and may be infinite;
    no finiteness or attainment assumption is hidden in the dense example.

    Source instances must validate the order/comparator and the semantic
    region claims. This is not an executable certificate checker, a heap-full
    gate, a Rust correspondence proof, or a floating-point model. *)

From Stdlib Require Import Arith Lia QArith micromega.Lqa.
From Liblevenshtein.TemporalAutomata Require Import CertifiedKnnHeap.
Set Implicit Arguments.

(** Total order on equivalence classes. The fields are premises supplied and
    proved by each instance; they are not project-level axioms. *)
Record certificate_order (Carrier : Type) := {
  order_equiv : Carrier -> Carrier -> Prop;
  order_le : Carrier -> Carrier -> Prop;
  order_lt : Carrier -> Carrier -> Prop;
  order_equiv_refl : forall x, order_equiv x x;
  order_equiv_sym : forall x y, order_equiv x y -> order_equiv y x;
  order_equiv_trans : forall x y z,
    order_equiv x y -> order_equiv y z -> order_equiv x z;
  order_le_refl : forall x, order_le x x;
  order_le_trans : forall x y z, order_le x y -> order_le y z -> order_le x z;
  order_le_total : forall x y, order_le x y \/ order_le y x;
  order_le_antisym : forall x y,
    order_le x y -> order_le y x -> order_equiv x y;
  order_le_cases : forall x y,
    order_le x y <-> order_lt x y \/ order_equiv x y;
  order_lt_cases : forall x y,
    order_lt x y <-> order_le x y /\ ~ order_equiv x y
}.

Arguments order_equiv {Carrier} _ _ _.
Arguments order_le {Carrier} _ _ _.
Arguments order_lt {Carrier} _ _ _.

Section OrderConsequences.
  Context {Carrier : Type} (ordering : certificate_order Carrier).

  Lemma certificate_equiv_implies_le : forall x y,
    order_equiv ordering x y -> order_le ordering x y.
  Proof.
    intros x y Hequal.
    apply (proj2 (order_le_cases ordering x y)); now right.
  Qed.

  Lemma certificate_lt_implies_le : forall x y,
    order_lt ordering x y -> order_le ordering x y.
  Proof.
    intros x y Hless.
    exact (proj1 (proj1 (order_lt_cases ordering x y) Hless)).
  Qed.

  Lemma certificate_lt_irreflexive : forall x, ~ order_lt ordering x x.
  Proof.
    intros x Hless.
    destruct (proj1 (order_lt_cases ordering x x) Hless) as [_ Hneq].
    apply Hneq, order_equiv_refl.
  Qed.

  Lemma certificate_le_lt_trans : forall x y z,
    order_le ordering x y -> order_lt ordering y z -> order_lt ordering x z.
  Proof.
    intros x y z Hxy Hyz.
    destruct (proj1 (order_lt_cases ordering y z) Hyz) as [Hyz_le Hyz_neq].
    apply (proj2 (order_lt_cases ordering x z)); split.
    - eapply order_le_trans; eauto.
    - intro Hxz; apply Hyz_neq.
      apply order_le_antisym; [exact Hyz_le |].
      eapply order_le_trans; [|exact Hxy].
      apply certificate_equiv_implies_le, order_equiv_sym; exact Hxz.
  Qed.

  Lemma certificate_lt_le_trans : forall x y z,
    order_lt ordering x y -> order_le ordering y z -> order_lt ordering x z.
  Proof.
    intros x y z Hxy Hyz.
    destruct (proj1 (order_lt_cases ordering x y) Hxy) as [Hxy_le Hxy_neq].
    apply (proj2 (order_lt_cases ordering x z)); split.
    - eapply order_le_trans; eauto.
    - intro Hxz; apply Hxy_neq.
      apply order_le_antisym; [exact Hxy_le |].
      eapply order_le_trans; [exact Hyz |].
      apply certificate_equiv_implies_le, order_equiv_sym; exact Hxz.
  Qed.
End OrderConsequences.

Definition generic_rank_le {Cost Tie : Type}
    (cost_order : certificate_order Cost) (tie_order : certificate_order Tie)
    (left right : Cost * Tie) : Prop :=
  order_lt cost_order (fst left) (fst right) \/
  order_equiv cost_order (fst left) (fst right) /\
    order_le tie_order (snd left) (snd right).

Lemma generic_rank_le_reflexive : forall Cost Tie
    (cost_order : certificate_order Cost) (tie_order : certificate_order Tie)
    (rank : Cost * Tie),
  generic_rank_le cost_order tie_order rank rank.
Proof.
  intros; right; split; [apply order_equiv_refl | apply order_le_refl].
Qed.

Lemma generic_rank_le_transitive : forall Cost Tie
    (cost_order : certificate_order Cost) (tie_order : certificate_order Tie)
    (first middle last : Cost * Tie),
  generic_rank_le cost_order tie_order first middle ->
  generic_rank_le cost_order tie_order middle last ->
  generic_rank_le cost_order tie_order first last.
Proof.
  intros Cost Tie cost_order tie_order first middle last Hfirst Hlast.
  destruct Hfirst as [Hfirst | [Hfirst Hfirst_tie]];
    destruct Hlast as [Hlast | [Hlast Hlast_tie]].
  - left; eapply certificate_le_lt_trans;
      [eapply certificate_lt_implies_le; exact Hfirst | exact Hlast].
  - left; eapply certificate_lt_le_trans;
      [exact Hfirst | apply certificate_equiv_implies_le; exact Hlast].
  - left; eapply certificate_le_lt_trans;
      [apply certificate_equiv_implies_le; exact Hfirst | exact Hlast].
  - right; split.
    + eapply order_equiv_trans; eauto.
    + eapply order_le_trans; eauto.
Qed.

Inductive rank_certificate (Cost Tie : Type) : Type :=
| RankUnknown
| RankEmpty
| RankGlobal (lower : Cost) (tie_floor : Tie)
| RankConditioned (lower : Cost) (tie_floor : Tie)
| RankStrictCost (lower : Cost).

Arguments RankUnknown {Cost Tie}.
Arguments RankEmpty {Cost Tie}.
Arguments RankGlobal {Cost Tie} _ _.
Arguments RankConditioned {Cost Tie} _ _.
Arguments RankStrictCost {Cost Tie} _.

Section Interpretation.
  Context {Cost Tie : Type}.
  Variable cost_order : certificate_order Cost.
  Variable tie_order : certificate_order Tie.

  Definition certificate_denotes (certificate : rank_certificate Cost Tie)
      (candidate : Cost * Tie) : Prop :=
    match certificate with
    | RankUnknown => True
    | RankEmpty => False
    | RankGlobal lower tie_floor =>
        order_le cost_order lower (fst candidate) /\
        order_le tie_order tie_floor (snd candidate)
    | RankConditioned lower tie_floor =>
        order_le cost_order lower (fst candidate) /\
        (order_equiv cost_order lower (fst candidate) ->
         order_le tie_order tie_floor (snd candidate))
    | RankStrictCost lower => order_lt cost_order lower (fst candidate)
    end.

  Definition certificate_sound (region : Cost * Tie -> Prop)
      (certificate : rank_certificate Cost Tie) : Prop :=
    forall candidate, region candidate -> certificate_denotes certificate candidate.

  (** A strict cut supports every threshold whose cost is at most the cut,
      regardless of its tie. This is deliberately not a synthetic rank. *)
  Definition certificate_floor (certificate : rank_certificate Cost Tie)
      (threshold : Cost * Tie) : Prop :=
    match certificate with
    | RankUnknown => False
    | RankEmpty => True
    | RankGlobal lower tie_floor | RankConditioned lower tie_floor =>
        generic_rank_le cost_order tie_order threshold (lower, tie_floor)
    | RankStrictCost lower => order_le cost_order (fst threshold) lower
    end.

  Theorem conditioned_denotation_is_lex_floor : forall lower tie_floor candidate,
    certificate_denotes (RankConditioned lower tie_floor) candidate <->
    generic_rank_le cost_order tie_order (lower, tie_floor) candidate.
  Proof.
    intros lower tie_floor [cost tie]; simpl.
    unfold generic_rank_le; simpl; split.
    - intros [Hcost Htie].
      destruct (proj1 (order_le_cases cost_order lower cost) Hcost)
        as [Hless | Hequal].
      + now left.
      + right; split; [exact Hequal | now apply Htie].
    - intros [Hless | [Hequal Htie]].
      + split; [now apply certificate_lt_implies_le |].
        intro Hequal; exfalso.
        destruct (proj1 (order_lt_cases cost_order lower cost) Hless)
          as [_ Hneq].
        contradiction.
      + split; [now apply certificate_equiv_implies_le |].
        intro; exact Htie.
  Qed.

  Theorem global_denotation_implies_conditioned : forall lower tie_floor candidate,
    certificate_denotes (RankGlobal lower tie_floor) candidate ->
    certificate_denotes (RankConditioned lower tie_floor) candidate.
  Proof. intros lower tie_floor candidate [Hcost Htie]; split; auto. Qed.

  Theorem strict_denotation_implies_any_conditioned_floor :
    forall lower tie_floor candidate,
      certificate_denotes (RankStrictCost lower) candidate ->
      certificate_denotes (RankConditioned lower tie_floor) candidate.
  Proof.
    intros lower tie_floor candidate Hstrict.
    apply conditioned_denotation_is_lex_floor; now left.
  Qed.

  Theorem certificate_interpretation_sound : forall region certificate threshold,
    certificate_sound region certificate ->
    certificate_floor certificate threshold ->
    forall candidate, region candidate ->
      generic_rank_le cost_order tie_order threshold candidate.
  Proof.
    intros region certificate threshold Hsound Hfloor candidate Hin.
    specialize (Hsound candidate Hin).
    destruct certificate as [| |lower tie_floor|lower tie_floor|lower];
      simpl in Hsound, Hfloor.
    - contradiction.
    - contradiction.
    - eapply generic_rank_le_transitive; [exact Hfloor |].
      apply conditioned_denotation_is_lex_floor.
      apply global_denotation_implies_conditioned; exact Hsound.
    - eapply generic_rank_le_transitive; [exact Hfloor |].
      now apply conditioned_denotation_is_lex_floor.
    - left; eapply certificate_le_lt_trans; eauto.
  Qed.

  Theorem unknown_is_sound_for_every_region : forall region,
    certificate_sound region RankUnknown.
  Proof. intros region candidate Hin; exact I. Qed.

  Theorem empty_is_sound_iff_region_is_empty : forall region,
    certificate_sound region RankEmpty <->
    (forall candidate, ~ region candidate).
  Proof. intros region; split; intros H candidate Hin; exact (H candidate Hin). Qed.

  Corollary inhabited_region_cannot_have_empty_certificate :
    forall (region : Cost * Tie -> Prop) candidate,
    region candidate -> ~ certificate_sound region RankEmpty.
  Proof. intros region candidate Hin Hempty; exact (Hempty candidate Hin). Qed.

  (** Any ordinary tie key admits its own equality-cost boundary point.
      Even a greatest element of the actual tie carrier cannot encode d>L. *)
  Theorem ordinary_tie_floor_is_not_a_strict_cut : forall lower tie_floor,
    certificate_denotes (RankConditioned lower tie_floor) (lower, tie_floor) /\
    ~ certificate_denotes (RankStrictCost lower) (lower, tie_floor).
  Proof.
    intros lower tie_floor; split.
    - apply conditioned_denotation_is_lex_floor, generic_rank_le_reflexive.
    - apply certificate_lt_irreflexive.
  Qed.
End Interpretation.

(** This natural specialization agrees exactly with the existing heap order. *)
Definition natural_certificate_order : certificate_order nat.
Proof.
  refine {| order_equiv := @eq nat; order_le := Nat.le; order_lt := Nat.lt |};
    intros; lia.
Defined.

Theorem natural_rank_semantics_agree_with_heap : forall left right,
  generic_rank_le natural_certificate_order natural_certificate_order
    (entry_cost left, entry_tie left) (entry_cost right, entry_tie right) <->
  rank_le left right.
Proof. intros; reflexivity. Qed.

Module DenseRationalCosts.
  Local Open Scope Q_scope.

  Definition rational_certificate_order : certificate_order Q.
  Proof.
    refine {| order_equiv := Qeq; order_le := Qle; order_lt := Qlt |}.
    - exact Qeq_refl.
    - exact Qeq_sym.
    - exact Qeq_trans.
    - exact Qle_refl.
    - exact Qle_trans.
    - intros left right.
      destruct (Qlt_le_dec left right) as [Hless | Hreverse].
      + left; now apply Qlt_le_weak.
      + right; exact Hreverse.
    - exact Qle_antisym.
    - exact Qle_lteq.
    - exact Qlt_leneq.
  Defined.

  Definition denotes :=
    certificate_denotes rational_certificate_order natural_certificate_order.
  Definition sound :=
    certificate_sound rational_certificate_order natural_certificate_order.
  Definition rank_lower :=
    generic_rank_le rational_certificate_order natural_certificate_order.

  Definition above_cut (lower : Q) (candidate : Q * nat) : Prop :=
    lower < fst candidate.

  Theorem rational_costs_are_dense : forall lower upper : Q,
    lower < upper -> exists middle, lower < middle /\ middle < upper.
  Proof.
    intros lower upper Hless.
    exists ((lower + upper) / 2); split.
    - apply Qlt_shift_div_l.
      + reflexivity.
      + setoid_replace (lower * 2) with (lower + lower) by ring.
        exact (proj2 (Qplus_lt_r lower upper lower) Hless).
    - apply Qlt_shift_div_r.
      + reflexivity.
      + setoid_replace (upper * 2) with (upper + upper) by ring.
        exact (proj2 (Qplus_lt_l lower upper upper) Hless).
  Qed.

  Theorem strict_certificate_for_the_entire_open_region : forall lower,
    sound (above_cut lower) (RankStrictCost lower).
  Proof. intros lower candidate Hin; exact Hin. Qed.

  Example strict_cut_accepts_fractional_cost :
    denotes (RankStrictCost 0) ((1#2), 0%nat).
  Proof. change (0 < (1#2)); lra. Qed.

  Example conditioned_floor_need_not_bound_global_ties :
    denotes (RankConditioned 5 9%nat) (6, 1%nat) /\
    ~ denotes (RankGlobal 5 9%nat) (6, 1%nat).
  Proof.
    split.
    - change ((5 <= 6) /\ (5 == 6 -> (9 <= 1)%nat)).
      split; [lra | intro Hfalse; exfalso; lra].
    - intros [_ Htie].
      change (9 <= 1)%nat in Htie; lia.
  Qed.

  Theorem strict_cut_bounds_every_tie_at_its_cost : forall lower threshold_tie,
    forall candidate, above_cut lower candidate ->
      rank_lower (lower, threshold_tie) candidate.
  Proof.
    intros lower threshold_tie candidate Hin; left; exact Hin.
  Qed.

  (** A fabricated next cost always overlooks a rational between the cut
      and that value. Therefore advancing the cost is not a sound encoding. *)
  Theorem every_proposed_next_cost_has_a_counterexample :
    forall lower proposed_next tie,
      lower < proposed_next ->
      exists candidate : Q * nat,
        denotes (RankStrictCost lower) candidate /\
        ~ rank_lower (proposed_next, tie) candidate.
  Proof.
    intros lower proposed_next tie Hnext.
    destruct (@rational_costs_are_dense lower proposed_next Hnext)
      as [middle [Hab Hbc]].
    exists (middle, tie); split; [exact Hab |].
    unfold rank_lower, generic_rank_le; simpl.
    intros [Hreverse | [Hequal Htie]]; lra.
  Qed.

  Example advancing_zero_to_one_loses_a_valid_candidate :
    denotes (RankStrictCost 0) ((1#2), 0%nat) /\
    ~ rank_lower (1, 0%nat) ((1#2), 0%nat).
  Proof.
    split; [exact strict_cut_accepts_fractional_cost |].
    unfold rank_lower, generic_rank_le; simpl.
    intros [Hreverse | [Hequal Htie]]; lra.
  Qed.

  Theorem natural_ties_have_no_greatest_key : forall greatest : nat,
    exists larger, (greatest < larger)%nat.
  Proof. intro greatest; exists (S greatest); lia. Qed.

  Theorem no_ordinary_tie_encodes_the_strict_cut : forall lower tie,
    denotes (RankConditioned lower tie) (lower, tie) /\
    ~ denotes (RankStrictCost lower) (lower, tie).
  Proof. intros; apply ordinary_tie_floor_is_not_a_strict_cut. Qed.

  Example unknown_does_not_assert_a_strict_cut :
    denotes RankUnknown (0, 0%nat) /\
    ~ denotes (RankStrictCost 0) (0, 0%nat).
  Proof.
    split; [exact I |].
    change (~ (0 < 0)); lra.
  Qed.

  Example inhabited_strict_region_still_rejects_empty :
    sound (above_cut 0) (RankStrictCost 0) /\
    sound (above_cut 0) RankUnknown /\
    ~ sound (above_cut 0) RankEmpty.
  Proof.
    split; [apply strict_certificate_for_the_entire_open_region |].
    split; [apply unknown_is_sound_for_every_region |].
    intro Hempty.
    apply (Hempty ((1#2), 0%nat)).
    change (0 < (1#2)); lra.
  Qed.
End DenseRationalCosts.
