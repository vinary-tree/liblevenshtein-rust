(** * Language-product conformance

    This trusted module states the algebraic obligations used by
    [src/transducer/language/product.rs] without introducing parameters,
    axioms, or hypotheses.  State sets and transition relations are executable
    Boolean predicates; each theorem quantifies over the concrete predicates
    supplied by a caller.
*)

From Stdlib Require Import Bool List Nat Lia.
Import ListNotations.

Definition LPStateSet : Type := nat -> bool.
Definition LPTransition : Type := nat -> nat -> nat -> bool.

Definition lp_union (left right : LPStateSet) : LPStateSet :=
  fun state => left state || right state.

Definition lp_difference (states cheaper : LPStateSet) : LPStateSet :=
  fun state => states state && negb (cheaper state).

Definition lp_move
    (transition : LPTransition)
    (unit : nat)
    (states : LPStateSet)
    (target : nat) : Prop :=
  exists source,
    states source = true /\ transition source unit target = true.

Definition lp_accepts (states finals : LPStateSet) : Prop :=
  exists state, states state = true /\ finals state = true.

(** Relational image distributes over union.  The Rust property test exercises
    the same law for [SmallDfa::step] and NFA epsilon-closures. *)
Theorem lp_move_distributes_over_union :
  forall transition unit left right target,
    lp_move transition unit (lp_union left right) target <->
    lp_move transition unit left target \/
    lp_move transition unit right target.
Proof.
  intros transition unit left right target.
  unfold lp_move, lp_union.
  split.
  - intros [source [Hmember Hedge]].
    apply orb_true_iff in Hmember.
    destruct Hmember as [Hleft | Hright].
    + left. exists source. auto.
    + right. exists source. auto.
  - intros [[source [Hleft Hedge]] | [source [Hright Hedge]]].
    + exists source. rewrite Hleft. auto.
    + exists source. rewrite Hright, orb_true_r. auto.
Qed.

(** Removing from a dearer level every state already present in a cheaper
    level preserves whether either level accepts.  Repeated pairwise use is the
    induction step for the entire fixed-cost frontier. *)
Theorem lp_canonicalization_preserves_acceptance :
  forall cheaper dearer finals,
    lp_accepts cheaper finals \/ lp_accepts dearer finals <->
    lp_accepts cheaper finals \/
    lp_accepts (lp_difference dearer cheaper) finals.
Proof.
  intros cheaper dearer finals. unfold lp_accepts, lp_difference.
  split.
  - intros [Hcheap | [state [Hdear Hfinal]]].
    + left. exact Hcheap.
    + destruct (cheaper state) eqn:Hcheap.
      * left. exists state. auto.
      * right. exists state. simpl. rewrite Hdear, Hcheap. auto.
  - intros [Hcheap | [state [Hdiff Hfinal]]].
    + left. exact Hcheap.
    + right. exists state. apply andb_true_iff in Hdiff. tauto.
Qed.

(** Canonicalization makes adjacent exact-cost levels disjoint in the cheaper
    direction. *)
Theorem lp_difference_excludes_cheaper :
  forall states cheaper state,
    lp_difference states cheaper state = true ->
    cheaper state = false.
Proof.
  intros states cheaper state H.
  unfold lp_difference in H.
  apply andb_true_iff in H. destruct H as [_ Hnot].
  apply negb_true_iff in Hnot. exact Hnot.
Qed.

(** A [u8] edit budget produces exactly [k + 1] exact-cost levels and never
    more than 256 levels. *)
Theorem lp_frontier_level_bound : forall max_distance,
  max_distance <= 255 ->
  1 <= S max_distance <= 256.
Proof. intros. lia. Qed.

Theorem lp_repeat_has_exact_frontier_length : forall max_distance,
  length (repeat ((fun _ : nat => false) : LPStateSet) (S max_distance)) =
    S max_distance.
Proof. intros. apply repeat_length. Qed.
