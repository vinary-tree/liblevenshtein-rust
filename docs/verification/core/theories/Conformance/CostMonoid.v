(** * Ordered cost-monoid conformance

    This assumption-free model proves the algebraic laws used by the Rust
    [CostMonoid] implementations over non-negative mathematical reals extended
    with an explicit top element.  The Rust [WeightedCost] carrier is IEEE-754
    [f64], so its finite addition has a separately documented rounding trust
    boundary; this file proves the real-number specification, not a false claim
    of bitwise floating-point associativity.
*)

From Stdlib Require Import Reals Lra.

Open Scope R_scope.

Inductive ExtendedCost : Type :=
| Finite : R -> ExtendedCost
| Top : ExtendedCost.

Definition weighted_combine (left right : ExtendedCost) : ExtendedCost :=
  match left, right with
  | Finite a, Finite b => Finite (a + b)
  | _, _ => Top
  end.

Definition bottleneck_combine (left right : ExtendedCost) : ExtendedCost :=
  match left, right with
  | Finite a, Finite b => Finite (Rmax a b)
  | _, _ => Top
  end.

Definition cost_le (left right : ExtendedCost) : Prop :=
  match left, right with
  | Finite a, Finite b => a <= b
  | Finite _, Top => True
  | Top, Top => True
  | Top, Finite _ => False
  end.

Theorem cost_le_total : forall left right,
  cost_le left right \/ cost_le right left.
Proof.
  intros [a |] [b |]; simpl; try tauto.
  destruct (Rle_dec a b); [left | right]; lra.
Qed.

Theorem cost_le_transitive : forall a b c,
  cost_le a b -> cost_le b c -> cost_le a c.
Proof.
  intros [a |] [b |] [c |]; simpl; try tauto; lra.
Qed.

Theorem weighted_l1_associative : forall a b c,
  weighted_combine (weighted_combine a b) c =
  weighted_combine a (weighted_combine b c).
Proof.
  intros [a |] [b |] [c |]; simpl; try reflexivity.
  f_equal. ring.
Qed.

Theorem weighted_l1_zero_identity : forall a,
  weighted_combine (Finite 0) a = a /\
  weighted_combine a (Finite 0) = a.
Proof.
  intros [a |]; simpl; split; try reflexivity; f_equal; ring.
Qed.

Theorem weighted_l2_left_monotone : forall a b w,
  a <= b -> a + w <= b + w.
Proof. intros; lra. Qed.

Theorem weighted_l2_right_monotone : forall a b w,
  a <= b -> w + a <= w + b.
Proof. intros; lra. Qed.

Theorem weighted_l4_inflation : forall accumulated step,
  0 <= step -> accumulated <= accumulated + step.
Proof. intros; lra. Qed.

Theorem weighted_l7_top_absorbing : forall cost,
  weighted_combine cost Top = Top /\
  weighted_combine Top cost = Top.
Proof. intros [cost |]; split; reflexivity. Qed.

Theorem bottleneck_l1_associative : forall a b c,
  bottleneck_combine (bottleneck_combine a b) c =
  bottleneck_combine a (bottleneck_combine b c).
Proof.
  intros [a |] [b |] [c |]; simpl; try reflexivity.
  f_equal. unfold Rmax. repeat destruct Rle_dec; lra.
Qed.

Theorem bottleneck_l1_zero_identity : forall a,
  0 <= a ->
  bottleneck_combine (Finite 0) (Finite a) = Finite a /\
  bottleneck_combine (Finite a) (Finite 0) = Finite a.
Proof.
  intros a H. simpl. unfold Rmax.
  repeat destruct Rle_dec; split; try reflexivity; f_equal; lra.
Qed.

Theorem bottleneck_l2_left_monotone : forall a b w,
  a <= b -> Rmax a w <= Rmax b w.
Proof.
  intros. unfold Rmax. repeat destruct Rle_dec; lra.
Qed.

Theorem bottleneck_l2_right_monotone : forall a b w,
  a <= b -> Rmax w a <= Rmax w b.
Proof.
  intros. unfold Rmax. repeat destruct Rle_dec; lra.
Qed.

Theorem bottleneck_l4_inflation : forall accumulated step,
  accumulated <= Rmax accumulated step.
Proof.
  intros. unfold Rmax. destruct Rle_dec; lra.
Qed.

Theorem bottleneck_l7_top_absorbing : forall cost,
  bottleneck_combine cost Top = Top /\
  bottleneck_combine Top cost = Top.
Proof. intros [cost |]; split; reflexivity. Qed.

(** [within] is [cost_le], so its downward-closure law is exactly transitivity. *)
Theorem within_l6_downward_closed : forall cheaper dearer threshold,
  cost_le cheaper dearer ->
  cost_le dearer threshold ->
  cost_le cheaper threshold.
Proof. exact cost_le_transitive. Qed.
