(** Assumption-free obligations for symmetrically banded Dynamic Time Warping.

    The executable kernel accumulates squared local deviations, substitutes
    exact interval minima at trie edges, rejects endpoints outside a required
    Sakoe--Chiba band, and applies cumulative LB_Keogh before constructing the
    next dynamic-programming column.  These theorems cover the arithmetic
    refinement and pruning boundary.  The final example also certifies a
    concrete triangle-inequality violation, so no metric axiom is available to
    the index. *)

From Stdlib Require Import Reals Lra List Lia PeanoNat.

Import ListNotations.
Open Scope R_scope.

Definition rmin (x y : R) : R := if Rle_dec x y then x else y.
Definition min3 (x y z : R) : R := rmin x (rmin y z).
Definition sq (x : R) : R := x * x.

Lemma sq_nonnegative : forall x, 0 <= sq x.
Proof. intros x; unfold sq; nra. Qed.

Lemma rmin_monotone : forall a b c d,
  a <= c -> b <= d -> rmin a b <= rmin c d.
Proof.
  intros a b c d Hac Hbd. unfold rmin.
  destruct (Rle_dec a b); destruct (Rle_dec c d); lra.
Qed.

Lemma rmin_nonnegative : forall a b,
  0 <= a -> 0 <= b -> 0 <= rmin a b.
Proof.
  intros a b Ha Hb. unfold rmin. destruct (Rle_dec a b); lra.
Qed.

Definition interval_dist (x lo hi : R) : R :=
  if Rlt_dec x lo then lo - x
  else if Rlt_dec hi x then x - hi
  else 0.

Lemma interval_dist_nonnegative : forall x lo hi,
  0 <= interval_dist x lo hi.
Proof.
  intros x lo hi. unfold interval_dist.
  destruct (Rlt_dec x lo); destruct (Rlt_dec hi x); lra.
Qed.

Lemma interval_dist_admissible : forall x lo hi y,
  lo <= hi -> lo <= y <= hi ->
  interval_dist x lo hi <= Rabs (x - y).
Proof.
  intros x lo hi y Hbox [Hlo Hhi]. unfold interval_dist.
  destruct (Rlt_dec x lo) as [Hxlo | Hxlo].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec hi x) as [Hhix | Hhix].
    + rewrite Rabs_right; lra.
    + pose proof (Rabs_pos (x - y)). lra.
Qed.

Lemma interval_dist_degenerate : forall x y,
  interval_dist x y y = Rabs (x - y).
Proof.
  intros x y. unfold interval_dist.
  destruct (Rlt_dec x y) as [Hxy | Hxy].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec y x) as [Hyx | Hyx].
    + rewrite Rabs_right; lra.
    + assert (x = y) by lra. subst. replace (y - y) with 0 by ring.
      rewrite Rabs_R0. reflexivity.
Qed.

Lemma square_monotone_nonnegative : forall a b,
  0 <= a -> a <= b -> sq a <= sq b.
Proof. intros a b Ha Hab. unfold sq. nra. Qed.

Lemma square_absolute : forall x, sq (Rabs x) = sq x.
Proof.
  intros x. unfold sq. destruct (Rcase_abs x) as [Hnegative | Hnonnegative].
  - rewrite Rabs_left; nra.
  - rewrite Rabs_right; nra.
Qed.

Theorem squared_interval_cost_admissible : forall x lo hi y,
  lo <= hi -> lo <= y <= hi ->
  sq (interval_dist x lo hi) <= sq (x - y).
Proof.
  intros x lo hi y Hbox Hy.
  assert (Hsquare : sq (x - y) = sq (Rabs (x - y))).
  { symmetry. apply square_absolute. }
  rewrite Hsquare.
  apply square_monotone_nonnegative.
  - apply interval_dist_nonnegative.
  - now apply interval_dist_admissible.
Qed.

Theorem squared_point_interval_exact : forall x y,
  sq (interval_dist x y y) = sq (x - y).
Proof.
  intros x y. rewrite interval_dist_degenerate.
  apply square_absolute.
Qed.

Definition dtw_step (north west east local : R) : R :=
  min3 north west east + local.

Theorem interval_dtw_step_admissible : forall n w e n' w' e' x lo hi y,
  n <= n' -> w <= w' -> e <= e' -> lo <= hi -> lo <= y <= hi ->
  dtw_step n w e (sq (interval_dist x lo hi))
  <= dtw_step n' w' e' (sq (x - y)).
Proof.
  intros n w e n' w' e' x lo hi y Hn Hw He Hbox Hy.
  unfold dtw_step, min3.
  assert (Hmin : rmin n (rmin w e) <= rmin n' (rmin w' e')).
  { apply rmin_monotone; [exact Hn |]. apply rmin_monotone; assumption. }
  pose proof (squared_interval_cost_admissible x lo hi y Hbox Hy).
  lra.
Qed.

Theorem dtw_step_nonnegative : forall n w e local,
  0 <= n -> 0 <= w -> 0 <= e -> 0 <= local ->
  0 <= dtw_step n w e local.
Proof.
  intros n w e local Hn Hw He Hlocal. unfold dtw_step, min3.
  pose proof (rmin_nonnegative w e Hw He).
  pose proof (rmin_nonnegative n (rmin w e) Hn H).
  lra.
Qed.

Fixpoint sum_costs (costs : list R) : R :=
  match costs with
  | [] => 0
  | cost :: tail => cost + sum_costs tail
  end.

Theorem prefix_keogh_sum_admissible : forall bounds exact_links,
  Forall2 Rle bounds exact_links ->
  sum_costs bounds <= sum_costs exact_links.
Proof.
  intros bounds exact_links Hlinks. induction Hlinks; simpl; lra.
Qed.

Theorem prefix_first_gate_prunes_soundly : forall prefix exact cutoff,
  prefix <= exact -> cutoff < prefix -> cutoff < exact.
Proof. intros prefix exact cutoff Hadmissible Hreject; lra. Qed.

Definition nat_distance (m n : nat) : nat :=
  if Nat.leb m n then n - m else m - n.

Definition endpoint_in_band (m n width : nat) : Prop :=
  (nat_distance m n <= width)%nat.

Theorem excessive_length_gap_is_unreachable : forall m n width,
  (width < nat_distance m n)%nat -> ~ endpoint_in_band m n width.
Proof. intros m n width Hgap Hin; unfold endpoint_in_band in Hin; lia. Qed.

Theorem squared_local_cost_is_symmetric : forall x y,
  sq (x - y) = sq (y - x).
Proof. intros x y; unfold sq; ring. Qed.

Definition singleton_cost (x y : R) : R := sq (x - y).
Definition singleton_pair_cost (x y1 y2 : R) : R :=
  sq (x - y1) + sq (x - y2).

Example band_one_dtw_triangle_counterexample :
  sqrt (singleton_pair_cost 0 1 1)
  > sqrt (singleton_cost 0 1) + sqrt (singleton_pair_cost 1 1 1).
Proof.
  replace (singleton_pair_cost 0 1 1) with 2 by
    (unfold singleton_pair_cost, sq; ring).
  replace (singleton_cost 0 1) with 1 by
    (unfold singleton_cost, sq; ring).
  replace (singleton_pair_cost 1 1 1) with 0 by
    (unfold singleton_pair_cost, sq; ring).
  rewrite sqrt_1, sqrt_0, Rplus_0_r.
  assert (Hsqrt_nonnegative : 0 <= sqrt 2) by apply sqrt_pos.
  pose proof (Rsqr_sqrt 2 ltac:(lra)) as Hsqrt_square.
  unfold Rsqr in Hsqrt_square.
  nra.
Qed.
