(** Machine-checked kernel obligations for one-dimensional discrete Fréchet.

    The executable kernel replaces scalar links by exact interval minima and
    accumulates links with a bottleneck maximum. This development proves that
    the relaxed recurrence is monotone, that its endpoint and one-sided
    Hausdorff candidate bounds are admissible, and that the local triangle
    obligation survives bottleneck composition. It contains no axioms or
    admitted obligations. *)

From Stdlib Require Import Reals Lra List.

Import ListNotations.
Open Scope R_scope.

Definition rmin (x y : R) : R := if Rle_dec x y then x else y.
Definition rmax (x y : R) : R := if Rle_dec x y then y else x.
Definition min3 (x y z : R) : R := rmin x (rmin y z).

Lemma rmin_le_left : forall x y, rmin x y <= x.
Proof. intros x y; unfold rmin; destruct (Rle_dec x y); lra. Qed.

Lemma rmin_le_right : forall x y, rmin x y <= y.
Proof. intros x y; unfold rmin; destruct (Rle_dec x y); lra. Qed.

Lemma rmin_monotone : forall a b c d,
  a <= c -> b <= d -> rmin a b <= rmin c d.
Proof.
  intros a b c d Hac Hbd. unfold rmin.
  destruct (Rle_dec a b); destruct (Rle_dec c d); lra.
Qed.

Lemma rmax_monotone : forall a b c d,
  a <= c -> b <= d -> rmax a b <= rmax c d.
Proof.
  intros a b c d Hac Hbd. unfold rmax.
  destruct (Rle_dec a b); destruct (Rle_dec c d); lra.
Qed.

Lemma rmax_inflates_left : forall prefix step,
  prefix <= rmax prefix step.
Proof. intros prefix step; unfold rmax; destruct (Rle_dec prefix step); lra. Qed.

Lemma rmax_nonnegative : forall x y,
  0 <= x -> 0 <= y -> 0 <= rmax x y.
Proof. intros x y Hx Hy; unfold rmax; destruct (Rle_dec x y); lra. Qed.

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

Definition frechet_step (north west east link : R) : R :=
  rmax (min3 north west east) link.

Theorem frechet_step_monotone : forall n w e link n' w' e' link',
  n <= n' -> w <= w' -> e <= e' -> link <= link' ->
  frechet_step n w e link <= frechet_step n' w' e' link'.
Proof.
  intros n w e link n' w' e' link' Hn Hw He Hlink.
  unfold frechet_step, min3.
  apply rmax_monotone; [apply rmin_monotone; [exact Hn |] | exact Hlink].
  apply rmin_monotone; assumption.
Qed.

Theorem interval_frechet_step_admissible : forall n w e n' w' e' x lo hi y,
  n <= n' -> w <= w' -> e <= e' -> lo <= hi -> lo <= y <= hi ->
  frechet_step n w e (interval_dist x lo hi)
  <= frechet_step n' w' e' (Rabs (x - y)).
Proof.
  intros n w e n' w' e' x lo hi y Hn Hw He Hbox Hy.
  apply frechet_step_monotone; try assumption.
  now apply interval_dist_admissible.
Qed.

Theorem point_interval_frechet_step_exact : forall n w e x y,
  frechet_step n w e (interval_dist x y y)
  = frechet_step n w e (Rabs (x - y)).
Proof. intros; now rewrite interval_dist_degenerate. Qed.

Definition endpoint_bound (x_first x_last y_first y_last : R) : R :=
  rmax (Rabs (x_first - y_first)) (Rabs (x_last - y_last)).

Theorem endpoint_bound_admissible : forall x_first x_last y_first y_last exact,
  Rabs (x_first - y_first) <= exact ->
  Rabs (x_last - y_last) <= exact ->
  endpoint_bound x_first x_last y_first y_last <= exact.
Proof.
  intros x_first x_last y_first y_last exact Hfirst Hlast.
  unfold endpoint_bound, rmax. destruct (Rle_dec (Rabs (x_first - y_first))
    (Rabs (x_last - y_last))); assumption.
Qed.

Fixpoint nearest (x : R) (ys : list R) : R :=
  match ys with
  | [] => 0
  | y :: tail =>
      match tail with
      | [] => Rabs (x - y)
      | _ => rmin (Rabs (x - y)) (nearest x tail)
      end
  end.

Lemma nearest_le_member : forall x ys y,
  ys <> [] -> In y ys -> nearest x ys <= Rabs (x - y).
Proof.
  intros x ys. induction ys as [|head tail IH]; intros y Hnonempty Hin.
  - contradiction.
  - destruct tail as [|next rest].
    + simpl in *. destruct Hin as [Heq | []]. subst. lra.
    + simpl in *. destruct Hin as [Heq | Hin].
      * subst. apply rmin_le_left.
      * eapply Rle_trans; [apply rmin_le_right |].
        apply IH; [discriminate | exact Hin].
Qed.

Fixpoint list_max (values : list R) : R :=
  match values with
  | [] => 0
  | value :: tail => rmax value (list_max tail)
  end.

Lemma list_max_upper : forall values upper,
  0 <= upper -> Forall (fun value => value <= upper) values ->
  list_max values <= upper.
Proof.
  intros values upper Hupper Hall. induction Hall as [|value tail Hvalue Hall IH].
  - simpl. exact Hupper.
  - simpl. unfold rmax. destruct (Rle_dec value (list_max tail)); assumption.
Qed.

Definition one_sided_hausdorff (xs ys : list R) : R :=
  list_max (map (fun x => nearest x ys) xs).

Theorem one_sided_hausdorff_admissible : forall xs ys exact,
  ys <> [] -> 0 <= exact ->
  (forall x, In x xs -> exists y, In y ys /\ Rabs (x - y) <= exact) ->
  one_sided_hausdorff xs ys <= exact.
Proof.
  intros xs ys exact Hys Hexact Hcovered. unfold one_sided_hausdorff.
  apply list_max_upper; [exact Hexact |].
  apply Forall_forall. intros nearest_x Hin.
  apply in_map_iff in Hin. destruct Hin as [x [Heq Hxin]]. subst nearest_x.
  destruct (Hcovered x Hxin) as [y [Hy Hlink]].
  eapply Rle_trans; [apply nearest_le_member; eassumption | exact Hlink].
Qed.

Theorem bottleneck_triangle_composition_step : forall prefix_xy prefix_yz prefix_xz
  link_xy link_yz link_xz,
  0 <= prefix_xy -> 0 <= prefix_yz -> 0 <= link_xy -> 0 <= link_yz ->
  prefix_xz <= prefix_xy + prefix_yz -> link_xz <= link_xy + link_yz ->
  rmax prefix_xz link_xz
  <= rmax prefix_xy link_xy + rmax prefix_yz link_yz.
Proof.
  intros prefix_xy prefix_yz prefix_xz link_xy link_yz link_xz
    Hpxy Hpyz Hlxy Hlyz Hprefix Hlink.
  unfold rmax in *. repeat destruct Rle_dec; lra.
Qed.

Lemma zero_absolute_link_identifies_points : forall x y,
  Rabs (x - y) = 0 -> x = y.
Proof.
  intros x y Hzero. destruct (Req_EM_T (x - y) 0) as [Heq | Hneq].
  - lra.
  - exfalso. exact (Rabs_no_R0 (x - y) Hneq Hzero).
Qed.

Lemma duplicate_stutter_has_zero_link : forall x,
  Rabs (x - x) = 0.
Proof. intros x. replace (x - x) with 0 by ring. apply Rabs_R0. Qed.

Theorem bottleneck_zero_identifies_each_link : forall prefix link,
  0 <= prefix -> 0 <= link -> rmax prefix link = 0 ->
  prefix = 0 /\ link = 0.
Proof.
  intros prefix link Hprefix Hlink Hzero. unfold rmax in Hzero.
  destruct (Rle_dec prefix link); split; lra.
Qed.
