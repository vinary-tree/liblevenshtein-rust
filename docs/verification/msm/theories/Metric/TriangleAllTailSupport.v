(** * MSM Triangle All-Tail Support Lemmas *)

From Stdlib Require Import List Lia Psatz QArith.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction CFunctionBounds MsmDistance Symmetry.

Definition msm_triangle_cell_bound
    (x : Q) (xs : list Q) (y : Q) (ys : list Q)
    (z : Q) (zs : list Q) (c_const : Q) (i j k : nat) : Prop :=
  msm_matrix_cell x xs z zs c_const i k <=
  msm_matrix_cell x xs y ys c_const i j +
  msm_matrix_cell y ys z zs c_const j k.

Definition msm_cell_potential
    (x : Q) (xs : list Q) (y : Q) (ys : list Q)
    (c_const : Q) (i j : nat) : Q :=
  msm_matrix_cell x xs y ys c_const i j -
    half_abs (nth i (x :: xs) 0) (nth j (y :: ys) 0).

Lemma msm_triangle_cell_base :
  forall x xs y ys z zs c_const,
    msm_triangle_cell_bound x xs y ys z zs c_const 0 0 0.
Proof.
  intros x xs y ys z zs c_const.
  unfold msm_triangle_cell_bound.
  rewrite !msm_matrix_cell_base.
  apply Qabs_triangle_diff.
Qed.

Definition msm_triangle_cell_potential_bound
    (x : Q) (xs : list Q) (y : Q) (ys : list Q)
    (z : Q) (zs : list Q) (c_const : Q) (i j k : nat) : Prop :=
  msm_cell_potential x xs z zs c_const i k <=
  msm_cell_potential x xs y ys c_const i j +
  msm_cell_potential y ys z zs c_const j k.

Lemma msm_triangle_cell_potential_base :
  forall x xs y ys z zs c_const,
    msm_triangle_cell_potential_bound x xs y ys z zs c_const 0 0 0.
Proof.
  intros x xs y ys z zs c_const.
  unfold msm_triangle_cell_potential_bound, msm_cell_potential.
  rewrite !msm_matrix_cell_base.
  simpl.
  unfold half_abs.
  pose proof (Qabs_triangle_diff x y z) as Htri.
  psatz Q.
Qed.

Definition move_potential (a_prev b_prev a b : Q) : Q :=
  Qabs_diff a b - half_abs a b + half_abs a_prev b_prev.

Definition merge_potential (c_const a_prev a b : Q) : Q :=
  c_func c_const a a_prev b - half_abs a b + half_abs a_prev b.

Definition split_potential (c_const a b_prev b : Q) : Q :=
  c_func c_const b a b_prev - half_abs a b + half_abs a b_prev.

Lemma move_potential_halves : forall a_prev b_prev a b,
  move_potential a_prev b_prev a b ==
  half_abs a b + half_abs a_prev b_prev.
Proof.
  intros a_prev b_prev a b.
  unfold move_potential, half_abs.
  ring.
Qed.

Lemma merge_potential_exact : forall c_const a_prev a b,
  merge_potential c_const a_prev a b ==
  c_const + half_abs a a_prev.
Proof.
  intros c_const a_prev a b.
  unfold merge_potential.
  rewrite (c_func_singleton_potential_delta c_const a a_prev b).
  reflexivity.
Qed.

Lemma split_potential_exact : forall c_const a b_prev b,
  split_potential c_const a b_prev b ==
  c_const + half_abs b b_prev.
Proof.
  intros c_const a b_prev b.
  unfold split_potential.
  setoid_replace (c_func c_const b a b_prev)
    with (c_func c_const b b_prev a)
    by (rewrite c_func_symm_bc; reflexivity).
  setoid_replace (half_abs a b) with (half_abs b a)
    by (unfold half_abs; rewrite Qabs_diff_symm; reflexivity).
  setoid_replace (half_abs a b_prev) with (half_abs b_prev a)
    by (unfold half_abs; rewrite Qabs_diff_symm; reflexivity).
  rewrite (c_func_singleton_potential_delta c_const b b_prev a).
  reflexivity.
Qed.

Lemma merge_potential_target_irrelevant :
  forall c_const a_prev a b b',
    merge_potential c_const a_prev a b ==
    merge_potential c_const a_prev a b'.
Proof.
  intros c_const a_prev a b b'.
  rewrite !merge_potential_exact.
  reflexivity.
Qed.

Lemma split_potential_source_irrelevant :
  forall c_const a a' b_prev b,
    split_potential c_const a b_prev b ==
    split_potential c_const a' b_prev b.
Proof.
  intros c_const a a' b_prev b.
  rewrite !split_potential_exact.
  reflexivity.
Qed.

Lemma move_potential_nonneg : forall a_prev b_prev a b,
  0 <= move_potential a_prev b_prev a b.
Proof.
  intros a_prev b_prev a b.
  rewrite move_potential_halves.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat; apply Qhalf_nonneg; apply Qabs_diff_nonneg.
Qed.

Lemma merge_potential_nonneg : forall c_const a_prev a b,
  0 <= c_const ->
  0 <= merge_potential c_const a_prev a b.
Proof.
  intros c_const a_prev a b Hc.
  rewrite merge_potential_exact.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat.
  - exact Hc.
  - apply Qhalf_nonneg. apply Qabs_diff_nonneg.
Qed.

Lemma split_potential_nonneg : forall c_const a b_prev b,
  0 <= c_const ->
  0 <= split_potential c_const a b_prev b.
Proof.
  intros c_const a b_prev b Hc.
  rewrite split_potential_exact.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat.
  - exact Hc.
  - apply Qhalf_nonneg. apply Qabs_diff_nonneg.
Qed.

Lemma half_abs_triangle : forall a b c,
  half_abs a c <= half_abs a b + half_abs b c.
Proof.
  intros a b c.
  unfold half_abs.
  setoid_replace (Qabs_diff a b * (1#2) + Qabs_diff b c * (1#2))
    with ((Qabs_diff a b + Qabs_diff b c) * (1#2))
    by ring.
  apply Qmult_le_compat_r.
  - apply Qabs_triangle_diff.
  - unfold Qle. simpl. lia.
Qed.

Lemma msm_triangle_cell_potential_to_bound :
  forall x xs y ys z zs c_const i j k,
    msm_triangle_cell_potential_bound x xs y ys z zs c_const i j k ->
    msm_triangle_cell_bound x xs y ys z zs c_const i j k.
Proof.
  intros x xs y ys z zs c_const i j k Hpot.
  unfold msm_triangle_cell_potential_bound in Hpot.
  unfold msm_cell_potential in Hpot.
  unfold msm_triangle_cell_bound.
  set (xi := nth i (x :: xs) 0) in *.
  set (yj := nth j (y :: ys) 0) in *.
  set (zk := nth k (z :: zs) 0) in *.
  set (DXZ := msm_matrix_cell x xs z zs c_const i k) in *.
  set (DXY := msm_matrix_cell x xs y ys c_const i j) in *.
  set (DYZ := msm_matrix_cell y ys z zs c_const j k) in *.
  setoid_replace DXZ with ((DXZ - half_abs xi zk) + half_abs xi zk) by ring.
  eapply Qle_trans.
  - apply Qplus_le_compat.
    + exact Hpot.
    + apply (half_abs_triangle xi yj zk).
  - ring_simplify.
    apply Qle_refl.
Qed.

Lemma move_move_potential_triangle :
  forall x_prev y_prev z_prev x y z,
    move_potential x_prev z_prev x z <=
    move_potential x_prev y_prev x y +
    move_potential y_prev z_prev y z.
Proof.
  intros x_prev y_prev z_prev x y z.
  rewrite !move_potential_halves.
  setoid_replace
    (half_abs x y + half_abs x_prev y_prev +
     (half_abs y z + half_abs y_prev z_prev))
    with
    ((half_abs x y + half_abs y z) +
     (half_abs x_prev y_prev + half_abs y_prev z_prev))
    by ring.
  apply Qplus_le_compat; apply half_abs_triangle.
Qed.

Lemma move_merge_potential_triangle :
  forall c_const x_prev y_prev x y z,
    merge_potential c_const x_prev x z <=
    move_potential x_prev y_prev x y +
    merge_potential c_const y_prev y z.
Proof.
  intros c_const x_prev y_prev x y z.
  unfold move_potential, merge_potential.
  pose proof (c_func_diag_potential_step c_const x x_prev y y_prev z) as Hstep.
  setoid_replace
    (Qabs_diff x y - half_abs x y + half_abs x_prev y_prev +
     (c_func c_const y y_prev z - half_abs y z + half_abs y_prev z))
    with
    (Qabs_diff x y - half_abs x y +
     c_func c_const y y_prev z - half_abs y z + half_abs y_prev z +
     half_abs x_prev y_prev)
    by ring.
  exact Hstep.
Qed.

Lemma merge_move_potential_triangle :
  forall c_const x_prev y_prev z_prev x y z,
    0 <= c_const ->
    merge_potential c_const x_prev x z <=
    merge_potential c_const x_prev x y +
    move_potential y_prev z_prev y z.
Proof.
  intros c_const x_prev y_prev z_prev x y z Hc.
  rewrite !merge_potential_exact.
  setoid_replace (c_const + half_abs x x_prev)
    with ((c_const + half_abs x x_prev) + 0) at 1 by ring.
  apply Qplus_le_compat.
  - apply Qle_refl.
  - apply move_potential_nonneg.
Qed.

Lemma move_split_potential_triangle :
  forall c_const x_prev y_prev z_prev x y z,
    0 <= c_const ->
    split_potential c_const x z_prev z <=
    move_potential x_prev y_prev x y +
    split_potential c_const y z_prev z.
Proof.
  intros c_const x_prev y_prev z_prev x y z Hc.
  rewrite !split_potential_exact.
  setoid_replace (c_const + half_abs z z_prev)
    with (0 + (c_const + half_abs z z_prev)) at 1 by ring.
  apply Qplus_le_compat.
  - apply move_potential_nonneg.
  - apply Qle_refl.
Qed.

Lemma merge_merge_potential_triangle :
  forall c_const x_prev y_prev x y z,
    0 <= c_const ->
    merge_potential c_const x_prev x z <=
    merge_potential c_const x_prev x y +
    merge_potential c_const y_prev y z.
Proof.
  intros c_const x_prev y_prev x y z Hc.
  rewrite !merge_potential_exact.
  setoid_replace (c_const + half_abs x x_prev)
    with ((c_const + half_abs x x_prev) + 0) at 1 by ring.
  apply Qplus_le_compat.
  - apply Qle_refl.
  - setoid_replace 0 with (0 + 0) by ring.
    apply Qplus_le_compat.
    + exact Hc.
    + apply Qhalf_nonneg. apply Qabs_diff_nonneg.
Qed.

Lemma merge_split_potential_triangle :
  forall c_const x_prev z_prev x y z,
    0 <= c_const ->
    split_potential c_const x z_prev z <=
    merge_potential c_const x_prev x y +
    split_potential c_const y z_prev z.
Proof.
  intros c_const x_prev z_prev x y z Hc.
  rewrite !split_potential_exact.
  setoid_replace (c_const + half_abs z z_prev)
    with (0 + (c_const + half_abs z z_prev)) at 1 by ring.
  apply Qplus_le_compat.
  - apply merge_potential_nonneg. exact Hc.
  - apply Qle_refl.
Qed.

Lemma split_split_potential_triangle :
  forall c_const x y_prev z_prev y z,
    0 <= c_const ->
    split_potential c_const x z_prev z <=
    split_potential c_const x y_prev y +
    split_potential c_const y z_prev z.
Proof.
  intros c_const x y_prev z_prev y z Hc.
  rewrite !split_potential_exact.
  setoid_replace (c_const + half_abs z z_prev)
    with (0 + (c_const + half_abs z z_prev)) at 1 by ring.
  apply Qplus_le_compat.
  - setoid_replace 0 with (0 + 0) by ring.
    apply Qplus_le_compat.
    + exact Hc.
    + apply Qhalf_nonneg. apply Qabs_diff_nonneg.
  - apply Qle_refl.
Qed.

Lemma split_move_potential_triangle :
  forall c_const x y_prev z_prev y z,
    split_potential c_const x z_prev z <=
    split_potential c_const x y_prev y +
    move_potential y_prev z_prev y z.
Proof.
  intros c_const x y_prev z_prev y z.
  rewrite !split_potential_exact.
  rewrite move_potential_halves.
  setoid_replace
    (c_const + half_abs y y_prev +
      (half_abs y z + half_abs y_prev z_prev))
    with
    (c_const + (half_abs y y_prev + half_abs y z + half_abs y_prev z_prev))
    by ring.
  apply Qplus_le_compat.
  - apply Qle_refl.
  - eapply Qle_trans.
    + apply half_abs_triangle with (b := y).
    + setoid_replace
        (half_abs y y_prev + half_abs y z + half_abs y_prev z_prev)
        with (half_abs y z + (half_abs y y_prev + half_abs y_prev z_prev))
        by ring.
      apply Qplus_le_compat.
      * unfold half_abs. rewrite Qabs_diff_symm. apply Qle_refl.
      * apply half_abs_triangle.
Qed.

Lemma split_merge_potential_sum_nonneg :
  forall c_const x y_prev z y,
    0 <= c_const ->
    0 <= split_potential c_const x y_prev y +
         merge_potential c_const y_prev y z.
Proof.
  intros c_const x y_prev z y Hc.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat.
  - apply split_potential_nonneg. exact Hc.
  - apply merge_potential_nonneg. exact Hc.
Qed.

Lemma Qle_minus_compat_r_local : forall a b c,
  a <= b ->
  a - c <= b - c.
Proof.
  intros a b c Hab.
  setoid_replace (a - c) with (a + - c) by ring.
  setoid_replace (b - c) with (b + - c) by ring.
  apply Qplus_le_compat.
  - exact Hab.
  - apply Qle_refl.
Qed.

Lemma msm_matrix_cell_move_potential_step :
  forall x xs y ys c_const i j,
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const (S i) (S j) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) <=
    msm_matrix_cell x xs y ys c_const i j -
      half_abs (nth i (x :: xs) 0) (nth j (y :: ys) 0) +
    move_potential
      (nth i (x :: xs) 0) (nth j (y :: ys) 0)
      (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0).
Proof.
  intros x xs y ys c_const i j Hi Hj.
  rewrite msm_matrix_cell_interior by assumption.
  unfold move_potential.
  setoid_replace
    (msm_matrix_cell x xs y ys c_const i j -
      half_abs (nth i (x :: xs) 0) (nth j (y :: ys) 0) +
      (Qabs_diff (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) -
       half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) +
       half_abs (nth i (x :: xs) 0) (nth j (y :: ys) 0)))
    with
    (msm_matrix_cell x xs y ys c_const i j +
      Qabs_diff (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0))
    by ring.
  apply Qle_minus_compat_r_local.
  apply Qmin3_le_a.
Qed.

Lemma msm_matrix_cell_merge_potential_step :
  forall x xs y ys c_const i j,
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const (S i) (S j) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) <=
    msm_matrix_cell x xs y ys c_const i (S j) -
      half_abs (nth i (x :: xs) 0) (nth (S j) (y :: ys) 0) +
    merge_potential c_const
      (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0)
      (nth (S j) (y :: ys) 0).
Proof.
  intros x xs y ys c_const i j Hi Hj.
  rewrite msm_matrix_cell_interior by assumption.
  unfold merge_potential.
  setoid_replace
    (msm_matrix_cell x xs y ys c_const i (S j) -
      half_abs (nth i (x :: xs) 0) (nth (S j) (y :: ys) 0) +
      (c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0)
         (nth (S j) (y :: ys) 0) -
       half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) +
       half_abs (nth i (x :: xs) 0) (nth (S j) (y :: ys) 0)))
    with
    (msm_matrix_cell x xs y ys c_const i (S j) +
      c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0)
        (nth (S j) (y :: ys) 0) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0))
    by ring.
  apply Qle_minus_compat_r_local.
  apply Qmin3_le_b.
Qed.

Lemma msm_matrix_cell_split_potential_step :
  forall x xs y ys c_const i j,
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const (S i) (S j) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) <=
    msm_matrix_cell x xs y ys c_const (S i) j -
      half_abs (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0) +
    split_potential c_const
      (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0)
      (nth (S j) (y :: ys) 0).
Proof.
  intros x xs y ys c_const i j Hi Hj.
  rewrite msm_matrix_cell_interior by assumption.
  unfold split_potential.
  setoid_replace
    (msm_matrix_cell x xs y ys c_const (S i) j -
      half_abs (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0) +
      (c_func c_const (nth (S j) (y :: ys) 0) (nth (S i) (x :: xs) 0)
         (nth j (y :: ys) 0) -
       half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0) +
       half_abs (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0)))
    with
    (msm_matrix_cell x xs y ys c_const (S i) j +
      c_func c_const (nth (S j) (y :: ys) 0) (nth (S i) (x :: xs) 0)
        (nth j (y :: ys) 0) -
      half_abs (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0))
    by ring.
  apply Qle_minus_compat_r_local.
  apply Qmin3_le_c.
Qed.

Lemma msm_matrix_cell_first_row_potential_eq :
  forall x xs y ys c_const j,
    (S j < length (y :: ys))%nat ->
    msm_cell_potential x xs y ys c_const 0 (S j) ==
    msm_cell_potential x xs y ys c_const 0 j +
      split_potential c_const x
        (nth j (y :: ys) 0) (nth (S j) (y :: ys) 0).
Proof.
  intros x xs y ys c_const j Hj.
  unfold msm_cell_potential, split_potential.
  rewrite msm_matrix_cell_first_row by exact Hj.
  change (nth 0 (x :: xs) 0) with x.
  ring.
Qed.

Lemma msm_matrix_cell_first_col_potential_eq :
  forall x xs y ys c_const i,
    (S i < length (x :: xs))%nat ->
    msm_cell_potential x xs y ys c_const (S i) 0 ==
    msm_cell_potential x xs y ys c_const i 0 +
      merge_potential c_const
        (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0) y.
Proof.
  intros x xs y ys c_const i Hi.
  unfold msm_cell_potential, merge_potential.
  rewrite msm_matrix_cell_first_col by exact Hi.
  change (nth 0 (y :: ys) 0) with y.
  ring.
Qed.

Lemma msm_matrix_cell_first_row_potential_step :
  forall x xs y ys c_const j,
    (S j < length (y :: ys))%nat ->
    msm_cell_potential x xs y ys c_const 0 (S j) <=
    msm_cell_potential x xs y ys c_const 0 j +
      split_potential c_const x
        (nth j (y :: ys) 0) (nth (S j) (y :: ys) 0).
Proof.
  intros x xs y ys c_const j Hj.
  rewrite msm_matrix_cell_first_row_potential_eq by exact Hj.
  apply Qle_refl.
Qed.

Lemma msm_matrix_cell_first_col_potential_step :
  forall x xs y ys c_const i,
    (S i < length (x :: xs))%nat ->
    msm_cell_potential x xs y ys c_const (S i) 0 <=
    msm_cell_potential x xs y ys c_const i 0 +
      merge_potential c_const
        (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0) y.
Proof.
  intros x xs y ys c_const i Hi.
  rewrite msm_matrix_cell_first_col_potential_eq by exact Hi.
  apply Qle_refl.
Qed.

Lemma Qplus_Qmin3_Qmin3_glb :
  forall a b c d e f lower,
    lower <= a + d ->
    lower <= a + e ->
    lower <= a + f ->
    lower <= b + d ->
    lower <= b + e ->
    lower <= b + f ->
    lower <= c + d ->
    lower <= c + e ->
    lower <= c + f ->
    lower <= Qmin3 a b c + Qmin3 d e f.
Proof.
  intros a b c d e f lower Had Hae Haf Hbd Hbe Hbf Hcd Hce Hcf.
  unfold Qmin3, Qmin2.
  repeat match goal with
  | |- context[Qle_bool ?x ?y] => destruct (Qle_bool x y)
  end;
  try assumption;
  try (setoid_replace (d + a) with (a + d) by ring; assumption);
  try (setoid_replace (e + a) with (a + e) by ring; assumption);
  try (setoid_replace (f + a) with (a + f) by ring; assumption);
  try (setoid_replace (d + b) with (b + d) by ring; assumption);
  try (setoid_replace (e + b) with (b + e) by ring; assumption);
  try (setoid_replace (f + b) with (b + f) by ring; assumption);
  try (setoid_replace (d + c) with (c + d) by ring; assumption);
  try (setoid_replace (e + c) with (c + e) by ring; assumption);
  try (setoid_replace (f + c) with (c + f) by ring; assumption).
Qed.

Lemma Qle_pair_sub_add_back :
  forall lower a d h1 h2,
    lower <= (a - h1) + (d - h2) ->
    lower + (h1 + h2) <= a + d.
Proof.
  intros lower a d h1 h2 H.
  psatz Q.
Qed.

Lemma Qplus_Qmin3_Qmin3_sub_glb :
  forall a b c d e f h1 h2 lower,
    lower <= (a - h1) + (d - h2) ->
    lower <= (a - h1) + (e - h2) ->
    lower <= (a - h1) + (f - h2) ->
    lower <= (b - h1) + (d - h2) ->
    lower <= (b - h1) + (e - h2) ->
    lower <= (b - h1) + (f - h2) ->
    lower <= (c - h1) + (d - h2) ->
    lower <= (c - h1) + (e - h2) ->
    lower <= (c - h1) + (f - h2) ->
    lower <= (Qmin3 a b c - h1) + (Qmin3 d e f - h2).
Proof.
  intros a b c d e f h1 h2 lower Had Hae Haf Hbd Hbe Hbf Hcd Hce Hcf.
  assert (Hsum : lower + (h1 + h2) <= Qmin3 a b c + Qmin3 d e f).
  { apply Qplus_Qmin3_Qmin3_glb;
      apply Qle_pair_sub_add_back; assumption. }
  apply (proj1 (Qplus_le_l lower
    ((Qmin3 a b c - h1) + (Qmin3 d e f - h2)) (h1 + h2))).
  psatz Q.
Qed.

Lemma msm_triangle_cell_potential_00k :
  forall x xs y ys z zs c_const k,
    (k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c_const 0 0 k.
Proof.
  intros x xs y ys z zs c_const k.
  induction k as [|k IH]; intros Hk.
  - apply msm_triangle_cell_potential_base.
  - unfold msm_triangle_cell_potential_bound.
    rewrite (msm_matrix_cell_first_row_potential_eq x xs z zs c_const k)
      by exact Hk.
    rewrite (msm_matrix_cell_first_row_potential_eq y ys z zs c_const k)
      by exact Hk.
    setoid_replace
      (msm_cell_potential x xs y ys c_const 0 0 +
       (msm_cell_potential y ys z zs c_const 0 k +
        split_potential c_const y (nth k (z :: zs) 0)
          (nth (S k) (z :: zs) 0)))
      with
      ((msm_cell_potential x xs y ys c_const 0 0 +
        msm_cell_potential y ys z zs c_const 0 k) +
       split_potential c_const y (nth k (z :: zs) 0)
         (nth (S k) (z :: zs) 0))
      by ring.
    apply Qplus_le_compat.
    + assert (Hk' : (k < length (z :: zs))%nat) by lia.
      exact (IH Hk').
    + rewrite (split_potential_source_irrelevant
                 c_const x y (nth k (z :: zs) 0)
                 (nth (S k) (z :: zs) 0)).
      apply Qle_refl.
Qed.

Lemma msm_triangle_cell_potential_i00 :
  forall x xs y ys z zs c_const i,
    (i < length (x :: xs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c_const i 0 0.
Proof.
  intros x xs y ys z zs c_const i.
  induction i as [|i IH]; intros Hi.
  - apply msm_triangle_cell_potential_base.
  - unfold msm_triangle_cell_potential_bound.
    rewrite (msm_matrix_cell_first_col_potential_eq x xs z zs c_const i)
      by exact Hi.
    rewrite (msm_matrix_cell_first_col_potential_eq x xs y ys c_const i)
      by exact Hi.
    setoid_replace
      ((msm_cell_potential x xs y ys c_const i 0 +
        merge_potential c_const (nth i (x :: xs) 0)
          (nth (S i) (x :: xs) 0) y) +
       msm_cell_potential y ys z zs c_const 0 0)
      with
      ((msm_cell_potential x xs y ys c_const i 0 +
        msm_cell_potential y ys z zs c_const 0 0) +
       merge_potential c_const (nth i (x :: xs) 0)
         (nth (S i) (x :: xs) 0) y)
      by ring.
    apply Qplus_le_compat.
    + assert (Hi' : (i < length (x :: xs))%nat) by lia.
      exact (IH Hi').
    + rewrite (merge_potential_target_irrelevant
                 c_const (nth i (x :: xs) 0)
                 (nth (S i) (x :: xs) 0) z y).
      apply Qle_refl.
Qed.

Lemma msm_triangle_cell_potential_0j0 :
  forall x xs y ys z zs c_const j,
    0 <= c_const ->
    (j < length (y :: ys))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c_const 0 j 0.
Proof.
  intros x xs y ys z zs c_const j Hc.
  induction j as [|j IH]; intros Hj.
  - apply msm_triangle_cell_potential_base.
  - unfold msm_triangle_cell_potential_bound.
    rewrite (msm_matrix_cell_first_row_potential_eq x xs y ys c_const j)
      by exact Hj.
    rewrite (msm_matrix_cell_first_col_potential_eq y ys z zs c_const j)
      by exact Hj.
    eapply Qle_trans.
    + assert (Hj' : (j < length (y :: ys))%nat) by lia.
      exact (IH Hj').
    + setoid_replace
        ((msm_cell_potential x xs y ys c_const 0 j +
          split_potential c_const x (nth j (y :: ys) 0)
            (nth (S j) (y :: ys) 0)) +
         (msm_cell_potential y ys z zs c_const j 0 +
          merge_potential c_const (nth j (y :: ys) 0)
            (nth (S j) (y :: ys) 0) z))
        with
        ((msm_cell_potential x xs y ys c_const 0 j +
          msm_cell_potential y ys z zs c_const j 0) +
         (split_potential c_const x (nth j (y :: ys) 0)
            (nth (S j) (y :: ys) 0) +
          merge_potential c_const (nth j (y :: ys) 0)
            (nth (S j) (y :: ys) 0) z))
        by ring.
      apply Qle_plus_nonneg_r.
      apply split_merge_potential_sum_nonneg.
      exact Hc.
Qed.

Lemma msm_triangle_cell_split_merge_relax :
  forall x xs y ys z zs c_const i j k,
    0 <= c_const ->
    msm_triangle_cell_potential_bound x xs y ys z zs c_const i j k ->
    msm_cell_potential x xs z zs c_const i k <=
      (msm_cell_potential x xs y ys c_const i j +
       split_potential c_const
         (nth i (x :: xs) 0)
         (nth j (y :: ys) 0)
         (nth (S j) (y :: ys) 0)) +
      (msm_cell_potential y ys z zs c_const j k +
       merge_potential c_const
         (nth j (y :: ys) 0)
         (nth (S j) (y :: ys) 0)
         (nth k (z :: zs) 0)).
Proof.
  intros x xs y ys z zs c_const i j k Hc Hih.
  unfold msm_triangle_cell_potential_bound in Hih.
  eapply Qle_trans.
  - exact Hih.
  - setoid_replace
      ((msm_cell_potential x xs y ys c_const i j +
        split_potential c_const (nth i (x :: xs) 0)
          (nth j (y :: ys) 0) (nth (S j) (y :: ys) 0)) +
       (msm_cell_potential y ys z zs c_const j k +
        merge_potential c_const (nth j (y :: ys) 0)
          (nth (S j) (y :: ys) 0) (nth k (z :: zs) 0)))
      with
      ((msm_cell_potential x xs y ys c_const i j +
        msm_cell_potential y ys z zs c_const j k) +
       (split_potential c_const (nth i (x :: xs) 0)
          (nth j (y :: ys) 0) (nth (S j) (y :: ys) 0) +
        merge_potential c_const (nth j (y :: ys) 0)
          (nth (S j) (y :: ys) 0) (nth k (z :: zs) 0)))
      by ring.
    apply Qle_plus_nonneg_r.
    apply split_merge_potential_sum_nonneg.
    exact Hc.
Qed.

(** * Interior potential expansion, cube symmetry, and the missing faces/interior

    The lemmas below close the all-tail triangle case. They express an interior
    [msm_cell_potential] as the [Qmin3] of its three predecessor potentials plus
    the corresponding operation potential, then drive a single strong induction
    on [i + j + k] through the base case, the three index-cube edges (already
    proven above), the three faces, and the strict interior. *)

(** Subtracting a constant distributes over [Qmin2]/[Qmin3]. *)
Lemma Qminus_le_mono_l : forall a b h, a <= b -> a - h <= b - h.
Proof.
  intros a b h H. unfold Qminus.
  apply Qplus_le_compat; [exact H | apply Qle_refl].
Qed.

Lemma Qmin2_minus : forall a b h, Qmin2 a b - h == Qmin2 (a - h) (b - h).
Proof.
  intros a b h. unfold Qmin2.
  destruct (Qle_bool a b) eqn:Hab; destruct (Qle_bool (a - h) (b - h)) eqn:Hab'.
  - reflexivity.
  - exfalso.
    apply Qle_bool_iff in Hab. apply Qle_bool_false_lt in Hab'.
    apply (Qlt_irrefl (a - h)).
    apply Qle_lt_trans with (y := b - h); [apply Qminus_le_mono_l; exact Hab | exact Hab'].
  - exfalso.
    apply Qle_bool_false_lt in Hab. apply Qle_bool_iff in Hab'.
    apply (Qlt_irrefl (b - h)).
    apply Qlt_le_trans with (y := a - h).
    + unfold Qminus. apply Qplus_lt_le_compat; [exact Hab | apply Qle_refl].
    + exact Hab'.
  - reflexivity.
Qed.

Lemma Qmin3_minus : forall a b c h, Qmin3 a b c - h == Qmin3 (a - h) (b - h) (c - h).
Proof.
  intros a b c h. unfold Qmin3.
  rewrite (Qmin2_minus a (Qmin2 b c) h).
  apply Qmin2_wd; [reflexivity | apply Qmin2_minus].
Qed.

(** An interior cell potential equals the [Qmin3] of its three predecessor
    potentials, each shifted by the matching operation potential. *)
Lemma msm_cell_potential_interior_eq :
  forall x xs y ys c i j,
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    msm_cell_potential x xs y ys c (S i) (S j) ==
    Qmin3
      (msm_cell_potential x xs y ys c i j +
         move_potential (nth i (x :: xs) 0) (nth j (y :: ys) 0)
                        (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0))
      (msm_cell_potential x xs y ys c i (S j) +
         merge_potential c (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0)
                           (nth (S j) (y :: ys) 0))
      (msm_cell_potential x xs y ys c (S i) j +
         split_potential c (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0)
                           (nth (S j) (y :: ys) 0)).
Proof.
  intros x xs y ys c i j Hi Hj.
  unfold msm_cell_potential.
  rewrite (msm_matrix_cell_interior x xs y ys c i j Hi Hj).
  rewrite Qmin3_minus.
  apply Qmin3_wd;
    unfold move_potential, merge_potential, split_potential; ring.
Qed.

(** The cell potential is symmetric under transposing the two series. *)
Lemma msm_cell_potential_sym :
  forall x xs y ys c i j,
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    msm_cell_potential x xs y ys c i j ==
    msm_cell_potential y ys x xs c j i.
Proof.
  intros x xs y ys c i j Hi Hj.
  unfold msm_cell_potential.
  rewrite (msm_matrix_cell_sym x xs y ys c i j Hi Hj).
  unfold half_abs.
  rewrite (Qabs_diff_symm (nth i (x :: xs) 0) (nth j (y :: ys) 0)).
  reflexivity.
Qed.

(** Reversing source/target (swapping X and Z, keeping the middle Y) preserves
    the triangle cell bound; lets us obtain the [k = 0] face from the [i = 0]
    face by transposition. *)
Lemma msm_triangle_cell_potential_reverse :
  forall x xs y ys z zs c i j k,
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    (k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound z zs y ys x xs c k j i ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i j k.
Proof.
  intros x xs y ys z zs c i j k Hi Hj Hk H.
  unfold msm_triangle_cell_potential_bound in *.
  rewrite (msm_cell_potential_sym x xs z zs c i k Hi Hk).
  rewrite (msm_cell_potential_sym x xs y ys c i j Hi Hj).
  rewrite (msm_cell_potential_sym y ys z zs c j k Hj Hk).
  (* Goal: P_ZX(k,i) <= P_YX(j,i) + P_ZY(k,j); H: P_ZX(k,i) <= P_ZY(k,j) + P_YX(j,i) *)
  eapply Qle_trans; [exact H | rewrite Qplus_comm; apply Qle_refl].
Qed.

(** ** Face [j = 0]: XY is a first column, YZ a first row, XZ full.
    Induct on [i]; the XZ cell only ever needs its merge branch, whose potential
    matches XY's by [merge_potential_target_irrelevant]. *)
Lemma msm_triangle_cell_potential_i0k :
  forall x xs y ys z zs c i k,
    0 <= c ->
    (i < length (x :: xs))%nat ->
    (k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i 0 k.
Proof.
  intros x xs y ys z zs c.
  induction i as [|i' IH]; intros k Hc Hi Hk.
  - apply msm_triangle_cell_potential_00k. exact Hk.
  - assert (Hi' : (i' < length (x :: xs))%nat) by lia.
    unfold msm_triangle_cell_potential_bound.
    rewrite (msm_matrix_cell_first_col_potential_eq x xs y ys c i' Hi).
    destruct k as [|k'].
    + assert (H0 : (0 < length (z :: zs))%nat) by (simpl; lia).
      rewrite (msm_matrix_cell_first_col_potential_eq x xs z zs c i' Hi).
      rewrite (merge_potential_target_irrelevant c (nth i' (x :: xs) 0)
                 (nth (S i') (x :: xs) 0) z y).
      pose proof (IH 0%nat Hc Hi' H0) as Hih.
      unfold msm_triangle_cell_potential_bound in Hih.
      unfold msm_cell_potential in *. psatz Q.
    + pose proof (msm_matrix_cell_merge_potential_step x xs z zs c i' k' Hi Hk) as Hstep.
      rewrite (merge_potential_target_irrelevant c (nth i' (x :: xs) 0)
                 (nth (S i') (x :: xs) 0) (nth (S k') (z :: zs) 0) y) in Hstep.
      pose proof (IH (S k') Hc Hi' Hk) as Hih.
      unfold msm_triangle_cell_potential_bound in Hih.
      unfold msm_cell_potential in *. psatz Q.
Qed.

(** ** Face [i = 0]: XY and XZ are first rows, YZ is full.
    Strong induction on [j + k]; expand YZ via [msm_cell_potential_interior_eq]
    and split the single [Qmin3] with [Qplus_Qmin3_glb]. *)
Lemma msm_triangle_cell_potential_0jk :
  forall x xs y ys z zs c j k,
    0 <= c ->
    (j < length (y :: ys))%nat ->
    (k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c 0 j k.
Proof.
  intros x xs y ys z zs c.
  assert (Hstrong : forall n j k, (j + k <= n)%nat -> 0 <= c ->
            (j < length (y :: ys))%nat -> (k < length (z :: zs))%nat ->
            msm_triangle_cell_potential_bound x xs y ys z zs c 0 j k).
  { induction n as [|n IHn]; intros j k Hn Hc Hj Hk.
    - assert (Hjk : j = 0%nat /\ k = 0%nat) by lia.
      destruct Hjk as [Hj0 Hk0]. subst.
      apply msm_triangle_cell_potential_00k. exact Hk.
    - destruct j as [|j']; destruct k as [|k'].
      + apply msm_triangle_cell_potential_00k. exact Hk.
      + apply msm_triangle_cell_potential_00k. exact Hk.
      + apply msm_triangle_cell_potential_0j0; [exact Hc | exact Hj].
      + assert (Hj' : (j' < length (y :: ys))%nat) by lia.
        assert (Hk' : (k' < length (z :: zs))%nat) by lia.
        unfold msm_triangle_cell_potential_bound.
        rewrite (msm_cell_potential_interior_eq y ys z zs c j' k' Hj Hk).
        apply Qplus_Qmin3_glb.
        * (* YZ move branch *)
          rewrite (msm_matrix_cell_first_row_potential_eq x xs z zs c k' Hk).
          rewrite (msm_matrix_cell_first_row_potential_eq x xs y ys c j' Hj).
          pose proof (IHn j' k' ltac:(lia) Hc Hj' Hk') as Hih.
          unfold msm_triangle_cell_potential_bound in Hih.
          pose proof (split_move_potential_triangle c x (nth j' (y :: ys) 0)
                        (nth k' (z :: zs) 0) (nth (S j') (y :: ys) 0)
                        (nth (S k') (z :: zs) 0)) as Hpair.
          unfold msm_cell_potential in *. psatz Q.
        * (* YZ merge branch *)
          rewrite (msm_matrix_cell_first_row_potential_eq x xs y ys c j' Hj).
          pose proof (IHn j' (S k') ltac:(lia) Hc Hj' Hk) as Hih.
          unfold msm_triangle_cell_potential_bound in Hih.
          pose proof (split_potential_nonneg c x (nth j' (y :: ys) 0)
                        (nth (S j') (y :: ys) 0) Hc) as Hsp.
          pose proof (merge_potential_nonneg c (nth j' (y :: ys) 0)
                        (nth (S j') (y :: ys) 0) (nth (S k') (z :: zs) 0) Hc) as Hmp.
          unfold msm_cell_potential in *. psatz Q.
        * (* YZ split branch *)
          rewrite (msm_matrix_cell_first_row_potential_eq x xs z zs c k' Hk).
          rewrite (split_potential_source_irrelevant c x (nth (S j') (y :: ys) 0)
                     (nth k' (z :: zs) 0) (nth (S k') (z :: zs) 0)).
          pose proof (IHn (S j') k' ltac:(lia) Hc Hj Hk') as Hih.
          unfold msm_triangle_cell_potential_bound in Hih.
          unfold msm_cell_potential in *. psatz Q. }
  intros j k Hc Hj Hk. apply (Hstrong (j + k)%nat j k); [lia | exact Hc | exact Hj | exact Hk].
Qed.

(** ** Face [k = 0]: obtained from the [i = 0] face by source/target reversal. *)
Lemma msm_triangle_cell_potential_ij0 :
  forall x xs y ys z zs c i j,
    0 <= c ->
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i j 0.
Proof.
  intros x xs y ys z zs c i j Hc Hi Hj.
  assert (Hk0 : (0 < length (z :: zs))%nat) by (simpl; lia).
  apply (msm_triangle_cell_potential_reverse x xs y ys z zs c i j 0 Hi Hj Hk0).
  apply (msm_triangle_cell_potential_0jk z zs y ys x xs c j i Hc Hj Hi).
Qed.

(** ** Strict interior: all three indices positive.
    Expand XY and YZ via [msm_cell_potential_interior_eq], split the double
    [Qmin3] with [Qplus_Qmin3_Qmin3_glb], and discharge the nine combinations.
    Six predecessor bounds are supplied as hypotheses (the strong induction in
    [msm_triangle_cell_potential_bound_all] provides them). *)
Lemma msm_triangle_cell_potential_interior :
  forall x xs y ys z zs c i j k,
    0 <= c ->
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    (S k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i j k ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i j (S k) ->
    msm_triangle_cell_potential_bound x xs y ys z zs c (S i) (S j) k ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i (S j) (S k) ->
    msm_triangle_cell_potential_bound x xs y ys z zs c (S i) j k ->
    msm_triangle_cell_potential_bound x xs y ys z zs c (S i) j (S k) ->
    msm_triangle_cell_potential_bound x xs y ys z zs c (S i) (S j) (S k).
Proof.
  intros x xs y ys z zs c i j k Hc Hi Hj Hk IH1 IH2 IH3 IH4 IH5 IH6.
  unfold msm_triangle_cell_potential_bound.
  rewrite (msm_cell_potential_interior_eq x xs y ys c i j Hi Hj).
  rewrite (msm_cell_potential_interior_eq y ys z zs c j k Hj Hk).
  apply Qplus_Qmin3_Qmin3_glb.
  - (* G1 (move,move): IH1, XZ move, move_move *)
    pose proof (msm_matrix_cell_move_potential_step x xs z zs c i k Hi Hk) as Hstep.
    pose proof (move_move_potential_triangle (nth i (x :: xs) 0) (nth j (y :: ys) 0)
                  (nth k (z :: zs) 0) (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0)
                  (nth (S k) (z :: zs) 0)) as Hpair.
    unfold msm_triangle_cell_potential_bound in IH1.
    unfold msm_cell_potential in *. psatz Q.
  - (* G2 (move,merge): IH2, XZ merge, move_merge *)
    pose proof (msm_matrix_cell_merge_potential_step x xs z zs c i k Hi Hk) as Hstep.
    pose proof (move_merge_potential_triangle c (nth i (x :: xs) 0) (nth j (y :: ys) 0)
                  (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0)
                  (nth (S k) (z :: zs) 0)) as Hpair.
    unfold msm_triangle_cell_potential_bound in IH2.
    unfold msm_cell_potential in *. psatz Q.
  - (* G3 (move,split): IH3, XZ split + split_source_irrel + XY move *)
    pose proof (msm_matrix_cell_split_potential_step x xs z zs c i k Hi Hk) as Hstep.
    rewrite (split_potential_source_irrelevant c (nth (S i) (x :: xs) 0)
               (nth (S j) (y :: ys) 0) (nth k (z :: zs) 0) (nth (S k) (z :: zs) 0)) in Hstep.
    pose proof (msm_matrix_cell_move_potential_step x xs y ys c i j Hi Hj) as Hxy.
    unfold msm_triangle_cell_potential_bound in IH3.
    unfold msm_cell_potential in *. psatz Q.
  - (* G4 (merge,move): IH4, XZ merge + merge_target_irrel + YZ move *)
    pose proof (msm_matrix_cell_merge_potential_step x xs z zs c i k Hi Hk) as Hstep.
    rewrite (merge_potential_target_irrelevant c (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0)
               (nth (S k) (z :: zs) 0) (nth (S j) (y :: ys) 0)) in Hstep.
    pose proof (msm_matrix_cell_move_potential_step y ys z zs c j k Hj Hk) as Hyz.
    unfold msm_triangle_cell_potential_bound in IH4.
    unfold msm_cell_potential in *. psatz Q.
  - (* G5 (merge,merge): IH4, XZ merge + merge_target_irrel + YZ merge *)
    pose proof (msm_matrix_cell_merge_potential_step x xs z zs c i k Hi Hk) as Hstep.
    rewrite (merge_potential_target_irrelevant c (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0)
               (nth (S k) (z :: zs) 0) (nth (S j) (y :: ys) 0)) in Hstep.
    pose proof (msm_matrix_cell_merge_potential_step y ys z zs c j k Hj Hk) as Hyz.
    unfold msm_triangle_cell_potential_bound in IH4.
    unfold msm_cell_potential in *. psatz Q.
  - (* G6 (merge,split): IH4, XZ merge + merge_target_irrel + YZ split *)
    pose proof (msm_matrix_cell_merge_potential_step x xs z zs c i k Hi Hk) as Hstep.
    rewrite (merge_potential_target_irrelevant c (nth i (x :: xs) 0) (nth (S i) (x :: xs) 0)
               (nth (S k) (z :: zs) 0) (nth (S j) (y :: ys) 0)) in Hstep.
    pose proof (msm_matrix_cell_split_potential_step y ys z zs c j k Hj Hk) as Hyz.
    unfold msm_triangle_cell_potential_bound in IH4.
    unfold msm_cell_potential in *. psatz Q.
  - (* G7 (split,move): IH5, XZ split + split_move *)
    pose proof (msm_matrix_cell_split_potential_step x xs z zs c i k Hi Hk) as Hstep.
    pose proof (split_move_potential_triangle c (nth (S i) (x :: xs) 0) (nth j (y :: ys) 0)
                  (nth k (z :: zs) 0) (nth (S j) (y :: ys) 0) (nth (S k) (z :: zs) 0)) as Hpair.
    unfold msm_triangle_cell_potential_bound in IH5.
    unfold msm_cell_potential in *. psatz Q.
  - (* G8 (split,merge): IH6 via the relax lemma (exactly this branch) *)
    exact (msm_triangle_cell_split_merge_relax x xs y ys z zs c (S i) j (S k) Hc IH6).
  - (* G9 (split,split): IH3, XZ split + split_source_irrel + XY split *)
    pose proof (msm_matrix_cell_split_potential_step x xs z zs c i k Hi Hk) as Hstep.
    rewrite (split_potential_source_irrelevant c (nth (S i) (x :: xs) 0)
               (nth (S j) (y :: ys) 0) (nth k (z :: zs) 0) (nth (S k) (z :: zs) 0)) in Hstep.
    pose proof (msm_matrix_cell_split_potential_step x xs y ys c i j Hi Hj) as Hxy.
    unfold msm_triangle_cell_potential_bound in IH3.
    unfold msm_cell_potential in *. psatz Q.
Qed.

(** ** All cells: single strong induction on [i + j + k] dispatching to the
    base case, the three edges, the three faces, and the strict interior. *)
Lemma msm_triangle_cell_potential_bound_all :
  forall x xs y ys z zs c i j k,
    0 <= c ->
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    (k < length (z :: zs))%nat ->
    msm_triangle_cell_potential_bound x xs y ys z zs c i j k.
Proof.
  intros x xs y ys z zs c.
  assert (Hstrong : forall n i j k, (i + j + k <= n)%nat -> 0 <= c ->
            (i < length (x :: xs))%nat -> (j < length (y :: ys))%nat ->
            (k < length (z :: zs))%nat ->
            msm_triangle_cell_potential_bound x xs y ys z zs c i j k).
  { induction n as [|n IHn]; intros i j k Hn Hc Hi Hj Hk.
    - assert (Hijk : i = 0%nat /\ j = 0%nat /\ k = 0%nat) by lia.
      destruct Hijk as [Hi0 [Hj0 Hk0]]. subst.
      apply msm_triangle_cell_potential_base.
    - destruct i as [|i']; destruct j as [|j']; destruct k as [|k'].
      + apply msm_triangle_cell_potential_base.
      + apply msm_triangle_cell_potential_00k. exact Hk.
      + apply msm_triangle_cell_potential_0j0; [exact Hc | exact Hj].
      + apply msm_triangle_cell_potential_0jk; [exact Hc | exact Hj | exact Hk].
      + apply msm_triangle_cell_potential_i00. exact Hi.
      + apply msm_triangle_cell_potential_i0k; [exact Hc | exact Hi | exact Hk].
      + apply msm_triangle_cell_potential_ij0; [exact Hc | exact Hi | exact Hj].
      + apply (msm_triangle_cell_potential_interior x xs y ys z zs c i' j' k' Hc Hi Hj Hk).
        * apply (IHn i' j' k'); [lia | exact Hc | lia | lia | lia].
        * apply (IHn i' j' (S k')); [lia | exact Hc | lia | lia | exact Hk].
        * apply (IHn (S i') (S j') k'); [lia | exact Hc | exact Hi | exact Hj | lia].
        * apply (IHn i' (S j') (S k')); [lia | exact Hc | lia | exact Hj | exact Hk].
        * apply (IHn (S i') j' k'); [lia | exact Hc | exact Hi | lia | lia].
        * apply (IHn (S i') j' (S k')); [lia | exact Hc | exact Hi | lia | exact Hk]. }
  intros i j k Hc Hi Hj Hk.
  apply (Hstrong (i + j + k)%nat i j k); [lia | exact Hc | exact Hi | exact Hj | exact Hk].
Qed.
