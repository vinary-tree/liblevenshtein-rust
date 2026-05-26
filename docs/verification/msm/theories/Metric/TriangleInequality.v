(** * MSM Triangle Inequality

    This module proves the triangle inequality for the MSM metric:

    MSM(X, Z) <= MSM(X, Y) + MSM(Y, Z)

    The proof uses trace composition: given optimal traces T1: X -> Y and T2: Y -> Z,
    we construct a trace T3: X -> Z with cost at most cost(T1) + cost(T2).

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia Psatz.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction CFunctionBounds MsmDistance.
From Liblevenshtein.MSM Require Import Symmetry TriangleAllTailSupport.

(** * Triangle Inequality: case dispatch

    Earlier revisions delegated the all-tail (all three series non-empty) case to
    an [MsmTriangleEvidence] record carrying the trace-composition obligation from
    Stefan et al., "The move-split-merge metric for time series", IEEE TKDE 25.6
    (2012): 1425-1438. That obligation is now discharged directly by
    [msm_triangle_all_tails] (proved below from the matrix-cell potential bound in
    [TriangleAllTailSupport]), so the record and its [MsmTriangleRemainingCase]
    index have been removed: every case is now proved unconditionally. *)

(** * Trace Composition for MSM *)

(** Given traces T1: X -> Y and T2: Y -> Z, we need to compose them into T3: X -> Z.

    The key challenge is that MSM operations involve context (the C function uses
    adjacent values). When composing traces, we must carefully handle this context.
*)

(** * Direct DP-Based Proof Approach *)

(** Instead of trace composition, we can prove triangle inequality directly
    on the DP recurrence by showing:

    For all i, j, k:
    Cost_XZ(i, k) <= Cost_XY(i, j) + Cost_YZ(j, k)

    where the intermediate j ranges over all valid positions in Y.
*)

(** Helper: |a - c| <= |a - b| + |b - c| (triangle inequality for absolute values) *)
Lemma qabs_triangle : forall a b c,
  Qabs (a - c) <= Qabs (a - b) + Qabs (b - c).
Proof.
  intros a b c.
  setoid_replace (a - c) with ((a - b) + (b - c)) by ring.
  apply Qabs_triangle.
Qed.

(** The Move operation satisfies triangle inequality *)
Lemma move_triangle : forall x y z,
  Qabs_diff x z <= Qabs_diff x y + Qabs_diff y z.
Proof.
  intros x y z.
  unfold Qabs_diff.
  apply qabs_triangle.
Qed.

Lemma last_nonempty_default_irrelevant : forall (A : Type) (a : A) l d1 d2,
  last (a :: l) d1 = last (a :: l) d2.
Proof.
  intros A a l.
  revert a.
  induction l as [|b l IH]; intros a d1 d2.
  - reflexivity.
  - simpl. apply IH.
Qed.

Lemma last_snoc : forall (A : Type) (l : list A) a d,
  last (l ++ [a]) d = a.
Proof.
  induction l as [|x xs IH]; intros a d.
  - reflexivity.
  - simpl.
    destruct xs as [|y ys].
    + reflexivity.
    + simpl. apply IH.
Qed.

Lemma qhalf_le_compat : forall a b,
  a <= b ->
  a * (1#2) <= b * (1#2).
Proof.
  intros a b Hab.
  apply Qmult_le_compat_r.
  - exact Hab.
  - unfold Qle. simpl. lia.
Qed.

Lemma Qle_minus_compat_r : forall a b c,
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

Lemma msm_compute_rows_singleton_target_potential :
  forall xs x_prev y z cost_y cost_z c_const,
    cost_z - Qabs_diff x_prev z * (1#2) <=
      cost_y - Qabs_diff x_prev y * (1#2) + Qabs_diff y z * (1#2) ->
    last (msm_compute_rows xs x_prev [z] z [cost_z] c_const) 0 -
      Qabs_diff (last (x_prev :: xs) x_prev) z * (1#2) <=
    last (msm_compute_rows xs x_prev [y] y [cost_y] c_const) 0 -
      Qabs_diff (last (x_prev :: xs) x_prev) y * (1#2) +
      Qabs_diff y z * (1#2).
Proof.
  induction xs as [|x xs IH]; intros x_prev y z cost_y cost_z c_const Hpot.
  - simpl. exact Hpot.
  - simpl.
    replace (match xs with
             | [] => x
             | _ :: _ => last xs x_prev
             end)
      with (last (x :: xs) x).
    2:{
      destruct xs as [|x2 xs2]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
    }
    apply IH.
    setoid_replace
      (cost_z + c_func c_const x x_prev z - Qabs_diff x z * (1#2))
      with
      ((cost_z - Qabs_diff x_prev z * (1#2)) +
       (c_const + Qabs_diff x x_prev * (1#2))).
    2:{
      pose proof (c_func_singleton_target_potential c_const x x_prev z) as Hz.
      setoid_replace (c_func c_const x x_prev z)
        with
        (c_const + (Qabs_diff x x_prev + Qabs_diff x z) * (1#2) -
          Qabs_diff x_prev z * (1#2)).
      - ring.
      - rewrite <- Hz. ring.
    }
    setoid_replace
      (cost_y + c_func c_const x x_prev y - Qabs_diff x y * (1#2) +
       Qabs_diff y z * (1#2))
      with
      ((cost_y - Qabs_diff x_prev y * (1#2) + Qabs_diff y z * (1#2)) +
       (c_const + Qabs_diff x x_prev * (1#2))).
    2:{
      pose proof (c_func_singleton_target_potential c_const x x_prev y) as Hy.
      setoid_replace (c_func c_const x x_prev y)
        with
        (c_const + (Qabs_diff x x_prev + Qabs_diff x y) * (1#2) -
          Qabs_diff x_prev y * (1#2)).
      - ring.
      - rewrite <- Hy. ring.
    }
    apply Qplus_le_compat.
    + exact Hpot.
    + apply Qle_refl.
Qed.

Lemma msm_compute_rows_singleton_target_bound :
  forall xs x_prev y z cost_y cost_z c_const,
    cost_z - Qabs_diff x_prev z * (1#2) <=
      cost_y - Qabs_diff x_prev y * (1#2) + Qabs_diff y z * (1#2) ->
    last (msm_compute_rows xs x_prev [z] z [cost_z] c_const) 0 <=
    last (msm_compute_rows xs x_prev [y] y [cost_y] c_const) 0 +
      Qabs_diff y z.
Proof.
  intros xs x_prev y z cost_y cost_z c_const Hpot.
  pose proof (msm_compute_rows_singleton_target_potential
                xs x_prev y z cost_y cost_z c_const Hpot) as Hfinal.
  set (x_last := last (x_prev :: xs) x_prev) in *.
  setoid_replace
    (last (msm_compute_rows xs x_prev [z] z [cost_z] c_const) 0)
    with
    ((last (msm_compute_rows xs x_prev [z] z [cost_z] c_const) 0 -
       Qabs_diff x_last z * (1#2)) +
      Qabs_diff x_last z * (1#2))
    by ring.
  eapply Qle_trans.
  - apply Qplus_le_compat.
    + exact Hfinal.
    + apply qhalf_le_compat.
      apply move_triangle.
  - setoid_replace
      ((last (msm_compute_rows xs x_prev [y] y [cost_y] c_const) 0 -
          Qabs_diff x_last y * (1#2) + Qabs_diff y z * (1#2)) +
        (Qabs_diff x_last y + Qabs_diff y z) * (1#2))
      with
      (last (msm_compute_rows xs x_prev [y] y [cost_y] c_const) 0 +
        Qabs_diff y z)
      by ring.
    apply Qle_refl.
Qed.

Lemma msm_triangle_source_tail : forall x x' xs y z c (Hc : 0 <= c),
  msm_distance (x :: x' :: xs) [z] {| msm_c := c; msm_c_nonneg := Hc |} <=
  msm_distance (x :: x' :: xs) [y] {| msm_c := c; msm_c_nonneg := Hc |} +
  msm_distance [y] [z] {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x x' xs y z c Hc.
  change (last (msm_compute_rows (x' :: xs) x [z] z [Qabs_diff x z] c) 0 <=
          last (msm_compute_rows (x' :: xs) x [y] y [Qabs_diff x y] c) 0 +
          Qabs_diff y z).
  apply msm_compute_rows_singleton_target_bound.
  fold (Qabs_diff x z).
  fold (Qabs_diff x y).
  fold (Qabs_diff y z).
  ring_simplify.
  setoid_replace ((1#2) * Qabs_diff x z)
    with (Qabs_diff x z * (1#2)) by ring.
  setoid_replace ((1#2) * Qabs_diff x y + (1#2) * Qabs_diff y z)
    with ((Qabs_diff x y + Qabs_diff y z) * (1#2)) by ring.
  apply qhalf_le_compat.
  apply move_triangle.
Qed.

Lemma msm_init_row_snoc_last : forall x1 y_prev ys y_new prev_cost c_const,
  last (msm_init_row x1 y_prev (ys ++ [y_new]) prev_cost c_const) 0 ==
  last (msm_init_row x1 y_prev ys prev_cost c_const) 0 +
  c_func c_const y_new x1 (last (y_prev :: ys) y_prev).
Proof.
  intros x1 y_prev ys.
  revert y_prev.
  induction ys as [|y ys IH]; intros y_prev y_new prev_cost c_const.
  - reflexivity.
  - rewrite (msm_init_row_cons_last x1 y_prev y (ys ++ [y_new]) prev_cost c_const).
    rewrite (msm_init_row_cons_last x1 y_prev y ys prev_cost c_const).
    replace (last (y_prev :: y :: ys) y_prev)
      with (last (y :: ys) y).
    + apply IH.
    + destruct ys as [|q qs]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
Qed.

Lemma msm_distance_singleton_source_snoc :
  forall y z zs z_new c (Hc : 0 <= c),
    msm_distance [y] ((z :: zs) ++ [z_new]) {| msm_c := c; msm_c_nonneg := Hc |} ==
    msm_distance [y] (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} +
    c_func c z_new y (last (z :: zs) z).
Proof.
  intros y z zs z_new c Hc.
  simpl.
  apply msm_init_row_snoc_last.
Qed.

Lemma msm_init_row_last_ge_start : forall x1 y_prev Y_tail prev_cost c_const,
  0 <= c_const ->
  prev_cost <= last (msm_init_row x1 y_prev Y_tail prev_cost c_const) 0.
Proof.
  intros x1 y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_cost c_const Hc.
  - simpl. apply Qle_refl.
  - rewrite msm_init_row_cons_last.
    eapply Qle_trans.
    + apply Qle_plus_nonneg_r.
      apply c_func_nonneg. exact Hc.
    + apply IH. exact Hc.
Qed.

Lemma msm_distance_singleton_left_ge_first : forall x y ys cfg,
  Qabs_diff x y <= msm_distance [x] (y :: ys) cfg.
Proof.
  intros x y ys cfg.
  simpl.
  apply msm_init_row_last_ge_start.
  apply msm_c_nonneg.
Qed.

Lemma msm_compute_rows_singleton_last_ge_start : forall xs x_prev y cost c_const,
  0 <= c_const ->
  cost <= last (msm_compute_rows xs x_prev [y] y [cost] c_const) 0.
Proof.
  induction xs as [|x xs IH]; intros x_prev y cost c_const Hc.
  - simpl. apply Qle_refl.
  - change (cost <=
      last (msm_compute_rows xs x [y] y
              [cost + c_func c_const x x_prev y] c_const) 0).
    eapply Qle_trans.
    + apply Qle_plus_nonneg_r.
      apply c_func_nonneg. exact Hc.
    + apply IH. exact Hc.
Qed.

Lemma msm_distance_singleton_right_ge_first : forall y ys z cfg,
  Qabs_diff y z <= msm_distance (y :: ys) [z] cfg.
Proof.
  intros y ys z cfg.
  simpl.
  apply msm_compute_rows_singleton_last_ge_start.
  apply msm_c_nonneg.
Qed.

Lemma msm_triangle_singleton_ends : forall x y ys z cfg,
  msm_distance [x] [z] cfg <=
  msm_distance [x] (y :: ys) cfg + msm_distance (y :: ys) [z] cfg.
Proof.
  intros x y ys z cfg.
  simpl.
  eapply Qle_trans.
  - apply move_triangle.
  - apply Qplus_le_compat.
    + apply msm_distance_singleton_left_ge_first.
    + apply msm_distance_singleton_right_ge_first.
Qed.

Lemma msm_triangle_target_tail_from_source_tail :
  (forall x x' xs y z c (Hc : 0 <= c),
    msm_distance (x :: x' :: xs) [z] {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: x' :: xs) [y] {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance [y] [z] {| msm_c := c; msm_c_nonneg := Hc |}) ->
  forall x y z z' zs c (Hc : 0 <= c),
    msm_distance [x] (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance [x] [y] {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance [y] (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros Hsource x y z z' zs c Hc.
  set (cfg := {| msm_c := c; msm_c_nonneg := Hc |}).
  change (msm_distance [x] (z :: z' :: zs) cfg <=
          msm_distance [x] [y] cfg + msm_distance [y] (z :: z' :: zs) cfg).
  rewrite (msm_symmetric [x] (z :: z' :: zs) cfg).
  rewrite (msm_symmetric [x] [y] cfg).
  rewrite (msm_symmetric [y] (z :: z' :: zs) cfg).
  setoid_replace
    (msm_distance [y] [x] cfg + msm_distance (z :: z' :: zs) [y] cfg)
    with
    (msm_distance (z :: z' :: zs) [y] cfg + msm_distance [y] [x] cfg)
    by ring.
  exact (Hsource z z' zs y x c Hc).
Qed.

Lemma msm_triangle_middle_target_tail_from_source_middle_tail :
  (forall x x' xs y y' ys z c (Hc : 0 <= c),
    msm_distance (x :: x' :: xs) [z] {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: x' :: xs) (y :: y' :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: y' :: ys) [z] {| msm_c := c; msm_c_nonneg := Hc |}) ->
  forall x y y' ys z z' zs c (Hc : 0 <= c),
    msm_distance [x] (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance [x] (y :: y' :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: y' :: ys) (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros Hsource_middle x y y' ys z z' zs c Hc.
  set (cfg := {| msm_c := c; msm_c_nonneg := Hc |}).
  change (msm_distance [x] (z :: z' :: zs) cfg <=
          msm_distance [x] (y :: y' :: ys) cfg +
          msm_distance (y :: y' :: ys) (z :: z' :: zs) cfg).
  rewrite (msm_symmetric [x] (z :: z' :: zs) cfg).
  rewrite (msm_symmetric [x] (y :: y' :: ys) cfg).
  rewrite (msm_symmetric (y :: y' :: ys) (z :: z' :: zs) cfg).
  setoid_replace
    (msm_distance (y :: y' :: ys) [x] cfg +
     msm_distance (z :: z' :: zs) (y :: y' :: ys) cfg)
    with
    (msm_distance (z :: z' :: zs) (y :: y' :: ys) cfg +
     msm_distance (y :: y' :: ys) [x] cfg)
    by ring.
  exact (Hsource_middle z z' zs y y' ys x c Hc).
Qed.

(** Extending the source by one sample is bounded by the merge branch of the
    executable recurrence. *)
Lemma msm_compute_row_last_le_merge : forall x_i x_prev y_prev Y_tail
    prev_row cost_left c_const,
  length prev_row = S (length Y_tail) ->
  cost_left <= hd 0 prev_row + c_func c_const x_i x_prev y_prev ->
  last (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 <=
  last prev_row 0 + c_func c_const x_i x_prev (last (y_prev :: Y_tail) y_prev).
Proof.
  intros x_i x_prev y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_row cost_left c_const Hlen Hleft.
  - destruct prev_row as [|p ps]; simpl in Hlen; [discriminate |].
    destruct ps as [|p' ps']; simpl in Hlen; [|discriminate].
    simpl in *. exact Hleft.
  - destruct prev_row as [|cost_diag prev_tail]; simpl in Hlen; [discriminate |].
    destruct prev_tail as [|cost_up rest]; simpl in Hlen; [discriminate |].
    simpl.
    set (cost_j := Qmin3 (cost_diag + Qabs_diff x_i y)
                           (cost_up + c_func c_const x_i x_prev y)
                           (cost_left + c_func c_const y x_i y_prev)).
    replace
      (match msm_compute_row x_i x_prev y ys (cost_up :: rest) cost_j c_const with
       | [] => cost_left
       | _ :: _ =>
           last (msm_compute_row x_i x_prev y ys (cost_up :: rest) cost_j c_const) 0
       end)
      with (last (msm_compute_row x_i x_prev y ys (cost_up :: rest) cost_j c_const) 0).
    2:{
      destruct (msm_compute_row x_i x_prev y ys (cost_up :: rest) cost_j c_const)
        as [|row_hd row_tl] eqn:Hrow; [|reflexivity].
      pose proof (msm_compute_row_length x_i x_prev y ys (cost_up :: rest)
                    cost_j c_const ltac:(simpl; lia)) as Hrow_len.
      rewrite Hrow in Hrow_len. simpl in Hrow_len. discriminate.
    }
    replace (match rest with
             | [] => cost_up
             | _ :: _ => last rest 0
             end)
      with (last (cost_up :: rest) 0) by (destruct rest; reflexivity).
    replace (match ys with
             | [] => y
             | _ :: _ => last ys y_prev
             end)
      with (last (y :: ys) y).
    2:{
      destruct ys as [|q qs]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
    }
    apply (IH y (cost_up :: rest) cost_j c_const).
    + simpl. lia.
    + subst cost_j. apply Qmin3_le_b.
Qed.

Lemma msm_compute_rows_snoc_source : forall xs x_prev Y y1 row x_new c_const,
  msm_compute_rows (xs ++ [x_new]) x_prev Y y1 row c_const =
  msm_compute_row x_new (last (x_prev :: xs) x_prev) y1 (tl Y)
    (msm_compute_rows xs x_prev Y y1 row c_const)
    (hd 0 (msm_compute_rows xs x_prev Y y1 row c_const) +
       c_func c_const x_new (last (x_prev :: xs) x_prev) y1)
    c_const.
Proof.
  induction xs as [|x xs IH]; intros x_prev Y y1 row x_new c_const.
  - reflexivity.
  - simpl. rewrite IH.
    replace (match xs with
             | [] => x
             | _ :: _ => last xs x_prev
             end)
      with (last (x :: xs) x).
    + reflexivity.
    + destruct xs as [|x' xs']; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
Qed.

Lemma msm_distance_snoc_source_le : forall x xs y ys x_new cfg,
  msm_distance ((x :: xs) ++ [x_new]) (y :: ys) cfg <=
  msm_distance (x :: xs) (y :: ys) cfg +
  c_func (msm_c cfg) x_new (last (x :: xs) x) (last (y :: ys) y).
Proof.
  intros x xs y ys x_new cfg.
  simpl.
  set (first_row := msm_init_row x y ys (Qabs_diff x y) (msm_c cfg)).
  rewrite msm_compute_rows_snoc_source.
  set (row_before := msm_compute_rows xs x (y :: ys) y first_row (msm_c cfg)).
  apply msm_compute_row_last_le_merge.
  - subst row_before first_row.
    apply msm_compute_rows_length.
    apply msm_init_row_length.
  - apply Qle_refl.
Qed.

Lemma msm_distance_snoc_target_le : forall x xs y ys y_new cfg,
  msm_distance (x :: xs) ((y :: ys) ++ [y_new]) cfg <=
  msm_distance (x :: xs) (y :: ys) cfg +
  c_func (msm_c cfg) y_new (last (x :: xs) x) (last (y :: ys) y).
Proof.
  intros x xs y ys y_new cfg.
  rewrite (msm_symmetric (x :: xs) ((y :: ys) ++ [y_new]) cfg).
  eapply Qle_trans.
  - apply msm_distance_snoc_source_le.
  - rewrite (msm_symmetric (y :: ys) (x :: xs) cfg).
    rewrite c_func_symm_bc.
    apply Qle_refl.
Qed.

Lemma msm_triangle_singleton_middle_target_potential :
  forall x xs y z zs c (Hc : 0 <= c),
    msm_distance (x :: xs) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} -
      Qabs_diff (last (x :: xs) x) (last (z :: zs) z) * (1#2) <=
    msm_distance (x :: xs) [y] {| msm_c := c; msm_c_nonneg := Hc |} -
      Qabs_diff (last (x :: xs) x) y * (1#2) +
    msm_distance [y] (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} -
      Qabs_diff y (last (z :: zs) z) * (1#2).
Proof.
  intros x xs y z zs c Hc.
  induction zs using rev_ind.
  - simpl.
    fold (Qabs_diff x z).
    fold (Qabs_diff x y).
    fold (Qabs_diff y z).
    replace (match xs with
             | [] => x
             | _ :: _ => last xs x
             end)
      with (last (x :: xs) x).
    2:{
      destruct xs as [|x2 xs2]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
    }
    setoid_replace
      (last (msm_compute_rows xs x [y] y [Qabs_diff x y] c) 0 -
        Qabs_diff (last (x :: xs) x) y * (1#2) +
        Qabs_diff y z - Qabs_diff y z * (1#2))
      with
      (last (msm_compute_rows xs x [y] y [Qabs_diff x y] c) 0 -
        Qabs_diff (last (x :: xs) x) y * (1#2) +
        Qabs_diff y z * (1#2))
      by ring.
    apply msm_compute_rows_singleton_target_potential.
    ring_simplify.
    setoid_replace ((1#2) * Qabs_diff x z)
      with (Qabs_diff x z * (1#2)) by ring.
    setoid_replace ((1#2) * Qabs_diff x y + (1#2) * Qabs_diff y z)
      with ((Qabs_diff x y + Qabs_diff y z) * (1#2)) by ring.
    apply qhalf_le_compat.
    apply move_triangle.
  - rewrite app_comm_cons.
    rewrite !last_snoc.
    set (cfg := {| msm_c := c; msm_c_nonneg := Hc |}).
    set (X := x :: xs).
    set (Zold := z :: zs).
    set (x_last := last X x).
    set (z_last := last Zold z).
    pose proof (msm_distance_snoc_target_le x xs z zs x0 cfg) as Hsnoc_x.
    pose proof (msm_distance_singleton_source_snoc y z zs x0 c Hc) as Hsnoc_y.
    change {| msm_c := c; msm_c_nonneg := Hc |} with cfg in Hsnoc_y.
    change (z :: zs) with Zold in Hsnoc_y.
    change (last Zold z) with z_last in Hsnoc_y.
    change (msm_distance X (Zold ++ [x0]) cfg -
              Qabs_diff x_last x0 * (1#2) <=
            msm_distance X [y] cfg - Qabs_diff x_last y * (1#2) +
            msm_distance [y] (Zold ++ [x0]) cfg -
              Qabs_diff y x0 * (1#2)).
    set (DXnew := msm_distance X (Zold ++ [x0]) cfg).
    set (DXold := msm_distance X Zold cfg).
    set (DXY := msm_distance X [y] cfg).
    set (DYnew := msm_distance [y] (Zold ++ [x0]) cfg).
    set (DYold := msm_distance [y] Zold cfg).
    set (Cx := c_func c x0 x_last z_last).
    set (Cy := c_func c x0 y z_last).
    set (Hxz_old := Qabs_diff x_last z_last * (1#2)).
    set (Hx_new := Qabs_diff x_last x0 * (1#2)).
    set (Hxy := Qabs_diff x_last y * (1#2)).
    set (Hyz_old := Qabs_diff y z_last * (1#2)).
    set (Hy_new := Qabs_diff y x0 * (1#2)).
    assert (Hsnoc_x' : DXnew <= DXold + Cx).
    { subst DXnew DXold Cx X Zold x_last z_last cfg. exact Hsnoc_x. }
    assert (Hsnoc_y' : DYnew == DYold + Cy).
    { subst DYnew DYold Cy Zold z_last cfg. exact Hsnoc_y. }
    assert (HIH :
      DXold - Hxz_old <= DXY - Hxy + DYold - Hyz_old).
    { subst DXold DXY DYold Hxz_old Hxy Hyz_old X Zold x_last z_last cfg.
      exact IHzs. }
    assert (Hstep : Cx + Hxz_old - Hx_new == Cy + Hyz_old - Hy_new).
    { subst Cx Cy Hxz_old Hx_new Hyz_old Hy_new.
      apply c_func_target_append_potential_step. }
    psatz Q.
Qed.

Lemma msm_triangle_source_target_tail :
  forall x x' xs y z z' zs c (Hc : 0 <= c),
    msm_distance (x :: x' :: xs) (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: x' :: xs) [y] {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance [y] (z :: z' :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x x' xs y z z' zs c Hc.
  pose proof (msm_triangle_singleton_middle_target_potential
                x (x' :: xs) y z (z' :: zs) c Hc) as Hpot.
  set (cfg := {| msm_c := c; msm_c_nonneg := Hc |}) in *.
  set (X := x :: x' :: xs) in *.
  set (Z := z :: z' :: zs) in *.
  set (x_last := last X x) in *.
  set (z_last := last Z z) in *.
  eapply Qle_trans.
  - setoid_replace (msm_distance X Z cfg)
      with
      ((msm_distance X Z cfg - Qabs_diff x_last z_last * (1#2)) +
       Qabs_diff x_last z_last * (1#2))
      by ring.
    apply Qplus_le_compat.
    + exact Hpot.
    + apply qhalf_le_compat.
      apply move_triangle.
  - setoid_replace
      ((msm_distance X [y] cfg - Qabs_diff x_last y * (1#2) +
        msm_distance [y] Z cfg - Qabs_diff y z_last * (1#2)) +
       (Qabs_diff x_last y + Qabs_diff y z_last) * (1#2))
      with
      (msm_distance X [y] cfg + msm_distance [y] Z cfg)
      by ring.
    apply Qle_refl.
Qed.

Fixpoint singleton_target_costs
    (prev : Q) (ys : list Q) (z cost c_const : Q) : list Q :=
  match ys with
  | [] => [cost]
  | y :: ys' =>
      cost :: singleton_target_costs y ys' z
        (cost + c_func c_const y prev z) c_const
  end.

Fixpoint row_singleton_target_potential
    (x z sx : Q) (ys row ycosts : list Q) : Prop :=
  match ys, row, ycosts with
  | [], [], [] => True
  | y :: ys', cell :: row', sy :: ycosts' =>
      sx - half_abs x z <= cell - half_abs x y + sy - half_abs y z /\
      row_singleton_target_potential x z sx ys' row' ycosts'
  | _, _, _ => False
  end.

Lemma singleton_target_costs_not_nil :
  forall prev ys z cost c_const,
    singleton_target_costs prev ys z cost c_const <> [].
Proof.
  intros prev ys z cost c_const.
  destruct ys; simpl; discriminate.
Qed.

Lemma singleton_target_costs_hd :
  forall prev ys z cost c_const,
    hd 0 (singleton_target_costs prev ys z cost c_const) = cost.
Proof.
  intros prev ys z cost c_const.
  destruct ys; reflexivity.
Qed.

Lemma singleton_target_costs_last : forall y ys z cost c_const,
  last (singleton_target_costs y ys z cost c_const) 0 ==
  last (msm_compute_rows ys y [z] z [cost] c_const) 0.
Proof.
  intros y ys.
  revert y.
  induction ys as [|y' ys IH]; intros y z cost c_const.
  - reflexivity.
  - simpl.
    destruct (singleton_target_costs y' ys z
                (cost + c_func c_const y' y z) c_const) as [|q qs] eqn:Hcosts.
    + exfalso.
      exact (singleton_target_costs_not_nil y' ys z
               (cost + c_func c_const y' y z) c_const Hcosts).
    + specialize (IH y' z (cost + c_func c_const y' y z) c_const).
      rewrite Hcosts in IH.
      simpl in IH.
      exact IH.
Qed.

Lemma row_singleton_target_potential_hd :
  forall x z sx y ys row ycosts,
    row_singleton_target_potential x z sx (y :: ys) row ycosts ->
    sx - half_abs x z <= hd 0 row - half_abs x y + hd 0 ycosts - half_abs y z.
Proof.
  intros x z sx y ys row ycosts Hpot.
  destruct row as [|cell row']; [contradiction |].
  destruct ycosts as [|sy ycosts']; [contradiction |].
  simpl in Hpot.
  exact (proj1 Hpot).
Qed.

Lemma row_singleton_target_potential_last :
  forall x z sx y ys row ycosts,
    row_singleton_target_potential x z sx (y :: ys) row ycosts ->
    sx - half_abs x z <=
      last row 0 - half_abs x (last (y :: ys) y) +
      last ycosts 0 - half_abs (last (y :: ys) y) z.
Proof.
  intros x z sx y ys.
  revert y.
  induction ys as [|y' ys IH]; intros y row ycosts Hpot.
  - destruct row as [|cell row']; [contradiction |].
    destruct ycosts as [|sy ycosts']; [contradiction |].
    simpl in Hpot.
    destruct Hpot as [Hhead Htail].
    destruct row' as [|cell' row'']; destruct ycosts' as [|sy' ycosts''];
      simpl in Htail; try contradiction.
    exact Hhead.
  - destruct row as [|cell row']; [contradiction |].
    destruct ycosts as [|sy ycosts']; [contradiction |].
    simpl in Hpot.
    destruct Hpot as [_ Htail].
    destruct row' as [|cell' row'']; [contradiction |].
    destruct ycosts' as [|sy' ycosts'']; [contradiction |].
    specialize (IH y' (cell' :: row'') (sy' :: ycosts'') Htail).
    simpl.
    replace (match row'' with
             | [] => cell'
             | _ :: _ => last row'' 0
             end)
      with (last (cell' :: row'') 0)
      by (destruct row''; reflexivity).
    replace (match ycosts'' with
             | [] => sy'
             | _ :: _ => last ycosts'' 0
             end)
      with (last (sy' :: ycosts'') 0)
      by (destruct ycosts''; reflexivity).
    replace (match ys with
             | [] => y'
             | _ :: _ => last ys y
             end)
      with (last (y' :: ys) y').
    2:{
      destruct ys as [|q qs]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
    }
    exact IH.
Qed.

Lemma msm_init_row_source_middle_potential_from :
  forall x y ys z row_cost y_cost c_const,
    0 <= c_const ->
    Qabs_diff x z - half_abs x z <=
      row_cost - half_abs x y + y_cost - half_abs y z ->
    row_singleton_target_potential x z (Qabs_diff x z)
      (y :: ys)
      (msm_init_row x y ys row_cost c_const)
      (singleton_target_costs y ys z y_cost c_const).
Proof.
  intros x y ys.
  revert y.
  induction ys as [|y' ys IH]; intros y z row_cost y_cost c_const Hc Hbase.
  - simpl. split; [exact Hbase | exact I].
  - simpl. split; [exact Hbase |].
    apply IH.
    + exact Hc.
    + eapply Qle_trans.
      * exact Hbase.
      * pose proof
          (c_func_left_potential_step_nonneg c_const x y' y z Hc) as Hstep.
        setoid_replace
          (row_cost + c_func c_const y' x y -
            half_abs x y' +
            (y_cost + c_func c_const y' y z) -
            half_abs y' z)
          with
          ((row_cost - half_abs x y + y_cost - half_abs y z) +
           (c_func c_const y' x y - half_abs x y' +
            c_func c_const y' y z - half_abs y' z +
            half_abs x y + half_abs y z))
          by ring.
        apply Qle_plus_nonneg_r.
        exact Hstep.
Qed.

Lemma msm_compute_row_source_middle_potential_from :
  forall x x_prev z sx_prev sx y ys prev_row cost_left y_cost c_const,
    0 <= c_const ->
    sx == sx_prev + c_func c_const x x_prev z ->
    row_singleton_target_potential x_prev z sx_prev (y :: ys) prev_row
      (singleton_target_costs y ys z y_cost c_const) ->
    sx - half_abs x z <= cost_left - half_abs x y + y_cost - half_abs y z ->
    row_singleton_target_potential x z sx (y :: ys)
      (msm_compute_row x x_prev y ys prev_row cost_left c_const)
      (singleton_target_costs y ys z y_cost c_const).
Proof.
  intros x x_prev z sx_prev sx y ys.
  revert y.
  induction ys as [|y' ys IH]; intros y prev_row cost_left y_cost c_const
                                  Hc Hsx Hprev Hleft.
  - simpl. split; [exact Hleft | exact I].
  - destruct prev_row as [|cost_diag [|cost_up rest]].
    + simpl in Hprev. contradiction.
    + simpl in Hprev. destruct Hprev as [_ Hbad]. contradiction.
    + simpl in Hprev.
      destruct Hprev as [Hdiag Htail].
      simpl.
      split; [exact Hleft |].
      set (sy := y_cost + c_func c_const y' y z).
      pose proof
        (row_singleton_target_potential_hd
           x_prev z sx_prev y' ys (cost_up :: rest)
           (singleton_target_costs y' ys z sy c_const) Htail) as Hup.
      simpl in Hup.
      rewrite singleton_target_costs_hd in Hup.
      set (cost_j := Qmin3 (cost_diag + Qabs_diff x y')
                            (cost_up + c_func c_const x x_prev y')
                            (cost_left + c_func c_const y' x y)).
      apply IH.
      * exact Hc.
      * exact Hsx.
      * exact Htail.
      * subst cost_j sy.
        setoid_replace
          (Qmin3 (cost_diag + Qabs_diff x y')
             (cost_up + c_func c_const x x_prev y')
             (cost_left + c_func c_const y' x y) -
            half_abs x y' +
            (y_cost + c_func c_const y' y z) -
            half_abs y' z)
          with
          ((- half_abs x y' +
             (y_cost + c_func c_const y' y z) -
             half_abs y' z) +
           Qmin3 (cost_diag + Qabs_diff x y')
             (cost_up + c_func c_const x x_prev y')
             (cost_left + c_func c_const y' x y))
          by ring.
        apply Qplus_Qmin3_glb.
        -- setoid_replace (sx - half_abs x z)
             with
             ((sx_prev - half_abs x_prev z) +
              (c_func c_const x x_prev z - half_abs x z +
               half_abs x_prev z)).
           2:{ rewrite Hsx. ring. }
           setoid_replace
             ((- half_abs x y' +
                (y_cost + c_func c_const y' y z) -
                half_abs y' z) +
              (cost_diag + Qabs_diff x y'))
             with
             ((cost_diag - half_abs x_prev y + y_cost - half_abs y z) +
              (Qabs_diff x y' - half_abs x y' +
               c_func c_const y' y z - half_abs y' z +
               half_abs y z + half_abs x_prev y))
             by ring.
           apply Qplus_le_compat.
           ++ exact Hdiag.
           ++ apply c_func_diag_potential_step.
        -- setoid_replace (sx - half_abs x z)
             with
             ((sx_prev - half_abs x_prev z) +
              (c_func c_const x x_prev z - half_abs x z +
               half_abs x_prev z)).
           2:{ rewrite Hsx. ring. }
           setoid_replace
             ((- half_abs x y' +
                (y_cost + c_func c_const y' y z) -
                half_abs y' z) +
              (cost_up + c_func c_const x x_prev y'))
             with
             ((cost_up - half_abs x_prev y' +
                 (y_cost + c_func c_const y' y z) - half_abs y' z) +
              (c_func c_const x x_prev y' - half_abs x y' +
               half_abs x_prev y'))
             by ring.
           apply Qplus_le_compat.
           ++ exact Hup.
           ++ rewrite (c_func_singleton_context_switch c_const x x_prev y' z).
              apply Qle_refl.
        -- eapply Qle_trans.
           ++ exact Hleft.
           ++ pose proof
                (c_func_left_potential_step_nonneg c_const x y' y z Hc)
                as Hstep.
              setoid_replace
                ((- half_abs x y' +
                   (y_cost + c_func c_const y' y z) -
                   half_abs y' z) +
                 (cost_left + c_func c_const y' x y))
                with
                ((cost_left - half_abs x y + y_cost - half_abs y z) +
                 (c_func c_const y' x y - half_abs x y' +
                  c_func c_const y' y z - half_abs y' z +
                  half_abs x y + half_abs y z))
                by ring.
              apply Qle_plus_nonneg_r.
              exact Hstep.
Qed.

Lemma msm_compute_rows_source_middle_potential_from :
  forall xs x_prev y ys z sx row y_cost c_const,
    0 <= c_const ->
    row_singleton_target_potential x_prev z sx (y :: ys) row
      (singleton_target_costs y ys z y_cost c_const) ->
    row_singleton_target_potential
      (last (x_prev :: xs) x_prev) z
      (last (msm_compute_rows xs x_prev [z] z [sx] c_const) 0)
      (y :: ys)
      (msm_compute_rows xs x_prev (y :: ys) y row c_const)
      (singleton_target_costs y ys z y_cost c_const).
Proof.
  induction xs as [|x xs IH]; intros x_prev y ys z sx row y_cost c_const Hc Hpot.
  - simpl. exact Hpot.
  - simpl.
    replace (match xs with
             | [] => x
             | _ :: _ => last xs x_prev
             end)
      with (last (x :: xs) x).
    2:{
      destruct xs as [|q qs]; [reflexivity |].
      simpl. apply last_nonempty_default_irrelevant.
    }
    apply IH.
    + exact Hc.
    + apply msm_compute_row_source_middle_potential_from with (sx_prev := sx).
      * exact Hc.
      * reflexivity.
      * exact Hpot.
      * pose proof
          (row_singleton_target_potential_hd
             x_prev z sx y ys row
             (singleton_target_costs y ys z y_cost c_const) Hpot) as Hfirst.
        setoid_replace
          (sx + c_func c_const x x_prev z - half_abs x z)
          with
          ((sx - half_abs x_prev z) +
           (c_func c_const x x_prev z - half_abs x z +
            half_abs x_prev z))
          by ring.
        setoid_replace
          (hd 0 row + c_func c_const x x_prev y - half_abs x y +
            y_cost - half_abs y z)
          with
          ((hd 0 row - half_abs x_prev y + y_cost - half_abs y z) +
           (c_func c_const x x_prev y - half_abs x y +
            half_abs x_prev y))
          by ring.
        apply Qplus_le_compat.
        -- rewrite singleton_target_costs_hd in Hfirst.
           exact Hfirst.
        -- rewrite (c_func_singleton_context_switch c_const x x_prev y z).
           apply Qle_refl.
Qed.

Lemma msm_triangle_source_middle_tail_potential :
  forall x xs y ys z c (Hc : 0 <= c),
    msm_distance (x :: xs) [z] {| msm_c := c; msm_c_nonneg := Hc |} -
      half_abs (last (x :: xs) x) z <=
    msm_distance (x :: xs) (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} -
      half_abs (last (x :: xs) x) (last (y :: ys) y) +
    msm_distance (y :: ys) [z] {| msm_c := c; msm_c_nonneg := Hc |} -
      half_abs (last (y :: ys) y) z.
Proof.
  intros x xs y ys z c Hc.
  change
    (last (msm_compute_rows xs x [z] z [Qabs_diff x z] c) 0 -
       half_abs (last (x :: xs) x) z <=
     last (msm_compute_rows xs x (y :: ys) y
            (msm_init_row x y ys (Qabs_diff x y) c) c) 0 -
       half_abs (last (x :: xs) x) (last (y :: ys) y) +
     last (msm_compute_rows ys y [z] z [Qabs_diff y z] c) 0 -
       half_abs (last (y :: ys) y) z).
  pose proof
    (msm_init_row_source_middle_potential_from
       x y ys z (Qabs_diff x y) (Qabs_diff y z) c Hc) as Hinit.
  assert (Hbase :
    Qabs_diff x z - half_abs x z <=
      Qabs_diff x y - half_abs x y + Qabs_diff y z - half_abs y z).
  { unfold half_abs.
    setoid_replace (Qabs_diff x z - Qabs_diff x z * (1#2))
      with (Qabs_diff x z * (1#2)) by ring.
    setoid_replace
      (Qabs_diff x y - Qabs_diff x y * (1#2) +
       Qabs_diff y z - Qabs_diff y z * (1#2))
      with ((Qabs_diff x y + Qabs_diff y z) * (1#2))
      by ring.
    apply qhalf_le_compat.
    apply move_triangle. }
  specialize (Hinit Hbase).
  pose proof
    (msm_compute_rows_source_middle_potential_from
       xs x y ys z (Qabs_diff x z)
       (msm_init_row x y ys (Qabs_diff x y) c)
       (Qabs_diff y z) c Hc Hinit) as Hrows.
  pose proof
    (row_singleton_target_potential_last
       (last (x :: xs) x) z
       (last (msm_compute_rows xs x [z] z [Qabs_diff x z] c) 0)
       y ys
       (msm_compute_rows xs x (y :: ys) y
          (msm_init_row x y ys (Qabs_diff x y) c) c)
       (singleton_target_costs y ys z (Qabs_diff y z) c)
       Hrows) as Hlast.
  rewrite (singleton_target_costs_last y ys z (Qabs_diff y z) c) in Hlast.
  exact Hlast.
Qed.

Lemma msm_triangle_source_middle_tail :
  forall x x' xs y y' ys z c (Hc : 0 <= c),
    msm_distance (x :: x' :: xs) [z] {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: x' :: xs) (y :: y' :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: y' :: ys) [z] {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x x' xs y y' ys z c Hc.
  pose proof
    (msm_triangle_source_middle_tail_potential
       x (x' :: xs) y (y' :: ys) z c Hc) as Hpot.
  set (cfg := {| msm_c := c; msm_c_nonneg := Hc |}) in *.
  set (X := x :: x' :: xs) in *.
  set (Y := y :: y' :: ys) in *.
  set (x_last := last X x) in *.
  set (y_last := last Y y) in *.
  eapply Qle_trans.
  - setoid_replace (msm_distance X [z] cfg)
      with
      ((msm_distance X [z] cfg - half_abs x_last z) +
       half_abs x_last z)
      by ring.
    apply Qplus_le_compat.
    + exact Hpot.
    + unfold half_abs.
      apply qhalf_le_compat.
      apply (move_triangle x_last y_last z).
  - unfold half_abs.
    ring_simplify.
    apply Qle_refl.
Qed.

(** Every non-empty DP cell still contains the initial move cost plus
    non-negative recurrence costs. *)
Lemma msm_init_row_ge_floor : forall x1 y_prev Y_tail prev_cost c_const floor,
  0 <= c_const ->
  floor <= prev_cost ->
  Forall (fun cost => floor <= cost)
    (msm_init_row x1 y_prev Y_tail prev_cost c_const).
Proof.
  intros x1 y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_cost c_const floor Hc Hprev.
  - simpl. constructor; [exact Hprev | constructor].
  - simpl. constructor.
    + exact Hprev.
    + apply IH.
      * exact Hc.
      * eapply Qle_trans.
        -- exact Hprev.
        -- apply Qle_plus_nonneg_r.
           apply c_func_nonneg. exact Hc.
Qed.

Lemma msm_compute_row_ge_floor : forall x_i x_prev y_prev Y_tail
    prev_row cost_left c_const floor,
  0 <= c_const ->
  Forall (fun cost => floor <= cost) prev_row ->
  floor <= cost_left ->
  Forall (fun cost => floor <= cost)
    (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const).
Proof.
  intros x_i x_prev y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_row cost_left c_const floor
                                    Hc Hprev Hleft.
  - simpl. constructor; [exact Hleft | constructor].
  - destruct prev_row as [|cost_diag [|cost_up rest]].
    + simpl. constructor; [exact Hleft | constructor].
    + simpl. constructor; [exact Hleft | constructor].
    + simpl. constructor.
      * exact Hleft.
      * apply IH.
        -- exact Hc.
        -- inversion Hprev as [|? ? _ Htail]. exact Htail.
        -- apply Qmin3_glb.
           ++ eapply Qle_trans.
              ** inversion Hprev as [|? ? Hdiag _]. exact Hdiag.
              ** apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
           ++ eapply Qle_trans.
              ** inversion Hprev as [|? ? _ Htail].
                 inversion Htail as [|? ? Hup _]. exact Hup.
              ** apply Qle_plus_nonneg_r.
                 apply c_func_nonneg. exact Hc.
           ++ eapply Qle_trans.
              ** exact Hleft.
              ** apply Qle_plus_nonneg_r.
                 apply c_func_nonneg. exact Hc.
Qed.

Lemma msm_compute_row_not_nil : forall x_i x_prev y_prev Y_tail
    prev_row cost_left c_const,
  msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const <> [].
Proof.
  intros x_i x_prev y_prev Y_tail prev_row cost_left c_const.
  destruct Y_tail as [|y ys]; destruct prev_row as [|p [|q rest]];
    simpl; discriminate.
Qed.

Lemma msm_compute_rows_ge_floor : forall xs x_prev Y y1 row c_const floor,
  0 <= c_const ->
  row <> [] ->
  Forall (fun cost => floor <= cost) row ->
  Forall (fun cost => floor <= cost)
    (msm_compute_rows xs x_prev Y y1 row c_const).
Proof.
  induction xs as [|x xs IH]; intros x_prev Y y1 row c_const floor Hc Hrow_nonempty Hrow.
  - simpl. exact Hrow.
  - simpl. apply IH.
    + exact Hc.
    + apply msm_compute_row_not_nil.
    + apply msm_compute_row_ge_floor.
      * exact Hc.
      * exact Hrow.
      * eapply Qle_trans with (y := hd 0 row).
        -- destruct row as [|r rs].
           ++ contradiction.
           ++ simpl. rewrite Forall_forall in Hrow.
              apply Hrow. left. reflexivity.
        -- apply Qle_plus_nonneg_r.
           apply c_func_nonneg. exact Hc.
Qed.

Lemma msm_distance_nonempty_ge_first : forall x xs y ys cfg,
  Qabs_diff x y <= msm_distance (x :: xs) (y :: ys) cfg.
Proof.
  intros x xs y ys cfg.
  simpl.
  set (first_row := msm_init_row x y ys (Qabs_diff x y) (msm_c cfg)).
  set (final_row := msm_compute_rows xs x (y :: ys) y first_row (msm_c cfg)).
  assert (Hfinal_len : length final_row = S (length ys)).
  { subst final_row first_row.
    apply msm_compute_rows_length.
    apply msm_init_row_length. }
  rewrite (last_eq_nth_by_length final_row 0 (length ys) Hfinal_len).
  assert (Hfloor : Forall (fun cost => Qabs_diff x y <= cost) final_row).
  { subst final_row first_row.
    apply msm_compute_rows_ge_floor.
    - apply msm_c_nonneg.
    - apply msm_init_row_not_nil.
    - apply msm_init_row_ge_floor.
      + apply msm_c_nonneg.
      + apply Qle_refl. }
  apply Forall_nth.
  - exact Hfloor.
  - rewrite Hfinal_len. lia.
Qed.

Lemma one_Q_nonneg : 0 <= 1#1.
Proof.
  unfold Qle. simpl. lia.
Qed.

Definition msm_triangle_counter_cfg : MsmConfig :=
  {| msm_c := 1#1; msm_c_nonneg := one_Q_nonneg |}.

(** The executable empty-series extension is not a full metric over all lists:
    the empty middle point can make the triangle inequality false. *)
Lemma msm_triangle_empty_middle_counterexample :
  ~ (msm_distance [0#1] [100#1] msm_triangle_counter_cfg <=
     msm_distance [0#1] [] msm_triangle_counter_cfg +
     msm_distance [] [100#1] msm_triangle_counter_cfg).
Proof.
  intro H.
  vm_compute in H.
  exact (H eq_refl).
Qed.

(** * Decomposed Triangle Dispatch Cases *)

Lemma msm_triangle_empty_ends_via_middle : forall y ys c (Hc : 0 <= c),
  msm_distance [] [] {| msm_c := c; msm_c_nonneg := Hc |} <=
  msm_distance [] (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
  msm_distance (y :: ys) [] {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros y ys c Hc.
  simpl.
  assert (Hlen : 0 <= inject_Z (Z.of_nat (length (y :: ys))) * c).
  { apply Qmult_le_0_compat.
    - apply inject_Z_of_nat_nonneg.
    - exact Hc. }
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat; exact Hlen.
Qed.

Lemma msm_triangle_empty_source_via_middle :
  forall y ys z zs c (Hc : 0 <= c),
    msm_distance [] (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance [] (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: ys) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros y ys z zs c Hc.
  change (qnat_mul (length (z :: zs)) c <=
          qnat_mul (length (y :: ys)) c +
          msm_distance (y :: ys) (z :: zs)
            {| msm_c := c; msm_c_nonneg := Hc |}).
  exact (msm_distance_length_target_lower (y :: ys) (z :: zs)
           {| msm_c := c; msm_c_nonneg := Hc |}).
Qed.

Lemma msm_triangle_empty_target_via_middle :
  forall x xs y ys c (Hc : 0 <= c),
    msm_distance (x :: xs) [] {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: xs) (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: ys) [] {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x xs y ys c Hc.
  change (qnat_mul (length (x :: xs)) c <=
          msm_distance (x :: xs) (y :: ys)
            {| msm_c := c; msm_c_nonneg := Hc |} +
          qnat_mul (length (y :: ys)) c).
  setoid_replace
    (msm_distance (x :: xs) (y :: ys)
       {| msm_c := c; msm_c_nonneg := Hc |} +
     qnat_mul (length (y :: ys)) c)
    with
    (qnat_mul (length (y :: ys)) c +
     msm_distance (x :: xs) (y :: ys)
       {| msm_c := c; msm_c_nonneg := Hc |}) by ring.
  exact (msm_distance_length_source_lower (x :: xs) (y :: ys)
           {| msm_c := c; msm_c_nonneg := Hc |}).
Qed.

(** The remaining all-tail case, proved directly from the matrix-cell triangle
    bound at the corner [(length xs, length ys, length zs)]. The three distances
    are rewritten to corner cells by [msm_distance_matrix_cell_last], reduced to
    the potential bound by [msm_triangle_cell_potential_to_bound], and closed by
    [msm_triangle_cell_potential_bound_all]. This discharges the former
    [MsmTriangleEvidence.msm_triangle_nonempty] obligation. *)
Lemma msm_triangle_all_tails :
  forall x xs y ys z zs c (Hc : 0 <= c),
    msm_distance (x :: xs) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: xs) (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: ys) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x xs y ys z zs c Hc.
  rewrite (msm_distance_matrix_cell_last x xs z zs {| msm_c := c; msm_c_nonneg := Hc |}).
  rewrite (msm_distance_matrix_cell_last x xs y ys {| msm_c := c; msm_c_nonneg := Hc |}).
  rewrite (msm_distance_matrix_cell_last y ys z zs {| msm_c := c; msm_c_nonneg := Hc |}).
  simpl (msm_c _).
  apply msm_triangle_cell_potential_to_bound.
  apply msm_triangle_cell_potential_bound_all;
    [ exact Hc | simpl; lia | simpl; lia | simpl; lia ].
Qed.

Lemma msm_triangle_nonempty_dispatch :
  forall x xs y ys z zs c (Hc : 0 <= c),
    msm_distance (x :: xs) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |} <=
    msm_distance (x :: xs) (y :: ys) {| msm_c := c; msm_c_nonneg := Hc |} +
    msm_distance (y :: ys) (z :: zs) {| msm_c := c; msm_c_nonneg := Hc |}.
Proof.
  intros x xs y ys z zs c Hc.
  destruct xs as [|x' xs']; destruct ys as [|y' ys']; destruct zs as [|z' zs'].
  - apply move_triangle.
  - apply msm_triangle_target_tail_from_source_tail.
    apply msm_triangle_source_tail.
  - change (msm_distance [x] [z] {| msm_c := c; msm_c_nonneg := Hc |} <=
            msm_distance [x] (y :: y' :: ys') {| msm_c := c; msm_c_nonneg := Hc |} +
            msm_distance (y :: y' :: ys') [z] {| msm_c := c; msm_c_nonneg := Hc |}).
    apply msm_triangle_singleton_ends.
  - apply msm_triangle_middle_target_tail_from_source_middle_tail.
    intros a a' as_ b b' bs d c0 Hc0.
    apply msm_triangle_source_middle_tail.
  - apply msm_triangle_source_tail.
  - apply msm_triangle_source_target_tail.
  - apply msm_triangle_source_middle_tail.
  - exact (msm_triangle_all_tails x (x' :: xs') y (y' :: ys') z (z' :: zs') c Hc).
Qed.

(** * Main Triangle Inequality *)

(** The triangle inequality for MSM.
    This is the most complex proof as it requires showing that
    optimal traces can be composed without increasing total cost. *)

Theorem msm_triangle : forall X Y Z cfg,
  Y <> [] ->
  msm_distance X Z cfg <= msm_distance X Y cfg + msm_distance Y Z cfg.
Proof.
  intros X Y Z cfg Hmiddle.
  destruct cfg as [c Hc].
  simpl.

  (* Case analysis on the structure of the series *)
  destruct Y as [|y ys].
  - contradiction.
  - destruct X as [|x xs]; destruct Z as [|z zs].
    + (* [], y::ys, [] *)
      exact (msm_triangle_empty_ends_via_middle y ys c Hc).
    + (* [], y::ys, z::zs *)
      exact (msm_triangle_empty_source_via_middle y ys z zs c Hc).
    + (* x::xs, y::ys, [] *)
      exact (msm_triangle_empty_target_via_middle x xs y ys c Hc).
    + (* x::xs, y::ys, z::zs *)
      exact (msm_triangle_nonempty_dispatch x xs y ys z zs c Hc).
Qed.
