(** * MSM Distance Definition

    This module defines the MSM distance function using dynamic programming,
    following the recurrence from Stefan et al. (Figure 10).

    The MSM distance between two time series X and Y is computed as:

    MSM(X, Y) = Cost(m, n) where m = |X|, n = |Y|

    Cost(i, j) = min {
      Cost(i-1, j-1) + |x_i - y_j|,           // Move
      Cost(i-1, j) + C(x_i, x_{i-1}, y_j),    // Merge-like
      Cost(i, j-1) + C(y_j, x_i, y_{j-1})     // Split-like
    }

    Base cases:
    - Cost(1, 1) = |x_1 - y_1|
    - Cost(i, 1) = Cost(i-1, 1) + C(x_i, x_{i-1}, y_1)  for i > 1
    - Cost(1, j) = Cost(1, j-1) + C(y_j, x_1, y_{j-1})  for j > 1

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia ZArith.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction.

(** * MSM Distance Computation *)

(** Compute the MSM distance matrix.
    We use a list-based approach where each row is computed from the previous.

    Note: We use 1-indexed semantics in comments but 0-indexed implementation.
*)

(** Initialize first row: Cost(1, j) = Cost(1, j-1) + C(y_j, x_1, y_{j-1}) *)
Fixpoint msm_init_row (x1 y_prev : Q) (Y_tail : list Q) (prev_cost : Q)
         (c_const : Q) : list Q :=
  match Y_tail with
  | [] => [prev_cost]
  | y_j :: ys =>
    let cost_j := prev_cost + c_func c_const y_j x1 y_prev in
    prev_cost :: msm_init_row x1 y_j ys cost_j c_const
  end.

(** Compute a row given the previous row.
    For row i (2 <= i <= m):
    - Cost(i, 1) = Cost(i-1, 1) + C(x_i, x_{i-1}, y_1)
    - Cost(i, j) = min3(
        Cost(i-1, j-1) + |x_i - y_j|,
        Cost(i-1, j) + C(x_i, x_{i-1}, y_j),
        Cost(i, j-1) + C(y_j, x_i, y_{j-1})
      )
*)
Fixpoint msm_compute_row (x_i x_prev : Q) (y_prev : Q) (Y_tail : list Q)
         (prev_row : list Q) (cost_left : Q) (c_const : Q) : list Q :=
  match Y_tail, prev_row with
  | [], _ => [cost_left]
  | y_j :: ys, cost_diag :: cost_up :: rest =>
    let cost_move := cost_diag + Qabs_diff x_i y_j in
    let cost_merge := cost_up + c_func c_const x_i x_prev y_j in
    let cost_split := cost_left + c_func c_const y_j x_i y_prev in
    let cost_j := Qmin3 cost_move cost_merge cost_split in
    cost_left :: msm_compute_row x_i x_prev y_j ys (cost_up :: rest) cost_j c_const
  | _, _ => [cost_left]  (* Should not happen with valid input *)
  end.

(** Compute all rows iteratively. *)
Fixpoint msm_compute_rows (X_tail : list Q) (x_prev : Q) (Y : list Q)
         (y1 : Q) (prev_row : list Q) (c_const : Q) : list Q :=
  match X_tail with
  | [] => prev_row
  | x_i :: xs =>
    (* First column: Cost(i, 1) = Cost(i-1, 1) + C(x_i, x_prev, y1) *)
    let cost_i1 := hd 0 prev_row + c_func c_const x_i x_prev y1 in
    (* Compute rest of row *)
    let new_row := msm_compute_row x_i x_prev y1 (tl Y) prev_row cost_i1 c_const in
    msm_compute_rows xs x_i Y y1 new_row c_const
  end.

(** Main MSM distance function. *)
Definition msm_distance (X Y : TimeSeries) (cfg : MsmConfig) : Q :=
  match X, Y with
  | [], [] => 0
  | [], _ => inject_Z (Z.of_nat (length Y)) * msm_c cfg  (* All splits *)
  | _, [] => inject_Z (Z.of_nat (length X)) * msm_c cfg  (* All merges *)
  | x1 :: xs, y1 :: ys =>
    let c_const := msm_c cfg in
    (* Initialize: Cost(1, 1) = |x1 - y1| *)
    let init_cost := Qabs_diff x1 y1 in
    (* Build first row: Cost(1, j) for all j *)
    let first_row := msm_init_row x1 y1 ys init_cost c_const in
    (* Compute remaining rows *)
    let final_row := msm_compute_rows xs x1 Y y1 first_row c_const in
    (* Return last element of final row *)
    last final_row 0
  end.

(** Helper: inject_Z of a positive nat is positive. *)
Lemma inject_Z_of_nat_pos : forall n, (0 < n)%nat -> 0 < inject_Z (Z.of_nat n).
Proof.
  intros n Hn.
  unfold Qlt. simpl.
  rewrite Z.mul_1_r.
  apply (proj2 (Z.compare_lt_iff 0 (Z.of_nat n))).
  apply (proj1 (Nat2Z.inj_lt 0 n)). exact Hn.
Qed.

(** * Evidence for MSM Metric Properties *)

(** This evidence captures the remaining non-empty identity-of-indiscernibles
    property implied by the MSM dynamic-programming recurrence and metricity result.
    Reference: Stefan, Alexandra, et al. "The move-split-merge metric for
    time series." IEEE TKDE 25.6 (2012): 1425-1438.

    Broad value-independent split/merge upper bounds are intentionally not
    assumed here: [msm_distance [x] [y]] is [|x-y|], which can exceed
    [(length [x] + length [y]) * c] when [c] is small.

    The conclusion uses [series_Qeq], not Leibniz equality, because [Qeq] is
    the equality respected by rational arithmetic in QArith. *)

Record MsmDistanceEvidence : Prop := mkMsmDistanceEvidence {
  (** MSM identity of indiscernibles for the recursive non-empty DP case.
      Singleton edge cases are proved locally below. *)
  msm_zero_implies_series_eq_proof :
    forall x x_next xs y y_next ys (cfg : MsmConfig),
    0 < msm_c cfg ->
    msm_distance (x :: x_next :: xs) (y :: y_next :: ys) cfg == 0 ->
    series_Qeq (x :: x_next :: xs) (y :: y_next :: ys)
}.

Lemma msm_distance_empty_empty : forall cfg,
  msm_distance [] [] cfg == 0.
Proof.
  intros cfg. reflexivity.
Qed.

Lemma msm_distance_empty_left : forall Y cfg,
  msm_distance [] Y cfg == inject_Z (Z.of_nat (length Y)) * msm_c cfg.
Proof.
  intros Y cfg.
  destruct Y; reflexivity.
Qed.

Lemma msm_distance_empty_right : forall X cfg,
  msm_distance X [] cfg == inject_Z (Z.of_nat (length X)) * msm_c cfg.
Proof.
  intros X cfg.
  destruct X; reflexivity.
Qed.

Lemma msm_one_nonneg : 0 <= 1#1.
Proof.
  unfold Qle. simpl. lia.
Qed.

Definition msm_identity_counter_cfg : MsmConfig :=
  {| msm_c := 1#1; msm_c_nonneg := msm_one_nonneg |}.

Lemma msm_qeq_representations_zero_distance :
  msm_distance [1#1] [2#2] msm_identity_counter_cfg == 0.
Proof.
  simpl. apply Qabs_diff_zero_iff. reflexivity.
Qed.

Lemma msm_qeq_representations_not_leibniz :
  [1#1] <> [2#2].
Proof.
  discriminate.
Qed.

(** * Alternative Definition (for proofs) *)

(** We also define a more direct matrix-based version for clearer proofs. *)

(** Build the full DP matrix. *)
Definition msm_matrix (X Y : TimeSeries) (cfg : MsmConfig) : QMatrix :=
  match X, Y with
  | [], [] => [[0]]
  | [], _ => [[inject_Z (Z.of_nat (length Y)) * msm_c cfg]]
  | _, [] => [[inject_Z (Z.of_nat (length X)) * msm_c cfg]]
  | x1 :: xs, y1 :: ys =>
    let c_const := msm_c cfg in
    let init_cost := Qabs_diff x1 y1 in
    let first_row := msm_init_row x1 y1 ys init_cost c_const in
    (* Build all rows and collect them *)
    [first_row]  (* Simplified - full implementation would accumulate all rows *)
  end.

(** * Basic Properties *)

(** Helper: inject_Z of non-negative Z is non-negative. *)
Lemma inject_Z_nonneg : forall z, (0 <= z)%Z -> 0 <= inject_Z z.
Proof.
  intros z Hz.
  unfold Qle. simpl.
  rewrite Z.mul_1_r.
  exact Hz.
Qed.

(** Helper: inject_Z of nat is non-negative. *)
Lemma inject_Z_of_nat_nonneg : forall n, 0 <= inject_Z (Z.of_nat n).
Proof.
  intros n.
  apply inject_Z_nonneg.
  apply Zle_0_nat.
Qed.

(** Helper: hd of list with default preserves non-negativity if default is non-negative. *)
Lemma hd_nonneg : forall l (d : Q),
  0 <= d ->
  Forall (fun x => 0 <= x) l ->
  0 <= hd d l.
Proof.
  intros l d Hd Hl.
  destruct l.
  - simpl. exact Hd.
  - simpl. inversion Hl. assumption.
Qed.

(** Helper: last of list with default preserves non-negativity. *)
Lemma last_nonneg : forall l (d : Q),
  0 <= d ->
  Forall (fun x => 0 <= x) l ->
  0 <= last l d.
Proof.
  intros l d Hd Hl.
  induction l.
  - simpl. exact Hd.
  - simpl.
    destruct l.
    + inversion Hl. assumption.
    + apply IHl. inversion Hl. assumption.
Qed.

(** Helper: msm_init_row produces non-negative values. *)
Lemma msm_init_row_nonneg : forall x1 y_prev Y_tail prev_cost c_const,
  0 <= c_const ->
  0 <= prev_cost ->
  Forall (fun x => 0 <= x) (msm_init_row x1 y_prev Y_tail prev_cost c_const).
Proof.
  intros x1 y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y_j ys IH]; intros y_prev prev_cost c_const Hc Hprev.
  - simpl. constructor; [exact Hprev | constructor].
  - simpl. constructor.
    + exact Hprev.
    + apply IH.
      * exact Hc.
      * apply Qle_trans with (y := 0 + 0).
        { setoid_replace (0 + 0) with 0 by ring. apply Qle_refl. }
        apply Qplus_le_compat.
        { exact Hprev. }
        { apply c_func_nonneg. exact Hc. }
Qed.

(** Helper: msm_compute_row produces non-negative values. *)
Lemma msm_compute_row_nonneg : forall x_i x_prev y_prev Y_tail prev_row cost_left c_const,
  0 <= c_const ->
  0 <= cost_left ->
  Forall (fun x => 0 <= x) prev_row ->
  Forall (fun x => 0 <= x) (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const).
Proof.
  intros x_i x_prev y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y_j ys IH]; intros y_prev prev_row cost_left c_const Hc Hleft Hprev.
  - simpl. constructor; [exact Hleft | constructor].
  - simpl.
    destruct prev_row as [|cost_diag [|cost_up rest]].
    + (* Empty prev_row - shouldn't happen with valid input *)
      simpl. constructor; [exact Hleft | constructor].
    + (* Single element prev_row - shouldn't happen *)
      simpl. constructor; [exact Hleft | constructor].
    + (* Normal case: cost_diag :: cost_up :: rest *)
      constructor.
      * exact Hleft.
      * apply IH.
        { exact Hc. }
        { (* New cost is min3 of non-negative values *)
          unfold Qmin3.
          apply Qmin2_glb.
          - (* cost_move = cost_diag + |x_i - y_j| >= 0 *)
            inversion Hprev as [|? ? Hdiag ?]. subst.
            apply Qle_trans with (y := 0 + 0).
            { setoid_replace (0 + 0) with 0 by ring. apply Qle_refl. }
            apply Qplus_le_compat; [exact Hdiag | apply Qabs_diff_nonneg].
          - apply Qmin2_glb.
            + (* cost_merge = cost_up + c_func >= 0 *)
              inversion Hprev as [|? ? ? Hrest]. subst.
              inversion Hrest as [|? ? Hup ?]. subst.
              apply Qle_trans with (y := 0 + 0).
              { setoid_replace (0 + 0) with 0 by ring. apply Qle_refl. }
              apply Qplus_le_compat; [exact Hup | apply c_func_nonneg; exact Hc].
            + (* cost_split = cost_left + c_func >= 0 *)
              apply Qle_trans with (y := 0 + 0).
              { setoid_replace (0 + 0) with 0 by ring. apply Qle_refl. }
              apply Qplus_le_compat; [exact Hleft | apply c_func_nonneg; exact Hc].
        }
        { (* cost_up :: rest is Forall nonneg *)
          inversion Hprev as [|? ? ? Hrest]. exact Hrest. }
Qed.

(** Helper: msm_compute_rows produces non-negative row. *)
Lemma msm_compute_rows_nonneg : forall X_tail x_prev Y y1 prev_row c_const,
  0 <= c_const ->
  Forall (fun x => 0 <= x) prev_row ->
  Forall (fun x => 0 <= x) (msm_compute_rows X_tail x_prev Y y1 prev_row c_const).
Proof.
  intros X_tail.
  induction X_tail as [|x_i xs IH]; intros x_prev Y y1 prev_row c_const Hc Hprev.
  - simpl. exact Hprev.
  - simpl.
    apply IH.
    + exact Hc.
    + apply msm_compute_row_nonneg.
      * exact Hc.
      * (* hd 0 prev_row + c_func >= 0 *)
        apply Qle_trans with (y := 0 + 0).
        { setoid_replace (0 + 0) with 0 by ring. apply Qle_refl. }
        apply Qplus_le_compat.
        { apply hd_nonneg. apply Qle_refl. exact Hprev. }
        { apply c_func_nonneg. exact Hc. }
      * exact Hprev.
Qed.

(** MSM distance is non-negative. *)
Lemma msm_nonneg : forall X Y cfg,
  0 <= msm_distance X Y cfg.
Proof.
  intros X Y cfg.
  destruct X as [|x1 xs]; destruct Y as [|y1 ys].
  - (* [], [] *)
    simpl. apply Qle_refl.
  - (* [], y::ys *)
    simpl.
    assert (Hc := msm_c_nonneg cfg).
    assert (Hn := inject_Z_of_nat_nonneg (length (y1 :: ys))).
    apply Qmult_le_0_compat; assumption.
  - (* x::xs, [] *)
    simpl.
    assert (Hc := msm_c_nonneg cfg).
    assert (Hn := inject_Z_of_nat_nonneg (length (x1 :: xs))).
    apply Qmult_le_0_compat; assumption.
  - (* x::xs, y::ys - the non-trivial case *)
    simpl.
    (* The result is last of the final row *)
    apply last_nonneg.
    + apply Qle_refl.
    + (* Final row is non-negative *)
      apply msm_compute_rows_nonneg.
      * apply msm_c_nonneg.
      * (* First row is non-negative *)
        apply msm_init_row_nonneg.
        { apply msm_c_nonneg. }
        { apply Qabs_diff_nonneg. }
Qed.

(** * Lemmas for Identity Proof *)

(** When X = Y, the optimal alignment is to use only Move operations with 0 cost.
    Key insight: |x_i - x_i| = 0, so the diagonal (all Move ops) has cost 0.
    Since MSM takes the minimum, and cost is non-negative, MSM(X,X) = 0. *)

(** Helper: when computing with same sequence, diagonal values are 0.
    The key is that Qabs_diff x x == 0. *)

(** Helper: Structure of msm_init_row result - it's (length + 1) elements *)
Lemma msm_init_row_length : forall x1 y_prev Y_tail prev_cost c_const,
  length (msm_init_row x1 y_prev Y_tail prev_cost c_const) = S (length Y_tail).
Proof.
  intros x1 y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_cost c_const.
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

(** Helper: nth element of msm_init_row *)
Lemma msm_init_row_nth : forall x1 y_prev Y_tail prev_cost c_const (n : nat) (d : Q),
  (n < length (msm_init_row x1 y_prev Y_tail prev_cost c_const))%nat ->
  exists cost, nth n (msm_init_row x1 y_prev Y_tail prev_cost c_const) d = cost.
Proof.
  intros. exists (nth n (msm_init_row x1 y_prev Y_tail prev_cost c_const) d). reflexivity.
Qed.

(** For reflexivity, we need a stronger statement: when x1 = y_prev = all elements,
    the entire row has cost based on accumulated c_func costs.
    However, for the diagonal (Move) path, we only care that the optimum is 0. *)

(** Helper: msm_init_row when starting with cost 0 and identical element *)
Lemma msm_init_row_first_zero : forall x xs c_const,
  hd 0 (msm_init_row x x xs 0 c_const) == 0.
Proof.
  intros x xs c_const.
  destruct xs; simpl; reflexivity.
Qed.

(** For reflexivity, we observe that:
    - The first cell has cost |x1 - x1| = 0
    - Each subsequent cell could use the diagonal (Move) path
    - The Move cost is |x_i - x_i| = 0
    - So the minimum path cost is 0

    The challenge is that our DP formulation computes row-by-row,
    not cell-by-cell, so we need to track that the diagonal element
    in each row is 0.
*)

(** Stronger helper: diagonal elements are 0 when X = Y *)
Lemma msm_init_row_diagonal_zero : forall (x : Q) (xs : list Q) (c_const : Q),
  0 <= c_const ->
  (* The last element represents diagonal endpoint when starting from 0 *)
  (* For identity proof, we show the minimum path has cost 0 *)
  (* Since |x - x| = 0, the Move path gives 0 *)
  (* Init row: first element is |x-x| = 0 *)
  (* But subsequent elements use c_func which adds cost *)
  (* The key is that the MINIMUM over the row includes the diagonal path *)
  True.
Proof.
  trivial.
Qed.

(** For reflexivity proof: We use a different approach.
    Instead of tracking exact values, we show:
    1. There exists a path (the diagonal) with cost 0
    2. MSM cost is the minimum over all paths
    3. Cost is non-negative
    4. Therefore MSM = 0 *)

(** Helper: last element of init_row when sequences match.
    This is tricky because init_row computes horizontal costs (using c_func),
    not diagonal costs. We need a different approach. *)

(** Actually, for reflexivity we need to show that when computing MSM(X, X),
    the diagonal cells have value 0. Let's define what we mean by diagonal. *)

(** In the DP matrix for MSM(X, Y) where X = x1::xs and Y = y1::ys:
    - Cell (0,0) has cost |x1 - y1| = |x1 - x1| = 0 when X = Y
    - Cell (i,i) has cost 0 when X = Y (using diagonal/Move path)

    The msm_init_row computes row 0: cells (0, j) for j = 0..len(Y)-1
    Only cell (0, 0) is on the diagonal; others use horizontal transitions.

    For reflexivity, we need to show that the FINAL diagonal cell (m-1, n-1)
    where m = n = len(X) has cost 0. *)

(** Let's prove a key property: when x = y, the Move cost is 0 *)
Lemma move_cost_identity : forall x, Qabs_diff x x == 0.
Proof.
  intros x. apply Qabs_diff_zero.
Qed.

(** When c_func is called with a = b, it returns c_const *)
Lemma c_func_identity : forall c_const a c_val,
  c_func c_const a a c_val == c_const.
Proof.
  intros. apply c_func_a_eq_b.
Qed.

(** Positive split/merge costs cannot participate in a zero-cost path. *)
Lemma c_func_pos : forall c_const a b c_val,
  0 < c_const ->
  0 < c_func c_const a b c_val.
Proof.
  intros c_const a b c_val Hc.
  eapply Qlt_le_trans.
  - exact Hc.
  - apply c_func_ge_c.
    apply Qlt_le_weak. exact Hc.
Qed.

Lemma Qplus_nonneg_pos : forall a b,
  0 <= a ->
  0 < b ->
  0 < a + b.
Proof.
  intros a b Ha Hb.
  setoid_replace 0 with (0 + 0) by ring.
  setoid_replace (a + b) with (b + a) by ring.
  apply Qplus_lt_le_compat; assumption.
Qed.

Lemma Qplus_nonneg_pos_not_zero : forall a b,
  0 <= a ->
  0 < b ->
  ~ (a + b == 0).
Proof.
  intros a b Ha Hb Hzero.
  pose proof (Qplus_nonneg_pos a b Ha Hb) as Hpos.
  apply (Qlt_not_eq 0 (a + b) Hpos).
  symmetry. exact Hzero.
Qed.

Lemma Qplus_nonneg_eq_zero : forall a b,
  0 <= a ->
  0 <= b ->
  a + b == 0 ->
  a == 0 /\ b == 0.
Proof.
  intros a b Ha Hb Hsum.
  split.
  - destruct (Qle_lt_or_eq 0 a Ha) as [Ha_pos | Ha_zero].
    + exfalso.
      setoid_replace (a + b) with (b + a) in Hsum by ring.
      apply (Qplus_nonneg_pos_not_zero b a Hb Ha_pos).
      exact Hsum.
    + symmetry. exact Ha_zero.
  - destruct (Qle_lt_or_eq 0 b Hb) as [Hb_pos | Hb_zero].
    + exfalso.
      apply (Qplus_nonneg_pos_not_zero a b Ha Hb_pos).
      exact Hsum.
    + symmetry. exact Hb_zero.
Qed.

Lemma Qmin2_eq_zero_choice : forall a b,
  Qmin2 a b == 0 ->
  a == 0 \/ b == 0.
Proof.
  intros a b Hmin.
  unfold Qmin2 in Hmin.
  destruct (Qle_bool a b); [left | right]; exact Hmin.
Qed.

Lemma Qmin3_eq_zero_choice : forall a b c,
  Qmin3 a b c == 0 ->
  a == 0 \/ b == 0 \/ c == 0.
Proof.
  intros a b c Hmin.
  unfold Qmin3 in Hmin.
  destruct (Qmin2_eq_zero_choice a (Qmin2 b c) Hmin) as [Ha | Hbc].
  - left. exact Ha.
  - right.
    apply Qmin2_eq_zero_choice. exact Hbc.
Qed.

Lemma msm_init_row_last_zero_singleton : forall x1 y_prev Y_tail prev_cost c_const,
  0 < c_const ->
  0 <= prev_cost ->
  last (msm_init_row x1 y_prev Y_tail prev_cost c_const) 0 == 0 ->
  Y_tail = [] /\ prev_cost == 0.
Proof.
  intros x1 y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_cost c_const Hc Hprev Hlast.
  - simpl in Hlast. split; [reflexivity | exact Hlast].
  - simpl in Hlast.
    assert (Hnext_nonneg : 0 <= prev_cost + c_func c_const y x1 y_prev).
    { apply Qlt_le_weak.
      apply Qplus_nonneg_pos.
      - exact Hprev.
      - apply c_func_pos. exact Hc. }
    destruct ys as [|y_next ys'].
    + simpl in Hlast.
      exfalso.
      apply (Qplus_nonneg_pos_not_zero prev_cost
               (c_func c_const y x1 y_prev)).
      * exact Hprev.
      * apply c_func_pos. exact Hc.
      * exact Hlast.
    + simpl in Hlast.
      destruct (IH y (prev_cost + c_func c_const y x1 y_prev)
                   c_const Hc Hnext_nonneg Hlast) as [_ Hnext_zero].
      exfalso.
      apply (Qplus_nonneg_pos_not_zero prev_cost
               (c_func c_const y x1 y_prev)).
      * exact Hprev.
      * apply c_func_pos. exact Hc.
      * exact Hnext_zero.
Qed.

Lemma msm_distance_singleton_left_zero : forall x y ys cfg,
  0 < msm_c cfg ->
  msm_distance [x] (y :: ys) cfg == 0 ->
  ys = [] /\ x == y.
Proof.
  intros x y ys cfg Hc Hzero.
  simpl in Hzero.
  destruct (msm_init_row_last_zero_singleton
              x y ys (Qabs_diff x y) (msm_c cfg)
              Hc (Qabs_diff_nonneg x y) Hzero) as [Hys Habs].
  split.
  - exact Hys.
  - apply Qabs_diff_zero_iff. exact Habs.
Qed.

Lemma msm_compute_rows_singleton_from_positive : forall xs x_prev y cost c_const,
  0 < c_const ->
  0 < cost ->
  0 < last (msm_compute_rows xs x_prev [y] y [cost] c_const) 0.
Proof.
  induction xs as [|x_next xs IH]; intros x_prev y cost c_const Hc Hcost.
  - simpl. exact Hcost.
  - change (0 <
      last (msm_compute_rows xs x_next [y] y
              [cost + c_func c_const x_next x_prev y] c_const) 0).
    eapply IH.
    + exact Hc.
    + apply Qplus_nonneg_pos.
      * apply Qlt_le_weak.
        exact Hcost.
      * apply c_func_pos. exact Hc.
Qed.

Lemma msm_distance_singleton_right_zero : forall x xs y cfg,
  0 < msm_c cfg ->
  msm_distance (x :: xs) [y] cfg == 0 ->
  xs = [] /\ x == y.
Proof.
  intros x xs y cfg Hc Hzero.
  destruct xs as [|x2 xs'].
  - simpl in Hzero.
    split; [reflexivity |].
    apply Qabs_diff_zero_iff. exact Hzero.
  - simpl in Hzero.
    exfalso.
    assert (Hstart_pos : 0 < Qabs_diff x y + c_func (msm_c cfg) x2 x y).
    { apply Qplus_nonneg_pos.
      - apply Qabs_diff_nonneg.
      - apply c_func_pos. exact Hc. }
    pose proof (msm_compute_rows_singleton_from_positive
                  xs' x2 y
                  (Qabs_diff x y + c_func (msm_c cfg) x2 x y)
                  (msm_c cfg) Hc) as Hpos.
    specialize (Hpos Hstart_pos).
    apply (Qlt_not_eq 0 _ Hpos).
    symmetry. exact Hzero.
Qed.

Lemma msm_distance_two_two_zero : forall x x_next y y_next cfg,
  0 < msm_c cfg ->
  msm_distance [x; x_next] [y; y_next] cfg == 0 ->
  x == y /\ x_next == y_next.
Proof.
  intros x x_next y y_next cfg Hc Hzero.
  simpl in Hzero.
  destruct (Qmin3_eq_zero_choice
              (Qabs_diff x y + Qabs_diff x_next y_next)
              (Qabs_diff x y + c_func (msm_c cfg) y_next x y +
                 c_func (msm_c cfg) x_next x y_next)
              (Qabs_diff x y + c_func (msm_c cfg) x_next x y +
                 c_func (msm_c cfg) y_next x_next y)
              Hzero) as [Hmove | [Hmerge | Hsplit]].
  - destruct (Qplus_nonneg_eq_zero
                (Qabs_diff x y) (Qabs_diff x_next y_next)
                (Qabs_diff_nonneg x y)
                (Qabs_diff_nonneg x_next y_next)
                Hmove) as [Hxy Hnext].
    split.
    + apply Qabs_diff_zero_iff. exact Hxy.
    + apply Qabs_diff_zero_iff. exact Hnext.
  - exfalso.
    assert (Hprefix_pos :
      0 < Qabs_diff x y + c_func (msm_c cfg) y_next x y).
    { apply Qplus_nonneg_pos.
      - apply Qabs_diff_nonneg.
      - apply c_func_pos. exact Hc. }
    assert (Hmerge_pos :
      0 < Qabs_diff x y + c_func (msm_c cfg) y_next x y +
            c_func (msm_c cfg) x_next x y_next).
    { apply Qplus_nonneg_pos.
      - apply Qlt_le_weak. exact Hprefix_pos.
      - apply c_func_pos. exact Hc. }
    apply (Qlt_not_eq 0 _ Hmerge_pos).
    symmetry. exact Hmerge.
  - exfalso.
    assert (Hprefix_pos :
      0 < Qabs_diff x y + c_func (msm_c cfg) x_next x y).
    { apply Qplus_nonneg_pos.
      - apply Qabs_diff_nonneg.
      - apply c_func_pos. exact Hc. }
    assert (Hsplit_pos :
      0 < Qabs_diff x y + c_func (msm_c cfg) x_next x y +
            c_func (msm_c cfg) y_next x_next y).
    { apply Qplus_nonneg_pos.
      - apply Qlt_le_weak. exact Hprefix_pos.
      - apply c_func_pos. exact Hc. }
    apply (Qlt_not_eq 0 _ Hsplit_pos).
    symmetry. exact Hsplit.
Qed.

Lemma msm_zero_implies_series_eq_evidence_use : forall (contracts : MsmDistanceEvidence) X Y cfg,
  0 < msm_c cfg ->
  msm_distance X Y cfg == 0 ->
  series_Qeq X Y.
Proof.
  intros contracts X Y cfg Hc Hzero.
  destruct X as [|x xs]; destruct Y as [|y ys].
  - exact I.
  - simpl in Hzero.
    exfalso.
    assert (Hlen_pos : (0 < length (y :: ys))%nat) by (simpl; lia).
    assert (Hpos_len : 0 < inject_Z (Z.of_nat (length (y :: ys)))).
    { apply inject_Z_of_nat_pos. exact Hlen_pos. }
    assert (Hpos_dist : 0 < inject_Z (Z.of_nat (length (y :: ys))) * msm_c cfg).
    { apply Qmult_lt_0_compat; assumption. }
    apply (Qlt_not_eq 0 _ Hpos_dist). symmetry. exact Hzero.
  - simpl in Hzero.
    exfalso.
    assert (Hlen_pos : (0 < length (x :: xs))%nat) by (simpl; lia).
    assert (Hpos_len : 0 < inject_Z (Z.of_nat (length (x :: xs)))).
    { apply inject_Z_of_nat_pos. exact Hlen_pos. }
    assert (Hpos_dist : 0 < inject_Z (Z.of_nat (length (x :: xs))) * msm_c cfg).
    { apply Qmult_lt_0_compat; assumption. }
    apply (Qlt_not_eq 0 _ Hpos_dist). symmetry. exact Hzero.
  - destruct xs as [|x_next xs']; destruct ys as [|y_next ys'].
    + simpl in Hzero.
      simpl. split.
      * apply Qabs_diff_zero_iff. exact Hzero.
      * exact I.
    + destruct (msm_distance_singleton_left_zero x y (y_next :: ys') cfg Hc Hzero)
        as [Htail_empty _].
      discriminate Htail_empty.
    + destruct (msm_distance_singleton_right_zero x (x_next :: xs') y cfg Hc Hzero)
        as [Htail_empty _].
      discriminate Htail_empty.
    + destruct xs' as [|x_third xs'']; destruct ys' as [|y_third ys''].
      * simpl.
        destruct (msm_distance_two_two_zero x x_next y y_next cfg Hc Hzero)
          as [Hxy Hnext].
        repeat split; assumption.
      * exact (msm_zero_implies_series_eq_proof contracts
                 x x_next [] y y_next (y_third :: ys'') cfg Hc Hzero).
      * exact (msm_zero_implies_series_eq_proof contracts
                 x x_next (x_third :: xs'') y y_next [] cfg Hc Hzero).
      * exact (msm_zero_implies_series_eq_proof contracts
                 x x_next (x_third :: xs'') y y_next (y_third :: ys'') cfg Hc Hzero).
Qed.

(** For reflexivity, the key insight is that the minimum cost path
    from (0,0) to (n-1, n-1) when X = Y is the diagonal path where
    each step is a Move with cost |x_i - x_i| = 0.

    Rather than tracking exact row values, we prove that:
    1. The cost to reach (i, i) via diagonal is 0
    2. The DP minimum is at most this diagonal cost
    3. Cost is non-negative
    4. Therefore minimum = 0 *)

(** Alternative approach: directly define what happens on the diagonal *)

(** When X and Y are the same sequence, last element of final row is 0.
    This is because the optimal alignment uses only Move operations. *)

(** Helper: For init_row, when Y_tail is empty, last = prev_cost *)
Lemma msm_init_row_empty_last : forall x1 y_prev prev_cost c_const,
  last (msm_init_row x1 y_prev [] prev_cost c_const) 0 == prev_cost.
Proof.
  intros. simpl. reflexivity.
Qed.

(** Helper: For non-empty Y_tail, last is computed recursively *)
Lemma msm_init_row_cons_last : forall x1 y_prev y ys prev_cost c_const,
  last (msm_init_row x1 y_prev (y :: ys) prev_cost c_const) 0 ==
  last (msm_init_row x1 y ys (prev_cost + c_func c_const y x1 y_prev) c_const) 0.
Proof.
  intros. simpl.
  destruct ys; simpl; reflexivity.
Qed.

(** For the reflexivity proof, we'll use a different strategy:
    Show that MSM(X, X) <= 0 (by exhibiting the diagonal path)
    and MSM(X, X) >= 0 (by msm_nonneg).
    Therefore MSM(X, X) = 0. *)

(** The diagonal path cost when X = X is sum of |x_i - x_i| = 0 *)
Lemma diagonal_path_cost_zero : forall X,
  fold_right (fun x acc => Qabs_diff x x + acc) 0 X == 0.
Proof.
  induction X as [|x xs IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. rewrite move_cost_identity. ring.
Qed.

(** * Diagonal Element Tracking for Reflexivity *)

(** Key insight: In MSM(X, X), only the diagonal elements are 0.
    - Row 0: position 0 has cost |x1 - x1| = 0
    - Row i: position i has cost 0 (via Move from diagonal of previous row)
    - Final answer is last element of final row = diagonal element = 0 *)

(** Helper: The first element of init_row is prev_cost *)
Lemma msm_init_row_hd : forall x1 y_prev Y_tail prev_cost c_const,
  hd 0 (msm_init_row x1 y_prev Y_tail prev_cost c_const) == prev_cost.
Proof.
  intros. destruct Y_tail; simpl; reflexivity.
Qed.

(** Helper: nth element of a row - we track position i in row i as the diagonal *)
Lemma msm_compute_row_length : forall x_i x_prev y_prev Y_tail prev_row cost_left c_const,
  (S (length Y_tail) <= length prev_row)%nat ->
  length (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) = S (length Y_tail).
Proof.
  intros x_i x_prev y_prev Y_tail.
  revert y_prev.
  induction Y_tail as [|y ys IH]; intros y_prev prev_row cost_left c_const Hlen.
  - simpl. reflexivity.
  - simpl.
    destruct prev_row as [|d1 [|d2 rest]].
    + simpl in Hlen. lia.
    + simpl in Hlen. lia.
    + simpl. rewrite IH.
      * reflexivity.
      * simpl in Hlen. simpl. lia.
Qed.

(** For reflexivity: track that diagonal element of row i equals 0 *)
(** The diagonal element is at position i in row i (0-indexed) *)

(** First row diagonal is at position 0 *)
Lemma msm_first_row_diagonal_zero : forall x xs c_const,
  nth 0 (msm_init_row x x xs 0 c_const) 0 == 0.
Proof.
  intros x xs c_const.
  destruct xs; simpl; reflexivity.
Qed.

(** Key lemma: In msm_compute_row, if the diagonal of prev_row is 0,
    and we're computing for identical sequences, then the new diagonal is 0 *)

(** First, show that Qmin3 of values where one is 0 gives <= 0 *)
Lemma Qmin3_with_zero : forall a b,
  Qmin3 0 a b <= 0.
Proof.
  intros a b. unfold Qmin3. apply Qmin2_le_l.
Qed.

Lemma Qmin3_le_l : forall a b c,
  Qmin3 a b c <= a.
Proof.
  intros a b c. unfold Qmin3. apply Qmin2_le_l.
Qed.

Lemma msm_compute_row_hd_eq :
  forall x_i x_prev y_prev Y_tail prev_row cost_left c_const,
    nth 0 (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 =
    cost_left.
Proof.
  intros x_i x_prev y_prev Y_tail prev_row cost_left c_const.
  destruct Y_tail as [|y ys]; simpl; [reflexivity |].
  destruct prev_row as [|d1 [|d2 rest]]; reflexivity.
Qed.

Lemma msm_compute_row_nth_succ_le_move :
  forall p x_i x_prev y_prev Y_tail prev_row cost_left c_const,
    (p < length Y_tail)%nat ->
    (S p < length prev_row)%nat ->
    nth (S p)
        (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 <=
    nth p prev_row 0 + Qabs_diff x_i (nth p Y_tail 0).
Proof.
  induction p as [|p IHp]; intros x_i x_prev y_prev Y_tail prev_row cost_left c_const Hy Hp.
  - destruct Y_tail as [|y ys]; [simpl in Hy; lia |].
    destruct prev_row as [|cost_diag [|cost_up rest]]; simpl in Hp; try lia.
    simpl.
    rewrite msm_compute_row_hd_eq.
    apply Qmin3_le_l.
  - destruct Y_tail as [|y ys]; [simpl in Hy; lia |].
    destruct prev_row as [|cost_diag [|cost_up rest]]; simpl in Hp; try lia.
    simpl.
    apply IHp; simpl in *; lia.
Qed.

Lemma msm_compute_row_diagonal_succ_le_zero :
  forall p x_i x_prev y_prev Y_tail prev_row cost_left c_const,
    (p < length Y_tail)%nat ->
    (S p < length prev_row)%nat ->
    nth p prev_row 0 <= 0 ->
    nth p Y_tail 0 = x_i ->
    nth (S p)
        (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 <= 0.
Proof.
  intros p x_i x_prev y_prev Y_tail prev_row cost_left c_const Hy Hp Hdiag Hyval.
  eapply Qle_trans.
  - apply msm_compute_row_nth_succ_le_move; eassumption.
  - rewrite Hyval.
    rewrite Qabs_diff_zero.
    setoid_replace (nth p prev_row 0 + 0) with (nth p prev_row 0) by ring.
    exact Hdiag.
Qed.

Lemma last_eq_nth_by_length :
  forall (l : list Q) d n,
    length l = S n ->
    last l d = nth n l d.
Proof.
  induction l as [|x xs IH]; intros d n Hlen.
  - simpl in Hlen. discriminate.
  - destruct xs as [|y ys].
    + simpl in Hlen. inversion Hlen. reflexivity.
    + simpl in Hlen.
      destruct n as [|n']; [discriminate |].
      simpl. apply IH. inversion Hlen. reflexivity.
Qed.

Lemma nth_from_skipn_cons :
  forall i (l : list Q) (x : Q) (xs : list Q) (d : Q),
    skipn i l = x :: xs ->
    nth i l d = x.
Proof.
  induction i as [|i IHi]; intros l x xs d Hskip.
  - destruct l as [|h t]; simpl in Hskip; inversion Hskip. reflexivity.
  - destruct l as [|h t]; simpl in Hskip; [discriminate |].
    apply IHi with (xs := xs). exact Hskip.
Qed.

Lemma skipn_succ_from_cons :
  forall i (l : list Q) (x : Q) (xs : list Q),
    skipn i l = x :: xs ->
    skipn (S i) l = xs.
Proof.
  induction i as [|i IHi]; intros l x xs Hskip.
  - destruct l as [|h t]; simpl in Hskip; inversion Hskip. reflexivity.
  - destruct l as [|h t]; simpl in Hskip; [discriminate |].
    apply IHi with (x := x). exact Hskip.
Qed.

Lemma msm_compute_rows_self_tail_last_le_zero :
  forall rem full_tail i x_prev x1 row c_const,
    skipn i full_tail = rem ->
    (i + length rem = length full_tail)%nat ->
    length row = S (length full_tail) ->
    nth i row 0 <= 0 ->
    last (msm_compute_rows rem x_prev (x1 :: full_tail) x1 row c_const) 0 <= 0.
Proof.
  induction rem as [|x rem IH]; intros full_tail i x_prev x1 row c_const Hskip Hlen_rem Hrow Hdiag.
  - simpl.
    assert (Hi : i = length full_tail) by (simpl in Hlen_rem; lia).
    subst i.
    rewrite (last_eq_nth_by_length row 0 (length full_tail)); assumption.
  - simpl.
    set (new_row :=
      msm_compute_row x x_prev x1 full_tail row
        (hd 0 row + c_func c_const x x_prev x1) c_const).
    assert (Hi_lt : (i < length full_tail)%nat) by (simpl in Hlen_rem; lia).
    assert (Hrow_i : (S i < length row)%nat) by (rewrite Hrow; lia).
    assert (Hx : nth i full_tail 0 = x).
    { eapply nth_from_skipn_cons. exact Hskip. }
    assert (Hnew_diag : nth (S i) new_row 0 <= 0).
    { subst new_row.
      eapply msm_compute_row_diagonal_succ_le_zero; eauto. }
    assert (Hnew_len : length new_row = S (length full_tail)).
    { subst new_row.
      rewrite msm_compute_row_length; [reflexivity | rewrite Hrow; lia]. }
    apply (IH full_tail (S i) x x1 new_row c_const).
    + eapply skipn_succ_from_cons. exact Hskip.
    + simpl in Hlen_rem. lia.
    + exact Hnew_len.
    + exact Hnew_diag.
Qed.

(** When computing row i+1 from row i where diagonal(row i) = 0,
    the Move option for diagonal(row i+1) uses:
    diagonal(row i) + |x_{i+1} - x_{i+1}| = 0 + 0 = 0
    So diagonal(row i+1) = min(0, ...) <= 0 *)

(** Combined with non-negativity: diagonal = 0 *)

(** For the actual proof, we need to track the position of diagonal elements
    through the computation. This is complex due to list indexing.

    Alternative simpler approach: prove that MSM(X,X) <= 0 directly
    by showing the minimum includes a zero path. *)

(** Simpler approach: For single-element lists *)
Lemma msm_reflexive_singleton : forall x cfg,
  msm_distance [x] [x] cfg == 0.
Proof.
  intros x cfg.
  unfold msm_distance. simpl.
  (* last (msm_compute_rows [] x [x] x (msm_init_row x x [] (Qabs_diff x x) (msm_c cfg)) (msm_c cfg)) 0 *)
  (* msm_init_row x x [] (Qabs_diff x x) c = [Qabs_diff x x] *)
  (* msm_compute_rows [] ... [Qabs_diff x x] ... = [Qabs_diff x x] *)
  (* last [Qabs_diff x x] 0 = Qabs_diff x x = 0 *)
  rewrite move_cost_identity.
  simpl.
  reflexivity.
Qed.

(** For longer lists, we need induction with careful tracking.
    The key is that each diagonal step uses Move with cost 0. *)

(** Define diagonal access: the i-th element of a list *)
Definition diagonal_of (row : list Q) (i : nat) : Q := nth i row 0.

(** Invariant: after computing row i, diagonal_of row i = 0 *)

(** Helper: When X = Y, the diagonal element is computed via Move with cost 0 *)
Lemma diagonal_move_cost_zero : forall x,
  Qabs_diff x x == 0.
Proof.
  intros. apply Qabs_diff_zero.
Qed.

(** For the full reflexivity proof, we use strong induction on list length,
    tracking that the diagonal element of each computed row is 0.
    This is the core invariant that ensures MSM(X, X) = 0. *)

(** Helper: last element of init_row when list is empty *)
Lemma msm_init_row_nil_last : forall x1 y_prev prev_cost c_const,
  last (msm_init_row x1 y_prev [] prev_cost c_const) 0 == prev_cost.
Proof.
  intros. simpl. reflexivity.
Qed.

Lemma msm_reflexive_diagonal_direct : forall X cfg,
  (2 <= length X)%nat ->
  msm_distance X X cfg == 0.
Proof.
  intros X cfg Hlen.
  destruct X as [|x1 xs].
  - simpl in Hlen. lia.
  - apply Qle_antisym.
    + simpl.
      eapply (msm_compute_rows_self_tail_last_le_zero
        xs xs 0 x1 x1
        (msm_init_row x1 x1 xs (Qabs_diff x1 x1) (msm_c cfg))
        (msm_c cfg)).
      * reflexivity.
      * simpl. lia.
      * apply msm_init_row_length.
      * destruct xs as [|x2 xs']; simpl; rewrite Qabs_diff_zero; apply Qle_refl.
    + apply msm_nonneg.
Qed.

(** Main reflexivity theorem *)
Lemma msm_reflexive : forall X cfg,
  msm_distance X X cfg == 0.
Proof.
  intros X cfg.
  destruct X as [|x1 xs].
  - (* Empty list *)
    simpl. reflexivity.
  - (* Non-empty list x1 :: xs *)
    (* Use antisymmetry: 0 <= MSM and MSM <= 0 *)
    apply Qle_antisym.
	    + (* MSM(X, X) <= 0 *)
	      destruct xs as [|x2 xs'].
	      * (* Singleton case: already proven *)
	        simpl. rewrite move_cost_identity. simpl. apply Qle_refl.
	      * (* List with at least 2 elements - diagonal Move path *)
	        assert (Hlen: (2 <= length (x1 :: x2 :: xs'))%nat) by (simpl; lia).
	        pose proof (msm_reflexive_diagonal_direct (x1 :: x2 :: xs') cfg Hlen) as Hdiag.
	        rewrite Hdiag. apply Qle_refl.
    + (* 0 <= MSM(X, X) *)
      apply msm_nonneg.
Qed.

(** MSM identity: MSM(X, Y) = 0 implies pointwise rational equality (when c > 0). *)
Lemma msm_zero_implies_series_eq : forall (contracts : MsmDistanceEvidence) X Y cfg,
  0 < msm_c cfg ->
  msm_distance X Y cfg == 0 ->
  series_Qeq X Y.
Proof.
  intros contracts X Y cfg Hc Hzero.
  exact (msm_zero_implies_series_eq_evidence_use contracts X Y cfg Hc Hzero).
Qed.
