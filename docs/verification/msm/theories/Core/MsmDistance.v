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
  - (* x::xs, y::ys *)
    simpl.
    (* The final value is built from non-negative operations *)
    (* This requires induction on the structure *)
    admit.
Admitted.

(** * Lemmas for Identity Proof *)

(** When X = Y, the optimal alignment is to use only Move operations with 0 cost. *)
Lemma msm_first_row_same : forall (x : Q) (xs : list Q) (c_const cost : Q),
  cost == Qabs_diff x x ->
  forall y ys,
  y = x -> ys = xs ->
  (* Each cell in first row has cost 0 when sequences match *)
  True. (* Placeholder for actual lemma *)
Proof.
  intros. trivial.
Qed.

(** Helper: last element of init_row when sequences match *)
Lemma msm_init_row_same_last : forall (x : Q) (xs : list Q) (c_const : Q),
  0 <= c_const ->
  last (msm_init_row x x xs 0 c_const) 0 == 0.
Proof.
  intros x xs c_const Hc.
  induction xs as [|x' xs' IH].
  - simpl. reflexivity.
  - simpl.
    (* Need more careful analysis of list structure *)
    admit.
Admitted.

(** MSM is reflexive: MSM(X, X) = 0 *)
Lemma msm_reflexive : forall X cfg,
  msm_distance X X cfg == 0.
Proof.
  intros X cfg.
  destruct X as [|x1 xs].
  - simpl. reflexivity.
  - simpl.
    (* When X = X, the diagonal path (all Move operations) gives cost 0
       because |x_i - x_i| = 0 for all i. *)
    (* This requires showing that msm_init_row and msm_compute_rows
       produce 0 on the diagonal when given identical sequences. *)
    admit.
Admitted.

(** MSM identity: MSM(X, Y) = 0 implies X = Y (when c > 0) *)
Lemma msm_zero_implies_equal : forall X Y cfg,
  0 < msm_c cfg ->
  msm_distance X Y cfg == 0 ->
  X = Y.
Proof.
  intros X Y cfg Hc Hmsm.
  (* If X ≠ Y, then either:
     1. They have different lengths -> need split/merge with cost >= c > 0
     2. Same length but different values -> need move with cost |x_i - y_i| > 0 *)
  admit.
Admitted.
