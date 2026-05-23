(** * MSM Identity Property

    This module proves the identity property of the MSM metric:

    1. MSM(X, X) = 0  (reflexivity)
    2. MSM(X, Y) = 0 -> X = Y  (identity of indiscernibles, partially)

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance.

(** * Reflexivity: MSM(X, X) = 0 *)

(** When X = Y, the optimal alignment uses only Move operations.
    Since each Move(i, i) has cost |x_i - x_i| = 0, the total cost is 0. *)

(** Helper: Qabs_diff of same element is 0 *)
Lemma Qabs_diff_same : forall x, Qabs_diff x x == 0.
Proof.
  intros x. unfold Qabs_diff.
  setoid_replace (x - x) with 0 by ring.
  apply Qabs_case; intros _; reflexivity.
Qed.

(** MSM is reflexive - use the lemma from MsmDistance *)
Theorem msm_reflexive' : forall (contracts : MsmDistanceContracts) X cfg,
  msm_distance X X cfg == 0.
Proof.
  intros contracts X cfg.
  apply (msm_reflexive contracts).
Qed.

(** * Partial Converse: MSM(X, Y) = 0 -> X = Y (for non-empty series) *)

(** This direction requires showing that:
    - If Move cost is 0, then x_i = y_j
    - If Split/Merge is used, cost >= c > 0
    - Therefore, if total cost is 0, only Moves were used with matching values *)

Lemma msm_zero_implies_same_length' : forall (contracts : MsmDistanceContracts) X Y cfg,
  0 < msm_c cfg ->
  msm_distance X Y cfg == 0 ->
  length X = length Y.
Proof.
  intros contracts X Y cfg Hc Hzero.
  f_equal.
  apply (msm_zero_implies_equal contracts) with cfg; assumption.
Qed.

Lemma msm_zero_implies_equal' : forall (contracts : MsmDistanceContracts) X Y cfg,
  0 < msm_c cfg ->
  msm_distance X Y cfg == 0 ->
  X = Y.
Proof.
  intros contracts X Y cfg Hc Hzero.
  exact (msm_zero_implies_equal contracts X Y cfg Hc Hzero).
Qed.

(** * Main Identity Theorem *)

Theorem msm_identity : forall (contracts : MsmDistanceContracts) X Y cfg,
  0 < msm_c cfg ->
  (msm_distance X Y cfg == 0 <-> X = Y).
Proof.
  intros contracts X Y cfg Hc.
  split.
  - (* MSM(X, Y) = 0 -> X = Y *)
    intros Hzero.
    exact (msm_zero_implies_equal contracts X Y cfg Hc Hzero).
  - (* X = Y -> MSM(X, Y) = 0 *)
    intros Heq. subst Y.
    apply (msm_reflexive contracts).
Qed.

(** * Non-negativity *)

Theorem msm_nonneg' : forall X Y cfg,
  0 <= msm_distance X Y cfg.
Proof.
  intros X Y cfg.
  apply msm_nonneg.
Qed.
