(** * MSM Core Definitions

    This module contains the foundational type definitions for the
    Move-Split-Merge (MSM) metric for time series.

    We use rational numbers (Q) for exact arithmetic, avoiding the
    axioms required for real numbers (R).

    Part of: Liblevenshtein.MSM

    Reference: Stefan, Alexandra, et al. "The move-split-merge metric for time series."
               IEEE TKDE 25.6 (2012): 1425-1438.
*)

From Stdlib Require Import List Nat Arith.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.

(** * Type Definitions *)

(** A time series is a list of rational values. *)
Definition TimeSeries := list Q.

(** Pointwise rational equality for time series.

    Coq's [Q] has a setoid equality [(==)] distinct from Leibniz equality:
    unreduced rationals can denote the same value without being syntactically
    equal. MSM identity over rational-valued series must therefore conclude
    this relation rather than raw list equality. *)
Fixpoint series_Qeq (X Y : TimeSeries) : Prop :=
  match X, Y with
  | [], [] => True
  | x :: xs, y :: ys => x == y /\ series_Qeq xs ys
  | _, _ => False
  end.

Lemma series_Qeq_refl : forall X, series_Qeq X X.
Proof.
  induction X as [|x xs IH].
  - exact I.
  - simpl. split; [reflexivity | exact IH].
Qed.

Lemma series_Qeq_length : forall X Y,
  series_Qeq X Y -> length X = length Y.
Proof.
  induction X as [|x xs IH]; intros Y Hxy.
  - destruct Y; [reflexivity | contradiction].
  - destruct Y as [|y ys]; [contradiction |].
    simpl in Hxy. destruct Hxy as [_ Htail].
    simpl. f_equal. apply IH. exact Htail.
Qed.

(** MSM configuration: the split/merge cost constant c. *)
Record MsmConfig := mkMsmConfig {
  msm_c : Q;          (* Split/merge cost constant *)
  msm_c_nonneg : 0 <= msm_c  (* c must be non-negative *)
}.

(** Dynamic programming matrix for MSM computation. *)
Definition QMatrix := list (list Q).

(** * Helper Functions *)

(** Safe list access with default value. *)
Definition nth_q (l : list Q) (i : nat) (default : Q) : Q :=
  nth i l default.

(** Safe matrix access with default value. *)
Definition get_q (m : QMatrix) (i j : nat) (default : Q) : Q :=
  nth_q (nth i m []) j default.

(** Minimum of two rationals. *)
Definition Qmin2 (a b : Q) : Q :=
  if Qle_bool a b then a else b.

(** Minimum of three rationals. *)
Definition Qmin3 (a b c : Q) : Q :=
  Qmin2 a (Qmin2 b c).

(** * Basic Lemmas *)

(** Helper: Qle_bool false implies > *)
Lemma Qle_bool_false_lt : forall a b, Qle_bool a b = false -> b < a.
Proof.
  intros a b H.
  (* We want to show b < a given Qle_bool a b = false *)
  (* Qlt_le_dec b a gives: {b < a} + {a <= b} *)
  destruct (Qlt_le_dec b a) as [Hlt | Hle].
  - (* b < a - this is what we want *)
    exact Hlt.
  - (* a <= b - but H says Qle_bool a b = false, contradiction *)
    exfalso.
    apply Qle_bool_iff in Hle. rewrite Hle in H. discriminate.
Qed.

Lemma Qmin2_comm : forall a b, Qmin2 a b == Qmin2 b a.
Proof.
  intros a b. unfold Qmin2.
  destruct (Qle_bool a b) eqn:Hab;
  destruct (Qle_bool b a) eqn:Hba.
  - apply Qle_bool_iff in Hab.
    apply Qle_bool_iff in Hba.
    apply Qle_antisym; assumption.
  - reflexivity.
  - reflexivity.
  - apply Qle_bool_false_lt in Hab.
    apply Qle_bool_false_lt in Hba.
    exfalso. apply (Qlt_irrefl a). apply Qlt_trans with b; assumption.
Qed.

Lemma Qmin2_le_l : forall a b, Qmin2 a b <= a.
Proof.
  intros a b. unfold Qmin2.
  destruct (Qle_bool a b) eqn:H.
  - apply Qle_refl.
  - apply Qle_bool_false_lt in H. apply Qlt_le_weak. assumption.
Qed.

Lemma Qmin2_le_r : forall a b, Qmin2 a b <= b.
Proof.
  intros a b. unfold Qmin2.
  destruct (Qle_bool a b) eqn:H.
  - apply Qle_bool_iff in H. assumption.
  - apply Qle_refl.
Qed.

Lemma Qmin2_glb : forall a b c, c <= a -> c <= b -> c <= Qmin2 a b.
Proof.
  intros a b c Ha Hb. unfold Qmin2.
  destruct (Qle_bool a b) eqn:H; assumption.
Qed.

Lemma Qmin3_le_a : forall a b c, Qmin3 a b c <= a.
Proof.
  intros. unfold Qmin3.
  apply Qle_trans with (y := a).
  - apply Qmin2_le_l.
  - apply Qle_refl.
Qed.

Lemma Qmin3_le_b : forall a b c, Qmin3 a b c <= b.
Proof.
  intros. unfold Qmin3.
  apply Qle_trans with (y := Qmin2 b c).
  - apply Qmin2_le_r.
  - apply Qmin2_le_l.
Qed.

Lemma Qmin3_le_c : forall a b c, Qmin3 a b c <= c.
Proof.
  intros. unfold Qmin3.
  apply Qle_trans with (y := Qmin2 b c).
  - apply Qmin2_le_r.
  - apply Qmin2_le_r.
Qed.

(** * Time Series Operations *)

(** Get value at index i (0-indexed). Returns 0 for out-of-bounds. *)
Definition ts_get (X : TimeSeries) (i : nat) : Q :=
  nth_q X i 0.

(** Length of a time series. *)
Definition ts_length (X : TimeSeries) : nat :=
  length X.

(** Absolute difference between two values. *)
Definition Qabs_diff (a b : Q) : Q :=
  Qabs (a - b).

(** * Lemmas about Qabs *)

Lemma Qabs_diff_nonneg : forall a b, 0 <= Qabs_diff a b.
Proof.
  intros. unfold Qabs_diff. apply Qabs_nonneg.
Qed.

Lemma Qabs_diff_symm : forall a b, Qabs_diff a b == Qabs_diff b a.
Proof.
  intros a b. unfold Qabs_diff.
  rewrite <- Qabs_opp.
  f_equiv. ring.
Qed.

Lemma Qabs_diff_zero : forall a, Qabs_diff a a == 0.
Proof.
  intros a. unfold Qabs_diff.
  setoid_replace (a - a) with 0 by ring.
  (* Qabs 0 == 0 *)
  apply Qabs_case; intros _; reflexivity.
Qed.

Lemma Qabs_diff_zero_iff : forall a b, Qabs_diff a b == 0 <-> a == b.
Proof.
  intros a b. split.
  - (* |a - b| == 0 -> a == b *)
    intros H. unfold Qabs_diff in H.
    (* Use Qabs_case to analyze |a - b| *)
    destruct (Qlt_le_dec 0 (a - b)) as [Hpos | Hneg].
    + (* a - b > 0, so |a - b| = a - b *)
      rewrite Qabs_pos in H by (apply Qlt_le_weak; assumption).
      (* a - b == 0 and a - b > 0 is a contradiction, but let's derive a == b *)
      (* Actually H : a - b == 0, so a == b *)
      apply Qplus_inj_r with (z := -b). ring_simplify. exact H.
    + destruct (Qeq_dec (a - b) 0) as [Heq | Hneq].
      * (* a - b == 0 *)
        apply Qplus_inj_r with (z := -b). ring_simplify. exact Heq.
      * (* a - b < 0 *)
        assert (Hlt : a - b < 0).
        { destruct (Qle_lt_or_eq _ _ Hneg) as [Hlt | Heq].
          - exact Hlt.
          - exfalso. apply Hneq. exact Heq. }
        rewrite Qabs_neg in H by (apply Qlt_le_weak; assumption).
        (* -(a - b) == 0 means a - b == 0 *)
        assert (Hab : a - b == 0).
        { setoid_replace (a - b) with (- - (a - b)) by ring. rewrite H. ring. }
        apply Qplus_inj_r with (z := -b). ring_simplify. exact Hab.
  - (* a == b -> |a - b| == 0 *)
    intros H. unfold Qabs_diff.
    setoid_rewrite H. setoid_replace (b - b) with 0 by ring.
    apply Qabs_case; intros _; reflexivity.
Qed.

Lemma Qabs_triangle_diff : forall a b c,
  Qabs_diff a c <= Qabs_diff a b + Qabs_diff b c.
Proof.
  intros a b c. unfold Qabs_diff.
  setoid_replace (a - c) with ((a - b) + (b - c)) by ring.
  apply Qabs_triangle.
Qed.
