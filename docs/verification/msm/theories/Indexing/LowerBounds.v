(** * MSM Lower Bounds

    This module proves that various lower bounds are valid for the MSM metric.
    These lower bounds enable efficient pruning during similarity search.

    Lower bounds proven in this executable model:
    1. Trivial L1 lower bounds for empty-side alignments
    2. Length difference (weighted by c) for empty-side alignments
    3. Combined pruning bounds for empty-side alignments

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance.

(** * Euclidean Distance as Lower Bound *)

(** Helper: zip two lists and apply a function (like map2) *)
Definition zip_with {A B C : Type} (f : A -> B -> C) (l1 : list A) (l2 : list B) : list C :=
  map (fun p => f (fst p) (snd p)) (combine l1 l2).

(** Euclidean distance between two series of the same length *)
Definition euclidean_dist (X Y : TimeSeries) : Q :=
  let squared_diffs := zip_with (fun x y => (x - y) * (x - y)) X Y in
  (* We'd need sqrt, so we use squared distance for now *)
  fold_right Qplus 0 squared_diffs.

(** L2 norm squared *)
Definition l2_squared (X Y : TimeSeries) : Q :=
  fold_right Qplus 0 (zip_with (fun x y => (x - y) * (x - y)) X Y).

(** L1 norm (sum of absolute differences) *)
Definition l1_dist (X Y : TimeSeries) : Q :=
  fold_right Qplus 0 (zip_with (fun x y => Qabs (x - y)) X Y).

(** The broad claim "L1 is always a lower bound for MSM" is not asserted here:
    split/merge paths can be cheaper than a pointwise diagonal path. The current
    executable proof keeps only the empty-side facts that follow directly from
    [msm_distance] non-negativity. *)

Lemma l1_lower_bound_empty_left : forall Y cfg,
  l1_dist [] Y <= msm_distance [] Y cfg.
Proof.
  intros Y cfg.
  unfold l1_dist, zip_with.
  destruct Y as [|y ys]; simpl.
  - apply Qle_refl.
  - apply Qmult_le_0_compat.
    + apply inject_Z_nonneg. lia.
    + apply msm_c_nonneg.
Qed.

Lemma l1_lower_bound_empty_right : forall X cfg,
  l1_dist X [] <= msm_distance X [] cfg.
Proof.
  intros X cfg.
  unfold l1_dist, zip_with.
  destruct X as [|x xs]; simpl.
  - apply Qle_refl.
  - apply Qmult_le_0_compat.
    + apply inject_Z_nonneg. lia.
    + apply msm_c_nonneg.
Qed.

(** * Length Difference as Lower Bound *)

(** Length-based lower bound: if lengths differ, we need split/merge operations *)
Definition length_lb (X Y : TimeSeries) (c : Q) : Q :=
  inject_Z (Z.of_nat (Nat.max (length X) (length Y) - Nat.min (length X) (length Y))) * c.

(** Nat absolute difference *)
Definition nat_abs_diff (n m : nat) : nat :=
  Nat.max n m - Nat.min n m.

(** Alternative formulation using nat_abs_diff *)
Definition length_lb' (X Y : TimeSeries) (c : Q) : Q :=
  inject_Z (Z.of_nat (nat_abs_diff (length X) (length Y))) * c.

(** Length lower bounds for the executable empty-side base cases. *)
Lemma length_lower_bound_empty_left : forall Y cfg,
  length_lb' [] Y (msm_c cfg) <= msm_distance [] Y cfg.
Proof.
  intros Y [c Hc].
  unfold length_lb', nat_abs_diff.
  destruct Y as [|y ys]; simpl.
  - setoid_replace (inject_Z 0 * c) with 0 by ring.
    apply Qle_refl.
  - apply Qle_refl.
Qed.

Lemma length_lower_bound_empty_right : forall X cfg,
  length_lb' X [] (msm_c cfg) <= msm_distance X [] cfg.
Proof.
  intros X [c Hc].
  unfold length_lb', nat_abs_diff.
  destruct X as [|x xs]; simpl.
  - setoid_replace (inject_Z 0 * c) with 0 by ring.
    apply Qle_refl.
  - apply Qle_refl.
Qed.

(** * Combined Lower Bound *)

(** Combined lower bound: maximum of all individual bounds *)
Definition combined_lb (X Y : TimeSeries) (c : Q) : Q :=
  Qmax (l1_dist X Y) (length_lb' X Y c).

(** Combined lower bounds for empty-side cases. *)
Theorem combined_lower_bound_empty_left : forall Y cfg,
  combined_lb [] Y (msm_c cfg) <= msm_distance [] Y cfg.
Proof.
  intros Y cfg.
  unfold combined_lb.
  apply Q.max_lub.
  - apply l1_lower_bound_empty_left.
  - apply length_lower_bound_empty_left.
Qed.

Theorem combined_lower_bound_empty_right : forall X cfg,
  combined_lb X [] (msm_c cfg) <= msm_distance X [] cfg.
Proof.
  intros X cfg.
  unfold combined_lb.
  apply Q.max_lub.
  - apply l1_lower_bound_empty_right.
  - apply length_lower_bound_empty_right.
Qed.

(** * Search Correctness *)

(** Using the proved empty-side lower bounds for search is sound. *)
Theorem lb_prune_sound_empty_left : forall Y cfg threshold,
  threshold < combined_lb [] Y (msm_c cfg) ->
  threshold < msm_distance [] Y cfg.
Proof.
  intros Y cfg threshold Hlb.
  apply Qlt_le_trans with (y := combined_lb [] Y (msm_c cfg)).
  - assumption.
  - apply combined_lower_bound_empty_left.
Qed.

Theorem lb_prune_sound_empty_right : forall X cfg threshold,
  threshold < combined_lb X [] (msm_c cfg) ->
  threshold < msm_distance X [] cfg.
Proof.
  intros X cfg threshold Hlb.
  apply Qlt_le_trans with (y := combined_lb X [] (msm_c cfg)).
  - assumption.
  - apply combined_lower_bound_empty_right.
Qed.

(** * Tightness Analysis *)

(** The L1 bound is tight when series have the same length and optimal
    alignment is purely diagonal (only Move operations). *)

Lemma l1_bound_tight_empty : forall cfg,
  l1_dist [] [] == msm_distance [] [] cfg.
Proof.
  reflexivity.
Qed.

(** The length bound is tight when series have very different values,
    so split/merge is always cheaper than moves. *)

(** * Summary *)

(** Current status:

    1. Empty-side L1 and length lower bounds are proved directly.
    2. Empty-side combined pruning is sound.
    3. Broad L1/combined lower bounds for non-empty MSM series need a
       trace-optimality development before they can be reinstated.
*)
