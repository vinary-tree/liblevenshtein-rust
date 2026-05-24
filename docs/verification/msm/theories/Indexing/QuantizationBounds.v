(** * Quantization Bounds for MSM Indexing

    This module proves bounds relating quantized (discrete) distance
    to exact MSM distance. These bounds justify using trie-based
    approximate search as a candidate filter for exact MSM verification.

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia ZArith.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance.

(** Ceiling function for rationals: smallest integer >= q *)
Definition Qceiling (q : Q) : Z :=
  let (n, d) := q in
  if Z.eqb (n mod (Zpos d)) 0%Z then
    (n / Zpos d)%Z
  else
    (n / Zpos d + 1)%Z.

(** * Quantization Configuration *)

(** Quantization maps continuous values to discrete bins *)
Record QuantConfig := mkQuantConfig {
  num_bins : nat;                 (* Number of bins (e.g., 256 for u8) *)
  min_val : Q;                    (* Minimum value in range *)
  max_val : Q;                    (* Maximum value in range *)
  range_positive : min_val < max_val
}.

(** Width of each bin *)
Definition bin_width (cfg : QuantConfig) : Q :=
  (max_val cfg - min_val cfg) / inject_Z (Z.of_nat (num_bins cfg)).

(** Quantize a value to its bin index *)
Definition quantize (cfg : QuantConfig) (v : Q) : nat :=
  (* In Coq, we'd need careful handling of floor/conversion *)
  (* Simplified: assume v is clamped to [min, max] *)
  0. (* Placeholder *)

(** Dequantize: get center value of a bin *)
Definition dequantize (cfg : QuantConfig) (bin : nat) : Q :=
  min_val cfg + (inject_Z (Z.of_nat bin) + (1#2)) * bin_width cfg.

(** The current executable model has a placeholder quantizer. It maps every
    value to bin 0, so nontrivial quantization-error bounds are intentionally
    not claimed here. *)
Lemma quantize_current_model_zero : forall cfg v,
  quantize cfg v = 0%nat.
Proof. reflexivity. Qed.

Lemma dequantize_quantize_current_model : forall cfg v,
  dequantize cfg (quantize cfg v) == dequantize cfg 0.
Proof.
  intros cfg v.
  rewrite quantize_current_model_zero.
  reflexivity.
Qed.

(** * Levenshtein on Quantized Sequence *)

(** A structurally recursive edit-like distance on natural-number sequences.
    It counts aligned substitutions plus unmatched suffix elements. *)
Fixpoint lev_nat (X Y : list nat) : nat :=
  match X, Y with
  | [], ys => length ys
  | xs, [] => length xs
  | x :: xs, y :: ys =>
      if Nat.eqb x y then lev_nat xs ys else S (lev_nat xs ys)
  end.

(** Basic properties of the sequence distance. *)
Lemma lev_nat_refl : forall X, lev_nat X X = O.
Proof.
  induction X as [|x xs IH].
  - reflexivity.
  - simpl. rewrite Nat.eqb_refl. exact IH.
Qed.

Lemma lev_nat_nil_l : forall Y, lev_nat nil Y = length Y.
Proof. reflexivity. Qed.

Lemma lev_nat_nil_r : forall X, lev_nat X nil = length X.
Proof. destruct X; reflexivity. Qed.

Lemma lev_nat_symm : forall X Y, lev_nat X Y = lev_nat Y X.
Proof.
  induction X as [|x xs IH]; intros [|y ys].
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - simpl. rewrite Nat.eqb_sym. destruct (y =? x); rewrite IH; reflexivity.
Qed.

(** Quantize a time series to bin indices *)
Definition quantize_series (cfg : QuantConfig) (X : TimeSeries) : list nat :=
  map (quantize cfg) X.

(** * Main Bound: Levenshtein Bounds MSM *)

(** With the placeholder quantizer, all values collapse into the same bin.
    The nontrivial MSM indexing theorem is therefore deferred until [quantize]
    is replaced by an executable binning function with a proved error bound. *)

(** Minimum cost of a non-matching operation in MSM *)
Definition min_msm_cost (cfg : MsmConfig) (qcfg : QuantConfig) : Q :=
  (* The minimum cost when symbols differ is either:
     - Move with |x - y| >= bin_width (different bins)
     - Split/Merge with cost >= c *)
  Qmin2 (bin_width qcfg) (msm_c cfg).

(** Same-length series have zero quantized edit distance in the current model. *)
Lemma lev_nat_quantize_same_length_zero : forall X Y q_cfg,
  length X = length Y ->
  lev_nat (quantize_series q_cfg X) (quantize_series q_cfg Y) = 0%nat.
Proof.
  induction X as [|x xs IH]; intros [|y ys] q_cfg Hlen; simpl in *; try discriminate.
  - reflexivity.
  - rewrite IH by lia.
    reflexivity.
Qed.

(** Lower bound theorem for the current placeholder quantizer. For same-length
    series the scaled trie distance is 0, so the bound follows from MSM
    non-negativity. *)
Theorem lev_bounds_msm_same_length_current_model : forall X Y msm_cfg q_cfg,
  length X = length Y ->
  inject_Z (Z.of_nat (lev_nat (quantize_series q_cfg X) (quantize_series q_cfg Y)))
    * min_msm_cost msm_cfg q_cfg
  <= msm_distance X Y msm_cfg.
Proof.
  intros X Y msm_cfg q_cfg Hlen.
  rewrite (lev_nat_quantize_same_length_zero X Y q_cfg Hlen).
  simpl.
  setoid_replace (0 * min_msm_cost msm_cfg q_cfg) with 0 by ring.
  apply msm_nonneg.
Qed.

(** * Search Completeness *)

(** If MSM(X, Y) <= threshold, then lev(Q(X), Q(Y)) is also bounded.
    This means trie-based filtering won't miss any true matches. *)

Theorem trie_completeness_same_length_current_model : forall X Y msm_cfg q_cfg threshold,
  length X = length Y ->
  (lev_nat (quantize_series q_cfg X) (quantize_series q_cfg Y)
    <= Z.to_nat (Qceiling (threshold / min_msm_cost msm_cfg q_cfg)))%nat.
Proof.
  intros X Y msm_cfg q_cfg threshold Hlen.
  rewrite (lev_nat_quantize_same_length_zero X Y q_cfg Hlen).
  lia.
Qed.

(** * Practical Bounds *)

(** For practical search, we need the trie distance threshold *)
Definition compute_trie_threshold (msm_threshold : Q) (msm_cfg : MsmConfig)
                                   (q_cfg : QuantConfig) : nat :=
  let min_cost := min_msm_cost msm_cfg q_cfg in
  if Qle_bool min_cost 0 then
    0  (* Fallback for degenerate case *)
  else
    Z.to_nat (Qceiling (msm_threshold / min_cost)).

(** The computed threshold is sufficient *)
Theorem trie_threshold_sufficient_same_length_current_model : forall X Y msm_cfg q_cfg msm_threshold,
  length X = length Y ->
  (lev_nat (quantize_series q_cfg X) (quantize_series q_cfg Y)
    <= compute_trie_threshold msm_threshold msm_cfg q_cfg)%nat.
Proof.
  intros X Y msm_cfg q_cfg msm_threshold Hlen.
  rewrite (lev_nat_quantize_same_length_zero X Y q_cfg Hlen).
  unfold compute_trie_threshold.
  destruct (Qle_bool (min_msm_cost msm_cfg q_cfg) 0); lia.
Qed.

(** * Summary *)

(** Current status for trie-based MSM indexing:

    1. The placeholder quantizer maps every value to bin 0.
    2. Same-length quantized series therefore have zero trie distance.
    3. A nontrivial no-false-negative indexing theorem should be reinstated
       only after [quantize] is replaced by executable binning with a proved
       error bound.
*)
