(** * Quantization Bounds for MSM Indexing

    Executable uniform binning with the per-bin interval [bin_bounds] consumed by
    the interval-MSM transducer, and the soundness theorem [quantize_in_bin_bounds]:
    every value lies inside the interval of the bin it quantizes to.

    This replaces the earlier placeholder quantizer (which mapped everything to
    bin 0). With a real binning function and a proved per-bin error bound, the
    no-false-negative indexing theorem deferred by that placeholder is now
    discharged: [bin_bounds] feeds [Indexing.IntervalColumn] (whose
    [interval_cell_le_matrix] needs exactly [in_itv (target value) (its bin)]),
    so interval pruning over the trie never drops a true MSM match.

    The extreme bins extend to +/- infinity (modelled as [None] endpoints in
    [Qitv]): everything at or below [min_val] folds into bin 0 (no lower bound),
    everything at or above [max_val] folds into the last bin (no upper bound).
    Interior bins use the closed span [[min + b*w, min + (b+1)*w]]; the closed
    upper endpoint is a harmless over-approximation, matching the Rust
    `QuantizationConfig::bin_bounds`.

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia ZArith Psatz.
From Stdlib Require Import QArith Qabs Qminmax Qround.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance IntervalCost.

(** Quantization configuration: [num_bins] uniform bins over [[min_val, max_val]]. *)
Record QuantConfig := mkQuantConfig {
  num_bins : nat;
  min_val : Q;
  max_val : Q;
  range_positive : min_val < max_val;
  bins_positive : (0 < num_bins)%nat
}.

Definition bin_width (cfg : QuantConfig) : Q :=
  (max_val cfg - min_val cfg) / inject_Z (Z.of_nat (num_bins cfg)).

(** Quantize to a bin index, folding out-of-range values into the extreme bins. *)
Definition quantize (cfg : QuantConfig) (v : Q) : nat :=
  if Qle_bool v (min_val cfg) then 0%nat
  else if Qle_bool (max_val cfg) v then (num_bins cfg - 1)%nat
  else Z.to_nat (Qfloor ((v - min_val cfg) / bin_width cfg)).

(** Per-bin interval [[lo, hi]] (extreme bins unbounded), as a [Qitv]. *)
Definition bin_bounds (cfg : QuantConfig) (b : nat) : Qitv :=
  ((if Nat.eqb b 0 then None
    else Some (min_val cfg + inject_Z (Z.of_nat b) * bin_width cfg)),
   (if Nat.eqb b (num_bins cfg - 1) then None
    else Some (min_val cfg + inject_Z (Z.of_nat (S b)) * bin_width cfg))).

(** * Arithmetic facts *)

Lemma num_bins_Q_pos : forall cfg, 0 < inject_Z (Z.of_nat (num_bins cfg)).
Proof.
  intro cfg. pose proof (bins_positive cfg) as Hnb.
  unfold Qlt; simpl; lia.
Qed.

Lemma num_bins_Q_nonzero : forall cfg, ~ inject_Z (Z.of_nat (num_bins cfg)) == 0.
Proof.
  intros cfg H. pose proof (bins_positive cfg) as Hnb.
  unfold Qeq in H; simpl in H; lia.
Qed.

Lemma range_pos : forall cfg, 0 < max_val cfg - min_val cfg.
Proof. intro cfg. pose proof (range_positive cfg). psatz Q. Qed.

Lemma bin_width_pos : forall cfg, 0 < bin_width cfg.
Proof.
  intro cfg. unfold bin_width.
  apply Qlt_shift_div_l.
  - apply num_bins_Q_pos.
  - rewrite Qmult_0_l. apply range_pos.
Qed.

Lemma bin_width_nonneg : forall cfg, 0 <= bin_width cfg.
Proof. intro cfg. apply Qlt_le_weak. apply bin_width_pos. Qed.

Lemma bin_width_ne0 : forall cfg, ~ bin_width cfg == 0.
Proof.
  intro cfg. apply Qnot_eq_sym. apply Qlt_not_eq. apply bin_width_pos.
Qed.

(** [a / b * b == a] for nonzero [b]. *)
Lemma div_mult_cancel : forall a b, ~ b == 0 -> a / b * b == a.
Proof.
  intros a b Hb. unfold Qdiv.
  rewrite <- Qmult_assoc.
  rewrite (Qmult_comm (/ b) b).
  rewrite Qmult_inv_r by exact Hb.
  ring.
Qed.

Lemma num_bins_times_width : forall cfg,
  inject_Z (Z.of_nat (num_bins cfg)) * bin_width cfg == max_val cfg - min_val cfg.
Proof.
  intro cfg. unfold bin_width.
  rewrite Qmult_comm. apply div_mult_cancel. apply num_bins_Q_nonzero.
Qed.

Lemma min_plus_all_bins : forall cfg,
  min_val cfg + inject_Z (Z.of_nat (num_bins cfg)) * bin_width cfg == max_val cfg.
Proof.
  intro cfg. rewrite num_bins_times_width. ring.
Qed.

Lemma Qfloor_nonneg : forall r, 0 <= r -> (0 <= Qfloor r)%Z.
Proof.
  intros r Hr. destruct (Z_lt_le_dec (Qfloor r) 0) as [Hlt | Hle]; [| exact Hle].
  exfalso.
  pose proof (Qlt_floor r) as Hf.
  assert (Hle0 : inject_Z (Qfloor r + 1) <= 0) by (unfold Qle; simpl; lia).
  apply (Qlt_not_le r 0); [| exact Hr].
  apply Qlt_le_trans with (inject_Z (Qfloor r + 1)); assumption.
Qed.

(** * Soundness: a value lies in the interval of the bin it quantizes to *)

Theorem quantize_in_bin_bounds : forall cfg v,
  in_itv v (bin_bounds cfg (quantize cfg v)).
Proof.
  intros cfg v.
  pose proof (bin_width_pos cfg) as Hw.
  pose proof (bin_width_nonneg cfg) as Hw0.
  pose proof (bin_width_ne0 cfg) as Hwne.
  pose proof (bins_positive cfg) as Hnb.
  unfold quantize.
  destruct (Qle_bool v (min_val cfg)) eqn:Hmin.
  - (* v <= min => bin 0 *)
    apply Qle_bool_iff in Hmin.
    unfold bin_bounds, in_itv. cbn [fst snd].
    split; [ exact I |].
    destruct (Nat.eqb 0 (num_bins cfg - 1)); [ exact I |].
    apply Qle_trans with (min_val cfg); [ exact Hmin |].
    rewrite <- (Qplus_0_r (min_val cfg)) at 1.
    apply Qplus_le_compat; [ apply Qle_refl |].
    change (Z.of_nat 1) with 1%Z. rewrite Qmult_1_l. exact Hw0.
  - destruct (Qle_bool (max_val cfg) v) eqn:Hmax.
    + (* max <= v => bin num_bins - 1 *)
      apply Qle_bool_iff in Hmax.
      unfold bin_bounds, in_itv. cbn [fst snd].
      split.
      * destruct (Nat.eqb (num_bins cfg - 1) 0); [ exact I |].
        apply Qle_trans with (max_val cfg); [| exact Hmax ].
        rewrite <- min_plus_all_bins.
        apply Qplus_le_compat; [ apply Qle_refl |].
        apply Qmult_le_compat_r; [| exact Hw0 ].
        unfold Qle; simpl; lia.
      * rewrite Nat.eqb_refl. exact I.
    + (* interior: min < v < max *)
      assert (Hvmin : min_val cfg < v).
      { destruct (Qlt_le_dec (min_val cfg) v) as [Hlt | Hle]; [ exact Hlt |].
        assert (Qle_bool v (min_val cfg) = true) by (apply Qle_bool_iff; exact Hle).
        rewrite Hmin in H. discriminate. }
      assert (Hvmax : v < max_val cfg).
      { destruct (Qlt_le_dec v (max_val cfg)) as [Hlt | Hle]; [ exact Hlt |].
        assert (Qle_bool (max_val cfg) v = true) by (apply Qle_bool_iff; exact Hle).
        rewrite Hmax in H. discriminate. }
      set (r := (v - min_val cfg) / bin_width cfg).
      assert (Hr0 : 0 <= r).
      { unfold r. apply Qle_shift_div_l; [ exact Hw |]. rewrite Qmult_0_l. psatz Q. }
      set (b := Z.to_nat (Qfloor r)).
      assert (Hbz : inject_Z (Z.of_nat b) == inject_Z (Qfloor r)).
      { unfold b. rewrite Z2Nat.id; [ reflexivity | apply Qfloor_nonneg; exact Hr0 ]. }
      assert (Hcancel : r * bin_width cfg == v - min_val cfg).
      { unfold r. apply div_mult_cancel. exact Hwne. }
      assert (Hlow : min_val cfg + inject_Z (Z.of_nat b) * bin_width cfg <= v).
      { rewrite Hbz.
        apply Qle_trans with (min_val cfg + r * bin_width cfg).
        - apply Qplus_le_compat; [ apply Qle_refl |].
          apply Qmult_le_compat_r; [ apply Qfloor_le | exact Hw0 ].
        - rewrite Hcancel. psatz Q. }
      assert (Hhigh : v <= min_val cfg + inject_Z (Z.of_nat (S b)) * bin_width cfg).
      { rewrite Nat2Z.inj_succ. unfold Z.succ. rewrite inject_Z_plus. rewrite Hbz.
        apply Qle_trans with (min_val cfg + r * bin_width cfg).
        - rewrite Hcancel. psatz Q.
        - apply Qplus_le_compat; [ apply Qle_refl |].
          apply Qmult_le_compat_r; [| exact Hw0 ].
          apply Qle_trans with (inject_Z (Qfloor r + 1)).
          + apply Qlt_le_weak. apply Qlt_floor.
          + rewrite inject_Z_plus. apply Qle_refl. }
      unfold bin_bounds, in_itv. cbn [fst snd].
      split.
      * destruct (Nat.eqb b 0); [ exact I | exact Hlow ].
      * destruct (Nat.eqb b (num_bins cfg - 1)); [ exact I | exact Hhigh ].
Qed.
