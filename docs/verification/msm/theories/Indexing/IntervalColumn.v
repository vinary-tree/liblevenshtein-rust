(** * Interval-Relaxed MSM Column: admissibility and pruning soundness

    Defines the interval-relaxed MSM dynamic-program cell [interval_cell] (the
    Coq analogue of the Rust `step_interval_column` family) and proves that it
    lower-bounds the exact MSM matrix cell [msm_matrix_cell] for every concrete
    target whose values lie in the per-column bins, hence that pruning on the
    column minimum never drops a true match.

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia Psatz.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance Symmetry IntervalCost.

(** Fuel-parameterised interval DP cell. [fuel] only needs to exceed [i + j];
    [interval_cell] supplies exactly enough. The recurrence mirrors
    [msm_matrix_cell]: rows [i] index the (scalar) query [X = x :: xs]; columns
    [j] index the target, whose [j]-th element is known only to lie in bin
    [nth j B]. Each exact per-element cost is replaced by its admissible lower
    bound from [IntervalCost]. *)
Fixpoint icell (fuel : nat) (X : list Q) (B : list Qitv) (c : Q) (i j : nat) : Q :=
  match fuel with
  | O => 0
  | S f =>
    match i, j with
    | O, O => interval_dist (nth 0 X 0) (nth 0 B (None, None))
    | O, S j' =>
        icell f X B c 0 j'
        + c_func_split_lb c (nth (S j') B (None, None)) (nth 0 X 0) (nth j' B (None, None))
    | S i', O =>
        icell f X B c i' 0
        + c_func_merge_lb c (nth (S i') X 0) (nth i' X 0) (nth 0 B (None, None))
    | S i', S j' =>
        Qmin3
          (icell f X B c i' j'
           + interval_dist (nth (S i') X 0) (nth (S j') B (None, None)))
          (icell f X B c i' (S j')
           + c_func_merge_lb c (nth (S i') X 0) (nth i' X 0) (nth (S j') B (None, None)))
          (icell f X B c (S i') j'
           + c_func_split_lb c (nth (S j') B (None, None)) (nth (S i') X 0) (nth j' B (None, None)))
    end
  end.

Definition interval_cell (X : list Q) (B : list Qitv) (c : Q) (i j : nat) : Q :=
  icell (S (i + j)) X B c i j.

(** One-step unfolding lemmas (definitional). *)
Lemma icell_unfold_00 : forall f X B c,
  icell (S f) X B c 0 0 = interval_dist (nth 0 X 0) (nth 0 B (None, None)).
Proof. reflexivity. Qed.

Lemma icell_unfold_0S : forall f X B c j,
  icell (S f) X B c 0 (S j) =
  icell f X B c 0 j
  + c_func_split_lb c (nth (S j) B (None, None)) (nth 0 X 0) (nth j B (None, None)).
Proof. reflexivity. Qed.

Lemma icell_unfold_S0 : forall f X B c i,
  icell (S f) X B c (S i) 0 =
  icell f X B c i 0
  + c_func_merge_lb c (nth (S i) X 0) (nth i X 0) (nth 0 B (None, None)).
Proof. reflexivity. Qed.

Lemma icell_unfold_SS : forall f X B c i j,
  icell (S f) X B c (S i) (S j) =
  Qmin3
    (icell f X B c i j + interval_dist (nth (S i) X 0) (nth (S j) B (None, None)))
    (icell f X B c i (S j) + c_func_merge_lb c (nth (S i) X 0) (nth i X 0) (nth (S j) B (None, None)))
    (icell f X B c (S i) j + c_func_split_lb c (nth (S j) B (None, None)) (nth (S i) X 0) (nth j B (None, None))).
Proof. reflexivity. Qed.

(** Fuel irrelevance: any two fuels exceeding the index sum give the same cell. *)
Lemma icell_irrel : forall f1 X B c i j,
  (i + j < f1)%nat -> forall f2, (i + j < f2)%nat ->
  icell f1 X B c i j == icell f2 X B c i j.
Proof.
  induction f1 as [|f1 IH]; intros X B c i j H1 f2 H2; [ lia |].
  destruct f2 as [|f2]; [ lia |].
  destruct i as [|i']; destruct j as [|j']; cbn [icell].
  - reflexivity.
  - rewrite (IH X B c 0%nat j' ltac:(lia) f2 ltac:(lia)). reflexivity.
  - rewrite (IH X B c i' 0%nat ltac:(lia) f2 ltac:(lia)). reflexivity.
  - apply Qmin3_wd.
    + rewrite (IH X B c i' j' ltac:(lia) f2 ltac:(lia)). reflexivity.
    + rewrite (IH X B c i' (S j') ltac:(lia) f2 ltac:(lia)). reflexivity.
    + rewrite (IH X B c (S i') j' ltac:(lia) f2 ltac:(lia)). reflexivity.
Qed.

(** Recurrence lemmas (the Coq analogue of `step_interval_column`). All fuels
    exceed the relevant index sums, so [icell_irrel] reconciles the differing
    fuel expressions that arise because [Nat.add] does not reduce on a variable
    first argument. *)
Lemma interval_cell_base : forall X B c,
  interval_cell X B c 0 0 == interval_dist (nth 0 X 0) (nth 0 B (None, None)).
Proof. intros. unfold interval_cell. rewrite icell_unfold_00. reflexivity. Qed.

Lemma interval_cell_first_row : forall X B c j,
  interval_cell X B c 0 (S j) ==
  interval_cell X B c 0 j
  + c_func_split_lb c (nth (S j) B (None, None)) (nth 0 X 0) (nth j B (None, None)).
Proof.
  intros. unfold interval_cell. rewrite icell_unfold_0S.
  setoid_replace (icell (0 + S j) X B c 0 j) with (icell (S (0 + j)) X B c 0 j)
    by (apply icell_irrel; lia).
  reflexivity.
Qed.

Lemma interval_cell_first_col : forall X B c i,
  interval_cell X B c (S i) 0 ==
  interval_cell X B c i 0
  + c_func_merge_lb c (nth (S i) X 0) (nth i X 0) (nth 0 B (None, None)).
Proof.
  intros. unfold interval_cell. rewrite icell_unfold_S0.
  setoid_replace (icell (S i + 0) X B c i 0) with (icell (S (i + 0)) X B c i 0)
    by (apply icell_irrel; lia).
  reflexivity.
Qed.

Lemma interval_cell_interior : forall X B c i j,
  interval_cell X B c (S i) (S j) ==
  Qmin3
    (interval_cell X B c i j
     + interval_dist (nth (S i) X 0) (nth (S j) B (None, None)))
    (interval_cell X B c i (S j)
     + c_func_merge_lb c (nth (S i) X 0) (nth i X 0) (nth (S j) B (None, None)))
    (interval_cell X B c (S i) j
     + c_func_split_lb c (nth (S j) B (None, None)) (nth (S i) X 0) (nth j B (None, None))).
Proof.
  intros X B c i j. unfold interval_cell. rewrite icell_unfold_SS.
  apply Qmin3_wd.
  - setoid_replace (icell (S i + S j) X B c i j) with (icell (S (i + j)) X B c i j)
      by (apply icell_irrel; lia).
    reflexivity.
  - reflexivity.
  - setoid_replace (icell (S i + S j) X B c (S i) j) with (icell (S (S i + j)) X B c (S i) j)
      by (apply icell_irrel; lia).
    reflexivity.
Qed.

(** Monotonicity of [Qmin3] in all three arguments. *)
Lemma Qmin3_mono : forall a1 a2 a3 b1 b2 b3,
  a1 <= b1 -> a2 <= b2 -> a3 <= b3 -> Qmin3 a1 a2 a3 <= Qmin3 b1 b2 b3.
Proof.
  intros a1 a2 a3 b1 b2 b3 H1 H2 H3. apply Qmin2_glb.
  - apply Qle_trans with a1; [ apply Qmin3_le_a | exact H1 ].
  - apply Qmin2_glb.
    + apply Qle_trans with a2; [ apply Qmin3_le_b | exact H2 ].
    + apply Qle_trans with a3; [ apply Qmin3_le_c | exact H3 ].
Qed.

(** * Admissibility: the interval cell lower-bounds the exact matrix cell

    For every concrete target [y :: ys] whose [k]-th element lies in bin
    [nth k B], the interval-relaxed DP cell never exceeds the exact MSM matrix
    cell. Proved by strong induction on [i + j], discharging each of the four
    recurrence cases with the matching per-element bound from [IntervalCost]
    and [Qmin3] monotonicity. *)
Lemma interval_cell_le_matrix : forall n x xs y ys B c i j,
  (i + j <= n)%nat -> 0 <= c -> length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  (i < length (x :: xs))%nat -> (j < length (y :: ys))%nat ->
  interval_cell (x :: xs) B c i j <= msm_matrix_cell x xs y ys c i j.
Proof.
  induction n as [|n IH]; intros x xs y ys B c i j Hn Hc HBlen Hbins Hi Hj.
  - assert (i = 0)%nat as -> by lia. assert (j = 0)%nat as -> by lia.
    rewrite interval_cell_base, msm_matrix_cell_base. cbn [nth].
    assert (Hb0 : in_itv (nth 0 (y :: ys) 0) (nth 0 B (None, None)))
      by (apply Hbins; cbn [length]; lia).
    cbn [nth] in Hb0. apply interval_dist_le_move. exact Hb0.
  - destruct i as [|i']; destruct j as [|j'].
    + rewrite interval_cell_base, msm_matrix_cell_base. cbn [nth].
      assert (Hb0 : in_itv (nth 0 (y :: ys) 0) (nth 0 B (None, None)))
        by (apply Hbins; cbn [length]; lia).
      cbn [nth] in Hb0. apply interval_dist_le_move. exact Hb0.
    + (* first row *)
      rewrite interval_cell_first_row.
      rewrite msm_matrix_cell_first_row by exact Hj.
      apply Qplus_le_compat.
      * apply IH; try assumption; lia.
      * cbn [nth]. apply c_func_split_lb_le;
          [ exact Hc | apply Hbins; lia | apply Hbins; lia ].
    + (* first col *)
      rewrite interval_cell_first_col.
      rewrite msm_matrix_cell_first_col by exact Hi.
      apply Qplus_le_compat.
      * apply IH; try assumption; lia.
      * apply c_func_merge_lb_le; [ exact Hc | apply Hbins; cbn [length]; lia ].
    + (* interior *)
      rewrite interval_cell_interior.
      rewrite msm_matrix_cell_interior by assumption.
      apply Qmin3_mono.
      * apply Qplus_le_compat.
        -- apply IH; try assumption; lia.
        -- apply interval_dist_le_move. apply Hbins; lia.
      * apply Qplus_le_compat.
        -- apply IH; try assumption; lia.
        -- apply c_func_merge_lb_le; [ exact Hc | apply Hbins; lia ].
      * apply Qplus_le_compat.
        -- apply IH; try assumption; lia.
        -- apply c_func_split_lb_le; [ exact Hc | apply Hbins; lia | apply Hbins; lia ].
Qed.

(** * Column minimum and pruning soundness *)

(** Minimum interval cell over rows [0 .. rows] in column [j]. *)
Fixpoint col_min (X : list Q) (B : list Qitv) (c : Q) (j rows : nat) : Q :=
  match rows with
  | O => interval_cell X B c 0 j
  | S r => Qmin2 (col_min X B c j r) (interval_cell X B c (S r) j)
  end.

(** The subtree lower bound carried by a trie node at depth [j + 1]. *)
Definition column_lower_bound (X : list Q) (B : list Qitv) (c : Q) (j : nat) : Q :=
  col_min X B c j (length X - 1).

Lemma col_min_le_row : forall rows X B c j i,
  (i <= rows)%nat -> col_min X B c j rows <= interval_cell X B c i j.
Proof.
  induction rows as [|r IH]; intros X B c j i Hi.
  - assert (i = 0)%nat as -> by lia. apply Qle_refl.
  - destruct (Nat.eq_dec i (S r)) as [->|Hne].
    + cbn [col_min]. apply Qmin2_le_r.
    + cbn [col_min]. apply Qle_trans with (col_min X B c j r).
      * apply Qmin2_le_l.
      * apply IH. lia.
Qed.

(** The interval value at the final cell lower-bounds the exact MSM distance. *)
Lemma interval_final_le_msm : forall x xs y ys B cfg,
  length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  interval_cell (x :: xs) B (msm_c cfg) (length xs) (length ys)
  <= msm_distance (x :: xs) (y :: ys) cfg.
Proof.
  intros x xs y ys B cfg HBlen Hbins.
  rewrite msm_distance_matrix_cell_last.
  apply (interval_cell_le_matrix (length xs + length ys)).
  - lia.
  - apply msm_c_nonneg.
  - exact HBlen.
  - exact Hbins.
  - cbn [length]; lia.
  - cbn [length]; lia.
Qed.

(** The node subtree bound lower-bounds the exact MSM distance of its reference. *)
Theorem column_lower_bound_le_msm : forall x xs y ys B cfg,
  length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  column_lower_bound (x :: xs) B (msm_c cfg) (length ys)
  <= msm_distance (x :: xs) (y :: ys) cfg.
Proof.
  intros x xs y ys B cfg HBlen Hbins.
  unfold column_lower_bound.
  apply Qle_trans with (interval_cell (x :: xs) B (msm_c cfg) (length xs) (length ys)).
  - replace (length (x :: xs) - 1)%nat with (length xs) by (cbn [length]; lia).
    apply col_min_le_row. lia.
  - apply interval_final_le_msm; assumption.
Qed.

(** Pruning soundness (no false negatives): if the subtree bound exceeds the
    threshold, every reference below has MSM distance above it. Matches the
    shape of [Indexing.LowerBounds.lb_prune_sound_empty_left]. *)
Theorem lb_prune_sound_msm : forall x xs y ys B cfg threshold,
  length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  threshold < column_lower_bound (x :: xs) B (msm_c cfg) (length ys) ->
  threshold < msm_distance (x :: xs) (y :: ys) cfg.
Proof.
  intros x xs y ys B cfg threshold HBlen Hbins Hlb.
  apply Qlt_le_trans with (column_lower_bound (x :: xs) B (msm_c cfg) (length ys)).
  - exact Hlb.
  - apply column_lower_bound_le_msm; assumption.
Qed.

(** * Subtree bound: the column minimum bounds every deeper cell

    A trie node at depth [j + 1] prunes its whole subtree on [column_lower_bound].
    The references below it are deeper finals -- cells [(length xs, j')] with
    [j' >= j]. We show the column minimum is non-decreasing in [j] (every later
    cell is reached only by adding non-negative MSM costs), so the node bound
    lower-bounds every deeper cell. This is the formal content of "any DP path to
    a deeper final must cross the current column." *)

Lemma Qle_Qmin3 : forall m a b c, m <= a -> m <= b -> m <= c -> m <= Qmin3 a b c.
Proof. intros m a b c Ha Hb Hc. apply Qmin2_glb; [ exact Ha | apply Qmin2_glb; assumption ]. Qed.

Lemma Qmin2_mono : forall a b a' b', a <= a' -> b <= b' -> Qmin2 a b <= Qmin2 a' b'.
Proof.
  intros a b a' b' Ha Hb. apply Qmin2_glb.
  - apply Qle_trans with a; [ apply Qmin2_le_l | exact Ha ].
  - apply Qle_trans with b; [ apply Qmin2_le_r | exact Hb ].
Qed.

(** Minimum exact matrix cell over rows [0 .. rows] in column [j]. *)
Fixpoint mcol_min (x : Q) (xs : list Q) (y : Q) (ys : list Q) (c : Q) (j rows : nat) : Q :=
  match rows with
  | O => msm_matrix_cell x xs y ys c 0 j
  | S r => Qmin2 (mcol_min x xs y ys c j r) (msm_matrix_cell x xs y ys c (S r) j)
  end.

Lemma mcol_min_le_row : forall rows x xs y ys c j i,
  (i <= rows)%nat -> mcol_min x xs y ys c j rows <= msm_matrix_cell x xs y ys c i j.
Proof.
  induction rows as [|r IH]; intros x xs y ys c j i Hi.
  - assert (i = 0)%nat as -> by lia. apply Qle_refl.
  - destruct (Nat.eq_dec i (S r)) as [->|Hne].
    + cbn [mcol_min]. apply Qmin2_le_r.
    + cbn [mcol_min]. apply Qle_trans with (mcol_min x xs y ys c j r).
      * apply Qmin2_le_l.
      * apply IH. lia.
Qed.

(** Every cell in column [S j] is at least the column-[j] minimum. *)
Lemma mcol_min_le_cell_succ : forall i x xs y ys c j,
  0 <= c -> (S j < length (y :: ys))%nat -> (i < length (x :: xs))%nat ->
  mcol_min x xs y ys c j (length xs) <= msm_matrix_cell x xs y ys c i (S j).
Proof.
  induction i as [|i' IHi]; intros x xs y ys c j Hc Hj Hi;
    assert (HLx : length (x :: xs) = S (length xs)) by reflexivity.
  - rewrite msm_matrix_cell_first_row by exact Hj.
    apply Qle_trans with (msm_matrix_cell x xs y ys c 0 j).
    + apply mcol_min_le_row. lia.
    + apply Qle_plus_nonneg_r. apply c_func_nonneg. exact Hc.
  - rewrite msm_matrix_cell_interior by (try exact Hi; exact Hj).
    apply Qle_Qmin3.
    + apply Qle_trans with (msm_matrix_cell x xs y ys c i' j).
      * apply mcol_min_le_row. lia.
      * apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
    + apply Qle_trans with (msm_matrix_cell x xs y ys c i' (S j)).
      * apply IHi; [ exact Hc | exact Hj | lia ].
      * apply Qle_plus_nonneg_r. apply c_func_nonneg. exact Hc.
    + apply Qle_trans with (msm_matrix_cell x xs y ys c (S i') j).
      * apply mcol_min_le_row. lia.
      * apply Qle_plus_nonneg_r. apply c_func_nonneg. exact Hc.
Qed.

Lemma le_mcol_min : forall rows x xs y ys c j m,
  (forall i, (i <= rows)%nat -> m <= msm_matrix_cell x xs y ys c i (S j)) ->
  m <= mcol_min x xs y ys c (S j) rows.
Proof.
  induction rows as [|r IH]; intros x xs y ys c j m H; cbn [mcol_min].
  - apply H. lia.
  - apply Qmin2_glb; [ apply IH; intros i Hi; apply H; lia | apply H; lia ].
Qed.

Lemma mcol_min_le_succ : forall x xs y ys c j,
  0 <= c -> (S j < length (y :: ys))%nat ->
  mcol_min x xs y ys c j (length xs) <= mcol_min x xs y ys c (S j) (length xs).
Proof.
  intros x xs y ys c j Hc Hj. apply le_mcol_min. intros i Hi.
  apply mcol_min_le_cell_succ; [ exact Hc | exact Hj | cbn [length]; lia ].
Qed.

Lemma mcol_min_le_deeper : forall d x xs y ys c j,
  0 <= c -> (j + d < length (y :: ys))%nat ->
  mcol_min x xs y ys c j (length xs) <= mcol_min x xs y ys c (j + d) (length xs).
Proof.
  induction d as [|d IH]; intros x xs y ys c j Hc Hd.
  - replace (j + 0)%nat with j by lia. apply Qle_refl.
  - replace (j + S d)%nat with (S (j + d)) by lia.
    apply Qle_trans with (mcol_min x xs y ys c (j + d) (length xs)).
    + apply IH; [ exact Hc | lia ].
    + apply mcol_min_le_succ; [ exact Hc | lia ].
Qed.

(** The interval column minimum lower-bounds the exact matrix column minimum. *)
Lemma col_min_le_mcol_min : forall rows x xs y ys B c j,
  0 <= c -> length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  (j < length (y :: ys))%nat -> (rows < length (x :: xs))%nat ->
  col_min (x :: xs) B c j rows <= mcol_min x xs y ys c j rows.
Proof.
  induction rows as [|r IH]; intros x xs y ys B c j Hc HBlen Hbins Hj Hrows; cbn [col_min mcol_min].
  - apply (interval_cell_le_matrix (0 + j)); try assumption; lia.
  - apply Qmin2_mono.
    + apply IH; try assumption. lia.
    + apply (interval_cell_le_matrix (S r + j)); try assumption; lia.
Qed.

(** Subtree pruning bound: the depth-[j+1] node bound lower-bounds every deeper
    cell [(length xs, j')], [j' >= j], in the same DP -- hence the exact MSM
    distance of every reference reachable below the node. *)
Theorem column_lb_le_deeper : forall x xs y ys B c j j',
  0 <= c -> length B = length (y :: ys) ->
  (forall k, (k < length (y :: ys))%nat -> in_itv (nth k (y :: ys) 0) (nth k B (None, None))) ->
  (j <= j')%nat -> (j' < length (y :: ys))%nat ->
  column_lower_bound (x :: xs) B c j <= msm_matrix_cell x xs y ys c (length xs) j'.
Proof.
  intros x xs y ys B c j j' Hc HBlen Hbins Hjj' Hj'.
  unfold column_lower_bound.
  replace (length (x :: xs) - 1)%nat with (length xs) by (cbn [length]; lia).
  apply Qle_trans with (mcol_min x xs y ys c j (length xs)).
  - apply col_min_le_mcol_min; try assumption; [ lia | cbn [length]; lia ].
  - apply Qle_trans with (mcol_min x xs y ys c j' (length xs)).
    + replace j' with (j + (j' - j))%nat by lia.
      apply mcol_min_le_deeper; [ exact Hc | lia ].
    + apply mcol_min_le_row. lia.
Qed.
