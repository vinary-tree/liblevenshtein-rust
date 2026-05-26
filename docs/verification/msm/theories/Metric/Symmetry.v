(** * MSM Symmetry Property

    This module proves the symmetry property of the MSM metric:

    MSM(X, Y) = MSM(Y, X)

    This is non-trivial because the C function is NOT symmetric in its arguments:
    C(a, b, c) ≠ C(a, c, b) in general

    However, the overall MSM distance IS symmetric because of how the
    split and merge operations complement each other when swapping X and Y.

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance.

(** * Symmetry of Base Operations *)

(** Qabs_diff is symmetric *)
Lemma Qabs_diff_symmetric : forall a b,
  Qabs_diff a b == Qabs_diff b a.
Proof.
  intros a b. unfold Qabs_diff.
  rewrite <- Qabs_opp.
  f_equiv. ring.
Qed.

(** * Key Insight: Role Reversal

    When we compute MSM(X, Y) vs MSM(Y, X):
    - Move operations are symmetric: |x_i - y_j| = |y_j - x_i|
    - Split in MSM(X, Y) corresponds to Merge in MSM(Y, X)
    - Merge in MSM(X, Y) corresponds to Split in MSM(Y, X)

    The key is showing that these role reversals preserve total cost.
*)

(** * Trace-Based Symmetry Proof Approach

    A trace is a sequence of operations (Move/Split/Merge) that transforms X to Y.
    For each trace from X to Y, there's a "reversed" trace from Y to X with the same cost.
*)

Inductive MsmOp : Type :=
  | OpMove : Q -> Q -> MsmOp    (* Move from value a to value b *)
  | OpSplit : Q -> Q -> Q -> MsmOp  (* Split: a splits after seeing b, targeting c *)
  | OpMerge : Q -> Q -> Q -> MsmOp. (* Merge: a merges after seeing b, targeting c *)

(** Cost of a single operation *)
Definition op_cost (c_const : Q) (op : MsmOp) : Q :=
  match op with
  | OpMove a b => Qabs_diff a b
  | OpSplit a b c => c_func c_const a b c
  | OpMerge a b c => c_func c_const a b c
  end.

(** Reverse an operation: swaps roles of source and target *)
Definition reverse_op (op : MsmOp) : MsmOp :=
  match op with
  | OpMove a b => OpMove b a
  | OpSplit a b c => OpMerge a c b  (* Split becomes Merge with swapped context *)
  | OpMerge a b c => OpSplit a c b  (* Merge becomes Split with swapped context *)
  end.

(** Key lemma: reversing an operation preserves its cost *)
Lemma reverse_op_cost : forall c_const op,
  0 <= c_const ->
  op_cost c_const (reverse_op op) == op_cost c_const op.
Proof.
  intros c_const op Hc.
  destruct op.
  - (* OpMove *)
    simpl. apply Qabs_diff_symmetric.
  - (* OpSplit *)
    simpl.
    (* Need to show c_func c_const q1 q2 q == c_func c_const q1 q0 q2 *)
    (* This is the tricky part - the C function IS symmetric in b and c_val *)
    apply c_func_symm_bc.
  - (* OpMerge *)
    simpl.
    apply c_func_symm_bc.
Qed.

(** A trace is a list of operations *)
Definition Trace := list MsmOp.

(** Total cost of a trace *)
Definition trace_cost (c_const : Q) (tr : Trace) : Q :=
  fold_right (fun op acc => op_cost c_const op + acc) 0 tr.

(** Reverse a trace *)
Definition reverse_trace (tr : Trace) : Trace :=
  rev (map reverse_op tr).

(** Helper: fold_right preserves Qeq for proper initial values *)
Lemma fold_right_Qeq_init : forall (f : MsmOp -> Q -> Q) l init1 init2,
  (forall op acc1 acc2, acc1 == acc2 -> f op acc1 == f op acc2) ->
  init1 == init2 ->
  fold_right f init1 l == fold_right f init2 l.
Proof.
  intros f l init1 init2 Hf Hinit.
  induction l as [|x xs IH].
  - simpl. exact Hinit.
  - simpl. apply Hf. exact IH.
Qed.

(** Helper: for additive f, fold_right f init l == init + fold_right f 0 l *)
Lemma fold_right_additive : forall (g : MsmOp -> Q) l init,
  fold_right (fun op acc => g op + acc) init l ==
  init + fold_right (fun op acc => g op + acc) 0 l.
Proof.
  intros g l init.
  induction l as [|x xs IH].
  - simpl. ring.
  - simpl.
    rewrite IH.
    ring.
Qed.

(** Helper: fold_right sum is the same regardless of order (for commutative +) *)
Lemma fold_right_sum_rev : forall (g : MsmOp -> Q) l,
  fold_right (fun op acc => g op + acc) 0 (rev l) ==
  fold_right (fun op acc => g op + acc) 0 l.
Proof.
  intros g l.
  induction l as [|x xs IH].
  - simpl. reflexivity.
  - simpl.
    rewrite fold_right_app.
    simpl.
    rewrite fold_right_additive.
    rewrite IH.
    ring.
Qed.

(** Reversing a trace preserves cost *)
Lemma reverse_trace_cost : forall c_const tr,
  0 <= c_const ->
  trace_cost c_const (reverse_trace tr) == trace_cost c_const tr.
Proof.
  intros c_const tr Hc.
  unfold reverse_trace, trace_cost.
  (* Use the fact that summing over reversed list gives same result *)
  rewrite fold_right_sum_rev.
  (* Now show that applying reverse_op to each element preserves total cost *)
  induction tr as [|op tr' IH].
  - simpl. reflexivity.
  - simpl.
    rewrite IH.
    rewrite reverse_op_cost by assumption.
    reflexivity.
Qed.

(** Singleton/source-column symmetry is executable: the first-row recurrence
    and the first-column recurrence accumulate the same costs, with the two
    context arguments of [c_func] swapped. *)
Lemma msm_init_row_compute_rows_singleton_sym :
  forall x y ys acc_l acc_r c_const,
    acc_l == acc_r ->
    last (msm_init_row x y ys acc_l c_const) 0 ==
    last (msm_compute_rows ys y [x] x [acc_r] c_const) 0.
Proof.
  intros x y ys.
  revert y.
  induction ys as [|y' ys IH]; intros y acc_l acc_r c_const Hacc.
  - simpl. exact Hacc.
  - rewrite msm_init_row_cons_last.
    simpl.
    apply IH.
    rewrite Hacc.
    rewrite c_func_symm_bc.
    reflexivity.
Qed.

Lemma msm_distance_singleton_left_sym : forall x y ys cfg,
  msm_distance [x] (y :: ys) cfg == msm_distance (y :: ys) [x] cfg.
Proof.
  intros x y ys cfg.
  simpl.
  apply msm_init_row_compute_rows_singleton_sym.
  apply Qabs_diff_symmetric.
Qed.

Lemma msm_distance_singleton_right_sym : forall x xs y cfg,
  msm_distance (x :: xs) [y] cfg == msm_distance [y] (x :: xs) cfg.
Proof.
  intros x xs y cfg.
  symmetry.
  apply msm_distance_singleton_left_sym.
Qed.

Lemma msm_distance_two_two_sym : forall x x' y y' cfg,
  msm_distance [x; x'] [y; y'] cfg == msm_distance [y; y'] [x; x'] cfg.
Proof.
  intros x x' y y' cfg.
  simpl.
  apply Qle_antisym.
  - apply Qmin3_glb.
    + setoid_replace (Qabs_diff y x + Qabs_diff y' x') with
        (Qabs_diff x y + Qabs_diff x' y').
      * apply Qmin3_le_a.
      * rewrite Qabs_diff_symmetric.
        rewrite (Qabs_diff_symmetric y' x').
        ring.
    + setoid_replace
        (Qabs_diff y x +
         c_func (msm_c cfg) x' y x + c_func (msm_c cfg) y' y x')
        with
        (Qabs_diff x y +
         c_func (msm_c cfg) x' x y + c_func (msm_c cfg) y' x' y).
      * apply Qmin3_le_c.
      * rewrite Qabs_diff_symmetric.
        rewrite c_func_symm_bc.
        rewrite (c_func_symm_bc (msm_c cfg) y' y x').
        ring.
    + setoid_replace
        (Qabs_diff y x +
         c_func (msm_c cfg) y' y x + c_func (msm_c cfg) x' y' x)
        with
        (Qabs_diff x y +
         c_func (msm_c cfg) y' x y + c_func (msm_c cfg) x' x y').
      * apply Qmin3_le_b.
      * rewrite Qabs_diff_symmetric.
        rewrite c_func_symm_bc.
        rewrite (c_func_symm_bc (msm_c cfg) x' y' x).
        ring.
  - apply Qmin3_glb.
    + setoid_replace (Qabs_diff x y + Qabs_diff x' y') with
        (Qabs_diff y x + Qabs_diff y' x').
      * apply Qmin3_le_a.
      * rewrite Qabs_diff_symmetric.
        rewrite (Qabs_diff_symmetric x' y').
        ring.
    + setoid_replace
        (Qabs_diff x y +
         c_func (msm_c cfg) y' x y + c_func (msm_c cfg) x' x y')
        with
        (Qabs_diff y x +
         c_func (msm_c cfg) y' y x + c_func (msm_c cfg) x' y' x).
      * apply Qmin3_le_c.
      * rewrite Qabs_diff_symmetric.
        rewrite c_func_symm_bc.
        rewrite (c_func_symm_bc (msm_c cfg) x' x y').
        ring.
    + setoid_replace
        (Qabs_diff x y +
         c_func (msm_c cfg) x' x y + c_func (msm_c cfg) y' x' y)
        with
        (Qabs_diff y x +
         c_func (msm_c cfg) x' y x + c_func (msm_c cfg) y' y x').
      * apply Qmin3_le_b.
      * rewrite Qabs_diff_symmetric.
        rewrite c_func_symm_bc.
        rewrite (c_func_symm_bc (msm_c cfg) y' x' y).
        ring.
Qed.

Lemma Qmin2_wd : forall a a' b b',
  a == a' ->
  b == b' ->
  Qmin2 a b == Qmin2 a' b'.
Proof.
  intros a a' b b' Ha Hb.
  apply Qle_antisym.
  - apply Qmin2_glb.
    + eapply Qle_trans.
      * apply Qmin2_le_l.
      * rewrite Ha. apply Qle_refl.
    + eapply Qle_trans.
      * apply Qmin2_le_r.
      * rewrite Hb. apply Qle_refl.
  - apply Qmin2_glb.
    + eapply Qle_trans.
      * apply Qmin2_le_l.
      * rewrite <- Ha. apply Qle_refl.
    + eapply Qle_trans.
      * apply Qmin2_le_r.
      * rewrite <- Hb. apply Qle_refl.
Qed.

Lemma Qmin3_wd : forall a a' b b' c c',
  a == a' ->
  b == b' ->
  c == c' ->
  Qmin3 a b c == Qmin3 a' b' c'.
Proof.
  intros a a' b b' c c' Ha Hb Hc.
  unfold Qmin3.
  apply Qmin2_wd.
  - exact Ha.
  - apply Qmin2_wd; assumption.
Qed.

Lemma Qmin3_swap_bc : forall a b c,
  Qmin3 a b c == Qmin3 a c b.
Proof.
  intros a b c.
  unfold Qmin3.
  apply Qmin2_wd.
  - reflexivity.
  - apply Qmin2_comm.
Qed.

Lemma msm_two_left_next_cost_sym :
  forall x x' y_prev y first_l first_r second_l second_r c_const,
    first_l == first_r ->
    second_l == second_r ->
    Qmin3 (first_l + Qabs_diff x' y)
      (first_l + c_func c_const y x y_prev + c_func c_const x' x y)
      (second_l + c_func c_const y x' y_prev) ==
    Qmin3 (first_r + Qabs_diff y x')
      (second_r + c_func c_const y y_prev x')
      (first_r + c_func c_const y y_prev x + c_func c_const x' y x).
Proof.
  intros x x' y_prev y first_l first_r second_l second_r c_const Hfirst Hsecond.
  eapply Qeq_trans.
  - apply Qmin3_wd.
    + rewrite Hfirst.
      rewrite Qabs_diff_symmetric.
      reflexivity.
    + rewrite Hfirst.
      rewrite (c_func_symm_bc c_const y x y_prev).
      rewrite (c_func_symm_bc c_const x' x y).
      reflexivity.
    + rewrite Hsecond.
      rewrite (c_func_symm_bc c_const y x' y_prev).
      reflexivity.
  - apply Qmin3_swap_bc.
Qed.

Lemma msm_init_row_not_nil : forall x y ys prev_cost c_const,
  msm_init_row x y ys prev_cost c_const <> [].
Proof.
  intros x y ys prev_cost c_const.
  destruct ys; discriminate.
Qed.

Lemma msm_compute_row_init_row_cons_last :
  forall x x' y_prev y ys first_cost second_cost c_const,
    last (msm_compute_row x' x y_prev (y :: ys)
            (first_cost ::
             msm_init_row x y ys
               (first_cost + c_func c_const y x y_prev) c_const)
            second_cost c_const) 0 ==
    last (msm_compute_row x' x y ys
            (msm_init_row x y ys
               (first_cost + c_func c_const y x y_prev) c_const)
            (Qmin3 (first_cost + Qabs_diff x' y)
               (first_cost + c_func c_const y x y_prev +
                c_func c_const x' x y)
               (second_cost + c_func c_const y x' y_prev))
            c_const) 0.
Proof.
  intros x x' y_prev y ys first_cost second_cost c_const.
  destruct ys as [|y_next ys']; simpl.
  - reflexivity.
  - destruct (msm_init_row x y_next ys'
                (first_cost + c_func c_const y x y_prev +
                 c_func c_const y_next x y) c_const) as [|cost_up rest] eqn:Hinit.
    + exfalso.
      pose proof (msm_init_row_not_nil x y_next ys'
                    (first_cost + c_func c_const y x y_prev +
                     c_func c_const y_next x y) c_const) as Hnot_nil.
      apply Hnot_nil. exact Hinit.
    + simpl.
      reflexivity.
Qed.

Lemma msm_two_left_row_compute_rows_sym :
  forall x x' y_prev ys first_l first_r second_l second_r c_const,
    first_l == first_r ->
    second_l == second_r ->
    last (msm_compute_row x' x y_prev ys
            (msm_init_row x y_prev ys first_l c_const)
            second_l c_const) 0 ==
    last (msm_compute_rows ys y_prev [x; x'] x
            [first_r; second_r] c_const) 0.
Proof.
  intros x x' y_prev ys.
  revert y_prev.
  induction ys as [|y ys IH];
    intros y_prev first_l first_r second_l second_r c_const Hfirst Hsecond.
  - simpl. exact Hsecond.
  - eapply Qeq_trans.
    + apply msm_compute_row_init_row_cons_last.
    + simpl.
      apply IH.
      * rewrite Hfirst.
        rewrite c_func_symm_bc.
        reflexivity.
      * apply msm_two_left_next_cost_sym; assumption.
Qed.

Lemma msm_distance_two_left_sym : forall x x' y y' ys cfg,
  msm_distance [x; x'] (y :: y' :: ys) cfg ==
  msm_distance (y :: y' :: ys) [x; x'] cfg.
Proof.
  intros x x' y y' ys cfg.
  simpl.
  change
    (last (msm_compute_row x' x y (y' :: ys)
             (Qabs_diff x y ::
              msm_init_row x y' ys
                (Qabs_diff x y + c_func (msm_c cfg) y' x y)
                (msm_c cfg))
             (Qabs_diff x y + c_func (msm_c cfg) x' x y)
             (msm_c cfg)) 0 ==
     last (msm_compute_rows ys y' [x; x'] x
             [(Qabs_diff y x + c_func (msm_c cfg) y' y x)%Q;
              Qmin3 (Qabs_diff y x + Qabs_diff y' x')
                (Qabs_diff y x + c_func (msm_c cfg) x' y x +
                 c_func (msm_c cfg) y' y x')
                (Qabs_diff y x + c_func (msm_c cfg) y' y x +
                 c_func (msm_c cfg) x' y' x)]
             (msm_c cfg)) 0).
  eapply Qeq_trans.
  - apply msm_compute_row_init_row_cons_last.
  - apply msm_two_left_row_compute_rows_sym.
    + rewrite Qabs_diff_symmetric.
      rewrite c_func_symm_bc.
      reflexivity.
    + apply msm_two_left_next_cost_sym.
      * apply Qabs_diff_symmetric.
      * rewrite Qabs_diff_symmetric.
        rewrite c_func_symm_bc.
        reflexivity.
Qed.

Lemma msm_distance_two_right_sym : forall x x' xs y y' cfg,
  msm_distance (x :: x' :: xs) [y; y'] cfg ==
  msm_distance [y; y'] (x :: x' :: xs) cfg.
Proof.
  intros x x' xs y y' cfg.
  symmetry.
  apply msm_distance_two_left_sym.
Qed.

Lemma msm_distance_three_three_sym : forall x x' x'' y y' y'' cfg,
  msm_distance [x; x'; x''] [y; y'; y''] cfg ==
  msm_distance [y; y'; y''] [x; x'; x''] cfg.
Proof.
  intros x x' x'' y y' y'' cfg.
  change
    (Qmin3
       (msm_distance [x; x'] [y; y'] cfg + Qabs_diff x'' y'')
       (msm_distance [x; x'] [y; y'; y''] cfg + c_func (msm_c cfg) x'' x' y'')
       (msm_distance [x; x'; x''] [y; y'] cfg + c_func (msm_c cfg) y'' x'' y') ==
     Qmin3
       (msm_distance [y; y'] [x; x'] cfg + Qabs_diff y'' x'')
       (msm_distance [y; y'] [x; x'; x''] cfg + c_func (msm_c cfg) y'' y' x'')
       (msm_distance [y; y'; y''] [x; x'] cfg + c_func (msm_c cfg) x'' y'' x')).
  eapply Qeq_trans.
  - apply Qmin3_wd.
    + rewrite (msm_distance_two_two_sym x x' y y' cfg).
      rewrite Qabs_diff_symmetric.
      reflexivity.
    + rewrite (msm_distance_two_left_sym x x' y y' [y''] cfg).
      rewrite c_func_symm_bc.
      reflexivity.
    + rewrite (msm_distance_two_right_sym x x' [x''] y y' cfg).
      rewrite c_func_symm_bc.
      reflexivity.
  - apply Qmin3_swap_bc.
Qed.

(** * Matrix View of the Executable DP *)

(** The executable distance keeps only the current row.  For symmetry, it is
    useful to retain the rows so that we can prove that every cell in the
    X/Y matrix equals the transposed cell in the Y/X matrix. *)
Fixpoint msm_compute_rows_matrix (X_tail : list Q) (x_prev : Q) (Y : list Q)
         (y1 : Q) (prev_row : list Q) (c_const : Q) : list (list Q) :=
  match X_tail with
  | [] => [prev_row]
  | x_i :: xs =>
    let cost_i1 := hd 0 prev_row + c_func c_const x_i x_prev y1 in
    let new_row := msm_compute_row x_i x_prev y1 (tl Y) prev_row cost_i1 c_const in
    prev_row :: msm_compute_rows_matrix xs x_i Y y1 new_row c_const
  end.

Definition msm_matrix_nonempty (x : Q) (xs : list Q)
    (y : Q) (ys : list Q) (c_const : Q) : list (list Q) :=
  msm_compute_rows_matrix xs x (y :: ys) y
    (msm_init_row x y ys (Qabs_diff x y) c_const) c_const.

Definition msm_matrix_cell (x : Q) (xs : list Q)
    (y : Q) (ys : list Q) (c_const : Q) (i j : nat) : Q :=
  nth j (nth i (msm_matrix_nonempty x xs y ys c_const) []) 0.

Lemma last_eq_nth_by_length_any :
  forall (A : Type) (l : list A) d n,
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

Lemma msm_compute_rows_matrix_length :
  forall X_tail x_prev Y y1 row c_const,
    length (msm_compute_rows_matrix X_tail x_prev Y y1 row c_const) =
    S (length X_tail).
Proof.
  induction X_tail as [|x xs IH]; intros x_prev Y y1 row c_const.
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

Lemma msm_compute_rows_matrix_last :
  forall X_tail x_prev Y y1 row c_const,
    last (msm_compute_rows_matrix X_tail x_prev Y y1 row c_const) [] =
    msm_compute_rows X_tail x_prev Y y1 row c_const.
Proof.
  induction X_tail as [|x xs IH]; intros x_prev Y y1 row c_const.
  - simpl. reflexivity.
  - simpl.
    set (cost_i1 := hd 0 row + c_func c_const x x_prev y1).
    set (new_row := msm_compute_row x x_prev y1 (tl Y) row cost_i1 c_const).
    destruct (msm_compute_rows_matrix xs x Y y1 new_row c_const) as [|r rs] eqn:Hrows.
    + pose proof (msm_compute_rows_matrix_length xs x Y y1 new_row c_const) as Hlen.
      rewrite Hrows in Hlen. simpl in Hlen. discriminate.
    + simpl.
      specialize (IH x Y y1 new_row c_const).
      rewrite Hrows in IH.
      exact IH.
Qed.

Lemma msm_compute_rows_matrix_row_length :
  forall X_tail x_prev y ys row c_const i,
    length row = S (length ys) ->
    (i < length (msm_compute_rows_matrix X_tail x_prev (y :: ys) y row c_const))%nat ->
    length (nth i (msm_compute_rows_matrix X_tail x_prev (y :: ys) y row c_const) []) =
    S (length ys).
Proof.
  induction X_tail as [|x xs IH]; intros x_prev y ys row c_const i Hrow_len Hi.
  - destruct i as [|i']; simpl in *; [exact Hrow_len | lia].
  - simpl in Hi |- *.
    set (cost_i1 := hd 0 row + c_func c_const x x_prev y).
    set (new_row := msm_compute_row x x_prev y ys row cost_i1 c_const).
    destruct i as [|i'].
    + exact Hrow_len.
    + apply IH.
      * subst new_row.
        apply msm_compute_row_length.
        rewrite Hrow_len. lia.
      * apply Nat.succ_lt_mono. exact Hi.
Qed.

Lemma msm_compute_rows_matrix_nth_0 :
  forall X_tail x_prev Y y1 row c_const,
    nth 0 (msm_compute_rows_matrix X_tail x_prev Y y1 row c_const) [] = row.
Proof.
  intros X_tail x_prev Y y1 row c_const.
  destruct X_tail; reflexivity.
Qed.

Lemma msm_compute_rows_matrix_row_succ :
  forall X_tail x_prev y ys row c_const i,
    (S i < length (msm_compute_rows_matrix X_tail x_prev (y :: ys) y row c_const))%nat ->
    nth (S i) (msm_compute_rows_matrix X_tail x_prev (y :: ys) y row c_const) [] =
    let rows := msm_compute_rows_matrix X_tail x_prev (y :: ys) y row c_const in
    let X_full := x_prev :: X_tail in
    let row_i := nth i rows [] in
    msm_compute_row (nth (S i) X_full 0) (nth i X_full 0) y ys row_i
      (hd 0 row_i + c_func c_const (nth (S i) X_full 0) (nth i X_full 0) y)
      c_const.
Proof.
  induction X_tail as [|x xs IH]; intros x_prev y ys row c_const i Hi.
  - simpl in Hi. lia.
  - destruct i as [|i'].
    + simpl. apply msm_compute_rows_matrix_nth_0.
    + simpl in Hi |- *.
      set (cost_i1 := hd 0 row + c_func c_const x x_prev y).
      set (new_row := msm_compute_row x x_prev y ys row cost_i1 c_const).
      apply IH.
      apply Nat.succ_lt_mono. exact Hi.
Qed.

Lemma msm_init_row_nth_succ_eq :
  forall j x y ys prev_cost c_const,
    (j < length ys)%nat ->
    nth (S j) (msm_init_row x y ys prev_cost c_const) 0 ==
    nth j (msm_init_row x y ys prev_cost c_const) 0 +
      c_func c_const (nth j ys 0) x (nth j (y :: ys) 0).
Proof.
  induction j as [|j IH]; intros x y ys prev_cost c_const Hj.
  - destruct ys as [|y' ys']; [simpl in Hj; lia |].
    destruct ys' as [|y'' ys'']; simpl; reflexivity.
  - destruct ys as [|y' ys']; [simpl in Hj; lia |].
    simpl.
    apply IH. simpl in Hj. lia.
Qed.

Lemma msm_compute_row_nth_succ_eq :
  forall j x_i x_prev y_prev Y_tail prev_row cost_left c_const,
    (j < length Y_tail)%nat ->
    (S j < length prev_row)%nat ->
    nth (S j)
        (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 ==
    Qmin3
      (nth j prev_row 0 + Qabs_diff x_i (nth j Y_tail 0))
      (nth (S j) prev_row 0 + c_func c_const x_i x_prev (nth j Y_tail 0))
      (nth j
         (msm_compute_row x_i x_prev y_prev Y_tail prev_row cost_left c_const) 0 +
       c_func c_const (nth j Y_tail 0) x_i (nth j (y_prev :: Y_tail) 0)).
Proof.
  induction j as [|j IH]; intros x_i x_prev y_prev Y_tail prev_row cost_left c_const
                                Hj Hprev.
  - destruct Y_tail as [|y ys]; [simpl in Hj; lia |].
    destruct prev_row as [|cost_diag [|cost_up rest]]; simpl in Hprev; try lia.
    simpl.
    rewrite msm_compute_row_hd_eq.
    reflexivity.
  - destruct Y_tail as [|y ys]; [simpl in Hj; lia |].
    destruct prev_row as [|cost_diag [|cost_up rest]]; simpl in Hprev; try lia.
    simpl.
    apply IH; simpl in *; lia.
Qed.

Lemma msm_matrix_cell_base :
  forall x xs y ys c_const,
    msm_matrix_cell x xs y ys c_const 0 0 == Qabs_diff x y.
Proof.
  intros x xs y ys c_const.
  unfold msm_matrix_cell, msm_matrix_nonempty.
  destruct xs; simpl; destruct ys; simpl; reflexivity.
Qed.

Lemma msm_matrix_cell_first_row :
  forall x xs y ys c_const j,
    (S j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const 0 (S j) ==
    msm_matrix_cell x xs y ys c_const 0 j +
      c_func c_const (nth (S j) (y :: ys) 0) x (nth j (y :: ys) 0).
Proof.
  intros x xs y ys c_const j Hj.
  unfold msm_matrix_cell, msm_matrix_nonempty.
  destruct xs; simpl;
    change (nth (S j) (msm_init_row x y ys (Qabs_diff x y) c_const) 0 ==
            nth j (msm_init_row x y ys (Qabs_diff x y) c_const) 0 +
            c_func c_const (nth j ys 0) x (nth j (y :: ys) 0));
    apply msm_init_row_nth_succ_eq; simpl in Hj; lia.
Qed.

Lemma msm_matrix_cell_first_col :
  forall x xs y ys c_const i,
    (S i < length (x :: xs))%nat ->
    msm_matrix_cell x xs y ys c_const (S i) 0 ==
    msm_matrix_cell x xs y ys c_const i 0 +
      c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0) y.
Proof.
  intros x xs y ys c_const i Hi.
  unfold msm_matrix_cell, msm_matrix_nonempty.
  rewrite msm_compute_rows_matrix_row_succ.
  - rewrite msm_compute_row_hd_eq.
    destruct (nth i
      (msm_compute_rows_matrix xs x (y :: ys) y
         (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) as [|q qs];
      reflexivity.
  - rewrite msm_compute_rows_matrix_length. simpl in Hi. exact Hi.
Qed.

Lemma msm_matrix_cell_interior :
  forall x xs y ys c_const i j,
    (S i < length (x :: xs))%nat ->
    (S j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const (S i) (S j) ==
    Qmin3
      (msm_matrix_cell x xs y ys c_const i j +
       Qabs_diff (nth (S i) (x :: xs) 0) (nth (S j) (y :: ys) 0))
      (msm_matrix_cell x xs y ys c_const i (S j) +
       c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0)
         (nth (S j) (y :: ys) 0))
      (msm_matrix_cell x xs y ys c_const (S i) j +
       c_func c_const (nth (S j) (y :: ys) 0) (nth (S i) (x :: xs) 0)
         (nth j (y :: ys) 0)).
Proof.
  intros x xs y ys c_const i j Hi Hj.
  unfold msm_matrix_cell, msm_matrix_nonempty.
  rewrite msm_compute_rows_matrix_row_succ.
  - change (nth (S j)
      (msm_compute_row (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0) y ys
         (nth i
            (msm_compute_rows_matrix xs x (y :: ys) y
               (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) [])
         (hd 0
            (nth i
               (msm_compute_rows_matrix xs x (y :: ys) y
                  (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) +
          c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0) y)
         c_const) 0 ==
      Qmin3
        (nth j
           (nth i
              (msm_compute_rows_matrix xs x (y :: ys) y
                 (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) 0 +
         Qabs_diff (nth (S i) (x :: xs) 0) (nth j ys 0))
        (nth (S j)
           (nth i
              (msm_compute_rows_matrix xs x (y :: ys) y
                 (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) 0 +
         c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0)
           (nth j ys 0))
        (nth j
           (msm_compute_row (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0) y ys
              (nth i
                 (msm_compute_rows_matrix xs x (y :: ys) y
                    (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) [])
              (hd 0
                 (nth i
                    (msm_compute_rows_matrix xs x (y :: ys) y
                       (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) +
               c_func c_const (nth (S i) (x :: xs) 0) (nth i (x :: xs) 0) y)
              c_const) 0 +
         c_func c_const (nth j ys 0) (nth (S i) (x :: xs) 0)
           (nth j (y :: ys) 0))).
    apply msm_compute_row_nth_succ_eq.
    + simpl in Hj. lia.
    + assert (Hrow_len :
        length
          (nth i
             (msm_compute_rows_matrix xs x (y :: ys) y
                (msm_init_row x y ys (Qabs_diff x y) c_const) c_const) []) =
        S (length ys)).
      { apply msm_compute_rows_matrix_row_length.
        - apply msm_init_row_length.
        - rewrite msm_compute_rows_matrix_length. simpl in Hi. lia. }
      rewrite Hrow_len. simpl in Hj. lia.
  - rewrite msm_compute_rows_matrix_length. simpl in Hi. exact Hi.
Qed.

Lemma msm_distance_matrix_cell_last :
  forall x xs y ys cfg,
    msm_distance (x :: xs) (y :: ys) cfg ==
    msm_matrix_cell x xs y ys (msm_c cfg) (length xs) (length ys).
Proof.
  intros x xs y ys cfg.
  unfold msm_matrix_cell, msm_matrix_nonempty.
  simpl.
  set (first_row := msm_init_row x y ys (Qabs_diff x y) (msm_c cfg)).
  set (rows := msm_compute_rows_matrix xs x (y :: ys) y first_row (msm_c cfg)).
  set (final_row := msm_compute_rows xs x (y :: ys) y first_row (msm_c cfg)).
  assert (Hrows_len : length rows = S (length xs)).
  { subst rows. apply msm_compute_rows_matrix_length. }
  assert (Hnth_row : nth (length xs) rows [] = final_row).
  { rewrite <- (last_eq_nth_by_length_any (list Q) rows [] (length xs) Hrows_len).
    subst rows final_row.
    apply msm_compute_rows_matrix_last. }
  rewrite Hnth_row.
  assert (Hfirst_len : length first_row = S (length ys)).
  { subst first_row. apply msm_init_row_length. }
  assert (Hfinal_len : length final_row = S (length ys)).
  { subst final_row. apply msm_compute_rows_length. exact Hfirst_len. }
  rewrite <- (last_eq_nth_by_length final_row 0 (length ys) Hfinal_len).
  reflexivity.
Qed.

Lemma msm_matrix_cell_sym_measure :
  forall n x xs y ys c_const i j,
    (i + j <= n)%nat ->
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const i j ==
    msm_matrix_cell y ys x xs c_const j i.
Proof.
  induction n as [|n IH]; intros x xs y ys c_const i j Hmeasure Hi Hj.
  - assert (Hi0 : i = 0%nat) by lia.
    assert (Hj0 : j = 0%nat) by lia.
    subst i j.
    rewrite !msm_matrix_cell_base.
    apply Qabs_diff_symmetric.
  - destruct i as [|i']; destruct j as [|j'].
    + rewrite !msm_matrix_cell_base.
      apply Qabs_diff_symmetric.
    + rewrite msm_matrix_cell_first_row by exact Hj.
      rewrite msm_matrix_cell_first_col by exact Hj.
      rewrite (IH x xs y ys c_const 0%nat j') by (simpl in *; lia).
      rewrite c_func_symm_bc.
      reflexivity.
    + rewrite msm_matrix_cell_first_col by exact Hi.
      rewrite msm_matrix_cell_first_row by exact Hi.
      rewrite (IH x xs y ys c_const i' 0%nat) by (simpl in *; lia).
      rewrite c_func_symm_bc.
      reflexivity.
    + rewrite msm_matrix_cell_interior by assumption.
      rewrite msm_matrix_cell_interior by assumption.
      eapply Qeq_trans.
      * apply Qmin3_wd.
        -- rewrite (IH x xs y ys c_const i' j') by (simpl in *; lia).
           rewrite Qabs_diff_symmetric.
           reflexivity.
        -- rewrite (IH x xs y ys c_const i' (S j')) by (simpl in *; lia).
           rewrite c_func_symm_bc.
           reflexivity.
        -- rewrite (IH x xs y ys c_const (S i') j') by (simpl in *; lia).
           rewrite c_func_symm_bc.
           reflexivity.
      * apply Qmin3_swap_bc.
Qed.

Lemma msm_matrix_cell_sym :
  forall x xs y ys c_const i j,
    (i < length (x :: xs))%nat ->
    (j < length (y :: ys))%nat ->
    msm_matrix_cell x xs y ys c_const i j ==
    msm_matrix_cell y ys x xs c_const j i.
Proof.
  intros x xs y ys c_const i j Hi Hj.
  apply (msm_matrix_cell_sym_measure (i + j) x xs y ys c_const i j);
    [lia | exact Hi | exact Hj].
Qed.

(** * Main Symmetry Theorem *)

(** Helper: length symmetry *)
Lemma length_sym : forall {A : Type} (l : list A),
  length l = length l.
Proof.
  reflexivity.
Qed.

(** Helper: multiplication is commutative for Qeq *)
Lemma Qmult_comm_eq : forall a b, a * b == b * a.
Proof.
  intros a b. ring.
Qed.

(** MSM symmetry follows from the transposed-cell invariant above. *)
Theorem msm_symmetric : forall X Y cfg,
  msm_distance X Y cfg == msm_distance Y X cfg.
Proof.
  intros X Y cfg.
  destruct X as [|x xs]; destruct Y as [|y ys].
  - (* [], [] *)
    simpl. reflexivity.
  - (* [], y::ys *)
    simpl.
    reflexivity.
  - (* x::xs, [] *)
    simpl.
    reflexivity.
  - rewrite (msm_distance_matrix_cell_last x xs y ys cfg).
    rewrite (msm_distance_matrix_cell_last y ys x xs cfg).
    apply msm_matrix_cell_sym; simpl; lia.
Qed.
