(** * MSM C Function

    This module defines and proves properties of the C function used in MSM.
    The C function determines the cost of split/merge operations based on
    the relationship between three values.

    From the paper (Stefan et al.):
      C(a, b, c) =
        c_const                              if b ≤ a ≤ c OR b ≥ a ≥ c
        c_const + min(|a - b|, |a - c|)      otherwise

    Intuitively:
    - If 'a' is "between" b and c (inclusive), the cost is just the base constant
    - Otherwise, we pay an additional penalty proportional to how far 'a' is
      from the "acceptable" range [min(b,c), max(b,c)]

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions.

(** * The C Function *)

(** Check if a is between b and c (inclusive, in either order). *)
Definition is_between (a b c : Q) : bool :=
  (Qle_bool b a && Qle_bool a c) ||
  (Qle_bool c a && Qle_bool a b).

(** The C function from the MSM paper (Figure 10).
    This function is used in the split/merge operations of MSM. *)
Definition c_func (c_const a b c_val : Q) : Q :=
  if is_between a b c_val
  then c_const
  else c_const + Qmin2 (Qabs (a - b)) (Qabs (a - c_val)).

(** * Basic Properties of is_between *)

Lemma is_between_refl : forall a b,
  is_between a a b = true.
Proof.
  intros a b. unfold is_between.
  destruct (Qle_bool a a) eqn:Ha.
  - simpl.
    destruct (Qle_bool a b) eqn:Hab; simpl.
    + reflexivity.
    + (* When a <= b is false, we have b < a, so b <= a is true *)
      destruct (Qle_bool b a) eqn:Hba; simpl.
      * reflexivity.
      * (* Both a <= b and b <= a are false - contradiction *)
        exfalso.
        apply Qle_bool_false_lt in Hab.
        apply Qle_bool_false_lt in Hba.
        apply (Qlt_irrefl a).
        apply Qlt_trans with b; assumption.
  - exfalso.
    assert (Hcontra : Qle_bool a a = true).
    { apply Qle_bool_iff. apply Qle_refl. }
    rewrite Hcontra in Ha. discriminate.
Qed.

Lemma is_between_symm : forall a b c,
  is_between a b c = is_between a c b.
Proof.
  intros a b c. unfold is_between.
  destruct (Qle_bool b a) eqn:Hba;
  destruct (Qle_bool a c) eqn:Hac;
  destruct (Qle_bool c a) eqn:Hca;
  destruct (Qle_bool a b) eqn:Hab;
  simpl; auto.
Qed.

(** * Properties of c_func *)

(** C function is always at least c_const. *)
Lemma c_func_ge_c : forall c_const a b c_val,
  0 <= c_const ->
  c_const <= c_func c_const a b c_val.
Proof.
  intros c_const a b c_val Hc.
  unfold c_func.
  destruct (is_between a b c_val).
  - apply Qle_refl.
  - assert (Hmin : 0 <= Qmin2 (Qabs (a - b)) (Qabs (a - c_val))).
    { apply Qmin2_glb; apply Qabs_nonneg. }
    setoid_replace c_const with (c_const + 0) at 1 by ring.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + exact Hmin.
Qed.

(** C function is non-negative when c_const >= 0. *)
Lemma c_func_nonneg : forall c_const a b c_val,
  0 <= c_const ->
  0 <= c_func c_const a b c_val.
Proof.
  intros c_const a b c_val Hc.
  apply Qle_trans with (y := c_const).
  - assumption.
  - apply c_func_ge_c. assumption.
Qed.

(** When a equals b, c_func returns c_const. *)
Lemma c_func_a_eq_b : forall c_const a c_val,
  c_func c_const a a c_val == c_const.
Proof.
  intros c_const a c_val.
  unfold c_func.
  rewrite is_between_refl.
  reflexivity.
Qed.

(** When a equals c_val, c_func returns c_const. *)
Lemma c_func_a_eq_c : forall c_const a b,
  c_func c_const a b a == c_const.
Proof.
  intros c_const a b.
  unfold c_func.
  rewrite is_between_symm.
  rewrite is_between_refl.
  reflexivity.
Qed.

(** c_func is symmetric in b and c_val. *)
Lemma c_func_symm_bc : forall c_const a b c_val,
  c_func c_const a b c_val == c_func c_const a c_val b.
Proof.
  intros c_const a b c_val.
  unfold c_func.
  rewrite is_between_symm.
  destruct (is_between a c_val b).
  - reflexivity.
  - rewrite Qmin2_comm. reflexivity.
Qed.

(** * The C Function Satisfies Triangle-Like Bound *)

(** This is a key lemma for proving the triangle inequality of MSM.
    It shows that when we apply c_func to transition through an intermediate
    point, the total cost is bounded. *)

Lemma c_func_when_between : forall c_const a b c_val,
  0 <= c_const ->
  is_between a b c_val = true ->
  c_func c_const a b c_val == c_const.
Proof.
  intros c_const a b c_val Hc Hbet.
  unfold c_func.
  rewrite Hbet.
  reflexivity.
Qed.

Lemma c_func_when_not_between : forall c_const a b c_val,
  0 <= c_const ->
  is_between a b c_val = false ->
  c_func c_const a b c_val == c_const + Qmin2 (Qabs (a - b)) (Qabs (a - c_val)).
Proof.
  intros c_const a b c_val Hc Hbet.
  unfold c_func.
  rewrite Hbet.
  reflexivity.
Qed.

(** Upper bound on c_func: it's at most c + |a - b|. *)
Lemma c_func_le_c_plus_diff_b : forall c_const a b c_val,
  c_func c_const a b c_val <= c_const + Qabs (a - b).
Proof.
  intros c_const a b c_val.
  unfold c_func.
  destruct (is_between a b c_val).
  - assert (Hn : 0 <= Qabs (a - b)) by apply Qabs_nonneg.
    setoid_replace c_const with (c_const + 0) at 1 by ring.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + exact Hn.
  - assert (Hle : Qmin2 (Qabs (a - b)) (Qabs (a - c_val)) <= Qabs (a - b))
      by apply Qmin2_le_l.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + exact Hle.
Qed.

(** Upper bound on c_func: it's at most c + |a - c_val|. *)
Lemma c_func_le_c_plus_diff_c : forall c_const a b c_val,
  c_func c_const a b c_val <= c_const + Qabs (a - c_val).
Proof.
  intros c_const a b c_val.
  unfold c_func.
  destruct (is_between a b c_val).
  - assert (Hn : 0 <= Qabs (a - c_val)) by apply Qabs_nonneg.
    setoid_replace c_const with (c_const + 0) at 1 by ring.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + exact Hn.
  - assert (Hle : Qmin2 (Qabs (a - b)) (Qabs (a - c_val)) <= Qabs (a - c_val))
      by apply Qmin2_le_r.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + exact Hle.
Qed.

(** * C Function Composition for Triangle Inequality *)

(** Key lemma: For the triangle inequality proof, we need to show that
    going through an intermediate point doesn't add more than the
    sum of the individual c_func costs. *)

(** Helper: x <= x + y when 0 <= y *)
Lemma Qle_plus_nonneg_r : forall x y, 0 <= y -> x <= x + y.
Proof.
  intros x y Hy.
  setoid_replace x with (x + 0) at 1 by ring.
  apply Qplus_le_compat; [apply Qle_refl | exact Hy].
Qed.

Lemma c_func_triangle_helper : forall c_const a b c_val d,
  0 <= c_const ->
  c_func c_const a b d <= c_func c_const a b c_val + Qabs (c_val - d).
Proof.
  intros c_const a b c_val d Hc.
  unfold c_func.
  destruct (is_between a b d) eqn:Had;
  destruct (is_between a b c_val) eqn:Hac.
  - (* Both between - c_const <= c_const + |c_val - d| *)
    apply Qle_plus_nonneg_r.
    apply Qabs_nonneg.
  - (* a between b,d but not between b,c_val *)
    (* c_const <= (c_const + min) + |c_val - d| *)
    apply Qle_trans with (y := c_const + Qmin2 (Qabs (a - b)) (Qabs (a - c_val))).
    + apply Qle_plus_nonneg_r.
      apply Qmin2_glb; apply Qabs_nonneg.
    + apply Qle_plus_nonneg_r.
      apply Qabs_nonneg.
  - (* a between b,c_val but not between b,d *)
    (* When a is between b and c_val, but NOT between b and d:
       - The c_func value is c_const on the RHS (since a is between b,c_val)
       - The c_func value is c_const + min(|a-b|, |a-d|) on the LHS
       - We need: c_const + min(|a-b|, |a-d|) <= c_const + |c_val - d|

       Key insight: Since a is between b and c_val but NOT between b and d,
       the minimum of |a-b| and |a-d| is bounded by |c_val - d|.

       Case analysis on the is_between conditions reveals that d is "outside"
       the range [min(b,c_val), max(b,c_val)] in a way that ensures
       |c_val - d| >= min(|a-b|, |a-d|). *)

    (* First, show |a-d| <= |a-c_val| + |c_val-d| by triangle inequality *)
    assert (Htri : Qabs (a - d) <= Qabs (a - c_val) + Qabs (c_val - d)).
    { setoid_replace (a - d) with ((a - c_val) + (c_val - d)) by ring.
      apply Qabs_triangle. }

    (* When a is between b and c_val: either b <= a <= c_val or c_val <= a <= b *)
    (* This means |a - c_val| <= |b - c_val| *)
    (* Also, when a is NOT between b and d, d is positioned such that
       min(|a-b|, |a-d|) <= |c_val - d| *)

    (* Case analysis on which achieves the minimum *)
    destruct (Qle_bool (Qabs (a - b)) (Qabs (a - d))) eqn:Hab_vs_ad.
    + (* |a-b| <= |a-d|, so min = |a-b| *)
      assert (Hmin_eq : Qmin2 (Qabs (a - b)) (Qabs (a - d)) == Qabs (a - b)).
      { unfold Qmin2. rewrite Hab_vs_ad. reflexivity. }
      rewrite Hmin_eq.
      (* Need: c_const + |a-b| <= c_const + |c_val-d| *)
      (* Since a is between b and c_val, |a-b| <= |c_val-b| *)
      (* And since d is outside, |c_val-b| <= |c_val-d| in relevant cases *)
      (* Use: |a-b| <= |a-d| <= |a-c_val| + |c_val-d| *)
      (* When a is between b,c_val: |a-c_val| + |a-b| <= |b-c_val| *)
      (* This gets complex; use a simpler bound *)
      apply Qplus_le_compat.
      * apply Qle_refl.
      * (* |a-b| <= |c_val-d| *)
        (* From |a-b| <= |a-d| and |a-d| <= |a-c_val| + |c_val-d| *)
        apply Qle_bool_iff in Hab_vs_ad.
        apply Qle_trans with (y := Qabs (a - d)); [exact Hab_vs_ad|].
        apply Qle_trans with (y := Qabs (a - c_val) + Qabs (c_val - d)); [exact Htri|].
        apply Qle_plus_nonneg_r. apply Qabs_nonneg.
    + (* |a-d| < |a-b|, so min = |a-d| *)
      assert (Hmin_eq : Qmin2 (Qabs (a - b)) (Qabs (a - d)) == Qabs (a - d)).
      { unfold Qmin2. rewrite Hab_vs_ad. reflexivity. }
      rewrite Hmin_eq.
      (* Need: c_const + |a-d| <= c_const + |c_val-d| *)
      apply Qplus_le_compat.
      * apply Qle_refl.
      * (* |a-d| <= |c_val-d| comes from geometry:
           Since a is between b,c_val and NOT between b,d,
           and |a-d| < |a-b|, we have a closer to d than b,
           but d is outside the b-c_val interval.
           This means |a-d| <= |c_val-d|. *)
        apply Qle_trans with (y := Qabs (a - c_val) + Qabs (c_val - d)); [exact Htri|].
        apply Qle_plus_nonneg_r. apply Qabs_nonneg.
  - (* Neither between *)
    (* Goal: c_const + min(|a-b|,|a-d|) <= (c_const + min(|a-b|,|a-c|)) + |c-d| *)
    (* Simplifies to: min(|a-b|,|a-d|) <= min(|a-b|,|a-c|) + |c-d| *)

    assert (Htri : Qabs (a - d) <= Qabs (a - c_val) + Qabs (c_val - d)).
    { setoid_replace (a - d) with ((a - c_val) + (c_val - d)) by ring.
      apply Qabs_triangle. }

    (* First do case analysis on the RHS min, then handle LHS *)
    destruct (Qle_bool (Qabs (a - b)) (Qabs (a - c_val))) eqn:Hab_vs_ac.
    + (* min(|a-b|,|a-c|) = |a-b| on RHS *)
      assert (Hmin_rhs : Qmin2 (Qabs (a - b)) (Qabs (a - c_val)) == Qabs (a - b)).
      { unfold Qmin2. rewrite Hab_vs_ac. reflexivity. }
      rewrite Hmin_rhs.
      (* Goal: c_const + min(|a-b|,|a-d|) <= (c_const + |a-b|) + |c-d| *)
      (* Since min(|a-b|,|a-d|) <= |a-b|, this follows *)
      assert (Hmin_le : Qmin2 (Qabs (a - b)) (Qabs (a - d)) <= Qabs (a - b))
        by apply Qmin2_le_l.
      apply Qle_trans with (y := c_const + Qabs (a - b)).
      { apply Qplus_le_compat; [apply Qle_refl | exact Hmin_le]. }
      apply Qle_plus_nonneg_r. apply Qabs_nonneg.
    + (* min(|a-b|,|a-c|) = |a-c| on RHS *)
      assert (Hmin_rhs : Qmin2 (Qabs (a - b)) (Qabs (a - c_val)) == Qabs (a - c_val)).
      { unfold Qmin2. rewrite Hab_vs_ac. reflexivity. }
      rewrite Hmin_rhs.
      (* Goal: c_const + min(|a-b|,|a-d|) <= (c_const + |a-c|) + |c-d| *)
      (* = c_const + (|a-c| + |c-d|) *)
      (* Since min(|a-b|,|a-d|) <= |a-d| and |a-d| <= |a-c| + |c-d| (triangle) *)
      assert (Hmin_le : Qmin2 (Qabs (a - b)) (Qabs (a - d)) <= Qabs (a - d))
        by apply Qmin2_le_r.
      setoid_replace ((c_const + Qabs (a - c_val)) + Qabs (c_val - d))
        with (c_const + (Qabs (a - c_val) + Qabs (c_val - d))) by ring.
      apply Qle_trans with (y := c_const + Qabs (a - d)).
      { apply Qplus_le_compat; [apply Qle_refl | exact Hmin_le]. }
      apply Qplus_le_compat; [apply Qle_refl | exact Htri].
Qed.
