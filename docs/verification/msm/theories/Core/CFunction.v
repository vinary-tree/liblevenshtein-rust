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
    (* c_const + min(|a-b|, |a-d|) <= c_const + |c_val - d| *)
    assert (Hle1 : Qmin2 (Qabs (a - b)) (Qabs (a - d)) <= Qabs (a - d)) by apply Qmin2_le_r.
    assert (Htri : Qabs (a - d) <= Qabs (a - c_val) + Qabs (c_val - d)).
    { setoid_replace (a - d) with ((a - c_val) + (c_val - d)) by ring.
      apply Qabs_triangle. }
    apply Qle_trans with (y := c_const + Qabs (a - d)).
    + apply Qplus_le_compat; [apply Qle_refl | exact Hle1].
    + apply Qle_trans with (y := c_const + (Qabs (a - c_val) + Qabs (c_val - d))).
      * apply Qplus_le_compat; [apply Qle_refl | exact Htri].
      * (* c_const + (|a-c| + |c-d|) = (c_const + |a-c|) + |c-d|
           We need c_const + (|a-c| + |c-d|) <= c_const + |c-d|
           But that's not true! We have LHS >= RHS.
           This case needs different handling. *)
        (* Actually, the goal is:
           c_const + min <= c_const + |c-d|
           and we have |a-d| <= |a-c| + |c-d|
           We need to reconsider this case... *)
        admit. (* This case requires more careful analysis *)
  - (* Neither between *)
    (* Goal: c_const + min(|a-b|,|a-d|) <= (c_const + min(|a-b|,|a-c|)) + |c-d| *)
    assert (Hle1 : Qmin2 (Qabs (a - b)) (Qabs (a - d)) <= Qabs (a - d)) by apply Qmin2_le_r.
    assert (Htri : Qabs (a - d) <= Qabs (a - c_val) + Qabs (c_val - d)).
    { setoid_replace (a - d) with ((a - c_val) + (c_val - d)) by ring.
      apply Qabs_triangle. }
    assert (Hle2 : Qmin2 (Qabs (a - b)) (Qabs (a - c_val)) <= Qabs (a - c_val))
      by apply Qmin2_le_r.
    (* Transitivity chain:
       min(|a-b|,|a-d|) <= |a-d| <= |a-c| + |c-d|
       and min(|a-b|,|a-c|) + |c-d| >= |a-c| + |c-d| - (|a-c| - min)
       This is complex; use admit for now *)
    admit.
Admitted.
