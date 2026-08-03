(** * IEEE-754 boundary for additive [WeightedCost]

    [CostMonoid.v] proves the exact algebra over non-negative mathematical
    reals with an explicit top element. This companion uses Flocq's binary
    floating-point model to expose the round-to-nearest-even error of one
    binary64 addition, then proves how two such rounded additions bound the
    difference between the two three-term parenthesizations. It deliberately
    does not assert false bitwise associativity for arbitrary [f64] values.
*)

From Stdlib Require Import Lra Psatz Reals ZArith.
From Flocq.Core Require Import Core.
From Flocq.Prop Require Import Relative.

Open Scope R_scope.

Definition binary64_round (value : R) : R :=
  round radix2 (FLT_exp (-1074) 53) ZnearestE value.

Definition binary64_relative_bound : R :=
  u_ro radix2 53 / (1 + u_ro radix2 53).

Definition binary64_absolute_bound : R :=
  / 2 * bpow radix2 (-1074).

(** Flocq's gradual-underflow theorem gives a relative component and a tiny
    absolute component for every rounded real value. The unbounded FLT model
    covers finite binary64 operations; Rust tests separately exclude overflow
    by requiring both evaluated parenthesizations to remain finite. *)
Theorem binary64_round_error_components : forall value,
  exists relative_error absolute_error,
    Rabs relative_error <= binary64_relative_bound /\
    Rabs absolute_error <= binary64_absolute_bound /\
    relative_error * absolute_error = 0 /\
    binary64_round value =
      value * (1 + relative_error) + absolute_error.
Proof.
  intros value.
  unfold binary64_round, binary64_relative_bound,
    binary64_absolute_bound.
  exact (relative_error_N_FLT'_ex radix2 (-1074) 53 ltac:(lia)
    (fun exponent => negb (Z.even exponent)) value).
Qed.

Theorem binary64_round_absolute_error : forall value,
  Rabs (binary64_round value - value) <=
  binary64_relative_bound * Rabs value + binary64_absolute_bound.
Proof.
  intros value.
  destruct (binary64_round_error_components value)
    as [relative [absolute [Hrelative [Habsolute [_ Hround]]]]].
  rewrite Hround.
  replace (value * (1 + relative) + absolute - value)
    with (value * relative + absolute) by ring.
  eapply Rle_trans; [apply Rabs_triang |].
  rewrite Rabs_mult.
  apply Rplus_le_compat.
  - rewrite (Rmult_comm binary64_relative_bound (Rabs value)).
    apply Rmult_le_compat_l; [apply Rabs_pos | exact Hrelative].
  - exact Habsolute.
Qed.

(** A backend-independent composition theorem. [relative] and [absolute]
    are the certified one-round bounds. The right-hand side is a symbolic
    forward-error envelope, so no decimal approximation of binary64 constants
    enters the trusted proof. *)
Theorem two_rounded_additions_reassociation_envelope : forall
    (rounder : R -> R) relative absolute a b c,
  0 <= relative ->
  0 <= absolute ->
  0 <= a -> 0 <= b -> 0 <= c ->
  (forall value,
      Rabs (rounder value - value) <= relative * Rabs value + absolute) ->
  Rabs
    (rounder (rounder (a + b) + c) -
     rounder (a + rounder (b + c))) <=
  (4 * relative + 2 * relative * relative) * Rmax 1 (a + b + c) +
  (2 * relative + 4) * absolute.
Proof.
  intros rounder relative absolute a b c
    Hrelative Habsolute Ha Hb Hc Hround.
  set (sum := a + b + c).
  set (left_pair := rounder (a + b)).
  set (right_pair := rounder (b + c)).
  set (left_total := rounder (left_pair + c)).
  set (right_total := rounder (a + right_pair)).
  assert (Hsum : 0 <= sum) by (unfold sum; lra).
  assert (Hab_sum : a + b <= sum) by (unfold sum; lra).
  assert (Hbc_sum : b + c <= sum) by (unfold sum; lra).
  assert (Hsum_max : sum <= Rmax 1 sum) by apply Rmax_r.
  assert (Hleft_pair :
      Rabs (left_pair - (a + b)) <= relative * (a + b) + absolute).
  { unfold left_pair.
    specialize (Hround (a + b)).
    assert (Hab : 0 <= a + b) by lra.
    rewrite (Rabs_pos_eq (a + b) Hab) in Hround; exact Hround. }
  assert (Hright_pair :
      Rabs (right_pair - (b + c)) <= relative * (b + c) + absolute).
  { unfold right_pair.
    specialize (Hround (b + c)).
    assert (Hbc : 0 <= b + c) by lra.
    rewrite (Rabs_pos_eq (b + c) Hbc) in Hround; exact Hround. }
  assert (Hleft_input :
      Rabs (left_pair + c) <=
      sum + relative * (a + b) + absolute).
  { replace (left_pair + c)
      with ((left_pair - (a + b)) + sum) by (unfold sum; ring).
    eapply Rle_trans; [apply Rabs_triang |].
    rewrite (Rabs_pos_eq sum Hsum); lra. }
  assert (Hright_input :
      Rabs (a + right_pair) <=
      sum + relative * (b + c) + absolute).
  { replace (a + right_pair)
      with (sum + (right_pair - (b + c))) by (unfold sum; ring).
    eapply Rle_trans; [apply Rabs_triang |].
    rewrite (Rabs_pos_eq sum Hsum); lra. }
  assert (Hleft_round :
      Rabs (left_total - (left_pair + c)) <=
      relative * (sum + relative * (a + b) + absolute) + absolute).
  { unfold left_total.
    eapply Rle_trans; [apply Hround |].
    nra. }
  assert (Hright_round :
      Rabs (right_total - (a + right_pair)) <=
      relative * (sum + relative * (b + c) + absolute) + absolute).
  { unfold right_total.
    eapply Rle_trans; [apply Hround |].
    nra. }
  assert (Hleft_exact :
      Rabs (left_total - sum) <=
      (2 * relative + relative * relative) * sum +
      (relative + 2) * absolute).
  { replace (left_total - sum)
      with ((left_total - (left_pair + c)) +
            (left_pair - (a + b))) by (unfold sum; ring).
    eapply Rle_trans; [apply Rabs_triang |].
    nra. }
  assert (Hright_exact :
      Rabs (right_total - sum) <=
      (2 * relative + relative * relative) * sum +
      (relative + 2) * absolute).
  { replace (right_total - sum)
      with ((right_total - (a + right_pair)) +
            (right_pair - (b + c))) by (unfold sum; ring).
    eapply Rle_trans; [apply Rabs_triang |].
    nra. }
  change (Rabs (left_total - right_total) <=
    (4 * relative + 2 * relative * relative) * Rmax 1 sum +
    (2 * relative + 4) * absolute).
  replace (left_total - right_total)
    with ((left_total - sum) + -(right_total - sum)) by ring.
  eapply Rle_trans; [apply Rabs_triang |].
  rewrite Rabs_Ropp.
  nra.
Qed.

Theorem binary64_three_term_reassociation_envelope : forall a b c,
  0 <= a -> 0 <= b -> 0 <= c ->
  Rabs
    (binary64_round (binary64_round (a + b) + c) -
     binary64_round (a + binary64_round (b + c))) <=
  (4 * binary64_relative_bound +
   2 * binary64_relative_bound * binary64_relative_bound) *
    Rmax 1 (a + b + c) +
  (2 * binary64_relative_bound + 4) * binary64_absolute_bound.
Proof.
  intros a b c Ha Hb Hc.
  apply two_rounded_additions_reassociation_envelope; try assumption.
  - unfold binary64_relative_bound.
    unfold Rdiv; apply Rmult_le_pos.
    + apply u_ro_pos.
    + apply Rlt_le; apply Rinv_0_lt_compat.
      pose proof (u_ro_pos radix2 53); lra.
  - unfold binary64_absolute_bound.
    apply Rmult_le_pos; [lra | apply bpow_ge_0].
  - apply binary64_round_absolute_error.
Qed.
