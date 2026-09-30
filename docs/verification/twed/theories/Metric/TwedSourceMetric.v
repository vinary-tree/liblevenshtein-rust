(** * The ground product metric for physical-time TWED

    This file proves only the sample-space metric used by the source TWED
    theorem. The metricity of the entire sequence recurrence, in particular
    its triangle inequality, requires a separate alignment/DP proof. *)

From Stdlib Require Import Reals Lra Ring.
Open Scope R_scope.

Definition timed_sample := (R * R)%type.

Definition timed_sample_distance (nu : R)
    (first second : timed_sample) : R :=
  Rabs (fst first - fst second) +
  nu * Rabs (snd first - snd second).

Lemma absolute_difference_triangle : forall first middle last,
  Rabs (first - last) <=
  Rabs (first - middle) + Rabs (middle - last).
Proof.
  intros first middle last.
  replace (first - last) with
    ((first - middle) + (middle - last)) by ring.
  apply Rabs_triang.
Qed.

Theorem timed_sample_distance_nonnegative : forall nu first second,
  0 <= nu -> 0 <= timed_sample_distance nu first second.
Proof.
  intros nu [first_value first_time] [second_value second_time] Hnu.
  unfold timed_sample_distance; simpl.
  pose proof (Rabs_pos (first_value - second_value)).
  pose proof (Rabs_pos (first_time - second_time)).
  nra.
Qed.

Theorem timed_sample_distance_reflexive : forall nu point,
  timed_sample_distance nu point point = 0.
Proof.
  intros nu [value time].
  unfold timed_sample_distance; simpl.
  replace (value - value) with 0 by ring.
  replace (time - time) with 0 by ring.
  rewrite Rabs_R0; ring.
Qed.

Theorem timed_sample_distance_symmetric : forall nu first second,
  timed_sample_distance nu first second =
  timed_sample_distance nu second first.
Proof.
  intros nu [first_value first_time] [second_value second_time].
  unfold timed_sample_distance; simpl.
  replace (second_value - first_value) with
    (-(first_value - second_value)) by ring.
  replace (second_time - first_time) with
    (-(first_time - second_time)) by ring.
  now rewrite !Rabs_Ropp.
Qed.

Theorem timed_sample_distance_separates : forall nu first second,
  0 < nu ->
  timed_sample_distance nu first second = 0 -> first = second.
Proof.
  intros nu [first_value first_time] [second_value second_time]
    Hnu Hdistance.
  unfold timed_sample_distance in Hdistance; simpl in Hdistance.
  pose proof (Rabs_pos (first_value - second_value)) as Hvalue_nonnegative.
  pose proof (Rabs_pos (first_time - second_time)) as Htime_nonnegative.
  assert (Hvalue_zero : Rabs (first_value - second_value) = 0) by nra.
  assert (Htime_zero : Rabs (first_time - second_time) = 0) by nra.
  assert (Hvalue_difference : first_value - second_value = 0).
  { destruct (Req_dec (first_value - second_value) 0) as [Hequal | Hnot].
    - exact Hequal.
    - exfalso; apply (Rabs_no_R0 _ Hnot); exact Hvalue_zero. }
  assert (Htime_difference : first_time - second_time = 0).
  { destruct (Req_dec (first_time - second_time) 0) as [Hequal | Hnot].
    - exact Hequal.
    - exfalso; apply (Rabs_no_R0 _ Hnot); exact Htime_zero. }
  f_equal; lra.
Qed.

Theorem timed_sample_distance_triangle : forall nu first middle last,
  0 <= nu ->
  timed_sample_distance nu first last <=
  timed_sample_distance nu first middle +
  timed_sample_distance nu middle last.
Proof.
  intros nu [first_value first_time] [middle_value middle_time]
    [last_value last_time] Hnu.
  unfold timed_sample_distance; simpl.
  pose proof (absolute_difference_triangle
    first_value middle_value last_value) as Hvalue.
  pose proof (absolute_difference_triangle
    first_time middle_time last_time) as Htime.
  nra.
Qed.

Theorem positive_stiffness_gives_timed_sample_metric : forall nu,
  0 < nu ->
  (forall first second, 0 <= timed_sample_distance nu first second) /\
  (forall point, timed_sample_distance nu point point = 0) /\
  (forall first second,
    timed_sample_distance nu first second =
    timed_sample_distance nu second first) /\
  (forall first second,
    timed_sample_distance nu first second = 0 <-> first = second) /\
  (forall first middle last,
    timed_sample_distance nu first last <=
    timed_sample_distance nu first middle +
    timed_sample_distance nu middle last).
Proof.
  intros nu Hnu.
  repeat split.
  - intros; apply timed_sample_distance_nonnegative; lra.
  - apply timed_sample_distance_reflexive.
  - apply timed_sample_distance_symmetric.
  - apply timed_sample_distance_separates; exact Hnu.
  - intro Hequal; subst; apply timed_sample_distance_reflexive.
  - intros; apply timed_sample_distance_triangle; lra.
Qed.
