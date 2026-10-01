(** * Counterexample to the 2008 report's printed TWED boundary

    Marteau, "Time Warp Edit Distance", technical report VALORIA.2008.1V5,
    arXiv:0802.3522v5, Equation (1) and its initialization (PDF pp. 3-4):
    https://arxiv.org/pdf/0802.3522v5#page=3

    The printed one-empty axes sum the point distances without the gap
    penalty, although each interior deletion or insertion adds lambda.
    This module computes the resulting 1-by-2 grid for a strict-origin
    counterexample to the printed Proposition 1. It does not model the
    library's cumulative axes, which do include lambda. All costs here
    are exact natural numbers, so no binary64 claim is involved. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Definition timed_point : Type := (nat * nat)%type.
Definition point_value (point : timed_point) : nat := fst point.
Definition point_time (point : timed_point) : nat := snd point.

(** Absolute scalar difference plus unit-stiffness absolute time difference. *)
Definition point_distance (left right : timed_point) : nat :=
  (Nat.max (point_value left) (point_value right) -
   Nat.min (point_value left) (point_value right)) +
  (Nat.max (point_time left) (point_time right) -
   Nat.min (point_time left) (point_time right)).

Definition origin : timed_point := (0, 0).
Definition gap_penalty : nat := 3.

(** The 2008 printed one-empty boundary: no gap penalty in this sum. *)
Fixpoint printed_boundary_from
    (previous : timed_point) (series : list timed_point) : nat :=
  match series with
  | [] => 0
  | current :: tail =>
      point_distance current previous + printed_boundary_from current tail
  end.

(** The same report's interior three-way recurrence. [north] is the
    (i-1,j) cell, [west] the (i,j-1) cell, and [northwest] the (i-1,j-1)
    cell. The two edit alternatives add lambda; the match alternative
    compares both current and predecessor point pairs. *)
Definition printed_interior_cell
    (north west northwest : nat)
    (left_current left_previous right_current right_previous : timed_point)
    : nat :=
  Nat.min
    (north + point_distance left_current left_previous + gap_penalty)
    (Nat.min
      (west + point_distance right_current right_previous + gap_penalty)
      (northwest + point_distance left_current right_current +
       point_distance left_previous right_previous)).

Definition a1 : timed_point := (0, 1).
Definition c1 : timed_point := (1, 1).
Definition c2 : timed_point := (1, 2).

Definition series_a : list timed_point := [a1].
Definition series_b : list timed_point := [].
Definition series_c : list timed_point := [c1; c2].

Fixpoint strictly_increasing_times (series : list timed_point) : Prop :=
  match series with
  | [] => True
  | first :: rest =>
      match rest with
      | [] => True
      | second :: _ =>
          point_time first < point_time second /\
          strictly_increasing_times rest
      end
  end.

Definition strict_after_origin (series : list timed_point) : Prop :=
  match series with
  | [] => True
  | first :: _ =>
      point_time origin < point_time first /\
      strictly_increasing_times series
  end.

Theorem witnesses_satisfy_strict_origin :
  strict_after_origin series_a /\
  strict_after_origin series_b /\
  strict_after_origin series_c.
Proof.
  unfold strict_after_origin,
    series_a, series_b, series_c, a1, c1, c2, origin, point_time.
  simpl. repeat split; lia.
Qed.

(** The exact 1-by-2 dynamic-programming grid for A versus C. *)
Definition d00 : nat := 0.
Definition d01 : nat := printed_boundary_from origin [c1].
Definition d02 : nat := printed_boundary_from origin series_c.
Definition d10 : nat := printed_boundary_from origin series_a.
Definition d11 : nat :=
  printed_interior_cell d01 d10 d00 a1 origin c1 origin.
Definition d12 : nat :=
  printed_interior_cell d02 d11 d01 a1 origin c2 c1.

Definition distance_a_b : nat := printed_boundary_from origin series_a.
Definition distance_b_c : nat := printed_boundary_from origin series_c.
Definition distance_a_c : nat := d12.

Theorem printed_grid_values :
  (d00, d01, d02, d10, d11, d12) = (0, 2, 3, 1, 1, 5).
Proof. reflexivity. Qed.

Theorem printed_pair_distances :
  distance_a_b = 1 /\ distance_b_c = 3 /\ distance_a_c = 5.
Proof. repeat split; reflexivity. Qed.

Theorem printed_2008_triangle_failure :
  distance_a_c > distance_a_b + distance_b_c.
Proof. vm_compute; lia. Qed.

Corollary printed_2008_is_not_metric_on_strict_origin_series :
  ~ (distance_a_c <= distance_a_b + distance_b_c).
Proof. pose proof printed_2008_triangle_failure; lia. Qed.
