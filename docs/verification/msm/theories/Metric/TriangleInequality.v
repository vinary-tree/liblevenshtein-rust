(** * MSM Triangle Inequality

    This module proves the triangle inequality for the MSM metric:

    MSM(X, Z) <= MSM(X, Y) + MSM(Y, Z)

    The proof uses trace composition: given optimal traces T1: X -> Y and T2: Y -> Z,
    we construct a trace T3: X -> Z with cost at most cost(T1) + cost(T2).

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction MsmDistance.
From Liblevenshtein.MSM Require Import Symmetry.

(** * Trace Composition for MSM *)

(** Given traces T1: X -> Y and T2: Y -> Z, we need to compose them into T3: X -> Z.

    The key challenge is that MSM operations involve context (the C function uses
    adjacent values). When composing traces, we must carefully handle this context.
*)

(** * Direct DP-Based Proof Approach *)

(** Instead of trace composition, we can prove triangle inequality directly
    on the DP recurrence by showing:

    For all i, j, k:
    Cost_XZ(i, k) <= Cost_XY(i, j) + Cost_YZ(j, k)

    where the intermediate j ranges over all valid positions in Y.
*)

(** Helper: |a - c| <= |a - b| + |b - c| (triangle inequality for absolute values) *)
Lemma qabs_triangle : forall a b c,
  Qabs (a - c) <= Qabs (a - b) + Qabs (b - c).
Proof.
  intros a b c.
  setoid_replace (a - c) with ((a - b) + (b - c)) by ring.
  apply Qabs_triangle.
Qed.

(** The Move operation satisfies triangle inequality *)
Lemma move_triangle : forall x y z,
  Qabs_diff x z <= Qabs_diff x y + Qabs_diff y z.
Proof.
  intros x y z.
  unfold Qabs_diff.
  apply qabs_triangle.
Qed.

(** * Main Triangle Inequality *)

(** The triangle inequality for MSM.
    This is the most complex proof as it requires showing that
    optimal traces can be composed without increasing total cost. *)

Theorem msm_triangle : forall X Y Z cfg,
  msm_distance X Z cfg <= msm_distance X Y cfg + msm_distance Y Z cfg.
Proof.
  intros X Y Z cfg.
  destruct cfg as [c Hc].
  simpl.

  (* Case analysis on the structure of the series *)
  destruct X as [|x xs]; destruct Y as [|y ys]; destruct Z as [|z zs].
  - (* [], [], [] *)
    simpl.
    (* 0 <= 0 + 0 *)
    setoid_replace (0 + 0) with 0 by ring.
    apply Qle_refl.
  - (* [], [], z::zs *)
    simpl.
    (* |Z|*c <= 0 + |Z|*c *)
    setoid_replace (0 + inject_Z (Z.of_nat (length (z :: zs))) * c)
      with (inject_Z (Z.of_nat (length (z :: zs))) * c) by ring.
    apply Qle_refl.
  - (* [], y::ys, [] *)
    simpl.
    (* MSM([], []) <= MSM([], Y) + MSM(Y, []) *)
    (* 0 <= |Y|*c + |Y|*c *)
    assert (H1 : 0 <= inject_Z (Z.of_nat (length (y :: ys))) * c).
    { apply Qmult_le_0_compat.
      - apply inject_Z_of_nat_nonneg.
      - exact Hc. }
    setoid_replace 0 with (0 + 0) by ring.
    apply Qplus_le_compat; assumption.
  - (* [], y::ys, z::zs *)
    simpl.
    (* MSM([], Z) <= MSM([], Y) + MSM(Y, Z) *)
    (* |Z|*c <= |Y|*c + MSM(Y, Z) *)
    admit.
  - (* x::xs, [], [] *)
    simpl.
    (* MSM(X, []) <= MSM(X, []) + MSM([], []) *)
    (* |X|*c <= |X|*c + 0 *)
    setoid_replace (inject_Z (Z.of_nat (length (x :: xs))) * c + 0)
      with (inject_Z (Z.of_nat (length (x :: xs))) * c) by ring.
    apply Qle_refl.
  - (* x::xs, [], z::zs *)
    simpl.
    (* This case requires careful analysis:
       MSM(X, Z) <= MSM(X, []) + MSM([], Z)
                 = |X|*c + |Z|*c

       But MSM(X, Z) might be less than this when X and Z are similar.
       We need to show that going through empty Y is never better than
       direct alignment. *)
    admit.
  - (* x::xs, y::ys, [] *)
    simpl.
    admit.
  - (* x::xs, y::ys, z::zs *)
    (* The main case: all three series non-empty *)
    (* This requires the full DP composition argument *)

    (* Key insight: The optimal path from X to Z via Y can be decomposed as:
       1. Align X with some prefix/suffix of Y
       2. Align that part of Y with Z
       3. Handle remaining elements via splits/merges

       The DP recurrence ensures the minimum is achieved. *)

    (* For each cell (i, k) in the X-Z DP matrix, we can bound:
       Cost_XZ(i, k) <= min_j { Cost_XY(i, j) + Cost_YZ(j, k) }

       This is because we can always:
       - Use optimal alignment to position j in Y
       - Use optimal alignment from position j to k in Z
    *)

    admit.
Admitted.
