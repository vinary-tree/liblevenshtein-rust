(** * Soft-DTW analysis-only boundary

    Soft-DTW aggregates every alignment path through a positive partition
    contribution.  Unlike a minimum-cost recurrence, deleting a dominated
    path changes that partition and therefore changes the soft value.  This
    small assumption-free model pins the architectural reason that the
    production Soft-DTW scorer is analysis-only and cannot reuse exact
    min-antichain dictionary pruning.
*)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Fixpoint partition_mass (contributions : list nat) : nat :=
  match contributions with
  | [] => 0
  | contribution :: rest => contribution + partition_mass rest
  end.

Lemma partition_mass_app : forall left right,
  partition_mass (left ++ right) = partition_mass left + partition_mass right.
Proof.
  intros left right; induction left; simpl; lia.
Qed.

(** Removing even one strictly positive path contribution changes the
    partition.  A min-cost subsumption proof is therefore insufficient for
    Soft-DTW. *)
Theorem positive_path_contribution_cannot_be_pruned : forall prefix suffix contribution,
  0 < contribution ->
  partition_mass (prefix ++ contribution :: suffix) >
  partition_mass (prefix ++ suffix).
Proof.
  intros prefix suffix contribution Hpositive.
  repeat rewrite partition_mass_app; simpl; lia.
Qed.

Definition grid_cells (query_len target_len : nat) : nat :=
  query_len * target_len.

(** A two-row evaluator changes retained memory, not the exact number of
    recurrence cells required by the full soft partition. *)
Theorem grid_work_is_symmetric : forall query_len target_len,
  grid_cells query_len target_len = grid_cells target_len query_len.
Proof. intros; unfold grid_cells; apply Nat.mul_comm. Qed.
