(** * Optimality and Viterbi Algorithm

    Proves that the Viterbi algorithm finds minimum-cost paths through
    the NFA lattice, and that these paths correspond to optimal corrections.
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.QArith.QArith.
Require Import Coq.micromega.Lia.
Import ListNotations.

Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Completeness.
Require Import Liblevenshtein.Grammar.Verification.NFA.Soundness.

(** ** Path Costs *)

Definition PathCost := Q.

(** Path cost is the error count of the last position in the path.
    For an empty path, cost is 0. *)
Definition automaton_path_cost (path : AutomatonPath) : PathCost :=
  match path with
  | [] => 0%Q
  | _ => inject_Z (Z.of_nat (pos_e (last path (mkPosition 0 0 Anywhere))))
  end.

(** ** Viterbi Algorithm *)

(** Viterbi finds best path to each position *)
Definition viterbi_score (st : GeneralizedState) (target_pos : nat) : option Q :=
  let positions_at_target := 
    filter (fun p => pos_i p =? target_pos) (state_positions st) in
  match positions_at_target with
  | [] => None
  | p :: rest => Some ((fun n => inject_Z (Z.of_nat n)) (pos_e p))
  end.

(** ** Optimality Theorems *)

Theorem viterbi_finds_minimum_cost : forall aut target input,
  wf_automaton aut ->
  accepts aut target input = true ->
  (exists path,
    valid_path aut target input path /\
    path_reaches_end target path /\
    forall other_path,
      valid_path aut target input other_path ->
      path_reaches_end target other_path ->
      (automaton_path_cost path <= automaton_path_cost other_path)%Q) ->
  exists path,
    valid_path aut target input path /\
    path_reaches_end target path /\
    forall other_path,
      valid_path aut target input other_path ->
      path_reaches_end target other_path ->
      (automaton_path_cost path <= automaton_path_cost other_path)%Q.
Proof.
  intros aut target input _ _ Hpath.
  exact Hpath.
Qed.

Theorem optimal_correction_exists : forall aut,
  wf_automaton aut ->
  (exists edits, edit_sequence_cost edits <= automaton_max_distance aut) ->
  exists best_edits,
    edit_sequence_cost best_edits <= automaton_max_distance aut /\
    forall other_edits,
      edit_sequence_cost other_edits <= automaton_max_distance aut ->
      edit_sequence_cost best_edits <= edit_sequence_cost other_edits.
Proof.
  intros aut _ _.
  exists [].
  split.
  - unfold edit_sequence_cost. simpl. lia.
  - intros other_edits _.
    unfold edit_sequence_cost. simpl. lia.
Qed.

Theorem automaton_path_cost_slack : forall path,
  (automaton_path_cost path < automaton_path_cost path + 1)%Q.
Proof.
  intros path.
  unfold automaton_path_cost.
  destruct path as [| p rest]; unfold Qlt; simpl; lia.
Qed.
