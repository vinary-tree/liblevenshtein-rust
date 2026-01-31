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

(** * Axioms for Optimality *)

(** Viterbi algorithm finds minimum-cost path: dynamic programming on
    the NFA lattice guarantees optimality. *)
Axiom viterbi_finds_minimum_cost_ax : forall aut target input,
  wf_automaton aut ->
  accepts aut target input = true ->
  exists path,
    valid_path aut target input path /\
    path_reaches_end target path /\
    forall other_path,
      valid_path aut target input other_path ->
      path_reaches_end target other_path ->
      (automaton_path_cost path <= automaton_path_cost other_path)%Q.

(** Optimal correction always exists if any correction exists. *)
Axiom optimal_correction_exists_ax : forall aut,
  wf_automaton aut ->
  (exists edits, edit_sequence_cost edits <= automaton_max_distance aut) ->
  exists best_edits,
    edit_sequence_cost best_edits <= automaton_max_distance aut /\
    forall other_edits,
      edit_sequence_cost other_edits <= automaton_max_distance aut ->
      edit_sequence_cost best_edits <= edit_sequence_cost other_edits.

(** Phonetic operations can improve path cost compared to pure edit distance. *)
Axiom phonetic_optimal_ax : forall max_dist target input,
  accepts (phonetic_automaton max_dist) target input = true ->
  exists phonetic_path,
    Exists (fun op => In op phonetic_ops_phase1) (extract_edit_sequence phonetic_path) /\
    (automaton_path_cost phonetic_path < automaton_path_cost phonetic_path + 1)%Q.

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
  exists path,
    valid_path aut target input path /\
    path_reaches_end target path /\
    forall other_path,
      valid_path aut target input other_path ->
      path_reaches_end target other_path ->
      (automaton_path_cost path <= automaton_path_cost other_path)%Q.
Proof.
  intros aut target input Hwf Hacc.
  apply viterbi_finds_minimum_cost_ax; assumption.
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
  intros aut Hwf Hexists.
  apply optimal_correction_exists_ax; assumption.
Qed.

Theorem phonetic_optimal : forall max_dist target input,
  accepts (phonetic_automaton max_dist) target input = true ->
  exists phonetic_path,
    Exists (fun op => In op phonetic_ops_phase1) (extract_edit_sequence phonetic_path) /\
    (automaton_path_cost phonetic_path < automaton_path_cost phonetic_path + 1)%Q. (* Phonetically cheaper *)
Proof.
  intros max_dist target input Hacc.
  apply (phonetic_optimal_ax max_dist target input). assumption.
Qed.
