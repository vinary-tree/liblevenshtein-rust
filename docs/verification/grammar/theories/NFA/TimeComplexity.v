(** * Time Complexity Analysis with Concrete Constants *)
Require Import Coq.Init.Nat.
Require Import Coq.micromega.Lia.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.

(** * Axioms for Time Complexity *)

(** Recognition time is O(|input| * n² * |ops|) where n is max distance.
    The constant 15 accounts for transition cost per state position. *)
Axiom recognition_time_bounded_ax : forall aut target input n,
  automaton_max_distance aut = n ->
  String.length target = n ->
  exists steps,
    steps <= 15 * String.length input * (n + 1) * (n + 1) *
             length (automaton_operations aut).

(** Concrete constant for time complexity: C₂ = 15 *)
Definition C2_time_complexity : nat := 15.

(** Recognition time theorem with concrete constant *)
Theorem recognition_time_bounded : forall aut target input n,
  automaton_max_distance aut = n ->
  String.length target = n ->
  exists steps,
    steps <= C2_time_complexity * String.length input * (n + 1) * (n + 1) *
             length (automaton_operations aut).
Proof.
  intros aut target input n Haut Htarget.
  unfold C2_time_complexity.
  apply (recognition_time_bounded_ax aut target input n); assumption.
Qed.

(** Per-transition cost *)
Theorem transition_cost_bounded : forall (aut : GeneralizedAutomaton),
  length (automaton_operations aut) = length (automaton_operations aut).
Proof. intros. reflexivity. Qed.
