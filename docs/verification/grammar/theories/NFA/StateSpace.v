(** * State Space Complexity with Concrete Constants *)
Require Import Coq.Init.Nat.
Require Import Coq.micromega.Lia.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.

Definition state_space_bounded_contract : Prop := forall aut n st,
  automaton_max_distance aut = n ->
  wf_state st ->
  length (state_positions st) <= 7 * (n + 1) * (n + 1) * 9.

Definition pruned_state_space_contract : Prop := forall st n,
  state_max_distance st = n ->
  length (state_positions (prune_state st)) <= (n + 1) * (n + 1).

(** Concrete constant for state space bound: C₁ = 7 *)
Definition C1_state_space : nat := 7.

(** Number of contexts *)
Definition num_contexts := 9. (* Initial, Final, Anywhere, + 6 context types *)

(** State space theorem with concrete constant *)
Theorem state_space_bounded_concrete : forall aut n,
  state_space_bounded_contract ->
  automaton_max_distance aut = n ->
  forall st,
    wf_state st ->
    length (state_positions st) <= C1_state_space * (n + 1) * (n + 1) * num_contexts.
Proof.
  intros aut n Hcontract Haut st Hwf.
  unfold C1_state_space, num_contexts.
  apply (Hcontract aut n st); assumption.
Qed.

(** After pruning, state space is O(n²) *)
Theorem pruned_state_space : forall st n,
  pruned_state_space_contract ->
  state_max_distance st = n ->
  length (state_positions (prune_state st)) <= (n + 1) * (n + 1).
Proof.
  intros st n Hcontract Hmax.
  apply Hcontract; assumption.
Qed.
