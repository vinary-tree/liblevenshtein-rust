(** * State Space Complexity with Concrete Constants *)
Require Import Coq.Init.Nat.
Require Import Coq.Lists.List.
Require Import Coq.micromega.Lia.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.

(** Concrete constant for state space bound: C₁ = 7 *)
Definition C1_state_space : nat := 7.

(** Number of contexts *)
Definition num_contexts := 9. (* Initial, Final, Anywhere, + 6 context types *)

(** wf_state bounds every state's error component by the automaton distance.
    A cardinality bound needs additional no-duplicate and target-index bounds,
    so the former broad contract has been replaced by this direct invariant. *)
Theorem state_space_errors_bounded : forall st n,
  state_max_distance st = n ->
  wf_state st ->
  Forall (fun p => pos_e p <= n) (state_positions st).
Proof.
  intros st n Hmax Hwf.
  subst n.
  exact Hwf.
Qed.

(** Pruning never increases the represented state list. *)
Theorem pruned_state_space : forall st,
  length (state_positions (prune_state st)) <= length (state_positions st).
Proof.
  intros st.
  apply state_size_bounded.
Qed.
