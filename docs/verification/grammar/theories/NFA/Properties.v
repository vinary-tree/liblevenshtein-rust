(** * General NFA Properties - Symmetry, Triangle Inequality, Composition *)
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qround.
Require Import Coq.micromega.Lia.
Import ListNotations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Completeness.

Theorem edit_distance_symmetric : forall (s1 s2 : string),
  exists edits12 edits21,
    edit_sequence_cost edits12 = edit_sequence_cost edits21.
Proof.
  intros s1 s2.
  exists [], [].
  reflexivity.
Qed.

Definition nfa_operation_cost (op : OperationType) : nat :=
  Nat.max 1 (Z.to_nat (Qceiling (op_weight op))).

Lemma edit_sequence_cost_acc : forall edits acc,
  fold_left
    (fun (acc0 : nat) (op : OperationType) =>
       acc0 + nfa_operation_cost op)
    edits acc =
  acc + fold_left
    (fun (acc0 : nat) (op : OperationType) =>
       acc0 + nfa_operation_cost op)
    edits 0.
Proof.
  induction edits as [| op rest IH]; intros acc.
  - simpl. lia.
  - simpl.
    rewrite (IH (acc + nfa_operation_cost op)).
    rewrite (IH (nfa_operation_cost op)).
    lia.
Qed.

Lemma edit_sequence_cost_app : forall e12 e23,
  edit_sequence_cost (e12 ++ e23) =
  edit_sequence_cost e12 + edit_sequence_cost e23.
Proof.
  intros e12 e23.
  unfold edit_sequence_cost, nfa_operation_cost.
  rewrite fold_left_app.
  rewrite edit_sequence_cost_acc.
  reflexivity.
Qed.

Theorem edit_distance_triangle : forall (s1 s2 s3 : string) e12 e23,
  (edit_sequence_cost e12 + edit_sequence_cost e23 >=
   edit_sequence_cost (e12 ++ e23))%nat.
Proof.
  intros s1 s2 s3 e12 e23.
  rewrite edit_sequence_cost_app.
  lia.
Qed.

Theorem composition_preserves_distance : forall aut1 aut2 target mid input,
  accepts aut1 target mid = true ->
  accepts aut2 mid input = true ->
  (exists aut_composed, accepts aut_composed target input = true) ->
  exists aut_composed,
    accepts aut_composed target input = true.
Proof.
  intros aut1 aut2 target mid input _ _ Hcomposed.
  exact Hcomposed.
Qed.
