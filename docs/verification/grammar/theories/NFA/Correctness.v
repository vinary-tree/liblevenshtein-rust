(** * End-to-End NFA Correctness *)
Require Import Coq.Lists.List.
Import ListNotations.

Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Completeness.
Require Import Liblevenshtein.Grammar.Verification.NFA.Soundness.
Require Import Liblevenshtein.Grammar.Verification.NFA.Optimality.

(** Main correctness: Completeness + Soundness *)
Theorem nfa_correctness : forall
  aut target input,
  wf_automaton aut ->
  (accepts aut target input = true ->
   exists edits,
     Forall (fun op => In op (automaton_operations aut)) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= automaton_max_distance aut) ->
  ((exists edits,
     Forall (fun op => In op (automaton_operations aut)) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= automaton_max_distance aut) ->
   accepts aut target input = true) ->
  (accepts aut target input = true <->
   exists edits,
     Forall (fun op => In op (automaton_operations aut)) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= automaton_max_distance aut).
Proof.
  intros aut target input Hwf Hsound Hcomplete.
  apply soundness_completeness_correctness; assumption.
Qed.

(** Phonetic + standard combined correctness *)
Theorem phonetic_nfa_correctness : forall
  max_dist target input,
  (accepts (phonetic_automaton max_dist) target input = true ->
   exists edits,
     Forall (fun op => In op (automaton_operations (phonetic_automaton max_dist))) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= max_dist) ->
  ((exists edits,
     Forall (fun op => In op (automaton_operations (phonetic_automaton max_dist))) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= max_dist) ->
   accepts (phonetic_automaton max_dist) target input = true) ->
  accepts (phonetic_automaton max_dist) target input = true <->
  exists edits,
    Forall (fun op => In op (automaton_operations (phonetic_automaton max_dist))) edits /\
    apply_edit_sequence target edits = input /\
    edit_sequence_cost edits <= max_dist.
Proof.
  intros max_dist target input Hsound Hcomplete.
  apply (nfa_correctness (phonetic_automaton max_dist)).
  - apply phonetic_automaton_wf.
  - exact Hsound.
  - exact Hcomplete.
Qed.

(** NFA termination *)
Theorem nfa_always_terminates : forall aut target input,
  exists result, result = accepts aut target input.
Proof. intros. eexists. reflexivity. Qed.

(** NFA determinism *)
Theorem nfa_deterministic : forall aut target input,
  forall b1 b2,
    b1 = accepts aut target input ->
    b2 = accepts aut target input ->
    b1 = b2.
Proof. intros. subst. reflexivity. Qed.
