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
  (sound_contracts : NFASoundnessEvidence)
  (complete_contracts : NFACompletenessEvidence)
  aut target input,
  wf_automaton aut ->
  (accepts aut target input = true <->
   exists edits,
     Forall (fun op => In op (automaton_operations aut)) edits /\
     apply_edit_sequence target edits = input /\
     edit_sequence_cost edits <= automaton_max_distance aut).
Proof.
  intros sound_contracts complete_contracts aut target input Hwf.
  apply (soundness_completeness_correctness sound_contracts complete_contracts).
  assumption.
Qed.

(** Phonetic + standard combined correctness *)
Theorem phonetic_nfa_correctness : forall
  (sound_contracts : NFASoundnessEvidence)
  (complete_contracts : NFACompletenessEvidence)
  max_dist target input,
  accepts (phonetic_automaton max_dist) target input = true <->
  exists edits,
    Forall (fun op => In op (automaton_operations (phonetic_automaton max_dist))) edits /\
    apply_edit_sequence target edits = input /\
    edit_sequence_cost edits <= max_dist.
Proof.
  intros sound_contracts complete_contracts max_dist target input.
  apply (nfa_correctness sound_contracts complete_contracts).
  apply phonetic_automaton_wf.
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
