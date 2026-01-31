(** * Integration with Grammar Correction Layer 1 *)
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.QArith.QArith.
Import ListNotations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Completeness.
Require Import Liblevenshtein.Grammar.Verification.NFA.Soundness.

(** Extend Layer 1 with phonetic NFA *)
Definition layer1_with_phonetic (max_dist : nat) (use_phonetic : bool) :=
  if use_phonetic
  then phonetic_automaton max_dist
  else standard_automaton max_dist.

(** * Axioms for Layer 1 Integration *)

(** Layer 1 with phonetic is complete: any phonetic edit sequence within
    max_dist is accepted by the phonetic automaton. *)
Axiom layer1_phonetic_completeness_ax : forall max_dist target input,
  (exists edits,
    Forall (fun op => In op phonetic_ops_phase1) edits /\
    edit_sequence_cost edits <= max_dist) ->
  accepts (phonetic_automaton max_dist) target input = true.

(** Layer 1 is sound: acceptance implies an edit sequence exists. *)
Axiom layer1_phonetic_soundness_ax : forall max_dist target input use_phonetic,
  accepts (layer1_with_phonetic max_dist use_phonetic) target input = true ->
  exists edits, edit_sequence_cost edits <= max_dist.

(** Layer 1 completeness with phonetic flag *)
Theorem layer1_phonetic_completeness : forall max_dist target input use_phonetic,
  use_phonetic = true ->
  (exists edits,
    Forall (fun op => In op phonetic_ops_phase1) edits /\
    edit_sequence_cost edits <= max_dist) ->
  accepts (layer1_with_phonetic max_dist use_phonetic) target input = true.
Proof.
  intros max_dist target input use_phonetic Hphon Hedits.
  unfold layer1_with_phonetic. rewrite Hphon.
  apply layer1_phonetic_completeness_ax. assumption.
Qed.

(** Layer 1 soundness with phonetic flag *)
Theorem layer1_phonetic_soundness : forall max_dist target input use_phonetic,
  accepts (layer1_with_phonetic max_dist use_phonetic) target input = true ->
  exists edits, edit_sequence_cost edits <= max_dist.
Proof.
  intros max_dist target input use_phonetic Hacc.
  exact (layer1_phonetic_soundness_ax max_dist target input use_phonetic Hacc).
Qed.

(** Lattice construction from NFA states *)
Definition nfa_states_to_lattice (st : GeneralizedState) : nat :=
  length (state_positions st).

Theorem nfa_lattice_wf : forall aut target input,
  wf_automaton aut ->
  let final := run_automaton_from aut target input 0 
                 (initial_state (automaton_max_distance aut))
                 (String.length input + 1) in
  wf_state final.
Proof. intros. apply run_automaton_preserves_wf; auto. apply initial_state_wf. Qed.
