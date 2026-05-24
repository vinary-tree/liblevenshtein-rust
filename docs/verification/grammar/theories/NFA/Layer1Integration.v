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

(** Layer 1 completeness with phonetic flag *)
Theorem layer1_phonetic_completeness : forall
  (contracts : NFACompletenessEvidence)
  max_dist target input use_phonetic edits,
  use_phonetic = true ->
  Forall (fun op => In op phonetic_ops_phase1) edits ->
  apply_edit_sequence target edits = input ->
  edit_sequence_cost edits <= max_dist ->
  accepts (layer1_with_phonetic max_dist use_phonetic) target input = true.
Proof.
  intros contracts max_dist target input use_phonetic edits Hphon Hall Happly Hcost.
  unfold layer1_with_phonetic. rewrite Hphon.
  apply (phonetic_completeness contracts) with (edits := edits); auto.
  apply Forall_impl with (P := fun op => In op phonetic_ops_phase1);
    [| exact Hall].
  intros op Hin. unfold phonetic_edit. apply phonetic_cost_less_than_standard.
  exact Hin.
Qed.

(** Layer 1 soundness with phonetic flag *)
Theorem layer1_phonetic_soundness : forall
  (contracts : NFASoundnessEvidence)
  max_dist target input use_phonetic,
  accepts (layer1_with_phonetic max_dist use_phonetic) target input = true ->
  exists edits, edit_sequence_cost edits <= max_dist.
Proof.
  intros contracts max_dist target input use_phonetic Hacc.
  assert (Hwf : wf_automaton (layer1_with_phonetic max_dist use_phonetic)).
  { unfold layer1_with_phonetic.
    destruct use_phonetic.
    - apply phonetic_automaton_wf.
    - apply standard_automaton_wf_c. }
  destruct (nfa_soundness contracts
              (layer1_with_phonetic max_dist use_phonetic)
              target input Hwf Hacc) as [edits [_ [_ Hcost]]].
  exists edits.
  unfold layer1_with_phonetic in Hcost.
  destruct use_phonetic; simpl in Hcost; exact Hcost.
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
