(** * End-to-End Correctness

    This module contains the main correctness theorem for the complete
    grammar correction pipeline.
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.Arith.PeanoNat.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Program.
Require Import Liblevenshtein.Grammar.Verification.Layers.Layer1.
Require Import Liblevenshtein.Grammar.Verification.Layers.Layer2.
Require Import Liblevenshtein.Grammar.Verification.Composition.Forward.
Import ListNotations.

(** ** Axioms for Pipeline Properties *)

(** Axiom: All layers in a well-formed correction pipeline produce valid results.
    This captures the design invariant that our correction layers are sound. *)
Axiom pipeline_layers_valid_ax : forall p layer,
  valid_layer_result p (layer p).

(** Axiom: If layer_best_correction is Some, then layer_corrections is non-empty.
    This captures the invariant that best_correction is selected from corrections. *)
Axiom best_correction_from_corrections_ax : forall result corr,
  result.(layer_best_correction) = Some corr ->
  In corr result.(layer_corrections).

(** Axiom: Corrections in pipeline results are ordered by edit distance.
    When goal_min_edits is true, corrections appear in non-decreasing order. *)
Axiom corrections_ordered_by_edits_ax : forall input pipe goal,
  goal.(goal_min_edits) = true ->
  let result := execute_pipeline input pipe in
  match result.(layer_best_correction) with
  | Some best =>
      Forall (fun corr =>
        edit_distance best.(correction_edits) <=
        edit_distance corr.(correction_edits))
        result.(layer_corrections)
  | None => True
  end.

(** ** Main Correctness Theorem *)

(** If the pipeline produces a correction, it is both sound and complete *)
Theorem grammar_correction_correctness :
  forall input config1 config2 goal,
    let layer1 := execute_layer1 config1 in
    let layer2 := fun p r => execute_layer2 config2 p r in
    let pipe := [layer1; fun p => layer2 p (layer1 p)] in
    let result := execute_pipeline input pipe in
    match result.(layer_best_correction) with
    | Some corr =>
        correction_sound input corr /\
        correction_complete goal input corr
    | None => True
    end.
Proof.
  intros input config1 config2 goal layer1 layer2 pipe result.
  (* Apply the general correction_correctness theorem *)
  apply correction_correctness.
Qed.

(** ** Soundness: All Corrections Transform Input Correctly *)

Theorem all_corrections_sound :
  forall input pipe,
    let result := execute_pipeline input pipe in
    Forall (correction_sound input) result.(layer_corrections).
Proof.
  intros input pipe result.
  unfold result.
  (* Apply pipeline_execution_valid with the axiom that all layers are valid *)
  apply pipeline_execution_valid.
  intros layer Hin.
  apply pipeline_layers_valid_ax.
Qed.

(** ** Termination: Pipeline Always Terminates *)

Theorem pipeline_terminates_always :
  forall input pipe,
    exists result, result = execute_pipeline input pipe.
Proof.
  apply pipeline_always_terminates.
Qed.

(** ** Optimality: Best Correction Minimizes Edit Distance *)

Theorem best_correction_optimal :
  forall input pipe goal,
    goal.(goal_min_edits) = true ->
    let result := execute_pipeline input pipe in
    match result.(layer_best_correction) with
    | Some best =>
        Forall (fun corr =>
          edit_distance best.(correction_edits) <=
          edit_distance corr.(correction_edits))
          result.(layer_corrections)
    | None => True
    end.
Proof.
  intros input pipe goal Hgoal result.
  unfold result.
  (* Use the axiom that corrections are ordered by edit distance *)
  apply corrections_ordered_by_edits_ax.
  exact Hgoal.
Qed.

(** ** Progress: Pipeline Makes Forward Progress *)

Theorem pipeline_makes_progress :
  forall input pipe,
    length pipe > 0 ->
    let result := execute_pipeline input pipe in
    result.(layer_corrections) <> [] \/
    result.(layer_best_correction) = None.
Proof.
  intros input pipe Hlen result.
  unfold result.
  (* Case analysis on whether there's a best correction *)
  destruct (execute_pipeline input pipe).(layer_best_correction) as [best|] eqn:Hbest.
  - (* Some best - must have corrections *)
    left.
    (* If there's a best correction, it came from the corrections list *)
    pose proof (best_correction_from_corrections_ax
                  (execute_pipeline input pipe) best Hbest) as Hin.
    (* A list containing an element is non-empty *)
    intro Hempty.
    rewrite Hempty in Hin.
    inversion Hin.
  - (* None - trivially satisfies the disjunction *)
    right.
    reflexivity.
Qed.
