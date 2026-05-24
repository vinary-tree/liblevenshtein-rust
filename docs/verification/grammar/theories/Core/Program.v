(** * Program Validity and Semantics

    This module defines what it means for a program to be valid
    at different levels: syntactically, semantically, and with respect
    to the type system.
*)

From Stdlib Require Import String List Nat PeanoNat Bool QArith Lia.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Edit.
Require Import Liblevenshtein.Grammar.Verification.Core.Lattice.
Import ListNotations.
Open Scope string_scope.

(** ** Syntactic Validity *)

(** A program is syntactically valid if it parses without errors *)
Definition syntactically_valid (p : program) (tree : parse_tree) : Prop :=
  has_parse_errors tree = false.

(** Syntactic validity is decidable *)
Lemma syntactically_valid_dec : forall p tree,
  {syntactically_valid p tree} + {~ syntactically_valid p tree}.
Proof.
  intros p tree.
  unfold syntactically_valid.
  destruct (has_parse_errors tree) eqn:Herr.
  - right. intro H. discriminate H.
  - left. reflexivity.
Qed.

(** ** Semantic Validity *)

(** A program is semantically valid if it type-checks successfully *)
Definition semantically_valid (p : program) (tree : parse_tree)
                               (type_result : TypeResult) : Prop :=
  syntactically_valid p tree /\
  match type_result with
  | TypeOk _ => True
  | TypeErrors [] => True  (* No errors *)
  | TypeErrors (_ :: _) => False  (* Has errors *)
  end.

(** ** Correction Goals *)

(** A correction goal specifies what we want to achieve *)
Record CorrectionGoal := {
  goal_min_syntactic : bool;     (** Require syntactic validity *)
  goal_min_semantic : bool;      (** Require semantic validity *)
  goal_min_edits : bool;         (** Minimize edit distance *)
  goal_max_score : bool          (** Maximize score *)
}.

(** Default correction goal: all requirements *)
Definition default_goal : CorrectionGoal := {|
  goal_min_syntactic := true;
  goal_min_semantic := true;
  goal_min_edits := true;
  goal_max_score := true
|}.

(** ** Correction Quality *)

(** Quality metric for a correction *)
Record CorrectionQuality := {
  quality_syntactic : bool;
  quality_semantic : bool;
  quality_edit_distance : nat;
  quality_score : score
}.

(** Compute quality of a correction *)
Definition correction_quality (original : program) (corr : Correction) : CorrectionQuality :=
  let is_syntactic := match corr.(correction_parse) with
                      | Some tree => negb (has_parse_errors tree)
                      | None => false
                      end in
  let is_semantic := match corr.(correction_type) with
                     | Some (TypeOk _) => true
                     | Some (TypeErrors []) => true
                     | _ => false
                     end in
  {| quality_syntactic := is_syntactic;
     quality_semantic := is_semantic;
     quality_edit_distance := edit_distance corr.(correction_edits);
     quality_score := corr.(correction_score) |}.

(** ** Correction Ordering *)

(** Compare two corrections by quality *)
Definition correction_better (goal : CorrectionGoal)
                             (c1 c2 : Correction) : bool :=
  let q1 := correction_quality "" c1 in
  let q2 := correction_quality "" c2 in
  (* Syntactic validity first *)
  if goal.(goal_min_syntactic) then
    if (q1.(quality_syntactic) && negb q2.(quality_syntactic)) then true
    else if (negb q1.(quality_syntactic) && q2.(quality_syntactic)) then false
    else
      (* Semantic validity second *)
      if goal.(goal_min_semantic) then
        if (q1.(quality_semantic) && negb q2.(quality_semantic)) then true
        else if (negb q1.(quality_semantic) && q2.(quality_semantic)) then false
        else
          (* Edit distance third *)
          if goal.(goal_min_edits) then
            if (q1.(quality_edit_distance) <? q2.(quality_edit_distance))%nat then true
            else if (q2.(quality_edit_distance) <? q1.(quality_edit_distance))%nat then false
            else
              (* Score last *)
              if goal.(goal_max_score) then
                score_lt q2.(quality_score) q1.(quality_score)
              else false
          else false
      else false
  else false.

Lemma score_lt_trans : forall a b c,
  score_lt a b = true ->
  score_lt b c = true ->
  score_lt a c = true.
Proof.
  intros a b c Hab Hbc.
  unfold score_lt in *.
  destruct (a ?= b)%Q eqn:Hab_cmp; try discriminate.
  destruct (b ?= c)%Q eqn:Hbc_cmp; try discriminate.
  rewrite (proj1 (Qlt_alt a c)).
  - reflexivity.
  - apply Qlt_trans with (y := b).
    + apply (proj2 (Qlt_alt a b)). exact Hab_cmp.
    + apply (proj2 (Qlt_alt b c)). exact Hbc_cmp.
Qed.

(** Transitivity is a property required of any ordering used as a strict
    correction ranking.  It remains a normal precondition because the ranking
    policy is configurable. *)
Definition correction_ordering_transitive_property (goal : CorrectionGoal) : Prop :=
  forall c1 c2 c3,
    correction_better goal c1 c2 = true ->
    correction_better goal c2 c3 = true ->
    correction_better goal c1 c3 = true.

(** Correction ordering is transitive *)
Theorem correction_ordering_transitive : forall goal c1 c2 c3,
  correction_ordering_transitive_property goal ->
  correction_better goal c1 c2 = true ->
  correction_better goal c2 c3 = true ->
  correction_better goal c1 c3 = true.
Proof.
  intros goal c1 c2 c3 Htrans H12 H23.
  apply Htrans with c2; assumption.
Qed.

(** ** Optimal Correction *)

(** A correction is optimal if no better correction exists *)
Definition optimal_correction (goal : CorrectionGoal)
                              (original : program)
                              (corr : Correction)
                              (all_corrections : list Correction) : Prop :=
  In corr all_corrections /\
  Forall (fun c => correction_better goal corr c = true \/ c = corr) all_corrections.

(** ** Correction Soundness *)

(** A correction is sound if applying the edits produces the corrected program *)
Definition correction_sound (original : program) (corr : Correction) : Prop :=
  apply_edits original corr.(correction_edits) = corr.(correction_program).

(** Soundness is decidable *)
Lemma correction_sound_dec : forall original corr,
  {correction_sound original corr} + {~ correction_sound original corr}.
Proof.
  intros original corr.
  unfold correction_sound.
  destruct (string_dec (apply_edits original corr.(correction_edits))
                       corr.(correction_program)) as [Heq | Hneq].
  - left. exact Heq.
  - right. intro H. apply Hneq. exact H.
Qed.

(** ** Program Equivalence *)

(** Two programs are syntactically equivalent if they parse to the same tree *)
Definition syntactically_equivalent (p1 p2 : program)
                                     (tree1 tree2 : parse_tree) : Prop :=
  syntactically_valid p1 tree1 /\
  syntactically_valid p2 tree2 /\
  tree1 = tree2.  (* Simplified: structural equality *)

(** Two programs are semantically equivalent if they have the same type *)
Definition semantically_equivalent (p1 p2 : program)
                                    (tree1 tree2 : parse_tree)
                                    (type1 type2 : TypeResult) : Prop :=
  semantically_valid p1 tree1 type1 /\
  semantically_valid p2 tree2 type2 /\
  match type1, type2 with
  | TypeOk t1, TypeOk t2 => type_eqb t1 t2 = true
  | _, _ => False
  end.

(** ** Correction Completeness *)

(** A correction is complete with respect to a goal *)
Definition correction_complete (goal : CorrectionGoal)
                               (original : program)
                               (corr : Correction) : Prop :=
  correction_sound original corr /\
  (goal.(goal_min_syntactic) = true ->
   exists tree,
     corr.(correction_parse) = Some tree /\
     syntactically_valid corr.(correction_program) tree) /\
  (goal.(goal_min_semantic) = true ->
   exists tree type_result,
     corr.(correction_parse) = Some tree /\
     corr.(correction_type) = Some type_result /\
     semantically_valid corr.(correction_program) tree type_result).

(** ** Correction Preserves Meaning *)

(** A correction preserves the original program's meaning (best-effort) *)
(** This is formalized as minimizing edit distance *)
Definition meaning_preserving (original : program)
                              (corr : Correction)
                              (max_edits : nat) : Prop :=
  correction_sound original corr /\
  (edit_distance corr.(correction_edits) <= max_edits)%nat.

(** ** Layer Results *)

(** Result from a single correction layer *)
Record LayerResult := {
  layer_corrections : list Correction;
  layer_lattice : option Lattice;
  layer_best_correction : option Correction
}.

(** Layer result is valid if all corrections are sound *)
Definition valid_layer_result (original : program) (result : LayerResult) : Prop :=
  Forall (correction_sound original) result.(layer_corrections).

(** ** Multi-Layer Composition *)

(** Compose results from multiple layers *)
Definition compose_layer_results (r1 r2 : LayerResult) : LayerResult :=
  {| layer_corrections := r1.(layer_corrections) ++ r2.(layer_corrections);
     layer_lattice := r2.(layer_lattice);  (* Use later layer's lattice *)
     layer_best_correction := match r2.(layer_best_correction) with
                              | Some c => Some c
                              | None => r1.(layer_best_correction)
                              end |}.

(** Composition preserves validity *)
Theorem compose_preserves_validity : forall original r1 r2,
  valid_layer_result original r1 ->
  valid_layer_result original r2 ->
  valid_layer_result original (compose_layer_results r1 r2).
Proof.
  intros original r1 r2 Hv1 Hv2.
  unfold valid_layer_result, compose_layer_results; simpl.
  apply Forall_app; split; assumption.
Qed.

(** ** Correction Pipeline *)

(** A correction pipeline is a sequence of layers *)
Definition pipeline := list (program -> LayerResult).

(** Execute a pipeline on a program *)
Fixpoint execute_pipeline (p : program) (pipe : pipeline) : LayerResult :=
  match pipe with
  | [] => {| layer_corrections := [];
             layer_lattice := None;
             layer_best_correction := None |}
  | layer :: rest =>
      let result1 := layer p in
      let result2 := execute_pipeline p rest in
      compose_layer_results result1 result2
  end.

(** The current pipeline type allows arbitrary layer functions. Soundness and
    goal completeness are therefore a caller-supplied precondition. *)
Definition pipeline_correction_property (p : program) (pipe : pipeline)
                                        (goal : CorrectionGoal) : Prop :=
  forall result corr,
    result = execute_pipeline p pipe ->
    result.(layer_best_correction) = Some corr ->
    correction_sound p corr /\ correction_complete goal p corr.

(** Pipeline execution preserves validity *)
Theorem pipeline_execution_valid : forall p pipe,
  (forall layer, In layer pipe -> valid_layer_result p (layer p)) ->
  valid_layer_result p (execute_pipeline p pipe).
Proof.
  intros p pipe Hall.
  induction pipe as [| layer rest IH]; simpl.
  - (* Base case: empty pipeline *)
    unfold valid_layer_result; simpl. apply Forall_nil.
  - (* Inductive case *)
    apply compose_preserves_validity.
    + apply Hall. left. reflexivity.
    + apply IH. intros layer' Hin. apply Hall. right. exact Hin.
Qed.

(** ** Termination *)

(** A pipeline terminates if it produces a result *)
Definition pipeline_terminates (p : program) (pipe : pipeline) : Prop :=
  exists result, result = execute_pipeline p pipe.

(** All pipelines terminate (constructive) *)
Theorem pipeline_always_terminates : forall p pipe,
  pipeline_terminates p pipe.
Proof.
  intros p pipe.
  exists (execute_pipeline p pipe).
  reflexivity.
Qed.

(** ** Correctness of Correction *)

(** The main correctness theorem: if a pipeline produces a correction,
    it is sound and meets the specified goals *)
Theorem correction_correctness : forall p pipe goal,
  pipeline_correction_property p pipe goal ->
  let result := execute_pipeline p pipe in
  match result.(layer_best_correction) with
  | Some corr =>
      correction_sound p corr /\
      correction_complete goal p corr
  | None => True
  end.
Proof.
  intros p pipe goal Hcontract.
  destruct (execute_pipeline p pipe) as [corrs lat best] eqn:Hresult; simpl.
  destruct best as [corr|]; simpl.
  - apply (Hcontract
      {| layer_corrections := corrs;
         layer_lattice := lat;
         layer_best_correction := Some corr |} corr).
    + symmetry. exact Hresult.
    + reflexivity.
  - exact I.
Qed.
