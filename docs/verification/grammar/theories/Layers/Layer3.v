(** * Layer 3: Type Checking

    This layer performs type checking on successfully parsed programs.
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Program.
Import ListNotations.

Record Layer3Config := {
  strict_mode : bool
}.

Definition default_layer3_config : Layer3Config := {|
  strict_mode := true
|}.

Definition type_check_program (tree : parse_tree) : TypeResult :=
  if has_parse_errors tree then
    TypeErrors
      [{| error_msg := "parse error";
          error_span := node_span tree |}]
  else
    TypeOk TyProcess.

Definition type_check_correction (corr : Correction) : Correction :=
  match corr.(correction_parse) with
  | Some tree =>
      {| correction_program := corr.(correction_program);
         correction_score := corr.(correction_score);
         correction_edits := corr.(correction_edits);
         correction_parse := corr.(correction_parse);
         correction_type := Some (type_check_program tree) |}
  | None => corr
  end.

Definition type_check_best (best : option Correction) : option Correction :=
  match best with
  | Some corr => Some (type_check_correction corr)
  | None => None
  end.

Definition execute_layer3 (config : Layer3Config) (input : program)
                         (layer2_result : LayerResult)
    : LayerResult :=
  {| layer_corrections :=
       map type_check_correction layer2_result.(layer_corrections);
     layer_lattice := layer2_result.(layer_lattice);
     layer_best_correction :=
       type_check_best layer2_result.(layer_best_correction) |}.

Lemma type_check_correction_preserves_soundness :
  forall input corr,
    correction_sound input corr ->
    correction_sound input (type_check_correction corr).
Proof.
  intros input corr Hsound.
  unfold type_check_correction.
  destruct corr as [program score edits parse typ].
  destruct parse as [tree |]; simpl in *; exact Hsound.
Qed.

Theorem layer3_preserves_validity :
  forall (config : Layer3Config) (input : program) (layer2_result : LayerResult),
    valid_layer_result input layer2_result ->
    valid_layer_result input (execute_layer3 config input layer2_result).
Proof.
  intros config input layer2_result Hvalid.
  destruct layer2_result as [corrections lat best].
  unfold execute_layer3, valid_layer_result in *; simpl in *.
  induction corrections as [| corr rest IH].
  - constructor.
  - inversion Hvalid as [| ? ? Hsound Hrest]; subst.
    constructor.
    + apply type_check_correction_preserves_soundness.
      exact Hsound.
    + apply IH. exact Hrest.
Qed.
