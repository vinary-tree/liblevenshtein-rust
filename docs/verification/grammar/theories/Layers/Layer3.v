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

Definition type_check_program (_ : parse_tree) : TypeResult := TypeErrors [].

Definition execute_layer3 (config : Layer3Config) (input : program)
                         (layer2_result : LayerResult)
    : LayerResult :=
  layer2_result.  (* Placeholder *)

(** The Layer 3 type-checking pass is not yet implemented: [execute_layer3] is
    currently the identity on the incoming [LayerResult]. We record exactly that,
    rather than a vacuous [True] "soundness" claim; real type-soundness awaits a
    real [type_check_program]. *)
Theorem layer3_is_passthrough :
  forall (config : Layer3Config) (input : program) (layer2_result : LayerResult),
    execute_layer3 config input layer2_result = layer2_result.
Proof. reflexivity. Qed.
