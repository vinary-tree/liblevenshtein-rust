(** * Layer 4: Semantic Repair

    This layer attempts to repair type errors through program transformations.
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Program.
Import ListNotations.

Record Layer4Config := {
  max_repairs : nat
}.

Definition default_layer4_config : Layer4Config := {|
  max_repairs := 3
|}.

Definition execute_layer4 (config : Layer4Config) (input : program)
                         (layer3_result : LayerResult)
    : LayerResult :=
  layer3_result.  (* Placeholder *)

(** Layer 4 (semantic repair) is not yet implemented: [execute_layer4] is the
    identity passthrough. We state exactly that instead of a vacuous [True]. *)
Theorem layer4_is_passthrough : forall config input layer3_result,
  execute_layer4 config input layer3_result = layer3_result.
Proof. reflexivity. Qed.
