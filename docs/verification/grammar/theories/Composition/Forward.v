(** * Forward Composition of Layers

    This module proves that layers can be composed sequentially
    while preserving correctness properties.
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Program.
Require Import Liblevenshtein.Grammar.Verification.Layers.Layer1.
Require Import Liblevenshtein.Grammar.Verification.Layers.Layer2.
Import ListNotations.

(** ** Sequential Composition *)

Definition compose_forward (layer1 : program -> LayerResult)
                           (layer2 : program -> LayerResult -> LayerResult)
                          (input : program) : LayerResult :=
  let result1 := layer1 input in
  layer2 input result1.

(** ** Composition Preserves Validity *)

Theorem forward_composition_valid : forall layer1 layer2 input,
  valid_layer_result input (layer1 input) ->
  valid_layer_result input (layer2 input (layer1 input)) ->
  valid_layer_result input (compose_forward layer1 layer2 input).
Proof.
  intros layer1 layer2 input Hv1 Hv2.
  unfold compose_forward.
  exact Hv2.
Qed.
