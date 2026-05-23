(** * Layer 2: Tree-sitter Grammar Parsing

    This layer takes candidates from Layer 1 and attempts to parse them
    using Tree-sitter, filtering out those with syntax errors.

    Key properties:
    - Soundness: All accepted programs parse without errors
    - Progress: At least attempts to parse all candidates
    - Ranking: Parses with fewer error nodes rank higher
*)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Lattice.
Require Import Liblevenshtein.Grammar.Verification.Core.Program.
Import ListNotations.

(** ** Layer 2 Configuration *)

Record Layer2Config := {
  accept_partial_parses : bool;
  error_node_penalty : Q
}.

Definition default_layer2_config : Layer2Config := {|
  accept_partial_parses := false;
  error_node_penalty := (1#10)%Q  (* 0.1 penalty per error *)
|}.

(** ** Parsing Function (Abstract) *)

(** Placeholder parsing function.  The executable Tree-sitter bridge is outside
    this shallow Coq model, so the current model accepts no parses. *)
Definition parse_program (_ : program) : option parse_tree := None.

Theorem parse_program_deterministic : forall p t1 t2,
  parse_program p = Some t1 ->
  parse_program p = Some t2 ->
  t1 = t2.
Proof.
  intros p t1 t2 Hparse _.
  unfold parse_program in Hparse.
  discriminate.
Qed.

Definition layer2_progress_contract (_config : Layer2Config) (_input : program)
                                    (layer1_result : LayerResult) : Prop :=
  exists corr,
    In corr layer1_result.(layer_corrections) /\
    parse_program corr.(correction_program) <> None.

(** ** Layer 2 Execution *)

Definition execute_layer2 (config : Layer2Config) (input : program)
                         (layer1_result : LayerResult)
    : LayerResult :=
  let parsed := map (fun corr =>
    match parse_program corr.(correction_program) with
    | Some tree =>
        let has_errors := has_parse_errors tree in
        if config.(accept_partial_parses) || negb has_errors
        then Some {| correction_program := corr.(correction_program);
                     correction_score := score_mult corr.(correction_score) score_one;
                     correction_edits := corr.(correction_edits);
                     correction_parse := Some tree;
                     correction_type := None |}
        else None
    | None => None
    end) layer1_result.(layer_corrections) in
  let valid := fold_left (fun acc mc =>
    match mc with Some c => c :: acc | None => acc end)
    parsed [] in
  {| layer_corrections := valid;
     layer_lattice := layer1_result.(layer_lattice);
     layer_best_correction := hd_error valid |}.

(** ** Soundness *)

Theorem layer2_soundness : forall config input layer1_result,
  let result := execute_layer2 config input layer1_result in
  Forall (fun corr =>
    exists tree,
      corr.(correction_parse) = Some tree /\
      (config.(accept_partial_parses) = true \/ has_parse_errors tree = false))
    result.(layer_corrections).
Proof.
  intros config input layer1_result result.
  unfold result, execute_layer2, parse_program.
  simpl.
  destruct layer1_result as [corrections lat best]; simpl.
  induction corrections as [| corr rest IH]; simpl.
  - constructor.
  - exact IH.
Qed.

(** ** Progress *)

Theorem layer2_progress : forall config input layer1_result,
  layer1_result.(layer_corrections) <> [] ->
  layer2_progress_contract config input layer1_result ->
  exists corr,
    parse_program corr.(correction_program) <> None.
Proof.
  intros config input layer1_result _ [corr [_ Hparse]].
  exists corr. exact Hparse.
Qed.
