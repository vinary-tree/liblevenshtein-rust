(** * Core Types for Grammar Correction Verification

    This module defines the foundational types used throughout the
    grammar correction verification, including programs, characters,
    positions, and scores.
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.micromega.Lia.
Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qround.
Import ListNotations.
Open Scope nat_scope.

(** ** Basic Types *)

(** A character in the source code *)
Definition char := ascii.

(** A program is represented as a string *)
Definition program := string.

(** Position in a program (line, column) *)
Record Position := {
  pos_line : nat;
  pos_col : nat
}.

(** Span in a program (start position, end position) *)
Record Span := {
  span_start : Position;
  span_end : Position
}.

(** ** Scores and Probabilities *)

(** Scores are represented as rational numbers between 0 and 1 *)
Definition score := Q.

Definition score_zero : score := 0%Q.
Definition score_one : score := 1%Q.

(** Score comparison *)
Definition score_lt (s1 s2 : score) : bool :=
  match (s1 ?= s2)%Q with
  | Lt => true
  | _ => false
  end.

Definition score_le (s1 s2 : score) : bool :=
  match (s1 ?= s2)%Q with
  | Gt => false
  | _ => true
  end.

(** Score arithmetic *)
Definition score_mult (s1 s2 : score) : score := (s1 * s2)%Q.

Definition score_add (s1 s2 : score) : score := (s1 + s2)%Q.

(** ** Edit Operations *)

(** Edit distance and operations *)
Inductive EditOp : Type :=
  | Insertion (c : char) (pos : Position)
  | Deletion (c : char) (pos : Position)
  | Substitution (old_c new_c : char) (pos : Position)
  | Transposition (c1 c2 : char) (pos : Position).

(** A sequence of edit operations *)
Definition edit_sequence := list EditOp.

(** Compute the edit distance (number of operations) *)
Fixpoint edit_distance (edits : edit_sequence) : nat :=
  match edits with
  | [] => 0
  | _ :: rest => S (edit_distance rest)
  end.

(** ** Parse Trees *)

(** Node kind in parse tree (simplified) *)
Inductive NodeKind : Type :=
  | NK_Program
  | NK_Statement
  | NK_Expression
  | NK_Identifier (name : string)
  | NK_Literal (value : string)
  | NK_BinaryOp (op : string)
  | NK_UnaryOp (op : string)
  | NK_Block
  | NK_Error.

(** Parse tree node (inductive for recursion) *)
Inductive ParseNode : Type :=
  | MkParseNode (kind : NodeKind) (span : Span) (children : list ParseNode).

(** Accessors for ParseNode *)
Definition node_kind (n : ParseNode) : NodeKind :=
  match n with MkParseNode k _ _ => k end.

Definition node_span (n : ParseNode) : Span :=
  match n with MkParseNode _ s _ => s end.

Definition node_children (n : ParseNode) : list ParseNode :=
  match n with MkParseNode _ _ c => c end.

(** Parse tree is the root node *)
Definition parse_tree := ParseNode.

(** Check if parse tree contains errors *)
Fixpoint has_parse_errors (tree : parse_tree) : bool :=
  match node_kind tree with
  | NK_Error => true
  | _ => existsb has_parse_errors (node_children tree)
  end.

(** ** Type System *)

(** Types in the object language type system (simplified for Rholang) *)
Inductive GType : Type :=
  | TyUnit
  | TyBool
  | TyInt
  | TyString
  | TyName
  | TyProcess
  | TyChannel (t : GType)
  | TyList (t : GType)
  | TyTuple (arity : nat)
  | TyUnknown
  | TyError.

(** Type environment maps variables to types *)
Definition type_env := list (string * GType).

(** Type error *)
Record TypeError := {
  error_msg : string;
  error_span : Span
}.

(** Type checking result *)
Inductive TypeResult : Type :=
  | TypeOk (t : GType)
  | TypeErrors (errors : list TypeError).

(** ** Lattice Structures *)

(** Lattice node represents a position in the error correction lattice *)
Record LatticeNode := {
  lattice_position : nat;              (** Position in input *)
  lattice_text : string;                (** Text fragment at this node *)
  lattice_score : score;                (** Cumulative score *)
  lattice_edits : edit_sequence         (** Edits applied to reach this node *)
}.

(** Lattice edge connects two nodes *)
Record LatticeEdge := {
  edge_from : nat;                      (** Source node index *)
  edge_to : nat;                        (** Target node index *)
  edge_weight : score                   (** Edge weight (probability) *)
}.

(** Lattice is a collection of nodes and edges *)
Record Lattice := {
  lattice_nodes : list LatticeNode;
  lattice_edges : list LatticeEdge;
  lattice_start : nat;                  (** Start node index *)
  lattice_end : nat                     (** End node index *)
}.

(** ** Correction Candidates *)

(** A correction candidate *)
Record Correction := {
  correction_program : program;
  correction_score : score;
  correction_edits : edit_sequence;
  correction_parse : option parse_tree;
  correction_type : option TypeResult
}.

(** ** Utility Functions *)

(** String equality (decidable) *)
Definition string_eqb (s1 s2 : string) : bool :=
  if string_dec s1 s2 then true else false.

(** Position equality *)
Definition position_eqb (p1 p2 : Position) : bool :=
  (p1.(pos_line) =? p2.(pos_line)) && (p1.(pos_col) =? p2.(pos_col)).

(** Type equality (simplified) *)
Fixpoint type_eqb (t1 t2 : GType) : bool :=
  match t1, t2 with
  | TyUnit, TyUnit => true
  | TyBool, TyBool => true
  | TyInt, TyInt => true
  | TyString, TyString => true
  | TyName, TyName => true
  | TyProcess, TyProcess => true
  | TyChannel t1', TyChannel t2' => type_eqb t1' t2'
  | TyList t1', TyList t2' => type_eqb t1' t2'
  | TyTuple n1, TyTuple n2 => n1 =? n2
  | TyUnknown, TyUnknown => true
  | TyError, TyError => true
  | _, _ => false
  end.

(** ** Basic Properties *)

(** Score properties *)
Lemma score_mult_comm : forall s1 s2,
  score_mult s1 s2 == score_mult s2 s1.
Proof.
  intros. unfold score_mult. ring.
Qed.

Lemma score_mult_assoc : forall s1 s2 s3,
  score_mult (score_mult s1 s2) s3 == score_mult s1 (score_mult s2 s3).
Proof.
  intros. unfold score_mult. ring.
Qed.

Lemma score_mult_one_l : forall s,
  score_mult score_one s == s.
Proof.
  intros. unfold score_mult, score_one. ring.
Qed.

Lemma score_mult_zero_l : forall s,
  score_mult score_zero s == score_zero.
Proof.
  intros. unfold score_mult, score_zero. ring.
Qed.

(** Edit distance is always non-negative *)
Lemma edit_distance_nonneg : forall edits,
  (0 <= edit_distance edits)%nat.
Proof.
  intros. induction edits; simpl; lia.
Qed.

(** Edit distance of concatenated sequences *)
Lemma edit_distance_app : forall e1 e2,
  edit_distance (e1 ++ e2) = (edit_distance e1 + edit_distance e2)%nat.
Proof.
  intros e1 e2. induction e1; simpl.
  - reflexivity.
  - rewrite IHe1. lia.
Qed.

(** Type equality is reflexive *)
Lemma type_eqb_refl : forall t,
  type_eqb t t = true.
Proof.
  induction t; simpl; auto.
  all: try match goal with
           | H : type_eqb _ _ = true |- type_eqb _ _ = true => exact H
           end.
  apply Nat.eqb_refl.
Qed.

(** ** Well-formedness Conditions *)

(** A position is well-formed if line and column are valid *)
Definition wf_position (p : Position) : Prop :=
  p.(pos_line) > 0 /\ p.(pos_col) > 0.

(** A span is well-formed if start precedes end *)
Definition wf_span (s : Span) : Prop :=
  wf_position s.(span_start) /\
  wf_position s.(span_end) /\
  (s.(span_start).(pos_line) < s.(span_end).(pos_line) \/
   (s.(span_start).(pos_line) = s.(span_end).(pos_line) /\
    s.(span_start).(pos_col) <= s.(span_end).(pos_col))).

(** A score is well-formed if it's between 0 and 1 *)
Definition wf_score (s : score) : Prop :=
  (score_zero <= s)%Q /\ (s <= score_one)%Q.

(** A lattice is well-formed if:
    - All node indices are valid
    - All edges connect valid nodes
    - Start and end nodes exist
    - All scores are well-formed
*)
Definition wf_lattice (lat : Lattice) : Prop :=
  lat.(lattice_start) < length lat.(lattice_nodes) /\
  lat.(lattice_end) < length lat.(lattice_nodes) /\
  Forall (fun e =>
    e.(edge_from) < length lat.(lattice_nodes) /\
    e.(edge_to) < length lat.(lattice_nodes) /\
    wf_score e.(edge_weight)
  ) lat.(lattice_edges) /\
  Forall (fun n => wf_score n.(lattice_score)) lat.(lattice_nodes).
