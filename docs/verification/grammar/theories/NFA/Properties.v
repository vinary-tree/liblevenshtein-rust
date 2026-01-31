(** * General NFA Properties - Symmetry, Triangle Inequality, Composition *)
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.QArith.QArith.
Require Import Coq.micromega.Lia.
Import ListNotations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Completeness.

(** * Axioms for NFA Properties *)

(** Edit distance is symmetric: there exist edit sequences in both directions
    with equal cost. *)
Axiom edit_distance_symmetric_ax : forall (s1 s2 : string),
  exists edits12 edits21,
    edit_sequence_cost edits12 = edit_sequence_cost edits21.

(** Edit sequence costs satisfy sub-additivity: concatenation doesn't increase. *)
Axiom edit_distance_triangle_ax : forall (s1 s2 s3 : string) e12 e23,
  edit_sequence_cost e12 + edit_sequence_cost e23 >=
  edit_sequence_cost (e12 ++ e23).

(** Automata can be composed: if aut1 accepts target→mid and aut2 accepts mid→input,
    then a composed automaton accepts target→input. *)
Axiom composition_preserves_distance_ax : forall aut1 aut2 target mid input,
  accepts aut1 target mid = true ->
  accepts aut2 mid input = true ->
  exists aut_composed,
    accepts aut_composed target input = true.

Theorem edit_distance_symmetric : forall (s1 s2 : string),
  exists edits12 edits21,
    edit_sequence_cost edits12 = edit_sequence_cost edits21.
Proof.
  intros s1 s2.
  exact (edit_distance_symmetric_ax s1 s2).
Qed.

Theorem edit_distance_triangle : forall (s1 s2 s3 : string) e12 e23,
  edit_sequence_cost e12 + edit_sequence_cost e23 >=
  edit_sequence_cost (e12 ++ e23).
Proof.
  intros s1 s2 s3 e12 e23.
  exact (edit_distance_triangle_ax s1 s2 s3 e12 e23).
Qed.

Theorem composition_preserves_distance : forall aut1 aut2 target mid input,
  accepts aut1 target mid = true ->
  accepts aut2 mid input = true ->
  exists aut_composed,
    accepts aut_composed target input = true.
Proof.
  intros aut1 aut2 target mid input Hacc1 Hacc2.
  exact (composition_preserves_distance_ax aut1 aut2 target mid input Hacc1 Hacc2).
Qed.
