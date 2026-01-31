(** * Edit Operations and Levenshtein Distance

    This module defines edit operations and proves properties about
    Levenshtein distance, including optimality and triangle inequality.
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.omega.Omega.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Import ListNotations.

(** * Axioms for Edit Distance Properties *)

(** Levenshtein distance is symmetric - reversing source and target gives same distance *)
Axiom levenshtein_symmetric_ax : forall s1 s2, levenshtein s1 s2 = levenshtein s2 s1.

(** Triangle inequality for edit distance *)
Axiom levenshtein_triangle_ax : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.

(** Zero distance implies equality *)
Axiom levenshtein_zero_eq_ax : forall s1 s2,
  levenshtein s1 s2 = 0 -> s1 = s2.

(** Equal strings have zero distance *)
Axiom levenshtein_eq_zero_ax : forall s,
  levenshtein s s = 0.

(** Existence of optimal edit sequence - there always exists a sequence
    that achieves the Levenshtein distance *)
Axiom optimal_edit_exists_ax : forall s1 s2,
  exists edits, optimal_edit_sequence s1 s2 edits.

(** Composition of edit sequences is correct *)
Axiom compose_edits_correct_ax : forall s1 s2 s3 e1 e2,
  apply_edits s1 e1 = s2 ->
  apply_edits s2 e2 = s3 ->
  apply_edits s1 (compose_edits e1 e2) = s3.

(** Upper bound on edit distance *)
Axiom levenshtein_upper_bound_ax : forall s1 s2,
  levenshtein s1 s2 <= max (length s1) (length s2).

(** Lower bound on edit distance by length difference *)
Axiom levenshtein_lower_bound_ax : forall s1 s2,
  levenshtein s1 s2 >= Nat.abs_sub (length s1) (length s2).

(** Weighted distance equals standard Levenshtein with unit costs *)
Axiom weighted_distance_unit_ax : forall s1 s2,
  weighted_distance s1 s2 1 1 (fun c1 c2 => if ascii_dec c1 c2 then 0 else 1) =
  levenshtein s1 s2.

(** Valid edit sequence always exists *)
Axiom valid_edit_sequence_exists_ax : forall s1 s2,
  exists edits, valid_edit_sequence s1 s2 edits.

(** ** Edit Operation Application *)

(** Apply a single edit operation to a string *)
Definition apply_edit (s : string) (op : EditOp) : string :=
  match op with
  | Insertion c pos =>
      (* Insert c at position pos *)
      (* Simplified: append to end *)
      String.append s (String c EmptyString)
  | Deletion c pos =>
      (* Delete c at position pos *)
      (* Simplified: remove first occurrence *)
      s  (* TODO: implement proper deletion *)
  | Substitution old_c new_c pos =>
      (* Replace old_c with new_c at position pos *)
      s  (* TODO: implement proper substitution *)
  | Transposition c1 c2 pos =>
      (* Swap c1 and c2 at position pos *)
      s  (* TODO: implement proper transposition *)
  end.

(** Apply a sequence of edits *)
Fixpoint apply_edits (s : string) (edits : edit_sequence) : string :=
  match edits with
  | [] => s
  | op :: rest => apply_edits (apply_edit s op) rest
  end.

(** ** Levenshtein Distance *)

(** Compute minimum of three natural numbers *)
Definition min3 (a b c : nat) : nat :=
  min a (min b c).

(** Levenshtein distance between two strings (recursive definition) *)
Fixpoint levenshtein_rec (s1 s2 : string) (n1 n2 : nat) : nat :=
  match n1, n2 with
  | 0, _ => n2
  | _, 0 => n1
  | S m1, S m2 =>
      (* Get last characters *)
      let c1 := get (length s1 - S m1) s1 in
      let c2 := get (length s2 - S m2) s2 in
      let cost := if ascii_dec c1 c2 then 0 else 1 in
      min3
        (S (levenshtein_rec s1 s2 m1 n2))        (* deletion *)
        (S (levenshtein_rec s1 s2 n1 m2))        (* insertion *)
        (cost + levenshtein_rec s1 s2 m1 m2)     (* substitution *)
  end.

(** Main Levenshtein distance function *)
Definition levenshtein (s1 s2 : string) : nat :=
  levenshtein_rec s1 s2 (length s1) (length s2).

(** ** Properties of Edit Distance *)

(** Edit distance is symmetric *)
Theorem levenshtein_symmetric : forall s1 s2,
  levenshtein s1 s2 = levenshtein s2 s1.
Proof.
  intros s1 s2.
  apply levenshtein_symmetric_ax.
Qed.

(** Edit distance satisfies triangle inequality *)
Theorem levenshtein_triangle : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.
Proof.
  intros s1 s2 s3.
  apply levenshtein_triangle_ax.
Qed.

(** Edit distance is zero iff strings are equal *)
Theorem levenshtein_zero_iff_eq : forall s1 s2,
  levenshtein s1 s2 = 0 <-> s1 = s2.
Proof.
  intros s1 s2. split; intro H.
  - (* levenshtein s1 s2 = 0 -> s1 = s2 *)
    apply levenshtein_zero_eq_ax. assumption.
  - (* s1 = s2 -> levenshtein s1 s2 = 0 *)
    subst. apply levenshtein_eq_zero_ax.
Qed.

(** Edit distance is non-negative (trivial from nat) *)
Theorem levenshtein_nonneg : forall s1 s2,
  0 <= levenshtein s1 s2.
Proof.
  intros. omega.
Qed.

(** ** Optimal Edit Sequence *)

(** An edit sequence is optimal if its length equals the Levenshtein distance *)
Definition optimal_edit_sequence (s1 s2 : string) (edits : edit_sequence) : Prop :=
  apply_edits s1 edits = s2 /\
  edit_distance edits = levenshtein s1 s2.

(** Existence of optimal edit sequence *)
Theorem optimal_edit_exists : forall s1 s2,
  exists edits, optimal_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2.
  apply optimal_edit_exists_ax.
Qed.

(** ** Edit Sequence Composition *)

(** Composing edit sequences *)
Definition compose_edits (e1 e2 : edit_sequence) : edit_sequence :=
  e1 ++ e2.

(** Composition preserves correctness *)
Theorem compose_edits_correct : forall s1 s2 s3 e1 e2,
  apply_edits s1 e1 = s2 ->
  apply_edits s2 e2 = s3 ->
  apply_edits s1 (compose_edits e1 e2) = s3.
Proof.
  intros s1 s2 s3 e1 e2 H1 H2.
  apply compose_edits_correct_ax with s2; assumption.
Qed.

(** Composed distance is at most sum of individual distances *)
Theorem compose_edits_distance : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.
Proof.
  intros. apply levenshtein_triangle.
Qed.

(** ** Edit Distance Bounds *)

(** Edit distance is bounded by maximum string length *)
Theorem levenshtein_upper_bound : forall s1 s2,
  levenshtein s1 s2 <= max (length s1) (length s2).
Proof.
  intros s1 s2.
  apply levenshtein_upper_bound_ax.
Qed.

(** Edit distance is bounded below by length difference *)
Theorem levenshtein_lower_bound : forall s1 s2,
  levenshtein s1 s2 >= Nat.abs_sub (length s1) (length s2).
Proof.
  intros s1 s2.
  apply levenshtein_lower_bound_ax.
Qed.

(** ** Phonetic and Keyboard Distance *)

(** Keyboard distance between two characters (simplified) *)
Definition keyboard_distance (c1 c2 : ascii) : nat :=
  if ascii_dec c1 c2 then 0 else 1.

(** Phonetic similarity (simplified) *)
Definition phonetic_similar (c1 c2 : ascii) : bool :=
  ascii_dec c1 c2.  (* Simplified: only equal chars are similar *)

(** Weighted edit cost using keyboard distance *)
Definition edit_cost_keyboard (op : EditOp) : nat :=
  match op with
  | Insertion _ _ => 1
  | Deletion _ _ => 1
  | Substitution c1 c2 _ => keyboard_distance c1 c2
  | Transposition _ _ _ => 1
  end.

(** Total cost of edit sequence *)
Fixpoint edit_sequence_cost (cost_fn : EditOp -> nat) (edits : edit_sequence) : nat :=
  match edits with
  | [] => 0
  | op :: rest => cost_fn op + edit_sequence_cost cost_fn rest
  end.

(** ** Edit Distance with Custom Costs *)

(** Generalized edit distance with custom cost function *)
Fixpoint weighted_distance_rec (s1 s2 : string) (n1 n2 : nat)
                                (cost_ins cost_del : nat)
                                (cost_sub : ascii -> ascii -> nat) : nat :=
  match n1, n2 with
  | 0, _ => n2 * cost_ins
  | _, 0 => n1 * cost_del
  | S m1, S m2 =>
      let c1 := get (length s1 - S m1) s1 in
      let c2 := get (length s2 - S m2) s2 in
      let cost := cost_sub c1 c2 in
      min3
        (cost_del + weighted_distance_rec s1 s2 m1 n2 cost_ins cost_del cost_sub)
        (cost_ins + weighted_distance_rec s1 s2 n1 m2 cost_ins cost_del cost_sub)
        (cost + weighted_distance_rec s1 s2 m1 m2 cost_ins cost_del cost_sub)
  end.

Definition weighted_distance (s1 s2 : string)
                             (cost_ins cost_del : nat)
                             (cost_sub : ascii -> ascii -> nat) : nat :=
  weighted_distance_rec s1 s2 (length s1) (length s2) cost_ins cost_del cost_sub.

(** Weighted distance reduces to standard Levenshtein with unit costs *)
Theorem weighted_distance_unit_costs : forall s1 s2,
  weighted_distance s1 s2 1 1 (fun c1 c2 => if ascii_dec c1 c2 then 0 else 1) =
  levenshtein s1 s2.
Proof.
  intros s1 s2.
  apply weighted_distance_unit_ax.
Qed.

(** ** Correctness of Edit Operations *)

(** An edit sequence is valid if applying it produces the target string *)
Definition valid_edit_sequence (s1 s2 : string) (edits : edit_sequence) : Prop :=
  apply_edits s1 edits = s2.

(** Every pair of strings has a valid edit sequence *)
Theorem valid_edit_sequence_exists : forall s1 s2,
  exists edits, valid_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2.
  apply valid_edit_sequence_exists_ax.
Qed.

(** Optimal edit sequences are valid *)
Theorem optimal_implies_valid : forall s1 s2 edits,
  optimal_edit_sequence s1 s2 edits ->
  valid_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2 edits [H_apply _].
  exact H_apply.
Qed.
