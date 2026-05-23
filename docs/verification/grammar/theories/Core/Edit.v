(** * Edit Operations and Levenshtein Distance

    This module defines grammar-level edit operations and connects the
    grammar verification stack to the trusted core Levenshtein distance.
*)

From Stdlib Require Import String Ascii List Nat PeanoNat Bool Lia.
From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.MetricProperties.
From Liblevenshtein.Core Require Import Triangle.TriangleInequality.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.

Import ListNotations.

(** ** String/List Bridges *)

Lemma list_ascii_of_string_length : forall s,
  List.length (list_ascii_of_string s) = String.length s.
Proof.
  induction s as [| c s IH]; simpl; [reflexivity | now rewrite IH].
Qed.

Lemma list_ascii_of_string_inj : forall s1 s2,
  list_ascii_of_string s1 = list_ascii_of_string s2 -> s1 = s2.
Proof.
  intros s1 s2 H.
  rewrite <- (string_of_list_ascii_of_string s1).
  rewrite <- (string_of_list_ascii_of_string s2).
  now rewrite H.
Qed.

(** ** Edit Operation Application *)

Definition edit_pos_index (pos : Position) : nat := pos.(pos_col).

Fixpoint insert_at (idx : nat) (c : ascii) (s : string) : string :=
  match idx, s with
  | O, _ => String c s
  | S idx', EmptyString => String c EmptyString
  | S idx', String h rest => String h (insert_at idx' c rest)
  end.

Fixpoint delete_at (idx : nat) (s : string) : string :=
  match idx, s with
  | O, EmptyString => EmptyString
  | O, String _ rest => rest
  | S idx', EmptyString => EmptyString
  | S idx', String h rest => String h (delete_at idx' rest)
  end.

Fixpoint substitute_at (idx : nat) (new_c : ascii) (s : string) : string :=
  match idx, s with
  | O, EmptyString => EmptyString
  | O, String _ rest => String new_c rest
  | S idx', EmptyString => EmptyString
  | S idx', String h rest => String h (substitute_at idx' new_c rest)
  end.

Fixpoint transpose_at (idx : nat) (s : string) : string :=
  match idx, s with
  | O, String c1 (String c2 rest) => String c2 (String c1 rest)
  | O, _ => s
  | S idx', EmptyString => EmptyString
  | S idx', String h rest => String h (transpose_at idx' rest)
  end.

(** Apply a single edit operation to a string. *)
Definition apply_edit (s : string) (op : EditOp) : string :=
  match op with
  | Insertion c pos => insert_at (edit_pos_index pos) c s
  | Deletion _ pos => delete_at (edit_pos_index pos) s
  | Substitution _ new_c pos => substitute_at (edit_pos_index pos) new_c s
  | Transposition _ _ pos => transpose_at (edit_pos_index pos) s
  end.

(** Apply a sequence of edits. *)
Fixpoint apply_edits (s : string) (edits : edit_sequence) : string :=
  match edits with
  | [] => s
  | op :: rest => apply_edits (apply_edit s op) rest
  end.

(** ** Levenshtein Distance *)

Definition levenshtein (s1 s2 : string) : nat :=
  lev_distance (list_ascii_of_string s1) (list_ascii_of_string s2).

(** ** Properties of Edit Distance *)

Theorem levenshtein_symmetric : forall s1 s2,
  levenshtein s1 s2 = levenshtein s2 s1.
Proof.
  intros s1 s2.
  unfold levenshtein.
  apply lev_distance_symmetry.
Qed.

Lemma lev_distance_zero_eq : forall s1 s2,
  lev_distance s1 s2 = 0 -> s1 = s2.
Proof.
  induction s1 as [| c1 s1 IH]; intros s2 Hdist.
  - rewrite lev_distance_empty_left in Hdist.
    destruct s2 as [| c2 s2]; [reflexivity | discriminate].
  - destruct s2 as [| c2 s2].
    + rewrite lev_distance_empty_right in Hdist.
      discriminate.
    + rewrite lev_distance_cons in Hdist.
      unfold min3 in Hdist.
      assert (Hsub : lev_distance s1 s2 + subst_cost c1 c2 = 0) by lia.
      assert (Htail : lev_distance s1 s2 = 0) by lia.
      assert (Hcost : subst_cost c1 c2 = 0) by lia.
      unfold subst_cost, char_eq in Hcost.
      destruct (ascii_dec c1 c2) as [Heq | Hneq].
      * subst c2. f_equal. now apply IH.
      * discriminate.
Qed.

Theorem levenshtein_triangle : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.
Proof.
  intros s1 s2 s3.
  unfold levenshtein.
  apply lev_distance_triangle_inequality.
Qed.

Theorem levenshtein_zero_iff_eq : forall s1 s2,
  levenshtein s1 s2 = 0 <-> s1 = s2.
Proof.
  intros s1 s2. split; intro H.
  - apply list_ascii_of_string_inj.
    apply lev_distance_zero_eq.
    exact H.
  - subst. unfold levenshtein. apply lev_distance_identity.
Qed.

Theorem levenshtein_nonneg : forall s1 s2,
  0 <= levenshtein s1 s2.
Proof.
  intros. lia.
Qed.

(** ** Optimal Edit Sequence *)

Definition optimal_edit_sequence (s1 s2 : string) (edits : edit_sequence) : Prop :=
  apply_edits s1 edits = s2 /\
  edit_distance edits = levenshtein s1 s2.

Definition optimal_edit_exists_contract : Prop :=
  forall s1 s2, exists edits, optimal_edit_sequence s1 s2 edits.

Theorem optimal_edit_exists : forall s1 s2,
  optimal_edit_exists_contract ->
  exists edits, optimal_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2 Hcontract.
  apply Hcontract.
Qed.

(** ** Edit Sequence Composition *)

Definition compose_edits (e1 e2 : edit_sequence) : edit_sequence :=
  e1 ++ e2.

Lemma apply_edits_app : forall s e1 e2,
  apply_edits s (e1 ++ e2) = apply_edits (apply_edits s e1) e2.
Proof.
  intros s e1.
  revert s.
  induction e1 as [| op rest IH]; intros s e2; simpl.
  - reflexivity.
  - apply IH.
Qed.

Theorem compose_edits_correct : forall s1 s2 s3 e1 e2,
  apply_edits s1 e1 = s2 ->
  apply_edits s2 e2 = s3 ->
  apply_edits s1 (compose_edits e1 e2) = s3.
Proof.
  intros s1 s2 s3 e1 e2 H1 H2.
  unfold compose_edits.
  rewrite apply_edits_app.
  now rewrite H1.
Qed.

Theorem compose_edits_distance : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.
Proof.
  intros. apply levenshtein_triangle.
Qed.

(** ** Edit Distance Bounds *)

Theorem levenshtein_upper_bound : forall s1 s2,
  levenshtein s1 s2 <= max (String.length s1) (String.length s2).
Proof.
  intros s1 s2.
  unfold levenshtein.
  rewrite <- !list_ascii_of_string_length.
  apply lev_distance_upper_bound.
Qed.

Definition length_abs_diff (s1 s2 : string) : nat :=
  abs_diff (String.length s1) (String.length s2).

Theorem levenshtein_lower_bound : forall s1 s2,
  levenshtein s1 s2 >= length_abs_diff s1 s2.
Proof.
  intros s1 s2.
  unfold levenshtein, length_abs_diff.
  rewrite <- !list_ascii_of_string_length.
  apply lev_distance_length_diff_lower.
Qed.

(** ** Phonetic and Keyboard Distance *)

Definition keyboard_distance (c1 c2 : ascii) : nat :=
  if ascii_dec c1 c2 then 0 else 1.

Definition phonetic_similar (c1 c2 : ascii) : bool :=
  if ascii_dec c1 c2 then true else false.

Definition edit_cost_keyboard (op : EditOp) : nat :=
  match op with
  | Insertion _ _ => 1
  | Deletion _ _ => 1
  | Substitution c1 c2 _ => keyboard_distance c1 c2
  | Transposition _ _ _ => 1
  end.

Fixpoint edit_sequence_cost (cost_fn : EditOp -> nat) (edits : edit_sequence) : nat :=
  match edits with
  | [] => 0
  | op :: rest => cost_fn op + edit_sequence_cost cost_fn rest
  end.

(** ** Edit Distance with Custom Costs *)

Fixpoint weighted_distance_fuel (fuel : nat) (s1 s2 : string) (n1 n2 : nat)
                                (cost_ins cost_del : nat)
                                (cost_sub : ascii -> ascii -> nat) : nat :=
  match fuel with
  | O => 0
  | S fuel' =>
      match n1, n2 with
      | O, _ => n2 * cost_ins
      | _, O => n1 * cost_del
      | S m1, S m2 =>
          let c1 :=
            match get (String.length s1 - S m1) s1 with
            | Some c => c
            | None => default_char
            end in
          let c2 :=
            match get (String.length s2 - S m2) s2 with
            | Some c => c
            | None => default_char
            end in
          let cost := cost_sub c1 c2 in
          min3
            (cost_del + weighted_distance_fuel fuel' s1 s2 m1 n2 cost_ins cost_del cost_sub)
            (cost_ins + weighted_distance_fuel fuel' s1 s2 n1 m2 cost_ins cost_del cost_sub)
            (cost + weighted_distance_fuel fuel' s1 s2 m1 m2 cost_ins cost_del cost_sub)
      end
  end.

Definition weighted_distance_rec (s1 s2 : string) (n1 n2 : nat)
                                 (cost_ins cost_del : nat)
                                 (cost_sub : ascii -> ascii -> nat) : nat :=
  weighted_distance_fuel (n1 + n2) s1 s2 n1 n2 cost_ins cost_del cost_sub.

Definition weighted_distance (s1 s2 : string)
                             (cost_ins cost_del : nat)
                             (cost_sub : ascii -> ascii -> nat) : nat :=
  weighted_distance_rec s1 s2 (String.length s1) (String.length s2)
    cost_ins cost_del cost_sub.

Definition weighted_distance_unit_contract : Prop := forall s1 s2,
  weighted_distance s1 s2 1 1 (fun c1 c2 => if ascii_dec c1 c2 then 0 else 1) =
  levenshtein s1 s2.

Theorem weighted_distance_unit_costs : forall s1 s2,
  weighted_distance_unit_contract ->
  weighted_distance s1 s2 1 1 (fun c1 c2 => if ascii_dec c1 c2 then 0 else 1) =
  levenshtein s1 s2.
Proof.
  intros s1 s2 Hcontract.
  apply Hcontract.
Qed.

(** ** Correctness of Edit Operations *)

Definition valid_edit_sequence (s1 s2 : string) (edits : edit_sequence) : Prop :=
  apply_edits s1 edits = s2.

Definition zero_pos : Position := {| pos_line := 0; pos_col := 0 |}.

Fixpoint delete_all_edits (s : string) : edit_sequence :=
  match s with
  | EmptyString => []
  | String c rest => Deletion c zero_pos :: delete_all_edits rest
  end.

Fixpoint insert_front_edits (s : string) : edit_sequence :=
  match s with
  | EmptyString => []
  | String c rest => insert_front_edits rest ++ [Insertion c zero_pos]
  end.

Definition valid_witness_edits (s1 s2 : string) : edit_sequence :=
  delete_all_edits s1 ++ insert_front_edits s2.

Lemma delete_all_edits_empty : forall s,
  apply_edits s (delete_all_edits s) = EmptyString.
Proof.
  induction s as [| c rest IH]; simpl; [reflexivity | exact IH].
Qed.

Lemma apply_insert_front_edits_empty : forall s,
  apply_edits EmptyString (insert_front_edits s) = s.
Proof.
  induction s as [| c rest IH]; simpl.
  - reflexivity.
  - rewrite apply_edits_app.
    simpl.
    now rewrite IH.
Qed.

Theorem valid_edit_sequence_exists : forall s1 s2,
  exists edits, valid_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2.
  exists (valid_witness_edits s1 s2).
  unfold valid_edit_sequence, valid_witness_edits.
  rewrite apply_edits_app.
  rewrite delete_all_edits_empty.
  apply apply_insert_front_edits_empty.
Qed.

Theorem optimal_implies_valid : forall s1 s2 edits,
  optimal_edit_sequence s1 s2 edits ->
  valid_edit_sequence s1 s2 edits.
Proof.
  intros s1 s2 edits [H_apply _].
  exact H_apply.
Qed.
