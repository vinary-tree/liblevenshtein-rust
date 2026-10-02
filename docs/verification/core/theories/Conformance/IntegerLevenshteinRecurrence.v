(** * Exact-integer Standard Levenshtein recurrence bridge

    ORC Section 11.1 specifies the fixed-query update with three unit-edit
    alternatives. This module represents a row as a total nat-indexed
    function, so it isolates the arithmetic and terminal-coordinate proof
    from array allocation and positional-state representation. Its reference
    is the independently organized head-edit [lev_distance] from the core
    verification library; [lev_distance_snoc] bridges head edits to the
    streaming, last-label recurrence.

    The checked-add model below uses unbounded naturals followed by an
    explicit maximum test. It is a conservative all-candidates-fit
    precondition, not a claim about a particular Rust overflow path. The
    [Char] carrier is this core library's ASCII model; Unicode decoding
    and a production Rust array layout remain separate correspondence
    obligations. *)

From Stdlib Require Import Arith Lia List Ascii.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import DPMatrix.SnocLemmas.

Definition reference_cell
    (query target : list Char) (index : nat) : nat :=
  lev_distance (firstn index query) target.

(** The seed is the cost of deleting each query prefix against empty
    target. Only coordinates through [length query] are observable. *)
Definition seed_row (index : nat) : nat := index.

(** One cell consumes one new target label. [previous] is the old row;
    the recursive call computes the immediately preceding cell in the
    new row. This is ORC's old-same, new-left, old-diagonal order. *)
Fixpoint transition_cell
    (query : list Char) (label : Char)
    (previous : nat -> nat) (index : nat) : nat :=
  match index with
  | 0 => previous 0 + 1
  | S prior_index =>
      min3
        (previous (S prior_index) + 1)
        (transition_cell query label previous prior_index + 1)
        (previous prior_index +
         subst_cost (nth prior_index query default_char) label)
  end.

Definition stream_row
    (query target : list Char) : nat -> nat :=
  fold_left (fun row label => transition_cell query label row)
    target seed_row.

Definition terminal_cost
    (query : list Char) (row : nat -> nat) : nat :=
  row (length query).

Lemma seed_row_refines_empty_target : forall query index,
  index <= length query ->
  seed_row index = reference_cell query [] index.
Proof.
  intros query index Hindex.
  unfold seed_row, reference_cell.
  rewrite lev_distance_empty_right.
  rewrite length_firstn.
  rewrite Nat.min_l by lia.
  reflexivity.
Qed.

(** The reference's snoc lemma lists the horizontal and vertical edits in
    the opposite order; minimum is commutative in those arguments. *)
Lemma min3_swap_first_two : forall first second third,
  min3 first second third = min3 second first third.
Proof.
  intros first second third.
  unfold min3.
  rewrite Nat.min_assoc.
  rewrite (Nat.min_comm first second).
  rewrite <- Nat.min_assoc.
  reflexivity.
Qed.

Theorem one_label_transition_refines_reference :
  forall query target label previous index,
    (forall k, k <= length query ->
      previous k = reference_cell query target k) ->
    index <= length query ->
    transition_cell query label previous index =
      reference_cell query (target ++ [label]) index.
Proof.
  intros query target label previous index Hprevious Hindex.
  induction index as [|prior_index IH].
  - unfold reference_cell in *.
    simpl transition_cell.
    rewrite (Hprevious 0) by lia.
    simpl firstn.
    rewrite !lev_distance_empty_left.
    rewrite length_app. simpl. lia.
  - assert (Hprior : prior_index < length query) by lia.
    simpl transition_cell.
    rewrite (Hprevious (S prior_index)) by lia.
    rewrite IH by lia.
    rewrite (Hprevious prior_index) by lia.
    unfold reference_cell.
    rewrite (firstn_S_snoc query prior_index default_char Hprior).
    rewrite lev_distance_snoc.
    apply min3_swap_first_two.
Qed.

Theorem streamed_row_refines_reference :
  forall query suffix prefix previous index,
    (forall k, k <= length query ->
      previous k = reference_cell query prefix k) ->
    index <= length query ->
    fold_left (fun row label => transition_cell query label row)
      suffix previous index =
      reference_cell query (prefix ++ suffix) index.
Proof.
  intros query suffix.
  induction suffix as [|label rest IH];
    intros prefix previous index Hprevious Hindex.
  - simpl. rewrite app_nil_r. now apply Hprevious.
  - simpl.
    replace (prefix ++ label :: rest)
      with ((prefix ++ [label]) ++ rest)
      by (rewrite <- app_assoc; reflexivity).
    apply IH; [|exact Hindex].
    intros k Hk.
    now apply one_label_transition_refines_reference.
Qed.

Theorem streamed_terminal_is_reference_edit_distance :
  forall query target,
    terminal_cost query (stream_row query target) =
    lev_distance query target.
Proof.
  intros query target.
  unfold terminal_cost, stream_row.
  pose proof (streamed_row_refines_reference
    query target [] seed_row (length query)
    (seed_row_refines_empty_target query)
    (Nat.le_refl (length query))) as Hterminal.
  simpl in Hterminal.
  unfold reference_cell in Hterminal.
  rewrite firstn_all in Hterminal.
  exact Hterminal.
Qed.

Corollary empty_target_returns_query_length : forall query,
  terminal_cost query (stream_row query []) = length query.
Proof.
  intro query.
  rewrite streamed_terminal_is_reference_edit_distance.
  apply lev_distance_empty_right.
Qed.

Corollary empty_query_returns_target_length : forall target,
  terminal_cost [] (stream_row [] target) = length target.
Proof.
  intro target.
  rewrite streamed_terminal_is_reference_edit_distance.
  apply lev_distance_empty_left.
Qed.

(** Checked finite-integer primitives. Rejection is explicit rather than
    silently wrapping or treating a machine maximum as an exact score. *)
Definition checked_add
    (maximum left right : nat) : option nat :=
  if Nat.leb (left + right) maximum
  then Some (left + right)
  else None.

Definition checked_transition
    (maximum upper horizontal diagonal replacement_weight : nat)
    : option nat :=
  match checked_add maximum upper 1,
        checked_add maximum horizontal 1,
        checked_add maximum diagonal replacement_weight with
  | Some delete_or_insert, Some insert_or_delete, Some consume =>
      Some (min3 delete_or_insert insert_or_delete consume)
  | _, _, _ => None
  end.

Lemma checked_add_sound : forall maximum left right result,
  checked_add maximum left right = Some result ->
  result = left + right /\ result <= maximum.
Proof.
  intros maximum left right result Hchecked.
  unfold checked_add in Hchecked.
  destruct (Nat.leb (left + right) maximum) eqn:Hlimit;
    try discriminate.
  inversion Hchecked; subst.
  split; [reflexivity |].
  apply Nat.leb_le in Hlimit.
  exact Hlimit.
Qed.

Lemma checked_add_complete : forall maximum left right,
  left + right <= maximum ->
  checked_add maximum left right = Some (left + right).
Proof.
  intros maximum left right Hbound.
  unfold checked_add.
  assert (Htest : Nat.leb (left + right) maximum = true)
    by (apply Nat.leb_le; exact Hbound).
  rewrite Htest.
  reflexivity.
Qed.

Theorem checked_transition_refines_exact_minimum :
  forall maximum upper horizontal diagonal replacement_weight result,
    checked_transition maximum upper horizontal diagonal
      replacement_weight = Some result ->
    result = min3 (upper + 1) (horizontal + 1)
      (diagonal + replacement_weight) /\
    result <= maximum.
Proof.
  intros maximum upper horizontal diagonal weight result Hchecked.
  unfold checked_transition in Hchecked.
  destruct (checked_add maximum upper 1) as [up|] eqn:Hup;
    try discriminate.
  destruct (checked_add maximum horizontal 1) as [left|]
    eqn:Hleft; try discriminate.
  destruct (checked_add maximum diagonal weight) as [diagonal_cost|]
    eqn:Hdiagonal; try discriminate.
  inversion Hchecked; subst result.
  apply checked_add_sound in Hup as [Hup_eq Hup_bound].
  apply checked_add_sound in Hleft as [Hleft_eq Hleft_bound].
  apply checked_add_sound in Hdiagonal as
    [Hdiagonal_eq Hdiagonal_bound].
  subst up left diagonal_cost.
  split; [reflexivity |].
  unfold min3. pose proof (Nat.le_min_l (upper + 1)
    (Nat.min (horizontal + 1) (diagonal + weight))).
  lia.
Qed.

Theorem checked_transition_accepts_bounded_candidates :
  forall maximum upper horizontal diagonal replacement_weight,
    upper + 1 <= maximum ->
    horizontal + 1 <= maximum ->
    diagonal + replacement_weight <= maximum ->
    checked_transition maximum upper horizontal diagonal
      replacement_weight =
      Some (min3 (upper + 1) (horizontal + 1)
        (diagonal + replacement_weight)).
Proof.
  intros maximum upper horizontal diagonal weight Hupper Hleft Hdiagonal.
  unfold checked_transition.
  rewrite (checked_add_complete maximum upper 1 Hupper).
  rewrite (checked_add_complete maximum horizontal 1 Hleft).
  rewrite (checked_add_complete maximum diagonal weight Hdiagonal).
  reflexivity.
Qed.

(** A successful checked cell agrees with the corresponding semantic
    prefix cell when its predecessor row has already been certified. The
    horizontal operand is the new row's preceding cell, making the
    coordinate convention explicit. *)
Theorem checked_cell_refines_reference :
  forall query target label previous index maximum result,
    (forall k, k <= length query ->
      previous k = reference_cell query target k) ->
    S index <= length query ->
    checked_transition maximum
      (previous (S index))
      (transition_cell query label previous index)
      (previous index)
      (subst_cost (nth index query default_char) label) = Some result ->
    result = reference_cell query (target ++ [label]) (S index) /\
    result <= maximum.
Proof.
  intros query target label previous index maximum result
    Hprevious Hindex Hchecked.
  destruct (checked_transition_refines_exact_minimum
    maximum (previous (S index))
    (transition_cell query label previous index)
    (previous index)
    (subst_cost (nth index query default_char) label)
    result Hchecked) as [Hresult Hbound].
  split; [|exact Hbound].
  rewrite Hresult.
  change (transition_cell query label previous (S index) =
    reference_cell query (target ++ [label]) (S index)).
  apply one_label_transition_refines_reference; assumption.
Qed.

(** A certificate for this kernel must bind the replacement weight to
    the actual equality predicate. Arbitrary weighted substitutions are
    outside standard unit Levenshtein. *)
Definition accept_unit_substitution_weight
    (query_label target_label : Char) (supplied_weight : nat)
    : option nat :=
  if Nat.eqb supplied_weight (subst_cost query_label target_label)
  then Some supplied_weight
  else None.

Theorem accepted_weight_is_standard :
  forall query_label target_label supplied_weight,
    accept_unit_substitution_weight query_label target_label
      supplied_weight = Some supplied_weight ->
    supplied_weight = subst_cost query_label target_label.
Proof.
  intros query_label target_label supplied_weight Haccepted.
  unfold accept_unit_substitution_weight in Haccepted.
  destruct (Nat.eqb supplied_weight
    (subst_cost query_label target_label)) eqn:Hequal;
    try discriminate.
  apply Nat.eqb_eq in Hequal.
  exact Hequal.
Qed.

Definition control_a : Char := Ascii.ascii_of_nat 0.
Definition control_b : Char := Ascii.ascii_of_nat 1.

Example off_by_one_terminal_read_changes_score :
  terminal_cost [control_a]
      (transition_cell [control_a] control_a seed_row) = 0 /\
  transition_cell [control_a] control_a seed_row 0 = 1.
Proof. split; reflexivity. Qed.

Example unsupported_weight_two_is_rejected :
  subst_cost control_a control_b = 1 /\
  accept_unit_substitution_weight control_a control_b 2 = None.
Proof. split; reflexivity. Qed.

Example overflowing_candidate_is_rejected_even_when_minimum_fits :
  checked_transition 2 2 0 0 0 = None /\
  min3 (2 + 1) (0 + 1) (0 + 0) = 0.
Proof. split; reflexivity. Qed.
