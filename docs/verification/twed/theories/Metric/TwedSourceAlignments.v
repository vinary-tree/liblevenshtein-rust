(** * Finite paths for the 2007 infinite-axis TWED recurrence

    The 2007 Marteau submission (arXiv:cs/0703033v5, Equation (9)) initializes
    every positive one-empty axis cell to infinity. Its first finite edit
    therefore matches the first samples. In the anchored transfer those
    first samples are the common anchors. Later edits retain the source
    predecessor context. This module models finite lawful paths only:

      https://arxiv.org/pdf/cs/0703033v5#page=15

    [lawful_path i j cost] means that exactly [i] left and [j] right samples
    have been consumed at exact natural-number cost [cost]. The natural
    carrier gives a finite path witness and is not a proof of the real
    sequence metric, recurrence/path equality, or binary64 correspondence. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Definition timed_sample : Type := (nat * nat)%type.
Definition sample_value (sample : timed_sample) : nat := fst sample.
Definition sample_time (sample : timed_sample) : nat := snd sample.
Definition source_zero_predecessor : timed_sample := (0, 0).
Definition common_anchor : timed_sample := (0, 1).

(** Source-domain admission is separate from edit-path construction. For
    the common anchor, this predicate requires every later timestamp to
    be strictly greater than its predecessor. *)
Fixpoint times_after
    (previous_time : nat) (tail : list timed_sample) : Prop :=
  match tail with
  | [] => True
  | current :: rest =>
      previous_time < sample_time current /\
      times_after (sample_time current) rest
  end.

Definition well_timed_anchor_tail (tail : list timed_sample) : Prop :=
  times_after (sample_time common_anchor) tail.

Definition point_distance (left right : timed_sample) : nat :=
  (Nat.max (sample_value left) (sample_value right) -
   Nat.min (sample_value left) (sample_value right)) +
  (Nat.max (sample_time left) (sample_time right) -
   Nat.min (sample_time left) (sample_time right)).

(** For the first source sample the report uses a conventional zero
    predecessor. At every later index the predecessor must actually be
    present in the source. A missing timestamp is therefore not invented. *)
Definition predecessor_at
    (source : nat -> option timed_sample) (next_index : nat)
    : option timed_sample :=
  match next_index with
  | 0 => Some source_zero_predecessor
  | S previous_index => source previous_index
  end.

Definition delete_charge
    (gap_penalty : nat) (current previous : timed_sample) : nat :=
  point_distance current previous + gap_penalty.

Definition match_charge
    (left_current left_previous right_current right_previous : timed_sample)
    : nat :=
  point_distance left_current right_current +
  point_distance left_previous right_previous.

Section Paths.
  Variables left right : nat -> option timed_sample.
  Variable gap_penalty : nat.

  (** Both deletion constructors require the other coordinate to be
      positive. This makes the source's infinite positive axes absent from
      the finite-path relation, rather than representing infinity by a
      large finite number. The lookup premises retain exact predecessor
      samples and timestamps for every charged step. *)
  Inductive lawful_path : nat -> nat -> nat -> Prop :=
  | PathOrigin : lawful_path 0 0 0
  | PathDeleteLeft : forall i j cost current previous,
      lawful_path i (S j) cost ->
      left i = Some current ->
      predecessor_at left i = Some previous ->
      lawful_path (S i) (S j)
        (cost + delete_charge gap_penalty current previous)
  | PathInsertRight : forall i j cost current previous,
      lawful_path (S i) j cost ->
      right j = Some current ->
      predecessor_at right j = Some previous ->
      lawful_path (S i) (S j)
        (cost + delete_charge gap_penalty current previous)
  | PathMatch : forall i j cost
      left_current left_previous right_current right_previous,
      lawful_path i j cost ->
      left i = Some left_current ->
      predecessor_at left i = Some left_previous ->
      right j = Some right_current ->
      predecessor_at right j = Some right_previous ->
      lawful_path (S i) (S j)
        (cost + match_charge
          left_current left_previous right_current right_previous).

  (** One last edit, with its predecessor path and all lookup evidence.
      This is a useful elimination interface for later DP proofs. *)
  Inductive last_step : nat -> nat -> nat -> Prop :=
  | LastDeleteLeft : forall i j cost current previous,
      lawful_path i (S j) cost ->
      left i = Some current ->
      predecessor_at left i = Some previous ->
      last_step (S i) (S j)
        (cost + delete_charge gap_penalty current previous)
  | LastInsertRight : forall i j cost current previous,
      lawful_path (S i) j cost ->
      right j = Some current ->
      predecessor_at right j = Some previous ->
      last_step (S i) (S j)
        (cost + delete_charge gap_penalty current previous)
  | LastMatch : forall i j cost
      left_current left_previous right_current right_previous,
      lawful_path i j cost ->
      left i = Some left_current ->
      predecessor_at left i = Some left_previous ->
      right j = Some right_current ->
      predecessor_at right j = Some right_previous ->
      last_step (S i) (S j)
        (cost + match_charge
          left_current left_previous right_current right_previous).

  Theorem lawful_path_decomposes : forall i j cost,
    lawful_path i j cost ->
    (i = 0 /\ j = 0 /\ cost = 0) \/ last_step i j cost.
  Proof.
    intros i j cost Hpath.
    destruct Hpath.
    - left; repeat split; reflexivity.
    - right; eapply LastDeleteLeft; eauto.
    - right; eapply LastInsertRight; eauto.
    - right; eapply LastMatch; eauto.
  Qed.

  Theorem last_step_reconstructs_path : forall i j cost,
    last_step i j cost -> lawful_path i j cost.
  Proof.
    intros i j cost Hstep.
    destruct Hstep.
    - eapply PathDeleteLeft; eauto.
    - eapply PathInsertRight; eauto.
    - eapply PathMatch; eauto.
  Qed.

  Corollary positive_cell_iff_last_step : forall i j cost,
    lawful_path (S i) (S j) cost <-> last_step (S i) (S j) cost.
  Proof.
    intros i j cost; split.
    - intro Hpath.
      destruct (lawful_path_decomposes _ _ _ Hpath)
        as [[Hi [_ _]] | Hstep]; [lia | exact Hstep].
    - apply last_step_reconstructs_path.
  Qed.

  Theorem finite_path_boundary_shape : forall i j cost,
    lawful_path i j cost ->
    (i = 0 /\ j = 0) \/ (0 < i /\ 0 < j).
  Proof.
    intros i j cost Hpath.
    destruct Hpath.
    - left; auto.
    - right; lia.
    - right; lia.
    - right; lia.
  Qed.

  Corollary positive_source_axes_have_no_finite_path : forall i cost,
    ~ lawful_path (S i) 0 cost /\
    ~ lawful_path 0 (S i) cost.
  Proof.
    intros i cost; split; intro Hpath.
    - destruct (finite_path_boundary_shape _ _ _ Hpath)
        as [[Hleft Hright] | [Hleft Hright]]; lia.
    - destruct (finite_path_boundary_shape _ _ _ Hpath)
        as [[Hleft Hright] | [Hleft Hright]]; lia.
  Qed.

  (** Every consumed left sample had a real predecessor lookup, except
      index zero whose conventional predecessor is explicit above. *)
  Theorem path_left_predecessors_exist : forall i j cost,
    lawful_path i j cost ->
    forall k, k < i ->
      exists previous, predecessor_at left k = Some previous.
  Proof.
    intros i j cost Hpath.
    induction Hpath as
      [|i j cost current previous Hpath IH Hcurrent Hprevious
       |i j cost current previous Hpath IH Hcurrent Hprevious
       |i j cost left_current left_previous right_current right_previous
          Hpath IH Hleft Hleft_previous Hright Hright_previous];
      intros k Hlt.
    - lia.
    - destruct (Nat.eq_dec k i) as [Heq | Hneq].
      + subst k. exists previous. exact Hprevious.
      + apply IH. lia.
    - apply IH. exact Hlt.
    - destruct (Nat.eq_dec k i) as [Heq | Hneq].
      + subst k. exists left_previous. exact Hleft_previous.
      + apply IH. lia.
  Qed.

  Lemma dense_predecessor_exists : forall source bound index,
    (forall k, k < bound -> exists point, source k = Some point) ->
    index < bound ->
    exists previous, predecessor_at source index = Some previous.
  Proof.
    intros source bound [|previous_index] Hdense Hlt.
    - exists source_zero_predecessor. reflexivity.
    - simpl. apply Hdense. lia.
  Qed.

  Lemma extend_left_by : forall bound count i j,
    (forall k, k < bound -> exists point, left k = Some point) ->
    i + count <= bound ->
    (exists cost, lawful_path i (S j) cost) ->
    exists cost, lawful_path (i + count) (S j) cost.
  Proof.
    intros bound count; induction count as [|count IH];
      intros i j Hdense Hbound [cost Hpath].
    - replace (i + 0) with i by lia. now exists cost.
    - assert (Hlt : i < bound) by lia.
      destruct (Hdense i Hlt) as [current Hcurrent].
      destruct (dense_predecessor_exists left bound i Hdense Hlt)
        as [previous Hprevious].
      assert (Hnext : exists next_cost,
        lawful_path (S i) (S j) next_cost).
      { exists (cost + delete_charge gap_penalty current previous).
        eapply PathDeleteLeft; eauto. }
      replace (i + S count) with (S i + count) by lia.
      apply (IH (S i) j Hdense); [lia | exact Hnext].
  Qed.

  Lemma extend_right_by : forall bound count i j,
    (forall k, k < bound -> exists point, right k = Some point) ->
    j + count <= bound ->
    (exists cost, lawful_path (S i) j cost) ->
    exists cost, lawful_path (S i) (j + count) cost.
  Proof.
    intros bound count; induction count as [|count IH];
      intros i j Hdense Hbound [cost Hpath].
    - replace (j + 0) with j by lia. now exists cost.
    - assert (Hlt : j < bound) by lia.
      destruct (Hdense j Hlt) as [current Hcurrent].
      destruct (dense_predecessor_exists right bound j Hdense Hlt)
        as [previous Hprevious].
      assert (Hnext : exists next_cost,
        lawful_path (S i) (S j) next_cost).
      { exists (cost + delete_charge gap_penalty current previous).
        eapply PathInsertRight; eauto. }
      replace (j + S count) with (S j + count) by lia.
      apply (IH i (S j) Hdense); [lia | exact Hnext].
  Qed.

  Theorem dense_finite_inputs_have_path : forall left_len right_len,
    0 < left_len -> 0 < right_len ->
    (forall i, i < left_len -> exists point, left i = Some point) ->
    (forall j, j < right_len -> exists point, right j = Some point) ->
    exists cost, lawful_path left_len right_len cost.
  Proof.
    intros left_len right_len Hleft_len Hright_len Hleft_dense Hright_dense.
    destruct (Hleft_dense 0 Hleft_len) as [first_left Hfirst_left].
    destruct (Hright_dense 0 Hright_len) as [first_right Hfirst_right].
    assert (Hfirst : exists cost, lawful_path 1 1 cost).
    { exists (match_charge first_left source_zero_predecessor
        first_right source_zero_predecessor).
      eapply PathMatch with (i := 0) (j := 0) (cost := 0)
        (left_current := first_left)
        (left_previous := source_zero_predecessor)
        (right_current := first_right)
        (right_previous := source_zero_predecessor);
        simpl; try reflexivity; try assumption.
      constructor. }
    assert (Hleft_bound : 1 + (left_len - 1) <= left_len) by lia.
    destruct (extend_left_by left_len (left_len - 1) 1 0
      Hleft_dense Hleft_bound Hfirst) as [left_cost Hleft_path].
    replace (1 + (left_len - 1)) with left_len in Hleft_path by lia.
    assert (Hright_start : exists cost,
      lawful_path (S (left_len - 1)) 1 cost).
    { replace (S (left_len - 1)) with left_len by lia.
      now exists left_cost. }
    assert (Hright_bound : 1 + (right_len - 1) <= right_len) by lia.
    destruct (extend_right_by right_len (right_len - 1)
      (left_len - 1) 1 Hright_dense Hright_bound Hright_start)
      as [final_cost Hfinal].
    replace (S (left_len - 1)) with left_len in Hfinal by lia.
    replace (1 + (right_len - 1)) with right_len in Hfinal by lia.
    now exists final_cost.
  Qed.
End Paths.

Definition list_source (series : list timed_sample)
    : nat -> option timed_sample :=
  fun index => nth_error series index.

Lemma list_source_is_dense : forall series index,
  index < length series ->
  exists point, list_source series index = Some point.
Proof.
  intros series; induction series as [|head tail IH];
    intros [|index] Hlt; unfold list_source in *; simpl in *; try lia.
  - exists head. reflexivity.
  - apply IH. lia.
Qed.

(** An empty physical tail is allowed: its source sequence is the one
    common anchor. The theorem supplies a finite alignment for every pair
    of such finite anchored lists, with no metric claim. *)
Theorem finite_anchored_lists_have_path :
  forall gap_penalty left_tail right_tail,
    exists cost,
      lawful_path
        (list_source (common_anchor :: left_tail))
        (list_source (common_anchor :: right_tail))
        gap_penalty
        (length (common_anchor :: left_tail))
        (length (common_anchor :: right_tail)) cost.
Proof.
  intros gap_penalty left_tail right_tail.
  eapply dense_finite_inputs_have_path.
  - simpl; lia.
  - simpl; lia.
  - intros index Hlt. now apply list_source_is_dense.
  - intros index Hlt. now apply list_source_is_dense.
Qed.

Corollary well_timed_anchored_lists_have_path :
  forall gap_penalty left_tail right_tail,
    well_timed_anchor_tail left_tail ->
    well_timed_anchor_tail right_tail ->
    exists cost,
      lawful_path
        (list_source (common_anchor :: left_tail))
        (list_source (common_anchor :: right_tail))
        gap_penalty
        (length (common_anchor :: left_tail))
        (length (common_anchor :: right_tail)) cost.
Proof.
  intros gap_penalty left_tail right_tail Hleft_times Hright_times.
  apply finite_anchored_lists_have_path.
Qed.

(** A malformed sparse source can have a current point at index two while
    the timestamp-bearing predecessor at index one is absent. This is not
    a valid finite anchored list. The path guard rejects consuming that
    point, even though the current lookup succeeds. *)
Definition missing_predecessor_source (index : nat)
    : option timed_sample :=
  match index with
  | 0 => Some common_anchor
  | 2 => Some (1, 3)
  | _ => None
  end.

Lemma current_exists_but_predecessor_is_absent :
  missing_predecessor_source 2 = Some (1, 3) /\
  predecessor_at missing_predecessor_source 2 = None.
Proof. split; reflexivity. Qed.

Theorem missing_predecessor_timestamp_rejects_consumption :
  forall right gap_penalty j cost,
    ~ lawful_path missing_predecessor_source right gap_penalty 3 j cost.
Proof.
  intros right gap_penalty j cost Hpath.
  assert (Hindex : 2 < 3) by lia.
  destruct (@path_left_predecessors_exist
    missing_predecessor_source right gap_penalty 3 j cost
    Hpath 2 Hindex) as [previous Hprevious].
  simpl in Hprevious. discriminate.
Qed.
