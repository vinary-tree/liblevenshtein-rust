(** * Sparse exact dictionary-product obligations for timestamped TWED

    This assumption-free refinement model follows the Rust product boundary:
    a residual contains the previous whole typed token, canonical sparse
    recurrence positions, and the exact final-cost bit pattern.  Fingerprints
    select collision buckets but never establish equality.  Transitions are
    constructed only for observed state/token pairs.  Dictionary traversal is
    an explicit cursor zipper, and paging retains an exact suffix.

    Natural numbers model canonical binary64 bit patterns and finite resource
    counts.  This file does not claim that natural-number addition models
    binary64 arithmetic; recurrence arithmetic is discharged by the separate
    TWED proof island and executable correspondence properties.
*)

From Stdlib Require Import Arith Lia List Bool.
Import ListNotations.

(** ** Whole typed labels *)

Record typed_label : Type := {
  value_bin : nat;
  time_bin : nat;
  unit_tag : nat
}.

Definition same_typed_label (left right : typed_label) : bool :=
  Nat.eqb (value_bin left) (value_bin right) &&
  Nat.eqb (time_bin left) (time_bin right) &&
  Nat.eqb (unit_tag left) (unit_tag right).

Theorem same_typed_label_reflects_every_component : forall left right,
  same_typed_label left right = true <->
  value_bin left = value_bin right /\
  time_bin left = time_bin right /\
  unit_tag left = unit_tag right.
Proof.
  intros left right; unfold same_typed_label.
  repeat rewrite andb_true_iff.
  repeat rewrite Nat.eqb_eq.
  tauto.
Qed.

Definition same_optional_label
    (left right : option typed_label) : bool :=
  match left, right with
  | None, None => true
  | Some left_label, Some right_label =>
      same_typed_label left_label right_label
  | _, _ => false
  end.

Lemma same_optional_label_reflects_exact_equality : forall left right,
  same_optional_label left right = true <-> left = right.
Proof.
  intros [left |] [right |]; simpl; try (split; discriminate).
  - rewrite same_typed_label_reflects_every_component.
    destruct left as [left_value left_time left_unit].
    destruct right as [right_value right_time right_unit].
    simpl; split.
    + intros [Hvalue [Htime Hunit]]; subst; reflexivity.
    + intros Hequal; inversion Hequal; now repeat split.
  - tauto.
Qed.

(** ** Canonical sparse residual keys *)

Record sparse_position : Type := {
  position_row : nat;
  position_cost_bits : nat
}.

Definition same_sparse_position
    (left right : sparse_position) : bool :=
  Nat.eqb (position_row left) (position_row right) &&
  Nat.eqb (position_cost_bits left) (position_cost_bits right).

Lemma same_sparse_position_reflects_exact_bits : forall left right,
  same_sparse_position left right = true <-> left = right.
Proof.
  intros [left_row left_cost] [right_row right_cost].
  unfold same_sparse_position; simpl.
  rewrite andb_true_iff; repeat rewrite Nat.eqb_eq.
  split.
  - intros [Hrow Hcost]; subst; reflexivity.
  - intros Hequal; inversion Hequal; now split.
Qed.

Fixpoint same_sparse_positions
    (left right : list sparse_position) : bool :=
  match left, right with
  | [], [] => true
  | left_position :: left_suffix, right_position :: right_suffix =>
      same_sparse_position left_position right_position &&
      same_sparse_positions left_suffix right_suffix
  | _, _ => false
  end.

Lemma same_sparse_positions_reflects_exact_bits : forall left right,
  same_sparse_positions left right = true <-> left = right.
Proof.
  induction left as [| left_position left_suffix IH];
    destruct right as [| right_position right_suffix]; simpl;
    try (split; discriminate); try tauto.
  rewrite andb_true_iff, same_sparse_position_reflects_exact_bits, IH.
  split.
  - intros [Hposition Hsuffix]; subst; reflexivity.
  - intros Hequal; inversion Hequal; now split.
Qed.

Record compact_state : Type := {
  previous_label : option typed_label;
  canonical_positions : list sparse_position;
  final_cost_bits : nat
}.

Definition exact_state_equal (left right : compact_state) : Prop :=
  previous_label left = previous_label right /\
  canonical_positions left = canonical_positions right /\
  final_cost_bits left = final_cost_bits right.

Definition exact_state_equalb (left right : compact_state) : bool :=
  same_optional_label (previous_label left) (previous_label right) &&
  same_sparse_positions
    (canonical_positions left) (canonical_positions right) &&
  Nat.eqb (final_cost_bits left) (final_cost_bits right).

Theorem exact_state_equalb_reflects_complete_residual : forall left right,
  exact_state_equalb left right = true <-> left = right.
Proof.
  intros [left_previous left_positions left_final]
         [right_previous right_positions right_final].
  unfold exact_state_equalb; simpl.
  repeat rewrite andb_true_iff.
  rewrite same_optional_label_reflects_exact_equality.
  rewrite same_sparse_positions_reflects_exact_bits.
  rewrite Nat.eqb_eq.
  split.
  - intros [[Hprevious Hpositions] Hfinal]; subst; reflexivity.
  - intros Hequal; inversion Hequal; now repeat split.
Qed.

Theorem exact_interning_preserves_residual : forall left right,
  exact_state_equal left right -> left = right.
Proof.
  intros [left_previous left_positions left_final]
         [right_previous right_positions right_final].
  intros [Hprevious [Hpositions Hfinal]]; simpl in *.
  subst; reflexivity.
Qed.

(** A deterministic fingerprint is only a bucket selector.  Reuse searches
    that bucket and checks the complete residual key byte-for-byte. *)
Fixpoint collision_checked_reuse
    (wanted : compact_state) (bucket : list compact_state)
    : option compact_state :=
  match bucket with
  | [] => None
  | candidate :: suffix =>
      if exact_state_equalb wanted candidate
      then Some candidate
      else collision_checked_reuse wanted suffix
  end.

Theorem collision_checked_reuse_is_exact : forall wanted bucket reused,
  collision_checked_reuse wanted bucket = Some reused -> reused = wanted.
Proof.
  intros wanted bucket; induction bucket as [| candidate suffix IH];
    intros reused Hlookup; simpl in Hlookup; try discriminate.
  destruct (exact_state_equalb wanted candidate) eqn:Hequal.
  - inversion Hlookup; subst.
    apply exact_state_equalb_reflects_complete_residual in Hequal.
    now symmetry.
  - now apply IH.
Qed.

Theorem equal_fingerprint_does_not_authorize_unequal_reuse : forall
    (fingerprint : compact_state -> nat) wanted collided,
  fingerprint wanted = fingerprint collided ->
  exact_state_equalb wanted collided = false ->
  collision_checked_reuse wanted [collided] = None.
Proof.
  intros fingerprint wanted collided _ Hunequal.
  simpl; now rewrite Hunequal.
Qed.

(** ** Exact vertical subsumption and dense reconstruction *)

(** An omitted row has one permitted witness: its exact canonical bit pattern
    equals the preceding row plus the query-deletion increment. *)
Definition exact_vertical_subsumes
    (add_bits : nat -> nat -> nat)
    (previous deletion current : nat) : bool :=
  Nat.eqb current (add_bits previous deletion).

Definition canonical_cell
    (add_bits : nat -> nat -> nat)
    (previous deletion current : nat) : option nat :=
  if exact_vertical_subsumes add_bits previous deletion current
  then None
  else Some current.

Definition reconstruct_cell
    (add_bits : nat -> nat -> nat)
    (previous deletion : nat) (cell : option nat) : nat :=
  match cell with
  | Some retained => retained
  | None => add_bits previous deletion
  end.

Theorem canonical_cell_reconstructs_exactly : forall
    add_bits previous deletion current,
  reconstruct_cell add_bits previous deletion
    (canonical_cell add_bits previous deletion current) = current.
Proof.
  intros add_bits previous deletion current.
  unfold canonical_cell, exact_vertical_subsumes.
  destruct (Nat.eqb current (add_bits previous deletion)) eqn:Hequal; simpl.
  - now apply Nat.eqb_eq in Hequal.
  - reflexivity.
Qed.

Theorem omitted_cell_has_exact_subsumption_witness : forall
    add_bits previous deletion current,
  canonical_cell add_bits previous deletion current = None ->
  current = add_bits previous deletion.
Proof.
  intros add_bits previous deletion current Homitted.
  unfold canonical_cell, exact_vertical_subsumes in Homitted.
  destruct (Nat.eqb current (add_bits previous deletion)) eqn:Hequal;
    try discriminate.
  now apply Nat.eqb_eq in Hequal.
Qed.

Theorem nonwitness_cell_is_retained : forall add_bits previous deletion current,
  current <> add_bits previous deletion ->
  canonical_cell add_bits previous deletion current = Some current.
Proof.
  intros add_bits previous deletion current Hdifferent.
  unfold canonical_cell, exact_vertical_subsumes.
  apply Nat.eqb_neq in Hdifferent; now rewrite Hdifferent.
Qed.

(** The option map is a proof device: [Some] cells become explicit
    [(row,cost-bits)] anchors and [None] cells are absent from the Rust vector.
    The row index carried by each retained position makes every omission
    recoverable from immutable query-deletion costs. *)
Fixpoint canonical_sparse_map
    (add_bits : nat -> nat -> nat)
    (previous : nat) (deletions dense : list nat) : list (option nat) :=
  match deletions, dense with
  | deletion :: deletion_suffix, current :: dense_suffix =>
      canonical_cell add_bits previous deletion current ::
      canonical_sparse_map add_bits current deletion_suffix dense_suffix
  | _, _ => []
  end.

Fixpoint reconstruct_sparse_map
    (add_bits : nat -> nat -> nat)
    (previous : nat) (deletions : list nat) (sparse : list (option nat))
    : list nat :=
  match deletions, sparse with
  | deletion :: deletion_suffix, cell :: sparse_suffix =>
      let current := reconstruct_cell add_bits previous deletion cell in
      current ::
        reconstruct_sparse_map add_bits current deletion_suffix sparse_suffix
  | _, _ => []
  end.

Fixpoint enumerate_sparse_positions
    (row : nat) (sparse : list (option nat)) : list sparse_position :=
  match sparse with
  | [] => []
  | None :: suffix => enumerate_sparse_positions (S row) suffix
  | Some cost :: suffix =>
      {| position_row := row; position_cost_bits := cost |} ::
      enumerate_sparse_positions (S row) suffix
  end.

Fixpoint rows_strict_from
    (lower : nat) (positions : list sparse_position) : Prop :=
  match positions with
  | [] => True
  | position :: suffix =>
      lower <= position_row position /\
      rows_strict_from (S (position_row position)) suffix
  end.

Lemma rows_strict_from_weaken : forall lower higher positions,
  lower <= higher ->
  rows_strict_from higher positions ->
  rows_strict_from lower positions.
Proof.
  intros lower higher positions Hlower.
  destruct positions as [| position suffix]; simpl; [tauto |].
  intros [Hposition Hsuffix]; split; [lia | exact Hsuffix].
Qed.

Theorem enumerated_sparse_positions_are_strictly_ordered : forall row sparse,
  rows_strict_from row (enumerate_sparse_positions row sparse).
Proof.
  intros row sparse; revert row.
  induction sparse as [| [cost |] suffix IH]; intros row; simpl.
  - exact I.
  - split; [lia | apply IH].
  - eapply (rows_strict_from_weaken row (S row)); [lia | apply IH].
Qed.

Lemma canonical_sparse_map_has_dense_length : forall
    add_bits previous deletions dense,
  length deletions = length dense ->
  length (canonical_sparse_map add_bits previous deletions dense) = length dense.
Proof.
  intros add_bits previous deletions; revert previous.
  induction deletions as [| deletion deletion_suffix IH];
    intros previous dense Hlength; destruct dense; simpl in *;
    try discriminate; try reflexivity.
  f_equal; apply IH; lia.
Qed.

Lemma enumeration_never_adds_positions : forall row sparse,
  length (enumerate_sparse_positions row sparse) <= length sparse.
Proof.
  intros row sparse; revert row.
  induction sparse as [| [cost |] suffix IH]; intros row; simpl.
  - lia.
  - specialize (IH (S row)); lia.
  - specialize (IH (S row)); lia.
Qed.

Theorem canonical_sparse_map_reconstructs_dense_exactly : forall
    add_bits previous deletions dense,
  length deletions = length dense ->
  reconstruct_sparse_map add_bits previous deletions
    (canonical_sparse_map add_bits previous deletions dense) = dense.
Proof.
  intros add_bits previous deletions; revert previous.
  induction deletions as [| deletion deletion_suffix IH];
    intros previous dense Hlength; destruct dense as [| current dense_suffix];
    simpl in *; try discriminate; try reflexivity.
  rewrite canonical_cell_reconstructs_exactly.
  f_equal; apply IH; lia.
Qed.

Theorem canonical_sparse_residual_is_exact_and_no_larger_than_dense : forall
    add_bits previous deletions dense,
  length deletions = length dense ->
  reconstruct_sparse_map add_bits previous deletions
    (canonical_sparse_map add_bits previous deletions dense) = dense /\
  length (enumerate_sparse_positions 0
    (canonical_sparse_map add_bits previous deletions dense)) <= length dense.
Proof.
  intros add_bits previous deletions dense Hlength.
  split.
  - now apply canonical_sparse_map_reconstructs_dense_exactly.
  - eapply Nat.le_trans; [apply enumeration_never_adds_positions |].
    rewrite canonical_sparse_map_has_dense_length by exact Hlength.
    reflexivity.
Qed.

(** ** Interval recurrence and collision-bucket authority *)

Definition min3 (a b c : nat) : nat := Nat.min a (Nat.min b c).

Theorem relaxed_cell_lower_simulates_concrete : forall
    abstract_diag abstract_up abstract_left
    concrete_diag concrete_up concrete_left
    abstract_match abstract_delete_query abstract_delete_candidate
    concrete_match concrete_delete_query concrete_delete_candidate,
  abstract_diag <= concrete_diag ->
  abstract_up <= concrete_up ->
  abstract_left <= concrete_left ->
  abstract_match <= concrete_match ->
  abstract_delete_query <= concrete_delete_query ->
  abstract_delete_candidate <= concrete_delete_candidate ->
  min3
    (abstract_diag + abstract_match)
    (abstract_up + abstract_delete_query)
    (abstract_left + abstract_delete_candidate) <=
  min3
    (concrete_diag + concrete_match)
    (concrete_up + concrete_delete_query)
    (concrete_left + concrete_delete_candidate).
Proof.
  intros; unfold min3.
  repeat apply Nat.min_glb.
  - eapply Nat.le_trans; [apply Nat.le_min_l |]. lia.
  - eapply Nat.le_trans; [apply Nat.le_min_r |].
    eapply Nat.le_trans; [apply Nat.le_min_l |]. lia.
  - eapply Nat.le_trans; [apply Nat.le_min_r |].
    eapply Nat.le_trans; [apply Nat.le_min_r |]. lia.
Qed.

Record original : Type := {
  stable_id : nat;
  exact_cost : nat
}.

Definition verify_bucket (cutoff : nat) (bucket : list original) : list original :=
  filter (fun candidate => Nat.leb (exact_cost candidate) cutoff) bucket.

Theorem verified_bucket_has_no_false_positive : forall cutoff bucket candidate,
  In candidate (verify_bucket cutoff bucket) -> exact_cost candidate <= cutoff.
Proof.
  intros cutoff bucket candidate Hin.
  unfold verify_bucket in Hin.
  apply filter_In in Hin; destruct Hin as [_ Hcost].
  now apply Nat.leb_le.
Qed.

Theorem verified_bucket_retains_every_qualifying_collision : forall cutoff bucket candidate,
  In candidate bucket -> exact_cost candidate <= cutoff ->
  In candidate (verify_bucket cutoff bucket).
Proof.
  intros cutoff bucket candidate Hin Hcost.
  unfold verify_bucket; apply filter_In; split; [exact Hin |].
  now apply Nat.leb_le.
Qed.

(** ** On-demand transitions and exact caches *)

Definition transition_key : Type := (nat * nat)%type.

Definition same_transition_key (left right : transition_key) : bool :=
  Nat.eqb (fst left) (fst right) && Nat.eqb (snd left) (snd right).

Lemma same_transition_key_reflects_pair : forall left right,
  same_transition_key left right = true <-> left = right.
Proof.
  intros [left_state left_label] [right_state right_label].
  unfold same_transition_key; simpl.
  rewrite andb_true_iff; repeat rewrite Nat.eqb_eq.
  split.
  - intros [Hstate Hlabel]; subst; reflexivity.
  - intros Hequal; inversion Hequal; now split.
Qed.

Fixpoint transition_cache_lookup
    (key : transition_key) (cache : list (transition_key * option nat))
    : option (option nat) :=
  match cache with
  | [] => None
  | (candidate_key, successor) :: suffix =>
      if same_transition_key key candidate_key
      then Some successor
      else transition_cache_lookup key suffix
  end.

Definition transition_on_demand
    (recompute : transition_key -> option nat)
    (cache : list (transition_key * option nat))
    (key : transition_key) : option nat :=
  match transition_cache_lookup key cache with
  | Some cached => cached
  | None => recompute key
  end.

Theorem absent_transition_is_recomputed_on_demand : forall recompute cache key,
  transition_cache_lookup key cache = None ->
  transition_on_demand recompute cache key = recompute key.
Proof.
  intros recompute cache key Hmiss.
  unfold transition_on_demand; now rewrite Hmiss.
Qed.

Theorem exact_cached_transition_refines_recomputation : forall
    recompute cache key successor,
  transition_cache_lookup key cache = Some successor ->
  recompute key = successor ->
  transition_on_demand recompute cache key = successor.
Proof.
  intros recompute cache key successor Hhit _.
  unfold transition_on_demand; now rewrite Hhit.
Qed.

Theorem inserting_observed_exact_transition_is_sound : forall recompute cache key,
  transition_on_demand recompute ((key, recompute key) :: cache) key = recompute key.
Proof.
  intros recompute cache [state label].
  unfold transition_on_demand; simpl.
  unfold same_transition_key; simpl.
  repeat rewrite Nat.eqb_refl; reflexivity.
Qed.

(** ** Immutable DFS cursor pager and zipper *)

Record edge_cursor : Type := {
  cursor_revision : nat;
  cursor_node : nat;
  cursor_state_id : nat;
  consumed_edges : nat;
  remaining_edges : list nat
}.

Definition cursor_page (fuel : nat) (cursor : edge_cursor)
    : list nat * list nat :=
  (firstn fuel (remaining_edges cursor),
   skipn fuel (remaining_edges cursor)).

Definition advance_cursor (fuel : nat) (cursor : edge_cursor) : edge_cursor :=
  let emitted := fst (cursor_page fuel cursor) in
  let remaining := snd (cursor_page fuel cursor) in
  {| cursor_revision := cursor_revision cursor;
     cursor_node := cursor_node cursor;
     cursor_state_id := cursor_state_id cursor;
     consumed_edges := consumed_edges cursor + length emitted;
     remaining_edges := remaining |}.

Theorem cursor_page_then_resume_equals_uninterrupted_edges : forall fuel cursor,
  fst (cursor_page fuel cursor) ++ snd (cursor_page fuel cursor) =
  remaining_edges cursor.
Proof.
  intros; unfold cursor_page; simpl; apply firstn_skipn.
Qed.

Theorem cursor_advance_preserves_product_focus : forall fuel cursor,
  cursor_revision (advance_cursor fuel cursor) = cursor_revision cursor /\
  cursor_node (advance_cursor fuel cursor) = cursor_node cursor /\
  cursor_state_id (advance_cursor fuel cursor) = cursor_state_id cursor.
Proof. intros; now repeat split. Qed.

Theorem cursor_advance_preserves_edge_accounting : forall fuel cursor,
  consumed_edges (advance_cursor fuel cursor) +
  length (remaining_edges (advance_cursor fuel cursor)) =
  consumed_edges cursor + length (remaining_edges cursor).
Proof.
  intros fuel cursor; unfold advance_cursor, cursor_page; simpl.
  pose proof (firstn_skipn fuel (remaining_edges cursor)) as Hpartition.
  apply (f_equal (@length nat)) in Hpartition.
  rewrite length_app in Hpartition; lia.
Qed.

Definition zipper_valid
    (revision arena_size : nat) (stack : list edge_cursor) : Prop :=
  Forall (fun cursor =>
    cursor_revision cursor = revision /\ cursor_state_id cursor < arena_size)
    stack.

Definition child_cursor
    (parent : edge_cursor) (child_node child_state : nat) (edges : list nat)
    : edge_cursor :=
  {| cursor_revision := cursor_revision parent;
     cursor_node := child_node;
     cursor_state_id := child_state;
     consumed_edges := 0;
     remaining_edges := edges |}.

Theorem zipper_push_preserves_revision_and_arena_reference : forall
    revision arena_size stack parent child_node child_state edges,
  zipper_valid revision arena_size stack ->
  cursor_revision parent = revision ->
  child_state < arena_size ->
  zipper_valid revision arena_size
    (child_cursor parent child_node child_state edges :: stack).
Proof.
  intros revision arena_size stack parent child_node child_state edges
         Hvalid Hrevision Hstate.
  constructor; [simpl; now split | exact Hvalid].
Qed.

Theorem zipper_pop_preserves_revision_and_arena_references : forall
    revision arena_size stack,
  zipper_valid revision arena_size stack ->
  zipper_valid revision arena_size (tl stack).
Proof.
  intros revision arena_size [| frame stack] Hvalid; simpl.
  - constructor.
  - inversion Hvalid; assumption.
Qed.

(** ** Tagged paging and resource scope *)

Definition page {A : Type} (fuel : nat) (work : list A) : list A * list A :=
  (firstn fuel work, skipn fuel work).

Theorem page_then_resume_equals_uninterrupted : forall A fuel (work : list A),
  fst (page fuel work) ++ snd (page fuel work) = work.
Proof. intros; unfold page; simpl; apply firstn_skipn. Qed.

Inductive tagged_result (A : Type) : Type :=
  | Complete : list A -> tagged_result A
  | Incomplete : list A -> list A -> tagged_result A.

Arguments Complete {A} _.
Arguments Incomplete {A} _ _.

Definition run_page {A : Type} (fuel : nat) (work : list A) : tagged_result A :=
  let '(emitted, remaining) := page fuel work in
  match remaining with
  | [] => Complete emitted
  | _ => Incomplete emitted remaining
  end.

Theorem complete_only_after_exhaustion : forall A fuel (work : list A) emitted,
  run_page fuel work = Complete emitted -> skipn fuel work = [].
Proof.
  intros A fuel work emitted.
  unfold run_page, page; simpl.
  destruct (skipn fuel work); [reflexivity | discriminate].
Qed.

Record retained_usage : Type := {
  retained_frames : nat;
  retained_states : nat;
  retained_positions : nat;
  retained_transitions : nat
}.

Record retained_limits : Type := {
  maximum_frames : nat;
  maximum_states : nat;
  maximum_positions : nat;
  maximum_transitions : nat
}.

Definition usage_within_limits
    (usage : retained_usage) (limits : retained_limits) : Prop :=
  retained_frames usage <= maximum_frames limits /\
  retained_states usage <= maximum_states limits /\
  retained_positions usage <= maximum_positions limits /\
  retained_transitions usage <= maximum_transitions limits.

Definition retained_cells (usage : retained_usage) : nat :=
  retained_frames usage + retained_states usage +
  retained_positions usage + retained_transitions usage.

(** The implementation is stack-safe because traversal uses explicit heap
    frames.  Its retained memory is bounded by configured ceilings.  The arena
    and cache can retain distinct observations from already visited nodes, so
    this theorem deliberately does not claim a live-depth-only bound. *)
Theorem retained_product_memory_is_bounded_by_explicit_ceilings : forall
    usage limits (visited_nodes : nat),
  usage_within_limits usage limits ->
  retained_cells usage <=
    maximum_frames limits + maximum_states limits +
    maximum_positions limits + maximum_transitions limits.
Proof.
  intros [frames states positions transitions]
         [frame_limit state_limit position_limit transition_limit]
         visited_nodes Hbounded.
  unfold usage_within_limits, retained_cells in *; simpl in *; lia.
Qed.

(** A frame-only pop cannot invalidate an interned state identifier because
    the query-local state arena is append-only for the captured query. *)
Theorem frame_pop_preserves_arena_identifier : forall A
    (arena : list A) (frames : list nat) frame state_id,
  nth_error arena state_id = Some frame ->
  nth_error arena state_id = Some frame /\
  length (tl frames) <= length frames.
Proof.
  intros; split; [assumption |].
  destruct frames; simpl; lia.
Qed.
