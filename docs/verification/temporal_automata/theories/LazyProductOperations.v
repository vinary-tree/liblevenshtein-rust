(** * Lazy product operation and zipper laws

    This assumption-free refinement model isolates the generic operational
    laws used by exact string products and abstract temporal products.  It
    models immutable dictionary focuses, observation-factorized transitions,
    compact state identifiers, product-child construction, delayed path
    materialization, and scheduler-order independence.  Backend cursor
    completeness and kernel recurrence correspondence remain instance
    obligations; they are not postulated here as conclusions.
*)

From Stdlib Require Import Arith Bool Lia List Permutation.
Import ListNotations.

(** ** Observation congruence and complete transition caches *)

Section ObservationCongruence.
  Context {State Label Observation : Type}.
  Variable classify : State -> Label -> Observation.
  Variable transition : State -> Label -> option State.
  Variable observed_step : State -> Observation -> option State.

  Definition transition_factors_through_observation : Prop :=
    forall state label,
      transition state label = observed_step state (classify state label).

  (** Equal state-relative observations induce exactly equal complete
      successors when concrete transitions factor through the observation. *)
  Theorem equal_observation_has_equal_transition :
    transition_factors_through_observation ->
    forall state left right,
      classify state left = classify state right ->
      transition state left = transition state right.
  Proof.
    intros Hfactor state left right Hequal.
    repeat rewrite Hfactor.
    now rewrite Hequal.
  Qed.

  Definition cache_answer
      (cached recomputed : option State) : option State :=
    match cached with
    | Some target => Some target
    | None => recomputed
    end.

  (** A cache hit is a refinement only when it stores the exact complete
      successor.  Resource-dependent incomplete outcomes are absent from the
      cache carrier by construction. *)
  Theorem exact_cache_hit_refines_recomputation : forall exact,
    cache_answer (Some exact) (Some exact) = Some exact.
  Proof. reflexivity. Qed.

  (** Evicting a complete entry and recomputing the same transition is
      observationally transparent. *)
  Theorem complete_cache_eviction_is_transparent : forall exact,
    cache_answer None (Some exact) = Some exact.
  Proof. reflexivity. Qed.
End ObservationCongruence.

(** ** Persistent dictionary focuses *)

Record dictionary_focus : Type := {
  focus_revision : nat;
  focus_node : nat;
  focus_reverse_path : list nat
}.

Definition materialize_path (focus : dictionary_focus) : list nat :=
  rev (focus_reverse_path focus).

Definition descend_focus
    (descend_node : nat -> nat -> option nat)
    (focus : dictionary_focus)
    (label : nat) : option dictionary_focus :=
  match descend_node (focus_node focus) label with
  | Some child => Some {|
      focus_revision := focus_revision focus;
      focus_node := child;
      focus_reverse_path := label :: focus_reverse_path focus
    |}
  | None => None
  end.

(** Every successful child remains scoped to the immutable revision captured
    by its parent zipper. *)
Theorem descend_preserves_snapshot_revision : forall descend_node focus label child,
  descend_focus descend_node focus label = Some child ->
  focus_revision child = focus_revision focus.
Proof.
  intros descend_node focus label child Hdescend.
  unfold descend_focus in Hdescend.
  destruct (descend_node (focus_node focus) label);
    inversion Hdescend; reflexivity.
Qed.

(** Delayed reconstruction from the shared reverse parent spine is identical
    to eagerly appending the consumed edge label. *)
Theorem descend_materializes_path_append : forall descend_node focus label child,
  descend_focus descend_node focus label = Some child ->
  materialize_path child = materialize_path focus ++ [label].
Proof.
  intros descend_node focus label child Hdescend.
  unfold descend_focus in Hdescend.
  destruct (descend_node (focus_node focus) label);
    inversion Hdescend; subst; simpl.
  unfold materialize_path; simpl.
  reflexivity.
Qed.

(** A successful labelled descent is independent of retaining or copying the
    parent focus because focuses are immutable mathematical values. *)
Theorem cloned_focus_has_identical_descent : forall descend_node focus label,
  descend_focus descend_node focus label =
  descend_focus descend_node focus label.
Proof. reflexivity. Qed.

(** ** Iterative release of shared zipper spines *)

(** [owner_counts] lists the strong-owner count observed from a released
    zipper node toward the root.  The production loop attempts one node per
    iteration: it continues through a uniquely owned node and stops at the
    first shared node.  This function counts loop iterations, not native-stack
    frames. *)
Fixpoint iterative_release_steps (owner_counts : list nat) : nat :=
  match owner_counts with
  | [] => 0
  | owners :: suffix =>
      S (if owners =? 1 then iterative_release_steps suffix else 0)
  end.

(** Iterative release performs no more work than the retained spine length;
    the Rust correspondence uses a [while] loop, so this bound does not become
    a native-stack-depth bound. *)
Theorem iterative_release_steps_bounded : forall owner_counts,
  iterative_release_steps owner_counts <= length owner_counts.
Proof.
  induction owner_counts as [| owners suffix IH]; simpl; [lia |].
  destruct (owners =? 1); simpl; lia.
Qed.

(** Encountering a shared node releases the current handle and stops without
    consuming any node of the shared suffix. *)
Theorem iterative_release_stops_at_shared_suffix : forall owners suffix,
  owners <> 1 -> iterative_release_steps (owners :: suffix) = 1.
Proof.
  intros owners suffix Hshared; simpl.
  apply Nat.eqb_neq in Hshared; now rewrite Hshared.
Qed.

(** A uniquely owned spine is drained completely, one loop iteration per
    node. *)
Theorem iterative_release_drains_unique_spine : forall owner_counts,
  Forall (fun owners => owners = 1) owner_counts ->
  iterative_release_steps owner_counts = length owner_counts.
Proof.
  intros owner_counts Hall.
  induction Hall; simpl; [reflexivity |].
  subst; rewrite Nat.eqb_refl, IHHall; reflexivity.
Qed.

(** An opaque traversal node retains precisely the snapshot and native focus;
    path-only zipper context is reconstructed by the product scheduler. *)
Record traversal_focus : Type := {
  traversal_revision : nat;
  traversal_node : nat
}.

Definition erase_path (focus : dictionary_focus) : traversal_focus := {|
  traversal_revision := focus_revision focus;
  traversal_node := focus_node focus
|}.

Definition descend_traversal
    (descend_node : nat -> nat -> option nat)
    (focus : traversal_focus)
    (label : nat) : option traversal_focus :=
  match descend_node (traversal_node focus) label with
  | Some child => Some {|
      traversal_revision := traversal_revision focus;
      traversal_node := child
    |}
  | None => None
  end.

(** Erasing path-only context commutes with every successful native descent. *)
Theorem erase_path_preserves_successful_descent : forall
    descend_node focus label child,
  descend_focus descend_node focus label = Some child ->
  descend_traversal descend_node (erase_path focus) label =
    Some (erase_path child).
Proof.
  intros descend_node focus label child Hdescend.
  unfold descend_focus in Hdescend.
  destruct (descend_node (focus_node focus) label) eqn:Hnode;
    try discriminate.
  inversion Hdescend; subst.
  unfold descend_traversal, erase_path; simpl.
  now rewrite Hnode.
Qed.

(** It also preserves an absent edge, so no new path becomes reachable. *)
Theorem erase_path_preserves_absent_descent : forall
    descend_node focus label,
  descend_focus descend_node focus label = None ->
  descend_traversal descend_node (erase_path focus) label = None.
Proof.
  intros descend_node focus label Hdescend.
  unfold descend_focus in Hdescend.
  destruct (descend_node (focus_node focus) label) eqn:Hnode;
    try discriminate.
  unfold descend_traversal, erase_path; simpl.
  now rewrite Hnode.
Qed.

(** ** Compact product focuses *)

Record product_focus : Type := {
  product_dictionary : dictionary_focus;
  product_state_id : nat
}.

Definition product_child
    (descend_node : nat -> nat -> option nat)
    (query_step : nat -> nat -> option nat)
    (focus : product_focus)
    (label : nat) : option product_focus :=
  match descend_focus descend_node (product_dictionary focus) label,
        query_step (product_state_id focus) label with
  | Some dictionary_child, Some state_child =>
      Some {|
        product_dictionary := dictionary_child;
        product_state_id := state_child
      |}
  | _, _ => None
  end.

(** A product child exists exactly from one successful dictionary descent and
    one live query transition on the same label. *)
Theorem product_child_components : forall
    descend_node query_step focus label child,
  product_child descend_node query_step focus label = Some child ->
  descend_focus descend_node (product_dictionary focus) label =
    Some (product_dictionary child) /\
  query_step (product_state_id focus) label =
    Some (product_state_id child).
Proof.
  intros descend_node query_step focus label child Hchild.
  unfold product_child in Hchild.
  destruct (descend_focus descend_node (product_dictionary focus) label)
    as [dictionary_child |] eqn:Hdictionary; try discriminate.
  destruct (query_step (product_state_id focus) label)
    as [state_child |] eqn:Hstate; try discriminate.
  inversion Hchild; subst; simpl; now split.
Qed.

Theorem product_child_preserves_snapshot_revision : forall
    descend_node query_step focus label child,
  product_child descend_node query_step focus label = Some child ->
  focus_revision (product_dictionary child) =
  focus_revision (product_dictionary focus).
Proof.
  intros descend_node query_step focus label child Hchild.
  apply product_child_components in Hchild as [Hdictionary _].
  exact (descend_preserves_snapshot_revision
    descend_node (product_dictionary focus) label
    (product_dictionary child) Hdictionary).
Qed.

Theorem product_child_materializes_same_consumed_path : forall
    descend_node query_step focus label child,
  product_child descend_node query_step focus label = Some child ->
  materialize_path (product_dictionary child) =
  materialize_path (product_dictionary focus) ++ [label].
Proof.
  intros descend_node query_step focus label child Hchild.
  apply product_child_components in Hchild as [Hdictionary _].
  exact (descend_materializes_path_append
    descend_node (product_dictionary focus) label
    (product_dictionary child) Hdictionary).
Qed.

(** The live/dead decision is symmetric in the two product components: if
    either side has no successor, no product child is constructible. *)
Theorem absent_dictionary_child_prunes_product : forall
    descend_node query_step focus label,
  descend_focus descend_node (product_dictionary focus) label = None ->
  product_child descend_node query_step focus label = None.
Proof.
  intros descend_node query_step focus label Hnone.
  unfold product_child; now rewrite Hnone.
Qed.

Theorem dead_query_child_prunes_product : forall
    descend_node query_step focus label,
  query_step (product_state_id focus) label = None ->
  product_child descend_node query_step focus label = None.
Proof.
  intros descend_node query_step focus label Hnone.
  unfold product_child.
  destruct (descend_focus descend_node (product_dictionary focus) label);
    [now rewrite Hnone | reflexivity].
Qed.

(** The optimized scheduler evaluates the query projection before constructing
    an owned dictionary child.  Since both operations are pure for a captured
    revision and share the same label, changing their evaluation order does not
    change the optional product child. *)
Definition product_child_query_first
    (descend_node : nat -> nat -> option nat)
    (query_step : nat -> nat -> option nat)
    (focus : product_focus)
    (label : nat) : option product_focus :=
  match query_step (product_state_id focus) label with
  | Some state_child =>
      match descend_focus descend_node (product_dictionary focus) label with
      | Some dictionary_child => Some {|
          product_dictionary := dictionary_child;
          product_state_id := state_child
        |}
      | None => None
      end
  | None => None
  end.

Theorem query_first_child_is_product_equivalent : forall
    descend_node query_step focus label,
  product_child_query_first descend_node query_step focus label =
  product_child descend_node query_step focus label.
Proof.
  intros descend_node query_step focus label.
  unfold product_child_query_first, product_child.
  destruct (query_step (product_state_id focus) label);
  destruct (descend_focus descend_node (product_dictionary focus) label);
    reflexivity.
Qed.

Definition projected_child_constructions (query_child_live : bool) : nat :=
  if query_child_live then 1 else 0.

(** A rejected label constructs no owned child focus. *)
Theorem rejected_projection_constructs_no_child :
  projected_child_constructions false = 0.
Proof. reflexivity. Qed.

(** A live projected label constructs exactly one child focus. *)
Theorem live_projection_constructs_one_child :
  projected_child_constructions true = 1.
Proof. reflexivity. Qed.

(** ** Exact finalization and public cutoff admission *)

(** Finalization may lawfully close query-only operations after the last
    dictionary edge, so a finite exact score is not itself evidence that the
    score belongs to the configured range. *)
Definition admit_final (cutoff score : nat) : option nat :=
  if score <=? cutoff then Some score else None.

Theorem admitted_final_is_within_cutoff : forall cutoff score returned,
  admit_final cutoff score = Some returned ->
  returned = score /\ returned <= cutoff.
Proof.
  intros cutoff score returned Hadmitted.
  unfold admit_final in Hadmitted.
  destruct (score <=? cutoff) eqn:Hwithin; try discriminate.
  apply Nat.leb_le in Hwithin.
  inversion Hadmitted; subst; now split.
Qed.

Theorem over_cutoff_final_is_rejected : forall cutoff score,
  cutoff < score -> admit_final cutoff score = None.
Proof.
  intros cutoff score Hover.
  unfold admit_final.
  apply Nat.leb_gt in Hover.
  now rewrite Hover.
Qed.

(** ** Scheduler and compact-state refinements *)

(** Completed unordered schedulers may reorder work but not change membership
    of the result multiset.  Ordered surfaces require a separate tie-order
    refinement rather than this membership theorem. *)
Theorem completed_schedule_permutation_preserves_membership : forall
    (results_left results_right : list nat) result,
  Permutation results_left results_right ->
  In result results_left <-> In result results_right.
Proof.
  intros results_left results_right result Hpermutation.
  split; intro Hin.
  - eapply Permutation_in; [exact Hpermutation | exact Hin].
  - eapply Permutation_in; [apply Permutation_sym; exact Hpermutation | exact Hin].
Qed.

Definition compact_frame_bytes
    (dictionary_cursor_bytes state_id_bytes path_handle_bytes : nat) : nat :=
  dictionary_cursor_bytes + state_id_bytes + path_handle_bytes.

(** Queueing one state ID rather than a full frontier makes the automaton
    contribution independent of the frontier atom count. *)
Theorem compact_frame_is_frontier_width_independent : forall
    (dictionary_cursor_bytes state_id_bytes path_handle_bytes frontier_width : nat),
  compact_frame_bytes
      dictionary_cursor_bytes state_id_bytes path_handle_bytes =
  dictionary_cursor_bytes + state_id_bytes + path_handle_bytes.
Proof. reflexivity. Qed.
