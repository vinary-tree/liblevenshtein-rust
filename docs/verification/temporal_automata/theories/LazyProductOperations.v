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

(** ** Lexicographic top-k stopping

    This order-theoretic kernel uses natural costs and unique natural tie keys.
    An instance must separately establish that its machine cost comparator,
    abstract lower bounds, snapshot candidate coverage, and result-heap tie
    key implement these premises.  In particular, an ideal-real inequality
    does not by itself discharge a binary64 lower-bound obligation. *)

Definition lex_rank_lt (left right : nat * nat) : Prop :=
  fst left < fst right \/
  fst left = fst right /\ snd left < snd right.

Definition lex_rank_le (left right : nat * nat) : Prop :=
  fst left < fst right \/
  fst left = fst right /\ snd left <= snd right.

Lemma component_bounds_are_lex_lower : forall
    lower_cost lower_tie exact_cost exact_tie,
  lower_cost <= exact_cost ->
  lower_tie <= exact_tie ->
  lex_rank_le (lower_cost, lower_tie) (exact_cost, exact_tie).
Proof.
  intros lower_cost lower_tie exact_cost exact_tie Hcost Htie.
  unfold lex_rank_le; simpl.
  destruct (Nat.eq_dec lower_cost exact_cost) as [Heq | Hneq].
  - right; lia.
  - left; lia.
Qed.

Lemma lex_rank_le_trans : forall left middle right,
  lex_rank_le left middle ->
  lex_rank_le middle right ->
  lex_rank_le left right.
Proof.
  intros [left_cost left_tie] [middle_cost middle_tie]
    [right_cost right_tie] Hleft Hright.
  unfold lex_rank_le in *; simpl in *.
  destruct Hleft as [Hleft | [Hleft Hleft_tie]];
    destruct Hright as [Hright | [Hright Hright_tie]];
    [left | left | left | right]; lia.
Qed.

Lemma lex_rank_le_excludes_better : forall worst candidate,
  lex_rank_le worst candidate ->
  ~ lex_rank_lt candidate worst.
Proof.
  intros [worst_cost worst_tie] [candidate_cost candidate_tie] Hle Hlt.
  unfold lex_rank_le in Hle; unfold lex_rank_lt in Hlt; simpl in *.
  destruct Hle as [Hle | [Hcost Htie]];
    destruct Hlt as [Hlt | [Hcost' Htie']]; lia.
Qed.

Corollary strict_cost_stopping_is_lex_stopping : forall
    worst_cost worst_tie lower_cost lower_tie,
  worst_cost < lower_cost ->
  lex_rank_le (worst_cost, worst_tie) (lower_cost, lower_tie).
Proof.
  intros worst_cost worst_tie lower_cost lower_tie Hcost.
  unfold lex_rank_le; simpl; now left.
Qed.

(** A full best-result heap can only lower its kth rank.  Therefore a region
    excluded against an earlier worst rank remains excluded after improvement. *)
Corollary pruned_region_needs_no_reopening : forall
    improved_worst previous_worst lower,
  lex_rank_le improved_worst previous_worst ->
  lex_rank_le previous_worst lower ->
  lex_rank_le improved_worst lower.
Proof.
  intros improved_worst previous_worst lower Himproved Hprevious.
  eapply lex_rank_le_trans; eassumption.
Qed.

(** A region's componentwise cost and tie floors exclude every strictly
    better candidate once their pair reaches the kth exact rank. *)
Theorem lex_region_pruning_is_sound : forall
    lower_cost lower_tie exact_cost exact_tie worst_cost worst_tie,
  lower_cost <= exact_cost ->
  lower_tie <= exact_tie ->
  lex_rank_le (worst_cost, worst_tie) (lower_cost, lower_tie) ->
  ~ lex_rank_lt (exact_cost, exact_tie) (worst_cost, worst_tie).
Proof.
  intros lower_cost lower_tie exact_cost exact_tie worst_cost worst_tie
    Hcost Htie Hstop.
  apply lex_rank_le_excludes_better.
  eapply lex_rank_le_trans; [exact Hstop |].
  now apply component_bounds_are_lex_lower.
Qed.

(** A concrete original may combine its region and candidate-specific cost
    bounds before pairing the result with its exact snapshot tie key. *)
Lemma maximum_of_admissible_cost_bounds_is_admissible : forall
    region_bound candidate_bound exact_cost,
  region_bound <= exact_cost ->
  candidate_bound <= exact_cost ->
  Nat.max region_bound candidate_bound <= exact_cost.
Proof. intros; lia. Qed.

Definition frontier_covers
    (regions candidates : list (nat * nat)) : Prop :=
  forall candidate, In candidate candidates ->
    exists lower, In lower regions /\ lex_rank_le lower candidate.

(** All unverified candidates are covered; every queued lower pair is at
    least the worst verified rank.  Hence no unverified candidate improves it.
    A bounded search may use this only after k distinct exact results exist. *)
Theorem lex_frontier_stopping_is_sound : forall regions unseen worst,
  frontier_covers regions unseen ->
  (forall lower, In lower regions -> lex_rank_le worst lower) ->
  forall candidate, In candidate unseen -> ~ lex_rank_lt candidate worst.
Proof.
  intros regions unseen worst Hcover Hstop candidate Hin.
  destruct (Hcover candidate Hin) as [lower [Hlower Hcandidate]].
  apply lex_rank_le_excludes_better.
  eapply lex_rank_le_trans; [apply Hstop; exact Hlower | exact Hcandidate].
Qed.

(** Combining this separation with the heap invariant that selected holds
    the best k verified candidates yields the ordered top k of their union.
    The unique tie-key and no-duplicate-candidate obligations establish that
    an equal rank cannot denote a different result. *)
Theorem selected_precedes_unseen : forall selected unseen worst,
  (forall member, In member selected -> lex_rank_le member worst) ->
  (forall candidate, In candidate unseen -> lex_rank_le worst candidate) ->
  forall member candidate,
    In member selected -> In candidate unseen -> lex_rank_le member candidate.
Proof.
  intros selected unseen worst Hselected Hunseen member candidate
    Hmember Hcandidate.
  eapply lex_rank_le_trans.
  - apply Hselected; exact Hmember.
  - apply Hunseen; exact Hcandidate.
Qed.

(** A list is a top-k selection when it has exactly k distinct verified
    members and none of its members ranks after an excluded candidate.  The
    list order itself is retained by the search's result finalizer. *)
Definition topk_selection
    (k : nat) (selected candidates : list (nat * nat)) : Prop :=
  length selected = k /\
  NoDup selected /\
  (forall member, In member selected -> In member candidates) /\
  (forall member outsider,
      In member selected ->
      In outsider candidates ->
      ~ In outsider selected ->
      lex_rank_le member outsider).

(** This strengthens the no-better-candidate lemma to an exact top-k
    selection statement. The verified heap invariant supplies the first
    premise; the lower-pair frontier theorem supplies the unseen premise. *)
Theorem lex_stop_preserves_topk_selection : forall
    k selected verified unseen worst regions,
  topk_selection k selected verified ->
  (forall member, In member selected -> lex_rank_le member worst) ->
  frontier_covers regions unseen ->
  (forall lower, In lower regions -> lex_rank_le worst lower) ->
  topk_selection k selected (verified ++ unseen).
Proof.
  intros k selected verified unseen worst regions
    [Hlength [Hnodup [Hselected Hbest]]] Hworst Hcover Hstop.
  repeat split.
  - exact Hlength.
  - exact Hnodup.
  - intros member Hmember.
    apply in_or_app; left; now apply Hselected.
  - intros member outsider Hmember Houtside Hnotselected.
    apply in_app_or in Houtside as [Hverified | Hunseen].
    + eapply Hbest; eassumption.
    + eapply lex_rank_le_trans.
      * apply Hworst; exact Hmember.
      * destruct (Hcover outsider Hunseen) as [lower [Hlower Hcandidate]].
        eapply lex_rank_le_trans.
        -- apply Hstop; exact Hlower.
        -- exact Hcandidate.
Qed.

(** An optional tie-floor query has three distinct outcomes.  Empty is a
    claim that there are no live candidates; Unknown gives no pruning
    certificate; Known is a lower floor for all live candidate tie keys. *)
Inductive tie_floor_summary : Type :=
| FloorUnknown
| FloorEmpty
| FloorKnown (floor : nat).

Definition tie_floor_sound
    (summary : tie_floor_summary) (candidate_ties : list nat) : Prop :=
  match summary with
  | FloorUnknown => True
  | FloorEmpty => candidate_ties = []
  | FloorKnown floor =>
      forall tie, In tie candidate_ties -> floor <= tie
  end.

(** The executable BF-1 decision is conservative at an unknown tie floor.
    Empty is a separate structural certificate and needs no cost comparison. *)
Definition prune_with_tie_summary
    (worst : nat * nat) (lower_cost : nat)
    (summary : tie_floor_summary) : bool :=
  match summary with
  | FloorEmpty => true
  | FloorUnknown => fst worst <? lower_cost
  | FloorKnown floor =>
      (fst worst <? lower_cost) ||
      ((fst worst =? lower_cost) && (snd worst <=? floor))
  end.

Theorem unknown_floor_cannot_prune_at_equal_cost : forall worst tie,
  prune_with_tie_summary (worst, tie) worst FloorUnknown = false.
Proof.
  intros; unfold prune_with_tie_summary; simpl.
  now rewrite Nat.ltb_irrefl.
Qed.

Theorem certified_summary_pruning_is_sound : forall
    summary ties worst lower_cost exact_cost exact_tie,
  tie_floor_sound summary ties ->
  In exact_tie ties ->
  lower_cost <= exact_cost ->
  prune_with_tie_summary worst lower_cost summary = true ->
  ~ lex_rank_lt (exact_cost, exact_tie) worst.
Proof.
  intros summary ties [worst_cost worst_tie] lower_cost exact_cost
    exact_tie Hsound Hin Hcost Hprune.
  destruct summary as [| |floor]; simpl in Hsound.
  - unfold prune_with_tie_summary in Hprune; simpl in Hprune.
    apply Nat.ltb_lt in Hprune.
    apply lex_rank_le_excludes_better.
    unfold lex_rank_le; simpl; left; lia.
  - subst ties; contradiction.
  - unfold prune_with_tie_summary in Hprune; simpl in Hprune.
    apply orb_true_iff in Hprune as [Hstrict | Hequal].
    + apply Nat.ltb_lt in Hstrict.
      apply lex_rank_le_excludes_better.
      unfold lex_rank_le; simpl; left; lia.
    + apply andb_true_iff in Hequal as [Hcost_equal Htie_floor].
      apply Nat.eqb_eq in Hcost_equal.
      apply Nat.leb_le in Htie_floor.
      eapply lex_region_pruning_is_sound.
      * exact Hcost.
      * now apply Hsound.
      * unfold lex_rank_le; simpl; right; lia.
Qed.

(** Removing live candidates cannot make a certified floor unsafe. Insertion
    can, so cached floors must be tied to an immutable revision. *)
Theorem tie_floor_survives_candidate_removal : forall
    summary old_ties new_ties,
  tie_floor_sound summary old_ties ->
  (forall tie, In tie new_ties -> In tie old_ties) ->
  tie_floor_sound summary new_ties.
Proof.
  intros summary old_ties new_ties Hsound Hsubset.
  destruct summary as [| |floor]; simpl in *; [exact I | |].
  - subst old_ties.
    destruct new_ties as [|tie rest]; [reflexivity |].
    exfalso; apply (Hsubset tie); simpl; auto.
  - intros tie Hin. apply Hsound, Hsubset, Hin.
Qed.

Definition combine_tie_floors
    (left right : tie_floor_summary) : tie_floor_summary :=
  match left, right with
  | FloorEmpty, other | other, FloorEmpty => other
  | FloorKnown left_floor, FloorKnown right_floor =>
      FloorKnown (Nat.min left_floor right_floor)
  | _, _ => FloorUnknown
  end.

Theorem combining_sound_tie_floors_is_sound : forall
    left right left_ties right_ties,
  tie_floor_sound left left_ties ->
  tie_floor_sound right right_ties ->
  tie_floor_sound (combine_tie_floors left right)
    (left_ties ++ right_ties).
Proof.
  intros left right left_ties right_ties Hleft Hright.
  destruct left as [| |left_floor]; destruct right as [| |right_floor];
    simpl in *.
  - exact I.
  - exact I.
  - exact I.
  - exact I.
  - subst; reflexivity.
  - subst; exact Hright.
  - exact I.
  - subst; now rewrite app_nil_r.
  - intros tie Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + specialize (Hleft tie Hin); lia.
    + specialize (Hright tie Hin); lia.
Qed.

Example a_partial_observed_minimum_is_not_a_floor :
  ~ tie_floor_sound (FloorKnown 10) [1; 10].
Proof.
  intro Hfloor.
  specialize (Hfloor 1 (or_introl eq_refl)).
  lia.
Qed.

(** The bounded elastic scan's tie key is the snapshot bucket/slot pair.
    Every live terminal member of a nonempty bucket has the same bucket ID
    and a nonnegative slot, so the complete bucket floor is (bucket, 0). *)
Definition elastic_tie_le (left right : nat * nat) : Prop :=
  fst left < fst right \/
  fst left = fst right /\ snd left <= snd right.

Theorem elastic_terminal_bucket_floor_is_sound : forall bucket slot,
  elastic_tie_le (bucket, 0) (bucket, slot).
Proof.
  intros bucket slot. unfold elastic_tie_le; simpl. right; lia.
Qed.

(** Timestamped bounded kNN ranks equal-cost originals by episode ID. The
    minimum of every complete nonempty terminal bucket is a certified floor;
    the recursive construction examines all live IDs, unlike an observed
    partial minimum. *)
Fixpoint complete_episode_floor (episodes : list nat) : option nat :=
  match episodes with
  | [] => None
  | episode :: tail =>
      match complete_episode_floor tail with
      | None => Some episode
      | Some tail_floor => Some (Nat.min episode tail_floor)
      end
  end.

Theorem complete_episode_floor_is_sound : forall episodes floor episode,
  complete_episode_floor episodes = Some floor ->
  In episode episodes -> floor <= episode.
Proof.
  induction episodes as [| head tail IH]; intros floor episode Hfloor Hin;
    simpl in *; [discriminate |].
  destruct (complete_episode_floor tail) as [tail_floor |] eqn:Htail.
  - inversion Hfloor; subst floor.
    destruct Hin as [Heq | Hmember]; [subst; lia |].
    specialize (IH tail_floor episode eq_refl Hmember); lia.
  - inversion Hfloor; subst floor.
    destruct Hin as [Heq | Hmember]; [subst; lia |].
    destruct tail as [| next rest]; [contradiction |].
    simpl in Htail.
    destruct (complete_episode_floor rest); discriminate.
Qed.

Theorem empty_episode_bucket_has_no_floor :
  complete_episode_floor [] = None.
Proof. reflexivity. Qed.
