(** * Lazy weighted frontiers for temporal automata

    This assumption-free development isolates the reusable correctness spine
    shared by exact temporal automata.  It deliberately says nothing about an
    unproved kernel-specific subsumption heuristic.  A position may eliminate
    another position only when an epsilon path witnesses a simulation.

    Costs are natural numbers in this refinement model.  The production
    implementation uses validated non-negative binary64 values; its exact
    representable-domain and rounding obligations are executable properties,
    not a claim that binary64 addition is associative.
*)

From Stdlib Require Import Arith Bool Lia List Permutation.
Import ListNotations.

(** A transition label is either epsilon (it consumes no target unit) or one
    concrete target-unit class. *)
Inductive transition_label : Type :=
  | Epsilon
  | Consume (label : nat).

(** [run step source word target cost] is a finite, stack-independent path in
    a weighted transition system.  The inductive definition is mathematical;
    the Rust evaluator uses an explicit worklist. *)
Inductive run
    (step : nat -> transition_label -> nat -> nat -> Prop)
    : nat -> list nat -> nat -> nat -> Prop :=
  | RunRefl : forall state, run step state [] state 0
  | RunEpsilon : forall source middle target edge_cost suffix_cost word,
      step source Epsilon middle edge_cost ->
      run step middle word target suffix_cost ->
      run step source word target (edge_cost + suffix_cost)
  | RunConsume : forall source middle target edge_cost suffix_cost label word,
      step source (Consume label) middle edge_cost ->
      run step middle word target suffix_cost ->
      run step source (label :: word) target (edge_cost + suffix_cost).

(** An epsilon run can be prepended to every suffix run without changing the
    consumed word. *)
Theorem epsilon_run_composition : forall step source middle target prefix_cost suffix_cost word,
  run step source [] middle prefix_cost ->
  run step middle word target suffix_cost ->
  run step source word target (prefix_cost + suffix_cost).
Proof.
  intros step source middle target prefix_cost suffix_cost word Hprefix.
  remember (@nil nat) as empty eqn:Hempty.
  induction Hprefix; intros Hsuffix; subst.
  - simpl. exact Hsuffix.
  - replace (edge_cost + suffix_cost0 + suffix_cost)
      with (edge_cost + (suffix_cost0 + suffix_cost)) by lia.
    eapply RunEpsilon; [exact H |].
    apply IHHprefix; [reflexivity | exact Hsuffix].
  - discriminate Hempty.
Qed.

(** Position [p] dominates [q] only when a concrete epsilon run reaches [q]
    and the accumulated cost through that run is no greater than [q]'s own
    accumulated cost. *)
Definition epsilon_dominates
    (step : nat -> transition_label -> nat -> nat -> Prop)
    (p_cost p q_cost q : nat) : Prop :=
  exists reach_cost,
    run step p [] q reach_cost /\ p_cost + reach_cost <= q_cost.

(** The load-bearing pruning theorem: every accepting continuation of a
    dominated position has a no-more-expensive accepting continuation from its
    dominator, over exactly the same future target word. *)
Theorem epsilon_dominance_is_residual_simulation : forall
    step p_cost p q_cost q final word suffix_cost,
  epsilon_dominates step p_cost p q_cost q ->
  run step q word final suffix_cost ->
  exists simulated_cost,
    run step p word final simulated_cost /\
    p_cost + simulated_cost <= q_cost + suffix_cost.
Proof.
  intros step p_cost p q_cost q final word suffix_cost
         [reach_cost [Hreach Hcost]] Hsuffix.
  exists (reach_cost + suffix_cost).
  split.
  - eapply epsilon_run_composition; eauto.
  - lia.
Qed.

(** Production canonicalization recognizes the cheapest and most local proof
    island: when a cell is exactly its immediate vertical epsilon predecessor
    plus the kernel's non-negative deletion/merge/link cost, retaining that
    later cell adds no residual behavior. *)
Theorem equal_vertical_cell_is_epsilon_dominated : forall
    step p q edge_cost p_cost q_cost,
  step p Epsilon q edge_cost ->
  q_cost = p_cost + edge_cost ->
  epsilon_dominates step p_cost p q_cost q.
Proof.
  intros step p q edge_cost p_cost q_cost Hstep Hequal.
  exists edge_cost; split.
  - replace edge_cost with (edge_cost + 0) by lia.
    eapply RunEpsilon; [exact Hstep | apply RunRefl].
  - lia.
Qed.

(** Strict dominance removes equivalence cycles: equivalent residual
    representatives do not eliminate each other merely because both directions
    are simulable. *)
Definition strict_dominates
    (dominates : nat -> nat -> bool) (left right : nat) : bool :=
  dominates left right && negb (dominates right left).

Definition canonical_member
    (dominates : nat -> nat -> bool) (universe : list nat) (candidate : nat) : bool :=
  negb (existsb (fun other => strict_dominates dominates other candidate) universe).

Definition canonical_frontier
    (dominates : nat -> nat -> bool) (universe : list nat) : list nat :=
  filter (canonical_member dominates universe) (nodup Nat.eq_dec universe).

Lemma existsb_permutation : forall
    (predicate : nat -> bool) (left right : list nat),
  Permutation left right -> existsb predicate left = existsb predicate right.
Proof.
  intros predicate left right Hperm.
  induction Hperm; simpl; try reflexivity.
  - now rewrite IHHperm.
  - destruct (predicate x), (predicate y), (existsb predicate l); reflexivity.
  - now rewrite IHHperm1, IHHperm2.
Qed.

Lemma canonical_member_permutation : forall dominates left right candidate,
  Permutation left right ->
  canonical_member dominates left candidate =
  canonical_member dominates right candidate.
Proof.
  intros dominates left right candidate Hperm.
  unfold canonical_member.
  now rewrite (existsb_permutation
    (fun other => strict_dominates dominates other candidate) left right Hperm).
Qed.

(** Canonicalization is independent of candidate-generation order.  The list
    order may differ, but membership in the interned state key is identical;
    the Rust boundary additionally sorts by the total position order. *)
Theorem canonical_frontier_permutation_invariant : forall dominates left right candidate,
  Permutation left right ->
  In candidate (canonical_frontier dominates left) <->
  In candidate (canonical_frontier dominates right).
Proof.
  intros dominates left right candidate Hperm.
  unfold canonical_frontier.
  repeat rewrite filter_In.
  repeat rewrite nodup_In.
  split; intros [Hin Hcanonical]; split.
  - eapply Permutation_in; eauto.
  - rewrite <- (canonical_member_permutation dominates left right candidate Hperm).
    exact Hcanonical.
  - eapply Permutation_in; [apply Permutation_sym; exact Hperm | exact Hin].
  - rewrite (canonical_member_permutation dominates left right candidate Hperm).
    exact Hcanonical.
Qed.

(** Query-local state identifiers are representation-only names for canonical
    frontiers.  If two generated candidate collections have the same members
    up to permutation, interning either collection denotes exactly the same
    residual automaton state.  This is the semantic condition required by the
    Rust collision-checked fingerprint table: fingerprints may collide, while
    canonical frontier equality remains authoritative. *)
Definition same_interned_frontier
    (dominates : nat -> nat -> bool) (left right : list nat) : Prop :=
  forall candidate,
    In candidate (canonical_frontier dominates left) <->
    In candidate (canonical_frontier dominates right).

Theorem canonical_interning_is_permutation_sound : forall dominates left right,
  Permutation left right ->
  same_interned_frontier dominates left right.
Proof.
  intros dominates left right Hpermutation candidate.
  now apply canonical_frontier_permutation_invariant.
Qed.

(** Reusing an interned identifier is sound only after exact canonical-key
    equality.  A hash/fingerprint equality is intentionally absent from this
    theorem and therefore cannot justify state reuse by itself. *)
Theorem exact_canonical_key_reuse_is_sound : forall
    dominates left right,
  canonical_frontier dominates left = canonical_frontier dominates right ->
  same_interned_frontier dominates left right.
Proof.
  intros dominates left right Hequal candidate.
  unfold same_interned_frontier.
  now rewrite Hequal.
Qed.

(** Every retained representative has no strict dominator in the generated
    universe. *)
Theorem canonical_frontier_is_antichain : forall dominates universe left right,
  In left (canonical_frontier dominates universe) ->
  In right (canonical_frontier dominates universe) ->
  strict_dominates dominates left right = false.
Proof.
  intros dominates universe left right Hleft Hright.
  unfold canonical_frontier in Hleft, Hright.
  apply filter_In in Hleft as [Hleft_in _].
  apply nodup_In in Hleft_in.
  apply filter_In in Hright as [Hright_in Hright_canonical].
  apply nodup_In in Hright_in.
  unfold canonical_member in Hright_canonical.
  apply negb_true_iff in Hright_canonical.
  destruct (strict_dominates dominates left right) eqn:Hstrict;
    [exfalso | reflexivity].
  assert (existsb
      (fun other => strict_dominates dominates other right)
      universe = true) as Hexists.
  { apply existsb_exists. exists left. now split. }
  rewrite Hexists in Hright_canonical. discriminate.
Qed.

(** The additive recurrence used by ERP, TWED, MSM, and DTW is represented by
    three incoming candidates. *)
Definition min3 (left middle right : nat) : nat :=
  Nat.min left (Nat.min middle right).

Lemma min3_monotone : forall a b c x y z,
  a <= x -> b <= y -> c <= z -> min3 a b c <= min3 x y z.
Proof.
  intros a b c x y z Ha Hb Hc.
  unfold min3.
  apply Nat.min_le_compat; [exact Ha |].
  now apply Nat.min_le_compat.
Qed.

Definition additive_cell
    (diagonal above left substitution deletion insertion : nat) : nat :=
  min3 (diagonal + substitution) (above + deletion) (left + insertion).

(** One interval-relaxed additive cell lower-bounds its concrete cell when
    predecessor cells and all local operation costs are lower bounds. *)
Theorem interval_additive_step_is_lower_simulation : forall
    abstract_diagonal abstract_above abstract_left
    concrete_diagonal concrete_above concrete_left
    abstract_substitution abstract_deletion abstract_insertion
    concrete_substitution concrete_deletion concrete_insertion,
  abstract_diagonal <= concrete_diagonal ->
  abstract_above <= concrete_above ->
  abstract_left <= concrete_left ->
  abstract_substitution <= concrete_substitution ->
  abstract_deletion <= concrete_deletion ->
  abstract_insertion <= concrete_insertion ->
  additive_cell
    abstract_diagonal abstract_above abstract_left
    abstract_substitution abstract_deletion abstract_insertion <=
  additive_cell
    concrete_diagonal concrete_above concrete_left
    concrete_substitution concrete_deletion concrete_insertion.
Proof.
  intros; unfold additive_cell; apply min3_monotone; lia.
Qed.

(** A point interval supplies equal predecessor and operation costs, so the
    relaxed recurrence is exactly the concrete recurrence. *)
Theorem point_interval_additive_step_is_exact : forall
    diagonal above left substitution deletion insertion,
  additive_cell diagonal above left substitution deletion insertion =
  additive_cell diagonal above left substitution deletion insertion.
Proof. reflexivity. Qed.

Definition bottleneck_cell
    (diagonal above left link : nat) : nat :=
  Nat.max link (min3 diagonal above left).

(** The discrete-Frechet bottleneck recurrence has the same lower-simulation
    property because both [min] and [max] are monotone. *)
Theorem interval_bottleneck_step_is_lower_simulation : forall
    abstract_diagonal abstract_above abstract_left abstract_link
    concrete_diagonal concrete_above concrete_left concrete_link,
  abstract_diagonal <= concrete_diagonal ->
  abstract_above <= concrete_above ->
  abstract_left <= concrete_left ->
  abstract_link <= concrete_link ->
  bottleneck_cell abstract_diagonal abstract_above abstract_left abstract_link <=
  bottleneck_cell concrete_diagonal concrete_above concrete_left concrete_link.
Proof.
  intros; unfold bottleneck_cell.
  apply Nat.max_le_compat; [assumption |].
  now apply min3_monotone.
Qed.

(** Exact verification is the authority boundary for an abstract candidate.
    An admissible bound can admit a false positive, but exact rescoring cannot
    emit one beyond the inclusive cutoff. *)
Definition verify_candidate (exact cutoff : nat) : bool := exact <=? cutoff.

Theorem exact_leaf_verification_has_no_false_positives : forall exact cutoff,
  verify_candidate exact cutoff = true -> exact <= cutoff.
Proof. intros; now apply Nat.leb_le. Qed.

Theorem abstract_rejection_is_safe : forall lower exact cutoff,
  lower <= exact -> cutoff <? lower = true -> cutoff < exact.
Proof.
  intros lower exact cutoff Hlower Hreject.
  apply Nat.ltb_lt in Hreject; lia.
Qed.

(** A page is a prefix and a continuation is the unconsumed suffix. *)
Definition page {A : Type} (limit : nat) (remaining : list A) : list A * list A :=
  (firstn limit remaining, skipn limit remaining).

(** Resuming after any page partition is observationally equal to the
    uninterrupted result sequence. *)
Theorem page_then_resume_equals_uninterrupted : forall A limit (remaining : list A),
  fst (page limit remaining) ++ snd (page limit remaining) = remaining.
Proof.
  intros; unfold page; simpl; apply firstn_skipn.
Qed.

Inductive tagged_outcome (A : Type) : Type :=
  | Complete : list A -> tagged_outcome A
  | Incomplete : list A -> tagged_outcome A.

Arguments Complete {A} _.
Arguments Incomplete {A} _.

Definition finish_or_pause {A : Type} (exhausted : bool) (results : list A) : tagged_outcome A :=
  if exhausted then Complete results else Incomplete results.

(** Completion is equivalent to observed exhaustion; in particular an empty
    partial page cannot be mistaken for complete absence. *)
Theorem complete_if_and_only_if_exhausted : forall A exhausted (results : list A),
  (exists complete_results,
      finish_or_pause exhausted results = Complete complete_results) <->
  exhausted = true.
Proof.
  intros A [] results; simpl.
  - split; intros; [reflexivity | now exists results].
  - split.
    + intros [complete_results H]. discriminate.
    + intros H. discriminate.
Qed.

Record generation_usage : Type := {
  current_positions : nat;
  next_positions : nat;
  cached_transitions : nat
}.

Definition retained_positions (usage : generation_usage) : nat :=
  current_positions usage + next_positions usage + cached_transitions usage.

Definition generation_bounded
    (frontier_limit cache_limit : nat) (usage : generation_usage) : Prop :=
  current_positions usage <= frontier_limit /\
  next_positions usage <= frontier_limit /\
  cached_transitions usage <= cache_limit.

(** Two-generation reclamation bounds retained state independently of the
    number of target units already consumed. *)
Theorem generational_retention_is_prefix_independent : forall
    (consumed_prefix frontier_limit cache_limit : nat) usage,
  generation_bounded frontier_limit cache_limit usage ->
  retained_positions usage <= 2 * frontier_limit + cache_limit.
Proof.
  intros consumed_prefix frontier_limit cache_limit
         [current next cached] [Hcurrent [Hnext Hcached]].
  unfold retained_positions; simpl in *; lia.
Qed.

(** Sparse generated-transition storage retains exactly one cell per observed
    source/class pair; query width does not multiply unobserved cells. *)
Definition nat_pair_eq_dec : forall left right : nat * nat,
    {left = right} + {left <> right}.
Proof. decide equality; apply Nat.eq_dec. Defined.

Definition sparse_transition_cells (observed_pairs : list (nat * nat)) : nat :=
  length (nodup nat_pair_eq_dec observed_pairs).

Theorem sparse_transition_storage_is_observation_bounded : forall observed_pairs,
  sparse_transition_cells observed_pairs <= length observed_pairs.
Proof.
  intros; unfold sparse_transition_cells.
  apply NoDup_incl_length.
  - apply NoDup_nodup.
  - intros pair Hin; now apply nodup_In in Hin.
Qed.

(** The optimized bounded product separates two stores: explicit DFS frames
    contain compact identifiers, while a query-local exact interner retains
    every distinct canonical residual constructed so far.  Popping a frame
    therefore does not pop an arena state. *)
Definition arena_references_valid {B : Type}
    (frames : list nat) (states : list B) : Prop :=
  Forall (fun state_id => state_id < length states) frames.

(** Reaching an already-interned residual pushes one identifier and allocates
    no state. *)
Theorem push_reused_state_preserves_valid_references : forall
    B (states : list B) frames state_id,
  arena_references_valid frames states ->
  state_id < length states ->
  arena_references_valid (state_id :: frames) states.
Proof.
  intros B states frames state_id Hvalid Hid.
  now constructor.
Qed.

(** A fresh residual is committed to the arena before its new identifier is
    placed in a frame. Existing identifiers remain valid because arena indices
    never move during a query. *)
Theorem push_fresh_state_preserves_valid_references : forall
    B (state : B) states frames,
  arena_references_valid frames states ->
  arena_references_valid (length states :: frames) (state :: states).
Proof.
  intros B state states frames Hvalid.
  constructor; simpl; [lia |].
  induction Hvalid as [| state_id remaining Hid Hremaining IH].
  - constructor.
  - constructor; [simpl; lia | exact IH].
Qed.

(** Iterative DFS pop removes only the top frame. Every remaining identifier
    still names the same immutable arena entry. *)
Theorem pop_frame_preserves_valid_references : forall
    B (states : list B) frames,
  arena_references_valid frames states ->
  arena_references_valid (tl frames) states.
Proof.
  intros B states frames Hvalid.
  destruct frames as [| state_id remaining].
  - constructor.
  - simpl; inversion Hvalid; assumption.
Qed.

Definition retained_arena_bytes
    (state_count column_width cell_bytes : nat) : nat :=
  state_count * column_width * cell_bytes.

(** The explicit state budget, rather than live DFS depth, bounds the exact
    interner independently of the number of dictionary nodes already visited. *)
Theorem interned_arena_retention_is_history_independent : forall
    (visited_nodes state_count max_states column_width cell_bytes : nat),
  state_count <= max_states ->
  retained_arena_bytes state_count column_width cell_bytes <=
  retained_arena_bytes max_states column_width cell_bytes.
Proof.
  intros visited_nodes state_count max_states column_width cell_bytes Hstates.
  unfold retained_arena_bytes.
  apply Nat.mul_le_mono_r.
  apply Nat.mul_le_mono_r.
  exact Hstates.
Qed.

(** The prospective scratch gate is transactional: a rejected child leaves
    the retained byte count exactly unchanged. *)
Definition retain_child_if_fits
    (limit current prospective : nat) : nat :=
  if prospective <=? limit then prospective else current.

Theorem rejected_child_preflight_is_atomic : forall limit current prospective,
  limit < prospective ->
  retain_child_if_fits limit current prospective = current.
Proof.
  intros limit current prospective Hreject.
  unfold retain_child_if_fits.
  apply Nat.leb_gt in Hreject.
  now rewrite Hreject.
Qed.

(** Sparse row construction charges its unit of work before evaluating or
    retaining the row.  [None] therefore denotes an atomic pause, not an
    over-budget transition whose accounting was discovered afterward. *)
Definition charge_sparse_row (limit used : nat) : option nat :=
  if S used <=? limit then Some (S used) else None.

Theorem admitted_sparse_row_never_exceeds_budget : forall limit used charged,
  charge_sparse_row limit used = Some charged ->
  charged = S used /\ charged <= limit.
Proof.
  intros limit used charged Hadmitted.
  unfold charge_sparse_row in Hadmitted.
  destruct (S used <=? limit) eqn:Hfits; try discriminate.
  inversion Hadmitted; subst.
  split; [reflexivity | now apply Nat.leb_le].
Qed.

Theorem rejected_sparse_row_is_pre_evaluation : forall limit used,
  limit < S used ->
  charge_sparse_row limit used = None.
Proof.
  intros limit used Hreject.
  unfold charge_sparse_row.
  apply Nat.leb_gt in Hreject.
  now rewrite Hreject.
Qed.

(** A consuming recurrence row can depend horizontally on the same previous
    row or diagonally on its immediate successor.  The sparse scheduler merges
    exactly those two seed sets before each kernel performs its vertical
    zero-target-consumption closure. *)
Definition neighbor_seed (active : list nat) (row : nat) : Prop :=
  In row active \/ exists predecessor, In predecessor active /\ row = S predecessor.

Theorem same_row_predecessor_is_scheduled : forall active row,
  In row active -> neighbor_seed active row.
Proof. intros active row Hactive; now left. Qed.

Theorem diagonal_predecessor_is_scheduled : forall active predecessor,
  In predecessor active -> neighbor_seed active (S predecessor).
Proof.
  intros active predecessor Hactive; right.
  now exists predecessor.
Qed.

Definition vertical_reachable (active : list nat) (row : nat) : Prop :=
  exists seed, neighbor_seed active seed /\ seed <= row.

Theorem scheduled_seed_is_vertically_reachable : forall active seed,
  neighbor_seed active seed -> vertical_reachable active seed.
Proof.
  intros active seed Hseed; exists seed; split; [exact Hseed | lia].
Qed.

Theorem vertical_successor_remains_reachable : forall active row,
  vertical_reachable active row -> vertical_reachable active (S row).
Proof.
  intros active row [seed [Hseed Hle]].
  exists seed; split; [exact Hseed | lia].
Qed.

(** A rolling query stores at most its preregistered window and emits owned
    snapshots of exactly that width. Consumed stream history never appears in
    the retained-state bound. *)
Definition rolling_retained (window consumed : nat) : nat :=
  Nat.min window consumed.

Theorem rolling_retention_is_stream_length_independent : forall window consumed,
  rolling_retained window consumed <= window.
Proof. intros; unfold rolling_retained; apply Nat.le_min_l. Qed.

Definition rolling_emit (window consumed : nat) : option nat :=
  if window <=? consumed then Some window else None.

Theorem emitted_rolling_window_has_registered_width : forall window consumed width,
  rolling_emit window consumed = Some width -> width = window.
Proof.
  intros window consumed width Hemit.
  unfold rolling_emit in Hemit.
  destruct (window <=? consumed); inversion Hemit; reflexivity.
Qed.

(** Snapshot acceptance is a conjunction over the content digest and every
    preregistered semantic configuration field. A dictionary payload alone is
    never sufficient. *)
Record snapshot_binding : Type := {
  dictionary_digest : nat;
  originals_digest : nat;
  kernel_digest : nat;
  quantizer_digest : nat;
  fold_digest : nat;
  transform_digest : nat
}.

Definition snapshot_accepts
    (expected observed : snapshot_binding) (checksum_ok : bool) : bool :=
  checksum_ok &&
  Nat.eqb (dictionary_digest expected) (dictionary_digest observed) &&
  Nat.eqb (originals_digest expected) (originals_digest observed) &&
  Nat.eqb (kernel_digest expected) (kernel_digest observed) &&
  Nat.eqb (quantizer_digest expected) (quantizer_digest observed) &&
  Nat.eqb (fold_digest expected) (fold_digest observed) &&
  Nat.eqb (transform_digest expected) (transform_digest observed).

Theorem checksum_mismatch_fails_closed : forall expected observed,
  snapshot_accepts expected observed false = false.
Proof. reflexivity. Qed.

Theorem fold_mismatch_fails_closed : forall expected observed checksum_ok,
  fold_digest expected <> fold_digest observed ->
  snapshot_accepts expected observed checksum_ok = false.
Proof.
  intros expected observed checksum_ok Hmismatch.
  unfold snapshot_accepts.
  apply Nat.eqb_neq in Hmismatch.
  now rewrite Hmismatch; repeat rewrite andb_false_r.
Qed.

(** A replayable additive witness is a finite list of local edge costs.  The
    executable certificate stores operation tags rather than trusting these
    costs; replay recomputes each edge from the original operands and kernel
    configuration. *)
Fixpoint replay_additive (initial : nat) (edges : list nat) : nat :=
  match edges with
  | [] => initial
  | edge :: suffix => replay_additive (initial + edge) suffix
  end.

Inductive locally_certified_trace : nat -> list nat -> nat -> Prop :=
  | CertifiedNil : forall cost,
      locally_certified_trace cost [] cost
  | CertifiedCons : forall source edge edges target,
      locally_certified_trace (source + edge) edges target ->
      locally_certified_trace source (edge :: edges) target.

(** Local predecessor equalities telescope: replaying every checked edge
    returns exactly the DP value at the terminal cell. *)
Theorem locally_certified_trace_replays_exactly : forall source edges target,
  locally_certified_trace source edges target ->
  replay_additive source edges = target.
Proof.
  intros source edges target Htrace.
  induction Htrace; simpl; [reflexivity | exact IHHtrace].
Qed.

(** Every reverse grid step consumes at least one operand coordinate.  Thus an
    iterative traceback is bounded by the starting measure and cannot recurse
    or cycle. *)
Definition grid_measure (query_index target_index : nat) : nat :=
  query_index + target_index.

Theorem monotone_traceback_step_decreases_measure : forall
    query_index target_index previous_query previous_target,
  previous_query <= query_index ->
  previous_target <= target_index ->
  previous_query < query_index \/ previous_target < target_index ->
  grid_measure previous_query previous_target <
  grid_measure query_index target_index.
Proof.
  intros query_index target_index previous_query previous_target
         Hquery Htarget [Hquery_strict | Htarget_strict];
    unfold grid_measure; lia.
Qed.

(** Reserving the worst-case monotone path length before extraction is a hard
    witness-memory gate; a rejected reservation retains zero witness bytes. *)
Definition reserve_witness (limit requested : nat) : option nat :=
  if requested <=? limit then Some requested else None.

Theorem rejected_witness_reservation_is_atomic : forall limit requested,
  limit < requested -> reserve_witness limit requested = None.
Proof.
  intros limit requested Hreject.
  unfold reserve_witness.
  apply Nat.leb_gt in Hreject.
  now rewrite Hreject.
Qed.
