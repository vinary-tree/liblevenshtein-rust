(** * Generic elastic-kernel pruning conformance

    This assumption-free development isolates the four obligations implemented
    by [ElasticKernel].  Costs are modeled by natural numbers here: the theorem
    premises are the kernel proofs (K1--K4), and the conclusions are exactly the
    pruning decisions made by the Rust range and best-first walkers.

    The real-valued additive and bottleneck monoids are proved separately in
    [CostMonoid.v].  MSM's interval-column instance is proved in
    [Liblevenshtein.MSM.Indexing.IntervalColumn].
*)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

(** K2 for the additive natural-number model: appending a lawful step cannot
    reduce accumulated cost. *)
Theorem k2_additive_inflation : forall accumulated step,
  accumulated <= accumulated + step.
Proof. intros; lia. Qed.

(** Repeated non-negative extensions preserve the same inflation invariant. *)
Theorem k2_additive_path_inflation : forall accumulated steps,
  accumulated <= fold_left Nat.add steps accumulated.
Proof.
  intros accumulated steps.
  revert accumulated.
  induction steps as [| step rest IH]; intros accumulated; simpl.
  - lia.
  - eapply Nat.le_trans.
    + apply k2_additive_inflation.
    + apply IH.
Qed.

(** The bottleneck model has the same load-bearing inflation property. *)
Theorem k2_bottleneck_inflation : forall accumulated step,
  accumulated <= Nat.max accumulated step.
Proof. intros; apply Nat.le_max_l. Qed.

(** K1 plus a failed inclusive cutoff implies that every represented exact
    descendant is outside the cutoff. *)
Theorem k1_subtree_prune_sound : forall node_bound descendant_exact cutoff,
  node_bound <= descendant_exact ->
  cutoff < node_bound ->
  cutoff < descendant_exact.
Proof. intros; lia. Qed.

(** K4 justifies skipping exact leaf scoring when the candidate-level bound is
    already outside the cutoff. *)
Theorem k4_candidate_prune_sound : forall candidate_bound exact cutoff,
  candidate_bound <= exact ->
  cutoff < candidate_bound ->
  cutoff < exact.
Proof. intros; lia. Qed.

(** K3 makes emission sound: an exact scorer may emit only its exact value and
    only inside the inclusive cutoff. *)
Theorem k3_exact_rescore_no_false_positive : forall reported exact cutoff,
  reported = exact ->
  reported <= cutoff ->
  exact <= cutoff.
Proof. intros; subst; assumption. Qed.

(** K1 and K4 compose: either pruning stage is sufficient to reject an exact
    candidate beyond the cutoff. *)
Theorem two_stage_pruning_sound : forall node_bound candidate_bound exact cutoff,
  node_bound <= exact ->
  candidate_bound <= exact ->
  cutoff < node_bound \/ cutoff < candidate_bound ->
  cutoff < exact.
Proof. intros; lia. Qed.

(** Best-first termination is sound once the minimum queued bound exceeds the
    current kth exact distance. *)
Theorem best_first_cutoff_sound : forall popped_bound queued_bound exact kth,
  popped_bound <= queued_bound ->
  queued_bound <= exact ->
  kth < popped_bound ->
  kth < exact.
Proof. intros; lia. Qed.

(** Empty/nonempty costs must be functions of the concrete sequence.  This
    executable witness distinguishes two ERP-style running gap sums and pins
    why the Rust trait receives the nonempty slice rather than exposing a
    nullary constant. *)
Fixpoint empty_gap_cost (gap : nat) (series : list nat) : nat :=
  match series with
  | [] => 0
  | value :: rest => Nat.add (Nat.sub value gap) (Nat.sub gap value)
                     + empty_gap_cost gap rest
  end.

Example empty_gap_cost_is_sequence_dependent :
  empty_gap_cost 1 [1; 1] = 0 /\ empty_gap_cost 1 [2; 2] = 2.
Proof. split; reflexivity. Qed.

(** Observational search counters partition every edge decision without
    influencing it.  This datatype mirrors the two Rust branches after an edge
    is inspected. *)
Inductive edge_decision : Type :=
  | PrefixPruned
  | ColumnBuilt.

Fixpoint prefix_pruned_count (decisions : list edge_decision) : nat :=
  match decisions with
  | [] => 0
  | PrefixPruned :: rest => S (prefix_pruned_count rest)
  | ColumnBuilt :: rest => prefix_pruned_count rest
  end.

Fixpoint column_built_count (decisions : list edge_decision) : nat :=
  match decisions with
  | [] => 0
  | PrefixPruned :: rest => column_built_count rest
  | ColumnBuilt :: rest => S (column_built_count rest)
  end.

Theorem edge_accounting_partition : forall decisions,
  length decisions = prefix_pruned_count decisions + column_built_count decisions.
Proof.
  induction decisions as [| decision rest IH]; simpl.
  - reflexivity.
  - destruct decision; simpl in *; lia.
Qed.

(** Final candidates have the analogous exclusive partition: a K4 rejection
    or an exact recurrence call. *)
Inductive candidate_decision : Type :=
  | CandidateBoundPruned
  | ExactEvaluated.

Fixpoint candidate_bound_pruned_count (decisions : list candidate_decision) : nat :=
  match decisions with
  | [] => 0
  | CandidateBoundPruned :: rest => S (candidate_bound_pruned_count rest)
  | ExactEvaluated :: rest => candidate_bound_pruned_count rest
  end.

Fixpoint exact_evaluation_count (decisions : list candidate_decision) : nat :=
  match decisions with
  | [] => 0
  | CandidateBoundPruned :: rest => exact_evaluation_count rest
  | ExactEvaluated :: rest => S (exact_evaluation_count rest)
  end.

Theorem candidate_accounting_partition : forall decisions,
  length decisions =
    candidate_bound_pruned_count decisions + exact_evaluation_count decisions.
Proof.
  induction decisions as [| decision rest IH]; simpl.
  - reflexivity.
  - destruct decision; simpl in *; lia.
Qed.

(** Cutoff abandonments are a filtered subset of exact evaluations. *)
Fixpoint true_count (outcomes : list bool) : nat :=
  match outcomes with
  | [] => 0
  | true :: rest => S (true_count rest)
  | false :: rest => true_count rest
  end.

Theorem filtered_counter_is_subset : forall outcomes,
  true_count outcomes <= length outcomes.
Proof.
  induction outcomes as [| outcome rest IH]; simpl.
  - lia.
  - destruct outcome; simpl in *; lia.
Qed.
