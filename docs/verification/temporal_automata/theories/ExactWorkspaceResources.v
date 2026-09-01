(** Assumption-free logical resource and outcome-classification proof island for
    the reusable exact point workspace.

    Natural numbers model checked Rust byte arithmetic after overflow checks.
    The file proves the plan-first peak equation, exact preflight boundary,
    retained-plus-later composition, reuse invariance, and the complete tagged
    decision table. It does not model allocator bookkeeping, kernel arithmetic,
    or whole-program correspondence. *)

From Stdlib Require Import Arith Bool Lia.

Record plan_storage : Type := {
  plan_retained : nat;
  plan_construction_peak : nat;
  plan_peak_covers_retained : plan_retained <= plan_construction_peak
}.

(** [frontier] is the logical byte size of two cost generations and two
    active-row generations. *)
Definition workspace_retained (plan : plan_storage) (frontier : nat) : nat :=
  plan_retained plan + frontier.

(** Construction is deliberately ordered: build the plan, release its
    transient scratch, then allocate the retained frontier. *)
Definition workspace_construction_peak
    (plan : plan_storage) (frontier : nat) : nat :=
  Nat.max (plan_construction_peak plan) (workspace_retained plan frontier).

Theorem plan_first_peak_is_max :
  forall plan frontier,
    workspace_construction_peak plan frontier =
    Nat.max (plan_construction_peak plan)
            (plan_retained plan + frontier).
Proof. reflexivity. Qed.

Theorem workspace_retained_is_within_construction_peak :
  forall plan frontier,
    workspace_retained plan frontier <=
    workspace_construction_peak plan frontier.
Proof.
  intros plan frontier.
  unfold workspace_construction_peak.
  apply Nat.le_max_r.
Qed.

Definition construction_preflight
    (plan : plan_storage) (frontier limit : nat) : bool :=
  Nat.leb (workspace_construction_peak plan frontier) limit.

Theorem accepted_construction_preflight_is_within_limit :
  forall plan frontier limit,
    construction_preflight plan frontier limit = true ->
    workspace_construction_peak plan frontier <= limit.
Proof.
  intros plan frontier limit accepted.
  now apply Nat.leb_le.
Qed.

Theorem rejected_construction_preflight_exceeds_limit :
  forall plan frontier limit,
    construction_preflight plan frontier limit = false ->
    limit < workspace_construction_peak plan frontier.
Proof.
  intros plan frontier limit rejected.
  now apply Nat.leb_gt.
Qed.

(** Later search state is allocated only after workspace construction. The
    ledger must therefore retain the larger of the construction peak and the
    live workspace plus later arena/queue state. *)
Definition session_peak
    (plan : plan_storage) (frontier later : nat) : nat :=
  Nat.max (workspace_construction_peak plan frontier)
          (workspace_retained plan frontier + later).

Theorem post_construction_peak_reduces_to_plan_or_live_state :
  forall plan frontier later,
    session_peak plan frontier later =
    Nat.max (plan_construction_peak plan)
            (workspace_retained plan frontier + later).
Proof.
  intros plan frontier later.
  unfold session_peak, workspace_construction_peak.
  rewrite <- Nat.max_assoc.
  replace (Nat.max (workspace_retained plan frontier)
                   (workspace_retained plan frontier + later))
    with (workspace_retained plan frontier + later).
  2: symmetry; apply Nat.max_r; lia.
  reflexivity.
Qed.

Definition post_construction_preflight
    (plan : plan_storage) (frontier later limit : nat) : bool :=
  Nat.leb (workspace_retained plan frontier + later) limit.

Theorem accepted_post_construction_state_is_within_limit :
  forall plan frontier later limit,
    post_construction_preflight plan frontier later limit = true ->
    workspace_retained plan frontier + later <= limit.
Proof.
  intros plan frontier later limit accepted.
  now apply Nat.leb_le.
Qed.

(** Resetting and reusing a workspace changes live cell values, not its
    retained storage. [candidate] makes the quantified reuse explicit. *)
Definition retained_after_reuse
    (plan : plan_storage) (frontier candidate : nat) : nat :=
  workspace_retained plan frontier.

Theorem candidate_reuse_preserves_retained_storage :
  forall plan frontier candidate,
    retained_after_reuse plan frontier candidate =
    workspace_retained plan frontier.
Proof. reflexivity. Qed.

Inductive cutoff_kind : Type :=
| FiniteCutoff
| TopCutoff.

Inductive exact_observation : Type :=
| FiniteCost (within_finite_cutoff : bool)
| TopCost
| InvalidCost.

Inductive exact_classification : Type :=
| WithinCutoff
| AboveCutoff
| NoFiniteAlignment
| NumericFailure.

(** Structural impossibility has priority over the numeric sentinel. For a
    structurally possible alignment, TOP under a finite cutoff is safely above
    cutoff; TOP under a TOP cutoff is ambiguous with overflow and fails closed. *)
Definition classify_exact
    (structurally_possible : bool)
    (cutoff : cutoff_kind)
    (observed : exact_observation) : exact_classification :=
  if negb structurally_possible then NoFiniteAlignment
  else
    match cutoff, observed with
    | FiniteCutoff, FiniteCost true => WithinCutoff
    | FiniteCutoff, FiniteCost false => AboveCutoff
    | FiniteCutoff, TopCost => AboveCutoff
    | FiniteCutoff, InvalidCost => NumericFailure
    | TopCutoff, FiniteCost _ => WithinCutoff
    | TopCutoff, TopCost => NumericFailure
    | TopCutoff, InvalidCost => NumericFailure
    end.

Theorem structural_impossibility_is_no_finite_alignment :
  forall cutoff observed,
    classify_exact false cutoff observed = NoFiniteAlignment.
Proof. reflexivity. Qed.

Theorem finite_cutoff_within_is_within :
  classify_exact true FiniteCutoff (FiniteCost true) = WithinCutoff.
Proof. reflexivity. Qed.

Theorem finite_cutoff_above_is_above :
  classify_exact true FiniteCutoff (FiniteCost false) = AboveCutoff.
Proof. reflexivity. Qed.

Theorem finite_cutoff_top_is_above :
  classify_exact true FiniteCutoff TopCost = AboveCutoff.
Proof. reflexivity. Qed.

Theorem top_cutoff_finite_is_within :
  forall within,
    classify_exact true TopCutoff (FiniteCost within) = WithinCutoff.
Proof. reflexivity. Qed.

Theorem top_cutoff_top_fails_closed :
  classify_exact true TopCutoff TopCost = NumericFailure.
Proof. reflexivity. Qed.

Theorem invalid_numeric_state_fails_closed :
  forall cutoff,
    classify_exact true cutoff InvalidCost = NumericFailure.
Proof. destruct cutoff; reflexivity. Qed.
