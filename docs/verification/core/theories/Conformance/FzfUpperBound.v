(** * Capacity-sensitive fzf upper bound derived from the recurrence *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition max3 (completed unstarted active : Z) : Z :=
  Z.max completed (Z.max unstarted active).

(** Rust omits an infeasible alternative. In the integer model we replace it
    with [completed], which cannot increase [max3]. *)
Definition feasible_term (feasible : bool) (completed term : Z) : Z :=
  if feasible then term else completed.

Definition fzf_bound
    (completed unstarted active : Z)
    (query_len active_remaining capacity beta : Z) : Z :=
  max3 completed
    (feasible_term (query_len <=? capacity) completed unstarted)
    (feasible_term (active_remaining <=? capacity) completed
       (active + active_remaining * beta)).

(** These are precisely the ways a child alternative arises from
    FuzzyMatchV2: retain a completion; take a non-increasing gap; take a match
    whose gain is at most [beta]; or begin a local alignment that was unstarted
    in the parent. *)
Inductive recurrence_projection
    (completed unstarted active : Z)
    (query_len active_remaining capacity beta : Z) : Z -> Prop :=
| projection_completed :
    recurrence_projection completed unstarted active
      query_len active_remaining capacity beta completed
| projection_gap : forall child_score,
    0 <= beta ->
    active_remaining <= capacity ->
    child_score <= active ->
    recurrence_projection completed unstarted active
      query_len active_remaining capacity beta
      (child_score + active_remaining * beta)
| projection_match : forall child_score child_remaining,
    0 <= beta ->
    active_remaining = child_remaining + 1 ->
    active_remaining <= capacity ->
    child_score <= active + beta ->
    recurrence_projection completed unstarted active
      query_len active_remaining capacity beta
      (child_score + child_remaining * beta)
| projection_start : forall child_projection,
    query_len <= capacity ->
    child_projection <= unstarted ->
    recurrence_projection completed unstarted active
      query_len active_remaining capacity beta child_projection.

Lemma max3_completed : forall c u a, c <= max3 c u a.
Proof. intros; unfold max3; apply Z.le_max_l. Qed.

Lemma max3_unstarted : forall c u a, u <= max3 c u a.
Proof. intros; unfold max3; eapply Z.le_trans; [apply Z.le_max_l|apply Z.le_max_r]. Qed.

Lemma max3_active : forall c u a, a <= max3 c u a.
Proof. intros; unfold max3; eapply Z.le_trans; [apply Z.le_max_r|apply Z.le_max_r]. Qed.

Theorem recurrence_projection_is_bounded :
  forall completed unstarted active query_len active_remaining capacity beta score,
  recurrence_projection completed unstarted active
    query_len active_remaining capacity beta score ->
  score <= fzf_bound completed unstarted active
    query_len active_remaining capacity beta.
Proof.
  intros completed unstarted active query_len active_remaining capacity beta score Hstep.
  inversion Hstep; subst; unfold fzf_bound.
  - apply max3_completed.
  - assert (Hactive_bool : (active_remaining <=? capacity) = true)
      by (apply Z.leb_le; assumption).
    unfold feasible_term. rewrite Hactive_bool.
    destruct (query_len <=? capacity) eqn:Hquery;
      (apply Z.le_trans with (active + active_remaining * beta);
       [lia | apply max3_active]).
  - assert (Hactive_bool : (child_remaining + 1 <=? capacity) = true)
      by (apply Z.leb_le; assumption).
    unfold feasible_term. rewrite Hactive_bool.
    destruct (query_len <=? capacity) eqn:Hquery;
      (apply Z.le_trans with (active + (child_remaining + 1) * beta);
       [nia | apply max3_active]).
  - assert (Hquery_bool : (query_len <=? capacity) = true)
      by (apply Z.leb_le; assumption).
    unfold feasible_term. rewrite Hquery_bool.
    destruct (active_remaining <=? capacity);
      (apply Z.le_trans with unstarted;
       [assumption | apply max3_unstarted]).
Qed.

Theorem branch_and_bound_prune_is_sound :
  forall completed unstarted active query_len active_remaining capacity beta score cutoff,
  recurrence_projection completed unstarted active
    query_len active_remaining capacity beta score ->
  fzf_bound completed unstarted active
    query_len active_remaining capacity beta < cutoff ->
  score < cutoff.
Proof.
  intros. pose proof (recurrence_projection_is_bounded _ _ _ _ _ _ _ _ H). lia.
Qed.

Example unstarted_capacity_is_load_bearing :
  fzf_bound 0 40 (-20) 3 2 3 20 = 40 /\
  fzf_bound 0 40 (-20) 3 2 2 20 = 20.
Proof. split; reflexivity. Qed.

Theorem fzf_arc_delta_telescope : forall initial middle final,
  initial + (middle - initial) + (final - middle) = final.
Proof. intros; lia. Qed.
