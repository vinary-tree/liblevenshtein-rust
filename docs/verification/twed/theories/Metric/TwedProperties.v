(** Machine-checked arithmetic obligations for unit-spaced TWED.

    The production kernel relaxes two adjacent target samples to two closed
    intervals. This development proves exactness and admissibility of those
    local box minima, monotonicity of the additive DP step, the length lower
    bound over arbitrary edit scripts, and additive script concatenation. The
    full Marteau metric theorem is not silently postulated here: the executable
    triangle property is independently exercised by generated Rust tests.
    This file contains no axioms or admitted obligations. *)

From Stdlib Require Import Reals Lra List.

Import ListNotations.
Open Scope R_scope.

Definition rmin (x y : R) : R := if Rle_dec x y then x else y.
Definition min3 (x y z : R) : R := rmin x (rmin y z).

Lemma rmin_monotone : forall a b c d,
  a <= c -> b <= d -> rmin a b <= rmin c d.
Proof.
  intros a b c d Hac Hbd. unfold rmin.
  destruct (Rle_dec a b); destruct (Rle_dec c d); lra.
Qed.

Lemma min3_monotone : forall a b c a' b' c',
  a <= a' -> b <= b' -> c <= c' -> min3 a b c <= min3 a' b' c'.
Proof.
  intros a b c a' b' c' Ha Hb Hc. unfold min3.
  apply rmin_monotone; [exact Ha |].
  now apply rmin_monotone.
Qed.

Definition interval_dist (x lo hi : R) : R :=
  if Rlt_dec x lo then lo - x
  else if Rlt_dec hi x then x - hi
  else 0.

Definition interval_gap (lo1 hi1 lo2 hi2 : R) : R :=
  if Rlt_dec hi1 lo2 then lo2 - hi1
  else if Rlt_dec hi2 lo1 then lo1 - hi2
  else 0.

Lemma interval_dist_nonnegative : forall x lo hi,
  0 <= interval_dist x lo hi.
Proof.
  intros x lo hi. unfold interval_dist.
  destruct (Rlt_dec x lo); destruct (Rlt_dec hi x); lra.
Qed.

Lemma interval_dist_admissible : forall x lo hi y,
  lo <= hi -> lo <= y <= hi ->
  interval_dist x lo hi <= Rabs (x - y).
Proof.
  intros x lo hi y Hbox [Hlo Hhi]. unfold interval_dist.
  destruct (Rlt_dec x lo) as [Hxlo | Hxlo].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec hi x) as [Hhix | Hhix].
    + rewrite Rabs_right; lra.
    + pose proof (Rabs_pos (x - y)). lra.
Qed.

Lemma interval_dist_degenerate : forall x y,
  interval_dist x y y = Rabs (x - y).
Proof.
  intros x y. unfold interval_dist.
  destruct (Rlt_dec x y) as [Hxy | Hxy].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec y x) as [Hyx | Hyx].
    + rewrite Rabs_right; lra.
    + assert (x = y) by lra. subst. replace (y - y) with 0 by ring.
      rewrite Rabs_R0. reflexivity.
Qed.

Lemma interval_gap_nonnegative : forall lo1 hi1 lo2 hi2,
  0 <= interval_gap lo1 hi1 lo2 hi2.
Proof.
  intros lo1 hi1 lo2 hi2. unfold interval_gap.
  destruct (Rlt_dec hi1 lo2); destruct (Rlt_dec hi2 lo1); lra.
Qed.

Lemma interval_gap_admissible : forall lo1 hi1 lo2 hi2 x y,
  lo1 <= hi1 -> lo2 <= hi2 ->
  lo1 <= x <= hi1 -> lo2 <= y <= hi2 ->
  interval_gap lo1 hi1 lo2 hi2 <= Rabs (x - y).
Proof.
  intros lo1 hi1 lo2 hi2 x y Hbox1 Hbox2 [Hxlo Hxhi] [Hylo Hyhi].
  unfold interval_gap.
  destruct (Rlt_dec hi1 lo2) as [Hbefore | Hbefore].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec hi2 lo1) as [Hafter | Hafter].
    + rewrite Rabs_right; lra.
    + pose proof (Rabs_pos (x - y)). lra.
Qed.

Lemma interval_gap_degenerate : forall x y,
  interval_gap x x y y = Rabs (x - y).
Proof.
  intros x y. unfold interval_gap.
  destruct (Rlt_dec x y) as [Hxy | Hxy].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec y x) as [Hyx | Hyx].
    + rewrite Rabs_right; lra.
    + assert (x = y) by lra. subst. replace (y - y) with 0 by ring.
      rewrite Rabs_R0. reflexivity.
Qed.

Definition match_relaxed
  (xcur xprev clo chi plo phi displacement nu : R) : R :=
  interval_dist xcur clo chi + interval_dist xprev plo phi
  + 2 * nu * displacement.

Definition match_exact
  (xcur xprev ycur yprev displacement nu : R) : R :=
  Rabs (xcur - ycur) + Rabs (xprev - yprev)
  + 2 * nu * displacement.

Theorem match_interval_admissible : forall
  xcur xprev clo chi plo phi ycur yprev displacement nu,
  clo <= chi -> plo <= phi ->
  clo <= ycur <= chi -> plo <= yprev <= phi ->
  match_relaxed xcur xprev clo chi plo phi displacement nu
  <= match_exact xcur xprev ycur yprev displacement nu.
Proof.
  intros xcur xprev clo chi plo phi ycur yprev displacement nu
    Hcurrent Hprevious Hycurrent Hyprevious.
  unfold match_relaxed, match_exact.
  pose proof (interval_dist_admissible xcur clo chi ycur Hcurrent Hycurrent).
  pose proof (interval_dist_admissible xprev plo phi yprev Hprevious Hyprevious).
  lra.
Qed.

Theorem match_point_intervals_exact : forall
  xcur xprev ycur yprev displacement nu,
  match_relaxed xcur xprev ycur ycur yprev yprev displacement nu
  = match_exact xcur xprev ycur yprev displacement nu.
Proof.
  intros. unfold match_relaxed, match_exact.
  now rewrite !interval_dist_degenerate.
Qed.

Definition delete_relaxed (clo chi plo phi nu lambda : R) : R :=
  interval_gap clo chi plo phi + nu + lambda.

Definition delete_exact (current previous nu lambda : R) : R :=
  Rabs (current - previous) + nu + lambda.

Theorem delete_interval_admissible : forall
  clo chi plo phi current previous nu lambda,
  clo <= chi -> plo <= phi ->
  clo <= current <= chi -> plo <= previous <= phi ->
  delete_relaxed clo chi plo phi nu lambda
  <= delete_exact current previous nu lambda.
Proof.
  intros clo chi plo phi current previous nu lambda
    Hcurrent Hprevious Hcur Hprev.
  unfold delete_relaxed, delete_exact.
  pose proof (interval_gap_admissible clo chi plo phi current previous
    Hcurrent Hprevious Hcur Hprev).
  lra.
Qed.

Theorem delete_point_intervals_exact : forall current previous nu lambda,
  delete_relaxed current current previous previous nu lambda
  = delete_exact current previous nu lambda.
Proof.
  intros. unfold delete_relaxed, delete_exact.
  now rewrite interval_gap_degenerate.
Qed.

Definition twed_step
  (north diagonal west delete_x match_leaf delete_y : R) : R :=
  min3 (north + delete_x) (diagonal + match_leaf) (west + delete_y).

Theorem twed_step_monotone : forall
  north diagonal west delete_x match_leaf delete_y
  north' diagonal' west' delete_x' match_leaf' delete_y',
  north <= north' -> diagonal <= diagonal' -> west <= west' ->
  delete_x <= delete_x' -> match_leaf <= match_leaf' -> delete_y <= delete_y' ->
  twed_step north diagonal west delete_x match_leaf delete_y
  <= twed_step north' diagonal' west' delete_x' match_leaf' delete_y'.
Proof.
  intros. unfold twed_step. apply min3_monotone; lra.
Qed.

Inductive twed_edit : Type :=
| MatchEdit (xcur xprev ycur yprev displacement : R)
| DeleteLeft (current previous : R)
| DeleteRight (current previous : R).

Definition edit_left_length (edit : twed_edit) : R :=
  match edit with MatchEdit _ _ _ _ _ | DeleteLeft _ _ => 1 | DeleteRight _ _ => 0 end.

Definition edit_right_length (edit : twed_edit) : R :=
  match edit with MatchEdit _ _ _ _ _ | DeleteRight _ _ => 1 | DeleteLeft _ _ => 0 end.

Definition edit_deletions (edit : twed_edit) : R :=
  match edit with MatchEdit _ _ _ _ _ => 0 | _ => 1 end.

Definition edit_well_formed (edit : twed_edit) : Prop :=
  match edit with MatchEdit _ _ _ _ displacement => 0 <= displacement | _ => True end.

Definition edit_cost (nu lambda : R) (edit : twed_edit) : R :=
  match edit with
  | MatchEdit xcur xprev ycur yprev displacement =>
      match_exact xcur xprev ycur yprev displacement nu
  | DeleteLeft current previous | DeleteRight current previous =>
      delete_exact current previous nu lambda
  end.

Fixpoint script_left_length (script : list twed_edit) : R :=
  match script with [] => 0 | edit :: tail => edit_left_length edit + script_left_length tail end.

Fixpoint script_right_length (script : list twed_edit) : R :=
  match script with [] => 0 | edit :: tail => edit_right_length edit + script_right_length tail end.

Fixpoint script_deletions (script : list twed_edit) : R :=
  match script with [] => 0 | edit :: tail => edit_deletions edit + script_deletions tail end.

Fixpoint script_cost (nu lambda : R) (script : list twed_edit) : R :=
  match script with [] => 0 | edit :: tail => edit_cost nu lambda edit + script_cost nu lambda tail end.

Lemma edit_length_gap_le_deletions : forall edit,
  Rabs (edit_left_length edit - edit_right_length edit) <= edit_deletions edit.
Proof.
  intros [xcur xprev ycur yprev displacement | current previous | current previous];
    simpl.
  - replace (1 - 1) with 0 by ring. rewrite Rabs_R0. lra.
  - rewrite Rabs_right; lra.
  - rewrite Rabs_left; lra.
Qed.

Theorem script_length_gap_le_deletions : forall script,
  Rabs (script_left_length script - script_right_length script)
  <= script_deletions script.
Proof.
  intros script. induction script as [|edit tail IH]; simpl.
  - replace (0 - 0) with 0 by ring. rewrite Rabs_R0. lra.
  - replace
      (edit_left_length edit + script_left_length tail -
       (edit_right_length edit + script_right_length tail))
      with
      ((edit_left_length edit - edit_right_length edit) +
       (script_left_length tail - script_right_length tail)) by ring.
    eapply Rle_trans.
    + apply Rabs_triang.
    + pose proof (edit_length_gap_le_deletions edit). lra.
Qed.

Lemma edit_deletion_penalty_bound : forall nu lambda edit,
  0 <= nu -> 0 <= lambda -> edit_well_formed edit ->
  lambda * edit_deletions edit <= edit_cost nu lambda edit.
Proof.
  intros nu lambda [xcur xprev ycur yprev displacement | current previous |
    current previous] Hnu Hlambda Hwell; simpl in *; unfold match_exact, delete_exact.
  - pose proof (Rabs_pos (xcur - ycur)).
    pose proof (Rabs_pos (xprev - yprev)). nra.
  - pose proof (Rabs_pos (current - previous)). nra.
  - pose proof (Rabs_pos (current - previous)). nra.
Qed.

Theorem script_deletion_penalty_bound : forall nu lambda script,
  0 <= nu -> 0 <= lambda -> Forall edit_well_formed script ->
  lambda * script_deletions script <= script_cost nu lambda script.
Proof.
  intros nu lambda script Hnu Hlambda Hwell.
  induction Hwell as [|edit tail Hedit Htail IH]; simpl.
  - nra.
  - pose proof (edit_deletion_penalty_bound nu lambda edit Hnu Hlambda Hedit).
    nra.
Qed.

Theorem twed_length_lower_bound : forall nu lambda script,
  0 <= nu -> 0 <= lambda -> Forall edit_well_formed script ->
  lambda * Rabs (script_left_length script - script_right_length script)
  <= script_cost nu lambda script.
Proof.
  intros nu lambda script Hnu Hlambda Hwell.
  pose proof (script_length_gap_le_deletions script) as Hlength.
  pose proof (script_deletion_penalty_bound nu lambda script Hnu Hlambda Hwell)
    as Hpenalty.
  nra.
Qed.

Theorem script_cost_app : forall nu lambda left right,
  script_cost nu lambda (left ++ right)
  = script_cost nu lambda left + script_cost nu lambda right.
Proof.
  intros nu lambda left right. induction left as [|edit tail IH]; simpl; lra.
Qed.

Theorem concatenated_script_cost_is_additive : forall nu lambda xy yz cxy cyz,
  script_cost nu lambda xy = cxy -> script_cost nu lambda yz = cyz ->
  script_cost nu lambda (xy ++ yz) = cxy + cyz.
Proof.
  intros nu lambda xy yz cxy cyz Hxy Hyz.
  rewrite script_cost_app, Hxy, Hyz. reflexivity.
Qed.

Lemma positive_stiffness_is_nonzero : forall nu, 0 < nu -> nu <> 0.
Proof. intros nu Hpositive Heq. subst. lra. Qed.

Example zero_parameter_unequal_length_witness :
  script_left_length
    [DeleteLeft 0 0; MatchEdit 1 0 1 0 0] = 2 /\
  script_right_length
    [DeleteLeft 0 0; MatchEdit 1 0 1 0 0] = 1 /\
  script_cost 0 0
    [DeleteLeft 0 0; MatchEdit 1 0 1 0 0] = 0.
Proof.
  split; [simpl; lra |]. split; [simpl; lra |].
  simpl. unfold match_exact, delete_exact.
  replace (0 - 0) with 0 by ring.
  replace (1 - 1) with 0 by ring.
  repeat rewrite Rabs_R0. lra.
Qed.

(** Explicit-timestamp TWED replaces the unit-grid stiffness constant by the
    elapsed physical time in one canonical unit. *)
Definition physical_delete_exact
    (current previous time previous_time nu lambda : R) : R :=
  Rabs (current - previous) + nu * (time - previous_time) + lambda.

Theorem physical_delete_is_nonnegative : forall
    current previous time previous_time nu lambda,
  previous_time <= time -> 0 <= nu -> 0 <= lambda ->
  0 <= physical_delete_exact
    current previous time previous_time nu lambda.
Proof.
  intros current previous time previous_time nu lambda Htime Hnu Hlambda.
  unfold physical_delete_exact.
  pose proof (Rabs_pos (current - previous)). nra.
Qed.

(** Unit-spaced physical timestamps reproduce the unit-grid deletion leaf
    exactly; this is the local correspondence used by the executable
    generated test. *)
Theorem unit_elapsed_physical_delete_is_unit_grid : forall
    current previous previous_time nu lambda,
  physical_delete_exact
    current previous (previous_time + 1) previous_time nu lambda =
  delete_exact current previous nu lambda.
Proof.
  intros. unfold physical_delete_exact, delete_exact. ring.
Qed.

Definition physical_match_exact
    (xcur xprev xtime xprev_time
     ycur yprev ytime yprev_time nu : R) : R :=
  Rabs (xcur - ycur) + Rabs (xprev - yprev) +
  nu * (Rabs (xtime - ytime) + Rabs (xprev_time - yprev_time)).

Theorem physical_match_is_nonnegative : forall
    xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu,
  0 <= nu ->
  0 <= physical_match_exact
    xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu.
Proof.
  intros xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu Hnu.
  unfold physical_match_exact.
  pose proof (Rabs_pos (xcur - ycur)).
  pose proof (Rabs_pos (xprev - yprev)).
  pose proof (Rabs_pos (xtime - ytime)).
  pose proof (Rabs_pos (xprev_time - yprev_time)). nra.
Qed.

(** The online constructor's monotonicity gate makes every committed elapsed
    target-time term nonnegative. *)
Theorem validated_timestamp_step_has_nonnegative_elapsed_time : forall
    previous_time current_time,
  previous_time < current_time -> 0 <= current_time - previous_time.
Proof. intros; lra. Qed.

(** The explicit-time interval deletion leaf composes independent value and
    physical-time box distances.  This theorem is over finite mathematical
    reals.  The executable boundary separately admits infinite value-box
    endpoints for clamped quantizer bins, while requiring finite timestamps. *)
Definition physical_delete_relaxed
    (current_lo current_hi previous_lo previous_hi
     time_lo time_hi previous_time_lo previous_time_hi nu lambda : R) : R :=
  interval_gap current_lo current_hi previous_lo previous_hi
  + nu * interval_gap time_lo time_hi previous_time_lo previous_time_hi
  + lambda.

Theorem physical_delete_interval_admissible : forall
    current_lo current_hi previous_lo previous_hi
    time_lo time_hi previous_time_lo previous_time_hi
    current previous time previous_time nu lambda,
  current_lo <= current_hi -> previous_lo <= previous_hi ->
  time_lo <= time_hi -> previous_time_lo <= previous_time_hi ->
  current_lo <= current <= current_hi ->
  previous_lo <= previous <= previous_hi ->
  time_lo <= time <= time_hi ->
  previous_time_lo <= previous_time <= previous_time_hi ->
  previous_time <= time -> 0 <= nu ->
  physical_delete_relaxed
    current_lo current_hi previous_lo previous_hi
    time_lo time_hi previous_time_lo previous_time_hi nu lambda
  <= physical_delete_exact current previous time previous_time nu lambda.
Proof.
  intros current_lo current_hi previous_lo previous_hi
    time_lo time_hi previous_time_lo previous_time_hi
    current previous time previous_time nu lambda
    Hcurrent_box Hprevious_box Htime_box Hprevious_time_box
    Hcurrent Hprevious Htime Hprevious_time Hmonotone Hnu.
  unfold physical_delete_relaxed, physical_delete_exact.
  pose proof (interval_gap_admissible
    current_lo current_hi previous_lo previous_hi current previous
    Hcurrent_box Hprevious_box Hcurrent Hprevious) as Hvalue.
  pose proof (interval_gap_admissible
    time_lo time_hi previous_time_lo previous_time_hi time previous_time
    Htime_box Hprevious_time_box Htime Hprevious_time) as Hphysical_time.
  rewrite Rabs_right in Hphysical_time by lra.
  nra.
Qed.

Theorem physical_delete_point_intervals_exact : forall
    current previous time previous_time nu lambda,
  previous_time <= time ->
  physical_delete_relaxed
    current current previous previous
    time time previous_time previous_time nu lambda
  = physical_delete_exact current previous time previous_time nu lambda.
Proof.
  intros current previous time previous_time nu lambda Htime.
  unfold physical_delete_relaxed, physical_delete_exact.
  rewrite !interval_gap_degenerate.
  replace (Rabs (time - previous_time)) with (time - previous_time)
    by (rewrite Rabs_right; lra).
  reflexivity.
Qed.

Definition physical_match_relaxed
    (xcur xprev xtime xprev_time
     ycur_lo ycur_hi yprev_lo yprev_hi
     ytime_lo ytime_hi yprev_time_lo yprev_time_hi nu : R) : R :=
  interval_dist xcur ycur_lo ycur_hi
  + interval_dist xprev yprev_lo yprev_hi
  + nu * (interval_dist xtime ytime_lo ytime_hi
          + interval_dist xprev_time yprev_time_lo yprev_time_hi).

Theorem physical_match_interval_admissible : forall
    xcur xprev xtime xprev_time
    ycur_lo ycur_hi yprev_lo yprev_hi
    ytime_lo ytime_hi yprev_time_lo yprev_time_hi
    ycur yprev ytime yprev_time nu,
  ycur_lo <= ycur_hi -> yprev_lo <= yprev_hi ->
  ytime_lo <= ytime_hi -> yprev_time_lo <= yprev_time_hi ->
  ycur_lo <= ycur <= ycur_hi -> yprev_lo <= yprev <= yprev_hi ->
  ytime_lo <= ytime <= ytime_hi ->
  yprev_time_lo <= yprev_time <= yprev_time_hi ->
  0 <= nu ->
  physical_match_relaxed
    xcur xprev xtime xprev_time
    ycur_lo ycur_hi yprev_lo yprev_hi
    ytime_lo ytime_hi yprev_time_lo yprev_time_hi nu
  <= physical_match_exact
    xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu.
Proof.
  intros xcur xprev xtime xprev_time
    ycur_lo ycur_hi yprev_lo yprev_hi
    ytime_lo ytime_hi yprev_time_lo yprev_time_hi
    ycur yprev ytime yprev_time nu
    Hycur_box Hyprev_box Hytime_box Hyprev_time_box
    Hycur Hyprev Hytime Hyprev_time Hnu.
  unfold physical_match_relaxed, physical_match_exact.
  pose proof (interval_dist_admissible
    xcur ycur_lo ycur_hi ycur Hycur_box Hycur) as Hcurrent.
  pose proof (interval_dist_admissible
    xprev yprev_lo yprev_hi yprev Hyprev_box Hyprev) as Hprevious.
  pose proof (interval_dist_admissible
    xtime ytime_lo ytime_hi ytime Hytime_box Hytime) as Hcurrent_time.
  pose proof (interval_dist_admissible
    xprev_time yprev_time_lo yprev_time_hi yprev_time
    Hyprev_time_box Hyprev_time) as Hprevious_time.
  nra.
Qed.

Theorem physical_match_point_intervals_exact : forall
    xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu,
  physical_match_relaxed
    xcur xprev xtime xprev_time
    ycur ycur yprev yprev ytime ytime yprev_time yprev_time nu
  = physical_match_exact
    xcur xprev xtime xprev_time ycur yprev ytime yprev_time nu.
Proof.
  intros. unfold physical_match_relaxed, physical_match_exact.
  now rewrite !interval_dist_degenerate.
Qed.
