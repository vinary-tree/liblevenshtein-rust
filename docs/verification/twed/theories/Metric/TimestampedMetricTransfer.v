(** * Timestamped TWED anchor and cumulative-boundary transfer

    The first section is polymorphic in the cost carrier and proves that the
    published infinite-positive-axis recurrence, after one forced common
    anchor match, has exactly the cumulative boundary and interior cells.
    It proves uniqueness of that source grid, not merely that one proposed
    embedding happens to satisfy the equations.

    The second section proves that shifting and prepending the common anchor
    is injective and preserves strict timestamp order. The metric-pullback
    theorem is conditional on source metric laws on the common-anchor image
    and on explicit equality between the chosen recurrence score and that
    pullback. Neither source metricity nor this concrete score binding is
    inferred from the published proposition. No metricity axiom is introduced
    for the library's broader origin-equal domain or binary64 execution. *)

From Stdlib Require Import Arith Lia List Reals Lra.
Import ListNotations.

Section AnchorRecurrence.
  Context {Cost : Type}.
  Variable zero : Cost.
  Variable add : Cost -> Cost -> Cost.
  Variable minimum : Cost -> Cost -> Cost.
  Context (add_zero_zero : add zero zero = zero).
  Variable delete_left delete_right : nat -> Cost.
  Variable match_cost : nat -> nat -> Cost.

  Definition min3 (a b c : Cost) : Cost :=
    minimum a (minimum b c).

  Definition option_add (value : option Cost) (charge : Cost)
      : option Cost :=
    option_map (fun cost => add cost charge) value.

  Definition option_min (left right : option Cost) : option Cost :=
    match left, right with
    | None, other | other, None => other
    | Some a, Some b => Some (minimum a b)
    end.

  Definition option_min3 (a b c : option Cost) : option Cost :=
    option_min a (option_min b c).

  Variable cumulative : nat -> nat -> Cost.
  Context (cumulative_origin : cumulative 0 0 = zero).
  Context (cumulative_left_axis : forall i,
    cumulative (S i) 0 = add (cumulative i 0) (delete_left (S i))).
  Context (cumulative_right_axis : forall j,
    cumulative 0 (S j) = add (cumulative 0 j) (delete_right (S j))).
  Context (cumulative_inner : forall i j,
    cumulative (S i) (S j) = min3
      (add (cumulative i (S j)) (delete_left (S i)))
      (add (cumulative (S i) j) (delete_right (S j)))
      (add (cumulative i j) (match_cost (S i) (S j)))).

  (** Index one denotes the common anchor in each source sequence. An axis
      with exactly one positive index is the anchored cumulative boundary. *)
  Definition anchored_grid (i j : nat) : option Cost :=
    match i, j with
    | 0, 0 => Some zero
    | 0, S _ | S _, 0 => None
    | S i', S j' => Some (cumulative i' j')
    end.

  Definition shifted_left (i : nat) : Cost :=
    match i with 0 => zero | S previous => delete_left (S previous) end.
  Definition shifted_right (j : nat) : Cost :=
    match j with 0 => zero | S previous => delete_right (S previous) end.
  Definition shifted_match (i j : nat) : Cost :=
    match i, j with
    | 0, 0 => zero
    | S previous, S prior => match_cost (S previous) (S prior)
    | _, _ => zero
    end.

  Definition source_cell
      (grid : nat -> nat -> option Cost) (i j : nat) : option Cost :=
    option_min3
      (option_add (grid i (S j)) (shifted_left i))
      (option_add (grid (S i) j) (shifted_right j))
      (option_add (grid i j) (shifted_match i j)).

  Lemma anchored_axes : forall i j,
    anchored_grid 0 0 = Some zero /\
    anchored_grid (S i) 0 = None /\
    anchored_grid 0 (S j) = None.
  Proof. intros; repeat split; reflexivity. Qed.

  Theorem anchored_grid_satisfies_source_recurrence : forall i j,
    anchored_grid (S i) (S j) = source_cell anchored_grid i j.
  Proof.
    intros [|i] [|j]; unfold source_cell, anchored_grid,
      shifted_left, shifted_right, shifted_match,
      option_min3, option_min, option_add, min3; simpl.
    - now rewrite cumulative_origin, add_zero_zero.
    - now rewrite cumulative_right_axis.
    - now rewrite cumulative_left_axis.
    - now rewrite cumulative_inner.
  Qed.

  (** Every grid with the same infinite axes and source recurrence equals the
      embedded cumulative grid. Induction on rows, then columns, follows the
      north/west/diagonal dependency order of the executable DP. *)
  Theorem source_grid_is_unique : forall source,
    source 0 0 = Some zero ->
    (forall i, source (S i) 0 = None) ->
    (forall j, source 0 (S j) = None) ->
    (forall i j, source (S i) (S j) = source_cell source i j) ->
    forall i j, source i j = anchored_grid i j.
  Proof.
    intros source Horigin Hleft Hright Hstep i.
    induction i as [|i IHi]; intro j.
    - destruct j as [|j]; [exact Horigin | now rewrite Hright].
    - induction j as [|j IHj].
      + now rewrite Hleft.
      + rewrite Hstep, anchored_grid_satisfies_source_recurrence.
        unfold source_cell.
        rewrite IHi, IHj, IHi.
        reflexivity.
  Qed.

  Corollary forced_anchor_match_equals_cumulative_score : forall source,
    source 0 0 = Some zero ->
    (forall i, source (S i) 0 = None) ->
    (forall j, source 0 (S j) = None) ->
    (forall i j, source (S i) (S j) = source_cell source i j) ->
    forall i j, source (S i) (S j) = Some (cumulative i j).
  Proof.
    intros source Horigin Hleft Hright Hstep i j.
    rewrite (source_grid_is_unique source Horigin Hleft Hright Hstep).
    reflexivity.
  Qed.
End AnchorRecurrence.

Open Scope R_scope.

Definition sample := (R * R)%type.
Definition sample_time (point : sample) : R := snd point.

Definition shift_sample (origin : R) (point : sample) : sample :=
  (fst point, 1 + sample_time point - origin).

Definition shift_time (origin time : R) : R := 1 + time - origin.

Lemma shifted_elapsed_time : forall origin current previous,
  shift_time origin current - shift_time origin previous =
    current - previous.
Proof. intros; unfold shift_time; lra. Qed.

Lemma shifted_cross_time : forall origin left right,
  Rabs (shift_time origin left - shift_time origin right) =
    Rabs (left - right).
Proof. intros; now rewrite shifted_elapsed_time. Qed.

Definition physical_delete
    (nu lambda : R) (current previous : sample) : R :=
  Rabs (fst current - fst previous) +
  nu * (sample_time current - sample_time previous) + lambda.

Definition physical_match
    (nu : R) (left previous_left right previous_right : sample) : R :=
  Rabs (fst left - fst right) +
  Rabs (fst previous_left - fst previous_right) +
  nu * (Rabs (sample_time left - sample_time right) +
        Rabs (sample_time previous_left - sample_time previous_right)).

Lemma shifted_delete_cost_is_original : forall origin nu lambda current previous,
  physical_delete nu lambda
    (shift_sample origin current) (shift_sample origin previous) =
  physical_delete nu lambda current previous.
Proof.
  intros origin nu lambda [current_value current_time]
    [previous_value previous_time].
  unfold physical_delete, shift_sample, sample_time; simpl.
  replace ((1 + current_time - origin) - (1 + previous_time - origin))
    with (current_time - previous_time) by lra.
  reflexivity.
Qed.

Lemma shifted_match_cost_is_original : forall origin nu
    left previous_left right previous_right,
  physical_match nu
    (shift_sample origin left) (shift_sample origin previous_left)
    (shift_sample origin right) (shift_sample origin previous_right) =
  physical_match nu left previous_left right previous_right.
Proof.
  intros origin nu [lv lt] [lpv lpt] [rv rt] [rpv rpt].
  unfold physical_match, shift_sample, sample_time; simpl.
  replace ((1 + lt - origin) - (1 + rt - origin))
    with (lt - rt) by lra.
  replace ((1 + lpt - origin) - (1 + rpt - origin))
    with (lpt - rpt) by lra.
  reflexivity.
Qed.

Lemma shifted_first_delete_cost_is_original : forall origin nu lambda first,
  physical_delete nu lambda
    (shift_sample origin first) (0, 1) =
  physical_delete nu lambda first (0, origin).
Proof.
  intros origin nu lambda [value time].
  unfold physical_delete, shift_sample, sample_time; simpl.
  replace ((1 + time - origin) - 1) with (time - origin) by lra.
  reflexivity.
Qed.

Lemma shifted_first_match_cost_is_original : forall origin nu left right,
  physical_match nu
    (shift_sample origin left) (0, 1)
    (shift_sample origin right) (0, 1) =
  physical_match nu left (0, origin) right (0, origin).
Proof.
  intros origin nu [lv lt] [rv rt].
  unfold physical_match, shift_sample, sample_time; simpl.
  replace ((1 + lt - origin) - (1 + rt - origin))
    with (lt - rt) by lra.
  replace (1 - 1) with 0 by ring.
  replace (origin - origin) with 0 by ring.
  now rewrite !Rabs_R0.
Qed.

Lemma common_anchor_match_cost_is_zero : forall nu,
  physical_match nu (0, 1) (0, 1) (0, 1) (0, 1) = 0.
Proof.
  intro nu; unfold physical_match, sample_time; simpl.
  repeat rewrite Rminus_diag_eq by reflexivity.
  repeat rewrite Rabs_R0. ring.
Qed.

(** The old constructor permits the first target sample at the origin.
    Its deletion and the subsequent match are both free at zero gap penalty,
    even though the two raw series differ. This is the concrete identity
    obstruction excluded by the strict-origin pullback domain. *)
Definition origin_counterexample_left : list sample := [(1, 1)].
Definition origin_counterexample_right : list sample := [(0, 0); (1, 1)].

Theorem origin_equal_raw_series_are_distinct :
  origin_counterexample_left <> origin_counterexample_right.
Proof. discriminate. Qed.

Theorem generic_origin_equal_raw_series_are_distinct :
  forall (Point : Type) (sentinel later : Point),
    [later] <> [sentinel; later].
Proof. intros; discriminate. Qed.

Theorem origin_equal_extra_sample_has_zero_cost_script :
  physical_delete 1 0 (0, 0) (0, 0) +
  physical_match 1 (1, 1) (0, 0) (1, 1) (0, 0) = 0.
Proof.
  unfold physical_delete, physical_match, sample_time; simpl.
  repeat rewrite Rminus_diag_eq by reflexivity.
  repeat rewrite Rabs_R0.
  ring.
Qed.

Definition finite_min (left right : R) : R :=
  if Rle_dec left right then left else right.

Lemma finite_min_nonnegative : forall left right,
  0 <= left -> 0 <= right -> 0 <= finite_min left right.
Proof.
  intros left right Hleft Hright.
  unfold finite_min; destruct (Rle_dec left right); lra.
Qed.

Lemma finite_min_zero_right : forall left,
  0 <= left -> finite_min left 0 = 0.
Proof.
  intros left Hleft.
  unfold finite_min; destruct (Rle_dec left 0); lra.
Qed.

(** The 1-by-2 cumulative DP grid for the distinct raw series. The final
    diagonal predecessor is the zero-cost deletion of the origin-equal sample.
    All other alternatives are nonnegative, so the minimum is exactly zero. *)
Theorem origin_equal_cumulative_recurrence_is_zero :
  let anchor := (0, 0) in
  let left := (1, 1) in
  let first_right := (0, 0) in
  let second_right := (1, 1) in
  let d01 := physical_delete 1 0 first_right anchor in
  let d10 := physical_delete 1 0 left anchor in
  let d02 := d01 + physical_delete 1 0 second_right first_right in
  let d11 := finite_min
    (d01 + physical_delete 1 0 left anchor)
    (finite_min (d10 + physical_delete 1 0 first_right anchor)
      (physical_match 1 left anchor first_right anchor)) in
  finite_min
    (d02 + physical_delete 1 0 left anchor)
    (finite_min (d11 + physical_delete 1 0 second_right first_right)
      (d01 + physical_match 1 left anchor second_right first_right)) = 0.
Proof.
  cbv beta zeta.
  unfold physical_delete, physical_match, sample_time; simpl.
  repeat rewrite Rminus_diag_eq by reflexivity.
  repeat rewrite Rabs_R0.
  replace (1 - 0) with 1 by ring.
  replace (0 - 1) with (-1) by ring.
  repeat rewrite Rabs_R1.
  replace (1 - 1) with 0 by ring.
  repeat rewrite Rabs_R0.
  unfold finite_min.
  repeat (destruct (Rle_dec _ _)); lra.
Qed.

(** The same origin-equal obstruction is independent of the shape of a
    sample. In particular, it applies to typed vectors under any
    nonnegative point cost with zero self-distance. The following is the
    entire 1-by-2 cumulative grid, not only one zero-cost script. *)
Section GenericGroundOriginControl.
  Context {Point : Type}.
  Variable ground : Point -> Point -> R.
  Context (ground_nonnegative : forall left right,
    0 <= ground left right).
  Context (ground_reflexive : forall point,
    ground point point = 0).

  Definition generic_delete (current previous : Point)
      (current_time previous_time : R) : R :=
    ground current previous + Rabs (current_time - previous_time).

  Definition generic_match (current_left previous_left : Point)
      (time_left previous_time_left : R)
      (current_right previous_right : Point)
      (time_right previous_time_right : R) : R :=
    ground current_left current_right +
    ground previous_left previous_right +
    Rabs (time_left - time_right) +
    Rabs (previous_time_left - previous_time_right).

  Lemma generic_delete_nonnegative : forall current previous time earlier,
    0 <= generic_delete current previous time earlier.
  Proof.
    intros; unfold generic_delete.
    pose proof (ground_nonnegative current previous).
    pose proof (Rabs_pos (time - earlier)). lra.
  Qed.

  Lemma generic_match_nonnegative : forall current_left previous_left
      time_left previous_time_left current_right previous_right
      time_right previous_time_right,
    0 <= generic_match current_left previous_left time_left
      previous_time_left current_right previous_right time_right
      previous_time_right.
  Proof.
    intros; unfold generic_match.
    pose proof (ground_nonnegative current_left current_right).
    pose proof (ground_nonnegative previous_left previous_right).
    pose proof (Rabs_pos (time_left - time_right)).
    pose proof (Rabs_pos (previous_time_left - previous_time_right)).
    lra.
  Qed.

  Theorem generic_origin_equal_cumulative_recurrence_is_zero :
    forall sentinel later,
      let d01 := generic_delete sentinel sentinel 0 0 in
      let d10 := generic_delete later sentinel 1 0 in
      let d02 := d01 + generic_delete later sentinel 1 0 in
      let d11 := finite_min
        (d01 + generic_delete later sentinel 1 0)
        (finite_min
          (d10 + generic_delete sentinel sentinel 0 0)
          (generic_match later sentinel 1 0 sentinel sentinel 0 0)) in
      finite_min
        (d02 + generic_delete later sentinel 1 0)
        (finite_min
          (d11 + generic_delete later sentinel 1 0)
          (d01 + generic_match later sentinel 1 0 later sentinel 1 0)) = 0.
  Proof.
    intros sentinel later.
    cbv beta zeta.
    set (zero_step := generic_delete sentinel sentinel 0 0).
    set (later_step := generic_delete later sentinel 1 0).
    set (first_match :=
      generic_match later sentinel 1 0 sentinel sentinel 0 0).
    set (last_match :=
      generic_match later sentinel 1 0 later sentinel 1 0).
    assert (Hzero : zero_step = 0).
    { unfold zero_step, generic_delete.
      rewrite ground_reflexive.
      replace (0 - 0) with 0 by ring.
      rewrite Rabs_R0. ring. }
    assert (Hlast : last_match = 0).
    { unfold last_match, generic_match.
      repeat rewrite ground_reflexive.
      replace (1 - 1) with 0 by ring.
      replace (0 - 0) with 0 by ring.
      repeat rewrite Rabs_R0. ring. }
    assert (Hlater : 0 <= later_step).
    { unfold later_step; apply generic_delete_nonnegative. }
    assert (Hfirst : 0 <= first_match).
    { unfold first_match; apply generic_match_nonnegative. }
    assert (Hmiddle :
      0 <= finite_min (zero_step + later_step)
        (finite_min (later_step + zero_step) first_match)).
    { apply finite_min_nonnegative; [lra |].
      apply finite_min_nonnegative; lra. }
    rewrite Hzero in Hmiddle.
    rewrite Hzero, Hlast.
    replace (0 + 0) with 0 by ring.
    rewrite finite_min_zero_right.
    - apply finite_min_zero_right; lra.
    - lra.
  Qed.
End GenericGroundOriginControl.

Definition anchor_series (origin : R) (series : list sample) : list sample :=
  (0, 1) :: map (shift_sample origin) series.

Fixpoint strict_times (series : list sample) : Prop :=
  match series with
  | [] => True
  | first :: rest =>
      match rest with
      | [] => True
      | second :: _ =>
          sample_time first < sample_time second /\ strict_times rest
      end
  end.

Definition strict_after_origin (origin : R) (series : list sample) : Prop :=
  match series with
  | [] => True
  | first :: _ => origin < sample_time first /\ strict_times series
  end.

(** The raw Rust constructor requires a nonempty series, first time no
    earlier than its origin, and strictly increasing successive times. The
    additive strict wrapper checks the one missing strict inequality. *)
Definition raw_constructor_domain (origin : R) (series : list sample) : Prop :=
  match series with
  | [] => False
  | first :: _ => origin <= sample_time first /\ strict_times series
  end.

Definition strict_wrapper_domain (origin : R) (series : list sample) : Prop :=
  raw_constructor_domain origin series /\
  match series with
  | [] => False
  | first :: _ => origin < sample_time first
  end.

Theorem strict_wrapper_enters_transfer_domain : forall origin series,
  strict_wrapper_domain origin series -> strict_after_origin origin series.
Proof.
  intros origin [| first rest] Hdomain; simpl in *; [tauto |].
  destruct Hdomain as [[_ Htimes] Hfirst].
  now split.
Qed.

Theorem origin_equal_counterexample_passes_raw_constructor :
  raw_constructor_domain 0 origin_counterexample_right.
Proof.
  unfold raw_constructor_domain, origin_counterexample_right,
    strict_times, sample_time; simpl; split; lra.
Qed.

Theorem origin_equal_counterexample_fails_strict_wrapper :
  ~ strict_wrapper_domain 0 origin_counterexample_right.
Proof.
  unfold strict_wrapper_domain, origin_counterexample_right,
    sample_time; simpl. lra.
Qed.

Lemma shift_sample_injective : forall origin first second,
  shift_sample origin first = shift_sample origin second -> first = second.
Proof.
  intros origin [first_value first_time] [second_value second_time] Hequal.
  unfold shift_sample, sample_time in Hequal; simpl in Hequal.
  inversion Hequal; subst.
  assert (first_time = second_time) by lra.
  subst; reflexivity.
Qed.

Theorem anchor_series_injective : forall origin first second,
  anchor_series origin first = anchor_series origin second -> first = second.
Proof.
  intros origin first second Hequal.
  apply (f_equal (@tl sample)) in Hequal.
  unfold anchor_series in Hequal; simpl in Hequal.
  revert second Hequal.
  induction first as [| head tail IH];
    intros [| other remaining] Hequal; simpl in Hequal;
    try discriminate; [reflexivity |].
  pose proof (f_equal (hd (0, 0)) Hequal) as Hhead.
  pose proof (f_equal (@tl sample) Hequal) as Htail.
  simpl in Hhead, Htail.
  apply shift_sample_injective in Hhead.
  apply IH in Htail.
  now subst.
Qed.

Lemma shifted_strict_times : forall origin series,
  strict_times series -> strict_times (map (shift_sample origin) series).
Proof.
  intros origin series.
  induction series as [|first [|second tail] IH]; simpl; auto.
  intros [Htime Htail]; split.
  - unfold sample_time, shift_sample in *; simpl in *; lra.
  - apply IH. exact Htail.
Qed.

Theorem anchor_series_has_strict_times : forall origin series,
  strict_after_origin origin series ->
  strict_times (anchor_series origin series).
Proof.
  intros origin [|first tail] Hdomain; simpl; [exact I |].
  destruct Hdomain as [Hfirst Hstrict].
  split.
  - unfold sample_time, shift_sample in *; simpl in *; lra.
  - change (strict_times (map (shift_sample origin) (first :: tail))).
    now apply shifted_strict_times.
Qed.

(** Source metric laws are needed only on this image. Empty physical input
    maps to the nonempty anchor-only source series; the empty source series
    is excluded, even though it satisfies [strict_times]. *)
Definition anchored_source_domain (origin : R) (anchored : list sample) : Prop :=
  exists physical,
    strict_after_origin origin physical /\
    anchored = anchor_series origin physical.

Lemma anchor_series_enters_source_domain : forall origin physical,
  strict_after_origin origin physical ->
  anchored_source_domain origin (anchor_series origin physical).
Proof.
  intros origin physical Hdomain.
  exists physical; split; [exact Hdomain | reflexivity].
Qed.

Lemma anchored_source_domain_is_nonempty : forall origin anchored,
  anchored_source_domain origin anchored -> anchored <> [].
Proof.
  intros origin anchored [physical [Hdomain Hanchored]].
  subst anchored; unfold anchor_series; discriminate.
Qed.

Lemma anchored_source_domain_has_strict_times : forall origin anchored,
  anchored_source_domain origin anchored -> strict_times anchored.
Proof.
  intros origin anchored [physical [Hdomain Hanchored]].
  subst anchored; now apply anchor_series_has_strict_times.
Qed.

Example empty_physical_input_admitted_but_empty_source_excluded : forall origin,
  anchored_source_domain origin [(0, 1)] /\
  strict_times ([] : list sample) /\
  ~ anchored_source_domain origin [].
Proof.
  intro origin; split.
  - change (anchored_source_domain origin (anchor_series origin [])).
    apply anchor_series_enters_source_domain; exact I.
  - split; [exact I |].
    intro Hdomain.
    apply (anchored_source_domain_is_nonempty origin [] Hdomain).
    reflexivity.
Qed.

(** [source_distance] is a real-valued view of finite source scores on the
    common-anchor image. Outside that image its values are unconstrained and
    need not form a metric; in particular no finite real value is claimed to
    represent the source's infinite empty-to-nonempty boundary.

    Source metricity is an explicit premise. To qualify a recurrence score,
    the final theorem additionally requires equality with this pullback.
    Establishing that equality for a concrete recurrence must bind its source
    scores to the option-valued grid and instantiate the recurrence theorem
    above; merely naming a function a recurrence score does not do so. *)
Section MetricPullback.
  Variable source_distance : list sample -> list sample -> R.
  Variable origin : R.
  Context (source_nonnegative : forall x y,
    anchored_source_domain origin x -> anchored_source_domain origin y ->
    0 <= source_distance x y).
  Context (source_symmetric : forall x y,
    anchored_source_domain origin x -> anchored_source_domain origin y ->
    source_distance x y = source_distance y x).
  Context (source_identity : forall x y,
    anchored_source_domain origin x -> anchored_source_domain origin y ->
    source_distance x y = 0 <-> x = y).
  Context (source_triangle : forall x y z,
    anchored_source_domain origin x -> anchored_source_domain origin y ->
    anchored_source_domain origin z ->
    source_distance x z <= source_distance x y + source_distance y z).

  Definition cumulative_distance (x y : list sample) : R :=
    source_distance (anchor_series origin x) (anchor_series origin y).

  Lemma anchored_source_pullback_is_metric : forall x y z,
    strict_after_origin origin x ->
    strict_after_origin origin y ->
    strict_after_origin origin z ->
    0 <= cumulative_distance x y /\
    cumulative_distance x y = cumulative_distance y x /\
    (cumulative_distance x y = 0 <-> x = y) /\
    cumulative_distance x z <=
      cumulative_distance x y + cumulative_distance y z.
  Proof.
    intros x y z Hx Hy Hz.
    unfold cumulative_distance.
    pose proof (anchor_series_enters_source_domain origin x Hx) as Hax.
    pose proof (anchor_series_enters_source_domain origin y Hy) as Hay.
    pose proof (anchor_series_enters_source_domain origin z Hz) as Haz.
    split.
    - now apply source_nonnegative.
    - split.
      + now apply source_symmetric.
      + split.
        * split.
          -- intro Hzero.
             apply anchor_series_injective with (origin := origin).
             now apply (proj1 (source_identity _ _ Hax Hay)).
          -- intro Hequal. subst.
             apply (proj2 (source_identity _ _ Hay Hay)); reflexivity.
        * now apply source_triangle.
  Qed.

  Variable cumulative_recurrence_score : list sample -> list sample -> R.
  Context (source_recurrence_correspondence : forall x y,
    strict_after_origin origin x -> strict_after_origin origin y ->
    cumulative_recurrence_score x y = cumulative_distance x y).

  Theorem strict_origin_pullback_is_metric : forall x y z,
    strict_after_origin origin x ->
    strict_after_origin origin y ->
    strict_after_origin origin z ->
    0 <= cumulative_recurrence_score x y /\
    cumulative_recurrence_score x y = cumulative_recurrence_score y x /\
    (cumulative_recurrence_score x y = 0 <-> x = y) /\
    cumulative_recurrence_score x z <=
      cumulative_recurrence_score x y + cumulative_recurrence_score y z.
  Proof.
    intros x y z Hx Hy Hz.
    rewrite (source_recurrence_correspondence x y Hx Hy),
      (source_recurrence_correspondence y x Hy Hx),
      (source_recurrence_correspondence x z Hx Hz),
      (source_recurrence_correspondence y z Hy Hz).
    now apply anchored_source_pullback_is_metric.
  Qed.
End MetricPullback.
