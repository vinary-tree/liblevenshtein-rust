(** * Exact-rational center/radius enclosure (NC-1)

    Distances and endpoints here are mathematical rationals. The metric
    record exposes every law used by the enclosure theorem. A region member
    must be proved to lie within the stated radius of the same center;
    naming a center and radius alone does not establish that premise.

    This is neither a binary64 rounding theorem nor a Rust interval-bound
    correspondence. A producer must prove its actual region membership,
    center score, radius, arithmetic operation graph, and outward endpoint
    conversion. Inward-rounded endpoint controls below show why copying
    these exact inequalities into floating-point code is unsound. *)

From Stdlib Require Import Bool QArith micromega.Lqa.
Local Open Scope Q_scope.

Record rational_metric (Point : Type) := {
  metric_distance : Point -> Point -> Q;
  metric_nonnegative : forall first second,
    0 <= metric_distance first second;
  metric_reflexive : forall point,
    metric_distance point point == 0;
  metric_symmetric : forall first second,
    metric_distance first second == metric_distance second first;
  metric_separates : forall first second,
    metric_distance first second == 0 -> first = second;
  metric_triangle : forall first middle last,
    metric_distance first last <=
      metric_distance first middle + metric_distance middle last
}.

Arguments metric_distance {Point} _ _ _.
Arguments metric_nonnegative {Point} _ _ _.
Arguments metric_reflexive {Point} _ _.
Arguments metric_symmetric {Point} _ _ _.
Arguments metric_separates {Point} _ _ _ _.
Arguments metric_triangle {Point} _ _ _ _.

Section Enclosure.
  Context {Point : Type}.
  Variable metric : rational_metric Point.

  Definition member_of_ball (center : Point) (radius : Q)
      (member : Point) : Prop :=
    metric_distance metric center member <= radius.

  (** This exact lower endpoint is max(0, d(query,center)-radius). *)
  Definition ball_lower (query center : Point) (radius : Q) : Q :=
    if Qlt_le_dec (metric_distance metric query center) radius
    then 0
    else metric_distance metric query center - radius.

  Definition ball_upper (query center : Point) (radius : Q) : Q :=
    metric_distance metric query center + radius.

  Theorem nc1_reverse_triangle_to_center :
    forall query center member radius,
      member_of_ball center radius member ->
      metric_distance metric query center <=
        metric_distance metric query member + radius.
  Proof.
    intros query center member radius Hmember.
    unfold member_of_ball in Hmember.
    pose proof (metric_triangle metric query member center) as Htriangle.
    pose proof (metric_symmetric metric member center) as Hsymmetric.
    lra.
  Qed.

  Theorem nc1_lower_endpoint_nonnegative :
    forall query center radius,
      0 <= ball_lower query center radius.
  Proof.
    intros query center radius.
    unfold ball_lower.
    destruct (Qlt_le_dec (metric_distance metric query center) radius); lra.
  Qed.

  Theorem nc1_lower_endpoint_correct :
    forall query center member radius,
      0 <= radius -> member_of_ball center radius member ->
      ball_lower query center radius <=
        metric_distance metric query member.
  Proof.
    intros query center member radius Hradius Hmember.
    pose proof (nc1_reverse_triangle_to_center
      query center member radius Hmember) as Hreverse.
    pose proof (metric_nonnegative metric query member) as Hnonnegative.
    unfold ball_lower.
    destruct (Qlt_le_dec (metric_distance metric query center) radius); lra.
  Qed.

  Theorem nc1_upper_endpoint_correct :
    forall query center member radius,
      0 <= radius -> member_of_ball center radius member ->
      metric_distance metric query member <=
        ball_upper query center radius.
  Proof.
    intros query center member radius Hradius Hmember.
    unfold member_of_ball in Hmember.
    unfold ball_upper.
    pose proof (metric_triangle metric query center member) as Htriangle.
    lra.
  Qed.

  Theorem nc1_exact_rational_enclosure :
    forall query center member radius,
      0 <= radius -> member_of_ball center radius member ->
      0 <= ball_lower query center radius /\
      ball_lower query center radius <=
        metric_distance metric query member /\
      metric_distance metric query member <=
        ball_upper query center radius.
  Proof.
    intros query center member radius Hradius Hmember.
    repeat split.
    - apply nc1_lower_endpoint_nonnegative.
    - apply nc1_lower_endpoint_correct; assumption.
    - apply nc1_upper_endpoint_correct; assumption.
  Qed.

  Corollary nc1_strict_cutoff_prune_is_sound :
    forall query center member radius cutoff,
      0 <= radius -> member_of_ball center radius member ->
      cutoff < ball_lower query center radius ->
      cutoff < metric_distance metric query member.
  Proof.
    intros query center member radius cutoff Hradius Hmember Hcutoff.
    pose proof (nc1_lower_endpoint_correct
      query center member radius Hradius Hmember) as Hlower.
    lra.
  Qed.
End Enclosure.

(** A two-point metric is enough to expose both missing-membership and
    inward-endpoint errors. The controls are exact-rational counterexamples,
    not simulated IEEE operations. *)
Definition two_point_distance (first second : bool) : Q :=
  if Bool.eqb first second then 0 else 2.

Definition two_point_metric : rational_metric bool.
Proof.
  refine {| metric_distance := two_point_distance |}.
  - intros [] []; unfold two_point_distance; simpl; lra.
  - intros []; unfold two_point_distance; simpl; reflexivity.
  - intros [] []; unfold two_point_distance; simpl; reflexivity.
  - intros [] [] Hzero; unfold two_point_distance in Hzero;
      simpl in Hzero; try reflexivity; lra.
  - intros [] [] []; unfold two_point_distance; simpl; lra.
Defined.

Example unscoped_region_breaks_upper_endpoint :
  ~ (forall member : bool,
      metric_distance two_point_metric false member <=
        ball_upper two_point_metric false false 0).
Proof.
  intro Hall.
  specialize (Hall true).
  change (2 <= 0) in Hall.
  lra.
Qed.

Example inward_lower_rounding_breaks_enclosure :
  member_of_ball two_point_metric false 2 true /\
  ~ (1 <= metric_distance two_point_metric true true).
Proof.
  split.
  - change (2 <= 2). lra.
  - change (~ (1 <= 0)). lra.
Qed.

Example inward_upper_rounding_breaks_enclosure :
  member_of_ball two_point_metric false 2 true /\
  ~ (metric_distance two_point_metric false true <= 1).
Proof.
  split.
  - change (2 <= 2). lra.
  - change (~ (2 <= 1)). lra.
Qed.
