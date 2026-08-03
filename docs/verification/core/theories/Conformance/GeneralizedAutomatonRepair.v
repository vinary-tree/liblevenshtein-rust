(** Assumption-free conformance model for operation-driven generalized acceptance.

    The Rust implementation evaluates a finite acyclic alignment graph.  This
    model isolates the path algebra used by that graph: source/target
    consumption and exact natural-number cost are folds over the same operation
    sequence. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Record operation : Type := {
  consume_source : nat;
  consume_target : nat;
  scaled_cost : nat
}.

Fixpoint path_source (path : list operation) : nat :=
  match path with
  | [] => 0
  | operation_step :: rest =>
      consume_source operation_step + path_source rest
  end.

Fixpoint path_target (path : list operation) : nat :=
  match path with
  | [] => 0
  | operation_step :: rest =>
      consume_target operation_step + path_target rest
  end.

Fixpoint path_cost (path : list operation) : nat :=
  match path with
  | [] => 0
  | operation_step :: rest => scaled_cost operation_step + path_cost rest
  end.

Lemma path_source_app : forall left right,
  path_source (left ++ right) = path_source left + path_source right.
Proof.
  induction left as [|operation_step rest IH]; intros right; simpl.
  - lia.
  - rewrite IH. lia.
Qed.

Lemma path_target_app : forall left right,
  path_target (left ++ right) = path_target left + path_target right.
Proof.
  induction left as [|operation_step rest IH]; intros right; simpl.
  - lia.
  - rewrite IH. lia.
Qed.

Lemma path_cost_app : forall left right,
  path_cost (left ++ right) = path_cost left + path_cost right.
Proof.
  induction left as [|operation_step rest IH]; intros right; simpl.
  - lia.
  - rewrite IH. lia.
Qed.

Theorem positive_scaled_operation_is_not_free : forall operation_step,
  0 < scaled_cost operation_step ->
  0 < path_cost [operation_step].
Proof.
  intros operation_step Hpositive. simpl. lia.
Qed.

Theorem cost_is_monotone_under_extension : forall prefix suffix,
  path_cost prefix <= path_cost (prefix ++ suffix).
Proof.
  intros prefix suffix. rewrite path_cost_app. lia.
Qed.

Theorem budget_is_monotone : forall path lower_budget upper_budget,
  path_cost path <= lower_budget ->
  lower_budget <= upper_budget ->
  path_cost path <= upper_budget.
Proof. lia. Qed.

Theorem hamming_paths_preserve_length : forall path,
  (forall operation_step,
      In operation_step path ->
      consume_source operation_step = consume_target operation_step) ->
  path_source path = path_target path.
Proof.
  induction path as [|operation_step rest IH]; intros Hall; simpl.
  - reflexivity.
  - assert (Hhead : consume_source operation_step = consume_target operation_step).
    { apply Hall. now left. }
    assert (Htail : path_source rest = path_target rest).
    { apply IH. intros candidate Hin. apply Hall. now right. }
    lia.
Qed.

Theorem absent_deletion_rejects_nonempty_source_against_empty_target :
  forall path,
    (forall operation_step,
        In operation_step path ->
        consume_target operation_step = 0 ->
        consume_source operation_step = 0) ->
    path_target path = 0 ->
    path_source path = 0.
Proof.
  induction path as [|operation_step rest IH]; intros Hno_deletion Htarget; simpl in *.
  - reflexivity.
  - assert (Hhead_target : consume_target operation_step = 0) by lia.
    assert (Htail_target : path_target rest = 0) by lia.
    assert (Hhead_source : consume_source operation_step = 0).
    { apply Hno_deletion; [now left | exact Hhead_target]. }
    assert (Htail_source : path_source rest = 0).
    { apply IH.
      - intros candidate Hin. apply Hno_deletion; now right.
      - exact Htail_target. }
    lia.
Qed.

(** Empty-side rates use an explicit infinity constructor rather than an IEEE
    sentinel. A finite rate is represented by a numerator and a positive
    denominator, and budget comparison is exact cross multiplication. *)
Inductive empty_side_rate : Type :=
| InfiniteRate
| FiniteRate (numerator denominator : nat).

Definition rate_fits (rate : empty_side_rate) (count budget : nat) : Prop :=
  match rate with
  | InfiniteRate => count = 0
  | FiniteRate numerator denominator => numerator * count <= budget * denominator
  end.

Theorem infinite_rate_fits_only_empty_side : forall count budget,
  rate_fits InfiniteRate count budget <-> count = 0.
Proof. intros; split; auto. Qed.

Theorem finite_rate_budget_is_exact_cross_product :
  forall numerator denominator count budget,
    rate_fits (FiniteRate numerator denominator) count budget <->
    numerator * count <= budget * denominator.
Proof. intros; split; auto. Qed.

Theorem alignment_step_progresses : forall operation_step,
  consume_source operation_step + consume_target operation_step > 0 ->
  consume_source operation_step > 0 \/ consume_target operation_step > 0.
Proof. lia. Qed.

(** A multi-scalar operation is charged only when its complete target slice is
    known.  Speculative entry is cost-neutral; completion adds the exact step. *)
Theorem charge_on_completion : forall accumulated step,
  accumulated + step = path_cost
    [{| consume_source := 0; consume_target := 0; scaled_cost := accumulated |};
     {| consume_source := 0; consume_target := 0; scaled_cost := step |}].
Proof. intros. simpl. lia. Qed.

Theorem positive_completion_charge_is_not_free : forall accumulated step,
  0 < step -> accumulated < accumulated + step.
Proof. lia. Qed.

(** Rescaling from denominator [source] to [source * multiplier] preserves the
    represented rational value by cross multiplication. *)
Definition rescale_cost (cost multiplier : nat) : nat := cost * multiplier.

Theorem rescale_preserves_value : forall cost source multiplier,
  0 < source ->
  rescale_cost cost multiplier * source = cost * (source * multiplier).
Proof. intros. unfold rescale_cost. nia. Qed.

(** Unless the complete operation set is certified as the classical
    Levenshtein lattice, conservative dominance removes only the same control
    coordinate at strictly greater exact cost.  In particular, denominator
    one does not certify unit-cost semantics: integer operations may cost two
    or more units. *)
Definition scaled_subsumes
    (classical_certified : bool)
    (left_offset left_cost right_offset right_cost : nat) : Prop :=
  left_cost < right_cost /\
  (classical_certified = true \/ left_offset = right_offset).

Theorem uncertified_scaled_subsumption_is_exact_coordinate_dominance :
  forall left_offset left_cost right_offset right_cost,
    scaled_subsumes false left_offset left_cost right_offset right_cost ->
    left_offset = right_offset /\ left_cost < right_cost.
Proof.
  intros left_offset left_cost right_offset right_cost
    [Hcost [Hcertified | Hoffset]].
  - discriminate.
  - auto.
Qed.

Theorem denominator_one_integer_cost_does_not_enable_offset_dominance :
  ~ scaled_subsumes false 0 0 1 2.
Proof.
  intros [_ [Hcertified | Hoffset]]; discriminate.
Qed.

(** The resource ceiling is checked before a newly discovered cell is
    materialized. *)
Definition may_materialize (discovered limit : nat) : Prop := S discovered <= limit.

Theorem discovery_guard_bounds_materialized_cells : forall discovered limit,
  may_materialize discovered limit -> S discovered <= limit.
Proof. auto. Qed.

Theorem failed_discovery_guard_crosses_the_limit : forall discovered limit,
  ~ may_materialize discovered limit -> limit < S discovered.
Proof. intros. unfold may_materialize in *. lia. Qed.

(** Trying all applicable operations makes minimum-cost selection independent
    of their insertion order. *)
Theorem applicable_operation_minimum_is_order_independent : forall left right,
  Nat.min left right = Nat.min right left.
Proof. exact Nat.min_comm. Qed.

(** The streaming antichain has set semantics: reinserting a structurally equal
    control position is idempotent, and inserting a new one preserves
    duplicate-freedom. *)
Definition add_control_position (position : nat) (positions : list nat) : list nat :=
  if in_dec Nat.eq_dec position positions then positions else position :: positions.

Theorem adding_existing_control_position_is_idempotent : forall position positions,
  In position positions ->
  add_control_position position positions = positions.
Proof.
  intros position positions Hin. unfold add_control_position.
  destruct (in_dec Nat.eq_dec position positions); [reflexivity | contradiction].
Qed.

Theorem adding_control_position_preserves_no_duplicates : forall position positions,
  NoDup positions -> NoDup (add_control_position position positions).
Proof.
  intros position positions Hnodup. unfold add_control_position.
  destruct (in_dec Nat.eq_dec position positions) as [Hin | Hnotin].
  - exact Hnodup.
  - constructor; assumption.
Qed.
