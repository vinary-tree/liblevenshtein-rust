(** * Representation-preserving subsumption fallback

    Phase 11 deliberately retains separate integer and floating position
    representations.  This model proves that factoring their common
    Standard/OSA/MergeSplit decision tree through [shared_subsumes] is
    extensionally equal to the legacy per-mode organization for every cost
    carrier and every carrier-specific comparison predicate.
*)

From Stdlib Require Import Arith Bool Lia.

Inductive mode : Type := Standard | Transposition | MergeSplit.

Definition index_distance (left right : nat) : nat :=
  if left <=? right then right - left else left - right.

Section GenericCarrier.
  Variable Cost : Type.
  Variable non_greater : Cost -> Cost -> Prop.
  Variable realignment_fits : Cost -> Cost -> nat -> Cost -> Prop.
  Variable strictly_less : Cost -> Cost -> Prop.

  Definition shared_subsumes
      (kind : mode)
      (li : nat) (lc : Cost) (ls : bool)
      (ri : nat) (rc : Cost) (rs : bool)
      (query_length : nat) (index_step : Cost) : Prop :=
    non_greater lc rc /\
    match kind with
    | Standard => realignment_fits lc rc (index_distance li ri) index_step
    | Transposition =>
        match ls, rs with
        | true, true => li = ri
        | false, false => realignment_fits lc rc (index_distance li ri) index_step
        | _, _ => False
        end
    | MergeSplit =>
        ls = rs /\
        li <= query_length /\
        ~ (ls = true /\ li >= query_length /\ ri < query_length) /\
        strictly_less lc rc /\
        li = ri
    end.

  Definition legacy_subsumes
      (kind : mode)
      (li : nat) (lc : Cost) (ls : bool)
      (ri : nat) (rc : Cost) (rs : bool)
      (query_length : nat) (index_step : Cost) : Prop :=
    match kind with
    | Standard =>
        non_greater lc rc /\
        realignment_fits lc rc (index_distance li ri) index_step
    | Transposition =>
        match ls, rs with
        | true, true => non_greater lc rc /\ li = ri
        | false, false =>
            non_greater lc rc /\
            realignment_fits lc rc (index_distance li ri) index_step
        | _, _ => False
        end
    | MergeSplit =>
        non_greater lc rc /\
        ls = rs /\
        li <= query_length /\
        ~ (ls = true /\ li >= query_length /\ ri < query_length) /\
        strictly_less lc rc /\
        li = ri
    end.

  Theorem shared_subsumes_is_legacy_subsumes : forall
      kind li lc ls ri rc rs query_length index_step,
    shared_subsumes kind li lc ls ri rc rs query_length index_step <->
    legacy_subsumes kind li lc ls ri rc rs query_length index_step.
  Proof.
    intros [] li lc [] ri rc [] query_length index_step;
      unfold shared_subsumes, legacy_subsumes; simpl; tauto.
  Qed.

  Theorem transposition_mixed_continuations_never_subsume : forall
      li lc ri rc query_length index_step,
    ~ shared_subsumes Transposition li lc true ri rc false query_length index_step /\
    ~ shared_subsumes Transposition li lc false ri rc true query_length index_step.
  Proof.
    intros; unfold shared_subsumes; simpl; tauto.
  Qed.

  Theorem merge_split_requires_same_index_and_kind : forall
      li lc ls ri rc rs query_length index_step,
    shared_subsumes MergeSplit li lc ls ri rc rs query_length index_step ->
    li = ri /\ ls = rs.
  Proof.
    intros; unfold shared_subsumes in *; simpl in *; tauto.
  Qed.
End GenericCarrier.
