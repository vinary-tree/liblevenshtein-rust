(** * Validation precedence and failure atomicity for a complete session

    A prepared action retains its complete predecessor and accumulates charged
    units and distinct executed-event identities.  It cannot mutate semantic
    state before publication.  Every failure at this action boundary returns
    Incomplete with the predecessor and its cumulative ledger.  A successful
    publication followed by a failing peak check is two actions, not a
    rollback of that publication.  A source correspondence proof must
    show that each Rust fallible branch implements one of these actions;
    this file does not infer that fact from a successful model check. *)

From Stdlib Require Import Arith Lia List.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedSearchSession CertifiedSessionResume.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Module TimestampedRangeValidation.
  (** These four predicates are observations of the public
      TimestampedTwedIndex::search_range_bounded validation path.  The
      source-to-predicate mapping remains an instance obligation. *)
  Record checks := {
    same_unit : bool;
    same_origin_bits : bool;
    query_length_allowed : bool;
    cutoff_allowed : bool
  }.

  Inductive validation_error :=
  | MixedUnits | MixedOrigins | QueryTooLong | InvalidCutoff.

  Definition first_error (request : checks) : option validation_error :=
    if same_unit request then
      if same_origin_bits request then
        if query_length_allowed request then
          if cutoff_allowed request then None else Some InvalidCutoff
        else Some QueryTooLong
      else Some MixedOrigins
    else Some MixedUnits.

  Theorem unit_error_precedes_all_other_errors : forall request,
    same_unit request = false -> first_error request = Some MixedUnits.
  Proof. intros [unit origin len cutoff] H; simpl in *; now rewrite H. Qed.

  Theorem origin_error_precedes_length_and_cutoff : forall request,
    same_unit request = true -> same_origin_bits request = false ->
    first_error request = Some MixedOrigins.
  Proof.
    intros [unit origin len cutoff] Hunit Horigin; simpl in *.
    now rewrite Hunit, Horigin.
  Qed.

  Theorem length_error_precedes_cutoff : forall request,
    same_unit request = true -> same_origin_bits request = true ->
    query_length_allowed request = false ->
    first_error request = Some QueryTooLong.
  Proof.
    intros [unit origin len cutoff] Hunit Horigin Hlen; simpl in *.
    now rewrite Hunit, Horigin, Hlen.
  Qed.

  Theorem cutoff_error_requires_earlier_validity : forall request,
    same_unit request = true -> same_origin_bits request = true ->
    query_length_allowed request = true ->
    cutoff_allowed request = false ->
    first_error request = Some InvalidCutoff.
  Proof.
    intros [unit origin len cutoff] Hunit Horigin Hlen Hcutoff; simpl in *.
    now rewrite Hunit, Horigin, Hlen, Hcutoff.
  Qed.

  Theorem admission_requires_all_four_checks : forall request,
    first_error request = None ->
    same_unit request = true /\ same_origin_bits request = true /\
    query_length_allowed request = true /\ cutoff_allowed request = true.
  Proof.
    intros [unit origin len cutoff] H; simpl in *.
    destruct unit, origin, len, cutoff; simpl in H;
      try discriminate; repeat split; reflexivity.
  Qed.

  Example simultaneous_unit_and_cutoff_fault_reports_unit :
    first_error
      {| same_unit := false; same_origin_bits := true;
         query_length_allowed := true; cutoff_allowed := false |} =
      Some MixedUnits.
  Proof. reflexivity. Qed.
End TimestampedRangeValidation.

Section FailureAction.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Let Runtime := session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost Score Evidence.
  Let Configuration := (Runtime * Ghost)%type.

  Definition replace_ledger_status (runtime : Runtime)
      (ledger : work_ledger) (status : session_status) : Runtime :=
    {| runtime_identity := runtime_identity runtime;
       runtime_pending := runtime_pending runtime;
       runtime_private := runtime_private runtime;
       runtime_selected := runtime_selected runtime;
       runtime_emission := runtime_emission runtime;
       runtime_arena := runtime_arena runtime;
       runtime_cache := runtime_cache runtime;
       runtime_ledger := ledger;
       runtime_reconstruction := runtime_reconstruction runtime;
       runtime_status := status |}.

  Definition with_ledger_status (configuration : Configuration)
      (ledger : work_ledger) (status : session_status) : Configuration :=
    (replace_ledger_status (fst configuration) ledger status,
     snd configuration).

  Definition semantic_contents (configuration : Configuration)
      : Configuration :=
    with_ledger_status configuration empty_ledger Active.

  Lemma semantic_contents_ignores_ledger_and_status :
    forall configuration ledger status,
      semantic_contents (with_ledger_status configuration ledger status) =
      semantic_contents configuration.
  Proof.
    intros [runtime ghost] ledger status.
    unfold semantic_contents, with_ledger_status; simpl.
    destruct runtime; reflexivity.
  Qed.

  Record prepared_action := {
    predecessor : Configuration;
    charged_increment : nat;
    executed_events : list nat;
    events_distinct : NoDup executed_events;
    execution_covered : length executed_events <= charged_increment
  }.

  Definition prepare (predecessor : Configuration) : prepared_action :=
    {| predecessor := predecessor;
       charged_increment := 0;
       executed_events := [];
       events_distinct := @NoDup_nil nat;
       execution_covered := Nat.le_refl 0 |}.

  Definition charge (action : prepared_action) (amount : nat)
      : prepared_action :=
    {| predecessor := predecessor action;
       charged_increment := charged_increment action + amount;
       executed_events := executed_events action;
       events_distinct := events_distinct action;
       execution_covered :=
         Nat.le_trans _ _ _ (execution_covered action)
           (Nat.le_add_r (charged_increment action) amount) |}.

  Definition execute (action : prepared_action) (event : nat)
      (reserved : length (executed_events action) < charged_increment action)
      (fresh : ~ In event (executed_events action)) : prepared_action :=
    {| predecessor := predecessor action;
       charged_increment := charged_increment action;
       executed_events := event :: executed_events action;
       events_distinct := @NoDup_cons nat event (executed_events action)
         fresh (events_distinct action);
       execution_covered := reserved |}.

  Definition accounted_ledger (action : prepared_action) : work_ledger :=
    let prior := runtime_ledger (fst (predecessor action)) in
    {| work_executed := work_executed prior + length (executed_events action);
       work_reserved := work_reserved prior;
       work_charged := work_charged prior + charged_increment action;
       bytes_live := bytes_live prior;
       bytes_peak := bytes_peak prior;
       allocation_count := allocation_count prior |}.

  Inductive stop_reason :=
  | BudgetRejected | ArithmeticRejected | AllocationRejected
  | NumericRejected | StoredDataRejected.

  Inductive disposition := RetainContinuation | EndContinuation.

  Inductive action_outcome :=
  | InvalidRequest : TimestampedRangeValidation.validation_error ->
      action_outcome
  | Incomplete : stop_reason -> Configuration ->
      option Configuration -> work_ledger -> action_outcome
  | Complete : Configuration -> action_outcome.

  Definition failure (action : prepared_action) (reason : stop_reason)
      (handling : disposition) : action_outcome :=
    let prior := predecessor action in
    let ledger := accounted_ledger action in
    let prefix := with_ledger_status prior ledger (Failed 0) in
    let continuation :=
      match handling with
      | RetainContinuation =>
          Some (with_ledger_status prior ledger (Suspended 0))
      | EndContinuation => None
      end in
    Incomplete reason prefix continuation ledger.

  Definition reject_or_prepare (checks : TimestampedRangeValidation.checks)
      (prior : Configuration) :
      action_outcome + prepared_action :=
    match TimestampedRangeValidation.first_error checks with
    | Some error => inl (InvalidRequest error)
    | None => inr (prepare prior)
    end.

  Theorem invalid_request_does_not_start_or_charge :
    forall checks prior error,
      TimestampedRangeValidation.first_error checks = Some error ->
      reject_or_prepare checks prior = inl (InvalidRequest error).
  Proof. intros checks prior error H; unfold reject_or_prepare; now rewrite H. Qed.

  Theorem failed_action_keeps_predecessor_contents :
    forall action reason handling prefix continuation ledger,
      failure action reason handling =
        Incomplete reason prefix continuation ledger ->
      semantic_contents prefix =
        semantic_contents (predecessor action) /\
      (forall resumed, continuation = Some resumed ->
        semantic_contents resumed =
          semantic_contents (predecessor action)).
  Proof.
    intros action reason handling prefix continuation ledger Hfailure.
    unfold failure in Hfailure.
    destruct handling; inversion Hfailure; subst; split.
    - apply semantic_contents_ignores_ledger_and_status.
    - intros resumed H; inversion H; subst.
      apply semantic_contents_ignores_ledger_and_status.
    - apply semantic_contents_ignores_ledger_and_status.
    - intros resumed H; discriminate.
  Qed.

  Theorem failed_action_reports_all_charges_and_work :
    forall action reason handling prefix continuation ledger,
      failure action reason handling =
        Incomplete reason prefix continuation ledger ->
      work_charged ledger =
        work_charged (runtime_ledger (fst (predecessor action))) +
          charged_increment action /\
      work_executed ledger =
        work_executed (runtime_ledger (fst (predecessor action))) +
          length (executed_events action) /\
      work_charged (runtime_ledger (fst (predecessor action))) <=
        work_charged ledger /\
      work_executed (runtime_ledger (fst (predecessor action))) <=
        work_executed ledger.
  Proof.
    intros action reason handling prefix continuation ledger Hfailure.
    unfold failure in Hfailure; destruct handling; inversion Hfailure;
      subst; simpl; repeat split; lia.
  Qed.

  Theorem failed_action_cannot_claim_complete :
    forall action reason handling completed,
      failure action reason handling <> Complete completed.
  Proof.
    intros action reason handling completed H.
    unfold failure in H; destruct handling; discriminate.
  Qed.

  Theorem failed_action_report_matches_retained_ledger :
    forall action reason prefix resumed ledger,
      failure action reason RetainContinuation =
        Incomplete reason prefix (Some resumed) ledger ->
      runtime_ledger (fst resumed) = ledger /\
      runtime_status (fst resumed) = Suspended 0 /\
      runtime_status (fst prefix) = Failed 0.
  Proof.
    intros action reason prefix resumed ledger Hfailure.
    unfold failure in Hfailure; inversion Hfailure; subst.
    repeat split; reflexivity.
  Qed.

  Theorem retained_failure_preserves_session_abstraction :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid action,
      session_abstraction region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid
        (fst (predecessor action)) (snd (predecessor action)) ->
      session_abstraction region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid
        (fst (with_ledger_status (predecessor action)
          (accounted_ledger action) (Suspended 0)))
        (snd (with_ledger_status (predecessor action)
          (accounted_ledger action) (Suspended 0))).
  Proof.
    intros region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid action Hinv.
    pose proof (execution_covered action) as Hcovered.
    unfold session_abstraction, exact_ownership, ownership_projection,
      borrowed_parent_live, selected_capacity, complete_cache_scoped in *.
    destruct (predecessor action) as [runtime ghost] eqn:Hpredecessor.
    destruct runtime as [identity pending private selected emission arena
      cache prior_ledger reconstruction status].
    unfold with_ledger_status, replace_ledger_status, accounted_ledger.
    rewrite Hpredecessor in *.
    simpl in *.
    repeat split; try tauto; try lia; discriminate.
  Qed.

  Theorem charged_failure_cannot_roll_back_reported_usage :
    forall action amount reason handling prefix continuation ledger,
      amount > 0 ->
      failure (charge action amount) reason handling =
        Incomplete reason prefix continuation ledger ->
      work_charged ledger >
        work_charged (runtime_ledger (fst (predecessor action))).
  Proof.
    intros action amount reason handling prefix continuation ledger
      Hpositive Hfailure.
    pose proof (failed_action_reports_all_charges_and_work Hfailure)
      as [Hcharge _].
    simpl in Hcharge; lia.
  Qed.

  Theorem executed_failure_cannot_erase_event :
    forall action event reserved fresh reason handling prefix continuation ledger,
      failure (execute action event reserved fresh) reason handling =
        Incomplete reason prefix continuation ledger ->
      work_executed ledger =
        work_executed (runtime_ledger (fst (predecessor action))) +
          S (length (executed_events action)).
  Proof.
    intros action event reserved fresh reason handling prefix continuation
      ledger Hfailure.
    pose proof (failed_action_reports_all_charges_and_work Hfailure)
      as [_ [Hwork _]].
    exact Hwork.
  Qed.
End FailureAction.

Module FailureControls.
  Import PublicationControl.

  Definition before_charge := prepare awaiting.
  Definition after_charge := charge before_charge 1.

  Definition after_execution :=
    execute after_charge 41 (ltac:(simpl; lia)) (ltac:(simpl; tauto)).

  Example allocation_failure_before_charge_keeps_zero_new_usage :
    exists prefix ledger,
      failure before_charge AllocationRejected EndContinuation =
        Incomplete AllocationRejected prefix None ledger /\
      semantic_contents prefix = semantic_contents awaiting /\
      ledger = runtime_ledger (fst awaiting).
  Proof.
    eexists; eexists; repeat split; reflexivity.
  Qed.

  Example allocation_failure_after_execution_reports_prior_work :
    exists prefix resumed ledger,
      failure after_execution AllocationRejected RetainContinuation =
        Incomplete AllocationRejected prefix (Some resumed) ledger /\
      semantic_contents prefix = semantic_contents awaiting /\
      semantic_contents resumed = semantic_contents awaiting /\
      work_executed ledger = 2 /\ work_charged ledger = 2.
  Proof.
    eexists; eexists; eexists; repeat split; reflexivity.
  Qed.

  Example charged_allocation_failure_cannot_report_complete :
    forall final,
      failure after_charge AllocationRejected RetainContinuation <>
        Complete final.
  Proof. intros; apply failed_action_cannot_claim_complete. Qed.

  Example charged_allocation_failure_cannot_undo_charge :
    forall prefix continuation,
      failure after_charge AllocationRejected RetainContinuation <>
        Incomplete AllocationRejected prefix continuation
          (runtime_ledger (fst awaiting)).
  Proof.
    intros prefix continuation Heq.
    pose proof (charged_failure_cannot_roll_back_reported_usage
      (action := before_charge) (amount := 1) (reason := AllocationRejected)
      (handling := RetainContinuation) (prefix := prefix)
      (continuation := continuation)
      (ledger := runtime_ledger (fst awaiting))) as Hpositive.
    specialize (Hpositive ltac:(lia) Heq).
    unfold before_charge in Hpositive; simpl in Hpositive; lia.
  Qed.

  Example allocation_failure_cannot_publish_partial_success :
    forall continuation ledger,
      failure before_charge AllocationRejected EndContinuation <>
        Incomplete AllocationRejected completed continuation ledger.
  Proof.
    intros continuation ledger Hfailure.
    pose proof (failed_action_keeps_predecessor_contents Hfailure)
      as [Hsemantics _].
    unfold semantic_contents, with_ledger_status,
      replace_ledger_status in Hsemantics.
    simpl in Hsemantics.
    discriminate Hsemantics.
  Qed.
End FailureControls.
