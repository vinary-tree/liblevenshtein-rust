(** * Suspension and resumption of complete search sessions

    A page boundary changes only the status of a retained continuation.  The
    pending occurrence cursors, private phase, snapshot identity, selected
    results, cache, arena, reconstruction, emission cursor, and cumulative
    ledger remain in that continuation.  The logical execution theorem below
    is conditional on a deterministic productive relation; connecting that
    relation to each Rust continuation is a separate correspondence task. *)

From Stdlib Require Import Arith List.
From Liblevenshtein.TemporalAutomata Require Import CertifiedSearchSession.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Section Resume.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Let Runtime := session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost Score Evidence.
  Let Configuration := (Runtime * Ghost)%type.

  Definition replace_status (runtime : Runtime) (status : session_status)
      : Runtime :=
    {| runtime_identity := runtime_identity runtime;
       runtime_pending := runtime_pending runtime;
       runtime_private := runtime_private runtime;
       runtime_selected := runtime_selected runtime;
       runtime_emission := runtime_emission runtime;
       runtime_arena := runtime_arena runtime;
       runtime_cache := runtime_cache runtime;
       runtime_ledger := runtime_ledger runtime;
       runtime_reconstruction := runtime_reconstruction runtime;
       runtime_status := status |}.

  Definition replace_configuration_status (configuration : Configuration)
      (status : session_status) : Configuration :=
    (replace_status (fst configuration) status, snd configuration).

  Definition logical_view (configuration : Configuration) : Configuration :=
    replace_configuration_status configuration Active.

  Definition suspend (configuration : Configuration) (reason : nat)
      : Configuration :=
    replace_configuration_status configuration (Suspended reason).

  Definition resume (configuration : Configuration) : Configuration :=
    replace_configuration_status configuration Active.

  Lemma replace_status_twice : forall runtime first second,
    replace_status (replace_status runtime first) second =
      replace_status runtime second.
  Proof. intros [??????????] first second; reflexivity. Qed.

  Lemma replace_status_same : forall runtime,
    replace_status runtime (runtime_status runtime) = runtime.
  Proof. intros [??????????]; reflexivity. Qed.

  Lemma logical_view_replace : forall configuration status,
    logical_view (replace_configuration_status configuration status) =
      logical_view configuration.
  Proof.
    intros [runtime ghost] status.
    unfold logical_view, replace_configuration_status; simpl.
    now rewrite replace_status_twice.
  Qed.

  Lemma resume_after_suspend : forall configuration reason,
    runtime_status (fst configuration) = Active ->
    resume (suspend configuration reason) = configuration.
  Proof.
    intros [runtime ghost] reason Hactive.
    unfold resume, suspend, replace_configuration_status; simpl.
    rewrite replace_status_twice.
    rewrite <- Hactive.
    now rewrite replace_status_same.
  Qed.

  Lemma suspension_preserves_logical_view : forall configuration reason,
    logical_view (suspend configuration reason) = logical_view configuration.
  Proof. intros; apply logical_view_replace. Qed.

  Lemma resumption_preserves_logical_view : forall configuration,
    logical_view (resume configuration) = logical_view configuration.
  Proof. intros; apply logical_view_replace. Qed.

  Lemma suspension_preserves_private_and_ledger :
    forall configuration reason,
      runtime_private (fst (suspend configuration reason)) =
        runtime_private (fst configuration) /\
      runtime_pending (fst (suspend configuration reason)) =
        runtime_pending (fst configuration) /\
      runtime_identity (fst (suspend configuration reason)) =
        runtime_identity (fst configuration) /\
      runtime_ledger (fst (suspend configuration reason)) =
        runtime_ledger (fst configuration) /\
      snd (suspend configuration reason) = snd configuration.
  Proof. intros [runtime ghost] reason; repeat split; reflexivity. Qed.

  Lemma resumption_preserves_private_and_ledger :
    forall configuration,
      runtime_private (fst (resume configuration)) =
        runtime_private (fst configuration) /\
      runtime_pending (fst (resume configuration)) =
        runtime_pending (fst configuration) /\
      runtime_identity (fst (resume configuration)) =
        runtime_identity (fst configuration) /\
      runtime_ledger (fst (resume configuration)) =
        runtime_ledger (fst configuration) /\
      snd (resume configuration) = snd configuration.
  Proof. intros [runtime ghost]; repeat split; reflexivity. Qed.

  Definition same_ownership (region_of :
      session_identity Contract Snapshot Query ->
      occurrence Node Residual Path -> list nat)
      (first second : Configuration) : Prop :=
    ownership_projection region_of (fst first) (snd first) =
      ownership_projection region_of (fst second) (snd second).

  Lemma replace_status_preserves_ownership : forall region_of configuration status,
    same_ownership region_of configuration
      (replace_configuration_status configuration status).
  Proof. intros region_of [runtime ghost] status; reflexivity. Qed.

  Definition nonterminal_invariant
      (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      (universe : list nat) (authoritative : nat -> Score -> Prop)
      (evidence_sound : nat -> Evidence -> Prop)
      (arena_valid : Arena -> Residual -> Prop)
      (cache_valid : session_identity Contract Snapshot Query ->
        Key -> Value -> Prop)
      (reconstruction_valid : Reconstruction -> Path -> nat -> Prop)
      (configuration : Configuration) : Prop :=
    session_abstraction region_of universe authoritative evidence_sound
      arena_valid cache_valid reconstruction_valid
      (fst configuration) (snd configuration).

  Lemma nonterminal_status_preserves_abstraction :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid configuration status,
      status <> Completed ->
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid configuration ->
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid
        (replace_configuration_status configuration status).
  Proof.
    intros region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid [runtime ghost] status Hstatus Hinv.
    unfold nonterminal_invariant, session_abstraction, exact_ownership,
      ownership_projection, borrowed_parent_live, selected_capacity,
      complete_cache_scoped in *.
    simpl in *.
    destruct runtime as [identity pending private selected emission arena
      cache ledger reconstruction old_status].
    simpl in *.
    repeat split; tauto.
  Qed.

  Corollary suspend_preserves_abstraction :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid configuration reason,
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid configuration ->
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid
        (suspend configuration reason).
  Proof.
    intros; eapply nonterminal_status_preserves_abstraction; eauto.
    discriminate.
  Qed.

  Corollary resume_preserves_abstraction :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid configuration,
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid configuration ->
      nonterminal_invariant region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid (resume configuration).
  Proof.
    intros; eapply nonterminal_status_preserves_abstraction; eauto.
    discriminate.
  Qed.

  Variable productive : Configuration -> Configuration -> Prop.

  Inductive paged_step : Configuration -> Configuration -> Prop :=
  | PagedWork : forall first last,
      runtime_status (fst first) = Active ->
      (runtime_status (fst last) = Active \/
       runtime_status (fst last) = Completed) ->
      productive (logical_view first) (logical_view last) ->
      paged_step first last
  | PagedSuspend : forall configuration reason,
      runtime_status (fst configuration) = Active ->
      paged_step configuration (suspend configuration reason)
  | PagedResume : forall configuration reason,
      runtime_status (fst configuration) = Suspended reason ->
      paged_step configuration (resume configuration).

  Inductive finite_steps {A : Type} (step : A -> A -> Prop) : A -> A -> Prop :=
  | StepsDone : forall state, finite_steps step state state
  | StepsMore : forall first middle last,
      step first middle ->
      finite_steps step middle last ->
      finite_steps step first last.

  Theorem paged_execution_erases_to_logical_execution :
    forall first last,
      finite_steps paged_step first last ->
      finite_steps productive (logical_view first) (logical_view last).
  Proof.
    intros first last Hsteps; induction Hsteps.
    - constructor.
    - destruct H.
      + eapply StepsMore; eauto.
      + rewrite suspension_preserves_logical_view in IHHsteps.
        exact IHHsteps.
      + rewrite resumption_preserves_logical_view in IHHsteps.
        exact IHHsteps.
  Qed.

  Definition logical_terminal (state : Configuration) : Prop :=
    forall next, ~ productive state next.

  Hypothesis productive_deterministic : forall first second third,
    productive first second -> productive first third -> second = third.

  Lemma deterministic_terminal_result_unique :
    forall start first last,
      finite_steps productive start first ->
      finite_steps productive start last ->
      logical_terminal first -> logical_terminal last ->
      first = last.
  Proof.
    intros start first last Hfirst.
    revert last.
    induction Hfirst as [state | start middle first Hstep Htail IH];
      intros last Hlast Hterminal_first Hterminal_last.
    - inversion Hlast; subst; [reflexivity |].
      exfalso; eapply Hterminal_first; eauto.
    - inversion Hlast; subst.
      + exfalso; eapply Hterminal_last; eauto.
      + assert (middle = middle0) by
          (eapply productive_deterministic; eauto).
        subst middle0.
        eapply IH; eauto.
  Qed.

  Definition completed_results (configuration : Configuration) :=
    (selected_entries (runtime_selected (fst configuration)),
     ghost_emitted_results (snd configuration)).

  Theorem completed_paged_and_unpaged_results_agree :
    forall initial paged_final unpaged_final,
      finite_steps paged_step initial paged_final ->
      finite_steps productive (logical_view initial)
        (logical_view unpaged_final) ->
      runtime_status (fst paged_final) = Completed ->
      runtime_status (fst unpaged_final) = Completed ->
      logical_terminal (logical_view paged_final) ->
      logical_terminal (logical_view unpaged_final) ->
      completed_results paged_final = completed_results unpaged_final.
  Proof.
    intros initial paged_final unpaged_final Hp Hu _ _ Hpt Hut.
    pose proof (paged_execution_erases_to_logical_execution Hp)
      as Hprojected.
    pose proof (deterministic_terminal_result_unique
      Hprojected Hu Hpt Hut) as Heq.
    unfold completed_results in *.
    destruct paged_final as [paged_runtime paged_ghost].
    destruct unpaged_final as [unpaged_runtime unpaged_ghost].
    simpl in *.
    inversion Heq; reflexivity.
  Qed.

  Theorem reset_pending_cannot_be_a_pause : forall configuration bad reason,
    runtime_pending (fst bad) <> runtime_pending (fst configuration) ->
    bad <> suspend configuration reason.
  Proof.
    intros configuration bad reason Hchanged Heq; subst bad.
    apply Hchanged; reflexivity.
  Qed.

  Theorem recharge_cannot_be_a_resume : forall configuration bad,
    runtime_ledger (fst bad) <> runtime_ledger (fst configuration) ->
    bad <> resume configuration.
  Proof.
    intros configuration bad Hchanged Heq; subst bad.
    apply Hchanged; reflexivity.
  Qed.
End Resume.
