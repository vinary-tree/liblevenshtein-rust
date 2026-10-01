(** * Checked exact-expression certificates as finite score traces

    For a fixed exact-natural, score-only request, each accepted instruction
    evaluates a target expression whose value equals its source expression in
    every environment.  The concrete machine emits target values; the
    reference emits source values.  Administrative steps emit no event and
    may stutter.  This bridge covers the expression fragment, not Rust code,
    witnesses, resources, or total termination. *)

From Stdlib Require Import Arith List.
Require Import CertifiedContracts CertificateChecking CertifiedMetricExecution.
Import ListNotations.

Record checked_instruction := {
  instruction_source : cost_expression;
  instruction_target : cost_expression;
  instruction_certificate : scoped_certificate
}.

Record executable_state := {
  executable_cursor : nat;
  executable_history : list nat;
  administrative_count : nat
}.

Record reference_state := {
  reference_cursor : nat;
  reference_history : list nat
}.

Definition emit_executable (state : executable_state)
    (score : nat) : executable_state :=
  {| executable_cursor := S (executable_cursor state);
     executable_history := executable_history state ++ [score];
     administrative_count := administrative_count state |}.

Definition bump_administrative (state : executable_state)
    : executable_state :=
  {| executable_cursor := executable_cursor state;
     executable_history := executable_history state;
     administrative_count := S (administrative_count state) |}.

Definition emit_reference (state : reference_state)
    (score : nat) : reference_state :=
  {| reference_cursor := S (reference_cursor state);
     reference_history := reference_history state ++ [score] |}.

Definition histories_related (concrete : executable_state)
    (abstract : reference_state) : Prop :=
  executable_cursor concrete = reference_cursor abstract /\
  executable_history concrete = reference_history abstract.

Section CheckedProgram.
  Variable requested : realization_scope.
  Variable environment : nat -> nat.
  Variable program : list checked_instruction.

  Inductive executable_step :
      executable_state -> list nat -> executable_state -> Prop :=
  | EmitChecked : forall state instruction,
      nth_error program (executable_cursor state) = Some instruction ->
      accept_scoped requested
        (instruction_source instruction)
        (instruction_target instruction)
        Equivalent (instruction_certificate instruction) = true ->
      executable_step state
        [evaluate environment (instruction_target instruction)]
        (emit_executable state
          (evaluate environment (instruction_target instruction)))
  | Administrative : forall state,
      executable_step state [] (bump_administrative state).

  Inductive reference_step :
      reference_state -> list nat -> reference_state -> Prop :=
  | EmitSource : forall state instruction,
      nth_error program (reference_cursor state) = Some instruction ->
      reference_step state
        [evaluate environment (instruction_source instruction)]
        (emit_reference state
          (evaluate environment (instruction_source instruction))).

  Theorem accepted_instruction_steps_refine_reference :
    step_refinement executable_step reference_step histories_related.
  Proof.
    intros concrete events next abstract Hrelated Hstep.
    destruct Hrelated as [Hcursor Hhistory].
    inversion Hstep as
      [state instruction Hlookup Haccepted | state]; subst.
    - pose proof (accepted_scoped_certificate_sound
        requested (instruction_source instruction)
        (instruction_target instruction) Equivalent
        (instruction_certificate instruction) Haccepted)
        as [_ Hscore].
      specialize (Hscore environment).
      unfold relation_holds in Hscore; simpl in Hscore.
      assert (Hreference_lookup :
        nth_error program (reference_cursor abstract) = Some instruction).
      { rewrite <- Hcursor. exact Hlookup. }
      exists (emit_reference abstract
        (evaluate environment (instruction_source instruction))).
      split.
      + rewrite Hscore.
        eapply execution_cons with
          (events := [evaluate environment
            (instruction_source instruction)])
          (later := []).
        * apply EmitSource. exact Hreference_lookup.
        * constructor.
      + split; simpl; [now rewrite Hcursor |].
        now rewrite Hhistory, Hscore.
    - exists abstract; split; [constructor |].
      split; simpl; assumption.
  Qed.

  Definition initial_executable : executable_state :=
    {| executable_cursor := 0; executable_history := [];
       administrative_count := 0 |}.

  Definition initial_reference : reference_state :=
    {| reference_cursor := 0; reference_history := [] |}.

  Definition executable_complete (state : executable_state) : Prop :=
    executable_cursor state = length program.

  Definition reference_complete (state : reference_state) : Prop :=
    reference_cursor state = length program.

  Theorem accepted_program_trace_preserves_completed_scores :
    forall events final_concrete,
      execution executable_step initial_executable events final_concrete ->
      executable_complete final_concrete ->
      exists final_reference,
        execution reference_step initial_reference events final_reference /\
        reference_complete final_reference /\
        executable_history final_concrete =
          reference_history final_reference.
  Proof.
    intros events final_concrete Hrun Hcomplete.
    assert (Hinitial :
      histories_related initial_executable initial_reference).
    { split; reflexivity. }
    destruct (local_certificates_lift_to_traces
      _ _ executable_step reference_step histories_related
      accepted_instruction_steps_refine_reference
      initial_executable events final_concrete Hrun
      initial_reference Hinitial)
      as [final_reference [Hreference Hrelated]].
    destruct Hrelated as [Hcursor Hhistory].
    exists final_reference; repeat split; try assumption.
    unfold executable_complete in Hcomplete.
    unfold reference_complete. now rewrite <- Hcursor.
  Qed.

  CoInductive infinite_administration : executable_state -> Prop :=
  | MoreAdministration : forall state,
      executable_step state [] (bump_administrative state) ->
      infinite_administration (bump_administrative state) ->
      infinite_administration state.

  CoFixpoint administration_forever (state : executable_state)
      : infinite_administration state :=
    MoreAdministration state (Administrative state)
      (administration_forever (bump_administrative state)).

  Lemma nonempty_program_starts_incomplete :
    program <> [] -> ~ executable_complete initial_executable.
  Proof.
    intros Hnonempty Hcomplete.
    unfold executable_complete, initial_executable in Hcomplete; simpl in Hcomplete.
    destruct program as [|instruction remaining];
      [contradiction | discriminate Hcomplete].
  Qed.
End CheckedProgram.

Module TraceControls.
  Definition instruction : checked_instruction :=
    {| instruction_source := Addition (InputVar 0) (Constant 0);
       instruction_target := InputVar 0;
       instruction_certificate :=
         singleton_certificate sample_forward_step |}.

  Definition program : list checked_instruction := [instruction].
  Definition environment (_ : nat) : nat := 7.

  Example checked_instruction_is_accepted :
    accept_scoped sample_realization_scope
      (instruction_source instruction)
      (instruction_target instruction) Equivalent
      (instruction_certificate instruction) = true.
  Proof. exact valid_exact_replay_is_accepted. Qed.

  Example checked_program_has_completed_trace :
    execution
      (executable_step sample_realization_scope environment program)
      initial_executable [7] (emit_executable initial_executable 7) /\
    executable_complete program (emit_executable initial_executable 7).
  Proof.
    split.
    - change (execution
        (executable_step sample_realization_scope environment program)
        initial_executable
        ([evaluate environment (instruction_target instruction)] ++ [])
        (emit_executable initial_executable
          (evaluate environment (instruction_target instruction)))).
      eapply execution_cons.
      + apply EmitChecked with (instruction := instruction).
        * reflexivity.
        * exact checked_instruction_is_accepted.
      + constructor.
    - reflexivity.
  Qed.

  Example checked_program_has_matching_reference :
    exists final_reference,
      execution
        (reference_step environment program)
        initial_reference [7] final_reference /\
      reference_complete program final_reference /\
      executable_history (emit_executable initial_executable 7) =
        reference_history final_reference.
  Proof.
    eapply accepted_program_trace_preserves_completed_scores.
    - exact (proj1 checked_program_has_completed_trace).
    - exact (proj2 checked_program_has_completed_trace).
  Qed.

  Example checked_program_also_has_infinite_silent_trace :
    infinite_administration sample_realization_scope environment program
      initial_executable /\
    ~ executable_complete program initial_executable.
  Proof.
    split.
    - apply administration_forever.
    - apply nonempty_program_starts_incomplete. discriminate.
  Qed.
End TraceControls.
