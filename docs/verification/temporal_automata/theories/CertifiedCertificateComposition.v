(** * Accepted local certificates compose across a finite ordered trace

    The concrete and reference states are the exact-natural, score-only
    expression machines in CertifiedCertificateTrace. A checked emission
    advances one program cursor and appends one score. An administrative
    step emits no event and leaves the score history untouched. The theorem
    below relates explicit initial states, the ordered event list, and
    completed terminal states. Its two-instruction instance emits distinct
    scores and includes silent steps. A separate coinductive control shows
    that finite-trace preservation does not supply progress or finalization.

    This is not a checker for the whole ordered-residual calculus, a witness
    or resource observation, a Rust correspondence, or a termination proof. *)

From Stdlib Require Import Arith List.
Require Import CertifiedContracts CertificateChecking
  CertifiedMetricExecution CertifiedCertificateTrace.
Import ListNotations.

Definition local_exact_certificate_accepted (requested : realization_scope)
    (instruction : checked_instruction) : Prop :=
  accept_scoped requested (instruction_source instruction)
    (instruction_target instruction) Equivalent
    (instruction_certificate instruction) = true.

Definition executable_ordered_scores (environment : nat -> nat)
    (program : list checked_instruction) : list nat :=
  map (fun instruction => evaluate environment
    (instruction_target instruction)) program.

Definition reference_ordered_scores (environment : nat -> nat)
    (program : list checked_instruction) : list nat :=
  map (fun instruction => evaluate environment
    (instruction_source instruction)) program.

Theorem accepted_local_certificates_preserve_program_score_order :
  forall requested environment program,
    Forall (local_exact_certificate_accepted requested) program ->
    executable_ordered_scores environment program =
      reference_ordered_scores environment program.
Proof.
  intros requested environment program Haccepted.
  induction Haccepted as [|instruction remaining Hhead Htail IH];
    simpl; [reflexivity |].
  f_equal.
  - apply (accepted_exact_chain_preserves_every_score requested
      (instruction_source instruction) (instruction_target instruction)
      (instruction_certificate instruction) Hhead environment).
  - exact IH.
Qed.

Lemma finite_executable_trace_appends_events :
  forall requested environment program first events last,
    execution (executable_step requested environment program)
      first events last ->
    executable_history last = executable_history first ++ events.
Proof.
  intros requested environment program first events last Htrace.
  induction Htrace as
    [state | first emitted middle later last Hstep Htail IH].
  - simpl. now rewrite app_nil_r.
  - rewrite IH.
    inversion Hstep; subst; simpl.
    + now rewrite <- app_assoc.
    + reflexivity.
Qed.

Theorem accepted_local_certificates_compose_completed_trace :
  forall requested environment program events final_concrete,
    Forall (local_exact_certificate_accepted requested) program ->
    execution (executable_step requested environment program)
      (initial_executable) events final_concrete ->
    executable_complete program final_concrete ->
    histories_related initial_executable initial_reference /\
    exists final_reference,
      execution (reference_step environment program)
        initial_reference events final_reference /\
      reference_complete program final_reference /\
      histories_related final_concrete final_reference /\
      events = reference_history final_reference /\
      executable_ordered_scores environment program =
        reference_ordered_scores environment program.
Proof.
  intros requested environment program events final_concrete
    Haccepted Htrace Hcomplete.
  assert (Hinitial : histories_related initial_executable initial_reference).
  { split; reflexivity. }
  split; [exact Hinitial |].
  pose proof (finite_executable_trace_appends_events
    requested environment program initial_executable events final_concrete
    Htrace) as Hordered.
  simpl in Hordered.
  destruct (accepted_program_trace_preserves_completed_scores
    requested environment program events final_concrete Htrace Hcomplete)
    as [final_reference [Hreference [Hreference_complete Hhistory]]].
  exists final_reference.
  split; [exact Hreference |].
  split; [exact Hreference_complete |].
  split.
  - split; [|exact Hhistory].
    unfold executable_complete in Hcomplete.
    unfold reference_complete in Hreference_complete.
    now rewrite Hcomplete, Hreference_complete.
  - split.
    + rewrite <- Hhistory. symmetry. exact Hordered.
    + exact (accepted_local_certificates_preserve_program_score_order
        requested environment program Haccepted).
Qed.

Module TwoInstructionControl.
  Definition requested : realization_scope := sample_realization_scope.

  Definition environment (index : nat) : nat :=
    if Nat.eqb index 0 then 7 else 3.

  Definition first_instruction : checked_instruction :=
    TraceControls.instruction.

  Definition second_step : scoped_step :=
    {| recorded_rule := NamedExact AddZeroRight;
       recorded_premises :=
         [ExactNaturalArithmetic; ScoreOnlyObservation; ExactPattern];
       recorded_direction := Forward;
       recorded_source := Addition (InputVar 1) (Constant 0);
       recorded_target := InputVar 1;
       recorded_scope := requested;
       recorded_arithmetic := ExactNaturals;
       recorded_witness_effect := NoWitnessEffect |}.

  Definition second_instruction : checked_instruction :=
    {| instruction_source := Addition (InputVar 1) (Constant 0);
       instruction_target := InputVar 1;
       instruction_certificate := singleton_certificate second_step |}.

  Definition program : list checked_instruction :=
    [first_instruction; second_instruction].

  Definition after_first : executable_state :=
    emit_executable initial_executable 7.

  Definition after_private_step : executable_state :=
    bump_administrative after_first.

  Definition after_second : executable_state :=
    emit_executable after_private_step 3.

  Definition completed : executable_state :=
    bump_administrative after_second.

  Example both_local_certificates_are_accepted :
    Forall (local_exact_certificate_accepted requested) program.
  Proof.
    constructor.
    - exact TraceControls.checked_instruction_is_accepted.
    - constructor; [reflexivity | constructor].
  Qed.

  Example ordered_scores_are_distinct_and_equal :
    executable_ordered_scores environment program = [7; 3] /\
    reference_ordered_scores environment program = [7; 3].
  Proof. split; reflexivity. Qed.

  Example two_checked_emissions_with_silent_steps :
    execution (executable_step requested environment program)
      initial_executable [7; 3] completed.
  Proof.
    eapply execution_cons with
      (events := [7]) (later := [3]) (middle := after_first).
    - apply EmitChecked with (instruction := first_instruction);
        reflexivity.
    - eapply execution_cons with
        (events := []) (later := [3]) (middle := after_private_step).
      + apply Administrative.
      + eapply execution_cons with
          (events := [3]) (later := []) (middle := after_second).
        * apply EmitChecked with (instruction := second_instruction);
            reflexivity.
        * eapply execution_cons with
            (events := []) (later := []) (middle := completed).
          -- apply Administrative.
          -- constructor.
  Qed.

  Example explicit_completed_observation :
    executable_complete program completed /\
    executable_history completed = [7; 3] /\
    administrative_count completed = 2.
  Proof. repeat split; reflexivity. Qed.

  Example two_checked_instructions_compose_to_reference :
    exists final_reference,
      execution (reference_step environment program)
        initial_reference [7; 3] final_reference /\
      reference_complete program final_reference /\
      histories_related completed final_reference /\
      reference_history final_reference = [7; 3].
  Proof.
    destruct (accepted_local_certificates_compose_completed_trace
      requested environment program [7; 3] completed
      both_local_certificates_are_accepted
      two_checked_emissions_with_silent_steps
      (proj1 explicit_completed_observation))
      as [_ [final_reference [Hrun [Hcomplete [Hrelated [Hordered _]]]]]].
    exists final_reference.
    split; [exact Hrun |].
    split; [exact Hcomplete |].
    split; [exact Hrelated |].
    symmetry; exact Hordered.
  Qed.

  Example accepted_prefix_can_stutter_forever_without_finalization :
    infinite_administration requested environment program after_first /\
    ~ executable_complete program after_first.
  Proof.
    split.
    - apply administration_forever.
    - unfold executable_complete, after_first, program.
      simpl; discriminate.
  Qed.
End TwoInstructionControl.
