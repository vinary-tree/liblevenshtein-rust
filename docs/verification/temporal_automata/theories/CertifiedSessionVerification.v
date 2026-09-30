(** * Conditional verification transitions for a complete search session

    Supplied interpretations of the captured snapshot give each original
    payload and tie key. An abstract verifier result has one of five disjoint
    tags; [run_verifier] consumes that result rather than executing scoring.
    Its soundness
    relation binds a successful score and witness to that payload and query;
    rejection tags justify exclusions; numeric and resource errors make no
    exact-score claim. This file does not assert that a Rust verifier meets
    that relation or that a particular binary64 operation graph is exact.
    Failed 1 and Failed 2 are distinct abstract status markers, not source
    error discriminants. Failure continuation and publication are separate
    obligations. The cutoff relation is a parameter; the finite control
    below supplies one inclusive instance. *)

From Stdlib Require Import Arith Lia List Permutation.
From Liblevenshtein.TemporalAutomata Require Import CertifiedSearchSession.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Inductive verification_exclusion :=
| BeyondCutoffEvidence
| NoAlignmentEvidence.

Inductive verifier_result (Score Tie Witness Error : Type) :=
| VerifiedWithin : Score -> Tie -> Witness -> verifier_result Score Tie Witness Error
| VerifiedBeyond : verifier_result Score Tie Witness Error
| VerifiedNoAlignment : verifier_result Score Tie Witness Error
| NumericFailure : Error -> verifier_result Score Tie Witness Error
| ResourceFailure : Error -> verifier_result Score Tie Witness Error.

Arguments VerifiedWithin {Score Tie Witness Error} _ _ _.
Arguments VerifiedBeyond {Score Tie Witness Error}.
Arguments VerifiedNoAlignment {Score Tie Witness Error}.
Arguments NumericFailure {Score Tie Witness Error} _.
Arguments ResourceFailure {Score Tie Witness Error} _.

Record verification_claim (Score Tie Witness : Type) := {
  claim_original : nat;
  claim_score : Score;
  claim_tie : Tie;
  claim_witness : Witness
}.

Arguments claim_original {Score Tie Witness} _.
Arguments claim_score {Score Tie Witness} _.
Arguments claim_tie {Score Tie Witness} _.
Arguments claim_witness {Score Tie Witness} _.

Definition claim_rank {Score Tie Witness}
    (claim : verification_claim Score Tie Witness) : ranked_result Score :=
  {| result_original := claim_original claim;
     result_score := claim_score claim |}.

Section Verification.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Payload Tie Witness Error : Type}.

  Let Runtime := session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost Score verification_exclusion.
  Let Claim := verification_claim Score Tie Witness.
  Let Decision := verifier_result Score Tie Witness Error.

  Variable payload_of : Snapshot -> nat -> Payload.
  Variable exact_score : Query -> Payload -> option Score.
  Variable within_cutoff : Score -> Score -> Prop.
  Variable tie_key : Snapshot -> nat -> Tie.
  Variable witness_valid : Query -> Payload -> Score -> Witness -> Prop.
  Variable cutoff : Score.

  Definition exact_for (identity : session_identity Contract Snapshot Query)
      (original : nat) : option Score :=
    exact_score (session_query identity)
      (payload_of (session_snapshot identity) original).

  Definition authoritative_for
      (identity : session_identity Contract Snapshot Query)
      (original : nat) (score : Score) : Prop :=
    exact_for identity original = Some score.

  Definition evidence_for
      (identity : session_identity Contract Snapshot Query)
      (original : nat) (evidence : verification_exclusion) : Prop :=
    match evidence with
    | BeyondCutoffEvidence =>
        exists score, exact_for identity original = Some score /\
          ~ within_cutoff score cutoff
    | NoAlignmentEvidence => exact_for identity original = None
    end.

  Definition claim_valid
      (identity : session_identity Contract Snapshot Query)
      (claim : Claim) : Prop :=
    authoritative_for identity (claim_original claim) (claim_score claim) /\
    claim_tie claim = tie_key (session_snapshot identity)
      (claim_original claim) /\
    witness_valid (session_query identity)
      (payload_of (session_snapshot identity) (claim_original claim))
      (claim_score claim) (claim_witness claim).

  Definition decision_sound
      (identity : session_identity Contract Snapshot Query)
      (original : nat) (decision : Decision) : Prop :=
    match decision with
    | VerifiedWithin score tie witness =>
        authoritative_for identity original score /\
        within_cutoff score cutoff /\
        tie = tie_key (session_snapshot identity) original /\
        witness_valid (session_query identity)
          (payload_of (session_snapshot identity) original) score witness
    | VerifiedBeyond => evidence_for identity original BeyondCutoffEvidence
    | VerifiedNoAlignment =>
        evidence_for identity original NoAlignmentEvidence
    | NumericFailure _ | ResourceFailure _ => True
    end.

  Definition provenance (runtime : Runtime) (ghost : Ghost)
      (claims : list Claim) : Prop :=
    Forall (claim_valid (runtime_identity runtime)) claims /\
    map claim_rank claims = ghost_verified_history ghost.

  Definition with_phase_and_status (before : Runtime)
      (phase : private_phase Node Residual Path Score Key)
      (status : session_status) : Runtime :=
    {| runtime_identity := runtime_identity before;
       runtime_pending := runtime_pending before;
       runtime_private := phase;
       runtime_selected := runtime_selected before;
       runtime_emission := runtime_emission before;
       runtime_arena := runtime_arena before;
       runtime_cache := runtime_cache before;
       runtime_ledger := runtime_ledger before;
       runtime_reconstruction := runtime_reconstruction before;
       runtime_status := status |}.

  Definition with_verified (ghost : Ghost) (result : ranked_result Score)
      : Ghost :=
    {| ghost_verified_history := result :: ghost_verified_history ghost;
       ghost_emitted_results := ghost_emitted_results ghost;
       ghost_exclusions := ghost_exclusions ghost |}.

  Definition with_exclusion (ghost : Ghost) (original : nat)
      (evidence : verification_exclusion) : Ghost :=
    {| ghost_verified_history := ghost_verified_history ghost;
       ghost_emitted_results := ghost_emitted_results ghost;
       ghost_exclusions :=
         {| exclusion_original := original;
            exclusion_evidence := evidence |} :: ghost_exclusions ghost |}.

  Record verification_output := {
    next_runtime : Runtime;
    next_ghost : Ghost;
    next_claims : list Claim
  }.

  Definition run_verifier (before : Runtime) (ghost : Ghost)
      (claims : list Claim) (original : nat) (decision : Decision)
      : verification_output :=
    match decision with
    | VerifiedWithin score tie witness =>
        let claim := {| claim_original := original;
                        claim_score := score;
                        claim_tie := tie;
                        claim_witness := witness |} in
        {| next_runtime := with_phase_and_status before
             (AwaitingPublication (claim_rank claim)) Active;
           next_ghost := with_verified ghost (claim_rank claim);
           next_claims := claim :: claims |}
    | VerifiedBeyond =>
        {| next_runtime := with_phase_and_status before NoPrivate Active;
           next_ghost := with_exclusion ghost original BeyondCutoffEvidence;
           next_claims := claims |}
    | VerifiedNoAlignment =>
        {| next_runtime := with_phase_and_status before NoPrivate Active;
           next_ghost := with_exclusion ghost original NoAlignmentEvidence;
           next_claims := claims |}
    | NumericFailure _ =>
        {| next_runtime := with_phase_and_status before
             (ScoringCandidate original) (Failed 1);
           next_ghost := ghost;
           next_claims := claims |}
    | ResourceFailure _ =>
        {| next_runtime := with_phase_and_status before
             (ScoringCandidate original) (Failed 2);
           next_ghost := ghost;
           next_claims := claims |}
    end.

  Theorem only_success_prepends_a_rank :
    forall before ghost claims original decision,
      match decision with
      | VerifiedWithin score _ _ =>
          exists result,
            result_original result = original /\
            result_score result = score /\
            ghost_verified_history
              (next_ghost (run_verifier before ghost claims original decision)) =
              result :: ghost_verified_history ghost
      | _ =>
          ghost_verified_history
            (next_ghost (run_verifier before ghost claims original decision)) =
            ghost_verified_history ghost
      end.
  Proof.
    intros before ghost claims original decision.
    destruct decision; simpl; try reflexivity.
    eexists; repeat split; reflexivity.
  Qed.

  Theorem failed_verification_retains_private_ownership :
    forall (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      (before : Runtime) (ghost : Ghost) claims original error,
      runtime_private before = ScoringCandidate original ->
      ownership_projection region_of
        (next_runtime (run_verifier before ghost claims original
          (NumericFailure error)))
        (next_ghost (run_verifier before ghost claims original
          (NumericFailure error))) =
        ownership_projection region_of before ghost /\
      ownership_projection region_of
        (next_runtime (run_verifier before ghost claims original
          (ResourceFailure error)))
        (next_ghost (run_verifier before ghost claims original
          (ResourceFailure error))) =
        ownership_projection region_of before ghost.
  Proof.
    intros region_of before ghost claims original error Hprivate.
    unfold run_verifier, ownership_projection, with_phase_and_status; simpl.
    now rewrite Hprivate.
  Qed.

  Theorem verification_preserves_provenance :
    forall before ghost claims original decision,
      provenance before ghost claims ->
      decision_sound (runtime_identity before) original decision ->
      provenance
        (next_runtime (run_verifier before ghost claims original decision))
        (next_ghost (run_verifier before ghost claims original decision))
        (next_claims (run_verifier before ghost claims original decision)).
  Proof.
    intros before ghost claims original decision [Hclaims Hmap] Hsound.
    destruct decision; simpl; split; try assumption.
    - constructor; [|exact Hclaims].
      unfold claim_valid; simpl in *; tauto.
    - simpl; now rewrite Hmap.
  Qed.

  Lemma verifier_evidence_excludes_winner :
    forall identity original evidence,
      evidence_for identity original evidence ->
      ~ (exists score, authoritative_for identity original score /\
          within_cutoff score cutoff).
  Proof.
    intros identity original [|] Hevidence [score [Hexact Hwithin]];
      simpl in Hevidence; unfold authoritative_for in Hexact.
    - destruct Hevidence as [actual [Hactual Hnot]].
      rewrite Hexact in Hactual; inversion Hactual; subst.
      contradiction.
    - rewrite Hevidence in Hexact; discriminate.
  Qed.

  Theorem verifier_preserves_exact_ownership :
    forall (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      universe (before : Runtime) (ghost : Ghost) claims original decision,
      runtime_private before = ScoringCandidate original ->
      exact_ownership universe
        (ownership_projection region_of before ghost) ->
      exact_ownership universe
        (ownership_projection region_of
          (next_runtime (run_verifier before ghost claims original decision))
          (next_ghost (run_verifier before ghost claims original decision))).
  Proof.
    intros region_of universe before ghost claims original decision
      Hprivate Howned.
    destruct decision; simpl.
    - unfold ownership_projection in *; simpl in *.
      now rewrite Hprivate in Howned.
    - change (exact_ownership universe
        (publish_excluded (ownership_projection region_of before ghost)
          [] [] original)).
      eapply publish_excluded_preserves_exact_ownership; [|exact Howned].
      unfold ownership_projection; now rewrite Hprivate.
    - change (exact_ownership universe
        (publish_excluded (ownership_projection region_of before ghost)
          [] [] original)).
      eapply publish_excluded_preserves_exact_ownership; [|exact Howned].
      unfold ownership_projection; now rewrite Hprivate.
    - unfold ownership_projection in *; simpl in *.
      now rewrite Hprivate in Howned.
    - unfold ownership_projection in *; simpl in *.
      now rewrite Hprivate in Howned.
  Qed.

  Theorem private_verification_candidate_is_in_the_universe :
    forall (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      universe (before : Runtime) (ghost : Ghost) original,
      runtime_private before = ScoringCandidate original ->
      exact_ownership universe
        (ownership_projection region_of before ghost) ->
      In original universe.
  Proof.
    intros region_of universe before ghost original Hprivate [_ Hperm].
    eapply Permutation_in; [exact Hperm |].
    unfold owned_originals, ownership_projection; simpl.
    rewrite Hprivate; simpl.
    apply in_or_app; right.
    simpl; now left.
  Qed.

  Theorem verifier_preserves_sound_exclusion_evidence :
    forall (before : Runtime) (ghost : Ghost) claims original decision,
      Forall (fun entry => evidence_for (runtime_identity before)
        (exclusion_original entry) (exclusion_evidence entry))
        (ghost_exclusions ghost) ->
      decision_sound (runtime_identity before) original decision ->
      Forall (fun entry => evidence_for
        (runtime_identity
          (next_runtime (run_verifier before ghost claims original decision)))
        (exclusion_original entry) (exclusion_evidence entry))
        (ghost_exclusions
          (next_ghost (run_verifier before ghost claims original decision))).
  Proof.
    intros before ghost claims original decision Hprevious Hsound.
    destruct decision; simpl in Hsound |- *.
    - exact Hprevious.
    - constructor; [exact Hsound | exact Hprevious].
    - constructor; [exact Hsound | exact Hprevious].
    - exact Hprevious.
    - exact Hprevious.
  Qed.

  Theorem verification_transition_preserves_session_abstraction :
    forall (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      universe arena_valid cache_valid reconstruction_valid
      (before : Runtime) (ghost : Ghost) claims original decision,
      runtime_private before = ScoringCandidate original ->
      runtime_status before = Active ->
      session_abstraction region_of universe
        (authoritative_for (runtime_identity before))
        (evidence_for (runtime_identity before)) arena_valid cache_valid
        reconstruction_valid before ghost ->
      provenance before ghost claims ->
      decision_sound (runtime_identity before) original decision ->
      let after := run_verifier before ghost claims original decision in
      session_abstraction region_of universe
        (authoritative_for (runtime_identity before))
        (evidence_for (runtime_identity before)) arena_valid cache_valid
        reconstruction_valid (next_runtime after) (next_ghost after) /\
      provenance (next_runtime after) (next_ghost after)
        (next_claims after).
  Proof.
    intros region_of universe arena_valid cache_valid reconstruction_valid
      before ghost claims original decision Hprivate Hactive Hinv Hprovenance
      Hsound.
    split.
    - unfold session_abstraction in Hinv |- *.
      assert (Howned_after : exact_ownership universe
        (ownership_projection region_of
          (next_runtime (run_verifier before ghost claims original decision))
          (next_ghost (run_verifier before ghost claims original decision)))).
      { eapply verifier_preserves_exact_ownership; eauto.
        exact (proj1 Hinv). }
      destruct Hinv as
        [Howned [Hborrow [Hcapacity [Hcache_scope [Harena [Hcache_valid
        [Hselected [Hhistory [Hselected_history [Hexclusions [Hwork
        [Hbytes [Hemission Hcompleted]]]]]]]]]]]]].
      rewrite Hprivate in Harena; simpl in Harena.
      destruct decision; simpl in Hsound, Howned_after |- *;
        repeat (first [assumption | split]);
        try discriminate;
        try (eapply verifier_preserves_sound_exclusion_evidence; eauto).
      all: try (constructor; [exact (proj1 Hsound) | exact Hhistory]).
      all: try (eapply Forall_impl; [|exact Hselected_history];
        simpl; firstorder).
      all: try (constructor; [exact Hsound | exact Hexclusions]).
    - eapply verification_preserves_provenance; eauto.
  Qed.

  Theorem no_finite_score_cannot_be_admitted_as_verified :
    forall identity original score tie witness,
      exact_for identity original = None ->
      ~ decision_sound identity original
        (VerifiedWithin score tie witness).
  Proof.
    intros identity original score tie witness Hnone Hsound.
    destruct Hsound as [Hexact _].
    unfold authoritative_for in Hexact.
    now rewrite Hnone in Hexact.
  Qed.

  Theorem a_within_cutoff_score_cannot_be_rejected_as_beyond :
    forall identity original score,
      authoritative_for identity original score ->
      within_cutoff score cutoff ->
      ~ decision_sound identity original VerifiedBeyond.
  Proof.
    intros identity original score Hexact Hwithin Hsound.
    eapply verifier_evidence_excludes_winner; [exact Hsound |].
    now exists score.
  Qed.

  Theorem rejected_or_failed_verification_creates_no_rank :
    forall (before : Runtime) (ghost : Ghost) claims original error,
      ghost_verified_history
        (next_ghost (run_verifier before ghost claims original
          VerifiedBeyond)) = ghost_verified_history ghost /\
      ghost_verified_history
        (next_ghost (run_verifier before ghost claims original
          VerifiedNoAlignment)) = ghost_verified_history ghost /\
      ghost_verified_history
        (next_ghost (run_verifier before ghost claims original
          (NumericFailure error))) = ghost_verified_history ghost /\
      ghost_verified_history
        (next_ghost (run_verifier before ghost claims original
          (ResourceFailure error))) = ghost_verified_history ghost /\
      runtime_status
        (next_runtime (run_verifier before ghost claims original
          (NumericFailure error))) = Failed 1 /\
      runtime_status
        (next_runtime (run_verifier before ghost claims original
          (ResourceFailure error))) = Failed 2.
  Proof. intros; repeat split; reflexivity. Qed.

End Verification.

(** A finite natural cost or a symbolic infinity. The reference evaluator in
    this control only returns finite costs. The inclusive cutoff accepts
    finite costs at equality and never admits symbolic infinity as a score. *)
Inductive control_score := FiniteCost : nat -> control_score | InfinityCost.

Definition control_within (score cutoff : control_score) : Prop :=
  match score, cutoff with
  | FiniteCost value, FiniteCost limit => value <= limit
  | FiniteCost _, InfinityCost => True
  | InfinityCost, _ => False
  end.

Lemma control_cutoff_is_inclusive_for_finite_scores :
  forall value limit,
    control_within (FiniteCost value) (FiniteCost limit) <-> value <= limit.
Proof. reflexivity. Qed.

Definition control_identity : session_identity unit unit unit :=
  {| session_contract := tt;
     session_snapshot := tt;
     session_query := tt;
     session_revision := 0 |}.

Definition control_payload (_ : unit) (original : nat) : nat := original.

Definition control_exact (_ : unit) (payload : nat) : option control_score :=
  match payload with
  | 0 => Some (FiniteCost 7)
  | _ => Some (FiniteCost 4)
  end.

Definition control_tie (_ : unit) (original : nat) : nat := original.

Definition control_witness (_ : unit) (payload : nat)
    (_ : control_score) (witness : nat) : Prop := witness = payload.

Definition control_decision_sound (original : nat)
    (decision : verifier_result control_score nat nat nat) : Prop :=
  @decision_sound control_score unit unit unit nat nat nat nat
    control_payload control_exact control_within control_tie control_witness
    (FiniteCost 5) control_identity original decision.

Example finite_control_accepts_exact_success :
  control_decision_sound 1 (VerifiedWithin (FiniteCost 4) 1 1).
Proof. simpl; repeat split; try reflexivity; lia. Qed.

Example finite_control_accepts_genuine_above_cutoff :
  control_decision_sound 0 VerifiedBeyond.
Proof.
  simpl. exists (FiniteCost 7).
  split; [reflexivity |].
  simpl; lia.
Qed.

Example finite_control_accepts_resource_failure_as_failure :
  control_decision_sound 1 (ResourceFailure 9).
Proof. exact I. Qed.

Lemma control_exact_never_returns_infinity :
  forall original, control_exact tt original <> Some InfinityCost.
Proof. intros [|original]; discriminate. Qed.

Theorem control_rejects_infinite_rank_coercion :
  forall original tie witness,
    ~ control_decision_sound original
      (VerifiedWithin InfinityCost tie witness).
Proof.
  intros original tie witness Hsound.
  unfold control_decision_sound, decision_sound,
    authoritative_for, exact_for in Hsound.
  destruct Hsound as [Hexact _].
  simpl in Hexact.
  now apply (control_exact_never_returns_infinity original).
Qed.

Example beyond_cutoff_cannot_be_coerced_to_exact_infinity :
  control_decision_sound 0 VerifiedBeyond /\
  ~ control_decision_sound 0 (VerifiedWithin InfinityCost 0 0).
Proof.
  split.
  - apply finite_control_accepts_genuine_above_cutoff.
  - apply control_rejects_infinite_rank_coercion.
Qed.

Example allocation_failure_cannot_be_coerced_to_exact_infinity :
  control_decision_sound 1 (ResourceFailure 9) /\
  ~ control_decision_sound 1 (VerifiedWithin InfinityCost 1 1).
Proof.
  split.
  - apply finite_control_accepts_resource_failure_as_failure.
  - apply control_rejects_infinite_rank_coercion.
Qed.
