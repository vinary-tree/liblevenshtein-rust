(** * Phase-indexed best-k certificates for complete search sessions

    The heap proof stores natural-cost ranks with explicit tie keys. The
    complete session stores original IDs and scores. The tie-key function is
    a parameter whose binding to the captured operation contract remains an
    instance obligation. These maps connect the two proof representations;
    they do not establish a source layout or a machine-score correspondence.

    A successful verification prepends ghost history before its private
    AwaitingPublication phase changes the selected heap. During that phase
    the old best-k certificate covers the history tail. Publication consumes
    one best_k_step and yields a certificate for the full history. This file
    proves the rank bridge only: ownership transfer, exclusion evidence,
    source heap mutators, and emitted-result finalization remain separate.
    Canonical selection is relative to verified history, not a claim that
    the unverified or excluded part of the universe cannot improve it. *)

From Stdlib Require Import Arith Lia List Permutation.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedKnnHeap CertifiedSearchSession.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Definition entry_of_result (tie_of : nat -> nat)
    (result : ranked_result nat) : ranked_entry :=
  {| entry_original := result_original result;
     entry_cost := result_score result;
     entry_tie := tie_of (result_original result) |}.

Definition result_of_entry (entry : ranked_entry) : ranked_result nat :=
  {| result_original := entry_original entry;
     result_score := entry_cost entry |}.

Definition canonical_tie (tie_of : nat -> nat)
    (entry : ranked_entry) : Prop :=
  entry_tie entry = tie_of (entry_original entry).

Lemma result_entry_round_trip : forall tie_of result,
  result_of_entry (entry_of_result tie_of result) = result.
Proof. intros tie_of [original score]; reflexivity. Qed.

Lemma entry_result_round_trip : forall tie_of entry,
  canonical_tie tie_of entry ->
  entry_of_result tie_of (result_of_entry entry) = entry.
Proof.
  intros tie_of [original cost tie] Hcanonical.
  unfold canonical_tie in Hcanonical; simpl in Hcanonical.
  subst tie; reflexivity.
Qed.

Lemma mapped_results_have_canonical_ties : forall tie_of results,
  Forall (canonical_tie tie_of)
    (map (entry_of_result tie_of) results).
Proof.
  intros tie_of results; induction results as [|result rest IH]; simpl.
  - constructor.
  - constructor; [reflexivity | exact IH].
Qed.

Lemma canonical_entries_round_trip : forall tie_of entries,
  Forall (canonical_tie tie_of) entries ->
  map (entry_of_result tie_of) (map result_of_entry entries) = entries.
Proof.
  intros tie_of entries Hcanonical.
  induction Hcanonical as [|entry rest Hentry Hrest IH].
  - reflexivity.
  - change (entry_of_result tie_of (result_of_entry entry) ::
      map (entry_of_result tie_of) (map result_of_entry rest) = entry :: rest).
    f_equal.
    + now apply entry_result_round_trip.
    + exact IH.
Qed.

Lemma mapped_result_entries_round_trip : forall tie_of results,
  map result_of_entry (map (entry_of_result tie_of) results) = results.
Proof.
  intros tie_of results; induction results as [|result rest IH].
  - reflexivity.
  - change (result_of_entry (entry_of_result tie_of result) ::
      map result_of_entry (map (entry_of_result tie_of) rest) = result :: rest).
    f_equal.
    + apply result_entry_round_trip.
    + exact IH.
Qed.

Definition fresh_result (tie_of : nat -> nat)
    (result : ranked_result nat) (history : list (ranked_result nat)) : Prop :=
  ~ In (result_original result) (map result_original history) /\
  ~ In (tie_of (result_original result))
    (map (fun previous => tie_of (result_original previous)) history).

(** Injectivity is sufficient, but the publication rule only needs freshness
    against this history. An instance may establish that local fact directly. *)
Lemma injective_ties_preserve_freshness : forall tie_of result history,
  (forall left right, tie_of left = tie_of right -> left = right) ->
  ~ In (result_original result) (map result_original history) ->
  fresh_result tie_of result history.
Proof.
  intros tie_of result history Hinjective Hfresh.
  split; [exact Hfresh |].
  intro Hin; apply in_map_iff in Hin as [previous [Hequal Hin]].
  apply Hfresh.
  assert (Horiginal : result_original previous = result_original result).
  { apply Hinjective; exact Hequal. }
  rewrite <- Horiginal; apply in_map; exact Hin.
Qed.

Definition session_heap_certificate (tie_of : nat -> nat)
    (capacity : nat) (history selected : list (ranked_result nat))
    (rejected : list ranked_entry) : Prop :=
  best_k_certificate capacity
    (map (entry_of_result tie_of) history)
    (map (entry_of_result tie_of) selected) rejected.

Theorem certified_session_selection_is_canonical :
  forall tie_of capacity history selected rejected,
    session_heap_certificate tie_of capacity history selected rejected ->
    map (entry_of_result tie_of) selected =
      firstn capacity (sort_ranked
        (map (entry_of_result tie_of) history)).
Proof.
  intros tie_of capacity history selected rejected Hcertificate.
  unfold session_heap_certificate in Hcertificate.
  now apply certified_best_k_is_canonical with (rejected := rejected).
Qed.

Corollary certified_session_result_list_is_canonical :
  forall tie_of capacity history selected rejected,
    session_heap_certificate tie_of capacity history selected rejected ->
    selected = map result_of_entry
      (firstn capacity (sort_ranked
        (map (entry_of_result tie_of) history))).
Proof.
  intros tie_of capacity history selected rejected Hcertificate.
  pose proof (certified_session_selection_is_canonical Hcertificate) as Hrank.
  apply (f_equal (map result_of_entry)) in Hrank.
  now rewrite mapped_result_entries_round_trip in Hrank.
Qed.

Lemma certified_selected_entries_have_canonical_ties :
  forall tie_of capacity verified selected rejected,
    best_k_certificate capacity verified selected rejected ->
    Forall (canonical_tie tie_of) verified ->
    Forall (canonical_tie tie_of) selected.
Proof.
  intros tie_of capacity verified selected rejected Hcertificate Hcanonical.
  apply Forall_forall; intros entry Hin.
  apply Forall_forall with (x := entry) in Hcanonical; [exact Hcanonical |].
  eapply Permutation_in.
  - apply Permutation_sym; exact (best_cover Hcertificate).
  - apply in_or_app; now left.
Qed.

(** This is the publication boundary after a verified rank has been staged.
    The rank is already in ghost history, but has not yet entered the heap. *)
Theorem publication_step_preserves_session_heap_certificate :
  forall tie_of capacity history selected rejected result
    next_selected next_rejected,
    session_heap_certificate tie_of capacity history selected rejected ->
    ~ In (result_original result) (map result_original history) ->
    ~ In (tie_of (result_original result))
      (map (fun previous => tie_of (result_original previous)) history) ->
    best_k_step capacity (entry_of_result tie_of result)
      (map (entry_of_result tie_of) selected) rejected
      next_selected next_rejected ->
    session_heap_certificate tie_of capacity (result :: history)
      (map result_of_entry next_selected) next_rejected.
Proof.
  intros tie_of capacity history selected rejected result next_selected
    next_rejected Hcertificate Hfresh_original Hfresh_tie Hstep.
  unfold session_heap_certificate in *.
  assert (Hnext : best_k_certificate capacity
    (entry_of_result tie_of result :: map (entry_of_result tie_of) history)
    next_selected next_rejected).
  { eapply best_k_step_preserves_certificate.
    - exact Hcertificate.
    - rewrite map_map; exact Hfresh_original.
    - rewrite map_map; exact Hfresh_tie.
    - exact Hstep. }
  assert (Hcanonical : Forall (canonical_tie tie_of) next_selected).
  { eapply certified_selected_entries_have_canonical_ties;
      [exact Hnext |].
    constructor; [reflexivity | apply mapped_results_have_canonical_ties]. }
  simpl.
  now rewrite (canonical_entries_round_trip Hcanonical).
Qed.

Section RuntimePhase.
  Context {Node Residual Path Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Let Runtime := session_runtime Node Residual Path nat Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost nat Evidence.

  (** [rejected] is proof-only rank history, not a second production heap. *)
  Definition phase_heap_certificate (tie_of : nat -> nat)
      (runtime : Runtime) (ghost : Ghost)
      (rejected : list ranked_entry) : Prop :=
    match runtime_selected runtime with
    | RangeSelected _ => False
    | KnnSelected capacity selected =>
        match runtime_private runtime with
        | AwaitingPublication pending =>
            exists history,
              ghost_verified_history ghost = pending :: history /\
              session_heap_certificate tie_of capacity history selected
                rejected /\ fresh_result tie_of pending history
        | _ => session_heap_certificate tie_of capacity
            (ghost_verified_history ghost) selected rejected
        end
    end.

  Theorem phase_heap_selection_is_canonical :
    forall tie_of (runtime : Runtime) (ghost : Ghost) rejected
      capacity selected,
      runtime_selected runtime = KnnSelected capacity selected ->
      phase_heap_certificate tie_of runtime ghost rejected ->
      match runtime_private runtime with
      | AwaitingPublication pending =>
          exists history,
            ghost_verified_history ghost = pending :: history /\
            map (entry_of_result tie_of) selected =
              firstn capacity (sort_ranked
                (map (entry_of_result tie_of) history))
      | _ => map (entry_of_result tie_of) selected =
          firstn capacity (sort_ranked
            (map (entry_of_result tie_of)
              (ghost_verified_history ghost)))
      end.
  Proof.
    intros tie_of runtime ghost rejected capacity selected Hselected Hphase.
    unfold phase_heap_certificate in Hphase.
    rewrite Hselected in Hphase.
    destruct (runtime_private runtime) as
      [|index children|original|pending|key]; simpl in *;
      try (now apply certified_session_selection_is_canonical
        with (rejected := rejected)).
    destruct Hphase as [history [Hhistory [Hcertificate Hfresh]]].
    exists history; split; [exact Hhistory |].
    now apply certified_session_selection_is_canonical
      with (rejected := rejected).
  Qed.

  Theorem phase_heap_history_has_fresh_identities :
    forall tie_of (runtime : Runtime) (ghost : Ghost) rejected,
      phase_heap_certificate tie_of runtime ghost rejected ->
      NoDup (map result_original (ghost_verified_history ghost)) /\
      NoDup (map (fun result => tie_of (result_original result))
        (ghost_verified_history ghost)).
  Proof.
    intros tie_of runtime ghost rejected Hphase.
    unfold phase_heap_certificate in Hphase.
    destruct (runtime_selected runtime) as [range | capacity selected];
      [contradiction |].
    destruct (runtime_private runtime) as
      [|index children|original|pending|key].
    all: try (unfold session_heap_certificate in Hphase;
      pose proof (best_distinct_originals Hphase) as Horiginals;
      pose proof (best_distinct_ties Hphase) as Hties;
      rewrite map_map in Horiginals, Hties;
      split; [exact Horiginals | exact Hties]).
    destruct Hphase as [history [Hhistory [Hcertificate [Hfresh_id Hfresh_tie]]]].
    unfold session_heap_certificate in Hcertificate.
    pose proof (best_distinct_originals Hcertificate) as Horiginals.
    pose proof (best_distinct_ties Hcertificate) as Hties.
    rewrite map_map in Horiginals, Hties.
    rewrite Hhistory; simpl; split; constructor; assumption.
  Qed.

  (** Only the rank projection is updated here. Exact-score authority and
      witness validity belong to the verifier theorem. Rejecting/displacing
      an original also needs an ownership/exclusion action in the full model. *)
  Definition stage_heap_result (before : Runtime) (result : ranked_result nat)
      : Runtime :=
    {| runtime_identity := runtime_identity before;
       runtime_pending := runtime_pending before;
       runtime_private := AwaitingPublication result;
       runtime_selected := runtime_selected before;
       runtime_emission := runtime_emission before;
       runtime_arena := runtime_arena before;
       runtime_cache := runtime_cache before;
       runtime_ledger := runtime_ledger before;
       runtime_reconstruction := runtime_reconstruction before;
       runtime_status := runtime_status before |}.

  Definition record_heap_result (ghost : Ghost) (result : ranked_result nat)
      : Ghost :=
    {| ghost_verified_history := result :: ghost_verified_history ghost;
       ghost_emitted_results := ghost_emitted_results ghost;
       ghost_exclusions := ghost_exclusions ghost |}.

  Definition publish_heap_result (before : Runtime) (capacity : nat)
      (selected : list ranked_entry) : Runtime :=
    {| runtime_identity := runtime_identity before;
       runtime_pending := runtime_pending before;
       runtime_private := NoPrivate;
       runtime_selected := KnnSelected capacity (map result_of_entry selected);
       runtime_emission := runtime_emission before;
       runtime_arena := runtime_arena before;
       runtime_cache := runtime_cache before;
       runtime_ledger := runtime_ledger before;
       runtime_reconstruction := runtime_reconstruction before;
       runtime_status := runtime_status before |}.

  Theorem staging_preserves_phase_heap_certificate :
    forall tie_of (before : Runtime) (ghost : Ghost) rejected result
      capacity selected,
      runtime_private before = ScoringCandidate (result_original result) ->
      runtime_selected before = KnnSelected capacity selected ->
      phase_heap_certificate tie_of before ghost rejected ->
      fresh_result tie_of result (ghost_verified_history ghost) ->
      phase_heap_certificate tie_of (stage_heap_result before result)
        (record_heap_result ghost result) rejected.
  Proof.
    intros tie_of before ghost rejected result capacity selected
      Hprivate Hselected Hphase Hfresh.
    unfold phase_heap_certificate in Hphase.
    rewrite Hselected, Hprivate in Hphase.
    unfold phase_heap_certificate, stage_heap_result, record_heap_result; simpl.
    rewrite Hselected.
    exists (ghost_verified_history ghost).
    split; [reflexivity |].
    split; assumption.
  Qed.

  Theorem publication_preserves_phase_heap_certificate :
    forall tie_of (before : Runtime) (ghost : Ghost) rejected pending
      capacity selected next_selected next_rejected,
      runtime_private before = AwaitingPublication pending ->
      runtime_selected before = KnnSelected capacity selected ->
      phase_heap_certificate tie_of before ghost rejected ->
      best_k_step capacity (entry_of_result tie_of pending)
        (map (entry_of_result tie_of) selected) rejected
        next_selected next_rejected ->
      phase_heap_certificate tie_of
        (publish_heap_result before capacity next_selected) ghost next_rejected.
  Proof.
    intros tie_of before ghost rejected pending capacity selected
      next_selected next_rejected Hprivate Hselected Hphase Hstep.
    unfold phase_heap_certificate in Hphase.
    rewrite Hselected, Hprivate in Hphase.
    destruct Hphase as [history [Hhistory [Hcertificate [Hfresh_id Hfresh_tie]]]].
    unfold phase_heap_certificate, publish_heap_result; simpl.
    rewrite Hhistory.
    eapply publication_step_preserves_session_heap_certificate; eauto.
  Qed.

  (** Guards describe the rank-phase projection of an active session action.
      They do not replace verifier soundness or exclusion evidence. *)
  Inductive session_heap_step (tie_of : nat -> nat) :
      Runtime -> Ghost -> list ranked_entry ->
      Runtime -> Ghost -> list ranked_entry -> Prop :=
  | HeapStageRank : forall before ghost rejected result capacity selected,
      runtime_status before = Active ->
      runtime_private before = ScoringCandidate (result_original result) ->
      runtime_selected before = KnnSelected capacity selected ->
      fresh_result tie_of result (ghost_verified_history ghost) ->
      session_heap_step tie_of before ghost rejected
        (stage_heap_result before result) (record_heap_result ghost result)
        rejected
  | HeapPublishRank : forall before ghost rejected pending capacity selected
      next_selected next_rejected,
      runtime_status before = Active ->
      runtime_private before = AwaitingPublication pending ->
      runtime_selected before = KnnSelected capacity selected ->
      best_k_step capacity (entry_of_result tie_of pending)
        (map (entry_of_result tie_of) selected) rejected
        next_selected next_rejected ->
      session_heap_step tie_of before ghost rejected
        (publish_heap_result before capacity next_selected) ghost next_rejected.

  Theorem session_heap_step_preserves_phase_certificate :
    forall tie_of before ghost rejected after next_ghost next_rejected,
      session_heap_step tie_of before ghost rejected after next_ghost
        next_rejected ->
      phase_heap_certificate tie_of before ghost rejected ->
      phase_heap_certificate tie_of after next_ghost next_rejected.
  Proof.
    intros tie_of before ghost rejected after next_ghost next_rejected
      Hstep Hphase.
    inversion Hstep; subst.
    - eapply staging_preserves_phase_heap_certificate; eauto.
    - eapply publication_preserves_phase_heap_certificate; eauto.
  Qed.

  Theorem awaiting_publication_has_a_certified_heap_step :
    forall tie_of (before : Runtime) (ghost : Ghost) rejected pending
      capacity selected,
      runtime_status before = Active ->
      runtime_private before = AwaitingPublication pending ->
      runtime_selected before = KnnSelected capacity selected ->
      phase_heap_certificate tie_of before ghost rejected ->
      exists next_selected next_rejected,
        session_heap_step tie_of before ghost rejected
          (publish_heap_result before capacity next_selected) ghost
          next_rejected /\
        phase_heap_certificate tie_of
          (publish_heap_result before capacity next_selected) ghost
          next_rejected.
  Proof.
    intros tie_of before ghost rejected pending capacity selected
      Hactive Hprivate Hselected Hphase.
    pose proof Hphase as Htail.
    unfold phase_heap_certificate in Htail.
    rewrite Hselected, Hprivate in Htail.
    destruct Htail as [history [Hhistory [Hcertificate Hfresh]]].
    unfold session_heap_certificate in Hcertificate.
    destruct (@best_k_step_exists capacity (entry_of_result tie_of pending)
      (map (entry_of_result tie_of) history)
      (map (entry_of_result tie_of) selected) rejected Hcertificate)
      as [next_selected [next_rejected Hstep]].
    exists next_selected, next_rejected; split.
    - eapply HeapPublishRank; eauto.
    - eapply publication_preserves_phase_heap_certificate; eauto.
  Qed.

  Theorem completed_session_heap_is_canonical :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid tie_of
      (runtime : Runtime) (ghost : Ghost) rejected capacity selected,
      session_abstraction region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid runtime ghost ->
      runtime_status runtime = Completed ->
      runtime_selected runtime = KnnSelected capacity selected ->
      phase_heap_certificate tie_of runtime ghost rejected ->
      selected = map result_of_entry
        (firstn capacity (sort_ranked
          (map (entry_of_result tie_of) (ghost_verified_history ghost)))).
  Proof.
    intros region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid tie_of runtime ghost rejected
      capacity selected Habstraction Hcompleted Hselected Hphase.
    assert (Hprivate : runtime_private runtime = NoPrivate).
    { unfold session_abstraction in Habstraction; tauto. }
    unfold phase_heap_certificate in Hphase.
    rewrite Hselected, Hprivate in Hphase.
    now apply certified_session_result_list_is_canonical
      with (rejected := rejected).
  Qed.
End RuntimePhase.

Definition pending_control_rank : ranked_entry :=
  {| entry_original := 3; entry_cost := 2; entry_tie := 3 |}.

Example staged_verification_is_not_premature_heap_publication :
  best_k_certificate 1 [] [] [] /\
  ~ best_k_certificate 1 [pending_control_rank] [] [].
Proof.
  split; [apply empty_best_k_certificate |].
  intro Hbad.
  pose proof (Permutation_length (best_cover Hbad)) as Hlength.
  simpl in Hlength; discriminate.
Qed.

Module HeapPhaseControls.
  Definition Runtime :=
    session_runtime unit unit unit nat unit unit unit unit unit unit unit.
  Definition Ghost := session_ghost nat unit.
  Definition tie_of (original : nat) := original.
  Definition result : ranked_result nat :=
    {| result_original := 3; result_score := 2 |}.

  Definition before : Runtime :=
    {| runtime_identity := compact_identity;
       runtime_pending := [];
       runtime_private := ScoringCandidate 3;
       runtime_selected := KnnSelected 1 [];
       runtime_emission :=
         {| committed_output_count := 0; next_page_index := 0 |};
       runtime_arena := tt;
       runtime_cache := [];
       runtime_ledger := empty_ledger;
       runtime_reconstruction := tt;
       runtime_status := Active |}.

  Definition ghost : Ghost :=
    {| ghost_verified_history := [];
       ghost_emitted_results := [];
       ghost_exclusions := [] |}.

  Definition staged := stage_heap_result before result.
  Definition staged_ghost := record_heap_result ghost result.
  Definition published :=
    publish_heap_result staged 1 [entry_of_result tie_of result].
  Definition prematurely_cleared := publish_heap_result staged 1 [].

  Lemma before_has_heap_certificate :
    phase_heap_certificate tie_of before ghost [].
  Proof. apply empty_best_k_certificate. Qed.

  Lemma staging_is_a_legal_rank_step :
    session_heap_step tie_of before ghost [] staged staged_ghost [].
  Proof.
    eapply HeapStageRank with (capacity := 1) (selected := []).
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - split; simpl; tauto.
  Qed.

  Lemma staged_has_heap_certificate :
    phase_heap_certificate tie_of staged staged_ghost [].
  Proof.
    eapply session_heap_step_preserves_phase_certificate.
    - exact staging_is_a_legal_rank_step.
    - exact before_has_heap_certificate.
  Qed.

  Lemma publication_is_a_legal_rank_step :
    session_heap_step tie_of staged staged_ghost [] published staged_ghost [].
  Proof.
    eapply HeapPublishRank with (pending := result) (selected := []).
    - reflexivity.
    - reflexivity.
    - reflexivity.
    - change (best_k_step 1 (entry_of_result tie_of result) [] []
        (insert_ranked (entry_of_result tie_of result) []) []).
      apply BestFill; simpl; lia.
  Qed.

  Example fresh_staged_result_is_published_canonically :
    phase_heap_certificate tie_of staged staged_ghost [] /\
    phase_heap_certificate tie_of published staged_ghost [] /\
    runtime_selected published = KnnSelected 1 [result].
  Proof.
    split; [exact staged_has_heap_certificate |].
    split.
    - eapply session_heap_step_preserves_phase_certificate.
      + exact publication_is_a_legal_rank_step.
      + exact staged_has_heap_certificate.
    - reflexivity.
  Qed.

  Example clearing_private_before_heap_publication_is_rejected :
    phase_heap_certificate tie_of staged staged_ghost [] /\
    ~ phase_heap_certificate tie_of prematurely_cleared staged_ghost [].
  Proof.
    split; [exact staged_has_heap_certificate |].
    intro Hbad.
    change (best_k_certificate 1 [entry_of_result tie_of result] [] [])
      in Hbad.
    pose proof (Permutation_length (best_cover Hbad)) as Hlength.
    simpl in Hlength; discriminate.
  Qed.

  Example duplicate_original_cannot_be_staged_twice :
    ~ fresh_result tie_of result [result].
  Proof.
    intros [Horiginal _]; apply Horiginal; simpl; auto.
  Qed.

  Example distinct_originals_do_not_justify_a_colliding_tie :
    result_original result <> 4 /\
    ~ fresh_result (fun _ => 0)
      {| result_original := 4; result_score := 1 |} [result].
  Proof.
    split; [discriminate |].
    intros [_ Htie]; apply Htie; simpl; auto.
  Qed.
End HeapPhaseControls.
