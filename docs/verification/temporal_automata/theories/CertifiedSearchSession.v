(** * Candidate ownership in a finite search session

    Each natural number names one original occurrence, even when dictionary
    nodes are shared. A private work item takes ownership before processing;
    publication moves it atomically to verified or soundly excluded. These
    generic laws do not establish the Rust scheduler or a concrete rank bound.
    In particular, a complete result theorem needs a separate proof that all
    pending/private work has been discharged and that exclusions are sound. *)

From Stdlib Require Import Arith Lia List Permutation.
Import ListNotations.

Record search_session := {
  pending_originals : list nat;
  private_originals : list nat;
  verified_originals : list nat;
  excluded_originals : list nat
}.

Definition owned_originals (session : search_session) : list nat :=
  pending_originals session ++ private_originals session ++
  verified_originals session ++ excluded_originals session.

Definition exact_ownership (universe : list nat) (session : search_session)
    : Prop :=
  NoDup universe /\ Permutation (owned_originals session) universe.

Definition exclusions_sound (winner : nat -> Prop) (session : search_session)
    : Prop :=
  forall original, In original (excluded_originals session) -> ~ winner original.

Definition begin_private (session : search_session)
    (prefix suffix : list nat) (original : nat) : search_session :=
  {| pending_originals := prefix ++ suffix;
     private_originals := original :: private_originals session;
     verified_originals := verified_originals session;
     excluded_originals := excluded_originals session |}.

Definition publish_verified (session : search_session)
    (prefix suffix : list nat) (original : nat) : search_session :=
  {| pending_originals := pending_originals session;
     private_originals := prefix ++ suffix;
     verified_originals := original :: verified_originals session;
     excluded_originals := excluded_originals session |}.

Definition publish_excluded (session : search_session)
    (prefix suffix : list nat) (original : nat) : search_session :=
  {| pending_originals := pending_originals session;
     private_originals := prefix ++ suffix;
     verified_originals := verified_originals session;
     excluded_originals := original :: excluded_originals session |}.

Lemma move_one_across_boundary :
  forall (prefix suffix destination tail : list nat) (original : nat),
  Permutation
    ((prefix ++ original :: suffix) ++ destination ++ tail)
    ((prefix ++ suffix) ++ (original :: destination) ++ tail).
Proof.
  intros prefix suffix destination tail original.
  repeat rewrite app_assoc.
  apply Permutation_app_tail.
  repeat rewrite <- app_assoc.
  apply Permutation_app_head.
  apply Permutation_middle.
Qed.

Theorem begin_private_preserves_exact_ownership :
  forall universe session prefix suffix original,
    pending_originals session = prefix ++ original :: suffix ->
    exact_ownership universe session ->
    exact_ownership universe (begin_private session prefix suffix original).
Proof.
  intros universe session prefix suffix original Hpending [Hnodup Hperm].
  split; [exact Hnodup |].
  unfold exact_ownership, owned_originals, begin_private in *; simpl.
  rewrite Hpending in Hperm.
  eapply Permutation_trans; [|exact Hperm].
  symmetry. apply move_one_across_boundary.
Qed.

Theorem begin_private_preserves_sound_exclusions :
  forall winner session prefix suffix original,
    exclusions_sound winner session ->
    exclusions_sound winner (begin_private session prefix suffix original).
Proof. intros; exact H. Qed.

Theorem publish_verified_preserves_exact_ownership :
  forall universe session prefix suffix original,
    private_originals session = prefix ++ original :: suffix ->
    exact_ownership universe session ->
    exact_ownership universe (publish_verified session prefix suffix original).
Proof.
  intros universe session prefix suffix original Hprivate [Hnodup Hperm].
  split; [exact Hnodup |].
  unfold owned_originals, publish_verified in *; simpl.
  rewrite Hprivate in Hperm.
  eapply Permutation_trans; [|exact Hperm].
  apply Permutation_app_head.
  symmetry. apply move_one_across_boundary.
Qed.

Theorem publish_verified_preserves_sound_exclusions :
  forall winner session prefix suffix original,
    exclusions_sound winner session ->
    exclusions_sound winner (publish_verified session prefix suffix original).
Proof. intros; exact H. Qed.

Theorem publish_excluded_preserves_exact_ownership :
  forall universe session prefix suffix original,
    private_originals session = prefix ++ original :: suffix ->
    exact_ownership universe session ->
    exact_ownership universe (publish_excluded session prefix suffix original).
Proof.
  intros universe session prefix suffix original Hprivate [Hnodup Hperm].
  split; [exact Hnodup |].
  unfold owned_originals, publish_excluded in *; simpl.
  rewrite Hprivate in Hperm.
  eapply Permutation_trans; [|exact Hperm].
  apply Permutation_app_head.
  repeat rewrite <- app_assoc.
  apply Permutation_app_head.
  symmetry.
  rewrite (app_assoc suffix (verified_originals session)
    (original :: excluded_originals session)).
  simpl.
  rewrite (app_assoc suffix (verified_originals session)
    (excluded_originals session)).
  apply Permutation_middle.
Qed.

Theorem publish_excluded_preserves_sound_exclusions :
  forall winner session prefix suffix original,
    ~ winner original ->
    exclusions_sound winner session ->
    exclusions_sound winner (publish_excluded session prefix suffix original).
Proof.
  intros winner session prefix suffix original Hnot Hsound candidate Hin.
  simpl in Hin. destruct Hin as [Heq | Hin].
  - subst; exact Hnot.
  - now apply Hsound.
Qed.

Inductive ownership_step (winner : nat -> Prop)
    : search_session -> search_session -> Prop :=
| ownership_begin : forall session prefix suffix original,
    pending_originals session = prefix ++ original :: suffix ->
    ownership_step winner session
      (begin_private session prefix suffix original)
| ownership_verify : forall session prefix suffix original,
    private_originals session = prefix ++ original :: suffix ->
    ownership_step winner session
      (publish_verified session prefix suffix original)
| ownership_exclude : forall session prefix suffix original,
    private_originals session = prefix ++ original :: suffix ->
    ~ winner original ->
    ownership_step winner session
      (publish_excluded session prefix suffix original).

Inductive ownership_steps (winner : nat -> Prop)
    : search_session -> search_session -> Prop :=
| ownership_refl : forall session, ownership_steps winner session session
| ownership_next : forall first middle last,
    ownership_step winner first middle ->
    ownership_steps winner middle last ->
    ownership_steps winner first last.

Theorem ownership_step_preserves_invariant :
  forall universe winner first last,
    ownership_step winner first last ->
    exact_ownership universe first /\ exclusions_sound winner first ->
    exact_ownership universe last /\ exclusions_sound winner last.
Proof.
  intros universe winner first last Hstep [Howned Hexcluded].
  inversion Hstep; subst.
  - split.
    + eapply begin_private_preserves_exact_ownership; eauto.
    + eapply begin_private_preserves_sound_exclusions; eauto.
  - split.
    + eapply publish_verified_preserves_exact_ownership; eauto.
    + eapply publish_verified_preserves_sound_exclusions; eauto.
  - split.
    + eapply publish_excluded_preserves_exact_ownership; eauto.
    + eapply publish_excluded_preserves_sound_exclusions; eauto.
Qed.

Theorem finite_session_preserves_invariant :
  forall universe winner first last,
    ownership_steps winner first last ->
    exact_ownership universe first /\ exclusions_sound winner first ->
    exact_ownership universe last /\ exclusions_sound winner last.
Proof.
  intros universe winner first last Hsteps.
  induction Hsteps; intros Hinvariant; [exact Hinvariant |].
  apply IHHsteps.
  eapply ownership_step_preserves_invariant; eauto.
Qed.

Theorem completed_session_contains_every_winner :
  forall universe winner session,
    exact_ownership universe session ->
    exclusions_sound winner session ->
    pending_originals session = [] ->
    private_originals session = [] ->
    forall original, In original universe -> winner original ->
      In original (verified_originals session).
Proof.
  intros universe winner session [_ Hperm] Hsound Hpending Hprivate
    original Huniverse Hwinner.
  assert (In original (owned_originals session)) as Howned.
  { eapply Permutation_in.
    - apply Permutation_sym; exact Hperm.
    - exact Huniverse. }
  unfold owned_originals in Howned.
  rewrite Hpending, Hprivate in Howned; simpl in Howned.
  apply in_app_or in Howned as [Hverified | Hexcluded].
  - exact Hverified.
  - exfalso. exact (Hsound original Hexcluded Hwinner).
Qed.

Theorem exact_ownership_prevents_duplicate_originals :
  forall universe session,
    exact_ownership universe session ->
    NoDup (owned_originals session).
Proof.
  intros universe session [Hnodup Hperm].
  eapply Permutation_NoDup; [apply Permutation_sym; exact Hperm | exact Hnodup].
Qed.

Definition initial_session (universe : list nat) : search_session :=
  {| pending_originals := universe;
     private_originals := [];
     verified_originals := [];
     excluded_originals := [] |}.

Definition public_only_originals (session : search_session) : list nat :=
  pending_originals session ++ verified_originals session ++
  excluded_originals session.

Example omitting_private_work_loses_an_original :
  let private := begin_private (initial_session [0]) [] [] 0 in
  exact_ownership [0] private /\
  public_only_originals private = [] /\
  ~ Permutation (public_only_originals private) [0].
Proof.
  simpl; repeat split.
  - constructor; [intro H; inversion H | constructor].
  - reflexivity.
  - intro Hperm.
    apply Permutation_length in Hperm; simpl in Hperm; discriminate.
Qed.

Lemma initial_session_invariant : forall universe winner,
  NoDup universe ->
  exact_ownership universe (initial_session universe) /\
  exclusions_sound winner (initial_session universe).
Proof.
  intros universe winner Hnodup; split.
  - split; [exact Hnodup |].
    unfold owned_originals, initial_session; simpl.
    now rewrite app_nil_r.
  - intros original Hin; inversion Hin.
Qed.

Theorem finite_completed_run_retains_every_winner :
  forall universe winner final,
    NoDup universe ->
    ownership_steps winner (initial_session universe) final ->
    pending_originals final = [] ->
    private_originals final = [] ->
    forall original, In original universe -> winner original ->
      In original (verified_originals final).
Proof.
  intros universe winner final Hnodup Hrun Hpending Hprivate original
    Huniverse Hwinner.
  pose proof (finite_session_preserves_invariant universe winner
    (initial_session universe) final Hrun
    (initial_session_invariant universe winner Hnodup))
    as [Howned Hexcluded].
  eapply completed_session_contains_every_winner; eauto.
Qed.

Theorem completed_verified_filter_has_exact_winner_membership :
  forall universe winnerb final,
    NoDup universe ->
    ownership_steps (fun original => winnerb original = true)
      (initial_session universe) final ->
    pending_originals final = [] ->
    private_originals final = [] ->
    forall original,
      In original (filter winnerb (verified_originals final)) <->
      In original (filter winnerb universe).
Proof.
  intros universe winnerb final Hnodup Hrun Hpending Hprivate original.
  pose proof (finite_session_preserves_invariant universe
    (fun id => winnerb id = true) (initial_session universe) final Hrun
    (initial_session_invariant universe _ Hnodup))
    as [[_ Hperm] _].
  split; intro Hin.
  - apply filter_In in Hin as [Hverified Hwinner].
    apply filter_In; split; [|exact Hwinner].
    assert (In original (owned_originals final)) as Howned.
    { unfold owned_originals; rewrite Hpending, Hprivate; simpl.
      apply in_or_app; now left. }
    eapply Permutation_in; eauto.
  - apply filter_In in Hin as [Huniverse Hwinner].
    apply filter_In; split; [|exact Hwinner].
    eapply finite_completed_run_retains_every_winner; eauto.
Qed.

(** A cursor-shaped specialization of the independent private-owner law.
    [Scoring] is ephemeral work after the cursor advances. [AwaitingResult]
    corresponds to a retained exact match awaiting a result-page slot. The
    published continuation stores the latter as [pending_match]. A pause is
    a stutter on candidate ownership; it supplies no termination theorem. *)
Inductive cursor_phase :=
| CursorIdle
| Scoring : nat -> cursor_phase
| AwaitingResult : nat -> cursor_phase.

Definition phase_originals (phase : cursor_phase) : list nat :=
  match phase with
  | CursorIdle => []
  | Scoring original | AwaitingResult original => [original]
  end.

Record range_cursor_state := {
  cursor_revision : nat;
  cursor_unseen : list nat;
  cursor_phase_state : cursor_phase;
  (** Reverse encounter history, not the Rust output Vec order. *)
  cursor_results : list nat;
  cursor_rejected : list nat
}.

Definition cursor_projection (state : range_cursor_state) : search_session :=
  {| pending_originals := cursor_unseen state;
     private_originals := phase_originals (cursor_phase_state state);
     verified_originals := cursor_results state;
     excluded_originals := cursor_rejected state |}.

Definition cursor_invariant (universe : list nat) (revision : nat)
    (winner : nat -> Prop) (state : range_cursor_state) : Prop :=
  cursor_revision state = revision /\
  exact_ownership universe (cursor_projection state) /\
  exclusions_sound winner (cursor_projection state).

Inductive cursor_step (winner : nat -> Prop)
    : range_cursor_state -> range_cursor_state -> Prop :=
| cursor_pick : forall revision original remaining results rejected,
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := original :: remaining;
         cursor_phase_state := CursorIdle; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := Scoring original; cursor_results := results;
         cursor_rejected := rejected |}
| cursor_reject : forall revision original remaining results rejected,
    ~ winner original ->
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := Scoring original; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := CursorIdle; cursor_results := results;
         cursor_rejected := original :: rejected |}
| cursor_accept : forall revision original remaining results rejected,
    winner original ->
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := Scoring original; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := AwaitingResult original; cursor_results := results;
         cursor_rejected := rejected |}
| cursor_publish : forall revision original remaining results rejected,
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := AwaitingResult original; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := CursorIdle; cursor_results := original :: results;
         cursor_rejected := rejected |}
| cursor_pause_idle : forall revision remaining results rejected,
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := CursorIdle; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := CursorIdle; cursor_results := results;
         cursor_rejected := rejected |}
| cursor_pause_pending : forall revision original remaining results rejected,
    cursor_step winner
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := AwaitingResult original; cursor_results := results;
         cursor_rejected := rejected |}
      {| cursor_revision := revision; cursor_unseen := remaining;
         cursor_phase_state := AwaitingResult original; cursor_results := results;
         cursor_rejected := rejected |}.

Theorem cursor_step_preserves_snapshot : forall winner first last,
  cursor_step winner first last ->
  cursor_revision last = cursor_revision first.
Proof. intros winner first last Hstep; inversion Hstep; reflexivity. Qed.

Theorem cursor_step_projects_to_ownership : forall winner first last,
  cursor_step winner first last ->
  cursor_projection first = cursor_projection last \/
  ownership_step winner (cursor_projection first) (cursor_projection last).
Proof.
  intros winner first last Hstep; inversion Hstep; subst; simpl.
  - right. apply ownership_begin with (prefix := []) (suffix := remaining)
      (original := original). reflexivity.
  - right. apply ownership_exclude with (prefix := []) (suffix := [])
      (original := original); [reflexivity | assumption].
  - left; reflexivity.
  - right. apply ownership_verify with (prefix := []) (suffix := [])
      (original := original). reflexivity.
  - left; reflexivity.
  - left; reflexivity.
Qed.

Theorem cursor_step_preserves_invariant :
  forall universe revision winner first last,
    cursor_step winner first last ->
    cursor_invariant universe revision winner first ->
    cursor_invariant universe revision winner last.
Proof.
  intros universe revision winner first last Hstep
    [Hrevision [Howned Hexcluded]].
  split.
  - rewrite (cursor_step_preserves_snapshot _ _ _ Hstep).
    exact Hrevision.
  - destruct (cursor_step_projects_to_ownership _ _ _ Hstep)
      as [Hequal | Hmove].
    + now rewrite <- Hequal.
    + eapply ownership_step_preserves_invariant; eauto.
Qed.

Definition cursor_result_sound (winner : nat -> Prop)
    (state : range_cursor_state) : Prop :=
  Forall winner (cursor_results state) /\
  match cursor_phase_state state with
  | AwaitingResult original => winner original
  | _ => True
  end.

Theorem cursor_step_preserves_result_sound :
  forall winner first last,
    cursor_step winner first last ->
    cursor_result_sound winner first ->
    cursor_result_sound winner last.
Proof.
  intros winner first last Hstep Hsound.
  inversion Hstep; subst; simpl in *;
    destruct Hsound as [Hresults Hpending].
  - now split.
  - now split.
  - now split.
  - split; [constructor; assumption | exact I].
  - now split.
  - now split.
Qed.

Inductive cursor_steps (winner : nat -> Prop)
    : range_cursor_state -> range_cursor_state -> Prop :=
| cursor_steps_refl : forall state, cursor_steps winner state state
| cursor_steps_next : forall first middle last,
    cursor_step winner first middle ->
    cursor_steps winner middle last ->
    cursor_steps winner first last.

Definition initial_cursor (revision : nat) (universe : list nat)
    : range_cursor_state :=
  {| cursor_revision := revision; cursor_unseen := universe;
     cursor_phase_state := CursorIdle; cursor_results := [];
     cursor_rejected := [] |}.

Theorem initial_cursor_invariant : forall universe revision winner,
  NoDup universe ->
  cursor_invariant universe revision winner (initial_cursor revision universe).
Proof.
  intros universe revision winner Hnodup.
  split; [reflexivity |].
  unfold cursor_projection, initial_cursor; simpl.
  apply initial_session_invariant; exact Hnodup.
Qed.

Theorem cursor_run_preserves_invariant :
  forall universe revision winner first last,
    cursor_steps winner first last ->
    cursor_invariant universe revision winner first ->
    cursor_invariant universe revision winner last.
Proof.
  intros universe revision winner first last Hsteps.
  induction Hsteps; intros Hinvariant; [exact Hinvariant |].
  apply IHHsteps.
  eapply cursor_step_preserves_invariant; eauto.
Qed.

Theorem cursor_run_preserves_result_sound :
  forall winner first last,
    cursor_steps winner first last ->
    cursor_result_sound winner first ->
    cursor_result_sound winner last.
Proof.
  intros winner first last Hsteps.
  induction Hsteps; intros Hsound; [exact Hsound |].
  apply IHHsteps.
  eapply cursor_step_preserves_result_sound; eauto.
Qed.

Theorem exhausted_cursor_contains_every_winner :
  forall universe revision winner final,
    NoDup universe ->
    cursor_steps winner (initial_cursor revision universe) final ->
    cursor_unseen final = [] ->
    cursor_phase_state final = CursorIdle ->
    forall original, In original universe -> winner original ->
      In original (cursor_results final).
Proof.
  intros universe revision winner final Hnodup Hrun Hunseen Hidle
    original Huniverse Hwinner.
  pose proof (cursor_run_preserves_invariant universe revision winner
    (initial_cursor revision universe) final Hrun
    (initial_cursor_invariant universe revision winner Hnodup))
    as [_ [Howned Hexcluded]].
  unfold cursor_projection in Howned, Hexcluded; simpl in *.
  change (In original (verified_originals (cursor_projection final))).
  eapply completed_session_contains_every_winner; eauto.
  simpl; now rewrite Hidle.
Qed.

Theorem exhausted_cursor_has_exact_range_membership :
  forall universe revision winner final,
    NoDup universe ->
    cursor_steps winner (initial_cursor revision universe) final ->
    cursor_unseen final = [] ->
    cursor_phase_state final = CursorIdle ->
    forall original,
      In original (cursor_results final) <->
      In original universe /\ winner original.
Proof.
  intros universe revision winner final Hnodup Hrun Hunseen Hidle original.
  pose proof (cursor_run_preserves_invariant universe revision winner
    (initial_cursor revision universe) final Hrun
    (initial_cursor_invariant universe revision winner Hnodup))
    as [_ [[_ Hperm] _]].
  pose proof (cursor_run_preserves_result_sound winner
    (initial_cursor revision universe) final Hrun)
    as Hsound.
  assert (cursor_result_sound winner (initial_cursor revision universe))
    as Hinitial by (split; constructor).
  specialize (Hsound Hinitial).
  destruct Hsound as [Hresults _].
  split.
  - intro Hin.
    split.
    + assert (In original (owned_originals (cursor_projection final)))
        as Howned.
      { unfold owned_originals, cursor_projection; simpl.
        rewrite Hunseen, Hidle; simpl.
        apply in_or_app; now left. }
      eapply Permutation_in; eauto.
    + apply Forall_forall with (x := original) in Hresults; assumption.
  - intros [Huniverse Hwinner].
    eapply exhausted_cursor_contains_every_winner; eauto.
Qed.

Example awaiting_result_is_exclusive_private_ownership :
  let waiting :=
    {| cursor_revision := 1; cursor_unseen := [];
       cursor_phase_state := AwaitingResult 0;
       cursor_results := []; cursor_rejected := [] |} in
  exact_ownership [0] (cursor_projection waiting) /\
  public_only_originals (cursor_projection waiting) = [].
Proof.
  simpl; split.
  - split.
    + constructor; [intro H; inversion H | constructor].
    + reflexivity.
  - reflexivity.
Qed.

(** * Complete session state and its erased proof history

    The earlier [search_session] is an ownership quotient. The types below
    name the runtime components needed by a whole-search proof. A snapshot-
    indexed interpretation gives each compact occurrence its original region;
    that region is not a stored runtime list. Borrowed split work does not
    take a second copy of its
    parent's region; scoring and a result awaiting publication do own one
    original exclusively. The [session_ghost] is a proof view and is absent
    from [session_runtime] and its storage measure. *)
Module CompleteState.
Set Implicit Arguments.

Record occurrence (Node Residual Path : Type) := {
  occurrence_node : Node;
  occurrence_residual : Residual;
  occurrence_path : Path;
  occurrence_cursor : nat
}.

Record ranked_result (Score : Type) := {
  result_original : nat;
  result_score : Score
}.

Arguments result_original {Score} _.
Arguments result_score {Score} _.

Inductive selected_results (Score : Type) :=
| RangeSelected : list (ranked_result Score) -> selected_results Score
| KnnSelected : nat -> list (ranked_result Score) -> selected_results Score.

Arguments RangeSelected {Score} _.
Arguments KnnSelected {Score} _ _.

Definition selected_entries {Score} (selected : selected_results Score)
    : list (ranked_result Score) :=
  match selected with
  | RangeSelected entries | KnnSelected _ entries => entries
  end.

Inductive private_phase (Node Residual Path Score CacheKey : Type) :=
| NoPrivate
| BorrowedSplit : nat -> list (occurrence Node Residual Path) ->
    private_phase Node Residual Path Score CacheKey
| ScoringCandidate : nat -> private_phase Node Residual Path Score CacheKey
| AwaitingPublication : ranked_result Score ->
    private_phase Node Residual Path Score CacheKey
| BuildingCacheEntry : CacheKey ->
    private_phase Node Residual Path Score CacheKey.

Arguments NoPrivate {Node Residual Path Score CacheKey}.
Arguments BorrowedSplit {Node Residual Path Score CacheKey} _ _.
Arguments ScoringCandidate {Node Residual Path Score CacheKey} _.
Arguments AwaitingPublication {Node Residual Path Score CacheKey} _.
Arguments BuildingCacheEntry {Node Residual Path Score CacheKey} _.

Definition exclusively_private {Node Residual Path Score CacheKey}
    (phase : private_phase Node Residual Path Score CacheKey) : list nat :=
  match phase with
  | ScoringCandidate original => [original]
  | AwaitingPublication result => [result_original result]
  | NoPrivate | BorrowedSplit _ _ | BuildingCacheEntry _ => []
  end.

Record session_identity (Contract Snapshot Query : Type) := {
  session_contract : Contract;
  session_snapshot : Snapshot;
  session_query : Query;
  session_revision : nat
}.

Record complete_cache_entry (Contract Snapshot Query Key Value : Type) := {
  cache_scope : session_identity Contract Snapshot Query;
  cache_key : Key;
  cache_value : Value
}.

Record work_ledger := {
  work_executed : nat;
  work_reserved : nat;
  work_charged : nat;
  bytes_live : nat;
  bytes_peak : nat;
  allocation_count : nat
}.

Record emission_state := {
  committed_output_count : nat;
  next_page_index : nat
}.

Inductive session_status :=
| Active
| Suspended : nat -> session_status
| Completed
| Failed : nat -> session_status.

Record session_runtime
    (Node Residual Path Score Contract Snapshot Query Key Value Arena
     Reconstruction : Type) := {
  runtime_identity : session_identity Contract Snapshot Query;
  runtime_pending : list (occurrence Node Residual Path);
  runtime_private : private_phase Node Residual Path Score Key;
  runtime_selected : selected_results Score;
  runtime_emission : emission_state;
  runtime_arena : Arena;
  runtime_cache : list (complete_cache_entry Contract Snapshot Query Key Value);
  runtime_ledger : work_ledger;
  runtime_reconstruction : Reconstruction;
  runtime_status : session_status
}.

Record exclusion (Evidence : Type) := {
  exclusion_original : nat;
  exclusion_evidence : Evidence
}.

Arguments exclusion_original {Evidence} _.
Arguments exclusion_evidence {Evidence} _.

Record session_ghost (Score Evidence : Type) := {
  ghost_verified_history : list (ranked_result Score);
  ghost_emitted_results : list (ranked_result Score);
  ghost_exclusions : list (exclusion Evidence)
}.

Definition pending_region_ids {Node Residual Path}
    (region_of : occurrence Node Residual Path -> list nat)
    (pending : list (occurrence Node Residual Path)) : list nat :=
  concat (map region_of pending).

Definition selected_ids {Score} (selected : selected_results Score)
    : list nat := map result_original (selected_entries selected).

Definition staged_occurrences {Node Residual Path Score Key}
    (phase : private_phase Node Residual Path Score Key)
    : list (occurrence Node Residual Path) :=
  match phase with
  | BorrowedSplit _ children => children
  | _ => []
  end.

Section Abstraction.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Let Runtime := session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost Score Evidence.

  Definition ownership_projection
      (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      (runtime : Runtime) (ghost : Ghost)
      : search_session :=
    {| pending_originals :=
         pending_region_ids (region_of (runtime_identity runtime))
           (runtime_pending runtime);
       private_originals := exclusively_private (runtime_private runtime);
       verified_originals := selected_ids (runtime_selected runtime) ++
         map result_original (ghost_emitted_results ghost);
       excluded_originals :=
         map exclusion_original (ghost_exclusions ghost) |}.

  Definition borrowed_parent_live (runtime : Runtime) : Prop :=
    match runtime_private runtime with
    | BorrowedSplit index _ =>
        exists parent, nth_error (runtime_pending runtime) index = Some parent
    | _ => True
    end.

  Definition selected_capacity (runtime : Runtime) : Prop :=
    match runtime_selected runtime with
    | RangeSelected _ => True
    | KnnSelected capacity entries => length entries <= capacity
    end.

  Definition complete_cache_scoped (runtime : Runtime) : Prop :=
    Forall (fun entry => cache_scope entry = runtime_identity runtime)
      (runtime_cache runtime).

  Definition session_abstraction
      (region_of : session_identity Contract Snapshot Query ->
        occurrence Node Residual Path -> list nat)
      (universe : list nat)
      (authoritative : nat -> Score -> Prop)
      (evidence_sound : nat -> Evidence -> Prop)
      (arena_valid : Arena -> Residual -> Prop)
      (cache_valid : session_identity Contract Snapshot Query ->
        Key -> Value -> Prop)
      (reconstruction_valid : Reconstruction -> Path -> nat -> Prop)
      (runtime : Runtime) (ghost : Ghost) : Prop :=
    exact_ownership universe
      (ownership_projection region_of runtime ghost) /\
    borrowed_parent_live runtime /\
    selected_capacity runtime /\
    complete_cache_scoped runtime /\
    Forall (fun occurrence =>
      arena_valid (runtime_arena runtime) (occurrence_residual occurrence) /\
      Forall (reconstruction_valid (runtime_reconstruction runtime)
        (occurrence_path occurrence))
        (region_of (runtime_identity runtime) occurrence))
      (runtime_pending runtime ++ staged_occurrences (runtime_private runtime)) /\
    Forall (fun entry =>
      cache_valid (runtime_identity runtime)
        (cache_key entry) (cache_value entry))
      (runtime_cache runtime) /\
    Forall (fun result =>
      authoritative (result_original result) (result_score result))
      (selected_entries (runtime_selected runtime) ++
       ghost_emitted_results ghost) /\
    Forall (fun result =>
      authoritative (result_original result) (result_score result))
      (ghost_verified_history ghost) /\
    Forall (fun result => In result (ghost_verified_history ghost))
      (selected_entries (runtime_selected runtime) ++
       ghost_emitted_results ghost) /\
    Forall (fun entry =>
      evidence_sound (exclusion_original entry) (exclusion_evidence entry))
      (ghost_exclusions ghost) /\
    work_executed (runtime_ledger runtime) <=
      work_charged (runtime_ledger runtime) /\
    bytes_live (runtime_ledger runtime) <=
      bytes_peak (runtime_ledger runtime) /\
    committed_output_count (runtime_emission runtime) =
      length (ghost_emitted_results ghost) /\
    (runtime_status runtime = Completed ->
      runtime_pending runtime = [] /\
      runtime_private runtime = NoPrivate).

  Theorem abstraction_covers_each_original_once :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid runtime ghost,
      session_abstraction region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid runtime ghost ->
      NoDup (owned_originals (ownership_projection region_of runtime ghost)) /\
      Permutation (owned_originals
        (ownership_projection region_of runtime ghost))
        universe.
  Proof.
    intros region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid runtime ghost [Hown _].
    split.
    - now apply exact_ownership_prevents_duplicate_originals with universe.
    - exact (proj2 Hown).
  Qed.

  Theorem borrowed_split_keeps_parent_ownership :
    forall region_of (runtime : Runtime) (ghost : Ghost) index children,
      runtime_private runtime = BorrowedSplit index children ->
      private_originals
        (ownership_projection region_of runtime ghost) = [].
  Proof. intros; unfold ownership_projection; now rewrite H. Qed.

  Theorem verification_history_is_not_an_owner :
    forall region_of (runtime : Runtime) (first later : Ghost),
      ghost_emitted_results first = ghost_emitted_results later ->
      ghost_exclusions first = ghost_exclusions later ->
      ownership_projection region_of runtime first =
      ownership_projection region_of runtime later.
  Proof.
    intros region_of runtime first later Hemitted Hexcluded.
    unfold ownership_projection; now rewrite Hemitted, Hexcluded.
  Qed.
End Abstraction.

(** Byte measures below must count allocated capacities, shared arena storage,
    private scratch, and continuation ownership when instantiated for Rust.
    The functions receive runtime fields only. The ledger is itself retained
    state; its counters do not replace a physical byte measure. *)
Record storage_measure
    (Node Residual Path Score Contract Snapshot Query Key Value Arena
     Reconstruction : Type) := {
  identity_bytes : session_identity Contract Snapshot Query -> nat;
  pending_bytes : list (occurrence Node Residual Path) -> nat;
  private_bytes : private_phase Node Residual Path Score Key -> nat;
  selected_bytes : selected_results Score -> nat;
  emission_bytes : emission_state -> nat;
  arena_bytes : Arena -> nat;
  cache_bytes : list (complete_cache_entry Contract Snapshot Query Key Value) -> nat;
  ledger_bytes : work_ledger -> nat;
  reconstruction_bytes : Reconstruction -> nat;
  status_bytes : session_status -> nat
}.

Definition runtime_storage_bytes
    {Node Residual Path Score Contract Snapshot Query Key Value Arena
     Reconstruction}
    (measure : storage_measure Node Residual Path Score Contract Snapshot
      Query Key Value Arena Reconstruction)
    (runtime : session_runtime Node Residual Path Score Contract Snapshot
      Query Key Value Arena Reconstruction) : nat :=
  identity_bytes measure (runtime_identity runtime) +
  pending_bytes measure (runtime_pending runtime) +
  private_bytes measure (runtime_private runtime) +
  selected_bytes measure (runtime_selected runtime) +
  emission_bytes measure (runtime_emission runtime) +
  arena_bytes measure (runtime_arena runtime) +
  cache_bytes measure (runtime_cache runtime) +
  ledger_bytes measure (runtime_ledger runtime) +
  reconstruction_bytes measure (runtime_reconstruction runtime) +
  status_bytes measure (runtime_status runtime).

Record proof_view
    (Node Residual Path Score Contract Snapshot Query Key Value Arena
     Reconstruction Evidence : Type) := {
  view_runtime : session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction;
  view_ghost : session_ghost Score Evidence
}.

Definition view_storage_bytes
    {Node Residual Path Score Contract Snapshot Query Key Value Arena
     Reconstruction Evidence}
    (measure : storage_measure Node Residual Path Score Contract Snapshot
      Query Key Value Arena Reconstruction)
    (view : proof_view Node Residual Path Score Contract Snapshot Query Key
      Value Arena Reconstruction Evidence) : nat :=
  runtime_storage_bytes measure (view_runtime view).

Theorem erased_ghost_storage_invariant :
  forall Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence
    (measure : storage_measure Node Residual Path Score Contract Snapshot
      Query Key Value Arena Reconstruction)
    (first later : proof_view Node Residual Path Score Contract Snapshot
      Query Key Value Arena Reconstruction Evidence),
    view_runtime first = view_runtime later ->
    view_storage_bytes measure first = view_storage_bytes measure later.
Proof.
  intros Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence measure first later Hruntime.
  unfold view_storage_bytes; now rewrite Hruntime.
Qed.

(** This finite control has two exact verifications in its proof history, but
    only the winning one occupies the concrete best-one store. The displaced
    original belongs to the ghost exclusion class. The example does not prove
    that any particular Rust heap comparison or exclusion is correct. *)
Definition compact_identity : session_identity unit unit unit :=
  {| session_contract := tt; session_snapshot := tt;
     session_query := tt; session_revision := 0 |}.

Definition empty_ledger : work_ledger :=
  {| work_executed := 0; work_reserved := 0; work_charged := 0;
     bytes_live := 0; bytes_peak := 0; allocation_count := 0 |}.

Definition compact_runtime :
    session_runtime unit unit unit nat unit unit unit unit unit unit unit :=
  {| runtime_identity := compact_identity;
     runtime_pending := [];
     runtime_private := NoPrivate;
     runtime_selected :=
       KnnSelected 1 [{| result_original := 0; result_score := 0 |}];
     runtime_emission :=
       {| committed_output_count := 0; next_page_index := 0 |};
     runtime_arena := tt;
     runtime_cache := [];
     runtime_ledger := empty_ledger;
     runtime_reconstruction := tt;
     runtime_status := Completed |}.

Definition compact_ghost : session_ghost nat unit :=
  {| ghost_verified_history :=
       [{| result_original := 0; result_score := 0 |};
        {| result_original := 1; result_score := 1 |}];
     ghost_emitted_results := [];
     ghost_exclusions :=
       [{| exclusion_original := 1; exclusion_evidence := tt |}] |}.

Example one_slot_heap_with_two_ghost_verifications :
  length (selected_entries (runtime_selected compact_runtime)) = 1 /\
  length (ghost_verified_history compact_ghost) = 2 /\
  session_abstraction (fun _ _ => []) [0; 1]
    (fun original score => score = original)
    (fun original _ => original = 1)
    (fun _ _ => True) (fun _ _ _ => True) (fun _ _ _ => True)
    compact_runtime compact_ghost.
Proof.
  split; [reflexivity |].
  split; [reflexivity |].
  unfold session_abstraction, exact_ownership, ownership_projection,
    borrowed_parent_live, selected_capacity, complete_cache_scoped.
  simpl.
  split.
  - split.
    + repeat constructor; simpl; intuition discriminate.
    + apply Permutation_refl.
  - split; [exact I |].
    split; [lia |].
    split; [constructor |].
    split; [constructor |].
    split; [constructor |].
    split; [constructor; [reflexivity | constructor] |].
    split; [repeat constructor |].
    split; [constructor; [simpl; left; reflexivity | constructor] |].
    split; [constructor; [reflexivity | constructor] |].
    split; [lia |].
    split; [lia |].
    split; [reflexivity |].
    intros _; split; reflexivity.
Qed.

End CompleteState.
