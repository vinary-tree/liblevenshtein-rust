(** * Candidate ownership in a finite search session

    Each natural number names one original occurrence, even when dictionary
    nodes are shared. A private work item takes ownership before processing;
    publication moves it atomically to verified or soundly excluded. These
    generic laws do not establish the Rust scheduler or a concrete rank bound.
    In particular, a complete result theorem needs a separate proof that all
    pending/private work has been discharged and that exclusions are sound. *)

From Stdlib Require Import Arith List Permutation.
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
