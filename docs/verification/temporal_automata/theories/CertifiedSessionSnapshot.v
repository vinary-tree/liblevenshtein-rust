(** * Captured-revision reads and path-sensitive original identity

    The finite catalog retains immutable snapshot images.  Every accepted
    bucket, summary, cache, or exact-payload observation is recomputed from
    the session's captured image and includes its revision, path, slot, and
    original identity.  Publishing a newer image cannot alter the older one.
    A Rust correspondence must still bind these images and observations to
    actual dictionary roots, bucket storage, and immutable borrows. *)

From Stdlib Require Import Arith Bool Lia List.
Import ListNotations.

Record occurrence_key := {
  key_path : list nat;
  key_slot : nat
}.

Definition occurrence_key_eq_dec (left right : occurrence_key) :
    {left = right} + {left <> right}.
Proof.
  decide equality; try apply Nat.eq_dec.
  apply list_eq_dec; apply Nat.eq_dec.
Defined.

Record snapshot_row := {
  row_key : occurrence_key;
  row_node : nat;
  row_original : nat;
  row_payload : nat;
  row_summary : nat;
  row_cache : nat
}.

Record snapshot_image := {
  image_revision : nat;
  image_rows : list snapshot_row
}.

Definition snapshot_well_formed (image : snapshot_image) : Prop :=
  NoDup (map row_key (image_rows image)) /\
  NoDup (map row_original (image_rows image)).

Fixpoint lookup_row (key : occurrence_key) (rows : list snapshot_row)
    : option snapshot_row :=
  match rows with
  | [] => None
  | row :: rest =>
      if occurrence_key_eq_dec key (row_key row) then Some row
      else lookup_row key rest
  end.

Fixpoint lookup_image (revision : nat) (catalog : list snapshot_image)
    : option snapshot_image :=
  match catalog with
  | [] => None
  | image :: rest =>
      if Nat.eqb revision (image_revision image) then Some image
      else lookup_image revision rest
  end.

Lemma lookup_image_append_other :
  forall catalog captured newer,
    image_revision newer <> captured ->
    lookup_image captured (catalog ++ [newer]) =
      lookup_image captured catalog.
Proof.
  induction catalog as [|head rest IH]; intros captured newer Hother; simpl.
  - apply Nat.eqb_neq in Hother. now rewrite Nat.eqb_sym, Hother.
  - destruct (Nat.eqb captured (image_revision head)); auto.
Qed.

Lemma nodup_append_fresh : forall (values : list nat) value,
  NoDup values -> ~ In value values -> NoDup (values ++ [value]).
Proof.
  intros values value Hnodup.
  induction Hnodup as [|head rest Hnot Htail IH]; intro Hfresh; simpl.
  - constructor; [intro H; inversion H | constructor].
  - constructor.
    + intro H; apply in_app_or in H as [Hin | Hin].
      * exact (Hnot Hin).
      * simpl in Hin; destruct Hin as [Heq | []].
        apply Hfresh; left; symmetry; exact Heq.
    + apply IH. intro Hin; apply Hfresh; right; exact Hin.
Qed.

Inductive read_kind := BucketRead | SummaryRead | CacheRead | PayloadRead.

Definition row_value (kind : read_kind) (row : snapshot_row) : nat :=
  match kind with
  | BucketRead => row_original row
  | SummaryRead => row_summary row
  | CacheRead => row_cache row
  | PayloadRead => row_payload row
  end.

Record snapshot_observation := {
  observed_revision : nat;
  observed_key : occurrence_key;
  observed_kind : read_kind;
  observed_original : nat;
  observed_value : nat
}.

Definition make_observation (image : snapshot_image) (kind : read_kind)
    (key : occurrence_key) : option snapshot_observation :=
  match lookup_row key (image_rows image) with
  | None => None
  | Some row => Some
      {| observed_revision := image_revision image;
         observed_key := key;
         observed_kind := kind;
         observed_original := row_original row;
         observed_value := row_value kind row |}
  end.

Definition accept_observation (image : snapshot_image)
    (observation : snapshot_observation) : bool :=
  Nat.eqb (observed_revision observation) (image_revision image) &&
  match lookup_row (observed_key observation) (image_rows image) with
  | None => false
  | Some row =>
      Nat.eqb (observed_original observation) (row_original row) &&
      Nat.eqb (observed_value observation)
        (row_value (observed_kind observation) row)
  end.

Theorem accepted_observation_has_captured_referent :
  forall image observation,
    accept_observation image observation = true ->
    observed_revision observation = image_revision image /\
    exists row,
      lookup_row (observed_key observation) (image_rows image) = Some row /\
      observed_original observation = row_original row /\
      observed_value observation = row_value (observed_kind observation) row.
Proof.
  intros image observation Haccepted.
  unfold accept_observation in Haccepted.
  apply andb_true_iff in Haccepted as [Hrevision Hrow].
  apply Nat.eqb_eq in Hrevision.
  split; [exact Hrevision |].
  destruct (lookup_row (observed_key observation) (image_rows image))
    as [row |] eqn:Hlookup; [|discriminate].
  apply andb_true_iff in Hrow as [Horiginal Hvalue].
  apply Nat.eqb_eq in Horiginal.
  apply Nat.eqb_eq in Hvalue.
  exists row; repeat split; assumption.
Qed.

Theorem constructed_observation_is_accepted :
  forall image kind key observation,
    make_observation image kind key = Some observation ->
    accept_observation image observation = true.
Proof.
  intros image kind key observation Hmake.
  unfold make_observation in Hmake.
  destruct (lookup_row key (image_rows image)) as [row |] eqn:Hlookup;
    [|discriminate].
  inversion Hmake; subst; unfold accept_observation; simpl.
  rewrite Nat.eqb_refl, Hlookup; simpl.
  now rewrite Nat.eqb_refl, Nat.eqb_refl.
Qed.

Record snapshot_session := {
  captured_image : snapshot_image;
  available_images : list snapshot_image;
  pending_keys : list occurrence_key;
  observed_reads : list snapshot_observation
}.

Definition key_in_image (image : snapshot_image)
    (key : occurrence_key) : Prop :=
  exists row, lookup_row key (image_rows image) = Some row.

Definition snapshot_invariant (session : snapshot_session) : Prop :=
  snapshot_well_formed (captured_image session) /\
  NoDup (map image_revision (available_images session)) /\
  lookup_image (image_revision (captured_image session))
    (available_images session) = Some (captured_image session) /\
  Forall (key_in_image (captured_image session)) (pending_keys session) /\
  Forall (fun observation =>
    accept_observation (captured_image session) observation = true)
    (observed_reads session).

Definition initial_session (image : snapshot_image)
    (pending : list occurrence_key) : snapshot_session :=
  {| captured_image := image;
     available_images := [image];
     pending_keys := pending;
     observed_reads := [] |}.

Theorem initial_snapshot_invariant : forall image pending,
  snapshot_well_formed image ->
  Forall (key_in_image image) pending ->
  snapshot_invariant (initial_session image pending).
Proof.
  intros image pending Hwell Hpending.
  destruct Hwell as [Hkeys Horiginals].
  unfold snapshot_invariant, initial_session, snapshot_well_formed; simpl.
  repeat split.
  - exact Hkeys.
  - exact Horiginals.
  - constructor; [intro H; inversion H | constructor].
  - now rewrite Nat.eqb_refl.
  - exact Hpending.
  - constructor.
Qed.

Definition record_read (session : snapshot_session)
    (observation : snapshot_observation) : snapshot_session :=
  {| captured_image := captured_image session;
     available_images := available_images session;
     pending_keys := pending_keys session;
     observed_reads := observation :: observed_reads session |}.

Definition publish_new_image (session : snapshot_session)
    (newer : snapshot_image) : snapshot_session :=
  {| captured_image := captured_image session;
     available_images := available_images session ++ [newer];
     pending_keys := pending_keys session;
     observed_reads := observed_reads session |}.

Definition update_pending (session : snapshot_session)
    (remaining : list occurrence_key) : snapshot_session :=
  {| captured_image := captured_image session;
     available_images := available_images session;
     pending_keys := remaining;
     observed_reads := observed_reads session |}.

Inductive snapshot_step : snapshot_session -> snapshot_session -> Prop :=
| ReadCaptured : forall session kind key observation,
    make_observation (captured_image session) kind key = Some observation ->
    snapshot_step session (record_read session observation)
| PublishNew : forall session newer,
    image_revision newer <> image_revision (captured_image session) ->
    ~ In (image_revision newer)
      (map image_revision (available_images session)) ->
    snapshot_step session (publish_new_image session newer)
| ChangePending : forall session remaining,
    Forall (key_in_image (captured_image session)) remaining ->
    snapshot_step session (update_pending session remaining).

Theorem snapshot_step_preserves_invariant : forall first last,
  snapshot_step first last ->
  snapshot_invariant first -> snapshot_invariant last.
Proof.
  intros first last Hstep Hinv.
  destruct Hinv as [Hwell [Hcatalog [Hcaptured [Hpending Hreads]]]].
  destruct Hwell as [Hkeys Horiginals].
  inversion Hstep; subst; unfold snapshot_invariant,
    snapshot_well_formed, record_read, publish_new_image, update_pending;
    simpl.
  - repeat split; try assumption.
    constructor; [eapply constructed_observation_is_accepted; eauto |
      exact Hreads].
  - rewrite map_app; simpl.
    repeat split; try assumption.
    + apply nodup_append_fresh; assumption.
    + rewrite lookup_image_append_other; assumption.
  - repeat split; assumption.
Qed.

Inductive snapshot_steps : snapshot_session -> snapshot_session -> Prop :=
| SnapshotDone : forall session, snapshot_steps session session
| SnapshotMore : forall first middle last,
    snapshot_step first middle -> snapshot_steps middle last ->
    snapshot_steps first last.

Lemma snapshot_step_keeps_captured_image : forall first last,
  snapshot_step first last ->
  captured_image last = captured_image first.
Proof.
  intros first last Hstep; inversion Hstep; reflexivity.
Qed.

Theorem snapshot_steps_keep_captured_image : forall first last,
  snapshot_steps first last ->
  captured_image last = captured_image first.
Proof.
  intros first last Hsteps.
  induction Hsteps as [session | first middle last Hstep Hsteps IH].
  - reflexivity.
  - rewrite IH. now apply snapshot_step_keeps_captured_image.
Qed.

Theorem reachable_sessions_keep_captured_revision : forall first last,
  snapshot_invariant first -> snapshot_steps first last ->
  snapshot_invariant last /\
  Forall (fun observation =>
    observed_revision observation =
      image_revision (captured_image last)) (observed_reads last).
Proof.
  intros first last Hinv Hsteps.
  induction Hsteps as [session | first middle last Hstep Hsteps IH].
  - split; [exact Hinv |].
    destruct Hinv as [_ [_ [_ [_ Hreads]]]].
    induction Hreads; constructor; auto.
    apply accepted_observation_has_captured_referent in H.
    exact (proj1 H).
  - apply IH.
    eapply snapshot_step_preserves_invariant; eauto.
Qed.

Theorem reachable_sessions_keep_initial_revision : forall first last,
  snapshot_invariant first -> snapshot_steps first last ->
  captured_image last = captured_image first /\
  Forall (fun observation =>
    observed_revision observation = image_revision (captured_image first))
    (observed_reads last).
Proof.
  intros first last Hinv Hsteps.
  pose proof (snapshot_steps_keep_captured_image first last Hsteps)
    as Hcaptured.
  pose proof (reachable_sessions_keep_captured_revision first last Hinv Hsteps)
    as [_ Hreads].
  rewrite Hcaptured in Hreads.
  split; assumption.
Qed.

Module SnapshotControls.
  Definition left_key : occurrence_key :=
    {| key_path := [0]; key_slot := 0 |}.
  Definition right_key : occurrence_key :=
    {| key_path := [1]; key_slot := 0 |}.

  Definition old_left : snapshot_row :=
    {| row_key := left_key; row_node := 7; row_original := 0;
       row_payload := 11; row_summary := 2; row_cache := 5 |}.
  Definition old_right : snapshot_row :=
    {| row_key := right_key; row_node := 7; row_original := 1;
       row_payload := 12; row_summary := 3; row_cache := 6 |}.
  Definition new_left : snapshot_row :=
    {| row_key := left_key; row_node := 7; row_original := 2;
       row_payload := 99; row_summary := 9; row_cache := 19 |}.

  Definition old_image : snapshot_image :=
    {| image_revision := 0; image_rows := [old_left; old_right] |}.
  Definition new_image : snapshot_image :=
    {| image_revision := 1; image_rows := [new_left] |}.

  Example shared_node_does_not_identify_originals :
    row_node old_left = row_node old_right /\
    row_key old_left <> row_key old_right /\
    row_original old_left <> row_original old_right /\
    lookup_row left_key (image_rows old_image) = Some old_left /\
    lookup_row right_key (image_rows old_image) = Some old_right.
  Proof. vm_compute; repeat split; congruence. Qed.

  Lemma old_image_well_formed : snapshot_well_formed old_image.
  Proof.
    unfold snapshot_well_formed, old_image; simpl; split;
      repeat constructor; simpl; intuition discriminate.
  Qed.

  Definition initial : snapshot_session :=
    initial_session old_image [left_key; right_key].

  Lemma initial_valid : snapshot_invariant initial.
  Proof.
    unfold initial.
    apply initial_snapshot_invariant; [apply old_image_well_formed |].
    repeat constructor; unfold key_in_image; eexists; reflexivity.
  Qed.

  Definition old_summary : snapshot_observation :=
    {| observed_revision := 0; observed_key := left_key;
       observed_kind := SummaryRead; observed_original := 0;
       observed_value := 2 |}.
  Definition old_payload : snapshot_observation :=
    {| observed_revision := 0; observed_key := right_key;
       observed_kind := PayloadRead; observed_original := 1;
       observed_value := 12 |}.
  Definition new_bucket : snapshot_observation :=
    {| observed_revision := 1; observed_key := left_key;
       observed_kind := BucketRead; observed_original := 2;
       observed_value := 2 |}.
  Definition falsely_retagged_bucket : snapshot_observation :=
    {| observed_revision := 0; observed_key := left_key;
       observed_kind := BucketRead; observed_original := 2;
       observed_value := 2 |}.
  Definition old_cache : snapshot_observation :=
    {| observed_revision := 0; observed_key := left_key;
       observed_kind := CacheRead; observed_original := 0;
       observed_value := 5 |}.

  Example old_read_survives_newer_revision :
    lookup_image 0 [old_image; new_image] = Some old_image /\
    accept_observation old_image old_payload = true /\
    accept_observation old_image old_cache = true.
  Proof. repeat split; reflexivity. Qed.

  Example new_bucket_with_old_summary_or_cache_is_rejected :
    accept_observation old_image new_bucket = false /\
    accept_observation old_image old_summary = true /\
    accept_observation old_image old_cache = true /\
    ~ Forall (fun observation =>
        accept_observation old_image observation = true)
        [new_bucket; old_summary; old_cache].
  Proof.
    repeat split; try reflexivity.
    intro Hvalid.
    apply Forall_forall with (x := new_bucket) in Hvalid;
      [discriminate Hvalid | simpl; auto].
  Qed.

  Example retagging_new_payload_as_old_revision_is_rejected :
    accept_observation old_image falsely_retagged_bucket = false.
  Proof. reflexivity. Qed.

  Definition after_summary := record_read initial old_summary.
  Definition after_new := publish_new_image after_summary new_image.
  Definition final := record_read after_new old_payload.

  Example old_session_reads_after_new_publication :
    snapshot_steps initial final.
  Proof.
    unfold final, after_new, after_summary.
    eapply SnapshotMore.
    - apply ReadCaptured with (kind := SummaryRead) (key := left_key).
      reflexivity.
    - eapply SnapshotMore.
      + apply (PublishNew _ new_image); simpl; [lia |].
        simpl; intuition lia.
      + eapply SnapshotMore.
        * apply ReadCaptured with (kind := PayloadRead) (key := right_key).
          reflexivity.
        * constructor.
  Qed.

  Example reachable_old_session_never_uses_new_revision :
    snapshot_invariant final /\
    Forall (fun observation => observed_revision observation = 0)
      (observed_reads final).
  Proof.
    pose proof (reachable_sessions_keep_captured_revision
      initial final initial_valid old_session_reads_after_new_publication)
      as [Hvalid Hreads].
    split; [exact Hvalid | exact Hreads].
  Qed.
End SnapshotControls.
