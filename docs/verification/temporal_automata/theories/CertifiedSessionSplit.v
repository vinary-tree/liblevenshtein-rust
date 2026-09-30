(** * Atomic split publication in a complete search session

    A borrowed split keeps its parent occurrence public while child and
    terminal work is prepared. Publication replaces that parent with pending
    terminal and child handles in one abstract step. Their snapshot-relative
    regions must match a package accepted by the complete partition checker.
    The theorem preserves original multiplicity and sound exclusions; it does
    not assert that the Rust edge iterator supplies these premises. *)

From Stdlib Require Import Arith Lia List Permutation.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRegionPartition CertifiedSearchSession.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Lemma pending_region_ids_app :
  forall Node Residual Path
    (region : occurrence Node Residual Path -> list nat) left right,
    pending_region_ids region (left ++ right) =
    pending_region_ids region left ++ pending_region_ids region right.
Proof.
  intros Node Residual Path region left.
  unfold pending_region_ids.
  induction left as [|first rest IH]; intro right; simpl.
  - reflexivity.
  - now rewrite IH, app_assoc.
Qed.

Lemma replace_middle_preserves_permutation :
  forall (before old replacement after : list nat),
    Permutation old replacement ->
    Permutation (before ++ old ++ after)
      (before ++ replacement ++ after).
Proof.
  intros before old replacement after Hreplace.
  apply Permutation_app; [reflexivity |].
  now apply Permutation_app.
Qed.

Section SplitPublication.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Let Runtime := session_runtime Node Residual Path Score Contract Snapshot
    Query Key Value Arena Reconstruction.
  Let Ghost := session_ghost Score Evidence.
  Let Occurrence := occurrence Node Residual Path.

  Definition publish_split (before : Runtime)
      (prefix terminal children suffix : list Occurrence) : Runtime :=
    {| runtime_identity := runtime_identity before;
       runtime_pending := prefix ++ terminal ++ children ++ suffix;
       runtime_private := NoPrivate;
       runtime_selected := runtime_selected before;
       runtime_emission := runtime_emission before;
       runtime_arena := runtime_arena before;
       runtime_cache := runtime_cache before;
       runtime_ledger := runtime_ledger before;
       runtime_reconstruction := runtime_reconstruction before;
       runtime_status := runtime_status before |}.

  Lemma split_publication_keeps_snapshot_identity :
    forall before prefix terminal children suffix,
      runtime_identity
        (publish_split before prefix terminal children suffix) =
      runtime_identity before.
  Proof. reflexivity. Qed.

  Lemma accepted_split_replaces_pending_region :
    forall (region : Occurrence -> list nat)
      prefix parent suffix terminal children (package : split_package),
      pending_region_ids region terminal = terminal_originals package ->
      map region children = child_original_regions package ->
      accept_split Nat.eq_dec (region parent) package = true ->
      Permutation
        (pending_region_ids region (prefix ++ parent :: suffix))
        (pending_region_ids region
          (prefix ++ terminal ++ children ++ suffix)).
  Proof.
    intros region prefix parent suffix terminal children package
      Hterminal Hchildren Haccept.
    apply accepted_split_has_exact_original_coverage in Haccept.
    assert (Hchild_regions : pending_region_ids region children =
        concat (child_original_regions package)).
    { unfold pending_region_ids; exact (f_equal (@concat nat) Hchildren). }
    repeat rewrite pending_region_ids_app.
    simpl.
    rewrite Hterminal, Hchild_regions.
    change (Permutation
      (pending_region_ids region prefix ++ region parent ++
        pending_region_ids region suffix)
      (pending_region_ids region prefix ++
        terminal_originals package ++
        concat (child_original_regions package) ++
        pending_region_ids region suffix)).
    replace
      (pending_region_ids region prefix ++
       terminal_originals package ++
       concat (child_original_regions package) ++
       pending_region_ids region suffix)
      with
      (pending_region_ids region prefix ++ flattened_split package ++
       pending_region_ids region suffix).
    - now apply replace_middle_preserves_permutation.
    - unfold flattened_split; repeat rewrite app_assoc; reflexivity.
  Qed.

  Theorem split_publication_preserves_exact_ownership :
    forall (region_of : session_identity Contract Snapshot Query ->
        Occurrence -> list nat)
      universe (before : Runtime) (ghost : Ghost)
      prefix parent suffix terminal children (package : split_package),
      exact_ownership universe
        (ownership_projection region_of before ghost) ->
      runtime_pending before = prefix ++ parent :: suffix ->
      runtime_private before = BorrowedSplit (length prefix) children ->
      pending_region_ids (region_of (runtime_identity before)) terminal =
        terminal_originals package ->
      map (region_of (runtime_identity before)) children =
        child_original_regions package ->
      accept_split Nat.eq_dec
        (region_of (runtime_identity before) parent) package = true ->
      exact_ownership universe
        (ownership_projection region_of
          (publish_split before prefix terminal children suffix) ghost).
  Proof.
    intros region_of universe before ghost prefix parent suffix terminal
      children package [Hnodup Howned] Hpending Hprivate
      Hterminal Hchildren Haccept.
    split; [exact Hnodup |].
    pose proof (accepted_split_replaces_pending_region
      (region_of (runtime_identity before)) prefix parent suffix terminal
      children package Hterminal Hchildren Haccept)
      as Hpending_regions.
    unfold owned_originals, ownership_projection in *.
    unfold publish_split; simpl.
    rewrite Hpending in Howned.
    rewrite Hprivate in Howned; simpl in Howned.
    repeat rewrite app_nil_r in Howned.
    eapply Permutation_trans; [|exact Howned].
    repeat rewrite app_nil_r.
    apply Permutation_app; [|reflexivity].
    now apply Permutation_sym.
  Qed.

  Theorem split_publication_preserves_sound_exclusions :
    forall winner (region_of : session_identity Contract Snapshot Query ->
        Occurrence -> list nat)
      (before : Runtime) (ghost : Ghost) prefix terminal children suffix,
      exclusions_sound winner (ownership_projection region_of before ghost) ->
      exclusions_sound winner
        (ownership_projection region_of
          (publish_split before prefix terminal children suffix) ghost).
  Proof.
    intros winner region_of before ghost prefix terminal children suffix
      Hsound.
    unfold exclusions_sound, ownership_projection in *.
    unfold publish_split; simpl in *.
    exact Hsound.
  Qed.
End SplitPublication.

(** Both child handles can name the same physical node. Their paths remain
    distinct, and the snapshot-relative region interpretation assigns each
    original once. The interpretation itself is an external premise. *)
Definition shared_node_split_occurrence (path : list nat) :
    occurrence nat nat (list nat) :=
  {| occurrence_node := 7;
     occurrence_residual := 0;
     occurrence_path := path;
     occurrence_cursor := 0 |}.

Definition shared_node_split_region
    (handle : occurrence nat nat (list nat)) : list nat :=
  match occurrence_path handle with
  | [] => [0; 1]
  | [0] => [0]
  | [1] => [1]
  | _ => []
  end.

Example split_publication_retains_distinct_paths_at_shared_node :
  occurrence_node (shared_node_split_occurrence [0]) =
    occurrence_node (shared_node_split_occurrence [1]) /\
  occurrence_path (shared_node_split_occurrence [0]) <>
    occurrence_path (shared_node_split_occurrence [1]) /\
  Permutation
    (pending_region_ids shared_node_split_region
      [shared_node_split_occurrence []])
    (pending_region_ids shared_node_split_region
      [shared_node_split_occurrence [0]; shared_node_split_occurrence [1]]).
Proof.
  split; [reflexivity |].
  split; [discriminate |].
  change (Permutation
    (pending_region_ids shared_node_split_region
      ([] ++ shared_node_split_occurrence [] :: []))
    (pending_region_ids shared_node_split_region
      ([] ++ [] ++
        [shared_node_split_occurrence [0]; shared_node_split_occurrence [1]]
        ++ []))).
  apply accepted_split_replaces_pending_region with
    (package := {| terminal_originals := [];
                   child_original_regions := [[0]; [1]] |});
    reflexivity.
Qed.

(** The existing partition checker rejects each malformed package. These
    controls are attached here to the publication theorem's acceptance gate. *)
Example split_omitting_live_terminal_is_rejected :
  @accept_split nat Nat.eq_dec [0; 1]
    {| terminal_originals := [];
       child_original_regions := [[1]] |} = false.
Proof. reflexivity. Qed.

Example split_omitting_collision_member_is_rejected :
  @accept_split nat Nat.eq_dec [0; 1]
    {| terminal_originals := [0];
       child_original_regions := [] |} = false.
Proof. reflexivity. Qed.

Example split_assigning_one_original_twice_is_rejected :
  @accept_split nat Nat.eq_dec [0; 1]
    {| terminal_originals := [0];
       child_original_regions := [[0; 1]] |} = false.
Proof. reflexivity. Qed.
