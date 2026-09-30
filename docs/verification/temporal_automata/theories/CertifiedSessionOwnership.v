(** * Initial ownership for complete CBC sessions

    A snapshot-relative region interpretation assigns original occurrences to
    compact pending handles. The root must cover the snapshot's original
    universe exactly once. Private scoring takes exclusive ownership after a
    cursor advances; staged split children borrow the still-public parent.
    Node handles may be shared and therefore cannot be used as original IDs.
    These theorems do not identify the interpretation with a Rust dictionary. *)

From Stdlib Require Import Arith Lia List Permutation.
From Liblevenshtein.TemporalAutomata Require Import CertifiedSearchSession.
Import ListNotations.
Import CompleteState.
Set Implicit Arguments.

Section InitialSession.
  Context {Node Residual Path Score Contract Snapshot Query Key Value Arena
    Reconstruction Evidence : Type}.

  Variable identity : session_identity Contract Snapshot Query.
  Variable root : occurrence Node Residual Path.
  Variable arena : Arena.
  Variable reconstruction : Reconstruction.
  Variable empty_selected : selected_results Score.

  Definition initialized_runtime :
      session_runtime Node Residual Path Score Contract Snapshot Query Key
        Value Arena Reconstruction :=
    {| runtime_identity := identity;
       runtime_pending := [root];
       runtime_private := NoPrivate;
       runtime_selected := empty_selected;
       runtime_emission :=
         {| committed_output_count := 0; next_page_index := 0 |};
       runtime_arena := arena;
       runtime_cache := [];
       runtime_ledger := empty_ledger;
       runtime_reconstruction := reconstruction;
       runtime_status := Active |}.

  Definition initialized_ghost : session_ghost Score Evidence :=
    {| ghost_verified_history := [];
       ghost_emitted_results := [];
       ghost_exclusions := [] |}.

  Theorem initialized_session_has_exact_ownership :
    forall region_of universe,
      NoDup universe ->
      region_of identity root = universe ->
      selected_entries empty_selected = [] ->
      exact_ownership universe
        (ownership_projection region_of initialized_runtime
          initialized_ghost).
  Proof.
    intros region_of universe Hnodup Hregion Hselected.
    unfold exact_ownership, ownership_projection, initialized_runtime,
      initialized_ghost, owned_originals, pending_region_ids, selected_ids.
    simpl. rewrite Hregion, Hselected. simpl.
    repeat rewrite app_nil_r.
    split; [exact Hnodup | apply Permutation_refl].
  Qed.

  Theorem initialized_session_satisfies_abstraction :
    forall region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid,
      NoDup universe ->
      region_of identity root = universe ->
      selected_entries empty_selected = [] ->
      (match empty_selected with
       | RangeSelected _ => True
       | KnnSelected capacity entries => length entries <= capacity
       end) ->
      arena_valid arena (occurrence_residual root) ->
      Forall (reconstruction_valid reconstruction (occurrence_path root))
        universe ->
      session_abstraction region_of universe authoritative evidence_sound
        arena_valid cache_valid reconstruction_valid
        initialized_runtime initialized_ghost.
  Proof.
    intros region_of universe authoritative evidence_sound arena_valid
      cache_valid reconstruction_valid Hnodup Hregion Hempty Hcapacity
      Harena Hreconstruction.
    unfold session_abstraction.
    split.
    - eapply initialized_session_has_exact_ownership; eauto.
    - unfold borrowed_parent_live, selected_capacity, complete_cache_scoped,
        initialized_runtime, initialized_ghost; simpl.
      rewrite Hempty.
      split; [exact I |].
      split; [exact Hcapacity |].
      split; [constructor |].
      split.
      + constructor.
        * split.
          -- exact Harena.
          -- rewrite Hregion; exact Hreconstruction.
        * constructor.
      + split; [constructor |].
        split; [constructor |].
        split; [constructor |].
        split; [constructor |].
        split; [constructor |].
        split; [lia |].
        split; [lia |].
        split; [reflexivity |].
        intros Hcompleted; discriminate.
  Qed.
End InitialSession.

(** Two pending occurrences use the same physical node handle. Their path and
    cursor contexts identify distinct originals, and the region interpretation
    keeps both. Collapsing them to one node would lose an original. *)
Definition shared_occurrence (cursor : nat) : occurrence nat unit nat :=
  {| occurrence_node := 7;
     occurrence_residual := tt;
     occurrence_path := cursor;
     occurrence_cursor := cursor |}.

Definition shared_region (_ : session_identity unit unit unit)
    (item : occurrence nat unit nat) : list nat :=
  if Nat.eqb (occurrence_cursor item) 0 then [0] else [1].

Definition shared_identity : session_identity unit unit unit :=
  {| session_contract := tt; session_snapshot := tt;
     session_query := tt; session_revision := 0 |}.

Example shared_node_has_two_originals :
  occurrence_node (shared_occurrence 0) =
    occurrence_node (shared_occurrence 1) /\
  pending_region_ids (shared_region shared_identity)
    [shared_occurrence 0; shared_occurrence 1] = [0; 1] /\
  NoDup (pending_region_ids (shared_region shared_identity)
    [shared_occurrence 0; shared_occurrence 1]).
Proof.
  simpl; repeat split; try reflexivity.
  repeat constructor; simpl; intuition discriminate.
Qed.

Example coalescing_shared_node_loses_an_original :
  ~ Permutation
    (pending_region_ids (shared_region shared_identity)
      [shared_occurrence 0])
    (pending_region_ids (shared_region shared_identity)
      [shared_occurrence 0; shared_occurrence 1]).
Proof.
  intro Hperm.
  apply Permutation_length in Hperm; simpl in Hperm; discriminate.
Qed.

(** A collision bucket can also have the same node and the same path for two
    different slots. The slot cursor, not a node or path key, distinguishes
    their original identities. *)
Definition collision_occurrence (slot : nat) : occurrence nat unit nat :=
  {| occurrence_node := 7;
     occurrence_residual := tt;
     occurrence_path := 0;
     occurrence_cursor := slot |}.

Example collision_slots_remain_distinct :
  occurrence_node (collision_occurrence 0) =
    occurrence_node (collision_occurrence 1) /\
  occurrence_path (collision_occurrence 0) =
    occurrence_path (collision_occurrence 1) /\
  pending_region_ids (shared_region shared_identity)
    [collision_occurrence 0; collision_occurrence 1] = [0; 1].
Proof. simpl; repeat split; reflexivity. Qed.

(** A cursor may remove an original from every public pending region before
    its exact score is committed. Private ownership is then indispensable. *)
Definition private_runtime :
    session_runtime unit unit unit nat unit unit unit unit unit unit unit :=
  {| runtime_identity := compact_identity;
     runtime_pending := [];
     runtime_private := ScoringCandidate 0;
     runtime_selected := RangeSelected [];
     runtime_emission :=
       {| committed_output_count := 0; next_page_index := 0 |};
     runtime_arena := tt;
     runtime_cache := [];
     runtime_ledger := empty_ledger;
     runtime_reconstruction := tt;
     runtime_status := Active |}.

Definition private_ghost : session_ghost nat unit :=
  {| ghost_verified_history := [];
     ghost_emitted_results := [];
     ghost_exclusions := [] |}.

Example omitting_exclusive_private_loses_an_original :
  let owned := ownership_projection (fun _ _ => [])
    private_runtime private_ghost in
  exact_ownership [0] owned /\
  public_only_originals owned = [] /\
  ~ Permutation (public_only_originals owned) [0].
Proof.
  simpl.
  split.
  - split.
    + repeat constructor; simpl; intuition discriminate.
    + apply Permutation_refl.
  - split; [reflexivity |].
    intro Hperm.
    apply Permutation_length in Hperm; simpl in Hperm; discriminate.
Qed.
