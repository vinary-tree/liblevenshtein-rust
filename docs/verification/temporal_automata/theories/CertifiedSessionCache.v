(** * Complete transition caches and append-only residual arenas

    This is a conditional operational model of a query-local cache. A
    fingerprint is only a lookup hint. Cache entries bind a full scope,
    source-state identifier, and label, and hold only completed semantic
    answers. Allocation failure lives outside that answer type. The model
    counts logical cells; byte, reserve, and source-code correspondence are
    separate obligations. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.
Set Implicit Arguments.

Inductive completion := Dead | Live (target_id : nat).

Inductive transition_outcome :=
| Finished (result : completion)
| ResourceFailure.

Record transition_report := {
  report_outcome : transition_outcome;
  report_charged_work : nat;
  report_allocations : nat
}.

Definition same_semantic_outcome
    (first second : transition_report) : Prop :=
  report_outcome first = report_outcome second.

Example equal_semantic_answers_need_not_equal_resource_reports :
  same_semantic_outcome
    {| report_outcome := Finished (Live 0);
       report_charged_work := 1;
       report_allocations := 0 |}
    {| report_outcome := Finished (Live 0);
       report_charged_work := 4;
       report_allocations := 1 |} /\
  {| report_outcome := Finished (Live 0);
     report_charged_work := 1;
     report_allocations := 0 |} <>
  {| report_outcome := Finished (Live 0);
     report_charged_work := 4;
     report_allocations := 1 |}.
Proof. split; [reflexivity | discriminate]. Qed.

Record cached_transition (Scope Label : Type) := {
  cached_scope : Scope;
  cached_source : nat;
  cached_label : Label;
  cached_answer : completion
}.

Record published_successor (Scope Label Cursor Witness : Type) := {
  publication_scope : Scope;
  publication_source : nat;
  publication_label : Label;
  publication_target : nat;
  publication_cursor : Cursor;
  publication_witness : Witness
}.

Record cache_world (Scope Label Fingerprint Residual Cursor Witness : Type) := {
  world_arena : list Residual;
  world_hints : list (Fingerprint * nat);
  world_cache : list (cached_transition Scope Label);
  world_publications : list (published_successor Scope Label Cursor Witness)
}.

Section CacheModel.
  Context {Scope Label Fingerprint Residual Cursor Witness : Type}.

  Variable scope_eq_dec : forall left right : Scope,
    {left = right} + {left <> right}.
  Variable label_eq_dec : forall left right : Label,
    {left = right} + {left <> right}.
  Variable fingerprint_eq_dec : forall left right : Fingerprint,
    {left = right} + {left <> right}.
  Variable residual_eq_dec : forall left right : Residual,
    {left = right} + {left <> right}.
  Variable fingerprint_of : Residual -> Fingerprint.
  Variable semantic_step : Scope -> Residual -> Label -> option Residual.
  Variable cursor_valid : Scope -> Residual -> Cursor -> Prop.
  Variable witness_valid : Scope -> Residual -> Witness -> Prop.

  Let Entry := cached_transition Scope Label.
  Let Publication := published_successor Scope Label Cursor Witness.
  Let World := cache_world Scope Label Fingerprint Residual Cursor Witness.

  Definition answer_valid (arena : list Residual) (scope : Scope)
      (source : Residual) (label : Label) (answer : completion) : Prop :=
    match answer with
    | Dead => semantic_step scope source label = None
    | Live target_id =>
        exists target,
          nth_error arena target_id = Some target /\
          semantic_step scope source label = Some target
    end.

  Definition entry_valid (arena : list Residual) (entry : Entry) : Prop :=
    exists source,
      nth_error arena (cached_source entry) = Some source /\
      answer_valid arena (cached_scope entry) source
        (cached_label entry) (cached_answer entry).

  Definition hint_valid (arena : list Residual)
      (hint : Fingerprint * nat) : Prop :=
    exists state,
      nth_error arena (snd hint) = Some state /\
      fst hint = fingerprint_of state.

  Definition publication_valid (arena : list Residual)
      (publication : Publication) : Prop :=
    exists source target,
      nth_error arena (publication_source publication) = Some source /\
      nth_error arena (publication_target publication) = Some target /\
      semantic_step (publication_scope publication) source
        (publication_label publication) = Some target /\
      cursor_valid (publication_scope publication) target
        (publication_cursor publication) /\
      witness_valid (publication_scope publication) target
        (publication_witness publication).

  Definition world_valid (world : World) : Prop :=
    Forall (hint_valid (world_arena world)) (world_hints world) /\
    Forall (entry_valid (world_arena world)) (world_cache world) /\
    Forall (publication_valid (world_arena world))
      (world_publications world).

  Fixpoint lookup_cache (scope : Scope) (source_id : nat)
      (label : Label) (entries : list Entry) : option completion :=
    match entries with
    | [] => None
    | entry :: rest =>
        if scope_eq_dec scope (cached_scope entry) then
          if Nat.eq_dec source_id (cached_source entry) then
            if label_eq_dec label (cached_label entry) then
              Some (cached_answer entry)
            else lookup_cache scope source_id label rest
          else lookup_cache scope source_id label rest
        else lookup_cache scope source_id label rest
    end.

  Lemma lookup_cache_witness :
    forall scope source_id label entries answer,
      lookup_cache scope source_id label entries = Some answer ->
      exists entry,
        In entry entries /\
        cached_scope entry = scope /\
        cached_source entry = source_id /\
        cached_label entry = label /\
        cached_answer entry = answer.
  Proof.
    intros scope source_id label entries.
    induction entries as [|entry rest IH]; intros answer Hlookup;
      simpl in Hlookup; [discriminate |].
    destruct (scope_eq_dec scope (cached_scope entry)) as [Hscope | Hscope].
    - destruct (Nat.eq_dec source_id (cached_source entry))
        as [Hsource | Hsource].
      + destruct (label_eq_dec label (cached_label entry))
          as [Hlabel | Hlabel].
        * inversion Hlookup; subst answer.
          exists entry; split; [now left |].
          repeat split; congruence.
        * destruct (IH answer Hlookup)
            as [found [Hin [Hfound_scope [Hfound_source
              [Hfound_label Hfound_answer]]]]].
          exists found; split; [now right |].
          repeat split; assumption.
      + destruct (IH answer Hlookup)
          as [found [Hin [Hfound_scope [Hfound_source
            [Hfound_label Hfound_answer]]]]].
        exists found; split; [now right |].
        repeat split; assumption.
    - destruct (IH answer Hlookup)
        as [found [Hin [Hfound_scope [Hfound_source
          [Hfound_label Hfound_answer]]]]].
      exists found; split; [now right |].
      repeat split; assumption.
  Qed.

  Theorem cache_hit_is_semantically_complete :
    forall world scope source_id source label answer,
      world_valid world ->
      nth_error (world_arena world) source_id = Some source ->
      lookup_cache scope source_id label (world_cache world) = Some answer ->
      answer_valid (world_arena world) scope source label answer.
  Proof.
    intros world scope source_id source label answer
      [_ [Hcache _]] Hsource Hlookup.
    destruct (lookup_cache_witness scope source_id label
      (world_cache world) Hlookup)
      as [entry [Hin [Hscope [Hid [Hlabel Hanswer]]]]].
    apply Forall_forall with (x := entry) in Hcache; [|exact Hin].
    destruct Hcache as [stored [Hstored Hvalid]].
    rewrite Hscope, Hlabel, Hanswer in Hvalid.
    rewrite Hid in Hstored.
    rewrite Hsource in Hstored; inversion Hstored; subst stored.
    exact Hvalid.
  Qed.

  Lemma append_preserves_lookup :
    forall (arena : list Residual) (fresh : Residual) state_id state,
    nth_error arena state_id = Some state ->
    nth_error (arena ++ [fresh]) state_id = Some state.
  Proof.
    intros arena fresh state_id state Hlookup.
    rewrite nth_error_app1; [exact Hlookup |].
    apply nth_error_Some; congruence.
  Qed.

  Lemma append_new_lookup :
    forall (arena : list Residual) (fresh : Residual),
    nth_error (arena ++ [fresh]) (length arena) = Some fresh.
  Proof.
    intros arena fresh.
    rewrite nth_error_app2 by lia.
    now rewrite Nat.sub_diag.
  Qed.

  Lemma answer_valid_after_append : forall arena fresh scope source label answer,
    answer_valid arena scope source label answer ->
    answer_valid (arena ++ [fresh]) scope source label answer.
  Proof.
    intros arena fresh scope source label [|target_id] Hvalid; simpl in *.
    - exact Hvalid.
    - destruct Hvalid as [target [Htarget Hstep]].
      exists target; split; [now apply append_preserves_lookup | exact Hstep].
  Qed.

  Lemma entry_valid_after_append : forall arena fresh entry,
    entry_valid arena entry -> entry_valid (arena ++ [fresh]) entry.
  Proof.
    intros arena fresh entry [source [Hsource Hanswer]].
    exists source; split.
    - now apply append_preserves_lookup.
    - now apply answer_valid_after_append.
  Qed.

  Lemma hint_valid_after_append : forall arena fresh hint,
    hint_valid arena hint -> hint_valid (arena ++ [fresh]) hint.
  Proof.
    intros arena fresh hint [state [Hstate Hfingerprint]].
    exists state; split; [now apply append_preserves_lookup |
      exact Hfingerprint].
  Qed.

  Lemma publication_valid_after_append : forall arena fresh publication,
    publication_valid arena publication ->
    publication_valid (arena ++ [fresh]) publication.
  Proof.
    intros arena fresh publication
      [source [target [Hsource [Htarget [Hstep [Hcursor Hwitness]]]]]].
    exists source, target.
    split; [now apply append_preserves_lookup |].
    split; [now apply append_preserves_lookup |].
    repeat split; assumption.
  Qed.

  Definition insert_complete_entry (world : World) (entry : Entry)
      (store : bool) : World :=
    {| world_arena := world_arena world;
       world_hints := world_hints world;
       world_cache := if store then entry :: world_cache world
                      else world_cache world;
       world_publications := world_publications world |}.

  Theorem insertion_of_complete_entry_preserves_validity :
    forall world entry store,
      world_valid world ->
      entry_valid (world_arena world) entry ->
      world_valid (insert_complete_entry world entry store).
  Proof.
    intros world entry [|] [Hhints [Hcache Hpublished]] Hentry;
      unfold insert_complete_entry, world_valid; simpl;
      repeat split; try assumption.
    now constructor.
  Qed.

  Definition evict_entries (world : World) (keep : Entry -> bool) : World :=
    {| world_arena := world_arena world;
       world_hints := world_hints world;
       world_cache := filter keep (world_cache world);
       world_publications := world_publications world |}.

  Theorem eviction_preserves_world_validity : forall world keep,
    world_valid world -> world_valid (evict_entries world keep).
  Proof.
    intros world keep [Hhints [Hcache Hpublished]].
    unfold evict_entries, world_valid; simpl.
    repeat split; try assumption.
    apply Forall_forall; intros entry Hin.
    apply filter_In in Hin as [Hmember _].
    now apply Forall_forall with (x := entry) in Hcache.
  Qed.

  Definition publish_successor (world : World)
      (publication : Publication) : World :=
    {| world_arena := world_arena world;
       world_hints := world_hints world;
       world_cache := world_cache world;
       world_publications := publication :: world_publications world |}.

  Theorem publication_preserves_world_validity : forall world publication,
    world_valid world ->
    publication_valid (world_arena world) publication ->
    world_valid (publish_successor world publication).
  Proof.
    intros world publication [Hhints [Hcache Hpublished]] Hvalid.
    unfold publish_successor, world_valid; simpl.
    repeat split; try assumption; now constructor.
  Qed.

  Inductive publication_action : World -> World -> Prop :=
  | PublishChecked : forall world publication,
      publication_valid (world_arena world) publication ->
      publication_action world (publish_successor world publication).

  Theorem publication_action_preserves_world_validity :
    forall world next,
      world_valid world ->
      publication_action world next ->
      world_valid next.
  Proof.
    intros world next Hworld Haction.
    inversion Haction; subst.
    now apply publication_preserves_world_validity.
  Qed.

  Theorem missing_target_id_cannot_be_published : forall arena publication,
    nth_error arena (publication_target publication) = None ->
    ~ publication_valid arena publication.
  Proof.
    intros arena publication Hmissing
      [source [target [_ [Htarget _]]]].
    rewrite Hmissing in Htarget; discriminate.
  Qed.

  Theorem premature_publication_breaks_world_validity :
    forall world publication,
      nth_error (world_arena world)
        (publication_target publication) = None ->
      ~ world_valid (publish_successor world publication).
  Proof.
    intros world publication Hmissing [_ [_ Hpublished]].
    unfold publish_successor in Hpublished; simpl in Hpublished.
    inversion Hpublished as [|? ? Hfirst _]; subst.
    eapply missing_target_id_cannot_be_published; eauto.
  Qed.

  Fixpoint lookup_interned (hint : Fingerprint) (state : Residual)
      (arena : list Residual) (hints : list (Fingerprint * nat))
      : option nat :=
    match hints with
    | [] => None
    | (candidate_hint, state_id) :: rest =>
        if fingerprint_eq_dec hint candidate_hint then
          match nth_error arena state_id with
          | Some stored =>
              if residual_eq_dec state stored then Some state_id
              else lookup_interned hint state arena rest
          | None => lookup_interned hint state arena rest
          end
        else lookup_interned hint state arena rest
    end.

  Lemma interned_hit_is_exact :
    forall hint state arena hints state_id,
      lookup_interned hint state arena hints = Some state_id ->
      nth_error arena state_id = Some state.
  Proof.
    intros hint state arena hints.
    induction hints as [|[candidate_hint candidate_id] rest IH];
      intros state_id Hlookup; simpl in Hlookup; [discriminate |].
    destruct (fingerprint_eq_dec hint candidate_hint) as [_ |];
      [|now apply IH].
    destruct (nth_error arena candidate_id) as [stored |]
      eqn:Hcandidate; [|now apply IH].
    destruct (residual_eq_dec state stored) as [Hequal |];
      [|now apply IH].
    inversion Hlookup; subst state_id stored; exact Hcandidate.
  Qed.

  Definition index_complete (world : World) : Prop :=
    forall state_id state,
      nth_error (world_arena world) state_id = Some state ->
      In (fingerprint_of state, state_id) (world_hints world).

  Definition strong_world_valid (world : World) : Prop :=
    world_valid world /\ index_complete world /\ NoDup (world_arena world).

  Lemma lookup_interned_some_after_cons :
    forall hint state arena first rest,
      (exists state_id,
        lookup_interned hint state arena rest = Some state_id) ->
      exists state_id,
        lookup_interned hint state arena (first :: rest) = Some state_id.
  Proof.
    intros hint state arena [first_hint first_id] rest
      [found Hfound].
    simpl.
    destruct (fingerprint_eq_dec hint first_hint) as [_ |].
    - destruct (nth_error arena first_id) as [stored |].
      + destruct (residual_eq_dec state stored) as [_ |].
        * now exists first_id.
        * now exists found.
      + now exists found.
    - now exists found.
  Qed.

  Lemma lookup_interned_finds_indexed_state :
    forall hint state arena hints state_id,
      In (hint, state_id) hints ->
      nth_error arena state_id = Some state ->
      exists found,
        lookup_interned hint state arena hints = Some found.
  Proof.
    intros hint state arena hints.
    induction hints as [|first rest IH]; intros state_id Hin Hlookup.
    - contradiction.
    - destruct Hin as [Hfirst | Hrest].
      + subst first; simpl.
        destruct (fingerprint_eq_dec hint hint) as [_ | Hneq];
          [|contradiction].
        rewrite Hlookup.
        destruct (residual_eq_dec state state) as [_ | Hneq];
          [now exists state_id | contradiction].
      + apply lookup_interned_some_after_cons.
        eapply IH; eauto.
  Qed.

  Lemma interned_miss_is_new : forall world state,
    index_complete world ->
    lookup_interned (fingerprint_of state) state
      (world_arena world) (world_hints world) = None ->
    ~ In state (world_arena world).
  Proof.
    intros world state Hindex Hmiss Hin.
    apply In_nth_error in Hin.
    destruct Hin as [state_id Hlookup].
    pose proof (Hindex state_id state Hlookup) as Hhint.
    destruct (lookup_interned_finds_indexed_state
      (fingerprint_of state) (world_arena world)
      (world_hints world) state_id Hhint Hlookup)
      as [found Hfound].
    rewrite Hmiss in Hfound; discriminate.
  Qed.

  Definition append_interned (world : World) (state : Residual) : World :=
    {| world_arena := world_arena world ++ [state];
       world_hints :=
         (fingerprint_of state, length (world_arena world)) ::
           world_hints world;
       world_cache := world_cache world;
       world_publications := world_publications world |}.

  Theorem append_interned_preserves_world_valid : forall world state,
    world_valid world -> world_valid (append_interned world state).
  Proof.
    intros world state [Hhints [Hcache Hpublished]].
    unfold append_interned, world_valid; simpl.
    repeat split.
    - constructor.
      + exists state; split; [apply append_new_lookup | reflexivity].
      + induction Hhints as [|hint rest Hvalid Hrest IH];
          constructor; [now apply hint_valid_after_append | exact IH].
    - induction Hcache as [|entry rest Hvalid Hrest IH];
        constructor; [now apply entry_valid_after_append | exact IH].
    - induction Hpublished as [|publication rest Hvalid Hrest IH];
        constructor; [now apply publication_valid_after_append | exact IH].
  Qed.

  Lemma append_interned_preserves_index_complete : forall world state,
    index_complete world -> index_complete (append_interned world state).
  Proof.
    intros world state Hindex state_id stored Hlookup.
    unfold append_interned in *; simpl in *.
    destruct (lt_dec state_id (length (world_arena world)))
      as [Hold | Hnew].
    - rewrite nth_error_app1 in Hlookup by lia.
      right; now apply Hindex.
    - assert (Hrange : state_id < length (world_arena world) + 1).
      { assert (Hnotnone :
          nth_error (world_arena world ++ [state]) state_id <> None).
        { rewrite Hlookup; discriminate. }
        apply nth_error_Some in Hnotnone.
        rewrite length_app in Hnotnone; simpl in Hnotnone; lia. }
      assert (Hequal : state_id = length (world_arena world)) by lia.
      subst state_id.
      rewrite append_new_lookup in Hlookup; inversion Hlookup; subst stored.
      now left.
  Qed.

  Lemma nodup_append_fresh : forall (arena : list Residual) state,
    NoDup arena -> ~ In state arena -> NoDup (arena ++ [state]).
  Proof.
    intros arena state Hnodup Hfresh.
    induction Hnodup as [|first rest Hnot Htail IH]; simpl.
    - constructor; [intro Hin; contradiction | constructor].
    - constructor.
      + intro Hin.
        apply in_app_or in Hin.
        destruct Hin as [Hold | [Hequal | []]].
        * now apply Hnot.
        * subst first; apply Hfresh; now left.
      + apply IH; intro Hin; apply Hfresh; now right.
  Qed.

  Theorem fresh_intern_preserves_strong_world_valid : forall world state,
    strong_world_valid world ->
    lookup_interned (fingerprint_of state) state
      (world_arena world) (world_hints world) = None ->
    strong_world_valid (append_interned world state).
  Proof.
    intros world state [Hworld [Hindex Hnodup]] Hmiss.
    unfold strong_world_valid; split; [|split].
    - now apply append_interned_preserves_world_valid.
    - now apply append_interned_preserves_index_complete.
    - apply nodup_append_fresh; [exact Hnodup |].
      eapply interned_miss_is_new; eauto.
  Qed.

  Definition make_entry (scope : Scope) (source_id : nat)
      (label : Label) (answer : completion) : Entry :=
    {| cached_scope := scope;
       cached_source := source_id;
       cached_label := label;
       cached_answer := answer |}.

  Lemma complete_entry_insertion_preserves_strong_world_valid :
    forall world entry store,
      strong_world_valid world ->
      entry_valid (world_arena world) entry ->
      strong_world_valid (insert_complete_entry world entry store).
  Proof.
    intros world entry store [Hworld [Hindex Hnodup]] Hentry.
    unfold strong_world_valid; split; [|split].
    - now apply insertion_of_complete_entry_preserves_validity.
    - exact Hindex.
    - exact Hnodup.
  Qed.

  Theorem eviction_preserves_strong_world_valid : forall world keep,
    strong_world_valid world ->
    strong_world_valid (evict_entries world keep).
  Proof.
    intros world keep [Hworld [Hindex Hnodup]].
    unfold strong_world_valid; split; [|split].
    - now apply eviction_preserves_world_validity.
    - exact Hindex.
    - exact Hnodup.
  Qed.

  Theorem publication_preserves_strong_world_valid :
    forall world publication,
      strong_world_valid world ->
      publication_valid (world_arena world) publication ->
      strong_world_valid (publish_successor world publication).
  Proof.
    intros world publication [Hworld [Hindex Hnodup]] Hvalid.
    unfold strong_world_valid; split; [|split].
    - now apply publication_preserves_world_validity.
    - exact Hindex.
    - exact Hnodup.
  Qed.

  Inductive transition_action (scope : Scope) (source_id : nat)
      (label : Label) : World -> World -> transition_outcome -> Prop :=
  | CacheHit : forall world answer,
      lookup_cache scope source_id label (world_cache world) = Some answer ->
      transition_action scope source_id label world world (Finished answer)
  | CacheMissDead : forall world source store,
      lookup_cache scope source_id label (world_cache world) = None ->
      nth_error (world_arena world) source_id = Some source ->
      semantic_step scope source label = None ->
      transition_action scope source_id label world
        (insert_complete_entry world
          (make_entry scope source_id label Dead) store)
        (Finished Dead)
  | CacheMissLiveReused : forall world source target target_id store,
      lookup_cache scope source_id label (world_cache world) = None ->
      nth_error (world_arena world) source_id = Some source ->
      semantic_step scope source label = Some target ->
      lookup_interned (fingerprint_of target) target
        (world_arena world) (world_hints world) = Some target_id ->
      transition_action scope source_id label world
        (insert_complete_entry world
          (make_entry scope source_id label (Live target_id)) store)
        (Finished (Live target_id))
  | CacheMissLiveFresh : forall world source target store,
      lookup_cache scope source_id label (world_cache world) = None ->
      nth_error (world_arena world) source_id = Some source ->
      semantic_step scope source label = Some target ->
      lookup_interned (fingerprint_of target) target
        (world_arena world) (world_hints world) = None ->
      transition_action scope source_id label world
        (insert_complete_entry (append_interned world target)
          (make_entry scope source_id label
            (Live (length (world_arena world)))) store)
        (Finished (Live (length (world_arena world))))
  | CacheMissResourceFailure : forall world source,
      lookup_cache scope source_id label (world_cache world) = None ->
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label world world ResourceFailure.

  Lemma dead_entry_valid :
    forall (world : World) (scope : Scope) source_id source label,
    nth_error (world_arena world) source_id = Some source ->
    semantic_step scope source label = None ->
    entry_valid (world_arena world)
      (make_entry scope source_id label Dead).
  Proof.
    intros world scope source_id source label Hsource Hdead.
    exists source; split; [exact Hsource | exact Hdead].
  Qed.

  Lemma live_entry_valid :
    forall (world : World) (scope : Scope)
      source_id source label target_id target,
      nth_error (world_arena world) source_id = Some source ->
      nth_error (world_arena world) target_id = Some target ->
      semantic_step scope source label = Some target ->
      entry_valid (world_arena world)
        (make_entry scope source_id label (Live target_id)).
  Proof.
    intros world scope source_id source label target_id target
      Hsource Htarget Hstep.
    exists source; split; [exact Hsource |].
    exists target; now split.
  Qed.

  Theorem transition_action_preserves_strong_world_valid :
    forall scope source_id label world next outcome,
      strong_world_valid world ->
      transition_action scope source_id label world next outcome ->
      strong_world_valid next.
  Proof.
    intros scope source_id label world next outcome Hworld Haction.
    destruct Haction.
    - exact Hworld.
    - eapply complete_entry_insertion_preserves_strong_world_valid;
        [exact Hworld |].
      eapply dead_entry_valid; eauto.
    - eapply complete_entry_insertion_preserves_strong_world_valid;
        [exact Hworld |].
      eapply live_entry_valid; eauto.
      now apply interned_hit_is_exact in H2.
    - eapply complete_entry_insertion_preserves_strong_world_valid.
      + eapply fresh_intern_preserves_strong_world_valid; eauto.
      + eapply live_entry_valid.
        * apply append_preserves_lookup; eauto.
        * apply append_new_lookup.
        * eauto.
    - exact Hworld.
  Qed.

  Theorem completed_action_refines_semantic_step :
    forall scope source_id source label world next answer,
      strong_world_valid world ->
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label world next
        (Finished answer) ->
      answer_valid (world_arena next) scope source label answer.
  Proof.
    intros scope source_id source label world next answer
      [Hworld _] Hsource Haction.
    inversion Haction; subst; clear Haction.
    - eapply cache_hit_is_semantically_complete.
      + exact Hworld.
      + exact Hsource.
      + eassumption.
    - assert (Hsame : source0 = source) by congruence.
      subst source0; unfold answer_valid; simpl; exact H4.
    - assert (Hsame : source0 = source).
      { congruence. }
      subst source0.
      exists target; split.
      + eapply interned_hit_is_exact; eassumption.
      + eassumption.
    - assert (Hsame : source0 = source).
      { congruence. }
      subst source0.
      exists target; split; [apply append_new_lookup | eassumption].
  Qed.

  Theorem transition_preserves_source_referent :
    forall scope source_id source label world next outcome,
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label world next outcome ->
      nth_error (world_arena next) source_id = Some source.
  Proof.
    intros scope source_id source label world next outcome Hsource Haction.
    destruct Haction; simpl; try exact Hsource.
    now apply append_preserves_lookup.
  Qed.

  Theorem resource_failure_does_not_publish :
    forall scope source_id label world next,
      transition_action scope source_id label world next ResourceFailure ->
      next = world.
  Proof.
    intros scope source_id label world next Haction.
    now inversion Haction.
  Qed.

  Theorem resource_failure_is_not_semantic_dead :
    ResourceFailure <> Finished Dead.
  Proof. discriminate. Qed.

  Theorem changed_scope_cannot_reuse_single_entry :
    forall scope source_id label entry,
      scope <> cached_scope entry ->
      lookup_cache scope source_id label [entry] = None.
  Proof.
    intros scope source_id label entry Hdifferent.
    simpl.
    destruct (scope_eq_dec scope (cached_scope entry)) as [Hequal |];
      [contradiction | reflexivity].
  Qed.

  Theorem completed_action_after_eviction_is_semantic :
    forall scope source_id source label world keep next answer,
      strong_world_valid world ->
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label
        (evict_entries world keep) next (Finished answer) ->
      answer_valid (world_arena next) scope source label answer.
  Proof.
    intros scope source_id source label world keep next answer
      Hworld Hsource Haction.
    eapply completed_action_refines_semantic_step
      with (world := evict_entries world keep).
    - now apply eviction_preserves_strong_world_valid.
    - exact Hsource.
    - exact Haction.
  Qed.

  Theorem live_publication_valid_after_transition :
    forall scope source_id source label world next target_id cursor witness,
      strong_world_valid world ->
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label world next
        (Finished (Live target_id)) ->
      (forall target,
        nth_error (world_arena next) target_id = Some target ->
        cursor_valid scope target cursor /\
        witness_valid scope target witness) ->
      publication_valid (world_arena next)
        {| publication_scope := scope;
           publication_source := source_id;
           publication_label := label;
           publication_target := target_id;
           publication_cursor := cursor;
           publication_witness := witness |}.
  Proof.
    intros scope source_id source label world next target_id cursor witness
      Hworld Hsource Haction Hmetadata.
    pose proof (completed_action_refines_semantic_step
      Hworld Hsource Haction) as Hanswer.
    destruct Hanswer as [target [Htarget Hstep]].
    destruct (Hmetadata target Htarget) as [Hcursor Hwitness].
    exists source, target.
    repeat split; try assumption.
    eapply transition_preserves_source_referent; eauto.
  Qed.

  Theorem publish_completed_live_successor_preserves_validity :
    forall scope source_id source label world next target_id cursor witness,
      strong_world_valid world ->
      nth_error (world_arena world) source_id = Some source ->
      transition_action scope source_id label world next
        (Finished (Live target_id)) ->
      (forall target,
        nth_error (world_arena next) target_id = Some target ->
        cursor_valid scope target cursor /\
        witness_valid scope target witness) ->
      strong_world_valid
        (publish_successor next
          {| publication_scope := scope;
             publication_source := source_id;
             publication_label := label;
             publication_target := target_id;
             publication_cursor := cursor;
             publication_witness := witness |}).
  Proof.
    intros scope source_id source label world next target_id cursor witness
      Hworld Hsource Haction Hmetadata.
    apply publication_preserves_strong_world_valid.
    - eapply transition_action_preserves_strong_world_valid; eauto.
    - eapply live_publication_valid_after_transition; eauto.
  Qed.

  Theorem every_published_id_has_frontier_and_metadata :
    forall world publication,
      world_valid world ->
      In publication (world_publications world) ->
      exists target,
        nth_error (world_arena world)
          (publication_target publication) = Some target /\
        cursor_valid (publication_scope publication) target
          (publication_cursor publication) /\
        witness_valid (publication_scope publication) target
          (publication_witness publication).
  Proof.
    intros world publication [_ [_ Hpublished]] Hin.
    apply Forall_forall with (x := publication) in Hpublished;
      [|exact Hin].
    destruct Hpublished as
      [source [target [_ [Htarget [_ [Hcursor Hwitness]]]]]].
    exists target; repeat split; assumption.
  Qed.

End CacheModel.

Example fingerprint_collision_is_not_identity :
  lookup_interned Nat.eq_dec Nat.eq_dec 0 20 [10] [(0, 0)] = None.
Proof. reflexivity. Qed.

Definition dead_entry_control : cached_transition nat nat :=
  {| cached_scope := 7;
     cached_source := 0;
     cached_label := 3;
     cached_answer := Dead |}.

Example cached_dead_is_distinct_from_a_miss :
  lookup_cache Nat.eq_dec Nat.eq_dec 7 0 3 [dead_entry_control] =
    Some Dead /\
  lookup_cache Nat.eq_dec Nat.eq_dec 7 0 3 [] = None.
Proof. split; reflexivity. Qed.

Example changed_scope_rejects_cached_answer :
  lookup_cache Nat.eq_dec Nat.eq_dec 8 0 3 [dead_entry_control] = None.
Proof. reflexivity. Qed.
