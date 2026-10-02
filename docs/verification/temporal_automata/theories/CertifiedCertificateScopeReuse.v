(** * Checked certificate reuse across request scopes

    This module specializes the finite exact-natural, score-only checker.
    Exact-scope reuse replays a certificate at the identical request. The
    weaker mode replays only a cutoff-independent score relation: it requires
    every stable field to agree and permits a changed cutoff, but it never
    reuses a cached prune decision without the separate cutoff gate.

    All fields are proof-level tokens. Binding them and the score environment
    to a live query, parameter values, snapshot, numeric implementation, or
    Rust operation is a separate source-correspondence obligation. *)

From Stdlib Require Import Arith Bool Lia.
Require Import CertifiedContracts CertificateChecking CertificateScope.

Inductive certificate_reuse_mode :=
| ExactScopeReuse
| CutoffIndependentScoreReuse.

Definition check_certificate_reuse (mode : certificate_reuse_mode)
    (requested : scoped_request) (lookup_key : nat)
    (origin target : cost_expression) (kind : relation_kind)
    (packet : reusable_score_proof) : bool :=
  match mode with
  | ExactScopeReuse =>
      accept_same_scope_score_reuse requested lookup_key
        origin target kind packet
  | CutoffIndependentScoreReuse =>
      accept_score_reuse requested lookup_key origin target kind packet
  end.

Theorem accepted_reuse_has_stable_scope_and_score_relation :
  forall mode requested lookup_key origin target kind packet,
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = true ->
    stable_part requested = stable_part (captured_request packet) /\
    certificate_scope (captured_chain packet) =
      realization_of (captured_request packet) /\
    forall environment, relation_holds origin target kind environment.
Proof.
  intros mode requested lookup_key origin target kind packet Haccept.
  destruct mode; simpl in Haccept.
  - unfold accept_same_scope_score_reuse in Haccept.
    destruct (scoped_request_eq_dec requested (captured_request packet));
      [|discriminate].
    eapply accepted_score_reuse_has_stable_identity; exact Haccept.
  - eapply accepted_score_reuse_has_stable_identity; exact Haccept.
Qed.

(** Each named stable field is retained even when the cutoff changes. The
    parameter token is checked along with the separate gap and stiffness
    values, so token equality alone is insufficient. *)
Theorem accepted_reuse_binds_every_stable_field :
  forall mode requested lookup_key origin target kind packet,
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = true ->
    let old := stable_part (captured_request packet) in
    let new := stable_part requested in
    bound_domain new = bound_domain old /\
    bound_query new = bound_query old /\
    bound_parameter_token new = bound_parameter_token old /\
    bound_gap new = bound_gap old /\
    bound_stiffness new = bound_stiffness old /\
    bound_snapshot new = bound_snapshot old /\
    bound_revision new = bound_revision old /\
    bound_arithmetic new = bound_arithmetic old /\
    bound_observation new = bound_observation old /\
    bound_label_context new = bound_label_context old.
Proof.
  intros mode requested lookup_key origin target kind packet Haccept.
  destruct (accepted_reuse_has_stable_scope_and_score_relation
    _ _ _ _ _ _ _ Haccept) as [Hstable _].
  simpl.
  repeat split; now rewrite Hstable.
Qed.

Theorem exact_reuse_requires_identical_request :
  forall requested lookup_key origin target kind packet,
    check_certificate_reuse ExactScopeReuse requested lookup_key
      origin target kind packet = true ->
    requested = captured_request packet.
Proof.
  intros requested lookup_key origin target kind packet Haccept.
  simpl in Haccept.
  apply accepted_same_scope_score_reuse_has_exact_request in Haccept.
  exact (proj1 Haccept).
Qed.

(** A changed projection of any stable field is enough to reject either
    mode, even when the lookup hint collides. Instantiate [project] with
    domain, query, parameter token, gap, stiffness, snapshot, revision,
    arithmetic, observation, or label context. *)
Theorem changed_stable_projection_rejects_reuse :
  forall (A : Type) (project : stable_request -> A)
    mode requested lookup_key origin target kind packet,
    project (stable_part requested) <>
      project (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros A project mode requested lookup_key origin target kind packet
    Hchanged.
  destruct (check_certificate_reuse mode requested lookup_key
    origin target kind packet) eqn:Haccept; [|reflexivity].
  exfalso; apply Hchanged.
  destruct (accepted_reuse_has_stable_scope_and_score_relation
    _ _ _ _ _ _ _ Haccept) as [Hstable _].
  now rewrite Hstable.
Qed.

Corollary changed_query_rejects_reuse :
  forall mode requested lookup_key origin target kind packet,
    bound_query (stable_part requested) <>
      bound_query (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros; eapply changed_stable_projection_rejects_reuse
    with (project := bound_query); eassumption.
Qed.

Corollary changed_gap_rejects_reuse :
  forall mode requested lookup_key origin target kind packet,
    bound_gap (stable_part requested) <>
      bound_gap (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros; eapply changed_stable_projection_rejects_reuse
    with (project := bound_gap); eassumption.
Qed.

Corollary changed_stiffness_rejects_reuse :
  forall mode requested lookup_key origin target kind packet,
    bound_stiffness (stable_part requested) <>
      bound_stiffness (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros; eapply changed_stable_projection_rejects_reuse
    with (project := bound_stiffness); eassumption.
Qed.

Corollary changed_revision_rejects_reuse :
  forall mode requested lookup_key origin target kind packet,
    bound_revision (stable_part requested) <>
      bound_revision (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros; eapply changed_stable_projection_rejects_reuse
    with (project := bound_revision); eassumption.
Qed.

Corollary changed_arithmetic_rejects_reuse :
  forall mode requested lookup_key origin target kind packet,
    bound_arithmetic (stable_part requested) <>
      bound_arithmetic (stable_part (captured_request packet)) ->
    check_certificate_reuse mode requested lookup_key
      origin target kind packet = false.
Proof.
  intros; eapply changed_stable_projection_rejects_reuse
    with (project := bound_arithmetic); eassumption.
Qed.

Theorem exact_reuse_rejects_changed_cutoff :
  forall requested lookup_key origin target kind packet,
    bound_cutoff requested <>
      bound_cutoff (captured_request packet) ->
    check_certificate_reuse ExactScopeReuse requested lookup_key
      origin target kind packet = false.
Proof.
  intros requested lookup_key origin target kind packet Hchanged.
  destruct (check_certificate_reuse ExactScopeReuse requested lookup_key
    origin target kind packet) eqn:Haccept; [|reflexivity].
  exfalso; apply Hchanged.
  now rewrite (exact_reuse_requires_identical_request
    _ _ _ _ _ _ Haccept).
Qed.

(** Relation reuse does not carry a prune verdict. A cached strict-prune
    decision is transferred only to the same or a lower cutoff; a raised
    cutoff needs a fresh comparison with the reused lower expression. *)
Definition check_cached_decision_reuse (requested : scoped_request)
    (lookup_key : nat) (environment : nat -> nat)
    (origin lower : cost_expression) (packet : reusable_score_proof) : bool :=
  accept_cached_prune requested lookup_key environment origin lower packet.

Theorem raised_cutoff_rejects_cached_decision :
  forall requested lookup_key environment origin lower packet,
    bound_cutoff (captured_request packet) < bound_cutoff requested ->
    check_cached_decision_reuse requested lookup_key environment
      origin lower packet = false.
Proof.
  intros requested lookup_key environment origin lower packet Hraised.
  unfold check_cached_decision_reuse.
  now apply increased_cutoff_cannot_reuse_cached_prune.
Qed.

Theorem accepted_cached_decision_reuse_is_a_strict_prune :
  forall requested lookup_key environment origin lower packet,
    check_cached_decision_reuse requested lookup_key environment
      origin lower packet = true ->
    bound_cutoff requested < evaluate environment origin.
Proof.
  intros requested lookup_key environment origin lower packet Haccept.
  unfold check_cached_decision_reuse in Haccept.
  now apply accepted_cached_prune_is_sound in Haccept.
Qed.

Example same_scope_exact_reuse_accepts_sample :
  check_certificate_reuse ExactScopeReuse
    (test_request 2 4 6 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = true.
Proof. reflexivity. Qed.

Example same_key_changed_query_is_rejected_here :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 8 4 6 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_gap_is_rejected_here :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 2 8 6 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_stiffness_is_rejected_here :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 2 4 8 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_stale_revision_is_rejected_here :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 2 4 6 12 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_rounded_profile_is_rejected_here :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 2 4 6 11 7 RoundedBinary64) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example raised_cutoff_replays_relation_but_not_decision :
  check_certificate_reuse CutoffIndependentScoreReuse
    (test_request 2 4 6 11 8 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = true /\
  check_cached_decision_reuse
    (test_request 2 4 6 11 8 ExactNaturals) 17 (fun _ => 9)
    test_origin test_lower test_packet = false.
Proof. split; reflexivity. Qed.

Example raised_cutoff_after_fresh_check_can_prune :
  accept_rechecked_prune (test_request 2 4 6 11 8 ExactNaturals)
    17 (fun _ => 9) test_origin test_lower test_packet = true.
Proof. reflexivity. Qed.
