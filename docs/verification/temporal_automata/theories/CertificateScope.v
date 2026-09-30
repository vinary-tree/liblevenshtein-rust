(** * Scope-safe reuse of finite score certificates

    These rules apply only to the exact-natural, score-only expression
    checker in CertificateChecking. A cache key is a lookup hint, never an
    authority. The request below is supplied by the caller from authoritative
    query and snapshot data; binding that data to Rust is a separate proof.
    A pruning transfer below uses one score environment for both cutoffs;
    a runtime instance must prove that changing the cutoff cannot change the
    exact score inputs. In particular, this file does not authorize reuse of
    whole range/kNN results or of witness and resource observations. *)

From Stdlib Require Import Arith Bool Lia.
Require Import CertifiedContracts CertificateChecking.

Record stable_request := {
  bound_domain : nat;
  bound_query : nat;
  bound_parameter_token : nat;
  bound_gap : nat;
  bound_stiffness : nat;
  bound_snapshot : nat;
  bound_revision : nat;
  bound_arithmetic : arithmetic_profile;
  bound_observation : observation_profile;
  bound_label_context : nat
}.

Record scoped_request := {
  stable_part : stable_request;
  bound_cutoff : nat
}.

Definition stable_request_eq_dec (lhs rhs : stable_request) :
    {lhs = rhs} + {lhs <> rhs}.
Proof.
  decide equality; try apply Nat.eq_dec;
    try apply arithmetic_profile_eq_dec;
    try apply observation_profile_eq_dec.
Defined.

Definition scoped_request_eq_dec (lhs rhs : scoped_request) :
    {lhs = rhs} + {lhs <> rhs}.
Proof.
  decide equality; try apply Nat.eq_dec; apply stable_request_eq_dec.
Defined.

Definition realization_of (request : scoped_request) : realization_scope :=
  let stable := stable_part request in
  {| base_scope :=
       {| scope_domain := bound_domain stable;
          scope_query := bound_query stable;
          scope_parameters := bound_parameter_token stable;
          scope_arithmetic := bound_arithmetic stable;
          scope_snapshot := bound_snapshot stable;
          scope_cutoff := bound_cutoff request;
          scope_observation := bound_observation stable |};
     scoped_revision := bound_revision stable;
     scoped_label_context := bound_label_context stable |}.

(** The parameter token remains an opaque field of the earlier checker.
    Gap and stiffness are compared separately here, so a colliding or stale
    parameter token cannot authorize reuse across either changed value. *)
Record reusable_score_proof := {
  captured_request : scoped_request;
  lookup_hint : nat;
  captured_chain : scoped_certificate
}.

Definition accept_score_reuse (request : scoped_request) (key : nat)
    (origin target : cost_expression) (kind : relation_kind)
    (packet : reusable_score_proof) : bool :=
  if Nat.eqb key (lookup_hint packet) then
  if stable_request_eq_dec (stable_part request)
       (stable_part (captured_request packet)) then
    accept_scoped (realization_of (captured_request packet))
      origin target kind (captured_chain packet)
  else false else false.

(** A whole-operation result can depend on the inclusive cutoff. This helper
    gates score-proof reuse on exact request equality; it does not certify
    whole-operation output. *)
Definition accept_same_scope_score_reuse (request : scoped_request) (key : nat)
    (origin target : cost_expression) (kind : relation_kind)
    (packet : reusable_score_proof) : bool :=
  if scoped_request_eq_dec request (captured_request packet) then
    accept_score_reuse request key origin target kind packet
  else false.

Theorem accepted_score_reuse_has_stable_identity :
  forall request key origin target kind packet,
    accept_score_reuse request key origin target kind packet = true ->
    stable_part request = stable_part (captured_request packet) /\
    certificate_scope (captured_chain packet) =
      realization_of (captured_request packet) /\
    forall environment, relation_holds origin target kind environment.
Proof.
  intros request key origin target kind packet Haccept.
  unfold accept_score_reuse in Haccept.
  destruct (Nat.eqb key (lookup_hint packet)); [|discriminate].
  destruct (stable_request_eq_dec (stable_part request)
    (stable_part (captured_request packet))) as [Hstable | Hstable];
    [|discriminate].
  destruct (accepted_scoped_certificate_sound _ _ _ _ _ Haccept)
    as [Hscope Hsemantic].
  repeat split; assumption.
Qed.

Theorem accepted_same_scope_score_reuse_has_exact_request :
  forall request key origin target kind packet,
    accept_same_scope_score_reuse request key origin target kind packet = true ->
    request = captured_request packet /\
    forall environment, relation_holds origin target kind environment.
Proof.
  intros request key origin target kind packet Haccept.
  unfold accept_same_scope_score_reuse in Haccept.
  destruct (scoped_request_eq_dec request (captured_request packet))
    as [Hequal | Hdifferent]; [|discriminate].
  split; [exact Hequal |].
  destruct (accepted_score_reuse_has_stable_identity
    _ _ _ _ _ _ Haccept) as [_ [_ Hsemantic]].
  exact Hsemantic.
Qed.

Theorem accepted_score_reuse_preserves_query_parameters_and_revision :
  forall request key origin target kind packet,
    accept_score_reuse request key origin target kind packet = true ->
    bound_query (stable_part request) =
      bound_query (stable_part (captured_request packet)) /\
    bound_gap (stable_part request) =
      bound_gap (stable_part (captured_request packet)) /\
    bound_stiffness (stable_part request) =
      bound_stiffness (stable_part (captured_request packet)) /\
    bound_revision (stable_part request) =
      bound_revision (stable_part (captured_request packet)) /\
    bound_arithmetic (stable_part request) =
      bound_arithmetic (stable_part (captured_request packet)).
Proof.
  intros request key origin target kind packet Haccept.
  destruct (accepted_score_reuse_has_stable_identity
    _ _ _ _ _ _ Haccept) as [Hstable _].
  now rewrite Hstable.
Qed.

(** A cutoff-independent lower expression may be reused as a relation.
    Pruning still compares its value against the *new* cutoff. *)
Definition accept_rechecked_prune (request : scoped_request) (key : nat)
    (environment : nat -> nat) (origin lower : cost_expression)
    (packet : reusable_score_proof) : bool :=
  accept_score_reuse request key origin lower LowerThanSource packet &&
    Nat.ltb (bound_cutoff request) (evaluate environment lower).

Theorem accepted_rechecked_prune_is_sound :
  forall request key environment origin lower packet,
    accept_rechecked_prune request key environment origin lower packet = true ->
    bound_cutoff request < evaluate environment origin.
Proof.
  intros request key environment origin lower packet Haccept.
  unfold accept_rechecked_prune in Haccept.
  apply andb_true_iff in Haccept as [Hreuse Hcutoff].
  apply Nat.ltb_lt in Hcutoff.
  destruct (accepted_score_reuse_has_stable_identity
    _ _ _ _ _ _ Hreuse) as [_ [_ Hlower]].
  unfold relation_holds in Hlower.
  specialize (Hlower environment). lia.
Qed.

(** Reuse of a cached *decision* is narrower: a lower cutoff preserves a
    previously justified strict prune for the same exact score environment.
    This checker replays the old evidence; no constant-time runtime cache
    implementation or environment correspondence is claimed. An increased
    cutoff needs a fresh inequality check even when the score proof remains
    reusable. *)
Definition accept_cached_prune (request : scoped_request) (key : nat)
    (environment : nat -> nat) (origin lower : cost_expression)
    (packet : reusable_score_proof) : bool :=
  Nat.leb (bound_cutoff request) (bound_cutoff (captured_request packet)) &&
    accept_rechecked_prune (captured_request packet) key environment
      origin lower packet &&
    (if stable_request_eq_dec (stable_part request)
          (stable_part (captured_request packet)) then true else false).

Theorem accepted_cached_prune_is_sound :
  forall request key environment origin lower packet,
    accept_cached_prune request key environment origin lower packet = true ->
    bound_cutoff request < evaluate environment origin.
Proof.
  intros request key environment origin lower packet Haccept.
  unfold accept_cached_prune in Haccept.
  apply andb_true_iff in Haccept as [Hprefix _].
  apply andb_true_iff in Hprefix as [Horder Hold].
  apply Nat.leb_le in Horder.
  apply accepted_rechecked_prune_is_sound in Hold. lia.
Qed.

Theorem increased_cutoff_cannot_reuse_cached_prune :
  forall request key environment origin lower packet,
    bound_cutoff (captured_request packet) < bound_cutoff request ->
    accept_cached_prune request key environment origin lower packet = false.
Proof.
  intros request key environment origin lower packet Hincrease.
  unfold accept_cached_prune.
  assert (Hfalse :
    Nat.leb (bound_cutoff request)
      (bound_cutoff (captured_request packet)) = false).
  { apply Nat.leb_gt. exact Hincrease. }
  now rewrite Hfalse.
Qed.

Definition test_request (query gap stiffness revision cutoff : nat)
    (arithmetic : arithmetic_profile) : scoped_request :=
  {| stable_part :=
       {| bound_domain := 1; bound_query := query;
          bound_parameter_token := 3; bound_gap := gap;
          bound_stiffness := stiffness; bound_snapshot := 5;
          bound_revision := revision; bound_arithmetic := arithmetic;
          bound_observation := ScoreOnly; bound_label_context := 13 |};
     bound_cutoff := cutoff |}.

Definition test_packet : reusable_score_proof :=
  {| captured_request := test_request 2 4 6 11 7 ExactNaturals;
     lookup_hint := 17;
     captured_chain := singleton_certificate sample_lower_step |}.

Definition test_origin := Addition (InputVar 0) (InputVar 1).
Definition test_lower := InputVar 0.

Example unchanged_score_proof_is_reusable :
  accept_score_reuse (test_request 2 4 6 11 8 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = true.
Proof. reflexivity. Qed.

Example same_key_changed_query_is_rejected :
  accept_score_reuse (test_request 8 4 6 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_gap_is_rejected :
  accept_score_reuse (test_request 2 8 6 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_stiffness_is_rejected :
  accept_score_reuse (test_request 2 4 8 11 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_stale_revision_is_rejected :
  accept_score_reuse (test_request 2 4 6 12 7 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_arithmetic_is_rejected :
  accept_score_reuse (test_request 2 4 6 11 7 RoundedBinary64) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example same_key_changed_cutoff_fails_exact_scope_gate :
  accept_same_scope_score_reuse (test_request 2 4 6 11 8 ExactNaturals) 17
    test_origin test_lower LowerThanSource test_packet = false.
Proof. reflexivity. Qed.

Example increased_cutoff_cannot_reuse_old_prune_decision :
  accept_cached_prune (test_request 2 4 6 11 8 ExactNaturals) 17
    (fun _ => 9) test_origin test_lower test_packet = false.
Proof. reflexivity. Qed.

Example increased_cutoff_can_prune_after_fresh_check :
  accept_rechecked_prune (test_request 2 4 6 11 8 ExactNaturals) 17
    (fun _ => 9) test_origin test_lower test_packet = true.
Proof. reflexivity. Qed.

Example changed_score_environment_breaks_cached_decision_transfer :
  accept_cached_prune (test_request 2 4 6 11 6 ExactNaturals) 17
    (fun _ => 9) test_origin test_lower test_packet = true /\
  ~ (6 < evaluate (fun _ => 0) test_origin).
Proof. split; [reflexivity | simpl; lia]. Qed.
