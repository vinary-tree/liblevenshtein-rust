(** * Replayable exact-range certificate laws

    This assumption-free model isolates the trust boundary of the bounded
    exact-range certificate in [time_series::elastic::walker].  A verifier
    does not accept recorded lower bounds on authority: it recomputes the
    canonical evidence stream for the exact query, cutoff, and snapshot
    binding, validates every K1--K4 decision, checks every logical resource
    ceiling, and requires structural equality with the supplied certificate.

    Kernel recurrence correspondence, dictionary paging completeness, and
    snapshot hashing are separate instance obligations.  This file proves
    the generic replay, mutation-rejection, resource, determinism, and
    completion-phase laws without axioms or admitted propositions.
*)

From Stdlib Require Import Arith Bool Lia List PeanoNat.
Import ListNotations.

(** ** Query/snapshot binding *)

Record certificate_binding : Type := {
  binding_snapshot : option nat;
  binding_query_words : list nat;
  binding_cutoff : nat
}.

Definition option_nat_eqb (left right : option nat) : bool :=
  match left, right with
  | Some left_value, Some right_value => Nat.eqb left_value right_value
  | None, None => true
  | _, _ => false
  end.

Fixpoint list_nat_eqb (left right : list nat) : bool :=
  match left, right with
  | [], [] => true
  | left_head :: left_tail, right_head :: right_tail =>
      Nat.eqb left_head right_head && list_nat_eqb left_tail right_tail
  | _, _ => false
  end.

Definition binding_eqb
    (left right : certificate_binding) : bool :=
  option_nat_eqb (binding_snapshot left) (binding_snapshot right)
  && list_nat_eqb (binding_query_words left) (binding_query_words right)
  && Nat.eqb (binding_cutoff left) (binding_cutoff right).

Lemma option_nat_eqb_true_iff : forall left right,
  option_nat_eqb left right = true <-> left = right.
Proof.
  intros [left |] [right |]; simpl; try (split; intro H; congruence).
  rewrite Nat.eqb_eq. split; intro H; [now subst | now inversion H].
Qed.

Lemma list_nat_eqb_true_iff : forall left right,
  list_nat_eqb left right = true <-> left = right.
Proof.
  induction left as [| left_head left_tail IH];
    intros [| right_head right_tail]; simpl; try (split; intro H; congruence).
  rewrite Bool.andb_true_iff, Nat.eqb_eq, IH.
  split.
  - intros [Hhead Htail]. now subst.
  - intro Hequal. inversion Hequal; subst. now split.
Qed.

Theorem binding_eqb_true_iff : forall left right,
  binding_eqb left right = true <-> left = right.
Proof.
  intros [left_snapshot left_query left_cutoff]
         [right_snapshot right_query right_cutoff].
  unfold binding_eqb; simpl.
  rewrite !Bool.andb_true_iff.
  rewrite option_nat_eqb_true_iff, list_nat_eqb_true_iff, Nat.eqb_eq.
  split.
  - intros [[Hsnapshot Hquery] Hcutoff]. now subst.
  - intro Hequal. inversion Hequal; subst. repeat split; reflexivity.
Qed.

(** ** K1--K4 evidence *)

Inductive range_evidence : Type :=
| PrefixPruned (path : list nat) (lower_bound : nat)
| SubtreePruned (path : list nat) (lower_bound : nat)
| TerminalPruned (path : list nat) (lower_bound : nat)
| CandidatePruned (path : list nat) (stable_id lower_bound : nat)
| ExactCandidate
    (path : list nat)
    (stable_id candidate_bound : nat)
    (exact : option nat)
    (survived : bool).

Definition evidence_path (record : range_evidence) : list nat :=
  match record with
  | PrefixPruned path _
  | SubtreePruned path _
  | TerminalPruned path _
  | CandidatePruned path _ _
  | ExactCandidate path _ _ _ _ => path
  end.

(** K1, K2, terminal, and K4 pruning require a strict lower-bound
    separation from the closed cutoff.  K3 is reached only after K4 admits
    the candidate.  A returned exact score is present exactly for a survivor;
    cutoff abandonment is represented by [None,false]. *)
Definition valid_evidence (cutoff : nat) (record : range_evidence) : Prop :=
  match record with
  | PrefixPruned _ lower_bound
  | SubtreePruned _ lower_bound
  | TerminalPruned _ lower_bound
  | CandidatePruned _ _ lower_bound => cutoff < lower_bound
  | ExactCandidate _ _ candidate_bound exact survived =>
      candidate_bound <= cutoff /\
      match exact, survived with
      | Some score, true => score <= cutoff
      | None, false => True
      | _, _ => False
      end
  end.

Definition exact_survivor_id (record : range_evidence) : option nat :=
  match record with
  | ExactCandidate _ stable_id _ (Some _) true => Some stable_id
  | _ => None
  end.

Definition survivor_ids (records : list range_evidence) : list nat :=
  fold_right
    (fun record accumulated =>
      match exact_survivor_id record with
      | Some stable_id => stable_id :: accumulated
      | None => accumulated
      end)
    [] records.

(** ** Exact logical resource accounting *)

Record certificate_limits : Type := {
  limit_records : nat;
  limit_path_bytes : nat;
  limit_work_units : nat;
  limit_witness_bytes : nat
}.

Record certificate_usage : Type := {
  usage_work_units : nat;
  usage_witness_bytes : nat
}.

Definition total_path_bytes (records : list range_evidence) : nat :=
  fold_right (fun record total => length (evidence_path record) + total) 0 records.

Definition expected_witness_bytes
    (query_word_bytes record_header_bytes : nat)
    (binding : certificate_binding)
    (records : list range_evidence) : nat :=
  query_word_bytes * length (binding_query_words binding)
  + record_header_bytes * length records
  + total_path_bytes records.

Definition within_limits
    (query_word_bytes record_header_bytes : nat)
    (limits : certificate_limits)
    (binding : certificate_binding)
    (records : list range_evidence)
    (usage : certificate_usage) : Prop :=
  length records <= limit_records limits /\
  total_path_bytes records <= limit_path_bytes limits /\
  usage_work_units usage <= limit_work_units limits /\
  usage_witness_bytes usage =
    expected_witness_bytes query_word_bytes record_header_bytes binding records /\
  usage_witness_bytes usage <= limit_witness_bytes limits.

Record range_certificate : Type := {
  certificate_binding_value : certificate_binding;
  certificate_evidence : list range_evidence;
  certificate_usage_value : certificate_usage
}.

(** The production verifier recomputes [expected] from the captured exact
    index.  Structural equality therefore includes decision kind, path,
    stable ID, bound, exact score, survivor flag, and canonical order. *)
Definition replays
    (query_word_bytes record_header_bytes : nat)
    (limits : certificate_limits)
    (expected_binding : certificate_binding)
    (expected : list range_evidence)
    (candidate : range_certificate) : Prop :=
  certificate_binding_value candidate = expected_binding /\
  certificate_evidence candidate = expected /\
  Forall (valid_evidence (binding_cutoff expected_binding)) expected /\
  within_limits query_word_bytes record_header_bytes limits
    expected_binding expected (certificate_usage_value candidate).

Theorem replay_binds_exact_query_cutoff_and_snapshot : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  replays query_word_bytes record_header_bytes limits
    expected_binding expected candidate ->
  certificate_binding_value candidate = expected_binding.
Proof. intros; now destruct H as [Hbinding _]. Qed.

Theorem replay_reproduces_canonical_evidence : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  replays query_word_bytes record_header_bytes limits
    expected_binding expected candidate ->
  certificate_evidence candidate = expected.
Proof. intros; now destruct H as [_ [Hevidence _]]. Qed.

Theorem replay_validates_every_k1_through_k4_decision : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  replays query_word_bytes record_header_bytes limits
    expected_binding expected candidate ->
  Forall (valid_evidence (binding_cutoff expected_binding))
    (certificate_evidence candidate).
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding
    expected candidate Hreplay.
  destruct Hreplay as [_ [Hevidence [Hvalid _]]].
  now rewrite Hevidence.
Qed.

Theorem replay_survivors_are_exactly_recomputed_survivors : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  replays query_word_bytes record_header_bytes limits
    expected_binding expected candidate ->
  survivor_ids (certificate_evidence candidate) = survivor_ids expected.
Proof.
  intros. apply replay_reproduces_canonical_evidence in H. now rewrite H.
Qed.

(** Any alteration that makes the supplied evidence unequal to the exact
    recomputation is rejected.  This one theorem covers altered paths, bounds,
    stable IDs, decision tags, exact scores, survivor flags, and ordering. *)
Theorem any_evidence_mutation_is_rejected : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  certificate_evidence candidate <> expected ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hdifferent Hreplay.
  apply Hdifferent.
  now apply (replay_reproduces_canonical_evidence
    query_word_bytes record_header_bytes limits expected_binding expected candidate).
Qed.

Theorem any_binding_mutation_is_rejected : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  certificate_binding_value candidate <> expected_binding ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hdifferent Hreplay.
  apply Hdifferent.
  now apply (replay_binds_exact_query_cutoff_and_snapshot
    query_word_bytes record_header_bytes limits expected_binding expected candidate).
Qed.

Theorem replay_never_exceeds_any_declared_ceiling : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  replays query_word_bytes record_header_bytes limits
    expected_binding expected candidate ->
  length (certificate_evidence candidate) <= limit_records limits /\
  total_path_bytes (certificate_evidence candidate) <= limit_path_bytes limits /\
  usage_work_units (certificate_usage_value candidate) <= limit_work_units limits /\
  usage_witness_bytes (certificate_usage_value candidate) <= limit_witness_bytes limits.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hreplay.
  destruct Hreplay as [_ [Hevidence [_ Hlimits]]].
  destruct Hlimits as [Hrecords [Hpaths [Hwork [_ Hwitness]]]].
  subst expected. now repeat split.
Qed.

Theorem record_limit_violation_fails_closed : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  limit_records limits < length expected ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hover Hreplay.
  destruct Hreplay as [_ [_ [_ [Hrecords _]]]]. lia.
Qed.

Theorem path_limit_violation_fails_closed : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  limit_path_bytes limits < total_path_bytes expected ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hover Hreplay.
  destruct Hreplay as [_ [_ [_ [_ [Hpaths _]]]]]. lia.
Qed.

Theorem work_limit_violation_fails_closed : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  limit_work_units limits < usage_work_units (certificate_usage_value candidate) ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hover Hreplay.
  destruct Hreplay as [_ [_ [_ [_ [_ [Hwork _]]]]]]. lia.
Qed.

Theorem witness_limit_violation_fails_closed : forall
    query_word_bytes record_header_bytes limits expected_binding expected candidate,
  limit_witness_bytes limits < usage_witness_bytes (certificate_usage_value candidate) ->
  ~ replays query_word_bytes record_header_bytes limits
      expected_binding expected candidate.
Proof.
  intros query_word_bytes record_header_bytes limits expected_binding expected
    candidate Hover Hreplay.
  destruct Hreplay as [_ [_ [_ [_ [_ [_ [_ Hwitness]]]]]]]. lia.
Qed.

(** ** Deterministic construction and completion phase *)

Definition canonical_certificate
    (binding : certificate_binding)
    (records : list range_evidence)
    (usage : certificate_usage) : range_certificate := {|
  certificate_binding_value := binding;
  certificate_evidence := records;
  certificate_usage_value := usage
|}.

Theorem equal_inputs_construct_equal_certificates : forall binding records usage,
  canonical_certificate binding records usage =
  canonical_certificate binding records usage.
Proof. reflexivity. Qed.

Inductive traversal_phase : Type :=
| Running
| Exhausted
| Failed.

Definition issue_certificate
    (phase : traversal_phase)
    (certificate : range_certificate) : option range_certificate :=
  match phase with
  | Exhausted => Some certificate
  | Running | Failed => None
  end.

Theorem certificate_is_issued_only_after_exhaustion : forall phase certificate issued,
  issue_certificate phase certificate = Some issued ->
  phase = Exhausted /\ issued = certificate.
Proof.
  intros [| |] certificate issued Hissued; simpl in Hissued;
    try discriminate.
  inversion Hissued; now split.
Qed.

Theorem running_query_cannot_issue_complete_empty : forall certificate,
  issue_certificate Running certificate = None.
Proof. reflexivity. Qed.

Theorem failed_query_cannot_issue_complete_empty : forall certificate,
  issue_certificate Failed certificate = None.
Proof. reflexivity. Qed.
