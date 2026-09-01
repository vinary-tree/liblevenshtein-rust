(** * Evidence typing for bounded approximate MSM search

    PAA candidate generation is heuristic.  Exact MSM reranking establishes
    score soundness for emitted candidates, but recall is certified only when
    both the candidate pool and exact decisions cover the captured index.
    This assumption-free model pins the tag-construction and score-authority
    laws implemented by [ApproxMsmSearchOutcome].
*)

From Stdlib Require Import Arith Bool List Lia.
Import ListNotations.

Record coverage : Type := {
  indexed_entries : nat;
  candidate_entries : nat;
  exact_reranked : nat
}.

Definition recall_certificate (c : coverage) : bool :=
  Nat.eqb (candidate_entries c) (indexed_entries c) &&
  Nat.eqb (exact_reranked c) (indexed_entries c).

Lemma recall_certificate_reflects_full_exact_coverage : forall c,
  recall_certificate c = true <->
  candidate_entries c = indexed_entries c /\
  exact_reranked c = indexed_entries c.
Proof.
  intros c; unfold recall_certificate.
  rewrite andb_true_iff.
  repeat rewrite Nat.eqb_eq.
  reflexivity.
Qed.

Inductive successful_outcome (A : Type) : Type :=
  | Exhaustive : A -> coverage -> successful_outcome A
  | Advisory : A -> coverage -> successful_outcome A.

Arguments Exhaustive {A} _ _.
Arguments Advisory {A} _ _.

Definition classify_success {A : Type} (result : A) (c : coverage) :
    successful_outcome A :=
  if recall_certificate c
  then Exhaustive result c
  else Advisory result c.

(** The implementation can construct the exhaustive tag exactly when its
    coverage certificate reports full candidate and exact-decision coverage. *)
Theorem exhaustive_tag_if_and_only_if_full_reranking : forall A (result : A) c,
  (exists returned coverage_value,
      classify_success result c = Exhaustive returned coverage_value) <->
  candidate_entries c = indexed_entries c /\
  exact_reranked c = indexed_entries c.
Proof.
  intros A result c.
  destruct (recall_certificate c) eqn:Hcertificate.
  - pose proof Hcertificate as Hfull.
    apply recall_certificate_reflects_full_exact_coverage in Hfull.
    split.
    + intros; exact Hfull.
    + intros; exists result, c; unfold classify_success; rewrite Hcertificate; reflexivity.
  - split.
    + intros [returned [coverage_value Heq]].
      unfold classify_success in Heq; rewrite Hcertificate in Heq; discriminate.
    + intros Hfull.
      apply recall_certificate_reflects_full_exact_coverage in Hfull.
      rewrite Hfull in Hcertificate; discriminate.
Qed.

Definition proves_recall {A : Type} (outcome : successful_outcome A) : bool :=
  match outcome with
  | Exhaustive _ c => recall_certificate c
  | Advisory _ _ => false
  end.

Theorem classified_success_proves_recall_if_and_only_if_full_reranking :
  forall A (result : A) c,
    proves_recall (classify_success result c) = true <->
    candidate_entries c = indexed_entries c /\
    exact_reranked c = indexed_entries c.
Proof.
  intros A result c.
  destruct (recall_certificate c) eqn:Hcertificate.
  - unfold classify_success; rewrite Hcertificate; simpl; rewrite Hcertificate.
    split.
    + intros _.
      apply recall_certificate_reflects_full_exact_coverage in Hcertificate.
      exact Hcertificate.
    + intros _; reflexivity.
  - unfold classify_success; rewrite Hcertificate; simpl.
    split; [discriminate |].
    intros Hfull.
    apply recall_certificate_reflects_full_exact_coverage in Hfull.
    rewrite Hfull in Hcertificate; discriminate.
Qed.

Theorem advisory_never_proves_recall : forall A (result : A) c,
  proves_recall (Advisory result c) = false.
Proof. reflexivity. Qed.

(** An empty advisory result remains non-evidence.  The payload cardinality is
    deliberately irrelevant to [proves_recall]. *)
Corollary empty_advisory_never_proves_absence : forall A c,
  proves_recall (Advisory (@nil A) c) = false.
Proof. reflexivity. Qed.

Record neighbor : Type := {
  neighbor_index : nat;
  neighbor_distance : nat
}.

Definition emit_exact (exact_distance : nat -> nat) (index : nat) : neighbor :=
  {| neighbor_index := index;
     neighbor_distance := exact_distance index |}.

Definition score_sound (exact_distance : nat -> nat) (candidate : neighbor) : Prop :=
  neighbor_distance candidate = exact_distance (neighbor_index candidate).

Theorem exact_verifier_is_score_authority : forall exact_distance index,
  score_sound exact_distance (emit_exact exact_distance index).
Proof. reflexivity. Qed.

Theorem every_mapped_emission_is_exact : forall exact_distance indices candidate,
  In candidate (map (emit_exact exact_distance) indices) ->
  score_sound exact_distance candidate.
Proof.
  intros exact_distance indices candidate Hin.
  apply in_map_iff in Hin.
  destruct Hin as [index [Heq _]].
  subst candidate; apply exact_verifier_is_score_authority.
Qed.

(** A proper candidate pool cannot satisfy the exhaustive certificate,
    independently of how many heuristic features were inspected. *)
Theorem proper_candidate_pool_has_no_recall_certificate : forall c,
  candidate_entries c < indexed_entries c ->
  recall_certificate c = false.
Proof.
  intros c Hproper.
  unfold recall_certificate.
  apply andb_false_intro1.
  apply Nat.eqb_neq; lia.
Qed.
