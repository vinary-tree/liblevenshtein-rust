(** Assumption-free proof island for the complete elastic snapshot protocol.

    This file proves only abstract phase ordering and finite key-set facts. It
    does not model SHA-256 collision resistance, operating-system rename
    semantics, PersistentARTrie, or whole-program correspondence. *)

From Stdlib Require Import List.
From Stdlib Require Import Sorting.Permutation.
Import ListNotations.

Inductive decode_phase : Type :=
| Framed
| ChecksumVerified
| SemanticallyDecoded
| BijectionVerified
| Accepted.

Definition checksum_holds (phase : decode_phase) : bool :=
  match phase with
  | Framed => false
  | _ => true
  end.

Definition semantics_available (phase : decode_phase) : bool :=
  match phase with
  | Framed | ChecksumVerified => false
  | _ => true
  end.

Definition acceptance_available (phase : decode_phase) : bool :=
  match phase with
  | Accepted => true
  | _ => false
  end.

Theorem semantics_require_checksum :
  forall phase,
    semantics_available phase = true -> checksum_holds phase = true.
Proof. destruct phase; simpl; congruence. Qed.

Theorem acceptance_requires_checksum :
  forall phase,
    acceptance_available phase = true -> checksum_holds phase = true.
Proof. destruct phase; simpl; congruence. Qed.

Inductive publication_phase : Type :=
| Absent
| Staging
| Sealed
| GenerationPublished
| ManifestPublished.

Definition generation_is_sealed (phase : publication_phase) : bool :=
  match phase with
  | Sealed | GenerationPublished | ManifestPublished => true
  | _ => false
  end.

Definition manifest_is_visible (phase : publication_phase) : bool :=
  match phase with
  | ManifestPublished => true
  | _ => false
  end.

Theorem visible_manifest_names_sealed_generation :
  forall phase,
    manifest_is_visible phase = true -> generation_is_sealed phase = true.
Proof. destruct phase; simpl; congruence. Qed.

Definition exact_key_bijection {A : Type} (buckets terminals : list A) : Prop :=
  NoDup buckets /\ NoDup terminals /\
  forall key, In key buckets <-> In key terminals.

Theorem exact_key_bijection_permutation :
  forall (A : Type) (buckets terminals : list A),
    exact_key_bijection buckets terminals -> Permutation buckets terminals.
Proof.
  intros A buckets terminals [Hb [Ht Heq]].
  apply NoDup_Permutation; assumption.
Qed.

Theorem exact_key_bijection_cardinality :
  forall (A : Type) (buckets terminals : list A),
    exact_key_bijection buckets terminals ->
    length buckets = length terminals.
Proof.
  intros A buckets terminals H.
  apply Permutation_length.
  now apply exact_key_bijection_permutation.
Qed.

Definition semantic_identity_equal (left right : list nat) : Prop := left = right.

Theorem changed_manifest_invalidates_semantic_identity :
  forall left right,
    left <> right -> ~ semantic_identity_equal left right.
Proof. unfold semantic_identity_equal; auto. Qed.
