(** * Original identity and natural-rank tie instances

    The exact-natural cost profile reuses the lawful lexicographic comparison
    from [CertifiedGenericRankOrder]. An elastic original is a captured
    dictionary path plus a terminal-bucket slot. Physical compressed-node
    identity is deliberately absent from this key: distinct paths may share
    a node, and one terminal bucket may retain several originals. A source
    encounter ordinal can serve as the natural tie key only when it is
    injective on admitted path/slot originals.

    A timestamped TWED original is represented abstractly. Its episode ID
    can serve as the tie key only when IDs are unique on the admitted captured
    originals. The executable comparator must agree with the specified
    natural lexicographic comparison (or satisfy a separately proved lawful
    binary64 rank-order contract).

    Remaining source obligations: capture a fixed dictionary revision;
    prove path/slot enumeration and ordinal assignment against Rust buckets;
    prove episode-ID uniqueness, retention, and no duplicate/missing originals;
    bind exact scores and the selected numeric authority, including the
    actual f64::total_cmp semantics for timestamped TWED; prove the Rust
    comparator, heap, result sorting, and cutoff/partial-result behaviors
    refine those ranks. None of these Rust or binary64 facts is asserted
    here. *)

From Stdlib Require Import Arith Lia List.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedGenericRankOrder CertifiedRankCertificateTypes.
Import ListNotations.

Definition elastic_occurrence (Label : Type) : Type :=
  (list Label * nat)%type.

Definition occurrence_path {Label : Type}
    (original : elastic_occurrence Label) : list Label := fst original.

Definition occurrence_slot {Label : Type}
    (original : elastic_occurrence Label) : nat := snd original.

Theorem elastic_path_slot_identifies_original :
  forall (Label : Type) (left right : elastic_occurrence Label),
    occurrence_path left = occurrence_path right ->
    occurrence_slot left = occurrence_slot right ->
    left = right.
Proof.
  intros Label [left_path left_slot] [right_path right_slot]
    Hpath Hslot.
  simpl in Hpath, Hslot. subst. reflexivity.
Qed.

Definition natural_original_rank {Original : Type}
    (score tie : Original -> nat) (original : Original) : nat * nat :=
  (score original, tie original).

Theorem natural_original_rank_comparison_sound :
  rank_comparison_sound natural_certificate_order
    natural_certificate_order
    NaturalRankInstance.compare_natural_rank.
Proof. exact NaturalRankInstance.natural_comparison_is_lawful. Qed.

(** A source comparison function is qualified by a pointwise equality
    premise. This is an obligation to prove about the chosen implementation,
    not an assertion that Rust presently satisfies it. *)
Theorem assumed_source_comparison_is_lawful :
  forall source_compare,
    (forall left right,
      source_compare left right =
        NaturalRankInstance.compare_natural_rank left right) ->
    rank_comparison_sound natural_certificate_order
      natural_certificate_order source_compare.
Proof.
  intros source_compare Hagrees left right.
  rewrite Hagrees.
  apply natural_original_rank_comparison_sound.
Qed.

Theorem injective_tie_makes_equal_comparison_identify_original :
  forall (Original : Type) (score tie : Original -> nat)
    (admitted : Original -> Prop) left right,
    (forall first second,
      admitted first -> admitted second ->
      tie first = tie second -> first = second) ->
    admitted left -> admitted right ->
    NaturalRankInstance.compare_natural_rank
      (natural_original_rank score tie left)
      (natural_original_rank score tie right) = Eq ->
    left = right.
Proof.
  intros Original score tie admitted left right Hunique
    Hleft Hright Hcomparison.
  unfold natural_original_rank in Hcomparison.
  unfold NaturalRankInstance.compare_natural_rank in Hcomparison.
  simpl in Hcomparison.
  destruct (Nat.compare (score left) (score right))
    eqn:Hcost; try discriminate.
  apply Nat.compare_eq_iff in Hcomparison.
  eapply Hunique; eauto.
Qed.

Section ElasticRank.
  Context {Label : Type}.
  Variable admitted : elastic_occurrence Label -> Prop.
  Variable score_of : elastic_occurrence Label -> nat.
  Variable encounter_ordinal : elastic_occurrence Label -> nat.

  (** Must be established from one captured original enumeration. Merely
      assigning a compressed-node ID does not satisfy this premise. *)
  Variable ordinal_unique : forall left right,
    admitted left -> admitted right ->
    encounter_ordinal left = encounter_ordinal right -> left = right.

  Definition elastic_rank (original : elastic_occurrence Label) : nat * nat :=
    natural_original_rank score_of encounter_ordinal original.

  Lemma elastic_rank_uses_encounter_ordinal : forall original,
    snd (elastic_rank original) = encounter_ordinal original.
  Proof. reflexivity. Qed.

  Theorem elastic_equal_comparison_identifies_occurrence :
    forall left right,
      admitted left -> admitted right ->
      NaturalRankInstance.compare_natural_rank
        (elastic_rank left) (elastic_rank right) = Eq ->
      left = right.
  Proof.
    intros left right Hleft Hright Hcompare.
    unfold elastic_rank in Hcompare.
    eapply injective_tie_makes_equal_comparison_identify_original;
      eauto.
  Qed.

  Theorem elastic_rank_comparison_is_lawful :
    forall left right,
      match NaturalRankInstance.compare_natural_rank
        (elastic_rank left) (elastic_rank right) with
      | Eq => generic_rank_equiv natural_certificate_order
          natural_certificate_order (elastic_rank left) (elastic_rank right)
      | Lt => generic_rank_strict natural_certificate_order
          natural_certificate_order (elastic_rank left) (elastic_rank right)
      | Gt => generic_rank_strict natural_certificate_order
          natural_certificate_order (elastic_rank right) (elastic_rank left)
      end.
  Proof. intros; apply natural_original_rank_comparison_sound. Qed.

  Theorem elastic_assumed_source_equal_identifies_occurrence :
    forall source_compare,
      (forall first second,
        source_compare first second =
          NaturalRankInstance.compare_natural_rank first second) ->
      forall left right,
        admitted left -> admitted right ->
        source_compare (elastic_rank left) (elastic_rank right) = Eq ->
        left = right.
  Proof.
    intros source_compare Hagrees left right Hleft Hright Hcompare.
    rewrite Hagrees in Hcompare.
    eapply elastic_equal_comparison_identifies_occurrence; eauto.
  Qed.
End ElasticRank.

Section TimestampedEpisodeRank.
  Context {Episode : Type}.
  Variable admitted : Episode -> Prop.
  Variable score_of : Episode -> nat.
  Variable episode_id : Episode -> nat.

  Variable episode_ids_unique : forall left right,
    admitted left -> admitted right ->
    episode_id left = episode_id right -> left = right.

  Definition timestamped_rank (episode : Episode) : nat * nat :=
    natural_original_rank score_of episode_id episode.

  Lemma timestamped_rank_uses_episode_id : forall episode,
    snd (timestamped_rank episode) = episode_id episode.
  Proof. reflexivity. Qed.

  Theorem timestamped_equal_comparison_identifies_episode :
    forall left right,
      admitted left -> admitted right ->
      NaturalRankInstance.compare_natural_rank
        (timestamped_rank left) (timestamped_rank right) = Eq ->
      left = right.
  Proof.
    intros left right Hleft Hright Hcompare.
    unfold timestamped_rank in Hcompare.
    eapply injective_tie_makes_equal_comparison_identify_original;
      eauto.
  Qed.

  Theorem timestamped_rank_comparison_is_lawful :
    forall left right,
      match NaturalRankInstance.compare_natural_rank
        (timestamped_rank left) (timestamped_rank right) with
      | Eq => generic_rank_equiv natural_certificate_order
          natural_certificate_order (timestamped_rank left)
          (timestamped_rank right)
      | Lt => generic_rank_strict natural_certificate_order
          natural_certificate_order (timestamped_rank left)
          (timestamped_rank right)
      | Gt => generic_rank_strict natural_certificate_order
          natural_certificate_order (timestamped_rank right)
          (timestamped_rank left)
      end.
  Proof. intros; apply natural_original_rank_comparison_sound. Qed.

  Theorem timestamped_assumed_source_equal_identifies_episode :
    forall source_compare,
      (forall first second,
        source_compare first second =
          NaturalRankInstance.compare_natural_rank first second) ->
      forall left right,
        admitted left -> admitted right ->
        source_compare (timestamped_rank left)
          (timestamped_rank right) = Eq ->
        left = right.
  Proof.
    intros source_compare Hagrees left right Hleft Hright Hcompare.
    rewrite Hagrees in Hcompare.
    eapply timestamped_equal_comparison_identifies_episode; eauto.
  Qed.
End TimestampedEpisodeRank.

(** Two originals share one terminal bucket and one compressed node, yet
    occupy different slots and compare in the declared tie order. The node
    model is intentionally constant, so this control cannot silently use
    physical node identity as original identity. *)
Definition collision_first : elastic_occurrence nat := ([7], 0).
Definition collision_second : elastic_occurrence nat := ([7], 1).
Definition control_compressed_node (_ : list nat) : nat := 42.
Definition control_collision_score (_ : elastic_occurrence nat) : nat := 5.
Definition control_collision_tie
    (original : elastic_occurrence nat) : nat := occurrence_slot original.

Example one_bucket_two_originals_remain_distinct :
  occurrence_path collision_first = occurrence_path collision_second /\
  control_compressed_node (occurrence_path collision_first) =
    control_compressed_node (occurrence_path collision_second) /\
  collision_first <> collision_second /\
  NaturalRankInstance.compare_natural_rank
    (natural_original_rank control_collision_score control_collision_tie
      collision_first)
    (natural_original_rank control_collision_score control_collision_tie
      collision_second) = Lt.
Proof. repeat split; compute; congruence. Qed.
