(** * Per-original threshold coverage and equality-slice rank floors

    This exact-natural finite model treats each bucket slot as an original.
    [search_candidates] is S_Q(c): it retains every enumerated original whose
    certified per-original lower bound is at most the inclusive cutoff.
    Soundness requires that each bound is no greater than that original's
    exact cost. A representative for a bucket is insufficient.

    The conditioned tie floor below is the minimum tie key among the exact
    equality-cost slice of a fully enumerated candidate region. Its use as
    an executable certificate additionally requires a producer to establish
    that region's cost floor, complete snapshot enumeration, exact-score or
    sound equality-slice evidence, and source tie-key identity. The strict
    cost cut when the equality slice is empty has the same completeness
    premise. No binary64, quantizer, Rust traversal, or resource behavior is
    inferred from these natural-number theorems. *)

From Stdlib Require Import Arith Arith.Compare_dec Lia List.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder.
Import ListNotations.

Record original_occurrence := {
  original_bucket : nat;
  original_slot : nat;
  original_exact_cost : nat;
  original_lower_bound : nat;
  original_tie : nat
}.

Definition occurrence_key (original : original_occurrence) : nat * nat :=
  (original_bucket original, original_slot original).

Definition unique_original_keys (all : list original_occurrence) : Prop :=
  NoDup (map occurrence_key all).

Definition per_original_bounds_sound (all : list original_occurrence) : Prop :=
  forall original, In original all ->
    original_lower_bound original <= original_exact_cost original.

Definition exact_eligible (all : list original_occurrence) (cutoff : nat)
    : list original_occurrence :=
  filter (fun original => Nat.leb (original_exact_cost original) cutoff) all.

Definition search_candidates (all : list original_occurrence) (cutoff : nat)
    : list original_occurrence :=
  filter (fun original => Nat.leb (original_lower_bound original) cutoff) all.

Theorem every_eligible_original_is_searched : forall all cutoff original,
  per_original_bounds_sound all ->
  In original (exact_eligible all cutoff) ->
  In original (search_candidates all cutoff).
Proof.
  intros all cutoff original Hsound Heligible.
  apply filter_In in Heligible as [Hall Hcost].
  apply Nat.leb_le in Hcost.
  apply filter_In; split; [exact Hall |].
  apply Nat.leb_le.
  pose proof (Hsound original Hall); lia.
Qed.

Corollary exact_cost_at_cutoff_is_in_s_q : forall all cutoff original,
  per_original_bounds_sound all ->
  In original all ->
  original_exact_cost original <= cutoff ->
  In original (search_candidates all cutoff).
Proof.
  intros all cutoff original Hsound Hall Hcost.
  apply every_eligible_original_is_searched; [exact Hsound |].
  apply filter_In; split; [exact Hall | now apply Nat.leb_le].
Qed.

Theorem search_candidates_are_enumerated_originals : forall all cutoff,
  incl (search_candidates all cutoff) all.
Proof.
  intros all cutoff original Hcandidate.
  apply filter_In in Hcandidate as [Hall _].
  exact Hall.
Qed.

Definition equality_slice (region : list original_occurrence) (floor : nat)
    : list original_occurrence :=
  filter (fun original => Nat.eqb (original_exact_cost original) floor) region.

Fixpoint minimum_tie_from (seed : nat) (rest : list original_occurrence)
    : nat :=
  match rest with
  | [] => seed
  | original :: tail =>
      Nat.min (original_tie original) (minimum_tie_from seed tail)
  end.

Definition equality_slice_minimum_tie (slice : list original_occurrence)
    : option nat :=
  match slice with
  | [] => None
  | first :: rest =>
      Some (minimum_tie_from (original_tie first) rest)
  end.

Lemma minimum_tie_from_le_seed : forall seed rest,
  minimum_tie_from seed rest <= seed.
Proof.
  intros seed rest; induction rest as [|original tail IH]; simpl; [lia |].
  eapply Nat.le_trans; [apply Nat.le_min_r | exact IH].
Qed.

Lemma minimum_tie_from_le_member : forall seed rest original,
  In original rest ->
  minimum_tie_from seed rest <= original_tie original.
Proof.
  intros seed rest; induction rest as [|head tail IH];
    intros original Hin; simpl in Hin; [contradiction |].
  simpl. destruct Hin as [Hequal | Hin].
  - subst original. apply Nat.le_min_l.
  - eapply Nat.le_trans; [apply Nat.le_min_r | now apply IH].
Qed.

Lemma minimum_tie_from_is_seed_or_member : forall seed rest,
  minimum_tie_from seed rest = seed \/
  exists original, In original rest /\
    minimum_tie_from seed rest = original_tie original.
Proof.
  intros seed rest; induction rest as [|head tail IH]; simpl.
  - now left.
  - destruct (le_dec (original_tie head)
      (minimum_tie_from seed tail)) as [Hleft | Hright].
    + rewrite Nat.min_l by lia.
      right; exists head; split; [now left | reflexivity].
    + rewrite Nat.min_r by lia.
      destruct IH as [Hseed | [original [Hin Hequal]]].
      * now left.
      * right; exists original; split; [now right | exact Hequal].
Qed.

Lemma equality_slice_minimum_tie_bounds_each_member :
  forall slice floor original,
    equality_slice_minimum_tie slice = Some floor ->
    In original slice -> floor <= original_tie original.
Proof.
  intros [|first rest] floor original Hminimum Hin;
    simpl in Hminimum, Hin; [discriminate |].
  inversion Hminimum; subst floor.
  destruct Hin as [Hequal | Hin].
  - subst original. apply minimum_tie_from_le_seed.
  - now apply minimum_tie_from_le_member.
Qed.

Lemma equality_slice_minimum_tie_is_attained :
  forall slice floor,
    equality_slice_minimum_tie slice = Some floor ->
    exists original, In original slice /\ original_tie original = floor.
Proof.
  intros [|first rest] floor Hminimum; simpl in Hminimum;
    [discriminate |].
  inversion Hminimum; subst floor.
  destruct (minimum_tie_from_is_seed_or_member
    (original_tie first) rest) as
      [Hseed | [original [Hin Hequal]]].
  - exists first; split; [now left | symmetry; exact Hseed].
  - exists original; split; [now right | symmetry; exact Hequal].
Qed.

Lemma nonempty_equality_slice_has_minimum_tie : forall slice,
  slice <> [] ->
  exists floor, equality_slice_minimum_tie slice = Some floor.
Proof.
  intros [|first rest] Hnonempty; [contradiction |].
  eexists; reflexivity.
Qed.

Definition region_cost_floor (region : list original_occurrence)
    (floor : nat) : Prop :=
  forall original, In original region ->
    floor <= original_exact_cost original.

(** This conclusion is the semantic [RankConditioned] denotation. It is
    deliberately conditional on the exact equality slice being complete. *)
Theorem equality_slice_minimum_gives_conditioned_tie_floor :
  forall region cost_floor tie_floor,
    region_cost_floor region cost_floor ->
    equality_slice_minimum_tie (equality_slice region cost_floor) =
      Some tie_floor ->
    forall original, In original region ->
      certificate_denotes natural_certificate_order
        natural_certificate_order
        (RankConditioned cost_floor tie_floor)
        (original_exact_cost original, original_tie original).
Proof.
  intros region cost_floor tie_floor Hfloor Hminimum original Hin.
  simpl. split; [apply Hfloor; exact Hin |].
  intro Hequal.
  assert (Hslice : In original (equality_slice region cost_floor)).
  { apply filter_In; split; [exact Hin |].
    apply Nat.eqb_eq. symmetry; exact Hequal. }
  eapply equality_slice_minimum_tie_bounds_each_member; eauto.
Qed.

Corollary nonempty_equality_slice_produces_conditioned_floor :
  forall region cost_floor,
    region_cost_floor region cost_floor ->
    equality_slice region cost_floor <> [] ->
    exists tie_floor,
      equality_slice_minimum_tie (equality_slice region cost_floor) =
        Some tie_floor /\
      forall original, In original region ->
        certificate_denotes natural_certificate_order
          natural_certificate_order
          (RankConditioned cost_floor tie_floor)
          (original_exact_cost original, original_tie original).
Proof.
  intros region cost_floor Hfloor Hnonempty.
  destruct (nonempty_equality_slice_has_minimum_tie
    _ Hnonempty) as [tie_floor Hminimum].
  exists tie_floor; split; [exact Hminimum |].
  eapply equality_slice_minimum_gives_conditioned_tie_floor; eauto.
Qed.

Theorem empty_equality_slice_gives_strict_cost_cut :
  forall region cost_floor,
    region_cost_floor region cost_floor ->
    equality_slice region cost_floor = [] ->
    forall original, In original region ->
      certificate_denotes natural_certificate_order
        natural_certificate_order
        (RankStrictCost cost_floor)
        (original_exact_cost original, original_tie original).
Proof.
  intros region cost_floor Hfloor Hempty original Hin.
  simpl.
  pose proof (Hfloor original Hin) as Hlower.
  destruct (Nat.eq_dec (original_exact_cost original) cost_floor)
    as [Hequal | Hneq]; [|lia].
  assert (Hslice : In original (equality_slice region cost_floor)).
  { apply filter_In; split; [exact Hin |].
    apply Nat.eqb_eq; exact Hequal. }
  rewrite Hempty in Hslice; contradiction.
Qed.

(** A partial scan has [Unknown] semantics. [Empty] requires a proof that
    the full source region is empty; seeing no member so far is insufficient. *)
Inductive enumeration_summary :=
| EnumerationUnknown
| EnumerationEmpty
| EnumerationKnown (members : list original_occurrence).

Definition enumeration_summary_sound
    (all : list original_occurrence) (summary : enumeration_summary) : Prop :=
  match summary with
  | EnumerationUnknown => True
  | EnumerationEmpty => all = []
  | EnumerationKnown members =>
      forall original, In original members <-> In original all
  end.

Theorem unknown_summary_is_sound_for_any_enumeration : forall all,
  enumeration_summary_sound all EnumerationUnknown.
Proof. intros; exact I. Qed.

Theorem empty_summary_requires_full_absence : forall all,
  enumeration_summary_sound all EnumerationEmpty <-> all = [].
Proof. intros; reflexivity. Qed.

Corollary incomplete_nonempty_region_cannot_claim_empty : forall all,
  all <> [] -> ~ enumeration_summary_sound all EnumerationEmpty.
Proof. intros all Hnonempty Hempty; apply Hnonempty; exact Hempty. Qed.

Definition bucket_first : original_occurrence :=
  {| original_bucket := 4; original_slot := 0;
     original_exact_cost := 8; original_lower_bound := 0;
     original_tie := 10 |}.

Definition bucket_second : original_occurrence :=
  {| original_bucket := 4; original_slot := 1;
     original_exact_cost := 1; original_lower_bound := 0;
     original_tie := 11 |}.

Example two_slots_one_bucket_are_distinct :
  original_bucket bucket_first = original_bucket bucket_second /\
  occurrence_key bucket_first <> occurrence_key bucket_second.
Proof. split; compute; congruence. Qed.

Example representative_only_misses_an_eligible_original :
  per_original_bounds_sound [bucket_first; bucket_second] /\
  In bucket_second (exact_eligible [bucket_first; bucket_second] 2) /\
  In bucket_second (search_candidates [bucket_first; bucket_second] 2) /\
  ~ In bucket_second [bucket_first] /\
  ~ enumeration_summary_sound [bucket_first; bucket_second]
      (EnumerationKnown [bucket_first]).
Proof.
  repeat split.
  - intros original Hin.
    simpl in Hin.
    destruct Hin as [Hequal | [Hequal | Hfalse]].
    + rewrite <- Hequal; simpl; lia.
    + rewrite <- Hequal; simpl; lia.
    + contradiction.
  - simpl; auto.
  - simpl; auto.
  - simpl; intros [Hequal | Hfalse]; [discriminate | contradiction].
  - intro Hknown.
    specialize (Hknown bucket_second).
    assert (Hlisted : In bucket_second [bucket_first]).
    { apply (proj2 Hknown). simpl; auto. }
    simpl in Hlisted.
    destruct Hlisted as [Hequal | Hfalse]; [discriminate | contradiction].
Qed.
