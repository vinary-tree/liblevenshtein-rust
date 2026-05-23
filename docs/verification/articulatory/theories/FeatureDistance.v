(** * Articulatory Feature Distance

    This module defines and proves properties of articulatory feature-based
    distance between phonemes. Unlike standard edit distance, this measures
    phonetic similarity based on articulatory features.

    Key Theorems:
    1. Symmetry: articulatory_distance(a, b) = articulatory_distance(b, a)
    2. Boundedness: 0 <= articulatory_distance(a, b) <= 1
    3. Identity: articulatory_distance(a, a) = 0

    Note: Triangle inequality may NOT hold for articulatory distance.

    Corresponds to: src/phonetic/feature_distance.rs
                    src/transducer/articulatory_costs.rs
*)

Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qabs.
Require Import Coq.micromega.Lia.
Require Import Coq.micromega.Lra.
Import ListNotations.

(** * Articulatory Features *)

(** Place of articulation *)
Inductive Place : Type :=
  | Bilabial    (* p, b, m *)
  | Labiodental (* f, v *)
  | Dental      (* th, dh *)
  | Alveolar    (* t, d, n, s, z, l, r *)
  | Postalveolar (* sh, zh, ch, j *)
  | Palatal     (* y *)
  | Velar       (* k, g, ng *)
  | Glottal     (* h *)
  | UnknownPlace.

(** Manner of articulation *)
Inductive Manner : Type :=
  | Plosive     (* p, b, t, d, k, g *)
  | Nasal       (* m, n, ng *)
  | Fricative   (* f, v, s, z, sh, zh, th, dh, h *)
  | Affricate   (* ch, j *)
  | Approximant (* w, y, r, l *)
  | Lateral     (* l *)
  | UnknownManner.

(** Voicing *)
Inductive Voice : Type :=
  | Voiced    (* b, d, g, v, z, zh, dh, j, m, n, ng, w, y, r, l *)
  | Voiceless (* p, t, k, f, s, sh, th, ch, h *)
  | UnknownVoice.

(** Complete phoneme feature bundle *)
Record PhonemeFeatures : Type := mkFeatures {
  feat_place : Place;
  feat_manner : Manner;
  feat_voice : Voice
}.

(** * Feature Distance Functions *)

(** Distance between places of articulation
    Based on physical proximity in the vocal tract *)
Definition place_distance (p1 p2 : Place) : Q :=
  match p1, p2 with
  | Bilabial, Bilabial => 0
  | Bilabial, Labiodental => 1#10
  | Bilabial, Dental => 2#10
  | Bilabial, Alveolar => 3#10
  | Bilabial, Postalveolar => 4#10
  | Bilabial, Palatal => 5#10
  | Bilabial, Velar => 6#10
  | Bilabial, Glottal => 7#10

  | Labiodental, Labiodental => 0
  | Labiodental, Bilabial => 1#10
  | Labiodental, Dental => 1#10
  | Labiodental, Alveolar => 2#10
  | Labiodental, Postalveolar => 3#10
  | Labiodental, Palatal => 4#10
  | Labiodental, Velar => 5#10
  | Labiodental, Glottal => 6#10

  | Dental, Dental => 0
  | Dental, Bilabial => 2#10
  | Dental, Labiodental => 1#10
  | Dental, Alveolar => 1#10
  | Dental, Postalveolar => 2#10
  | Dental, Palatal => 3#10
  | Dental, Velar => 4#10
  | Dental, Glottal => 5#10

  | Alveolar, Alveolar => 0
  | Alveolar, Bilabial => 3#10
  | Alveolar, Labiodental => 2#10
  | Alveolar, Dental => 1#10
  | Alveolar, Postalveolar => 1#10
  | Alveolar, Palatal => 2#10
  | Alveolar, Velar => 3#10
  | Alveolar, Glottal => 4#10

  | Postalveolar, Postalveolar => 0
  | Postalveolar, Bilabial => 4#10
  | Postalveolar, Labiodental => 3#10
  | Postalveolar, Dental => 2#10
  | Postalveolar, Alveolar => 1#10
  | Postalveolar, Palatal => 1#10
  | Postalveolar, Velar => 2#10
  | Postalveolar, Glottal => 3#10

  | Palatal, Palatal => 0
  | Palatal, Bilabial => 5#10
  | Palatal, Labiodental => 4#10
  | Palatal, Dental => 3#10
  | Palatal, Alveolar => 2#10
  | Palatal, Postalveolar => 1#10
  | Palatal, Velar => 1#10
  | Palatal, Glottal => 2#10

  | Velar, Velar => 0
  | Velar, Bilabial => 6#10
  | Velar, Labiodental => 5#10
  | Velar, Dental => 4#10
  | Velar, Alveolar => 3#10
  | Velar, Postalveolar => 2#10
  | Velar, Palatal => 1#10
  | Velar, Glottal => 1#10

  | Glottal, Glottal => 0
  | Glottal, Bilabial => 7#10
  | Glottal, Labiodental => 6#10
  | Glottal, Dental => 5#10
  | Glottal, Alveolar => 4#10
  | Glottal, Postalveolar => 3#10
  | Glottal, Palatal => 2#10
  | Glottal, Velar => 1#10

  | UnknownPlace, UnknownPlace => 0
  | UnknownPlace, _ => 1
  | _, UnknownPlace => 1
  end.

(** Distance between manners of articulation *)
Definition manner_distance (m1 m2 : Manner) : Q :=
  match m1, m2 with
  | Plosive, Plosive => 0
  | Plosive, Nasal => 2#10
  | Plosive, Fricative => 3#10
  | Plosive, Affricate => 2#10
  | Plosive, Approximant => 4#10
  | Plosive, Lateral => 4#10

  | Nasal, Nasal => 0
  | Nasal, Plosive => 2#10
  | Nasal, Fricative => 3#10
  | Nasal, Affricate => 3#10
  | Nasal, Approximant => 2#10
  | Nasal, Lateral => 2#10

  | Fricative, Fricative => 0
  | Fricative, Plosive => 3#10
  | Fricative, Nasal => 3#10
  | Fricative, Affricate => 1#10
  | Fricative, Approximant => 2#10
  | Fricative, Lateral => 2#10

  | Affricate, Affricate => 0
  | Affricate, Plosive => 2#10
  | Affricate, Nasal => 3#10
  | Affricate, Fricative => 1#10
  | Affricate, Approximant => 3#10
  | Affricate, Lateral => 3#10

  | Approximant, Approximant => 0
  | Approximant, Plosive => 4#10
  | Approximant, Nasal => 2#10
  | Approximant, Fricative => 2#10
  | Approximant, Affricate => 3#10
  | Approximant, Lateral => 1#10

  | Lateral, Lateral => 0
  | Lateral, Plosive => 4#10
  | Lateral, Nasal => 2#10
  | Lateral, Fricative => 2#10
  | Lateral, Affricate => 3#10
  | Lateral, Approximant => 1#10

  | UnknownManner, UnknownManner => 0
  | UnknownManner, _ => 1
  | _, UnknownManner => 1
  end.

(** Distance for voicing *)
Definition voice_distance (v1 v2 : Voice) : Q :=
  match v1, v2 with
  | Voiced, Voiced => 0
  | Voiceless, Voiceless => 0
  | Voiced, Voiceless => 2#10
  | Voiceless, Voiced => 2#10
  | UnknownVoice, UnknownVoice => 0
  | UnknownVoice, _ => 1
  | _, UnknownVoice => 1
  end.

(** * Articulatory Distance *)

(** Feature weights for combining distances *)
Definition place_weight : Q := 4#10.
Definition manner_weight : Q := 4#10.
Definition voice_weight : Q := 2#10.

(** Combined articulatory distance *)
Definition articulatory_distance (f1 f2 : PhonemeFeatures) : Q :=
  place_weight * place_distance (feat_place f1) (feat_place f2) +
  manner_weight * manner_distance (feat_manner f1) (feat_manner f2) +
  voice_weight * voice_distance (feat_voice f1) (feat_voice f2).

(** * Symmetry Proof *)

(** Place distance is symmetric *)
Lemma place_distance_symmetric : forall p1 p2,
  place_distance p1 p2 = place_distance p2 p1.
Proof.
  intros p1 p2.
  destruct p1; destruct p2; reflexivity.
Qed.

(** Manner distance is symmetric *)
Lemma manner_distance_symmetric : forall m1 m2,
  manner_distance m1 m2 = manner_distance m2 m1.
Proof.
  intros m1 m2.
  destruct m1; destruct m2; reflexivity.
Qed.

(** Voice distance is symmetric *)
Lemma voice_distance_symmetric : forall v1 v2,
  voice_distance v1 v2 = voice_distance v2 v1.
Proof.
  intros v1 v2.
  destruct v1; destruct v2; reflexivity.
Qed.

(** Main symmetry theorem *)
Theorem articulatory_symmetric : forall f1 f2,
  articulatory_distance f1 f2 = articulatory_distance f2 f1.
Proof.
  intros f1 f2.
  unfold articulatory_distance.
  rewrite place_distance_symmetric.
  rewrite manner_distance_symmetric.
  rewrite voice_distance_symmetric.
  reflexivity.
Qed.

(** * Identity Proof *)

(** Component identities, including the public char-level behavior where an
    unknown character compared with itself has distance zero before feature
    lookup falls back to maximum distance for unknown non-identical chars. *)
Lemma place_distance_identity : forall p,
  place_distance p p = 0.
Proof.
  intros p. destruct p; reflexivity.
Qed.

Lemma manner_distance_identity : forall m,
  manner_distance m m = 0.
Proof.
  intros m. destruct m; reflexivity.
Qed.

Lemma voice_distance_identity : forall v,
  voice_distance v v = 0.
Proof.
  intros v. destruct v; reflexivity.
Qed.

(** Same phoneme has distance 0 *)
Theorem articulatory_identity : forall f,
  articulatory_distance f f == 0.
Proof.
  intros f.
  unfold articulatory_distance.
  destruct f as [p m v].
  cbn.
  rewrite (place_distance_identity p).
  rewrite (manner_distance_identity m).
  rewrite (voice_distance_identity v).
  ring.
Qed.

(** * Boundedness Proof *)

(** Component distances are bounded *)
Lemma place_distance_bounded : forall p1 p2,
  0 <= place_distance p1 p2 <= 1.
Proof.
  intros p1 p2.
  destruct p1; destruct p2; split; discriminate || reflexivity.
Qed.

Lemma manner_distance_bounded : forall m1 m2,
  0 <= manner_distance m1 m2 <= 1.
Proof.
  intros m1 m2.
  destruct m1; destruct m2; split; discriminate || reflexivity.
Qed.

Lemma voice_distance_bounded : forall v1 v2,
  0 <= voice_distance v1 v2 <= 1.
Proof.
  intros v1 v2.
  destruct v1; destruct v2; split; discriminate || reflexivity.
Qed.

(** Weights sum to 1 *)
Lemma weights_sum_to_one :
  place_weight + manner_weight + voice_weight == 1.
Proof.
  unfold place_weight, manner_weight, voice_weight.
  reflexivity.
Qed.

(** Main boundedness theorem *)
Theorem articulatory_bounded : forall f1 f2,
  0 <= articulatory_distance f1 f2 <= 1.
Proof.
  intros f1 f2.
  unfold articulatory_distance.
  destruct (place_distance_bounded (feat_place f1) (feat_place f2)) as [Hp_lo Hp_hi].
  destruct (manner_distance_bounded (feat_manner f1) (feat_manner f2)) as [Hm_lo Hm_hi].
  destruct (voice_distance_bounded (feat_voice f1) (feat_voice f2)) as [Hv_lo Hv_hi].
  split.
  - (* 0 <= distance *)
    replace 0 with (0 + 0)%Q by reflexivity.
    apply Qplus_le_compat.
    + replace 0 with (0 + 0)%Q by reflexivity.
      apply Qplus_le_compat.
      * apply Qmult_le_0_compat; [unfold place_weight; discriminate | exact Hp_lo].
      * apply Qmult_le_0_compat; [unfold manner_weight; discriminate | exact Hm_lo].
    + apply Qmult_le_0_compat; [unfold voice_weight; discriminate | exact Hv_lo].
  - (* distance <= 1 *)
    setoid_replace 1 with (place_weight * 1 + manner_weight * 1 + voice_weight * 1).
    + apply Qplus_le_compat.
      * apply Qplus_le_compat.
        -- apply Qmult_le_compat_nonneg.
           ++ split; [unfold place_weight; discriminate | apply Qle_refl].
           ++ split; [exact Hp_lo | exact Hp_hi].
        -- apply Qmult_le_compat_nonneg.
           ++ split; [unfold manner_weight; discriminate | apply Qle_refl].
           ++ split; [exact Hm_lo | exact Hm_hi].
      * apply Qmult_le_compat_nonneg.
        -- split; [unfold voice_weight; discriminate | apply Qle_refl].
        -- split; [exact Hv_lo | exact Hv_hi].
    + unfold place_weight, manner_weight, voice_weight. ring.
Qed.

(** * Triangle Inequality Analysis *)

(** This module proves symmetry, identity, and boundedness only.  It does not
    claim a metric-space triangle theorem for the full articulatory model.  The
    common b/t/k sanity check below is a tight triangle case, not a
    counterexample. *)

(** Example phonemes for counter-example *)
Definition phoneme_b := mkFeatures Bilabial Plosive Voiced.
Definition phoneme_k := mkFeatures Velar Plosive Voiceless.
Definition phoneme_t := mkFeatures Alveolar Plosive Voiceless.

(** Compute specific distances *)
Example dist_b_k : articulatory_distance phoneme_b phoneme_k == 28#100.
Proof. compute. reflexivity. Qed.

Example dist_b_t : articulatory_distance phoneme_b phoneme_t == 16#100.
Proof. compute. reflexivity. Qed.

Example dist_t_k : articulatory_distance phoneme_t phoneme_k == 12#100.
Proof. compute. reflexivity. Qed.

(** The b/t/k triple is tight: 28/100 = 16/100 + 12/100. *)
Example triangle_b_t_k_tight :
  articulatory_distance phoneme_b phoneme_k ==
    articulatory_distance phoneme_b phoneme_t +
    articulatory_distance phoneme_t phoneme_k.
Proof.
  compute.
  reflexivity.
Qed.

(** * Cost Blending *)

(** Blend articulatory distance with standard edit cost *)
Definition blended_cost (art_dist : Q) (standard_cost : Q) (blend : Q) : Q :=
  blend * art_dist + (1 - blend) * standard_cost.

(** Blending is bounded when inputs are bounded *)
Lemma blended_cost_bounded : forall art_dist std_cost blend,
  0 <= art_dist <= 1 ->
  0 <= std_cost <= 1 ->
  0 <= blend <= 1 ->
  0 <= blended_cost art_dist std_cost blend <= 1.
Proof.
  intros art_dist std_cost blend [Ha_lo Ha_hi] [Hs_lo Hs_hi] [Hb_lo Hb_hi].
  unfold blended_cost.
  assert (Hone_minus_blend_lo : 0 <= 1 - blend).
  { apply (proj1 (Qle_minus_iff blend 1)). exact Hb_hi. }
  split.
  - (* 0 <= blended_cost *)
    replace 0 with (0 + 0)%Q by reflexivity.
    apply Qplus_le_compat.
    + apply Qmult_le_0_compat; assumption.
    + apply Qmult_le_0_compat; assumption.
  - (* blended_cost <= 1 *)
    eapply Qle_trans.
    + apply Qplus_le_compat.
      * apply Qmult_le_compat_nonneg.
        -- split; [exact Hb_lo | apply Qle_refl].
        -- split; [exact Ha_lo | exact Ha_hi].
      * apply Qmult_le_compat_nonneg.
        -- split; [exact Hone_minus_blend_lo | apply Qle_refl].
        -- split; [exact Hs_lo | exact Hs_hi].
    + setoid_replace (blend * 1 + (1 - blend) * 1) with 1.
      * apply Qle_refl.
      * ring.
Qed.

(** Blending is symmetric in first argument (from articulatory) *)
Lemma blended_cost_art_symmetric : forall f1 f2 std_cost blend,
  blended_cost (articulatory_distance f1 f2) std_cost blend =
  blended_cost (articulatory_distance f2 f1) std_cost blend.
Proof.
  intros f1 f2 std_cost blend.
  unfold blended_cost.
  rewrite articulatory_symmetric.
  reflexivity.
Qed.

(** * Character to Features Mapping *)

(** Map ASCII characters to phoneme features *)
(** This is a simplified mapping for common English consonants *)
Definition char_to_features (c : ascii) : option PhonemeFeatures :=
  (* Using ASCII codes - simplified *)
  match nat_of_ascii c with
  | 98%nat  => Some (mkFeatures Bilabial Plosive Voiced)      (* b *)
  | 100%nat => Some (mkFeatures Alveolar Plosive Voiced)      (* d *)
  | 102%nat => Some (mkFeatures Labiodental Fricative Voiceless) (* f *)
  | 103%nat => Some (mkFeatures Velar Plosive Voiced)         (* g *)
  | 107%nat => Some (mkFeatures Velar Plosive Voiceless)      (* k *)
  | 108%nat => Some (mkFeatures Alveolar Lateral Voiced)      (* l *)
  | 109%nat => Some (mkFeatures Bilabial Nasal Voiced)        (* m *)
  | 110%nat => Some (mkFeatures Alveolar Nasal Voiced)        (* n *)
  | 112%nat => Some (mkFeatures Bilabial Plosive Voiceless)   (* p *)
  | 114%nat => Some (mkFeatures Alveolar Approximant Voiced)  (* r *)
  | 115%nat => Some (mkFeatures Alveolar Fricative Voiceless) (* s *)
  | 116%nat => Some (mkFeatures Alveolar Plosive Voiceless)   (* t *)
  | 118%nat => Some (mkFeatures Labiodental Fricative Voiced) (* v *)
  | 122%nat => Some (mkFeatures Alveolar Fricative Voiced)    (* z *)
  | _ => None
  end.

(** Character distance when features available *)
Definition char_articulatory_distance (c1 c2 : ascii) : option Q :=
  match char_to_features c1, char_to_features c2 with
  | Some f1, Some f2 => Some (articulatory_distance f1 f2)
  | _, _ => None
  end.

(** Character distance is symmetric *)
Lemma char_distance_symmetric : forall c1 c2 d,
  char_articulatory_distance c1 c2 = Some d ->
  char_articulatory_distance c2 c1 = Some d.
Proof.
  intros c1 c2 d H.
  unfold char_articulatory_distance in *.
  destruct (char_to_features c1) as [f1|]; [| discriminate].
  destruct (char_to_features c2) as [f2|]; [| discriminate].
  inversion H. subst.
  rewrite articulatory_symmetric. reflexivity.
Qed.
