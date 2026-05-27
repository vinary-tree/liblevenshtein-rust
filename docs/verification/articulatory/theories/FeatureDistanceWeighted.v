(** * Weighted Articulatory Feature Distance — faithful 7-dimension model

    A faithful Coq model of [feature_set_distance_weighted]
    (src/phonetic/feature_distance.rs), capturing ALL SEVEN weight dimensions of
    the Rust [FeatureDistanceWeights] struct, the vowel path
    (height / backness / rounding), the vowel-vs-consonant maximum, and the
    explicit [.min(1.0)] cap (modeled with [Qmin]).

    Companion to FeatureDistance.v (the 3-weight consonant model). Where that
    file generalizes the historical idealized model, this file mirrors the Rust
    function field-for-field. Levels (place, vowel height, vowel backness) are
    rationals (Rust uses integer levels and |level difference|); we use [Qabs] of
    the difference, which is the same quantity.

    Proven (admission-free) for [feature_set_distance_w7]:
      - fsd7_symmetric    : symmetry (all weights)
      - fsd7_identity     : a phoneme against itself is 0 (all weights)
      - fsd7_nonneg       : non-negativity (non-negative weights)
      - fsd7_bounded      : <= 1 for ALL weights (directly from the cap)
      - fsd7_monotone     : per-dimension monotonicity (non-strict, through the cap)

    The monotonicity is the formal backing of the Rust [weighted_*_monotonic]
    tests; the non-strict (<=) form is exactly why the proptest monotonicity
    property uses [>=] (the cap saturates increases).
*)

Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qabs.
Require Import Coq.QArith.Qminmax.
Require Import Coq.Bool.Bool.
Require Import Coq.micromega.Lqa.

Local Open Scope Q_scope.

(** ** Manner of articulation *)

Inductive Manner :=
  | MStop | MFricative | MAffricate | MNasal | MApproximant | MLateral.

Definition manner_eqb (a b : Manner) : bool :=
  match a, b with
  | MStop, MStop | MFricative, MFricative | MAffricate, MAffricate
  | MNasal, MNasal | MApproximant, MApproximant | MLateral, MLateral => true
  | _, _ => false
  end.

(** Curated manner base distances (a subset of the Rust MANNER_DISTANCES table),
    written symmetrically; pairs absent here fall back to the default weight. *)
Definition manner_base_opt (m1 m2 : Manner) : option Q :=
  match m1, m2 with
  | MStop, MFricative | MFricative, MStop => Some (3#10)
  | MStop, MAffricate | MAffricate, MStop => Some (2#10)
  | MStop, MNasal | MNasal, MStop => Some (3#10)
  | MStop, MApproximant | MApproximant, MStop => Some (4#10)
  | MFricative, MAffricate | MAffricate, MFricative => Some (2#10)
  | MFricative, MApproximant | MApproximant, MFricative => Some (3#10)
  | MNasal, MApproximant | MApproximant, MNasal => Some (3#10)
  | MApproximant, MLateral | MLateral, MApproximant => Some (1#10)
  | _, _ => None
  end.

(** ** Phonemes *)

(** A consonant carries voicing, a place level, and a manner. (Real phonemes
    always have place and manner; the Rust missing-feature fallback to
    [manner_default] is a defensive edge case not modeled here, so that identity
    holds cleanly. [manner_default] remains meaningful: it is the cost of two
    distinct manners absent from the curated table.) *)
Record Consonant := mkCons {
  c_voiced : bool;
  c_place  : Q;
  c_manner : Manner
}.

Record Vowel := mkVowel {
  v_height : Q;
  v_back   : Q;
  v_round  : bool
}.

Inductive Phoneme := Cons (_ : Consonant) | Vow (_ : Vowel).

(** ** Seven per-dimension weights (mirrors Rust FeatureDistanceWeights). *)
Record FeatureWeights7 := mkW7 {
  w7_voicing        : Q;
  w7_place          : Q;
  w7_manner_default : Q;
  w7_manner_scale   : Q;
  w7_vheight        : Q;
  w7_vback          : Q;
  w7_vround         : Q
}.

(** The built-in IPA defaults (= Rust FeatureDistanceWeights::standard). *)
Definition standard_weights7 : FeatureWeights7 :=
  mkW7 (1#10) (15#100) (5#10) 1 (15#100) (15#100) (1#10).

Definition weights7_nonneg (w : FeatureWeights7) : Prop :=
  0 <= w7_voicing w /\ 0 <= w7_place w /\ 0 <= w7_manner_default w /\
  0 <= w7_manner_scale w /\ 0 <= w7_vheight w /\ 0 <= w7_vback w /\
  0 <= w7_vround w.

Definition weights7_le (w w' : FeatureWeights7) : Prop :=
  w7_voicing w <= w7_voicing w' /\ w7_place w <= w7_place w' /\
  w7_manner_default w <= w7_manner_default w' /\ w7_manner_scale w <= w7_manner_scale w' /\
  w7_vheight w <= w7_vheight w' /\ w7_vback w <= w7_vback w' /\
  w7_vround w <= w7_vround w'.

(** ** Component distances *)

(** A binary feature contributes its weight iff the two values differ. *)
Definition bool_diff_cost (w : Q) (b1 b2 : bool) : Q :=
  if Bool.eqb b1 b2 then 0 else w.

Definition place_dist (w : FeatureWeights7) (a b : Q) : Q :=
  w7_place w * Qabs (a - b).

Definition manner_dist (w : FeatureWeights7) (m1 m2 : Manner) : Q :=
  if manner_eqb m1 m2 then 0
  else match manner_base_opt m1 m2 with
       | Some d => w7_manner_scale w * d
       | None => w7_manner_default w
       end.

Definition cons_body (w : FeatureWeights7) (c1 c2 : Consonant) : Q :=
  bool_diff_cost (w7_voicing w) (c_voiced c1) (c_voiced c2)
  + place_dist w (c_place c1) (c_place c2)
  + manner_dist w (c_manner c1) (c_manner c2).

Definition vowel_body (w : FeatureWeights7) (u1 u2 : Vowel) : Q :=
  w7_vheight w * Qabs (v_height u1 - v_height u2)
  + w7_vback w * Qabs (v_back u1 - v_back u2)
  + bool_diff_cost (w7_vround w) (v_round u1) (v_round u2).

(** Pre-cap body: matching categories combine their dimensions; a vowel and a
    consonant are maximally distant (1, the Rust cross-category branch). *)
Definition body (w : FeatureWeights7) (p1 p2 : Phoneme) : Q :=
  match p1, p2 with
  | Cons c1, Cons c2 => cons_body w c1 c2
  | Vow u1, Vow u2 => vowel_body w u1 u2
  | _, _ => 1
  end.

(** Full weighted distance: the body, capped at 1 (Rust [.min(1.0)]). *)
Definition feature_set_distance_w7 (w : FeatureWeights7) (p1 p2 : Phoneme) : Q :=
  Qmin 1 (body w p1 p2).

(** ** Small reusable facts *)

Lemma Qplus_nonneg : forall x y, 0 <= x -> 0 <= y -> 0 <= x + y.
Proof.
  intros x y Hx Hy.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat; assumption.
Qed.

Lemma qabs_sub_comm : forall a b, Qabs (a - b) == Qabs (b - a).
Proof.
  intros a b. setoid_replace (b - a) with (- (a - b)) by ring.
  symmetry. apply Qabs_opp.
Qed.

Lemma qabs_self_zero : forall a, Qabs (a - a) == 0.
Proof.
  intro a. setoid_replace (a - a) with 0 by ring.
  apply Qabs_pos. apply Qle_refl.
Qed.

(** Monotone congruence for the cap: [Qmin 1] is monotone in its argument. *)
Lemma qmin_mono_r : forall a x y, x <= y -> Qmin a x <= Qmin a y.
Proof.
  intros a x y H. apply Q.min_glb.
  - apply Q.le_min_l.
  - apply (Qle_trans _ x). apply Q.le_min_r. exact H.
Qed.

(** ** Symmetry of the components *)

Lemma manner_eqb_sym : forall a b, manner_eqb a b = manner_eqb b a.
Proof. intros a b. destruct a; destruct b; reflexivity. Qed.

Lemma manner_eqb_refl : forall m, manner_eqb m m = true.
Proof. intro m. destruct m; reflexivity. Qed.

Lemma manner_base_opt_sym : forall a b, manner_base_opt a b = manner_base_opt b a.
Proof. intros a b. destruct a; destruct b; reflexivity. Qed.

Lemma manner_base_opt_nonneg : forall a b d, manner_base_opt a b = Some d -> 0 <= d.
Proof.
  intros a b d H. destruct a; destruct b; cbn in H;
    (discriminate H || (injection H as <-; lra)).
Qed.

Lemma bool_diff_cost_sym : forall w b1 b2, bool_diff_cost w b1 b2 = bool_diff_cost w b2 b1.
Proof. intros w b1 b2. unfold bool_diff_cost. destruct b1; destruct b2; reflexivity. Qed.

Lemma bool_diff_cost_refl : forall w b, bool_diff_cost w b b = 0.
Proof. intros w b. unfold bool_diff_cost. destruct b; reflexivity. Qed.

Lemma manner_dist_sym : forall w m1 m2, manner_dist w m1 m2 = manner_dist w m2 m1.
Proof.
  intros w m1 m2. unfold manner_dist.
  rewrite (manner_eqb_sym m1 m2), (manner_base_opt_sym m1 m2).
  reflexivity.
Qed.

Lemma manner_dist_refl : forall w m, manner_dist w m m = 0.
Proof. intros w m. unfold manner_dist. rewrite manner_eqb_refl. reflexivity. Qed.

Lemma place_dist_sym : forall w a b, place_dist w a b == place_dist w b a.
Proof. intros w a b. unfold place_dist. rewrite (qabs_sub_comm a b). reflexivity. Qed.

Lemma place_dist_refl : forall w a, place_dist w a a == 0.
Proof. intros w a. unfold place_dist. rewrite (qabs_self_zero a). ring. Qed.

(** ** Non-negativity of the components *)

Lemma bool_diff_cost_nonneg : forall w b1 b2, 0 <= w -> 0 <= bool_diff_cost w b1 b2.
Proof. intros w b1 b2 H. unfold bool_diff_cost. destruct (Bool.eqb b1 b2). apply Qle_refl. exact H. Qed.

Lemma place_dist_nonneg : forall w a b, 0 <= w7_place w -> 0 <= place_dist w a b.
Proof. intros w a b H. unfold place_dist. apply Qmult_le_0_compat. exact H. apply Qabs_nonneg. Qed.

Lemma manner_dist_nonneg : forall w m1 m2,
  0 <= w7_manner_scale w -> 0 <= w7_manner_default w -> 0 <= manner_dist w m1 m2.
Proof.
  intros w m1 m2 Hs Hd. unfold manner_dist.
  destruct (manner_eqb m1 m2). apply Qle_refl.
  destruct (manner_base_opt m1 m2) eqn:E.
  - apply Qmult_le_0_compat. exact Hs. eapply manner_base_opt_nonneg. exact E.
  - exact Hd.
Qed.

(** ** Monotonicity of the components *)

Lemma bool_diff_cost_mono : forall x y b1 b2, x <= y -> bool_diff_cost x b1 b2 <= bool_diff_cost y b1 b2.
Proof. intros x y b1 b2 H. unfold bool_diff_cost. destruct (Bool.eqb b1 b2). apply Qle_refl. exact H. Qed.

Lemma manner_dist_mono : forall w w' m1 m2,
  w7_manner_scale w <= w7_manner_scale w' ->
  w7_manner_default w <= w7_manner_default w' ->
  manner_dist w m1 m2 <= manner_dist w' m1 m2.
Proof.
  intros w w' m1 m2 Hsc Hdef. unfold manner_dist.
  destruct (manner_eqb m1 m2). apply Qle_refl.
  destruct (manner_base_opt m1 m2) eqn:E.
  - apply Qmult_le_compat_r. exact Hsc. eapply manner_base_opt_nonneg. exact E.
  - exact Hdef.
Qed.

(** ** Main theorems *)

(** Symmetry of the body (all weights). *)
Lemma body_sym : forall w p1 p2, body w p1 p2 == body w p2 p1.
Proof.
  intros w p1 p2. destruct p1 as [c1|u1]; destruct p2 as [c2|u2]; cbn [body].
  - unfold cons_body.
    rewrite (bool_diff_cost_sym (w7_voicing w) (c_voiced c1) (c_voiced c2)).
    rewrite (place_dist_sym w (c_place c1) (c_place c2)).
    rewrite (manner_dist_sym w (c_manner c1) (c_manner c2)).
    reflexivity.
  - reflexivity.
  - reflexivity.
  - unfold vowel_body.
    rewrite (bool_diff_cost_sym (w7_vround w) (v_round u1) (v_round u2)).
    rewrite (qabs_sub_comm (v_height u1) (v_height u2)).
    rewrite (qabs_sub_comm (v_back u1) (v_back u2)).
    reflexivity.
Qed.

Theorem fsd7_symmetric : forall w p1 p2,
  feature_set_distance_w7 w p1 p2 == feature_set_distance_w7 w p2 p1.
Proof.
  intros w p1 p2. unfold feature_set_distance_w7.
  rewrite (body_sym w p1 p2). reflexivity.
Qed.

(** Identity: a phoneme against itself is 0 (all weights). *)
Theorem fsd7_identity : forall w p, feature_set_distance_w7 w p p == 0.
Proof.
  intros w p. unfold feature_set_distance_w7.
  assert (Hb : body w p p == 0).
  { destruct p as [c|u]; cbn [body].
    - unfold cons_body.
      rewrite (bool_diff_cost_refl (w7_voicing w) (c_voiced c)).
      rewrite (manner_dist_refl w (c_manner c)).
      rewrite (place_dist_refl w (c_place c)).
      ring.
    - unfold vowel_body.
      rewrite (bool_diff_cost_refl (w7_vround w) (v_round u)).
      rewrite (qabs_self_zero (v_height u)).
      rewrite (qabs_self_zero (v_back u)).
      ring. }
  rewrite Hb.
  apply Qle_antisym.
  - apply Q.le_min_r.
  - apply Q.min_glb. lra. apply Qle_refl.
Qed.

(** Non-negativity (non-negative weights). *)
Lemma body_nonneg : forall w p1 p2, weights7_nonneg w -> 0 <= body w p1 p2.
Proof.
  intros w p1 p2 [Hvo [Hpl [Hmd [Hms [Hvh [Hvb Hvr]]]]]].
  destruct p1 as [c1|u1]; destruct p2 as [c2|u2]; cbn [body].
  - unfold cons_body.
    apply Qplus_nonneg;
      [ apply Qplus_nonneg;
          [ apply bool_diff_cost_nonneg; exact Hvo
          | apply place_dist_nonneg; exact Hpl ]
      | apply manner_dist_nonneg; [exact Hms | exact Hmd] ].
  - lra.
  - lra.
  - unfold vowel_body.
    apply Qplus_nonneg;
      [ apply Qplus_nonneg;
          [ apply Qmult_le_0_compat; [exact Hvh | apply Qabs_nonneg]
          | apply Qmult_le_0_compat; [exact Hvb | apply Qabs_nonneg] ]
      | apply bool_diff_cost_nonneg; exact Hvr ].
Qed.

Theorem fsd7_nonneg : forall w p1 p2, weights7_nonneg w -> 0 <= feature_set_distance_w7 w p1 p2.
Proof.
  intros w p1 p2 Hw. unfold feature_set_distance_w7.
  apply Q.min_glb. lra. apply body_nonneg. exact Hw.
Qed.

(** Boundedness <= 1 holds for ALL weights — it is enforced directly by the cap. *)
Theorem fsd7_bounded : forall w p1 p2, feature_set_distance_w7 w p1 p2 <= 1.
Proof. intros w p1 p2. unfold feature_set_distance_w7. apply Q.le_min_l. Qed.

(** Per-dimension monotonicity: raising weights componentwise never decreases the
    distance. Non-strict (<=) because of the cap. *)
Lemma body_mono : forall w w' p1 p2, weights7_le w w' -> body w p1 p2 <= body w' p1 p2.
Proof.
  intros w w' p1 p2 [Hvo [Hpl [Hmd [Hms [Hvh [Hvb Hvr]]]]]].
  destruct p1 as [c1|u1]; destruct p2 as [c2|u2]; cbn [body].
  - unfold cons_body.
    apply Qplus_le_compat;
      [ apply Qplus_le_compat;
          [ apply bool_diff_cost_mono; exact Hvo
          | apply Qmult_le_compat_r; [exact Hpl | apply Qabs_nonneg] ]
      | apply manner_dist_mono; [exact Hms | exact Hmd] ].
  - apply Qle_refl.
  - apply Qle_refl.
  - unfold vowel_body.
    apply Qplus_le_compat;
      [ apply Qplus_le_compat;
          [ apply Qmult_le_compat_r; [exact Hvh | apply Qabs_nonneg]
          | apply Qmult_le_compat_r; [exact Hvb | apply Qabs_nonneg] ]
      | apply bool_diff_cost_mono; exact Hvr ].
Qed.

Theorem fsd7_monotone : forall w w' p1 p2,
  weights7_le w w' -> feature_set_distance_w7 w p1 p2 <= feature_set_distance_w7 w' p1 p2.
Proof.
  intros w w' p1 p2 Hle. unfold feature_set_distance_w7.
  apply qmin_mono_r. apply body_mono. exact Hle.
Qed.

(** ** Non-vacuity: concrete distances under the standard weights *)

(** A vowel and a consonant are maximally distant. *)
Example cross_category_max :
  feature_set_distance_w7 standard_weights7
    (Cons (mkCons true (0#1) MStop)) (Vow (mkVowel (0#1) (0#1) false)) == 1.
Proof. vm_compute. reflexivity. Qed.

(** Two vowels differing by one height step: 1 * vowel_height_step = 0.15. *)
Example vowel_one_height_step :
  feature_set_distance_w7 standard_weights7
    (Vow (mkVowel (1#1) (0#1) false)) (Vow (mkVowel (2#1) (0#1) false)) == 15#100.
Proof. vm_compute. reflexivity. Qed.

(** Standard weights are non-negative (so the conditional theorems apply). *)
Example standard_weights7_nonneg : weights7_nonneg standard_weights7.
Proof. unfold weights7_nonneg, standard_weights7; cbn. repeat split; lra. Qed.
