(** * Interval-Relaxed MSM Per-Element Cost Lower Bounds

    The interval-MSM transducer ([crate::time_series::msm_transducer]) walks a
    trie of *quantized* reference series. Along a trie branch each target element
    is known only up to its quantization bin: an interval [lo, hi] rather than a
    scalar. To prune soundly it evaluates the MSM recurrence with each per-element
    cost replaced by an admissible lower bound -- the minimum the true cost could
    take for any concrete value(s) inside the bin interval(s).

    This module defines those three closed-form bounds and proves, with no
    axioms, that each is:

    - admissible  (<= the true cost for every concrete value in the interval), and
    - exact       (achieved by some concrete value, hence the *tightest* lower
                   bound -- mirroring the Rust property tests
                   `merge_lb_is_min_over_c` / `split_lb_is_min_over_box`).

    Quantization extreme bins extend to +/- infinity (everything below the range
    folds into bin 0, everything above into the last bin). We model a bin
    interval as [Qitv] = [(option Q * option Q)] with [None] = unbounded, and
    write the bound functions so that an unbounded endpoint is routed
    structurally to the penalty-0 / overlap branch -- every subtraction stays
    inside [Q], so no [oo - oo] is ever evaluated. This faithfully mirrors the
    Rust invariant documented in `msm_interval.rs`.

    The MSM C function (see [CFunction.v]) is
      [c_func c a b cv] = c                              if [is_between a b cv]
                        = c + min(|a-b|, |a-cv|)         otherwise,
    where [is_between a b cv] holds iff [a] lies between [b] and [cv] inclusive.

    Part of: Liblevenshtein.MSM
*)

From Stdlib Require Import List Nat Arith Lia Psatz.
From Stdlib Require Import QArith Qabs Qminmax.
Import ListNotations.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction CFunctionBounds.

(** * Maximum of two rationals (dual of [Qmin2]) *)

Definition Qmax2 (a b : Q) : Q := if Qle_bool a b then b else a.

Lemma Qmax2_le_l : forall a b, a <= Qmax2 a b.
Proof.
  intros a b. unfold Qmax2. destruct (Qle_bool a b) eqn:H.
  - apply Qle_bool_iff in H. exact H.
  - apply Qle_refl.
Qed.

Lemma Qmax2_le_r : forall a b, b <= Qmax2 a b.
Proof.
  intros a b. unfold Qmax2. destruct (Qle_bool a b) eqn:H.
  - apply Qle_refl.
  - apply Qle_bool_false_lt in H. apply Qlt_le_weak. exact H.
Qed.

Lemma Qmax2_lub : forall a b c, a <= c -> b <= c -> Qmax2 a b <= c.
Proof.
  intros a b c Ha Hb. unfold Qmax2. destruct (Qle_bool a b); assumption.
Qed.

Lemma Qmax2_nonneg : forall a b, 0 <= a -> 0 <= b -> 0 <= Qmax2 a b.
Proof.
  intros a b Ha _. apply Qle_trans with a. exact Ha. apply Qmax2_le_l.
Qed.

(** Monotonicity of [Qmin2] in its second argument. *)
Lemma Qmin2_mono_r : forall x p q, p <= q -> Qmin2 x p <= Qmin2 x q.
Proof.
  intros x p q Hpq. apply Qmin2_glb.
  - apply Qmin2_le_l.
  - apply Qle_trans with p. apply Qmin2_le_r. exact Hpq.
Qed.

(** [Qmin2] respects [Qeq] (it is not registered as a setoid morphism). *)
Lemma Qmin2_wd : forall a a' b b', a == a' -> b == b' -> Qmin2 a b == Qmin2 a' b'.
Proof.
  intros a a' b b' Ha Hb. apply Qle_antisym; apply Qmin2_glb.
  - apply Qle_trans with a; [ apply Qmin2_le_l | rewrite Ha; apply Qle_refl ].
  - apply Qle_trans with b; [ apply Qmin2_le_r | rewrite Hb; apply Qle_refl ].
  - apply Qle_trans with a'; [ apply Qmin2_le_l | rewrite Ha; apply Qle_refl ].
  - apply Qle_trans with b'; [ apply Qmin2_le_r | rewrite Hb; apply Qle_refl ].
Qed.

(** [x <= |x|] over Q (used to bound a signed gap by an absolute difference). *)
Lemma Qle_Qabs_self : forall x, x <= Qabs x.
Proof. exact Qle_Qabs. Qed.

(** * Interval type with optional (unbounded) endpoints *)

Definition Qitv : Type := (option Q * option Q)%type.

(** [v] lies in the (closed, possibly unbounded) interval [iv]. *)
Definition in_itv (v : Q) (iv : Qitv) : Prop :=
  (match fst iv with Some lo => lo <= v | None => True end) /\
  (match snd iv with Some hi => v <= hi | None => True end).

(** An interval is non-empty when it contains some value. *)
Definition itv_nonempty (iv : Qitv) : Prop := exists v, in_itv v iv.

(** * Move bound: distance from a scalar to an interval *)

Definition below_gap (lo : option Q) (v : Q) : Q :=
  match lo with Some l => Qmax2 0 (l - v) | None => 0 end.

Definition above_gap (hi : option Q) (v : Q) : Q :=
  match hi with Some h => Qmax2 0 (v - h) | None => 0 end.

(** Admissible lower bound on the MSM Move cost |v - y| for [y] in [iv]:
    the distance from [v] to the interval (0 iff [v] is inside). *)
Definition interval_dist (v : Q) (iv : Qitv) : Q :=
  Qmax2 (below_gap (fst iv) v) (above_gap (snd iv) v).

Lemma below_gap_nonneg : forall lo v, 0 <= below_gap lo v.
Proof. intros [l|] v; simpl; [ apply Qmax2_le_l | apply Qle_refl ]. Qed.

Lemma above_gap_nonneg : forall hi v, 0 <= above_gap hi v.
Proof. intros [h|] v; simpl; [ apply Qmax2_le_l | apply Qle_refl ]. Qed.

Lemma interval_dist_nonneg : forall v iv, 0 <= interval_dist v iv.
Proof.
  intros v iv. unfold interval_dist.
  apply Qle_trans with (below_gap (fst iv) v).
  - apply below_gap_nonneg.
  - apply Qmax2_le_l.
Qed.

(** Admissibility of the Move bound. *)
Lemma interval_dist_le_move : forall v y iv,
  in_itv y iv -> interval_dist v iv <= Qabs_diff v y.
Proof.
  intros v y [lo hi] [Hlo Hhi]. cbn [fst snd] in *.
  unfold interval_dist. cbn [fst snd].
  apply Qmax2_lub.
  - destruct lo as [l|]; cbn [below_gap].
    + apply Qmax2_lub.
      * apply Qabs_diff_nonneg.
      * apply Qle_trans with (y - v).
        -- psatz Q.
        -- rewrite Qabs_diff_symm. unfold Qabs_diff. apply Qle_Qabs.
    + apply Qabs_diff_nonneg.
  - destruct hi as [h|]; cbn [above_gap].
    + apply Qmax2_lub.
      * apply Qabs_diff_nonneg.
      * apply Qle_trans with (v - y).
        -- psatz Q.
        -- unfold Qabs_diff. apply Qle_Qabs.
    + apply Qabs_diff_nonneg.
Qed.

(** * Merge bound: free [c_val] over an interval, scalar [a], [b] *)

(** "[a <= hi]" with [None] (= +inf) always satisfied. *)
Definition le_hi (a : Q) (hi : option Q) : bool :=
  match hi with Some h => Qle_bool a h | None => true end.

(** "[lo <= a]" with [None] (= -inf) always satisfied. *)
Definition ge_lo (a : Q) (lo : option Q) : bool :=
  match lo with Some l => Qle_bool l a | None => true end.

(** There exists [cv] in [iv] placing [a] between [b] and [cv]. *)
Definition merge_between (a b : Q) (iv : Qitv) : bool :=
  (Qle_bool b a && le_hi a (snd iv)) || (Qle_bool a b && ge_lo a (fst iv)).

(** Admissible lower bound on the Merge cost [c_func c a b cv] for [cv] in [iv]. *)
Definition c_func_merge_lb (c_const a b : Q) (iv : Qitv) : Q :=
  if merge_between a b iv
  then c_const
  else c_const + Qmin2 (Qabs_diff a b) (interval_dist a iv).

(** If some concrete [cv] in [iv] makes [a] between [b] and [cv], the guard fires. *)
Lemma is_between_in_itv_merge : forall a b iv cv,
  in_itv cv iv -> is_between a b cv = true -> merge_between a b iv = true.
Proof.
  intros a b [lo hi] cv [Hlo Hhi] Hbet. simpl in *.
  unfold merge_between, is_between in *. simpl in *.
  apply orb_true_iff in Hbet. apply orb_true_iff.
  destruct Hbet as [H1 | H2].
  - apply andb_true_iff in H1. destruct H1 as [Hba Hacv]. left.
    apply andb_true_iff. split; [ exact Hba |].
    destruct hi as [h|]; simpl; [| reflexivity].
    apply Qle_bool_iff in Hacv. apply Qle_bool_iff.
    apply Qle_trans with cv; [ exact Hacv | exact Hhi ].
  - apply andb_true_iff in H2. destruct H2 as [Hcva Hab]. right.
    apply andb_true_iff. split; [ exact Hab |].
    destruct lo as [l|]; simpl; [| reflexivity].
    apply Qle_bool_iff in Hcva. apply Qle_bool_iff.
    apply Qle_trans with cv; [ exact Hlo | exact Hcva ].
Qed.

(** Admissibility of the Merge bound. *)
Lemma c_func_merge_lb_le : forall c_const a b iv cv,
  0 <= c_const -> in_itv cv iv ->
  c_func_merge_lb c_const a b iv <= c_func c_const a b cv.
Proof.
  intros c_const a b iv cv Hc Hin.
  unfold c_func_merge_lb.
  destruct (merge_between a b iv) eqn:Hguard.
  - apply c_func_ge_c. exact Hc.
  - assert (Hbet : is_between a b cv = false).
    { destruct (is_between a b cv) eqn:E; [| reflexivity].
      rewrite (is_between_in_itv_merge a b iv cv Hin E) in Hguard. discriminate. }
    unfold c_func. rewrite Hbet.
    apply Qplus_le_compat; [ apply Qle_refl |].
    change (Qabs (a - b)) with (Qabs_diff a b).
    change (Qabs (a - cv)) with (Qabs_diff a cv).
    apply Qmin2_mono_r.
    apply interval_dist_le_move. exact Hin.
Qed.

(** * Split bound: free [a] over [a_iv], scalar [b], free [c_val] over [c_iv]

    In the MSM split term [c_func c (y_j) (x_i) (y_{j-1})] the free arguments are
    [a = y_j] (current target bin [a_iv]) and [c_val = y_{j-1}] (previous target
    bin [c_iv]); [b = x_i] is the scalar query element. The set of [a] that lie
    "between [b] and some [cv] in [c_iv]" is the contiguous union
    [min(b, c_lo), max(b, c_hi)]. The bound is [c] plus the gap from [a_iv] to
    that union (0 when they overlap). We express the penalty uniformly as the
    maximum of the "forced-above" and "forced-below" gaps -- both 0 on overlap. *)

Definition opt_min_b (b : Q) (lo : option Q) : option Q :=
  match lo with Some l => Some (Qmin2 b l) | None => None end.

Definition opt_max_b (b : Q) (hi : option Q) : option Q :=
  match hi with Some h => Some (Qmax2 b h) | None => None end.

(** Gap [max(0, lo_ep - upper_ep)]; 0 whenever either endpoint is unbounded. *)
Definition opt_below_gap (lo_ep upper_ep : option Q) : Q :=
  match lo_ep, upper_ep with
  | Some l, Some u => Qmax2 0 (l - u)
  | _, _ => 0
  end.

Definition c_func_split_lb (c_const : Q) (a_iv : Qitv) (b : Q) (c_iv : Qitv) : Q :=
  let ulo := opt_min_b b (fst c_iv) in
  let uhi := opt_max_b b (snd c_iv) in
  c_const
  + Qmax2 (opt_below_gap (fst a_iv) uhi) (opt_below_gap ulo (snd a_iv)).

Lemma opt_below_gap_nonneg : forall lo_ep upper_ep, 0 <= opt_below_gap lo_ep upper_ep.
Proof.
  intros [l|] [u|]; simpl; try apply Qle_refl. apply Qmax2_le_l.
Qed.

(** Admissibility of the Split bound. *)
Lemma c_func_split_lb_le : forall c_const a_iv b c_iv av cv,
  0 <= c_const -> in_itv av a_iv -> in_itv cv c_iv ->
  c_func_split_lb c_const a_iv b c_iv <= c_func c_const av b cv.
Proof.
  intros c_const [alo ahi] b [clo chi] av cv Hc [Hal Hah] [Hcl Hch].
  cbn [fst snd] in *.
  unfold c_func_split_lb. cbn [fst snd].
  (* Bound each gap below |av-b| and |av-cv| (admissibility), and show both gaps
     vanish when av is between b and cv. The gap terms are written explicitly so
     the asserts match the goal syntactically. *)
  assert (Hga_b : opt_below_gap alo (opt_max_b b chi) <= Qabs (av - b)).
  { destruct alo as [al|]; destruct chi as [ch|]; cbn [opt_below_gap opt_max_b];
      try apply Qabs_nonneg.
    apply Qmax2_lub; [ apply Qabs_nonneg |].
    apply Qle_trans with (av - b); [| apply Qle_Qabs ].
    assert (Hb : b <= Qmax2 b ch) by apply Qmax2_le_l. psatz Q. }
  assert (Hga_c : opt_below_gap alo (opt_max_b b chi) <= Qabs (av - cv)).
  { destruct alo as [al|]; destruct chi as [ch|]; cbn [opt_below_gap opt_max_b];
      try apply Qabs_nonneg.
    apply Qmax2_lub; [ apply Qabs_nonneg |].
    apply Qle_trans with (av - cv); [| apply Qle_Qabs ].
    assert (Hc2 : cv <= Qmax2 b ch) by (apply Qle_trans with ch; [ exact Hch | apply Qmax2_le_r ]).
    psatz Q. }
  assert (Hgb_b : opt_below_gap (opt_min_b b clo) ahi <= Qabs (av - b)).
  { destruct clo as [cl|]; destruct ahi as [ah|]; cbn [opt_below_gap opt_min_b];
      try apply Qabs_nonneg.
    apply Qmax2_lub; [ apply Qabs_nonneg |].
    rewrite <- (Qabs_opp (av - b)).
    setoid_replace (- (av - b)) with (b - av) by ring.
    apply Qle_trans with (b - av); [| apply Qle_Qabs ].
    assert (Hb : Qmin2 b cl <= b) by apply Qmin2_le_l. psatz Q. }
  assert (Hgb_c : opt_below_gap (opt_min_b b clo) ahi <= Qabs (av - cv)).
  { destruct clo as [cl|]; destruct ahi as [ah|]; cbn [opt_below_gap opt_min_b];
      try apply Qabs_nonneg.
    apply Qmax2_lub; [ apply Qabs_nonneg |].
    rewrite <- (Qabs_opp (av - cv)).
    setoid_replace (- (av - cv)) with (cv - av) by ring.
    apply Qle_trans with (cv - av); [| apply Qle_Qabs ].
    assert (Hc2 : Qmin2 b cl <= cv) by (apply Qle_trans with cl; [ apply Qmin2_le_r | exact Hcl ]).
    psatz Q. }
  unfold c_func.
  destruct (is_between av b cv) eqn:Ebet.
  - (* between: both gaps are 0, so the penalty is <= 0 <= c_func - c. *)
    assert (Hga0 : opt_below_gap alo (opt_max_b b chi) <= 0).
    { destruct alo as [al|]; destruct chi as [ch|]; cbn [opt_below_gap opt_max_b];
        try apply Qle_refl.
      apply Qmax2_lub; [ apply Qle_refl |].
      apply is_between_true_cases in Ebet.
      assert (Hav_uhi : av <= Qmax2 b ch).
      { destruct Ebet as [[Hba Havc] | [Hcva Havb]].
        - apply Qle_trans with ch; [ apply Qle_trans with cv; [exact Havc | exact Hch] | apply Qmax2_le_r ].
        - apply Qle_trans with b; [ exact Havb | apply Qmax2_le_l ]. }
      psatz Q. }
    assert (Hgb0 : opt_below_gap (opt_min_b b clo) ahi <= 0).
    { destruct clo as [cl|]; destruct ahi as [ah|]; cbn [opt_below_gap opt_min_b];
        try apply Qle_refl.
      apply Qmax2_lub; [ apply Qle_refl |].
      apply is_between_true_cases in Ebet.
      assert (Hulo_av : Qmin2 b cl <= av).
      { destruct Ebet as [[Hba Havc] | [Hcva Havb]].
        - apply Qle_trans with b; [ apply Qmin2_le_l | exact Hba ].
        - apply Qle_trans with cv; [ apply Qle_trans with cl; [apply Qmin2_le_r | exact Hcl] | exact Hcva ]. }
      psatz Q. }
    apply Qle_trans with (c_const + 0).
    + apply Qplus_le_compat; [ apply Qle_refl | apply Qmax2_lub; assumption ].
    + psatz Q.
  - (* not between: penalty <= min(|av-b|, |av-cv|). *)
    apply Qplus_le_compat; [ apply Qle_refl |].
    apply Qmax2_lub; apply Qmin2_glb; assumption.
Qed.

(** * Exactness (tightness): each bound is achieved by some concrete value

    Together with the admissibility lemmas above, these establish that each
    closed-form bound is exactly the minimum of the true cost over the interval
    box -- the "exact minimum" claim of `msm_interval.rs` and the Rust property
    tests `merge_lb_is_min_over_c` / `split_lb_is_min_over_box`. *)

Lemma interval_dist_ge_below_some : forall l hi v, l - v <= interval_dist v (Some l, hi).
Proof.
  intros l hi v. unfold interval_dist. cbn [fst snd].
  apply Qle_trans with (below_gap (Some l) v); [| apply Qmax2_le_l ].
  cbn [below_gap]. apply Qmax2_le_r.
Qed.

Lemma interval_dist_ge_above_some : forall lo h v, v - h <= interval_dist v (lo, Some h).
Proof.
  intros lo h v. unfold interval_dist. cbn [fst snd].
  apply Qle_trans with (above_gap (Some h) v); [| apply Qmax2_le_r ].
  cbn [above_gap]. apply Qmax2_le_r.
Qed.

(** Exactness of the Move bound: the distance is attained by a concrete [y]. *)
Lemma interval_dist_tight : forall v iv,
  itv_nonempty iv -> exists y, in_itv y iv /\ interval_dist v iv == Qabs_diff v y.
Proof.
  intros v [lo hi] [w [Hwl Hwh]]. cbn [fst snd] in Hwl, Hwh.
  destruct lo as [l|]; destruct hi as [h|].
  - assert (Hlh : l <= h) by (apply Qle_trans with w; assumption).
    destruct (Qlt_le_dec v l) as [Hvl | Hlv].
    + exists l. split; [ split; [ apply Qle_refl | exact Hlh ] |].
      apply Qle_antisym.
      * apply interval_dist_le_move. split; [ apply Qle_refl | exact Hlh ].
      * unfold Qabs_diff. rewrite Qabs_neg by psatz Q.
        apply Qle_trans with (l - v); [ psatz Q | apply interval_dist_ge_below_some ].
    + destruct (Qlt_le_dec h v) as [Hhv | Hvh].
      * exists h. split; [ split; [ exact Hlh | apply Qle_refl ] |].
        apply Qle_antisym.
        -- apply interval_dist_le_move. split; [ exact Hlh | apply Qle_refl ].
        -- unfold Qabs_diff. rewrite Qabs_pos by psatz Q.
           apply Qle_trans with (v - h); [ psatz Q | apply interval_dist_ge_above_some ].
      * exists v. split; [ split; [ exact Hlv | exact Hvh ] |].
        apply Qle_antisym.
        -- apply interval_dist_le_move. split; [ exact Hlv | exact Hvh ].
        -- apply Qle_trans with 0;
             [ rewrite Qabs_diff_zero; apply Qle_refl
             | apply interval_dist_nonneg ].
  - destruct (Qlt_le_dec v l) as [Hvl | Hlv].
    + exists l. split; [ split; [ apply Qle_refl | exact I ] |].
      apply Qle_antisym.
      * apply interval_dist_le_move. split; [ apply Qle_refl | exact I ].
      * unfold Qabs_diff. rewrite Qabs_neg by psatz Q.
        apply Qle_trans with (l - v); [ psatz Q | apply interval_dist_ge_below_some ].
    + exists v. split; [ split; [ exact Hlv | exact I ] |].
      apply Qle_antisym.
      * apply interval_dist_le_move. split; [ exact Hlv | exact I ].
      * apply Qle_trans with 0;
          [ rewrite Qabs_diff_zero; apply Qle_refl
          | apply interval_dist_nonneg ].
  - destruct (Qlt_le_dec h v) as [Hhv | Hvh].
    + exists h. split; [ split; [ exact I | apply Qle_refl ] |].
      apply Qle_antisym.
      * apply interval_dist_le_move. split; [ exact I | apply Qle_refl ].
      * unfold Qabs_diff. rewrite Qabs_pos by psatz Q.
        apply Qle_trans with (v - h); [ psatz Q | apply interval_dist_ge_above_some ].
    + exists v. split; [ split; [ exact I | exact Hvh ] |].
      apply Qle_antisym.
      * apply interval_dist_le_move. split; [ exact I | exact Hvh ].
      * apply Qle_trans with 0;
          [ rewrite Qabs_diff_zero; apply Qle_refl
          | apply interval_dist_nonneg ].
  - exists v. split; [ split; exact I |].
    apply Qle_antisym.
    + apply interval_dist_le_move. split; exact I.
    + apply Qle_trans with 0;
        [ rewrite Qabs_diff_zero; apply Qle_refl
        | apply interval_dist_nonneg ].
Qed.

(** Push a scalar into an interval's lower (resp. upper) endpoint. *)
Definition clamp_lo (lo : option Q) (a : Q) : Q :=
  match lo with Some l => Qmax2 l a | None => a end.
Definition clamp_hi (hi : option Q) (a : Q) : Q :=
  match hi with Some h => Qmin2 h a | None => a end.

(** When the Merge guard fires, a concrete in-interval witness makes [a]
    between [b] and it (so the C cost is exactly [c_const]). *)
Lemma merge_between_witness : forall a b iv,
  itv_nonempty iv -> merge_between a b iv = true ->
  exists cv, in_itv cv iv /\ is_between a b cv = true.
Proof.
  intros a b [lo hi] [w [Hwl Hwh]] Hg. cbn [fst snd] in Hwl, Hwh.
  unfold merge_between in Hg. apply orb_true_iff in Hg. destruct Hg as [HA | HB].
  - apply andb_true_iff in HA. destruct HA as [Hba Hhic]. apply Qle_bool_iff in Hba.
    exists (clamp_lo lo a). split.
    + split; cbn [fst snd].
      * destruct lo as [l|]; cbn [clamp_lo]; [ apply Qmax2_le_l | exact I ].
      * destruct hi as [h|]; cbn [le_hi] in Hhic; [| exact I].
        apply Qle_bool_iff in Hhic.
        destruct lo as [l|]; cbn [clamp_lo].
        -- apply Qmax2_lub; [ apply Qle_trans with w; assumption | exact Hhic ].
        -- exact Hhic.
    + unfold is_between. apply orb_true_iff. left. apply andb_true_iff. split.
      * apply Qle_bool_iff. exact Hba.
      * apply Qle_bool_iff. destruct lo as [l|]; cbn [clamp_lo];
          [ apply Qmax2_le_r | apply Qle_refl ].
  - apply andb_true_iff in HB. destruct HB as [Hab Hloc]. apply Qle_bool_iff in Hab.
    exists (clamp_hi hi a). split.
    + split; cbn [fst snd].
      * destruct lo as [l|]; cbn [ge_lo] in Hloc; [| exact I].
        apply Qle_bool_iff in Hloc.
        destruct hi as [h|]; cbn [clamp_hi].
        -- apply Qmin2_glb; [ apply Qle_trans with w; assumption | exact Hloc ].
        -- exact Hloc.
      * destruct hi as [h|]; cbn [clamp_hi]; [ apply Qmin2_le_l | exact I ].
    + unfold is_between. apply orb_true_iff. right. apply andb_true_iff. split.
      * apply Qle_bool_iff. destruct hi as [h|]; cbn [clamp_hi];
          [ apply Qmin2_le_r | apply Qle_refl ].
      * apply Qle_bool_iff. exact Hab.
Qed.

(** Exactness of the Merge bound. *)
Lemma c_func_merge_lb_tight : forall c_const a b iv,
  0 <= c_const -> itv_nonempty iv ->
  exists cv, in_itv cv iv /\ c_func_merge_lb c_const a b iv == c_func c_const a b cv.
Proof.
  intros c_const a b iv Hc Hne.
  destruct (merge_between a b iv) eqn:Hg.
  - destruct (merge_between_witness a b iv Hne Hg) as [cv [Hin Hbet]].
    exists cv. split; [ exact Hin |].
    unfold c_func_merge_lb. rewrite Hg. unfold c_func. rewrite Hbet. reflexivity.
  - destruct (interval_dist_tight a iv Hne) as [y [Hyin Hyeq]].
    exists y. split; [ exact Hyin |].
    assert (Hybet : is_between a b y = false).
    { destruct (is_between a b y) eqn:E; [| reflexivity].
      rewrite (is_between_in_itv_merge a b iv y Hyin E) in Hg. discriminate. }
    unfold c_func_merge_lb. rewrite Hg. unfold c_func. rewrite Hybet.
    change (Qabs (a - b)) with (Qabs_diff a b).
    change (Qabs (a - y)) with (Qabs_diff a y).
    assert (Hm : Qmin2 (Qabs_diff a b) (interval_dist a iv)
                 == Qmin2 (Qabs_diff a b) (Qabs_diff a y))
      by (apply Qmin2_wd; [ reflexivity | exact Hyeq ]).
    rewrite Hm. reflexivity.
Qed.

(** * Split bound exactness: a concrete witness attains [c_func_split_lb]. *)

(** [Qmax2] respects [Qeq] (not registered as a setoid morphism). *)
Lemma Qmax2_wd : forall a a' b b', a == a' -> b == b' -> Qmax2 a b == Qmax2 a' b'.
Proof.
  intros a a' b b' Ha Hb. apply Qle_antisym; apply Qmax2_lub.
  - apply Qle_trans with a'; [ rewrite Ha; apply Qle_refl | apply Qmax2_le_l ].
  - apply Qle_trans with b'; [ rewrite Hb; apply Qle_refl | apply Qmax2_le_r ].
  - apply Qle_trans with a; [ rewrite Ha; apply Qle_refl | apply Qmax2_le_l ].
  - apply Qle_trans with b; [ rewrite Hb; apply Qle_refl | apply Qmax2_le_r ].
Qed.

(** [a] between [b] and [cv] from the ordering [b <= a <= cv]. *)
Lemma is_between_le_le : forall a b cv, b <= a -> a <= cv -> is_between a b cv = true.
Proof.
  intros a b cv H1 H2. unfold is_between. apply orb_true_iff. left.
  apply andb_true_iff. split; apply Qle_bool_iff; assumption.
Qed.

(** [a] between [b] and [cv] from the ordering [cv <= a <= b]. *)
Lemma is_between_ge_ge : forall a b cv, cv <= a -> a <= b -> is_between a b cv = true.
Proof.
  intros a b cv H1 H2. unfold is_between. apply orb_true_iff. right.
  apply andb_true_iff. split; apply Qle_bool_iff; assumption.
Qed.

Lemma opt_below_gap_zero_le : forall l u, opt_below_gap (Some l) (Some u) == 0 -> l <= u.
Proof.
  intros l u H. cbn [opt_below_gap] in H. unfold Qmax2 in H.
  destruct (Qle_bool 0 (l - u)) eqn:E.
  - apply Qle_bool_iff in E. psatz Q.
  - apply Qle_bool_false_lt in E. psatz Q.
Qed.

Lemma opt_below_gap_pos : forall lo_ep up_ep,
  0 < opt_below_gap lo_ep up_ep -> exists l u, lo_ep = Some l /\ up_ep = Some u /\ u < l.
Proof.
  intros [l|] [u|] H; cbn [opt_below_gap] in H;
    try (exfalso; apply (Qlt_irrefl 0); exact H).
  exists l, u. repeat split. unfold Qmax2 in H.
  destruct (Qle_bool 0 (l - u)) eqn:E; [ psatz Q | exfalso; apply (Qlt_irrefl 0); exact H ].
Qed.

Lemma opt_below_gap_val : forall l u, u < l -> opt_below_gap (Some l) (Some u) == l - u.
Proof.
  intros l u H. cbn [opt_below_gap]. unfold Qmax2.
  destruct (Qle_bool 0 (l - u)) eqn:E; [ reflexivity |].
  apply Qle_bool_false_lt in E. exfalso. psatz Q.
Qed.

Lemma min_sub_eq : forall a b cv, Qmin2 (a - b) (a - cv) == a - Qmax2 b cv.
Proof.
  intros a b cv. unfold Qmin2, Qmax2.
  destruct (Qle_bool b cv) eqn:Ebc; destruct (Qle_bool (a - b) (a - cv)) eqn:E.
  - apply Qle_bool_iff in Ebc; apply Qle_bool_iff in E. psatz Q.
  - reflexivity.
  - reflexivity.
  - apply Qle_bool_false_lt in Ebc; apply Qle_bool_false_lt in E. psatz Q.
Qed.

Lemma min_sub_eq2 : forall a b cv, Qmin2 (b - a) (cv - a) == Qmin2 b cv - a.
Proof.
  intros a b cv. unfold Qmin2.
  destruct (Qle_bool b cv) eqn:Ebc; destruct (Qle_bool (b - a) (cv - a)) eqn:E.
  - reflexivity.
  - apply Qle_bool_iff in Ebc; apply Qle_bool_false_lt in E. psatz Q.
  - apply Qle_bool_false_lt in Ebc; apply Qle_bool_iff in E. psatz Q.
  - reflexivity.
Qed.

Lemma c_func_both_above : forall cst a b cv,
  b < a -> cv < a -> c_func cst a b cv == cst + (a - Qmax2 b cv).
Proof.
  intros cst a b cv Hb Hcv. unfold c_func.
  assert (Hbet : is_between a b cv = false).
  { unfold is_between. apply orb_false_iff; split; apply andb_false_iff.
    - right. destruct (Qle_bool a cv) eqn:E;
        [ apply Qle_bool_iff in E; exfalso; psatz Q | reflexivity ].
    - right. destruct (Qle_bool a b) eqn:E;
        [ apply Qle_bool_iff in E; exfalso; psatz Q | reflexivity ]. }
  rewrite Hbet.
  assert (Hmin : Qmin2 (Qabs (a - b)) (Qabs (a - cv)) == a - Qmax2 b cv).
  { transitivity (Qmin2 (a - b) (a - cv)).
    - apply Qmin2_wd; apply Qabs_pos; psatz Q.
    - apply min_sub_eq. }
  rewrite Hmin. reflexivity.
Qed.

Lemma c_func_both_below : forall cst a b cv,
  a < b -> a < cv -> c_func cst a b cv == cst + (Qmin2 b cv - a).
Proof.
  intros cst a b cv Hb Hcv. unfold c_func.
  assert (Hbet : is_between a b cv = false).
  { unfold is_between. apply orb_false_iff; split; apply andb_false_iff.
    - left. destruct (Qle_bool b a) eqn:E;
        [ apply Qle_bool_iff in E; exfalso; psatz Q | reflexivity ].
    - left. destruct (Qle_bool cv a) eqn:E;
        [ apply Qle_bool_iff in E; exfalso; psatz Q | reflexivity ]. }
  rewrite Hbet.
  assert (Hmin : Qmin2 (Qabs (a - b)) (Qabs (a - cv)) == Qmin2 b cv - a).
  { transitivity (Qmin2 (b - a) (cv - a)).
    - apply Qmin2_wd; (rewrite Qabs_neg by psatz Q); ring.
    - apply min_sub_eq2. }
  rewrite Hmin. reflexivity.
Qed.

Lemma clamp_hi_le : forall hi x, clamp_hi hi x <= x.
Proof. intros [h|] x; cbn [clamp_hi]; [ apply Qmin2_le_r | apply Qle_refl ]. Qed.

Lemma clamp_lo_ge : forall lo x, x <= clamp_lo lo x.
Proof. intros [l|] x; cbn [clamp_lo]; [ apply Qmax2_le_r | apply Qle_refl ]. Qed.

Lemma clamp_in_itv : forall lo hi x,
  itv_nonempty (lo, hi) -> in_itv (clamp_lo lo (clamp_hi hi x)) (lo, hi).
Proof.
  intros lo hi x [w [Hwl Hwh]]. cbn [fst snd] in Hwl, Hwh. split; cbn [fst snd].
  - destruct lo as [l|]; cbn [clamp_lo]; [ apply Qmax2_le_l | exact I ].
  - destruct hi as [h|]; [| exact I].
    destruct lo as [l|]; cbn [clamp_lo clamp_hi].
    + apply Qmax2_lub; [ apply Qle_trans with w; assumption | apply Qmin2_le_l ].
    + apply Qmin2_le_l.
Qed.

(** In the overlap regime (both gaps 0), a concrete in-box pair is between. *)
Lemma split_overlap_witness : forall b alo ahi clo chi,
  itv_nonempty (alo, ahi) -> itv_nonempty (clo, chi) ->
  opt_below_gap alo (opt_max_b b chi) == 0 ->
  opt_below_gap (opt_min_b b clo) ahi == 0 ->
  exists av cv, in_itv av (alo, ahi) /\ in_itv cv (clo, chi) /\ is_between av b cv = true.
Proof.
  intros b alo ahi clo chi Hane Hcne Hga0v Hgb0v.
  exists (clamp_lo alo (clamp_hi ahi b)).
  exists (clamp_lo clo (clamp_hi chi (clamp_lo alo (clamp_hi ahi b)))).
  remember (clamp_lo alo (clamp_hi ahi b)) as av eqn:Havdef.
  split; [ rewrite Havdef; apply clamp_in_itv; exact Hane |].
  split; [ apply clamp_in_itv; exact Hcne |].
  destruct (Qlt_le_dec b av) as [Hbav | Havle].
  - (* b < av: av <= cv, between via b <= av <= cv *)
    assert (Hav_chi : av <= clamp_hi chi av).
    { destruct chi as [ch0|]; cbn [clamp_hi]; [| apply Qle_refl].
      apply Qmin2_glb; [| apply Qle_refl].
      destruct alo as [al|];
        [| exfalso; rewrite Havdef in Hbav; cbn [clamp_lo] in Hbav;
           pose proof (clamp_hi_le ahi b); psatz Q ].
      assert (Hclh : clamp_hi ahi b <= b) by apply clamp_hi_le.
      assert (Haveq : av == al).
      { rewrite Havdef; cbn [clamp_lo]; unfold Qmax2;
        destruct (Qle_bool al (clamp_hi ahi b)) eqn:E; [| reflexivity];
        exfalso; rewrite Havdef in Hbav; cbn [clamp_lo] in Hbav; unfold Qmax2 in Hbav;
        rewrite E in Hbav; psatz Q. }
      assert (Halb : b < al) by (rewrite Haveq in Hbav; exact Hbav).
      cbn [opt_max_b] in Hga0v.
      pose proof (opt_below_gap_zero_le al (Qmax2 b ch0) Hga0v) as Halmax.
      rewrite Haveq.
      assert (Hbch : b <= ch0).
      { destruct (Qlt_le_dec ch0 b) as [H|H]; [ exfalso | exact H ].
        assert (Qmax2 b ch0 == b) by
          (unfold Qmax2; destruct (Qle_bool b ch0) eqn:E2;
           [ apply Qle_bool_iff in E2; psatz Q | reflexivity ]).
        psatz Q. }
      assert (Hmx : Qmax2 b ch0 == ch0) by
        (unfold Qmax2; destruct (Qle_bool b ch0) eqn:E2;
         [ reflexivity | apply Qle_bool_false_lt in E2; psatz Q ]).
      psatz Q. }
    assert (Hav_le_cv : av <= clamp_lo clo (clamp_hi chi av))
      by (apply Qle_trans with (clamp_hi chi av); [ exact Hav_chi | apply clamp_lo_ge ]).
    apply is_between_le_le; [ apply Qlt_le_weak; exact Hbav | exact Hav_le_cv ].
  - destruct (Qle_lt_or_eq _ _ Havle) as [Hlt | Heq].
    + (* av < b: cv <= av, between via cv <= av <= b *)
      assert (Hcv_le_av : clamp_lo clo (clamp_hi chi av) <= av).
      { destruct clo as [cl0|]; cbn [clamp_lo]; [| apply clamp_hi_le].
        apply Qmax2_lub; [| apply clamp_hi_le].
        destruct ahi as [ah|];
          [| exfalso; rewrite Havdef in Hlt; cbn [clamp_lo clamp_hi] in Hlt;
             pose proof (clamp_lo_ge alo b); psatz Q ].
        assert (Hahb : ah <= b).
        { destruct (Qlt_le_dec b ah) as [H|H]; [| exact H].
          exfalso. rewrite Havdef in Hlt; cbn [clamp_hi] in Hlt.
          assert (Hqmb : Qmin2 ah b == b)
            by (unfold Qmin2; destruct (Qle_bool ah b) eqn:E;
                [ apply Qle_bool_iff in E; psatz Q | reflexivity ]).
          assert (b <= clamp_lo alo (Qmin2 ah b))
            by (apply Qle_trans with (Qmin2 ah b);
                [ rewrite Hqmb; apply Qle_refl | apply clamp_lo_ge ]).
          psatz Q. }
        assert (Hqab : Qmin2 ah b == ah)
          by (unfold Qmin2; destruct (Qle_bool ah b) eqn:E;
              [ reflexivity | apply Qle_bool_false_lt in E; psatz Q ]).
        assert (Hah_le_av : ah <= av).
        { rewrite Havdef; cbn [clamp_hi].
          apply Qle_trans with (Qmin2 ah b);
            [ rewrite Hqab; apply Qle_refl | apply clamp_lo_ge ]. }
        cbn [opt_min_b] in Hgb0v.
        pose proof (opt_below_gap_zero_le (Qmin2 b cl0) ah Hgb0v) as Hmincl.
        assert (Hclb : cl0 <= b).
        { destruct (Qlt_le_dec b cl0) as [H|H]; [ exfalso | exact H ].
          assert (Qmin2 b cl0 == b) by
            (unfold Qmin2; destruct (Qle_bool b cl0) eqn:E2;
             [ reflexivity | apply Qle_bool_false_lt in E2; psatz Q ]).
          psatz Q. }
        assert (Hmn : Qmin2 b cl0 == cl0) by
          (unfold Qmin2; destruct (Qle_bool b cl0) eqn:E2;
           [ apply Qle_bool_iff in E2; psatz Q | reflexivity ]).
        psatz Q. }
      apply is_between_ge_ge; [ exact Hcv_le_av | apply Qlt_le_weak; exact Hlt ].
    + (* av == b: total order picks a between side *)
      assert (Hba : b <= av) by (rewrite Heq; apply Qle_refl).
      assert (Hab : av <= b) by (rewrite Heq; apply Qle_refl).
      destruct (Qlt_le_dec (clamp_lo clo (clamp_hi chi av)) b) as [Hcb | Hbc].
      * apply is_between_ge_ge;
          [ apply Qlt_le_weak; apply Qlt_le_trans with b; [ exact Hcb | exact Hba ]
          | exact Hab ].
      * apply is_between_le_le;
          [ exact Hba | apply Qle_trans with b; [ exact Hab | exact Hbc ] ].
Qed.

Lemma c_func_split_lb_tight : forall c_const a_iv b c_iv,
  0 <= c_const -> itv_nonempty a_iv -> itv_nonempty c_iv ->
  exists av cv, in_itv av a_iv /\ in_itv cv c_iv /\
    c_func_split_lb c_const a_iv b c_iv == c_func c_const av b cv.
Proof.
  intros c_const [alo ahi] b [clo chi] Hc Hane Hcne.
  destruct Hane as [aw [Hawl Hawh]]. destruct Hcne as [cw [Hcwl Hcwh]].
  cbn [fst snd] in Hawl, Hawh, Hcwl, Hcwh.
  unfold c_func_split_lb. cbn [fst snd].
  destruct (Qlt_le_dec 0 (opt_below_gap alo (opt_max_b b chi))) as [Hga | Hga0].
  - destruct (opt_below_gap_pos _ _ Hga) as [al [muh [Halo [Hmuh Hlt]]]]. subst alo.
    destruct chi as [ch|]; [| cbn [opt_max_b] in Hmuh; discriminate ].
    cbn [opt_max_b] in Hmuh. injection Hmuh as Hmuh. subst muh.
    assert (Hbch_b : b <= Qmax2 b ch) by apply Qmax2_le_l.
    assert (Hbch_c : ch <= Qmax2 b ch) by apply Qmax2_le_r.
    exists al, ch. repeat split.
    + apply Qle_refl.
    + destruct ahi as [ah|]; [ apply Qle_trans with aw; assumption | exact I ].
    + destruct clo as [cl|]; [ apply Qle_trans with cw; assumption | exact I ].
    + apply Qle_refl.
    + assert (Hgb0 : opt_below_gap (opt_min_b b clo) ahi == 0).
      { destruct (opt_min_b b clo) as [ul|] eqn:Eul; destruct ahi as [ah|];
          cbn [opt_below_gap]; try reflexivity.
        unfold Qmax2. destruct (Qle_bool 0 (ul - ah)) eqn:E; [| reflexivity].
        apply Qle_bool_iff in E. exfalso.
        assert (ul <= b).
        { destruct clo as [cl|]; cbn [opt_min_b] in Eul; [| discriminate].
          injection Eul as Eul; subst ul. apply Qmin2_le_l. }
        psatz Q. }
      rewrite (c_func_both_above c_const al b ch) by psatz Q.
      assert (Hgaval : opt_below_gap (Some al) (opt_max_b b (Some ch)) == al - Qmax2 b ch)
        by (cbn [opt_max_b]; apply opt_below_gap_val; exact Hlt).
      assert (Hqm : Qmax2 (opt_below_gap (Some al) (opt_max_b b (Some ch)))
                          (opt_below_gap (opt_min_b b clo) ahi) == al - Qmax2 b ch).
      { rewrite (Qmax2_wd _ (al - Qmax2 b ch) _ 0 Hgaval Hgb0).
        unfold Qmax2 at 1; destruct (Qle_bool (al - Qmax2 b ch) 0) eqn:E;
          [ apply Qle_bool_iff in E; exfalso; psatz Q | reflexivity ]. }
      rewrite Hqm. ring.
  - destruct (Qlt_le_dec 0 (opt_below_gap (opt_min_b b clo) ahi)) as [Hgb | Hgb0].
    + destruct (opt_below_gap_pos _ _ Hgb) as [mul [ah [Hmul [Hahi Hlt]]]]. subst ahi.
      destruct clo as [cl|]; [| cbn [opt_min_b] in Hmul; discriminate ].
      cbn [opt_min_b] in Hmul. injection Hmul as Hmul. subst mul.
      assert (Hbcl_b : Qmin2 b cl <= b) by apply Qmin2_le_l.
      assert (Hbcl_c : Qmin2 b cl <= cl) by apply Qmin2_le_r.
      exists ah, cl. repeat split.
      * destruct alo as [al|]; [ apply Qle_trans with aw; assumption | exact I ].
      * apply Qle_refl.
      * apply Qle_refl.
      * destruct chi as [ch|]; [ apply Qle_trans with cw; assumption | exact I ].
      * assert (Hga0' : opt_below_gap alo (opt_max_b b chi) == 0).
        { destruct alo as [al|] eqn:Ealo; destruct (opt_max_b b chi) as [uh|] eqn:Euh;
            cbn [opt_below_gap]; try reflexivity.
          unfold Qmax2. destruct (Qle_bool 0 (al - uh)) eqn:E; [| reflexivity].
          apply Qle_bool_iff in E. exfalso.
          assert (b <= uh).
          { destruct chi as [ch|]; cbn [opt_max_b] in Euh; [| discriminate].
            injection Euh as Euh; subst uh. apply Qmax2_le_l. }
          psatz Q. }
        rewrite (c_func_both_below c_const ah b cl) by psatz Q.
        assert (Hgbval : opt_below_gap (opt_min_b b (Some cl)) (Some ah) == Qmin2 b cl - ah)
          by (cbn [opt_min_b]; apply opt_below_gap_val; exact Hlt).
        assert (Hqm : Qmax2 (opt_below_gap alo (opt_max_b b chi))
                            (opt_below_gap (opt_min_b b (Some cl)) (Some ah)) == Qmin2 b cl - ah).
        { rewrite (Qmax2_wd _ 0 _ (Qmin2 b cl - ah) Hga0' Hgbval).
          unfold Qmax2 at 1; destruct (Qle_bool 0 (Qmin2 b cl - ah)) eqn:E;
            [ reflexivity | apply Qle_bool_false_lt in E; exfalso; psatz Q ]. }
        rewrite Hqm. ring.
    + assert (Hga0v : opt_below_gap alo (opt_max_b b chi) == 0)
        by (apply Qle_antisym; [ exact Hga0 | apply opt_below_gap_nonneg ]).
      assert (Hgb0v : opt_below_gap (opt_min_b b clo) ahi == 0)
        by (apply Qle_antisym; [ exact Hgb0 | apply opt_below_gap_nonneg ]).
      destruct (split_overlap_witness b alo ahi clo chi
                 (ex_intro _ aw (conj Hawl Hawh)) (ex_intro _ cw (conj Hcwl Hcwh)) Hga0v Hgb0v)
        as [av [cv [Havin [Hcvin Hbet]]]].
      exists av, cv. repeat split;
        [ exact (proj1 Havin) | exact (proj2 Havin)
        | exact (proj1 Hcvin) | exact (proj2 Hcvin) |].
      unfold c_func. rewrite Hbet.
      assert (Hmax0 : Qmax2 (opt_below_gap alo (opt_max_b b chi))
                            (opt_below_gap (opt_min_b b clo) ahi) == 0).
      { rewrite (Qmax2_wd _ 0 _ 0 Hga0v Hgb0v). unfold Qmax2. reflexivity. }
      rewrite Hmax0. ring.
Qed.
