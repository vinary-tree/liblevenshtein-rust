(** * MSM C Function Bounds for Triangle Proofs *)

From Stdlib Require Import Bool Psatz.
From Stdlib Require Import QArith Qabs Qminmax.
From Liblevenshtein.MSM Require Import MsmDefinitions CFunction.

Definition half_abs (a b : Q) : Q :=
  Qabs_diff a b * (1#2).

Lemma is_between_true_cases : forall a b t,
  is_between a b t = true ->
  (b <= a /\ a <= t) \/ (t <= a /\ a <= b).
Proof.
  intros a b t Hbetween.
  unfold is_between in Hbetween.
  apply orb_true_iff in Hbetween.
  destruct Hbetween as [Hleft | Hright].
  - apply andb_true_iff in Hleft.
    destruct Hleft as [Hba Hat].
    left. split; apply Qle_bool_iff; assumption.
  - apply andb_true_iff in Hright.
    destruct Hright as [Hta Hab].
    right. split; apply Qle_bool_iff; assumption.
Qed.

Lemma is_between_false_above : forall a b t,
  b <= a ->
  is_between a b t = false ->
  t < a.
Proof.
  intros a b t Hba Hbetween.
  unfold is_between in Hbetween.
  assert (Hba_bool : Qle_bool b a = true) by (apply Qle_bool_iff; exact Hba).
  apply orb_false_iff in Hbetween.
  destruct Hbetween as [Hleft _].
  apply andb_false_iff in Hleft.
  destruct Hleft as [Hba_false | Hat_false].
  - rewrite Hba_bool in Hba_false. discriminate.
  - apply Qle_bool_false_lt in Hat_false. exact Hat_false.
Qed.

Lemma is_between_false_below : forall a b t,
  a <= b ->
  is_between a b t = false ->
  a < t.
Proof.
  intros a b t Hab Hbetween.
  unfold is_between in Hbetween.
  assert (Hab_bool : Qle_bool a b = true) by (apply Qle_bool_iff; exact Hab).
  apply orb_false_iff in Hbetween.
  destruct Hbetween as [_ Hright].
  apply andb_false_iff in Hright.
  destruct Hright as [Hta_false | Hab_false].
  - apply Qle_bool_false_lt in Hta_false. exact Hta_false.
  - rewrite Hab_bool in Hab_false. discriminate.
Qed.

Lemma Qabs_interval_right_le : forall z a y,
  z < a ->
  a <= y ->
  Qabs (a - z) <= Qabs (y - z).
Proof.
  intros z a y Hza Hay.
  rewrite Qabs_pos by (apply Qlt_le_weak; psatz Q).
  rewrite Qabs_pos by (apply Qlt_le_weak; psatz Q).
  psatz Q.
Qed.

Lemma Qabs_interval_left_le : forall y a z,
  y <= a ->
  a < z ->
  Qabs (a - z) <= Qabs (y - z).
Proof.
  intros y a z Hya Haz.
  rewrite Qabs_neg by (apply Qlt_le_weak; psatz Q).
  rewrite Qabs_neg by (apply Qlt_le_weak; psatz Q).
  psatz Q.
Qed.

Lemma Qmin2_lipschitz_r : forall u v w d,
  0 <= d ->
  v <= w + d ->
  Qmin2 u v <= Qmin2 u w + d.
Proof.
  intros u v w d Hd Hv.
  change (Qmin2 u v <= (if Qle_bool u w then u else w) + d).
  destruct (Qle_bool u w).
  - eapply Qle_trans.
    + apply Qmin2_le_l.
    + apply Qle_plus_nonneg_r. exact Hd.
  - eapply Qle_trans.
    + apply Qmin2_le_r.
    + exact Hv.
Qed.

Lemma c_func_between_not_between_bound : forall a b y z,
  is_between a b y = true ->
  is_between a b z = false ->
  Qmin2 (Qabs (a - b)) (Qabs (a - z)) <= Qabs (y - z).
Proof.
  intros a b y z Hy Hz.
  destruct (is_between_true_cases a b y Hy) as [[Hba Hay] | [Hya Hab]].
  - eapply Qle_trans.
    + apply Qmin2_le_r.
    + apply Qabs_interval_right_le.
      * eapply is_between_false_above; eassumption.
      * exact Hay.
  - eapply Qle_trans.
    + apply Qmin2_le_r.
    + apply Qabs_interval_left_le.
      * exact Hya.
      * eapply is_between_false_below; eassumption.
Qed.

Lemma c_func_lipschitz_context : forall c_const a b y z,
  c_func c_const a b z <= c_func c_const a b y + Qabs_diff y z.
Proof.
  intros c_const a b y z.
  unfold c_func.
  destruct (is_between a b z) eqn:Hz;
  destruct (is_between a b y) eqn:Hy;
  unfold Qabs_diff.
  - apply Qle_plus_nonneg_r. apply Qabs_nonneg.
  - eapply Qle_trans.
    + apply Qle_plus_nonneg_r. apply Qmin2_glb; apply Qabs_nonneg.
    + apply Qle_plus_nonneg_r.
      apply Qabs_nonneg.
  - apply Qplus_le_compat.
    + apply Qle_refl.
    + eapply c_func_between_not_between_bound; eassumption.
  - setoid_replace
      (c_const + Qmin2 (Qabs (a - b)) (Qabs (a - y)) + Qabs (y - z))
      with
      (c_const + (Qmin2 (Qabs (a - b)) (Qabs (a - y)) + Qabs (y - z)))
      by ring.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + apply Qmin2_lipschitz_r.
      * apply Qabs_nonneg.
      * setoid_replace (a - z) with ((a - y) + (y - z)) by ring.
        apply Qabs_triangle.
Qed.

Lemma Qmin2_eq_l : forall a b,
  a <= b ->
  Qmin2 a b == a.
Proof.
  intros a b Hab.
  unfold Qmin2.
  destruct (Qle_bool a b) eqn:Hle.
  - reflexivity.
  - apply Qle_bool_false_lt in Hle.
    exfalso. exact (Qlt_not_le _ _ Hle Hab).
Qed.

Lemma Qmin2_eq_r : forall a b,
  b <= a ->
  Qmin2 a b == b.
Proof.
  intros a b Hba.
  unfold Qmin2.
  destruct (Qle_bool a b) eqn:Hle.
  - apply Qle_bool_iff in Hle.
    apply Qle_antisym; assumption.
  - reflexivity.
Qed.

Lemma Qabs_diff_add_between : forall x y z,
  x <= y ->
  y <= z ->
  Qabs_diff x z == Qabs_diff x y + Qabs_diff y z.
Proof.
  intros x y z Hxy Hyz.
  unfold Qabs_diff.
  rewrite Qabs_neg by psatz Q.
  rewrite Qabs_neg by psatz Q.
  rewrite Qabs_neg by psatz Q.
  ring.
Qed.

Lemma is_between_false_cases : forall a b t,
  is_between a b t = false ->
  (a < b /\ a < t) \/ (b < a /\ t < a).
Proof.
  intros a b t Hbetween.
  unfold is_between in Hbetween.
  apply orb_false_iff in Hbetween.
  destruct Hbetween as [Hleft Hright].
  apply andb_false_iff in Hleft.
  apply andb_false_iff in Hright.
  destruct (Qlt_le_dec a b) as [Hab_lt | Hba].
  - left. split; [exact Hab_lt |].
    destruct Hright as [Hta_false | Hab_false].
    + apply Qle_bool_false_lt in Hta_false. exact Hta_false.
    + assert (Hab_bool : Qle_bool a b = true)
        by (apply Qle_bool_iff; apply Qlt_le_weak; exact Hab_lt).
      rewrite Hab_bool in Hab_false. discriminate.
  - right.
    destruct Hleft as [Hba_false | Hat_false].
    + assert (Hba_bool : Qle_bool b a = true)
        by (apply Qle_bool_iff; exact Hba).
      rewrite Hba_bool in Hba_false. discriminate.
    + apply Qle_bool_false_lt in Hat_false.
      split.
      * destruct Hright as [Hta_false | Hab_false].
        -- assert (Hta_bool : Qle_bool t a = true)
             by (apply Qle_bool_iff; apply Qlt_le_weak; exact Hat_false).
           rewrite Hta_bool in Hta_false. discriminate.
        -- apply Qle_bool_false_lt in Hab_false. exact Hab_false.
      * exact Hat_false.
Qed.

Lemma c_func_between_abs_sum : forall a b t,
  is_between a b t = true ->
  Qabs_diff b t == Qabs_diff a b + Qabs_diff a t.
Proof.
  intros a b t Hbetween.
  destruct (is_between_true_cases a b t Hbetween) as [[Hba Hat] | [Hta Hab]].
  - pose proof (Qabs_diff_add_between b a t Hba Hat) as Hsum.
    rewrite Hsum.
    rewrite Qabs_diff_symm.
    reflexivity.
  - pose proof (Qabs_diff_add_between t a b Hta Hab) as Hsum.
    rewrite (Qabs_diff_symm b t).
    rewrite Hsum.
    rewrite (Qabs_diff_symm t a).
    ring.
Qed.

Lemma c_func_not_between_min_half : forall a b t,
  is_between a b t = false ->
  Qmin2 (Qabs_diff a b) (Qabs_diff a t) +
    Qabs_diff b t * (1#2) ==
  (Qabs_diff a b + Qabs_diff a t) * (1#2).
Proof.
  intros a b t Hbetween.
  destruct (is_between_false_cases a b t Hbetween) as [[Hab Hat] | [Hba Hta]].
  - destruct (Qle_bool b t) eqn:Hbt.
    + apply Qle_bool_iff in Hbt.
      assert (Hsum : Qabs_diff a t == Qabs_diff a b + Qabs_diff b t).
      { apply Qabs_diff_add_between; [apply Qlt_le_weak; exact Hab | exact Hbt]. }
      rewrite (Qmin2_eq_l (Qabs_diff a b) (Qabs_diff a t)).
      * rewrite Hsum. ring.
      * rewrite Hsum. apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
    + apply Qle_bool_false_lt in Hbt.
      assert (Hsum : Qabs_diff a b == Qabs_diff a t + Qabs_diff b t).
      { rewrite (Qabs_diff_symm b t).
        apply Qabs_diff_add_between.
        - apply Qlt_le_weak. exact Hat.
        - apply Qlt_le_weak. exact Hbt. }
      rewrite (Qmin2_eq_r (Qabs_diff a b) (Qabs_diff a t)).
      * rewrite Hsum. ring.
      * rewrite Hsum.
        apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
  - destruct (Qle_bool b t) eqn:Hbt.
    + apply Qle_bool_iff in Hbt.
      assert (Hsum : Qabs_diff a b == Qabs_diff b t + Qabs_diff a t).
      { rewrite Qabs_diff_symm.
        rewrite (Qabs_diff_symm a t).
        apply Qabs_diff_add_between.
        - exact Hbt.
        - apply Qlt_le_weak. exact Hta. }
      rewrite (Qmin2_eq_r (Qabs_diff a b) (Qabs_diff a t)).
      * rewrite Hsum. ring.
      * rewrite Hsum.
        setoid_replace (Qabs_diff b t + Qabs_diff a t)
          with (Qabs_diff a t + Qabs_diff b t) by ring.
        apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
    + apply Qle_bool_false_lt in Hbt.
      assert (Hsum : Qabs_diff a t == Qabs_diff b t + Qabs_diff a b).
      { rewrite Qabs_diff_symm.
        rewrite (Qabs_diff_symm b t).
        rewrite (Qabs_diff_symm a b).
        apply Qabs_diff_add_between.
        - apply Qlt_le_weak. exact Hbt.
        - apply Qlt_le_weak. exact Hba. }
      rewrite (Qmin2_eq_l (Qabs_diff a b) (Qabs_diff a t)).
      * rewrite Hsum. ring.
      * rewrite Hsum.
        setoid_replace (Qabs_diff b t + Qabs_diff a b)
          with (Qabs_diff a b + Qabs_diff b t) by ring.
        apply Qle_plus_nonneg_r. apply Qabs_diff_nonneg.
Qed.

Lemma c_func_singleton_target_potential : forall c_const a b t,
  c_func c_const a b t + Qabs_diff b t * (1#2) ==
  c_const + (Qabs_diff a b + Qabs_diff a t) * (1#2).
Proof.
  intros c_const a b t.
  unfold c_func.
  destruct (is_between a b t) eqn:Hbetween.
  - rewrite (c_func_between_abs_sum a b t Hbetween).
    ring.
  - fold (Qabs_diff a b).
    fold (Qabs_diff a t).
    setoid_replace
      (c_const + Qmin2 (Qabs_diff a b) (Qabs_diff a t) +
         Qabs_diff b t * (1#2))
      with
      (c_const +
        (Qmin2 (Qabs_diff a b) (Qabs_diff a t) +
           Qabs_diff b t * (1#2)))
      by ring.
    rewrite (c_func_not_between_min_half a b t Hbetween).
    ring.
Qed.

Lemma c_func_target_append_potential_step : forall c_const a b y t,
  c_func c_const a b t + Qabs_diff b t * (1#2) -
    Qabs_diff b a * (1#2) ==
  c_func c_const a y t + Qabs_diff y t * (1#2) -
    Qabs_diff y a * (1#2).
Proof.
  intros c_const a b y t.
  pose proof (c_func_singleton_target_potential c_const a b t) as Hb.
  pose proof (c_func_singleton_target_potential c_const a y t) as Hy.
  setoid_replace (Qabs_diff b a) with (Qabs_diff a b)
    by (rewrite Qabs_diff_symm; reflexivity).
  setoid_replace (Qabs_diff y a) with (Qabs_diff a y)
    by (rewrite Qabs_diff_symm; reflexivity).
  setoid_replace
    (c_func c_const a b t + Qabs_diff b t * (1#2) -
      Qabs_diff a b * (1#2))
    with
    (c_const + (Qabs_diff a b + Qabs_diff a t) * (1#2) -
      Qabs_diff a b * (1#2)).
  2:{ rewrite <- Hb. ring. }
  setoid_replace
    (c_func c_const a y t + Qabs_diff y t * (1#2) -
      Qabs_diff a y * (1#2))
    with
    (c_const + (Qabs_diff a y + Qabs_diff a t) * (1#2) -
      Qabs_diff a y * (1#2)).
  2:{ rewrite <- Hy. ring. }
  ring.
Qed.

Lemma c_func_endpoint_step_bound : forall c_const a b t,
  0 <= c_const ->
  Qabs_diff a t <= Qabs_diff b t + c_func c_const a b t.
Proof.
  intros c_const a b t Hc.
  pose proof (c_func_singleton_target_potential c_const a b t) as Hpot.
  pose proof (Qabs_triangle_diff a b t) as Htri.
  setoid_replace (Qabs_diff b t + c_func c_const a b t)
    with
    ((c_func c_const a b t + Qabs_diff b t * (1#2)) +
      Qabs_diff b t * (1#2))
    by ring.
  rewrite Hpot.
  psatz Q.
Qed.

Lemma Qhalf_nonneg : forall a,
  0 <= a ->
  0 <= a * (1#2).
Proof.
  intros a Ha.
  apply Qmult_le_0_compat.
  - exact Ha.
  - unfold Qle. simpl. lia.
Qed.

Lemma Qhalf_le_self : forall a,
  0 <= a ->
  a * (1#2) <= a.
Proof.
  intros a Ha.
  setoid_replace (a * (1#2)) with ((1#2) * a) by ring.
  setoid_replace a with (1 * a) at 2 by ring.
  apply Qmult_le_compat_r.
  - unfold Qle. simpl. lia.
  - exact Ha.
Qed.

Lemma Qle_sub_nonneg : forall a b,
  b <= a ->
  0 <= a - b.
Proof.
  intros a b Hba.
  apply (proj1 (Qplus_le_l 0 (a - b) b)).
  ring_simplify.
  exact Hba.
Qed.

Lemma half_abs_endpoint_step_bound : forall c_const a b t,
  0 <= c_const ->
  half_abs a t <= half_abs b t + c_func c_const a b t.
Proof.
  intros c_const a b t Hc.
  unfold half_abs.
  pose proof (c_func_endpoint_step_bound c_const a b t Hc) as Hend.
  pose proof (c_func_nonneg c_const a b t Hc) as HC.
  eapply Qle_trans.
  - apply Qmult_le_compat_r.
    + exact Hend.
    + unfold Qle. simpl. lia.
  - setoid_replace
      ((Qabs_diff b t + c_func c_const a b t) * (1#2))
      with
      (Qabs_diff b t * (1#2) + c_func c_const a b t * (1#2))
      by ring.
    apply Qplus_le_compat.
    + apply Qle_refl.
    + apply Qhalf_le_self. exact HC.
Qed.

Lemma c_func_singleton_potential_delta : forall c_const a b t,
  c_func c_const a b t - half_abs a t + half_abs b t ==
  c_const + half_abs a b.
Proof.
  intros c_const a b t.
  unfold half_abs.
  pose proof (c_func_singleton_target_potential c_const a b t) as Hpot.
  setoid_replace
    (c_func c_const a b t - Qabs_diff a t * (1#2) +
      Qabs_diff b t * (1#2))
    with
    ((c_func c_const a b t + Qabs_diff b t * (1#2)) -
      Qabs_diff a t * (1#2))
    by ring.
  rewrite Hpot.
  ring.
Qed.

Lemma c_func_singleton_context_switch : forall c_const a b u t,
  c_func c_const a b t - half_abs a t + half_abs b t ==
  c_func c_const a b u - half_abs a u + half_abs b u.
Proof.
  intros c_const a b u t.
  rewrite (c_func_singleton_potential_delta c_const a b t).
  rewrite (c_func_singleton_potential_delta c_const a b u).
  reflexivity.
Qed.

Lemma c_func_left_potential_step_nonneg : forall c_const x y y_prev z,
  0 <= c_const ->
  0 <=
    c_func c_const y x y_prev - half_abs x y +
    c_func c_const y y_prev z - half_abs y z +
    half_abs x y_prev + half_abs y_prev z.
Proof.
  intros c_const x y y_prev z Hc.
  pose proof (half_abs_endpoint_step_bound c_const y y_prev x Hc) as Hxy.
  pose proof (half_abs_endpoint_step_bound c_const y y_prev z Hc) as Hyz.
  setoid_replace (half_abs y x) with (half_abs x y) in Hxy
    by (unfold half_abs; rewrite Qabs_diff_symm; reflexivity).
  setoid_replace (half_abs y_prev x) with (half_abs x y_prev) in Hxy
    by (unfold half_abs; rewrite Qabs_diff_symm; reflexivity).
  setoid_replace (c_func c_const y y_prev x) with
    (c_func c_const y x y_prev) in Hxy
    by (rewrite c_func_symm_bc; reflexivity).
  assert (Hleft : 0 <=
    c_func c_const y x y_prev - half_abs x y + half_abs x y_prev).
  { setoid_replace
      (c_func c_const y x y_prev - half_abs x y + half_abs x y_prev)
      with
      (half_abs x y_prev + c_func c_const y x y_prev - half_abs x y)
      by ring.
    apply Qle_sub_nonneg.
    exact Hxy. }
  assert (Hright : 0 <=
    c_func c_const y y_prev z - half_abs y z + half_abs y_prev z).
  { setoid_replace
      (c_func c_const y y_prev z - half_abs y z + half_abs y_prev z)
      with
      (half_abs y_prev z + c_func c_const y y_prev z - half_abs y z)
      by ring.
    apply Qle_sub_nonneg.
    exact Hyz. }
  setoid_replace
    (c_func c_const y x y_prev - half_abs x y +
      c_func c_const y y_prev z - half_abs y z +
      half_abs x y_prev + half_abs y_prev z)
    with
    ((c_func c_const y x y_prev - half_abs x y + half_abs x y_prev) +
     (c_func c_const y y_prev z - half_abs y z + half_abs y_prev z))
    by ring.
  setoid_replace 0 with (0 + 0) by ring.
  apply Qplus_le_compat; assumption.
Qed.

Lemma c_func_diag_potential_step : forall c_const x x_prev y y_prev z,
  c_func c_const x x_prev z - half_abs x z + half_abs x_prev z <=
  Qabs_diff x y - half_abs x y +
  c_func c_const y y_prev z - half_abs y z + half_abs y_prev z +
  half_abs x_prev y_prev.
Proof.
  intros c_const x x_prev y y_prev z.
  pose proof (Qabs_triangle_diff x y x_prev) as Hxyx.
  pose proof (Qabs_triangle_diff y y_prev x_prev) as Hyypx.
  setoid_replace (Qabs_diff y_prev x_prev) with
    (Qabs_diff x_prev y_prev) in Hyypx
    by (rewrite Qabs_diff_symm; reflexivity).
  assert (Hfull :
    Qabs_diff x x_prev <=
    Qabs_diff x y + Qabs_diff y y_prev + Qabs_diff x_prev y_prev).
  { eapply Qle_trans.
    - exact Hxyx.
    - setoid_replace
        (Qabs_diff x y + Qabs_diff y y_prev + Qabs_diff x_prev y_prev)
        with
        (Qabs_diff x y + (Qabs_diff y y_prev + Qabs_diff x_prev y_prev))
        by ring.
      apply Qplus_le_compat.
      + apply Qle_refl.
      + exact Hyypx. }
  setoid_replace
    (c_func c_const x x_prev z - half_abs x z + half_abs x_prev z)
    with
    (c_const + half_abs x x_prev).
  2:{
    rewrite (c_func_singleton_potential_delta c_const x x_prev z).
    ring.
  }
  setoid_replace
    (Qabs_diff x y - half_abs x y +
      c_func c_const y y_prev z - half_abs y z + half_abs y_prev z +
      half_abs x_prev y_prev)
    with
    (Qabs_diff x y - half_abs x y +
      (c_const + half_abs y y_prev) + half_abs x_prev y_prev).
  2:{
    setoid_replace
      (Qabs_diff x y - half_abs x y +
        c_func c_const y y_prev z - half_abs y z + half_abs y_prev z +
        half_abs x_prev y_prev)
      with
      (Qabs_diff x y - half_abs x y +
        (c_func c_const y y_prev z - half_abs y z + half_abs y_prev z) +
        half_abs x_prev y_prev)
      by ring.
    rewrite (c_func_singleton_potential_delta c_const y y_prev z).
    ring.
  }
  setoid_replace (Qabs_diff x y - half_abs x y)
    with (half_abs x y)
    by (unfold half_abs; ring).
  setoid_replace
    (half_abs x y + (c_const + half_abs y y_prev) +
      half_abs x_prev y_prev)
    with
    (c_const +
      (half_abs x y + half_abs y y_prev + half_abs x_prev y_prev))
    by ring.
  apply Qplus_le_compat.
  - apply Qle_refl.
  - unfold half_abs.
    eapply Qle_trans.
    + apply Qmult_le_compat_r.
      * exact Hfull.
      * unfold Qle. simpl. lia.
    + ring_simplify.
      apply Qle_refl.
Qed.
