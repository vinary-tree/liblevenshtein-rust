(** Machine-checked arithmetic obligations for Edit distance with Real Penalty.

    This development models a concrete ERP alignment as a list of match,
    deletion, and insertion edits. It proves the gap-mass lower bound for every
    alignment, and proves that zero-cost alignments identify sequences only
    after quotienting away occurrences of the gap value [g]. *)

From Stdlib Require Import Reals Lra List.

Import ListNotations.
Open Scope R_scope.

Definition interval_dist (x lo hi : R) : R :=
  if Rlt_dec x lo then lo - x
  else if Rlt_dec hi x then x - hi
  else 0.

Lemma interval_dist_nonnegative : forall x lo hi,
  0 <= interval_dist x lo hi.
Proof.
  intros x lo hi. unfold interval_dist.
  destruct (Rlt_dec x lo); destruct (Rlt_dec hi x); lra.
Qed.

Lemma interval_dist_admissible : forall x lo hi y,
  lo <= hi -> lo <= y <= hi ->
  interval_dist x lo hi <= Rabs (x - y).
Proof.
  intros x lo hi y Hbox [Hlo Hhi]. unfold interval_dist.
  destruct (Rlt_dec x lo) as [Hxlo | Hxlo].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec hi x) as [Hhix | Hhix].
    + rewrite Rabs_right; lra.
    + pose proof (Rabs_pos (x - y)). lra.
Qed.

Lemma interval_dist_degenerate : forall x y,
  interval_dist x y y = Rabs (x - y).
Proof.
  intros x y. unfold interval_dist.
  destruct (Rlt_dec x y) as [Hxy | Hxy].
  - rewrite Rabs_left; lra.
  - destruct (Rlt_dec y x) as [Hyx | Hyx].
    + rewrite Rabs_right; lra.
    + assert (x = y) by lra. subst. replace (y - y) with 0 by ring.
      rewrite Rabs_R0. reflexivity.
Qed.

Lemma reverse_abs_difference : forall x y g,
  Rabs (Rabs (x - g) - Rabs (y - g)) <= Rabs (x - y).
Proof.
  intros x y g.
  pose proof (Rabs_triang_inv2 (x - g) (y - g)) as H.
  replace ((x - g) - (y - g)) with (x - y) in H by ring.
  exact H.
Qed.

Inductive erp_edit : Type :=
| MatchEdit (x y : R)
| DeleteEdit (x : R)
| InsertEdit (y : R).

Definition edit_left_mass (g : R) (edit : erp_edit) : R :=
  match edit with
  | MatchEdit x _ | DeleteEdit x => Rabs (x - g)
  | InsertEdit _ => 0
  end.

Definition edit_right_mass (g : R) (edit : erp_edit) : R :=
  match edit with
  | MatchEdit _ y | InsertEdit y => Rabs (y - g)
  | DeleteEdit _ => 0
  end.

Definition edit_cost (g : R) (edit : erp_edit) : R :=
  match edit with
  | MatchEdit x y => Rabs (x - y)
  | DeleteEdit x => Rabs (x - g)
  | InsertEdit y => Rabs (y - g)
  end.

Fixpoint script_left_mass (g : R) (script : list erp_edit) : R :=
  match script with
  | [] => 0
  | edit :: tail => edit_left_mass g edit + script_left_mass g tail
  end.

Fixpoint script_right_mass (g : R) (script : list erp_edit) : R :=
  match script with
  | [] => 0
  | edit :: tail => edit_right_mass g edit + script_right_mass g tail
  end.

Fixpoint script_cost (g : R) (script : list erp_edit) : R :=
  match script with
  | [] => 0
  | edit :: tail => edit_cost g edit + script_cost g tail
  end.

Lemma edit_potential_bound : forall g edit,
  Rabs (edit_left_mass g edit - edit_right_mass g edit) <= edit_cost g edit.
Proof.
  intros g [x y | x | y]; simpl.
  - apply reverse_abs_difference.
  - rewrite Rminus_0_r, Rabs_pos_eq; [reflexivity | apply Rabs_pos].
  - rewrite Rminus_0_l, Rabs_Ropp, Rabs_pos_eq; [reflexivity | apply Rabs_pos].
Qed.

Lemma edit_cost_nonnegative : forall g edit, 0 <= edit_cost g edit.
Proof.
  intros g [x y | x | y]; simpl; apply Rabs_pos.
Qed.

Lemma script_cost_nonnegative : forall g script, 0 <= script_cost g script.
Proof.
  intros g script. induction script as [|edit tail IH]; simpl.
  - lra.
  - pose proof (edit_cost_nonnegative g edit). lra.
Qed.

Theorem script_gap_mass_bound : forall g script,
  Rabs (script_left_mass g script - script_right_mass g script)
  <= script_cost g script.
Proof.
  intros g script. induction script as [|edit tail IH]; simpl.
  - replace (0 - 0) with 0 by ring. rewrite Rabs_R0. lra.
  - replace
      (edit_left_mass g edit + script_left_mass g tail -
       (edit_right_mass g edit + script_right_mass g tail))
      with
      ((edit_left_mass g edit - edit_right_mass g edit) +
       (script_left_mass g tail - script_right_mass g tail)) by ring.
    eapply Rle_trans.
    + apply Rabs_triang.
    + pose proof (edit_potential_bound g edit). lra.
Qed.

Fixpoint source_of (script : list erp_edit) : list R :=
  match script with
  | [] => []
  | MatchEdit x _ :: tail | DeleteEdit x :: tail => x :: source_of tail
  | InsertEdit _ :: tail => source_of tail
  end.

Fixpoint target_of (script : list erp_edit) : list R :=
  match script with
  | [] => []
  | MatchEdit _ y :: tail | InsertEdit y :: tail => y :: target_of tail
  | DeleteEdit _ :: tail => target_of tail
  end.

Fixpoint gap_mass (g : R) (series : list R) : R :=
  match series with
  | [] => 0
  | value :: tail => Rabs (value - g) + gap_mass g tail
  end.

Lemma source_mass_projection : forall g script,
  script_left_mass g script = gap_mass g (source_of script).
Proof.
  intros g script. induction script as [|[x y | x | y] tail IH]; simpl; lra.
Qed.

Lemma target_mass_projection : forall g script,
  script_right_mass g script = gap_mass g (target_of script).
Proof.
  intros g script. induction script as [|[x y | x | y] tail IH]; simpl; lra.
Qed.

Theorem erp_candidate_lower_bound : forall g script,
  Rabs (gap_mass g (source_of script) - gap_mass g (target_of script))
  <= script_cost g script.
Proof.
  intros g script.
  rewrite <- source_mass_projection, <- target_mass_projection.
  apply script_gap_mass_bound.
Qed.

Fixpoint strip_gap (g : R) (series : list R) : list R :=
  match series with
  | [] => []
  | value :: tail =>
      if Req_EM_T value g then strip_gap g tail
      else value :: strip_gap g tail
  end.

Definition erp_quotient_equiv (g : R) (x y : list R) : Prop :=
  strip_gap g x = strip_gap g y.

Lemma erp_quotient_reflexive : forall g x, erp_quotient_equiv g x x.
Proof. intros g x. reflexivity. Qed.

Lemma erp_quotient_symmetric : forall g x y,
  erp_quotient_equiv g x y -> erp_quotient_equiv g y x.
Proof. intros g x y H. symmetry. exact H. Qed.

Lemma erp_quotient_transitive : forall g x y z,
  erp_quotient_equiv g x y -> erp_quotient_equiv g y z ->
  erp_quotient_equiv g x z.
Proof. intros g x y z Hxy Hyz. unfold erp_quotient_equiv in *. congruence. Qed.

Lemma strip_gap_head : forall g tail, strip_gap g (g :: tail) = strip_gap g tail.
Proof.
  intros g tail. simpl. destruct (Req_EM_T g g); [reflexivity | contradiction].
Qed.

Lemma abs_zero_implies_zero : forall value,
  Rabs value = 0 -> value = 0.
Proof.
  intros value Habs. destruct (Req_EM_T value 0) as [Hzero | Hnonzero].
  - exact Hzero.
  - exfalso. exact (Rabs_no_R0 value Hnonzero Habs).
Qed.

Theorem zero_cost_alignment_has_quotient_identity : forall g script,
  script_cost g script = 0 ->
  erp_quotient_equiv g (source_of script) (target_of script).
Proof.
  intros g script. induction script as [|edit tail IH]; intros Hzero; simpl in *.
  - reflexivity.
  - pose proof (edit_cost_nonnegative g edit) as Hedit.
    pose proof (script_cost_nonnegative g tail) as Htail.
    assert (Hedit_zero : edit_cost g edit = 0) by lra.
    assert (Htail_zero : script_cost g tail = 0) by lra.
    specialize (IH Htail_zero). unfold erp_quotient_equiv in IH |- *.
    destruct edit as [x y | x | y]; simpl in *.
    + apply abs_zero_implies_zero in Hedit_zero.
      assert (x = y) by lra. subst y.
      destruct (Req_EM_T x g); simpl.
      * exact IH.
      * f_equal. exact IH.
    + apply abs_zero_implies_zero in Hedit_zero.
      assert (x = g) by lra. subst x.
      destruct (Req_EM_T g g); [exact IH | contradiction].
    + apply abs_zero_implies_zero in Hedit_zero.
      assert (y = g) by lra. subst y.
      destruct (Req_EM_T g g); [exact IH | contradiction].
Qed.
