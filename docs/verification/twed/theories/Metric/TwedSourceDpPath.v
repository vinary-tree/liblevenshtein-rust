(** * Finite natural-cost DP and lawful alignment paths

    This file imports the exact finite-path model for the 2007 infinite-axis
    TWED recurrence. It establishes the origin, axis, and interior equations
    for a separate executable-on-paper dynamic program. The carrier is nat;
    None represents an unavailable/infinite cell or an absent source point.
    No sequence metric law or binary64 correspondence is asserted. *)

From Stdlib Require Import Arith Lia List.
From Liblevenshtein.TWED.Metric Require Import TwedSourceAlignments.
Import ListNotations.

Definition option_add (cost charge : option nat) : option nat :=
  match cost, charge with
  | Some first, Some second => Some (first + second)
  | _, _ => None
  end.

Definition option_min (first second : option nat) : option nat :=
  match first, second with
  | Some first_cost, Some second_cost =>
      Some (Nat.min first_cost second_cost)
  | Some cost, None | None, Some cost => Some cost
  | None, None => None
  end.

Definition option_min3 (first second third : option nat) : option nat :=
  option_min first (option_min second third).

Lemma option_add_some_inv : forall first second sum,
  option_add first second = Some sum ->
  exists left right,
    first = Some left /\ second = Some right /\ sum = left + right.
Proof.
  intros [left|] [right|] sum H; simpl in H; try discriminate.
  inversion H; subst. exists left, right. repeat split; reflexivity.
Qed.

Lemma option_min_attains_operand : forall first second minimum,
  option_min first second = Some minimum ->
  first = Some minimum \/ second = Some minimum.
Proof.
  intros [left|] [right|] minimum H; simpl in H; try discriminate.
  - destruct (le_dec left right) as [Hle | Hgt].
    + rewrite Nat.min_l in H by lia.
      inversion H; subst. left; reflexivity.
    + rewrite Nat.min_r in H by lia.
      inversion H; subst. right; reflexivity.
  - inversion H; subst. left; reflexivity.
  - inversion H; subst. right; reflexivity.
Qed.

Lemma option_min3_attains_operand : forall first second third minimum,
  option_min3 first second third = Some minimum ->
  first = Some minimum \/
  second = Some minimum \/ third = Some minimum.
Proof.
  intros first second third minimum H.
  unfold option_min3 in H.
  destruct (option_min_attains_operand _ _ _ H) as [Hfirst | Hrest].
  - now left.
  - apply option_min_attains_operand in Hrest.
    destruct Hrest as [Hsecond | Hthird]; [now right; left | now right; right].
Qed.

Lemma option_min_at_most_left : forall first second candidate,
  first = Some candidate ->
  exists minimum,
    option_min first second = Some minimum /\ minimum <= candidate.
Proof.
  intros first [other|] candidate Hfirst; subst first; simpl.
  - exists (Nat.min candidate other); split; [reflexivity | apply Nat.le_min_l].
  - exists candidate; split; [reflexivity | lia].
Qed.

Lemma option_min_at_most_right : forall first second candidate,
  second = Some candidate ->
  exists minimum,
    option_min first second = Some minimum /\ minimum <= candidate.
Proof.
  intros [other|] second candidate Hsecond; subst second; simpl.
  - exists (Nat.min other candidate); split; [reflexivity | apply Nat.le_min_r].
  - exists candidate; split; [reflexivity | lia].
Qed.

Lemma option_min3_at_most_left : forall first second third candidate,
  first = Some candidate ->
  exists minimum,
    option_min3 first second third = Some minimum /\
    minimum <= candidate.
Proof.
  intros first second third candidate Hfirst.
  unfold option_min3.
  now apply option_min_at_most_left.
Qed.

Lemma option_min3_at_most_middle : forall first second third candidate,
  second = Some candidate ->
  exists minimum,
    option_min3 first second third = Some minimum /\
    minimum <= candidate.
Proof.
  intros first second third candidate Hsecond.
  destruct (option_min_at_most_left second third candidate Hsecond)
    as [inner [Hinner Hbound]].
  destruct (option_min_at_most_right first (option_min second third)
    inner Hinner) as [outer [Houter Houter_bound]].
  exists outer; split; [exact Houter | lia].
Qed.

Lemma option_min3_at_most_right : forall first second third candidate,
  third = Some candidate ->
  exists minimum,
    option_min3 first second third = Some minimum /\
    minimum <= candidate.
Proof.
  intros first second third candidate Hthird.
  destruct (option_min_at_most_right second third candidate Hthird)
    as [inner [Hinner Hbound]].
  destruct (option_min_at_most_right first (option_min second third)
    inner Hinner) as [outer [Houter Houter_bound]].
  exists outer; split; [exact Houter | lia].
Qed.

Section SourceDP.
  Variables left right : nat -> option timed_sample.
  Variable gap_penalty : nat.

  Definition left_charge (index : nat) : option nat :=
    match left index, predecessor_at left index with
    | Some current, Some previous =>
        Some (delete_charge gap_penalty current previous)
    | _, _ => None
    end.

  Definition right_charge (index : nat) : option nat :=
    match right index, predecessor_at right index with
    | Some current, Some previous =>
        Some (delete_charge gap_penalty current previous)
    | _, _ => None
    end.

  Definition pair_charge (left_index right_index : nat) : option nat :=
    match left left_index, predecessor_at left left_index,
          right right_index, predecessor_at right right_index with
    | Some left_current, Some left_previous,
      Some right_current, Some right_previous =>
        Some (match_charge
          left_current left_previous right_current right_previous)
    | _, _, _, _ => None
    end.

  Lemma left_charge_some_inv : forall index charge,
    left_charge index = Some charge ->
    exists current previous,
      left index = Some current /\
      predecessor_at left index = Some previous /\
      charge = delete_charge gap_penalty current previous.
  Proof.
    intros index charge Hcharge.
    unfold left_charge in Hcharge.
    destruct (left index) as [current|] eqn:Hcurrent; try discriminate.
    destruct (predecessor_at left index) as [previous|] eqn:Hprevious;
      try discriminate.
    inversion Hcharge; subst.
    exists current, previous. repeat split; assumption || reflexivity.
  Qed.

  Lemma right_charge_some_inv : forall index charge,
    right_charge index = Some charge ->
    exists current previous,
      right index = Some current /\
      predecessor_at right index = Some previous /\
      charge = delete_charge gap_penalty current previous.
  Proof.
    intros index charge Hcharge.
    unfold right_charge in Hcharge.
    destruct (right index) as [current|] eqn:Hcurrent; try discriminate.
    destruct (predecessor_at right index) as [previous|] eqn:Hprevious;
      try discriminate.
    inversion Hcharge; subst.
    exists current, previous. repeat split; assumption || reflexivity.
  Qed.

  Lemma pair_charge_some_inv : forall i j charge,
    pair_charge i j = Some charge ->
    exists left_current left_previous right_current right_previous,
      left i = Some left_current /\
      predecessor_at left i = Some left_previous /\
      right j = Some right_current /\
      predecessor_at right j = Some right_previous /\
      charge = match_charge
        left_current left_previous right_current right_previous.
  Proof.
    intros i j charge Hcharge.
    unfold pair_charge in Hcharge.
    destruct (left i) as [left_current|] eqn:Hleft; try discriminate.
    destruct (predecessor_at left i) as [left_previous|]
      eqn:Hleft_previous; try discriminate.
    destruct (right j) as [right_current|] eqn:Hright; try discriminate.
    destruct (predecessor_at right j) as [right_previous|]
      eqn:Hright_previous; try discriminate.
    inversion Hcharge; subst.
    exists left_current, left_previous, right_current, right_previous.
    repeat split; assumption || reflexivity.
  Qed.

  (** The outer recursion is on rows. The inner recursion is on columns;
      [north] retains the previous row, and [row] is the current row.
      Thus no mutually nonstructural recursion is needed. *)
  Fixpoint source_dp (i : nat) : nat -> option nat :=
    match i with
    | 0 => fun j => match j with 0 => Some 0 | S _ => None end
    | S previous_i =>
        let north := source_dp previous_i in
        fix row (j : nat) : option nat :=
          match j with
          | 0 => None
          | S previous_j =>
              option_min3
                (option_add (north (S previous_j))
                  (left_charge previous_i))
                (option_add (row previous_j)
                  (right_charge previous_j))
                (option_add (north previous_j)
                  (pair_charge previous_i previous_j))
          end
    end.

  Lemma source_dp_origin : source_dp 0 0 = Some 0.
  Proof. reflexivity. Qed.

  Lemma source_dp_positive_axes : forall i j,
    source_dp (S i) 0 = None /\ source_dp 0 (S j) = None.
  Proof. intros; split; reflexivity. Qed.

  Lemma source_dp_interior : forall i j,
    source_dp (S i) (S j) =
      option_min3
        (option_add (source_dp i (S j)) (left_charge i))
        (option_add (source_dp (S i) j) (right_charge j))
        (option_add (source_dp i j) (pair_charge i j)).
  Proof. intros; reflexivity. Qed.

  Definition north_candidate (i j : nat) : option nat :=
    option_add (source_dp i (S j)) (left_charge i).

  Definition west_candidate (i j : nat) : option nat :=
    option_add (source_dp (S i) j) (right_charge j).

  Definition diagonal_candidate (i j : nat) : option nat :=
    option_add (source_dp i j) (pair_charge i j).

  Lemma source_dp_interior_candidates : forall i j,
    source_dp (S i) (S j) =
      option_min3
        (north_candidate i j)
        (west_candidate i j)
        (diagonal_candidate i j).
  Proof. intros; apply source_dp_interior. Qed.

  (** Every finite DP cell chooses an actual lawful predecessor branch. *)
  Theorem finite_dp_cell_has_attaining_path : forall i j best,
    source_dp i j = Some best ->
    lawful_path left right gap_penalty i j best.
  Proof.
    intro i; induction i as [|i IHi]; intros j.
    - destruct j as [|j]; intros best Hdp; simpl in Hdp.
      + inversion Hdp; subst. constructor.
      + discriminate.
    - induction j as [|j IHj]; intros best Hdp.
      + simpl in Hdp; discriminate.
      + rewrite source_dp_interior_candidates in Hdp.
        destruct (option_min3_attains_operand _ _ _ _ Hdp)
          as [Hnorth | [Hwest | Hdiagonal]].
        * unfold north_candidate in Hnorth.
          destruct (option_add_some_inv _ _ _ Hnorth)
            as [prior_cost [charge [Hprior [Hcharge Hsum]]]].
          destruct (left_charge_some_inv _ _ Hcharge)
            as [current [previous [Hcurrent [Hprevious Hcharge_eq]]]].
          subst best charge.
          eapply PathDeleteLeft.
          -- apply IHi. exact Hprior.
          -- exact Hcurrent.
          -- exact Hprevious.
        * unfold west_candidate in Hwest.
          destruct (option_add_some_inv _ _ _ Hwest)
            as [prior_cost [charge [Hprior [Hcharge Hsum]]]].
          destruct (right_charge_some_inv _ _ Hcharge)
            as [current [previous [Hcurrent [Hprevious Hcharge_eq]]]].
          subst best charge.
          eapply PathInsertRight.
          -- apply IHj. exact Hprior.
          -- exact Hcurrent.
          -- exact Hprevious.
        * unfold diagonal_candidate in Hdiagonal.
          destruct (option_add_some_inv _ _ _ Hdiagonal)
            as [prior_cost [charge [Hprior [Hcharge Hsum]]]].
          destruct (pair_charge_some_inv _ _ _ Hcharge)
            as [left_current [left_previous [right_current
              [right_previous [Hleft [Hleft_previous
                [Hright [Hright_previous Hcharge_eq]]]]]]]].
          subst best charge.
          eapply PathMatch.
          -- apply IHi. exact Hprior.
          -- exact Hleft.
          -- exact Hleft_previous.
          -- exact Hright.
          -- exact Hright_previous.
  Qed.

  (** Induction on the path shows that the DP result is no larger than
      any finite lawful alignment cost. *)
  Theorem source_dp_dominates_every_path : forall i j cost,
    lawful_path left right gap_penalty i j cost ->
    exists best, source_dp i j = Some best /\ best <= cost.
  Proof.
    intros i j cost Hpath.
    induction Hpath as
      [|i j cost current previous Hpath IH Hcurrent Hprevious
       |i j cost current previous Hpath IH Hcurrent Hprevious
       |i j cost left_current left_previous right_current right_previous
          Hpath IH Hleft Hleft_previous Hright Hright_previous].
    - exists 0. split; [reflexivity | lia].
    - destruct IH as [prior_best [Hprior Hbound]].
      assert (Hcharge : left_charge i =
        Some (delete_charge gap_penalty current previous)).
      { unfold left_charge. now rewrite Hcurrent, Hprevious. }
      assert (Hcandidate : north_candidate i j =
        Some (prior_best + delete_charge gap_penalty current previous)).
      { unfold north_candidate. now rewrite Hprior, Hcharge. }
      destruct (option_min3_at_most_left
        (north_candidate i j) (west_candidate i j)
        (diagonal_candidate i j) _ Hcandidate)
        as [best [Hminimum Hminimum_bound]].
      exists best; split.
      + rewrite source_dp_interior_candidates. exact Hminimum.
      + lia.
    - destruct IH as [prior_best [Hprior Hbound]].
      assert (Hcharge : right_charge j =
        Some (delete_charge gap_penalty current previous)).
      { unfold right_charge. now rewrite Hcurrent, Hprevious. }
      assert (Hcandidate : west_candidate i j =
        Some (prior_best + delete_charge gap_penalty current previous)).
      { unfold west_candidate. now rewrite Hprior, Hcharge. }
      destruct (option_min3_at_most_middle
        (north_candidate i j) (west_candidate i j)
        (diagonal_candidate i j) _ Hcandidate)
        as [best [Hminimum Hminimum_bound]].
      exists best; split.
      + rewrite source_dp_interior_candidates. exact Hminimum.
      + lia.
    - destruct IH as [prior_best [Hprior Hbound]].
      assert (Hcharge : pair_charge i j =
        Some (match_charge
          left_current left_previous right_current right_previous)).
      { unfold pair_charge.
        now rewrite Hleft, Hleft_previous, Hright, Hright_previous. }
      assert (Hcandidate : diagonal_candidate i j =
        Some (prior_best + match_charge
          left_current left_previous right_current right_previous)).
      { unfold diagonal_candidate. now rewrite Hprior, Hcharge. }
      destruct (option_min3_at_most_right
        (north_candidate i j) (west_candidate i j)
        (diagonal_candidate i j) _ Hcandidate)
        as [best [Hminimum Hminimum_bound]].
      exists best; split.
      + rewrite source_dp_interior_candidates. exact Hminimum.
      + lia.
  Qed.

  Theorem finite_dp_path_equivalence : forall i j,
    match source_dp i j with
    | None => forall cost,
        ~ lawful_path left right gap_penalty i j cost
    | Some best =>
        lawful_path left right gap_penalty i j best /\
        forall cost,
          lawful_path left right gap_penalty i j cost -> best <= cost
    end.
  Proof.
    intros i j.
    destruct (source_dp i j) as [best|] eqn:Hdp.
    - split.
      + apply finite_dp_cell_has_attaining_path. exact Hdp.
      + intros cost Hpath.
        destruct (source_dp_dominates_every_path _ _ _ Hpath)
          as [actual [Hactual Hbound]].
        rewrite Hdp in Hactual. inversion Hactual; subst. exact Hbound.
    - intros cost Hpath.
      destruct (source_dp_dominates_every_path _ _ _ Hpath)
        as [actual [Hactual _]].
      rewrite Hdp in Hactual. discriminate.
  Qed.

  Corollary finite_source_minimum_is_attained : forall i j,
    (exists cost, lawful_path left right gap_penalty i j cost) ->
    exists best,
      source_dp i j = Some best /\
      lawful_path left right gap_penalty i j best /\
      forall cost,
        lawful_path left right gap_penalty i j cost -> best <= cost.
  Proof.
    intros i j [cost Hpath].
    destruct (source_dp_dominates_every_path _ _ _ Hpath)
      as [best [Hdp _]].
    exists best. split; [exact Hdp |].
    pose proof (finite_dp_path_equivalence i j) as Hcertificate.
    rewrite Hdp in Hcertificate.
    exact Hcertificate.
  Qed.
End SourceDP.

(** Relaxing the common-anchor boundary changes the answer. Here the
    left source is just the anchor and the right source is the anchor
    followed by (0,2). At gap penalty three, the lawful DP must first
    match anchors and then insert the later point at cost 1+3=4. An
    altered, gap-free positive axis lets the path first consume the right
    anchor at cost 1 and then match the left anchor to the later point at
    current-point cost 1 and predecessor-point cost 1, for a total of 3. *)
Definition short_left := list_source [common_anchor].
Definition short_right := list_source [common_anchor; (0, 2)].

Definition altered_d00 : nat := 0.
Definition altered_d01 : nat := point_distance common_anchor
  source_zero_predecessor.
Definition altered_d02 : nat := altered_d01 +
  point_distance (0, 2) common_anchor.
Definition altered_d10 : nat := altered_d01.
Definition altered_d11 : nat := Nat.min
  (altered_d01 + delete_charge 3 common_anchor source_zero_predecessor)
  (Nat.min
    (altered_d10 + delete_charge 3 common_anchor source_zero_predecessor)
    (altered_d00 + match_charge common_anchor
      source_zero_predecessor common_anchor source_zero_predecessor)).
Definition altered_d12 : nat := Nat.min
  (altered_d02 + delete_charge 3 common_anchor source_zero_predecessor)
  (Nat.min
    (altered_d11 + delete_charge 3 (0, 2) common_anchor)
    (altered_d01 + match_charge common_anchor
      source_zero_predecessor (0, 2) common_anchor)).

Theorem relaxed_anchor_boundary_changes_score :
  altered_d12 = 3 /\
  source_dp short_left short_right 3 1 2 = Some 4.
Proof. split; reflexivity. Qed.
