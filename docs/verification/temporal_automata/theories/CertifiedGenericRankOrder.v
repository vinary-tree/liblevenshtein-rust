(** * Generic lexicographic rank order and lawful comparison

    Costs and tie keys use the quotient-aware order interface from the rank
    certificate module. Rank equality means equivalence in both coordinates;
    it need not be Leibniz equality of representations. A comparator is
    admitted only together with a proof connecting every comparison result
    to that order. Min/max retain an actual input and satisfy their order
    laws independently of which equivalent representation is retained.

    Injective tie keys identify originals only within the explicitly supplied
    rank function and domain. A source instance must bind those functions,
    its comparison code, numeric authority, and captured snapshot. The
    unordered-value control models the failure of ordinary NaN comparisons;
    it is not an IEEE bit-level model. An explicitly chosen total ordering of
    NaN representations would be a different semantic contract. *)

From Stdlib Require Import Arith Lia.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedKnnHeap CertifiedRankCertificateTypes.
Set Implicit Arguments.

Section RankLaws.
  Context {Cost Tie : Type}.
  Variable cost_order : certificate_order Cost.
  Variable tie_order : certificate_order Tie.

  Definition generic_rank_equiv (left right : Cost * Tie) : Prop :=
    order_equiv cost_order (fst left) (fst right) /\
    order_equiv tie_order (snd left) (snd right).

  Definition generic_rank_strict (left right : Cost * Tie) : Prop :=
    generic_rank_le cost_order tie_order left right /\
    ~ generic_rank_equiv left right.

  Lemma rank_equiv_reflexive : forall rank, generic_rank_equiv rank rank.
  Proof. intro rank; split; apply order_equiv_refl. Qed.

  Lemma rank_equiv_symmetric : forall left right,
    generic_rank_equiv left right -> generic_rank_equiv right left.
  Proof.
    intros left right [Hcost Htie]; split; now apply order_equiv_sym.
  Qed.

  Lemma rank_equiv_transitive : forall first middle last,
    generic_rank_equiv first middle -> generic_rank_equiv middle last ->
    generic_rank_equiv first last.
  Proof.
    intros first middle last [Hcost Htie] [Hnext_cost Hnext_tie]; split;
      eapply order_equiv_trans; eauto.
  Qed.

  Lemma rank_equiv_implies_le : forall left right,
    generic_rank_equiv left right ->
    generic_rank_le cost_order tie_order left right.
  Proof.
    intros left right [Hcost Htie]; constructor 2; split; [exact Hcost |].
    now apply certificate_equiv_implies_le.
  Qed.

  Lemma rank_le_implies_cost_le : forall left right,
    generic_rank_le cost_order tie_order left right ->
    order_le cost_order (fst left) (fst right).
  Proof.
    intros left right [Hcost | [Hcost Htie]].
    - now apply certificate_lt_implies_le.
    - now apply certificate_equiv_implies_le.
  Qed.

  Theorem generic_rank_le_total : forall left right,
    generic_rank_le cost_order tie_order left right \/
    generic_rank_le cost_order tie_order right left.
  Proof.
    intros left right.
    destruct (order_le_total cost_order (fst left) (fst right)) as [Hle | Hle].
    - destruct (proj1 (order_le_cases cost_order _ _) Hle) as [Hlt | Heq].
      + constructor 1; now constructor 1.
      + destruct (order_le_total tie_order (snd left) (snd right)) as [Ht | Ht].
        * constructor 1; constructor 2; now split.
        * constructor 2; constructor 2; split; [now apply order_equiv_sym | exact Ht].
    - destruct (proj1 (order_le_cases cost_order _ _) Hle) as [Hlt | Heq].
      + constructor 2; now constructor 1.
      + destruct (order_le_total tie_order (snd left) (snd right)) as [Ht | Ht].
        * constructor 1; constructor 2; split; [now apply order_equiv_sym | exact Ht].
        * constructor 2; constructor 2; now split.
  Qed.

  Theorem rank_le_antisymmetric_up_to_equiv : forall left right,
    generic_rank_le cost_order tie_order left right ->
    generic_rank_le cost_order tie_order right left ->
    generic_rank_equiv left right.
  Proof.
    intros left right Hforward Hbackward.
    pose proof (rank_le_implies_cost_le Hforward) as Hcost_forward.
    pose proof (rank_le_implies_cost_le Hbackward) as Hcost_backward.
    assert (Hnot_forward : ~ order_lt cost_order (fst left) (fst right)).
    { intro Hlt.
      apply (@certificate_lt_irreflexive Cost cost_order (fst left)).
      eapply certificate_lt_le_trans; eauto. }
    assert (Hnot_backward : ~ order_lt cost_order (fst right) (fst left)).
    { intro Hlt.
      apply (@certificate_lt_irreflexive Cost cost_order (fst right)).
      eapply certificate_lt_le_trans; eauto. }
    destruct Hforward as [Hlt | [Hcost Htie]]; [contradiction |].
    destruct Hbackward as [Hlt | [Hcost_back Htie_back]]; [contradiction |].
    split; [exact Hcost | eapply order_le_antisym; eauto].
  Qed.

  Theorem rank_strict_is_lexicographic : forall left right,
    generic_rank_strict left right <->
    order_lt cost_order (fst left) (fst right) \/
    (order_equiv cost_order (fst left) (fst right) /\
     order_lt tie_order (snd left) (snd right)).
  Proof.
    intros left right; split.
    - intros [[Hcost | [Hcost Htie]] Hneq].
      + now constructor 1.
      + constructor 2; split; [exact Hcost |].
        apply (proj2 (order_lt_cases tie_order _ _)); split; [exact Htie |].
        intro Htie_eq; apply Hneq; now split.
    - intros [Hcost | [Hcost Htie]].
      + split; [now constructor 1 |].
        intros [Heq Htie].
        exact (proj2 (proj1 (order_lt_cases cost_order _ _) Hcost) Heq).
      + split.
        * constructor 2; split; [exact Hcost | now apply certificate_lt_implies_le].
        * intros [Heq Htie_eq].
          exact (proj2 (proj1 (order_lt_cases tie_order _ _) Htie) Htie_eq).
  Qed.

  Theorem rank_le_is_strict_or_equal : forall left right,
    generic_rank_le cost_order tie_order left right <->
    generic_rank_strict left right \/ generic_rank_equiv left right.
  Proof.
    intros left right; split.
    - intros [Hcost | [Hcost Htie]].
      + constructor 1; apply rank_strict_is_lexicographic; now constructor 1.
      + destruct (proj1 (order_le_cases tie_order _ _) Htie) as [Hlt | Heq].
        * constructor 1; apply rank_strict_is_lexicographic; constructor 2; now split.
        * constructor 2; now split.
    - intros [[Hle Hneq] | Heq]; [exact Hle | now apply rank_equiv_implies_le].
  Qed.

  (** Package the derived rank laws for reuse by every ordered-certificate
      consumer. The imported generic proof provides weak transitivity. *)
  Definition generic_rank_order : certificate_order (Cost * Tie).
  Proof.
    refine {| order_equiv := generic_rank_equiv;
              order_le := generic_rank_le cost_order tie_order;
              order_lt := generic_rank_strict |}.
    - exact rank_equiv_reflexive.
    - exact rank_equiv_symmetric.
    - exact rank_equiv_transitive.
    - intro rank; apply generic_rank_le_reflexive.
    - intros; eapply generic_rank_le_transitive; eauto.
    - exact generic_rank_le_total.
    - exact rank_le_antisymmetric_up_to_equiv.
    - exact rank_le_is_strict_or_equal.
    - intros; reflexivity.
  Defined.

  Theorem rank_strict_irreflexive : forall rank, ~ generic_rank_strict rank rank.
  Proof.
    intros rank [Hle Hneq]; apply Hneq, rank_equiv_reflexive.
  Qed.

  Theorem rank_strict_asymmetric : forall left right,
    generic_rank_strict left right -> ~ generic_rank_strict right left.
  Proof.
    intros left right [Hle Hneq] [Hback Hneq_back].
    apply Hneq; now apply rank_le_antisymmetric_up_to_equiv.
  Qed.

  Theorem rank_strict_transitive : forall first middle last,
    generic_rank_strict first middle -> generic_rank_strict middle last ->
    generic_rank_strict first last.
  Proof.
    intros first middle last Hfirst Hlast.
    eapply (@certificate_le_lt_trans (Cost * Tie) generic_rank_order).
    - exact (proj1 Hfirst).
    - exact Hlast.
  Qed.

  Theorem rank_trichotomy : forall left right,
    generic_rank_strict left right \/ generic_rank_equiv left right \/
    generic_rank_strict right left.
  Proof.
    intros left right; destruct (generic_rank_le_total left right) as [Hle | Hle].
    - apply rank_le_is_strict_or_equal in Hle as [Hlt | Heq]; auto.
    - apply rank_le_is_strict_or_equal in Hle as [Hlt | Heq].
      + auto.
      + constructor 2; constructor 1; now apply rank_equiv_symmetric.
  Qed.

  Definition rank_comparison_sound
      (compare : Cost * Tie -> Cost * Tie -> comparison) : Prop :=
    forall left right,
      match compare left right with
      | Eq => generic_rank_equiv left right
      | Lt => generic_rank_strict left right
      | Gt => generic_rank_strict right left
      end.

  Record lawful_rank_comparator := {
    compare_rank : Cost * Tie -> Cost * Tie -> comparison;
    compare_rank_sound : rank_comparison_sound compare_rank
  }.

  Theorem comparison_equal_iff_rank_equiv : forall comparator left right,
    compare_rank comparator left right = Eq <-> generic_rank_equiv left right.
  Proof.
    intros comparator left right.
    pose proof (compare_rank_sound comparator left right) as Hsound.
    destruct (compare_rank comparator left right); simpl in Hsound.
    - split; [intro; exact Hsound | intro; reflexivity].
    - split; [discriminate |].
      intro Heq; exfalso; exact (proj2 Hsound Heq).
    - split; [discriminate |].
      intro Heq; exfalso; apply (proj2 Hsound).
      now apply rank_equiv_symmetric.
  Qed.

  Theorem comparison_less_iff_rank_strict : forall comparator left right,
    compare_rank comparator left right = Lt <-> generic_rank_strict left right.
  Proof.
    intros comparator left right.
    pose proof (compare_rank_sound comparator left right) as Hsound.
    destruct (compare_rank comparator left right); simpl in Hsound.
    - split; [discriminate |].
      intro Hlt; exfalso; exact (proj2 Hlt Hsound).
    - split; [intro; exact Hsound | intro; reflexivity].
    - split; [discriminate |].
      intro Hlt; exfalso; eapply rank_strict_asymmetric; eauto.
  Qed.

  Corollary lawful_comparison_reflexive : forall comparator rank,
    compare_rank comparator rank rank = Eq.
  Proof. intros; apply comparison_equal_iff_rank_equiv, rank_equiv_reflexive. Qed.

  Definition rank_minimum (comparator : lawful_rank_comparator)
      (left right : Cost * Tie) : Cost * Tie :=
    match compare_rank comparator left right with Gt => right | _ => left end.

  Definition rank_maximum (comparator : lawful_rank_comparator)
      (left right : Cost * Tie) : Cost * Tie :=
    match compare_rank comparator left right with Lt => right | _ => left end.

  Theorem rank_minimum_selects_an_input : forall comparator left right,
    rank_minimum comparator left right = left \/
    rank_minimum comparator left right = right.
  Proof.
    intros; unfold rank_minimum; destruct (compare_rank comparator left right); auto.
  Qed.

  Theorem rank_maximum_selects_an_input : forall comparator left right,
    rank_maximum comparator left right = left \/
    rank_maximum comparator left right = right.
  Proof.
    intros; unfold rank_maximum; destruct (compare_rank comparator left right); auto.
  Qed.

  Theorem rank_minimum_below_both : forall comparator left right,
    generic_rank_le cost_order tie_order (rank_minimum comparator left right) left /\
    generic_rank_le cost_order tie_order (rank_minimum comparator left right) right.
  Proof.
    intros comparator left right.
    pose proof (compare_rank_sound comparator left right) as Hsound.
    unfold rank_minimum; destruct (compare_rank comparator left right);
      simpl in Hsound.
    - split; [apply generic_rank_le_reflexive | now apply rank_equiv_implies_le].
    - split; [apply generic_rank_le_reflexive | exact (proj1 Hsound)].
    - split; [exact (proj1 Hsound) | apply generic_rank_le_reflexive].
  Qed.

  Theorem rank_maximum_above_both : forall comparator left right,
    generic_rank_le cost_order tie_order left (rank_maximum comparator left right) /\
    generic_rank_le cost_order tie_order right (rank_maximum comparator left right).
  Proof.
    intros comparator left right.
    pose proof (compare_rank_sound comparator left right) as Hsound.
    unfold rank_maximum; destruct (compare_rank comparator left right);
      simpl in Hsound.
    - split; [apply generic_rank_le_reflexive |].
      apply rank_equiv_implies_le; now apply rank_equiv_symmetric.
    - split; [exact (proj1 Hsound) | apply generic_rank_le_reflexive].
    - split; [apply generic_rank_le_reflexive | exact (proj1 Hsound)].
  Qed.

  Theorem rank_minimum_is_greatest_lower_bound : forall comparator left right lower,
    generic_rank_le cost_order tie_order lower left ->
    generic_rank_le cost_order tie_order lower right ->
    generic_rank_le cost_order tie_order lower (rank_minimum comparator left right).
  Proof.
    intros comparator left right lower Hleft Hright.
    unfold rank_minimum; destruct (compare_rank comparator left right); assumption.
  Qed.

  Theorem rank_maximum_is_least_upper_bound : forall comparator left right upper,
    generic_rank_le cost_order tie_order left upper ->
    generic_rank_le cost_order tie_order right upper ->
    generic_rank_le cost_order tie_order (rank_maximum comparator left right) upper.
  Proof.
    intros comparator left right upper Hleft Hright.
    unfold rank_maximum; destruct (compare_rank comparator left right); assumption.
  Qed.

  Theorem rank_minimum_commutative_up_to_equiv : forall comparator left right,
    generic_rank_equiv (rank_minimum comparator left right)
      (rank_minimum comparator right left).
  Proof.
    intros comparator left right.
    destruct (rank_minimum_below_both comparator left right) as [Hll Hlr].
    destruct (rank_minimum_below_both comparator right left) as [Hrr Hrl].
    apply rank_le_antisymmetric_up_to_equiv;
      apply rank_minimum_is_greatest_lower_bound; assumption.
  Qed.

  Theorem rank_maximum_commutative_up_to_equiv : forall comparator left right,
    generic_rank_equiv (rank_maximum comparator left right)
      (rank_maximum comparator right left).
  Proof.
    intros comparator left right.
    destruct (rank_maximum_above_both comparator left right) as [Hll Hlr].
    destruct (rank_maximum_above_both comparator right left) as [Hrr Hrl].
    apply rank_le_antisymmetric_up_to_equiv;
      apply rank_maximum_is_least_upper_bound; assumption.
  Qed.

  Theorem rank_minimum_idempotent : forall comparator rank,
    rank_minimum comparator rank rank = rank.
  Proof.
    intros; unfold rank_minimum; now rewrite lawful_comparison_reflexive.
  Qed.

  Theorem rank_maximum_idempotent : forall comparator rank,
    rank_maximum comparator rank rank = rank.
  Proof.
    intros; unfold rank_maximum; now rewrite lawful_comparison_reflexive.
  Qed.

  Section OriginalIdentity.
    Context {Original : Type}.
    Variable rank_of : Original -> Cost * Tie.
    Variable admitted : Original -> Prop.
    Context (tie_identifies_original : forall left right,
      admitted left -> admitted right ->
      order_equiv tie_order (snd (rank_of left)) (snd (rank_of right)) ->
      left = right).

    Theorem equivalent_admitted_ranks_identify_original : forall left right,
      admitted left -> admitted right ->
      generic_rank_equiv (rank_of left) (rank_of right) -> left = right.
    Proof.
      intros left right Hleft Hright [_ Htie].
      eapply tie_identifies_original; eauto.
    Qed.

    Theorem mutually_ordered_admitted_ranks_identify_original : forall left right,
      admitted left -> admitted right ->
      generic_rank_le cost_order tie_order (rank_of left) (rank_of right) ->
      generic_rank_le cost_order tie_order (rank_of right) (rank_of left) ->
      left = right.
    Proof.
      intros left right Hleft Hright Hforward Hback.
      eapply equivalent_admitted_ranks_identify_original; eauto.
      now apply rank_le_antisymmetric_up_to_equiv.
    Qed.

    Theorem equal_comparison_identifies_admitted_original :
      forall comparator left right,
        admitted left -> admitted right ->
        compare_rank comparator (rank_of left) (rank_of right) = Eq -> left = right.
    Proof.
      intros comparator left right Hleft Hright Hequal.
      eapply equivalent_admitted_ranks_identify_original; eauto.
      now apply comparison_equal_iff_rank_equiv in Hequal.
    Qed.

    Theorem distinct_admitted_originals_have_strict_ranks : forall left right,
      admitted left -> admitted right -> left <> right ->
      generic_rank_strict (rank_of left) (rank_of right) \/
      generic_rank_strict (rank_of right) (rank_of left).
    Proof.
      intros left right Hleft Hright Hdifferent.
      destruct (rank_trichotomy (rank_of left) (rank_of right))
        as [Hless | [Hequal | Hgreater]]; auto.
      exfalso; apply Hdifferent.
      eapply equivalent_admitted_ranks_identify_original; eauto.
    Qed.
  End OriginalIdentity.
End RankLaws.

Module NaturalRankInstance.
  Definition compare_natural_rank (left right : nat * nat) : comparison :=
    match Nat.compare (fst left) (fst right) with
    | Eq => Nat.compare (snd left) (snd right)
    | Lt => Lt
    | Gt => Gt
    end.

  Theorem natural_comparison_is_lawful :
    rank_comparison_sound natural_certificate_order natural_certificate_order
      compare_natural_rank.
  Proof.
    intros [left_cost left_tie] [right_cost right_tie].
    unfold compare_natural_rank; simpl.
    destruct (Nat.compare left_cost right_cost) eqn:Hcost.
    - apply Nat.compare_eq_iff in Hcost.
      destruct (Nat.compare left_tie right_tie) eqn:Htie.
      + apply Nat.compare_eq_iff in Htie.
        unfold generic_rank_equiv; simpl; now split.
      + apply Nat.compare_lt_iff in Htie.
        unfold generic_rank_strict, generic_rank_equiv, generic_rank_le;
          simpl; lia.
      + apply Nat.compare_gt_iff in Htie.
        unfold generic_rank_strict, generic_rank_equiv, generic_rank_le;
          simpl; lia.
    - apply Nat.compare_lt_iff in Hcost.
      unfold generic_rank_strict, generic_rank_equiv, generic_rank_le;
        simpl; lia.
    - apply Nat.compare_gt_iff in Hcost.
      unfold generic_rank_strict, generic_rank_equiv, generic_rank_le;
        simpl; lia.
  Qed.

  Definition natural_rank_comparator :
      lawful_rank_comparator natural_certificate_order natural_certificate_order :=
    {| compare_rank := compare_natural_rank;
       compare_rank_sound := natural_comparison_is_lawful |}.

  Theorem natural_strict_order_agrees_with_heap : forall left right,
    generic_rank_strict natural_certificate_order natural_certificate_order
      (entry_cost left, entry_tie left) (entry_cost right, entry_tie right) <->
    rank_lt left right.
  Proof.
    intros left right.
    unfold generic_rank_strict, generic_rank_equiv, generic_rank_le, rank_lt;
      simpl; lia.
  Qed.

  Theorem natural_id_ties_make_equal_comparisons_identify_originals :
    forall (score_of : nat -> nat) left right,
      compare_rank natural_rank_comparator
        (score_of left, left) (score_of right, right) = Eq -> left = right.
  Proof.
    intros score_of left right Hequal.
    apply comparison_equal_iff_rank_equiv in Hequal.
    change (score_of left = score_of right /\ left = right) in Hequal.
    tauto.
  Qed.

  Example equal_cost_comparison_uses_tie_key :
    compare_rank natural_rank_comparator (5, 1) (5, 9) = Lt /\
    rank_minimum natural_rank_comparator (6, 1) (5, 9) = (5, 9) /\
    rank_maximum natural_rank_comparator (6, 1) (5, 9) = (6, 1).
  Proof. repeat split; reflexivity. Qed.

  Example always_equal_comparator_is_not_lawful :
    ~ rank_comparison_sound natural_certificate_order natural_certificate_order
      (fun _ _ => Eq).
  Proof.
    intro Hbad; specialize (Hbad (0, 0) (1, 0)).
    change (0 = 1 /\ 0 = 0) in Hbad; lia.
  Qed.

  Definition colliding_rank_of (_ : nat) : nat * nat := (5, 0).

  Example lawful_comparison_alone_does_not_identify_originals :
    1 <> 2 /\
    compare_rank natural_rank_comparator
      (colliding_rank_of 1) (colliding_rank_of 2) = Eq.
  Proof. split; [discriminate | reflexivity]. Qed.
End NaturalRankInstance.

(** An unordered token has exactly the relevant NaN obstruction: even its
    comparison with itself is not <=. It cannot implement this order interface. *)
Definition unordered_numeric_le (left right : option nat) : Prop :=
  match left, right with
  | Some x, Some y => x <= y
  | _, _ => False
  end.

Theorem unordered_nan_semantics_cannot_supply_a_lawful_order :
  ~ exists ordering : certificate_order (option nat),
      forall left right,
        order_le ordering left right <-> unordered_numeric_le left right.
Proof.
  intros [ordering Hcorrespondence].
  pose proof (order_le_refl ordering None) as Hreflexive.
  apply (proj1 (Hcorrespondence None None)) in Hreflexive.
  exact Hreflexive.
Qed.
