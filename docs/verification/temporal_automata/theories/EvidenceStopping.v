(** * Feasible evidence completions and information-relative stopping

    The universe is an explicit finite list of eligible original identities.
    Identities are naturals and are also the injective tie keys. Costs use a
    supplied lawful order, possibly with quotient equality. Evidence consists
    of exact verified observations and closed cost intervals; repeated exact
    observations are permitted and must agree semantically.

    A completion is a total cost assignment, but evidence and results concern
    only the captured finite universe. No metric, recurrence, snapshot, or
    machine-arithmetic relation is inferred. Membership is decidable because
    identities are naturals. No decidability of arbitrary cost propositions,
    endpoint-attainability axiom, or search over all completions is assumed.

    The constructive consistency theorem selects a recorded exact value or
    an interval's lower endpoint under explicit coherence and bound premises.
    Safe stopping itself includes existence of a feasible completion, so an
    inconsistent evidence set can never authorize a vacuous result. BF-4 is
    proved for a full verified best-k selection; an underfull selection needs
    exhaustion of eligible unresolved identities. These are semantic stopping
    theorems, not an executable evidence solver or a Rust correspondence. *)

From Stdlib Require Import Arith Lia List Sorting.Sorted.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder.
Import ListNotations.
Set Implicit Arguments.

Section EvidenceModel.
  Context {Cost : Type}.
  Variable cost_order : certificate_order Cost.

  Definition completion := nat -> Cost.

  Definition original_rank (assignment : completion) (original : nat)
      : Cost * nat := (assignment original, original).

  Definition original_le (assignment : completion) (lhs rhs : nat) : Prop :=
    generic_rank_le cost_order natural_certificate_order
      (original_rank assignment lhs) (original_rank assignment rhs).

  Definition original_lt (assignment : completion) (lhs rhs : nat) : Prop :=
    generic_rank_strict cost_order natural_certificate_order
      (original_rank assignment lhs) (original_rank assignment rhs).

  Lemma original_le_reflexive : forall assignment original,
    original_le assignment original original.
  Proof. intros; apply generic_rank_le_reflexive. Qed.

  Lemma original_le_transitive : forall assignment first middle last,
    original_le assignment first middle ->
    original_le assignment middle last -> original_le assignment first last.
  Proof. intros; eapply generic_rank_le_transitive; eauto. Qed.

  Lemma not_earlier_implies_reverse_order : forall assignment lhs rhs,
    ~ original_lt assignment lhs rhs -> original_le assignment rhs lhs.
  Proof.
    intros assignment lhs rhs Hnot.
    destruct (@generic_rank_le_total Cost nat cost_order
      natural_certificate_order
      (original_rank assignment lhs) (original_rank assignment rhs))
      as [Hforward | Hreverse]; [|exact Hreverse].
    apply rank_le_is_strict_or_equal in Hforward as [Hstrict | Hequal].
    - exfalso; apply Hnot; exact Hstrict.
    - apply rank_equiv_implies_le, rank_equiv_symmetric; exact Hequal.
  Qed.

  Lemma strict_order_conflicts_with_reverse : forall assignment lhs rhs,
    original_lt assignment lhs rhs -> ~ original_le assignment rhs lhs.
  Proof.
    intros assignment lhs rhs [Hforward Hneq] Hreverse.
    apply Hneq; eapply rank_le_antisymmetric_up_to_equiv; eauto.
  Qed.

  Record stopping_evidence := {
    eligible_originals : list nat;
    verified_observations : list (nat * Cost);
    lower_endpoint : nat -> Cost;
    upper_endpoint : nat -> Cost
  }.

  Definition verified_originals (evidence : stopping_evidence) : list nat :=
    map fst (verified_observations evidence).

  Definition evidence_well_formed (evidence : stopping_evidence) : Prop :=
    NoDup (eligible_originals evidence) /\
    incl (verified_originals evidence) (eligible_originals evidence).

  (** [feasible_completion evidence] is the set M(I). Source identity
      membership is checked separately from semantic score consistency. *)
  Definition feasible_completion (evidence : stopping_evidence)
      (assignment : completion) : Prop :=
    evidence_well_formed evidence /\
    (forall original score,
      In (original, score) (verified_observations evidence) ->
      order_equiv cost_order (assignment original) score) /\
    (forall original, In original (eligible_originals evidence) ->
      order_le cost_order (lower_endpoint evidence original)
        (assignment original) /\
      order_le cost_order (assignment original)
        (upper_endpoint evidence original)).

  Definition evidence_consistent (evidence : stopping_evidence) : Prop :=
    exists assignment, feasible_completion evidence assignment.

  Fixpoint lookup_verified (original : nat) (observations : list (nat * Cost))
      : option Cost :=
    match observations with
    | [] => None
    | (identity, score) :: rest =>
        if Nat.eq_dec original identity then Some score
        else lookup_verified original rest
    end.

  Definition verified_scores_coherent (evidence : stopping_evidence) : Prop :=
    forall original first second,
      In (original, first) (verified_observations evidence) ->
      In (original, second) (verified_observations evidence) ->
      order_equiv cost_order first second.

  Lemma lookup_verified_returns_observation : forall observations original score,
    lookup_verified original observations = Some score ->
    In (original, score) observations.
  Proof.
    intro observations; induction observations as [|[identity value] rest IH];
      intros original score Hlookup; simpl in Hlookup; [discriminate |].
    destruct (Nat.eq_dec original identity) as [Hequal | Hdifferent].
    - inversion Hlookup; subst; simpl; auto.
    - simpl; constructor 2; now apply IH.
  Qed.

  Lemma observed_identity_has_lookup : forall observations original score,
    In (original, score) observations ->
    exists actual, lookup_verified original observations = Some actual.
  Proof.
    intro observations; induction observations as [|[identity value] rest IH];
      intros original score Hin; simpl in Hin; [contradiction |].
    simpl; destruct (Nat.eq_dec original identity) as [Hequal | Hdifferent].
    - exists value; reflexivity.
    - destruct Hin as [Hequal | Hin].
      + exfalso; apply Hdifferent.
        exact (eq_sym (f_equal (@fst nat Cost) Hequal)).
      + eapply IH; exact Hin.
  Qed.

  Definition endpoint_completion (evidence : stopping_evidence) : completion :=
    fun original =>
      match lookup_verified original (verified_observations evidence) with
      | Some score => score
      | None => lower_endpoint evidence original
      end.

  Theorem coherent_bounded_evidence_has_a_completion : forall evidence,
    evidence_well_formed evidence ->
    verified_scores_coherent evidence ->
    (forall original, In original (eligible_originals evidence) ->
      order_le cost_order (lower_endpoint evidence original)
        (upper_endpoint evidence original)) ->
    (forall original score,
      In (original, score) (verified_observations evidence) ->
      order_le cost_order (lower_endpoint evidence original) score /\
      order_le cost_order score (upper_endpoint evidence original)) ->
    feasible_completion evidence (endpoint_completion evidence).
  Proof.
    intros evidence Hformed Hcoherent Hintervals Hobserved_bounds.
    split; [exact Hformed |]; split.
    - intros original score Hin.
      destruct (@observed_identity_has_lookup
        (verified_observations evidence) original score Hin)
        as [actual Hlookup].
      unfold endpoint_completion; rewrite Hlookup.
      apply (Hcoherent original actual score); [|exact Hin].
      eapply lookup_verified_returns_observation; exact Hlookup.
    - intros original Hin; unfold endpoint_completion.
      destruct (lookup_verified original (verified_observations evidence))
        as [score|] eqn:Hlookup.
      + apply Hobserved_bounds.
        eapply lookup_verified_returns_observation; exact Hlookup.
      + split; [apply order_le_refl | now apply Hintervals].
  Qed.

  Corollary coherent_bounded_evidence_is_consistent : forall evidence,
    evidence_well_formed evidence ->
    verified_scores_coherent evidence ->
    (forall original, In original (eligible_originals evidence) ->
      order_le cost_order (lower_endpoint evidence original)
        (upper_endpoint evidence original)) ->
    (forall original score,
      In (original, score) (verified_observations evidence) ->
      order_le cost_order (lower_endpoint evidence original) score /\
      order_le cost_order score (upper_endpoint evidence original)) ->
    evidence_consistent evidence.
  Proof.
    intros evidence Hformed Hcoherent Hintervals Hobserved.
    exists (endpoint_completion evidence).
    now apply coherent_bounded_evidence_has_a_completion.
  Qed.

  Theorem feasible_completions_agree_on_verified_ranks :
    forall evidence first second original,
      feasible_completion evidence first ->
      feasible_completion evidence second ->
      In original (verified_originals evidence) ->
      generic_rank_equiv cost_order natural_certificate_order
        (original_rank first original) (original_rank second original).
  Proof.
    intros evidence first second original [_ [Hfirst _]] [_ [Hsecond _]] Hin.
    unfold verified_originals in Hin.
    apply in_map_iff in Hin as [[identity score] [Hidentity Hobserved]].
    simpl in Hidentity; subst identity.
    split; [|apply order_equiv_refl].
    eapply order_equiv_trans.
    - apply Hfirst; exact Hobserved.
    - apply order_equiv_sym, Hsecond; exact Hobserved.
  Qed.

  (** This definition gives a completed ordered best-k independently of any
      stopping test. Every omitted eligible original requires a full selection
      and follows each selected original. Its size theorem below recovers
      min(k, population) for a duplicate-free finite universe. *)
  Record best_k_over (universe : list nat) (assignment : completion)
      (capacity : nat) (selected : list nat) : Prop := {
    chosen_distinct : NoDup selected;
    chosen_eligible : incl selected universe;
    chosen_ordered : StronglySorted (original_le assignment) selected;
    chosen_capacity : length selected <= capacity;
    omitted_follows_full_selection : forall original,
      In original universe -> ~ In original selected ->
      length selected = capacity /\
      forall chosen, In chosen selected -> original_le assignment chosen original
  }.

  Definition full_best_k universe assignment capacity selected : Prop :=
    best_k_over universe assignment capacity selected /\
    length selected = capacity /\ 0 < capacity.

  Definition underfull_best_k universe assignment capacity selected : Prop :=
    best_k_over universe assignment capacity selected /\
    length selected < capacity.

  Theorem best_k_has_required_cardinality :
    forall universe assignment capacity selected,
      NoDup universe -> best_k_over universe assignment capacity selected ->
      length selected = Nat.min capacity (length universe).
  Proof.
    intros universe assignment capacity selected Hdistinct
      [Hselected_distinct Hincluded Hordered Hcapacity Homitted].
    pose proof (@NoDup_incl_length nat selected universe
      Hselected_distinct Hincluded) as Hsize.
    destruct (Nat.eq_dec (length selected) capacity) as [Hfull | Hunder].
    - rewrite <- Hfull; symmetry; apply Nat.min_l; exact Hsize.
    - assert (Hreverse : incl universe selected).
      { intros original Hin.
        destruct (in_dec Nat.eq_dec original selected) as [Hchosen | Hnot];
          [exact Hchosen |].
        destruct (Homitted original Hin Hnot) as [Hfull _]; contradiction. }
      pose proof (@NoDup_incl_length nat universe selected
        Hdistinct Hreverse) as Hreverse_size.
      assert (Hequal : length selected = length universe) by lia.
      rewrite Hequal; symmetry; apply Nat.min_r; lia.
  Qed.

  Lemma zero_capacity_selects_nothing : forall universe assignment,
    best_k_over universe assignment 0 [].
  Proof.
    intros; constructor.
    - constructor.
    - intros original Hin; contradiction.
    - constructor.
    - simpl; lia.
    - intros original Hin Hnot; split; [reflexivity |].
      intros chosen Hchosen; contradiction.
  Qed.

  (** The local best-k premise concerns only verified originals. Requiring it
      for every feasible completion does not assume the unknown answer: all
      such completions agree on the verified ranks by the theorem above.
      A concrete verified-heap producer must discharge this premise. *)
  Definition full_verified_choice (evidence : stopping_evidence)
      (capacity : nat) (selected : list nat) (worst : nat) : Prop :=
    length selected = capacity /\ 0 < capacity /\ In worst selected /\
    forall assignment, feasible_completion evidence assignment ->
      best_k_over (verified_originals evidence) assignment capacity selected /\
      forall chosen, In chosen selected -> original_le assignment chosen worst.

  Definition underfull_verified_choice (evidence : stopping_evidence)
      (capacity : nat) (selected : list nat) : Prop :=
    length selected < capacity /\
    forall assignment, feasible_completion evidence assignment ->
      best_k_over (verified_originals evidence) assignment capacity selected.

  Definition unresolved_influence (evidence : stopping_evidence)
      (assignment : completion) (worst : nat) : Prop :=
    exists original,
      In original (eligible_originals evidence) /\
      ~ In original (verified_originals evidence) /\
      original_lt assignment original worst.

  Definition no_feasible_unresolved_influence (evidence : stopping_evidence)
      (worst : nat) : Prop :=
    forall assignment, feasible_completion evidence assignment ->
      ~ unresolved_influence evidence assignment worst.

  Definition universally_safe_to_stop (evidence : stopping_evidence)
      (capacity : nat) (selected : list nat) : Prop :=
    evidence_consistent evidence /\
    forall assignment, feasible_completion evidence assignment ->
      best_k_over (eligible_originals evidence) assignment capacity selected.

  Theorem bf4_full_safety_characterization :
    forall evidence capacity selected worst,
      evidence_consistent evidence ->
      full_verified_choice evidence capacity selected worst ->
      (universally_safe_to_stop evidence capacity selected <->
       no_feasible_unresolved_influence evidence worst).
  Proof.
    intros evidence capacity selected worst Hconsistent
      [Hfull [Hpositive [Hworst Hverified]]]; split.
    - intros [_ Hsafe] assignment Hmodel
        [original [Heligible [Hunverified Hearlier]]].
      destruct (Hsafe assignment Hmodel)
        as [_ _ _ _ Homitted].
      destruct (Hverified assignment Hmodel)
        as [[_ Hselected_verified _ _ _] Hmaximum].
      assert (Hnotselected : ~ In original selected).
      { intro Hselected; apply Hunverified, Hselected_verified; exact Hselected. }
      destruct (Homitted original Heligible Hnotselected) as [_ Hafter].
      eapply strict_order_conflicts_with_reverse; [exact Hearlier |].
      apply Hafter; exact Hworst.
    - intros Hno; split; [exact Hconsistent |].
      intros assignment Hmodel.
      destruct (Hverified assignment Hmodel)
        as [[Hdistinct Hselected_verified Hordered Hcapacity Hverified_after]
            Hmaximum].
      destruct Hmodel as [[Huniverse Hverified_eligible] [Hscores Hbounds]].
      assert (Hmodel : feasible_completion evidence assignment).
      { split; [now split | now split]. }
      constructor.
      + exact Hdistinct.
      + intros original Hin; apply Hverified_eligible, Hselected_verified; exact Hin.
      + exact Hordered.
      + exact Hcapacity.
      + intros original Heligible Hnotselected; split; [exact Hfull |].
        intros chosen Hchosen.
        destruct (in_dec Nat.eq_dec original (verified_originals evidence))
          as [Hknown | Hunknown].
        * destruct (Hverified_after original Hknown Hnotselected) as [_ Hafter].
          apply Hafter; exact Hchosen.
        * eapply original_le_transitive; [apply Hmaximum; exact Hchosen |].
          apply not_earlier_implies_reverse_order.
          intro Hearlier; apply (Hno assignment Hmodel).
          exists original; split; [exact Heligible |].
          split; [exact Hunknown | exact Hearlier].
  Qed.

  Definition eligible_unresolved_exhausted (evidence : stopping_evidence) : Prop :=
    incl (eligible_originals evidence) (verified_originals evidence).

  Theorem underfull_safety_requires_exhaustion :
    forall evidence capacity selected,
      evidence_consistent evidence ->
      underfull_verified_choice evidence capacity selected ->
      (universally_safe_to_stop evidence capacity selected <->
       eligible_unresolved_exhausted evidence).
  Proof.
    intros evidence capacity selected Hconsistent [Hunder Hverified]; split.
    - intros [_ Hsafe].
      destruct Hconsistent as [assignment Hmodel].
      destruct (Hsafe assignment Hmodel) as [_ _ _ _ Homitted].
      destruct (Hverified assignment Hmodel) as [_ Hselected_verified _ _ _].
      intros original Heligible.
      destruct (in_dec Nat.eq_dec original selected) as [Hchosen | Hnot].
      + apply Hselected_verified; exact Hchosen.
      + destruct (Homitted original Heligible Hnot) as [Hfull _]; lia.
    - intros Hexhausted; split; [exact Hconsistent |].
      intros assignment Hmodel.
      destruct (Hverified assignment Hmodel)
        as [Hdistinct Hselected_verified Hordered Hcapacity Homitted].
      destruct Hmodel as [[Huniverse Hverified_eligible] _].
      constructor.
      + exact Hdistinct.
      + intros original Hin; apply Hverified_eligible, Hselected_verified; exact Hin.
      + exact Hordered.
      + exact Hcapacity.
      + intros original Heligible Hnotselected.
        apply Homitted; [apply Hexhausted; exact Heligible | exact Hnotselected].
  Qed.

  Corollary inconsistent_evidence_cannot_authorize_stopping :
    forall evidence capacity selected,
      ~ evidence_consistent evidence ->
      ~ universally_safe_to_stop evidence capacity selected.
  Proof. intros evidence capacity selected Hbad [Hconsistent _]; contradiction. Qed.

  Lemma singleton_verified_choice : forall evidence original,
    verified_originals evidence = [original] ->
    full_verified_choice evidence 1 [original] original.
  Proof.
    intros evidence original Hverified.
    split; [reflexivity |]; split; [lia |]; split; [simpl; auto |].
    intros assignment Hmodel; split.
    - rewrite Hverified; constructor.
      + constructor; [simpl; tauto | constructor].
      + intros candidate Hin; exact Hin.
      + constructor; constructor.
      + simpl; lia.
      + intros candidate Hin Hnot; contradiction.
    - intros chosen [Hequal | Hfalse]; [subst chosen | contradiction].
      apply original_le_reflexive.
  Qed.
End EvidenceModel.

Module NaturalEvidenceControls.
  Definition contradictory_scores : @stopping_evidence nat :=
    {| eligible_originals := [0];
       verified_observations := [(0, 1); (0, 2)];
       lower_endpoint := fun _ => 0;
       upper_endpoint := fun _ => 10 |}.

  Example contradictory_score_evidence_is_well_formed :
    evidence_well_formed contradictory_scores.
  Proof.
    split.
    - constructor; [simpl; tauto | constructor].
    - intros original Hin; simpl in *; tauto.
  Qed.

  Example contradictory_scores_have_no_completion :
    ~ evidence_consistent natural_certificate_order contradictory_scores.
  Proof.
    intros [assignment [_ [Hscores _]]].
    pose proof (Hscores 0 1 (or_introl eq_refl)) as Hone.
    pose proof (Hscores 0 2 (or_intror (or_introl eq_refl))) as Htwo.
    change (assignment 0 = 1) in Hone.
    change (assignment 0 = 2) in Htwo; lia.
  Qed.

  Definition impossible_interval : @stopping_evidence nat :=
    {| eligible_originals := [0]; verified_observations := [];
       lower_endpoint := fun _ => 2; upper_endpoint := fun _ => 1 |}.

  Example impossible_interval_evidence_is_well_formed :
    evidence_well_formed impossible_interval.
  Proof.
    split.
    - constructor; [simpl; tauto | constructor].
    - intros original Hin; simpl in Hin; contradiction.
  Qed.

  Example impossible_interval_has_no_completion :
    ~ evidence_consistent natural_certificate_order impossible_interval.
  Proof.
    intros [assignment [_ [_ Hbounds]]].
    specialize (Hbounds 0 (or_introl eq_refl)).
    change (2 <= assignment 0 /\ assignment 0 <= 1) in Hbounds; lia.
  Qed.

  Example contradictory_scores_cannot_authorize_any_result :
    forall capacity selected,
      ~ universally_safe_to_stop natural_certificate_order
        contradictory_scores capacity selected.
  Proof.
    intros; apply inconsistent_evidence_cannot_authorize_stopping.
    exact contradictory_scores_have_no_completion.
  Qed.

  Definition separated_evidence : @stopping_evidence nat :=
    {| eligible_originals := [0; 1]; verified_observations := [(0, 5)];
       lower_endpoint := fun original => if Nat.eq_dec original 0 then 5 else 8;
       upper_endpoint := fun _ => 10 |}.

  Definition separated_completion (original : nat) : nat :=
    if Nat.eq_dec original 0 then 5 else 8.

  Example separated_evidence_has_a_feasible_completion :
    feasible_completion natural_certificate_order
      separated_evidence separated_completion.
  Proof.
    split.
    - split.
      + constructor; [simpl; lia |].
        constructor; [simpl; tauto | constructor].
      + intros original Hin; simpl in *; tauto.
    - split.
      + intros original score Hin; simpl in Hin.
        destruct Hin as [Hequal | Hfalse]; [inversion Hequal; subst | contradiction].
        reflexivity.
      + intros original Hin; simpl in Hin.
        destruct Hin as [Hequal | [Hequal | Hfalse]].
        * subst original; change (5 <= 5 /\ 5 <= 10); lia.
        * subst original; change (8 <= 8 /\ 8 <= 10); lia.
        * contradiction.
  Qed.

  Example separated_evidence_has_no_unresolved_influence :
    no_feasible_unresolved_influence natural_certificate_order separated_evidence 0.
  Proof.
    intros assignment [_ [Hscores Hbounds]]
      [original [Heligible [Hunverified Hearlier]]].
    simpl in Heligible.
    destruct Heligible as [Hequal | [Hequal | Hfalse]];
      [subst original | subst original | contradiction].
    - apply Hunverified; simpl; auto.
    - pose proof (Hscores 0 5 (or_introl eq_refl)) as Hzero.
      pose proof (Hbounds 1 (or_intror (or_introl eq_refl))) as Hone.
      change (assignment 0 = 5) in Hzero.
      change (8 <= assignment 1 /\ assignment 1 <= 10) in Hone.
      change ((assignment 1 < assignment 0 \/
        (assignment 1 = assignment 0 /\ 1 <= 0)) /\
        ~ (assignment 1 = assignment 0 /\ 1 = 0)) in Hearlier.
      lia.
  Qed.

  Example inhabited_full_selection_is_safe :
    universally_safe_to_stop natural_certificate_order separated_evidence 1 [0].
  Proof.
    assert (Hconsistent :
      evidence_consistent natural_certificate_order separated_evidence).
    { exists separated_completion; exact separated_evidence_has_a_feasible_completion. }
    assert (Hchoice : full_verified_choice natural_certificate_order
      separated_evidence 1 [0] 0).
    { apply singleton_verified_choice; reflexivity. }
    apply (proj2 (@bf4_full_safety_characterization nat natural_certificate_order
      separated_evidence 1 [0] 0 Hconsistent Hchoice)).
    exact separated_evidence_has_no_unresolved_influence.
  Qed.

  (** The very same consistent evidence has no rank beating original zero,
      but capacity two still has room for the expensive unresolved original. *)
  Example inhabited_underfull_stopping_is_rejected :
    ~ universally_safe_to_stop natural_certificate_order separated_evidence 2 [0].
  Proof.
    intros [_ Hsafe].
    destruct (Hsafe separated_completion
      separated_evidence_has_a_feasible_completion) as [_ _ _ _ Homitted].
    assert (Hin : In 1 (eligible_originals separated_evidence)) by (simpl; auto).
    assert (Hnot : ~ In 1 [0]) by (simpl; lia).
    destruct (Homitted 1 Hin Hnot) as [Hfull _]; simpl in Hfull; lia.
  Qed.

  Definition permissive_evidence : @stopping_evidence nat :=
    {| eligible_originals := [0; 1]; verified_observations := [(0, 5)];
       lower_endpoint := fun _ => 0; upper_endpoint := fun _ => 10 |}.

  Definition improving_completion (original : nat) : nat :=
    if Nat.eq_dec original 0 then 5 else 4.

  Example improving_assignment_is_feasible :
    feasible_completion natural_certificate_order
      permissive_evidence improving_completion.
  Proof.
    split.
    - split.
      + constructor; [simpl; lia |].
        constructor; [simpl; tauto | constructor].
      + intros original Hin; simpl in *; tauto.
    - split.
      + intros original score Hin; simpl in Hin.
        destruct Hin as [Hequal | Hfalse]; [inversion Hequal; subst | contradiction].
        reflexivity.
      + intros original Hin; simpl in Hin.
        destruct Hin as [Hequal | [Hequal | Hfalse]].
        * subst original; change (0 <= 5 /\ 5 <= 10); lia.
        * subst original; change (0 <= 4 /\ 4 <= 10); lia.
        * contradiction.
  Qed.

  Example inhabited_full_selection_can_be_unsafe :
    full_verified_choice natural_certificate_order permissive_evidence 1 [0] 0 /\
    evidence_consistent natural_certificate_order permissive_evidence /\
    unresolved_influence natural_certificate_order
      permissive_evidence improving_completion 0 /\
    ~ universally_safe_to_stop natural_certificate_order permissive_evidence 1 [0].
  Proof.
    split; [apply singleton_verified_choice; reflexivity |].
    split; [exists improving_completion; exact improving_assignment_is_feasible |].
    split.
    - exists 1; split; [simpl; auto |]; split; [simpl; lia |].
      change ((4 < 5 \/ (4 = 5 /\ 1 <= 0)) /\ ~ (4 = 5 /\ 1 = 0)).
      lia.
    - intros [_ Hsafe].
      destruct (Hsafe improving_completion improving_assignment_is_feasible)
        as [_ _ _ _ Homitted].
      assert (Hin : In 1 (eligible_originals permissive_evidence)) by (simpl; auto).
      assert (Hnot : ~ In 1 [0]) by (simpl; lia).
      destruct (Homitted 1 Hin Hnot) as [_ Hafter].
      specialize (Hafter 0 (or_introl eq_refl)).
      change (5 < 4 \/ (5 = 4 /\ 0 <= 1)) in Hafter; lia.
  Qed.
End NaturalEvidenceControls.
