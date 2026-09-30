(** * Certified execution and optimization kernels

    Parametric finite-trace simulation, a decreasing rank for administrative
    steps, cost-conditioned tie bounds, potential accounting, and finite
    portfolio selection. These are implications from explicit local premises,
    not certificates for the Rust implementation or its binary64 arithmetic.
    No axiom or admitted proof is introduced. *)

From Stdlib Require Import Arith Bool Lia List.
Import ListNotations.

Section Traces.
  Context {Event : Type}.

  Inductive execution {State : Type}
      (step : State -> list Event -> State -> Prop)
      : State -> list Event -> State -> Prop :=
  | execution_refl : forall state, execution step state [] state
  | execution_cons : forall first events middle later last,
      step first events middle -> execution step middle later last ->
      execution step first (events ++ later) last.

  Lemma execution_append : forall State step (first middle last : State) left right,
    execution step first left middle -> execution step middle right last ->
    execution step first (left ++ right) last.
  Proof.
    intros State step first middle last left right Hleft Hright.
    induction Hleft; simpl; [exact Hright |].
    rewrite <- app_assoc. econstructor; eauto.
  Qed.

  Definition step_refinement {Concrete Abstract : Type}
      (concrete_step : Concrete -> list Event -> Concrete -> Prop)
      (abstract_step : Abstract -> list Event -> Abstract -> Prop)
      (related : Concrete -> Abstract -> Prop) : Prop :=
    forall concrete events next abstract,
      related concrete abstract -> concrete_step concrete events next ->
      exists following,
        execution abstract_step abstract events following /\ related next following.

  Theorem local_certificates_lift_to_traces : forall Concrete Abstract
      concrete_step abstract_step (related : Concrete -> Abstract -> Prop),
    step_refinement concrete_step abstract_step related ->
    forall first events last,
      execution concrete_step first events last ->
      forall abstract, related first abstract ->
      exists following,
        execution abstract_step abstract events following /\ related last following.
  Proof.
    intros Concrete Abstract concrete_step abstract_step related Hlocal
      first events last Hrun.
    induction Hrun; intros abstract Hrelated.
    - exists abstract; split; [constructor | exact Hrelated].
    - destruct (Hlocal _ _ _ _ Hrelated H) as [middle' [Hstep Hmiddle]].
      destruct (IHHrun _ Hmiddle) as [last' [Htail Hlast]].
      exists last'; split; [eapply execution_append; eauto | exact Hlast].
  Qed.

  Theorem step_certificates_compose : forall First Middle Last
      first_step middle_step last_step
      (left : First -> Middle -> Prop) (right : Middle -> Last -> Prop),
    step_refinement first_step middle_step left ->
    step_refinement middle_step last_step right ->
    step_refinement first_step last_step
      (fun first last => exists middle, left first middle /\ right middle last).
  Proof.
    intros First Middle Last first_step middle_step last_step left right
      Hleft Hright first events next last [middle [Hl Hr]] Hstep.
    destruct (Hleft _ _ _ _ Hl Hstep) as [next_middle [Hmiddle Hnext]].
    destruct (local_certificates_lift_to_traces _ _ _ _ _ Hright
      _ _ _ Hmiddle _ Hr) as [next_last [Hlast Hnext_last]].
    exists next_last; split; [exact Hlast |].
    exists next_middle; now split.
  Qed.
End Traces.

(** Finite trace preservation by itself allows an infinite silent loop.
    A natural-valued rank gives an explicit bound on consecutive internal steps. *)
Section AdministrativeProgress.
  Context {State : Type}.
  Variable internal_step : State -> State -> Prop.
  Variable rank : State -> nat.

  Inductive internal_steps : nat -> State -> State -> Prop :=
  | internal_zero : forall state, internal_steps 0 state state
  | internal_next : forall count first middle last,
      internal_step first middle -> internal_steps count middle last ->
      internal_steps (S count) first last.

  Theorem decreasing_rank_bounds_internal_steps :
    (forall first last, internal_step first last -> rank last < rank first) ->
    forall count first last, internal_steps count first last ->
      count + rank last <= rank first.
  Proof.
    intros Hdecrease count first last Hsteps.
    induction Hsteps; [lia |].
    specialize (Hdecrease _ _ H); lia.
  Qed.
End AdministrativeProgress.

(** BF-2: the secondary bound is needed only on the equality-cost slice. *)
Definition ranked_le (left right : nat * nat) : Prop :=
  fst left < fst right \/
  (fst left = fst right /\ snd left <= snd right).

Definition ranked_lt (left right : nat * nat) : Prop :=
  fst left < fst right \/
  (fst left = fst right /\ snd left < snd right).

Theorem equality_slice_characterizes_lex_lower_bound : forall lower floor cost tie,
  lower <= cost ->
  (ranked_le (lower, floor) (cost, tie) <-> (lower = cost -> floor <= tie)).
Proof.
  intros; unfold ranked_le; simpl. lia.
Qed.

Record ranked_candidate := {
  candidate_cost : nat;
  candidate_tie : nat;
  candidate_lower : nat
}.

Definition candidate_bound_sound (candidate : ranked_candidate) : Prop :=
  candidate_lower candidate <= candidate_cost candidate.

Definition eligible_at (threshold : nat) (candidates : list ranked_candidate)
    : list ranked_candidate :=
  filter (fun candidate => candidate_lower candidate <=? threshold) candidates.

Theorem eligible_set_covers_exact_threshold : forall candidates threshold candidate,
  Forall candidate_bound_sound candidates -> In candidate candidates ->
  candidate_cost candidate <= threshold ->
  In candidate (eligible_at threshold candidates).
Proof.
  intros candidates threshold candidate Hbounds Hin Hcost.
  apply Forall_forall with (x := candidate) in Hbounds; [|exact Hin].
  unfold eligible_at; apply filter_In; split; [exact Hin |].
  apply Nat.leb_le; unfold candidate_bound_sound in Hbounds; lia.
Qed.

Definition floor_on (floor : nat) (candidates : list ranked_candidate) : Prop :=
  forall candidate, In candidate candidates -> floor <= candidate_tie candidate.

Theorem threshold_floor_supplies_lex_lower_bound : forall candidates lower floor,
  Forall candidate_bound_sound candidates ->
  (forall candidate, In candidate candidates -> lower <= candidate_cost candidate) ->
  floor_on floor (eligible_at lower candidates) ->
  forall candidate, In candidate candidates ->
    ranked_le (lower, floor) (candidate_cost candidate, candidate_tie candidate).
Proof.
  intros candidates lower floor Hbounds Hlower Hfloor candidate Hin.
  apply equality_slice_characterizes_lex_lower_bound; [now apply Hlower |].
  intro Heq. apply Hfloor.
  eapply eligible_set_covers_exact_threshold; eauto; lia.
Qed.

Theorem threshold_floor_pruning_is_sound : forall candidates lower floor worst_tie,
  Forall candidate_bound_sound candidates ->
  (forall candidate, In candidate candidates -> lower <= candidate_cost candidate) ->
  floor_on floor (eligible_at lower candidates) -> worst_tie <= floor ->
  forall candidate, In candidate candidates ->
    ~ ranked_lt (candidate_cost candidate, candidate_tie candidate) (lower, worst_tie).
Proof.
  intros candidates lower floor worst_tie Hbounds Hlower Hfloor Hworst candidate Hin.
  pose proof (threshold_floor_supplies_lex_lower_bound _ _ _
    Hbounds Hlower Hfloor _ Hin) as Hrank.
  unfold ranked_le in Hrank; unfold ranked_lt; simpl in *; lia.
Qed.

Theorem eligible_sets_shrink_with_threshold : forall candidates low high candidate,
  low <= high -> In candidate (eligible_at low candidates) ->
  In candidate (eligible_at high candidates).
Proof.
  intros candidates low high candidate Hle Hin.
  unfold eligible_at in *; apply filter_In in Hin as [Hin Hlow].
  apply filter_In; split; [exact Hin |].
  apply Nat.leb_le; apply Nat.leb_le in Hlow; lia.
Qed.

Example equality_slice_can_improve_a_global_floor :
  let expensive := {| candidate_cost := 6; candidate_tie := 1; candidate_lower := 6 |} in
  let equal := {| candidate_cost := 5; candidate_tie := 9; candidate_lower := 5 |} in
  eligible_at 5 [expensive; equal] = [equal] /\
  ~ ranked_le (5, 7) (5, 1) /\ ranked_le (5, 7) (5, 9).
Proof. simpl; unfold ranked_le; simpl; repeat split; lia. Qed.

Example a_threshold_floor_cannot_be_reused_at_a_larger_threshold :
  let expensive := {| candidate_cost := 6; candidate_tie := 1; candidate_lower := 6 |} in
  let equal := {| candidate_cost := 5; candidate_tie := 9; candidate_lower := 5 |} in
  floor_on 9 (eligible_at 5 [expensive; equal]) /\
  ~ floor_on 9 (eligible_at 6 [expensive; equal]).
Proof.
  simpl; unfold floor_on; simpl; split.
  - intros candidate [Heq | []]; subst; simpl; lia.
  - intro Hfloor; specialize (Hfloor _ (or_introl eq_refl)); simpl in Hfloor; lia.
Qed.

(** Parent bounds remain valid for every child subset, allowing a max to
    monotonize priorities without modifying the score recurrence. *)
Theorem inherited_maximum_bound_is_sound : forall parent local exact,
  parent <= exact -> local <= exact ->
  Nat.max parent local <= exact /\ parent <= Nat.max parent local.
Proof. intros; lia. Qed.

(** Equality-conditioned floors cannot be combined coordinatewise: a floor
    proved only at cost five says nothing about ties at cost six. Combining
    complete lexicographic certificates by their lexicographic maximum is safe. *)
Definition ranked_maximum (left right : nat * nat) : nat * nat :=
  if (fst left <? fst right) ||
     ((fst left =? fst right) && (snd left <=? snd right))
  then right else left.

Theorem maximum_of_rank_certificates_is_sound : forall left right exact,
  ranked_le left exact -> ranked_le right exact ->
  ranked_le (ranked_maximum left right) exact.
Proof.
  intros left right exact Hleft Hright.
  unfold ranked_maximum; destruct ((_ <? _) || ((_ =? _) && (_ <=? _)));
    assumption.
Qed.

Theorem ranked_maximum_strengthens_both_inputs : forall left right,
  ranked_le left (ranked_maximum left right) /\
  ranked_le right (ranked_maximum left right).
Proof.
  intros [lc ltie] [rc rtie]; unfold ranked_maximum, ranked_le; simpl.
  destruct ((lc <? rc) || ((lc =? rc) && (ltie <=? rtie))) eqn:Hchoose.
  - apply orb_true_iff in Hchoose as [Hlt | Heq].
    + apply Nat.ltb_lt in Hlt; simpl; lia.
    + apply andb_true_iff in Heq as [Heq Htie].
      apply Nat.eqb_eq in Heq; apply Nat.leb_le in Htie; simpl; lia.
  - apply orb_false_iff in Hchoose as [Hlt Hrest].
    apply Nat.ltb_ge in Hlt.
    apply andb_false_iff in Hrest as [Hneq | Htie].
    + apply Nat.eqb_neq in Hneq; simpl; lia.
    + apply Nat.leb_gt in Htie; simpl; lia.
Qed.

Example coordinatewise_maximum_of_conditioned_floors_is_unsound :
  ranked_le (5, 9) (6, 1) /\ ranked_le (6, 1) (6, 1) /\
  ~ ranked_le (Nat.max 5 6, Nat.max 9 1) (6, 1).
Proof. unfold ranked_le; simpl; lia. Qed.

(** Quantitative refinement: charges and released credits telescope across
    arbitrary executions. Peak live memory requires a different invariant. *)
Section PotentialAccounting.
  Context {State : Type}.
  Variable charged_step : State -> nat -> nat -> State -> Prop.
  Variable potential : State -> nat.

  Inductive charged_execution : State -> nat -> nat -> State -> Prop :=
  | charged_zero : forall state, charged_execution state 0 0 state
  | charged_next : forall first work credit middle later_work later_credit last,
      charged_step first work credit middle ->
      charged_execution middle later_work later_credit last ->
      charged_execution first (work + later_work) (credit + later_credit) last.

  Theorem local_potential_bounds_total_work :
    (forall first work credit last, charged_step first work credit last ->
       work + potential last <= credit + potential first) ->
    forall first work credit last, charged_execution first work credit last ->
      work + potential last <= credit + potential first.
  Proof.
    intros Hlocal first work credit last Hrun.
    induction Hrun; [lia |].
    specialize (Hlocal _ _ _ _ H); lia.
  Qed.
End PotentialAccounting.

(** Finite portfolio optimality is relative to an exact declared cost model.
    It proves no claim about unmeasured execution time or absent candidates. *)
Section Portfolio.
  Context {Artifact : Type}.
  Variable objective : Artifact -> nat.

  Fixpoint select_minimum (baseline : Artifact) (choices : list Artifact) : Artifact :=
    match choices with
    | [] => baseline
    | candidate :: tail =>
        let selected := select_minimum baseline tail in
        if objective selected <=? objective candidate then selected else candidate
    end.

  Theorem selected_minimum_is_available : forall baseline choices,
    In (select_minimum baseline choices) (baseline :: choices).
  Proof.
    intros baseline choices; induction choices as [| candidate tail IH]; simpl.
    - now left.
    - destruct (objective (select_minimum baseline tail) <=? objective candidate).
      + destruct IH as [Hbase | Hin]; [now left | right; now right].
      + right; now left.
  Qed.

  Theorem selected_minimum_is_model_optimal : forall baseline choices,
    objective (select_minimum baseline choices) <= objective baseline /\
    (forall candidate, In candidate choices ->
      objective (select_minimum baseline choices) <= objective candidate).
  Proof.
    intros baseline choices; induction choices as [| candidate tail [Hbase Htail]];
      simpl; [split; [lia | intros ? []] |].
    destruct (objective (select_minimum baseline tail) <=? objective candidate)
      eqn:Hchoose.
    - apply Nat.leb_le in Hchoose. split; [exact Hbase |].
      intros other [Heq | Hin]; [subst; exact Hchoose | now apply Htail].
    - apply Nat.leb_gt in Hchoose. split; [lia |].
      intros other [Heq | Hin]; [subst; lia | specialize (Htail _ Hin); lia].
  Qed.

  Theorem portfolio_selection_preserves_certification : forall
      (certified : Artifact -> Prop) baseline choices,
    certified baseline -> Forall certified choices ->
    certified (select_minimum baseline choices).
  Proof.
    intros certified baseline choices Hbase Hchoices.
    pose proof (selected_minimum_is_available baseline choices) as Hin.
    destruct Hin as [Heq | Hin]; [now rewrite <- Heq |].
    now apply (proj1 (Forall_forall certified choices) Hchoices).
  Qed.
End Portfolio.
