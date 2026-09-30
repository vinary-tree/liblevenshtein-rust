(** * Scoped certificates for the finite exact-natural expression fragment

    This checker consumes the expression language and proved rewrite rules in
    CertifiedContracts. It does not accept an arbitrary ORC realization node,
    machine program, rounded-arithmetic rule, or witness transformation. A
    future extension must add a rule with its own soundness proof. *)

From Stdlib Require Import Arith Bool Lia List.
Require Import CertifiedContracts.
Import ListNotations.

Inductive rewrite_direction := Forward | Backward.
Inductive witness_effect := NoWitnessEffect | PreservesWitness | ChangesWitness.
Inductive relation_kind := Equivalent | LowerThanSource.

(** An unknown wire identifier has a representation, but no replay case. *)
Inductive rule_identity :=
| NamedExact : exact_rule -> rule_identity
| NamedLower : lower_rule -> rule_identity
| UnknownRule : nat -> rule_identity.

Inductive premise_name :=
| ExactNaturalArithmetic
| ScoreOnlyObservation
| ExactPattern
| NaturalNonnegativity
| OmittedSummandNonnegativity.

Definition premise_name_eq_dec (lhs rhs : premise_name) :
    {lhs = rhs} + {lhs <> rhs}.
Proof. decide equality. Defined.

Record realization_scope := {
  base_scope : contract_scope;
  scoped_revision : nat;
  scoped_label_context : nat
}.

Definition realization_scope_eq_dec (lhs rhs : realization_scope) :
    {lhs = rhs} + {lhs <> rhs}.
Proof.
  decide equality; try apply Nat.eq_dec; apply scope_eq_dec.
Defined.

(** The caller supplies exactly the premises named by the rule. The replay
    algorithm checks the actual pattern and carrier; the names alone are not
    treated as evidence. *)
Definition required_premises (rule : rule_identity) : option (list premise_name) :=
  match rule with
  | NamedExact _ =>
      Some [ExactNaturalArithmetic; ScoreOnlyObservation; ExactPattern]
  | NamedLower ZeroBound =>
      Some [ExactNaturalArithmetic; ScoreOnlyObservation;
            NaturalNonnegativity]
  | NamedLower _ =>
      Some [ExactNaturalArithmetic; ScoreOnlyObservation;
            OmittedSummandNonnegativity]
  | UnknownRule _ => None
  end.

Definition advance_kind (old effect : relation_kind) : relation_kind :=
  match old, effect with
  | Equivalent, Equivalent => Equivalent
  | _, _ => LowerThanSource
  end.

Definition rule_result (rule : rule_identity) (direction : rewrite_direction)
    (source target : cost_expression) : option relation_kind :=
  match rule, direction with
  | NamedExact named, Forward =>
      match exact_rewrite named source with
      | Some expected =>
          if expression_eq_dec expected target then Some Equivalent else None
      | None => None
      end
  | NamedExact named, Backward =>
      match exact_rewrite named target with
      | Some expected =>
          if expression_eq_dec expected source then Some Equivalent else None
      | None => None
      end
  | NamedLower named, Forward =>
      match construct_lower named source with
      | Some expected =>
          if expression_eq_dec expected target then Some LowerThanSource else None
      | None => None
      end
  | NamedLower _, Backward | UnknownRule _, _ => None
  end.

Lemma rule_result_sound : forall rule direction source target effect environment,
  rule_result rule direction source target = Some effect ->
  match effect with
  | Equivalent => evaluate environment target = evaluate environment source
  | LowerThanSource => evaluate environment target <= evaluate environment source
  end.
Proof.
  intros rule direction source target effect environment Hresult.
  destruct rule as [named | named | unknown]; destruct direction;
    simpl in Hresult; try discriminate.
  - destruct (exact_rewrite named source) as [expected |] eqn:Hrewrite;
      [|discriminate].
    destruct (expression_eq_dec expected target) as [Heq | Hneq];
      [|discriminate].
    inversion Hresult; subst.
    symmetry; eapply exact_rewrite_sound; eauto.
  - destruct (exact_rewrite named target) as [expected |] eqn:Hrewrite;
      [|discriminate].
    destruct (expression_eq_dec expected source) as [Heq | Hneq];
      [|discriminate].
    inversion Hresult; subst.
    eapply exact_rewrite_sound; eauto.
  - destruct (construct_lower named source) as [expected |] eqn:Hrewrite;
      [|discriminate].
    destruct (expression_eq_dec expected target) as [Heq | Hneq];
      [|discriminate].
    inversion Hresult; subst.
    eapply construct_lower_sound; eauto.
Qed.

Record scoped_step := {
  recorded_rule : rule_identity;
  recorded_premises : list premise_name;
  recorded_direction : rewrite_direction;
  recorded_source : cost_expression;
  recorded_target : cost_expression;
  recorded_scope : realization_scope;
  recorded_arithmetic : arithmetic_profile;
  recorded_witness_effect : witness_effect
}.

Definition replay_step (requested : realization_scope)
    (current : cost_expression) (old_kind : relation_kind)
    (step : scoped_step) : option (cost_expression * relation_kind) :=
  if realization_scope_eq_dec requested (recorded_scope step) then
  if arithmetic_profile_eq_dec (recorded_arithmetic step) ExactNaturals then
  if arithmetic_profile_eq_dec (scope_arithmetic (base_scope requested))
       ExactNaturals then
  if observation_profile_eq_dec
       (scope_observation (base_scope requested)) ScoreOnly then
  match recorded_witness_effect step with
  | NoWitnessEffect =>
      if expression_eq_dec current (recorded_source step) then
      match required_premises (recorded_rule step) with
      | Some required =>
          if list_eq_dec premise_name_eq_dec
               required (recorded_premises step) then
          match rule_result (recorded_rule step)
              (recorded_direction step) (recorded_source step)
              (recorded_target step) with
          | Some effect =>
              Some (recorded_target step, advance_kind old_kind effect)
          | None => None
          end else None
      | None => None
      end else None
  | PreservesWitness | ChangesWitness => None
  end else None else None else None else None.

Definition named_rule (rule : rule_identity) : Prop :=
  match rule with UnknownRule _ => False | _ => True end.

Theorem replay_step_requires_exact_scope :
  forall requested current old_kind step next new_kind,
    replay_step requested current old_kind step = Some (next, new_kind) ->
    recorded_scope step = requested.
Proof.
  intros requested current old_kind step next new_kind Hstep.
  unfold replay_step in Hstep.
  destruct (realization_scope_eq_dec requested (recorded_scope step))
    as [Hscope | Hscope]; [symmetry; exact Hscope | discriminate].
Qed.

Theorem replay_step_identifies_exact_rule_and_premises :
  forall requested current old_kind step next new_kind,
    replay_step requested current old_kind step = Some (next, new_kind) ->
    exists required effect,
      required_premises (recorded_rule step) = Some required /\
      recorded_premises step = required /\
      named_rule (recorded_rule step) /\
      rule_result (recorded_rule step) (recorded_direction step)
        (recorded_source step) (recorded_target step) = Some effect.
Proof.
  intros requested current old_kind step next new_kind Hstep.
  unfold replay_step in Hstep.
  destruct (realization_scope_eq_dec requested (recorded_scope step));
    [|discriminate].
  destruct (arithmetic_profile_eq_dec (recorded_arithmetic step)
    ExactNaturals); [|discriminate].
  destruct (arithmetic_profile_eq_dec
    (scope_arithmetic (base_scope requested)) ExactNaturals);
    [|discriminate].
  destruct (observation_profile_eq_dec
    (scope_observation (base_scope requested)) ScoreOnly);
    [|discriminate].
  destruct (recorded_witness_effect step); try discriminate.
  destruct (expression_eq_dec current (recorded_source step));
    [|discriminate].
  destruct (required_premises (recorded_rule step))
    as [required |] eqn:Hrequired; [|discriminate].
  destruct (list_eq_dec premise_name_eq_dec required
    (recorded_premises step)) as [Hnames | Hnames]; [|discriminate].
  destruct (rule_result (recorded_rule step) (recorded_direction step)
    (recorded_source step) (recorded_target step))
    as [effect |] eqn:Hrule; [|discriminate].
  exists required, effect; repeat split; try assumption.
  - symmetry; exact Hnames.
  - destruct (recorded_rule step); simpl; auto.
    simpl in Hrequired; discriminate.
Qed.

Theorem unknown_rule_never_replays :
  forall requested current old_kind step next new_kind identifier,
    recorded_rule step = UnknownRule identifier ->
    replay_step requested current old_kind step <> Some (next, new_kind).
Proof.
  intros requested current old_kind step next new_kind identifier Hunknown Hstep.
  destruct (replay_step_identifies_exact_rule_and_premises
    _ _ _ _ _ _ Hstep) as [required [effect [Hrequired [_ [Hnamed _]]]]].
  rewrite Hunknown in Hnamed. exact Hnamed.
Qed.

Theorem wrong_premise_list_never_replays :
  forall requested current old_kind step next new_kind required,
    required_premises (recorded_rule step) = Some required ->
    recorded_premises step <> required ->
    replay_step requested current old_kind step <> Some (next, new_kind).
Proof.
  intros requested current old_kind step next new_kind required
    Hrequired Hwrong Hstep.
  destruct (replay_step_identifies_exact_rule_and_premises
    _ _ _ _ _ _ Hstep) as [actual [effect [Hactual [Hnames _]]]].
  rewrite Hrequired in Hactual; congruence.
Qed.

Theorem reversed_lower_rule_never_replays :
  forall requested current old_kind step next new_kind named,
    recorded_rule step = NamedLower named ->
    recorded_direction step = Backward ->
    replay_step requested current old_kind step <> Some (next, new_kind).
Proof.
  intros requested current old_kind step next new_kind named
    Hnamed Hbackward Hstep.
  destruct (replay_step_identifies_exact_rule_and_premises
    _ _ _ _ _ _ Hstep) as [required [effect [_ [_ [_ Hrule]]]]].
  rewrite Hnamed, Hbackward in Hrule.
  discriminate Hrule.
Qed.

Theorem rounded_step_never_replays :
  forall requested current old_kind step next new_kind,
    recorded_arithmetic step = RoundedBinary64 ->
    replay_step requested current old_kind step <> Some (next, new_kind).
Proof.
  intros requested current old_kind step next new_kind Hrounded Hstep.
  unfold replay_step in Hstep.
  destruct (realization_scope_eq_dec requested (recorded_scope step));
    [|discriminate].
  destruct (arithmetic_profile_eq_dec (recorded_arithmetic step)
    ExactNaturals) as [Hexact | Hnot_exact]; [|discriminate].
  rewrite Hrounded in Hexact; discriminate Hexact.
Qed.

Definition relation_holds (origin current : cost_expression)
    (kind : relation_kind) (environment : nat -> nat) : Prop :=
  match kind with
  | Equivalent => evaluate environment current = evaluate environment origin
  | LowerThanSource => evaluate environment current <= evaluate environment origin
  end.

Lemma replay_step_preserves_relation :
  forall requested origin current old_kind step next new_kind,
    replay_step requested current old_kind step = Some (next, new_kind) ->
    (forall environment, relation_holds origin current old_kind environment) ->
    forall environment, relation_holds origin next new_kind environment.
Proof.
  intros requested origin current old_kind step next new_kind Hstep Hold environment.
  unfold replay_step in Hstep.
  destruct (realization_scope_eq_dec requested (recorded_scope step));
    [|discriminate].
  destruct (arithmetic_profile_eq_dec (recorded_arithmetic step)
    ExactNaturals); [|discriminate].
  destruct (arithmetic_profile_eq_dec
    (scope_arithmetic (base_scope requested)) ExactNaturals);
    [|discriminate].
  destruct (observation_profile_eq_dec
    (scope_observation (base_scope requested)) ScoreOnly);
    [|discriminate].
  destruct (recorded_witness_effect step); try discriminate.
  destruct (expression_eq_dec current (recorded_source step))
    as [Hsource | Hsource]; [|discriminate].
  destruct (required_premises (recorded_rule step))
    as [required |] eqn:Hrequired; [|discriminate].
  destruct (list_eq_dec premise_name_eq_dec required
    (recorded_premises step)); [|discriminate].
  destruct (rule_result (recorded_rule step) (recorded_direction step)
    (recorded_source step) (recorded_target step))
    as [effect |] eqn:Hrule; [|discriminate].
  inversion Hstep; subst next new_kind; subst current.
  pose proof (rule_result_sound _ _ _ _ _ environment Hrule) as Hsound.
  specialize (Hold environment).
  destruct old_kind, effect; unfold relation_holds in *; simpl in *; lia.
Qed.

(** This inductive relation is finite because every constructor consumes one
    list cell. The executable replay below computes exactly this relation. *)
Inductive finite_replay (requested : realization_scope) :
    cost_expression -> relation_kind -> list scoped_step ->
    cost_expression -> relation_kind -> Prop :=
| ReplayDone : forall current kind,
    finite_replay requested current kind [] current kind
| ReplayMore : forall current kind step remaining next next_kind final final_kind,
    replay_step requested current kind step = Some (next, next_kind) ->
    finite_replay requested next next_kind remaining final final_kind ->
    finite_replay requested current kind (step :: remaining) final final_kind.

Fixpoint replay_steps (requested : realization_scope)
    (current : cost_expression) (kind : relation_kind)
    (steps : list scoped_step) : option (cost_expression * relation_kind) :=
  match steps with
  | [] => Some (current, kind)
  | step :: remaining =>
      match replay_step requested current kind step with
      | Some (next, next_kind) =>
          replay_steps requested next next_kind remaining
      | None => None
      end
  end.

Theorem replay_steps_iff_finite_replay :
  forall requested steps source kind target final_kind,
    replay_steps requested source kind steps = Some (target, final_kind) <->
    finite_replay requested source kind steps target final_kind.
Proof.
  intros requested steps source kind target final_kind.
  split.
  - intros Hcomputed.
    revert source kind target final_kind Hcomputed.
    induction steps as [|step remaining IH];
      intros source kind target final_kind Hcomputed; simpl in Hcomputed.
    + inversion Hcomputed; subst. constructor.
    + destruct (replay_step requested source kind step)
        as [[next next_kind] |] eqn:Hstep; [|discriminate].
      eapply ReplayMore; [exact Hstep |].
      eapply IH; exact Hcomputed.
  - intro Hrelation.
    induction Hrelation; simpl; [reflexivity |].
    now rewrite H, IHHrelation.
Qed.

Theorem finite_replay_sound :
  forall requested steps origin source kind target final_kind,
    finite_replay requested source kind steps target final_kind ->
    (forall environment, relation_holds origin source kind environment) ->
    forall environment, relation_holds origin target final_kind environment.
Proof.
  intros requested steps origin source kind target final_kind Hreplay.
  induction Hreplay; intros Horigin environment.
  - apply Horigin.
  - apply IHHreplay.
    eapply replay_step_preserves_relation; eauto.
Qed.

Record scoped_certificate := {
  certificate_scope : realization_scope;
  certificate_steps : list scoped_step
}.

Definition accept_scoped (requested : realization_scope)
    (origin target : cost_expression) (claimed_kind : relation_kind)
    (certificate : scoped_certificate) : bool :=
  if realization_scope_eq_dec requested (certificate_scope certificate) then
  if arithmetic_profile_eq_dec
       (scope_arithmetic (base_scope requested)) ExactNaturals then
  if observation_profile_eq_dec
       (scope_observation (base_scope requested)) ScoreOnly then
    match replay_steps requested origin Equivalent (certificate_steps certificate) with
    | Some (result, result_kind) =>
        if expression_eq_dec result target then
        match claimed_kind, result_kind with
        | Equivalent, Equivalent | LowerThanSource, LowerThanSource => true
        | _, _ => false
        end else false
    | None => false
    end
  else false else false else false.

Theorem accepted_scoped_certificate_sound :
  forall requested origin target claimed_kind certificate,
    accept_scoped requested origin target claimed_kind certificate = true ->
    certificate_scope certificate = requested /\
    forall environment, relation_holds origin target claimed_kind environment.
Proof.
  intros requested origin target claimed_kind certificate Haccept.
  unfold accept_scoped in Haccept.
  destruct (realization_scope_eq_dec requested
    (certificate_scope certificate)) as [Hscope | Hscope]; [|discriminate].
  destruct (arithmetic_profile_eq_dec
    (scope_arithmetic (base_scope requested)) ExactNaturals);
    [|discriminate].
  destruct (observation_profile_eq_dec
    (scope_observation (base_scope requested)) ScoreOnly);
    [|discriminate].
  destruct (replay_steps requested origin Equivalent
    (certificate_steps certificate)) as [[result result_kind] |]
    eqn:Hreplay; [|discriminate].
  destruct (expression_eq_dec result target) as [Htarget | Htarget];
    [|discriminate].
  destruct claimed_kind, result_kind; simpl in Haccept; try discriminate.
  - split; [symmetry; exact Hscope |].
    intro environment; subst result.
    eapply finite_replay_sound;
      [apply replay_steps_iff_finite_replay; exact Hreplay |].
    intro another_environment; unfold relation_holds; reflexivity.
  - split; [symmetry; exact Hscope |].
    intro environment; subst result.
    eapply finite_replay_sound;
      [apply replay_steps_iff_finite_replay; exact Hreplay |].
    intro another_environment; unfold relation_holds; reflexivity.
Qed.

Definition sample_realization_scope : realization_scope :=
  {| base_scope := sample_scope;
     scoped_revision := 11;
     scoped_label_context := 13 |}.

Definition sample_forward_step : scoped_step :=
  {| recorded_rule := NamedExact AddZeroRight;
     recorded_premises :=
       [ExactNaturalArithmetic; ScoreOnlyObservation; ExactPattern];
     recorded_direction := Forward;
     recorded_source := Addition (InputVar 0) (Constant 0);
     recorded_target := InputVar 0;
     recorded_scope := sample_realization_scope;
     recorded_arithmetic := ExactNaturals;
     recorded_witness_effect := NoWitnessEffect |}.

Definition sample_lower_step : scoped_step :=
  {| recorded_rule := NamedLower AdditionLeftBound;
     recorded_premises :=
       [ExactNaturalArithmetic; ScoreOnlyObservation;
        OmittedSummandNonnegativity];
     recorded_direction := Forward;
     recorded_source := Addition (InputVar 0) (InputVar 1);
     recorded_target := InputVar 0;
     recorded_scope := sample_realization_scope;
     recorded_arithmetic := ExactNaturals;
     recorded_witness_effect := NoWitnessEffect |}.

Definition singleton_certificate (step : scoped_step) : scoped_certificate :=
  {| certificate_scope := sample_realization_scope;
     certificate_steps := [step] |}.

Example valid_exact_replay_is_accepted :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (Constant 0)) (InputVar 0)
    Equivalent (singleton_certificate sample_forward_step) = true.
Proof. reflexivity. Qed.

Example valid_lower_replay_is_accepted :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (InputVar 1)) (InputVar 0)
    LowerThanSource (singleton_certificate sample_lower_step) = true.
Proof. reflexivity. Qed.

Example lower_replay_cannot_claim_equivalence :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (InputVar 1)) (InputVar 0)
    Equivalent (singleton_certificate sample_lower_step) = false.
Proof. reflexivity. Qed.

Example missing_premise_is_rejected :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (Constant 0)) (InputVar 0)
    Equivalent
    (singleton_certificate
      {| recorded_rule := NamedExact AddZeroRight;
         recorded_premises := [ExactNaturalArithmetic; ExactPattern];
         recorded_direction := Forward;
         recorded_source := Addition (InputVar 0) (Constant 0);
         recorded_target := InputVar 0;
         recorded_scope := sample_realization_scope;
         recorded_arithmetic := ExactNaturals;
         recorded_witness_effect := NoWitnessEffect |}) = false.
Proof. reflexivity. Qed.

Example reversed_lower_simulation_is_rejected :
  accept_scoped sample_realization_scope
    (InputVar 0) (Addition (InputVar 0) (InputVar 1))
    LowerThanSource
    (singleton_certificate
      {| recorded_rule := NamedLower AdditionLeftBound;
         recorded_premises :=
           [ExactNaturalArithmetic; ScoreOnlyObservation;
            OmittedSummandNonnegativity];
         recorded_direction := Backward;
         recorded_source := InputVar 0;
         recorded_target := Addition (InputVar 0) (InputVar 1);
         recorded_scope := sample_realization_scope;
         recorded_arithmetic := ExactNaturals;
         recorded_witness_effect := NoWitnessEffect |}) = false.
Proof. reflexivity. Qed.

Example arithmetic_profile_mismatch_is_rejected :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (Constant 0)) (InputVar 0)
    Equivalent
    (singleton_certificate
      {| recorded_rule := NamedExact AddZeroRight;
         recorded_premises :=
           [ExactNaturalArithmetic; ScoreOnlyObservation; ExactPattern];
         recorded_direction := Forward;
         recorded_source := Addition (InputVar 0) (Constant 0);
         recorded_target := InputVar 0;
         recorded_scope := sample_realization_scope;
         recorded_arithmetic := RoundedBinary64;
         recorded_witness_effect := NoWitnessEffect |}) = false.
Proof. reflexivity. Qed.

Example changed_revision_is_rejected :
  accept_scoped
    {| base_scope := sample_scope;
       scoped_revision := 12;
       scoped_label_context := 13 |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0)
    Equivalent (singleton_certificate sample_forward_step) = false.
Proof. reflexivity. Qed.

Example unsupported_witness_effect_is_rejected :
  accept_scoped sample_realization_scope
    (Addition (InputVar 0) (Constant 0)) (InputVar 0)
    Equivalent
    (singleton_certificate
      {| recorded_rule := NamedExact AddZeroRight;
         recorded_premises :=
           [ExactNaturalArithmetic; ScoreOnlyObservation; ExactPattern];
         recorded_direction := Forward;
         recorded_source := Addition (InputVar 0) (Constant 0);
         recorded_target := InputVar 0;
         recorded_scope := sample_realization_scope;
         recorded_arithmetic := ExactNaturals;
         recorded_witness_effect := PreservesWitness |}) = false.
Proof. reflexivity. Qed.

Example unsupported_rule_is_rejected :
  accept_scoped sample_realization_scope (InputVar 0) (InputVar 0)
    Equivalent
    (singleton_certificate
      {| recorded_rule := UnknownRule 41;
         recorded_premises := [];
         recorded_direction := Forward;
         recorded_source := InputVar 0;
         recorded_target := InputVar 0;
         recorded_scope := sample_realization_scope;
         recorded_arithmetic := ExactNaturals;
         recorded_witness_effect := NoWitnessEffect |}) = false.
Proof. reflexivity. Qed.
