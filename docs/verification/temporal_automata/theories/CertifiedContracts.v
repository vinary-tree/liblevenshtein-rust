(** * A small checked construction language for ordered-cost expressions

    This file is a complete checker for its declared finite expression
    language. It illustrates the CBC acceptance seam. It does not model
    binary64, dictionary ownership, Rust execution, or arbitrary ORC IR.
    Those require separate instance and correspondence theorems. *)

From Stdlib Require Import Arith Bool Lia List.
Import ListNotations.

Inductive arithmetic_profile := ExactNaturals | RoundedBinary64.

Inductive observation_profile := ScoreOnly | OrderedResult | CanonicalWitness.

Definition observation_profile_eq_dec (lhs rhs : observation_profile) :
    {lhs = rhs} + {lhs <> rhs}.
Proof. decide equality. Defined.

Definition arithmetic_profile_eq_dec (lhs rhs : arithmetic_profile) :
    {lhs = rhs} + {lhs <> rhs}.
Proof. decide equality. Defined.

Record contract_scope := {
  scope_domain : nat;
  scope_query : nat;
  scope_parameters : nat;
  scope_arithmetic : arithmetic_profile;
  scope_snapshot : nat;
  scope_cutoff : nat;
  scope_observation : observation_profile
}.

Definition scope_eq_dec (lhs rhs : contract_scope) :
    {lhs = rhs} + {lhs <> rhs}.
Proof.
  decide equality;
    try apply Nat.eq_dec;
    try apply arithmetic_profile_eq_dec;
    try apply observation_profile_eq_dec.
Defined.

Inductive cost_expression :=
| InputVar : nat -> cost_expression
| Constant : nat -> cost_expression
| Addition : cost_expression -> cost_expression -> cost_expression
| Minimum : cost_expression -> cost_expression -> cost_expression.

Definition expression_eq_dec (lhs rhs : cost_expression) :
    {lhs = rhs} + {lhs <> rhs}.
Proof. decide equality; apply Nat.eq_dec. Defined.

Fixpoint evaluate (environment : nat -> nat) (term : cost_expression) : nat :=
  match term with
  | InputVar index => environment index
  | Constant value => value
  | Addition lhs rhs => evaluate environment lhs + evaluate environment rhs
  | Minimum lhs rhs => Nat.min (evaluate environment lhs) (evaluate environment rhs)
  end.

(** Exact rules carry the entire target syntax. The checker independently
    recomputes that target from the source and named rule. *)
Inductive exact_rule :=
| AddZeroRight
| AddZeroLeft
| SwapMinimum
| AssociateAddition.

Definition exact_rewrite (rule : exact_rule) (source : cost_expression)
    : option cost_expression :=
  match rule, source with
  | AddZeroRight, Addition lhs (Constant 0) => Some lhs
  | AddZeroLeft, Addition (Constant 0) rhs => Some rhs
  | SwapMinimum, Minimum lhs rhs => Some (Minimum rhs lhs)
  | AssociateAddition, Addition (Addition first middle) last =>
      Some (Addition first (Addition middle last))
  | _, _ => None
  end.

Lemma exact_rewrite_sound : forall rule source target environment,
  exact_rewrite rule source = Some target ->
  evaluate environment source = evaluate environment target.
Proof.
  intros rule source target environment Hrewrite.
  destruct rule; destruct source; simpl in Hrewrite; try discriminate.
  - destruct source2; try discriminate.
    destruct n; try discriminate.
    inversion Hrewrite; subst; simpl; lia.
  - destruct source1; try discriminate.
    destruct n; try discriminate.
    inversion Hrewrite; subst; simpl; lia.
  - inversion Hrewrite; subst; simpl.
    now rewrite Nat.min_comm.
  - destruct source1; try discriminate.
    inversion Hrewrite; subst; simpl; lia.
Qed.

Record exact_step := {
  step_rule : exact_rule;
  step_source : cost_expression;
  step_target : cost_expression
}.

Fixpoint replay_exact (current : cost_expression) (steps : list exact_step)
    : option cost_expression :=
  match steps with
  | [] => Some current
  | step :: remaining =>
      if expression_eq_dec current (step_source step) then
        match exact_rewrite (step_rule step) (step_source step) with
        | Some expected =>
            if expression_eq_dec expected (step_target step) then
              replay_exact (step_target step) remaining
            else None
        | None => None
        end
      else None
  end.

Lemma replay_exact_sound : forall steps source target environment,
  replay_exact source steps = Some target ->
  evaluate environment source = evaluate environment target.
Proof.
  induction steps as [|step remaining IH];
    intros source target environment Hreplay; simpl in Hreplay.
  - inversion Hreplay; reflexivity.
  - destruct (expression_eq_dec source (step_source step))
      as [Hsource | Hsource]; [|discriminate].
    destruct (exact_rewrite (step_rule step) (step_source step))
      as [expected |] eqn:Hrule; [|discriminate].
    destruct (expression_eq_dec expected (step_target step))
      as [Htarget | Htarget]; [|discriminate].
    subst source expected.
    eapply eq_trans.
    + eapply exact_rewrite_sound; eauto.
    + eapply IH; eauto.
Qed.

Record exact_certificate := {
  exact_scope : contract_scope;
  exact_steps : list exact_step
}.

Definition accept_exact (requested : contract_scope)
    (source target : cost_expression) (certificate : exact_certificate) : bool :=
  if arithmetic_profile_eq_dec (scope_arithmetic requested) ExactNaturals then
  if observation_profile_eq_dec (scope_observation requested) ScoreOnly then
  if scope_eq_dec requested (exact_scope certificate) then
    match replay_exact source (exact_steps certificate) with
    | Some result => if expression_eq_dec result target then true else false
    | None => false
    end
  else false else false else false.

Theorem accepted_exact_requires_exact_natural_arithmetic :
  forall requested source target certificate,
    accept_exact requested source target certificate = true ->
    scope_arithmetic requested = ExactNaturals.
Proof.
  intros requested source target certificate Haccept.
  unfold accept_exact in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [assumption | discriminate].
Qed.

Theorem accepted_exact_requires_score_only_observation :
  forall requested source target certificate,
    accept_exact requested source target certificate = true ->
    scope_observation requested = ScoreOnly.
Proof.
  intros requested source target certificate Haccept.
  unfold accept_exact in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [|discriminate].
  destruct (observation_profile_eq_dec (scope_observation requested)
    ScoreOnly); [assumption | discriminate].
Qed.

Theorem accepted_exact_certificate_preserves_every_environment :
  forall requested source target certificate,
    accept_exact requested source target certificate = true ->
    exact_scope certificate = requested /\
    forall environment, evaluate environment source = evaluate environment target.
Proof.
  intros requested source target certificate Haccept.
  unfold accept_exact in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [|discriminate].
  destruct (observation_profile_eq_dec (scope_observation requested)
    ScoreOnly); [|discriminate].
  destruct (scope_eq_dec requested (exact_scope certificate))
    as [Hscope | Hscope]; [|discriminate].
  destruct (replay_exact source (exact_steps certificate))
    as [result |] eqn:Hreplay; [|discriminate].
  destruct (expression_eq_dec result target)
    as [Hequal | Hequal]; [|discriminate].
  split; [symmetry; exact Hscope |].
  intros environment; subst result.
  eapply replay_exact_sound; eauto.
Qed.

Record integer_measure_specification := {
  measure_scope : contract_scope;
  measure_expression : cost_expression
}.

Definition accept_realization (specification : integer_measure_specification)
    (artifact : cost_expression) (certificate : exact_certificate) : bool :=
  accept_exact (measure_scope specification) (measure_expression specification)
    artifact certificate.

Theorem accepted_realization_implements_its_measure :
  forall specification artifact certificate,
    accept_realization specification artifact certificate = true ->
    forall environment,
      evaluate environment (measure_expression specification) =
      evaluate environment artifact.
Proof.
  intros specification artifact certificate Haccept environment.
  unfold accept_realization in Haccept.
  destruct (accepted_exact_certificate_preserves_every_environment
    _ _ _ _ Haccept) as [_ Hsemantics].
  now apply Hsemantics.
Qed.

Theorem accepted_exact_certificates_compose_semantically :
  forall requested first middle last left_certificate right_certificate,
    accept_exact requested first middle left_certificate = true ->
    accept_exact requested middle last right_certificate = true ->
    forall environment, evaluate environment first = evaluate environment last.
Proof.
  intros requested first middle last left_certificate right_certificate
    Hleft Hright environment.
  destruct (accepted_exact_certificate_preserves_every_environment
    _ _ _ _ Hleft) as [_ Hleft_semantics].
  destruct (accepted_exact_certificate_preserves_every_environment
    _ _ _ _ Hright) as [_ Hright_semantics].
  now rewrite Hleft_semantics, Hright_semantics.
Qed.

Definition score_trace (term : cost_expression)
    (inputs : list (nat -> nat)) : list nat :=
  map (fun environment => evaluate environment term) inputs.

Theorem accepted_exact_certificate_preserves_finite_score_trace :
  forall requested source target certificate,
    accept_exact requested source target certificate = true ->
    forall inputs, score_trace source inputs = score_trace target inputs.
Proof.
  intros requested source target certificate Haccept inputs.
  destruct (accepted_exact_certificate_preserves_every_environment
    _ _ _ _ Haccept) as [_ Hvalue].
  unfold score_trace.
  induction inputs as [|environment rest IH]; simpl; [reflexivity |].
  now rewrite Hvalue, IH.
Qed.

(** A lower-bound certificate has a separate language and conclusion. It
    cannot be passed to accept_exact. Natural-number nonnegativity is an
    explicit property of this cost carrier. *)
Inductive lower_rule :=
| ZeroBound
| AdditionLeftBound
| AdditionRightBound.

Definition construct_lower (rule : lower_rule) (exact : cost_expression)
    : option cost_expression :=
  match rule, exact with
  | ZeroBound, _ => Some (Constant 0)
  | AdditionLeftBound, Addition lhs _ => Some lhs
  | AdditionRightBound, Addition _ rhs => Some rhs
  | _, _ => None
  end.

Lemma construct_lower_sound : forall rule exact lower environment,
  construct_lower rule exact = Some lower ->
  evaluate environment lower <= evaluate environment exact.
Proof.
  intros rule exact lower environment Hconstruct.
  destruct rule; destruct exact; simpl in Hconstruct; try discriminate;
    inversion Hconstruct; subst; simpl; lia.
Qed.

Record lower_certificate := {
  lower_scope : contract_scope;
  lower_rule_name : lower_rule
}.

Definition accept_lower (requested : contract_scope)
    (exact lower : cost_expression) (certificate : lower_certificate) : bool :=
  if arithmetic_profile_eq_dec (scope_arithmetic requested) ExactNaturals then
  if observation_profile_eq_dec (scope_observation requested) ScoreOnly then
  if scope_eq_dec requested (lower_scope certificate) then
    match construct_lower (lower_rule_name certificate) exact with
    | Some result => if expression_eq_dec result lower then true else false
    | None => false
    end
  else false else false else false.

Theorem accepted_lower_requires_exact_natural_arithmetic :
  forall requested exact lower certificate,
    accept_lower requested exact lower certificate = true ->
    scope_arithmetic requested = ExactNaturals.
Proof.
  intros requested exact lower certificate Haccept.
  unfold accept_lower in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [assumption | discriminate].
Qed.

Theorem accepted_lower_requires_score_only_observation :
  forall requested exact lower certificate,
    accept_lower requested exact lower certificate = true ->
    scope_observation requested = ScoreOnly.
Proof.
  intros requested exact lower certificate Haccept.
  unfold accept_lower in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [|discriminate].
  destruct (observation_profile_eq_dec (scope_observation requested)
    ScoreOnly); [assumption | discriminate].
Qed.

Theorem accepted_lower_certificate_bounds_every_environment :
  forall requested exact lower certificate,
    accept_lower requested exact lower certificate = true ->
    lower_scope certificate = requested /\
    forall environment, evaluate environment lower <= evaluate environment exact.
Proof.
  intros requested exact lower certificate Haccept.
  unfold accept_lower in Haccept.
  destruct (arithmetic_profile_eq_dec (scope_arithmetic requested)
    ExactNaturals); [|discriminate].
  destruct (observation_profile_eq_dec (scope_observation requested)
    ScoreOnly); [|discriminate].
  destruct (scope_eq_dec requested (lower_scope certificate))
    as [Hscope | Hscope]; [|discriminate].
  destruct (construct_lower (lower_rule_name certificate) exact)
    as [result |] eqn:Hlower; [|discriminate].
  destruct (expression_eq_dec result lower)
    as [Hequal | Hequal]; [|discriminate].
  split; [symmetry; exact Hscope |].
  intros environment; subst result.
  eapply construct_lower_sound; eauto.
Qed.

Definition sample_scope : contract_scope :=
  {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
     scope_arithmetic := ExactNaturals; scope_snapshot := 5;
     scope_cutoff := 7; scope_observation := ScoreOnly |}.

Definition sample_exact : exact_certificate :=
  {| exact_scope := sample_scope;
     exact_steps := [{| step_rule := AddZeroRight;
                         step_source := Addition (InputVar 0) (Constant 0);
                         step_target := InputVar 0 |}] |}.

Example valid_exact_certificate_is_accepted :
  accept_exact sample_scope (Addition (InputVar 0) (Constant 0))
    (InputVar 0) sample_exact = true.
Proof. reflexivity. Qed.

Example changed_snapshot_is_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := ExactNaturals; scope_snapshot := 8;
       scope_cutoff := 7; scope_observation := ScoreOnly |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example changed_query_is_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 8; scope_parameters := 3;
       scope_arithmetic := ExactNaturals; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := ScoreOnly |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example changed_parameters_are_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 2; scope_parameters := 8;
       scope_arithmetic := ExactNaturals; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := ScoreOnly |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example changed_arithmetic_is_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := RoundedBinary64; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := ScoreOnly |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example matching_rounded_scopes_cannot_authorize_natural_rewrites :
  let rounded_scope :=
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := RoundedBinary64; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := ScoreOnly |} in
  accept_exact rounded_scope
    (Addition (Addition (InputVar 0) (InputVar 1)) (InputVar 2))
    (Addition (InputVar 0) (Addition (InputVar 1) (InputVar 2)))
    {| exact_scope := rounded_scope;
       exact_steps :=
         [{| step_rule := AssociateAddition;
             step_source := Addition
               (Addition (InputVar 0) (InputVar 1)) (InputVar 2);
             step_target := Addition
               (InputVar 0) (Addition (InputVar 1) (InputVar 2)) |}] |} = false.
Proof. reflexivity. Qed.

Example changed_observation_is_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := ExactNaturals; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := CanonicalWitness |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example matching_witness_scopes_cannot_authorize_score_only_rewrites :
  let witness_scope :=
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := ExactNaturals; scope_snapshot := 5;
       scope_cutoff := 7; scope_observation := CanonicalWitness |} in
  accept_exact witness_scope (Addition (InputVar 0) (Constant 0))
    (InputVar 0)
    {| exact_scope := witness_scope;
       exact_steps := exact_steps sample_exact |} = false.
Proof. reflexivity. Qed.

Example changed_cutoff_is_rejected :
  accept_exact
    {| scope_domain := 1; scope_query := 2; scope_parameters := 3;
       scope_arithmetic := ExactNaturals; scope_snapshot := 5;
       scope_cutoff := 8; scope_observation := ScoreOnly |}
    (Addition (InputVar 0) (Constant 0)) (InputVar 0) sample_exact = false.
Proof. reflexivity. Qed.

Example missing_rewrite_is_rejected :
  accept_exact sample_scope (Addition (InputVar 0) (Constant 0))
    (InputVar 0) {| exact_scope := sample_scope; exact_steps := [] |} = false.
Proof. reflexivity. Qed.

Example fabricated_target_is_rejected :
  accept_exact sample_scope (Addition (InputVar 0) (Constant 0))
    (Constant 17) sample_exact = false.
Proof. reflexivity. Qed.

Example wrong_rule_is_rejected :
  accept_exact sample_scope (Addition (InputVar 0) (Constant 0))
    (InputVar 0)
    {| exact_scope := sample_scope;
       exact_steps := [{| step_rule := SwapMinimum;
                          step_source := Addition (InputVar 0) (Constant 0);
                          step_target := InputVar 0 |}] |} = false.
Proof. reflexivity. Qed.

Example broken_chain_is_rejected :
  accept_exact sample_scope (Addition (InputVar 0) (Constant 0))
    (InputVar 1)
    {| exact_scope := sample_scope;
       exact_steps := [
         {| step_rule := AddZeroRight;
            step_source := Addition (InputVar 0) (Constant 0);
            step_target := InputVar 0 |};
         {| step_rule := AddZeroRight;
            step_source := Addition (InputVar 1) (Constant 0);
            step_target := InputVar 1 |}] |} = false.
Proof. reflexivity. Qed.

Example separate_lower_certificate_is_accepted :
  accept_lower sample_scope (Addition (InputVar 0) (InputVar 1))
    (InputVar 0)
    {| lower_scope := sample_scope; lower_rule_name := AdditionLeftBound |} = true.
Proof. reflexivity. Qed.

(** A lower certificate is not accepted at the exact checker boundary. *)
Fail Check (fun (certificate : lower_certificate) =>
  accept_exact sample_scope (InputVar 0) (InputVar 0) certificate).
