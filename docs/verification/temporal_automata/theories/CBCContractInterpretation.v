(** * Profile-indexed CBC operation contracts

    This is an interpretation language, not a checker for arbitrary ORC IR.
    Every operation instance must prove its reference laws and bind the
    abstract numerical, failure, continuation, and accounting relations to
    its executable. No production correspondence follows from this file. *)

From Stdlib Require Import Arith Lia List Sorting.Sorted Sorting.Permutation.
Import ListNotations.
Set Implicit Arguments.

Inductive observation_profile :=
| OnlineScore | CompletedRange | CompletedKnn
| CanonicalWitness | ResumableOperation | ResourceSafety.

Inductive search_mode := RangeSearch | KnnSearch.

Inductive observed_score (Cost Failure : Type) :=
| ExactScore : Cost -> observed_score Cost Failure
| BeyondCutoff : observed_score Cost Failure
| ScoreFailure : Failure -> observed_score Cost Failure
| ScoreIncomplete : Failure -> observed_score Cost Failure.
Arguments ExactScore {Cost Failure} _.
Arguments BeyondCutoff {Cost Failure}.
Arguments ScoreFailure {Cost Failure} _.
Arguments ScoreIncomplete {Cost Failure} _.

Inductive operation_outcome (A Failure : Type) :=
| Completed : A -> operation_outcome A Failure
| Incomplete : option A -> Failure -> operation_outcome A Failure
| Failed : Failure -> operation_outcome A Failure.
Arguments Completed {A Failure} _.
Arguments Incomplete {A Failure} _ _.
Arguments Failed {A Failure} _.

Record scored_original (Original Identity Cost : Type) := {
  scored_original_id : Original;
  scored_identity : Identity;
  scored_cost : Cost
}.

Record continuation_view (Snapshot Original Failure Continuation : Type) := {
  view_snapshot : Snapshot;
  view_published : list Original;
  view_private : list Original;
  view_remaining : list Original;
  view_excluded : list Original;
  view_reason : option Failure;
  view_continuation : option Continuation
}.

Record resource_view := {
  executed_work : nat;
  reserved_work : nat;
  charged_work : nat;
  live_bytes : nat;
  peak_bytes : nat;
  page_work : nat;
  page_results : nat
}.

Record resource_limits := {
  cumulative_work_limit : option nat;
  peak_byte_limit : option nat;
  page_work_limit : option nat;
  page_result_limit : option nat
}.

Definition under_limit (value : nat) (limit : option nat) : Prop :=
  match limit with Some ceiling => value <= ceiling | None => True end.

Definition within_limits (limits : resource_limits) (view : resource_view)
    : Prop :=
  live_bytes view <= peak_bytes view /\
  under_limit (charged_work view + reserved_work view)
    (cumulative_work_limit limits) /\
  under_limit (peak_bytes view) (peak_byte_limit limits) /\
  under_limit (page_work view) (page_work_limit limits) /\
  under_limit (page_results view) (page_result_limit limits).

Fixpoint monotone_resource_views (views : list resource_view) : Prop :=
  match views with
  | first :: ((second :: _) as rest) =>
      charged_work first <= charged_work second /\
      peak_bytes first <= peak_bytes second /\
      monotone_resource_views rest
  | _ => True
  end.

Section Contract.
  Context {Input Parameters Numeric Cost DomainError Original Identity
    RangeTie KnnTie Witness Failure Snapshot Continuation ResourceEvent : Type}.

  Definition result := scored_original Original Identity Cost.
  Definition session_view :=
    continuation_view Snapshot Original Failure Continuation.

  Definition observation_request (selected : observation_profile) : Type :=
    match selected with
    | OnlineScore => Input
    | CanonicalWitness => Original
    | _ => unit
    end.

  Definition observation_payload (selected : observation_profile) : Type :=
    match selected with
    | OnlineScore => list (Input * observed_score Cost Failure)
    | CompletedRange | CompletedKnn =>
        operation_outcome (list result) Failure
    | CanonicalWitness =>
        operation_outcome (option (result * Witness)) Failure
    | ResumableOperation => operation_outcome session_view Failure
    | ResourceSafety => list ResourceEvent * list resource_view
    end.

  Record measure_specification := {
    valid_input : Input -> Prop;
    validate_input : Input -> option DomainError;
    validation_sound : forall input,
      validate_input input = None -> valid_input input;
    validation_complete : forall input,
      valid_input input -> validate_input input = None;
    quotient_identity : Input -> Input -> Prop;
    quotient_refl : forall input,
      valid_input input -> quotient_identity input input;
    quotient_sym : forall first second,
      quotient_identity first second -> quotient_identity second first;
    quotient_trans : forall first middle last,
      quotient_identity first middle -> quotient_identity middle last ->
      quotient_identity first last;
    fixed_parameters : Parameters;
    numeric_authority : Numeric;
    measure : Numeric -> Parameters -> Input -> Input -> Cost;
    valid_cost : Cost -> Prop;
    cost_zero : Cost;
    cost_add : Cost -> Cost -> Cost;
    cost_le : Cost -> Cost -> Prop;
    cost_lt : Cost -> Cost -> Prop;
    cost_le_reflexive : forall cost,
      valid_cost cost -> cost_le cost cost;
    cost_le_transitive : forall first middle last,
      cost_le first middle -> cost_le middle last -> cost_le first last;
    cost_le_total : forall first second,
      valid_cost first -> valid_cost second ->
      cost_le first second \/ cost_le second first;
    cost_lt_characterization : forall first second,
      valid_cost first -> valid_cost second ->
      (cost_lt first second <->
        cost_le first second /\ ~ cost_le second first);
    zero_valid : valid_cost cost_zero;
    measure_valid : forall first second,
      valid_input first -> valid_input second ->
      valid_cost (measure numeric_authority fixed_parameters first second)
  }.

  Definition cost_equivalent (spec : measure_specification)
      (first second : Cost) : Prop :=
    cost_le spec first second /\ cost_le spec second first.

  Definition reference_cost (spec : measure_specification)
      (query input : Input) : Cost :=
    measure spec (numeric_authority spec) (fixed_parameters spec) query input.

  Record operation_contract := {
    operation_measure : measure_specification;
    fixed_query : Input;
    fixed_query_valid : valid_input operation_measure fixed_query;
    captured_snapshot : Snapshot;
    snapshot_originals : Snapshot -> list Original;
    original_identity : Original -> Identity;
    snapshot_identity_unique :
      NoDup (map original_identity (snapshot_originals captured_snapshot));
    original_input : Original -> Input;
    eligible_original : Original -> Prop;
    eligible_input_valid : forall original,
      eligible_original original ->
      valid_input operation_measure (original_input original);
    range_tie : Original -> RangeTie;
    range_tie_before : RangeTie -> RangeTie -> Prop;
    knn_tie : Original -> KnnTie;
    knn_tie_before : KnnTie -> KnnTie -> Prop;
    inclusive_cutoff : Cost;
    requested_k : nat;
    ideal_range : list result;
    ideal_knn_order : list result;
    attempted_prefixes : Input -> list Input;
    committed_prefixes : Input -> list Input;
    is_prefix : Input -> Input -> Prop;
    attempted_prefixes_well_formed : forall input,
      attempted_prefixes input <> [] /\
      last (attempted_prefixes input) input = input /\
      NoDup (attempted_prefixes input) /\
      Forall (fun prefix => is_prefix prefix input) (attempted_prefixes input);
    witness_feasible : Original -> Witness -> Prop;
    witness_cost : Witness -> Cost;
    witness_tie_before : Witness -> Witness -> Prop;
    domain_failure : DomainError -> Failure;
    allowed_failure : forall selected,
      observation_request selected -> Failure -> Prop;
    allowed_incomplete : forall selected,
      observation_request selected -> Failure -> Prop;
    terminal_incomplete : Failure -> session_view -> Prop;
    continuation_denotes : Continuation -> list Original -> Prop;
    session_mode : search_mode;
    resource_accounting :
      list ResourceEvent -> list resource_view -> Prop;
    limits : resource_limits;
    selected_profile : observation_profile -> Prop;
    selected_nonempty : exists selected, selected_profile selected;
    reference_observe : forall selected,
      observation_request selected -> observation_payload selected
  }.

  Definition candidate_score (contract : operation_contract)
      (original : Original) : Cost :=
    reference_cost (operation_measure contract) (fixed_query contract)
      (original_input contract original).

  Definition common_result_law (contract : operation_contract)
      (row : result) : Prop :=
    In (scored_original_id row)
      (snapshot_originals contract (captured_snapshot contract)) /\
    eligible_original contract (scored_original_id row) /\
    scored_identity row = original_identity contract (scored_original_id row) /\
    scored_cost row = candidate_score contract (scored_original_id row).

  Definition range_eligible (contract : operation_contract)
      (original : Original) : Prop :=
    In original (snapshot_originals contract (captured_snapshot contract)) /\
    eligible_original contract original /\
    cost_le (operation_measure contract) (candidate_score contract original)
      (inclusive_cutoff contract).

  Definition knn_eligible (contract : operation_contract)
      (original : Original) : Prop :=
    In original (snapshot_originals contract (captured_snapshot contract)) /\
    eligible_original contract original.

  Definition range_rank_less (contract : operation_contract)
      (first second : result) : Prop :=
    range_tie_before contract
      (range_tie contract (scored_original_id first))
      (range_tie contract (scored_original_id second)).

  Definition knn_rank_less (contract : operation_contract)
      (first second : result) : Prop :=
    cost_lt (operation_measure contract)
      (scored_cost first) (scored_cost second) \/
    (cost_equivalent (operation_measure contract)
       (scored_cost first) (scored_cost second) /\
     knn_tie_before contract
       (knn_tie contract (scored_original_id first))
       (knn_tie contract (scored_original_id second))).

  Definition original_of (rows : list result) : list Original :=
    map (fun row : result => scored_original_id row) rows.

  Definition resume_goal (contract : operation_contract) : list Original :=
    match session_mode contract with
    | RangeSearch => original_of (ideal_range contract)
    | KnnSearch =>
        original_of (firstn (requested_k contract) (ideal_knn_order contract))
    end.

  Definition ideal_range_law (contract : operation_contract) : Prop :=
    Forall (common_result_law contract) (ideal_range contract) /\
    NoDup (original_of (ideal_range contract)) /\
    StronglySorted (range_rank_less contract) (ideal_range contract) /\
    forall original,
      In original (original_of (ideal_range contract)) <->
      range_eligible contract original.

  Definition ideal_knn_law (contract : operation_contract) : Prop :=
    Forall (common_result_law contract) (ideal_knn_order contract) /\
    NoDup (original_of (ideal_knn_order contract)) /\
    StronglySorted (knn_rank_less contract) (ideal_knn_order contract) /\
    forall original,
      In original (original_of (ideal_knn_order contract)) <->
      knn_eligible contract original.

  Definition online_event_law (contract : operation_contract)
      (prefix : Input) (event : observed_score Cost Failure) : Prop :=
    match event with
    | ExactScore score =>
        validate_input (operation_measure contract) prefix = None /\
        score = reference_cost (operation_measure contract)
          (fixed_query contract) prefix /\
        cost_le (operation_measure contract) score
          (inclusive_cutoff contract)
    | BeyondCutoff =>
        validate_input (operation_measure contract) prefix = None /\
        ~ cost_le (operation_measure contract)
            (reference_cost (operation_measure contract)
              (fixed_query contract) prefix)
            (inclusive_cutoff contract)
    | ScoreFailure reason =>
        (exists error,
          validate_input (operation_measure contract) prefix = Some error /\
          reason = domain_failure contract error) \/
        allowed_failure contract OnlineScore prefix reason
    | ScoreIncomplete reason =>
        allowed_incomplete contract OnlineScore prefix reason
    end.

  Definition online_observation_law (contract : operation_contract)
      (input : Input) (events : list (Input * observed_score Cost Failure))
      : Prop :=
    map fst events = attempted_prefixes contract input /\
    Forall (fun event =>
      is_prefix contract (fst event) input /\
      online_event_law contract (fst event) (snd event)) events /\
    Forall (fun event =>
      match snd event with
      | ScoreFailure _ | ScoreIncomplete _ => False
      | _ => True
      end)
      (removelast events) /\
    match rev events with
    | [] => False
    | (_, ScoreFailure _) :: _ | (_, ScoreIncomplete _) :: _ =>
        committed_prefixes contract input =
          removelast (attempted_prefixes contract input)
    | _ => committed_prefixes contract input = attempted_prefixes contract input
    end.

  Definition partial_rows_law (contract : operation_contract)
      (ideal rows : list result) (rank : result -> result -> Prop) : Prop :=
    incl rows ideal /\
    NoDup (original_of rows) /\
    StronglySorted rank rows.

  Definition range_observation_law (contract : operation_contract)
      (outcome : operation_outcome (list result) Failure) : Prop :=
    match outcome with
    | Completed rows => rows = ideal_range contract
    | Incomplete partial reason =>
        allowed_incomplete contract CompletedRange tt reason /\
        match partial with
        | Some rows => partial_rows_law contract (ideal_range contract)
            rows (range_rank_less contract)
        | None => True
        end
    | Failed reason => allowed_failure contract CompletedRange tt reason
    end.

  Definition knn_observation_law (contract : operation_contract)
      (outcome : operation_outcome (list result) Failure) : Prop :=
    match outcome with
    | Completed rows =>
        rows = firstn (requested_k contract) (ideal_knn_order contract)
    | Incomplete partial reason =>
        allowed_incomplete contract CompletedKnn tt reason /\
        match partial with
        | Some rows => partial_rows_law contract (ideal_knn_order contract)
            rows (knn_rank_less contract)
        | None => True
        end
    | Failed reason => allowed_failure contract CompletedKnn tt reason
    end.

  Definition canonical_witness_law (contract : operation_contract)
      (original : Original) (witness : Witness) : Prop :=
    witness_feasible contract original witness /\
    witness_cost contract witness = candidate_score contract original /\
    forall alternative,
      witness_feasible contract original alternative ->
      cost_le (operation_measure contract)
        (candidate_score contract original) (witness_cost contract alternative) /\
      (cost_equivalent (operation_measure contract)
         (witness_cost contract alternative)
         (candidate_score contract original) ->
       ~ witness_tie_before contract alternative witness).

  Definition partial_witness_law (contract : operation_contract)
      (original : Original) (row : result) (witness : Witness) : Prop :=
    In original
      (snapshot_originals contract (captured_snapshot contract)) /\
    scored_original_id row = original /\
    scored_identity row = original_identity contract original /\
    witness_feasible contract original witness /\
    scored_cost row = witness_cost contract witness.

  Definition witness_observation_law (contract : operation_contract)
      (original : Original)
      (outcome : operation_outcome (option (result * Witness)) Failure) : Prop :=
    match outcome with
    | Completed None => ~ range_eligible contract original
    | Completed (Some (row, witness)) =>
        scored_original_id row = original /\
        common_result_law contract row /\
        range_eligible contract original /\
        canonical_witness_law contract original witness
    | Incomplete partial reason =>
        allowed_incomplete contract CanonicalWitness original reason /\
        match partial with
        | Some (Some (row, witness)) =>
            partial_witness_law contract original row witness
        | _ => True
        end
    | Failed reason => allowed_failure contract CanonicalWitness original reason
    end.

  Definition session_partition_law (contract : operation_contract)
      (view : session_view) : Prop :=
    view_snapshot view = captured_snapshot contract /\
    Permutation
      (view_published view ++ view_private view ++
       view_remaining view ++ view_excluded view)
      (snapshot_originals contract (captured_snapshot contract)) /\
    (forall original,
      In original (view_published view) ->
      In original (resume_goal contract)) /\
    (forall original,
      In original (view_excluded view) ->
      ~ In original (resume_goal contract)).

  Definition resume_observation_law (contract : operation_contract)
      (outcome : operation_outcome session_view Failure) : Prop :=
    match outcome with
    | Completed view =>
        session_partition_law contract view /\
        view_private view = [] /\ view_remaining view = [] /\
        view_continuation view = None /\ view_reason view = None
    | Incomplete partial reason =>
        allowed_incomplete contract ResumableOperation tt reason /\
        match partial with
        | Some view =>
            session_partition_law contract view /\
            view_reason view = Some reason /\
            match view_continuation view with
            | Some continuation =>
                continuation_denotes contract continuation
                  (view_private view ++ view_remaining view)
            | None => terminal_incomplete contract reason view
            end
        | None => True
        end
    | Failed reason => allowed_failure contract ResumableOperation tt reason
    end.

  Definition resource_observation_law (contract : operation_contract)
      (trace : list ResourceEvent * list resource_view) : Prop :=
    resource_accounting contract (fst trace) (snd trace) /\
    Forall (within_limits (limits contract)) (snd trace) /\
    monotone_resource_views (snd trace).

  Definition observation_law (contract : operation_contract)
      (selected : observation_profile) :
      observation_request selected -> observation_payload selected -> Prop :=
    match selected as profile return
      observation_request profile -> observation_payload profile -> Prop with
    | OnlineScore => online_observation_law contract
    | CompletedRange => fun _ => range_observation_law contract
    | CompletedKnn => fun _ => knn_observation_law contract
    | CanonicalWitness => witness_observation_law contract
    | ResumableOperation => fun _ => resume_observation_law contract
    | ResourceSafety => fun _ => resource_observation_law contract
    end.

  Record lawful_reference (contract : operation_contract) : Prop := {
    lawful_range_ideal : ideal_range_law contract;
    lawful_knn_ideal : ideal_knn_law contract;
    lawful_range_rank_irreflexive : forall row,
      common_result_law contract row ->
      ~ range_rank_less contract row row;
    lawful_range_rank_transitive : forall first middle last,
      range_rank_less contract first middle ->
      range_rank_less contract middle last ->
      range_rank_less contract first last;
    lawful_range_rank_total : forall first second,
      common_result_law contract first ->
      common_result_law contract second ->
      scored_original_id first <> scored_original_id second ->
      range_rank_less contract first second \/
      range_rank_less contract second first;
    lawful_knn_rank_irreflexive : forall row,
      common_result_law contract row ->
      ~ knn_rank_less contract row row;
    lawful_knn_rank_transitive : forall first middle last,
      knn_rank_less contract first middle ->
      knn_rank_less contract middle last ->
      knn_rank_less contract first last;
    lawful_knn_rank_total : forall first second,
      common_result_law contract first ->
      common_result_law contract second ->
      scored_original_id first <> scored_original_id second ->
      knn_rank_less contract first second \/
      knn_rank_less contract second first;
    lawful_selected_observations : forall selected request,
      selected_profile contract selected ->
      observation_law contract selected request
        (reference_observe contract selected request)
  }.

  Record exact_realization (contract : operation_contract)
      (implementation : forall selected,
        observation_request selected -> observation_payload selected) : Prop := {
    exact_reference_lawful : lawful_reference contract;
    exact_selected_observations : forall selected request,
      selected_profile contract selected ->
      implementation selected request =
        reference_observe contract selected request
  }.

  Definition lower_simulation (contract : operation_contract)
      (lower : Original -> Cost) : Prop :=
    forall original, knn_eligible contract original ->
      cost_le (operation_measure contract)
        (lower original) (candidate_score contract original).

  Record metric_geometry (spec : measure_specification) : Prop := {
    geometry_nonnegative : forall first second,
      valid_input spec first -> valid_input spec second ->
      cost_le spec (cost_zero spec) (reference_cost spec first second);
    geometry_reflexive : forall input,
      valid_input spec input ->
      cost_equivalent spec (reference_cost spec input input) (cost_zero spec);
    geometry_symmetric : forall first second,
      valid_input spec first -> valid_input spec second ->
      cost_equivalent spec (reference_cost spec first second)
        (reference_cost spec second first);
    geometry_separates : forall first second,
      valid_input spec first -> valid_input spec second ->
      (cost_equivalent spec (reference_cost spec first second) (cost_zero spec)
       <-> quotient_identity spec first second);
    geometry_triangle : forall first middle last,
      valid_input spec first -> valid_input spec middle ->
      valid_input spec last ->
      cost_le spec (reference_cost spec first last)
        (cost_add spec
          (reference_cost spec first middle)
          (reference_cost spec middle last))
  }.

  Definition resource_refinement (contract : operation_contract)
      (implementation : list ResourceEvent * list resource_view) : Prop :=
    resource_observation_law contract implementation.

  Theorem exact_realization_preserves_selected_law :
    forall contract implementation,
      exact_realization contract implementation ->
      forall selected request,
        selected_profile contract selected ->
        observation_law contract selected request
          (implementation selected request).
  Proof.
    intros contract implementation [Hlaw Hequal]
      selected request Hselected.
    rewrite (Hequal selected request Hselected).
    destruct Hlaw as [_ _ _ _ _ _ _ _ Hselected_reference].
    exact (Hselected_reference selected request Hselected).
  Qed.

  Theorem online_observation_cannot_omit_attempted_prefixes :
    forall contract input,
      ~ online_observation_law contract input [].
  Proof.
    intros contract input [Hprefixes _].
    simpl in Hprefixes.
    destruct (attempted_prefixes_well_formed contract input) as [Hnonempty _].
    symmetry in Hprefixes. contradiction.
  Qed.

  Theorem completed_knn_uses_its_own_ranked_universe :
    forall contract rows,
      knn_observation_law contract (Completed rows) ->
      rows = firstn (requested_k contract) (ideal_knn_order contract).
  Proof. intros; exact H. Qed.

  Theorem range_partial_cannot_contain_unjustified_row :
    forall contract rows reason row,
      range_observation_law contract (Incomplete (Some rows) reason) ->
      In row rows -> ~ In row (ideal_range contract) -> False.
  Proof.
    intros contract rows reason row [Hreason [Hsubset _]] Hin Hforeign.
    apply Hforeign. now apply Hsubset.
  Qed.

  Theorem completed_session_cannot_exclude_a_goal_original :
    forall contract view original,
      resume_observation_law contract (Completed view) ->
      In original (view_excluded view) ->
      In original (resume_goal contract) -> False.
  Proof.
    intros contract view original [Hpartition _] Hexcluded Hgoal.
    destruct Hpartition as [_ [_ [_ Hexclusion]]].
    exact (Hexclusion original Hexcluded Hgoal).
  Qed.

End Contract.

(** A one-original, nonempty instance makes the lower/exact distinction
    observable. Its reference score is one; zero is a sound lower filter. *)
Section LowerNotExactControl.

Definition control_spec :
    @measure_specification nat unit unit nat unit.
Proof.
  refine {|
    valid_input := fun _ => True;
    validate_input := fun _ => None;
    quotient_identity := eq;
    fixed_parameters := tt;
    numeric_authority := tt;
    measure := fun _ _ _ _ => 1;
    valid_cost := fun _ => True;
    cost_zero := 0;
    cost_add := Nat.add;
    cost_le := le;
    cost_lt := lt
  |}; intros; simpl in *; try exact I; try reflexivity; try lia.
Defined.

Definition control_row : scored_original nat nat nat :=
  {| scored_original_id := 0; scored_identity := 0; scored_cost := 1 |}.

Definition control_limits : resource_limits :=
  {| cumulative_work_limit := None;
     peak_byte_limit := None;
     page_work_limit := None;
     page_result_limit := None |}.

Definition control_operation :
    @operation_contract nat unit unit nat unit nat nat nat nat
      unit unit unit unit unit.
Proof.
  refine {|
    operation_measure := control_spec;
    fixed_query := 0;
    captured_snapshot := tt;
    snapshot_originals := fun _ => [0];
    original_identity := fun original => original;
    original_input := fun _ => 0;
    eligible_original := fun original => original = 0;
    range_tie := fun original => original;
    range_tie_before := lt;
    knn_tie := fun original => original;
    knn_tie_before := lt;
    inclusive_cutoff := 1;
    requested_k := 1;
    ideal_range := [control_row];
    ideal_knn_order := [control_row];
    attempted_prefixes := fun input => [input];
    committed_prefixes := fun input => [input];
    is_prefix := eq;
    witness_feasible := fun original _ => original = 0;
    witness_cost := fun _ => 1;
    witness_tie_before := fun _ _ => False;
    domain_failure := fun _ => tt;
    allowed_failure := fun _ _ _ => False;
    allowed_incomplete := fun selected _ _ =>
      match selected with CompletedRange => True | _ => False end;
    terminal_incomplete := fun _ _ => False;
    continuation_denotes := fun _ _ => False;
    session_mode := RangeSearch;
    resource_accounting := fun events views => events = [] /\ views = [];
    limits := control_limits;
    selected_profile := fun selected => selected = OnlineScore;
    reference_observe := fun selected =>
      match selected as profile return
        @observation_request nat nat profile ->
        @observation_payload nat nat nat nat unit unit unit unit unit profile with
      | OnlineScore => fun input => [(input, ExactScore 1)]
      | CompletedRange => fun _ => Completed [control_row]
      | CompletedKnn => fun _ => Completed [control_row]
      | CanonicalWitness => fun _ => Completed (Some (control_row, tt))
      | ResumableOperation => fun _ => Failed tt
      | ResourceSafety => fun _ => ([], [])
      end
  |}; simpl; intros; try exact I; try reflexivity; try congruence.
  - repeat constructor; simpl; congruence.
  - repeat split; simpl; try congruence; repeat constructor; simpl; auto.
  - now exists OnlineScore.
Defined.

Lemma control_ideal_range : ideal_range_law control_operation.
Proof.
  unfold ideal_range_law, common_result_law, range_eligible,
    original_of, candidate_score, reference_cost, range_rank_less.
  simpl. repeat split; simpl; try tauto; try lia;
    try (repeat constructor; simpl; auto; congruence).
Qed.

Lemma control_ideal_knn : ideal_knn_law control_operation.
Proof.
  unfold ideal_knn_law, common_result_law, knn_eligible,
    original_of, candidate_score, reference_cost, knn_rank_less.
  simpl. repeat split; simpl; try tauto; try lia;
    try (repeat constructor; simpl; auto; congruence).
Qed.

Lemma control_reference_lawful : lawful_reference control_operation.
Proof.
  constructor.
  - exact control_ideal_range.
  - exact control_ideal_knn.
  - intros row Hrow. unfold range_rank_less; simpl; lia.
  - intros first middle last Hfirst Hsecond.
    unfold range_rank_less in *; simpl in *; lia.
  - intros first second Hfirst Hsecond Hdistinct.
    unfold common_result_law in Hfirst, Hsecond.
    simpl in Hfirst, Hsecond.
    destruct Hfirst as [Hfirst _].
    destruct Hsecond as [Hsecond _].
    simpl in Hfirst, Hsecond.
    lia.
  - intros row Hrow.
    unfold knn_rank_less, cost_equivalent; simpl; lia.
  - intros first middle last Hfirst Hsecond.
    unfold knn_rank_less, cost_equivalent in *; simpl in *; lia.
  - intros first second Hfirst Hsecond Hdistinct.
    unfold common_result_law in Hfirst, Hsecond.
    simpl in Hfirst, Hsecond.
    destruct Hfirst as [Hfirst _].
    destruct Hsecond as [Hsecond _].
    simpl in Hfirst, Hsecond.
    lia.
  - intros selected request Hselected.
    destruct selected; try discriminate.
    unfold observation_law, online_observation_law,
      online_event_law; simpl.
    repeat split; try reflexivity; repeat constructor; simpl; auto.
Qed.

Definition control_lower_implementation :
    forall selected,
      @observation_request nat nat selected ->
      @observation_payload nat nat nat nat unit unit unit unit unit selected :=
  fun selected =>
    match selected as profile return
      @observation_request nat nat profile ->
      @observation_payload nat nat nat nat unit unit unit unit unit profile with
    | OnlineScore => fun input => [(input, ExactScore 0)]
    | CompletedRange => fun _ => Completed [control_row]
    | CompletedKnn => fun _ => Completed [control_row]
    | CanonicalWitness => fun _ => Completed (Some (control_row, tt))
    | ResumableOperation => fun _ => Failed tt
    | ResourceSafety => fun _ => ([], [])
    end.

Lemma control_zero_is_a_sound_lower_simulation :
  lower_simulation control_operation (fun _ => 0).
Proof.
  intros original Heligible.
  unfold candidate_score, reference_cost; simpl; lia.
Qed.

Theorem control_zero_lower_is_not_exact :
  ~ exact_realization control_operation control_lower_implementation.
Proof.
  intros [_ Hequal].
  specialize (Hequal OnlineScore 0 eq_refl).
  simpl in Hequal. discriminate.
Qed.

Definition control_foreign_row : scored_original nat nat nat :=
  {| scored_original_id := 1; scored_identity := 1; scored_cost := 1 |}.

Example foreign_completed_knn_row_is_rejected :
  ~ knn_observation_law control_operation
      (Completed [control_foreign_row]).
Proof. simpl; discriminate. Qed.

Example omitted_online_prefix_is_rejected :
  ~ online_observation_law control_operation 0 [].
Proof. apply online_observation_cannot_omit_attempted_prefixes. Qed.

Definition control_bad_session : continuation_view unit nat unit unit :=
  {| view_snapshot := tt;
     view_published := [];
     view_private := [];
     view_remaining := [];
     view_excluded := [0];
     view_reason := None;
     view_continuation := None |}.

Example unsound_completed_exclusion_is_rejected :
  ~ resume_observation_law control_operation
      (Completed control_bad_session).
Proof.
  intro Haccepted.
  eapply completed_session_cannot_exclude_a_goal_original
    with (original := 0) in Haccepted; simpl in *; auto.
Qed.

Example unjustified_partial_range_row_is_rejected :
  ~ range_observation_law control_operation
      (Incomplete (Some [control_foreign_row]) tt).
Proof.
  intro Haccepted.
  eapply range_partial_cannot_contain_unjustified_row
    with (row := control_foreign_row) in Haccepted; simpl in *; auto.
  intros [Hequal | []]. discriminate.
Qed.

Example valid_partial_range_row_is_accepted :
  range_observation_law control_operation
    (Incomplete (Some [control_row]) tt).
Proof.
  simpl. repeat split; try exact I; try (repeat constructor); auto.
  destruct H as [Hequal | []]. exact Hequal.
Qed.

Fail Check (fun (certificate :
  lower_simulation control_operation (fun _ => 0)) =>
  (certificate :
    exact_realization control_operation control_lower_implementation)).

End LowerNotExactControl.
