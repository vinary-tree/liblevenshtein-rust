(** * Refinements for adaptive ordered residual machines

    This file proves the representation-switching law, a monotone sparse
    lookup refinement, and the semantic/resource separation for bound-first
    candidate evaluation. It contains no axioms or admitted proofs. The
    concrete Rust conversion, scheduler, and ledger correspondences remain
    separate instance obligations. *)

From Stdlib Require Import Arith Bool Lia List.
Import ListNotations.

(** ** R2a: switching representations during a run *)

Section RepresentationSwitching.
  Context {Label Output Dense Sparse : Type}.
  Variable specification : list Label -> Output.
  Variable dense_behavior : Dense -> list Label -> Output.
  Variable sparse_behavior : Sparse -> list Label -> Output.
  Variable dense_step : Dense -> Label -> Dense.
  Variable sparse_step : Sparse -> Label -> Sparse.
  Variable to_dense : Sparse -> Dense.
  Variable to_sparse : Dense -> Sparse.
  Variable dense_seed : Dense.

  Context (seed_correct : forall suffix,
    dense_behavior dense_seed suffix = specification suffix).
  Context (dense_derivative : forall state label suffix,
    dense_behavior (dense_step state label) suffix =
      dense_behavior state (label :: suffix)).
  Context (sparse_derivative : forall state label suffix,
    sparse_behavior (sparse_step state label) suffix =
      sparse_behavior state (label :: suffix)).
  Context (to_dense_correct : forall state suffix,
    dense_behavior (to_dense state) suffix = sparse_behavior state suffix).
  Context (to_sparse_correct : forall state suffix,
    sparse_behavior (to_sparse state) suffix = dense_behavior state suffix).

  Inductive layout_state := DenseState (state : Dense) | SparseState (state : Sparse).
  Inductive event := Consume (label : Label) | UseDense | UseSparse.

  Definition behavior (state : layout_state) : list Label -> Output :=
    match state with
    | DenseState dense => dense_behavior dense
    | SparseState sparse => sparse_behavior sparse
    end.

  Definition advance (state : layout_state) (action : event) : layout_state :=
    match action, state with
    | Consume label, DenseState dense => DenseState (dense_step dense label)
    | Consume label, SparseState sparse => SparseState (sparse_step sparse label)
    | UseDense, DenseState dense => DenseState dense
    | UseDense, SparseState sparse => DenseState (to_dense sparse)
    | UseSparse, SparseState sparse => SparseState sparse
    | UseSparse, DenseState dense => SparseState (to_sparse dense)
    end.

  Definition consumed (action : event) : list Label :=
    match action with Consume label => [label] | _ => [] end.

  Fixpoint execute (state : layout_state) (actions : list event) : layout_state :=
    match actions with
    | [] => state
    | action :: tail => execute (advance state action) tail
    end.

  Fixpoint word (actions : list event) : list Label :=
    match actions with
    | [] => []
    | action :: tail => consumed action ++ word tail
    end.

  Lemma advance_preserves_residual : forall state action suffix,
    behavior (advance state action) suffix =
      behavior state (consumed action ++ suffix).
  Proof.
    intros [dense | sparse] [label | |] suffix; simpl;
      try apply dense_derivative; try apply sparse_derivative;
      try apply to_dense_correct; try apply to_sparse_correct;
      reflexivity.
  Qed.

  Theorem arbitrary_adaptive_switches_preserve_score : forall actions suffix,
    behavior (execute (DenseState dense_seed) actions) suffix =
      specification (word actions ++ suffix).
  Proof.
    assert (general : forall state actions suffix,
      behavior (execute state actions) suffix =
        behavior state (word actions ++ suffix)).
    { intros state actions; revert state.
      induction actions as [| action tail IH]; intros state suffix; simpl.
      - reflexivity.
      - rewrite IH, advance_preserves_residual.
        now rewrite app_assoc. }
    intros actions suffix.
    rewrite general. apply seed_correct.
  Qed.
End RepresentationSwitching.

(** ** CL-1: cursor lookup on sorted sparse positions *)

Section MonotoneCursor.
  Variable Value : Type.

  Fixpoint strictly_ascending (source : list (nat * Value)) : Prop :=
    match source with
    | [] => True
    | (row, _) :: tail =>
        Forall (fun entry => row < fst entry) tail /\ strictly_ascending tail
    end.

  Fixpoint drop_before (requested : nat) (source : list (nat * Value))
      : list (nat * Value) :=
    match source with
    | [] => []
    | (row, value) :: tail =>
        if row <? requested then drop_before requested tail
        else (row, value) :: tail
    end.

  Fixpoint binary_lookup (requested : nat) (source : list (nat * Value))
      : option Value :=
    match source with
    | [] => None
    | (row, value) :: tail =>
        if row =? requested then Some value else binary_lookup requested tail
    end.

  Definition cursor_lookup (requested : nat) (source : list (nat * Value))
      : option Value :=
    match drop_before requested source with
    | [] => None
    | (row, value) :: _ => if row =? requested then Some value else None
    end.

  Lemma drop_before_preserves_later_lookup : forall source low high,
    low <= high ->
    binary_lookup high (drop_before low source) = binary_lookup high source.
  Proof.
    induction source as [| [row value] tail IH]; intros low high Horder;
      simpl; [reflexivity |].
    destruct (row <? low) eqn:Hrow.
    - apply Nat.ltb_lt in Hrow.
      assert (Hneq : (row =? high) = false) by (apply Nat.eqb_neq; lia).
      rewrite Hneq. now apply IH.
    - reflexivity.
  Qed.

  Lemma ascending_tail : forall row value tail,
    strictly_ascending ((row, value) :: tail) -> strictly_ascending tail.
  Proof. intros row value tail [_ Htail]; exact Htail. Qed.

  Lemma drop_before_ascending : forall source requested,
    strictly_ascending source ->
    strictly_ascending (drop_before requested source).
  Proof.
    induction source as [| [row value] tail IH]; intros requested Hsorted;
      simpl; [exact I |].
    destruct (row <? requested).
    - apply IH. now apply ascending_tail in Hsorted.
    - exact Hsorted.
  Qed.

  Lemma drop_before_head_ge : forall source requested row value tail,
    drop_before requested source = (row, value) :: tail ->
    requested <= row.
  Proof.
    induction source as [| [first first_value] rest IH];
      intros requested row value tail Hdrop; simpl in Hdrop;
      [discriminate |].
    destruct (first <? requested) eqn:Hfirst.
    - now apply IH in Hdrop.
    - inversion Hdrop; subst. now apply Nat.ltb_ge in Hfirst.
  Qed.

  Lemma cursor_lookup_equals_binary_lookup : forall source requested,
    strictly_ascending source ->
    cursor_lookup requested source = binary_lookup requested source.
  Proof.
    intros source requested Hsorted.
    unfold cursor_lookup.
    rewrite <- (drop_before_preserves_later_lookup source requested requested)
      by lia.
    remember (drop_before requested source) as remaining eqn:Hremaining.
    pose proof (drop_before_ascending source requested Hsorted) as Hsuffix.
    rewrite <- Hremaining in Hsuffix.
    destruct remaining as [| [row value] tail]; simpl; [reflexivity |].
    assert (Hge : requested <= row).
    { eapply drop_before_head_ge. symmetry; exact Hremaining. }
    destruct (row =? requested) eqn:Heq; [reflexivity |].
    destruct Hsuffix as [Hall _].
    clear Hremaining Hsorted source.
    induction tail as [| [next next_value] rest IH]; simpl; [reflexivity |].
    inversion Hall as [| ? ? Hgt Hrest]; subst.
    assert (Hneq : (next =? requested) = false)
      by (apply Nat.eqb_neq; simpl in Hgt; lia).
    rewrite Hneq.
    apply IH. exact Hrest.
  Qed.

  Fixpoint ordered_requests (requests : list nat) : Prop :=
    match requests with
    | [] => True
    | requested :: tail =>
        Forall (fun later => requested <= later) tail /\ ordered_requests tail
    end.

  Definition requests_for_row (row : nat) : list nat :=
    match row with
    | 0 => [0]
    | S previous => [previous; S previous]
    end.

  Fixpoint evaluated_requests (rows : list nat) : list nat :=
    match rows with
    | [] => []
    | row :: tail => requests_for_row row ++ evaluated_requests tail
    end.

  Fixpoint strictly_ascending_rows (rows : list nat) : Prop :=
    match rows with
    | [] => True
    | row :: tail =>
        Forall (fun later => row < later) tail /\
        strictly_ascending_rows tail
    end.

  Lemma requests_for_row_ordered : forall row,
    ordered_requests (requests_for_row row).
  Proof. intros [|previous]; simpl; repeat split; auto with arith. Qed.

  Lemma requests_for_row_at_most_row : forall row request,
    In request (requests_for_row row) -> request <= row.
  Proof.
    intros [|previous] request Hin; simpl in Hin.
    - destruct Hin as [Heq | []]; subst; lia.
    - destruct Hin as [Heq | [Heq | []]]; subst; lia.
  Qed.

  Lemma evaluated_requests_after_row : forall lower rows,
    Forall (fun row => lower < row) rows ->
    Forall (fun request => lower <= request) (evaluated_requests rows).
  Proof.
    intros lower rows Hall.
    induction Hall as [| row tail Hgreater Hrest IH]; simpl;
      [constructor |].
    apply Forall_app; split; [| exact IH].
    destruct row as [|previous]; [lia |].
    simpl. constructor; [lia |]. constructor; [lia | constructor].
  Qed.

  Lemma ordered_requests_append : forall left right,
    ordered_requests left -> ordered_requests right ->
    (forall x y, In x left -> In y right -> x <= y) ->
    ordered_requests (left ++ right).
  Proof.
    induction left as [| head tail IH]; intros right Hleft Hright Hcross;
      simpl in *; [exact Hright |].
    destruct Hleft as [Hhead Htail].
    split.
    - apply Forall_app; split; [exact Hhead |].
      apply Forall_forall; intros y Hy.
      apply Hcross with (x := head); [now left | exact Hy].
    - apply IH; [exact Htail | exact Hright |].
      intros x y Hx Hy.
      apply Hcross; [now right | exact Hy].
  Qed.

  Theorem ascending_evaluated_rows_generate_ordered_requests : forall rows,
    strictly_ascending_rows rows ->
    ordered_requests (evaluated_requests rows).
  Proof.
    induction rows as [| row tail IH]; intro Hrows; simpl;
      [exact I |].
    destruct Hrows as [Hall Htail].
    apply ordered_requests_append.
    - apply requests_for_row_ordered.
    - now apply IH.
    - intros x y Hx Hy.
      pose proof (requests_for_row_at_most_row row x Hx) as Hxle.
      pose proof (evaluated_requests_after_row row tail Hall) as Hrequests.
      apply Forall_forall with (x := y) in Hrequests; [lia | exact Hy].
  Qed.

  Definition next_merged_row
      (scheduled vertical : option nat) : option nat :=
    match scheduled, vertical with
    | None, other | other, None => other
    | Some first_row, Some second_row => Some (Nat.min first_row second_row)
    end.

  Lemma scheduled_vertical_merge_advances : forall last scheduled vertical next,
    (forall row, scheduled = Some row -> last < row) ->
    (forall row, vertical = Some row -> row = S last) ->
    next_merged_row scheduled vertical = Some next ->
    last < next.
  Proof.
    intros last [scheduled |] [vertical |] next Hscheduled Hvertical Hnext;
      simpl in Hnext.
    - inversion Hnext; subst.
      specialize (Hscheduled scheduled eq_refl).
      specialize (Hvertical vertical eq_refl).
      lia.
    - inversion Hnext; subst.
      now apply (Hscheduled next eq_refl).
    - inversion Hnext; subst.
      specialize (Hvertical next eq_refl). lia.
    - discriminate.
  Qed.

  Fixpoint cursor_answers
      (source : list (nat * Value)) (requests : list nat)
      : list (option Value) :=
    match requests with
    | [] => []
    | requested :: tail =>
        cursor_lookup requested source ::
        cursor_answers (drop_before requested source) tail
    end.

  Theorem ordered_cursor_matches_independent_lookups : forall source requests,
    strictly_ascending source -> ordered_requests requests ->
    cursor_answers source requests =
      map (fun requested => binary_lookup requested source) requests.
  Proof.
    intros source requests Hsorted.
    revert source Hsorted.
    induction requests as [| requested tail IH];
      intros source Hsorted Hordered; simpl; [reflexivity |].
    destruct Hordered as [Hall Htail].
    rewrite (cursor_lookup_equals_binary_lookup source requested Hsorted).
    f_equal.
    rewrite (IH (drop_before requested source)).
    2: { now apply drop_before_ascending. }
    2: { exact Htail. }
    apply map_ext_in.
    intros later Hin.
    apply Forall_forall with (x := later) in Hall; [| exact Hin].
    now apply drop_before_preserves_later_lookup.
  Qed.

  Fixpoint drop_count (requested : nat) (source : list (nat * Value)) : nat :=
    match source with
    | [] => 0
    | (row, _) :: tail =>
        if row <? requested then S (drop_count requested tail) else 0
    end.

  Lemma drop_count_partition : forall source requested,
    drop_count requested source + length (drop_before requested source) =
      length source.
  Proof.
    induction source as [| [row value] tail IH]; intros requested;
      simpl; [reflexivity |].
    destruct (row <? requested); simpl; rewrite ?IH; lia.
  Qed.

  Fixpoint cursor_comparisons
      (source : list (nat * Value)) (requests : list nat) : nat :=
    match requests with
    | [] => 0
    | requested :: tail =>
        S (drop_count requested source) +
        cursor_comparisons (drop_before requested source) tail
    end.

  Lemma cursor_comparisons_telescopes : forall source requests,
    cursor_comparisons source requests +
      length (fold_left (fun cursor requested => drop_before requested cursor)
        requests source) = length source + length requests.
  Proof.
    intros source requests; revert source.
    induction requests as [| requested tail IH]; intros source; simpl;
      [lia |].
    pose proof (IH (drop_before requested source)).
    pose proof (drop_count_partition source requested).
    lia.
  Qed.

  Corollary cursor_comparisons_linear : forall source requests,
    cursor_comparisons source requests <= length source + length requests.
  Proof.
    intros source requests.
    pose proof (cursor_comparisons_telescopes source requests).
    lia.
  Qed.
End MonotoneCursor.

(** ** RA-1: bound-first evaluation and transactional charging *)

Record candidate := {
  exact_cost : nat;
  lower_cost : nat;
  bound_work : nat;
  exact_work : nat
}.

Definition admissible (entry : candidate) : Prop :=
  lower_cost entry <= exact_cost entry.

Definition exact_accepts (cutoff : nat) (entry : candidate) : bool :=
  exact_cost entry <=? cutoff.

Definition bound_first_accepts (cutoff : nat) (entry : candidate) : bool :=
  if cutoff <? lower_cost entry then false else exact_accepts cutoff entry.

Theorem admissible_bound_first_preserves_exact_result : forall cutoff entry,
  admissible entry ->
  bound_first_accepts cutoff entry = exact_accepts cutoff entry.
Proof.
  intros cutoff entry Hadmissible.
  unfold bound_first_accepts, exact_accepts, admissible in *.
  destruct (cutoff <? lower_cost entry) eqn:Hreject; [| reflexivity].
  apply Nat.ltb_lt in Hreject.
  symmetry. apply Nat.leb_gt. lia.
Qed.

Corollary admissible_bound_first_preserves_completed_search : forall
    cutoff entries,
  Forall admissible entries ->
  filter (bound_first_accepts cutoff) entries =
    filter (exact_accepts cutoff) entries.
Proof.
  intros cutoff entries Hall.
  induction Hall as [| entry tail Hadmissible Htail IH]; simpl;
    [reflexivity |].
  now rewrite (admissible_bound_first_preserves_exact_result
    cutoff entry Hadmissible), IH.
Qed.

Inductive phase := NeedBound | NeedExact | Finished.

Record candidate_state := {
  current_phase : phase;
  charged_work : nat;
  emitted : bool
}.

Definition step_candidate
    (cutoff limit : nat) (entry : candidate) (state : candidate_state)
    : candidate_state :=
  match current_phase state with
  | NeedBound =>
      if charged_work state + bound_work entry <=? limit then
        if cutoff <? lower_cost entry then
          {| current_phase := Finished;
             charged_work := charged_work state + bound_work entry;
             emitted := emitted state |}
        else
          {| current_phase := NeedExact;
             charged_work := charged_work state + bound_work entry;
             emitted := emitted state |}
      else state
  | NeedExact =>
      if charged_work state + exact_work entry <=? limit then
        {| current_phase := Finished;
           charged_work := charged_work state + exact_work entry;
           emitted := emitted state || exact_accepts cutoff entry |}
      else state
  | Finished => state
  end.

Theorem rejected_preflight_is_atomic : forall cutoff limit entry state,
  current_phase state <> Finished ->
  (match current_phase state with
   | NeedBound => limit < charged_work state + bound_work entry
   | NeedExact => limit < charged_work state + exact_work entry
   | Finished => False
   end) ->
  step_candidate cutoff limit entry state = state.
Proof.
  intros cutoff limit entry [phase charged emitted] Hphase Hlimit;
    destruct phase; simpl in *; try contradiction;
    unfold step_candidate; simpl.
  - assert (Hfalse : (charged + bound_work entry <=? limit) = false)
      by (apply Nat.leb_gt; lia).
    now rewrite Hfalse.
  - assert (Hfalse : (charged + exact_work entry <=? limit) = false)
      by (apply Nat.leb_gt; lia).
    now rewrite Hfalse.
Qed.

Theorem accepted_step_never_exceeds_limit : forall cutoff limit entry state,
  charged_work state <= limit ->
  charged_work (step_candidate cutoff limit entry state) <= limit.
Proof.
  intros cutoff limit entry [phase charged emitted] Hcharged.
  unfold step_candidate; destruct phase; simpl in *.
  - destruct (charged + bound_work entry <=? limit) eqn:Hbound.
    + apply Nat.leb_le in Hbound.
      destruct (cutoff <? lower_cost entry); simpl; lia.
    + simpl; lia.
  - destruct (charged + exact_work entry <=? limit) eqn:Hexact.
    + apply Nat.leb_le in Hexact. simpl; lia.
    + simpl; lia.
  - lia.
Qed.

Theorem completed_candidate_matches_exact : forall cutoff limit entry charged,
  admissible entry ->
  charged + bound_work entry + exact_work entry <= limit ->
  let initial := {| current_phase := NeedBound;
                    charged_work := charged; emitted := false |} in
  let final := step_candidate cutoff limit entry
                 (step_candidate cutoff limit entry initial) in
  current_phase final = Finished /\
  emitted final = exact_accepts cutoff entry.
Proof.
  intros cutoff limit entry charged Hadmissible Hlimit; simpl.
  unfold step_candidate; simpl.
  assert (Hbound : charged + bound_work entry <= limit) by lia.
  assert (Hboundb : (charged + bound_work entry <=? limit) = true)
    by (apply Nat.leb_le; exact Hbound).
  rewrite Hboundb.
  destruct (cutoff <? lower_cost entry) eqn:Hreject; simpl.
  - split; [reflexivity |].
    unfold exact_accepts, admissible in *.
    apply Nat.ltb_lt in Hreject.
    symmetry. apply Nat.leb_gt; lia.
  - assert (Hexactb : (charged + bound_work entry + exact_work entry <=? limit) = true)
      by (apply Nat.leb_le; exact Hlimit).
    rewrite Hexactb.
    simpl. now split.
Qed.

Theorem paused_exact_phase_resumes_without_repeating_bound : forall
    cutoff entry small_limit large_limit,
  lower_cost entry <= cutoff ->
  bound_work entry <= small_limit ->
  small_limit < bound_work entry + exact_work entry ->
  bound_work entry + exact_work entry <= large_limit ->
  let initial := {| current_phase := NeedBound;
                    charged_work := 0; emitted := false |} in
  let after_bound := step_candidate cutoff small_limit entry initial in
  current_phase after_bound = NeedExact /\
  charged_work after_bound = bound_work entry /\
  step_candidate cutoff small_limit entry after_bound = after_bound /\
  step_candidate cutoff large_limit entry after_bound =
    step_candidate cutoff large_limit entry
      (step_candidate cutoff large_limit entry initial).
Proof.
  intros cutoff entry small_limit large_limit Hcandidate Hbound Hpause Hlarge.
  assert (Hboundb : (bound_work entry <=? small_limit) = true)
    by (apply Nat.leb_le; exact Hbound).
  assert (Hlargeb : (bound_work entry <=? large_limit) = true)
    by (apply Nat.leb_le; lia).
  assert (Hreject : (cutoff <? lower_cost entry) = false)
    by (apply Nat.ltb_ge; exact Hcandidate).
  assert (Hpauseb : (bound_work entry + exact_work entry <=? small_limit) = false)
    by (apply Nat.leb_gt; exact Hpause).
  assert (Hexactb : (bound_work entry + exact_work entry <=? large_limit) = true)
    by (apply Nat.leb_le; exact Hlarge).
  unfold step_candidate; simpl.
  rewrite Hboundb, Hreject; simpl.
  rewrite Hpauseb, Hlargeb; simpl.
  rewrite Hexactb; simpl.
  repeat split; reflexivity.
Qed.

Definition legacy_full_precharge
    (limit : nat) (entry : candidate) : bool :=
  bound_work entry + exact_work entry <=? limit.

Example a_smaller_budget_changes_the_resource_outcome :
  let entry := {| exact_cost := 8; lower_cost := 8;
                  bound_work := 1; exact_work := 100 |} in
  current_phase (step_candidate 5 1 entry
    {| current_phase := NeedBound; charged_work := 0; emitted := false |})
    = Finished /\ legacy_full_precharge 1 entry = false.
Proof. now split. Qed.

(** ** RA-2: charge each sparse edge primitive before private execution *)

Record edge_operation := {
  operation_work : nat;
  operation_delta : nat
}.

Fixpoint total_work (operations : list edge_operation) : nat :=
  match operations with
  | [] => 0
  | operation :: tail => operation_work operation + total_work tail
  end.

Fixpoint total_delta (operations : list edge_operation) : nat :=
  match operations with
  | [] => 0
  | operation :: tail => operation_delta operation + total_delta tail
  end.

Lemma total_work_append : forall first second,
  total_work (first ++ second) = total_work first + total_work second.
Proof.
  induction first as [|operation tail IH]; intros second; simpl;
    [lia | rewrite IH; lia].
Qed.

Lemma total_delta_append : forall first second,
  total_delta (first ++ second) = total_delta first + total_delta second.
Proof.
  induction first as [|operation tail IH]; intros second; simpl;
    [lia | rewrite IH; lia].
Qed.

Record edge_state := {
  published_value : nat;
  private_value : nat;
  remaining_operations : list edge_operation;
  edge_charged : nat;
  edge_complete : bool
}.

Definition begin_edge (initial : nat) (operations : list edge_operation)
    : edge_state :=
  {| published_value := initial;
     private_value := initial;
     remaining_operations := operations;
     edge_charged := 0;
     edge_complete := false |}.

Definition advance_edge (limit : nat) (state : edge_state) : edge_state :=
  if edge_complete state then state else
  match remaining_operations state with
  | [] =>
      {| published_value := private_value state;
         private_value := private_value state;
         remaining_operations := [];
         edge_charged := edge_charged state;
         edge_complete := true |}
  | operation :: tail =>
      if edge_charged state + operation_work operation <=? limit then
        {| published_value := published_value state;
           private_value := private_value state + operation_delta operation;
           remaining_operations := tail;
           edge_charged := edge_charged state + operation_work operation;
           edge_complete := false |}
      else state
  end.

Definition edge_refines
    (initial : nat) (all : list edge_operation) (state : edge_state) : Prop :=
  if edge_complete state then
    remaining_operations state = [] /\
    published_value state = initial + total_delta all /\
    edge_charged state = total_work all
  else
    exists processed,
      all = processed ++ remaining_operations state /\
      private_value state = initial + total_delta processed /\
      published_value state = initial /\
      edge_charged state = total_work processed.

Theorem fresh_edge_refines_pure_execution : forall initial all,
  edge_refines initial all (begin_edge initial all).
Proof.
  intros initial all; unfold edge_refines, begin_edge; simpl.
  exists []; simpl; repeat split; lia.
Qed.

Theorem edge_preflight_failure_keeps_every_field : forall limit state operation tail,
  edge_complete state = false ->
  remaining_operations state = operation :: tail ->
  limit < edge_charged state + operation_work operation ->
  advance_edge limit state = state.
Proof.
  intros limit state operation tail Hactive Hremaining Hlimit.
  unfold advance_edge; rewrite Hactive, Hremaining.
  assert (Hreject :
    (edge_charged state + operation_work operation <=? limit) = false)
    by (apply Nat.leb_gt; exact Hlimit).
  now rewrite Hreject.
Qed.

Theorem private_edge_step_keeps_published_value : forall limit state,
  edge_complete state = false ->
  remaining_operations state <> [] ->
  published_value (advance_edge limit state) = published_value state.
Proof.
  intros limit state Hactive Hremaining.
  unfold advance_edge; rewrite Hactive.
  destruct (remaining_operations state) as [|operation tail] eqn:Hops;
    [contradiction |].
  destruct (edge_charged state + operation_work operation <=? limit);
    reflexivity.
Qed.

Theorem edge_step_preserves_pure_refinement : forall initial all limit state,
  edge_refines initial all state ->
  edge_refines initial all (advance_edge limit state).
Proof.
  intros initial all limit state Hrefine.
  unfold edge_refines in *.
  destruct (edge_complete state) eqn:Hcomplete.
  - unfold advance_edge; rewrite Hcomplete, Hcomplete.
    exact Hrefine.
  - destruct Hrefine as [processed [Hall [Hprivate [Hpublished Hcharged]]]].
    unfold advance_edge; rewrite Hcomplete.
    destruct (remaining_operations state) as [|operation tail] eqn:Hremaining.
    + simpl. split; [reflexivity |]. split.
      * rewrite Hprivate.
        subst all.
        now rewrite app_nil_r.
      * rewrite Hcharged.
        subst all.
        now rewrite app_nil_r.
    + destruct (edge_charged state + operation_work operation <=? limit)
        eqn:Haffordable.
      * simpl. exists (processed ++ [operation]).
        repeat split.
        -- rewrite <- app_assoc; simpl. exact Hall.
        -- rewrite Hprivate, total_delta_append; simpl; lia.
        -- exact Hpublished.
        -- rewrite Hcharged, total_work_append; simpl; lia.
      * rewrite Hcomplete. exists processed.
        split; [now rewrite Hremaining |].
        repeat split; assumption.
Qed.

Corollary completed_edge_publishes_exactly_the_pure_result : forall
    initial all state,
  edge_refines initial all state ->
  edge_complete state = true ->
  published_value state = initial + total_delta all /\
  edge_charged state = total_work all.
Proof.
  intros initial all state Hrefine Hcomplete.
  unfold edge_refines in Hrefine; now rewrite Hcomplete in Hrefine.
Qed.

Fixpoint run_edge_pages (limits : list nat) (state : edge_state) : edge_state :=
  match limits with
  | [] => state
  | limit :: tail => run_edge_pages tail (advance_edge limit state)
  end.

Theorem arbitrary_page_boundaries_preserve_private_edge_refinement : forall
    initial all limits,
  edge_refines initial all
    (run_edge_pages limits (begin_edge initial all)).
Proof.
  intros initial all limits.
  assert (general : forall state pages,
    edge_refines initial all state ->
    edge_refines initial all (run_edge_pages pages state)).
  { intros state pages; revert state.
    induction pages as [|limit tail IH]; intros state Hstate; simpl;
      [exact Hstate |].
    apply IH. now apply edge_step_preserves_pure_refinement. }
  apply general. apply fresh_edge_refines_pure_execution.
Qed.

Corollary completed_paged_edge_equals_uninterrupted_pure_edge : forall
    initial all limits,
  edge_complete (run_edge_pages limits (begin_edge initial all)) = true ->
  published_value (run_edge_pages limits (begin_edge initial all)) =
    initial + total_delta all /\
  edge_charged (run_edge_pages limits (begin_edge initial all)) =
    total_work all.
Proof.
  intros initial all limits Hcomplete.
  eapply completed_edge_publishes_exactly_the_pure_result;
    [apply arbitrary_page_boundaries_preserve_private_edge_refinement |
     exact Hcomplete].
Qed.
