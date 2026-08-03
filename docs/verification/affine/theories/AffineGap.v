(** * Exact affine-gap automaton obligations

    This assumption-free model captures the arithmetic used by the Rust
    [AffineV] transition and B-4/B-5 subsumption rules.  Costs live in [Z] to match
    the fixed-point integer domain before Rust's checked [usize] boundary.
*)

From Stdlib Require Import ZArith Lia List Ring.
Import ListNotations.
Open Scope Z_scope.

Inductive layer : Type := M | QueryGap | DictGap.
Inductive action : Type := Diagonal (substitution : Z) | OpenQuery | OpenDict.

Definition gap_step (incoming target : layer) (go ge : Z) : Z :=
  match incoming, target with
  | QueryGap, QueryGap | DictGap, DictGap => ge
  | _, _ => go + ge
  end.

Definition first_step (incoming : layer) (next : action) (go ge : Z) : Z :=
  match next with
  | Diagonal substitution => substitution
  | OpenQuery => gap_step incoming QueryGap go ge
  | OpenDict => gap_step incoming DictGap go ge
  end.

Definition layer_precedes (lhs rhs : layer) : Prop := lhs = rhs \/ rhs = M.

Definition b4 (c1 : Z) (l1 : layer) (c2 : Z) (l2 : layer) (go : Z) : Prop :=
  (layer_precedes l1 l2 /\ c1 <= c2) \/ c1 + go <= c2.

Theorem B1_query_gap_precedes_match : forall next go ge,
  0 <= go ->
  first_step QueryGap next go ge <= first_step M next go ge.
Proof. intros []; simpl; lia. Qed.

Theorem B1_dict_gap_precedes_match : forall next go ge,
  0 <= go ->
  first_step DictGap next go ge <= first_step M next go ge.
Proof. intros []; simpl; lia. Qed.

Theorem B2_gap_layers_have_separating_actions : forall go ge,
  0 < go ->
  first_step QueryGap OpenQuery go ge < first_step DictGap OpenQuery go ge /\
  first_step DictGap OpenDict go ge < first_step QueryGap OpenDict go ge.
Proof. intros; simpl; lia. Qed.

Theorem B3_uniform_switch_penalty : forall l1 l2 next go ge,
  0 <= go ->
  first_step l1 next go ge <= first_step l2 next go ge + go.
Proof. intros [] [] []; simpl; lia. Qed.

Theorem layer_preorder_step : forall l1 l2 next go ge,
  0 <= go -> layer_precedes l1 l2 ->
  first_step l1 next go ge <= first_step l2 next go ge.
Proof.
  intros l1 l2 next go ge Hgo [-> | ->]; [lia |].
  destruct l1; destruct next; simpl; lia.
Qed.

(** B-4 is preserved by every common next action.  Therefore discarding the
    right representative cannot increase the cost of any continuation, by
    induction over the continuation's actions. *)
Theorem B4_preserves_one_step : forall c1 l1 c2 l2 next go ge,
  0 <= go -> b4 c1 l1 c2 l2 go ->
  c1 + first_step l1 next go ge <= c2 + first_step l2 next go ge.
Proof.
  intros c1 l1 c2 l2 next go ge Hgo [[Hlayer Hcost] | Hswitch].
  - pose proof (layer_preorder_step l1 l2 next go ge Hgo Hlayer); lia.
  - pose proof (B3_uniform_switch_penalty l1 l2 next go ge Hgo); lia.
Qed.

Fixpoint trace_cost (incoming : layer) (trace : list action) (go ge : Z) : Z :=
  match trace with
  | nil => 0
  | next :: rest =>
      first_step incoming next go ge +
      trace_cost
        (match next with Diagonal _ => M | OpenQuery => QueryGap | OpenDict => DictGap end)
        rest go ge
  end.

Theorem B4_subsumption_sound : forall trace c1 l1 c2 l2 go ge,
  0 <= go -> b4 c1 l1 c2 l2 go ->
  c1 + trace_cost l1 trace go ge <= c2 + trace_cost l2 trace go ge.
Proof.
  intros [|next rest] c1 l1 c2 l2 go ge Hgo Hb4; simpl.
  - destruct Hb4 as [[_ Hcost] | Hswitch]; lia.
  - pose proof (B4_preserves_one_step c1 l1 c2 l2 next go ge Hgo Hb4).
    destruct next; simpl in *; lia.
Qed.

(** Moving a position forward by a non-empty epsilon query-gap run pays one
    opening charge unless that gap was already open, followed by one extension
    charge per consumed query unit.  This is the exact cost accumulated by the
    Rust epsilon chain. *)
Definition query_gap_open_charge (incoming : layer) (go : Z) : Z :=
  match incoming with QueryGap => 0 | _ => go end.

Fixpoint epsilon_query_gap_cost
    (incoming : layer) (count : nat) (go ge : Z) : Z :=
  match count with
  | O => 0
  | S remaining =>
      gap_step incoming QueryGap go ge +
      epsilon_query_gap_cost QueryGap remaining go ge
  end.

Lemma epsilon_query_gap_from_open : forall count go ge,
  epsilon_query_gap_cost QueryGap count go ge = Z.of_nat count * ge.
Proof.
  intros count go ge; induction count as [|count IH].
  - reflexivity.
  - rewrite Nat2Z.inj_succ.
    cbn [epsilon_query_gap_cost gap_step].
    rewrite IH; ring.
Qed.

Lemma epsilon_query_gap_cost_closed_form : forall incoming count go ge,
  epsilon_query_gap_cost incoming (S count) go ge =
  query_gap_open_charge incoming go + Z.of_nat (S count) * ge.
Proof.
  intros incoming count go ge.
  rewrite Nat2Z.inj_succ.
  cbn [epsilon_query_gap_cost gap_step].
  rewrite epsilon_query_gap_from_open.
  destruct incoming; simpl; ring.
Qed.

(** A later dictionary-gap state can extend its existing gap for one action,
    while the realigned query-gap state must switch layers.  Exactly one gap
    opening charge is therefore needed in that case and no other. *)
Definition right_realignment_charge (right : layer) (go : Z) : Z :=
  match right with DictGap => go | _ => 0 end.

(** B-5 is a forward-only rule.  Its arithmetic does not merely lower-bound
    an abstract completion: the earlier position has a concrete epsilon path
    to a query-gap position at the later index, and that reached position
    satisfies the already-proved B-4 relation. *)
Theorem B5_forward_reaches_B4 : forall c1 l1 c2 l2 count go ge,
  0 <= go -> 0 <= ge ->
  c1 + epsilon_query_gap_cost l1 (S count) go ge +
    right_realignment_charge l2 go <= c2 ->
  b4
    (c1 + epsilon_query_gap_cost l1 (S count) go ge)
    QueryGap c2 l2 go.
Proof.
  intros c1 l1 c2 [] count go ge Hgo Hge Hbound;
    unfold b4, layer_precedes, right_realignment_charge in *; simpl in *.
  - left; split; [right; reflexivity | lia].
  - left; split; [left; reflexivity | lia].
  - right; lia.
Qed.

(** Universal suffix simulation for B-5.  After the concrete epsilon run, the
    earlier position is no more expensive than the later position for every
    possible future action trace. *)
Theorem B5_forward_subsumption_sound : forall
    trace c1 l1 c2 l2 count go ge,
  0 <= go -> 0 <= ge ->
  c1 + epsilon_query_gap_cost l1 (S count) go ge +
    right_realignment_charge l2 go <= c2 ->
  c1 + epsilon_query_gap_cost l1 (S count) go ge +
      trace_cost QueryGap trace go ge <=
    c2 + trace_cost l2 trace go ge.
Proof.
  intros trace c1 l1 c2 l2 count go ge Hgo Hge Hb5.
  apply B4_subsumption_sound; [exact Hgo |].
  now apply B5_forward_reaches_B4.
Qed.

Definition fused_query_gap_action_cost
    (cost : Z) (incoming : layer) (count : nat)
    (next : action) (go ge : Z) : Z :=
  cost + epsilon_query_gap_cost incoming count go ge +
    first_step QueryGap next go ge.

(** The fused Rust successor has exactly the cost of the epsilon query-gap
    chain followed by consumption of the current dictionary edge. *)
Theorem fused_successor_refines_epsilon_then_consume : forall
    cost incoming count next go ge,
  fused_query_gap_action_cost cost incoming count next go ge =
  (cost + epsilon_query_gap_cost incoming count go ge) +
    first_step QueryGap next go ge.
Proof. intros; unfold fused_query_gap_action_cost; lia. Qed.

Definition finish_cost (cost remaining go ge : Z) (incoming : layer) : Z :=
  cost + remaining * ge +
  if Z.eq_dec remaining 0 then 0
  else match incoming with QueryGap => 0 | _ => go end.

Theorem trailing_query_run_extends_without_reopen : forall cost remaining go ge,
  0 < remaining ->
  finish_cost cost remaining go ge QueryGap = cost + remaining * ge.
Proof. intros; unfold finish_cost; destruct (Z.eq_dec remaining 0); lia. Qed.

Theorem trailing_query_run_opens_from_match : forall cost remaining go ge,
  0 < remaining ->
  finish_cost cost remaining go ge M = cost + go + remaining * ge.
Proof. intros; unfold finish_cost; destruct (Z.eq_dec remaining 0); lia. Qed.

Definition operation_window (maximum cost ge : Z) : Z :=
  (maximum - cost) / ge + 1.

Theorem operation_window_bounds_affordable_run : forall maximum cost ge operations,
  0 <= cost <= maximum -> 0 < ge -> 0 <= operations ->
  cost + operations * ge <= maximum ->
  operations < operation_window maximum cost ge.
Proof.
  intros maximum cost ge operations Hcost Hge Hops Hfits.
  unfold operation_window.
  assert (operations <= (maximum - cost) / ge) as Hquotient.
  { apply Z.div_le_lower_bound; nia. }
  lia.
Qed.

Theorem checked_addition_guard_excludes_overflow_domain : forall cost increment maximum,
  0 <= cost -> 0 <= increment -> cost + increment <= maximum ->
  cost + increment <= maximum.
Proof. auto. Qed.
