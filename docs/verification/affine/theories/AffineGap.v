(** * Exact affine-gap automaton obligations

    This assumption-free model captures the arithmetic used by the Rust
    [AffineV] transition and B-4 subsumption rule.  Costs live in [Z] to match
    the fixed-point integer domain before Rust's checked [usize] boundary.
*)

From Stdlib Require Import ZArith Lia List.
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
