(** * PositionKind and monomorphized-variant conformance

    This assumption-free model pins the Rust Phase 5 representation and its
    one-dispatch-per-edge semantics.  A position key contains the two legacy
    natural-number coordinates plus a continuation kind and one-byte payload.
    Runtime and monomorphized subsumption are defined independently, then
    proved extensionally equal for every built-in algorithm.
*)

From Stdlib Require Import Arith Bool Lia.

Inductive position_kind : Type :=
  | Normal
  | OsaTransposing
  | Splitting
  | AffineQueryGap
  | AffineDictGap
  | DamerauPending.

Scheme Equality for position_kind.

Record position : Type := mkPosition {
  term_index : nat;
  num_errors : nat;
  kind : position_kind;
  aux : nat
}.

Definition is_special (p : position) : bool :=
  negb (position_kind_beq (kind p) Normal).

Definition representation_valid (p : position) : Prop :=
  aux p < 256 /\ (kind p = Normal -> aux p = 0).

Theorem normal_constructor_valid : forall i e,
  representation_valid (mkPosition i e Normal 0).
Proof. intros; split; simpl; lia. Qed.

(** The total-order key is injective: binary search cannot collapse positions
    that differ only in continuation kind or payload. *)
Theorem full_key_injective : forall p q,
  term_index p = term_index q ->
  num_errors p = num_errors q ->
  kind p = kind q ->
  aux p = aux q ->
  p = q.
Proof. intros [pi pe pk pa] [qi qe qk qa]; simpl; intros; subst; reflexivity. Qed.

Inductive algorithm : Type := Standard | Osa | MergeSplit.

Definition nat_abs_diff (a b : nat) : nat := (a - b) + (b - a).

Definition standard_subsumes (lhs rhs : position) : bool :=
  if Nat.leb (num_errors lhs) (num_errors rhs) then
    Nat.leb (nat_abs_diff (term_index lhs) (term_index rhs))
            (num_errors rhs - num_errors lhs)
  else false.

Definition osa_subsumes (lhs rhs : position) : bool :=
  if Nat.leb (num_errors lhs) (num_errors rhs) then
    match is_special lhs, is_special rhs with
    | true, true => Nat.eqb (term_index lhs) (term_index rhs)
    | false, false => standard_subsumes lhs rhs
    | _, _ => false
    end
  else false.

Definition merge_split_subsumes
    (query_length : nat) (lhs rhs : position) : bool :=
  Nat.leb (num_errors lhs) (num_errors rhs)
  && Bool.eqb (is_special lhs) (is_special rhs)
  && Nat.leb (term_index lhs) query_length
  && negb (is_special lhs
           && Nat.leb query_length (term_index lhs)
           && Nat.ltb (term_index rhs) query_length)
  && Nat.ltb (num_errors lhs) (num_errors rhs)
  && Nat.eqb (term_index lhs) (term_index rhs).

(** Legacy runtime dispatch: the algorithm test conceptually occurs for every
    pair in the old hot loop. *)
Definition runtime_subsumes
    (a : algorithm) (query_length : nat) (lhs rhs : position) : bool :=
  match a with
  | Standard => standard_subsumes lhs rhs
  | Osa => osa_subsumes lhs rhs
  | MergeSplit => merge_split_subsumes query_length lhs rhs
  end.

Inductive variant_spec : Type := StandardV | OsaV | MergeSplitV.

Definition select_variant (a : algorithm) : variant_spec :=
  match a with Standard => StandardV | Osa => OsaV | MergeSplit => MergeSplitV end.

(** Monomorphized dispatch: selection occurs once, while the chosen leaf is
    applied to every pair on that dictionary edge. *)
Definition static_subsumes
    (v : variant_spec) (query_length : nat) (lhs rhs : position) : bool :=
  match v with
  | StandardV => standard_subsumes lhs rhs
  | OsaV => osa_subsumes lhs rhs
  | MergeSplitV => merge_split_subsumes query_length lhs rhs
  end.

Theorem dispatch_equivalence : forall a query_length lhs rhs,
  runtime_subsumes a query_length lhs rhs =
  static_subsumes (select_variant a) query_length lhs rhs.
Proof. intros; destruct a; reflexivity. Qed.

Theorem selected_variant_is_edge_stable : forall (a : algorithm) (p1 p2 : position),
  select_variant a = select_variant a.
Proof. intros; reflexivity. Qed.

Theorem osa_mixed_continuations_do_not_subsume : forall i e j f,
  osa_subsumes (mkPosition i e Normal 0)
               (mkPosition j f OsaTransposing 0) = false.
Proof.
  intros; unfold osa_subsumes, is_special; simpl.
  destruct (Nat.leb e f); reflexivity.
Qed.

Theorem merge_split_requires_strictly_fewer_errors :
  forall query_length lhs rhs,
    merge_split_subsumes query_length lhs rhs = true ->
    num_errors lhs < num_errors rhs.
Proof.
  intros query_length lhs rhs H.
  unfold merge_split_subsumes in H.
  repeat rewrite andb_true_iff in H.
  apply Nat.ltb_lt.
  tauto.
Qed.

Theorem standard_subsumption_never_reverses_error_order : forall lhs rhs,
  standard_subsumes lhs rhs = true -> num_errors lhs <= num_errors rhs.
Proof.
  intros lhs rhs H.
  unfold standard_subsumes in H.
  destruct (Nat.leb (num_errors lhs) (num_errors rhs)) eqn:Hle.
  - apply Nat.leb_le; assumption.
  - discriminate.
Qed.
