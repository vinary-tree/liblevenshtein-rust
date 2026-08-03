(** * Streaming unrestricted Damerau--Levenshtein obligations

    This assumption-free model mirrors the Rust [DamerauV] transition leaf.
    A pending representative remembers the positive query-endpoint offset
    [delta].  Entry prepays one transposition plus [delta - 1] query deletions;
    each intervening dictionary unit costs one insertion; resolution changes
    no cost.  The arithmetic is exactly the Lowrance--Wagner macro term.
*)

From Stdlib Require Import Arith Bool Lia PeanoNat.

Inductive continuation : Type := Normal | Pending (delta : nat).

Record position : Type := mkPosition {
  term_index : nat;
  errors : nat;
  layer : continuation
}.

Definition valid (p : position) : Prop :=
  match layer p with
  | Normal => True
  | Pending delta => 1 <= delta /\ delta < 256
  end.

Definition entry (p : position) (delta budget : nat) : option position :=
  match layer p with
  | Pending _ => None
  | Normal =>
      if (1 <=? delta) && (delta <? 256) && (errors p + delta <=? budget)
      then Some (mkPosition (term_index p) (errors p + delta) (Pending delta))
      else None
  end.

Definition extend (p : position) (budget : nat) : option position :=
  match layer p with
  | Normal => None
  | Pending delta =>
      if errors p + 1 <=? budget
      then Some (mkPosition (term_index p) (errors p + 1) (Pending delta))
      else None
  end.

Definition resolve (p : position) : option position :=
  match layer p with
  | Normal => None
  | Pending delta =>
      Some (mkPosition (term_index p + delta + 1) (errors p) Normal)
  end.

Definition epsilon_successor (p : position) : option position :=
  match layer p with Normal => Some p | Pending _ => None end.

Definition nat_abs_diff (a b : nat) : nat := (a - b) + (b - a).

Definition damerau_subsumes (lhs rhs : position) : bool :=
  if errors lhs <=? errors rhs then
    match layer lhs, layer rhs with
    | Normal, Normal =>
        nat_abs_diff (term_index lhs) (term_index rhs)
          <=? errors rhs - errors lhs
    | Pending dl, Pending dr =>
        Nat.eqb (term_index lhs) (term_index rhs) && Nat.eqb dl dr
    | _, _ => false
    end
  else false.

Theorem entry_preserves_budget : forall p delta budget q,
  entry p delta budget = Some q -> errors q <= budget.
Proof.
  intros p delta budget q H.
  unfold entry in H.
  destruct (layer p) as [|old_delta]; try discriminate.
  destruct (1 <=? delta) eqn:Hpositive; try discriminate.
  destruct (delta <? 256) eqn:Hbyte; try discriminate.
  destruct (errors p + delta <=? budget) eqn:Hbudget; try discriminate.
  inversion H; subst; simpl.
  apply Nat.leb_le; exact Hbudget.
Qed.

Theorem entry_creates_valid_pending : forall p delta budget q,
  valid p -> entry p delta budget = Some q -> valid q.
Proof.
  intros p delta budget q Hvalid H.
  unfold entry in H.
  destruct (layer p) as [|old_delta]; try discriminate.
  destruct (1 <=? delta) eqn:Hpositive; try discriminate.
  destruct (delta <? 256) eqn:Hbyte; try discriminate.
  destruct (errors p + delta <=? budget) eqn:Hbudget; try discriminate.
  inversion H; subst; simpl.
  split; [apply Nat.leb_le | apply Nat.ltb_lt]; assumption.
Qed.

Theorem extend_preserves_delta_and_adds_one : forall i e delta budget q,
  extend (mkPosition i e (Pending delta)) budget = Some q ->
  layer q = Pending delta /\ errors q = e + 1 /\ term_index q = i.
Proof.
  intros i e delta budget q H; unfold extend in H; simpl in H.
  destruct (e + 1 <=? budget) eqn:Hguard; try discriminate.
  inversion H; subst; simpl; repeat split; reflexivity.
Qed.

Theorem pending_has_no_epsilon_successor : forall i e delta,
  epsilon_successor (mkPosition i e (Pending delta)) = None.
Proof. reflexivity. Qed.

Theorem resolve_advances_exact_endpoint : forall i e delta,
  resolve (mkPosition i e (Pending delta)) =
  Some (mkPosition (i + delta + 1) e Normal).
Proof. reflexivity. Qed.

(** [delta] equals one transposition plus [delta-1] query deletions.
    Adding [between] dictionary insertions yields Lowrance--Wagner's term. *)
Theorem macro_cost_equivalent : forall delta between,
  1 <= delta ->
  delta + between = (delta - 1) + between + 1.
Proof. intros; lia. Qed.

Theorem mixed_continuations_never_subsume : forall i e j f delta,
  damerau_subsumes (mkPosition i e Normal)
                    (mkPosition j f (Pending delta)) = false /\
  damerau_subsumes (mkPosition j f (Pending delta))
                    (mkPosition i e Normal) = false.
Proof.
  intros; unfold damerau_subsumes; simpl.
  destruct (e <=? f), (f <=? e); auto.
Qed.

Theorem pending_subsumption_requires_same_key : forall i e j f dl dr,
  damerau_subsumes (mkPosition i e (Pending dl))
                    (mkPosition j f (Pending dr)) = true ->
  e <= f /\ i = j /\ dl = dr.
Proof.
  intros i e j f dl dr H.
  unfold damerau_subsumes in H; simpl in H.
  destruct (e <=? f) eqn:He; try discriminate.
  apply Nat.leb_le in He.
  apply andb_true_iff in H as [Hi Hd].
  apply Nat.eqb_eq in Hi; apply Nat.eqb_eq in Hd; auto.
Qed.

(** At most [k] live diagonals times [k] deltas gives the claimed quadratic
    frontier bound.  This is an upper bound, not an assertion that every pair
    is reachable. *)
Theorem frontier_quadratic_bound : forall diagonals deltas k,
  diagonals <= k -> deltas <= k -> diagonals * deltas <= k * k.
Proof. intros; nia. Qed.
