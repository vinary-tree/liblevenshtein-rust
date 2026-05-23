(******************************************************************************)
(* Invariants.v - Position Constructor Correctness                           *)
(*                                                                            *)
(* This file proves that position constructors correctly maintain invariants *)
(* and that all invariant checks are decidable. These properties ensure that *)
(* positions created during automaton operation remain valid.                *)
(*                                                                            *)
(* Key theorems:                                                              *)
(*   - Constructor correctness: Valid inputs → valid positions               *)
(*   - Invariant decidability: Can check invariants computationally          *)
(*   - Accessor safety: Reading position fields preserves invariants         *)
(*                                                                            *)
(* References:                                                                *)
(*   - Theory: docs/research/weighted-levenshtein-automata/README.md         *)
(*   - Implementation: src/transducer/generalized/position.rs                *)
(*   - Core definitions: Core.v                                              *)
(*                                                                            *)
(* Authors: Formal Verification Team                                          *)
(* Date: 2025-11-17                                                          *)
(******************************************************************************)

From Stdlib Require Import ZArith.
From Stdlib Require Import Arith.
From Stdlib Require Import Lia.
From Stdlib Require Import Bool.
From Stdlib Require Import Ascii.

Require Import Core.

Open Scope Z_scope.

(******************************************************************************)
(* SECTION 1: Smart Constructors                                             *)
(*                                                                            *)
(* Smart constructors check preconditions and return Option Position.        *)
(* These mirror the Rust `new_*` methods that return Result<Position, Error> *)
(*                                                                            *)
(* DESIGN: We use Option instead of dependent types for simplicity:          *)
(*   - Some p: Valid position created successfully                           *)
(*   - None: Invalid inputs (violate preconditions)                          *)
(*                                                                            *)
(* RUST CORRESPONDENCE: src/transducer/generalized/position.rs:150-400       *)
(******************************************************************************)

(**
  Constructor for I-type (non-final) positions

  PRECONDITIONS:
    1. |offset| ≤ errors (can reach diagonal)
    2. -n ≤ offset ≤ n (within bounds)
    3. errors ≤ n (valid error budget)

  GEOMETRIC MEANING: Position must be reachable from start within budget

  RUST: GeneralizedPosition::new_i(offset, errors, max_distance)
*)
Definition new_i (offset : Z) (errors : nat) (max_distance : nat) : option Position :=
  if (Z.abs offset <=? Z.of_nat errors) &&
     ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarINonFinal offset errors max_distance None)
  else None.

(**
  Constructor for M-type (final) positions

  PRECONDITIONS:
    1. errors ≥ -offset - n (can reach end from position)
    2. -2n ≤ offset ≤ 0 (past term end, within bounds)
    3. errors ≤ n (valid error budget)

  GEOMETRIC MEANING: Position past term end, can still reach query end

  RUST: GeneralizedPosition::new_m(offset, errors, max_distance)
*)
Definition new_m (offset : Z) (errors : nat) (max_distance : nat) : option Position :=
  if (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) &&
     ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarMFinal offset errors max_distance None)
  else None.

(**
  Constructor for I-type transposing positions

  PRECONDITIONS: Same as I-type + must provide entry character

  USAGE: Created when entering transposition operation
  Entry character stores the first character of the transposed pair

  RUST: GeneralizedPosition::new_i_transposing(offset, errors, n, entry_char)
*)
Definition new_i_transposing (offset : Z) (errors : nat) (max_distance : nat)
                              (entry_char : ascii) : option Position :=
  if (Z.abs offset <=? Z.of_nat errors) &&
     ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarITransposing offset errors max_distance (Some entry_char))
  else None.

(**
  Constructor for M-type transposing positions

  PRECONDITIONS: Same as M-type + must provide entry character

  RUST: GeneralizedPosition::new_m_transposing(offset, errors, n, entry_char)
*)
Definition new_m_transposing (offset : Z) (errors : nat) (max_distance : nat)
                              (entry_char : ascii) : option Position :=
  if (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) &&
     ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarMTransposing offset errors max_distance (Some entry_char))
  else None.

(**
  Constructor for I-type splitting positions

  PRECONDITIONS: Same as I-type + must provide entry character

  USAGE: Created when entering phonetic split operation
  Entry character stores the first input character for validation

  RUST: GeneralizedPosition::new_i_splitting(offset, errors, n, entry_char)
*)
Definition new_i_splitting (offset : Z) (errors : nat) (max_distance : nat)
                            (entry_char : ascii) : option Position :=
  if (Z.abs offset <=? Z.of_nat errors) &&
     ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarISplitting offset errors max_distance (Some entry_char))
  else None.

(**
  Constructor for M-type splitting positions

  PRECONDITIONS: Same as M-type + must provide entry character

  RUST: GeneralizedPosition::new_m_splitting(offset, errors, n, entry_char)
*)
Definition new_m_splitting (offset : Z) (errors : nat) (max_distance : nat)
                            (entry_char : ascii) : option Position :=
  if (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) &&
     ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) &&
     (errors <=? max_distance)%nat
  then Some (mkPosition VarMSplitting offset errors max_distance (Some entry_char))
  else None.

(******************************************************************************)
(* SECTION 2: Constructor Correctness Theorems                               *)
(*                                                                            *)
(* These theorems prove that smart constructors maintain invariants:         *)
(*   If constructor returns Some p, then p satisfies its invariant           *)
(*                                                                            *)
(* PROOF STRATEGY: Unfold definitions, case analysis on conditionals         *)
(******************************************************************************)

(**
  THEOREM: new_i Constructor Correctness

  STATEMENT: If new_i succeeds, the result satisfies i_invariant

  INTUITION: The constructor checks exactly the conditions required by
             the invariant, so success implies validity.

  PROOF STRATEGY:
    1. Assume new_i(...) = Some p
    2. Unfold new_i definition
    3. Case analysis on the if-then-else
    4. The "then" branch creates position with exact values checked
    5. Show i_invariant holds for those values

  IMPLEMENTATION IMPACT: Justifies that Rust new_i() error checking is
                          sufficient to guarantee valid positions.
*)
Theorem new_i_correct : forall offset errors max_distance p,
  new_i offset errors max_distance = Some p ->
  i_invariant p.
Proof.
  intros offset errors max_distance p Hnew.
  unfold new_i in Hnew.
  (* Case analysis on the conditional *)
  destruct (Z.abs offset <=? Z.of_nat errors) eqn:Habs; [|discriminate].
  destruct ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  (* All conditions true, so Some was returned *)
  injection Hnew as Heq.
  rewrite <- Heq.

  (* Prove i_invariant for constructed position *)
  unfold i_invariant.
  simpl.

  (* Extract conditions from boolean tests *)
  apply Z.leb_le in Habs.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof *)
  repeat split; auto.
Qed.

(**
  THEOREM: new_m Constructor Correctness

  STATEMENT: If new_m succeeds, the result satisfies m_invariant

  PROOF STRATEGY: Similar to new_i_correct
*)
Theorem new_m_correct : forall offset errors max_distance p,
  new_m offset errors max_distance = Some p ->
  m_invariant p.
Proof.
  intros offset errors max_distance p Hnew.
  unfold new_m in Hnew.
  (* Case analysis on the conditional *)
  destruct (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) eqn:Hreach; [|discriminate].
  destruct ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  (* All conditions true *)
  injection Hnew as Heq.
  rewrite <- Heq.

  (* Prove m_invariant *)
  unfold m_invariant.
  simpl.

  (* Extract conditions *)
  apply Z.geb_le in Hreach.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof of all four conjuncts *)
  split. { reflexivity. }  (* variant *)
  split. { lia. } (* errors ≥ -offset - n (flip Hreach) *)
  split. { split; [exact Hlo | exact Hhi]. } (* bounds *)
  exact Herr. (* errors ≤ n *)
Qed.

(**
  THEOREM: new_i_transposing Constructor Correctness

  STATEMENT: If new_i_transposing succeeds, result satisfies invariant

  ADDITIONAL PROPERTY: entry_char is Some (required for transposition)
*)
Theorem new_i_transposing_correct : forall offset errors max_distance entry_char p,
  new_i_transposing offset errors max_distance entry_char = Some p ->
  i_transposing_invariant p.
Proof.
  intros offset errors max_distance entry_char p Hnew.
  unfold new_i_transposing in Hnew.
  (* Case analysis *)
  destruct (Z.abs offset <=? Z.of_nat errors) eqn:Habs; [|discriminate].
  destruct ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  injection Hnew as Heq.
  rewrite <- Heq.

  (* Prove i_transposing_invariant *)
  unfold i_transposing_invariant.
  simpl.

  (* Extract conditions *)
  apply Z.leb_le in Habs.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof of all conjuncts *)
  split. { reflexivity. }  (* variant *)
  split. { exact Habs. } (* |offset| ≤ errors *)
  split. { split; [exact Hlo | exact Hhi]. } (* bounds *)
  split. { exact Herr. } (* errors ≤ n *)
  (* entry_char <> None *)
  intro H. discriminate H.
Qed.

(**
  THEOREM: new_m_transposing Constructor Correctness
*)
Theorem new_m_transposing_correct : forall offset errors max_distance entry_char p,
  new_m_transposing offset errors max_distance entry_char = Some p ->
  m_transposing_invariant p.
Proof.
  intros offset errors max_distance entry_char p Hnew.
  unfold new_m_transposing in Hnew.
  destruct (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) eqn:Hreach; [|discriminate].
  destruct ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  injection Hnew as Heq.
  rewrite <- Heq.

  unfold m_transposing_invariant.
  simpl.

  apply Z.geb_le in Hreach.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof of all conjuncts *)
  split. { reflexivity. }  (* variant *)
  split. { lia. } (* errors ≥ -offset - n *)
  split. { split; [exact Hlo | exact Hhi]. } (* bounds *)
  split. { exact Herr. } (* errors ≤ n *)
  (* entry_char <> None: after simpl, entry_char p = Some entry_char *)
  intro H. discriminate H.
Qed.

(**
  THEOREM: new_i_splitting Constructor Correctness
*)
Theorem new_i_splitting_correct : forall offset errors max_distance entry_char p,
  new_i_splitting offset errors max_distance entry_char = Some p ->
  i_splitting_invariant p.
Proof.
  intros offset errors max_distance entry_char p Hnew.
  unfold new_i_splitting in Hnew.
  destruct (Z.abs offset <=? Z.of_nat errors) eqn:Habs; [|discriminate].
  destruct ((-Z.of_nat max_distance <=? offset) && (offset <=? Z.of_nat max_distance)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  injection Hnew as Heq.
  rewrite <- Heq.

  unfold i_splitting_invariant.
  simpl.

  apply Z.leb_le in Habs.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof of all conjuncts *)
  split. { reflexivity. }  (* variant *)
  split. { exact Habs. } (* |offset| ≤ errors *)
  split. { split; [exact Hlo | exact Hhi]. } (* bounds *)
  split. { exact Herr. } (* errors ≤ n *)
  (* entry_char <> None *)
  intro H. discriminate H.
Qed.

(**
  THEOREM: new_m_splitting Constructor Correctness
*)
Theorem new_m_splitting_correct : forall offset errors max_distance entry_char p,
  new_m_splitting offset errors max_distance entry_char = Some p ->
  m_splitting_invariant p.
Proof.
  intros offset errors max_distance entry_char p Hnew.
  unfold new_m_splitting in Hnew.
  destruct (Z.of_nat errors >=? (- offset - Z.of_nat max_distance)) eqn:Hreach; [|discriminate].
  destruct ((-Z.of_nat (2 * max_distance) <=? offset) && (offset <=? 0)) eqn:Hbounds; [|discriminate].
  destruct (errors <=? max_distance)%nat eqn:Herr; [|discriminate].

  injection Hnew as Heq.
  rewrite <- Heq.

  unfold m_splitting_invariant.
  simpl.

  apply Z.geb_le in Hreach.
  apply andb_true_iff in Hbounds as [Hlo Hhi].
  apply Z.leb_le in Hlo.
  apply Z.leb_le in Hhi.
  apply Nat.leb_le in Herr.

  (* Construct proof of all conjuncts *)
  split. { reflexivity. }  (* variant *)
  split. { lia. } (* errors ≥ -offset - n *)
  split. { split; [exact Hlo | exact Hhi]. } (* bounds *)
  split. { exact Herr. } (* errors ≤ n *)
  (* entry_char <> None *)
  intro H. discriminate H.
Qed.

(******************************************************************************)
(* SECTION 3: Invariant Decidability                                         *)
(*                                                                            *)
(* Prove that invariants can be checked computationally (decidable).         *)
(* This justifies runtime invariant checking in the Rust implementation.     *)
(******************************************************************************)

(**
  Boolean version of i_invariant for computational checking
*)
Definition i_invariant_b (p : Position) : bool :=
  match variant p with
  | VarINonFinal =>
      (Z.abs (offset p) <=? Z.of_nat (errors p)) &&
      ((-Z.of_nat (max_distance p) <=? offset p) && (offset p <=? Z.of_nat (max_distance p))) &&
      (errors p <=? max_distance p)%nat
  | _ => false
  end.

(**
  Boolean version of m_invariant
*)
Definition m_invariant_b (p : Position) : bool :=
  match variant p with
  | VarMFinal =>
      (Z.of_nat (errors p) >=? (-(offset p) - Z.of_nat (max_distance p))) &&
      ((-Z.of_nat (2 * max_distance p) <=? offset p) && (offset p <=? 0)) &&
      (errors p <=? max_distance p)%nat
  | _ => false
  end.

(**
  THEOREM: i_invariant is decidable

  STATEMENT: Boolean check reflects the Prop invariant

  PRACTICAL USE: Can use i_invariant_b in property tests to validate positions
*)
Theorem i_invariant_decidable : forall p,
  i_invariant_b p = true <-> i_invariant p.
Proof.
  intro p.
  unfold i_invariant_b, i_invariant.
  split; intro H.

  (* true -> Prop *)
  - destruct (variant p) eqn:Hvar; try discriminate.
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff in H1 as [Habs Hbounds].
    apply andb_true_iff in Hbounds as [Hlo Hhi].
    apply Z.leb_le in Habs.
    apply Z.leb_le in Hlo.
    apply Z.leb_le in Hhi.
    apply Nat.leb_le in H2.
    repeat split; auto.

  (* Prop -> true *)
  - destruct H as [Hvar [Habs [Hbounds Herr]]].
    rewrite Hvar.
    apply andb_true_iff; split.
    + apply andb_true_iff; split.
      * apply Z.leb_le. exact Habs.
      * apply andb_true_iff; split.
        -- apply Z.leb_le. lia.
        -- apply Z.leb_le. lia.
    + apply Nat.leb_le. exact Herr.
Qed.

(**
  THEOREM: m_invariant is decidable
*)
Theorem m_invariant_decidable : forall p,
  m_invariant_b p = true <-> m_invariant p.
Proof.
  intro p.
  unfold m_invariant_b, m_invariant.
  split; intro H.

  (* true -> Prop *)
  - destruct (variant p) eqn:Hvar; try discriminate.
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff in H1 as [Hreach Hbounds].
    apply andb_true_iff in Hbounds as [Hlo Hhi].
    apply Z.geb_le in Hreach.
    apply Z.leb_le in Hlo.
    apply Z.leb_le in Hhi.
    apply Nat.leb_le in H2.
    split. { reflexivity. }
    split. { lia. }
    split. { split; [exact Hlo | exact Hhi]. }
    exact H2.

  (* Prop -> true *)
  - destruct H as [Hvar [Hreach [Hbounds Herr]]].
    rewrite Hvar.
    apply andb_true_iff; split.
    + apply andb_true_iff; split.
      * apply Z.geb_le. lia.  (* Flip inequality *)
      * apply andb_true_iff; split.
        -- apply Z.leb_le. lia.
        -- apply Z.leb_le. lia.
    + apply Nat.leb_le. exact Herr.
Qed.

(******************************************************************************)
(* SECTION 4: Accessor Safety                                                *)
(*                                                                            *)
(* Prove that reading position fields maintains invariants.                  *)
(* These are simple but ensure position immutability is safe.                *)
(******************************************************************************)

(**
  LEMMA: Valid positions have bounded errors
*)
Lemma valid_position_errors_bounded : forall p,
  valid_position p -> (errors p <= max_distance p)%nat.
Proof.
  intros p Hvalid.
  unfold valid_position in Hvalid.
  destruct (variant p);
    (* All invariants have errors ≤ max_distance as conjunct *)
    try (destruct Hvalid as [_ [_ [_ H]]]; exact H);
    (* For transposing/splitting variants, extra entry_char conjunct *)
    destruct Hvalid as [_ [_ [_ [H _]]]]; exact H.
Qed.

(**
  LEMMA: Valid I-type positions have bounded offset
*)
Lemma valid_i_offset_bounded : forall p,
  i_invariant p ->
  -Z.of_nat (max_distance p) <= offset p <= Z.of_nat (max_distance p).
Proof.
  intros p [_ [_ [H _]]].
  exact H.
Qed.

(**
  LEMMA: Valid I-type positions can reach diagonal
*)
Lemma valid_i_reachable : forall p,
  i_invariant p ->
  Z.abs (offset p) <= Z.of_nat (errors p).
Proof.
  intros p [_ [H _]].
  exact H.
Qed.

(******************************************************************************)
(* SECTION 5: Invariant Relationships                                        *)
(*                                                                            *)
(* Prove properties relating different invariants.                           *)
(******************************************************************************)

(**
  LEMMA: I-type positions with zero errors must be on diagonal

  INTUITION: If no errors remaining, position must match (offset = 0)
             This is because |offset| ≤ errors, so errors=0 → offset=0
*)
Lemma i_zero_errors_on_diagonal : forall p,
  i_invariant p ->
  errors p = 0%nat ->
  offset p = 0.
Proof.
  intros p [_ [Hreach _]] Hzero.
  rewrite Hzero in Hreach.
  simpl in Hreach.
  (* |offset p| ≤ 0, and |x| ≥ 0 always, so offset p = 0 *)
  destruct (Z.abs_spec (offset p)) as [[Hpos Habs_eq] | [Hneg Habs_eq]].
  - (* offset p ≥ 0, so |offset p| = offset p *)
    rewrite Habs_eq in Hreach.
    lia.
  - (* offset p < 0, so |offset p| = -offset p *)
    rewrite Habs_eq in Hreach.
    lia.
Qed.

(**
  LEMMA: M-type positions with zero errors remain within the reachable
  overshoot interval.

  The stronger historical claim `offset = -max_distance` is not implied by
  `m_invariant`: zero-error M positions may have any offset in the allowed
  interval as long as the reachability lower bound is satisfied.
*)
Lemma m_zero_errors_reachable_bounds : forall p,
  m_invariant p ->
  errors p = 0%nat ->
  -Z.of_nat (max_distance p) <= offset p <= 0.
Proof.
  intros p [_ [Hreach [Hbounds _]]] Hzero.
  rewrite Hzero in Hreach.
  simpl in Hreach.
  lia.
Qed.

(******************************************************************************)
(* END OF INVARIANTS.V                                                        *)
(*                                                                            *)
(* This file establishes constructor correctness and invariant decidability. *)
(* Key results:                                                               *)
(*   - All six constructors proven to maintain invariants                    *)
(*   - Invariants are computationally checkable                              *)
(*   - Position accessors are safe                                           *)
(*                                                                            *)
(* NEXT STEPS:                                                                *)
(*   - Operations.v: Define standard edit operations                         *)
(*   - Transitions.v: Prove successor functions correct                      *)
(*   - State.v: Prove anti-chain preservation                                *)
(******************************************************************************)
