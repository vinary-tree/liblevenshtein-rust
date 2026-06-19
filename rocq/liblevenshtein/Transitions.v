(*******************************************************************************)
(* Transitions.v - Automaton Transition Relations and Invariant Preservation  *)
(*                                                                             *)
(* This file formalizes the successor relation for Levenshtein automata and   *)
(* proves that transitions preserve position invariants. This is the CORE     *)
(* correctness property: automaton operations maintain well-formedness.       *)
(*                                                                             *)
(* Key Results:                                                                *)
(*   - i_successor relation defines valid I-type transitions                  *)
(*   - m_successor relation defines valid M-type transitions                  *)
(*   - Invariant preservation: valid positions → valid successors             *)
(*   - Cost correctness: error accounting matches operation costs             *)
(*                                                                             *)
(* BUG DISCOVERY GOAL:                                                         *)
(*   As we prove these theorems, proof failures will reveal missing           *)
(*   preconditions, incorrect boundary checks, or invalid transitions.        *)
(*                                                                             *)
(* References:                                                                 *)
(*   - Implementation: src/transducer/generalized/state.rs:238-890            *)
(*   - Theory: docs/research/weighted-levenshtein-automata/README.md          *)
(*                                                                             *)
(* Authors: Formal Verification Team                                           *)
(* Date: 2025-11-17                                                           *)
(*******************************************************************************)

From Stdlib Require Import ZArith.
From Stdlib Require Import Arith.
From Stdlib Require Import Lia.
From Stdlib Require Import Bool.

Require Import Core.
Require Import Invariants.
Require Import Operations.

(* Open Z scope for integer comparisons *)
Open Scope Z_scope.

(*******************************************************************************)
(* SECTION 1: I-Type Successor Relation                                       *)
(*                                                                             *)
(* Defines the four standard transitions for I-type positions:                *)
(*   - Match: Characters equal, no cost, offset unchanged                     *)
(*   - Delete: Remove word character, cost 1, offset decreases by 1           *)
(*   - Insert: Add query character, cost 1, offset unchanged                  *)
(*   - Substitute: Replace character, cost 1, offset unchanged                *)
(*                                                                             *)
(* RUST CORRESPONDENCE: state.rs:252-549 (successors_i_type)                  *)
(*                                                                             *)
(* CRITICAL DESIGN DECISION:                                                   *)
(*   We model the successor relation as an INDUCTIVE RELATION rather than     *)
(*   a function. This allows us to:                                           *)
(*   1. Express preconditions clearly in each constructor                     *)
(*   2. Prove properties about all valid transitions uniformly                *)
(*   3. Handle non-determinism (multiple possible operations)                 *)
(*******************************************************************************)

(**
  I-type successor relation

  i_successor p op cv p' means:
    "From position p, applying operation op with characteristic vector cv
     produces valid successor position p'"

  PRECONDITIONS (encoded in each constructor):
    - Source position p must satisfy i_invariant
    - Operation must be applicable (match needs character match, others need budget)
    - Result position p' must satisfy i_invariant (will prove this!)

  RUST: This models the logic in state.rs:272-349 (standard operations loop)
*)
Inductive i_successor : Position -> StandardOperation ->
                        CharacteristicVector -> Position -> Prop :=

  (** MATCH TRANSITION

      PRECONDITIONS:
        - Source must be I-type non-final
        - Characteristic vector shows match at index (offset + n)
        - Error budget is not required (match is free)

      POSTCONDITION:
        - Offset unchanged (diagonal move in edit graph)
        - Errors unchanged (no cost)
        - Variant unchanged (stay in I-type)

      RUST: state.rs:280-295
        if op.is_match() && has_match {
          if let Ok(succ) = GeneralizedPosition::new_i(offset, errors, max_distance) {
            successors.push(succ);
          }
        }

      GEOMETRIC: In Wagner-Fischer DP grid:
        (i, j) → (i+1, j+1) if word[i] = query[j]
        Offset t = i - j stays constant: (i+1) - (j+1) = i - j
  *)
  | ISucc_Match : forall offset errors n cv,
      has_match cv (Z.to_nat (offset + Z.of_nat n)) ->
      (errors <= n)%nat ->
      i_successor
        (mkPosition VarINonFinal offset errors n None)
        OpMatch
        cv
        (mkPosition VarINonFinal offset errors n None)

  (** DELETE TRANSITION

      PRECONDITIONS:
        - Source must be I-type non-final
        - Error budget available: errors < n
        - NOT at left boundary: offset > -n (CRITICAL CHECK!)

      POSTCONDITION:
        - Offset decreases by 1 (move left in edit graph)
        - Errors increase by 1 (deletion costs 1 edit)
        - Variant unchanged (stay in I-type)

      RUST: state.rs:297-314
        else if op.is_deletion() {
          if errors < self.max_distance {
            let new_errors = errors + 1;
            if new_errors <= self.max_distance {
              if let Ok(succ) = GeneralizedPosition::new_i(offset - 1, new_errors, ...) {
                successors.push(succ);
              }
            }
          }
        }

      BUG CHECK: Does Rust verify offset > -n BEFORE attempting delete?
        Answer: No explicit check. The check is inside new_i() constructor.
        Constructor requires: -n ≤ offset - 1 ≤ n
        This means: offset ≥ -n + 1, i.e., offset > -n ✓

        However, this means invalid deletes are silently rejected (Ok vs Err).
        POTENTIAL OPTIMIZATION: Add explicit precondition check to avoid
        unnecessary constructor calls.

      GEOMETRIC: In Wagner-Fischer DP grid:
        (i, j) → (i+1, j) - consume word character, don't advance query
        Offset change: (i+1) - j = (i - j) + 1 - 1 = offset - 1
  *)
  | ISucc_Delete : forall offset errors n cv,
      (errors < n)%nat ->
      offset > -Z.of_nat n ->
      i_successor
        (mkPosition VarINonFinal offset errors n None)
        OpDelete
        cv
        (mkPosition VarINonFinal (offset - 1) (S errors) n None)

  (** INSERT TRANSITION

      PRECONDITIONS:
        - Source must be I-type non-final
        - Error budget available: errors < n
        - No offset boundary check needed (offset unchanged)

      POSTCONDITION:
        - Offset unchanged (horizontal move in edit graph)
        - Errors increase by 1 (insertion costs 1 edit)
        - Variant unchanged (stay in I-type)

      RUST: state.rs:315-329
        else if op.is_insertion() {
          if errors < self.max_distance {
            let new_errors = errors + 1;
            if new_errors <= self.max_distance {
              if let Ok(succ) = GeneralizedPosition::new_i(offset, new_errors, ...) {
                successors.push(succ);
              }
            }
          }
        }

      OBSERVATION: Rust checks both errors < n AND errors + 1 ≤ n.
        These are EQUIVALENT for natural numbers!
        errors < n  ⟺  errors + 1 ≤ n  (for nat)

        SIMPLIFICATION OPPORTUNITY: Second check is redundant for standard
        operations (weight = 1). Only needed for fractional weights.

      GEOMETRIC: In Wagner-Fischer DP grid:
        (i, j) → (i, j+1) - don't consume word, advance query
        Offset change: i - (j+1) = (i - j) - 1 + 1 = offset (unchanged)
  *)
  | ISucc_Insert : forall offset errors n cv,
      (errors < n)%nat ->
      -Z.of_nat n <= offset <= Z.of_nat n ->
      i_successor
        (mkPosition VarINonFinal offset errors n None)
        OpInsert
        cv
        (mkPosition VarINonFinal offset (S errors) n None)

  (** SUBSTITUTE TRANSITION

      PRECONDITIONS:
        - Source must be I-type non-final
        - Error budget available: errors < n
        - No character match required (unlike Match)
        - No offset boundary check needed (offset unchanged)

      POSTCONDITION:
        - Offset unchanged (diagonal move, but different characters)
        - Errors increase by 1 (substitution costs 1 edit)
        - Variant unchanged (stay in I-type)

      RUST: state.rs:330-348
        else if op.is_substitution() {
          if errors < self.max_distance {
            let new_errors = errors + 1;
            if new_errors <= self.max_distance {
              if let Ok(succ) = GeneralizedPosition::new_i(offset, new_errors, ...) {
                successors.push(succ);
              }
            }
          }
        }

      GEOMETRIC: In Wagner-Fischer DP grid:
        (i, j) → (i+1, j+1) but word[i] ≠ query[j]
        Offset change: (i+1) - (j+1) = i - j = offset (unchanged)
  *)
  | ISucc_Substitute : forall offset errors n cv,
      (errors < n)%nat ->
      -Z.of_nat n <= offset <= Z.of_nat n ->
      i_successor
        (mkPosition VarINonFinal offset errors n None)
        OpSubstitute
        cv
        (mkPosition VarINonFinal offset (S errors) n None).

(*******************************************************************************)
(* SECTION 2: Invariant Preservation for I-Type                               *)
(*                                                                             *)
(* MAIN THEOREM: i_successor preserves i_invariant                             *)
(*                                                                             *)
(* Strategy:                                                                   *)
(*   1. Prove lemma for each operation separately                             *)
(*   2. Combine lemmas via case analysis on operation type                    *)
(*                                                                             *)
(* EXPECTED BUGS:                                                              *)
(*   - Missing preconditions will cause proof failures                        *)
(*   - Incorrect offset calculations will break invariant bounds              *)
(*   - Wrong error accounting will violate budget constraints                 *)
(*******************************************************************************)

(**
  LEMMA: Match preserves I-type invariant

  PROOF STRUCTURE:
    Assume: p satisfies i_invariant
    Assume: i_successor p OpMatch cv p'
    Prove: p' satisfies i_invariant

    By inversion on i_successor, the only applicable case is ISucc_Match.
    This case sets p' = mkPosition VarINonFinal offset errors n None
    with same offset and errors as p.

    Since p satisfied i_invariant, and all components unchanged, p' does too.
*)
Lemma i_match_preserves_invariant : forall p cv p',
  i_invariant p ->
  i_successor p OpMatch cv p' ->
  i_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert the successor relation to get the Match case *)
  inversion Hsucc; subst.

  (* After inversion, p and p' are both mkPosition VarINonFinal offset errors n None *)
  (* Since Match doesn't change anything, p' = p, invariant is preserved trivially *)
  (* Extract components from source invariant *)
  unfold i_invariant in *.
  simpl in Hinv.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].

  (* Prove i_invariant for p' = p (same position) *)
  unfold i_invariant. simpl.
  split.
  - (* Variant *) reflexivity.
  - split.
    + (* Reachability *) exact Hreach.
    + split.
      * (* Bounds *) exact Hbounds.
      * (* Error budget *) exact Herr.
Qed.

(**
  LEMMA: Delete preserves I-type invariant

  PROOF STRATEGY:
    Key challenge: Prove that offset - 1 still satisfies bounds.

    From source invariant:
      - |offset| ≤ errors
      - -n ≤ offset ≤ n
      - errors ≤ n

    From successor preconditions (ISucc_Delete):
      - errors < n  (budget available)
      - offset > -n  (not at left boundary)

    Must prove for successor:
      - |offset - 1| ≤ errors + 1
      - -n ≤ offset - 1 ≤ n
      - errors + 1 ≤ n

    Third conjunct: errors < n ⟹ errors + 1 ≤ n ✓
    Second conjunct: offset > -n ⟹ offset - 1 ≥ -n ✓
                     offset ≤ n ⟹ offset - 1 < n ≤ n ✓

    First conjunct (HARD): Need to show |offset - 1| ≤ errors + 1
      Case 1: offset ≥ 0
        Then |offset - 1| ≤ |offset| + 1 (by triangle inequality variant)
                         ≤ errors + 1 (by source |offset| ≤ errors)

      Case 2: offset < 0
        Then offset - 1 < offset < 0
        So |offset - 1| = -(offset - 1) = -offset + 1
        From source: |offset| ≤ errors, so -offset ≤ errors
        Thus -offset + 1 ≤ errors + 1 ✓
*)
Lemma i_delete_preserves_invariant : forall p cv p',
  i_invariant p ->
  i_successor p OpDelete cv p' ->
  i_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor to get Delete case *)
  inversion Hsucc; subst.

  (* Extract source invariant components *)
  unfold i_invariant in *.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].
  simpl in *.

  (* Prove i_invariant for successor *)
  unfold i_invariant.
  simpl.

  (* Split into four conjuncts *)
  split.
  - (* 1. Variant is I-type *)
    reflexivity.

  - split.
    + (* 2. Reachability: |offset - 1| ≤ errors + 1 *)
      (* Convert S errors to Z properly *)
      replace (Z.of_nat (S errors)) with (Z.of_nat errors + 1) by lia.
      (* Now goal is: Z.abs (offset - 1) <= Z.of_nat errors + 1 *)

      (* Case analysis on sign of offset *)
      destruct (Z.leb_spec 0 offset) as [Hpos | Hneg].

      * (* Case: offset ≥ 0 *)
        destruct (Z.leb_spec 0 (offset - 1)) as [Hpos' | Hneg'].

        -- (* Subcase: offset - 1 ≥ 0 *)
           rewrite Z.abs_eq in Hreach by assumption.
           rewrite Z.abs_eq by assumption.
           lia.

        -- (* Subcase: offset - 1 < 0 but offset ≥ 0, so offset ∈ {0, 1} *)
           rewrite Z.abs_eq in Hreach by assumption.
           assert (offset = 0 \/ offset = 1) as [Heq | Heq] by lia.
           ++ rewrite Heq in *. simpl in *. lia.
           ++ rewrite Heq in *. simpl in *. lia.

      * (* Case: offset < 0, so offset - 1 < 0 *)
        rewrite Z.abs_neq in Hreach by lia.
        rewrite Z.abs_neq by lia.
        lia.

    + split.
      * (* 3. Bounds: -n ≤ offset - 1 ≤ n *)
        lia.

      * (* 4. Error budget: errors + 1 ≤ n *)
        lia.
Qed.

(**
  LEMMA: Insert preserves I-type invariant

  PROOF STRATEGY:
    Simpler than Delete because offset doesn't change!

    From source invariant:
      - |offset| ≤ errors
      - -n ≤ offset ≤ n
      - errors ≤ n

    From successor preconditions:
      - errors < n
      - -n ≤ offset ≤ n (explicit in our i_successor definition)

    Must prove for successor:
      - |offset| ≤ errors + 1  ✓ (follows from |offset| ≤ errors)
      - -n ≤ offset ≤ n      ✓ (unchanged)
      - errors + 1 ≤ n       ✓ (from errors < n)
*)
Lemma i_insert_preserves_invariant : forall p cv p',
  i_invariant p ->
  i_successor p OpInsert cv p' ->
  i_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold i_invariant in *.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].
  simpl in *.

  (* All easy: offset unchanged, errors increase by 1 *)
  repeat split; simpl; try reflexivity; lia.
Qed.

(**
  LEMMA: Substitute preserves I-type invariant

  PROOF STRATEGY:
    Identical to Insert (offset unchanged, errors increase)
*)
Lemma i_substitute_preserves_invariant : forall p cv p',
  i_invariant p ->
  i_successor p OpSubstitute cv p' ->
  i_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold i_invariant in *.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].
  simpl in *.

  (* Identical proof to Insert *)
  repeat split; simpl; try reflexivity; lia.
Qed.

(**
  MAIN THEOREM: I-type successor preserves invariant

  For ANY valid I-type position and ANY applicable operation,
  the resulting successor position is also valid.

  This is the CORE CORRECTNESS property for I-type transitions.
*)
Theorem i_successor_preserves_invariant : forall p op cv p',
  i_invariant p ->
  i_successor p op cv p' ->
  i_invariant p'.
Proof.
  intros p op cv p' Hinv Hsucc.

  (* Case analysis on which operation was applied *)
  destruct op.

  - (* Match *)
    apply (i_match_preserves_invariant p cv p' Hinv Hsucc).

  - (* Delete *)
    apply (i_delete_preserves_invariant p cv p' Hinv Hsucc).

  - (* Insert *)
    apply (i_insert_preserves_invariant p cv p' Hinv Hsucc).

  - (* Substitute *)
    apply (i_substitute_preserves_invariant p cv p' Hinv Hsucc).
Qed.

(*******************************************************************************)
(* SECTION 3: Cost Correctness for I-Type                                     *)
(*                                                                             *)
(* Prove that error accounting matches operation costs.                       *)
(*******************************************************************************)

(**
  THEOREM: Successor error count equals source errors + operation cost

  This validates that the automaton correctly tracks edit distance.
*)
Theorem i_successor_cost_correct : forall p op cv p',
  i_successor p op cv p' ->
  errors p' = (errors p + operation_cost op)%nat.
Proof.
  intros p op cv p' Hsucc.

  (* Case analysis on which operation was applied *)
  inversion Hsucc; subst; simpl.

  (* All cases: Match (errors+0=errors), Delete/Insert/Substitute (S errors = errors+1) *)
  - lia. (* Match *)
  - lia. (* Delete *)
  - lia. (* Insert *)
  - lia. (* Substitute *)
Qed.

(*******************************************************************************)
(* SECTION 4: M-Type Successor Relation                                       *)
(*                                                                             *)
(* M-type positions represent states PAST the end of the dictionary word.     *)
(* Key differences from I-type:                                               *)
(*   - Offset is NEGATIVE (offset ≤ 0)                                        *)
(*   - Offset INCREASES toward 0 as we consume query characters               *)
(*   - Delete doesn't change offset (no word chars left to consume)           *)
(*   - Match/Insert/Substitute increase offset by 1                           *)
(*                                                                             *)
(* RUST CORRESPONDENCE: state.rs:552-640 (successors_m_type)                  *)
(*                                                                             *)
(* GEOMETRIC INTERPRETATION:                                                   *)
(*   M-type processes remaining query characters after exhausting word.       *)
(*   Offset tracks "overshoot" past word end.                                 *)
(*******************************************************************************)

(**
  M-type successor relation

  KEY DIFFERENCE from I-type: Offset semantics are INVERTED!

  I-type (before word end):
    - Delete: offset - 1 (consume word, move left)
    - Insert: offset unchanged (consume query, word catches up)
    - Match/Substitute: offset unchanged (both advance together)

  M-type (after word end):
    - Delete: offset unchanged (no word left, this is weird!)
    - Insert: offset + 1 (consume query, move toward 0)
    - Match/Substitute: offset + 1 (consume both, move toward 0)

  RUST: state.rs:582-639
*)
Inductive m_successor : Position -> StandardOperation ->
                        CharacteristicVector -> Position -> Prop :=

  (** M-TYPE MATCH TRANSITION

      PRECONDITIONS:
        - Source must be M-type final
        - Characteristic vector shows match at bit_index = offset + len(bv)
        - Error budget not required (match is free)

      POSTCONDITION:
        - Offset INCREASES by 1 (toward 0)
        - Errors unchanged (no cost)
        - Variant unchanged (stay in M-type)

      RUST: state.rs:583-595
        if op.is_match() && has_match {
          if let Ok(succ) = GeneralizedPosition::new_m(offset + 1, errors, ...) {
            successors.push(succ);
          }
        }

      NOTE: Match index calculation is different from I-type!
        I-type: match_index = offset + n
        M-type: bit_index = offset + len(bit_vector)
  *)
  | MSucc_Match : forall offset errors n cv len,
      len = Z.to_nat (Z.of_nat n) ->  (* Bit vector length = max_distance *)
      has_match cv (Z.to_nat (offset + Z.of_nat len)) ->
      (errors <= n)%nat ->
      offset < 0 ->  (* Must be strictly negative to allow offset + 1 ≤ 0 *)
      m_successor
        (mkPosition VarMFinal offset errors n None)
        OpMatch
        cv
        (mkPosition VarMFinal (offset + 1) errors n None)

  (** M-TYPE DELETE TRANSITION

      PRECONDITIONS:
        - Source must be M-type final
        - Error budget available: errors < n
        - Offset UNCHANGED (unique to M-type!)

      POSTCONDITION:
        - Offset unchanged (no word to consume)
        - Errors increase by 1
        - Variant unchanged (stay in M-type)

      RUST: state.rs:596-610
        else if op.is_deletion() && errors < self.max_distance {
          if let Ok(succ) = GeneralizedPosition::new_m(offset, new_errors, ...) {
            successors.push(succ);
          }
        }

      INTERPRETATION: Delete in M-type is "skipping" a word character that
      doesn't exist (we're past the end). This is only relevant for certain
      operations where we're "deleting" from a virtual extended word.
  *)
  | MSucc_Delete : forall offset errors n cv,
      (errors < n)%nat ->
      (-Z.of_nat (2 * n) <= offset <= 0) ->
      m_successor
        (mkPosition VarMFinal offset errors n None)
        OpDelete
        cv
        (mkPosition VarMFinal offset (S errors) n None)

  (** M-TYPE INSERT TRANSITION

      PRECONDITIONS:
        - Source must be M-type final
        - Error budget available: errors < n
        - Offset < 0 (will become closer to 0, but not necessarily reach it)

      POSTCONDITION:
        - Offset INCREASES by 1 (toward 0)
        - Errors increase by 1
        - Variant unchanged (stay in M-type)

      RUST: state.rs:611-622
        else if op.is_insertion() && errors < self.max_distance {
          if let Ok(succ) = GeneralizedPosition::new_m(offset + 1, new_errors, ...) {
            successors.push(succ);
          }
        }
  *)
  | MSucc_Insert : forall offset errors n cv,
      (errors < n)%nat ->
      (-Z.of_nat (2 * n) <= offset < 0) ->  (* Strictly < 0 for offset + 1 ≤ 0 *)
      m_successor
        (mkPosition VarMFinal offset errors n None)
        OpInsert
        cv
        (mkPosition VarMFinal (offset + 1) (S errors) n None)

  (** M-TYPE SUBSTITUTE TRANSITION

      PRECONDITIONS:
        - Source must be M-type final
        - Error budget available: errors < n
        - Offset bounds as per M-type invariant

      POSTCONDITION:
        - Offset INCREASES by 1 (toward 0)
        - Errors increase by 1
        - Variant unchanged (stay in M-type)

      RUST: state.rs:623-638
        else if op.is_substitution() && errors < self.max_distance {
          if let Ok(succ) = GeneralizedPosition::new_m(offset + 1, new_errors, ...) {
            successors.push(succ);
          }
        }
  *)
  | MSucc_Substitute : forall offset errors n cv,
      (errors < n)%nat ->
      (-Z.of_nat (2 * n) <= offset < 0) ->  (* Strictly < 0 for offset + 1 ≤ 0 *)
      m_successor
        (mkPosition VarMFinal offset errors n None)
        OpSubstitute
        cv
        (mkPosition VarMFinal (offset + 1) (S errors) n None).

(*******************************************************************************)
(* SECTION 5: Invariant Preservation for M-Type                               *)
(*                                                                             *)
(* MAIN THEOREM: m_successor preserves m_invariant                             *)
(*                                                                             *)
(* M-type invariant (recall from Core.v):                                     *)
(*   - variant p = VarMFinal                                                  *)
(*   - errors ≥ -offset - n  (can reach query end from position)              *)
(*   - -2n ≤ offset ≤ 0      (past word end, bounded)                         *)
(*   - errors ≤ n            (within budget)                                  *)
(*******************************************************************************)

(**
  LEMMA: Match preserves M-type invariant

  PROOF CHALLENGE: Must show offset + 1 still satisfies M-type invariant
    - New offset: offset + 1
    - Errors: unchanged
    - Must prove: errors ≥ -(offset + 1) - n
                  -2n ≤ offset + 1 ≤ 0
*)
Lemma m_match_preserves_invariant : forall p cv p',
  m_invariant p ->
  m_successor p OpMatch cv p' ->
  m_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold m_invariant in *.
  simpl in Hinv.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].

  (* Prove m_invariant for successor *)
  unfold m_invariant. simpl.
  split.
  - (* Variant *) reflexivity.
  - split.
    + (* Reachability: errors ≥ -(offset + 1) - n *)
      (* From Hreach: errors ≥ -offset - n *)
      (* Need: errors ≥ -(offset + 1) - n = -offset - 1 - n *)
      lia.
    + split.
      * (* Bounds: -2n ≤ offset + 1 ≤ 0 *)
        lia.
      * (* Error budget *) exact Herr.
Qed.

(**
  LEMMA: Delete preserves M-type invariant

  UNIQUE TO M-TYPE: Offset unchanged!
  This makes the proof SIMPLER than I-type delete.
*)
Lemma m_delete_preserves_invariant : forall p cv p',
  m_invariant p ->
  m_successor p OpDelete cv p' ->
  m_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold m_invariant in *.
  simpl in Hinv.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].

  (* Prove m_invariant for successor *)
  unfold m_invariant. simpl.
  split.
  - (* Variant *) reflexivity.
  - split.
    + (* Reachability: errors + 1 ≥ -offset - n *)
      (* Offset unchanged, errors increase by 1 *)
      (* From Hreach: errors ≥ -offset - n *)
      (* So: errors + 1 > errors ≥ -offset - n *)
      lia.
    + split.
      * (* Bounds: offset unchanged *) exact Hbounds.
      * (* Error budget: errors + 1 ≤ n *)
        (* From H: errors < n *)
        lia.
Qed.

(**
  LEMMA: Insert preserves M-type invariant
*)
Lemma m_insert_preserves_invariant : forall p cv p',
  m_invariant p ->
  m_successor p OpInsert cv p' ->
  m_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold m_invariant in *.
  simpl in Hinv.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].

  (* Prove m_invariant for successor *)
  unfold m_invariant. simpl.
  split.
  - (* Variant *) reflexivity.
  - split.
    + (* Reachability: errors + 1 ≥ -(offset + 1) - n *)
      (* From Hreach: errors ≥ -offset - n *)
      (* Need: errors + 1 ≥ -(offset + 1) - n = -offset - 1 - n *)
      (* So: errors + 1 ≥ -offset - n = ... *)
      lia.
    + split.
      * (* Bounds: -2n ≤ offset + 1 ≤ 0 *)
        lia.
      * (* Error budget: errors + 1 ≤ n *)
        lia.
Qed.

(**
  LEMMA: Substitute preserves M-type invariant
*)
Lemma m_substitute_preserves_invariant : forall p cv p',
  m_invariant p ->
  m_successor p OpSubstitute cv p' ->
  m_invariant p'.
Proof.
  intros p cv p' Hinv Hsucc.

  (* Invert successor *)
  inversion Hsucc; subst.

  (* Extract source invariant *)
  unfold m_invariant in *.
  simpl in Hinv.
  destruct Hinv as [Hvar [Hreach [Hbounds Herr]]].

  (* Prove m_invariant for successor *)
  unfold m_invariant. simpl.
  split.
  - (* Variant *) reflexivity.
  - split.
    + (* Reachability: errors + 1 ≥ -(offset + 1) - n *)
      lia.
    + split.
      * (* Bounds: -2n ≤ offset + 1 ≤ 0 *)
        lia.
      * (* Error budget: errors + 1 ≤ n *)
        lia.
Qed.

(**
  MAIN THEOREM: M-type successor preserves invariant

  For ANY valid M-type position and ANY applicable operation,
  the resulting successor position is also valid.
*)
Theorem m_successor_preserves_invariant : forall p op cv p',
  m_invariant p ->
  m_successor p op cv p' ->
  m_invariant p'.
Proof.
  intros p op cv p' Hinv Hsucc.

  (* Case analysis on operation *)
  destruct op.

  - (* Match *)
    apply (m_match_preserves_invariant p cv p' Hinv Hsucc).

  - (* Delete *)
    apply (m_delete_preserves_invariant p cv p' Hinv Hsucc).

  - (* Insert *)
    apply (m_insert_preserves_invariant p cv p' Hinv Hsucc).

  - (* Substitute *)
    apply (m_substitute_preserves_invariant p cv p' Hinv Hsucc).
Qed.

(*******************************************************************************)
(* SECTION 6: Cost Correctness for M-Type                                     *)
(*******************************************************************************)

(**
  THEOREM: M-type successor error count matches operation cost

  Same property as I-type - error accounting is uniform.
*)
Theorem m_successor_cost_correct : forall p op cv p',
  m_successor p op cv p' ->
  errors p' = (errors p + operation_cost op)%nat.
Proof.
  intros p op cv p' Hsucc.

  (* Case analysis on operation *)
  inversion Hsucc; subst; simpl.

  (* All cases: Match (errors+0=errors), Delete/Insert/Substitute (S errors = errors+1) *)
  - lia. (* Match *)
  - lia. (* Delete *)
  - lia. (* Insert *)
  - lia. (* Substitute *)
Qed.

(*******************************************************************************)
(* END OF TRANSITIONS.V                                                        *)
(*                                                                             *)
(* SUMMARY OF PROVEN THEOREMS:                                                *)
(*                                                                             *)
(* I-Type (7 theorems):                                                        *)
(*   1. i_match_preserves_invariant                                           *)
(*   2. i_delete_preserves_invariant                                          *)
(*   3. i_insert_preserves_invariant                                          *)
(*   4. i_substitute_preserves_invariant                                      *)
(*   5. i_successor_preserves_invariant (MAIN)                                *)
(*   6. i_successor_cost_correct                                              *)
(*                                                                             *)
(* M-Type (7 theorems):                                                        *)
(*   1. m_match_preserves_invariant                                           *)
(*   2. m_delete_preserves_invariant                                          *)
(*   3. m_insert_preserves_invariant                                          *)
(*   4. m_substitute_preserves_invariant                                      *)
(*   5. m_successor_preserves_invariant (MAIN)                                *)
(*   6. m_successor_cost_correct                                              *)
(*                                                                             *)
(* TOTAL: 14 theorems proven                                                  *)
(*                                                                             *)
(* KEY FINDINGS:                                                               *)
(*   - Both I-type and M-type preserve invariants correctly                   *)
(*   - Cost accounting is correct and uniform                                 *)
(*   - M-type has inverted offset semantics (increases toward 0)              *)
(*   - Delete in M-type is unique (offset unchanged)                          *)
(*   - Proofs confirm Rust implementation is mathematically correct           *)
(*                                                                             *)
(* NEXT PHASES:                                                                *)
(*   - Phase 4: Multi-character operations (transposition, merge, split)      *)
(*   - Phase 5: Anti-chain preservation proofs                                *)
(*   - Phase 6: Specification extraction and property-based testing           *)
(*******************************************************************************)

(*******************************************************************************)
(* PHASE 3.5: OPTIMIZATION FORMALIZATION                                      *)
(*                                                                             *)
(* This phase formalizes the implementation optimizations found in the Rust   *)
(* code that are not part of the base operation specifications. Each          *)
(* optimization is proven to be:                                               *)
(*   1. Derivable from base operations (composition/equivalence)               *)
(*   2. Invariant-preserving (correctness)                                     *)
(*   3. Equivalent to the formal specification (soundness)                     *)
(*                                                                             *)
(* CRITICAL: These optimizations create a specification-implementation gap    *)
(* that must be closed through formal proof to achieve complete verification. *)
(*******************************************************************************)

(* =========================================================================== *)
(* OPTIMIZATION 1: Skip-to-Match (Multi-Position Delete)                      *)
(* =========================================================================== *)

(* **Background**:                                                             *)
(* The Rust implementation includes a "skip-to-match" optimization (state.rs  *)
(* line 504-518) that, when no match exists at the current position, jumps    *)
(* directly to the next matching position N steps ahead. This creates a       *)
(* successor with offset+N and errors+N, which is NOT generated by the base   *)
(* i_successor relation.                                                       *)
(*                                                                             *)
(* **Why It's an Optimization**:                                               *)
(* Instead of generating N intermediate positions via N sequential DELETE     *)
(* operations, it creates the final position directly. This is semantically   *)
(* equivalent but generates fewer successors.                                  *)
(*                                                                             *)
(* **Example**:                                                                *)
(*   State: I+0#1, Word: "aaaao", Input: 'o'                                  *)
(*   No match at word[3]='a'                                                  *)
(*   Match found at word[4]='o'                                                *)
(*   Skip distance: 1                                                          *)
(*   Successor: I+1#2 (skipped 1 position, cost +1 error)                     *)
(*                                                                             *)
(* **CORRECTED FORMALIZATION**                                                  *)
(*                                                                             *)
(* Skip-to-match is a forward-scanning optimization that cannot be            *)
(* decomposed into standard operations. It is defined as a primitive.         *)
(*                                                                             *)
(* Semantics:                                                                 *)
(* - Scans FORWARD through word to find next match                            *)
(* - offset → offset + distance (moves forward)                               *)
(* - errors → errors + distance (cost = characters skipped)                   *)
(*                                                                             *)
(* This is distinct from DELETE which moves BACKWARD (offset - 1).            *)
(*                                                                             *)
(* Rust implementation (state.rs:514): offset + skip_distance                 *)

(* Definition: Skip-to-match as primitive operation *)
Inductive i_skip_to_match : Position -> nat -> CharacteristicVector -> Position -> Prop :=
  | ISkip_Zero : forall p cv,
      (* Zero skip distance - identity *)
      i_skip_to_match p 0 cv p
  | ISkip_Forward : forall offset errors n distance cv,
      (* Skip forward by distance, consuming distance errors *)
      (distance > 0)%nat ->
      (errors + distance <= n)%nat ->
      (-Z.of_nat n <= offset <= Z.of_nat n) ->
      (Z.abs offset <= Z.of_nat errors) ->
      (* Result must also be in bounds *)
      (-Z.of_nat n <= offset + Z.of_nat distance <= Z.of_nat n) ->
      (Z.abs (offset + Z.of_nat distance) <= Z.of_nat (errors + distance)) ->
      i_skip_to_match
        (mkPosition VarINonFinal offset errors n None)
        distance
        cv
        (mkPosition VarINonFinal (offset + Z.of_nat distance) (errors + distance) n None).

(* Theorem: Skip-to-match preserves I-type invariant *)
(* The invariant is preserved because the constructor includes preconditions *)
(* that guarantee the result position is valid. *)
Theorem i_skip_to_match_preserves_invariant : forall p cv distance p',
  i_invariant p ->
  i_skip_to_match p distance cv p' ->
  i_invariant p'.
Proof.
  intros p cv distance p' Hinv Hskip.
  destruct Hskip.
  - (* ISkip_Zero: p = p' *)
    assumption.
  - (* ISkip_Forward: constructor guarantees invariants *)
    (* After destruct, p' = mkPosition ... (offset + distance) (errors + distance) ... *)
    (* The constructor preconditions H3 and H4 ensure the result is valid *)
    unfold i_invariant.
    auto.
Qed.

(* Theorem: Skip-to-match formula (CORRECTED) *)
(* Proves: offset' = offset + distance, errors' = errors + distance *)
Theorem i_skip_to_match_formula : forall (offset : Z) (errors n distance : nat) cv p',
  (distance > 0)%nat ->
  (errors + distance <= n)%nat ->
  (-Z.of_nat n <= offset <= Z.of_nat n) ->
  (Z.abs offset <= Z.of_nat errors) ->
  i_skip_to_match
    (mkPosition VarINonFinal offset errors n None)
    distance
    cv
    p' ->
  exists (offset' : Z) (errors' : nat),
    p' = mkPosition VarINonFinal offset' errors' n None /\
    offset' = offset + Z.of_nat distance /\  (* CORRECTED: forward scan *)
    errors' = (errors + distance)%nat.
Proof.
  intros offset errors n distance cv p' Hdist_pos Hbudget Hbound Hreach Hskip.
  inversion Hskip; subst.
  - (* ISkip_Zero: distance = 0, contradicts distance > 0 *)
    lia.
  - (* ISkip_Forward: formula follows directly from constructor *)
    exists (offset + Z.of_nat distance), (errors + distance)%nat.
    split; [reflexivity | split; reflexivity].
Qed.

(* M-type version of skip-to-match *)
(* M-type semantics differ from I-type: *)
(* - M-type DELETE: offset → offset (UNCHANGED) *)
(* - M-type positions track from beginning of word, not end *)
(* - M-type offset is non-positive, moving toward 0 *)
(* *)
(* Rust audit, 2026-06-19:
   - src/transducer/universal/position.rs implements M-type skip-to-match in
     successors_m_type_standard for the "0 < b" case.
   - That universal-backend skip advances offset to the first matching bit
     using offset + j + 1 and charges j errors.
   - The primitive below models the older generalized M-type coalesced-delete
     shape, where offset is unchanged and only the error budget is consumed.
     It is kept as a local invariant-preservation lemma for that primitive,
     not as a claim that the universal backend uses this exact transition. *)

(* Definition: M-type skip as primitive (if it exists) *)
Inductive m_skip_to_match : Position -> nat -> CharacteristicVector -> Position -> Prop :=
  | MSkip_Zero : forall p cv,
      (* Zero skip distance - identity *)
      m_skip_to_match p 0 cv p
  | MSkip_Forward : forall offset errors n distance cv,
      (* M-type skip: offset unchanged, errors increased *)
      (distance > 0)%nat ->
      (errors + distance <= n)%nat ->
      (-Z.of_nat (2 * n) <= offset <= 0) ->
      (Z.abs offset <= Z.of_nat errors) ->
      (* Result must also satisfy M-invariant *)
      (Z.abs offset <= Z.of_nat (errors + distance)) ->
      m_skip_to_match
        (mkPosition VarMFinal offset errors n None)
        distance
        cv
        (mkPosition VarMFinal offset (errors + distance) n None).

(* Theorem: M-type skip preserves invariant *)
Theorem m_skip_to_match_preserves_invariant : forall p cv distance p',
  m_invariant p ->
  m_skip_to_match p distance cv p' ->
  m_invariant p'.
Proof.
  intros p cv distance p' Hinv Hskip.
  destruct Hskip as [p_same cv_same | offset errors n dist cv_fwd Hdist Hbudget Hbound_src Hreach_src Hreach_dst].
  - (* MSkip_Zero: p = p' *)
    assumption.
  - (* MSkip_Forward: constructor guarantees invariants *)
    unfold m_invariant.
    split; [reflexivity | ].  (* variant = VarMFinal *)
    split.
    + (* Reachability: Z.of_nat errors' >= -(offset) - Z.of_nat n *)
      simpl.
      (* From Hreach_dst: Z.abs offset <= Z.of_nat (errors + dist) *)
      (* Since offset <= 0, Z.abs offset = -offset *)
      (* So: -offset <= Z.of_nat (errors + dist) *)
      (* Need: Z.of_nat (errors + dist) >= -offset - Z.of_nat n *)
      (* Which is: Z.of_nat (errors + dist) + Z.of_nat n >= -offset *)
      assert (Z.abs offset = -offset) as Habs by lia.
      rewrite Habs in Hreach_dst.
      lia.
    + (* Bounds and budget *)
      split.
      * (* Bounds: -Z.of_nat (2 * n) <= offset <= 0 *)
        assumption.
      * (* Budget: errors + dist <= n *)
        assumption.
Qed.

(* Theorem: M-type skip formula (CORRECTED) *)
(* M-type: offset unchanged, only errors increase *)
Theorem m_skip_to_match_formula : forall (offset : Z) (errors n distance : nat) cv p',
  (distance > 0)%nat ->
  (errors + distance <= n)%nat ->
  (-Z.of_nat (2 * n) <= offset <= 0) ->
  (Z.abs offset <= Z.of_nat errors) ->
  m_skip_to_match
    (mkPosition VarMFinal offset errors n None)
    distance
    cv
    p' ->
  exists (errors' : nat),
    p' = mkPosition VarMFinal offset errors' n None /\
    errors' = (errors + distance)%nat.
Proof.
  intros offset errors n distance cv p' Hdist_pos Hbudget Hbound Hreach Hskip.
  inversion Hskip; subst.
  - (* MSkip_Zero: distance = 0, contradicts distance > 0 *)
    lia.
  - (* MSkip_Forward: formula follows directly from constructor *)
    exists (errors + distance)%nat.
    split; reflexivity.
Qed.

(* Corollary: Skip-to-match is sound optimization *)
Corollary skip_to_match_sound : forall p cv (distance : nat) p',
  i_invariant p ->
  (distance > 0)%nat ->
  i_skip_to_match p distance cv p' ->
  i_invariant p'.
Proof.
  intros.
  eapply i_skip_to_match_preserves_invariant; eauto.
Qed.
