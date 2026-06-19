(*
 * Formal Specification of Phonetic Split Operations
 *
 * This module defines phonetic split operations as primitive operations
 * distinct from standard Levenshtein operations (Match, Delete, Insert, Substitute).
 *
 * Key insight: Like skip-to-match, phonetic splits are NOT decomposable into
 * standard operations. They represent a distinct class of multi-character
 * transformations with their own semantics.
 *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
From Stdlib Require Import Ascii.
Require Import Core.
Require Import Operations.
Require Import Transitions.

Open Scope Z_scope.

(*
 * Phonetic Split Lifecycle
 * ========================
 *
 * Phonetic splits are multi-step operations with three phases:
 *
 * 1. ENTRY: From regular I-type position, enter ISplitting state
 *    - Cost: Applied upon entry (typically fractional, truncates to 0)
 *    - Offset: Unchanged (like MATCH operations)
 *    - Entry character: Recorded for validation
 *
 * 2. PROGRESS: Remain in ISplitting state while consuming input
 *    - Validate that input matches target pattern (e.g., 'f' + 'ph')
 *    - No cost during progress
 *    - Position remains stable
 *
 * 3. COMPLETION: Exit to regular I-type or M-type position
 *    - Offset: Unchanged (advancement happens via sliding subword window)
 *    - Errors: Same as entry (cost was applied at entry)
 *    - Transition to I-type if within word, M-type if past word end
 *
 * Example: "graf" → "graphe" with f→ph split
 * -------
 *
 * Start: I+0#0 (processing 'gra')
 *
 * Step 1 (Entry): At position word[3]='f', input='p'
 *   - Check: Can apply f→ph phonetic split
 *   - Enter: ISplitting+0#0_f (offset unchanged, like MATCH)
 *   - Cost: 0 (fractional weight 0.15 truncates to 0)
 *
 * Step 2 (Progress): At ISplitting+0#0_f, input='h'
 *   - Check: 'f' + 'ph' matches pattern
 *   - Remain: ISplitting+0#0_f (consuming 'h')
 *
 * Step 3 (Completion): Complete split
 *   - Create: I+0#0 (offset unchanged)
 *   - Position is past end of word → converts to M-type
 *   - Result: M+0#0
 *   - Note: Advancement happens via sliding subword window
 *
 * Step 4 (Standard Operation): From M+0#0, input='e'
 *   - INSERT operation (cost 1)
 *   - Result: M+1#1
 *
 * Total cost: 0 (split) + 1 (insert) = 1 error
 *)

(*
 * Split Entry Relation
 * ====================
 *
 * Defines when an I-type position can enter a phonetic splitting state.
 *)

Inductive i_split_entry : Position -> ascii -> nat -> Position -> Prop :=
  | ISplitEntry : forall offset errors n entry_char split_cost,
      (* Preconditions *)
      (-Z.of_nat n <= offset <= Z.of_nat n) ->
      (Z.abs offset <= Z.of_nat errors) ->
      (errors <= n)%nat ->
      (errors + split_cost <= n)%nat ->  (* Budget check *)

      (* Transition: I-type → ISplitting *)
      (* UPDATED: offset unchanged (like MATCH operations) *)
      (* Previous preconditions (offset > -n, fractional cost constraint) *)
      (* are no longer needed since offset is not decremented *)
      i_split_entry
        (mkPosition VarINonFinal offset errors n None)
        entry_char
        split_cost
        (mkPosition VarISplitting offset (errors + split_cost) n (Some entry_char)).

(*
 * Split Progress Relation
 * =======================
 *
 * Defines progression within splitting state (consuming additional input).
 * Progress consumes one input character while retaining the splitting state;
 * entry and completion carry the operation identity and budget accounting.
 *)

Inductive i_split_progress : Position -> ascii -> Position -> Prop :=
  | ISplitProgress : forall offset errors n entry_char input_char,
      (* Remain in same splitting state *)
      i_split_progress
        (mkPosition VarISplitting offset errors n (Some entry_char))
        input_char
        (mkPosition VarISplitting offset errors n (Some entry_char)).

(*
 * Split Completion Relation
 * =========================
 *
 * Defines when a splitting state can exit to a regular position.
 *)

Inductive i_split_completion : Position -> nat -> Position -> Prop :=
  | ISplitComplete_ToI : forall offset errors n entry_char result_offset,
      (* Complete split, return to I-type *)
      (* UPDATED: offset unchanged (like MATCH operations) *)
      result_offset = offset ->
      (* Precondition: Result position must satisfy I-invariant *)
      (Z.abs result_offset <= Z.of_nat errors) ->
      (-Z.of_nat n <= result_offset <= Z.of_nat n) ->
      i_split_completion
        (mkPosition VarISplitting offset errors n (Some entry_char))
        n
        (mkPosition VarINonFinal result_offset errors n None)

  | ISplitComplete_ToM : forall offset errors n entry_char result_offset m_offset,
      (* Complete split, transition to M-type (past word end) *)
      (* UPDATED: offset unchanged (like MATCH operations) *)
      result_offset = offset ->
      m_offset = result_offset - Z.of_nat n ->
      (* Preconditions: Result position must satisfy M-invariant *)
      (-Z.of_nat (2 * n) <= m_offset <= 0) ->
      (Z.of_nat errors >= -m_offset - Z.of_nat n) ->
      i_split_completion
        (mkPosition VarISplitting offset errors n (Some entry_char))
        n
        (mkPosition VarMFinal m_offset errors n None).

(*
 * Complete Split Operation (Entry → Completion)
 * =============================================
 *
 * This relation combines entry and completion for simple splits
 * that don't require intermediate progress steps.
 *)

Inductive i_phonetic_split : Position -> ascii -> nat -> Position -> Prop :=
  | IPhoneticSplit : forall p1 p2 p3 entry_char split_cost n,
      i_split_entry p1 entry_char split_cost p2 ->
      i_split_completion p2 n p3 ->
      i_phonetic_split p1 entry_char split_cost p3.

(*
 * M-Type Phonetic Splits
 * ======================
 *
 * Similar definitions for M-type positions.
 * Simpler because M-type is already past word end.
 *)

Inductive m_split_entry : Position -> ascii -> nat -> Position -> Prop :=
  | MSplitEntry : forall offset errors n entry_char split_cost,
      (* Preconditions *)
      (-Z.of_nat (2 * n) <= offset <= 0) ->
      (Z.of_nat errors >= -offset - Z.of_nat n) ->
      (errors <= n)%nat ->
      (errors + split_cost <= n)%nat ->

      (* Transition: M-type → MSplitting *)
      (* UPDATED: offset unchanged (like MATCH operations) *)
      m_split_entry
        (mkPosition VarMFinal offset errors n None)
        entry_char
        split_cost
        (mkPosition VarMSplitting offset (errors + split_cost) n (Some entry_char)).

Inductive m_split_completion : Position -> Position -> Prop :=
  | MSplitComplete : forall offset errors n entry_char result_offset,
      (* UPDATED: offset unchanged (like MATCH operations) *)
      result_offset = offset ->
      (* Completion is allowed only when the resulting M-type position is valid. *)
      (Z.of_nat errors >= -result_offset - Z.of_nat n) ->
      (-Z.of_nat (2 * n) <= result_offset <= 0) ->
      m_split_completion
        (mkPosition VarMSplitting offset errors n (Some entry_char))
        (mkPosition VarMFinal result_offset errors n None).

Inductive m_phonetic_split : Position -> ascii -> nat -> Position -> Prop :=
  | MPhoneticSplit : forall p1 p2 p3 entry_char split_cost,
      m_split_entry p1 entry_char split_cost p2 ->
      m_split_completion p2 p3 ->
      m_phonetic_split p1 entry_char split_cost p3.

(*
 * Invariant Definitions for Splitting States
 * ==========================================
 *
 * PHASE 4 UPDATE: Relaxed invariants for intermediate splitting states.
 *
 * Splitting states are temporary (entry → progress → completion).
 * We use a relaxed reachability constraint: |offset| ≤ errors + 1
 *
 * Note: With the updated offset semantics (offset unchanged during entry/completion),
 * the standard invariant |offset| ≤ errors would also work. However, we keep
 * the relaxed version for additional safety margin and consistency with the
 * previously proven theorems.
 *
 * The +1 buffer provides flexibility for edge cases and potential future extensions.
 *)

Definition i_splitting_invariant (p : Position) : Prop :=
  variant p = VarISplitting /\
  let n := max_distance p in
  let offset := offset p in
  let errors := errors p in
  (* RELAXED: |offset| ≤ errors + 1 instead of |offset| ≤ errors *)
  Z.abs offset <= Z.of_nat errors + 1 /\
  -Z.of_nat n <= offset <= Z.of_nat n /\
  (errors <= n)%nat.

Definition m_splitting_invariant (p : Position) : Prop :=
  variant p = VarMSplitting /\
  let n := max_distance p in
  let offset := offset p in
  let errors := errors p in
  (* RELAXED: errors + 1 instead of errors *)
  Z.of_nat errors + 1 >= -offset - Z.of_nat n /\
  -Z.of_nat (2 * n) <= offset <= 0 /\
  (errors <= n)%nat.

(*
 * Theorems: Invariant Preservation
 * ================================
 *)

(* Theorem: Entering split preserves I-type invariant *)
(* UPDATED: Simpler proof with unchanged offset semantics *)
Theorem i_split_entry_preserves_invariant : forall p entry_char cost p',
  i_invariant p ->
  i_split_entry p entry_char cost p' ->
  i_splitting_invariant p'.
Proof.
  intros p entry_char cost p' Hinv Hentry.
  inversion Hentry; subst.
  unfold i_splitting_invariant. simpl.
  split; [reflexivity | ].
  split.
  - (* Reachability: |offset| ≤ errors + cost + 1 (relaxed) *)
    (* From H0: |offset| ≤ errors (source I-type invariant) *)
    (* Since cost ≥ 0, we have errors ≤ errors + cost ≤ errors + cost + 1 *)
    (* Therefore: |offset| ≤ errors ≤ errors + cost + 1 *)
    lia.
  - split.
    + (* Bounds: -n ≤ offset ≤ n *)
      (* From H: precondition of entry *)
      assumption.
    + (* Budget: errors + cost ≤ n *)
      assumption.
Qed.

(* Theorem: Completing split preserves invariant *)
Theorem i_split_completion_preserves_invariant : forall p n p',
  i_splitting_invariant p ->
  i_split_completion p n p' ->
  (i_invariant p' \/ m_invariant p').
Proof.
  intros p n' p' Hinv Hcompletion.
  inversion Hcompletion; subst.
  - (* Case: Complete to I-type *)
    left.
    unfold i_invariant. simpl.
    unfold i_splitting_invariant in Hinv.
    destruct Hinv as [_ [_ [_ Hbudget]]].
    split; [reflexivity | ].
    split.
    + (* Reachability: |result_offset| ≤ errors *)
      (* From H0: precondition of completion *)
      assumption.
    + split.
      * (* Bounds: -n ≤ result_offset ≤ n *)
        (* From H1: precondition of completion *)
        assumption.
      * (* Budget: errors ≤ n *)
        assumption.
  - (* Case: Complete to M-type *)
    right.
    unfold m_invariant. simpl.
    unfold i_splitting_invariant in Hinv.
    destruct Hinv as [_ [_ [_ Hbudget]]].
    split; [reflexivity | ].
    split.
    + (* M-type reachability *)
      (* From H2: precondition of completion *)
      assumption.
    + split.
      * (* M-type bounds *)
        (* From H1: precondition of completion *)
        assumption.
      * (* Budget *)
        assumption.
Qed.

(* Theorem: Phonetic split preserves invariants *)
Theorem i_phonetic_split_preserves_invariant : forall p entry_char cost p',
  i_invariant p ->
  i_phonetic_split p entry_char cost p' ->
  (i_invariant p' \/ m_invariant p').
Proof.
  intros p entry_char cost p' Hinv Hsplit.
  inversion Hsplit; subst.
  (* Entry preserves invariant *)
  assert (Hsplitting: i_splitting_invariant p2) by
    (eapply i_split_entry_preserves_invariant; eauto).
  (* Completion preserves invariant *)
  eapply i_split_completion_preserves_invariant; eauto.
Qed.

(* Theorem: Entering an M-type split preserves the relaxed splitting invariant *)
Theorem m_split_entry_preserves_invariant : forall p entry_char cost p',
  m_invariant p ->
  m_split_entry p entry_char cost p' ->
  m_splitting_invariant p'.
Proof.
  intros p entry_char cost p' Hinv Hentry.
  inversion Hentry; subst.
  unfold m_splitting_invariant. simpl.
  split; [reflexivity | ].
  split.
  - (* Reachability: source reachability remains true after adding split cost. *)
    lia.
  - split.
    + assumption.
    + assumption.
Qed.

(* Theorem: Completing an M-type split restores the ordinary M invariant *)
Theorem m_split_completion_preserves_invariant : forall p p',
  m_splitting_invariant p ->
  m_split_completion p p' ->
  m_invariant p'.
Proof.
  intros p p' Hinv Hcompletion.
  inversion Hcompletion; subst.
  unfold m_splitting_invariant in Hinv.
  destruct Hinv as [_ [_ [_ Hbudget]]].
  unfold m_invariant. simpl.
  split; [reflexivity | ].
  split.
  - assumption.
  - split; [assumption | exact Hbudget].
Qed.

(* Theorem: M-type phonetic split preserves invariants *)
Theorem m_phonetic_split_preserves_invariant : forall p entry_char cost p',
  m_invariant p ->
  m_phonetic_split p entry_char cost p' ->
  m_invariant p'.
Proof.
  intros p entry_char cost p' Hinv Hsplit.
  inversion Hsplit; subst.
  assert (Hsplitting: m_splitting_invariant p2) by
    (eapply m_split_entry_preserves_invariant; eauto).
  eapply m_split_completion_preserves_invariant; eauto.
Qed.

(*
 * Theorems: Composition Properties
 * ================================
 *)

(* Phonetic I-splits compose with ordinary I-successor transitions. *)
Theorem i_phonetic_split_composes_with_i_successor : forall p entry_char cost p_split op cv p',
  i_invariant p ->
  i_phonetic_split p entry_char cost p_split ->
  i_successor p_split op cv p' ->
  i_invariant p'.
Proof.
  intros p entry_char cost p_split op cv p' Hinv Hsplit Hsucc.
  pose proof (i_phonetic_split_preserves_invariant
                p entry_char cost p_split Hinv Hsplit) as Hsplit_inv.
  destruct Hsplit_inv as [Hi | Hm].
  - eapply i_successor_preserves_invariant; eauto.
  - unfold m_invariant in Hm.
    destruct Hm as [Hvariant _].
    inversion Hsucc; subst; simpl in Hvariant; discriminate.
Qed.

(* Phonetic M-splits compose with ordinary M-successor transitions. *)
Theorem m_phonetic_split_composes_with_m_successor : forall p entry_char cost p_split op cv p',
  m_invariant p ->
  m_phonetic_split p entry_char cost p_split ->
  m_successor p_split op cv p' ->
  m_invariant p'.
Proof.
  intros p entry_char cost p_split op cv p' Hinv Hsplit Hsucc.
  pose proof (m_phonetic_split_preserves_invariant
                p entry_char cost p_split Hinv Hsplit) as Hsplit_inv.
  eapply m_successor_preserves_invariant; eauto.
Qed.

(* Consecutive I-type phonetic splits preserve invariants when the intermediate
   split completes back to I-type, which is the only state where another I-split
   can start. *)
Theorem consecutive_i_phonetic_splits_preserve_invariant :
  forall p entry1 cost1 p_mid entry2 cost2 p',
    i_invariant p ->
    i_phonetic_split p entry1 cost1 p_mid ->
    i_invariant p_mid ->
    i_phonetic_split p_mid entry2 cost2 p' ->
    (i_invariant p' \/ m_invariant p').
Proof.
  intros p entry1 cost1 p_mid entry2 cost2 p' Hinv Hsplit1 Hmid Hsplit2.
  eapply i_phonetic_split_preserves_invariant; eauto.
Qed.

(* Consecutive M-type phonetic splits preserve the M invariant. *)
Theorem consecutive_m_phonetic_splits_preserve_invariant :
  forall p entry1 cost1 p_mid entry2 cost2 p',
    m_invariant p ->
    m_phonetic_split p entry1 cost1 p_mid ->
    m_phonetic_split p_mid entry2 cost2 p' ->
    m_invariant p'.
Proof.
  intros p entry1 cost1 p_mid entry2 cost2 p' Hinv Hsplit1 Hsplit2.
  assert (Hmid: m_invariant p_mid) by
    (eapply m_phonetic_split_preserves_invariant; eauto).
  eapply m_phonetic_split_preserves_invariant; eauto.
Qed.

(* Cost accounting for a complete I-type phonetic split. *)
Theorem i_phonetic_split_cost_correct : forall p entry_char cost p',
  i_phonetic_split p entry_char cost p' ->
  errors p' = (errors p + cost)%nat.
Proof.
  intros p entry_char cost p' Hsplit.
  destruct Hsplit as [p1 p2 p3 entry split_cost n Hentry Hcompletion].
  inversion Hentry; subst.
  inversion Hcompletion; subst; reflexivity.
Qed.

(* Cost accounting for a complete M-type phonetic split. *)
Theorem m_phonetic_split_cost_correct : forall p entry_char cost p',
  m_phonetic_split p entry_char cost p' ->
  errors p' = (errors p + cost)%nat.
Proof.
  intros p entry_char cost p' Hsplit.
  destruct Hsplit as [p1 p2 p3 entry split_cost Hentry Hcompletion].
  inversion Hentry; subst.
  inversion Hcompletion; subst; reflexivity.
Qed.

(* Additive accounting across an I-type phonetic split followed by a standard
   transition. *)
Theorem i_phonetic_split_then_i_successor_cost_correct :
  forall p entry_char cost p_split op cv p',
    i_phonetic_split p entry_char cost p_split ->
    i_successor p_split op cv p' ->
    errors p' = (errors p + cost + operation_cost op)%nat.
Proof.
  intros p entry_char cost p_split op cv p' Hsplit Hsucc.
  rewrite (i_successor_cost_correct p_split op cv p' Hsucc).
  rewrite (i_phonetic_split_cost_correct p entry_char cost p_split Hsplit).
  lia.
Qed.

(* Additive accounting across an M-type phonetic split followed by a standard
   transition. *)
Theorem m_phonetic_split_then_m_successor_cost_correct :
  forall p entry_char cost p_split op cv p',
    m_phonetic_split p entry_char cost p_split ->
    m_successor p_split op cv p' ->
    errors p' = (errors p + cost + operation_cost op)%nat.
Proof.
  intros p entry_char cost p_split op cv p' Hsplit Hsucc.
  rewrite (m_successor_cost_correct p_split op cv p' Hsucc).
  rewrite (m_phonetic_split_cost_correct p entry_char cost p_split Hsplit).
  lia.
Qed.
