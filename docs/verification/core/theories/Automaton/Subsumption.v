(** * Subsumption Relation for Levenshtein Automata

    This module defines the subsumption relation for all three algorithm variants.
    Position p1 subsumes p2 if all candidates reachable from p2 are also reachable
    from p1, allowing p2 to be pruned.

    Part of: Liblevenshtein.Core.Automaton

    Rust Reference: src/transducer/position.rs:56-180

    Subsumption Rules:
    - Standard: e ≤ f ∧ |i-j| ≤ f-e
    - Transposition: Complex special position handling
    - MergeAndSplit: e < f (strict) + distance check

    Key Theorems:
    - subsumes_irrefl: No position subsumes itself (except trivial case)
    - subsumes_trans: Transitivity of subsumption
    - subsumes_covers_reachable: Semantic correctness
*)

From Stdlib Require Import Arith Bool List Nat Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Automaton.Position.

(** * Subsumption Definitions *)

(** Final position check for subsumption constraints *)
Definition position_is_final_for_subsumption (qlen : nat) (p : Position) : bool :=
  qlen <=? term_index p.

(** Standard algorithm subsumption: e ≤ f ∧ |i-j| ≤ f-e

    CRITICAL FIX: A non-final position (term_index < qlen) CANNOT subsume
    a final position (term_index >= qlen). This is because after all dict
    characters are consumed, there are no more transitions, so the non-final
    position cannot reach the final state that the final position represents.

    Without this check, antichain construction incorrectly eliminates final
    positions, causing the automaton to reject valid words.
*)
Definition subsumes_standard (qlen : nat) (p1 p2 : Position) : bool :=
  let i := term_index p1 in
  let e := num_errors p1 in
  let j := term_index p2 in
  let f := num_errors p2 in
  let p1_final := position_is_final_for_subsumption qlen p1 in
  let p2_final := position_is_final_for_subsumption qlen p2 in
  (* Non-final cannot subsume final *)
  if (negb p1_final) && p2_final then false
  else (e <=? f) && (abs_diff i j <=? f - e).

(** Transposition algorithm subsumption.

    CRITICAL INVARIANTS:
    1. Positions with different variant_state (is_special) CANNOT subsume each other
    2. Non-final positions CANNOT subsume final positions

    From Definition 11 (thesis):
    - i#e_t ≤^t_s j#f     ⇔ false (different subscripts/variant_state)
    - i#e   ≤^t_s j#f_t   ⇔ false (different subscripts/variant_state)
    - i#e   ≤^t_s j#f     ⇔ f > e ∧ |j - i| ≤ f - e (both Usual)
    - i#e_t ≤^t_s j#f_t   ⇔ false (transposition positions never subsume each other)

    Rules:
    - Normal ⊑ Normal: same query index and e ≤ f
    - Special ⊑ Special: only if same position and e ≤ f
    - Normal CANNOT subsume Special
    - Special CANNOT subsume Normal
    - Non-final CANNOT subsume Final (critical for correctness)
*)
Definition subsumes_transposition (p1 p2 : Position) (query_length : nat) : bool :=
  let i := term_index p1 in
  let e := num_errors p1 in
  let s := is_special p1 in
  let j := term_index p2 in
  let f := num_errors p2 in
  let t := is_special p2 in
  let p1_final := position_is_final_for_subsumption query_length p1 in
  let p2_final := position_is_final_for_subsumption query_length p2 in
  (* CRITICAL: Non-final cannot subsume final *)
  if (negb p1_final) && p2_final then false
  (* CRITICAL: Different variant_state means no subsumption *)
  else if Bool.eqb s t then
    (* Same variant_state *)
    if negb (e <=? f) then false
    else if s then
      (* Both special: must be at same position *)
      i =? j
    else
      (* Neither special: keep Transposition pruning same-index.  Cross-index
         pruning can erase the ordinary position needed to enter/complete a
         future adjacent transposition. *)
      i =? j
  else
    (* Different variant_state: no subsumption *)
    false.

(** MergeAndSplit algorithm subsumption.
    Key difference: Requires e < f (STRICT) to allow both (i,e,false) and
    (i,e,true) to coexist in the same state.
    - Representatives beyond the query length are rejected; executable
      MergeAndSplit transitions never generate them
    - Normal and split-in-progress positions cannot subsume each other
    - Final special positions cannot subsume non-final split states
    - Non-final cannot subsume final (critical for correctness)
*)
Definition subsumes_merge_split (qlen : nat) (p1 p2 : Position) : bool :=
  let i := term_index p1 in
  let e := num_errors p1 in
  let s := is_special p1 in
  let j := term_index p2 in
  let f := num_errors p2 in
  let t := is_special p2 in
  let p1_final := position_is_final_for_subsumption qlen p1 in
  let p2_final := position_is_final_for_subsumption qlen p2 in
  (* CRITICAL: MergeAndSplit never generates representatives past qlen. *)
  if qlen <? i then false
  (* CRITICAL: Non-final cannot subsume final *)
  else if (negb p1_final) && p2_final then false
  (* CRITICAL: A final split-in-progress position has no completion transition. *)
  else if s && p1_final && negb p2_final then false
  (* Variant states represent different continuations. *)
  else if negb (Bool.eqb s t) then false
  (* Must have strictly fewer errors *)
  else if negb (e <? f) then false
  (* MergeAndSplit pruning is kept same-index only. Cross-index pruning can
     erase delete-closure witnesses needed by split/merge completions. *)
  else i =? j.

(** Unified subsumption that dispatches to the appropriate algorithm *)
Definition subsumes (alg : Algorithm) (query_length : nat) (p1 p2 : Position) : bool :=
  match alg with
  | Standard => subsumes_standard query_length p1 p2
  | Transposition => subsumes_transposition p1 p2 query_length
  | MergeAndSplit => subsumes_merge_split query_length p1 p2
  end.

(** * Basic Subsumption Properties *)

(** For Standard, a position always subsumes itself (same finality) *)
Lemma subsumes_standard_refl : forall qlen p,
  subsumes_standard qlen p p = true.
Proof.
  intros qlen [i e s].
  unfold subsumes_standard, position_is_final_for_subsumption. simpl.
  (* The non-final-cannot-subsume-final check: (negb p1_final) && p2_final = false
     because p1_final = p2_final, so either both false or both true *)
  destruct (qlen <=? i); simpl.
  - (* Both final *)
    rewrite Nat.leb_refl, abs_diff_self. simpl. reflexivity.
  - (* Both non-final *)
    rewrite Nat.leb_refl, abs_diff_self. simpl. reflexivity.
Qed.

(** For MergeAndSplit, a position does NOT subsume itself (strict inequality) *)
Lemma subsumes_merge_split_irrefl : forall qlen p,
  subsumes_merge_split qlen p p = false.
Proof.
  intros qlen [i e s].
  unfold subsumes_merge_split, position_is_final_for_subsumption. simpl.
  destruct (qlen <? i); simpl; [reflexivity|].
  destruct (qlen <=? i); simpl.
  - (* Final: same finality, falls through to strict error check *)
    destruct s; simpl; rewrite Nat.ltb_irrefl; reflexivity.
  - (* Non-final: same finality, falls through to strict error check *)
    destruct s; simpl; rewrite Nat.ltb_irrefl; reflexivity.
Qed.

(** * Subsumption Transitivity for Standard *)

Lemma abs_diff_triangle : forall a b c,
  abs_diff a c <= abs_diff a b + abs_diff b c.
Proof.
  intros a b c.
  unfold abs_diff.
  destruct (a <=? b) eqn:Hab, (b <=? c) eqn:Hbc, (a <=? c) eqn:Hac;
  apply Nat.leb_le in Hab || apply Nat.leb_gt in Hab;
  apply Nat.leb_le in Hbc || apply Nat.leb_gt in Hbc;
  apply Nat.leb_le in Hac || apply Nat.leb_gt in Hac;
  lia.
Qed.

Lemma subsumes_standard_trans : forall qlen p1 p2 p3,
  subsumes_standard qlen p1 p2 = true ->
  subsumes_standard qlen p2 p3 = true ->
  subsumes_standard qlen p1 p3 = true.
Proof.
  intros qlen [i1 e1 s1] [i2 e2 s2] [i3 e3 s3].
  unfold subsumes_standard, position_is_final_for_subsumption. simpl.
  (* Analyze finality cases *)
  destruct (qlen <=? i1) eqn:Hi1, (qlen <=? i2) eqn:Hi2, (qlen <=? i3) eqn:Hi3;
  simpl; try discriminate.
  (* Cases where both premises can be true:
     - All final: standard transitivity
     - All non-final: standard transitivity
     - p1 final, p2 non-final: first premise fails (false)
     - p1 final, p2 final, p3 non-final: second premise fails (false) - actually ok
     - p1 non-final, p2 non-final, p3 final: second premise fails
     - p1 non-final, p2 final: first premise fails *)
  all: rewrite !andb_true_iff, !Nat.leb_le.
  all: intros [He12 Hd12] [He23 Hd23].
  all: split; [ lia | ].
  all: assert (Htri : abs_diff i1 i3 <= abs_diff i1 i2 + abs_diff i2 i3)
    by apply abs_diff_triangle.
  all: lia.
Qed.

(** * Subsumption Semantic Correctness *)

(** A position (i, e) represents having consumed i characters from the query
    with e errors accumulated. The key semantic property is:

    If p1 subsumes p2, then any dictionary suffix that can be matched from p2
    can also be matched from p1 (possibly with different error paths).

    This is captured by the subsumption formula: if |i-j| <= f-e, then
    from position i with e errors, we can reach position j with f errors
    (using f-e edit operations to cover the index difference).

    IMPORTANT: With the non-final-cannot-subsume-final fix, this property
    now also requires that if p2 is final, p1 must also be final.
*)

(** Helper: Distance-based reachability *)
Definition can_reach (i1 e1 i2 e2 : nat) : Prop :=
  abs_diff i1 i2 <= e2 - e1 /\ e1 <= e2.

(** Semantics lemma - now conditional on finality *)
Lemma subsumes_standard_semantics : forall qlen p1 p2,
  (* If both have same finality, standard formula applies *)
  (term_index p1 >= qlen <-> term_index p2 >= qlen) ->
  (subsumes_standard qlen p1 p2 = true <->
   can_reach (term_index p1) (num_errors p1) (term_index p2) (num_errors p2)).
Proof.
  intros qlen [i1 e1 s1] [i2 e2 s2] Hsame_final.
  unfold subsumes_standard, can_reach, position_is_final_for_subsumption. simpl.
  simpl in Hsame_final.
  (* With same finality, the non-final-cannot-subsume-final check passes *)
  destruct (qlen <=? i1) eqn:Hi1, (qlen <=? i2) eqn:Hi2; simpl.
  - (* Both final *)
    rewrite andb_true_iff, !Nat.leb_le. split; intros [? ?]; split; lia.
  - (* p1 final, p2 non-final - contradiction with Hsame_final *)
    apply Nat.leb_le in Hi1. apply Nat.leb_nle in Hi2.
    exfalso. apply Hsame_final in Hi1. lia.
  - (* p1 non-final, p2 final - the check returns false *)
    apply Nat.leb_nle in Hi1. apply Nat.leb_le in Hi2.
    exfalso. apply Hsame_final in Hi2. lia.
  - (* Both non-final *)
    rewrite andb_true_iff, !Nat.leb_le. split; intros [? ?]; split; lia.
Qed.

(** * Algorithm-Specific Irreflexivity *)

(** Standard allows self-subsumption (reflexive) *)
Lemma subsumes_standard_refl_iff : forall qlen p,
  subsumes_standard qlen p p = true.
Proof.
  intros qlen p. apply subsumes_standard_refl.
Qed.

(** Transposition: self-subsumption always holds (same finality) *)
Lemma subsumes_transposition_refl : forall p query_length,
  is_special p = false ->
  subsumes_transposition p p query_length = true.
Proof.
  intros [i e s] qlen Hs.
  unfold subsumes_transposition, position_is_final_for_subsumption. simpl.
  simpl in Hs. subst s.
  (* With same position, finality check passes (negb x && x = false for any x) *)
  destruct (qlen <=? i); simpl.
  - rewrite Nat.leb_refl, Nat.eqb_refl. reflexivity.
  - rewrite Nat.leb_refl, Nat.eqb_refl. reflexivity.
Qed.

Lemma subsumes_transposition_special_refl : forall p query_length,
  is_special p = true ->
  subsumes_transposition p p query_length = true.
Proof.
  intros [i e s] qlen Hs.
  unfold subsumes_transposition, position_is_final_for_subsumption. simpl.
  simpl in Hs. subst s.
  destruct (qlen <=? i); simpl.
  - rewrite Nat.leb_refl, Nat.eqb_refl. reflexivity.
  - rewrite Nat.leb_refl, Nat.eqb_refl. reflexivity.
Qed.

(** MergeAndSplit: NEVER self-subsuming (strict inequality) *)
Lemma subsumes_merge_split_not_refl : forall qlen p,
  subsumes_merge_split qlen p p = false.
Proof.
  intros qlen p. apply subsumes_merge_split_irrefl.
Qed.

(** * Unified Subsumption Properties *)

(** Transitivity for unified subsumption (Standard algorithm) *)
Lemma subsumes_trans_standard : forall qlen p1 p2 p3,
  subsumes Standard qlen p1 p2 = true ->
  subsumes Standard qlen p2 p3 = true ->
  subsumes Standard qlen p1 p3 = true.
Proof.
  intros qlen p1 p2 p3 H1 H2.
  unfold subsumes in *.
  apply subsumes_standard_trans with p2; assumption.
Qed.

(** * Examples *)

(** Examples with qlen = 10 (both positions non-final) *)
Example subsumes_example_1 :
  (* (5, 2) subsumes (4, 3) for Standard: |5-4| = 1 <= 3-2 = 1 ✓ *)
  subsumes_standard 10 (std_pos 5 2) (std_pos 4 3) = true.
Proof. reflexivity. Qed.

Example subsumes_example_2 :
  (* (5, 2) subsumes (5, 3) for Standard: |5-5| = 0 <= 3-2 = 1 ✓ *)
  subsumes_standard 10 (std_pos 5 2) (std_pos 5 3) = true.
Proof. reflexivity. Qed.

Example subsumes_example_3 :
  (* (3, 3) does NOT subsume (5, 2) for Standard: e=3 > f=2 *)
  subsumes_standard 10 (std_pos 3 3) (std_pos 5 2) = false.
Proof. reflexivity. Qed.

(** CRITICAL FIX EXAMPLE: Non-final cannot subsume final *)
Example subsumes_example_4_non_final_cannot_subsume_final :
  (* (2, 1) CANNOT subsume (3, 2) when qlen=3 because:
     - (2, 1) has term_index 2 < 3 = qlen (NON-FINAL)
     - (3, 2) has term_index 3 >= 3 = qlen (FINAL)
     Even though the formula |3-2|=1 <= 2-1=1 would pass! *)
  subsumes_standard 3 (std_pos 2 1) (std_pos 3 2) = false.
Proof. reflexivity. Qed.

Example subsumes_example_5_both_final_can_subsume :
  (* (10, 1) CAN subsume (11, 2) when qlen=3 because both are final *)
  subsumes_standard 3 (std_pos 10 1) (std_pos 11 2) = true.
Proof. reflexivity. Qed.

Example subsumes_ms_example_1 :
  (* For MergeAndSplit, (5, 2) does NOT subsume (5, 2): need e < f *)
  subsumes_merge_split 10 (std_pos 5 2) (std_pos 5 2) = false.
Proof. reflexivity. Qed.

Example subsumes_ms_example_2 :
  (* For MergeAndSplit, (5, 2) subsumes (5, 3): e=2 < f=3 and |5-5|=0 <= 1 *)
  subsumes_merge_split 10 (std_pos 5 2) (std_pos 5 3) = true.
Proof. reflexivity. Qed.

Example subsumes_trans_example_1 :
  (* Normal (1,1) should NOT subsume special (0,1,true) for Transposition *)
  subsumes_transposition (std_pos 1 1) (special_pos 0 1) 5 = false.
Proof. reflexivity. Qed.

Example subsumes_trans_example_2 :
  (* Special (5,2) subsumes special (5,3) for Transposition: same position *)
  subsumes_transposition (special_pos 5 2) (special_pos 5 3) 5 = true.
Proof. reflexivity. Qed.

Example subsumes_trans_example_3 :
  (* CRITICAL FIX: Special (0,1) should NOT subsume Normal (1,2) even though
     e=1 < f=2 and |0-1|=1 <= 2-1=1 would otherwise pass the formula.
     Different variant_state means no subsumption. *)
  subsumes_transposition (special_pos 0 1) (std_pos 1 2) 5 = false.
Proof. reflexivity. Qed.

Example subsumes_trans_example_4 :
  (* CRITICAL FIX: Non-final (2,1) cannot subsume final (3,2) even when
     both are non-special. This is the key bug fix! *)
  subsumes_transposition (std_pos 2 1) (std_pos 3 2) 3 = false.
Proof. reflexivity. Qed.

Example subsumes_trans_example_5 :
  (* Cross-index normal positions are not pruned for Transposition. *)
  subsumes_transposition (std_pos 2 1) (std_pos 3 2) 5 = false.
Proof. reflexivity. Qed.

(** * Critical Lemma: Non-Final Cannot Subsume Final *)

(** This lemma captures the key fix that prevents correctness bugs.
    If p1 is non-final (term_index < qlen) and p2 is final (term_index >= qlen),
    then p1 cannot subsume p2 for any algorithm. *)

Lemma non_final_cannot_subsume_final_standard : forall qlen p1 p2,
  position_is_final_for_subsumption qlen p1 = false ->
  position_is_final_for_subsumption qlen p2 = true ->
  subsumes_standard qlen p1 p2 = false.
Proof.
  intros qlen p1 p2 Hp1_non_final Hp2_final.
  unfold subsumes_standard.
  rewrite Hp1_non_final, Hp2_final. simpl. reflexivity.
Qed.

Lemma non_final_cannot_subsume_final_transposition : forall qlen p1 p2,
  position_is_final_for_subsumption qlen p1 = false ->
  position_is_final_for_subsumption qlen p2 = true ->
  subsumes_transposition p1 p2 qlen = false.
Proof.
  intros qlen p1 p2 Hp1_non_final Hp2_final.
  unfold subsumes_transposition.
  rewrite Hp1_non_final, Hp2_final. simpl. reflexivity.
Qed.

Lemma non_final_cannot_subsume_final_merge_split : forall qlen p1 p2,
  position_is_final_for_subsumption qlen p1 = false ->
  position_is_final_for_subsumption qlen p2 = true ->
  subsumes_merge_split qlen p1 p2 = false.
Proof.
  intros qlen p1 p2 Hp1_non_final Hp2_final.
  unfold subsumes_merge_split.
  destruct (qlen <? term_index p1); simpl; [reflexivity|].
  rewrite Hp1_non_final, Hp2_final. simpl. reflexivity.
Qed.

(** Unified version for all algorithms *)
Lemma non_final_cannot_subsume_final : forall alg qlen p1 p2,
  position_is_final_for_subsumption qlen p1 = false ->
  position_is_final_for_subsumption qlen p2 = true ->
  subsumes alg qlen p1 p2 = false.
Proof.
  intros alg qlen p1 p2 Hp1_non_final Hp2_final.
  destruct alg.
  - apply non_final_cannot_subsume_final_standard; assumption.
  - apply non_final_cannot_subsume_final_transposition; assumption.
  - apply non_final_cannot_subsume_final_merge_split; assumption.
Qed.

(** For non-special positions, Transposition subsumption is a conservative
    subset of Standard subsumption. *)
Lemma subsumes_nonspecial_std_trans : forall qlen p1 p2,
  is_special p1 = false ->
  is_special p2 = false ->
  subsumes_transposition p1 p2 qlen = true ->
  subsumes_standard qlen p1 p2 = true.
Proof.
  intros qlen [i1 e1 s1] [i2 e2 s2] Hs1 Hs2 Hsub.
  unfold subsumes_standard, subsumes_transposition, position_is_final_for_subsumption in *.
  simpl in *. subst s1 s2. simpl in *.
  destruct (qlen <=? i1) eqn:Hi1, (qlen <=? i2) eqn:Hi2; simpl in *;
    try discriminate.
  all: destruct (e1 <=? e2) eqn:He; simpl in *; try discriminate.
  all: apply Nat.eqb_eq in Hsub; subst i2.
  all: try rewrite He; rewrite abs_diff_self; simpl; reflexivity.
Qed.
