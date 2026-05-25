(** * Merge-Split Distance

    This module defines the merge-split distance function, which extends
    standard Levenshtein distance with merge and split operations.

    Part of: Liblevenshtein.Core

    Operations:
    - Insertion: cost 1
    - Deletion: cost 1
    - Substitution: cost 1 (0 for match)
    - Merge: cost 1 (two adjacent source chars → one target char)
    - Split: cost 1 (one source char → two target chars)

    Design: AXIOM-FREE generic semantics.
    The merge/split predicates are decidable boolean functions that accept
    every 2-to-1 merge and every 1-to-2 split, matching the Rust executable.

    Key property: merge_split_distance(s1, s2) <= lev_distance(s1, s2)

    ** PROOF STATUS SUMMARY (Dec 2025) **

    COMPLETED (Qed):
    - ms_le_standard      : merge_split_distance <= lev_distance
    - ms_sym              : Symmetry
    - ms_length_diff_lower: Lower bound by length difference
    - ms_empty_left/right : Base cases for empty strings
    - ms_single           : Single character case
    - ms_seq_exists       : Optimal edit sequences exist
    - ms_upper_bound      : Distance <= cost of any valid sequence
    - min6_swap_12_45     : Helper for symmetry

    TRIANGLE INEQUALITY:
    - ms_triangle_via_trace is provided by Composition/MergeSplitComposition.v

    REMOVED:
    - ms_seq_compose      : Sequence composition (model limitation - deleted)
    - ms_triangle_via_seq : Depended on ms_seq_compose (deleted)
    - ms_eq_lev_when_no_merge_split : FALSE - double-subst optimization
      Correct relationship: merge_split_distance <= lev_distance (proven)
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia Wf_nat.
From Stdlib Require Import Recdef.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.MetricProperties.

(** * Merge/Split Predicate Definitions (Generic)

    The Rust [merge_and_split_distance] implementation treats every 2-to-1
    merge and every 1-to-2 split as an available edit operation with cost 1.
    These predicates preserve the older proof interface while making that
    generic executable semantics explicit in Coq.
*)

(** Can characters c1, c2 merge to form target character d?
    Generic merge/split semantics accepts any such triple. *)
Definition can_merge (c1 c2 d : Char) : bool :=
  true.

(** Can character c split into target characters d1, d2?
    Generic merge/split semantics accepts any such triple. *)
Definition can_split (c d1 d2 : Char) : bool :=
  true.

(** * Helper: Check if merge or split applies and compute cost *)

(** Merge cost under generic semantics. *)
Definition merge_cost (c1 c2 d : Char) : nat :=
  if can_merge c1 c2 d then 1 else 100.

(** Split cost under generic semantics. *)
Definition split_cost (c d1 d2 : Char) : nat :=
  if can_split c d1 d2 then 1 else 100.

(** * Minimum Helpers *)

Definition min5 (a b c d e : nat) : nat :=
  min (min (min a b) (min c d)) e.

Definition min6 (a b c d e f : nat) : nat :=
  min (min5 a b c d e) f.

(** * Measure for Termination *)

Definition ms_measure (p : list Char * list Char) : nat :=
  length (fst p) + length (snd p).

(** * Main Definition using Function

    The merge-split distance extends Levenshtein with:
    - Merge: c1::c2::rest → d::rest' costs 1 if can_merge c1 c2 d
    - Split: c::rest → d1::d2::rest' costs 1 if can_split c d1 d2
*)
Function merge_split_pair (p : list Char * list Char) {measure ms_measure p} : nat :=
  match p with
  | ([], s2) => length s2
  | (s1, []) => length s1
  | ([c1], [d1]) =>
      if char_eq c1 d1 then 0 else 1
  | ([c1], d1 :: d2 :: s2') =>
      (* One source char, 2+ target chars - potential split *)
      let standard := min3
        (merge_split_pair ([], d1 :: d2 :: s2') + 1)       (* delete c1 *)
        (merge_split_pair ([c1], d2 :: s2') + 1)           (* insert d1 *)
        (merge_split_pair ([], d2 :: s2') + subst_cost c1 d1) in  (* subst *)
      min standard (merge_split_pair ([], s2') + split_cost c1 d1 d2)  (* split: c1 → d1, d2 *)
  | (c1 :: c2 :: s1', [d1]) =>
      (* 2+ source chars, one target char - potential merge *)
      let standard := min3
        (merge_split_pair (c2 :: s1', [d1]) + 1)              (* delete c1 *)
        (merge_split_pair (c1 :: c2 :: s1', []) + 1)          (* insert d1 *)
        (merge_split_pair (c2 :: s1', []) + subst_cost c1 d1) in (* subst *)
      min standard (merge_split_pair (s1', []) + merge_cost c1 c2 d1)  (* merge *)
  | (c1 :: c2 :: s1', d1 :: d2 :: s2') =>
      (* Both 2+ chars - merge and split both possible *)
      min6 (merge_split_pair (c2 :: s1', d1 :: d2 :: s2') + 1)   (* delete c1 *)
           (merge_split_pair (c1 :: c2 :: s1', d2 :: s2') + 1)   (* insert d1 *)
           (merge_split_pair (c2 :: s1', d2 :: s2') + subst_cost c1 d1)  (* subst c1→d1 *)
           (merge_split_pair (s1', d2 :: s2') + merge_cost c1 c2 d1)  (* merge: c1c2 → d1, consume c1,c2,d1 *)
           (merge_split_pair (c2 :: s1', s2') + split_cost c1 d1 d2)  (* split: c1 → d1,d2, consume c1,d1,d2 *)
           (merge_split_pair (s1', s2') + subst_cost c1 d1 + subst_cost c2 d2)  (* double subst: c1→d1, c2→d2 *)
  end.
Proof.
  (* Termination proofs - each recursive call decreases measure *)
  all: intros; unfold ms_measure; simpl; lia.
Defined.

(** Wrapper function with standard signature *)
Definition merge_split_distance (s1 s2 : list Char) : nat :=
  merge_split_pair (s1, s2).

(** * Unfolding Lemmas *)

Lemma ms_empty_left :
  forall s, merge_split_distance [] s = length s.
Proof.
  intro s.
  unfold merge_split_distance.
  rewrite merge_split_pair_equation.
  reflexivity.
Qed.

Lemma ms_empty_right :
  forall s, merge_split_distance s [] = length s.
Proof.
  intro s.
  unfold merge_split_distance.
  rewrite merge_split_pair_equation.
  destruct s as [| c1 s'].
  - reflexivity.
  - destruct s' as [| c2 s''].
    + reflexivity.
    + reflexivity.
Qed.

Lemma ms_single :
  forall c1 d1, merge_split_distance [c1] [d1] =
    if char_eq c1 d1 then 0 else 1.
Proof.
  intros c1 d1.
  unfold merge_split_distance.
  rewrite merge_split_pair_equation.
  reflexivity.
Qed.

(** * Key Property: Merge-Split Distance ≤ Standard Levenshtein *)

Lemma ms_le_standard : forall s1 s2,
  merge_split_distance s1 s2 <= lev_distance s1 s2.
Proof.
  (* Strong induction on |s1| + |s2|.
     Key insight: Each branch in merge_split_pair includes all standard
     Levenshtein options (del, ins, subst) plus additional merge/split options.
     Taking min with additional options can only decrease the result. *)
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite ms_empty_left.
    rewrite lev_distance_empty_left.
    lia.
  - destruct s2 as [| d1 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite ms_empty_right.
      rewrite lev_distance_empty_right.
      lia.
    + destruct s1' as [| c2 s1''].
      * (* s1 = [c1] *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = [c1], s2 = [d1] *)
           rewrite ms_single.
           rewrite lev_distance_cons.
           rewrite lev_distance_empty_left.
           rewrite lev_distance_empty_right.
           rewrite lev_distance_empty_left.
           simpl.
           unfold min3, subst_cost, char_eq.
           destruct (ascii_dec c1 d1); simpl; lia.
        -- (* s1 = [c1], s2 = d1::d2::s2'' *)
           (* Both use min3 with del, ins, subst *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           rewrite lev_distance_cons.
           unfold min3.
           assert (Hdel : merge_split_pair ([], d1 :: d2 :: s2'') <=
                          lev_distance [] (d1 :: d2 :: s2'')).
           { apply IH with (m := 0 + length (d1 :: d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : merge_split_pair ([c1], d2 :: s2'') <=
                          lev_distance [c1] (d2 :: s2'')).
           { apply IH with (m := 1 + length (d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : merge_split_pair ([], d2 :: s2'') <=
                          lev_distance [] (d2 :: s2'')).
           { apply IH with (m := 0 + length (d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           lia.
      * (* s1 = c1::c2::s1'' *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = c1::c2::s1'', s2 = [d1] *)
           (* merge_split uses min(min3, merge), lev uses min3 *)
           (* min(X, Y) <= X when X = min3(...) *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           rewrite lev_distance_cons.
           unfold min3.
           assert (Hdel : merge_split_pair (c2 :: s1'', [d1]) <=
                          lev_distance (c2 :: s1'') [d1]).
           { apply IH with (m := length (c2 :: s1'') + 1).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : merge_split_pair (c1 :: c2 :: s1'', []) <=
                          lev_distance (c1 :: c2 :: s1'') []).
           { apply IH with (m := length (c1 :: c2 :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : merge_split_pair (c2 :: s1'', []) <=
                          lev_distance (c2 :: s1'') []).
           { apply IH with (m := length (c2 :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           (* min (min3 ...) merge <= min3 lev ... because min3 ms <= min3 lev *)
           etransitivity.
           ++ apply Nat.le_min_l.  (* drop the merge branch *)
           ++ lia.
        -- (* s1 = c1::c2::s1'', s2 = d1::d2::s2'' - main case *)
           (* min6 includes del, ins, subst as first 3 elements *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           rewrite lev_distance_cons.
           unfold min6, min5, min3.
           assert (Hdel : merge_split_pair (c2 :: s1'', d1 :: d2 :: s2'') <=
                          lev_distance (c2 :: s1'') (d1 :: d2 :: s2'')).
           { apply IH with (m := length (c2 :: s1'') + length (d1 :: d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : merge_split_pair (c1 :: c2 :: s1'', d2 :: s2'') <=
                          lev_distance (c1 :: c2 :: s1'') (d2 :: s2'')).
           { apply IH with (m := length (c1 :: c2 :: s1'') + length (d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : merge_split_pair (c2 :: s1'', d2 :: s2'') <=
                          lev_distance (c2 :: s1'') (d2 :: s2'')).
           { apply IH with (m := length (c2 :: s1'') + length (d2 :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           (* min6 includes del, ins, subst. min6 <= min(del, ins, subst) *)
           remember (merge_split_pair (c2 :: s1'', d1 :: d2 :: s2'') + 1) as msD.
           remember (merge_split_pair (c1 :: c2 :: s1'', d2 :: s2'') + 1) as msI.
           remember (merge_split_pair (c2 :: s1'', d2 :: s2'') + subst_cost c1 d1) as msS.
           remember (lev_distance (c2 :: s1'') (d1 :: d2 :: s2'') + 1) as lD.
           remember (lev_distance (c1 :: c2 :: s1'') (d2 :: s2'') + 1) as lI.
           remember (lev_distance (c2 :: s1'') (d2 :: s2'') + subst_cost c1 d1) as lS.
           assert (HmsD_lD : msD <= lD) by lia.
           assert (HmsI_lI : msI <= lI) by lia.
           assert (HmsS_lS : msS <= lS) by lia.
           (* min6 structure: min (min (min (min msD msI) (min msS merge)) split) double *)
           (* This is <= msD, msI, msS, etc. *)
           assert (Hmin6_msD : min (min (min (min msD msI) (min msS
                   (merge_split_pair (s1'', d1 :: d2 :: s2'') + merge_cost c1 c2 d1)))
                   (merge_split_pair (c1 :: c2 :: s1'', s2'') + split_cost d1 c1 c2))
                   (merge_split_pair (c2 :: s1'', s2'') + subst_cost c1 d1 + subst_cost c2 d2)
                 <= msD).
           { etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_l|].
             apply Nat.le_min_l. }
           assert (Hmin6_msI : min (min (min (min msD msI) (min msS
                   (merge_split_pair (s1'', d1 :: d2 :: s2'') + merge_cost c1 c2 d1)))
                   (merge_split_pair (c1 :: c2 :: s1'', s2'') + split_cost d1 c1 c2))
                   (merge_split_pair (c2 :: s1'', s2'') + subst_cost c1 d1 + subst_cost c2 d2)
                 <= msI).
           { etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_l|].
             apply Nat.le_min_r. }
           assert (Hmin6_msS : min (min (min (min msD msI) (min msS
                   (merge_split_pair (s1'', d1 :: d2 :: s2'') + merge_cost c1 c2 d1)))
                   (merge_split_pair (c1 :: c2 :: s1'', s2'') + split_cost d1 c1 c2))
                   (merge_split_pair (c2 :: s1'', s2'') + subst_cost c1 d1 + subst_cost c2 d2)
                 <= msS).
           { etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_l|].
             etransitivity; [apply Nat.le_min_r|].
             apply Nat.le_min_l. }
           (* Now case split on which of lD, lI, lS is minimum *)
           destruct (Nat.le_ge_cases lD lI) as [HlD_lI | HlD_lI];
           destruct (Nat.le_ge_cases lD lS) as [HlD_lS | HlD_lS];
           destruct (Nat.le_ge_cases lI lS) as [HlI_lS | HlI_lS]; lia.
Qed.

(** * Identity of Indiscernibles *)

Lemma char_eq_refl : forall c, char_eq c c = true.
Proof.
  intro c. unfold char_eq.
  destruct (ascii_dec c c) as [_ | Hneq].
  - reflexivity.
  - exfalso. apply Hneq. reflexivity.
Qed.

(** char_eq is symmetric *)
Lemma char_eq_sym : forall c d, char_eq c d = char_eq d c.
Proof.
  intros c d. unfold char_eq.
  destruct (ascii_dec c d) as [Heq | Hneq];
  destruct (ascii_dec d c) as [Heq' | Hneq'].
  - reflexivity.
  - exfalso. apply Hneq'. symmetry. exact Heq.
  - exfalso. apply Hneq. symmetry. exact Heq'.
  - reflexivity.
Qed.

(** char_eq = true implies equality *)
Lemma char_eq_true : forall c1 c2, char_eq c1 c2 = true -> c1 = c2.
Proof.
  intros c1 c2 H. unfold char_eq in H.
  destruct (ascii_dec c1 c2) as [Heq|Hneq].
  - exact Heq.
  - discriminate.
Qed.

(** subst_cost is symmetric *)
Lemma subst_cost_sym : forall c d, subst_cost c d = subst_cost d c.
Proof.
  intros c d. unfold subst_cost.
  rewrite char_eq_sym. reflexivity.
Qed.

(** merge_cost and split_cost are related symmetrically:
    merge_cost c1 c2 d = split_cost d c1 c2 *)
Lemma merge_split_cost_sym : forall c1 c2 d,
  merge_cost c1 c2 d = split_cost d c1 c2.
Proof.
  intros c1 c2 d. unfold merge_cost, split_cost, can_split.
  reflexivity.
Qed.

Lemma ms_same : forall s,
  merge_split_distance s s = 0.
Proof.
  intro s.
  remember (length s) as n eqn:Hlen.
  revert s Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s Hlen.
  destruct s as [| c1 s'].
  - (* s = [] *)
    apply ms_empty_left.
  - destruct s' as [| c2 s''].
    + (* s = [c1] *)
      rewrite ms_single.
      rewrite char_eq_refl. reflexivity.
    + (* s = c1 :: c2 :: s'' *)
      (* Use the unfolding for cons2 case *)
      unfold merge_split_distance.
      rewrite merge_split_pair_equation.
      (* The diagonal (subst) branch: d(c2::s'', c2::s'') + subst c1 c1 = 0 + 0 = 0 *)
      unfold min6, min5.
      assert (Hsubst : subst_cost c1 c1 = 0).
      { unfold subst_cost. rewrite char_eq_refl. reflexivity. }
      (* Need IH for the diagonal recursive call *)
      assert (Hsub : merge_split_pair (c2 :: s'', c2 :: s'') = 0).
      { assert (Hlen' : length (c2 :: s'') < n).
        { simpl in Hlen. simpl. lia. }
        specialize (IH (length (c2 :: s'')) Hlen' (c2 :: s'') eq_refl).
        unfold merge_split_distance in IH. exact IH. }
      rewrite Hsub, Hsubst.
      (* Now the third position in min6 (diagonal branch) is 0 + 0 = 0 *)
      (* Goal: min(min(min(min(min a b) (min 0 d)) e) f) = 0 *)
      apply Nat.le_antisymm; [| lia].
      (* Need to show min6 a b 0 d e f <= 0 *)
      (* min(min(min(min(min a b) (min 0 d)) e) f) <= min(min(min 0 d) e) f <= min 0 d <= 0 *)
      etransitivity; [apply Nat.le_min_l|].
      etransitivity; [apply Nat.le_min_l|].
      etransitivity; [apply Nat.le_min_r|].
      apply Nat.le_min_l.
Qed.

(** * Merge Example *)

(** Under generic semantics:
    - merge_split_distance [c1; c2] [d] = 1 (via merge)
    - lev_distance [c1; c2] [d] = 2 (needs delete + subst or subst + delete) *)

Lemma ms_merge_when_applicable : forall c1 c2 d,
  can_merge c1 c2 d = true ->
  merge_split_distance [c1; c2] [d] = 1.
Proof.
  intros c1 c2 d Hmerge.
  (* Use the multi_single case: (c1 :: c2 :: [], [d]) *)
  unfold merge_split_distance.
  rewrite merge_split_pair_equation.
  (* The merge branch: d([], []) + merge_cost c1 c2 d = 0 + 1 *)
  (* Standard branches: all require at least 2 ops *)
  (* Delete c1: d([c2], [d]) + 1 >= 1 + 1 = 2 *)
  (* Insert d: d([c1;c2], []) + 1 = 2 + 1 = 3 *)
  (* Subst c1 for d: d([c2], []) + subst_cost >= 1 + 0 = 1 (but this branch is min'd later) *)
  unfold merge_cost.
  rewrite Hmerge.
  (* Now the merge branch is: d([], []) + 1 *)
  simpl.
  repeat rewrite merge_split_pair_equation.
  simpl.
  unfold min3.
  unfold subst_cost, char_eq.
  (* Need to case split on character equality *)
  destruct (ascii_dec c1 d) as [Heq1|Hneq1];
  destruct (ascii_dec c2 d) as [Heq2|Hneq2]; simpl; try lia.
Qed.

(** * Split Example (symmetric to merge) *)

Lemma ms_split_when_applicable : forall c d1 d2,
  can_split c d1 d2 = true ->
  merge_split_distance [c] [d1; d2] = 1.
Proof.
  intros c d1 d2 Hsplit.
  (* Use the single_multi case with newly added split branch *)
  unfold merge_split_distance.
  rewrite merge_split_pair_equation.
  (* The split branch: ms([], []) + split_cost c d1 d2 = 0 + 1 = 1 *)
  unfold split_cost.
  rewrite Hsplit.
  simpl.
  repeat rewrite merge_split_pair_equation.
  simpl.
  unfold min3.
  (* The split branch gives 1, other branches give >= 1, so min is 1 *)
  unfold subst_cost, char_eq.
  destruct (ascii_dec c d1); lia.
Qed.

(** * Metric Properties *)

(* Helper lemma: min6 is permutation invariant for swapping (1,2) and (4,5) *)
Lemma min6_swap_12_45 : forall a b c d e f,
  min6 a b c d e f = min6 b a c e d f.
Proof.
  intros a b c d e f.
  unfold min6, min5.
  lia.
Qed.

Lemma ms_sym : forall s1 s2,
  merge_split_distance s1 s2 = merge_split_distance s2 s1.
Proof.
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.

  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite ms_empty_left, ms_empty_right. reflexivity.
  - destruct s2 as [| d1 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite ms_empty_right, ms_empty_left. reflexivity.
    + destruct s1' as [| c2 s1''].
      * (* s1 = [c1] *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = [c1], s2 = [d1] *)
           rewrite ms_single, ms_single.
           rewrite char_eq_sym. reflexivity.
        -- (* s1 = [c1], s2 = d1::d2::s2'' *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           set (lhs := min _ _).
           rewrite merge_split_pair_equation.
           unfold lhs. clear lhs.
           assert (IH1: merge_split_pair (nil, d1 :: d2 :: s2'') =
                       merge_split_pair (d1 :: d2 :: s2'', nil)).
           { fold merge_split_distance. apply IH with (m := length (d1::d2::s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH2: merge_split_pair (c1 :: nil, d2 :: s2'') =
                       merge_split_pair (d2 :: s2'', c1 :: nil)).
           { fold merge_split_distance. apply IH with (m := 1 + length (d2::s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH3: merge_split_pair (nil, d2 :: s2'') =
                       merge_split_pair (d2 :: s2'', nil)).
           { fold merge_split_distance. apply IH with (m := length (d2::s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH4: merge_split_pair (nil, s2'') = merge_split_pair (s2'', nil)).
           { fold merge_split_distance. apply IH with (m := length s2'').
             simpl in *; lia. simpl; lia. }
           rewrite IH1, IH2, IH3, IH4.
           rewrite subst_cost_sym.
           unfold split_cost, merge_cost, can_split.
           unfold min3.
           set (A := merge_split_pair (d1 :: d2 :: s2'', nil)).
           set (B := merge_split_pair (d2 :: s2'', c1 :: nil)).
           set (C := merge_split_pair (d2 :: s2'', nil)).
           set (D := merge_split_pair (s2'', nil)).
           assert (Hcomm: min (A + 1) (min (B + 1) (C + subst_cost d1 c1)) =
                         min (B + 1) (min (A + 1) (C + subst_cost d1 c1))).
           { lia. }
           rewrite Hcomm. reflexivity.
      * destruct s2' as [| d2 s2''].
        -- (* s1 = c1::c2::s1'', s2 = [d1] *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           set (lhs := min _ _).
           rewrite merge_split_pair_equation.
           unfold lhs. clear lhs.
           assert (IH1: merge_split_pair (c2 :: s1'', d1 :: nil) =
                       merge_split_pair (d1 :: nil, c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := length (c2::s1'') + 1).
             simpl in *; lia. simpl; lia. }
           assert (IH2: merge_split_pair (c1 :: c2 :: s1'', nil) =
                       merge_split_pair (nil, c1 :: c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := length (c1::c2::s1'')).
             simpl in *; lia. simpl; lia. }
           assert (IH3: merge_split_pair (c2 :: s1'', nil) =
                       merge_split_pair (nil, c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := length (c2::s1'')).
             simpl in *; lia. simpl; lia. }
           assert (IH4: merge_split_pair (s1'', nil) = merge_split_pair (nil, s1'')).
           { fold merge_split_distance. apply IH with (m := length s1'').
             simpl in *; lia. simpl; lia. }
           rewrite IH1, IH2, IH3, IH4.
           rewrite subst_cost_sym.
           unfold merge_cost, split_cost, can_split.
           unfold min3.
           set (A := merge_split_pair (d1 :: nil, c2 :: s1'')).
           set (B := merge_split_pair (nil, c1 :: c2 :: s1'')).
           set (C := merge_split_pair (nil, c2 :: s1'')).
           set (D := merge_split_pair (nil, s1'')).
           assert (Hcomm: min (A + 1) (min (B + 1) (C + subst_cost d1 c1)) =
                         min (B + 1) (min (A + 1) (C + subst_cost d1 c1))).
           { lia. }
           rewrite Hcomm. reflexivity.
        -- (* s1 = c1::c2::s1'', s2 = d1::d2::s2'' - main case *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           set (lhs := min6 _ _ _ _ _ _).
           rewrite merge_split_pair_equation.
           unfold lhs. clear lhs.
           (* Establish all IH results *)
           assert (IH_del: merge_split_pair (c2 :: s1'', d1 :: d2 :: s2'') =
                          merge_split_pair (d1 :: d2 :: s2'', c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := S (length s1'') + S (S (length s2''))).
             simpl in *; lia. simpl; lia. }
           assert (IH_ins: merge_split_pair (c1 :: c2 :: s1'', d2 :: s2'') =
                          merge_split_pair (d2 :: s2'', c1 :: c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := S (S (length s1'')) + S (length s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH_sub: merge_split_pair (c2 :: s1'', d2 :: s2'') =
                          merge_split_pair (d2 :: s2'', c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := S (length s1'') + S (length s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH_merge: merge_split_pair (s1'', d2 :: s2'') =
                            merge_split_pair (d2 :: s2'', s1'')).
           { fold merge_split_distance. apply IH with (m := length s1'' + S (length s2'')).
             simpl in *; lia. simpl; lia. }
           assert (IH_split: merge_split_pair (c2 :: s1'', s2'') =
                            merge_split_pair (s2'', c2 :: s1'')).
           { fold merge_split_distance. apply IH with (m := S (length s1'') + length s2'').
             simpl in *; lia. simpl; lia. }
           assert (IH_double: merge_split_pair (s1'', s2'') =
                             merge_split_pair (s2'', s1'')).
           { fold merge_split_distance. apply IH with (m := length s1'' + length s2'').
             simpl in *; lia. simpl; lia. }
           rewrite IH_del, IH_ins, IH_sub, IH_merge, IH_split, IH_double.
           rewrite subst_cost_sym.
           rewrite (subst_cost_sym c2 d2).
           assert (Hmerge_split: split_cost d1 c1 c2 = merge_cost c1 c2 d1).
           { unfold split_cost, merge_cost, can_split. reflexivity. }
           assert (Hsplit_merge: split_cost c1 d1 d2 = merge_cost d1 d2 c1).
           { unfold split_cost, merge_cost, can_split. reflexivity. }
           rewrite Hmerge_split, Hsplit_merge.
           (* Goal: min6 A B C D E F = min6 B A C E D F *)
           apply min6_swap_12_45.
Qed.

(** Length-difference lower bound: ms(s1,s2) >= ||s1| - |s2||.
    This holds because each operation changes the length difference by at most 1
    and costs at least 0 (for subst of matching chars). To change length by n,
    we need at least n operations that aren't free substs.

    PROOF STATUS: proved by well-founded induction over combined input length.
    The key insight is that:
    - Delete/Insert/Merge/Split each change length by 1, cost 1
    - Subst preserves length, costs 0 or 1
    - Double-subst changes length by 0, costs 0-2
    So minimum cost to change length difference by k is at least k.
*)
Lemma ms_length_diff_lower : forall s1 s2,
  merge_split_distance s1 s2 >= abs_diff (length s1) (length s2).
Proof.
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.

  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite ms_empty_left. unfold abs_diff. simpl.
    destruct (0 <=? length s2); lia.
  - destruct s2 as [| d1 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite ms_empty_right. unfold abs_diff. simpl.
      destruct (S (length s1') <=? 0) eqn:E; [apply Nat.leb_le in E; lia | lia].

    + (* s1 = c1::s1', s2 = d1::s2' - both non-empty *)
      (* Key: abs_diff (S m) (S n) = abs_diff m n *)
      assert (Habs_eq: abs_diff (length (c1 :: s1')) (length (d1 :: s2')) =
                       abs_diff (length s1') (length s2')).
      { unfold abs_diff. simpl.
        destruct (length s1' <=? length s2') eqn:Hcmp;
        destruct (S (length s1') <=? S (length s2')) eqn:Hcmp2.
        - reflexivity.
        - apply Nat.leb_le in Hcmp. apply Nat.leb_gt in Hcmp2. lia.
        - apply Nat.leb_gt in Hcmp. apply Nat.leb_le in Hcmp2. lia.
        - reflexivity. }
      rewrite Habs_eq.

      (* Now show ms(c1::s1', d1::s2') >= abs_diff (length s1') (length s2') *)
      (* Case split on structure to match merge_split_pair cases *)
      destruct s1' as [| c2 s1''].
      * (* s1 = [c1], s2 = d1::s2' *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = [c1], s2 = [d1] *)
           rewrite ms_single.
           unfold abs_diff. simpl.
           destruct (0 <=? 0); lia.
        -- (* s1 = [c1], s2 = d1::d2::s2'' *)
           (* After Habs_eq rewrite, we need to show:
              ms([c1], d1::d2::s2'') >= abs_diff (length []) (length (d2::s2''))
                                      = abs_diff 0 (S (length s2''))
                                      = S (length s2'') *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           (* min (min3 (delete+1) (insert+1) (subst)) (split) *)
           (* All four branches need to be >= S(length s2'') *)
           set (target := abs_diff 0 (S (length s2''))).
           assert (Hdel: merge_split_pair ([], d1 :: d2 :: s2'') + 1 >= target).
           { rewrite merge_split_pair_equation. unfold target, abs_diff. simpl. lia. }
           assert (Hsubst_bound: subst_cost c1 d1 <= 1).
           { unfold subst_cost. destruct (char_eq c1 d1); lia. }
           assert (Hsubst_br: merge_split_pair ([], d2 :: s2'') + subst_cost c1 d1 >= target).
           { rewrite merge_split_pair_equation. unfold target, abs_diff. simpl. lia. }
           assert (Hins_br: merge_split_pair ([c1], d2 :: s2'') + 1 >= target).
           { assert (IHins: merge_split_pair ([c1], d2 :: s2'') >=
                           abs_diff (length [c1]) (length (d2 :: s2''))).
             { fold merge_split_distance.
               apply (IH (1 + S (length s2''))). simpl in *; lia. simpl. lia. }
             unfold target, abs_diff in *. simpl in *. lia. }
           (* Split branch: ms([], s2'') + split_cost c1 d1 d2 *)
           assert (Hsplit_br: merge_split_pair ([], s2'') + split_cost c1 d1 d2 >= target).
           { rewrite merge_split_pair_equation. unfold target, abs_diff, split_cost. simpl.
             destruct (can_split c1 d1 d2); lia. }
           (* min3 a b c = min(a, min(b, c)) *)
           unfold min3.
           apply Nat.min_case.
           ++ (* min3 branch *)
              apply Nat.min_case; [exact Hdel |].
              apply Nat.min_case; [exact Hins_br | exact Hsubst_br].
           ++ (* split branch *)
              exact Hsplit_br.

      * (* s1 = c1::c2::s1'', s2 = d1::s2' *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = c1::c2::s1'', s2 = [d1] *)
           (* After Habs_eq rewrite, we need:
              ms(c1::c2::s1'', [d1]) >= abs_diff (length (c2::s1'')) (length [])
                                      = abs_diff (S (length s1'')) 0
                                      = S (length s1'') *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           set (target := abs_diff (S (length s1'')) 0).
           assert (Hsubst_bound: subst_cost c1 d1 <= 1).
           { unfold subst_cost. destruct (char_eq c1 d1); lia. }
           (* Delete: ms(c2::s1'', [d1]) + 1 *)
           assert (IHdel: merge_split_pair (c2 :: s1'', [d1]) + 1 >= target).
           { assert (IH1: merge_split_pair (c2 :: s1'', [d1]) >=
                         abs_diff (length (c2 :: s1'')) (length [d1])).
             { fold merge_split_distance.
               apply (IH (S (length s1'') + 1)). simpl in *; lia. simpl. lia. }
             (* abs_diff (S m) 1 = if S m <= 1 then 1 - S m else S m - 1 = if m = 0 then 0 else m *)
             (* target = abs_diff (S m) 0 = S m *)
             (* So IH1 + 1 >= S m needs: if m = 0 then 0 + 1 >= 1 ✓ else m + 1 >= S m ✓ *)
             unfold target, abs_diff in *. simpl in *.
             destruct (length s1'' <=? 0) eqn:E1;
             destruct (S (length s1'') <=? 1) eqn:E2;
             try (apply Nat.leb_le in E1); try (apply Nat.leb_gt in E1);
             try (apply Nat.leb_le in E2); try (apply Nat.leb_gt in E2);
             lia. }
           (* Insert: ms(c1::c2::s1'', []) + 1 = S(S(length s1'')) + 1 *)
           assert (Hins: merge_split_pair (c1 :: c2 :: s1'', []) + 1 >= target).
           { rewrite merge_split_pair_equation. unfold target, abs_diff. simpl. lia. }
           (* Subst: ms(c2::s1'', []) + subst = S(length s1'') + subst *)
           assert (Hsubst: merge_split_pair (c2 :: s1'', []) + subst_cost c1 d1 >= target).
           { (* ms(c2::s1'', []) = length(c2::s1'') = S(length s1'') *)
             (* target = abs_diff (S(length s1'')) 0 = S(length s1'') *)
             (* Need: S(length s1'') + subst_cost >= S(length s1'') which is clear *)
             rewrite merge_split_pair_equation.
             destruct s1'' as [| c3 s1'''].
             - (* s1'' = [] so c2::s1'' = [c2] *)
               simpl. unfold target, abs_diff. simpl.
               assert (Hsub_ge0: subst_cost c1 d1 >= 0) by lia. lia.
             - (* s1'' = c3::s1''' *)
               simpl. unfold target, abs_diff. simpl.
               assert (Hsub_ge0: subst_cost c1 d1 >= 0) by lia. lia. }
           (* Merge: ms(s1'', []) + merge_cost *)
           assert (Hmerge: merge_split_pair (s1'', []) + merge_cost c1 c2 d1 >= target).
           { (* ms(s1'', []) = length s1'', target = S(length s1'') *)
             (* merge_cost = 1 if can_merge else 100 *)
             (* Need: length s1'' + merge_cost >= S(length s1'') *)
             unfold target, abs_diff.
             (* merge_split_pair (s, []) = length s by definition *)
             pose proof (ms_empty_right s1'') as Hms_eq.
             unfold merge_split_distance in Hms_eq.
             rewrite Hms_eq.
             unfold merge_cost.
             destruct (can_merge c1 c2 d1) eqn:Hcan; simpl; lia. }
           (* Now show min(min3(...), merge) >= target *)
           (* min(min3(d,i,s), m) >= target when all >= target *)
           apply Nat.min_case.
           { unfold min3. apply Nat.min_case; [exact IHdel |].
             apply Nat.min_case; [exact Hins | exact Hsubst]. }
           { exact Hmerge. }

        -- (* s1 = c1::c2::s1'', s2 = d1::d2::s2'' - main case *)
           (* After rewrite, goal is: ms(c1::c2::s1'', d1::d2::s2'') >= abs_diff (S len1) (S len2)
              where len1 = length s1'', len2 = length s2''.
              Since abs_diff (S len1) (S len2) = abs_diff len1 len2, we work with the latter. *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.

           set (len1 := length s1'').
           set (len2 := length s2'').

           (* First establish the key equality for target conversion *)
           assert (Htarget_eq: abs_diff (S len1) (S len2) = abs_diff len1 len2).
           { unfold abs_diff.
             destruct (len1 <=? len2) eqn:E1;
             destruct (S len1 <=? S len2) eqn:E2.
             - reflexivity.
             - apply Nat.leb_le in E1. apply Nat.leb_gt in E2. lia.
             - apply Nat.leb_gt in E1. apply Nat.leb_le in E2. lia.
             - reflexivity. }

           (* The goal after unfold is: min6 ... >= abs_diff (S len1) (S len2) *)
           (* We rewrite to use abs_diff len1 len2 as target *)
           set (target := abs_diff len1 len2).
           assert (Hgoal_eq: abs_diff (length (c2 :: s1'')) (length (d2 :: s2'')) = target).
           { simpl. unfold len1, len2, target. exact Htarget_eq. }
           rewrite Hgoal_eq.

           (* Now prove each branch >= target *)

           (* Branch 1: Delete - ms(c2::s1'', d1::d2::s2'') + 1 *)
           assert (Hdel: merge_split_pair (c2 :: s1'', d1 :: d2 :: s2'') + 1 >= target).
           { assert (IHdel: merge_split_pair (c2 :: s1'', d1 :: d2 :: s2'') >=
                           abs_diff (length (c2 :: s1'')) (length (d1 :: d2 :: s2''))).
             { fold merge_split_distance.
               apply (IH (S len1 + S (S len2))). simpl in Hlen. unfold len1, len2. lia.
               simpl. unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length (c2 :: s1'')) (length (d1 :: d2 :: s2'')) =
                             abs_diff len1 (S len2)).
             { simpl. unfold abs_diff, len1, len2.
               destruct (length s1'' <=? S (length s2'')) eqn:E1;
               destruct (S (length s1'') <=? S (S (length s2''))) eqn:E2.
               - reflexivity.
               - apply Nat.leb_le in E1. apply Nat.leb_gt in E2. lia.
               - apply Nat.leb_gt in E1. apply Nat.leb_le in E2. lia.
               - reflexivity. }
             rewrite Hlen_eq in IHdel.
             pose proof (abs_diff_succ_bound len1 len2) as Hbound.
             unfold target. lia. }

           (* Branch 2: Insert - ms(c1::c2::s1'', d2::s2'') + 1 *)
           assert (Hins: merge_split_pair (c1 :: c2 :: s1'', d2 :: s2'') + 1 >= target).
           { assert (IHins: merge_split_pair (c1 :: c2 :: s1'', d2 :: s2'') >=
                           abs_diff (length (c1 :: c2 :: s1'')) (length (d2 :: s2''))).
             { fold merge_split_distance.
               apply (IH (S (S len1) + S len2)). simpl in Hlen. unfold len1, len2. lia.
               simpl. unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length (c1 :: c2 :: s1'')) (length (d2 :: s2'')) =
                             abs_diff (S len1) len2).
             { simpl. unfold abs_diff, len1, len2.
               destruct (S (length s1'') <=? length s2'') eqn:E1;
               destruct (S (S (length s1'')) <=? S (length s2'')) eqn:E2.
               - reflexivity.
               - apply Nat.leb_le in E1. apply Nat.leb_gt in E2. lia.
               - apply Nat.leb_gt in E1. apply Nat.leb_le in E2. lia.
               - reflexivity. }
             rewrite Hlen_eq in IHins.
             pose proof (abs_diff_succ_bound_fst len1 len2) as Hbound.
             unfold target. lia. }

           (* Branch 3: Subst - ms(c2::s1'', d2::s2'') + subst_cost c1 d1 *)
           assert (Hsubst: merge_split_pair (c2 :: s1'', d2 :: s2'') + subst_cost c1 d1 >= target).
           { assert (IHsubst: merge_split_pair (c2 :: s1'', d2 :: s2'') >=
                             abs_diff (length (c2 :: s1'')) (length (d2 :: s2''))).
             { fold merge_split_distance.
               apply (IH (S len1 + S len2)). simpl in Hlen. unfold len1, len2. lia.
               simpl. unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length (c2 :: s1'')) (length (d2 :: s2'')) = target).
             { simpl. unfold len1, len2, target. exact Htarget_eq. }
             rewrite Hlen_eq in IHsubst. lia. }

           (* Branch 4: Merge - ms(s1'', d2::s2'') + merge_cost c1 c2 d1 *)
           assert (Hmerge: merge_split_pair (s1'', d2 :: s2'') + merge_cost c1 c2 d1 >= target).
           { (* First establish IH bound - needed for both cases *)
             assert (IHmerge: merge_split_pair (s1'', d2 :: s2'') >=
                             abs_diff (length s1'') (length (d2 :: s2''))).
             { fold merge_split_distance.
               apply (IH (len1 + S len2)). simpl in Hlen. unfold len1, len2. lia.
               simpl. unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length s1'') (length (d2 :: s2'')) =
                             abs_diff len1 (S len2)).
             { simpl. unfold len1, len2. reflexivity. }
             rewrite Hlen_eq in IHmerge.
             (* merge_cost >= 1 (either 1 or 100) *)
             assert (Hmc_ge1: merge_cost c1 c2 d1 >= 1).
             { unfold merge_cost. destruct (can_merge c1 c2 d1); lia. }
             (* Need: abs_diff len1 (S len2) + 1 >= target = abs_diff len1 len2 *)
             unfold target, abs_diff in *.
             destruct (len1 <=? S len2) eqn:E1;
             destruct (len1 <=? len2) eqn:E2.
             - apply Nat.leb_le in E1. apply Nat.leb_le in E2. lia.
             - apply Nat.leb_le in E1. apply Nat.leb_gt in E2. lia.
             - apply Nat.leb_gt in E1. apply Nat.leb_le in E2. lia.
             - apply Nat.leb_gt in E1. apply Nat.leb_gt in E2. lia. }

           (* Branch 5: Split - ms(c2::s1'', s2'') + split_cost c1 d1 d2 *)
           assert (Hsplit: merge_split_pair (c2 :: s1'', s2'') + split_cost c1 d1 d2 >= target).
           { (* First establish IH bound - needed for both cases *)
             assert (IHsplit: merge_split_pair (c2 :: s1'', s2'') >=
                             abs_diff (length (c2 :: s1'')) (length s2'')).
             { fold merge_split_distance.
               apply (IH (S len1 + len2)). simpl in Hlen. unfold len1, len2. lia.
               simpl. unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length (c2 :: s1'')) (length s2'') =
                             abs_diff (S len1) len2).
             { simpl. unfold len1, len2. reflexivity. }
             rewrite Hlen_eq in IHsplit.
             (* split_cost >= 1 (either 1 or 100) *)
             assert (Hsc_ge1: split_cost c1 d1 d2 >= 1).
             { unfold split_cost. destruct (can_split c1 d1 d2); lia. }
             (* Need: abs_diff (S len1) len2 + 1 >= target = abs_diff len1 len2 *)
             unfold target, abs_diff in *.
             destruct (S len1 <=? len2) eqn:E1;
             destruct (len1 <=? len2) eqn:E2.
             - apply Nat.leb_le in E1. apply Nat.leb_le in E2. lia.
             - apply Nat.leb_le in E1. apply Nat.leb_gt in E2. lia.
             - apply Nat.leb_gt in E1. apply Nat.leb_le in E2. lia.
             - apply Nat.leb_gt in E1. apply Nat.leb_gt in E2. lia. }

           (* Branch 6: Double-subst - ms(s1'', s2'') + subst_cost c1 d1 + subst_cost c2 d2 *)
           assert (Hdouble: merge_split_pair (s1'', s2'') + subst_cost c1 d1 + subst_cost c2 d2 >= target).
           { assert (IHdouble: merge_split_pair (s1'', s2'') >=
                              abs_diff (length s1'') (length s2'')).
             { fold merge_split_distance.
               apply (IH (len1 + len2)). simpl in Hlen. unfold len1, len2. lia.
               unfold len1, len2. lia. }
             assert (Hlen_eq: abs_diff (length s1'') (length s2'') = target).
             { unfold len1, len2, target. reflexivity. }
             rewrite Hlen_eq in IHdouble.
             (* subst_cost >= 0, so ms >= target implies ms + subst_cost + subst_cost >= target *)
             lia. }

           (* Now use min6 property: min6 a b c d e f >= x when all branches >= x *)
           (* lia can solve this directly after unfolding min6/min5 *)
           unfold min6, min5, min3.
           lia.
Qed.

Lemma ms_nonneg : forall s1 s2,
  merge_split_distance s1 s2 >= 0.
Proof.
  intros. lia.
Qed.

(** * Edit Sequence Infrastructure for Triangle Inequality *)

(** Edit operations for merge-split distance *)
Inductive ms_op : Type :=
  | MSDelete : Char -> ms_op                     (* delete char from source *)
  | MSInsert : Char -> ms_op                     (* insert char into target *)
  | MSSubst : Char -> Char -> ms_op              (* substitute source char with target char *)
  | MSMerge : Char -> Char -> Char -> ms_op      (* merge two source chars into one target char *)
  | MSSplit : Char -> Char -> Char -> ms_op.     (* split one source char into two target chars *)

(** Cost of a single edit operation *)
Definition ms_op_cost (op : ms_op) : nat :=
  match op with
  | MSDelete _ => 1
  | MSInsert _ => 1
  | MSSubst c d => subst_cost c d
  | MSMerge c1 c2 d => merge_cost c1 c2 d
  | MSSplit c d1 d2 => split_cost c d1 d2
  end.

(** Total cost of an edit sequence *)
Fixpoint ms_seq_cost (ops : list ms_op) : nat :=
  match ops with
  | [] => 0
  | op :: rest => ms_op_cost op + ms_seq_cost rest
  end.

(** Apply a single operation to (source, target) pair.
    Returns Some (new_source, new_target) if valid, None otherwise. *)
Definition apply_ms_op (op : ms_op) (src tgt : list Char) : option (list Char * list Char) :=
  match op, src, tgt with
  | MSDelete c, c' :: src', tgt' =>
      if char_eq c c' then Some (src', tgt') else None
  | MSInsert d, src', d' :: tgt' =>
      if char_eq d d' then Some (src', tgt') else None
  | MSSubst c d, c' :: src', d' :: tgt' =>
      if andb (char_eq c c') (char_eq d d') then Some (src', tgt') else None
  | MSMerge c1 c2 d, c1' :: c2' :: src', d' :: tgt' =>
      if andb (andb (char_eq c1 c1') (char_eq c2 c2')) (char_eq d d')
      then Some (src', tgt') else None
  | MSSplit c d1 d2, c' :: src', d1' :: d2' :: tgt' =>
      if andb (andb (char_eq c c') (char_eq d1 d1')) (char_eq d2 d2')
      then Some (src', tgt') else None
  | _, _, _ => None
  end.

(** Apply a sequence of operations *)
Fixpoint apply_ms_seq (ops : list ms_op) (src tgt : list Char) : option (list Char * list Char) :=
  match ops with
  | [] => Some (src, tgt)
  | op :: rest =>
      match apply_ms_op op src tgt with
      | Some (src', tgt') => apply_ms_seq rest src' tgt'
      | None => None
      end
  end.

(** A sequence is valid if it transforms (s1, s2) to ([], []) *)
Definition ms_seq_valid (ops : list ms_op) (s1 s2 : list Char) : Prop :=
  apply_ms_seq ops s1 s2 = Some ([], []).

(** Key lemma: Empty sequence is valid only for empty strings *)
Lemma ms_seq_empty_valid : forall s1 s2,
  ms_seq_valid [] s1 s2 <-> s1 = [] /\ s2 = [].
Proof.
  intros s1 s2. unfold ms_seq_valid. simpl.
  split; intros H.
  - inversion H. split; reflexivity.
  - destruct H as [H1 H2]. subst. reflexivity.
Qed.

(** Sequence cost is additive under concatenation *)
Lemma ms_seq_cost_app : forall ops1 ops2,
  ms_seq_cost (ops1 ++ ops2) = ms_seq_cost ops1 + ms_seq_cost ops2.
Proof.
  intros ops1 ops2.
  induction ops1 as [| op ops1' IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. lia.
Qed.

(** Sequence application is compositional *)
Lemma apply_ms_seq_app : forall ops1 ops2 src tgt src' tgt',
  apply_ms_seq ops1 src tgt = Some (src', tgt') ->
  apply_ms_seq (ops1 ++ ops2) src tgt = apply_ms_seq ops2 src' tgt'.
Proof.
  intros ops1 ops2 src tgt src' tgt' H.
  induction ops1 as [| op ops1' IH] in src, tgt, src', tgt', H |- *.
  - simpl in *. inversion H. subst. reflexivity.
  - simpl in *.
    destruct (apply_ms_op op src tgt) as [[s1 t1]|] eqn:E; [|discriminate].
    apply IH. exact H.
Qed.

(** For any strings s1, s2, there exists a valid edit sequence.
    This follows from the structure of merge_split_pair which always terminates. *)
Lemma ms_seq_exists : forall s1 s2,
  exists ops, ms_seq_valid ops s1 s2 /\ ms_seq_cost ops = merge_split_distance s1 s2.
Proof.
  intros s1 s2.
  (* Proof by strong induction on |s1| + |s2|, following merge_split_pair structure *)
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.

  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    (* Need |s2| insert operations *)
    exists (map MSInsert s2). split.
    + unfold ms_seq_valid.
      clear IH Hlen n.
      induction s2 as [| d s2' IHs2'].
      * simpl. reflexivity.
      * simpl. rewrite char_eq_refl. exact IHs2'.
    + rewrite ms_empty_left.
      clear IH Hlen n.
      induction s2 as [| d s2' IHs2'].
      * simpl. reflexivity.
      * simpl. rewrite IHs2'. reflexivity.

  - destruct s2 as [| d1 s2'].
    + (* s2 = [] *)
      (* Need |s1| delete operations *)
      exists (map MSDelete (c1 :: s1')). split.
      * unfold ms_seq_valid.
        simpl. rewrite char_eq_refl.
        clear IH Hlen n.
        induction s1' as [| c s1'' IHs1''].
        -- simpl. reflexivity.
        -- simpl. rewrite char_eq_refl. exact IHs1''.
      * rewrite ms_empty_right. simpl.
        clear IH Hlen n.
        induction s1' as [| c s1'' IHs1''].
        -- simpl. reflexivity.
        -- simpl. rewrite IHs1''. reflexivity.

    + (* s1 = c1::s1', s2 = d1::s2' - case analysis on structure *)
      destruct s1' as [| c2 s1''].
      * (* s1 = [c1] - single source *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = [c1], s2 = [d1] - single/single *)
           unfold merge_split_distance. rewrite merge_split_pair_equation.
           destruct (char_eq c1 d1) eqn:Heq.
           ++ (* c1 = d1, cost = 0, but still need subst operation to consume chars *)
              apply char_eq_true in Heq. subst d1.
              exists [MSSubst c1 c1]. split.
              ** unfold ms_seq_valid. simpl. rewrite !char_eq_refl. reflexivity.
              ** simpl. unfold subst_cost. rewrite char_eq_refl. lia.
           ++ (* c1 <> d1, cost = 1 = subst_cost c1 d1 *)
              exists [MSSubst c1 d1]. split.
              ** unfold ms_seq_valid. simpl. rewrite !char_eq_refl. reflexivity.
              ** simpl. unfold subst_cost. rewrite Heq. lia.

        -- (* s1 = [c1], s2 = d1::d2::s2'' - single/multi, potential split *)
           (* IH for the three standard branches *)
           assert (IH_del: exists ops, ms_seq_valid ops [] (d1::d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance [] (d1::d2::s2'')).
           { apply (IH (0 + length (d1::d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_ins: exists ops, ms_seq_valid ops [c1] (d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance [c1] (d2::s2'')).
           { apply (IH (1 + length (d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_sub: exists ops, ms_seq_valid ops [] (d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance [] (d2::s2'')).
           { apply (IH (0 + length (d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_spl: exists ops, ms_seq_valid ops [] s2'' /\
                           ms_seq_cost ops = merge_split_distance [] s2'').
           { apply (IH (0 + length s2'')). simpl in Hlen. simpl. lia. reflexivity. }

           destruct IH_del as [ops_del [Hv_del Hc_del]].
           destruct IH_ins as [ops_ins [Hv_ins Hc_ins]].
           destruct IH_sub as [ops_sub [Hv_sub Hc_sub]].
           destruct IH_spl as [ops_spl [Hv_spl Hc_spl]].

           (* ms([c1], d1::d2::s2'') = min (min3 del ins sub) split *)
           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           fold (merge_split_distance [] (d1::d2::s2'')).
           fold (merge_split_distance [c1] (d2::s2'')).
           fold (merge_split_distance [] (d2::s2'')).
           fold (merge_split_distance [] s2'').

           (* Define abbreviations for branch costs *)
           remember (merge_split_distance [] (d1::d2::s2'') + 1) as del eqn:Hdel_def.
           remember (merge_split_distance [c1] (d2::s2'') + 1) as ins eqn:Hins_def.
           remember (merge_split_distance [] (d2::s2'') + subst_cost c1 d1) as sub eqn:Hsub_def.
           remember (merge_split_distance [] s2'' + split_cost c1 d1 d2) as spl eqn:Hspl_def.

           (* Decide which branch wins using Nat.min_dec *)
           destruct (Nat.min_dec (min3 del ins sub) spl) as [Hstd | Hspl_win].
           ++ (* Standard branch wins *)
              destruct (Nat.min_dec del (Nat.min ins sub)) as [Hdel_win | Hins_sub].
              ** (* Delete wins *)
                 exists (MSDelete c1 :: ops_del). split.
                 --- unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_del.
                 --- simpl. rewrite Hc_del.
                     rewrite Hstd. unfold min3. rewrite <- Hdel_win, Hdel_def. lia.
              ** destruct (Nat.min_dec ins sub) as [Hins_win | Hsub_win].
                 --- (* Insert wins *)
                     exists (MSInsert d1 :: ops_ins). split.
                     +++ unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_ins.
                     +++ simpl. rewrite Hc_ins.
                         rewrite Hstd. unfold min3. rewrite <- Hins_sub, <- Hins_win, Hins_def. lia.
                 --- (* Subst wins *)
                     exists (MSSubst c1 d1 :: ops_sub). split.
                     +++ unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_sub.
                     +++ simpl. rewrite Hc_sub.
                         rewrite Hstd. unfold min3. rewrite <- Hins_sub, <- Hsub_win, Hsub_def. lia.
           ++ (* Split wins *)
              exists (MSSplit c1 d1 d2 :: ops_spl). split.
              ** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_spl.
              ** simpl. rewrite Hc_spl.
                 rewrite Hspl_win.
                 rewrite Hspl_def.
                 unfold split_cost, can_split. lia.

      * (* s1 = c1::c2::s1'' - multi source *)
        destruct s2' as [| d2 s2''].
        -- (* s1 = c1::c2::s1'', s2 = [d1] - multi/single, potential merge *)
           (* IH for the standard branches and merge *)
           assert (IH_del: exists ops, ms_seq_valid ops (c2::s1'') [d1] /\
                           ms_seq_cost ops = merge_split_distance (c2::s1'') [d1]).
           { apply (IH (length (c2::s1'') + 1)). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_ins: exists ops, ms_seq_valid ops (c1::c2::s1'') [] /\
                           ms_seq_cost ops = merge_split_distance (c1::c2::s1'') []).
           { apply (IH (length (c1::c2::s1'') + 0)). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_sub: exists ops, ms_seq_valid ops (c2::s1'') [] /\
                           ms_seq_cost ops = merge_split_distance (c2::s1'') []).
           { apply (IH (length (c2::s1'') + 0)). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_mrg: exists ops, ms_seq_valid ops s1'' [] /\
                           ms_seq_cost ops = merge_split_distance s1'' []).
           { apply (IH (length s1'' + 0)). simpl in Hlen. simpl. lia. reflexivity. }

           destruct IH_del as [ops_del [Hv_del Hc_del]].
           destruct IH_ins as [ops_ins [Hv_ins Hc_ins]].
           destruct IH_sub as [ops_sub [Hv_sub Hc_sub]].
           destruct IH_mrg as [ops_mrg [Hv_mrg Hc_mrg]].

           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           fold (merge_split_distance (c2::s1'') [d1]).
           fold (merge_split_distance (c1::c2::s1'') []).
           fold (merge_split_distance (c2::s1'') []).
           fold (merge_split_distance s1'' []).

           set (del := merge_split_distance (c2::s1'') [d1] + 1).
           set (ins := merge_split_distance (c1::c2::s1'') [] + 1).
           set (sub := merge_split_distance (c2::s1'') [] + subst_cost c1 d1).
           set (mrg := merge_split_distance s1'' [] + merge_cost c1 c2 d1).

           destruct (Nat.min_dec (min3 del ins sub) mrg) as [Hstd | Hmrg].
           ++ (* Standard branch wins *)
              destruct (Nat.min_dec del (Nat.min ins sub)) as [Hdel | Hins_sub].
              ** (* Delete wins *)
                 exists (MSDelete c1 :: ops_del). split.
                 --- unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_del.
                 --- simpl. rewrite Hc_del.
                     rewrite Hstd. unfold min3. rewrite <- Hdel. unfold del. lia.
              ** destruct (Nat.min_dec ins sub) as [Hins | Hsub].
                 --- (* Insert wins *)
                     exists (MSInsert d1 :: ops_ins). split.
                     +++ unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_ins.
                     +++ simpl. rewrite Hc_ins.
                         rewrite Hstd. unfold min3. rewrite <- Hins_sub, <- Hins. unfold ins. lia.
                 --- (* Subst wins *)
                     exists (MSSubst c1 d1 :: ops_sub). split.
                     +++ unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_sub.
                     +++ simpl. rewrite Hc_sub.
                         rewrite Hstd. unfold min3. rewrite <- Hins_sub, <- Hsub. unfold sub. lia.
           ++ (* Merge wins *)
              exists (MSMerge c1 c2 d1 :: ops_mrg). split.
              ** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_mrg.
              ** simpl. rewrite Hc_mrg.
                 rewrite Hmrg. unfold mrg, merge_cost, can_merge. lia.

        -- (* s1 = c1::c2::s1'', s2 = d1::d2::s2'' - multi/multi, all 6 branches *)
           (* IH for all 6 branches *)
           assert (IH_del: exists ops, ms_seq_valid ops (c2::s1'') (d1::d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance (c2::s1'') (d1::d2::s2'')).
           { apply (IH (length (c2::s1'') + length (d1::d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_ins: exists ops, ms_seq_valid ops (c1::c2::s1'') (d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance (c1::c2::s1'') (d2::s2'')).
           { apply (IH (length (c1::c2::s1'') + length (d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_sub: exists ops, ms_seq_valid ops (c2::s1'') (d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance (c2::s1'') (d2::s2'')).
           { apply (IH (length (c2::s1'') + length (d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_mrg: exists ops, ms_seq_valid ops s1'' (d2::s2'') /\
                           ms_seq_cost ops = merge_split_distance s1'' (d2::s2'')).
           { apply (IH (length s1'' + length (d2::s2''))). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_spl: exists ops, ms_seq_valid ops (c2::s1'') s2'' /\
                           ms_seq_cost ops = merge_split_distance (c2::s1'') s2'').
           { apply (IH (length (c2::s1'') + length s2'')). simpl in Hlen. simpl. lia. reflexivity. }
           assert (IH_dbl: exists ops, ms_seq_valid ops s1'' s2'' /\
                           ms_seq_cost ops = merge_split_distance s1'' s2'').
           { apply (IH (length s1'' + length s2'')). simpl in Hlen. simpl. lia. reflexivity. }

           destruct IH_del as [ops_del [Hv_del Hc_del]].
           destruct IH_ins as [ops_ins [Hv_ins Hc_ins]].
           destruct IH_sub as [ops_sub [Hv_sub Hc_sub]].
           destruct IH_mrg as [ops_mrg [Hv_mrg Hc_mrg]].
           destruct IH_spl as [ops_spl [Hv_spl Hc_spl]].
           destruct IH_dbl as [ops_dbl [Hv_dbl Hc_dbl]].

           unfold merge_split_distance.
           rewrite merge_split_pair_equation.
           fold (merge_split_distance (c2::s1'') (d1::d2::s2'')).
           fold (merge_split_distance (c1::c2::s1'') (d2::s2'')).
           fold (merge_split_distance (c2::s1'') (d2::s2'')).
           fold (merge_split_distance s1'' (d2::s2'')).
           fold (merge_split_distance (c2::s1'') s2'').
           fold (merge_split_distance s1'' s2'').

           set (del := merge_split_distance (c2::s1'') (d1::d2::s2'') + 1).
           set (ins := merge_split_distance (c1::c2::s1'') (d2::s2'') + 1).
           set (sub := merge_split_distance (c2::s1'') (d2::s2'') + subst_cost c1 d1).
           set (mrg := merge_split_distance s1'' (d2::s2'') + merge_cost c1 c2 d1).
           set (spl := merge_split_distance (c2::s1'') s2'' + split_cost c1 d1 d2).
           set (dbl := merge_split_distance s1'' s2'' + subst_cost c1 d1 + subst_cost c2 d2).

           (* min6 a b c d e f = min (min5 a b c d e) f *)
           (* min5 a b c d e = min (min (min a b) (min c d)) e *)
           unfold min6. unfold min5.

           (* Case analysis on which branch wins *)
           destruct (Nat.min_dec (Nat.min (Nat.min (Nat.min del ins) (Nat.min sub mrg)) spl) dbl) as [Hmin5 | Hdbl].
           ++ (* min5 wins over double-subst *)
              destruct (Nat.min_dec (Nat.min (Nat.min del ins) (Nat.min sub mrg)) spl) as [Hmin4 | Hspl].
              ** (* min4 wins over split *)
                 destruct (Nat.min_dec (Nat.min del ins) (Nat.min sub mrg)) as [Hdel_ins | Hsub_mrg].
                 --- (* del or ins wins *)
                     destruct (Nat.min_dec del ins) as [Hdel | Hins].
                     +++ (* Delete wins *)
                         exists (MSDelete c1 :: ops_del). split.
                         *** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_del.
                         *** simpl. rewrite Hc_del.
                             rewrite <- Hmin5, <- Hmin4, <- Hdel_ins, <- Hdel. unfold del. lia.
                     +++ (* Insert wins *)
                         exists (MSInsert d1 :: ops_ins). split.
                         *** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_ins.
                         *** simpl. rewrite Hc_ins.
                             rewrite <- Hmin5, <- Hmin4, <- Hdel_ins, <- Hins. unfold ins. lia.
                 --- (* sub or mrg wins *)
                     destruct (Nat.min_dec sub mrg) as [Hsub | Hmrg].
                     +++ (* Subst wins *)
                         exists (MSSubst c1 d1 :: ops_sub). split.
                         *** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_sub.
                         *** simpl. rewrite Hc_sub.
                             rewrite <- Hmin5, <- Hmin4, <- Hsub_mrg, <- Hsub. unfold sub. lia.
                     +++ (* Merge wins *)
                         exists (MSMerge c1 c2 d1 :: ops_mrg). split.
                         *** unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_mrg.
                         *** simpl. rewrite Hc_mrg.
                             rewrite Hmin5, Hmin4, Hsub_mrg, Hmrg.
                             unfold mrg, merge_cost, can_merge. lia.
              ** (* Split wins *)
                 exists (MSSplit c1 d1 d2 :: ops_spl). split.
                 --- unfold ms_seq_valid in *. simpl. rewrite !char_eq_refl. exact Hv_spl.
                 --- simpl. rewrite Hc_spl.
                     rewrite Hmin5, Hspl.
                     unfold spl, split_cost, can_split. lia.
           ++ (* Double-subst wins *)
              (* Need two subst operations: c1→d1 and c2→d2 *)
              exists (MSSubst c1 d1 :: MSSubst c2 d2 :: ops_dbl). split.
              ** unfold ms_seq_valid in *. simpl.
                 rewrite !char_eq_refl. simpl. rewrite !char_eq_refl.
                 exact Hv_dbl.
              ** simpl. rewrite Hc_dbl.
                 rewrite <- Hdbl. unfold dbl. lia.
Qed.

(** Helper: Each operation corresponds to a branch of the DP computation.
    When apply_ms_op op s1 s2 = Some (s1', s2'), we have
    ms(s1, s2) <= ms_op_cost op + ms(s1', s2'). *)
Lemma ms_op_le_branch : forall op src tgt src' tgt',
  apply_ms_op op src tgt = Some (src', tgt') ->
  merge_split_distance src tgt <= ms_op_cost op + merge_split_distance src' tgt'.
Proof.
  intros op src tgt src' tgt' Hop.
  destruct op; simpl in Hop.

  - (* MSDelete c: src = c'::src_rest, result = (src_rest, tgt) *)
    destruct src as [| c' src_rest]; [discriminate|].
    destruct (char_eq c c') eqn:Hceq; [|discriminate].
    injection Hop as Hsrc Htgt. rewrite <- Hsrc, <- Htgt.
    (* Goal: ms(c'::src_rest, tgt) <= 1 + ms(src_rest, tgt) *)
    unfold merge_split_distance.
    destruct src_rest as [| c2 src''].
    + (* src = [c'] *)
      destruct tgt as [| d1 tgt_rest].
      * rewrite merge_split_pair_equation. simpl. lia.
      * destruct tgt_rest as [| d2 tgt''].
        -- rewrite merge_split_pair_equation. simpl.
           unfold subst_cost. destruct (char_eq c' d1); lia.
        -- (* single-multi case: min (min3 del ins subst) split *)
           rewrite merge_split_pair_equation.
           (* The delete branch = ms([], d1::d2::tgt'') + 1 *)
           (* Goal: min(...) <= 1 + ms([], d1::d2::tgt'') *)
           (* Since n + 1 = 1 + n, del_branch = RHS *)
           set (del := merge_split_pair ([], d1 :: d2 :: tgt'')).
           assert (Hdel: min (min3 (del + 1)
                                   (merge_split_pair ([c'], d2 :: tgt'') + 1)
                                   (merge_split_pair ([], d2 :: tgt'') + subst_cost c' d1))
                             (merge_split_pair ([], tgt'') + split_cost c' d1 d2)
                         <= del + 1).
           { unfold min3. lia. }
           simpl ms_op_cost. lia.
    + (* src = c'::c2::src'' *)
      destruct tgt as [| d1 tgt_rest].
      * (* tgt = [] *)
        fold (merge_split_distance (c' :: c2 :: src'') []).
        fold (merge_split_distance (c2 :: src'') []).
        rewrite !ms_empty_right. simpl. lia.
      * destruct tgt_rest as [| d2 tgt''].
        -- (* multi-single case *)
           rewrite merge_split_pair_equation.
           set (del := merge_split_pair (c2 :: src'', [d1])).
           assert (Hdel: min (min3 (del + 1)
                                   (merge_split_pair (c' :: c2 :: src'', []) + 1)
                                   (merge_split_pair (c2 :: src'', []) + subst_cost c' d1))
                             (merge_split_pair (src'', []) + merge_cost c' c2 d1)
                         <= del + 1).
           { unfold min3. lia. }
           simpl ms_op_cost. lia.
        -- (* multi-multi case: min6 branches, first is delete *)
           rewrite merge_split_pair_equation.
           set (del := merge_split_pair (c2 :: src'', d1 :: d2 :: tgt'')).
           assert (Hdel: min6 (del + 1)
                              (merge_split_pair (c' :: c2 :: src'', d2 :: tgt'') + 1)
                              (merge_split_pair (c2 :: src'', d2 :: tgt'') + subst_cost c' d1)
                              (merge_split_pair (src'', d2 :: tgt'') + merge_cost c' c2 d1)
                              (merge_split_pair (c2 :: src'', tgt'') + split_cost c' d1 d2)
                              (merge_split_pair (src'', tgt'') + subst_cost c' d1 + subst_cost c2 d2)
                         <= del + 1).
           { unfold min6, min5. lia. }
           simpl ms_op_cost. lia.

  - (* MSInsert c: tgt = d'::tgt_rest, result = (src, tgt_rest) *)
    destruct tgt as [| d' tgt_rest]; [discriminate|].
    destruct (char_eq c d') eqn:Hdeq; [|discriminate].
    injection Hop as Hsrc Htgt. rewrite <- Hsrc, <- Htgt.
    (* Goal: ms(src, d'::tgt_rest) <= 1 + ms(src, tgt_rest) *)
    unfold merge_split_distance.
    destruct src as [| c1 src_rest].
    + (* src = [] *)
      fold (merge_split_distance [] (d' :: tgt_rest)).
      fold (merge_split_distance [] tgt_rest).
      rewrite !ms_empty_left. simpl. lia.
    + destruct src_rest as [| c2 src''].
      * (* src = [c1] *)
        destruct tgt_rest as [| d2 tgt''].
        -- rewrite merge_split_pair_equation. simpl.
           unfold subst_cost. destruct (char_eq c1 d'); lia.
        -- (* single-multi case: insert is 2nd branch of min3 *)
           rewrite merge_split_pair_equation.
           set (ins := merge_split_pair ([c1], d2 :: tgt'')).
           assert (Hins: min (min3 (merge_split_pair ([], d' :: d2 :: tgt'') + 1)
                                    (ins + 1)
                                    (merge_split_pair ([], d2 :: tgt'') + subst_cost c1 d'))
                              (merge_split_pair ([], tgt'') + split_cost c1 d' d2)
                          <= ins + 1).
           { unfold min3. lia. }
           simpl ms_op_cost. lia.
      * (* src = c1::c2::src'' *)
        destruct tgt_rest as [| d2 tgt''].
        -- (* multi-single case: insert is 2nd branch of min3 *)
           rewrite merge_split_pair_equation.
           set (ins := merge_split_pair (c1 :: c2 :: src'', [])).
           assert (Hins: min (min3 (merge_split_pair (c2 :: src'', [d']) + 1)
                                    (ins + 1)
                                    (merge_split_pair (c2 :: src'', []) + subst_cost c1 d'))
                              (merge_split_pair (src'', []) + merge_cost c1 c2 d')
                          <= ins + 1).
           { unfold min3. lia. }
           simpl ms_op_cost. lia.
        -- (* multi-multi case: insert is 2nd of min6 *)
           rewrite merge_split_pair_equation.
           set (ins := merge_split_pair (c1 :: c2 :: src'', d2 :: tgt'')).
           assert (Hins: min6 (merge_split_pair (c2 :: src'', d' :: d2 :: tgt'') + 1)
                               (ins + 1)
                               (merge_split_pair (c2 :: src'', d2 :: tgt'') + subst_cost c1 d')
                               (merge_split_pair (src'', d2 :: tgt'') + merge_cost c1 c2 d')
                               (merge_split_pair (c2 :: src'', tgt'') + split_cost c1 d' d2)
                               (merge_split_pair (src'', tgt'') + subst_cost c1 d' + subst_cost c2 d2)
                          <= ins + 1).
           { unfold min6, min5. lia. }
           simpl ms_op_cost. lia.

  - (* MSSubst c c0: src = c'::src_rest, tgt = d'::tgt_rest, result = (src_rest, tgt_rest) *)
    destruct src as [| c' src_rest]; [discriminate|].
    destruct tgt as [| d' tgt_rest]; [discriminate|].
    destruct (andb (char_eq c c') (char_eq c0 d')) eqn:Hboth; [|discriminate].
    apply andb_prop in Hboth. destruct Hboth as [Hceq Hc0eq].
    (* Use char_eq_true to derive actual equalities *)
    apply char_eq_true in Hceq. apply char_eq_true in Hc0eq.
    subst c c0.
    injection Hop as Hsrc Htgt. rewrite <- Hsrc, <- Htgt.
    (* Goal: ms(c'::src_rest, d'::tgt_rest) <= subst_cost c' d' + ms(src_rest, tgt_rest) *)
    unfold merge_split_distance.
    destruct src_rest as [| c2 src''].
    + (* src = [c'] *)
      destruct tgt_rest as [| d2 tgt''].
      * (* single-single case: [c'] vs [d'] *)
        rewrite merge_split_pair_equation.
        simpl ms_op_cost.
        (* Goal: (if char_eq c' d' then 0 else 1) <= subst_cost c' d' + 0 *)
        unfold subst_cost. rewrite Nat.add_0_r.
        destruct (char_eq c' d'); reflexivity.
      * (* single-multi: subst is 3rd branch of min3 *)
        rewrite merge_split_pair_equation.
        set (sub := merge_split_pair ([], d2 :: tgt'')).
        assert (Hsub: min (min3 (merge_split_pair ([], d' :: d2 :: tgt'') + 1)
                                 (merge_split_pair ([c'], d2 :: tgt'') + 1)
                                 (sub + subst_cost c' d'))
                           (merge_split_pair ([], tgt'') + split_cost c' d' d2)
                       <= sub + subst_cost c' d').
        { unfold min3. lia. }
        simpl ms_op_cost. lia.
    + (* src = c'::c2::src'' *)
      destruct tgt_rest as [| d2 tgt''].
      * (* multi-single: subst is 3rd branch of min3 *)
        rewrite merge_split_pair_equation.
        set (sub := merge_split_pair (c2 :: src'', [])).
        assert (Hsub: min (min3 (merge_split_pair (c2 :: src'', [d']) + 1)
                                 (merge_split_pair (c' :: c2 :: src'', []) + 1)
                                 (sub + subst_cost c' d'))
                           (merge_split_pair (src'', []) + merge_cost c' c2 d')
                       <= sub + subst_cost c' d').
        { unfold min3. lia. }
        simpl ms_op_cost. lia.
      * (* multi-multi: subst is 3rd of min6 *)
        rewrite merge_split_pair_equation.
        set (sub := merge_split_pair (c2 :: src'', d2 :: tgt'')).
        assert (Hsub: min6 (merge_split_pair (c2 :: src'', d' :: d2 :: tgt'') + 1)
                            (merge_split_pair (c' :: c2 :: src'', d2 :: tgt'') + 1)
                            (sub + subst_cost c' d')
                            (merge_split_pair (src'', d2 :: tgt'') + merge_cost c' c2 d')
                            (merge_split_pair (c2 :: src'', tgt'') + split_cost c' d' d2)
                            (merge_split_pair (src'', tgt'') + subst_cost c' d' + subst_cost c2 d2)
                       <= sub + subst_cost c' d').
        { unfold min6, min5. lia. }
        simpl ms_op_cost. lia.

  - (* MSMerge c c0 c1: src = c1'::c2'::src_rest, tgt = d'::tgt_rest, result = (src_rest, tgt_rest) *)
    (* Variables: c = first source char, c0 = second source char, c1 = target char *)
    destruct src as [| c1' src_tail]; [discriminate|].
    destruct src_tail as [| c2' src_rest]; [discriminate|].
    destruct tgt as [| d' tgt_rest]; [discriminate|].
    destruct (andb (andb (char_eq c c1') (char_eq c0 c2')) (char_eq c1 d')) eqn:Hall; [|discriminate].
    apply andb_prop in Hall. destruct Hall as [H12 Hd].
    apply andb_prop in H12. destruct H12 as [Hc Hc0].
    (* Use char_eq_true to derive actual equalities *)
    apply char_eq_true in Hc. apply char_eq_true in Hc0. apply char_eq_true in Hd.
    subst c c0 c1.
    injection Hop as Hsrc Htgt. rewrite <- Hsrc, <- Htgt.
    (* Goal: ms(c1'::c2'::src_rest, d'::tgt_rest) <= merge_cost c1' c2' d' + ms(src_rest, tgt_rest) *)
    unfold merge_split_distance.
    destruct tgt_rest as [| d2 tgt''].
    + (* tgt = [d'] - multi-single case: merge is 4th branch *)
      rewrite merge_split_pair_equation.
      set (mrg := merge_split_pair (src_rest, [])).
      assert (Hmrg: min (min3 (merge_split_pair (c2' :: src_rest, [d']) + 1)
                               (merge_split_pair (c1' :: c2' :: src_rest, []) + 1)
                               (merge_split_pair (c2' :: src_rest, []) + subst_cost c1' d'))
                         (mrg + merge_cost c1' c2' d')
                     <= mrg + merge_cost c1' c2' d').
      { apply Nat.le_min_r. }
      simpl ms_op_cost. lia.
    + (* tgt = d'::d2::tgt'' - multi-multi: merge is 4th of min6 *)
      rewrite merge_split_pair_equation.
      set (mrg := merge_split_pair (src_rest, d2 :: tgt'')).
      assert (Hmrg: min6 (merge_split_pair (c2' :: src_rest, d' :: d2 :: tgt'') + 1)
                          (merge_split_pair (c1' :: c2' :: src_rest, d2 :: tgt'') + 1)
                          (merge_split_pair (c2' :: src_rest, d2 :: tgt'') + subst_cost c1' d')
                          (mrg + merge_cost c1' c2' d')
                          (merge_split_pair (c2' :: src_rest, tgt'') + split_cost c1' d' d2)
                          (merge_split_pair (src_rest, tgt'') + subst_cost c1' d' + subst_cost c2' d2)
                     <= mrg + merge_cost c1' c2' d').
      { unfold min6, min5. lia. }
      simpl ms_op_cost. lia.

  - (* MSSplit c c0 c1: src = c'::src_rest, tgt = d1'::d2'::tgt_rest, result = (src_rest, tgt_rest) *)
    (* Variables: c = source char, c0 = first target char, c1 = second target char *)
    destruct src as [| c' src_rest]; [discriminate|].
    destruct tgt as [| d1' tgt_tail]; [discriminate|].
    destruct tgt_tail as [| d2' tgt_rest]; [discriminate|].
    destruct (andb (andb (char_eq c c') (char_eq c0 d1')) (char_eq c1 d2')) eqn:Hall; [|discriminate].
    apply andb_prop in Hall. destruct Hall as [H_c0 Hc1].
    apply andb_prop in H_c0. destruct H_c0 as [Hc Hc0].
    (* Use char_eq_true to derive actual equalities *)
    apply char_eq_true in Hc. apply char_eq_true in Hc0. apply char_eq_true in Hc1.
    subst c c0 c1.
    injection Hop as Hsrc Htgt. rewrite <- Hsrc, <- Htgt.
    (* Goal: ms(c'::src_rest, d1'::d2'::tgt_rest) <= split_cost c' d1' d2' + ms(src_rest, tgt_rest) *)
    unfold merge_split_distance.
    destruct src_rest as [| c2 src''].
    + (* src = [c'] - single-multi case: split is 4th branch (after min3) *)
      rewrite merge_split_pair_equation.
      set (spl := merge_split_pair ([], tgt_rest)).
      assert (Hspl: min (min3 (merge_split_pair ([], d1' :: d2' :: tgt_rest) + 1)
                               (merge_split_pair ([c'], d2' :: tgt_rest) + 1)
                               (merge_split_pair ([], d2' :: tgt_rest) + subst_cost c' d1'))
                         (spl + split_cost c' d1' d2')
                     <= spl + split_cost c' d1' d2').
      { apply Nat.le_min_r. }
      simpl ms_op_cost. lia.
    + (* src = c'::c2::src'' - multi-multi: split is 5th of min6 *)
      rewrite merge_split_pair_equation.
      set (spl := merge_split_pair (c2 :: src'', tgt_rest)).
      assert (Hspl: min6 (merge_split_pair (c2 :: src'', d1' :: d2' :: tgt_rest) + 1)
                          (merge_split_pair (c' :: c2 :: src'', d2' :: tgt_rest) + 1)
                          (merge_split_pair (c2 :: src'', d2' :: tgt_rest) + subst_cost c' d1')
                          (merge_split_pair (src'', d2' :: tgt_rest) + merge_cost c' c2 d1')
                          (spl + split_cost c' d1' d2')
                          (merge_split_pair (src'', tgt_rest) + subst_cost c' d1' + subst_cost c2 d2')
                     <= spl + split_cost c' d1' d2').
      { unfold min6, min5. lia. }
      simpl ms_op_cost. lia.
Qed.

(** The merge-split distance is bounded above by any valid edit sequence cost.

    This follows from ms_op_le_branch: each operation gives a branch of the
    DP computation, and since ms takes the minimum, ms <= sum of op costs. *)
Lemma ms_upper_bound : forall ops s1 s2,
  ms_seq_valid ops s1 s2 ->
  merge_split_distance s1 s2 <= ms_seq_cost ops.
Proof.
  intros ops.
  induction ops as [| op ops' IH].
  - (* Empty sequence: only valid for [], [] with cost 0 *)
    intros s1 s2 Hvalid.
    apply ms_seq_empty_valid in Hvalid.
    destruct Hvalid as [H1 H2]. subst.
    rewrite ms_same. simpl. lia.
  - (* op :: ops' *)
    intros s1 s2 Hvalid.
    unfold ms_seq_valid in Hvalid. simpl in Hvalid.
    destruct (apply_ms_op op s1 s2) as [[s1' s2']|] eqn:Hop; [|discriminate].
    simpl.
    (* Use IH on the remaining ops *)
    assert (IH': merge_split_distance s1' s2' <= ms_seq_cost ops').
    { apply IH. unfold ms_seq_valid. exact Hvalid. }
    (* Use ms_op_le_branch to get the key inequality *)
    assert (Hbranch: merge_split_distance s1 s2 <= ms_op_cost op + merge_split_distance s1' s2').
    { apply ms_op_le_branch. exact Hop. }
    (* Combine: ms(s1,s2) <= op_cost + ms(s1',s2') <= op_cost + cost(ops') *)
  lia.
Qed.

(** Applying an operation is stable under appending untouched suffixes. *)
Lemma apply_ms_op_app_suffix : forall op src tgt src' tgt' src_tail tgt_tail,
  apply_ms_op op src tgt = Some (src', tgt') ->
  apply_ms_op op (src ++ src_tail) (tgt ++ tgt_tail) =
    Some (src' ++ src_tail, tgt' ++ tgt_tail).
Proof.
  intros op src tgt src' tgt' src_tail tgt_tail Hop.
  destruct op; simpl in Hop.
  - destruct src as [|c' src_rest]; [discriminate|].
    destruct (char_eq c c') eqn:Heq; [|discriminate].
    inversion Hop. subst. simpl. rewrite Heq. reflexivity.
  - destruct tgt as [|d' tgt_rest]; [discriminate|].
    destruct (char_eq c d') eqn:Heq; [|discriminate].
    inversion Hop. subst.
    simpl. rewrite Heq. reflexivity.
  - destruct src as [|c' src_rest]; [discriminate|].
    destruct tgt as [|d' tgt_rest]; [discriminate|].
    destruct (andb (char_eq c c') (char_eq c0 d')) eqn:Heq; [|discriminate].
    inversion Hop. subst. simpl. rewrite Heq. reflexivity.
  - destruct src as [|c1' [|c2' src_rest]]; [discriminate|discriminate|].
    destruct tgt as [|d' tgt_rest]; [discriminate|].
    destruct (andb (andb (char_eq c c1') (char_eq c0 c2')) (char_eq c1 d')) eqn:Heq;
      [|discriminate].
    inversion Hop. subst. simpl. rewrite Heq. reflexivity.
  - destruct src as [|c' src_rest]; [discriminate|].
    destruct tgt as [|d1' [|d2' tgt_rest]]; [discriminate|discriminate|].
    destruct (andb (andb (char_eq c c') (char_eq c0 d1')) (char_eq c1 d2')) eqn:Heq;
      [|discriminate].
    inversion Hop. subst. simpl. rewrite Heq. reflexivity.
Qed.

(** Applying a valid prefix edit sequence is stable under appending untouched suffixes. *)
Lemma apply_ms_seq_app_suffix : forall ops src tgt src' tgt' src_tail tgt_tail,
  apply_ms_seq ops src tgt = Some (src', tgt') ->
  apply_ms_seq ops (src ++ src_tail) (tgt ++ tgt_tail) =
    Some (src' ++ src_tail, tgt' ++ tgt_tail).
Proof.
  induction ops as [|op ops IH]; intros src tgt src' tgt' src_tail tgt_tail Hseq.
  - simpl in Hseq. inversion Hseq. subst. reflexivity.
  - simpl in Hseq.
    destruct (apply_ms_op op src tgt) as [[src1 tgt1]|] eqn:Hop; [|discriminate].
    simpl.
    rewrite (apply_ms_op_app_suffix op src tgt src1 tgt1 src_tail tgt_tail Hop).
    apply IH. exact Hseq.
Qed.

(** A final operation on appended suffixes gives an upper bound for the larger strings. *)
Lemma ms_distance_append_op_bound : forall op s1 s2 tail1 tail2,
  apply_ms_op op tail1 tail2 = Some ([], []) ->
  merge_split_distance (s1 ++ tail1) (s2 ++ tail2) <=
    merge_split_distance s1 s2 + ms_op_cost op.
Proof.
  intros op s1 s2 tail1 tail2 Hop.
  destruct (ms_seq_exists s1 s2) as [ops [Hvalid Hcost]].
  apply Nat.le_trans with (ms_seq_cost (ops ++ [op])).
  - apply ms_upper_bound.
    unfold ms_seq_valid in *.
    rewrite (apply_ms_seq_app ops [op] (s1 ++ tail1) (s2 ++ tail2) tail1 tail2).
    + simpl. rewrite Hop. reflexivity.
    + change (Some (tail1, tail2)) with (Some ([] ++ tail1, [] ++ tail2)).
      apply apply_ms_seq_app_suffix. exact Hvalid.
  - rewrite ms_seq_cost_app. simpl. lia.
Qed.

Lemma ms_distance_delete_last : forall s1 s2 c,
  merge_split_distance (s1 ++ [c]) s2 <= merge_split_distance s1 s2 + 1.
Proof.
  intros s1 s2 c.
  replace (merge_split_distance (s1 ++ [c]) s2)
    with (merge_split_distance (s1 ++ [c]) (s2 ++ [])) by (rewrite app_nil_r; reflexivity).
  eapply Nat.le_trans.
  - apply (ms_distance_append_op_bound (MSDelete c) s1 s2 [c] []).
    simpl. rewrite char_eq_refl. reflexivity.
  - simpl. lia.
Qed.

Lemma ms_distance_insert_last : forall s1 s2 c,
  merge_split_distance s1 (s2 ++ [c]) <= merge_split_distance s1 s2 + 1.
Proof.
  intros s1 s2 c.
  replace (merge_split_distance s1 (s2 ++ [c]))
    with (merge_split_distance (s1 ++ []) (s2 ++ [c])) by (rewrite app_nil_r; reflexivity).
  eapply Nat.le_trans.
  - apply (ms_distance_append_op_bound (MSInsert c) s1 s2 [] [c]).
    simpl. rewrite char_eq_refl. reflexivity.
  - simpl. lia.
Qed.

Lemma ms_distance_subst_last : forall s1 s2 c d,
  merge_split_distance (s1 ++ [c]) (s2 ++ [d]) <=
    merge_split_distance s1 s2 + subst_cost c d.
Proof.
  intros s1 s2 c d.
  apply (ms_distance_append_op_bound (MSSubst c d) s1 s2 [c] [d]).
  simpl. rewrite !char_eq_refl. reflexivity.
Qed.

Lemma ms_distance_match_last : forall s1 s2 c,
  merge_split_distance (s1 ++ [c]) (s2 ++ [c]) <=
    merge_split_distance s1 s2.
Proof.
  intros s1 s2 c.
  eapply Nat.le_trans.
  - apply ms_distance_subst_last.
  - unfold subst_cost. rewrite char_eq_refl. lia.
Qed.

Lemma ms_distance_merge_last : forall s1 s2 c1 c2 d,
  merge_split_distance (s1 ++ [c1; c2]) (s2 ++ [d]) <=
    merge_split_distance s1 s2 + 1.
Proof.
  intros s1 s2 c1 c2 d.
  eapply Nat.le_trans.
  - apply (ms_distance_append_op_bound (MSMerge c1 c2 d) s1 s2 [c1; c2] [d]).
    simpl. rewrite !char_eq_refl. reflexivity.
  - simpl. unfold merge_cost, can_merge. lia.
Qed.

Lemma ms_distance_split_last : forall s1 s2 c d1 d2,
  merge_split_distance (s1 ++ [c]) (s2 ++ [d1; d2]) <=
    merge_split_distance s1 s2 + 1.
Proof.
  intros s1 s2 c d1 d2.
  eapply Nat.le_trans.
  - apply (ms_distance_append_op_bound (MSSplit c d1 d2) s1 s2 [c] [d1; d2]).
    simpl. rewrite !char_eq_refl. reflexivity.
  - simpl. unfold split_cost, can_split. lia.
Qed.

(** Standard Levenshtein can simulate every merge-split operation with at
    most twice its merge-split cost. *)

Lemma subst_cost_le_one : forall c d, subst_cost c d <= 1.
Proof.
  intros c d.
  unfold subst_cost.
  destruct (char_eq c d); lia.
Qed.

Lemma merge_cost_at_least_one : forall c1 c2 d, 1 <= merge_cost c1 c2 d.
Proof.
  intros c1 c2 d.
  unfold merge_cost.
  destruct (can_merge c1 c2 d); lia.
Qed.

Lemma split_cost_at_least_one : forall c d1 d2, 1 <= split_cost c d1 d2.
Proof.
  intros c d1 d2.
  unfold split_cost.
  destruct (can_split c d1 d2); lia.
Qed.

Lemma lev_delete_step_bound : forall c s1 s2,
  lev_distance (c :: s1) s2 <= 1 + lev_distance s1 s2.
Proof.
  intros c s1 s2.
  destruct s2 as [| d s2].
  - rewrite lev_distance_empty_right.
    rewrite lev_distance_empty_right.
    simpl. lia.
  - rewrite lev_distance_cons.
    unfold min3. lia.
Qed.

Lemma lev_insert_step_bound : forall d s1 s2,
  lev_distance s1 (d :: s2) <= 1 + lev_distance s1 s2.
Proof.
  intros d s1 s2.
  rewrite lev_distance_symmetry.
  rewrite (lev_distance_symmetry s1 s2).
  apply lev_delete_step_bound.
Qed.

Lemma lev_subst_step_bound : forall c d s1 s2,
  lev_distance (c :: s1) (d :: s2) <= subst_cost c d + lev_distance s1 s2.
Proof.
  intros c d s1 s2.
  rewrite lev_distance_cons.
  unfold min3. lia.
Qed.

Lemma lev_merge_step_bound : forall c1 c2 d s1 s2,
  lev_distance (c1 :: c2 :: s1) (d :: s2) <= 2 + lev_distance s1 s2.
Proof.
  intros c1 c2 d s1 s2.
  pose proof (lev_subst_step_bound c1 d (c2 :: s1) s2) as Hsub.
  pose proof (lev_delete_step_bound c2 s1 s2) as Hdel.
  pose proof (subst_cost_le_one c1 d) as Hcost.
  lia.
Qed.

Lemma lev_split_step_bound : forall c d1 d2 s1 s2,
  lev_distance (c :: s1) (d1 :: d2 :: s2) <= 2 + lev_distance s1 s2.
Proof.
  intros c d1 d2 s1 s2.
  pose proof (lev_subst_step_bound c d1 s1 (d2 :: s2)) as Hsub.
  pose proof (lev_insert_step_bound d2 s1 s2) as Hins.
  pose proof (subst_cost_le_one c d1) as Hcost.
  lia.
Qed.

Lemma lev_ms_op_step_bound : forall op s1 s2 s1' s2',
  apply_ms_op op s1 s2 = Some (s1', s2') ->
  lev_distance s1 s2 <= 2 * ms_op_cost op + lev_distance s1' s2'.
Proof.
  intros op s1 s2 s1' s2' Hop.
  destruct op as [c | d | c d | c1 c2 d | c d1 d2].
  - destruct s1 as [| a s1_tail]; simpl in Hop; try discriminate.
    destruct (char_eq c a) eqn:Hca; try discriminate.
    injection Hop as Hs1 Hs2. subst s1' s2'.
    pose proof (lev_delete_step_bound a s1_tail s2) as Hdel.
    simpl. lia.
  - destruct s2 as [| b s2_tail]; simpl in Hop; try discriminate.
    destruct (char_eq d b) eqn:Hdb; try discriminate.
    injection Hop as Hs1 Hs2. subst s1' s2'.
    pose proof (lev_insert_step_bound b s1 s2_tail) as Hins.
    simpl. lia.
  - destruct s1 as [| a s1_tail]; simpl in Hop; try discriminate.
    destruct s2 as [| b s2_tail]; simpl in Hop; try discriminate.
    destruct (andb (char_eq c a) (char_eq d b)) eqn:Hchars; try discriminate.
    apply andb_true_iff in Hchars as [Hca Hdb].
    apply char_eq_true in Hca.
    apply char_eq_true in Hdb.
    subst c d.
    injection Hop as Hs1 Hs2. subst s1' s2'.
    pose proof (lev_subst_step_bound a b s1_tail s2_tail) as Hsub.
    simpl. lia.
  - destruct s1 as [| a [| a2 s1_rest]]; simpl in Hop; try discriminate.
    destruct s2 as [| b s2_tail]; simpl in Hop; try discriminate.
    destruct (andb (andb (char_eq c1 a) (char_eq c2 a2)) (char_eq d b)) eqn:Hchars; try discriminate.
    injection Hop as Hs1 Hs2. subst s1' s2'.
    pose proof (lev_merge_step_bound a a2 b s1_rest s2_tail) as Hmerge.
    pose proof (merge_cost_at_least_one c1 c2 d) as Hcost.
    simpl. lia.
  - destruct s1 as [| a s1_tail]; simpl in Hop; try discriminate.
    destruct s2 as [| b [| b2 s2_rest]]; simpl in Hop; try discriminate.
    destruct (andb (andb (char_eq c a) (char_eq d1 b)) (char_eq d2 b2)) eqn:Hchars; try discriminate.
    injection Hop as Hs1 Hs2. subst s1' s2'.
    pose proof (lev_split_step_bound a b b2 s1_tail s2_rest) as Hsplit.
    pose proof (split_cost_at_least_one c d1 d2) as Hcost.
    simpl. lia.
Qed.

Lemma lev_distance_ms_seq_bound : forall ops s1 s2,
  ms_seq_valid ops s1 s2 ->
  lev_distance s1 s2 <= 2 * ms_seq_cost ops.
Proof.
  induction ops as [| op ops IH]; intros s1 s2 Hvalid.
  - apply ms_seq_empty_valid in Hvalid as [Hs1 Hs2].
    subst s1 s2.
    rewrite lev_distance_empty_left.
    simpl. lia.
  - unfold ms_seq_valid in Hvalid.
    simpl in Hvalid.
    destruct (apply_ms_op op s1 s2) as [[s1' s2']|] eqn:Hop; try discriminate.
    assert (Hrest : ms_seq_valid ops s1' s2').
    { unfold ms_seq_valid. exact Hvalid. }
    pose proof (IH s1' s2' Hrest) as IHbound.
    pose proof (lev_ms_op_step_bound op s1 s2 s1' s2' Hop) as Hstep.
    simpl. lia.
Qed.

Theorem lev_distance_ms_bound : forall query dict,
  lev_distance query dict <= 2 * merge_split_distance query dict.
Proof.
  intros query dict.
  destruct (ms_seq_exists query dict) as [ops [Hvalid Hcost]].
  pose proof (lev_distance_ms_seq_bound ops query dict Hvalid) as Hbound.
  rewrite <- Hcost.
  exact Hbound.
Qed.

(** *** TRIANGLE INEQUALITY NOTE ***

    Triangle inequality: ms(s1, s3) <= ms(s1, s2) + ms(s2, s3).

    The theorem used by downstream modules is
    `Composition.MergeSplitComposition.ms_triangle_via_trace`. The earlier
    direct-induction attempt was removed from this foundational module.

    SEMANTIC SOUNDNESS:
    1. ms(s1, s2) represents the minimum cost to transform s1 into s2
    2. ms(s2, s3) represents the minimum cost to transform s2 into s3
    3. Any path s1 → s2 → s3 has total cost ms(s1,s2) + ms(s2,s3)
    4. ms(s1, s3) is the MINIMUM over ALL paths from s1 to s3
    5. Therefore ms(s1, s3) <= cost of any specific path, including via s2
*)
(** The merge-split triangle inequality is provided by the trace-composition
    theorem `Composition.MergeSplitComposition.ms_triangle_via_trace`.
    Keeping the earlier direct-induction attempt here created a proof-maintenance
    burden in a foundational module without adding a usable dependency. *)

(** NOTE: A FALSE lemma ms_eq_lev_when_no_merge_split was previously here.
    It was removed because it is FALSE: the double-subst optimization in
    merge_split_pair can produce strictly lower costs than standard
    Levenshtein distance even when merge/split predicates always return false.

    COUNTEREXAMPLE: s1 = [c1, c2], s2 = [d1, c2, c2] where d2 = c2
    - single-subst: lev([c2], [c2, c2]) + subst(c1, d1) = 1 + subst(c1, d1)
    - double-subst: lev([c2], [c2]) + subst(c1, d1) + 0 = 0 + subst(c1, d1)
    Result: double-subst < single-subst

    CORRECT RELATIONSHIP: merge_split_distance <= lev_distance
    (proven in ms_le_standard)
*)
