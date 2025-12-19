(** * Main Correctness Theorem for Levenshtein Automata

    This module provides the unified correctness theorem combining
    soundness and completeness.

    Part of: Liblevenshtein.Core.Automaton

    Main Result:
    - automaton_correct: automaton_accepts alg query n dict = true <->
                         lev_distance query dict <= n (for Standard algorithm)

    This establishes that Levenshtein automata are a correct implementation
    of approximate string matching with edit distance bounds.
*)

From Stdlib Require Import Arith Bool List Nat Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.
From Liblevenshtein.Core Require Import Automaton.Position.
From Liblevenshtein.Core Require Import Automaton.State.
From Liblevenshtein.Core Require Import Automaton.Acceptance.
From Liblevenshtein.Core Require Import Automaton.Soundness.
From Liblevenshtein.Core Require Import Automaton.Completeness.

(** * Main Correctness Theorem *)

(** The Standard Levenshtein automaton is correct:
    it accepts a dictionary word if and only if the edit distance
    from the query to that word is within the specified bound.

    This is the combination of soundness and completeness:
    - Soundness: accepts -> distance <= n (no false positives)
    - Completeness: distance <= n -> accepts (no false negatives)
*)
Theorem automaton_correct_standard : forall query dict n,
  automaton_accepts Standard query n dict = true <->
  lev_distance query dict <= n.
Proof.
  intros query dict n.
  split.
  - (* Soundness: accepts -> distance <= n *)
    apply automaton_sound_standard.
  - (* Completeness: distance <= n -> accepts *)
    apply automaton_complete_standard.
Qed.

(** For Transposition algorithm:

    The Transposition algorithm uses Damerau-Levenshtein distance
    which allows transposition of adjacent characters at cost 1.

    The full correctness theorem:
    automaton_accepts Transposition query n dict = true <->
    damerau_lev_distance query dict <= n
*)
Theorem automaton_correct_transposition : forall query dict n,
  automaton_accepts Transposition query n dict = true <->
  damerau_lev_distance query dict <= n.
Proof.
  intros query dict n.
  split.
  - (* Soundness *)
    apply automaton_sound_transposition.
  - (* Completeness *)
    apply automaton_complete_transposition.
Qed.

(** Soundness direction *)
Theorem automaton_correct_transposition_sound : forall query dict n,
  automaton_accepts Transposition query n dict = true ->
  damerau_lev_distance query dict <= n.
Proof.
  apply automaton_sound_transposition.
Qed.

(** Completeness direction *)
Theorem automaton_correct_transposition_complete : forall query dict n,
  damerau_lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  apply automaton_complete_transposition.
Qed.

(** Fallback with standard Levenshtein (always works since damerau <= lev) *)
Corollary automaton_correct_transposition_complete_lev : forall query dict n,
  lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  apply automaton_complete_transposition_lev.
Qed.

(** For MergeAndSplit algorithm:

    The MergeAndSplit algorithm uses merge-split distance which allows:
    - Merge: two adjacent query chars → one dict char (cost 1)
    - Split: one query char → two dict chars (cost 1)

    The full correctness theorem:
    automaton_accepts MergeAndSplit query n dict = true <->
    merge_split_distance query dict <= n
*)
Theorem automaton_correct_merge_split : forall query dict n,
  automaton_accepts MergeAndSplit query n dict = true <->
  merge_split_distance query dict <= n.
Proof.
  intros query dict n.
  split.
  - (* Soundness *)
    apply automaton_sound_merge_split.
  - (* Completeness *)
    apply automaton_complete_merge_split.
Qed.

(** Soundness direction *)
Theorem automaton_correct_merge_split_sound : forall query dict n,
  automaton_accepts MergeAndSplit query n dict = true ->
  merge_split_distance query dict <= n.
Proof.
  apply automaton_sound_merge_split.
Qed.

(** Completeness direction *)
Theorem automaton_correct_merge_split_complete : forall query dict n,
  merge_split_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  apply automaton_complete_merge_split.
Qed.

(** Fallback with standard Levenshtein (always works since merge_split <= lev) *)
Corollary automaton_correct_merge_split_complete_lev : forall query dict n,
  lev_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  apply automaton_complete_merge_split_lev.
Qed.

(** * Corollaries *)

(** The automaton correctly classifies all strings *)
Corollary automaton_classification : forall query dict n,
  (automaton_accepts Standard query n dict = true /\
   lev_distance query dict <= n) \/
  (automaton_accepts Standard query n dict = false /\
   lev_distance query dict > n).
Proof.
  intros query dict n.
  destruct (automaton_accepts Standard query n dict) eqn:Hacc.
  - (* accepts = true *)
    left. split.
    + reflexivity.
    + apply automaton_correct_standard. exact Hacc.
  - (* accepts = false *)
    right. split.
    + reflexivity.
    + (* If not accepting, distance must be > n *)
      destruct (Nat.le_gt_cases (lev_distance query dict) n) as [Hle | Hgt].
      * (* distance <= n but not accepting - contradiction *)
        apply automaton_correct_standard in Hle.
        rewrite Hle in Hacc. discriminate.
      * exact Hgt.
Qed.

(** The automaton never misclassifies *)
Corollary no_misclassification : forall query dict n,
  ~(automaton_accepts Standard query n dict = true /\
    lev_distance query dict > n) /\
  ~(automaton_accepts Standard query n dict = false /\
    lev_distance query dict <= n).
Proof.
  intros query dict n.
  split.
  - (* No false positives *)
    intros [Hacc Hgt].
    apply automaton_correct_standard in Hacc.
    lia.
  - (* No false negatives *)
    intros [Hacc Hle].
    apply automaton_correct_standard in Hle.
    rewrite Hle in Hacc. discriminate.
Qed.

(** * Distance Computation Correctness *)

(** The automaton correctly computes the minimum distance *)
Theorem automaton_distance_correct : forall query dict n d,
  automaton_distance Standard query n dict = Some d ->
  d = lev_distance query dict \/
  (d <= n /\ lev_distance query dict <= d).
Proof.
  intros query dict n d Hdist.
  (* The automaton returns the minimum error count among final positions.
     We combine soundness and completeness:
     1. From Some d: automaton accepted, so lev_distance <= n (by soundness)
     2. d is the minimum num_errors among final positions
     3. lev_distance <= d (from reachable_final_to_distance)
     4. Since lev_distance <= n (from soundness), completeness gives optimal path
     5. Therefore d = lev_distance

     Strategy: Show lev_distance <= d, then show d = lev_distance using the
     fact that if lev_distance <= n, the optimal path is explored. *)

  (* Step 1: From Some d, the automaton accepted *)
  assert (Haccepts : automaton_accepts Standard query n dict = true).
  { unfold automaton_distance in Hdist.
    destruct (automaton_run_from_initial Standard query n dict) as [final|] eqn:Hrun.
    - (* Some final - check if state_accepting_distance returns Some *)
      unfold automaton_accepts. rewrite Hrun.
      unfold state_is_final.
      destruct (state_accepting_distance final) as [d'|] eqn:Hdist'.
      + (* Some d' means there's a final position *)
        apply accepting_distance_achieves in Hdist'.
        destruct Hdist' as [p [Hin [Hfinal _]]].
        rewrite existsb_exists.
        exists p. split; [exact Hin | exact Hfinal].
      + (* None means no final position, but Hdist says Some d - contradiction *)
        simpl in Hdist. discriminate.
    - (* None - contradiction with Hdist *)
      simpl in Hdist. discriminate. }

  (* Step 2: From soundness, lev_distance <= n *)
  assert (Hle_n : lev_distance query dict <= n).
  { apply automaton_sound_standard. exact Haccepts. }

  (* Step 3: Use completeness - since lev_distance <= n, d = lev_distance *)
  left.
  (* The proof requires showing that:
     - The automaton explores the optimal path (from completeness)
     - The minimum error count equals lev_distance
     This requires the full completeness infrastructure including
     reachable_final_implies_accepts which is currently Admitted.

     For now, we use the weaker result that follows from soundness. *)

  (* From the soundness proof infrastructure, we know that any position p
     in the final state satisfies lev_distance <= num_errors p.
     The minimum d satisfies lev_distance <= d.

     From completeness, if lev_distance <= n, then the optimal path is explored,
     giving a position with num_errors = lev_distance. So d <= lev_distance.

     Together: d = lev_distance. *)

  (* This proof depends on reachable_final_implies_accepts being true.
     With the current admitted lemmas, we cannot complete this proof. *)
Admitted.

(** * Decidability *)

(** Edit distance comparison is decidable *)
Lemma lev_distance_decidable : forall query dict n,
  {lev_distance query dict <= n} + {lev_distance query dict > n}.
Proof.
  intros query dict n.
  destruct (lev_distance query dict <=? n) eqn:Hle.
  - left. apply Nat.leb_le. exact Hle.
  - right. apply Nat.leb_gt. exact Hle.
Qed.

(** Automaton acceptance is decidable (trivially, as it's a boolean) *)
Lemma automaton_accepts_decidable : forall alg query n dict,
  {automaton_accepts alg query n dict = true} +
  {automaton_accepts alg query n dict = false}.
Proof.
  intros alg query n dict.
  destruct (automaton_accepts alg query n dict).
  - left. reflexivity.
  - right. reflexivity.
Qed.

(** * Monotonicity Properties *)

(** Increasing the distance bound preserves acceptance *)
Lemma automaton_accepts_monotone : forall alg query dict n m,
  n <= m ->
  automaton_accepts alg query n dict = true ->
  automaton_accepts alg query m dict = true.
Proof.
  intros alg query dict n m Hle Hacc.
  destruct alg.
  - (* Standard *)
    apply automaton_correct_standard in Hacc.
    apply automaton_correct_standard.
    lia.
  - (* Transposition *)
    apply automaton_correct_transposition_sound in Hacc.
    apply automaton_correct_transposition_complete.
    lia.
  - (* MergeAndSplit *)
    apply automaton_correct_merge_split_sound in Hacc.
    apply automaton_correct_merge_split_complete.
    lia.
Qed.

(** * Summary *)

(** This module establishes the correctness of Levenshtein automata:

    1. **Soundness** (automaton_sound): If the automaton accepts,
       the edit distance is within bound. No false positives.

    2. **Completeness** (automaton_complete): If the edit distance
       is within bound, the automaton accepts. No false negatives.

    3. **Correctness**: The automaton accepts if and only if the
       distance is within bound:
       - Standard: lev_distance (automaton_correct_standard)
       - Transposition: damerau_lev_distance (automaton_correct_transposition)
       - MergeAndSplit: merge_split_distance (automaton_correct_merge_split)

    4. **Monotonicity**: Increasing the bound preserves acceptance.

    5. **Decidability**: All relevant predicates are decidable.

    **Distance Functions Defined**:
    - lev_distance: Standard Levenshtein distance (insert, delete, substitute)
    - damerau_lev_distance: Damerau-Levenshtein distance (+ transposition)
    - merge_split_distance: Merge-split distance (+ merge and split operations)

    **Key Relationships**:
    - damerau_lev_distance <= lev_distance (transposition can only help)
    - merge_split_distance <= lev_distance (merge/split can only help)

    Remaining work for complete verification:
    - Complete the reachable_final_to_distance proof in Soundness.v
    - Complete the optimal_sequence_exists proof in Completeness.v
    - Complete the traceable_implies_reachable proof in Completeness.v
    - Complete the helper lemmas in Soundness.v and Completeness.v
*)

