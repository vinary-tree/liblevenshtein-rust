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
Theorem automaton_correct_standard : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  automaton_accepts Standard query n dict = true <->
  lev_distance query dict <= n.
Proof.
  intros complete_contracts query dict n.
  split.
  - (* Soundness: accepts -> distance <= n *)
    apply automaton_sound_standard.
  - (* Completeness: distance <= n -> accepts *)
    apply (automaton_complete_standard complete_contracts).
Qed.

(** For Transposition algorithm:

    The Transposition algorithm uses Damerau-Levenshtein distance
    which allows transposition of adjacent characters at cost 1.

    The full correctness theorem:
    automaton_accepts Transposition query n dict = true <->
    damerau_lev_distance query dict <= n
*)
Theorem automaton_correct_transposition : forall
  (sound_contracts : AutomatonSoundnessContracts)
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  automaton_accepts Transposition query n dict = true <->
  damerau_lev_distance query dict <= n.
Proof.
  intros sound_contracts complete_contracts query dict n.
  split.
  - (* Soundness *)
    apply (automaton_sound_transposition sound_contracts).
  - (* Completeness *)
    apply (automaton_complete_transposition complete_contracts).
Qed.

(** Soundness direction *)
Theorem automaton_correct_transposition_sound : forall (sound_contracts : AutomatonSoundnessContracts) query dict n,
  automaton_accepts Transposition query n dict = true ->
  damerau_lev_distance query dict <= n.
Proof.
  intros sound_contracts query dict n Haccept.
  apply (automaton_sound_transposition sound_contracts).
  exact Haccept.
Qed.

(** Completeness direction *)
Theorem automaton_correct_transposition_complete : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  damerau_lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros complete_contracts query dict n Hdist.
  apply (automaton_complete_transposition complete_contracts).
  exact Hdist.
Qed.

(** Fallback with standard Levenshtein (always works since damerau <= lev) *)
Corollary automaton_correct_transposition_complete_lev : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros complete_contracts query dict n Hdist.
  apply (automaton_complete_transposition_lev complete_contracts).
  exact Hdist.
Qed.

(** For MergeAndSplit algorithm:

    The MergeAndSplit algorithm uses merge-split distance which allows:
    - Merge: two adjacent query chars → one dict char (cost 1)
    - Split: one query char → two dict chars (cost 1)

    The full correctness theorem:
    automaton_accepts MergeAndSplit query n dict = true <->
    merge_split_distance query dict <= n
*)
Theorem automaton_correct_merge_split : forall
  (sound_contracts : AutomatonSoundnessContracts)
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  automaton_accepts MergeAndSplit query n dict = true <->
  merge_split_distance query dict <= n.
Proof.
  intros sound_contracts complete_contracts query dict n.
  split.
  - (* Soundness *)
    apply (automaton_sound_merge_split sound_contracts).
  - (* Completeness *)
    apply (automaton_complete_merge_split complete_contracts).
Qed.

(** Soundness direction *)
Theorem automaton_correct_merge_split_sound : forall (sound_contracts : AutomatonSoundnessContracts) query dict n,
  automaton_accepts MergeAndSplit query n dict = true ->
  merge_split_distance query dict <= n.
Proof.
  intros sound_contracts query dict n Haccept.
  apply (automaton_sound_merge_split sound_contracts).
  exact Haccept.
Qed.

(** Completeness direction *)
Theorem automaton_correct_merge_split_complete : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  merge_split_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros complete_contracts query dict n Hdist.
  apply (automaton_complete_merge_split complete_contracts).
  exact Hdist.
Qed.

(** Fallback with standard Levenshtein (always works since merge_split <= lev) *)
Corollary automaton_correct_merge_split_complete_lev : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  lev_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros complete_contracts query dict n Hdist.
  apply (automaton_complete_merge_split_lev complete_contracts).
  exact Hdist.
Qed.

(** * Corollaries *)

(** The automaton correctly classifies all strings *)
Corollary automaton_classification : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  (automaton_accepts Standard query n dict = true /\
   lev_distance query dict <= n) \/
  (automaton_accepts Standard query n dict = false /\
   lev_distance query dict > n).
Proof.
  intros complete_contracts query dict n.
  destruct (automaton_accepts Standard query n dict) eqn:Hacc.
  - (* accepts = true *)
    left. split.
    + reflexivity.
    + apply (proj1 (automaton_correct_standard complete_contracts query dict n)).
      exact Hacc.
  - (* accepts = false *)
    right. split.
    + reflexivity.
    + (* If not accepting, distance must be > n *)
      destruct (Nat.le_gt_cases (lev_distance query dict) n) as [Hle | Hgt].
      * (* distance <= n but not accepting - contradiction *)
        pose proof (proj2 (automaton_correct_standard complete_contracts query dict n) Hle) as Haccept.
        rewrite Haccept in Hacc. discriminate.
      * exact Hgt.
Qed.

(** The automaton never misclassifies *)
Corollary no_misclassification : forall
  (complete_contracts : AutomatonCompletenessCoreContracts)
  query dict n,
  ~(automaton_accepts Standard query n dict = true /\
    lev_distance query dict > n) /\
  ~(automaton_accepts Standard query n dict = false /\
    lev_distance query dict <= n).
Proof.
  intros complete_contracts query dict n.
  split.
  - (* No false positives *)
    intros [Hacc Hgt].
    apply (proj1 (automaton_correct_standard complete_contracts query dict n)) in Hacc.
    lia.
  - (* No false negatives *)
    intros [Hacc Hle].
    pose proof (proj2 (automaton_correct_standard complete_contracts query dict n) Hle) as Haccept.
    rewrite Haccept in Hacc. discriminate.
Qed.

(** * Distance Computation Correctness *)

(** The automaton correctly computes the minimum distance. *)
Definition automaton_distance_correct_contract : Prop :=
  forall query dict n d,
  automaton_distance Standard query n dict = Some d ->
  d = lev_distance query dict \/
  (d <= n /\ lev_distance query dict <= d).

Lemma automaton_distance_correct :
  automaton_distance_correct_contract ->
  forall query dict n d,
    automaton_distance Standard query n dict = Some d ->
    d = lev_distance query dict \/
    (d <= n /\ lev_distance query dict <= d).
Proof.
  intros contract query dict n d Hdist.
  exact (contract query dict n d Hdist).
Qed.

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
Lemma automaton_accepts_monotone : forall
  (sound_contracts : AutomatonSoundnessContracts)
  (complete_contracts : AutomatonCompletenessCoreContracts)
  alg query dict n m,
  n <= m ->
  automaton_accepts alg query n dict = true ->
  automaton_accepts alg query m dict = true.
Proof.
  intros sound_contracts complete_contracts alg query dict n m Hle Hacc.
  destruct alg.
  - (* Standard *)
    apply (proj1 (automaton_correct_standard complete_contracts query dict n)) in Hacc.
    apply (proj2 (automaton_correct_standard complete_contracts query dict m)).
    lia.
  - (* Transposition *)
    apply (automaton_correct_transposition_sound sound_contracts) in Hacc.
    apply (automaton_correct_transposition_complete complete_contracts).
    lia.
  - (* MergeAndSplit *)
    apply (automaton_correct_merge_split_sound sound_contracts) in Hacc.
    apply (automaton_correct_merge_split_complete complete_contracts).
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

    Remaining contract obligations for fully instantiated verification:
    - Provide AutomatonSoundnessContracts for algorithm-specific trace soundness.
    - Provide AutomatonCompletenessCoreContracts for reachability completeness.
    - Provide automaton_distance_correct_contract for exact distance reporting.
*)
