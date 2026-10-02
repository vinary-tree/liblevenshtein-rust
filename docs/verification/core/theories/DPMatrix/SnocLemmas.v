(** * Snoc Lemmas for Levenshtein Distance

    This module provides lemmas about list suffix operations (snoc = cons on the right)
    and proves lev_distance_snoc - the key property connecting Wagner-Fischer
    recurrence to the recursive lev_distance definition.

    Part of: Liblevenshtein.Core

    Decomposed from: Distance.v (lines 8131-8334)
*)

From Coq Require Import String List Arith Ascii Bool Nat Lia Wf_nat.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.

(** * List Decomposition Lemmas *)

(** firstn (S n) l can be decomposed as prefix ++ [last_elem] *)
Lemma firstn_S_decompose :
  forall {A : Type} (l : list A) (n : nat),
    S n <= length l ->
    exists (prefix : list A) (c : A),
      firstn (S n) l = prefix ++ [c] /\
      prefix = firstn n l /\
      length prefix = n.
Proof.
  intros A l n.
  revert l.
  induction n as [| n' IH]; intros l H.
  - (* n = 0 *)
    destruct l as [| x xs].
    + simpl in H. lia.
    + exists [], x. simpl.
      split; [reflexivity | split; [reflexivity | reflexivity]].
  - (* n = S n' *)
    destruct l as [| x xs].
    + simpl in H. lia.
    + simpl in H.
      assert (H': S n' <= length xs) by lia.
      specialize (IH xs H').
      destruct IH as [prefix [c [H_eq [H_prefix H_len]]]].
      exists (x :: prefix), c.
      split.
      * simpl. f_equal. exact H_eq.
      * split.
        -- simpl. f_equal. exact H_prefix.
        -- simpl. f_equal. exact H_len.
Qed.

(** Key property: firstn (S n) l = firstn n l ++ [nth n l d] *)
Lemma firstn_S_snoc :
  forall {A : Type} (l : list A) (n : nat) (d : A),
    n < length l ->
    firstn (S n) l = firstn n l ++ [nth n l d].
Proof.
  intros A l n d H.
  revert l H.
  induction n as [| n' IH]; intros l H.
  - (* n = 0 *)
    destruct l as [| x xs].
    + simpl in H. lia.
    + simpl. reflexivity.
  - (* n = S n' *)
    destruct l as [| x xs].
    + simpl in H. lia.
    + simpl. f_equal.
      apply IH. simpl in H. lia.
Qed.

(** Helper: length of list with appended singleton *)
Lemma length_app_singleton : forall {A : Type} (l : list A) (x : A),
  length (l ++ [x]) = S (length l).
Proof. intros. rewrite length_app. simpl. lia. Qed.

(** Helper lemma: firstn of full length is identity *)
Lemma firstn_all :
  forall {A : Type} (l : list A),
    firstn (length l) l = l.
Proof.
  intros A l.
  induction l as [| x xs IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

(** * min3 Arithmetic Lemmas *)

Lemma min3_add_right : forall a b c k,
  min3 a b c + k = min3 (a + k) (b + k) (c + k).
Proof.
  intros a b c k.
  unfold min3.
  rewrite Nat.add_min_distr_r.
  rewrite Nat.add_min_distr_r.
  reflexivity.
Qed.

(** Taking the minimum of a 3-by-3 matrix is independent of whether its
    rows or columns are folded first. This proof exposes each of the nine
    entries through the order laws, without splitting on minimum operands. *)
Lemma min3_matrix_transpose :
  forall a b c d e f g h i,
    min3 (min3 a b c) (min3 d e f) (min3 g h i) =
    min3 (min3 a d g) (min3 b e h) (min3 c f i).
Proof.
  intros a b c d e f g h i.
  unfold min3.
  apply Nat.le_antisymm.
  - repeat apply Nat.min_glb.
    all: repeat rewrite Nat.min_le_iff; auto 12 using Nat.le_refl.
  - repeat apply Nat.min_glb.
    all: repeat rewrite Nat.min_le_iff; auto 12 using Nat.le_refl.
Qed.

(** Key arithmetic lemma for the inductive case of lev_distance_snoc.
    Shows that the min3 structure rearranges correctly when we swap
    front-processing (d cost) and back-processing (e cost). *)
Lemma min3_snoc_key :
  forall x1 x2 x3 y1 y2 y3 z1 z2 z3 d e,
    min3 (min3 (x1+1) (x2+1) (x3+d) + 1)
         (min3 (y1+1) (y2+1) (y3+d) + 1)
         (min3 (z1+1) (z2+1) (z3+d) + e)
    = min3 (min3 (x1+1) (y1+1) (z1+e) + 1)
           (min3 (x2+1) (y2+1) (z2+e) + 1)
           (min3 (x3+1) (y3+1) (z3+e) + d).
Proof.
  intros.
  repeat rewrite min3_add_right.
  rewrite min3_matrix_transpose.
  f_equal; f_equal; lia.
Qed.

(** * Suffix Version of lev_distance Recurrence *)

(** Auxiliary lemma with explicit bound for strong induction *)
Lemma lev_distance_snoc_aux :
  forall n s1 s2 c1 c2,
    length s1 + length s2 <= n ->
    lev_distance (s1 ++ [c1]) (s2 ++ [c2]) =
      min3 (lev_distance s1 (s2 ++ [c2]) + 1)
           (lev_distance (s1 ++ [c1]) s2 + 1)
           (lev_distance s1 s2 + subst_cost c1 c2).
Proof.
  intro n. induction n as [n IH] using lt_wf_ind.
  intros s1 s2 c1 c2 Hlen.

  destruct s1 as [| a s1']; destruct s2 as [| b s2'].

  - (* Case: s1 = [], s2 = [] *)
    simpl. rewrite lev_distance_cons.
    rewrite !lev_distance_empty_left, !lev_distance_empty_right.
    simpl. reflexivity.

  - (* Case: s1 = [], s2 = b :: s2' *)
    simpl app. rewrite lev_distance_cons. rewrite !lev_distance_empty_left.

    assert (IH_raw : lev_distance [c1] (s2' ++ [c2]) =
                     min3 (lev_distance [] (s2' ++ [c2]) + 1)
                          (lev_distance [c1] s2' + 1)
                          (lev_distance [] s2' + subst_cost c1 c2)).
    { change [c1] with ([] ++ [c1]). apply (IH (length s2')).
      - simpl in *. lia.
      - simpl. lia. }
    rewrite !lev_distance_empty_left in IH_raw.

    rewrite IH_raw.
    rewrite lev_distance_cons. rewrite !lev_distance_empty_left.

    simpl length.
    rewrite !length_app_singleton.

    set (x := lev_distance [c1] s2').
    set (nn := length s2').
    set (d := subst_cost c1 c2).
    set (e := subst_cost c1 b).

    unfold min3. lia.

  - (* Case: s1 = a :: s1', s2 = [] *)
    simpl app. rewrite lev_distance_cons. rewrite !lev_distance_empty_right.

    assert (IH_raw : lev_distance (s1' ++ [c1]) [c2] =
                     min3 (lev_distance s1' [c2] + 1)
                          (lev_distance (s1' ++ [c1]) [] + 1)
                          (lev_distance s1' [] + subst_cost c1 c2)).
    { change [c2] with ([] ++ [c2]). apply (IH (length s1')).
      - simpl in *. lia.
      - simpl. lia. }
    rewrite !lev_distance_empty_right in IH_raw.

    rewrite IH_raw.
    rewrite lev_distance_cons. rewrite !lev_distance_empty_right.

    simpl length.
    rewrite !length_app_singleton.

    set (x := lev_distance s1' [c2]).
    set (nn := length s1').
    set (d := subst_cost c1 c2).
    set (e := subst_cost a c2).

    unfold min3. lia.

  - (* Case: s1 = a :: s1', s2 = b :: s2' - the main inductive case *)
    simpl app. rewrite lev_distance_cons.

    (* IH1: For lev_distance (s1' ++ [c1]) (b :: s2' ++ [c2]) *)
    assert (IH1: lev_distance (s1' ++ [c1]) (b :: s2' ++ [c2]) =
                 min3 (lev_distance s1' (b :: s2' ++ [c2]) + 1)
                      (lev_distance (s1' ++ [c1]) (b :: s2') + 1)
                      (lev_distance s1' (b :: s2') + subst_cost c1 c2)).
    { change (b :: s2' ++ [c2]) with ((b :: s2') ++ [c2]).
      apply (IH (length s1' + length (b :: s2'))).
      - simpl in *. lia.
      - simpl. lia. }

    (* IH2: For lev_distance (a :: s1' ++ [c1]) (s2' ++ [c2]) *)
    assert (IH2: lev_distance (a :: s1' ++ [c1]) (s2' ++ [c2]) =
                 min3 (lev_distance (a :: s1') (s2' ++ [c2]) + 1)
                      (lev_distance (a :: s1' ++ [c1]) s2' + 1)
                      (lev_distance (a :: s1') s2' + subst_cost c1 c2)).
    { change (a :: s1' ++ [c1]) with ((a :: s1') ++ [c1]).
      apply (IH (length (a :: s1') + length s2')).
      - simpl in *. lia.
      - simpl. lia. }

    (* IH3: For lev_distance (s1' ++ [c1]) (s2' ++ [c2]) *)
    assert (IH3: lev_distance (s1' ++ [c1]) (s2' ++ [c2]) =
                 min3 (lev_distance s1' (s2' ++ [c2]) + 1)
                      (lev_distance (s1' ++ [c1]) s2' + 1)
                      (lev_distance s1' s2' + subst_cost c1 c2)).
    { apply (IH (length s1' + length s2')).
      - simpl in *. lia.
      - simpl. lia. }

    rewrite IH1, IH2, IH3.
    rewrite (lev_distance_cons a b s1' (s2' ++ [c2])).
    rewrite (lev_distance_cons a b (s1' ++ [c1]) s2').
    rewrite (lev_distance_cons a b s1' s2').

    apply min3_snoc_key.
Qed.

(** Suffix version of lev_distance_cons: recursion on the last elements.
    This is the key property that connects Wagner-Fischer recurrence to lev_distance.

    This property follows from the fact that edit distance can be computed either
    by peeling from the front (as in lev_distance_cons) or from the back.
    Both approaches yield the same result because edit distance measures the
    minimum-cost alignment, and the order of processing doesn't affect the optimal cost.

    The proof uses strong induction on length s1 + length s2 with the key insight
    that the min3 structure rearranges correctly (min3_snoc_key lemma). *)
Lemma lev_distance_snoc :
  forall (s1 s2 : list Char) (c1 c2 : Char),
    lev_distance (s1 ++ [c1]) (s2 ++ [c2]) =
      min3 (lev_distance s1 (s2 ++ [c2]) + 1)
           (lev_distance (s1 ++ [c1]) s2 + 1)
           (lev_distance s1 s2 + subst_cost c1 c2).
Proof.
  intros s1 s2 c1 c2.
  apply (lev_distance_snoc_aux (length s1 + length s2)). lia.
Qed.
