(** * Damerau-Levenshtein Distance Definition

    This module defines the Damerau-Levenshtein distance function, which extends
    standard Levenshtein distance with transposition of adjacent characters.

    Part of: Liblevenshtein.Core

    Operations:
    - Insertion: cost 1
    - Deletion: cost 1
    - Substitution: cost 1 (0 for match)
    - Transposition: cost 1 (swap adjacent characters)

    Key property: damerau_lev_distance(s1, s2) <= lev_distance(s1, s2)
    since transposition is always at least as good as two substitutions.

    NOTE: This module provides the distance function and all basic lemmas.
    The recurrence is the restricted adjacent-transposition variant; the
    unrestricted triangle inequality is false for this executable model, with a
    formal counterexample in Composition/DamerauComposition.v.
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia Wf_nat.
From Stdlib Require Import Recdef.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.MetricProperties.

(** * Helper Functions *)

(** Absolute difference of two natural numbers *)
Definition abs_diff (n m : nat) : nat :=
  if n <=? m then m - n else n - m.

Lemma abs_diff_sym : forall n m, abs_diff n m = abs_diff m n.
Proof.
  intros n m. unfold abs_diff.
  destruct (n <=? m) eqn:Hnm; destruct (m <=? n) eqn:Hmn.
  - apply Nat.leb_le in Hnm. apply Nat.leb_le in Hmn. lia.
  - reflexivity.
  - reflexivity.
  - apply Nat.leb_nle in Hnm. apply Nat.leb_nle in Hmn. lia.
Qed.

Lemma abs_diff_bounds : forall n m,
  n <= m + abs_diff n m /\ m <= n + abs_diff n m.
Proof.
  intros n m. unfold abs_diff.
  destruct (n <=? m) eqn:Hnm.
  - apply Nat.leb_le in Hnm. lia.
  - apply Nat.leb_nle in Hnm. lia.
Qed.

Lemma abs_diff_0_l : forall n, abs_diff 0 n = n.
Proof.
  intros n. unfold abs_diff. simpl. lia.
Qed.

Lemma abs_diff_0_r : forall n, abs_diff n 0 = n.
Proof.
  intros n. unfold abs_diff.
  destruct (n <=? 0) eqn:H.
  - apply Nat.leb_le in H. lia.
  - lia.
Qed.

Lemma abs_diff_le : forall n m, abs_diff n m <= Nat.max n m.
Proof.
  intros n m. unfold abs_diff.
  destruct (n <=? m) eqn:H.
  - apply Nat.leb_le in H. lia.
  - apply Nat.leb_nle in H. lia.
Qed.

Lemma abs_diff_ge_diff : forall n m, n - m <= abs_diff n m.
Proof.
  intros n m. unfold abs_diff.
  destruct (n <=? m) eqn:H.
  - apply Nat.leb_le in H. lia.
  - apply Nat.leb_nle in H. lia.
Qed.

(** Character equality reflexivity *)
Lemma char_eq_refl : forall c, char_eq c c = true.
Proof.
  intro c. unfold char_eq.
  destruct (ascii_dec c c) as [Heq | Hneq].
  - reflexivity.
  - exfalso. apply Hneq. reflexivity.
Qed.

(** Character equality symmetry *)
Lemma char_eq_sym : forall c1 c2, char_eq c1 c2 = char_eq c2 c1.
Proof.
  intros c1 c2. unfold char_eq.
  destruct (ascii_dec c1 c2) as [H12 | H12];
  destruct (ascii_dec c2 c1) as [H21 | H21]; try reflexivity.
  - exfalso. apply H21. symmetry. exact H12.
  - exfalso. apply H12. symmetry. exact H21.
Qed.

(** Substitution cost symmetry *)
Lemma subst_cost_sym : forall c1 c2, subst_cost c1 c2 = subst_cost c2 c1.
Proof.
  intros c1 c2. unfold subst_cost.
  rewrite char_eq_sym. reflexivity.
Qed.

(** Minimum of four natural numbers *)
Definition min4 (a b c d : nat) : nat :=
  min (min a b) (min c d).

(** Minimum of five natural numbers *)
Definition min5 (a b c d e : nat) : nat :=
  min (min4 a b c d) e.

(** min3 with first two arguments swapped equals min3 *)
Lemma min3_swap12 : forall a b c,
  min a (min b c) = min b (min a c).
Proof.
  intros a b c.
  (* Both sides compute min(a, b, c), just with different associativity *)
  destruct (Nat.le_ge_cases a b) as [Hab | Hab];
  destruct (Nat.le_ge_cases a c) as [Hac | Hac];
  destruct (Nat.le_ge_cases b c) as [Hbc | Hbc]; lia.
Qed.

(** min4 with swapped first two arguments *)
Lemma min4_swap12 : forall a b c d,
  min4 a b c d = min4 b a c d.
Proof.
  intros a b c d.
  unfold min4.
  destruct (Nat.le_ge_cases a b) as [Hab | Hab]; lia.
Qed.

(** Transposition cost: returns 1 if c1=d2 and c1'=d1, else a large value *)
Definition trans_cost_calc (c1 c1' d1 d2 : Char) : nat :=
  if andb (char_eq c1 d2) (char_eq c1' d1) then 1 else 100.

(** Transposition cost symmetry *)
Lemma trans_cost_calc_sym : forall c1 c1' c2 c2',
  trans_cost_calc c1 c1' c2 c2' = trans_cost_calc c2 c2' c1 c1'.
Proof.
  intros c1 c1' c2 c2'. unfold trans_cost_calc.
  rewrite (char_eq_sym c1 c2').
  rewrite (char_eq_sym c1' c2).
  rewrite Bool.andb_comm.
  reflexivity.
Qed.

(** * Measure for Termination *)

Definition damerau_measure (p : list Char * list Char) : nat :=
  length (fst p) + length (snd p).

(** * Main Definition using Function

    To avoid the "nested-recursive" issue, we structure the function
    so that all recursive calls appear at the top level of each branch,
    not inside conditionals. The transposition cost is added separately.
*)
Function damerau_lev_pair (p : list Char * list Char) {measure damerau_measure p} : nat :=
  match p with
  | ([], s2) => length s2
  | (s1, []) => length s1
  | ([c1], [c2]) =>
      if char_eq c1 c2 then 0 else 1
  | ([c1], c2 :: c2' :: s2') =>
      min3 (damerau_lev_pair ([], c2 :: c2' :: s2') + 1)
           (damerau_lev_pair ([c1], c2' :: s2') + 1)
           (damerau_lev_pair ([], c2' :: s2') + subst_cost c1 c2)
  | (c1 :: c1' :: s1', [c2]) =>
      min3 (damerau_lev_pair (c1' :: s1', [c2]) + 1)
           (damerau_lev_pair (c1 :: c1' :: s1', []) + 1)
           (damerau_lev_pair (c1' :: s1', []) + subst_cost c1 c2)
  | (c1 :: c1' :: s1', c2 :: c2' :: s2') =>
      (* Both strings have ≥2 characters - transposition possible *)
      (* Standard Damerau-Levenshtein: del, ins, subst, trans *)
      min4 (damerau_lev_pair (c1' :: s1', c2 :: c2' :: s2') + 1)   (* del *)
           (damerau_lev_pair (c1 :: c1' :: s1', c2' :: s2') + 1)   (* ins *)
           (damerau_lev_pair (c1' :: s1', c2' :: s2') + subst_cost c1 c2)  (* subst *)
           (damerau_lev_pair (s1', s2') + trans_cost_calc c1 c1' c2 c2')  (* transposition *)
  end.
Proof.
  (* Termination proofs - each recursive call decreases measure *)
  all: intros; unfold damerau_measure; simpl; lia.
Defined.

(** Wrapper function with standard signature *)
Definition damerau_lev_distance (s1 s2 : list Char) : nat :=
  damerau_lev_pair (s1, s2).

(** * Unfolding Lemmas *)

Lemma damerau_lev_empty_left :
  forall s, damerau_lev_distance [] s = length s.
Proof.
  intro s.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  reflexivity.
Qed.

Lemma damerau_lev_empty_right :
  forall s, damerau_lev_distance s [] = length s.
Proof.
  intro s.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  destruct s as [| c s'].
  - reflexivity.
  - destruct s' as [| c' s''].
    + reflexivity.
    + reflexivity.
Qed.

Lemma damerau_lev_single :
  forall c1 c2, damerau_lev_distance [c1] [c2] =
    if char_eq c1 c2 then 0 else 1.
Proof.
  intros c1 c2.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  reflexivity.
Qed.

(** Unfolding for single vs multi (2+) *)
Lemma damerau_lev_single_multi : forall c1 c2 c2' s2',
  damerau_lev_distance [c1] (c2 :: c2' :: s2') =
  min3 (damerau_lev_distance [] (c2 :: c2' :: s2') + 1)
       (damerau_lev_distance [c1] (c2' :: s2') + 1)
       (damerau_lev_distance [] (c2' :: s2') + subst_cost c1 c2).
Proof.
  intros.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  reflexivity.
Qed.

(** Unfolding for multi (2+) vs single *)
Lemma damerau_lev_multi_single : forall c1 c1' s1' c2,
  damerau_lev_distance (c1 :: c1' :: s1') [c2] =
  min3 (damerau_lev_distance (c1' :: s1') [c2] + 1)
       (damerau_lev_distance (c1 :: c1' :: s1') [] + 1)
       (damerau_lev_distance (c1' :: s1') [] + subst_cost c1 c2).
Proof.
  intros.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  reflexivity.
Qed.

(** * Key Lemma: Transposition Case *)

Lemma damerau_lev_cons2 : forall c1 c1' s1' c2 c2' s2',
  damerau_lev_distance (c1 :: c1' :: s1') (c2 :: c2' :: s2') =
  min4 (damerau_lev_distance (c1' :: s1') (c2 :: c2' :: s2') + 1)
       (damerau_lev_distance (c1 :: c1' :: s1') (c2' :: s2') + 1)
       (damerau_lev_distance (c1' :: s1') (c2' :: s2') + subst_cost c1 c2)
       (damerau_lev_distance s1' s2' + trans_cost_calc c1 c1' c2 c2').
Proof.
  intros.
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  reflexivity.
Qed.

(** * Key Property: Damerau-Levenshtein ≤ Standard Levenshtein *)

Lemma damerau_lev_le_standard : forall s1 s2,
  damerau_lev_distance s1 s2 <= lev_distance s1 s2.
Proof.
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite damerau_lev_empty_left.
    rewrite lev_distance_empty_left.
    lia.
  - destruct s2 as [| c2 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite damerau_lev_empty_right.
      rewrite lev_distance_empty_right.
      lia.
    + destruct s1' as [| c1' s1''].
      * (* s1 = [c1] *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = [c1], s2 = [c2] *)
           rewrite damerau_lev_single.
           rewrite lev_distance_cons.
           (* lev_distance [c1] [c2] = min3 (lev [] [c2] + 1) (lev [c1] [] + 1) (lev [] [] + subst) *)
           rewrite lev_distance_empty_left.
           rewrite lev_distance_empty_right.
           rewrite lev_distance_empty_left.
           simpl.
           (* Now: (if char_eq c1 c2 then 0 else 1) <= min3 2 2 (subst_cost c1 c2) *)
           unfold min3, subst_cost, char_eq.
           destruct (ascii_dec c1 c2); simpl; lia.
        -- (* s1 = [c1], s2 = c2::c2'::s2'' *)
           rewrite damerau_lev_single_multi.
           rewrite lev_distance_cons.
           unfold min3.
           (* Both have structure: min A (min B C) *)
           (* We prove: min D (min I S_d) <= min D' (min I' S_l) *)
           (* where D = del, I = ins, S_d/S_l = subst with dam/lev subdistances *)
           assert (Hdel : damerau_lev_distance [] (c2 :: c2' :: s2'') <=
                          lev_distance [] (c2 :: c2' :: s2'')).
           { apply IH with (m := 0 + length (c2 :: c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : damerau_lev_distance [c1] (c2' :: s2'') <=
                          lev_distance [c1] (c2' :: s2'')).
           { apply IH with (m := 1 + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : damerau_lev_distance [] (c2' :: s2'') <=
                          lev_distance [] (c2' :: s2'')).
           { apply IH with (m := 0 + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           (* Now use lia with all bounds established *)
           lia.
      * (* s1 = c1::c1'::s1'' *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = c1::c1'::s1'', s2 = [c2] *)
           rewrite damerau_lev_multi_single.
           rewrite lev_distance_cons.
           unfold min3.
           assert (Hdel : damerau_lev_distance (c1' :: s1'') [c2] <=
                          lev_distance (c1' :: s1'') [c2]).
           { apply IH with (m := length (c1' :: s1'') + 1).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : damerau_lev_distance (c1 :: c1' :: s1'') [] <=
                          lev_distance (c1 :: c1' :: s1'') []).
           { apply IH with (m := length (c1 :: c1' :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : damerau_lev_distance (c1' :: s1'') [] <=
                          lev_distance (c1' :: s1'') []).
           { apply IH with (m := length (c1' :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           lia.
        -- (* s1 = c1::c1'::s1'', s2 = c2::c2'::s2'' - main case *)
           rewrite damerau_lev_cons2.
           rewrite lev_distance_cons.
           (* min4 (del, ins, subst, trans) vs min3 (del, ins, subst) *)
           (* Strategy: Show min4 is minimum of 4 options that includes del, ins, subst.
              Since taking min of more options can only make it smaller or equal,
              and each damerau option is <= corresponding lev option by IH,
              the result follows. *)
           unfold min4, min3.
           (* Establish IH bounds for the three shared operations *)
           assert (Hdel : damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') <=
                          lev_distance (c1' :: s1'') (c2 :: c2' :: s2'')).
           { apply IH with (m := length (c1' :: s1'') + length (c2 :: c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hins : damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') <=
                          lev_distance (c1 :: c1' :: s1'') (c2' :: s2'')).
           { apply IH with (m := length (c1 :: c1' :: s1'') + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (Hsub : damerau_lev_distance (c1' :: s1'') (c2' :: s2'') <=
                          lev_distance (c1' :: s1'') (c2' :: s2'')).
           { apply IH with (m := length (c1' :: s1'') + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           (* Let D = del+1, I = ins+1, S = subst+subdist *)
           (* We need: min(min dD dI)(min dS T) <= min lD (min lI lS) *)
           (* where T = trans. *)
           remember (damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1) as dD eqn:HdD.
           remember (damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1) as dI eqn:HdI.
           remember (damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2) as dS eqn:HdS.
           remember (lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1) as lD eqn:HlD.
           remember (lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1) as lI eqn:HlI.
           remember (lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2) as lS eqn:HlS.
           assert (HdD_lD : dD <= lD) by lia.
           assert (HdI_lI : dI <= lI) by lia.
           assert (HdS_lS : dS <= lS) by lia.
           (* min4 = min(min dD dI)(min dS T) where T = trans term *)
           (* Show: min(min dD dI)(min dS T) <= dD, <= dI, <= dS *)
           assert (Hmin4_dD : min (min dD dI) (min dS
                   (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2'))
                 <= dD).
           { etransitivity; [apply Nat.le_min_l|]. apply Nat.le_min_l. }
           assert (Hmin4_dI : min (min dD dI) (min dS
                   (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2'))
                 <= dI).
           { etransitivity; [apply Nat.le_min_l|]. apply Nat.le_min_r. }
           assert (Hmin4_dS : min (min dD dI) (min dS
                   (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2'))
                 <= dS).
           { etransitivity; [apply Nat.le_min_r|]. apply Nat.le_min_l. }
           (* Now we have: min4 <= dD <= lD, min4 <= dI <= lI, min4 <= dS <= lS *)
           (* We need: min4 <= min lD (min lI lS) *)
           destruct (Nat.le_ge_cases lD lI) as [HlD_lI | HlD_lI];
           destruct (Nat.le_ge_cases lD lS) as [HlD_lS | HlD_lS];
           destruct (Nat.le_ge_cases lI lS) as [HlI_lS | HlI_lS]; lia.
Qed.

(** * Identity of Indiscernibles *)

Lemma damerau_lev_same : forall s,
  damerau_lev_distance s s = 0.
Proof.
  intro s.
  (* Use strong induction on length of s *)
  remember (length s) as n eqn:Hlen.
  revert s Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s Hlen.
  destruct s as [| c s'].
  - (* Base case: s = [] *)
    apply damerau_lev_empty_left.
  - destruct s' as [| c' s''].
    + (* Single element: s = [c] *)
      rewrite damerau_lev_single.
      rewrite char_eq_refl. reflexivity.
    + (* At least two elements: s = c :: c' :: s'' *)
      rewrite damerau_lev_cons2.
      unfold min4.
      (* The substitution branch: d(c'::s'', c'::s'') + subst_cost c c *)
      (* By IH, d(c'::s'', c'::s'') = 0 *)
      (* subst_cost c c = 0 *)
      assert (Hsubst : subst_cost c c = 0).
      { unfold subst_cost. rewrite char_eq_refl. reflexivity. }
      (* Need IH for the subterm c' :: s'' which has length n - 1 *)
      assert (Hsub : damerau_lev_distance (c' :: s'') (c' :: s'') = 0).
      { apply IH with (m := length (c' :: s'')).
        - simpl in Hlen. simpl. lia.
        - reflexivity. }
      (* Now show min evaluates to 0 *)
      rewrite Hsubst.
      rewrite Hsub.
      (* The structure is: min4 a b c d = min (min a b) (min c d)
         where c = 0 + 0 = 0 after rewriting.
         The goal is to show this equals 0. *)
      assert (Hz : 0 + 0 = 0) by lia.
      rewrite Hz.
      (* Goal: min (min a b) (min 0 d) = 0 *)
      (* We show min (min a b) (min 0 d) <= 0 *)
      apply Nat.le_antisymm.
      * (* min (min a b) (min 0 d) <= 0 *)
        etransitivity.
        -- apply Nat.le_min_r.
        -- (* min 0 d <= 0 *)
           apply Nat.le_min_l.
      * lia.
Qed.

(** * Transposition Example *)

Lemma damerau_transpose_distinct : forall a b : Char,
  a <> b ->
  damerau_lev_distance [a; b] [b; a] = 1.
Proof.
  intros a b Hneq.
  (* Compute by unfolding definitions *)
  unfold damerau_lev_distance.
  rewrite damerau_lev_pair_equation.
  (* Unfold all nested calls *)
  simpl.
  repeat rewrite damerau_lev_pair_equation.
  simpl.
  (* Now deal with char_eq calls *)
  unfold subst_cost, trans_cost_calc, char_eq.
  destruct (ascii_dec a a) as [_ | Ha]; [| exfalso; apply Ha; reflexivity].
  destruct (ascii_dec b b) as [_ | Hb]; [| exfalso; apply Hb; reflexivity].
  destruct (ascii_dec a b) as [Hab | _]; [contradiction |].
  destruct (ascii_dec b a) as [Hba | _]; [subst; contradiction |].
  simpl.
  unfold min4.
  reflexivity.
Qed.

(** Standard Levenshtein gives 2 for transposition case *)
Lemma lev_transpose_distinct : forall a b : Char,
  a <> b ->
  lev_distance [a; b] [b; a] = 2.
Proof.
  intros a b Hneq.
  (* Standard Levenshtein for [a; b] vs [b; a] *)
  unfold lev_distance.
  rewrite lev_distance_pair_equation.
  simpl.
  repeat rewrite lev_distance_pair_equation.
  simpl.
  unfold subst_cost, char_eq.
  destruct (ascii_dec a a) as [_ | Ha]; [| exfalso; apply Ha; reflexivity].
  destruct (ascii_dec b b) as [_ | Hb]; [| exfalso; apply Hb; reflexivity].
  destruct (ascii_dec a b) as [Hab | _]; [contradiction |].
  destruct (ascii_dec b a) as [Hba | _]; [subst; contradiction |].
  simpl.
  unfold min3.
  reflexivity.
Qed.

(** * Metric Properties *)

Lemma damerau_lev_sym : forall s1 s2,
  damerau_lev_distance s1 s2 = damerau_lev_distance s2 s1.
Proof.
  (* Use strong induction on |s1| + |s2| *)
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite damerau_lev_empty_left.
    rewrite damerau_lev_empty_right.
    reflexivity.
  - destruct s2 as [| c2 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite damerau_lev_empty_left.
      rewrite damerau_lev_empty_right.
      reflexivity.
    + (* s1 = c1::s1', s2 = c2::s2' *)
      destruct s1' as [| c1' s1''].
      * (* s1 = [c1] *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = [c1], s2 = [c2] *)
           rewrite damerau_lev_single.
           rewrite damerau_lev_single.
           rewrite char_eq_sym.
           reflexivity.
        -- (* s1 = [c1], s2 = c2::c2'::s2'' *)
           rewrite damerau_lev_single_multi.
           rewrite damerau_lev_multi_single.
           unfold min3.
           (* Use IH on subterms *)
           assert (H1 : damerau_lev_distance [] (c2 :: c2' :: s2'') =
                        damerau_lev_distance (c2 :: c2' :: s2'') []).
           { apply IH with (m := 0 + length (c2 :: c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H2 : damerau_lev_distance [c1] (c2' :: s2'') =
                        damerau_lev_distance (c2' :: s2'') [c1]).
           { apply IH with (m := 1 + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H3 : damerau_lev_distance [] (c2' :: s2'') =
                        damerau_lev_distance (c2' :: s2'') []).
           { apply IH with (m := 0 + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           rewrite H1, H2, H3.
           rewrite subst_cost_sym.
           (* Structure: min A (min B C) = min B (min A C) *)
           apply min3_swap12.
      * (* s1 = c1::c1'::s1'' *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = c1::c1'::s1'', s2 = [c2] *)
           rewrite damerau_lev_multi_single.
           rewrite damerau_lev_single_multi.
           unfold min3.
           assert (H1 : damerau_lev_distance (c1' :: s1'') [c2] =
                        damerau_lev_distance [c2] (c1' :: s1'')).
           { apply IH with (m := length (c1' :: s1'') + 1).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H2 : damerau_lev_distance (c1 :: c1' :: s1'') [] =
                        damerau_lev_distance [] (c1 :: c1' :: s1'')).
           { apply IH with (m := length (c1 :: c1' :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H3 : damerau_lev_distance (c1' :: s1'') [] =
                        damerau_lev_distance [] (c1' :: s1'')).
           { apply IH with (m := length (c1' :: s1'') + 0).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           rewrite H1, H2, H3.
           rewrite subst_cost_sym.
           (* Structure: min A (min B C) = min B (min A C) *)
           apply min3_swap12.
        -- (* s1 = c1::c1'::s1'', s2 = c2::c2'::s2'' - main case *)
           (* With standard DL definition (no double diagonal), the structure is symmetric *)
           rewrite damerau_lev_cons2.
           rewrite damerau_lev_cons2.
           (* Use IH on all subterms *)
           assert (H1 : damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') =
                        damerau_lev_distance (c2 :: c2' :: s2'') (c1' :: s1'')).
           { apply IH with (m := length (c1' :: s1'') + length (c2 :: c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H2 : damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') =
                        damerau_lev_distance (c2' :: s2'') (c1 :: c1' :: s1'')).
           { apply IH with (m := length (c1 :: c1' :: s1'') + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H3 : damerau_lev_distance (c1' :: s1'') (c2' :: s2'') =
                        damerau_lev_distance (c2' :: s2'') (c1' :: s1'')).
           { apply IH with (m := length (c1' :: s1'') + length (c2' :: s2'')).
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           assert (H4 : damerau_lev_distance s1'' s2'' =
                        damerau_lev_distance s2'' s1'').
           { apply IH with (m := length s1'' + length s2'').
             - simpl in Hlen. simpl. lia.
             - simpl. lia. }
           rewrite H1, H2, H3, H4.
           rewrite subst_cost_sym.
           rewrite trans_cost_calc_sym.
           (* Now: min4 (b+1) (a+1) (c+sc) (d+tc) = min4 (a+1) (b+1) (c+sc) (d+tc) *)
           (* This follows from min4_swap12 *)
           apply min4_swap12.
Qed.

Lemma damerau_lev_nonneg : forall s1 s2,
  damerau_lev_distance s1 s2 >= 0.
Proof.
  intros. lia.
Qed.

(** Length difference bound: edit distance is at least the length difference *)
Lemma damerau_lev_length_bound : forall s1 s2,
  damerau_lev_distance s1 s2 >= abs_diff (length s1) (length s2).
Proof.
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite damerau_lev_empty_left.
    rewrite abs_diff_0_l. lia.
  - destruct s2 as [| c2 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite damerau_lev_empty_right.
      rewrite abs_diff_0_r. lia.
    + destruct s1' as [| c1' s1''].
      * (* s1 = [c1] *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = [c1], s2 = [c2] *)
           rewrite damerau_lev_single.
           simpl. unfold abs_diff. simpl.
           destruct (char_eq c1 c2); lia.
        -- (* s1 = [c1], s2 = c2::c2'::s2'' *)
           rewrite damerau_lev_single_multi.
           (* min3 (del+1) (ins+1) (sub+cost) >= abs_diff 1 (S (S (length s2''))) *)
           (* abs_diff 1 (S (S (length s2''))) = S (length s2'') *)
           unfold min3.
           (* Rewrite empty distance *)
           rewrite damerau_lev_empty_left.
           (* Need: min (del+1) (min (ins+1) (sub+cost)) >= S (length s2'') *)
           simpl length.
           (* IH for ins: d([c1], c2'::s2'') >= abs_diff 1 (S (length s2'')) = length s2'' *)
           assert (Hins : damerau_lev_distance [c1] (c2' :: s2'') >= length s2'').
           { assert (Hm : 1 + S (length s2'') < n).
             { simpl in Hlen. simpl. lia. }
             assert (Hih := IH (1 + S (length s2'')) Hm [c1] (c2' :: s2'')).
             simpl in Hih.
             assert (Hlen' : 1 + S (length s2'') = 1 + S (length s2'')) by lia.
             specialize (Hih Hlen').
             unfold abs_diff in Hih. simpl in Hih.
             rewrite Nat.sub_0_r in Hih. exact Hih. }
           (* Show: min (S (S len) + 1) (min (d+1) (S len + cost)) >= S len *)
           (* Use Nat.min_glb to handle each branch *)
           assert (Hdel_bound : S (length s2'') <= S (S (length s2'')) + 1) by lia.
           assert (Hins_bound : S (length s2'') <= damerau_lev_distance [c1] (c2' :: s2'') + 1) by lia.
           assert (Hsub_bound : S (length s2'') <= S (length s2'') + subst_cost c1 c2).
           { unfold subst_cost. destruct (char_eq c1 c2); lia. }
           (* abs_diff 1 (S (S len)) = S len since 1 <= S (S len) *)
           assert (Habs : abs_diff 1 (S (S (length s2''))) = S (length s2'')).
           { unfold abs_diff. simpl. lia. }
           rewrite Habs.
           (* min a (min b c) >= k follows from a >= k, b >= k, c >= k *)
           apply Nat.min_glb.
           ++ exact Hdel_bound.
           ++ apply Nat.min_glb.
              ** exact Hins_bound.
              ** exact Hsub_bound.
      * (* s1 = c1::c1'::s1'' *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = c1::c1'::s1'', s2 = [c2] *)
           rewrite damerau_lev_multi_single.
           unfold min3.
           simpl length.
           (* abs_diff (S (S len)) 1 = S len since S (S len) >= 1 *)
           assert (Habs : abs_diff (S (S (length s1''))) 1 = S (length s1'')).
           { unfold abs_diff. simpl. lia. }
           rewrite Habs.
           (* IH for the delete case: d(c1'::s1'', [c2]) >= abs_diff (S (length s1'')) 1 *)
           assert (Hdel : damerau_lev_distance (c1' :: s1'') [c2] >= length s1'').
           { assert (Hm : S (length s1'') + 1 < n).
             { simpl in Hlen. simpl. lia. }
             assert (Hih := IH (S (length s1'') + 1) Hm (c1' :: s1'') [c2]).
             simpl in Hih.
             assert (Hlen' : S (length s1'') + 1 = S (length s1'') + 1) by lia.
             specialize (Hih Hlen').
             (* abs_diff (S len) 1 = len because S len >= 1 *)
             assert (Habs' : abs_diff (S (length s1'')) 1 = length s1'').
             { unfold abs_diff.
               destruct (length s1'') as [| len'] eqn:Hlen''.
               - simpl. reflexivity.
               - simpl. lia. }
             rewrite Habs' in Hih. exact Hih. }
           (* Base cases for empty right *)
           assert (Hemp1 : damerau_lev_distance (c1 :: c1' :: s1'') [] = S (S (length s1''))).
           { rewrite damerau_lev_empty_right. simpl. reflexivity. }
           assert (Hemp2 : damerau_lev_distance (c1' :: s1'') [] = S (length s1'')).
           { rewrite damerau_lev_empty_right. simpl. reflexivity. }
           (* Bounds for each branch *)
           assert (Hdel_bound : S (length s1'') <= damerau_lev_distance (c1' :: s1'') [c2] + 1) by lia.
           assert (Hins_bound : S (length s1'') <= damerau_lev_distance (c1 :: c1' :: s1'') [] + 1).
           { rewrite Hemp1. lia. }
           assert (Hsub_bound : S (length s1'') <= damerau_lev_distance (c1' :: s1'') [] + subst_cost c1 c2).
           { rewrite Hemp2. unfold subst_cost. destruct (char_eq c1 c2); lia. }
           apply Nat.min_glb.
           ++ exact Hdel_bound.
           ++ apply Nat.min_glb.
              ** exact Hins_bound.
              ** exact Hsub_bound.
        -- (* s1 = c1::c1'::s1'', s2 = c2::c2'::s2'' - main case *)
           rewrite damerau_lev_cons2.
           unfold min4.
           simpl length.
           (* The target: abs_diff (S (S len1)) (S (S len2)) = abs_diff len1 len2 *)
           set (len1 := length s1'').
           set (len2 := length s2'').
           assert (Htarget : abs_diff (S (S len1)) (S (S len2)) = abs_diff len1 len2).
           { unfold abs_diff.
             destruct (S (S len1) <=? S (S len2)) eqn:Hcmp1;
             destruct (len1 <=? len2) eqn:Hcmp2;
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_le in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_le in Hcmp2; lia). }
           rewrite Htarget.
           (* IH for all subterms - stated in terms of abs_diff len1 len2 *)
           assert (Hdel : damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 >=
                          abs_diff len1 len2).
           { assert (Hih : damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') >=
                           abs_diff (S len1) (S (S len2))).
             { apply IH with (m := S len1 + S (S len2)).
               - unfold len1, len2 in *. simpl in Hlen. simpl. lia.
               - unfold len1, len2. simpl. lia. }
             unfold abs_diff in *.
             destruct (S len1 <=? S (S len2)) eqn:Hcmp1;
             destruct (len1 <=? len2) eqn:Hcmp2;
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_le in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_le in Hcmp2; lia). }
           assert (Hins : damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1 >=
                          abs_diff len1 len2).
           { assert (Hih : damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') >=
                           abs_diff (S (S len1)) (S len2)).
             { apply IH with (m := S (S len1) + S len2).
               - unfold len1, len2 in *. simpl in Hlen. simpl. lia.
               - unfold len1, len2. simpl. lia. }
             unfold abs_diff in *.
             destruct (S (S len1) <=? S len2) eqn:Hcmp1;
             destruct (len1 <=? len2) eqn:Hcmp2;
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_le in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_le in Hcmp2; lia). }
           assert (Hsub : damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2 >=
                          abs_diff len1 len2).
           { assert (Hih : damerau_lev_distance (c1' :: s1'') (c2' :: s2'') >=
                           abs_diff (S len1) (S len2)).
             { apply IH with (m := S len1 + S len2).
               - unfold len1, len2 in *. simpl in Hlen. simpl. lia.
               - unfold len1, len2. simpl. lia. }
             unfold abs_diff in *. unfold subst_cost.
             destruct (char_eq c1 c2);
             destruct (S len1 <=? S len2) eqn:Hcmp1;
             destruct (len1 <=? len2) eqn:Hcmp2;
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_le in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_le in Hcmp1; apply Nat.leb_gt in Hcmp2; lia);
             try (apply Nat.leb_gt in Hcmp1; apply Nat.leb_le in Hcmp2; lia). }
           assert (Htrans : damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2' >=
                            abs_diff len1 len2).
           { assert (Hih : damerau_lev_distance s1'' s2'' >= abs_diff len1 len2).
             { apply IH with (m := len1 + len2).
               - unfold len1, len2 in *. simpl in Hlen. simpl. lia.
               - unfold len1, len2. simpl. lia. }
             unfold trans_cost_calc.
             destruct (char_eq c1 c2'); destruct (char_eq c1' c2); lia. }
           (* Use Nat.min_glb to handle min4 = min (min a b) (min c d) *)
           apply Nat.min_glb.
           ++ apply Nat.min_glb.
              ** exact Hdel.
              ** exact Hins.
           ++ apply Nat.min_glb.
              ** exact Hsub.
              ** exact Htrans.
Qed.

(** Auxiliary lemma for triangle inequality: removing the first character
    from the source string changes distance by at most 1.
    Proof: d(s', s2) <= 1 + d(c::s', s2) because we can insert c (cost 1)
    then transform c::s' to s2.

    This lemma uses strong induction on |s'| + |s2| because some recursive
    cases require applying the lemma to smaller target strings. *)
Lemma damerau_lev_remove_first_source : forall c s' s2,
  damerau_lev_distance s' s2 <= damerau_lev_distance (c :: s') s2 + 1.
Proof.
  intros c s' s2.
  (* Strong induction on |s'| + |s2| *)
  remember (length s' + length s2) as n eqn:Hlen.
  revert c s' s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros c s' s2 Hlen.
  destruct s2 as [| c2 s2'].
  - (* s2 = [] *)
    rewrite !damerau_lev_empty_right. simpl. lia.
  - destruct s' as [| c' s''].
    + (* s' = [] *)
      rewrite damerau_lev_empty_left.
      pose proof (damerau_lev_length_bound [c] (c2 :: s2')) as Hbd.
      simpl in Hbd. unfold abs_diff in Hbd. simpl in Hbd. simpl. lia.
    + (* s' = c'::s'' *)
      destruct s'' as [| c'' s'''].
      * (* s' = [c'] *)
        destruct s2' as [| c2' s2''].
        -- (* s' = [c'], s2 = [c2] *)
           rewrite !damerau_lev_single.
           rewrite damerau_lev_multi_single.
           unfold min3.
           rewrite damerau_lev_empty_right. simpl length.
           destruct (char_eq c' c2); simpl; lia.
        -- (* s' = [c'], s2 = c2::c2'::s2'' *)
           (* Goal: d([c'], c2::c2'::s2'') <= d([c;c'], c2::c2'::s2'') + 1 *)
           (* Expand RHS using damerau_lev_cons2 *)
           rewrite damerau_lev_cons2.
           (* RHS = min4 del_R ins_R sub_R trans_R + 1 *)

           (* Name the branches *)
           set (del_R := damerau_lev_distance [c'] (c2 :: c2' :: s2'') + 1).
           set (ins_R := damerau_lev_distance [c; c'] (c2' :: s2'') + 1).
           set (sub_R := damerau_lev_distance [c'] (c2' :: s2'') + subst_cost c c2).
           set (trans_R := damerau_lev_distance [] s2'' + trans_cost_calc c c' c2 c2').

           (* Establish bounds for each branch *)

           (* H1: LHS <= del_R + 1 (trivial since del_R = LHS + 1) *)
           assert (H1 : damerau_lev_distance [c'] (c2 :: c2' :: s2'') <= del_R + 1).
           { unfold del_R. lia. }

           (* H2: LHS <= ins_R + 1 *)
           assert (H2 : damerau_lev_distance [c'] (c2 :: c2' :: s2'') <= ins_R + 1).
           { rewrite damerau_lev_single_multi.
             (* LHS = min3 del_L ins_L sub_L <= ins_L *)
             etransitivity; [apply Nat.le_min_r; apply Nat.le_min_l |].
             unfold ins_R.
             (* Use IH: d([c'], c2'::s2'') <= d([c;c'], c2'::s2'') + 1 *)
             assert (HIH : damerau_lev_distance [c'] (c2' :: s2'') <=
                           damerau_lev_distance [c; c'] (c2' :: s2'') + 1).
             { apply IH with (m := length [c'] + length (c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - reflexivity. }
             lia. }

           (* H3: LHS <= sub_R + 1 *)
           assert (H3 : damerau_lev_distance [c'] (c2 :: c2' :: s2'') <= sub_R + 1).
           { rewrite damerau_lev_single_multi.
             etransitivity; [apply Nat.le_min_r; apply Nat.le_min_l |].
             unfold sub_R.
             (* d([c'], c2'::s2'') + 1 <= d([c'], c2'::s2'') + subst_cost + 1 *)
             unfold subst_cost. destruct (char_eq c c2); lia. }

           (* H4: LHS <= trans_R + 1 *)
           assert (H4 : damerau_lev_distance [c'] (c2 :: c2' :: s2'') <= trans_R + 1).
           { unfold trans_R, trans_cost_calc.
             destruct (andb (char_eq c c2') (char_eq c' c2)) eqn:Htrans.
             - (* trans_cost = 1, meaning c = c2' and c' = c2 *)
               (* Use sub_L bound: LHS <= d([],c2'::s2'') + subst_cost c' c2 *)
               rewrite damerau_lev_single_multi.
               etransitivity; [apply Nat.le_min_r; apply Nat.le_min_r |].
               (* Now goal: d([], c2'::s2'') + subst_cost c' c2 <= d([], s2'') + 1 + 1 *)
               (* Since char_eq c' c2 = true, subst_cost c' c2 = 0 *)
               apply andb_prop in Htrans. destruct Htrans as [_ Hc'c2].
               unfold subst_cost. rewrite Hc'c2. simpl.
               (* Goal: d([], c2'::s2'') + 0 <= d([], s2'') + 2 *)
               rewrite !damerau_lev_empty_left.
               simpl length.
               (* Goal: S (length s2'') + 0 <= length s2'' + 2 *)
               lia.
             - (* trans_cost = 100 *)
               (* Use del_L bound: LHS <= d([], c2::c2'::s2'') + 1 *)
               rewrite damerau_lev_single_multi.
               etransitivity; [apply Nat.le_min_l |].
               (* Goal: d([], c2::c2'::s2'') + 1 <= d([], s2'') + 100 + 1 *)
               rewrite !damerau_lev_empty_left.
               simpl length. lia. }

           (* Case analysis on min4 *)
           unfold min4.
           destruct (Nat.min_dec (min del_R ins_R) (min sub_R trans_R)) as [Hmin_outer | Hmin_outer].
           ++ rewrite Hmin_outer.
              destruct (Nat.min_dec del_R ins_R) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H1.
              ** rewrite Hmin_inner. exact H2.
           ++ rewrite Hmin_outer.
              destruct (Nat.min_dec sub_R trans_R) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H3.
              ** rewrite Hmin_inner. exact H4.
      * (* s' = c'::c''::s''' *)
        destruct s2' as [| c2' s2''].
        -- (* s2 = [c2] *)
           pose proof (damerau_lev_length_bound (c' :: c'' :: s''') [c2]) as HlenLHS.
           pose proof (damerau_lev_length_bound (c :: c' :: c'' :: s''') [c2]) as HlenRHS.
           simpl length in HlenLHS, HlenRHS.
           rewrite damerau_lev_multi_single.
           unfold min3.
           rewrite damerau_lev_empty_right.
           simpl length.
           etransitivity; [apply Nat.le_min_r; apply Nat.le_min_l |].
           unfold abs_diff in HlenRHS.
           assert (Hfalse : S (S (S (length s'''))) <=? 1 = false) by reflexivity.
           rewrite Hfalse in HlenRHS. lia.
        -- (* s2 = c2::c2'::s2'' - main case *)
           (* Goal: d(c'::c''::s''', c2::c2'::s2'') <= d(c::c'::c''::s''', c2::c2'::s2'') + 1 *)
           (* Both sides use damerau_lev_cons2 *)
           rewrite (damerau_lev_cons2 c c' (c'' :: s''') c2 c2' s2'').

           (* Name the RHS branches *)
           set (del_R := damerau_lev_distance (c' :: c'' :: s''') (c2 :: c2' :: s2'') + 1).
           set (ins_R := damerau_lev_distance (c :: c' :: c'' :: s''') (c2' :: s2'') + 1).
           set (sub_R := damerau_lev_distance (c' :: c'' :: s''') (c2' :: s2'') + subst_cost c c2).
           set (trans_R := damerau_lev_distance (c'' :: s''') s2'' + trans_cost_calc c c' c2 c2').

           (* Establish bounds *)

           (* H1: LHS <= del_R + 1 (trivial since del_R = LHS + 1) *)
           assert (H1 : damerau_lev_distance (c' :: c'' :: s''') (c2 :: c2' :: s2'') <= del_R + 1).
           { unfold del_R. lia. }

           (* H2: LHS <= ins_R + 1 *)
           assert (H2 : damerau_lev_distance (c' :: c'' :: s''') (c2 :: c2' :: s2'') <= ins_R + 1).
           { rewrite damerau_lev_cons2.
             (* LHS = min4 ... <= ins_L = d(c'::c''::s''', c2'::s2'') + 1 *)
             etransitivity; [unfold min4; apply Nat.le_min_l; apply Nat.le_min_r |].
             unfold ins_R.
             (* Use IH: d(c'::c''::s''', c2'::s2'') <= d(c::c'::c''::s''', c2'::s2'') + 1 *)
             assert (HIH : damerau_lev_distance (c' :: c'' :: s''') (c2' :: s2'') <=
                           damerau_lev_distance (c :: c' :: c'' :: s''') (c2' :: s2'') + 1).
             { apply IH with (m := length (c' :: c'' :: s''') + length (c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - reflexivity. }
             lia. }

           (* H3: LHS <= sub_R + 1 *)
           assert (H3 : damerau_lev_distance (c' :: c'' :: s''') (c2 :: c2' :: s2'') <= sub_R + 1).
           { rewrite damerau_lev_cons2.
             etransitivity; [unfold min4; apply Nat.le_min_l; apply Nat.le_min_r |].
             unfold sub_R.
             unfold subst_cost. destruct (char_eq c c2); lia. }

           (* H4: LHS <= trans_R + 1 *)
           assert (H4 : damerau_lev_distance (c' :: c'' :: s''') (c2 :: c2' :: s2'') <= trans_R + 1).
           { unfold trans_R, trans_cost_calc.
             destruct (andb (char_eq c c2') (char_eq c' c2)) eqn:Htrans.
             - (* trans_cost = 1, meaning c = c2' and c' = c2 *)
               (* Use sub_L bound: LHS <= d(c''::s''', c2'::s2'') + subst_cost c' c2 *)
               rewrite damerau_lev_cons2.
               (* Goal: min4 del_L ins_L sub_L trans_L <= trans_R + 1 *)
               (* min4 a b c d = min (min a b) (min c d), we want <= c (sub_L) *)
               etransitivity.
               + (* min4 ... <= sub_L *)
                 unfold min4.
                 etransitivity; [apply Nat.le_min_r |].
                 apply Nat.le_min_l.
               + (* sub_L <= trans_R + 1 *)
                 (* Since c' = c2, subst_cost c' c2 = 0 *)
                 apply andb_prop in Htrans. destruct Htrans as [_ Hc'c2].
                 unfold subst_cost. rewrite Hc'c2. simpl.
               (* Now: d(c''::s''', c2'::s2'') <= d(c''::s''', s2'') + 2 *)
               (* Show d(c''::s''', c2'::s2'') <= d(c''::s''', s2'') + 1 using insert branch *)
               (* Goal: d(c''::s''', c2'::s2'') + 0 <= d(c''::s''', s2'') + 2 *)
               (* Case analysis on s2'' to determine target structure *)
               destruct s2'' as [| c2'' s2'''].
               ++ (* s2'' = [] *)
                  (* d(c''::s''', [c2']) + 0 <= d(c''::s''', []) + 2 *)
                  rewrite damerau_lev_empty_right.
                  (* Goal: d(c''::s''', [c2']) + 0 <= length (c''::s''') + 2 *)
                  (* Using: d_damerau <= d_lev <= max(|s1|, |s2|) *)
                  pose proof (damerau_lev_le_standard (c'' :: s''') [c2']) as Hle_std.
                  pose proof (lev_distance_upper_bound (c'' :: s''') [c2']) as Hle_bound.
                  simpl length in Hle_bound.
                  (* Hle_bound : lev_distance <= max(1 + length s''', 1) *)
                  rewrite Nat.max_comm in Hle_bound.
                  destruct s'''; simpl in *; lia.
               ++ (* s2'' = c2''::s2''' *)
                  (* Goal: d(c''::s''', c2'::c2''::s2''') + 0 <= d(c''::s''', c2''::s2''') + 2 *)
                  (* By insert branch: d(s1, c::s2) = min(..., d(s1, s2)+1, ...) <= d(s1,s2)+1 *)
                  (* So d(c''::s''', c2'::c2''::s2''') <= d(c''::s''', c2''::s2''')+1 < d(...)+2 *)
                  destruct s''' as [| c''' s''''].
                  ** (* s''' = [], source = [c''] *)
                     rewrite damerau_lev_single_multi.
                     unfold min3.
                     (* Goal: min del_L (min ins_L sub_L) + 0 <= d([c''],c2''::s2''') + 2 *)
                     (* Step 1: min ... <= ins_L = d([c''],c2''::s2''')+1 *)
                     (* Step 2: ins_L + 0 <= d([c''],c2''::s2''') + 2 *)
                     transitivity (damerau_lev_distance [c''] (c2'' :: s2''') + 1 + 0).
                     --- (* min ... + 0 <= ins + 0 *)
                         apply Nat.add_le_mono_r.
                         etransitivity; [apply Nat.le_min_r |].
                         apply Nat.le_min_l.
                     --- (* ins + 0 <= d(...) + 2 *)
                         lia.
                  ** (* s''' = c'''::s'''', source has 2+ elements *)
                     rewrite damerau_lev_cons2.
                     unfold min4.
                     (* Goal: min (min del ins) (min sub trans) + 0 <= d(...) + 2 *)
                     (* Use insert branch: ins = d(c''::c'''::s'''', c2''::s2''') + 1 *)
                     transitivity (damerau_lev_distance (c'' :: c''' :: s'''') (c2'' :: s2''') + 1 + 0).
                     --- (* min ... + 0 <= ins + 0 *)
                         apply Nat.add_le_mono_r.
                         etransitivity; [apply Nat.le_min_l |].
                         apply Nat.le_min_r.
                     --- (* ins + 0 <= d(...) + 2 *)
                         lia.
             - (* trans_cost = 100 *)
               (* Goal: d(c'::c''::s''', c2::c2'::s2'') <= d(c''::s''', s2'') + 100 + 1 *)
               (* Strategy: d(c'::c''::s''', c2::c2'::s2'') <= d(c''::s''', s2'') + 4 (via insert path) *)
               (* Since 4 <= 101, we're done *)
               (* First, bound the LHS via del_L branch *)
               rewrite damerau_lev_cons2.
               etransitivity; [unfold min4; apply Nat.le_min_l; apply Nat.le_min_l |].
               (* Now: d(c''::s''', c2::c2'::s2'') + 1 <= d(c''::s''', s2'') + 101 *)
               (* Use upper bounds on damerau_lev_distance *)
               pose proof (damerau_lev_le_standard (c'' :: s''') (c2 :: c2' :: s2'')) as Hle_std.
               pose proof (lev_distance_upper_bound (c'' :: s''') (c2 :: c2' :: s2'')) as Hub_LHS.
               pose proof (damerau_lev_nonneg (c'' :: s''') s2'') as Hnn_RHS.
               simpl length in Hub_LHS.
               (* Hub_LHS: lev_distance <= max(1 + |s'''|, 2 + |s2''|) *)
               (* For trans_cost = 100, RHS has + 100 + 1 = + 101, which is plenty *)
               (* max(1 + |s'''|, 2 + |s2''|) + 1 <= 101 needs |s'''|, |s2''| bounded *)
               (* Actually, use the structural bound: d(s1, s2) <= |s1| + |s2| *)
               pose proof (damerau_lev_length_bound (c'' :: s''') (c2 :: c2' :: s2'')) as Hlen_LHS.
               simpl length in Hlen_LHS.
               (* Since d >= abs_diff, and we need d + 1 <= d' + 101 *)
               (* d' >= 0, so d + 1 <= max(1+|s'''|, 2+|s2''|) + 1 <= some bound *)
               (* Key: LHS = d(c''::s''', c2::c2'::s2'') <= d(c''::s''', s2'') + 2 by 2 insertions *)
               (* So LHS + 1 <= d(c''::s''', s2'') + 3 <= RHS since 3 <= 101 *)
               assert (Hadd_bound : damerau_lev_distance (c'' :: s''') (c2 :: c2' :: s2'') <=
                                    damerau_lev_distance (c'' :: s''') s2'' + 2).
               { destruct s2'' as [| c2''' s2''''].
                 + (* s2'' = [] *)
                   rewrite damerau_lev_empty_right.
                   pose proof (damerau_lev_le_standard (c'' :: s''') [c2; c2']) as Hstd.
                   pose proof (lev_distance_upper_bound (c'' :: s''') [c2; c2']) as Hub.
                   simpl length in Hub.
                   destruct s'''; simpl in *; lia.
                 + (* s2'' = c2'''::s2'''' *)
                   (* d(src, c2::c2'::c2'''::s2'''') <= d(src, c2'''::s2'''') + 2 *)
                   (* Use insert branches twice:
                      1. d(src, c2::c2'::c2'''::s2'''') <= d(src, c2'::c2'''::s2'''') + 1
                      2. d(src, c2'::c2'''::s2'''') <= d(src, c2'''::s2'''') + 1 *)
                   destruct s''' as [| c''' s'''''].
                   * (* s''' = [], src = [c''] *)
                     (* Step 1: d([c''], c2::c2'::c2'''::s2'''') <= d([c''], c2'::c2'''::s2'''') + 1 *)
                     rewrite damerau_lev_single_multi.
                     unfold min3.
                     transitivity (damerau_lev_distance [c''] (c2' :: c2''' :: s2'''') + 1).
                     { etransitivity; [apply Nat.le_min_r |]. apply Nat.le_min_l. }
                     (* Step 2: d([c''], c2'::c2'''::s2'''') + 1 <= d([c''], c2'''::s2'''') + 2 *)
                     rewrite damerau_lev_single_multi.
                     unfold min3.
                     transitivity (damerau_lev_distance [c''] (c2''' :: s2'''') + 1 + 1).
                     { apply Nat.add_le_mono_r. etransitivity; [apply Nat.le_min_r |]. apply Nat.le_min_l. }
                     { lia. }
                   * (* s''' = c'''::s''''', src = c''::c'''::s''''' *)
                     (* Step 1: d(src, c2::c2'::c2'''::s2'''') <= d(src, c2'::c2'''::s2'''') + 1 *)
                     rewrite damerau_lev_cons2.
                     unfold min4.
                     transitivity (damerau_lev_distance (c'' :: c''' :: s''''') (c2' :: c2''' :: s2'''') + 1).
                     { etransitivity; [apply Nat.le_min_l |]. apply Nat.le_min_r. }
                     (* Step 2: d(src, c2'::c2'''::s2'''') + 1 <= d(src, c2'''::s2'''') + 2 *)
                     rewrite damerau_lev_cons2.
                     unfold min4.
                     transitivity (damerau_lev_distance (c'' :: c''' :: s''''') (c2''' :: s2'''') + 1 + 1).
                     { apply Nat.add_le_mono_r. etransitivity; [apply Nat.le_min_l |]. apply Nat.le_min_r. }
                     { lia. } }
               lia. }

           (* Case analysis on min4 *)
           unfold min4.
           destruct (Nat.min_dec (min del_R ins_R) (min sub_R trans_R)) as [Hmin_outer | Hmin_outer].
           ++ rewrite Hmin_outer.
              destruct (Nat.min_dec del_R ins_R) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H1.
              ** rewrite Hmin_inner. exact H2.
           ++ rewrite Hmin_outer.
              destruct (Nat.min_dec sub_R trans_R) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H3.
              ** rewrite Hmin_inner. exact H4.
Qed.

(** Symmetric lemma: removing first character from target changes distance by at most 1 *)
Lemma damerau_lev_remove_first_target : forall s c s',
  damerau_lev_distance s s' <= damerau_lev_distance s (c :: s') + 1.
Proof.
  intros s c s'.
  (* By symmetry with remove_first_source, using damerau_lev_sym *)
  rewrite damerau_lev_sym.
  rewrite (damerau_lev_sym s (c :: s')).
  apply damerau_lev_remove_first_source.
Qed.

(** Adding a character to the source increases distance by at most 1.
    d(c::s, s') <= d(s, s') + 1 because we can delete c first. *)
Lemma damerau_lev_add_first_source : forall c s s',
  damerau_lev_distance (c :: s) s' <= damerau_lev_distance s s' + 1.
Proof.
  intros c s s'.
  destruct s' as [| c' s''].
  - (* s' = [] *)
    rewrite !damerau_lev_empty_right. simpl. lia.
  - (* s' = c'::s'' *)
    destruct s as [| c'' s'''].
    + (* s = [] *)
      rewrite damerau_lev_empty_left.
      (* d([c], c'::s'') <= d([], c'::s'') + 1 = |c'::s''| + 1 *)
      destruct s'' as [| c''' s'''].
      * (* s'' = [], so target is [c'] *)
        rewrite damerau_lev_single.
        simpl. destruct (char_eq c c'); lia.
      * (* s'' = c'''::s''', so target is c'::c'''::s''' *)
        rewrite damerau_lev_single_multi. unfold min3.
        (* d([c], c'::c'''::s''') = min3(del, ins, subst) *)
        (* where del = d([], c'::c'''::s''') + 1 = |c'::c'''::s'''| + 1 *)
        (* We want: LHS <= |c'::c'''::s'''| + 1 *)
        (* Take delete branch: d([], c'::c'''::s''') + 1 = |c'::c'''::s'''| + 1 *)
        etransitivity; [apply Nat.le_min_l |].
        rewrite damerau_lev_empty_left. simpl. lia.
    + (* s = c''::s''' - need to case on s'' for target *)
      destruct s'' as [| c'''' s''''].
      * (* s'' = [], target is [c'] *)
        (* d(c::c''::s''', [c']) <= d(c''::s''', [c']) + 1 *)
        rewrite damerau_lev_multi_single. unfold min3.
        (* delete branch = d(c''::s''', []) + 1 = |c''::s'''| + 1 *)
        (* We want: LHS <= d(c''::s''', [c']) + 1 *)
        (* Take delete branch: d(c''::s''', []) + 1 and show <= d(c''::s''', [c']) + 1 *)
        apply Nat.le_min_l.
      * (* s'' = c''''::s'''', target is c'::c''''::s'''' *)
        (* d(c::c''::s''', c'::c''''::s'''') - use damerau_lev_cons2 *)
        rewrite damerau_lev_cons2.
        (* min4 a b c d = min (min a b) (min c d) where a = delete_branch *)
        unfold min4.
        etransitivity; [apply Nat.le_min_l |].
        apply Nat.le_min_l.
Qed.

(** Adding a character to the target increases distance by at most 1.
    d(s, c::s') <= d(s, s') + 1 because we can insert c. *)
Lemma damerau_lev_add_first_target : forall s c s',
  damerau_lev_distance s (c :: s') <= damerau_lev_distance s s' + 1.
Proof.
  intros s c s'.
  (* By symmetry with add_first_source *)
  rewrite damerau_lev_sym.
  rewrite (damerau_lev_sym s s').
  apply damerau_lev_add_first_source.
Qed.

(** * Relationship: Standard Levenshtein bounded by 2× Damerau-Levenshtein

    Each operation in Damerau-Levenshtein corresponds to at most 2 operations
    in standard Levenshtein:
    - Insertion: 1 op in both
    - Deletion: 1 op in both
    - Substitution: 1 op in both
    - Transposition: 1 op in DL, but requires 2 ops (del+ins or 2 subs) in L

    Therefore: lev_distance s1 s2 <= 2 * damerau_lev_distance s1 s2

    Proof sketch for main case (both strings have 2+ characters):
    - For del, ins, sub options: IH gives lev <= 2 * damerau
    - For trans option when c1=c2' and c1'=c2:
      The lev sub path simulates transposition via double-substitution,
      giving lev(c1'::s1'', c1::s2'') + 1 <= lev(s1'', s2'') + 2
                                           <= 2 * damerau(s1'', s2'') + 2
                                           = 2 * d_trans
*)
Lemma lev_le_double_damerau : forall s1 s2,
  lev_distance s1 s2 <= 2 * damerau_lev_distance s1 s2.
Proof.
  (* Strong induction on |s1| + |s2|.
     Key insight: Each Damerau operation corresponds to at most 2 Lev operations.
     - del, ins, sub: 1 op in both (IH applies)
     - trans: 1 op in Damerau, but 2 subs in Lev (simulating transposition) *)
  intros s1 s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [] *)
    rewrite lev_distance_empty_left.
    rewrite damerau_lev_empty_left.
    lia.
  - destruct s2 as [| c2 s2'].
    + (* s1 = c1::s1', s2 = [] *)
      rewrite lev_distance_empty_right.
      rewrite damerau_lev_empty_right.
      lia.
    + (* s1 = c1::s1', s2 = c2::s2' *)
      destruct s1' as [| c1' s1''].
      * (* s1 = [c1], s2 = c2::s2' *)
        (* No transposition possible in Damerau when |s1| = 1 *)
        destruct s2' as [| c2' s2''].
        -- (* s2 = [c2] *)
           rewrite lev_distance_cons.
           rewrite damerau_lev_single.
           (* lev [c1] [c2] = min3 (lev [] [c2] + 1) (lev [c1] [] + 1) (lev [] [] + cost) *)
           (* damerau [c1] [c2] = if c1=c2 then 0 else 1 = subst_cost c1 c2 *)
           unfold min3.
           rewrite lev_distance_empty_left.
           rewrite lev_distance_empty_right.
           rewrite (lev_distance_empty_left []).
           simpl length.
           unfold subst_cost. destruct (char_eq c1 c2); simpl; lia.
        -- (* s2 = c2::c2'::s2'' *)
           rewrite lev_distance_cons.
           rewrite damerau_lev_single_multi.
           unfold min3.
           (* Each lev branch <= 2 * corresponding damerau branch *)
           assert (Hdel : lev_distance [] (c2 :: c2' :: s2'') + 1 <=
                          2 * (damerau_lev_distance [] (c2 :: c2' :: s2'') + 1)).
           { rewrite lev_distance_empty_left, damerau_lev_empty_left. lia. }
           assert (Hins : lev_distance [c1] (c2' :: s2'') + 1 <=
                          2 * (damerau_lev_distance [c1] (c2' :: s2'') + 1)).
           { assert (HIH : lev_distance [c1] (c2' :: s2'') <= 2 * damerau_lev_distance [c1] (c2' :: s2'')).
             { apply IH with (m := 1 + length (c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - simpl. lia. }
             lia. }
           assert (Hsub : lev_distance [] (c2' :: s2'') + subst_cost c1 c2 <=
                          2 * (damerau_lev_distance [] (c2' :: s2'') + subst_cost c1 c2)).
           { rewrite lev_distance_empty_left, damerau_lev_empty_left.
             unfold subst_cost. destruct (char_eq c1 c2); lia. }
           (* Now show min of lev <= 2 * min of damerau *)
           etransitivity.
           ++ apply Nat.min_le_compat.
              ** exact Hdel.
              ** apply Nat.min_le_compat; [exact Hins | exact Hsub].
           ++ rewrite <- Nat.mul_min_distr_l.
              rewrite <- Nat.mul_min_distr_l.
              lia.
      * (* s1 = c1::c1'::s1'', s2 = c2::s2' *)
        destruct s2' as [| c2' s2''].
        -- (* s1 = c1::c1'::s1'', s2 = [c2] - symmetric *)
           rewrite lev_distance_cons.
           rewrite damerau_lev_multi_single.
           unfold min3.
           assert (Hdel : lev_distance (c1' :: s1'') [c2] + 1 <=
                          2 * (damerau_lev_distance (c1' :: s1'') [c2] + 1)).
           { assert (HIH : lev_distance (c1' :: s1'') [c2] <=
                           2 * damerau_lev_distance (c1' :: s1'') [c2]).
             { apply IH with (m := length (c1' :: s1'') + 1).
               - simpl in Hlen. simpl. lia.
               - simpl. lia. }
             lia. }
           assert (Hins : lev_distance (c1 :: c1' :: s1'') [] + 1 <=
                          2 * (damerau_lev_distance (c1 :: c1' :: s1'') [] + 1)).
           { rewrite lev_distance_empty_right, damerau_lev_empty_right. lia. }
           assert (Hsub : lev_distance (c1' :: s1'') [] + subst_cost c1 c2 <=
                          2 * (damerau_lev_distance (c1' :: s1'') [] + subst_cost c1 c2)).
           { rewrite lev_distance_empty_right, damerau_lev_empty_right.
             unfold subst_cost. destruct (char_eq c1 c2); simpl; lia. }
           etransitivity.
           ++ apply Nat.min_le_compat.
              ** exact Hdel.
              ** apply Nat.min_le_compat; [exact Hins | exact Hsub].
           ++ rewrite <- Nat.mul_min_distr_l.
              rewrite <- Nat.mul_min_distr_l.
              lia.
        -- (* s1 = c1::c1'::s1'', s2 = c2::c2'::s2'' - main case with transposition *)
           rewrite lev_distance_cons.
           rewrite damerau_lev_cons2.
           unfold min3, min4.
           (* IH bounds for del, ins, sub branches *)
           assert (Hdel : lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 <=
                          2 * (damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1)).
           { assert (HIH : lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') <=
                           2 * damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'')).
             { apply IH with (m := length (c1' :: s1'') + length (c2 :: c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - simpl. lia. }
             lia. }
           assert (Hins : lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1 <=
                          2 * (damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1)).
           { assert (HIH : lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') <=
                           2 * damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'')).
             { apply IH with (m := length (c1 :: c1' :: s1'') + length (c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - simpl. lia. }
             lia. }
           assert (Hsub : lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2 <=
                          2 * (damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2)).
           { assert (HIH : lev_distance (c1' :: s1'') (c2' :: s2'') <=
                           2 * damerau_lev_distance (c1' :: s1'') (c2' :: s2'')).
             { apply IH with (m := length (c1' :: s1'') + length (c2' :: s2'')).
               - simpl in Hlen. simpl. lia.
               - simpl. lia. }
             unfold subst_cost. destruct (char_eq c1 c2); lia. }
           (* Key approach: Case on which branch achieves damerau's minimum.
              For del/ins/sub branches: l_X <= 2*d_X and lev <= l_X, damerau = d_X
              For trans branch: lev <= lev(s1'', s2'') + 2 <= 2*(damerau(s1'', s2'') + 1) *)

           (* First, bound lev against the transposition branch when trans applies *)
           assert (Htrans_bound :
             lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
             2 * (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2')).
           { unfold trans_cost_calc.
             destruct (char_eq c1 c2') eqn:Heq1; destruct (char_eq c1' c2) eqn:Heq2;
             simpl andb.
             - (* c1 = c2' and c1' = c2: transposition applies with cost 1 *)
               (* lev(c1::c1'::s1'', c2::c2'::s2'') <= lev(s1'', s2'') + 2 *)
               (* because we can use two substitutions to simulate transposition *)
               unfold char_eq in Heq1, Heq2.
               destruct (ascii_dec c1 c2'); [| discriminate].
               destruct (ascii_dec c1' c2); [| discriminate].
               subst c2' c2.
               (* Goal: lev(c1::c1'::s1'', c1'::c1::s2'') <= 2*(damerau(s1'',s2'')+1) *)
               (* lev <= lev(c1'::s1'', c1::s2'') + 1 (via sub branch, cost >= 0) *)
               assert (Hstep1 : lev_distance (c1 :: c1' :: s1'') (c1' :: c1 :: s2'') <=
                                lev_distance (c1' :: s1'') (c1 :: s2'') + 1).
               { rewrite lev_distance_cons. unfold min3.
                 etransitivity; [apply Nat.le_min_r; apply Nat.le_min_r |].
                 unfold subst_cost. destruct (char_eq c1 c1'); lia. }
               (* lev(c1'::s1'', c1::s2'') <= lev(s1'', s2'') + 1 *)
               assert (Hstep2 : lev_distance (c1' :: s1'') (c1 :: s2'') <=
                                lev_distance s1'' s2'' + 1).
               { rewrite lev_distance_cons. unfold min3.
                 etransitivity; [apply Nat.le_min_r; apply Nat.le_min_r |].
                 unfold subst_cost. destruct (char_eq c1' c1); lia. }
               (* lev(s1'', s2'') <= 2 * damerau(s1'', s2'') by IH *)
               assert (HIH_trans : lev_distance s1'' s2'' <= 2 * damerau_lev_distance s1'' s2'').
               { apply IH with (m := length s1'' + length s2'').
                 - simpl in Hlen. simpl. lia.
                 - reflexivity. }
               lia.
            - (* c1 = c2' but c1' ≠ c2: trans_cost = 100 *)
              (* Strategy: lev(full) <= lev(s1'', s2'') + 4 <= 2*damerau(s1'', s2'') + 4 *)
              (* And 2*damerau(s1'', s2'') + 4 <= 2*(damerau(s1'', s2'') + 100) since 4 <= 200 *)
              assert (HIH'' : lev_distance s1'' s2'' <= 2 * damerau_lev_distance s1'' s2'').
              { apply IH with (m := length s1'' + length s2'').
                - simpl in Hlen. simpl. lia.
                - reflexivity. }
              (* lev(c1::c1'::s1'', c2::c2'::s2'') <= lev(c1'::s1'', c2'::s2'') + 2 *)
              assert (Hstep1 : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                               lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 <=
                                    lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
                { rewrite lev_distance_cons. unfold min3.
                  (* min3(a,b,c) <= b, so min3(a,b,c)+1 <= b+1 = lev+1+1 = lev+2 *)
                  assert (Hmin_le_mid : min (lev_distance s1'' (c2 :: c2' :: s2'') + 1)
                            (min (lev_distance (c1' :: s1'') (c2' :: s2'') + 1)
                                 (lev_distance s1'' (c2' :: s2'') + subst_cost c1' c2))
                          <= lev_distance (c1' :: s1'') (c2' :: s2'') + 1).
                  { etransitivity; [apply Nat.le_min_r |]. apply Nat.le_min_l. }
                  lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              (* lev(c1'::s1'', c2'::s2'') <= lev(s1'', s2'') + 2 *)
              assert (Hstep2 : lev_distance (c1' :: s1'') (c2' :: s2'') <=
                               lev_distance s1'' s2'' + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance s1'' (c2' :: s2'') + 1 <=
                                    lev_distance s1'' s2'' + 2).
                { (* Adding a char to s2 increases distance by at most 1 *)
                  (* lev(s1'', c2'::s2'') <= lev(s1'', s2'') + 1 by inserting c2' *)
                  destruct s1'' as [| ch s1'''].
                  - (* s1'' = [] *)
                    rewrite lev_distance_empty_left. simpl.
                    rewrite lev_distance_empty_left. lia.
                  - (* s1'' = ch :: s1''' *)
                    rewrite lev_distance_cons. unfold min3.
                    (* The insert branch gives: lev(ch::s1''', s2') + 1 *)
                    assert (Hins_branch : min (lev_distance s1''' (c2' :: s2'') + 1)
                              (min (lev_distance (ch :: s1''') s2'' + 1)
                                   (lev_distance s1''' s2'' + subst_cost ch c2'))
                            <= lev_distance (ch :: s1''') s2'' + 1).
                    { etransitivity; [apply Nat.le_min_r|]. apply Nat.le_min_l. }
                    lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              (* Combine: lev(full) <= lev(s1'', s2'') + 4 <= 2*d(s1'', s2'') + 4 *)
              assert (Hcombined : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                                  2 * damerau_lev_distance s1'' s2'' + 4).
              { lia. }
              (* trans_cost = 100 in this case *)
              simpl. lia.
            - (* c1 ≠ c2': trans_cost = 100 *)
              (* Same strategy as above *)
              assert (HIH'' : lev_distance s1'' s2'' <= 2 * damerau_lev_distance s1'' s2'').
              { apply IH with (m := length s1'' + length s2'').
                - simpl in Hlen. simpl. lia.
                - reflexivity. }
              assert (Hstep1 : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                               lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 <=
                                    lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
                { rewrite lev_distance_cons. unfold min3.
                  assert (Hmin_le_mid : min (lev_distance s1'' (c2 :: c2' :: s2'') + 1)
                            (min (lev_distance (c1' :: s1'') (c2' :: s2'') + 1)
                                 (lev_distance s1'' (c2' :: s2'') + subst_cost c1' c2))
                          <= lev_distance (c1' :: s1'') (c2' :: s2'') + 1).
                  { etransitivity; [apply Nat.le_min_r |]. apply Nat.le_min_l. }
                  lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              assert (Hstep2 : lev_distance (c1' :: s1'') (c2' :: s2'') <=
                               lev_distance s1'' s2'' + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance s1'' (c2' :: s2'') + 1 <=
                                    lev_distance s1'' s2'' + 2).
                { destruct s1'' as [| ch s1'''].
                  - rewrite lev_distance_empty_left. simpl.
                    rewrite lev_distance_empty_left. lia.
                  - rewrite lev_distance_cons. unfold min3.
                    assert (Hins_branch : min (lev_distance s1''' (c2' :: s2'') + 1)
                              (min (lev_distance (ch :: s1''') s2'' + 1)
                                   (lev_distance s1''' s2'' + subst_cost ch c2'))
                            <= lev_distance (ch :: s1''') s2'' + 1).
                    { etransitivity; [apply Nat.le_min_r|]. apply Nat.le_min_l. }
                    lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              simpl. lia.
            - (* Neither: trans_cost = 100 *)
              (* Same strategy *)
              assert (HIH'' : lev_distance s1'' s2'' <= 2 * damerau_lev_distance s1'' s2'').
              { apply IH with (m := length s1'' + length s2'').
                - simpl in Hlen. simpl. lia.
                - reflexivity. }
              assert (Hstep1 : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                               lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 <=
                                    lev_distance (c1' :: s1'') (c2' :: s2'') + 2).
                { rewrite lev_distance_cons. unfold min3.
                  assert (Hmin_le_mid : min (lev_distance s1'' (c2 :: c2' :: s2'') + 1)
                            (min (lev_distance (c1' :: s1'') (c2' :: s2'') + 1)
                                 (lev_distance s1'' (c2' :: s2'') + subst_cost c1' c2))
                          <= lev_distance (c1' :: s1'') (c2' :: s2'') + 1).
                  { etransitivity; [apply Nat.le_min_r |]. apply Nat.le_min_l. }
                  lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              assert (Hstep2 : lev_distance (c1' :: s1'') (c2' :: s2'') <=
                               lev_distance s1'' s2'' + 2).
              { rewrite lev_distance_cons. unfold min3.
                assert (Hdel_step : lev_distance s1'' (c2' :: s2'') + 1 <=
                                    lev_distance s1'' s2'' + 2).
                { destruct s1'' as [| ch s1'''].
                  - rewrite lev_distance_empty_left. simpl.
                    rewrite lev_distance_empty_left. lia.
                  - rewrite lev_distance_cons. unfold min3.
                    assert (Hins_branch : min (lev_distance s1''' (c2' :: s2'') + 1)
                              (min (lev_distance (ch :: s1''') s2'' + 1)
                                   (lev_distance s1''' s2'' + subst_cost ch c2'))
                            <= lev_distance (ch :: s1''') s2'' + 1).
                    { etransitivity; [apply Nat.le_min_r|]. apply Nat.le_min_l. }
                    lia. }
                etransitivity; [apply Nat.le_min_l |]. exact Hdel_step. }
              simpl. lia. }

           (* Now the main bound: lev <= 2 * damerau *)
           (* lev = min3(l_del, l_ins, l_sub), damerau = min4(d_del, d_ins, d_sub, d_trans) *)
           (* We have: l_del <= 2*d_del, l_ins <= 2*d_ins, l_sub <= 2*d_sub, and Htrans_bound *)
           (* Since damerau = min4(...) <= each of d_del, d_ins, d_sub, d_trans *)
           (* Case: damerau achieved by d_del => 2*damerau = 2*d_del >= l_del >= lev ✓ *)
           (* Case: damerau achieved by d_ins => 2*damerau = 2*d_ins >= l_ins >= lev ✓ *)
           (* Case: damerau achieved by d_sub => 2*damerau = 2*d_sub >= l_sub >= lev ✓ *)
           (* Case: damerau achieved by d_trans => use Htrans_bound *)

           (* Clean approach: show lev <= 2 * each branch of damerau, then use min4 property *)
           assert (Hlev_le_del : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                                 2 * (damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1)).
           { etransitivity; [| exact Hdel].
             rewrite lev_distance_cons. unfold min3.
             etransitivity; [apply Nat.le_min_l |]. lia. }
           assert (Hlev_le_ins : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                                 2 * (damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1)).
           { etransitivity; [| exact Hins].
             rewrite lev_distance_cons. unfold min3.
             etransitivity; [apply Nat.le_min_r; apply Nat.le_min_l |]. lia. }
           assert (Hlev_le_sub : lev_distance (c1 :: c1' :: s1'') (c2 :: c2' :: s2'') <=
                                 2 * (damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2)).
           { etransitivity; [| exact Hsub].
             rewrite lev_distance_cons. unfold min3.
             etransitivity; [apply Nat.le_min_r; apply Nat.le_min_r |]. lia. }

           (* Now: lev <= 2 * each of the 4 branches of min4 *)
           (* Goal is: min(l_del, min(l_ins, l_sub)) <= 2 * min(min(d_del, d_ins), min(d_sub, d_trans)) *)

           (* Prove LHS <= 2*d_X for each branch *)
           set (LHS := min (lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1)
                       (min (lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1)
                            (lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2))).

           assert (H1 : LHS <= 2 * (damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1)).
           { unfold LHS. etransitivity; [apply Nat.le_min_l |]. exact Hdel. }

           assert (H2 : LHS <= 2 * (damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1)).
           { unfold LHS.
             etransitivity; [apply Nat.le_min_r |].
             etransitivity; [apply Nat.le_min_l |]. exact Hins. }

           assert (H3 : LHS <= 2 * (damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2)).
           { unfold LHS.
             etransitivity; [apply Nat.le_min_r |].
             etransitivity; [apply Nat.le_min_r |]. exact Hsub. }

           assert (H4 : LHS <= 2 * (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2')).
           { (* LHS = min(...) equals lev_distance(c1::c1'::s1'', c2::c2'::s2'') by lev_distance_cons *)
             (* Rewrite Htrans_bound to use the same form as LHS *)
             rewrite lev_distance_cons in Htrans_bound. unfold min3 in Htrans_bound.
             exact Htrans_bound. }

           (* Goal: LHS <= 2 * min(min(d_del, d_ins), min(d_sub, d_trans)) *)
           (* Use: 2 * min(a,b) >= 2*a if min(a,b)=a, and >= 2*b if min(a,b)=b *)
           (* Since LHS <= 2*d_X for all X, LHS <= 2*min(...) regardless of which achieves min *)

           (* Simplify: Let d1 = min(d_del, d_ins), d2 = min(d_sub, d_trans) *)
           set (d_del := damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1).
           set (d_ins := damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1).
           set (d_sub := damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2).
           set (d_trans := damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2').

           (* Case analysis on outer min *)
           destruct (Nat.min_dec (min d_del d_ins) (min d_sub d_trans)) as [Hmin_outer | Hmin_outer].
           ++ (* min achieves left: min(...) = min(d_del, d_ins) *)
              rewrite Hmin_outer.
              (* Case analysis on inner min *)
              destruct (Nat.min_dec d_del d_ins) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H1.
              ** rewrite Hmin_inner. exact H2.
           ++ (* min achieves right: min(...) = min(d_sub, d_trans) *)
              rewrite Hmin_outer.
              destruct (Nat.min_dec d_sub d_trans) as [Hmin_inner | Hmin_inner].
              ** rewrite Hmin_inner. exact H3.
              ** rewrite Hmin_inner. exact H4.
Qed.
