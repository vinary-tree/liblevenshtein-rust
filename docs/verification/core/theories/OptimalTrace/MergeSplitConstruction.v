(** * Optimal Merge-Split Trace Construction

    This module defines the optimal MS trace construction via DP backtracking.
    The optimal trace achieves exactly the merge_split_distance cost.

    Part of: Liblevenshtein.Core

    Design: Follows the pattern of OptimalTrace/Construction.v but with
    6-way branching for merge-split operations.
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia Wf_nat.
From Stdlib Require Import Program.Wf.
From Stdlib Require Import Recdef.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.
From Liblevenshtein.Core Require Import Trace.MergeSplitTrace.

(** * Base case lemmas for merge_split_pair *)

(** merge_split_pair ([], B) = length B *)
Lemma merge_split_pair_nil_l : forall B,
  merge_split_pair ([], B) = length B.
Proof.
  intros B.
  rewrite merge_split_pair_equation.
  reflexivity.
Qed.

(** merge_split_pair (A, []) = length A *)
Lemma merge_split_pair_nil_r : forall A,
  merge_split_pair (A, []) = length A.
Proof.
  intros A.
  destruct A as [|c rest].
  - rewrite merge_split_pair_equation. reflexivity.
  - rewrite merge_split_pair_equation.
    destruct rest; reflexivity.
Qed.

(** * Optimal MS Trace Definition *)

(** Measure for well-founded recursion *)
Definition ms_optimal_trace_measure (p : list Char * list Char) : nat :=
  length (fst p) + length (snd p).

(** Helper to determine which branch of min6 wins *)
Definition min6_branch (a b c d e f : nat) : nat :=
  if (a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f) then 1
  else if (b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f) then 2
  else if (c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f) then 3
  else if (d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f) then 4
  else if (e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f) then 5
  else 6.

(** min6_branch always returns at least 1 *)
Lemma min6_branch_ge_1 : forall a b c d e f, 1 <= min6_branch a b c d e f.
Proof.
  intros. unfold min6_branch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [lia|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [lia|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [lia|].
  destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [lia|].
  destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; lia.
Qed.

(** min6_branch never returns 0 *)
Lemma min6_branch_not_0 : forall a b c d e f, min6_branch a b c d e f <> 0.
Proof.
  intros. pose proof (min6_branch_ge_1 a b c d e f). lia.
Qed.

(** min6_branch = 1 implies min6 equals first argument *)
Lemma min6_branch_eq_1_implies : forall a b c d e f,
  min6_branch a b c d e f = 1 ->
  min6 a b c d e f = a.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1.
  - (* E1 = true: a <= all others *)
    apply andb_prop in E1 as [E1' Hf].
    apply andb_prop in E1' as [E1'' He].
    apply andb_prop in E1'' as [E1''' Hd].
    apply andb_prop in E1''' as [Hab Hc].
    apply Nat.leb_le in Hab. apply Nat.leb_le in Hc. apply Nat.leb_le in Hd.
    apply Nat.leb_le in He. apply Nat.leb_le in Hf.
    unfold min6, min5. lia.
  - (* E1 = false: contradicts Hbranch = 1 *)
    destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [discriminate|].
    destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [discriminate|].
    destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [discriminate|].
    destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; discriminate.
Qed.

(** min6_branch = 2 implies min6 equals second argument *)
Lemma min6_branch_eq_2_implies : forall a b c d e f,
  min6_branch a b c d e f = 2 ->
  min6 a b c d e f = b.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2.
  - apply andb_prop in E2 as [E2' Hf].
    apply andb_prop in E2' as [E2'' He].
    apply andb_prop in E2'' as [E2''' Hd].
    apply andb_prop in E2''' as [Hba Hc].
    apply Nat.leb_le in Hba. apply Nat.leb_le in Hc. apply Nat.leb_le in Hd.
    apply Nat.leb_le in He. apply Nat.leb_le in Hf.
    unfold min6, min5. lia.
  - destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [discriminate|].
    destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [discriminate|].
    destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; discriminate.
Qed.

(** min6_branch = 3 implies min6 equals third argument *)
Lemma min6_branch_eq_3_implies : forall a b c d e f,
  min6_branch a b c d e f = 3 ->
  min6 a b c d e f = c.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3.
  - apply andb_prop in E3 as [E3' Hf].
    apply andb_prop in E3' as [E3'' He].
    apply andb_prop in E3'' as [E3''' Hd].
    apply andb_prop in E3''' as [Hca Hb].
    apply Nat.leb_le in Hca. apply Nat.leb_le in Hb. apply Nat.leb_le in Hd.
    apply Nat.leb_le in He. apply Nat.leb_le in Hf.
    unfold min6, min5. lia.
  - destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [discriminate|].
    destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; discriminate.
Qed.

(** min6_branch = 4 implies min6 equals fourth argument *)
Lemma min6_branch_eq_4_implies : forall a b c d e f,
  min6_branch a b c d e f = 4 ->
  min6 a b c d e f = d.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [discriminate|].
  destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4.
  - apply andb_prop in E4 as [E4' Hf].
    apply andb_prop in E4' as [E4'' He].
    apply andb_prop in E4'' as [E4''' Hc].
    apply andb_prop in E4''' as [Hda Hb].
    apply Nat.leb_le in Hda. apply Nat.leb_le in Hb. apply Nat.leb_le in Hc.
    apply Nat.leb_le in He. apply Nat.leb_le in Hf.
    unfold min6, min5. lia.
  - destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; discriminate.
Qed.

(** min6_branch = 5 implies min6 equals fifth argument *)
Lemma min6_branch_eq_5_implies : forall a b c d e f,
  min6_branch a b c d e f = 5 ->
  min6 a b c d e f = e.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [discriminate|].
  destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [discriminate|].
  destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5.
  - apply andb_prop in E5 as [E5' Hf].
    apply andb_prop in E5' as [E5'' Hd].
    apply andb_prop in E5'' as [E5''' Hc].
    apply andb_prop in E5''' as [Hea Hb].
    apply Nat.leb_le in Hea. apply Nat.leb_le in Hb. apply Nat.leb_le in Hc.
    apply Nat.leb_le in Hd. apply Nat.leb_le in Hf.
    unfold min6, min5. lia.
  - discriminate.
Qed.

(** min6_branch returns at most 6 *)
Lemma min6_branch_le_6 : forall a b c d e f, min6_branch a b c d e f <= 6.
Proof.
  intros. unfold min6_branch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool; [lia|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool; [lia|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool; [lia|].
  destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool; [lia|].
  destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool; lia.
Qed.

(** min6_branch >= 6 implies = 6 (for default branch handling) *)
Lemma min6_branch_ge_6_eq_6 : forall a b c d e f n,
  min6_branch a b c d e f = S (S (S (S (S (S n))))) ->
  min6_branch a b c d e f = 6.
Proof.
  intros a b c d e f n H.
  pose proof (min6_branch_le_6 a b c d e f) as Hle.
  lia.
Qed.

(** min6_branch = 6 implies min6 equals sixth argument *)
Lemma min6_branch_eq_6_implies : forall a b c d e f,
  min6_branch a b c d e f = 6 ->
  min6 a b c d e f = f.
Proof.
  intros a b c d e f Hbranch.
  unfold min6_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d) && (a <=? e) && (a <=? f))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d) && (b <=? e) && (b <=? f))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d) && (c <=? e) && (c <=? f))%bool eqn:E3; [discriminate|].
  destruct ((d <=? a) && (d <=? b) && (d <=? c) && (d <=? e) && (d <=? f))%bool eqn:E4; [discriminate|].
  destruct ((e <=? a) && (e <=? b) && (e <=? c) && (e <=? d) && (e <=? f))%bool eqn:E5; [discriminate|].
  (* Now we know none of a,b,c,d,e are the minimum, so f must be *)
  (* From E1..E5 being false, derive that f < each of a,b,c,d,e *)
  unfold min6, min5.
  (* Since no branch condition is true, we know for each variable there exists
     some other variable strictly smaller. By pigeon-hole, f must be among
     the smallest. *)
  assert (Ha: ~ (a <= b /\ a <= c /\ a <= d /\ a <= e /\ a <= f)).
  { intros [H1 [H2 [H3 [H4 H5]]]].
    destruct (Nat.leb_spec0 a b); destruct (Nat.leb_spec0 a c);
    destruct (Nat.leb_spec0 a d); destruct (Nat.leb_spec0 a e);
    destruct (Nat.leb_spec0 a f); simpl in E1; try discriminate; lia. }
  assert (Hb: ~ (b <= a /\ b <= c /\ b <= d /\ b <= e /\ b <= f)).
  { intros [H1 [H2 [H3 [H4 H5]]]].
    destruct (Nat.leb_spec0 b a); destruct (Nat.leb_spec0 b c);
    destruct (Nat.leb_spec0 b d); destruct (Nat.leb_spec0 b e);
    destruct (Nat.leb_spec0 b f); simpl in E2; try discriminate; lia. }
  assert (Hc: ~ (c <= a /\ c <= b /\ c <= d /\ c <= e /\ c <= f)).
  { intros [H1 [H2 [H3 [H4 H5]]]].
    destruct (Nat.leb_spec0 c a); destruct (Nat.leb_spec0 c b);
    destruct (Nat.leb_spec0 c d); destruct (Nat.leb_spec0 c e);
    destruct (Nat.leb_spec0 c f); simpl in E3; try discriminate; lia. }
  assert (Hd: ~ (d <= a /\ d <= b /\ d <= c /\ d <= e /\ d <= f)).
  { intros [H1 [H2 [H3 [H4 H5]]]].
    destruct (Nat.leb_spec0 d a); destruct (Nat.leb_spec0 d b);
    destruct (Nat.leb_spec0 d c); destruct (Nat.leb_spec0 d e);
    destruct (Nat.leb_spec0 d f); simpl in E4; try discriminate; lia. }
  assert (He: ~ (e <= a /\ e <= b /\ e <= c /\ e <= d /\ e <= f)).
  { intros [H1 [H2 [H3 [H4 H5]]]].
    destruct (Nat.leb_spec0 e a); destruct (Nat.leb_spec0 e b);
    destruct (Nat.leb_spec0 e c); destruct (Nat.leb_spec0 e d);
    destruct (Nat.leb_spec0 e f); simpl in E5; try discriminate; lia. }
  lia.
Qed.

(** Helper to determine which branch of min(min3, x) wins *)
Definition min4_branch (a b c d : nat) : nat :=
  if (a <=? b) && (a <=? c) && (a <=? d) then 1
  else if (b <=? a) && (b <=? c) && (b <=? d) then 2
  else if (c <=? a) && (c <=? b) && (c <=? d) then 3
  else 4.

(** min4_branch always returns at least 1 *)
Lemma min4_branch_ge_1 : forall a b c d, 1 <= min4_branch a b c d.
Proof.
  intros. unfold min4_branch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool eqn:E1; [lia|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool eqn:E2; [lia|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool eqn:E3; lia.
Qed.

(** min4_branch never returns 0 *)
Lemma min4_branch_not_0 : forall a b c d, min4_branch a b c d <> 0.
Proof.
  intros. pose proof (min4_branch_ge_1 a b c d). lia.
Qed.

(** Helper: When first branch wins, min(min3 a b c) d = a *)
Lemma min4_eq_first : forall a b c d,
  (a <=? b) && (a <=? c) && (a <=? d) = true ->
  min (min3 a b c) d = a.
Proof.
  intros a b c d H.
  apply andb_prop in H as [H1 H2].
  apply andb_prop in H1 as [H3 H4].
  apply Nat.leb_le in H2.
  apply Nat.leb_le in H3.
  apply Nat.leb_le in H4.
  unfold min3. lia.
Qed.

(** Helper: When second branch wins, min(min3 a b c) d = b *)
Lemma min4_eq_second : forall a b c d,
  (a <=? b) && (a <=? c) && (a <=? d) = false ->
  (b <=? a) && (b <=? c) && (b <=? d) = true ->
  min (min3 a b c) d = b.
Proof.
  intros a b c d Hf H.
  apply andb_prop in H as [H1 H2].
  apply andb_prop in H1 as [H3 H4].
  apply Nat.leb_le in H2.
  apply Nat.leb_le in H3.
  apply Nat.leb_le in H4.
  unfold min3. lia.
Qed.

(** Helper: When third branch wins, min(min3 a b c) d = c *)
Lemma min4_eq_third : forall a b c d,
  (a <=? b) && (a <=? c) && (a <=? d) = false ->
  (b <=? a) && (b <=? c) && (b <=? d) = false ->
  (c <=? a) && (c <=? b) && (c <=? d) = true ->
  min (min3 a b c) d = c.
Proof.
  intros a b c d Hf1 Hf2 H.
  apply andb_prop in H as [H1 H2].
  apply andb_prop in H1 as [H3 H4].
  apply Nat.leb_le in H2.
  apply Nat.leb_le in H3.
  apply Nat.leb_le in H4.
  unfold min3. lia.
Qed.

(** Helper: When fourth branch wins (default), min(min3 a b c) d = d *)
Lemma min4_eq_fourth : forall a b c d,
  (a <=? b) && (a <=? c) && (a <=? d) = false ->
  (b <=? a) && (b <=? c) && (b <=? d) = false ->
  (c <=? a) && (c <=? b) && (c <=? d) = false ->
  min (min3 a b c) d = d.
Proof.
  intros a b c d Hf1 Hf2 Hf3.
  unfold min3.
  (* The hypotheses tell us: a is not the minimum, b is not the minimum,
     c is not the minimum. By process of elimination, d must be. *)
  (* First, we derive that d < each of a, b, c *)
  assert (Ha: ~ (a <= b /\ a <= c /\ a <= d)).
  { intros [H1 [H2 H3]].
    destruct (Nat.leb_spec0 a b); destruct (Nat.leb_spec0 a c); destruct (Nat.leb_spec0 a d);
    simpl in Hf1; try discriminate; lia. }
  assert (Hb: ~ (b <= a /\ b <= c /\ b <= d)).
  { intros [H1 [H2 H3]].
    destruct (Nat.leb_spec0 b a); destruct (Nat.leb_spec0 b c); destruct (Nat.leb_spec0 b d);
    simpl in Hf2; try discriminate; lia. }
  assert (Hc: ~ (c <= a /\ c <= b /\ c <= d)).
  { intros [H1 [H2 H3]].
    destruct (Nat.leb_spec0 c a); destruct (Nat.leb_spec0 c b); destruct (Nat.leb_spec0 c d);
    simpl in Hf3; try discriminate; lia. }
  lia.
Qed.

(** min4_branch = 1 implies min equals first argument *)
Lemma min4_branch_eq_1_implies : forall a b c d,
  min4_branch a b c d = 1 ->
  min (min3 a b c) d = a.
Proof.
  intros a b c d Hbranch.
  unfold min4_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool eqn:E1.
  - apply min4_eq_first. exact E1.
  - (* E1 = false means first branch not taken, so result is 2, 3, or 4 *)
    destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool eqn:E2; [discriminate|].
    destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool eqn:E3; discriminate.
Qed.

(** min4_branch = 2 implies min equals second argument *)
Lemma min4_branch_eq_2_implies : forall a b c d,
  min4_branch a b c d = 2 ->
  min (min3 a b c) d = b.
Proof.
  intros a b c d Hbranch.
  unfold min4_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool eqn:E2.
  - apply min4_eq_second; assumption.
  - destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool eqn:E3; discriminate.
Qed.

(** min4_branch = 3 implies min equals third argument *)
Lemma min4_branch_eq_3_implies : forall a b c d,
  min4_branch a b c d = 3 ->
  min (min3 a b c) d = c.
Proof.
  intros a b c d Hbranch.
  unfold min4_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool eqn:E3.
  - apply min4_eq_third; assumption.
  - discriminate.
Qed.

(** min4_branch = 4 implies min equals fourth argument *)
Lemma min4_branch_eq_4_implies : forall a b c d,
  min4_branch a b c d = 4 ->
  min (min3 a b c) d = d.
Proof.
  intros a b c d Hbranch.
  unfold min4_branch in Hbranch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool eqn:E1; [discriminate|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool eqn:E2; [discriminate|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool eqn:E3; [discriminate|].
  apply min4_eq_fourth; assumption.
Qed.

(** min4_branch returns at most 4 *)
Lemma min4_branch_le_4 : forall a b c d, min4_branch a b c d <= 4.
Proof.
  intros. unfold min4_branch.
  destruct ((a <=? b) && (a <=? c) && (a <=? d))%bool; [lia|].
  destruct ((b <=? a) && (b <=? c) && (b <=? d))%bool; [lia|].
  destruct ((c <=? a) && (c <=? b) && (c <=? d))%bool; lia.
Qed.

(** min4_branch >= 4 implies = 4 *)
Lemma min4_branch_ge_4_eq_4 : forall a b c d n,
  min4_branch a b c d = S (S (S (S n))) ->
  min4_branch a b c d = 4.
Proof.
  intros a b c d n H.
  pose proof (min4_branch_le_4 a b c d) as Hle.
  lia.
Qed.

(** Shift trace positions: add offset to source positions *)
Definition ms_trace_shift_A (offset : nat) (T : MSTrace) : MSTrace :=
  map (fun e =>
    match e with
    | MSMatch i j => MSMatch (i + offset) j
    | MSMerge2 i1 i2 j => MSMerge2 (i1 + offset) (i2 + offset) j
    | MSSplit2 i j1 j2 => MSSplit2 (i + offset) j1 j2
    | MSDouble i1 i2 j1 j2 => MSDouble (i1 + offset) (i2 + offset) j1 j2
    end) T.

(** Shift trace positions: add offset to target positions *)
Definition ms_trace_shift_B (offset : nat) (T : MSTrace) : MSTrace :=
  map (fun e =>
    match e with
    | MSMatch i j => MSMatch i (j + offset)
    | MSMerge2 i1 i2 j => MSMerge2 i1 i2 (j + offset)
    | MSSplit2 i j1 j2 => MSSplit2 i (j1 + offset) (j2 + offset)
    | MSDouble i1 i2 j1 j2 => MSDouble i1 i2 (j1 + offset) (j2 + offset)
    end) T.

(** Shift both positions *)
Definition ms_trace_shift_AB (offsetA offsetB : nat) (T : MSTrace) : MSTrace :=
  ms_trace_shift_B offsetB (ms_trace_shift_A offsetA T).

(** Optimal MS trace construction via DP backtracking

    For each case, we determine which branch of the DP wins and construct
    the corresponding trace element.

    Branch structure for main case (c1::c2::s1', d1::d2::s2'):
    1. Delete c1: no trace element (position 1 in A not matched)
    2. Insert d1: no trace element (position 1 in B not matched)
    3. Subst c1→d1: MSMatch 1 1
    4. Merge c1,c2→d1: MSMerge2 1 2 1
    5. Split c1→d1,d2: MSSplit2 1 1 2
    6. Double-subst: MSDouble 1 2 1 2
*)
Function ms_optimal_trace_pair (p : list Char * list Char)
  {measure ms_optimal_trace_measure p} : MSTrace :=
  match p with
  | ([], _) => []
  | (_, []) => []
  | ([c1], [d1]) =>
      (* Single char each: match if equal, otherwise subst (still match position) *)
      [MSMatch 1 1]
  | ([c1], d1 :: d2 :: s2') =>
      (* Single source, 2+ target: potential split *)
      let cost_del := merge_split_pair ([], d1 :: d2 :: s2') + 1 in
      let cost_ins := merge_split_pair ([c1], d2 :: s2') + 1 in
      let cost_sub := merge_split_pair ([], d2 :: s2') + subst_cost c1 d1 in
      let cost_spl := merge_split_pair ([], s2') + split_cost c1 d1 d2 in
      let branch := min4_branch cost_del cost_ins cost_sub cost_spl in
      match branch with
      | 1 => (* delete *) ms_trace_shift_B 1 (ms_optimal_trace_pair ([], d1 :: d2 :: s2'))
      | 2 => (* insert *) ms_trace_shift_B 1 (ms_optimal_trace_pair ([c1], d2 :: s2'))
      | 3 => (* subst *) MSMatch 1 1 :: ms_trace_shift_B 1 (ms_optimal_trace_pair ([], d2 :: s2'))
      | _ => (* split *) MSSplit2 1 1 2 :: ms_trace_shift_B 2 (ms_optimal_trace_pair ([], s2'))
      end
  | (c1 :: c2 :: s1', [d1]) =>
      (* 2+ source, single target: potential merge *)
      let cost_del := merge_split_pair (c2 :: s1', [d1]) + 1 in
      let cost_ins := merge_split_pair (c1 :: c2 :: s1', []) + 1 in
      let cost_sub := merge_split_pair (c2 :: s1', []) + subst_cost c1 d1 in
      let cost_mer := merge_split_pair (s1', []) + merge_cost c1 c2 d1 in
      let branch := min4_branch cost_del cost_ins cost_sub cost_mer in
      match branch with
      | 1 => (* delete *) ms_trace_shift_A 1 (ms_optimal_trace_pair (c2 :: s1', [d1]))
      | 2 => (* insert *) ms_trace_shift_A 1 (ms_optimal_trace_pair (c1 :: c2 :: s1', []))
      | 3 => (* subst *) MSMatch 1 1 :: ms_trace_shift_A 1 (ms_optimal_trace_pair (c2 :: s1', []))
      | _ => (* merge *) MSMerge2 1 2 1 :: ms_trace_shift_A 2 (ms_optimal_trace_pair (s1', []))
      end
  | (c1 :: c2 :: s1', d1 :: d2 :: s2') =>
      (* Main case: all 6 branches possible *)
      let cost_del := merge_split_pair (c2 :: s1', d1 :: d2 :: s2') + 1 in
      let cost_ins := merge_split_pair (c1 :: c2 :: s1', d2 :: s2') + 1 in
      let cost_sub := merge_split_pair (c2 :: s1', d2 :: s2') + subst_cost c1 d1 in
      let cost_mer := merge_split_pair (s1', d2 :: s2') + merge_cost c1 c2 d1 in
      let cost_spl := merge_split_pair (c2 :: s1', s2') + split_cost c1 d1 d2 in
      let cost_dbl := merge_split_pair (s1', s2') + subst_cost c1 d1 + subst_cost c2 d2 in
      let branch := min6_branch cost_del cost_ins cost_sub cost_mer cost_spl cost_dbl in
      match branch with
      | 1 => (* delete c1 *)
          ms_trace_shift_A 1 (ms_optimal_trace_pair (c2 :: s1', d1 :: d2 :: s2'))
      | 2 => (* insert d1 *)
          ms_trace_shift_B 1 (ms_optimal_trace_pair (c1 :: c2 :: s1', d2 :: s2'))
      | 3 => (* subst c1→d1 *)
          MSMatch 1 1 :: ms_trace_shift_AB 1 1 (ms_optimal_trace_pair (c2 :: s1', d2 :: s2'))
      | 4 => (* merge c1,c2→d1 *)
          MSMerge2 1 2 1 :: ms_trace_shift_AB 2 1 (ms_optimal_trace_pair (s1', d2 :: s2'))
      | 5 => (* split c1→d1,d2 *)
          MSSplit2 1 1 2 :: ms_trace_shift_AB 1 2 (ms_optimal_trace_pair (c2 :: s1', s2'))
      | _ => (* double-subst c1→d1, c2→d2 *)
          MSDouble 1 2 1 2 :: ms_trace_shift_AB 2 2 (ms_optimal_trace_pair (s1', s2'))
      end
  end.
Proof.
  (* Termination proofs - each recursive call decreases measure *)
  all: intros; unfold ms_optimal_trace_measure; simpl; try lia.
Defined.

(** Wrapper: optimal MS trace for given strings A, B *)
Definition ms_optimal_trace (A B : list Char) : MSTrace :=
  ms_optimal_trace_pair (A, B).

(** * Shift Lemmas *)

Lemma ms_trace_shift_A_nil : forall n,
  ms_trace_shift_A n [] = [].
Proof. reflexivity. Qed.

Lemma ms_trace_shift_B_nil : forall n,
  ms_trace_shift_B n [] = [].
Proof. reflexivity. Qed.

Lemma ms_trace_shift_AB_nil : forall n m,
  ms_trace_shift_AB n m [] = [].
Proof.
  intros. unfold ms_trace_shift_AB.
  rewrite ms_trace_shift_A_nil, ms_trace_shift_B_nil. reflexivity.
Qed.

(** Empty input produces empty trace *)
Lemma ms_optimal_trace_pair_nil_l : forall B,
  ms_optimal_trace_pair ([], B) = [].
Proof.
  intro B. rewrite ms_optimal_trace_pair_equation.
  reflexivity.
Qed.

Lemma ms_optimal_trace_pair_nil_r : forall A,
  ms_optimal_trace_pair (A, []) = [].
Proof.
  intro A. rewrite ms_optimal_trace_pair_equation.
  destruct A as [|c [|c2 A']]; reflexivity.
Qed.

(** * Position Bounds Lemmas *)

(** All positions in optimal trace are >= 1 *)
Definition ms_trace_positions_ge_1 (T : MSTrace) : Prop :=
  Forall (fun e =>
    match e with
    | MSMatch i j => i >= 1 /\ j >= 1
    | MSMerge2 i1 i2 j => i1 >= 1 /\ i2 >= 1 /\ j >= 1
    | MSSplit2 i j1 j2 => i >= 1 /\ j1 >= 1 /\ j2 >= 1
    | MSDouble i1 i2 j1 j2 => i1 >= 1 /\ i2 >= 1 /\ j1 >= 1 /\ j2 >= 1
    end) T.

Lemma ms_trace_shift_A_preserves_ge1 : forall T n,
  ms_trace_positions_ge_1 T ->
  n >= 0 ->
  ms_trace_positions_ge_1 (ms_trace_shift_A n T).
Proof.
  intros T n HT Hn.
  unfold ms_trace_positions_ge_1 in *.
  induction T as [| e rest IH].
  - constructor.
  - simpl. constructor.
    + inversion HT as [| ? ? He Hrest]; subst.
      destruct e; simpl in *; lia.
    + apply IH. inversion HT; assumption.
Qed.

Lemma ms_trace_shift_B_preserves_ge1 : forall T n,
  ms_trace_positions_ge_1 T ->
  n >= 0 ->
  ms_trace_positions_ge_1 (ms_trace_shift_B n T).
Proof.
  intros T n HT Hn.
  unfold ms_trace_positions_ge_1 in *.
  induction T as [| e rest IH].
  - constructor.
  - simpl. constructor.
    + inversion HT as [| ? ? He Hrest]; subst.
      destruct e; simpl in *; lia.
    + apply IH. inversion HT; assumption.
Qed.

(** Optimal trace has positions >= 1 *)
Lemma ms_optimal_trace_pair_positions_ge1 : forall p,
  ms_trace_positions_ge_1 (ms_optimal_trace_pair p).
Proof.
  intro p.
  remember (ms_optimal_trace_measure p) as n eqn:Hn.
  revert p Hn.
  induction n as [n IH] using lt_wf_ind.
  intros [A B] Hn.
  rewrite ms_optimal_trace_pair_equation.
  destruct A as [| c1 [| c2 s1']];
  destruct B as [| d1 [| d2 s2']];
  try (constructor; fail).
  - (* [c1], [d1] *)
    unfold ms_trace_positions_ge_1. constructor; [lia | constructor].
  - (* [c1], d1::d2::s2' *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch.
    + (* 0: _ case = split *)
      simpl. constructor; [lia|].
      constructor. (* tail is [] after simpl *)
    + (* 1: delete *)
      simpl. constructor. (* trace is [] after simpl *)
    + (* 2: insert *)
      simpl. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure ([c1], d2 :: s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 3: subst *)
      simpl. constructor; [lia|].
      constructor. (* tail is [] after simpl *)
    + (* 4+: _ case = split *)
      simpl. constructor; [lia|].
      constructor. (* tail is [] after simpl *)
  - (* c1::c2::s1', [d1] *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch.
    + (* 0: _ case = merge *)
      constructor; [simpl; lia|].
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil. constructor.
    + (* 1: delete *)
      simpl. apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (c2 :: s1', [d1])).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 2: insert *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil. constructor.
    + (* 3: subst *)
      constructor; [simpl; lia|].
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil. constructor.
    + (* 4+: _ case = merge *)
      constructor; [simpl; lia|].
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil. constructor.
  - (* c1::c2::s1', d1::d2::s2' - main case *)
    cbv zeta.
    destruct (min6_branch _ _ _ _ _ _) as [|[|[|[|[|[|]]]]]] eqn:Ebranch; simpl.
    + (* 0: matches _ case = double-subst: recursive on (s1', s2') *)
      constructor; [lia |].
      unfold ms_trace_shift_AB. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (s1', s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 1: delete: recursive on (c2 :: s1', d1 :: d2 :: s2') *)
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 2: insert: recursive on (c1 :: c2 :: s1', d2 :: s2') *)
      apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 3: subst: recursive on (c2 :: s1', d2 :: s2') *)
      constructor; [lia |].
      unfold ms_trace_shift_AB. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (c2 :: s1', d2 :: s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 4: merge: recursive on (s1', d2 :: s2') *)
      constructor; [lia |].
      unfold ms_trace_shift_AB. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (s1', d2 :: s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 5: split: recursive on (c2 :: s1', s2') *)
      constructor; [lia |].
      unfold ms_trace_shift_AB. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (c2 :: s1', s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
    + (* 6+: matches _ case = double-subst: recursive on (s1', s2') *)
      constructor; [lia |].
      unfold ms_trace_shift_AB. apply ms_trace_shift_B_preserves_ge1; [|lia].
      apply ms_trace_shift_A_preserves_ge1; [|lia].
      apply IH with (m := ms_optimal_trace_measure (s1', s2')).
      * unfold ms_optimal_trace_measure in *; simpl in *; lia.
      * reflexivity.
Qed.

(** * Existence Theorem *)

(** There exists an MS trace with positions >= 1 *)
Theorem ms_optimal_trace_positions_exist : forall (A B : list Char),
  exists T : MSTrace,
    ms_trace_positions_ge_1 T.
Proof.
  intros A B.
  exists (ms_optimal_trace A B).
  apply ms_optimal_trace_pair_positions_ge1.
Qed.

(** * Shift and Position Lemmas *)

(** Shifting A positions by n adds n to each position in the trace's A positions *)
Lemma ms_trace_shift_A_positions_A : forall T n,
  ms_trace_positions_A (ms_trace_shift_A n T) =
  map (fun i => i + n) (ms_trace_positions_A T).
Proof.
  induction T as [| e rest IH]; intros n.
  - reflexivity.
  - simpl. rewrite IH.
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl;
    f_equal; try reflexivity.
Qed.

(** Shifting A positions doesn't change B positions *)
Lemma ms_trace_shift_A_positions_B : forall T n,
  ms_trace_positions_B (ms_trace_shift_A n T) = ms_trace_positions_B T.
Proof.
  induction T as [| e rest IH]; intros n.
  - reflexivity.
  - simpl. rewrite IH.
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Shifting B positions by n adds n to each position in the trace's B positions *)
Lemma ms_trace_shift_B_positions_B : forall T n,
  ms_trace_positions_B (ms_trace_shift_B n T) =
  map (fun j => j + n) (ms_trace_positions_B T).
Proof.
  induction T as [| e rest IH]; intros n.
  - reflexivity.
  - simpl. rewrite IH.
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl;
    f_equal; try reflexivity.
Qed.

(** Shifting B positions doesn't change A positions *)
Lemma ms_trace_shift_B_positions_A : forall T n,
  ms_trace_positions_A (ms_trace_shift_B n T) = ms_trace_positions_A T.
Proof.
  induction T as [| e rest IH]; intros n.
  - reflexivity.
  - simpl. rewrite IH.
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Length of shifted positions equals original *)
Lemma map_add_length : forall (l : list nat) n,
  length (map (fun i => i + n) l) = length l.
Proof.
  intros. rewrite map_length. reflexivity.
Qed.

(** Shift A preserves position count *)
Lemma ms_trace_shift_A_positions_A_length : forall T n,
  length (ms_trace_positions_A (ms_trace_shift_A n T)) =
  length (ms_trace_positions_A T).
Proof.
  intros. rewrite ms_trace_shift_A_positions_A, map_add_length. reflexivity.
Qed.

(** Shift B preserves position count *)
Lemma ms_trace_shift_B_positions_B_length : forall T n,
  length (ms_trace_positions_B (ms_trace_shift_B n T)) =
  length (ms_trace_positions_B T).
Proof.
  intros. rewrite ms_trace_shift_B_positions_B, map_add_length. reflexivity.
Qed.

(** * Change Cost and Shifting *)

(** Helper: nth with successor index on cons list *)
Lemma nth_cons_succ : forall (A : Type) n (c : A) l d,
  nth (S n) (c :: l) d = nth n l d.
Proof. reflexivity. Qed.

(** Key lemma: for 1-indexed positions i >= 1, accessing position i in (c::A')
    after shifting gives the same result as accessing position (i-1) in A'.
    Note: nth (i + 1 - 1) (c::A') d = nth i (c::A') d = nth (i-1) A' d (for i >= 1)
*)
Lemma nth_shift_cons : forall (T : Type) (c : T) A' d i,
  i >= 1 ->
  nth (i + 1 - 1) (c :: A') d = nth (i - 1) A' d.
Proof.
  intros T c A' d i Hi.
  replace (i + 1 - 1) with i by lia.
  destruct i as [| i']; [lia |].
  simpl. f_equal. lia.
Qed.

(** Helper for rewriting nth with shifted index - handles simpl'd form *)
Lemma nth_shift_cons_alt : forall (T : Type) (c : T) A' d j,
  j >= 1 ->
  match j + 1 - 1 with
  | 0 => c
  | S m => nth m A' d
  end = nth (j - 1) A' d.
Proof.
  intros T c A' d j Hj.
  replace (j + 1 - 1) with j by lia.
  destruct j as [| j']; [lia |].
  simpl. f_equal. lia.
Qed.

(** When we shift A positions by 1, the change cost with (c::A') equals
    the change cost of the unshifted trace with A'. This requires that
    all positions are >= 1 (1-indexed and valid).

    For valid traces, position (i+1) in (c::A') accesses character at
    index i, which is the same as position i in A' (which accesses index i-1).
*)
Lemma ms_element_cost_shift_A_1 : forall A' B c e,
  (match e with
   | MSMatch i _ => i >= 1
   | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
   | MSSplit2 i _ _ => i >= 1
   | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
   end) ->
  ms_element_cost (c :: A') B (
    match e with
    | MSMatch i j => MSMatch (i + 1) j
    | MSMerge2 i1 i2 j => MSMerge2 (i1 + 1) (i2 + 1) j
    | MSSplit2 i j1 j2 => MSSplit2 (i + 1) j1 j2
    | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 1) (i2 + 1) j1 j2
    end
  ) = ms_element_cost A' B e.
Proof.
  intros A' B c e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *)
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hpos).
    reflexivity.
  - (* MSMerge2 *)
    destruct Hpos as [Hi1 Hi2].
    unfold merge_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    reflexivity.
  - (* MSSplit2 *)
    unfold split_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hpos).
    reflexivity.
  - (* MSDouble *)
    destruct Hpos as [Hi1 Hi2].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    reflexivity.
Qed.

(** Similar for B shifts - shift by 1 *)
Lemma ms_element_cost_shift_B_1 : forall A B' d e,
  (match e with
   | MSMatch _ j => j >= 1
   | MSMerge2 _ _ j => j >= 1
   | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
   | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost A (d :: B') (
    match e with
    | MSMatch i j => MSMatch i (j + 1)
    | MSMerge2 i1 i2 j => MSMerge2 i1 i2 (j + 1)
    | MSSplit2 i j1 j2 => MSSplit2 i (j1 + 1) (j2 + 1)
    | MSDouble i1 i2 j1 j2 => MSDouble i1 i2 (j1 + 1) (j2 + 1)
    end
  ) = ms_element_cost A B' e.
Proof.
  intros A B' d e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *)
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j Hpos).
    reflexivity.
  - (* MSMerge2 *)
    unfold merge_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j Hpos).
    reflexivity.
  - (* MSSplit2 *)
    destruct Hpos as [Hj1 Hj2].
    unfold split_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
  - (* MSDouble *)
    destruct Hpos as [Hj1 Hj2].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
Qed.

(** * Base Case Cost Lemmas *)

(** Empty trace costs *)
Lemma ms_trace_cost_nil : forall A B,
  ms_trace_cost A B [] = length A + length B.
Proof.
  intros A B.
  unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
  unfold ms_trace_positions_A, ms_trace_positions_B. simpl.
  lia.
Qed.

(** Single match element cost *)
Lemma ms_trace_cost_single_match : forall c d,
  ms_trace_cost [c] [d] [MSMatch 1 1] = subst_cost c d.
Proof.
  intros c d.
  unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
  unfold ms_trace_positions_A, ms_trace_positions_B. simpl.
  unfold ms_element_positions_A, ms_element_positions_B. simpl.
  unfold ms_element_cost. simpl.
  lia.
Qed.

(** * Trace Cost with Shift Lemmas *)

(** Helper: shift a single element by 1 in both A and B positions *)
Definition ms_element_shift_AB_1 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 1) (j + 1)
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 1) (i2 + 1) (j + 1)
  | MSSplit2 i j1 j2 => MSSplit2 (i + 1) (j1 + 1) (j2 + 1)
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 1) (i2 + 1) (j1 + 1) (j2 + 1)
  end.

(** Helper: shift a single element by (2,1) - for merge branch *)
Definition ms_element_shift_AB_2_1 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 2) (j + 1)
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 2) (i2 + 2) (j + 1)
  | MSSplit2 i j1 j2 => MSSplit2 (i + 2) (j1 + 1) (j2 + 1)
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 2) (i2 + 2) (j1 + 1) (j2 + 1)
  end.

(** Helper: shift a single element by (1,2) - for split branch *)
Definition ms_element_shift_AB_1_2 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 1) (j + 2)
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 1) (i2 + 1) (j + 2)
  | MSSplit2 i j1 j2 => MSSplit2 (i + 1) (j1 + 2) (j2 + 2)
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 1) (i2 + 1) (j1 + 2) (j2 + 2)
  end.

(** Helper: shift a single element by (2,2) - for double-subst branch *)
Definition ms_element_shift_AB_2_2 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 2) (j + 2)
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 2) (i2 + 2) (j + 2)
  | MSSplit2 i j1 j2 => MSSplit2 (i + 2) (j1 + 2) (j2 + 2)
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 2) (i2 + 2) (j1 + 2) (j2 + 2)
  end.

(** Element cost is preserved when shifting both positions by 1 *)
Lemma ms_element_cost_shift_AB_1 : forall A' B' c d e,
  (match e with
   | MSMatch i j => i >= 1 /\ j >= 1
   | MSMerge2 i1 i2 j => i1 >= 1 /\ i2 >= 1 /\ j >= 1
   | MSSplit2 i j1 j2 => i >= 1 /\ j1 >= 1 /\ j2 >= 1
   | MSDouble i1 i2 j1 j2 => i1 >= 1 /\ i2 >= 1 /\ j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost (c :: A') (d :: B') (ms_element_shift_AB_1 e) =
  ms_element_cost A' B' e.
Proof.
  intros A' B' c d e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *)
    destruct Hpos as [Hi Hj].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hi).
    rewrite (nth_shift_cons_alt Char d B' default_char j Hj).
    reflexivity.
  - (* MSMerge2 *)
    destruct Hpos as [Hi1 [Hi2 Hj]].
    unfold merge_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    rewrite (nth_shift_cons_alt Char d B' default_char j Hj).
    reflexivity.
  - (* MSSplit2 *)
    destruct Hpos as [Hi [Hj1 Hj2]].
    unfold split_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hi).
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
  - (* MSDouble *)
    destruct Hpos as [Hi1 [Hi2 [Hj1 Hj2]]].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
Qed.

(** Helper: shifting index by 2 in a 2-cons list *)
Lemma nth_shift_cons_cons_2_alt : forall (X : Type) (c1 c2 : X) L def i,
  i >= 1 ->
  nth (i + 2 - 1) (c1 :: c2 :: L) def = nth (i - 1) L def.
Proof.
  intros X c1 c2 L def i Hi.
  assert (H: i + 2 - 1 = S (S (i - 1))) by lia.
  rewrite H. simpl. reflexivity.
Qed.

(** Element cost is preserved when shifting by (2,1) - for merge branch *)
Lemma ms_element_cost_shift_AB_2_1 : forall A' B' c1 c2 d e,
  (match e with
   | MSMatch i j => i >= 1 /\ j >= 1
   | MSMerge2 i1 i2 j => i1 >= 1 /\ i2 >= 1 /\ j >= 1
   | MSSplit2 i j1 j2 => i >= 1 /\ j1 >= 1 /\ j2 >= 1
   | MSDouble i1 i2 j1 j2 => i1 >= 1 /\ i2 >= 1 /\ j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost (c1 :: c2 :: A') (d :: B') (ms_element_shift_AB_2_1 e) =
  ms_element_cost A' B' e.
Proof.
  intros A' B' c1 c2 d e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl ms_element_shift_AB_2_1; unfold ms_element_cost.
  - (* MSMatch *)
    simpl in Hpos. destruct Hpos as [Hi Hj].
    assert (HA: i + 2 - 1 = S (S (i - 1))) by lia.
    assert (HB: j + 1 - 1 = S (j - 1)) by lia.
    rewrite HA, HB. simpl. reflexivity.
  - (* MSMerge2 *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 Hj]].
    unfold merge_cost.
    assert (HA1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (HA2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    assert (HB: j + 1 - 1 = S (j - 1)) by lia.
    rewrite HA1, HA2, HB. simpl. reflexivity.
  - (* MSSplit2 *)
    simpl in Hpos. destruct Hpos as [Hi [Hj1 Hj2]].
    unfold split_cost.
    assert (HA: i + 2 - 1 = S (S (i - 1))) by lia.
    assert (HB1: j1 + 1 - 1 = S (j1 - 1)) by lia.
    assert (HB2: j2 + 1 - 1 = S (j2 - 1)) by lia.
    rewrite HA, HB1, HB2. simpl. reflexivity.
  - (* MSDouble *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 [Hj1 Hj2]]].
    assert (HA1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (HA2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    assert (HB1: j1 + 1 - 1 = S (j1 - 1)) by lia.
    assert (HB2: j2 + 1 - 1 = S (j2 - 1)) by lia.
    rewrite HA1, HB1, HA2, HB2. simpl. reflexivity.
Qed.

(** Element cost is preserved when shifting by (1,2) - for split branch *)
Lemma ms_element_cost_shift_AB_1_2 : forall A' B' c d1 d2 e,
  (match e with
   | MSMatch i j => i >= 1 /\ j >= 1
   | MSMerge2 i1 i2 j => i1 >= 1 /\ i2 >= 1 /\ j >= 1
   | MSSplit2 i j1 j2 => i >= 1 /\ j1 >= 1 /\ j2 >= 1
   | MSDouble i1 i2 j1 j2 => i1 >= 1 /\ i2 >= 1 /\ j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost (c :: A') (d1 :: d2 :: B') (ms_element_shift_AB_1_2 e) =
  ms_element_cost A' B' e.
Proof.
  intros A' B' c d1 d2 e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl ms_element_shift_AB_1_2; unfold ms_element_cost.
  - (* MSMatch *)
    simpl in Hpos. destruct Hpos as [Hi Hj].
    assert (HA: i + 1 - 1 = S (i - 1)) by lia.
    assert (HB: j + 2 - 1 = S (S (j - 1))) by lia.
    rewrite HA, HB. simpl. reflexivity.
  - (* MSMerge2 *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 Hj]].
    unfold merge_cost.
    assert (HA1: i1 + 1 - 1 = S (i1 - 1)) by lia.
    assert (HA2: i2 + 1 - 1 = S (i2 - 1)) by lia.
    assert (HB: j + 2 - 1 = S (S (j - 1))) by lia.
    rewrite HA1, HA2, HB. simpl. reflexivity.
  - (* MSSplit2 *)
    simpl in Hpos. destruct Hpos as [Hi [Hj1 Hj2]].
    unfold split_cost.
    assert (HA: i + 1 - 1 = S (i - 1)) by lia.
    assert (HB1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (HB2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite HA, HB1, HB2. simpl. reflexivity.
  - (* MSDouble *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 [Hj1 Hj2]]].
    assert (HA1: i1 + 1 - 1 = S (i1 - 1)) by lia.
    assert (HA2: i2 + 1 - 1 = S (i2 - 1)) by lia.
    assert (HB1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (HB2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite HA1, HB1, HA2, HB2. simpl. reflexivity.
Qed.

(** Element cost is preserved when shifting by (2,2) - for double-subst branch *)
Lemma ms_element_cost_shift_AB_2_2 : forall A' B' c1 c2 d1 d2 e,
  (match e with
   | MSMatch i j => i >= 1 /\ j >= 1
   | MSMerge2 i1 i2 j => i1 >= 1 /\ i2 >= 1 /\ j >= 1
   | MSSplit2 i j1 j2 => i >= 1 /\ j1 >= 1 /\ j2 >= 1
   | MSDouble i1 i2 j1 j2 => i1 >= 1 /\ i2 >= 1 /\ j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost (c1 :: c2 :: A') (d1 :: d2 :: B') (ms_element_shift_AB_2_2 e) =
  ms_element_cost A' B' e.
Proof.
  intros A' B' c1 c2 d1 d2 e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl ms_element_shift_AB_2_2; unfold ms_element_cost.
  - (* MSMatch *)
    simpl in Hpos. destruct Hpos as [Hi Hj].
    assert (HA: i + 2 - 1 = S (S (i - 1))) by lia.
    assert (HB: j + 2 - 1 = S (S (j - 1))) by lia.
    rewrite HA, HB. simpl. reflexivity.
  - (* MSMerge2 *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 Hj]].
    unfold merge_cost.
    assert (HA1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (HA2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    assert (HB: j + 2 - 1 = S (S (j - 1))) by lia.
    rewrite HA1, HA2, HB. simpl. reflexivity.
  - (* MSSplit2 *)
    simpl in Hpos. destruct Hpos as [Hi [Hj1 Hj2]].
    unfold split_cost.
    assert (HA: i + 2 - 1 = S (S (i - 1))) by lia.
    assert (HB1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (HB2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite HA, HB1, HB2. simpl. reflexivity.
  - (* MSDouble *)
    simpl in Hpos. destruct Hpos as [Hi1 [Hi2 [Hj1 Hj2]]].
    assert (HA1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (HA2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    assert (HB1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (HB2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite HA1, HB1, HA2, HB2. simpl. reflexivity.
Qed.

(** ms_trace_shift_AB 1 1 equals mapping ms_element_shift_AB_1 *)
Lemma ms_trace_shift_AB_1_map : forall T,
  ms_trace_shift_AB 1 1 T = map ms_element_shift_AB_1 T.
Proof.
  intros T.
  unfold ms_trace_shift_AB, ms_trace_shift_A, ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** ms_trace_shift_AB 2 1 equals mapping ms_element_shift_AB_2_1 *)
Lemma ms_trace_shift_AB_2_1_map : forall T,
  ms_trace_shift_AB 2 1 T = map ms_element_shift_AB_2_1 T.
Proof.
  intros T.
  unfold ms_trace_shift_AB, ms_trace_shift_A, ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** ms_trace_shift_AB 1 2 equals mapping ms_element_shift_AB_1_2 *)
Lemma ms_trace_shift_AB_1_2_map : forall T,
  ms_trace_shift_AB 1 2 T = map ms_element_shift_AB_1_2 T.
Proof.
  intros T.
  unfold ms_trace_shift_AB, ms_trace_shift_A, ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** ms_trace_shift_AB 2 2 equals mapping ms_element_shift_AB_2_2 *)
Lemma ms_trace_shift_AB_2_2_map : forall T,
  ms_trace_shift_AB 2 2 T = map ms_element_shift_AB_2_2 T.
Proof.
  intros T.
  unfold ms_trace_shift_AB, ms_trace_shift_A, ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** Helper for fold_left with non-zero initial accumulator *)
Lemma fold_left_add_init : forall (f : MSTraceElement -> nat) (T : list MSTraceElement) (init : nat),
  fold_left (fun acc e => acc + f e) T init = init + fold_left (fun acc e => acc + f e) T 0.
Proof.
  intros f T init.
  revert init.
  induction T as [| e rest IH] using rev_ind.
  - simpl. lia.
  - intro init. rewrite !fold_left_app. simpl.
    rewrite IH. rewrite (IH 0). lia.
Qed.

(** Change cost is preserved when shifting all positions by 1 *)
Lemma ms_trace_change_cost_shift_AB_1 : forall A' B' c d T,
  ms_trace_positions_ge_1 T ->
  ms_trace_change_cost (c :: A') (d :: B') (ms_trace_shift_AB 1 1 T) =
  ms_trace_change_cost A' B' T.
Proof.
  intros A' B' c d T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_AB_1_map.
  (* Now we have: fold_left (fun acc e => acc + ms_element_cost (c::A') (d::B') e) (map shift T) 0
                = fold_left (fun acc e => acc + ms_element_cost A' B' e) T 0 *)
  induction T as [| e rest IH] using rev_ind.
  - reflexivity.
  - rewrite map_app. simpl.
    rewrite !fold_left_app. simpl.
    (* Extract Hpos for rest and e *)
    unfold ms_trace_positions_ge_1 in Hpos.
    apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
    inversion He_list as [| ? ? He _]; subst; clear He_list.
    specialize (IH Hrest).
    rewrite IH.
    rewrite (ms_element_cost_shift_AB_1 A' B' c d e He).
    reflexivity.
Qed.

(** Change cost is preserved when shifting by (2,1) - for merge branch *)
Lemma ms_trace_change_cost_shift_AB_2_1 : forall A' B' c1 c2 d T,
  ms_trace_positions_ge_1 T ->
  ms_trace_change_cost (c1 :: c2 :: A') (d :: B') (ms_trace_shift_AB 2 1 T) =
  ms_trace_change_cost A' B' T.
Proof.
  intros A' B' c1 c2 d T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_AB_2_1_map.
  induction T as [| e rest IH] using rev_ind.
  - reflexivity.
  - rewrite map_app. simpl.
    rewrite !fold_left_app. simpl.
    unfold ms_trace_positions_ge_1 in Hpos.
    apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
    inversion He_list as [| ? ? He _]; subst; clear He_list.
    specialize (IH Hrest).
    rewrite IH.
    rewrite (ms_element_cost_shift_AB_2_1 A' B' c1 c2 d e He).
    reflexivity.
Qed.

(** Change cost is preserved when shifting by (1,2) - for split branch *)
Lemma ms_trace_change_cost_shift_AB_1_2 : forall A' B' c d1 d2 T,
  ms_trace_positions_ge_1 T ->
  ms_trace_change_cost (c :: A') (d1 :: d2 :: B') (ms_trace_shift_AB 1 2 T) =
  ms_trace_change_cost A' B' T.
Proof.
  intros A' B' c d1 d2 T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_AB_1_2_map.
  induction T as [| e rest IH] using rev_ind.
  - reflexivity.
  - rewrite map_app. simpl.
    rewrite !fold_left_app. simpl.
    unfold ms_trace_positions_ge_1 in Hpos.
    apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
    inversion He_list as [| ? ? He _]; subst; clear He_list.
    specialize (IH Hrest).
    rewrite IH.
    rewrite (ms_element_cost_shift_AB_1_2 A' B' c d1 d2 e He).
    reflexivity.
Qed.

(** Change cost is preserved when shifting by (2,2) - for double-subst branch *)
Lemma ms_trace_change_cost_shift_AB_2_2 : forall A' B' c1 c2 d1 d2 T,
  ms_trace_positions_ge_1 T ->
  ms_trace_change_cost (c1 :: c2 :: A') (d1 :: d2 :: B') (ms_trace_shift_AB 2 2 T) =
  ms_trace_change_cost A' B' T.
Proof.
  intros A' B' c1 c2 d1 d2 T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_AB_2_2_map.
  induction T as [| e rest IH] using rev_ind.
  - reflexivity.
  - rewrite map_app. simpl.
    rewrite !fold_left_app. simpl.
    unfold ms_trace_positions_ge_1 in Hpos.
    apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
    inversion He_list as [| ? ? He _]; subst; clear He_list.
    specialize (IH Hrest).
    rewrite IH.
    rewrite (ms_element_cost_shift_AB_2_2 A' B' c1 c2 d1 d2 e He).
    reflexivity.
Qed.

(** Shifted positions preserve length *)
Lemma ms_trace_shift_AB_positions_A_length : forall T,
  length (ms_trace_positions_A (ms_trace_shift_AB 1 1 T)) = length (ms_trace_positions_A T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_1_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

Lemma ms_trace_shift_AB_positions_B_length : forall T,
  length (ms_trace_positions_B (ms_trace_shift_AB 1 1 T)) = length (ms_trace_positions_B T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_1_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** Position length preserved under (2,1) shift *)
Lemma ms_trace_shift_AB_2_1_positions_A_length : forall T,
  length (ms_trace_positions_A (ms_trace_shift_AB 2 1 T)) = length (ms_trace_positions_A T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_2_1_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

Lemma ms_trace_shift_AB_2_1_positions_B_length : forall T,
  length (ms_trace_positions_B (ms_trace_shift_AB 2 1 T)) = length (ms_trace_positions_B T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_2_1_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** Position length preserved under (1,2) shift *)
Lemma ms_trace_shift_AB_1_2_positions_A_length : forall T,
  length (ms_trace_positions_A (ms_trace_shift_AB 1 2 T)) = length (ms_trace_positions_A T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_1_2_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

Lemma ms_trace_shift_AB_1_2_positions_B_length : forall T,
  length (ms_trace_positions_B (ms_trace_shift_AB 1 2 T)) = length (ms_trace_positions_B T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_1_2_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** Position length preserved under (2,2) shift *)
Lemma ms_trace_shift_AB_2_2_positions_A_length : forall T,
  length (ms_trace_positions_A (ms_trace_shift_AB 2 2 T)) = length (ms_trace_positions_A T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_2_2_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

Lemma ms_trace_shift_AB_2_2_positions_B_length : forall T,
  length (ms_trace_positions_B (ms_trace_shift_AB 2 2 T)) = length (ms_trace_positions_B T).
Proof.
  intro T.
  rewrite ms_trace_shift_AB_2_2_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite !length_app.
    f_equal.
    + destruct e; reflexivity.
    + exact IH.
Qed.

(** Aliases for 1,1 shift position length preservation *)
Definition ms_trace_shift_AB_1_1_positions_A_length := ms_trace_shift_AB_positions_A_length.
Definition ms_trace_shift_AB_1_1_positions_B_length := ms_trace_shift_AB_positions_B_length.

(** Delete cost with prepended match element *)
Lemma ms_trace_delete_cost_cons_match : forall c A' T,
  ms_trace_delete_cost (c :: A') (MSMatch 1 1 :: ms_trace_shift_AB 1 1 T) =
  ms_trace_delete_cost A' T.
Proof.
  intros c A' T.
  unfold ms_trace_delete_cost, ms_trace_positions_A. simpl.
  (* Goal: S (length A') - S (length (flat_map ... (shift T))) = length A' - length (flat_map ... T) *)
  rewrite ms_trace_shift_AB_positions_A_length.
  (* Now both sides have same flat_map length *)
  reflexivity.
Qed.

(** Insert cost with prepended match element *)
Lemma ms_trace_insert_cost_cons_match : forall d B' T,
  ms_trace_insert_cost (d :: B') (MSMatch 1 1 :: ms_trace_shift_AB 1 1 T) =
  ms_trace_insert_cost B' T.
Proof.
  intros d B' T.
  unfold ms_trace_insert_cost, ms_trace_positions_B. simpl.
  rewrite ms_trace_shift_AB_positions_B_length.
  reflexivity.
Qed.

(** Delete cost with prepended merge element: MSMerge2 1 2 1 consumes positions 1,2 in A *)
Lemma ms_trace_delete_cost_cons_merge : forall c1 c2 A' T,
  length (ms_trace_positions_A T) <= length A' ->
  ms_trace_delete_cost (c1 :: c2 :: A') (MSMerge2 1 2 1 :: ms_trace_shift_AB 2 1 T) =
  ms_trace_delete_cost A' T.
Proof.
  intros c1 c2 A' T Hlen.
  unfold ms_trace_delete_cost, ms_trace_positions_A. simpl.
  rewrite ms_trace_shift_AB_2_1_positions_A_length.
  (* Both sides now: length A' - length (flat_map ms_element_positions_A T) *)
  reflexivity.
Qed.

(** Insert cost with prepended merge element: MSMerge2 1 2 1 consumes position 1 in B *)
Lemma ms_trace_insert_cost_cons_merge : forall d B' T,
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_insert_cost (d :: B') (MSMerge2 1 2 1 :: ms_trace_shift_AB 2 1 T) =
  ms_trace_insert_cost B' T.
Proof.
  intros d B' T Hlen.
  unfold ms_trace_insert_cost, ms_trace_positions_B. simpl.
  rewrite ms_trace_shift_AB_2_1_positions_B_length.
  reflexivity.
Qed.

(** Delete cost with prepended split element: MSSplit2 1 1 2 consumes position 1 in A *)
Lemma ms_trace_delete_cost_cons_split : forall c A' T,
  length (ms_trace_positions_A T) <= length A' ->
  ms_trace_delete_cost (c :: A') (MSSplit2 1 1 2 :: ms_trace_shift_AB 1 2 T) =
  ms_trace_delete_cost A' T.
Proof.
  intros c A' T Hlen.
  unfold ms_trace_delete_cost, ms_trace_positions_A. simpl.
  rewrite ms_trace_shift_AB_1_2_positions_A_length.
  reflexivity.
Qed.

(** Insert cost with prepended split element: MSSplit2 1 1 2 consumes positions 1,2 in B *)
Lemma ms_trace_insert_cost_cons_split : forall d1 d2 B' T,
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_insert_cost (d1 :: d2 :: B') (MSSplit2 1 1 2 :: ms_trace_shift_AB 1 2 T) =
  ms_trace_insert_cost B' T.
Proof.
  intros d1 d2 B' T Hlen.
  unfold ms_trace_insert_cost, ms_trace_positions_B. simpl.
  rewrite ms_trace_shift_AB_1_2_positions_B_length.
  reflexivity.
Qed.

(** Delete cost with prepended double element: MSDouble 1 2 1 2 consumes positions 1,2 in A *)
Lemma ms_trace_delete_cost_cons_double : forall c1 c2 A' T,
  length (ms_trace_positions_A T) <= length A' ->
  ms_trace_delete_cost (c1 :: c2 :: A') (MSDouble 1 2 1 2 :: ms_trace_shift_AB 2 2 T) =
  ms_trace_delete_cost A' T.
Proof.
  intros c1 c2 A' T Hlen.
  unfold ms_trace_delete_cost, ms_trace_positions_A. simpl.
  rewrite ms_trace_shift_AB_2_2_positions_A_length.
  reflexivity.
Qed.

(** Insert cost with prepended double element: MSDouble 1 2 1 2 consumes positions 1,2 in B *)
Lemma ms_trace_insert_cost_cons_double : forall d1 d2 B' T,
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_insert_cost (d1 :: d2 :: B') (MSDouble 1 2 1 2 :: ms_trace_shift_AB 2 2 T) =
  ms_trace_insert_cost B' T.
Proof.
  intros d1 d2 B' T Hlen.
  unfold ms_trace_insert_cost, ms_trace_positions_B. simpl.
  rewrite ms_trace_shift_AB_2_2_positions_B_length.
  reflexivity.
Qed.

(** The full trace cost decomposition for subst branch *)
Lemma ms_trace_cost_cons_match : forall c d A' B' T,
  ms_trace_positions_ge_1 T ->
  ms_trace_cost (c :: A') (d :: B') (MSMatch 1 1 :: ms_trace_shift_AB 1 1 T) =
  subst_cost c d + ms_trace_cost A' B' T.
Proof.
  intros c d A' B' T Hpos.
  unfold ms_trace_cost.
  (* Break into change + delete + insert costs *)
  (* Change cost for (MSMatch 1 1 :: shifted T) = subst c d + change(shifted T)
                                                = subst c d + change(T) *)
  assert (Hchange: ms_trace_change_cost (c :: A') (d :: B') (MSMatch 1 1 :: ms_trace_shift_AB 1 1 T) =
                   subst_cost c d + ms_trace_change_cost A' B' T).
  { unfold ms_trace_change_cost at 1. simpl.
    unfold ms_element_cost at 1. simpl.
    (* Now goal: fold_left ... (shift T) (subst c d) = subst c d + change_cost A' B' T *)
    rewrite fold_left_add_init.
    f_equal.
    (* Now need: fold_left ... (shift T) 0 = change_cost A' B' T *)
    apply ms_trace_change_cost_shift_AB_1. exact Hpos. }
  rewrite Hchange.
  (* Delete and insert costs use position counting - independent of actual characters *)
  rewrite (ms_trace_delete_cost_cons_match c A' T).
  rewrite (ms_trace_insert_cost_cons_match d B' T).
  lia.
Qed.

(** * Delete/Insert Branch Lemmas *)

(** For delete branch: we shift A positions by 1, adding 1 to delete cost *)

(** Helper: shift by 1 in A only *)
Definition ms_element_shift_A_1 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 1) j
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 1) (i2 + 1) j
  | MSSplit2 i j1 j2 => MSSplit2 (i + 1) j1 j2
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 1) (i2 + 1) j1 j2
  end.

Lemma ms_trace_shift_A_1_map : forall T,
  ms_trace_shift_A 1 T = map ms_element_shift_A_1 T.
Proof.
  intro T. unfold ms_trace_shift_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite IH. destruct e; reflexivity.
Qed.

(** Element cost preserved when shifting A positions by 1 *)
Lemma ms_element_cost_shift_A_1_full : forall A' B c e,
  (match e with
   | MSMatch i _ => i >= 1
   | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
   | MSSplit2 i _ _ => i >= 1
   | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
   end) ->
  ms_element_cost (c :: A') B (ms_element_shift_A_1 e) = ms_element_cost A' B e.
Proof.
  intros A' B c e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *)
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hpos).
    reflexivity.
  - (* MSMerge2 *)
    destruct Hpos as [Hi1 Hi2].
    unfold merge_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    reflexivity.
  - (* MSSplit2 *)
    unfold split_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i Hpos).
    reflexivity.
  - (* MSDouble *)
    destruct Hpos as [Hi1 Hi2].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char c A' default_char i1 Hi1).
    rewrite (nth_shift_cons_alt Char c A' default_char i2 Hi2).
    reflexivity.
Qed.

(** Change cost preserved when shifting A positions by 1 *)
Lemma ms_trace_change_cost_shift_A_1 : forall A' B c T,
  Forall (fun e =>
    match e with
    | MSMatch i _ => i >= 1
    | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
    | MSSplit2 i _ _ => i >= 1
    | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
    end) T ->
  ms_trace_change_cost (c :: A') B (ms_trace_shift_A 1 T) =
  ms_trace_change_cost A' B T.
Proof.
  intros A' B c T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_A_1_map.
  induction T as [| e rest IH] using rev_ind; [reflexivity |].
  rewrite map_app. simpl.
  rewrite !fold_left_app. simpl.
  apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
  inversion He_list as [| ? ? He _]; subst; clear He_list.
  specialize (IH Hrest).
  rewrite IH.
  rewrite (ms_element_cost_shift_A_1_full A' B c e He).
  reflexivity.
Qed.

(** Positions length preserved when shifting A *)
Lemma ms_trace_shift_A_positions_A_length_1 : forall T,
  length (ms_trace_positions_A (ms_trace_shift_A 1 T)) = length (ms_trace_positions_A T).
Proof.
  intro T. rewrite ms_trace_shift_A_1_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

Lemma ms_trace_shift_A_positions_B_length_1 : forall T,
  length (ms_trace_positions_B (ms_trace_shift_A 1 T)) = length (ms_trace_positions_B T).
Proof.
  intro T. rewrite ms_trace_shift_A_1_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

(** Delete branch: trace cost with shifted A positions adds 1 *)
Lemma ms_trace_cost_shift_A_delete : forall c A' B T,
  Forall (fun e =>
    match e with
    | MSMatch i _ => i >= 1
    | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
    | MSSplit2 i _ _ => i >= 1
    | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
    end) T ->
  length (ms_trace_positions_A T) <= length A' ->
  ms_trace_cost (c :: A') B (ms_trace_shift_A 1 T) =
  1 + ms_trace_cost A' B T.
Proof.
  intros c A' B T Hpos Hbound.
  unfold ms_trace_cost, ms_trace_delete_cost, ms_trace_insert_cost.
  (* Change cost is preserved *)
  rewrite (ms_trace_change_cost_shift_A_1 A' B c T Hpos).
  (* Positions lengths preserved under shift *)
  rewrite ms_trace_shift_A_positions_A_length_1.
  rewrite ms_trace_shift_A_positions_B_length_1.
  (* Delete cost: S(len A') - len(positions) = 1 + (len A' - len(positions)) *)
  simpl length. lia.
Qed.

(** Similarly for insert branch: shift B positions by 1 *)
Definition ms_element_shift_B_1 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch i (j + 1)
  | MSMerge2 i1 i2 j => MSMerge2 i1 i2 (j + 1)
  | MSSplit2 i j1 j2 => MSSplit2 i (j1 + 1) (j2 + 1)
  | MSDouble i1 i2 j1 j2 => MSDouble i1 i2 (j1 + 1) (j2 + 1)
  end.

Lemma ms_trace_shift_B_1_map : forall T,
  ms_trace_shift_B 1 T = map ms_element_shift_B_1 T.
Proof.
  intro T. unfold ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite IH. destruct e; reflexivity.
Qed.

(** Element cost preserved when shifting B positions by 1 *)
Lemma ms_element_cost_shift_B_1_full : forall A B' d e,
  (match e with
   | MSMatch _ j => j >= 1
   | MSMerge2 _ _ j => j >= 1
   | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
   | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost A (d :: B') (ms_element_shift_B_1 e) = ms_element_cost A B' e.
Proof.
  intros A B' d e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *)
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j Hpos).
    reflexivity.
  - (* MSMerge2 *)
    unfold merge_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j Hpos).
    reflexivity.
  - (* MSSplit2 *)
    destruct Hpos as [Hj1 Hj2].
    unfold split_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
  - (* MSDouble *)
    destruct Hpos as [Hj1 Hj2].
    unfold ms_element_cost.
    rewrite (nth_shift_cons_alt Char d B' default_char j1 Hj1).
    rewrite (nth_shift_cons_alt Char d B' default_char j2 Hj2).
    reflexivity.
Qed.

(** Change cost preserved when shifting B positions by 1 *)
Lemma ms_trace_change_cost_shift_B_1 : forall A B' d T,
  Forall (fun e =>
    match e with
    | MSMatch _ j => j >= 1
    | MSMerge2 _ _ j => j >= 1
    | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
    | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
    end) T ->
  ms_trace_change_cost A (d :: B') (ms_trace_shift_B 1 T) =
  ms_trace_change_cost A B' T.
Proof.
  intros A B' d T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_B_1_map.
  induction T as [| e rest IH] using rev_ind; [reflexivity |].
  rewrite map_app. simpl.
  rewrite !fold_left_app. simpl.
  apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
  inversion He_list as [| ? ? He _]; subst; clear He_list.
  specialize (IH Hrest).
  rewrite IH.
  rewrite (ms_element_cost_shift_B_1_full A B' d e He).
  reflexivity.
Qed.

(** Positions length preserved when shifting B *)
Lemma ms_trace_shift_B_positions_A_length_1 : forall T,
  length (ms_trace_positions_A (ms_trace_shift_B 1 T)) = length (ms_trace_positions_A T).
Proof.
  intro T. rewrite ms_trace_shift_B_1_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

Lemma ms_trace_shift_B_positions_B_length_1 : forall T,
  length (ms_trace_positions_B (ms_trace_shift_B 1 T)) = length (ms_trace_positions_B T).
Proof.
  intro T. rewrite ms_trace_shift_B_1_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

(** Insert branch: trace cost with shifted B positions adds 1 *)
Lemma ms_trace_cost_shift_B_insert : forall A d B' T,
  Forall (fun e =>
    match e with
    | MSMatch _ j => j >= 1
    | MSMerge2 _ _ j => j >= 1
    | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
    | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
    end) T ->
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_cost A (d :: B') (ms_trace_shift_B 1 T) =
  1 + ms_trace_cost A B' T.
Proof.
  intros A d B' T Hpos Hbound.
  unfold ms_trace_cost, ms_trace_delete_cost, ms_trace_insert_cost.
  (* Change cost is preserved *)
  rewrite (ms_trace_change_cost_shift_B_1 A B' d T Hpos).
  (* Positions lengths preserved under shift *)
  rewrite ms_trace_shift_B_positions_A_length_1.
  rewrite ms_trace_shift_B_positions_B_length_1.
  (* Insert cost: S(len B') - len(positions) = 1 + (len B' - len(positions)) *)
  simpl length. lia.
Qed.

(** * Merge Branch Lemma *)

(** Helper: shift by 2 in A only *)
Definition ms_element_shift_A_2 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i + 2) j
  | MSMerge2 i1 i2 j => MSMerge2 (i1 + 2) (i2 + 2) j
  | MSSplit2 i j1 j2 => MSSplit2 (i + 2) j1 j2
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 + 2) (i2 + 2) j1 j2
  end.

Lemma ms_trace_shift_A_2_map : forall T,
  ms_trace_shift_A 2 T = map ms_element_shift_A_2 T.
Proof.
  intro T. unfold ms_trace_shift_A.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite IH. destruct e; reflexivity.
Qed.

Lemma nth_shift_cons_cons_alt : forall (X : Type) (c d : X) (L : list X) (def : X) (i : nat),
  i >= 1 ->
  nth (i + 2 - 1) (c :: d :: L) def = nth (i - 1) L def.
Proof.
  intros X c d L def i Hi.
  replace (i + 2 - 1) with (S (S (i - 1))) by lia.
  simpl. reflexivity.
Qed.

(** Helper for nth with shift by 2 *)
Lemma nth_plus_2_minus_1 : forall (X : Type) (c1 c2 : X) (A' : list X) (def : X) (i : nat),
  i >= 1 ->
  nth (i + 2 - 1) (c1 :: c2 :: A') def = nth (i - 1) A' def.
Proof.
  intros X c1 c2 A' def i Hi.
  assert (H: i + 2 - 1 = S (S (i - 1))) by lia.
  rewrite H. simpl. reflexivity.
Qed.

(** Element cost preserved when shifting A positions by 2 *)
Lemma ms_element_cost_shift_A_2_full : forall A' B c1 c2 e,
  (match e with
   | MSMatch i _ => i >= 1
   | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
   | MSSplit2 i _ _ => i >= 1
   | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
   end) ->
  ms_element_cost (c1 :: c2 :: A') B (ms_element_shift_A_2 e) = ms_element_cost A' B e.
Proof.
  intros A' B c1 c2 e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl ms_element_shift_A_2; unfold ms_element_cost.
  - (* MSMatch *)
    simpl in Hpos.
    assert (H: i + 2 - 1 = S (S (i - 1))) by lia. rewrite H. simpl. reflexivity.
  - (* MSMerge2 *)
    simpl in Hpos. destruct Hpos as [Hi1 Hi2].
    unfold merge_cost.
    assert (H1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (H2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    rewrite H1, H2. simpl. reflexivity.
  - (* MSSplit2 *)
    simpl in Hpos. unfold split_cost.
    assert (H: i + 2 - 1 = S (S (i - 1))) by lia. rewrite H. simpl. reflexivity.
  - (* MSDouble *)
    simpl in Hpos. destruct Hpos as [Hi1 Hi2].
    assert (H1: i1 + 2 - 1 = S (S (i1 - 1))) by lia.
    assert (H2: i2 + 2 - 1 = S (S (i2 - 1))) by lia.
    rewrite H1, H2. simpl. reflexivity.
Qed.

(** Change cost preserved when shifting A positions by 2 *)
Lemma ms_trace_change_cost_shift_A_2 : forall A' B c1 c2 T,
  Forall (fun e =>
    match e with
    | MSMatch i _ => i >= 1
    | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
    | MSSplit2 i _ _ => i >= 1
    | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
    end) T ->
  ms_trace_change_cost (c1 :: c2 :: A') B (ms_trace_shift_A 2 T) =
  ms_trace_change_cost A' B T.
Proof.
  intros A' B c1 c2 T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_A_2_map.
  induction T as [| e rest IH] using rev_ind; [reflexivity |].
  rewrite map_app. simpl.
  rewrite !fold_left_app. simpl.
  apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
  inversion He_list as [| ? ? He _]; subst; clear He_list.
  specialize (IH Hrest).
  rewrite IH.
  rewrite (ms_element_cost_shift_A_2_full A' B c1 c2 e He).
  reflexivity.
Qed.

(** Positions length preserved when shifting A by 2 *)
Lemma ms_trace_shift_A_positions_A_length_2 : forall T,
  length (ms_trace_positions_A (ms_trace_shift_A 2 T)) = length (ms_trace_positions_A T).
Proof.
  intro T. rewrite ms_trace_shift_A_2_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

Lemma ms_trace_shift_A_positions_B_length_2 : forall T,
  length (ms_trace_positions_B (ms_trace_shift_A 2 T)) = length (ms_trace_positions_B T).
Proof.
  intro T. rewrite ms_trace_shift_A_2_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

(** Merge branch: MSMerge2 1 2 1 + shift A by 2, B by 1

    For the merge branch in the main recursive case, the trace element MSMerge2 1 2 1
    consumes positions 1,2 in A (the characters c1, c2) and position 1 in B (character d),
    representing the merge of c1,c2 -> d.

    The recursive trace T covers the remaining strings (A', B').
    Shifting by (2, 1) adjusts positions appropriately.

    ADMITTED: Full proof requires careful handling of change cost decomposition
    with the shifted trace. The semantic correctness follows from:
    1. MSMerge2 1 2 1 contributes merge_cost c1 c2 d
    2. Shifted T contributes ms_trace_cost A' B' T
    3. Delete cost: (2 + |A'|) - (2 + |pos_A(T)|) = |A'| - |pos_A(T)|
    4. Insert cost: (1 + |B'|) - (1 + |pos_B(T)|) = |B'| - |pos_B(T)|
*)
Lemma ms_trace_cost_cons_merge : forall c1 c2 A' d B' T,
  ms_trace_positions_ge_1 T ->
  length (ms_trace_positions_A T) <= length A' ->
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_cost (c1 :: c2 :: A') (d :: B') (MSMerge2 1 2 1 :: ms_trace_shift_AB 2 1 T) =
  merge_cost c1 c2 d + ms_trace_cost A' B' T.
Proof.
  intros c1 c2 A' d B' T Hpos HlenA HlenB.
  unfold ms_trace_cost.
  (* Change cost for (MSMerge2 1 2 1 :: shifted T) = merge_cost c1 c2 d + change(shifted T)
                                                   = merge_cost c1 c2 d + change(T) *)
  assert (Hchange: ms_trace_change_cost (c1 :: c2 :: A') (d :: B') (MSMerge2 1 2 1 :: ms_trace_shift_AB 2 1 T) =
                   merge_cost c1 c2 d + ms_trace_change_cost A' B' T).
  { unfold ms_trace_change_cost at 1. simpl.
    unfold ms_element_cost at 1. simpl.
    (* Now goal: fold_left ... (shift T) (merge_cost c1 c2 d) = merge_cost c1 c2 d + change_cost A' B' T *)
    rewrite fold_left_add_init.
    f_equal.
    (* Now need: fold_left ... (shift T) 0 = change_cost A' B' T *)
    apply ms_trace_change_cost_shift_AB_2_1. exact Hpos. }
  rewrite Hchange.
  (* Delete and insert costs use position counting *)
  rewrite (ms_trace_delete_cost_cons_merge c1 c2 A' T HlenA).
  rewrite (ms_trace_insert_cost_cons_merge d B' T HlenB).
  lia.
Qed.

(** * Split Branch Lemma *)

(** Helper: shift by 2 in B only *)
Definition ms_element_shift_B_2 (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch i (j + 2)
  | MSMerge2 i1 i2 j => MSMerge2 i1 i2 (j + 2)
  | MSSplit2 i j1 j2 => MSSplit2 i (j1 + 2) (j2 + 2)
  | MSDouble i1 i2 j1 j2 => MSDouble i1 i2 (j1 + 2) (j2 + 2)
  end.

Lemma ms_trace_shift_B_2_map : forall T,
  ms_trace_shift_B 2 T = map ms_element_shift_B_2 T.
Proof.
  intro T. unfold ms_trace_shift_B.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl. rewrite IH. destruct e; reflexivity.
Qed.

(** Element cost preserved when shifting B positions by 2 *)
Lemma ms_element_cost_shift_B_2_full : forall A B' d1 d2 e,
  (match e with
   | MSMatch _ j => j >= 1
   | MSMerge2 _ _ j => j >= 1
   | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
   | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
   end) ->
  ms_element_cost A (d1 :: d2 :: B') (ms_element_shift_B_2 e) = ms_element_cost A B' e.
Proof.
  intros A B' d1 d2 e Hpos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl ms_element_shift_B_2; unfold ms_element_cost.
  - (* MSMatch *)
    simpl in Hpos.
    assert (H: j + 2 - 1 = S (S (j - 1))) by lia. rewrite H. simpl. reflexivity.
  - (* MSMerge2 *)
    simpl in Hpos. unfold merge_cost.
    assert (H: j + 2 - 1 = S (S (j - 1))) by lia. rewrite H. simpl. reflexivity.
  - (* MSSplit2 *)
    simpl in Hpos. destruct Hpos as [Hj1 Hj2]. unfold split_cost.
    assert (H1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (H2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite H1, H2. simpl. reflexivity.
  - (* MSDouble *)
    simpl in Hpos. destruct Hpos as [Hj1 Hj2].
    assert (H1: j1 + 2 - 1 = S (S (j1 - 1))) by lia.
    assert (H2: j2 + 2 - 1 = S (S (j2 - 1))) by lia.
    rewrite H1, H2. simpl. reflexivity.
Qed.

(** Change cost preserved when shifting B positions by 2 *)
Lemma ms_trace_change_cost_shift_B_2 : forall A B' d1 d2 T,
  Forall (fun e =>
    match e with
    | MSMatch _ j => j >= 1
    | MSMerge2 _ _ j => j >= 1
    | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
    | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
    end) T ->
  ms_trace_change_cost A (d1 :: d2 :: B') (ms_trace_shift_B 2 T) =
  ms_trace_change_cost A B' T.
Proof.
  intros A B' d1 d2 T Hpos.
  unfold ms_trace_change_cost.
  rewrite ms_trace_shift_B_2_map.
  induction T as [| e rest IH] using rev_ind; [reflexivity |].
  rewrite map_app. simpl.
  rewrite !fold_left_app. simpl.
  apply Forall_app in Hpos. destruct Hpos as [Hrest He_list].
  inversion He_list as [| ? ? He _]; subst; clear He_list.
  specialize (IH Hrest).
  rewrite IH.
  rewrite (ms_element_cost_shift_B_2_full A B' d1 d2 e He).
  reflexivity.
Qed.

(** Positions length preserved when shifting B by 2 *)
Lemma ms_trace_shift_B_positions_A_length_2 : forall T,
  length (ms_trace_positions_A (ms_trace_shift_B 2 T)) = length (ms_trace_positions_A T).
Proof.
  intro T. rewrite ms_trace_shift_B_2_map.
  unfold ms_trace_positions_A.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

Lemma ms_trace_shift_B_positions_B_length_2 : forall T,
  length (ms_trace_positions_B (ms_trace_shift_B 2 T)) = length (ms_trace_positions_B T).
Proof.
  intro T. rewrite ms_trace_shift_B_2_map.
  unfold ms_trace_positions_B.
  induction T as [| e rest IH]; [reflexivity |].
  simpl. rewrite !length_app. f_equal; [destruct e; reflexivity | exact IH].
Qed.

(** Split branch: MSSplit2 1 1 2 + shift A by 1, B by 2 *)
Lemma ms_trace_cost_cons_split : forall c A' d1 d2 B' T,
  ms_trace_positions_ge_1 T ->
  length (ms_trace_positions_A T) <= length A' ->
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_cost (c :: A') (d1 :: d2 :: B') (MSSplit2 1 1 2 :: ms_trace_shift_AB 1 2 T) =
  split_cost c d1 d2 + ms_trace_cost A' B' T.
Proof.
  intros c A' d1 d2 B' T Hpos HlenA HlenB.
  unfold ms_trace_cost.
  (* Change cost for (MSSplit2 1 1 2 :: shifted T) = split_cost c d1 d2 + change(shifted T)
                                                   = split_cost c d1 d2 + change(T) *)
  assert (Hchange: ms_trace_change_cost (c :: A') (d1 :: d2 :: B') (MSSplit2 1 1 2 :: ms_trace_shift_AB 1 2 T) =
                   split_cost c d1 d2 + ms_trace_change_cost A' B' T).
  { unfold ms_trace_change_cost at 1. simpl.
    unfold ms_element_cost at 1. simpl.
    (* Now goal: fold_left ... (shift T) (split_cost c d1 d2) = split_cost c d1 d2 + change_cost A' B' T *)
    rewrite fold_left_add_init.
    f_equal.
    (* Now need: fold_left ... (shift T) 0 = change_cost A' B' T *)
    apply ms_trace_change_cost_shift_AB_1_2. exact Hpos. }
  rewrite Hchange.
  (* Delete and insert costs use position counting *)
  rewrite (ms_trace_delete_cost_cons_split c A' T HlenA).
  rewrite (ms_trace_insert_cost_cons_split d1 d2 B' T HlenB).
  lia.
Qed.

(** * Double-Subst Branch Lemma *)

(** Double-subst branch: MSDouble 1 2 1 2 + shift A by 2, B by 2 *)
Lemma ms_trace_cost_cons_double : forall c1 c2 A' d1 d2 B' T,
  ms_trace_positions_ge_1 T ->
  length (ms_trace_positions_A T) <= length A' ->
  length (ms_trace_positions_B T) <= length B' ->
  ms_trace_cost (c1 :: c2 :: A') (d1 :: d2 :: B') (MSDouble 1 2 1 2 :: ms_trace_shift_AB 2 2 T) =
  subst_cost c1 d1 + subst_cost c2 d2 + ms_trace_cost A' B' T.
Proof.
  intros c1 c2 A' d1 d2 B' T Hpos HlenA HlenB.
  unfold ms_trace_cost.
  (* Change cost for (MSDouble 1 2 1 2 :: shifted T) = subst_cost c1 d1 + subst_cost c2 d2 + change(shifted T)
                                                     = subst_cost c1 d1 + subst_cost c2 d2 + change(T) *)
  assert (Hchange: ms_trace_change_cost (c1 :: c2 :: A') (d1 :: d2 :: B') (MSDouble 1 2 1 2 :: ms_trace_shift_AB 2 2 T) =
                   subst_cost c1 d1 + subst_cost c2 d2 + ms_trace_change_cost A' B' T).
  { unfold ms_trace_change_cost at 1. simpl.
    unfold ms_element_cost at 1. simpl.
    (* Now goal: fold_left ... (shift T) (subst_cost c1 d1 + subst_cost c2 d2) = ... + change_cost A' B' T *)
    rewrite fold_left_add_init.
    f_equal.
    (* Now need: fold_left ... (shift T) 0 = change_cost A' B' T *)
    apply ms_trace_change_cost_shift_AB_2_2. exact Hpos. }
  rewrite Hchange.
  (* Delete and insert costs use position counting *)
  rewrite (ms_trace_delete_cost_cons_double c1 c2 A' T HlenA).
  rewrite (ms_trace_insert_cost_cons_double d1 d2 B' T HlenB).
  lia.
Qed.

(** * Conversion Lemmas for Position Predicates *)

(** ms_trace_positions_ge_1 implies the A-only Forall variant *)
Lemma ms_positions_ge_1_A_only : forall T,
  ms_trace_positions_ge_1 T ->
  Forall (fun e =>
    match e with
    | MSMatch i _ => i >= 1
    | MSMerge2 i1 i2 _ => i1 >= 1 /\ i2 >= 1
    | MSSplit2 i _ _ => i >= 1
    | MSDouble i1 i2 _ _ => i1 >= 1 /\ i2 >= 1
    end) T.
Proof.
  intros T H.
  unfold ms_trace_positions_ge_1 in H.
  induction T as [| e rest IH].
  - constructor.
  - inversion H; subst.
    constructor.
    + destruct e; simpl in *; tauto.
    + apply IH. assumption.
Qed.

(** ms_trace_positions_ge_1 implies the B-only Forall variant *)
Lemma ms_positions_ge_1_B_only : forall T,
  ms_trace_positions_ge_1 T ->
  Forall (fun e =>
    match e with
    | MSMatch _ j => j >= 1
    | MSMerge2 _ _ j => j >= 1
    | MSSplit2 _ j1 j2 => j1 >= 1 /\ j2 >= 1
    | MSDouble _ _ j1 j2 => j1 >= 1 /\ j2 >= 1
    end) T.
Proof.
  intros T H.
  unfold ms_trace_positions_ge_1 in H.
  induction T as [| e rest IH].
  - constructor.
  - inversion H; subst.
    constructor.
    + destruct e; simpl in *; tauto.
    + apply IH. assumption.
Qed.

(** * Position Bounds for Optimal Trace *)

(** The number of A-positions in optimal trace is bounded by |A| *)
Lemma ms_optimal_trace_pair_positions_A_bound : forall p,
  length (ms_trace_positions_A (ms_optimal_trace_pair p)) <= length (fst p).
Proof.
  intro p.
  remember (ms_optimal_trace_measure p) as n eqn:Hn.
  revert p Hn.
  induction n as [n IH] using lt_wf_ind.
  intros [A B] Hn.
  rewrite ms_optimal_trace_pair_equation.
  destruct A as [| c1 [| c2 s1']];
  destruct B as [| d1 [| d2 s2']];
  try (simpl; lia).
  - (* [c1], d1::d2::s2' *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch; simpl.
    + (* delete branch - results in [] *)
      unfold ms_trace_positions_A. simpl. lia.
    + (* delete branch *)
      unfold ms_trace_positions_A. simpl. lia.
    + (* insert branch *)
      rewrite ms_trace_shift_B_positions_A_length_1.
      specialize (IH (ms_optimal_trace_measure ([c1], d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure ([c1], d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt ([c1], d2 :: s2') eq_refl). simpl in IH. lia.
    + (* subst branch *)
      unfold ms_trace_positions_A, ms_element_positions_A. simpl. lia.
    + (* split branch *)
      unfold ms_trace_positions_A, ms_element_positions_A. simpl. lia.
  - (* c1::c2::s1', [d1] *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch; simpl.
    + (* merge branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_A, ms_element_positions_A. simpl. lia.
    + (* delete branch *)
      rewrite ms_trace_shift_A_positions_A_length_1.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', [d1]))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', [d1]) < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', [d1]) eq_refl). simpl in IH. lia.
    + (* insert branch - already simplified to 0 <= ... *)
      lia.
    + (* subst branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_A, ms_element_positions_A. simpl. lia.
    + (* merge branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_A, ms_element_positions_A. simpl. lia.
  - (* c1::c2::s1', d1::d2::s2' - main case *)
    cbv zeta.
    destruct (min6_branch _ _ _ _ _ _) as [|[|[|[|[|[|]]]]]] eqn:Ebranch; simpl.
    + (* double-subst branch *)
      unfold ms_trace_positions_A at 1, ms_element_positions_A. simpl.
      rewrite ms_trace_shift_AB_2_2_positions_A_length.
      specialize (IH (ms_optimal_trace_measure (s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', s2') eq_refl). simpl in IH. lia.
    + (* delete branch *)
      rewrite ms_trace_shift_A_positions_A_length_1.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', d1 :: d2 :: s2') eq_refl). simpl in IH. lia.
    + (* insert branch *)
      rewrite ms_trace_shift_B_positions_A_length_1.
      specialize (IH (ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c1 :: c2 :: s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* subst branch *)
      unfold ms_trace_positions_A at 1, ms_element_positions_A. simpl.
      rewrite ms_trace_shift_AB_1_1_positions_A_length.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* merge branch *)
      unfold ms_trace_positions_A at 1, ms_element_positions_A. simpl.
      rewrite ms_trace_shift_AB_2_1_positions_A_length.
      specialize (IH (ms_optimal_trace_measure (s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* split branch *)
      unfold ms_trace_positions_A at 1, ms_element_positions_A. simpl.
      rewrite ms_trace_shift_AB_1_2_positions_A_length.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', s2') eq_refl). simpl in IH. lia.
    + (* double-subst branch (default) *)
      unfold ms_trace_positions_A at 1, ms_element_positions_A. simpl.
      rewrite ms_trace_shift_AB_2_2_positions_A_length.
      specialize (IH (ms_optimal_trace_measure (s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', s2') eq_refl). simpl in IH. lia.
Qed.

(** The number of B-positions in optimal trace is bounded by |B| *)
Lemma ms_optimal_trace_pair_positions_B_bound : forall p,
  length (ms_trace_positions_B (ms_optimal_trace_pair p)) <= length (snd p).
Proof.
  intro p.
  remember (ms_optimal_trace_measure p) as n eqn:Hn.
  revert p Hn.
  induction n as [n IH] using lt_wf_ind.
  intros [A B] Hn.
  rewrite ms_optimal_trace_pair_equation.
  destruct A as [| c1 [| c2 s1']];
  destruct B as [| d1 [| d2 s2']];
  try (simpl; lia).
  - (* [c1], d1::d2::s2' *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch; simpl.
    + (* delete branch - results in [] *)
      unfold ms_trace_positions_B. simpl. lia.
    + (* delete branch *)
      unfold ms_trace_positions_B. simpl. lia.
    + (* insert branch *)
      rewrite ms_trace_shift_B_positions_B_length_1.
      specialize (IH (ms_optimal_trace_measure ([c1], d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure ([c1], d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt ([c1], d2 :: s2') eq_refl). simpl in IH. lia.
    + (* subst branch *)
      unfold ms_trace_positions_B, ms_element_positions_B. simpl. lia.
    + (* split branch *)
      unfold ms_trace_positions_B, ms_element_positions_B. simpl. lia.
  - (* c1::c2::s1', [d1] *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch; simpl.
    + (* merge branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_B, ms_element_positions_B. simpl. lia.
    + (* delete branch *)
      rewrite ms_trace_shift_A_positions_B_length_1.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', [d1]))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', [d1]) < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', [d1]) eq_refl). simpl in IH. lia.
    + (* insert branch - already simplified to 0 <= ... *)
      lia.
    + (* subst branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_B, ms_element_positions_B. simpl. lia.
    + (* merge branch - empty recursive *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_positions_B, ms_element_positions_B. simpl. lia.
  - (* c1::c2::s1', d1::d2::s2' - main case *)
    cbv zeta.
    destruct (min6_branch _ _ _ _ _ _) as [|[|[|[|[|[|]]]]]] eqn:Ebranch; simpl.
    + (* double-subst branch *)
      unfold ms_trace_positions_B at 1, ms_element_positions_B. simpl.
      rewrite ms_trace_shift_AB_2_2_positions_B_length.
      specialize (IH (ms_optimal_trace_measure (s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', s2') eq_refl). simpl in IH. lia.
    + (* delete branch *)
      rewrite ms_trace_shift_A_positions_B_length_1.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', d1 :: d2 :: s2') eq_refl). simpl in IH. lia.
    + (* insert branch *)
      rewrite ms_trace_shift_B_positions_B_length_1.
      specialize (IH (ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c1 :: c2 :: s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* subst branch *)
      unfold ms_trace_positions_B at 1, ms_element_positions_B. simpl.
      rewrite ms_trace_shift_AB_1_1_positions_B_length.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* merge branch *)
      unfold ms_trace_positions_B at 1, ms_element_positions_B. simpl.
      rewrite ms_trace_shift_AB_2_1_positions_B_length.
      specialize (IH (ms_optimal_trace_measure (s1', d2 :: s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', d2 :: s2') eq_refl). simpl in IH. lia.
    + (* split branch *)
      unfold ms_trace_positions_B at 1, ms_element_positions_B. simpl.
      rewrite ms_trace_shift_AB_1_2_positions_B_length.
      specialize (IH (ms_optimal_trace_measure (c2 :: s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (c2 :: s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (c2 :: s1', s2') eq_refl). simpl in IH. lia.
    + (* double-subst branch (default) *)
      unfold ms_trace_positions_B at 1, ms_element_positions_B. simpl.
      rewrite ms_trace_shift_AB_2_2_positions_B_length.
      specialize (IH (ms_optimal_trace_measure (s1', s2'))).
      assert (Hlt: ms_optimal_trace_measure (s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
      specialize (IH Hlt (s1', s2') eq_refl). simpl in IH. lia.
Qed.

(** * Cost Equality Theorem *)

(** The optimal trace has cost equal to merge_split_distance.
    This is the key theorem for the triangle inequality proof.

    Proof strategy: Strong induction on |A| + |B|, matching the cases
    of ms_optimal_trace_pair with the cases of merge_split_pair.
    At each step, the constructed trace element matches the winning
    branch of the min6/min4/min3 computation.
*)
Theorem ms_optimal_trace_cost_eq : forall (A B : list Char),
  ms_trace_cost A B (ms_optimal_trace A B) = merge_split_distance A B.
Proof.
  intros A B.
  unfold ms_optimal_trace, merge_split_distance.
  remember (ms_optimal_trace_measure (A, B)) as n eqn:Hn.
  revert A B Hn.
  induction n as [n IH] using lt_wf_ind.
  intros A B Hn.
  rewrite ms_optimal_trace_pair_equation, merge_split_pair_equation.
  destruct A as [| c1 [| c2 s1']];
  destruct B as [| d1 [| d2 s2']].
  - (* [], [] *)
    rewrite ms_trace_cost_nil. simpl. lia.
  - (* [], [d1] *)
    rewrite ms_trace_cost_nil. simpl. lia.
  - (* [], d1::d2::s2' *)
    rewrite ms_trace_cost_nil. simpl. lia.
  - (* [c1], [] *)
    rewrite ms_trace_cost_nil. simpl. lia.
  - (* [c1], [d1] *)
    rewrite ms_trace_cost_single_match. unfold subst_cost. reflexivity.
  - (* [c1], d1::d2::s2' *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch.
    + (* 0: impossible - min4_branch >= 1 *)
      exfalso. exact (min4_branch_not_0 _ _ _ _ Ebranch).
    + (* 1: delete branch - empty source after delete *)
      rewrite ms_optimal_trace_pair_nil_l.
      rewrite ms_trace_shift_B_nil.
      simpl ms_optimal_trace_pair.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      unfold ms_trace_positions_A, ms_trace_positions_B. cbn [flat_map map fold_left length abs_diff].
      (* LHS is now 1 + S (S (length s2')), RHS is min (min3 ...) *)
      (* Use min4_branch_eq_1_implies to simplify the min(min3 ...) expression *)
      pose proof (min4_branch_eq_1_implies _ _ _ _ Ebranch) as Hmin.
      rewrite Hmin.
      (* Now RHS = merge_split_pair ([], d1::d2::s2') + 1 *)
      rewrite merge_split_pair_nil_l. simpl. lia.
    + (* 2: insert branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair ([c1], d2 :: s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_shift_B_insert.
      * (* Main goal *)
        specialize (IH (ms_optimal_trace_measure ([c1], d2 :: s2'))).
        assert (Hlt: ms_optimal_trace_measure ([c1], d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt ([c1]) (d2 :: s2') eq_refl).
        rewrite IH.
        (* Need: 1 + merge_split_pair ([c1], d2::s2') = merge_split_pair ([c1], d1::d2::s2') *)
        (* RHS is already expanded from line 2172; use Hmin to simplify *)
        pose proof (min4_branch_eq_2_implies _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin. (* Now: 1 + x = x + 1 *) lia.
      * apply ms_positions_ge_1_B_only. assumption.
      * pose proof (ms_optimal_trace_pair_positions_B_bound ([c1], d2 :: s2')) as Hbound.
        simpl in Hbound. simpl. exact Hbound.
    + (* 3: subst branch *)
      rewrite ms_optimal_trace_pair_nil_l.
      rewrite ms_trace_shift_B_nil.
      simpl ms_optimal_trace_pair.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      unfold ms_trace_positions_A, ms_trace_positions_B.
      unfold ms_element_positions_A, ms_element_positions_B.
      unfold ms_element_cost. cbn [flat_map map fold_left length abs_diff].
      (* LHS simplified, RHS is min (min3 ...) *)
      pose proof (min4_branch_eq_3_implies _ _ _ _ Ebranch) as Hmin.
      rewrite Hmin.
      (* Now RHS = subst_cost c1 d1 + merge_split_pair ([], d2::s2') *)
      rewrite merge_split_pair_nil_l. simpl. lia.
    + (* 4+: split branch (default) *)
      rewrite ms_optimal_trace_pair_nil_l.
      rewrite ms_trace_shift_B_nil.
      simpl ms_optimal_trace_pair.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      unfold ms_trace_positions_A, ms_trace_positions_B.
      unfold ms_element_positions_A, ms_element_positions_B.
      unfold ms_element_cost. cbn [flat_map map fold_left length abs_diff].
      (* LHS simplified, RHS is min (min3 ...) *)
      (* Ebranch matches S(S(S(S _))), so min4_branch >= 4. Since max is 4, it equals 4. *)
      pose proof (min4_branch_ge_4_eq_4 _ _ _ _ _ Ebranch) as Hbranch4.
      pose proof (min4_branch_eq_4_implies _ _ _ _ Hbranch4) as Hmin.
      rewrite Hmin.
      (* Now RHS = split_cost c1 d1 d2 + merge_split_pair ([], s2') *)
      rewrite merge_split_pair_nil_l. simpl. lia.
  - (* c1::c2::s1', [] *)
    rewrite ms_trace_cost_nil. simpl. lia.
  - (* c1::c2::s1', [d1] *)
    cbv zeta.
    destruct (min4_branch _ _ _ _) as [|[|[|[|]]]] eqn:Ebranch.
    + (* 0: impossible - min4_branch >= 1 *)
      exfalso. exact (min4_branch_not_0 _ _ _ _ Ebranch).
    + (* 1: delete branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (c2 :: s1', [d1])))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_shift_A_delete.
      * (* Main goal *)
        specialize (IH (ms_optimal_trace_measure (c2 :: s1', [d1]))).
        assert (Hlt: ms_optimal_trace_measure (c2 :: s1', [d1]) < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt (c2 :: s1') [d1] eq_refl).
        rewrite IH.
        pose proof (min4_branch_eq_1_implies _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        (* Branch 1 = delete, goal: 1 + merge_split_pair(c2::s1', [d1]) = 1 + merge_split_pair(c2::s1', [d1]) *)
        lia.
      * apply ms_positions_ge_1_A_only. assumption.
      * pose proof (ms_optimal_trace_pair_positions_A_bound (c2 :: s1', [d1])) as Hbound.
        simpl in Hbound. exact Hbound.
    + (* 2: insert branch - B goes to empty *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      rewrite ms_trace_cost_nil.
      pose proof (min4_branch_eq_2_implies _ _ _ _ Ebranch) as Hmin.
      rewrite Hmin.
      rewrite merge_split_pair_nil_r. simpl. lia.
    + (* 3: subst branch *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      unfold ms_trace_positions_A, ms_trace_positions_B.
      unfold ms_element_positions_A, ms_element_positions_B.
      unfold ms_element_cost. cbn [flat_map map fold_left length abs_diff].
      pose proof (min4_branch_eq_3_implies _ _ _ _ Ebranch) as Hmin.
      rewrite Hmin.
      rewrite merge_split_pair_nil_r. simpl. lia.
    + (* 4+: merge branch (default) *)
      rewrite ms_optimal_trace_pair_nil_r, ms_trace_shift_A_nil.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      unfold ms_trace_positions_A, ms_trace_positions_B.
      unfold ms_element_positions_A, ms_element_positions_B.
      unfold ms_element_cost. cbn [flat_map map fold_left length abs_diff].
      pose proof (min4_branch_ge_4_eq_4 _ _ _ _ _ Ebranch) as Hbranch4.
      pose proof (min4_branch_eq_4_implies _ _ _ _ Hbranch4) as Hmin.
      rewrite Hmin.
      rewrite merge_split_pair_nil_r. simpl. lia.
  - (* c1::c2::s1', d1::d2::s2' - main case *)
    cbv zeta.
    destruct (min6_branch _ _ _ _ _ _) as [|[|[|[|[|[|]]]]]] eqn:Ebranch.
    + (* 0: impossible - min6_branch >= 1 *)
      exfalso. exact (min6_branch_not_0 _ _ _ _ _ _ Ebranch).
    + (* 1: delete branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (c2 :: s1', d1 :: d2 :: s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_shift_A_delete.
      * specialize (IH (ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2'))).
        assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d1 :: d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt (c2 :: s1') (d1 :: d2 :: s2') eq_refl).
        rewrite IH.
        pose proof (min6_branch_eq_1_implies _ _ _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        lia.
      * apply ms_positions_ge_1_A_only. assumption.
      * pose proof (ms_optimal_trace_pair_positions_A_bound (c2 :: s1', d1 :: d2 :: s2')) as Hbound.
        simpl in Hbound. exact Hbound.
    + (* 2: insert branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (c1 :: c2 :: s1', d2 :: s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_shift_B_insert.
      * specialize (IH (ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2'))).
        assert (Hlt: ms_optimal_trace_measure (c1 :: c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt (c1 :: c2 :: s1') (d2 :: s2') eq_refl).
        rewrite IH.
        pose proof (min6_branch_eq_2_implies _ _ _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        lia.
      * apply ms_positions_ge_1_B_only. assumption.
      * pose proof (ms_optimal_trace_pair_positions_B_bound (c1 :: c2 :: s1', d2 :: s2')) as Hbound.
        simpl in Hbound. exact Hbound.
    + (* 3: subst branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (c2 :: s1', d2 :: s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_cons_match.
      * specialize (IH (ms_optimal_trace_measure (c2 :: s1', d2 :: s2'))).
        assert (Hlt: ms_optimal_trace_measure (c2 :: s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt (c2 :: s1') (d2 :: s2') eq_refl).
        rewrite IH.
        pose proof (min6_branch_eq_3_implies _ _ _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        lia.
      * assumption.
    + (* 4: merge branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (s1', d2 :: s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_cons_merge.
      * specialize (IH (ms_optimal_trace_measure (s1', d2 :: s2'))).
        assert (Hlt: ms_optimal_trace_measure (s1', d2 :: s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt s1' (d2 :: s2') eq_refl).
        rewrite IH.
        pose proof (min6_branch_eq_4_implies _ _ _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        lia.
      * assumption.
      * pose proof (ms_optimal_trace_pair_positions_A_bound (s1', d2 :: s2')) as Hbound. simpl in Hbound. exact Hbound.
      * pose proof (ms_optimal_trace_pair_positions_B_bound (s1', d2 :: s2')) as Hbound. simpl in Hbound. exact Hbound.
    + (* 5: split branch *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (c2 :: s1', s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_cons_split.
      * specialize (IH (ms_optimal_trace_measure (c2 :: s1', s2'))).
        assert (Hlt: ms_optimal_trace_measure (c2 :: s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt (c2 :: s1') s2' eq_refl).
        rewrite IH.
        pose proof (min6_branch_eq_5_implies _ _ _ _ _ _ Ebranch) as Hmin.
        rewrite Hmin.
        lia.
      * assumption.
      * pose proof (ms_optimal_trace_pair_positions_A_bound (c2 :: s1', s2')) as Hbound. simpl in Hbound. exact Hbound.
      * pose proof (ms_optimal_trace_pair_positions_B_bound (c2 :: s1', s2')) as Hbound. simpl in Hbound. exact Hbound.
    + (* 6+: double-subst branch (default) *)
      assert (Hpos: ms_trace_positions_ge_1 (ms_optimal_trace_pair (s1', s2')))
        by apply ms_optimal_trace_pair_positions_ge1.
      rewrite ms_trace_cost_cons_double.
      * specialize (IH (ms_optimal_trace_measure (s1', s2'))).
        assert (Hlt: ms_optimal_trace_measure (s1', s2') < n) by (unfold ms_optimal_trace_measure in *; simpl in *; lia).
        specialize (IH Hlt s1' s2' eq_refl).
        rewrite IH.
        pose proof (min6_branch_ge_6_eq_6 _ _ _ _ _ _ _ Ebranch) as Hbranch6.
        pose proof (min6_branch_eq_6_implies _ _ _ _ _ _ Hbranch6) as Hmin.
        rewrite Hmin.
        lia.
      * assumption.
      * pose proof (ms_optimal_trace_pair_positions_A_bound (s1', s2')) as Hbound. simpl in Hbound. exact Hbound.
      * pose proof (ms_optimal_trace_pair_positions_B_bound (s1', s2')) as Hbound. simpl in Hbound. exact Hbound.
Qed.

(** Combined existence theorem with cost equality *)
Theorem ms_optimal_trace_exists : forall (A B : list Char),
  exists T : MSTrace,
    ms_trace_positions_ge_1 T /\
    ms_trace_cost A B T = merge_split_distance A B.
Proof.
  intros A B.
  exists (ms_optimal_trace A B).
  split.
  - apply ms_optimal_trace_pair_positions_ge1.
  - apply ms_optimal_trace_cost_eq.
Qed.
