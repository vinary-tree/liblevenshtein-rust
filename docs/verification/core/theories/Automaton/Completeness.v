(** * Completeness of Levenshtein Automata

    This module proves that if the actual Levenshtein distance is <= n,
    then the automaton accepts. This is the "completeness" direction of correctness.

    Part of: Liblevenshtein.Core.Automaton

    Key Theorem:
    - automaton_complete: lev_distance(query, dict) <= n -> accepts(query, n, dict)

    Proof Strategy:
    - If lev_distance(query, dict) = d <= n, there exists an optimal edit sequence
    - This edit sequence corresponds to a path through the automaton
    - The automaton explores all paths up to cost n
    - Therefore the optimal path will be found and the automaton accepts
*)

From Stdlib Require Import Arith Bool List Nat Lia Ascii Program.Equality.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.
From Liblevenshtein.Core Require Import Automaton.Position.
From Liblevenshtein.Core Require Import Automaton.CharVector.
From Liblevenshtein.Core Require Import Automaton.State.
From Liblevenshtein.Core Require Import Automaton.Transition.
From Liblevenshtein.Core Require Import Automaton.Acceptance.
From Liblevenshtein.Core Require Import Automaton.Soundness.
From Liblevenshtein.Core Require Import Automaton.AntiChain.
From Liblevenshtein.Core Require Import Automaton.Subsumption.
From Liblevenshtein.Core Require Import Triangle.TriangleInequality.
From Liblevenshtein.Core Require Import Core.MetricProperties.

(** * Edit Operations *)

(** An edit operation transforms one string pair state to another *)
Inductive edit_op : Type :=
  | Edit_Match    : Char -> edit_op   (* Match: consume both, cost 0 *)
  | Edit_Substitute : Char -> Char -> edit_op  (* Substitute: consume both, cost 1 *)
  | Edit_Delete   : Char -> edit_op   (* Delete from query: consume query only, cost 1 *)
  | Edit_Insert   : Char -> edit_op.  (* Insert from dict: consume dict only, cost 1 *)

(** Cost of an edit operation *)
Definition edit_cost (op : edit_op) : nat :=
  match op with
  | Edit_Match _ => 0
  | Edit_Substitute _ _ => 1
  | Edit_Delete _ => 1
  | Edit_Insert _ => 1
  end.

(** An edit sequence is a list of edit operations *)
Definition edit_sequence := list edit_op.

(** Total cost of an edit sequence *)
Definition sequence_cost (ops : edit_sequence) : nat :=
  fold_right (fun op acc => edit_cost op + acc) 0 ops.

(** * Edit Sequence Validity *)

(** Apply an edit operation to string pair indices *)
Definition apply_edit_op (op : edit_op) (qi di : nat) : (nat * nat) :=
  match op with
  | Edit_Match _ => (S qi, S di)
  | Edit_Substitute _ _ => (S qi, S di)
  | Edit_Delete _ => (S qi, di)
  | Edit_Insert _ => (qi, S di)
  end.

(** Apply a sequence of edit operations *)
Fixpoint apply_edit_sequence (ops : edit_sequence) (qi di : nat) : (nat * nat) :=
  match ops with
  | [] => (qi, di)
  | op :: rest =>
      let (qi', di') := apply_edit_op op qi di in
      apply_edit_sequence rest qi' di'
  end.

(** An edit sequence is valid for (query, dict) if:
    1. It starts at (0, 0)
    2. Each operation is consistent with the characters at current positions
    3. It ends at (|query|, |dict|) *)

(** Check if an edit operation is valid at given position *)
Definition valid_edit_op_at (query dict : list Char) (qi di : nat) (op : edit_op) : Prop :=
  match op with
  | Edit_Match c =>
      nth_error query qi = Some c /\ nth_error dict di = Some c
  | Edit_Substitute c1 c2 =>
      nth_error query qi = Some c1 /\ nth_error dict di = Some c2 /\ c1 <> c2
  | Edit_Delete c =>
      nth_error query qi = Some c
  | Edit_Insert c =>
      nth_error dict di = Some c
  end.

(** An edit sequence is valid for transforming query to dict *)
Inductive valid_edit_sequence : list Char -> list Char -> nat -> nat -> edit_sequence -> Prop :=
  | valid_empty : forall query dict,
      valid_edit_sequence query dict (length query) (length dict) []
  | valid_cons : forall query dict qi di qi' di' op ops,
      valid_edit_op_at query dict qi di op ->
      apply_edit_op op qi di = (qi', di') ->
      valid_edit_sequence query dict qi' di' ops ->
      valid_edit_sequence query dict qi di (op :: ops).

(** * Helper Lemma: Edit Distance Bounds *)

(** min3 is less than or equal to its first argument *)
Lemma min3_le_first : forall a b c, min3 a b c <= a.
Proof.
  intros a b c. unfold min3. lia.
Qed.

(** lev(c::query, dict) <= lev(query, dict) + 1 (delete c) *)
Lemma lev_distance_cons_query_le : forall c query dict,
  lev_distance (c :: query) dict <= lev_distance query dict + 1.
Proof.
  intros c query dict.
  destruct dict as [| d dict'].
  - do 2 rewrite lev_distance_empty_right. simpl.
    destruct (length query); lia.
  - rewrite lev_distance_cons.
    (* min3(del, ins, subst) where del = lev(query, d::dict') + 1 *)
    apply Nat.le_trans with (lev_distance query (d :: dict') + 1).
    + apply min3_le_first.
    + lia.
Qed.

(** Helper: inserting one char costs at most 1 *)
(** lev(s, c::s) <= 1 - inserting c at the front costs 1 *)
Lemma lev_distance_insert_single : forall c s,
  lev_distance s (c :: s) <= 1.
Proof.
  intros c s.
  destruct s as [| a s'].
  - (* s = [] *)
    rewrite lev_distance_empty_left. simpl. lia.
  - (* s = a :: s' *)
    rewrite lev_distance_cons.
    (* min3(delete a + 1, insert c + 1, subst a c)
       where insert c means lev_distance (a::s') (a::s') = 0, cost = 0 + 1 = 1 *)
    unfold min3.
    assert (H : lev_distance (a :: s') (a :: s') = 0).
    { apply lev_distance_identity. }
    lia.
Qed.

(** Helper: deleting one char costs at most 1 *)
(** lev(c::s, s) <= 1 - deleting c at the front costs 1 *)
Lemma lev_distance_delete_single : forall c s,
  lev_distance (c :: s) s <= 1.
Proof.
  intros c s.
  destruct s as [| a s'].
  - (* s = [] *)
    rewrite lev_distance_empty_right. simpl. lia.
  - (* s = a :: s' *)
    rewrite lev_distance_cons.
    (* delete c leaves s, then lev(s, a::s') where s = a::s', so lev_distance s s = 0 *)
    unfold min3.
    assert (H : lev_distance (a :: s') (a :: s') = 0).
    { apply lev_distance_identity. }
    lia.
Qed.

(** Adding a character to the query increases distance by at most 1 *)
(** lev(query, dict) <= lev(c::query, dict) + 1 *)
(** This is a key lemma: dropping a char from query can increase distance by at most 1 *)
Lemma lev_distance_drop_query_char : forall c query dict,
  lev_distance query dict <= lev_distance (c :: query) dict + 1.
Proof.
  intros c query dict.
  (* Use triangle inequality: lev(A,C) <= lev(A,B) + lev(B,C)
     with A = query, B = c :: query, C = dict *)
  pose proof (lev_distance_triangle_inequality query (c :: query) dict) as Htri.
  (* Htri: lev query dict <= lev query (c::query) + lev (c::query) dict *)
  pose proof (lev_distance_insert_single c query) as Hins.
  (* Hins: lev query (c::query) <= 1 *)
  lia.
Qed.

(** Adding a character to the dict increases distance by at most 1 *)
(** lev(query, dict) <= lev(query, c::dict) + 1 *)
(** This is a key lemma: dropping a char from dict can increase distance by at most 1 *)
Lemma lev_distance_drop_dict_char : forall c query dict,
  lev_distance query dict <= lev_distance query (c :: dict) + 1.
Proof.
  intros c query dict.
  (* Use triangle inequality: lev(A,C) <= lev(A,B) + lev(B,C)
     with A = query, B = query, C = dict vs c::dict
     Actually, we need: lev(query, dict) <= lev(query, c::dict) + 1
     Triangle with A = query, B = c::dict, C = dict:
       lev(query, dict) <= lev(query, c::dict) + lev(c::dict, dict) *)
  pose proof (lev_distance_triangle_inequality query (c :: dict) dict) as Htri.
  (* Htri: lev query dict <= lev query (c::dict) + lev (c::dict) dict *)
  pose proof (lev_distance_delete_single c dict) as Hdel.
  (* Hdel: lev (c::dict) dict <= 1 *)
  lia.
Qed.

(** * Index Shifting Lemmas *)

(** Shifting edit operation validity when prepending to query *)
Lemma valid_edit_op_shift_query : forall c query dict qi di op,
  valid_edit_op_at query dict qi di op ->
  valid_edit_op_at (c :: query) dict (S qi) di op.
Proof.
  intros c query dict qi di op Hvalid.
  destruct op; simpl in *.
  - (* Edit_Match *)
    destruct Hvalid as [Hq Hd].
    split; [| exact Hd].
    simpl. exact Hq.
  - (* Edit_Substitute *)
    destruct Hvalid as [Hq [Hd Hne]].
    split; [| split; [exact Hd | exact Hne]].
    simpl. exact Hq.
  - (* Edit_Delete *)
    simpl. exact Hvalid.
  - (* Edit_Insert *)
    exact Hvalid.
Qed.

(** Shifting edit operation validity when prepending to dict *)
Lemma valid_edit_op_shift_dict : forall c query dict qi di op,
  valid_edit_op_at query dict qi di op ->
  valid_edit_op_at query (c :: dict) qi (S di) op.
Proof.
  intros c query dict qi di op Hvalid.
  destruct op; simpl in *.
  - (* Edit_Match *)
    destruct Hvalid as [Hq Hd].
    split; [exact Hq |].
    simpl. exact Hd.
  - (* Edit_Substitute *)
    destruct Hvalid as [Hq [Hd Hne]].
    split; [exact Hq | split; [| exact Hne]].
    simpl. exact Hd.
  - (* Edit_Delete *)
    exact Hvalid.
  - (* Edit_Insert *)
    simpl. exact Hvalid.
Qed.

(** Shifting a valid edit sequence when prepending to query *)
Lemma valid_sequence_shift_query : forall c query dict qi di ops,
  valid_edit_sequence query dict qi di ops ->
  valid_edit_sequence (c :: query) dict (S qi) di ops.
Proof.
  intros c query dict qi di ops Hvalid.
  induction Hvalid as [query' dict' | query' dict' qi' di' qi'' di'' op ops' Hop_valid Happly Hrest IH].
  - (* Empty sequence: ends at (length query, length dict) *)
    (* Need to end at (S (length query), length dict) = (length (c :: query), length dict) *)
    simpl. constructor.
  - (* Cons case *)
    apply valid_cons with (qi' := S qi'') (di' := di'').
    + apply valid_edit_op_shift_query. exact Hop_valid.
    + destruct op; simpl in Happly; inversion Happly; simpl; reflexivity.
    + exact IH.
Qed.

(** Shifting a valid edit sequence when prepending to dict *)
Lemma valid_sequence_shift_dict : forall c query dict qi di ops,
  valid_edit_sequence query dict qi di ops ->
  valid_edit_sequence query (c :: dict) qi (S di) ops.
Proof.
  intros c query dict qi di ops Hvalid.
  induction Hvalid as [query' dict' | query' dict' qi' di' qi'' di'' op ops' Hop_valid Happly Hrest IH].
  - (* Empty sequence: ends at (length query, length dict) *)
    (* Need to end at (length query, S (length dict)) = (length query, length (c :: dict)) *)
    simpl. constructor.
  - (* Cons case *)
    apply valid_cons with (qi' := qi'') (di' := S di'').
    + apply valid_edit_op_shift_dict. exact Hop_valid.
    + destruct op; simpl in Happly; inversion Happly; simpl; reflexivity.
    + exact IH.
Qed.

(** * Optimal Edit Sequences *)

(** There exists an optimal edit sequence achieving the Levenshtein distance *)
Lemma optimal_sequence_exists : forall query dict,
  exists ops,
    valid_edit_sequence query dict 0 0 ops /\
    sequence_cost ops = lev_distance query dict.
Proof.
  (* Strong induction on |query| + |dict| *)
  intros query dict.
  remember (length query + length dict) as n eqn:Hlen.
  revert query dict Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros query dict Hlen.
  destruct query as [| c1 query'].
  - (* query = [] *)
    (* All inserts: Edit_Insert for each dict char *)
    exists (map Edit_Insert dict).
    split.
    + (* valid_edit_sequence [] dict 0 0 (map Edit_Insert dict) *)
      clear IH Hlen n.
      induction dict as [| d dict' IHd'].
      * simpl. constructor.
      * simpl.
        apply valid_cons with (qi' := 0) (di' := 1).
        -- unfold valid_edit_op_at. simpl. reflexivity.
        -- simpl. reflexivity.
        -- (* Use IH shifted: [] dict' 0 0 -> [] (d::dict') 0 1 *)
           apply valid_sequence_shift_dict. exact IHd'.
    + (* sequence_cost = lev_distance [] dict = length dict *)
      rewrite lev_distance_empty_left.
      clear IH Hlen n.
      induction dict as [| d dict' IHd].
      * simpl. reflexivity.
      * simpl. rewrite IHd. reflexivity.
  - destruct dict as [| c2 dict'].
    + (* query = c1::query', dict = [] *)
      (* All deletes: Edit_Delete for each query char *)
      exists (map Edit_Delete (c1 :: query')).
      split.
      * (* valid_edit_sequence *)
        clear IH Hlen n.
        (* First prove the general case: for any query, map Edit_Delete is valid *)
        assert (Hgen : forall q, valid_edit_sequence q [] 0 0 (map Edit_Delete q)).
        { induction q as [| qc q' IHq'].
          - simpl. constructor.
          - simpl.
            apply valid_cons with (qi' := 1) (di' := 0).
            + unfold valid_edit_op_at. simpl. reflexivity.
            + simpl. reflexivity.
            + apply valid_sequence_shift_query. exact IHq'. }
        apply Hgen.
      * (* sequence_cost = lev_distance (c1::query') [] = length (c1::query') *)
        rewrite lev_distance_empty_right.
        simpl.
        clear IH Hlen n.
        induction query' as [| c' query'' IHq'].
        -- simpl. reflexivity.
        -- simpl. rewrite IHq'. reflexivity.
    + (* query = c1::query', dict = c2::dict' - main inductive case *)
      (* Get IH for the three recursive calls *)
      assert (IH_del : exists ops_del,
                valid_edit_sequence query' (c2 :: dict') 0 0 ops_del /\
                sequence_cost ops_del = lev_distance query' (c2 :: dict')).
      { apply IH with (m := length query' + length (c2 :: dict')).
        - simpl in Hlen. simpl. lia.
        - reflexivity. }
      assert (IH_ins : exists ops_ins,
                valid_edit_sequence (c1 :: query') dict' 0 0 ops_ins /\
                sequence_cost ops_ins = lev_distance (c1 :: query') dict').
      { apply IH with (m := length (c1 :: query') + length dict').
        - simpl in Hlen. simpl. lia.
        - reflexivity. }
      assert (IH_sub : exists ops_sub,
                valid_edit_sequence query' dict' 0 0 ops_sub /\
                sequence_cost ops_sub = lev_distance query' dict').
      { apply IH with (m := length query' + length dict').
        - simpl in Hlen. simpl. lia.
        - reflexivity. }
      destruct IH_del as [ops_del [Hvalid_del Hcost_del]].
      destruct IH_ins as [ops_ins [Hvalid_ins Hcost_ins]].
      destruct IH_sub as [ops_sub [Hvalid_sub Hcost_sub]].
      (* Compute lev_distance and find minimum branch *)
      rewrite lev_distance_cons.
      unfold min3.
      (* Case analysis on which branch achieves minimum *)
      destruct (Nat.le_ge_cases (lev_distance query' (c2 :: dict') + 1)
                                (lev_distance (c1 :: query') dict' + 1)) as [Hdel_ins | Hins_del].
      * (* delete branch <= insert branch *)
        destruct (Nat.le_ge_cases (lev_distance query' (c2 :: dict') + 1)
                                  (lev_distance query' dict' + subst_cost c1 c2)) as [Hdel_sub | Hsub_del].
        -- (* delete branch is optimal *)
           exists (Edit_Delete c1 :: ops_del).
           split.
           ++ apply valid_cons with (qi' := 1) (di' := 0).
              ** unfold valid_edit_op_at. simpl. reflexivity.
              ** simpl. reflexivity.
              ** (* Shift ops_del from (0,0) on query' to (1,0) on c1::query' *)
                 apply valid_sequence_shift_query. exact Hvalid_del.
           ++ simpl. rewrite Hcost_del.
              (* Goal: 1 + lev_distance query' (c2::dict') = min ... *)
              assert (Hmin : min (lev_distance query' (c2 :: dict') + 1)
                                 (min (lev_distance (c1 :: query') dict' + 1)
                                      (lev_distance query' dict' + subst_cost c1 c2))
                           = lev_distance query' (c2 :: dict') + 1).
              { apply Nat.min_l. apply Nat.min_glb; [exact Hdel_ins | exact Hdel_sub]. }
              lia.
        -- (* subst/match branch is optimal *)
           destruct (char_eq c1 c2) eqn:Heq_chars.
           ++ (* c1 = c2: match *)
              exists (Edit_Match c1 :: ops_sub).
              split.
              ** apply valid_cons with (qi' := 1) (di' := 1).
                 --- unfold valid_edit_op_at. simpl.
                     apply char_eq_eq in Heq_chars.
                     subst c2. split; reflexivity.
                 --- simpl. reflexivity.
                 --- (* Shift ops_sub from (0,0) on query'/dict' to (1,1) on c1::query'/c2::dict' *)
                     apply valid_sequence_shift_query.
                     apply valid_sequence_shift_dict.
                     exact Hvalid_sub.
              ** simpl. rewrite Hcost_sub.
                 unfold subst_cost. rewrite Heq_chars. simpl.
                 rewrite Nat.add_0_r.
                 (* Goal: 0 + lev_distance query' dict' = min ... *)
                 (* Since c1 = c2, we have subst_cost c1 c2 = 0, so subst branch is cheapest *)
                 (* Hsub_del: lev_distance query' dict' + subst_cost c1 c2 <= lev_distance query' (c2 :: dict') + 1 *)
                 (* With char_eq c1 c2 = true, subst_cost c1 c2 = 0 *)
                 unfold subst_cost in Hsub_del. rewrite Heq_chars in Hsub_del. simpl in Hsub_del.
                 rewrite Nat.add_0_r in Hsub_del.
                 apply char_eq_eq in Heq_chars. subst c2.
                 (* Use the fact that for the min, if we have matching chars, subst_cost is 0
                    and lev_distance query' dict' is optimal *)
                 (* The minimum is achieved by the subst/match branch *)
                 assert (Hbound_del : lev_distance query' dict' <= lev_distance query' (c1 :: dict') + 1)
                   by exact Hsub_del.
                 assert (Hbound_ins : lev_distance query' dict' <= lev_distance (c1 :: query') dict' + 1)
                   by apply lev_distance_drop_query_char.
                 (* Now we can establish the min *)
                 assert (Hmin_inner : min (lev_distance (c1 :: query') dict' + 1)
                                          (lev_distance query' dict')
                                   = lev_distance query' dict')
                   by (apply Nat.min_r; lia).
                 assert (Hmin_outer : min (lev_distance query' (c1 :: dict') + 1)
                                          (lev_distance query' dict')
                                   = lev_distance query' dict')
                   by (apply Nat.min_r; lia).
                 lia.
           ++ (* c1 <> c2: substitute *)
              exists (Edit_Substitute c1 c2 :: ops_sub).
              split.
              ** apply valid_cons with (qi' := 1) (di' := 1).
                 --- unfold valid_edit_op_at. simpl.
                     assert (Hneq : c1 <> c2).
                     { intro Heq. subst. rewrite char_eq_refl in Heq_chars. discriminate. }
                     split; [reflexivity | split; [reflexivity | exact Hneq]].
                 --- simpl. reflexivity.
                 --- (* Shift ops_sub from (0,0) on query'/dict' to (1,1) on c1::query'/c2::dict' *)
                     apply valid_sequence_shift_query.
                     apply valid_sequence_shift_dict.
                     exact Hvalid_sub.
              ** simpl. rewrite Hcost_sub.
                 unfold subst_cost. rewrite Heq_chars.
                 (* Simplify Hsub_del: subst_cost c1 c2 = 1 since c1 <> c2 *)
                 unfold subst_cost in Hsub_del. rewrite Heq_chars in Hsub_del. simpl in Hsub_del.
                 assert (Hmin : min (lev_distance query' (c2 :: dict') + 1)
                                    (min (lev_distance (c1 :: query') dict' + 1)
                                         (lev_distance query' dict' + 1))
                              = lev_distance query' dict' + 1).
                 { (* From Hsub_del: subst <= delete, i.e., query' dict' + 1 <= query' (c2::dict') + 1 *)
                   (* From Hdel_ins: delete <= insert, i.e., query' (c2::dict') + 1 <= (c1::query') dict' + 1 *)
                   (* By transitivity: query' dict' + 1 <= (c1::query') dict' + 1 *)
                   assert (Hinner : lev_distance query' dict' + 1 <= lev_distance (c1 :: query') dict' + 1)
                     by lia.
                   rewrite (Nat.min_r _ _ Hinner).
                   rewrite (Nat.min_r _ _ Hsub_del).
                   reflexivity. }
                 lia.
      * (* insert branch <= delete branch *)
        destruct (Nat.le_ge_cases (lev_distance (c1 :: query') dict' + 1)
                                  (lev_distance query' dict' + subst_cost c1 c2)) as [Hins_sub | Hsub_ins].
        -- (* insert branch is optimal *)
           exists (Edit_Insert c2 :: ops_ins).
           split.
           ++ apply valid_cons with (qi' := 0) (di' := 1).
              ** unfold valid_edit_op_at. simpl. reflexivity.
              ** simpl. reflexivity.
              ** (* Shift ops_ins from (0,0) on c1::query'/dict' to (0,1) on c1::query'/c2::dict' *)
                 apply valid_sequence_shift_dict. exact Hvalid_ins.
           ++ simpl. rewrite Hcost_ins.
              assert (Hmin : min (lev_distance query' (c2 :: dict') + 1)
                                 (min (lev_distance (c1 :: query') dict' + 1)
                                      (lev_distance query' dict' + subst_cost c1 c2))
                           = lev_distance (c1 :: query') dict' + 1).
              { (* First simplify inner min using Hins_sub *)
                rewrite (Nat.min_l _ _ Hins_sub).
                (* Now prove outer min: insert <= delete *)
                apply Nat.min_r. exact Hins_del. }
              lia.
        -- (* subst/match branch is optimal via insert comparison *)
           destruct (char_eq c1 c2) eqn:Heq_chars.
           ++ (* c1 = c2: match *)
              exists (Edit_Match c1 :: ops_sub).
              split.
              ** apply valid_cons with (qi' := 1) (di' := 1).
                 --- unfold valid_edit_op_at. simpl.
                     apply char_eq_eq in Heq_chars.
                     subst c2. split; reflexivity.
                 --- simpl. reflexivity.
                 --- (* Shift ops_sub from (0,0) on query'/dict' to (1,1) on c1::query'/c2::dict' *)
                     apply valid_sequence_shift_query.
                     apply valid_sequence_shift_dict.
                     exact Hvalid_sub.
              ** simpl. rewrite Hcost_sub.
                 unfold subst_cost. rewrite Heq_chars. simpl.
                 rewrite Nat.add_0_r.
                 (* Simplify Hsub_ins with c1=c2 so subst_cost=0 *)
                 unfold subst_cost in Hsub_ins. rewrite Heq_chars in Hsub_ins. simpl in Hsub_ins.
                 rewrite Nat.add_0_r in Hsub_ins.
                 apply char_eq_eq in Heq_chars. subst c2.
                 assert (Hmin : min (lev_distance query' (c1 :: dict') + 1)
                                    (min (lev_distance (c1 :: query') dict' + 1)
                                         (lev_distance query' dict'))
                              = lev_distance query' dict').
                 { (* Inner min: need query' dict' <= (c1::query') dict' + 1 *)
                   assert (Hinner : lev_distance query' dict' <= lev_distance (c1 :: query') dict' + 1)
                     by exact Hsub_ins.
                   rewrite (Nat.min_r _ _ Hinner).
                   (* Outer min: need query' dict' <= query' (c1::dict') + 1 *)
                   apply Nat.min_r.
                   apply lev_distance_drop_dict_char. }
                 lia.
           ++ (* c1 <> c2: substitute *)
              exists (Edit_Substitute c1 c2 :: ops_sub).
              split.
              ** apply valid_cons with (qi' := 1) (di' := 1).
                 --- unfold valid_edit_op_at. simpl.
                     assert (Hneq : c1 <> c2).
                     { intro Heq. subst. rewrite char_eq_refl in Heq_chars. discriminate. }
                     split; [reflexivity | split; [reflexivity | exact Hneq]].
                 --- simpl. reflexivity.
                 --- (* Shift ops_sub from (0,0) on query'/dict' to (1,1) on c1::query'/c2::dict' *)
                     apply valid_sequence_shift_query.
                     apply valid_sequence_shift_dict.
                     exact Hvalid_sub.
              ** simpl. rewrite Hcost_sub.
                 unfold subst_cost. rewrite Heq_chars.
                 (* Simplify Hsub_ins: subst_cost c1 c2 = 1 since c1 <> c2 *)
                 unfold subst_cost in Hsub_ins. rewrite Heq_chars in Hsub_ins. simpl in Hsub_ins.
                 assert (Hmin : min (lev_distance query' (c2 :: dict') + 1)
                                    (min (lev_distance (c1 :: query') dict' + 1)
                                         (lev_distance query' dict' + 1))
                              = lev_distance query' dict' + 1).
                 { (* Inner min: subst <= insert, i.e., query' dict' + 1 <= (c1::query') dict' + 1 *)
                   rewrite (Nat.min_r _ _ Hsub_ins).
                   (* Outer min: subst <= delete follows from Hsub_ins and Hins_del by transitivity *)
                   apply Nat.min_r. lia. }
                 lia.
Qed.

(** * Edit Sequence to Automaton Path *)

(** Key insight: Each edit operation corresponds to an automaton transition:
    - Edit_Match: Consuming dict char triggers match transition
    - Edit_Substitute: Consuming dict char triggers substitute transition
    - Edit_Delete: Epsilon closure handles deletions
    - Edit_Insert: Consuming dict char triggers insert transition

    The automaton explores all valid edit sequences up to cost n.
*)

(** An edit sequence can be traced through the automaton *)
Definition sequence_traceable (alg : Algorithm) (query : list Char) (n : nat)
                               (dict : list Char) (ops : edit_sequence) : Prop :=
  valid_edit_sequence query dict 0 0 ops /\
  sequence_cost ops <= n.

(** Generalized reachability lemma: from any valid starting position *)
Lemma traceable_implies_reachable_gen : forall query dict qi di ops n e,
  valid_edit_sequence query dict qi di ops ->
  sequence_cost ops + e <= n ->
  di <= length dict ->
  position_reachable query n (firstn di dict) (std_pos qi e) ->
  exists p,
    position_reachable query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= sequence_cost ops + e.
Proof.
  intros query dict qi di ops n e Hvalid.
  revert n e.
  induction Hvalid as [query' dict' | query' dict' qi' di' qi'' di'' op ops' Hop Happly Hrest IH];
    intros n e Hcost Hdi Hreach.
  - (* valid_empty: at end position *)
    exists (std_pos (length query') e).
    simpl in *.
    rewrite firstn_all2 in Hreach by lia.
    split; [exact Hreach |].
    split; [reflexivity |].
    split; [reflexivity |].
    lia.
  - (* valid_cons: op :: ops' *)
    simpl in Hcost.
    destruct op as [c | c1 c2 | c | c].
    + (* Edit_Match c *)
      simpl in Happly. inversion Happly as [[Hqi' Hdi']]. clear Happly.
      simpl in Hop.
      destruct Hop as [Hq Hd].
      (* Match: position advances in both query and dict, no error increase *)
      assert (Hd_bound : di' < length dict').
      { apply nth_error_Some. rewrite Hd. discriminate. }
      assert (Hdi'' : di'' <= length dict').
      { subst di''. lia. }
      assert (Hqi'_bound : qi' < length query').
      { apply nth_error_Some. rewrite Hq. discriminate. }
      (* Build reachable for position (qi'', e) with dict_prefix = firstn di'' dict' *)
      assert (Hreach' : position_reachable query' n (firstn di'' dict') (std_pos qi'' e)).
      { subst qi'' di''.
        (* firstn (S di') dict' = firstn di' dict' ++ [c] when nth di' dict' = Some c *)
        rewrite (firstn_S_snoc_nth_error _ _ _ Hd).
        apply reach_match with (c := c).
        - simpl. exact Hreach.
        - simpl. exact Hqi'_bound.
        - exact Hq. }
      specialize (IH n e).
      assert (Hcost' : sequence_cost ops' + e <= n) by lia.
      apply IH; [exact Hcost' | exact Hdi'' | exact Hreach'].
    + (* Edit_Substitute c1 c2 *)
      simpl in Happly. inversion Happly as [[Hqi' Hdi']]. clear Happly.
      simpl in Hop.
      simpl in Hcost.  (* Simplify edit_cost (Edit_Substitute c1 c2) = 1 *)
      destruct Hop as [Hq [Hd Hneq]].
      (* Substitute: position advances in both, error increases by 1 *)
      assert (Hd_bound : di' < length dict').
      { apply nth_error_Some. rewrite Hd. discriminate. }
      assert (Hdi'' : di'' <= length dict').
      { subst di''. lia. }
      assert (Hqi'_bound : qi' < length query').
      { apply nth_error_Some. rewrite Hq. discriminate. }
      (* Use reach_substitute to advance *)
      assert (Hreach' : position_reachable query' n (firstn di'' dict') (std_pos qi'' (S e))).
      { subst qi'' di''.
        rewrite (firstn_S_snoc_nth_error _ _ _ Hd).
        apply reach_substitute with (c := c2) (c' := c1).
        - simpl. exact Hreach.
        - simpl. exact Hqi'_bound.
        - exact Hq.
        - intro Heq. apply Hneq. symmetry. exact Heq.
        - lia. }
      specialize (IH n (S e)).
      assert (Hcost' : sequence_cost ops' + S e <= n) by lia.
      destruct (IH Hcost' Hdi'' Hreach') as [p [Hp1 [Hp2 [Hp3 Hp4]]]].
      exists p. split; [exact Hp1 |]. split; [exact Hp2 |]. split; [exact Hp3 |].
      simpl. lia.
    + (* Edit_Delete c *)
      simpl in Happly. inversion Happly as [[Hqi'_eq Hdi']]. clear Happly.
      simpl in Hop.
      simpl in Hcost.  (* Simplify edit_cost (Edit_Delete c) = 1 *)
      (* Delete: position advances in query only, error increases by 1 *)
      assert (Hdi'' : di'' <= length dict').
      { rewrite <- Hdi'. exact Hdi. }
      assert (Hqi'_bound : qi' < length query').
      { apply nth_error_Some. rewrite Hop. discriminate. }
      (* Use reach_delete to advance *)
      assert (Hreach' : position_reachable query' n (firstn di'' dict') (std_pos qi'' (S e))).
      { subst qi'' di''.
        apply reach_delete.
        - exact Hreach.
        - simpl. lia.
        - lia. }
      specialize (IH n (S e)).
      assert (Hcost' : sequence_cost ops' + S e <= n) by lia.
      destruct (IH Hcost' Hdi'' Hreach') as [p [Hp1 [Hp2 [Hp3 Hp4]]]].
      exists p. split; [exact Hp1 |]. split; [exact Hp2 |]. split; [exact Hp3 |].
      simpl. lia.
    + (* Edit_Insert c *)
      simpl in Happly. inversion Happly as [[Hqi' Hdi']]. clear Happly.
      simpl in Hop.
      simpl in Hcost.  (* Simplify edit_cost (Edit_Insert c) = 1 *)
      (* Insert: position advances in dict only, error increases by 1 *)
      assert (Hd_bound : di' < length dict').
      { apply nth_error_Some. rewrite Hop. discriminate. }
      assert (Hdi'' : di'' <= length dict').
      { subst di''. lia. }
      (* Use reach_insert to advance *)
      assert (Hreach' : position_reachable query' n (firstn di'' dict') (std_pos qi'' (S e))).
      { subst qi'' di''.
        rewrite (firstn_S_snoc_nth_error _ _ _ Hop).
        apply reach_insert.
        - simpl. exact Hreach.
        - lia. }
      specialize (IH n (S e)).
      assert (Hcost' : sequence_cost ops' + S e <= n) by lia.
      destruct (IH Hcost' Hdi'' Hreach') as [p [Hp1 [Hp2 [Hp3 Hp4]]]].
      exists p. split; [exact Hp1 |]. split; [exact Hp2 |]. split; [exact Hp3 |].
      simpl. lia.
Qed.

(** If a sequence is traceable, the automaton will find it *)
Lemma traceable_implies_reachable : forall query dict n ops,
  valid_edit_sequence query dict 0 0 ops ->
  sequence_cost ops <= n ->
  exists p,
    position_reachable query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= sequence_cost ops.
Proof.
  intros query dict n ops Hvalid Hcost.
  destruct (traceable_implies_reachable_gen query dict 0 0 ops n 0
              Hvalid ltac:(simpl; lia) ltac:(lia) (reach_initial query n))
    as [p [Hp1 [Hp2 [Hp3 Hp4]]]].
  exists p. split; [exact Hp1 |]. split; [exact Hp2 |]. split; [exact Hp3 |].
  lia.
Qed.

(** * Subsumption and Position Containment *)

(** Position subsumption: p1 subsumes p2 if p1 is "at least as good" *)
Definition position_subsumes (p1 p2 : Position) : Prop :=
  term_index p1 = term_index p2 /\
  is_special p1 = is_special p2 /\
  num_errors p1 <= num_errors p2.

(** A position list contains (or subsumes) another position *)
Definition positions_contain (ps : list Position) (p : Position) : Prop :=
  exists p', In p' ps /\ position_subsumes p' p.

(** A position list represents another position under the executable
    algorithm-specific antichain subsumption relation. This is weaker than
    [positions_contain]: Standard subsumption may represent a position at a
    nearby query index when the saved error budget pays for the offset. *)
Definition positions_subsume (alg : Algorithm) (qlen : nat) (ps : list Position) (p : Position) : Prop :=
  exists p', In p' ps /\ subsumes alg qlen p' p = true.

(** MergeAndSplit uses a strict subsumption relation, so exact self-subsumption
    is intentionally false. For completeness we therefore track a pruning chain:
    a position is covered when it is present, or when a covered representative
    subsumes it. *)
Inductive positions_cover_merge_split (qlen : nat) (ps : list Position) :
  Position -> Prop :=
  | cover_ms_in : forall p,
      In p ps ->
      positions_cover_merge_split qlen ps p
  | cover_ms_sub : forall p q,
      positions_cover_merge_split qlen ps q ->
      subsumes MergeAndSplit qlen q p = true ->
      positions_cover_merge_split qlen ps p.

Lemma positions_cover_merge_split_monotone : forall qlen ps ps' p,
  incl ps ps' ->
  positions_cover_merge_split qlen ps p ->
  positions_cover_merge_split qlen ps' p.
Proof.
  intros qlen ps ps' p Hincl Hcover.
  induction Hcover.
  - apply cover_ms_in. apply Hincl. exact H.
  - eapply cover_ms_sub; eauto.
Qed.

Lemma positions_cover_merge_split_subsumed_by_any : forall qlen ps p,
  subsumed_by_any MergeAndSplit qlen p ps = true ->
  positions_cover_merge_split qlen ps p.
Proof.
  intros qlen ps p Hsub.
  apply subsumed_by_any_correct in Hsub as [q [Hq_in Hq_sub]].
  eapply cover_ms_sub.
  - apply cover_ms_in. exact Hq_in.
  - exact Hq_sub.
Qed.

Lemma positions_cover_merge_split_remove_subsumed : forall qlen r ps p,
  positions_cover_merge_split qlen ps p ->
  positions_cover_merge_split qlen (r :: remove_subsumed MergeAndSplit qlen r ps) p.
Proof.
  intros qlen r ps p Hcover.
  induction Hcover as [p Hin | p q Hcover IH Hsub_qp].
  - destruct (subsumes MergeAndSplit qlen r p) eqn:Hrsubp.
    + eapply cover_ms_sub.
      * apply cover_ms_in. simpl. left. reflexivity.
      * exact Hrsubp.
    + apply cover_ms_in. simpl. right.
      apply in_remove_subsumed_if_not_subsumed; assumption.
  - eapply cover_ms_sub; eauto.
Qed.

Lemma state_insert_covers_inserted_merge_split : forall qlen p s,
  algorithm s = MergeAndSplit ->
  query_length s = qlen ->
  positions_cover_merge_split qlen (positions (state_insert p s)) p.
Proof.
  intros qlen p s Halg Hqlen.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any MergeAndSplit qlen p (positions s)) eqn:Hsub.
  - apply positions_cover_merge_split_monotone with (ps := positions s).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + apply positions_cover_merge_split_subsumed_by_any. exact Hsub.
  - apply cover_ms_in.
    apply fold_right_sorted_insert_preserves_In.
    simpl. left. reflexivity.
Qed.

Lemma state_insert_preserves_cover_merge_split : forall qlen p s target,
  algorithm s = MergeAndSplit ->
  query_length s = qlen ->
  positions_cover_merge_split qlen (positions s) target ->
  positions_cover_merge_split qlen (positions (state_insert p s)) target.
Proof.
  intros qlen p s target Halg Hqlen Hcover.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any MergeAndSplit qlen p (positions s)) eqn:Hsub.
  - apply positions_cover_merge_split_monotone with (ps := positions s).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + exact Hcover.
  - apply positions_cover_merge_split_monotone
      with (ps := p :: remove_subsumed MergeAndSplit qlen p (positions s)).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + apply positions_cover_merge_split_remove_subsumed. exact Hcover.
Qed.

Lemma fold_state_insert_preserves_cover_merge_split : forall qlen inserts s p,
  algorithm s = MergeAndSplit ->
  query_length s = qlen ->
  positions_cover_merge_split qlen (positions s) p ->
  positions_cover_merge_split qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hcover.
  - simpl. exact Hcover.
  - simpl.
    apply IH.
    + unfold state_insert. simpl. exact Halg.
    + unfold state_insert. simpl. exact Hqlen.
    + apply state_insert_preserves_cover_merge_split with (qlen := qlen);
        assumption.
Qed.

Lemma fold_state_insert_covers_member_merge_split : forall qlen inserts s p,
  algorithm s = MergeAndSplit ->
  query_length s = qlen ->
  In p inserts ->
  positions_cover_merge_split qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hin.
  - inversion Hin.
  - simpl in Hin |- *.
    destruct Hin as [Heq | Hin_rest].
    + subst q.
      apply fold_state_insert_preserves_cover_merge_split.
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * apply state_insert_covers_inserted_merge_split with (qlen := qlen);
          assumption.
    + apply (IH (state_insert q s) p).
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * exact Hin_rest.
Qed.

Lemma transition_state_merge_split_covers_closed_position : forall s c query n s' p,
  transition_state MergeAndSplit s c query n = Some s' ->
  In p
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  positions_cover_merge_split (query_length s) (positions s') p.
Proof.
  intros s c query n s' p Htrans Hin.
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions MergeAndSplit (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  fold min_i in Hin.
  fold cv in Hin.
  fold trans_positions in Hin.
  fold closed_positions in Hin.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  apply fold_state_insert_covers_member_merge_split.
  - unfold empty_state. reflexivity.
  - unfold empty_state. reflexivity.
  - exact Hin.
Qed.

Lemma positions_cover_merge_split_final : forall qlen ps p,
  positions_cover_merge_split qlen ps p ->
  position_is_final qlen p = true ->
  existsb (position_is_final qlen) ps = true.
Proof.
  intros qlen ps p Hcover Hfinal.
  induction Hcover as [p Hin | p q Hcover IH Hsub].
  - rewrite existsb_exists. exists p. split; assumption.
  - destruct (position_is_final qlen q) eqn:Hq_final.
    + apply IH. reflexivity.
    + exfalso.
      assert (Hq_sub_final : position_is_final_for_subsumption qlen q = false).
      { unfold position_is_final_for_subsumption, position_is_final in *.
        exact Hq_final. }
      assert (Hp_sub_final : position_is_final_for_subsumption qlen p = true).
      { unfold position_is_final_for_subsumption, position_is_final in *.
        exact Hfinal. }
      pose proof (non_final_cannot_subsume_final
                    MergeAndSplit qlen q p Hq_sub_final Hp_sub_final) as Hnot.
      rewrite Hsub in Hnot. discriminate.
Qed.

Lemma covered_final_state_accepts_merge_split : forall (query : list Char) s p,
  query_length s = length query ->
  positions_cover_merge_split (length query) (positions s) p ->
  term_index p = length query ->
  state_is_final s = true.
Proof.
  intros query s p Hqlen Hcover Hterm.
  unfold state_is_final.
  rewrite Hqlen.
  apply positions_cover_merge_split_final with (p := p).
  - exact Hcover.
  - unfold position_is_final. rewrite Nat.leb_le. lia.
Qed.

(** Transposition uses ordinary self-subsumption for same-index positions, but
    the proof still needs a cover relation rather than raw membership: antichain
    insertion may replace a position by an equal-index lower-error
    representative. *)
Inductive positions_cover_transposition (qlen : nat) (ps : list Position) :
  Position -> Prop :=
  | cover_trans_in : forall p,
      In p ps ->
      positions_cover_transposition qlen ps p
  | cover_trans_sub : forall p q,
      positions_cover_transposition qlen ps q ->
      subsumes Transposition qlen q p = true ->
      positions_cover_transposition qlen ps p.

Lemma positions_cover_transposition_monotone : forall qlen ps ps' p,
  incl ps ps' ->
  positions_cover_transposition qlen ps p ->
  positions_cover_transposition qlen ps' p.
Proof.
  intros qlen ps ps' p Hincl Hcover.
  induction Hcover.
  - apply cover_trans_in. apply Hincl. exact H.
  - eapply cover_trans_sub; eauto.
Qed.

Lemma positions_cover_transposition_subsumed_by_any : forall qlen ps p,
  subsumed_by_any Transposition qlen p ps = true ->
  positions_cover_transposition qlen ps p.
Proof.
  intros qlen ps p Hsub.
  apply subsumed_by_any_correct in Hsub as [q [Hq_in Hq_sub]].
  eapply cover_trans_sub.
  - apply cover_trans_in. exact Hq_in.
  - exact Hq_sub.
Qed.

Lemma positions_cover_transposition_remove_subsumed : forall qlen r ps p,
  positions_cover_transposition qlen ps p ->
  positions_cover_transposition qlen
    (r :: remove_subsumed Transposition qlen r ps) p.
Proof.
  intros qlen r ps p Hcover.
  induction Hcover as [p Hin | p q Hcover IH Hsub_qp].
  - destruct (subsumes Transposition qlen r p) eqn:Hrsubp.
    + eapply cover_trans_sub.
      * apply cover_trans_in. simpl. left. reflexivity.
      * exact Hrsubp.
    + apply cover_trans_in. simpl. right.
      apply in_remove_subsumed_if_not_subsumed; assumption.
  - eapply cover_trans_sub; eauto.
Qed.

Lemma state_insert_covers_inserted_transposition : forall qlen p s,
  algorithm s = Transposition ->
  query_length s = qlen ->
  positions_cover_transposition qlen (positions (state_insert p s)) p.
Proof.
  intros qlen p s Halg Hqlen.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any Transposition qlen p (positions s)) eqn:Hsub.
  - apply positions_cover_transposition_monotone with (ps := positions s).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + apply positions_cover_transposition_subsumed_by_any. exact Hsub.
  - apply cover_trans_in.
    apply fold_right_sorted_insert_preserves_In.
    simpl. left. reflexivity.
Qed.

Lemma state_insert_preserves_cover_transposition : forall qlen p s target,
  algorithm s = Transposition ->
  query_length s = qlen ->
  positions_cover_transposition qlen (positions s) target ->
  positions_cover_transposition qlen (positions (state_insert p s)) target.
Proof.
  intros qlen p s target Halg Hqlen Hcover.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any Transposition qlen p (positions s)) eqn:Hsub.
  - apply positions_cover_transposition_monotone with (ps := positions s).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + exact Hcover.
  - apply positions_cover_transposition_monotone
      with (ps := p :: remove_subsumed Transposition qlen p (positions s)).
    + intros q Hq. apply fold_right_sorted_insert_preserves_In. exact Hq.
    + apply positions_cover_transposition_remove_subsumed. exact Hcover.
Qed.

Lemma fold_state_insert_preserves_cover_transposition : forall qlen inserts s p,
  algorithm s = Transposition ->
  query_length s = qlen ->
  positions_cover_transposition qlen (positions s) p ->
  positions_cover_transposition qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hcover.
  - simpl. exact Hcover.
  - simpl.
    apply IH.
    + unfold state_insert. simpl. exact Halg.
    + unfold state_insert. simpl. exact Hqlen.
    + apply state_insert_preserves_cover_transposition with (qlen := qlen);
        assumption.
Qed.

Lemma fold_state_insert_covers_member_transposition : forall qlen inserts s p,
  algorithm s = Transposition ->
  query_length s = qlen ->
  In p inserts ->
  positions_cover_transposition qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hin.
  - inversion Hin.
  - simpl in Hin |- *.
    destruct Hin as [Heq | Hin_rest].
    + subst q.
      apply fold_state_insert_preserves_cover_transposition.
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * apply state_insert_covers_inserted_transposition with (qlen := qlen);
          assumption.
    + apply (IH (state_insert q s) p).
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * exact Hin_rest.
Qed.

Lemma transition_state_transposition_covers_closed_position :
  forall s c query n s' p,
  transition_state Transposition s c query n = Some s' ->
  In p
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  positions_cover_transposition (query_length s) (positions s') p.
Proof.
  intros s c query n s' p Htrans Hin.
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions Transposition (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  fold min_i in Hin.
  fold cv in Hin.
  fold trans_positions in Hin.
  fold closed_positions in Hin.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  apply fold_state_insert_covers_member_transposition.
  - unfold empty_state. reflexivity.
  - unfold empty_state. reflexivity.
  - exact Hin.
Qed.

Lemma positions_cover_transposition_final : forall qlen ps p,
  positions_cover_transposition qlen ps p ->
  position_is_final qlen p = true ->
  existsb (position_is_final qlen) ps = true.
Proof.
  intros qlen ps p Hcover Hfinal.
  induction Hcover as [p Hin | p q Hcover IH Hsub].
  - rewrite existsb_exists. exists p. split; assumption.
  - destruct (position_is_final qlen q) eqn:Hq_final.
    + apply IH. reflexivity.
    + exfalso.
      assert (Hq_sub_final : position_is_final_for_subsumption qlen q = false).
      { unfold position_is_final_for_subsumption, position_is_final in *.
        exact Hq_final. }
      assert (Hp_sub_final : position_is_final_for_subsumption qlen p = true).
      { unfold position_is_final_for_subsumption, position_is_final in *.
        exact Hfinal. }
      pose proof (non_final_cannot_subsume_final
                    Transposition qlen q p Hq_sub_final Hp_sub_final) as Hnot.
      rewrite Hsub in Hnot. discriminate.
Qed.

Lemma covered_final_state_accepts_transposition : forall (query : list Char) s p,
  query_length s = length query ->
  positions_cover_transposition (length query) (positions s) p ->
  term_index p = length query ->
  state_is_final s = true.
Proof.
  intros query s p Hqlen Hcover Hterm.
  unfold state_is_final.
  rewrite Hqlen.
  apply positions_cover_transposition_final with (p := p).
  - exact Hcover.
  - unfold position_is_final. rewrite Nat.leb_le. lia.
Qed.

(** Exact membership gives executable Standard representation. *)
Lemma positions_subsume_standard_refl_in : forall qlen ps p,
  In p ps ->
  positions_subsume Standard qlen ps p.
Proof.
  intros qlen ps p Hin.
  exists p. split; [exact Hin |].
  simpl. apply subsumes_standard_refl.
Qed.

(** Standard subsumption is stable across one non-final delete successor.
    The final-position guard is the only reason for the explicit
    [S i < qlen] premise: non-final representatives are intentionally not
    allowed to subsume final positions. *)
Lemma subsumes_standard_delete_successor_nonfinal : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  S i < qlen ->
  subsumes Standard qlen p (std_pos (S i) (S e)) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hnonfinal.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct (qlen <=? j) eqn:Hj_final; simpl in *.
  - rewrite Bool.andb_true_iff in Hsub.
    destruct Hsub as [Herr Hdist].
    apply Nat.leb_le in Herr.
    apply Nat.leb_le in Hdist.
    rewrite Bool.andb_true_iff.
    split.
    + apply Nat.leb_le. lia.
    + apply Nat.leb_le.
      assert (Htri : abs_diff j (S i) <= abs_diff j i + 1).
      { unfold abs_diff.
        destruct (j <=? i) eqn:Hji, (j <=? S i) eqn:HjSi;
          apply Nat.leb_le in Hji || apply Nat.leb_gt in Hji;
          apply Nat.leb_le in HjSi || apply Nat.leb_gt in HjSi;
          lia. }
      eapply Nat.le_trans; [exact Htri |].
      destruct e' as [|e0]; simpl in *; lia.
  - destruct (qlen <=? i) eqn:Hi_final; simpl in *; [discriminate|].
    assert (Hsucc_nonfinal : (qlen <=? S i) = false).
    { apply Nat.leb_gt. lia. }
    rewrite Hsucc_nonfinal. simpl.
    rewrite Bool.andb_true_iff in Hsub.
    destruct Hsub as [Herr Hdist].
    apply Nat.leb_le in Herr.
    apply Nat.leb_le in Hdist.
    rewrite Bool.andb_true_iff.
    split.
    + apply Nat.leb_le. lia.
    + apply Nat.leb_le.
      assert (Htri : abs_diff j (S i) <= abs_diff j i + 1).
      { unfold abs_diff.
        destruct (j <=? i) eqn:Hji, (j <=? S i) eqn:HjSi;
          apply Nat.leb_le in Hji || apply Nat.leb_gt in Hji;
          apply Nat.leb_le in HjSi || apply Nat.leb_gt in HjSi;
          lia. }
      eapply Nat.le_trans; [exact Htri |].
      destruct e' as [|e0]; simpl in *; lia.
Qed.

Lemma positions_subsume_standard_delete_successor_nonfinal : forall qlen ps i e,
  positions_subsume Standard qlen ps (std_pos i e) ->
  S i < qlen ->
  positions_subsume Standard qlen ps (std_pos (S i) (S e)).
Proof.
  intros qlen ps i e [p [Hin Hsub]] Hnonfinal.
  exists p. split; [exact Hin |].
  exact (subsumes_standard_delete_successor_nonfinal qlen p i e Hsub Hnonfinal).
Qed.

(** Generalized delete-successor representation arithmetic. If a Standard
    representative subsumes [(i,e)] and the exact reachable path deletes to
    [(S i, S e)], then either the representative is already far enough ahead,
    or a finite delete chain from the representative reaches a position that
    subsumes the exact delete successor. This handles the final-position case
    where non-final representatives are intentionally forbidden from subsuming
    final positions. *)
Lemma subsumes_standard_delete_successor_chain : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  term_index p <= qlen ->
  S i <= qlen ->
  exists k,
    term_index p + k <= qlen /\
    num_errors p + k <= S e /\
    subsumes Standard qlen
      (std_pos (term_index p + k) (num_errors p + k))
      (std_pos (S i) (S e)) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hp_bound Hstep.
  simpl in Hp_bound.
  unfold subsumes in Hsub. simpl in Hsub.
  unfold subsumes_standard in Hsub.
  unfold position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  destruct (j <=? S i) eqn:Hj_le_succ.
  - apply Nat.leb_le in Hj_le_succ.
    exists (S i - j).
    assert (Hidx : j + (S i - j) = S i) by lia.
    assert (Herr_succ : e' + (S i - j) <= S e).
    { change (e' + (S i - j) <= S e).
      unfold abs_diff in Hdist.
      destruct (j <=? i) eqn:Hj_le_i.
      * apply Nat.leb_le in Hj_le_i.
        replace (S i - j) with (S (i - j)) by lia.
        assert (Hsum : e' + (i - j) <= e).
        { eapply Nat.le_trans.
          - apply Nat.add_le_mono_l. exact Hdist.
          - rewrite Nat.add_comm.
            rewrite Nat.sub_add by exact Herr.
            lia. }
        replace (e' + S (i - j)) with (S (e' + (i - j))) by lia.
        apply le_n_S. exact Hsum.
      * apply Nat.leb_gt in Hj_le_i.
        assert (j = S i) by lia.
        subst j. simpl. lia. }
    split.
    + change (j + (S i - j) <= qlen).
      rewrite Hidx. exact Hstep.
    + split.
      * exact Herr_succ.
      * change (subsumes Standard qlen
          (std_pos (j + (S i - j)) (e' + (S i - j)))
          (std_pos (S i) (S e)) = true).
        rewrite Hidx.
        unfold subsumes. simpl.
        unfold subsumes_standard, position_is_final_for_subsumption. simpl.
        destruct ((negb (qlen <=? S i)) && (qlen <=? S i)) eqn:Hsame_final.
        { destruct (qlen <=? S i); discriminate. }
        apply Bool.andb_true_iff.
        split.
        -- apply Nat.leb_le. exact Herr_succ.
        -- rewrite abs_diff_self. apply Nat.leb_le. lia.
  - apply Nat.leb_gt in Hj_le_succ.
    exists 0.
    split.
    + simpl. lia.
    + split.
      * simpl. lia.
      * unfold subsumes. simpl.
        unfold subsumes_standard, position_is_final_for_subsumption. simpl.
        replace (j + 0) with j by lia.
        replace (e' + 0) with e' by lia.
        destruct ((negb (qlen <=? j)) && (qlen <=? S i)) eqn:Hfinal_succ.
        { apply andb_true_iff in Hfinal_succ.
          destruct Hfinal_succ as [Hj_nonfinal Hsucc_final].
          apply Bool.negb_true_iff in Hj_nonfinal.
          apply Nat.leb_gt in Hj_nonfinal.
          apply Nat.leb_le in Hsucc_final.
          lia. }
        apply Bool.andb_true_iff.
        split.
        -- apply Nat.leb_le. lia.
        -- apply Nat.leb_le.
           unfold abs_diff in *.
           destruct (j <=? i) eqn:Hj_le_i; [apply Nat.leb_le in Hj_le_i; lia|].
           apply Nat.leb_gt in Hj_le_i.
           destruct (j <=? S i) eqn:Hj_le_si; [apply Nat.leb_le in Hj_le_si; lia|].
           apply Nat.leb_gt in Hj_le_si.
           assert (Hgap : j - S i <= j - i) by lia.
           assert (Herr_mono : e - e' <= S e - e') by lia.
           destruct e' as [|e0]; simpl in *; lia.
Qed.

(** Helper: Standard subsumption implies error bound. *)
Lemma subsumes_standard_errors : forall qlen p1 p2,
  subsumes_standard qlen p1 p2 = true ->
  num_errors p1 <= num_errors p2.
Proof.
  intros qlen p1 p2 Hsub.
  unfold subsumes_standard in Hsub.
  destruct ((negb (position_is_final_for_subsumption qlen p1)) &&
            (position_is_final_for_subsumption qlen p2)); [discriminate|].
  apply andb_prop in Hsub. destruct Hsub as [He _].
  apply Nat.leb_le. exact He.
Qed.

(** Standard insert successors preserve executable representation. This is the
    local arithmetic fact used when a represented predecessor consumes a
    dictionary character via insertion: the representative consumes the same
    character with one additional error, and the Standard distance slack is
    unchanged. *)
Lemma subsumes_standard_insert_successor : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  subsumes Standard qlen (std_pos (term_index p) (S (num_errors p)))
    (std_pos i (S e)) = true.
Proof.
  intros qlen [j e' sp] i e Hsub.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  rewrite Bool.andb_true_iff.
  split.
  - apply Nat.leb_le. lia.
  - apply Nat.leb_le. lia.
Qed.

(** If a Standard representative sits at the same query index as the exact
    predecessor it subsumes, then advancing both by a match preserves
    subsumption. *)
Lemma subsumes_standard_match_successor_same_index : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  term_index p = i ->
  subsumes Standard qlen (std_pos (S i) (num_errors p))
    (std_pos (S i) e) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hidx.
  simpl in Hidx. subst j.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct ((negb (qlen <=? i)) && (qlen <=? i)) eqn:Hfinal;
    [destruct (qlen <=? i); discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr _].
  apply Nat.leb_le in Herr.
  destruct ((negb (qlen <=? S i)) && (qlen <=? S i)) eqn:Hfinal_succ;
    [destruct (qlen <=? S i); discriminate|].
  rewrite Bool.andb_true_iff.
  split.
  - apply Nat.leb_le. exact Herr.
  - apply Nat.leb_le. rewrite abs_diff_self. lia.
Qed.

(** The same-index substitution successor is also monotone under Standard
    subsumption. Both sides pay one error, so the existing error ordering is
    preserved. *)
Lemma subsumes_standard_substitute_successor_same_index : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  term_index p = i ->
  subsumes Standard qlen (std_pos (S i) (S (num_errors p)))
    (std_pos (S i) (S e)) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hidx.
  simpl in Hidx. subst j.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct ((negb (qlen <=? i)) && (qlen <=? i)) eqn:Hfinal;
    [destruct (qlen <=? i); discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr _].
  apply Nat.leb_le in Herr.
  destruct ((negb (qlen <=? S i)) && (qlen <=? S i)) eqn:Hfinal_succ;
    [destruct (qlen <=? S i); discriminate|].
  rewrite Bool.andb_true_iff.
  split.
  - apply Nat.leb_le. lia.
  - apply Nat.leb_le. rewrite abs_diff_self. lia.
Qed.

(** If the surviving representative is ahead of a matched predecessor, it can
    consume the dictionary character by insertion. The saved error slack that
    paid for the index offset also pays for the inserted character. *)
Lemma subsumes_standard_match_successor_ahead_insert : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  i < term_index p ->
  term_index p <= qlen ->
  subsumes Standard qlen
    (std_pos (term_index p) (S (num_errors p)))
    (std_pos (S i) e) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hahead Hbound.
  simpl in *.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  assert (Herr_strict : e' < e).
  { unfold abs_diff in Hdist.
    destruct (j <=? i) eqn:Hj_le_i.
    - apply Nat.leb_le in Hj_le_i. lia.
    - apply Nat.leb_gt in Hj_le_i. lia. }
  destruct ((negb (qlen <=? j)) && (qlen <=? S i)) eqn:Hfinal_succ.
  { apply Bool.andb_true_iff in Hfinal_succ.
    destruct Hfinal_succ as [Hj_nonfinal Htarget_final].
    apply Bool.negb_true_iff in Hj_nonfinal.
    apply Nat.leb_gt in Hj_nonfinal.
    apply Nat.leb_le in Htarget_final.
    lia. }
  rewrite Bool.andb_true_iff.
  split.
  - destruct e as [|e0]; simpl; [lia | apply Nat.leb_le; lia].
  - apply Nat.leb_le.
    unfold abs_diff in *.
    destruct (j <=? i) eqn:Hj_le_i.
    + apply Nat.leb_le in Hj_le_i. lia.
    + apply Nat.leb_gt in Hj_le_i.
      destruct (j <=? S i) eqn:Hj_le_si.
      * apply Nat.leb_le in Hj_le_si. lia.
      * apply Nat.leb_gt in Hj_le_si. lia.
Qed.

(** The ahead-representative substitution case is weaker arithmetically: both
    the representative insert successor and the requested substitution successor
    add one error. *)
Lemma subsumes_standard_substitute_successor_ahead_insert : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  i < term_index p ->
  term_index p <= qlen ->
  subsumes Standard qlen
    (std_pos (term_index p) (S (num_errors p)))
    (std_pos (S i) (S e)) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hahead Hbound.
  simpl in *.
  unfold subsumes in *. simpl in *.
  unfold subsumes_standard in *.
  unfold position_is_final_for_subsumption in *.
  simpl in *.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  destruct ((negb (qlen <=? j)) && (qlen <=? S i)) eqn:Hfinal_succ.
  { apply Bool.andb_true_iff in Hfinal_succ.
    destruct Hfinal_succ as [Hj_nonfinal Htarget_final].
    apply Bool.negb_true_iff in Hj_nonfinal.
    apply Nat.leb_gt in Hj_nonfinal.
    apply Nat.leb_le in Htarget_final.
    lia. }
  rewrite Bool.andb_true_iff.
  split.
  - apply Nat.leb_le. lia.
  - apply Nat.leb_le.
    unfold abs_diff in *.
    destruct (j <=? i) eqn:Hj_le_i.
    + apply Nat.leb_le in Hj_le_i. lia.
    + apply Nat.leb_gt in Hj_le_i.
      destruct (j <=? S i) eqn:Hj_le_si.
      * apply Nat.leb_le in Hj_le_si. lia.
      * apply Nat.leb_gt in Hj_le_si. lia.
Qed.

(** Same-index Standard subsumption is exactly monotone in the error count. *)
Lemma subsumes_standard_same_index_error_widen : forall qlen i e1 e2,
  e1 <= e2 ->
  subsumes Standard qlen (std_pos i e1) (std_pos i e2) = true.
Proof.
  intros qlen i e1 e2 Herr.
  unfold subsumes. simpl.
  unfold subsumes_standard, position_is_final_for_subsumption. simpl.
  destruct ((negb (qlen <=? i)) && (qlen <=? i)) eqn:Hfinal;
    [destruct (qlen <=? i); discriminate|].
  rewrite Bool.andb_true_iff.
  split.
  - apply Nat.leb_le. exact Herr.
  - apply Nat.leb_le. rewrite abs_diff_self. lia.
Qed.

(** If a representative is behind the exact predecessor, the Standard
    subsumption slack pays for the delete steps needed to catch up to the
    predecessor's query index. *)
Lemma subsumes_standard_catch_up_delete_chain_same_index : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  term_index p <= i ->
  subsumes Standard qlen
    (std_pos i (num_errors p + (i - term_index p)))
    (std_pos i e) = true.
Proof.
  intros qlen [j e' sp] i e Hsub Hbehind.
  simpl in *.
  unfold subsumes in Hsub. simpl in Hsub.
  unfold subsumes_standard in Hsub.
  unfold position_is_final_for_subsumption in Hsub. simpl in Hsub.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  apply subsumes_standard_same_index_error_widen.
  unfold abs_diff in Hdist.
  destruct (j <=? i) eqn:Hj_le_i.
  - apply Nat.leb_le in Hj_le_i. lia.
  - apply Nat.leb_gt in Hj_le_i. lia.
Qed.

Lemma subsumes_standard_behind_error_slack : forall qlen p i e,
  subsumes Standard qlen p (std_pos i e) = true ->
  term_index p <= i ->
  num_errors p + (i - term_index p) <= e.
Proof.
  intros qlen [j e' sp] i e Hsub Hbehind.
  simpl in *.
  unfold subsumes in Hsub. simpl in Hsub.
  unfold subsumes_standard in Hsub.
  unfold position_is_final_for_subsumption in Hsub. simpl in Hsub.
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hdist].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hdist.
  unfold abs_diff in Hdist.
  destruct (j <=? i) eqn:Hj_le_i.
  - apply Nat.leb_le in Hj_le_i. lia.
  - apply Nat.leb_gt in Hj_le_i. lia.
Qed.

(** Variant of [term_index_minus_min_bounded] for a reachable target that is
    not itself retained in the state.  Standard antichain representatives may
    be behind an exact predecessor; after delete catch-up the exact predecessor
    is reachable in the same dictionary prefix but may be pruned from the state.
    The shared prefix bounds still keep it inside the [2*n] vector window from
    the retained state's minimum term index. *)
Lemma reachable_term_index_minus_state_min_bounded : forall
  query n dict_prefix positions init p anchor,
  (forall p0, In p0 positions -> position_reachable query n dict_prefix p0) ->
  (forall p0, In p0 positions -> is_special p0 = false) ->
  position_reachable query n dict_prefix p ->
  is_special p = false ->
  num_errors p <= n ->
  In anchor positions ->
  term_index anchor < init ->
  term_index p - fold_left Nat.min (map term_index positions) init <= 2 * n.
Proof.
  intros query n dict_prefix positions init p anchor Hreach Hspec
         Hp_reach Hp_spec Hp_err Hanchor Hanchor_lt.
  set (min_i := fold_left Nat.min (map term_index positions) init).
  assert (Hne : positions <> []).
  { intro Hempty. rewrite Hempty in Hanchor. contradiction. }
  assert (Hmin_lt_init : min_i < init).
  { unfold min_i.
    eapply Nat.le_lt_trans.
    - apply min_i_le_term_index. exact Hanchor.
    - exact Hanchor_lt. }
  destruct (list_has_min_term_index positions Hne) as [p_min [Hin_min Hmin_prop]].
  assert (Hmin_le_pmin : min_i <= term_index p_min).
  { unfold min_i. apply min_i_le_term_index. exact Hin_min. }
  assert (Hpmin_le_min : term_index p_min <= min_i).
  { assert (Hin_min_i : In min_i (map term_index positions)).
    { unfold min_i in Hmin_lt_init |- *.
      apply fold_left_min_in_list. exact Hmin_lt_init. }
    apply in_map_iff in Hin_min_i.
    destruct Hin_min_i as [q [Heq_q Hin_q]].
    specialize (Hmin_prop q Hin_q).
    lia. }
  assert (Hmin_eq : min_i = term_index p_min) by lia.
  assert (Hp_upper : term_index p <= length dict_prefix + n).
  { pose proof (reachable_term_index_upper_bound query n dict_prefix p Hp_reach)
      as Hupper.
    lia. }
  assert (Hpmin_err : num_errors p_min <= n).
  { apply reachable_implies_edit_distance with
      (query := query) (dict_prefix := dict_prefix).
    - apply Hreach. exact Hin_min.
    - apply Hspec. exact Hin_min. }
  assert (Hdict_lower : length dict_prefix <= term_index p_min + n).
  { pose proof (reachable_term_index_lower_bound query n dict_prefix p_min
                  (Hreach p_min Hin_min)) as Hlower.
    lia. }
  rewrite Hmin_eq.
  lia.
Qed.

(** Reachability is closed under a bounded chain of delete steps. *)
Lemma position_reachable_delete_chain : forall query n dict i e k,
  position_reachable query n dict (std_pos i e) ->
  i + k <= length query ->
  e + k <= n ->
  position_reachable query n dict (std_pos (i + k) (e + k)).
Proof.
  intros query n dict i e k Hreach.
  revert i e Hreach.
  induction k as [|k IH]; intros i e Hreach Hterm Herr.
  - replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    exact Hreach.
  - replace (i + S k) with (S (i + k)) by lia.
    replace (e + S k) with (S (e + k)) by lia.
    apply reach_delete.
    + apply (IH i e Hreach); lia.
    + lia.
    + lia.
Qed.

(** Inserting a Standard position represents the inserted position: either it
    survives insertion, or an existing antichain member subsumes it. *)
Lemma state_insert_represents_inserted_standard : forall qlen q s,
  algorithm s = Standard ->
  query_length s = qlen ->
  positions_subsume Standard qlen (positions (state_insert q s)) q.
Proof.
  intros qlen q s Halg Hqlen.
  unfold positions_subsume.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any Standard qlen q (positions s)) eqn:Hsub.
  - apply subsumed_by_any_correct in Hsub as [p' [Hin Hsub']].
    exists p'. split.
    + apply fold_right_sorted_insert_preserves_In. exact Hin.
    + exact Hsub'.
  - exists q. split.
    + apply fold_right_sorted_insert_preserves_In. simpl. left. reflexivity.
    + simpl. apply subsumes_standard_refl.
Qed.

(** Standard antichain insertion preserves representation of any previously
    represented position. If the new position removes the old witness, the new
    position subsumes that witness and Standard transitivity preserves the
    representation. *)
Lemma state_insert_preserves_positions_subsume_standard : forall qlen q s p,
  algorithm s = Standard ->
  query_length s = qlen ->
  positions_subsume Standard qlen (positions s) p ->
  positions_subsume Standard qlen (positions (state_insert q s)) p.
Proof.
  intros qlen q s p Halg Hqlen [r [Hin Hrsub]].
  unfold positions_subsume.
  unfold state_insert. simpl.
  rewrite Halg, Hqlen.
  unfold antichain_insert.
  destruct (subsumed_by_any Standard qlen q (positions s)) eqn:Hsub_q.
  - exists r. split.
    + apply fold_right_sorted_insert_preserves_In. exact Hin.
    + exact Hrsub.
  - destruct (subsumes Standard qlen q r) eqn:Hqsubr.
    + exists q. split.
      * apply fold_right_sorted_insert_preserves_In. simpl. left. reflexivity.
      * eapply subsumes_trans_standard; eauto.
    + exists r. split.
      * apply fold_right_sorted_insert_preserves_In.
        simpl. right.
        apply in_remove_subsumed_if_not_subsumed; [exact Hin | exact Hqsubr].
      * exact Hrsub.
Qed.

(** Folding [state_insert] over more Standard positions preserves any existing
    executable representation. *)
Lemma fold_state_insert_preserves_positions_subsume_standard : forall qlen inserts s p,
  algorithm s = Standard ->
  query_length s = qlen ->
  positions_subsume Standard qlen (positions s) p ->
  positions_subsume Standard qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hsub.
  - simpl. exact Hsub.
  - simpl.
    apply IH.
    + unfold state_insert. simpl. exact Halg.
    + unfold state_insert. simpl. exact Hqlen.
    + apply state_insert_preserves_positions_subsume_standard with
        (qlen := qlen); assumption.
Qed.

(** Folding [state_insert] over a list represents every inserted member. *)
Lemma fold_state_insert_represents_member_standard : forall qlen inserts s p,
  algorithm s = Standard ->
  query_length s = qlen ->
  In p inserts ->
  positions_subsume Standard qlen
    (positions (fold_left (fun s0 q => state_insert q s0) inserts s)) p.
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hin.
  - inversion Hin.
  - simpl in Hin |- *.
    destruct Hin as [Heq | Hin_rest].
    + subst q.
      apply fold_state_insert_preserves_positions_subsume_standard.
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * apply state_insert_represents_inserted_standard with (qlen := qlen);
          assumption.
    + apply (IH (state_insert q s) p).
      * unfold state_insert. simpl. exact Halg.
      * unfold state_insert. simpl. exact Hqlen.
      * exact Hin_rest.
Qed.

(** After a Standard transition, every member of the epsilon-closed candidate
    list is represented in the folded antichain output state. *)
Lemma transition_state_standard_represents_closed_position : forall s c query n s' p,
  transition_state Standard s c query n = Some s' ->
  In p
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  positions_subsume Standard (query_length s) (positions s') p.
Proof.
  intros s c query n s' p Htrans Hin.
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions Standard (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  fold min_i in Hin.
  fold cv in Hin.
  fold trans_positions in Hin.
  fold closed_positions in Hin.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  apply fold_state_insert_represents_member_standard.
  - unfold empty_state. reflexivity.
  - unfold empty_state. reflexivity.
  - exact Hin.
Qed.

Lemma transition_state_standard_represents_match_cv : forall s c query n s' i e,
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < query_length s ->
  cv_at
    (characteristic_vector c query
       (fold_left Nat.min (map term_index (positions s)) (query_length s))
       (2 * n + 6))
    (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = true ->
  positions_subsume Standard (query_length s) (positions s') (std_pos (S i) e).
Proof.
  intros s c query n s' i e Htrans Hin Hi_lt Hcv.
  apply (transition_state_standard_represents_closed_position s c query n s').
  - exact Htrans.
  - apply epsilon_closure_includes_input.
    unfold transition_state_positions.
    apply in_flat_map.
    exists (std_pos i e). split; [exact Hin |].
    unfold transition_position.
    apply transition_standard_produces_match.
    + exact Hi_lt.
    + change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin.
    + exact Hcv.
Qed.

Lemma transition_state_standard_represents_substitute_cv : forall s c query n s' i e,
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < query_length s ->
  e < n ->
  cv_at
    (characteristic_vector c query
       (fold_left Nat.min (map term_index (positions s)) (query_length s))
       (2 * n + 6))
    (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = false ->
  positions_subsume Standard (query_length s) (positions s') (std_pos (S i) (S e)).
Proof.
  intros s c query n s' i e Htrans Hin Hi_lt He_lt Hcv.
  apply (transition_state_standard_represents_closed_position s c query n s').
  - exact Htrans.
  - apply epsilon_closure_includes_input.
    unfold transition_state_positions.
    apply in_flat_map.
    exists (std_pos i e). split; [exact Hin |].
    unfold transition_position.
    apply transition_standard_produces_substitute.
    + exact Hi_lt.
    + change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin.
    + exact Hcv.
    + exact He_lt.
Qed.

Lemma transition_state_standard_represents_insert_exact : forall s c query n s' i e,
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  e < n ->
  positions_subsume Standard (query_length s) (positions s') (std_pos i (S e)).
Proof.
  intros s c query n s' i e Htrans Hin He_lt.
  apply (transition_state_standard_represents_closed_position s c query n s').
  - exact Htrans.
  - apply epsilon_closure_includes_input.
    unfold transition_state_positions.
    apply in_flat_map.
    exists (std_pos i e). split; [exact Hin |].
    unfold transition_position.
    apply transition_standard_produces_insert.
    exact He_lt.
Qed.

Lemma transition_state_standard_represents_match_exact : forall
  query n dict s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  positions_subsume Standard (query_length s) (positions s') (std_pos (S i) e).
Proof.
  intros query n dict s c s' i e Hqlen Hall_reach Hall_spec Htrans Hin Hlt Hnth.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - exact Hall_spec.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at (characteristic_vector c query min_i (2 * n + 6))
      (i - min_i) = true).
  {
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  eapply (transition_state_standard_represents_match_cv s c query n s' i e).
  - exact Htrans.
  - exact Hin.
  - rewrite Hqlen. exact Hlt.
  - exact Hcv.
Qed.

Lemma transition_state_standard_represents_substitute_exact : forall
  query n dict s c c' s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  positions_subsume Standard (query_length s) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n dict s c c' s' i e Hqlen Hall_reach Hall_spec
         Htrans Hin Hlt Hnth Hneq He_lt.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - exact Hall_spec.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = false).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    apply char_matches_at_false_iff.
    intros [q [Hnth_q Heq]].
    rewrite Hnth in Hnth_q.
    injection Hnth_q as Hq. subst q.
    apply Hneq. exact Heq. }
  eapply (transition_state_standard_represents_substitute_cv s c query n s' i e).
  - exact Htrans.
  - exact Hin.
  - rewrite Hqlen. exact Hlt.
  - exact He_lt.
  - exact Hcv.
Qed.

(** Exact Standard successors are present in the epsilon-closed transition
    candidate list before antichain pruning. These lemmas are the concrete
    counterpart to the [positions_subsume] representation lemmas above. *)
Lemma transition_state_standard_closed_insert_exact : forall
  s c query n i e,
  In (std_pos i e) (positions s) ->
  e < n ->
  In (std_pos i (S e))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin He_lt.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin |].
  unfold transition_position.
  apply transition_standard_produces_insert.
  exact He_lt.
Qed.

(** A represented predecessor with spare error budget has its insert successor
    represented after the next Standard transition. If the old representative
    is not at the exact predecessor index, its own insert successor carries the
    same Standard subsumption slack, and transitivity lifts the folded-state
    representation to the requested successor. *)
Lemma transition_state_standard_represents_insert_represented : forall
  s c query n s' i e,
  (forall p, In p (positions s) -> is_special p = false) ->
  transition_state Standard s c query n = Some s' ->
  positions_subsume Standard (query_length s) (positions s) (std_pos i e) ->
  e < n ->
  positions_subsume Standard (query_length s) (positions s') (std_pos i (S e)).
Proof.
  intros s c query n s' i e Hall_spec Htrans [p' [Hin' Hsub']] He_lt.
  assert (Hspec' : is_special p' = false).
  { apply Hall_spec. exact Hin'. }
  assert (Herr' : num_errors p' < n).
  { pose proof (subsumes_standard_errors (query_length s) p' (std_pos i e)
                 Hsub') as Herr_le.
    simpl in Herr_le. lia. }
  assert (Hp'_std : p' = std_pos (term_index p') (num_errors p')).
  { destruct p' as [j e' sp]. simpl in Hspec'. subst sp.
    unfold std_pos. simpl. reflexivity. }
  pose (p_ins := std_pos (term_index p') (S (num_errors p'))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp'_std in Hin'.
    apply transition_state_standard_closed_insert_exact; assumption. }
  destruct (transition_state_standard_represents_closed_position
              s c query n s' p_ins Htrans Hclosed) as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - unfold p_ins.
    apply subsumes_standard_insert_successor.
    exact Hsub'.
Qed.

Lemma transition_state_standard_closed_match_exact : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  In (std_pos (S i) e)
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hall_spec Hin Hlt Hnth.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - exact Hall_spec.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at (characteristic_vector c query min_i (2 * n + 6))
      (i - min_i) = true).
  {
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin |].
  unfold transition_position.
  apply transition_standard_produces_match.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    apply min_i_le_term_index. exact Hin.
  - exact Hcv.
Qed.

Lemma transition_state_standard_closed_substitute_exact : forall
  query n dict s c c' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  In (std_pos (S i) (S e))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c c' i e Hqlen Hall_reach Hall_spec
         Hin Hlt Hnth Hneq He_lt.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - exact Hall_spec.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = false).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    apply char_matches_at_false_iff.
    intros [q [Hnth_q Heq]].
    rewrite Hnth in Hnth_q.
    injection Hnth_q as Hq. subst q.
    apply Hneq. exact Heq. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin |].
  unfold transition_position.
  apply transition_standard_produces_substitute.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    apply min_i_le_term_index. exact Hin.
  - exact Hcv.
  - exact He_lt.
Qed.

Lemma transition_state_standard_closed_index_match_exact : forall
  s c query n i e j,
  In (std_pos i e) (positions s) ->
  i < query_length s ->
  e < n ->
  index_of_match
    (characteristic_vector c query
       (fold_left Nat.min (map term_index (positions s)) (query_length s))
       (2 * n + 6))
    (i - fold_left Nat.min (map term_index (positions s)) (query_length s))
    (Nat.min (n - e + 1)
       (length
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6)) -
        (i - fold_left Nat.min (map term_index (positions s)) (query_length s)))) =
    Some j ->
  In (std_pos (S (i + j)) (e + j))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e j Hin Hlt He_lt Hidx.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin |].
  unfold transition_position.
  apply transition_standard_produces_index_match.
  - exact Hlt.
  - exact He_lt.
  - exact Hidx.
Qed.

(** Represented match preservation for the same-index case.  Antichain pruning
    may keep a lower-error representative instead of the exact predecessor; if
    the representative is at the same query index, its exact match successor is
    generated and still represents the requested match successor. *)
Lemma transition_state_standard_represents_match_represented_same_index : forall
  query n dict s c s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep = i ->
  i < length query ->
  nth_error query i = Some c ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) e).
Proof.
  intros query n dict s c s' i e p_rep Hqlen Hall_reach Hall_spec Htrans
         Hin_rep Hsub_rep Hidx Hlt Hnth.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hin_std : In (std_pos i (num_errors p_rep)) (positions s)).
  { rewrite <- Hidx. rewrite <- Hp_rep_std. exact Hin_rep. }
  destruct (transition_state_standard_represents_match_exact
              query n dict s c s' i (num_errors p_rep)
              Hqlen Hall_reach Hall_spec Htrans Hin_std Hlt Hnth)
    as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  rewrite <- Hqlen.
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - rewrite Hqlen.
    apply subsumes_standard_match_successor_same_index with (p := p_rep).
    + exact Hsub_rep.
    + exact Hidx.
Qed.

(** Represented substitution preservation for the same-index case. *)
Lemma transition_state_standard_represents_substitute_represented_same_index : forall
  query n dict s c c' s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep = i ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n dict s c c' s' i e p_rep Hqlen Hall_reach Hall_spec Htrans
         Hin_rep Hsub_rep Hidx Hlt Hnth Hneq He_lt.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hin_std : In (std_pos i (num_errors p_rep)) (positions s)).
  { rewrite <- Hidx. rewrite <- Hp_rep_std. exact Hin_rep. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_errors (length query) p_rep (std_pos i e)
                 Hsub_rep) as Herr_le.
    simpl in Herr_le. lia. }
  destruct (transition_state_standard_represents_substitute_exact
              query n dict s c c' s' i (num_errors p_rep)
              Hqlen Hall_reach Hall_spec Htrans Hin_std Hlt Hnth Hneq
              Hrep_err_lt)
    as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  rewrite <- Hqlen.
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - rewrite Hqlen.
    apply subsumes_standard_substitute_successor_same_index with (p := p_rep).
    + exact Hsub_rep.
    + exact Hidx.
Qed.

(** Exact same-index containment is not preserved by Standard antichain
    insertion. The insert position [(0,1)] is reachable after consuming a
    matching one-character dictionary, but [(1,0)] subsumes it and remains as
    the only state position. *)
Lemma standard_exact_positions_contain_counterexample :
  exists query dict n s_mid p,
    automaton_run_from_initial Standard query n dict = Some s_mid /\
    position_reachable query n dict p /\
    is_special p = false /\
    num_errors p <= n /\
    ~ positions_contain (positions s_mid) p.
Proof.
  exists [default_char], [default_char], 1,
         (mkState [std_pos 1 0] Standard 1),
         (std_pos 0 1).
  split.
  - vm_compute. reflexivity.
  - split.
    + change [default_char] with ([] ++ [default_char]).
      apply reach_insert.
      * apply reach_initial.
      * simpl. lia.
    + split.
      * unfold std_pos. simpl. reflexivity.
      * split.
        -- simpl. lia.
        -- intros [p' [Hin Hsub]].
           simpl in Hin.
           destruct Hin as [Heq | []]. subst p'.
           unfold position_subsumes in Hsub. simpl in Hsub. lia.
Qed.

(** Regression check for Standard lookahead deletion: deleting a middle query
    character is accepted within distance one. *)
Lemma standard_middle_delete_accepts :
  let a := Ascii.ascii_of_nat 97 in
  let b := Ascii.ascii_of_nat 98 in
  let c := Ascii.ascii_of_nat 99 in
  lev_distance [a; b; c] [a; c] <= 1 /\
  automaton_accepts Standard [a; b; c] 1 [a; c] = true.
Proof.
  vm_compute.
  split; [lia | reflexivity].
Qed.

(** * Helper Lemmas for Containment Proofs *)

(** position_subsumes is reflexive *)
Lemma position_subsumes_refl : forall p,
  position_subsumes p p.
Proof.
  intros p.
  unfold position_subsumes.
  repeat split; lia.
Qed.

(** Containment via exact membership *)
Lemma positions_contain_In : forall ps p,
  In p ps ->
  positions_contain ps p.
Proof.
  intros ps p Hin.
  exists p. split.
  - exact Hin.
  - apply position_subsumes_refl.
Qed.

(** delete_step produces the expected position *)
Lemma delete_step_correct : forall i e n qlen,
  S i <= qlen ->
  e < n ->
  delete_step (std_pos i e) n qlen = Some (std_pos (S i) (S e)).
Proof.
  intros i e n qlen Hi He.
  unfold delete_step.
  (* std_pos i e has is_special = false, so first branch is skipped *)
  simpl is_special. simpl term_index. simpl num_errors.
  (* Case split on the combined boolean condition *)
  destruct ((S i <=? qlen) && (e <? n)) eqn:Hcond.
  - (* Condition is true *)
    reflexivity.
  - (* Condition is false, but this contradicts Hi and He *)
    apply andb_false_iff in Hcond.
    destruct Hcond as [Hleb | Hltb].
    + apply Nat.leb_nle in Hleb. lia.
    + apply Nat.ltb_nlt in Hltb. lia.
Qed.

(** Helper: If position p is in positions and delete_step p produces Some p',
    then p' is in new_positions of epsilon_closure_aux *)
Lemma delete_step_in_flat_map : forall p p' positions n qlen,
  In p positions ->
  delete_step p n qlen = Some p' ->
  In p' (flat_map (fun p0 => match delete_step p0 n qlen with
                             | Some p1 => [p1]
                             | None => []
                             end) positions).
Proof.
  intros p p' positions n qlen Hin Hdel.
  apply in_flat_map.
  exists p. split.
  - exact Hin.
  - rewrite Hdel. left. reflexivity.
Qed.

(** epsilon_closure_aux always returns a superset of its input *)
Lemma epsilon_closure_aux_includes_input : forall fuel positions n qlen,
  incl positions (epsilon_closure_aux positions n qlen fuel).
Proof.
  induction fuel as [| fuel' IH]; intros positions n qlen.
  - (* fuel = 0 *)
    simpl. unfold incl. auto.
  - (* fuel = S fuel' *)
    simpl.
    set (new := flat_map (fun p => match delete_step p n qlen with
                                   | Some p' => [p']
                                   | None => []
                                   end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + (* new is empty *)
      unfold incl. auto.
    + (* new is non-empty - recurse *)
      unfold incl. intros p Hp.
      apply IH.
      apply in_or_app. left. exact Hp.
Qed.

(** Epsilon closure is closed under delete_step.
    If p is in the epsilon closure and delete_step p = Some p',
    then p' is also in the epsilon closure.

    This follows from the fixpoint structure of epsilon_closure_aux:
    it iterates until no new positions can be generated. *)

(** Helper: epsilon_closure_aux with fuel >= 1 returns positions closed under
    one step of delete_step. For positions reachable by k delete steps,
    we need fuel >= k. *)
Lemma epsilon_closure_aux_extends_one_delete : forall fuel positions n qlen p p',
  In p positions ->
  delete_step p n qlen = Some p' ->
  fuel >= 1 ->
  In p' (epsilon_closure_aux positions n qlen fuel).
Proof.
  intros fuel positions n qlen p p' Hin Hdel Hfuel.
  destruct fuel as [| fuel'].
  - lia.
  - simpl.
    set (new := flat_map (fun p0 => match delete_step p0 n qlen with
                                    | Some p1 => [p1]
                                    | None => []
                                    end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + (* new is empty, but p' should be in new - contradiction *)
      exfalso.
      assert (Hcontra : In p' new).
      { unfold new. apply delete_step_in_flat_map with (p := p).
        - exact Hin.
        - exact Hdel. }
      destruct new as [| x xs].
      * inversion Hcontra.
      * discriminate Hnil.
    + (* new is non-empty *)
      apply epsilon_closure_aux_includes_input.
      apply in_or_app. right.
      unfold new. apply delete_step_in_flat_map with (p := p).
      * exact Hin.
      * exact Hdel.
Qed.

(** Key helper: Starting from a position in the original list, k delete steps
    can be captured in epsilon_closure_aux with fuel >= k. *)
Lemma epsilon_closure_aux_reaches_deletes : forall k fuel positions n qlen i e,
  In (std_pos i e) positions ->
  i + k <= qlen ->
  e + k <= n ->
  fuel >= k ->
  In (std_pos (i + k) (e + k)) (epsilon_closure_aux positions n qlen fuel).
Proof.
  induction k as [| k' IH]; intros fuel positions n qlen i e Hin Hbound_i Hbound_e Hfuel.
  - (* k = 0 *)
    replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    apply epsilon_closure_aux_includes_input.
    exact Hin.
  - (* k = S k' *)
    destruct fuel as [| fuel'].
    + lia.
    + (* fuel = S fuel' >= S k' *)
      simpl.
      set (new := flat_map (fun p => match delete_step p n qlen with
                                     | Some p' => [p']
                                     | None => []
                                     end) positions) in *.
      destruct (is_nil new) eqn:Hnil.
      * (* new is empty - but we need at least one delete step *)
        (* Since (std_pos i e) is in positions and we need delete_step to work,
           (std_pos (S i) (S e)) should be in new *)
        (* But new is empty, which means delete_step failed on all positions *)
        (* This happens when all positions have num_errors >= n or term_index > qlen *)
        (* But we have e + S k' <= n, so e + 1 <= n, so e < n *)
        (* And i + S k' <= qlen, so i + 1 <= qlen, so S i <= qlen *)
        (* So delete_step (std_pos i e) = Some (std_pos (S i) (S e)) should succeed *)
        exfalso.
        assert (Hdel : delete_step (std_pos i e) n qlen = Some (std_pos (S i) (S e))).
        { apply delete_step_correct; lia. }
        assert (Hin_new : In (std_pos (S i) (S e)) new).
        { unfold new. apply delete_step_in_flat_map with (p := std_pos i e); auto. }
        destruct new as [| x xs].
        -- inversion Hin_new.
        -- discriminate Hnil.
      * (* new is non-empty *)
        (* We need: (std_pos (i + S k') (e + S k')) in epsilon_closure_aux (positions ++ new) ... fuel' *)
        (* Use IH with (std_pos (S i) (S e)) in positions ++ new *)
        replace (i + S k') with (S i + k') by lia.
        replace (e + S k') with (S e + k') by lia.
        apply IH with (e := S e).
        -- (* (std_pos (S i) (S e)) is in positions ++ new *)
           apply in_or_app. right.
           unfold new.
           apply delete_step_in_flat_map with (p := std_pos i e).
           ++ exact Hin.
           ++ apply delete_step_correct; lia.
        -- lia.
        -- lia.
        -- lia.
Qed.

(** Epsilon closure captures delete chains.
    If position (i, e) is in the input, then all delete-reachable positions
    (i+k, e+k) are in the epsilon closure.

    The proof uses the fact that epsilon_closure with fuel = S n can capture
    up to n delete steps, and k <= n since e + k <= n and e >= 0. *)
Lemma epsilon_closure_reaches_deletes : forall positions n qlen i e k,
  In (std_pos i e) positions ->
  i + k <= qlen ->
  e + k <= n ->
  In (std_pos (i + k) (e + k)) (epsilon_closure positions n qlen).
Proof.
  intros positions n qlen i e k Hin Hbound_i Hbound_e.
  unfold epsilon_closure.
  apply epsilon_closure_aux_reaches_deletes with (e := e).
  - exact Hin.
  - exact Hbound_i.
  - exact Hbound_e.
  - (* fuel = S n >= k since e + k <= n and e >= 0 implies k <= n *)
    lia.
Qed.

(** A successful executable delete step is exactly a one-cell Standard
    delete move. Keeping this fact near the closure lemmas avoids depending
    on later state-level invariants. *)
Lemma delete_step_source : forall p n qlen p',
  delete_step p n qlen = Some p' ->
  p' = std_pos (S (term_index p)) (S (num_errors p)) /\
  is_special p = false /\
  S (term_index p) <= qlen /\
  num_errors p < n.
Proof.
  intros [i e sp] n qlen p' Hdel.
  unfold delete_step in Hdel.
  destruct sp; simpl in Hdel; [discriminate Hdel|].
  destruct qlen as [|qlen']; simpl in Hdel; [discriminate Hdel|].
  destruct ((i <=? qlen') && (e <? n)) eqn:Hcond; [| discriminate Hdel].
  inversion Hdel; subst; clear Hdel.
  apply andb_true_iff in Hcond.
  destruct Hcond as [Hi He].
  apply Nat.leb_le in Hi.
  apply Nat.ltb_lt in He.
  split; [reflexivity|].
  split; [reflexivity|].
  split.
  - simpl. lia.
  - simpl. lia.
Qed.

(** Every member of a bounded epsilon closure originates from one of the
    input positions by a finite Standard delete chain. *)
Lemma epsilon_closure_aux_source_deletes : forall fuel positions n qlen p,
  (forall p0, In p0 positions -> is_special p0 = false) ->
  In p (epsilon_closure_aux positions n qlen fuel) ->
  exists p0 k,
    In p0 positions /\
    p = std_pos (term_index p0 + k) (num_errors p0 + k).
Proof.
  induction fuel as [|fuel' IH]; intros positions n qlen p Hnonspec Hin.
  - simpl in Hin.
    exists p, 0.
    split; [exact Hin|].
    destruct p as [i e sp]. simpl in *.
    specialize (Hnonspec (mkPosition i e sp) Hin).
    destruct sp; simpl in Hnonspec; [discriminate|].
    replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    reflexivity.
  - simpl in Hin.
    set (new := flat_map (fun p0 : Position =>
                            match delete_step p0 n qlen with
                            | Some p' => [p']
                            | None => []
                            end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + exists p, 0.
      split; [exact Hin|].
      destruct p as [i e sp]. simpl in *.
      specialize (Hnonspec (mkPosition i e sp) Hin).
      destruct sp; simpl in Hnonspec; [discriminate|].
      replace (i + 0) with i by lia.
      replace (e + 0) with e by lia.
      reflexivity.
    + assert (Hnonspec_app : forall q,
          In q (positions ++ new) -> is_special q = false).
      { intros q Hq.
        apply in_app_or in Hq.
        destruct Hq as [Hq_old | Hq_new].
        - apply Hnonspec. exact Hq_old.
        - unfold new in Hq_new.
          apply in_flat_map in Hq_new.
          destruct Hq_new as [q0 [Hq0 Hdel_in]].
          destruct (delete_step q0 n qlen) as [q'|] eqn:Hdel.
          + destruct Hdel_in as [Heq | []]. subst q.
            destruct (delete_step_source q0 n qlen q' Hdel)
              as [Hq' [_ [_ _]]].
            rewrite Hq'. reflexivity.
          + contradiction. }
      destruct (IH (positions ++ new) n qlen p Hnonspec_app Hin)
        as [p0 [k [Hp0_in Hp]]].
      apply in_app_or in Hp0_in.
      destruct Hp0_in as [Hp0_old | Hp0_new].
      * exists p0, k. split; assumption.
      * unfold new in Hp0_new.
        apply in_flat_map in Hp0_new.
        destruct Hp0_new as [p_base [Hp_base Hdel_in]].
        destruct (delete_step p_base n qlen) as [p_del|] eqn:Hdel.
        -- destruct Hdel_in as [Hp0_eq | []]. subst p0.
           destruct (delete_step_source p_base n qlen p_del Hdel)
             as [Hp_del [_ [_ _]]].
           exists p_base, (S k).
           split; [exact Hp_base|].
           rewrite Hp.
           rewrite Hp_del.
           simpl. f_equal; lia.
        -- contradiction.
Qed.

Lemma epsilon_closure_source_deletes : forall positions n qlen p,
  (forall p0, In p0 positions -> is_special p0 = false) ->
  In p (epsilon_closure positions n qlen) ->
  exists p0 k,
    In p0 positions /\
    p = std_pos (term_index p0 + k) (num_errors p0 + k).
Proof.
  intros positions n qlen p Hnonspec Hin.
  unfold epsilon_closure in Hin.
  eapply epsilon_closure_aux_source_deletes; eauto.
Qed.

(** Epsilon closure is closed under any further valid delete chain from one
    of its own Standard members. The proof factors through the original
    source position, then reuses [epsilon_closure_reaches_deletes]. *)
Lemma epsilon_closure_member_reaches_deletes : forall positions n qlen p k,
  (forall p0, In p0 positions -> is_special p0 = false) ->
  In p (epsilon_closure positions n qlen) ->
  term_index p + k <= qlen ->
  num_errors p + k <= n ->
  In (std_pos (term_index p + k) (num_errors p + k))
     (epsilon_closure positions n qlen).
Proof.
  intros positions n qlen p k Hnonspec Hin Hterm Herr.
  destruct (epsilon_closure_source_deletes positions n qlen p Hnonspec Hin)
    as [p0 [k0 [Hp0_in Hp]]].
  rewrite Hp in Hterm, Herr |- *.
  simpl in Hterm, Herr |- *.
  replace (term_index p0 + k0 + k) with (term_index p0 + (k0 + k)) by lia.
  replace (num_errors p0 + k0 + k) with (num_errors p0 + (k0 + k)) by lia.
  eapply epsilon_closure_reaches_deletes
    with (i := term_index p0) (e := num_errors p0) (k := k0 + k).
  - assert (Hp0_std : p0 = std_pos (term_index p0) (num_errors p0)).
    { destruct p0 as [i0 e0 sp0]. simpl in *.
      specialize (Hnonspec (mkPosition i0 e0 sp0) Hp0_in).
      destruct sp0; simpl in Hnonspec; [discriminate|].
      reflexivity. }
    rewrite <- Hp0_std. exact Hp0_in.
  - lia.
  - lia.
Qed.

Lemma epsilon_closure_aux_source_deletes_nonspecial : forall fuel positions n qlen p,
  In p (epsilon_closure_aux positions n qlen fuel) ->
  is_special p = false ->
  exists p0 k,
    In p0 positions /\
    is_special p0 = false /\
    p = std_pos (term_index p0 + k) (num_errors p0 + k).
Proof.
  induction fuel as [|fuel' IH]; intros positions n qlen p Hin Hspec.
  - simpl in Hin.
    exists p, 0.
    repeat split; try exact Hin; try exact Hspec.
    destruct p as [i e sp]. simpl in *.
    destruct sp; simpl in Hspec; [discriminate|].
    replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    reflexivity.
  - simpl in Hin.
    set (new := flat_map (fun p0 : Position =>
                            match delete_step p0 n qlen with
                            | Some p' => [p']
                            | None => []
                            end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + exists p, 0.
      repeat split; try exact Hin; try exact Hspec.
      destruct p as [i e sp]. simpl in *.
      destruct sp; simpl in Hspec; [discriminate|].
      replace (i + 0) with i by lia.
      replace (e + 0) with e by lia.
      reflexivity.
    + destruct (IH (positions ++ new) n qlen p Hin Hspec)
        as [p0 [k [Hp0_in [Hp0_spec Hp]]]].
      apply in_app_or in Hp0_in.
      destruct Hp0_in as [Hp0_old | Hp0_new].
      * exists p0, k. repeat split; assumption.
      * unfold new in Hp0_new.
        apply in_flat_map in Hp0_new.
        destruct Hp0_new as [p_base [Hp_base Hdel_in]].
        destruct (delete_step p_base n qlen) as [p_del|] eqn:Hdel.
        -- destruct Hdel_in as [Hp0_eq | []]. subst p0.
           destruct (delete_step_source p_base n qlen p_del Hdel)
             as [Hp_del [Hp_base_spec [_ _]]].
           exists p_base, (S k).
           repeat split; try assumption.
           rewrite Hp.
           rewrite Hp_del.
           simpl. f_equal; lia.
        -- contradiction.
Qed.

Lemma epsilon_closure_member_reaches_deletes_nonspecial : forall positions n qlen p k,
  In p (epsilon_closure positions n qlen) ->
  is_special p = false ->
  term_index p + k <= qlen ->
  num_errors p + k <= n ->
  In (std_pos (term_index p + k) (num_errors p + k))
     (epsilon_closure positions n qlen).
Proof.
  intros positions n qlen p k Hin Hspec Hterm Herr.
  unfold epsilon_closure in Hin |- *.
  destruct (epsilon_closure_aux_source_deletes_nonspecial
              (S n) positions n qlen p Hin Hspec)
    as [p0 [k0 [Hp0_in [Hp0_spec Hp]]]].
  rewrite Hp in Hterm, Herr |- *.
  simpl in Hterm, Herr |- *.
  replace (term_index p0 + k0 + k) with
    (term_index p0 + (k0 + k)) by lia.
  replace (num_errors p0 + k0 + k) with
    (num_errors p0 + (k0 + k)) by lia.
  assert (Hp0_std_in : In (std_pos (term_index p0) (num_errors p0)) positions).
  { destruct p0 as [i0 e0 sp0]. simpl in *.
    destruct sp0; simpl in Hp0_spec; [discriminate|].
    exact Hp0_in. }
  exact (epsilon_closure_aux_reaches_deletes
           (k0 + k) (S n) positions n qlen
           (term_index p0) (num_errors p0)
           Hp0_std_in ltac:(lia) ltac:(lia) ltac:(lia)).
Qed.

Lemma initial_closed_delete_chain_represented : forall n qlen p k,
  In p (epsilon_closure [initial_position] n qlen) ->
  term_index p + k <= qlen ->
  num_errors p + k <= n ->
  positions_subsume Standard qlen
    (epsilon_closure [initial_position] n qlen)
    (std_pos (term_index p + k) (num_errors p + k)).
Proof.
  intros n qlen p k Hin Hterm Herr.
  apply positions_subsume_standard_refl_in.
  apply epsilon_closure_member_reaches_deletes.
  - intros p0 Hp0.
    simpl in Hp0.
    destruct Hp0 as [Hp0 | []].
    subst p0. reflexivity.
  - exact Hin.
  - exact Hterm.
  - exact Herr.
Qed.

(** * Containment Preservation Lemmas *)

(** Helper: position_subsumes respects delete chains.
    If p' subsumes p, then delete(p') subsumes delete(p). *)
Lemma position_subsumes_delete : forall i e e',
  e' <= e ->
  S i <= e' + (S i - i) + (S e' - e') ->
  position_subsumes (std_pos (S i) (S e')) (std_pos (S i) (S e)).
Proof.
  intros i e e' Hle _.
  unfold position_subsumes. simpl.
  repeat split; lia.
Qed.

(** Epsilon closure extends containment through delete chains.
    If the positions contain (i, e) via subsumption, then after epsilon closure,
    they contain (i+k, e+k) for valid k. *)
Lemma epsilon_closure_extends_contain : forall positions n qlen i e k,
  positions_contain positions (std_pos i e) ->
  i + k <= qlen ->
  e + k <= n ->
  positions_contain (epsilon_closure positions n qlen) (std_pos (i + k) (e + k)).
Proof.
  intros positions n qlen i e k Hcont Hbound_i Hbound_e.
  destruct Hcont as [p' [Hin' Hsub']].
  unfold position_subsumes in Hsub'.
  destruct Hsub' as [Hterm [Hspec Herr]].
  (* p' = std_pos i e' where e' <= e *)
  destruct p' as [i' e' b'].
  simpl in Hterm, Hspec, Herr.
  subst i'. destruct b' eqn:Hb'; try discriminate.
  (* p' = std_pos i e' with e' <= e *)
  (* By epsilon_closure_reaches_deletes, (i + k, e' + k) is in the epsilon closure *)
  assert (Hin_closed : In (std_pos (i + k) (e' + k)) (epsilon_closure positions n qlen)).
  { apply epsilon_closure_reaches_deletes with (e := e').
    - exact Hin'.
    - exact Hbound_i.
    - lia. (* e' + k <= e + k <= n *)
  }
  (* Now show (i + k, e' + k) subsumes (i + k, e + k) *)
  exists (std_pos (i + k) (e' + k)).
  split.
  - exact Hin_closed.
  - unfold position_subsumes. simpl. repeat split; lia.
Qed.

(** Helper: positions_contain is transitive with position_subsumes *)
Lemma positions_contain_trans : forall ps p1 p2,
  positions_contain ps p1 ->
  position_subsumes p1 p2 ->
  positions_contain ps p2.
Proof.
  intros ps p1 p2 [p' [Hin Hsub1]] Hsub2.
  exists p'.
  split.
  - exact Hin.
  - (* position_subsumes is transitive *)
    unfold position_subsumes in *.
    destruct Hsub1 as [Hi1 [Hs1 He1]].
    destruct Hsub2 as [Hi2 [Hs2 He2]].
    repeat split; try lia; try congruence.
Qed.

(** * Antichain Containment Lemmas *)

(** Key observation: The automaton's subsumes function allows term_index
    differences (|i1 - i2| <= e2 - e1), but position_subsumes requires
    same term_index. For completeness, we need a weaker property:

    If a final position (term_index >= qlen) is generated by transitions,
    then the antichain contains SOME final position.

    This is because non-final positions can only subsume final positions
    if they have strictly lower error counts, and error counts are bounded.
*)

(** NOTE: fold_state_insert_has_same_index was originally here but is not
    provable as stated because Standard subsumption allows |i1 - i2| <= e2 - e1,
    meaning a position at index i can be subsumed by one at a different index j.

    What we actually need for completeness is fold_state_insert_has_final:
    if the input contains a final position, the output contains a final position.
    This weaker property IS provable and sufficient (see fold_state_insert_has_final
    defined later in this file). *)

(** * Final Position Preservation for Standard Algorithm *)

(** Key insight for Standard algorithm:

    The Standard algorithm's antichain construction preserves final positions.
    This is because:
    1. If a final position (i, e) with i >= qlen is in closed_positions
    2. And some non-final (j, e') with j < qlen subsumes it
    3. Then i - j <= e - e', so e' < e
    4. After epsilon_closure, positions (j+k, e'+k) for k = 0..i-j are generated
    5. In particular, (i, e' + (i-j)) is generated with e' + (i-j) <= e
    6. This is final (term_index = i >= qlen) and either equals (i, e) or has lower errors

    Therefore, a non-final position can only subsume a final position if
    an equal or better final position is also present.
*)

(** Note: Proving that epsilon_closure output is closed under delete is complex
    because the closure computation is bounded by fuel = S n. For positions
    added during closure, we'd need to track the remaining fuel.

    The key property we need is different: if a non-final position in
    closed_positions can subsume a final position, then there's also
    a final position with lower or equal errors in closed_positions. *)

(** Helper: position_is_final equals position_is_final_for_subsumption *)
Lemma position_is_final_eq_subsumption : forall qlen p,
  position_is_final qlen p = position_is_final_for_subsumption qlen p.
Proof.
  intros qlen p.
  unfold position_is_final, position_is_final_for_subsumption.
  reflexivity.
Qed.

(** Simpler approach: Show that Standard antichain preserves existsb is_final.

    PROOF COMPLETED after the subsumption fix (Dec 2024):
    The key insight is that non-final positions CANNOT subsume final positions.
    Therefore, when inserting a non-final position, all final positions survive
    in remove_subsumed. *)
Lemma antichain_insert_preserves_final_standard : forall qlen p positions,
  existsb (position_is_final qlen) positions = true ->
  existsb (position_is_final qlen) (antichain_insert Standard qlen p positions) = true.
Proof.
  intros qlen p positions Hfinal.
  unfold antichain_insert.
  destruct (subsumed_by_any Standard qlen p positions) eqn:Hsub.
  - (* p is subsumed, positions unchanged *)
    exact Hfinal.
  - (* p is inserted, some positions may be removed *)
    (* Case 1: p itself is final - then result has p which is final *)
    (* Case 2: p is not final - need to show the final position wasn't removed *)
    destruct (position_is_final qlen p) eqn:Hp_final.
    + (* p is final, so result has a final position (p) *)
      apply existsb_exists. exists p. split.
      * simpl. left. reflexivity.
      * exact Hp_final.
    + (* p is not final - but non-final cannot subsume final! *)
      (* So all final positions in positions survive in remove_subsumed *)
      simpl. apply orb_true_intro. right.
      (* Since position_is_final = position_is_final_for_subsumption, we can
         directly use the lemma from AntiChain *)
      apply remove_subsumed_preserves_existsb_final_subsumption.
      * (* p is non-final *)
        unfold position_is_final_for_subsumption, position_is_final in Hp_final.
        exact Hp_final.
      * (* positions has a final position *)
        unfold position_is_final_for_subsumption, position_is_final in Hfinal.
        exact Hfinal.
Qed.

(** The full preservation lemma requires knowing that positions came from
    epsilon_closure, which ensures that if a non-final position can subsume
    a final position, there's also a final position with lower or equal errors. *)

(** Generalized version: antichain_insert preserves final positions for ANY algorithm
    because all algorithms have the non-final-cannot-subsume-final property. *)
Lemma antichain_insert_preserves_final : forall alg qlen p positions,
  existsb (position_is_final qlen) positions = true ->
  existsb (position_is_final qlen) (antichain_insert alg qlen p positions) = true.
Proof.
  intros alg qlen p positions Hfinal.
  unfold antichain_insert.
  destruct (subsumed_by_any alg qlen p positions) eqn:Hsub.
  - (* p is subsumed, positions unchanged *)
    exact Hfinal.
  - (* p is inserted, some positions may be removed *)
    destruct (position_is_final qlen p) eqn:Hp_final.
    + (* p is final *)
      apply existsb_exists. exists p. split.
      * simpl. left. reflexivity.
      * exact Hp_final.
    + (* p is not final - use the key property *)
      simpl. apply orb_true_intro. right.
      apply remove_subsumed_preserves_existsb_final_subsumption.
      * unfold position_is_final_for_subsumption, position_is_final in Hp_final.
        exact Hp_final.
      * unfold position_is_final_for_subsumption, position_is_final in Hfinal.
        exact Hfinal.
Qed.

(** Folding antichain_insert from accumulator with final position *)
Lemma fold_antichain_insert_from_final_acc : forall alg qlen positions acc,
  existsb (position_is_final qlen) acc = true ->
  existsb (position_is_final qlen)
    (fold_left (fun acc' p => antichain_insert alg qlen p acc') positions acc) = true.
Proof.
  intros alg qlen positions.
  induction positions as [| p rest IH]; intros acc Hfinal.
  - (* Empty list *)
    simpl. exact Hfinal.
  - (* p :: rest *)
    simpl. apply IH.
    apply antichain_insert_preserves_final. exact Hfinal.
Qed.

(** Helper: inserting a final position produces a final accumulator *)
Lemma antichain_insert_final_produces_final : forall alg qlen p acc,
  position_is_final qlen p = true ->
  existsb (position_is_final qlen) (antichain_insert alg qlen p acc) = true.
Proof.
  intros alg qlen p acc Hp_final.
  unfold antichain_insert.
  destruct (subsumed_by_any alg qlen p acc) eqn:Hsub.
  - (* p is subsumed by something in acc *)
    apply subsumed_by_any_correct in Hsub.
    destruct Hsub as [p' [Hin' Hsub']].
    apply existsb_exists. exists p'. split.
    + exact Hin'.
    + destruct (position_is_final qlen p') eqn:Hp'_final.
      * reflexivity.
      * (* contradiction: non-final p' subsumes final p *)
        exfalso.
        assert (Hfalse : subsumes alg qlen p' p = false).
        { apply non_final_cannot_subsume_final.
          - unfold position_is_final_for_subsumption, position_is_final in Hp'_final.
            exact Hp'_final.
          - unfold position_is_final_for_subsumption, position_is_final in Hp_final.
            exact Hp_final. }
        rewrite Hsub' in Hfalse. discriminate.
  - (* p not subsumed *)
    apply existsb_exists. exists p. split.
    + simpl. left. reflexivity.
    + exact Hp_final.
Qed.

(** Folding antichain_insert preserves final positions (main lemma) *)
Lemma fold_antichain_insert_preserves_final : forall alg qlen positions acc,
  existsb (position_is_final qlen) positions = true ->
  existsb (position_is_final qlen)
    (fold_left (fun acc' p => antichain_insert alg qlen p acc') positions acc) = true.
Proof.
  intros alg qlen positions.
  induction positions as [| p rest IH]; intros acc Hfinal.
  - (* Empty list - contradiction *)
    simpl in Hfinal. discriminate.
  - (* p :: rest *)
    simpl.
    rewrite existsb_exists in Hfinal.
    destruct Hfinal as [q [Hin Hq_final]].
    simpl in Hin. destruct Hin as [Heq | Hin'].
    + (* q = p, so p is final *)
      subst q.
      (* After inserting p, the accumulator has a final position *)
      (* Then the rest of the fold preserves it *)
      apply fold_antichain_insert_from_final_acc.
      apply antichain_insert_final_produces_final. exact Hq_final.
    + (* q in rest *)
      apply IH.
      apply existsb_exists. exists q. split; [exact Hin' | exact Hq_final].
Qed.

(** Helper: state_insert preserves final positions *)
Lemma state_insert_preserves_final : forall p s,
  existsb (position_is_final (query_length s)) (positions s) = true ->
  existsb (position_is_final (query_length s)) (positions (state_insert p s)) = true.
Proof.
  intros p s Hfinal.
  unfold state_insert. simpl.
  (* state_insert = fold_right sorted_insert [] (antichain_insert ...) *)
  apply fold_right_sorted_insert_preserves_existsb.
  apply antichain_insert_preserves_final.
  exact Hfinal.
Qed.

(** Helper: state_insert of a final position produces a state with a final position *)
Lemma state_insert_final_produces_final : forall p s,
  position_is_final (query_length s) p = true ->
  existsb (position_is_final (query_length s)) (positions (state_insert p s)) = true.
Proof.
  intros p s Hp_final.
  unfold state_insert. simpl.
  apply fold_right_sorted_insert_preserves_existsb.
  unfold antichain_insert.
  destruct (subsumed_by_any (algorithm s) (query_length s) p (positions s)) eqn:Hsub.
  - (* p is subsumed by something in positions s *)
    (* The subsumer must be final (since non-final cannot subsume final) *)
    apply subsumed_by_any_correct in Hsub.
    destruct Hsub as [p' [Hin' Hsub']].
    apply existsb_exists. exists p'. split.
    + exact Hin'.
    + (* p' must be final since p is final and p' subsumes p *)
      destruct (position_is_final (query_length s) p') eqn:Hp'_final.
      * reflexivity.
      * (* contradiction: non-final p' subsumes final p *)
        exfalso.
        assert (Hfalse : subsumes (algorithm s) (query_length s) p' p = false).
        { apply non_final_cannot_subsume_final.
          - unfold position_is_final_for_subsumption, position_is_final in Hp'_final.
            exact Hp'_final.
          - unfold position_is_final_for_subsumption, position_is_final in Hp_final.
            exact Hp_final. }
        rewrite Hsub' in Hfalse. discriminate.
  - (* p is not subsumed, so p is in the result *)
    apply existsb_exists. exists p. split.
    + simpl. left. reflexivity.
    + exact Hp_final.
Qed.

(** Helper: fold state_insert preserves final positions with explicit qlen tracking *)
Lemma fold_state_insert_preserves_final_aux : forall qlen pos_list s,
  query_length s = qlen ->
  existsb (position_is_final qlen) (Automaton.State.positions s) = true ->
  existsb (position_is_final qlen)
    (Automaton.State.positions (fold_left (fun s' p => state_insert p s') pos_list s)) = true.
Proof.
  intros qlen pos_list.
  induction pos_list as [| p rest IH]; intros s Hqlen Hfinal.
  - simpl. exact Hfinal.
  - simpl.
    apply IH.
    + unfold state_insert. simpl. exact Hqlen.
    + (* Goal: existsb (position_is_final qlen) (positions (state_insert p s)) = true *)
      (* We know: Hfinal : existsb (position_is_final qlen) (positions s) = true *)
      (* Use state_insert_preserves_final, which uses query_length s *)
      rewrite <- Hqlen.
      (* Goal: existsb (position_is_final (query_length s)) (positions (state_insert p s)) = true *)
      apply state_insert_preserves_final.
      rewrite Hqlen. exact Hfinal.
Qed.

(** Query length is preserved through fold_left state_insert *)
Lemma fold_state_insert_preserves_query_length : forall pos_list s,
  query_length (fold_left (fun s' p => state_insert p s') pos_list s) = query_length s.
Proof.
  intros pos_list.
  induction pos_list as [| p rest IH]; intros s.
  - simpl. reflexivity.
  - simpl. rewrite IH. unfold state_insert. simpl. reflexivity.
Qed.

(** Helper: fold state_insert on closed_positions produces a state with a final position
    when closed_positions contains a final position. Uses strong induction. *)
Lemma fold_state_insert_has_final : forall alg qlen closed_positions init_state,
  query_length init_state = qlen ->
  algorithm init_state = alg ->
  existsb (position_is_final qlen) closed_positions = true ->
  existsb (position_is_final qlen)
    (positions (fold_left (fun s p => state_insert p s) closed_positions init_state)) = true.
Proof.
  intros alg qlen closed_positions.
  induction closed_positions as [| p rest IH]; intros init_state Hqlen Halg Hfinal.
  - (* empty list - contradiction since no final position *)
    simpl in Hfinal. discriminate.
  - simpl.
    simpl in Hfinal.
    destruct (position_is_final qlen p) eqn:Hp_final.
    + (* p is final - insert it and preserve through rest *)
      assert (Hqlen_after : query_length (state_insert p init_state) = qlen).
      { unfold state_insert. simpl. exact Hqlen. }
      assert (Hhas_final : existsb (position_is_final qlen)
                             (positions (state_insert p init_state)) = true).
      { rewrite <- Hqlen.
        apply state_insert_final_produces_final.
        rewrite Hqlen. exact Hp_final. }
      apply (fold_state_insert_preserves_final_aux qlen rest (state_insert p init_state)).
      * exact Hqlen_after.
      * exact Hhas_final.
    + (* p is not final - final position must be in rest *)
      apply IH.
      * unfold state_insert. simpl. exact Hqlen.
      * unfold state_insert. simpl. exact Halg.
      * exact Hfinal.
Qed.

Lemma fold_state_insert_accepting : forall alg qlen closed_positions,
  existsb (position_is_final qlen) closed_positions = true ->
  state_is_final (fold_left (fun s p => state_insert p s) closed_positions (empty_state alg qlen)) = true.
Proof.
  intros alg qlen closed_positions Hfinal.
  unfold state_is_final.
  (* Convert query_length of fold result to qlen *)
  assert (Hqlen_fold : query_length (fold_left (fun s p => state_insert p s)
                                      closed_positions (empty_state alg qlen)) = qlen).
  { rewrite fold_state_insert_preserves_query_length.
    unfold empty_state. simpl. reflexivity. }
  rewrite Hqlen_fold.
  apply (fold_state_insert_has_final alg qlen closed_positions (empty_state alg qlen)).
  - unfold empty_state. simpl. reflexivity.
  - unfold empty_state. simpl. reflexivity.
  - exact Hfinal.
Qed.

(** ** Weaker Completeness Properties

    The full `positions_contain` invariant is not preserved through antichain
    building because positions at different indices can subsume each other.
    However, for completeness, we only need weaker properties:

    1. The automaton doesn't go dead (closed_positions non-empty) when
       there's a reachable position with bounded errors
    2. If a FINAL position is reachable, the state contains SOME final position

    These weaker properties are sufficient for completeness proofs.
*)

(** Key property: transition always produces at least an insert position
    when the predecessor has errors < n. This ensures the automaton never
    goes dead for reachable positions with bounded errors. *)
Lemma transition_produces_insert_bounded : forall p cv min_i n qlen,
  is_special p = false ->
  num_errors p < n ->
  exists p', In p' (transition_position_standard p cv min_i n qlen) /\ num_errors p' <= n.
Proof.
  intros p cv min_i n qlen Hnonspec Hbound.
  (* Insert transition is always available when num_errors < n *)
  destruct p as [i e is_spec]. simpl in *.
  subst is_spec.
  (* Now p = mkPosition i e false = std_pos i e *)
  exists (std_pos i (S e)).
  split.
  - (* std_pos i e = mkPosition i e false definitionally *)
    change (mkPosition i e false) with (std_pos i e).
    apply transition_standard_produces_insert.
    exact Hbound.
  - simpl. lia.
Qed.

(** Stronger version: insert produces exactly e+1 errors *)
Lemma transition_produces_insert_exact : forall p cv min_i n qlen,
  is_special p = false ->
  num_errors p < n ->
  exists p', In p' (transition_position_standard p cv min_i n qlen) /\
             num_errors p' = S (num_errors p) /\ is_special p' = false.
Proof.
  intros p cv min_i n qlen Hnonspec Hbound.
  destruct p as [i e is_spec]. simpl in *. subst is_spec.
  exists (std_pos i (S e)).
  split; [| split].
  - change (mkPosition i e false) with (std_pos i e).
    apply transition_standard_produces_insert. exact Hbound.
  - unfold std_pos. simpl. reflexivity.
  - unfold std_pos. simpl. reflexivity.
Qed.

(** If a state has a position with errors < n, transition_state_positions is non-empty *)
Lemma transition_state_positions_nonempty_standard : forall positions cv min_i n qlen,
  (exists p, In p positions /\ is_special p = false /\ num_errors p < n) ->
  transition_state_positions Standard positions cv min_i n qlen <> [].
Proof.
  intros positions cv min_i n qlen [p [Hin [Hspec Herr]]].
  unfold transition_state_positions.
  intro Hempty.
  (* flat_map on non-empty list with non-empty function output is non-empty *)
  destruct (transition_produces_insert_bounded p cv min_i n qlen Hspec Herr)
    as [p' [Hin' _]].
  (* p' is in transition_position_standard p, so it's in flat_map result *)
  assert (Hin_flat : In p' (flat_map (fun p0 => transition_position Standard p0 cv min_i n qlen) positions)).
  { apply in_flat_map. exists p. split.
    - exact Hin.
    - (* transition_position Standard p = transition_position_standard p *)
      unfold transition_position. exact Hin'. }
  rewrite Hempty in Hin_flat. contradiction.
Qed.

(** Epsilon closure on non-empty list with bounded errors is non-empty *)
Lemma epsilon_closure_nonempty_bounded : forall positions n qlen,
  positions <> [] ->
  (forall p, In p positions -> num_errors p <= n) ->
  epsilon_closure positions n qlen <> [].
Proof.
  intros positions n qlen Hnonempty Hbound.
  unfold epsilon_closure.
  (* epsilon_closure_aux always includes the original positions *)
  destruct positions as [| p rest].
  - contradiction.
  - (* Non-empty input gives non-empty output *)
    intro Hempty.
    assert (Hin : In p (epsilon_closure_aux (p :: rest) n qlen (S n))).
    { apply epsilon_closure_aux_includes_input. left. reflexivity. }
    rewrite Hempty in Hin. contradiction.
Qed.

(** If a state has a non-special position with errors < n, transition_state returns Some *)
Lemma transition_state_not_dead_standard : forall s c query n,
  algorithm s = Standard ->
  query_length s = length query ->
  (exists p, In p (positions s) /\ is_special p = false /\ num_errors p < n) ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros s c query n Halg Hqlen [p [Hin [Hspec Herr]]].
  unfold transition_state.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  set (cv := characteristic_vector c query min_i (2 * n + 6)).
  set (trans_pos := transition_state_positions Standard (positions s) cv min_i n (query_length s)).
  set (closed_pos := epsilon_closure trans_pos n (query_length s)).
  (* Show trans_pos is non-empty *)
  assert (Htrans_nonempty : trans_pos <> []).
  { apply transition_state_positions_nonempty_standard.
    exists p. split; [exact Hin | split; [exact Hspec | exact Herr]]. }
  (* Show closed_pos is non-empty *)
  assert (Hclosed_nonempty : closed_pos <> []).
  { unfold closed_pos.
    destruct trans_pos as [| tp rest] eqn:Htrans_eq.
    - contradiction.
    - (* Non-empty trans_pos gives non-empty epsilon_closure *)
      intro Hempty.
      unfold epsilon_closure in Hempty.
      assert (Hin_closure : In tp (epsilon_closure_aux (tp :: rest) n (query_length s) (S n))).
      { apply epsilon_closure_aux_includes_input. left. reflexivity. }
      rewrite Hempty in Hin_closure. contradiction. }
  (* Therefore is_nil closed_pos = false *)
  destruct closed_pos as [| cp crest] eqn:Hclosed_eq.
  - contradiction.
  - exists (fold_left (fun s0 p0 => state_insert p0 s0) (cp :: crest) (empty_state Standard (query_length s))).
    reflexivity.
Qed.

(** If a represented Standard position exactly matches the consumed character,
    the transition is non-dead even when its error count is already [n]. *)
Lemma transition_state_not_dead_standard_match : forall query n dict s c p,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  In p (positions s) ->
  is_special p = false ->
  term_index p < length query ->
  nth_error query (term_index p) = Some c ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros query n dict s c p Hqlen Hall_reach Hall_spec Hin Hspec Hlt Hnth.
  unfold transition_state.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  set (cv := characteristic_vector c query min_i (2 * n + 6)).
  set (trans_pos := transition_state_positions Standard (positions s) cv min_i n (query_length s)).
  set (closed_pos := epsilon_closure trans_pos n (query_length s)).
  assert (Hmin_le : min_i <= term_index p).
  { unfold min_i. apply min_i_le_term_index. exact Hin. }
  assert (Hoffset_bound : term_index p - min_i < 2 * n + 6).
  { assert (Hbounded : term_index p - min_i <= 2 * n).
    { unfold min_i.
      apply term_index_minus_min_bounded with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - exact Hall_spec.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    lia. }
  assert (Hcv : cv_at cv (term_index p - min_i) = true).
  { unfold cv.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hsum : min_i + (term_index p - min_i) = term_index p) by lia.
    rewrite Hsum.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  assert (Hin_trans :
    In (std_pos (S (term_index p)) (num_errors p)) trans_pos).
  { unfold trans_pos, transition_state_positions.
    apply in_flat_map.
    exists p. split; [exact Hin |].
    unfold transition_position.
    assert (Hp_std : p = std_pos (term_index p) (num_errors p)).
    { destruct p as [ti ne sp].
      unfold is_special in Hspec. simpl in Hspec.
      destruct sp; try discriminate.
      unfold std_pos. simpl. reflexivity. }
    rewrite Hp_std.
    apply transition_standard_produces_match.
    - rewrite Hqlen. exact Hlt.
    - exact Hmin_le.
    - exact Hcv. }
  assert (Htrans_nonempty : trans_pos <> []).
  { intro Hempty. rewrite Hempty in Hin_trans. contradiction. }
  assert (Hclosed_nonempty : closed_pos <> []).
  { unfold closed_pos.
    destruct trans_pos as [| tp rest] eqn:Htrans_eq.
    - contradiction.
    - intro Hempty.
      unfold epsilon_closure in Hempty.
      assert (Hin_closure : In tp (epsilon_closure_aux (tp :: rest) n (query_length s) (S n))).
      { apply epsilon_closure_aux_includes_input. left. reflexivity. }
      rewrite Hempty in Hin_closure. contradiction. }
  destruct closed_pos as [| cp crest] eqn:Hclosed_eq.
  - contradiction.
  - exists (fold_left (fun s0 p0 => state_insert p0 s0) (cp :: crest) (empty_state Standard (query_length s))).
    reflexivity.
Qed.

(** If a Standard subsumer represents a different query index, it must have
    strictly fewer errors; the saved error budget pays for the index offset. *)
Lemma subsumes_standard_diff_index_lt_errors : forall qlen p' i e,
  subsumes Standard qlen p' (std_pos i e) = true ->
  term_index p' <> i ->
  num_errors p' < e.
Proof.
  intros qlen [i' e' sp'] i e Hsub Hdiff.
  simpl in Hdiff.
  unfold subsumes in Hsub. simpl in Hsub.
  unfold subsumes_standard in Hsub. simpl in Hsub.
  destruct ((negb (position_is_final_for_subsumption qlen {| term_index := i'; num_errors := e'; is_special := sp' |}))
            && (position_is_final_for_subsumption qlen (std_pos i e))) eqn:Hfinal;
    [discriminate|].
  rewrite Bool.andb_true_iff in Hsub.
  destruct Hsub as [Herr Hoffset].
  apply Nat.leb_le in Herr.
  apply Nat.leb_le in Hoffset.
  assert (Habs_pos : 0 < abs_diff i' i).
  { unfold abs_diff.
    destruct (i' <=? i) eqn:Hle.
    - apply Nat.leb_le in Hle. lia.
    - apply Nat.leb_gt in Hle. lia. }
  assert (Herr_gap : 0 < e - e') by lia.
  assert (Hneq_err : e' <> e).
  { intro Heq. subst. rewrite Nat.sub_diag in Herr_gap. lia. }
  destruct (Nat.lt_ge_cases e' e) as [Hlt | Hge].
  - exact Hlt.
  - assert (Heq : e' = e) by lia. contradiction.
Qed.

(** Ahead represented match preservation.  A representative that has advanced
    past the exact predecessor cannot necessarily match the same query
    character, but it can always consume the dictionary character by insertion;
    Standard subsumption arithmetic then shows that insert successor represents
    the requested match successor. *)
Lemma transition_state_standard_represents_match_represented_ahead_insert : forall
  query n s c s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  (forall p0, In p0 (positions s) -> term_index p0 <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i < term_index p_rep ->
  e <= n ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) e).
Proof.
  intros query n s c s' i e p_rep Hqlen Hall_spec Hstate_bound Htrans
         Hin_rep Hsub_rep Hahead Herr.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_diff_index_lt_errors
                  (length query) p_rep i e Hsub_rep ltac:(lia)) as Hlt.
    lia. }
  pose (p_ins := std_pos (term_index p_rep) (S (num_errors p_rep))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp_rep_std in Hin_rep.
    apply transition_state_standard_closed_insert_exact; assumption. }
  destruct (transition_state_standard_represents_closed_position
              s c query n s' p_ins Htrans Hclosed) as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  rewrite <- Hqlen.
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - unfold p_ins. rewrite Hqlen.
    apply subsumes_standard_match_successor_ahead_insert.
    + exact Hsub_rep.
    + exact Hahead.
    + apply Hstate_bound. exact Hin_rep.
Qed.

(** Ahead represented substitution preservation via the same insert successor. *)
Lemma transition_state_standard_represents_substitute_represented_ahead_insert : forall
  query n s c s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  (forall p0, In p0 (positions s) -> term_index p0 <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i < term_index p_rep ->
  e < n ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n s c s' i e p_rep Hqlen Hall_spec Hstate_bound Htrans
         Hin_rep Hsub_rep Hahead He_lt.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_errors (length query) p_rep (std_pos i e)
                 Hsub_rep) as Herr_le.
    simpl in Herr_le. lia. }
  pose (p_ins := std_pos (term_index p_rep) (S (num_errors p_rep))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp_rep_std in Hin_rep.
    apply transition_state_standard_closed_insert_exact; assumption. }
  destruct (transition_state_standard_represents_closed_position
              s c query n s' p_ins Htrans Hclosed) as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  rewrite <- Hqlen.
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - unfold p_ins. rewrite Hqlen.
    apply subsumes_standard_substitute_successor_ahead_insert.
    + exact Hsub_rep.
    + exact Hahead.
    + apply Hstate_bound. exact Hin_rep.
Qed.

(** Combined represented match preservation for representatives that are not
    behind the exact predecessor. *)
Lemma transition_state_standard_represents_match_represented_not_behind : forall
  query n dict s c s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  (forall p0, In p0 (positions s) -> term_index p0 <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i <= term_index p_rep ->
  e <= n ->
  i < length query ->
  nth_error query i = Some c ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) e).
Proof.
  intros query n dict s c s' i e p_rep Hqlen Hall_reach Hall_spec
         Hstate_bound Htrans Hin_rep Hsub_rep Hnot_behind Herr Hlt Hnth.
  destruct (Nat.eq_dec (term_index p_rep) i) as [Hsame | Hneq].
  - eapply transition_state_standard_represents_match_represented_same_index; eauto.
  - eapply (transition_state_standard_represents_match_represented_ahead_insert
              query n s c s' i e p_rep); eauto.
    lia.
Qed.

(** Combined represented substitution preservation for representatives that are
    not behind the exact predecessor. *)
Lemma transition_state_standard_represents_substitute_represented_not_behind : forall
  query n dict s c c' s' i e p_rep,
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  (forall p0, In p0 (positions s) -> term_index p0 <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i <= term_index p_rep ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  positions_subsume Standard (length query) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n dict s c c' s' i e p_rep Hqlen Hall_reach Hall_spec
         Hstate_bound Htrans Hin_rep Hsub_rep Hnot_behind Hlt Hnth Hneq_ch He_lt.
  destruct (Nat.eq_dec (term_index p_rep) i) as [Hsame | Hneq_idx].
  - eapply transition_state_standard_represents_substitute_represented_same_index; eauto.
  - eapply (transition_state_standard_represents_substitute_represented_ahead_insert
              query n s c s' i e p_rep); eauto.
    lia.
Qed.

(** A represented predecessor with spare error budget is enough to make the
    next Standard transition non-dead. *)
Lemma transition_state_not_dead_standard_represented_error_lt : forall
  query n s c i e,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  positions_subsume Standard (length query) (positions s) (std_pos i e) ->
  e < n ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros query n s c i e Halg Hqlen Hall_spec Hrep Herr.
  destruct Hrep as [p' [Hin' Hsub']].
  assert (Hspec' : is_special p' = false).
  { apply Hall_spec. exact Hin'. }
  assert (Herr' : num_errors p' < n).
  { unfold subsumes in Hsub'. simpl in Hsub'.
    apply subsumes_standard_errors in Hsub'. simpl in Hsub'. lia. }
  apply (transition_state_not_dead_standard s c query n Halg Hqlen).
  exists p'. repeat split; assumption.
Qed.

(** A represented exact-match predecessor is enough to make the next Standard
    transition non-dead, even when the represented position has no spare error
    budget. If the representative is at a different query index, Standard
    subsumption guarantees it has strictly fewer errors and can take an insert
    transition instead. *)
Lemma transition_state_not_dead_standard_represented_match : forall
  query n dict s c i e,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall p0, In p0 (positions s) -> position_reachable query n dict p0) ->
  (forall p0, In p0 (positions s) -> is_special p0 = false) ->
  positions_subsume Standard (length query) (positions s) (std_pos i e) ->
  e <= n ->
  i < length query ->
  nth_error query i = Some c ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros query n dict s c i e Halg Hqlen Hall_reach Hall_spec Hrep Herr Hlt Hnth.
  destruct Hrep as [p' [Hin' Hsub']].
  assert (Hspec' : is_special p' = false).
  { apply Hall_spec. exact Hin'. }
  destruct (Nat.eq_dec (term_index p') i) as [Hsame | Hdiff].
  - apply (transition_state_not_dead_standard_match query n dict s c p').
    + exact Hqlen.
    + exact Hall_reach.
    + exact Hall_spec.
    + exact Hin'.
    + exact Hspec'.
    + rewrite Hsame. exact Hlt.
    + rewrite Hsame. exact Hnth.
  - assert (Herr_lt : num_errors p' < n).
    { assert (Hlt_err : num_errors p' < e).
      { eapply subsumes_standard_diff_index_lt_errors; eauto. }
      lia. }
    apply (transition_state_not_dead_standard s c query n Halg Hqlen).
    exists p'. repeat split; assumption.
Qed.

(** Helper: antichain_insert preserves min error bound.
    If positions has a position with errors <= e, and p has errors <= e,
    then antichain_insert result has a position with errors <= e. *)
Lemma antichain_insert_preserves_min_error : forall qlen p positions e,
  num_errors p <= e ->
  exists p', In p' (antichain_insert Standard qlen p positions) /\ num_errors p' <= e.
Proof.
  intros qlen p positions e Hp_err.
  unfold antichain_insert.
  destruct (subsumed_by_any Standard qlen p positions) eqn:Hsub.
  - (* p is subsumed by something in positions *)
    (* The subsuming position has errors <= errors of p <= e *)
    unfold subsumed_by_any in Hsub.
    induction positions as [| q rest IH].
    + simpl in Hsub. discriminate.
    + simpl in Hsub.
      destruct (subsumes Standard qlen q p) eqn:Hq_sub.
      * (* q subsumes p *)
        exists q. split.
        -- left. reflexivity.
        -- unfold subsumes in Hq_sub. simpl in Hq_sub.
           apply subsumes_standard_errors in Hq_sub. lia.
      * (* q doesn't subsume p, look in rest *)
        (* subsumes Standard qlen q p = subsumes_standard qlen q p = false *)
        unfold subsumes in Hq_sub. simpl in Hsub.
        rewrite Hq_sub in Hsub.
        destruct IH as [p' [Hin' Herr']].
        -- exact Hsub.
        -- exists p'. split; [right; exact Hin' | exact Herr'].
  - (* p is not subsumed - p survives *)
    exists p. split.
    + left. reflexivity.
    + exact Hp_err.
Qed.

(** Helper: remove_subsumed keeps positions that aren't subsumed by q *)
Lemma remove_subsumed_keeps_not_subsumed : forall qlen q p positions,
  In p positions ->
  subsumes Standard qlen q p = false ->
  In p (remove_subsumed Standard qlen q positions).
Proof.
  intros qlen q p positions Hin Hnsub.
  induction positions as [| r rest IH].
  - inversion Hin.
  - simpl in Hin. destruct Hin as [Heq | Hin'].
    + subst r. simpl. unfold subsumes in Hnsub. simpl in Hnsub. rewrite Hnsub.
      left. reflexivity.
    + simpl. destruct (subsumes_standard qlen q r) eqn:Hsub.
      * apply IH. exact Hin'.
      * right. apply IH. exact Hin'.
Qed.

(** Helper: antichain_insert preserves existing witnesses with bounded errors.
    If positions has a position with errors <= e, then after antichain_insert,
    the result still has a position with errors <= e. *)
Lemma antichain_insert_preserves_existing_witness : forall qlen q positions e,
  (exists p, In p positions /\ num_errors p <= e) ->
  exists p', In p' (antichain_insert Standard qlen q positions) /\ num_errors p' <= e.
Proof.
  intros qlen q positions e [p [Hin Herr]].
  unfold antichain_insert.
  destruct (subsumed_by_any Standard qlen q positions) eqn:Hsub.
  - (* q is subsumed, positions unchanged *)
    exists p. split; assumption.
  - (* q is not subsumed: result is q :: remove_subsumed q positions *)
    destruct (subsumes Standard qlen q p) eqn:Hq_sub_p.
    + (* q subsumes p, so q has errors <= p's errors <= e *)
      exists q. split.
      * left. reflexivity.
      * unfold subsumes in Hq_sub_p. simpl in Hq_sub_p.
        apply subsumes_standard_errors in Hq_sub_p. lia.
    + (* q doesn't subsume p, so p survives in remove_subsumed *)
      exists p. split.
      * right. apply remove_subsumed_keeps_not_subsumed; assumption.
      * exact Herr.
Qed.

(** Helper: positions of state_insert relate to antichain_insert *)
Lemma positions_state_insert : forall p s,
  positions (state_insert p s) = fold_right sorted_insert [] (antichain_insert (algorithm s) (query_length s) p (positions s)).
Proof.
  intros p s. unfold state_insert. simpl. reflexivity.
Qed.

(** Helper: if p is in antichain_insert result, it's in fold_right sorted_insert result *)
Lemma in_antichain_in_sorted : forall p alg qlen q positions,
  In p (antichain_insert alg qlen q positions) ->
  In p (fold_right sorted_insert [] (antichain_insert alg qlen q positions)).
Proof.
  intros p alg qlen q positions Hin.
  apply fold_right_sorted_insert_preserves_In. exact Hin.
Qed.

(** Helper: state_insert preserves witness with bounded errors *)
Lemma state_insert_preserves_witness : forall qlen q s e,
  algorithm s = Standard ->
  query_length s = qlen ->
  (exists p, In p (positions s) /\ num_errors p <= e) ->
  exists p', In p' (positions (state_insert q s)) /\ num_errors p' <= e.
Proof.
  intros qlen q s e Halg Hqlen [p [Hin Herr]].
  pose proof (antichain_insert_preserves_existing_witness qlen q (positions s) e) as H.
  destruct H as [p' [Hin' Herr']].
  { exists p. split; assumption. }
  exists p'. split.
  - rewrite positions_state_insert, Halg, Hqlen.
    apply in_antichain_in_sorted. exact Hin'.
  - exact Herr'.
Qed.

(** Helper: positions in antichain_insert result come from input list or new position *)
Lemma in_antichain_insert_origin : forall alg qlen q positions p,
  In p (antichain_insert alg qlen q positions) ->
  p = q \/ In p positions.
Proof.
  intros alg qlen q positions p Hin.
  unfold antichain_insert in Hin.
  destruct (subsumed_by_any alg qlen q positions) eqn:Hsub.
  - (* q subsumed: result is positions unchanged *)
    right. exact Hin.
  - (* q not subsumed: result is q :: remove_subsumed *)
    simpl in Hin. destruct Hin as [Heq | Hin'].
    + left. symmetry. exact Heq.
    + right. apply in_remove_subsumed in Hin'. destruct Hin' as [Hin'' _]. exact Hin''.
Qed.

(** Helper: In p (sorted_insert q positions) -> p = q \/ In p positions *)
Lemma in_sorted_insert_origin : forall q positions p,
  In p (sorted_insert q positions) ->
  p = q \/ In p positions.
Proof.
  intros q positions.
  induction positions as [| r rest IH]; intros p Hin.
  - simpl in Hin. destruct Hin as [Heq | []]. left. symmetry. exact Heq.
  - simpl in Hin.
    destruct (position_ltb q r) eqn:Hlt.
    + (* q < r: result is q :: r :: rest *)
      simpl in Hin. destruct Hin as [Heq | Hin'].
      * left. symmetry. exact Heq.
      * right. exact Hin'.
    + destruct (position_eqb q r) eqn:Heqb.
      * (* q = r: result is r :: rest *)
        right. exact Hin.
      * (* q > r: result is r :: sorted_insert q rest *)
        simpl in Hin. destruct Hin as [Heq | Hin'].
        -- right. left. exact Heq.
        -- apply IH in Hin'. destruct Hin' as [Heq' | Hin''].
           ++ left. exact Heq'.
           ++ right. right. exact Hin''.
Qed.

(** Helper: positions in fold_right sorted_insert come from input list *)
Lemma in_fold_sorted_insert_origin : forall positions p,
  In p (fold_right sorted_insert [] positions) ->
  In p positions.
Proof.
  intros positions.
  induction positions as [| q rest IH]; intros p Hin.
  - simpl in Hin. inversion Hin.
  - simpl in Hin.
    apply in_sorted_insert_origin in Hin.
    destruct Hin as [Heq | Hin'].
    + left. symmetry. exact Heq.
    + right. apply IH. exact Hin'.
Qed.

(** Helper: positions in state_insert come from input state or new position *)
Lemma in_state_insert_origin : forall q s p,
  In p (positions (state_insert q s)) ->
  p = q \/ In p (positions s).
Proof.
  intros q s p Hin.
  rewrite positions_state_insert in Hin.
  apply in_fold_sorted_insert_origin in Hin.
  apply in_antichain_insert_origin in Hin. exact Hin.
Qed.

(** Helper: all positions in fold_left state_insert come from closed_positions or init_state *)
Lemma in_fold_state_insert_origin_general : forall closed_positions init_state p,
  In p (positions (fold_left (fun s q => state_insert q s) closed_positions init_state)) ->
  In p closed_positions \/ In p (positions init_state).
Proof.
  intros closed_positions.
  induction closed_positions as [| q rest IH]; intros init_state p Hin.
  - (* Base: fold_left on [] is identity *)
    simpl in Hin. right. exact Hin.
  - (* Step: fold_left on (q :: rest) *)
    simpl in Hin.
    apply IH in Hin.
    destruct Hin as [Hin_rest | Hin_state].
    + (* p came from rest *)
      left. right. exact Hin_rest.
    + (* p came from state_insert q init_state *)
      apply in_state_insert_origin in Hin_state.
      destruct Hin_state as [Heq | Hin_init].
      * (* p = q *)
        left. left. symmetry. exact Heq.
      * (* p from init_state *)
        right. exact Hin_init.
Qed.

(** Corollary: when init_state is empty, positions come from closed_positions *)
Lemma in_fold_state_insert_origin : forall closed_positions init_state p,
  positions init_state = [] ->
  In p (positions (fold_left (fun s q => state_insert q s) closed_positions init_state)) ->
  In p closed_positions.
Proof.
  intros closed_positions init_state p Hinit Hin.
  apply in_fold_state_insert_origin_general in Hin.
  destruct Hin as [Hin_closed | Hin_init].
  - exact Hin_closed.
  - rewrite Hinit in Hin_init. inversion Hin_init.
Qed.

(** Helper: fold_left state_insert preserves min error bound.
    Key insight: subsumption in Standard requires e1 <= e2, so if we insert
    positions with errors <= e, the final antichain has min error <= e.

    Technical note: This requires showing that:
    1. When we insert a position with errors <= e, either it survives or something
       with errors <= e already exists (since subsumption requires e1 <= e2)
    2. Processing subsequent positions doesn't remove our witness

    Full proof requires tracking that Standard never produces special positions,
    and that subsumption in Standard preserves error bounds. *)

(** Helper: state_insert preserves algorithm *)
Lemma algorithm_state_insert : forall p s,
  algorithm (state_insert p s) = algorithm s.
Proof.
  intros p s. unfold state_insert. simpl. reflexivity.
Qed.

(** Helper: state_insert preserves query_length *)
Lemma query_length_state_insert : forall p s,
  query_length (state_insert p s) = query_length s.
Proof.
  intros p s. unfold state_insert. simpl. reflexivity.
Qed.

Lemma fold_state_insert_preserves_min_error : forall qlen closed_positions init_state e,
  algorithm init_state = Standard ->
  query_length init_state = qlen ->
  (forall p, In p closed_positions -> is_special p = false) ->  (* All positions non-special *)
  positions init_state = [] ->  (* Start from empty state *)
  (exists p, In p closed_positions /\ num_errors p <= e) ->
  exists p', In p' (positions (fold_left (fun s p => state_insert p s) closed_positions init_state)) /\
             is_special p' = false /\ num_errors p' <= e.
Proof.
  (* The proof follows from:
     1. antichain_insert_preserves_min_error: when inserting p with errors <= e,
        the result has some position with errors <= e
     2. Subsequent insertions don't remove positions with smaller errors
        (subsumption requires e1 <= e2, so a position with errors <= e
        can only be subsumed by something with errors <= e)
     3. Standard algorithm only produces non-special positions *)
  intros qlen closed_positions init_state e Halg Hqlen Hnonspec Hinit_empty [p [Hin Herr]].

  (* Key insight: once we insert a position with errors <= e,
     subsequent inserts preserve this bound. We track this via an invariant. *)

  (* First, show there exists a position with errors <= e in the final state *)
  assert (Herr_preserved : exists p', In p' (positions (fold_left (fun s q => state_insert q s)
                                                        closed_positions init_state)) /\
                           num_errors p' <= e).
  { (* Proof by induction with invariant:
       Either we haven't yet processed our witness p, or the state contains a position with errors <= e *)
    revert Hin.
    generalize dependent init_state.
    induction closed_positions as [| q rest IH]; intros init_state Halg Hqlen Hinit_empty Hin_p.
    - (* Base case: closed_positions = [], but p ∈ [] is false *)
      inversion Hin_p.
    - (* Step case: closed_positions = q :: rest *)
      simpl.
      simpl in Hin_p. destruct Hin_p as [Heq | Hin_rest].
      + (* p = q is the head *)
        subst q.
        (* After inserting p, the state has a position with errors <= e *)
        (* Then subsequent inserts preserve this *)
        assert (Hafter_p : exists p', In p' (positions (state_insert p init_state)) /\
                                      num_errors p' <= e).
        { (* Use antichain_insert_preserves_min_error + bridge to state positions *)
          destruct (antichain_insert_preserves_min_error qlen p (positions init_state) e Herr)
            as [p' [Hin_ac Herr']].
          exists p'. split; [| exact Herr'].
          rewrite positions_state_insert.
          rewrite Halg, Hqlen.
          apply fold_right_sorted_insert_preserves_In.
          exact Hin_ac. }
        destruct Hafter_p as [p0 [Hin0 Herr0]].
        (* Now show subsequent inserts preserve this witness *)
        clear IH.
        (* We have p0 in state_insert p init_state with errors <= e.
           Need to show fold_left over rest preserves a position with errors <= e.
           We generalize over the state to allow induction. *)
        remember (state_insert p init_state) as s0 eqn:Hs0.
        assert (Halg0 : algorithm s0 = Standard).
        { rewrite Hs0. rewrite algorithm_state_insert. exact Halg. }
        assert (Hqlen0 : query_length s0 = qlen).
        { rewrite Hs0. rewrite query_length_state_insert. exact Hqlen. }
        clear Hs0 Halg Hqlen Hinit_empty.
        revert s0 p0 Hin0 Herr0 Halg0 Hqlen0.
        induction rest as [| r rest' IHrest]; intros s0 p0 Hin0 Herr0 Halg0 Hqlen0;
          [simpl; exists p0; split; assumption |].
        (* Step: Insert r, then process rest' *)
        simpl.
        assert (Hpreserved : exists p1, In p1 (positions (state_insert r s0)) /\ num_errors p1 <= e).
        { apply state_insert_preserves_witness with (qlen := query_length s0).
          - rewrite Halg0. reflexivity.
          - reflexivity.
          - exists p0. split; assumption. }
        destruct Hpreserved as [p1 [Hin1 Herr1]].
        eapply IHrest; [| exact Hin1 | exact Herr1 |
                        rewrite algorithm_state_insert; exact Halg0 |
                        rewrite query_length_state_insert; exact Hqlen0 ].
        (* Non-special property for p :: rest' *)
        intros q Hq_in.
        apply Hnonspec. simpl in Hq_in |- *. destruct Hq_in as [Heq | Hq_in'];
          [left; exact Heq | right; right; exact Hq_in'].
      + (* p is in rest, recurse *)
        assert (Halg' : algorithm (state_insert q init_state) = Standard).
        { rewrite algorithm_state_insert. exact Halg. }
        assert (Hqlen' : query_length (state_insert q init_state) = qlen).
        { rewrite query_length_state_insert. exact Hqlen. }
        assert (Hinit_empty' : positions (state_insert q init_state) =
                               fold_right sorted_insert [] [q]).
        { rewrite positions_state_insert. unfold antichain_insert.
          rewrite Hinit_empty. reflexivity. }
        (* This is not [] but we need to track differently *)
        (* Actually, we should use a generalized IH that doesn't require empty init_state *)
        (* For now, use a different approach - just track that error bound is preserved *)
        clear IH.
        (* Use that error bound is preserved through fold_left *)
        generalize dependent (state_insert q init_state).
        induction rest as [| r rest' IHrest']; intros s' Halg' Hqlen' Hinit_empty'.
        { (* Base case: rest = [] contradicts Hin_rest : In p [] *)
          inversion Hin_rest. }
        (* Step case: rest = r :: rest' *)
        simpl in Hin_rest |- *.
        destruct Hin_rest as [Hp_eq | Hin_rest'].
        -- (* p = r *)
           subst r.
           assert (Hafter_p : exists p', In p' (positions (state_insert p s')) /\
                                         num_errors p' <= e).
           { destruct (antichain_insert_preserves_min_error (query_length s') p (positions s') e Herr)
               as [p' [Hin_ac Herr']].
             exists p'. split; [| exact Herr'].
               rewrite positions_state_insert.
               apply fold_right_sorted_insert_preserves_In.
               rewrite Halg'. exact Hin_ac. }
           destruct Hafter_p as [pw [Hin_w Herr_w]].
           clear IHrest'.
           (* Process rest' with witness pw *)
           assert (Hgen: forall s_cur : State, algorithm s_cur = Standard -> query_length s_cur = qlen ->
                forall pi, In pi (positions s_cur) -> num_errors pi <= e ->
                exists pf, In pf (positions (fold_left (fun s0 q => state_insert q s0) rest' s_cur)) /\
                           num_errors pf <= e).
           { clear s' Halg' Hqlen' Hinit_empty' pw Hin_w Herr_w.
             induction rest' as [| r' rest'' IHrest'']; intros s_cur Halg_cur Hqlen_cur pi Hin_pi Herr_pi;
               [simpl; exists pi; split; assumption |].
             simpl.
             assert (Hpreserved : exists pi', In pi' (positions (state_insert r' s_cur)) /\
                                              num_errors pi' <= e)
               by (apply state_insert_preserves_witness with (qlen := query_length s_cur);
                   [exact Halg_cur | reflexivity | exists pi; split; assumption]).
             destruct Hpreserved as [pi' [Hin_pi' Herr_pi']].
             assert (Hnonspec' : forall p0 : Position, In p0 (q :: p :: rest'') -> is_special p0 = false)
               by (intros p0 Hin0; apply Hnonspec; simpl in *; tauto).
             assert (Halg_ins' : algorithm (state_insert r' s_cur) = Standard)
               by (rewrite algorithm_state_insert; exact Halg_cur).
             assert (Hqlen_ins' : query_length (state_insert r' s_cur) = qlen)
               by (rewrite query_length_state_insert; exact Hqlen_cur).
             exact (IHrest'' Hnonspec' (state_insert r' s_cur) Halg_ins' Hqlen_ins' pi' Hin_pi' Herr_pi'). }
           assert (Halg_ins : algorithm (state_insert p s') = Standard)
             by (rewrite algorithm_state_insert; exact Halg').
           assert (Hqlen_ins : query_length (state_insert p s') = qlen)
             by (rewrite query_length_state_insert; exact Hqlen').
           exact (Hgen (state_insert p s') Halg_ins Hqlen_ins pw Hin_w Herr_w).
        -- (* p in rest' - recursive case *)
           (* Use the same Hgen approach: prove a general statement about fold_left *)
           (* Key: when p eventually appears in the list, we'll get a witness *)
           (* We prove: for any state s, if p is in remaining positions, and alg/qlen are right,
              then fold_left produces a position with errors <= e *)
           assert (Hgen_rest': forall rest_pos s_cur,
                     algorithm s_cur = Standard ->
                     query_length s_cur = qlen ->
                     In p rest_pos ->
                     exists pf, In pf (positions (fold_left (fun s0 q0 => state_insert q0 s0) rest_pos s_cur)) /\
                                num_errors pf <= e).
           { clear IHrest' s' Halg' Hqlen' Hinit_empty' Hin_rest'.
             induction rest_pos as [| r' rest'' IHrest'']; intros s_cur Halg_cur Hqlen_cur Hin_p';
               [inversion Hin_p' |].
             simpl in Hin_p' |- *.
             destruct Hin_p' as [Heq | Hin_rest''].
             - (* p = r' - insert p now, then use preservation *)
               subst r'.
               assert (Hafter_p : exists pw, In pw (positions (state_insert p s_cur)) /\
                                             num_errors pw <= e).
               { destruct (antichain_insert_preserves_min_error (query_length s_cur) p (positions s_cur) e Herr)
                   as [pw [Hin_ac Herr_w]].
                 exists pw. split; [| exact Herr_w].
                 rewrite positions_state_insert.
                 apply fold_right_sorted_insert_preserves_In.
                 rewrite Halg_cur. exact Hin_ac. }
               destruct Hafter_p as [pw [Hin_w Herr_w]].
               (* Now fold_left over rest'' preserves the witness *)
               (* Use a general lemma about state_insert_preserves_witness through fold_left *)
               clear IHrest''.
               assert (Hpres_fold: forall l s_any pw_any,
                         algorithm s_any = Standard ->
                         query_length s_any = qlen ->
                         In pw_any (positions s_any) ->
                         num_errors pw_any <= e ->
                         exists pf, In pf (positions (fold_left (fun s0 q0 => state_insert q0 s0) l s_any)) /\
                                    num_errors pf <= e).
               { clear s_cur Halg_cur Hqlen_cur pw Hin_w Herr_w.
                 induction l as [| r'' l' IHl']; intros s_any pw_any Halg_any Hqlen_any Hin_any Herr_any;
                   [simpl; exists pw_any; split; assumption |].
                 simpl.
                 assert (Hpres' : exists pi', In pi' (positions (state_insert r'' s_any)) /\
                                              num_errors pi' <= e).
                 { apply state_insert_preserves_witness with (qlen := query_length s_any).
                   - exact Halg_any.
                   - reflexivity.
                   - exists pw_any. split; assumption. }
                 destruct Hpres' as [pi' [Hin_pi' Herr_pi']].
                 apply IHl' with (pw_any := pi').
                 - rewrite algorithm_state_insert. exact Halg_any.
                 - rewrite query_length_state_insert. exact Hqlen_any.
                 - exact Hin_pi'.
                 - exact Herr_pi'. }
               apply Hpres_fold with (pw_any := pw).
               + rewrite algorithm_state_insert. exact Halg_cur.
               + rewrite query_length_state_insert. exact Hqlen_cur.
               + exact Hin_w.
               + exact Herr_w.
             - (* p in rest'' - recurse *)
               apply IHrest'' with (s_cur := state_insert r' s_cur).
               + rewrite algorithm_state_insert. exact Halg_cur.
               + rewrite query_length_state_insert. exact Hqlen_cur.
               + exact Hin_rest''. }
           (* Apply Hgen_rest' for rest' starting from state_insert r s' *)
           apply Hgen_rest' with (rest_pos := rest') (s_cur := state_insert r s').
           ++ rewrite algorithm_state_insert. exact Halg'.
           ++ rewrite query_length_state_insert. exact Hqlen'.
           ++ exact Hin_rest'.
  }

  (* Now show the position with errors <= e also has is_special = false *)
  destruct Herr_preserved as [p' [Hin' Herr']].
  exists p'. split; [exact Hin' |].
  split.
  - (* Show is_special p' = false *)
    (* p' comes from closed_positions (since init_state is empty) *)
    apply in_fold_state_insert_origin in Hin'; [| exact Hinit_empty].
    apply Hnonspec. exact Hin'.
  - exact Herr'.
Qed.

(** Helper: algorithm of empty_state *)
Lemma algorithm_empty_state : forall alg qlen,
  algorithm (empty_state alg qlen) = alg.
Proof.
  intros alg qlen. unfold empty_state. simpl. reflexivity.
Qed.

(** Helper: fold_left state_insert preserves algorithm *)
(* Note: algorithm_state_insert already defined earlier *)
Lemma algorithm_fold_state_insert : forall l s,
  algorithm (fold_left (fun s0 p0 => state_insert p0 s0) l s) = algorithm s.
Proof.
  induction l as [| p rest IH]; intros s.
  - simpl. reflexivity.
  - simpl. rewrite IH. apply algorithm_state_insert.
Qed.

(** transition_state preserves algorithm type *)
Lemma transition_state_preserves_algorithm : forall alg s c query n s',
  transition_state alg s c query n = Some s' ->
  algorithm s' = alg.
Proof.
  intros alg s c query n s' Htrans.
  unfold transition_state in Htrans.
  destruct (is_nil _); [discriminate|].
  injection Htrans as Heq. subst s'.
  rewrite algorithm_fold_state_insert.
  apply algorithm_empty_state.
Qed.

(** Helper: query_length of empty_state *)
Lemma query_length_empty_state : forall alg qlen,
  query_length (empty_state alg qlen) = qlen.
Proof.
  intros alg qlen. unfold empty_state. simpl. reflexivity.
Qed.

(** Helper: fold_left state_insert preserves query_length *)
(* Note: query_length_state_insert already defined earlier *)
Lemma query_length_fold_state_insert : forall l s,
  query_length (fold_left (fun s0 p0 => state_insert p0 s0) l s) = query_length s.
Proof.
  induction l as [| p rest IH]; intros s.
  - simpl. reflexivity.
  - simpl. rewrite IH. apply query_length_state_insert.
Qed.

(** transition_state preserves query_length *)
Lemma transition_state_preserves_query_length : forall alg s c query n s',
  transition_state alg s c query n = Some s' ->
  query_length s' = query_length s.
Proof.
  intros alg s c query n s' Htrans.
  unfold transition_state in Htrans.
  destruct (is_nil _); [discriminate|].
  injection Htrans as Heq. subst s'.
  rewrite query_length_fold_state_insert.
  apply query_length_empty_state.
Qed.

(** Running the automaton preserves the algorithm field. *)
Lemma automaton_run_preserves_algorithm : forall alg query n dict s s',
  algorithm s = alg ->
  automaton_run alg query n dict s = Some s' ->
  algorithm s' = alg.
Proof.
  induction dict as [|c dict IH]; intros s s' Halg Hrun.
  - simpl in Hrun. injection Hrun as Heq. subst s'. exact Halg.
  - simpl in Hrun.
    destruct (transition_state alg s c query n) as [s_mid|] eqn:Htrans;
      try discriminate.
    apply (IH s_mid s').
    + apply transition_state_preserves_algorithm in Htrans.
      exact Htrans.
    + exact Hrun.
Qed.

(** Helper: std_pos is non-special *)
Lemma std_pos_not_special : forall i e, is_special (std_pos i e) = false.
Proof.
  intros i e. unfold std_pos, is_special. reflexivity.
Qed.

(** Helper: transition_position_standard produces only non-special positions *)
Lemma transition_standard_nonspecial : forall p cv min_i n qlen p',
  In p' (transition_position_standard p cv min_i n qlen) ->
  is_special p' = false.
Proof.
  intros p cv min_i n qlen p' Hin.
  apply transition_position_standard_non_special' in Hin.
  exact Hin.
Qed.

(** Helper: transition_position Standard = transition_position_standard for non-special *)
Lemma transition_position_standard_eq : forall p cv min_i n qlen,
  is_special p = false ->
  transition_position Standard p cv min_i n qlen = transition_position_standard p cv min_i n qlen.
Proof.
  intros p cv min_i n qlen Hnonspec.
  unfold transition_position.
  reflexivity.
Qed.

(** Helper: delete_step produces only non-special positions *)
Lemma delete_step_nonspecial : forall p n qlen p',
  delete_step p n qlen = Some p' ->
  is_special p' = false.
Proof.
  intros p n qlen p' Hdel.
  unfold delete_step in Hdel.
  destruct (is_special p); [discriminate|].
  destruct ((S (term_index p) <=? qlen) && (num_errors p <? n)); [| discriminate].
  injection Hdel as Heq. subst p'. apply std_pos_not_special.
Qed.

(** Helper: epsilon_closure_aux preserves non-special property *)
Lemma epsilon_closure_aux_nonspecial : forall fuel positions n qlen p,
  (forall q, In q positions -> is_special q = false) ->
  In p (epsilon_closure_aux positions n qlen fuel) ->
  is_special p = false.
Proof.
  induction fuel as [| fuel' IH]; intros positions n qlen p Hnonspec_input Hin.
  - (* fuel = 0 *)
    simpl in Hin. apply Hnonspec_input. exact Hin.
  - (* fuel = S fuel' *)
    simpl in Hin.
    set (new := flat_map (fun p0 => match delete_step p0 n qlen with
                                    | Some p' => [p']
                                    | None => []
                                    end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + (* new is empty *)
      apply Hnonspec_input. exact Hin.
    + (* new is non-empty *)
      apply IH with (positions := positions ++ new) (n := n) (qlen := qlen).
      * (* All positions in positions ++ new are non-special *)
        intros q Hq_in.
        apply in_app_or in Hq_in.
        destruct Hq_in as [Hq_orig | Hq_new].
        -- apply Hnonspec_input. exact Hq_orig.
        -- (* q is in new, produced by delete_step *)
           unfold new in Hq_new.
           apply in_flat_map in Hq_new.
           destruct Hq_new as [q0 [Hq0_in Hq_del]].
           destruct (delete_step q0 n qlen) as [q'|] eqn:Hdel.
           ++ destruct Hq_del as [Heq | Hcontra]; [| inversion Hcontra].
              subst q. apply delete_step_nonspecial with (p := q0) (n := n) (qlen := qlen).
              exact Hdel.
           ++ inversion Hq_del.
      * exact Hin.
Qed.

(** Helper: epsilon_closure preserves non-special property *)
Lemma epsilon_closure_nonspecial : forall positions n qlen p,
  (forall q, In q positions -> is_special q = false) ->
  In p (epsilon_closure positions n qlen) ->
  is_special p = false.
Proof.
  intros positions n qlen p Hnonspec_input Hin.
  unfold epsilon_closure in Hin.
  apply epsilon_closure_aux_nonspecial with (fuel := S n) (positions := positions) (n := n) (qlen := qlen).
  - exact Hnonspec_input.
  - exact Hin.
Qed.

(** Helper: transition_state_positions for Standard produces only non-special positions *)
Lemma transition_state_positions_standard_nonspecial : forall positions cv min_i n qlen p,
  In p (transition_state_positions Standard positions cv min_i n qlen) ->
  is_special p = false.
Proof.
  intros positions cv min_i n qlen p Hin.
  unfold transition_state_positions in Hin.
  apply in_flat_map in Hin.
  destruct Hin as [q [Hq_in Hp_trans]].
  unfold transition_position in Hp_trans.
  apply transition_standard_nonspecial with (p := q) (cv := cv) (min_i := min_i) (n := n) (qlen := qlen).
  exact Hp_trans.
Qed.

(** Standard transitions do not advance past [qlen] when their inputs are
    already query-bounded. *)
Lemma transition_standard_term_bounded : forall p cv min_i n qlen p',
  term_index p <= qlen ->
  min_i <= term_index p ->
  (forall j, cv_at cv j = true -> min_i + j < qlen) ->
  In p' (transition_position_standard p cv min_i n qlen) ->
  term_index p' <= qlen.
Proof.
  intros p cv min_i n qlen p' Hp_bound Hmin_le Hcv_bound Hin.
  unfold transition_position_standard in Hin.
  destruct (is_special p); [inversion Hin|].
  apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - destruct (term_index p <? qlen) eqn:Hlt; [| inversion Hin].
    apply Nat.ltb_lt in Hlt.
    destruct (num_errors p <? n) eqn:Herr.
    + set (offset := term_index p - min_i) in *.
      set (limit := Nat.min (n - num_errors p + 1) (length cv - offset)) in *.
      change (In p'
        match index_of_match cv offset limit with
        | Some 0 => [std_pos (S (term_index p)) (num_errors p)]
        | Some (S j0) =>
            [std_pos (S (term_index p)) (S (num_errors p));
             std_pos (term_index p + S (S j0)) (num_errors p + S j0)]
        | None => [std_pos (S (term_index p)) (S (num_errors p))]
        end) in Hin.
      destruct (index_of_match cv offset limit) as [[|j]|] eqn:Hidx;
        simpl in Hin.
      * destruct Hin as [Heq | []]. subst p'. simpl. lia.
      * destruct Hin as [Heq | [Heq | []]].
        -- subst p'. simpl. lia.
        -- subst p'. simpl.
           pose proof (index_of_match_some_cv_at cv offset limit (S j) Hidx) as Hcvj.
           pose proof (Hcv_bound _ Hcvj) as Hquery.
           unfold offset in Hquery. lia.
      * destruct Hin as [Heq | []]. subst p'. simpl. lia.
    + change (In p' (if cv_at cv (term_index p - min_i) then
                       [std_pos (S (term_index p)) (num_errors p)]
                     else [])) in Hin.
      destruct (cv_at cv (term_index p - min_i)) eqn:Hcv; simpl in Hin; try contradiction.
      destruct Hin as [Heq | []]. subst p'. simpl. lia.
  - destruct (num_errors p <? n) eqn:Herr; [| inversion Hin].
    destruct Hin as [Heq | []]. subst p'. simpl. exact Hp_bound.
Qed.

Lemma delete_step_term_bounded : forall p n qlen p',
  delete_step p n qlen = Some p' ->
  term_index p' <= qlen.
Proof.
  intros p n qlen p' Hdel.
  unfold delete_step in Hdel.
  destruct (is_special p); [discriminate|].
  destruct ((S (term_index p) <=? qlen) && (num_errors p <? n)) eqn:Hcond;
    [| discriminate].
  apply andb_prop in Hcond. destruct Hcond as [Hle _].
  apply Nat.leb_le in Hle.
  injection Hdel as Heq. subst p'. simpl. exact Hle.
Qed.

Lemma epsilon_closure_aux_term_bounded : forall fuel positions n qlen p,
  (forall q, In q positions -> term_index q <= qlen) ->
  In p (epsilon_closure_aux positions n qlen fuel) ->
  term_index p <= qlen.
Proof.
  induction fuel as [|fuel' IH]; intros positions n qlen p Hbound Hin.
  - simpl in Hin. apply Hbound. exact Hin.
  - simpl in Hin.
    set (new := flat_map (fun p0 => match delete_step p0 n qlen with
                                    | Some p' => [p']
                                    | None => []
                                    end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + apply Hbound. exact Hin.
    + apply IH with (positions := positions ++ new) (n := n).
      * intros q Hq.
        apply in_app_or in Hq.
        destruct Hq as [Hq_old | Hq_new].
        -- apply Hbound. exact Hq_old.
        -- unfold new in Hq_new.
           apply in_flat_map in Hq_new.
           destruct Hq_new as [q0 [Hq0_in Hq_del]].
           destruct (delete_step q0 n qlen) as [q'|] eqn:Hdel.
           ++ destruct Hq_del as [Heq | []]. subst q.
              apply delete_step_term_bounded with (p := q0) (n := n).
              exact Hdel.
           ++ inversion Hq_del.
      * exact Hin.
Qed.

Lemma epsilon_closure_term_bounded : forall positions n qlen p,
  (forall q, In q positions -> term_index q <= qlen) ->
  In p (epsilon_closure positions n qlen) ->
  term_index p <= qlen.
Proof.
  intros positions n qlen p Hbound Hin.
  unfold epsilon_closure in Hin.
  eapply epsilon_closure_aux_term_bounded; eauto.
Qed.

Lemma transition_state_positions_standard_term_bounded : forall positions cv min_i n qlen p,
  (forall q, In q positions -> term_index q <= qlen) ->
  (forall q, In q positions -> min_i <= term_index q) ->
  (forall j, cv_at cv j = true -> min_i + j < qlen) ->
  In p (transition_state_positions Standard positions cv min_i n qlen) ->
  term_index p <= qlen.
Proof.
  intros positions cv min_i n qlen p Hbound Hmin_bound Hcv_bound Hin.
  unfold transition_state_positions in Hin.
  apply in_flat_map in Hin.
  destruct Hin as [q [Hq_in Hp_trans]].
  unfold transition_position in Hp_trans.
  eapply transition_standard_term_bounded.
  - apply Hbound. exact Hq_in.
  - apply Hmin_bound. exact Hq_in.
  - exact Hcv_bound.
  - exact Hp_trans.
Qed.

(** A Standard transition output is delete-closed up to executable
    representation: if a folded survivor can still delete within the query and
    error bounds, then the folded state represents that delete-chain endpoint. *)
Lemma transition_state_standard_delete_chain_represented : forall
  s c query n s' p k,
  transition_state Standard s c query n = Some s' ->
  In p (positions s') ->
  term_index p + k <= query_length s ->
  num_errors p + k <= n ->
  positions_subsume Standard (query_length s) (positions s')
    (std_pos (term_index p + k) (num_errors p + k)).
Proof.
  intros s c query n s' p k Htrans Hin Hterm Herr.
  assert (Htrans_orig := Htrans).
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions Standard (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  assert (Hp_closed : In p closed_positions).
  { apply in_fold_state_insert_origin with
      (init_state := empty_state Standard (query_length s)).
    - unfold empty_state. reflexivity.
    - exact Hin. }
  assert (Htarget_closed :
    In (std_pos (term_index p + k) (num_errors p + k)) closed_positions).
  { unfold closed_positions.
    apply epsilon_closure_member_reaches_deletes.
    - intros q Hq.
      unfold trans_positions in Hq.
      eapply transition_state_positions_standard_nonspecial. exact Hq.
    - exact Hp_closed.
    - exact Hterm.
    - exact Herr. }
  eapply transition_state_standard_represents_closed_position.
  - exact Htrans_orig.
  - exact Htarget_closed.
Qed.

Definition state_delete_chain_represented (n : nat) (s : State) : Prop :=
  forall p k,
    In p (positions s) ->
    term_index p + k <= query_length s ->
    num_errors p + k <= n ->
    positions_subsume Standard (query_length s) (positions s)
      (std_pos (term_index p + k) (num_errors p + k)).

Lemma transition_state_standard_state_delete_chain_represented : forall
  s c query n s',
  transition_state Standard s c query n = Some s' ->
  state_delete_chain_represented n s'.
Proof.
  intros s c query n s' Htrans p k Hin Hterm Herr.
  assert (Hqlen : query_length s' = query_length s).
  { eapply transition_state_preserves_query_length. exact Htrans. }
  rewrite Hqlen in Hterm.
  rewrite Hqlen.
  eapply transition_state_standard_delete_chain_represented; eauto.
Qed.

Lemma automaton_run_standard_delete_chain_represented_from_state : forall
  query n dict s final,
  state_delete_chain_represented n s ->
  automaton_run Standard query n dict s = Some final ->
  state_delete_chain_represented n final.
Proof.
  induction dict as [|c rest IH]; intros s final Hclosed Hrun.
  - simpl in Hrun. injection Hrun as Hfinal. subst final. exact Hclosed.
  - simpl in Hrun.
    destruct (transition_state Standard s c query n) as [s_mid|] eqn:Htrans;
      [| discriminate].
    apply (IH s_mid final).
    + apply transition_state_standard_state_delete_chain_represented with
        (s := s) (c := c) (query := query).
      exact Htrans.
    + exact Hrun.
Qed.

(** Key lemma: transition preserves "has position with bounded errors".
    If input has position with errors e < n, output has position with errors <= e + 1.
    This is critical for the induction in automaton_run_with_slack. *)
Lemma transition_state_has_bounded_error : forall s c query n e,
  algorithm s = Standard ->
  query_length s = length query ->
  (exists p, In p (positions s) /\ is_special p = false /\ num_errors p <= e) ->
  e < n ->
  forall s', transition_state Standard s c query n = Some s' ->
  (exists p', In p' (positions s') /\ is_special p' = false /\ num_errors p' <= S e).
Proof.
  intros s c query n e Halg Hqlen [p [Hin [Hspec Hperr]]] He_lt s' Htrans.
  (* The transition produces at least the insert position with errors = num_errors p + 1 *)
  (* This position goes through transition_state_positions, epsilon_closure, and fold_state_insert *)
  (* After fold_state_insert, some position with errors <= num_errors p + 1 <= S e exists *)
  unfold transition_state in Htrans.
  (* Extract the components *)
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_pos := transition_state_positions Standard (positions s) cv min_i n (query_length s)) in *.
  set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
  destruct (is_nil closed_pos) eqn:Hnil in Htrans; [discriminate|].
  injection Htrans as Hs'_eq.

  (* Step 1: Get insert position from transition_produces_insert_exact *)
  assert (Herr_bound : num_errors p < n) by lia.
  destruct (transition_produces_insert_exact p cv min_i n (query_length s) Hspec Herr_bound)
    as [p_ins [Hin_trans [Hp_ins_err Hp_ins_spec]]].

  (* Step 2: p_ins is in trans_pos (via flat_map) *)
  assert (Hin_trans_pos : In p_ins trans_pos).
  { unfold trans_pos, transition_state_positions.
    apply in_flat_map. exists p. split.
    - exact Hin.
    - unfold transition_position. exact Hin_trans. }

  (* Step 3: p_ins is in closed_pos (epsilon_closure includes input) *)
  assert (Hin_closed : In p_ins closed_pos).
  { unfold closed_pos. unfold epsilon_closure.
    apply epsilon_closure_aux_includes_input. exact Hin_trans_pos. }

  (* Step 4: All positions in closed_pos are non-special for Standard *)
  (* For Standard algorithm:
     - transition_position_standard produces only std_pos (is_special = false)
     - delete_step produces only std_pos (is_special = false)
     - epsilon_closure_aux only applies delete_step, so preserves this *)
  assert (Hnonspec_closed : forall q, In q closed_pos -> is_special q = false).
  { unfold closed_pos.
    intros q Hq_in.
    apply epsilon_closure_nonspecial with (positions := trans_pos) (n := n) (qlen := query_length s).
    - (* All positions in trans_pos are non-special *)
      intros q0 Hq0_in. unfold trans_pos in Hq0_in.
      apply transition_state_positions_standard_nonspecial with
        (positions := positions s) (cv := cv) (min_i := min_i) (n := n) (qlen := query_length s).
      exact Hq0_in.
    - exact Hq_in. }

  (* Step 5: Apply fold_state_insert_preserves_min_error *)
  rewrite <- Hs'_eq.
  assert (Halg_empty : algorithm (empty_state Standard (query_length s)) = Standard).
  { unfold empty_state. reflexivity. }
  assert (Hqlen_empty : query_length (empty_state Standard (query_length s)) = query_length s).
  { unfold empty_state. reflexivity. }
  assert (Hinit_empty : positions (empty_state Standard (query_length s)) = []).
  { unfold empty_state. reflexivity. }

  (* We have p_ins in closed_pos with num_errors = S (num_errors p) <= S e *)
  assert (Hp_ins_bound : num_errors p_ins <= S e).
  { rewrite Hp_ins_err. lia. }

  destruct (fold_state_insert_preserves_min_error (query_length s) closed_pos
             (empty_state Standard (query_length s)) (S e)
             Halg_empty Hqlen_empty Hnonspec_closed Hinit_empty)
    as [pf [Hin_final [Hspec_final Herr_final]]].
  { exists p_ins. split; [exact Hin_closed | exact Hp_ins_bound]. }

  exists pf. split; [| split]; assumption.
Qed.

(** Stronger invariant: if we have "slack" (errors + dict_length <= n),
    then automaton_run succeeds and maintains bounded errors. *)
Lemma automaton_run_with_slack : forall query n dict s e,
  algorithm s = Standard ->
  query_length s = length query ->
  (exists p, In p (positions s) /\ is_special p = false /\ num_errors p <= e) ->
  e + length dict <= n ->
  exists final,
    automaton_run Standard query n dict s = Some final /\
    (exists p', In p' (positions final) /\ is_special p' = false /\ num_errors p' <= e + length dict).
Proof.
  intros query n dict.
  induction dict as [| c rest IH]; intros s e Halg Hqlen [p [Hin [Hspec Hperr]]] Hslack.

  - (* dict = [] *)
    simpl. exists s. split; [reflexivity|].
    exists p. split; [exact Hin | split; [exact Hspec | simpl; lia]].

  - (* dict = c :: rest *)
    simpl.
    simpl in Hslack. (* e + S (length rest) <= n, so e < n *)
    assert (He_lt_n : e < n) by lia.
    (* Use transition_state_not_dead_standard since we have errors e < n *)
    assert (Hpred_bound : num_errors p < n) by lia.
    destruct (transition_state_not_dead_standard s c query n Halg Hqlen) as [s' Hs'].
    { exists p. split; [exact Hin | split; [exact Hspec | exact Hpred_bound]]. }
    (* Prove algorithm and query_length properties for s' using helper lemmas *)
    assert (Halg' : algorithm s' = Standard).
    { apply (transition_state_preserves_algorithm Standard s c query n s' Hs'). }
    assert (Hqlen' : query_length s' = length query).
    { rewrite (transition_state_preserves_query_length Standard s c query n s' Hs'). exact Hqlen. }
    rewrite Hs'.
    (* After transition, s' has position with errors <= S e *)
    (* Now apply IH with e' = S e *)
    assert (Htrans_bound : exists p', In p' (positions s') /\ is_special p' = false /\ num_errors p' <= S e).
    { apply (transition_state_has_bounded_error s c query n e Halg Hqlen).
      - exists p. split; [exact Hin | split; [exact Hspec | exact Hperr]].
      - exact He_lt_n.
      - exact Hs'. }
    destruct Htrans_bound as [p' [Hin' [Hspec' Hp'err]]].
    assert (Hslack' : S e + length rest <= n) by lia.
    destruct (IH s' (S e) Halg' Hqlen') as [final [Hrun Hfinal]].
    + exists p'. split; [exact Hin' | split; [exact Hspec' | exact Hp'err]].
    + exact Hslack'.
    + exists final. split; [exact Hrun|].
      destruct Hfinal as [p'' [Hin'' [Hspec'' Hp''err]]].
      exists p''. split; [exact Hin'' | split; [exact Hspec'' | simpl in *; lia]].
Qed.

(** Note: A lemma automaton_run_preserves_bounded_error was previously here but was
    removed because its statement was too strong - it claimed the automaton never
    goes dead for ANY dictionary, but when errors + dict_length > n and all remaining
    characters mismatch, the automaton must die. Current Standard completeness
    uses the can-complete invariant below, which follows a concrete completion
    path rather than arbitrary bounded-error states. *)

(** Helper: Positions reachable via empty dictionary prefix are on the diagonal.

    This is because only reach_initial and reach_delete don't extend the
    dictionary prefix. Starting from (0,0), applying reach_delete k times
    gives (k,k). *)
Lemma reachable_empty_prefix_diagonal : forall query n i e,
  position_reachable query n [] (std_pos i e) -> i = e.
Proof.
  intros query n i e Hreach.
  remember [] as dp eqn:Hdp.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hreach as [
    | dp' i' e' Hreach' IH Hbound_i Hbound_e
    | dp' c i' e' Hreach' IH Hlt Hnth
    | dp' c c' i' e' Hreach' IH Hlt Hnth Hneq Hbound_e
    | dp' c i' e' Hreach' IH Hbound_e
  ]; intros i e Hp.
  - (* reach_initial: dp = [], p = initial_position = (0, 0) *)
    injection Hp. intros. subst. reflexivity.
  - (* reach_delete: dp' = [], p = std_pos (S i') (S e') *)
    injection Hp. intros He Hi. subst i e.
    f_equal. apply IH.
    + exact Hdp.
    + reflexivity.
  - (* reach_match: dp' ++ [c] = [] is impossible *)
    destruct dp'; discriminate.
  - (* reach_substitute: dp' ++ [c] = [] is impossible *)
    destruct dp'; discriminate.
  - (* reach_insert: dp' ++ [c] = [] is impossible *)
    destruct dp'; discriminate.
Qed.

(** The Standard representation bridge is fully local for the empty dictionary
    prefix: the initial epsilon closure contains exactly the delete chain from
    the origin, so every empty-prefix reachable position is represented by
    reflexive Standard subsumption. *)
Lemma position_subsumed_from_empty_run : forall query n s_mid p,
  automaton_run_from_initial Standard query n [] = Some s_mid ->
  position_reachable query n [] p ->
  is_special p = false ->
  num_errors p <= n ->
  positions_subsume Standard (length query) (positions s_mid) p.
Proof.
  intros query n s_mid [i e sp] Hrun Hreach Hspec Herr.
  unfold automaton_run_from_initial in Hrun.
  simpl in Hrun.
  injection Hrun as Hs_mid. subst s_mid.
  simpl in Hspec. destruct sp; try discriminate.
  simpl in *.
  assert (Hdiag : i = e).
  { apply (reachable_empty_prefix_diagonal query n i e). exact Hreach. }
  subst i.
  apply positions_subsume_standard_refl_in.
  simpl.
  replace e with (0 + e) at 1 by lia.
  replace e with (0 + e) at 2 by lia.
  apply epsilon_closure_reaches_deletes.
  - simpl. left. reflexivity.
  - pose proof (reachable_term_index_bound_query query n [] (std_pos e e) Hreach) as Hbound.
    simpl in Hbound. lia.
  - exact Herr.
Qed.

(** Successful Standard runs only contain query-bounded positions. *)
Lemma standard_run_positions_term_bounded : forall query n dict final,
  automaton_run_from_initial Standard query n dict = Some final ->
  forall p,
    In p (positions final) ->
    term_index p <= length query.
Proof.
  intros query n dict final Hrun p Hin.
  unfold automaton_run_from_initial in Hrun.
  set (init_closed :=
    mkState
      (epsilon_closure (positions (initial_state Standard (length query))) n (length query))
      Standard (length query)) in *.
  assert (Hqlen_init : query_length init_closed = length query).
  { unfold init_closed. simpl. reflexivity. }
  assert (Hinit_reach : forall p0, In p0 (positions init_closed) ->
    position_reachable query n [] p0 /\ is_special p0 = false).
  { intros p0 Hin0.
    unfold init_closed in Hin0. simpl in Hin0.
    apply initial_closed_state_reachable. exact Hin0. }
  assert (Hreach : position_reachable query n dict p).
  { pose proof (automaton_run_preserves_reachable_standard
      query n [] dict init_closed final Hqlen_init Hrun Hinit_reach p Hin) as Hp.
    simpl in Hp. exact Hp. }
  apply reachable_term_index_bound_query with (n := n) (dict_prefix := dict).
  exact Hreach.
Qed.

Lemma initial_closed_state_delete_chain_represented : forall n qlen,
  state_delete_chain_represented n
    (mkState (epsilon_closure [initial_position] n qlen) Standard qlen).
Proof.
  intros n qlen p k Hin Hterm Herr.
  simpl in Hin, Hterm |- *.
  apply initial_closed_delete_chain_represented; assumption.
Qed.

Lemma automaton_run_from_initial_standard_delete_chain_represented : forall
  query n dict final,
  automaton_run_from_initial Standard query n dict = Some final ->
  state_delete_chain_represented n final.
Proof.
  intros query n dict final Hrun.
  unfold automaton_run_from_initial in Hrun.
  apply (automaton_run_standard_delete_chain_represented_from_state
           query n dict
           (mkState (epsilon_closure [initial_position] n (length query))
                    Standard (length query))
           final).
  - apply initial_closed_state_delete_chain_represented.
  - exact Hrun.
Qed.

Lemma represented_delete_successor_from_run : forall query n dict s_mid i e,
  automaton_run_from_initial Standard query n dict = Some s_mid ->
  positions_subsume Standard (length query) (positions s_mid) (std_pos i e) ->
  S i <= length query ->
  S e <= n ->
  positions_subsume Standard (length query) (positions s_mid)
    (std_pos (S i) (S e)).
Proof.
  intros query n dict s_mid i e Hrun [p' [Hin' Hsub']] Hstep Herr_succ.
  pose proof (standard_run_positions_term_bounded query n dict s_mid Hrun
                p' Hin') as Hp'_bound.
  destruct (subsumes_standard_delete_successor_chain
              (length query) p' i e Hsub' Hp'_bound Hstep)
    as [k [Hk_term [Hk_err Hk_sub]]].
  assert (Hqlen_mid : query_length s_mid = length query).
  { pose proof Hrun as Hrun_unfold.
    unfold automaton_run_from_initial in Hrun_unfold.
    rewrite (automaton_run_preserves_query_length Standard query n dict
               (mkState (epsilon_closure [initial_position] n (length query))
                        Standard (length query))
               s_mid Hrun_unfold).
    reflexivity. }
  pose proof (automaton_run_from_initial_standard_delete_chain_represented
                query n dict s_mid Hrun) as Hclosed.
  destruct (Hclosed p' k Hin' ltac:(rewrite Hqlen_mid; exact Hk_term)
                      ltac:(lia)) as [r [Hr_in Hr_sub]].
  rewrite Hqlen_mid in Hr_sub.
  exists r. split; [exact Hr_in|].
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - exact Hk_sub.
Qed.

(** The delete-successor bridge above only needs two local state facts: all
    concrete representatives are query-bounded, and the folded state is closed
    under executable delete-chain representation.  Keeping this state-local
    version separate lets the can-complete invariant avoid referring back to a
    whole run from the initial state. *)
Lemma represented_delete_successor_from_closed_state : forall (query : list Char) n s i e,
  query_length s = length query ->
  (forall p, In p (positions s) -> term_index p <= length query) ->
  state_delete_chain_represented n s ->
  positions_subsume Standard (length query) (positions s) (std_pos i e) ->
  S i <= length query ->
  S e <= n ->
  positions_subsume Standard (length query) (positions s)
    (std_pos (S i) (S e)).
Proof.
  intros query n s i e Hqlen Hbound Hclosed [p' [Hin' Hsub']]
         Hstep Herr_succ.
  destruct (subsumes_standard_delete_successor_chain
              (length query) p' i e Hsub' (Hbound p' Hin') Hstep)
    as [k [Hk_term [Hk_err Hk_sub]]].
  destruct (Hclosed p' k Hin' ltac:(rewrite Hqlen; exact Hk_term)
                     ltac:(lia)) as [r [Hr_in Hr_sub]].
  exists r. split; [exact Hr_in |].
  rewrite Hqlen in Hr_sub.
  eapply subsumes_trans_standard.
  - exact Hr_sub.
  - exact Hk_sub.
Qed.

(** Iterated form of [represented_delete_successor_from_closed_state].  This
    is the catch-up bridge needed when a retained Standard antichain
    representative is behind an exact predecessor: every bounded delete-chain
    endpoint is still represented by the same folded state. *)
Lemma represented_delete_chain_from_closed_state : forall
  (query : list Char) n s i e k,
  query_length s = length query ->
  (forall p, In p (positions s) -> term_index p <= length query) ->
  state_delete_chain_represented n s ->
  positions_subsume Standard (length query) (positions s) (std_pos i e) ->
  i + k <= length query ->
  e + k <= n ->
  positions_subsume Standard (length query) (positions s)
    (std_pos (i + k) (e + k)).
Proof.
  intros query n s i e k Hqlen Hbound Hclosed Hrep Hterm Herr.
  induction k as [|k' IH].
  - replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    exact Hrep.
  - replace (i + S k') with (S (i + k')) by lia.
    replace (e + S k') with (S (e + k')) by lia.
    apply (represented_delete_successor_from_closed_state
             query n s (i + k') (e + k')).
    + exact Hqlen.
    + exact Hbound.
    + exact Hclosed.
    + apply IH; lia.
    + lia.
    + lia.
Qed.

(** A behind representative can be advanced by bounded delete steps to the
    exact predecessor's query index.  The resulting same-index endpoint still
    subsumes the original predecessor, so later match/substitute reasoning can
    use the simpler same-index arithmetic. *)
Lemma represented_behind_catch_up_from_closed_state : forall
  (query : list Char) n s i e p_rep,
  query_length s = length query ->
  (forall p, In p (positions s) -> term_index p <= length query) ->
  state_delete_chain_represented n s ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep <= i ->
  i <= length query ->
  e <= n ->
  let e_catch := num_errors p_rep + (i - term_index p_rep) in
  positions_subsume Standard (length query) (positions s)
    (std_pos i e_catch) /\
  subsumes Standard (length query) (std_pos i e_catch) (std_pos i e) = true.
Proof.
  intros query n s i e p_rep Hqlen Hbound Hclosed Hin_rep Hsub Hbehind
         Hi_bound Herr.
  set (k := i - term_index p_rep).
  set (e_catch := num_errors p_rep + k).
  assert (Hcatch_sub :
    subsumes Standard (length query) (std_pos i e_catch) (std_pos i e) = true).
  { unfold e_catch, k.
    apply subsumes_standard_catch_up_delete_chain_same_index; assumption. }
  assert (Hcatch_err : e_catch <= e).
  { change (num_errors (std_pos i e_catch) <= num_errors (std_pos i e)).
    apply subsumes_standard_errors with (qlen := length query).
    exact Hcatch_sub. }
  split.
  - unfold e_catch, k.
    pose proof (Hclosed p_rep (i - term_index p_rep) Hin_rep) as Hrep_catch.
    simpl in Hrep_catch.
    replace (term_index p_rep + (i - term_index p_rep)) with i in Hrep_catch by lia.
    rewrite Hqlen in Hrep_catch.
    apply Hrep_catch.
    + lia.
    + lia.
  - exact Hcatch_sub.
Qed.

(** Inverting a successful append run exposes the state produced by the prefix.
    This is the converse shape of [run_concat] needed by prefix-induction
    completeness arguments. *)
Lemma automaton_run_app_some_inv : forall alg (query : list Char) n dict1 dict2 s final,
  automaton_run alg query n (dict1 ++ dict2) s = Some final ->
  exists mid,
    automaton_run alg query n dict1 s = Some mid /\
    automaton_run alg query n dict2 mid = Some final.
Proof.
  intros alg query n dict1.
  induction dict1 as [|c rest IH]; intros dict2 s final Hrun.
  - simpl in Hrun.
    exists s. split; [reflexivity | exact Hrun].
  - simpl in Hrun.
    destruct (transition_state alg s c query n) as [s1|] eqn:Htrans;
      [| discriminate].
    destruct (IH dict2 s1 final Hrun) as [mid [Hprefix Hsuffix]].
    exists mid. split.
    + simpl. rewrite Htrans. exact Hprefix.
    + exact Hsuffix.
Qed.

(** * Option C: Can-Complete-To-Final Approach (December 2025)

    The fundamental insight: Instead of tracking that each specific reachable
    position survives antichain construction (which fails because subsumption
    allows different term_indices), we track a weaker property:

    "The state contains SOME position that can complete to a final position"

    Key properties:
    1. Subsumption preserves ability to complete to final
    2. Antichain construction preserves this property
    3. Transitions preserve this property when input has completable positions

    This approach sidesteps the mismatch between:
    - position_subsumes (requires exact term_index)
    - subsumes_standard (allows |i1 - i2| <= e2 - e1)
*)

(** Forward reachability: position p can reach position p_final by processing
    the remaining dictionary characters 'remaining'.

    This is the dual of position_reachable - instead of tracking how we got to p,
    it tracks where p can go from here.
*)
Inductive can_reach (query : list Char) (n : nat) :
  Position -> list Char -> Position -> Prop :=
  | can_reach_done : forall p,
      can_reach query n p [] p
  | can_reach_delete : forall p remaining p_final i e,
      p = std_pos i e ->
      S i <= length query ->
      e < n ->
      can_reach query n (std_pos (S i) (S e)) remaining p_final ->
      can_reach query n p remaining p_final
  | can_reach_match : forall p c remaining p_final i e,
      p = std_pos i e ->
      i < length query ->
      nth_error query i = Some c ->
      can_reach query n (std_pos (S i) e) remaining p_final ->
      can_reach query n p (c :: remaining) p_final
  | can_reach_substitute : forall p c c' remaining p_final i e,
      p = std_pos i e ->
      i < length query ->
      nth_error query i = Some c' ->
      c <> c' ->
      e < n ->
      can_reach query n (std_pos (S i) (S e)) remaining p_final ->
      can_reach query n p (c :: remaining) p_final
  | can_reach_insert : forall p c remaining p_final i e,
      p = std_pos i e ->
      e < n ->
      can_reach query n (std_pos i (S e)) remaining p_final ->
      can_reach query n p (c :: remaining) p_final.

(** A position can complete to a final position: there exists a remaining
    dictionary suffix and a final position that p can reach. *)
Definition can_complete_to_final (query : list Char) (n : nat) (remaining : list Char) (p : Position) : Prop :=
  exists p_final,
    can_reach query n p remaining p_final /\
    term_index p_final = length query /\
    num_errors p_final <= n /\
    is_special p_final = false.

(** A state has the can-complete property if at least one of its positions
    can complete to final. *)
Definition state_has_completable (query : list Char) (n : nat) (remaining : list Char) (s : State) : Prop :=
  exists p, In p (positions s) /\ can_complete_to_final query n remaining p.

(** MergeAndSplit completion relation. This is the same backwards view as
    [can_reach], extended with the merge and split transitions implemented by
    [transition_position_merge_split]. *)
Inductive can_reach_ms (query : list Char) (n : nat) :
  Position -> list Char -> Position -> Prop :=
  | can_reach_ms_done : forall p,
      can_reach_ms query n p [] p
  | can_reach_ms_delete : forall p remaining p_final i e,
      p = std_pos i e ->
      S i <= length query ->
      e < n ->
      can_reach_ms query n (std_pos (S i) (S e)) remaining p_final ->
      can_reach_ms query n p remaining p_final
  | can_reach_ms_match : forall p c remaining p_final i e,
      p = std_pos i e ->
      i < length query ->
      nth_error query i = Some c ->
      can_reach_ms query n (std_pos (S i) e) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final
  | can_reach_ms_substitute : forall p c c' remaining p_final i e,
      p = std_pos i e ->
      i < length query ->
      nth_error query i = Some c' ->
      c <> c' ->
      e < n ->
      can_reach_ms query n (std_pos (S i) (S e)) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final
  | can_reach_ms_insert : forall p c remaining p_final i e,
      p = std_pos i e ->
      e < n ->
      can_reach_ms query n (std_pos i (S e)) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final
  | can_reach_ms_merge : forall p c remaining p_final i e,
      p = std_pos i e ->
      S i < length query ->
      e < n ->
      can_reach_ms query n (std_pos (S (S i)) (S e)) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final
  | can_reach_ms_enter_split : forall p c remaining p_final i e,
      p = std_pos i e ->
      e < n ->
      can_reach_ms query n (special_pos i (S e)) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final
  | can_reach_ms_complete_split : forall p c remaining p_final i e,
      p = special_pos i e ->
      i < length query ->
      can_reach_ms query n (std_pos (S i) e) remaining p_final ->
      can_reach_ms query n p (c :: remaining) p_final.

Definition can_complete_to_final_ms
  (query : list Char) (n : nat) (remaining : list Char) (p : Position) : Prop :=
  exists p_final,
    can_reach_ms query n p remaining p_final /\
    term_index p_final = length query /\
    num_errors p_final <= n /\
    is_special p_final = false.

Definition state_has_ms_completable
  (query : list Char) (n : nat) (remaining : list Char) (s : State) : Prop :=
  exists p, In p (positions s) /\ can_complete_to_final_ms query n remaining p.

(** Basic property: final positions can trivially complete (with empty remaining) *)
Lemma final_position_can_complete : forall query n p,
  term_index p = length query ->
  num_errors p <= n ->
  is_special p = false ->
  can_complete_to_final query n [] p.
Proof.
  intros query n p Hterm Herr Hspec.
  exists p. repeat split.
  - apply can_reach_done.
  - exact Hterm.
  - exact Herr.
  - exact Hspec.
Qed.

(** Reading a character at [i] exposes it as the head of [skipn i]. *)
Lemma skipn_nth_error_cons : forall {A : Type} (l : list A) i x,
  nth_error l i = Some x ->
  skipn i l = x :: skipn (S i) l.
Proof.
  intros A l.
  induction l as [|h t IH]; intros i x Hnth.
  - destruct i; discriminate.
  - destruct i as [|i'].
    + simpl in Hnth. inversion Hnth. subst. reflexivity.
    + simpl in Hnth. simpl. apply IH. exact Hnth.
Qed.

(** The recursive Damerau distance is complete for the semantic
    [position_reachable_damerau] relation.  The proof follows the executable
    recurrence and constructs the corresponding front-to-back path. *)
Lemma min3_plus_le_cases : forall a b c e n,
  min3 a b c + e <= n ->
  a + e <= n \/ b + e <= n \/ c + e <= n.
Proof.
  intros a b c e n H.
  unfold min3 in H.
  destruct (Nat.min_dec a (min b c)) as [Hmin | Hmin].
  - rewrite Hmin in H. left. exact H.
  - rewrite Hmin in H.
    destruct (Nat.min_dec b c) as [Hbc | Hbc].
    + rewrite Hbc in H. right. left. exact H.
    + rewrite Hbc in H. right. right. exact H.
Qed.

Lemma min4_plus_le_cases : forall a b c d e n,
  min4 a b c d + e <= n ->
  a + e <= n \/ b + e <= n \/ c + e <= n \/ d + e <= n.
Proof.
  intros a b c d e n H.
  unfold min4 in H.
  destruct (Nat.min_dec (min a b) (min c d)) as [Houter | Houter].
  - rewrite Houter in H.
    destruct (Nat.min_dec a b) as [Hab | Hab].
    + rewrite Hab in H. left. exact H.
    + rewrite Hab in H. right. left. exact H.
  - rewrite Houter in H.
    destruct (Nat.min_dec c d) as [Hcd | Hcd].
    + rewrite Hcd in H. right. right. left. exact H.
    + rewrite Hcd in H. right. right. right. exact H.
Qed.

Lemma skipn_nil_index : forall {A : Type} (l : list A) i,
  i <= length l ->
  skipn i l = [] ->
  i = length l.
Proof.
  intros A l i Hle Hskip.
  assert (Hlen : length (skipn i l) = 0).
  { rewrite Hskip. reflexivity. }
  rewrite length_skipn in Hlen. lia.
Qed.

Lemma nth_error_of_skipn_cons : forall {A : Type} (l : list A) i x xs,
  skipn i l = x :: xs ->
  nth_error l i = Some x.
Proof.
  intros A l i x xs Hskip.
  replace i with (i + 0) by lia.
  rewrite <- nth_error_skipn.
  rewrite Hskip. reflexivity.
Qed.

Lemma skipn_succ_of_skipn_cons : forall {A : Type} (l : list A) i x xs,
  skipn i l = x :: xs ->
  skipn (S i) l = xs.
Proof.
  intros A l i x xs Hskip.
  pose proof (nth_error_of_skipn_cons l i x xs Hskip) as Hnth.
  pose proof (skipn_nth_error_cons l i x Hnth) as Hcons.
  rewrite Hskip in Hcons. injection Hcons as Htail.
  symmetry. exact Htail.
Qed.

Lemma nth_error_second_of_skipn_cons2 : forall {A : Type} (l : list A) i x y xs,
  skipn i l = x :: y :: xs ->
  nth_error l (S i) = Some y.
Proof.
  intros A l i x y xs Hskip.
  replace (S i) with (i + 1) by lia.
  rewrite <- nth_error_skipn.
  rewrite Hskip. reflexivity.
Qed.

Lemma skipn_two_of_skipn_cons2 : forall {A : Type} (l : list A) i x y xs,
  skipn i l = x :: y :: xs ->
  skipn (S (S i)) l = xs.
Proof.
  intros A l i x y xs Hskip.
  change xs with (skipn 2 (x :: y :: xs)).
  rewrite <- Hskip.
  rewrite skipn_skipn.
  replace (2 + i) with (S (S i)) by lia.
  reflexivity.
Qed.

Lemma subst_cost_le_one_local : forall c d,
  subst_cost c d <= 1.
Proof.
  intros c d. unfold subst_cost. destruct (char_eq c d); lia.
Qed.

Lemma damerau_lev_distance_reachable_final_gen :
  forall m query dict qi di n e,
  length (skipn qi query) + length (skipn di dict) = m ->
  qi <= length query ->
  di <= length dict ->
  damerau_lev_distance (skipn qi query) (skipn di dict) + e <= n ->
  position_reachable_damerau query n (firstn di dict) (std_pos qi e) ->
  exists p,
    position_reachable_damerau query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= n.
Proof.
  induction m as [m IH] using lt_wf_ind.
  intros query dict qi di n e Hmeasure Hqi Hdi Hdist Hreach.
  destruct (skipn qi query) as [|qc qtail] eqn:Hq.
  - destruct (skipn di dict) as [|dc dtail] eqn:Hd.
    + assert (Hqi_end : qi = length query)
        by (apply skipn_nil_index; assumption).
      assert (Hdi_end : di = length dict)
        by (apply skipn_nil_index; assumption).
      exists (std_pos qi e).
      rewrite Hdi_end in Hreach.
      rewrite firstn_all in Hreach.
      split.
      * exact Hreach.
      * split.
        -- simpl. exact Hqi_end.
        -- split.
           ++ reflexivity.
           ++ simpl. rewrite damerau_lev_empty_left in Hdist. simpl in Hdist. lia.
    + pose proof (nth_error_of_skipn_cons dict di dc dtail Hd) as Hnth_d.
      pose proof (skipn_succ_of_skipn_cons dict di dc dtail Hd) as Hdtail.
      assert (Hdi' : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
	      assert (Hinsert : e < n).
	      { rewrite damerau_lev_empty_left in Hdist. simpl in Hdist. lia. }
      assert (Hreach' :
        position_reachable_damerau query n (firstn (S di) dict)
          (std_pos qi (S e))).
      { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
        apply reach_damerau_insert.
        - exact Hreach.
        - exact Hinsert. }
      eapply IH with
        (m := length (skipn qi query) + length (skipn (S di) dict));
        try exact Hreach'; try exact Hqi; try exact Hdi'.
	      * rewrite Hq, Hdtail. simpl in Hmeasure. simpl. lia.
	      * reflexivity.
	      * rewrite Hq, Hdtail.
	        rewrite damerau_lev_empty_left.
	        rewrite damerau_lev_empty_left in Hdist. simpl in Hdist. lia.
  - pose proof (nth_error_of_skipn_cons query qi qc qtail Hq) as Hnth_q.
    pose proof (skipn_succ_of_skipn_cons query qi qc qtail Hq) as Hqtail.
    assert (Hqi' : S qi <= length query).
    { apply nth_error_Some. rewrite Hnth_q. discriminate. }
	    destruct (skipn di dict) as [|dc dtail] eqn:Hd.
	    + assert (Hdelete : e < n).
	      { rewrite damerau_lev_empty_right in Hdist. simpl in Hdist. lia. }
      assert (Hreach' :
        position_reachable_damerau query n (firstn di dict)
          (std_pos (S qi) (S e))).
      { apply reach_damerau_delete.
        - exact Hreach.
        - exact Hqi'.
        - exact Hdelete. }
      eapply IH with
        (m := length (skipn (S qi) query) + length (skipn di dict));
        try exact Hreach'; try exact Hqi'; try exact Hdi.
	      * rewrite Hqtail, Hd. simpl in Hmeasure. simpl. lia.
	      * reflexivity.
	      * rewrite Hqtail, Hd.
	        rewrite damerau_lev_empty_right.
	        rewrite damerau_lev_empty_right in Hdist. simpl in Hdist. lia.
    + pose proof (nth_error_of_skipn_cons dict di dc dtail Hd) as Hnth_d.
      pose proof (skipn_succ_of_skipn_cons dict di dc dtail Hd) as Hdtail.
      assert (Hdi' : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
      destruct qtail as [|qc2 qtail2].
      * destruct dtail as [|dc2 dtail2].
	        -- rewrite damerau_lev_single in Hdist.
           destruct (char_eq qc dc) eqn:Hsame.
           ++ apply char_eq_eq in Hsame. subst dc.
              assert (Hreach' :
                position_reachable_damerau query n (firstn (S di) dict)
                  (std_pos (S qi) e)).
              { rewrite (firstn_S_snoc_nth_error dict di qc Hnth_d).
                apply reach_damerau_match with (c := qc).
                - exact Hreach.
                - apply nth_error_Some. rewrite Hnth_q. discriminate.
                - exact Hnth_q. }
              eapply IH with
                (m := length (skipn (S qi) query) +
                      length (skipn (S di) dict));
                try exact Hreach'; try exact Hqi'; try exact Hdi'.
	              ** rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
	              ** rewrite Hqtail, Hdtail.
	                 rewrite damerau_lev_empty_left. simpl. simpl in Hdist. lia.
           ++ assert (Hsub : e < n) by (simpl in Hdist; lia).
              assert (Hneq : dc <> qc).
              { intro Heq. subst dc. rewrite char_eq_refl in Hsame. discriminate. }
              assert (Hreach' :
                position_reachable_damerau query n (firstn (S di) dict)
                  (std_pos (S qi) (S e))).
              { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                apply reach_damerau_substitute with (c := dc) (c' := qc).
                - exact Hreach.
                - apply nth_error_Some. rewrite Hnth_q. discriminate.
                - exact Hnth_q.
                - exact Hneq.
                - exact Hsub. }
              eapply IH with
                (m := length (skipn (S qi) query) +
                      length (skipn (S di) dict));
                try exact Hreach'; try exact Hqi'; try exact Hdi'.
	              ** rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
	              ** rewrite Hqtail, Hdtail.
	                 rewrite damerau_lev_empty_left. simpl. simpl in Hdist. lia.
        -- assert (Hq_one : skipn qi query = [qc]) by exact Hq.
           assert (Hd_multi : skipn di dict = dc :: dc2 :: dtail2) by exact Hd.
	           rewrite damerau_lev_single_multi in Hdist.
           destruct (min3_plus_le_cases _ _ _ e n Hdist)
             as [Hdel | [Hins | Hsubcase]].
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn di dict)
                  (std_pos (S qi) (S e))).
              { apply reach_damerau_delete.
                - exact Hreach.
                - exact Hqi'.
                - lia. }
              eapply IH with
                (m := length (skipn (S qi) query) + length (skipn di dict));
                try exact Hreach'; try exact Hqi'; try exact Hdi.
	              ** rewrite Hqtail, Hd. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
		              ** rewrite Hqtail, Hd.
                         rewrite damerau_lev_empty_left.
                         rewrite damerau_lev_empty_left in Hdel.
                         simpl in Hdel. simpl. lia.
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn (S di) dict)
                  (std_pos qi (S e))).
              { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                apply reach_damerau_insert.
                - exact Hreach.
                - lia. }
              eapply IH with
                (m := length (skipn qi query) +
                      length (skipn (S di) dict));
                try exact Hreach'; try exact Hqi; try exact Hdi'.
	              ** rewrite Hq, Hdtail. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
		              ** rewrite Hq, Hdtail. simpl. lia.
           ++ destruct (char_eq qc dc) eqn:Hsame.
              ** apply char_eq_eq in Hsame. subst dc.
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) e)).
                 { rewrite (firstn_S_snoc_nth_error dict di qc Hnth_d).
                   apply reach_damerau_match with (c := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail, Hdtail.
	                     unfold subst_cost in Hsubcase.
	                     rewrite char_eq_refl in Hsubcase. simpl in Hsubcase.
	                     simpl. lia.
              ** assert (Hneq : dc <> qc).
                 { intro Heq. subst dc. rewrite char_eq_refl in Hsame. discriminate. }
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) (S e))).
                 { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                   apply reach_damerau_substitute with (c := dc) (c' := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q.
                   - exact Hneq.
                   - unfold subst_cost in Hsubcase. rewrite Hsame in Hsubcase. lia. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
		                 --- rewrite Hqtail, Hdtail.
		                     unfold subst_cost in Hsubcase.
		                     rewrite Hsame in Hsubcase.
                             rewrite damerau_lev_empty_left.
                             rewrite damerau_lev_empty_left in Hsubcase.
                             simpl in Hsubcase. simpl. lia.
      * pose proof (nth_error_second_of_skipn_cons2 query qi qc qc2 qtail2 Hq)
          as Hnth_q2.
        pose proof (skipn_two_of_skipn_cons2 query qi qc qc2 qtail2 Hq)
          as Hqtail2.
        assert (Hqi2 : S (S qi) <= length query).
        { apply nth_error_Some. rewrite Hnth_q2. discriminate. }
        destruct dtail as [|dc2 dtail2].
        -- assert (Hq_multi : skipn qi query = qc :: qc2 :: qtail2) by exact Hq.
           assert (Hd_one : skipn di dict = [dc]) by exact Hd.
	           rewrite damerau_lev_multi_single in Hdist.
           destruct (min3_plus_le_cases _ _ _ e n Hdist)
             as [Hdel | [Hins | Hsubcase]].
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn di dict)
                  (std_pos (S qi) (S e))).
              { apply reach_damerau_delete.
                - exact Hreach.
                - exact Hqi'.
                - lia. }
              eapply IH with
                (m := length (skipn (S qi) query) + length (skipn di dict));
                try exact Hreach'; try exact Hqi'; try exact Hdi.
	              ** rewrite Hqtail, Hd. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
	              ** rewrite Hqtail, Hd. simpl. lia.
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn (S di) dict)
                  (std_pos qi (S e))).
              { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                apply reach_damerau_insert.
                - exact Hreach.
                - lia. }
              eapply IH with
                (m := length (skipn qi query) +
                      length (skipn (S di) dict));
                try exact Hreach'; try exact Hqi; try exact Hdi'.
	              ** rewrite Hq, Hdtail. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
		              ** rewrite Hq, Hdtail.
                         rewrite damerau_lev_empty_right.
                         rewrite damerau_lev_empty_right in Hins.
                         simpl in Hins. simpl. lia.
           ++ destruct (char_eq qc dc) eqn:Hsame.
              ** apply char_eq_eq in Hsame. subst dc.
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) e)).
                 { rewrite (firstn_S_snoc_nth_error dict di qc Hnth_d).
                   apply reach_damerau_match with (c := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail, Hdtail.
	                     unfold subst_cost in Hsubcase.
	                     rewrite char_eq_refl in Hsubcase. simpl in Hsubcase.
	                     simpl. lia.
              ** assert (Hneq : dc <> qc).
                 { intro Heq. subst dc. rewrite char_eq_refl in Hsame. discriminate. }
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) (S e))).
                 { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                   apply reach_damerau_substitute with (c := dc) (c' := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q.
                   - exact Hneq.
                   - unfold subst_cost in Hsubcase. rewrite Hsame in Hsubcase. lia. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail, Hdtail.
	                     unfold subst_cost in Hsubcase.
	                     rewrite Hsame in Hsubcase. simpl. lia.
        -- pose proof (nth_error_second_of_skipn_cons2 dict di dc dc2 dtail2 Hd)
             as Hnth_d2.
           pose proof (skipn_two_of_skipn_cons2 dict di dc dc2 dtail2 Hd)
             as Hdtail2.
           assert (Hdi2 : S (S di) <= length dict).
           { apply nth_error_Some. rewrite Hnth_d2. discriminate. }
	           rewrite damerau_lev_cons2 in Hdist.
           destruct (min4_plus_le_cases _ _ _ _ e n Hdist)
             as [Hdel | [Hins | [Hsubcase | Htranscase]]].
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn di dict)
                  (std_pos (S qi) (S e))).
              { apply reach_damerau_delete.
                - exact Hreach.
                - exact Hqi'.
                - lia. }
              eapply IH with
                (m := length (skipn (S qi) query) + length (skipn di dict));
                try exact Hreach'; try exact Hqi'; try exact Hdi.
	              ** rewrite Hqtail, Hd. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
	              ** rewrite Hqtail, Hd. simpl. lia.
           ++ assert (Hreach' :
                position_reachable_damerau query n (firstn (S di) dict)
                  (std_pos qi (S e))).
              { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                apply reach_damerau_insert.
                - exact Hreach.
                - lia. }
              eapply IH with
                (m := length (skipn qi query) +
                      length (skipn (S di) dict));
                try exact Hreach'; try exact Hqi; try exact Hdi'.
	              ** rewrite Hq, Hdtail. simpl in Hmeasure. simpl. lia.
	              ** reflexivity.
	              ** rewrite Hq, Hdtail. simpl. lia.
           ++ destruct (char_eq qc dc) eqn:Hsame.
              ** apply char_eq_eq in Hsame. subst dc.
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) e)).
                 { rewrite (firstn_S_snoc_nth_error dict di qc Hnth_d).
                   apply reach_damerau_match with (c := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail, Hdtail.
	                     unfold subst_cost in Hsubcase.
	                     rewrite char_eq_refl in Hsubcase. simpl in Hsubcase.
	                     simpl. lia.
              ** assert (Hneq : dc <> qc).
                 { intro Heq. subst dc. rewrite char_eq_refl in Hsame. discriminate. }
                 assert (Hreach' :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (std_pos (S qi) (S e))).
                 { rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                   apply reach_damerau_substitute with (c := dc) (c' := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q.
                   - exact Hneq.
                   - unfold subst_cost in Hsubcase. rewrite Hsame in Hsubcase. lia. }
                 eapply IH with
                   (m := length (skipn (S qi) query) +
                         length (skipn (S di) dict));
                   try exact Hreach'; try exact Hqi'; try exact Hdi'.
	                 --- rewrite Hqtail, Hdtail. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail, Hdtail.
	                     unfold subst_cost in Hsubcase.
	                     rewrite Hsame in Hsubcase. simpl. lia.
           ++ unfold trans_cost_calc in Htranscase.
              destruct (andb (char_eq qc dc2) (char_eq qc2 dc)) eqn:Hswap.
              ** apply andb_prop in Hswap as [Hqc_dc2 Hqc2_dc].
                 apply char_eq_eq in Hqc_dc2.
                 apply char_eq_eq in Hqc2_dc.
                 subst dc dc2.
                 assert (Henter :
                   position_reachable_damerau query n (firstn (S di) dict)
                     (special_pos qi (S e))).
                 { rewrite (firstn_S_snoc_nth_error dict di qc2 Hnth_d).
                   apply reach_damerau_enter_transpose with (c_next := qc).
                   - exact Hreach.
                   - apply nth_error_Some. rewrite Hnth_q2. discriminate.
                   - exact Hnth_q2.
                   - exact Hnth_q.
                   - lia. }
                 assert (Hcomplete :
                   position_reachable_damerau query n (firstn (S (S di)) dict)
                     (std_pos (S (S qi)) (S e))).
                 { rewrite (firstn_S_snoc_nth_error dict (S di) qc Hnth_d2).
                   apply reach_damerau_complete_transpose.
                   - exact Henter.
                   - apply nth_error_Some. rewrite Hnth_q. discriminate.
                   - exact Hnth_q. }
                 eapply IH with
                   (m := length (skipn (S (S qi)) query) +
                         length (skipn (S (S di)) dict));
                   try exact Hcomplete; try exact Hqi2; try exact Hdi2.
	                 --- rewrite Hqtail2, Hdtail2. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail2, Hdtail2.
	                     simpl in Htranscase. simpl. lia.
              ** assert (Hstep1 :
                   (exists e1,
                     e1 <= S e /\
                     position_reachable_damerau query n (firstn (S di) dict)
                       (std_pos (S qi) e1))).
                 { destruct (char_eq qc dc) eqn:Hsame.
                   - apply char_eq_eq in Hsame. subst dc.
                     exists e. split; [lia|].
                     rewrite (firstn_S_snoc_nth_error dict di qc Hnth_d).
                     apply reach_damerau_match with (c := qc).
                     + exact Hreach.
                     + apply nth_error_Some. rewrite Hnth_q. discriminate.
                     + exact Hnth_q.
                   - exists (S e). split; [lia|].
                     assert (Hneq : dc <> qc).
                     { intro Heq. subst dc. rewrite char_eq_refl in Hsame. discriminate. }
                     rewrite (firstn_S_snoc_nth_error dict di dc Hnth_d).
                     apply reach_damerau_substitute with (c := dc) (c' := qc).
                     + exact Hreach.
                     + apply nth_error_Some. rewrite Hnth_q. discriminate.
                     + exact Hnth_q.
                     + exact Hneq.
                     + simpl in Htranscase. lia. }
                 destruct Hstep1 as [e1 [He1_bound Hreach1]].
                 assert (Hstep2 :
                   exists e2,
                     e2 <= S (S e) /\
                     position_reachable_damerau query n (firstn (S (S di)) dict)
                       (std_pos (S (S qi)) e2)).
                 { destruct (char_eq qc2 dc2) eqn:Hsame2.
                   - apply char_eq_eq in Hsame2. subst dc2.
                     exists e1. split; [lia|].
                     rewrite (firstn_S_snoc_nth_error dict (S di) qc2 Hnth_d2).
                     apply reach_damerau_match with (c := qc2).
                     + exact Hreach1.
                     + apply nth_error_Some. rewrite Hnth_q2. discriminate.
                     + exact Hnth_q2.
                   - exists (S e1). split; [lia|].
                     assert (Hneq : dc2 <> qc2).
                     { intro Heq. subst dc2. rewrite char_eq_refl in Hsame2. discriminate. }
                     rewrite (firstn_S_snoc_nth_error dict (S di) dc2 Hnth_d2).
                     apply reach_damerau_substitute with (c := dc2) (c' := qc2).
                     + exact Hreach1.
                     + apply nth_error_Some. rewrite Hnth_q2. discriminate.
                     + exact Hnth_q2.
                     + exact Hneq.
                     + simpl in Htranscase. lia. }
                 destruct Hstep2 as [e2 [He2_bound Hreach2]].
                 eapply IH with
                   (m := length (skipn (S (S qi)) query) +
                         length (skipn (S (S di)) dict));
                   try exact Hreach2; try exact Hqi2; try exact Hdi2.
	                 --- rewrite Hqtail2, Hdtail2. simpl in Hmeasure. simpl. lia.
	                 --- reflexivity.
	                 --- rewrite Hqtail2, Hdtail2.
	                     simpl in Htranscase. lia.
Qed.

Theorem transposition_reachable_final : forall query dict n,
  damerau_lev_distance query dict <= n ->
  exists p,
    position_reachable_damerau query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= n.
Proof.
  intros query dict n Hdist.
  replace query with (skipn 0 query) at 1 by reflexivity.
  replace dict with (skipn 0 dict) at 1 by reflexivity.
  eapply damerau_lev_distance_reachable_final_gen
    with (m := length query + length dict) (qi := 0) (di := 0) (e := 0).
  - simpl. reflexivity.
  - lia.
  - lia.
  - simpl. lia.
  - simpl. apply reach_damerau_initial.
Qed.

(** A valid edit sequence can be read backward as a [can_reach] completion.
    This is the can-complete analogue of [traceable_implies_reachable_gen]. *)
Lemma valid_sequence_can_reach_gen : forall query dict qi di ops n e,
  valid_edit_sequence query dict qi di ops ->
  sequence_cost ops + e <= n ->
  di <= length dict ->
  exists p_final,
    can_reach query n (std_pos qi e) (skipn di dict) p_final /\
    term_index p_final = length query /\
    is_special p_final = false /\
    num_errors p_final <= sequence_cost ops + e.
Proof.
  intros query dict qi di ops n e Hvalid.
  revert n e.
  induction Hvalid as
    [query' dict'
    | query' dict' qi' di' qi'' di'' op ops'
        Hop Happly Hrest IH];
    intros n e Hcost Hdi.
  - exists (std_pos (length query') e).
    rewrite skipn_all2 by lia.
    repeat split; simpl; try lia.
    apply can_reach_done.
  - simpl in Hcost.
    destruct op as [c | c1 c2 | c | c].
    + simpl in Hcost.
      simpl in Happly. inversion Happly as [[Hqi Hdi']]. subst qi'' di''.
      simpl in Hop. destruct Hop as [Hq Hd].
      assert (Hdict_tail : S di' <= length dict').
      { assert (Hlt : di' < length dict').
        { apply nth_error_Some. rewrite Hd. discriminate. }
        lia. }
      assert (Hcost_tail : sequence_cost ops' + e <= n) by lia.
      destruct (IH n e Hcost_tail Hdict_tail)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite (skipn_nth_error_cons dict' di' c Hd).
      repeat split; try exact Hterm; try exact Hspec; try exact Herr.
      apply (can_reach_match query' n (std_pos qi' e) c
               (skipn (S di') dict') p_final qi' e).
      * reflexivity.
      * apply nth_error_Some. rewrite Hq. discriminate.
      * exact Hq.
      * exact Hreach.
    + simpl in Hcost.
      simpl in Happly. inversion Happly as [[Hqi Hdi']]. subst qi'' di''.
      simpl in Hop. destruct Hop as [Hq [Hd Hneq]].
      assert (Hdict_tail : S di' <= length dict').
      { assert (Hlt : di' < length dict').
        { apply nth_error_Some. rewrite Hd. discriminate. }
        lia. }
      assert (Hcost_tail : sequence_cost ops' + S e <= n) by lia.
      destruct (IH n (S e) Hcost_tail Hdict_tail)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite (skipn_nth_error_cons dict' di' c2 Hd).
      repeat split; try exact Hterm; try exact Hspec; simpl; try lia.
      apply (can_reach_substitute query' n (std_pos qi' e) c2 c1
               (skipn (S di') dict') p_final qi' e).
      * reflexivity.
      * apply nth_error_Some. rewrite Hq. discriminate.
      * exact Hq.
      * intro Heq. apply Hneq. symmetry. exact Heq.
      * lia.
      * exact Hreach.
    + simpl in Hcost.
      simpl in Happly. inversion Happly as [[Hqi Hdi']]. subst qi'' di''.
      simpl in Hop.
      assert (Hcost_tail : sequence_cost ops' + S e <= n) by lia.
      destruct (IH n (S e) Hcost_tail Hdi)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      repeat split; try exact Hterm; try exact Hspec; simpl; try lia.
      apply (can_reach_delete query' n (std_pos qi' e)
               (skipn di' dict') p_final qi' e).
      * reflexivity.
      * assert (Hlt : qi' < length query').
        { apply nth_error_Some. rewrite Hop. discriminate. }
        lia.
      * lia.
      * exact Hreach.
    + simpl in Hcost.
      simpl in Happly. inversion Happly as [[Hqi Hdi']]. subst qi'' di''.
      simpl in Hop.
      assert (Hdict_tail : S di' <= length dict').
      { assert (Hlt : di' < length dict').
        { apply nth_error_Some. rewrite Hop. discriminate. }
        lia. }
      assert (Hcost_tail : sequence_cost ops' + S e <= n) by lia.
      destruct (IH n (S e) Hcost_tail Hdict_tail)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite (skipn_nth_error_cons dict' di' c Hop).
      repeat split; try exact Hterm; try exact Hspec; simpl; try lia.
      apply (can_reach_insert query' n (std_pos qi' e) c
               (skipn (S di') dict') p_final qi' e).
      * reflexivity.
      * lia.
      * exact Hreach.
Qed.

Lemma valid_sequence_can_complete_initial : forall query dict n ops,
  valid_edit_sequence query dict 0 0 ops ->
  sequence_cost ops <= n ->
  can_complete_to_final query n dict initial_position.
Proof.
  intros query dict n ops Hvalid Hcost.
  destruct (valid_sequence_can_reach_gen query dict 0 0 ops n 0
              Hvalid ltac:(simpl; lia) ltac:(lia))
    as [p_final [Hreach [Hterm [Hspec Herr]]]].
  unfold can_complete_to_final.
  exists p_final.
  unfold initial_position.
  simpl in Hreach.
  repeat split; try exact Hterm; try exact Hspec; try lia.
  exact Hreach.
Qed.

(** A valid merge/split edit sequence can be read as a semantic
    MergeAndSplit-reachable path. This is the sequence-side bridge needed by
    direct MergeAndSplit completeness: [ms_seq_exists] supplies the optimal
    sequence, while this lemma turns that sequence into the reachability shape
    used by the automaton proofs. *)
Lemma ms_valid_sequence_reachable_merge_split_gen : forall query dict ops qi di n e,
  apply_ms_seq ops (skipn qi query) (skipn di dict) = Some ([], []) ->
  ms_seq_cost ops + e <= n ->
  qi <= length query ->
  di <= length dict ->
  position_reachable_merge_split query n (firstn di dict) (std_pos qi e) ->
  exists p,
    position_reachable_merge_split query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= ms_seq_cost ops + e.
Proof.
  intros query dict ops.
  induction ops as [|op rest IH]; intros qi di n e Hvalid Hcost Hqi Hdi Hreach.
  - simpl in Hvalid.
    injection Hvalid as Hsrc Htgt.
    assert (Hqi_end : qi = length query).
    { assert (Hlen : length (skipn qi query) = 0) by (rewrite Hsrc; reflexivity).
      rewrite length_skipn in Hlen. lia. }
    assert (Hdi_end : di = length dict).
    { assert (Hlen : length (skipn di dict) = 0) by (rewrite Htgt; reflexivity).
      rewrite length_skipn in Hlen. lia. }
    exists (std_pos qi e).
    repeat split; simpl; try lia.
    rewrite firstn_all2 in Hreach by lia.
    exact Hreach.
  - simpl in Hvalid, Hcost.
    destruct op as [c_del | c_ins | c_src c_tgt | c1 c2 d | c_split d1 d2].
    + (* Delete from source/query. *)
      destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_del q) eqn:Hchar; [|discriminate].
      apply char_eq_eq in Hchar. subst q.
      assert (Hnth_q : nth_error query qi = Some c_del).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hqtail_eq : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_del Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hqtail_eq in Hvalid.
      assert (Hreach' :
        position_reachable_merge_split query n (firstn di dict) (std_pos (S qi) (S e))).
      { apply reach_ms_delete.
        - exact Hreach.
        - apply nth_error_Some. rewrite Hnth_q. discriminate.
        - simpl in Hcost. lia. }
      assert (Hcost_tail : ms_seq_cost rest + S e <= n) by (simpl in Hcost; lia).
      assert (Hqi_tail_bound : S qi <= length query).
      { apply nth_error_Some. rewrite Hnth_q. discriminate. }
      destruct (IH (S qi) di n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi Hreach')
        as [p [Hp [Hterm [Hspec Herr]]]].
      exists p. repeat split; try assumption; simpl; lia.
    + (* Insert into target/dictionary. *)
      destruct (skipn di dict) as [|d dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_ins d) eqn:Hchar; [|discriminate].
      apply char_eq_eq in Hchar. subst d.
      assert (Hnth_d : nth_error dict di = Some c_ins).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hdtail_eq : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di c_ins Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hdtail_eq in Hvalid.
      assert (Hreach' :
        position_reachable_merge_split query n (firstn (S di) dict) (std_pos qi (S e))).
      { rewrite (firstn_S_snoc_nth_error dict di c_ins Hnth_d).
        apply reach_ms_insert.
        - exact Hreach.
        - simpl in Hcost. lia. }
      assert (Hcost_tail : ms_seq_cost rest + S e <= n) by (simpl in Hcost; lia).
      assert (Hdi_tail_bound : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
      destruct (IH qi (S di) n (S e) Hvalid Hcost_tail Hqi Hdi_tail_bound Hreach')
        as [p [Hp [Hterm [Hspec Herr]]]].
      exists p. repeat split; try assumption; simpl; lia.
    + (* Match/substitute. *)
      destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      destruct (skipn di dict) as [|d dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_src q) eqn:Hsrc; [|discriminate].
      destruct (char_eq c_tgt d) eqn:Htgt; [|discriminate].
      apply char_eq_eq in Hsrc. subst q.
      apply char_eq_eq in Htgt. subst d.
      assert (Hnth_q : nth_error query qi = Some c_src).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hnth_d : nth_error dict di = Some c_tgt).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hqtail_eq : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_src Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      assert (Hdtail_eq : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di c_tgt Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hqtail_eq, Hdtail_eq in Hvalid.
      destruct (char_eq c_src c_tgt) eqn:Hsame.
      * apply char_eq_eq in Hsame. subst c_tgt.
        assert (Hreach' :
          position_reachable_merge_split query n (firstn (S di) dict) (std_pos (S qi) e)).
        { rewrite (firstn_S_snoc_nth_error dict di c_src Hnth_d).
          apply reach_ms_match with (c := c_src).
          - exact Hreach.
          - apply nth_error_Some. rewrite Hnth_q. discriminate.
          - exact Hnth_q. }
        assert (Hcost_tail : ms_seq_cost rest + e <= n).
        { simpl in Hcost. unfold subst_cost in Hcost. rewrite char_eq_refl in Hcost. lia. }
        assert (Hqi_tail_bound : S qi <= length query).
        { apply nth_error_Some. rewrite Hnth_q. discriminate. }
        assert (Hdi_tail_bound : S di <= length dict).
        { apply nth_error_Some. rewrite Hnth_d. discriminate. }
        destruct (IH (S qi) (S di) n e Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound Hreach')
          as [p [Hp [Hterm [Hspec Herr]]]].
        exists p. repeat split; try assumption; simpl; unfold subst_cost; rewrite char_eq_refl; lia.
      * assert (Hreach' :
          position_reachable_merge_split query n (firstn (S di) dict) (std_pos (S qi) (S e))).
        { rewrite (firstn_S_snoc_nth_error dict di c_tgt Hnth_d).
          apply reach_ms_substitute with (c := c_tgt) (c' := c_src).
          - exact Hreach.
          - apply nth_error_Some. rewrite Hnth_q. discriminate.
          - exact Hnth_q.
          - simpl in Hcost. unfold subst_cost in Hcost. rewrite Hsame in Hcost. lia. }
        assert (Hcost_tail : ms_seq_cost rest + S e <= n).
        { simpl in Hcost. unfold subst_cost in Hcost. rewrite Hsame in Hcost. lia. }
        assert (Hqi_tail_bound : S qi <= length query).
        { apply nth_error_Some. rewrite Hnth_q. discriminate. }
        assert (Hdi_tail_bound : S di <= length dict).
        { apply nth_error_Some. rewrite Hnth_d. discriminate. }
        destruct (IH (S qi) (S di) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound Hreach')
          as [p [Hp [Hterm [Hspec Herr]]]].
        exists p. repeat split; try assumption; simpl; unfold subst_cost; rewrite Hsame; lia.
    + (* Merge two source/query characters into one target/dictionary char. *)
      destruct (skipn qi query) as [|q1 qtail1] eqn:Hqtail1; [discriminate|].
      destruct qtail1 as [|q2 qtail2]; [discriminate|].
      destruct (skipn di dict) as [|d' dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c1 q1) eqn:Hc1; [|discriminate].
      destruct (char_eq c2 q2) eqn:Hc2; [|discriminate].
      destruct (char_eq d d') eqn:Hd; [|discriminate].
      apply char_eq_eq in Hc1. subst q1.
      apply char_eq_eq in Hc2. subst q2.
      apply char_eq_eq in Hd. subst d'.
      assert (Hnth_q1 : nth_error query qi = Some c1).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail1. reflexivity. }
      assert (Hnth_q2 : nth_error query (S qi) = Some c2).
      { replace (S qi) with (qi + 1) by lia.
        rewrite <- (nth_error_skipn qi query 1). rewrite Hqtail1. reflexivity. }
      assert (Hnth_d : nth_error dict di = Some d).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hquery_tail : qtail2 = skipn (S (S qi)) query).
      { change qtail2 with (skipn 2 (c1 :: c2 :: qtail2)).
        rewrite <- Hqtail1.
        rewrite skipn_skipn.
        replace (2 + qi) with (S (S qi)) by lia.
        reflexivity. }
      assert (Hdict_tail : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di d Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hquery_tail, Hdict_tail in Hvalid.
      assert (Hreach' :
        position_reachable_merge_split query n (firstn (S di) dict)
          (std_pos (S (S qi)) (S e))).
      { rewrite (firstn_S_snoc_nth_error dict di d Hnth_d).
        apply reach_ms_merge.
        - exact Hreach.
        - apply nth_error_Some. rewrite Hnth_q2. discriminate.
        - simpl in Hcost. unfold merge_cost, can_merge in Hcost. lia. }
      assert (Hcost_tail : ms_seq_cost rest + S e <= n).
      { simpl in Hcost. unfold merge_cost, can_merge in Hcost. lia. }
      assert (Hqi_tail_bound : S (S qi) <= length query).
      { apply nth_error_Some. rewrite Hnth_q2. discriminate. }
      assert (Hdi_tail_bound : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
      destruct (IH (S (S qi)) (S di) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound Hreach')
        as [p [Hp [Hterm [Hspec Herr]]]].
      exists p. repeat split; try assumption; simpl; unfold merge_cost, can_merge; lia.
    + (* Split one source/query character into two target/dictionary chars. *)
      destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      destruct (skipn di dict) as [|d1' dtail1] eqn:Hdtail1; [discriminate|].
      destruct dtail1 as [|d2' dtail2]; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_split q) eqn:Hsrc; [|discriminate].
      destruct (char_eq d1 d1') eqn:Hd1; [|discriminate].
      destruct (char_eq d2 d2') eqn:Hd2; [|discriminate].
      apply char_eq_eq in Hsrc. subst q.
      apply char_eq_eq in Hd1. subst d1'.
      apply char_eq_eq in Hd2. subst d2'.
      assert (Hnth_q : nth_error query qi = Some c_split).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hnth_d1 : nth_error dict di = Some d1).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail1. reflexivity. }
      assert (Hnth_d2 : nth_error dict (S di) = Some d2).
      { replace (S di) with (di + 1) by lia.
        rewrite <- (nth_error_skipn di dict 1). rewrite Hdtail1. reflexivity. }
      assert (Hquery_tail : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_split Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      assert (Hdict_tail : dtail2 = skipn (S (S di)) dict).
      { change dtail2 with (skipn 2 (d1 :: d2 :: dtail2)).
        rewrite <- Hdtail1.
        rewrite skipn_skipn.
        replace (2 + di) with (S (S di)) by lia.
        reflexivity. }
      rewrite Hquery_tail, Hdict_tail in Hvalid.
      assert (Hreach_split :
        position_reachable_merge_split query n (firstn (S di) dict)
          (special_pos qi (S e))).
      { rewrite (firstn_S_snoc_nth_error dict di d1 Hnth_d1).
        apply reach_ms_enter_split.
        - exact Hreach.
        - simpl in Hcost. unfold split_cost, can_split in Hcost. lia. }
      assert (Hreach' :
        position_reachable_merge_split query n (firstn (S (S di)) dict)
          (std_pos (S qi) (S e))).
      { rewrite (firstn_S_snoc_nth_error dict (S di) d2 Hnth_d2).
        apply reach_ms_complete_split.
        - exact Hreach_split.
        - apply nth_error_Some. rewrite Hnth_q. discriminate. }
      assert (Hcost_tail : ms_seq_cost rest + S e <= n).
      { simpl in Hcost. unfold split_cost, can_split in Hcost. lia. }
      assert (Hqi_tail_bound : S qi <= length query).
      { apply nth_error_Some. rewrite Hnth_q. discriminate. }
      assert (Hdi_tail_bound : S (S di) <= length dict).
      { apply nth_error_Some. rewrite Hnth_d2. discriminate. }
      destruct (IH (S qi) (S (S di)) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound Hreach')
        as [p [Hp [Hterm [Hspec Herr]]]].
      exists p. repeat split; try assumption; simpl; unfold split_cost, can_split; lia.
Qed.

Lemma ms_valid_sequence_reachable_merge_split : forall query dict ops n,
  ms_seq_valid ops query dict ->
  ms_seq_cost ops <= n ->
  exists p,
    position_reachable_merge_split query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= ms_seq_cost ops.
Proof.
  intros query dict ops n Hvalid Hcost.
  unfold ms_seq_valid in Hvalid.
  destruct (ms_valid_sequence_reachable_merge_split_gen
              query dict ops 0 0 n 0 Hvalid ltac:(simpl; lia)
              ltac:(lia) ltac:(lia))
    as [p [Hreach [Hterm [Hspec Herr]]]].
  - simpl. apply reach_ms_initial.
  - exists p. repeat split; try assumption; simpl; lia.
Qed.

Lemma merge_split_distance_reachable_final : forall query dict n,
  merge_split_distance query dict <= n ->
  exists p,
    position_reachable_merge_split query n dict p /\
    term_index p = length query /\
    is_special p = false /\
    num_errors p <= n.
Proof.
  intros query dict n Hdist.
  destruct (ms_seq_exists query dict) as [ops [Hvalid Hcost]].
  destruct (ms_valid_sequence_reachable_merge_split query dict ops n) as
    [p [Hreach [Hterm [Hspec Herr]]]].
  - exact Hvalid.
  - lia.
  - exists p. repeat split; try assumption.
    rewrite Hcost in Herr. lia.
Qed.

(** The same valid merge/split sequence can be read backwards as a completion
    witness for the executable MergeAndSplit automaton. *)
Lemma ms_valid_sequence_can_reach_ms_gen : forall query dict ops qi di n e,
  apply_ms_seq ops (skipn qi query) (skipn di dict) = Some ([], []) ->
  ms_seq_cost ops + e <= n ->
  qi <= length query ->
  di <= length dict ->
  exists p_final,
    can_reach_ms query n (std_pos qi e) (skipn di dict) p_final /\
    term_index p_final = length query /\
    is_special p_final = false /\
    num_errors p_final <= ms_seq_cost ops + e.
Proof.
  intros query dict ops.
  induction ops as [|op rest IH]; intros qi di n e Hvalid Hcost Hqi Hdi.
  - simpl in Hvalid.
    injection Hvalid as Hsrc Htgt.
    assert (Hqi_end : qi = length query).
    { assert (Hlen : length (skipn qi query) = 0) by (rewrite Hsrc; reflexivity).
      rewrite length_skipn in Hlen. lia. }
    assert (Hdi_end : di = length dict).
    { assert (Hlen : length (skipn di dict) = 0) by (rewrite Htgt; reflexivity).
      rewrite length_skipn in Hlen. lia. }
    exists (std_pos qi e).
    rewrite Htgt.
    repeat split; simpl; try lia.
    apply can_reach_ms_done.
  - simpl in Hvalid, Hcost.
    destruct op as [c_del | c_ins | c_src c_tgt | c1 c2 d | c_split d1 d2].
    + destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_del q) eqn:Hchar; [|discriminate].
      apply char_eq_eq in Hchar. subst q.
      assert (Hnth_q : nth_error query qi = Some c_del).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hqtail_eq : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_del Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hqtail_eq in Hvalid.
      assert (Hcost_tail : ms_seq_cost rest + S e <= n) by (simpl in Hcost; lia).
      assert (Hqi_tail_bound : S qi <= length query).
      { apply nth_error_Some. rewrite Hnth_q. discriminate. }
      destruct (IH (S qi) di n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      repeat split; try exact Hterm; try exact Hspec; simpl; try lia.
      apply (can_reach_ms_delete query n (std_pos qi e)
               (skipn di dict) p_final qi e).
      * reflexivity.
      * exact Hqi_tail_bound.
      * lia.
      * exact Hreach.
    + destruct (skipn di dict) as [|d dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_ins d) eqn:Hchar; [|discriminate].
      apply char_eq_eq in Hchar. subst d.
      assert (Hnth_d : nth_error dict di = Some c_ins).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hdtail_eq : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di c_ins Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hdtail_eq in Hvalid.
      assert (Hcost_tail : ms_seq_cost rest + S e <= n) by (simpl in Hcost; lia).
      assert (Hdi_tail_bound : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
      destruct (IH qi (S di) n (S e) Hvalid Hcost_tail Hqi Hdi_tail_bound)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite Hdtail_eq.
      repeat split; try exact Hterm; try exact Hspec; simpl; try lia.
      apply (can_reach_ms_insert query n (std_pos qi e) c_ins
               (skipn (S di) dict) p_final qi e).
      * reflexivity.
      * lia.
      * exact Hreach.
    + destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      destruct (skipn di dict) as [|d dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_src q) eqn:Hsrc; [|discriminate].
      destruct (char_eq c_tgt d) eqn:Htgt; [|discriminate].
      apply char_eq_eq in Hsrc. subst q.
      apply char_eq_eq in Htgt. subst d.
      assert (Hnth_q : nth_error query qi = Some c_src).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hnth_d : nth_error dict di = Some c_tgt).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hqtail_eq : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_src Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      assert (Hdtail_eq : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di c_tgt Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hqtail_eq, Hdtail_eq in Hvalid.
      destruct (char_eq c_src c_tgt) eqn:Hsame.
      * apply char_eq_eq in Hsame. subst c_tgt.
        assert (Hcost_tail : ms_seq_cost rest + e <= n).
        { simpl in Hcost. unfold subst_cost in Hcost. rewrite char_eq_refl in Hcost. lia. }
        assert (Hqi_tail_bound : S qi <= length query).
        { apply nth_error_Some. rewrite Hnth_q. discriminate. }
        assert (Hdi_tail_bound : S di <= length dict).
        { apply nth_error_Some. rewrite Hnth_d. discriminate. }
        destruct (IH (S qi) (S di) n e Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound)
          as [p_final [Hreach [Hterm [Hspec Herr]]]].
        exists p_final.
        rewrite Hdtail_eq.
        split.
        -- apply (can_reach_ms_match query n (std_pos qi e) c_src
                    (skipn (S di) dict) p_final qi e).
           ++ reflexivity.
           ++ apply nth_error_Some. rewrite Hnth_q. discriminate.
           ++ exact Hnth_q.
           ++ exact Hreach.
        -- repeat split; try exact Hterm; try exact Hspec.
           simpl. unfold subst_cost. rewrite char_eq_refl. lia.
      * assert (Hcost_tail : ms_seq_cost rest + S e <= n).
        { simpl in Hcost. unfold subst_cost in Hcost. rewrite Hsame in Hcost. lia. }
        assert (Hqi_tail_bound : S qi <= length query).
        { apply nth_error_Some. rewrite Hnth_q. discriminate. }
        assert (Hdi_tail_bound : S di <= length dict).
        { apply nth_error_Some. rewrite Hnth_d. discriminate. }
        destruct (IH (S qi) (S di) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound)
          as [p_final [Hreach [Hterm [Hspec Herr]]]].
        exists p_final.
        rewrite Hdtail_eq.
        split.
        -- apply (can_reach_ms_substitute query n (std_pos qi e) c_tgt c_src
                    (skipn (S di) dict) p_final qi e).
           ++ reflexivity.
           ++ apply nth_error_Some. rewrite Hnth_q. discriminate.
           ++ exact Hnth_q.
           ++ intro Heq. subst c_tgt. rewrite char_eq_refl in Hsame. discriminate.
           ++ lia.
           ++ exact Hreach.
        -- repeat split; try exact Hterm; try exact Hspec.
           simpl. unfold subst_cost. rewrite Hsame. lia.
    + destruct (skipn qi query) as [|q1 qtail1] eqn:Hqtail1; [discriminate|].
      destruct qtail1 as [|q2 qtail2]; [discriminate|].
      destruct (skipn di dict) as [|d' dtail] eqn:Hdtail; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c1 q1) eqn:Hc1; [|discriminate].
      destruct (char_eq c2 q2) eqn:Hc2; [|discriminate].
      destruct (char_eq d d') eqn:Hd; [|discriminate].
      apply char_eq_eq in Hc1. subst q1.
      apply char_eq_eq in Hc2. subst q2.
      apply char_eq_eq in Hd. subst d'.
      assert (Hnth_q2 : nth_error query (S qi) = Some c2).
      { replace (S qi) with (qi + 1) by lia.
        rewrite <- (nth_error_skipn qi query 1). rewrite Hqtail1. reflexivity. }
      assert (Hnth_d : nth_error dict di = Some d).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail. reflexivity. }
      assert (Hquery_tail : qtail2 = skipn (S (S qi)) query).
      { change qtail2 with (skipn 2 (c1 :: c2 :: qtail2)).
        rewrite <- Hqtail1.
        rewrite skipn_skipn.
        replace (2 + qi) with (S (S qi)) by lia.
        reflexivity. }
      assert (Hdict_tail : dtail = skipn (S di) dict).
      { pose proof (skipn_nth_error_cons dict di d Hnth_d) as Hskip.
        rewrite Hdtail in Hskip. inversion Hskip. reflexivity. }
      rewrite Hquery_tail, Hdict_tail in Hvalid.
      assert (Hcost_tail : ms_seq_cost rest + S e <= n).
      { simpl in Hcost. unfold merge_cost, can_merge in Hcost. lia. }
      assert (Hqi_tail_bound : S (S qi) <= length query).
      { apply nth_error_Some. rewrite Hnth_q2. discriminate. }
      assert (Hdi_tail_bound : S di <= length dict).
      { apply nth_error_Some. rewrite Hnth_d. discriminate. }
      destruct (IH (S (S qi)) (S di) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite Hdict_tail.
      split.
      * apply (can_reach_ms_merge query n (std_pos qi e) d
                 (skipn (S di) dict) p_final qi e).
        -- reflexivity.
        -- lia.
        -- lia.
        -- exact Hreach.
      * repeat split; try exact Hterm; try exact Hspec.
        simpl. unfold merge_cost, can_merge. lia.
    + destruct (skipn qi query) as [|q qtail] eqn:Hqtail; [discriminate|].
      destruct (skipn di dict) as [|d1' dtail1] eqn:Hdtail1; [discriminate|].
      destruct dtail1 as [|d2' dtail2]; [discriminate|].
      simpl in Hvalid.
      destruct (char_eq c_split q) eqn:Hsrc; [|discriminate].
      destruct (char_eq d1 d1') eqn:Hd1; [|discriminate].
      destruct (char_eq d2 d2') eqn:Hd2; [|discriminate].
      apply char_eq_eq in Hsrc. subst q.
      apply char_eq_eq in Hd1. subst d1'.
      apply char_eq_eq in Hd2. subst d2'.
      assert (Hnth_q : nth_error query qi = Some c_split).
      { replace qi with (qi + 0) by lia.
        rewrite <- (nth_error_skipn qi query 0). rewrite Hqtail. reflexivity. }
      assert (Hnth_d1 : nth_error dict di = Some d1).
      { replace di with (di + 0) by lia.
        rewrite <- (nth_error_skipn di dict 0). rewrite Hdtail1. reflexivity. }
      assert (Hnth_d2 : nth_error dict (S di) = Some d2).
      { replace (S di) with (di + 1) by lia.
        rewrite <- (nth_error_skipn di dict 1). rewrite Hdtail1. reflexivity. }
      assert (Hquery_tail : qtail = skipn (S qi) query).
      { pose proof (skipn_nth_error_cons query qi c_split Hnth_q) as Hskip.
        rewrite Hqtail in Hskip. inversion Hskip. reflexivity. }
      assert (Hdict_tail : dtail2 = skipn (S (S di)) dict).
      { change dtail2 with (skipn 2 (d1 :: d2 :: dtail2)).
        rewrite <- Hdtail1.
        rewrite skipn_skipn.
        replace (2 + di) with (S (S di)) by lia.
        reflexivity. }
      rewrite Hquery_tail, Hdict_tail in Hvalid.
      assert (Hcost_tail : ms_seq_cost rest + S e <= n).
      { simpl in Hcost. unfold split_cost, can_split in Hcost. lia. }
      assert (Hqi_tail_bound : S qi <= length query).
      { apply nth_error_Some. rewrite Hnth_q. discriminate. }
      assert (Hdi_tail_bound : S (S di) <= length dict).
      { apply nth_error_Some. rewrite Hnth_d2. discriminate. }
      destruct (IH (S qi) (S (S di)) n (S e) Hvalid Hcost_tail Hqi_tail_bound Hdi_tail_bound)
        as [p_final [Hreach [Hterm [Hspec Herr]]]].
      exists p_final.
      rewrite <- Hdict_tail in Hreach.
      split.
      * apply (can_reach_ms_enter_split query n (std_pos qi e) d1
                 (d2 :: dtail2) p_final qi e).
        -- reflexivity.
        -- lia.
        -- apply (can_reach_ms_complete_split query n (special_pos qi (S e)) d2
                   dtail2 p_final qi (S e)).
           ++ reflexivity.
           ++ apply nth_error_Some. rewrite Hnth_q. discriminate.
           ++ exact Hreach.
      * repeat split; try exact Hterm; try exact Hspec.
        simpl. unfold split_cost, can_split. lia.
Qed.

Lemma ms_valid_sequence_can_complete_initial : forall query dict n ops,
  ms_seq_valid ops query dict ->
  ms_seq_cost ops <= n ->
  can_complete_to_final_ms query n dict initial_position.
Proof.
  intros query dict n ops Hvalid Hcost.
  unfold ms_seq_valid in Hvalid.
  destruct (ms_valid_sequence_can_reach_ms_gen query dict ops 0 0 n 0
              Hvalid ltac:(simpl; lia) ltac:(lia) ltac:(lia))
    as [p_final [Hreach [Hterm [Hspec Herr]]]].
  unfold can_complete_to_final_ms.
  exists p_final.
  unfold initial_position.
  simpl in Hreach.
  repeat split; try exact Hterm; try exact Hspec; try lia.
  exact Hreach.
Qed.

Lemma merge_split_bound_initial_closed_has_ms_completable : forall query dict n,
  merge_split_distance query dict <= n ->
  state_has_ms_completable query n dict
    (mkState (epsilon_closure [initial_position] n (length query))
             MergeAndSplit (length query)).
Proof.
  intros query dict n Hdist.
  destruct (ms_seq_exists query dict) as [ops [Hvalid Hcost]].
  exists initial_position.
  split.
  - simpl. apply epsilon_closure_includes_input. simpl. left. reflexivity.
  - apply ms_valid_sequence_can_complete_initial with (ops := ops).
    + exact Hvalid.
    + lia.
Qed.

Lemma can_reach_ms_errors_monotone : forall query n p remaining p_final,
  can_reach_ms query n p remaining p_final ->
  num_errors p <= num_errors p_final.
Proof.
  intros query n p remaining p_final Hreach.
  induction Hreach; subst; simpl in *; lia.
Qed.

Lemma can_reach_ms_term_index_monotone : forall query n p remaining p_final,
  can_reach_ms query n p remaining p_final ->
  term_index p <= term_index p_final.
Proof.
  intros query n p remaining p_final Hreach.
  induction Hreach; subst; simpl in *; lia.
Qed.

Lemma can_reach_ms_empty_source_not_special : forall query n p p_final,
  can_reach_ms query n p [] p_final ->
  is_special p_final = false ->
  is_special p = false.
Proof.
  intros query n p p_final Hreach Hfinal_spec.
  remember ([] : list Char) as remaining eqn:Hremaining.
  revert Hremaining.
  induction Hreach; intros Hremaining; subst; simpl in *; try discriminate; try reflexivity.
  exact Hfinal_spec.
Qed.

Lemma can_reach_ms_empty_remaining_errors : forall query n p p_final,
  can_reach_ms query n p [] p_final ->
  is_special p = false ->
  num_errors p_final = num_errors p + (term_index p_final - term_index p).
Proof.
  intros query n p p_final Hreach Hspec.
  remember ([] : list Char) as remaining eqn:Hremaining.
  revert Hremaining Hspec.
  induction Hreach; intros Hremaining Hspec0; subst; simpl in *.
  - lia.
  - specialize (IHHreach eq_refl eq_refl).
    pose proof (can_reach_ms_term_index_monotone _ _ _ _ _ Hreach) as Hmono.
    simpl in *. lia.
  - discriminate.
  - discriminate.
  - discriminate.
  - discriminate.
  - discriminate.
  - discriminate.
Qed.

Lemma can_reach_ms_lower_errors_aux : forall query n p remaining p_final diff,
  can_reach_ms query n p remaining p_final ->
  num_errors p_final <= n ->
  diff <= num_errors p ->
  exists p_final',
    can_reach_ms query n
      (if is_special p
       then special_pos (term_index p) (num_errors p - diff)
       else std_pos (term_index p) (num_errors p - diff))
      remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' = num_errors p_final - diff /\
    is_special p_final' = is_special p_final.
Proof.
  intros query n p remaining p_final diff Hreach.
  induction Hreach; intros Hfinal_err Hdiff; subst; simpl in *.
  - destruct p as [i e sp]; simpl in *.
    destruct sp.
    + exists (special_pos i (e - diff)).
      repeat split; simpl; try lia.
      apply can_reach_ms_done.
    + exists (std_pos i (e - diff)).
      repeat split; simpl; try lia.
      apply can_reach_ms_done.
  - specialize (IHHreach Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    replace (match diff with 0 => S e | S l => e - l end)
      with (S (e - diff)) in Hreach' by (destruct diff; simpl; lia).
    apply (can_reach_ms_delete query n (std_pos i (e - diff))
             remaining p_final' i (e - diff)); try reflexivity; try lia.
    exact Hreach'.
  - specialize (IHHreach Hfinal_err Hdiff).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    apply (can_reach_ms_match query n (std_pos i (e - diff)) c
             remaining p_final' i (e - diff)); assumption || reflexivity.
  - specialize (IHHreach Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    replace (match diff with 0 => S e | S l => e - l end)
      with (S (e - diff)) in Hreach' by (destruct diff; simpl; lia).
    apply (can_reach_ms_substitute query n (std_pos i (e - diff)) c c'
             remaining p_final' i (e - diff)); try assumption; try reflexivity; try lia.
  - specialize (IHHreach Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    replace (match diff with 0 => S e | S l => e - l end)
      with (S (e - diff)) in Hreach' by (destruct diff; simpl; lia).
    apply (can_reach_ms_insert query n (std_pos i (e - diff)) c
             remaining p_final' i (e - diff)); try reflexivity; try lia.
    exact Hreach'.
  - specialize (IHHreach Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    replace (match diff with 0 => S e | S l => e - l end)
      with (S (e - diff)) in Hreach' by (destruct diff; simpl; lia).
    apply (can_reach_ms_merge query n (std_pos i (e - diff)) c
             remaining p_final' i (e - diff)); try assumption; try reflexivity; try lia.
  - specialize (IHHreach Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    replace (match diff with 0 => S e | S l => e - l end)
      with (S (e - diff)) in Hreach' by (destruct diff; simpl; lia).
    apply (can_reach_ms_enter_split query n (std_pos i (e - diff)) c
             remaining p_final' i (e - diff)); try reflexivity; try lia.
    exact Hreach'.
  - specialize (IHHreach Hfinal_err Hdiff).
    destruct IHHreach as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
    exists p_final'. repeat split; try assumption; simpl; try lia.
    apply (can_reach_ms_complete_split query n (special_pos i (e - diff)) c
             remaining p_final' i (e - diff)); try assumption; try reflexivity.
Qed.

Lemma can_reach_ms_lower_errors : forall query n p remaining p_final e',
  can_reach_ms query n p remaining p_final ->
  num_errors p_final <= n ->
  e' <= num_errors p ->
  exists p_final',
    can_reach_ms query n
      (if is_special p then special_pos (term_index p) e'
       else std_pos (term_index p) e') remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' = num_errors p_final - (num_errors p - e') /\
    is_special p_final' = is_special p_final.
Proof.
  intros query n p remaining p_final e' Hreach Hfinal_err He'.
  pose proof (can_reach_ms_lower_errors_aux query n p remaining p_final
                (num_errors p - e') Hreach Hfinal_err ltac:(lia))
    as [p_final' [Hreach' [Hterm [Herr Hspec]]]].
  exists p_final'. repeat split; try assumption.
  destruct p as [i ep sp]; simpl in *.
  destruct sp; replace (ep - (ep - e')) with e' in Hreach' by lia;
    exact Hreach'.
Qed.

Lemma can_reach_ms_prepend_deletes : forall query n i e k remaining p_final,
  i + k <= length query ->
  e + k <= n ->
  can_reach_ms query n (std_pos (i + k) (e + k)) remaining p_final ->
  can_reach_ms query n (std_pos i e) remaining p_final.
Proof.
  intros query n i e k.
  revert i e.
  induction k as [|k IH]; intros i e remaining p_final Hterm Herr Hreach.
  - replace (i + 0) with i in Hreach by lia.
    replace (e + 0) with e in Hreach by lia.
    exact Hreach.
  - apply (can_reach_ms_delete query n (std_pos i e) remaining p_final i e).
    + reflexivity.
    + lia.
    + lia.
    + replace (S i) with (i + 1) by lia.
      replace (S e) with (e + 1) by lia.
      replace (i + S k) with ((i + 1) + k) in Hreach by lia.
      replace (e + S k) with ((e + 1) + k) in Hreach by lia.
      apply (IH (i + 1) (e + 1)); try lia.
      exact Hreach.
Qed.

Lemma can_complete_ms_same_index_lower_errors : forall query n remaining p p',
  can_complete_to_final_ms query n remaining p ->
  term_index p' = term_index p ->
  is_special p' = is_special p ->
  num_errors p' <= num_errors p ->
  can_complete_to_final_ms query n remaining p'.
Proof.
  intros query n remaining p p'
         [p_final [Hreach [Hterm [Herr Hspec]]]]
         Hidx Hspecial Herr_le.
  destruct (can_reach_ms_lower_errors query n p remaining p_final
              (num_errors p') Hreach Herr Herr_le)
    as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
  exists p_final'. repeat split; try lia.
  - destruct p as [i e sp], p' as [i' e' sp']; simpl in *.
    subst i' sp'.
    destruct sp; simpl in *; exact Hreach'.
  - rewrite Hspec'. exact Hspec.
Qed.

Lemma can_complete_ms_prepend_deletes : forall query n remaining i e k,
  i + k <= length query ->
  e + k <= n ->
  can_complete_to_final_ms query n remaining (std_pos (i + k) (e + k)) ->
  can_complete_to_final_ms query n remaining (std_pos i e).
Proof.
  intros query n remaining i e k Hterm Herr
         [p_final [Hreach [Hfinal_term [Hfinal_err Hfinal_spec]]]].
  exists p_final. repeat split; try assumption.
  apply can_reach_ms_prepend_deletes with (k := k); assumption.
Qed.

Lemma can_complete_ms_behind_std : forall query n remaining i e j f,
  i <= j ->
  e + (j - i) <= f ->
  j <= length query ->
  can_complete_to_final_ms query n remaining (std_pos j f) ->
  can_complete_to_final_ms query n remaining (std_pos i e).
Proof.
  intros query n remaining i e j f Hidx Herr Hbound Hcomplete.
  pose (k := j - i).
  assert (Hsame :
    can_complete_to_final_ms query n remaining (std_pos (i + k) (e + k))).
  { replace (i + k) with j by (unfold k; lia).
    apply can_complete_ms_same_index_lower_errors with (p := std_pos j f).
    - exact Hcomplete.
    - reflexivity.
    - reflexivity.
    - simpl. unfold k. exact Herr. }
  apply can_complete_ms_prepend_deletes with (k := k).
  - unfold k. lia.
  - destruct Hcomplete as [p_final [Hreach [_ [Hfinal_err _]]]].
    pose proof (can_reach_ms_errors_monotone query n (std_pos j f)
                  remaining p_final Hreach) as Hmono.
    simpl in Hmono. unfold k in *. lia.
  - exact Hsame.
Qed.

Lemma can_complete_ms_special_inv : forall query n remaining i e,
  can_complete_to_final_ms query n remaining (special_pos i e) ->
  exists c rest,
    remaining = c :: rest /\
    i < length query /\
    can_complete_to_final_ms query n rest (std_pos (S i) e).
Proof.
  intros query n remaining i e
         [p_final [Hreach [Hterm [Herr Hspec]]]].
  destruct remaining as [|c rest].
  - dependent destruction Hreach; simpl in Hspec; discriminate.
  - dependent destruction Hreach; try discriminate.
    exists c, rest. split; [reflexivity|].
    split; [assumption|].
    unfold can_complete_to_final_ms.
    exists p_final. repeat split; assumption.
Qed.

Local Lemma can_reach_ms_higher_index : forall
  (query : list Char) (n : nat) p remaining p_final i' e',
  can_reach_ms query n p remaining p_final ->
  term_index p_final = length query ->
  is_special p_final = false ->
  num_errors p_final <= n ->
  i' > term_index p ->
  i' <= length query ->
  e' <= num_errors p ->
  i' - term_index p <= num_errors p - e' ->
  exists p_final',
    can_reach_ms query n (std_pos i' e') remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' <= n /\
    is_special p_final' = false.
Proof.
  intros query n p remaining p_final i' e' Hreach.
  revert i' e'.
  induction Hreach; intros i' e' Hterm_final Hfinal_spec Hfinal_err
                          Hi'_gt Hi'_qlen He'_le Hdiff; subst; simpl in *.
  - lia.
  - destruct (Nat.eq_dec i' (S i)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      assert (He'_tail : e' <= S e) by lia.
      destruct (can_reach_ms_lower_errors query n (std_pos (S i) (S e))
                  remaining p_final e' Hreach Hfinal_err He'_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hreach'; try exact Hterm';
        try lia.
      rewrite Hspec'. exact Hfinal_spec.
    + assert (Hi'_gt_tail : i' > S i) by lia.
      assert (He'_le_tail : e' <= S e) by lia.
      assert (Hdiff_tail : i' - S i <= S e - e') by lia.
      destruct (IHHreach i' e' Hterm_final Hfinal_spec Hfinal_err
                  Hi'_gt_tail Hi'_qlen He'_le_tail Hdiff_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; assumption.
  - assert (He'_lt : e' < n).
    { pose proof (can_reach_ms_errors_monotone query n (std_pos (S i) e)
                    remaining p_final Hreach) as Hmono.
      simpl in Hmono. lia. }
    destruct (Nat.eq_dec i' (S i)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      assert (Hinsert_tail_err : S e' <= e) by lia.
      destruct (can_reach_ms_lower_errors query n (std_pos (S i) e)
                  remaining p_final (S e') Hreach Hfinal_err
                  Hinsert_tail_err)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try lia.
      * apply (can_reach_ms_insert query n (std_pos (S i) e') c
                  remaining p_final' (S i) e'); try reflexivity; try lia.
        exact Hreach'.
      * rewrite Hspec'. exact Hfinal_spec.
    + assert (Hi'_gt_tail : i' > S i) by lia.
      assert (Hinsert_tail_err : S e' <= e) by lia.
      assert (Hdiff_tail : i' - S i <= e - S e') by lia.
      destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                  Hi'_gt_tail Hi'_qlen Hinsert_tail_err Hdiff_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr';
        try exact Hspec'.
      apply (can_reach_ms_insert query n (std_pos i' e') c
               remaining p_final' i' e'); try reflexivity; try lia.
      exact Hreach'.
  - assert (He'_lt : e' < n) by lia.
    destruct (Nat.eq_dec i' (S i)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      assert (Hsubst_tail_err : S e' <= S e) by lia.
      destruct (can_reach_ms_lower_errors query n (std_pos (S i) (S e))
                  remaining p_final (S e') Hreach Hfinal_err
                  Hsubst_tail_err)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try lia.
      * apply (can_reach_ms_insert query n (std_pos (S i) e') c
                  remaining p_final' (S i) e'); try reflexivity; try lia.
        exact Hreach'.
      * rewrite Hspec'. exact Hfinal_spec.
    + assert (Hi'_gt_tail : i' > S i) by lia.
      assert (Hsubst_tail_err : S e' <= S e) by lia.
      assert (Hdiff_tail : i' - S i <= S e - S e') by lia.
      destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                  Hi'_gt_tail Hi'_qlen Hsubst_tail_err Hdiff_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr';
        try exact Hspec'.
      apply (can_reach_ms_insert query n (std_pos i' e') c
               remaining p_final' i' e'); try reflexivity; try lia.
      exact Hreach'.
  - assert (He'_lt : e' < n) by lia.
    assert (Hinsert_tail_err : S e' <= S e) by lia.
    assert (Hdiff_tail : i' - i <= S e - S e') by lia.
    destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                Hi'_gt Hi'_qlen Hinsert_tail_err Hdiff_tail)
      as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split; try exact Hterm'; try exact Herr';
      try exact Hspec'.
    apply (can_reach_ms_insert query n (std_pos i' e') c
             remaining p_final' i' e'); try reflexivity; try lia.
    exact Hreach'.
  - assert (He'_lt : e' < n) by lia.
    destruct (Nat.le_gt_cases i' (S (S i))) as [Hi'_le_tail | Hi'_gt_tail].
    + pose (k := S (S i) - i').
      assert (Hidx : i' + k = S (S i)) by (unfold k; lia).
      assert (Herr_target : S e' + k <= S e) by (unfold k; lia).
      assert (Hlower :
        exists p_final',
          can_reach_ms query n (std_pos (S (S i)) (S e' + k))
            remaining p_final' /\
          term_index p_final' = term_index p_final /\
          num_errors p_final' = num_errors p_final - (S e - (S e' + k)) /\
          is_special p_final' = is_special p_final).
      { apply (can_reach_ms_lower_errors query n (std_pos (S (S i)) (S e))
                  remaining p_final (S e' + k) Hreach Hfinal_err).
        exact Herr_target. }
      destruct Hlower as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      assert (Hprep :
        can_reach_ms query n (std_pos i' (S e')) remaining p_final').
      { apply can_reach_ms_prepend_deletes with (k := k).
        - rewrite Hidx. lia.
        - lia.
        - simpl in Hreach'.
          replace (S (S i)) with (i' + k) in Hreach' by lia.
          exact Hreach'. }
      exists p_final'. repeat split; try exact Hterm'; try lia.
      * apply (can_reach_ms_insert query n (std_pos i' e') c
                  remaining p_final' i' e'); try reflexivity; try lia.
        exact Hprep.
      * rewrite Hspec'. exact Hfinal_spec.
    + assert (Hmerge_tail_err : S e' <= S e) by lia.
      assert (Hdiff_tail : i' - S (S i) <= S e - S e') by lia.
      destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                  Hi'_gt_tail Hi'_qlen Hmerge_tail_err Hdiff_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr';
        try exact Hspec'.
      apply (can_reach_ms_insert query n (std_pos i' e') c
               remaining p_final' i' e'); try reflexivity; try lia.
      exact Hreach'.
  - assert (He'_lt : e' < n) by lia.
    assert (Hinsert_tail_err : S e' <= S e) by lia.
    assert (Hdiff_tail : i' - i <= S e - S e') by lia.
    destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                Hi'_gt Hi'_qlen Hinsert_tail_err Hdiff_tail)
      as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split; try exact Hterm'; try exact Herr';
      try exact Hspec'.
    apply (can_reach_ms_insert query n (std_pos i' e') c
             remaining p_final' i' e'); try reflexivity; try lia.
    exact Hreach'.
  - assert (He'_lt : e' < n).
    { pose proof (can_reach_ms_errors_monotone query n (std_pos (S i) e)
                    remaining p_final Hreach) as Hmono.
      simpl in Hmono. lia. }
    destruct (Nat.eq_dec i' (S i)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      assert (Hsplit_tail_err : S e' <= e) by lia.
      destruct (can_reach_ms_lower_errors query n (std_pos (S i) e)
                  remaining p_final (S e') Hreach Hfinal_err
                  Hsplit_tail_err)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try lia.
      * apply (can_reach_ms_insert query n (std_pos (S i) e') c
                  remaining p_final' (S i) e'); try reflexivity; try lia.
        exact Hreach'.
      * rewrite Hspec'. exact Hfinal_spec.
    + assert (Hi'_gt_tail : i' > S i) by lia.
      assert (Hsplit_tail_err : S e' <= e) by lia.
      assert (Hdiff_tail : i' - S i <= e - S e') by lia.
      destruct (IHHreach i' (S e') Hterm_final Hfinal_spec Hfinal_err
                  Hi'_gt_tail Hi'_qlen Hsplit_tail_err Hdiff_tail)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr';
        try exact Hspec'.
      apply (can_reach_ms_insert query n (std_pos i' e') c
               remaining p_final' i' e'); try reflexivity; try lia.
      exact Hreach'.
Qed.

Local Lemma can_reach_ms_from_ahead : forall
  query n i e remaining p_final i' e',
  can_reach_ms query n (std_pos i e) remaining p_final ->
  i' >= i ->
  i' - i <= e - e' ->
  e' <= e ->
  i' <= length query ->
  num_errors p_final <= n ->
  term_index p_final = length query ->
  is_special p_final = false ->
  exists p_final',
    can_reach_ms query n (std_pos i' e') remaining p_final' /\
    term_index p_final' = length query /\
    num_errors p_final' <= n /\
    is_special p_final' = false.
Proof.
  intros query n i e remaining p_final i' e' Hreach Hi'_ge Hdiff
         He'_le Hi'_qlen Hfinal_err Hterm_final Hfinal_spec.
  destruct (Nat.eq_dec i' i) as [Hi'_eq | Hi'_neq].
  - subst i'.
    destruct (can_reach_ms_lower_errors query n (std_pos i e)
                remaining p_final e' Hreach Hfinal_err He'_le)
      as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. split; [exact Hreach'|].
    split.
    + lia.
    + split.
      * rewrite Herr'. lia.
      * rewrite Hspec'. exact Hfinal_spec.
  - assert (Hi'_gt : i' > i) by lia.
    destruct (can_reach_ms_higher_index query n (std_pos i e)
                remaining p_final i' e' Hreach Hterm_final Hfinal_spec
                Hfinal_err Hi'_gt Hi'_qlen He'_le Hdiff)
      as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split; try exact Hreach'; try exact Herr';
      try exact Hspec'.
    lia.
Qed.

Local Lemma subsumption_preserves_can_complete_ms : forall
  query n remaining p p',
  can_complete_to_final_ms query n remaining p ->
  subsumes_merge_split (length query) p' p = true ->
  is_special p = false ->
  is_special p' = false ->
  term_index p' <= length query ->
  can_complete_to_final_ms query n remaining p'.
Proof.
  intros query n remaining p p' Hcomplete Hsub Hspec Hspec' Hp'_bound.
  destruct p as [i e sp], p' as [i' e' sp']; simpl in *.
  subst sp sp'.
  unfold subsumes_merge_split in Hsub.
  unfold position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (length query <? i') eqn:Hp'_past.
  { apply Nat.ltb_lt in Hp'_past. lia. }
  destruct ((negb (length query <=? i')) && (length query <=? i))
    eqn:Hfinal_check; [discriminate|].
  destruct (negb (e' <? e)) eqn:Herr_check; [discriminate|].
  apply Bool.negb_false_iff in Herr_check.
  apply Nat.ltb_lt in Herr_check.
  apply Nat.eqb_eq in Hsub. subst i'.
  eapply can_complete_ms_same_index_lower_errors.
  - exact Hcomplete.
  - reflexivity.
  - reflexivity.
  - simpl. lia.
Qed.

Local Lemma subsumes_merge_split_representative_bound : forall qlen p p',
  subsumes_merge_split qlen p' p = true ->
  term_index p' <= qlen.
Proof.
  intros qlen p p' Hsub.
  unfold subsumes_merge_split in Hsub.
  destruct (qlen <? term_index p') eqn:Hpast.
  - discriminate.
  - apply Nat.ltb_ge. exact Hpast.
Qed.

Local Lemma subsumption_preserves_can_complete_ms_special : forall
  query n remaining i e p',
  can_complete_to_final_ms query n remaining (special_pos i e) ->
  subsumes_merge_split (length query) p' (special_pos i e) = true ->
  is_special p' = true ->
  term_index p' <= length query ->
  can_complete_to_final_ms query n remaining p'.
Proof.
  intros query n remaining i e p' Hcomplete Hsub Hspec' Hp'_bound.
  destruct p' as [i' e' sp']; simpl in Hspec'. subst sp'.
  unfold subsumes_merge_split in Hsub.
  unfold position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (length query <? i') eqn:Hp'_past.
  { simpl in Hsub. discriminate Hsub. }
  simpl in Hsub.
  destruct ((negb (length query <=? i')) && (length query <=? i))
    eqn:Hfinal_check.
  { simpl in Hsub. discriminate Hsub. }
  simpl in Hsub.
  destruct ((length query <=? i') && negb (length query <=? i))
    eqn:Hspecial_final_check.
  { simpl in Hsub. discriminate Hsub. }
  simpl in Hsub.
  destruct (negb (e' <? e)) eqn:Herr_check.
  { simpl in Hsub. discriminate Hsub. }
  simpl in Hsub.
  apply Bool.negb_false_iff in Herr_check.
  apply Nat.ltb_lt in Herr_check.
  apply Nat.eqb_eq in Hsub. subst i'.
  eapply can_complete_ms_same_index_lower_errors.
  - exact Hcomplete.
  - reflexivity.
  - reflexivity.
  - simpl. lia.
Qed.

Local Lemma subsumption_preserves_can_complete_ms_any : forall
  query n remaining p p',
  can_complete_to_final_ms query n remaining p ->
  subsumes_merge_split (length query) p' p = true ->
  term_index p' <= length query ->
  can_complete_to_final_ms query n remaining p'.
Proof.
  intros query n remaining p p' Hcomplete Hsub Hp'_bound.
  destruct p as [i e sp], p' as [i' e' sp']; simpl in *.
  destruct sp.
  - destruct sp' eqn:Hsp'.
    + eapply subsumption_preserves_can_complete_ms_special.
      * exact Hcomplete.
      * simpl. exact Hsub.
      * reflexivity.
      * simpl. exact Hp'_bound.
    + unfold subsumes_merge_split in Hsub.
      unfold position_is_final_for_subsumption in Hsub.
	      simpl in Hsub.
	      destruct (length query <? i') eqn:Hp'_past; [discriminate|].
	      destruct ((negb (length query <=? i')) && (length query <=? i));
	        simpl in Hsub; [discriminate|].
	      destruct ((length query <=? i') && negb (length query <=? i));
	        simpl in Hsub; discriminate.
  - destruct sp' eqn:Hsp'.
    + unfold subsumes_merge_split in Hsub.
      unfold position_is_final_for_subsumption in Hsub.
	      simpl in Hsub.
	      destruct (length query <? i') eqn:Hp'_past; [discriminate|].
	      destruct ((negb (length query <=? i')) && (length query <=? i));
	        simpl in Hsub; [discriminate|].
	      destruct ((length query <=? i') && negb (length query <=? i));
	        simpl in Hsub; discriminate.
    + eapply subsumption_preserves_can_complete_ms.
      * exact Hcomplete.
      * simpl. exact Hsub.
      * reflexivity.
      * reflexivity.
      * simpl. exact Hp'_bound.
Qed.

Lemma positions_cover_merge_split_has_ms_completable : forall
  query n remaining ps p,
  (forall q, In q ps -> term_index q <= length query) ->
  positions_cover_merge_split (length query) ps p ->
  can_complete_to_final_ms query n remaining p ->
  exists q, In q ps /\ can_complete_to_final_ms query n remaining q.
Proof.
  intros query n remaining ps p Hbound Hcover.
  induction Hcover as [p Hin | p q Hcover IH Hsub]; intros Hcomplete.
  - exists p. split; assumption.
  - assert (Hq_bound : term_index q <= length query).
    { eapply subsumes_merge_split_representative_bound. exact Hsub. }
    apply IH.
    eapply subsumption_preserves_can_complete_ms_any.
    + exact Hcomplete.
    + exact Hsub.
    + exact Hq_bound.
Qed.

Lemma reachable_merge_split_term_index_upper_bound : forall query n dict_prefix p,
  position_reachable_merge_split query n dict_prefix p ->
  term_index p <= length dict_prefix + num_errors p.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach; simpl in *; try rewrite length_app in *; simpl in *; lia.
Qed.

Lemma reachable_merge_split_term_index_lower_bound : forall query n dict_prefix p,
  position_reachable_merge_split query n dict_prefix p ->
  length dict_prefix <= term_index p + num_errors p.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach; simpl in *; try rewrite length_app in *; simpl in *; lia.
Qed.

Lemma state_positions_spread_bound_merge_split : forall query n dict_prefix positions,
  (forall p, In p positions -> position_reachable_merge_split query n dict_prefix p) ->
  (forall p, In p positions -> num_errors p <= n) ->
  forall p1 p2,
    In p1 positions -> In p2 positions ->
    term_index p2 <= term_index p1 + 2 * n.
Proof.
  intros query n dict_prefix positions Hreach Herr p1 p2 Hin1 Hin2.
  assert (Hupper : term_index p2 <= length dict_prefix + num_errors p2).
  { eapply reachable_merge_split_term_index_upper_bound.
    apply Hreach. exact Hin2. }
  assert (Hlower : length dict_prefix <= term_index p1 + num_errors p1).
  { eapply reachable_merge_split_term_index_lower_bound.
    apply Hreach. exact Hin1. }
  assert (He1 : num_errors p1 <= n) by (apply Herr; exact Hin1).
  assert (He2 : num_errors p2 <= n) by (apply Herr; exact Hin2).
  lia.
Qed.

Lemma term_index_minus_min_bounded_merge_split : forall query n dict_prefix positions init p,
  (forall p0, In p0 positions -> position_reachable_merge_split query n dict_prefix p0) ->
  term_index p < init ->
  In p positions ->
  positions <> [] ->
  term_index p - fold_left Nat.min (map term_index positions) init <= 2 * n.
Proof.
  intros query n dict_prefix positions init p Hreach Hlt_init Hin Hne.
  set (min_i := fold_left Nat.min (map term_index positions) init).
  assert (Hmin_le_p : min_i <= term_index p).
  { unfold min_i. apply min_i_le_term_index. exact Hin. }
  assert (Hmin_lt_init : min_i < init).
  { apply Nat.le_lt_trans with (term_index p); assumption. }
  destruct (list_has_min_term_index positions Hne) as [p_min [Hin_min Hmin_prop]].
  assert (Hmin_le_pmin : min_i <= term_index p_min).
  { unfold min_i. apply min_i_le_term_index. exact Hin_min. }
  assert (Hpmin_le_min : term_index p_min <= min_i).
  { assert (Hin_min_i : In min_i (map term_index positions)).
    { unfold min_i. apply fold_left_min_in_list. exact Hmin_lt_init. }
    apply in_map_iff in Hin_min_i.
    destruct Hin_min_i as [q [Heq_q Hin_q]].
    specialize (Hmin_prop q Hin_q).
    lia. }
  assert (Hmin_eq : min_i = term_index p_min) by lia.
  rewrite Hmin_eq.
  assert (Hspread : term_index p <= term_index p_min + 2 * n).
  { apply state_positions_spread_bound_merge_split with
      (query := query) (dict_prefix := dict_prefix) (positions := positions).
    - exact Hreach.
    - intros p0 Hin0.
      eapply reachable_merge_split_implies_edit_distance.
      apply Hreach. exact Hin0.
    - exact Hin_min.
    - exact Hin. }
  assert (Hp_ge_pmin : term_index p_min <= term_index p).
  { apply Hmin_prop. exact Hin. }
  lia.
Qed.

Lemma transition_state_merge_split_closed_insert_exact : forall
  s c query n i e,
  In (std_pos i e) (positions s) ->
  e < n ->
  In (std_pos i (S e))
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin He_lt.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_insert.
  exact He_lt.
Qed.

Lemma transition_state_merge_split_closed_match_exact : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  In (std_pos (S i) e)
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hin Hlt Hnth.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded_merge_split with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = true).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_match.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    unfold min_i. apply min_i_le_term_index. exact Hin.
  - exact Hcv.
Qed.

Lemma transition_state_merge_split_closed_substitute_exact : forall
  query n dict s c c' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  In (std_pos (S i) (S e))
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c c' i e Hqlen Hall_reach Hin Hlt Hnth Hneq He_lt.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded_merge_split with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = false).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    apply char_matches_at_false_iff.
    intros [q [Hnth_q Heq]].
    rewrite Hnth in Hnth_q.
    injection Hnth_q as Hq. subst q.
    apply Hneq. exact Heq. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_substitute.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    unfold min_i. apply min_i_le_term_index. exact Hin.
  - exact Hcv.
  - exact He_lt.
Qed.

Lemma transition_state_transposition_closed_insert_exact : forall
  s c query n i e,
  In (std_pos i e) (positions s) ->
  e < n ->
  In (std_pos i (S e))
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin He_lt.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_transposition.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_insert.
  exact He_lt.
Qed.

Lemma transition_state_transposition_closed_match_exact : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  In (std_pos (S i) e)
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hin Hlt Hnth.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n + 5).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded_damerau with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = true).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_transposition.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_match.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    unfold min_i. apply min_i_le_term_index. exact Hin.
  - exact Hcv.
Qed.

Lemma transition_state_transposition_closed_substitute_exact : forall
  query n dict s c c' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  In (std_pos (S i) (S e))
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c c' i e Hqlen Hall_reach Hin Hlt Hnth Hneq He_lt.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n + 5).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply term_index_minus_min_bounded_damerau with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - rewrite Hqlen. exact Hlt.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    lia. }
  assert (Hcv :
    cv_at
      (characteristic_vector c query
         (fold_left Nat.min (map term_index (positions s)) (query_length s))
         (2 * n + 6))
      (i - fold_left Nat.min (map term_index (positions s)) (query_length s)) = false).
  { fold min_i.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    assert (Hmin_le : min_i <= i).
    { unfold min_i.
      change i with (term_index (std_pos i e)).
      apply min_i_le_term_index. exact Hin. }
    assert (Hsum : min_i + (i - min_i) = i) by lia.
    rewrite Hsum.
    apply char_matches_at_false_iff.
    intros [q [Hnth_q Heq]].
    rewrite Hnth in Hnth_q.
    injection Hnth_q as Hq. subst q.
    apply Hneq. exact Heq. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_transposition.
  simpl.
  apply in_or_app. left.
  apply transition_standard_produces_substitute.
  - rewrite Hqlen. exact Hlt.
  - change i with (term_index (std_pos i e)).
    unfold min_i. apply min_i_le_term_index. exact Hin.
  - exact Hcv.
  - exact He_lt.
Qed.

Lemma term_index_minus_min_bounded_damerau_enter : forall
  query n dict_prefix positions init i e,
  (forall p0, In p0 positions ->
              position_reachable_damerau query n dict_prefix p0) ->
  In (std_pos i e) positions ->
  positions <> [] ->
  i < init ->
  e < n ->
  i - fold_left Nat.min (map term_index positions) init <= 2 * n + 4.
Proof.
  intros query n dict_prefix positions init i e Hreach Hin Hne Hlt_init He_lt.
  set (min_i := fold_left Nat.min (map term_index positions) init).
  assert (Hmin_le_p : min_i <= i).
  { unfold min_i.
    change i with (term_index (std_pos i e)).
    apply min_i_le_term_index. exact Hin. }
  assert (Hmin_lt_init : min_i < init) by lia.
  destruct (list_has_min_term_index positions Hne) as [p_min [Hin_min Hmin_prop]].
  assert (Hpmin_le_min : term_index p_min <= min_i).
  { assert (Hin_min_i : In min_i (map term_index positions)).
    { unfold min_i. apply fold_left_min_in_list. exact Hmin_lt_init. }
    apply in_map_iff in Hin_min_i.
    destruct Hin_min_i as [q [Heq_q Hin_q]].
    specialize (Hmin_prop q Hin_q). lia. }
  assert (Hmin_le_pmin : min_i <= term_index p_min).
  { unfold min_i. apply min_i_le_term_index. exact Hin_min. }
  assert (Hmin_eq : min_i = term_index p_min) by lia.
  pose proof (reachable_damerau_term_index_upper_bound
                query n dict_prefix (std_pos i e) (Hreach _ Hin)) as Hupper.
  pose proof (reachable_damerau_term_index_lower_bound
                query n dict_prefix p_min (Hreach _ Hin_min)) as Hlower.
  pose proof (reachable_damerau_implies_edit_distance
                query n dict_prefix p_min (Hreach _ Hin_min)) as Herr_min.
  simpl in Hupper.
  fold min_i.
  rewrite Hmin_eq.
  lia.
Qed.

Lemma transition_state_transposition_closed_enter_exact : forall
  query n dict s c c_next i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  In (std_pos i e) (positions s) ->
  S i < length query ->
  nth_error query (S i) = Some c ->
  nth_error query i = Some c_next ->
  e < n ->
  In (special_pos i (S e))
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c c_next i e Hqlen Hall_reach Hin Hlt Hnth_next _ He_lt.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hmin_le : min_i <= i).
  { unfold min_i.
    change i with (term_index (std_pos i e)).
    apply min_i_le_term_index. exact Hin. }
  assert (Hoffset_bound : S (i - min_i) < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n + 4).
    { unfold min_i.
      eapply term_index_minus_min_bounded_damerau_enter
        with (query := query) (dict_prefix := dict).
      - exact Hall_reach.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction.
      - rewrite Hqlen. lia.
      - exact He_lt. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at (characteristic_vector c query min_i (2 * n + 6))
      (S (i - min_i)) = true).
  {
    rewrite cv_at_char_matches by exact Hoffset_bound.
    replace (min_i + S (i - min_i)) with (S i) by lia.
    unfold char_matches_at.
    rewrite Hnth_next.
    apply char_eq_refl. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_transposition.
  simpl.
  apply in_or_app. right.
  assert (Hguard : ((i + 1 <? query_length s) && (e <? n)) = true).
  { apply andb_true_iff. split.
    - apply Nat.ltb_lt. rewrite Hqlen. lia.
  - apply Nat.ltb_lt. exact He_lt. }
  rewrite Hguard.
  simpl.
  replace (S i - fold_left Nat.min (map term_index (positions s)) (query_length s))
    with (S (i - fold_left Nat.min (map term_index (positions s)) (query_length s)))
    by (fold min_i; lia).
  fold min_i.
  replace (n + (n + 0) + 6) with (2 * n + 6) by lia.
  rewrite Hcv.
  simpl. left. reflexivity.
Qed.

Lemma transition_state_transposition_closed_complete_exact : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  In (special_pos i e) (positions s) ->
  S i < length query ->
  nth_error query i = Some c ->
  In (std_pos (S (S i)) e)
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hin Hlt Hnth.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  assert (Hmin_le : min_i <= i).
  { unfold min_i.
    change i with (term_index (special_pos i e)).
    apply min_i_le_term_index. exact Hin. }
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n + 5).
    { unfold min_i.
      change i with (term_index (special_pos i e)).
      apply term_index_minus_min_bounded_damerau with
        (query := query) (dict_prefix := dict) (positions := positions s).
      - exact Hall_reach.
      - simpl. rewrite Hqlen. lia.
      - exact Hin.
      - intro Hempty. rewrite Hempty in Hin. contradiction. }
    fold min_i in Hbounded. lia. }
  assert (Hcv :
    cv_at (characteristic_vector c query min_i (2 * n + 6))
      (i - min_i) = true).
  {
    rewrite cv_at_char_matches by exact Hoffset_bound.
    replace (min_i + (i - min_i)) with i by lia.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (special_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_transposition.
  simpl.
  assert (Hguard : (i + 1 <? query_length s) = true).
  { apply Nat.ltb_lt. rewrite Hqlen. lia. }
  rewrite Hguard.
  fold min_i.
  replace (n + (n + 0) + 6) with (2 * n + 6) by lia.
  rewrite Hcv.
  simpl. left. reflexivity.
Qed.

Lemma transition_state_merge_split_closed_merge_exact : forall
  s c query n i e,
  In (std_pos i e) (positions s) ->
  S i < query_length s ->
  e < n ->
  In (std_pos (S (S i)) (S e))
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin Hi He.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  apply in_or_app. right.
  destruct ((i + 1 <? query_length s) && (e <? n)) eqn:Hcond.
  - simpl. left. f_equal; lia.
  - apply Bool.andb_false_iff in Hcond.
    destruct Hcond as [Hbad | Hbad].
    + apply Nat.ltb_nlt in Hbad. lia.
    + apply Nat.ltb_nlt in Hbad. lia.
Qed.

Lemma transition_state_merge_split_closed_enter_split_exact : forall
  s c query n i e,
  In (std_pos i e) (positions s) ->
  e < n ->
  In (special_pos i (S e))
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin He.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (std_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  apply in_or_app. right.
  assert (He_b : (e <? n) = true) by (apply Nat.ltb_lt; exact He).
  destruct ((i + 1 <? query_length s) && (e <? n)) eqn:Hmerge.
  - simpl. right.
    rewrite He_b. left. reflexivity.
  - simpl.
    rewrite He_b. left. reflexivity.
Qed.

Lemma transition_state_merge_split_closed_complete_split_exact : forall
  s c query n i e,
  In (special_pos i e) (positions s) ->
  i < query_length s ->
  In (std_pos (S i) e)
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)).
Proof.
  intros s c query n i e Hin Hi.
  apply epsilon_closure_includes_input.
  unfold transition_state_positions.
  apply in_flat_map.
  exists (special_pos i e). split; [exact Hin|].
  unfold transition_position.
  simpl.
  unfold transition_position_merge_split.
  simpl.
  assert (Hi_b : (i <? query_length s) = true) by (apply Nat.ltb_lt; exact Hi).
  rewrite Hi_b.
  simpl. left. reflexivity.
Qed.

Lemma transition_state_merge_split_covers_insert_exact : forall
  s c query n s' i e,
  transition_state MergeAndSplit s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  e < n ->
  positions_cover_merge_split (query_length s) (positions s') (std_pos i (S e)).
Proof.
  intros s c query n s' i e Htrans Hin He.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - apply transition_state_merge_split_closed_insert_exact; assumption.
Qed.

Lemma transition_state_merge_split_covers_match_exact : forall
  query n dict s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  positions_cover_merge_split (query_length s) (positions s') (std_pos (S i) e).
Proof.
  intros query n dict s c s' i e Hqlen Hall_reach Htrans Hin Hlt Hnth.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - eapply transition_state_merge_split_closed_match_exact; eauto.
Qed.

Lemma transition_state_merge_split_covers_substitute_exact : forall
  query n dict s c c' s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  positions_cover_merge_split (query_length s) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n dict s c c' s' i e Hqlen Hall_reach Htrans Hin Hlt Hnth Hneq He.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - eapply transition_state_merge_split_closed_substitute_exact; eauto.
Qed.

Lemma transition_state_merge_split_covers_merge_exact : forall
  s c query n s' i e,
  transition_state MergeAndSplit s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  S i < query_length s ->
  e < n ->
  positions_cover_merge_split (query_length s) (positions s') (std_pos (S (S i)) (S e)).
Proof.
  intros s c query n s' i e Htrans Hin Hi He.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - apply transition_state_merge_split_closed_merge_exact; assumption.
Qed.

Lemma transition_state_merge_split_covers_enter_split_exact : forall
  s c query n s' i e,
  transition_state MergeAndSplit s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  e < n ->
  positions_cover_merge_split (query_length s) (positions s') (special_pos i (S e)).
Proof.
  intros s c query n s' i e Htrans Hin He.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - apply transition_state_merge_split_closed_enter_split_exact; assumption.
Qed.

Lemma transition_state_merge_split_covers_complete_split_exact : forall
  s c query n s' i e,
  transition_state MergeAndSplit s c query n = Some s' ->
  In (special_pos i e) (positions s) ->
  i < query_length s ->
  positions_cover_merge_split (query_length s) (positions s') (std_pos (S i) e).
Proof.
  intros s c query n s' i e Htrans Hin Hi.
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans.
  - apply transition_state_merge_split_closed_complete_split_exact; assumption.
Qed.

Definition state_delete_chain_covered_merge_split (n : nat) (s : State) : Prop :=
  forall p k,
    In p (positions s) ->
    is_special p = false ->
    term_index p + k <= query_length s ->
    num_errors p + k <= n ->
    positions_cover_merge_split (query_length s) (positions s)
      (std_pos (term_index p + k) (num_errors p + k)).

Local Lemma subsumes_merge_split_std_inv : forall qlen p i e,
  subsumes MergeAndSplit qlen p (std_pos i e) = true ->
  exists e', p = std_pos i e' /\ e' < e.
Proof.
  intros qlen [j e' sp] i e Hsub.
  unfold subsumes in Hsub.
  simpl in Hsub.
  unfold subsumes_merge_split in Hsub.
  unfold position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (qlen <? j) eqn:Hpast; [discriminate|].
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  destruct (sp && (qlen <=? j) && negb (qlen <=? i)) eqn:Hspecial_final;
    [discriminate|].
  destruct (negb (Bool.eqb sp false)) eqn:Hvariant; [discriminate|].
  destruct sp; simpl in Hvariant; try discriminate.
  destruct (negb (e' <? e)) eqn:Herr; [discriminate|].
  apply Bool.negb_false_iff in Herr.
  apply Nat.ltb_lt in Herr.
  apply Nat.eqb_eq in Hsub. subst j.
  exists e'. split; [reflexivity | exact Herr].
Qed.

Local Lemma subsumes_merge_split_special_inv : forall qlen p i e,
  subsumes MergeAndSplit qlen p (special_pos i e) = true ->
  exists e', p = special_pos i e' /\ e' < e.
Proof.
  intros qlen [j e' sp] i e Hsub.
  unfold subsumes in Hsub.
  simpl in Hsub.
  unfold subsumes_merge_split in Hsub.
  unfold position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (qlen <? j) eqn:Hpast; [discriminate|].
  destruct ((negb (qlen <=? j)) && (qlen <=? i)) eqn:Hfinal;
    [discriminate|].
  destruct (sp && (qlen <=? j) && negb (qlen <=? i)) eqn:Hspecial_final;
    [discriminate|].
  destruct (negb (Bool.eqb sp true)) eqn:Hvariant; [discriminate|].
  destruct sp; simpl in Hvariant; try discriminate.
  destruct (negb (e' <? e)) eqn:Herr; [discriminate|].
  apply Bool.negb_false_iff in Herr.
  apply Nat.ltb_lt in Herr.
  apply Nat.eqb_eq in Hsub. subst j.
  exists e'. split; [reflexivity | exact Herr].
Qed.

Local Lemma subsumes_merge_split_std_same_index_lt : forall qlen i e1 e2,
  i <= qlen ->
  e1 < e2 ->
  subsumes MergeAndSplit qlen (std_pos i e1) (std_pos i e2) = true.
Proof.
  intros qlen i e1 e2 Hbound Herr.
  unfold subsumes, subsumes_merge_split, position_is_final_for_subsumption.
  simpl.
  assert (Hpast : (qlen <? i) = false) by (apply Nat.ltb_ge; exact Hbound).
  rewrite Hpast.
  destruct (qlen <=? i); simpl.
  - destruct (negb (e1 <? e2)) eqn:Hlt; [|apply Nat.eqb_refl].
    apply Bool.negb_true_iff in Hlt.
    apply Nat.ltb_ge in Hlt. lia.
  - destruct (negb (e1 <? e2)) eqn:Hlt; [|apply Nat.eqb_refl].
    apply Bool.negb_true_iff in Hlt.
    apply Nat.ltb_ge in Hlt. lia.
Qed.

Local Lemma subsumes_merge_split_special_same_index_lt : forall qlen i e1 e2,
  i <= qlen ->
  e1 < e2 ->
  subsumes MergeAndSplit qlen (special_pos i e1) (special_pos i e2) = true.
Proof.
  intros qlen i e1 e2 Hbound Herr.
  unfold subsumes, subsumes_merge_split, position_is_final_for_subsumption.
  simpl.
  assert (Hpast : (qlen <? i) = false) by (apply Nat.ltb_ge; exact Hbound).
  rewrite Hpast.
  destruct (qlen <=? i); simpl.
  - destruct (negb (e1 <? e2)) eqn:Hlt; [|apply Nat.eqb_refl].
    apply Bool.negb_true_iff in Hlt.
    apply Nat.ltb_ge in Hlt. lia.
  - destruct (negb (e1 <? e2)) eqn:Hlt; [|apply Nat.eqb_refl].
    apply Bool.negb_true_iff in Hlt.
    apply Nat.ltb_ge in Hlt. lia.
Qed.

Local Lemma positions_cover_merge_split_to_state_has_ms_completable : forall
  query n remaining ps p,
  (forall q, In q ps -> term_index q <= length query) ->
  positions_cover_merge_split (length query) ps p ->
  can_complete_to_final_ms query n remaining p ->
  exists q, In q ps /\ can_complete_to_final_ms query n remaining q.
Proof.
  intros query n remaining ps p Hbound Hcover Hcomplete.
  eapply positions_cover_merge_split_has_ms_completable; eauto.
Qed.

Local Lemma transition_state_merge_split_output_has_ms_completable : forall
  query n remaining s' p,
  (forall q, In q (positions s') -> term_index q <= length query) ->
  positions_cover_merge_split (length query) (positions s') p ->
  can_complete_to_final_ms query n remaining p ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n remaining s' p Hbound Hcover Hcomplete.
  destruct (positions_cover_merge_split_to_state_has_ms_completable
              query n remaining (positions s') p Hbound Hcover Hcomplete)
    as [q [Hq_in Hq_complete]].
  exists q. split; assumption.
Qed.

Lemma positions_cover_merge_split_delete_successor_covered : forall
  (query : list Char) n s i e,
  query_length s = length query ->
  state_delete_chain_covered_merge_split n s ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  S i <= length query ->
  S e <= n ->
  positions_cover_merge_split (length query) (positions s)
    (std_pos (S i) (S e)).
Proof.
  intros query n s i e Hqlen Hclosed Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hterm Herr.
  - subst p.
    pose proof (Hclosed (std_pos i e) 1 Hin eq_refl) as Hnext.
    simpl in Hnext.
    rewrite Hqlen in Hnext.
    replace (i + 1) with (S i) in Hnext by lia.
    replace (e + 1) with (S e) in Hnext by lia.
    apply Hnext; lia.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcover_next :
      positions_cover_merge_split (length query) (positions s)
        (std_pos (S i) (S e'))).
    { apply IH; try reflexivity; lia. }
    eapply cover_ms_sub.
    + exact Hcover_next.
    + apply subsumes_merge_split_std_same_index_lt; lia.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_insert_covered : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  e < n ->
  can_complete_to_final_ms query n remaining (std_pos i (S e)) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hout_bound Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp He_lt Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_insert_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining (std_pos i (S e'))).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; try lia; exact Hcomplete'.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_match_covered : forall
  query n dict remaining s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final_ms query n remaining (std_pos (S i) e) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e Hqlen Hall_reach Hout_bound
         Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_match_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining (std_pos (S i) e')).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; eauto.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_substitute_covered : forall
  query n dict remaining s c c' s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final_ms query n remaining (std_pos (S i) (S e)) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e Hqlen Hall_reach Hout_bound
         Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth Hneq He_lt Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_substitute_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining (std_pos (S i) (S e'))).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; try lia; eauto.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_merge_covered : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  S i < length query ->
  e < n ->
  can_complete_to_final_ms query n remaining (std_pos (S (S i)) (S e)) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hout_bound Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hi He_lt Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_merge_exact; eauto.
      rewrite Hqlen. exact Hi.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining
        (std_pos (S (S i)) (S e'))).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; try lia; exact Hcomplete'.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_enter_split_covered : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  e < n ->
  can_complete_to_final_ms query n remaining (special_pos i (S e)) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hout_bound Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp He_lt Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_enter_split_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining (special_pos i (S e'))).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; try lia; exact Hcomplete'.
Qed.

Lemma transition_state_merge_split_preserves_ms_complete_complete_split_covered : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) (special_pos i e) ->
  i < length query ->
  can_complete_to_final_ms query n remaining (std_pos (S i) e) ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hout_bound Htrans Hcover.
  remember (special_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hi Hcomplete.
  - subst p.
    eapply transition_state_merge_split_output_has_ms_completable.
    + exact Hout_bound.
    + rewrite <- Hqlen.
      eapply transition_state_merge_split_covers_complete_split_exact; eauto.
      rewrite Hqlen. exact Hi.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_merge_split_special_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final_ms query n remaining (std_pos (S i) e')).
    { eapply can_complete_ms_same_index_lower_errors.
      - exact Hcomplete.
      - reflexivity.
      - reflexivity.
      - simpl. lia. }
    eapply IH; try reflexivity; eauto.
Qed.

Local Lemma transition_state_merge_split_succeeds_from_closed_member : forall
  s c query n p,
  In p
    (epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros s c query n p Hin.
  unfold transition_state.
  set (closed_positions :=
    epsilon_closure
       (transition_state_positions MergeAndSplit (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s)) n (query_length s)) in *.
  destruct closed_positions as [|p0 rest].
  - contradiction.
  - exists (fold_left (fun s0 p1 => state_insert p1 s0)
              (p0 :: rest) (empty_state MergeAndSplit (query_length s))).
    reflexivity.
Qed.

Lemma transition_state_merge_split_succeeds_from_std_error_cover : forall
  query n s c i e,
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  e < n ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros query n s c i e Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp He_lt.
  - subst p.
    eapply transition_state_merge_split_succeeds_from_closed_member.
    eapply transition_state_merge_split_closed_insert_exact; eauto.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    eapply IH; try reflexivity; lia.
Qed.

Lemma transition_state_merge_split_succeeds_from_match_cover : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_merge_split query n dict p0) ->
  positions_cover_merge_split (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth.
  - subst p.
    eapply transition_state_merge_split_succeeds_from_closed_member.
    eapply transition_state_merge_split_closed_match_exact; eauto.
  - subst p.
    destruct (subsumes_merge_split_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    eapply IH; eauto.
Qed.

Lemma transition_state_merge_split_succeeds_from_complete_split_cover : forall
  query n s c i e,
  query_length s = length query ->
  positions_cover_merge_split (length query) (positions s) (special_pos i e) ->
  i < length query ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros query n s c i e Hqlen Hcover.
  remember (special_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hi.
  - subst p.
    eapply transition_state_merge_split_succeeds_from_closed_member.
    eapply transition_state_merge_split_closed_complete_split_exact; eauto.
    rewrite Hqlen. exact Hi.
  - subst p.
    destruct (subsumes_merge_split_special_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_lt]].
    subst q.
    eapply IH; eauto.
Qed.

Lemma transition_state_merge_split_succeeds_from_can_reach_covered : forall
  query n dict c remaining s p p_final,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_merge_split query n dict q) ->
  positions_cover_merge_split (length query) (positions s) p ->
  can_reach_ms query n p (c :: remaining) p_final ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros query n dict c remaining s p p_final Hqlen Hall_reach Hcover Hreach.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s dict Hqlen Hall_reach Hcover.
  induction Hreach; intros c0 remaining0 Hinput s0 dict0 Hqlen Hall_reach Hcover.
  - discriminate.
  - subst p.
    eapply transition_state_merge_split_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_match_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_merge_split_succeeds_from_complete_split_cover; eauto.
Qed.

Lemma transition_state_merge_split_preserves_can_reach_covered : forall
  query n dict c remaining s s' p p_final,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_merge_split query n dict q) ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  state_delete_chain_covered_merge_split n s ->
  transition_state MergeAndSplit s c query n = Some s' ->
  positions_cover_merge_split (length query) (positions s) p ->
  can_reach_ms query n p (c :: remaining) p_final ->
  term_index p_final = length query ->
  num_errors p_final <= n ->
  is_special p_final = false ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' p p_final Hqlen Hall_reach Hout_bound
         Hclosed Htrans Hcover Hreach Hterm_final Hfinal_err Hfinal_spec.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s s' dict Hqlen Hall_reach Hout_bound Hclosed
         Htrans Hcover.
  induction Hreach; intros c0 remaining0 Hinput s0 s' dict0 Hqlen
         Hall_reach Hout_bound Hclosed Htrans Hcover.
  - discriminate.
  - subst p.
    assert (Hcover_del :
      positions_cover_merge_split (length query) (positions s0)
        (std_pos (S i) (S e))).
    { eapply positions_cover_merge_split_delete_successor_covered; eauto; lia. }
    eapply IHHreach; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (std_pos (S i) e)).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_match_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (std_pos (S i) (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_substitute_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (std_pos i (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_insert_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (std_pos (S (S i)) (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_merge_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (special_pos i (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_enter_split_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final_ms query n remaining0 (std_pos (S i) e)).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_merge_split_preserves_ms_complete_complete_split_covered; eauto.
Qed.

Lemma transition_state_merge_split_preserves_state_has_ms_completable : forall
  query n dict c remaining s s',
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_merge_split query n dict q) ->
  (forall q, In q (positions s') -> term_index q <= length query) ->
  state_delete_chain_covered_merge_split n s ->
  transition_state MergeAndSplit s c query n = Some s' ->
  state_has_ms_completable query n (c :: remaining) s ->
  state_has_ms_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' Hqlen Hall_reach Hout_bound
         Hclosed Htrans [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_merge_split_preserves_can_reach_covered
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply cover_ms_in. exact Hin.
Qed.

Lemma transition_state_merge_split_succeeds_from_state_has_ms_completable : forall
  query n dict c remaining s,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_merge_split query n dict q) ->
  state_has_ms_completable query n (c :: remaining) s ->
  exists s', transition_state MergeAndSplit s c query n = Some s'.
Proof.
  intros query n dict c remaining s Hqlen Hall_reach [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_merge_split_succeeds_from_can_reach_covered
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply cover_ms_in. exact Hin.
Qed.

Lemma transition_state_merge_split_delete_chain_covered : forall
  s c query n s' p k,
  transition_state MergeAndSplit s c query n = Some s' ->
  In p (positions s') ->
  is_special p = false ->
  term_index p + k <= query_length s ->
  num_errors p + k <= n ->
  positions_cover_merge_split (query_length s) (positions s')
    (std_pos (term_index p + k) (num_errors p + k)).
Proof.
  intros s c query n s' p k Htrans Hin Hspec Hterm Herr.
  assert (Htrans_orig := Htrans).
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions MergeAndSplit (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  assert (Hp_closed : In p closed_positions).
  { apply in_fold_state_insert_origin with
      (init_state := empty_state MergeAndSplit (query_length s)).
    - unfold empty_state. reflexivity.
    - exact Hin. }
  assert (Htarget_closed :
    In (std_pos (term_index p + k) (num_errors p + k)) closed_positions).
  { unfold closed_positions.
    apply epsilon_closure_member_reaches_deletes_nonspecial;
      assumption. }
  eapply transition_state_merge_split_covers_closed_position.
  - exact Htrans_orig.
  - exact Htarget_closed.
Qed.

Lemma transition_state_merge_split_state_delete_chain_covered : forall
  s c query n s',
  transition_state MergeAndSplit s c query n = Some s' ->
  state_delete_chain_covered_merge_split n s'.
Proof.
  intros s c query n s' Htrans p k Hin Hspec Hterm Herr.
  assert (Hqlen : query_length s' = query_length s).
  { eapply transition_state_preserves_query_length. exact Htrans. }
  rewrite Hqlen in Hterm.
  rewrite Hqlen.
  eapply transition_state_merge_split_delete_chain_covered; eauto.
Qed.

Lemma initial_closed_state_delete_chain_covered_merge_split : forall n qlen,
  state_delete_chain_covered_merge_split n
    (mkState (epsilon_closure [initial_position] n qlen)
             MergeAndSplit qlen).
Proof.
  intros n qlen p k Hin Hspec Hterm Herr.
  simpl in *.
  apply cover_ms_in.
  apply epsilon_closure_member_reaches_deletes_nonspecial; assumption.
Qed.

Lemma state_has_ms_completable_empty_accepts_merge_split : forall query n s,
  query_length s = length query ->
  state_delete_chain_covered_merge_split n s ->
  state_has_ms_completable query n [] s ->
  state_is_final s = true.
Proof.
  intros query n s Hqlen Hclosed
         [p [Hp_in [p_final [Hreach [Hterm_final [Herr_final Hspec_final]]]]]].
  pose proof (can_reach_ms_empty_source_not_special
                query n p p_final Hreach Hspec_final) as Hsource_spec.
  pose proof (can_reach_ms_term_index_monotone
                query n p [] p_final Hreach) as Hterm_mono.
  pose proof (can_reach_ms_empty_remaining_errors
                query n p p_final Hreach Hsource_spec) as Herr_exact.
  set (k := term_index p_final - term_index p).
  assert (Hk_term : term_index p + k <= query_length s).
  { unfold k. rewrite Hqlen, Hterm_final. lia. }
  assert (Hk_err : num_errors p + k <= n).
  { unfold k. rewrite Herr_exact in Herr_final. exact Herr_final. }
  pose proof (Hclosed p k Hp_in Hsource_spec Hk_term Hk_err) as Hcover.
  assert (Htarget_eq :
    std_pos (term_index p + k) (num_errors p + k) = p_final).
  { destruct p_final as [i_f e_f sp_f].
    simpl in Hterm_final, Herr_final, Hspec_final, Hterm_mono, Herr_exact.
    destruct sp_f; simpl in Hspec_final; try discriminate.
    unfold k, std_pos. simpl.
    f_equal; lia. }
  rewrite Hqlen in Hcover.
  eapply covered_final_state_accepts_merge_split.
  - exact Hqlen.
  - rewrite Htarget_eq in Hcover. exact Hcover.
  - exact Hterm_final.
Qed.

Lemma automaton_run_merge_split_delete_chain_covered_from_state : forall
  query n dict s final,
  state_delete_chain_covered_merge_split n s ->
  automaton_run MergeAndSplit query n dict s = Some final ->
  state_delete_chain_covered_merge_split n final.
Proof.
  induction dict as [|c rest IH]; intros s final Hclosed Hrun.
  - simpl in Hrun. injection Hrun as Hfinal. subst final. exact Hclosed.
  - simpl in Hrun.
    destruct (transition_state MergeAndSplit s c query n) as [s_mid|] eqn:Htrans;
      [| discriminate].
    apply (IH s_mid final).
    + eapply transition_state_merge_split_state_delete_chain_covered.
      exact Htrans.
    + exact Hrun.
Qed.

Lemma automaton_run_merge_split_preserves_ms_completable_from_state : forall
  query n remaining dict_prefix s,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_merge_split query n dict_prefix q) ->
  state_delete_chain_covered_merge_split n s ->
  state_has_ms_completable query n remaining s ->
  exists final,
    automaton_run MergeAndSplit query n remaining s = Some final /\
    state_has_ms_completable query n [] final.
Proof.
  intros query n remaining.
  induction remaining as [|c rest IH]; intros dict_prefix s Hqlen Hall_reach
         Hclosed Hcomplete.
  - exists s. split; [reflexivity | exact Hcomplete].
  - destruct (transition_state_merge_split_succeeds_from_state_has_ms_completable
                query n dict_prefix c rest s Hqlen Hall_reach Hcomplete)
      as [s_mid Htrans].
    assert (Hrun_one : automaton_run MergeAndSplit query n [c] s = Some s_mid).
    { simpl. rewrite Htrans. reflexivity. }
    assert (Hall_reach_mid : forall q, In q (positions s_mid) ->
      position_reachable_merge_split query n (dict_prefix ++ [c]) q).
    { intros q Hq.
      eapply automaton_run_preserves_reachable_merge_split.
      - exact Hqlen.
      - exact Hrun_one.
      - exact Hall_reach.
      - exact Hq. }
    assert (Hout_bound : forall q, In q (positions s_mid) ->
      term_index q <= length query).
    { intros q Hq.
      eapply reachable_merge_split_term_index_bound_query.
      apply Hall_reach_mid. exact Hq. }
    assert (Hcomplete_mid : state_has_ms_completable query n rest s_mid).
    { eapply transition_state_merge_split_preserves_state_has_ms_completable; eauto. }
    assert (Hqlen_mid : query_length s_mid = length query).
    { rewrite (transition_state_preserves_query_length MergeAndSplit s c query n s_mid Htrans).
      exact Hqlen. }
    assert (Hclosed_mid : state_delete_chain_covered_merge_split n s_mid).
    { eapply transition_state_merge_split_state_delete_chain_covered.
      exact Htrans. }
    destruct (IH (dict_prefix ++ [c]) s_mid Hqlen_mid Hall_reach_mid
                Hclosed_mid Hcomplete_mid) as [final [Hrun_rest Hcomplete_final]].
    exists final. split.
    + simpl. rewrite Htrans. exact Hrun_rest.
    + exact Hcomplete_final.
Qed.

Lemma automaton_run_merge_split_completable_from_ms_bound : forall query dict n,
  merge_split_distance query dict <= n ->
  exists final,
    automaton_run_from_initial MergeAndSplit query n dict = Some final /\
    state_has_ms_completable query n [] final.
Proof.
  intros query dict n Hdist.
  unfold automaton_run_from_initial.
  set (init_closed :=
    mkState (epsilon_closure (positions (initial_state MergeAndSplit (length query)))
                         n (length query)) MergeAndSplit (length query)).
  assert (Hinit_eq :
    init_closed =
    mkState (epsilon_closure [initial_position] n (length query))
            MergeAndSplit (length query)).
  { unfold init_closed, initial_state. reflexivity. }
  rewrite Hinit_eq.
  apply automaton_run_merge_split_preserves_ms_completable_from_state
    with (dict_prefix := []).
  - reflexivity.
  - intros p Hp.
    eapply epsilon_closure_preserves_reachable_merge_split
      with (positions := [initial_position]).
    + intros p0 Hp0.
      simpl in Hp0. destruct Hp0 as [Hp0 | []]. subst p0.
      apply reach_ms_initial.
    + exact Hp.
  - apply initial_closed_state_delete_chain_covered_merge_split.
  - apply merge_split_bound_initial_closed_has_ms_completable.
    exact Hdist.
Qed.

Lemma automaton_run_from_initial_merge_split_delete_chain_covered : forall
  query n dict final,
  automaton_run_from_initial MergeAndSplit query n dict = Some final ->
  state_delete_chain_covered_merge_split n final.
Proof.
  intros query n dict final Hrun.
  unfold automaton_run_from_initial in Hrun.
  apply (automaton_run_merge_split_delete_chain_covered_from_state
           query n dict
           (mkState (epsilon_closure [initial_position] n (length query))
                    MergeAndSplit (length query))
           final).
  - apply initial_closed_state_delete_chain_covered_merge_split.
  - exact Hrun.
Qed.

Lemma automaton_run_merge_split_final_ms_completable_accepts : forall query n dict final,
  automaton_run_from_initial MergeAndSplit query n dict = Some final ->
  state_has_ms_completable query n [] final ->
  state_is_final final = true.
Proof.
  intros query n dict final Hrun Hcomplete.
  assert (Hqlen : query_length final = length query).
  { unfold automaton_run_from_initial in Hrun.
    rewrite (automaton_run_preserves_query_length MergeAndSplit query n dict
               (mkState (epsilon_closure [initial_position] n (length query))
                        MergeAndSplit (length query))
               final Hrun).
    reflexivity. }
  apply state_has_ms_completable_empty_accepts_merge_split with
    (query := query) (n := n).
  - exact Hqlen.
  - eapply automaton_run_from_initial_merge_split_delete_chain_covered.
    exact Hrun.
  - exact Hcomplete.
Qed.

Lemma initial_closed_has_completable_from_sequence : forall query dict n ops,
  valid_edit_sequence query dict 0 0 ops ->
  sequence_cost ops <= n ->
  state_has_completable query n dict
    (mkState (epsilon_closure [initial_position] n (length query))
             Standard (length query)).
Proof.
  intros query dict n ops Hvalid Hcost.
  exists initial_position.
  split.
  - simpl. apply epsilon_closure_includes_input. simpl. left. reflexivity.
  - apply valid_sequence_can_complete_initial with (ops := ops); assumption.
Qed.

Lemma lev_bound_initial_closed_has_completable : forall query dict n,
  lev_distance query dict <= n ->
  state_has_completable query n dict
    (mkState (epsilon_closure [initial_position] n (length query))
             Standard (length query)).
Proof.
  intros query dict n Hdist.
  destruct (optimal_sequence_exists query dict) as [ops [Hvalid Hcost]].
  apply initial_closed_has_completable_from_sequence with (ops := ops).
  - exact Hvalid.
  - lia.
Qed.

(** Key lemma: if p can reach p_final, and p' = (i - k, e - k) is valid,
    then p' can also reach p_final by first doing k delete operations.

    Preconditions:
    - k <= term_index p (so i - k >= 0)
    - k <= num_errors p (so e - k >= 0)
    - term_index p + k <= length query (so deletes have room)
    - num_errors p + k <= n (so error budget allows deletes)
*)
Lemma can_reach_prepend_deletes : forall query n p p_final remaining k,
  can_reach query n p remaining p_final ->
  p = std_pos (term_index p) (num_errors p) ->
  is_special p = false ->
  k <= term_index p ->
  k <= num_errors p ->
  term_index p <= length query ->
  num_errors p <= n ->
  exists p',
    p' = std_pos (term_index p - k) (num_errors p - k) /\
    can_reach query n p' remaining p_final.
Proof.
  intros query n p p_final remaining k Hreach Hp Hspec Hk_i Hk_e Hterm Herr.
  destruct p as [i e is_spec]. simpl in *.
  subst is_spec.
  (* Base case and inductive case: go backwards from p to p' *)
  induction k as [| k' IH].
  - (* k = 0: p' = p *)
    exists (std_pos i e).
    replace (i - 0) with i by lia.
    replace (e - 0) with e by lia.
    split; [reflexivity | exact Hreach].
  - (* k = S k': first get to (i - k', e - k'), then add one more delete *)
    assert (Hk'_bounds : k' <= i /\ k' <= e) by lia.
    destruct Hk'_bounds as [Hk'_i Hk'_e].
    specialize (IH Hk'_i Hk'_e).
    destruct IH as [p_mid [Hp_mid Hreach_mid]].
    (* p_mid = std_pos (i - k') (e - k') *)
    (* We need p' = std_pos (i - S k') (e - S k') *)
    exists (std_pos (i - S k') (e - S k')).
    split; [reflexivity |].
    (* Apply delete from p' to get to p_mid *)
    assert (Hsucc_i : S (i - S k') = i - k') by lia.
    assert (Hsucc_e : S (e - S k') = e - k') by lia.
    apply (can_reach_delete query n (std_pos (i - S k') (e - S k'))
                            remaining p_final (i - S k') (e - S k')).
    + reflexivity.
    + (* S (i - S k') = i - k' <= i <= length query *) lia.
    + (* e - S k' < n: from k' < k <= e and e <= n, so e - S k' < e <= n *)
      lia.
    + rewrite Hsucc_i, Hsucc_e.
      rewrite <- Hp_mid.
      exact Hreach_mid.
Qed.

(** Source position in can_reach is always non-special.
    All can_reach constructors either have p = std_pos i e (explicitly non-special)
    or p = p_final where p_final is non-special by other hypotheses. *)
Lemma can_reach_source_not_special : forall query n p remaining p_final,
  can_reach query n p remaining p_final ->
  is_special p_final = false ->
  is_special p = false.
Proof.
  intros query n p remaining p_final Hreach Hspec_final.
  induction Hreach.
  - (* can_reach_done: p = p_final *)
    exact Hspec_final.
  - (* can_reach_delete: p = std_pos i e *)
    subst p. unfold std_pos. simpl. reflexivity.
  - (* can_reach_match: p = std_pos i e *)
    subst p. unfold std_pos. simpl. reflexivity.
  - (* can_reach_substitute: p = std_pos i e *)
    subst p. unfold std_pos. simpl. reflexivity.
  - (* can_reach_insert: p = std_pos i e *)
    subst p. unfold std_pos. simpl. reflexivity.
Qed.

(** Errors are monotonically increasing along can_reach paths.
    Each operation either keeps errors the same (match) or increases by 1
    (delete, substitute, insert). *)
Lemma can_reach_errors_monotone : forall query n p remaining p_final,
  can_reach query n p remaining p_final ->
  num_errors p <= num_errors p_final.
Proof.
  intros query n p remaining p_final Hreach.
  induction Hreach.
  - (* can_reach_done: p = p_final *)
    lia.
  - (* can_reach_delete: from (i, e), next is (S i, S e) *)
    (* H: p = std_pos i e, IH: S e <= num_errors p_final *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_match: from (i, e), next is (S i, e) *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_substitute: from (i, e), next is (S i, S e) *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_insert: from (i, e), next is (i, S e) *)
    subst p. simpl. simpl in IHHreach. lia.
Qed.

(** Term index is monotonically increasing along can_reach paths.
    Each operation either keeps term_index the same (insert) or increases by 1
    (delete, match, substitute). *)
Lemma can_reach_term_index_monotone : forall query n p remaining p_final,
  can_reach query n p remaining p_final ->
  term_index p <= term_index p_final.
Proof.
  intros query n p remaining p_final Hreach.
  induction Hreach.
  - (* can_reach_done: p = p_final *)
    lia.
  - (* can_reach_delete: from (i, e), next is (S i, S e) *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_match: from (i, e), next is (S i, e) *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_substitute: from (i, e), next is (S i, S e) *)
    subst p. simpl. simpl in IHHreach. lia.
  - (* can_reach_insert: from (i, e), next is (i, S e) *)
    subst p. simpl. simpl in IHHreach. lia.
Qed.

(** For can_reach with empty remaining, errors increase exactly by term_index increase.
    With remaining = [], only delete operations are available, and each delete
    increments both term_index and num_errors by 1. So the total error increase
    equals the total term_index increase. *)
Lemma can_reach_empty_remaining_errors : forall query n p p_final,
  can_reach query n p [] p_final ->
  num_errors p_final = num_errors p + (term_index p_final - term_index p).
Proof.
  intros query n p p_final Hreach.
  remember ([] : list Char) as remaining eqn:Hremaining.
  revert Hremaining.
  induction Hreach; intros Hremaining; subst.
  - simpl. lia.
  - subst. simpl.
    specialize (IHHreach eq_refl).
    match goal with
    | H : can_reach _ _ (std_pos (S _) (S _)) [] _ |- _ =>
        pose proof (can_reach_term_index_monotone _ _ _ _ _ H) as Hmono
    end.
    simpl in *. lia.
  - discriminate.
  - discriminate.
  - discriminate.
Qed.

(** Helper: can_reach is monotonic in error count.
    If (i, e) can reach p_final, then (i, e') with e' <= e can also reach
    a final position (with correspondingly fewer errors).

    The key insight is that each error-increasing operation checks e < n.
    If we start with e' <= e, then at each step where the original path
    had error e + k, we have e' + k <= e + k. So if e + k < n was satisfied
    in the original, e' + k < n is also satisfied.

    The resulting p_final' has num_errors = num_errors p_final - (e - e').
*)
Lemma can_reach_lower_errors_aux : forall query n p remaining p_final diff,
  can_reach query n p remaining p_final ->
  is_special p = false ->
  num_errors p_final <= n ->
  diff <= num_errors p ->
  exists p_final',
    can_reach query n (std_pos (term_index p) (num_errors p - diff)) remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' = num_errors p_final - diff /\
    is_special p_final' = false.
Proof.
  intros query n p remaining p_final diff Hreach.
  induction Hreach; intros Hspec Hfinal_err Hdiff.
  - (* can_reach_done: p = p_final, remaining = [] *)
    simpl in *.
    exists (std_pos (term_index p) (num_errors p - diff)).
    repeat split; auto.
    apply can_reach_done.
  - (* can_reach_delete *)
    subst p. simpl in *.
    (* Original: (i, e) -> (S i, S e) via delete *)
    (* With diff reduction: (i, e - diff) -> (S i, S (e - diff)) via delete *)
    assert (He_diff_lt_n : e - diff < n) by lia.
    specialize (IHHreach eq_refl Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'.
    simpl in Hreach', Hterm', Herr', Hspec'.
    repeat split; try assumption; try lia.
    (* IH gives: can_reach from (S i, S e - diff), need: can_reach from (i, e - diff) *)
    (* Apply delete constructor: (i, e-diff) -> (S i, S(e-diff)) *)
    apply (can_reach_delete query n (std_pos i (e - diff)) remaining p_final' i (e - diff)).
    + reflexivity.
    + assumption. (* S i <= length query *)
    + exact He_diff_lt_n.
    + (* Hreach' has S e - diff, goal needs S (e - diff), these are equal when diff <= e *)
      replace (S (e - diff)) with (S e - diff) by lia.
      exact Hreach'.
  - (* can_reach_match *)
    subst p. simpl in *.
    (* Match doesn't change error count *)
    specialize (IHHreach eq_refl Hfinal_err Hdiff).
    destruct IHHreach as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'.
    simpl in Hreach', Hterm', Herr', Hspec'.
    repeat split; try assumption; try lia.
    apply (can_reach_match query n (std_pos i (e - diff)) c remaining p_final' i (e - diff)).
    + reflexivity.
    + assumption. (* i < length query *)
    + assumption. (* nth_error query i = Some c *)
    + exact Hreach'.
  - (* can_reach_substitute *)
    subst p. simpl in *.
    (* Substitute: (i, e) -> (S i, S e) *)
    assert (He_diff_lt_n : e - diff < n) by lia.
    specialize (IHHreach eq_refl Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'.
    simpl in Hreach', Hterm', Herr', Hspec'.
    repeat split; try assumption; try lia.
    (* Apply substitute constructor *)
    apply (can_reach_substitute query n (std_pos i (e - diff)) c c' remaining p_final' i (e - diff)).
    + reflexivity.
    + assumption. (* i < length query *)
    + assumption. (* nth_error query i = Some c' *)
    + assumption. (* c <> c' *)
    + exact He_diff_lt_n.
    + replace (S (e - diff)) with (S e - diff) by lia.
      exact Hreach'.
  - (* can_reach_insert *)
    subst p. simpl in *.
    (* Insert: (i, e) -> (i, S e) *)
    assert (He_diff_lt_n : e - diff < n) by lia.
    specialize (IHHreach eq_refl Hfinal_err ltac:(lia)).
    destruct IHHreach as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'.
    simpl in Hreach', Hterm', Herr', Hspec'.
    repeat split; try assumption; try lia.
    (* Apply insert constructor *)
    apply (can_reach_insert query n (std_pos i (e - diff)) c remaining p_final' i (e - diff)).
    + reflexivity.
    + exact He_diff_lt_n.
    + replace (S (e - diff)) with (S e - diff) by lia.
      exact Hreach'.
Qed.

(** Main corollary: lower errors version for std_pos *)
Lemma can_reach_lower_errors : forall query n i e remaining p_final e',
  can_reach query n (std_pos i e) remaining p_final ->
  e' <= e ->
  num_errors p_final <= n ->
  exists p_final',
    can_reach query n (std_pos i e') remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' = num_errors p_final - (e - e') /\
    is_special p_final' = false.
Proof.
  intros query n i e remaining p_final e' Hreach He'_le Hfinal_err.
  pose proof (can_reach_lower_errors_aux query n (std_pos i e) remaining p_final (e - e')
    Hreach eq_refl Hfinal_err ltac:(simpl; lia)) as Haux.
  simpl in Haux.
  replace (e - (e - e')) with e' in Haux by lia.
  exact Haux.
Qed.

(** If a non-special position is ahead in the query with enough saved error
    budget, it can simulate the original completion path. When the original
    path consumes a dictionary character that the ahead position can no longer
    match at the same query index, the ahead position consumes it with INSERT;
    the arithmetic premise [i' - i <= e - e'] pays for those inserts. *)
Lemma can_reach_higher_index : forall
  (query : list Char) (n i e i' e' : nat) (remaining : list Char) (p_final : Position),
  can_reach query n (std_pos i e) remaining p_final ->
  term_index p_final = length query ->
  num_errors p_final <= n ->
  i' > i ->
  i' <= length query ->
  e' <= e ->
  i' - i <= e - e' ->
  exists p_final',
    can_reach query n (std_pos i' e') remaining p_final' /\
    term_index p_final' = term_index p_final /\
    num_errors p_final' <= n /\
    is_special p_final' = false.
Proof.
  intros query n i e i' e' remaining p_final Hreach.
  remember (std_pos i e) as p eqn:Hp.
  revert i e i' e' Hp.
  induction Hreach; intros i0 e0 i' e' Hp Hterm_final Hfinal_err
                          Hi'_gt Hi'_qlen He'_le Hdiff.
  - rewrite Hp in Hterm_final. simpl in Hterm_final. lia.
  -
    assert (Htail_start : can_reach query n (std_pos (S i0) (S e0)) remaining p_final).
    { match goal with
      | Hmem : ?p0 = std_pos i0 e0,
        Hstart : ?p0 = std_pos ?j ?f,
        Htail : can_reach query n (std_pos (S ?j) (S ?f)) remaining p_final |- _ =>
          rewrite Hmem in Hstart; inversion Hstart; subst; exact Htail
      end. }
    destruct (Nat.eq_dec i' (S i0)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      pose proof (can_reach_lower_errors query n (S i0) (S e0) remaining p_final e'
        Htail_start ltac:(lia) Hfinal_err) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hreach'; try exact Hterm'; try exact Hspec'.
      rewrite Herr'. lia.
    + pose proof (IHHreach (S i0) (S e0) i' e'
        ltac:(match goal with
              | Hmem : ?p0 = std_pos i0 e0,
                Hstart : ?p0 = std_pos ?j ?f |- std_pos (S ?j) (S ?f) = std_pos (S i0) (S e0) =>
                  rewrite Hmem in Hstart; inversion Hstart; subst; reflexivity
              end)
        Hterm_final Hfinal_err ltac:(lia) Hi'_qlen ltac:(lia) ltac:(lia)) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; assumption.
  -
    assert (Htail_start : can_reach query n (std_pos (S i0) e0) remaining p_final).
    { match goal with
      | Hmem : ?p0 = std_pos i0 e0,
        Hstart : ?p0 = std_pos ?j ?f,
        Htail : can_reach query n (std_pos (S ?j) ?f) remaining p_final |- _ =>
          rewrite Hmem in Hstart; inversion Hstart; subst; exact Htail
      end. }
    pose proof (can_reach_errors_monotone query n (std_pos (S i0) e0) remaining p_final Htail_start) as Herr_mono.
    simpl in Herr_mono.
    destruct (Nat.eq_dec i' (S i0)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      pose proof (can_reach_lower_errors query n (S i0) e0 remaining p_final (S e')
        Htail_start ltac:(lia) Hfinal_err) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Hspec'.
      * eapply can_reach_insert; [reflexivity | lia | exact Hreach'].
      * rewrite Herr'. lia.
    + pose proof (IHHreach (S i0) e0 i' (S e')
        ltac:(match goal with
              | Hmem : ?p0 = std_pos i0 e0,
                Hstart : ?p0 = std_pos ?j ?f |- std_pos (S ?j) ?f = std_pos (S i0) e0 =>
                  rewrite Hmem in Hstart; inversion Hstart; subst; reflexivity
              end)
        Hterm_final Hfinal_err ltac:(lia) Hi'_qlen ltac:(lia) ltac:(lia)) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr'; try exact Hspec'.
      eapply can_reach_insert; [reflexivity | lia | exact Hreach'].
  -
    assert (Htail_start : can_reach query n (std_pos (S i0) (S e0)) remaining p_final).
    { match goal with
      | Hmem : ?p0 = std_pos i0 e0,
        Hstart : ?p0 = std_pos ?j ?f,
        Htail : can_reach query n (std_pos (S ?j) (S ?f)) remaining p_final |- _ =>
          rewrite Hmem in Hstart; inversion Hstart; subst; exact Htail
      end. }
    pose proof (can_reach_errors_monotone query n (std_pos (S i0) (S e0)) remaining p_final Htail_start) as Herr_mono.
    simpl in Herr_mono.
    destruct (Nat.eq_dec i' (S i0)) as [Hi'_eq | Hi'_neq].
    + subst i'.
      pose proof (can_reach_lower_errors query n (S i0) (S e0) remaining p_final (S e')
        Htail_start ltac:(lia) Hfinal_err) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Hspec'.
      * eapply can_reach_insert; [reflexivity | lia | exact Hreach'].
      * rewrite Herr'. lia.
    + pose proof (IHHreach (S i0) (S e0) i' (S e')
        ltac:(match goal with
              | Hmem : ?p0 = std_pos i0 e0,
                Hstart : ?p0 = std_pos ?j ?f |- std_pos (S ?j) (S ?f) = std_pos (S i0) (S e0) =>
                  rewrite Hmem in Hstart; inversion Hstart; subst; reflexivity
              end)
        Hterm_final Hfinal_err ltac:(lia) Hi'_qlen ltac:(lia) ltac:(lia)) as
        [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; try exact Hterm'; try exact Herr'; try exact Hspec'.
      eapply can_reach_insert; [reflexivity | lia | exact Hreach'].
  -
    assert (Htail_start : can_reach query n (std_pos i0 (S e0)) remaining p_final).
    { match goal with
      | Hmem : ?p0 = std_pos i0 e0,
        Hstart : ?p0 = std_pos ?j ?f,
        Htail : can_reach query n (std_pos ?j (S ?f)) remaining p_final |- _ =>
          rewrite Hmem in Hstart; inversion Hstart; subst; exact Htail
      end. }
    pose proof (can_reach_errors_monotone query n (std_pos i0 (S e0)) remaining p_final Htail_start) as Herr_mono.
    simpl in Herr_mono.
    pose proof (IHHreach i0 (S e0) i' (S e')
      ltac:(match goal with
            | Hmem : ?p0 = std_pos i0 e0,
              Hstart : ?p0 = std_pos ?j ?f |- std_pos ?j (S ?f) = std_pos i0 (S e0) =>
                rewrite Hmem in Hstart; inversion Hstart; subst; reflexivity
            end)
      Hterm_final Hfinal_err Hi'_gt Hi'_qlen ltac:(lia) ltac:(lia)) as
      [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split; try exact Hterm'; try exact Herr'; try exact Hspec'.
    eapply can_reach_insert; [reflexivity | lia | exact Hreach'].
Qed.

(** Helper: construct a path of k deletes from position (i, e) to (i+k, e+k).
    Used when remaining = [] to consume query characters via deletion. *)
Lemma can_reach_deletes : forall query n i e k,
  i + k <= length query ->
  e + k <= n ->
  can_reach query n (std_pos i e) [] (std_pos (i + k) (e + k)).
Proof.
  intros query n i e k.
  generalize dependent e.
  generalize dependent i.
  induction k as [| k' IHk].
  - (* k = 0 *)
    intros i e Hi_k He_k.
    replace (i + 0) with i by lia.
    replace (e + 0) with e by lia.
    apply can_reach_done.
  - (* k = S k' *)
    intros i e Hi_k He_k.
    apply (can_reach_delete query n (std_pos i e) [] (std_pos (i + S k') (e + S k')) i e).
    + reflexivity.
    + lia.
    + lia.
    + (* IH: from (S i, S e), do k' deletes to reach (S i + k', S e + k') *)
      replace (i + S k') with (S i + k') by lia.
      replace (e + S k') with (S e + k') by lia.
      apply IHk; lia.
Qed.

(** Helper: can_reach from a position that is "ahead" in the query.
    If (i, e) can reach p_final via remaining, and (i', e') satisfies:
    - i' >= i (ahead in query)
    - i' - i <= e - e' (position advance bounded by error savings)
    - e' <= e
    Then (i', e') can also reach some final position via remaining.

    The proof uses INSERT operations to consume dictionary characters
    without advancing the query position, catching up to the original path.

    Precondition: p_final is a final position (term_index = length query).
*)
Lemma can_reach_from_ahead : forall
  query n i e remaining p_final i' e',
  can_reach query n (std_pos i e) remaining p_final ->
  i' >= i ->
  i' - i <= e - e' ->
  e' <= e ->
  i' <= length query ->
  num_errors p_final <= n ->
  term_index p_final = length query ->  (* Added: p_final is final *)
  exists p_final',
    can_reach query n (std_pos i' e') remaining p_final' /\
    term_index p_final' = length query /\
    num_errors p_final' <= n /\
    is_special p_final' = false.
Proof.
  intros query n i e remaining p_final i' e' Hreach Hi'_ge Hdiff He'_le Hi'_qlen
         Hfinal_err Hterm_final.
  (* First handle the special case i' = i using can_reach_lower_errors *)
  destruct (Nat.eq_dec i' i) as [Hi'_eq | Hi'_neq].
  - (* i' = i: use can_reach_lower_errors directly *)
    subst i'.
    pose proof (can_reach_lower_errors query n i e remaining p_final e'
      Hreach He'_le Hfinal_err) as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split.
    + exact Hreach'.
    + rewrite Hterm'. exact Hterm_final.
    + (* num_errors p_final' = num_errors p_final - (e - e') <= n *)
      destruct p_final as [i_f e_f sp_f]. simpl in Herr', Hfinal_err. lia.
    + exact Hspec'.
  - (* i' > i: use INSERT to consume dict chars, then recurse *)
    assert (Hi'_gt : i' > i) by lia.
    pose proof (can_reach_higher_index query n i e i' e' remaining p_final
                  Hreach Hterm_final Hfinal_err Hi'_gt Hi'_qlen He'_le Hdiff) as
	      [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. repeat split.
    + exact Hreach'.
    + rewrite Hterm'. exact Hterm_final.
    + exact Herr'.
    + exact Hspec'.
Qed.

(** Key theorem: Subsumption preserves can_complete_to_final.

    If position p can complete to final, and p' subsumes p (via subsumes_standard),
    then p' can also complete to final.

    The key insight is that subsumption gives us "slack" in term_index that
    we can use up with delete operations to reach the same path as p.

    Precondition term_index p' <= length query is needed because:
    - When i' > length query, no path to final (term_index = length query) exists
    - In automaton contexts, this is always satisfied since positions have bounded term_index
*)
Lemma subsumption_preserves_can_complete : forall
  query n remaining p p',
  can_complete_to_final query n remaining p ->
  subsumes_standard (length query) p' p = true ->
  is_special p = false ->
  is_special p' = false ->
  term_index p' <= length query ->
  can_complete_to_final query n remaining p'.
Proof.
  intros query n remaining p p' Hcomplete Hsub Hspec Hspec' Hi'_qlen.
  destruct Hcomplete as [p_final [Hreach [Hterm_final [Herr_final Hspec_final]]]].
  destruct p as [i e is_spec].
  destruct p' as [i' e' is_spec'].
  simpl in Hspec, Hspec'. subst is_spec is_spec'.
  (* Extract subsumption conditions from subsumes_standard *)
  (* subsumes_standard structure:
     if (negb p1_final) && p2_final then false
     else (e' <=? e) && (abs_diff i' i <=? e - e')
  *)
  unfold subsumes_standard in Hsub.
  simpl in Hsub.
  (* The if-then-else: destruct on the condition *)
  destruct ((negb (position_is_final_for_subsumption (length query) (mkPosition i' e' false)))
            && (position_is_final_for_subsumption (length query) (mkPosition i e false))) eqn:Hfinal_check.
  - (* Condition is true, so subsumes_standard returns false *)
    (* But Hsub says it's true - contradiction *)
    discriminate.
  - (* Condition is false, so we're in the else branch *)
    (* Hsub = (e' <=? e) && (abs_diff i' i <=? e - e') = true *)
    rewrite Bool.andb_true_iff in Hsub.
    destruct Hsub as [Herr_le Hdist_le].
    apply Nat.leb_le in Herr_le.
    apply Nat.leb_le in Hdist_le.
    (* e' <= e and |i' - i| <= e - e' *)
    unfold abs_diff in Hdist_le.
    destruct (i' <? i) eqn:Hi'_lt_i.
    + (* i' < i: abs_diff = i - i', so i - i' <= e - e' *)
      apply Nat.ltb_lt in Hi'_lt_i.
      (* p' = (i', e') can reach p_final via path:
         1. (i', e') --k deletes--> (i, e' + k) where k = i - i'
         2. (i, e' + k) --lower errors version of p's path--> p_final'

         Since e' + (i - i') <= e, we can use can_reach_lower_errors.
         Then can_reach_prepend_deletes gives us the full path from (i', e'). *)

      (* Step 1: Use can_reach_lower_errors to get path from (i, e' + (i - i')) to p_final' *)
      (* First, simplify Hdist_le using i' < i *)
      assert (Hdist_le_simple : i - i' <= e - e').
      { (* Since i' < i, we have i' <=? i = true, so abs_diff reduces to i - i' *)
        assert (Hi'_le_i_bool : (i' <=? i) = true) by (apply Nat.leb_le; lia).
        rewrite Hi'_le_i_bool in Hdist_le. exact Hdist_le. }
      assert (He_intermed_le_e : e' + (i - i') <= e).
      { (* From Hdist_le_simple: i - i' <= e - e'
           With Herr_le: e' <= e, so e - e' is well-defined (non-negative).
           Therefore: e' + (i - i') <= e' + (e - e') = e *)
        lia. }
      pose proof (can_reach_lower_errors query n i e remaining p_final (e' + (i - i'))
        Hreach He_intermed_le_e Herr_final) as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].

      (* Step 2: Use can_reach_prepend_deletes to prepend (i - i') deletes *)
      assert (Hk_i : i - i' <= i) by lia.
      assert (Hk_e_intermed : i - i' <= e' + (i - i')) by lia.
      assert (Hi_qlen : i <= length query).
      { (* From can_reach_term_index_monotone: term_index p <= term_index p_final *)
        pose proof (can_reach_term_index_monotone query n (mkPosition i e false) remaining p_final Hreach) as Hmono.
        simpl in Hmono. rewrite Hterm_final in Hmono. exact Hmono. }
      assert (He_intermed_n : e' + (i - i') <= n).
      { (* From can_reach_errors_monotone: e <= num_errors p_final *)
        (* From Herr_final: num_errors p_final <= n *)
        (* So e <= n, and since e' + (i - i') <= e, we have e' + (i - i') <= n *)
        assert (He_le_pfinal : e <= num_errors p_final).
        { apply (can_reach_errors_monotone query n (mkPosition i e false) remaining p_final).
          exact Hreach. }
        lia. }
      pose proof (can_reach_prepend_deletes query n (std_pos i (e' + (i - i'))) p_final' remaining (i - i')
        Hreach' eq_refl eq_refl Hk_i Hk_e_intermed Hi_qlen He_intermed_n) as [p'' [Hp'' Hreach'']].

      (* p'' = (i - (i - i'), (e' + (i - i')) - (i - i')) = (i', e') *)
      assert (Hp''_eq : p'' = std_pos i' e').
      { rewrite Hp''. unfold std_pos. simpl.
        f_equal.
        - (* i - (i - i') = i' *)
          (* Since i' < i, we have i' <= i *)
          lia.
        - (* (e' + (i - i')) - (i - i') = e' *)
          lia. }
      rewrite Hp''_eq in Hreach''.

      (* Now we have can_reach from (i', e') to p_final' *)
      exists p_final'. repeat split.
      * exact Hreach''.
      * rewrite Hterm'. exact Hterm_final.
      * rewrite Herr'. lia.
      * exact Hspec'.
    + (* i' >= i: abs_diff = i' - i, so i' - i <= e - e' *)
      apply Nat.ltb_ge in Hi'_lt_i.
      (* p' = (i', e') is at or ahead of p = (i, e) in the query *)
      (* Use can_reach_from_ahead to construct the path *)
      simpl in Hi'_qlen.
      (* Simplify Hdist_le: since i <= i', abs_diff reduces to i' - i *)
      assert (Hdist_le_simple : i' - i <= e - e').
      { unfold abs_diff in Hdist_le.
        destruct (i' <=? i) eqn:Hi'_le_i.
        - (* i' <= i and i <= i', so i' = i *)
          apply Nat.leb_le in Hi'_le_i.
          assert (Hi_eq : i' = i) by lia.
          subst i'. simpl. lia.
        - (* i' > i, so abs_diff = i' - i *)
          exact Hdist_le. }
      pose proof (can_reach_from_ahead query n i e remaining p_final i' e'
        Hreach Hi'_lt_i Hdist_le_simple Herr_le Hi'_qlen Herr_final Hterm_final)
        as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
      exists p_final'. repeat split; auto.
Qed.

Lemma can_complete_same_index_lower_errors : forall query n remaining i e e',
  i <= length query ->
  e' <= e ->
  can_complete_to_final query n remaining (std_pos i e) ->
  can_complete_to_final query n remaining (std_pos i e').
Proof.
  intros query n remaining i e e' Hi Herr Hcomplete.
  eapply subsumption_preserves_can_complete.
  - exact Hcomplete.
  - apply subsumes_standard_same_index_error_widen. exact Herr.
  - reflexivity.
  - reflexivity.
  - simpl. exact Hi.
Qed.

Lemma can_complete_prepend_deletes : forall query n remaining i e k,
  i + k <= length query ->
  e + k <= n ->
  can_complete_to_final query n remaining (std_pos (i + k) (e + k)) ->
  can_complete_to_final query n remaining (std_pos i e).
Proof.
  intros query n remaining i e k Hterm Herr
         [p_final [Hreach [Hfinal_term [Hfinal_err Hfinal_spec]]]].
  destruct (can_reach_prepend_deletes
              query n (std_pos (i + k) (e + k)) p_final remaining k)
    as [p' [Hp' Hreach']].
  - exact Hreach.
  - reflexivity.
  - reflexivity.
  - simpl. lia.
  - simpl. lia.
  - simpl. exact Hterm.
  - simpl. exact Herr.
  - assert (Hp'_start : p' = std_pos i e).
    { rewrite Hp'. unfold std_pos. simpl. f_equal; lia. }
    exists p_final. repeat split; try assumption.
    rewrite <- Hp'_start. exact Hreach'.
Qed.

Local Lemma subsumes_transposition_std_inv : forall qlen p i e,
  subsumes Transposition qlen p (std_pos i e) = true ->
  exists e', p = std_pos i e' /\ e' <= e.
Proof.
  intros qlen [j e' sp] i e Hsub.
  unfold subsumes, subsumes_transposition, position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (negb (qlen <=? j) && (qlen <=? i)) eqn:Hfinal; [discriminate|].
  destruct (Bool.eqb sp false) eqn:Hvariant; [|discriminate].
  destruct sp; simpl in Hvariant; try discriminate.
  destruct (negb (e' <=? e)) eqn:Herr; [discriminate|].
  apply Bool.negb_false_iff in Herr.
  apply Nat.leb_le in Herr.
  apply Nat.eqb_eq in Hsub. subst j.
  exists e'. split; [reflexivity | exact Herr].
Qed.

Local Lemma subsumes_transposition_std_same_index_le : forall qlen i e1 e2,
  e1 <= e2 ->
  subsumes Transposition qlen (std_pos i e1) (std_pos i e2) = true.
Proof.
  intros qlen i e1 e2 Herr.
  unfold subsumes, subsumes_transposition, position_is_final_for_subsumption.
  simpl.
  destruct (negb (qlen <=? i) && (qlen <=? i)) eqn:Hfinal.
  - apply Bool.andb_true_iff in Hfinal.
    destruct Hfinal as [Hnot Hyes].
    apply Bool.negb_true_iff in Hnot.
    rewrite Hyes in Hnot. discriminate.
  - assert (Herr_bool : (e1 <=? e2) = true).
    { apply Nat.leb_le. exact Herr. }
    rewrite Herr_bool. simpl. apply Nat.eqb_refl.
Qed.

Local Lemma subsumes_transposition_special_inv : forall qlen p i e,
  subsumes Transposition qlen p (special_pos i e) = true ->
  exists e', p = special_pos i e' /\ e' <= e.
Proof.
  intros qlen [j e' sp] i e Hsub.
  unfold subsumes, subsumes_transposition, position_is_final_for_subsumption in Hsub.
  simpl in Hsub.
  destruct (negb (qlen <=? j) && (qlen <=? i)) eqn:Hfinal; [discriminate|].
  destruct (Bool.eqb sp true) eqn:Hvariant; [|discriminate].
  destruct sp; simpl in Hvariant; try discriminate.
  destruct (negb (e' <=? e)) eqn:Herr; [discriminate|].
  apply Bool.negb_false_iff in Herr.
  apply Nat.leb_le in Herr.
  apply Nat.eqb_eq in Hsub. subst j.
  exists e'. split; [reflexivity | exact Herr].
Qed.

Local Lemma subsumes_transposition_special_same_index_le : forall qlen i e1 e2,
  e1 <= e2 ->
  subsumes Transposition qlen (special_pos i e1) (special_pos i e2) = true.
Proof.
  intros qlen i e1 e2 Herr.
  unfold subsumes, subsumes_transposition, position_is_final_for_subsumption.
  simpl.
  destruct (negb (qlen <=? i) && (qlen <=? i)) eqn:Hfinal.
  - apply Bool.andb_true_iff in Hfinal.
    destruct Hfinal as [Hnot Hyes].
    apply Bool.negb_true_iff in Hnot.
    rewrite Hyes in Hnot. discriminate.
  - assert (Herr_bool : (e1 <=? e2) = true).
    { apply Nat.leb_le. exact Herr. }
    rewrite Herr_bool. simpl. apply Nat.eqb_refl.
Qed.

Definition state_delete_chain_covered_transposition (n : nat) (s : State) : Prop :=
  forall p k,
    In p (positions s) ->
    is_special p = false ->
    term_index p + k <= query_length s ->
    num_errors p + k <= n ->
    positions_cover_transposition (query_length s) (positions s)
      (std_pos (term_index p + k) (num_errors p + k)).

Local Lemma positions_cover_transposition_to_state_has_completable : forall
  query n remaining ps p,
  positions_cover_transposition (length query) ps p ->
  can_complete_to_final query n remaining p ->
  exists q, In q ps /\ can_complete_to_final query n remaining q.
Proof.
  intros query n remaining ps p Hcover.
  induction Hcover as [p Hin | p q Hcover IH Hsub]; intros Hcomplete.
  - exists p. split; assumption.
  - destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
    pose proof (can_reach_source_not_special query n p remaining p_final Hreach Hspec)
      as Hp_spec.
    destruct p as [i e sp]; simpl in Hp_spec. destruct sp; try discriminate.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_q]].
    subst q.
    apply IH.
    destruct (can_reach_lower_errors query n i e remaining p_final e'
                Hreach Herr_q Herr)
      as [p_final' [Hreach' [Hterm' [Herr' Hspec']]]].
    exists p_final'. split; [exact Hreach'|].
    split; [lia|].
    split; [lia|exact Hspec'].
Qed.

Local Lemma transition_state_transposition_output_has_completable : forall
  query n remaining s' p,
  positions_cover_transposition (length query) (positions s') p ->
  can_complete_to_final query n remaining p ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s' p Hcover Hcomplete.
  destruct (positions_cover_transposition_to_state_has_completable
              query n remaining (positions s') p Hcover Hcomplete)
    as [q [Hq_in Hq_complete]].
  exists q. split; assumption.
Qed.

Lemma positions_cover_transposition_delete_successor_covered : forall
  (query : list Char) n s i e,
  query_length s = length query ->
  state_delete_chain_covered_transposition n s ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  S i <= length query ->
  S e <= n ->
  positions_cover_transposition (length query) (positions s)
    (std_pos (S i) (S e)).
Proof.
  intros query n s i e Hqlen Hclosed Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hterm Herr.
  - subst p.
    pose proof (Hclosed (std_pos i e) 1 Hin eq_refl) as Hnext.
    simpl in Hnext.
    rewrite Hqlen in Hnext.
    replace (i + 1) with (S i) in Hnext by lia.
    replace (e + 1) with (S e) in Hnext by lia.
    apply Hnext; lia.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover_next :
      positions_cover_transposition (length query) (positions s)
        (std_pos (S i) (S e'))).
    { apply IH; try reflexivity; lia. }
    eapply cover_trans_sub.
    + exact Hcover_next.
    + apply subsumes_transposition_std_same_index_le. lia.
Qed.

Lemma transition_state_transposition_preserves_complete_insert_covered : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i <= length query ->
  e < n ->
  can_complete_to_final query n remaining (std_pos i (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hi_bound He_lt Hcomplete.
  - subst p.
    eapply transition_state_transposition_output_has_completable.
    + rewrite <- Hqlen.
      eapply transition_state_transposition_covers_closed_position; eauto.
      apply transition_state_transposition_closed_insert_exact; [exact Hin | exact He_lt].
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final query n remaining (std_pos i (S e'))).
    { apply (can_complete_same_index_lower_errors query n remaining i (S e) (S e')).
      - exact Hi_bound.
      - lia.
      - exact Hcomplete. }
    eapply IH; try reflexivity; try lia; exact Hcomplete'.
Qed.

Lemma transition_state_transposition_preserves_complete_match_covered : forall
  query n dict remaining s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e Hqlen Hall_reach Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth Hcomplete.
  - subst p.
    eapply transition_state_transposition_output_has_completable.
    + rewrite <- Hqlen.
      eapply transition_state_transposition_covers_closed_position; eauto.
      eapply transition_state_transposition_closed_match_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final query n remaining (std_pos (S i) e')).
    { eapply can_complete_same_index_lower_errors.
      - simpl. lia.
      - exact Herr_le.
      - exact Hcomplete. }
    eapply IH; try reflexivity; eauto.
Qed.

Lemma transition_state_transposition_preserves_complete_substitute_covered : forall
  query n dict remaining s c c' s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e Hqlen Hall_reach Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth Hneq He_lt Hcomplete.
  - subst p.
    eapply transition_state_transposition_output_has_completable.
    + rewrite <- Hqlen.
      eapply transition_state_transposition_covers_closed_position; eauto.
      eapply transition_state_transposition_closed_substitute_exact; eauto.
    + exact Hcomplete.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcomplete' :
      can_complete_to_final query n remaining (std_pos (S i) (S e'))).
    { apply (can_complete_same_index_lower_errors query n remaining (S i) (S e) (S e')).
      - simpl. lia.
      - lia.
      - exact Hcomplete. }
    eapply IH; try reflexivity; try lia; eauto.
Qed.

Local Lemma transition_state_transposition_succeeds_from_closed_member : forall
  s c query n p,
  In p
    (epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros s c query n p Hin.
  unfold transition_state.
  set (closed_positions :=
    epsilon_closure
       (transition_state_positions Transposition (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s)) n (query_length s)) in *.
  destruct closed_positions as [|p0 rest].
  - contradiction.
  - exists (fold_left (fun s0 p1 => state_insert p1 s0)
              (p0 :: rest) (empty_state Transposition (query_length s))).
    reflexivity.
Qed.

Lemma transition_state_transposition_succeeds_from_std_error_cover : forall
  query n s c i e,
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  e < n ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros query n s c i e Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp He_lt.
  - subst p.
    eapply transition_state_transposition_succeeds_from_closed_member.
    eapply transition_state_transposition_closed_insert_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    eapply IH; try reflexivity; lia.
Qed.

Lemma transition_state_transposition_succeeds_from_match_cover : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth.
  - subst p.
    eapply transition_state_transposition_succeeds_from_closed_member.
    eapply transition_state_transposition_closed_match_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq _]].
	    subst q.
	    eapply IH; eauto.
Qed.

Lemma transition_state_transposition_covers_insert_covered : forall
  query n s c s' i e,
  query_length s = length query ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  e < n ->
  positions_cover_transposition (length query) (positions s') (std_pos i (S e)).
Proof.
  intros query n s c s' i e Hqlen Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp He_lt.
  - subst p.
    rewrite <- Hqlen.
    eapply transition_state_transposition_covers_closed_position.
    + exact Htrans.
    + apply transition_state_transposition_closed_insert_exact; assumption.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover' :
      positions_cover_transposition (length query) (positions s')
        (std_pos i (S e'))).
    { apply IH; try reflexivity; lia. }
    eapply cover_trans_sub.
    + exact Hcover'.
    + apply subsumes_transposition_std_same_index_le. lia.
Qed.

Lemma transition_state_transposition_covers_match_covered : forall
  query n dict s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c ->
  positions_cover_transposition (length query) (positions s') (std_pos (S i) e).
Proof.
  intros query n dict s c s' i e Hqlen Hall_reach Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth.
  - subst p.
    rewrite <- Hqlen.
    eapply transition_state_transposition_covers_closed_position.
    + exact Htrans.
    + eapply transition_state_transposition_closed_match_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover' :
      positions_cover_transposition (length query) (positions s')
        (std_pos (S i) e')).
    { eapply IH; try reflexivity; eauto. }
    eapply cover_trans_sub.
    + exact Hcover'.
    + apply subsumes_transposition_std_same_index_le. exact Herr_le.
Qed.

Lemma transition_state_transposition_covers_substitute_covered : forall
  query n dict s c c' s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  positions_cover_transposition (length query) (positions s') (std_pos (S i) (S e)).
Proof.
  intros query n dict s c c' s' i e Hqlen Hall_reach Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth Hneq He_lt.
  - subst p.
    rewrite <- Hqlen.
    eapply transition_state_transposition_covers_closed_position.
    + exact Htrans.
    + eapply transition_state_transposition_closed_substitute_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover' :
      positions_cover_transposition (length query) (positions s')
        (std_pos (S i) (S e'))).
    { eapply IH; try reflexivity; try lia; eauto. }
    eapply cover_trans_sub.
    + exact Hcover'.
    + apply subsumes_transposition_std_same_index_le. lia.
Qed.

Lemma transition_state_transposition_covers_enter_covered : forall
  query n dict s c c_next s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (std_pos i e) ->
  S i < length query ->
  nth_error query (S i) = Some c ->
  nth_error query i = Some c_next ->
  e < n ->
  positions_cover_transposition (length query) (positions s') (special_pos i (S e)).
Proof.
  intros query n dict s c c_next s' i e Hqlen Hall_reach Htrans Hcover.
  remember (std_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth_next Hnth_cur He_lt.
  - subst p.
    rewrite <- Hqlen.
    eapply transition_state_transposition_covers_closed_position.
    + exact Htrans.
    + eapply transition_state_transposition_closed_enter_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_std_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover' :
      positions_cover_transposition (length query) (positions s')
        (special_pos i (S e'))).
    { eapply IH; try reflexivity; try lia; eauto. }
    eapply cover_trans_sub.
    + exact Hcover'.
    + apply subsumes_transposition_special_same_index_le. lia.
Qed.

Lemma transition_state_transposition_covers_complete_covered : forall
  query n dict s c s' i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) (special_pos i e) ->
  S i < length query ->
  nth_error query i = Some c ->
  positions_cover_transposition (length query) (positions s')
    (std_pos (S (S i)) e).
Proof.
  intros query n dict s c s' i e Hqlen Hall_reach Htrans Hcover.
  remember (special_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth.
  - subst p.
    rewrite <- Hqlen.
    eapply transition_state_transposition_covers_closed_position.
    + exact Htrans.
    + eapply transition_state_transposition_closed_complete_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_special_inv (length query) q i e Hsub)
      as [e' [Hq_eq Herr_le]].
    subst q.
    assert (Hcover' :
      positions_cover_transposition (length query) (positions s')
        (std_pos (S (S i)) e')).
    { eapply IH; try reflexivity; eauto. }
    eapply cover_trans_sub.
    + exact Hcover'.
    + apply subsumes_transposition_std_same_index_le. exact Herr_le.
Qed.

Lemma transition_state_transposition_succeeds_from_complete_cover : forall
  query n dict s c i e,
  query_length s = length query ->
  (forall p0, In p0 (positions s) ->
              position_reachable_damerau query n dict p0) ->
  positions_cover_transposition (length query) (positions s) (special_pos i e) ->
  S i < length query ->
  nth_error query i = Some c ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros query n dict s c i e Hqlen Hall_reach Hcover.
  remember (special_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hcover as [p Hin | p q Hcover IH Hsub];
    intros i e Hp Hlt Hnth.
  - subst p.
    eapply transition_state_transposition_succeeds_from_closed_member.
    eapply transition_state_transposition_closed_complete_exact; eauto.
  - subst p.
    destruct (subsumes_transposition_special_inv (length query) q i e Hsub)
      as [e' [Hq_eq _]].
    subst q.
    eapply IH; eauto.
Qed.

Lemma transition_state_transposition_succeeds_from_can_reach_covered : forall
  query n dict c remaining s p p_final,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_damerau query n dict q) ->
  positions_cover_transposition (length query) (positions s) p ->
  can_reach query n p (c :: remaining) p_final ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros query n dict c remaining s p p_final Hqlen Hall_reach Hcover Hreach.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s dict Hqlen Hall_reach Hcover.
  induction Hreach; intros c0 remaining0 Hinput s0 dict0 Hqlen Hall_reach Hcover.
  - discriminate.
  - subst p.
    eapply transition_state_transposition_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_transposition_succeeds_from_match_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_transposition_succeeds_from_std_error_cover; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_transposition_succeeds_from_std_error_cover; eauto.
Qed.

Lemma transition_state_transposition_preserves_can_reach_covered : forall
  query n dict c remaining s s' p p_final,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_damerau query n dict q) ->
  state_delete_chain_covered_transposition n s ->
  transition_state Transposition s c query n = Some s' ->
  positions_cover_transposition (length query) (positions s) p ->
  can_reach query n p (c :: remaining) p_final ->
  term_index p_final = length query ->
  num_errors p_final <= n ->
  is_special p_final = false ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' p p_final Hqlen Hall_reach
         Hclosed Htrans Hcover Hreach Hterm_final Hfinal_err Hfinal_spec.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s s' dict Hqlen Hall_reach Hclosed Htrans Hcover.
  induction Hreach; intros c0 remaining0 Hinput s0 s' dict0 Hqlen
         Hall_reach Hclosed Htrans Hcover.
  - discriminate.
  - subst p.
    assert (Hcover_del :
      positions_cover_transposition (length query) (positions s0)
        (std_pos (S i) (S e))).
    { eapply positions_cover_transposition_delete_successor_covered; eauto; lia. }
    eapply IHHreach; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos (S i) e)).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_transposition_preserves_complete_match_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos (S i) (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_transposition_preserves_complete_substitute_covered; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos i (S e))).
    { exists p_final. repeat split; assumption. }
    assert (Hi_bound : i <= length query).
    { pose proof (can_reach_term_index_monotone
                    query n (std_pos i (S e)) remaining0 p_final Hreach) as Hmono.
      simpl in Hmono. lia. }
    eapply transition_state_transposition_preserves_complete_insert_covered; eauto.
Qed.

Lemma transition_state_transposition_preserves_state_has_completable : forall
  query n dict c remaining s s',
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_damerau query n dict q) ->
  state_delete_chain_covered_transposition n s ->
  transition_state Transposition s c query n = Some s' ->
  state_has_completable query n (c :: remaining) s ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' Hqlen Hall_reach Hclosed Htrans
         [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_transposition_preserves_can_reach_covered
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply cover_trans_in. exact Hin.
Qed.

Lemma transition_state_transposition_succeeds_from_state_has_completable : forall
  query n dict c remaining s,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_damerau query n dict q) ->
  state_has_completable query n (c :: remaining) s ->
  exists s', transition_state Transposition s c query n = Some s'.
Proof.
  intros query n dict c remaining s Hqlen Hall_reach [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_transposition_succeeds_from_can_reach_covered
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply cover_trans_in. exact Hin.
Qed.

Lemma transition_state_transposition_delete_chain_covered : forall
  s c query n s' p k,
  transition_state Transposition s c query n = Some s' ->
  In p (positions s') ->
  is_special p = false ->
  term_index p + k <= query_length s ->
  num_errors p + k <= n ->
  positions_cover_transposition (query_length s) (positions s')
    (std_pos (term_index p + k) (num_errors p + k)).
Proof.
  intros s c query n s' p k Htrans Hin Hspec Hterm Herr.
  assert (Htrans_orig := Htrans).
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions :=
    transition_state_positions Transposition (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  assert (Hp_closed : In p closed_positions).
  { apply in_fold_state_insert_origin with
      (init_state := empty_state Transposition (query_length s)).
    - unfold empty_state. reflexivity.
    - exact Hin. }
  assert (Htarget_closed :
    In (std_pos (term_index p + k) (num_errors p + k)) closed_positions).
  { unfold closed_positions.
    apply epsilon_closure_member_reaches_deletes_nonspecial;
      assumption. }
  eapply transition_state_transposition_covers_closed_position.
  - exact Htrans_orig.
  - exact Htarget_closed.
Qed.

Lemma transition_state_transposition_state_delete_chain_covered : forall
  s c query n s',
  transition_state Transposition s c query n = Some s' ->
  state_delete_chain_covered_transposition n s'.
Proof.
  intros s c query n s' Htrans p k Hin Hspec Hterm Herr.
  assert (Hqlen : query_length s' = query_length s).
  { eapply transition_state_preserves_query_length. exact Htrans. }
  rewrite Hqlen in Hterm.
  rewrite Hqlen.
  eapply transition_state_transposition_delete_chain_covered; eauto.
Qed.

Lemma initial_closed_state_delete_chain_covered_transposition : forall n qlen,
  state_delete_chain_covered_transposition n
    (mkState (epsilon_closure [initial_position] n qlen)
             Transposition qlen).
Proof.
  intros n qlen p k Hin Hspec Hterm Herr.
  simpl in *.
  apply cover_trans_in.
  apply epsilon_closure_member_reaches_deletes_nonspecial; assumption.
Qed.

Lemma state_has_completable_empty_accepts_transposition : forall query n s,
  query_length s = length query ->
  state_delete_chain_covered_transposition n s ->
  state_has_completable query n [] s ->
  state_is_final s = true.
Proof.
  intros query n s Hqlen Hclosed
         [p [Hp_in [p_final [Hreach [Hterm_final [Herr_final Hspec_final]]]]]].
  pose proof (can_reach_source_not_special query n p [] p_final Hreach Hspec_final)
    as Hsource_spec.
  pose proof (can_reach_term_index_monotone query n p [] p_final Hreach)
    as Hterm_mono.
  pose proof (can_reach_empty_remaining_errors query n p p_final Hreach)
    as Herr_exact.
  set (k := term_index p_final - term_index p).
  assert (Hk_term : term_index p + k <= query_length s).
  { unfold k. rewrite Hqlen, Hterm_final. lia. }
  assert (Hk_err : num_errors p + k <= n).
  { unfold k. rewrite Herr_exact in Herr_final. exact Herr_final. }
  pose proof (Hclosed p k Hp_in Hsource_spec Hk_term Hk_err) as Hcover.
  assert (Htarget_eq :
    std_pos (term_index p + k) (num_errors p + k) = p_final).
  { destruct p_final as [i_f e_f sp_f].
    simpl in Hterm_final, Herr_final, Hspec_final, Hterm_mono, Herr_exact.
    destruct sp_f; simpl in Hspec_final; try discriminate.
    unfold k, std_pos. simpl.
    f_equal; lia. }
  rewrite Hqlen in Hcover.
  eapply covered_final_state_accepts_transposition.
  - exact Hqlen.
  - rewrite Htarget_eq in Hcover. exact Hcover.
  - exact Hterm_final.
Qed.

Lemma automaton_run_transposition_delete_chain_covered_from_state : forall
  query n dict s final,
  state_delete_chain_covered_transposition n s ->
  automaton_run Transposition query n dict s = Some final ->
  state_delete_chain_covered_transposition n final.
Proof.
  induction dict as [|c rest IH]; intros s final Hclosed Hrun.
  - simpl in Hrun. injection Hrun as Hfinal. subst final. exact Hclosed.
  - simpl in Hrun.
    destruct (transition_state Transposition s c query n) as [s_mid|] eqn:Htrans;
      [| discriminate].
    apply (IH s_mid final).
    + eapply transition_state_transposition_state_delete_chain_covered.
      exact Htrans.
    + exact Hrun.
Qed.

Lemma initial_closed_has_completable_from_lev_bound_transposition :
  forall query dict n,
  lev_distance query dict <= n ->
  state_has_completable query n dict
    (mkState (epsilon_closure [initial_position] n (length query))
             Transposition (length query)).
Proof.
  intros query dict n Hdist.
  destruct (optimal_sequence_exists query dict) as [ops [Hvalid Hcost]].
  exists initial_position.
  split.
  - simpl. apply epsilon_closure_includes_input. simpl. left. reflexivity.
  - apply valid_sequence_can_complete_initial with (ops := ops).
    + exact Hvalid.
    + lia.
Qed.

Lemma automaton_run_transposition_preserves_completable_from_state : forall
  query n remaining dict_prefix s,
  query_length s = length query ->
  (forall q, In q (positions s) ->
              position_reachable_damerau query n dict_prefix q) ->
  state_delete_chain_covered_transposition n s ->
  state_has_completable query n remaining s ->
  exists final,
    automaton_run Transposition query n remaining s = Some final /\
    state_has_completable query n [] final.
Proof.
  intros query n remaining.
  induction remaining as [|c rest IH]; intros dict_prefix s Hqlen Hall_reach
         Hclosed Hcomplete.
  - exists s. split; [reflexivity | exact Hcomplete].
  - destruct (transition_state_transposition_succeeds_from_state_has_completable
                query n dict_prefix c rest s Hqlen Hall_reach Hcomplete)
      as [s_mid Htrans].
    assert (Hrun_one : automaton_run Transposition query n [c] s = Some s_mid).
    { simpl. rewrite Htrans. reflexivity. }
    assert (Hall_reach_mid : forall q, In q (positions s_mid) ->
      position_reachable_damerau query n (dict_prefix ++ [c]) q).
    { intros q Hq.
      eapply automaton_run_preserves_reachable_damerau_transposition.
      - exact Hqlen.
      - exact Hrun_one.
      - exact Hall_reach.
      - exact Hq. }
    assert (Hcomplete_mid : state_has_completable query n rest s_mid).
    { eapply transition_state_transposition_preserves_state_has_completable; eauto. }
    assert (Hqlen_mid : query_length s_mid = length query).
    { rewrite (transition_state_preserves_query_length Transposition s c query n s_mid Htrans).
      exact Hqlen. }
    assert (Hclosed_mid : state_delete_chain_covered_transposition n s_mid).
    { eapply transition_state_transposition_state_delete_chain_covered.
      exact Htrans. }
    destruct (IH (dict_prefix ++ [c]) s_mid Hqlen_mid Hall_reach_mid
                Hclosed_mid Hcomplete_mid) as [final [Hrun_rest Hcomplete_final]].
    exists final. split.
    + simpl. rewrite Htrans. exact Hrun_rest.
    + exact Hcomplete_final.
Qed.

Lemma automaton_run_transposition_completable_from_lev_bound : forall query dict n,
  lev_distance query dict <= n ->
  exists final,
    automaton_run_from_initial Transposition query n dict = Some final /\
    state_has_completable query n [] final.
Proof.
  intros query dict n Hdist.
  unfold automaton_run_from_initial.
  set (init_closed :=
    mkState (epsilon_closure (positions (initial_state Transposition (length query)))
                         n (length query)) Transposition (length query)).
  assert (Hinit_eq :
    init_closed =
    mkState (epsilon_closure [initial_position] n (length query))
            Transposition (length query)).
  { unfold init_closed, initial_state. reflexivity. }
  rewrite Hinit_eq.
  apply automaton_run_transposition_preserves_completable_from_state
    with (dict_prefix := []).
  - reflexivity.
  - intros p Hp.
    eapply epsilon_closure_preserves_reachable_damerau
      with (positions := [initial_position]).
    + intros p0 Hp0.
      simpl in Hp0. destruct Hp0 as [Hp0 | []]. subst p0.
      apply reach_damerau_initial.
    + exact Hp.
  - apply initial_closed_state_delete_chain_covered_transposition.
  - apply initial_closed_has_completable_from_lev_bound_transposition.
    exact Hdist.
Qed.

Lemma automaton_run_from_initial_transposition_delete_chain_covered : forall
  query n dict final,
  automaton_run_from_initial Transposition query n dict = Some final ->
  state_delete_chain_covered_transposition n final.
Proof.
  intros query n dict final Hrun.
  unfold automaton_run_from_initial in Hrun.
  apply (automaton_run_transposition_delete_chain_covered_from_state
           query n dict
           (mkState (epsilon_closure [initial_position] n (length query))
                    Transposition (length query))
           final).
  - apply initial_closed_state_delete_chain_covered_transposition.
  - exact Hrun.
Qed.

Lemma automaton_run_transposition_final_completable_accepts : forall query n dict final,
  automaton_run_from_initial Transposition query n dict = Some final ->
  state_has_completable query n [] final ->
  state_is_final final = true.
Proof.
  intros query n dict final Hrun Hcomplete.
  assert (Hqlen : query_length final = length query).
  { unfold automaton_run_from_initial in Hrun.
    rewrite (automaton_run_preserves_query_length Transposition query n dict
               (mkState (epsilon_closure [initial_position] n (length query))
                        Transposition (length query))
               final Hrun).
    reflexivity. }
  apply state_has_completable_empty_accepts_transposition with
    (query := query) (n := n).
  - exact Hqlen.
  - eapply automaton_run_from_initial_transposition_delete_chain_covered.
    exact Hrun.
  - exact Hcomplete.
Qed.

Lemma automaton_run_from_initial_snoc : forall alg query n dict c s s',
  automaton_run_from_initial alg query n dict = Some s ->
  transition_state alg s c query n = Some s' ->
  automaton_run_from_initial alg query n (dict ++ [c]) = Some s'.
Proof.
  intros alg query n dict c s s' Hrun Htrans.
  assert (Hsnoc : forall init mid,
    automaton_run alg query n dict init = Some mid ->
    transition_state alg mid c query n = Some s' ->
    automaton_run alg query n (dict ++ [c]) init = Some s').
  { clear s Hrun Htrans.
    induction dict as [|d rest IH]; intros init mid Hrun_mid Htrans_mid.
    - simpl in Hrun_mid. injection Hrun_mid as Hmid. subst mid.
      simpl. rewrite Htrans_mid. reflexivity.
    - simpl in Hrun_mid |- *.
      destruct (transition_state alg init d query n) as [next|] eqn:Hstep.
	      + exact (IH next mid Hrun_mid Htrans_mid).
	      + discriminate. }
  unfold automaton_run_from_initial in *.
  eapply Hsnoc; eauto.
Qed.

Lemma automaton_run_from_initial_query_length : forall alg query n dict final,
  automaton_run_from_initial alg query n dict = Some final ->
  query_length final = length query.
Proof.
  intros alg query n dict final Hrun.
  unfold automaton_run_from_initial in Hrun.
  rewrite (automaton_run_preserves_query_length alg query n dict
             (mkState
                (epsilon_closure (positions (initial_state alg (length query)))
                   n (length query)) alg (length query))
             final Hrun).
  reflexivity.
Qed.

Lemma automaton_run_from_initial_transposition_positions_reachable : forall
  query n dict final,
  automaton_run_from_initial Transposition query n dict = Some final ->
  forall p, In p (positions final) ->
    position_reachable_damerau query n dict p.
Proof.
  intros query n dict final Hrun p Hin.
  unfold automaton_run_from_initial, initial_state in Hrun.
  simpl in Hrun.
  assert (Hinit : forall p0,
    In p0 (positions
      (mkState (epsilon_closure [initial_position] n (length query))
               Transposition (length query))) ->
    position_reachable_damerau query n [] p0).
  { intros p0 Hp0. simpl in Hp0.
    apply initial_epsilon_reachable_damerau. exact Hp0. }
  pose proof
    (automaton_run_preserves_reachable_damerau_transposition
       query n [] dict
       (mkState (epsilon_closure [initial_position] n (length query))
                Transposition (length query))
       final eq_refl Hrun Hinit p Hin) as Hreachable.
  simpl in Hreachable.
  exact Hreachable.
Qed.

Lemma automaton_run_from_initial_transposition_covers_reachable_damerau :
  forall query n dict p,
  position_reachable_damerau query n dict p ->
  exists final,
    automaton_run_from_initial Transposition query n dict = Some final /\
    positions_cover_transposition (length query) (positions final) p.
Proof.
  intros query n dict p Hreach.
  induction Hreach.
  - exists (mkState (epsilon_closure [initial_position] n (length query))
              Transposition (length query)).
    split.
    + unfold automaton_run_from_initial, initial_state. simpl. reflexivity.
    + simpl. apply cover_trans_in.
      apply epsilon_closure_includes_input. simpl. left. reflexivity.
  - destruct IHHreach as [final [Hrun Hcover]].
    exists final. split; [exact Hrun|].
    assert (Hqlen : query_length final = length query).
    { eapply automaton_run_from_initial_query_length. exact Hrun. }
    assert (Hclosed : state_delete_chain_covered_transposition n final).
    { eapply automaton_run_from_initial_transposition_delete_chain_covered.
      exact Hrun. }
    eapply positions_cover_transposition_delete_successor_covered; eauto; lia.
  - destruct IHHreach as [s [Hrun Hcover]].
    assert (Hqlen : query_length s = length query).
    { eapply automaton_run_from_initial_query_length. exact Hrun. }
    assert (Hall_reach : forall q, In q (positions s) ->
      position_reachable_damerau query n dp q).
    { eapply automaton_run_from_initial_transposition_positions_reachable.
      exact Hrun. }
    destruct (transition_state_transposition_succeeds_from_match_cover
                query n dp s c i e Hqlen Hall_reach Hcover)
      as [s_next Htrans]; eauto.
    exists s_next. split.
    + eapply automaton_run_from_initial_snoc; eauto.
    + eapply transition_state_transposition_covers_match_covered; eauto.
  - destruct IHHreach as [s [Hrun Hcover]].
    assert (Hqlen : query_length s = length query).
    { eapply automaton_run_from_initial_query_length. exact Hrun. }
    assert (Hall_reach : forall q, In q (positions s) ->
      position_reachable_damerau query n dp q).
    { eapply automaton_run_from_initial_transposition_positions_reachable.
      exact Hrun. }
    destruct (transition_state_transposition_succeeds_from_std_error_cover
                query n s c i e Hcover)
      as [s_next Htrans]; eauto.
    exists s_next. split.
    + eapply automaton_run_from_initial_snoc; eauto.
    + eapply transition_state_transposition_covers_substitute_covered; eauto.
  - destruct IHHreach as [s [Hrun Hcover]].
    assert (Hqlen : query_length s = length query).
    { eapply automaton_run_from_initial_query_length. exact Hrun. }
    destruct (transition_state_transposition_succeeds_from_std_error_cover
                query n s c i e Hcover)
      as [s_next Htrans]; eauto.
    exists s_next. split.
    + eapply automaton_run_from_initial_snoc; eauto.
    + eapply transition_state_transposition_covers_insert_covered; eauto.
  - destruct IHHreach as [s [Hrun Hcover]].
    assert (Hqlen : query_length s = length query).
    { eapply automaton_run_from_initial_query_length. exact Hrun. }
    assert (Hall_reach : forall q, In q (positions s) ->
      position_reachable_damerau query n dp q).
    { eapply automaton_run_from_initial_transposition_positions_reachable.
      exact Hrun. }
    destruct (transition_state_transposition_succeeds_from_std_error_cover
                query n s c i e Hcover)
      as [s_next Htrans]; eauto.
    exists s_next. split.
    + eapply automaton_run_from_initial_snoc; eauto.
    + eapply transition_state_transposition_covers_enter_covered; eauto.
	  - destruct IHHreach as [s [Hrun Hcover]].
	    assert (Hqlen : query_length s = length query).
	    { eapply automaton_run_from_initial_query_length. exact Hrun. }
	    assert (Hall_reach : forall q, In q (positions s) ->
	      position_reachable_damerau query n dp q).
	    { eapply automaton_run_from_initial_transposition_positions_reachable.
	      exact Hrun. }
	    assert (Hsi : S i < length query).
	    { inversion Hreach; subst; simpl in *; try discriminate; assumption. }
	    assert (Hci : nth_error query i = Some c).
	    { match goal with
	      | H : nth_error query i = Some c |- _ => exact H
	      end. }
	    destruct (transition_state_transposition_succeeds_from_complete_cover
	                query n dp s c i e Hqlen Hall_reach Hcover Hsi Hci)
	      as [s_next Htrans].
	    exists s_next. split.
    + eapply automaton_run_from_initial_snoc; eauto.
    + eapply transition_state_transposition_covers_complete_covered; eauto.
Qed.

Lemma reachable_damerau_final_implies_accepts_transposition : forall query dict n p,
  position_reachable_damerau query n dict p ->
  term_index p = length query ->
  is_special p = false ->
  num_errors p <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros query dict n p Hreach Hterm _ _.
  destruct (automaton_run_from_initial_transposition_covers_reachable_damerau
              query n dict p Hreach)
    as [final [Hrun Hcover]].
  unfold automaton_accepts.
  rewrite Hrun.
  eapply covered_final_state_accepts_transposition.
  - eapply automaton_run_from_initial_query_length. exact Hrun.
  - exact Hcover.
  - exact Hterm.
Qed.

Lemma can_complete_match_skip_candidate : forall
  query n remaining i e pi pe j,
  pi <= i ->
  pe + (i - pi) <= e ->
  i < length query ->
  e <= n ->
  j <= i - pi ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  can_complete_to_final query n remaining (std_pos (S (pi + j)) (pe + j)).
Proof.
  intros query n remaining i e pi pe j Hbehind Hcatch Herr_term Herr_err
         Hj Hcomplete.
  set (e_catch := pe + (i - pi)).
  assert (Hcomplete_catch :
    can_complete_to_final query n remaining (std_pos (S i) e_catch)).
  { apply can_complete_same_index_lower_errors with (e := e).
    - simpl. lia.
    - unfold e_catch. exact Hcatch.
    - exact Hcomplete. }
  pose (k := i - (pi + j)).
  assert (Hterm : S (pi + j) + k <= length query).
  { unfold k. lia. }
  assert (Herror : pe + j + k <= n).
  { unfold k. unfold e_catch in Hcomplete_catch |- *. lia. }
  replace (std_pos (S i) e_catch)
    with (std_pos (S (pi + j) + k) (pe + j + k)) in Hcomplete_catch.
  - apply can_complete_prepend_deletes with (k := k).
    + exact Hterm.
    + exact Herror.
    + exact Hcomplete_catch.
  - unfold k, e_catch, std_pos. simpl. f_equal; lia.
Qed.

Lemma can_complete_substitute_behind_match_candidate : forall
  query n remaining i e pi pe,
  pi <= i ->
  pe + (i - pi) <= e ->
  i < length query ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  can_complete_to_final query n remaining (std_pos (S pi) pe).
Proof.
  intros query n remaining i e pi pe Hbehind Hcatch Hterm Herr Hcomplete.
  set (d := i - pi).
  assert (Hcomplete_catch :
    can_complete_to_final query n remaining (std_pos (S i) (pe + d))).
  { apply can_complete_same_index_lower_errors with (e := S e).
    - simpl. lia.
    - unfold d. lia.
    - exact Hcomplete. }
  replace (std_pos (S i) (pe + d))
    with (std_pos (S pi + d) (pe + d)) in Hcomplete_catch
    by (unfold d, std_pos; simpl; f_equal; lia).
  apply can_complete_prepend_deletes with (k := d).
  - unfold d. lia.
  - unfold d. lia.
  - exact Hcomplete_catch.
Qed.

Lemma can_complete_substitute_behind_substitute_candidate : forall
  query n remaining i e pi pe,
  pi <= i ->
  pe + (i - pi) <= e ->
  i < length query ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  can_complete_to_final query n remaining (std_pos (S pi) (S pe)).
Proof.
  intros query n remaining i e pi pe Hbehind Hcatch Hterm Herr Hcomplete.
  set (d := i - pi).
  assert (Hcomplete_catch :
    can_complete_to_final query n remaining (std_pos (S i) (S (pe + d)))).
  { apply can_complete_same_index_lower_errors with (e := S e).
    - simpl. lia.
    - unfold d. lia.
    - exact Hcomplete. }
  replace (std_pos (S i) (S (pe + d)))
    with (std_pos (S pi + d) (S pe + d)) in Hcomplete_catch
    by (unfold d, std_pos; simpl; f_equal; lia).
  apply can_complete_prepend_deletes with (k := d).
  - unfold d. lia.
  - unfold d. lia.
  - exact Hcomplete_catch.
Qed.

(** Helper: antichain_insert preserves can-complete property.

    If the current antichain has a completable position, or the new position
    is completable, then the result has a completable position.

    Preconditions on term_index bounds are needed because subsumption_preserves_can_complete
    requires term_index p' <= length query for the subsuming position. The
    non-specialness of an existing subsuming witness comes from the existing
    antichain invariant, not from Standard subsumption itself, which does not
    inspect [is_special].
*)
Lemma antichain_insert_preserves_can_complete : forall
  query n remaining p pos_list,
  is_special p = false ->
  term_index p <= length query ->
  (forall q, In q pos_list -> term_index q <= length query) ->
  (forall q, In q pos_list -> is_special q = false) ->
  (can_complete_to_final query n remaining p \/
   exists q, In q pos_list /\ is_special q = false /\ can_complete_to_final query n remaining q) ->
  exists p', In p' (antichain_insert Standard (length query) p pos_list) /\
             is_special p' = false /\
             can_complete_to_final query n remaining p'.
Proof.
  intros query n remaining p pos_list Hspec_p Hp_qlen Hpos_list_qlen Hpos_list_nonspec Hcomplete.
  unfold antichain_insert.
  destruct (subsumed_by_any Standard (length query) p pos_list) eqn:Hsub.
  - (* p is subsumed - pos_list unchanged *)
    destruct Hcomplete as [Hp_complete | [q [Hq_in [Hq_spec Hq_complete]]]].
      + (* p is completable and subsumed - find what subsumes it *)
        apply subsumed_by_any_correct in Hsub as [p' [Hp'_in Hp'_sub]].
      (* The subsuming position can also complete to final via subsumption_preserves_can_complete *)
      exists p'. split; [exact Hp'_in |].
      (* Need to show p' is non-special and can complete *)
      (* For now, use the fact that p' subsumes completable p *)
      split.
      * (* p' is non-special by the existing antichain invariant. *)
        apply Hpos_list_nonspec. exact Hp'_in.
      * (* p' can complete since it subsumes p which can complete *)
        apply (subsumption_preserves_can_complete query n remaining p p'
               Hp_complete Hp'_sub Hspec_p
               (Hpos_list_nonspec p' Hp'_in)
               (Hpos_list_qlen p' Hp'_in)).
    + (* Some q in pos_list is completable - it survives *)
      exists q. split; [exact Hq_in | split; [exact Hq_spec | exact Hq_complete]].
  - (* p is not subsumed - p is inserted, some positions may be removed *)
    destruct Hcomplete as [Hp_complete | [q [Hq_in [Hq_spec Hq_complete]]]].
    + (* p is completable - p is in the result *)
      exists p.
      split.
      * (* p is in the result: either directly or via remove_subsumed *)
        unfold remove_subsumed.
        (* p :: filter (...) positions, so p is definitely in *)
        left. reflexivity.
      * split; [exact Hspec_p | exact Hp_complete].
    + (* q is completable - either q survives or p subsumes q *)
      destruct (subsumes Standard (length query) p q) eqn:Hp_sub_q.
      * (* p subsumes q - p can complete (by subsumption_preserves_can_complete) *)
        exists p.
        split; [left; reflexivity |].
        split; [exact Hspec_p |].
        (* p subsumes q and q can complete -> p can complete *)
        apply (subsumption_preserves_can_complete query n remaining q p Hq_complete Hp_sub_q Hq_spec Hspec_p Hp_qlen).
      * (* p does not subsume q - q survives in remove_subsumed *)
        exists q.
        split.
        -- right.
           apply in_remove_subsumed_if_not_subsumed; [exact Hq_in | exact Hp_sub_q].
        -- split; [exact Hq_spec | exact Hq_complete].
Qed.

(** Lifting the antichain can-complete invariant through [state_insert].
    This packages the sorted-list wrapper around [antichain_insert], so later
    transition proofs can reason at the state level. *)
Lemma state_insert_preserves_can_complete_standard : forall
  query n remaining p s,
  algorithm s = Standard ->
  query_length s = length query ->
  is_special p = false ->
  term_index p <= length query ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (can_complete_to_final query n remaining p \/
   exists q, In q (positions s) /\ is_special q = false /\
             can_complete_to_final query n remaining q) ->
  exists q, In q (positions (state_insert p s)) /\
            is_special q = false /\
            can_complete_to_final query n remaining q.
Proof.
  intros query n remaining p s Halg Hqlen Hp_spec Hp_bound
         Hstate_bound Hstate_spec Hcomplete.
  destruct (antichain_insert_preserves_can_complete
              query n remaining p (positions s)
              Hp_spec Hp_bound Hstate_bound Hstate_spec Hcomplete)
    as [q [Hq_in [Hq_spec Hq_complete]]].
  exists q. split.
  - rewrite positions_state_insert.
    rewrite Halg, Hqlen.
    apply fold_right_sorted_insert_preserves_In.
    exact Hq_in.
  - split; assumption.
Qed.

Lemma state_insert_preserves_state_has_completable_standard : forall
  query n remaining p s,
  algorithm s = Standard ->
  query_length s = length query ->
  is_special p = false ->
  term_index p <= length query ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  state_has_completable query n remaining s ->
  state_has_completable query n remaining (state_insert p s).
Proof.
  intros query n remaining p s Halg Hqlen Hp_spec Hp_bound
         Hstate_bound Hstate_spec [q [Hq_in Hq_complete]].
  unfold state_has_completable.
  destruct (state_insert_preserves_can_complete_standard
              query n remaining p s Halg Hqlen Hp_spec Hp_bound
              Hstate_bound Hstate_spec) as [r [Hr_in [_ Hr_complete]]].
  - right.
    exists q. repeat split.
    + exact Hq_in.
    + apply Hstate_spec. exact Hq_in.
    + exact Hq_complete.
  - exists r. split; assumption.
Qed.

Lemma state_insert_adds_can_complete_standard : forall
  query n remaining p s,
  algorithm s = Standard ->
  query_length s = length query ->
  is_special p = false ->
  term_index p <= length query ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  can_complete_to_final query n remaining p ->
  state_has_completable query n remaining (state_insert p s).
Proof.
  intros query n remaining p s Halg Hqlen Hp_spec Hp_bound
         Hstate_bound Hstate_spec Hp_complete.
  unfold state_has_completable.
  destruct (state_insert_preserves_can_complete_standard
              query n remaining p s Halg Hqlen Hp_spec Hp_bound
              Hstate_bound Hstate_spec) as [q [Hq_in [_ Hq_complete]]].
  - left. exact Hp_complete.
  - exists q. split; assumption.
Qed.

Lemma fold_state_insert_preserves_state_has_completable_standard : forall
  query n remaining inserts s,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall p, In p inserts -> is_special p = false) ->
  (forall p, In p inserts -> term_index p <= length query) ->
  (forall p, In p (positions s) -> is_special p = false) ->
  (forall p, In p (positions s) -> term_index p <= length query) ->
  state_has_completable query n remaining s ->
  state_has_completable query n remaining
    (fold_left (fun s0 p => state_insert p s0) inserts s).
Proof.
  induction inserts as [|p rest IH]; intros s Halg Hqlen Hinserts_spec
         Hinserts_bound Hstate_spec Hstate_bound Hcomplete.
  - simpl. exact Hcomplete.
  - simpl.
    apply IH.
    + rewrite algorithm_state_insert. exact Halg.
    + rewrite query_length_state_insert. exact Hqlen.
    + intros q Hq. apply Hinserts_spec. simpl. right. exact Hq.
    + intros q Hq. apply Hinserts_bound. simpl. right. exact Hq.
    + intros q Hq.
      apply in_state_insert_origin in Hq.
      destruct Hq as [Heq | Hq_old].
      * subst q. apply Hinserts_spec. simpl. left. reflexivity.
      * apply Hstate_spec. exact Hq_old.
    + intros q Hq.
      apply in_state_insert_origin in Hq.
      destruct Hq as [Heq | Hq_old].
      * subst q. apply Hinserts_bound. simpl. left. reflexivity.
      * apply Hstate_bound. exact Hq_old.
    + apply state_insert_preserves_state_has_completable_standard.
      * exact Halg.
      * exact Hqlen.
      * apply Hinserts_spec. simpl. left. reflexivity.
      * apply Hinserts_bound. simpl. left. reflexivity.
      * exact Hstate_bound.
      * exact Hstate_spec.
      * exact Hcomplete.
Qed.

Lemma fold_state_insert_has_can_complete_member_standard : forall
  query n remaining inserts s p,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall q, In q inserts -> is_special q = false) ->
  (forall q, In q inserts -> term_index q <= length query) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  In p inserts ->
  can_complete_to_final query n remaining p ->
  state_has_completable query n remaining
    (fold_left (fun s0 q => state_insert q s0) inserts s).
Proof.
  induction inserts as [|q rest IH]; intros s p Halg Hqlen Hinserts_spec
         Hinserts_bound Hstate_spec Hstate_bound Hp_in Hp_complete.
  - inversion Hp_in.
  - simpl in Hp_in |- *.
    destruct Hp_in as [Hp_eq | Hp_in_rest].
    + subst q.
      apply fold_state_insert_preserves_state_has_completable_standard.
      * rewrite algorithm_state_insert. exact Halg.
      * rewrite query_length_state_insert. exact Hqlen.
      * intros r Hr. apply Hinserts_spec. simpl. right. exact Hr.
      * intros r Hr. apply Hinserts_bound. simpl. right. exact Hr.
      * intros r Hr.
        apply in_state_insert_origin in Hr.
        destruct Hr as [Heq | Hr_old].
        -- subst r. apply Hinserts_spec. simpl. left. reflexivity.
        -- apply Hstate_spec. exact Hr_old.
      * intros r Hr.
        apply in_state_insert_origin in Hr.
        destruct Hr as [Heq | Hr_old].
        -- subst r. apply Hinserts_bound. simpl. left. reflexivity.
        -- apply Hstate_bound. exact Hr_old.
      * apply state_insert_adds_can_complete_standard.
        -- exact Halg.
        -- exact Hqlen.
        -- apply Hinserts_spec. simpl. left. reflexivity.
        -- apply Hinserts_bound. simpl. left. reflexivity.
        -- exact Hstate_bound.
        -- exact Hstate_spec.
        -- exact Hp_complete.
    + apply (IH (state_insert q s) p).
      * rewrite algorithm_state_insert. exact Halg.
      * rewrite query_length_state_insert. exact Hqlen.
      * intros r Hr. apply Hinserts_spec. simpl. right. exact Hr.
      * intros r Hr. apply Hinserts_bound. simpl. right. exact Hr.
      * intros r Hr.
        apply in_state_insert_origin in Hr.
        destruct Hr as [Heq | Hr_old].
        -- subst r. apply Hinserts_spec. simpl. left. reflexivity.
        -- apply Hstate_spec. exact Hr_old.
      * intros r Hr.
        apply in_state_insert_origin in Hr.
        destruct Hr as [Heq | Hr_old].
        -- subst r. apply Hinserts_bound. simpl. left. reflexivity.
        -- apply Hstate_bound. exact Hr_old.
      * exact Hp_in_rest.
      * exact Hp_complete.
Qed.

(** If a Standard transition succeeds and one of its epsilon-closed generated
    positions can complete on the remaining suffix, the folded output state
    keeps some completable representative. *)
Lemma transition_state_standard_has_can_complete_closed_member : forall
  query n remaining s c s' p,
  query_length s = length query ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s)) ->
  can_complete_to_final query n remaining p ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' p Hqlen Hstate_bound Htrans Hin Hp_complete.
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_positions := transition_state_positions Standard (positions s) cv min_i n (query_length s)) in *.
  set (closed_positions := epsilon_closure trans_positions n (query_length s)) in *.
  fold min_i in Hin.
  fold cv in Hin.
  fold trans_positions in Hin.
  fold closed_positions in Hin.
  destruct (is_nil closed_positions) eqn:Hnil; [discriminate|].
  injection Htrans as Hs'. subst s'.
  apply (fold_state_insert_has_can_complete_member_standard
           query n remaining closed_positions
           (empty_state Standard (query_length s)) p).
  - unfold empty_state. reflexivity.
  - unfold empty_state. simpl. exact Hqlen.
  - intros q Hq.
    unfold closed_positions in Hq.
    eapply epsilon_closure_nonspecial; [| exact Hq].
    intros q0 Hq0.
    unfold trans_positions in Hq0.
    eapply transition_state_positions_standard_nonspecial. exact Hq0.
  - intros q Hq.
    rewrite <- Hqlen.
    unfold closed_positions in Hq.
    eapply epsilon_closure_term_bounded; [| exact Hq].
    intros q0 Hq0.
    unfold trans_positions in Hq0.
	    eapply transition_state_positions_standard_term_bounded.
	    + intros q1 Hq1. rewrite Hqlen. apply Hstate_bound. exact Hq1.
	    + intros q1 Hq1. unfold min_i. apply min_i_le_term_index. exact Hq1.
	    + intros j Hcvj.
	      unfold cv in Hcvj.
	      pose proof (cv_at_true_in_bounds _ _ Hcvj) as Hj_bound.
	      rewrite (char_vector_length c query min_i (2 * n + 6)) in Hj_bound.
	      pose proof (cv_match_nth_error c query min_i (2 * n + 6) j Hj_bound Hcvj)
	        as Hnth.
	      rewrite Hqlen.
	      apply (proj1 (@nth_error_Some Char query (min_i + j))).
	      rewrite Hnth. discriminate.
	    + exact Hq0.
  - intros q Hq. unfold empty_state in Hq. simpl in Hq. contradiction.
  - intros q Hq. unfold empty_state in Hq. simpl in Hq. contradiction.
  - exact Hin.
  - exact Hp_complete.
Qed.

(** Insert-step preservation for represented predecessors. This is the
    can-complete version of [transition_state_standard_represents_insert_represented]:
    if an antichain member represents the exact insert predecessor, the
    representative's own insert successor is generated, can complete by Standard
    subsumption, and survives folding as a completable state witness. *)
Lemma transition_state_standard_preserves_can_complete_insert_represented : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  positions_subsume Standard (length query) (positions s) (std_pos i e) ->
  e < n ->
  can_complete_to_final query n remaining (std_pos i (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hall_spec Hstate_bound
         Htrans [p' [Hin' Hsub']] He_lt Hcomplete.
  assert (Hspec' : is_special p' = false).
  { apply Hall_spec. exact Hin'. }
  assert (Hbound' : term_index p' <= length query).
  { apply Hstate_bound. exact Hin'. }
  assert (Herr' : num_errors p' < n).
  { pose proof (subsumes_standard_errors (length query) p' (std_pos i e)
                 Hsub') as Herr_le.
    simpl in Herr_le. lia. }
  assert (Hp'_std : p' = std_pos (term_index p') (num_errors p')).
  { destruct p' as [j e' sp]. simpl in Hspec'. subst sp.
    unfold std_pos. simpl. reflexivity. }
  pose (p_ins := std_pos (term_index p') (S (num_errors p'))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp'_std in Hin'.
    apply transition_state_standard_closed_insert_exact; assumption. }
  assert (Hcomplete_ins : can_complete_to_final query n remaining p_ins).
  { unfold p_ins.
    apply (subsumption_preserves_can_complete query n remaining
             (std_pos i (S e))
             (std_pos (term_index p') (S (num_errors p')))).
    - exact Hcomplete.
    - apply subsumes_standard_insert_successor.
      exact Hsub'.
    - reflexivity.
    - reflexivity.
    - exact Hbound'. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_ins.
Qed.

Lemma transition_state_standard_preserves_can_complete_insert_exact : forall
  query n remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  e < n ->
  can_complete_to_final query n remaining (std_pos i (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e Hqlen Hstate_bound Htrans Hin He_lt Hcomplete.
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - apply transition_state_standard_closed_insert_exact.
    + exact Hin.
    + exact He_lt.
  - exact Hcomplete.
Qed.

Lemma transition_state_standard_preserves_can_complete_match_exact : forall
  query n dict remaining s c s' i e,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e Hqlen Hall_reach Hall_spec
         Hstate_bound Htrans Hin Hlt Hnth Hcomplete.
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - eapply transition_state_standard_closed_match_exact; eauto.
  - exact Hcomplete.
Qed.

Lemma transition_state_standard_preserves_can_complete_substitute_exact : forall
  query n dict remaining s c c' s' i e,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In (std_pos i e) (positions s) ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e Hqlen Hall_reach Hall_spec
         Hstate_bound Htrans Hin Hlt Hnth Hneq He_lt Hcomplete.
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - eapply transition_state_standard_closed_substitute_exact; eauto.
  - exact Hcomplete.
Qed.

(** Same-index represented match preservation.  This is the represented
    analogue of the exact match lemma for the case where the surviving
    antichain representative has the same term index as the exact predecessor. *)
Lemma transition_state_standard_preserves_can_complete_match_represented_same_index : forall
  query n dict remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep = i ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hidx Hlt Hnth
         Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hin_std : In (std_pos i (num_errors p_rep)) (positions s)).
  { rewrite <- Hidx. rewrite <- Hp_rep_std. exact Hin_rep. }
  assert (Hclosed : In (std_pos (S i) (num_errors p_rep))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { eapply transition_state_standard_closed_match_exact; eauto. }
  assert (Hsub_succ :
    subsumes_standard (length query)
      (std_pos (S i) (num_errors p_rep)) (std_pos (S i) e) = true).
  { change (subsumes Standard (length query)
      (std_pos (S i) (num_errors p_rep)) (std_pos (S i) e) = true).
    eapply subsumes_standard_match_successor_same_index; eauto. }
  assert (Hcomplete_rep :
    can_complete_to_final query n remaining (std_pos (S i) (num_errors p_rep))).
  { eapply subsumption_preserves_can_complete.
    - exact Hcomplete.
    - exact Hsub_succ.
    - reflexivity.
    - reflexivity.
    - simpl. lia. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_rep.
Qed.

(** Same-index represented substitution preservation.  The representative may
    have fewer errors than the exact predecessor; after both successors pay the
    substitution error, the representative successor still subsumes the exact
    successor and can complete. *)
Lemma transition_state_standard_preserves_can_complete_substitute_represented_same_index : forall
  query n dict remaining s c c' s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep = i ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hidx Hlt Hnth
         Hneq He_lt Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hin_std : In (std_pos i (num_errors p_rep)) (positions s)).
  { rewrite <- Hidx. rewrite <- Hp_rep_std. exact Hin_rep. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_errors (length query) p_rep (std_pos i e)
                 Hsub_rep) as Herr_le.
    simpl in Herr_le. lia. }
  assert (Hclosed : In (std_pos (S i) (S (num_errors p_rep)))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { eapply transition_state_standard_closed_substitute_exact; eauto. }
  assert (Hsub_succ :
    subsumes_standard (length query)
      (std_pos (S i) (S (num_errors p_rep))) (std_pos (S i) (S e)) = true).
  { change (subsumes Standard (length query)
      (std_pos (S i) (S (num_errors p_rep))) (std_pos (S i) (S e)) = true).
    eapply subsumes_standard_substitute_successor_same_index; eauto. }
  assert (Hcomplete_rep :
    can_complete_to_final query n remaining
      (std_pos (S i) (S (num_errors p_rep)))).
  { eapply subsumption_preserves_can_complete.
    - exact Hcomplete.
    - exact Hsub_succ.
    - reflexivity.
    - reflexivity.
    - simpl. lia. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_rep.
Qed.

(** Ahead represented match preservation for the can-complete invariant.  The
    representative consumes the head dictionary character as an insert and then
    inherits the requested match successor's completion via Standard
    subsumption. *)
Lemma transition_state_standard_preserves_can_complete_match_represented_ahead_insert : forall
  query n remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i < term_index p_rep ->
  e <= n ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e p_rep Hqlen Hall_spec Hstate_bound
         Htrans Hin_rep Hsub_rep Hahead Herr Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_diff_index_lt_errors
                  (length query) p_rep i e Hsub_rep ltac:(lia)) as Hlt.
    lia. }
  pose (p_ins := std_pos (term_index p_rep) (S (num_errors p_rep))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp_rep_std in Hin_rep.
    apply transition_state_standard_closed_insert_exact; assumption. }
  assert (Hsub_succ :
    subsumes_standard (length query) p_ins (std_pos (S i) e) = true).
  { unfold p_ins.
    change (subsumes Standard (length query)
      (std_pos (term_index p_rep) (S (num_errors p_rep)))
      (std_pos (S i) e) = true).
    apply subsumes_standard_match_successor_ahead_insert.
    - exact Hsub_rep.
    - exact Hahead.
    - apply Hstate_bound. exact Hin_rep. }
  assert (Hcomplete_ins : can_complete_to_final query n remaining p_ins).
  { eapply subsumption_preserves_can_complete.
    - exact Hcomplete.
    - exact Hsub_succ.
    - reflexivity.
    - unfold p_ins. reflexivity.
    - unfold p_ins. simpl. apply Hstate_bound. exact Hin_rep. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_ins.
Qed.

(** Ahead represented substitution preservation for the can-complete invariant. *)
Lemma transition_state_standard_preserves_can_complete_substitute_represented_ahead_insert : forall
  query n remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i < term_index p_rep ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n remaining s c s' i e p_rep Hqlen Hall_spec Hstate_bound
         Htrans Hin_rep Hsub_rep Hahead He_lt Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  assert (Hrep_err_lt : num_errors p_rep < n).
  { pose proof (subsumes_standard_errors (length query) p_rep (std_pos i e)
                 Hsub_rep) as Herr_le.
    simpl in Herr_le. lia. }
  pose (p_ins := std_pos (term_index p_rep) (S (num_errors p_rep))).
  assert (Hclosed : In p_ins
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { unfold p_ins.
    rewrite Hp_rep_std in Hin_rep.
    apply transition_state_standard_closed_insert_exact; assumption. }
  assert (Hsub_succ :
    subsumes_standard (length query) p_ins (std_pos (S i) (S e)) = true).
  { unfold p_ins.
    change (subsumes Standard (length query)
      (std_pos (term_index p_rep) (S (num_errors p_rep)))
      (std_pos (S i) (S e)) = true).
    apply subsumes_standard_substitute_successor_ahead_insert.
    - exact Hsub_rep.
    - exact Hahead.
    - apply Hstate_bound. exact Hin_rep. }
  assert (Hcomplete_ins : can_complete_to_final query n remaining p_ins).
  { eapply subsumption_preserves_can_complete.
    - exact Hcomplete.
    - exact Hsub_succ.
    - reflexivity.
    - unfold p_ins. reflexivity.
    - unfold p_ins. simpl. apply Hstate_bound. exact Hin_rep. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_ins.
Qed.

(** Combined can-complete match preservation for representatives that are not
    behind the exact predecessor. *)
Lemma transition_state_standard_preserves_can_complete_match_represented_not_behind : forall
  query n dict remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i <= term_index p_rep ->
  e <= n ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e p_rep Hqlen Hall_reach Hall_spec
         Hstate_bound Htrans Hin_rep Hsub_rep Hnot_behind Herr Hlt Hnth Hcomplete.
  destruct (Nat.eq_dec (term_index p_rep) i) as [Hsame | Hneq].
  - eapply transition_state_standard_preserves_can_complete_match_represented_same_index; eauto.
  - eapply (transition_state_standard_preserves_can_complete_match_represented_ahead_insert
              query n remaining s c s' i e p_rep); eauto.
    lia.
Qed.

(** Behind represented match preservation.  If the retained representative is
    behind the exact predecessor, Standard's lookahead search finds the same
    consumed character at some offset no later than the exact predecessor.  The
    generated skip-to-match candidate can then delete-catch-up to the exact
    match successor's completion path. *)
Lemma transition_state_standard_preserves_can_complete_match_represented_behind_skip : forall
  query n dict remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep < i ->
  e <= n ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hbehind Herr Hlt Hnth
         Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  pose (pi := term_index p_rep).
  pose (pe := num_errors p_rep).
  pose (d := i - pi).
  pose (e_catch := pe + d).
  assert (Hcatch : e_catch <= e).
  { unfold e_catch, pe, d, pi.
    apply (subsumes_standard_behind_error_slack (length query)).
    - exact Hsub_rep.
    - lia. }
  assert (Hd_pos : 0 < d) by (unfold d, pi; lia).
  assert (Hpe_lt : pe < n) by (unfold pe, d, e_catch in *; lia).
  assert (Hreach_rep : position_reachable query n dict (std_pos pi pe)).
  { unfold pi, pe. rewrite <- Hp_rep_std. apply Hall_reach. exact Hin_rep. }
  assert (Htarget_reach : position_reachable query n dict (std_pos i e_catch)).
  { assert (Htarget_eq : std_pos i e_catch = std_pos (pi + d) (pe + d)).
    { unfold e_catch, d, pi, pe, std_pos. simpl. f_equal; lia. }
    rewrite Htarget_eq.
    apply position_reachable_delete_chain.
    - exact Hreach_rep.
    - unfold d, pi. lia.
    - unfold e_catch in *. lia. }
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  set (cv := characteristic_vector c query min_i (2 * n + 6)).
  set (limit := Nat.min (n - pe + 1) (length cv - (pi - min_i))).
  assert (Hmin_le_pi : min_i <= pi).
  { unfold min_i, pi. apply min_i_le_term_index. exact Hin_rep. }
  assert (Hmin_le_i : min_i <= i) by lia.
  assert (Hoffset_bound : i - min_i < 2 * n + 6).
  { assert (Hbounded : i - min_i <= 2 * n).
    { unfold min_i.
      change i with (term_index (std_pos i e_catch)).
      eapply reachable_term_index_minus_state_min_bounded.
      - exact Hall_reach.
      - exact Hall_spec.
      - exact Htarget_reach.
      - reflexivity.
      - simpl. unfold e_catch. lia.
      - exact Hin_rep.
      - rewrite Hqlen. unfold pi. lia. }
    lia. }
  assert (Hcv_target : cv_at cv (i - min_i) = true).
  { unfold cv.
    rewrite cv_at_char_matches by exact Hoffset_bound.
    replace (min_i + (i - min_i)) with i by lia.
    unfold char_matches_at.
    rewrite Hnth.
    apply char_eq_refl. }
  assert (Hcv_search : cv_at cv (pi - min_i + d) = true).
  { replace (pi - min_i + d) with (i - min_i) by (unfold d; lia).
    exact Hcv_target. }
  assert (Hd_limit : d < limit).
  { assert (Hd_err : d < n - pe + 1) by (unfold d, e_catch in *; lia).
    assert (Hd_window : d < length cv - (pi - min_i)).
    { unfold cv. rewrite char_vector_length.
      replace (pi - min_i + d) with (i - min_i) by (unfold d; lia).
      lia. }
    unfold limit.
    destruct (Nat.le_gt_cases (n - pe + 1) (length cv - (pi - min_i)))
      as [Hmin_left | Hmin_right].
    - rewrite Nat.min_l by exact Hmin_left. exact Hd_err.
    - rewrite Nat.min_r by lia. exact Hd_window. }
  destruct (index_of_match_finds_at_or_before cv (pi - min_i) limit d
              Hd_limit Hcv_search) as [j [Hidx Hj_le]].
  assert (Hclosed : In (std_pos (S (pi + j)) (pe + j))
    (epsilon_closure
       (transition_state_positions Standard (positions s)
          (characteristic_vector c query
             (fold_left Nat.min (map term_index (positions s)) (query_length s))
             (2 * n + 6))
          (fold_left Nat.min (map term_index (positions s)) (query_length s))
          n (query_length s))
       n (query_length s))).
  { eapply transition_state_standard_closed_index_match_exact.
    - unfold pi, pe. rewrite <- Hp_rep_std. exact Hin_rep.
    - rewrite Hqlen. unfold pi in *. lia.
    - unfold pe in *. exact Hpe_lt.
    - unfold cv, limit, min_i in Hidx. exact Hidx. }
  assert (Hcomplete_skip :
    can_complete_to_final query n remaining (std_pos (S (pi + j)) (pe + j))).
  { eapply can_complete_match_skip_candidate with
      (i := i) (e := e) (pi := pi) (pe := pe).
    - unfold pi. lia.
    - unfold e_catch, d in Hcatch. exact Hcatch.
    - exact Hlt.
    - exact Herr.
    - unfold d in Hj_le. exact Hj_le.
    - exact Hcomplete. }
  eapply transition_state_standard_has_can_complete_closed_member.
  - exact Hqlen.
  - exact Hstate_bound.
  - exact Htrans.
  - exact Hclosed.
  - exact Hcomplete_skip.
Qed.

(** Complete represented-match preservation, split only on the retained
    representative's relative query index. *)
Lemma transition_state_standard_preserves_can_complete_match_represented : forall
  query n dict remaining s c s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  e <= n ->
  i < length query ->
  nth_error query i = Some c ->
  can_complete_to_final query n remaining (std_pos (S i) e) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Herr Hlt Hnth Hcomplete.
  destruct (Nat.le_gt_cases i (term_index p_rep)) as [Hnot_behind | Hbehind].
  - eapply transition_state_standard_preserves_can_complete_match_represented_not_behind; eauto.
  - eapply transition_state_standard_preserves_can_complete_match_represented_behind_skip; eauto.
Qed.

(** Combined can-complete substitution preservation for representatives that
    are not behind the exact predecessor. *)
Lemma transition_state_standard_preserves_can_complete_substitute_represented_not_behind : forall
  query n dict remaining s c c' s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i <= term_index p_rep ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hnot_behind Hlt Hnth
         Hneq_ch He_lt Hcomplete.
  destruct (Nat.eq_dec (term_index p_rep) i) as [Hsame | Hneq_idx].
  - eapply transition_state_standard_preserves_can_complete_substitute_represented_same_index; eauto.
  - eapply (transition_state_standard_preserves_can_complete_substitute_represented_ahead_insert
              query n remaining s c s' i e p_rep); eauto.
    lia.
Qed.

(** Behind represented substitution preservation.  If the representative's
    current character matches the consumed dictionary character, the match
    successor has enough saved error budget to catch up by deletes.  Otherwise,
    Standard emits the representative's substitute successor, which has the
    same catch-up property. *)
Lemma transition_state_standard_preserves_can_complete_substitute_represented_behind : forall
  query n dict remaining s c c' s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  term_index p_rep < i ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hbehind Hlt Hnth
         Hneq_ch He_lt Hcomplete.
  assert (Hspec_rep : is_special p_rep = false).
  { apply Hall_spec. exact Hin_rep. }
  assert (Hp_rep_std : p_rep = std_pos (term_index p_rep) (num_errors p_rep)).
  { destruct p_rep as [j e' sp]. simpl in Hspec_rep. subst sp.
    unfold std_pos. simpl. reflexivity. }
  pose (pi := term_index p_rep).
  pose (pe := num_errors p_rep).
  pose (d := i - pi).
  pose (e_catch := pe + d).
  assert (Hcatch : e_catch <= e).
  { unfold e_catch, pe, d, pi.
    apply (subsumes_standard_behind_error_slack (length query)).
    - exact Hsub_rep.
    - lia. }
  assert (Hpe_lt : pe < n) by (unfold pe, d, e_catch in *; lia).
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)).
  set (cv := characteristic_vector c query min_i (2 * n + 6)).
  assert (Hmin_le_pi : min_i <= pi).
  { unfold min_i, pi. apply min_i_le_term_index. exact Hin_rep. }
  assert (Hin_std : In (std_pos pi pe) (positions s)).
  { unfold pi, pe. rewrite <- Hp_rep_std. exact Hin_rep. }
  destruct (cv_at cv (pi - min_i)) eqn:Hcv.
  - assert (Hclosed : In (std_pos (S pi) pe)
      (epsilon_closure
         (transition_state_positions Standard (positions s)
            (characteristic_vector c query
               (fold_left Nat.min (map term_index (positions s)) (query_length s))
               (2 * n + 6))
            (fold_left Nat.min (map term_index (positions s)) (query_length s))
            n (query_length s))
         n (query_length s))).
    { apply epsilon_closure_includes_input.
      unfold transition_state_positions.
      apply in_flat_map.
      exists (std_pos pi pe). split; [exact Hin_std |].
      unfold transition_position.
      apply transition_standard_produces_match.
      - rewrite Hqlen. unfold pi. lia.
      - exact Hmin_le_pi.
      - unfold cv in Hcv. exact Hcv. }
    assert (Hcomplete_match :
      can_complete_to_final query n remaining (std_pos (S pi) pe)).
    { eapply can_complete_substitute_behind_match_candidate with
        (i := i) (e := e) (pi := pi) (pe := pe).
      - unfold pi. lia.
      - unfold e_catch, d in Hcatch. exact Hcatch.
      - exact Hlt.
      - exact He_lt.
      - exact Hcomplete. }
    eapply transition_state_standard_has_can_complete_closed_member.
    + exact Hqlen.
    + exact Hstate_bound.
    + exact Htrans.
    + exact Hclosed.
    + exact Hcomplete_match.
  - assert (Hclosed : In (std_pos (S pi) (S pe))
      (epsilon_closure
         (transition_state_positions Standard (positions s)
            (characteristic_vector c query
               (fold_left Nat.min (map term_index (positions s)) (query_length s))
               (2 * n + 6))
            (fold_left Nat.min (map term_index (positions s)) (query_length s))
            n (query_length s))
         n (query_length s))).
    { apply epsilon_closure_includes_input.
      unfold transition_state_positions.
      apply in_flat_map.
      exists (std_pos pi pe). split; [exact Hin_std |].
      unfold transition_position.
      apply transition_standard_produces_substitute.
      - rewrite Hqlen. unfold pi. lia.
      - exact Hmin_le_pi.
      - unfold cv in Hcv. exact Hcv.
      - exact Hpe_lt. }
    assert (Hcomplete_sub :
      can_complete_to_final query n remaining (std_pos (S pi) (S pe))).
    { eapply can_complete_substitute_behind_substitute_candidate with
        (i := i) (e := e) (pi := pi) (pe := pe).
      - unfold pi. lia.
      - unfold e_catch, d in Hcatch. exact Hcatch.
      - exact Hlt.
      - exact He_lt.
      - exact Hcomplete. }
    eapply transition_state_standard_has_can_complete_closed_member.
    + exact Hqlen.
    + exact Hstate_bound.
    + exact Htrans.
    + exact Hclosed.
    + exact Hcomplete_sub.
Qed.

(** Complete represented-substitution preservation, split on the retained
    representative's relative query index. *)
Lemma transition_state_standard_preserves_can_complete_substitute_represented : forall
  query n dict remaining s c c' s' i e p_rep,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  transition_state Standard s c query n = Some s' ->
  In p_rep (positions s) ->
  subsumes Standard (length query) p_rep (std_pos i e) = true ->
  i < length query ->
  nth_error query i = Some c' ->
  c <> c' ->
  e < n ->
  can_complete_to_final query n remaining (std_pos (S i) (S e)) ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict remaining s c c' s' i e p_rep Hqlen Hall_reach
         Hall_spec Hstate_bound Htrans Hin_rep Hsub_rep Hlt Hnth Hneq_ch
         He_lt Hcomplete.
  destruct (Nat.le_gt_cases i (term_index p_rep)) as [Hnot_behind | Hbehind].
  - eapply transition_state_standard_preserves_can_complete_substitute_represented_not_behind; eauto.
  - eapply transition_state_standard_preserves_can_complete_substitute_represented_behind; eauto.
Qed.

(** A represented completable source remains completable after consuming one
    dictionary character.  Leading delete steps are discharged against the
    current state's delete-chain representation; the first consuming step then
    uses the represented match/substitute/insert preservation lemmas above. *)
Lemma transition_state_standard_preserves_can_reach_represented : forall
  query n dict c remaining s s' p p_final,
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  state_delete_chain_represented n s ->
  transition_state Standard s c query n = Some s' ->
  positions_subsume Standard (length query) (positions s) p ->
  can_reach query n p (c :: remaining) p_final ->
  term_index p_final = length query ->
  num_errors p_final <= n ->
  is_special p_final = false ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' p p_final Hqlen Hall_reach
         Hall_spec Hstate_bound Hclosed Htrans Hrep Hreach_path
         Hfinal_term Hfinal_err Hfinal_spec.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s s' dict Hqlen Hall_reach Hall_spec Hstate_bound
         Hclosed Htrans Hrep.
  induction Hreach_path; intros c0 remaining0 Hinput s0 s' dict0 Hqlen
         Hall_reach Hall_spec Hstate_bound Hclosed Htrans Hrep.
  - discriminate.
  - subst p.
    assert (Hrep_del :
      positions_subsume Standard (length query) (positions s0)
        (std_pos (S i) (S e))).
    { eapply represented_delete_successor_from_closed_state.
      - exact Hqlen.
      - exact Hstate_bound.
      - exact Hclosed.
      - exact Hrep.
      - lia.
      - lia. }
    eapply IHHreach_path; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    destruct Hrep as [p_rep [Hin_rep Hsub_rep]].
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos (S i) e)).
    { exists p_final. repeat split; assumption. }
    assert (He_le : e <= n).
    { pose proof (can_reach_errors_monotone query n (std_pos (S i) e)
                  remaining0 p_final Hreach_path) as Herr_mono.
      simpl in Herr_mono. lia. }
    eapply transition_state_standard_preserves_can_complete_match_represented; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    destruct Hrep as [p_rep [Hin_rep Hsub_rep]].
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos (S i) (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_standard_preserves_can_complete_substitute_represented; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (Hcomplete :
      can_complete_to_final query n remaining0 (std_pos i (S e))).
    { exists p_final. repeat split; assumption. }
    eapply transition_state_standard_preserves_can_complete_insert_represented; eauto.
Qed.

(** State-level one-character preservation for the can-complete invariant. *)
Lemma transition_state_standard_preserves_state_has_completable : forall
  query n dict c remaining s s',
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  state_delete_chain_represented n s ->
  transition_state Standard s c query n = Some s' ->
  state_has_completable query n (c :: remaining) s ->
  state_has_completable query n remaining s'.
Proof.
  intros query n dict c remaining s s' Hqlen Hall_reach Hall_spec
         Hstate_bound Hclosed Htrans [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_standard_preserves_can_reach_represented
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply positions_subsume_standard_refl_in. exact Hin.
Qed.

(** Non-dead counterpart of the one-character preservation lemma. *)
Lemma transition_state_standard_succeeds_from_can_reach_represented : forall
  query n dict c remaining s p p_final,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  state_delete_chain_represented n s ->
  positions_subsume Standard (length query) (positions s) p ->
  can_reach query n p (c :: remaining) p_final ->
  num_errors p_final <= n ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros query n dict c remaining s p p_final Halg Hqlen Hall_reach
         Hall_spec Hstate_bound Hclosed Hrep Hreach_path Hfinal_err.
  remember (c :: remaining) as input eqn:Hinput.
  revert c remaining Hinput s dict Halg Hqlen Hall_reach Hall_spec Hstate_bound
         Hclosed Hrep.
  induction Hreach_path; intros c0 remaining0 Hinput s0 dict0 Halg Hqlen
         Hall_reach Hall_spec Hstate_bound Hclosed Hrep.
  - discriminate.
  - subst p.
    assert (Hrep_del :
      positions_subsume Standard (length query) (positions s0)
        (std_pos (S i) (S e))).
    { eapply represented_delete_successor_from_closed_state.
      - exact Hqlen.
      - exact Hstate_bound.
      - exact Hclosed.
      - exact Hrep.
      - lia.
      - lia. }
    eapply IHHreach_path; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    assert (He_le : e <= n).
    { pose proof (can_reach_errors_monotone query n (std_pos (S i) e)
                  remaining0 p_final Hreach_path) as Herr_mono.
      simpl in Herr_mono. lia. }
    eapply transition_state_not_dead_standard_represented_match; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_not_dead_standard_represented_error_lt; eauto.
  - injection Hinput as Hc_eq Hrem_eq. subst c remaining.
    subst p.
    eapply transition_state_not_dead_standard_represented_error_lt; eauto.
Qed.

Lemma transition_state_standard_succeeds_from_state_has_completable : forall
  query n dict c remaining s,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  state_delete_chain_represented n s ->
  state_has_completable query n (c :: remaining) s ->
  exists s', transition_state Standard s c query n = Some s'.
Proof.
  intros query n dict c remaining s Halg Hqlen Hall_reach Hall_spec
         Hstate_bound Hclosed [p [Hin Hcomplete]].
  destruct Hcomplete as [p_final [Hreach [Hterm [Herr Hspec]]]].
  eapply transition_state_standard_succeeds_from_can_reach_represented
    with (dict := dict) (p := p) (p_final := p_final); eauto.
  apply positions_subsume_standard_refl_in. exact Hin.
Qed.

(** If the suffix is empty, a completable state already represents a final
    Standard position.  Empty-suffix [can_reach] paths consist only of delete
    moves, and [state_delete_chain_represented] carries that final delete
    endpoint through antichain pruning. *)
Lemma state_has_completable_empty_accepts_standard : forall query n s,
  query_length s = length query ->
  (forall p, In p (positions s) -> is_special p = false) ->
  state_delete_chain_represented n s ->
  state_has_completable query n [] s ->
  state_is_final s = true.
Proof.
  intros query n s Hqlen Hall_spec Hclosed
         [p [Hp_in [p_final [Hreach [Hterm_final [Herr_final Hspec_final]]]]]].
  pose proof (can_reach_term_index_monotone query n p [] p_final Hreach)
    as Hterm_mono.
  pose proof (can_reach_empty_remaining_errors query n p p_final Hreach)
    as Herr_exact.
  set (k := term_index p_final - term_index p).
  assert (Hk_term : term_index p + k <= query_length s).
  { unfold k. rewrite Hqlen, Hterm_final. lia. }
  assert (Hk_err : num_errors p + k <= n).
  { unfold k. rewrite Herr_exact in Herr_final. exact Herr_final. }
  destruct (Hclosed p k Hp_in Hk_term Hk_err) as [p' [Hp'_in Hp'_sub]].
  assert (Htarget_eq :
    std_pos (term_index p + k) (num_errors p + k) = p_final).
  { destruct p_final as [i_f e_f sp_f].
    simpl in Hterm_final, Herr_final, Hspec_final, Hterm_mono, Herr_exact.
    subst sp_f.
    unfold k, std_pos. simpl.
    f_equal; lia. }
  rewrite Htarget_eq in Hp'_sub.
  unfold state_is_final.
  rewrite existsb_exists.
  exists p'. split; [exact Hp'_in |].
  unfold position_is_final.
  rewrite Nat.leb_le.
  destruct (position_is_final_for_subsumption (query_length s) p') eqn:Hp'_final.
  - unfold position_is_final_for_subsumption in Hp'_final.
    apply Nat.leb_le in Hp'_final. exact Hp'_final.
  - exfalso.
    assert (Hfinal_target :
      position_is_final_for_subsumption (query_length s) p_final = true).
    { unfold position_is_final_for_subsumption.
      rewrite Nat.leb_le. rewrite Hqlen, Hterm_final. lia. }
    pose proof (non_final_cannot_subsume_final
                  Standard (query_length s) p' p_final
                  Hp'_final Hfinal_target) as Hnot_sub.
    rewrite Hp'_sub in Hnot_sub. discriminate.
Qed.

(** Run-level preservation of the can-complete invariant from any reachable
    Standard state satisfying the local state invariants. *)
Lemma automaton_run_standard_preserves_completable_from_state : forall
  query n remaining dict_prefix s,
  algorithm s = Standard ->
  query_length s = length query ->
  (forall q, In q (positions s) -> position_reachable query n dict_prefix q) ->
  (forall q, In q (positions s) -> is_special q = false) ->
  (forall q, In q (positions s) -> term_index q <= length query) ->
  state_delete_chain_represented n s ->
  state_has_completable query n remaining s ->
  exists final,
    automaton_run Standard query n remaining s = Some final /\
    state_has_completable query n [] final.
Proof.
  intros query n remaining.
  induction remaining as [|c rest IH]; intros dict_prefix s Halg Hqlen
         Hall_reach Hall_spec Hstate_bound Hclosed Hcomplete.
  - exists s. split; [reflexivity | exact Hcomplete].
  - destruct (transition_state_standard_succeeds_from_state_has_completable
                query n dict_prefix c rest s Halg Hqlen Hall_reach Hall_spec
                Hstate_bound Hclosed Hcomplete) as [s_mid Htrans].
    assert (Hcomplete_mid : state_has_completable query n rest s_mid).
    { eapply transition_state_standard_preserves_state_has_completable; eauto. }
    assert (Hrun_one : automaton_run Standard query n [c] s = Some s_mid).
    { simpl. rewrite Htrans. reflexivity. }
    assert (Halg_mid : algorithm s_mid = Standard).
    { apply (transition_state_preserves_algorithm Standard s c query n s_mid Htrans). }
    assert (Hqlen_mid : query_length s_mid = length query).
    { rewrite (transition_state_preserves_query_length Standard s c query n s_mid Htrans).
      exact Hqlen. }
    assert (Hall_reach_mid : forall q, In q (positions s_mid) ->
      position_reachable query n (dict_prefix ++ [c]) q).
    { intros q Hq.
      eapply automaton_run_preserves_reachable_standard.
      - exact Hqlen.
      - exact Hrun_one.
      - intros p Hp. split; [apply Hall_reach | apply Hall_spec]; exact Hp.
      - exact Hq. }
    assert (Hall_spec_mid : forall q, In q (positions s_mid) ->
      is_special q = false).
    { apply (standard_run_positions_non_special query n [c] s s_mid Hrun_one).
      exact Hall_spec. }
    assert (Hstate_bound_mid : forall q, In q (positions s_mid) ->
      term_index q <= length query).
    { intros q Hq.
      apply reachable_term_index_bound_query with
        (n := n) (dict_prefix := dict_prefix ++ [c]).
      apply Hall_reach_mid. exact Hq. }
    assert (Hclosed_mid : state_delete_chain_represented n s_mid).
    { eapply transition_state_standard_state_delete_chain_represented.
      exact Htrans. }
    destruct (IH (dict_prefix ++ [c]) s_mid Halg_mid Hqlen_mid
                Hall_reach_mid Hall_spec_mid Hstate_bound_mid Hclosed_mid
                Hcomplete_mid) as [final [Hrun_rest Hcomplete_final]].
    exists final. split.
    + simpl. rewrite Htrans. exact Hrun_rest.
    + exact Hcomplete_final.
Qed.

(** If the Levenshtein bound holds, the Standard run reaches a final state
    carrying the can-complete invariant with an empty remaining suffix. *)
Lemma automaton_run_standard_completable_from_lev_bound : forall query dict n,
  lev_distance query dict <= n ->
  exists final,
    automaton_run_from_initial Standard query n dict = Some final /\
    state_has_completable query n [] final.
Proof.
  intros query dict n Hdist.
  unfold automaton_run_from_initial.
  set (init_closed :=
    mkState (epsilon_closure (positions (initial_state Standard (length query)))
                         n (length query)) Standard (length query)).
  assert (Hinit_eq :
    init_closed =
    mkState (epsilon_closure [initial_position] n (length query))
            Standard (length query)).
  { unfold init_closed, initial_state. reflexivity. }
  rewrite Hinit_eq.
  apply automaton_run_standard_preserves_completable_from_state with
    (dict_prefix := []).
  - reflexivity.
  - reflexivity.
  - intros p Hp.
    apply initial_closed_state_reachable in Hp.
    destruct Hp as [Hreach _]. exact Hreach.
  - intros p Hp.
    apply initial_closed_state_reachable in Hp.
    destruct Hp as [_ Hspec]. exact Hspec.
  - intros p Hp.
    apply reachable_term_index_bound_query with (n := n) (dict_prefix := []).
    apply initial_closed_state_reachable in Hp.
    destruct Hp as [Hreach _]. exact Hreach.
  - apply initial_closed_state_delete_chain_represented.
  - apply lev_bound_initial_closed_has_completable.
    exact Hdist.
Qed.

(** Run-level final acceptance bridge for the can-complete invariant.  The
    remaining missing Standard-completeness step is to preserve
    [state_has_completable] through each consumed dictionary character; once
    that invariant reaches the empty suffix, finality is local. *)
Lemma automaton_run_standard_final_completable_accepts : forall query n dict final,
  automaton_run_from_initial Standard query n dict = Some final ->
  state_has_completable query n [] final ->
  state_is_final final = true.
Proof.
  intros query n dict final Hrun Hcomplete.
  assert (Hqlen : query_length final = length query).
  { unfold automaton_run_from_initial in Hrun.
    rewrite (automaton_run_preserves_query_length Standard query n dict
               (mkState (epsilon_closure [initial_position] n (length query))
                        Standard (length query))
               final Hrun).
    reflexivity. }
  assert (Hall_spec : forall p, In p (positions final) -> is_special p = false).
  { unfold automaton_run_from_initial in Hrun.
    apply (standard_run_positions_non_special query n dict
             (mkState (epsilon_closure [initial_position] n (length query))
                      Standard (length query)) final Hrun).
    intros p Hp.
    apply initial_closed_state_reachable in Hp.
    destruct Hp as [_ Hspec]. exact Hspec. }
  apply state_has_completable_empty_accepts_standard with (query := query) (n := n).
  - exact Hqlen.
  - exact Hall_spec.
  - apply automaton_run_from_initial_standard_delete_chain_represented with
      (query := query) (dict := dict).
    exact Hrun.
  - exact Hcomplete.
Qed.

(** Unconditional Standard completeness through the can-complete invariant. *)
Theorem automaton_complete_standard_can_complete : forall query dict n,
  lev_distance query dict <= n ->
  automaton_accepts Standard query n dict = true.
Proof.
  intros query dict n Hdist.
  destruct (automaton_run_standard_completable_from_lev_bound query dict n Hdist)
    as [final [Hrun Hcomplete]].
  unfold automaton_accepts.
  rewrite Hrun.
  apply automaton_run_standard_final_completable_accepts with
    (query := query) (n := n) (dict := dict).
  - exact Hrun.
  - exact Hcomplete.
Qed.

(** Now we can prove the main completeness property using Option C approach *)

(** Helper: if the Standard run produces a state and a final position is
    reachable within the bound, unconditional Standard completeness makes that
    produced state accepting. *)
Lemma automaton_final_state_accepts_standard : forall
  query n dict final p,
  automaton_run_from_initial Standard query n dict = Some final ->
  position_reachable query n dict p ->
  term_index p = length query ->
  is_special p = false ->
  num_errors p <= n ->
  state_is_final final = true.
Proof.
  intros query n dict final p Hrun Hreach Hfinal Hspec Herr.
  assert (Hdist : lev_distance query dict <= n).
  { pose proof (reachable_final_to_distance query dict n p Hreach Hspec Hfinal)
      as Hdist.
    lia. }
  pose proof (automaton_complete_standard_can_complete query dict n Hdist) as Haccept.
  unfold automaton_accepts in Haccept.
  rewrite Hrun in Haccept.
  exact Haccept.
Qed.

(** Simplified version for the main completeness proof *)
Lemma reachable_final_implies_accepts : forall
  query dict n p,
  position_reachable query n dict p ->
  term_index p = length query ->
  is_special p = false ->
  num_errors p <= n ->
  automaton_accepts Standard query n dict = true.
Proof.
  intros query dict n p Hreach Hfinal Hspec Herr.
  apply automaton_complete_standard_can_complete.
  pose proof (reachable_final_to_distance query dict n p Hreach Hspec Hfinal)
    as Hdist.
  lia.
Qed.

(** * Helper Lemmas for Algorithm Inclusion *)

(** flat_map respects inclusion: if f(x) ⊆ g(x) for all x, then flat_map f l ⊆ flat_map g l *)
Lemma flat_map_incl : forall {A B : Type} (f g : A -> list B) (l : list A),
  (forall x, In x l -> incl (f x) (g x)) ->
  incl (flat_map f l) (flat_map g l).
Proof.
  intros A B f g l Hincl.
  unfold incl.
  intros y Hin.
  apply in_flat_map in Hin.
  destruct Hin as [x [Hinx Hiny]].
  apply in_flat_map.
  exists x. split.
  - exact Hinx.
  - apply Hincl; assumption.
Qed.

(** Transposition's transition_state_positions includes Standard's for non-special input *)
Lemma transition_state_positions_incl_standard_transposition :
  forall positions cv min_i n qlen,
  (forall p, In p positions -> is_special p = false) ->
  incl (transition_state_positions Standard positions cv min_i n qlen)
       (transition_state_positions Transposition positions cv min_i n qlen).
Proof.
  intros positions cv min_i n qlen Hall.
  unfold transition_state_positions.
  apply flat_map_incl.
  intros p Hin.
  unfold transition_position.
  specialize (Hall p Hin).
  apply transposition_includes_standard.
  exact Hall.
Qed.

(** Similarly for MergeAndSplit *)
Lemma transition_state_positions_incl_standard_merge_split :
  forall positions cv min_i n qlen,
  (forall p, In p positions -> is_special p = false) ->
  incl (transition_state_positions Standard positions cv min_i n qlen)
       (transition_state_positions MergeAndSplit positions cv min_i n qlen).
Proof.
  intros positions cv min_i n qlen Hall.
  unfold transition_state_positions.
  apply flat_map_incl.
  intros p Hin.
  unfold transition_position.
  specialize (Hall p Hin).
  apply merge_split_includes_standard.
  exact Hall.
Qed.

(** Epsilon closure preserves inclusion.
    This is a key lemma showing that if positions1 ⊆ positions2, then
    their epsilon closures maintain this relationship.

    The proof uses induction on fuel, with the key insight that:
    1. Original positions are included in the output (epsilon_closure_includes_input)
    2. New positions from delete_step are monotonic in the input positions
    3. Even when is_nil differs between branches, the inclusion holds
       because epsilon_closure_aux returns a superset of its input *)

Lemma epsilon_closure_aux_incl : forall fuel positions1 positions2 n qlen,
  incl positions1 positions2 ->
  incl (epsilon_closure_aux positions1 n qlen fuel)
       (epsilon_closure_aux positions2 n qlen fuel).
Proof.
  induction fuel as [| fuel' IH]; intros positions1 positions2 n qlen Hincl.
  - (* fuel = 0 *)
    simpl. exact Hincl.
  - (* fuel = S fuel' *)
    simpl.
    set (new1 := flat_map (fun p => match delete_step p n qlen with
                                    | Some p' => [p']
                                    | None => []
                                    end) positions1) in *.
    set (new2 := flat_map (fun p => match delete_step p n qlen with
                                    | Some p' => [p']
                                    | None => []
                                    end) positions2) in *.
    (* Key fact: new1 ⊆ new2 because positions1 ⊆ positions2 *)
    assert (Hnew_incl : incl new1 new2).
    { unfold new1, new2.
      unfold incl. intros p Hp.
      apply in_flat_map in Hp.
      destruct Hp as [p0 [Hin0 Hp0]].
      apply in_flat_map.
      exists p0. split.
      - apply Hincl. exact Hin0.
      - exact Hp0. }

    destruct (is_nil new1) eqn:Hnil1.
    + (* new1 is empty *)
      destruct (is_nil new2) eqn:Hnil2.
      * (* Both empty - return originals *)
        exact Hincl.
      * (* new1 empty, new2 non-empty *)
        (* Result is IH applied to (positions2 ++ new2) *)
        (* We need: positions1 ⊆ epsilon_closure_aux (positions2 ++ new2) fuel' *)
        (* Use transitivity: positions1 ⊆ positions2 ⊆ positions2++new2 ⊆ closure *)
        intros p Hp.
        apply epsilon_closure_aux_includes_input.
        apply in_or_app. left.
        apply Hincl. exact Hp.
    + (* new1 is non-empty *)
      destruct (is_nil new2) eqn:Hnil2.
      * (* new1 non-empty, new2 empty - contradiction *)
        (* If new1 is non-empty but new2 is empty, and new1 ⊆ new2, contradiction *)
        unfold is_nil in Hnil1.
        destruct new1 as [| p1 rest1] eqn:Hnew1eq.
        -- discriminate Hnil1.
        -- (* new1 = p1 :: rest1, but incl (p1::rest1) new2 and is_nil new2 = true *)
           unfold is_nil in Hnil2.
           destruct new2 as [| p2 rest2] eqn:Hnew2eq.
           ++ (* new2 = [] but incl (p1::rest1) [] is False *)
              unfold incl in Hnew_incl.
              specialize (Hnew_incl p1 (in_eq p1 rest1)).
              simpl in Hnew_incl. contradiction.
           ++ discriminate Hnil2.
      * (* Both non-empty - recurse *)
        apply IH.
        intros p Hp.
        apply in_app_or in Hp.
        apply in_or_app.
        destruct Hp as [Hp | Hp].
        -- left. apply Hincl. exact Hp.
        -- right. apply Hnew_incl. exact Hp.
Qed.

Lemma epsilon_closure_incl : forall positions1 positions2 n qlen,
  incl positions1 positions2 ->
  incl (epsilon_closure positions1 n qlen) (epsilon_closure positions2 n qlen).
Proof.
  intros positions1 positions2 n qlen Hincl.
  unfold epsilon_closure.
  apply epsilon_closure_aux_incl.
  exact Hincl.
Qed.

(** Key lemma: if positions1 ⊆ positions2 and both are non-empty,
    and positions1 produces a non-empty antichain state,
    then positions2 also produces a non-empty state containing all final positions
    from positions1 (up to subsumption). *)

(** Simpler approach: use the fact that final positions are determined by term_index.
    If Standard has a final position (term_index >= qlen), and Transposition has
    positions that include all Standard positions (before antichain filtering),
    then Transposition also has a final position. *)

(** Final position preservation under inclusion *)
Lemma final_position_preserved : forall qlen positions1 positions2,
  incl positions1 positions2 ->
  existsb (position_is_final qlen) positions1 = true ->
  existsb (position_is_final qlen) positions2 = true.
Proof.
  intros qlen positions1 positions2 Hincl Hfinal.
  rewrite existsb_exists in Hfinal.
  rewrite existsb_exists.
  destruct Hfinal as [p [Hin Hpfinal]].
  exists p. split.
  - apply Hincl. exact Hin.
  - exact Hpfinal.
Qed.

(** ** Transition step helpers for Standard ⊆ Transposition *)

(** Helper: closed_positions of Transposition include those of Standard
    when starting from states where Standard positions ⊆ Transposition positions *)
Lemma closed_positions_incl_standard_transposition :
  forall positions_std positions_trans cv min_i n qlen,
  (forall p, In p positions_std -> is_special p = false) ->
  incl positions_std positions_trans ->
  incl (epsilon_closure (transition_state_positions Standard positions_std cv min_i n qlen) n qlen)
       (epsilon_closure (transition_state_positions Transposition positions_trans cv min_i n qlen) n qlen).
Proof.
  intros positions_std positions_trans cv min_i n qlen Hnonspec Hincl.
  apply epsilon_closure_incl.
  unfold incl. intros p Hp.
  unfold transition_state_positions in *.
  apply in_flat_map in Hp.
  destruct Hp as [p0 [Hin0 Hp0]].
  apply in_flat_map.
  exists p0. split.
  - apply Hincl. exact Hin0.
  - unfold transition_position.
    specialize (Hnonspec p0 Hin0).
    apply transposition_includes_standard.
    + exact Hnonspec.
    + exact Hp0.
Qed.

(** Helper: if Standard's closed_positions is non-empty, so is Transposition's *)
Lemma transition_non_empty_standard_transposition :
  forall positions_std positions_trans cv min_i n qlen,
  (forall p, In p positions_std -> is_special p = false) ->
  incl positions_std positions_trans ->
  is_nil (epsilon_closure (transition_state_positions Standard positions_std cv min_i n qlen) n qlen) = false ->
  is_nil (epsilon_closure (transition_state_positions Transposition positions_trans cv min_i n qlen) n qlen) = false.
Proof.
  intros positions_std positions_trans cv min_i n qlen Hnonspec Hincl Hnonempty.
  pose proof (closed_positions_incl_standard_transposition
                positions_std positions_trans cv min_i n qlen Hnonspec Hincl) as Hclosed_incl.
  remember (epsilon_closure (transition_state_positions Standard positions_std cv min_i n qlen) n qlen) as closed_std.
  remember (epsilon_closure (transition_state_positions Transposition positions_trans cv min_i n qlen) n qlen) as closed_trans.
  unfold is_nil in *.
  destruct closed_std as [| p_std rest_std].
  - discriminate Hnonempty.
  - destruct closed_trans as [| p_trans rest_trans].
    + (* Transposition's closed is empty but Standard's is not - contradiction *)
      unfold incl in Hclosed_incl.
      specialize (Hclosed_incl p_std (in_eq p_std rest_std)).
      simpl in Hclosed_incl. contradiction.
    + reflexivity.
Qed.

(** Helper: if Standard's closed_positions has a final position, Transposition's state does too *)
Lemma transition_final_standard_transposition :
  forall positions_std positions_trans cv min_i n qlen alg_trans,
  (forall p, In p positions_std -> is_special p = false) ->
  incl positions_std positions_trans ->
  let closed_std := epsilon_closure (transition_state_positions Standard positions_std cv min_i n qlen) n qlen in
  let closed_trans := epsilon_closure (transition_state_positions Transposition positions_trans cv min_i n qlen) n qlen in
  existsb (position_is_final qlen) closed_std = true ->
  is_nil closed_trans = false ->
  state_is_final (fold_left (fun s p => state_insert p s) closed_trans (empty_state alg_trans qlen)) = true.
Proof.
  intros positions_std positions_trans cv min_i n qlen alg_trans Hnonspec Hincl.
  intros closed_std closed_trans Hfinal_std Hnonempty_trans.
  unfold state_is_final.
  assert (Hqlen_fold : query_length (fold_left (fun s p => state_insert p s) closed_trans (empty_state alg_trans qlen)) = qlen).
  { rewrite fold_state_insert_preserves_query_length.
    unfold empty_state. simpl. reflexivity. }
  rewrite Hqlen_fold.
  apply (fold_state_insert_has_final alg_trans qlen closed_trans (empty_state alg_trans qlen)).
  - unfold empty_state. simpl. reflexivity.
  - unfold empty_state. simpl. reflexivity.
  - (* Need: existsb (position_is_final qlen) closed_trans = true *)
    apply final_position_preserved with (positions1 := closed_std).
    + apply closed_positions_incl_standard_transposition; assumption.
    + exact Hfinal_std.
Qed.

(** Helper: fold_left Nat.min is monotonic in the accumulator *)
Lemma fold_left_min_mono : forall l init1 init2,
  init1 <= init2 ->
  fold_left Nat.min l init1 <= fold_left Nat.min l init2.
Proof.
  intros l.
  induction l as [| x rest IH].
  - simpl. auto.
  - simpl. intros init1 init2 Hle.
    apply IH. lia.
Qed.

(** Helper: adding an element to a list can only decrease the fold_left Nat.min *)
Lemma fold_left_min_cons_le : forall l init x,
  fold_left Nat.min (x :: l) init <= fold_left Nat.min l init.
Proof.
  intros l init x.
  simpl.
  (* fold_left Nat.min l (Nat.min x init) <= fold_left Nat.min l init *)
  apply fold_left_min_mono.
  lia.
Qed.

(** Helper: min_i for a superset is <= min_i for a subset *)
Lemma min_i_incl : forall positions1 positions2 init,
  incl positions1 positions2 ->
  fold_left Nat.min (map term_index positions2) init <=
  fold_left Nat.min (map term_index positions1) init.
Proof.
  intros positions1 positions2 init Hincl.
  (* Key insight: The fold_left Nat.min over positions2 is bounded by either init
     or any element in map term_index positions2. Since positions1 ⊆ positions2,
     every element in positions1 is also in positions2. *)
  set (min1 := fold_left Nat.min (map term_index positions1) init).
  set (min2 := fold_left Nat.min (map term_index positions2) init).

  (* Case analysis: Is min1 < init or min1 = init? *)
  destruct (Nat.lt_ge_cases min1 init) as [Hlt | Hge].
  - (* min1 < init: min1 is in map term_index positions1 *)
    assert (Hin_min1 : In min1 (map term_index positions1)).
    { apply fold_left_min_in_list. exact Hlt. }
    (* So there's some p in positions1 with term_index p = min1 *)
    apply in_map_iff in Hin_min1.
    destruct Hin_min1 as [p [Heq Hinp]].
    (* Since p ∈ positions1 and positions1 ⊆ positions2, p ∈ positions2 *)
    assert (Hinp2 : In p positions2).
    { apply Hincl. exact Hinp. }
    (* So term_index p = min1 is in map term_index positions2 *)
    assert (Hin_min1_2 : In min1 (map term_index positions2)).
    { apply in_map_iff. exists p. split; [exact Heq | exact Hinp2]. }
    (* By fold_left_min_le_elem, min2 <= min1 *)
    apply fold_left_min_le_elem. exact Hin_min1_2.
  - (* min1 >= init: by fold_left_min_le_init, min1 <= init, so min1 = init *)
    assert (Heq : min1 = init).
    { pose proof (fold_left_min_le_init (map term_index positions1) init) as Hle.
      fold min1 in Hle. lia. }
    (* Now min2 <= init = min1 by fold_left_min_le_init *)
    rewrite Heq.
    apply fold_left_min_le_init.
Qed.

(** Reverse of fold_state_insert_has_final: if the fold result is final,
    then the input list contains a final position.

    This follows because positions in the fold result come from the input list
    (via fold_state_insert_positions from Soundness.v). *)
Lemma fold_state_insert_final_reverse : forall alg qlen positions,
  state_is_final (fold_left (fun s p => state_insert p s) positions (empty_state alg qlen)) = true ->
  existsb (position_is_final qlen) positions = true.
Proof.
  intros alg qlen positions Hfinal.
  unfold state_is_final in Hfinal.
  rewrite fold_state_insert_preserves_query_length in Hfinal.
  simpl in Hfinal.
  rewrite existsb_exists in Hfinal.
  destruct Hfinal as [p_final [Hin_fold Hp_final]].
  (* p_final is in positions of the fold result, so by fold_state_insert_positions,
     it's in either [] or positions. Since [] is empty, p_final ∈ positions. *)
  apply fold_state_insert_positions in Hin_fold.
  destruct Hin_fold as [Hin_empty | Hin_positions].
  - simpl in Hin_empty. contradiction.
  - rewrite existsb_exists. exists p_final. split; [exact Hin_positions | exact Hp_final].
Qed.

(** Helper: if l1 ⊆ l2 and l1 is non-empty, then l2 is non-empty. *)
Lemma incl_not_nil : forall {A : Type} (l1 l2 : list A),
  incl l1 l2 ->
  is_nil l1 = false ->
  is_nil l2 = false.
Proof.
  intros A l1 l2 Hincl Hnonempty.
  destruct l1 as [| x xs].
  - simpl in Hnonempty. discriminate.
  - destruct l2 as [| y ys].
    + (* l1 = x::xs but l2 = [], contradiction via incl *)
      assert (Hin : In x (x :: xs)) by (left; reflexivity).
      apply Hincl in Hin. simpl in Hin. contradiction.
    + simpl. reflexivity.
Qed.

Lemma standard_accepts_implies_transposition_accepts : forall
  query n dict,
  automaton_accepts Standard query n dict = true ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros query n dict Haccept.
  destruct (automaton_run_transposition_completable_from_lev_bound
              query dict n (automaton_sound_standard query dict n Haccept))
    as [final [Hrun Hcomplete]].
  unfold automaton_accepts.
  rewrite Hrun.
  eapply automaton_run_transposition_final_completable_accepts; eauto.
Qed.

(** Similar lemma for Transposition algorithm.

    For the Transposition algorithm, positions can include special positions
    that represent transposition-in-progress states. The proof requires showing
    that the automaton explores all transposition paths.

    Since position_reachable uses only Standard operations (match, substitute,
    delete, insert), any Standard-reachable position is also reachable in
    Transposition. Therefore, if Standard accepts, Transposition also accepts. *)
Lemma reachable_final_implies_accepts_transposition : forall
  query dict n p,
  position_reachable query n dict p ->
  term_index p = length query ->
  is_special p = false ->
  num_errors p <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros query dict n p Hreach Hfinal Hspec Herr.
  (* Use the fact that Standard acceptance implies Transposition acceptance *)
  apply standard_accepts_implies_transposition_accepts.
  apply (reachable_final_implies_accepts query dict n p); assumption.
Qed.

Lemma standard_accepts_implies_merge_split_accepts : forall query n dict,
  automaton_accepts Standard query n dict = true ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros query n dict Haccept.
  assert (Hdist : merge_split_distance query dict <= n).
  { apply Nat.le_trans with (lev_distance query dict).
    - apply ms_le_standard.
    - apply automaton_sound_standard. exact Haccept. }
  destruct (automaton_run_merge_split_completable_from_ms_bound query dict n Hdist)
    as [final [Hrun Hcomplete]].
  unfold automaton_accepts.
  rewrite Hrun.
  eapply automaton_run_merge_split_final_ms_completable_accepts; eauto.
Qed.

(** Similar lemma for MergeAndSplit algorithm.

    For the MergeAndSplit algorithm, positions can include special positions
    that represent merge/split-in-progress states. The proof requires showing
    that the automaton explores all merge/split paths.

    Since position_reachable uses only Standard operations, any Standard-reachable
    position is also reachable in MergeAndSplit. *)
Lemma reachable_final_implies_accepts_merge_split : forall query dict n p,
  position_reachable query n dict p ->
  term_index p = length query ->
  is_special p = false ->
  num_errors p <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros query dict n p Hreach Hfinal Hspec Herr.
  apply standard_accepts_implies_merge_split_accepts.
  apply (reachable_final_implies_accepts query dict n p); assumption.
Qed.

(** * Main Completeness Theorem *)

(** If lev_distance <= n, the automaton accepts for Standard algorithm *)
Theorem automaton_complete_standard : forall
  query dict n,
  lev_distance query dict <= n ->
  automaton_accepts Standard query n dict = true.
Proof.
  intros query dict n Hdist.
  apply automaton_complete_standard_can_complete.
  exact Hdist.
Qed.

(** Transposition completeness using Damerau-Levenshtein distance.

    *** BUG FIXED (2024-12) ***
    The previous bug where non-final positions could subsume final positions
    has been fixed in Subsumption.v. The counterexample (query="abc", dict="ba", n=2)
    now works correctly.

    The Transposition algorithm can perform transposition of adjacent characters
    in addition to standard Levenshtein operations. This means:
    - damerau_lev_distance <= lev_distance (transposition can only help)
    - If damerau_lev_distance <= n, the automaton accepts

    IMPORTANT: The naive strategy "damerau <= n → lev <= n → Standard accepts" is INVALID.
    Since damerau <= lev (not lev <= damerau), we CANNOT derive lev <= n from damerau <= n.
    When transposition helps (damerau < lev), Standard might NOT accept.

    Correct proof strategy requires showing Transposition directly explores Damerau paths:
    1. Define position_reachable_damerau with transposition constructor
    2. Show optimal Damerau edit sequence maps to reachable_damerau path
    3. Show automaton explores transposition via special position mechanism:
       - Enter: (i, e) → (i, e+1)_special when query[i+1] = c
       - Exit: (i, e)_special → (i+2, e) when query[i] = c
       Together: query[i]query[i+1] matched as query[i+1]query[i] at cost 1
*)
Theorem automaton_complete_transposition : forall
  query dict n,
  damerau_lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros query dict n Hdist.
  (* NOTE: The proof strategy "damerau ≤ n → lev ≤ n → Standard accepts" is INVALID
     because damerau ≤ lev (not lev ≤ damerau). We cannot derive lev ≤ n from damerau ≤ n.

     For cases where damerau < lev (transposition helps), Standard might NOT accept,
     but Transposition should. This requires a direct proof showing that Transposition's
     special positions correctly handle transposition operations.

     Proof requirements:
     1. Define Damerau-reachable positions (including transposition transitions)
     2. Show optimal Damerau edit sequence corresponds to reachable path
     3. Show reachable path leads to accepting Transposition state

     Alternative approach: Show Transposition automaton directly explores Damerau paths
     through its special position mechanism:
     - Enter special: (i, e) → (i, e+1)_special when query[i+1] = c (remember query[i+1])
     - Exit special: (i, e)_special → (i+2, e) when query[i] = c (complete transposition)
     Together: query[i]query[i+1] matched as query[i+1]query[i] at cost 1

     This is fundamentally different from Standard which cannot do transposition. *)
  destruct (transposition_reachable_final query dict n Hdist)
    as [p [Hreach [Hterm [Hspec Herr]]]].
  eapply reachable_damerau_final_implies_accepts_transposition; eauto.
Qed.

(** Transposition also accepts strings within standard Levenshtein distance,
    since damerau_lev_distance <= lev_distance. *)
Corollary automaton_complete_transposition_lev : forall
  query dict n,
  lev_distance query dict <= n ->
  automaton_accepts Transposition query n dict = true.
Proof.
  intros query dict n Hdist.
  destruct (automaton_run_transposition_completable_from_lev_bound query dict n Hdist)
    as [final [Hrun Hcomplete]].
  unfold automaton_accepts.
  rewrite Hrun.
  eapply automaton_run_transposition_final_completable_accepts; eauto.
Qed.

(** MergeAndSplit completeness using merge-split distance.

    The MergeAndSplit algorithm can perform merge (2 query chars -> 1 dict char)
    and split (1 query char -> 2 dict chars) in addition to standard operations.
    This means:
    - merge_split_distance <= lev_distance (merge/split can only help)
    - If merge_split_distance <= n, the automaton accepts
*)
Theorem automaton_complete_merge_split : forall
  query dict n,
  merge_split_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros query dict n Hdist.
  destruct (automaton_run_merge_split_completable_from_ms_bound query dict n Hdist)
    as [final [Hrun Hcomplete]].
  unfold automaton_accepts.
  rewrite Hrun.
  eapply automaton_run_merge_split_final_ms_completable_accepts; eauto.
Qed.

(** MergeAndSplit also accepts strings within standard Levenshtein distance,
    since merge_split_distance <= lev_distance. *)
Corollary automaton_complete_merge_split_lev : forall
  query dict n,
  lev_distance query dict <= n ->
  automaton_accepts MergeAndSplit query n dict = true.
Proof.
  intros query dict n Hdist.
  apply automaton_complete_merge_split.
  (* Need: merge_split_distance <= lev_distance *)
  apply Nat.le_trans with (lev_distance query dict).
  - apply ms_le_standard.
  - exact Hdist.
Qed.

(** Unified completeness theorem (using standard Levenshtein as upper bound)

    Since:
    - damerau_lev_distance <= lev_distance
    - merge_split_distance <= lev_distance

    If lev_distance <= n, all algorithms accept.
    This is the "fallback" version using standard Levenshtein distance.
*)
Theorem automaton_complete : forall
  alg query dict n,
  lev_distance query dict <= n ->
  automaton_accepts alg query n dict = true.
Proof.
  intros alg query dict n Hdist.
  destruct alg.
  - apply automaton_complete_standard. exact Hdist.
  - apply automaton_complete_transposition_lev. exact Hdist.
  - apply automaton_complete_merge_split_lev. exact Hdist.
Qed.

(** * Corollaries *)

(** No false negatives: within distance implies accepting *)
Corollary no_false_negatives : forall
  alg query dict n,
  lev_distance query dict <= n ->
  automaton_accepts alg query n dict = true.
Proof.
  exact automaton_complete.
Qed.

(** * Helper Lemmas *)

(** Delete sequence for consuming remaining query characters *)
Lemma delete_sequence_valid : forall query qi,
  qi <= length query ->
  exists ops,
    valid_edit_sequence query [] qi 0 ops /\
    sequence_cost ops = length query - qi.
Proof.
  intros query qi.
  (* Induction on remaining characters to delete *)
  remember (length query - qi) as remaining eqn:Hrem.
  revert qi Hrem.
  induction remaining as [| remaining' IH]; intros qi Hrem Hqi.
  - (* Base case: qi = length query, no more chars to delete *)
    assert (Hqi_eq : qi = length query) by lia.
    exists [].
    split.
    + rewrite Hqi_eq. constructor.
    + simpl. lia.
  - (* Inductive case: delete one char, then continue *)
    assert (Hqi_lt : qi < length query) by lia.
    (* Get the character at position qi *)
    destruct (nth_error query qi) as [c|] eqn:Hnth.
    2: { (* nth_error returns None: contradiction with qi < length query *)
         apply nth_error_None in Hnth. lia. }
    (* Apply IH for qi + 1 *)
    assert (HIH : exists ops', valid_edit_sequence query [] (S qi) 0 ops' /\
                                sequence_cost ops' = remaining').
    { apply IH with (qi := S qi).
      - lia.
      - lia. }
    destruct HIH as [ops' [Hvalid' Hcost']].
    (* Construct the sequence: Edit_Delete c :: ops' *)
    exists (Edit_Delete c :: ops').
    split.
    + apply valid_cons with (qi' := S qi) (di' := 0).
      * (* valid_edit_op_at query [] qi 0 (Edit_Delete c) *)
        unfold valid_edit_op_at. exact Hnth.
      * (* apply_edit_op (Edit_Delete c) qi 0 = (S qi, 0) *)
        simpl. reflexivity.
      * exact Hvalid'.
    + simpl. rewrite Hcost'. lia.
Qed.

(** Insert sequence for consuming remaining dict characters *)
Lemma insert_sequence_valid : forall dict di,
  di <= length dict ->
  exists ops,
    valid_edit_sequence [] dict 0 di ops /\
    sequence_cost ops = length dict - di.
Proof.
  intros dict di.
  (* Induction on remaining characters to insert *)
  remember (length dict - di) as remaining eqn:Hrem.
  revert di Hrem.
  induction remaining as [| remaining' IH]; intros di Hrem Hdi.
  - (* Base case: di = length dict, no more chars to insert *)
    assert (Hdi_eq : di = length dict) by lia.
    exists [].
    split.
    + rewrite Hdi_eq. simpl. constructor.
    + simpl. lia.
  - (* Inductive case: insert one char, then continue *)
    assert (Hdi_lt : di < length dict) by lia.
    (* Get the character at position di *)
    destruct (nth_error dict di) as [c|] eqn:Hnth.
    2: { (* nth_error returns None: contradiction with di < length dict *)
         apply nth_error_None in Hnth. lia. }
    (* Apply IH for di + 1 *)
    assert (HIH : exists ops', valid_edit_sequence [] dict 0 (S di) ops' /\
                                sequence_cost ops' = remaining').
    { apply IH with (di := S di).
      - lia.
      - lia. }
    destruct HIH as [ops' [Hvalid' Hcost']].
    (* Construct the sequence: Edit_Insert c :: ops' *)
    exists (Edit_Insert c :: ops').
    split.
    + apply valid_cons with (qi' := 0) (di' := S di).
      * (* valid_edit_op_at [] dict 0 di (Edit_Insert c) *)
        unfold valid_edit_op_at. exact Hnth.
      * (* apply_edit_op (Edit_Insert c) 0 di = (0, S di) *)
        simpl. reflexivity.
      * exact Hvalid'.
    + simpl. rewrite Hcost'. lia.
Qed.

(** * Properties of Edit Sequences *)

(** Concatenating valid sequences *)
Lemma valid_sequence_concat : forall query dict qi1 di1 qi2 di2 ops1 ops2,
  valid_edit_sequence query dict qi1 di1 ops1 ->
  apply_edit_sequence ops1 qi1 di1 = (qi2, di2) ->
  valid_edit_sequence query dict qi2 di2 ops2 ->
  valid_edit_sequence query dict qi1 di1 (ops1 ++ ops2).
Proof.
  intros query dict qi1 di1 qi2 di2 ops1.
  revert qi1 di1 qi2 di2.
  induction ops1 as [| op ops1' IH]; intros qi1 di1 qi2 di2 ops2 Hvalid1 Happly Hvalid2.
  - (* ops1 = [] *)
    simpl in Happly. inversion Happly. subst.
    simpl. exact Hvalid2.
  - (* ops1 = op :: ops1' *)
    simpl. simpl in Happly.
    inversion Hvalid1 as [| ? ? ? ? qi1' di1' ? ? Hvalid_op Hop Hvalid_rest]. subst.
    rewrite Hop in Happly.
    apply valid_cons with (qi' := qi1') (di' := di1').
    + exact Hvalid_op.
    + exact Hop.
    + apply IH with (qi2 := qi2) (di2 := di2).
      * exact Hvalid_rest.
      * exact Happly.
      * exact Hvalid2.
Qed.

(** Cost of concatenated sequences *)
Lemma sequence_cost_concat : forall ops1 ops2,
  sequence_cost (ops1 ++ ops2) = sequence_cost ops1 + sequence_cost ops2.
Proof.
  intros ops1 ops2.
  induction ops1 as [| op ops1' IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. lia.
Qed.
