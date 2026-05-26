(** * Transition Functions for Levenshtein Automata

    This module defines the transition functions for all three algorithm variants:
    Standard, Transposition, and MergeAndSplit.

    Part of: Liblevenshtein.Core.Automaton

    Rust Reference: src/transducer/transition.rs:176-495

    Key Definitions:
    - transition_standard: Standard Levenshtein transitions
    - transition_transposition: Damerau-Levenshtein transitions
    - transition_merge_split: Merge-and-Split transitions
    - epsilon_closure: Compute all deletion reachable positions
    - transition_state: State-level transition function

    Key Theorems:
    - initial_state_contains_origin
    - transition_preserves_wf
    - transition_generates_successors
*)

From Stdlib Require Import Arith Bool List Nat Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Automaton.Position.
From Liblevenshtein.Core Require Import Automaton.Subsumption.
From Liblevenshtein.Core Require Import Automaton.AntiChain.
From Liblevenshtein.Core Require Import Automaton.CharVector.
From Liblevenshtein.Core Require Import Automaton.State.

(** * Helper Functions *)

(** Get characteristic vector value at offset from current position *)
Definition cv_at (cv : list bool) (offset : nat) : bool :=
  nth offset cv false.

(** If cv_at returns true, the offset must be in bounds *)
Lemma cv_at_true_in_bounds : forall cv offset,
  cv_at cv offset = true -> offset < length cv.
Proof.
  intros cv offset H.
  unfold cv_at in H.
  destruct (le_lt_dec (length cv) offset) as [Hge | Hlt].
  - (* offset >= length cv: contradiction *)
    rewrite nth_overflow in H by exact Hge.
    discriminate.
  - exact Hlt.
Qed.

(** Find the first [true] entry at or after [start], searching at most [limit]
    positions.  The result is the relative offset from [start], matching the
    Rust [index_of_match] helper used by the Standard transition. *)
Fixpoint index_of_match (cv : list bool) (start limit : nat) : option nat :=
  match limit with
  | 0 => None
  | S limit' =>
      if cv_at cv start then Some 0
      else option_map S (index_of_match cv (S start) limit')
  end.

Lemma index_of_match_lt_limit : forall cv start limit j,
  index_of_match cv start limit = Some j ->
  j < limit.
Proof.
  intros cv start limit.
  revert start.
  induction limit as [|limit' IH]; intros start j Hidx.
  - simpl in Hidx. discriminate.
  - simpl in Hidx.
    destruct (cv_at cv start) eqn:Hcv.
    + injection Hidx as Hj. subst j. lia.
    + destruct (index_of_match cv (S start) limit') as [j'|] eqn:Hnext;
        [| discriminate].
      injection Hidx as Hj. subst j.
      pose proof (IH (S start) j' Hnext). lia.
Qed.

Lemma index_of_match_zero_cv_at : forall cv start limit,
  index_of_match cv start limit = Some 0 ->
  cv_at cv start = true.
Proof.
  intros cv start limit Hidx.
  destruct limit as [|limit']; simpl in Hidx; [discriminate|].
  destruct (cv_at cv start) eqn:Hcv.
  - reflexivity.
  - destruct (index_of_match cv (S start) limit'); discriminate.
Qed.

Lemma index_of_match_some_cv_at : forall cv start limit j,
  index_of_match cv start limit = Some j ->
  cv_at cv (start + j) = true.
Proof.
  intros cv start limit.
  revert start.
  induction limit as [|limit' IH]; intros start j Hidx.
  - simpl in Hidx. discriminate.
  - simpl in Hidx.
    destruct (cv_at cv start) eqn:Hcv.
    + injection Hidx as Hj. subst j.
      replace (start + 0) with start by lia. exact Hcv.
    + destruct (index_of_match cv (S start) limit') as [j'|] eqn:Hnext;
        [| discriminate].
      injection Hidx as Hj. subst j.
      replace (start + S j') with (S start + j') by lia.
      apply IH. exact Hnext.
Qed.

Lemma index_of_match_finds_at_or_before : forall cv start limit d,
  d < limit ->
  cv_at cv (start + d) = true ->
  exists j,
    index_of_match cv start limit = Some j /\
    j <= d.
Proof.
  intros cv start limit.
  revert start.
  induction limit as [|limit' IH]; intros start d Hd Hcv.
  - lia.
  - simpl.
    destruct (cv_at cv start) eqn:Hhead.
    + exists 0. split; [reflexivity | lia].
    + destruct d as [|d'].
      * rewrite Nat.add_0_r in Hcv.
        rewrite Hhead in Hcv. discriminate.
      * assert (Hd' : d' < limit') by lia.
        assert (Hcv' : cv_at cv (S start + d') = true).
        { replace (S start + d') with (start + S d') by lia.
          exact Hcv. }
        destruct (IH (S start) d' Hd' Hcv') as [j [Hidx Hj]].
        rewrite Hidx.
        exists (S j). split; [reflexivity | lia].
Qed.

Lemma index_of_match_successor_cv_at_false : forall cv start limit j,
  index_of_match cv start limit = Some (S j) ->
  cv_at cv start = false.
Proof.
  intros cv start limit j Hidx.
  destruct limit as [|limit']; simpl in Hidx; [discriminate|].
  destruct (cv_at cv start) eqn:Hcv.
  - discriminate.
  - reflexivity.
Qed.

Lemma index_of_match_none_head_false : forall cv start limit,
  0 < limit ->
  index_of_match cv start limit = None ->
  cv_at cv start = false.
Proof.
  intros cv start limit Hpos Hidx.
  destruct limit as [|limit']; [lia|].
  simpl in Hidx.
  destruct (cv_at cv start) eqn:Hcv.
  - discriminate.
  - reflexivity.
Qed.

(** Check if we can advance query index (not past query length) *)
Definition can_advance (i qlen : nat) : bool :=
  i <? qlen.

(** Check if we're within max distance *)
Definition within_distance (e n : nat) : bool :=
  e <=? n.

(** * Standard Algorithm Transitions *)

(** For position (i, e), consuming dictionary character with characteristic vector cv:
    - Match: if cv[i - min_i] = true, generate (i+1, e)
    - Substitute: if cv[i - min_i] = false and e < n, generate (i+1, e+1)
    - Insert: generate (i, e+1) (advance dict but not query)
    - Delete: handled via epsilon closure

    Note: min_i is the minimum term_index used when creating the cv.
    The cv is created as characteristic_vector c query min_i window, so
    cv[offset] = char_matches_at c query (min_i + offset).
    For position p with term_index i, we use offset = i - min_i to check
    if c matches query[i].
*)
Definition transition_position_standard (p : Position) (cv : list bool) (min_i n qlen : nat) : list Position :=
  let i := term_index p in
  let e := num_errors p in
  let offset := i - min_i in  (* Offset into cv for this position *)
  (* Only process non-special positions *)
  if is_special p then []
  else
    let candidates :=
      (* Match or Substitute *)
      (if i <? qlen then
        if e <? n then
          let limit := Nat.min (n - e + 1) (length cv - offset) in
          match index_of_match cv offset limit with
          | Some 0 => [std_pos (S i) e]  (* Immediate match *)
          | Some (S j) =>
              [std_pos (S i) (S e);
               std_pos (i + S (S j)) (e + S j)]
          | None => [std_pos (S i) (S e)]
          end
        else if cv_at cv offset then
          [std_pos (S i) e]
        else []
      else [])
      ++
      (* Insert: advance dict but not query *)
      (if e <? n then [std_pos i (S e)] else [])
    in candidates.

(** * Transposition Algorithm Transitions *)

(** For Transposition (Damerau-Levenshtein), we add:
    - Enter special state: (i, e) → (i, e+1)_special if query[i+1] matches
    - Complete transposition: (i, e)_special → (i+2, e) if query[i] matches

    Note: min_i is the minimum term_index used when creating the cv.
    For position with term_index i, cv[i - min_i] gives char_matches_at c query i.
*)
Definition transition_position_transposition (p : Position) (cv : list bool) (min_i n qlen : nat) : list Position :=
  let i := term_index p in
  let e := num_errors p in
  let offset := i - min_i in  (* Base offset for this position *)
  if is_special p then
    (* Special position: complete transposition *)
    (* Check if query[i] matches (offset = i - min_i) *)
    if i + 1 <? qlen then
      if cv_at cv offset then
        [std_pos (S (S i)) e]  (* Complete: advance by 2, keep same errors *)
      else []
    else []
  else
    (* Normal position: standard transitions + enter special *)
    let standard := transition_position_standard p cv min_i n qlen in
    let enter_special :=
      if (i + 1 <? qlen) && (e <? n) then
        (* Check if query[i+1] matches (offset + 1) *)
        if cv_at cv (S offset) then
          [special_pos i (S e)]  (* Enter special state *)
        else []
      else []
    in standard ++ enter_special.

(** * MergeAndSplit Algorithm Transitions *)

(** For MergeAndSplit, we add:
    - Merge: (i, e) → (i+2, e+1) - two query chars matched by one dict char
    - Enter split: (i, e) → (i, e+1)_special
    - Complete split: (i, e)_special → (i+1, e) - one query char matched by two dict chars

    Note: min_i is the minimum term_index used when creating the cv.
*)
Definition transition_position_merge_split (p : Position) (cv : list bool) (min_i n qlen : nat) : list Position :=
  let i := term_index p in
  let e := num_errors p in
  let offset := i - min_i in  (* Offset into cv for this position *)
  if is_special p then
    (* Special position: complete split unconditionally once entered. *)
    if i <? qlen then
      [std_pos (S i) e]
    else []
  else
    (* Normal position: standard transitions + generic merge + enter split *)
    let standard := transition_position_standard p cv min_i n qlen in
    let merge :=
      if (i + 1 <? qlen) && (e <? n) then
        [std_pos (S (S i)) (S e)]
      else []
    in
    let enter_split :=
      if e <? n then
        [special_pos i (S e)]  (* Enter split state *)
      else []
    in standard ++ merge ++ enter_split.

(** * Unified Transition Function *)

Definition transition_position (alg : Algorithm) (p : Position) (cv : list bool) (min_i n qlen : nat) : list Position :=
  match alg with
  | Standard => transition_position_standard p cv min_i n qlen
  | Transposition => transition_position_transposition p cv min_i n qlen
  | MergeAndSplit => transition_position_merge_split p cv min_i n qlen
  end.

(** * Epsilon Closure (Delete Operations) *)

(** Compute positions reachable via delete operations (advance query without consuming dict).
    This is done as a fixpoint until no new positions are found or we exceed max distance. *)

(** Helper to check if a list is empty *)
Definition is_nil {A : Type} (l : list A) : bool :=
  match l with
  | [] => true
  | _ :: _ => false
  end.

(** Single deletion step: advance query index, increment errors *)
Definition delete_step (p : Position) (n qlen : nat) : option Position :=
  let i := term_index p in
  let e := num_errors p in
  if is_special p then None  (* Can't delete from special position *)
  else if (S i <=? qlen) && (e <? n) then
    Some (std_pos (S i) (S e))
  else None.

(** Compute all deletion-reachable positions (bounded by n iterations) *)
Fixpoint epsilon_closure_aux (positions : list Position) (n qlen : nat) (fuel : nat) : list Position :=
  match fuel with
  | 0 => positions
  | S fuel' =>
      let new_positions :=
        flat_map (fun p =>
          match delete_step p n qlen with
          | Some p' => [p']
          | None => []
          end) positions
      in
      if is_nil new_positions then positions
      else epsilon_closure_aux (positions ++ new_positions) n qlen fuel'
  end.

Definition epsilon_closure (positions : list Position) (n qlen : nat) : list Position :=
  epsilon_closure_aux positions n qlen (S n).  (* n+1 iterations is sufficient *)

(** * State-Level Transition *)

(** Transition a state by processing all positions.
    min_i is the minimum term_index used when creating cv, so each position p
    accesses cv at offset (term_index p - min_i) to check for character match. *)
Definition transition_state_positions (alg : Algorithm) (positions : list Position)
                                       (cv : list bool) (min_i n qlen : nat) : list Position :=
  flat_map (fun p => transition_position alg p cv min_i n qlen) positions.

(** Full state transition: compute transitions, apply epsilon closure, build new state *)
Definition transition_state (alg : Algorithm) (s : State) (c : Char) (query : list Char) (n : nat) : option State :=
  let positions := positions s in
  let qlen := query_length s in
  (* Compute characteristic vector based on minimum term_index in state *)
  let min_i := fold_left Nat.min (map term_index positions) qlen in
  (* Window size: 2*n+2 for Standard, 2*n+6 for Transposition (due to larger spread bound) *)
  let window := 2 * n + 6 in
  let cv := characteristic_vector c query min_i window in
  (* Compute transitions - pass min_i so each position uses correct cv offset *)
  let trans_positions := transition_state_positions alg positions cv min_i n qlen in
  (* Apply epsilon closure *)
  let closed_positions := epsilon_closure trans_positions n qlen in
  (* Build new state (antichain_insert handles subsumption) *)
  if is_nil closed_positions then None
  else Some (fold_left (fun s p => state_insert p s)
                       closed_positions
                       (empty_state alg qlen)).

(** * Properties *)

(** Standard transition preserves bounds when input is bounded *)
Lemma transition_standard_bounded : forall p cv min_i n qlen p',
  num_errors p <= n ->
  In p' (transition_position_standard p cv min_i n qlen) ->
  num_errors p' <= n.
Proof.
  intros p cv min_i n qlen p' Hbound Hin.
  unfold transition_position_standard in Hin.
  destruct (is_special p) eqn:Hspec.
  - simpl in Hin. contradiction.
  - apply in_app_or in Hin.
    destruct Hin as [Hin | Hin].
    + (* Match or Substitute *)
      destruct (term_index p <? qlen) eqn:Hlt; simpl in Hin; try contradiction.
      destruct (num_errors p <? n) eqn:He.
      * set (limit := Nat.min (n - num_errors p + 1)
                            (length cv - (term_index p - min_i))) in *.
        destruct (index_of_match cv (term_index p - min_i) limit) as [[|j]|] eqn:Hidx;
          simpl in Hin.
        -- destruct Hin as [Heq | []]. subst. simpl. exact Hbound.
        -- destruct Hin as [Heq | [Heq | []]]; subst; simpl.
           ++ apply Nat.ltb_lt in He. lia.
           ++ pose proof (index_of_match_lt_limit cv (term_index p - min_i)
                            limit (S j) Hidx) as Hj_lt.
              unfold limit in Hj_lt.
              apply Nat.ltb_lt in He. lia.
        -- destruct Hin as [Heq | []]. subst. simpl.
           apply Nat.ltb_lt in He. lia.
      * destruct (cv_at cv (term_index p - min_i)) eqn:Hcv; simpl in Hin;
          try contradiction.
        destruct Hin as [Heq | []]. subst. simpl. exact Hbound.
    + (* Insert *)
      destruct (num_errors p <? n) eqn:He; simpl in Hin; try contradiction.
      destruct Hin as [Heq | Hin]; try contradiction.
      subst. simpl.
      apply Nat.ltb_lt in He. lia.
Qed.

(** Delete step preserves bounds *)
Lemma delete_step_bounded : forall p n qlen p',
  num_errors p <= n ->
  delete_step p n qlen = Some p' ->
  num_errors p' <= n.
Proof.
  intros p n qlen p' Hbound Hdel.
  unfold delete_step in Hdel.
  destruct (is_special p) eqn:Hspec; try discriminate.
  destruct ((S (term_index p) <=? qlen) && (num_errors p <? n)) eqn:Hcond; try discriminate.
  inversion Hdel. subst. simpl.
  apply andb_prop in Hcond. destruct Hcond as [_ He].
  apply Nat.ltb_lt in He. lia.
Qed.

(** Epsilon closure aux preserves error bounds *)
Lemma epsilon_closure_aux_bounded : forall fuel positions n qlen p,
  (forall p0, In p0 positions -> num_errors p0 <= n) ->
  In p (epsilon_closure_aux positions n qlen fuel) ->
  num_errors p <= n.
Proof.
  induction fuel as [| fuel' IH]; intros positions n qlen p Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    destruct (is_nil (flat_map _ positions)) eqn:Hnil.
    + apply Hall. exact Hin.
    + apply IH in Hin.
      * exact Hin.
      * intros p0 Hin0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall. exact Hin0.
        -- rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           destruct (delete_step p1 n qlen) as [p'|] eqn:Hdel.
           ++ simpl in Hin0. destruct Hin0 as [Heq | Hin0]; try contradiction.
              subst. apply delete_step_bounded with (p := p1) (qlen := qlen).
              ** apply Hall. exact Hin1.
              ** exact Hdel.
           ++ simpl in Hin0. contradiction.
Qed.

(** Epsilon closure preserves error bounds *)
Lemma epsilon_closure_bounded : forall positions n qlen p,
  (forall p0, In p0 positions -> num_errors p0 <= n) ->
  In p (epsilon_closure positions n qlen) ->
  num_errors p <= n.
Proof.
  intros positions n qlen p Hall Hin.
  unfold epsilon_closure in Hin.
  apply epsilon_closure_aux_bounded with (positions := positions) (fuel := S n) (qlen := qlen).
  - exact Hall.
  - exact Hin.
Qed.

(** For epsilon closure starting from std_pos 0 0, all positions have term_index = num_errors.
    This is because delete_step from (i, e) produces (i+1, e+1), preserving the invariant. *)
Lemma epsilon_closure_from_origin_term_eq_errors_aux :
  forall fuel positions n qlen p,
    (forall p0, In p0 positions -> term_index p0 = num_errors p0) ->
    In p (epsilon_closure_aux positions n qlen fuel) ->
    term_index p = num_errors p.
Proof.
  induction fuel as [| fuel' IH]; intros positions n qlen p Hinv Hin.
  - (* fuel = 0 *)
    simpl in Hin. apply Hinv. exact Hin.
  - (* fuel = S fuel' *)
    simpl in Hin.
    set (new := flat_map (fun p0 => match delete_step p0 n qlen with
                                    | Some p' => [p']
                                    | None => []
                                    end) positions) in *.
    destruct (is_nil new) eqn:Hnil.
    + (* new is empty, positions unchanged *)
      apply Hinv. exact Hin.
    + (* new is non-empty, recurse *)
      apply IH with (positions := positions ++ new) (n := n) (qlen := qlen); auto.
      intros p0 Hp0. apply in_app_or in Hp0. destruct Hp0 as [Hp0 | Hp0].
      * (* p0 in original positions *)
        apply Hinv. exact Hp0.
      * (* p0 in new, produced by delete_step *)
        unfold new in Hp0.
        apply in_flat_map in Hp0.
        destruct Hp0 as [p1 [Hp1 Hdel]].
        destruct (delete_step p1 n qlen) as [p'|] eqn:Hdel_eq.
        -- (* delete_step p1 = Some p' *)
           simpl in Hdel. destruct Hdel as [Heq | []].
           subst p0.
           unfold delete_step in Hdel_eq.
           destruct (is_special p1) eqn:Hspec.
           ++ discriminate.
           ++ destruct ((S (term_index p1) <=? qlen) && (num_errors p1 <? n)) eqn:Hcond.
              ** injection Hdel_eq as Heq.
                 rewrite <- Heq. simpl.
                 (* term_index (std_pos (S i) (S e)) = S i, num_errors = S e *)
                 (* By Hinv, term_index p1 = num_errors p1 *)
                 specialize (Hinv p1 Hp1). lia.
              ** discriminate.
        -- simpl in Hdel. contradiction.
Qed.

Lemma epsilon_closure_from_origin_term_eq_errors :
  forall n qlen p,
    In p (epsilon_closure [std_pos 0 0] n qlen) ->
    term_index p = num_errors p.
Proof.
  intros n qlen p Hin.
  unfold epsilon_closure in Hin.
  apply epsilon_closure_from_origin_term_eq_errors_aux with
      (fuel := S n) (positions := [std_pos 0 0]) (n := n) (qlen := qlen); auto.
  intros p0 Hp0. simpl in Hp0. destruct Hp0 as [Heq | []].
  subst p0. simpl. reflexivity.
Qed.

(** Corollary: term_index is bounded by n for epsilon_closure from origin *)
Lemma epsilon_closure_from_origin_term_bounded :
  forall n qlen p,
    In p (epsilon_closure [std_pos 0 0] n qlen) ->
    term_index p <= n.
Proof.
  intros n qlen p Hin.
  assert (Heq : term_index p = num_errors p).
  { apply epsilon_closure_from_origin_term_eq_errors with (n := n) (qlen := qlen). exact Hin. }
  rewrite Heq.
  apply epsilon_closure_bounded with (positions := [std_pos 0 0]) (n := n) (qlen := qlen); auto.
  intros p0 Hp0. simpl in Hp0. destruct Hp0 as [Heq0 | []].
  subst p0. simpl. lia.
Qed.

(** Helper: std_pos 0 0 is always in the epsilon closure from [std_pos 0 0].
    This is because the original position is always preserved. *)
Lemma epsilon_closure_aux_preserves_original :
  forall fuel positions n qlen p,
    In p positions ->
    In p (epsilon_closure_aux positions n qlen fuel).
Proof.
  induction fuel as [| fuel' IH]; intros positions n qlen p Hin.
  - simpl. exact Hin.
  - simpl.
    set (new := flat_map (fun p0 : Position =>
                            match delete_step p0 n qlen with
                            | Some p' => [p']
                            | None => []
                            end) positions).
    destruct (is_nil new).
    + exact Hin.
    + apply IH. apply in_or_app. left. exact Hin.
Qed.

Lemma std_pos_0_0_in_epsilon_closure :
  forall n qlen,
    In (std_pos 0 0) (epsilon_closure [std_pos 0 0] n qlen).
Proof.
  intros n qlen.
  unfold epsilon_closure.
  apply epsilon_closure_aux_preserves_original.
  simpl. left. reflexivity.
Qed.

(** Helper: fold_left Nat.min over a list containing 0 returns 0 *)
Lemma fold_left_min_contains_zero :
  forall (l : list nat) acc,
    In 0 l \/ acc = 0 ->
    fold_left Nat.min l acc = 0.
Proof.
  induction l as [| h t IH]; intros acc Hzero; simpl.
  - destruct Hzero as [[] | Hacc]. exact Hacc.
  - apply IH.
    destruct Hzero as [Hin | Hacc].
    + destruct Hin as [Hh | Ht].
      * subst h. right. lia.
      * left. exact Ht.
    + subst acc. right. lia.
Qed.

(** The minimum term_index in epsilon_closure from [std_pos 0 0] is 0 *)
Lemma epsilon_closure_from_origin_min_is_zero :
  forall n qlen,
    fold_left Nat.min (map term_index (epsilon_closure [std_pos 0 0] n qlen)) qlen = 0.
Proof.
  intros n qlen.
  (* std_pos 0 0 is always in the epsilon closure with term_index 0 *)
  pose proof (std_pos_0_0_in_epsilon_closure n qlen) as Hin.
  (* Therefore 0 is in the map of term_indices *)
  assert (Hzero : In 0 (map term_index (epsilon_closure [std_pos 0 0] n qlen))).
  { apply in_map_iff. exists (std_pos 0 0). split.
    - reflexivity.
    - exact Hin. }
  (* fold_left Nat.min over a list containing 0 returns 0 *)
  apply fold_left_min_contains_zero.
  left. exact Hzero.
Qed.

(** * Algorithm Inclusion Lemmas *)

(** Transposition includes all Standard transitions for non-special positions.
    This is because transition_position_transposition for non-special positions
    computes standard ++ enter_special, so standard positions are included. *)
Lemma transposition_includes_standard : forall p cv min_i n qlen,
  is_special p = false ->
  incl (transition_position_standard p cv min_i n qlen)
       (transition_position_transposition p cv min_i n qlen).
Proof.
  intros p cv min_i n qlen Hspec.
  unfold transition_position_transposition.
  rewrite Hspec.
  intros x Hin.
  apply in_or_app. left. exact Hin.
Qed.

(** MergeAndSplit includes all Standard transitions for non-special positions.
    This is because transition_position_merge_split for non-special positions
    computes standard ++ merge ++ enter_split, so standard positions are included. *)
Lemma merge_split_includes_standard : forall p cv min_i n qlen,
  is_special p = false ->
  incl (transition_position_standard p cv min_i n qlen)
       (transition_position_merge_split p cv min_i n qlen).
Proof.
  intros p cv min_i n qlen Hspec.
  unfold transition_position_merge_split.
  rewrite Hspec.
  intros x Hin.
  apply in_or_app. left. exact Hin.
Qed.

(** Unified inclusion: extended algorithms include Standard for non-special positions *)
Lemma extended_includes_standard : forall alg p cv min_i n qlen,
  alg <> Standard ->
  is_special p = false ->
  incl (transition_position_standard p cv min_i n qlen)
       (transition_position alg p cv min_i n qlen).
Proof.
  intros alg p cv min_i n qlen Halg Hspec.
  destruct alg.
  - (* Standard - contradicts Halg *)
    contradiction.
  - (* Transposition *)
    apply transposition_includes_standard. exact Hspec.
  - (* MergeAndSplit *)
    apply merge_split_includes_standard. exact Hspec.
Qed.

(** * Initial State Correctness *)

(** The initial state (0, 0) is correct starting point *)
Lemma initial_position_valid : forall qlen,
  qlen > 0 ->
  term_index initial_position = 0 /\
  num_errors initial_position = 0.
Proof.
  intros qlen Hqlen.
  unfold initial_position, std_pos. simpl.
  split; reflexivity.
Qed.

(** * Examples *)

Example transition_standard_match :
  (* Position (0,0) with matching char at position 0 gives (1,0) *)
  (* min_i = 0, so offset = 0 - 0 = 0, cv[0] = true *)
  In (std_pos 1 0) (transition_position_standard (std_pos 0 0) [true] 0 2 5).
Proof.
  unfold transition_position_standard, cv_at, std_pos. simpl.
  left. reflexivity.
Qed.

Example transition_standard_substitute :
  (* Position (0,0) with non-matching char gives (1,1) *)
  (* min_i = 0, so offset = 0 - 0 = 0, cv[0] = false *)
  In (std_pos 1 1) (transition_position_standard (std_pos 0 0) [false] 0 2 5).
Proof.
  unfold transition_position_standard, cv_at, std_pos. simpl.
  left. reflexivity.
Qed.

Example transition_standard_insert :
  (* Position (0,0) can also give (0,1) via insert *)
  (* min_i = 0, so offset = 0 - 0 = 0, cv[0] = false (insert doesn't check cv) *)
  In (std_pos 0 1) (transition_position_standard (std_pos 0 0) [false] 0 2 5).
Proof.
  unfold transition_position_standard, cv_at, std_pos. simpl.
  right. left. reflexivity.
Qed.

(** * General Transition Correspondence Lemmas *)

(** Standard transition produces match result when CV indicates match *)
Lemma transition_standard_produces_match : forall i e cv min_i n qlen,
  i < qlen ->
  min_i <= i ->
  cv_at cv (i - min_i) = true ->
  In (std_pos (S i) e) (transition_position_standard (std_pos i e) cv min_i n qlen).
Proof.
  intros i e cv min_i n qlen Hi_lt Hmin_le Hcv.
  unfold transition_position_standard, std_pos. simpl.
  (* i <? qlen = true by Hi_lt *)
  assert (Hi_lt_b : (i <? qlen) = true) by (apply Nat.ltb_lt; exact Hi_lt).
  rewrite Hi_lt_b.
  destruct (e <? n) eqn:He_lt.
  - set (limit := Nat.min (n - e + 1) (length cv - (i - min_i))).
    assert (Hlimit_pos : 0 < limit).
    { unfold limit.
      pose proof (cv_at_true_in_bounds cv (i - min_i) Hcv) as Hoff.
      apply Nat.ltb_lt in He_lt. lia. }
    destruct limit as [|limit']; [lia|].
    simpl. rewrite Hcv. left. reflexivity.
  - rewrite Hcv. left. reflexivity.
Qed.

(** If the Standard lookahead search finds a match at relative offset [j],
    the executable transition emits the skip-to-match successor. The [j = 0]
    case is the ordinary match transition. *)
Lemma transition_standard_produces_index_match : forall i e cv min_i n qlen j,
  i < qlen ->
  e < n ->
  index_of_match cv (i - min_i)
    (Nat.min (n - e + 1) (length cv - (i - min_i))) = Some j ->
  In (std_pos (S (i + j)) (e + j))
     (transition_position_standard (std_pos i e) cv min_i n qlen).
Proof.
  intros i e cv min_i n qlen j Hi_lt He_lt Hidx.
  unfold transition_position_standard, std_pos. simpl.
  assert (Hi_lt_b : (i <? qlen) = true) by (apply Nat.ltb_lt; exact Hi_lt).
  rewrite Hi_lt_b.
  assert (He_lt_b : (e <? n) = true) by (apply Nat.ltb_lt; exact He_lt).
  rewrite He_lt_b.
  rewrite Hidx.
  destruct j as [|j'].
  - simpl. replace (S (i + 0)) with (S i) by lia.
    replace (e + 0) with e by lia.
    left. reflexivity.
  - simpl. replace (S (i + S j')) with (i + S (S j')) by lia.
    replace (e + S j') with (e + S j') by lia.
    right. left. reflexivity.
Qed.

(** Standard transition produces substitute result when CV indicates mismatch and e < n *)
Lemma transition_standard_produces_substitute : forall i e cv min_i n qlen,
  i < qlen ->
  min_i <= i ->
  cv_at cv (i - min_i) = false ->
  e < n ->
  In (std_pos (S i) (S e)) (transition_position_standard (std_pos i e) cv min_i n qlen).
Proof.
  intros i e cv min_i n qlen Hi_lt Hmin_le Hcv He_lt.
  unfold transition_position_standard, std_pos. simpl.
  assert (Hi_lt_b : (i <? qlen) = true) by (apply Nat.ltb_lt; exact Hi_lt).
  assert (He_lt_b : (e <? n) = true) by (apply Nat.ltb_lt; exact He_lt).
  rewrite Hi_lt_b, He_lt_b.
  set (limit := Nat.min (n - e + 1) (length cv - (i - min_i))).
  destruct (index_of_match cv (i - min_i) limit) as [[|j]|] eqn:Hidx.
  - pose proof (index_of_match_zero_cv_at cv (i - min_i) limit Hidx) as Hzero.
    rewrite Hcv in Hzero. discriminate.
  - simpl. left. reflexivity.
  - simpl. left. reflexivity.
Qed.

(** Standard transition produces insert result when e < n *)
Lemma transition_standard_produces_insert : forall i e cv min_i n qlen,
  e < n ->
  In (std_pos i (S e)) (transition_position_standard (std_pos i e) cv min_i n qlen).
Proof.
  intros i e cv min_i n qlen He_lt.
  unfold transition_position_standard, std_pos. simpl.
  assert (He_lt_b : (e <? n) = true) by (apply Nat.ltb_lt; exact He_lt).
  destruct (i <? qlen) eqn:Hi_lt.
  - (* i < qlen *)
    rewrite He_lt_b.
    destruct (index_of_match cv (i - min_i)
                (Nat.min (n - e + 1) (length cv - (i - min_i)))) as [[|j]|] eqn:Hidx;
      simpl.
    + right. left. reflexivity.
    + right. right. left. reflexivity.
    + right. left. reflexivity.
  - (* i >= qlen *)
    (* Insert is only element in [] ++ [insert] *)
    rewrite He_lt_b. left. reflexivity.
Qed.
