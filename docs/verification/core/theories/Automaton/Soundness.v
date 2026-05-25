(** * Soundness of Levenshtein Automata

    This module proves that if the automaton accepts a dictionary word
    with distance d, then the actual Levenshtein distance is <= d.
    This is the "soundness" direction of correctness.

    Part of: Liblevenshtein.Core.Automaton

    Key Theorem:
    - automaton_sound: accepts(query, n, dict) -> lev_distance(query, dict) <= n

    Proof Strategy:
    - Define position invariant: (i, e) means we've consumed i query chars
      with e errors while matching a prefix of dict
    - Show transitions preserve this invariant
    - Show accepting means we've matched the entire query with errors <= n
*)

From Stdlib Require Import Arith Bool List Nat Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.MinLemmas.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.
From Liblevenshtein.Core Require Import Automaton.Position.
From Liblevenshtein.Core Require Import Automaton.CharVector.
From Liblevenshtein.Core Require Import Automaton.State.
From Liblevenshtein.Core Require Import Automaton.Transition.
From Liblevenshtein.Core Require Import Automaton.AntiChain.
From Liblevenshtein.Core Require Import Automaton.Acceptance.
From Liblevenshtein.Core Require Import DPMatrix.SnocLemmas.

(** * Position Invariant *)

(** A position (i, e, s) in state after consuming dict prefix dp represents:
    - We've aligned query[0..i-1] with some portion of dp
    - The alignment has cost e (number of errors)
    - If s=true, we're in a special state for transposition/split *)

(** Position reachability predicate:
    reachable query dict_prefix n p means position p can be reached
    by processing dict_prefix from the initial state. *)
Inductive position_reachable (query : list Char) (n : nat) :
  list Char -> Position -> Prop :=
  | reach_initial :
      position_reachable query n [] initial_position
  | reach_delete : forall dp i e,
      position_reachable query n dp (std_pos i e) ->
      S i <= length query ->
      e < n ->
      position_reachable query n dp (std_pos (S i) (S e))
  | reach_match : forall dp c i e,
      position_reachable query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c ->
      position_reachable query n (dp ++ [c]) (std_pos (S i) e)
  | reach_substitute : forall dp c c' i e,
      position_reachable query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c' ->
      c <> c' ->
      e < n ->
      position_reachable query n (dp ++ [c]) (std_pos (S i) (S e))
  | reach_insert : forall dp c i e,
      position_reachable query n dp (std_pos i e) ->
      e < n ->
      position_reachable query n (dp ++ [c]) (std_pos i (S e)).

(** * Damerau-Levenshtein Position Reachability

    Extends position_reachable with transposition operations.
    Transposition is modeled as a two-step process:
    1. Enter special state when we see a dict char that matches query[i+1]
    2. Complete transposition when we see the next dict char matching query[i]

    This captures the Damerau-Levenshtein transposition at cost 1.
*)
Inductive position_reachable_damerau (query : list Char) (n : nat) :
  list Char -> Position -> Prop :=
  | reach_damerau_initial :
      position_reachable_damerau query n [] initial_position
  | reach_damerau_delete : forall dp i e,
      position_reachable_damerau query n dp (std_pos i e) ->
      S i <= length query ->
      e < n ->
      position_reachable_damerau query n dp (std_pos (S i) (S e))
  | reach_damerau_match : forall dp c i e,
      position_reachable_damerau query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c ->
      position_reachable_damerau query n (dp ++ [c]) (std_pos (S i) e)
  | reach_damerau_substitute : forall dp c c' i e,
      position_reachable_damerau query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c' ->
      c <> c' ->
      e < n ->
      position_reachable_damerau query n (dp ++ [c]) (std_pos (S i) (S e))
  | reach_damerau_insert : forall dp c i e,
      position_reachable_damerau query n dp (std_pos i e) ->
      e < n ->
      position_reachable_damerau query n (dp ++ [c]) (std_pos i (S e))
  | reach_damerau_enter_transpose : forall dp c c_next i e,
      (* From (i, e) non-special, see dict char c that matches query[i+1] *)
      position_reachable_damerau query n dp (std_pos i e) ->
      S i < length query ->  (* Need at least 2 more query chars *)
      nth_error query (S i) = Some c ->  (* c matches query[i+1] *)
      nth_error query i = Some c_next ->  (* save query[i] for later check *)
      e < n ->
      position_reachable_damerau query n (dp ++ [c]) (special_pos i (S e))
  | reach_damerau_complete_transpose : forall dp c i e,
      (* From (i, e) special, see dict char c that matches query[i] *)
      position_reachable_damerau query n dp (special_pos i e) ->
      i < length query ->
      nth_error query i = Some c ->  (* c matches query[i], completing swap *)
      position_reachable_damerau query n (dp ++ [c]) (std_pos (S (S i)) e).

(** MergeSplit reachability.

    Special positions represent a split-in-progress: the first output
    dictionary character has been consumed and the split cost has already been
    paid; the next transition may consume the second output character and
    advance the query by one. *)
Inductive position_reachable_merge_split (query : list Char) (n : nat) :
  list Char -> Position -> Prop :=
  | reach_ms_initial :
      position_reachable_merge_split query n [] initial_position
  | reach_ms_delete : forall dp i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      S i <= length query ->
      e < n ->
      position_reachable_merge_split query n dp (std_pos (S i) (S e))
  | reach_ms_match : forall dp c i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c ->
      position_reachable_merge_split query n (dp ++ [c]) (std_pos (S i) e)
  | reach_ms_substitute : forall dp c c' i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      i < length query ->
      nth_error query i = Some c' ->
      e < n ->
      position_reachable_merge_split query n (dp ++ [c]) (std_pos (S i) (S e))
  | reach_ms_insert : forall dp c i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      e < n ->
      position_reachable_merge_split query n (dp ++ [c]) (std_pos i (S e))
  | reach_ms_merge : forall dp c i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      S i < length query ->
      e < n ->
      position_reachable_merge_split query n (dp ++ [c]) (std_pos (S (S i)) (S e))
  | reach_ms_enter_split : forall dp c i e,
      position_reachable_merge_split query n dp (std_pos i e) ->
      e < n ->
      position_reachable_merge_split query n (dp ++ [c]) (special_pos i (S e))
  | reach_ms_complete_split : forall dp c i e,
      position_reachable_merge_split query n dp (special_pos i e) ->
      i < length query ->
      position_reachable_merge_split query n (dp ++ [c]) (std_pos (S i) e).

(** * Special Position Tracking Infrastructure

    Special positions (from enter_transpose) need to be tracked to prove
    soundness. The key insight is that special positions ALWAYS originate
    from non-special positions via enter_transpose, and the transition that
    creates them records the necessary information for complete_transpose.

    We define an invariant that tracks this origin relationship. *)

(** Definition: A special position originated from a non-special position. *)
Definition enters_special_from (p_src p_dst : Position) : Prop :=
  is_special p_src = false /\
  is_special p_dst = true /\
  term_index p_dst = term_index p_src /\
  num_errors p_dst = S (num_errors p_src).

(** Invariant: All special positions in a state originated from non-special.
    This captures that special positions were created via enter_transpose. *)
Definition special_positions_originated (positions : list Position) : Prop :=
  forall p, In p positions ->
    is_special p = true ->
    exists p', is_special p' = false /\
               num_errors p' < num_errors p /\
               term_index p' = term_index p.

Lemma initial_state_no_special : forall qlen,
  Forall (fun p => is_special p = false) (positions (initial_state Transposition qlen)).
Proof.
  intros qlen.
  unfold initial_state, initial_position, std_pos.
  simpl. constructor; [reflexivity | constructor].
Qed.

(** * MergeAndSplit Soundness Contracts

    The MergeAndSplit algorithm uses specialized operations for character merging
    (two query chars map to one dict char) and splitting (one query char maps to
    two dict chars).

    These contracts capture the soundness property: if the automaton accepts,
    the actual merge-split distance is bounded by the acceptance threshold. *)

(** * Damerau Reachability Implies Standard Reachability

    Since every DL operation sequence corresponds to a standard sequence
    (with transpositions replaced by 2 ops), we can convert reachability.
    However, the error bound increases because transpositions cost 1 in DL
    but 2 in standard Levenshtein. *)

(** Damerau positions have bounded errors *)
Lemma reachable_damerau_implies_edit_distance : forall query n dp p,
  position_reachable_damerau query n dp p ->
  num_errors p <= n.
Proof.
  intros query n dp p Hreach.
  induction Hreach.
  - (* initial *) unfold initial_position, std_pos. simpl. lia.
  - (* delete *) simpl. lia.
  - (* match *) simpl. apply IHHreach.
  - (* substitute *) simpl. lia.
  - (* insert *) simpl. lia.
  - (* enter transpose *) simpl. lia.
  - (* complete transpose *) simpl. apply IHHreach.
Qed.

(** * Damerau Reachability to Distance Bound

    Key lemma for transposition soundness: if we can reach a final position
    (term_index = length query) via position_reachable_damerau with error
    count e, then damerau_lev_distance(query, dict_prefix) <= e.

    The proof follows by induction on the reachability derivation:
    - reach_damerau_initial: Both empty, distance 0
    - reach_damerau_delete: Advances query index (deletion)
    - reach_damerau_match: Character match
    - reach_damerau_substitute: Character mismatch (substitution)
    - reach_damerau_insert: Insert dict char (insertion)
    - reach_damerau_enter_transpose: Enter transposition state
    - reach_damerau_complete_transpose: Complete transposition

    Each constructor corresponds to a valid Damerau-Levenshtein edit operation.
*)

(** Helper: adding a char to s1 increases lev distance by at most 1.
    Used to prove the length bound. *)
Lemma lev_distance_delete_bound_early : forall c s1 s2,
  lev_distance (c :: s1) s2 <= lev_distance s1 s2 + 1.
Proof.
  intros c s1 s2.
  destruct s2 as [| d s2'].
  - rewrite lev_distance_empty_right.
    rewrite lev_distance_empty_right.
    simpl. lia.
  - rewrite lev_distance_cons.
    unfold min3.
    apply Nat.le_min_l.
Qed.

(** Helper: Levenshtein distance is bounded by sum of lengths. *)
Lemma lev_distance_length_bound_early : forall s1 s2,
  lev_distance s1 s2 <= length s1 + length s2.
Proof.
  induction s1 as [| c s1' IH].
  - intros s2. rewrite lev_distance_empty_left. lia.
  - intros s2.
    apply Nat.le_trans with (lev_distance s1' s2 + 1).
    + apply lev_distance_delete_bound_early.
    + specialize (IH s2). simpl length. lia.
Qed.

(** Helper: Damerau-Levenshtein distance bounds by string lengths.
    Follows from damerau_lev_le_standard and the standard length bound. *)
Lemma damerau_lev_distance_length_bound : forall s1 s2,
  damerau_lev_distance s1 s2 <= length s1 + length s2.
Proof.
  intros s1 s2.
  (* Use transitivity: damerau_lev <= lev <= |s1| + |s2| *)
  apply Nat.le_trans with (lev_distance s1 s2).
  - apply damerau_lev_le_standard.
  - apply lev_distance_length_bound_early.
Qed.

(** * Distance Bound from Position *)

(** Key lemma: if we can reach position (i, e), then we can edit
    query[0..i-1] into the dictionary prefix with cost e *)
Lemma reachable_implies_edit_distance : forall query dict_prefix n p,
  position_reachable query n dict_prefix p ->
  is_special p = false ->
  num_errors p <= n.
Proof.
  intros query dict_prefix n p Hreach Hspec.
  induction Hreach; simpl in *.
  - (* initial *) lia.
  - (* delete *) lia.
  - (* match *) apply IHHreach. exact Hspec.
  - (* substitute *) lia.
  - (* insert *) lia.
Qed.

(** * Key Soundness Lemma: Reachability to Distance *)

(** The central lemma for soundness: if we can reach a final position (i.e.,
    term_index = query_length) after processing dictionary prefix dp with
    error count e, then lev_distance(query, dp) <= e.

    This is proved by induction on the position_reachable derivation:
    - reach_initial: Both query and dp are empty, distance is 0
    - reach_delete: Advances query index with +1 error (deletion)
    - reach_match: Character match, no error increase
    - reach_substitute: Character mismatch with +1 error (substitution)
    - reach_insert: Insert dict char with +1 error (insertion)

    The key insight is that each constructor corresponds to a valid edit operation,
    and the accumulated errors form an upper bound on the edit distance. *)

(** * Generalized Soundness: Partial Prefix Approach *)

(** The key insight is that position_reachable maintains an invariant about
    partial alignments. When we reach position (i, e) with dictionary prefix dp,
    the edit distance from the first i characters of query to dp is <= e.

    We use firstn and related lemmas from DPMatrix.SnocLemmas. *)

(** Helper for term_index bounds *)
Lemma term_index_bound : forall (query : list Char) (i : nat),
  i < length query ->
  exists c : Char, nth_error query i = Some c.
Proof.
  intros query i Hi.
  destruct (nth_error query i) as [c|] eqn:Hnth.
  - exists c. reflexivity.
  - apply nth_error_None in Hnth. lia.
Qed.

(** Key lemma: firstn (S i) query = firstn i query ++ [query[i]] when i < length query
    This version uses nth_error to specify the element at position i. *)
Lemma firstn_S_snoc_nth_error : forall {A : Type} (l : list A) (i : nat) (x : A),
  nth_error l i = Some x ->
  firstn (S i) l = firstn i l ++ [x].
Proof.
  intros A l.
  induction l as [| h t IH]; intros i x Hnth.
  - (* l = [] - contradiction since nth_error [] i = None *)
    destruct i; discriminate.
  - (* l = h :: t *)
    destruct i as [| i'].
    + (* i = 0 *)
      simpl in Hnth. inversion Hnth. subst.
      simpl. reflexivity.
    + (* i = S i' *)
      simpl in Hnth. simpl.
      f_equal. apply IH. exact Hnth.
Qed.

(** * Levenshtein Distance Monotonicity Lemmas *)

(** Deletion bound: adding a char to s1 increases distance by at most 1 *)
Lemma lev_distance_delete_bound : forall c s1 s2,
  lev_distance (c :: s1) s2 <= lev_distance s1 s2 + 1.
Proof.
  intros c s1 s2.
  destruct s2 as [| d s2'].
  - rewrite lev_distance_empty_right.
    rewrite lev_distance_empty_right.
    simpl. lia.
  - rewrite lev_distance_cons.
    unfold min3.
    apply Nat.le_min_l.
Qed.

(** Deletion from query: adding char to END of s1 increases distance by at most 1.
    Proved using strong induction on |s1| + |s2|.
    Key insight: Each branch of the min3 recurrence satisfies the bound by IH. *)
Lemma lev_distance_snoc_left_bound_aux : forall n s1 c s2,
  length s1 + length s2 <= n ->
  lev_distance (s1 ++ [c]) s2 <= lev_distance s1 s2 + 1.
Proof.
  intro n. induction n as [n IH] using lt_wf_ind.
  intros s1 c s2 Hlen.
  destruct s1 as [| d s1']; destruct s2 as [| e s2'].
  - (* s1 = [], s2 = [] *)
    (* Goal: lev_distance [c] [] <= lev_distance [] [] + 1 *)
    (* lev_distance [c] [] = 1, lev_distance [] [] = 0, so 1 <= 0 + 1 = 1 *)
    rewrite !lev_distance_empty_right. simpl length. lia.
  - (* s1 = [], s2 = e :: s2' *)
    simpl. apply lev_distance_delete_bound.
  - (* s1 = d :: s1', s2 = [] *)
    (* Goal: lev_distance (d :: s1' ++ [c]) [] <= lev_distance (d :: s1') [] + 1 *)
    rewrite !lev_distance_empty_right.
    simpl app. simpl length. rewrite length_app. simpl length. lia.
  - (* s1 = d :: s1', s2 = e :: s2' - main inductive case *)
    simpl app.
    rewrite (lev_distance_cons d e (s1' ++ [c]) s2').
    rewrite (lev_distance_cons d e s1' s2').
    unfold min3.
    (* IH on deletion branch *)
    assert (IH1: lev_distance (s1' ++ [c]) (e :: s2') <= lev_distance s1' (e :: s2') + 1).
    { apply (IH (length s1' + S (length s2'))).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* IH on insertion branch *)
    assert (IH2: lev_distance (d :: (s1' ++ [c])) s2' <= lev_distance (d :: s1') s2' + 1).
    { change (d :: (s1' ++ [c])) with ((d :: s1') ++ [c]).
      apply (IH (S (length s1') + length s2')).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* IH on diagonal branch *)
    assert (IH3: lev_distance (s1' ++ [c]) s2' <= lev_distance s1' s2' + 1).
    { apply (IH (length s1' + length s2')).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* Now show min3 LHS <= min3 RHS + 1 *)
    (* LHS branches: del_L = IH1 arg + 1, ins_L = IH2 arg + 1, diag_L = IH3 arg + subst_cost *)
    (* RHS branches: del_R = lev_distance s1' (e :: s2') + 1, ins_R = ..., diag_R = ... *)
    (* Each LHS branch is <= corresponding RHS branch + 1 *)
    (* del_L <= del_R + 1 by IH1 *)
    (* ins_L <= ins_R + 1 by IH2 *)
    (* diag_L <= diag_R + 1 by IH3 *)
    (* Therefore min3 del_L ins_L diag_L <= min3 (del_R+1) (ins_R+1) (diag_R+1) = min3 del_R ins_R diag_R + 1 *)
    lia.
Qed.

Lemma lev_distance_snoc_left_bound : forall s1 c s2,
  lev_distance (s1 ++ [c]) s2 <= lev_distance s1 s2 + 1.
Proof.
  intros s1 c s2.
  apply (lev_distance_snoc_left_bound_aux (length s1 + length s2)). lia.
Qed.

(** Insertion bound: adding a char to s2 increases distance by at most 1 *)
Lemma lev_distance_insert_bound : forall s1 c s2,
  lev_distance s1 (c :: s2) <= lev_distance s1 s2 + 1.
Proof.
  intros s1 c s2.
  destruct s1 as [| d s1'].
  - rewrite lev_distance_empty_left.
    rewrite lev_distance_empty_left.
    simpl. lia.
  - rewrite lev_distance_cons.
    unfold min3.
    apply Nat.le_trans with (lev_distance (d :: s1') s2 + 1).
    + apply Nat.min_le_iff. right. apply Nat.le_min_l.
    + lia.
Qed.

(** Append version of insertion bound.
    Appending a character to s2 increases distance by at most 1.
    Proved using strong induction on |s1| + |s2|.
    Key insight: Each branch of the min3 recurrence satisfies the bound by IH. *)
Lemma lev_distance_append_char_bound_aux : forall n s1 s2 c,
  length s1 + length s2 <= n ->
  lev_distance s1 (s2 ++ [c]) <= lev_distance s1 s2 + 1.
Proof.
  intro n. induction n as [n IH] using lt_wf_ind.
  intros s1 s2 c Hlen.
  destruct s1 as [| d s1']; destruct s2 as [| e s2'].
  - (* s1 = [], s2 = [] *)
    (* Goal: lev_distance [] [c] <= lev_distance [] [] + 1 *)
    (* Both sides use lev_distance_empty_left: 1 <= 0 + 1 *)
    rewrite !lev_distance_empty_left. simpl length. lia.
  - (* s1 = [], s2 = e :: s2' *)
    rewrite !lev_distance_empty_left.
    simpl. rewrite length_app. simpl. lia.
  - (* s1 = d :: s1', s2 = [] *)
    (* Goal: lev_distance (d :: s1') [c] <= lev_distance (d :: s1') [] + 1 *)
    (* Use lev_distance_insert_bound: lev_distance s1 (c :: s2) <= lev_distance s1 s2 + 1 *)
    (* This gives lev_distance (d :: s1') [c] <= lev_distance (d :: s1') [] + 1 directly *)
    apply lev_distance_insert_bound.
  - (* s1 = d :: s1', s2 = e :: s2' - main inductive case *)
    simpl app.
    rewrite (lev_distance_cons d e s1' (s2' ++ [c])).
    rewrite (lev_distance_cons d e s1' s2').
    unfold min3.
    (* IH on deletion branch *)
    assert (IH1: lev_distance s1' (e :: (s2' ++ [c])) <= lev_distance s1' (e :: s2') + 1).
    { change (e :: (s2' ++ [c])) with ((e :: s2') ++ [c]).
      apply (IH (length s1' + S (length s2'))).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* IH on insertion branch *)
    assert (IH2: lev_distance (d :: s1') (s2' ++ [c]) <= lev_distance (d :: s1') s2' + 1).
    { apply (IH (S (length s1') + length s2')).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* IH on diagonal branch *)
    assert (IH3: lev_distance s1' (s2' ++ [c]) <= lev_distance s1' s2' + 1).
    { apply (IH (length s1' + length s2')).
      - simpl in Hlen. lia.
      - simpl. lia. }
    (* Each LHS branch is <= corresponding RHS branch + 1 *)
    lia.
Qed.

Lemma lev_distance_append_char_bound : forall s1 s2 c,
  lev_distance s1 (s2 ++ [c]) <= lev_distance s1 s2 + 1.
Proof.
  intros s1 s2 c.
  apply (lev_distance_append_char_bound_aux (length s1 + length s2)). lia.
Qed.

(** Match bound: matching last character.
    If both strings end with the same character, matching that character
    doesn't add to the distance.

    Uses lev_distance_snoc directly. The diagonal branch has subst_cost c c = 0,
    so the result follows from the min3 structure. *)
Lemma lev_distance_match_last : forall s1 s2 c,
  lev_distance (s1 ++ [c]) (s2 ++ [c]) <= lev_distance s1 s2.
Proof.
  intros s1 s2 c.
  rewrite lev_distance_snoc.
  unfold min3.
  unfold subst_cost. rewrite char_eq_refl.
  (* LHS = min (lev_distance s1 (s2 ++ [c]) + 1)
              (min (lev_distance (s1 ++ [c]) s2 + 1)
                   (lev_distance s1 s2 + 0)) *)
  (* We need to show this is <= lev_distance s1 s2 *)
  (* The diagonal branch is exactly lev_distance s1 s2 + 0 = lev_distance s1 s2 *)
  (* And min takes the minimum, so result <= diagonal = lev_distance s1 s2 *)
  lia.
Qed.

(** Substitution bound: different last characters add at most 1 to distance.
    Uses lev_distance_snoc directly. The diagonal branch has subst_cost c d = 1
    when c <> d, giving the +1 bound. *)
Lemma lev_distance_subst_last : forall s1 s2 c d,
  c <> d ->
  lev_distance (s1 ++ [c]) (s2 ++ [d]) <= lev_distance s1 s2 + 1.
Proof.
  intros s1 s2 c d Hneq.
  rewrite lev_distance_snoc.
  unfold min3.
  unfold subst_cost.
  destruct (char_eq c d) eqn:Heq.
  - (* c = d - contradiction with Hneq *)
    unfold char_eq in Heq.
    destruct (Ascii.ascii_dec c d) as [Heq' | Hneq'].
    + contradiction.
    + discriminate.
  - (* c <> d, so subst_cost = 1 *)
    (* LHS = min (lev_distance s1 (s2 ++ [d]) + 1)
                (min (lev_distance (s1 ++ [c]) s2 + 1)
                     (lev_distance s1 s2 + 1)) *)
    (* The diagonal branch is exactly lev_distance s1 s2 + 1, and min <= diagonal *)
    lia.
Qed.

(** Generalized soundness lemma for partial alignments.
    This is the key invariant maintained by position_reachable:
    If we reach position (i, e) with dict dp, then
    lev_distance (firstn i query) dp <= e.

    This generalizes the final position case where i = |query|. *)
Lemma reachable_partial_distance : forall query n dp p,
  position_reachable query n dp p ->
  is_special p = false ->
  lev_distance (firstn (term_index p) query) dp <= num_errors p.
Proof.
  intros query n dp p Hreach.
  induction Hreach as [
    | dp' i e Hreach' IH Hle He
    | dp' c i e Hreach' IH Hi Hnth
    | dp' c c' i e Hreach' IH Hi Hnth He
    | dp' c i e Hreach' IH He
  ]; intros Hspec.
  - (* reach_initial: position (0, 0), dp = [] *)
    unfold initial_position, std_pos. simpl.
    rewrite lev_distance_unfold. reflexivity.
  - (* reach_delete: position (S i, S e), dp unchanged *)
    (* By IH: lev_distance (firstn i query) dp' <= e
       Goal: lev_distance (firstn (S i) query) dp' <= S e
       We need to show: firstn (S i) query = firstn i query ++ [c] for some c
       Then use lev_distance_snoc_left_bound. *)
    simpl term_index. simpl num_errors. simpl is_special in Hspec.
    assert (Hbound : i < length query) by lia.
    destruct (term_index_bound query i Hbound) as [c Hnth_c].
    rewrite (firstn_S_snoc_nth_error query i c Hnth_c).
    apply Nat.le_trans with (lev_distance (firstn i query) dp' + 1).
    + apply lev_distance_snoc_left_bound.
    + specialize (IH Hspec). simpl term_index in IH. simpl num_errors in IH.
      lia.
  - (* reach_match: position (S i, e), dp extended by c *)
    (* By IH: lev_distance (firstn i query) dp' <= e
       query[i] = c, so firstn (S i) query = firstn i query ++ [c]
       Goal: lev_distance (firstn (S i) query) (dp' ++ [c]) <= e
       Use lev_distance_match_last. *)
    simpl term_index. simpl num_errors. simpl is_special in Hspec.
    rewrite (firstn_S_snoc_nth_error query i c Hnth).
    apply Nat.le_trans with (lev_distance (firstn i query) dp').
    + apply lev_distance_match_last.
    + specialize (IH Hspec). simpl term_index in IH. simpl num_errors in IH.
      exact IH.
  - (* reach_substitute: position (S i, S e), dp extended by c, c ≠ query[i] *)
    (* By IH: lev_distance (firstn i query) dp' <= e
       query[i] = c' ≠ c, so firstn (S i) query = firstn i query ++ [c']
       Goal: lev_distance (firstn (S i) query) (dp' ++ [c]) <= S e
       Use lev_distance_subst_last. *)
    simpl term_index. simpl num_errors. simpl is_special in Hspec.
    rewrite (firstn_S_snoc_nth_error query i c' Hnth).
    apply Nat.le_trans with (lev_distance (firstn i query) dp' + 1).
    + apply lev_distance_subst_last.
      (* Need to show c' <> c. We have c <> c' from Hneq *)
      intro Heq. symmetry in Heq. contradiction.
    + specialize (IH Hspec). simpl term_index in IH. simpl num_errors in IH.
      lia.
  - (* reach_insert: position (i, S e), dp extended by c *)
    (* By IH: lev_distance (firstn i query) dp' <= e
       Goal: lev_distance (firstn i query) (dp' ++ [c]) <= S e
       Use lev_distance_append_char_bound. *)
    simpl term_index. simpl num_errors. simpl is_special in Hspec.
    apply Nat.le_trans with (lev_distance (firstn i query) dp' + 1).
    + apply lev_distance_append_char_bound.
    + specialize (IH Hspec). simpl term_index in IH. simpl num_errors in IH.
      lia.
Qed.

(** Corollary: when term_index = length query, we get the full distance bound *)
Lemma reachable_partial_to_full : forall query n dp p,
  position_reachable query n dp p ->
  is_special p = false ->
  term_index p = length query ->
  lev_distance query dp <= num_errors p.
Proof.
  intros query n dp p Hreach Hspec Hfinal.
  rewrite <- (firstn_all query).
  rewrite <- Hfinal.
  apply (reachable_partial_distance query n dp p Hreach Hspec).
Qed.

(** The main soundness lemma: reachable final position bounds distance.
    This is now a direct corollary of reachable_partial_to_full. *)
Lemma reachable_final_to_distance : forall query dict n p,
  position_reachable query n dict p ->
  is_special p = false ->
  term_index p = length query ->
  lev_distance query dict <= num_errors p.
Proof.
  intros query dict n p Hreach Hspec Hfinal.
  apply (reachable_partial_to_full query n dict p Hreach Hspec Hfinal).
Qed.

(** MergeSplit-reachable positions have bounded error counts. *)
Lemma reachable_merge_split_implies_edit_distance : forall query n dp p,
  position_reachable_merge_split query n dp p ->
  num_errors p <= n.
Proof.
  intros query n dp p Hreach.
  induction Hreach; simpl in *; try lia; exact IHHreach.
Qed.

(** MergeSplit-reachable positions never advance past the query. *)
Lemma reachable_merge_split_term_index_bound_query : forall query n dict_prefix p,
  position_reachable_merge_split query n dict_prefix p ->
  term_index p <= length query.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - unfold initial_position. simpl. lia.
  - simpl. exact H.
  - simpl. lia.
  - simpl. lia.
  - simpl. exact IHHreach.
  - simpl. lia.
  - simpl. exact IHHreach.
  - simpl. lia.
Qed.

(** MergeSplit partial-distance invariant.

    For ordinary positions, [num_errors] bounds the merge-split distance between
    the consumed query prefix and dictionary prefix. For split-in-progress
    special positions, the second conjunct records the predecessor before the
    first split output character; this is what lets complete-split consume the
    second output character without adding another error. *)
Lemma reachable_merge_split_partial_distance_strong : forall query n dp p,
  position_reachable_merge_split query n dp p ->
  merge_split_distance (firstn (term_index p) query) dp <= num_errors p /\
  (is_special p = true ->
   forall q,
     nth_error query (term_index p) = Some q ->
     exists dp0 d1 e0,
       dp = dp0 ++ [d1] /\
       p = special_pos (term_index p) (S e0) /\
       num_errors p = S e0 /\
       merge_split_distance (firstn (term_index p) query) dp0 <= e0).
Proof.
  intros query n dp p Hreach.
  induction Hreach as [
    | dp' i e Hreach' IH Hle He
    | dp' c i e Hreach' IH Hi Hnth
    | dp' c c' i e Hreach' IH Hi Hnth He
    | dp' c i e Hreach' IH He
    | dp' c i e Hreach' IH Hi He
    | dp' c i e Hreach' IH He
    | dp' c i e Hreach' IH Hi
  ].
  - split.
    + unfold initial_position, std_pos. simpl. rewrite ms_empty_left. reflexivity.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      assert (Hlt : i < length query) by lia.
      destruct (term_index_bound query i Hlt) as [q Hnth].
      rewrite (firstn_S_snoc_nth_error query i q Hnth).
      eapply Nat.le_trans.
      * apply ms_distance_delete_last.
      * lia.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      rewrite (firstn_S_snoc_nth_error query i c Hnth).
      eapply Nat.le_trans.
      * apply ms_distance_match_last.
      * exact IHdist.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      rewrite (firstn_S_snoc_nth_error query i c' Hnth).
      eapply Nat.le_trans.
      * apply ms_distance_subst_last.
      * pose proof (subst_cost_le_one c' c). lia.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      eapply Nat.le_trans.
      * apply ms_distance_insert_last.
      * lia.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      destruct (term_index_bound query i) as [q1 Hnth1]; [lia|].
      destruct (term_index_bound query (S i)) as [q2 Hnth2]; [lia|].
      replace (firstn (S (S i)) query) with (firstn i query ++ [q1; q2]).
      * eapply Nat.le_trans.
        -- apply ms_distance_merge_last.
        -- lia.
      * rewrite (firstn_S_snoc_nth_error query (S i) q2 Hnth2).
        rewrite (firstn_S_snoc_nth_error query i q1 Hnth1).
        rewrite <- app_assoc. reflexivity.
    + intros Hspec. simpl in Hspec. discriminate.
  - destruct IH as [IHdist _].
    simpl term_index in IHdist. simpl num_errors in IHdist.
    split.
    + simpl term_index. simpl num_errors.
      eapply Nat.le_trans.
      * apply ms_distance_insert_last.
      * lia.
    + intros _ q Hnth.
      exists dp', c, e.
      repeat split; simpl; try reflexivity.
      exact IHdist.
  - destruct IH as [_ IHspecial].
    split.
    + simpl term_index. simpl num_errors.
      destruct (term_index_bound query i Hi) as [q Hnth].
      specialize (IHspecial eq_refl q Hnth) as [dp0 [d1 [e0 [Hdp [Hp [Herr Hprev]]]]]].
      simpl in Hp, Herr.
      simpl term_index in Hprev.
      subst e.
      rewrite Hdp.
      rewrite (firstn_S_snoc_nth_error query i q Hnth).
      replace ((dp0 ++ [d1]) ++ [c]) with (dp0 ++ [d1; c]) by (rewrite <- app_assoc; reflexivity).
      eapply Nat.le_trans.
      * apply ms_distance_split_last.
      * lia.
    + intros Hspec. simpl in Hspec. discriminate.
Qed.

Lemma reachable_merge_split_partial_distance : forall query n dp p,
  position_reachable_merge_split query n dp p ->
  merge_split_distance (firstn (term_index p) query) dp <= num_errors p.
Proof.
  intros query n dp p Hreach.
  apply (proj1 (reachable_merge_split_partial_distance_strong query n dp p Hreach)).
Qed.

Lemma reachable_merge_split_final_to_distance : forall query dict n p,
  position_reachable_merge_split query n dict p ->
  term_index p = length query ->
  merge_split_distance query dict <= num_errors p.
Proof.
  intros query dict n p Hreach Hfinal.
  rewrite <- (firstn_all query).
  rewrite <- Hfinal.
  apply reachable_merge_split_partial_distance with (n := n).
  exact Hreach.
Qed.

(** * Min Bound Helper Lemmas *)

(** Helper: if a_i <= b_i for i=1,2,3, then min3(a) <= min3(b) *)
Lemma min3_le_min3 : forall a1 a2 a3 b1 b2 b3,
  a1 <= b1 -> a2 <= b2 -> a3 <= b3 ->
  min a1 (min a2 a3) <= min b1 (min b2 b3).
Proof.
  intros. apply Nat.min_le_compat; [assumption|].
  apply Nat.min_le_compat; assumption.
Qed.

(** Helper: if a_i <= b_i for i=1,2,3, then min4(a1,a2,a3,a4) <= min3(b1,b2,b3)
    This drops a4 from consideration. *)
Lemma min4_le_min3 : forall a1 a2 a3 a4 b1 b2 b3,
  a1 <= b1 -> a2 <= b2 -> a3 <= b3 ->
  min (min a1 a2) (min a3 a4) <= min b1 (min b2 b3).
Proof.
  intros.
  etransitivity.
  { apply Nat.min_le_compat_l. apply Nat.le_min_l. }
  (* Now: min (min a1 a2) a3 <= min b1 (min b2 b3) *)
  (* Use <- min_assoc: min (min n m) p = min n (min m p) *)
  rewrite <- Nat.min_assoc.
  apply min3_le_min3; assumption.
Qed.

(** * Damerau-Levenshtein Distance Monotonicity Lemmas

    These lemmas follow directly from the DL recurrence. The deletion option
    in damerau_lev_distance (c::s1) s2 is exactly damerau_lev_distance s1 s2 + 1,
    so the min is at most that value.

    Note: These cannot be proved by chaining through standard Levenshtein because
    DL <= Lev (not the reverse). They must be proved directly from the DL recurrence.
*)

(** Deletion bound: adding a char at beginning of s1 increases DL distance by at most 1.
    Follows from the DL recurrence: one option is (delete c) + recurse on s1. *)
Lemma damerau_lev_distance_delete_bound : forall c s1 s2,
  damerau_lev_distance (c :: s1) s2 <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros c s1 s2.
  destruct s1 as [| c1 s1'].
  - (* s1 = [], so c::s1 = [c] *)
    destruct s2 as [| c2 s2'].
    + (* s2 = [] *)
      rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_left.
      simpl. lia.
    + destruct s2' as [| c2' s2''].
      * (* s2 = [c2] *)
        rewrite damerau_lev_single. rewrite damerau_lev_empty_left.
        simpl. destruct (char_eq c c2); lia.
      * (* s2 = c2::c2'::s2'' *)
        rewrite damerau_lev_single_multi.
        (* First option is dam([], c2::c2'::s2'') + 1 *)
        unfold min3. lia.
  - (* s1 = c1::s1', so c::s1 = c::c1::s1' *)
    destruct s2 as [| c2 s2'].
    + (* s2 = [] *)
      rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_right.
      simpl. lia.
    + destruct s2' as [| c2' s2''].
      * (* s2 = [c2] *)
        rewrite damerau_lev_multi_single.
        (* First option is dam(c1::s1', [c2]) + 1 *)
        unfold min3. lia.
      * (* s2 = c2::c2'::s2'' *)
        rewrite damerau_lev_cons2.
        (* First option is dam(c1::s1', c2::c2'::s2'') + 1 *)
        unfold min4. lia.
Qed.

(** Adding char to end of s1 increases DL distance by at most 1.
    By strong induction on combined length, using the cons bound. *)
Lemma damerau_lev_distance_snoc_left_bound : forall s1 c s2,
  damerau_lev_distance (s1 ++ [c]) s2 <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros s1 c s2.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| a s1'].
  - (* s1 = [], s1 ++ [c] = [c] *)
    simpl. apply damerau_lev_distance_delete_bound.
  - (* s1 = a :: s1' *)
    destruct s2 as [| b s2'].
    + (* s2 = [] *)
      rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_right.
      rewrite length_app. simpl. lia.
    + (* s2 = b :: s2' *)
      destruct s1' as [| a' s1''].
      * (* s1 = [a], so s1++[c] = [a;c] *)
        destruct s2' as [| b' s2''].
        -- (* s2 = [b], so we have [a;c] vs [b] - use multi_single *)
           simpl app.
           rewrite damerau_lev_multi_single.
           rewrite damerau_lev_single.
           unfold min3, subst_cost.
           (* First option is dam([c], [b]) + 1 *)
           rewrite damerau_lev_single.
           rewrite damerau_lev_empty_right.
           rewrite damerau_lev_empty_right.
           destruct (char_eq c b); destruct (char_eq a b); simpl; lia.
        -- (* s2 = b::b'::s2'', so we have [a;c] vs (b::b'::s2'') - use cons2 *)
           simpl app.
           (* Goal: dam([a;c], b::b'::s2'') <= dam([a], b::b'::s2'') + 1 *)
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance [a] (b :: b' :: s2'') =
                          min3 (damerau_lev_distance [] (b :: b' :: s2'') + 1)
                               (damerau_lev_distance [a] (b' :: s2'') + 1)
                               (damerau_lev_distance [] (b' :: s2'') + subst_cost a b)).
           { apply damerau_lev_single_multi. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* Use delete_bound and IH *)
           assert (H1 : damerau_lev_distance [c] (b :: b' :: s2'') <=
                        damerau_lev_distance [] (b :: b' :: s2'') + 1).
           { apply damerau_lev_distance_delete_bound. }
           assert (H2 : damerau_lev_distance [a; c] (b' :: s2'') <=
                        damerau_lev_distance [a] (b' :: s2'') + 1).
           { change [a; c] with ([a] ++ [c]).
             apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance [c] (b' :: s2'') <=
                        damerau_lev_distance [] (b' :: s2'') + 1).
           { apply damerau_lev_distance_delete_bound. }
           (* min4 <= min3 + 1: each opt <= corresponding rhs + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1)
                                              (damerau_lev_distance [a] (b' :: s2'') + 1 + 1))
                                        (damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1)).
           ++ apply Nat.le_trans with (min (min (damerau_lev_distance [c] (b :: b' :: s2'') + 1)
                                                 (damerau_lev_distance [a; c] (b' :: s2'') + 1))
                                           (damerau_lev_distance [c] (b' :: s2'') + subst_cost a b)).
              ** apply Nat.min_le_compat; [apply le_n | apply Nat.le_min_l].
              ** apply Nat.min_le_compat.
                 { apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2]. }
                 { lia. }
           ++ rewrite Nat.min_assoc. lia.
      * (* s1 = a::a'::s1'', so s1++[c] has at least 3 chars *)
        destruct s2' as [| b' s2''].
        -- (* s2 = [b], goal: dam(a::a'::(s1''++[c]), [b]) <= dam(a::a'::s1'', [b]) + 1 *)
           simpl app.
           rewrite damerau_lev_multi_single.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') [b] =
                          min3 (damerau_lev_distance (a' :: s1'') [b] + 1)
                               (damerau_lev_distance (a :: a' :: s1'') [] + 1)
                               (damerau_lev_distance (a' :: s1'') [] + subst_cost a b)).
           { apply damerau_lev_multi_single. }
           rewrite Hrhs. clear Hrhs.
           unfold min3.
           (* Use IH for H1, and direct length for H2, H3 *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) [b] + 1 <=
                        damerau_lev_distance (a' :: s1'') [b] + 1 + 1).
           { assert (Heq : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             assert (HIH : damerau_lev_distance (a' :: s1'' ++ [c]) [b] <=
                           damerau_lev_distance (a' :: s1'') [b] + 1).
             { rewrite Heq. apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
             lia. }
           (* opt2, opt3 and rhs2, rhs3 are all lengths - simplify *)
           rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_right.
           rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_right.
           (* Now: min(opt1, min(opt2, opt3)) <= min(rhs1, min(rhs2, rhs3)) + 1 *)
           (* opt2 = len(a::a'::(s1''++[c])) + 1, rhs2 = len(a::a'::s1'') + 1 *)
           (* opt3 = len(a'::(s1''++[c])) + subst, rhs3 = len(a'::s1'') + subst *)
           (* All opt_i <= rhs_i + 1 *)
           apply Nat.le_trans with (min (damerau_lev_distance (a' :: s1'') [b] + 1 + 1)
                                        (min (length (a :: a' :: s1'') + 1 + 1)
                                             (length (a' :: s1'') + subst_cost a b + 1))).
           ++ (* min(opts) <= min(rhss + 1) *)
              apply Nat.min_le_compat.
              ** exact H1.
              ** apply Nat.min_le_compat; simpl; rewrite length_app; simpl; lia.
           ++ (* min(rhss + 1) = min(rhss) + 1 *)
              rewrite <- Nat.add_min_distr_r. rewrite <- Nat.add_min_distr_r. lia.
        -- (* s2 = b::b'::s2'', goal: dam(a::a'::(s1''++[c]), b::b'::s2'') <= dam(a::a'::s1'', ...) + 1 *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') (b :: b' :: s2'') =
                          min4 (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1)
                               (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1)
                               (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b)
                               (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b')).
           { apply damerau_lev_cons2. }
           rewrite Hrhs. clear Hrhs.
           unfold min4.
           (* Use IH on all 4 options *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) (b :: b' :: s2'') <=
                        damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq1. apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'' ++ [c]) (b' :: s2'') <=
                        damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq2 : a :: a' :: s1'' ++ [c] = (a :: a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq2. apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (a' :: s1'' ++ [c]) (b' :: s2'') <=
                        damerau_lev_distance (a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq3 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq3. apply (IH (length s1'' + length s2'' + 2)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H4 : damerau_lev_distance (s1'' ++ [c]) s2'' <=
                        damerau_lev_distance s1'' s2'' + 1).
           { apply (IH (length s1'' + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           (* min4(lhs) <= min4(rhs) + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1 + 1)
                                              (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1 + 1))
                                        (min (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b + 1)
                                             (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b' + 1))).
           ++ apply Nat.min_le_compat.
              ** apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
              ** apply Nat.min_le_compat; lia.
           ++ rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              lia.
Qed.

(** Insertion bound: adding a char at beginning of s2 increases DL distance by at most 1.
    Follows from the DL recurrence: one option is (insert c) + recurse on s2. *)
Lemma damerau_lev_distance_insert_bound : forall s1 c s2,
  damerau_lev_distance s1 (c :: s2) <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros s1 c s2.
  destruct s2 as [| c2 s2'].
  - (* s2 = [], so c::s2 = [c] *)
    destruct s1 as [| c1 s1'].
    + (* s1 = [] *)
      rewrite damerau_lev_empty_left. rewrite damerau_lev_empty_left.
      simpl. lia.
    + destruct s1' as [| c1' s1''].
      * (* s1 = [c1] *)
        rewrite damerau_lev_single. rewrite damerau_lev_empty_right.
        simpl. destruct (char_eq c1 c); lia.
      * (* s1 = c1::c1'::s1'' *)
        rewrite damerau_lev_multi_single.
        (* Second option is dam(c1::c1'::s1'', []) + 1 *)
        unfold min3. lia.
  - (* s2 = c2::s2', so c::s2 = c::c2::s2' *)
    destruct s1 as [| c1 s1'].
    + (* s1 = [] *)
      rewrite damerau_lev_empty_left. rewrite damerau_lev_empty_left.
      simpl. lia.
    + destruct s1' as [| c1' s1''].
      * (* s1 = [c1] *)
        rewrite damerau_lev_single_multi.
        (* Second option is dam([c1], c2::s2') + 1 *)
        unfold min3. lia.
      * (* s1 = c1::c1'::s1'' *)
        rewrite damerau_lev_cons2.
        (* Second option is dam(c1::c1'::s1'', c2::s2') + 1 *)
        unfold min4. lia.
Qed.

(** Adding char to end of s2 increases DL distance by at most 1.
    By strong induction on combined length, using the cons bound. *)
Lemma damerau_lev_distance_append_char_bound : forall s1 s2 c,
  damerau_lev_distance s1 (s2 ++ [c]) <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros s1 s2 c.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s2 as [| b s2'].
  - (* s2 = [], s2 ++ [c] = [c] *)
    simpl. apply damerau_lev_distance_insert_bound.
  - (* s2 = b :: s2' *)
    destruct s1 as [| a s1'].
    + (* s1 = [] *)
      rewrite damerau_lev_empty_left. rewrite damerau_lev_empty_left.
      rewrite length_app. simpl. lia.
    + (* s1 = a :: s1' *)
      destruct s2' as [| b' s2''].
      * (* s2 = [b], so s2++[c] = [b;c] *)
        destruct s1' as [| a' s1''].
        -- (* s1 = [a], so LHS = dam([a], [b;c]) - use single_multi *)
           simpl app.
           rewrite damerau_lev_single_multi.
           rewrite damerau_lev_single.
           unfold min3, subst_cost.
           rewrite damerau_lev_empty_left.
           rewrite damerau_lev_single.
           rewrite damerau_lev_empty_left.
           destruct (char_eq a b); destruct (char_eq a c); simpl; lia.
        -- (* s1 = a::a'::s1'', s2 = [b], goal: dam(a::a'::s1'', [b;c]) <= dam(a::a'::s1'', [b]) + 1 *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') [b] =
                          min3 (damerau_lev_distance (a' :: s1'') [b] + 1)
                               (damerau_lev_distance (a :: a' :: s1'') [] + 1)
                               (damerau_lev_distance (a' :: s1'') [] + subst_cost a b)).
           { apply damerau_lev_multi_single. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* Use IH and insert_bound *)
           assert (H1 : damerau_lev_distance (a' :: s1'') [b; c] <=
                        damerau_lev_distance (a' :: s1'') [b] + 1).
           { change [b; c] with ([b] ++ [c]).
             apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'') [c] <=
                        damerau_lev_distance (a :: a' :: s1'') [] + 1).
           { apply damerau_lev_distance_insert_bound. }
           assert (H3 : damerau_lev_distance (a' :: s1'') [c] <=
                        damerau_lev_distance (a' :: s1'') [] + 1).
           { apply damerau_lev_distance_insert_bound. }
           (* min4 has 4 opts, min3 has 3 - case split on trans_cost *)
           set (rhs1 := damerau_lev_distance (a' :: s1'') [b] + 1).
           set (rhs2 := damerau_lev_distance (a :: a' :: s1'') [] + 1).
           set (rhs3 := damerau_lev_distance (a' :: s1'') [] + subst_cost a b).
           set (opt4 := damerau_lev_distance s1'' [] + trans_cost_calc a a' b c).
           destruct (andb (char_eq a c) (char_eq a' b)) eqn:Htrans.
           ++ (* trans_cost = 1: opt4 = |s1''| + 1 <= rhs3 + 1 *)
              assert (H4 : opt4 <= rhs3 + 1).
              { unfold opt4, rhs3. rewrite damerau_lev_empty_right. rewrite damerau_lev_empty_right. simpl.
                unfold trans_cost_calc. rewrite Htrans. unfold subst_cost.
                destruct (char_eq a b); simpl; lia. }
              (* Show opt3 = dam(a'::s1'',[c]) + subst <= rhs3 + 1 using H3 *)
              assert (H3' : damerau_lev_distance (a' :: s1'') [c] + subst_cost a b <= rhs3 + 1).
              { unfold rhs3. lia. }
              apply Nat.le_trans with (min (min (rhs1 + 1) (rhs2 + 1)) (min (rhs3 + 1) (rhs3 + 1))).
              ** apply Nat.min_le_compat.
                 --- apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
                 --- apply Nat.min_le_compat; [exact H3' | exact H4].
              ** rewrite Nat.min_id.
                 rewrite <- Nat.add_min_distr_r.  (* inner: min(rhs1+1, rhs2+1) -> min rhs1 rhs2 + 1 *)
                 rewrite <- Nat.add_min_distr_r.  (* outer: min(min rhs1 rhs2 + 1, rhs3+1) -> min3 + 1 *)
                 rewrite Nat.min_assoc.
                 lia.
           ++ (* trans_cost = 100: opt4 is huge, so min4 = min(opt1, opt2, opt3) *)
              (* opt3 bound: dam(a'::s1'',[c]) + subst <= rhs3 + 1 *)
              assert (H3' : damerau_lev_distance (a' :: s1'') [c] + subst_cost a b <= rhs3 + 1).
              { unfold rhs3. lia. }
              (* Show min4 <= min(min(rhs1+1, rhs2+1), rhs3+1) = min3 + 1 *)
              (* Since opt4 is large, min(opt3, opt4) <= opt3 <= rhs3+1, not min(opt4,...) <= ??? *)
              apply Nat.le_trans with (min (min (rhs1 + 1) (rhs2 + 1)) (rhs3 + 1)).
              ** apply Nat.min_le_compat.
                 --- (* min(opt1, opt2) <= min(rhs1+1, rhs2+1) *)
                     apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
                 --- (* min(opt3, opt4) <= rhs3 + 1 - use opt3 as the bound since opt4 is huge *)
                     apply Nat.le_trans with (damerau_lev_distance (a' :: s1'') [c] + subst_cost a b).
                     +++ apply Nat.le_min_l.
                     +++ exact H3'.
              ** (* min(min(rhs1+1, rhs2+1), rhs3+1) <= min3 + 1 *)
                 rewrite <- Nat.add_min_distr_r.
                 rewrite <- Nat.add_min_distr_r.
                 rewrite Nat.min_assoc.
                 lia.
      * (* s2 = b::b'::s2'', so s2++[c] has at least 3 chars *)
        destruct s1' as [| a' s1''].
        -- (* s1 = [a], s2 = b::b'::s2'', goal: dam([a], (b::b'::s2'')++[c]) <= dam([a], b::b'::s2'') + 1 *)
           simpl app.
           rewrite damerau_lev_single_multi.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance [a] (b :: b' :: s2'') =
                          min3 (damerau_lev_distance [] (b :: b' :: s2'') + 1)
                               (damerau_lev_distance [a] (b' :: s2'') + 1)
                               (damerau_lev_distance [] (b' :: s2'') + subst_cost a b)).
           { apply damerau_lev_single_multi. }
           rewrite Hrhs. clear Hrhs.
           unfold min3.
           (* Use IH *)
           assert (H1 : damerau_lev_distance [a] (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance [a] (b' :: s2'') + 1).
           { assert (Heq1 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1. apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           (* LHS = min (dam([], b::b'::s2''++[c])+1) (min (dam([a], b'::s2''++[c])+1) (dam([],b'::s2''++[c])+subst)) *)
           (* RHS = min (dam([], b::b'::s2'')+1) (min (dam([a], b'::s2'')+1) (dam([], b'::s2'')+subst)) + 1 *)
           (* Show all 3 LHS options are bounded by corresponding RHS options + 1 *)
           assert (HA : damerau_lev_distance [] (b :: b' :: s2'' ++ [c]) + 1 <=
                        damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1).
           { rewrite damerau_lev_empty_left. rewrite damerau_lev_empty_left.
             (* b::b'::(s2''++[c]) has length 2 + length s2'' + 1 = length (b::b'::s2'') + 1 *)
             assert (Hlen1 : length (b :: b' :: s2'' ++ [c]) = length (b :: b' :: s2'') + 1).
             { simpl. rewrite length_app. simpl. lia. }
             lia. }
           assert (HC : damerau_lev_distance [] (b' :: s2'' ++ [c]) + subst_cost a b <=
                        damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1).
           { rewrite damerau_lev_empty_left. rewrite damerau_lev_empty_left.
             assert (Hlen2 : length (b' :: s2'' ++ [c]) = length (b' :: s2'') + 1).
             { simpl. rewrite length_app. simpl. lia. }
             lia. }
           apply Nat.le_trans with (min (damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1)
                                        (min (damerau_lev_distance [a] (b' :: s2'') + 1 + 1)
                                             (damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1))).
           { apply Nat.min_le_compat; [exact HA |].
             apply Nat.min_le_compat; [apply Nat.add_le_mono_r; exact H1 | exact HC]. }
           rewrite <- Nat.add_min_distr_r.
           rewrite <- Nat.add_min_distr_r.
           lia.
        -- (* s1 = a::a'::s1'', s2 = b::b'::s2'', goal: dam(a::a'::s1'', (b::b'::s2'')++[c]) <= dam(...) + 1 *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') (b :: b' :: s2'') =
                          min4 (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1)
                               (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1)
                               (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b)
                               (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b')).
           { apply damerau_lev_cons2. }
           rewrite Hrhs. clear Hrhs.
           unfold min4.
           (* Use IH *)
           assert (H1 : damerau_lev_distance (a' :: s1'') (b :: b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1).
           { assert (Heq1 : b :: b' :: s2'' ++ [c] = (b :: b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1. apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'') (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq2 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq2. apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (a' :: s1'') (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq3 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq3. apply (IH (length s1'' + length s2'' + 2)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H4 : damerau_lev_distance s1'' (s2'' ++ [c]) <=
                        damerau_lev_distance s1'' s2'' + 1).
           { apply (IH (length s1'' + length s2'')); [simpl in Hlen; lia | reflexivity]. }
           (* min4(lhs) <= min4(rhs) + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1 + 1)
                                              (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1 + 1))
                                        (min (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b + 1)
                                             (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b' + 1))).
           ++ apply Nat.min_le_compat.
              ** apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
              ** apply Nat.min_le_compat; lia.
           ++ rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              lia.
Qed.

(** Match bound: matching last character doesn't increase DL distance.
    By strong induction: the diagonal case in the recurrence has cost 0 when chars match. *)
Lemma damerau_lev_distance_match_last : forall s1 s2 c,
  damerau_lev_distance (s1 ++ [c]) (s2 ++ [c]) <= damerau_lev_distance s1 s2.
Proof.
  intros s1 s2 c.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| a s1'].
  - (* s1 = [] *)
    destruct s2 as [| b s2'].
    + (* s1 = [], s2 = [] *)
      simpl. rewrite damerau_lev_single. rewrite damerau_lev_empty_left.
      rewrite char_eq_refl. lia.
    + (* s1 = [], s2 = b::s2' *)
      simpl. rewrite damerau_lev_empty_left.
      (* Goal: dam([c], (b::s2')++[c]) <= length (b::s2') *)
      destruct s2' as [| b' s2''].
      * (* s2 = [b], goal: dam([c], [b,c]) <= 1 *)
        simpl. rewrite damerau_lev_single_multi.
        (* dam([c],[b,c]) = min3(dam([],[b,c])+1, dam([c],[c])+1, dam([],[c])+subst(c,b)) *)
        rewrite damerau_lev_empty_left.
        rewrite damerau_lev_same.
        unfold min3, subst_cost, char_eq.
        destruct (Ascii.ascii_dec c b); simpl; lia.
      * (* s2 = b::b'::s2'', goal: dam([c], b::b'::(s2''++[c])) <= 2 + length s2'' *)
        simpl.
        rewrite damerau_lev_single_multi.
        rewrite damerau_lev_empty_left.
        unfold min3.
        (* Use IH: dam([c], b'::(s2''++[c])) <= 1 + length s2'' *)
        assert (HIH : damerau_lev_distance [c] (b' :: s2'' ++ [c]) <= 1 + length s2'').
        { assert (Heq : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
          rewrite Heq.
          change [c] with ([] ++ [c]).
          assert (Hm : 1 + length s2'' < n).
          { simpl in Hlen. lia. }
          apply Nat.le_trans with (damerau_lev_distance [] (b' :: s2'')).
          - apply (IH (1 + length s2'') Hm [] (b' :: s2'')).
            simpl. reflexivity.
          - rewrite damerau_lev_empty_left. reflexivity. }
        (* Second option: dam([c], b'::(s2''++[c])) + 1 <= (1 + length s2'') + 1 = 2 + length s2'' *)
        etransitivity.
        { apply Nat.le_min_r. }
        etransitivity.
        { apply Nat.le_min_l. }
        lia.
  - (* s1 = a::s1' *)
    destruct s2 as [| b s2'].
    + (* s1 = a::s1', s2 = [] *)
      simpl. rewrite damerau_lev_empty_right.
      (* Goal: dam((a::s1')++[c], [c]) <= length (a::s1') *)
      (* Symmetric to the s1=[], s2=b::s2' case *)
      destruct s1' as [| a' s1''].
      * (* s1' = [], goal: dam([a,c], [c]) <= 1 *)
        simpl. rewrite damerau_lev_multi_single.
        rewrite damerau_lev_same.
        rewrite damerau_lev_empty_right.
        unfold min3, subst_cost, char_eq.
        destruct (Ascii.ascii_dec a c); simpl; lia.
      * (* s1' = a'::s1'', goal: dam(a::a'::(s1''++[c]), [c]) <= 2 + length s1'' *)
        simpl.
        rewrite damerau_lev_multi_single.
        rewrite damerau_lev_empty_right.
        unfold min3.
        (* Use IH: dam(a'::(s1''++[c]), [c]) <= 1 + length s1'' *)
        assert (HIH : damerau_lev_distance (a' :: s1'' ++ [c]) [c] <= 1 + length s1'').
        { assert (Heq : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
          rewrite Heq.
          change [c] with ([] ++ [c]).
          assert (Hm : 1 + length s1'' < n).
          { simpl in Hlen. lia. }
          apply Nat.le_trans with (damerau_lev_distance (a' :: s1'') []).
          - apply (IH (1 + length s1'') Hm (a' :: s1'') []).
            simpl. lia.
          - rewrite damerau_lev_empty_right. reflexivity. }
        (* First option: dam(a'::(s1''++[c]),[c]) + 1 <= (1 + length s1'') + 1 = 2 + length s1'' *)
        etransitivity.
        { apply Nat.le_min_l. }
        lia.
    + (* s1 = a::s1', s2 = b::s2' *)
      destruct s1' as [| a' s1''].
      * (* s1 = [a] *)
        destruct s2' as [| b' s2''].
        -- (* s1 = [a], s2 = [b] *)
           (* Goal: dam([a,c], [b,c]) <= dam([a], [b]) = subst_cost(a,b) *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* min4(dam([c],[b,c])+1, dam([a,c],[c])+1,
                  dam([c],[c])+subst_cost(a,b), dam([],[])+trans_cost(a,c,b,c))
              <= dam([a],[b]) *)
           (* First rewrite dam([c],[c]) = 0 and dam([],[]) = 0 *)
           rewrite damerau_lev_same.
           rewrite (damerau_lev_empty_right []).
           (* Now rewrite dam([a],[b]) *)
           rewrite damerau_lev_single.
           unfold min4. simpl.
           (* Now: min(min A B)(min (subst_cost a b) C) <= subst_cost a b *)
           apply Nat.le_trans with (min (0 + subst_cost a b) (0 + trans_cost_calc a c b c)).
           { apply Nat.le_min_r. }
           apply Nat.le_min_l.
        -- (* s1 = [a], s2 = b::b'::s2'' *)
           (* Goal: dam([a,c], b::b'::(s2''++[c])) <= dam([a], b::b'::s2'') *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* LHS is now min4 structure; protect it before expanding RHS *)
           (* Assert the RHS expansion to avoid rewriting wrong term *)
           assert (Hrhs : damerau_lev_distance [a] (b :: b' :: s2'') =
                          min3 (damerau_lev_distance [] (b :: b' :: s2'') + 1)
                               (damerau_lev_distance [a] (b' :: s2'') + 1)
                               (damerau_lev_distance [] (b' :: s2'') + subst_cost a b)).
           { apply damerau_lev_single_multi. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* LHS: min(min opt1 opt2)(min opt3 opt4) where:
              opt1 = dam([c], b::b'::(s2''++[c])) + 1
              opt2 = dam([a,c], b'::(s2''++[c])) + 1
              opt3 = dam([c], b'::(s2''++[c])) + subst(a,b)
              opt4 = dam([], s2''++[c]) + trans(a,c,b,b')
              RHS: min(x)(min y z) where:
              x = dam([], b::b'::s2'') + 1
              y = dam([a], b'::s2'') + 1
              z = dam([], b'::s2'') + subst(a,b)
              IH gives: opt1 <= x, opt2 <= y, opt3 <= z *)
           (* First establish IH bounds *)
           assert (H1 : damerau_lev_distance [c] (b :: b' :: s2'' ++ [c]) <=
                        damerau_lev_distance [] (b :: b' :: s2'')).
           { assert (Heq1 : b :: b' :: s2'' ++ [c] = (b :: b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1. change [c] with ([] ++ [c]).
             apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance [a; c] (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance [a] (b' :: s2'')).
           { assert (Heq2 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq2. change [a; c] with ([a] ++ [c]).
             apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance [c] (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance [] (b' :: s2'')).
           { assert (Heq3 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq3. change [c] with ([] ++ [c]).
             apply (IH (1 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           (* Goal: min4(opt1,opt2,opt3,opt4) <= min3(x,y,z) *)
           (* where opt1<=x, opt2<=y, opt3<=z (from H1, H2, H3) *)
           (* Strategy: min(min(opt1,opt2),min(opt3,opt4)) <= min(min(x,y),z) = min(x,min(y,z)) *)
           (* Since min(opt3,opt4) <= opt3 <= z, we don't need to bound opt4 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance [] (b :: b' :: s2'') + 1)
                                              (damerau_lev_distance [a] (b' :: s2'') + 1))
                                        (damerau_lev_distance [] (b' :: s2'') + subst_cost a b)).
           ++ (* min4(opts) <= min(min(x,y),z) where opt1<=x, opt2<=y, opt3<=z *)
              apply Nat.min_le_compat.
              ** (* min(opt1,opt2) <= min(x,y) *)
                 apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
              ** (* min(opt3,opt4) <= z *)
                 apply Nat.le_trans with (damerau_lev_distance [c] (b' :: s2'' ++ [c]) + subst_cost a b).
                 { apply Nat.le_min_l. }
                 apply Nat.add_le_mono_r. exact H3.
           ++ (* min(min(x,y),z) <= min(x,min(y,z)) by associativity *)
              rewrite Nat.min_assoc. apply le_n.
      * (* s1 = a::a'::s1'' *)
        destruct s2' as [| b' s2''].
        -- (* s2 = [b] *)
           (* Symmetric case: dam(a::a'::(s1''++[c]), [b,c]) <= dam(a::a'::s1'', [b]) *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert the RHS expansion via damerau_lev_multi_single *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') [b] =
                          min3 (damerau_lev_distance (a' :: s1'') [b] + 1)
                               (damerau_lev_distance (a :: a' :: s1'') [] + 1)
                               (damerau_lev_distance (a' :: s1'') [] + subst_cost a b)).
           { apply damerau_lev_multi_single. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* LHS min4 has:
              opt1 = dam(a'::(s1''++[c]), [b,c]) + 1
              opt2 = dam(a::a'::(s1''++[c]), [c]) + 1
              opt3 = dam(a'::(s1''++[c]), [c]) + subst(a,b)
              opt4 = dam(s1''++[c], []) + trans(a,a',b,c)
              RHS min3 has:
              x = dam(a'::s1'', [b]) + 1
              y = dam(a::a'::s1'', []) + 1
              z = dam(a'::s1'', []) + subst(a,b) *)
           (* Establish IH bounds *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) [b; c] <=
                        damerau_lev_distance (a' :: s1'') [b]).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq1. change [b; c] with ([b] ++ [c]).
             apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'' ++ [c]) [c] <=
                        damerau_lev_distance (a :: a' :: s1'') []).
           { assert (Heq2 : a :: a' :: s1'' ++ [c] = (a :: a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq2. change [c] with ([] ++ [c]).
             apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (a' :: s1'' ++ [c]) [c] <=
                        damerau_lev_distance (a' :: s1'') []).
           { assert (Heq3 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq3. change [c] with ([] ++ [c]).
             apply (IH (1 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           (* min4(opts) <= min(min(x,y),z) <= min3(x,y,z) *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'') [b] + 1)
                                              (damerau_lev_distance (a :: a' :: s1'') [] + 1))
                                        (damerau_lev_distance (a' :: s1'') [] + subst_cost a b)).
           ++ apply Nat.min_le_compat.
              ** apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
              ** apply Nat.le_trans with (damerau_lev_distance (a' :: s1'' ++ [c]) [c] + subst_cost a b).
                 { apply Nat.le_min_l. }
                 apply Nat.add_le_mono_r. exact H3.
           ++ rewrite Nat.min_assoc. apply le_n.
        -- (* s1 = a::a'::s1'', s2 = b::b'::s2'' *)
           (* Main inductive case: dam(a::a'::(s1''++[c]), b::b'::(s2''++[c])) <=
                                   dam(a::a'::s1'', b::b'::s2'') *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert the RHS expansion via damerau_lev_cons2 *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') (b :: b' :: s2'') =
                          min4 (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1)
                               (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1)
                               (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b)
                               (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b')).
           { apply damerau_lev_cons2. }
           rewrite Hrhs. clear Hrhs.
           unfold min4.
           (* LHS: min(min(opt1,opt2),min(opt3,opt4)) with:
              opt1 = dam(a'::(s1''++[c]), b::b'::(s2''++[c])) + 1
              opt2 = dam(a::a'::(s1''++[c]), b'::(s2''++[c])) + 1
              opt3 = dam(a'::(s1''++[c]), b'::(s2''++[c])) + subst(a,b)
              opt4 = dam(s1''++[c], s2''++[c]) + trans(a,a',b,b')
              RHS: min(min(x,y),min(z,w)) with:
              x = dam(a'::s1'', b::b'::s2'') + 1
              y = dam(a::a'::s1'', b'::s2'') + 1
              z = dam(a'::s1'', b'::s2'') + subst(a,b)
              w = dam(s1'', s2'') + trans(a,a',b,b')
              IH gives: opt1<=x, opt2<=y, opt3<=z, opt4<=w *)
           (* Establish IH bounds *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) (b :: b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a' :: s1'') (b :: b' :: s2'')).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b :: b' :: s2'' ++ [c] = (b :: b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 3)).
             - simpl in Hlen. lia.
             - simpl. lia. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'' ++ [c]) (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a :: a' :: s1'') (b' :: s2'')).
           { assert (Heq1 : a :: a' :: s1'' ++ [c] = (a :: a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 3)).
             - simpl in Hlen. lia.
             - simpl. lia. }
           assert (H3 : damerau_lev_distance (a' :: s1'' ++ [c]) (b' :: s2'' ++ [c]) <=
                        damerau_lev_distance (a' :: s1'') (b' :: s2'')).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b' :: s2'' ++ [c] = (b' :: s2'') ++ [c]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 2)).
             - simpl in Hlen. lia.
             - simpl. lia. }
           assert (H4 : damerau_lev_distance (s1'' ++ [c]) (s2'' ++ [c]) <=
                        damerau_lev_distance s1'' s2'').
           { apply (IH (length s1'' + length s2'')).
             - simpl in Hlen. lia.
             - reflexivity. }
           (* min4(opts) <= min4(bounds) because each opt_i <= bound_i *)
           apply Nat.min_le_compat.
           ++ apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
           ++ apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H3 | exact H4].
Qed.

(** Substitution bound: different last characters add at most 1 to DL distance.
    By strong induction: diagonal option has cost 1 when chars differ. *)
Lemma damerau_lev_distance_subst_last : forall s1 s2 c d,
  c <> d ->
  damerau_lev_distance (s1 ++ [c]) (s2 ++ [d]) <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros s1 s2 c d Hneq.
  (* Strong induction on length s1 + length s2 *)
  remember (length s1 + length s2) as n eqn:Hlen.
  generalize dependent s2.
  generalize dependent s1.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| a s1'].
  - (* s1 = [] *)
    destruct s2 as [| b s2'].
    + (* s1 = [], s2 = [], goal: dam([c], [d]) <= 0 + 1 = 1 *)
      simpl.
      rewrite damerau_lev_single.
      unfold subst_cost, char_eq.
      destruct (Ascii.ascii_dec c d) as [Heq | _].
      * (* c = d contradicts hypothesis *)
        contradiction.
      * (* c <> d, so subst_cost = 1 *)
        lia.
    + (* s1 = [], s2 = b::s2', goal: dam([c], b::(s2'++[d])) <= 1 + length s2' + 1 *)
      destruct s2' as [| b' s2''].
      * (* s2 = [b], goal: dam([c], [b,d]) <= 2 *)
        simpl app.
        (* dam([c], [b,d]) = min3(dam([], [b,d])+1, dam([c], [d])+1, dam([], [d])+subst(c,b))
           We use the second option: dam([c], [d]) + 1 = subst(c,d) + 1 *)
        change [b; d] with (b :: d :: []).
        rewrite damerau_lev_single_multi.
        unfold min3.
        (* Use second term of the inner min: dam([c], [d]) + 1 *)
        apply Nat.le_trans with (damerau_lev_distance [c] [d] + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance [c] [d] + 1)
                                       (damerau_lev_distance [] [d] + subst_cost c b)).
          { apply Nat.le_min_r. }
          apply Nat.le_min_l. }
        rewrite damerau_lev_single.
        unfold subst_cost, char_eq.
        destruct (Ascii.ascii_dec c d); [contradiction | simpl; lia].
      * (* s2 = b::b'::s2'', goal: dam([c], b::b'::(s2''++[d])) <= 3 + length s2'' *)
        simpl app.
        rewrite damerau_lev_single_multi.
        unfold min3.
        (* Use IH on dam([c], b'::(s2''++[d])) <= dam([], b'::s2'') + 1 = 2 + length s2'' *)
        assert (Heq : b' :: s2'' ++ [d] = (b' :: s2'') ++ [d]) by reflexivity.
        assert (HIH : damerau_lev_distance [c] (b' :: s2'' ++ [d]) <=
                      damerau_lev_distance [] (b' :: s2'') + 1).
        { rewrite Heq. change [c] with ([] ++ [c]).
          apply (IH (1 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
        rewrite damerau_lev_empty_left in HIH.
        rewrite damerau_lev_empty_left.
        (* Goal: min(len(b::b'::s2''++[d])+1)(min(dam([c],b'::...)+1)(len+subst)) <= 3+len s2'' *)
        (* HIH: dam([c], ...) <= length (b' :: s2'') + 1 *)
        (* Use: second option dam([c], b'::...)+1 *)
        apply Nat.le_trans with (damerau_lev_distance [c] (b' :: s2'' ++ [d]) + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance [c] (b' :: s2'' ++ [d]) + 1)
                                       (damerau_lev_distance [] (b' :: s2'' ++ [d]) + subst_cost c b)).
          { apply Nat.le_min_r. }
          apply Nat.le_min_l. }
        (* Goal: dam([c], b' :: s2'' ++ [d]) + 1 <= 3 + length s2'' *)
        (* HIH: dam([c], ...) <= length (b' :: s2'') + 1 = S (length s2'') + 1 = 2 + length s2'' *)
        apply Nat.le_trans with (length (b' :: s2'') + 1 + 1).
        { apply Nat.add_le_mono_r. exact HIH. }
        simpl. lia.
  - (* s1 = a::s1' *)
    destruct s2 as [| b s2'].
    + (* s1 = a::s1', s2 = [], goal: dam(a::(s1'++[c]), [d]) <= 1 + length s1' + 1 *)
      simpl.
      destruct s1' as [| a' s1''].
      * (* s1 = [a], goal: dam([a,c], [d]) <= 2 *)
        (* [a,c] has 2 elements, [d] has 1. Use multi_single *)
        simpl.
        rewrite damerau_lev_multi_single.
        unfold min3.
        (* opt1 = dam([c], [d]) + 1 = subst(c,d) + 1 = 1 + 1 = 2 (if c <> d) *)
        etransitivity.
        { apply Nat.le_min_l. }
        rewrite damerau_lev_single.
        unfold subst_cost, char_eq.
        destruct (Ascii.ascii_dec c d); [contradiction | simpl; lia].
      * (* s1 = a::a'::s1'', goal: dam(a::a'::(s1''++[c]), [d]) <= 3 + length s1'' *)
        simpl.
        rewrite damerau_lev_multi_single.
        unfold min3.
        (* Use IH on dam(a'::(s1''++[c]), [d]) *)
        apply Nat.le_trans with (damerau_lev_distance (a' :: s1'' ++ [c]) [d] + 1).
        { apply Nat.le_min_l. }
        assert (Heq : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
        rewrite Heq. change [d] with ([] ++ [d]).
        assert (HIH : damerau_lev_distance ((a' :: s1'') ++ [c]) ([] ++ [d]) <=
                      damerau_lev_distance (a' :: s1'') [] + 1).
        { apply (IH (1 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
        rewrite damerau_lev_empty_right in HIH.
        simpl in HIH. simpl. lia.
    + (* s1 = a::s1', s2 = b::s2' *)
      destruct s1' as [| a' s1''].
      * (* s1 = [a] *)
        destruct s2' as [| b' s2''].
        -- (* s1 = [a], s2 = [b], goal: dam([a,c], [b,d]) <= dam([a],[b]) + 1 *)
           simpl app.
           (* Assert RHS value first to avoid wrong rewrite *)
           assert (Hrhs : damerau_lev_distance [a] [b] = subst_cost a b).
           { apply damerau_lev_single. }
           rewrite Hrhs. clear Hrhs.
           rewrite damerau_lev_cons2.
           (* opt3 = dam([c], [d]) + subst_cost a b *)
           unfold min4.
           apply Nat.le_trans with (damerau_lev_distance [c] [d] + subst_cost a b).
           { apply Nat.le_trans with (min (damerau_lev_distance [c] [d] + subst_cost a b)
                                          (damerau_lev_distance [] [] + trans_cost_calc a c b d)).
             { apply Nat.le_min_r. }
             apply Nat.le_min_l. }
           rewrite damerau_lev_single.
           unfold subst_cost, char_eq.
           destruct (Ascii.ascii_dec c d); [contradiction | simpl; lia].
        -- (* s1 = [a], s2 = b::b'::s2'' *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance [a] (b :: b' :: s2'') =
                          min3 (damerau_lev_distance [] (b :: b' :: s2'') + 1)
                               (damerau_lev_distance [a] (b' :: s2'') + 1)
                               (damerau_lev_distance [] (b' :: s2'') + subst_cost a b)).
           { apply damerau_lev_single_multi. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* Use IH bounds *)
           assert (H1 : damerau_lev_distance [c] (b :: b' :: s2'' ++ [d]) <=
                        damerau_lev_distance [] (b :: b' :: s2'') + 1).
           { assert (Heq1 : b :: b' :: s2'' ++ [d] = (b :: b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq1. change [c] with ([] ++ [c]).
             apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance [a; c] (b' :: s2'' ++ [d]) <=
                        damerau_lev_distance [a] (b' :: s2'') + 1).
           { assert (Heq2 : b' :: s2'' ++ [d] = (b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq2. change [a; c] with ([a] ++ [c]).
             apply (IH (2 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance [c] (b' :: s2'' ++ [d]) <=
                        damerau_lev_distance [] (b' :: s2'') + 1).
           { assert (Heq3 : b' :: s2'' ++ [d] = (b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq3. change [c] with ([] ++ [c]).
             apply (IH (1 + length s2'')); [simpl in Hlen; lia | simpl; lia]. }
           (* min4 <= min3 + 1 *)
           (* Strategy: show min4 <= min(opt1,opt2,opt3) <= min(rhs+1,...) = min3 + 1 *)
           (* H1: dam([c],...) <= dam([],b::b'::s2'') + 1 = rhs1 *)
           (* H2: dam([a,c],...) <= dam([a],b'::s2'') + 1 = rhs2 *)
           (* H3: dam([c],...) <= dam([],b'::s2'') + 1 *)
           (* opt1 = dam([c],...) + 1 <= rhs1 + 1 *)
           (* opt2 = dam([a,c],...) + 1 <= rhs2 + 1 *)
           (* opt3 = dam([c],...) + subst <= dam([],b'::s2'') + 1 + subst = rhs3 + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1)
                                              (damerau_lev_distance [a] (b' :: s2'') + 1 + 1))
                                        (damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1)).
           ++ (* min4 <= min(rhs1+1, rhs2+1, rhs3+1) *)
              apply Nat.le_trans with (min (min (damerau_lev_distance [c] (b :: b' :: s2'' ++ [d]) + 1)
                                                 (damerau_lev_distance [a; c] (b' :: s2'' ++ [d]) + 1))
                                           (damerau_lev_distance [c] (b' :: s2'' ++ [d]) + subst_cost a b)).
              ** (* min4 <= min(opt1,opt2,opt3) by showing min(min(A,B),min(C,D)) <= min(min(A,B),C) *)
                 apply Nat.min_le_compat.
                 { apply le_n. }  (* min(opt1,opt2) <= min(opt1,opt2) *)
                 { apply Nat.le_min_l. }  (* min(opt3,opt4) <= opt3 *)
              ** (* min(opt1,opt2,opt3) <= min(rhs1+1, rhs2+1, rhs3+1) using H1, H2, H3 *)
                 apply Nat.min_le_compat.
                 { apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2]. }
                 { (* dam([c],...) + subst <= dam([],b'::s2'') + subst + 1 *)
                   (* From H3: dam([c],...) <= dam([],b'::s2'') + 1 *)
                   lia. }
           ++ (* min(rhs1+1, rhs2+1, rhs3+1) <= min3 + 1 *)
              rewrite Nat.min_assoc.
              (* min(min(a+1+1, b+1+1), c+1) vs min(min(a+1, b+1), c) + 1 *)
              (* = min(min(a+2, b+2), c+1) vs min(min(a+1, b+1), c) + 1 *)
              (* Need: min(a+2, min(b+2, c+1)) <= min(a+1, min(b+1, c)) + 1 *)
              (* This holds since a+2 <= a+1+1, b+2 <= b+1+1, c+1 <= c+1 *)
              assert (Ha : damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1 <=
                           damerau_lev_distance [] (b :: b' :: s2'') + 1 + 1) by lia.
              assert (Hb : damerau_lev_distance [a] (b' :: s2'') + 1 + 1 <=
                           damerau_lev_distance [a] (b' :: s2'') + 1 + 1) by lia.
              assert (Hc : damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1 <=
                           damerau_lev_distance [] (b' :: s2'') + subst_cost a b + 1) by lia.
              lia.
      * (* s1 = a::a'::s1'' *)
        destruct s2' as [| b' s2''].
        -- (* s2 = [b], symmetric case *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') [b] =
                          min3 (damerau_lev_distance (a' :: s1'') [b] + 1)
                               (damerau_lev_distance (a :: a' :: s1'') [] + 1)
                               (damerau_lev_distance (a' :: s1'') [] + subst_cost a b)).
           { apply damerau_lev_multi_single. }
           rewrite Hrhs. clear Hrhs.
           unfold min4, min3.
           (* Use IH bounds *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) [b; d] <=
                        damerau_lev_distance (a' :: s1'') [b] + 1).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq1. change [b; d] with ([b] ++ [d]).
             apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'' ++ [c]) [d] <=
                        damerau_lev_distance (a :: a' :: s1'') [] + 1).
           { assert (Heq2 : a :: a' :: s1'' ++ [c] = (a :: a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq2. change [d] with ([] ++ [d]).
             apply (IH (2 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (a' :: s1'' ++ [c]) [d] <=
                        damerau_lev_distance (a' :: s1'') [] + 1).
           { assert (Heq3 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             rewrite Heq3. change [d] with ([] ++ [d]).
             apply (IH (1 + length s1'')); [simpl in Hlen; lia | simpl; lia]. }
           (* min4 <= min3 + 1 *)
           (* Same strategy as before: show each opt <= rhs + 1, then min4 <= min3 + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'') [b] + 1 + 1)
                                              (damerau_lev_distance (a :: a' :: s1'') [] + 1 + 1))
                                        (damerau_lev_distance (a' :: s1'') [] + subst_cost a b + 1)).
           ++ (* min4 <= min(rhs1+1, rhs2+1, rhs3+1) *)
              apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'' ++ [c]) [b; d] + 1)
                                                 (damerau_lev_distance (a :: a' :: s1'' ++ [c]) [d] + 1))
                                           (damerau_lev_distance (a' :: s1'' ++ [c]) [d] + subst_cost a b)).
              ** (* min4 <= min(opt1,opt2,opt3) *)
                 apply Nat.min_le_compat.
                 { apply le_n. }
                 { apply Nat.le_min_l. }
              ** (* min(opt1,opt2,opt3) <= min(rhs1+1, rhs2+1, rhs3+1) *)
                 apply Nat.min_le_compat.
                 { apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2]. }
                 { lia. }
           ++ (* min(rhs1+1, rhs2+1, rhs3+1) <= min3 + 1 *)
              rewrite Nat.min_assoc. lia.
        -- (* s1 = a::a'::s1'', s2 = b::b'::s2'' - main case *)
           simpl app.
           rewrite damerau_lev_cons2.
           (* Assert RHS expansion *)
           assert (Hrhs : damerau_lev_distance (a :: a' :: s1'') (b :: b' :: s2'') =
                          min4 (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1)
                               (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1)
                               (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b)
                               (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b')).
           { apply damerau_lev_cons2. }
           rewrite Hrhs. clear Hrhs.
           unfold min4.
           (* Use IH bounds on all 4 components *)
           assert (H1 : damerau_lev_distance (a' :: s1'' ++ [c]) (b :: b' :: s2'' ++ [d]) <=
                        damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b :: b' :: s2'' ++ [d] = (b :: b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (a :: a' :: s1'' ++ [c]) (b' :: s2'' ++ [d]) <=
                        damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq1 : a :: a' :: s1'' ++ [c] = (a :: a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b' :: s2'' ++ [d] = (b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (a' :: s1'' ++ [c]) (b' :: s2'' ++ [d]) <=
                        damerau_lev_distance (a' :: s1'') (b' :: s2'') + 1).
           { assert (Heq1 : a' :: s1'' ++ [c] = (a' :: s1'') ++ [c]) by reflexivity.
             assert (Heq2 : b' :: s2'' ++ [d] = (b' :: s2'') ++ [d]) by reflexivity.
             rewrite Heq1, Heq2.
             apply (IH (length s1'' + length s2'' + 2)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H4 : damerau_lev_distance (s1'' ++ [c]) (s2'' ++ [d]) <=
                        damerau_lev_distance s1'' s2'' + 1).
           { apply (IH (length s1'' + length s2'')); [simpl in Hlen; lia | reflexivity]. }
           (* min4(lhs_opts) <= min4(rhs_opts) + 1 *)
           (* Strategy: each lhs_opt <= rhs_opt + 1, so min4(lhs) <= min4(rhs) + 1 *)
           (* Use: min(a+1, b+1, c+1, d+1) = min(a,b,c,d) + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (a' :: s1'') (b :: b' :: s2'') + 1 + 1)
                                              (damerau_lev_distance (a :: a' :: s1'') (b' :: s2'') + 1 + 1))
                                        (min (damerau_lev_distance (a' :: s1'') (b' :: s2'') + subst_cost a b + 1)
                                             (damerau_lev_distance s1'' s2'' + trans_cost_calc a a' b b' + 1))).
           ++ (* min4(lhs) <= min4(rhs + 1) *)
              apply Nat.min_le_compat.
              ** apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2].
              ** apply Nat.min_le_compat; lia.
           ++ (* min4(rhs + 1) = min4(rhs) + 1 *)
              rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              rewrite <- Nat.add_min_distr_r.
              lia.
Qed.

(** Transposition snoc bound: adding transposed pairs increases DL distance by at most 1.
    This captures that transposing [a;b] to [b;a] costs 1 in Damerau-Levenshtein. *)
Lemma damerau_lev_distance_transpose_snoc : forall s1 s2 a b,
  a <> b ->
  damerau_lev_distance (s1 ++ [a; b]) (s2 ++ [b; a]) <= damerau_lev_distance s1 s2 + 1.
Proof.
  intros s1 s2 a b Hneq.
  remember (length s1 + length s2) as n eqn:Hlen.
  revert s1 s2 Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros s1 s2 Hlen.
  destruct s1 as [| c1 s1'].
  - (* s1 = [], s1 ++ [a;b] = [a;b] *)
    destruct s2 as [| c2 s2'].
    + (* s2 = [], s2 ++ [b;a] = [b;a] *)
      (* Goal: dam([a;b], [b;a]) <= dam([], []) + 1 = 0 + 1 = 1 *)
      simpl.
      (* dam([a;b], [b;a]) = 1 by damerau_transpose_distinct since a ≠ b *)
      rewrite damerau_transpose_distinct; [lia | exact Hneq].
    + (* s2 = c2::s2', s2 ++ [b;a] = c2::(s2' ++ [b;a]) *)
      destruct s2' as [| c2' s2''].
      * (* s2 = [c2], s2 ++ [b;a] = [c2;b;a] *)
        simpl app.
        rewrite damerau_lev_empty_left. simpl.
        (* dam([a;b], [c2;b;a]) <= 1 + 1 = 2 *)
        (* Use cons2: dam([a;b], [c2;b;a]) = min4(...) *)
        (* [a;b] = a::b::[], [c2;b;a] = c2::b::[a] *)
        rewrite damerau_lev_cons2.
        unfold min4.
        (* min4 structure: min (min opt1 opt2) (min opt3 opt4)
           opt1 = dam([b], [c2;b;a]) + 1
           opt2 = dam([a;b], [b;a]) + 1  <- we want this, = 1 + 1 = 2
           opt3 = dam([b], [b;a]) + subst_cost a c2
           opt4 = dam([], [a]) + trans_cost_calc a b c2 b *)
        apply Nat.le_trans with (damerau_lev_distance [a; b] [b; a] + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance [b] [c2; b; a] + 1)
                                       (damerau_lev_distance [a; b] [b; a] + 1)).
          { apply Nat.le_min_l. }
          apply Nat.le_min_r. }
        rewrite damerau_transpose_distinct; [lia | exact Hneq].
      * (* s2 = c2::c2'::s2'', s2 ++ [b;a] = c2::c2'::(s2''++[b;a]) *)
        simpl app.
        rewrite damerau_lev_empty_left. simpl.
        (* dam([a;b], c2::c2'::(s2''++[b;a])) <= length s2'' + 3 + 1 *)
        (* [a;b] = a::b::[], [c2::c2'::(s2''++[b;a])] with 2+ elements *)
        rewrite damerau_lev_cons2.
        unfold min4.
        (* min4 structure: min (min opt1 opt2) (min opt3 opt4)
           opt1 = dam([b], c2::c2'::(s2''++[b;a])) + 1
           opt2 = dam([a;b], c2'::(s2''++[b;a])) + 1  <- we want this
           opt3 = dam([b], c2'::(s2''++[b;a])) + subst_cost a c2
           opt4 = dam([], s2''++[b;a]) + trans_cost_calc a b c2 c2' *)
        apply Nat.le_trans with (damerau_lev_distance [a; b] (c2' :: s2'' ++ [b; a]) + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance [b] (c2 :: c2' :: s2'' ++ [b; a]) + 1)
                                       (damerau_lev_distance [a; b] (c2' :: s2'' ++ [b; a]) + 1)).
          { apply Nat.le_min_l. }
          apply Nat.le_min_r. }
        (* Now we need to bound dam([a;b], c2'::(s2''++[b;a])) *)
        (* Use IH with s1 = [], s2 = c2'::s2'' *)
        assert (Hlenx : length (@nil Char) + length (c2' :: s2'') < n).
        { simpl in Hlen. simpl. lia. }
        specialize (IH (length (@nil Char) + length (c2' :: s2'')) Hlenx (@nil Char) (c2' :: s2'')).
        assert (Hlen' : length (@nil Char) + length (c2' :: s2'') = length (@nil Char) + length (c2' :: s2'')) by reflexivity.
        specialize (IH Hlen').
        simpl in IH.
        (* IH: dam([a;b], c2'::(s2''++[b;a])) <= dam([], c2'::s2'') + 1 *)
        apply Nat.le_trans with (damerau_lev_distance [] (c2' :: s2'') + 1 + 1).
        { apply Nat.add_le_mono_r. exact IH. }
        rewrite damerau_lev_empty_left. simpl. lia.
  - (* s1 = c1::s1', s1 ++ [a;b] = c1::(s1' ++ [a;b]) *)
    destruct s2 as [| c2 s2'].
    + (* s2 = [], s2 ++ [b;a] = [b;a] *)
      simpl app.
      rewrite damerau_lev_empty_right. simpl.
      (* dam(c1::(s1'++[a;b]), [b;a]) <= length s1' + 2 *)
      (* Both lists have 2+ elements, so we can use cons2 after destructing s1' *)
      destruct s1' as [| c1' s1'tail].
      * (* s1' = [], first list is [c1;a;b] *)
        simpl.
        rewrite damerau_lev_cons2.
        unfold min4.
        (* opt1 = dam([a;b], [b;a]) + 1 = 1 + 1 = 2 *)
        apply Nat.le_trans with (damerau_lev_distance [a; b] [b; a] + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance [a; b] [b; a] + 1)
                                       (damerau_lev_distance [c1; a; b] [a] + 1)).
          { apply Nat.le_min_l. }
          apply Nat.le_min_l. }
        rewrite damerau_transpose_distinct; [lia | exact Hneq].
      * (* s1' = c1'::s1'tail, first list is c1::c1'::(s1'tail++[a;b]) *)
        cbn [app].
        rewrite damerau_lev_cons2.
        unfold min4.
        (* opt1 = dam(c1'::(s1'tail++[a;b]), [b;a]) + 1 *)
        apply Nat.le_trans with (damerau_lev_distance (c1' :: s1'tail ++ [a; b]) [b; a] + 1).
        { apply Nat.le_trans with (min (damerau_lev_distance (c1' :: s1'tail ++ [a; b]) [b; a] + 1)
                                       (damerau_lev_distance (c1 :: c1' :: s1'tail ++ [a; b]) [a] + 1)).
          { apply Nat.le_min_l. }
          apply Nat.le_min_l. }
        (* Use IH on s1' = c1'::s1'tail, s2 = [] *)
        apply Nat.le_trans with (damerau_lev_distance (c1' :: s1'tail) [] + 1 + 1).
        { apply Nat.add_le_mono_r.
          change (c1' :: s1'tail ++ [a; b]) with ((c1' :: s1'tail) ++ [a; b]).
          change [b; a] with ([] ++ [b; a]).
          assert (Hlt : length (c1' :: s1'tail) + 0 < n) by (simpl in *; lia).
          apply (IH (length (c1' :: s1'tail) + 0) Hlt (c1' :: s1'tail) []).
          simpl. lia. }
        rewrite damerau_lev_empty_right. simpl. lia.
    + (* s1 = c1::s1', s2 = c2::s2' *)
      destruct s1' as [| c1' s1''].
      * (* s1 = [c1], s1 ++ [a;b] = [c1;a;b] *)
        destruct s2' as [| c2' s2''].
        -- (* s2 = [c2], s2 ++ [b;a] = [c2;b;a] *)
           simpl app.
           rewrite damerau_lev_single. simpl.
           (* dam([c1;a;b], [c2;b;a]) <= subst(c1,c2) + 1 *)
           rewrite damerau_lev_cons2.
           unfold min4.
           (* min4 structure after cons2:
              opt1 = dam([a;b], [c2;b;a]) + 1
              opt2 = dam([c1;a;b], [b;a]) + 1
              opt3 = dam([a;b], [b;a]) + subst_cost c1 c2  <- we want this
              opt4 = dam([b], [a]) + trans_cost_calc c1 a c2 b *)
           apply Nat.le_trans with (damerau_lev_distance [a; b] [b; a] + subst_cost c1 c2).
           { apply Nat.le_trans with (min (damerau_lev_distance [a; b] [b; a] + subst_cost c1 c2)
                                          (damerau_lev_distance [b] [a] + trans_cost_calc c1 a c2 b)).
             { apply Nat.le_min_r. }
             apply Nat.le_min_l. }
           rewrite damerau_transpose_distinct; [|exact Hneq].
           unfold subst_cost. destruct (char_eq c1 c2); lia.
        -- (* s2 = c2::c2'::s2'', s2 ++ [b;a] = c2::c2'::(s2''++[b;a]) *)
           simpl app.
           rewrite damerau_lev_cons2.
           rewrite damerau_lev_single_multi.
           unfold min4, min3.
           (* Strategy: Show each LHS option_i <= corresponding RHS option_i + 1
              Then min(LHS) <= min(RHS_i + 1) = min(RHS) + 1 *)

           (* H1: lhs1 (delete c1) <= rhs1 + 1 *)
           assert (H1 : damerau_lev_distance [a; b] (c2 :: c2' :: s2'' ++ [b; a]) + 1 <=
                        (damerau_lev_distance [] (c2 :: c2' :: s2'') + 1) + 1).
           { apply Nat.add_le_mono_r.
             change [a; b] with ([] ++ [a; b]).
             assert (Heq : c2 :: c2' :: s2'' ++ [b; a] = (c2 :: c2' :: s2'') ++ [b; a]) by reflexivity.
             rewrite Heq.
             assert (Hsize : 0 + length (c2 :: c2' :: s2'') < n) by (simpl; simpl in Hlen; lia).
             apply (IH _ Hsize [] (c2 :: c2' :: s2'')). simpl; lia. }

           (* H2: lhs2 (insert c2) <= rhs2 + 1 *)
           assert (H2 : damerau_lev_distance [c1; a; b] (c2' :: s2'' ++ [b; a]) + 1 <=
                        (damerau_lev_distance [c1] (c2' :: s2'') + 1) + 1).
           { apply Nat.add_le_mono_r.
             change [c1; a; b] with ([c1] ++ [a; b]).
             assert (Heq : c2' :: s2'' ++ [b; a] = (c2' :: s2'') ++ [b; a]) by reflexivity.
             rewrite Heq.
             assert (Hsize : 1 + length (c2' :: s2'') < n) by (simpl; simpl in Hlen; lia).
             apply (IH _ Hsize [c1] (c2' :: s2'')). simpl; lia. }

           (* H3: lhs3 (substitute) <= rhs3 + 1 *)
           assert (H3 : damerau_lev_distance [a; b] (c2' :: s2'' ++ [b; a]) + subst_cost c1 c2 <=
                        (damerau_lev_distance [] (c2' :: s2'') + subst_cost c1 c2) + 1).
           { assert (HIH : damerau_lev_distance [a; b] (c2' :: s2'' ++ [b; a]) <=
                           damerau_lev_distance [] (c2' :: s2'') + 1).
             { change [a; b] with ([] ++ [a; b]).
               assert (Heq : c2' :: s2'' ++ [b; a] = (c2' :: s2'') ++ [b; a]) by reflexivity.
               rewrite Heq.
               assert (Hsize : 0 + length (c2' :: s2'') < n) by (simpl; simpl in Hlen; lia).
               apply (IH _ Hsize [] (c2' :: s2'')). simpl; lia. }
             lia. }

           (* Combine: min(min(lhs1,lhs2), min(lhs3,lhs4)) <= min(min(rhs1,rhs2),rhs3) + 1 *)
           (* Step 1: Drop lhs4 (transposition) since min4 <= min3 of first 3 options *)
           apply Nat.le_trans with (min (min (damerau_lev_distance [a; b] (c2 :: c2' :: s2'' ++ [b; a]) + 1)
                                             (damerau_lev_distance [c1; a; b] (c2' :: s2'' ++ [b; a]) + 1))
                                        (damerau_lev_distance [a; b] (c2' :: s2'' ++ [b; a]) + subst_cost c1 c2)).
           { apply Nat.min_le_compat_l. apply Nat.le_min_l. }

           (* Step 2: Show min(lhs1,lhs2,lhs3) <= min(rhs1+1,rhs2+1,rhs3+1) using H1,H2,H3 *)
           apply Nat.le_trans with (min (min ((damerau_lev_distance [] (c2 :: c2' :: s2'') + 1) + 1)
                                             ((damerau_lev_distance [c1] (c2' :: s2'') + 1) + 1))
                                        ((damerau_lev_distance [] (c2' :: s2'') + subst_cost c1 c2) + 1)).
           { apply Nat.min_le_compat.
             - apply Nat.min_le_compat; [exact H1 | exact H2].
             - exact H3. }

           (* Step 3: min(min(a+1,b+1),c+1) = min(min(a,b),c) + 1 by lia *)
           lia.
      * (* s1 = c1::c1'::s1'' *)
        destruct s2' as [| c2' s2''].
        -- (* s2 = [c2], so s2 ++ [b;a] = [c2;b;a] *)
           simpl app.
           (* LHS: dam(c1::c1'::(s1''++[a;b]), c2::b::[a]) - multi vs 3-element *)
           (* RHS: dam(c1::c1'::s1'', [c2]) + 1 - multi vs single *)
           rewrite damerau_lev_cons2.
           rewrite damerau_lev_multi_single.
           unfold min4, min3.
           (* Strategy: show each LHS opt <= corresponding RHS opt + 1 *)

           (* H1: delete opt: dam(c1'::(s1''++[a;b]), [c2;b;a]) + 1 <= rhs1 + 1 *)
           assert (H1 : damerau_lev_distance (c1' :: s1'' ++ [a; b]) [c2; b; a] + 1 <=
                        (damerau_lev_distance (c1' :: s1'') [c2] + 1) + 1).
           { apply Nat.add_le_mono_r.
             change (c1' :: s1'' ++ [a; b]) with ((c1' :: s1'') ++ [a; b]).
             change [c2; b; a] with ([c2] ++ [b; a]).
             assert (Hsize : length (c1' :: s1'') + length [c2] < n) by (simpl; simpl in Hlen; lia).
             apply (IH _ Hsize (c1' :: s1'') [c2]). simpl; lia. }

           (* H2: insert opt: dam(c1::c1'::(s1''++[a;b]), [b;a]) + 1 <= rhs2 + 1 *)
           assert (H2 : damerau_lev_distance (c1 :: c1' :: s1'' ++ [a; b]) [b; a] + 1 <=
                        (damerau_lev_distance (c1 :: c1' :: s1'') [] + 1) + 1).
           { apply Nat.add_le_mono_r.
             change (c1 :: c1' :: s1'' ++ [a; b]) with ((c1 :: c1' :: s1'') ++ [a; b]).
             change [b; a] with ([] ++ [b; a]).
             assert (Hsize : length (c1 :: c1' :: s1'') + 0 < n) by (simpl; simpl in Hlen; lia).
             apply (IH _ Hsize (c1 :: c1' :: s1'') []). simpl; lia. }

           (* H3: subst opt: dam(c1'::(s1''++[a;b]), [b;a]) + subst <= rhs3 + 1 *)
           assert (H3 : damerau_lev_distance (c1' :: s1'' ++ [a; b]) [b; a] + subst_cost c1 c2 <=
                        (damerau_lev_distance (c1' :: s1'') [] + subst_cost c1 c2) + 1).
           { assert (HIH : damerau_lev_distance (c1' :: s1'' ++ [a; b]) [b; a] <=
                           damerau_lev_distance (c1' :: s1'') [] + 1).
             { change (c1' :: s1'' ++ [a; b]) with ((c1' :: s1'') ++ [a; b]).
               change [b; a] with ([] ++ [b; a]).
               assert (Hsize : length (c1' :: s1'') + 0 < n) by (simpl; simpl in Hlen; lia).
               apply (IH _ Hsize (c1' :: s1'') []). simpl; lia. }
             lia. }

           (* Combine: min4 <= min3 + 1 *)
           apply Nat.le_trans with (min (min (damerau_lev_distance (c1' :: s1'' ++ [a; b]) [c2; b; a] + 1)
                                             (damerau_lev_distance (c1 :: c1' :: s1'' ++ [a; b]) [b; a] + 1))
                                        (damerau_lev_distance (c1' :: s1'' ++ [a; b]) [b; a] + subst_cost c1 c2)).
           { apply Nat.min_le_compat_l. apply Nat.le_min_l. }
           apply Nat.le_trans with (min (min ((damerau_lev_distance (c1' :: s1'') [c2] + 1) + 1)
                                             ((damerau_lev_distance (c1 :: c1' :: s1'') [] + 1) + 1))
                                        ((damerau_lev_distance (c1' :: s1'') [] + subst_cost c1 c2) + 1)).
           { apply Nat.min_le_compat.
             - apply Nat.min_le_compat; [exact H1 | exact H2].
             - exact H3. }
           lia.
        -- (* s1 = c1::c1'::s1'', s2 = c2::c2'::s2'' *)
           simpl app.
           rewrite damerau_lev_cons2.
           rewrite damerau_lev_cons2.
           unfold min4.
           (* Use IH on all 4 options *)
           assert (H1 : damerau_lev_distance (c1' :: s1'' ++ [a; b]) (c2 :: c2' :: s2'' ++ [b; a]) <=
                        damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1).
           { change (c1' :: s1'' ++ [a; b]) with ((c1' :: s1'') ++ [a; b]).
             change (c2 :: c2' :: s2'' ++ [b; a]) with ((c2 :: c2' :: s2'') ++ [b; a]).
             apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H2 : damerau_lev_distance (c1 :: c1' :: s1'' ++ [a; b]) (c2' :: s2'' ++ [b; a]) <=
                        damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1).
           { change (c1 :: c1' :: s1'' ++ [a; b]) with ((c1 :: c1' :: s1'') ++ [a; b]).
             change (c2' :: s2'' ++ [b; a]) with ((c2' :: s2'') ++ [b; a]).
             apply (IH (length s1'' + length s2'' + 3)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H3 : damerau_lev_distance (c1' :: s1'' ++ [a; b]) (c2' :: s2'' ++ [b; a]) <=
                        damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + 1).
           { change (c1' :: s1'' ++ [a; b]) with ((c1' :: s1'') ++ [a; b]).
             change (c2' :: s2'' ++ [b; a]) with ((c2' :: s2'') ++ [b; a]).
             apply (IH (length s1'' + length s2'' + 2)); [simpl in Hlen; lia | simpl; lia]. }
           assert (H4 : damerau_lev_distance (s1'' ++ [a; b]) (s2'' ++ [b; a]) <=
                        damerau_lev_distance s1'' s2'' + 1).
           { apply (IH (length s1'' + length s2'')); [simpl in Hlen; lia | reflexivity]. }
           apply Nat.le_trans with (min (min (damerau_lev_distance (c1' :: s1'') (c2 :: c2' :: s2'') + 1 + 1)
                                              (damerau_lev_distance (c1 :: c1' :: s1'') (c2' :: s2'') + 1 + 1))
                                        (min (damerau_lev_distance (c1' :: s1'') (c2' :: s2'') + subst_cost c1 c2 + 1)
                                             (damerau_lev_distance s1'' s2'' + trans_cost_calc c1 c1' c2 c2' + 1))).
           { apply Nat.min_le_compat.
             { apply Nat.min_le_compat; apply Nat.add_le_mono_r; [exact H1 | exact H2]. }
             { apply Nat.min_le_compat; [lia | lia]. } }
           rewrite <- Nat.add_min_distr_r.
           rewrite <- Nat.add_min_distr_r.
           rewrite <- Nat.add_min_distr_r.
           lia.
Qed.

(** * Damerau Reachability Partial Distance Invariant *)

(** Key invariant: if we reach position (i, e) via position_reachable_damerau,
    then damerau_lev_distance (firstn i query) dp <= e.

    This generalizes to the final position case where i = |query|.

    For special positions (transposition-in-progress), the invariant holds
    for the *previous* non-special position, and the error count accounts
    for the in-progress transposition.

    We prove a strengthened version that handles both cases, then derive
    the main lemma as a corollary.
*)

(** Strengthened lemma: handles both special and non-special positions.
    For special positions, we track the predecessor's invariant. *)
Lemma reachable_damerau_partial_distance_strong : forall query n dp p,
  position_reachable_damerau query n dp p ->
  (* For non-special positions: standard invariant *)
  (is_special p = false ->
   damerau_lev_distance (firstn (term_index p) query) dp <= num_errors p) /\
  (* For special positions: dp = dp0 ++ [c] where invariant holds for dp0,
     c = query[i+1], and we track the distinctness for transpose_snoc *)
  (is_special p = true ->
   exists dp0 c c',
     dp = dp0 ++ [c] /\
     S (term_index p) < length query /\
     nth_error query (S (term_index p)) = Some c /\
     nth_error query (term_index p) = Some c' /\
     damerau_lev_distance (firstn (term_index p) query) dp0 <= num_errors p - 1).
Proof.
  intros query n dp p Hreach.
  induction Hreach as [
    | dp' i e Hreach' IH Hle He
    | dp' c i e Hreach' IH Hi Hnth
    | dp' c c' i e Hreach' IH Hi Hnth Hneq He
    | dp' c i e Hreach' IH He
    | dp' c c_next i e Hreach' IH Hsi Hnth_next Hnth He
    | dp' c i e Hreach' IH Hi Hnth
  ].
  - (* reach_damerau_initial: position (0, 0), dp = [] *)
    split.
    + intros _. unfold initial_position, std_pos. simpl.
      rewrite damerau_lev_empty_left. simpl. reflexivity.
    + intros Hspec. unfold initial_position, std_pos in Hspec. simpl in Hspec.
      discriminate.
  - (* reach_damerau_delete: position (S i, S e), dp unchanged *)
    split.
    + intros Hspec. simpl term_index. simpl num_errors. simpl is_special in Hspec.
      assert (Hbound : i < length query) by lia.
      destruct (term_index_bound query i Hbound) as [c Hnth_c].
      rewrite (firstn_S_snoc_nth_error query i c Hnth_c).
      apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp' + 1).
      * apply damerau_lev_distance_snoc_left_bound.
      * destruct IH as [IH_nonspec _].
        specialize (IH_nonspec Hspec). simpl term_index in IH_nonspec.
        simpl num_errors in IH_nonspec. lia.
    + intros Hspec. simpl is_special in Hspec. discriminate.
  - (* reach_damerau_match: position (S i, e), dp extended by c *)
    split.
    + intros Hspec. simpl term_index. simpl num_errors. simpl is_special in Hspec.
      rewrite (firstn_S_snoc_nth_error query i c Hnth).
      apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp').
      * apply damerau_lev_distance_match_last.
      * destruct IH as [IH_nonspec _].
        specialize (IH_nonspec Hspec). simpl term_index in IH_nonspec.
        simpl num_errors in IH_nonspec. exact IH_nonspec.
    + intros Hspec. simpl is_special in Hspec. discriminate.
  - (* reach_damerau_substitute: position (S i, S e), dp extended by c, c ≠ query[i] *)
    split.
    + intros Hspec. simpl term_index. simpl num_errors. simpl is_special in Hspec.
      rewrite (firstn_S_snoc_nth_error query i c' Hnth).
      apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp' + 1).
      * apply damerau_lev_distance_subst_last.
        intro Heq. symmetry in Heq. contradiction.
      * destruct IH as [IH_nonspec _].
        specialize (IH_nonspec Hspec). simpl term_index in IH_nonspec.
        simpl num_errors in IH_nonspec. lia.
    + intros Hspec. simpl is_special in Hspec. discriminate.
  - (* reach_damerau_insert: position (i, S e), dp extended by c *)
    split.
    + intros Hspec. simpl term_index. simpl num_errors. simpl is_special in Hspec.
      apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp' + 1).
      * apply damerau_lev_distance_append_char_bound.
      * destruct IH as [IH_nonspec _].
        specialize (IH_nonspec Hspec). simpl term_index in IH_nonspec.
        simpl num_errors in IH_nonspec. lia.
    + intros Hspec. simpl is_special in Hspec. discriminate.
  - (* reach_damerau_enter_transpose: special position (i, S e), dp extended by c *)
    split.
    + intros Hspec. simpl is_special in Hspec. discriminate.
    + intros _. simpl term_index. simpl num_errors.
      exists dp', c, c_next.
      split. reflexivity.
      split. exact Hsi.
      split. exact Hnth_next.
      split. exact Hnth.
      (* Need: dam(firstn i query, dp') <= S e - 1 = e *)
      destruct IH as [IH_nonspec _].
      assert (Hspec_prev : is_special (std_pos i e) = false) by reflexivity.
      specialize (IH_nonspec Hspec_prev). simpl term_index in IH_nonspec.
      simpl num_errors in IH_nonspec. simpl.
      rewrite Nat.sub_0_r. exact IH_nonspec.
  - (* reach_damerau_complete_transpose: position (S (S i), e), dp extended by c *)
    split.
    + intros Hspec. simpl term_index. simpl num_errors. simpl is_special in Hspec.
      (* Use the special case of IH to get predecessor info *)
      destruct IH as [_ IH_spec].
      assert (Hspec_prev : is_special (special_pos i e) = true) by reflexivity.
      specialize (IH_spec Hspec_prev).
      destruct IH_spec as [dp0 [c_enter [c_next' [Hdp [Hlen [Hnth_enter [Hnth_next' Hdam]]]]]]].
      (* dp' = dp0 ++ [c_enter], c_enter = query[S i] = query[i+1] *)
      (* dp = dp' ++ [c] = dp0 ++ [c_enter] ++ [c] = dp0 ++ [c_enter; c] *)
      (* c = query[i] by Hnth, so c = c_next' *)
      simpl term_index in Hnth_next', Hnth_enter, Hlen, Hdam.
      simpl num_errors in Hdam.
      assert (Hc_eq : c = c_next').
      { rewrite Hnth in Hnth_next'. inversion Hnth_next'. reflexivity. }
      subst c_next'.
      (* firstn (S (S i)) query = firstn i query ++ [query[i]; query[i+1]]
                                = firstn i query ++ [c; c_enter] *)
      assert (Hi_bound : i < length query) by lia.
      assert (HSi_bound : S i < length query) by lia.
      destruct (term_index_bound query i Hi_bound) as [ci Hci].
      destruct (term_index_bound query (S i) HSi_bound) as [cSi HcSi].
      (* ci = query[i] = c, cSi = query[S i] = c_enter *)
      assert (Hci_eq : ci = c) by (rewrite Hci in Hnth; inversion Hnth; reflexivity).
      assert (HcSi_eq : cSi = c_enter) by (rewrite HcSi in Hnth_enter; inversion Hnth_enter; reflexivity).
      subst ci cSi.
      (* Rewrite firstn (S (S i)) query *)
      rewrite (firstn_S_snoc_nth_error query (S i) c_enter HcSi).
      rewrite (firstn_S_snoc_nth_error query i c Hci).
      (* Goal: dam(firstn i query ++ [c; c_enter], dp0 ++ [c_enter; c]) <= e *)
      (* dp = dp' ++ [c] = dp0 ++ [c_enter] ++ [c] *)
      rewrite Hdp.
      assert (Hassoc1 : (firstn i query ++ [c]) ++ [c_enter] = firstn i query ++ [c; c_enter]) by
        (rewrite <- app_assoc; reflexivity).
      assert (Hassoc2 : (dp0 ++ [c_enter]) ++ [c] = dp0 ++ [c_enter; c]) by
        (rewrite <- app_assoc; reflexivity).
      rewrite Hassoc1, Hassoc2.
      (* The executable automaton may enter a special state even when the two
         adjacent query characters are equal.  Distinct characters use the
         transposition branch; equal characters are bounded by two ordinary
         matching suffix steps. *)
      destruct (Ascii.ascii_dec c c_enter) as [Heq_chars | Hneq_chars].
      * subst c_enter.
        replace (firstn i query ++ [c; c]) with ((firstn i query ++ [c]) ++ [c])
          by (rewrite <- app_assoc; reflexivity).
        replace (dp0 ++ [c; c]) with ((dp0 ++ [c]) ++ [c])
          by (rewrite <- app_assoc; reflexivity).
        apply Nat.le_trans with
            (damerau_lev_distance (firstn i query ++ [c]) (dp0 ++ [c])).
        -- apply damerau_lev_distance_match_last.
        -- apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp0).
           ++ apply damerau_lev_distance_match_last.
           ++ lia.
      * apply Nat.le_trans with (damerau_lev_distance (firstn i query) dp0 + 1).
        -- apply damerau_lev_distance_transpose_snoc.
           exact Hneq_chars.
        -- destruct e as [| e'].
           ++ simpl in Hdam. exfalso. inversion Hreach'.
           ++ lia.
    + intros Hspec. simpl is_special in Hspec. discriminate.
Qed.

(** Main lemma: for non-special positions, the partial distance is bounded. *)
Lemma reachable_damerau_partial_distance : forall query n dp p,
  position_reachable_damerau query n dp p ->
  is_special p = false ->
  damerau_lev_distance (firstn (term_index p) query) dp <= num_errors p.
Proof.
  intros query n dp p Hreach Hspec.
  destruct (reachable_damerau_partial_distance_strong query n dp p Hreach) as [H _].
  exact (H Hspec).
Qed.

(** Corollary: when term_index = length query, we get the full DL distance bound *)
Lemma reachable_damerau_partial_to_full : forall query n dp p,
  position_reachable_damerau query n dp p ->
  is_special p = false ->
  term_index p = length query ->
  damerau_lev_distance query dp <= num_errors p.
Proof.
  intros query n dp p Hreach Hspec Hfinal.
  rewrite <- (firstn_all query).
  rewrite <- Hfinal.
  apply (reachable_damerau_partial_distance query n dp p Hreach Hspec).
Qed.

(** Reachable final position bounds Damerau-Levenshtein distance.
    This is the key soundness lemma for the Transposition algorithm.

    Note: For special positions (transposition-in-progress), the position
    doesn't directly represent a valid alignment endpoint. The lemma
    requires the final position to be non-special. *)
Lemma reachable_damerau_final_to_distance : forall query dict n p,
  position_reachable_damerau query n dict p ->
  is_special p = false ->
  term_index p = length query ->
  damerau_lev_distance query dict <= num_errors p.
Proof.
  intros query dict n p Hreach Hspec Hfinal.
  apply (reachable_damerau_partial_to_full query n dict p Hreach Hspec Hfinal).
Qed.

(** * Automaton Run Preserves Reachability *)

(** To use the soundness lemma, we need to show that positions in the final
    state after automaton_run are indeed reachable from the initial state.
    This requires showing that each transition step preserves reachability. *)

(** Initial state contains only reachable positions *)
Lemma initial_state_reachable : forall query n p,
  In p (positions (initial_state Standard (length query))) ->
  position_reachable query n [] p.
Proof.
  intros query n p Hin.
  unfold initial_state in Hin.
  simpl in Hin.
  destruct Hin as [Heq | []].
  subst p.
  apply reach_initial.
Qed.

(** * Transition Soundness *)

(** For the Standard algorithm, each transition type preserves reachability:
    - Match: if cv[0] = true and i < qlen, position (i+1, e) is reachable via reach_match
    - Substitute: if cv[0] = false and e < n, position (i+1, e+1) is reachable via reach_substitute
    - Insert: if e < n, position (i, e+1) is reachable via reach_insert
    These correspond to the edit operations in Levenshtein distance.

    The key insight is that the characteristic vector encodes whether the dictionary
    character matches query[i], and each transition rule directly corresponds to an
    edit operation constructor in position_reachable.
*)

(** Transition preserves reachability - case analysis on transition type.
    Here we use a per-position characteristic vector: cv is created starting at
    term_index p, so min_i = term_index p and the offset = i - min_i = 0. *)
Lemma transition_preserves_reachable_standard : forall query n dict_prefix c p p',
  position_reachable query n dict_prefix p ->
  is_special p = false ->
  In p' (transition_position_standard p (characteristic_vector c query (term_index p) (2*n+6)) (term_index p) n (length query)) ->
  position_reachable query n (dict_prefix ++ [c]) p'.
Proof.
  intros query n dict_prefix c p p' Hreach Hspec Hin_trans.
  unfold transition_position_standard in Hin_trans.
  rewrite Hspec in Hin_trans.
  (* The cv is characteristic_vector c query (term_index p) (2*n+6) with min_i = term_index p.
     So offset = term_index p - term_index p = 0.
     The cv check at offset 0 tells us char_matches_at c query (term_index p). *)
  set (cv := characteristic_vector c query (term_index p) (2 * n + 6)) in *.
  (* offset = term_index p - term_index p simplifies to 0 *)
  replace (term_index p - term_index p) with 0 in Hin_trans by lia.
  apply in_app_or in Hin_trans.
  destruct Hin_trans as [Hin | Hin].
  - (* Match or Substitute - depends on cv_at *)
    destruct (term_index p <? length query) eqn:Hlt.
    2: { simpl in Hin. contradiction. }
    destruct (cv_at cv 0) eqn:Hcv.
    + (* Match case *)
      destruct Hin as [Heq | []]. subst p'.
      destruct p as [i e s]. simpl in *.
      subst s.  (* s = false from Hspec *)
      apply reach_match with (c := c).
      * exact Hreach.
      * apply Nat.ltb_lt. exact Hlt.
      * unfold cv_at in Hcv. unfold cv in Hcv.
        rewrite char_vector_correct in Hcv by lia.
        rewrite Nat.add_0_r in Hcv.
        apply char_matches_at_iff in Hcv.
        destruct Hcv as [c' [Hnth Heq]]. subst. exact Hnth.
    + (* Substitute case *)
      destruct (num_errors p <? n) eqn:He.
      2: { simpl in Hin. contradiction. }
      destruct Hin as [Heq | []]. subst p'.
      destruct p as [i e s]. simpl in *.
      subst s.  (* s = false from Hspec *)
      destruct (nth_error query i) as [c'|] eqn:Hnth.
      * apply reach_substitute with (c' := c').
        -- exact Hreach.
        -- apply Nat.ltb_lt. exact Hlt.
        -- exact Hnth.
        -- intro Heq. subst c'.
           unfold cv_at in Hcv. unfold cv in Hcv.
           rewrite char_vector_correct in Hcv by lia.
           rewrite Nat.add_0_r in Hcv.
           unfold char_matches_at in Hcv. rewrite Hnth in Hcv.
           rewrite char_eq_refl in Hcv. discriminate.
        -- apply Nat.ltb_lt. exact He.
      * apply nth_error_None in Hnth.
        apply Nat.ltb_lt in Hlt. lia.
  - (* Insert case *)
    destruct (num_errors p <? n) eqn:He.
    2: { simpl in Hin. contradiction. }
    destruct Hin as [Heq | []]. subst p'.
    destruct p as [i e s]. simpl in *.
    subst s.  (* s = false from Hspec *)
    apply reach_insert.
    * exact Hreach.
    * apply Nat.ltb_lt. exact He.
Qed.

(** Epsilon closure preserves reachability via delete operations.
    The proof is by induction on the fuel parameter of epsilon_closure_aux.
    Each delete step produces a position reachable via reach_delete.

    Proof sketch:
    1. Base case (fuel = 0): positions unchanged, all are reachable by Hall.
    2. Inductive case: The new positions come from either:
       a. Original positions - reachable by Hall
       b. Delete steps from original positions - reachable by reach_delete
    3. By IH, any position in the final result is reachable.
*)
Lemma epsilon_preserves_reachable : forall query n dict_prefix positions p,
  (forall p0, In p0 positions -> position_reachable query n dict_prefix p0) ->
  In p (epsilon_closure positions n (length query)) ->
  position_reachable query n dict_prefix p.
Proof.
  intros query n dict_prefix positions p Hall Hin.
  unfold epsilon_closure in Hin.
  (* The proof requires careful induction over epsilon_closure_aux.
     Each step adds positions reachable via reach_delete. The key insight
     is that delete_step only produces valid positions that satisfy the
     reach_delete preconditions. *)
  remember (S n) as fuel eqn:Hfuel.
  clear Hfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    set (f := fun p0 : Position =>
                match delete_step p0 n (length query) with
                | Some p' => [p']
                | None => []
                end) in *.
    destruct (is_nil (flat_map f positions)) eqn:Hnil.
    + apply Hall. exact Hin.
    + apply IH in Hin.
      * exact Hin.
      * intros p0 Hin0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall. exact Hin0.
        -- rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           unfold f in Hin0.
           destruct p1 as [i1 e1 s1].
           destruct (delete_step (mkPosition i1 e1 s1) n (length query)) as [p'|] eqn:Hdel.
           ++ destruct Hin0 as [Heq | []]. subst p0.
              unfold delete_step in Hdel.
              simpl in Hdel.
              destruct s1; try discriminate.
              (* Case split on length query to simplify the conditional *)
              destruct (length query) as [|m] eqn:Hqlen.
              { (* length query = 0 *)
                simpl in Hdel. discriminate. }
              (* length query = S m *)
              simpl in Hdel.
              destruct ((i1 <=? m) && (e1 <? n)) eqn:Hcond; try discriminate.
              apply andb_prop in Hcond. destruct Hcond as [Hle' He].
              apply Nat.leb_le in Hle'. apply Nat.ltb_lt in He.
              inversion Hdel. subst p'.
              apply reach_delete.
              ** exact (Hall _ Hin1).
              ** rewrite Hqlen. lia.
              ** exact He.
           ++ destruct Hin0 as [].
Qed.

(** Epsilon closure on initial state gives reachable positions *)
Lemma initial_epsilon_reachable : forall query n p,
  In p (epsilon_closure (positions (initial_state Standard (length query))) n (length query)) ->
  position_reachable query n [] p.
Proof.
  intros query n p Hin.
  apply epsilon_preserves_reachable with (positions := positions (initial_state Standard (length query))).
  - intros p0 Hin0.
    apply initial_state_reachable. exact Hin0.
  - exact Hin.
Qed.

(** * Final Position Soundness *)

(** A final position means we've consumed the entire query *)
Lemma final_position_query_consumed : forall query n dict p,
  position_reachable query n dict p ->
  position_is_final (length query) p = true ->
  term_index p >= length query.
Proof.
  intros query n dict p Hreach Hfinal.
  apply position_final_iff in Hfinal.
  exact Hfinal.
Qed.

(** * Automaton Run Preserves Reachability *)

(** To complete soundness, we need to show that every position in the final
    state after automaton_run is reachable. This requires connecting the
    automaton run to position_reachable. *)

(** Transition on all positions preserves reachability.
    Uses per-position characteristic vectors where min_i = term_index p. *)
Lemma transition_state_positions_preserve_reachable : forall query n dict_prefix c positions p',
  (forall p, In p positions -> position_reachable query n dict_prefix p) ->
  (forall p, In p positions -> is_special p = false) ->
  In p' (flat_map (fun p => transition_position_standard p
                    (characteristic_vector c query (term_index p) (2*n+6)) (term_index p) n (length query)) positions) ->
  position_reachable query n (dict_prefix ++ [c]) p'.
Proof.
  intros query n dict_prefix c positions p' Hall Hspec Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [Hin_p Hin_p']].
  apply transition_preserves_reachable_standard with (p := p) (dict_prefix := dict_prefix).
  - apply Hall. exact Hin_p.
  - apply Hspec. exact Hin_p.
  - exact Hin_p'.
Qed.

(** Helper: sorted_insert only produces positions from p or the input list *)
Lemma sorted_insert_source : forall p acc p',
  In p' (sorted_insert p acc) ->
  p' = p \/ In p' acc.
Proof.
  intros p acc.
  induction acc as [| q rest IH]; intros p' Hin.
  - simpl in Hin. destruct Hin as [Heq | []]. subst. left. reflexivity.
  - simpl in Hin.
    destruct (position_ltb p q) eqn:Hlt.
    + destruct Hin as [Heq | Hin'].
      * subst. left. reflexivity.
      * right. exact Hin'.
    + destruct (position_eqb p q) eqn:Heqb.
      * destruct Hin as [Heq' | Hin'].
        -- subst. right. left. reflexivity.
        -- right. right. exact Hin'.
      * destruct Hin as [Heq' | Hin'].
        -- subst. right. left. reflexivity.
        -- apply IH in Hin'.
           destruct Hin' as [Hp | Hrest].
           ++ left. exact Hp.
           ++ right. right. exact Hrest.
Qed.

(** Positions in fold_right sorted_insert come from the input list *)
Lemma fold_right_sorted_insert_In : forall positions p,
  In p (fold_right sorted_insert [] positions) ->
  In p positions.
Proof.
  intros positions.
  induction positions as [| q rest IH]; intros p Hin.
  - simpl in Hin. contradiction.
  - simpl in Hin.
    apply sorted_insert_source in Hin.
    destruct Hin as [Heq | Hin'].
    + subst. left. reflexivity.
    + right. apply IH. exact Hin'.
Qed.

(** Positions in antichain_insert come from p or the original list *)
Lemma antichain_insert_source : forall alg qlen p positions p',
  In p' (antichain_insert alg qlen p positions) ->
  p' = p \/ In p' positions.
Proof.
  intros alg qlen p positions p' Hin.
  unfold antichain_insert in Hin.
  destruct (subsumed_by_any alg qlen p positions) eqn:Hsub.
  - right. exact Hin.
  - simpl in Hin. destruct Hin as [Heq | Hin'].
    + subst. left. reflexivity.
    + right. apply in_remove_subsumed in Hin'. destruct Hin' as [Hin' _]. exact Hin'.
Qed.

(** Positions in state_insert come from the new position or the original state *)
Lemma state_insert_source : forall p s p',
  In p' (Automaton.State.positions (state_insert p s)) ->
  p' = p \/ In p' (Automaton.State.positions s).
Proof.
  intros p s p' Hin.
  unfold state_insert in Hin. simpl in Hin.
  apply fold_right_sorted_insert_In in Hin.
  apply antichain_insert_source in Hin.
  exact Hin.
Qed.

(** Positions in folded state_insert are from the input list *)
Lemma fold_state_insert_positions : forall alg qlen positions init_positions p,
  In p (Automaton.State.positions
          (fold_left (fun s p0 => state_insert p0 s)
                     positions
                     (mkState init_positions alg qlen))) ->
  In p init_positions \/ In p positions.
Proof.
  intros alg qlen positions.
  induction positions as [| p0 rest IH]; intros init_positions p Hin.
  - simpl in Hin. left. exact Hin.
  - simpl in Hin.
    (* After state_insert, we have: state_insert p0 (mkState init_positions alg qlen) =
       mkState new_init alg qlen where new_init = positions of state_insert result *)
    remember (state_insert p0 (mkState init_positions alg qlen)) as new_state.
    assert (Hstate_eq : new_state = mkState (Automaton.State.positions new_state) alg qlen).
    { subst new_state. unfold state_insert. simpl. reflexivity. }
    rewrite Hstate_eq in Hin.
    apply IH in Hin.
    destruct Hin as [Hin_new | Hin_rest].
    + (* p is from new_state positions, which came from either p0 or init_positions *)
      subst new_state.
      apply state_insert_source in Hin_new. simpl in Hin_new.
      destruct Hin_new as [Hp0 | Hinit].
      * subst. right. left. reflexivity.
      * left. exact Hinit.
    + right. right. exact Hin_rest.
Qed.

(** Epsilon closure preserves position membership from transitions *)
Lemma epsilon_closure_includes_input : forall p positions n qlen,
  In p positions ->
  In p (epsilon_closure positions n qlen).
Proof.
  intros p positions n qlen Hin.
  unfold epsilon_closure.
  (* epsilon_closure_aux always includes original positions *)
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hin.
  induction fuel as [| fuel' IH]; intros positions Hin.
  - simpl. exact Hin.
  - simpl.
    destruct (is_nil _) eqn:Hnil.
    + exact Hin.
    + apply IH. apply in_or_app. left. exact Hin.
Qed.

(** Standard transition produces non-special positions *)
Lemma transition_position_standard_non_special' : forall p cv min_i n qlen p',
  In p' (transition_position_standard p cv min_i n qlen) ->
  is_special p' = false.
Proof.
  intros p cv min_i n qlen p' Hin.
  unfold transition_position_standard in Hin.
  destruct (is_special p) eqn:Hspec.
  - simpl in Hin. contradiction.
  - apply in_app_or in Hin.
    destruct Hin as [Hin | Hin].
    + (* Match/Substitute case *)
      destruct (term_index p <? qlen) eqn:Hlt; simpl in Hin; try contradiction.
      destruct (cv_at cv (term_index p - min_i)) eqn:Hcv.
      * destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
      * destruct (num_errors p <? n) eqn:He; simpl in Hin; try contradiction.
        destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
    + (* Insert case *)
      destruct (num_errors p <? n) eqn:He; simpl in Hin; try contradiction.
      destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
Qed.

(** Transition state positions are non-special for Standard *)
Lemma transition_state_positions_non_special' : forall positions cv min_i n qlen p',
  In p' (flat_map (fun p => transition_position Standard p cv min_i n qlen) positions) ->
  is_special p' = false.
Proof.
  intros positions cv min_i n qlen p' Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [_ Hin_p']].
  simpl in Hin_p'.
  apply transition_position_standard_non_special' in Hin_p'.
  exact Hin_p'.
Qed.

(** All positions in epsilon closure are non-special when starting from non-special *)
Lemma epsilon_closure_non_special' : forall positions n qlen p,
  (forall p0, In p0 positions -> is_special p0 = false) ->
  In p (epsilon_closure positions n qlen) ->
  is_special p = false.
Proof.
  intros positions n qlen p Hall Hin.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
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
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec; try discriminate.
              destruct (_ && _) eqn:Hcond; try discriminate.
              inversion Hdel. subst. simpl. reflexivity.
           ++ simpl in Hin0. contradiction.
Qed.

(** Positions in fold_left state_insert come from the input list.
    Since state_insert either keeps positions or adds from input,
    all positions in result are from the input list. *)
Lemma fold_state_insert_non_special' : forall alg qlen positions init_positions p,
  In p (Automaton.State.positions
          (fold_left (fun s p0 => state_insert p0 s)
                     positions
                     (mkState init_positions alg qlen))) ->
  (forall p0, In p0 init_positions -> is_special p0 = false) ->
  (forall p0, In p0 positions -> is_special p0 = false) ->
  is_special p = false.
Proof.
  intros alg qlen positions.
  induction positions as [| p0 rest IH]; intros init_positions p Hin Hinit Hpos.
  - simpl in Hin. apply Hinit. exact Hin.
  - simpl in Hin.
    (* The fold_left has form: fold_left f rest (state_insert p0 init) *)
    (* By IH with the new initial state being (state_insert p0 (mkState init_positions alg qlen)) *)
    apply IH with (init_positions := Automaton.State.positions (state_insert p0 (mkState init_positions alg qlen))).
    + exact Hin.
    + (* Positions in state_insert p0 init come from p0 or init_positions *)
      intros p1 Hin1.
      apply state_insert_source in Hin1.
      destruct Hin1 as [Heq | Hin1'].
      * subst. apply Hpos. left. reflexivity.
      * simpl in Hin1'. apply Hinit. exact Hin1'.
    + (* rest positions are non-special *)
      intros p1 Hin1. apply Hpos. right. exact Hin1.
Qed.

(** * Characteristic Vector Compatibility *)

(** Key insight: The transition_state implementation uses a single characteristic vector
    computed from min_i (minimum term_index in state), while per-position reasoning
    uses characteristic vectors at each position's term_index.

    Compatibility: For position p with term_index p >= min_i, accessing the cv
    at offset (term_index p - min_i) gives the same result as accessing a cv
    built starting at term_index p.

    This is because:
    cv = characteristic_vector c query min_i window
    cv_at cv (term_index p - min_i) = nth (term_index p - min_i) cv false
                                    = char_matches_at c query (min_i + (term_index p - min_i))
                                    = char_matches_at c query (term_index p)
*)

(** cv_at on shifted characteristic vector equals char_matches_at *)
Lemma cv_at_char_matches : forall c query min_i window i,
  i < window ->
  cv_at (characteristic_vector c query min_i window) i = char_matches_at c query (min_i + i).
Proof.
  intros c query min_i window i Hi.
  unfold cv_at.
  apply char_vector_correct. exact Hi.
Qed.

(** Compatibility: Using min_i-based cv with shifted offset gives correct result *)
Lemma cv_compatibility : forall c query min_i window term_i,
  min_i <= term_i ->
  term_i - min_i < window ->
  cv_at (characteristic_vector c query min_i window) (term_i - min_i) =
  cv_at (characteristic_vector c query term_i window) 0.
Proof.
  intros c query min_i window term_i Hle Hlt.
  rewrite cv_at_char_matches by exact Hlt.
  rewrite cv_at_char_matches by lia.
  f_equal. lia.
Qed.

(** The window size 2*n+2 is sufficient when the spread of term_indices is bounded.
    In a valid automaton state, the spread of term_indices between positions is
    bounded by 2*n (since positions can diverge by at most n in either direction
    from any common ancestor position). *)
Lemma window_sufficient : forall n min_i term_i,
  min_i <= term_i ->
  term_i - min_i <= 2 * n ->
  term_i - min_i < 2 * n + 2.
Proof.
  intros n min_i term_i Hle Hspread.
  lia.
Qed.

(** Damerau version: window size 2*n+6 is sufficient when spread is bounded by 2*n+5.
    The larger spread accounts for transposition edge cases where special positions
    can have different term_index relationships. *)
Lemma window_sufficient_damerau : forall n min_i term_i,
  min_i <= term_i ->
  term_i - min_i <= 2 * n + 5 ->
  term_i - min_i < 2 * n + 6.
Proof.
  intros n min_i term_i Hle Hspread.
  lia.
Qed.

(** Helper lemma: term_index is bounded above by length dict_prefix + num_errors.
    This is because each delete operation advances term_index without consuming dict. *)
Lemma reachable_term_index_upper_bound : forall query n dict_prefix p,
  position_reachable query n dict_prefix p ->
  term_index p <= length dict_prefix + num_errors p.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) at [] *)
    unfold initial_position. simpl. lia.
  - (* delete: (S i, S e) at dp, IH: i <= length dp + e *)
    simpl. simpl in IHHreach. lia.
  - (* match: (S i, e) at dp ++ [c], IH: i <= length dp + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
  - (* substitute: (S i, S e) at dp ++ [c], IH: i <= length dp + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
  - (* insert: (i, S e) at dp ++ [c], IH: i <= length dp + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
Qed.

(** Helper lemma: length dict_prefix is bounded by term_index + num_errors.
    This is because each insert operation consumes dict without advancing term_index. *)
Lemma reachable_term_index_lower_bound : forall query n dict_prefix p,
  position_reachable query n dict_prefix p ->
  length dict_prefix <= term_index p + num_errors p.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) at [] *)
    unfold initial_position. simpl. lia.
  - (* delete: (S i, S e) at dp, IH: length dp <= i + e *)
    simpl. simpl in IHHreach. lia.
  - (* match: (S i, e) at dp ++ [c], IH: length dp <= i + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
  - (* substitute: (S i, S e) at dp ++ [c], IH: length dp <= i + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
  - (* insert: (i, S e) at dp ++ [c], IH: length dp <= i + e *)
    simpl. simpl in IHHreach. rewrite app_length. simpl. lia.
Qed.

(** Key invariant: term_index of a reachable position is bounded by query length.
    This follows directly from the position_reachable definition:
    - Delete requires S i <= length query
    - Match/Substitute require i < length query (so S i <= length query)
    - Insert doesn't change term_index *)
Lemma reachable_term_index_bound_query : forall query n dict_prefix p,
  position_reachable query n dict_prefix p ->
  term_index p <= length query.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) *)
    unfold initial_position. simpl. lia.
  - (* delete: (S i, S e) - requires S i <= length query *)
    simpl. exact H.
  - (* match: (S i, e) - requires i < length query, so S i <= length query *)
    simpl. lia.
  - (* substitute: (S i, S e) - requires i < length query *)
    simpl. lia.
  - (* insert: (i, S e) - term_index unchanged *)
    simpl. exact IHHreach.
Qed.

(** Helper: Special positions are only created by enter_transpose, which requires
    S (term_index p) < length query. *)
Lemma reachable_damerau_special_bound : forall query n dict_prefix i e,
  position_reachable_damerau query n dict_prefix (special_pos i e) ->
  S i < length query.
Proof.
  intros query n dict_prefix i e Hreach.
  remember (special_pos i e) as p eqn:Hp.
  revert i e Hp.
  induction Hreach; intros i0 e0 Hp; try discriminate.
  - (* enter_transpose: only constructor that produces special_pos *)
    inversion Hp. subst. exact H.
Qed.

(** Damerau version of term_index bound *)
Lemma reachable_damerau_term_index_bound_query : forall query n dict_prefix p,
  position_reachable_damerau query n dict_prefix p ->
  term_index p <= length query.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) *)
    unfold initial_position. simpl. lia.
  - (* delete: (S i, S e) - requires S i <= length query *)
    simpl. exact H.
  - (* match: (S i, e) - requires i < length query, so S i <= length query *)
    simpl. lia.
  - (* substitute: (S i, S e) - requires i < length query *)
    simpl. lia.
  - (* insert: (i, S e) - term_index unchanged *)
    simpl. exact IHHreach.
  - (* enter_transpose: special_pos i (S e) - requires S i < length query, so i <= length query *)
    simpl. lia.
  - (* complete_transpose: std_pos (S (S i)) e *)
    simpl.
    (* The special position (i, e) came from enter_transpose which had S i < length query *)
    assert (Hspec_bound : S i < length query).
    { apply reachable_damerau_special_bound with (n := n) (dict_prefix := dp) (e := e).
      exact Hreach. }
    lia.
Qed.

(** Key invariant: In a valid automaton state, the spread of term_indices
    between any two positions is bounded by 2*n.

    This follows from the automaton structure:
    - From position (i, e), we can reach (i+1, e) via match/substitute,
      (i, e+1) via insert, or (i+1, e+1) via delete
    - The maximum "forward" movement without error is via consecutive matches
    - The maximum "backward" lag is via consecutive inserts (at most n)
    - So maximum spread is 2*n between any two positions in the same state

    For Standard algorithm specifically:
    - All positions start at (0, 0)
    - After processing k dictionary characters, a position (i, e) satisfies:
      - i - (k - e) is the number of extra deletes (advancing query without dict)
      - (k - e) - i is the number of extra inserts (advancing dict without query)
    - Both quantities are bounded by e <= n
    - So for any two positions p1, p2: |term_index p1 - term_index p2| <= n + n = 2n
*)
Lemma state_positions_spread_bound : forall query n dict_prefix positions,
  (forall p, In p positions -> position_reachable query n dict_prefix p) ->
  (forall p, In p positions -> num_errors p <= n) ->
  forall p1 p2,
    In p1 positions -> In p2 positions ->
    term_index p2 <= term_index p1 + 2 * n.
Proof.
  intros query n dict_prefix positions Hreach Herr p1 p2 Hin1 Hin2.
  (* Use the helper lemmas:
     - term_index p2 <= length dict_prefix + num_errors p2 (upper bound)
     - length dict_prefix <= term_index p1 + num_errors p1 (lower bound)

     So: term_index p2 <= (length dict_prefix) + n
                       <= (term_index p1 + n) + n = term_index p1 + 2*n *)
  assert (Hupper : term_index p2 <= length dict_prefix + num_errors p2).
  { eapply reachable_term_index_upper_bound. apply Hreach. exact Hin2. }
  assert (Hlower : length dict_prefix <= term_index p1 + num_errors p1).
  { eapply reachable_term_index_lower_bound. apply Hreach. exact Hin1. }
  assert (He1 : num_errors p1 <= n).
  { apply Herr. exact Hin1. }
  assert (He2 : num_errors p2 <= n).
  { apply Herr. exact Hin2. }
  lia.
Qed.

(** NOTE ON CHARACTERISTIC VECTOR HANDLING:

    The transition_state implementation uses a single characteristic vector computed
    from min_i (the minimum term_index in the state). Each position p accesses the
    cv at offset (term_index p - min_i) to check for character match at query[term_index p].

    This is correct because:
    - cv = characteristic_vector c query min_i window
    - cv[offset] = char_matches_at c query (min_i + offset)
    - For position p with term_index i, offset = i - min_i
    - So cv[i - min_i] = char_matches_at c query (min_i + (i - min_i)) = char_matches_at c query i

    The window size 2*n+2 is sufficient because the spread of term_indices in
    a valid automaton state is bounded by 2*n (see state_positions_spread_bound).
*)

(** fold_left min is bounded by the initial value *)
Lemma fold_left_min_le_init : forall (l : list nat) (init : nat),
  fold_left Nat.min l init <= init.
Proof.
  induction l as [| x rest IH]; intros init.
  - simpl. lia.
  - simpl.
    (* fold_left f (x :: rest) init = fold_left f rest (f init x) *)
    (* so: fold_left Nat.min rest (Nat.min init x) <= init *)
    specialize (IH (Nat.min init x)).
    (* IH: fold_left Nat.min rest (Nat.min init x) <= Nat.min init x *)
    apply Nat.le_trans with (Nat.min init x).
    + exact IH.
    + apply Nat.le_min_l.
Qed.

(** fold_left min is bounded by any element in the list *)
Lemma fold_left_min_le_elem : forall (l : list nat) (init x : nat),
  In x l ->
  fold_left Nat.min l init <= x.
Proof.
  induction l as [| y rest IH]; intros init x Hin.
  - contradiction.
  - simpl in Hin. destruct Hin as [Heq | Hin].
    + subst x. simpl.
      (* fold_left Nat.min rest (Nat.min init y) <= y *)
      pose proof (fold_left_min_le_init rest (Nat.min init y)) as H.
      apply Nat.le_trans with (Nat.min init y).
      * exact H.
      * apply Nat.le_min_r.
    + simpl. apply IH. exact Hin.
Qed.

(** If fold_left min returns a value less than init, that value is in the list *)
Lemma fold_left_min_in_list : forall (l : list nat) (init : nat),
  fold_left Nat.min l init < init ->
  In (fold_left Nat.min l init) l.
Proof.
  induction l as [| x rest IH]; intros init Hlt.
  - (* Empty list: contradiction since fold_left [] init = init *)
    simpl in Hlt. lia.
  - (* Cons case *)
    simpl in *.
    set (next := Nat.min init x) in *.
    destruct (Nat.lt_ge_cases (fold_left Nat.min rest next) next) as [Hlt_next | Hge_next].
    + (* fold result < next: it's in rest *)
      right. apply IH. exact Hlt_next.
    + (* fold result >= next: fold result = next (since fold result <= next) *)
      pose proof (fold_left_min_le_init rest next) as Hle.
      assert (Heq : fold_left Nat.min rest next = next) by lia.
      rewrite Heq.
      (* next = Nat.min init x. Show next is either init or x. *)
      (* Since fold result = next < init, we have next < init, so next = x *)
      unfold next.
      destruct (Nat.le_gt_cases x init) as [Hle_x | Hgt_x].
      * (* x <= init: min init x = x *)
        rewrite Nat.min_r by exact Hle_x. left. reflexivity.
      * (* x > init: min init x = init, but next < init is contradiction *)
        assert (Hmin_eq : Nat.min init x = init) by lia.
        unfold next in Hlt. rewrite Hmin_eq in Hlt.
        (* fold_left Nat.min rest init < init, but Heq says fold = next = init *)
        unfold next in Heq. rewrite Hmin_eq in Heq. lia.
Qed.

(** The min_i computed via fold_left is <= any position's term_index *)
Lemma min_i_le_term_index : forall positions init p,
  In p positions ->
  fold_left Nat.min (map term_index positions) init <= term_index p.
Proof.
  intros positions init p Hin.
  apply fold_left_min_le_elem.
  apply in_map. exact Hin.
Qed.

(** Helper: A non-empty list has a position with minimum term_index *)
Lemma list_has_min_term_index : forall (positions : list Position),
  positions <> [] ->
  exists p_min, In p_min positions /\
    forall p', In p' positions -> term_index p_min <= term_index p'.
Proof.
  induction positions as [| p0 rest IH]; intros Hne.
  - contradiction.
  - destruct rest as [| p1 rest'].
    + (* Single element *)
      exists p0. split.
      * left. reflexivity.
      * intros p' Hin'. simpl in Hin'. destruct Hin' as [Heq | []].
        subst. lia.
    + (* Multiple elements *)
      assert (Hrest_ne : p1 :: rest' <> []) by discriminate.
      specialize (IH Hrest_ne).
      destruct IH as [p_rest_min [Hin_rest_min Hmin_rest_prop]].
      destruct (Nat.le_gt_cases (term_index p0) (term_index p_rest_min)) as [Hle | Hgt].
      * (* p0 is smaller or equal to rest's minimum -> p0 is overall minimum *)
        exists p0. split.
        -- left. reflexivity.
        -- intros p' Hin'. destruct Hin' as [Heq | Hin'].
           ++ subst. lia.
           ++ apply Nat.le_trans with (term_index p_rest_min).
              ** exact Hle.
              ** apply Hmin_rest_prop. exact Hin'.
      * (* rest's minimum is smaller -> it's overall minimum *)
        exists p_rest_min. split.
        -- right. exact Hin_rest_min.
        -- intros p' Hin'. destruct Hin' as [Heq | Hin'].
           ++ subst. lia.
           ++ apply Hmin_rest_prop. exact Hin'.
Qed.

(** Helper: For any position in a state of reachable positions with bounded errors,
    the spread from min_i is bounded by 2*n.

    When there's a position with term_index < init, min_i equals the minimum term_index
    (not init), and the spread bound gives term_index p - min_i <= 2*n.

    Note: We use the fact that min_i <= term_index p < init implies min_i < init,
    so the initial value doesn't affect the result when we have at least one position
    with term_index < init. *)
Lemma term_index_minus_min_bounded : forall query n dict_prefix positions init p,
  (forall p0, In p0 positions -> position_reachable query n dict_prefix p0) ->
  (forall p0, In p0 positions -> is_special p0 = false) ->
  term_index p < init ->  (* Key: p has term_index < init, so min_i < init *)
  In p positions ->
  positions <> [] ->
  term_index p - fold_left Nat.min (map term_index positions) init <= 2 * n.
Proof.
  intros query n dict_prefix positions init p Hreach Hspec Hlt_init Hin Hne.
  set (min_i := fold_left Nat.min (map term_index positions) init).
  (* min_i <= term_index p since p is in positions *)
  assert (Hmin_le_p : min_i <= term_index p).
  { apply min_i_le_term_index. exact Hin. }
  (* min_i < init since min_i <= term_index p < init *)
  assert (Hmin_lt_init : min_i < init).
  { apply Nat.le_lt_trans with (term_index p); [exact Hmin_le_p | exact Hlt_init]. }
  (* Get the minimum position *)
  destruct (list_has_min_term_index positions Hne) as [p_min [Hin_min Hmin_prop]].
  (* min_i <= term_index p_min *)
  assert (Hmin_le_pmin : min_i <= term_index p_min).
  { apply min_i_le_term_index. exact Hin_min. }
  (* Since min_i < init and min_i = fold_left min ... init, the minimum is achieved
     by some element in the list, not by init. Since p_min has the minimum term_index,
     min_i = term_index p_min. We show this using antisymmetry: min_i <= term_index p_min
     and term_index p_min <= min_i. *)

  (* For term_index p_min <= min_i: We need to show the fold result is at least the minimum
     term_index in the list. This is because the fold with Nat.min can only decrease values,
     and it starts from init. If at any point we see term_index p_min, the result will be
     at most term_index p_min. But we need >= not <=.

     Actually, the fold result is the minimum of init and all elements. Since min_i < init,
     the fold result is the minimum of just the elements. And since p_min has the minimum
     term_index, fold result = term_index p_min.

     To prove term_index p_min <= min_i, note that min_i is the minimum of init and all
     term_indices. Since term_index p_min is in the list, min_i <= term_index p_min (already have).
     And min_i is one of {init} ∪ {term_indices}. Since min_i < init, min_i is some term_index.
     Since term_index p_min is the minimum term_index, term_index p_min <= min_i. *)

  (* We'll show term_index p_min <= min_i. Since we have min_i <= term_index p_min (from Hmin_le_pmin),
     proving term_index p_min <= min_i would give equality.

     The key insight: min_i is either init or some element's term_index.
     Since min_i < init, min_i is some element's term_index.
     Since p_min has the minimum term_index among all positions, min_i >= term_index p_min. *)
  assert (Hpmin_le_min : term_index p_min <= min_i).
  { (* By Hmin_le_pmin, we have min_i <= term_index p_min.
       If min_i < term_index p_min, then since min_i is a term_index (min_i < init),
       we'd have a position with term_index < term_index p_min, contradicting
       that p_min has the minimum term_index.

       So min_i = term_index p_min, hence term_index p_min <= min_i. *)

    (* Proof: assume for contradiction that term_index p_min > min_i.
       Since min_i < init (Hmin_lt_init), min_i is achieved by some element, not init.
       So there exists q in positions with term_index q = min_i < term_index p_min.
       But Hmin_prop says p_min has the minimum term_index, so term_index p_min <= term_index q.
       Contradiction: term_index p_min <= term_index q = min_i < term_index p_min. *)

    (* Since min_i <= term_index p_min (Hmin_le_pmin), we have min_i <= term_index p_min.
       The only way to get term_index p_min <= min_i is if they're equal.
       We established min_i < init, meaning min_i is determined by list elements.
       The fold_left Nat.min over the list with init > all elements gives the minimum element.
       Since p_min is the minimum element, min_i = term_index p_min. *)

    (* Direct argument: Since min_i <= term_index p_min and p_min is the position
       with minimum term_index, and min_i is the fold result over term_indices with
       init > all term_indices, min_i = min(term_indices) = term_index p_min. *)

    (* The key: fold_left Nat.min l init equals min({init} ∪ l).
       Since min_i < init, we have min_i = min(l) where l = map term_index positions.
       Since p_min has the minimum term_index in positions, min_i = term_index p_min.
       So term_index p_min <= min_i is actually equality, hence true. *)
    (* Use fold_left_min_in_list: since min_i < init, min_i is in map term_index positions *)
    assert (Hin_min_i : In min_i (map term_index positions)).
    { apply fold_left_min_in_list. exact Hmin_lt_init. }
    (* So min_i = term_index q for some q in positions *)
    apply in_map_iff in Hin_min_i.
    destruct Hin_min_i as [q [Heq_q Hin_q]].
    (* Since p_min has the minimum term_index, term_index p_min <= term_index q = min_i *)
    specialize (Hmin_prop q Hin_q).
    lia. }
  (* So min_i = term_index p_min *)
  assert (Hmin_eq : min_i = term_index p_min) by lia.
  rewrite Hmin_eq.
  (* Now use the spread bound. Goal: term_index p - term_index p_min <= 2 * n
     state_positions_spread_bound gives: term_index p <= term_index p_min + 2 * n
     These are equivalent since term_index p >= term_index p_min (p_min is minimum). *)
  assert (Hp_ge_pmin : term_index p_min <= term_index p).
  { apply Hmin_prop. exact Hin. }
  assert (Hspread : term_index p <= term_index p_min + 2 * n).
  { apply state_positions_spread_bound with (query := query) (dict_prefix := dict_prefix)
                                            (positions := positions).
    - exact Hreach.
    - intros p0 Hin0. apply reachable_implies_edit_distance with (query := query) (dict_prefix := dict_prefix).
      + apply Hreach. exact Hin0.
      + apply Hspec. exact Hin0.
    - exact Hin_min.
    - exact Hin. }
  lia.
Qed.

(** Damerau version of upper bound with position-dependent slack.
    Standard positions: term_index p <= length dp + num_errors p + 4
    Special positions:  term_index p <= length dp + num_errors p + 3

    The tighter bound for special positions allows complete_transpose to work:
    - enter_transpose: std -> special, IH gives +4, goal needs +3 (slack gained)
    - complete_transpose: special -> std, IH gives +3, goal needs +4 (exactly enough) *)
Lemma reachable_damerau_term_index_upper_bound : forall query n dict_prefix p,
  position_reachable_damerau query n dict_prefix p ->
  term_index p <= length dict_prefix + num_errors p + (if is_special p then 3 else 4).
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) at [], std_pos so is_special = false *)
    unfold initial_position. simpl. lia.
  - (* delete: std_pos (S i) (S e) at dp, std_pos so is_special = false *)
    simpl. simpl in IHHreach. lia.
  - (* match: std_pos (S i) e at dp ++ [c], is_special = false *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* substitute: std_pos (S i) (S e) at dp ++ [c], is_special = false *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* insert: std_pos i (S e) at dp ++ [c], is_special = false *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* enter_transpose: special_pos i (S e) at dp ++ [c], is_special = true *)
    (* IH: i <= length dp + e + 4 (predecessor is std_pos, is_special = false) *)
    (* Goal: i <= length (dp ++ [c]) + S e + 3 = length dp + 1 + e + 4 = length dp + e + 5 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* complete_transpose: std_pos (S (S i)) e at dp ++ [c], is_special = false *)
    (* IH: i <= length dp + e + 3 (predecessor is special_pos, is_special = true) *)
    (* Goal: S (S i) <= length (dp ++ [c]) + e + 4 = length dp + 1 + e + 4 = length dp + e + 5 *)
    (* From IH: i + 2 <= length dp + e + 5 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
Qed.

(** Damerau version of lower bound: length dict_prefix <= term_index p + num_errors p + 1
    The "+1" accounts for insert operations that don't advance term_index. *)
Lemma reachable_damerau_term_index_lower_bound : forall query n dict_prefix p,
  position_reachable_damerau query n dict_prefix p ->
  length dict_prefix <= term_index p + num_errors p + 1.
Proof.
  intros query n dict_prefix p Hreach.
  induction Hreach.
  - (* initial: (0, 0) at [] *)
    unfold initial_position. simpl. lia.
  - (* delete: (S i, S e) at dp, IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach. lia.
  - (* match: (S i, e) at dp ++ [c], IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* substitute: (S i, S e) at dp ++ [c], IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* insert: (i, S e) at dp ++ [c], IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* enter_transpose: special_pos (i, S e) at dp ++ [c], IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
  - (* complete_transpose: std_pos (S (S i), e) at dp ++ [c], IH: length dp <= i + e + 1 *)
    simpl. simpl in IHHreach.
    assert (Hlen: length (dp ++ [c]) = length dp + 1) by (rewrite length_app; reflexivity).
    lia.
Qed.

(** Damerau version of spread bound.
    The bound is 2*n + 5 to account for transposition edge cases:
    upper bound gives +4 slack (worst case for non-special), lower bound gives +1 slack.
    Total: 2n + 4 + 1 = 2n + 5 *)
Lemma state_positions_spread_bound_damerau : forall query n dict_prefix positions,
  (forall p, In p positions -> position_reachable_damerau query n dict_prefix p) ->
  (forall p, In p positions -> num_errors p <= n) ->
  forall p1 p2,
    In p1 positions -> In p2 positions ->
    term_index p2 <= term_index p1 + 2 * n + 5.
Proof.
  intros query n dict_prefix positions Hreach Herr p1 p2 Hin1 Hin2.
  (* Upper bound is conditional: +4 for std, +3 for special. Use worst case +4. *)
  assert (Hupper : term_index p2 <= length dict_prefix + num_errors p2 + 4).
  { pose proof (reachable_damerau_term_index_upper_bound _ _ _ _ (Hreach _ Hin2)) as H.
    destruct (is_special p2); simpl in H; lia. }
  assert (Hlower : length dict_prefix <= term_index p1 + num_errors p1 + 1).
  { eapply reachable_damerau_term_index_lower_bound. apply Hreach. exact Hin1. }
  assert (He1 : num_errors p1 <= n) by (apply Herr; exact Hin1).
  assert (He2 : num_errors p2 <= n) by (apply Herr; exact Hin2).
  lia.
Qed.

(** Damerau version of term_index_minus_min_bounded.
    The spread bound accounts for transposition edge cases with +5. *)
Lemma term_index_minus_min_bounded_damerau : forall query n dict_prefix positions init p,
  (forall p0, In p0 positions -> position_reachable_damerau query n dict_prefix p0) ->
  term_index p < init ->
  In p positions ->
  positions <> [] ->
  term_index p - fold_left Nat.min (map term_index positions) init <= 2 * n + 5.
Proof.
  intros query n dict_prefix positions init p Hreach Hlt_init Hin Hne.
  set (min_i := fold_left Nat.min (map term_index positions) init).
  (* min_i <= term_index p since p is in positions *)
  assert (Hmin_le_p : min_i <= term_index p).
  { apply min_i_le_term_index. exact Hin. }
  (* min_i < init since min_i <= term_index p < init *)
  assert (Hmin_lt_init : min_i < init).
  { apply Nat.le_lt_trans with (term_index p); [exact Hmin_le_p | exact Hlt_init]. }
  (* Get the minimum position *)
  destruct (list_has_min_term_index positions Hne) as [p_min [Hin_min Hmin_prop]].
  (* min_i <= term_index p_min *)
  assert (Hmin_le_pmin : min_i <= term_index p_min).
  { apply min_i_le_term_index. exact Hin_min. }
  (* Show term_index p_min <= min_i using fold_left_min_in_list *)
  assert (Hpmin_le_min : term_index p_min <= min_i).
  { assert (Hin_min_i : In min_i (map term_index positions)).
    { apply fold_left_min_in_list. exact Hmin_lt_init. }
    apply in_map_iff in Hin_min_i.
    destruct Hin_min_i as [q [Heq_q Hin_q]].
    specialize (Hmin_prop q Hin_q).
    lia. }
  (* So min_i = term_index p_min *)
  assert (Hmin_eq : min_i = term_index p_min) by lia.
  rewrite Hmin_eq.
  (* Now show spread bound *)
  assert (Hp_ge_pmin : term_index p_min <= term_index p).
  { apply Hmin_prop. exact Hin. }
  assert (Hspread : term_index p <= term_index p_min + 2 * n + 5).
  { apply state_positions_spread_bound_damerau with (query := query) (dict_prefix := dict_prefix)
                                            (positions := positions).
    - exact Hreach.
    - intros p0 Hin0. apply reachable_damerau_implies_edit_distance with (query := query) (dp := dict_prefix).
      apply Hreach. exact Hin0.
    - exact Hin_min.
    - exact Hin. }
  lia.
Qed.

(** Key lemma: Positions produced by standard transitions from reachable positions
    are themselves reachable. This is the core of the soundness argument.

    The proof works by showing that each transition type (match, substitute, insert)
    corresponds to a valid reachability constructor.

    With the corrected cv offset, the cv check at offset (term_index p - min_i)
    correctly determines whether c matches query[term_index p].

    Note: Requires query_length s = length query invariant. *)
Lemma transition_positions_reachable_standard : forall query n dict_prefix c s p',
  query_length s = length query ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable query n dict_prefix p /\ is_special p = false) ->
  let min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s) in
  let cv := characteristic_vector c query min_i (2 * n + 6) in
  In p' (transition_state_positions Standard (Automaton.State.positions s) cv min_i n (query_length s)) ->
  position_reachable query n (dict_prefix ++ [c]) p'.
Proof.
  intros query n dict_prefix c s p' Hqlen Hall min_i cv Hin.
  unfold transition_state_positions in Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [Hin_p Hin_p']].
  pose proof (Hall p Hin_p) as [Hreach Hspec]. (* Use pose proof to preserve Hall *)
  simpl in Hin_p'.
  unfold transition_position_standard in Hin_p'.
  destruct (is_special p) eqn:Hspec'; try contradiction.
  set (offset := term_index p - min_i) in *.
  apply in_app_or in Hin_p'.
  destruct Hin_p' as [Hin_match | Hin_ins].
  - (* Match or Substitute case *)
    destruct (term_index p <? query_length s) eqn:Hlt; try (simpl in Hin_match; contradiction).
    apply Nat.ltb_lt in Hlt. rewrite Hqlen in Hlt.
    destruct (cv_at cv offset) eqn:Hcv.
    + (* Match case - cv indicates character matches at query[term_index p] *)
      simpl in Hin_match. destruct Hin_match as [Heq | []]. subst.
      (* cv_at cv offset = true means char_matches_at c query (min_i + offset) = true
         = char_matches_at c query (term_index p) = true (since offset = term_index p - min_i)
         So c matches query[term_index p]. Use reach_match. *)
      assert (Hp : p = std_pos (term_index p) (num_errors p)).
      { destruct p as [ti ne sp].
        unfold is_special in Hspec'. simpl in Hspec'.
        destruct sp; try discriminate.
        unfold std_pos. simpl. reflexivity. }
      rewrite Hp in Hreach.
      apply reach_match with (c := c).
      * exact Hreach.
      * exact Hlt.
      * (* Need: nth_error query (term_index p) = Some c *)
        (* From Hcv: cv_at cv offset = true *)
        (* Since offset = term_index p - min_i and we have min_i <= term_index p for positions in state *)
        (* Step 1: min_i <= term_index p *)
        assert (Hmin_le : min_i <= term_index p).
        { apply min_i_le_term_index. exact Hin_p. }
        (* Step 2: offset < 2*n+2 (window size) *)
        assert (Hoffset_bound : offset < 2 * n + 2).
        { unfold offset. apply window_sufficient.
          - exact Hmin_le.
          - (* Use the spread bound lemma *)
            apply term_index_minus_min_bounded with
              (query := query) (dict_prefix := dict_prefix)
              (positions := Automaton.State.positions s).
            + intros p0 Hin0. apply Hall in Hin0. destruct Hin0 as [H _]. exact H.
            + intros p0 Hin0. apply Hall in Hin0. destruct Hin0 as [_ H]. exact H.
            + rewrite Hqlen. exact Hlt.  (* term_index p < length query = query_length s = init *)
            + exact Hin_p.
            + intro Hcontra. rewrite Hcontra in Hin_p. contradiction. }
        (* Step 3: Use cv_at_char_matches *)
        assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
        { apply cv_at_char_matches. lia. }
        (* Step 4: Simplify min_i + offset = term_index p *)
        assert (Hsum : min_i + offset = term_index p).
        { unfold offset. lia. }
        rewrite Hsum in Hcv_matches.
        (* Step 5: char_matches_at c query (term_index p) = true *)
        rewrite Hcv_matches in Hcv.
        (* Step 6: Use char_matches_at_iff *)
        apply char_matches_at_iff in Hcv.
        destruct Hcv as [c' [Hnth Heq]].
        subst c'. exact Hnth.
    + (* Substitute case - cv indicates no match at query[term_index p] *)
      destruct (num_errors p <? n) eqn:Herr; simpl in Hin_match; try contradiction.
      destruct Hin_match as [Heq | []]. subst.
      apply Nat.ltb_lt in Herr.
      (* cv_at cv offset = false means char_matches_at c query (term_index p) = false
         So c ≠ query[term_index p]. Use reach_substitute. *)
      assert (Hp : p = std_pos (term_index p) (num_errors p)).
      { destruct p as [ti ne sp].
        unfold is_special in Hspec'. simpl in Hspec'.
        destruct sp; try discriminate.
        unfold std_pos. simpl. reflexivity. }
      destruct (nth_error query (term_index p)) as [c'|] eqn:Hnth.
      * apply reach_substitute with (c' := c').
        -- rewrite Hp in Hreach. exact Hreach.
        -- exact Hlt.
        -- exact Hnth.
        -- (* Need: c ≠ c' *)
           intro Heq. subst c'.
           (* cv_at cv offset = false but nth_error query (term_index p) = Some c
              means char_matches_at c query (term_index p) should be true - contradiction *)
           (* Step 1: min_i <= term_index p *)
           assert (Hmin_le : min_i <= term_index p).
           { apply min_i_le_term_index. exact Hin_p. }
           (* Step 2: offset < 2*n+2 *)
           assert (Hoffset_bound : offset < 2 * n + 2).
           { unfold offset. apply window_sufficient.
             - exact Hmin_le.
             - (* Use the spread bound lemma *)
               apply term_index_minus_min_bounded with
                 (query := query) (dict_prefix := dict_prefix)
                 (positions := Automaton.State.positions s).
               + intros p0 Hin0. apply Hall in Hin0. destruct Hin0 as [H _]. exact H.
               + intros p0 Hin0. apply Hall in Hin0. destruct Hin0 as [_ H]. exact H.
               + rewrite Hqlen. exact Hlt.  (* term_index p < length query = query_length s = init *)
               + exact Hin_p.
               + intro Hcontra. rewrite Hcontra in Hin_p. contradiction. }
           (* Step 3: cv_at cv offset = char_matches_at c query (min_i + offset) *)
           assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
           { apply cv_at_char_matches. lia. }
           (* Step 4: min_i + offset = term_index p *)
           assert (Hsum : min_i + offset = term_index p).
           { unfold offset. lia. }
           rewrite Hsum in Hcv_matches.
           (* Step 5: char_matches_at c query (term_index p) should be true *)
           assert (Hmatch_true : char_matches_at c query (term_index p) = true).
           { unfold char_matches_at. rewrite Hnth. apply char_eq_refl. }
           (* Step 6: But cv_at cv offset = false, so we have false = true - contradiction *)
           rewrite Hcv_matches in Hcv. rewrite Hmatch_true in Hcv. discriminate.
        -- exact Herr.
      * (* nth_error query (term_index p) = None - contradiction with Hlt *)
        apply nth_error_None in Hnth. lia.
  - (* Insert case *)
    destruct (num_errors p <? n) eqn:Herr; simpl in Hin_ins; try contradiction.
    destruct Hin_ins as [Heq | []]. subst.
    apply Nat.ltb_lt in Herr.
    apply reach_insert.
    assert (Hp : p = std_pos (term_index p) (num_errors p)).
    { destruct p as [ti ne sp].
      unfold is_special in Hspec'. simpl in Hspec'.
      destruct sp; try discriminate.
      unfold std_pos. simpl. reflexivity. }
    rewrite Hp in Hreach.
    exact Hreach.
    exact Herr.
Qed.

(** Epsilon closure preserves reachability by adding delete transitions *)
Lemma epsilon_closure_preserves_reachable : forall query n dict_prefix positions p,
  (forall p0, In p0 positions -> position_reachable query n dict_prefix p0 /\ is_special p0 = false) ->
  In p (epsilon_closure positions n (length query)) ->
  position_reachable query n dict_prefix p.
Proof.
  intros query n dict_prefix positions p Hall Hin.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall in Hin. destruct Hin as [H _]. exact H.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
    + apply Hall in Hin. destruct Hin as [H _]. exact H.
    + apply IH in Hin.
      * exact Hin.
      * (* Need to show new positions are reachable *)
        intros p0 Hin0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall. exact Hin0.
        -- (* p0 is from delete_step applied to some position *)
           rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           destruct (delete_step p1 n (length query)) as [p'|] eqn:Hdel.
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec; try discriminate.
              destruct (_ && _) eqn:Hcond; try discriminate.
              inversion Hdel. subst. clear Hdel.
              apply Bool.andb_true_iff in Hcond.
              destruct Hcond as [Hlt Herr].
              apply Nat.ltb_lt in Hlt.
              apply Nat.ltb_lt in Herr.
              (* p1 is reachable, so delete_step p1 is reachable via reach_delete *)
              split.
              ** apply Hall in Hin1. destruct Hin1 as [Hreach1 Hspec1].
                 (* Since p1 is non-special, p1 = std_pos (term_index p1) (num_errors p1) *)
                 assert (Hp1 : p1 = std_pos (term_index p1) (num_errors p1)).
                 { destruct p1 as [ti ne sp].
                   unfold is_special in Hspec1. simpl in Hspec1.
                   destruct sp; try discriminate.
                   unfold std_pos. simpl. reflexivity. }
                 rewrite Hp1 in Hreach1.
                 apply reach_delete.
                 --- exact Hreach1.
                 --- exact Hlt.
                 --- exact Herr.
              ** simpl. reflexivity.
           ++ simpl in Hin0. contradiction.
Qed.

(** Key lemma: automaton_run preserves reachability.
    If all positions in state s are reachable with dict_prefix dp,
    then after running on dict, all positions are reachable with dp ++ dict.

    Note: This is for Standard algorithm with non-special positions.
    Requires query_length s = length query invariant. *)
Lemma automaton_run_preserves_reachable_standard : forall query n dict_prefix dict s final,
  query_length s = length query ->
  automaton_run Standard query n dict s = Some final ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable query n dict_prefix p /\ is_special p = false) ->
  (forall p, In p (Automaton.State.positions final) ->
             position_reachable query n (dict_prefix ++ dict) p).
Proof.
  intros query n dict_prefix dict.
  revert dict_prefix.
  induction dict as [| c rest IH]; intros dict_prefix s final Hqlen Hrun Hall.
  - (* dict = [] *)
    simpl in Hrun. inversion Hrun. subst.
    intros p Hin.
    rewrite app_nil_r.
    apply Hall in Hin. destruct Hin as [Hreach _]. exact Hreach.
  - (* dict = c :: rest *)
    simpl in Hrun.
    destruct (transition_state Standard s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    (* After transition, positions in s' are reachable with dict_prefix ++ [c] *)
    (* Then by IH, positions in final are reachable with dict_prefix ++ [c] ++ rest *)
    intros p Hin.
    (* Transform goal: dict_prefix ++ c :: rest = (dict_prefix ++ [c]) ++ rest *)
    assert (Heq : dict_prefix ++ c :: rest = (dict_prefix ++ [c]) ++ rest).
    { rewrite <- app_assoc. simpl. reflexivity. }
    rewrite Heq.
    (* First unfold transition_state to get the structure of s' *)
    unfold transition_state in Htrans.
    set (min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s)) in *.
    set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
    set (trans_pos := transition_state_positions Standard (Automaton.State.positions s) cv min_i n (query_length s)) in *.
    set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
    destruct (is_nil closed_pos) eqn:Hnil; [discriminate|].
    inversion Htrans. subst. clear Htrans.
    (* Now s' is the concrete fold expression *)
    (* query_length of the new state = query_length s *)
    (* Need a lemma: fold_left state_insert preserves query_length *)
    assert (Hqlen_pres : forall l s0, query_length (fold_left (fun s1 p1 => state_insert p1 s1) l s0) = query_length s0).
    { induction l as [| p1 l' IHl]; intros s0.
      - simpl. reflexivity.
      - simpl. rewrite IHl. unfold state_insert. simpl. reflexivity. }
    assert (Hqlen' : query_length (fold_left (fun s0 p0 => state_insert p0 s0) closed_pos (empty_state Standard (query_length s))) = length query).
    { rewrite Hqlen_pres. unfold empty_state. simpl. exact Hqlen. }
    apply (IH (dict_prefix ++ [c]) _ final Hqlen' Hrun).
    (* Need to show: positions in s' are reachable with dict_prefix ++ [c] *)
    intros p0 Hin0.
    (* Save a copy of Hin0 before destructive tactics *)
    pose proof Hin0 as Hin0_copy.
    (* Prove reachability *)
    assert (Hreach : position_reachable query n (dict_prefix ++ [c]) p0).
    { simpl in Hin0.
      apply fold_state_insert_positions in Hin0.
      destruct Hin0 as [Hin_init | Hin_closed].
      - contradiction.
      - unfold closed_pos in Hin_closed.
        apply epsilon_closure_preserves_reachable with (positions := trans_pos).
        + intros p1 Hin1.
          unfold trans_pos in Hin1.
          split.
          * apply transition_positions_reachable_standard with (s := s).
            -- exact Hqlen.
            -- exact Hall.
            -- exact Hin1.
          * apply transition_state_positions_non_special' in Hin1.
            exact Hin1.
        + rewrite <- Hqlen. exact Hin_closed. }
    (* Prove non-special *)
    assert (Hnonspec : is_special p0 = false).
    { simpl in Hin0_copy.
      apply fold_state_insert_positions in Hin0_copy.
      destruct Hin0_copy as [Hin_init | Hin_closed].
      - destruct Hin_init.
      - unfold closed_pos in Hin_closed.
        apply epsilon_closure_non_special' with (positions := trans_pos) (n := n) (qlen := query_length s) (p := p0).
        + intros p1 Hin1.
          unfold trans_pos in Hin1.
          apply transition_state_positions_non_special' in Hin1.
          exact Hin1.
        + exact Hin_closed. }
    (* Combine *)
    split; [exact Hreach | exact Hnonspec].
  + (* Final goal from IH: In p (positions final) *)
    exact Hin.
Qed.

(** All positions in epsilon closure are non-special when starting from non-special *)
Lemma epsilon_closure_non_special : forall positions n qlen p,
  (forall p0, In p0 positions -> is_special p0 = false) ->
  In p (epsilon_closure positions n qlen) ->
  is_special p = false.
Proof.
  intros positions n qlen p Hall Hin.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
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
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec; try discriminate.
              destruct (_ && _) eqn:Hcond; try discriminate.
              inversion Hdel. subst. simpl. reflexivity.
           ++ simpl in Hin0. contradiction.
Qed.

(** Initial state positions are reachable and non-special *)
Lemma initial_closed_state_reachable : forall query n p,
  In p (epsilon_closure (Automaton.State.positions (initial_state Standard (length query))) n (length query)) ->
  position_reachable query n [] p /\ is_special p = false.
Proof.
  intros query n p Hin.
  split.
  - apply initial_epsilon_reachable. exact Hin.
  - (* All positions from epsilon closure of initial state are non-special *)
    apply (epsilon_closure_non_special
             (Automaton.State.positions (initial_state Standard (length query)))
             n (length query) p).
    + intros p0 Hin0.
      unfold initial_state in Hin0. simpl in Hin0.
      destruct Hin0 as [Heq | []]. subst.
      unfold initial_position, std_pos. simpl. reflexivity.
    + exact Hin.
Qed.

(** Standard transition produces non-special positions *)
Lemma transition_position_standard_non_special : forall p cv min_i n qlen p',
  In p' (transition_position_standard p cv min_i n qlen) ->
  is_special p' = false.
Proof.
  intros p cv min_i n qlen p' Hin.
  unfold transition_position_standard in Hin.
  destruct (is_special p) eqn:Hspec.
  - simpl in Hin. contradiction.
  - apply in_app_or in Hin.
    destruct Hin as [Hin | Hin].
    + (* Match/Substitute case *)
      destruct (term_index p <? qlen) eqn:Hlt; simpl in Hin; try contradiction.
      destruct (cv_at cv (term_index p - min_i)) eqn:Hcv.
      * destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
      * destruct (num_errors p <? n) eqn:He; simpl in Hin; try contradiction.
        destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
    + (* Insert case *)
      destruct (num_errors p <? n) eqn:He; simpl in Hin; try contradiction.
      destruct Hin as [Heq | []]. subst. unfold std_pos. simpl. reflexivity.
Qed.

(** Transition state positions are non-special for Standard *)
Lemma transition_state_positions_non_special : forall positions cv min_i n qlen p',
  In p' (flat_map (fun p => transition_position Standard p cv min_i n qlen) positions) ->
  is_special p' = false.
Proof.
  intros positions cv min_i n qlen p' Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [_ Hin_p']].
  simpl in Hin_p'.
  apply transition_position_standard_non_special in Hin_p'.
  exact Hin_p'.
Qed.

(** Positions in fold_left state_insert come from the input list.
    Since state_insert either keeps positions or adds from input,
    all positions in result are from the input list. *)
Lemma fold_state_insert_non_special : forall alg qlen positions init_positions p,
  In p (Automaton.State.positions
          (fold_left (fun s p0 => state_insert p0 s)
                     positions
                     (mkState init_positions alg qlen))) ->
  (forall p0, In p0 init_positions -> is_special p0 = false) ->
  (forall p0, In p0 positions -> is_special p0 = false) ->
  is_special p = false.
Proof.
  intros alg qlen positions.
  induction positions as [| p0 rest IH]; intros init_positions p Hin Hinit Hpos.
  - simpl in Hin. apply Hinit. exact Hin.
  - simpl in Hin.
    (* The fold_left has form: fold_left f rest (state_insert p0 init) *)
    (* By IH with the new initial state being (state_insert p0 (mkState init_positions alg qlen)) *)
    apply IH with (init_positions := Automaton.State.positions (state_insert p0 (mkState init_positions alg qlen))).
    + exact Hin.
    + (* Positions in state_insert p0 init come from p0 or init_positions *)
      intros p1 Hin1.
      apply state_insert_source in Hin1.
      destruct Hin1 as [Heq | Hin1'].
      * subst. apply Hpos. left. reflexivity.
      * simpl in Hin1'. apply Hinit. exact Hin1'.
    + (* rest positions are non-special *)
      intros p1 Hin1. apply Hpos. right. exact Hin1.
Qed.

(** All positions in a Standard algorithm run are non-special.
    This is because transition_position_standard only produces std_pos positions,
    and epsilon_closure via delete_step also produces std_pos positions. *)
Lemma standard_run_positions_non_special : forall query n dict s final,
  automaton_run Standard query n dict s = Some final ->
  (forall p, In p (Automaton.State.positions s) -> is_special p = false) ->
  (forall p, In p (Automaton.State.positions final) -> is_special p = false).
Proof.
  intros query n dict.
  induction dict as [| c rest IH]; intros s final Hrun Hall.
  - simpl in Hrun. inversion Hrun. subst. exact Hall.
  - simpl in Hrun.
    destruct (transition_state Standard s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    apply (IH s' final Hrun).
    (* Show positions in s' are non-special *)
    intros p Hin.
    unfold transition_state in Htrans.
    (* transition_state computes:
       1. trans_positions from transition_state_positions (non-special by transition_state_positions_non_special)
       2. closed_positions from epsilon_closure (non-special by epsilon_closure_non_special)
       3. new state from fold_left state_insert *)
    set (min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s)) in *.
    set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
    set (trans_pos := transition_state_positions Standard (Automaton.State.positions s) cv min_i n (query_length s)) in *.
    set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
    destruct (is_nil closed_pos) eqn:Hnil.
    { discriminate. }
    inversion Htrans. subst. clear Htrans.
    simpl in Hin.
    apply fold_state_insert_non_special with
          (alg := Standard) (qlen := query_length s)
          (positions := closed_pos)
          (init_positions := []).
    + exact Hin.
    + intros p0 Hin0. contradiction.
    + intros p0 Hin0.
      unfold closed_pos in Hin0.
      apply (epsilon_closure_non_special trans_pos n (query_length s) p0).
      * intros p1 Hin1.
        unfold trans_pos in Hin1.
        apply transition_state_positions_non_special in Hin1.
        exact Hin1.
      * exact Hin0.
Qed.

(** * Transposition Reachability Infrastructure *)

(** ** Damerau Reachability for Epsilon Closure

    Like standard reachability, epsilon closure preserves Damerau reachability
    because delete_step produces the same non-special positions and uses
    reach_damerau_delete. *)

Lemma epsilon_closure_preserves_reachable_damerau : forall query n dict_prefix positions p,
  (forall p0, In p0 positions -> position_reachable_damerau query n dict_prefix p0) ->
  In p (epsilon_closure positions n (length query)) ->
  position_reachable_damerau query n dict_prefix p.
Proof.
  intros query n dict_prefix positions p Hall Hin.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
    + apply Hall. exact Hin.
    + apply IH in Hin.
      * exact Hin.
      * (* Need to show new positions are damerau-reachable *)
        intros p0 Hin0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall. exact Hin0.
        -- (* p0 is from delete_step applied to some position *)
           rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           destruct (delete_step p1 n (length query)) as [p'|] eqn:Hdel.
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec; try discriminate.
              destruct (_ && _) eqn:Hcond; try discriminate.
              inversion Hdel. subst. clear Hdel.
              apply Bool.andb_true_iff in Hcond.
              destruct Hcond as [Hlt Herr].
              apply Nat.ltb_lt in Hlt.
              apply Nat.ltb_lt in Herr.
              (* p1 is damerau-reachable, so delete_step p1 is reachable via reach_damerau_delete *)
              apply Hall in Hin1.
              (* Since p1 is non-special, p1 = std_pos (term_index p1) (num_errors p1) *)
              assert (Hp1 : p1 = std_pos (term_index p1) (num_errors p1)).
              { destruct p1 as [ti ne sp].
                unfold is_special in Hspec. simpl in Hspec.
                destruct sp; try discriminate.
                unfold std_pos. simpl. reflexivity. }
              rewrite Hp1 in Hin1.
              apply reach_damerau_delete.
              ** exact Hin1.
              ** exact Hlt.
              ** exact Herr.
           ++ simpl in Hin0. contradiction.
Qed.

(** Variant for non-special positions: if all non-special input positions are
    reachable, then all non-special output positions are reachable.
    This is useful when the input contains special positions that may not
    be semantically reachable (e.g., from enter_special when c = c_next). *)
Lemma epsilon_closure_preserves_reachable_damerau_nonspecial : forall query n dict_prefix positions p,
  (forall p0, In p0 positions -> is_special p0 = false -> position_reachable_damerau query n dict_prefix p0) ->
  In p (epsilon_closure positions n (length query)) ->
  is_special p = false ->
  position_reachable_damerau query n dict_prefix p.
Proof.
  intros query n dict_prefix positions p Hall Hin Hspec_p.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall; assumption.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
    + apply Hall; assumption.
    + apply IH in Hin.
      * exact Hin.
      * (* Need to show non-special positions in new list are damerau-reachable *)
        intros p0 Hin0 Hspec0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall; assumption.
        -- (* p0 is from delete_step applied to some position *)
           rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           destruct (delete_step p1 n (length query)) as [p'|] eqn:Hdel.
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec1; try discriminate.
              destruct (_ && _) eqn:Hcond; try discriminate.
              inversion Hdel. subst. clear Hdel.
              apply Bool.andb_true_iff in Hcond.
              destruct Hcond as [Hlt Herr].
              apply Nat.ltb_lt in Hlt.
              apply Nat.ltb_lt in Herr.
              (* p1 is non-special and in positions, so it's damerau-reachable *)
              assert (Hp1_reach : position_reachable_damerau query n dict_prefix p1).
              { apply Hall; assumption. }
              (* p1 = std_pos (term_index p1) (num_errors p1) since it's non-special *)
              assert (Hp1 : p1 = std_pos (term_index p1) (num_errors p1)).
              { destruct p1 as [ti ne sp].
                unfold is_special in Hspec1. simpl in Hspec1.
                destruct sp; try discriminate.
                unfold std_pos. simpl. reflexivity. }
              rewrite Hp1 in Hp1_reach.
              apply reach_damerau_delete.
              ** exact Hp1_reach.
              ** exact Hlt.
              ** exact Herr.
           ++ simpl in Hin0. contradiction.
Qed.

(** ** Initial State Damerau Reachability *)

Lemma initial_epsilon_reachable_damerau : forall query n p,
  In p (epsilon_closure (Automaton.State.positions (initial_state Standard (length query))) n (length query)) ->
  position_reachable_damerau query n [] p.
Proof.
  intros query n p Hin.
  apply epsilon_closure_preserves_reachable_damerau with
        (positions := Automaton.State.positions (initial_state Standard (length query))).
  - intros p0 Hin0.
    unfold initial_state in Hin0. simpl in Hin0.
    destruct Hin0 as [Heq | []]. subst.
    unfold initial_position, std_pos.
    apply reach_damerau_initial.
  - exact Hin.
Qed.

Lemma initial_closed_state_reachable_damerau : forall query n p,
  In p (epsilon_closure (Automaton.State.positions (initial_state Standard (length query))) n (length query)) ->
  position_reachable_damerau query n [] p /\ is_special p = false.
Proof.
  intros query n p Hin.
  split.
  - apply initial_epsilon_reachable_damerau. exact Hin.
  - (* All positions from epsilon closure of initial state are non-special *)
    apply (epsilon_closure_non_special
             (Automaton.State.positions (initial_state Standard (length query)))
             n (length query) p).
    + intros p0 Hin0.
      unfold initial_state in Hin0. simpl in Hin0.
      destruct Hin0 as [Heq | []]. subst.
      unfold initial_position, std_pos. simpl. reflexivity.
    + exact Hin.
Qed.

(** ** Transposition Transition Reachability (Non-Special Only)

    Shows that transposition transitions preserve Damerau reachability for
    NON-SPECIAL positions. Special positions generated when c = c_next
    (query[i] = query[i+1]) are not semantically reachable via
    reach_damerau_enter_transpose, but since special positions cannot be
    final/accepting, this doesn't affect soundness.

    This handles both standard transitions (via inclusion) and the
    special transposition transitions (enter_transpose, complete_transpose). *)

Lemma transition_positions_reachable_transposition : forall query n dict_prefix c s p',
  query_length s = length query ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable_damerau query n dict_prefix p) ->
  let min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s) in
  let cv := characteristic_vector c query min_i (2 * n + 6) in
  In p' (transition_state_positions Transposition (Automaton.State.positions s) cv min_i n (query_length s)) ->
  position_reachable_damerau query n (dict_prefix ++ [c]) p'.
Proof.
  intros query n dict_prefix c s p' Hqlen Hall min_i cv Hin.
  unfold transition_state_positions in Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [Hin_p Hin_p']].
  pose proof (Hall p Hin_p) as Hreach.
  simpl in Hin_p'.
  unfold transition_position_transposition in Hin_p'.
  set (offset := term_index p - min_i) in *.
  destruct (is_special p) eqn:Hspec.
  - (* Special position: complete transposition - p' must be std_pos (S (S i)) e *)
    destruct (term_index p + 1 <? query_length s) eqn:Hlt; try (simpl in Hin_p'; contradiction).
    apply Nat.ltb_lt in Hlt.
    destruct (cv_at cv offset) eqn:Hcv; try (simpl in Hin_p'; contradiction).
    simpl in Hin_p'. destruct Hin_p' as [Heq | []]. subst.
    (* cv_at cv offset = true means c matches query[term_index p] *)
    (* This corresponds to completing the transposition *)
    (* p is special position (i, e) where we saw query[i+1] before, now see query[i] *)
    assert (Hp : p = special_pos (term_index p) (num_errors p)).
    { destruct p as [ti ne sp].
      unfold is_special in Hspec. simpl in Hspec.
      destruct sp; try discriminate.
      unfold special_pos. simpl. reflexivity. }
    rewrite Hp in Hreach.
    rewrite Hqlen in Hlt.
    (* Need: nth_error query (term_index p) = Some c *)
    assert (Hmin_le : min_i <= term_index p).
    { apply min_i_le_term_index. exact Hin_p. }
    (* Derive offset bound from the fact that cv_at returned true *)
    assert (Hoffset_bound : offset < 2 * n + 6).
    { pose proof (cv_at_true_in_bounds cv offset Hcv) as Hbound.
      unfold cv in Hbound. rewrite char_vector_length in Hbound.
      exact Hbound. }
    assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
    { apply cv_at_char_matches. exact Hoffset_bound. }
    assert (Hsum : min_i + offset = term_index p).
    { unfold offset. lia. }
    rewrite Hsum in Hcv_matches.
    rewrite Hcv_matches in Hcv.
    apply char_matches_at_iff in Hcv.
    destruct Hcv as [c' [Hnth Heq']]. subst c'.
    apply reach_damerau_complete_transpose.
    + exact Hreach.
    + lia.  (* From Hlt : term_index p + 1 < length query *)
    + exact Hnth.
  - (* Non-special position: standard transitions + enter special *)
    apply in_app_or in Hin_p'.
    destruct Hin_p' as [Hin_std | Hin_enter].
    + (* Standard transition - reuse standard logic but with Damerau reachability *)
      (* The standard transitions (match/substitute/insert) correspond to
         reach_damerau_match/substitute/insert *)
      unfold transition_position_standard in Hin_std.
      rewrite Hspec in Hin_std.
      apply in_app_or in Hin_std.
      destruct Hin_std as [Hin_ms | Hin_ins].
      * (* Match or Substitute case *)
        destruct (term_index p <? query_length s) eqn:Hlt; try (simpl in Hin_ms; contradiction).
        apply Nat.ltb_lt in Hlt. rewrite Hqlen in Hlt.
        destruct (cv_at cv offset) eqn:Hcv.
        -- (* Match case *)
           (* Need to fold abbreviations before rewriting *)
           fold offset in Hin_ms.
           rewrite Hcv in Hin_ms. simpl in Hin_ms. destruct Hin_ms as [Heq | []]. subst.
           assert (Hp : p = std_pos (term_index p) (num_errors p)).
           { destruct p as [ti ne sp].
             unfold is_special in Hspec. simpl in Hspec.
             destruct sp; try discriminate.
             unfold std_pos. simpl. reflexivity. }
           rewrite Hp in Hreach.
           (* Need: nth_error query (term_index p) = Some c *)
           assert (Hmin_le : min_i <= term_index p).
           { apply min_i_le_term_index. exact Hin_p. }
           (* Derive offset bound from the fact that cv_at returned true *)
           assert (Hoffset_bound : offset < 2 * n + 6).
           { pose proof (cv_at_true_in_bounds cv offset Hcv) as Hbound.
             unfold cv in Hbound. rewrite char_vector_length in Hbound.
             exact Hbound. }
           assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
           { apply cv_at_char_matches. exact Hoffset_bound. }
           assert (Hsum : min_i + offset = term_index p).
           { unfold offset. lia. }
           rewrite Hsum in Hcv_matches.
           rewrite Hcv_matches in Hcv.
           apply char_matches_at_iff in Hcv.
           destruct Hcv as [c' [Hnth Heq']]. subst c'.
           apply reach_damerau_match with (c := c).
           ++ exact Hreach.
           ++ exact Hlt.
           ++ exact Hnth.
        -- (* Substitute case *)
           fold offset in Hin_ms.
           rewrite Hcv in Hin_ms. simpl in Hin_ms.
           destruct (num_errors p <? n) eqn:Herr; simpl in Hin_ms; try contradiction.
           destruct Hin_ms as [Heq | []]. subst.
           apply Nat.ltb_lt in Herr.
           assert (Hp : p = std_pos (term_index p) (num_errors p)).
           { destruct p as [ti ne sp].
             unfold is_special in Hspec. simpl in Hspec.
             destruct sp; try discriminate.
             unfold std_pos. simpl. reflexivity. }
           rewrite Hp in Hreach.
           destruct (nth_error query (term_index p)) as [c'|] eqn:Hnth.
           ++ apply reach_damerau_substitute with (c' := c').
              ** exact Hreach.
              ** exact Hlt.
              ** exact Hnth.
              ** (* Need: c ≠ c' *)
                 intro Heq. subst c'.
                 assert (Hmin_le : min_i <= term_index p).
                 { apply min_i_le_term_index. exact Hin_p. }
                 (* Derive offset bound using Damerau spread lemma *)
                 assert (Hoffset_bound : offset < 2 * n + 6).
                 { unfold offset. apply window_sufficient_damerau.
                   - exact Hmin_le.
                   - apply term_index_minus_min_bounded_damerau with
                       (query := query) (dict_prefix := dict_prefix)
                       (positions := Automaton.State.positions s).
                     + exact Hall.
                     + rewrite Hqlen. exact Hlt.
                     + exact Hin_p.
                     + intro Hcontra. rewrite Hcontra in Hin_p. contradiction. }
                 assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
                 { apply cv_at_char_matches. exact Hoffset_bound. }
                 assert (Hsum : min_i + offset = term_index p).
                 { unfold offset. lia. }
                 rewrite Hsum in Hcv_matches.
                 assert (Hmatch_true : char_matches_at c query (term_index p) = true).
                 { unfold char_matches_at. rewrite Hnth. apply char_eq_refl. }
                 rewrite Hcv_matches in Hcv. rewrite Hmatch_true in Hcv. discriminate.
              ** exact Herr.
           ++ apply nth_error_None in Hnth. lia.
      * (* Insert case *)
        destruct (num_errors p <? n) eqn:Herr; simpl in Hin_ins; try contradiction.
        destruct Hin_ins as [Heq | []]. subst.
        apply Nat.ltb_lt in Herr.
        assert (Hp : p = std_pos (term_index p) (num_errors p)).
        { destruct p as [ti ne sp].
          unfold is_special in Hspec. simpl in Hspec.
          destruct sp; try discriminate.
          unfold std_pos. simpl. reflexivity. }
        rewrite Hp in Hreach.
        apply reach_damerau_insert.
        -- exact Hreach.
        -- exact Herr.
    + (* Enter special state for transposition *)
      destruct ((term_index p + 1 <? query_length s) && (num_errors p <? n)) eqn:Hcond;
        try (simpl in Hin_enter; contradiction).
      destruct (cv_at cv (S offset)) eqn:Hcv; try (simpl in Hin_enter; contradiction).
      simpl in Hin_enter. destruct Hin_enter as [Heq | []]. subst.
      apply Bool.andb_true_iff in Hcond.
      destruct Hcond as [Hlt_next Herr].
      apply Nat.ltb_lt in Hlt_next.
      apply Nat.ltb_lt in Herr.
      rewrite Hqlen in Hlt_next.
      assert (Hp : p = std_pos (term_index p) (num_errors p)).
      { destruct p as [ti ne sp].
        unfold is_special in Hspec. simpl in Hspec.
        destruct sp; try discriminate.
        unfold std_pos. simpl. reflexivity. }
      rewrite Hp in Hreach.
      assert (Hmin_le : min_i <= term_index p).
      { apply min_i_le_term_index. exact Hin_p. }
      assert (Hoffset_bound : S offset < 2 * n + 6).
      { pose proof (cv_at_true_in_bounds cv (S offset) Hcv) as Hbound.
        unfold cv in Hbound. rewrite char_vector_length in Hbound.
        exact Hbound. }
      assert (Hcv_matches : cv_at cv (S offset) = char_matches_at c query (min_i + S offset)).
      { apply cv_at_char_matches. exact Hoffset_bound. }
      assert (Hsum : min_i + S offset = S (term_index p)).
      { unfold offset. lia. }
      rewrite Hsum in Hcv_matches.
      rewrite Hcv_matches in Hcv.
      apply char_matches_at_iff in Hcv.
      destruct Hcv as [c_at_next [Hnth_next Heq_next]]. subst c_at_next.
      destruct (term_index_bound query (term_index p)) as [c_next Hnth_cur].
      { lia. }
      apply reach_damerau_enter_transpose with (c_next := c_next).
      * exact Hreach.
      * lia.
      * exact Hnth_next.
      * exact Hnth_cur.
	      * exact Herr.
Qed.

Lemma transition_positions_reachable_merge_split : forall query n dict_prefix c s p',
  query_length s = length query ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable_merge_split query n dict_prefix p) ->
  let min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s) in
  let cv := characteristic_vector c query min_i (2 * n + 6) in
  In p' (transition_state_positions MergeAndSplit (Automaton.State.positions s) cv min_i n (query_length s)) ->
  position_reachable_merge_split query n (dict_prefix ++ [c]) p'.
Proof.
  intros query n dict_prefix c s p' Hqlen Hall min_i cv Hin.
  unfold transition_state_positions in Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [Hin_p Hin_p']].
  pose proof (Hall p Hin_p) as Hreach.
  unfold transition_position in Hin_p'. simpl in Hin_p'.
  unfold transition_position_merge_split in Hin_p'.
  set (offset := term_index p - min_i) in *.
  destruct (is_special p) eqn:Hspec.
  - (* Complete split. *)
    destruct (term_index p <? query_length s) eqn:Hlt; try (simpl in Hin_p'; contradiction).
    apply Nat.ltb_lt in Hlt. rewrite Hqlen in Hlt.
    simpl in Hin_p'. destruct Hin_p' as [Heq | []]. subst.
    assert (Hp : p = special_pos (term_index p) (num_errors p)).
    { destruct p as [ti ne sp].
      unfold is_special in Hspec. simpl in Hspec.
      destruct sp; try discriminate.
      unfold special_pos. simpl. reflexivity. }
    rewrite Hp in Hreach.
    apply reach_ms_complete_split.
    + exact Hreach.
    + exact Hlt.
  - (* Standard transitions, merge, or enter split. *)
    apply in_app_or in Hin_p'.
    destruct Hin_p' as [Hin_std | Hin_extra].
    + unfold transition_position_standard in Hin_std.
      rewrite Hspec in Hin_std.
      apply in_app_or in Hin_std.
      destruct Hin_std as [Hin_diag | Hin_ins].
      * destruct (term_index p <? query_length s) eqn:Hlt; try (simpl in Hin_diag; contradiction).
        apply Nat.ltb_lt in Hlt. rewrite Hqlen in Hlt.
        destruct (cv_at cv offset) eqn:Hcv.
        -- fold offset in Hin_diag.
           rewrite Hcv in Hin_diag. simpl in Hin_diag.
           destruct Hin_diag as [Heq | []]. subst.
           assert (Hp : p = std_pos (term_index p) (num_errors p)).
           { destruct p as [ti ne sp].
             unfold is_special in Hspec. simpl in Hspec.
             destruct sp; try discriminate.
             unfold std_pos. simpl. reflexivity. }
           rewrite Hp in Hreach.
           assert (Hoffset_bound : offset < 2 * n + 6).
           { pose proof (cv_at_true_in_bounds cv offset Hcv) as Hbound.
             unfold cv in Hbound. rewrite char_vector_length in Hbound.
             exact Hbound. }
           assert (Hcv_matches : cv_at cv offset = char_matches_at c query (min_i + offset)).
           { apply cv_at_char_matches. exact Hoffset_bound. }
           assert (Hmin_le : min_i <= term_index p).
           { apply min_i_le_term_index. exact Hin_p. }
           assert (Hsum : min_i + offset = term_index p).
           { unfold offset. lia. }
           rewrite Hsum in Hcv_matches.
           rewrite Hcv_matches in Hcv.
           apply char_matches_at_iff in Hcv.
           destruct Hcv as [c' [Hnth Heq']]. subst c'.
           apply reach_ms_match with (c := c); assumption.
        -- fold offset in Hin_diag.
           rewrite Hcv in Hin_diag. simpl in Hin_diag.
           destruct (num_errors p <? n) eqn:Herr; simpl in Hin_diag; try contradiction.
           destruct Hin_diag as [Heq | []]. subst.
           apply Nat.ltb_lt in Herr.
           assert (Hp : p = std_pos (term_index p) (num_errors p)).
           { destruct p as [ti ne sp].
             unfold is_special in Hspec. simpl in Hspec.
             destruct sp; try discriminate.
             unfold std_pos. simpl. reflexivity. }
           rewrite Hp in Hreach.
           destruct (term_index_bound query (term_index p) Hlt) as [c' Hnth].
           apply reach_ms_substitute with (c' := c'); assumption.
      * destruct (num_errors p <? n) eqn:Herr; simpl in Hin_ins; try contradiction.
        destruct Hin_ins as [Heq | []]. subst.
        apply Nat.ltb_lt in Herr.
        assert (Hp : p = std_pos (term_index p) (num_errors p)).
        { destruct p as [ti ne sp].
          unfold is_special in Hspec. simpl in Hspec.
          destruct sp; try discriminate.
          unfold std_pos. simpl. reflexivity. }
        rewrite Hp in Hreach.
        apply reach_ms_insert; assumption.
    + apply in_app_or in Hin_extra.
      destruct Hin_extra as [Hin_merge | Hin_split].
      * destruct ((term_index p + 1 <? query_length s) && (num_errors p <? n)) eqn:Hcond;
          try (simpl in Hin_merge; contradiction).
        simpl in Hin_merge. destruct Hin_merge as [Heq | []]. subst.
        apply Bool.andb_true_iff in Hcond.
        destruct Hcond as [Hlt_next Herr].
        apply Nat.ltb_lt in Hlt_next.
        apply Nat.ltb_lt in Herr.
        rewrite Hqlen in Hlt_next.
        assert (Hp : p = std_pos (term_index p) (num_errors p)).
        { destruct p as [ti ne sp].
          unfold is_special in Hspec. simpl in Hspec.
          destruct sp; try discriminate.
          unfold std_pos. simpl. reflexivity. }
        rewrite Hp in Hreach.
        apply reach_ms_merge; [exact Hreach | lia | exact Herr].
      * destruct (num_errors p <? n) eqn:Herr; simpl in Hin_split; try contradiction.
        destruct Hin_split as [Heq | []]. subst.
        apply Nat.ltb_lt in Herr.
        assert (Hp : p = std_pos (term_index p) (num_errors p)).
        { destruct p as [ti ne sp].
          unfold is_special in Hspec. simpl in Hspec.
          destruct sp; try discriminate.
          unfold std_pos. simpl. reflexivity. }
        rewrite Hp in Hreach.
        apply reach_ms_enter_split; assumption.
Qed.

Lemma epsilon_closure_preserves_reachable_merge_split : forall query n dict_prefix positions p,
  (forall p0, In p0 positions -> position_reachable_merge_split query n dict_prefix p0) ->
  In p (epsilon_closure positions n (length query)) ->
  position_reachable_merge_split query n dict_prefix p.
Proof.
  intros query n dict_prefix positions p Hall Hin.
  unfold epsilon_closure in Hin.
  remember (S n) as fuel.
  clear Heqfuel.
  revert positions Hall Hin.
  induction fuel as [| fuel' IH]; intros positions Hall Hin.
  - simpl in Hin. apply Hall. exact Hin.
  - simpl in Hin.
    destruct (is_nil _) eqn:Hnil.
    + apply Hall. exact Hin.
    + apply IH in Hin.
      * exact Hin.
      * intros p0 Hin0.
        apply in_app_or in Hin0.
        destruct Hin0 as [Hin0 | Hin0].
        -- apply Hall. exact Hin0.
        -- rewrite in_flat_map in Hin0.
           destruct Hin0 as [p1 [Hin1 Hin0]].
           destruct (delete_step p1 n (length query)) as [p_del|] eqn:Hdel.
           ++ simpl in Hin0. destruct Hin0 as [Heq | []]. subst.
              unfold delete_step in Hdel.
              destruct (is_special p1) eqn:Hspec; try discriminate.
              destruct ((S (term_index p1) <=? length query) && (num_errors p1 <? n)) eqn:Hcond;
                try discriminate.
              inversion Hdel. subst. clear Hdel.
              apply Bool.andb_true_iff in Hcond.
              destruct Hcond as [Hle Herr].
              apply Nat.leb_le in Hle.
              apply Nat.ltb_lt in Herr.
              assert (Hp : p1 = std_pos (term_index p1) (num_errors p1)).
              { destruct p1 as [ti ne sp].
                unfold is_special in Hspec. simpl in Hspec.
                destruct sp; try discriminate.
                unfold std_pos. simpl. reflexivity. }
              pose proof (Hall p1 Hin1) as Hreach.
              rewrite Hp in Hreach.
              apply reach_ms_delete.
              ** exact Hreach.
              ** exact Hle.
              ** exact Herr.
           ++ simpl in Hin0. contradiction.
Qed.

(** ** Query Length Preservation

    The automaton_run function preserves the query_length field of the state.
    This is because transition_state uses empty_state with the same query_length.
*)

Lemma transition_state_preserves_query_length : forall alg s c q m s',
  transition_state alg s c q m = Some s' ->
  query_length s' = query_length s.
Proof.
  intros alg s c q m s' Htrans.
  unfold transition_state in Htrans.
  destruct (is_nil _) eqn:Hnil; [discriminate|].
  inversion Htrans. subst.
  (* fold_left state_insert preserves query_length of base state *)
  assert (Hfold_pres : forall l s0, query_length (fold_left (fun s1 p1 => state_insert p1 s1) l s0) = query_length s0).
  { induction l as [| p1 l' IHl]; intros s0.
    - simpl. reflexivity.
    - simpl. rewrite IHl. unfold state_insert. simpl. reflexivity. }
  rewrite Hfold_pres. unfold empty_state. simpl. reflexivity.
Qed.

(** Transposition run reachability.

    The invariant tracks semantic Damerau reachability for every position that
    survives the antichain. Special positions no longer need their origin to
    remain in the same state: [position_reachable_damerau] itself records the
    pending transposition step, and antichain filtering only keeps positions
    produced by the transition/closure pipeline. *)
Lemma automaton_run_preserves_reachable_damerau_transposition :
  forall query n dict_prefix dict s final,
  query_length s = length query ->
  automaton_run Transposition query n dict s = Some final ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable_damerau query n dict_prefix p) ->
  (forall p, In p (Automaton.State.positions final) ->
             position_reachable_damerau query n (dict_prefix ++ dict) p).
Proof.
  intros query n dict_prefix dict.
  revert dict_prefix.
  induction dict as [| c rest IH]; intros dict_prefix s final Hqlen Hrun Hall.
  - simpl in Hrun. inversion Hrun. subst.
    intros p Hin.
    rewrite app_nil_r.
    apply Hall. exact Hin.
  - simpl in Hrun.
    destruct (transition_state Transposition s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    intros p Hin.
    assert (Heq : dict_prefix ++ c :: rest = (dict_prefix ++ [c]) ++ rest).
    { rewrite <- app_assoc. simpl. reflexivity. }
    rewrite Heq.
    assert (Hqlen' : query_length s' = length query).
    { rewrite (transition_state_preserves_query_length Transposition s c query n s' Htrans).
      exact Hqlen. }
    apply (IH (dict_prefix ++ [c]) s' final Hqlen' Hrun).
    intros p0 Hin0.
    unfold transition_state in Htrans.
    set (min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s)) in *.
    set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
    set (trans_pos := transition_state_positions Transposition (Automaton.State.positions s) cv min_i n (query_length s)) in *.
    set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
    destruct (is_nil closed_pos) eqn:Hnil; [discriminate|].
    inversion Htrans. subst. clear Htrans.
    simpl in Hin0.
    apply fold_state_insert_positions in Hin0.
    destruct Hin0 as [Hin_empty | Hin_closed].
    + contradiction.
    + unfold closed_pos in Hin_closed.
      rewrite Hqlen in Hin_closed.
      apply epsilon_closure_preserves_reachable_damerau with (positions := trans_pos).
      * intros p1 Hin1.
        unfold trans_pos in Hin1.
        apply transition_positions_reachable_transposition with (s := s).
        -- exact Hqlen.
        -- exact Hall.
        -- exact Hin1.
      * exact Hin_closed.
	  + exact Hin.
Qed.

(** MergeAndSplit run reachability.

    This mirrors the transposition invariant but uses the merge/split semantic
    reachability relation, including split-in-progress special positions. *)
Lemma automaton_run_preserves_reachable_merge_split :
  forall query n dict_prefix dict s final,
  query_length s = length query ->
  automaton_run MergeAndSplit query n dict s = Some final ->
  (forall p, In p (Automaton.State.positions s) ->
             position_reachable_merge_split query n dict_prefix p) ->
  (forall p, In p (Automaton.State.positions final) ->
             position_reachable_merge_split query n (dict_prefix ++ dict) p).
Proof.
  intros query n dict_prefix dict.
  revert dict_prefix.
  induction dict as [| c rest IH]; intros dict_prefix s final Hqlen Hrun Hall.
  - simpl in Hrun. inversion Hrun. subst.
    intros p Hin.
    rewrite app_nil_r.
    apply Hall. exact Hin.
  - simpl in Hrun.
    destruct (transition_state MergeAndSplit s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    intros p Hin.
    assert (Heq : dict_prefix ++ c :: rest = (dict_prefix ++ [c]) ++ rest).
    { rewrite <- app_assoc. simpl. reflexivity. }
    rewrite Heq.
    assert (Hqlen' : query_length s' = length query).
    { rewrite (transition_state_preserves_query_length MergeAndSplit s c query n s' Htrans).
      exact Hqlen. }
    apply (IH (dict_prefix ++ [c]) s' final Hqlen' Hrun).
    intros p0 Hin0.
    unfold transition_state in Htrans.
    set (min_i := fold_left Nat.min (map term_index (Automaton.State.positions s)) (query_length s)) in *.
    set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
    set (trans_pos := transition_state_positions MergeAndSplit (Automaton.State.positions s) cv min_i n (query_length s)) in *.
    set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
    destruct (is_nil closed_pos) eqn:Hnil; [discriminate|].
    inversion Htrans. subst. clear Htrans.
    simpl in Hin0.
    apply fold_state_insert_positions in Hin0.
    destruct Hin0 as [Hin_empty | Hin_closed].
    + contradiction.
    + unfold closed_pos in Hin_closed.
      rewrite Hqlen in Hin_closed.
      apply epsilon_closure_preserves_reachable_merge_split with (positions := trans_pos).
      * intros p1 Hin1.
        unfold trans_pos in Hin1.
        apply transition_positions_reachable_merge_split with (s := s).
        -- exact Hqlen.
        -- exact Hall.
        -- exact Hin1.
      * exact Hin_closed.
    + exact Hin.
Qed.

(** * Main Soundness Theorem *)

Lemma automaton_run_preserves_query_length : forall alg query n dict s final,
  automaton_run alg query n dict s = Some final ->
  query_length final = query_length s.
Proof.
  intros alg query n dict.
  induction dict as [| c rest IH]; intros s final Hrun.
  - simpl in Hrun. inversion Hrun. reflexivity.
  - simpl in Hrun.
    destruct (transition_state alg s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    rewrite (IH s' final Hrun).
    apply transition_state_preserves_query_length with (alg := alg) (c := c) (q := query) (m := n).
    exact Htrans.
Qed.

(** Automaton soundness: if the automaton accepts dict with max distance n,
    then the actual Levenshtein distance is at most n.

    For Standard algorithm, this uses the classic Levenshtein distance.
    For Transposition and MergeAndSplit, analogous theorems hold for
    Damerau-Levenshtein and merge-split distances respectively.
*)

Theorem automaton_sound_standard : forall query dict n,
  automaton_accepts Standard query n dict = true ->
  lev_distance query dict <= n.
Proof.
  intros query dict n Haccept.
  unfold automaton_accepts in Haccept.
  destruct (automaton_run_from_initial Standard query n dict) as [final|] eqn:Hrun.
  2: { discriminate. }
  (* We have a final accepting state *)
  apply state_final_has_final_position in Haccept.
  destruct Haccept as [p [Hin Hfinal]].
  (* Position p is in final state and is accepting *)
  (* By soundness, the path to p witnesses an alignment *)
  (* The alignment cost is num_errors p which is <= n *)
  (* Therefore lev_distance <= alignment cost <= n *)

  (* Step 1: Show p is reachable *)
  unfold automaton_run_from_initial in Hrun.
  set (init_closed := mkState (epsilon_closure
                       (Automaton.State.positions (initial_state Standard (length query)))
                       n (length query)) Standard (length query)) in *.

  (* Positions in init_closed are reachable with [] as dict_prefix *)
  assert (Hinit_reach : forall p0, In p0 (Automaton.State.positions init_closed) ->
                         position_reachable query n [] p0 /\ is_special p0 = false).
  { intros p0 Hin0. unfold init_closed in Hin0. simpl in Hin0.
    apply initial_closed_state_reachable. exact Hin0. }

  (* After running on dict, positions are reachable with dict as dict_prefix *)
  assert (Hreach : position_reachable query n dict p).
  { apply automaton_run_preserves_reachable_standard with
          (query := query) (dict_prefix := []) (s := init_closed) (final := final).
    - (* query_length init_closed = length query *)
      unfold init_closed. simpl. reflexivity.
    - exact Hrun.
    - exact Hinit_reach.
    - exact Hin. }

  (* Step 2: Use reachable_final_to_distance *)
  (* The final position has term_index >= length query *)
  apply position_final_iff in Hfinal.

  (* For final positions, term_index = length query for Standard algorithm *)
  (* (positions can't go past query length in our encoding) *)
  (* We have term_index p >= length query from Hfinal *)

  (* Step 3: Show p is non-special using Standard algorithm property *)
  assert (Hspec : is_special p = false).
  { (* Standard algorithm produces only non-special positions *)
    apply (standard_run_positions_non_special query n dict init_closed final Hrun).
    - intros p1 Hin1. unfold init_closed in Hin1. simpl in Hin1.
      apply initial_closed_state_reachable in Hin1.
      destruct Hin1 as [_ Hsp]. exact Hsp.
    - exact Hin. }

  (* Special case: if term_index p = length query exactly *)
  destruct (Nat.eq_dec (term_index p) (length query)) as [Heq | Hneq].
  - (* term_index p = length query - use reachable_final_to_distance *)
    apply Nat.le_trans with (num_errors p).
    + apply (reachable_final_to_distance query dict n p Hreach Hspec Heq).
    + (* num_errors p <= n by error bound invariant *)
      apply (reachable_implies_edit_distance query dict n p Hreach Hspec).
  - (* term_index p > length query *)
    (* This case is actually impossible by reachable_term_index_bound_query *)
    exfalso.
    assert (Hbound : term_index p <= length query).
    { apply reachable_term_index_bound_query with (n := n) (dict_prefix := dict). exact Hreach. }
    (* Need to show query_length final = length query *)
    assert (Hqlen_final : query_length final = length query).
    { (* query_length is preserved through automaton_run *)
      unfold automaton_run_from_initial in Hrun.
      (* init_closed has query_length = length query *)
      assert (Hqlen_init : query_length init_closed = length query).
      { unfold init_closed. simpl. reflexivity. }
      (* automaton_run preserves query_length *)
      rewrite (automaton_run_preserves_query_length Standard query n dict init_closed final Hrun).
      exact Hqlen_init. }
    (* Now we have all the pieces for contradiction *)
    (* Hfinal : query_length final <= term_index p, i.e., length query <= term_index p *)
    (* Hbound : term_index p <= length query *)
    (* Hneq : term_index p <> length query *)
    (* This is a contradiction *)
    rewrite Hqlen_final in Hfinal.
    lia.
Qed.

(** ** Special Position Invariant for Transposition

    Key invariant: In any state produced by running the Transposition automaton,
    all special positions p satisfy term_index p + 1 < query_length s.
    This means special positions can never be final (since final requires
    query_length s <= term_index p).
*)

(** Special positions in transition outputs satisfy the bound constraint *)
Lemma transition_transposition_special_bound : forall p cv min_i n qlen p',
  In p' (transition_position_transposition p cv min_i n qlen) ->
  is_special p' = true ->
  term_index p' + 1 < qlen.
Proof.
  intros p cv min_i n qlen p' Hin Hspec.
  unfold transition_position_transposition in Hin.
  destruct (is_special p) eqn:Hsp.
  - (* p is special: outputs are [std_pos (S (S i)) e] or [] *)
    destruct (term_index p + 1 <? qlen) eqn:Hlt; try (simpl in Hin; contradiction).
    destruct (cv_at cv (term_index p - min_i)) eqn:Hcv; try (simpl in Hin; contradiction).
    simpl in Hin. destruct Hin as [Heq | []]. subst.
    (* p' = std_pos (S (S (term_index p))) (num_errors p), which is not special *)
    unfold std_pos in Hspec. simpl in Hspec. discriminate.
  - (* p is not special: outputs are standard ++ enter_special *)
    (* The output for non-special p is:
       transition_position_standard p cv min_i n qlen ++ enter_special
       where enter_special = if (term_index p + 1 <? qlen) && (num_errors p <? n)
                             then if cv_at cv (S (term_index p - min_i))
                                  then [special_pos (term_index p) (S (num_errors p))]
                                  else []
                             else [] *)
    simpl in Hin.
    rewrite in_app_iff in Hin.
    destruct Hin as [Hin_std | Hin_enter].
    + (* p' in standard part: all outputs from transition_position_standard are std_pos *)
      (* transition_position_standard only produces std_pos (non-special) *)
      exfalso.
      assert (Hstd_spec : is_special p' = false).
      { apply transition_position_standard_non_special with (p := p) (cv := cv) (min_i := min_i) (n := n) (qlen := qlen).
        exact Hin_std. }
      rewrite Hspec in Hstd_spec. discriminate.
    + (* p' in enter_special part: only special_pos with term_index + 1 < qlen *)
      destruct ((term_index p + 1 <? qlen) && (num_errors p <? n)) eqn:Hcond.
      * destruct (cv_at cv (S (term_index p - min_i))) eqn:Hcv_next.
        -- simpl in Hin_enter.
           destruct Hin_enter as [Heq | []]. subst.
           unfold special_pos. simpl.
           apply andb_prop in Hcond. destruct Hcond as [Hlt _].
           apply Nat.ltb_lt in Hlt. exact Hlt.
        -- simpl in Hin_enter. contradiction.
      * simpl in Hin_enter. contradiction.
Qed.

(** Special positions in transition_state_positions for Transposition satisfy the bound *)
Lemma transition_state_positions_transposition_special_bound : forall positions cv min_i n qlen p',
  In p' (transition_state_positions Transposition positions cv min_i n qlen) ->
  is_special p' = true ->
  term_index p' + 1 < qlen.
Proof.
  intros positions cv min_i n qlen p' Hin Hspec.
  unfold transition_state_positions in Hin.
  rewrite in_flat_map in Hin.
  destruct Hin as [p [_ Hin_trans]].
  simpl in Hin_trans.
  apply transition_transposition_special_bound with (p := p) (cv := cv) (min_i := min_i) (n := n).
  - exact Hin_trans.
  - exact Hspec.
Qed.

(** Helper: delete_step produces non-special positions *)
Lemma delete_step_non_special : forall p n qlen p',
  delete_step p n qlen = Some p' ->
  is_special p' = false.
Proof.
  intros p n qlen p' Hdel.
  unfold delete_step in Hdel.
  destruct (is_special p) eqn:Hsp.
  - (* p is special - delete_step returns None *)
    discriminate.
  - (* p is not special *)
    destruct ((S (term_index p) <=? qlen) && (num_errors p <? n)) eqn:Hcond.
    + inversion Hdel. subst. reflexivity.
    + discriminate.
Qed.

(** Helper lemma: special positions in epsilon_closure_aux come from input *)
Lemma epsilon_closure_aux_special_in_original : forall positions n qlen fuel p,
  In p (epsilon_closure_aux positions n qlen fuel) ->
  is_special p = true ->
  In p positions.
Proof.
  intros positions n qlen fuel.
  revert positions.
  induction fuel as [| fuel' IH]; intros positions p Hin Hspec.
  - (* fuel = 0: epsilon_closure_aux returns positions unchanged *)
    simpl in Hin. exact Hin.
  - (* fuel = S fuel': either returns positions or recurses on positions ++ new_positions *)
    simpl in Hin.
    set (new_positions := flat_map (fun p0 : Position =>
           match delete_step p0 n qlen with
           | Some p' => [p']
           | None => []
           end) positions) in *.
    destruct (is_nil new_positions) eqn:Hnil.
    + (* new_positions is nil, result is just positions *)
      exact Hin.
    + (* new_positions is non-nil, recurse on positions ++ new_positions *)
      apply IH in Hin; [| exact Hspec].
      rewrite in_app_iff in Hin.
      destruct Hin as [Hin_orig | Hin_new].
      * exact Hin_orig.
      * (* p is in new_positions - but all of those are non-special *)
        unfold new_positions in Hin_new.
        rewrite in_flat_map in Hin_new.
        destruct Hin_new as [p0 [_ Hin_del]].
        destruct (delete_step p0 n qlen) as [p'|] eqn:Hdel.
        -- simpl in Hin_del. destruct Hin_del as [Heq | []]. subst p.
           pose proof (delete_step_non_special p0 n qlen p' Hdel) as Hnspec.
           rewrite Hnspec in Hspec. discriminate.
        -- simpl in Hin_del. contradiction.
Qed.

(** Helper: special positions in epsilon closure come from original list *)
Lemma epsilon_closure_special_in_original : forall positions n qlen p,
  In p (epsilon_closure positions n qlen) /\ is_special p = true ->
  In p positions.
Proof.
  intros positions n qlen p [Hin Hspec].
  unfold epsilon_closure in Hin.
  apply epsilon_closure_aux_special_in_original with (n := n) (qlen := qlen) (fuel := S n).
  - exact Hin.
  - exact Hspec.
Qed.

(** Epsilon closure preserves the special bound invariant *)
Lemma epsilon_closure_preserves_special_bound : forall positions n qlen p,
  (forall p0, In p0 positions -> is_special p0 = true -> term_index p0 + 1 < qlen) ->
  In p (epsilon_closure positions n qlen) ->
  is_special p = true ->
  term_index p + 1 < qlen.
Proof.
  intros positions n qlen p Hall Hin Hspec.
  (* Special positions in epsilon_closure must come from original positions,
     since delete_step only produces non-special (std_pos) positions. *)
  assert (Hin_orig : In p positions).
  { apply epsilon_closure_special_in_original with (n := n) (qlen := qlen).
    split; assumption. }
  apply Hall; assumption.
Qed.

(** Transition state for Transposition preserves special bound invariant *)
Lemma transition_state_transposition_special_bound : forall s c query n s',
  query_length s = length query ->
  (forall p, In p (positions s) -> is_special p = true -> term_index p + 1 < query_length s) ->
  transition_state Transposition s c query n = Some s' ->
  (forall p, In p (positions s') -> is_special p = true -> term_index p + 1 < query_length s').
Proof.
  intros s c query n s' Hqlen Hall Htrans p Hin Hspec.
  (* First establish query_length s' = query_length s *)
  assert (Hqlen' : query_length s' = query_length s).
  { apply transition_state_preserves_query_length in Htrans. exact Htrans. }
  rewrite Hqlen'.
  (* Now proceed with the proof *)
  unfold transition_state in Htrans.
  set (min_i := fold_left Nat.min (map term_index (positions s)) (query_length s)) in *.
  set (cv := characteristic_vector c query min_i (2 * n + 6)) in *.
  set (trans_pos := transition_state_positions Transposition (positions s) cv min_i n (query_length s)) in *.
  set (closed_pos := epsilon_closure trans_pos n (query_length s)) in *.
  destruct (is_nil closed_pos) eqn:Hnil; [discriminate|].
  inversion Htrans. subst. clear Htrans.
  simpl in Hin.
  apply fold_state_insert_positions in Hin.
  destruct Hin as [Hin_empty | Hin_closed].
  - (* In empty_state positions = [] *)
    contradiction.
  - (* In closed_pos *)
    assert (Htrans_bound : forall p0, In p0 trans_pos -> is_special p0 = true -> term_index p0 + 1 < query_length s).
    { intros p0 Hin0 Hspec0.
      apply transition_state_positions_transposition_special_bound with
            (positions := positions s) (cv := cv) (min_i := min_i) (n := n).
      exact Hin0. exact Hspec0. }
    apply epsilon_closure_preserves_special_bound with (positions := trans_pos) (n := n).
    + exact Htrans_bound.
    + exact Hin_closed.
    + exact Hspec.
Qed.

(** Automaton run preserves special bound invariant *)
Lemma automaton_run_transposition_special_bound : forall query n dict s final,
  query_length s = length query ->
  (forall p, In p (positions s) -> is_special p = true -> term_index p + 1 < query_length s) ->
  automaton_run Transposition query n dict s = Some final ->
  (forall p, In p (positions final) -> is_special p = true -> term_index p + 1 < query_length final).
Proof.
  intros query n dict.
  induction dict as [| c rest IH]; intros s final Hqlen Hall Hrun.
  - (* dict = [] *)
    simpl in Hrun. inversion Hrun. subst.
    exact Hall.
  - (* dict = c :: rest *)
    simpl in Hrun.
    destruct (transition_state Transposition s c query n) as [s'|] eqn:Htrans.
    2: { discriminate. }
    assert (Hqlen' : query_length s' = length query).
    { apply transition_state_preserves_query_length in Htrans.
      rewrite Htrans. exact Hqlen. }
    apply (IH s' final Hqlen').
    + (* Special bound preserved by transition *)
      apply transition_state_transposition_special_bound with (c := c) (query := query) (n := n) (s := s).
      * exact Hqlen.
      * exact Hall.
      * exact Htrans.
    + exact Hrun.
Qed.

(** Key lemma: In a final state from Transposition automaton run,
    special positions can never be final (accepting). *)
Lemma transposition_final_not_special : forall query n dict final p,
  automaton_run_from_initial Transposition query n dict = Some final ->
  In p (positions final) ->
  position_is_final (query_length final) p = true ->
  is_special p = false.
Proof.
  intros query n dict final p Hrun Hin Hfinal_pos.
  unfold automaton_run_from_initial in Hrun.
  set (init_closed := mkState (epsilon_closure
                       (Automaton.State.positions (initial_state Transposition (length query)))
                       n (length query)) Transposition (length query)) in *.
  (* Initial state has no special positions *)
  assert (Hinit_no_special : forall p0, In p0 (positions init_closed) ->
                                        is_special p0 = true ->
                                        term_index p0 + 1 < query_length init_closed).
  { intros p0 Hin0 Hspec0.
    unfold init_closed in Hin0. simpl in Hin0.
    (* Initial state for Transposition is [initial_position] = [std_pos 0 0] *)
    (* Epsilon closure only adds delete_step which creates std_pos (non-special) *)
    (* So there are no special positions in init_closed *)
    exfalso.
    assert (Hin_init : In p0 [initial_position]).
    { apply epsilon_closure_special_in_original with (n := n) (qlen := length query).
      split; assumption. }
    destruct Hin_init as [Heq | []].
    subst. unfold initial_position in Hspec0. simpl in Hspec0. discriminate. }
  (* query_length init_closed = length query *)
  assert (Hqlen_init : query_length init_closed = length query).
  { unfold init_closed. simpl. reflexivity. }
  (* Apply the run invariant *)
  assert (Hbound : forall p0, In p0 (positions final) ->
                              is_special p0 = true ->
                              term_index p0 + 1 < query_length final).
  { apply automaton_run_transposition_special_bound with (query := query) (n := n) (dict := dict) (s := init_closed).
    - exact Hqlen_init.
    - exact Hinit_no_special.
    - exact Hrun. }
  (* Now show p is not special *)
  destruct (is_special p) eqn:Hspec.
  - (* p is special - derive contradiction *)
    apply Hbound in Hin; [|exact Hspec].
    (* Hin : term_index p + 1 < query_length final *)
    (* Hfinal_pos : query_length final <= term_index p *)
    apply position_final_iff in Hfinal_pos.
    lia.
  - reflexivity.
Qed.

(** Transposition soundness using Damerau-Levenshtein distance.

    The Transposition algorithm accepts strings within Damerau-Levenshtein distance n.
    Since transposition can reduce the edit distance (e.g., "ab" to "ba" is
    distance 1 with transposition but distance 2 with standard operations),
    we use damerau_lev_distance, not lev_distance.
*)
Theorem automaton_sound_transposition : forall query dict n,
  automaton_accepts Transposition query n dict = true ->
  damerau_lev_distance query dict <= n.
Proof.
  intros query dict n Haccept.
  unfold automaton_accepts in Haccept.
  destruct (automaton_run_from_initial Transposition query n dict) as [final|] eqn:Hrun.
  2: { discriminate. }
  apply state_final_has_final_position in Haccept.
  destruct Haccept as [p [Hin Hfinal]].

  assert (Hreach : position_reachable_damerau query n dict p).
  { unfold automaton_run_from_initial in Hrun.
    set (init_closed := mkState (epsilon_closure
                         (Automaton.State.positions (initial_state Transposition (length query)))
                         n (length query)) Transposition (length query)) in *.
    assert (Hinit_reach : forall p0, In p0 (Automaton.State.positions init_closed) ->
                            position_reachable_damerau query n [] p0).
    { intros p0 Hin0.
      unfold init_closed in Hin0. simpl in Hin0.
      unfold initial_state in Hin0. simpl in Hin0.
      apply epsilon_closure_preserves_reachable_damerau with (positions := [initial_position]).
      - intros p1 Hin1.
        destruct Hin1 as [Heq | []]. subst.
        unfold initial_position, std_pos.
        apply reach_damerau_initial.
      - exact Hin0. }
    apply automaton_run_preserves_reachable_damerau_transposition with
        (query := query) (dict_prefix := []) (s := init_closed) (final := final).
    - unfold init_closed. simpl. reflexivity.
    - exact Hrun.
    - exact Hinit_reach.
    - exact Hin. }

  assert (Hspec : is_special p = false).
  { eapply transposition_final_not_special; eauto. }

  apply position_final_iff in Hfinal.
  assert (Hqlen_final : query_length final = length query).
  { unfold automaton_run_from_initial in Hrun.
    set (init_closed := mkState (epsilon_closure
                         (Automaton.State.positions (initial_state Transposition (length query)))
                         n (length query)) Transposition (length query)) in *.
    rewrite (automaton_run_preserves_query_length Transposition query n dict init_closed final Hrun).
    unfold init_closed. simpl. reflexivity. }

  assert (Hterm : term_index p = length query).
  { assert (Hbound : term_index p <= length query).
    { apply reachable_damerau_term_index_bound_query with (n := n) (dict_prefix := dict).
      exact Hreach. }
    rewrite Hqlen_final in Hfinal.
    lia. }

  apply Nat.le_trans with (num_errors p).
  - apply reachable_damerau_final_to_distance with (n := n); assumption.
  - apply reachable_damerau_implies_edit_distance with (query := query) (dp := dict).
    exact Hreach.
Qed.

(** Fallback: Standard Levenshtein bound from transposition acceptance.

    Note: lev_distance can be up to 2x damerau_lev_distance (each transposition
    costs 1 in Damerau-Levenshtein but 2 in standard Levenshtein). Thus if
    damerau_lev_distance <= n, then lev_distance <= 2n. *)
Corollary automaton_sound_transposition_lev : forall query dict n,
  automaton_accepts Transposition query n dict = true ->
  lev_distance query dict <= 2 * n.
Proof.
  intros query dict n Haccept.
  apply automaton_sound_transposition in Haccept.
  (* By soundness, damerau_lev_distance query dict <= n *)
  (* By lev_le_double_damerau, lev_distance <= 2 * damerau_lev_distance *)
  (* Therefore lev_distance <= 2 * n *)
  apply Nat.le_trans with (2 * damerau_lev_distance query dict).
  - apply lev_le_double_damerau.
  - lia.
Qed.

(** MergeAndSplit soundness using merge-split distance.

    The MergeAndSplit algorithm accepts strings within merge-split distance n.
    Merge (two query chars to one dict char) and split (one query char to
    two dict chars) can reduce the edit distance compared to standard operations.
*)
Theorem automaton_sound_merge_split : forall query dict n,
  automaton_accepts MergeAndSplit query n dict = true ->
  merge_split_distance query dict <= n.
Proof.
  intros query dict n Haccept.
  unfold automaton_accepts in Haccept.
  destruct (automaton_run_from_initial MergeAndSplit query n dict) as [final|] eqn:Hrun.
  2: { discriminate. }
  apply state_final_has_final_position in Haccept.
  destruct Haccept as [p [Hin Hfinal]].

  assert (Hreach : position_reachable_merge_split query n dict p).
  { unfold automaton_run_from_initial in Hrun.
    set (init_closed := mkState (epsilon_closure
                         (Automaton.State.positions (initial_state MergeAndSplit (length query)))
                         n (length query)) MergeAndSplit (length query)) in *.
    assert (Hinit_reach : forall p0, In p0 (Automaton.State.positions init_closed) ->
                            position_reachable_merge_split query n [] p0).
    { intros p0 Hin0.
      unfold init_closed in Hin0. simpl in Hin0.
      unfold initial_state in Hin0. simpl in Hin0.
      apply epsilon_closure_preserves_reachable_merge_split with (positions := [initial_position]).
      - intros p1 Hin1.
        destruct Hin1 as [Heq | []]. subst.
        unfold initial_position, std_pos.
        apply reach_ms_initial.
      - exact Hin0. }
    apply automaton_run_preserves_reachable_merge_split with
        (query := query) (dict_prefix := []) (s := init_closed) (final := final).
    - unfold init_closed. simpl. reflexivity.
    - exact Hrun.
    - exact Hinit_reach.
    - exact Hin. }

  apply position_final_iff in Hfinal.
  assert (Hqlen_final : query_length final = length query).
  { unfold automaton_run_from_initial in Hrun.
    set (init_closed := mkState (epsilon_closure
                         (Automaton.State.positions (initial_state MergeAndSplit (length query)))
                         n (length query)) MergeAndSplit (length query)) in *.
    rewrite (automaton_run_preserves_query_length MergeAndSplit query n dict init_closed final Hrun).
    unfold init_closed. simpl. reflexivity. }

  assert (Hterm : term_index p = length query).
  { assert (Hbound : term_index p <= length query).
    { apply reachable_merge_split_term_index_bound_query with (n := n) (dict_prefix := dict).
      exact Hreach. }
    rewrite Hqlen_final in Hfinal.
    lia. }

  apply Nat.le_trans with (num_errors p).
  - apply reachable_merge_split_final_to_distance with (n := n); assumption.
  - apply reachable_merge_split_implies_edit_distance with (query := query) (dp := dict).
    exact Hreach.
Qed.

(** Fallback: Standard Levenshtein bound from MergeAndSplit acceptance.

    Note: lev_distance can be up to 2x merge_split_distance (each merge or split
    costs 1 in MS but up to 2 in standard Levenshtein). Thus if
    merge_split_distance <= n, then lev_distance <= 2n. *)
Corollary automaton_sound_merge_split_lev : forall query dict n,
  automaton_accepts MergeAndSplit query n dict = true ->
  lev_distance query dict <= 2 * n.
Proof.
  intros query dict n Haccept.
  apply automaton_sound_merge_split in Haccept.
  (* Each merge-split operation is simulated by at most two standard
     Levenshtein operations. *)
  pose proof (lev_distance_ms_bound query dict) as Hbound.
  lia.
Qed.

(** Unified soundness theorem - each algorithm uses its appropriate distance

    - Standard: lev_distance
    - Transposition: damerau_lev_distance
    - MergeAndSplit: merge_split_distance
*)

(** For Standard algorithm: soundness with lev_distance *)
Corollary automaton_sound_standard_lev : forall query dict n,
  automaton_accepts Standard query n dict = true ->
  lev_distance query dict <= n.
Proof.
  apply automaton_sound_standard.
Qed.

(** * Corollaries *)

(** If automaton reports distance d, actual distance is <= d *)
Corollary automaton_distance_sound : forall query dict n d,
  automaton_distance Standard query n dict = Some d ->
  lev_distance query dict <= d.
Proof.
  intros query dict n d Hdist.
  unfold automaton_distance in Hdist.
  destruct (automaton_run_from_initial Standard query n dict) as [final|] eqn:Hrun.
  2: { discriminate. }
  (* The distance is the minimum error count among final positions *)
  (* Use accepting_distance_achieves to get a position p with num_errors p = d *)
  apply accepting_distance_achieves in Hdist.
  destruct Hdist as [p [Hin [Hfinal Herrors]]].

  (* Unfold automaton_run_from_initial to get automaton_run *)
  unfold automaton_run_from_initial in Hrun.
  set (init_closed := mkState (epsilon_closure
                     (Automaton.State.positions (initial_state Standard (length query)))
                     n (length query)) Standard (length query)) in *.

  (* Positions in init_closed are reachable with [] as dict_prefix *)
  assert (Hinit_reach : forall p0, In p0 (Automaton.State.positions init_closed) ->
                         position_reachable query n [] p0 /\ is_special p0 = false).
  { intros p0 Hin0. unfold init_closed in Hin0. simpl in Hin0.
    apply initial_closed_state_reachable. exact Hin0. }

  (* Step 1: Show p is reachable *)
  assert (Hreach : position_reachable query n dict p).
  { apply automaton_run_preserves_reachable_standard with
          (query := query) (dict_prefix := []) (s := init_closed) (final := final).
    - (* query_length init_closed = length query *)
      unfold init_closed. simpl. reflexivity.
    - exact Hrun.
    - exact Hinit_reach.
    - exact Hin. }

  (* Step 2: Show p is non-special *)
  assert (Hspec : is_special p = false).
  { apply (standard_run_positions_non_special query n dict init_closed final Hrun).
    - intros p1 Hin1. unfold init_closed in Hin1. simpl in Hin1.
      apply initial_closed_state_reachable in Hin1.
      destruct Hin1 as [_ Hsp]. exact Hsp.
    - exact Hin. }

  (* Step 3: Show term_index p = length query *)
  apply position_final_iff in Hfinal.
  assert (Hbound : term_index p <= length query).
  { apply reachable_term_index_bound_query with (n := n) (dict_prefix := dict). exact Hreach. }
  assert (Hqlen_final : query_length final = length query).
  { assert (Hqlen_init : query_length init_closed = length query).
    { unfold init_closed. simpl. reflexivity. }
    rewrite (automaton_run_preserves_query_length Standard query n dict init_closed final Hrun).
    exact Hqlen_init. }
  rewrite Hqlen_final in Hfinal.
  assert (Heq : term_index p = length query) by lia.

  (* Step 4: Apply reachable_final_to_distance *)
  rewrite <- Herrors.
  apply (reachable_final_to_distance query dict n p Hreach Hspec Heq).
Qed.

(** No false positives for Standard algorithm *)
Corollary no_false_positives_standard : forall query dict n,
  automaton_accepts Standard query n dict = true ->
  lev_distance query dict <= n.
Proof.
  apply automaton_sound_standard.
Qed.

(** No false positives for Transposition algorithm (using Damerau-Lev) *)
Corollary no_false_positives_transposition : forall query dict n,
  automaton_accepts Transposition query n dict = true ->
  damerau_lev_distance query dict <= n.
Proof.
  intros query dict n Haccept.
  apply automaton_sound_transposition.
  exact Haccept.
Qed.

(** No false positives for MergeAndSplit algorithm (using merge-split distance) *)
Corollary no_false_positives_merge_split : forall query dict n,
  automaton_accepts MergeAndSplit query n dict = true ->
  merge_split_distance query dict <= n.
Proof.
  intros query dict n Haccept.
  apply automaton_sound_merge_split.
  exact Haccept.
Qed.
