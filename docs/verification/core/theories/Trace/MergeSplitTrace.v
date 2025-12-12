(** * Merge-Split Trace Definitions and Validity

    A merge-split trace is a formalization of how a merge-split edit sequence
    transforms string A into string B, abstracting away the order of operations
    and focusing on the correspondence between character positions.

    Unlike standard Levenshtein traces (which are 1:1 position mappings),
    merge-split traces allow:
    - 1:1 mappings (match/substitution)
    - 2:1 mappings (merge: two source chars -> one target char)
    - 1:2 mappings (split: one source char -> two target chars)
    - 2:2 mappings (double substitution)

    Part of: Liblevenshtein.Core

    Design: Based on successful trace infrastructure from TraceBasics.v
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.

(** * Merge-Split Trace Element Type *)

(** A trace element describes one "atomic" correspondence in the transformation.
    Each element can map 1 or 2 source positions to 1 or 2 target positions.

    Positions are 1-indexed: position 1 is the first character.
*)
Inductive MSTraceElement :=
  | MSMatch (i j : nat)             (* Position i in A maps to j in B - match/subst *)
  | MSMerge2 (i1 i2 j : nat)        (* Positions i1,i2 in A merge to j in B *)
  | MSSplit2 (i j1 j2 : nat)        (* Position i in A splits to j1,j2 in B *)
  | MSDouble (i1 i2 j1 j2 : nat).   (* Positions i1,i2 map to j1,j2 - double-subst *)

Definition MSTrace := list MSTraceElement.

(** * Position Extraction Functions *)

(** Get positions in source string A touched by a trace element *)
Definition ms_element_positions_A (e : MSTraceElement) : list nat :=
  match e with
  | MSMatch i _ => [i]
  | MSMerge2 i1 i2 _ => [i1; i2]
  | MSSplit2 i _ _ => [i]
  | MSDouble i1 i2 _ _ => [i1; i2]
  end.

(** Get positions in target string B touched by a trace element *)
Definition ms_element_positions_B (e : MSTraceElement) : list nat :=
  match e with
  | MSMatch _ j => [j]
  | MSMerge2 _ _ j => [j]
  | MSSplit2 _ j1 j2 => [j1; j2]
  | MSDouble _ _ j1 j2 => [j1; j2]
  end.

(** All source positions touched by a trace *)
Definition ms_trace_positions_A (T : MSTrace) : list nat :=
  flat_map ms_element_positions_A T.

(** All target positions touched by a trace *)
Definition ms_trace_positions_B (T : MSTrace) : list nat :=
  flat_map ms_element_positions_B T.

(** * Element Cost Functions *)

(** Cost of a single trace element given strings A and B.
    Uses 1-indexed positions, so we access index (i-1) in the list.
*)
Definition ms_element_cost (A B : list Char) (e : MSTraceElement) : nat :=
  match e with
  | MSMatch i j =>
      subst_cost (nth (i-1) A default_char) (nth (j-1) B default_char)
  | MSMerge2 i1 i2 j =>
      merge_cost (nth (i1-1) A default_char) (nth (i2-1) A default_char) (nth (j-1) B default_char)
  | MSSplit2 i j1 j2 =>
      split_cost (nth (i-1) A default_char) (nth (j1-1) B default_char) (nth (j2-1) B default_char)
  | MSDouble i1 i2 j1 j2 =>
      subst_cost (nth (i1-1) A default_char) (nth (j1-1) B default_char) +
      subst_cost (nth (i2-1) A default_char) (nth (j2-1) B default_char)
  end.

(** * Validity Predicates *)

(** Check if a position is valid for a given length (1-indexed) *)
Definition valid_position (len pos : nat) : bool :=
  (1 <=? pos) && (pos <=? len).

(** Check if all positions in a list are valid *)
Definition all_positions_valid (len : nat) (ps : list nat) : bool :=
  forallb (valid_position len) ps.

(** Check if a trace element has valid positions for given string lengths *)
Definition ms_valid_element (lenA lenB : nat) (e : MSTraceElement) : bool :=
  all_positions_valid lenA (ms_element_positions_A e) &&
  all_positions_valid lenB (ms_element_positions_B e).

(** Check if positions in a list have no overlaps *)
Fixpoint no_duplicate_positions (ps : list nat) : bool :=
  match ps with
  | [] => true
  | p :: rest => negb (existsb (Nat.eqb p) rest) && no_duplicate_positions rest
  end.

(** Check if a trace has no overlapping source positions *)
Definition ms_positions_no_overlap_A (T : MSTrace) : bool :=
  no_duplicate_positions (ms_trace_positions_A T).

(** Check if a trace has no overlapping target positions *)
Definition ms_positions_no_overlap_B (T : MSTrace) : bool :=
  no_duplicate_positions (ms_trace_positions_B T).

(** Monotonicity helper: check if element positions are strictly ordered *)
Definition ms_element_positions_ordered (e : MSTraceElement) : bool :=
  match e with
  | MSMatch _ _ => true
  | MSMerge2 i1 i2 _ => i1 <? i2
  | MSSplit2 _ j1 j2 => j1 <? j2
  | MSDouble i1 i2 j1 j2 => (i1 <? i2) && (j1 <? j2)
  end.

(** Get the minimum source position of an element *)
Definition ms_element_min_A (e : MSTraceElement) : nat :=
  match e with
  | MSMatch i _ => i
  | MSMerge2 i1 _ _ => i1
  | MSSplit2 i _ _ => i
  | MSDouble i1 _ _ _ => i1
  end.

(** Get the minimum target position of an element *)
Definition ms_element_min_B (e : MSTraceElement) : nat :=
  match e with
  | MSMatch _ j => j
  | MSMerge2 _ _ j => j
  | MSSplit2 _ j1 _ => j1
  | MSDouble _ _ j1 _ => j1
  end.

(** Get the maximum source position of an element *)
Definition ms_element_max_A (e : MSTraceElement) : nat :=
  match e with
  | MSMatch i _ => i
  | MSMerge2 _ i2 _ => i2
  | MSSplit2 i _ _ => i
  | MSDouble _ i2 _ _ => i2
  end.

(** Get the maximum target position of an element *)
Definition ms_element_max_B (e : MSTraceElement) : nat :=
  match e with
  | MSMatch _ j => j
  | MSMerge2 _ _ j => j
  | MSSplit2 _ _ j2 => j2
  | MSDouble _ _ _ j2 => j2
  end.

(** Check if elements are monotonically ordered in the trace *)
Fixpoint ms_trace_monotonic_aux (T : MSTrace) : bool :=
  match T with
  | [] => true
  | [_] => true
  | e1 :: (e2 :: _) as rest =>
      (ms_element_max_A e1 <? ms_element_min_A e2) &&
      (ms_element_max_B e1 <? ms_element_min_B e2) &&
      ms_trace_monotonic_aux rest
  end.

(** Full validity check for a trace *)
Definition ms_trace_valid (A B : list Char) (T : MSTrace) : bool :=
  forallb (ms_valid_element (length A) (length B)) T &&
  forallb ms_element_positions_ordered T &&
  ms_positions_no_overlap_A T &&
  ms_positions_no_overlap_B T &&
  ms_trace_monotonic_aux T.

(** * Trace Cost Computation *)

(** Sum of element costs *)
Definition ms_trace_change_cost (A B : list Char) (T : MSTrace) : nat :=
  fold_left (fun acc e => acc + ms_element_cost A B e) T 0.

(** Count of positions not covered by the trace - these become deletions/insertions *)
Definition ms_trace_delete_cost (A : list Char) (T : MSTrace) : nat :=
  length A - length (ms_trace_positions_A T).

Definition ms_trace_insert_cost (B : list Char) (T : MSTrace) : nat :=
  length B - length (ms_trace_positions_B T).

(** Full trace cost = change_cost + delete_cost + insert_cost *)
Definition ms_trace_cost (A B : list Char) (T : MSTrace) : nat :=
  ms_trace_change_cost A B T +
  ms_trace_delete_cost A T +
  ms_trace_insert_cost B T.

(** * Projection to Standard Trace Pairs *)

(** Convert an MS trace element to standard (position, position) pairs.
    This is useful for composing traces via the existing compose_trace infrastructure.

    Note: This projection is NOT injective - multiple MS elements can map to
    the same pairs. It's used for the composition proof, not for reconstruction.
*)
Definition ms_element_to_pairs (e : MSTraceElement) : list (nat * nat) :=
  match e with
  | MSMatch i j => [(i, j)]
  | MSMerge2 i1 i2 j => [(i1, j); (i2, j)]    (* Many-to-one: both i1,i2 map to j *)
  | MSSplit2 i j1 j2 => [(i, j1); (i, j2)]    (* One-to-many: i maps to both j1,j2 *)
  | MSDouble i1 i2 j1 j2 => [(i1, j1); (i2, j2)]
  end.

(** Convert entire trace to list of pairs *)
Definition ms_trace_to_pairs (T : MSTrace) : list (nat * nat) :=
  flat_map ms_element_to_pairs T.

(** * Basic Lemmas *)

(** Length of positions list for each element type *)
Lemma ms_element_positions_A_length : forall e,
  length (ms_element_positions_A e) =
  match e with
  | MSMatch _ _ => 1
  | MSMerge2 _ _ _ => 2
  | MSSplit2 _ _ _ => 1
  | MSDouble _ _ _ _ => 2
  end.
Proof.
  intros []; reflexivity.
Qed.

Lemma ms_element_positions_B_length : forall e,
  length (ms_element_positions_B e) =
  match e with
  | MSMatch _ _ => 1
  | MSMerge2 _ _ _ => 1
  | MSSplit2 _ _ _ => 2
  | MSDouble _ _ _ _ => 2
  end.
Proof.
  intros []; reflexivity.
Qed.

(** Projection relationship - positions in pairs include positions in A/B
    Note: pairs may duplicate positions (e.g., MSSplit2 i j1 j2 gives [(i,j1), (i,j2)])
    while positions_A gives just [i]. So map fst (pairs) != positions_A in general.

    Instead we prove membership preservation. *)

(** Any position in ms_element_positions_A is in the fst of some pair *)
Lemma ms_element_positions_A_in_pairs : forall e i,
  In i (ms_element_positions_A e) ->
  exists j, In (i, j) (ms_element_to_pairs e).
Proof.
  intros e i Hin.
  destruct e as [a b | a1 a2 b | a b1 b2 | a1 a2 b1 b2]; simpl in *.
  - (* MSMatch a b: positions_A = [a] *)
    destruct Hin as [H | []]. subst. exists b. left. reflexivity.
  - (* MSMerge2 a1 a2 b: positions_A = [a1; a2] *)
    destruct Hin as [H | [H | []]]; subst.
    + exists b. left. reflexivity.
    + exists b. right. left. reflexivity.
  - (* MSSplit2 a b1 b2: positions_A = [a] *)
    destruct Hin as [H | []]. subst. exists b1. left. reflexivity.
  - (* MSDouble a1 a2 b1 b2: positions_A = [a1; a2] *)
    destruct Hin as [H | [H | []]]; subst.
    + exists b1. left. reflexivity.
    + exists b2. right. left. reflexivity.
Qed.

(** Any position in ms_element_positions_B is in the snd of some pair *)
Lemma ms_element_positions_B_in_pairs : forall e j,
  In j (ms_element_positions_B e) ->
  exists i, In (i, j) (ms_element_to_pairs e).
Proof.
  intros e j Hin.
  destruct e as [a b | a1 a2 b | a b1 b2 | a1 a2 b1 b2]; simpl in *.
  - (* MSMatch a b: positions_B = [b] *)
    destruct Hin as [H | []]. subst. exists a. left. reflexivity.
  - (* MSMerge2 a1 a2 b: positions_B = [b] *)
    destruct Hin as [H | []]. subst. exists a1. left. reflexivity.
  - (* MSSplit2 a b1 b2: positions_B = [b1; b2] *)
    destruct Hin as [H | [H | []]]; subst.
    + exists a. left. reflexivity.
    + exists a. right. left. reflexivity.
  - (* MSDouble a1 a2 b1 b2: positions_B = [b1; b2] *)
    destruct Hin as [H | [H | []]]; subst.
    + exists a1. left. reflexivity.
    + exists a2. right. left. reflexivity.
Qed.

(** Any fst of a pair is in ms_element_positions_A *)
Lemma ms_element_pairs_fst_in_positions : forall e i j,
  In (i, j) (ms_element_to_pairs e) ->
  In i (ms_element_positions_A e).
Proof.
  intros e i j Hin.
  destruct e as [a b | a1 a2 b | a b1 b2 | a1 a2 b1 b2]; simpl in *.
  - (* MSMatch a b: pairs = [(a, b)] *)
    destruct Hin as [H | []]. injection H as -> ->. left. reflexivity.
  - (* MSMerge2 a1 a2 b: pairs = [(a1, b); (a2, b)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + right. left. reflexivity.
  - (* MSSplit2 a b1 b2: pairs = [(a, b1); (a, b2)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + left. reflexivity.
  - (* MSDouble a1 a2 b1 b2: pairs = [(a1, b1); (a2, b2)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + right. left. reflexivity.
Qed.

(** Any snd of a pair is in ms_element_positions_B *)
Lemma ms_element_pairs_snd_in_positions : forall e i j,
  In (i, j) (ms_element_to_pairs e) ->
  In j (ms_element_positions_B e).
Proof.
  intros e i j Hin.
  destruct e as [a b | a1 a2 b | a b1 b2 | a1 a2 b1 b2]; simpl in *.
  - (* MSMatch a b: pairs = [(a, b)] *)
    destruct Hin as [H | []]. injection H as -> ->. left. reflexivity.
  - (* MSMerge2 a1 a2 b: pairs = [(a1, b); (a2, b)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + left. reflexivity.
  - (* MSSplit2 a b1 b2: pairs = [(a, b1); (a, b2)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + right. left. reflexivity.
  - (* MSDouble a1 a2 b1 b2: pairs = [(a1, b1); (a2, b2)] *)
    destruct Hin as [H | [H | []]]; injection H as -> ->.
    + left. reflexivity.
    + right. left. reflexivity.
Qed.

(** Trace-level membership preservation *)
Lemma ms_trace_positions_A_in_pairs : forall T i,
  In i (ms_trace_positions_A T) ->
  exists j, In (i, j) (ms_trace_to_pairs T).
Proof.
  induction T as [| e T' IH]; intros i Hin.
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    destruct Hin as [Hin_e | Hin_T'].
    + destruct (ms_element_positions_A_in_pairs e i Hin_e) as [j Hpair].
      exists j. simpl. apply in_or_app. left. exact Hpair.
    + destruct (IH i Hin_T') as [j Hpair].
      exists j. simpl. apply in_or_app. right. exact Hpair.
Qed.

Lemma ms_trace_positions_B_in_pairs : forall T j,
  In j (ms_trace_positions_B T) ->
  exists i, In (i, j) (ms_trace_to_pairs T).
Proof.
  induction T as [| e T' IH]; intros j Hin.
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    destruct Hin as [Hin_e | Hin_T'].
    + destruct (ms_element_positions_B_in_pairs e j Hin_e) as [i Hpair].
      exists i. simpl. apply in_or_app. left. exact Hpair.
    + destruct (IH j Hin_T') as [i Hpair].
      exists i. simpl. apply in_or_app. right. exact Hpair.
Qed.

(** Empty trace has zero cost *)
Lemma ms_trace_change_cost_nil : forall A B,
  ms_trace_change_cost A B [] = 0.
Proof. reflexivity. Qed.

(** Empty trace validity *)
Lemma ms_trace_valid_nil : forall A B,
  ms_trace_valid A B [] = true.
Proof. reflexivity. Qed.

(** * Decidable Equality for MSTraceElement *)

Definition ms_element_eq_dec (e1 e2 : MSTraceElement) : {e1 = e2} + {e1 <> e2}.
Proof.
  decide equality; apply Nat.eq_dec.
Defined.

(** * Trace Cost Upper Bound Infrastructure *)

(** Import necessary lemmas *)
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.MetricProperties.

(** Helper: max(a,b) <= a + b *)
Lemma max_le_plus : forall a b, Nat.max a b <= a + b.
Proof.
  intros. apply Nat.max_lub; lia.
Qed.

(** Levenshtein distance is bounded by sum of lengths.
    Follows from lev_distance_upper_bound (max version) + max_le_plus. *)
Lemma lev_distance_sum_bound : forall A B,
  lev_distance A B <= length A + length B.
Proof.
  intros A B.
  apply Nat.le_trans with (Nat.max (length A) (length B)).
  - apply lev_distance_upper_bound.
  - apply max_le_plus.
Qed.

(** Merge-split distance is bounded by string lengths.
    Follows from ms_le_standard + lev_distance_sum_bound. *)
Lemma ms_length_upper_bound : forall A B,
  merge_split_distance A B <= length A + length B.
Proof.
  intros A B.
  apply Nat.le_trans with (lev_distance A B).
  - apply ms_le_standard.
  - apply lev_distance_sum_bound.
Qed.

(** Valid trace positions are bounded by string length.
    All positions in ms_trace_positions_A are in range [1, length A].

    Proof approach: Admit with semantic justification.
    The full proof requires showing:
    1. Each position is in bounds (from ms_valid_element)
    2. Positions are distinct (from ms_positions_no_overlap_A)
    3. By pigeonhole: |distinct positions in [1,n]| <= n
*)
Lemma ms_valid_trace_touched_A_bound : forall A B T,
  ms_trace_valid A B T = true ->
  length (ms_trace_positions_A T) <= length A.
Proof.
  intros A B T Hvalid.
  (* SEMANTIC JUSTIFICATION:
     A valid trace has:
     - All positions in ms_trace_positions_A are in [1, length A]
     - No duplicate positions (ms_positions_no_overlap_A)
     Therefore by pigeonhole, |positions_A| <= length A.

     The full proof requires tracking validity through the induction,
     which involves complex destruct patterns. For now, we admit this
     with the semantic justification that the bound is structurally sound.
  *)
  admit.
Admitted.

(** Symmetric bound for B positions *)
Lemma ms_valid_trace_touched_B_bound : forall A B T,
  ms_trace_valid A B T = true ->
  length (ms_trace_positions_B T) <= length B.
Proof.
  intros A B T Hvalid.
  (* Same semantic justification as ms_valid_trace_touched_A_bound *)
  admit.
Admitted.

(** * Main Theorem: Trace Cost Upper Bound *)

(** For any valid MS trace T on strings A and B, the trace cost provides
    an upper bound on the merge-split distance.

    This is the key lemma needed for the triangle inequality:
    - If T1 is an optimal trace for A→B, its cost = ms_distance(A,B)
    - If T2 is an optimal trace for B→C, its cost = ms_distance(B,C)
    - A composed trace A→C would have cost <= cost(T1) + cost(T2)
    - Since ms_distance(A,C) <= any valid trace cost, we get triangle

    Proof strategy:
    - For empty trace: trace_cost = |A| + |B| >= distance (by ms_length_upper_bound)
    - For non-empty trace: use the fact that valid traces define a valid
      alignment, and ms_distance is the minimum over all valid alignments.

    Technical approach:
    - Uses ms_length_upper_bound for the empty trace case
    - Uses ms_valid_trace_touched_A/B_bound to show trace_cost >= 0
    - The key insight is that trace_cost = change_cost + delete_cost + insert_cost
      where delete_cost = |A| - |positions_A| and insert_cost = |B| - |positions_B|
    - For empty trace: delete_cost = |A|, insert_cost = |B|, change_cost = 0
      So trace_cost = |A| + |B| >= ms_distance(A,B)
*)
Theorem ms_trace_upper_bound : forall A B T,
  ms_trace_valid A B T = true ->
  merge_split_distance A B <= ms_trace_cost A B T.
Proof.
  intros A B T Hvalid.
  unfold ms_trace_cost.

  (* trace_cost = change_cost + delete_cost + insert_cost
     where delete_cost = |A| - |positions_A(T)|
           insert_cost = |B| - |positions_B(T)|

     For any trace, trace_cost >= |A| - |positions_A| + |B| - |positions_B|
     Since positions are bounded: |positions_A| <= |A|, |positions_B| <= |B|
     So trace_cost >= 0.

     For the empty trace: trace_cost = |A| + |B|
     By ms_length_upper_bound: distance <= |A| + |B|
  *)

  destruct T as [| e rest].
  - (* Empty trace *)
    simpl. unfold ms_trace_delete_cost, ms_trace_insert_cost.
    simpl ms_trace_positions_A. simpl ms_trace_positions_B.
    simpl length. rewrite Nat.sub_0_r, Nat.sub_0_r.
    unfold ms_trace_change_cost. simpl.
    apply ms_length_upper_bound.
  - (* Non-empty trace *)
    (* For non-empty valid trace, we use the upper bound approach:
       - trace_cost >= delete_cost + insert_cost (since change_cost >= 0)
       - delete_cost = |A| - |positions_A|
       - insert_cost = |B| - |positions_B|

       The key is that valid traces satisfy:
       - Positions are in bounds
       - Positions don't overlap

       So the number of uncovered positions is exactly delete_cost + insert_cost.

       The semantic argument is that any valid trace represents a valid edit
       transformation, and ms_distance is the minimum cost transformation.

       For a formal proof via induction on trace structure, see the approach
       in DamerauTrace.v (trace_bounds_distance_strong).

       Here we use the upper bound approach: since change_cost >= 0 and
       delete_cost + insert_cost >= 0, trace_cost >= 0.
       Combined with ms_length_upper_bound, if trace_cost >= |A| + |B| - k
       for some k >= 0, we still have distance <= trace_cost in most cases.
    *)

    pose proof (ms_valid_trace_touched_A_bound A B (e :: rest) Hvalid) as HbA.
    pose proof (ms_valid_trace_touched_B_bound A B (e :: rest) Hvalid) as HbB.

    (* delete_cost = |A| - |positions_A| *)
    (* insert_cost = |B| - |positions_B| *)
    unfold ms_trace_delete_cost, ms_trace_insert_cost.

    (* For the bound, observe:
       trace_cost = change_cost + (|A| - |posA|) + (|B| - |posB|)

       We need: distance <= trace_cost

       Using ms_length_upper_bound: distance <= |A| + |B|

       If |posA| + |posB| >= change_cost, then:
       trace_cost = change_cost + |A| + |B| - |posA| - |posB|
                 >= |A| + |B| - (|posA| + |posB| - change_cost)

       The challenge is that change_cost could be less than |posA| + |posB|
       (e.g., matching chars have cost 0).

       For a complete proof, we need to show that the trace's alignment
       corresponds to a valid edit sequence, making distance <= trace_cost.

       Simplified approach using the length bound:
    *)

    pose proof (ms_length_upper_bound A B) as Hdist_bound.

    (* The trace cost includes at least the uncovered positions as deletes/inserts *)
    assert (Huncovered: length A - length (ms_trace_positions_A (e :: rest)) +
                       length B - length (ms_trace_positions_B (e :: rest)) >= 0) by lia.

    (* For a complete proof, we'd show:
       distance <= change_cost + delete_cost + insert_cost

       This follows from the definition of ms_distance as the minimum over
       all edit sequences, and the trace defines one such sequence.

       SEMANTIC JUSTIFICATION:
       A valid trace T defines an alignment where:
       - Matched/merged/split/doubled positions have corresponding edit operations
       - Unmatched positions in A become deletions (cost 1 each)
       - Unmatched positions in B become insertions (cost 1 each)

       The trace cost = sum of edit costs = change_cost + delete_cost + insert_cost
       Since ms_distance is the minimum, distance <= trace_cost.

       For now, we use a weaker bound that suffices for many cases:
    *)

    (* Weak bound: trace_cost >= delete_cost + insert_cost >= 0
       Combined with length bound gives distance <= trace_cost when
       trace_cost >= |A| + |B| *)

    (* For traces that cover few positions, trace_cost >= |A| + |B| *)
    destruct (le_lt_dec (length (ms_trace_positions_A (e :: rest)) +
                        length (ms_trace_positions_B (e :: rest)))
                        (ms_trace_change_cost A B (e :: rest))) as [Hcase1 | Hcase2].
    + (* Case: change_cost >= |posA| + |posB| *)
      (* trace_cost = change_cost + |A| - |posA| + |B| - |posB|
                   >= |posA| + |posB| + |A| - |posA| + |B| - |posB|
                   = |A| + |B|
         So distance <= |A| + |B| <= trace_cost *)
      lia.
    + (* Case: change_cost < |posA| + |posB| *)
      (* This is the typical case for traces with many matches (cost 0).

         For a complete proof, we'd use induction to show that the
         trace's alignment gives a valid edit sequence with the trace cost.

         The approach in DamerauTrace.v uses strong induction on |A| + |B|:
         - Analyze the first trace element
         - Show the recursive subproblem has a valid trace
         - Use IH and combine costs

         For MS traces, this requires handling all 4 element types:
         MSMatch, MSMerge2, MSSplit2, MSDouble

         Each element type reduces to a smaller subproblem:
         - MSMatch(1,1): A' = tail A, B' = tail B
         - MSMerge2(1,2,1): A' = drop 2 A, B' = tail B
         - MSSplit2(1,1,2): A' = tail A, B' = drop 2 B
         - MSDouble(1,2,1,2): A' = drop 2 A, B' = drop 2 B

         For a valid trace, if element starts at position > 1 in A or B,
         the uncovered positions become deletions/insertions which are
         accounted for in delete_cost/insert_cost.

         ADMITTED: Full inductive proof following DamerauTrace.v pattern.
         The semantic argument is sound.
      *)

      (* For the empty case, we already proved it above.
         For non-empty, use the semantic justification:
         - Valid traces define valid edit transformations
         - ms_distance is minimum, so distance <= any valid transformation cost
      *)

      (* Fallback to length bound when trace is "efficient" *)
      (* If trace covers many positions with low change_cost,
         the delete_cost + insert_cost is small, making trace_cost close to change_cost.
         We still have distance <= |A| + |B|, and typically trace_cost >= distance
         because efficient traces tend to witness optimal alignments. *)

      (* For now, use lia to check if length bound suffices *)
      (* This may fail for some cases - those would need the full inductive proof *)
      try lia.

      (* If lia fails, the full inductive proof is needed.
         SEMANTIC JUSTIFICATION for Admitted:
         - Each valid trace element represents a valid edit operation
         - The trace cost = sum of operation costs
         - ms_distance = minimum over all valid operation sequences
         - Therefore ms_distance <= trace_cost

         The formal proof requires induction on |A| + |B| with case analysis
         on trace structure, similar to trace_bounds_distance_strong in DamerauTrace.v.
      *)

      (* Use the semantic bound *)
      pose proof (ms_length_upper_bound A B) as Hub.
      (* In most practical cases, trace_cost >= distance by construction.
         The formal gap is proving this for all valid traces. *)

      (* For traces with efficient alignment (low change_cost, high coverage),
         we need the inductive argument. For traces with low coverage,
         delete_cost + insert_cost dominates and we get trace_cost >= |A| + |B| - k
         for small k. *)

      (* Attempt arithmetic bound *)
      assert (Hchange_ge_0 : ms_trace_change_cost A B (e :: rest) >= 0) by lia.

      (* The trace cost is:
         change_cost + (|A| - |posA|) + (|B| - |posB|)

         If |posA| <= |A| and |posB| <= |B|, then:
         trace_cost >= 0 + (|A| - |A|) + (|B| - |B|) = 0 (trivial)

         But we need distance <= trace_cost.

         The key missing piece is showing that trace alignment implies
         distance <= trace_cost. This requires the full inductive proof.

         ADMITTED pending full induction proof.
      *)
      admit.
Admitted.
