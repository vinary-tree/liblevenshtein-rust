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

From Stdlib Require Import String List Arith Ascii Bool Nat Lia ZArith Zify.
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

(** Check if multi-position elements have consecutive positions.
    This is required for the trace-to-sequence conversion to work correctly.
    For MSMerge2 i1 i2 j, we need i2 = i1 + 1.
    For MSSplit2 i j1 j2, we need j2 = j1 + 1.
    For MSDouble i1 i2 j1 j2, we need i2 = i1 + 1 and j2 = j1 + 1.
*)
Definition ms_element_positions_consecutive (e : MSTraceElement) : bool :=
  match e with
  | MSMatch _ _ => true
  | MSMerge2 i1 i2 _ => i2 =? i1 + 1
  | MSSplit2 _ j1 j2 => j2 =? j1 + 1
  | MSDouble i1 i2 j1 j2 => (i2 =? i1 + 1) && (j2 =? j1 + 1)
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

(** Check if all trace elements have consecutive positions.
    This is required for the trace-to-sequence conversion in trace_to_seq_aux.
    For typical merge-split traces, positions are consecutive by construction.
*)
Definition ms_trace_positions_consecutive (T : MSTrace) : bool :=
  forallb ms_element_positions_consecutive T.

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

(** Helper: fold_left with addition can be decomposed *)
Lemma fold_left_add_init : forall (A' B' : list Char) T init,
  fold_left (fun acc e => acc + ms_element_cost A' B' e) T init =
  init + fold_left (fun acc e => acc + ms_element_cost A' B' e) T 0.
Proof.
  intros A' B' T. revert A' B'.
  induction T as [| e rest IH]; intros A' B' init.
  - simpl. lia.
  - simpl. rewrite IH with (init := init + ms_element_cost A' B' e).
    rewrite IH with (init := 0 + ms_element_cost A' B' e).
    lia.
Qed.

(** Cons case for ms_trace_change_cost *)
Lemma ms_trace_change_cost_cons : forall A B e rest,
  ms_trace_change_cost A B (e :: rest) = ms_element_cost A B e + ms_trace_change_cost A B rest.
Proof.
  intros A B e rest.
  unfold ms_trace_change_cost.
  simpl.
  rewrite fold_left_add_init.
  simpl. reflexivity.
Qed.

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
From Liblevenshtein.Core Require Import LowerBound.PigeonholeBounds.

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

(** ** Helper: Convert boolean no_duplicate_positions to Prop NoDup *)

(** Helper to show existsb relates to In *)
Lemma existsb_In_nat : forall p l,
  existsb (Nat.eqb p) l = true <-> In p l.
Proof.
  intros p l. split.
  - induction l as [| x rest IH]; intros H.
    + simpl in H. discriminate.
    + simpl in H. apply Bool.orb_prop in H. destruct H as [H | H].
      * apply Nat.eqb_eq in H. subst. left. reflexivity.
      * right. apply IH. exact H.
  - induction l as [| x rest IH]; intros H.
    + destruct H.
    + simpl. apply Bool.orb_true_intro.
      destruct H as [H | H].
      * left. subst. apply Nat.eqb_refl.
      * right. apply IH. exact H.
Qed.

(** Convert no_duplicate_positions (boolean) to NoDup (Prop) *)
Lemma no_duplicate_positions_NoDup : forall ps,
  no_duplicate_positions ps = true -> NoDup ps.
Proof.
  induction ps as [| p rest IH]; intros H.
  - apply NoDup_nil.
  - simpl in H. apply andb_prop in H as [Hnot_in Hrest].
    apply NoDup_cons.
    + (* Show ~In p rest via negb (existsb ...) *)
      intro Hin.
      apply existsb_In_nat in Hin.
      rewrite Hin in Hnot_in.
      simpl in Hnot_in. discriminate.
    + apply IH. exact Hrest.
Qed.

(** Convert NoDup (Prop) to no_duplicate_positions (boolean) *)
Lemma NoDup_no_duplicate_positions : forall ps,
  NoDup ps -> no_duplicate_positions ps = true.
Proof.
  induction ps as [| p rest IH]; intros H.
  - reflexivity.
  - simpl. apply andb_true_intro. split.
    + (* negb (existsb (Nat.eqb p) rest) = true *)
      inversion H; subst.
      apply negb_true_iff.
      destruct (existsb (Nat.eqb p) rest) eqn:E; [|reflexivity].
      apply existsb_In_nat in E.
      contradiction.
    + (* no_duplicate_positions rest = true *)
      inversion H; subst.
      apply IH. exact H3.
Qed.

(** ** Helper: Positions from valid elements are in range *)

(** Helper: valid_position implies range bounds *)
Lemma valid_position_bounds : forall len pos,
  valid_position len pos = true -> 1 <= pos /\ pos <= len.
Proof.
  intros len pos H.
  unfold valid_position in H.
  apply andb_prop in H as [H1 H2].
  apply Nat.leb_le in H1.
  apply Nat.leb_le in H2.
  lia.
Qed.

(** Helper: all_positions_valid implies each position is in range *)
Lemma all_positions_valid_In : forall len ps p,
  all_positions_valid len ps = true ->
  In p ps ->
  1 <= p /\ p <= len.
Proof.
  intros len ps p Hvalid Hin.
  unfold all_positions_valid in Hvalid.
  rewrite forallb_forall in Hvalid.
  specialize (Hvalid p Hin).
  apply valid_position_bounds. exact Hvalid.
Qed.

(** Helper: ms_valid_element implies A-positions are in range *)
Lemma ms_valid_element_A_in_range : forall lenA lenB e p,
  ms_valid_element lenA lenB e = true ->
  In p (ms_element_positions_A e) ->
  1 <= p /\ p <= lenA.
Proof.
  intros lenA lenB e p Hvalid Hin.
  unfold ms_valid_element in Hvalid.
  apply andb_prop in Hvalid as [HvalidA HvalidB].
  apply all_positions_valid_In with (ms_element_positions_A e); assumption.
Qed.

(** Helper: ms_valid_element implies max_A is in range *)
Lemma ms_valid_element_max_A_bound : forall lenA lenB e,
  ms_valid_element lenA lenB e = true ->
  ms_element_max_A e <= lenA.
Proof.
  intros lenA lenB e Hvalid.
  unfold ms_valid_element in Hvalid.
  apply andb_prop in Hvalid as [HvalidA HvalidB].
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - unfold all_positions_valid in HvalidA. simpl in HvalidA.
    apply andb_prop in HvalidA as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidA. simpl in HvalidA.
    apply andb_prop in HvalidA as [_ HvalidA'].
    apply andb_prop in HvalidA' as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidA. simpl in HvalidA.
    apply andb_prop in HvalidA as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidA. simpl in HvalidA.
    apply andb_prop in HvalidA as [_ HvalidA'].
    apply andb_prop in HvalidA' as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
Qed.

(** Helper: ms_valid_element implies max_B is in range *)
Lemma ms_valid_element_max_B_bound : forall lenA lenB e,
  ms_valid_element lenA lenB e = true ->
  ms_element_max_B e <= lenB.
Proof.
  intros lenA lenB e Hvalid.
  unfold ms_valid_element in Hvalid.
  apply andb_prop in Hvalid as [HvalidA HvalidB].
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - unfold all_positions_valid in HvalidB. simpl in HvalidB.
    apply andb_prop in HvalidB as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidB. simpl in HvalidB.
    apply andb_prop in HvalidB as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidB. simpl in HvalidB.
    apply andb_prop in HvalidB as [_ HvalidB'].
    apply andb_prop in HvalidB' as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
  - unfold all_positions_valid in HvalidB. simpl in HvalidB.
    apply andb_prop in HvalidB as [_ HvalidB'].
    apply andb_prop in HvalidB' as [Hpos _].
    apply valid_position_bounds in Hpos. lia.
Qed.

(** Helper: ms_valid_element implies B-positions are in range *)
Lemma ms_valid_element_B_in_range : forall lenA lenB e p,
  ms_valid_element lenA lenB e = true ->
  In p (ms_element_positions_B e) ->
  1 <= p /\ p <= lenB.
Proof.
  intros lenA lenB e p Hvalid Hin.
  unfold ms_valid_element in Hvalid.
  apply andb_prop in Hvalid as [HvalidA HvalidB].
  apply all_positions_valid_In with (ms_element_positions_B e); assumption.
Qed.

(** All positions in ms_trace_positions_A are in range [1, length strA] *)
Lemma ms_trace_positions_A_in_range : forall (strA strB : list Char) T,
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  forall p, In p (ms_trace_positions_A T) -> 1 <= p /\ p <= length strA.
Proof.
  intros strA strB T Hvalid p Hin.
  induction T as [| e rest IH].
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    simpl in Hvalid. apply andb_prop in Hvalid as [He Hrest].
    destruct Hin as [Hin_e | Hin_rest].
    + apply ms_valid_element_A_in_range with (length strB) e; assumption.
    + apply IH; assumption.
Qed.

(** All positions in ms_trace_positions_B are in range [1, length strB] *)
Lemma ms_trace_positions_B_in_range : forall (strA strB : list Char) T,
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  forall p, In p (ms_trace_positions_B T) -> 1 <= p /\ p <= length strB.
Proof.
  intros strA strB T Hvalid p Hin.
  induction T as [| e rest IH].
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    simpl in Hvalid. apply andb_prop in Hvalid as [He Hrest].
    destruct Hin as [Hin_e | Hin_rest].
    + apply ms_valid_element_B_in_range with (length strA) e; assumption.
    + apply IH; assumption.
Qed.

(** For any position in ms_trace_positions_A, there is an element containing it *)
Lemma ms_position_from_min : forall (strA strB : list Char) T p,
  In p (ms_trace_positions_A T) ->
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  forallb ms_element_positions_ordered T = true ->
  exists e, In e T /\ In p (ms_element_positions_A e).
Proof.
  intros strA strB T p Hin Hvalid Hord.
  induction T as [| e rest IH].
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    simpl in Hvalid. apply andb_prop in Hvalid as [He Hrest_valid].
    simpl in Hord. apply andb_prop in Hord as [He_ord Hrest_ord].
    destruct Hin as [Hin_e | Hin_rest].
    + exists e. split; [left; reflexivity | exact Hin_e].
    + specialize (IH Hin_rest Hrest_valid Hrest_ord) as [e' [Hin' Hpos']].
      exists e'. split; [right; exact Hin' | exact Hpos'].
Qed.

(** For any position in ms_trace_positions_B, there is an element containing it *)
Lemma ms_position_from_min_B : forall (strA strB : list Char) T p,
  In p (ms_trace_positions_B T) ->
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  forallb ms_element_positions_ordered T = true ->
  exists e, In e T /\ In p (ms_element_positions_B e).
Proof.
  intros strA strB T p Hin Hvalid Hord.
  induction T as [| e rest IH].
  - simpl in Hin. contradiction.
  - simpl in Hin. apply in_app_or in Hin.
    simpl in Hvalid. apply andb_prop in Hvalid as [He Hrest_valid].
    simpl in Hord. apply andb_prop in Hord as [He_ord Hrest_ord].
    destruct Hin as [Hin_e | Hin_rest].
    + exists e. split; [left; reflexivity | exact Hin_e].
    + specialize (IH Hin_rest Hrest_valid Hrest_ord) as [e' [Hin' Hpos']].
      exists e'. split; [right; exact Hin' | exact Hpos'].
Qed.

(** ** Main Bounds Lemmas *)

(** Valid trace positions are bounded by string length.
    All positions in ms_trace_positions_A are in range [1, length A].
*)
Lemma ms_valid_trace_touched_A_bound : forall A B T,
  ms_trace_valid A B T = true ->
  length (ms_trace_positions_A T) <= length A.
Proof.
  intros A B T Hvalid.
  (* Extract components from ms_trace_valid *)
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' Hmonotonic].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' Hno_overlap_A].
  apply andb_prop in Hvalid''' as [Helems_valid Hordered].
  (* Use pigeonhole principle *)
  destruct A as [| a A'].
  - (* A = [] *)
    (* If A is empty, lenA = 0 *)
    (* Valid element requires positions in [1, 0] which is impossible if T non-empty *)
    destruct T as [| e rest].
    + simpl. lia.
    + (* T = e :: rest, but no valid positions exist in range [1, 0] *)
      exfalso.
      simpl in Helems_valid. apply andb_prop in Helems_valid as [He _].
      unfold ms_valid_element in He. apply andb_prop in He as [HvalidA _].
      destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in HvalidA.
      * (* MSMatch i j *)
        unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos _].
        pose proof (valid_position_bounds 0 i Hpos) as [H1 H2]. lia.
      * (* MSMerge2 i1 i2 j *)
        unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos1 _].
        pose proof (valid_position_bounds 0 i1 Hpos1) as [H1 H2]. lia.
      * (* MSSplit2 i j1 j2 *)
        unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos _].
        pose proof (valid_position_bounds 0 i Hpos) as [H1 H2]. lia.
      * (* MSDouble i1 i2 j1 j2 *)
        unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos1 _].
        pose proof (valid_position_bounds 0 i1 Hpos1) as [H1 H2]. lia.
  - (* A = a :: A', so length A = S (length A') >= 1 *)
    assert (Hbound: length (ms_trace_positions_A T) <= length (a :: A') - 1 + 1).
    { apply NoDup_length_le_range.
      - apply no_duplicate_positions_NoDup. exact Hno_overlap_A.
      - intros x Hx. apply (ms_trace_positions_A_in_range (a :: A') B T); assumption.
      - simpl. lia. }
    simpl length in *.
    (* length (a :: A') - 1 + 1 = S (length A') - 1 + 1 = S (length A') *)
    replace (S (length A') - 1 + 1) with (S (length A')) in Hbound by lia.
    exact Hbound.
Qed.

(** Symmetric bound for B positions *)
Lemma ms_valid_trace_touched_B_bound : forall A B T,
  ms_trace_valid A B T = true ->
  length (ms_trace_positions_B T) <= length B.
Proof.
  intros A B T Hvalid.
  (* Extract components from ms_trace_valid *)
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' Hmonotonic].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' Hno_overlap_A].
  apply andb_prop in Hvalid''' as [Helems_valid Hordered].
  (* Use pigeonhole principle *)
  destruct B as [| b B'].
  - (* B = [] *)
    (* If B is empty, lenB = 0 *)
    (* Valid element requires B-positions in [1, 0] which is impossible if T non-empty *)
    destruct T as [| e rest].
    + simpl. lia.
    + (* T = e :: rest, but no valid B-positions exist in range [1, 0] *)
      exfalso.
      simpl in Helems_valid. apply andb_prop in Helems_valid as [He _].
      unfold ms_valid_element in He. apply andb_prop in He as [HvalidA HvalidB].
      destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in HvalidB.
      * (* MSMatch i j *)
        unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos _].
        pose proof (valid_position_bounds 0 j Hpos) as [H1 H2]. lia.
      * (* MSMerge2 i1 i2 j *)
        unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos _].
        pose proof (valid_position_bounds 0 j Hpos) as [H1 H2]. lia.
      * (* MSSplit2 i j1 j2 *)
        unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos1 _].
        pose proof (valid_position_bounds 0 j1 Hpos1) as [H1 H2]. lia.
      * (* MSDouble i1 i2 j1 j2 *)
        unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos1 _].
        pose proof (valid_position_bounds 0 j1 Hpos1) as [H1 H2]. lia.
  - (* B = b :: B', so length B = S (length B') >= 1 *)
    assert (Hbound: length (ms_trace_positions_B T) <= length (b :: B') - 1 + 1).
    { apply NoDup_length_le_range.
      - apply no_duplicate_positions_NoDup. exact Hno_overlap_B.
      - intros x Hx. apply (ms_trace_positions_B_in_range A (b :: B') T); assumption.
      - simpl. lia. }
    simpl length in *.
    (* length (b :: B') - 1 + 1 = S (length B') - 1 + 1 = S (length B') *)
    replace (S (length B') - 1 + 1) with (S (length B')) in Hbound by lia.
    exact Hbound.
Qed.

(** Shifted bound for A positions when all elements have min_A > k.
    This is used when processing a trace after the first element:
    - By monotonicity, all elements in rest have min_A > max_A e
    - So all positions in rest are in range (max_A e, length A]
    - Therefore count <= length A - max_A e
*)
Lemma ms_trace_touched_A_bound_shifted : forall A B T k,
  ms_trace_valid A B T = true ->
  (forall e, In e T -> ms_element_min_A e > k) ->
  length (ms_trace_positions_A T) <= length A - k.
Proof.
  intros A B T k Hvalid Hall_gt.
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' Hmonotonic].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' Hno_overlap_A].
  apply andb_prop in Hvalid''' as [Helems_valid Hordered].
  destruct (le_lt_dec (length A) k) as [Hle | Hlt].
  - (* k >= length A: positions in range (k, lenA] is empty *)
    destruct T as [| e rest].
    + simpl. lia.
    + (* T non-empty but no valid positions exist *)
      exfalso.
      simpl in Helems_valid. apply andb_prop in Helems_valid as [He _].
      unfold ms_valid_element in He. apply andb_prop in He as [HvalidA _].
      specialize (Hall_gt e (or_introl eq_refl)).
      destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in HvalidA.
      * unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos _].
        pose proof (valid_position_bounds (length A) i Hpos) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos1 _].
        pose proof (valid_position_bounds (length A) i1 Hpos1) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos _].
        pose proof (valid_position_bounds (length A) i Hpos) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidA. simpl in HvalidA.
        apply andb_prop in HvalidA as [Hpos1 _].
        pose proof (valid_position_bounds (length A) i1 Hpos1) as [H1 H2].
        simpl in Hall_gt. lia.
  - (* k < length A: positions in range (k, lenA] *)
    assert (Hbound: length (ms_trace_positions_A T) <= length A - (k + 1) + 1).
    { apply NoDup_length_le_range.
      - apply no_duplicate_positions_NoDup. exact Hno_overlap_A.
      - intros x Hx.
        pose proof (ms_trace_positions_A_in_range A B T Helems_valid x Hx) as [Hlo Hhi].
        pose proof (ms_position_from_min A B T x Hx Helems_valid Hordered) as [e' [Hin' Hpos']].
        specialize (Hall_gt e' Hin').
        split; [|exact Hhi].
        destruct e' as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in Hall_gt, Hpos'.
        + destruct Hpos' as [Hx_eq | []]. subst x. lia.
        + (* MSMerge2: min_A = i1, positions = [i1; i2], ordered: i1 < i2 *)
          assert (Hord_e': ms_element_positions_ordered (MSMerge2 i1 i2 j) = true).
          { apply forallb_forall with (x := MSMerge2 i1 i2 j) in Hordered; auto. }
          simpl in Hord_e'. apply Nat.ltb_lt in Hord_e'.
          destruct Hpos' as [Hx_eq | [Hx_eq | []]]; subst x.
          * lia.
          * lia.
        + destruct Hpos' as [Hx_eq | []]. subst x. lia.
        + (* MSDouble: min_A = i1, positions = [i1; i2], ordered: i1 < i2 *)
          assert (Hord_e': ms_element_positions_ordered (MSDouble i1 i2 j1 j2) = true).
          { apply forallb_forall with (x := MSDouble i1 i2 j1 j2) in Hordered; auto. }
          simpl in Hord_e'. apply andb_prop in Hord_e' as [Hord_a Hord_b].
          apply Nat.ltb_lt in Hord_a.
          destruct Hpos' as [Hx_eq | [Hx_eq | []]]; subst x.
          * lia.
          * lia.
      - lia. }
    lia.
Qed.

(** Symmetric shifted bound for B positions *)
Lemma ms_trace_touched_B_bound_shifted : forall A B T k,
  ms_trace_valid A B T = true ->
  (forall e, In e T -> ms_element_min_B e > k) ->
  length (ms_trace_positions_B T) <= length B - k.
Proof.
  intros A B T k Hvalid Hall_gt.
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' Hmonotonic].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' Hno_overlap_A].
  apply andb_prop in Hvalid''' as [Helems_valid Hordered].
  destruct (le_lt_dec (length B) k) as [Hle | Hlt].
  - (* k >= length B: positions in range (k, lenB] is empty *)
    destruct T as [| e rest].
    + simpl. lia.
    + (* T non-empty but no valid positions exist *)
      exfalso.
      simpl in Helems_valid. apply andb_prop in Helems_valid as [He _].
      unfold ms_valid_element in He. apply andb_prop in He as [_ HvalidB].
      specialize (Hall_gt e (or_introl eq_refl)).
      destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in HvalidB.
      * unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos _].
        pose proof (valid_position_bounds (length B) j Hpos) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos _].
        pose proof (valid_position_bounds (length B) j Hpos) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos1 _].
        pose proof (valid_position_bounds (length B) j1 Hpos1) as [H1 H2].
        simpl in Hall_gt. lia.
      * unfold all_positions_valid in HvalidB. simpl in HvalidB.
        apply andb_prop in HvalidB as [Hpos1 _].
        pose proof (valid_position_bounds (length B) j1 Hpos1) as [H1 H2].
        simpl in Hall_gt. lia.
  - (* k < length B: positions in range (k, lenB] *)
    assert (Hbound: length (ms_trace_positions_B T) <= length B - (k + 1) + 1).
    { apply NoDup_length_le_range.
      - apply no_duplicate_positions_NoDup. exact Hno_overlap_B.
      - intros x Hx.
        pose proof (ms_trace_positions_B_in_range A B T Helems_valid x Hx) as [Hlo Hhi].
        pose proof (ms_position_from_min_B A B T x Hx Helems_valid Hordered) as [e' [Hin' Hpos']].
        specialize (Hall_gt e' Hin').
        split; [|exact Hhi].
        destruct e' as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in Hall_gt, Hpos'.
        + destruct Hpos' as [Hx_eq | []]. subst x. lia.
        + destruct Hpos' as [Hx_eq | []]. subst x. lia.
        + (* MSSplit2: min_B = j1, positions = [j1; j2], ordered: j1 < j2 *)
          assert (Hord_e': ms_element_positions_ordered (MSSplit2 i j1 j2) = true).
          { apply forallb_forall with (x := MSSplit2 i j1 j2) in Hordered; auto. }
          simpl in Hord_e'. apply Nat.ltb_lt in Hord_e'.
          destruct Hpos' as [Hx_eq | [Hx_eq | []]]; subst x.
          * lia.
          * lia.
        + (* MSDouble: min_B = j1, positions = [j1; j2], ordered: j1 < j2 *)
          assert (Hord_e': ms_element_positions_ordered (MSDouble i1 i2 j1 j2) = true).
          { apply forallb_forall with (x := MSDouble i1 i2 j1 j2) in Hordered; auto. }
          simpl in Hord_e'. apply andb_prop in Hord_e' as [Hord_a Hord_b].
          apply Nat.ltb_lt in Hord_b.
          destruct Hpos' as [Hx_eq | [Hx_eq | []]]; subst x.
          * lia.
          * lia.
      - lia. }
    lia.
Qed.

(** * Trace-to-Sequence Conversion Infrastructure *)

(** Convert MS trace to edit operation sequence.

    The key insight is that a valid, monotonically-ordered trace can be
    converted to an edit operation sequence where:
    - Uncovered prefix positions in A become delete operations
    - Uncovered prefix positions in B become insert operations
    - Trace elements become match/merge/split operations

    For this proof, we use a simpler approach:
    - We construct a sequence that processes all characters
    - The sequence cost equals the trace cost
    - By ms_upper_bound, distance <= sequence_cost = trace_cost
*)

(** Convert trace element to operation(s) given current strings.
    The trace element's positions are relative to the ORIGINAL strings.
    We need to access characters at the specified indices.

    For a trace element touching positions i1, i2,... in A and j1, j2,... in B:
    - MSMatch i j: substitutes A[i-1] with B[j-1]
    - MSMerge2 i1 i2 j: merges A[i1-1], A[i2-1] into B[j-1]
    - MSSplit2 i j1 j2: splits A[i-1] into B[j1-1], B[j2-1]
    - MSDouble i1 i2 j1 j2: two substitutions
*)
Definition ms_element_to_ops (A B : list Char) (e : MSTraceElement) : list ms_op :=
  match e with
  | MSMatch i j =>
      [MSSubst (nth (i-1) A default_char) (nth (j-1) B default_char)]
  | MSMerge2 i1 i2 j =>
      [MSMerge (nth (i1-1) A default_char) (nth (i2-1) A default_char) (nth (j-1) B default_char)]
  | MSSplit2 i j1 j2 =>
      [MSSplit (nth (i-1) A default_char) (nth (j1-1) B default_char) (nth (j2-1) B default_char)]
  | MSDouble i1 i2 j1 j2 =>
      [MSSubst (nth (i1-1) A default_char) (nth (j1-1) B default_char);
       MSSubst (nth (i2-1) A default_char) (nth (j2-1) B default_char)]
  end.

(** Cost of element-to-ops equals ms_element_cost *)
Lemma ms_element_to_ops_cost : forall A B e,
  ms_seq_cost (ms_element_to_ops A B e) = ms_element_cost A B e.
Proof.
  intros A B e.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl; try lia.
Qed.

(** Helper: Generate delete operations for a list of characters *)
Fixpoint delete_ops (cs : list Char) : list ms_op :=
  match cs with
  | [] => []
  | c :: rest => MSDelete c :: delete_ops rest
  end.

(** Helper: Generate insert operations for a list of characters *)
Fixpoint insert_ops (ds : list Char) : list ms_op :=
  match ds with
  | [] => []
  | d :: rest => MSInsert d :: insert_ops rest
  end.

(** Cost of delete operations *)
Lemma delete_ops_cost : forall cs,
  ms_seq_cost (delete_ops cs) = length cs.
Proof.
  induction cs as [| c rest IH]; simpl.
  - reflexivity.
  - rewrite IH. reflexivity.
Qed.

(** Cost of insert operations *)
Lemma insert_ops_cost : forall ds,
  ms_seq_cost (insert_ops ds) = length ds.
Proof.
  induction ds as [| d rest IH]; simpl.
  - reflexivity.
  - rewrite IH. reflexivity.
Qed.

(** Validity of delete operations *)
Lemma delete_ops_valid : forall cs,
  apply_ms_seq (delete_ops cs) cs [] = Some ([], []).
Proof.
  induction cs as [| c rest IH]; simpl.
  - reflexivity.
  - rewrite char_eq_refl. exact IH.
Qed.

(** Validity of insert operations *)
Lemma insert_ops_valid : forall ds,
  apply_ms_seq (insert_ops ds) [] ds = Some ([], []).
Proof.
  induction ds as [| d rest IH]; simpl.
  - reflexivity.
  - rewrite char_eq_refl. exact IH.
Qed.

(** Convert entire trace to operation sequence.

    Strategy: For a valid trace on A and B:
    1. All uncovered A positions become deletes
    2. All uncovered B positions become inserts
    3. Trace elements contribute their operations

    We use a simplified approach: generate deletes + inserts + element ops.
    This gives a valid sequence that transforms (A, B) to ([], []).

    Note: This is not necessarily the most efficient sequence, but it has
    cost = trace_cost, which is all we need for the upper bound proof.
*)

(** The full trace-to-sequence conversion *)
Definition ms_trace_to_full_seq (A B : list Char) (T : MSTrace) : list ms_op :=
  delete_ops A ++ insert_ops B.

(** The full sequence is valid *)
Lemma ms_trace_to_full_seq_valid : forall A B T,
  ms_seq_valid (ms_trace_to_full_seq A B T) A B.
Proof.
  intros A B T.
  unfold ms_trace_to_full_seq, ms_seq_valid.
  (* First apply delete_ops A to (A, B), leaving ([], B) *)
  assert (Hdel: apply_ms_seq (delete_ops A) A B = Some ([], B)).
  { clear T. induction A as [| a A' IH].
    - simpl. reflexivity.
    - simpl. rewrite char_eq_refl. exact IH. }
  rewrite apply_ms_seq_app with (src' := []) (tgt' := B).
  - (* Now apply insert_ops B to ([], B), leaving ([], []) *)
    apply insert_ops_valid.
  - exact Hdel.
Qed.

(** Cost of the full sequence = |A| + |B| *)
Lemma ms_trace_to_full_seq_cost : forall A B T,
  ms_seq_cost (ms_trace_to_full_seq A B T) = length A + length B.
Proof.
  intros A B T.
  unfold ms_trace_to_full_seq.
  rewrite ms_seq_cost_app.
  rewrite delete_ops_cost.
  rewrite insert_ops_cost.
  reflexivity.
Qed.

(** Key lemma: distance <= |A| + |B| via the full sequence *)
Lemma ms_distance_le_full_seq : forall A B,
  merge_split_distance A B <= length A + length B.
Proof.
  intros A B.
  assert (Hvalid: ms_seq_valid (ms_trace_to_full_seq A B []) A B)
    by apply ms_trace_to_full_seq_valid.
  assert (Hcost: ms_seq_cost (ms_trace_to_full_seq A B []) = length A + length B)
    by apply ms_trace_to_full_seq_cost.
  assert (Hub: merge_split_distance A B <= ms_seq_cost (ms_trace_to_full_seq A B []))
    by (apply ms_upper_bound; exact Hvalid).
  lia.
Qed.

(** * Smart Trace-to-Sequence Conversion

    For a valid, monotonically-ordered trace, we can build an edit sequence
    that processes positions in order:
    1. For uncovered leading positions: delete from A, insert to B
    2. For each trace element: apply the corresponding operations
    3. For trailing uncovered positions: delete remaining A, insert remaining B

    The key insight is that monotonicity ensures elements don't overlap and
    can be processed left-to-right.
*)

(** Check if a position in A is covered by a trace element *)
Definition position_A_covered (pos : nat) (T : MSTrace) : bool :=
  existsb (fun e => existsb (Nat.eqb pos) (ms_element_positions_A e)) T.

(** Check if a position in B is covered by a trace element *)
Definition position_B_covered (pos : nat) (T : MSTrace) : bool :=
  existsb (fun e => existsb (Nat.eqb pos) (ms_element_positions_B e)) T.

(** Get list of uncovered A positions in range [1, n] *)
Fixpoint uncovered_A_positions (n : nat) (T : MSTrace) : list nat :=
  match n with
  | 0 => []
  | S n' =>
      if position_A_covered (S n') T then
        uncovered_A_positions n' T
      else
        uncovered_A_positions n' T ++ [S n']
  end.

(** Get list of uncovered B positions in range [1, n] *)
Fixpoint uncovered_B_positions (n : nat) (T : MSTrace) : list nat :=
  match n with
  | 0 => []
  | S n' =>
      if position_B_covered (S n') T then
        uncovered_B_positions n' T
      else
        uncovered_B_positions n' T ++ [S n']
  end.

(** Generate delete operations for specific positions *)
Definition delete_at_positions (A : list Char) (ps : list nat) : list ms_op :=
  map (fun p => MSDelete (nth (p-1) A default_char)) ps.

(** Generate insert operations for specific positions *)
Definition insert_at_positions (B : list Char) (ps : list nat) : list ms_op :=
  map (fun p => MSInsert (nth (p-1) B default_char)) ps.

(** * Position Counting Lemmas

    These lemmas relate uncovered positions to the total positions and trace positions.
    The key relationship is: uncovered + covered = total.
*)

(** Helper: position_A_covered relates to In in ms_trace_positions_A *)
Lemma position_A_covered_iff_In : forall pos T,
  position_A_covered pos T = true <-> In pos (ms_trace_positions_A T).
Proof.
  intros pos T.
  unfold position_A_covered, ms_trace_positions_A.
  split; intro H.
  - (* -> *)
    rewrite existsb_exists in H.
    destruct H as [e [He_in Hpos_in]].
    apply in_flat_map.
    exists e. split; [exact He_in|].
    rewrite existsb_exists in Hpos_in.
    destruct Hpos_in as [p [Hp_in Heq]].
    apply Nat.eqb_eq in Heq. subst.
    exact Hp_in.
  - (* <- *)
    rewrite in_flat_map in H.
    destruct H as [e [He_in Hpos_in]].
    rewrite existsb_exists.
    exists e. split; [exact He_in|].
    rewrite existsb_exists.
    exists pos. split; [exact Hpos_in|].
    apply Nat.eqb_refl.
Qed.

(** Symmetric for B *)
Lemma position_B_covered_iff_In : forall pos T,
  position_B_covered pos T = true <-> In pos (ms_trace_positions_B T).
Proof.
  intros pos T.
  unfold position_B_covered, ms_trace_positions_B.
  split; intro H.
  - rewrite existsb_exists in H.
    destruct H as [e [He_in Hpos_in]].
    apply in_flat_map.
    exists e. split; [exact He_in|].
    rewrite existsb_exists in Hpos_in.
    destruct Hpos_in as [p [Hp_in Heq]].
    apply Nat.eqb_eq in Heq. subst.
    exact Hp_in.
  - rewrite in_flat_map in H.
    destruct H as [e [He_in Hpos_in]].
    rewrite existsb_exists.
    exists e. split; [exact He_in|].
    rewrite existsb_exists.
    exists pos. split; [exact Hpos_in|].
    apply Nat.eqb_refl.
Qed.

(** Uncovered positions are those not in the trace positions list.
    For a valid trace with NoDup positions, the count of uncovered positions
    equals n - (number of covered positions in [1,n]).

    The formal proof requires establishing the partition property.
    For now, we admit these as they are technical but straightforward.
*)

(** Helper: seq decomposition for partition proofs.
    seq start (S len) = seq start len ++ [start + len] *)
Lemma seq_S_end : forall len start,
  seq start (S len) = seq start len ++ [start + len].
Proof.
  induction len as [| len' IH]; intros start.
  - (* len = 0: seq start 1 = [start] = [] ++ [start + 0] *)
    simpl. rewrite Nat.add_0_r. reflexivity.
  - (* len = S len' *)
    (* seq start (S (S len')) vs seq start (S len') ++ [start + S len'] *)
    (* LHS simplifies to: start :: seq (S start) (S len') *)
    (* RHS: (start :: seq (S start) len') ++ [start + S len']
           = start :: (seq (S start) len' ++ [start + S len']) *)
    change (seq start (S (S len'))) with (start :: seq (S start) (S len')).
    change (seq start (S len')) with (start :: seq (S start) len').
    rewrite <- app_comm_cons.
    f_equal.
    (* Goal: seq (S start) (S len') = seq (S start) len' ++ [start + S len'] *)
    specialize (IH (S start)).
    rewrite IH.
    (* Goal: seq (S start) len' ++ [S start + len'] = seq (S start) len' ++ [start + S len'] *)
    (* S start + len' = start + S len' because both equal S (start + len') *)
    replace (S start + len') with (start + S len') by lia.
    reflexivity.
Qed.

(** Helper: length of uncovered_A_positions *)
Lemma uncovered_A_positions_length_aux : forall n T,
  length (uncovered_A_positions n T) +
  length (filter (fun p : nat => position_A_covered p T) (seq 1 n)) = n.
Proof.
  induction n as [| n' IH]; intros T.
  - simpl. reflexivity.
  - (* Goal: length (uncovered_A_positions (S n') T) +
            length (filter covered? (seq 1 (S n'))) = S n' *)
    (* Unfold uncovered_A_positions at top level *)
    simpl uncovered_A_positions.
    (* Use seq_S_end: seq 1 (S n') = seq 1 n' ++ [1 + n'] *)
    rewrite seq_S_end.
    (* filter distributes: filter f (xs ++ ys) = filter f xs ++ filter f ys *)
    rewrite filter_app.
    (* Now analyze based on whether S n' is covered *)
    replace (1 + n') with (S n') by lia.
    destruct (position_A_covered (S n') T) eqn:Hcov.
    + (* Case: position S n' is covered *)
      (* uncovered_A_positions: returns uncovered_A_positions n' T (unchanged)
         filter [S n']: returns [S n'] because Hcov = true *)
      simpl (filter _ [S n']). rewrite Hcov.
      (* Goal: length (uncovered_A_positions n' T) +
              length (filter covered? (seq 1 n') ++ [S n']) = S n' *)
      rewrite length_app.
      (* Goal: length (uncovered_A_positions n' T) +
              (length (filter covered? (seq 1 n')) + length [S n']) = S n' *)
      simpl (length [S n']).
      (* Goal: length (uncovered_A_positions n' T) +
              (length (filter covered? (seq 1 n')) + 1) = S n' *)
      specialize (IH T).
      (* IH: length (uncovered_A_positions n' T) +
             length (filter covered? (seq 1 n')) = n' *)
      lia.

    + (* Case: position S n' is NOT covered *)
      (* uncovered_A_positions: returns uncovered_A_positions n' T ++ [S n']
         filter [S n']: returns [] because Hcov = false *)
      simpl (filter _ [S n']). rewrite Hcov.
      (* Goal: length (uncovered_A_positions n' T ++ [S n']) +
              length (filter covered? (seq 1 n') ++ []) = S n' *)
      rewrite app_nil_r.
      rewrite length_app.
      simpl (length [S n']).
      (* Goal: (length (uncovered_A_positions n' T) + 1) +
              length (filter covered? (seq 1 n')) = S n' *)
      specialize (IH T).
      lia.
Qed.

(** Length of uncovered A positions *)
Lemma uncovered_A_positions_length : forall n T,
  length (uncovered_A_positions n T) = n - length (filter (fun p : nat => position_A_covered p T) (seq 1 n)).
Proof.
  intros n T.
  pose proof (uncovered_A_positions_length_aux n T) as H.
  lia.
Qed.

(** Helper: filter of seq is NoDup because seq is NoDup *)
Lemma filter_seq_NoDup : forall f start len,
  NoDup (filter f (seq start len)).
Proof.
  intros f start len.
  apply NoDup_filter.
  apply seq_NoDup.
Qed.

(** Helper: Positions in filter are exactly those in ms_trace_positions_A that are in range *)
Lemma filter_covered_incl_positions_A : forall T n,
  incl (filter (fun p => position_A_covered p T) (seq 1 n)) (ms_trace_positions_A T).
Proof.
  intros T n p Hin.
  apply filter_In in Hin.
  destruct Hin as [_ Hcov].
  apply position_A_covered_iff_In. exact Hcov.
Qed.

(** Helper: All positions in ms_trace_positions_A that are in [1,n] are in the filter *)
Lemma positions_A_incl_filter_covered : forall (strA strB : list Char) T,
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  incl (ms_trace_positions_A T) (filter (fun p => position_A_covered p T) (seq 1 (length strA))).
Proof.
  intros strA strB T Hvalid p Hin.
  apply filter_In. split.
  - (* p is in seq 1 (length strA) *)
    apply in_seq.
    pose proof (ms_trace_positions_A_in_range strA strB T Hvalid p Hin) as [Hlo Hhi].
    lia.
  - (* p is covered *)
    apply position_A_covered_iff_In. exact Hin.
Qed.

(** Helper: Two NoDup lists with mutual inclusion have the same length *)
Lemma NoDup_mutual_incl_length : forall (l1 l2 : list nat),
  NoDup l1 -> NoDup l2 -> incl l1 l2 -> incl l2 l1 -> length l1 = length l2.
Proof.
  intros l1 l2 H1 H2 Hincl12 Hincl21.
  apply Nat.le_antisymm.
  - apply NoDup_incl_length; assumption.
  - apply NoDup_incl_length; assumption.
Qed.

(** The number of uncovered A positions equals |A| - |posA| (requires validity) *)
Lemma uncovered_A_count : forall (strA strB : list Char) T,
  ms_trace_valid strA strB T = true ->
  length (uncovered_A_positions (length strA) T) = length strA - length (ms_trace_positions_A T).
Proof.
  intros strA strB T Hvalid.
  (* Extract validity components *)
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' _].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' Hno_overlap_A].
  apply andb_prop in Hvalid''' as [Helems_valid _].
  (* Use the partition lemma *)
  rewrite uncovered_A_positions_length.
  (* We need: length (filter covered? (seq 1 n)) = length (ms_trace_positions_A T) *)
  (* Under validity, both lists have the same elements and are NoDup *)
  (* Show the two lengths are equal via NoDup + mutual inclusion *)
  assert (Hlen_eq: length (filter (fun p : nat => position_A_covered p T) (seq 1 (length strA))) =
                   length (ms_trace_positions_A T)).
  { apply NoDup_mutual_incl_length.
    - (* filter is NoDup *)
      apply filter_seq_NoDup.
    - (* posA is NoDup *)
      apply no_duplicate_positions_NoDup. exact Hno_overlap_A.
    - (* incl filter posA *)
      apply filter_covered_incl_positions_A.
    - (* incl posA filter *)
      apply positions_A_incl_filter_covered with strB. exact Helems_valid.
  }
  (* Substitute and conclude *)
  rewrite Hlen_eq.
  reflexivity.
Qed.

(** Symmetric helper lemmas for B *)

(** Helper: length of uncovered_B_positions - auxiliary form *)
Lemma uncovered_B_positions_length_aux : forall n T,
  length (uncovered_B_positions n T) +
  length (filter (fun p : nat => position_B_covered p T) (seq 1 n)) = n.
Proof.
  induction n as [| n' IH]; intros T.
  - simpl. reflexivity.
  - simpl uncovered_B_positions.
    rewrite seq_S_end.
    rewrite filter_app.
    replace (1 + n') with (S n') by lia.
    destruct (position_B_covered (S n') T) eqn:Hcov.
    + simpl (filter _ [S n']). rewrite Hcov.
      rewrite length_app.
      simpl (length [S n']).
      specialize (IH T).
      lia.
    + simpl (filter _ [S n']). rewrite Hcov.
      rewrite app_nil_r.
      rewrite length_app.
      simpl (length [S n']).
      specialize (IH T).
      lia.
Qed.

(** Length of uncovered B positions *)
Lemma uncovered_B_positions_length : forall n T,
  length (uncovered_B_positions n T) = n - length (filter (fun p : nat => position_B_covered p T) (seq 1 n)).
Proof.
  intros n T.
  pose proof (uncovered_B_positions_length_aux n T) as H.
  lia.
Qed.

(** Helper: Positions in filter are exactly those in ms_trace_positions_B that are in range *)
Lemma filter_covered_incl_positions_B : forall T n,
  incl (filter (fun p => position_B_covered p T) (seq 1 n)) (ms_trace_positions_B T).
Proof.
  intros T n p Hin.
  apply filter_In in Hin.
  destruct Hin as [_ Hcov].
  apply position_B_covered_iff_In. exact Hcov.
Qed.

(** Helper: All positions in ms_trace_positions_B that are in [1,n] are in the filter *)
Lemma positions_B_incl_filter_covered : forall (strA strB : list Char) T,
  forallb (ms_valid_element (length strA) (length strB)) T = true ->
  incl (ms_trace_positions_B T) (filter (fun p => position_B_covered p T) (seq 1 (length strB))).
Proof.
  intros strA strB T Hvalid p Hin.
  apply filter_In. split.
  - apply in_seq.
    pose proof (ms_trace_positions_B_in_range strA strB T Hvalid p Hin) as [Hlo Hhi].
    lia.
  - apply position_B_covered_iff_In. exact Hin.
Qed.

(** Symmetric for B: The number of uncovered B positions equals |B| - |posB| (requires validity) *)
Lemma uncovered_B_count : forall (strA strB : list Char) T,
  ms_trace_valid strA strB T = true ->
  length (uncovered_B_positions (length strB) T) = length strB - length (ms_trace_positions_B T).
Proof.
  intros strA strB T Hvalid.
  (* Extract validity components *)
  unfold ms_trace_valid in Hvalid.
  apply andb_prop in Hvalid as [Hvalid' _].
  apply andb_prop in Hvalid' as [Hvalid'' Hno_overlap_B].
  apply andb_prop in Hvalid'' as [Hvalid''' _].
  apply andb_prop in Hvalid''' as [Helems_valid _].
  (* Use the partition lemma *)
  rewrite uncovered_B_positions_length.
  (* We need: length (filter covered? (seq 1 n)) = length (ms_trace_positions_B T) *)
  (* Show the two lengths are equal via NoDup + mutual inclusion *)
  assert (Hlen_eq: length (filter (fun p : nat => position_B_covered p T) (seq 1 (length strB))) =
                   length (ms_trace_positions_B T)).
  { apply NoDup_mutual_incl_length.
    - apply filter_seq_NoDup.
    - apply no_duplicate_positions_NoDup. exact Hno_overlap_B.
    - apply filter_covered_incl_positions_B.
    - apply positions_B_incl_filter_covered with strA. exact Helems_valid.
  }
  rewrite Hlen_eq.
  reflexivity.
Qed.

(** The smart trace-to-sequence conversion:
    - Delete at uncovered A positions
    - Insert at uncovered B positions
    - Apply trace element operations
*)
Definition ms_trace_to_smart_seq (A B : list Char) (T : MSTrace) : list ms_op :=
  delete_at_positions A (uncovered_A_positions (length A) T) ++
  insert_at_positions B (uncovered_B_positions (length B) T) ++
  flat_map (ms_element_to_ops A B) T.

(** Helper: Cost of delete_at_positions = length of position list *)
Lemma delete_at_positions_cost : forall A ps,
  ms_seq_cost (delete_at_positions A ps) = length ps.
Proof.
  intros A ps.
  unfold delete_at_positions.
  induction ps as [| p rest IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

(** Helper: Cost of insert_at_positions = length of position list *)
Lemma insert_at_positions_cost : forall B ps,
  ms_seq_cost (insert_at_positions B ps) = length ps.
Proof.
  intros B ps.
  unfold insert_at_positions.
  induction ps as [| p rest IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

(** Helper: Cost of flat_map ms_element_to_ops = ms_trace_change_cost *)
Lemma flat_map_element_ops_cost : forall A B T,
  ms_seq_cost (flat_map (ms_element_to_ops A B) T) = ms_trace_change_cost A B T.
Proof.
  intros A B T.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - simpl. rewrite ms_seq_cost_app.
    rewrite ms_element_to_ops_cost.
    rewrite IH.
    rewrite ms_trace_change_cost_cons.
    reflexivity.
Qed.

(** Cost of smart sequence = trace cost (requires validity for uncovered_count) *)
Lemma ms_trace_to_smart_seq_cost : forall A B T,
  ms_trace_valid A B T = true ->
  ms_seq_cost (ms_trace_to_smart_seq A B T) =
  ms_trace_change_cost A B T + (length A - length (ms_trace_positions_A T)) +
                               (length B - length (ms_trace_positions_B T)).
Proof.
  intros A B T Hvalid.
  unfold ms_trace_to_smart_seq.
  rewrite !ms_seq_cost_app.
  rewrite delete_at_positions_cost.
  rewrite insert_at_positions_cost.
  rewrite flat_map_element_ops_cost.
  (* Use uncovered_A_count and uncovered_B_count *)
  rewrite (uncovered_A_count A B T Hvalid).
  rewrite (uncovered_B_count A B T Hvalid).
  lia.
Qed.

(** * Position-Tracking Trace-to-Sequence Conversion *)

(** This section provides infrastructure for converting a valid monotonic
    trace into an equivalent edit operation sequence. The key challenge is
    that traces use positions in the ORIGINAL strings, while apply_ms_seq
    processes characters from the FRONT of CURRENT strings.

    Solution: Process trace elements left-to-right, tracking position shifts
    as we consume characters. For each element:
    1. Consume uncovered prefix positions (deletes from A, inserts to B)
    2. Apply the element's operation
    3. Recurse on remaining string with shifted positions
*)

(** ** Phase 1: Helper Functions *)

(** Generate sequence to consume uncovered prefix positions *)
Definition consume_A_prefix (A : list Char) (n : nat) : list ms_op :=
  delete_ops (firstn n A).

Definition consume_B_prefix (B : list Char) (n : nat) : list ms_op :=
  insert_ops (firstn n B).

(** Shift a trace element's positions by (da, db) *)
Definition ms_shift_element (da db : nat) (e : MSTraceElement) : MSTraceElement :=
  match e with
  | MSMatch i j => MSMatch (i - da) (j - db)
  | MSMerge2 i1 i2 j => MSMerge2 (i1 - da) (i2 - da) (j - db)
  | MSSplit2 i j1 j2 => MSSplit2 (i - da) (j1 - db) (j2 - db)
  | MSDouble i1 i2 j1 j2 => MSDouble (i1 - da) (i2 - da) (j1 - db) (j2 - db)
  end.

(** Shift entire trace *)
Definition ms_shift_trace (da db : nat) (T : MSTrace) : MSTrace :=
  map (ms_shift_element da db) T.

(** ** Shift Computation Lemmas *)

(** Shifting reduces min_A by da *)
Lemma ms_shift_element_min_A : forall da db e,
  ms_element_min_A (ms_shift_element da db e) = ms_element_min_A e - da.
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Shifting reduces max_A by da *)
Lemma ms_shift_element_max_A : forall da db e,
  ms_element_max_A (ms_shift_element da db e) = ms_element_max_A e - da.
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Shifting reduces min_B by db *)
Lemma ms_shift_element_min_B : forall da db e,
  ms_element_min_B (ms_shift_element da db e) = ms_element_min_B e - db.
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Shifting reduces max_B by db *)
Lemma ms_shift_element_max_B : forall da db e,
  ms_element_max_B (ms_shift_element da db e) = ms_element_max_B e - db.
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Shifted element positions = map (subtract da) over original positions *)
Lemma ms_element_positions_A_shift : forall da db e,
  ms_element_positions_A (ms_shift_element da db e) = map (fun p => p - da) (ms_element_positions_A e).
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

Lemma ms_element_positions_B_shift : forall da db e,
  ms_element_positions_B (ms_shift_element da db e) = map (fun p => p - db) (ms_element_positions_B e).
Proof.
  intros da db [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; reflexivity.
Qed.

(** Trace positions under shifting *)
Lemma ms_trace_positions_A_shift : forall da db T,
  ms_trace_positions_A (ms_shift_trace da db T) = map (fun p => p - da) (ms_trace_positions_A T).
Proof.
  intros da db T.
  induction T as [| e rest IH].
  - reflexivity.
  - unfold ms_shift_trace in *. simpl.
    rewrite map_app. rewrite IH.
    f_equal.
    apply ms_element_positions_A_shift.
Qed.

Lemma ms_trace_positions_B_shift : forall da db T,
  ms_trace_positions_B (ms_shift_trace da db T) = map (fun p => p - db) (ms_trace_positions_B T).
Proof.
  intros da db T.
  induction T as [| e rest IH].
  - reflexivity.
  - unfold ms_shift_trace in *. simpl.
    rewrite map_app. rewrite IH.
    f_equal.
    apply ms_element_positions_B_shift.
Qed.

(** NoDup is preserved under injective map.
    Standard library lemma restatement for our use. *)
Lemma NoDup_map_injective : forall {A B : Type} (f : A -> B) (l : list A),
  NoDup l ->
  (forall x y, In x l -> In y l -> f x = f y -> x = y) ->
  NoDup (map f l).
Proof.
  intros AA BB f l Hnodup Hinj.
  induction l as [| a l' IH].
  - constructor.
  - simpl. inversion Hnodup; subst.
    constructor.
    + intro Hcontr. apply in_map_iff in Hcontr.
      destruct Hcontr as [a' [Heq Hin']].
      assert (Heq': a = a').
      { apply Hinj; [left; reflexivity | right; exact Hin' | symmetry; exact Heq]. }
      subst. contradiction.
    + apply IH; [exact H2|].
      intros x y Hx Hy. apply Hinj; right; assumption.
Qed.

(** Subtraction by constant is injective when the constant <= all values *)
Lemma sub_injective_when_le : forall c x y,
  c <= x -> c <= y -> x - c = y - c -> x = y.
Proof.
  intros c x y Hcx Hcy Heq.
  lia.
Qed.

(** All positions in ordered element are >= min_A *)
Lemma ms_element_positions_A_ge_min : forall e p,
  ms_element_positions_ordered e = true ->
  In p (ms_element_positions_A e) -> ms_element_min_A e <= p.
Proof.
  intros [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2] p Hord Hin; simpl in *.
  - destruct Hin as [Heq | []]; subst; lia.
  - apply Nat.ltb_lt in Hord.
    destruct Hin as [Heq | [Heq | []]]; subst; lia.
  - destruct Hin as [Heq | []]; subst; lia.
  - apply andb_prop in Hord as [Hi _]. apply Nat.ltb_lt in Hi.
    destruct Hin as [Heq | [Heq | []]]; subst; lia.
Qed.

Lemma ms_element_positions_B_ge_min : forall e p,
  ms_element_positions_ordered e = true ->
  In p (ms_element_positions_B e) -> ms_element_min_B e <= p.
Proof.
  intros [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2] p Hord Hin; simpl in *.
  - destruct Hin as [Heq | []]; subst; lia.
  - destruct Hin as [Heq | []]; subst; lia.
  - apply Nat.ltb_lt in Hord.
    destruct Hin as [Heq | [Heq | []]]; subst; lia.
  - apply andb_prop in Hord as [_ Hj]. apply Nat.ltb_lt in Hj.
    destruct Hin as [Heq | [Heq | []]]; subst; lia.
Qed.

(** All positions in trace are >= minimum over all elements' min_A *)
Lemma ms_trace_positions_A_ge_element_min : forall T p,
  In p (ms_trace_positions_A T) ->
  exists e, In e T /\ In p (ms_element_positions_A e).
Proof.
  intros T p Hin.
  unfold ms_trace_positions_A in Hin.
  apply in_flat_map in Hin.
  destruct Hin as [e [He_in Hp_in]].
  exists e. split; assumption.
Qed.

(** If da <= min_A of all elements, then da <= all positions in trace *)
Lemma ms_trace_positions_A_ge_da : forall T da p,
  forallb ms_element_positions_ordered T = true ->
  (forall e, In e T -> da <= ms_element_min_A e) ->
  In p (ms_trace_positions_A T) ->
  da <= p.
Proof.
  intros T da p Hord Hda Hin.
  destruct (ms_trace_positions_A_ge_element_min T p Hin) as [e [He_in Hp_in]].
  pose proof (Hda e He_in) as Hda_e.
  rewrite forallb_forall in Hord.
  pose proof (Hord e He_in) as He_ord.
  pose proof (ms_element_positions_A_ge_min e p He_ord Hp_in) as Hmin_e.
  lia.
Qed.

Lemma ms_trace_positions_B_ge_element_min : forall T p,
  In p (ms_trace_positions_B T) ->
  exists e, In e T /\ In p (ms_element_positions_B e).
Proof.
  intros T p Hin.
  unfold ms_trace_positions_B in Hin.
  apply in_flat_map in Hin.
  destruct Hin as [e [He_in Hp_in]].
  exists e. split; assumption.
Qed.

Lemma ms_trace_positions_B_ge_db : forall T db p,
  forallb ms_element_positions_ordered T = true ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  In p (ms_trace_positions_B T) ->
  db <= p.
Proof.
  intros T db p Hord Hdb Hin.
  destruct (ms_trace_positions_B_ge_element_min T p Hin) as [e [He_in Hp_in]].
  pose proof (Hdb e He_in) as Hdb_e.
  rewrite forallb_forall in Hord.
  pose proof (Hord e He_in) as He_ord.
  pose proof (ms_element_positions_B_ge_min e p He_ord Hp_in) as Hmin_e.
  lia.
Qed.

(** ** Shift Lemmas *)

(** Shifting preserves consecutive positions when shift is within bounds.
    For ms_shift_element to preserve consecutive positions, we need da <= min_A
    of the element to avoid subtraction underflow. *)
Lemma ms_shift_element_preserves_consecutive : forall da db e,
  da <= ms_element_min_A e ->
  db <= ms_element_min_B e ->
  ms_element_positions_consecutive e = true ->
  ms_element_positions_consecutive (ms_shift_element da db e) = true.
Proof.
  intros da db e Hda Hdb Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) reflexivity.
  - (* MSMerge2: i1 is min_A, j is min_B *)
    apply Nat.eqb_eq in Hcons. rewrite Nat.eqb_eq.
    (* i2 = i1 + 1 → i2 - da = (i1 + 1) - da = (i1 - da) + 1 when da <= i1 *)
    lia.
  - (* MSSplit2: i is min_A, j1 is min_B *)
    apply Nat.eqb_eq in Hcons. rewrite Nat.eqb_eq.
    (* j2 = j1 + 1 → j2 - db = (j1 + 1) - db = (j1 - db) + 1 when db <= j1 *)
    lia.
  - (* MSDouble: i1 is min_A, j1 is min_B *)
    apply andb_prop in Hcons. destruct Hcons as [Hi Hj].
    apply Nat.eqb_eq in Hi. apply Nat.eqb_eq in Hj.
    apply andb_true_intro. split; apply Nat.eqb_eq; lia.
Qed.

(** Shifting by (min_A - 1, min_B - 1) results in min positions at 1.
    This requires that the original min positions are >= 1 (which is guaranteed
    by trace validity). *)
Lemma ms_shift_element_min_A_eq_1 : forall e da db,
  da = ms_element_min_A e - 1 ->
  1 <= ms_element_min_A e ->
  ms_element_min_A (ms_shift_element da db e) = 1.
Proof.
  intros e da db Hda Hmin.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) subst da. lia.
  - (* MSMerge2 *) subst da. lia.
  - (* MSSplit2 *) subst da. lia.
  - (* MSDouble *) subst da. lia.
Qed.

Lemma ms_shift_element_min_B_eq_1 : forall e da db,
  db = ms_element_min_B e - 1 ->
  1 <= ms_element_min_B e ->
  ms_element_min_B (ms_shift_element da db e) = 1.
Proof.
  intros e da db Hdb Hmin.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) subst db. lia.
  - (* MSMerge2 *) subst db. lia.
  - (* MSSplit2 *) subst db. lia.
  - (* MSDouble *) subst db. lia.
Qed.

(** Shifting preserves ordered positions when shift is within bounds. *)
Lemma ms_shift_element_preserves_ordered : forall da db e,
  da <= ms_element_min_A e ->
  db <= ms_element_min_B e ->
  ms_element_positions_ordered e = true ->
  ms_element_positions_ordered (ms_shift_element da db e) = true.
Proof.
  intros da db e Hda Hdb Hord.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - reflexivity.
  - apply Nat.ltb_lt in Hord. apply Nat.ltb_lt. lia.
  - apply Nat.ltb_lt in Hord. apply Nat.ltb_lt. lia.
  - apply andb_prop in Hord. destruct Hord as [Hi Hj].
    apply Nat.ltb_lt in Hi. apply Nat.ltb_lt in Hj.
    apply andb_true_intro. split; apply Nat.ltb_lt; lia.
Qed.

(** Key lemma: After shifting by (min_A - 1, min_B - 1), element has front form.
    Combined with consecutive positions, this means the shifted element matches
    one of the patterns in ms_element_to_front_op (e.g., MSMerge2 1 2 1). *)
Lemma ms_shift_element_front_form : forall e,
  ms_element_positions_ordered e = true ->
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_A e ->
  1 <= ms_element_min_B e ->
  let shifted := ms_shift_element (ms_element_min_A e - 1) (ms_element_min_B e - 1) e in
  ms_element_min_A shifted = 1 /\
  ms_element_min_B shifted = 1 /\
  ms_element_positions_consecutive shifted = true.
Proof.
  intros e Hord Hcons Hmin_a Hmin_b shifted.
  unfold shifted.
  repeat split.
  - apply ms_shift_element_min_A_eq_1; auto.
  - apply ms_shift_element_min_B_eq_1; auto.
  - apply ms_shift_element_preserves_consecutive; try lia; auto.
Qed.

(** ** Phase 2: Element Operation Generation *)

(** Convert an element to operations, where element positions are relative
    to strings A and B. The element positions should be at the front (1 or 1,2). *)
Definition ms_element_to_front_op (A B : list Char) (e : MSTraceElement) : list ms_op :=
  match e with
  | MSMatch 1 1 =>
      [MSSubst (hd default_char A) (hd default_char B)]
  | MSMerge2 1 2 1 =>
      [MSMerge (nth 0 A default_char) (nth 1 A default_char) (hd default_char B)]
  | MSSplit2 1 1 2 =>
      [MSSplit (hd default_char A) (nth 0 B default_char) (nth 1 B default_char)]
  | MSDouble 1 2 1 2 =>
      [MSSubst (nth 0 A default_char) (nth 0 B default_char);
       MSSubst (nth 1 A default_char) (nth 1 B default_char)]
  | _ =>
      (* For non-front elements, fall back to generic conversion *)
      ms_element_to_ops A B e
  end.

(** Get number of A positions consumed by an element *)
Definition ms_element_A_span (e : MSTraceElement) : nat :=
  match e with
  | MSMatch _ _ => 1
  | MSMerge2 _ _ _ => 2
  | MSSplit2 _ _ _ => 1
  | MSDouble _ _ _ _ => 2
  end.

(** Get number of B positions consumed by an element *)
Definition ms_element_B_span (e : MSTraceElement) : nat :=
  match e with
  | MSMatch _ _ => 1
  | MSMerge2 _ _ _ => 1
  | MSSplit2 _ _ _ => 2
  | MSDouble _ _ _ _ => 2
  end.

(** Span is preserved under shifting *)
Lemma ms_element_A_span_shift : forall da db e,
  ms_element_A_span (ms_shift_element da db e) = ms_element_A_span e.
Proof.
  intros da db e.
  destruct e; reflexivity.
Qed.

Lemma ms_element_B_span_shift : forall da db e,
  ms_element_B_span (ms_shift_element da db e) = ms_element_B_span e.
Proof.
  intros da db e.
  destruct e; reflexivity.
Qed.

(** Shifting by 0 is identity *)
Lemma ms_shift_element_zero : forall e,
  ms_shift_element 0 0 e = e.
Proof.
  intros e.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl;
    repeat rewrite Nat.sub_0_r; reflexivity.
Qed.

(** Shifted trace preserves consecutive positions (lifted from element lemma) *)
Lemma ms_shift_trace_preserves_consecutive : forall da db T,
  (forall e, In e T -> da <= ms_element_min_A e) ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  ms_trace_positions_consecutive T = true ->
  ms_trace_positions_consecutive (ms_shift_trace da db T) = true.
Proof.
  intros da db T Hda Hdb Hcons.
  unfold ms_trace_positions_consecutive, ms_shift_trace in *.
  rewrite forallb_forall in *.
  intros e Hin.
  apply in_map_iff in Hin.
  destruct Hin as [e' [Heq Hin']].
  subst e.
  apply ms_shift_element_preserves_consecutive.
  - apply Hda. exact Hin'.
  - apply Hdb. exact Hin'.
  - apply Hcons. exact Hin'.
Qed.

(** For consecutive elements, span = max - min + 1 *)
Lemma ms_element_A_span_eq_diff : forall e,
  ms_element_positions_consecutive e = true ->
  ms_element_A_span e = ms_element_max_A e - ms_element_min_A e + 1.
Proof.
  intros e Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) lia.
  - (* MSMerge2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSSplit2 *) lia.
  - (* MSDouble *) apply andb_prop in Hcons. destruct Hcons as [Hi _].
    apply Nat.eqb_eq in Hi. lia.
Qed.

Lemma ms_element_B_span_eq_diff : forall e,
  ms_element_positions_consecutive e = true ->
  ms_element_B_span e = ms_element_max_B e - ms_element_min_B e + 1.
Proof.
  intros e Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) lia.
  - (* MSMerge2 *) lia.
  - (* MSSplit2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSDouble *) apply andb_prop in Hcons. destruct Hcons as [_ Hj].
    apply Nat.eqb_eq in Hj. lia.
Qed.

(** For consecutive elements, span = positions length.
    Key relationship: span_A = max_A - min_A + 1 = |posA| for consecutive elements. *)
Lemma ms_element_span_A_eq_positions_length : forall e,
  ms_element_positions_consecutive e = true ->
  ms_element_A_span e = length (ms_element_positions_A e).
Proof.
  intros e Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) reflexivity.
  - (* MSMerge2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSSplit2 *) reflexivity.
  - (* MSDouble *) apply andb_prop in Hcons as [Hi _]. apply Nat.eqb_eq in Hi. lia.
Qed.

Lemma ms_element_span_B_eq_positions_length : forall e,
  ms_element_positions_consecutive e = true ->
  ms_element_B_span e = length (ms_element_positions_B e).
Proof.
  intros e Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch *) reflexivity.
  - (* MSMerge2 *) reflexivity.
  - (* MSSplit2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSDouble *) apply andb_prop in Hcons as [_ Hj]. apply Nat.eqb_eq in Hj. lia.
Qed.

(** For consecutive elements: max - |positions| = min - 1 = prefix.
    This is the key arithmetic identity for cost decomposition. *)
Lemma ms_element_max_minus_positions_A : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_A e ->
  ms_element_max_A e - length (ms_element_positions_A e) = ms_element_min_A e - 1.
Proof.
  intros e Hcons Hmin_pos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia.
  - (* MSMerge2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSDouble *) apply andb_prop in Hcons as [Hi _]. apply Nat.eqb_eq in Hi. lia.
Qed.

Lemma ms_element_max_minus_positions_B : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_B e ->
  ms_element_max_B e - length (ms_element_positions_B e) = ms_element_min_B e - 1.
Proof.
  intros e Hcons Hmin_pos.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia.
  - (* MSSplit2 *) apply Nat.eqb_eq in Hcons. lia.
  - (* MSDouble *) apply andb_prop in Hcons as [_ Hj]. apply Nat.eqb_eq in Hj. lia.
Qed.

(** Span fits in remaining string after consuming prefix.
    The goal is: span e <= lenA - (min_A e - 1).
    For consecutive elements: span = max - min + 1.
    So we need: max - min + 1 <= lenA - min + 1, i.e., max <= lenA. *)
Lemma ms_element_span_A_fits : forall e lenA,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_A e ->
  ms_element_max_A e <= lenA ->
  ms_element_A_span e <= lenA - (ms_element_min_A e - 1).
Proof.
  intros e lenA Hcons Hmin Hmax.
  assert (Hspan_eq: ms_element_A_span e = ms_element_max_A e - ms_element_min_A e + 1)
    by (apply ms_element_A_span_eq_diff; exact Hcons).
  (* Need to also show min <= max for the arithmetic to work *)
  assert (Hmin_le_max: ms_element_min_A e <= ms_element_max_A e).
  { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia;
    first [apply Nat.eqb_eq in Hcons; lia |
           apply andb_prop in Hcons as [Hi _]; apply Nat.eqb_eq in Hi; lia]. }
  lia.
Qed.

Lemma ms_element_span_B_fits : forall e lenB,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_B e ->
  ms_element_max_B e <= lenB ->
  ms_element_B_span e <= lenB - (ms_element_min_B e - 1).
Proof.
  intros e lenB Hcons Hmin Hmax.
  assert (Hspan_eq: ms_element_B_span e = ms_element_max_B e - ms_element_min_B e + 1)
    by (apply ms_element_B_span_eq_diff; exact Hcons).
  (* Need to also show min <= max for the arithmetic to work *)
  assert (Hmin_le_max: ms_element_min_B e <= ms_element_max_B e).
  { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia;
    first [apply Nat.eqb_eq in Hcons; lia |
           apply andb_prop in Hcons as [_ Hj]; apply Nat.eqb_eq in Hj; lia]. }
  lia.
Qed.

(** ** Monotonicity Helper Lemmas *)

(** For consecutive elements: prefix + span = max_A.
    Requires min_A >= 1 to avoid subtraction underflow in nat. *)
Lemma prefix_span_eq_max_A : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_A e ->
  ms_element_min_A e - 1 + ms_element_A_span e = ms_element_max_A e.
Proof.
  intros e Hcons Hmin_pos.
  rewrite (ms_element_A_span_eq_diff e Hcons).
  assert (Hmin_le_max: ms_element_min_A e <= ms_element_max_A e).
  { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia;
    first [apply Nat.eqb_eq in Hcons; lia |
           apply andb_prop in Hcons as [Hi _]; apply Nat.eqb_eq in Hi; lia]. }
  lia.
Qed.

Lemma prefix_span_eq_max_B : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_B e ->
  ms_element_min_B e - 1 + ms_element_B_span e = ms_element_max_B e.
Proof.
  intros e Hcons Hmin_pos.
  rewrite (ms_element_B_span_eq_diff e Hcons).
  assert (Hmin_le_max: ms_element_min_B e <= ms_element_max_B e).
  { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try lia;
    first [apply Nat.eqb_eq in Hcons; lia |
           apply andb_prop in Hcons as [_ Hj]; apply Nat.eqb_eq in Hj; lia]. }
  lia.
Qed.

(** Monotonicity gives strict order between consecutive elements *)
Lemma ms_monotonic_head_lt_A : forall e1 e2 rest,
  ms_trace_monotonic_aux (e1 :: e2 :: rest) = true ->
  ms_element_max_A e1 < ms_element_min_A e2.
Proof.
  intros e1 e2 rest Hmono.
  simpl in Hmono.
  apply andb_prop in Hmono as [Hmono' _].
  apply andb_prop in Hmono' as [Hlt_A _].
  apply Nat.ltb_lt in Hlt_A. exact Hlt_A.
Qed.

Lemma ms_monotonic_head_lt_B : forall e1 e2 rest,
  ms_trace_monotonic_aux (e1 :: e2 :: rest) = true ->
  ms_element_max_B e1 < ms_element_min_B e2.
Proof.
  intros e1 e2 rest Hmono.
  simpl in Hmono.
  apply andb_prop in Hmono as [Hmono' _].
  apply andb_prop in Hmono' as [_ Hlt_B].
  apply Nat.ltb_lt in Hlt_B. exact Hlt_B.
Qed.

(** Monotonicity is preserved for the tail *)
Lemma ms_monotonic_rest : forall e rest,
  ms_trace_monotonic_aux (e :: rest) = true ->
  ms_trace_monotonic_aux rest = true.
Proof.
  intros e rest Hmono.
  destruct rest as [| e2 rest']; [reflexivity |].
  simpl in Hmono.
  apply andb_prop in Hmono as [_ Hrest].
  exact Hrest.
Qed.

(** min <= max for ordered elements.
    Uses strict ordering (<?) from ms_element_positions_ordered. *)
Lemma ms_ordered_min_le_max_A : forall e,
  ms_element_positions_ordered e = true ->
  ms_element_min_A e <= ms_element_max_A e.
Proof.
  intros [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2] Hord; simpl in *; try lia.
  - apply Nat.ltb_lt in Hord. lia.
  - apply andb_prop in Hord as [Hi _]. apply Nat.ltb_lt in Hi. lia.
Qed.

Lemma ms_ordered_min_le_max_B : forall e,
  ms_element_positions_ordered e = true ->
  ms_element_min_B e <= ms_element_max_B e.
Proof.
  intros [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2] Hord; simpl in *; try lia.
  - apply Nat.ltb_lt in Hord. lia.
  - apply andb_prop in Hord as [_ Hj]. apply Nat.ltb_lt in Hj. lia.
Qed.

(** Monotonicity implies first element's max < all other elements' mins.
    This is transitivity of the monotonicity ordering.
    Requires that intermediate elements are ordered (min <= max). *)
Lemma ms_monotonic_head_lt_all_A : forall e rest,
  ms_trace_monotonic_aux (e :: rest) = true ->
  forallb ms_element_positions_ordered rest = true ->
  forall e', In e' rest -> ms_element_max_A e < ms_element_min_A e'.
Proof.
  intros e rest. revert e.
  induction rest as [| e2 rest' IH]; intros e Hmono Hord e' Hin.
  - destruct Hin.
  - simpl in Hord. apply andb_prop in Hord as [He2_ord Hrest_ord].
    destruct Hin as [Heq | Hin'].
    + subst e'. exact (ms_monotonic_head_lt_A e e2 rest' Hmono).
    + pose proof (ms_monotonic_head_lt_A e e2 rest' Hmono) as Hlt_e_e2.
      pose proof (ms_monotonic_rest e (e2 :: rest') Hmono) as Hrest.
      pose proof (IH e2 Hrest Hrest_ord e' Hin') as Hlt_e2_e'.
      pose proof (ms_ordered_min_le_max_A e2 He2_ord) as Hmin_le_max.
      lia.
Qed.

Lemma ms_monotonic_head_lt_all_B : forall e rest,
  ms_trace_monotonic_aux (e :: rest) = true ->
  forallb ms_element_positions_ordered rest = true ->
  forall e', In e' rest -> ms_element_max_B e < ms_element_min_B e'.
Proof.
  intros e rest. revert e.
  induction rest as [| e2 rest' IH]; intros e Hmono Hord e' Hin.
  - destruct Hin.
  - simpl in Hord. apply andb_prop in Hord as [He2_ord Hrest_ord].
    destruct Hin as [Heq | Hin'].
    + subst e'. exact (ms_monotonic_head_lt_B e e2 rest' Hmono).
    + pose proof (ms_monotonic_head_lt_B e e2 rest' Hmono) as Hlt_e_e2.
      pose proof (ms_monotonic_rest e (e2 :: rest') Hmono) as Hrest.
      pose proof (IH e2 Hrest Hrest_ord e' Hin') as Hlt_e2_e'.
      pose proof (ms_ordered_min_le_max_B e2 He2_ord) as Hmin_le_max.
      lia.
Qed.

(** ** Shifted Trace Validity Preservation *)

(** When positions are shifted uniformly, ordering and monotonicity are preserved. *)

(** Shifted element preserves ordering *)
Lemma ms_shift_element_ordered : forall da db e,
  da <= ms_element_min_A e ->
  db <= ms_element_min_B e ->
  ms_element_positions_ordered e = true ->
  ms_element_positions_ordered (ms_shift_element da db e) = true.
Proof.
  intros da db e Hda Hdb Hord.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *; try reflexivity.
  - apply Nat.ltb_lt in Hord. apply Nat.ltb_lt. lia.
  - apply Nat.ltb_lt in Hord. apply Nat.ltb_lt. lia.
  - apply andb_prop in Hord as [Hi Hj].
    apply Nat.ltb_lt in Hi. apply Nat.ltb_lt in Hj.
    apply andb_true_intro. split; apply Nat.ltb_lt; lia.
Qed.

(** Shifted trace preserves ordering for all elements *)
Lemma ms_shift_trace_ordered : forall da db T,
  (forall e, In e T -> da <= ms_element_min_A e) ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  forallb ms_element_positions_ordered T = true ->
  forallb ms_element_positions_ordered (ms_shift_trace da db T) = true.
Proof.
  intros da db T Hda Hdb Hord.
  induction T as [| e rest IH]; [reflexivity |].
  simpl in *. apply andb_prop in Hord as [He_ord Hrest_ord].
  apply andb_true_intro. split.
  - apply ms_shift_element_ordered; [apply Hda | apply Hdb |]; try (left; reflexivity).
    exact He_ord.
  - apply IH.
    + intros e' Hin. apply Hda. right. exact Hin.
    + intros e' Hin. apply Hdb. right. exact Hin.
    + exact Hrest_ord.
Qed.

(** Shifted element validity: positions remain in bounds.
    Requires strict inequality to ensure shifted positions are >= 1.
    Also requires ordering to ensure multi-position elements work correctly. *)
Lemma ms_shift_element_valid : forall da db e lenA lenB,
  da < ms_element_min_A e ->
  db < ms_element_min_B e ->
  ms_element_positions_ordered e = true ->
  ms_valid_element lenA lenB e = true ->
  ms_valid_element (lenA - da) (lenB - db) (ms_shift_element da db e) = true.
Proof.
  intros da db e lenA lenB Hda Hdb Hord Hvalid.
  unfold ms_valid_element in *.
  apply andb_prop in Hvalid as [HvalidA HvalidB].
  apply andb_true_intro. split.
  - (* A positions valid *)
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
    + (* MSMatch *)
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidA as [Hi _].
      apply andb_true_intro. split; [| reflexivity].
      unfold valid_position in *. apply andb_prop in Hi as [Hi_lo Hi_hi].
      apply Nat.leb_le in Hi_lo. apply Nat.leb_le in Hi_hi.
      apply andb_true_intro. split; apply Nat.leb_le; lia.
    + (* MSMerge2: ordered means i1 < i2 *)
      apply Nat.ltb_lt in Hord.
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidA as [Hi1 Hi2'].
      apply andb_prop in Hi2' as [Hi2 _].
      unfold valid_position in *.
      apply andb_prop in Hi1 as [Hi1_lo Hi1_hi].
      apply andb_prop in Hi2 as [Hi2_lo Hi2_hi].
      apply Nat.leb_le in Hi1_lo. apply Nat.leb_le in Hi1_hi.
      apply Nat.leb_le in Hi2_lo. apply Nat.leb_le in Hi2_hi.
      apply andb_true_intro. split.
      * apply andb_true_intro. split; apply Nat.leb_le; lia.
      * apply andb_true_intro. split.
        -- apply andb_true_intro. split; apply Nat.leb_le; lia.
        -- reflexivity.
    + (* MSSplit2 *)
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidA as [Hi _].
      unfold valid_position in *. apply andb_prop in Hi as [Hi_lo Hi_hi].
      apply Nat.leb_le in Hi_lo. apply Nat.leb_le in Hi_hi.
      apply andb_true_intro. split; [| reflexivity].
      apply andb_true_intro. split; apply Nat.leb_le; lia.
    + (* MSDouble: ordered means i1 < i2 and j1 < j2 *)
      apply andb_prop in Hord as [Hi_ord Hj_ord].
      apply Nat.ltb_lt in Hi_ord.
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidA as [Hi1 Hi2'].
      apply andb_prop in Hi2' as [Hi2 _].
      unfold valid_position in *.
      apply andb_prop in Hi1 as [Hi1_lo Hi1_hi].
      apply andb_prop in Hi2 as [Hi2_lo Hi2_hi].
      apply Nat.leb_le in Hi1_lo. apply Nat.leb_le in Hi1_hi.
      apply Nat.leb_le in Hi2_lo. apply Nat.leb_le in Hi2_hi.
      apply andb_true_intro. split.
      * apply andb_true_intro. split; apply Nat.leb_le; lia.
      * apply andb_true_intro. split.
        -- apply andb_true_intro. split; apply Nat.leb_le; lia.
        -- reflexivity.
  - (* B positions valid *)
    destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
    + (* MSMatch *)
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidB as [Hj _].
      apply andb_true_intro. split; [| reflexivity].
      unfold valid_position in *. apply andb_prop in Hj as [Hj_lo Hj_hi].
      apply Nat.leb_le in Hj_lo. apply Nat.leb_le in Hj_hi.
      apply andb_true_intro. split; apply Nat.leb_le; lia.
    + (* MSMerge2 *)
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidB as [Hj _].
      unfold valid_position in *. apply andb_prop in Hj as [Hj_lo Hj_hi].
      apply Nat.leb_le in Hj_lo. apply Nat.leb_le in Hj_hi.
      apply andb_true_intro. split; [| reflexivity].
      apply andb_true_intro. split; apply Nat.leb_le; lia.
    + (* MSSplit2: ordered means j1 < j2 *)
      apply Nat.ltb_lt in Hord.
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidB as [Hj1 Hj2'].
      apply andb_prop in Hj2' as [Hj2 _].
      unfold valid_position in *.
      apply andb_prop in Hj1 as [Hj1_lo Hj1_hi].
      apply andb_prop in Hj2 as [Hj2_lo Hj2_hi].
      apply Nat.leb_le in Hj1_lo. apply Nat.leb_le in Hj1_hi.
      apply Nat.leb_le in Hj2_lo. apply Nat.leb_le in Hj2_hi.
      apply andb_true_intro. split.
      * apply andb_true_intro. split; apply Nat.leb_le; lia.
      * apply andb_true_intro. split.
        -- apply andb_true_intro. split; apply Nat.leb_le; lia.
        -- reflexivity.
    + (* MSDouble: ordered means i1 < i2 and j1 < j2 *)
      apply andb_prop in Hord as [Hi_ord Hj_ord].
      apply Nat.ltb_lt in Hj_ord.
      unfold all_positions_valid in *. simpl in *.
      apply andb_prop in HvalidB as [Hj1 Hj2'].
      apply andb_prop in Hj2' as [Hj2 _].
      unfold valid_position in *.
      apply andb_prop in Hj1 as [Hj1_lo Hj1_hi].
      apply andb_prop in Hj2 as [Hj2_lo Hj2_hi].
      apply Nat.leb_le in Hj1_lo. apply Nat.leb_le in Hj1_hi.
      apply Nat.leb_le in Hj2_lo. apply Nat.leb_le in Hj2_hi.
      apply andb_true_intro. split.
      * apply andb_true_intro. split; apply Nat.leb_le; lia.
      * apply andb_true_intro. split.
        -- apply andb_true_intro. split; apply Nat.leb_le; lia.
        -- reflexivity.
Qed.

(** Shifted trace preserves element validity for all elements.
    Requires strict inequality to ensure shifted positions are >= 1.
    Also requires ordering for all elements. *)
Lemma ms_shift_trace_elems_valid : forall da db T lenA lenB,
  (forall e, In e T -> da < ms_element_min_A e) ->
  (forall e, In e T -> db < ms_element_min_B e) ->
  forallb ms_element_positions_ordered T = true ->
  forallb (ms_valid_element lenA lenB) T = true ->
  forallb (ms_valid_element (lenA - da) (lenB - db)) (ms_shift_trace da db T) = true.
Proof.
  intros da db T lenA lenB Hda Hdb Hord Hvalid.
  induction T as [| e rest IH]; [reflexivity |].
  simpl in *. apply andb_prop in Hvalid as [He_valid Hrest_valid].
  apply andb_prop in Hord as [He_ord Hrest_ord].
  apply andb_true_intro. split.
  - apply ms_shift_element_valid; try (apply Hda; left; reflexivity);
    try (apply Hdb; left; reflexivity); assumption.
  - apply IH.
    + intros e' Hin. apply Hda. right. exact Hin.
    + intros e' Hin. apply Hdb. right. exact Hin.
    + exact Hrest_ord.
    + exact Hrest_valid.
Qed.

(** Shifted trace preserves no-overlap on positions.
    The key insight is that subtracting a constant from all positions
    preserves distinctness when the constant <= all values. *)
Lemma ms_shift_trace_no_overlap_A_gen : forall da db T,
  forallb ms_element_positions_ordered T = true ->
  (forall e, In e T -> da <= ms_element_min_A e) ->
  ms_positions_no_overlap_A T = true ->
  ms_positions_no_overlap_A (ms_shift_trace da db T) = true.
Proof.
  intros da db T Hord Hda Hno.
  unfold ms_positions_no_overlap_A.
  rewrite ms_trace_positions_A_shift.
  apply NoDup_no_duplicate_positions.
  apply NoDup_map_injective.
  - apply no_duplicate_positions_NoDup. exact Hno.
  - (* Injectivity: (x - da) = (y - da) → x = y when da <= x, da <= y *)
    intros x y Hx Hy Heq.
    apply sub_injective_when_le with da; auto.
    + apply ms_trace_positions_A_ge_da with T; assumption.
    + apply ms_trace_positions_A_ge_da with T; assumption.
Qed.

Lemma ms_shift_trace_no_overlap_B_gen : forall da db T,
  forallb ms_element_positions_ordered T = true ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  ms_positions_no_overlap_B T = true ->
  ms_positions_no_overlap_B (ms_shift_trace da db T) = true.
Proof.
  intros da db T Hord Hdb Hno.
  unfold ms_positions_no_overlap_B.
  rewrite ms_trace_positions_B_shift.
  apply NoDup_no_duplicate_positions.
  apply NoDup_map_injective.
  - apply no_duplicate_positions_NoDup. exact Hno.
  - intros x y Hx Hy Heq.
    apply sub_injective_when_le with db; auto.
    + apply ms_trace_positions_B_ge_db with T; assumption.
    + apply ms_trace_positions_B_ge_db with T; assumption.
Qed.

(** Shifted trace preserves monotonicity.
    Requires shift <= min for all elements and ordering. *)
Lemma ms_shift_trace_monotonic : forall da db T,
  (forall e, In e T -> da <= ms_element_min_A e) ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  forallb ms_element_positions_ordered T = true ->
  ms_trace_monotonic_aux T = true ->
  ms_trace_monotonic_aux (ms_shift_trace da db T) = true.
Proof.
  intros da db T Hda Hdb Hord Hmono.
  induction T as [| e1 rest IH].
  - reflexivity.
  - destruct rest as [| e2 rest'].
    + simpl. reflexivity.
    + simpl in Hmono. apply andb_prop in Hmono as [Hmono' Hrest_mono].
      apply andb_prop in Hmono' as [HltA HltB].
      apply Nat.ltb_lt in HltA. apply Nat.ltb_lt in HltB.
      simpl in Hord. apply andb_prop in Hord as [He1_ord Hrest_ord].
      simpl. apply andb_true_intro. split.
      * apply andb_true_intro. split; apply Nat.ltb_lt.
        -- (* max_A e1 - da < min_A e2 - da when max_A e1 < min_A e2 and da <= min_A e1 *)
           rewrite ms_shift_element_max_A, ms_shift_element_min_A.
           assert (Hda1: da <= ms_element_min_A e1) by (apply Hda; left; reflexivity).
           assert (Hda2: da <= ms_element_min_A e2) by (apply Hda; right; left; reflexivity).
           pose proof (ms_ordered_min_le_max_A e1 He1_ord) as Hmin_le_max.
           (* For nat subtraction, need da <= max_A e1 *)
           assert (Hda_max: da <= ms_element_max_A e1).
           { apply Nat.le_trans with (ms_element_min_A e1); assumption. }
           lia.
        -- rewrite ms_shift_element_max_B, ms_shift_element_min_B.
           assert (Hdb1: db <= ms_element_min_B e1) by (apply Hdb; left; reflexivity).
           assert (Hdb2: db <= ms_element_min_B e2) by (apply Hdb; right; left; reflexivity).
           pose proof (ms_ordered_min_le_max_B e1 He1_ord) as Hmin_le_max.
           (* For nat subtraction, need db <= max_B e1 *)
           assert (Hdb_max: db <= ms_element_max_B e1).
           { apply Nat.le_trans with (ms_element_min_B e1); assumption. }
           lia.
      * apply IH.
        -- intros e Hin. apply Hda. right. exact Hin.
        -- intros e Hin. apply Hdb. right. exact Hin.
        -- exact Hrest_ord.
        -- exact Hrest_mono.
Qed.

(** No-overlap is preserved for the rest of a trace *)
Lemma ms_positions_no_overlap_A_rest : forall e rest,
  ms_positions_no_overlap_A (e :: rest) = true ->
  ms_positions_no_overlap_A rest = true.
Proof.
  intros e rest Hno.
  unfold ms_positions_no_overlap_A in *.
  unfold ms_trace_positions_A in Hno. simpl in Hno.
  unfold ms_trace_positions_A.
  (* Use fact that NoDup (l1 ++ l2) -> NoDup l2 *)
  apply no_duplicate_positions_NoDup in Hno.
  apply NoDup_no_duplicate_positions.
  apply NoDup_app_remove_l in Hno.
  exact Hno.
Qed.

Lemma ms_positions_no_overlap_B_rest : forall e rest,
  ms_positions_no_overlap_B (e :: rest) = true ->
  ms_positions_no_overlap_B rest = true.
Proof.
  intros e rest Hno.
  unfold ms_positions_no_overlap_B in *.
  unfold ms_trace_positions_B in Hno. simpl in Hno.
  unfold ms_trace_positions_B.
  apply no_duplicate_positions_NoDup in Hno.
  apply NoDup_no_duplicate_positions.
  apply NoDup_app_remove_l in Hno.
  exact Hno.
Qed.

(** Comprehensive shifted trace validity preservation.
    When we shift a valid trace by (da, db) where:
    - da <= min_A for all elements (so positions remain >= 1)
    - db <= min_B for all elements
    - Shifted positions fit in new string lengths

    The resulting shifted trace is valid for shorter strings. *)
Lemma ms_shift_trace_valid : forall da db (A B A' B' : list Char) T,
  length A' = length A - da ->
  length B' = length B - db ->
  forallb ms_element_positions_ordered T = true ->
  (forall e, In e T -> da <= ms_element_min_A e) ->
  (forall e, In e T -> db <= ms_element_min_B e) ->
  (forall e, In e T -> da < ms_element_min_A e) ->
  (forall e, In e T -> db < ms_element_min_B e) ->
  forallb (ms_valid_element (length A) (length B)) T = true ->
  ms_positions_no_overlap_A T = true ->
  ms_positions_no_overlap_B T = true ->
  ms_trace_monotonic_aux T = true ->
  ms_trace_valid A' B' (ms_shift_trace da db T) = true.
Proof.
  intros da db A B A' B' T HlenA' HlenB' Hord Hda_le Hdb_le Hda_lt Hdb_lt Hvalid Hno_A Hno_B Hmono.
  unfold ms_trace_valid.
  apply andb_true_intro. split; [|apply ms_shift_trace_monotonic; assumption].
  apply andb_true_intro. split; [|apply ms_shift_trace_no_overlap_B_gen with (da := da); assumption].
  apply andb_true_intro. split; [|apply ms_shift_trace_no_overlap_A_gen with (db := db); assumption].
  apply andb_true_intro. split.
  - (* Element validity *)
    rewrite HlenA', HlenB'.
    apply ms_shift_trace_elems_valid; assumption.
  - (* Ordered positions preserved *)
    apply ms_shift_trace_ordered; assumption.
Qed.

(** ** Phase 3: Main Construction *)

(** Recursively build operation sequence from monotonic trace.

    Invariant: At each step, the trace T is valid for strings A and B,
    meaning all positions in T are within bounds of A and B.

    For termination, we track the total remaining length |A| + |B|.
*)
Fixpoint trace_to_seq_aux (fuel : nat) (A B : list Char) (T : MSTrace) : list ms_op :=
  match fuel with
  | 0 => delete_ops A ++ insert_ops B  (* Fallback: shouldn't happen with valid trace *)
  | S fuel' =>
      match T with
      | [] => delete_ops A ++ insert_ops B  (* Clean up remaining characters *)
      | e :: rest =>
          let min_a := ms_element_min_A e in
          let min_b := ms_element_min_B e in
          let prefix_a := min_a - 1 in  (* Uncovered positions before element in A *)
          let prefix_b := min_b - 1 in  (* Uncovered positions before element in B *)
          let A' := skipn prefix_a A in  (* A after consuming prefix *)
          let B' := skipn prefix_b B in  (* B after consuming prefix *)
          let span_a := ms_element_A_span e in
          let span_b := ms_element_B_span e in
          let A'' := skipn span_a A' in  (* A after element *)
          let B'' := skipn span_b B' in  (* B after element *)
          let shift_a := prefix_a + span_a in
          let shift_b := prefix_b + span_b in
          (* Consume uncovered prefix positions *)
          consume_A_prefix A prefix_a ++
          consume_B_prefix B prefix_b ++
          (* Apply element operation (positions shifted to front) *)
          ms_element_to_front_op A' B' (ms_shift_element prefix_a prefix_b e) ++
          (* Recurse on remaining *)
          trace_to_seq_aux fuel' A'' B'' (ms_shift_trace shift_a shift_b rest)
      end
  end.

(** Wrapper that provides sufficient fuel *)
Definition trace_to_seq (A B : list Char) (T : MSTrace) : list ms_op :=
  trace_to_seq_aux (length A + length B + 1) A B T.

(** ** Phase 4: Validity Lemmas *)

(** Helper: consume_A_prefix validity *)
Lemma consume_A_prefix_valid : forall A B n,
  n <= length A ->
  apply_ms_seq (consume_A_prefix A n) A B = Some (skipn n A, B).
Proof.
  intros A B n Hlen.
  unfold consume_A_prefix.
  induction n as [| n' IH] in A, Hlen |- *.
  - simpl. reflexivity.
  - destruct A as [| a A'].
    + simpl in Hlen. lia.
    + simpl. rewrite char_eq_refl.
      rewrite IH by (simpl in Hlen; lia).
      reflexivity.
Qed.

(** Helper: consume_B_prefix validity *)
Lemma consume_B_prefix_valid : forall A B n,
  n <= length B ->
  apply_ms_seq (consume_B_prefix B n) A B = Some (A, skipn n B).
Proof.
  intros A B n Hlen.
  unfold consume_B_prefix.
  induction n as [| n' IH] in B, Hlen |- *.
  - simpl. reflexivity.
  - destruct B as [| b B'].
    + simpl in Hlen. lia.
    + simpl. rewrite char_eq_refl.
      rewrite IH by (simpl in Hlen; lia).
      reflexivity.
Qed.

(** Helper: element operation validity for front-positioned elements

    NOTE: This lemma requires that after shifting, the element's positions
    are at the front of the strings (1 or 1,2). For non-consecutive positions
    (e.g., MSMerge2 1 3 1), additional interleaved deletions would be needed.

    The full proof requires handling all position patterns, which involves
    careful case analysis and position tracking. We admit this technical detail
    while documenting that the semantic correctness holds: any valid trace
    defines a valid alignment, and aligning characters at specified positions
    can be achieved through an appropriate sequence of operations.
*)
Lemma ms_element_front_op_valid : forall A B e,
  ms_shift_element (ms_element_min_A e - 1) (ms_element_min_B e - 1) e = e ->
  ms_element_min_A e = 1 ->
  ms_element_min_B e = 1 ->
  ms_element_A_span e <= length A ->
  ms_element_B_span e <= length B ->
  ms_element_positions_ordered e = true ->
  ms_element_positions_consecutive e = true ->
  apply_ms_seq (ms_element_to_front_op A B e) A B =
    Some (skipn (ms_element_A_span e) A, skipn (ms_element_B_span e) B).
Proof.
  intros A B e Hshift Hmin_a Hmin_b Hspan_a Hspan_b Hord Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2];
    simpl in Hmin_a, Hmin_b, Hord, Hcons.
  - (* MSMatch i j *)
    subst i j.
    destruct A as [| a A']; [simpl in Hspan_a; lia|].
    destruct B as [| b B']; [simpl in Hspan_b; lia|].
    simpl. rewrite !char_eq_refl. reflexivity.
  - (* MSMerge2 i1 i2 j *)
    (* From hypotheses: i1 = 1, j = 1, i2 = i1 + 1 = 2 *)
    subst i1 j.
    apply Nat.eqb_eq in Hcons. simpl in Hcons. subst i2.
    (* Now element is MSMerge2 1 2 1 *)
    simpl ms_element_A_span. simpl ms_element_B_span.
    (* Need: A has at least 2 chars, B has at least 1 char *)
    destruct A as [| a1 A']; [simpl in Hspan_a; lia|].
    destruct A' as [| a2 A'']; [simpl in Hspan_a; lia|].
    destruct B as [| b B']; [simpl in Hspan_b; lia|].
    simpl. rewrite !char_eq_refl. reflexivity.
  - (* MSSplit2 i j1 j2 *)
    (* From hypotheses: i = 1, j1 = 1, j2 = j1 + 1 = 2 *)
    subst i j1.
    apply Nat.eqb_eq in Hcons. simpl in Hcons. subst j2.
    (* Now element is MSSplit2 1 1 2 *)
    simpl ms_element_A_span. simpl ms_element_B_span.
    destruct A as [| a A']; [simpl in Hspan_a; lia|].
    destruct B as [| b1 B']; [simpl in Hspan_b; lia|].
    destruct B' as [| b2 B'']; [simpl in Hspan_b; lia|].
    simpl. rewrite !char_eq_refl. reflexivity.
  - (* MSDouble i1 i2 j1 j2 *)
    (* From hypotheses: i1 = 1, j1 = 1, i2 = 2, j2 = 2 *)
    subst i1 j1.
    apply andb_prop in Hcons. destruct Hcons as [Hi2 Hj2].
    apply Nat.eqb_eq in Hi2. apply Nat.eqb_eq in Hj2.
    simpl in Hi2, Hj2. subst i2 j2.
    (* Now element is MSDouble 1 2 1 2 *)
    simpl ms_element_A_span. simpl ms_element_B_span.
    destruct A as [| a1 A']; [simpl in Hspan_a; lia|].
    destruct A' as [| a2 A'']; [simpl in Hspan_a; lia|].
    destruct B as [| b1 B']; [simpl in Hspan_b; lia|].
    destruct B' as [| b2 B'']; [simpl in Hspan_b; lia|].
    (* ms_element_to_front_op generates [MSSubst a1 b1; MSSubst a2 b2]
       apply_ms_seq applies both, need to show result is (A'', B'') *)
    unfold ms_element_to_front_op. simpl.
    rewrite char_eq_refl. simpl.
    rewrite char_eq_refl. simpl.
    rewrite char_eq_refl. simpl.
    rewrite char_eq_refl. reflexivity.
Qed.

(** ** Phase 5: Cost Lemmas *)

(** Helper: consume_A_prefix cost *)
Lemma consume_A_prefix_cost : forall A n,
  n <= length A ->
  ms_seq_cost (consume_A_prefix A n) = n.
Proof.
  intros A n Hlen.
  unfold consume_A_prefix.
  rewrite delete_ops_cost.
  rewrite firstn_length.
  lia.
Qed.

(** Helper: consume_B_prefix cost *)
Lemma consume_B_prefix_cost : forall B n,
  n <= length B ->
  ms_seq_cost (consume_B_prefix B n) = n.
Proof.
  intros B n Hlen.
  unfold consume_B_prefix.
  rewrite insert_ops_cost.
  rewrite firstn_length.
  lia.
Qed.

(** Key lemma: Element cost is preserved when shifting both the element
    positions and the strings by the same amount.

    If A' = skipn da A and B' = skipn db B, then
    ms_element_cost A' B' (ms_shift_element da db e) = ms_element_cost A B e

    This works because:
    - Shifted element has positions (i - da, j - db) where i > da, j > db
    - Looking up position (i - da) in A' = skipn da A gives the same character
      as looking up position i in A (since nth (k-1) (skipn da xs) = nth (k+da-1) xs)

    NOTE: Requires STRICT inequality (da < i) for the nat arithmetic to work out.
    This is satisfied in practice because we shift by max_A e and rest has min_A > max_A e.
*)
Lemma ms_element_cost_shift : forall A B da db e,
  da < ms_element_min_A e ->
  db < ms_element_min_B e ->
  ms_element_positions_consecutive e = true ->
  ms_element_cost (skipn da A) (skipn db B) (ms_shift_element da db e) =
  ms_element_cost A B e.
Proof.
  intros A B da db e Hda Hdb Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - (* MSMatch i j: subst_cost is an if-expression *)
    unfold subst_cost.
    (* Replace each nth expression with its original form *)
    replace (nth (i - da - 1) (skipn da A) default_char) with (nth (i - 1) A default_char)
      by (rewrite nth_skipn; f_equal; lia).
    replace (nth (j - db - 1) (skipn db B) default_char) with (nth (j - 1) B default_char)
      by (rewrite nth_skipn; f_equal; lia).
    reflexivity.
  - (* MSMerge2 i1 i2 j: Hda is da < i1, Hcons gives i2 = i1 + 1 *)
    unfold merge_cost.
    (* Extract i2 = i1 + 1 from consecutive constraint *)
    apply Nat.eqb_eq in Hcons.
    (* Now Hcons : i2 = i1 + 1, and Hda : da < i1 implies da < i2 *)
    (* Each nth expression is equal after shifting *)
    replace (nth (i1 - da - 1) (skipn da A) default_char) with (nth (i1 - 1) A default_char)
      by (rewrite nth_skipn; f_equal; lia).
    replace (nth (i2 - da - 1) (skipn da A) default_char) with (nth (i2 - 1) A default_char)
      by (rewrite nth_skipn; f_equal; lia).
    replace (nth (j - db - 1) (skipn db B) default_char) with (nth (j - 1) B default_char)
      by (rewrite nth_skipn; f_equal; lia).
    reflexivity.
  - (* MSSplit2 i j1 j2: Hdb is db < j1, Hcons gives j2 = j1 + 1 *)
    unfold split_cost.
    (* Extract j2 = j1 + 1 from consecutive constraint *)
    apply Nat.eqb_eq in Hcons.
    (* Now Hcons : j2 = j1 + 1, and Hdb : db < j1 implies db < j2 *)
    (* Each nth expression is equal after shifting *)
    replace (nth (i - da - 1) (skipn da A) default_char) with (nth (i - 1) A default_char)
      by (rewrite nth_skipn; f_equal; lia).
    replace (nth (j1 - db - 1) (skipn db B) default_char) with (nth (j1 - 1) B default_char)
      by (rewrite nth_skipn; f_equal; lia).
    replace (nth (j2 - db - 1) (skipn db B) default_char) with (nth (j2 - 1) B default_char)
      by (rewrite nth_skipn; f_equal; lia).
    reflexivity.
  - (* MSDouble i1 i2 j1 j2: Hda is da < i1, Hdb is db < j1,
       Hcons gives (i2 =? i1 + 1) && (j2 =? j1 + 1) = true *)
    (* Extract consecutive relations from Hcons *)
    apply andb_prop in Hcons. destruct Hcons as [Hcons_a Hcons_b].
    apply Nat.eqb_eq in Hcons_a. (* i2 = i1 + 1 *)
    apply Nat.eqb_eq in Hcons_b. (* j2 = j1 + 1 *)
    (* Now with da < i1 and i2 = i1 + 1, we have da < i2
       Similarly with db < j1 and j2 = j1 + 1, we have db < j2 *)
    f_equal.
    + (* First subst_cost *)
      unfold subst_cost.
      replace (nth (i1 - da - 1) (skipn da A) default_char) with (nth (i1 - 1) A default_char)
        by (rewrite nth_skipn; f_equal; lia).
      replace (nth (j1 - db - 1) (skipn db B) default_char) with (nth (j1 - 1) B default_char)
        by (rewrite nth_skipn; f_equal; lia).
      reflexivity.
    + (* Second subst_cost *)
      unfold subst_cost.
      replace (nth (i2 - da - 1) (skipn da A) default_char) with (nth (i2 - 1) A default_char)
        by (rewrite nth_skipn; f_equal; lia).
      replace (nth (j2 - db - 1) (skipn db B) default_char) with (nth (j2 - 1) B default_char)
        by (rewrite nth_skipn; f_equal; lia).
      reflexivity.
Qed.

(** Trace change cost is preserved under coordinated shifting *)
Lemma ms_trace_change_cost_shift : forall A B da db T,
  (forall e, In e T -> da < ms_element_min_A e) ->
  (forall e, In e T -> db < ms_element_min_B e) ->
  ms_trace_positions_consecutive T = true ->
  ms_trace_change_cost (skipn da A) (skipn db B) (ms_shift_trace da db T) =
  ms_trace_change_cost A B T.
Proof.
  intros A B da db T Hda Hdb Hcons.
  induction T as [| e rest IH].
  - reflexivity.
  - simpl ms_shift_trace.
    rewrite !ms_trace_change_cost_cons.
    (* Extract consecutive constraint for head element *)
    unfold ms_trace_positions_consecutive in Hcons.
    rewrite forallb_forall in Hcons.
    assert (Hcons_e: ms_element_positions_consecutive e = true)
      by (apply Hcons; left; reflexivity).
    assert (Hcons_rest: ms_trace_positions_consecutive rest = true).
    { unfold ms_trace_positions_consecutive.
      rewrite forallb_forall.
      intros e' Hin'. apply Hcons. right. exact Hin'. }
    rewrite ms_element_cost_shift.
    + rewrite IH.
      * reflexivity.
      * intros e' Hin'. apply Hda. right. exact Hin'.
      * intros e' Hin'. apply Hdb. right. exact Hin'.
      * exact Hcons_rest.
    + apply Hda. left. reflexivity.
    + apply Hdb. left. reflexivity.
    + exact Hcons_e.
Qed.

(** Helper: element front op cost equals element cost

    NOTE: For elements at front positions (1 or 1,2), the cost of the
    generated operations equals the element cost. For non-consecutive
    positions, additional delete/insert operations would be needed.
*)
Lemma ms_element_front_op_cost : forall A B e,
  ms_element_min_A e = 1 ->
  ms_element_min_B e = 1 ->
  ms_element_positions_ordered e = true ->
  ms_element_positions_consecutive e = true ->
  ms_seq_cost (ms_element_to_front_op A B e) = ms_element_cost A B e.
Proof.
  intros A B e Hmin_a Hmin_b Hord Hcons.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2];
    simpl in Hmin_a, Hmin_b, Hord, Hcons.
  - (* MSMatch i j *)
    subst i j. simpl.
    rewrite Nat.add_0_r.
    unfold hd. destruct A; destruct B; reflexivity.
  - (* MSMerge2 i1 i2 j *)
    subst i1 j.
    apply Nat.eqb_eq in Hcons. simpl in Hcons. subst i2.
    (* Element is MSMerge2 1 2 1 *)
    simpl.
    (* ms_seq_cost [MSMerge a b c] = merge_cost a b c + 0 *)
    rewrite Nat.add_0_r.
    (* Need: merge_cost (nth 0 A d) (nth 1 A d) (hd d B) = merge_cost (nth 0 A d) (nth 1 A d) (nth 0 B d) *)
    (* hd d B = nth 0 B d by definition *)
    unfold hd. destruct B; reflexivity.
  - (* MSSplit2 i j1 j2 *)
    subst i j1.
    apply Nat.eqb_eq in Hcons. simpl in Hcons. subst j2.
    (* Element is MSSplit2 1 1 2 *)
    simpl.
    rewrite Nat.add_0_r.
    (* Need: split_cost (hd d A) (nth 0 B d) (nth 1 B d) = split_cost (nth 0 A d) (nth 0 B d) (nth 1 B d) *)
    unfold hd. destruct A; reflexivity.
  - (* MSDouble i1 i2 j1 j2 *)
    subst i1 j1.
    apply andb_prop in Hcons. destruct Hcons as [Hi2 Hj2].
    apply Nat.eqb_eq in Hi2. apply Nat.eqb_eq in Hj2.
    simpl in Hi2, Hj2. subst i2 j2.
    (* Element is MSDouble 1 2 1 2 *)
    simpl.
    (* ms_seq_cost [MSSubst a b; MSSubst c d] = subst_cost a b + subst_cost c d + 0 *)
    rewrite Nat.add_0_r.
    reflexivity.
Qed.

(** ** Main Validity Theorem for trace_to_seq *)

(** Key lemma: trace_to_seq produces a valid sequence that transforms A to B.
    Requires consecutive positions for elements to ensure the operation
    sequence correctly transforms the strings. *)
Lemma trace_to_seq_aux_valid : forall fuel A B T,
  length A + length B < fuel ->
  ms_trace_valid A B T = true ->
  ms_trace_positions_consecutive T = true ->
  ms_seq_valid (trace_to_seq_aux fuel A B T) A B.
Proof.
  intros fuel.
  induction fuel as [| fuel' IH]; intros A B T Hfuel Hvalid Hcons.
  - (* fuel = 0: impossible by Hfuel *)
    lia.
  - (* fuel = S fuel' *)
    destruct T as [| e rest].
    + (* T = []: delete all A, insert all B *)
      unfold ms_seq_valid. simpl.
      (* Apply delete_ops A then insert_ops B *)
      assert (Hdel: apply_ms_seq (delete_ops A) A B = Some ([], B)).
      { clear. induction A as [| a A' IH']; simpl.
        - reflexivity.
        - rewrite char_eq_refl. exact IH'. }
      rewrite apply_ms_seq_app with (src' := []) (tgt' := B) by exact Hdel.
      apply insert_ops_valid.
    + (* T = e :: rest *)
      (* Extract validity components *)
      unfold ms_trace_valid in Hvalid.
      apply andb_prop in Hvalid as [Hvalid' Hmono].
      apply andb_prop in Hvalid' as [Hvalid'' Hno_B].
      apply andb_prop in Hvalid'' as [Hvalid''' Hno_A].
      apply andb_prop in Hvalid''' as [Helems Hord_elems].
      simpl in Helems. apply andb_prop in Helems as [He_valid Hrest_valid].
      simpl in Hord_elems. apply andb_prop in Hord_elems as [He_ord Hrest_ord].

      (* Extract consecutive property *)
      unfold ms_trace_positions_consecutive in Hcons.
      simpl in Hcons. apply andb_prop in Hcons as [He_cons Hrest_cons].

      (* Get bounds from element validity *)
      unfold ms_valid_element in He_valid.
      apply andb_prop in He_valid as [HvalidA HvalidB].

      (* Get minimum positions *)
      assert (Hmin_a_pos: 1 <= ms_element_min_A e /\ ms_element_min_A e <= length A).
      { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos. }
      assert (Hmin_b_pos: 1 <= ms_element_min_B e /\ ms_element_min_B e <= length B).
      { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos. }

      (* Establish bounds for prefix consumption *)
      assert (Hprefix_a: ms_element_min_A e - 1 <= length A) by lia.
      assert (Hprefix_b: ms_element_min_B e - 1 <= length B) by lia.

      (* Show validity step by step *)
      simpl trace_to_seq_aux.
      unfold ms_seq_valid.

      (* Step 1: Apply consume_A_prefix *)
      assert (Hstep1: apply_ms_seq (consume_A_prefix A (ms_element_min_A e - 1)) A B =
                      Some (skipn (ms_element_min_A e - 1) A, B)).
      { apply consume_A_prefix_valid. exact Hprefix_a. }
      rewrite apply_ms_seq_app with
        (src' := skipn (ms_element_min_A e - 1) A) (tgt' := B) by exact Hstep1.

      (* Step 2: Apply consume_B_prefix *)
      assert (Hprefix_b': ms_element_min_B e - 1 <= length B) by lia.
      assert (Hstep2: apply_ms_seq (consume_B_prefix B (ms_element_min_B e - 1))
                        (skipn (ms_element_min_A e - 1) A) B =
                      Some (skipn (ms_element_min_A e - 1) A, skipn (ms_element_min_B e - 1) B)).
      { apply consume_B_prefix_valid. exact Hprefix_b'. }
      rewrite apply_ms_seq_app with
        (src' := skipn (ms_element_min_A e - 1) A)
        (tgt' := skipn (ms_element_min_B e - 1) B) by exact Hstep2.

      (* Step 3: Apply element operation using ms_element_front_op_valid
         The shifted element has:
         - min_A = 1 (by ms_shift_element_min_A_eq_1)
         - min_B = 1 (by ms_shift_element_min_B_eq_1)
         - consecutive positions (by ms_shift_element_preserves_consecutive)
         - ordered positions (by ms_shift_element_preserves_ordered)

         Step 4: Apply IH to recursion on rest
         Requires showing:
         - fuel decreases (by structure)
         - remaining strings are smaller
         - rest maintains validity (from original validity)
         - rest maintains consecutive positions (from Hrest_cons)

         The full proof requires tracking position invariants through the recursion.
         This is semantically correct but mechanically complex. *)

      (* Set up names for shifted element and remaining strings *)
      set (prefix_a := ms_element_min_A e - 1) in *.
      set (prefix_b := ms_element_min_B e - 1) in *.
      set (span_a := ms_element_A_span e) in *.
      set (span_b := ms_element_B_span e) in *.
      set (A' := skipn prefix_a A).
      set (B' := skipn prefix_b B).
      set (A'' := skipn span_a A').
      set (B'' := skipn span_b B').
      set (shifted_e := ms_shift_element prefix_a prefix_b e).

      (* Get max bounds from element validity *)
      (* HvalidA and HvalidB were extracted earlier from He_valid *)
      (* Reconstruct ms_valid_element for the max bound lemmas *)
      assert (He_elem_valid: ms_valid_element (length A) (length B) e = true).
      { unfold ms_valid_element. apply andb_true_intro. split; assumption. }
      pose proof (ms_valid_element_max_A_bound (length A) (length B) e He_elem_valid) as Hmax_a.
      pose proof (ms_valid_element_max_B_bound (length A) (length B) e He_elem_valid) as Hmax_b.

      (* Show shifted element properties *)
      assert (Hshifted_min_a: ms_element_min_A shifted_e = 1).
      { apply ms_shift_element_min_A_eq_1; unfold prefix_a; lia. }
      assert (Hshifted_min_b: ms_element_min_B shifted_e = 1).
      { apply ms_shift_element_min_B_eq_1; unfold prefix_b; lia. }
      assert (Hshifted_cons: ms_element_positions_consecutive shifted_e = true).
      { apply ms_shift_element_preserves_consecutive; unfold prefix_a, prefix_b; try lia; exact He_cons. }
      assert (Hshifted_ord: ms_element_positions_ordered shifted_e = true).
      { apply ms_shift_element_preserves_ordered; unfold prefix_a, prefix_b; try lia; exact He_ord. }

      (* Show spans fit in remaining strings *)
      assert (Hlen_A': length A' = length A - prefix_a).
      { unfold A', prefix_a. rewrite length_skipn. lia. }
      assert (Hlen_B': length B' = length B - prefix_b).
      { unfold B', prefix_b. rewrite length_skipn. lia. }
      assert (Hspan_a_fits: span_a <= length A').
      { unfold span_a. rewrite Hlen_A'. unfold prefix_a.
        apply ms_element_span_A_fits; [exact He_cons | lia | exact Hmax_a]. }
      assert (Hspan_b_fits: span_b <= length B').
      { unfold span_b. rewrite Hlen_B'. unfold prefix_b.
        apply ms_element_span_B_fits; [exact He_cons | lia | exact Hmax_b]. }

      (* Shifted element has spans fitting (spans preserved under shift) *)
      assert (Hshifted_span_a: ms_element_A_span shifted_e <= length A').
      { unfold shifted_e. rewrite ms_element_A_span_shift. exact Hspan_a_fits. }
      assert (Hshifted_span_b: ms_element_B_span shifted_e <= length B').
      { unfold shifted_e. rewrite ms_element_B_span_shift. exact Hspan_b_fits. }

      (* Show shifted element has shift-by-0 property (since min positions are 1) *)
      assert (Hshift_zero: ms_shift_element (ms_element_min_A shifted_e - 1)
                                            (ms_element_min_B shifted_e - 1) shifted_e = shifted_e).
      { rewrite Hshifted_min_a, Hshifted_min_b. simpl. apply ms_shift_element_zero. }

      (* Step 3: Apply element operation *)
      assert (Hstep3: apply_ms_seq (ms_element_to_front_op A' B' shifted_e) A' B' =
                      Some (A'', B'')).
      { unfold A'', B'.
        assert (Hgoal: apply_ms_seq (ms_element_to_front_op A' B' shifted_e) A' B' =
                       Some (skipn (ms_element_A_span shifted_e) A',
                             skipn (ms_element_B_span shifted_e) B')).
        { apply ms_element_front_op_valid; assumption. }
        unfold shifted_e in Hgoal.
        rewrite ms_element_A_span_shift, ms_element_B_span_shift in Hgoal.
        fold span_a span_b A' in Hgoal.
        exact Hgoal. }
      rewrite apply_ms_seq_app with (src' := A'') (tgt' := B'') by exact Hstep3.

      (* Step 4: Apply IH to recursion *)
      (* The recursive call uses shifted trace on remaining strings.
         The validity proof for the shifted trace is complex and requires
         showing that monotonicity ensures positions remain in bounds.
         We admit this technical detail while documenting the semantic correctness. *)

      (* Show fuel decreases *)
      assert (Hfuel_dec: length A'' + length B'' < fuel').
      { (* Spans are positive *)
        assert (Hspan_a_pos: 1 <= span_a).
        { unfold span_a. destruct e; simpl; lia. }
        assert (Hspan_b_pos: 1 <= span_b).
        { unfold span_b. destruct e; simpl; lia. }
        (* Show lengths decrease *)
        assert (Hlen_A'': length A'' <= length A' - 1).
        { unfold A''. rewrite length_skipn. lia. }
        assert (Hlen_B'': length B'' <= length B' - 1).
        { unfold B''. rewrite length_skipn. lia. }
        (* Use bounds from Hlen_A', Hlen_B' *)
        rewrite Hlen_A' in Hlen_A''.
        rewrite Hlen_B' in Hlen_B''.
        lia. }

      (* Apply IH - validity of shifted trace is the complex part *)
      apply IH.
      * exact Hfuel_dec.
      * (* ms_trace_valid A'' B'' (shifted rest) - requires monotonicity analysis *)
        (* Use ms_shift_trace_valid with shift = (max_A e, max_B e) *)
        pose proof (prefix_span_eq_max_A e He_cons (proj1 Hmin_a_pos)) as Hshift_a.
        pose proof (prefix_span_eq_max_B e He_cons (proj1 Hmin_b_pos)) as Hshift_b.
        fold prefix_a span_a in Hshift_a.
        fold prefix_b span_b in Hshift_b.
        (* Extract rest validity components *)
        pose proof (ms_positions_no_overlap_A_rest e rest Hno_A) as Hrest_no_A.
        pose proof (ms_positions_no_overlap_B_rest e rest Hno_B) as Hrest_no_B.
        pose proof (ms_monotonic_rest e rest Hmono) as Hrest_mono.
        (* Compute lengths of A'' and B'' *)
        assert (Hlen_A'': length A'' = length A - (prefix_a + span_a)).
        { unfold A'', A', prefix_a, span_a.
          rewrite length_skipn, length_skipn.
          (* Need to show prefix_a <= length A and span_a <= length A' *)
          assert (Hpref_a_bound: ms_element_min_A e - 1 <= length A) by lia.
          assert (HlenA'_eq: length (skipn (ms_element_min_A e - 1) A) = length A - (ms_element_min_A e - 1)).
          { rewrite length_skipn. lia. }
          (* span_a = max_A - min_A + 1 for consecutive elements *)
          rewrite (ms_element_A_span_eq_diff e He_cons).
          assert (Hspan_bound: ms_element_max_A e - ms_element_min_A e + 1 <=
                              length A - (ms_element_min_A e - 1)).
          { (* max_A e <= length A from element validity *)
            pose proof (ms_valid_element_max_A_bound (length A) (length B) e He_elem_valid) as Hmax_bound.
            lia. }
          lia. }
        assert (Hlen_B'': length B'' = length B - (prefix_b + span_b)).
        { unfold B'', B', prefix_b, span_b.
          rewrite length_skipn, length_skipn.
          assert (Hpref_b_bound: ms_element_min_B e - 1 <= length B) by lia.
          rewrite (ms_element_B_span_eq_diff e He_cons).
          pose proof (ms_valid_element_max_B_bound (length A) (length B) e He_elem_valid) as Hmax_bound.
          lia. }
        (* Apply ms_shift_trace_valid - shift by max positions of e *)
        rewrite Hshift_a in Hlen_A''.
        rewrite Hshift_b in Hlen_B''.
        (* Rewrite goal to use max positions *)
        rewrite Hshift_a, Hshift_b.
        apply (ms_shift_trace_valid (ms_element_max_A e) (ms_element_max_B e) A B A'' B'' rest).
        -- (* length A'' = length A - max_A e *)
           exact Hlen_A''.
        -- (* length B'' = length B - max_B e *)
           exact Hlen_B''.
        -- (* forallb ms_element_positions_ordered rest = true *)
           exact Hrest_ord.
        -- (* forall e', In e' rest -> max_A e <= ms_element_min_A e' *)
           intros e' Hin'.
           pose proof (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin') as Hlt.
           lia.
        -- (* forall e', In e' rest -> max_B e <= ms_element_min_B e' *)
           intros e' Hin'.
           pose proof (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin') as Hlt.
           lia.
        -- (* forall e', In e' rest -> max_A e < ms_element_min_A e' *)
           intros e' Hin'.
           exact (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin').
        -- (* forall e', In e' rest -> max_B e < ms_element_min_B e' *)
           intros e' Hin'.
           exact (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin').
        -- (* forallb (ms_valid_element (length A) (length B)) rest = true *)
           exact Hrest_valid.
        -- (* ms_positions_no_overlap_A rest = true *)
           exact Hrest_no_A.
        -- (* ms_positions_no_overlap_B rest = true *)
           exact Hrest_no_B.
        -- (* ms_trace_monotonic_aux rest = true *)
           exact Hrest_mono.
      * (* Consecutive positions preserved under shifting *)
        apply ms_shift_trace_preserves_consecutive.
        { (* Shift bounds for A positions *)
          intros e' Hin'.
          (* Show prefix_a + span_a = max_A e using consecutive property *)
          pose proof (prefix_span_eq_max_A e He_cons (proj1 Hmin_a_pos)) as Heq_max_a.
          unfold prefix_a, span_a in Heq_max_a.
          (* Monotonicity: max_A e < min_A e' for all e' in rest *)
          pose proof (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin') as Hlt.
          lia. }
        { (* Shift bounds for B positions *)
          intros e' Hin'.
          pose proof (prefix_span_eq_max_B e He_cons (proj1 Hmin_b_pos)) as Heq_max_b.
          unfold prefix_b, span_b in Heq_max_b.
          pose proof (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin') as Hlt.
          lia. }
        { (* Rest has consecutive positions *)
          exact Hrest_cons. }
Qed.

(** Arithmetic helper lemma for cost bound proof.
    This is extracted as a standalone lemma to help lia with a clean context.

    The goal has the form:
      (mina - 1) + (minb - 1) + elemCost + rec <=
        elemCost + change + (la - (pa + ra)) + (lb - (pb + rb))

    This reduces to (after canceling elemCost):
      (mina - 1) + (minb - 1) + rec <= change + (la - pa - ra) + (lb - pb - rb)

    From the IH we have:
      rec <= change + (la - ma - ra) + (lb - mb - rb)

    And from consecutive positions:
      ma = (mina - 1) + pa (so la - ma = la - mina + 1 - pa)
*)
Lemma cost_bound_arith : forall rec change elemCost la lb ma mb pa pb ra rb mina minb,
  rec <= change + (la - ma - ra) + (lb - mb - rb) ->
  ma = mina - 1 + pa ->
  mb = minb - 1 + pb ->
  ma <= la -> mb <= lb -> pa <= ma -> pb <= mb ->
  1 <= mina -> 1 <= minb ->
  ra <= la - ma -> rb <= lb - mb ->
  mina - 1 + (minb - 1 + (elemCost + rec)) <=
    elemCost + change + (la - (pa + ra)) + (lb - (pb + rb)).
Proof.
  intros rec change elemCost la lb ma mb pa pb ra rb mina minb
         Hrec Hma Hmb Hma_le Hmb_le Hpa Hpb Hmina Hminb Hra Hrb.
  (* Substitute to eliminate ma and mb *)
  subst ma mb.
  (* Now lia should be faster with explicit values *)
  lia.
Qed.

(** Helper: positions_A count is bounded by max_A for consecutive elements with min_A >= 1 *)
Lemma pos_le_max_A : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_A e ->
  length (ms_element_positions_A e) <= ms_element_max_A e.
Proof.
  intros e Hcons Hmin.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - lia.
  - apply Nat.eqb_eq in Hcons. lia.
  - lia.
  - apply andb_prop in Hcons. destruct Hcons as [Ha _].
    apply Nat.eqb_eq in Ha. lia.
Qed.

(** Helper: positions_B count is bounded by max_B for consecutive elements with min_B >= 1 *)
Lemma pos_le_max_B : forall e,
  ms_element_positions_consecutive e = true ->
  1 <= ms_element_min_B e ->
  length (ms_element_positions_B e) <= ms_element_max_B e.
Proof.
  intros e Hcons Hmin.
  destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
  - lia.
  - lia.
  - apply Nat.eqb_eq in Hcons. lia.
  - apply andb_prop in Hcons. destruct Hcons as [_ Hb].
    apply Nat.eqb_eq in Hb. lia.
Qed.

(** Cost bound for trace_to_seq_aux *)
Lemma trace_to_seq_aux_cost_bound : forall fuel A B T,
  length A + length B < fuel ->
  ms_trace_valid A B T = true ->
  ms_trace_positions_consecutive T = true ->
  ms_seq_cost (trace_to_seq_aux fuel A B T) <= ms_trace_cost A B T.
Proof.
  intros fuel.
  induction fuel as [| fuel' IH]; intros A B T Hfuel Hvalid Hcons.
  - lia.
  - destruct T as [| e rest].
    + (* T = []: sequence cost = |A| + |B| = trace_cost *)
      simpl.
      rewrite ms_seq_cost_app.
      rewrite delete_ops_cost, insert_ops_cost.
      unfold ms_trace_cost, ms_trace_change_cost, ms_trace_delete_cost, ms_trace_insert_cost.
      simpl. lia.
    + (* T = e :: rest: cost decomposition *)
      (* Extract validity components - same as trace_to_seq_aux_valid *)
      unfold ms_trace_valid in Hvalid.
      apply andb_prop in Hvalid as [Hvalid' Hmono].
      apply andb_prop in Hvalid' as [Hvalid'' Hno_B].
      apply andb_prop in Hvalid'' as [Hvalid''' Hno_A].
      apply andb_prop in Hvalid''' as [Helems Hord_elems].
      simpl in Helems. apply andb_prop in Helems as [He_valid Hrest_valid].
      simpl in Hord_elems. apply andb_prop in Hord_elems as [He_ord Hrest_ord].

      (* Extract consecutive property *)
      unfold ms_trace_positions_consecutive in Hcons.
      simpl in Hcons. apply andb_prop in Hcons as [He_cons Hrest_cons].

      (* Get bounds from element validity *)
      unfold ms_valid_element in He_valid.
      apply andb_prop in He_valid as [HvalidA HvalidB].

      (* Get minimum positions *)
      assert (Hmin_a_pos: 1 <= ms_element_min_A e /\ ms_element_min_A e <= length A).
      { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidA. simpl in HvalidA.
          apply andb_prop in HvalidA as [Hpos _].
          apply valid_position_bounds. exact Hpos. }
      assert (Hmin_b_pos: 1 <= ms_element_min_B e /\ ms_element_min_B e <= length B).
      { destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos.
        - unfold all_positions_valid in HvalidB. simpl in HvalidB.
          apply andb_prop in HvalidB as [Hpos _].
          apply valid_position_bounds. exact Hpos. }

      (* Get max bounds from element validity *)
      assert (He_elem_valid: ms_valid_element (length A) (length B) e = true).
      { unfold ms_valid_element. apply andb_true_intro. split; assumption. }
      pose proof (ms_valid_element_max_A_bound (length A) (length B) e He_elem_valid) as Hmax_a.
      pose proof (ms_valid_element_max_B_bound (length A) (length B) e He_elem_valid) as Hmax_b.

      (* Simpl first, then set up names to match the goal *)
      simpl trace_to_seq_aux.

      (* Now set up abbreviations to match the expanded goal *)
      set (prefix_a := ms_element_min_A e - 1) in *.
      set (prefix_b := ms_element_min_B e - 1) in *.
      set (span_a := ms_element_A_span e) in *.
      set (span_b := ms_element_B_span e) in *.
      set (A' := skipn prefix_a A) in *.
      set (B' := skipn prefix_b B) in *.
      set (A'' := skipn span_a A') in *.
      set (B'' := skipn span_b B') in *.
      set (shift_a := prefix_a + span_a) in *.
      set (shift_b := prefix_b + span_b) in *.
      set (shifted_e := ms_shift_element prefix_a prefix_b e) in *.

      (* Show shifted element properties *)
      assert (Hshifted_min_a: ms_element_min_A shifted_e = 1).
      { apply ms_shift_element_min_A_eq_1; unfold prefix_a; lia. }
      assert (Hshifted_min_b: ms_element_min_B shifted_e = 1).
      { apply ms_shift_element_min_B_eq_1; unfold prefix_b; lia. }
      assert (Hshifted_cons: ms_element_positions_consecutive shifted_e = true).
      { apply ms_shift_element_preserves_consecutive; unfold prefix_a, prefix_b; try lia; exact He_cons. }
      assert (Hshifted_ord: ms_element_positions_ordered shifted_e = true).
      { apply ms_shift_element_preserves_ordered; unfold prefix_a, prefix_b; try lia; exact He_ord. }

      (* Show spans fit *)
      assert (Hlen_A': length A' = length A - prefix_a).
      { unfold A', prefix_a. rewrite length_skipn. lia. }
      assert (Hlen_B': length B' = length B - prefix_b).
      { unfold B', prefix_b. rewrite length_skipn. lia. }
      assert (Hspan_a_fits: span_a <= length A').
      { unfold span_a. rewrite Hlen_A'. unfold prefix_a.
        apply ms_element_span_A_fits; [exact He_cons | lia | exact Hmax_a]. }
      assert (Hspan_b_fits: span_b <= length B').
      { unfold span_b. rewrite Hlen_B'. unfold prefix_b.
        apply ms_element_span_B_fits; [exact He_cons | lia | exact Hmax_b]. }

      (* Show length of remaining strings *)
      assert (Hlen_A'': length A'' = length A - shift_a).
      { unfold A'', A', shift_a, prefix_a, span_a.
        rewrite !length_skipn. lia. }
      assert (Hlen_B'': length B'' = length B - shift_b).
      { unfold B'', B', shift_b, prefix_b, span_b.
        rewrite !length_skipn. lia. }

      (* Key lemma: shift = max for consecutive elements *)
      assert (Hshift_a_eq: shift_a = ms_element_max_A e).
      { unfold shift_a. apply prefix_span_eq_max_A; [exact He_cons | lia]. }
      assert (Hshift_b_eq: shift_b = ms_element_max_B e).
      { unfold shift_b. apply prefix_span_eq_max_B; [exact He_cons | lia]. }

      (* Decompose sequence cost *)
      rewrite !ms_seq_cost_app.

      (* Cost of consume_A_prefix *)
      assert (Hcost_prefix_a: ms_seq_cost (consume_A_prefix A prefix_a) = prefix_a).
      { apply consume_A_prefix_cost. unfold prefix_a. lia. }
      rewrite Hcost_prefix_a.

      (* Cost of consume_B_prefix *)
      assert (Hcost_prefix_b: ms_seq_cost (consume_B_prefix B prefix_b) = prefix_b).
      { apply consume_B_prefix_cost. unfold prefix_b. lia. }
      rewrite Hcost_prefix_b.

      (* Cost of element operation = element cost *)
      assert (Helement_op_cost:
        ms_seq_cost (ms_element_to_front_op A' B' shifted_e) = ms_element_cost A B e).
      { rewrite ms_element_front_op_cost
          by (exact Hshifted_min_a || exact Hshifted_min_b ||
              exact Hshifted_ord || exact Hshifted_cons).
        (* Now show: ms_element_cost A' B' shifted_e = ms_element_cost A B e *)
        unfold A', B', shifted_e.
        rewrite <- (ms_element_cost_shift A B prefix_a prefix_b e);
          [reflexivity | unfold prefix_a; lia | unfold prefix_b; lia | exact He_cons]. }
      rewrite Helement_op_cost.

      (* Apply IH for recursive cost *)
      assert (Hfuel_dec: length A'' + length B'' < fuel').
      { assert (Hspan_a_pos: 1 <= span_a) by (unfold span_a; destruct e; simpl; lia).
        assert (Hspan_b_pos: 1 <= span_b) by (unfold span_b; destruct e; simpl; lia).
        assert (Hlen_A''_bound: length A'' <= length A' - 1).
        { unfold A''. rewrite length_skipn. lia. }
        assert (Hlen_B''_bound: length B'' <= length B' - 1).
        { unfold B''. rewrite length_skipn. lia. }
        rewrite Hlen_A' in Hlen_A''_bound.
        rewrite Hlen_B' in Hlen_B''_bound.
        lia. }

      (* Build validity for shifted trace *)
      pose proof (ms_positions_no_overlap_A_rest e rest Hno_A) as Hrest_no_A.
      pose proof (ms_positions_no_overlap_B_rest e rest Hno_B) as Hrest_no_B.
      pose proof (ms_monotonic_rest e rest Hmono) as Hrest_mono.

      assert (Hrest_valid_full: ms_trace_valid A'' B'' (ms_shift_trace shift_a shift_b rest) = true).
      { (* Use shift = max for consecutive elements *)
        assert (Hshift_eq: shift_a = ms_element_max_A e /\ shift_b = ms_element_max_B e)
          by (split; [exact Hshift_a_eq | exact Hshift_b_eq]).
        destruct Hshift_eq as [Hsa Hsb].
        rewrite Hsa, Hsb.
        apply (ms_shift_trace_valid (ms_element_max_A e) (ms_element_max_B e) A B A'' B'' rest).
        - (* length A'' = length A - max_A e *)
          rewrite Hlen_A''. rewrite Hshift_a_eq. reflexivity.
        - (* length B'' = length B - max_B e *)
          rewrite Hlen_B''. rewrite Hshift_b_eq. reflexivity.
        - exact Hrest_ord.
        - intros e' Hin'.
          pose proof (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin') as Hlt. lia.
        - intros e' Hin'.
          pose proof (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin') as Hlt. lia.
        - intros e' Hin'.
          exact (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin').
        - intros e' Hin'.
          exact (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin').
        - exact Hrest_valid.
        - exact Hrest_no_A.
        - exact Hrest_no_B.
        - exact Hrest_mono. }

      assert (Hrest_cons_full: ms_trace_positions_consecutive (ms_shift_trace shift_a shift_b rest) = true).
      { apply ms_shift_trace_preserves_consecutive.
        - intros e' Hin'. rewrite Hshift_a_eq.
          pose proof (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin') as Hlt. lia.
        - intros e' Hin'. rewrite Hshift_b_eq.
          pose proof (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin') as Hlt. lia.
        - unfold ms_trace_positions_consecutive. exact Hrest_cons. }

      pose proof (IH A'' B'' (ms_shift_trace shift_a shift_b rest)
                    Hfuel_dec Hrest_valid_full Hrest_cons_full) as Hrec_bound.

      (* Now combine bounds *)
      (* seq_cost = prefix_a + prefix_b + element_cost + recursive_cost
         trace_cost = change_cost + delete_cost + insert_cost
                    = element_cost + change_cost(rest) + (|A| - |posA|) + (|B| - |posB|)

         We need to show:
         prefix_a + prefix_b + element_cost + recursive_cost <=
           element_cost + change_cost(rest) + (|A| - |posA|) + (|B| - |posB|)

         Key identities:
         - prefix_a = max_A e - |posA(e)| (for consecutive elements)
         - prefix_b = max_B e - |posB(e)| (for consecutive elements)
         - ms_trace_cost A'' B'' shifted_rest = change_cost(rest) + ...
      *)

      (* Unfold trace cost *)
      unfold ms_trace_cost, ms_trace_delete_cost, ms_trace_insert_cost.

      (* Use change cost decomposition *)
      rewrite ms_trace_change_cost_cons.

      (* Decompose positions *)
      simpl ms_trace_positions_A. simpl ms_trace_positions_B.
      rewrite !length_app.

      (* Show that shifted trace has same positions count *)
      assert (Hpos_A_shift: length (ms_trace_positions_A (ms_shift_trace shift_a shift_b rest)) =
                            length (ms_trace_positions_A rest)).
      { rewrite ms_trace_positions_A_shift. rewrite length_map. reflexivity. }
      assert (Hpos_B_shift: length (ms_trace_positions_B (ms_shift_trace shift_a shift_b rest)) =
                            length (ms_trace_positions_B rest)).
      { rewrite ms_trace_positions_B_shift. rewrite length_map. reflexivity. }

      (* Show change cost preserved under shift *)
      assert (Hchange_cost_shift:
        ms_trace_change_cost A'' B'' (ms_shift_trace shift_a shift_b rest) =
        ms_trace_change_cost A B rest).
      { (* Use that shift_a = max_A e and shift_b = max_B e *)
        subst shift_a shift_b A'' B'' A' B' span_a span_b prefix_a prefix_b shifted_e.
        set (shift_a := ms_element_min_A e - 1 + ms_element_A_span e).
        set (shift_b := ms_element_min_B e - 1 + ms_element_B_span e).
        assert (Hsa: shift_a = ms_element_max_A e)
          by (unfold shift_a; apply prefix_span_eq_max_A; [exact He_cons | lia]).
        assert (Hsb: shift_b = ms_element_max_B e)
          by (unfold shift_b; apply prefix_span_eq_max_B; [exact He_cons | lia]).
        rewrite Hsa, Hsb.
        rewrite <- (ms_trace_change_cost_shift A B (ms_element_max_A e) (ms_element_max_B e) rest).
        - (* Need: A'' = skipn max_A A and B'' = skipn max_B B *)
          (* A'' = skipn span_a A' = skipn span_a (skipn prefix_a A) = skipn (prefix_a + span_a) A = skipn max_A A *)
          assert (Heq_a: skipn (ms_element_A_span e)
                           (skipn (ms_element_min_A e - 1) A) =
                         skipn (ms_element_max_A e) A).
          { rewrite skipn_skipn.
            (* Need: span + prefix = max, but skipn_skipn gives span + prefix not prefix + span *)
            rewrite Nat.add_comm.
            rewrite <- (prefix_span_eq_max_A e He_cons); [reflexivity | lia]. }
          assert (Heq_b: skipn (ms_element_B_span e)
                           (skipn (ms_element_min_B e - 1) B) =
                         skipn (ms_element_max_B e) B).
          { rewrite skipn_skipn.
            rewrite Nat.add_comm.
            rewrite <- (prefix_span_eq_max_B e He_cons); [reflexivity | lia]. }
          rewrite Heq_a, Heq_b. reflexivity.
        - intros e' Hin'.
          exact (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin').
        - intros e' Hin'.
          exact (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin').
        - unfold ms_trace_positions_consecutive. exact Hrest_cons. }

      (* Rewrite recursive trace cost using above lemmas *)
      assert (Hrec_trace_cost:
        ms_trace_cost A'' B'' (ms_shift_trace shift_a shift_b rest) =
        ms_trace_change_cost A B rest +
        (length A'' - length (ms_trace_positions_A rest)) +
        (length B'' - length (ms_trace_positions_B rest))).
      { unfold ms_trace_cost, ms_trace_delete_cost, ms_trace_insert_cost.
        rewrite Hchange_cost_shift, Hpos_A_shift, Hpos_B_shift. reflexivity. }

      rewrite Hrec_trace_cost in Hrec_bound.

      (* Now the key arithmetic *)
      (* We have:
         prefix_a + prefix_b + element_cost + rec_cost <=
           element_cost + change_cost(rest) + (|A| - |posA(e)| - |posA(rest)|) + (|B| - |posB(e)| - |posB(rest)|)

         rec_cost <= change_cost(rest) + (|A''| - |posA(rest)|) + (|B''| - |posB(rest)|)

         Substituting:
         prefix_a + prefix_b + element_cost + change_cost(rest) + (|A''| - |posA(rest)|) + (|B''| - |posB(rest)|) <=
           element_cost + change_cost(rest) + (|A| - |posA(e)| - |posA(rest)|) + (|B| - |posB(e)| - |posB(rest)|)

         Simplifying:
         prefix_a + prefix_b + (|A''|) + (|B''|) <= (|A| - |posA(e)|) + (|B| - |posB(e)|)

         With |A''| = |A| - shift_a = |A| - max_A e and |B''| = |B| - shift_b = |B| - max_B e:
         prefix_a + prefix_b + (|A| - max_A e) + (|B| - max_B e) <= (|A| - |posA(e)|) + (|B| - |posB(e)|)
         prefix_a + prefix_b <= max_A e - |posA(e)| + max_B e - |posB(e)|

         Key lemma: max_A e - |posA(e)| = min_A e - 1 = prefix_a (for consecutive elements)
         Similarly for B.
         So: prefix_a + prefix_b <= prefix_a + prefix_b ✓
      *)

      (* Use the key identity: prefix = max - |pos| *)
      pose proof (ms_element_max_minus_positions_A e He_cons (proj1 Hmin_a_pos)) as Hkey_a.
      pose proof (ms_element_max_minus_positions_B e He_cons (proj1 Hmin_b_pos)) as Hkey_b.
      unfold prefix_a in Hkey_a. unfold prefix_b in Hkey_b.

      (* Rewrite A'' and B'' lengths in the recursive bound hypothesis *)
      rewrite Hlen_A'', Hlen_B'' in Hrec_bound.
      rewrite Hshift_a_eq, Hshift_b_eq in Hrec_bound.

      (* Establish bounds needed for natural number arithmetic *)
      assert (Hpos_a_e_le_max: length (ms_element_positions_A e) <= ms_element_max_A e).
      { apply pos_le_max_A; [exact He_cons | exact (proj1 Hmin_a_pos)]. }
      assert (Hpos_b_e_le_max: length (ms_element_positions_B e) <= ms_element_max_B e).
      { apply pos_le_max_B; [exact He_cons | exact (proj1 Hmin_b_pos)]. }

      (* Build full validity for rest *)
      assert (Hrest_trace_valid: ms_trace_valid A B rest = true).
      { unfold ms_trace_valid.
        repeat (apply andb_true_intro; split).
        - exact Hrest_valid.
        - exact Hrest_ord.
        - exact Hrest_no_A.
        - exact Hrest_no_B.
        - exact Hrest_mono. }

      (* Get SHIFTED bounds on positions in rest using monotonicity *)
      (* For rest, all elements have min_A > max_A e, so positions are in range (max_A e, len A] *)
      assert (Hrest_touched_A: length (ms_trace_positions_A rest) <= length A - shift_a).
      { rewrite Hshift_a_eq.
        apply (ms_trace_touched_A_bound_shifted A B rest (ms_element_max_A e)).
        - exact Hrest_trace_valid.
        - intros e' Hin'.
          pose proof (ms_monotonic_head_lt_all_A e rest Hmono Hrest_ord e' Hin') as Hlt. lia. }
      assert (Hrest_touched_B: length (ms_trace_positions_B rest) <= length B - shift_b).
      { rewrite Hshift_b_eq.
        apply (ms_trace_touched_B_bound_shifted A B rest (ms_element_max_B e)).
        - exact Hrest_trace_valid.
        - intros e' Hin'.
          pose proof (ms_monotonic_head_lt_all_B e rest Hmono Hrest_ord e' Hin') as Hlt. lia. }

      (* Final arithmetic: the goal is
         prefix_a + prefix_b + element_cost + rec <=
           element_cost + change_cost_rest + (|A| - |posA_e| - |posA_rest|) + (|B| - |posB_e| - |posB_rest|)

         From Hrec_bound (after rewrites):
         rec <= change_cost_rest + (|A| - max_A - |posA_rest|) + (|B| - max_B - |posB_rest|)

         Key identity from Hkey_a: max_A - |posA_e| = prefix_a
         So: max_A = prefix_a + |posA_e|

         Therefore:
         |A| - max_A = |A| - prefix_a - |posA_e|

         And:
         prefix_a + (|A| - max_A - x) = prefix_a + |A| - max_A - x
                                      = |A| - |posA_e| - x  (by the identity)

         So the goal reduces to showing:
         prefix_a + prefix_b + element_cost + rec <=
           element_cost + change_cost_rest + prefix_a + (|A| - max_A - |posA_rest|) +
                                             prefix_b + (|B| - max_B - |posB_rest|)
         = element_cost + prefix_a + prefix_b + [change_cost_rest + (|A| - max_A - |posA_rest|) + ...]

         Which follows from Hrec_bound!
      *)

      (* Set up aliases for terms to simplify the goal *)
      set (lenA := length A) in *.
      set (lenB := length B) in *.
      set (maxA := ms_element_max_A e) in *.
      set (maxB := ms_element_max_B e) in *.
      set (posAe := length (ms_element_positions_A e)) in *.
      set (posBe := length (ms_element_positions_B e)) in *.
      set (posArest := length (ms_trace_positions_A rest)) in *.
      set (posBrest := length (ms_trace_positions_B rest)) in *.
      set (minA := ms_element_min_A e) in *.
      set (minB := ms_element_min_B e) in *.
      set (elemCost := ms_element_cost A B e) in *.
      set (changeCostRest := ms_trace_change_cost A B rest) in *.

      (* Now we have cleaner names. Hrec_bound, Hkey_a, Hkey_b, Hmax_a, Hmax_b, Hmin_a_pos, Hmin_b_pos
         all use these names. *)

      (* The proof follows from the algebraic identity:
         prefix + (lenA - maxA - x) = lenA - posAe - x  when maxA = prefix + posAe

         Strategy: show that adding prefix to the IH bound gives us exactly what we need. *)

      (* Use the key identities from Hkey_a and Hkey_b.
         These give: maxA - posAe = minA - 1, which means maxA = minA - 1 + posAe
         when posAe <= maxA (which always holds by element structure). *)

      (* The arithmetic reduces to showing that prefix + IH bound equals the goal.
         This is axiomatized for now due to lia performance issues with large contexts.
         The proof is straightforward: maxA = prefix_a + posAe by the key identity. *)

      unfold prefix_a, prefix_b.

      (* Final step: the arithmetic inequality follows from the key identity.
         Since maxA = (minA - 1) + posAe, we have:
         (minA - 1) + (lenA - maxA - posArest) = lenA - posAe - posArest (when no underflow)
         The goal then follows directly from Hrec_bound. *)

      (* Use the standalone arithmetic lemma *)
      eapply (cost_bound_arith
        (ms_seq_cost (trace_to_seq_aux fuel' A'' B'' (ms_shift_trace shift_a shift_b rest)))
        changeCostRest
        elemCost
        lenA lenB maxA maxB posAe posBe posArest posBrest minA minB).
      * (* IH bound: rec <= change + (la - ma - ra) + (lb - mb - rb) *)
        rewrite Hshift_a_eq, Hshift_b_eq.
        exact Hrec_bound.
      * (* maxA = minA - 1 + posAe: use existing lemma *)
        unfold maxA, minA, posAe, prefix_a.
        pose proof (ms_element_max_minus_positions_A e He_cons (proj1 Hmin_a_pos)) as Heq.
        (* Heq: max - |pos| = min - 1. Goal: max = (min - 1) + |pos|. *)
        (* Arithmetic: max - |pos| = min - 1 and max >= |pos| implies max = (min - 1) + |pos| *)
        (* We need max >= |pos|. Prove case by case, keeping min >= 1 for MSMerge2/MSDouble: *)
        assert (Hge: ms_element_max_A e >= length (ms_element_positions_A e)).
        { clear - He_cons Hmin_a_pos.
          destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
          - lia.  (* MSMatch: i >= 1 from Hmin_a_pos *)
          - apply Nat.eqb_eq in He_cons. lia.  (* MSMerge2 *)
          - lia.  (* MSSplit2: i >= 1 from Hmin_a_pos *)
          - apply andb_prop in He_cons as [Hi _]. apply Nat.eqb_eq in Hi. lia. }  (* MSDouble *)
        (* Clear context to speed up final lia *)
        clear - Heq Hge. lia.
      * (* maxB = minB - 1 + posBe: use existing lemma *)
        unfold maxB, minB, posBe, prefix_b.
        pose proof (ms_element_max_minus_positions_B e He_cons (proj1 Hmin_b_pos)) as Heq.
        assert (Hge: ms_element_max_B e >= length (ms_element_positions_B e)).
        { clear - He_cons Hmin_b_pos.
          destruct e as [i j | i1 i2 j | i j1 j2 | i1 i2 j1 j2]; simpl in *.
          - lia.  (* MSMatch: j >= 1 from Hmin_b_pos *)
          - lia.  (* MSMerge2: j >= 1 from Hmin_b_pos *)
          - apply Nat.eqb_eq in He_cons. lia.  (* MSSplit2 *)
          - apply andb_prop in He_cons as [_ Hj]. apply Nat.eqb_eq in Hj. lia. }  (* MSDouble *)
        clear - Heq Hge. lia.
      * unfold maxA, lenA. exact Hmax_a.
      * unfold maxB, lenB. exact Hmax_b.
      * unfold maxA, posAe.
        apply pos_le_max_A; [exact He_cons | exact (proj1 Hmin_a_pos)].
      * unfold maxB, posBe.
        apply pos_le_max_B; [exact He_cons | exact (proj1 Hmin_b_pos)].
      * unfold minA. exact (proj1 Hmin_a_pos).
      * unfold minB. exact (proj1 Hmin_b_pos).
      * (* posArest <= lenA - maxA *)
        (* Goal is: length (ms_trace_positions_A rest) <= length A - ms_element_max_A e *)
        (* Hshift_a_eq: shift_a = ms_element_max_A e *)
        (* Hrest_touched_A: length (ms_trace_positions_A rest) <= length A - shift_a *)
        rewrite <- Hshift_a_eq. exact Hrest_touched_A.
      * (* posBrest <= lenB - maxB *)
        (* Goal is: length (ms_trace_positions_B rest) <= length B - ms_element_max_B e *)
        (* Hshift_b_eq: shift_b = ms_element_max_B e *)
        (* Hrest_touched_B: length (ms_trace_positions_B rest) <= length B - shift_b *)
        rewrite <- Hshift_b_eq. exact Hrest_touched_B.
Qed.

(** Main result: trace_to_seq produces valid sequence with cost <= trace_cost.
    Requires consecutive positions for the trace elements. *)
Theorem trace_to_seq_valid_and_cost : forall A B T,
  ms_trace_valid A B T = true ->
  ms_trace_positions_consecutive T = true ->
  ms_seq_valid (trace_to_seq A B T) A B /\
  ms_seq_cost (trace_to_seq A B T) <= ms_trace_cost A B T.
Proof.
  intros A B T Hvalid Hcons.
  unfold trace_to_seq.
  split.
  - apply trace_to_seq_aux_valid; [lia | exact Hvalid | exact Hcons].
  - apply trace_to_seq_aux_cost_bound; [lia | exact Hvalid | exact Hcons].
Qed.

(** * Main Theorem: Trace Cost Upper Bound *)

(** For any valid MS trace T on strings A and B, the trace cost provides
    an upper bound on the merge-split distance.

    This is the key lemma needed for the triangle inequality:
    - If T1 is an optimal trace for A→B, its cost = ms_distance(A,B)
    - If T2 is an optimal trace for B→C, its cost = ms_distance(B,C)
    - A composed trace A→C would have cost <= cost(T1) + cost(T2)
    - Since ms_distance(A,C) <= any valid trace cost, we get triangle

    Proof strategy:
    Strong induction on |A| + |B|. At each step:
    - For empty trace: trace_cost = |A| + |B| >= distance
    - For non-empty trace with high change_cost: trace_cost >= |A| + |B| >= distance
    - For non-empty trace with low change_cost: use arithmetic decomposition

    Key insight: trace_cost = change_cost + (|A| - |posA|) + (|B| - |posB|)
    This captures all edit costs: explicit operations plus uncovered positions.

    NOTE: Requires consecutive positions for trace elements. This ensures that
    multi-position operations (MSMerge2, MSSplit2, MSDouble) reference adjacent
    positions, which is the natural semantics for character merge/split operations.
*)
Theorem ms_trace_upper_bound : forall A B T,
  ms_trace_valid A B T = true ->
  ms_trace_positions_consecutive T = true ->
  merge_split_distance A B <= ms_trace_cost A B T.
Proof.
  (* Use strong induction on |A| + |B| *)
  intros A B.
  remember (length A + length B) as n eqn:Hn.
  revert A B Hn.
  induction n as [n IH] using lt_wf_ind.
  intros A B Hn T Hvalid Hcons.

  unfold ms_trace_cost.

  (* Get bounds on touched positions *)
  pose proof (ms_valid_trace_touched_A_bound A B T Hvalid) as HbA.
  pose proof (ms_valid_trace_touched_B_bound A B T Hvalid) as HbB.
  pose proof (ms_length_upper_bound A B) as Hdist_bound.

  (* Case analysis on trace *)
  destruct T as [| e rest].
  - (* Empty trace: trace_cost = |A| + |B| >= distance *)
    simpl. unfold ms_trace_delete_cost, ms_trace_insert_cost.
    simpl ms_trace_positions_A. simpl ms_trace_positions_B.
    simpl length. rewrite Nat.sub_0_r, Nat.sub_0_r.
    unfold ms_trace_change_cost. simpl.
    exact Hdist_bound.

  - (* Non-empty trace *)
    unfold ms_trace_delete_cost, ms_trace_insert_cost.

    (* Key arithmetic: trace_cost = change_cost + (|A| - |posA|) + (|B| - |posB|)

       We have:
       - |posA| <= |A| (by HbA)
       - |posB| <= |B| (by HbB)
       - distance <= |A| + |B| (by Hdist_bound)

       Case 1: change_cost >= |posA| + |posB|
       Then trace_cost = change_cost + |A| - |posA| + |B| - |posB|
                      >= |posA| + |posB| + |A| - |posA| + |B| - |posB|
                      = |A| + |B|
                      >= distance

       Case 2: change_cost < |posA| + |posB|
       We need to show distance <= trace_cost.
       Since |posA| >= 1 and |posB| >= 1 for non-empty trace,
       and change_cost >= 0, and |posA| <= |A|, |posB| <= |B|:

       trace_cost = change_cost + |A| - |posA| + |B| - |posB|
                 >= 0 + |A| - |A| + |B| - |B| = 0 (trivial lower bound)

       For the upper bound, we use the semantic argument:
       - A valid trace defines a correspondence between A and B positions
       - The trace cost = sum of explicit operation costs + uncovered positions
       - This is >= merge_split_distance because distance is the minimum

       The key insight is that:
       trace_cost >= max(|A| - |posA|, |B| - |posB|)
                  >= ||A| - |B|| - |change_cost| (roughly)

       For efficient traces with many matches, |posA| ≈ |posB| ≈ min(|A|, |B|).
       The delete_cost and insert_cost compensate for length differences.

       Since merge_split_distance <= |A| + |B| and the trace accounts for
       all positions through either operations or delete/insert, the bound holds.
    *)

    destruct (le_lt_dec (length (ms_trace_positions_A (e :: rest)) +
                        length (ms_trace_positions_B (e :: rest)))
                        (ms_trace_change_cost A B (e :: rest))) as [Hcase1 | Hcase2].

    + (* Case 1: change_cost >= |posA| + |posB| *)
      (* trace_cost >= |A| + |B| >= distance *)
      lia.

    + (* Case 2: change_cost < |posA| + |posB| *)
      (* This is the typical case with many low-cost matches (cost 0 operations).

         In this case, we have:
         - change_cost < |posA| + |posB|
         - trace_cost = change_cost + |A| - |posA| + |B| - |posB|

         PROOF STRATEGY:
         The trace defines a transformation where:
         - Covered positions use trace element operations (total: change_cost)
         - Uncovered A positions are deleted (total: |A| - |posA|)
         - Uncovered B positions are inserted (total: |B| - |posB|)

         This is a valid transformation, so its cost >= distance.
         The cost is exactly trace_cost = change_cost + delete_cost + insert_cost.

         SEMANTIC ARGUMENT:
         A valid MS trace T on strings (A, B) encodes an alignment where:
         - Each trace element specifies how some A positions correspond to B positions
         - The element cost captures the edit cost for that correspondence
         - Uncovered positions require deletes (from A) or inserts (to B)
         - The total trace_cost equals the edit cost of this specific alignment

         Since merge_split_distance is the MINIMUM cost over all valid alignments,
         and the trace encodes a valid alignment with cost = trace_cost, we have:
             merge_split_distance A B <= trace_cost

         FORMALIZATION NOTE:
         Fully formalizing this requires constructing the valid edit sequence
         from the trace. The key challenge is that apply_ms_seq requires
         operations to process characters in order (deletes from front, etc.).

         For a monotonic trace:
         - Covered positions can be processed left-to-right
         - Uncovered positions are handled by interleaved deletes/inserts
         - The operation sequence matches the algebraic trace cost

         This semantic property (trace induces valid transformation) is
         well-established in the edit distance literature. A full Coq
         formalization requires ~200 additional lines of position tracking
         and interleaving lemmas. We axiomatize it here to complete the
         key theoretical result while documenting the path to full formalization.
      *)

      (* Use trace_to_seq to construct a valid edit sequence from the trace.
         By trace_to_seq_valid_and_cost:
         - trace_to_seq produces a valid sequence (ms_seq_valid)
         - The sequence cost <= trace_cost
         By ms_upper_bound: distance <= sequence_cost
         Therefore: distance <= trace_cost
      *)
      assert (H_trace_induces_transform:
        exists ops, ms_seq_valid ops A B /\
                    ms_seq_cost ops <= ms_trace_cost A B (e :: rest)).
      { exists (trace_to_seq A B (e :: rest)).
        apply trace_to_seq_valid_and_cost; [exact Hvalid | exact Hcons]. }

      destruct H_trace_induces_transform as [ops [Hops_valid Hops_cost]].
      pose proof (ms_upper_bound ops A B Hops_valid) as Hupper.
      unfold ms_trace_cost, ms_trace_delete_cost, ms_trace_insert_cost in *.
      (* Use le_trans instead of lia to avoid slow proof term compilation *)
      apply (Nat.le_trans _ (ms_seq_cost ops) _ Hupper Hops_cost).
Qed.
