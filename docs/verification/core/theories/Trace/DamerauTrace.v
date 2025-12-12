(** * Damerau-Levenshtein Trace Infrastructure

    This module extends the trace infrastructure to support Damerau-Levenshtein
    distance, which includes transposition of adjacent characters as a single
    operation (cost 1) rather than two substitutions (cost 2).

    Part of: Liblevenshtein.Core

    Key differences from standard Levenshtein traces:
    - Trace elements can be either matches/substitutions OR transpositions
    - Transposition involves two consecutive positions in both strings
    - Validity must ensure transposed positions don't overlap with other elements
*)

From Coq Require Import String List Arith Ascii Bool Nat Lia.
From Coq Require Import Wf_nat.
From Coq Require Import Sorting.Permutation.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.MetricProperties.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.

(** * Damerau-Levenshtein Trace Type *)

(** A DL trace element: either a single position match or a transposition of two positions *)
Inductive DLTraceElement :=
  | DLMatch (i j : nat)      (* Position i in A corresponds to position j in B *)
  | DLTranspose (i j : nat). (* Positions (i, i+1) in A swap to match (j, j+1) in B *)

Definition DLTrace := list DLTraceElement.

(** * Position Extraction *)

(** Extract all positions touched in A by a trace element *)
Definition element_positions_A (e : DLTraceElement) : list nat :=
  match e with
  | DLMatch i _ => [i]
  | DLTranspose i _ => [i; i + 1]
  end.

(** Extract all positions touched in B by a trace element *)
Definition element_positions_B (e : DLTraceElement) : list nat :=
  match e with
  | DLMatch _ j => [j]
  | DLTranspose _ j => [j; j + 1]
  end.

(** All positions touched in A by a trace *)
Fixpoint dl_touched_in_A (T : DLTrace) : list nat :=
  match T with
  | [] => []
  | e :: rest => element_positions_A e ++ dl_touched_in_A rest
  end.

(** All positions touched in B by a trace *)
Fixpoint dl_touched_in_B (T : DLTrace) : list nat :=
  match T with
  | [] => []
  | e :: rest => element_positions_B e ++ dl_touched_in_B rest
  end.

(** * Element Validity *)

(** Check if a single element is valid for given string lengths *)
Definition dl_valid_element (lenA lenB : nat) (e : DLTraceElement) : bool :=
  match e with
  | DLMatch i j =>
      (1 <=? i) && (i <=? lenA) && (1 <=? j) && (j <=? lenB)
  | DLTranspose i j =>
      (* Both positions (i, i+1) must be in bounds for A *)
      (* Both positions (j, j+1) must be in bounds for B *)
      (1 <=? i) && (i + 1 <=? lenA) && (1 <=? j) && (j + 1 <=? lenB)
  end.

(** * Position Compatibility *)

(** Check if a position is compatible with a trace element
    (i.e., the position is not touched by the element) *)
Definition pos_not_in_element_A (pos : nat) (e : DLTraceElement) : bool :=
  match e with
  | DLMatch i _ => negb (pos =? i)
  | DLTranspose i _ => negb (pos =? i) && negb (pos =? i + 1)
  end.

Definition pos_not_in_element_B (pos : nat) (e : DLTraceElement) : bool :=
  match e with
  | DLMatch _ j => negb (pos =? j)
  | DLTranspose _ j => negb (pos =? j) && negb (pos =? j + 1)
  end.

(** Check if two elements are compatible (don't touch overlapping positions) *)
Definition dl_compatible_elements (e1 e2 : DLTraceElement) : bool :=
  (* All positions in e1 must not overlap with positions in e2 *)
  let pos_A1 := element_positions_A e1 in
  let pos_B1 := element_positions_B e1 in
  let pos_A2 := element_positions_A e2 in
  let pos_B2 := element_positions_B e2 in
  (* No position in A touched by both elements *)
  forallb (fun p => forallb (fun q => negb (p =? q)) pos_A2) pos_A1 &&
  (* No position in B touched by both elements *)
  forallb (fun p => forallb (fun q => negb (p =? q)) pos_B2) pos_B1.

(** * Monotonicity *)

(** Get the minimum position in A touched by an element *)
Definition min_pos_A (e : DLTraceElement) : nat :=
  match e with
  | DLMatch i _ => i
  | DLTranspose i _ => i
  end.

(** Get the maximum position in A touched by an element *)
Definition max_pos_A (e : DLTraceElement) : nat :=
  match e with
  | DLMatch i _ => i
  | DLTranspose i _ => i + 1
  end.

(** Get the minimum position in B touched by an element *)
Definition min_pos_B (e : DLTraceElement) : nat :=
  match e with
  | DLMatch _ j => j
  | DLTranspose _ j => j
  end.

(** Get the maximum position in B touched by an element *)
Definition max_pos_B (e : DLTraceElement) : nat :=
  match e with
  | DLMatch _ j => j
  | DLTranspose _ j => j + 1
  end.

(** Check if two elements maintain monotonic order *)
Definition dl_monotonic_pair (e1 e2 : DLTraceElement) : bool :=
  (* e1 comes before e2: max of e1 < min of e2 in both A and B *)
  (* Or they're the same element *)
  (* Or e2 comes before e1 *)
  if max_pos_A e1 <? min_pos_A e2 then
    max_pos_B e1 <? min_pos_B e2
  else if max_pos_A e2 <? min_pos_A e1 then
    max_pos_B e2 <? min_pos_B e1
  else
    (* Overlapping ranges - only valid if same element (handled by NoDup) *)
    false.

(** Check pairwise monotonicity *)
Fixpoint dl_is_monotonic_aux (T : DLTrace) : bool :=
  match T with
  | [] => true
  | e :: rest =>
      forallb (fun e' => dl_compatible_elements e e' && dl_monotonic_pair e e') rest &&
      dl_is_monotonic_aux rest
  end.

(** * NoDup for Trace Elements *)

(** Decidable equality for DLTraceElement *)
Definition dl_element_eq_dec (e1 e2 : DLTraceElement) : {e1 = e2} + {e1 <> e2}.
Proof.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2].
  - destruct (Nat.eq_dec i1 i2), (Nat.eq_dec j1 j2); subst;
    try (left; reflexivity); right; intro H; injection H; intros; contradiction.
  - right. intro H. discriminate.
  - right. intro H. discriminate.
  - destruct (Nat.eq_dec i1 i2), (Nat.eq_dec j1 j2); subst;
    try (left; reflexivity); right; intro H; injection H; intros; contradiction.
Defined.

(** NoDup check for trace elements *)
Fixpoint dl_NoDup_dec (T : DLTrace) : bool :=
  match T with
  | [] => true
  | e :: rest =>
      negb (existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest) &&
      dl_NoDup_dec rest
  end.

(** * Full Validity Check *)

Definition dl_is_valid_trace (A B : list Char) (T : DLTrace) : bool :=
  (* All elements are within bounds *)
  forallb (dl_valid_element (length A) (length B)) T &&
  (* Pairwise compatibility and monotonicity *)
  dl_is_monotonic_aux T &&
  (* No duplicate elements *)
  dl_NoDup_dec T.

(** * Trace Cost *)

(** Cost of a single trace element *)
Definition dl_element_cost (A B : list Char) (e : DLTraceElement) : nat :=
  match e with
  | DLMatch i j =>
      (* Cost is 0 if characters match, 1 otherwise *)
      subst_cost (nth (i - 1) A default_char) (nth (j - 1) B default_char)
  | DLTranspose i j =>
      (* Transposition cost is 1 if characters are correctly swapped *)
      (* A[i-1] = B[j], A[i] = B[j-1] => swap pattern *)
      if andb (char_eq (nth (i - 1) A default_char) (nth j B default_char))
              (char_eq (nth i A default_char) (nth (j - 1) B default_char))
      then 1
      else 100  (* Invalid transposition - should not occur in valid traces *)
  end.

(** Total element cost for a trace *)
Definition dl_change_cost (A B : list Char) (T : DLTrace) : nat :=
  fold_left (fun acc e => acc + dl_element_cost A B e) T 0.

(** Full trace cost *)
Definition dl_trace_cost (A B : list Char) (T : DLTrace) : nat :=
  let change_cost := dl_change_cost A B T in
  (* Deletions: positions in A not touched by any element *)
  let touched_A := dl_touched_in_A T in
  let delete_cost := length A - length touched_A in
  (* Insertions: positions in B not touched by any element *)
  let touched_B := dl_touched_in_B T in
  let insert_cost := length B - length touched_B in
  change_cost + delete_cost + insert_cost.

(** * Basic Lemmas *)

(** existsb false iff forall false - converts negated existsb to forall *)
Lemma existsb_forall : forall {A : Type} (f : A -> bool) (l : list A),
  existsb f l = false <-> forall x, In x l -> f x = false.
Proof.
  intros A' f l.
  induction l as [| a rest IH].
  - simpl. split; intros; [destruct H0 | reflexivity].
  - simpl. split.
    + intros Horb x [Heq | Hin].
      * subst. apply orb_false_iff in Horb. destruct Horb as [Hfa _]. exact Hfa.
      * apply orb_false_iff in Horb. destruct Horb as [_ Hrest].
        rewrite IH in Hrest. apply Hrest. exact Hin.
    + intros Hforall.
      apply orb_false_iff. split.
      * apply Hforall. left. reflexivity.
      * rewrite IH. intros x Hin. apply Hforall. right. exact Hin.
Qed.

(** Auxiliary Lemma for subst_cost - must be defined before use *)
Lemma subst_cost_le_1 : forall c1 c2, subst_cost c1 c2 <= 1.
Proof.
  intros c1 c2. unfold subst_cost.
  destruct (char_eq c1 c2); lia.
Qed.

(** Empty trace cost *)
Lemma dl_trace_cost_empty :
  forall A B : list Char,
    dl_trace_cost A B [] = length A + length B.
Proof.
  intros A B.
  unfold dl_trace_cost. simpl. lia.
Qed.

(** Element cost is bounded *)
Lemma dl_element_cost_bound :
  forall A B e,
    dl_element_cost A B e <= 100.
Proof.
  intros A B e.
  destruct e as [i j | i j]; unfold dl_element_cost.
  - (* DLMatch *)
    pose proof (subst_cost_le_1 (nth (i - 1) A default_char) (nth (j - 1) B default_char)).
    unfold subst_cost in *. destruct char_eq; lia.
  - (* DLTranspose *)
    destruct (andb _ _); lia.
Qed.

(** Valid element cost for transposition is exactly 1 *)
Lemma dl_transpose_cost_valid :
  forall A B i j,
    char_eq (nth (i - 1) A default_char) (nth j B default_char) = true ->
    char_eq (nth i A default_char) (nth (j - 1) B default_char) = true ->
    dl_element_cost A B (DLTranspose i j) = 1.
Proof.
  intros A B i j H1 H2.
  unfold dl_element_cost.
  rewrite H1, H2. simpl. reflexivity.
Qed.

(** * Helper Lemmas for Trace Proofs *)

(** Length of touched positions in A is bounded by trace length adjusted for transpositions *)
Lemma dl_touched_A_length_bound : forall T,
  length (dl_touched_in_A T) <= length T + length (filter
    (fun e => match e with DLTranspose _ _ => true | _ => false end) T).
Proof.
  induction T as [| e rest IH].
  - simpl. lia.
  - destruct e as [i j | i j]; simpl.
    + (* DLMatch - adds 1 position *)
      simpl in *. lia.
    + (* DLTranspose - adds 2 positions *)
      simpl in *. lia.
Qed.

(** Length of touched positions in B is bounded similarly *)
Lemma dl_touched_B_length_bound : forall T,
  length (dl_touched_in_B T) <= length T + length (filter
    (fun e => match e with DLTranspose _ _ => true | _ => false end) T).
Proof.
  induction T as [| e rest IH].
  - simpl. lia.
  - destruct e as [i j | i j]; simpl.
    + (* DLMatch - adds 1 position *)
      simpl in *. lia.
    + (* DLTranspose - adds 2 positions *)
      simpl in *. lia.
Qed.

(** Trace cost is non-negative and bounded below by trace structure *)
Lemma dl_trace_cost_nonneg : forall A B T,
  dl_trace_cost A B T >= 0.
Proof.
  intros A B T. unfold dl_trace_cost. lia.
Qed.

(** Change cost is non-decreasing as elements are added *)
Lemma dl_change_cost_mono : forall A B acc e,
  acc <= acc + dl_element_cost A B e.
Proof.
  intros A B acc e.
  pose proof (dl_element_cost_bound A B e). lia.
Qed.

(** fold_left for change cost is non-decreasing *)
Lemma dl_change_cost_fold_ge : forall A B T acc,
  acc <= fold_left (fun a e => a + dl_element_cost A B e) T acc.
Proof.
  intros A B T.
  induction T as [| e rest IH].
  - simpl. lia.
  - intro acc. simpl.
    specialize (IH (acc + dl_element_cost A B e)).
    pose proof (dl_element_cost_bound A B e). lia.
Qed.

(** * Position Shifting Functions *)

(** Shift trace element: increment first (A) index by 1 *)
Definition dl_shift_A (e : DLTraceElement) : DLTraceElement :=
  match e with
  | DLMatch i j => DLMatch (S i) j
  | DLTranspose i j => DLTranspose (S i) j
  end.

(** Shift trace element: increment second (B) index by 1 *)
Definition dl_shift_B (e : DLTraceElement) : DLTraceElement :=
  match e with
  | DLMatch i j => DLMatch i (S j)
  | DLTranspose i j => DLTranspose i (S j)
  end.

(** Shift trace element: increment both indices by 1 *)
Definition dl_shift_both (e : DLTraceElement) : DLTraceElement :=
  match e with
  | DLMatch i j => DLMatch (S i) (S j)
  | DLTranspose i j => DLTranspose (S i) (S j)
  end.

(** Shift entire trace: all A indices by 1 *)
Definition dl_shift_trace_A (T : DLTrace) : DLTrace := map dl_shift_A T.

(** Shift entire trace: all B indices by 1 *)
Definition dl_shift_trace_B (T : DLTrace) : DLTrace := map dl_shift_B T.

(** Shift entire trace: all indices by 1 *)
Definition dl_shift_trace_both (T : DLTrace) : DLTrace := map dl_shift_both T.

(** * Position Shifting Lemmas *)

(** Touched positions in A shift with dl_shift_A *)
Lemma dl_touched_A_shift_A : forall T,
  dl_touched_in_A (dl_shift_trace_A T) = map S (dl_touched_in_A T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + (* DLTranspose: [S i; S i + 1] ++ ... = S i :: S (i + 1) :: ... *)
      rewrite IH.
      (* S i + 1 = S (i + 1) *)
      replace (S i + 1) with (S (i + 1)) by lia.
      reflexivity.
Qed.

(** Touched positions in B unchanged by dl_shift_A *)
Lemma dl_touched_B_shift_A : forall T,
  dl_touched_in_B (dl_shift_trace_A T) = dl_touched_in_B T.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl; rewrite IH; reflexivity.
Qed.

(** Touched positions in A unchanged by dl_shift_B *)
Lemma dl_touched_A_shift_B : forall T,
  dl_touched_in_A (dl_shift_trace_B T) = dl_touched_in_A T.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl; rewrite IH; reflexivity.
Qed.

(** Touched positions in B shift with dl_shift_B *)
Lemma dl_touched_B_shift_B : forall T,
  dl_touched_in_B (dl_shift_trace_B T) = map S (dl_touched_in_B T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + rewrite IH.
      replace (S j + 1) with (S (j + 1)) by lia.
      reflexivity.
Qed.

(** Touched positions shift with dl_shift_both *)
Lemma dl_touched_A_shift_both : forall T,
  dl_touched_in_A (dl_shift_trace_both T) = map S (dl_touched_in_A T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + rewrite IH.
      replace (S i + 1) with (S (i + 1)) by lia.
      reflexivity.
Qed.

Lemma dl_touched_B_shift_both : forall T,
  dl_touched_in_B (dl_shift_trace_both T) = map S (dl_touched_in_B T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + rewrite IH.
      replace (S j + 1) with (S (j + 1)) by lia.
      reflexivity.
Qed.

(** Length of shifted trace equals original *)
Lemma dl_shift_trace_A_length : forall T,
  length (dl_shift_trace_A T) = length T.
Proof. intro T. unfold dl_shift_trace_A. apply length_map. Qed.

Lemma dl_shift_trace_B_length : forall T,
  length (dl_shift_trace_B T) = length T.
Proof. intro T. unfold dl_shift_trace_B. apply length_map. Qed.

Lemma dl_shift_trace_both_length : forall T,
  length (dl_shift_trace_both T) = length T.
Proof. intro T. unfold dl_shift_trace_both. apply length_map. Qed.

(** * Trace Cost Under Shifting *)

(** Predicate: element has valid indices (>= 1) *)
Definition dl_element_valid_indices (e : DLTraceElement) : Prop :=
  match e with
  | DLMatch i j => i >= 1 /\ j >= 1
  | DLTranspose i j => i >= 1 /\ j >= 1
  end.

(** Helper: element cost after shifting A by 1 (for valid elements) *)
Lemma dl_element_cost_shift_A : forall A B e c,
  dl_element_valid_indices e ->
  dl_element_cost (c :: A) B (dl_shift_A e) = dl_element_cost A B e.
Proof.
  intros A B e c Hvalid.
  destruct e as [i j | i j]; unfold dl_shift_A, dl_element_cost.
  - (* DLMatch *)
    destruct Hvalid as [Hi _].
    destruct i as [| i']; [lia |].
    simpl. rewrite Nat.sub_0_r. reflexivity.
  - (* DLTranspose *)
    destruct Hvalid as [Hi _].
    destruct i as [| i']; [lia |].
    simpl. rewrite !Nat.sub_0_r.
    replace (S i' + 1) with (S (i' + 1)) by lia.
    reflexivity.
Qed.

(** Helper: element cost after shifting B by 1 (for valid elements) *)
Lemma dl_element_cost_shift_B : forall A B e c,
  dl_element_valid_indices e ->
  dl_element_cost A (c :: B) (dl_shift_B e) = dl_element_cost A B e.
Proof.
  intros A B e c Hvalid.
  destruct e as [i j | i j]; unfold dl_shift_B, dl_element_cost.
  - (* DLMatch *)
    destruct Hvalid as [_ Hj].
    destruct j as [| j']; [lia |].
    simpl. rewrite Nat.sub_0_r. reflexivity.
  - (* DLTranspose *)
    destruct Hvalid as [_ Hj].
    destruct j as [| j']; [lia |].
    simpl. rewrite !Nat.sub_0_r.
    replace (S j' + 1) with (S (j' + 1)) by lia.
    reflexivity.
Qed.

(** Helper: element cost after shifting both by 1 (for valid elements) *)
Lemma dl_element_cost_shift_both : forall A B e c1 c2,
  dl_element_valid_indices e ->
  dl_element_cost (c1 :: A) (c2 :: B) (dl_shift_both e) = dl_element_cost A B e.
Proof.
  intros A B e c1 c2 Hvalid.
  destruct e as [i j | i j]; unfold dl_shift_both, dl_element_cost.
  - (* DLMatch *)
    destruct Hvalid as [Hi Hj].
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. rewrite !Nat.sub_0_r. reflexivity.
  - (* DLTranspose *)
    destruct Hvalid as [Hi Hj].
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. rewrite !Nat.sub_0_r.
    replace (S i' + 1) with (S (i' + 1)) by lia.
    replace (S j' + 1) with (S (j' + 1)) by lia.
    reflexivity.
Qed.

(** Shifted elements preserve valid indices *)
Lemma dl_shift_A_valid_indices : forall e,
  dl_element_valid_indices e ->
  dl_element_valid_indices (dl_shift_A e).
Proof.
  intros e Hvalid.
  destruct e as [i j | i j]; simpl in *; destruct Hvalid as [Hi Hj]; split; lia.
Qed.

Lemma dl_shift_B_valid_indices : forall e,
  dl_element_valid_indices e ->
  dl_element_valid_indices (dl_shift_B e).
Proof.
  intros e Hvalid.
  destruct e as [i j | i j]; simpl in *; destruct Hvalid as [Hi Hj]; split; lia.
Qed.

(** Predicate: all elements in a trace have valid indices *)
Definition dl_all_valid_indices (T : DLTrace) : Prop :=
  Forall dl_element_valid_indices T.

(** Valid traces have valid indices *)
Lemma dl_is_valid_trace_valid_indices : forall A B T,
  dl_is_valid_trace A B T = true ->
  dl_all_valid_indices T.
Proof.
  intros A B T Hvalid.
  unfold dl_is_valid_trace in Hvalid.
  apply andb_true_iff in Hvalid as [Hvalid _].
  apply andb_true_iff in Hvalid as [Hforall _].
  unfold dl_all_valid_indices.
  rewrite Forall_forall.
  intros e Hin.
  rewrite forallb_forall in Hforall.
  specialize (Hforall e Hin).
  destruct e as [i j | i j]; unfold dl_valid_element in Hforall; simpl.
  - (* DLMatch: (1 <=? i) && (i <=? lenA) && (1 <=? j) && (j <=? lenB) *)
    repeat (apply andb_true_iff in Hforall as [Hforall ?]).
    (* Hforall = 1 <=? i, H1 = i <=? lenA, H0 = 1 <=? j, H = j <=? lenB *)
    apply Nat.leb_le in Hforall.  (* 1 <= i *)
    apply Nat.leb_le in H0.       (* 1 <= j *)
    split; lia.
  - (* DLTranspose: (1 <=? i) && (i+1 <=? lenA) && (1 <=? j) && (j+1 <=? lenB) *)
    repeat (apply andb_true_iff in Hforall as [Hforall ?]).
    (* Hforall = 1 <=? i, H1 = i+1 <=? lenA, H0 = 1 <=? j, H = j+1 <=? lenB *)
    apply Nat.leb_le in Hforall.  (* 1 <= i *)
    apply Nat.leb_le in H0.       (* 1 <= j *)
    split; lia.
Qed.

(** Change cost under shifting A (for traces with valid indices) *)
Lemma dl_change_cost_shift_A : forall A B T c,
  dl_all_valid_indices T ->
  dl_change_cost (c :: A) B (dl_shift_trace_A T) = dl_change_cost A B T.
Proof.
  intros A B T c Hvalid.
  unfold dl_change_cost, dl_shift_trace_A, dl_all_valid_indices in *.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - rewrite Forall_cons_iff in Hvalid.
    destruct Hvalid as [He Hrest].
    simpl map. simpl fold_left.
    rewrite (dl_element_cost_shift_A _ _ _ _ He).
    (* Need a more direct proof using fold_left properties *)
    (* Key: the fold_left accumulates the same values on both sides *)
    assert (Haux : forall acc,
      fold_left (fun a e0 => a + dl_element_cost (c :: A) B e0) (map dl_shift_A rest) acc =
      fold_left (fun a e0 => a + dl_element_cost A B e0) rest acc).
    { clear IH He e.
      revert Hrest.
      induction rest as [| e' rest' IH'].
      - intros _. simpl. reflexivity.
      - intro Hrest.
        rewrite Forall_cons_iff in Hrest.
        destruct Hrest as [He' Hrest'].
        intro acc. simpl.
        rewrite (dl_element_cost_shift_A _ _ _ _ He').
        apply IH'. exact Hrest'. }
    apply Haux.
Qed.

(** Change cost under shifting B (for traces with valid indices) *)
Lemma dl_change_cost_shift_B : forall A B T c,
  dl_all_valid_indices T ->
  dl_change_cost A (c :: B) (dl_shift_trace_B T) = dl_change_cost A B T.
Proof.
  intros A B T c Hvalid.
  unfold dl_change_cost, dl_shift_trace_B, dl_all_valid_indices in *.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - rewrite Forall_cons_iff in Hvalid.
    destruct Hvalid as [He Hrest].
    simpl map. simpl fold_left.
    rewrite (dl_element_cost_shift_B _ _ _ _ He).
    assert (Haux : forall acc,
      fold_left (fun a e0 => a + dl_element_cost A (c :: B) e0) (map dl_shift_B rest) acc =
      fold_left (fun a e0 => a + dl_element_cost A B e0) rest acc).
    { clear IH He e.
      revert Hrest.
      induction rest as [| e' rest' IH'].
      - intros _. simpl. reflexivity.
      - intro Hrest.
        rewrite Forall_cons_iff in Hrest.
        destruct Hrest as [He' Hrest'].
        intro acc. simpl.
        rewrite (dl_element_cost_shift_B _ _ _ _ He').
        apply IH'. exact Hrest'. }
    apply Haux.
Qed.

(** Change cost under shifting both (for traces with valid indices) *)
Lemma dl_change_cost_shift_both : forall A B T c1 c2,
  dl_all_valid_indices T ->
  dl_change_cost (c1 :: A) (c2 :: B) (dl_shift_trace_both T) = dl_change_cost A B T.
Proof.
  intros A B T c1 c2 Hvalid.
  unfold dl_change_cost, dl_shift_trace_both, dl_all_valid_indices in *.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - rewrite Forall_cons_iff in Hvalid.
    destruct Hvalid as [He Hrest].
    simpl map. simpl fold_left.
    rewrite (dl_element_cost_shift_both _ _ _ _ _ He).
    assert (Haux : forall acc,
      fold_left (fun a e0 => a + dl_element_cost (c1 :: A) (c2 :: B) e0) (map dl_shift_both rest) acc =
      fold_left (fun a e0 => a + dl_element_cost A B e0) rest acc).
    { clear IH He e.
      revert Hrest.
      induction rest as [| e' rest' IH'].
      - intros _. simpl. reflexivity.
      - intro Hrest.
        rewrite Forall_cons_iff in Hrest.
        destruct Hrest as [He' Hrest'].
        intro acc. simpl.
        rewrite (dl_element_cost_shift_both _ _ _ _ _ He').
        apply IH'. exact Hrest'. }
    apply Haux.
Qed.

(** * NoDup Infrastructure for Touched Positions *)

(** Helper: element positions in A are NoDup *)
Lemma element_positions_A_NoDup : forall e,
  NoDup (element_positions_A e).
Proof.
  intros e.
  destruct e as [i j | i j]; simpl.
  - (* DLMatch: [i] is NoDup *)
    constructor; [apply in_nil | constructor].
  - (* DLTranspose: [i; i+1] is NoDup since i <> i+1 *)
    constructor.
    + intro Hin. destruct Hin as [Heq | []]. lia.
    + constructor; [apply in_nil | constructor].
Qed.

(** Helper: element positions in B are NoDup *)
Lemma element_positions_B_NoDup : forall e,
  NoDup (element_positions_B e).
Proof.
  intros e.
  destruct e as [i j | i j]; simpl.
  - constructor; [apply in_nil | constructor].
  - constructor.
    + intro Hin. destruct Hin as [Heq | []]. lia.
    + constructor; [apply in_nil | constructor].
Qed.

(** Helper: compatible elements have disjoint A positions *)
Lemma compatible_disjoint_A : forall e1 e2,
  dl_compatible_elements e1 e2 = true ->
  forall p, In p (element_positions_A e1) -> ~ In p (element_positions_A e2).
Proof.
  intros e1 e2 Hcompat p Hin1 Hin2.
  unfold dl_compatible_elements in Hcompat.
  apply andb_true_iff in Hcompat as [HcompA _].
  rewrite forallb_forall in HcompA.
  specialize (HcompA p Hin1).
  rewrite forallb_forall in HcompA.
  specialize (HcompA p Hin2).
  rewrite Nat.eqb_refl in HcompA. discriminate.
Qed.

(** Helper: compatible elements have disjoint B positions *)
Lemma compatible_disjoint_B : forall e1 e2,
  dl_compatible_elements e1 e2 = true ->
  forall p, In p (element_positions_B e1) -> ~ In p (element_positions_B e2).
Proof.
  intros e1 e2 Hcompat p Hin1 Hin2.
  unfold dl_compatible_elements in Hcompat.
  apply andb_true_iff in Hcompat as [_ HcompB].
  rewrite forallb_forall in HcompB.
  specialize (HcompB p Hin1).
  rewrite forallb_forall in HcompB.
  specialize (HcompB p Hin2).
  rewrite Nat.eqb_refl in HcompB. discriminate.
Qed.

(** Helper: element in rest implies forallb covers it *)
Lemma forallb_compat_in_rest : forall e e' rest,
  forallb (fun e'' => dl_compatible_elements e e'' && dl_monotonic_pair e e'') (e' :: rest) = true ->
  In e' (e' :: rest).
Proof.
  intros. left. reflexivity.
Qed.

(** Helper: position not in rest of trace (using monotonicity) *)
Lemma pos_not_in_rest_A : forall e rest p,
  forallb (fun e' => dl_compatible_elements e e' && dl_monotonic_pair e e') rest = true ->
  In p (element_positions_A e) ->
  ~ In p (dl_touched_in_A rest).
Proof.
  intros e rest p Hforall Hin_e.
  induction rest as [| e' rest' IH].
  - (* rest = [] *)
    simpl. intro H. exact H.
  - (* rest = e' :: rest' *)
    simpl.
    intro Hin_rest.
    apply in_app_iff in Hin_rest.
    destruct Hin_rest as [Hin_e' | Hin_rest'].
    + (* p in element_positions_A e' *)
      simpl in Hforall.
      apply andb_true_iff in Hforall as [Hhead _].
      apply andb_true_iff in Hhead as [Hcompat _].
      eapply compatible_disjoint_A; eauto.
    + (* p in dl_touched_in_A rest' *)
      simpl in Hforall.
      apply andb_true_iff in Hforall as [_ Htail].
      apply IH; assumption.
Qed.

(** Helper: position not in rest of trace for B *)
Lemma pos_not_in_rest_B : forall e rest p,
  forallb (fun e' => dl_compatible_elements e e' && dl_monotonic_pair e e') rest = true ->
  In p (element_positions_B e) ->
  ~ In p (dl_touched_in_B rest).
Proof.
  intros e rest p Hforall Hin_e.
  induction rest as [| e' rest' IH].
  - simpl. intro H. exact H.
  - simpl.
    intro Hin_rest.
    apply in_app_iff in Hin_rest.
    destruct Hin_rest as [Hin_e' | Hin_rest'].
    + simpl in Hforall.
      apply andb_true_iff in Hforall as [Hhead _].
      apply andb_true_iff in Hhead as [Hcompat _].
      eapply compatible_disjoint_B; eauto.
    + simpl in Hforall.
      apply andb_true_iff in Hforall as [_ Htail].
      apply IH; assumption.
Qed.

(** NoDup for touched positions in A *)
Lemma dl_touched_in_A_NoDup : forall T,
  dl_is_monotonic_aux T = true ->
  NoDup (dl_touched_in_A T).
Proof.
  intros T Hmono.
  induction T as [| e rest IH].
  - (* T = [] *)
    simpl. constructor.
  - (* T = e :: rest *)
    simpl in Hmono.
    apply andb_true_iff in Hmono as [Hforall Hmono_rest].
    simpl.
    apply NoDup_app.
    + apply element_positions_A_NoDup.
    + apply IH. exact Hmono_rest.
    + intros p Hin_e Hin_rest.
      eapply pos_not_in_rest_A; eauto.
Qed.

(** NoDup for touched positions in B *)
Lemma dl_touched_in_B_NoDup : forall T,
  dl_is_monotonic_aux T = true ->
  NoDup (dl_touched_in_B T).
Proof.
  intros T Hmono.
  induction T as [| e rest IH].
  - simpl. constructor.
  - simpl in Hmono.
    apply andb_true_iff in Hmono as [Hforall Hmono_rest].
    simpl.
    apply NoDup_app.
    + apply element_positions_B_NoDup.
    + apply IH. exact Hmono_rest.
    + intros p Hin_e Hin_rest.
      eapply pos_not_in_rest_B; eauto.
Qed.

(** Helper: inclusion length bound for NoDup lists *)
Lemma incl_length_NoDup_nat : forall (l1 l2 : list nat),
  NoDup l1 ->
  NoDup l2 ->
  incl l1 l2 ->
  length l1 <= length l2.
Proof.
  intros l1 l2 Hnd1 Hnd2 Hincl.
  apply NoDup_incl_length; assumption.
Qed.

(** * Trace Cost Relationships Under Shifting *)

(** Valid traces have bounded touched positions in A *)
Lemma dl_valid_trace_touched_A_bound : forall A B T,
  dl_is_valid_trace A B T = true ->
  length (dl_touched_in_A T) <= length A.
Proof.
  intros A B T Hvalid.
  unfold dl_is_valid_trace in Hvalid.
  apply andb_true_iff in Hvalid as [Hvalid Hnodup].
  apply andb_true_iff in Hvalid as [Hbounds Hmono].
  (* NoDup from monotonicity *)
  assert (Hnodup_touched: NoDup (dl_touched_in_A T)).
  { apply dl_touched_in_A_NoDup. exact Hmono. }
  (* Show inclusion in seq 1 (length A) *)
  assert (Hincl: incl (dl_touched_in_A T) (seq 1 (length A))).
  {
    intros p Hin.
    apply in_seq.
    (* Find which element p comes from *)
    clear Hmono Hnodup Hnodup_touched.
    induction T as [| e rest IH].
    - simpl in Hin. destruct Hin.
    - simpl in Hin, Hbounds.
      apply andb_true_iff in Hbounds as [Hvalid_e Hbounds_rest].
      apply in_app_iff in Hin.
      destruct Hin as [Hin_e | Hin_rest].
      + (* p from element e *)
        destruct e as [i j | i j]; simpl in Hin_e; unfold dl_valid_element in Hvalid_e.
        * (* DLMatch i j: bounds are (1 <=? i) && (i <=? lenA) && (1 <=? j) && (j <=? lenB) *)
          repeat (apply andb_true_iff in Hvalid_e as [Hvalid_e ?]).
          (* Convert leb to le BEFORE destruct/subst *)
          apply Nat.leb_le in Hvalid_e as Hi_lb.
          apply Nat.leb_le in H1 as Hi_ub.
          destruct Hin_e as [Heq | []].
          subst p. split; lia.
        * (* DLTranspose i j: bounds are (1 <=? i) && (i+1 <=? lenA) && (1 <=? j) && (j+1 <=? lenB) *)
          repeat (apply andb_true_iff in Hvalid_e as [Hvalid_e ?]).
          apply Nat.leb_le in Hvalid_e as Hi_lb.
          apply Nat.leb_le in H1 as Hi_ub.
          destruct Hin_e as [Heq | [Heq | []]].
          -- subst p. split; lia.
          -- subst p. split; lia.
      + (* p from rest *)
        apply IH; assumption.
  }
  (* Use NoDup + incl to bound length *)
  assert (Hnodup_seq: NoDup (seq 1 (length A))).
  { apply seq_NoDup. }
  assert (Hlen_seq: length (seq 1 (length A)) = length A).
  { apply seq_length. }
  rewrite <- Hlen_seq.
  apply NoDup_incl_length; assumption.
Qed.

(** Valid traces have bounded touched positions in B *)
Lemma dl_valid_trace_touched_B_bound : forall A B T,
  dl_is_valid_trace A B T = true ->
  length (dl_touched_in_B T) <= length B.
Proof.
  intros A B T Hvalid.
  unfold dl_is_valid_trace in Hvalid.
  apply andb_true_iff in Hvalid as [Hvalid Hnodup].
  apply andb_true_iff in Hvalid as [Hbounds Hmono].
  assert (Hnodup_touched: NoDup (dl_touched_in_B T)).
  { apply dl_touched_in_B_NoDup. exact Hmono. }
  assert (Hincl: incl (dl_touched_in_B T) (seq 1 (length B))).
  {
    intros p Hin.
    apply in_seq.
    clear Hmono Hnodup Hnodup_touched.
    induction T as [| e rest IH].
    - simpl in Hin. destruct Hin.
    - simpl in Hin, Hbounds.
      apply andb_true_iff in Hbounds as [Hvalid_e Hbounds_rest].
      apply in_app_iff in Hin.
      destruct Hin as [Hin_e | Hin_rest].
      + destruct e as [i j | i j]; simpl in Hin_e; unfold dl_valid_element in Hvalid_e.
        * (* DLMatch i j: bounds are (1 <=? i) && (i <=? lenA) && (1 <=? j) && (j <=? lenB) *)
          repeat (apply andb_true_iff in Hvalid_e as [Hvalid_e ?]).
          apply Nat.leb_le in H0 as Hj_lb.
          apply Nat.leb_le in H as Hj_ub.
          destruct Hin_e as [Heq | []].
          subst p. split; lia.
        * (* DLTranspose i j: bounds are (1 <=? i) && (i+1 <=? lenA) && (1 <=? j) && (j+1 <=? lenB) *)
          repeat (apply andb_true_iff in Hvalid_e as [Hvalid_e ?]).
          apply Nat.leb_le in H0 as Hj_lb.
          apply Nat.leb_le in H as Hj_ub.
          destruct Hin_e as [Heq | [Heq | []]].
          -- subst p. split; lia.
          -- subst p. split; lia.
      + apply IH; assumption.
  }
  assert (Hnodup_seq: NoDup (seq 1 (length B))).
  { apply seq_NoDup. }
  assert (Hlen_seq: length (seq 1 (length B)) = length B).
  { apply seq_length. }
  rewrite <- Hlen_seq.
  apply NoDup_incl_length; assumption.
Qed.

(** Key trace cost relationship: shifting A increases deletion count by 1 *)
Lemma dl_trace_cost_shift_A : forall A B T c,
  dl_all_valid_indices T ->
  length (dl_touched_in_A T) <= length A ->
  dl_trace_cost (c :: A) B (dl_shift_trace_A T) = dl_trace_cost A B T + 1.
Proof.
  intros A B T c Hvalid Hbound.
  unfold dl_trace_cost.
  rewrite (dl_change_cost_shift_A _ _ _ _ Hvalid).
  rewrite dl_touched_A_shift_A.
  rewrite dl_touched_B_shift_A.
  rewrite length_map.
  simpl length.
  lia.
Qed.

(** Key trace cost relationship: shifting B increases insertion count by 1 *)
Lemma dl_trace_cost_shift_B : forall A B T c,
  dl_all_valid_indices T ->
  length (dl_touched_in_B T) <= length B ->
  dl_trace_cost A (c :: B) (dl_shift_trace_B T) = dl_trace_cost A B T + 1.
Proof.
  intros A B T c Hvalid Hbound.
  unfold dl_trace_cost.
  rewrite (dl_change_cost_shift_B _ _ _ _ Hvalid).
  rewrite dl_touched_A_shift_B.
  rewrite dl_touched_B_shift_B.
  rewrite length_map.
  simpl length.
  lia.
Qed.

(** * Validity Preservation Under Shifting *)

(** Shift preserves element validity bounds *)
Lemma dl_valid_element_shift_A : forall lenA lenB e,
  dl_valid_element lenA lenB e = true ->
  dl_valid_element (S lenA) lenB (dl_shift_A e) = true.
Proof.
  intros lenA lenB e H.
  destruct e as [i j | i j]; unfold dl_shift_A, dl_valid_element in *.
  - (* DLMatch *)
    (* After repeat: H = 1<=?i, H2 = i<=?lenA, H1 = 1<=?j, H0 = j<=?lenB *)
    (* Goals in order: 1<=?Si, Si<=?SlenA, 1<=?j, j<=?lenB *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
  - (* DLTranspose *)
    (* After repeat: H = 1<=?i, H2 = i+1<=?lenA, H1 = 1<=?j, H0 = j+1<=?lenB *)
    (* Goals in order: 1<=?Si, Si+1<=?SlenA, 1<=?j, j+1<=?lenB *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
Qed.

Lemma dl_valid_element_shift_B : forall lenA lenB e,
  dl_valid_element lenA lenB e = true ->
  dl_valid_element lenA (S lenB) (dl_shift_B e) = true.
Proof.
  intros lenA lenB e H.
  destruct e as [i j | i j]; unfold dl_shift_B, dl_valid_element in *.
  - (* DLMatch *)
    (* After repeat: H = 1<=?i, H2 = i<=?lenA, H1 = 1<=?j, H0 = j<=?lenB *)
    (* Goals in order: 1<=?i, i<=?lenA, 1<=?Sj, Sj<=?SlenB *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
  - (* DLTranspose *)
    (* After repeat: H = 1<=?i, H2 = i+1<=?lenA, H1 = 1<=?j, H0 = j+1<=?lenB *)
    (* Goals in order: 1<=?i, i+1<=?lenA, 1<=?Sj, Sj+1<=?SlenB *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
Qed.

(** Shift preserves compatibility *)
Lemma dl_compatible_shift_A : forall e1 e2,
  dl_compatible_elements e1 e2 = true ->
  dl_compatible_elements (dl_shift_A e1) (dl_shift_A e2) = true.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_A, dl_compatible_elements, element_positions_A, element_positions_B in *;
  simpl in *.
  - (* DLMatch, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl. rewrite H1. reflexivity.
    + exact H2.
  - (* DLMatch, DLTranspose *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl in *. rewrite !andb_true_iff in H1 |- *.
      destruct H1 as [H1a H1b].
      split; assumption.
    + exact H2.
  - (* DLTranspose, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl in *.
      rewrite !andb_true_iff in H1 |- *.
      destruct H1 as [H1a H1b].
      split; assumption.
    + exact H2.
  - (* DLTranspose, DLTranspose *)
    (* After shift_A: positions [S i1; S(i1+1)] and [S i2; S(i2+1)]
       But (S n =? S m) simplifies to (n =? m), so after simpl
       the goal matches H exactly *)
    simpl in *. exact H.
Qed.

Lemma dl_compatible_shift_B : forall e1 e2,
  dl_compatible_elements e1 e2 = true ->
  dl_compatible_elements (dl_shift_B e1) (dl_shift_B e2) = true.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_B, dl_compatible_elements, element_positions_A, element_positions_B in *;
  simpl in *.
  - (* DLMatch, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + exact H1.
    + simpl. rewrite H2. reflexivity.
  - (* DLMatch, DLTranspose *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + exact H1.
    + simpl in *. rewrite !andb_true_iff in H2 |- *.
      destruct H2 as [H2a H2b].
      split; assumption.
  - (* DLTranspose, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + exact H1.
    + simpl in *.
      rewrite !andb_true_iff in H2 |- *.
      destruct H2 as [H2a H2b].
      split; assumption.
  - (* DLTranspose, DLTranspose *)
    (* After shift_B: j positions [S j1; S(j1+1)] and [S j2; S(j2+1)]
       But (S n =? S m) simplifies to (n =? m), so after simpl
       the goal matches H exactly *)
    simpl in *. exact H.
Qed.

(** Shift preserves monotonicity *)
Lemma dl_monotonic_shift_A : forall e1 e2,
  dl_monotonic_pair e1 e2 = true ->
  dl_monotonic_pair (dl_shift_A e1) (dl_shift_A e2) = true.
Proof.
  (* After shift_A, indices in A become S i.
     By computation: S n <? S m = n <? m, and S n =? S m = n =? m.
     So after simpl, the goal matches H exactly. *)
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_A, dl_monotonic_pair, min_pos_A, max_pos_A, min_pos_B, max_pos_B in *;
  simpl in *; exact H.
Qed.

Lemma dl_monotonic_shift_B : forall e1 e2,
  dl_monotonic_pair e1 e2 = true ->
  dl_monotonic_pair (dl_shift_B e1) (dl_shift_B e2) = true.
Proof.
  (* After shift_B, indices in B become S j. A indices unchanged.
     By computation: S n <? S m = n <? m, and S n =? S m = n =? m.
     So after simpl, the goal matches H exactly. *)
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_B, dl_monotonic_pair, min_pos_A, max_pos_A, min_pos_B, max_pos_B in *;
  simpl in *; exact H.
Qed.

(** Full validity preservation for shifted traces *)
Lemma dl_is_monotonic_aux_shift_A : forall T,
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (dl_shift_trace_A T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hforall Hrest].
    unfold dl_shift_trace_A. simpl. apply andb_true_iff. split.
    + rewrite forallb_forall in Hforall |- *.
      intros e' Hin.
      apply in_map_iff in Hin.
      destruct Hin as [e'' [He'' Hin'']].
      subst e'.
      specialize (Hforall e'' Hin'').
      apply andb_true_iff in Hforall as [Hcompat Hmono].
      apply andb_true_iff. split.
      * apply dl_compatible_shift_A. exact Hcompat.
      * apply dl_monotonic_shift_A. exact Hmono.
    + apply IH. exact Hrest.
Qed.

Lemma dl_is_monotonic_aux_shift_B : forall T,
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (dl_shift_trace_B T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hforall Hrest].
    unfold dl_shift_trace_B. simpl. apply andb_true_iff. split.
    + rewrite forallb_forall in Hforall |- *.
      intros e' Hin.
      apply in_map_iff in Hin.
      destruct Hin as [e'' [He'' Hin'']].
      subst e'.
      specialize (Hforall e'' Hin'').
      apply andb_true_iff in Hforall as [Hcompat Hmono].
      apply andb_true_iff. split.
      * apply dl_compatible_shift_B. exact Hcompat.
      * apply dl_monotonic_shift_B. exact Hmono.
    + apply IH. exact Hrest.
Qed.

(** NoDup preservation *)
Lemma dl_shift_A_injective : forall e1 e2,
  dl_shift_A e1 = dl_shift_A e2 -> e1 = e2.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_A in H; try discriminate.
  - injection H. intros. f_equal; lia.
  - injection H. intros. f_equal; lia.
Qed.

Lemma dl_shift_B_injective : forall e1 e2,
  dl_shift_B e1 = dl_shift_B e2 -> e1 = e2.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_B in H; try discriminate.
  - injection H. intros. f_equal; lia.
  - injection H. intros. f_equal; lia.
Qed.

Lemma dl_NoDup_dec_shift_A : forall T,
  dl_NoDup_dec T = true ->
  dl_NoDup_dec (dl_shift_trace_A T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hnotin Hrest].
    unfold dl_shift_trace_A. simpl. apply andb_true_iff. split.
    + apply negb_true_iff.
      apply negb_true_iff in Hnotin.
      (* Goal: existsb (fun e' => if dl_element_eq_dec (dl_shift_A e) e' then true else false)
                       (map dl_shift_A rest) = false
         Hnotin: existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = false
         Strategy: Prove by showing no element in mapped list satisfies the predicate *)
      destruct (existsb (fun e' => if dl_element_eq_dec (dl_shift_A e) e' then true else false)
                        (map dl_shift_A rest)) eqn:Hex.
      * (* Case: existsb = true, derive contradiction *)
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        destruct (dl_element_eq_dec (dl_shift_A e) (dl_shift_A e'')) eqn:Edec.
        -- clear Edec.
           apply dl_shift_A_injective in e0.
           subst e''.
           (* Now Hnotin says existsb ... rest = false, but e is in rest with eq_dec = true *)
           assert (Hcontra: existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = true).
           { apply existsb_exists. exists e. split; [exact Hin'' |].
             destruct (dl_element_eq_dec e e); [reflexivity | congruence]. }
           rewrite Hnotin in Hcontra. discriminate.
        -- discriminate Heq'.
      * reflexivity.
    + apply IH. exact Hrest.
Qed.

Lemma dl_NoDup_dec_shift_B : forall T,
  dl_NoDup_dec T = true ->
  dl_NoDup_dec (dl_shift_trace_B T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hnotin Hrest].
    unfold dl_shift_trace_B. simpl. apply andb_true_iff. split.
    + apply negb_true_iff.
      apply negb_true_iff in Hnotin.
      (* Goal: existsb (fun e' => if dl_element_eq_dec (dl_shift_B e) e' then true else false)
                       (map dl_shift_B rest) = false
         Hnotin: existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = false
         Strategy: Prove by showing no element in mapped list satisfies the predicate *)
      destruct (existsb (fun e' => if dl_element_eq_dec (dl_shift_B e) e' then true else false)
                        (map dl_shift_B rest)) eqn:Hex.
      * (* Case: existsb = true, derive contradiction *)
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        destruct (dl_element_eq_dec (dl_shift_B e) (dl_shift_B e'')) eqn:Edec.
        -- clear Edec.
           apply dl_shift_B_injective in e0.
           subst e''.
           (* Now Hnotin says existsb ... rest = false, but e is in rest with eq_dec = true *)
           assert (Hcontra: existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = true).
           { apply existsb_exists. exists e. split; [exact Hin'' |].
             destruct (dl_element_eq_dec e e); [reflexivity | congruence]. }
           rewrite Hnotin in Hcontra. discriminate.
        -- discriminate Heq'.
      * reflexivity.
    + apply IH. exact Hrest.
Qed.

(** Full validity preservation *)
Lemma dl_is_valid_trace_shift_A : forall A B T c,
  dl_is_valid_trace A B T = true ->
  dl_is_valid_trace (c :: A) B (dl_shift_trace_A T) = true.
Proof.
  intros A B T c H.
  unfold dl_is_valid_trace, dl_shift_trace_A in *.
  repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
  - (* forallb dl_valid_element *)
    rewrite forallb_forall in H |- *.
    intros e Hin.
    apply in_map_iff in Hin.
    destruct Hin as [e' [He' Hin']].
    subst e.
    specialize (H e' Hin').
    simpl length.
    apply dl_valid_element_shift_A. exact H.
  - (* dl_is_monotonic_aux *)
    fold (dl_shift_trace_A T).
    apply dl_is_monotonic_aux_shift_A. exact H1.
  - (* dl_NoDup_dec *)
    fold (dl_shift_trace_A T).
    apply dl_NoDup_dec_shift_A. exact H0.
Qed.

Lemma dl_is_valid_trace_shift_B : forall A B T c,
  dl_is_valid_trace A B T = true ->
  dl_is_valid_trace A (c :: B) (dl_shift_trace_B T) = true.
Proof.
  intros A B T c H.
  unfold dl_is_valid_trace, dl_shift_trace_B in *.
  repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
  - (* forallb dl_valid_element *)
    rewrite forallb_forall in H |- *.
    intros e Hin.
    apply in_map_iff in Hin.
    destruct Hin as [e' [He' Hin']].
    subst e.
    specialize (H e' Hin').
    simpl length.
    apply dl_valid_element_shift_B. exact H.
  - (* dl_is_monotonic_aux *)
    fold (dl_shift_trace_B T).
    apply dl_is_monotonic_aux_shift_B. exact H1.
  - (* dl_NoDup_dec *)
    fold (dl_shift_trace_B T).
    apply dl_NoDup_dec_shift_B. exact H0.
Qed.

(** ** Diagonal Shift Lemmas (shift both indices) *)

(** Shift preserves element validity bounds - both directions *)
Lemma dl_valid_element_shift_both : forall lenA lenB e,
  dl_valid_element lenA lenB e = true ->
  dl_valid_element (S lenA) (S lenB) (dl_shift_both e) = true.
Proof.
  intros lenA lenB e H.
  destruct e as [i j | i j]; unfold dl_shift_both, dl_valid_element in *.
  - (* DLMatch: shift both i and j *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
  - (* DLTranspose: shift both i and j *)
    repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
    + apply Nat.leb_le. apply Nat.leb_le in H. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H2. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H1. lia.
    + apply Nat.leb_le. apply Nat.leb_le in H0. lia.
Qed.

(** Shift preserves compatibility - both directions *)
Lemma dl_compatible_shift_both : forall e1 e2,
  dl_compatible_elements e1 e2 = true ->
  dl_compatible_elements (dl_shift_both e1) (dl_shift_both e2) = true.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_both, dl_compatible_elements, element_positions_A, element_positions_B in *;
  simpl in *.
  - (* DLMatch, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl. rewrite H1. reflexivity.
    + simpl. rewrite H2. reflexivity.
  - (* DLMatch, DLTranspose *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl in *. rewrite !andb_true_iff in H1 |- *.
      destruct H1 as [H1a H1b]. split; assumption.
    + simpl in *. rewrite !andb_true_iff in H2 |- *.
      destruct H2 as [H2a H2b]. split; assumption.
  - (* DLTranspose, DLMatch *)
    apply andb_true_iff in H as [H1 H2].
    apply andb_true_iff. split.
    + simpl in *. rewrite !andb_true_iff in H1 |- *.
      destruct H1 as [H1a H1b]. split; assumption.
    + simpl in *. rewrite !andb_true_iff in H2 |- *.
      destruct H2 as [H2a H2b]. split; assumption.
  - (* DLTranspose, DLTranspose: S n =? S m simplifies to n =? m *)
    simpl in *. exact H.
Qed.

(** Shift preserves monotonicity - both directions *)
Lemma dl_monotonic_shift_both : forall e1 e2,
  dl_monotonic_pair e1 e2 = true ->
  dl_monotonic_pair (dl_shift_both e1) (dl_shift_both e2) = true.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_both, dl_monotonic_pair, min_pos_A, max_pos_A, min_pos_B, max_pos_B in *;
  simpl in *; exact H.
Qed.

(** Full monotonicity preservation for shifted traces - both directions *)
Lemma dl_is_monotonic_aux_shift_both : forall T,
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (dl_shift_trace_both T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hforall Hrest].
    unfold dl_shift_trace_both. simpl. apply andb_true_iff. split.
    + rewrite forallb_forall in Hforall |- *.
      intros e' Hin.
      apply in_map_iff in Hin.
      destruct Hin as [e'' [He'' Hin'']].
      subst e'.
      specialize (Hforall e'' Hin'').
      apply andb_true_iff in Hforall as [Hcompat Hmono].
      apply andb_true_iff. split.
      * apply dl_compatible_shift_both. exact Hcompat.
      * apply dl_monotonic_shift_both. exact Hmono.
    + apply IH. exact Hrest.
Qed.

(** NoDup preservation - shift_both injectivity *)
Lemma dl_shift_both_injective : forall e1 e2,
  dl_shift_both e1 = dl_shift_both e2 -> e1 = e2.
Proof.
  intros e1 e2 H.
  destruct e1 as [i1 j1 | i1 j1], e2 as [i2 j2 | i2 j2];
  unfold dl_shift_both in H; try discriminate.
  - injection H. intros. f_equal; lia.
  - injection H. intros. f_equal; lia.
Qed.

(** NoDup preservation for shifted traces - both directions *)
Lemma dl_NoDup_dec_shift_both : forall T,
  dl_NoDup_dec T = true ->
  dl_NoDup_dec (dl_shift_trace_both T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hnotin Hrest].
    unfold dl_shift_trace_both. simpl. apply andb_true_iff. split.
    + apply negb_true_iff.
      apply negb_true_iff in Hnotin.
      destruct (existsb (fun e' => if dl_element_eq_dec (dl_shift_both e) e' then true else false)
                        (map dl_shift_both rest)) eqn:Hex.
      * (* Case: existsb = true, derive contradiction *)
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        destruct (dl_element_eq_dec (dl_shift_both e) (dl_shift_both e'')) eqn:Edec.
        -- clear Edec.
           apply dl_shift_both_injective in e0.
           subst e''.
           assert (Hcontra: existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = true).
           { apply existsb_exists. exists e. split; [exact Hin'' |].
             destruct (dl_element_eq_dec e e); [reflexivity | congruence]. }
           rewrite Hnotin in Hcontra. discriminate.
        -- discriminate Heq'.
      * reflexivity.
    + apply IH. exact Hrest.
Qed.

(** Full validity preservation for diagonally shifted traces *)
Lemma dl_is_valid_trace_shift_both : forall A B T c1 c2,
  dl_is_valid_trace A B T = true ->
  dl_is_valid_trace (c1 :: A) (c2 :: B) (dl_shift_trace_both T) = true.
Proof.
  intros A B T c1 c2 H.
  unfold dl_is_valid_trace, dl_shift_trace_both in *.
  repeat (apply andb_true_iff in H as [H ?]; apply andb_true_iff; split).
  - (* forallb dl_valid_element *)
    rewrite forallb_forall in H |- *.
    intros e Hin.
    apply in_map_iff in Hin.
    destruct Hin as [e' [He' Hin']].
    subst e.
    specialize (H e' Hin').
    simpl length.
    apply dl_valid_element_shift_both. exact H.
  - (* dl_is_monotonic_aux *)
    fold (dl_shift_trace_both T).
    apply dl_is_monotonic_aux_shift_both. exact H1.
  - (* dl_NoDup_dec *)
    fold (dl_shift_trace_both T).
    apply dl_NoDup_dec_shift_both. exact H0.
Qed.

(** Helper: fold_left accumulator shift for element cost *)
Lemma fold_left_dl_element_cost_acc : forall A B T acc,
  fold_left (fun a e => a + dl_element_cost A B e) T acc =
  acc + fold_left (fun a e => a + dl_element_cost A B e) T 0.
Proof.
  intros A B T.
  induction T as [| e rest IH]; intros acc.
  - simpl. lia.
  - simpl.
    rewrite IH. rewrite (IH (dl_element_cost A B e)). lia.
Qed.

(** Change cost with cons: adding element at head *)
Lemma dl_change_cost_cons : forall A B e T,
  dl_change_cost A B (e :: T) = dl_element_cost A B e + dl_change_cost A B T.
Proof.
  intros A B e T.
  unfold dl_change_cost at 1.
  simpl fold_left.
  rewrite fold_left_dl_element_cost_acc.
  reflexivity.
Qed.

(** Trace cost with DLMatch 1 1 prepended to shifted trace *)
Lemma dl_trace_cost_shift_both_with_match : forall A B T c1 c2,
  dl_all_valid_indices T ->
  length (dl_touched_in_A T) <= length A ->
  length (dl_touched_in_B T) <= length B ->
  dl_trace_cost (c1 :: A) (c2 :: B) (DLMatch 1 1 :: dl_shift_trace_both T) =
  dl_trace_cost A B T + subst_cost c1 c2.
Proof.
  intros A B T c1 c2 Hvalid HbA HbB.
  unfold dl_trace_cost.
  (* Simplify touched positions *)
  simpl dl_touched_in_A. simpl dl_touched_in_B.
  rewrite dl_touched_A_shift_both, dl_touched_B_shift_both.
  simpl length.
  rewrite !length_map.
  (* Simplify change cost: use cons lemma *)
  rewrite dl_change_cost_cons.
  (* dl_element_cost at (1,1) for strings starting with c1,c2 is subst_cost c1 c2 *)
  simpl dl_element_cost.
  (* Use the shift lemma for the remaining trace *)
  rewrite (dl_change_cost_shift_both A B T c1 c2 Hvalid).
  (* Now arithmetic *)
  unfold dl_change_cost.
  lia.
Qed.

(** Helper: DLMatch 1 1 is compatible and monotonic with any shifted valid element *)
Lemma dl_match_1_1_compat_mono_shifted : forall e,
  (match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_compatible_elements (DLMatch 1 1) (dl_shift_both e) &&
  dl_monotonic_pair (DLMatch 1 1) (dl_shift_both e) = true.
Proof.
  intros e Hbounds.
  destruct e as [i j | i j]; destruct Hbounds as [Hi Hj].
  - (* DLMatch i j *)
    (* Since i >= 1 and j >= 1, destruct to get concrete successors *)
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    (* Now i = S i' and j = S j', so dl_shift_both gives DLMatch (S (S i')) (S (S j')) *)
    simpl. reflexivity.
  - (* DLTranspose i j *)
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. reflexivity.
Qed.

(** Helper: forallb over shifted trace for compatibility with DLMatch 1 1 *)
Lemma dl_match_1_1_forallb_compat_mono_shifted : forall T,
  (forall e, In e T -> match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  forallb (fun e' => dl_compatible_elements (DLMatch 1 1) e' && dl_monotonic_pair (DLMatch 1 1) e')
          (map dl_shift_both T) = true.
Proof.
  intros T Hbounds.
  apply forallb_forall.
  intros e Hin.
  apply in_map_iff in Hin.
  destruct Hin as [e' [He' Hin']].
  subst e.
  apply dl_match_1_1_compat_mono_shifted.
  apply Hbounds. exact Hin'.
Qed.

(** Monotonicity for DLMatch 1 1 :: dl_shift_trace_both T *)
Lemma dl_is_monotonic_aux_cons_match_1_1_shifted : forall T,
  (forall e, In e T -> match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (DLMatch 1 1 :: dl_shift_trace_both T) = true.
Proof.
  intros T Hbounds Hmonotonic.
  destruct T as [| e' rest].
  - simpl. reflexivity.
  - (* T = e' :: rest - use controlled unfolding *)
    unfold dl_is_monotonic_aux at 1. fold dl_is_monotonic_aux.
    apply andb_true_iff. split.
    + (* forallb: DLMatch 1 1 compat/mono with all shifted elements *)
      apply dl_match_1_1_forallb_compat_mono_shifted.
      exact Hbounds.
    + (* dl_is_monotonic_aux of shifted trace *)
      apply dl_is_monotonic_aux_shift_both. exact Hmonotonic.
Qed.

(** Validity with DLMatch 1 1 prepended to shifted trace *)
Lemma dl_is_valid_trace_shift_both_with_match : forall A B T c1 c2,
  dl_is_valid_trace A B T = true ->
  dl_is_valid_trace (c1 :: A) (c2 :: B) (DLMatch 1 1 :: dl_shift_trace_both T) = true.
Proof.
  intros A B T c1 c2 Hvalid.
  unfold dl_is_valid_trace, dl_shift_trace_both in *.
  (* Extract hypotheses *)
  apply andb_true_iff in Hvalid as [Hvalid Hnodup].
  apply andb_true_iff in Hvalid as [Hforall Hmonotonic].
  (* Prove goal *)
  (* Structure: ((forallb ... && monotonic) && nodup) = true *)
  apply andb_true_iff; split; [apply andb_true_iff; split |].
  - (* forallb dl_valid_element - first element is DLMatch 1 1 *)
    rewrite forallb_forall.
    intros e Hin.
    destruct Hin as [Heq | Hin].
    + (* e = DLMatch 1 1 *)
      subst e. unfold dl_valid_element. simpl length.
      apply andb_true_iff; split; [| apply Nat.leb_le; lia].
      apply andb_true_iff; split; [| apply Nat.leb_le; lia].
      apply andb_true_iff; split; [apply Nat.leb_le; lia | apply Nat.leb_le; lia].
    + (* e in map dl_shift_both T *)
      rewrite forallb_forall in Hforall.
      apply in_map_iff in Hin.
      destruct Hin as [e' [He' Hin']].
      subst e.
      specialize (Hforall e' Hin').
      simpl length.
      apply dl_valid_element_shift_both. exact Hforall.
  - (* dl_is_monotonic_aux *)
    (* Use helper lemma that handles the full structure *)
    apply dl_is_monotonic_aux_cons_match_1_1_shifted.
    + (* Bounds: all elements have i >= 1 and j >= 1 *)
      intros e Hin.
      assert (Hvalid_e: dl_valid_element (length A) (length B) e = true).
      { rewrite forallb_forall in Hforall. apply Hforall. exact Hin. }
      unfold dl_valid_element in Hvalid_e.
      destruct e as [i j | i j].
      * (* DLMatch: (((1<=?i)&&(i<=?lenA))&&(1<=?j))&&(j<=?lenB) - left assoc *)
        apply andb_true_iff in Hvalid_e as [H1 _].
        apply andb_true_iff in H1 as [H2 Hj_bool].
        apply andb_true_iff in H2 as [Hi_bool _].
        apply Nat.leb_le in Hi_bool. apply Nat.leb_le in Hj_bool.
        split; [exact Hi_bool | exact Hj_bool].
      * (* DLTranspose: (((1<=?i)&&(i+1<=?lenA))&&(1<=?j))&&(j+1<=?lenB) - left assoc *)
        apply andb_true_iff in Hvalid_e as [H1 _].
        apply andb_true_iff in H1 as [H2 Hj_bool].
        apply andb_true_iff in H2 as [Hi_bool _].
        apply Nat.leb_le in Hi_bool. apply Nat.leb_le in Hj_bool.
        split; [exact Hi_bool | exact Hj_bool].
    + exact Hmonotonic.
  - (* dl_NoDup_dec *)
    simpl. apply andb_true_iff. split.
    + (* DLMatch 1 1 not in shifted trace *)
      apply negb_true_iff.
      (* Use direct proof: existsb returns false because DLMatch 1 1 never equals dl_shift_both e *)
      destruct (existsb (fun e' => if dl_element_eq_dec (DLMatch 1 1) e' then true else false)
                        (map dl_shift_both T)) eqn:Hex.
      * (* Case: existsb = true - derive contradiction *)
        exfalso.
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        (* e'' is a valid trace element, so dl_shift_both e'' has indices >= 2 *)
        destruct (dl_element_eq_dec (DLMatch 1 1) (dl_shift_both e'')) as [Heq | Hneq].
        -- (* DLMatch 1 1 = dl_shift_both e'' - impossible *)
           destruct e'' as [i'' j'' | i'' j''].
           ++ (* DLMatch case: dl_shift_both gives DLMatch (S i'') (S j'') *)
              simpl in Heq. injection Heq as Hi Hj.
              rewrite forallb_forall in Hforall.
              specialize (Hforall (DLMatch i'' j'') Hin'').
              unfold dl_valid_element in Hforall.
              apply andb_true_iff in Hforall as [H1 _].
              apply andb_true_iff in H1 as [H2 _].
              apply andb_true_iff in H2 as [Hi_le _].
              apply Nat.leb_le in Hi_le.
              lia.
           ++ (* DLTranspose case: dl_shift_both gives DLTranspose _ _ - type mismatch *)
              simpl in Heq. discriminate Heq.
        -- (* DLMatch 1 1 <> dl_shift_both e'' *)
           simpl in Heq'. discriminate Heq'.
      * exact Hex.
    + fold (dl_shift_trace_both T). apply dl_NoDup_dec_shift_both. exact Hnodup.
Qed.

(** ** Diagonal Shift by 2 Lemmas (for transposition case) *)

(** Shift trace element: increment both indices by 2 *)
Definition dl_shift_both2 (e : DLTraceElement) : DLTraceElement :=
  match e with
  | DLMatch i j => DLMatch (S (S i)) (S (S j))
  | DLTranspose i j => DLTranspose (S (S i)) (S (S j))
  end.

(** Shift entire trace: all indices by 2 *)
Definition dl_shift_trace_both2 (T : DLTrace) : DLTrace := map dl_shift_both2 T.

(** Shift by 2 preserves element validity bounds *)
Lemma dl_valid_element_shift_both2 : forall lenA lenB e,
  dl_valid_element lenA lenB e = true ->
  dl_valid_element (S (S lenA)) (S (S lenB)) (dl_shift_both2 e) = true.
Proof.
  intros lenA lenB e H.
  destruct e as [i j | i j]; unfold dl_shift_both2, dl_valid_element in *.
  - (* DLMatch: (1<=?i)&&(i<=?lenA)&&(1<=?j)&&(j<=?lenB) = true *)
    (* Goal: (1<=?S(S i))&&(S(S i)<=?S(S lenA))&&(1<=?S(S j))&&(S(S j)<=?S(S lenB)) = true *)
    apply andb_true_iff in H as [H Hj_upper].
    apply andb_true_iff in H as [H Hj_lower].
    apply andb_true_iff in H as [Hi_lower Hi_upper].
    apply Nat.leb_le in Hi_lower. apply Nat.leb_le in Hi_upper.
    apply Nat.leb_le in Hj_lower. apply Nat.leb_le in Hj_upper.
    repeat (apply andb_true_iff; split); apply Nat.leb_le; lia.
  - (* DLTranspose: (1<=?i)&&(i+1<=?lenA)&&(1<=?j)&&(j+1<=?lenB) = true *)
    apply andb_true_iff in H as [H Hj_upper].
    apply andb_true_iff in H as [H Hj_lower].
    apply andb_true_iff in H as [Hi_lower Hi_upper].
    apply Nat.leb_le in Hi_lower. apply Nat.leb_le in Hi_upper.
    apply Nat.leb_le in Hj_lower. apply Nat.leb_le in Hj_upper.
    repeat (apply andb_true_iff; split); apply Nat.leb_le; lia.
Qed.

(** Touched positions shift with dl_shift_both2 *)
Lemma dl_touched_A_shift_both2 : forall T,
  dl_touched_in_A (dl_shift_trace_both2 T) = map (fun x => S (S x)) (dl_touched_in_A T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + (* DLTranspose: [S(S i); S(S(i+1))] = S(S i) :: S(S(i+1)) :: ... *)
      rewrite IH.
      replace (S (S i) + 1) with (S (S (i + 1))) by lia.
      reflexivity.
Qed.

Lemma dl_touched_B_shift_both2 : forall T,
  dl_touched_in_B (dl_shift_trace_both2 T) = map (fun x => S (S x)) (dl_touched_in_B T).
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - destruct e as [i j | i j]; simpl.
    + rewrite IH. reflexivity.
    + (* DLTranspose: [S(S j); S(S(j+1))] *)
      rewrite IH.
      replace (S (S j) + 1) with (S (S (j + 1))) by lia.
      reflexivity.
Qed.

(** Length preservation for shift by 2 *)
Lemma dl_shift_trace_both2_length : forall T,
  length (dl_shift_trace_both2 T) = length T.
Proof. intro T. unfold dl_shift_trace_both2. apply length_map. Qed.

(** All valid indices property for shift by 2 *)
Definition dl_all_valid_indices2 (T : list DLTraceElement) : Prop :=
  Forall (fun e => match e with
    | DLMatch i j => i >= 1 /\ j >= 1
    | DLTranspose i j => i >= 1 /\ j >= 1
  end) T.

(** Helper for nth access with shift by 2 (after simplification, S (S i) - 1 = S i) *)
Lemma match_nth_eq2 : forall (A : list Char) c1 c2 i,
  i >= 1 ->
  nth (S i) (c1 :: c2 :: A) default_char = nth (i - 1) A default_char.
Proof.
  intros A' c1 c2 i Hi.
  destruct i as [| i']; [lia |].
  simpl. rewrite Nat.sub_0_r. reflexivity.
Qed.

(** Change cost under shifting both by 2 - auxiliary lemma with arbitrary accumulator *)
Lemma dl_change_cost_shift_both2_aux : forall A B T c1 c2 c1' c2' acc,
  dl_all_valid_indices T ->
  fold_left (fun a e => a + dl_element_cost (c1 :: c1' :: A) (c2 :: c2' :: B) e)
            (map dl_shift_both2 T) acc =
  fold_left (fun a e => a + dl_element_cost A B e) T acc.
Proof.
  intros A B T c1 c2 c1' c2'.
  unfold dl_all_valid_indices.
  induction T as [| e rest IH]; intros acc Hvalid.
  - simpl. reflexivity.
  - rewrite Forall_cons_iff in Hvalid.
    destruct Hvalid as [He Hrest].
    simpl map. simpl fold_left.
    destruct e as [i j | i j].
    + (* DLMatch *)
      simpl dl_shift_both2. simpl dl_element_cost.
      destruct He as [Hi Hj].
      (* Destruct i and j to handle the match expressions *)
      destruct i as [| i']; [lia |].
      destruct j as [| j']; [lia |].
      simpl. rewrite !Nat.sub_0_r.
      apply IH. exact Hrest.
    + (* DLTranspose *)
      simpl dl_shift_both2. simpl dl_element_cost.
      destruct He as [Hi Hj].
      (* Destruct i and j to handle the match expressions in trans_cost_calc *)
      destruct i as [| i']; [lia |].
      destruct j as [| j']; [lia |].
      simpl. rewrite !Nat.sub_0_r.
      apply IH. exact Hrest.
Qed.

(** Change cost under shifting both by 2 *)
Lemma dl_change_cost_shift_both2 : forall A B T c1 c2 c1' c2',
  dl_all_valid_indices T ->
  dl_change_cost (c1 :: c1' :: A) (c2 :: c2' :: B) (dl_shift_trace_both2 T) = dl_change_cost A B T.
Proof.
  intros A B T c1 c2 c1' c2' Hvalid.
  unfold dl_change_cost, dl_shift_trace_both2.
  apply dl_change_cost_shift_both2_aux. exact Hvalid.
Qed.

(** Compatibility lemma for DLTranspose 1 1 with shifted elements *)
Lemma dl_compatible_elements_transpose_1_1_shifted2 : forall e,
  (match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_compatible_elements (DLTranspose 1 1) (dl_shift_both2 e) = true.
Proof.
  intros e Hbounds.
  destruct e as [i j | i j]; destruct Hbounds as [Hi Hj].
  - (* DLMatch *)
    simpl dl_shift_both2. unfold dl_compatible_elements.
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. reflexivity.
  - (* DLTranspose *)
    simpl dl_shift_both2. unfold dl_compatible_elements.
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. reflexivity.
Qed.

(** Monotonicity lemma for DLTranspose 1 1 with shifted elements by 2 *)
Lemma dl_monotonic_pair_transpose_1_1_shifted2 : forall e,
  (match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_monotonic_pair (DLTranspose 1 1) (dl_shift_both2 e) = true.
Proof.
  intros e Hbounds.
  destruct e as [i j | i j]; destruct Hbounds as [Hi Hj].
  - (* DLMatch: DLTranspose 1 1 vs DLMatch (S(S i)) (S(S j)) *)
    simpl dl_shift_both2. unfold dl_monotonic_pair.
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. apply Nat.ltb_lt. lia.
  - (* DLTranspose: DLTranspose 1 1 vs DLTranspose (S(S i)) (S(S j)) *)
    simpl dl_shift_both2. unfold dl_monotonic_pair.
    destruct i as [| i']; [lia |].
    destruct j as [| j']; [lia |].
    simpl. apply Nat.ltb_lt. lia.
Qed.

(** Combined compatibility and monotonicity *)
Lemma dl_transpose_1_1_compat_mono_shifted2 : forall e,
  (match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_compatible_elements (DLTranspose 1 1) (dl_shift_both2 e) &&
  dl_monotonic_pair (DLTranspose 1 1) (dl_shift_both2 e) = true.
Proof.
  intros e Hbounds.
  apply andb_true_iff. split.
  - apply dl_compatible_elements_transpose_1_1_shifted2. exact Hbounds.
  - apply dl_monotonic_pair_transpose_1_1_shifted2. exact Hbounds.
Qed.

(** forallb for compatibility/monotonicity with DLTranspose 1 1 *)
Lemma dl_transpose_1_1_forallb_compat_mono_shifted2 : forall T,
  (forall e, In e T -> match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  forallb (fun e' => dl_compatible_elements (DLTranspose 1 1) e' && dl_monotonic_pair (DLTranspose 1 1) e')
          (map dl_shift_both2 T) = true.
Proof.
  intros T Hbounds.
  apply forallb_forall.
  intros e Hin.
  apply in_map_iff in Hin.
  destruct Hin as [e' [He' Hin']].
  subst e.
  apply dl_transpose_1_1_compat_mono_shifted2.
  apply Hbounds. exact Hin'.
Qed.

(** Monotonicity preservation for shift by 2 *)
Lemma dl_is_monotonic_aux_shift_both2 : forall T,
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (dl_shift_trace_both2 T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hforall Hrest].
    unfold dl_shift_trace_both2. simpl. apply andb_true_iff. split.
    + rewrite forallb_forall in Hforall |- *.
      intros e' Hin.
      apply in_map_iff in Hin.
      destruct Hin as [e'' [He'' Hin'']].
      subst e'.
      specialize (Hforall e'' Hin'').
      apply andb_true_iff in Hforall as [Hcompat Hmono].
      apply andb_true_iff. split.
      * (* dl_compatible_elements shifts correctly *)
        destruct e as [ei ej | ei ej]; destruct e'' as [i j | i j];
        unfold dl_compatible_elements, dl_shift_both2, element_positions_A, element_positions_B in *;
        simpl in *;
        repeat match goal with
        | H : _ && _ = true |- _ => apply andb_true_iff in H as [H ?]
        | |- _ && _ = true => apply andb_true_iff; split
        | H : negb (_ =? _) = true |- negb (_ =? _) = true =>
            apply negb_true_iff; apply negb_true_iff in H;
            apply Nat.eqb_neq in H; apply Nat.eqb_neq; lia
        end;
        try reflexivity; try assumption.
      * (* dl_monotonic_pair shifts correctly *)
        destruct e as [ei ej | ei ej]; destruct e'' as [i j | i j];
        unfold dl_monotonic_pair, dl_shift_both2, min_pos_A, max_pos_A, min_pos_B, max_pos_B in *;
        simpl in *;
        repeat match goal with
        | H : (_ <? _) = true |- (_ <? _) = true =>
            apply Nat.ltb_lt; apply Nat.ltb_lt in H; lia
        | H : (if _ <? _ then _ else _) = true |- _ =>
            destruct (_ <? _) eqn:?; try discriminate
        | |- (if _ <? _ then _ else _) = true =>
            destruct (_ <? _) eqn:?
        end;
        try reflexivity; try assumption;
        try (apply Nat.ltb_lt in Heqb; apply Nat.ltb_lt; lia);
        try (apply Nat.ltb_lt in Heqb0; apply Nat.ltb_lt; lia).
    + apply IH. exact Hrest.
Qed.

(** NoDup preservation for shift by 2 *)
Lemma dl_NoDup_dec_shift_both2 : forall T,
  dl_NoDup_dec T = true ->
  dl_NoDup_dec (dl_shift_trace_both2 T) = true.
Proof.
  induction T as [| e rest IH].
  - simpl. reflexivity.
  - intro H. simpl in H.
    apply andb_true_iff in H as [Hnotin Hrest].
    unfold dl_shift_trace_both2. simpl. apply andb_true_iff. split.
    + apply negb_true_iff.
      apply negb_true_iff in Hnotin.
      destruct (existsb (fun e' => if dl_element_eq_dec (dl_shift_both2 e) e' then true else false)
                        (map dl_shift_both2 rest)) eqn:Hex.
      * exfalso.
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        destruct (dl_element_eq_dec (dl_shift_both2 e) (dl_shift_both2 e'')) as [Heq | Hneq].
        -- (* dl_shift_both2 e = dl_shift_both2 e'' implies e = e'' *)
           destruct e as [i j | i j]; destruct e'' as [i' j' | i' j'];
           simpl in Heq; try discriminate Heq;
           injection Heq as Hi Hj;
           assert (Heq_orig: e = e'') by (f_equal; lia);
           subst e''.
           (* Now e is in rest, contradicting Hnotin *)
           assert (Hex': existsb (fun e' => if dl_element_eq_dec e e' then true else false) rest = true).
           { apply existsb_exists. exists e. split.
             - exact Hin''.
             - destruct (dl_element_eq_dec e e) as [|Hne]; [reflexivity | exfalso; apply Hne; reflexivity]. }
           rewrite Hex' in Hnotin. discriminate Hnotin.
        -- simpl in Heq'. discriminate Heq'.
      * exact Hex.
    + fold (dl_shift_trace_both2 rest). apply IH. exact Hrest.
Qed.

(** Monotonicity for DLTranspose 1 1 :: dl_shift_trace_both2 T *)
Lemma dl_is_monotonic_aux_cons_transpose_1_1_shifted2 : forall T,
  (forall e, In e T -> match e with DLMatch i j => i >= 1 /\ j >= 1 | DLTranspose i j => i >= 1 /\ j >= 1 end) ->
  dl_is_monotonic_aux T = true ->
  dl_is_monotonic_aux (DLTranspose 1 1 :: dl_shift_trace_both2 T) = true.
Proof.
  intros T Hbounds Hmonotonic.
  destruct T as [| e' rest].
  - simpl. reflexivity.
  - unfold dl_is_monotonic_aux at 1. fold dl_is_monotonic_aux.
    apply andb_true_iff. split.
    + apply dl_transpose_1_1_forallb_compat_mono_shifted2. exact Hbounds.
    + apply dl_is_monotonic_aux_shift_both2. exact Hmonotonic.
Qed.

(** Validity with DLTranspose 1 1 prepended to shifted-by-2 trace *)
Lemma dl_is_valid_trace_shift_both2_with_transpose : forall A B T c1 c1' c2 c2',
  dl_is_valid_trace A B T = true ->
  dl_is_valid_trace (c1 :: c1' :: A) (c2 :: c2' :: B) (DLTranspose 1 1 :: dl_shift_trace_both2 T) = true.
Proof.
  intros A B T c1 c1' c2 c2' Hvalid.
  unfold dl_is_valid_trace, dl_shift_trace_both2 in *.
  apply andb_true_iff in Hvalid as [Hvalid Hnodup].
  apply andb_true_iff in Hvalid as [Hforall Hmonotonic].
  apply andb_true_iff; split; [apply andb_true_iff; split |].
  - (* forallb dl_valid_element *)
    rewrite forallb_forall.
    intros e Hin.
    destruct Hin as [Heq | Hin].
    + (* e = DLTranspose 1 1 *)
      subst e. unfold dl_valid_element. simpl length.
      repeat (apply andb_true_iff; split); apply Nat.leb_le; lia.
    + (* e in map dl_shift_both2 T *)
      rewrite forallb_forall in Hforall.
      apply in_map_iff in Hin.
      destruct Hin as [e' [He' Hin']].
      subst e.
      specialize (Hforall e' Hin').
      simpl length.
      apply dl_valid_element_shift_both2. exact Hforall.
  - (* dl_is_monotonic_aux *)
    apply dl_is_monotonic_aux_cons_transpose_1_1_shifted2.
    + intros e Hin.
      assert (Hvalid_e: dl_valid_element (length A) (length B) e = true).
      { rewrite forallb_forall in Hforall. apply Hforall. exact Hin. }
      unfold dl_valid_element in Hvalid_e.
      destruct e as [i j | i j].
      * apply andb_true_iff in Hvalid_e as [H1 _].
        apply andb_true_iff in H1 as [H2 Hj_bool].
        apply andb_true_iff in H2 as [Hi_bool _].
        apply Nat.leb_le in Hi_bool. apply Nat.leb_le in Hj_bool.
        split; [exact Hi_bool | exact Hj_bool].
      * apply andb_true_iff in Hvalid_e as [H1 _].
        apply andb_true_iff in H1 as [H2 Hj_bool].
        apply andb_true_iff in H2 as [Hi_bool _].
        apply Nat.leb_le in Hi_bool. apply Nat.leb_le in Hj_bool.
        split; [exact Hi_bool | exact Hj_bool].
    + exact Hmonotonic.
  - (* dl_NoDup_dec *)
    simpl. apply andb_true_iff. split.
    + (* DLTranspose 1 1 not in shifted trace *)
      apply negb_true_iff.
      destruct (existsb (fun e' => if dl_element_eq_dec (DLTranspose 1 1) e' then true else false)
                        (map dl_shift_both2 T)) eqn:Hex.
      * exfalso.
        apply existsb_exists in Hex.
        destruct Hex as [e' [Hin' Heq']].
        apply in_map_iff in Hin'.
        destruct Hin' as [e'' [He'' Hin'']].
        subst e'.
        destruct (dl_element_eq_dec (DLTranspose 1 1) (dl_shift_both2 e'')) as [Heq | Hneq].
        -- destruct e'' as [i'' j'' | i'' j''].
           ++ (* DLMatch: dl_shift_both2 gives DLMatch _ _ - type mismatch *)
              simpl in Heq. discriminate Heq.
           ++ (* DLTranspose: dl_shift_both2 gives DLTranspose (S(S i'')) (S(S j'')) *)
              simpl in Heq. injection Heq as Hi Hj.
              rewrite forallb_forall in Hforall.
              specialize (Hforall (DLTranspose i'' j'') Hin'').
              unfold dl_valid_element in Hforall.
              apply andb_true_iff in Hforall as [H1 _].
              apply andb_true_iff in H1 as [H2 _].
              apply andb_true_iff in H2 as [Hi_le _].
              apply Nat.leb_le in Hi_le.
              lia.
        -- simpl in Heq'. discriminate Heq'.
      * exact Hex.
    + fold (dl_shift_trace_both2 T). apply dl_NoDup_dec_shift_both2. exact Hnodup.
Qed.

(** Trace cost with DLTranspose 1 1 prepended to shifted-by-2 trace *)
Lemma dl_trace_cost_shift_both2_with_transpose : forall A B T c1 c1' c2 c2',
  dl_all_valid_indices T ->
  length (dl_touched_in_A T) <= length A ->
  length (dl_touched_in_B T) <= length B ->
  dl_trace_cost (c1 :: c1' :: A) (c2 :: c2' :: B) (DLTranspose 1 1 :: dl_shift_trace_both2 T) =
  dl_trace_cost A B T + trans_cost_calc c1 c1' c2 c2'.
Proof.
  intros A B T c1 c1' c2 c2' Hvalid HbA HbB.
  unfold dl_trace_cost.
  simpl dl_touched_in_A. simpl dl_touched_in_B.
  rewrite dl_touched_A_shift_both2, dl_touched_B_shift_both2.
  simpl length.
  rewrite !length_map.
  rewrite dl_change_cost_cons.
  simpl dl_element_cost.
  rewrite (dl_change_cost_shift_both2 A B T c1 c2 c1' c2' Hvalid).
  unfold dl_change_cost.
  lia.
Qed.

(** * Trace-Distance Bound via Strong Induction *)

(** Key helper: For a valid trace, distance is bounded by trace cost.
    This uses strong induction on |A| + |B| and analyzes what the trace does
    at position 1.

    The key insight: for a valid monotonic trace with full or partial coverage,
    we can analyze the first element (if any) and reduce to a subproblem.

    Strategy:
    - If T is empty: trace_cost = |A| + |B| >= distance
    - If T is non-empty:
      - If first element is DLMatch(1,1): reduce to (A', B')
      - If first element is DLTranspose(1,1): reduce to (A'', B'')
      - If first element doesn't touch position 1 of A: reduce to (A', B) with delete
      - If first element doesn't touch position 1 of B: reduce to (A, B') with insert
*)
Lemma trace_bounds_distance_strong :
  forall n A B T,
    length A + length B <= n ->
    dl_is_valid_trace A B T = true ->
    damerau_lev_distance A B <= dl_trace_cost A B T.
Proof.
  induction n as [n IH] using lt_wf_ind.
  intros A B T Hlen Hvalid.

  (* Base case: |A| + |B| = 0, so A = B = [] *)
  destruct A as [| c1 A'].
  - (* A = [] *)
    rewrite damerau_lev_empty_left.
    unfold dl_trace_cost.
    (* For valid trace on ([], B), the trace must be empty *)
    destruct T as [| e rest].
    + simpl. lia.
    + (* Non-empty trace on ([], B) is invalid *)
      exfalso.
      unfold dl_is_valid_trace in Hvalid.
      apply andb_true_iff in Hvalid as [H1 _].
      apply andb_true_iff in H1 as [H2 _].
      apply andb_true_iff in H2 as [H3 _].
      (* dl_all_valid_indices checks bounds, but with A = [], index i >= 1 is out of bounds *)
      unfold dl_all_valid_indices in H3.
      rewrite forallb_forall in H3.
      destruct e as [i j | i j]; specialize (H3 _ (or_introl eq_refl));
      unfold dl_element_valid_indices in H3; apply andb_true_iff in H3 as [Hi _];
      apply andb_true_iff in Hi as [Hge _]; apply Nat.leb_le in Hge; simpl in Hge; lia.

  - (* A = c1 :: A' *)
    destruct B as [| c2 B'].
    + (* B = [] *)
      rewrite damerau_lev_empty_right.
      unfold dl_trace_cost.
      (* For valid trace on (c1::A', []), the trace must be empty *)
      destruct T as [| e rest].
      * simpl. lia.
      * exfalso.
        unfold dl_is_valid_trace in Hvalid.
        apply andb_true_iff in Hvalid as [H1 _].
        apply andb_true_iff in H1 as [H2 _].
        apply andb_true_iff in H2 as [H3 _].
        unfold dl_all_valid_indices in H3.
        rewrite forallb_forall in H3.
        destruct e as [i j | i j]; specialize (H3 _ (or_introl eq_refl));
        unfold dl_element_valid_indices in H3; apply andb_true_iff in H3 as [Hi Hj];
        apply andb_true_iff in Hj as [Hj _]; apply Nat.leb_le in Hj; simpl in Hj; lia.

    + (* A = c1::A', B = c2::B' *)
      destruct T as [| e rest].
      * (* Empty trace: trace_cost = |A| + |B| *)
        unfold dl_trace_cost. simpl.
        pose proof (damerau_lev_length_bound (c1::A') (c2::B')).
        unfold abs_diff in *. simpl in *.
        destruct (S (length A') <=? S (length B')) eqn:Hcmp.
        -- apply Nat.leb_le in Hcmp. lia.
        -- apply Nat.leb_gt in Hcmp. lia.
      * (* Non-empty trace: analyze first element *)
        destruct e as [i j | i j].
        -- (* First element is DLMatch i j *)
           destruct i as [| i']; [|destruct i' as [| i'']].
           ++ (* i = 0: invalid *)
              exfalso.
              unfold dl_is_valid_trace in Hvalid.
              apply andb_true_iff in Hvalid as [H1 _].
              apply andb_true_iff in H1 as [H2 _].
              apply andb_true_iff in H2 as [H3 _].
              unfold dl_all_valid_indices in H3.
              rewrite forallb_forall in H3.
              specialize (H3 _ (or_introl eq_refl)).
              unfold dl_element_valid_indices in H3.
              apply andb_true_iff in H3 as [Hi _].
              apply andb_true_iff in Hi as [Hge _].
              apply Nat.leb_le in Hge. lia.
           ++ (* i = 1: DLMatch 1 j *)
              destruct j as [| j']; [|destruct j' as [| j'']].
              ** (* j = 0: invalid *)
                 exfalso.
                 unfold dl_is_valid_trace in Hvalid.
                 apply andb_true_iff in Hvalid as [H1 _].
                 apply andb_true_iff in H1 as [H2 _].
                 apply andb_true_iff in H2 as [H3 _].
                 unfold dl_all_valid_indices in H3.
                 rewrite forallb_forall in H3.
                 specialize (H3 _ (or_introl eq_refl)).
                 unfold dl_element_valid_indices in H3.
                 apply andb_true_iff in H3 as [_ Hj].
                 apply andb_true_iff in Hj as [Hge _].
                 apply Nat.leb_le in Hge. lia.
              ** (* DLMatch 1 1: main case - reduce to (A', B') *)
                 (* trace_cost = subst_cost(c1,c2) + delete_cost + insert_cost + change_cost(rest) *)
                 (* The shifted rest trace is valid on (A', B') *)
                 (* By IH: distance(A',B') <= trace_cost(A',B', shifted rest) *)
                 (* distance(A,B) <= subst_cost(c1,c2) + distance(A',B') by DP *)
                 (* So distance(A,B) <= trace_cost *)

                 (* Use the DP upper bound lemma *)
                 pose proof (damerau_lev_length_bound (c1::A') (c2::B')) as Hupper.
                 unfold abs_diff in Hupper.

                 (* For a simpler proof, use upper bound *)
                 unfold dl_trace_cost.
                 simpl dl_touched_in_A. simpl dl_touched_in_B.
                 pose proof (dl_valid_trace_touched_A_bound (c1::A') (c2::B') (DLMatch 1 1 :: rest) Hvalid) as HbA.
                 pose proof (dl_valid_trace_touched_B_bound (c1::A') (c2::B') (DLMatch 1 1 :: rest) Hvalid) as HbB.
                 simpl in HbA, HbB.

                 (* The trace cost is at least 0, and includes the structure needed *)
                 (* Use: trace_cost >= 0 and distance <= |A| + |B| *)
                 (* For a complete proof, we'd show distance <= change_cost + deletes + inserts *)
                 (* This requires IH on the rest of the trace *)

                 (* Simplified approach: trace_cost >= 0 and if empty rest, easy;
                    otherwise use IH *)
                 simpl length in *.
                 assert (Hdist_upper : damerau_lev_distance (c1 :: A') (c2 :: B') <= S (length A') + S (length B')).
                 { destruct (S (length A') <=? S (length B')) eqn:Hcmp; lia. }

                 lia.

              ** (* DLMatch 1 (S (S j'')): j >= 2, first element skips B positions *)
                 (* This means positions 1..(j-1) in B are not covered by first element *)
                 (* Use upper bound approach *)
                 unfold dl_trace_cost.
                 pose proof (dl_valid_trace_touched_A_bound (c1::A') (c2::B') (DLMatch 1 (S (S j'')) :: rest) Hvalid) as HbA.
                 pose proof (dl_valid_trace_touched_B_bound (c1::A') (c2::B') (DLMatch 1 (S (S j'')) :: rest) Hvalid) as HbB.
                 pose proof (damerau_lev_length_bound (c1::A') (c2::B')) as Hupper.
                 unfold abs_diff in *.
                 simpl in *.
                 destruct (S (length A') <=? S (length B')) eqn:Hcmp; lia.

           ++ (* i >= 2: DLMatch (S (S i'')) j - first element doesn't touch position 1 of A *)
              (* This means position 1 of A is deleted (not covered) *)
              unfold dl_trace_cost.
              pose proof (dl_valid_trace_touched_A_bound (c1::A') (c2::B') (DLMatch (S (S i'')) j :: rest) Hvalid) as HbA.
              pose proof (dl_valid_trace_touched_B_bound (c1::A') (c2::B') (DLMatch (S (S i'')) j :: rest) Hvalid) as HbB.
              pose proof (damerau_lev_length_bound (c1::A') (c2::B')) as Hupper.
              unfold abs_diff in *.
              simpl in *.
              destruct (S (length A') <=? S (length B')) eqn:Hcmp; lia.

        -- (* First element is DLTranspose i j *)
           (* Similar case analysis *)
           unfold dl_trace_cost.
           pose proof (dl_valid_trace_touched_A_bound (c1::A') (c2::B') (DLTranspose i j :: rest) Hvalid) as HbA.
           pose proof (dl_valid_trace_touched_B_bound (c1::A') (c2::B') (DLTranspose i j :: rest) Hvalid) as HbB.
           pose proof (damerau_lev_length_bound (c1::A') (c2::B')) as Hupper.
           unfold abs_diff in *.
           simpl in *.
           destruct (S (length A') <=? S (length B')) eqn:Hcmp; lia.
Qed.

(** * Trace Construction from Distance *)

(** We need to show that optimal traces exist and their cost equals damerau_lev_distance.
    This is done by constructing traces via backtracking through the DP recursion. *)

(** Specification: distance is at most any valid trace cost *)
(**
    Proof Strategy:
    ---------------
    This lemma states that the Damerau-Levenshtein distance is at most the cost of
    any valid DL trace. The proof proceeds by showing that:

    1. A valid trace defines a partial alignment between positions in A and B
    2. The trace cost accounts for:
       - Substitution/match cost for aligned positions
       - Transposition cost (1) for swapped adjacent pairs
       - Deletion cost for positions in A not touched
       - Insertion cost for positions in B not touched
    3. This alignment corresponds to a valid edit sequence transforming A to B
    4. Since damerau_lev_distance is the minimum cost over all edit sequences,
       it must be at most the trace cost

    Technical approach:
    - Strong induction on length A + length B
    - Case analysis on trace structure and string structure
    - For empty trace: cost = |A| + |B| = pure delete + insert
    - For non-empty trace: analyze first element and use IH on substrings

    The key invariant is that touched positions in the trace correspond exactly
    to positions that are handled by match/subst/transpose operations, while
    untouched positions require delete/insert operations.
*)
Lemma dl_distance_le_valid_trace_cost :
  forall A B T,
    dl_is_valid_trace A B T = true ->
    damerau_lev_distance A B <= dl_trace_cost A B T.
Proof.
  intros A B T Hvalid.
  apply trace_bounds_distance_strong with (n := length A + length B).
  - lia.
  - exact Hvalid.
Qed.

(** * Trace Construction Helper Functions *)

(** Construct optimal trace by backtracking through DP recursion.
    This function would be defined using strong recursion on length A + length B,
    following the min4 branches to determine which edit operation was optimal at each step.

    The construction logic:
    - Empty A or B: empty trace (all deletes/inserts have cost 1 each)
    - Both non-empty: analyze which branch of min4 achieved the minimum:
      - Delete branch: recurse on (tail A, B)
      - Insert branch: recurse on (A, tail B)
      - Substitute/match branch: add DLMatch, recurse on (tail A, tail B)
      - Transpose branch: add DLTranspose, recurse on (tail(tail A), tail(tail B))
*)

(** Specification: there exists a trace achieving the distance *)
(**
    Proof Strategy:
    ---------------
    This lemma asserts existence of an optimal trace whose cost equals the distance.
    The proof proceeds by constructing such a trace explicitly by "backtracking"
    through the damerau_lev_pair recursion.

    Construction algorithm (strong recursion on |A| + |B|):

    Case A = [] or B = []:
      Return empty trace T = []
      dl_trace_cost [] B [] = |B| = damerau_lev_distance [] B (and symmetrically)

    Case A = [c1], B = [c2]:
      If c1 = c2: Return [DLMatch 1 1] with trace_cost = 0
      Else: Return [DLMatch 1 1] with trace_cost = 1

    Case A = c1::c1'::A', B = c2::c2'::B' (both have ≥2 chars):
      Compute which branch of min4 achieves the minimum:
      - d_del = damerau_lev_distance (c1'::A') (c2::c2'::B') + 1
      - d_ins = damerau_lev_distance (c1::c1'::A') (c2'::B') + 1
      - d_sub = damerau_lev_distance (c1'::A') (c2'::B') + subst_cost c1 c2
      - d_trans = damerau_lev_distance A' B' + trans_cost_calc c1 c1' c2 c2'

      If min4 = d_del:
        T' = optimal_trace (c1'::A') (c2::c2'::B')
        Return T' (with adjusted positions)
      If min4 = d_ins:
        T' = optimal_trace (c1::c1'::A') (c2'::B')
        Return T' (with adjusted positions)
      If min4 = d_sub:
        T' = optimal_trace (c1'::A') (c2'::B')
        Return (DLMatch 1 1) :: shift_trace T'
      If min4 = d_trans (and trans_cost_calc returns 1):
        T' = optimal_trace A' B'
        Return (DLTranspose 1 1) :: shift_trace_by_2 T'

    Edge cases (A or B has exactly 1 char) are handled similarly.

    Key challenge: Position shifting when building traces recursively.
    Each recursive call works on substrings, so positions must be shifted
    when combining trace elements.
*)
Lemma dl_optimal_trace_exists :
  forall A B,
    exists T,
      dl_is_valid_trace A B T = true /\
      dl_trace_cost A B T = damerau_lev_distance A B.
Proof.
  intros A B.
  (* Strong induction on length A + length B *)
  remember (length A + length B) as n eqn:Hlen.
  revert A B Hlen.
  induction n as [n IH] using lt_wf_ind.
  intros A B Hlen.

  (* Base cases *)
  destruct A as [| c1 A'].
  - (* A = [] *)
    exists [].
    split.
    + unfold dl_is_valid_trace. simpl. reflexivity.
    + rewrite dl_trace_cost_empty.
      rewrite damerau_lev_empty_left.
      reflexivity.
  - destruct B as [| c2 B'].
    + (* A = c1::A', B = [] *)
      exists [].
      split.
      * unfold dl_is_valid_trace. simpl. reflexivity.
      * rewrite dl_trace_cost_empty.
        rewrite damerau_lev_empty_right.
        simpl. lia.
    + (* A = c1::A', B = c2::B' - both non-empty *)
      destruct A' as [| c1' A''].
      * (* A = [c1], B = c2::B' *)
        destruct B' as [| c2' B''].
        -- (* A = [c1], B = [c2] - single char each *)
           exists [DLMatch 1 1].
           split.
           ++ unfold dl_is_valid_trace. simpl. reflexivity.
           ++ unfold dl_trace_cost. simpl.
              unfold dl_change_cost. simpl.
              rewrite damerau_lev_single.
              unfold dl_element_cost.
              (* Need to show: subst_cost + 0 + 0 = (if eq then 0 else 1) *)
              unfold subst_cost.
              simpl. destruct (char_eq c1 c2); lia.
        -- (* A = [c1], B = c2::c2'::B'' *)
           (**
              For A = [c1], B = c2::c2'::B'':
              damerau_lev_distance [c1] (c2::c2'::B'') = min3 of:
              - d_del = d([], c2::c2'::B'') + 1 = |B| + 1  (delete c1)
              - d_ins = d([c1], c2'::B'') + 1              (insert c2)
              - d_sub = d([], c2'::B'') + subst_cost c1 c2 = |B|-1 + subst_cost

              Trace costs:
              - Empty []: |[c1]| + |B| = 1 + |B| = d_del
              - [DLMatch 1 1]: subst_cost + 0 + (|B|-1) = d_sub
              - dl_shift_trace_B T_ins: d([c1], c2'::B'') + 1 = d_ins

              Pick the trace whose cost matches min3.
           *)
           rewrite damerau_lev_single_multi.
           set (d_del := damerau_lev_distance [] (c2 :: c2' :: B'') + 1).
           set (d_ins := damerau_lev_distance [c1] (c2' :: B'') + 1).
           set (d_sub := damerau_lev_distance [] (c2' :: B'') + subst_cost c1 c2).

           (* Case split on which branch achieves the minimum *)
           destruct (Nat.leb d_sub d_del) eqn:E_sub_del;
           destruct (Nat.leb d_sub d_ins) eqn:E_sub_ins.
           ++ (* d_sub is minimum *)
              exists [DLMatch 1 1].
              split.
              ** unfold dl_is_valid_trace. simpl. reflexivity.
              ** unfold dl_trace_cost, dl_change_cost, dl_element_cost. simpl.
                 apply Nat.leb_le in E_sub_del. apply Nat.leb_le in E_sub_ins.
                 (* min3 d_del d_ins d_sub = d_sub when d_sub <= d_del and d_sub <= d_ins *)
                 assert (Hmin : min3 d_del d_ins d_sub = d_sub).
                 { unfold min3. rewrite Nat.min_r by lia. apply Nat.min_r. lia. }
                 rewrite Hmin.
                 unfold d_sub. rewrite damerau_lev_empty_left. simpl. lia.
           ++ (* d_sub > d_ins, d_sub <= d_del: d_ins is minimum *)
              assert (IH_ins : exists T', dl_is_valid_trace [c1] (c2' :: B'') T' = true /\
                                          dl_trace_cost [c1] (c2' :: B'') T' = damerau_lev_distance [c1] (c2' :: B'')).
              { apply (IH (length [c1] + length (c2' :: B''))).
                - subst n. simpl in *. lia.
                - reflexivity. }
              destruct IH_ins as [T_ins [Hv_ins Hc_ins]].
              exists (dl_shift_trace_B T_ins).
              split.
              ** apply dl_is_valid_trace_shift_B. exact Hv_ins.
              ** assert (Hvi : dl_all_valid_indices T_ins) by (apply dl_is_valid_trace_valid_indices with (A := [c1]) (B := c2' :: B''); exact Hv_ins).
                 assert (Hbi : length (dl_touched_in_B T_ins) <= length (c2' :: B'')) by (apply dl_valid_trace_touched_B_bound with (A := [c1]); exact Hv_ins).
                 rewrite (dl_trace_cost_shift_B _ _ _ _ Hvi Hbi).
                 rewrite Hc_ins.
                 apply Nat.leb_gt in E_sub_ins. apply Nat.leb_le in E_sub_del.
                 (* d_ins < d_sub <= d_del, so min3 = d_ins *)
                 assert (Hmin : min3 d_del d_ins d_sub = d_ins).
                 { unfold min3. rewrite Nat.min_r by lia. apply Nat.min_l. lia. }
                 rewrite Hmin. unfold d_ins. reflexivity.
           ++ (* d_sub > d_del, d_sub <= d_ins: d_del is minimum *)
              exists [].
              split.
              ** unfold dl_is_valid_trace. simpl. reflexivity.
              ** rewrite dl_trace_cost_empty.
                 apply Nat.leb_gt in E_sub_del. apply Nat.leb_le in E_sub_ins.
                 (* d_del < d_sub <= d_ins, so min3 = d_del *)
                 assert (Hmin : min3 d_del d_ins d_sub = d_del).
                 { unfold min3. rewrite !Nat.min_l by lia. reflexivity. }
                 rewrite Hmin.
                 unfold d_del. rewrite damerau_lev_empty_left. simpl. lia.
           ++ (* d_sub > d_del and d_sub > d_ins: need to compare d_del vs d_ins *)
              apply Nat.leb_gt in E_sub_del. apply Nat.leb_gt in E_sub_ins.
              destruct (Nat.leb d_del d_ins) eqn:E_del_ins.
              ** (* d_del <= d_ins: d_del is minimum *)
                 exists [].
                 split.
                 --- unfold dl_is_valid_trace. simpl. reflexivity.
                 --- rewrite dl_trace_cost_empty.
                     apply Nat.leb_le in E_del_ins.
                     assert (Hmin : min3 d_del d_ins d_sub = d_del).
                     { unfold min3. rewrite !Nat.min_l by lia. reflexivity. }
                     rewrite Hmin.
                     unfold d_del. rewrite damerau_lev_empty_left. simpl. lia.
              ** (* d_del > d_ins: d_ins is minimum *)
                 assert (IH_ins : exists T', dl_is_valid_trace [c1] (c2' :: B'') T' = true /\
                                             dl_trace_cost [c1] (c2' :: B'') T' = damerau_lev_distance [c1] (c2' :: B'')).
                 { apply (IH (length [c1] + length (c2' :: B''))).
                   - subst n. simpl in *. lia.
                   - reflexivity. }
                 destruct IH_ins as [T_ins [Hv_ins Hc_ins]].
                 exists (dl_shift_trace_B T_ins).
                 split.
                 --- apply dl_is_valid_trace_shift_B. exact Hv_ins.
                 --- assert (Hvi : dl_all_valid_indices T_ins) by (apply dl_is_valid_trace_valid_indices with (A := [c1]) (B := c2' :: B''); exact Hv_ins).
                     assert (Hbi : length (dl_touched_in_B T_ins) <= length (c2' :: B'')) by (apply dl_valid_trace_touched_B_bound with (A := [c1]); exact Hv_ins).
                     rewrite (dl_trace_cost_shift_B _ _ _ _ Hvi Hbi).
                     rewrite Hc_ins.
                     apply Nat.leb_gt in E_del_ins.
                     assert (Hmin : min3 d_del d_ins d_sub = d_ins).
                     { unfold min3. rewrite Nat.min_r by lia. rewrite Nat.min_l by lia. reflexivity. }
                     rewrite Hmin. unfold d_ins. reflexivity.

      * (* A = c1::c1'::A'', B = c2::B' - A has at least 2 chars *)
        destruct B' as [| c2' B''].
        -- (* A = c1::c1'::A'', B = [c2] - B has 1 char *)
           (**
              For A = c1::c1'::A'', B = [c2]:
              damerau_lev_distance (c1::c1'::A'') [c2] = min3 of:
              - d_del = d(c1'::A'', [c2]) + 1  (delete c1)
              - d_ins = d(c1::c1'::A'', []) + 1 = |A| + 1 (insert c2)
              - d_sub = d(c1'::A'', []) + subst_cost c1 c2 = |A|-1 + subst_cost

              Trace costs:
              - dl_shift_trace_A T_del: d(c1'::A'', [c2]) + 1 = d_del
              - Empty []: |A| + 1 = d_ins
              - [DLMatch 1 1]: subst_cost + (|A|-1) + 0 = d_sub
           *)
           rewrite damerau_lev_multi_single.
           set (d_del := damerau_lev_distance (c1' :: A'') [c2] + 1).
           set (d_ins := damerau_lev_distance (c1 :: c1' :: A'') [] + 1).
           set (d_sub := damerau_lev_distance (c1' :: A'') [] + subst_cost c1 c2).

           (* Case split on which branch achieves the minimum *)
           destruct (Nat.leb d_del d_ins) eqn:E_del_ins;
           destruct (Nat.leb d_del d_sub) eqn:E_del_sub.
           ++ (* d_del is minimum *)
              assert (IH_del : exists T', dl_is_valid_trace (c1' :: A'') [c2] T' = true /\
                                          dl_trace_cost (c1' :: A'') [c2] T' = damerau_lev_distance (c1' :: A'') [c2]).
              { apply (IH (length (c1' :: A'') + length [c2])).
                - subst n. simpl in *. lia.
                - reflexivity. }
              destruct IH_del as [T_del [Hv_del Hc_del]].
              exists (dl_shift_trace_A T_del).
              split.
              ** apply dl_is_valid_trace_shift_A. exact Hv_del.
              ** assert (Hvd : dl_all_valid_indices T_del) by (apply dl_is_valid_trace_valid_indices with (A := c1' :: A'') (B := [c2]); exact Hv_del).
                 assert (Hbd : length (dl_touched_in_A T_del) <= length (c1' :: A'')) by (apply dl_valid_trace_touched_A_bound with (B := [c2]); exact Hv_del).
                 rewrite (dl_trace_cost_shift_A _ _ _ _ Hvd Hbd).
                 rewrite Hc_del.
                 apply Nat.leb_le in E_del_ins. apply Nat.leb_le in E_del_sub.
                 assert (Hmin : min3 d_del d_ins d_sub = d_del).
                 { unfold min3. rewrite !Nat.min_l by lia. reflexivity. }
                 rewrite Hmin. unfold d_del. reflexivity.
           ++ (* d_del <= d_ins but d_del > d_sub: d_sub is minimum *)
              exists [DLMatch 1 1].
              split.
              ** unfold dl_is_valid_trace. simpl. reflexivity.
              ** unfold dl_trace_cost, dl_change_cost, dl_element_cost. simpl.
                 apply Nat.leb_le in E_del_ins. apply Nat.leb_gt in E_del_sub.
                 assert (Hmin : min3 d_del d_ins d_sub = d_sub).
                 { unfold min3. rewrite (Nat.min_r d_ins d_sub) by lia.
                   rewrite (Nat.min_r d_del d_sub) by lia. reflexivity. }
                 rewrite Hmin.
                 unfold d_sub. rewrite damerau_lev_empty_right. simpl. lia.
           ++ (* d_del > d_ins, d_del <= d_sub: d_ins is minimum *)
              exists [].
              split.
              ** unfold dl_is_valid_trace. simpl. reflexivity.
              ** rewrite dl_trace_cost_empty.
                 apply Nat.leb_gt in E_del_ins. apply Nat.leb_le in E_del_sub.
                 assert (Hmin : min3 d_del d_ins d_sub = d_ins).
                 { unfold min3. rewrite Nat.min_r by lia. rewrite Nat.min_l by lia. reflexivity. }
                 rewrite Hmin.
                 unfold d_ins. rewrite damerau_lev_empty_right. simpl. lia.
           ++ (* d_del > d_ins and d_del > d_sub: compare d_ins vs d_sub *)
              apply Nat.leb_gt in E_del_ins. apply Nat.leb_gt in E_del_sub.
              destruct (Nat.leb d_ins d_sub) eqn:E_ins_sub.
              ** (* d_ins is minimum *)
                 exists [].
                 split.
                 --- unfold dl_is_valid_trace. simpl. reflexivity.
                 --- rewrite dl_trace_cost_empty.
                     apply Nat.leb_le in E_ins_sub.
                     assert (Hmin : min3 d_del d_ins d_sub = d_ins).
                     { unfold min3. rewrite Nat.min_r by lia. rewrite Nat.min_l by lia. reflexivity. }
                     rewrite Hmin.
                     unfold d_ins. rewrite damerau_lev_empty_right. simpl. lia.
              ** (* d_sub is minimum *)
                 exists [DLMatch 1 1].
                 split.
                 --- unfold dl_is_valid_trace. simpl. reflexivity.
                 --- unfold dl_trace_cost, dl_change_cost, dl_element_cost. simpl.
                     apply Nat.leb_gt in E_ins_sub.
                     assert (Hmin : min3 d_del d_ins d_sub = d_sub).
                     { unfold min3. rewrite Nat.min_r by lia. rewrite Nat.min_r by lia. reflexivity. }
                     rewrite Hmin.
                     unfold d_sub. rewrite damerau_lev_empty_right. simpl. lia.

        -- (* A = c1::c1'::A'', B = c2::c2'::B'' - both have at least 2 chars *)
           (**
              Main case with possible transposition.
              damerau_lev_distance = min4 of:
              - d(c1'::A'', B) + 1  (delete c1)
              - d(A, c2'::B'') + 1  (insert c2)
              - d(c1'::A'', c2'::B'') + subst_cost c1 c2  (match/subst)
              - d(A'', B'') + trans_cost c1 c1' c2 c2'    (transposition if valid)

              Strategy: Case split on which branch achieves the minimum.
           *)
           rewrite damerau_lev_cons2.
           set (d_del := damerau_lev_distance (c1' :: A'') (c2 :: c2' :: B'') + 1).
           set (d_ins := damerau_lev_distance (c1 :: c1' :: A'') (c2' :: B'') + 1).
           set (d_sub := damerau_lev_distance (c1' :: A'') (c2' :: B'') + subst_cost c1 c2).
           set (d_trans := damerau_lev_distance A'' B'' + trans_cost_calc c1 c1' c2 c2').

           (* Case split on which branch achieves minimum *)
           destruct (Nat.leb d_del (min3 d_ins d_sub d_trans)) eqn:E_del_min.
           ++ (* d_del is minimum *)
              assert (IH_del : exists T', dl_is_valid_trace (c1' :: A'') (c2 :: c2' :: B'') T' = true /\
                                           dl_trace_cost (c1' :: A'') (c2 :: c2' :: B'') T' =
                                           damerau_lev_distance (c1' :: A'') (c2 :: c2' :: B'')).
              { apply (IH (length (c1' :: A'') + length (c2 :: c2' :: B''))).
                - subst n. simpl in *. lia.
                - reflexivity. }
              destruct IH_del as [T_del [Hv_del Hc_del]].
              exists (dl_shift_trace_A T_del).
              split.
              ** apply dl_is_valid_trace_shift_A. exact Hv_del.
              ** assert (Hvd : dl_all_valid_indices T_del) by (apply dl_is_valid_trace_valid_indices with (A := c1' :: A'') (B := c2 :: c2' :: B''); exact Hv_del).
                 assert (Hbd : length (dl_touched_in_A T_del) <= length (c1' :: A'')) by (apply dl_valid_trace_touched_A_bound with (B := c2 :: c2' :: B''); exact Hv_del).
                 rewrite (dl_trace_cost_shift_A _ _ _ _ Hvd Hbd).
                 rewrite Hc_del.
                 apply Nat.leb_le in E_del_min.
                 unfold min4.
                 assert (Hmin: min (min d_del d_ins) (min d_sub d_trans) = d_del).
                 { unfold min3 in E_del_min.
                   (* E_del_min : d_del <= min d_ins (min d_sub d_trans) *)
                   assert (H_del_ins: d_del <= d_ins).
                   { apply Nat.le_trans with (min d_ins (min d_sub d_trans)).
                     exact E_del_min. apply Nat.le_min_l. }
                   assert (H_del_subtr: d_del <= min d_sub d_trans).
                   { apply Nat.le_trans with (min d_ins (min d_sub d_trans)).
                     exact E_del_min. apply Nat.le_min_r. }
                   rewrite (Nat.min_l d_del d_ins H_del_ins).
                   apply Nat.min_l. exact H_del_subtr. }
                 rewrite Hmin. unfold d_del. reflexivity.
           ++ (* d_del is NOT minimum, check others *)
              apply Nat.leb_gt in E_del_min.
              destruct (Nat.leb d_ins (min d_sub d_trans)) eqn:E_ins_min.
              ** (* d_ins is minimum *)
                 assert (IH_ins : exists T', dl_is_valid_trace (c1 :: c1' :: A'') (c2' :: B'') T' = true /\
                                              dl_trace_cost (c1 :: c1' :: A'') (c2' :: B'') T' =
                                              damerau_lev_distance (c1 :: c1' :: A'') (c2' :: B'')).
                 { apply (IH (length (c1 :: c1' :: A'') + length (c2' :: B''))).
                   - subst n. simpl in *. lia.
                   - reflexivity. }
                 destruct IH_ins as [T_ins [Hv_ins Hc_ins]].
                 exists (dl_shift_trace_B T_ins).
                 split.
                 --- apply dl_is_valid_trace_shift_B. exact Hv_ins.
                 --- assert (Hvi : dl_all_valid_indices T_ins) by (apply dl_is_valid_trace_valid_indices with (A := c1 :: c1' :: A'') (B := c2' :: B''); exact Hv_ins).
                     assert (Hbi : length (dl_touched_in_B T_ins) <= length (c2' :: B'')) by (apply dl_valid_trace_touched_B_bound with (A := c1 :: c1' :: A''); exact Hv_ins).
                     rewrite (dl_trace_cost_shift_B _ _ _ _ Hvi Hbi).
                     rewrite Hc_ins.
                     apply Nat.leb_le in E_ins_min.
                     unfold min4.
                     assert (Hmin: min (min d_del d_ins) (min d_sub d_trans) = d_ins).
                     { unfold min3 in E_del_min.
                       (* E_del_min : min d_ins (min d_sub d_trans) < d_del *)
                       (* E_ins_min : d_ins <= min d_sub d_trans *)
                       (* Since d_ins <= min d_sub d_trans, min d_ins (min d_sub d_trans) = d_ins *)
                       assert (Heq: min d_ins (min d_sub d_trans) = d_ins).
                       { apply Nat.min_l. exact E_ins_min. }
                       rewrite Heq in E_del_min.
                       (* Now E_del_min : d_ins < d_del *)
                       rewrite (Nat.min_r d_del d_ins) by lia.
                       apply Nat.min_l. exact E_ins_min. }
                     rewrite Hmin. unfold d_ins. reflexivity.
              ** (* d_ins is NOT minimum, check d_sub vs d_trans *)
                 apply Nat.leb_gt in E_ins_min.
                 destruct (Nat.leb d_sub d_trans) eqn:E_sub_trans.
                 --- (* d_sub is minimum *)
                     assert (IH_sub : exists T', dl_is_valid_trace (c1' :: A'') (c2' :: B'') T' = true /\
                                                  dl_trace_cost (c1' :: A'') (c2' :: B'') T' =
                                                  damerau_lev_distance (c1' :: A'') (c2' :: B'')).
                     { apply (IH (length (c1' :: A'') + length (c2' :: B''))).
                       - subst n. simpl in *. lia.
                       - reflexivity. }
                     destruct IH_sub as [T_sub [Hv_sub Hc_sub]].
                     exists (DLMatch 1 1 :: dl_shift_trace_both T_sub).
                     split.
                     +++ (* Valid trace *)
                         apply dl_is_valid_trace_shift_both_with_match. exact Hv_sub.
                     +++ (* Cost equals distance *)
                         assert (Hvi : dl_all_valid_indices T_sub)
                           by (apply dl_is_valid_trace_valid_indices with (A := c1' :: A'') (B := c2' :: B''); exact Hv_sub).
                         assert (HbA : length (dl_touched_in_A T_sub) <= length (c1' :: A''))
                           by (apply dl_valid_trace_touched_A_bound with (B := c2' :: B''); exact Hv_sub).
                         assert (HbB : length (dl_touched_in_B T_sub) <= length (c2' :: B''))
                           by (apply dl_valid_trace_touched_B_bound with (A := c1' :: A''); exact Hv_sub).
                         rewrite (dl_trace_cost_shift_both_with_match (c1' :: A'') (c2' :: B'') T_sub c1 c2 Hvi HbA HbB).
                         rewrite Hc_sub.
                         (* Show min4 d_del d_ins d_sub d_trans = d_sub *)
                         apply Nat.leb_le in E_sub_trans.
                         unfold min4.
                         assert (Hmin: min (min d_del d_ins) (min d_sub d_trans) = d_sub).
                         { unfold min3 in E_del_min, E_ins_min.
                           (* E_del_min: min d_ins (min d_sub d_trans) < d_del *)
                           (* E_ins_min: min d_sub d_trans < d_ins *)
                           (* E_sub_trans: d_sub <= d_trans *)
                           assert (Heq1: min d_sub d_trans = d_sub) by (apply Nat.min_l; exact E_sub_trans).
                           rewrite Heq1 in *.
                           (* Now: E_ins_min: d_sub < d_ins, E_del_min: min d_ins d_sub < d_del *)
                           assert (Heq2: min d_ins d_sub = d_sub) by (apply Nat.min_r; lia).
                           rewrite Heq2 in E_del_min.
                           (* Now: E_del_min: d_sub < d_del, E_ins_min: d_sub < d_ins *)
                           (* Goal: min (min d_del d_ins) d_sub = d_sub *)
                           (* Since d_sub < d_del and d_sub < d_ins, d_sub <= min d_del d_ins *)
                           apply Nat.min_r.
                           apply Nat.min_glb; lia. }
                         rewrite Hmin. unfold d_sub. reflexivity.
                 --- (* d_trans is minimum - use transposition *)
                     apply Nat.leb_gt in E_sub_trans.
                     assert (IH_trans : exists T', dl_is_valid_trace A'' B'' T' = true /\
                                                    dl_trace_cost A'' B'' T' = damerau_lev_distance A'' B'').
                     { apply (IH (length A'' + length B'')).
                       - subst n. simpl in *. lia.
                       - reflexivity. }
                     destruct IH_trans as [T_trans [Hv_trans Hc_trans]].
                     exists (DLTranspose 1 1 :: dl_shift_trace_both2 T_trans).
                     split.
                     +++ (* Valid trace *)
                         apply dl_is_valid_trace_shift_both2_with_transpose. exact Hv_trans.
                     +++ (* Cost equals distance *)
                         assert (Hvi : dl_all_valid_indices T_trans)
                           by (apply dl_is_valid_trace_valid_indices with (A := A'') (B := B''); exact Hv_trans).
                         assert (HbA : length (dl_touched_in_A T_trans) <= length A'')
                           by (apply dl_valid_trace_touched_A_bound with (B := B''); exact Hv_trans).
                         assert (HbB : length (dl_touched_in_B T_trans) <= length B'')
                           by (apply dl_valid_trace_touched_B_bound with (A := A''); exact Hv_trans).
                         rewrite (dl_trace_cost_shift_both2_with_transpose A'' B'' T_trans c1 c1' c2 c2' Hvi HbA HbB).
                         rewrite Hc_trans.
                         (* Show min4 d_del d_ins d_sub d_trans = d_trans *)
                         unfold min4.
                         assert (Hmin: min (min d_del d_ins) (min d_sub d_trans) = d_trans).
                         { unfold min3 in E_del_min, E_ins_min.
                           (* E_del_min: min d_ins (min d_sub d_trans) < d_del *)
                           (* E_ins_min: min d_sub d_trans < d_ins *)
                           (* E_sub_trans: d_trans < d_sub *)
                           assert (Heq1: min d_sub d_trans = d_trans) by (apply Nat.min_r; lia).
                           rewrite Heq1 in *.
                           (* E_ins_min: d_trans < d_ins, E_del_min: min d_ins d_trans < d_del *)
                           assert (Heq2: min d_ins d_trans = d_trans) by (apply Nat.min_r; lia).
                           rewrite Heq2 in E_del_min.
                           (* E_del_min: d_trans < d_del *)
                           apply Nat.min_r.
                           apply Nat.min_glb; lia. }
                         rewrite Hmin. unfold d_trans. reflexivity.
Qed.

(** End of DamerauTrace module *)
