(** * Core Types for NFA/Phonetic Regex Layer

    This module defines the foundational types for the Generalized Levenshtein
    NFA with context-sensitive phonetic operations. This formalization builds on
    the theoretical framework from "Fast String Correction with Levenshtein Automata"
    (TCS 2011) and extends it with context-aware phonetic transformations.
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Coq.NArith.BinNat.
Require Import Coq.ZArith.BinInt.
Require Import Coq.micromega.Lia.
Import ListNotations.

Open Scope nat_scope.

(** ** Context Definitions *)

(** Context captures the linguistic environment for phonetic rules.
    This allows us to model context-sensitive phonetic transformations
    (e.g., "c" → "s" before front vowels {e, i, y}).
*)
Inductive Context : Type :=
  | Anywhere                                    (** No context restriction *)
  | Initial                                     (** Start of word *)
  | Final                                       (** End of word *)
  | BeforeVowel (vowels : list ascii)          (** Before specific vowels *)
  | AfterVowel (vowels : list ascii)           (** After specific vowels *)
  | BeforeConsonant (consonants : list ascii)  (** Before specific consonants *)
  | AfterConsonant (consonants : list ascii)   (** After specific consonants *)
  | BetweenVowels                              (** Between any vowels *)
  | InitialCluster.                            (** Part of initial consonant cluster *)

(** Helper: Check if character is a vowel *)
Definition is_vowel (c : ascii) : bool :=
  match c with
  | "a" | "e" | "i" | "o" | "u"
  | "A" | "E" | "I" | "O" | "U" => true
  | _ => false
  end%char.

(** Helper: Check if character is a consonant *)
Definition is_consonant (c : ascii) : bool :=
  negb (is_vowel c).

(** Check if context matches at a position in a string *)
Fixpoint context_matches (ctx : Context) (s : string) (pos : nat) : bool :=
  match ctx with
  | Anywhere => true
  | Initial => Nat.eqb pos 0
  | Final => Nat.eqb pos (String.length s)
  | BeforeVowel vowels =>
      match get pos s with
      | Some c => existsb (Ascii.eqb c) vowels
      | None => false
      end
  | AfterVowel vowels =>
      match pos with
      | 0 => false
      | S pos' =>
          match get pos' s with
          | Some c => existsb (Ascii.eqb c) vowels
          | None => false
          end
      end
  | BeforeConsonant consonants =>
      match get pos s with
      | Some c => existsb (Ascii.eqb c) consonants
      | None => false
      end
  | AfterConsonant consonants =>
      match pos with
      | 0 => false
      | S pos' =>
          match get pos' s with
          | Some c => existsb (Ascii.eqb c) consonants
          | None => false
          end
      end
  | BetweenVowels =>
      match pos with
      | 0 => false
      | S pos' =>
          match get pos' s, get pos s with
          | Some c1, Some c2 => is_vowel c1 && is_vowel c2
          | _, _ => false
          end
      end
  | InitialCluster =>
      (Nat.leb pos 2) &&
      match get pos s with
      | Some c => is_consonant c
      | None => false
      end
  end.

(** ** Characteristic Vectors *)

(** A characteristic vector encodes the positions where a character occurs in a word.
    For a word w and character a, cv(w, a) is a bit vector where bit i is 1 iff w[i] = a.

    We use N (arbitrary precision natural numbers) to represent bit vectors.
*)
Definition CharacteristicVector := N.

(** Empty characteristic vector *)
Definition cv_empty : CharacteristicVector := 0%N.

(** Set bit at position i *)
Definition cv_set_bit (cv : CharacteristicVector) (pos : nat) : CharacteristicVector :=
  N.lor cv (N.shiftl 1 (N.of_nat pos)).

(** Test bit at position i *)
Definition cv_test_bit (cv : CharacteristicVector) (pos : nat) : bool :=
  N.testbit cv (N.of_nat pos).

(** Build characteristic vector for character c in string s *)
Fixpoint build_cv (s : string) (c : ascii) (pos : nat) : CharacteristicVector :=
  match s with
  | EmptyString => cv_empty
  | String c' s' =>
      let cv_rest := build_cv s' c (S pos) in
      if Ascii.eqb c c' then cv_set_bit cv_rest pos else cv_rest
  end.

(** Main entry point: build CV from start of string *)
Definition characteristic_vector (s : string) (c : ascii) : CharacteristicVector :=
  build_cv s c 0.

(** ** Position Type *)

(** A position in the NFA represents a (position, error_count, context) triple.
    - i: position in the target word
    - e: number of errors consumed
    - ctx: current context for phonetic rules
*)
Record Position := mkPosition {
  pos_i : nat;        (** Position in target word *)
  pos_e : nat;        (** Error count *)
  pos_ctx : Context   (** Current context *)
}.

(** Position equality *)
Definition position_eqb (p1 p2 : Position) : bool :=
  (pos_i p1 =? pos_i p2) && (pos_e p1 =? pos_e p2).
  (* Note: We don't compare contexts for equality - they're metadata *)

(** Position ordering (for subsumption) *)
Definition position_le (p1 p2 : Position) : bool :=
  (pos_i p1 =? pos_i p2) && (pos_e p1 <=? pos_e p2).

(** ** Operation Types *)

(** An operation type represents a single edit operation with context.
    Following TCS 2011 Definition 5, an operation is a tuple ⟨t^x, t^y, t^w⟩:
    - t^x: number of characters consumed from first word
    - t^y: number of characters consumed from second word
    - t^w: cost (weight) of the operation

    We extend this with:
    - chars_x: actual characters from first word (for multi-char ops)
    - chars_y: actual characters from second word
    - context: linguistic context requirement
*)
Record OperationType := mkOperation {
  op_consume_x : nat;            (** Characters consumed from word 1 *)
  op_consume_y : nat;            (** Characters consumed from word 2 *)
  op_weight : Q;                 (** Operation cost *)
  op_chars_x : list ascii;       (** Expected characters from word 1 *)
  op_chars_y : list ascii;       (** Expected characters from word 2 *)
  op_context : Context;          (** Required linguistic context *)
  op_name : string               (** Human-readable name for debugging *)
}.

(** Standard edit operations *)

(** Match: consume identical character from both words, cost 0 *)
Definition op_match (c : ascii) : OperationType :=
  mkOperation 1 1 0%Q [c] [c] Anywhere "match".

(** Insert: insert character into word 2, cost 1 *)
Definition op_insert (c : ascii) : OperationType :=
  mkOperation 0 1 1%Q [] [c] Anywhere "insert".

(** Delete: delete character from word 1, cost 1 *)
Definition op_delete (c : ascii) : OperationType :=
  mkOperation 1 0 1%Q [c] [] Anywhere "delete".

(** Substitute: replace one character with another, cost 1 *)
Definition op_substitute (c1 c2 : ascii) : OperationType :=
  mkOperation 1 1 1%Q [c1] [c2] Anywhere "substitute".

(** Transpose: swap adjacent characters, cost 1 *)
Definition op_transpose (c1 c2 : ascii) : OperationType :=
  mkOperation 2 2 1%Q [c1; c2] [c2; c1] Anywhere "transpose".

(** ** Phonetic Operations *)

(** Phonetic consonant digraph: 2 characters → 1 phoneme, cost 0.15 *)
Definition op_phonetic_digraph (src1 src2 dst : ascii) (ctx : Context) : OperationType :=
  mkOperation 2 1 0.15%Q [src1; src2] [dst] ctx "phonetic_digraph".

(** Phonetic substitution: phonetically similar characters, cost 0.20 *)
Definition op_phonetic_subst (c1 c2 : ascii) (ctx : Context) : OperationType :=
  mkOperation 1 1 0.20%Q [c1] [c2] ctx "phonetic_subst".

(** Phonetic expansion: 1 phoneme → 2 characters, cost 0.15 *)
Definition op_phonetic_expand (src dst1 dst2 : ascii) (ctx : Context) : OperationType :=
  mkOperation 1 2 0.15%Q [src] [dst1; dst2] ctx "phonetic_expand".

(** Silent letter deletion: drop silent character, cost 0.10 *)
Definition op_silent_delete (c : ascii) (ctx : Context) : OperationType :=
  mkOperation 1 0 0.10%Q [c] [] ctx "silent_delete".

(** ** Operation Sets *)

(** A collection of operations defines the behavior of the automaton *)
Definition OperationSet := list OperationType.

(** Helper: Convert string to list of ascii *)
Fixpoint list_ascii_of_string (s : string) : list ascii :=
  match s with
  | EmptyString => []
  | String c s' => c :: list_ascii_of_string s'
  end.

(** Helper: Check if two lists of ascii are equal *)
Definition list_ascii_eqb (l1 l2 : list ascii) : bool :=
  (length l1 =? length l2) &&
  forallb (fun p => Ascii.eqb (fst p) (snd p)) (combine l1 l2).

(** Check if operation can apply at current position *)
Definition can_apply (op : OperationType) (s1 s2 : string) (i j : nat) : bool :=
  (* Check sufficient characters remain *)
  (i + op_consume_x op <=? String.length s1) &&
  (j + op_consume_y op <=? String.length s2) &&
  (* Check character match *)
  let chars1 := substring i (op_consume_x op) s1 in
  let chars2 := substring j (op_consume_y op) s2 in
  list_ascii_eqb (list_ascii_of_string chars1) (op_chars_x op) &&
  list_ascii_eqb (list_ascii_of_string chars2) (op_chars_y op) &&
  (* Check context *)
  context_matches (op_context op) s1 i.

(** ** Generalized State *)

(** A state in the Generalized Levenshtein NFA is a set of positions.
    Following TCS 2011 Definition 15, states are I-type (initial phase)
    or M-type (matching phase). We represent both with position sets.
*)
Record GeneralizedState := mkState {
  state_positions : list Position;   (** Set of (i, e, ctx) positions *)
  state_is_initial : bool;            (** Is this an initial state? *)
  state_max_distance : nat            (** Maximum edit distance allowed *)
}.

(** Empty state *)
Definition empty_state (max_dist : nat) : GeneralizedState :=
  mkState [] true max_dist.

(** Initial state: {(0, 0, Initial)} *)
Definition initial_state (max_dist : nat) : GeneralizedState :=
  mkState [mkPosition 0 0 Initial] true max_dist.

(** Check if position is in state *)
Definition state_contains (st : GeneralizedState) (p : Position) : bool :=
  existsb (position_eqb p) (state_positions st).

(** Add position to state *)
Definition state_add (st : GeneralizedState) (p : Position) : GeneralizedState :=
  if state_contains st p then st
  else mkState (p :: state_positions st) (state_is_initial st) (state_max_distance st).

(** State equality *)
Definition state_eqb (s1 s2 : GeneralizedState) : bool :=
  (length (state_positions s1) =? length (state_positions s2)) &&
  forallb (fun p => state_contains s2 p) (state_positions s1).

(** ** Subsumption *)

(** Position subsumption: p1 subsumes p2 if p1.i = p2.i and p1.e ≤ p2.e
    This is the key optimization from TCS 2011 Lemma 7.3.
*)
Definition position_subsumes (p1 p2 : Position) : bool :=
  (pos_i p1 =? pos_i p2) && (pos_e p1 <=? pos_e p2).

(** State subsumption *)
Definition state_subsumes (s1 s2 : GeneralizedState) : bool :=
  forallb (fun p2 =>
    existsb (fun p1 => position_subsumes p1 p2) (state_positions s1)
  ) (state_positions s2).

(** Prune subsumed positions from state *)
Fixpoint prune_subsumed_positions (positions : list Position) : list Position :=
  match positions with
  | [] => []
  | p :: rest =>
      let rest' := prune_subsumed_positions rest in
      if existsb (fun p' => position_subsumes p' p) rest' then rest'
      else p :: (filter (fun p' => negb (position_subsumes p p')) rest')
  end.

Definition prune_state (st : GeneralizedState) : GeneralizedState :=
  mkState (prune_subsumed_positions (state_positions st))
          (state_is_initial st)
          (state_max_distance st).

(** ** Prune State Specification

    The prune_state function removes subsumed positions from a state while
    preserving essential properties. This specification captures:
    1. Position inclusion: pruned positions are a subset of original
    2. Error bound preservation: all pruned positions respect the max distance
    3. Acceptance preservation: if original has accepting position, so does pruned
    4. Subsumption property: removed positions are subsumed by remaining ones *)

(** Positions in pruned state are a subset of positions in original state *)
Definition prune_state_incl (st : GeneralizedState) : Prop :=
  incl (state_positions (prune_state st)) (state_positions st).

(** Pruning preserves the error bound on positions *)
Definition prune_preserves_error_bound (st : GeneralizedState) : Prop :=
  Forall (fun p => pos_e p <= state_max_distance st)
         (state_positions (prune_state st)).

(** Pruning preserves the existence of accepting positions *)
Definition prune_preserves_acceptance (word_length : nat) (st : GeneralizedState) : Prop :=
  existsb (fun p => pos_i p =? word_length) (state_positions st) = true ->
  existsb (fun p => pos_i p =? word_length) (state_positions (prune_state st)) = true.

(** Removed positions are subsumed by remaining positions *)
Definition prune_subsumed_complete (st : GeneralizedState) : Prop :=
  forall p, In p (state_positions st) ->
    ~In p (state_positions (prune_state st)) ->
    exists p', In p' (state_positions (prune_state st)) /\
               position_subsumes p' p = true.

(** Full specification of prune_state *)
Definition prune_state_spec (st : GeneralizedState) : Prop :=
  prune_state_incl st /\
  prune_preserves_error_bound st /\
  (forall wl, prune_preserves_acceptance wl st) /\
  prune_subsumed_complete st.

Definition prune_state_satisfies_spec_contract : Prop := forall st,
  Forall (fun p => pos_e p <= state_max_distance st) (state_positions st) ->
  prune_state_spec st.

(** prune_subsumed_positions returns a sublist of the input. *)
Lemma prune_subsumed_is_sublist : forall positions p,
  In p (prune_subsumed_positions positions) -> In p positions.
Proof.
  induction positions as [| h rest IH]; intros p Hin; simpl in *.
  - contradiction.
  - destruct (existsb (fun p' : Position => position_subsumes p' h)
                      (prune_subsumed_positions rest)) eqn:Hsubsumed.
    + right. apply IH. exact Hin.
    + destruct Hin as [Heq | Hin].
      * left. exact Heq.
      * apply filter_In in Hin as [Hin _].
        right. apply IH. exact Hin.
Qed.

(** Corollaries that follow from the specification *)

Lemma prune_state_incl_holds : forall st,
  incl (state_positions (prune_state st)) (state_positions st).
Proof.
  intros st p Hin.
  unfold prune_state in Hin. simpl in Hin.
  apply prune_subsumed_is_sublist. exact Hin.
Qed.

(** ** Well-Formedness Conditions *)

(** A position is well-formed if errors don't exceed the bound *)
Definition wf_position (max_dist : nat) (p : Position) : Prop :=
  pos_e p <= max_dist.

(** A state is well-formed if all positions are well-formed *)
Definition wf_state (st : GeneralizedState) : Prop :=
  Forall (wf_position (state_max_distance st)) (state_positions st).

(** An operation is well-formed if it has reasonable bounds *)
Definition wf_operation (op : OperationType) : Prop :=
  (0 <= op_weight op)%Q /\
  op_consume_x op <= 3 /\  (* At most 3-char operations *)
  op_consume_y op <= 3 /\
  length (op_chars_x op) = op_consume_x op /\
  length (op_chars_y op) = op_consume_y op.

(** An operation set is well-formed if all operations are *)
Definition wf_operation_set (ops : OperationSet) : Prop :=
  Forall wf_operation ops.

(** ** Bounded Diagonal Property *)

(** An operation satisfies the bounded diagonal property if it preserves
    length difference within a constant bound. From TCS 2011 Theorem 8.2.
*)
Definition bounded_diagonal (c : nat) (op : OperationType) : Prop :=
  (Z.abs (Z.of_nat (op_consume_y op) - Z.of_nat (op_consume_x op)) <= Z.of_nat c)%Z.

(** 1-bounded diagonal: most common case *)
Definition is_1_bounded (op : OperationType) : Prop :=
  bounded_diagonal 1 op.

(** Operation set satisfies bounded diagonal if all operations do *)
Definition operation_set_bounded (c : nat) (ops : OperationSet) : Prop :=
  Forall (bounded_diagonal c) ops.

(** ** Basic Properties *)

(** Characteristic vector properties *)
Lemma cv_empty_no_bits : forall pos,
  cv_test_bit cv_empty pos = false.
Proof.
  intros. unfold cv_test_bit, cv_empty.
  apply N.bits_0.
Qed.

Lemma cv_set_test_eq : forall cv pos,
  cv_test_bit (cv_set_bit cv pos) pos = true.
Proof.
  intros. unfold cv_test_bit, cv_set_bit.
  rewrite N.lor_spec, N.shiftl_spec_high by lia.
  rewrite N.sub_diag. simpl. apply Bool.orb_true_r.
Qed.

Lemma cv_set_test_neq : forall cv pos1 pos2,
  pos1 <> pos2 ->
  cv_test_bit (cv_set_bit cv pos1) pos2 = cv_test_bit cv pos2.
Proof.
  intros cv pos1 pos2 Hneq.
  unfold cv_test_bit, cv_set_bit.
  rewrite N.lor_spec.
  (* Need to show the shifted bit doesn't affect position pos2 *)
  assert (H: N.testbit (N.shiftl 1 (N.of_nat pos1)) (N.of_nat pos2) = false).
  { (* Case analysis on the ordering of pos1 and pos2 *)
    destruct (Nat.lt_trichotomy pos1 pos2) as [Hlt | [Heq | Hgt]].
    - (* pos1 < pos2: use high spec *)
      (* N.shiftl_spec_high: forall a n m : N, (n <= m)%N ->
         N.testbit (N.shiftl a n) m = N.testbit a (m - n) *)
      rewrite N.shiftl_spec_high by lia.
      (* Goal: N.testbit 1 (N.of_nat pos2 - N.of_nat pos1) = false *)
      (* When pos1 < pos2, (N.of_nat pos2 - N.of_nat pos1)%N >= 1
         and N.testbit 1 k = false for k >= 1 *)
      assert (Hdiff: (N.of_nat pos2 - N.of_nat pos1 >= 1)%N) by lia.
      destruct (N.of_nat pos2 - N.of_nat pos1)%N as [|p] eqn:Heq_diff.
      + (* Difference is 0: contradiction since pos1 < pos2 *)
        exfalso. lia.
      + (* Difference is positive: N.testbit 1 (N.pos p) = false *)
        destruct p; reflexivity.
    - (* pos1 = pos2: contradiction with Hneq *)
      exfalso. apply Hneq. assumption.
    - (* pos1 > pos2: use low spec *)
      (* N.shiftl_spec_low: forall a n m : N, (m < n)%N ->
         N.testbit (N.shiftl a n) m = false *)
      rewrite N.shiftl_spec_low by lia.
      reflexivity.
  }
  rewrite H. rewrite Bool.orb_false_r. reflexivity.
Qed.

(** Position equality is reflexive *)
Lemma position_eqb_refl : forall p,
  position_eqb p p = true.
Proof.
  intros [i e ctx]. unfold position_eqb. simpl.
  rewrite Nat.eqb_refl, Nat.eqb_refl.
  reflexivity.
Qed.

(** Position subsumption is reflexive *)
Lemma position_subsumes_refl : forall p,
  position_subsumes p p = true.
Proof.
  intros [i e ctx]. unfold position_subsumes. simpl.
  rewrite Nat.eqb_refl, Nat.leb_refl.
  reflexivity.
Qed.

(** Position subsumption is transitive *)
Lemma position_subsumes_trans : forall p1 p2 p3,
  position_subsumes p1 p2 = true ->
  position_subsumes p2 p3 = true ->
  position_subsumes p1 p3 = true.
Proof.
  intros [i1 e1 ctx1] [i2 e2 ctx2] [i3 e3 ctx3].
  unfold position_subsumes. simpl.
  intros H12 H23.
  apply andb_true_iff in H12. destruct H12 as [Hi12 He12].
  apply andb_true_iff in H23. destruct H23 as [Hi23 He23].
  apply Nat.eqb_eq in Hi12. apply Nat.eqb_eq in Hi23.
  apply Nat.leb_le in He12. apply Nat.leb_le in He23.
  subst. rewrite Nat.eqb_refl. simpl.
  apply Nat.leb_le. lia.
Qed.

(** Standard operations are 1-bounded *)
Lemma op_match_1_bounded : forall c,
  is_1_bounded (op_match c).
Proof.
  intros c. unfold is_1_bounded, bounded_diagonal, op_match. simpl.
  (* |1 - 1| = 0 <= 1 *)
  lia.
Qed.

Lemma op_insert_1_bounded : forall c,
  is_1_bounded (op_insert c).
Proof.
  intros c. unfold is_1_bounded, bounded_diagonal, op_insert. simpl.
  (* |1 - 0| = 1 <= 1 *)
  lia.
Qed.

Lemma op_delete_1_bounded : forall c,
  is_1_bounded (op_delete c).
Proof.
  intros c. unfold is_1_bounded, bounded_diagonal, op_delete. simpl.
  (* |0 - 1| = 1 <= 1 *)
  lia.
Qed.

Lemma op_substitute_1_bounded : forall c1 c2,
  is_1_bounded (op_substitute c1 c2).
Proof.
  intros c1 c2. unfold is_1_bounded, bounded_diagonal, op_substitute. simpl.
  (* |1 - 1| = 0 <= 1 *)
  lia.
Qed.

Lemma op_transpose_1_bounded : forall c1 c2,
  is_1_bounded (op_transpose c1 c2).
Proof.
  intros c1 c2. unfold is_1_bounded, bounded_diagonal, op_transpose. simpl.
  (* |2 - 2| = 0 <= 1 *)
  lia.
Qed.
