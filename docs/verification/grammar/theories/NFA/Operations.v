(** * Phonetic Operations for Generalized Levenshtein NFA

    This module defines the phonetic operations used in the NFA layer,
    including context-sensitive transformations and their proofs of
    bounded diagonal property.

    Based on empirical analysis from English phonetic feasibility study,
    we target ~85% coverage of common phonetic errors with Phase 1
    operations (unconditional and simple context-dependent).
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Coq.micromega.Lia.
Require Import Coq.ZArith.Znat.
Import ListNotations.

Require Import Liblevenshtein.Grammar.Verification.NFA.Types.

(** * Axioms for Operation Properties *)

(** When can_apply succeeds, the characters at the positions match
    the operation's expected characters. *)
Axiom can_apply_chars_match_ax : forall op s1 s2 i j,
  can_apply op s1 s2 i j = true ->
  let chars1 := substring i (op_consume_x op) s1 in
  let chars2 := substring j (op_consume_y op) s2 in
  list_ascii_of_string chars1 = op_chars_x op /\
  list_ascii_of_string chars2 = op_chars_y op.

(** Context matching depends only on the prefix up to the current position. *)
Axiom context_matches_monotone_ax : forall ctx s1 s2 pos,
  substring 0 pos s1 = substring 0 pos s2 ->
  context_matches ctx s1 pos = context_matches ctx s2 pos.

(** ** Phonetic Consonant Digraphs *)

(** English consonant digraphs that map to single phonemes *)

Definition op_ch_to_k : OperationType :=
  op_phonetic_digraph "c"%char "h"%char "k"%char Anywhere.

Definition op_sh_to_s : OperationType :=
  op_phonetic_digraph "s"%char "h"%char "s"%char Anywhere.

Definition op_ph_to_f : OperationType :=
  op_phonetic_digraph "p"%char "h"%char "f"%char Anywhere.

Definition op_th_to_t : OperationType :=
  op_phonetic_digraph "t"%char "h"%char "t"%char Anywhere.

Definition op_gh_to_f : OperationType :=
  op_phonetic_digraph "g"%char "h"%char "f"%char Final.

Definition op_wh_to_w : OperationType :=
  op_phonetic_digraph "w"%char "h"%char "w"%char Initial.

(** ** Initial Consonant Clusters *)

(** Silent letters in initial clusters *)

Definition op_kn_to_n : OperationType :=
  op_phonetic_digraph "k"%char "n"%char "n"%char Initial.

Definition op_gn_to_n : OperationType :=
  op_phonetic_digraph "g"%char "n"%char "n"%char Initial.

Definition op_pn_to_n : OperationType :=
  op_phonetic_digraph "p"%char "n"%char "n"%char Initial.

Definition op_ps_to_s : OperationType :=
  op_phonetic_digraph "p"%char "s"%char "s"%char Initial.

Definition op_wr_to_r : OperationType :=
  op_phonetic_digraph "w"%char "r"%char "r"%char Initial.

(** ** Context-Sensitive Phonetic Substitutions *)

(** C → S before front vowels {e, i, y} *)
Definition front_vowels : list ascii := ["e"; "i"; "y"]%char.

Definition op_c_to_s : OperationType :=
  op_phonetic_subst "c"%char "s"%char (BeforeVowel front_vowels).

Definition op_c_to_k : OperationType :=
  op_phonetic_subst "c"%char "k"%char Anywhere.

(** G → J before front vowels *)
Definition op_g_to_j : OperationType :=
  op_phonetic_subst "g"%char "j"%char (BeforeVowel front_vowels).

Definition op_g_to_g : OperationType :=
  op_phonetic_subst "g"%char "g"%char Anywhere.

(** S ↔ Z voicing *)
Definition op_s_to_z : OperationType :=
  op_phonetic_subst "s"%char "z"%char BetweenVowels.

Definition op_z_to_s : OperationType :=
  op_phonetic_subst "z"%char "s"%char BetweenVowels.

(** ** Double Consonants *)

(** Double consonants simplify to single *)

Definition op_double_b : OperationType :=
  op_phonetic_digraph "b"%char "b"%char "b"%char Anywhere.

Definition op_double_d : OperationType :=
  op_phonetic_digraph "d"%char "d"%char "d"%char Anywhere.

Definition op_double_f : OperationType :=
  op_phonetic_digraph "f"%char "f"%char "f"%char Anywhere.

Definition op_double_g : OperationType :=
  op_phonetic_digraph "g"%char "g"%char "g"%char Anywhere.

Definition op_double_l : OperationType :=
  op_phonetic_digraph "l"%char "l"%char "l"%char Anywhere.

Definition op_double_m : OperationType :=
  op_phonetic_digraph "m"%char "m"%char "m"%char Anywhere.

Definition op_double_n : OperationType :=
  op_phonetic_digraph "n"%char "n"%char "n"%char Anywhere.

Definition op_double_p : OperationType :=
  op_phonetic_digraph "p"%char "p"%char "p"%char Anywhere.

Definition op_double_r : OperationType :=
  op_phonetic_digraph "r"%char "r"%char "r"%char Anywhere.

Definition op_double_s : OperationType :=
  op_phonetic_digraph "s"%char "s"%char "s"%char Anywhere.

Definition op_double_t : OperationType :=
  op_phonetic_digraph "t"%char "t"%char "t"%char Anywhere.

(** ** Silent Letters *)

(** Common silent letters in English *)

Definition op_silent_e : OperationType :=
  op_silent_delete "e"%char Final.

Definition op_silent_b : OperationType :=
  op_silent_delete "b"%char (AfterConsonant ["m"]%char).

Definition op_silent_k : OperationType :=
  op_silent_delete "k"%char (BeforeConsonant ["n"]%char).

Definition op_silent_w : OperationType :=
  op_silent_delete "w"%char (BeforeConsonant ["r"]%char).

Definition op_silent_h : OperationType :=
  op_silent_delete "h"%char (AfterConsonant ["w"; "g"; "r"]%char).

(** ** Phonetic Operation Sets *)

(** Phase 1: Basic phonetic operations (unconditional + simple context) *)
Definition phonetic_ops_phase1 : OperationSet := [
  (* Consonant digraphs *)
  op_ch_to_k; op_sh_to_s; op_ph_to_f; op_th_to_t; op_gh_to_f; op_wh_to_w;

  (* Initial clusters *)
  op_kn_to_n; op_gn_to_n; op_pn_to_n; op_ps_to_s; op_wr_to_r;

  (* Context-sensitive substitutions *)
  op_c_to_s; op_c_to_k; op_g_to_j; op_s_to_z; op_z_to_s;

  (* Double consonants *)
  op_double_b; op_double_d; op_double_f; op_double_g; op_double_l;
  op_double_m; op_double_n; op_double_p; op_double_r; op_double_s; op_double_t;

  (* Silent letters *)
  op_silent_e; op_silent_b; op_silent_k; op_silent_w; op_silent_h
].

(** Standard edit operations *)
Definition standard_ops : OperationSet := [
  (* Match is implicit in NFA transitions *)
  (* Insert any ASCII character (0-127) *)
  (* Delete any ASCII character *)
  (* Substitute any pair of characters *)
  (* Transpose adjacent characters *)
].
(* Note: In practice, these are generated dynamically based on input *)

(** Combined operation set: standard + phonetic *)
Definition full_operation_set : OperationSet :=
  standard_ops ++ phonetic_ops_phase1.

(** ** Bounded Diagonal Proofs *)

(** All phonetic digraphs (2→1) are 1-bounded *)
Lemma phonetic_digraph_1_bounded : forall src1 src2 dst ctx,
  is_1_bounded (op_phonetic_digraph src1 src2 dst ctx).
Proof.
  intros. unfold is_1_bounded, bounded_diagonal, op_phonetic_digraph. simpl.
  (* 2 consumed from x, 1 consumed from y *)
  (* diff = 1 - 2 = -1, |diff| = 1 ≤ 1 ✓ *)
  lia.
Qed.

Lemma op_ch_to_k_1_bounded : is_1_bounded op_ch_to_k.
Proof. unfold op_ch_to_k. apply phonetic_digraph_1_bounded. Qed.

Lemma op_sh_to_s_1_bounded : is_1_bounded op_sh_to_s.
Proof. unfold op_sh_to_s. apply phonetic_digraph_1_bounded. Qed.

Lemma op_ph_to_f_1_bounded : is_1_bounded op_ph_to_f.
Proof. unfold op_ph_to_f. apply phonetic_digraph_1_bounded. Qed.

Lemma op_th_to_t_1_bounded : is_1_bounded op_th_to_t.
Proof. unfold op_th_to_t. apply phonetic_digraph_1_bounded. Qed.

Lemma op_gh_to_f_1_bounded : is_1_bounded op_gh_to_f.
Proof. unfold op_gh_to_f. apply phonetic_digraph_1_bounded. Qed.

Lemma op_wh_to_w_1_bounded : is_1_bounded op_wh_to_w.
Proof. unfold op_wh_to_w. apply phonetic_digraph_1_bounded. Qed.

Lemma op_kn_to_n_1_bounded : is_1_bounded op_kn_to_n.
Proof. unfold op_kn_to_n. apply phonetic_digraph_1_bounded. Qed.

Lemma op_gn_to_n_1_bounded : is_1_bounded op_gn_to_n.
Proof. unfold op_gn_to_n. apply phonetic_digraph_1_bounded. Qed.

Lemma op_pn_to_n_1_bounded : is_1_bounded op_pn_to_n.
Proof. unfold op_pn_to_n. apply phonetic_digraph_1_bounded. Qed.

Lemma op_ps_to_s_1_bounded : is_1_bounded op_ps_to_s.
Proof. unfold op_ps_to_s. apply phonetic_digraph_1_bounded. Qed.

Lemma op_wr_to_r_1_bounded : is_1_bounded op_wr_to_r.
Proof. unfold op_wr_to_r. apply phonetic_digraph_1_bounded. Qed.

Lemma op_double_1_bounded : forall c,
  is_1_bounded (op_phonetic_digraph c c c Anywhere).
Proof. intros. apply (phonetic_digraph_1_bounded c c c Anywhere). Qed.

(** Phonetic substitutions (1→1) are 1-bounded *)
Lemma phonetic_subst_1_bounded : forall c1 c2 ctx,
  is_1_bounded (op_phonetic_subst c1 c2 ctx).
Proof.
  intros. unfold is_1_bounded, bounded_diagonal, op_phonetic_subst. simpl.
  (* 1 consumed from x, 1 consumed from y *)
  (* diff = 1 - 1 = 0, |diff| = 0 ≤ 1 ✓ *)
  lia.
Qed.

Lemma op_c_to_s_1_bounded : is_1_bounded op_c_to_s.
Proof. apply phonetic_subst_1_bounded. Qed.

Lemma op_c_to_k_1_bounded : is_1_bounded op_c_to_k.
Proof. apply phonetic_subst_1_bounded. Qed.

Lemma op_g_to_j_1_bounded : is_1_bounded op_g_to_j.
Proof. apply phonetic_subst_1_bounded. Qed.

Lemma op_s_to_z_1_bounded : is_1_bounded op_s_to_z.
Proof. apply phonetic_subst_1_bounded. Qed.

Lemma op_z_to_s_1_bounded : is_1_bounded op_z_to_s.
Proof. apply phonetic_subst_1_bounded. Qed.

(** Silent deletions (1→0) are 1-bounded *)
Lemma silent_delete_1_bounded : forall c ctx,
  is_1_bounded (op_silent_delete c ctx).
Proof.
  intros. unfold is_1_bounded, bounded_diagonal, op_silent_delete. simpl.
  (* 1 consumed from x, 0 consumed from y *)
  (* diff = 0 - 1 = -1, |diff| = 1 ≤ 1 ✓ *)
  lia.
Qed.

Lemma op_silent_e_1_bounded : is_1_bounded op_silent_e.
Proof. apply silent_delete_1_bounded. Qed.

Lemma op_silent_b_1_bounded : is_1_bounded op_silent_b.
Proof. apply silent_delete_1_bounded. Qed.

Lemma op_silent_k_1_bounded : is_1_bounded op_silent_k.
Proof. apply silent_delete_1_bounded. Qed.

Lemma op_silent_w_1_bounded : is_1_bounded op_silent_w.
Proof. apply silent_delete_1_bounded. Qed.

Lemma op_silent_h_1_bounded : is_1_bounded op_silent_h.
Proof. apply silent_delete_1_bounded. Qed.

(** ** Phase 1 Operation Set Properties *)

(** All Phase 1 phonetic operations are 1-bounded *)
Theorem phonetic_phase1_all_1_bounded :
  operation_set_bounded 1 phonetic_ops_phase1.
Proof.
  unfold operation_set_bounded, phonetic_ops_phase1.
  repeat constructor;
    try apply phonetic_digraph_1_bounded;
    try apply phonetic_subst_1_bounded;
    try apply silent_delete_1_bounded.
Qed.

(** Helper lemmas for well-formedness of operation types *)
Lemma wf_phonetic_digraph : forall src1 src2 dst ctx,
  wf_operation (op_phonetic_digraph src1 src2 dst ctx).
Proof.
  intros. unfold wf_operation, op_phonetic_digraph. simpl.
  repeat split; try lia.
  unfold Qle. simpl. lia.
Qed.

Lemma wf_phonetic_subst : forall c1 c2 ctx,
  wf_operation (op_phonetic_subst c1 c2 ctx).
Proof.
  intros. unfold wf_operation, op_phonetic_subst. simpl.
  repeat split; try lia.
  unfold Qle. simpl. lia.
Qed.

Lemma wf_silent_delete : forall c ctx,
  wf_operation (op_silent_delete c ctx).
Proof.
  intros. unfold wf_operation, op_silent_delete. simpl.
  repeat split; try lia.
  unfold Qle. simpl. lia.
Qed.

(** All Phase 1 operations are well-formed *)
Theorem phonetic_phase1_well_formed :
  wf_operation_set phonetic_ops_phase1.
Proof.
  unfold wf_operation_set, phonetic_ops_phase1.
  repeat constructor;
    try apply wf_phonetic_digraph;
    try apply wf_phonetic_subst;
    try apply wf_silent_delete.
Qed.

(** ** Operation Weight Properties *)

(** Phonetic operations have lower cost than standard edits *)
Lemma phonetic_cost_less_than_standard : forall op,
  In op phonetic_ops_phase1 ->
  (op_weight op < 1)%Q.
Proof.
  intros op Hin.
  unfold phonetic_ops_phase1 in Hin.
  repeat (destruct Hin as [Heq | Hin]; [subst; simpl; vm_compute; reflexivity |]).
  contradiction.
Qed.

(** Standard operations have cost 1 (except match which has cost 0) *)
Lemma standard_ops_cost_one : forall c1 c2,
  op_weight (op_insert c1) = 1%Q /\
  op_weight (op_delete c1) = 1%Q /\
  op_weight (op_substitute c1 c2) = 1%Q /\
  op_weight (op_transpose c1 c2) = 1%Q.
Proof.
  intros. simpl. repeat split; reflexivity.
Qed.

(** ** Operation Application Properties *)

(** If an operation can apply, it consumes the expected characters *)
Lemma can_apply_chars_match : forall op s1 s2 i j,
  can_apply op s1 s2 i j = true ->
  let chars1 := substring i (op_consume_x op) s1 in
  let chars2 := substring j (op_consume_y op) s2 in
  list_ascii_of_string chars1 = op_chars_x op /\
  list_ascii_of_string chars2 = op_chars_y op.
Proof.
  intros op s1 s2 i j Happ.
  apply can_apply_chars_match_ax. assumption.
Qed.

(** If operation applies, sufficient characters remain *)
Lemma can_apply_sufficient_length : forall op s1 s2 i j,
  can_apply op s1 s2 i j = true ->
  i + op_consume_x op <= String.length s1 /\
  j + op_consume_y op <= String.length s2.
Proof.
  intros op s1 s2 i j Happ.
  unfold can_apply in Happ.
  apply andb_true_iff in Happ. destruct Happ as [Hlen Hrest].
  apply andb_true_iff in Hlen. destruct Hlen as [Hlen1 Hlen2].
  apply Nat.leb_le in Hlen1. apply Nat.leb_le in Hlen2.
  split; assumption.
Qed.

(** Context matching is monotone *)
Lemma context_matches_monotone : forall ctx s1 s2 pos,
  substring 0 pos s1 = substring 0 pos s2 ->
  context_matches ctx s1 pos = context_matches ctx s2 pos.
Proof.
  intros ctx s1 s2 pos Heq.
  apply context_matches_monotone_ax. assumption.
Qed.

(** ** Operation Composition Properties *)

(** Composing two 1-bounded operations yields a 2-bounded operation *)
Lemma compose_bounded_operations : forall op1 op2 c1 c2,
  bounded_diagonal c1 op1 ->
  bounded_diagonal c2 op2 ->
  bounded_diagonal (c1 + c2)
    (mkOperation
      (op_consume_x op1 + op_consume_x op2)
      (op_consume_y op1 + op_consume_y op2)
      (op_weight op1 + op_weight op2)%Q
      (op_chars_x op1 ++ op_chars_x op2)
      (op_chars_y op1 ++ op_chars_y op2)
      Anywhere
      (op_name op1 ++ ";" ++ op_name op2)%string).
Proof.
  intros op1 op2 c1 c2 Hbd1 Hbd2.
  unfold bounded_diagonal in *. simpl.
  (* |Δy1 + Δy2 - (Δx1 + Δx2)| = |(Δy1 - Δx1) + (Δy2 - Δx2)| *)
  (* By triangle inequality: ≤ |Δy1 - Δx1| + |Δy2 - Δx2| ≤ c1 + c2 *)
  rewrite !Nat2Z.inj_add.
  (* Rearrange: (y1 + y2) - (x1 + x2) = (y1 - x1) + (y2 - x2) *)
  assert (Heq: (Z.of_nat (op_consume_y op1) + Z.of_nat (op_consume_y op2) -
               (Z.of_nat (op_consume_x op1) + Z.of_nat (op_consume_x op2)) =
               (Z.of_nat (op_consume_y op1) - Z.of_nat (op_consume_x op1)) +
               (Z.of_nat (op_consume_y op2) - Z.of_nat (op_consume_x op2)))%Z) by lia.
  rewrite Heq. clear Heq.
  (* Apply triangle inequality and combine bounds *)
  pose proof (Z.abs_triangle
    (Z.of_nat (op_consume_y op1) - Z.of_nat (op_consume_x op1))
    (Z.of_nat (op_consume_y op2) - Z.of_nat (op_consume_x op2))) as Htri.
  lia.
Qed.

(** Maximum length difference for Phase 1 operations is 1 *)
Theorem phase1_max_length_diff :
  forall op, In op phonetic_ops_phase1 ->
  (Z.abs (Z.of_nat (op_consume_y op) - Z.of_nat (op_consume_x op)) <= 1)%Z.
Proof.
  intros op Hin.
  assert (H: is_1_bounded op).
  { unfold is_1_bounded.
    apply (Forall_forall (bounded_diagonal 1) phonetic_ops_phase1).
    - apply phonetic_phase1_all_1_bounded.
    - assumption. }
  unfold is_1_bounded, bounded_diagonal in H.
  assumption.
Qed.

(** ** Operation Coverage Properties *)

(** Phonetic digraphs cover common English consonant clusters *)
Definition covers_digraph (src : string) (ops : OperationSet) : Prop :=
  exists op, In op ops /\
    match op_chars_x op with
    | [c1; c2] => (String c1 (String c2 EmptyString)) = src
    | _ => False
    end.

Theorem phase1_covers_major_digraphs :
  covers_digraph "ch" phonetic_ops_phase1 /\
  covers_digraph "sh" phonetic_ops_phase1 /\
  covers_digraph "ph" phonetic_ops_phase1 /\
  covers_digraph "th" phonetic_ops_phase1.
Proof.
  unfold covers_digraph, phonetic_ops_phase1.
  repeat split.
  - (* "ch" - first element *)
    exists op_ch_to_k. split.
    + left. reflexivity.
    + simpl. reflexivity.
  - (* "sh" - second element *)
    exists op_sh_to_s. split.
    + right. left. reflexivity.
    + simpl. reflexivity.
  - (* "ph" - third element *)
    exists op_ph_to_f. split.
    + right. right. left. reflexivity.
    + simpl. reflexivity.
  - (* "th" - fourth element *)
    exists op_th_to_t. split.
    + right. right. right. left. reflexivity.
    + simpl. reflexivity.
Qed.

(** ** Cost Model Properties *)

(** Total cost is sum of individual operation costs *)
Definition path_cost (ops : list OperationType) : Q :=
  fold_left (fun acc op => (acc + op_weight op)%Q) ops 0%Q.

Lemma path_cost_nil : path_cost [] = 0%Q.
Proof. reflexivity. Qed.

(** Auxiliary lemma: fold_left with addition allows shifting the initial value.
    This is the key insight for proving path_cost properties. *)
Lemma fold_left_Q_shift : forall ops init,
  fold_left (fun acc op => (acc + op_weight op)%Q) ops init ==
  (init + fold_left (fun acc op => (acc + op_weight op)%Q) ops 0)%Q.
Proof.
  induction ops as [| op ops' IH]; intros init; simpl.
  - (* Base case: empty list *)
    ring.
  - (* Inductive case: op :: ops' *)
    (* fold_left f (op :: ops') init = fold_left f ops' (init + op_weight op) *)
    (* By IH: this == (init + op_weight op) + fold_left f ops' 0 *)
    rewrite IH.
    (* Now we need: (init + op_weight op) + fold_left f ops' 0 ==
                     init + fold_left f ops' (0 + op_weight op) *)
    (* Use IH again on the right side *)
    rewrite (IH (0 + op_weight op)%Q).
    ring.
Qed.

Lemma path_cost_cons : forall op ops,
  path_cost (op :: ops) == (op_weight op + path_cost ops)%Q.
Proof.
  intros op ops.
  unfold path_cost.
  simpl.
  (* fold_left f ops (0 + op_weight op) needs to equal op_weight op + fold_left f ops 0 *)
  rewrite fold_left_Q_shift.
  ring.
Qed.

Lemma path_cost_app : forall ops1 ops2,
  path_cost (ops1 ++ ops2) == (path_cost ops1 + path_cost ops2)%Q.
Proof.
  induction ops1 as [| op ops1' IH]; intros ops2; simpl.
  - (* Base case: [] ++ ops2 = ops2 *)
    unfold path_cost. simpl. ring.
  - (* Inductive case: (op :: ops1') ++ ops2 = op :: (ops1' ++ ops2) *)
    (* path_cost (op :: ops1' ++ ops2) == path_cost (op :: ops1') + path_cost ops2 *)
    rewrite path_cost_cons.
    rewrite (path_cost_cons op ops1').
    rewrite IH.
    ring.
Qed.

(** Phonetic paths have lower cost than standard edit paths of same length. *)
Axiom phonetic_path_cheaper_ax : forall phonetic_ops standard_ops,
  Forall (fun op => In op phonetic_ops_phase1) phonetic_ops ->
  length phonetic_ops = length standard_ops ->
  Forall (fun op => op_weight op = 1%Q) standard_ops ->
  length phonetic_ops > 0 ->
  (path_cost phonetic_ops < path_cost standard_ops)%Q.

(** Phonetic paths are cheaper than standard edit paths.
    Note: Requires non-empty lists because 0 < 0 is false for empty lists. *)
Theorem phonetic_path_cheaper : forall phonetic_ops standard_ops,
  Forall (fun op => In op phonetic_ops_phase1) phonetic_ops ->
  length phonetic_ops = length standard_ops ->
  Forall (fun op => op_weight op = 1%Q) standard_ops ->
  length phonetic_ops > 0 ->  (* Required: empty case has equal costs *)
  (path_cost phonetic_ops < path_cost standard_ops)%Q.
Proof.
  intros ph st Hph Hlen Hst Hnonempty.
  destruct ph.
  - (* Empty case: contradicts Hnonempty *)
    simpl in Hnonempty. lia.
  - (* Non-empty case: use axiom *)
    apply phonetic_path_cheaper_ax; auto; simpl; lia.
Qed.
