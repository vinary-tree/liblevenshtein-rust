(** Standalone test for rule_id_unique closed-world proof *)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia.
From Stdlib Require Import PeanoNat QArith.
Import ListNotations.
Open Scope Q_scope.

(** * Phone Type (from rewrite_rules.v) *)
Module Phone.
  Inductive t : Set :=
  | Vowel : ascii -> t
  | Consonant : ascii -> t
  | Digraph : ascii -> ascii -> t
  | Silent : t.
End Phone.

Definition Phone := Phone.t.
Definition Vowel := Phone.Vowel.
Definition Consonant := Phone.Consonant.
Definition Digraph := Phone.Digraph.
Definition Silent := Phone.Silent.

(** * Context Type *)
Inductive Context : Set :=
| Initial : Context
| Final : Context
| BeforeVowel : list ascii -> Context
| AfterConsonant : list ascii -> Context
| BeforeConsonant : list ascii -> Context
| AfterVowel : list ascii -> Context
| Anywhere : Context.

(** * RewriteRule Record *)
Record RewriteRule : Set := mkRule {
  rule_id : nat;
  rule_name : string;
  pattern : list Phone;
  replacement : list Phone;
  context : Context;
  weight : Q;
}.

(** * ASCII constants *)
Definition a_ascii : ascii := "097".
Definition e_ascii : ascii := "101".
Definition i_ascii : ascii := "105".
Definition o_ascii : ascii := "111".
Definition u_ascii : ascii := "117".
Definition c_ascii : ascii := "099".
Definition h_ascii : ascii := "104".
Definition p_ascii : ascii := "112".
Definition s_ascii : ascii := "115".
Definition t_ascii : ascii := "116".
Definition k_ascii : ascii := "107".
Definition f_ascii : ascii := "102".
Definition g_ascii : ascii := "103".
Definition q_ascii : ascii := "113".
Definition w_ascii : ascii := "119".
Definition x_ascii : ascii := "120".
Definition y_ascii : ascii := "121".
Definition z_ascii : ascii := "122".

(** Front vowels *)
Definition front_vowels : list ascii := [e_ascii; i_ascii].

(** * Rule Definitions (from zompist_rules.v) *)

(** Rule 1 *)
Definition rule_ch_to_tsh : RewriteRule := {|
  rule_id := 1;
  rule_name := "ch → ç (tsh sound)";
  pattern := [Consonant c_ascii; Consonant h_ascii];
  replacement := [Digraph c_ascii h_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 2 *)
Definition rule_sh_to_sh : RewriteRule := {|
  rule_id := 2;
  rule_name := "sh → $ (sh sound)";
  pattern := [Consonant s_ascii; Consonant h_ascii];
  replacement := [Digraph s_ascii h_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 3 *)
Definition rule_ph_to_f : RewriteRule := {|
  rule_id := 3;
  rule_name := "ph → f";
  pattern := [Consonant p_ascii; Consonant h_ascii];
  replacement := [Consonant f_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 20 *)
Definition rule_c_to_s_before_front : RewriteRule := {|
  rule_id := 20;
  rule_name := "c → s / _[ie]";
  pattern := [Consonant c_ascii];
  replacement := [Consonant s_ascii];
  context := BeforeVowel front_vowels;
  weight := 0;
|}.

(** Rule 21 *)
Definition rule_c_to_k_elsewhere : RewriteRule := {|
  rule_id := 21;
  rule_name := "c → k (elsewhere)";
  pattern := [Consonant c_ascii];
  replacement := [Consonant k_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 22 *)
Definition rule_g_to_j_before_front : RewriteRule := {|
  rule_id := 22;
  rule_name := "g → j / _[ie]";
  pattern := [Consonant g_ascii];
  replacement := [Consonant "106"];
  context := BeforeVowel front_vowels;
  weight := 0;
|}.

(** Rule 33 *)
Definition rule_silent_e_final : RewriteRule := {|
  rule_id := 33;
  rule_name := "e → ∅ / _#";
  pattern := [Vowel e_ascii];
  replacement := [Silent];
  context := Final;
  weight := 0;
|}.

(** Rule 34 *)
Definition rule_gh_silent : RewriteRule := {|
  rule_id := 34;
  rule_name := "gh → ∅";
  pattern := [Consonant g_ascii; Consonant h_ascii];
  replacement := [Silent];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 100 *)
Definition phonetic_th_to_t : RewriteRule := {|
  rule_id := 100;
  rule_name := "th → t (phonetic)";
  pattern := [Consonant t_ascii; Consonant h_ascii];
  replacement := [Consonant t_ascii];
  context := Anywhere;
  weight := 15 # 100;
|}.

(** Rule 101 *)
Definition phonetic_qu_to_kw : RewriteRule := {|
  rule_id := 101;
  rule_name := "qu → kw (phonetic)";
  pattern := [Consonant q_ascii; Consonant u_ascii];
  replacement := [Consonant k_ascii; Consonant w_ascii];
  context := Anywhere;
  weight := 15 # 100;
|}.

(** Rule 102 *)
Definition phonetic_kw_to_qu : RewriteRule := {|
  rule_id := 102;
  rule_name := "kw → qu (phonetic reverse)";
  pattern := [Consonant k_ascii; Consonant w_ascii];
  replacement := [Consonant q_ascii; Consonant u_ascii];
  context := Anywhere;
  weight := 15 # 100;
|}.

(** Rule 200 *)
Definition rule_x_expand : RewriteRule := {|
  rule_id := 200;
  rule_name := "x → yy (expansion test)";
  pattern := [Consonant x_ascii];
  replacement := [Consonant y_ascii; Consonant y_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** Rule 201 *)
Definition rule_y_to_z : RewriteRule := {|
  rule_id := 201;
  rule_name := "y → z (transformation test)";
  pattern := [Consonant y_ascii];
  replacement := [Consonant z_ascii];
  context := Anywhere;
  weight := 0;
|}.

(** * Complete List of All Rules *)

Definition all_zompist_rules : list RewriteRule :=
  [ rule_ch_to_tsh; rule_sh_to_sh; rule_ph_to_f;
    rule_c_to_s_before_front; rule_c_to_k_elsewhere; rule_g_to_j_before_front;
    rule_silent_e_final; rule_gh_silent;
    phonetic_th_to_t; phonetic_qu_to_kw; phonetic_kw_to_qu;
    rule_x_expand; rule_y_to_z ].

Definition all_rule_ids : list nat := map rule_id all_zompist_rules.

(** Verify the IDs compute correctly *)
Eval compute in all_rule_ids.
(* Should be: [1; 2; 3; 20; 21; 22; 33; 34; 100; 101; 102; 200; 201] *)

(** * NoDup Proof *)

(** Prove NoDup for the literal list of IDs *)
Lemma NoDup_13_distinct_nats :
  NoDup [1%nat; 2%nat; 3%nat; 20%nat; 21%nat; 22%nat; 33%nat; 34%nat; 100%nat; 101%nat; 102%nat; 200%nat; 201%nat].
Proof.
  repeat constructor; simpl; intuition discriminate.
Qed.

(** Prove the IDs equal the expected list *)
Lemma all_rule_ids_eq :
  all_rule_ids = [1%nat; 2%nat; 3%nat; 20%nat; 21%nat; 22%nat; 33%nat; 34%nat; 100%nat; 101%nat; 102%nat; 200%nat; 201%nat].
Proof. reflexivity. Qed.

(** NoDup for rule IDs *)
Lemma all_rule_ids_NoDup : NoDup all_rule_ids.
Proof.
  rewrite all_rule_ids_eq.
  exact NoDup_13_distinct_nats.
Qed.

(** * Uniqueness Theorem *)

(** If two rules have the same ID and are in a NoDup-ID list, they must be equal *)
Lemma In_map_same_id_unique :
  forall (l : list RewriteRule) (r1 r2 : RewriteRule),
    NoDup (map rule_id l) ->
    In r1 l -> In r2 l ->
    rule_id r1 = rule_id r2 ->
    r1 = r2.
Proof.
  induction l as [| x xs IH]; intros r1 r2 Hnodup Hin1 Hin2 Heq.
  - destruct Hin1.
  - simpl in Hnodup. inversion Hnodup as [| ? ? Hnotin Hnodup_xs]. subst.
    destruct Hin1 as [H1 | H1]; destruct Hin2 as [H2 | H2].
    + congruence.
    + subst r1. exfalso.
      apply Hnotin. rewrite in_map_iff.
      exists r2. split; [symmetry; exact Heq | exact H2].
    + subst r2. exfalso.
      apply Hnotin. rewrite in_map_iff.
      exists r1. split; [exact Heq | exact H1].
    + apply IH; assumption.
Qed.

(** The closed-world theorem for Zompist rules *)
Theorem rule_id_unique_in_zompist :
  forall r1 r2 : RewriteRule,
    In r1 all_zompist_rules ->
    In r2 all_zompist_rules ->
    rule_id r1 = rule_id r2 ->
    r1 = r2.
Proof.
  intros r1 r2 Hin1 Hin2 Heq.
  apply (In_map_same_id_unique all_zompist_rules r1 r2 all_rule_ids_NoDup Hin1 Hin2 Heq).
Qed.

(** Check axiom dependencies *)
Print Assumptions rule_id_unique_in_zompist.

(** Should print "Closed under the global context" *)
