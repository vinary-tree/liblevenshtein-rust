(** Test: Prove rule_id_unique for closed-world Zompist rules *)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia QArith.
Import ListNotations.
Open Scope Q_scope.

(* Load the rewrite rules module directly *)
Load "rewrite_rules".

(* Include all zompist rule definitions inline since Load has issues with Require *)

Module ZompistRulesLocal.

(** ASCII constants for common characters *)
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

Definition front_vowels : list ascii := [e_ascii; i_ascii].

(** Rule definitions *)
Definition rule_ch_to_tsh : RewriteRule := {|
  rule_id := 1; rule_name := "ch → ç"; 
  pattern := [Consonant c_ascii; Consonant h_ascii];
  replacement := [Digraph c_ascii h_ascii]; context := Anywhere; weight := 0;
|}.

Definition rule_sh_to_sh : RewriteRule := {|
  rule_id := 2; rule_name := "sh → $";
  pattern := [Consonant s_ascii; Consonant h_ascii];
  replacement := [Digraph s_ascii h_ascii]; context := Anywhere; weight := 0;
|}.

Definition rule_ph_to_f : RewriteRule := {|
  rule_id := 3; rule_name := "ph → f";
  pattern := [Consonant p_ascii; Consonant h_ascii];
  replacement := [Consonant f_ascii]; context := Anywhere; weight := 0;
|}.

Definition rule_c_to_s_before_front : RewriteRule := {|
  rule_id := 20; rule_name := "c → s / _[ie]";
  pattern := [Consonant c_ascii]; replacement := [Consonant s_ascii];
  context := BeforeVowel front_vowels; weight := 0;
|}.

Definition rule_c_to_k_elsewhere : RewriteRule := {|
  rule_id := 21; rule_name := "c → k";
  pattern := [Consonant c_ascii]; replacement := [Consonant k_ascii];
  context := Anywhere; weight := 0;
|}.

Definition rule_g_to_j_before_front : RewriteRule := {|
  rule_id := 22; rule_name := "g → j / _[ie]";
  pattern := [Consonant g_ascii]; replacement := [Consonant "106"];
  context := BeforeVowel front_vowels; weight := 0;
|}.

Definition rule_silent_e_final : RewriteRule := {|
  rule_id := 33; rule_name := "e → ∅ / _#";
  pattern := [Vowel e_ascii]; replacement := [Silent];
  context := Final; weight := 0;
|}.

Definition rule_gh_silent : RewriteRule := {|
  rule_id := 34; rule_name := "gh → ∅";
  pattern := [Consonant g_ascii; Consonant h_ascii];
  replacement := [Silent]; context := Anywhere; weight := 0;
|}.

Definition phonetic_th_to_t : RewriteRule := {|
  rule_id := 100; rule_name := "th → t";
  pattern := [Consonant t_ascii; Consonant h_ascii];
  replacement := [Consonant t_ascii]; context := Anywhere; weight := 15 # 100;
|}.

Definition phonetic_qu_to_kw : RewriteRule := {|
  rule_id := 101; rule_name := "qu → kw";
  pattern := [Consonant q_ascii; Consonant u_ascii];
  replacement := [Consonant k_ascii; Consonant w_ascii]; context := Anywhere; weight := 15 # 100;
|}.

Definition phonetic_kw_to_qu : RewriteRule := {|
  rule_id := 102; rule_name := "kw → qu";
  pattern := [Consonant k_ascii; Consonant w_ascii];
  replacement := [Consonant q_ascii; Consonant u_ascii]; context := Anywhere; weight := 15 # 100;
|}.

Definition rule_x_expand : RewriteRule := {|
  rule_id := 200; rule_name := "x → yy";
  pattern := [Consonant x_ascii];
  replacement := [Consonant y_ascii; Consonant y_ascii]; context := Anywhere; weight := 0;
|}.

Definition rule_y_to_z : RewriteRule := {|
  rule_id := 201; rule_name := "y → z";
  pattern := [Consonant y_ascii]; replacement := [Consonant z_ascii];
  context := Anywhere; weight := 0;
|}.

(** Complete list of all rules *)
Definition all_zompist_rules : list RewriteRule :=
  [ rule_ch_to_tsh; rule_sh_to_sh; rule_ph_to_f;
    rule_c_to_s_before_front; rule_c_to_k_elsewhere; rule_g_to_j_before_front;
    rule_silent_e_final; rule_gh_silent;
    phonetic_th_to_t; phonetic_qu_to_kw; phonetic_kw_to_qu;
    rule_x_expand; rule_y_to_z ].

Definition all_rule_ids : list nat := map rule_id all_zompist_rules.

End ZompistRulesLocal.

Import ZompistRulesLocal.

(** Compute to verify IDs *)
Eval compute in all_rule_ids.

(** Helper: Check if a nat is in a list *)
Fixpoint nat_In (n : nat) (l : list nat) : bool :=
  match l with
  | [] => false
  | x :: xs => if Nat.eqb n x then true else nat_In n xs
  end.

(** NoDup decidable check *)
Fixpoint NoDup_dec_nat (l : list nat) : bool :=
  match l with
  | [] => true
  | x :: xs => if nat_In x xs then false else NoDup_dec_nat xs
  end.

(** Check IDs are unique by computation *)
Eval compute in NoDup_dec_nat all_rule_ids.

(** nat_In reflects In *)
Lemma nat_In_reflect : forall n l, nat_In n l = true <-> In n l.
Proof.
  intros n l. induction l as [| x xs IH].
  - simpl. split; discriminate.
  - simpl. destruct (Nat.eqb_spec n x).
    + subst. split; [left; auto | auto].
    + rewrite IH. split; [right; auto | intros [H | H]; [congruence | auto]].
Qed.

Lemma nat_In_reflect_false : forall n l, nat_In n l = false <-> ~In n l.
Proof.
  intros n l. rewrite <- nat_In_reflect.
  destruct (nat_In n l); split; auto; discriminate.
Qed.

(** NoDup_dec_nat reflects NoDup *)
Lemma NoDup_dec_nat_reflect : forall l, NoDup_dec_nat l = true <-> NoDup l.
Proof.
  intros l. induction l as [| x xs IH].
  - simpl. split; [constructor | auto].
  - simpl. destruct (nat_In x xs) eqn:E.
    + split; [discriminate | intros H; inversion H; apply nat_In_reflect in E; contradiction].
    + rewrite IH. split.
      * intros H. constructor; [apply nat_In_reflect_false; exact E | exact H].
      * intros H. inversion H; auto.
Qed.

(** Prove all rule IDs have no duplicates *)
Lemma all_rule_ids_NoDup : NoDup all_rule_ids.
Proof. apply NoDup_dec_nat_reflect. reflexivity. Qed.

(** Key lemma: If rules have the same ID, they are equal *)
Lemma rule_id_unique_in_zompist :
  forall r1 r2 : RewriteRule,
    In r1 all_zompist_rules -> In r2 all_zompist_rules ->
    rule_id r1 = rule_id r2 -> r1 = r2.
Proof.
  intros r1 r2 Hin1 Hin2 Heq.
  unfold all_zompist_rules in Hin1, Hin2.
  simpl in Hin1, Hin2.
  repeat (destruct Hin1 as [Hin1 | Hin1]);
  try (subst r1);
  repeat (destruct Hin2 as [Hin2 | Hin2]);
  try (subst r2);
  try reflexivity;
  try (simpl in Heq; discriminate);
  try contradiction.
Qed.

Print "SUCCESS: rule_id_unique_in_zompist proven!"
Print Assumptions rule_id_unique_in_zompist.

