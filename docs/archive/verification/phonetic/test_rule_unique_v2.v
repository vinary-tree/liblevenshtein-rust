From Stdlib Require Import String List Arith Ascii Bool Nat Lia.
From Stdlib Require Import PeanoNat.
Import ListNotations.

(* Load the phonetic definitions *)
Load "rewrite_rules".
Load "zompist_rules".

(* Define the complete list of all Zompist rules *)
Definition all_zompist_rules : list RewriteRule :=
  [ rule_ch_to_tsh; rule_sh_to_sh; rule_ph_to_f;
    rule_c_to_s_before_front; rule_c_to_k_elsewhere; rule_g_to_j_before_front;
    rule_silent_e_final; rule_gh_silent;
    phonetic_th_to_t; phonetic_qu_to_kw; phonetic_kw_to_qu;
    rule_x_expand; rule_y_to_z ].

(* Extract rule IDs *)
Definition all_rule_ids : list nat := map rule_id all_zompist_rules.

(* Verify the IDs *)
Eval compute in all_rule_ids.

(* Prove NoDup using decidable equality and computation *)
Lemma NoDup_13_distinct_nats :
  NoDup [1; 2; 3; 20; 21; 22; 33; 34; 100; 101; 102; 200; 201].
Proof.
  repeat constructor; simpl; intuition discriminate.
Qed.

(* Prove the IDs match the expected list *)
Lemma all_rule_ids_eq :
  all_rule_ids = [1; 2; 3; 20; 21; 22; 33; 34; 100; 101; 102; 200; 201].
Proof. reflexivity. Qed.

(* NoDup for rule IDs follows *)
Lemma all_rule_ids_NoDup : NoDup all_rule_ids.
Proof.
  rewrite all_rule_ids_eq.
  exact NoDup_13_distinct_nats.
Qed.

(* Helper: If two rules have the same ID and are in a NoDup-ID list, they're at the same position *)
Lemma In_map_same_id_unique :
  forall (l : list RewriteRule) (r1 r2 : RewriteRule),
    NoDup (map rule_id l) ->
    In r1 l -> In r2 l ->
    rule_id r1 = rule_id r2 ->
    r1 = r2.
Proof.
  induction l as [| x xs IH]; intros r1 r2 Hnodup Hin1 Hin2 Heq.
  - (* empty list *) destruct Hin1.
  - (* cons case *)
    simpl in Hnodup. inversion Hnodup as [| ? ? Hnotin Hnodup_xs]. subst.
    destruct Hin1 as [H1 | H1]; destruct Hin2 as [H2 | H2].
    + (* r1 = x, r2 = x *) congruence.
    + (* r1 = x, r2 in xs *)
      subst r1. exfalso.
      apply Hnotin. rewrite in_map_iff.
      exists r2. split; [exact Heq | exact H2].
    + (* r1 in xs, r2 = x *)
      subst r2. exfalso.
      apply Hnotin. rewrite in_map_iff.
      exists r1. split; [symmetry; exact Heq | exact H1].
    + (* r1 in xs, r2 in xs *)
      apply IH; assumption.
Qed.

(* The closed-world theorem *)
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

Print Assumptions rule_id_unique_in_zompist.
