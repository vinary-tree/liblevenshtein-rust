Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
Import ListNotations.

Lemma Phone_eqb_sym : forall p1 p2, Phone_eqb p1 p2 = Phone_eqb p2 p1.
Proof.
  intros p1 p2.
  destruct p1, p2; unfold Phone_eqb; simpl;
  try reflexivity;
  try (apply Ascii.eqb_sym).
  (* Digraph case *)
  f_equal; apply Ascii.eqb_sym.
Qed.
