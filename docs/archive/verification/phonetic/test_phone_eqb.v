Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
Import ListNotations.

Goal forall (ph_first : Phone) (pat_rest : list Phone) (s : PhoneticString) (p : nat) (s_ph : Phone),
  nth_error s p = Some s_ph ->
  Phone_eqb s_ph ph_first = true ->
  pattern_matches_at (ph_first :: pat_rest) s p = false ->
  False.
Proof.
  intros ph_first pat_rest s p s_ph H_nth H_eq H_no_match.
  unfold pattern_matches_at in H_no_match.
  simpl in H_no_match.
  rewrite H_nth in H_no_match.
  Show.
Abort.
