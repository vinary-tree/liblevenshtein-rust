Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Types.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Lib.
From Liblevenshtein.Phonetic.Verification Require Import Core.Rules.
From Liblevenshtein.Phonetic.Verification Require Import Patterns.PatternHelpers.
Import ListNotations.

Lemma leftmost_mismatch_before_transformation :
  forall pat s p pos i_left,
    (p < pos)%nat ->
    (pos < p + length pat)%nat ->
    (p <= i_left < p + length pat)%nat ->
    (* Pattern doesn't match *)
    pattern_matches_at pat s p = false ->
    (* i_left is leftmost mismatch *)
    (nth_error s i_left = None \/
     exists ph pat_ph,
       nth_error s i_left = Some ph /\
       nth_error pat (i_left - p) = Some pat_ph /\
       Phone_eqb ph pat_ph = false) ->
    (* All positions before i_left match *)
    (forall j, (p <= j < i_left)%nat ->
      exists s_ph pat_ph,
        nth_error s j = Some s_ph /\
        nth_error pat (j - p) = Some pat_ph /\
        Phone_eqb s_ph pat_ph = true) ->
    (* Then leftmost mismatch before transformation *)
    (i_left < pos)%nat.
Proof.
  intros pat s p pos i_left H_p_lt_pos H_overlap H_i_range H_no_match H_i_mismatch H_all_before.
  destruct (lt_dec i_left pos) as [H_lt | H_ge]; [exact H_lt |].
  exfalso.
  assert (H_prefix_len: (pos - p > 0)%nat) by lia.
  assert (H_prefix_matches: forall j, (p <= j < pos)%nat ->
    exists s_ph pat_ph,
      nth_error s j = Some s_ph /\
      nth_error pat (j - p) = Some pat_ph /\
      Phone_eqb s_ph pat_ph = true).
  { intros j H_j_range. apply H_all_before. lia. }

  Show. (* Show context BEFORE generalize *)
Abort.
