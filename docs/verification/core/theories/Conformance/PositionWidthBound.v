(** A full-range pair of machine-word coordinates cannot fit in one word.

    The model deliberately ignores the continuation tag and auxiliary byte:
    the two public [usize] coordinates alone already exceed an eight-byte
    representation on a 64-bit target.  The proof is a finite pigeonhole
    argument, independent of any particular Rust struct layout. *)

From Stdlib Require Import List Arith Lia.
Import ListNotations.

Lemma nodup_image_of_injective_range :
  forall (A B : Type) (f : A -> B) (xs : list A),
    NoDup xs ->
    (forall x y, In x xs -> In y xs -> f x = f y -> x = y) ->
    NoDup (map f xs).
Proof.
  intros A B f xs Hnodup.
  induction Hnodup as [| x xs Hnotin Hnodup IH]; intros Hinjective.
  - constructor.
  - simpl. constructor.
    + intro Hmapped.
      apply in_map_iff in Hmapped.
      destruct Hmapped as [y [Hequal Hy]].
      apply Hnotin.
      assert (x = y) as Hxy.
      { apply (Hinjective x y).
        - simpl; auto.
        - simpl; auto.
        - symmetry; exact Hequal. }
      subst; exact Hy.
    + apply IH.
      intros a b Ha Hb Hequal.
      apply (Hinjective a b); simpl; auto.
Qed.

Theorem full_range_pair_cannot_fit_one_word :
  forall (word_values : nat) (encode : nat -> nat -> nat),
    1 < word_values ->
    (forall index errors,
        index < word_values -> errors < word_values ->
        encode index errors < word_values) ->
    (forall i e j f,
        i < word_values -> e < word_values ->
        j < word_values -> f < word_values ->
        encode i e = encode j f -> i = j /\ e = f) ->
    False.
Proof.
  intros n encode Hn Hrange Hinjective.
  set (indices := seq 0 n).
  set (zeroes := map (fun i => encode i 0) indices).
  set (ones := map (fun i => encode i 1) indices).
  assert (Hzeroes : NoDup zeroes).
  { unfold zeroes.
    apply nodup_image_of_injective_range.
    - apply seq_NoDup.
    - intros i j Hi Hj Hequal.
      apply in_seq in Hi. apply in_seq in Hj.
      assert (i = j /\ 0 = 0) as Hpair.
      { apply Hinjective; try lia; exact Hequal. }
      exact (proj1 Hpair). }
  assert (Hones : NoDup ones).
  { unfold ones.
    apply nodup_image_of_injective_range.
    - apply seq_NoDup.
    - intros i j Hi Hj Hequal.
      apply in_seq in Hi. apply in_seq in Hj.
      assert (i = j /\ 1 = 1) as Hpair.
      { apply Hinjective; try lia; exact Hequal. }
      exact (proj1 Hpair). }
  assert (Hboth : NoDup (zeroes ++ ones)).
  { apply NoDup_app; try assumption.
    intros value Hz Ho.
    unfold zeroes in Hz. unfold ones in Ho.
    apply in_map_iff in Hz. apply in_map_iff in Ho.
    destruct Hz as [i [Hzi Hi]].
    destruct Ho as [j [Hoj Hj]].
    apply in_seq in Hi. apply in_seq in Hj.
    assert (i = j /\ 0 = 1) as Hpair.
    { apply Hinjective; lia. }
    destruct Hpair as [_ Hcontradiction].
    discriminate Hcontradiction. }
  assert (Hinclusion : incl (zeroes ++ ones) (seq 0 n)).
  { intros value Hvalue.
    apply in_app_or in Hvalue.
    destruct Hvalue as [Hvalue | Hvalue].
    - unfold zeroes in Hvalue.
      apply in_map_iff in Hvalue.
      destruct Hvalue as [i [Hvalue Hi]].
      apply in_seq in Hi.
      assert (encode i 0 < n) by (apply Hrange; lia).
      apply in_seq. subst value; simpl; lia.
    - unfold ones in Hvalue.
      apply in_map_iff in Hvalue.
      destruct Hvalue as [i [Hvalue Hi]].
      apply in_seq in Hi.
      assert (encode i 1 < n) by (apply Hrange; lia).
      apply in_seq. subst value; simpl; lia. }
  pose proof (NoDup_incl_length Hboth Hinclusion) as Hlength.
  unfold zeroes, ones, indices in Hlength.
  repeat rewrite length_app in Hlength.
  repeat rewrite length_map in Hlength.
  repeat rewrite length_seq in Hlength.
  lia.
Qed.

Corollary no_lossless_eight_byte_pair_of_usize :
  forall encode : nat -> nat -> nat,
    (forall i e, i < 2 ^ 64 -> e < 2 ^ 64 -> encode i e < 2 ^ 64) ->
    (forall i e j f,
        i < 2 ^ 64 -> e < 2 ^ 64 ->
        j < 2 ^ 64 -> f < 2 ^ 64 ->
        encode i e = encode j f -> i = j /\ e = f) ->
    False.
Proof.
  intros encode Hrange Hinjective.
  eapply (full_range_pair_cannot_fit_one_word (2 ^ 64) encode).
  - apply Nat.pow_gt_1; lia.
  - exact Hrange.
  - exact Hinjective.
Qed.
