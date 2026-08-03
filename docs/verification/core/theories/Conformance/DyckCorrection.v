(** * Exact multi-kind Dyck-correction invariants

    This assumption-free model mirrors the four backpointer constructors in
    [src/transducer/language/dyck.rs].  It proves that every reconstructed
    target is kind-sensitive Dyck, that zero-cost reconstruction is exactly a
    balanced identity, that deletion supplies a correction for every source,
    and that the typed Dyck grammar has the first-pair decomposition used by
    the interval recurrence.
*)

From Stdlib Require Import Arith Lia List PeanoNat.
Import ListNotations.

Definition replacement_cost (actual expected : nat) : nat :=
  if Nat.eq_dec actual expected then 0 else 1.

Lemma replacement_cost_zero_iff : forall actual expected,
  replacement_cost actual expected = 0 <-> actual = expected.
Proof.
  intros actual expected; unfold replacement_cost.
  destruct (Nat.eq_dec actual expected); [tauto | lia].
Qed.

(** Opening kind [r] is token [r]; its closer is [kinds + r]. *)
Inductive dyck (kinds : nat) : list nat -> Prop :=
  | dyck_empty : dyck kinds []
  | dyck_concat : forall left right,
      dyck kinds left ->
      dyck kinds right ->
      dyck kinds (left ++ right)
  | dyck_wrap : forall kind inner,
      kind < kinds ->
      dyck kinds inner ->
      dyck kinds (kind :: inner ++ [kinds + kind]).

(** A correction tree is the normal form reconstructed by the Rust interval
    DP.  The constructors are, in order: empty, delete-first, use the first
    source token with an inserted closer, pair two consumed source tokens, and
    insert an opener before a consumed closer. *)
Inductive correction_tree (kinds : nat) : list nat -> list nat -> nat -> Prop :=
  | correction_empty : correction_tree kinds [] [] 0
  | correction_delete : forall actual source target cost,
      correction_tree kinds source target cost ->
      correction_tree kinds (actual :: source) target (S cost)
  | correction_insert_close : forall actual source suffix kind suffix_cost,
      kind < kinds ->
      correction_tree kinds source suffix suffix_cost ->
      correction_tree kinds
        (actual :: source)
        (kind :: (kinds + kind) :: suffix)
        (replacement_cost actual kind + 1 + suffix_cost)
  | correction_pair :
      forall actual_open source_inner actual_close source_suffix
             target_inner target_suffix kind inner_cost suffix_cost,
      kind < kinds ->
      correction_tree kinds source_inner target_inner inner_cost ->
      correction_tree kinds source_suffix target_suffix suffix_cost ->
      correction_tree kinds
        (actual_open :: source_inner ++ actual_close :: source_suffix)
        (kind :: target_inner ++ (kinds + kind) :: target_suffix)
        (replacement_cost actual_open kind + inner_cost +
         replacement_cost actual_close (kinds + kind) + suffix_cost)
  | correction_insert_open :
      forall source_inner actual_close source_suffix
             target_inner target_suffix kind inner_cost suffix_cost,
      kind < kinds ->
      correction_tree kinds source_inner target_inner inner_cost ->
      correction_tree kinds source_suffix target_suffix suffix_cost ->
      correction_tree kinds
        (source_inner ++ actual_close :: source_suffix)
        (kind :: target_inner ++ (kinds + kind) :: target_suffix)
        (1 + inner_cost + replacement_cost actual_close (kinds + kind) + suffix_cost).

Theorem correction_target_is_dyck : forall kinds source target cost,
  correction_tree kinds source target cost -> dyck kinds target.
Proof.
  intros kinds source target cost H; induction H.
  - constructor.
  - exact IHcorrection_tree.
  - replace (kind :: (kinds + kind) :: suffix)
      with ((kind :: [] ++ [kinds + kind]) ++ suffix) by reflexivity.
    apply dyck_concat.
    + apply dyck_wrap; [assumption | constructor].
    + exact IHcorrection_tree.
  - replace (kind :: target_inner ++ (kinds + kind) :: target_suffix)
      with ((kind :: target_inner ++ [kinds + kind]) ++ target_suffix).
    + apply dyck_concat.
      * apply dyck_wrap; assumption.
      * exact IHcorrection_tree2.
    + simpl; rewrite <- app_assoc; reflexivity.
  - replace (kind :: target_inner ++ (kinds + kind) :: target_suffix)
      with ((kind :: target_inner ++ [kinds + kind]) ++ target_suffix).
    + apply dyck_concat.
      * apply dyck_wrap; assumption.
      * exact IHcorrection_tree2.
    + simpl; rewrite <- app_assoc; reflexivity.
Qed.

Lemma correction_cost_zero_is_balanced_identity :
  forall kinds source target cost,
    correction_tree kinds source target cost ->
    cost = 0 ->
    source = target /\ dyck kinds source.
Proof.
  intros kinds source target cost H; induction H; intros Hcost.
  - split; [reflexivity | constructor].
  - lia.
  - lia.
  - assert (Hopen : replacement_cost actual_open kind = 0) by lia.
    assert (Hinner : inner_cost = 0) by lia.
    assert (Hclose : replacement_cost actual_close (kinds + kind) = 0) by lia.
    assert (Hsuffix : suffix_cost = 0) by lia.
    apply (proj1 (replacement_cost_zero_iff actual_open kind)) in Hopen.
    apply (proj1 (replacement_cost_zero_iff actual_close (kinds + kind))) in Hclose.
    specialize (IHcorrection_tree1 Hinner).
    specialize (IHcorrection_tree2 Hsuffix).
    destruct IHcorrection_tree1 as [Hinner_eq Hinner_dyck].
    destruct IHcorrection_tree2 as [Hsuffix_eq Hsuffix_dyck].
    subst; split; [reflexivity |].
    replace (kind :: target_inner ++ (kinds + kind) :: target_suffix)
      with ((kind :: target_inner ++ [kinds + kind]) ++ target_suffix).
    + apply dyck_concat.
      * apply dyck_wrap; assumption.
      * assumption.
    + simpl; rewrite <- app_assoc; reflexivity.
  - lia.
Qed.

Theorem zero_cost_correction_is_balanced_identity :
  forall kinds source target,
    correction_tree kinds source target 0 ->
    source = target /\ dyck kinds source.
Proof.
  intros; eapply correction_cost_zero_is_balanced_identity; eauto.
Qed.

Theorem every_source_has_a_correction : forall kinds source,
  correction_tree kinds source [] (length source).
Proof.
  intros kinds source; induction source as [|token rest IH].
  - constructor.
  - simpl; constructor; exact IH.
Qed.

(** A relational minimum avoids hiding any executable or classical choice.
    It says that [cost] is attained and is no greater than every correction
    witness for the same source interval. *)
Definition minimum_correction_cost
    (kinds : nat) (source : list nat) (cost : nat) : Prop :=
  (exists target, correction_tree kinds source target cost) /\
  forall target other_cost,
    correction_tree kinds source target other_cost -> cost <= other_cost.

(** These are exactly the four candidates considered by one non-empty Rust
    interval cell.  Recursive costs carry their own minimum proof, matching
    the increasing-interval-length table invariant. *)
Inductive recurrence_candidate (kinds : nat) : list nat -> nat -> Prop :=
  | candidate_delete : forall actual source source_cost,
      minimum_correction_cost kinds source source_cost ->
      recurrence_candidate kinds (actual :: source) (S source_cost)
  | candidate_insert_close : forall actual source kind source_cost,
      kind < kinds ->
      minimum_correction_cost kinds source source_cost ->
      recurrence_candidate kinds (actual :: source)
        (replacement_cost actual kind + 1 + source_cost)
  | candidate_pair :
      forall actual_open source_inner actual_close source_suffix
             kind inner_cost suffix_cost,
      kind < kinds ->
      minimum_correction_cost kinds source_inner inner_cost ->
      minimum_correction_cost kinds source_suffix suffix_cost ->
      recurrence_candidate kinds
        (actual_open :: source_inner ++ actual_close :: source_suffix)
        (replacement_cost actual_open kind + inner_cost +
         replacement_cost actual_close (kinds + kind) + suffix_cost)
  | candidate_insert_open :
      forall source_inner actual_close source_suffix
             kind inner_cost suffix_cost,
      kind < kinds ->
      minimum_correction_cost kinds source_inner inner_cost ->
      minimum_correction_cost kinds source_suffix suffix_cost ->
      recurrence_candidate kinds
        (source_inner ++ actual_close :: source_suffix)
        (1 + inner_cost + replacement_cost actual_close (kinds + kind) + suffix_cost).

Definition minimum_recurrence_candidate
    (kinds : nat) (source : list nat) (cost : nat) : Prop :=
  recurrence_candidate kinds source cost /\
  forall other_cost,
    recurrence_candidate kinds source other_cost -> cost <= other_cost.

Lemma recurrence_candidate_is_a_correction : forall kinds source cost,
  recurrence_candidate kinds source cost ->
  exists target, correction_tree kinds source target cost.
Proof.
  intros kinds source cost Hcandidate; destruct Hcandidate.
  - destruct H as [[target Htree] _].
    exists target; constructor; exact Htree.
  - destruct H0 as [[target Htree] _].
    exists (kind :: (kinds + kind) :: target).
    econstructor; eauto.
  - destruct H0 as [[target_inner Hinner] _].
    destruct H1 as [[target_suffix Hsuffix] _].
    exists (kind :: target_inner ++ (kinds + kind) :: target_suffix).
    econstructor; eauto.
  - destruct H0 as [[target_inner Hinner] _].
    destruct H1 as [[target_suffix Hsuffix] _].
    exists (kind :: target_inner ++ (kinds + kind) :: target_suffix).
    econstructor; eauto.
Qed.

(** Every strict source subinterval already has an attained minimum when the
    interval table is filled in increasing length. *)
Definition strict_subintervals_minimized
    (kinds : nat) (source : list nat) : Prop :=
  forall subsource,
    length subsource < length source ->
    exists cost, minimum_correction_cost kinds subsource cost.

Lemma every_correction_has_a_no_more_expensive_candidate :
  forall kinds source target cost,
    source <> [] ->
    strict_subintervals_minimized kinds source ->
    correction_tree kinds source target cost ->
    exists candidate_cost,
      recurrence_candidate kinds source candidate_cost /\
      candidate_cost <= cost.
Proof.
  intros kinds source target cost Hnonempty Hsubproblems Htree.
  destruct Htree.
  - contradiction.
  - destruct (Hsubproblems source) as [best Hbest]; [simpl; lia |].
    exists (S best); split.
    + constructor; exact Hbest.
    + destruct Hbest as [_ Hleast].
      specialize (Hleast target cost Htree); lia.
  - destruct (Hsubproblems source) as [best Hbest]; [simpl; lia |].
    exists (replacement_cost actual kind + 1 + best); split.
    + econstructor; eauto.
    + destruct Hbest as [_ Hleast].
      specialize (Hleast suffix suffix_cost Htree); lia.
  - destruct (Hsubproblems source_inner) as [best_inner Hbest_inner].
    + simpl; rewrite length_app; simpl; lia.
    + destruct (Hsubproblems source_suffix) as [best_suffix Hbest_suffix].
      * simpl; rewrite length_app; simpl; lia.
      * exists (replacement_cost actual_open kind + best_inner +
                  replacement_cost actual_close (kinds + kind) + best_suffix).
        split.
        -- econstructor; eauto.
        -- destruct Hbest_inner as [_ Hleast_inner].
           destruct Hbest_suffix as [_ Hleast_suffix].
           specialize (Hleast_inner target_inner inner_cost Htree1).
           specialize (Hleast_suffix target_suffix suffix_cost Htree2).
           lia.
  - destruct (Hsubproblems source_inner) as [best_inner Hbest_inner].
    + simpl; rewrite length_app; simpl; lia.
    + destruct (Hsubproblems source_suffix) as [best_suffix Hbest_suffix].
      * simpl; rewrite length_app; simpl; lia.
      * exists (1 + best_inner +
                  replacement_cost actual_close (kinds + kind) + best_suffix).
        split.
        -- econstructor; eauto.
        -- destruct Hbest_inner as [_ Hleast_inner].
           destruct Hbest_suffix as [_ Hleast_suffix].
           specialize (Hleast_inner target_inner inner_cost Htree1).
           specialize (Hleast_suffix target_suffix suffix_cost Htree2).
           lia.
Qed.

(** Global optimality of the runtime recurrence.  Under the table invariant
    that strict subinterval cells already contain their exact minima, choosing
    the least of the four runtime candidate families is equivalent to being
    the exact distance to the entire typed Dyck language. *)
Theorem interval_recurrence_is_globally_exact :
  forall kinds source cost,
    source <> [] ->
    strict_subintervals_minimized kinds source ->
    (minimum_recurrence_candidate kinds source cost <->
     minimum_correction_cost kinds source cost).
Proof.
  intros kinds source cost Hnonempty Hsubproblems; split.
  - intros [Hcandidate Hleast_candidate].
    split.
    + apply recurrence_candidate_is_a_correction; exact Hcandidate.
    + intros target other_cost Hother.
      destruct (every_correction_has_a_no_more_expensive_candidate
        kinds source target other_cost Hnonempty Hsubproblems Hother)
        as [candidate_cost [Hcandidate_cost Hbound]].
      specialize (Hleast_candidate candidate_cost Hcandidate_cost); lia.
  - intros Hminimum.
    destruct Hminimum as [[target Htree] Hleast].
    destruct (every_correction_has_a_no_more_expensive_candidate
      kinds source target cost Hnonempty Hsubproblems Htree)
      as [candidate_cost [Hcandidate Hcandidate_le]].
    assert (Hcost_le : cost <= candidate_cost).
    { destruct (recurrence_candidate_is_a_correction
        kinds source candidate_cost Hcandidate) as [candidate_target Hcandidate_tree].
      apply (Hleast candidate_target candidate_cost Hcandidate_tree). }
    assert (Hequal : candidate_cost = cost) by lia.
    subst candidate_cost; split; [exact Hcandidate |].
    intros other_cost Hother_candidate.
    destruct (recurrence_candidate_is_a_correction
      kinds source other_cost Hother_candidate) as [other_target Hother_tree].
    apply (Hleast other_target other_cost Hother_tree).
Qed.

(** Every non-empty typed Dyck word exposes exactly the grammar shape
    enumerated by the interval split: one opening kind, a balanced interior,
    its same-kind closer, and a balanced suffix. *)
Theorem nonempty_dyck_first_pair_decomposition :
  forall kinds word,
    dyck kinds word ->
    word <> [] ->
    exists kind inner suffix,
      kind < kinds /\
      word = kind :: inner ++ (kinds + kind) :: suffix /\
      dyck kinds inner /\
      dyck kinds suffix.
Proof.
  intros kinds word Hdyck; induction Hdyck; intros Hnonempty.
  - contradiction.
  - destruct left as [|head tail].
    + simpl in *; apply IHHdyck2; assumption.
    + assert (Hleft : head :: tail <> []) by discriminate.
      destruct (IHHdyck1 Hleft)
        as [kind [inner [suffix [Hkind [Heq [Hinner Hsuffix]]]]]].
      exists kind, inner, (suffix ++ right).
      repeat split; try assumption.
      * rewrite Heq; simpl; f_equal; rewrite <- app_assoc; reflexivity.
      * apply dyck_concat; assumption.
  - exists kind, inner, [].
    repeat split; try assumption.
    constructor.
Qed.

Example cross_kind_pair_is_not_a_typed_wrap :
  forall kinds opening closing,
    opening < kinds ->
    closing < kinds ->
    opening <> closing ->
    kinds + opening <> kinds + closing.
Proof. intros; lia. Qed.
