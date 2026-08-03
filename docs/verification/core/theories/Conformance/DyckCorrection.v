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

(** Standard left-to-right unit-cost Levenshtein alignments. This relation is
    independent of the Dyck grammar and of the interval recurrence: it has the
    ordinary delete, insert, and consume-as (keep/substitute) columns. *)
Inductive levenshtein_alignment : list nat -> list nat -> nat -> Prop :=
  | alignment_empty : levenshtein_alignment [] [] 0
  | alignment_delete : forall actual source target cost,
      levenshtein_alignment source target cost ->
      levenshtein_alignment (actual :: source) target (S cost)
  | alignment_insert : forall expected source target cost,
      levenshtein_alignment source target cost ->
      levenshtein_alignment source (expected :: target) (S cost)
  | alignment_consume : forall actual expected source target cost,
      levenshtein_alignment source target cost ->
      levenshtein_alignment
        (actual :: source) (expected :: target)
        (replacement_cost actual expected + cost).

Lemma alignment_to_empty_cost : forall source cost,
  levenshtein_alignment source [] cost -> cost = length source.
Proof.
  intros source cost Halign.
  remember [] as target eqn:Htarget.
  induction Halign; inversion Htarget; subst; simpl; auto.
Qed.

Lemma alignment_append : forall
    left_source left_target left_cost
    right_source right_target right_cost,
  levenshtein_alignment left_source left_target left_cost ->
  levenshtein_alignment right_source right_target right_cost ->
  levenshtein_alignment
    (left_source ++ right_source)
    (left_target ++ right_target)
    (left_cost + right_cost).
Proof.
  intros left_source left_target left_cost right_source right_target right_cost
    Hleft Hright.
  induction Hleft; simpl.
  - exact Hright.
  - replace (S cost + right_cost) with (S (cost + right_cost)) by lia.
    constructor; exact IHHleft.
  - replace (S cost + right_cost) with (S (cost + right_cost)) by lia.
    constructor; exact IHHleft.
  - replace (replacement_cost actual expected + cost + right_cost)
      with (replacement_cost actual expected + (cost + right_cost)) by lia.
    constructor; exact IHHleft.
Qed.

(** Any standard alignment can be cut at a target boundary. Deletions at the
    boundary are assigned to the left segment, which is sufficient for the
    normalization proof below. *)
Lemma alignment_target_append_split : forall source left_target right_target cost,
  levenshtein_alignment source (left_target ++ right_target) cost ->
  exists left_source right_source left_cost right_cost,
    source = left_source ++ right_source /\
    cost = left_cost + right_cost /\
    levenshtein_alignment left_source left_target left_cost /\
    levenshtein_alignment right_source right_target right_cost.
Proof.
  intros source left_target right_target cost Halign.
  remember (left_target ++ right_target) as target eqn:Htarget.
  revert left_target right_target Htarget.
  induction Halign; intros left_target right_target Htarget.
  - symmetry in Htarget; apply app_eq_nil in Htarget.
    destruct Htarget as [-> ->].
    exists [], [], 0, 0; repeat split; constructor.
  - destruct (IHHalign left_target right_target Htarget)
      as [left_source [right_source [left_cost [right_cost
          [Hsource [Hcost [Hleft Hright]]]]]]].
    exists (actual :: left_source), right_source, (S left_cost), right_cost.
    split.
    + rewrite Hsource; reflexivity.
    + split.
      * rewrite Hcost; lia.
      * split.
        -- constructor; exact Hleft.
        -- exact Hright.
  - destruct left_target as [|left_head left_tail].
    + simpl in Htarget; subst right_target.
      exists [], source, 0, (S cost); simpl.
      repeat split.
      * constructor.
      * constructor; exact Halign.
    + simpl in Htarget; inversion Htarget; subst left_head target.
      destruct (IHHalign left_tail right_target eq_refl)
        as [left_source [right_source [left_cost [right_cost
            [Hsource [Hcost [Hleft Hright]]]]]]].
      exists left_source, right_source, (S left_cost), right_cost.
      split; [exact Hsource |].
      split.
      * rewrite Hcost; lia.
      * split.
        -- constructor; exact Hleft.
        -- exact Hright.
  - destruct left_target as [|left_head left_tail].
    + simpl in Htarget; subst right_target.
      exists [], (actual :: source), 0,
        (replacement_cost actual expected + cost).
      simpl; repeat split.
      * constructor.
      * constructor; exact Halign.
    + simpl in Htarget; inversion Htarget; subst left_head target.
      destruct (IHHalign left_tail right_target eq_refl)
        as [left_source [right_source [left_cost [right_cost
            [Hsource [Hcost [Hleft Hright]]]]]]].
      exists (actual :: left_source), right_source,
        (replacement_cost actual expected + left_cost), right_cost.
      split.
      * rewrite Hsource; reflexivity.
      * split.
        -- rewrite Hcost; lia.
        -- split.
           ++ constructor; exact Hleft.
           ++ exact Hright.
Qed.

(** An alignment to one target token contains either one insertion and only
    deletions, or one consumed source token and deletions around it. *)
Lemma alignment_to_singleton_shape : forall source expected cost,
  levenshtein_alignment source [expected] cost ->
  cost = S (length source) \/
  exists prefix actual suffix,
    source = prefix ++ actual :: suffix /\
    cost = length prefix + replacement_cost actual expected + length suffix.
Proof.
  intros source expected cost Halign.
  remember [expected] as target eqn:Htarget.
  induction Halign as
      [|deleted source target cost Halign IH
       |inserted source target cost Halign IH
       |actual consumed source target cost Halign IH];
    inversion Htarget; subst.
  - specialize (IH eq_refl).
    destruct IH as [Hinserted | Hconsumed].
    + left; simpl; lia.
    + right.
      destruct Hconsumed as [prefix [chosen [suffix [Hsource Hcost]]]].
      exists (deleted :: prefix), chosen, suffix; subst; simpl; split; [reflexivity | lia].
  - left.
    pose proof (alignment_to_empty_cost source cost Halign); simpl; lia.
  - right.
    pose proof (alignment_to_empty_cost source cost Halign) as Hempty.
    exists [], actual, source; simpl; split; [reflexivity | lia].
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

Lemma alignment_delete_all : forall source,
  levenshtein_alignment source [] (length source).
Proof.
  intros source; induction source as [|actual source IH]; simpl.
  - constructor.
  - constructor; exact IH.
Qed.

(** Every runtime backpointer tree is an ordinary unit-cost Levenshtein
    alignment to its reconstructed target. This is the soundness direction of
    the refinement to the independent edit semantics. *)
Theorem correction_tree_is_standard_alignment : forall kinds source target cost,
  correction_tree kinds source target cost ->
  levenshtein_alignment source target cost.
Proof.
  intros kinds source target cost Htree; induction Htree.
  - constructor.
  - constructor; exact IHHtree.
  - replace (replacement_cost actual kind + 1 + suffix_cost)
      with (replacement_cost actual kind + S suffix_cost) by lia.
    constructor; constructor; exact IHHtree.
  - pose proof (alignment_consume
      actual_close (kinds + kind) source_suffix target_suffix suffix_cost
      IHHtree2) as Hclose.
    pose proof (alignment_append
      source_inner target_inner inner_cost
      (actual_close :: source_suffix) ((kinds + kind) :: target_suffix)
      (replacement_cost actual_close (kinds + kind) + suffix_cost)
      IHHtree1 Hclose) as Hrest.
    replace (replacement_cost actual_open kind + inner_cost +
             replacement_cost actual_close (kinds + kind) + suffix_cost)
      with (replacement_cost actual_open kind +
            (inner_cost +
             (replacement_cost actual_close (kinds + kind) + suffix_cost))) by lia.
    constructor; exact Hrest.
  - pose proof (alignment_consume
      actual_close (kinds + kind) source_suffix target_suffix suffix_cost
      IHHtree2) as Hclose.
    pose proof (alignment_append
      source_inner target_inner inner_cost
      (actual_close :: source_suffix) ((kinds + kind) :: target_suffix)
      (replacement_cost actual_close (kinds + kind) + suffix_cost)
      IHHtree1 Hclose) as Hrest.
    replace (1 + inner_cost + replacement_cost actual_close (kinds + kind) + suffix_cost)
      with (S (inner_cost +
               (replacement_cost actual_close (kinds + kind) + suffix_cost))) by lia.
    constructor; exact Hrest.
Qed.

Lemma correction_delete_prefix : forall kinds prefix source target cost,
  correction_tree kinds source target cost ->
  correction_tree kinds (prefix ++ source) target (length prefix + cost).
Proof.
  intros kinds prefix; induction prefix as [|actual prefix IH];
    intros source target cost Htree; simpl.
  - exact Htree.
  - constructor; apply IH; exact Htree.
Qed.

(** Delete padding around two independently aligned regions. This packages the
    list reassociation needed when the target's first typed pair is removed or
    its opener/closer is assigned to a consumed source token. *)
Lemma alignment_deleted_padding_concat : forall
    prefix left_source middle right_source
    left_target right_target left_cost right_cost,
  levenshtein_alignment left_source left_target left_cost ->
  levenshtein_alignment right_source right_target right_cost ->
  levenshtein_alignment
    (prefix ++ left_source ++ middle ++ right_source)
    (left_target ++ right_target)
    (length prefix + left_cost + length middle + right_cost).
Proof.
  intros prefix left_source middle right_source left_target right_target
    left_cost right_cost Hleft Hright.
  pose proof (alignment_delete_all prefix) as Hprefix.
  pose proof (alignment_delete_all middle) as Hmiddle.
  pose proof (alignment_append
    prefix [] (length prefix)
    left_source left_target left_cost Hprefix Hleft) as Hprefix_left.
  pose proof (alignment_append
    (prefix ++ left_source) left_target (length prefix + left_cost)
    middle [] (length middle) Hprefix_left Hmiddle) as Hthrough_middle.
  pose proof (alignment_append
    ((prefix ++ left_source) ++ middle) (left_target ++ [])
    (length prefix + left_cost + length middle)
    right_source right_target right_cost Hthrough_middle Hright) as Hall.
  replace (prefix ++ left_source ++ middle ++ right_source)
    with (((prefix ++ left_source) ++ middle) ++ right_source)
    by (repeat rewrite app_assoc; reflexivity).
  replace (left_target ++ right_target)
    with ((left_target ++ []) ++ right_target) by (rewrite app_nil_r; reflexivity).
  exact Hall.
Qed.

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

Lemma alignment_first_pair_split : forall
    source kind inner close suffix cost,
  levenshtein_alignment
    source (kind :: inner ++ close :: suffix) cost ->
  exists open_source inner_source close_source suffix_source,
  exists open_cost inner_cost close_cost suffix_cost,
    source = open_source ++ inner_source ++ close_source ++ suffix_source /\
    cost = open_cost + inner_cost + close_cost + suffix_cost /\
    levenshtein_alignment open_source [kind] open_cost /\
    levenshtein_alignment inner_source inner inner_cost /\
    levenshtein_alignment close_source [close] close_cost /\
    levenshtein_alignment suffix_source suffix suffix_cost.
Proof.
  intros source kind inner close suffix cost Halign.
  change (levenshtein_alignment
    source ([kind] ++ (inner ++ ([close] ++ suffix))) cost) in Halign.
  destruct (alignment_target_append_split
      source [kind] (inner ++ ([close] ++ suffix)) cost Halign)
    as [open_source [rest_source [open_cost [rest_cost
        [Hsource [Hcost [Hopen Hrest]]]]]]].
  destruct (alignment_target_append_split
      rest_source inner ([close] ++ suffix) rest_cost Hrest)
    as [inner_source [close_suffix_source [inner_cost [close_suffix_cost
        [Hrest_source [Hrest_cost [Hinner Hclose_suffix]]]]]]].
  destruct (alignment_target_append_split
      close_suffix_source [close] suffix close_suffix_cost Hclose_suffix)
    as [close_source [suffix_source [close_cost [suffix_cost
        [Hclose_source [Hclose_cost [Hclose Hsuffix]]]]]]].
  exists open_source, inner_source, close_source, suffix_source.
  exists open_cost, inner_cost, close_cost, suffix_cost.
  subst; repeat split; try assumption; try lia.
Qed.

(** Normalization/completeness against independent edit semantics. The measure
    decreases when a consumed delimiter is removed from the source, when a
    typed pair is removed from the target, or both. The four cases below are
    exactly: consume/consume, insert/consume, consume/insert, insert/insert for
    the target's first typed pair. Extra source tokens in singleton alignments
    are ordinary deletions and are folded into the recursive interior/suffix. *)
Lemma standard_alignment_normalizes_by_measure : forall
    measure source target cost kinds,
  length source + length target <= measure ->
  levenshtein_alignment source target cost ->
  dyck kinds target ->
  exists normalized_target normalized_cost,
    correction_tree kinds source normalized_target normalized_cost /\
    normalized_cost <= cost.
Proof.
  induction measure as [measure IH] using lt_wf_ind.
  intros source target cost kinds Hmeasure Halign Hdyck.
  destruct target as [|target_head target_tail].
  - pose proof (alignment_to_empty_cost source cost Halign) as Hcost.
    exists [], (length source); split.
    + apply every_source_has_a_correction.
    + lia.
  - assert (Hnonempty : target_head :: target_tail <> []) by discriminate.
    destruct (nonempty_dyck_first_pair_decomposition
      kinds (target_head :: target_tail) Hdyck Hnonempty)
      as [kind [inner [suffix [Hkind [Htarget [Hinner_dyck Hsuffix_dyck]]]]]].
    rewrite Htarget in Halign, Hmeasure.
    destruct (alignment_first_pair_split
      source kind inner (kinds + kind) suffix cost Halign)
      as [open_source [inner_source [close_source [suffix_source
          [open_cost [inner_cost [close_cost [suffix_cost
          [Hsource [Hcost [Hopen [Hinner [Hclose Hsuffix]]]]]]]]]]]]].
    destruct (alignment_to_singleton_shape
      open_source kind open_cost Hopen) as
      [Hopen_inserted |
       [open_prefix [actual_open [open_tail [Hopen_source Hopen_cost]]]]].
    + destruct (alignment_to_singleton_shape
        close_source (kinds + kind) close_cost Hclose) as
        [Hclose_inserted |
         [close_prefix [actual_close [close_tail
           [Hclose_source Hclose_cost]]]]].
      * pose proof (alignment_deleted_padding_concat
          open_source inner_source close_source suffix_source
          inner suffix inner_cost suffix_cost Hinner Hsuffix) as Hreduced.
        assert (Hreduced_measure :
          length (open_source ++ inner_source ++ close_source ++ suffix_source) +
          length (inner ++ suffix) < measure).
        { rewrite <- Hsource.
          assert (Htarget_length :
            length (kind :: inner ++ (kinds + kind) :: suffix) =
            length (inner ++ suffix) + 2).
          { simpl; rewrite !length_app; simpl; lia. }
          rewrite Htarget_length in Hmeasure; lia. }
        assert (Hreduced_dyck : dyck kinds (inner ++ suffix)).
        { apply dyck_concat; assumption. }
        destruct (IH _ Hreduced_measure
          (open_source ++ inner_source ++ close_source ++ suffix_source)
          (inner ++ suffix)
          (length open_source + inner_cost + length close_source + suffix_cost)
          kinds (le_n _) Hreduced Hreduced_dyck)
          as [normalized [normalized_cost [Htree Hbound]]].
        exists normalized, normalized_cost.
        split.
        -- rewrite Hsource; exact Htree.
        -- lia.
      * subst close_source; subst close_cost.
        pose proof (alignment_deleted_padding_concat
          open_source inner_source close_prefix []
          inner [] inner_cost 0 Hinner alignment_empty) as Hinner_padded.
        repeat rewrite app_nil_r in Hinner_padded; simpl in Hinner_padded.
        replace (length open_source + inner_cost + length close_prefix + 0)
          with (length open_source + inner_cost + length close_prefix)
          in Hinner_padded by lia.
        pose proof (alignment_deleted_padding_concat
          close_tail suffix_source [] []
          suffix [] suffix_cost 0 Hsuffix alignment_empty) as Hsuffix_padded.
        repeat rewrite app_nil_r in Hsuffix_padded; simpl in Hsuffix_padded.
        replace (length close_tail + suffix_cost + 0 + 0)
          with (length close_tail + suffix_cost) in Hsuffix_padded by lia.
        assert (Hinner_measure :
          length (open_source ++ inner_source ++ close_prefix) + length inner < measure).
        { rewrite Hsource in Hmeasure.
          simpl in Hmeasure; rewrite !length_app in Hmeasure; simpl in Hmeasure.
          rewrite !length_app; lia. }
        assert (Hsuffix_measure :
          length (close_tail ++ suffix_source) + length suffix < measure).
        { rewrite Hsource in Hmeasure.
          simpl in Hmeasure; rewrite !length_app in Hmeasure; simpl in Hmeasure.
          rewrite !length_app; lia. }
        destruct (IH _ Hinner_measure
          (open_source ++ inner_source ++ close_prefix) inner
          (length open_source + inner_cost + length close_prefix)
          kinds (le_n _) Hinner_padded Hinner_dyck)
          as [normalized_inner [normalized_inner_cost
              [Hinner_tree Hinner_bound]]].
        destruct (IH _ Hsuffix_measure
          (close_tail ++ suffix_source) suffix
          (length close_tail + suffix_cost)
          kinds (le_n _) Hsuffix_padded Hsuffix_dyck)
          as [normalized_suffix [normalized_suffix_cost
              [Hsuffix_tree Hsuffix_bound]]].
        exists
          (kind :: normalized_inner ++ (kinds + kind) :: normalized_suffix),
          (1 + normalized_inner_cost +
           replacement_cost actual_close (kinds + kind) + normalized_suffix_cost).
        split.
        -- rewrite Hsource.
           replace
             (open_source ++ inner_source ++
              (close_prefix ++ actual_close :: close_tail) ++ suffix_source)
             with
             ((open_source ++ inner_source ++ close_prefix) ++
              actual_close :: (close_tail ++ suffix_source))
             by (repeat rewrite <- app_assoc; reflexivity).
           econstructor; eauto.
        -- lia.
    + subst open_source; subst open_cost.
      destruct (alignment_to_singleton_shape
        close_source (kinds + kind) close_cost Hclose) as
        [Hclose_inserted |
         [close_prefix [actual_close [close_tail
           [Hclose_source Hclose_cost]]]]].
      * subst close_cost.
        pose proof (alignment_deleted_padding_concat
          open_tail inner_source close_source suffix_source
          inner suffix inner_cost suffix_cost Hinner Hsuffix) as Hreduced.
        assert (Hreduced_measure :
          length (open_tail ++ inner_source ++ close_source ++ suffix_source) +
          length (inner ++ suffix) < measure).
        { rewrite Hsource in Hmeasure.
          simpl in Hmeasure; rewrite !length_app in Hmeasure; simpl in Hmeasure.
          rewrite !length_app; lia. }
        assert (Hreduced_dyck : dyck kinds (inner ++ suffix)).
        { apply dyck_concat; assumption. }
        destruct (IH _ Hreduced_measure
          (open_tail ++ inner_source ++ close_source ++ suffix_source)
          (inner ++ suffix)
          (length open_tail + inner_cost + length close_source + suffix_cost)
          kinds (le_n _) Hreduced Hreduced_dyck)
          as [normalized [normalized_cost [Htree Hbound]]].
        exists (kind :: (kinds + kind) :: normalized),
          (length open_prefix +
           (replacement_cost actual_open kind + 1 + normalized_cost)).
        split.
        -- rewrite Hsource.
           replace
             ((open_prefix ++ actual_open :: open_tail) ++
              inner_source ++ close_source ++ suffix_source)
             with
             (open_prefix ++ actual_open ::
              (open_tail ++ inner_source ++ close_source ++ suffix_source))
             by (repeat rewrite <- app_assoc; reflexivity).
           eapply correction_delete_prefix with (prefix := open_prefix).
           econstructor; eauto.
        -- lia.
      * subst close_source; subst close_cost.
        pose proof (alignment_deleted_padding_concat
          open_tail inner_source close_prefix []
          inner [] inner_cost 0 Hinner alignment_empty) as Hinner_padded.
        repeat rewrite app_nil_r in Hinner_padded; simpl in Hinner_padded.
        replace (length open_tail + inner_cost + length close_prefix + 0)
          with (length open_tail + inner_cost + length close_prefix)
          in Hinner_padded by lia.
        pose proof (alignment_deleted_padding_concat
          close_tail suffix_source [] []
          suffix [] suffix_cost 0 Hsuffix alignment_empty) as Hsuffix_padded.
        repeat rewrite app_nil_r in Hsuffix_padded; simpl in Hsuffix_padded.
        replace (length close_tail + suffix_cost + 0 + 0)
          with (length close_tail + suffix_cost) in Hsuffix_padded by lia.
        assert (Hinner_measure :
          length (open_tail ++ inner_source ++ close_prefix) + length inner < measure).
        { rewrite Hsource in Hmeasure.
          simpl in Hmeasure; rewrite !length_app in Hmeasure; simpl in Hmeasure.
          rewrite !length_app; lia. }
        assert (Hsuffix_measure :
          length (close_tail ++ suffix_source) + length suffix < measure).
        { rewrite Hsource in Hmeasure.
          simpl in Hmeasure; rewrite !length_app in Hmeasure; simpl in Hmeasure.
          rewrite !length_app; lia. }
        destruct (IH _ Hinner_measure
          (open_tail ++ inner_source ++ close_prefix) inner
          (length open_tail + inner_cost + length close_prefix)
          kinds (le_n _) Hinner_padded Hinner_dyck)
          as [normalized_inner [normalized_inner_cost
              [Hinner_tree Hinner_bound]]].
        destruct (IH _ Hsuffix_measure
          (close_tail ++ suffix_source) suffix
          (length close_tail + suffix_cost)
          kinds (le_n _) Hsuffix_padded Hsuffix_dyck)
          as [normalized_suffix [normalized_suffix_cost
              [Hsuffix_tree Hsuffix_bound]]].
        exists
          (kind :: normalized_inner ++ (kinds + kind) :: normalized_suffix),
          (length open_prefix +
           (replacement_cost actual_open kind + normalized_inner_cost +
            replacement_cost actual_close (kinds + kind) + normalized_suffix_cost)).
        split.
        -- rewrite Hsource.
           replace
             ((open_prefix ++ actual_open :: open_tail) ++
              inner_source ++
              (close_prefix ++ actual_close :: close_tail) ++ suffix_source)
             with
             (open_prefix ++ actual_open ::
              (open_tail ++ inner_source ++ close_prefix) ++
              actual_close :: (close_tail ++ suffix_source))
             by (repeat rewrite <- app_assoc; reflexivity).
           eapply correction_delete_prefix with (prefix := open_prefix).
           econstructor; eauto.
        -- lia.
Qed.

Theorem standard_alignment_normalizes_to_correction_tree : forall
    kinds source target cost,
  levenshtein_alignment source target cost ->
  dyck kinds target ->
  exists normalized_target normalized_cost,
    correction_tree kinds source normalized_target normalized_cost /\
    normalized_cost <= cost.
Proof.
  intros kinds source target cost Halign Hdyck.
  eapply standard_alignment_normalizes_by_measure
    with (measure := length source + length target); eauto.
Qed.

(** The independently specified distance to the typed Dyck language.  Neither
    this predicate nor [levenshtein_alignment] mentions interval cells,
    recurrence candidates, or correction-tree constructors. *)
Definition minimum_dyck_levenshtein_cost
    (kinds : nat) (source : list nat) (cost : nat) : Prop :=
  (exists target,
      dyck kinds target /\ levenshtein_alignment source target cost) /\
  forall target other_cost,
    dyck kinds target ->
    levenshtein_alignment source target other_cost ->
    cost <= other_cost.

(** Soundness plus normalization makes the algorithm-shaped correction-tree
    minimum extensionally equal to the independent Levenshtein minimum. *)
Theorem correction_minimum_equals_dyck_levenshtein_minimum : forall
    kinds source cost,
  minimum_correction_cost kinds source cost <->
  minimum_dyck_levenshtein_cost kinds source cost.
Proof.
  intros kinds source cost; split.
  - intros [[target Htree] Hleast].
    split.
    + exists target; split.
      * eapply correction_target_is_dyck; eauto.
      * eapply correction_tree_is_standard_alignment; eauto.
    + intros other_target other_cost Hdyck Halign.
      destruct (standard_alignment_normalizes_to_correction_tree
        kinds source other_target other_cost Halign Hdyck)
        as [normalized_target [normalized_cost [Hnormalized Hbound]]].
      specialize (Hleast normalized_target normalized_cost Hnormalized); lia.
  - intros [[target [Hdyck Halign]] Hleast].
    destruct (standard_alignment_normalizes_to_correction_tree
      kinds source target cost Halign Hdyck)
      as [normalized_target [normalized_cost [Hnormalized Hbound]]].
    assert (Hcost_le : cost <= normalized_cost).
    { apply (Hleast normalized_target normalized_cost).
      - eapply correction_target_is_dyck; eauto.
      - eapply correction_tree_is_standard_alignment; eauto. }
    assert (Hequal : normalized_cost = cost) by lia.
    subst normalized_cost; split.
    + exists normalized_target; exact Hnormalized.
    + intros other_target other_cost Hother.
      apply (Hleast other_target other_cost).
      * eapply correction_target_is_dyck; eauto.
      * eapply correction_tree_is_standard_alignment; eauto.
Qed.

(** The increasing-interval table invariant therefore proves exactness against
    ordinary Levenshtein edits, not merely against a recurrence-shaped
    specification.  Every recursive premise is a strict source subinterval,
    precisely matching the runtime fill order. *)
Theorem interval_recurrence_is_exact_standard_dyck_distance : forall
    kinds source cost,
  source <> [] ->
  strict_subintervals_minimized kinds source ->
  (minimum_recurrence_candidate kinds source cost <->
   minimum_dyck_levenshtein_cost kinds source cost).
Proof.
  intros kinds source cost Hnonempty Hsubproblems.
  split.
  - intro Hcandidate.
    apply correction_minimum_equals_dyck_levenshtein_minimum.
    apply (proj1 (interval_recurrence_is_globally_exact
      kinds source cost Hnonempty Hsubproblems)); exact Hcandidate.
  - intro Hminimum.
    apply (proj2 (interval_recurrence_is_globally_exact
      kinds source cost Hnonempty Hsubproblems)).
    apply correction_minimum_equals_dyck_levenshtein_minimum; exact Hminimum.
Qed.

Example cross_kind_pair_is_not_a_typed_wrap :
  forall kinds opening closing,
    opening < kinds ->
    closing < kinds ->
    opening <> closing ->
    kinds + opening <> kinds + closing.
Proof. intros; lia. Qed.
