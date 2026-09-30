(** * Best-k selection and a max-heap abstraction boundary

    Ranks use natural costs and natural tie keys. Each entry names one
    original occurrence. A source instance must bind the comparator to its
    actual arithmetic and tie rule and show the concrete heap mutators meet
    the abstract transition. Nothing in this proof adds a production scan. *)

From Stdlib Require Import Arith Lia List Permutation Sorting.Sorted.
Import ListNotations.
Set Implicit Arguments.

Record ranked_entry := {
  entry_original : nat;
  entry_cost : nat;
  entry_tie : nat
}.

Definition rank_lt (left right : ranked_entry) : Prop :=
  entry_cost left < entry_cost right \/
  entry_cost left = entry_cost right /\ entry_tie left < entry_tie right.

Definition rank_le (left right : ranked_entry) : Prop :=
  entry_cost left < entry_cost right \/
  entry_cost left = entry_cost right /\ entry_tie left <= entry_tie right.

Lemma rank_le_refl : forall entry, rank_le entry entry.
Proof. intros; unfold rank_le; right; lia. Qed.

Lemma rank_le_total : forall left right,
  rank_le left right \/ rank_le right left.
Proof.
  intros left right; unfold rank_le.
  destruct (Nat.lt_trichotomy (entry_cost left) (entry_cost right))
    as [Hless | [Hequal | Hgreater]].
  - left; now left.
  - destruct (le_dec (entry_tie left) (entry_tie right)).
    + left; right; lia.
    + right; right; lia.
  - right; now left.
Qed.

Lemma rank_le_trans : forall first middle last,
  rank_le first middle -> rank_le middle last -> rank_le first last.
Proof.
  intros first middle last Hfirst Hlast.
  unfold rank_le in *.
  destruct Hfirst as [Hfirst | [Hfirst Hfirst_tie]];
    destruct Hlast as [Hlast | [Hlast Hlast_tie]];
    [left | left | left | right]; lia.
Qed.

Lemma rank_lt_implies_le : forall left right,
  rank_lt left right -> rank_le left right.
Proof. intros left right [Hcost | [Hcost Htie]]; unfold rank_le; lia. Qed.

Lemma rank_not_lt_implies_reverse_le : forall left right,
  ~ rank_lt left right -> rank_le right left.
Proof.
  intros left right Hnot.
  unfold rank_lt in Hnot; unfold rank_le.
  destruct (Nat.lt_trichotomy (entry_cost right) (entry_cost left))
    as [Hless | [Hequal | Hgreater]].
  - now left.
  - right; lia.
  - exfalso; apply Hnot; now left.
Qed.

Definition rank_le_dec (left right : ranked_entry) :
    {rank_le left right} + {~ rank_le left right}.
Proof.
  unfold rank_le.
  destruct (lt_dec (entry_cost left) (entry_cost right)) as [Hlt | Hnlt].
  - left; now left.
  - destruct (Nat.eq_dec (entry_cost left) (entry_cost right))
      as [Heq | Hneq].
    + destruct (le_dec (entry_tie left) (entry_tie right))
        as [Hle | Hnle].
      * left; now right.
      * right; intros [Hcost | [_ Htie]]; lia.
    + right; intros [Hcost | [Hcost _]]; lia.
Defined.

Definition rank_lt_dec (left right : ranked_entry) :
    {rank_lt left right} + {~ rank_lt left right}.
Proof.
  unfold rank_lt.
  destruct (lt_dec (entry_cost left) (entry_cost right)) as [Hlt | Hnlt].
  - left; now left.
  - destruct (Nat.eq_dec (entry_cost left) (entry_cost right))
      as [Heq | Hneq].
    + destruct (lt_dec (entry_tie left) (entry_tie right))
        as [Hlt | Hnlt_tie].
      * left; now right.
      * right; intros [Hcost | [_ Htie]]; lia.
    + right; intros [Hcost | [Hcost _]]; lia.
Defined.

Fixpoint insert_ranked (entry : ranked_entry)
    (selected : list ranked_entry) : list ranked_entry :=
  match selected with
  | [] => [entry]
  | first :: rest =>
      if rank_le_dec entry first then entry :: first :: rest
      else first :: insert_ranked entry rest
  end.

Lemma insert_ranked_permutation : forall entry selected,
  Permutation (insert_ranked entry selected) (entry :: selected).
Proof.
  intros entry selected; induction selected as [|first rest IH]; simpl.
  - reflexivity.
  - destruct (rank_le_dec entry first); [reflexivity |].
    eapply Permutation_trans.
    + apply perm_skip; exact IH.
    + apply perm_swap.
Qed.

Lemma insert_ranked_forall : forall (P : ranked_entry -> Prop)
    entry selected,
  P entry -> Forall P selected -> Forall P (insert_ranked entry selected).
Proof.
  intros P entry selected Hentry Hselected.
  induction Hselected as [|first rest Hfirst Hrest IH]; simpl.
  - now constructor.
  - destruct (rank_le_dec entry first); constructor; auto.
Qed.

Lemma insert_ranked_preserves_sorted : forall entry selected,
  StronglySorted rank_le selected ->
  StronglySorted rank_le (insert_ranked entry selected).
Proof.
  intros entry selected Hsorted.
  induction Hsorted as [|first rest Htail IH Hfor]; simpl.
  - constructor; constructor.
  - destruct (rank_le_dec entry first) as [Hbefore | Hafter].
    + constructor.
      * now constructor.
      * constructor; [exact Hbefore |].
        eapply Forall_impl; [|exact Hfor].
        intros other Hfirst_other.
        eapply rank_le_trans; eauto.
    + constructor.
      * exact IH.
      * apply insert_ranked_forall.
        -- destruct (rank_le_total first entry) as [Hle | Hle];
             [exact Hle | contradiction].
        -- exact Hfor.
Qed.

Lemma insert_ranked_length : forall entry selected,
  length (insert_ranked entry selected) = S (length selected).
Proof.
  intros entry selected.
  pose proof (Permutation_length (insert_ranked_permutation entry selected))
    as Hlength.
  simpl in Hlength; exact Hlength.
Qed.

Lemma insert_ranked_members : forall entry selected member,
  In member (insert_ranked entry selected) <->
  member = entry \/ In member selected.
Proof.
  intros entry selected member.
  split; intro Hin.
  - assert (In member (entry :: selected)) as Hsource.
    { eapply Permutation_in; [apply insert_ranked_permutation | exact Hin]. }
    simpl in Hsource.
    destruct Hsource as [Hequal | Hmember].
    + left; now symmetry.
    + now right.
  - eapply Permutation_in.
    + apply Permutation_sym, insert_ranked_permutation.
    + simpl; destruct Hin as [Hequal | Hmember].
      * left; now symmetry.
      * now right.
Qed.

Fixpoint split_last (entries : list ranked_entry)
    : option (list ranked_entry * ranked_entry) :=
  match entries with
  | [] => None
  | [last] => Some ([], last)
  | first :: rest =>
      match split_last rest with
      | Some (prefix, last) => Some (first :: prefix, last)
      | None => None
      end
  end.

Lemma split_last_sound : forall entries prefix worst,
  split_last entries = Some (prefix, worst) ->
  entries = prefix ++ [worst].
Proof.
  induction entries as [|first [|second rest] IH];
    intros prefix worst Hsplit.
  - discriminate.
  - simpl in Hsplit; inversion Hsplit; reflexivity.
  - change
      (match split_last (second :: rest) with
       | Some (middle, last) => Some (first :: middle, last)
       | None => None
       end = Some (prefix, worst)) in Hsplit.
    destruct (split_last (second :: rest)) as [[middle last] |]
      eqn:Hmiddle; [|discriminate].
    inversion Hsplit; subst.
    specialize (IH middle worst eq_refl); simpl.
    now rewrite IH.
Qed.

Lemma split_last_complete : forall entries,
  entries <> [] ->
  exists prefix worst, split_last entries = Some (prefix, worst).
Proof.
  induction entries as [|first [|second rest] IH];
    intro Hnonempty.
  - contradiction.
  - exists [], first; reflexivity.
  - assert (Htail_nonempty : second :: rest <> []) by discriminate.
    destruct (IH Htail_nonempty) as
      [prefix [worst Hsplit]].
    exists (first :: prefix), worst.
    change
      (match split_last (second :: rest) with
       | Some (middle, last) => Some (first :: middle, last)
       | None => None
       end = Some (first :: prefix, worst)).
    now rewrite Hsplit.
Qed.

Lemma sorted_prefix_and_worst : forall prefix worst,
  StronglySorted rank_le (prefix ++ [worst]) ->
  StronglySorted rank_le prefix /\
  forall entry, In entry prefix -> rank_le entry worst.
Proof.
  induction prefix as [|first rest IH]; intros worst Hsorted; simpl in *.
  - split; [constructor | intros entry Hin; contradiction].
  - inversion Hsorted as [|? ? Htail Hfor]; subst.
    destruct (IH worst Htail) as [Hrest Hmax].
    split.
    + constructor; [exact Hrest |].
      apply Forall_forall; intros entry Hin.
      apply Forall_forall with (x := entry) in Hfor.
      * exact Hfor.
      * apply in_or_app; now left.
    + intros entry [Hequal | Hin].
      * subst entry.
        apply Forall_forall with (x := worst) in Hfor.
        -- exact Hfor.
        -- apply in_or_app; right; now left.
      * now apply Hmax.
Qed.

Record best_k_certificate (capacity : nat) (verified selected rejected
    : list ranked_entry) : Prop := {
  best_cover : Permutation verified (selected ++ rejected);
  best_sorted : StronglySorted rank_le selected;
  best_capacity : length selected <= capacity;
  best_fill_before_reject : length selected < capacity -> rejected = [];
  best_selected_precedes_rejected :
    forall retained discarded,
      In retained selected -> In discarded rejected ->
      rank_le retained discarded;
  best_distinct_originals : NoDup (map entry_original verified);
  best_distinct_ties : NoDup (map entry_tie verified)
}.

Lemma empty_best_k_certificate : forall capacity,
  best_k_certificate capacity [] [] [].
Proof.
  intro capacity; constructor; simpl; try constructor; try lia;
    intros; contradiction.
Qed.

Inductive best_k_step (capacity : nat) (entry : ranked_entry)
    : list ranked_entry -> list ranked_entry ->
      list ranked_entry -> list ranked_entry -> Prop :=
| BestZero : forall rejected,
    capacity = 0 ->
    best_k_step capacity entry [] rejected [] (entry :: rejected)
| BestFill : forall selected,
    length selected < capacity ->
    best_k_step capacity entry selected []
      (insert_ranked entry selected) []
| BestReplace : forall prefix worst rejected,
    length (prefix ++ [worst]) = capacity ->
    rank_lt entry worst ->
    best_k_step capacity entry (prefix ++ [worst]) rejected
      (insert_ranked entry prefix) (worst :: rejected)
| BestReject : forall prefix worst rejected,
    length (prefix ++ [worst]) = capacity ->
    ~ rank_lt entry worst ->
    best_k_step capacity entry (prefix ++ [worst]) rejected
      (prefix ++ [worst]) (entry :: rejected).

Lemma replace_best_cover : forall entry verified prefix worst rejected,
  Permutation verified ((prefix ++ [worst]) ++ rejected) ->
  Permutation (entry :: verified)
    (insert_ranked entry prefix ++ worst :: rejected).
Proof.
  intros entry verified prefix worst rejected Hcover.
  eapply Permutation_trans with
    (l' := entry :: ((prefix ++ [worst]) ++ rejected)).
  - apply perm_skip; exact Hcover.
  - rewrite <- (app_assoc prefix [worst] rejected).
    simpl.
    change (Permutation ((entry :: prefix) ++ worst :: rejected)
      (insert_ranked entry prefix ++ worst :: rejected)).
    apply Permutation_app; [|reflexivity].
    apply Permutation_sym, insert_ranked_permutation.
Qed.

Lemma reject_best_cover : forall (entry : ranked_entry)
    (verified selected rejected : list ranked_entry),
  Permutation verified (selected ++ rejected) ->
  Permutation (entry :: verified)
    (selected ++ entry :: rejected).
Proof.
  intros entry verified selected rejected Hcover.
  eapply Permutation_trans.
  - apply perm_skip; exact Hcover.
  - apply Permutation_middle.
Qed.

Theorem best_k_step_exists : forall capacity entry verified selected rejected,
  best_k_certificate capacity verified selected rejected ->
  exists next_selected next_rejected,
    best_k_step capacity entry selected rejected
      next_selected next_rejected.
Proof.
  intros capacity entry verified selected rejected Hbest.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Horder Hnodup Htie].
  destruct (Nat.eq_dec capacity 0) as [Hzero | Hpositive].
  - assert (selected = []) as Hempty.
    { destruct selected as [|first rest]; [reflexivity |].
      simpl in *; lia. }
    subst selected.
    exists [], (entry :: rejected).
    apply BestZero; exact Hzero.
  - destruct (lt_dec (length selected) capacity) as [Hunder | Hfull].
    + pose proof (Hfill Hunder) as Hrejected.
      subst rejected.
      exists (insert_ranked entry selected), [].
      now apply BestFill.
    + assert (Hlength : length selected = capacity).
      { lia. }
      assert (Hnonempty : selected <> []).
      { intro Hempty; subst selected; simpl in Hlength; lia. }
      destruct (split_last_complete Hnonempty)
        as [prefix [worst Hsplit]].
      assert (Hselected : selected = prefix ++ [worst]).
      { eapply split_last_sound; eauto. }
      subst selected.
      destruct (rank_lt_dec entry worst) as [Hbetter | Hnot_better].
      * exists (insert_ranked entry prefix), (worst :: rejected).
        now apply BestReplace.
      * exists (prefix ++ [worst]), (entry :: rejected).
        now apply BestReject.
Qed.

Lemma prepend_fresh_field_preserves_nodup :
  forall (field : ranked_entry -> nat) entry verified,
    ~ In (field entry) (map field verified) ->
    NoDup (map field verified) ->
    NoDup (map field (entry :: verified)).
Proof. intros; simpl; now constructor. Qed.

Lemma zero_capacity_preserves_best_k :
  forall entry verified rejected,
    best_k_certificate 0 verified [] rejected ->
    ~ In (entry_original entry) (map entry_original verified) ->
    ~ In (entry_tie entry) (map entry_tie verified) ->
    best_k_certificate 0 (entry :: verified) [] (entry :: rejected).
Proof.
  intros entry verified rejected Hbest Hfresh_original Hfresh_tie.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Horder Hnodup Htie].
  constructor.
  - eapply reject_best_cover; exact Hcover.
  - constructor.
  - simpl; lia.
  - simpl; lia.
  - intros retained discarded Hin; contradiction.
  - now apply prepend_fresh_field_preserves_nodup.
  - now apply prepend_fresh_field_preserves_nodup.
Qed.

Lemma underfull_admission_preserves_best_k :
  forall capacity entry verified selected,
    length selected < capacity ->
    best_k_certificate capacity verified selected [] ->
    ~ In (entry_original entry) (map entry_original verified) ->
    ~ In (entry_tie entry) (map entry_tie verified) ->
    best_k_certificate capacity (entry :: verified)
      (insert_ranked entry selected) [].
Proof.
  intros capacity entry verified selected Hunder Hbest
    Hfresh_original Hfresh_tie.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Horder Hnodup Htie].
  constructor.
  - rewrite app_nil_r in Hcover |- *.
    eapply Permutation_trans.
    + apply perm_skip; exact Hcover.
    + apply Permutation_sym, insert_ranked_permutation.
  - now apply insert_ranked_preserves_sorted.
  - rewrite insert_ranked_length; lia.
  - intros; reflexivity.
  - intros retained discarded _ Hin; contradiction.
  - now apply prepend_fresh_field_preserves_nodup.
  - now apply prepend_fresh_field_preserves_nodup.
Qed.

Lemma full_replacement_preserves_best_k :
  forall capacity entry verified prefix worst rejected,
    length (prefix ++ [worst]) = capacity ->
    rank_lt entry worst ->
    best_k_certificate capacity verified (prefix ++ [worst]) rejected ->
    ~ In (entry_original entry) (map entry_original verified) ->
    ~ In (entry_tie entry) (map entry_tie verified) ->
    best_k_certificate capacity (entry :: verified)
      (insert_ranked entry prefix) (worst :: rejected).
Proof.
  intros capacity entry verified prefix worst rejected Hfull Hbetter
    Hbest Hfresh_original Hfresh_tie.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Horder Hnodup Htie].
  destruct (sorted_prefix_and_worst prefix worst Hsorted)
    as [Hprefix Hprefix_worst].
  assert (Hworst_in : In worst (prefix ++ [worst])).
  { apply in_or_app; right; now left. }
  constructor.
  - eapply replace_best_cover; exact Hcover.
  - now apply insert_ranked_preserves_sorted.
  - rewrite insert_ranked_length in *.
    rewrite length_app in Hfull; simpl in Hfull; lia.
  - intro Hunder.
    rewrite insert_ranked_length in Hunder.
    rewrite length_app in Hfull; simpl in Hfull; lia.
  - intros retained discarded Hretained Hdiscarded.
    apply insert_ranked_members in Hretained.
    destruct Hretained as [Hnew | Hold].
    + subst retained.
      destruct Hdiscarded as [Hworst | Hrejected].
      * subst discarded; now apply rank_lt_implies_le.
      * eapply rank_le_trans with (middle := worst).
        -- now apply rank_lt_implies_le.
        -- eapply Horder; eauto.
    + destruct Hdiscarded as [Hworst | Hrejected].
      * subst discarded; now apply Hprefix_worst.
      * eapply Horder; [apply in_or_app; now left | exact Hrejected].
  - now apply prepend_fresh_field_preserves_nodup.
  - now apply prepend_fresh_field_preserves_nodup.
Qed.

Lemma full_rejection_preserves_best_k :
  forall capacity entry verified prefix worst rejected,
    length (prefix ++ [worst]) = capacity ->
    ~ rank_lt entry worst ->
    best_k_certificate capacity verified (prefix ++ [worst]) rejected ->
    ~ In (entry_original entry) (map entry_original verified) ->
    ~ In (entry_tie entry) (map entry_tie verified) ->
    best_k_certificate capacity (entry :: verified)
      (prefix ++ [worst]) (entry :: rejected).
Proof.
  intros capacity entry verified prefix worst rejected Hfull Hnot_better
    Hbest Hfresh_original Hfresh_tie.
  destruct Hbest as [Hcover Hsorted Hcap Hfill Horder Hnodup Htie].
  destruct (sorted_prefix_and_worst prefix worst Hsorted)
    as [_ Hprefix_worst].
  constructor.
  - eapply reject_best_cover; exact Hcover.
  - exact Hsorted.
  - lia.
  - lia.
  - intros retained discarded Hretained Hdiscarded.
    destruct Hdiscarded as [Hnew | Hold].
    + subst discarded.
      eapply rank_le_trans with (middle := worst).
      * apply in_app_or in Hretained.
        destruct Hretained as [Hprefix | [Hequal | Hempty]].
        -- now apply Hprefix_worst.
        -- subst retained; apply rank_le_refl.
        -- contradiction.
      * now apply rank_not_lt_implies_reverse_le.
    + eapply Horder; eauto.
  - now apply prepend_fresh_field_preserves_nodup.
  - now apply prepend_fresh_field_preserves_nodup.
Qed.

Theorem best_k_step_preserves_certificate :
  forall capacity entry verified selected rejected
    next_selected next_rejected,
    best_k_certificate capacity verified selected rejected ->
    ~ In (entry_original entry) (map entry_original verified) ->
    ~ In (entry_tie entry) (map entry_tie verified) ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    best_k_certificate capacity (entry :: verified)
      next_selected next_rejected.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected Hbest Hfresh_original Hfresh_tie Hstep.
  inversion Hstep; subst.
  - eapply zero_capacity_preserves_best_k; eauto.
  - eapply underfull_admission_preserves_best_k; eauto.
  - eapply full_replacement_preserves_best_k; eauto.
  - eapply full_rejection_preserves_best_k; eauto.
Qed.

Theorem best_k_cardinality :
  forall capacity verified selected rejected,
    best_k_certificate capacity verified selected rejected ->
    length selected = Nat.min capacity (length verified).
Proof.
  intros capacity verified selected rejected Hbest.
  destruct Hbest as [Hcover _ Hcap Hfill _ _ _].
  pose proof (Permutation_length Hcover) as Hlength.
  rewrite length_app in Hlength.
  destruct (lt_dec (length selected) capacity) as [Hunder | Hfull].
  - rewrite (Hfill Hunder) in Hlength; simpl in Hlength.
    rewrite Nat.min_r by lia; lia.
  - rewrite Nat.min_l by lia; lia.
Qed.

(** The cutoff is absent for zero capacity and for every underfilled heap.
    A source implementation may use a sentinel to represent this [None],
    but its pruning rule must have exactly the same gate. *)
Definition kth_rank (capacity : nat) (selected : list ranked_entry)
    : option ranked_entry :=
  match capacity with
  | 0 => None
  | S _ =>
      if Nat.eq_dec (length selected) capacity then
        match split_last selected with
        | Some (_, worst) => Some worst
        | None => None
        end
      else None
  end.

Lemma kth_rank_zero : forall selected, kth_rank 0 selected = None.
Proof. reflexivity. Qed.

Lemma kth_rank_underfull : forall capacity selected,
  length selected < capacity -> kth_rank capacity selected = None.
Proof.
  intros [|capacity] selected Hunder; [simpl in Hunder; lia |].
  unfold kth_rank; destruct (Nat.eq_dec (length selected) (S capacity));
    [lia | reflexivity].
Qed.

Lemma kth_rank_some_full : forall capacity selected worst,
  kth_rank capacity selected = Some worst ->
  capacity > 0 /\ length selected = capacity /\
  exists prefix, selected = prefix ++ [worst].
Proof.
  intros [|capacity] selected worst Hrank; [discriminate |].
  unfold kth_rank in Hrank.
  destruct (Nat.eq_dec (length selected) (S capacity)) as [Hfull |];
    [|discriminate].
  destruct (split_last selected) as [[prefix last] |] eqn:Hsplit;
    [|discriminate].
  inversion Hrank; subst last.
  repeat split; try lia.
  exists prefix; eapply split_last_sound; eauto.
Qed.

Lemma kth_rank_full_exists : forall capacity selected,
  capacity > 0 -> length selected = capacity ->
  exists worst, kth_rank capacity selected = Some worst.
Proof.
  intros [|capacity] selected Hpositive Hfull; [lia |].
  assert (Hnonempty : selected <> []).
  { intro Hempty; subst selected; simpl in Hfull; lia. }
  destruct (split_last_complete Hnonempty) as
    [prefix [worst Hsplit]].
  exists worst; unfold kth_rank.
  destruct (Nat.eq_dec (length selected) (S capacity)) as [_ | Hneq];
    [now rewrite Hsplit | contradiction].
Qed.

Theorem kth_rank_is_selected_maximum :
  forall capacity verified selected rejected worst,
    best_k_certificate capacity verified selected rejected ->
    kth_rank capacity selected = Some worst ->
    forall member, In member selected -> rank_le member worst.
Proof.
  intros capacity verified selected rejected worst Hbest Hrank
    member Hmember.
  destruct (kth_rank_some_full capacity selected Hrank)
    as [_ [_ [prefix Hselected]]].
  subst selected.
  destruct Hbest as [_ Hsorted _ _ _ _ _].
  destruct (sorted_prefix_and_worst prefix worst Hsorted)
    as [_ Hprefix].
  apply in_app_or in Hmember.
  destruct Hmember as [Hin | [Hequal | []]].
  - now apply Hprefix.
  - subst member; apply rank_le_refl.
Qed.

Theorem kth_rank_iff_full_best_k :
  forall capacity verified selected rejected,
    best_k_certificate capacity verified selected rejected ->
    (exists worst, kth_rank capacity selected = Some worst) <->
    capacity > 0 /\ capacity <= length verified.
Proof.
  intros capacity verified selected rejected Hbest.
  split.
  - intros [worst Hrank].
    destruct (kth_rank_some_full capacity selected Hrank)
      as [Hpositive [Hfull _]].
    pose proof (best_k_cardinality Hbest) as Hsize.
    split; [exact Hpositive |].
    rewrite Hfull in Hsize; lia.
  - intros [Hpositive Hseen].
    apply kth_rank_full_exists; [exact Hpositive |].
    rewrite (best_k_cardinality Hbest).
    rewrite Nat.min_l by lia; reflexivity.
Qed.

Lemma split_last_app_singleton : forall prefix worst,
  split_last (prefix ++ [worst]) = Some (prefix, worst).
Proof.
  induction prefix as [|first rest IH]; intro worst; simpl.
  - reflexivity.
  - destruct rest as [|second tail].
    + reflexivity.
    + rewrite IH; reflexivity.
Qed.

Lemma kth_rank_full_last : forall capacity prefix worst,
  capacity > 0 -> length (prefix ++ [worst]) = capacity ->
  kth_rank capacity (prefix ++ [worst]) = Some worst.
Proof.
  intros [|capacity] prefix worst Hpositive Hfull; [lia |].
  unfold kth_rank.
  destruct (Nat.eq_dec (length (prefix ++ [worst])) (S capacity))
    as [_ | Hneq]; [|contradiction].
  now rewrite split_last_app_singleton.
Qed.

Theorem full_kth_rank_nonincreasing :
  forall capacity entry verified selected rejected
    next_selected next_rejected previous current,
    best_k_certificate capacity verified selected rejected ->
    best_k_step capacity entry selected rejected
      next_selected next_rejected ->
    kth_rank capacity selected = Some previous ->
    kth_rank capacity next_selected = Some current ->
    rank_le current previous.
Proof.
  intros capacity entry verified selected rejected next_selected
    next_rejected previous current Hbest Hstep Hprevious Hcurrent.
  destruct Hstep.
  - subst capacity; discriminate.
  - destruct (kth_rank_some_full capacity selected Hprevious)
      as [_ [Hfull _]].
    lia.
  - assert (Hpositive : capacity > 0).
    { rewrite length_app in H; simpl in H; lia. }
    pose proof (kth_rank_full_last prefix worst Hpositive H)
      as Hbefore.
    rewrite Hbefore in Hprevious; inversion Hprevious; subst previous.
    destruct (kth_rank_some_full capacity
      (insert_ranked entry prefix) Hcurrent)
      as [_ [_ [next_prefix Hnext]]].
    assert (Hmember : In current (insert_ranked entry prefix)).
    { rewrite Hnext; apply in_or_app; right; now left. }
    apply insert_ranked_members in Hmember.
    destruct (sorted_prefix_and_worst prefix worst (best_sorted Hbest))
      as [_ Hprefix_worst].
    destruct Hmember as [Hnew | Hold].
    + subst current; now apply rank_lt_implies_le.
    + now apply Hprefix_worst.
  - rewrite Hprevious in Hcurrent.
    inversion Hcurrent; subst current; apply rank_le_refl.
Qed.

(** A concrete max heap may permute the retained entries. Its root must
    dominate every other retained rank. The proof below identifies the root
    with the sorted kth rank without requiring a sorted production heap. *)
Definition max_heap_representation (heap selected : list ranked_entry) : Prop :=
  Permutation heap selected /\
  match heap with
  | [] => True
  | root :: tail => Forall (fun member => rank_le member root) tail
  end.

Lemma rank_le_antisym_fields : forall left right,
  rank_le left right -> rank_le right left ->
  entry_cost left = entry_cost right /\
  entry_tie left = entry_tie right.
Proof.
  intros left right Hforward Hbackward.
  unfold rank_le in *.
  destruct Hforward as [Hcost | [Hcost Htie]];
    destruct Hbackward as [Hcost_back | [Hcost_back Htie_back]];
    lia.
Qed.

Theorem heap_root_has_kth_rank :
  forall capacity verified selected rejected heap worst,
    best_k_certificate capacity verified selected rejected ->
    max_heap_representation heap selected ->
    kth_rank capacity selected = Some worst ->
    exists root tail,
      heap = root :: tail /\
      entry_cost root = entry_cost worst /\
      entry_tie root = entry_tie worst.
Proof.
  intros capacity verified selected rejected heap worst Hbest
    [Hperm Hroot] Hkth.
  destruct (kth_rank_some_full capacity selected Hkth)
    as [Hpositive [Hfull [prefix Hselected]]].
  destruct heap as [|root tail].
  - pose proof (Permutation_length Hperm) as Hlength.
    simpl in Hlength; rewrite Hfull in Hlength; lia.
  - exists root, tail; split; [reflexivity |].
    apply rank_le_antisym_fields.
    + eapply kth_rank_is_selected_maximum; [exact Hbest | exact Hkth |].
      eapply Permutation_in; [exact Hperm | now left].
    + assert (Hworst_selected : In worst selected).
      { rewrite Hselected; apply in_or_app; right; now left. }
      assert (Hworst_heap : In worst (root :: tail)).
      { eapply Permutation_in; [apply Permutation_sym, Hperm |
        exact Hworst_selected]. }
      destruct Hworst_heap as [Hequal | Htail].
      * subst worst; apply rank_le_refl.
      * apply Forall_forall with (x := worst) in Hroot; assumption.
Qed.

Definition tie_zero : ranked_entry :=
  {| entry_original := 0; entry_cost := 5; entry_tie := 0 |}.
Definition tie_one : ranked_entry :=
  {| entry_original := 1; entry_cost := 5; entry_tie := 1 |}.
Definition later_worse : ranked_entry :=
  {| entry_original := 2; entry_cost := 6; entry_tie := 2 |}.

Example tie_comparator_control :
  insert_ranked tie_zero [tie_one] = [tie_zero; tie_one].
Proof. reflexivity. Qed.

(** A reversed equal-cost tie comparator retains the wrong singleton. *)
Example reversed_tie_mutant_rejected :
  ~ best_k_certificate 1 [tie_one; tie_zero] [tie_one] [tie_zero].
Proof.
  intro Hbad.
  destruct Hbad as [_ _ _ _ Horder _ _].
  specialize (Horder tie_one tie_zero (or_introl eq_refl)
    (or_introl eq_refl)).
  unfold rank_le, tie_one, tie_zero in Horder; simpl in Horder; lia.
Qed.

Example underfull_has_no_kth : kth_rank 2 [tie_zero] = None.
Proof. reflexivity. Qed.

(** Using a one-entry cutoff to discard the next candidate at k=2
    violates the fill-before-reject invariant. *)
Example premature_cutoff_mutant_rejected :
  ~ best_k_certificate 2 [later_worse; tie_zero]
      [tie_zero] [later_worse].
Proof.
  intro Hbad.
  destruct Hbad as [_ _ _ Hfill _ _ _].
  specialize (Hfill ltac:(simpl; lia)).
  discriminate.
Qed.

Example zero_capacity_has_no_kth : kth_rank 0 [] = None.
Proof. reflexivity. Qed.

Fixpoint sort_ranked (entries : list ranked_entry) : list ranked_entry :=
  match entries with
  | [] => []
  | entry :: rest => insert_ranked entry (sort_ranked rest)
  end.

Lemma sort_ranked_sorted : forall entries,
  StronglySorted rank_le (sort_ranked entries).
Proof.
  induction entries as [|entry rest IH]; simpl.
  - constructor.
  - now apply insert_ranked_preserves_sorted.
Qed.

Lemma sort_ranked_permutation : forall entries,
  Permutation (sort_ranked entries) entries.
Proof.
  induction entries as [|entry rest IH]; simpl.
  - reflexivity.
  - eapply Permutation_trans.
    + apply insert_ranked_permutation.
    + now apply perm_skip.
Qed.

Lemma nodup_field_injective_on_list :
  forall (field : ranked_entry -> nat) entries left right,
    NoDup (map field entries) ->
    In left entries -> In right entries ->
    field left = field right -> left = right.
Proof.
  intros field entries; induction entries as [|first rest IH];
    intros left right Hnodup Hleft Hright Hequal.
  - contradiction.
  - simpl in Hnodup, Hleft, Hright.
    inversion Hnodup as [|? ? Hnot Htail]; subst.
    destruct Hleft as [Hleft | Hleft];
      destruct Hright as [Hright | Hright].
    + congruence.
    + subst left.
      exfalso; apply Hnot.
      rewrite Hequal; apply in_map; exact Hright.
    + subst right.
      exfalso; apply Hnot.
      rewrite <- Hequal; apply in_map; exact Hleft.
    + eapply IH; eauto.
Qed.

Lemma strongly_sorted_app : forall left right,
  StronglySorted rank_le left ->
  StronglySorted rank_le right ->
  (forall x y, In x left -> In y right -> rank_le x y) ->
  StronglySorted rank_le (left ++ right).
Proof.
  intros left right Hleft Hright Hcross.
  induction Hleft as [|first rest Hsorted IH Hfor]; simpl.
  - exact Hright.
  - constructor.
    + apply IH.
      intros x y Hx Hy; apply Hcross; [now right | exact Hy].
    + apply Forall_forall; intros member Hmember.
      apply in_app_or in Hmember.
      destruct Hmember as [Hrest | Hright_member].
      * now apply Forall_forall with (x := member) in Hfor.
      * apply Hcross; [now left | exact Hright_member].
Qed.

Lemma strongly_sorted_permutation_unique : forall left right,
  StronglySorted rank_le left ->
  StronglySorted rank_le right ->
  Permutation left right ->
  NoDup (map entry_tie left) ->
  left = right.
Proof.
  induction left as [|first rest IH]; intros right Hleft Hright Hperm Hnodup.
  - destruct right as [|other remaining]; [reflexivity |].
    pose proof (Permutation_length Hperm) as Hlength;
      simpl in Hlength; discriminate.
  - destruct right as [|other remaining].
    + pose proof (Permutation_length Hperm) as Hlength;
        simpl in Hlength; discriminate.
    + inversion Hleft as [|? ? Hleft_tail Hleft_for]; subst.
      inversion Hright as [|? ? Hright_tail Hright_for]; subst.
      inversion Hnodup as [|? ? Hnot Htail_nodup]; subst.
      assert (Hother_left : In other (first :: rest)).
      { eapply Permutation_in; [apply Permutation_sym, Hperm | now left]. }
      assert (Hfirst_right : In first (other :: remaining)).
      { eapply Permutation_in; [exact Hperm | now left]. }
      assert (Hfirst_other : rank_le first other).
      { destruct Hother_left as [Hequal | Hrest].
        - subst other; apply rank_le_refl.
        - now apply Forall_forall with (x := other) in Hleft_for. }
      assert (Hother_first : rank_le other first).
      { destruct Hfirst_right as [Hequal | Hremaining].
        - subst first; apply rank_le_refl.
        - now apply Forall_forall with (x := first) in Hright_for. }
      destruct (rank_le_antisym_fields Hfirst_other Hother_first)
        as [_ Hsame_tie].
      assert (Hsame : first = other).
      { eapply nodup_field_injective_on_list;
          [exact Hnodup | now left | exact Hother_left | exact Hsame_tie]. }
      subst other.
      f_equal.
      eapply IH; [exact Hleft_tail | exact Hright_tail | | exact Htail_nodup].
      now apply Permutation_cons_inv in Hperm.
Qed.

Theorem certified_best_k_is_canonical :
  forall capacity verified selected rejected,
    best_k_certificate capacity verified selected rejected ->
    selected = firstn capacity (sort_ranked verified).
Proof.
  intros capacity verified selected rejected Hbest.
  destruct Hbest as [Hcover Hselected_sorted Hcap Hfill Hcross
    Horiginals Hties].
  assert (Hsorted_all :
    StronglySorted rank_le (selected ++ sort_ranked rejected)).
  { apply strongly_sorted_app; [exact Hselected_sorted |
      apply sort_ranked_sorted |].
    intros x y Hx Hy.
    eapply Hcross; [exact Hx |].
    eapply Permutation_in; [apply sort_ranked_permutation | exact Hy]. }
  assert (Hperm_all :
    Permutation (selected ++ sort_ranked rejected) (sort_ranked verified)).
  { eapply Permutation_trans.
    - apply Permutation_app; [reflexivity |
        apply sort_ranked_permutation].
    - eapply Permutation_trans; [apply Permutation_sym, Hcover |
        apply Permutation_sym, sort_ranked_permutation]. }
  assert (Hties_all :
    NoDup (map entry_tie (selected ++ sort_ranked rejected))).
  { eapply Permutation_NoDup.
    - apply Permutation_map.
      apply Permutation_sym.
      eapply Permutation_trans; [exact Hperm_all |
        apply sort_ranked_permutation].
    - exact Hties. }
  pose proof (strongly_sorted_permutation_unique
    Hsorted_all (sort_ranked_sorted verified) Hperm_all Hties_all)
    as Hsame.
  rewrite <- Hsame.
  rewrite firstn_app.
  destruct (lt_dec (length selected) capacity) as [Hunder | Hfull].
  - rewrite (Hfill Hunder); simpl.
    rewrite firstn_all2 by lia.
    rewrite firstn_nil, app_nil_r; reflexivity.
  - assert (Hequal : length selected = capacity) by lia.
    rewrite <- Hequal.
    rewrite firstn_all; simpl.
    rewrite Nat.sub_diag; simpl; rewrite app_nil_r; reflexivity.
Qed.
