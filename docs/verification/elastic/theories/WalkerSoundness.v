(** * Traversal-level soundness of the generic elastic walker

    This development proves the recursive search property promised by the
    [ElasticKernel] contract.  A node carries a lower bound and may carry an
    exact terminal distance.  The mutually inductive forest represents its
    dictionary children without assuming a branching factor.

    [tree_sound] is the executable proof interface:

    - K1: a terminal node's bound is no greater than its exact distance;
    - K2: a child's bound is no smaller than its parent's bound;
    - the same obligations hold recursively below every child.

    [walk_tree] is the depth-first range walker.  It rejects an entire subtree
    exactly when the node bound exceeds the inclusive cutoff.  The final
    theorem states traversal-level completeness: every exact terminal within
    the cutoff occurs in the walker's output.  There are no axioms, admitted
    obligations, or abstract oracle functions in this model.
*)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Inductive search_tree : Type :=
  | Node : nat -> option nat -> search_forest -> search_tree
with search_forest : Type :=
  | FNil : search_forest
  | FCons : search_tree -> search_forest -> search_forest.

Definition node_bound (tree : search_tree) : nat :=
  match tree with Node bound _ _ => bound end.

Definition terminal_values (terminal : option nat) : list nat :=
  match terminal with Some exact => [exact] | None => [] end.

Definition eligible_terminal (cutoff : nat) (terminal : option nat) : list nat :=
  match terminal with
  | Some exact => if exact <=? cutoff then [exact] else []
  | None => []
  end.

Fixpoint all_terminals (tree : search_tree) : list nat :=
  match tree with
  | Node _ terminal children => terminal_values terminal ++ all_forest_terminals children
  end
with all_forest_terminals (forest : search_forest) : list nat :=
  match forest with
  | FNil => []
  | FCons child rest => all_terminals child ++ all_forest_terminals rest
  end.

Fixpoint tree_sound (tree : search_tree) : Prop :=
  match tree with
  | Node bound terminal children =>
      (match terminal with Some exact => bound <= exact | None => True end) /\
      forest_sound bound children
  end
with forest_sound (parent_bound : nat) (forest : search_forest) : Prop :=
  match forest with
  | FNil => True
  | FCons child rest =>
      parent_bound <= node_bound child /\
      tree_sound child /\
      forest_sound parent_bound rest
  end.

Fixpoint walk_tree (cutoff : nat) (tree : search_tree) : list nat :=
  match tree with
  | Node bound terminal children =>
      if bound <=? cutoff
      then eligible_terminal cutoff terminal ++ walk_forest cutoff children
      else []
  end
with walk_forest (cutoff : nat) (forest : search_forest) : list nat :=
  match forest with
  | FNil => []
  | FCons child rest => walk_tree cutoff child ++ walk_forest cutoff rest
  end.

Lemma eligible_terminal_complete : forall terminal cutoff exact,
  In exact (terminal_values terminal) ->
  exact <= cutoff ->
  In exact (eligible_terminal cutoff terminal).
Proof.
  intros [value |] cutoff exact Hin Hcutoff; simpl in *.
  - destruct Hin as [Heq | []].
    subst value.
    apply Nat.leb_le in Hcutoff.
    rewrite Hcutoff; simpl; auto.
  - contradiction.
Qed.

Lemma eligible_terminal_sound : forall terminal cutoff exact,
  In exact (eligible_terminal cutoff terminal) ->
  In exact (terminal_values terminal) /\ exact <= cutoff.
Proof.
  intros [value |] cutoff exact Hin; simpl in *.
  - destruct (value <=? cutoff) eqn:Hguard; simpl in Hin; try contradiction.
    destruct Hin as [Heq | []].
    subst value.
    split; [simpl; auto | apply Nat.leb_le; exact Hguard].
  - contradiction.
Qed.

Theorem tree_bound_lower_bounds_every_terminal : forall tree exact,
  tree_sound tree ->
  In exact (all_terminals tree) ->
  node_bound tree <= exact
with forest_parent_lower_bounds_every_terminal : forall parent_bound forest exact,
  forest_sound parent_bound forest ->
  In exact (all_forest_terminals forest) ->
  parent_bound <= exact.
Proof.
  - intros [bound terminal children] exact Hsound Hin.
    simpl in Hsound, Hin |- *.
    destruct Hsound as [Hterminal Hchildren].
    apply in_app_or in Hin as [Hhere | Hbelow].
    + destruct terminal as [value |]; simpl in Hhere, Hterminal.
      * destruct Hhere as [Heq | []]. now subst value.
      * contradiction.
    + eapply forest_parent_lower_bounds_every_terminal; eauto.
  - intros parent_bound [|child rest] exact Hsound Hin.
    + contradiction.
    + simpl in Hsound, Hin.
      destruct Hsound as [Hedge [Hchild Hrest]].
      apply in_app_or in Hin as [Hin_child | Hin_rest].
      * eapply Nat.le_trans; [exact Hedge |].
        eapply tree_bound_lower_bounds_every_terminal; eauto.
      * eapply forest_parent_lower_bounds_every_terminal; eauto.
Qed.

Theorem rejected_subtree_contains_no_qualifying_terminal : forall tree cutoff exact,
  tree_sound tree ->
  cutoff < node_bound tree ->
  In exact (all_terminals tree) ->
  cutoff < exact.
Proof.
  intros tree cutoff exact Hsound Hreject Hin.
  pose proof (tree_bound_lower_bounds_every_terminal tree exact Hsound Hin).
  lia.
Qed.

Theorem walk_tree_complete : forall tree cutoff exact,
  tree_sound tree ->
  In exact (all_terminals tree) ->
  exact <= cutoff ->
  In exact (walk_tree cutoff tree)
with walk_forest_complete : forall parent_bound forest cutoff exact,
  forest_sound parent_bound forest ->
  In exact (all_forest_terminals forest) ->
  exact <= cutoff ->
  In exact (walk_forest cutoff forest).
Proof.
  - intros [bound terminal children] cutoff exact Hsound Hin Hcutoff.
    simpl in Hsound, Hin |- *.
    destruct Hsound as [Hterminal Hchildren].
    assert (bound <= exact) as Hbound.
    { apply in_app_or in Hin as [Hhere | Hbelow].
      - destruct terminal as [value |]; simpl in Hhere, Hterminal.
        + destruct Hhere as [Heq | []]. now subst value.
        + contradiction.
      - eapply forest_parent_lower_bounds_every_terminal; eauto. }
    assert (bound <= cutoff) as Hguard by lia.
    apply Nat.leb_le in Hguard.
    rewrite Hguard.
    apply in_app_or in Hin as [Hhere | Hbelow].
    + apply in_or_app; left.
      eapply eligible_terminal_complete; eauto.
    + apply in_or_app; right.
      eapply walk_forest_complete; eauto.
  - intros parent_bound [|child rest] cutoff exact Hsound Hin Hcutoff.
    + contradiction.
    + simpl in Hsound, Hin |- *.
      destruct Hsound as [Hedge [Hchild Hrest]].
      apply in_app_or in Hin as [Hin_child | Hin_rest].
      * apply in_or_app; left.
        eapply walk_tree_complete; eauto.
      * apply in_or_app; right.
        eapply walk_forest_complete; eauto.
Qed.

(** K1 plus K2 imply the advertised no-false-negative contract for the full
    recursive walker, not merely for one arithmetic pruning decision. *)
Corollary k1_k2_imply_no_false_negatives : forall tree cutoff exact,
  tree_sound tree ->
  In exact (all_terminals tree) ->
  exact <= cutoff ->
  In exact (walk_tree cutoff tree).
Proof. exact walk_tree_complete. Qed.

Theorem walk_tree_sound : forall tree cutoff exact,
  In exact (walk_tree cutoff tree) ->
  In exact (all_terminals tree) /\ exact <= cutoff
with walk_forest_sound : forall forest cutoff exact,
  In exact (walk_forest cutoff forest) ->
  In exact (all_forest_terminals forest) /\ exact <= cutoff.
Proof.
  - intros [bound terminal children] cutoff exact Hin.
    simpl in Hin |- *.
    destruct (bound <=? cutoff) eqn:Hguard; try contradiction.
    apply in_app_or in Hin as [Hhere | Hbelow].
    + apply eligible_terminal_sound in Hhere as [Hterminal Hcutoff].
      split; [apply in_or_app; left; exact Hterminal | exact Hcutoff].
    + apply walk_forest_sound in Hbelow as [Hforest Hcutoff].
      split; [apply in_or_app; right; exact Hforest | exact Hcutoff].
  - intros [|child rest] cutoff exact Hin.
    + contradiction.
    + simpl in Hin |- *.
      apply in_app_or in Hin as [Hin_child | Hin_rest].
      * apply walk_tree_sound in Hin_child as [Hchild Hcutoff].
        split; [apply in_or_app; left; exact Hchild | exact Hcutoff].
      * apply walk_forest_sound in Hin_rest as [Hrest Hcutoff].
        split; [apply in_or_app; right; exact Hrest | exact Hcutoff].
Qed.

Example shared_prefix_walk_keeps_both_qualifying_terminals :
  let tree := Node 1 None
    (FCons (Node 2 (Some 2) FNil)
      (FCons (Node 3 (Some 3) FNil)
        (FCons (Node 5 (Some 5) FNil) FNil))) in
  tree_sound tree /\ walk_tree 3 tree = [2; 3].
Proof. simpl; repeat split; lia. Qed.
