(** * Checked partition of original occurrences at a dictionary split

    The identifier names an original occurrence, not a physical dictionary node.
    A split package lists terminal originals and complete child regions.
    The checker accepts exactly when that flattened list is a permutation of
    the parent's originals. Parent enumeration and Rust cursor correspondence
    remain separate obligations. *)

From Stdlib Require Import Arith Bool List Permutation.
Import ListNotations.

Section PartitionChecker.
  Context {Original : Type}.
  Variable original_eq_dec : forall left right : Original,
    {left = right} + {left <> right}.

Fixpoint erase_original (original : Original) (members : list Original)
    : option (list Original) :=
  match members with
  | [] => None
  | first :: remaining =>
      if original_eq_dec original first then Some remaining
      else option_map (cons first) (erase_original original remaining)
  end.

Theorem erase_original_sound : forall original members rest,
  erase_original original members = Some rest ->
  Permutation members (original :: rest).
Proof.
  intros original members.
  induction members as [|first remaining IH]; intros rest Herase;
    simpl in Herase; [discriminate |].
  destruct (original_eq_dec original first) as [Heq | Hneq].
  - inversion Herase; subst. reflexivity.
  - destruct (erase_original original remaining) as [middle |]
      eqn:Hmiddle; [|discriminate].
    inversion Herase; subst rest.
    specialize (IH middle eq_refl).
    eapply Permutation_trans.
    + apply perm_skip; exact IH.
    + apply perm_swap.
Qed.

Theorem erase_original_exists : forall original members,
  In original members ->
  exists rest, erase_original original members = Some rest.
Proof.
  intros original members.
  induction members as [|first remaining IH]; intro Hin;
    simpl in Hin; [contradiction |].
  destruct Hin as [Heq | Hin].
  - subst first. exists remaining; simpl.
    destruct (original_eq_dec original original); [reflexivity | contradiction].
  - destruct (IH Hin) as [rest Hrest].
    simpl. destruct (original_eq_dec original first) as [Heq | Hneq].
    + exists remaining; reflexivity.
    + exists (first :: rest); now rewrite Hrest.
Qed.

Fixpoint consume_originals (parent emitted : list Original)
    : option (list Original) :=
  match emitted with
  | [] => Some parent
  | original :: remaining =>
      match erase_original original parent with
      | Some rest => consume_originals rest remaining
      | None => None
      end
  end.

Theorem consume_originals_sound : forall emitted parent rest,
  consume_originals parent emitted = Some rest ->
  Permutation parent (emitted ++ rest).
Proof.
  induction emitted as [|original remaining IH];
    intros parent rest Hconsume; simpl in Hconsume.
  - inversion Hconsume; subst; reflexivity.
  - destruct (erase_original original parent) as [middle |]
      eqn:Herase; [|discriminate].
    specialize (erase_original_sound _ _ _ Herase) as Hhead.
    specialize (IH _ _ Hconsume).
    eapply Permutation_trans; [exact Hhead |].
    simpl. apply perm_skip; exact IH.
Qed.

Theorem consume_originals_complete : forall emitted parent,
  Permutation parent emitted ->
  consume_originals parent emitted = Some [].
Proof.
  induction emitted as [|original remaining IH]; intros parent Hperm.
  - destruct parent as [|first rest]; [reflexivity |].
    apply Permutation_length in Hperm; simpl in Hperm; discriminate.
  - destruct (erase_original_exists original parent)
      as [rest Herase].
    { eapply Permutation_in; [apply Permutation_sym; exact Hperm |].
      simpl; now left. }
    simpl; rewrite Herase.
    apply IH.
    pose proof (erase_original_sound _ _ _ Herase) as Hremoved.
    eapply Permutation_cons_inv.
    eapply Permutation_trans; [apply Permutation_sym; exact Hremoved |].
    exact Hperm.
Qed.

Definition accept_permutation (parent emitted : list Original) : bool :=
  match consume_originals parent emitted with
  | Some [] => true
  | _ => false
  end.

Theorem accept_permutation_reflects_exact_coverage : forall parent emitted,
  accept_permutation parent emitted = true <->
  Permutation parent emitted.
Proof.
  intros parent emitted; split.
  - unfold accept_permutation.
    destruct (consume_originals parent emitted) as [[|extra rest] |]
      eqn:Hconsume; try discriminate.
    intro Haccepted.
    pose proof (consume_originals_sound _ _ _ Hconsume) as Hperm.
    now rewrite app_nil_r in Hperm.
  - intro Hperm. unfold accept_permutation.
    now rewrite (consume_originals_complete _ _ Hperm).
Qed.

Record split_package := {
  terminal_originals : list Original;
  child_original_regions : list (list Original)
}.

Definition flattened_split (package : split_package) : list Original :=
  terminal_originals package ++ concat (child_original_regions package).

Definition accept_split (parent : list Original) (package : split_package) : bool :=
  accept_permutation parent (flattened_split package).

Theorem accepted_split_has_exact_original_coverage : forall parent package,
  accept_split parent package = true <->
  Permutation parent (flattened_split package).
Proof.
  intros parent package; unfold accept_split.
  apply accept_permutation_reflects_exact_coverage.
Qed.

Theorem accepted_split_preserves_unique_originals : forall parent package,
  NoDup parent -> accept_split parent package = true ->
  NoDup (flattened_split package).
Proof.
  intros parent package Hunique Haccept.
  apply accepted_split_has_exact_original_coverage in Haccept.
  eapply Permutation_NoDup; eauto.
Qed.

Theorem accepted_split_has_no_missing_or_foreign_original :
  forall parent package original,
    accept_split parent package = true ->
    (In original parent <-> In original (flattened_split package)).
Proof.
  intros parent package original Haccept.
  apply accepted_split_has_exact_original_coverage in Haccept.
  split; intro Hin.
  - eapply Permutation_in; eauto.
  - eapply Permutation_in; [apply Permutation_sym; exact Haccept | exact Hin].
Qed.

Theorem accepted_split_replaces_parent_in_any_ownership_context :
  forall universe before parent after package,
    Permutation (before ++ parent ++ after) universe ->
    accept_split parent package = true ->
    Permutation (before ++ flattened_split package ++ after) universe.
Proof.
  intros universe before parent after package Hglobal Haccept.
  apply accepted_split_has_exact_original_coverage in Haccept.
  eapply Permutation_trans; [|exact Hglobal].
  apply Permutation_app_head.
  apply Permutation_app_tail.
  now apply Permutation_sym.
Qed.

End PartitionChecker.

Definition shared_node (_original : nat) : nat := 7.

Example terminal_and_children_partition_parent :
  @accept_split nat Nat.eq_dec [1; 2; 3]
    {| terminal_originals := [1]; child_original_regions := [[2]; [3]] |}
  = true.
Proof. reflexivity. Qed.

Example shared_physical_node_does_not_merge_originals :
  shared_node 1 = shared_node 2 /\
  @accept_split nat Nat.eq_dec [1; 2]
    {| terminal_originals := []; child_original_regions := [[1]; [2]] |}
  = true.
Proof. now split. Qed.

Example node_only_dedup_loses_a_collision_member :
  @accept_split nat Nat.eq_dec [1; 2]
    {| terminal_originals := []; child_original_regions := [[1]] |}
  = false.
Proof. reflexivity. Qed.

Example missing_terminal_is_rejected :
  @accept_split nat Nat.eq_dec [1; 2; 3]
    {| terminal_originals := []; child_original_regions := [[2]; [3]] |}
  = false.
Proof. reflexivity. Qed.

Example duplicate_original_is_rejected :
  @accept_split nat Nat.eq_dec [1; 2]
    {| terminal_originals := [1]; child_original_regions := [[1]; [2]] |}
  = false.
Proof. reflexivity. Qed.

Example foreign_original_is_rejected :
  @accept_split nat Nat.eq_dec [1; 2]
    {| terminal_originals := [1]; child_original_regions := [[4]] |}
  = false.
Proof. reflexivity. Qed.

(** A path and bucket slot form a logical original identity. The physical
    dictionary node can be shared by two paths without identifying them. *)
Definition path_slot_occurrence := (list nat * nat)%type.

Definition path_slot_eq_dec : forall left right : path_slot_occurrence,
    {left = right} + {left <> right}.
Proof.
  decide equality;
    try apply Nat.eq_dec;
    apply list_eq_dec; apply Nat.eq_dec.
Defined.

Lemma different_paths_have_distinct_original_occurrences :
  forall (left right : list nat) (slot : nat),
    left <> right -> (left, slot) <> (right, slot).
Proof.
  intros left right slot Hdifferent Hequal.
  inversion Hequal; contradiction.
Qed.

Definition shared_physical_node (_ : path_slot_occurrence) : nat := 7.

Example path_sensitive_split_retains_both_shared_node_occurrences :
  let first := ([1], 0) in
  let second := ([2], 0) in
  shared_physical_node first = shared_physical_node second /\
  first <> second /\
  @accept_split path_slot_occurrence path_slot_eq_dec [first; second]
    {| terminal_originals := [];
       child_original_regions := [[first]; [second]] |} = true.
Proof. simpl; repeat split; discriminate. Qed.

Example path_insensitive_split_loses_a_shared_node_occurrence :
  let first := ([1], 0) in
  let second := ([2], 0) in
  @accept_split path_slot_occurrence path_slot_eq_dec [first; second]
    {| terminal_originals := []; child_original_regions := [[first]] |}
  = false.
Proof. reflexivity. Qed.
