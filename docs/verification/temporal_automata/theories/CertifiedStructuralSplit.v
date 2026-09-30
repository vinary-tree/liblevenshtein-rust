(** * A construction that partitions original occurrences by one node decision

    A classifier sends each original to the terminal bucket or one labelled
    child. The builder processes every parent original exactly once. It
    produces a unique child bucket per label and a package accepted by the
    complete finite partition checker. The classifier's correspondence with
    captured dictionary traversal is a separate executable obligation. *)

From Stdlib Require Import Arith List Permutation.
Require Import CertifiedRegionPartition.
Import ListNotations.

Section StructuralSplit.
  Context {Original Label : Type}.
  Variable original_eq_dec : forall left right : Original,
    {left = right} + {left <> right}.
  Variable label_eq_dec : forall left right : Label,
    {left = right} + {left <> right}.

  Definition child_buckets := list (Label * list Original).

  Definition bucket_labels (buckets : child_buckets) : list Label :=
    map fst buckets.

  Definition bucket_originals (buckets : child_buckets) : list Original :=
    concat (map snd buckets).

  Fixpoint insert_child (label : Label) (original : Original)
      (buckets : child_buckets) : child_buckets :=
    match buckets with
    | [] => [(label, [original])]
    | (first_label, originals) :: remaining =>
        if label_eq_dec label first_label then
          (first_label, original :: originals) :: remaining
        else
          (first_label, originals) :: insert_child label original remaining
    end.

  Theorem insert_child_preserves_originals :
    forall label original buckets,
      Permutation
        (bucket_originals (insert_child label original buckets))
        (original :: bucket_originals buckets).
  Proof.
    intros label original buckets.
    induction buckets as [|[first originals] remaining IH]; simpl;
      [reflexivity |].
    destruct (label_eq_dec label first) as [Heq | Hneq]; simpl.
    - reflexivity.
    - eapply Permutation_trans.
      + apply Permutation_app_head; exact IH.
      + symmetry; apply Permutation_middle.
  Qed.

  Lemma inserted_bucket_labels : forall label original buckets other,
    In other (bucket_labels (insert_child label original buckets)) <->
    other = label \/ In other (bucket_labels buckets).
  Proof.
    intros label original buckets.
    induction buckets as [|[first originals] remaining IH];
      intro other; simpl.
    - firstorder congruence.
    - destruct (label_eq_dec label first) as [Heq | Hneq]; simpl.
      + subst label; firstorder congruence.
      + rewrite IH. tauto.
  Qed.

  Theorem insert_child_preserves_unique_labels :
    forall label original buckets,
      NoDup (bucket_labels buckets) ->
      NoDup (bucket_labels (insert_child label original buckets)).
  Proof.
    intros label original buckets.
    induction buckets as [|[first originals] remaining IH];
      intro Hunique; simpl in *.
    - constructor; [intro Hin; inversion Hin | constructor].
    - inversion Hunique as [|? ? Hnot Hrest]; subst.
      destruct (label_eq_dec label first) as [Heq | Hneq]; simpl.
      + exact Hunique.
      + constructor.
        * intro Hin. apply inserted_bucket_labels in Hin.
          destruct Hin as [Hequal | Hin].
          -- symmetry in Hequal; contradiction.
          -- contradiction.
        * apply IH; exact Hrest.
  Qed.

  Record labelled_split := {
    labelled_terminals : list Original;
    labelled_children : child_buckets
  }.

  Definition erase_labels (split : labelled_split) : split_package :=
    {| terminal_originals := labelled_terminals split;
       child_original_regions := map snd (labelled_children split) |}.

  Definition labelled_originals (split : labelled_split) : list Original :=
    labelled_terminals split ++ bucket_originals (labelled_children split).

  Fixpoint build_split (classify : Original -> option Label)
      (parent : list Original) : labelled_split :=
    match parent with
    | [] => {| labelled_terminals := []; labelled_children := [] |}
    | original :: remaining =>
        let tail := build_split classify remaining in
        match classify original with
        | None =>
            {| labelled_terminals := original :: labelled_terminals tail;
               labelled_children := labelled_children tail |}
        | Some label =>
            {| labelled_terminals := labelled_terminals tail;
               labelled_children :=
                 insert_child label original (labelled_children tail) |}
        end
    end.

  Theorem build_split_preserves_every_original :
    forall classify parent,
      Permutation (labelled_originals (build_split classify parent)) parent.
  Proof.
    intros classify parent.
    induction parent as [|original remaining IH]; simpl; [reflexivity |].
    destruct (build_split classify remaining) as [terminals children] eqn:Htail;
      simpl in *.
    destruct (classify original) as [label |]; simpl.
    - eapply Permutation_trans.
      + apply Permutation_app_head.
        apply insert_child_preserves_originals.
      + eapply Permutation_trans.
        * symmetry; apply Permutation_middle.
        * apply perm_skip; exact IH.
    - apply perm_skip; exact IH.
  Qed.

  Theorem build_split_has_unique_child_labels :
    forall classify parent,
      NoDup (bucket_labels (labelled_children (build_split classify parent))).
  Proof.
    intros classify parent.
    induction parent as [|original remaining IH]; simpl; [constructor |].
    destruct (classify original) as [label |]; simpl.
    - apply insert_child_preserves_unique_labels; exact IH.
    - exact IH.
  Qed.

  Theorem constructed_split_is_accepted : forall classify parent,
    accept_split original_eq_dec parent
      (erase_labels (build_split classify parent)) = true.
  Proof.
    intros classify parent.
    apply accepted_split_has_exact_original_coverage.
    unfold erase_labels, flattened_split, labelled_originals; simpl.
    apply Permutation_sym.
    apply build_split_preserves_every_original.
  Qed.

  Theorem constructed_split_preserves_unique_originals :
    forall classify parent,
      NoDup parent ->
      NoDup (labelled_originals (build_split classify parent)).
  Proof.
    intros classify parent Hunique.
    eapply Permutation_NoDup.
    - apply Permutation_sym; apply build_split_preserves_every_original.
    - exact Hunique.
  Qed.
End StructuralSplit.
