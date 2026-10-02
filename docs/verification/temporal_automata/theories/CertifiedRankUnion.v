(** * Complete finite union of scoped ordered-rank certificates

    A captured operation owns each original exactly once across child
    regions, live terminal originals, and private work. [Permutation] records
    the complete partition, and [NoDup] rules out duplicate ownership. Each
    piece supplies its own scoped semantic certificate. A common rank floor
    or strict cost cut is accepted only when every piece, including live and
    private pieces, supplies that consequence. Empty pieces may use
    [RankEmpty]; [RankUnknown] cannot be silently strengthened.

    This is a finite proof-level model. Source enumeration, actual bound
    producers, numeric authority, heap fullness, and Rust correspondence
    remain separate obligations. *)

From Stdlib Require Import Arith Lia List Sorting.Permutation.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedGenericRankOrder.
Import ListNotations.

Section ScopedUnion.
  Context {Scope Original Cost Tie : Type}.
  Variable cost_order : certificate_order Cost.
  Variable tie_order : certificate_order Tie.
  Variable rank_of : Original -> Cost * Tie.

  Record complete_partition := {
    partition_scope : Scope;
    partition_owned : list Original;
    partition_children : list (list Original);
    partition_live_terminals : list Original;
    partition_private_work : list Original;
    partition_owned_unique : NoDup partition_owned;
    partition_exact_cover : Permutation partition_owned
      (concat partition_children ++
       partition_live_terminals ++ partition_private_work)
  }.

  Record scoped_piece_certificate := {
    piece_scope : Scope;
    piece_claim : rank_certificate Cost Tie
  }.

  Definition piece_rank_region (members : list Original)
      (candidate : Cost * Tie) : Prop :=
    exists original, In original members /\ rank_of original = candidate.

  Definition piece_valid (partition : complete_partition)
      (members : list Original) (certificate : scoped_piece_certificate)
      : Prop :=
    piece_scope certificate = partition_scope partition /\
    certificate_sound cost_order tie_order
      (piece_rank_region members) (piece_claim certificate).

  Definition all_pieces_valid (partition : complete_partition)
      (child_certificates : list scoped_piece_certificate)
      (terminal_certificate private_certificate : scoped_piece_certificate)
      : Prop :=
    Forall2 (piece_valid partition) (partition_children partition)
      child_certificates /\
    piece_valid partition (partition_live_terminals partition)
      terminal_certificate /\
    piece_valid partition (partition_private_work partition)
      private_certificate.

  Definition supports_rank_floor (certificate : scoped_piece_certificate)
      (floor : Cost * Tie) : Prop :=
    certificate_floor cost_order tie_order (piece_claim certificate) floor.

  Definition all_pieces_support_rank_floor
      (child_certificates : list scoped_piece_certificate)
      (terminal_certificate private_certificate : scoped_piece_certificate)
      (floor : Cost * Tie) : Prop :=
    Forall (fun certificate => supports_rank_floor certificate floor)
      child_certificates /\
    supports_rank_floor terminal_certificate floor /\
    supports_rank_floor private_certificate floor.

  Lemma valid_piece_floor_bounds_each_original :
    forall partition members certificate floor original,
      piece_valid partition members certificate ->
      supports_rank_floor certificate floor ->
      In original members ->
      generic_rank_le cost_order tie_order floor (rank_of original).
  Proof.
    intros partition members certificate floor original
      [_ Hsound] Hfloor Hin.
    eapply certificate_interpretation_sound.
    - exact Hsound.
    - exact Hfloor.
    - exists original; split; [exact Hin | reflexivity].
  Qed.

  Lemma child_rank_floors_cover_each_child_original :
    forall partition children certificates floor original,
      Forall2 (piece_valid partition) children certificates ->
      Forall (fun certificate => supports_rank_floor certificate floor)
        certificates ->
      In original (concat children) ->
      generic_rank_le cost_order tie_order floor (rank_of original).
  Proof.
    intros partition children certificates floor original Hpieces.
    induction Hpieces as
      [|members certificate rest certificates Hvalid Hrest IH];
      intros Hfloors Hin; simpl in Hin; [contradiction |].
    inversion Hfloors as [|? ? Hhead Htail]; subst.
    apply in_app_iff in Hin as [Hmember | Hremaining].
    - eapply valid_piece_floor_bounds_each_original; eauto.
    - eapply IH; eauto.
  Qed.

  Theorem complete_partition_rank_floor_aggregation_sound :
    forall partition child_certificates terminal_certificate
      private_certificate floor original,
      all_pieces_valid partition child_certificates
        terminal_certificate private_certificate ->
      all_pieces_support_rank_floor child_certificates
        terminal_certificate private_certificate floor ->
      In original (partition_owned partition) ->
      generic_rank_le cost_order tie_order floor (rank_of original).
  Proof.
    intros partition child_certificates terminal_certificate
      private_certificate floor original
      [Hchildren [Hterminal Hprivate]]
      [Hchild_floors [Hterminal_floor Hprivate_floor]] Howned.
    eapply Permutation_in in Howned;
      [|exact (partition_exact_cover partition)].
    apply in_app_iff in Howned as [Hchild | Hrest].
    - eapply child_rank_floors_cover_each_child_original; eauto.
    - apply in_app_iff in Hrest as [Hlive | Hprivate_original].
      + eapply valid_piece_floor_bounds_each_original;
          [exact Hterminal | exact Hterminal_floor | exact Hlive].
      + eapply valid_piece_floor_bounds_each_original;
          [exact Hprivate | exact Hprivate_floor | exact Hprivate_original].
  Qed.

  (** The strict-cut gate accepts an empty piece, a stronger strict cut,
      or a global/conditioned cost floor already strictly above the cut.
      Unknown supplies no cut. *)
  Definition supports_strict_cut (certificate : scoped_piece_certificate)
      (cut : Cost) : Prop :=
    match piece_claim certificate with
    | RankUnknown => False
    | RankEmpty => True
    | RankGlobal lower _ | RankConditioned lower _ =>
        order_lt cost_order cut lower
    | RankStrictCost lower => order_le cost_order cut lower
    end.

  Lemma strict_cut_support_implies_denoted_cost_above_cut :
    forall certificate cut candidate,
      supports_strict_cut certificate cut ->
      certificate_denotes cost_order tie_order
        (piece_claim certificate) candidate ->
      order_lt cost_order cut (fst candidate).
  Proof.
    intros [scope certificate] cut candidate Hsupport Hdenotes.
    destruct certificate as [| |lower tie|lower tie|lower];
      simpl in Hsupport, Hdenotes; try contradiction.
    - eapply certificate_lt_le_trans; [exact Hsupport | exact (proj1 Hdenotes)].
    - eapply certificate_lt_le_trans; [exact Hsupport | exact (proj1 Hdenotes)].
    - eapply certificate_le_lt_trans; [exact Hsupport | exact Hdenotes].
  Qed.

  Lemma valid_piece_strict_cut_bounds_each_original :
    forall partition members certificate cut original,
      piece_valid partition members certificate ->
      supports_strict_cut certificate cut ->
      In original members ->
      order_lt cost_order cut (fst (rank_of original)).
  Proof.
    intros partition members certificate cut original
      [_ Hsound] Hsupport Hin.
    eapply strict_cut_support_implies_denoted_cost_above_cut;
      [exact Hsupport |].
    apply Hsound.
    exists original; split; [exact Hin | reflexivity].
  Qed.

  Lemma child_strict_cuts_cover_each_child_original :
    forall partition children certificates cut original,
      Forall2 (piece_valid partition) children certificates ->
      Forall (fun certificate => supports_strict_cut certificate cut)
        certificates ->
      In original (concat children) ->
      order_lt cost_order cut (fst (rank_of original)).
  Proof.
    intros partition children certificates cut original Hpieces.
    induction Hpieces as
      [|members certificate rest certificates Hvalid Hrest IH];
      intros Hcuts Hin; simpl in Hin; [contradiction |].
    inversion Hcuts as [|? ? Hhead Htail]; subst.
    apply in_app_iff in Hin as [Hmember | Hremaining].
    - eapply valid_piece_strict_cut_bounds_each_original; eauto.
    - eapply IH; eauto.
  Qed.

  Theorem complete_partition_strict_cut_aggregation_sound :
    forall partition child_certificates terminal_certificate
      private_certificate cut original,
      all_pieces_valid partition child_certificates
        terminal_certificate private_certificate ->
      Forall (fun certificate => supports_strict_cut certificate cut)
        child_certificates ->
      supports_strict_cut terminal_certificate cut ->
      supports_strict_cut private_certificate cut ->
      In original (partition_owned partition) ->
      order_lt cost_order cut (fst (rank_of original)).
  Proof.
    intros partition child_certificates terminal_certificate
      private_certificate cut original
      [Hchildren [Hterminal Hprivate]] Hchild_cuts Hterminal_cut
      Hprivate_cut Howned.
    eapply Permutation_in in Howned;
      [|exact (partition_exact_cover partition)].
    apply in_app_iff in Howned as [Hchild | Hrest].
    - eapply child_strict_cuts_cover_each_child_original; eauto.
    - apply in_app_iff in Hrest as [Hlive | Hprivate_original].
      + eapply valid_piece_strict_cut_bounds_each_original;
          [exact Hterminal | exact Hterminal_cut | exact Hlive].
      + eapply valid_piece_strict_cut_bounds_each_original;
          [exact Hprivate | exact Hprivate_cut | exact Hprivate_original].
  Qed.

  Lemma empty_piece_can_be_skipped_with_proof :
    forall partition certificate floor,
      piece_scope certificate = partition_scope partition ->
      piece_claim certificate = RankEmpty ->
      piece_valid partition [] certificate /\
      supports_rank_floor certificate floor /\
      supports_strict_cut certificate (fst floor).
  Proof.
    intros partition certificate floor Hscope Hempty.
    split.
    - unfold piece_valid; split; [exact Hscope |].
      intros candidate [original [Hin _]]; contradiction.
    - split.
      + unfold supports_rank_floor; rewrite Hempty; exact I.
      + unfold supports_strict_cut; rewrite Hempty; exact I.
  Qed.

  Theorem zero_children_still_require_live_and_private_floors :
    forall partition terminal_certificate private_certificate floor original,
      partition_children partition = [] ->
      all_pieces_valid partition [] terminal_certificate private_certificate ->
      supports_rank_floor terminal_certificate floor ->
      supports_rank_floor private_certificate floor ->
      In original (partition_owned partition) ->
      generic_rank_le cost_order tie_order floor (rank_of original).
  Proof.
    intros partition terminal_certificate private_certificate floor original
      Hzero Hvalid Hterminal Hprivate Howned.
    eapply complete_partition_rank_floor_aggregation_sound;
      [exact Hvalid | |exact Howned].
    split; [constructor | now split].
  Qed.

  Theorem true_unknown_live_piece_blocks_nonempty_union_floor :
    forall partition child_certificates terminal_certificate
      private_certificate floor,
      partition_live_terminals partition <> [] ->
      piece_claim terminal_certificate = RankUnknown ->
      ~ all_pieces_support_rank_floor child_certificates
          terminal_certificate private_certificate floor.
  Proof.
    intros partition child_certificates terminal_certificate
      private_certificate floor _ Hunknown [_ [Hterminal _]].
    unfold supports_rank_floor in Hterminal.
    rewrite Hunknown in Hterminal; exact Hterminal.
  Qed.
End ScopedUnion.

Module MissingLiveTerminalControl.
  Definition rank_of (original : nat) : nat * nat :=
    if Nat.eqb original 0 then (5, 1) else (5, 9).

  Definition partition : @complete_partition nat nat.
  Proof.
    refine {| partition_scope := 3;
              partition_owned := [1; 0];
              partition_children := [[1]];
              partition_live_terminals := [0];
              partition_private_work := [] |}.
    - constructor.
      + simpl; intros [Heq | Hfalse]; [discriminate | contradiction].
      + constructor; [simpl; tauto | constructor].
    - simpl; apply Permutation_refl.
  Defined.

  Definition child_certificate : @scoped_piece_certificate nat nat nat :=
    {| piece_scope := 3;
       piece_claim := RankConditioned 5 9 |}.

  Definition live_certificate : @scoped_piece_certificate nat nat nat :=
    {| piece_scope := 3;
       piece_claim := RankUnknown |}.

  Definition private_certificate : @scoped_piece_certificate nat nat nat :=
    {| piece_scope := 3;
       piece_claim := RankEmpty |}.

  Example control_partition_certificates_are_valid :
    all_pieces_valid natural_certificate_order natural_certificate_order
      rank_of partition [child_certificate] live_certificate
      private_certificate.
  Proof.
    unfold all_pieces_valid.
    cbn [partition_children partition_live_terminals partition_private_work].
    split.
    - constructor.
      + unfold piece_valid; split; [reflexivity |].
        intros candidate [original [Hin Hr]].
        simpl in Hin.
        destruct Hin as [Heq | []].
        subst original; rewrite <- Hr.
        unfold rank_of; simpl.
        split; [lia | intros; lia].
      + constructor.
    - split.
      + unfold piece_valid; split; [reflexivity |].
        intros candidate _; exact I.
      + unfold piece_valid; split; [reflexivity |].
        intros candidate [original [Hin _]].
        simpl in Hin; contradiction.
  Qed.

  Example omitted_early_tie_live_terminal_refutes_child_only_floor :
    In 0 (partition_owned partition) /\
    supports_rank_floor natural_certificate_order natural_certificate_order
      child_certificate (5, 7) /\
    ~ generic_rank_le natural_certificate_order natural_certificate_order
      (5, 7) (rank_of 0) /\
    ~ all_pieces_support_rank_floor natural_certificate_order
        natural_certificate_order [child_certificate]
        live_certificate private_certificate (5, 7).
  Proof.
    split; [simpl; auto |].
    split.
    - change (5 < 5 \/ (5 = 5 /\ 7 <= 9)).
      right; split; lia.
    - split.
      + unfold rank_of, generic_rank_le; simpl; lia.
      + intros [_ [Hlive _]].
        exact Hlive.
  Qed.
End MissingLiveTerminalControl.
