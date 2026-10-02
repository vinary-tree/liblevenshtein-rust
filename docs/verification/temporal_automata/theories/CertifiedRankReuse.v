(** * Scoped reuse of per-original threshold rank floors

    This finite exact-natural model reuses a floor over S_Q(c), the set of
    enumerated originals whose per-original lower bound is at most the
    inclusive cutoff. With a fixed bound function, lowering the cutoff
    shrinks S_Q(c), so a floor valid on the old set remains valid. If a
    bound function changes, reuse requires a separate inclusion proof.

    The score-certificate checker binds stable request tokens (query,
    parameters, snapshot/revision, arithmetic, observation, and labels),
    but it does not itself certify this rank floor or an actual Rust search
    region. The two facts are joined only under the explicit premises below.
    Per-original soundness, complete source enumeration, live tie identity,
    binary64 behavior, and executable producer correspondence remain
    separate obligations. No resource or asymptotic claim follows here. *)

From Stdlib Require Import Arith Bool Lia List.
From Liblevenshtein.TemporalAutomata Require Import
  CertifiedRankCertificateTypes CertifiedThresholdCoverage
  CertificateScope CertifiedCertificateScopeReuse.
Import ListNotations.

Definition candidates_for (all : list original_occurrence)
    (bound : original_occurrence -> nat) (cutoff : nat)
    : list original_occurrence :=
  filter (fun original => Nat.leb (bound original) cutoff) all.

Definition bound_function_sound (all : list original_occurrence)
    (bound : original_occurrence -> nat) : Prop :=
  forall original, In original all ->
    bound original <= original_exact_cost original.

Definition candidate_rank (original : original_occurrence) : nat * nat :=
  (original_exact_cost original, original_tie original).

Definition rank_floor_for (all : list original_occurrence)
    (bound : original_occurrence -> nat) (cutoff : nat)
    (floor : nat * nat) : Prop :=
  forall original, In original (candidates_for all bound cutoff) ->
    generic_rank_le natural_certificate_order natural_certificate_order
      floor (candidate_rank original).

Lemma candidate_membership_iff : forall all bound cutoff original,
  In original (candidates_for all bound cutoff) <->
  In original all /\ bound original <= cutoff.
Proof.
  intros; unfold candidates_for.
  rewrite filter_In, Nat.leb_le; reflexivity.
Qed.

Theorem decreasing_cutoff_shrinks_candidates :
  forall all bound new_cutoff old_cutoff,
    new_cutoff <= old_cutoff ->
    incl (candidates_for all bound new_cutoff)
      (candidates_for all bound old_cutoff).
Proof.
  intros all bound new_cutoff old_cutoff Hcut original Hin.
  apply candidate_membership_iff in Hin as [Hall Hbound].
  apply candidate_membership_iff; split; [exact Hall | lia].
Qed.

Corollary decreasing_cutoff_shrinks_s_q :
  forall all new_cutoff old_cutoff,
    new_cutoff <= old_cutoff ->
    incl (search_candidates all new_cutoff)
      (search_candidates all old_cutoff).
Proof.
  intros all new_cutoff old_cutoff Hcut.
  exact (decreasing_cutoff_shrinks_candidates all original_lower_bound
    new_cutoff old_cutoff Hcut).
Qed.

Lemma sound_bounds_cover_exact_eligible_originals :
  forall all bound cutoff original,
    bound_function_sound all bound ->
    In original all ->
    original_exact_cost original <= cutoff ->
    In original (candidates_for all bound cutoff).
Proof.
  intros all bound cutoff original Hsound Hall Hcost.
  apply candidate_membership_iff; split; [exact Hall |].
  pose proof (Hsound original Hall); lia.
Qed.

Theorem decreasing_cutoff_preserves_rank_floor :
  forall all bound new_cutoff old_cutoff floor,
    new_cutoff <= old_cutoff ->
    rank_floor_for all bound old_cutoff floor ->
    rank_floor_for all bound new_cutoff floor.
Proof.
  intros all bound new_cutoff old_cutoff floor Hcut Hfloor
    original Hin.
  apply Hfloor.
  eapply decreasing_cutoff_shrinks_candidates; eauto.
Qed.

(** The inclusion premise is indispensable when the bound producer changes.
    Separate soundness proofs alone do not establish this inclusion. *)
Theorem changed_bounds_reuse_requires_candidate_inclusion :
  forall all old_bound new_bound old_cutoff new_cutoff floor,
    incl (candidates_for all new_bound new_cutoff)
      (candidates_for all old_bound old_cutoff) ->
    rank_floor_for all old_bound old_cutoff floor ->
    rank_floor_for all new_bound new_cutoff floor.
Proof.
  intros all old_bound new_bound old_cutoff new_cutoff floor
    Hinclusion Hfloor original Hin.
  apply Hfloor, Hinclusion; exact Hin.
Qed.

Record scoped_rank_floor_claim := {
  claim_scope : stable_request;
  claim_originals : list original_occurrence;
  claim_bound : original_occurrence -> nat;
  claim_cutoff : nat;
  claim_floor : nat * nat;
  claim_valid : rank_floor_for claim_originals claim_bound
    claim_cutoff claim_floor
}.

Theorem lower_cutoff_reuses_scoped_rank_floor :
  forall (old : scoped_rank_floor_claim) new_cutoff,
    new_cutoff <= claim_cutoff old ->
    exists newer : scoped_rank_floor_claim,
      claim_scope newer = claim_scope old /\
      claim_originals newer = claim_originals old /\
      claim_bound newer = claim_bound old /\
      claim_cutoff newer = new_cutoff /\
      claim_floor newer = claim_floor old.
Proof.
  intros old new_cutoff Hcut.
  unshelve eexists {| claim_scope := claim_scope old;
    claim_originals := claim_originals old;
    claim_bound := claim_bound old;
    claim_cutoff := new_cutoff;
    claim_floor := claim_floor old |}.
  - eapply decreasing_cutoff_preserves_rank_floor;
      [exact Hcut | exact (claim_valid old)].
  - repeat split; reflexivity.
Qed.

(** A checked score relation establishes stable identity. The rank fact
    remains a separate, already proved premise. The captured cutoff must
    match that fact, and only a decrease is transferred automatically. *)
Theorem accepted_score_scope_and_decrease_reuse_rank_floor :
  forall requested lookup_key origin target kind packet
    (old : scoped_rank_floor_claim),
    check_certificate_reuse CutoffIndependentScoreReuse requested
      lookup_key origin target kind packet = true ->
    claim_scope old = stable_part (captured_request packet) ->
    claim_cutoff old = bound_cutoff (captured_request packet) ->
    bound_cutoff requested <= bound_cutoff (captured_request packet) ->
    claim_scope old = stable_part requested /\
    rank_floor_for (claim_originals old) (claim_bound old)
      (bound_cutoff requested) (claim_floor old).
Proof.
  intros requested lookup_key origin target kind packet old Haccept
    Hscope Hcutoff Hdecrease.
  destruct (accepted_reuse_has_stable_scope_and_score_relation
    _ _ _ _ _ _ _ Haccept) as [Hstable _].
  split.
  - rewrite Hscope; symmetry; exact Hstable.
  - eapply decreasing_cutoff_preserves_rank_floor
      with (old_cutoff := claim_cutoff old).
    + rewrite Hcutoff; exact Hdecrease.
    + exact (claim_valid old).
Qed.

(** Equality of the stable request covers all named fields, rather than
    relying on a lookup-key collision. *)
Corollary accepted_rank_reuse_binds_query_snapshot_and_profiles :
  forall requested lookup_key origin target kind packet,
    check_certificate_reuse CutoffIndependentScoreReuse requested
      lookup_key origin target kind packet = true ->
    let new := stable_part requested in
    let old := stable_part (captured_request packet) in
    bound_query new = bound_query old /\
    bound_snapshot new = bound_snapshot old /\
    bound_revision new = bound_revision old /\
    bound_arithmetic new = bound_arithmetic old /\
    bound_observation new = bound_observation old.
Proof.
  intros requested lookup_key origin target kind packet Haccept.
  destruct (accepted_reuse_has_stable_scope_and_score_relation
    _ _ _ _ _ _ _ Haccept) as [Hstable _].
  simpl; repeat split; now rewrite Hstable.
Qed.

Theorem changed_rank_scope_rejects_score_reuse :
  forall requested lookup_key origin target kind packet,
    bound_query (stable_part requested) <>
      bound_query (stable_part (captured_request packet)) \/
    bound_snapshot (stable_part requested) <>
      bound_snapshot (stable_part (captured_request packet)) \/
    bound_revision (stable_part requested) <>
      bound_revision (stable_part (captured_request packet)) \/
    bound_arithmetic (stable_part requested) <>
      bound_arithmetic (stable_part (captured_request packet)) \/
    bound_observation (stable_part requested) <>
      bound_observation (stable_part (captured_request packet)) ->
    check_certificate_reuse CutoffIndependentScoreReuse requested
      lookup_key origin target kind packet = false.
Proof.
  intros requested lookup_key origin target kind packet Hchange.
  destruct Hchange as [Hquery | [Hsnapshot | [Hrevision |
    [Harithmetic | Hobservation]]]].
  - eapply changed_stable_projection_rejects_reuse
      with (project := bound_query); exact Hquery.
  - eapply changed_stable_projection_rejects_reuse
      with (project := bound_snapshot); exact Hsnapshot.
  - eapply changed_stable_projection_rejects_reuse
      with (project := bound_revision); exact Hrevision.
  - eapply changed_stable_projection_rejects_reuse
      with (project := bound_arithmetic); exact Harithmetic.
  - eapply changed_stable_projection_rejects_reuse
      with (project := bound_observation); exact Hobservation.
Qed.

Definition later_tie_original : original_occurrence :=
  {| original_bucket := 0; original_slot := 0;
     original_exact_cost := 5; original_lower_bound := 1;
     original_tie := 9 |}.

Definition earlier_tie_original : original_occurrence :=
  {| original_bucket := 0; original_slot := 1;
     original_exact_cost := 5; original_lower_bound := 2;
     original_tie := 1 |}.

(** Both originals are in one bucket and have sound per-original bounds.
    Raising the inclusive threshold from 1 to 2 admits the earlier tie and
    invalidates the old floor (5,9). *)
Example increasing_cutoff_can_invalidate_tie_floor :
  per_original_bounds_sound [later_tie_original; earlier_tie_original] /\
  rank_floor_for [later_tie_original; earlier_tie_original]
    original_lower_bound 1 (5, 9) /\
  In earlier_tie_original
    (search_candidates [later_tie_original; earlier_tie_original] 2) /\
  ~ rank_floor_for [later_tie_original; earlier_tie_original]
      original_lower_bound 2 (5, 9).
Proof.
  split.
  - intros original Hin; simpl in Hin.
    destruct Hin as [Heq | [Heq | []]]; subst original; simpl; lia.
  - split.
    + intros original Hin.
      apply candidate_membership_iff in Hin as [Hall Hbound].
      simpl in Hall.
      destruct Hall as [Heq | [Heq | []]]; subst original;
        simpl in Hbound |- *.
      * apply generic_rank_le_reflexive.
      * lia.
    + split.
      * simpl; auto.
      * intros Hfloor.
        specialize (Hfloor earlier_tie_original).
        assert (Hin : In earlier_tie_original
          (candidates_for [later_tie_original; earlier_tie_original]
            original_lower_bound 2)).
        { apply candidate_membership_iff; simpl; split; [auto | lia]. }
        specialize (Hfloor Hin).
        change (5 < 5 \/ (5 = 5 /\ 9 <= 1)) in Hfloor.
        lia.
Qed.
