(** * fzf local-alignment upper-bound conformance

    fzf may start an alignment after an arbitrary dictionary prefix. A sound
    branch-and-bound value therefore retains active and unstarted alternatives.
*)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition fzf_bound (unstarted active : Z) : Z := Z.max unstarted active.

Theorem unstarted_alternative_is_retained : forall unstarted active,
  unstarted <= fzf_bound unstarted active.
Proof. intros; unfold fzf_bound; apply Z.le_max_l. Qed.

Theorem active_alternative_is_retained : forall unstarted active,
  active <= fzf_bound unstarted active.
Proof. intros; unfold fzf_bound; apply Z.le_max_r. Qed.

Theorem every_live_alignment_is_bounded : forall exact unstarted active,
  exact <= unstarted \/ exact <= active ->
  exact <= fzf_bound unstarted active.
Proof.
  intros exact unstarted active [Hunstarted | Hactive].
  - eapply Z.le_trans; [exact Hunstarted | apply unstarted_alternative_is_retained].
  - eapply Z.le_trans; [exact Hactive | apply active_alternative_is_retained].
Qed.

Theorem branch_and_bound_prune_is_sound : forall exact unstarted active cutoff,
  (exact <= unstarted \/ exact <= active) ->
  fzf_bound unstarted active < cutoff -> exact < cutoff.
Proof.
  intros. pose proof (every_live_alignment_is_bounded _ _ _ H). lia.
Qed.

Example active_only_bound_is_unsound :
  let active := 10 in let unstarted := 20 in let exact := 20 in
  active < exact /\ exact <= fzf_bound unstarted active.
Proof. simpl; split; reflexivity. Qed.

Theorem fzf_arc_delta_telescope : forall initial middle final,
  initial + (middle - initial) + (final - middle) = final.
Proof. intros; lia. Qed.

Theorem top_k_cutoff_monotone : forall old next,
  old <= next -> Z.max old next = next.
Proof. intros; apply Z.max_r; assumption. Qed.
