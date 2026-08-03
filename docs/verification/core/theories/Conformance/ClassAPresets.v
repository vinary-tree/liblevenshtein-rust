(** Assumption-free Class-A preset invariants.

    These results isolate the algebra mirrored by the Rust property suite:
    coordinate mismatch laws for Hamming; reversible and composable unit-cost
    insertion/deletion scripts; exact directional skip cost; and prefix-safe
    aggregation for OperationSet validation. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Definition mismatch (left right : nat) : nat :=
  if Nat.eq_dec left right then 0 else 1.

Lemma mismatch_identity : forall value, mismatch value value = 0.
Proof.
  intros value. unfold mismatch. destruct (Nat.eq_dec value value); congruence.
Qed.

Lemma mismatch_symmetric : forall left right,
  mismatch left right = mismatch right left.
Proof.
  intros left right. unfold mismatch.
  destruct (Nat.eq_dec left right); destruct (Nat.eq_dec right left); congruence.
Qed.

Lemma mismatch_triangle : forall left middle right,
  mismatch left right <= mismatch left middle + mismatch middle right.
Proof.
  intros left middle right. unfold mismatch.
  destruct (Nat.eq_dec left right);
  destruct (Nat.eq_dec left middle);
  destruct (Nat.eq_dec middle right); subst; simpl; lia.
Qed.

Fixpoint hamming_fold (left right : list nat) : nat :=
  match left, right with
  | left_head :: left_tail, right_head :: right_tail =>
      mismatch left_head right_head + hamming_fold left_tail right_tail
  | _, _ => 0
  end.

Lemma hamming_identity : forall values, hamming_fold values values = 0.
Proof.
  induction values as [|head tail IH]; simpl; [reflexivity |].
  rewrite mismatch_identity, IH. lia.
Qed.

Lemma hamming_symmetric : forall left right,
  hamming_fold left right = hamming_fold right left.
Proof.
  induction left as [|left_head left_tail IH]; destruct right as [|right_head right_tail];
    simpl; try reflexivity.
  rewrite mismatch_symmetric, IH. reflexivity.
Qed.

Theorem hamming_triangle : forall left middle right,
  length left = length middle ->
  length middle = length right ->
  hamming_fold left right <=
    hamming_fold left middle + hamming_fold middle right.
Proof.
  induction left as [|left_head left_tail IH];
    destruct middle as [|middle_head middle_tail];
    destruct right as [|right_head right_tail]; simpl; intros Hlm Hmr;
    try discriminate; try lia.
  specialize (IH middle_tail right_tail ltac:(lia) ltac:(lia)).
  pose proof (mismatch_triangle left_head middle_head right_head).
  lia.
Qed.

Inductive indel_step : Type :=
| Keep
| Insert
| Delete.

Definition step_source (step : indel_step) : nat :=
  match step with Keep | Delete => 1 | Insert => 0 end.

Definition step_target (step : indel_step) : nat :=
  match step with Keep | Insert => 1 | Delete => 0 end.

Definition step_cost (step : indel_step) : nat :=
  match step with Keep => 0 | Insert | Delete => 1 end.

Fixpoint script_source (script : list indel_step) : nat :=
  match script with [] => 0 | step :: rest => step_source step + script_source rest end.

Fixpoint script_target (script : list indel_step) : nat :=
  match script with [] => 0 | step :: rest => step_target step + script_target rest end.

Fixpoint script_cost (script : list indel_step) : nat :=
  match script with [] => 0 | step :: rest => step_cost step + script_cost rest end.

Lemma script_source_app : forall left right,
  script_source (left ++ right) = script_source left + script_source right.
Proof.
  induction left as [|step rest IH]; intros right; simpl; [lia | rewrite IH; lia].
Qed.

Lemma script_target_app : forall left right,
  script_target (left ++ right) = script_target left + script_target right.
Proof.
  induction left as [|step rest IH]; intros right; simpl; [lia | rewrite IH; lia].
Qed.

Lemma script_cost_app : forall left right,
  script_cost (left ++ right) = script_cost left + script_cost right.
Proof.
  induction left as [|step rest IH]; intros right; simpl; [lia | rewrite IH; lia].
Qed.

Definition inverse_step (step : indel_step) : indel_step :=
  match step with Keep => Keep | Insert => Delete | Delete => Insert end.

Lemma inverse_step_source : forall step,
  step_source (inverse_step step) = step_target step.
Proof. intros []; reflexivity. Qed.

Lemma inverse_step_target : forall step,
  step_target (inverse_step step) = step_source step.
Proof. intros []; reflexivity. Qed.

Lemma inverse_step_cost : forall step,
  step_cost (inverse_step step) = step_cost step.
Proof. intros []; reflexivity. Qed.

Definition reverse_script (script : list indel_step) : list indel_step :=
  map inverse_step (rev script).

Theorem reverse_script_preserves_cost : forall script,
  script_cost (reverse_script script) = script_cost script.
Proof.
  induction script as [|step rest IH]; simpl; [reflexivity |].
  assert (Hrev : rev (step :: rest) = rev rest ++ [step]).
  { change (rev ([step] ++ rest) = rev rest ++ [step]).
    rewrite rev_app_distr. reflexivity. }
  unfold reverse_script in *. rewrite Hrev, map_app, script_cost_app.
  simpl. rewrite IH, inverse_step_cost. lia.
Qed.

Theorem reverse_script_swaps_consumption : forall script,
  script_source (reverse_script script) = script_target script /\
  script_target (reverse_script script) = script_source script.
Proof.
  induction script as [|step rest [IHsource IHtarget]]; simpl.
  - split; reflexivity.
  - assert (Hrev : rev (step :: rest) = rev rest ++ [step]).
    { change (rev ([step] ++ rest) = rev rest ++ [step]).
      rewrite rev_app_distr. reflexivity. }
    unfold reverse_script in *. rewrite Hrev, map_app.
    rewrite script_source_app, script_target_app. simpl.
    rewrite IHsource, IHtarget, inverse_step_source, inverse_step_target. lia.
Qed.

Lemma indel_length_bounds_linear : forall script,
  script_source script <= script_target script + script_cost script /\
  script_target script <= script_source script + script_cost script.
Proof.
  induction script as [|step rest [IHsource IHtarget]]; simpl; [lia |].
  destruct step; simpl in *; lia.
Qed.

Theorem indel_length_lower_bounds : forall script,
  script_source script - script_target script <= script_cost script /\
  script_target script - script_source script <= script_cost script.
Proof.
  intros script. pose proof (indel_length_bounds_linear script) as [Hsource Htarget].
  rewrite Nat.le_sub_le_add_l, Nat.le_sub_le_add_l. split; assumption.
Qed.

Theorem concatenating_indel_scripts_adds_cost : forall first second,
  script_cost (first ++ second) = script_cost first + script_cost second.
Proof. exact script_cost_app. Qed.

Inductive skip_step : Type := SkipMatch | SkipSource.

Definition skip_source (step : skip_step) : nat := 1.
Definition skip_target (step : skip_step) : nat :=
  match step with SkipMatch => 1 | SkipSource => 0 end.
Definition skip_cost (step : skip_step) : nat :=
  match step with SkipMatch => 0 | SkipSource => 1 end.

Fixpoint skip_path_source (path : list skip_step) : nat :=
  match path with [] => 0 | step :: rest => skip_source step + skip_path_source rest end.
Fixpoint skip_path_target (path : list skip_step) : nat :=
  match path with [] => 0 | step :: rest => skip_target step + skip_path_target rest end.
Fixpoint skip_path_cost (path : list skip_step) : nat :=
  match path with [] => 0 | step :: rest => skip_cost step + skip_path_cost rest end.

Theorem bounded_skip_exact_length_difference : forall path,
  skip_path_source path = skip_path_target path + skip_path_cost path.
Proof.
  induction path as [|step rest IH]; simpl; [lia |]. destruct step; simpl in *; lia.
Qed.

Record declared_operation : Type := {
  declared_source : nat;
  declared_target : nat
}.

Definition declared_consumption (operation : declared_operation) : nat :=
  declared_source operation + declared_target operation.

Fixpoint aggregate_consumption (operations : list declared_operation) : nat :=
  match operations with
  | [] => 0
  | operation :: rest => declared_consumption operation + aggregate_consumption rest
  end.

Lemma aggregate_consumption_app : forall prefix suffix,
  aggregate_consumption (prefix ++ suffix) =
  aggregate_consumption prefix + aggregate_consumption suffix.
Proof.
  induction prefix as [|operation rest IH]; intros suffix; simpl; [lia | rewrite IH; lia].
Qed.

Theorem validated_total_bounds_every_prefix : forall prefix suffix limit,
  aggregate_consumption (prefix ++ suffix) <= limit ->
  aggregate_consumption prefix <= limit.
Proof.
  intros prefix suffix limit Hbound. rewrite aggregate_consumption_app in Hbound. lia.
Qed.

Theorem progressing_operation_advances_grid : forall operation,
  declared_consumption operation > 0 ->
  declared_source operation > 0 \/ declared_target operation > 0.
Proof. intros operation Hprogress. unfold declared_consumption in Hprogress. lia. Qed.
