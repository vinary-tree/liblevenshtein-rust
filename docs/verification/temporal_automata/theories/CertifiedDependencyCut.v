(** * Complete dependency cuts for nonnegative recurrence search

    A represented live cut intersects every unresolved accepting derivation.
    Expansion may remove a live dependency only after it enumerates every
    semantic successor; a resolved accepting endpoint is recorded separately.
    The graph's nodes may be cells from different retained generations or
    pending multi-label operations.  This is a finite-path theorem, not a
    proof that a particular Rust recurrence enumerates its successors. *)

From Stdlib Require Import Arith Lia List.
Import ListNotations.

Inductive semantic_path {V : Type} (edge : V -> V -> Prop)
    : V -> V -> list V -> Prop :=
| PathAt : forall x, semantic_path edge x x [x]
| PathThen : forall x y z tail,
    edge x y -> semantic_path edge y z (y :: tail) ->
    semantic_path edge x z (x :: y :: tail).

Lemma path_starts_in : forall (V : Type) (edge : V -> V -> Prop)
    start finish path,
  semantic_path edge start finish path -> In start path.
Proof.
  intros V edge start finish path Hpath.
  induction Hpath; simpl; auto.
Qed.

Lemma path_hit_has_successor : forall (V : Type) (edge : V -> V -> Prop)
    start finish path node,
  semantic_path edge start finish path -> In node path ->
  node = finish \/
  exists next, edge node next /\ In next path.
Proof.
  intros V edge start finish path node Hpath.
  induction Hpath as [x | x y z tail Hedge Htail IH]; intros Hin.
  - simpl in Hin. destruct Hin as [Heq | []]. now left.
  - simpl in Hin. destruct Hin as [Heq | Hin].
    + subst x. right. exists y; split; [exact Hedge | simpl; auto].
    + destruct (IH Hin) as [Heq | [next [Hedge' Hnext]]].
      * now left.
      * right. exists next; split; [exact Hedge' | simpl; auto].
Qed.

Definition cut_covers {V : Type} (edge : V -> V -> Prop)
    (root : V) (accepting : V -> Prop)
    (live resolved : list V) : Prop :=
  forall finish path,
    semantic_path edge root finish path -> accepting finish ->
    In finish resolved \/
    exists node, In node path /\ In node live.

Definition resolved_sound {V : Type} (accepting proven : V -> Prop)
    (resolved : list V) : Prop :=
  forall node, In node resolved -> accepting node -> proven node.

Definition cut_invariant {V : Type} (edge : V -> V -> Prop)
    (root : V) (accepting proven : V -> Prop)
    (live resolved : list V) : Prop :=
  cut_covers edge root accepting live resolved /\
  resolved_sound accepting proven resolved.

Lemma in_remove_other : forall (V : Type)
    (equal : forall a b : V, {a = b} + {a <> b})
    removed member values,
  member <> removed -> In member values ->
  In member (remove equal removed values).
Proof.
  intros V equal removed member values Hother Hin.
  induction values as [|head tail IH]; simpl in *; [contradiction |].
  destruct Hin as [Heq | Hin].
  - subst head. destruct (equal removed member) as [Heq | Hneq].
    + congruence.
    + simpl; auto.
  - destruct (equal removed head); simpl; auto.
Qed.

Theorem initial_cut_invariant : forall (V : Type)
    (edge : V -> V -> Prop) (root : V) (accepting proven : V -> Prop),
  cut_invariant edge root accepting proven [root] [].
Proof.
  intros V edge root accepting proven.
  split.
  - intros finish path Hpath _. right. exists root.
    split; [eapply path_starts_in; eauto | simpl; auto].
  - intros node Hin. contradiction.
Qed.

Theorem expansion_preserves_cut : forall (V : Type)
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (root : V) (accepting proven : V -> Prop)
    live resolved expanded,
  (forall next, edge expanded next -> In next (successors expanded)) ->
  In expanded live ->
  (accepting expanded -> proven expanded) ->
  cut_invariant edge root accepting proven live resolved ->
  cut_invariant edge root accepting proven
    (remove equal expanded live ++ successors expanded)
    (expanded :: resolved).
Proof.
  intros V equal edge successors root accepting proven live resolved
    expanded Hcomplete _ Hresolved [Hcovers Hsound].
  split.
  - intros finish path Hpath Haccept.
    destruct (Hcovers finish path Hpath Haccept)
      as [Hdone | [node [Hinpath Hinlive]]].
    + left; simpl; auto.
    + destruct (equal node expanded) as [Heq | Hneq].
      * subst node.
        destruct (path_hit_has_successor V edge root finish path
          expanded Hpath Hinpath) as [Heq | [next [Hedge Hnext]]].
        -- left; simpl; auto.
        -- right; exists next; split; [exact Hnext |].
           apply in_or_app. right. apply Hcomplete. exact Hedge.
      * right; exists node; split; [exact Hinpath |].
        apply in_or_app. left.
        eapply in_remove_other; eauto.
  - intros node Hin Haccept.
    simpl in Hin. destruct Hin as [Heq | Hin].
    + subst node. now apply Hresolved.
    + eapply Hsound; eauto.
Qed.

Inductive cut_step {V : Type}
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (accepting proven : V -> Prop)
    : (list V * list V) -> (list V * list V) -> Prop :=
| CutExpand : forall live resolved node,
    In node live ->
    (forall next, edge node next -> In next (successors node)) ->
    (accepting node -> proven node) ->
    cut_step equal edge successors accepting proven
      (live, resolved)
      (remove equal node live ++ successors node, node :: resolved).

Inductive cut_steps {V : Type}
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (accepting proven : V -> Prop)
    : (list V * list V) -> (list V * list V) -> Prop :=
| CutDone : forall state,
    cut_steps equal edge successors accepting proven state state
| CutMore : forall first middle last,
    cut_step equal edge successors accepting proven first middle ->
    cut_steps equal edge successors accepting proven middle last ->
    cut_steps equal edge successors accepting proven first last.

Theorem cut_step_preserves_invariant : forall (V : Type)
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (root : V) (accepting proven : V -> Prop) first last,
  cut_step equal edge successors accepting proven first last ->
  cut_invariant edge root accepting proven (fst first) (snd first) ->
  cut_invariant edge root accepting proven (fst last) (snd last).
Proof.
  intros V equal edge successors root accepting proven
    [live resolved] [live' resolved'] Hstep Hinv.
  inversion Hstep; subst; simpl in *.
  eapply expansion_preserves_cut; eauto.
Qed.

Theorem cut_steps_preserve_invariant : forall (V : Type)
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (root : V) (accepting proven : V -> Prop) first last,
  cut_steps equal edge successors accepting proven first last ->
  cut_invariant edge root accepting proven (fst first) (snd first) ->
  cut_invariant edge root accepting proven (fst last) (snd last).
Proof.
  intros V equal edge successors root accepting proven first last
    Hsteps Hinv.
  induction Hsteps as [state | first middle last Hstep Hsteps IH].
  - exact Hinv.
  - apply IH. eapply cut_step_preserves_invariant; eauto.
Qed.

Corollary reachable_cut_invariant : forall (V : Type)
    (equal : forall a b : V, {a = b} + {a <> b})
    (edge : V -> V -> Prop) (successors : V -> list V)
    (root : V) (accepting proven : V -> Prop) last,
  cut_steps equal edge successors accepting proven ([root], []) last ->
  cut_invariant edge root accepting proven (fst last) (snd last).
Proof.
  intros V equal edge successors root accepting proven last Hsteps.
  eapply cut_steps_preserve_invariant; eauto.
  apply initial_cut_invariant.
Qed.

Lemma path_cost_monotone : forall (V : Type)
    (edge : V -> V -> Prop) (cost : V -> nat),
  (forall from to, edge from to -> cost from <= cost to) ->
  forall start finish path node,
    semantic_path edge start finish path ->
    In node path -> cost node <= cost finish.
Proof.
  intros V edge cost Hedge start finish path node Hpath.
  revert node.
  induction Hpath as [x | x y z tail Hxy Htail IH]; intros node Hin.
  - simpl in Hin. destruct Hin as [Heq | []]. subst; lia.
  - simpl in Hin. destruct Hin as [Heq | Hin].
    + subst node. specialize (IH y (or_introl eq_refl)).
      specialize (Hedge x y Hxy). lia.
    + now apply IH.
Qed.

Fixpoint cut_floor {V : Type} (cost : V -> nat)
    (live : list V) : option nat :=
  match live with
  | [] => None
  | node :: tail =>
      match cut_floor cost tail with
      | None => Some (cost node)
      | Some floor => Some (Nat.min (cost node) floor)
      end
  end.

Lemma cut_floor_le_member : forall (V : Type) (cost : V -> nat)
    live node,
  In node live ->
  exists floor, cut_floor cost live = Some floor /\ floor <= cost node.
Proof.
  intros V cost live.
  induction live as [|head tail IH]; intros node Hin;
    [contradiction |].
  simpl in Hin. destruct Hin as [Heq | Hin].
  - subst node. simpl. destruct (cut_floor cost tail) as [floor |].
    + exists (Nat.min (cost head) floor); split; [reflexivity | lia].
    + exists (cost head); split; [reflexivity | lia].
  - destruct (IH node Hin) as [floor [Hfloor Hle]].
    simpl. rewrite Hfloor.
    exists (Nat.min (cost head) floor); split; [reflexivity | lia].
Qed.

Theorem complete_cut_bounds_unresolved_completion :
  forall (V : Type) (edge : V -> V -> Prop) (cost : V -> nat)
    (root : V) (accepting : V -> Prop) live resolved finish path,
  (forall from to, edge from to -> cost from <= cost to) ->
  cut_covers edge root accepting live resolved ->
  semantic_path edge root finish path ->
  accepting finish -> ~ In finish resolved ->
  exists floor,
    cut_floor cost live = Some floor /\ floor <= cost finish.
Proof.
  intros V edge cost root accepting live resolved finish path
    Hmono Hcovers Hpath Haccept Hopen.
  destruct (Hcovers finish path Hpath Haccept)
    as [Hdone | [node [Hinpath Hinlive]]]; [contradiction |].
  destruct (cut_floor_le_member V cost live node Hinlive)
    as [floor [Hfloor Hle]].
  exists floor; split; [exact Hfloor |].
  pose proof (path_cost_monotone V edge cost Hmono root finish path
    node Hpath Hinpath) as Hpathcost.
  lia.
Qed.

Corollary strict_cut_pruning_is_sound :
  forall (V : Type) (edge : V -> V -> Prop) (cost : V -> nat)
    (root : V) (accepting : V -> Prop)
    live resolved cutoff finish path,
  (forall from to, edge from to -> cost from <= cost to) ->
  cut_covers edge root accepting live resolved ->
  (forall node, In node live -> cutoff < cost node) ->
  semantic_path edge root finish path -> accepting finish ->
  ~ In finish resolved -> cutoff < cost finish.
Proof.
  intros V edge cost root accepting live resolved cutoff finish path
    Hmono Hcovers Hstrict Hpath Haccept Hopen.
  destruct (Hcovers finish path Hpath Haccept)
    as [Hdone | [node [Hinpath Hinlive]]]; [contradiction |].
  pose proof (Hstrict node Hinlive) as Hnode.
  pose proof (path_cost_monotone V edge cost Hmono root finish path
    node Hpath Hinpath) as Hend.
  lia.
Qed.

(** The finite control has an operation that consumes two target generations.
    It bypasses [CurrentRow], so that row alone is not a dependency cut. *)
Module BypassControl.
  Inductive vertex := Seed | CurrentRow | PendingMacro | SlowAccept | FastAccept.

  Definition equal (left right : vertex) : {left = right} + {left <> right}.
  Proof. decide equality. Defined.

  Definition generation (node : vertex) : nat :=
    match node with
    | Seed => 0 | CurrentRow => 1 | PendingMacro => 2
    | SlowAccept | FastAccept => 3
    end.

  Definition semantic_successors (node : vertex) : list vertex :=
    match node with
    | Seed => [CurrentRow; PendingMacro]
    | CurrentRow => [SlowAccept]
    | PendingMacro => [FastAccept]
    | SlowAccept | FastAccept => []
    end.

  Definition row_only_successors (node : vertex) : list vertex :=
    match node with
    | Seed => [CurrentRow]
    | _ => semantic_successors node
    end.

  Definition edge (from to : vertex) : Prop :=
    In to (semantic_successors from).

  Definition accepting (node : vertex) : Prop :=
    node = SlowAccept \/ node = FastAccept.

  Definition proven (node : vertex) : Prop := False.

  Definition cost (node : vertex) : nat :=
    match node with
    | Seed => 0 | CurrentRow => 6 | PendingMacro => 1
    | SlowAccept => 7 | FastAccept => 4
    end.

  Lemma successors_complete : forall from to,
    edge from to -> In to (semantic_successors from).
  Proof. auto. Qed.

  Example row_only_enumerator_omits_macro_edge :
    ~ (forall next, edge Seed next ->
        In next (row_only_successors Seed)).
  Proof.
    intro Hcomplete.
    specialize (Hcomplete PendingMacro).
    assert (Hedge : edge Seed PendingMacro).
    { unfold edge, semantic_successors; simpl; auto. }
    specialize (Hcomplete Hedge).
    simpl in Hcomplete. intuition discriminate.
  Qed.

  Lemma edge_cost_monotone : forall from to,
    edge from to -> cost from <= cost to.
  Proof.
    intros from to Hedge.
    unfold edge, semantic_successors in Hedge.
    destruct from, to; simpl in *;
      repeat match goal with
      | H : _ \/ _ |- _ => destruct H as [H | H]
      end;
      try contradiction; try discriminate; lia.
  Qed.

  Example complete_expansion_has_valid_cut :
    cut_invariant edge Seed accepting proven
      [CurrentRow; PendingMacro] [Seed].
  Proof.
    change (cut_invariant edge Seed accepting proven
      (remove equal Seed [Seed] ++ semantic_successors Seed)
      (Seed :: [])).
    apply expansion_preserves_cut.
    - apply successors_complete.
    - unfold edge, semantic_successors; simpl; auto.
    - unfold accepting; intros [H | H]; discriminate.
    - apply initial_cut_invariant.
  Qed.

  Example complete_expansion_is_reachable :
    cut_steps equal edge semantic_successors accepting proven
      ([Seed], []) ([CurrentRow; PendingMacro], [Seed]).
  Proof.
    change (cut_steps equal edge semantic_successors accepting proven
      ([Seed], [])
      (remove equal Seed [Seed] ++ semantic_successors Seed, Seed :: [])).
    eapply CutMore.
    - apply CutExpand.
      + simpl; auto.
      + apply successors_complete.
      + unfold accepting; intros [H | H]; discriminate.
    - constructor.
  Qed.

  Lemma fast_path :
    semantic_path edge Seed FastAccept [Seed; PendingMacro; FastAccept].
  Proof.
    eapply PathThen with (y := PendingMacro).
    - unfold edge, semantic_successors; simpl; auto.
    - eapply PathThen with (y := FastAccept).
      + unfold edge, semantic_successors; simpl; auto.
      + constructor.
  Qed.

  Example complete_cut_bounds_fast_completion :
    exists floor,
      cut_floor cost [CurrentRow; PendingMacro] = Some floor /\
      floor <= cost FastAccept.
  Proof.
    eapply complete_cut_bounds_unresolved_completion with
      (edge := edge) (root := Seed) (accepting := accepting)
      (resolved := [Seed])
      (path := [Seed; PendingMacro; FastAccept]).
    - exact edge_cost_monotone.
    - exact (proj1 complete_expansion_has_valid_cut).
    - exact fast_path.
    - right; reflexivity.
    - simpl; intuition discriminate.
  Qed.

  Example current_row_only_is_not_a_cut :
    ~ cut_covers edge Seed accepting [CurrentRow] [Seed].
  Proof.
    intro Hcut.
    specialize (Hcut FastAccept [Seed; PendingMacro; FastAccept]
      fast_path (or_intror eq_refl)).
    simpl in Hcut.
    destruct Hcut as [Hdone | [node [Hinpath Hinlive]]].
    - intuition discriminate.
    - destruct Hinlive as [Heq | []].
      subst node. simpl in Hinpath. intuition discriminate.
  Qed.

  Example incomplete_row_prune_loses_within_cutoff_result :
    (forall node, In node [CurrentRow] -> 5 < cost node) /\
    accepting FastAccept /\ cost FastAccept <= 5 /\
    semantic_path edge Seed FastAccept [Seed; PendingMacro; FastAccept].
  Proof.
    split.
    - intros node Hin. simpl in Hin.
      destruct Hin as [Heq | []]. subst node; simpl; lia.
    - split.
      + right; reflexivity.
      + split; [simpl; lia | exact fast_path].
  Qed.
End BypassControl.
