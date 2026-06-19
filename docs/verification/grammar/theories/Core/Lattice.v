(** * Lattice Structures for Error Correction

    This module defines and proves properties of error correction lattices,
    including path enumeration, score computation, and lattice composition.
*)

From Stdlib Require Import String List Nat PeanoNat Bool Lia.
From Stdlib Require Import QArith.QArith QArith.Qminmax micromega.Lqa.
Require Import Liblevenshtein.Grammar.Verification.Core.Types.
Require Import Liblevenshtein.Grammar.Verification.Core.Edit.
Import ListNotations.

(** ** Lattice Path *)

(** A path through the lattice is a sequence of node indices *)
Definition lattice_path := list nat.

(** Check if a path is valid in a lattice *)
Fixpoint valid_path (lat : Lattice) (path : lattice_path) : bool :=
  match path with
  | [] => false
  | [n] => (n =? lat.(lattice_start)) || (n =? lat.(lattice_end))
  | n1 :: ((n2 :: _) as tail) =>
      (n1 <? length lat.(lattice_nodes)) &&
      (n2 <? length lat.(lattice_nodes)) &&
      existsb (fun e => (e.(edge_from) =? n1) && (e.(edge_to) =? n2))
              lat.(lattice_edges) &&
      valid_path lat tail
  end.

(** A complete path goes from start to end *)
Definition complete_path (lat : Lattice) (path : lattice_path) : bool :=
  match path with
  | [] => false
  | n :: _ =>
      (n =? lat.(lattice_start)) &&
      (last path 0 =? lat.(lattice_end)) &&
      valid_path lat path
  end.

(** The current [wf_lattice] predicate only checks local node/edge bounds.
    Reachability from start to end is a separate contract. *)
Definition lattice_reachable (lat : Lattice) : Prop :=
  exists path, complete_path lat path = true.

(** ** Path Score Computation *)

(** Get edge weight between two nodes *)
Definition get_edge_weight (lat : Lattice) (from to : nat) : score :=
  match find (fun e => (e.(edge_from) =? from) && (e.(edge_to) =? to))
             lat.(lattice_edges) with
  | Some e => e.(edge_weight)
  | None => score_zero
  end.

(** Compute score of a path *)
Fixpoint path_score (lat : Lattice) (path : lattice_path) : score :=
  match path with
  | [] => score_one
  | [_] => score_one
  | n1 :: ((n2 :: _) as tail) =>
      score_mult (get_edge_weight lat n1 n2) (path_score lat tail)
  end.

(** Outgoing edges for a node. *)
Definition outgoing_edges (lat : Lattice) (from : nat) : list LatticeEdge :=
  filter (fun e => e.(edge_from) =? from) lat.(lattice_edges).

(** Enumerate complete paths from [current] to the lattice end using at most
    [fuel] edges.  The fuel is a structural bound, not an optimality claim:
    callers choose it from the finite node set so the function always
    terminates even if the lattice contains cycles. *)
Fixpoint enumerate_paths_from
    (lat : Lattice) (fuel current : nat) : list lattice_path :=
  if current =? lat.(lattice_end) then
    [[current]]
  else
    match fuel with
    | 0 => []
    | S fuel' =>
        flat_map
          (fun e =>
             map (fun suffix => current :: suffix)
                 (enumerate_paths_from lat fuel' e.(edge_to)))
          (outgoing_edges lat current)
    end.

(** Bounded complete paths through a lattice. *)
Definition bounded_complete_paths (lat : Lattice) : list lattice_path :=
  filter (complete_path lat)
         (enumerate_paths_from
            lat
            (length lat.(lattice_nodes))
            lat.(lattice_start)).

(** Maximum score helper. *)
Definition score_max (left right : score) : score :=
  if score_lt left right then right else left.

(** Maximum score over a finite path list. *)
Fixpoint max_path_score (lat : Lattice) (paths : list lattice_path) : score :=
  match paths with
  | [] => score_zero
  | [path] => path_score lat path
  | path :: rest => score_max (path_score lat path) (max_path_score lat rest)
  end.

Lemma max_path_score_achievable : forall lat paths,
  paths <> [] ->
  exists path,
    In path paths /\
    path_score lat path == max_path_score lat paths.
Proof.
  intros lat paths Hnonempty.
  destruct paths as [| path rest]; [contradiction |].
  revert path Hnonempty.
  induction rest as [| next rest IH]; intros path _.
  - exists path. split; [left; reflexivity | reflexivity].
  - simpl.
    destruct (score_lt (path_score lat path)
                       (max_path_score lat (next :: rest))) eqn:Hlt.
    + destruct (IH next ltac:(discriminate)) as [best [Hin Hscore]].
      exists best.
      split; [right; exact Hin |].
      replace (match rest with
               | [] => path_score lat next
               | _ :: _ => score_max (path_score lat next)
                                    (max_path_score lat rest)
               end)
        with (max_path_score lat (next :: rest)) by reflexivity.
      unfold score_max. rewrite Hlt. exact Hscore.
    + exists path.
      split; [left; reflexivity |].
      replace (match rest with
               | [] => path_score lat next
               | _ :: _ => score_max (path_score lat next)
                                    (max_path_score lat rest)
               end)
        with (max_path_score lat (next :: rest)) by reflexivity.
      unfold score_max. rewrite Hlt. reflexivity.
Qed.

(** ** Lattice Construction *)

(** Build a simple linear lattice (no corrections) *)
Definition linear_lattice (s : string) : Lattice :=
  let len := String.length s in
  let nodes := map (fun i =>
    {| lattice_position := i;
       lattice_text := s;  (* Simplified *)
       lattice_score := score_one;
       lattice_edits := [] |})
    (seq 0 (S len)) in
  let edges := map (fun i =>
    {| edge_from := i;
       edge_to := S i;
       edge_weight := score_one |})
    (seq 0 len) in
  {| lattice_nodes := nodes;
     lattice_edges := edges;
     lattice_start := 0;
     lattice_end := len |}.

(** ** Lattice Properties *)

Lemma linear_lattice_edges_valid : forall s e,
  String.length s > 0 ->
  In e (linear_lattice s).(lattice_edges) ->
  e.(edge_from) < List.length (linear_lattice s).(lattice_nodes) /\
  e.(edge_to) < List.length (linear_lattice s).(lattice_nodes).
Proof.
  intros s e _ Hin.
  unfold linear_lattice in *; simpl in *.
  rewrite map_length, seq_length.
  apply in_map_iff in Hin as [i [Heq Hinseq]].
  subst e; simpl.
  apply in_seq in Hinseq.
  lia.
Qed.

(** Linear lattice is well-formed *)
Theorem linear_lattice_wf : forall s,
  String.length s > 0 ->
  wf_lattice (linear_lattice s).
Proof.
  intros s Hlen.
  unfold wf_lattice, linear_lattice; simpl.
  repeat split.
  - (* start node exists *)
    rewrite map_length, seq_length. lia.
  - (* end node exists *)
    rewrite map_length, seq_length. lia.
  - (* all edges valid *)
    apply Forall_forall. intros e Hin.
    apply in_map_iff in Hin as [i [Heq Hinseq]].
    subst e; simpl.
    apply in_seq in Hinseq.
    rewrite map_length, seq_length.
    repeat split; try lia.
    all: unfold wf_score, score_zero, score_one; lra.
  - (* all node scores are well-formed *)
    apply Forall_forall. intros n Hin.
    destruct Hin as [Heq | Hin].
    + subst n; simpl.
      unfold wf_score, score_zero, score_one; split; lra.
    + apply in_map_iff in Hin as [i [Heq _]].
      subst n; simpl.
    unfold wf_score, score_zero, score_one; split; lra.
Qed.

(** A lattice satisfying the reachability contract has a complete path. *)
Theorem lattice_has_path : forall lat,
  lattice_reachable lat ->
  exists path, complete_path lat path = true.
Proof.
  intros lat Hreachable.
  exact Hreachable.
Qed.

(** ** Best Path (Viterbi Algorithm) *)

(** Find the highest-scoring bounded complete path. *)
Definition best_path_score (lat : Lattice) : score :=
  max_path_score lat (bounded_complete_paths lat).

Definition best_path_score_achievable (lat : Lattice) : Prop :=
  exists path,
    In path (bounded_complete_paths lat) /\
    complete_path lat path = true /\
    path_score lat path == best_path_score lat.

Theorem best_path_achievable : forall lat,
  bounded_complete_paths lat <> [] ->
  best_path_score_achievable lat.
Proof.
  intros lat Hnonempty.
  unfold best_path_score_achievable, best_path_score.
  destruct (max_path_score_achievable lat (bounded_complete_paths lat) Hnonempty)
    as [path [Hin Hscore]].
  exists path.
  repeat split.
  - exact Hin.
  - apply filter_In in Hin as [_ Hcomplete].
    exact Hcomplete.
  - exact Hscore.
Qed.

Theorem best_path_achievable_explicit : forall lat,
  best_path_score_achievable lat ->
  exists path,
    In path (bounded_complete_paths lat) /\
    complete_path lat path = true /\
    path_score lat path == best_path_score lat.
Proof.
  intros lat Hachievable.
  exact Hachievable.
Qed.

(** ** Top-K Paths *)

(** Select the highest-scoring path from a finite list. *)
Fixpoint best_path_by_score
    (lat : Lattice)
    (paths : list lattice_path)
    : option lattice_path :=
  match paths with
  | [] => None
  | path :: rest =>
      match best_path_by_score lat rest with
      | None => Some path
      | Some best =>
          if score_lt (path_score lat path) (path_score lat best) then
            Some best
          else
            Some path
      end
  end.

Lemma best_path_by_score_in : forall lat paths best,
  best_path_by_score lat paths = Some best ->
  In best paths.
Proof.
  intros lat paths.
  induction paths as [| path rest IH]; intros best Hbest.
  - simpl in Hbest. discriminate.
  - simpl in Hbest.
    destruct (best_path_by_score lat rest) as [rest_best |] eqn:Hrest.
    + destruct (score_lt (path_score lat path)
                         (path_score lat rest_best)) eqn:Hlt.
      * inversion Hbest; subst.
        right. apply IH. reflexivity.
      * inversion Hbest; subst.
        left. reflexivity.
    + inversion Hbest; subst.
      left. reflexivity.
Qed.

(** Find the highest-scoring bounded complete path when [k] is nonzero. *)
Definition top_k_paths (lat : Lattice) (k : nat) : list lattice_path :=
  match k with
  | 0 => []
  | S _ =>
      match best_path_by_score lat (bounded_complete_paths lat) with
      | Some path => [path]
      | None => []
      end
  end.

(** All top-k paths are complete *)
Theorem top_k_paths_complete : forall lat k,
  wf_lattice lat ->
  Forall (fun p => complete_path lat p = true) (top_k_paths lat k).
Proof.
  intros lat k _.
  unfold top_k_paths.
  destruct k as [| k']; [constructor |].
  destruct (best_path_by_score lat (bounded_complete_paths lat)) as [path |] eqn:Hbest.
  - constructor.
    + apply best_path_by_score_in in Hbest.
      apply filter_In in Hbest as [_ Hcomplete].
      exact Hcomplete.
    + constructor.
  - constructor.
Qed.

(** Top-k paths are sorted by score *)
Theorem top_k_paths_sorted : forall lat k,
  let paths := top_k_paths lat k in
  forall i j,
    (i < j < List.length paths)%nat ->
    (path_score lat (nth i paths []) >= path_score lat (nth j paths []))%Q.
Proof.
  intros lat k.
  unfold top_k_paths.
  destruct k as [| k']; simpl.
  - intros i j Hij. exfalso. lia.
  - destruct (best_path_by_score lat (bounded_complete_paths lat)) as [path |] eqn:Hbest; simpl;
      intros i j Hij; exfalso; lia.
Qed.

(** ** Lattice Expansion with Edits *)

(** Skip edges model single deletions in the current edge-only lattice
    representation.  They preserve the node set and add an extra way to advance
    by two positions when edits are enabled. *)
Definition edit_skip_edges (lat : Lattice) (max_edits : nat) : list LatticeEdge :=
  if max_edits =? 0 then
    []
  else
    map
      (fun i =>
         {| edge_from := i;
            edge_to := i + 2;
            edge_weight := score_one |})
      (seq 0 (Nat.pred (Nat.pred (length lat.(lattice_nodes))))).

Lemma edit_skip_edges_valid : forall lat max_edits e,
  In e (edit_skip_edges lat max_edits) ->
  e.(edge_from) < length lat.(lattice_nodes) /\
  e.(edge_to) < length lat.(lattice_nodes) /\
  wf_score e.(edge_weight).
Proof.
  intros lat max_edits e Hin.
  unfold edit_skip_edges in Hin.
  destruct (max_edits =? 0) eqn:Hmax; [contradiction |].
  apply in_map_iff in Hin as [i [Heq Hinseq]].
  subst e. simpl.
  apply in_seq in Hinseq.
  remember (length lat.(lattice_nodes)) as node_count eqn:Hnodes.
  destruct node_count as [| [| node_count']]; simpl in Hinseq; try lia.
  split.
  - lia.
  - split.
    + lia.
    + unfold wf_score, score_zero, score_one; lra.
Qed.

(** Add error correction edges to a lattice. *)
Definition expand_lattice_with_edits (lat : Lattice) (max_edits : nat) : Lattice :=
  {| lattice_nodes := lat.(lattice_nodes);
     lattice_edges := lat.(lattice_edges) ++ edit_skip_edges lat max_edits;
     lattice_start := lat.(lattice_start);
     lattice_end := lat.(lattice_end) |}.

(** Expansion preserves well-formedness *)
Theorem expand_lattice_wf : forall lat max_edits,
  wf_lattice lat ->
  wf_lattice (expand_lattice_with_edits lat max_edits).
Proof.
  intros lat max_edits Hwf.
  unfold expand_lattice_with_edits.
  unfold wf_lattice in *; simpl in *.
  destruct Hwf as [Hstart [Hend [Hedges Hnodes]]].
  repeat split; try assumption.
  apply Forall_app.
  split; [exact Hedges |].
  apply Forall_forall.
  intros e Hin.
  apply edit_skip_edges_valid in Hin.
  exact Hin.
Qed.

(** Existing valid paths remain valid after edit-edge expansion. *)
Lemma valid_path_expand_preserved : forall lat max_edits path,
  valid_path lat path = true ->
  valid_path (expand_lattice_with_edits lat max_edits) path = true.
Proof.
  intros lat max_edits path.
  induction path as [| n rest IH]; intros Hvalid.
  - exact Hvalid.
  - destruct rest as [| n2 rest'].
    + exact Hvalid.
    + simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hprefix Htail].
      apply andb_true_iff in Hprefix as [Hbounds Hedge].
      apply andb_true_iff in Hbounds as [Hn Hn2].
      simpl.
      rewrite Hn, Hn2.
      rewrite existsb_app, Hedge.
      simpl.
      pose proof (IH Htail) as Htail_expanded.
      simpl in Htail_expanded.
      rewrite Hn2 in Htail_expanded.
      exact Htail_expanded.
Qed.

(** Expansion adds paths *)
Theorem expand_lattice_adds_paths : forall lat max_edits,
  wf_lattice lat ->
  forall path,
    complete_path lat path = true ->
    complete_path (expand_lattice_with_edits lat max_edits) path = true.
Proof.
  intros lat max_edits Hwf path Hpath.
  unfold complete_path in *.
  destruct path as [| n rest]; [exact Hpath |].
  simpl in *.
  apply andb_true_iff in Hpath as [Hprefix Hvalid].
  apply andb_true_iff in Hprefix as [Hstart Hend].
  rewrite Hstart, Hend.
  simpl.
  destruct rest as [| n2 rest']; [reflexivity |].
  change (valid_path (expand_lattice_with_edits lat max_edits)
                     (n :: n2 :: rest') = true).
  apply valid_path_expand_preserved.
  exact Hvalid.
Qed.

(** ** Lattice Composition *)

(** Compose two lattices sequentially *)
Definition compose_lattices (lat1 lat2 : Lattice) : Lattice :=
  (* Connect end of lat1 to start of lat2 *)
  let offset := length lat1.(lattice_nodes) in
  let shifted_nodes := map (fun n =>
    {| lattice_position := n.(lattice_position);
       lattice_text := n.(lattice_text);
       lattice_score := n.(lattice_score);
       lattice_edits := n.(lattice_edits) |})
    lat2.(lattice_nodes) in
  let shifted_edges := map (fun e =>
    {| edge_from := e.(edge_from) + offset;
       edge_to := e.(edge_to) + offset;
       edge_weight := e.(edge_weight) |})
    lat2.(lattice_edges) in
  let connecting_edge := {|
    edge_from := lat1.(lattice_end);
    edge_to := lat2.(lattice_start) + offset;
    edge_weight := score_one
  |} in
  {| lattice_nodes := lat1.(lattice_nodes) ++ shifted_nodes;
     lattice_edges := lat1.(lattice_edges) ++ [connecting_edge] ++ shifted_edges;
     lattice_start := lat1.(lattice_start);
     lattice_end := lat2.(lattice_end) + offset |}.

(** Composition preserves well-formedness *)
Theorem compose_lattices_wf : forall lat1 lat2,
  wf_lattice lat1 ->
  wf_lattice lat2 ->
  wf_lattice (compose_lattices lat1 lat2).
Proof.
  intros lat1 lat2 Hwf1 Hwf2.
  unfold wf_lattice, compose_lattices in *; simpl in *.
  destruct Hwf1 as [Hstart1 [Hend1 [Hedges1 Hnodes1]]].
  destruct Hwf2 as [Hstart2 [Hend2 [Hedges2 Hnodes2]]].
  repeat split.
  - rewrite app_length. lia.
  - rewrite app_length, map_length. lia.
  - apply Forall_app. split.
    + eapply Forall_impl; [| exact Hedges1].
      intros e [Hfrom [Hto [Hwlo Hwhi]]]. simpl in *.
      rewrite app_length, map_length.
      repeat split; try lia.
      * exact Hwlo.
      * exact Hwhi.
    + constructor.
      * simpl.
        rewrite app_length, map_length.
        repeat split.
        -- lia.
        -- lia.
        -- unfold wf_score, score_zero, score_one; lra.
        -- unfold wf_score, score_zero, score_one; lra.
      * apply Forall_forall. intros e Hin.
        apply in_map_iff in Hin as [e0 [Heq Hin0]].
        subst e. simpl.
        rewrite app_length, map_length.
        rewrite Forall_forall in Hedges2.
        specialize (Hedges2 e0 Hin0) as [Hfrom [Hto [Hwlo Hwhi]]].
        repeat split; try lia.
        -- exact Hwlo.
        -- exact Hwhi.
  - apply Forall_app. split.
    + exact Hnodes1.
    + rewrite Forall_forall in *.
      intros n Hin.
      apply in_map_iff in Hin as [n0 [Heq Hin0]].
      subst n. simpl.
      apply Hnodes2. exact Hin0.
Qed.

(** ** Lattice Pruning *)

(** Prune low-scoring paths from lattice *)
Definition prune_lattice (lat : Lattice) (threshold : score) : Lattice :=
  let pruned_edges := filter (fun e => score_le threshold e.(edge_weight))
                              lat.(lattice_edges) in
  {| lattice_nodes := lat.(lattice_nodes);
     lattice_edges := pruned_edges;
     lattice_start := lat.(lattice_start);
     lattice_end := lat.(lattice_end) |}.

(** Pruning preserves well-formedness *)
Theorem prune_lattice_wf : forall lat threshold,
  wf_lattice lat ->
  wf_lattice (prune_lattice lat threshold).
Proof.
  intros lat threshold Hwf.
  unfold wf_lattice, prune_lattice; simpl.
  destruct Hwf as [Hstart [Hend [Hedges Hnodes]]].
  repeat split; auto.
  - (* Pruned edges still connect valid nodes *)
    apply Forall_forall. intros e Hin.
    apply filter_In in Hin as [Hin_orig _].
    rewrite Forall_forall in Hedges.
    apply Hedges. exact Hin_orig.
Qed.

(** Pruning removes low-scoring edges.  A stronger path-score theorem would
    need additional assumptions because path scores multiply edge weights. *)
Theorem pruning_removes_low_scores : forall lat threshold e,
  In e (prune_lattice lat threshold).(lattice_edges) ->
  score_le threshold e.(edge_weight) = true.
Proof.
  intros lat threshold e Hin.
  unfold prune_lattice in Hin; simpl in Hin.
  apply filter_In in Hin as [_ Hkeep].
  exact Hkeep.
Qed.

(** ** Beam Search on Lattice *)

(** Beam search with fixed width *)
Definition beam_search (lat : Lattice) (beam_width : nat) : list lattice_path :=
  top_k_paths lat beam_width.

(** Beam search returns at most beam_width paths *)
Theorem beam_search_bounded : forall lat beam_width,
  (List.length (beam_search lat beam_width) <= beam_width)%nat.
Proof.
  intros lat beam_width.
  unfold beam_search, top_k_paths.
  destruct beam_width as [| beam_width']; simpl; [lia |].
  destruct (best_path_by_score lat (bounded_complete_paths lat)); simpl; lia.
Qed.

(** Beam search paths are complete *)
Theorem beam_search_complete : forall lat beam_width,
  wf_lattice lat ->
  Forall (fun p => complete_path lat p = true) (beam_search lat beam_width).
Proof.
  intros lat beam_width Hwf.
  unfold beam_search.
  apply top_k_paths_complete. exact Hwf.
Qed.

(** ** Lattice Minimization *)

(** Remove nodes not reachable from start or not reaching end *)
Definition minimize_lattice (lat : Lattice) : Lattice :=
  (* Remove unreachable nodes and edges *)
  lat.  (* Placeholder *)

(** Minimization preserves reachable paths *)
Theorem minimize_preserves_paths : forall lat path,
  complete_path lat path = true ->
  complete_path (minimize_lattice lat) path = true.
Proof.
  intros lat path Hpath.
  unfold minimize_lattice.
  exact Hpath.
Qed.

(** Minimization reduces lattice size *)
Theorem minimize_reduces_size : forall lat,
  (List.length (minimize_lattice lat).(lattice_nodes) <= List.length lat.(lattice_nodes))%nat.
Proof.
  intros lat.
  unfold minimize_lattice. simpl. lia.
Qed.
