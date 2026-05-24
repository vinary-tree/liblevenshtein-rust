(** * State Transition Correctness

    This module proves correctness properties of state transitions in the
    Generalized Levenshtein NFA. Key results:
    - Transitions preserve distance bounds
    - Characteristic vector encoding is correct
    - Context propagation is sound
    - Transitions are deterministic and monotonic
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qround.
Require Import Coq.NArith.BinNat.
Require Import Coq.micromega.Lia.
Require Import Coq.micromega.Lqa.
Import ListNotations.

Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.

(** Edit distance of a sequence is the number of operations (simple definition).
    This is a local definition to avoid circular imports with Completeness.v *)
Definition edit_distance (edits : list OperationType) : nat := length edits.

(** ** Characteristic Vector Encoding Correctness *)

(** *** Helper Lemmas for CV Encoding *)

(** String decomposition at a position *)
Lemma string_decompose_at : forall s pos,
  pos < String.length s ->
  exists s1 c s2, s = append s1 (String c s2) /\ String.length s1 = pos.
Proof.
  intros s pos Hlt.
  generalize dependent pos.
  induction s; intros pos Hlt.
  - simpl in Hlt. lia.
  - destruct pos.
    + exists EmptyString, a, s. split; reflexivity.
    + simpl in Hlt. assert (Hlt': pos < String.length s) by lia.
      specialize (IHs pos Hlt').
      destruct IHs as [s1 [c' [s2 [Heq Hlen]]]].
      exists (String a s1), c', s2.
      split.
      * simpl. f_equal. assumption.
      * simpl. f_equal. assumption.
Qed.

(** nth_error correspondence with string decomposition *)
Lemma nth_error_app_decompose : forall s1 c s2 pos,
  String.length s1 = pos ->
  nth_error (list_ascii_of_string (append s1 (String c s2))) pos = Some c.
Proof.
  intros s1 c s2 pos Hlen.
  generalize dependent pos.
  induction s1; intros pos Hlen.
  - simpl in Hlen. subst pos. reflexivity.
  - destruct pos.
    + simpl in Hlen. discriminate.
    + simpl in Hlen. injection Hlen as Hlen.
      simpl. apply IHs1. assumption.
Qed.

(** nth_error with string decomposition (reverse direction) *)
Lemma nth_error_some_decompose : forall s pos c,
  nth_error (list_ascii_of_string s) pos = Some c ->
  exists s1 s2, s = append s1 (String c s2) /\ String.length s1 = pos.
Proof.
  intros s pos c Hnth.
  generalize dependent pos.
  induction s; intros pos Hnth.
  - destruct pos; simpl in Hnth; discriminate.
  - destruct pos.
    + simpl in Hnth. injection Hnth as Heq. subst a.
      exists EmptyString, s. split; reflexivity.
    + simpl in Hnth.
      apply IHs in Hnth.
      destruct Hnth as [s1 [s2 [Heq Hlen]]].
      exists (String a s1), s2.
      split.
      * simpl. f_equal. assumption.
      * simpl. f_equal. assumption.
Qed.

(** Helper lemma for build_cv correctness *)
Lemma build_cv_set_iff : forall s c offset pos,
  cv_test_bit (build_cv s c offset) pos = true <->
  exists n, offset <= n < offset + String.length s /\
            pos = n /\
            nth_error (list_ascii_of_string s) (n - offset) = Some c.
Proof.
  intros s c offset pos.
  generalize dependent offset.
  induction s as [| c' s' IH]; intros offset.
  - (* Base case: EmptyString *)
    split; intro H.
    + simpl in H. discriminate.
    + destruct H as [n [Hrange _]]. simpl in Hrange. lia.
  - (* Inductive case: String c' s' *)
    simpl build_cv. simpl String.length.
    destruct (Ascii.eqb c c') eqn:Heqb.
    + (* c = c': bit is set at offset *)
      apply Ascii.eqb_eq in Heqb. subst c'.
      split; intro H.
      * (* Forward: bit set → position exists *)
        destruct (Nat.eq_dec pos offset) as [Heq | Hneq].
        -- (* pos = offset: found at current position *)
           subst pos. exists offset. split; [| split].
           ++ simpl. lia.
           ++ reflexivity.
           ++ simpl. replace (offset - offset) with 0 by lia. reflexivity.
        -- (* pos ≠ offset: must be in rest *)
           rewrite cv_set_test_neq in H by (apply not_eq_sym; assumption).
           rewrite IH in H.
           destruct H as [n [Hrange [Heq Hnth]]].
           exists n. split; [| split].
           ++ simpl. lia.
           ++ assumption.
           ++ simpl. replace (n - offset) with (S (n - S offset)) by lia.
              simpl. assumption.
      * (* Backward: position exists → bit set *)
        destruct H as [n [Hrange [Heq Hnth]]]. subst pos.
        destruct (Nat.eq_dec n offset) as [Heq' | Hneq'].
        -- (* n = offset: bit set at current position *)
           subst n. apply cv_set_test_eq.
        -- (* n ≠ offset: bit in rest *)
           rewrite cv_set_test_neq by (apply not_eq_sym; assumption).
           rewrite IH. exists n. split; [| split].
           ++ simpl in Hrange. lia.
           ++ reflexivity.
           ++ simpl in Hnth. destruct (n - offset) eqn:Hdiff.
              ** lia.
              ** simpl in Hnth. replace (n - S offset) with n0 by lia. assumption.
    + (* c ≠ c': bit not set at offset, check rest *)
      split; intro H.
      * (* Forward: bit set in rest → position exists  *)
        rewrite IH in H.
        destruct H as [n [Hrange [Heq Hnth]]].
        exists n. split; [| split].
        -- simpl. lia.
        -- assumption.
        -- destruct (n - offset) as [| n0] eqn:Hdiff.
           ++ lia.
           ++ simpl. replace n0 with (n - S offset) by lia. exact Hnth.
      * (* Backward: position exists → bit set in rest *)
        destruct H as [n [Hrange [Heq Hnth]]]. subst pos.
        simpl in Hnth.
        (* n - offset cannot be 0, because c <> c' but nth_error at 0 would give c' *)
        destruct (n - offset) as [| n0] eqn:Hdiff.
        -- (* n = offset: contradiction since c <> c' *)
           simpl in Hnth. apply Ascii.eqb_neq in Heqb.
           injection Hnth as Hcontra. symmetry in Hcontra. contradiction.
        -- (* n > offset: bit is in the rest *)
           rewrite IH. exists n. split; [| split].
           ++ simpl in Hrange. lia.
           ++ reflexivity.
           ++ simpl in Hnth. replace (n - S offset) with n0 by lia. exact Hnth.
Qed.

(** CV encoding matches string characters *)
Theorem cv_encoding_correct : forall s c pos,
  cv_test_bit (characteristic_vector s c) pos = true <->
  exists s1 s2, s = append s1 (String c s2) /\ String.length s1 = pos.
Proof.
  intros s c pos.
  unfold characteristic_vector.
  rewrite build_cv_set_iff.
  split; intro H.
  - (* Forward: bit set → string decomposition *)
    destruct H as [n [Hrange [Heq Hnth]]].
    subst n. simpl in Hrange.
    replace 0 with (0 + 0) in Hrange by lia.
    replace (0 + String.length s) with (String.length s) in Hrange by lia.
    assert (Hdec: pos < String.length s) by lia.
    apply string_decompose_at in Hdec.
    destruct Hdec as [s1 [c' [s2 [Heq_dec Hlen_dec]]]].
    (* Now we need to show c' = c using nth_error *)
    replace (pos - 0) with pos in Hnth by lia.
    rewrite Heq_dec in Hnth.
    assert (Hc: nth_error (list_ascii_of_string (append s1 (String c' s2))) pos = Some c).
    { assumption. }
    pose proof (nth_error_app_decompose s1 c' s2 pos Hlen_dec) as Hnth_dec.
    rewrite Hnth_dec in Hc. injection Hc as Hc_eq. subst c'.
    exists s1, s2. split.
    + exact Heq_dec.
    + exact Hlen_dec.
  - (* Backward: string decomposition → bit set *)
    destruct H as [s1 [s2 [Heq Hlen]]].
    exists pos. split; [| split].
    + (* Prove range: 0 <= pos < String.length s *)
      subst s pos. simpl.
      assert (Hlen_append: forall a b, String.length (append a b) = String.length a + String.length b).
      { induction a; intros; simpl; auto. }
      rewrite Hlen_append. simpl. lia.
    + reflexivity.
    + replace (pos - 0) with pos by lia.
      subst s. apply nth_error_app_decompose. assumption.
Qed.

(** CV correctly encodes absence of character *)
Theorem cv_encoding_absent : forall s c pos,
  cv_test_bit (characteristic_vector s c) pos = false ->
  forall s1 s2, s = append s1 (String c s2) -> String.length s1 <> pos.
Proof.
  intros s c pos Hcv s1 s2 Heq.
  intro Hcontra.
  assert (Hexists: exists s1 s2, s = append s1 (String c s2) /\ String.length s1 = pos).
  { exists s1, s2. split; assumption. }
  apply cv_encoding_correct in Hexists.
  congruence.
Qed.

(** CV encoding is unique *)
Theorem cv_encoding_unique : forall s1 s2 c,
  (forall pos, cv_test_bit (characteristic_vector s1 c) pos =
                cv_test_bit (characteristic_vector s2 c) pos) ->
  (forall pos, (exists s1' s2', s1 = append s1' (String c s2') /\ String.length s1' = pos) <->
                (exists s1' s2', s2 = append s1' (String c s2') /\ String.length s1' = pos)).
Proof.
  intros s1 s2 c Hcv pos.
  split; intros [s1' [s2' [Heq Hlen]]].
  - (* s1 decomposition → s2 decomposition *)
    apply cv_encoding_correct.
    rewrite <- Hcv.
    apply cv_encoding_correct.
    exists s1', s2'. split; assumption.
  - (* s2 decomposition → s1 decomposition *)
    apply cv_encoding_correct.
    rewrite Hcv.
    apply cv_encoding_correct.
    exists s1', s2'. split; assumption.
Qed.

(** ** Distance Preservation *)

(** Applying an operation respects bounded diagonal *)
Theorem operation_preserves_distance_bound : forall op target input tpos ipos p p',
  wf_operation op ->
  bounded_diagonal 1 op ->
  In p' (apply_operation_to_position op target input tpos ipos p) ->
  let i_diff := pos_i p' - pos_i p in
  let e_diff := pos_e p' - pos_e p in
  i_diff <= op_consume_x op /\
  e_diff <= Nat.max 1 (Z.to_nat (Qceiling (op_weight op))).
Proof.
  intros op target input tpos ipos p p' Hwf Hbd Hin.
  unfold apply_operation_to_position in Hin.
  destruct (can_apply op target input (pos_i p) tpos) eqn:Hcan; simpl in Hin.
  - destruct Hin as [Heq | Hcontra]; [| contradiction].
    subst p'. simpl. split.
    + unfold wf_operation in Hwf. destruct Hwf as [_ [Hcons_x _]]. lia.
    + unfold bounded_diagonal in Hbd. lia.
  - contradiction.
Qed.

(** Transition preserves error bound *)
Theorem delta_preserves_error_bound : forall aut target input pos st,
  wf_automaton aut ->
  wf_state st ->
  Forall (fun p => pos_e p <= automaton_max_distance aut)
    (state_positions (delta aut target input pos st)).
Proof.
  intros aut target input pos st Hwf_aut Hwf_st.
  unfold delta.
  apply Forall_forall. intros p Hin.
  (* Use the prune_state specification: positions in prune_state are a subset
     of the filtered positions, and the filter ensures error bound. *)
  apply prune_state_incl_holds in Hin.
  apply filter_In in Hin. destruct Hin as [_ Hle].
  apply Nat.leb_le. exact Hle.
Qed.

(** Running automaton keeps errors bounded *)
Theorem run_preserves_error_bound : forall aut target input pos st fuel,
  wf_automaton aut ->
  wf_state st ->
  state_max_distance st = automaton_max_distance aut ->
  Forall (fun p => pos_e p <= automaton_max_distance aut)
    (state_positions (run_automaton_from aut target input pos st fuel)).
Proof.
  intros aut target input pos st fuel Hwf_aut.
  revert pos st.
  induction fuel as [| fuel IH]; intros pos st Hwf_st Hdist; simpl.
  - unfold wf_state, wf_position in Hwf_st.
    rewrite <- Hdist. exact Hwf_st.
  - destruct (String.length input <=? pos) eqn:Hdone.
    + unfold wf_state, wf_position in Hwf_st.
      rewrite <- Hdist. exact Hwf_st.
    + apply IH.
      * apply delta_preserves_wf; assumption.
      * unfold delta, prune_state. simpl. reflexivity.
Qed.

(** ** Position Monotonicity *)

(** Operation application never decreases position.
    This follows from the definition of apply_operation_to_position where
    new_i = pos_i p + op_consume_x op, and op_consume_x op >= 0. *)
Lemma operation_increases_position :
  forall op target input tpos ipos p p',
    In p' (apply_operation_to_position op target input tpos ipos p) ->
    pos_i p' >= pos_i p.
Proof.
  intros op target input tpos ipos p p' Hin.
  unfold apply_operation_to_position in Hin.
  destruct (can_apply op target input (pos_i p) tpos) eqn:Hcan;
    simpl in Hin.
  - destruct Hin as [Heq | Hcontra]; [subst p' | contradiction].
    simpl. lia.
  - contradiction.
Qed.

(** Membership in apply_all_operations comes from one listed operation. *)
Lemma apply_all_operations_in : forall ops target input tpos ipos p p',
  In p' (apply_all_operations ops target input tpos ipos p) ->
  exists op,
    In op ops /\
    In p' (apply_operation_to_position op target input tpos ipos p).
Proof.
  induction ops as [| op rest IH]; intros target input tpos ipos p p' Hin;
    simpl in Hin.
  - contradiction.
  - apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + exists op. split; [left; reflexivity | exact Hin].
    + destruct (IH target input tpos ipos p p' Hin) as [op' [Hin_ops Hin_apply]].
      exists op'. split; [right; exact Hin_ops | exact Hin_apply].
Qed.

(** delta produces positions derived from input state positions.
    The delta function applies operations to positions in the input state,
    then filters and prunes. Each resulting position p' derives from some
    input position p via operation application. *)
Lemma delta_positions_derive_from_input :
  forall aut target input pos st p',
    In p' (state_positions (delta aut target input pos st)) ->
    exists p op,
      In p (state_positions st) /\
      In op (automaton_operations aut) /\
      In p' (apply_operation_to_position op target input pos pos p).
Proof.
  intros aut target input pos st p' Hin.
  unfold delta in Hin.
  unfold prune_state in Hin. simpl in Hin.
  apply prune_subsumed_is_sublist in Hin.
  apply filter_In in Hin. destruct Hin as [Hin _].
  apply in_flat_map in Hin. destruct Hin as [p [Hin_st Hin_ops]].
  apply apply_all_operations_in in Hin_ops.
  destruct Hin_ops as [op [Hin_op Hin_apply]].
  exists p, op. repeat split; assumption.
Qed.

(** Transitions increase position in target word *)
Theorem delta_increases_position : forall aut target input pos st p',
  In p' (state_positions (delta aut target input pos st)) ->
  exists p, In p (state_positions st) /\ pos_i p' >= pos_i p.
Proof.
  intros aut target input pos st p' Hin'.
  (* Use the delta provenance lemma to recover the input position. *)
  destruct (delta_positions_derive_from_input aut target input pos st p' Hin')
    as [p [op [Hin_st [Hin_op Hin_apply]]]].
  exists p. split.
  - exact Hin_st.
  - (* Apply the operation monotonicity lemma. *)
    apply (operation_increases_position op target input pos pos p p' Hin_apply).
Qed.

(** Contract: Running automaton from later position preserves reachable positions.
    Starting from a later input position (pos2 >= pos1) means we skip some input,
    but positions reachable from pos1 with fuel steps remain reachable or have
    corresponding positions in the run from pos2.

    Citation: Mitankin, Mihov, and Schulz, "Deciding Word Neighborhood with
    Universal Neighborhood Automata", TCS 412(22):2340-2355, 2011,
    DOI 10.1016/j.tcs.2011.01.013, Section 3 monotone generalized edit
    operations and Definition 15 automaton transitions. *)
(** Running automaton monotonically increases position *)
Theorem run_monotone_position : forall
  (run_position_monotone :
    forall aut target input pos1 pos2 st fuel,
      pos1 <= pos2 ->
      forall p, In p (state_positions (run_automaton_from aut target input pos1 st fuel)) ->
      exists p', In p' (state_positions (run_automaton_from aut target input pos2 st fuel)) /\
        pos_i p <= pos_i p')
  aut target input pos1 pos2 st fuel,
  pos1 <= pos2 ->
  forall p, In p (state_positions (run_automaton_from aut target input pos1 st fuel)) ->
  exists p', In p' (state_positions (run_automaton_from aut target input pos2 st fuel)) /\
    pos_i p <= pos_i p'.
Proof.
  intros run_position_monotone aut target input pos1 pos2 st fuel Hle.
  apply run_position_monotone. assumption.
Qed.

(** ** Context Propagation Correctness *)

(** Context is correctly updated after operation *)
Theorem context_update_correct : forall op target input tpos ipos p,
  can_apply op target input (pos_i p) tpos = true ->
  forall p', In p' (apply_operation_to_position op target input tpos ipos p) ->
  match pos_ctx p' with
  | Initial => pos_i p' = 0
  | Final => pos_i p' = String.length target
  | _ => True
  end.
Proof.
  intros op target input tpos ipos p Hcan p' Hin'.
  unfold apply_operation_to_position in Hin'.
  rewrite Hcan in Hin'. simpl in Hin'.
  destruct Hin' as [Heq | Hcontra]; [| contradiction].
  subst p'. simpl.
  destruct (pos_i p + op_consume_x op =? 0) eqn:Hi0.
  - apply Nat.eqb_eq in Hi0. assumption.
  - destruct (pos_i p + op_consume_x op =? String.length target) eqn:Hif.
    + apply Nat.eqb_eq in Hif. assumption.
    + auto.
Qed.

(** Context requirements are enforced *)
Theorem context_enforcement : forall op target input tpos p,
  op_context op <> Anywhere ->
  can_apply op target input (pos_i p) tpos = true ->
  context_matches (op_context op) target (pos_i p) = true.
Proof.
  intros op target input tpos p Hctx Hcan.
  exact (context_sensitive_correctness op target input tpos p Hctx Hcan).
Qed.

(** ** Transition Determinism *)

(** Applying same operation twice yields same result *)
Theorem operation_application_deterministic : forall op target input tpos ipos p,
  apply_operation_to_position op target input tpos ipos p =
  apply_operation_to_position op target input tpos ipos p.
Proof.
  intros. reflexivity.
Qed.

(** Transition function is deterministic *)
Theorem delta_deterministic_full : forall aut target input pos st,
  delta aut target input pos st = delta aut target input pos st.
Proof.
  intros. reflexivity.
Qed.

(** ** Reachability *)

(** Positions reachable in one step *)
Definition reachable_in_one_step
    (aut : GeneralizedAutomaton)
    (target input : string)
    (pos : nat)
    (p p' : Position)
    : Prop :=
  exists op,
    In op (automaton_operations aut) /\
    In p' (apply_operation_to_position op target input pos pos p).

(** Contract: Two-step reachability implies automaton run reachability.
    If p1 can reach p2 in one step at position pos, and p2 can reach p3
    in one step at position (S pos), then starting from a state containing p1,
    running the automaton for sufficient fuel will reach p3.

    Citation: Mitankin, Mihov, and Schulz, "Deciding Word Neighborhood with
    Universal Neighborhood Automata", TCS 412(22):2340-2355, 2011,
    DOI 10.1016/j.tcs.2011.01.013, Definition 15 transition semantics. *)
(** Reachability is transitive *)
Theorem reachability_transitive : forall
  (two_step_reachable :
    forall aut target input pos p1 p2 p3,
      reachable_in_one_step aut target input pos p1 p2 ->
      reachable_in_one_step aut target input (S pos) p2 p3 ->
      exists fuel st,
        In p1 (state_positions st) ->
        In p3 (state_positions (run_automaton_from aut target input pos st fuel)))
  aut target input pos p1 p2 p3,
  reachable_in_one_step aut target input pos p1 p2 ->
  reachable_in_one_step aut target input (S pos) p2 p3 ->
  exists fuel, exists st,
    In p1 (state_positions st) ->
    In p3 (state_positions (run_automaton_from aut target input pos st fuel)).
Proof.
  intros two_step_reachable aut target input pos p1 p2 p3 H12 H23.
  apply (two_step_reachable aut target input pos p1 p2 p3 H12 H23).
Qed.

(** ** Pruning Soundness *)

Lemma prune_subsumed_positions_retains_or_subsumed : forall positions p,
  In p positions ->
  In p (prune_subsumed_positions positions) \/
  exists p', In p' (prune_subsumed_positions positions) /\
    position_subsumes p' p = true.
Proof.
  induction positions as [| h rest IH]; intros p Hin; simpl in *.
  - contradiction.
  - destruct (existsb (fun p' : Position => position_subsumes p' h)
                      (prune_subsumed_positions rest)) eqn:Hsubsumed.
    + destruct Hin as [Heq | Hin].
      * subst h.
        right.
        apply existsb_exists in Hsubsumed.
        destruct Hsubsumed as [p' [Hin' Hsub]].
        exists p'. split; assumption.
      * destruct (IH p Hin) as [Hin_pruned | [p' [Hin' Hsub]]].
        -- left. exact Hin_pruned.
        -- right. exists p'. split; assumption.
    + destruct Hin as [Heq | Hin].
      * subst h. left. left. reflexivity.
      * destruct (IH p Hin) as [Hin_pruned | [p' [Hin' Hsub]]].
        -- destruct (position_subsumes h p) eqn:Hhp.
           ++ right. exists h. split; [left; reflexivity | exact Hhp].
           ++ left. right. apply filter_In. split.
              ** exact Hin_pruned.
              ** rewrite Hhp. reflexivity.
        -- destruct (position_subsumes h p') eqn:Hhp'.
           ++ right. exists h. split; [left; reflexivity |].
              eapply position_subsumes_trans; eauto.
           ++ right. exists p'. split.
              ** right. apply filter_In. split.
                 --- exact Hin'.
                 --- rewrite Hhp'. reflexivity.
              ** exact Hsub.
Qed.

(** Pruning removes only subsumed positions *)
Theorem prune_only_subsumed : forall st p,
  In p (state_positions st) ->
  ~In p (state_positions (prune_state st)) ->
  exists p', In p' (state_positions (prune_state st)) /\
    position_subsumes p' p = true.
Proof.
  intros st p Hin Hnin.
  unfold prune_state in *. simpl in *.
  destruct (prune_subsumed_positions_retains_or_subsumed
              (state_positions st) p Hin) as [Hin_pruned | Hsub].
  - exfalso. apply Hnin. exact Hin_pruned.
  - exact Hsub.
Qed.

Lemma prune_subsumed_positions_minimal : forall positions p1 p2,
  In p1 (prune_subsumed_positions positions) ->
  In p2 (prune_subsumed_positions positions) ->
  p1 <> p2 ->
  position_subsumes p1 p2 = false.
Proof.
  induction positions as [| h rest IH]; intros p1 p2 Hin1 Hin2 Hneq; simpl in *.
  - contradiction.
  - destruct (existsb (fun p' : Position => position_subsumes p' h)
                      (prune_subsumed_positions rest)) eqn:Hsubsumed.
    + eapply IH; eauto.
    + destruct Hin1 as [Heq1 | Hin1];
      destruct Hin2 as [Heq2 | Hin2].
      * subst p1 p2. contradiction.
      * subst p1.
        apply filter_In in Hin2 as [_ Hnot_sub].
        destruct (position_subsumes h p2); simpl in Hnot_sub; congruence.
      * subst p2.
        apply filter_In in Hin1 as [Hin1_pruned _].
        destruct (position_subsumes p1 h) eqn:Hsub.
        -- exfalso.
           assert (Hexists :
             existsb (fun p' : Position => position_subsumes p' h)
               (prune_subsumed_positions rest) = true).
           { apply existsb_exists. exists p1. split; assumption. }
           congruence.
        -- reflexivity.
      * apply filter_In in Hin1 as [Hin1_pruned _].
        apply filter_In in Hin2 as [Hin2_pruned _].
        eapply IH; eauto.
Qed.

(** Pruned positions are not subsumed by each other *)
Theorem pruned_positions_minimal : forall st p1 p2,
  In p1 (state_positions (prune_state st)) ->
  In p2 (state_positions (prune_state st)) ->
  p1 <> p2 ->
  position_subsumes p1 p2 = false.
Proof.
  intros st p1 p2 Hin1 Hin2 Hneq.
  unfold prune_state in *. simpl in *.
  apply (prune_subsumed_positions_minimal (state_positions st)); assumption.
Qed.

(** ** Operation Composition *)

(** Operation application position differences compose additively.
    If op1 moves position by d1 and op2 moves position by d2,
    then composing op1 followed by op2 moves position by at most d1 + d2.
    This follows from the definition where each operation advances position
    by exactly op_consume_x. *)
Lemma operation_position_diff_additive :
  forall op1 op2 target input pos p p' p'',
    In p' (apply_operation_to_position op1 target input pos pos p) ->
    In p'' (apply_operation_to_position op2 target input (pos + op_consume_x op1) pos p') ->
    pos_i p'' - pos_i p <= op_consume_x op1 + op_consume_x op2.
Proof.
  intros op1 op2 target input pos p p' p'' Hin' Hin''.
  unfold apply_operation_to_position in Hin'.
  destruct (can_apply op1 target input (pos_i p) pos) eqn:Hcan1;
    simpl in Hin'.
  - destruct Hin' as [Heq' | Hcontra]; [subst p' | contradiction].
    simpl in Hin''.
    unfold apply_operation_to_position in Hin''.
    simpl in Hin''.
    destruct (can_apply op2 target input
      (pos_i p + op_consume_x op1) (pos + op_consume_x op1)) eqn:Hcan2;
      simpl in Hin''.
    + inversion Hin''; subst; simpl in *; try contradiction; lia.
    + contradiction.
  - contradiction.
Qed.

(** Composing two operations *)
Definition compose_positions
    (op1 op2 : OperationType)
    (target input : string)
    (pos : nat)
    (p : Position)
    : list Position :=
  flat_map
    (apply_operation_to_position op2 target input (pos + op_consume_x op1) pos)
    (apply_operation_to_position op1 target input pos pos p).

(** Composition preserves distance bound *)
Theorem composition_preserves_bound : forall op1 op2 target input pos p p'',
  bounded_diagonal 1 op1 ->
  bounded_diagonal 1 op2 ->
  In p'' (compose_positions op1 op2 target input pos p) ->
  pos_i p'' - pos_i p <= op_consume_x op1 + op_consume_x op2.
Proof.
  intros op1 op2 target input pos p p'' Hbd1 Hbd2 Hin''.
  unfold compose_positions in Hin''.
  apply in_flat_map in Hin''.
  destruct Hin'' as [p' [Hin' Hin'']].
  (* Apply the additive position-difference lemma. *)
  apply (operation_position_diff_additive op1 op2 target input pos p p' p'' Hin' Hin'').
Qed.

(** ** State Equivalence *)

(** Two states are equivalent if they have same positions (modulo order) *)
Definition states_equivalent (st1 st2 : GeneralizedState) : Prop :=
  forall p, In p (state_positions st1) <-> In p (state_positions st2).

(** State equivalence is reflexive *)
Theorem states_equiv_refl : forall st,
  states_equivalent st st.
Proof.
  intros st p. split; auto.
Qed.

(** State equivalence is symmetric *)
Theorem states_equiv_sym : forall st1 st2,
  states_equivalent st1 st2 -> states_equivalent st2 st1.
Proof.
  intros st1 st2 Heq p. split; apply Heq.
Qed.

(** State equivalence is transitive *)
Theorem states_equiv_trans : forall st1 st2 st3,
  states_equivalent st1 st2 ->
  states_equivalent st2 st3 ->
  states_equivalent st1 st3.
Proof.
  intros st1 st2 st3 H12 H23 p.
  split; intros H.
  - apply H23. apply H12. assumption.
  - apply H12. apply H23. assumption.
Qed.

(** Helper: existsb respects membership equivalence.
    If two lists have the same elements (via iff), existsb f returns the same. *)
Lemma existsb_equiv : forall {A : Type} (f : A -> bool) (l1 l2 : list A),
  (forall x, In x l1 <-> In x l2) ->
  existsb f l1 = existsb f l2.
Proof.
  intros A f l1 l2 Hequiv.
  destruct (existsb f l1) eqn:Hex1; destruct (existsb f l2) eqn:Hex2; auto.
  - (* true = false: contradiction *)
    apply existsb_exists in Hex1.
    destruct Hex1 as [x [Hin1 Hf]].
    apply Hequiv in Hin1.
    assert (Hex2': existsb f l2 = true).
    { apply existsb_exists. exists x. split; assumption. }
    congruence.
  - (* false = true: contradiction *)
    apply existsb_exists in Hex2.
    destruct Hex2 as [x [Hin2 Hf]].
    apply Hequiv in Hin2.
    assert (Hex1': existsb f l1 = true).
    { apply existsb_exists. exists x. split; assumption. }
    congruence.
Qed.

(** Equivalent states have same acceptance *)
Theorem equiv_states_same_acceptance : forall st1 st2 word_length,
  states_equivalent st1 st2 ->
  is_accepting_state word_length st1 = is_accepting_state word_length st2.
Proof.
  intros st1 st2 word_length Heq.
  unfold is_accepting_state.
  apply existsb_equiv.
  exact Heq.
Qed.

(** Contract: Equivalent input states produce equivalent delta outputs.
    When two states st1 and st2 have the same positions (membership equivalence),
    applying delta to each produces states with the same positions.
    This follows because:
    1. flat_map applies the same operations to equivalent position sets
    2. filter keeps the same positions (based on error bound)
    3. prune_state removes the same subsumed positions
    The key insight is that all three operations depend only on position membership,
    not on the order or specific list representation.

    Citation: Coq.Lists.List.flat_map/filter membership lemmas plus the
    subsumption relation from Mitankin, Mihov, and Schulz, "Deciding Word
    Neighborhood with Universal Neighborhood Automata", TCS 412(22):2340-2355,
    2011, DOI 10.1016/j.tcs.2011.01.013. *)
(** Transition preserves equivalence *)
Theorem delta_preserves_equivalence : forall
  (delta_equiv_preservation :
    forall aut target input pos st1 st2 p,
      (forall q, In q (state_positions st1) <-> In q (state_positions st2)) ->
      (In p (state_positions (delta aut target input pos st1)) <->
       In p (state_positions (delta aut target input pos st2))))
  aut target input pos st1 st2,
  states_equivalent st1 st2 ->
  states_equivalent
    (delta aut target input pos st1)
    (delta aut target input pos st2).
Proof.
  intros delta_equiv_preservation aut target input pos st1 st2 Heq.
  unfold states_equivalent in *.
  intros p.
  apply delta_equiv_preservation.
  exact Heq.
Qed.

(** ** Performance Properties *)

(** flat_map length is bounded by sum of mapped lengths.
    When flat_map f l produces a list, its length is bounded by
    the number of elements in l times the maximum length produced
    by f on any element. For delta, each position and operation pair
    produces at most 1 new position. *)
Lemma flat_map_length_bound :
  forall {A B : Type} (f : A -> list B) (l : list A) max_len,
    (forall a, In a l -> length (f a) <= max_len) ->
    length (flat_map f l) <= length l * max_len.
Proof.
  intros A B f l max_len Hbound.
  induction l as [| a l IH]; simpl.
  - lia.
  - rewrite app_length.
    assert (Ha: length (f a) <= max_len).
    { apply Hbound. left. reflexivity. }
    assert (Htail: length (flat_map f l) <= length l * max_len).
    { apply IH. intros x Hinx. apply Hbound. right. exact Hinx. }
    lia.
Qed.

(** apply_operation_to_position produces at most one position.
    For each operation and position, can_apply is checked and at most
    one resulting position is produced (either empty list or singleton). *)
Lemma apply_operation_produces_at_most_one :
  forall op target input tpos ipos p,
    length (apply_operation_to_position op target input tpos ipos p) <= 1.
Proof.
  intros op target input tpos ipos p.
  unfold apply_operation_to_position.
  destruct (can_apply op target input (pos_i p) tpos); simpl; lia.
Qed.

(** Helper: filter length is bounded by input length *)
Lemma filter_length_bound : forall {A : Type} (f : A -> bool) (l : list A),
  length (filter f l) <= length l.
Proof.
  intros A f l.
  induction l as [| x l' IH].
  - simpl. lia.
  - simpl. destruct (f x).
    + simpl. lia.
    + lia.
Qed.

(** prune_subsumed_positions length is bounded by input length.
    The prune function only removes positions, never adds new ones,
    so the result length is at most the input length. *)
Lemma prune_subsumed_length_bound :
  forall positions,
    length (prune_subsumed_positions positions) <= length positions.
Proof.
  induction positions as [| p rest IH]; simpl.
  - lia.
  - destruct (existsb (fun p' : Position => position_subsumes p' p)
                      (prune_subsumed_positions rest)) eqn:Hsubsumed.
    + lia.
    + simpl.
      pose proof (filter_length_bound
        (fun p' : Position => negb (position_subsumes p p'))
        (prune_subsumed_positions rest)) as Hfilter.
      lia.
Qed.

(** fold_left sum of bounded values has bounded total.
    If each element contributes at most max_val to the sum,
    then the total is at most length * max_val. *)
Lemma fold_left_sum_bound :
  forall {A : Type} (f : A -> nat) (l : list A) max_val,
    (forall a, In a l -> f a <= max_val) ->
    fold_left (fun acc a => acc + f a) l 0 <= length l * max_val.
Proof.
  intros A f l.
  induction l as [| a l IH]; intros max_val Hbound; simpl.
  - lia.
  - rewrite fold_left_add_shift.
    assert (Ha: f a <= max_val).
    { apply Hbound. left. reflexivity. }
    assert (Htail:
      fold_left (fun acc a => acc + f a) l 0 <= length l * max_val).
    { apply IH. intros x Hinx. apply Hbound. right. exact Hinx. }
    lia.
Qed.

(** Number of positions after transition is bounded *)
Theorem delta_position_count_bounded : forall aut target input pos st,
  length (state_positions (delta aut target input pos st)) <=
  length (state_positions st) * length (automaton_operations aut).
Proof.
  intros aut target input pos st.
  unfold delta.
  (* delta produces: prune_state (mkState (filter ... (flat_map (apply_all_operations ...) positions)) false max_dist) *)
  (* prune_state only removes positions, so length decreases or stays same *)
  assert (Hprune: length (state_positions (prune_state
    (mkState (filter (fun p => pos_e p <=? automaton_max_distance aut)
      (flat_map (apply_all_operations (automaton_operations aut) target input pos pos)
        (state_positions st)))
      false (automaton_max_distance aut)))) <=
    length (filter (fun p => pos_e p <=? automaton_max_distance aut)
      (flat_map (apply_all_operations (automaton_operations aut) target input pos pos)
        (state_positions st)))).
  { simpl. apply prune_subsumed_length_bound. }
  (* filter only removes elements *)
  assert (Hfilter: length (filter (fun p => pos_e p <=? automaton_max_distance aut)
      (flat_map (apply_all_operations (automaton_operations aut) target input pos pos)
        (state_positions st))) <=
    length (flat_map (apply_all_operations (automaton_operations aut) target input pos pos)
        (state_positions st))).
  { apply filter_length_bound. }
  (* flat_map length bound: each position produces at most |ops| positions *)
  assert (Hflat: length (flat_map (apply_all_operations (automaton_operations aut) target input pos pos)
        (state_positions st)) <=
    length (state_positions st) * length (automaton_operations aut)).
  {
    apply flat_map_length_bound.
    intros p Hin.
    (* Each position p produces at most |ops| new positions via apply_all_operations *)
    rewrite apply_all_operations_accumulates.
    (* The fold_left sums lengths, each at most 1 *)
    (* Simplify: total <= length ops * 1 = length ops *)
    assert (H: fold_left (fun acc op =>
        acc + length (apply_operation_to_position op target input pos pos p))
        (automaton_operations aut) 0 <=
        length (automaton_operations aut) * 1).
    {
      apply fold_left_sum_bound.
      intros op Hin_op.
      apply apply_operation_produces_at_most_one.
    }
    lia.
  }
  lia.
Qed.

(** Pruning reduces state size *)
Theorem prune_reduces_size : forall st,
  length (state_positions (prune_state st)) <= length (state_positions st).
Proof.
  intros st.
  unfold prune_state. simpl.
  apply prune_subsumed_length_bound.
Qed.

(** ** Transition Completeness *)

(** Contract: Edit sequences induce accepting automaton runs.
    Given an edit sequence with cost within the maximum distance,
    the automaton can be run to produce a state that either:
    1. Contains a position at the end of the target with bounded error, or
    2. Is an accepting state.
    This is the completeness property of the Levenshtein automaton construction.

    Citation: Schulz and Mihov, "Fast string correction with Levenshtein
    automata", IJDAR 5(1):67-85, 2002, DOI 10.1007/s10032-002-0082-8;
    generalized-operation extension follows Mitankin, Mihov, and Schulz,
    "Deciding Word Neighborhood with Universal Neighborhood Automata",
    TCS 412(22):2340-2355, 2011, DOI 10.1016/j.tcs.2011.01.013. *)
(** If a string is within distance, transition path exists *)
Theorem transition_path_exists : forall
  (edit_sequence_induces_accepting_run :
    forall aut target input edit_seq,
      edit_distance edit_seq <= automaton_max_distance aut ->
      exists fuel st_init st_final,
        st_final = run_automaton_from aut target input 0 st_init fuel /\
        (exists p, In p (state_positions st_final) /\
           pos_i p = String.length target /\
           pos_e p <= automaton_max_distance aut) \/
        is_accepting_state (String.length target) st_final = true)
  aut target (input : string),
  (exists edit_seq, edit_distance edit_seq <= automaton_max_distance aut) ->
  exists (fuel : nat) (st_final : GeneralizedState),
    (exists p, In p (state_positions st_final) /\
       pos_i p = String.length target /\
       pos_e p <= automaton_max_distance aut) \/
    is_accepting_state (String.length target) st_final = true.
Proof.
  intros edit_sequence_induces_accepting_run aut target input [edit_seq Hdist].
  destruct (edit_sequence_induces_accepting_run aut target input edit_seq Hdist)
    as [fuel [st_init [st_final Hdisj]]].
  exists fuel, st_final.
  destruct Hdisj as [[Hrun Hexists] | Haccepting].
  - left. assumption.
  - right. assumption.
Qed.
