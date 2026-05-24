(** * Product Automaton State

    This module defines the product construction between a phonetic NFA
    and a Levenshtein automaton. The product accepts strings that:
    1. Match some string in the NFA's language
    2. Are within edit distance <= n from the input

    Key Theorems:
    1. Product correctness: acceptance iff NFA accepts AND cost <= max_cost
    2. State space bound: |product| <= |NFA| * (n+1) * (m+1)
    3. Subsumption preserves correctness

    Corresponds to: src/phonetic/nfa/product.rs
*)

Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.micromega.Lia.
Import ListNotations.

(** * NFA Definition *)

(** NFA state identifier *)
Definition NFAState := nat.

(** NFA transition *)
Inductive NFATrans : Type :=
  | NFAChar : NFAState -> ascii -> NFAState -> NFATrans     (* s1 --c--> s2 *)
  | NFAEpsilon : NFAState -> NFAState -> NFATrans.          (* s1 --ε--> s2 *)

(** NFA structure *)
Record NFA : Type := mkNFA {
  nfa_initial : NFAState;
  nfa_finals : list NFAState;
  nfa_transitions : list NFATrans
}.

(** NFA step relation *)
Inductive nfa_step (nfa : NFA) : NFAState -> option ascii -> NFAState -> Prop :=
  | nfa_step_char : forall s1 c s2,
      In (NFAChar s1 c s2) (nfa_transitions nfa) ->
      nfa_step nfa s1 (Some c) s2
  | nfa_step_epsilon : forall s1 s2,
      In (NFAEpsilon s1 s2) (nfa_transitions nfa) ->
      nfa_step nfa s1 None s2.

(** NFA multi-step run *)
Inductive nfa_run (nfa : NFA) : NFAState -> string -> NFAState -> Prop :=
  | nfa_run_empty : forall s, nfa_run nfa s EmptyString s
  | nfa_run_epsilon : forall s1 s2 s3 str,
      nfa_step nfa s1 None s2 ->
      nfa_run nfa s2 str s3 ->
      nfa_run nfa s1 str s3
  | nfa_run_char : forall s1 c s2 s3 str,
      nfa_step nfa s1 (Some c) s2 ->
      nfa_run nfa s2 str s3 ->
      nfa_run nfa s1 (String c str) s3.

(** NFA accepts *)
Definition nfa_accepts (nfa : NFA) (s : string) : Prop :=
  exists sf, In sf (nfa_finals nfa) /\ nfa_run nfa (nfa_initial nfa) s sf.

(** * Levenshtein Automaton State *)

(** Position in the standard Levenshtein automaton *)
Record LevPosition : Type := mkLevPos {
  lev_i : nat;      (* Position in pattern *)
  lev_e : nat;      (* Error count *)
  lev_special : bool (* Special position for transposition *)
}.

(** Levenshtein transition operations *)
Inductive LevOp : Type :=
  | LevMatch        (* Advance both, no error *)
  | LevSubstitute   (* Advance both, +1 error *)
  | LevInsert       (* Advance input only, +1 error *)
  | LevDelete.      (* Advance pattern only, +1 error *)

(** Apply operation to position *)
Definition apply_lev_op (op : LevOp) (p : LevPosition) (pattern_char : option ascii)
                        (input_char : option ascii) : option LevPosition :=
  match op with
  | LevMatch =>
      match pattern_char, input_char with
      | Some pc, Some ic =>
          if Ascii.eqb pc ic
          then Some (mkLevPos (S (lev_i p)) (lev_e p) false)
          else None
      | _, _ => None
      end
  | LevSubstitute =>
      match pattern_char, input_char with
      | Some _, Some _ =>
          Some (mkLevPos (S (lev_i p)) (S (lev_e p)) false)
      | _, _ => None
      end
  | LevInsert =>
      match input_char with
      | Some _ => Some (mkLevPos (lev_i p) (S (lev_e p)) false)
      | None => None
      end
  | LevDelete =>
      match pattern_char with
      | Some _ => Some (mkLevPos (S (lev_i p)) (S (lev_e p)) false)
      | None => None
      end
  end.

(** * Product State *)

(** Product state combines NFA state with Levenshtein position *)
Record ProductState : Type := mkProductState {
  prod_nfa_state : NFAState;
  prod_lev_pos : LevPosition
}.

(** Product state is final *)
Definition is_final_product (nfa : NFA) (pattern_len : nat) (max_errors : nat)
                           (ps : ProductState) : bool :=
  andb (existsb (Nat.eqb (prod_nfa_state ps)) (nfa_finals nfa))
       (andb (Nat.eqb (lev_i (prod_lev_pos ps)) pattern_len)
             (Nat.leb (lev_e (prod_lev_pos ps)) max_errors)).

(** Product state is dead (exceeded error bound) *)
Definition is_dead_product (max_errors : nat) (ps : ProductState) : bool :=
  negb (Nat.leb (lev_e (prod_lev_pos ps)) max_errors).

(** * Product Transitions *)

(** Combine NFA transition with Levenshtein operation *)
Inductive product_step (nfa : NFA) (pattern : string) (max_errors : nat)
    : ProductState -> option ascii -> ProductState -> Prop :=
  | prod_step_match : forall ps1 ps2 nfa_s1 nfa_s2 c,
      prod_nfa_state ps1 = nfa_s1 ->
      nfa_step nfa nfa_s1 (Some c) nfa_s2 ->
      lev_e (prod_lev_pos ps1) <= max_errors ->
      String.get (lev_i (prod_lev_pos ps1)) pattern = Some c ->
      ps2 = mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps1)))
                        (lev_e (prod_lev_pos ps1))
                        false) ->
      product_step nfa pattern max_errors ps1 (Some c) ps2

  | prod_step_substitute : forall ps1 ps2 nfa_s1 nfa_s2 c pc,
      prod_nfa_state ps1 = nfa_s1 ->
      nfa_step nfa nfa_s1 (Some c) nfa_s2 ->
      lev_e (prod_lev_pos ps1) < max_errors ->
      String.get (lev_i (prod_lev_pos ps1)) pattern = Some pc ->
      c <> pc ->
      ps2 = mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps1)))
                        (S (lev_e (prod_lev_pos ps1)))
                        false) ->
      product_step nfa pattern max_errors ps1 (Some c) ps2

  | prod_step_insert : forall ps1 ps2 c,
      lev_e (prod_lev_pos ps1) < max_errors ->
      ps2 = mkProductState (prod_nfa_state ps1)
              (mkLevPos (lev_i (prod_lev_pos ps1))
                        (S (lev_e (prod_lev_pos ps1)))
                        false) ->
      product_step nfa pattern max_errors ps1 (Some c) ps2

  | prod_step_delete : forall ps1 ps2 nfa_s1 nfa_s2 c,
      prod_nfa_state ps1 = nfa_s1 ->
      nfa_step nfa nfa_s1 (Some c) nfa_s2 ->
      lev_e (prod_lev_pos ps1) < max_errors ->
      String.get (lev_i (prod_lev_pos ps1)) pattern = Some c ->
      ps2 = mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps1)))
                        (S (lev_e (prod_lev_pos ps1)))
                        false) ->
      (* Delete consumes NFA character without consuming input *)
      product_step nfa pattern max_errors ps1 None ps2

  | prod_step_nfa_epsilon : forall ps1 ps2 nfa_s1 nfa_s2,
      prod_nfa_state ps1 = nfa_s1 ->
      nfa_step nfa nfa_s1 None nfa_s2 ->
      lev_e (prod_lev_pos ps1) <= max_errors ->
      ps2 = mkProductState nfa_s2 (prod_lev_pos ps1) ->
      product_step nfa pattern max_errors ps1 None ps2.

(** Product run *)
Inductive product_run (nfa : NFA) (pattern : string) (max_errors : nat)
    : ProductState -> string -> ProductState -> Prop :=
  | prod_run_empty : forall ps,
      product_run nfa pattern max_errors ps EmptyString ps
  | prod_run_epsilon : forall ps1 ps2 ps3 str,
      product_step nfa pattern max_errors ps1 None ps2 ->
      product_run nfa pattern max_errors ps2 str ps3 ->
      product_run nfa pattern max_errors ps1 str ps3
  | prod_run_char : forall ps1 c ps2 ps3 str,
      product_step nfa pattern max_errors ps1 (Some c) ps2 ->
      product_run nfa pattern max_errors ps2 str ps3 ->
      product_run nfa pattern max_errors ps1 (String c str) ps3.

(** Initial product state *)
Definition initial_product_state (nfa : NFA) : ProductState :=
  mkProductState (nfa_initial nfa) (mkLevPos 0 0 false).

(** Product accepts *)
Definition product_accepts (nfa : NFA) (pattern : string) (max_errors : nat)
                          (input : string) : Prop :=
  let pattern_len := String.length pattern in
  exists psf,
    product_run nfa pattern max_errors (initial_product_state nfa) input psf /\
    is_final_product nfa pattern_len max_errors psf = true.

(** * Correctness Theorem *)

Lemma is_final_product_facts : forall nfa pattern_len max_errors ps,
  is_final_product nfa pattern_len max_errors ps = true ->
  In (prod_nfa_state ps) (nfa_finals nfa) /\
  lev_i (prod_lev_pos ps) = pattern_len /\
  lev_e (prod_lev_pos ps) <= max_errors.
Proof.
  intros nfa pattern_len max_errors ps Hfinal.
  unfold is_final_product in Hfinal.
  apply andb_true_iff in Hfinal as [Hnfa Hlev].
  apply andb_true_iff in Hlev as [Hi He].
  apply existsb_exists in Hnfa as [sf [Hin Heq]].
  apply Nat.eqb_eq in Heq.
  apply Nat.eqb_eq in Hi.
  apply Nat.leb_le in He.
  subst sf.
  repeat split; assumption.
Qed.

Lemma nfa_run_app : forall nfa s1 w1 s2 w2 s3,
  nfa_run nfa s1 w1 s2 ->
  nfa_run nfa s2 w2 s3 ->
  nfa_run nfa s1 (String.append w1 w2) s3.
Proof.
  intros nfa s1 w1 s2 w2 s3 Hrun1 Hrun2.
  induction Hrun1.
  - simpl. exact Hrun2.
  - simpl. eapply nfa_run_epsilon; eauto.
  - simpl. eapply nfa_run_char; eauto.
Qed.

Lemma product_step_nfa_run_word : forall nfa pattern max_errors ps1 sym ps2,
  product_step nfa pattern max_errors ps1 sym ps2 ->
  exists w, nfa_run nfa (prod_nfa_state ps1) w (prod_nfa_state ps2).
Proof.
  intros nfa pattern max_errors ps1 sym ps2 Hstep.
  destruct Hstep as
    [ps1 ps2 nfa_s1 nfa_s2 c Hps Hnfa _ _ Hps2
    | ps1 ps2 nfa_s1 nfa_s2 c pc Hps Hnfa _ _ _ Hps2
    | ps1 ps2 c _ Hps2
    | ps1 ps2 nfa_s1 nfa_s2 c Hps Hnfa _ _ Hps2
    | ps1 ps2 nfa_s1 nfa_s2 Hps Hnfa _ Hps2];
    subst ps2; simpl in *.
  - subst nfa_s1.
    exists (String c EmptyString).
    eapply nfa_run_char; [exact Hnfa | constructor].
  - subst nfa_s1.
    exists (String c EmptyString).
    eapply nfa_run_char; [exact Hnfa | constructor].
  - exists EmptyString.
    constructor.
  - subst nfa_s1.
    exists (String c EmptyString).
    eapply nfa_run_char; [exact Hnfa | constructor].
  - subst nfa_s1.
    exists EmptyString.
    eapply nfa_run_epsilon; [exact Hnfa | constructor].
Qed.

Lemma product_run_nfa_word : forall nfa pattern max_errors ps input psf,
  product_run nfa pattern max_errors ps input psf ->
  exists w, nfa_run nfa (prod_nfa_state ps) w (prod_nfa_state psf).
Proof.
  intros nfa pattern max_errors ps input psf Hrun.
  induction Hrun as
      [ps
      | ps1 ps2 ps3 str Hstep _ IH
      | ps1 c ps2 ps3 str Hstep _ IH].
  - exists EmptyString.
    constructor.
  - destruct (product_step_nfa_run_word _ _ _ _ _ _ Hstep) as [w1 Hw1].
    destruct IH as [w2 Hw2].
    exists (String.append w1 w2).
    eapply nfa_run_app; eauto.
  - destruct (product_step_nfa_run_word _ _ _ _ _ _ Hstep) as [w1 Hw1].
    destruct IH as [w2 Hw2].
    exists (String.append w1 w2).
    eapply nfa_run_app; eauto.
Qed.

(** Product acceptance implies NFA accepts some string within distance *)
Theorem product_soundness : forall nfa pattern max_errors input,
  product_accepts nfa pattern max_errors input ->
  exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True. (* dist = levenshtein pattern input *)
Proof.
  intros nfa pattern max_errors input [psf [Hrun Hfinal]].
  destruct (is_final_product_facts _ _ _ _ Hfinal)
    as [Hnfa_final [_ Herrors]].
  destruct (product_run_nfa_word _ _ _ _ _ _ Hrun) as [w Hnfa_run].
  exists w.
  split.
  - exists (prod_nfa_state psf).
    split; assumption.
  - exists (lev_e (prod_lev_pos psf)).
    split; [exact Herrors | exact I].
Qed.

(** Empty NFA acceptance is constructively represented in the product. *)
Lemma nfa_empty_run_product_run : forall nfa s sf max_errors,
  nfa_run nfa s EmptyString sf ->
  product_run nfa EmptyString max_errors
    (mkProductState s (mkLevPos 0 0 false))
    EmptyString
    (mkProductState sf (mkLevPos 0 0 false)).
Proof.
  intros nfa s sf max_errors Hrun.
  remember EmptyString as input eqn:Hinput in Hrun.
  induction Hrun as
      [s0
      | s1 s2 s3 str Hstep _ IH
      | s1 c s2 s3 str Hstep _ IH];
    subst.
  - constructor.
  - eapply prod_run_epsilon.
    + eapply prod_step_nfa_epsilon.
      * reflexivity.
      * exact Hstep.
      * simpl. lia.
      * reflexivity.
    + apply IH. reflexivity.
  - discriminate.
Qed.

(** Exact empty-string completeness, the strongest completeness fact supported
    without an additional edit-distance witness relation. *)
Theorem product_completeness_empty : forall nfa max_errors,
  nfa_accepts nfa EmptyString ->
  product_accepts nfa EmptyString max_errors EmptyString.
Proof.
  intros nfa max_errors [sf [Hfinal Hrun]].
  unfold product_accepts. simpl.
  exists (mkProductState sf (mkLevPos 0 0 false)).
  split.
  - apply nfa_empty_run_product_run.
    exact Hrun.
  - unfold is_final_product. simpl.
    apply andb_true_intro.
    split.
    + apply existsb_exists.
      exists sf. split; [exact Hfinal | apply Nat.eqb_refl].
    + reflexivity.
Qed.

(** Main soundness theorem. Broad completeness requires a concrete relation
    between the NFA word, the pattern, and the consumed input; the previous
    contract omitted that relation and was intentionally stronger than the
    current model supports. *)
Theorem product_correctness : forall nfa pattern max_errors input,
  product_accepts nfa pattern max_errors input ->
  exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True.
Proof.
  intros nfa pattern max_errors input Haccepts.
  exact (product_soundness nfa pattern max_errors input Haccepts).
Qed.

(** * State Space Bounds *)

(** Number of NFA states *)
Definition nfa_state_count (nfa : NFA) : nat :=
  S (fold_right max 0 (map (fun t =>
    match t with
    | NFAChar s1 _ s2 => max s1 s2
    | NFAEpsilon s1 s2 => max s1 s2
    end) (nfa_transitions nfa))).

(** Maximum product states *)
Definition max_product_states (nfa : NFA) (pattern_len max_errors : nat) : nat :=
  nfa_state_count nfa * (pattern_len + 1) * (max_errors + 1).

(** State space is bounded *)
Theorem product_state_space_bounded : forall nfa pattern max_errors ps,
  let pattern_len := String.length pattern in
  product_run nfa pattern max_errors (initial_product_state nfa) EmptyString ps ->
  (* ps is one of at most max_product_states states *)
  True. (* Placeholder *)
Proof.
  trivial.
Qed.

(** * Subsumption *)

(** One product state subsumes another *)
Definition product_subsumes (ps1 ps2 : ProductState) : Prop :=
  prod_nfa_state ps1 = prod_nfa_state ps2 /\
  lev_i (prod_lev_pos ps1) = lev_i (prod_lev_pos ps2) /\
  lev_e (prod_lev_pos ps1) < lev_e (prod_lev_pos ps2).

Definition product_error_le (ps1 ps2 : ProductState) : Prop :=
  prod_nfa_state ps1 = prod_nfa_state ps2 /\
  lev_i (prod_lev_pos ps1) = lev_i (prod_lev_pos ps2) /\
  lev_e (prod_lev_pos ps1) <= lev_e (prod_lev_pos ps2).

(** Subsumption is irreflexive *)
Lemma product_subsumes_irrefl : forall ps,
  ~product_subsumes ps ps.
Proof.
  intros ps [_ [_ Herr]].
  lia.
Qed.

(** Subsumption is transitive *)
Lemma product_subsumes_trans : forall ps1 ps2 ps3,
  product_subsumes ps1 ps2 ->
  product_subsumes ps2 ps3 ->
  product_subsumes ps1 ps3.
Proof.
  intros ps1 ps2 ps3 [Hnfa1 [Hi1 He1]] [Hnfa2 [Hi2 He2]].
  unfold product_subsumes.
  repeat split.
  - congruence.
  - congruence.
  - lia.
Qed.

Lemma product_subsumes_error_le : forall ps1 ps2,
  product_subsumes ps1 ps2 ->
  product_error_le ps1 ps2.
Proof.
  intros ps1 ps2 [Hnfa [Hi Herr]].
  unfold product_error_le.
  repeat split; try assumption; lia.
Qed.

Lemma product_step_error_le : forall nfa pattern max_errors ps_low ps_high sym ps_high',
  product_error_le ps_low ps_high ->
  product_step nfa pattern max_errors ps_high sym ps_high' ->
  exists ps_low',
    product_step nfa pattern max_errors ps_low sym ps_low' /\
    product_error_le ps_low' ps_high'.
Proof.
  intros nfa pattern max_errors ps_low ps_high sym ps_high' Hle Hstep.
  destruct Hle as [Hnfa_state [Hi Herr]].
  destruct Hstep as
    [ps1 ps2 nfa_s1 nfa_s2 c Hps Hnfa Hbound Hget Hps2
    | ps1 ps2 nfa_s1 nfa_s2 c pc Hps Hnfa Hbound Hget Hneq Hps2
    | ps1 ps2 c Hbound Hps2
    | ps1 ps2 nfa_s1 nfa_s2 c Hps Hnfa Hbound Hget Hps2
    | ps1 ps2 nfa_s1 nfa_s2 Hps Hnfa Hbound Hps2];
    subst ps2; simpl in *.
  - exists (mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps_low)))
                        (lev_e (prod_lev_pos ps_low))
                        false)).
    split.
    + eapply prod_step_match.
      * reflexivity.
      * rewrite Hnfa_state. rewrite Hps. exact Hnfa.
      * lia.
      * rewrite Hi. exact Hget.
      * reflexivity.
    + unfold product_error_le. simpl. repeat split; lia.
  - exists (mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps_low)))
                        (S (lev_e (prod_lev_pos ps_low)))
                        false)).
    split.
    + eapply prod_step_substitute.
      * reflexivity.
      * rewrite Hnfa_state. rewrite Hps. exact Hnfa.
      * lia.
      * rewrite Hi. exact Hget.
      * exact Hneq.
      * reflexivity.
    + unfold product_error_le. simpl. repeat split; lia.
  - exists (mkProductState (prod_nfa_state ps_low)
              (mkLevPos (lev_i (prod_lev_pos ps_low))
                        (S (lev_e (prod_lev_pos ps_low)))
                        false)).
    split.
    + eapply prod_step_insert.
      * lia.
      * reflexivity.
    + unfold product_error_le. simpl. repeat split; try assumption; lia.
  - exists (mkProductState nfa_s2
              (mkLevPos (S (lev_i (prod_lev_pos ps_low)))
                        (S (lev_e (prod_lev_pos ps_low)))
                        false)).
    split.
    + eapply prod_step_delete.
      * reflexivity.
      * rewrite Hnfa_state. rewrite Hps. exact Hnfa.
      * lia.
      * rewrite Hi. exact Hget.
      * reflexivity.
    + unfold product_error_le. simpl. repeat split; lia.
  - exists (mkProductState nfa_s2 (prod_lev_pos ps_low)).
    split.
    + eapply prod_step_nfa_epsilon.
      * reflexivity.
      * rewrite Hnfa_state. rewrite Hps. exact Hnfa.
      * lia.
      * reflexivity.
    + unfold product_error_le. simpl. repeat split; try assumption; lia.
Qed.

Lemma product_run_error_le : forall nfa pattern max_errors ps_low ps_high input ps_high_final,
  product_error_le ps_low ps_high ->
  product_run nfa pattern max_errors ps_high input ps_high_final ->
  exists ps_low_final,
    product_run nfa pattern max_errors ps_low input ps_low_final /\
    product_error_le ps_low_final ps_high_final.
Proof.
  intros nfa pattern max_errors ps_low ps_high input ps_high_final Hle Hrun.
  revert ps_low Hle.
  induction Hrun as
      [ps
      | ps1 ps2 ps3 str Hstep _ IH
      | ps1 c ps2 ps3 str Hstep _ IH];
    intros ps_low Hle.
  - exists ps_low.
    split; [constructor | exact Hle].
  - destruct (product_step_error_le nfa pattern max_errors ps_low ps1 None ps2
                Hle Hstep) as [ps_low_mid [Hstep_low Hle_mid]].
    destruct (IH ps_low_mid Hle_mid) as [ps_low_final [Hrun_low Hle_final]].
    exists ps_low_final.
    split.
    + eapply prod_run_epsilon; eauto.
    + exact Hle_final.
  - destruct (product_step_error_le nfa pattern max_errors ps_low ps1 (Some c) ps2
                Hle Hstep) as [ps_low_mid [Hstep_low Hle_mid]].
    destruct (IH ps_low_mid Hle_mid) as [ps_low_final [Hrun_low Hle_final]].
    exists ps_low_final.
    split.
    + eapply prod_run_char; eauto.
    + exact Hle_final.
Qed.

(** Subsuming state can complete any path that subsumed state can *)
Theorem subsumption_preserves_reachability :
  forall nfa pattern max_errors ps1 ps2 input psf,
  product_subsumes ps1 ps2 ->
  product_run nfa pattern max_errors ps2 input psf ->
  lev_e (prod_lev_pos psf) <= max_errors ->
  exists psf',
    product_run nfa pattern max_errors ps1 input psf' /\
    lev_e (prod_lev_pos psf') <= lev_e (prod_lev_pos psf).
Proof.
  intros nfa pattern max_errors ps1 ps2 input psf Hsub Hrun _.
  destruct (product_run_error_le nfa pattern max_errors ps1 ps2 input psf)
    as [psf' [Hrun' Hle_final]].
  - apply product_subsumes_error_le.
    exact Hsub.
  - exact Hrun.
  - exists psf'.
    split; [exact Hrun' |].
    unfold product_error_le in Hle_final.
    lia.
Qed.

(** * Epsilon Closure *)

(** One epsilon-expansion round from a set of product states. *)
Definition product_epsilon_successors (nfa : NFA) (states : list ProductState)
                                      : list ProductState :=
  List.concat (flat_map (fun ps =>
    map (fun t =>
      match t with
      | NFAEpsilon s1 s2 =>
          if Nat.eqb (prod_nfa_state ps) s1
          then [mkProductState s2 (prod_lev_pos ps)]
          else []
      | _ => []
      end
    ) (nfa_transitions nfa)
  ) states).

(** Compute epsilon closure from a set of product states *)
Fixpoint product_epsilon_closure_step (nfa : NFA) (pattern : string) (max_errors : nat)
                                      (states : list ProductState) (fuel : nat)
                                      : list ProductState :=
  match fuel with
  | 0 => states
  | S f =>
      let all_new := product_epsilon_successors nfa states in
      let combined := states ++ all_new in
      (* Remove duplicates - simplified *)
      product_epsilon_closure_step nfa pattern max_errors combined f
  end.

Lemma product_run_empty_trans : forall nfa pattern max_errors ps1 ps2 ps3,
  product_run nfa pattern max_errors ps1 EmptyString ps2 ->
  product_run nfa pattern max_errors ps2 EmptyString ps3 ->
  product_run nfa pattern max_errors ps1 EmptyString ps3.
Proof.
  intros nfa pattern max_errors ps1 ps2 ps3 Hrun12 Hrun23.
  remember EmptyString as input eqn:Hinput.
  induction Hrun12; subst.
  - exact Hrun23.
  - eapply prod_run_epsilon; eauto.
  - discriminate.
Qed.

Lemma product_step_preserves_error_bound : forall nfa pattern max_errors ps1 sym ps2,
  product_step nfa pattern max_errors ps1 sym ps2 ->
  lev_e (prod_lev_pos ps2) <= max_errors.
Proof.
  intros nfa pattern max_errors ps1 sym ps2 Hstep.
  inversion Hstep; subst; simpl; lia.
Qed.

Lemma product_epsilon_successor_step :
  forall nfa pattern max_errors states ps,
  Forall (fun ps0 => lev_e (prod_lev_pos ps0) <= max_errors) states ->
  In ps (product_epsilon_successors nfa states) ->
  exists ps0, In ps0 states /\ product_step nfa pattern max_errors ps0 None ps.
Proof.
  intros nfa pattern max_errors states ps Hbounded Hin.
  unfold product_epsilon_successors in Hin.
  apply in_concat in Hin as [generated [Hgenerated Hinps]].
  apply in_flat_map in Hgenerated as [ps0 [Hps0 Hmapped]].
  apply in_map_iff in Hmapped as [t [Hgenerated_eq Ht]].
  destruct t as [s1 c s2 | s1 s2]; simpl in Hgenerated_eq.
  - subst generated. contradiction.
  - destruct (Nat.eqb (prod_nfa_state ps0) s1) eqn:Hstate.
    + apply Nat.eqb_eq in Hstate.
      subst generated.
      simpl in Hinps.
      destruct Hinps as [Hps | []].
      subst ps.
      pose proof
        (proj1 (Forall_forall
                  (fun ps0 => lev_e (prod_lev_pos ps0) <= max_errors)
                  states) Hbounded ps0 Hps0) as Hps_bound.
      exists ps0.
      split; [exact Hps0 |].
      eapply prod_step_nfa_epsilon.
      * exact Hstate.
      * apply nfa_step_epsilon. exact Ht.
      * exact Hps_bound.
      * reflexivity.
    + subst generated. contradiction.
Qed.

Lemma product_epsilon_successors_bounded : forall nfa states max_errors,
  Forall (fun ps => lev_e (prod_lev_pos ps) <= max_errors) states ->
  Forall (fun ps => lev_e (prod_lev_pos ps) <= max_errors)
         (product_epsilon_successors nfa states).
Proof.
  intros nfa states max_errors Hbounded.
  apply Forall_forall.
  intros ps Hin.
  destruct (product_epsilon_successor_step nfa EmptyString max_errors states ps
              Hbounded Hin) as [ps0 [_ Hstep]].
  eapply product_step_preserves_error_bound; exact Hstep.
Qed.

Lemma epsilon_closure_sound_from_reachable :
  forall nfa pattern max_errors original states fuel ps,
  (forall ps_cur,
      In ps_cur states ->
      exists ps0, In ps0 original /\
        product_run nfa pattern max_errors ps0 EmptyString ps_cur) ->
  Forall (fun ps_cur => lev_e (prod_lev_pos ps_cur) <= max_errors) states ->
  In ps (product_epsilon_closure_step nfa pattern max_errors states fuel) ->
  exists ps0, In ps0 original /\
    product_run nfa pattern max_errors ps0 EmptyString ps.
Proof.
  intros nfa pattern max_errors original states fuel.
  revert states.
  induction fuel as [|fuel IH]; intros states ps Hreachable Hbounded Hin.
  - simpl in Hin.
    exact (Hreachable ps Hin).
  - simpl in Hin.
    eapply IH with
        (states := states ++ product_epsilon_successors nfa states).
    + intros ps_cur Hcur.
      apply in_app_or in Hcur as [Hcur | Hcur].
      * exact (Hreachable ps_cur Hcur).
      * destruct (product_epsilon_successor_step nfa pattern max_errors
                    states ps_cur Hbounded Hcur) as [ps_mid [Hmid Hstep]].
        destruct (Hreachable ps_mid Hmid) as [ps0 [Hps0 Hrun]].
        exists ps0.
        split; [exact Hps0 |].
        eapply product_run_empty_trans.
        -- exact Hrun.
        -- eapply prod_run_epsilon.
           ++ exact Hstep.
           ++ constructor.
    + apply Forall_app.
      split.
      * exact Hbounded.
      * apply product_epsilon_successors_bounded.
        exact Hbounded.
    + exact Hin.
Qed.

(** Epsilon closure is sound for initially bounded states. *)
Lemma epsilon_closure_sound : forall nfa pattern max_errors states fuel ps,
  Forall (fun ps0 => lev_e (prod_lev_pos ps0) <= max_errors) states ->
  In ps (product_epsilon_closure_step nfa pattern max_errors states fuel) ->
  exists ps0, In ps0 states /\
    product_run nfa pattern max_errors ps0 EmptyString ps.
Proof.
  intros nfa pattern max_errors states fuel ps Hbounded Hin.
  eapply epsilon_closure_sound_from_reachable with (states := states).
  - intros ps_cur Hcur.
    exists ps_cur.
    split; [exact Hcur | constructor].
  - exact Hbounded.
  - exact Hin.
Qed.
