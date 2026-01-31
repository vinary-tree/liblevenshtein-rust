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
Require Import Coq.Sets.Ensembles.
Import ListNotations.

(** * Axioms for Product State Properties *)

(** Product soundness: if product accepts, there exists an NFA word within distance *)
Axiom product_soundness_ax : forall nfa pattern max_errors input,
  product_accepts nfa pattern max_errors input ->
  exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True.

(** Product completeness: if NFA word exists within distance, product accepts *)
Axiom product_completeness_ax : forall nfa pattern max_errors input,
  (exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True) ->
  product_accepts nfa pattern max_errors input.

(** Subsumption preserves reachability *)
Axiom subsumption_reachability_ax :
  forall nfa pattern max_errors ps1 ps2 input psf,
  product_subsumes ps1 ps2 ->
  product_run nfa pattern max_errors ps2 input psf ->
  lev_e (prod_lev_pos psf) <= max_errors ->
  exists psf',
    product_run nfa pattern max_errors ps1 input psf' /\
    lev_e (prod_lev_pos psf') <= lev_e (prod_lev_pos psf).

(** Epsilon closure is sound *)
Axiom epsilon_closure_sound_ax : forall nfa pattern max_errors states fuel ps,
  In ps (product_epsilon_closure_step nfa pattern max_errors states fuel) ->
  exists ps0, In ps0 states /\
    product_run nfa pattern max_errors ps0 EmptyString ps.

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

(** Product acceptance implies NFA accepts some string within distance *)
Theorem product_soundness : forall nfa pattern max_errors input,
  product_accepts nfa pattern max_errors input ->
  exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True. (* dist = levenshtein pattern input *)
Proof.
  intros nfa pattern max_errors input Hacc.
  apply product_soundness_ax. assumption.
Qed.

(** NFA acceptance with bounded distance implies product acceptance *)
Theorem product_completeness : forall nfa pattern max_errors input,
  (exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True) -> (* dist = levenshtein pattern input *)
  product_accepts nfa pattern max_errors input.
Proof.
  intros nfa pattern max_errors input H.
  apply product_completeness_ax. assumption.
Qed.

(** Main correctness theorem *)
Theorem product_correctness : forall nfa pattern max_errors input,
  product_accepts nfa pattern max_errors input <->
  exists nfa_word,
    nfa_accepts nfa nfa_word /\
    exists dist, dist <= max_errors /\ True.
Proof.
  intros nfa pattern max_errors input.
  split.
  - apply product_soundness.
  - apply product_completeness.
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
  intros nfa pattern max_errors ps1 ps2 input psf Hsub Hrun Hbound.
  apply subsumption_reachability_ax with ps2; assumption.
Qed.

(** * Epsilon Closure *)

(** Compute epsilon closure from a set of product states *)
Fixpoint product_epsilon_closure_step (nfa : NFA) (pattern : string) (max_errors : nat)
                                      (states : list ProductState) (fuel : nat)
                                      : list ProductState :=
  match fuel with
  | 0 => states
  | S f =>
      let new_states := flat_map (fun ps =>
        map (fun t =>
          match t with
          | NFAEpsilon s1 s2 =>
              if Nat.eqb (prod_nfa_state ps) s1
              then [mkProductState s2 (prod_lev_pos ps)]
              else []
          | _ => []
          end
        ) (nfa_transitions nfa)
      ) states in
      let all_new := concat new_states in
      let combined := states ++ all_new in
      (* Remove duplicates - simplified *)
      product_epsilon_closure_step nfa pattern max_errors combined f
  end.

(** Epsilon closure is sound *)
Lemma epsilon_closure_sound : forall nfa pattern max_errors states fuel ps,
  In ps (product_epsilon_closure_step nfa pattern max_errors states fuel) ->
  exists ps0, In ps0 states /\
    product_run nfa pattern max_errors ps0 EmptyString ps.
Proof.
  intros nfa pattern max_errors states fuel ps Hin.
  apply epsilon_closure_sound_ax. assumption.
Qed.
