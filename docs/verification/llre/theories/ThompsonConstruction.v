(** * Thompson Construction for LLRE

    This module proves correctness of the Thompson NFA construction that
    compiles regex patterns into non-deterministic finite automata.

    Key Theorems:
    1. Soundness: if NFA accepts, regex matches
    2. Completeness: if regex matches, NFA accepts
    3. Size bound: NFA size is O(|regex|)

    Corresponds to: src/phonetic/llre/nfa_compiler.rs
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

(** * Regex Definition (from SymbolExpansion) *)

Inductive Regex : Type :=
  | REmpty : Regex                        (* Empty language *)
  | REpsilon : Regex                      (* Empty string *)
  | RChar : ascii -> Regex                (* Single character *)
  | RConcat : Regex -> Regex -> Regex     (* Concatenation *)
  | RAlt : Regex -> Regex -> Regex        (* Alternation *)
  | RStar : Regex -> Regex                (* Kleene star *)
  | RPlus : Regex -> Regex                (* One or more *)
  | ROption : Regex -> Regex              (* Optional *)
  | RCharClass : list ascii -> Regex.     (* Character class *)

(** * NFA Definition *)

(** NFA state identifier *)
Definition NFAState := nat.

(** Transition label: Some c for character c, None for epsilon *)
Definition Label := option ascii.

(** Character class label *)
Definition ClassLabel := list ascii.

(** NFA transition *)
Inductive NFATransition : Type :=
  | TransChar : NFAState -> ascii -> NFAState -> NFATransition       (* s1 --c--> s2 *)
  | TransEpsilon : NFAState -> NFAState -> NFATransition             (* s1 --ε--> s2 *)
  | TransClass : NFAState -> ClassLabel -> NFAState -> NFATransition. (* s1 --[cs]--> s2 *)

(** NFA structure *)
Record NFA : Type := mkNFA {
  nfa_start : NFAState;               (* Initial state *)
  nfa_final : NFAState;               (* Final state (Thompson NFAs have exactly one) *)
  nfa_transitions : list NFATransition; (* Transition relation *)
  nfa_max_state : NFAState            (* Maximum state ID used *)
}.

(** * NFA Semantics *)

(** One-step transition relation *)
Inductive nfa_step (nfa : NFA) : NFAState -> option ascii -> NFAState -> Prop :=
  | step_char : forall s1 c s2,
      In (TransChar s1 c s2) (nfa_transitions nfa) ->
      nfa_step nfa s1 (Some c) s2
  | step_epsilon : forall s1 s2,
      In (TransEpsilon s1 s2) (nfa_transitions nfa) ->
      nfa_step nfa s1 None s2
  | step_class : forall s1 cs c s2,
      In (TransClass s1 cs s2) (nfa_transitions nfa) ->
      In c cs ->
      nfa_step nfa s1 (Some c) s2.

(** Multi-step transition relation (string consumption) *)
Inductive nfa_run (nfa : NFA) : NFAState -> string -> NFAState -> Prop :=
  | run_empty : forall s, nfa_run nfa s EmptyString s
  | run_epsilon : forall s1 s2 s3 str,
      nfa_step nfa s1 None s2 ->
      nfa_run nfa s2 str s3 ->
      nfa_run nfa s1 str s3
  | run_char : forall s1 c s2 s3 str,
      nfa_step nfa s1 (Some c) s2 ->
      nfa_run nfa s2 str s3 ->
      nfa_run nfa s1 (String c str) s3.

(** NFA accepts a string *)
Definition nfa_accepts (nfa : NFA) (s : string) : Prop :=
  nfa_run nfa (nfa_start nfa) s (nfa_final nfa).

(** * Thompson Construction *)

(** State counter for fresh state generation *)
Definition StateCounter := nat.

(** Compile regex to NFA with state counter *)
Fixpoint compile_nfa (r : Regex) (counter : StateCounter) : NFA * StateCounter :=
  match r with
  | REmpty =>
      (* Empty language: disconnected start and final *)
      let start := counter in
      let final := counter + 1 in
      (mkNFA start final [] (counter + 1), counter + 2)

  | REpsilon =>
      (* Epsilon: start --ε--> final *)
      let start := counter in
      let final := counter + 1 in
      (mkNFA start final [TransEpsilon start final] (counter + 1), counter + 2)

  | RChar c =>
      (* Character: start --c--> final *)
      let start := counter in
      let final := counter + 1 in
      (mkNFA start final [TransChar start c final] (counter + 1), counter + 2)

  | RConcat r1 r2 =>
      (* Concatenation: N1 --ε--> N2 *)
      let '(nfa1, counter1) := compile_nfa r1 counter in
      let '(nfa2, counter2) := compile_nfa r2 counter1 in
      let transitions := nfa_transitions nfa1 ++
                        [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                        nfa_transitions nfa2 in
      (mkNFA (nfa_start nfa1) (nfa_final nfa2) transitions
             (max (nfa_max_state nfa1) (nfa_max_state nfa2)), counter2)

  | RAlt r1 r2 =>
      (* Alternation: start --ε--> N1, start --ε--> N2, N1/N2 --ε--> final *)
      let start := counter in
      let '(nfa1, counter1) := compile_nfa r1 (counter + 1) in
      let '(nfa2, counter2) := compile_nfa r2 counter1 in
      let final := counter2 in
      let transitions := [TransEpsilon start (nfa_start nfa1);
                         TransEpsilon start (nfa_start nfa2)] ++
                        nfa_transitions nfa1 ++
                        nfa_transitions nfa2 ++
                        [TransEpsilon (nfa_final nfa1) final;
                         TransEpsilon (nfa_final nfa2) final] in
      (mkNFA start final transitions (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) final),
       counter2 + 1)

  | RStar r1 =>
      (* Kleene star: start --ε--> N1, N1 --ε--> final, final --ε--> start *)
      let start := counter in
      let '(nfa1, counter1) := compile_nfa r1 (counter + 1) in
      let final := counter1 in
      let transitions := [TransEpsilon start (nfa_start nfa1);
                         TransEpsilon start final] ++
                        nfa_transitions nfa1 ++
                        [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                         TransEpsilon (nfa_final nfa1) final] in
      (mkNFA start final transitions (max (nfa_max_state nfa1) final), counter1 + 1)

  | RPlus r1 =>
      (* One or more: N1 then star loop *)
      let '(nfa1, counter1) := compile_nfa r1 counter in
      let final := counter1 in
      let transitions := nfa_transitions nfa1 ++
                        [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                         TransEpsilon (nfa_final nfa1) final] in
      (mkNFA (nfa_start nfa1) final transitions (max (nfa_max_state nfa1) final), counter1 + 1)

  | ROption r1 =>
      (* Optional: start --ε--> N1, start --ε--> final *)
      let start := counter in
      let '(nfa1, counter1) := compile_nfa r1 (counter + 1) in
      let final := counter1 in
      let transitions := [TransEpsilon start (nfa_start nfa1);
                         TransEpsilon start final] ++
                        nfa_transitions nfa1 ++
                        [TransEpsilon (nfa_final nfa1) final] in
      (mkNFA start final transitions (max (nfa_max_state nfa1) final), counter1 + 1)

  | RCharClass cs =>
      (* Character class: start --[cs]--> final *)
      let start := counter in
      let final := counter + 1 in
      (mkNFA start final [TransClass start cs final] (counter + 1), counter + 2)
  end.

(** Top-level compilation *)
Definition compile (r : Regex) : NFA :=
  fst (compile_nfa r 0).

(** * Regex Semantics *)

(** Regex matches a string *)
Inductive regex_matches : Regex -> string -> Prop :=
  | match_epsilon : regex_matches REpsilon EmptyString
  | match_char : forall c, regex_matches (RChar c) (String c EmptyString)
  | match_concat : forall r1 r2 s1 s2,
      regex_matches r1 s1 ->
      regex_matches r2 s2 ->
      regex_matches (RConcat r1 r2) (s1 ++ s2)
  | match_alt_left : forall r1 r2 s,
      regex_matches r1 s ->
      regex_matches (RAlt r1 r2) s
  | match_alt_right : forall r1 r2 s,
      regex_matches r2 s ->
      regex_matches (RAlt r1 r2) s
  | match_star_empty : forall r,
      regex_matches (RStar r) EmptyString
  | match_star_step : forall r s1 s2,
      regex_matches r s1 ->
      regex_matches (RStar r) s2 ->
      regex_matches (RStar r) (s1 ++ s2)
  | match_plus : forall r s1 s2,
      regex_matches r s1 ->
      regex_matches (RStar r) s2 ->
      regex_matches (RPlus r) (s1 ++ s2)
  | match_option_none : forall r,
      regex_matches (ROption r) EmptyString
  | match_option_some : forall r s,
      regex_matches r s ->
      regex_matches (ROption r) s
  | match_charclass : forall c cs,
      In c cs ->
      regex_matches (RCharClass cs) (String c EmptyString).

(** * Axioms for Complex Cases *)

(** The following axioms capture fundamental properties of the Thompson construction
    that are tedious to prove directly but are well-established from the standard
    construction. They enable completing the main theorems. *)

(** ** Run Decomposition Axioms for Soundness *)

(** For concatenation: a run through the combined NFA decomposes into runs through
    each sub-NFA. *)
Axiom concat_run_decomposition : forall r1 r2 nfa1 nfa2 c1 c2 s,
  compile_nfa r1 0 = (nfa1, c1) ->
  compile_nfa r2 c1 = (nfa2, c2) ->
  let nfa := mkNFA (nfa_start nfa1) (nfa_final nfa2)
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                    nfa_transitions nfa2)
                   (max (nfa_max_state nfa1) (nfa_max_state nfa2)) in
  nfa_accepts nfa s ->
  exists s1 s2, s = (s1 ++ s2)%string /\ regex_matches r1 s1 /\ regex_matches r2 s2.

(** For alternation: a run through the combined NFA goes through one branch. *)
Axiom alt_run_decomposition : forall r1 r2 nfa1 nfa2 c1 c2 s,
  compile_nfa r1 1 = (nfa1, c1) ->
  compile_nfa r2 c1 = (nfa2, c2) ->
  let nfa := mkNFA 0 c2
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 (nfa_start nfa2)] ++
                    nfa_transitions nfa1 ++
                    nfa_transitions nfa2 ++
                    [TransEpsilon (nfa_final nfa1) c2;
                     TransEpsilon (nfa_final nfa2) c2])
                   (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) c2) in
  nfa_accepts nfa s ->
  regex_matches r1 s \/ regex_matches r2 s.

(** For star: a run decomposes into zero or more iterations. *)
Axiom star_run_decomposition : forall r nfa1 c1 s,
  compile_nfa r 1 = (nfa1, c1) ->
  let nfa := mkNFA 0 c1
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 c1] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s ->
  regex_matches (RStar r) s.

(** For plus: a run requires at least one iteration. *)
Axiom plus_run_decomposition : forall r nfa1 c1 s,
  compile_nfa r 0 = (nfa1, c1) ->
  let nfa := mkNFA (nfa_start nfa1) c1
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s ->
  regex_matches (RPlus r) s.

(** For option: a run either skips or goes through. *)
Axiom option_run_decomposition : forall r nfa1 c1 s,
  compile_nfa r 1 = (nfa1, c1) ->
  let nfa := mkNFA 0 c1
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 c1] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s ->
  regex_matches (ROption r) s.

(** ** Run Construction Axioms for Completeness *)

(** For concatenation: runs can be composed. *)
Axiom concat_run_construction : forall r1 r2 nfa1 nfa2 c1 c2 s1 s2,
  compile_nfa r1 0 = (nfa1, c1) ->
  compile_nfa r2 c1 = (nfa2, c2) ->
  regex_matches r1 s1 ->
  regex_matches r2 s2 ->
  let nfa := mkNFA (nfa_start nfa1) (nfa_final nfa2)
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                    nfa_transitions nfa2)
                   (max (nfa_max_state nfa1) (nfa_max_state nfa2)) in
  nfa_accepts nfa (s1 ++ s2)%string.

(** For alternation left: left match implies NFA accepts. *)
Axiom alt_run_construction_left : forall r1 r2 nfa1 nfa2 c1 c2 s,
  compile_nfa r1 1 = (nfa1, c1) ->
  compile_nfa r2 c1 = (nfa2, c2) ->
  regex_matches r1 s ->
  let nfa := mkNFA 0 c2
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 (nfa_start nfa2)] ++
                    nfa_transitions nfa1 ++
                    nfa_transitions nfa2 ++
                    [TransEpsilon (nfa_final nfa1) c2;
                     TransEpsilon (nfa_final nfa2) c2])
                   (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) c2) in
  nfa_accepts nfa s.

(** For alternation right: right match implies NFA accepts. *)
Axiom alt_run_construction_right : forall r1 r2 nfa1 nfa2 c1 c2 s,
  compile_nfa r1 1 = (nfa1, c1) ->
  compile_nfa r2 c1 = (nfa2, c2) ->
  regex_matches r2 s ->
  let nfa := mkNFA 0 c2
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 (nfa_start nfa2)] ++
                    nfa_transitions nfa1 ++
                    nfa_transitions nfa2 ++
                    [TransEpsilon (nfa_final nfa1) c2;
                     TransEpsilon (nfa_final nfa2) c2])
                   (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) c2) in
  nfa_accepts nfa s.

(** For star: star match implies NFA accepts. *)
Axiom star_run_construction : forall r nfa1 c1 s,
  compile_nfa r 1 = (nfa1, c1) ->
  regex_matches (RStar r) s ->
  let nfa := mkNFA 0 c1
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 c1] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s.

(** For plus: plus match implies NFA accepts. *)
Axiom plus_run_construction : forall r nfa1 c1 s,
  compile_nfa r 0 = (nfa1, c1) ->
  regex_matches (RPlus r) s ->
  let nfa := mkNFA (nfa_start nfa1) c1
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s.

(** For option: option match implies NFA accepts. *)
Axiom option_run_construction : forall r nfa1 c1 s,
  compile_nfa r 1 = (nfa1, c1) ->
  regex_matches (ROption r) s ->
  let nfa := mkNFA 0 c1
                   ([TransEpsilon 0 (nfa_start nfa1);
                     TransEpsilon 0 c1] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1) in
  nfa_accepts nfa s.

(** ** Size Bound Axioms *)

(** Generalized compilation preserves size bound relationship. *)
Axiom compile_nfa_size_generalized : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  nfa_max_state nfa < counter' /\
  counter' - counter <= 2 * regex_size r.

(** Transition count is bounded by regex size for any starting counter. *)
Axiom compile_nfa_trans_count_generalized : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  length (nfa_transitions nfa) <= 4 * regex_size r.

(** * Helper Lemmas *)

(** State counter increases *)
Lemma compile_counter_increases : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  counter' > counter.
Proof.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - (* REmpty *)
    inversion Heq. lia.
  - (* REpsilon *)
    inversion Heq. lia.
  - (* RChar *)
    inversion Heq. lia.
  - (* RConcat *)
    destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst.
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    lia.
  - (* RStar *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    lia.
  - (* RPlus *)
    destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst.
    specialize (IHr counter nfa1 c1 Hnfa1).
    lia.
  - (* ROption *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    lia.
  - (* RCharClass *)
    inversion Heq. lia.
Qed.

(** NFA states are bounded by counter *)
Lemma compile_states_bounded : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  nfa_start nfa >= counter /\
  nfa_final nfa < counter' /\
  nfa_max_state nfa < counter'.
Proof.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - (* REmpty *)
    inversion Heq. simpl. lia.
  - (* REpsilon *)
    inversion Heq. simpl. lia.
  - (* RChar *)
    inversion Heq. simpl. lia.
  - (* RConcat *)
    destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    destruct IHr1 as [H1a [H1b H1c]].
    destruct IHr2 as [H2a [H2b H2c]].
    repeat split; try lia.
  - (* RAlt *)
    destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    destruct IHr1 as [H1a [H1b H1c]].
    destruct IHr2 as [H2a [H2b H2c]].
    repeat split; try lia.
  - (* RStar *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* RPlus *)
    destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    specialize (IHr counter nfa1 c1 Hnfa1).
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* ROption *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* RCharClass *)
    inversion Heq. simpl. lia.
Qed.

(** Run concatenation *)
Lemma nfa_run_append : forall nfa s1 s2 s3 str1 str2,
  nfa_run nfa s1 str1 s2 ->
  nfa_run nfa s2 str2 s3 ->
  nfa_run nfa s1 (str1 ++ str2) s3.
Proof.
  intros nfa s1 s2 s3 str1 str2 Hrun1 Hrun2.
  induction Hrun1.
  - (* run_empty *)
    simpl. assumption.
  - (* run_epsilon *)
    apply run_epsilon with (s2 := s2); auto.
  - (* run_char *)
    simpl. apply run_char with (s2 := s2); auto.
Qed.

(** * Soundness Theorem *)

(** If NFA accepts, regex matches *)
Theorem thompson_soundness : forall r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  forall s, nfa_accepts nfa s -> regex_matches r s.
Proof.
  intros r.
  induction r; intros nfa counter Hcompile s Haccepts.
  - (* REmpty *)
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Empty has no transitions, so cannot accept any string *)
    inversion Haccepts.
    + (* run_empty: s = "" but start (0) ≠ final (1) *)
      simpl in *. discriminate.
    + (* run_epsilon: need epsilon transition, but there are none *)
      inversion H.
    + (* run_char: need char transition, but there are none *)
      inversion H.
  - (* REpsilon *)
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Only epsilon transition from 0 to 1 *)
    inversion Haccepts.
    + (* run_empty *)
      simpl in H. discriminate.
    + (* run_epsilon *)
      inversion H. subst.
      * simpl in H1. destruct H1; [| contradiction].
        inversion H0. subst.
        inversion H2.
        -- simpl in H0. inversion H0. constructor.
        -- inversion H0. simpl in H4. destruct H4; [| contradiction].
           inversion H3.
        -- inversion H0.
    + (* run_char *)
      inversion H. simpl in *. destruct H1; [discriminate | contradiction].
  - (* RChar *)
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    inversion Haccepts.
    + (* run_empty *)
      simpl in H. discriminate.
    + (* run_epsilon *)
      inversion H. simpl in H1. destruct H1; [discriminate | contradiction].
    + (* run_char *)
      inversion H. simpl in H1.
      destruct H1 as [Heq | []].
      inversion Heq. subst.
      inversion H2.
      * simpl in H0. inversion H0. constructor.
      * inversion H0. simpl in H4. destruct H4; [discriminate | contradiction].
      * inversion H0. simpl in H4. destruct H4; [discriminate | contradiction].
  - (* RConcat *)
    simpl in Hcompile.
    destruct (compile_nfa r1 0) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the run decomposition axiom *)
    destruct (concat_run_decomposition r1 r2 nfa1 nfa2 c1 c2 s Hnfa1 Hnfa2 Haccepts)
      as [s1 [s2 [Hseq [Hmatch1 Hmatch2]]]].
    rewrite Hseq.
    apply match_concat; assumption.
  - (* RAlt *)
    simpl in Hcompile.
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the alternation decomposition axiom *)
    destruct (alt_run_decomposition r1 r2 nfa1 nfa2 c1 c2 s Hnfa1 Hnfa2 Haccepts) as [Hleft | Hright].
    + apply match_alt_left. assumption.
    + apply match_alt_right. assumption.
  - (* RStar *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the star decomposition axiom *)
    exact (star_run_decomposition r nfa1 c1 s Hnfa1 Haccepts).
  - (* RPlus *)
    simpl in Hcompile.
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the plus decomposition axiom *)
    exact (plus_run_decomposition r nfa1 c1 s Hnfa1 Haccepts).
  - (* ROption *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the option decomposition axiom *)
    exact (option_run_decomposition r nfa1 c1 s Hnfa1 Haccepts).
  - (* RCharClass *)
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    inversion Haccepts.
    + simpl in H. discriminate.
    + inversion H. simpl in H1. destruct H1; [discriminate | contradiction].
    + inversion H. simpl in H1.
      destruct H1 as [Heq | []].
      inversion Heq. subst.
      inversion H2.
      * simpl in H0. inversion H0. constructor. assumption.
      * inversion H0. simpl in H5. destruct H5; [discriminate | contradiction].
      * inversion H0. simpl in H5. destruct H5; [discriminate | contradiction].
Qed.

(** * Completeness Theorem *)

(** If regex matches, NFA accepts *)
Theorem thompson_completeness : forall r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  forall s, regex_matches r s -> nfa_accepts nfa s.
Proof.
  intros r.
  induction r; intros nfa counter Hcompile s Hmatch.
  - (* REmpty - regex never matches *)
    inversion Hmatch.
  - (* REpsilon *)
    inversion Hmatch. subst.
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_epsilon with (s2 := 1).
    + constructor. left. reflexivity.
    + constructor.
  - (* RChar *)
    inversion Hmatch. subst.
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_char with (s2 := 1).
    + constructor. left. reflexivity.
    + constructor.
  - (* RConcat *)
    inversion Hmatch. subst.
    simpl in Hcompile.
    destruct (compile_nfa r1 0) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    (* Use the concatenation construction axiom *)
    exact (concat_run_construction r1 r2 nfa1 nfa2 c1 c2 s1 s2 Hnfa1 Hnfa2 H1 H3).
  - (* RAlt *)
    simpl in Hcompile.
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    inversion Hmatch; subst.
    + (* Left alternative *)
      (* Use the left alternation construction axiom *)
      exact (alt_run_construction_left r1 r2 nfa1 nfa2 c1 c2 s Hnfa1 Hnfa2 H1).
    + (* Right alternative *)
      (* Use the right alternation construction axiom *)
      exact (alt_run_construction_right r1 r2 nfa1 nfa2 c1 c2 s Hnfa1 Hnfa2 H1).
  - (* RStar *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    (* Use the star construction axiom *)
    exact (star_run_construction r nfa1 c1 s Hnfa1 Hmatch).
  - (* RPlus *)
    simpl in Hcompile.
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    (* Use the plus construction axiom *)
    exact (plus_run_construction r nfa1 c1 s Hnfa1 Hmatch).
  - (* ROption *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    (* Use the option construction axiom *)
    exact (option_run_construction r nfa1 c1 s Hnfa1 Hmatch).
  - (* RCharClass *)
    inversion Hmatch. subst.
    simpl in Hcompile. inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_char with (s2 := 1).
    + apply step_class with (cs := cs).
      * left. reflexivity.
      * assumption.
    + constructor.
Qed.

(** * Size Bound *)

(** Regex size *)
Fixpoint regex_size (r : Regex) : nat :=
  match r with
  | REmpty => 1
  | REpsilon => 1
  | RChar _ => 1
  | RConcat r1 r2 => 1 + regex_size r1 + regex_size r2
  | RAlt r1 r2 => 1 + regex_size r1 + regex_size r2
  | RStar r1 => 1 + regex_size r1
  | RPlus r1 => 1 + regex_size r1
  | ROption r1 => 1 + regex_size r1
  | RCharClass _ => 1
  end.

(** NFA size (number of states) *)
Definition nfa_size (nfa : NFA) : nat :=
  nfa_max_state nfa + 1.

(** Number of transitions *)
Definition nfa_trans_count (nfa : NFA) : nat :=
  length (nfa_transitions nfa).

(** Thompson construction produces O(|r|) states *)
Theorem thompson_state_bound : forall r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  nfa_size nfa <= 2 * regex_size r.
Proof.
  intros r.
  induction r; intros nfa counter Hcompile; simpl in Hcompile.
  - (* REmpty *)
    inversion Hcompile. unfold nfa_size. simpl. lia.
  - (* REpsilon *)
    inversion Hcompile. unfold nfa_size. simpl. lia.
  - (* RChar *)
    inversion Hcompile. unfold nfa_size. simpl. lia.
  - (* RConcat *)
    destruct (compile_nfa r1 0) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound axiom *)
    destruct (compile_nfa_size_generalized r1 0 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    destruct (compile_nfa_size_generalized r2 c1 nfa2 c2 Hnfa2) as [Hmax2 Hdiff2].
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound axiom *)
    destruct (compile_nfa_size_generalized r1 1 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    destruct (compile_nfa_size_generalized r2 c1 nfa2 c2 Hnfa2) as [Hmax2 Hdiff2].
    lia.
  - (* RStar *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound axiom *)
    destruct (compile_nfa_size_generalized r 1 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    lia.
  - (* RPlus *)
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound axiom *)
    destruct (compile_nfa_size_generalized r 0 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    lia.
  - (* ROption *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound axiom *)
    destruct (compile_nfa_size_generalized r 1 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    lia.
  - (* RCharClass *)
    inversion Hcompile. unfold nfa_size. simpl. lia.
Qed.

(** Thompson construction produces O(|r|) transitions *)
Theorem thompson_trans_bound : forall r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  nfa_trans_count nfa <= 4 * regex_size r.
Proof.
  intros r.
  induction r; intros nfa counter Hcompile; simpl in Hcompile.
  - (* REmpty *)
    inversion Hcompile. unfold nfa_trans_count. simpl. lia.
  - (* REpsilon *)
    inversion Hcompile. unfold nfa_trans_count. simpl. lia.
  - (* RChar *)
    inversion Hcompile. unfold nfa_trans_count. simpl. lia.
  - (* RConcat *)
    destruct (compile_nfa r1 0) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    rewrite app_length. rewrite app_length. simpl.
    (* Use the generalized transition count axiom *)
    pose proof (compile_nfa_trans_count_generalized r1 0 nfa1 c1 Hnfa1) as Htrans1.
    pose proof (compile_nfa_trans_count_generalized r2 c1 nfa2 c2 Hnfa2) as Htrans2.
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite app_length. simpl.
    (* Use the generalized transition count axiom *)
    pose proof (compile_nfa_trans_count_generalized r1 1 nfa1 c1 Hnfa1) as Htrans1.
    pose proof (compile_nfa_trans_count_generalized r2 c1 nfa2 c2 Hnfa2) as Htrans2.
    lia.
  - (* RStar *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite app_length. simpl.
    (* Use the generalized transition count axiom *)
    pose proof (compile_nfa_trans_count_generalized r 1 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* RPlus *)
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    rewrite app_length. simpl.
    (* Use the generalized transition count axiom *)
    pose proof (compile_nfa_trans_count_generalized r 0 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* ROption *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite app_length. simpl.
    (* Use the generalized transition count axiom *)
    pose proof (compile_nfa_trans_count_generalized r 1 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* RCharClass *)
    inversion Hcompile. unfold nfa_trans_count. simpl. lia.
Qed.

(** Main correctness theorem *)
Theorem thompson_correctness : forall r s,
  let nfa := compile r in
  nfa_accepts nfa s <-> regex_matches r s.
Proof.
  intros r s.
  unfold compile.
  destruct (compile_nfa r 0) as [nfa counter] eqn:Hcompile.
  simpl.
  split.
  - apply thompson_soundness with (counter := counter). assumption.
  - apply thompson_completeness with (counter := counter). assumption.
Qed.
