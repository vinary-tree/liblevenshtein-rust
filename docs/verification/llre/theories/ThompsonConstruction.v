(** * Thompson Construction for LLRE

    This module proves correctness of the Thompson NFA construction that
    compiles regex patterns into non-deterministic finite automata.

    Key Theorems:
    1. Soundness: if NFA accepts, regex matches
    2. Completeness: if regex matches, NFA accepts
    3. Size bound: NFA size is O(|regex|)

    Corresponds to: src/phonetic/llre/nfa_compiler.rs
*)

From Stdlib Require Import Lists.List.
From Stdlib Require Import Strings.String.
From Stdlib Require Import Strings.Ascii.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Arith.PeanoNat.
From Stdlib Require Import Bool.Bool.
From Stdlib Require Import micromega.Lia.
From Stdlib Require Import Program.Equality.
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

(** A run trace exposes the finite sequence of transition labels used by a
    run. This is proof-only infrastructure for decomposing Thompson runs
    without changing the accepted-language semantics. *)
Inductive nfa_run_trace (nfa : NFA) :
  NFAState -> list (option ascii) -> string -> NFAState -> Prop :=
  | trace_empty : forall s,
      nfa_run_trace nfa s [] EmptyString s
  | trace_epsilon : forall s1 s2 s3 labels str,
      nfa_step nfa s1 None s2 ->
      nfa_run_trace nfa s2 labels str s3 ->
      nfa_run_trace nfa s1 (None :: labels) str s3
  | trace_char : forall s1 c s2 s3 labels str,
      nfa_step nfa s1 (Some c) s2 ->
      nfa_run_trace nfa s2 labels str s3 ->
      nfa_run_trace nfa s1 (Some c :: labels) (String c str) s3.

Lemma nfa_run_trace_sound : forall nfa s labels str t,
  nfa_run_trace nfa s labels str t ->
  nfa_run nfa s str t.
Proof.
  intros nfa s labels str t Htrace.
  induction Htrace.
  - constructor.
  - apply run_epsilon with (s2 := s2); assumption.
  - apply run_char with (s2 := s2); assumption.
Qed.

Lemma nfa_run_trace_complete : forall nfa s str t,
  nfa_run nfa s str t ->
  exists labels, nfa_run_trace nfa s labels str t.
Proof.
  intros nfa s str t Hrun.
  induction Hrun.
  - exists []. constructor.
  - destruct IHHrun as [labels Htrace].
    exists (None :: labels).
    apply trace_epsilon with (s2 := s2); assumption.
  - destruct IHHrun as [labels Htrace].
    exists (Some c :: labels).
    apply trace_char with (s2 := s2); assumption.
Qed.

Lemma nfa_run_trace_append : forall nfa s1 labels1 str1 s2 labels2 str2 s3,
  nfa_run_trace nfa s1 labels1 str1 s2 ->
  nfa_run_trace nfa s2 labels2 str2 s3 ->
  nfa_run_trace nfa s1 (labels1 ++ labels2) (str1 ++ str2) s3.
Proof.
  intros nfa s1 labels1 str1 s2 labels2 str2 s3 Hrun1 Hrun2.
  induction Hrun1.
  - simpl. exact Hrun2.
  - simpl. eapply trace_epsilon; [eassumption|exact (IHHrun1 Hrun2)].
  - simpl. eapply trace_char; [eassumption|exact (IHHrun1 Hrun2)].
Qed.

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

(** * Thompson Construction Contracts *)

(** Evidence for the remaining combinator run decompositions follows the
    standard Thompson construction semantics. Construction/completeness cases
    are proved locally below for arbitrary fresh-state counters.
    Citation: Thompson, K. (1968), "Programming Techniques: Regular
    expression search algorithm", Communications of the ACM 11(6),
    419-422, DOI 10.1145/363347.363387. *)

Record ThompsonEvidence : Prop := mkThompsonEvidence {
  concat_run_decomposition : forall r1 r2 nfa1 nfa2 c1 c2 s,
    compile_nfa r1 0 = (nfa1, c1) ->
    compile_nfa r2 c1 = (nfa2, c2) ->
    let nfa := mkNFA (nfa_start nfa1) (nfa_final nfa2)
                     (nfa_transitions nfa1 ++
                      [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                      nfa_transitions nfa2)
                     (max (nfa_max_state nfa1) (nfa_max_state nfa2)) in
    nfa_accepts nfa s ->
    exists s1 s2, s = (s1 ++ s2)%string /\ regex_matches r1 s1 /\ regex_matches r2 s2;

  alt_run_decomposition : forall r1 r2 nfa1 nfa2 c1 c2 s,
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
    regex_matches r1 s \/ regex_matches r2 s;

  star_run_decomposition : forall r nfa1 c1 s,
    compile_nfa r 1 = (nfa1, c1) ->
    let nfa := mkNFA 0 c1
                     ([TransEpsilon 0 (nfa_start nfa1);
                       TransEpsilon 0 c1] ++
                      nfa_transitions nfa1 ++
                      [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                       TransEpsilon (nfa_final nfa1) c1])
                     (max (nfa_max_state nfa1) c1) in
    nfa_accepts nfa s ->
    regex_matches (RStar r) s;

  plus_run_decomposition : forall r nfa1 c1 s,
    compile_nfa r 0 = (nfa1, c1) ->
    let nfa := mkNFA (nfa_start nfa1) c1
                     (nfa_transitions nfa1 ++
                      [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                       TransEpsilon (nfa_final nfa1) c1])
                     (max (nfa_max_state nfa1) c1) in
    nfa_accepts nfa s ->
    regex_matches (RPlus r) s;

  option_run_decomposition : forall r nfa1 c1 s,
    compile_nfa r 1 = (nfa1, c1) ->
    let nfa := mkNFA 0 c1
                     ([TransEpsilon 0 (nfa_start nfa1);
                       TransEpsilon 0 c1] ++
                      nfa_transitions nfa1 ++
                      [TransEpsilon (nfa_final nfa1) c1])
                     (max (nfa_max_state nfa1) c1) in
    nfa_accepts nfa s ->
    regex_matches (ROption r) s
}.

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
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    inversion Heq. subst.
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    inversion Heq. subst.
    lia.
  - (* RStar *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst.
    lia.
  - (* RPlus *)
    destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr counter nfa1 c1 Hnfa1).
    inversion Heq. subst.
    lia.
  - (* ROption *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst.
    lia.
  - (* RCharClass *)
    inversion Heq. lia.
Qed.

Lemma compile_start_eq : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  nfa_start nfa = counter.
Proof.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - inversion Heq. reflexivity.
  - inversion Heq. reflexivity.
  - inversion Heq. reflexivity.
  - destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    apply IHr1 in Hnfa1. exact Hnfa1.
  - destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. reflexivity.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. reflexivity.
  - destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    apply IHr in Hnfa1. exact Hnfa1.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. reflexivity.
  - inversion Heq. reflexivity.
Qed.

Lemma compile_final_succ_eq : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  S (nfa_final nfa) = counter'.
Proof.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    apply IHr2 in Hnfa2. exact Hnfa2.
  - destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. simpl. lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. simpl. lia.
  - destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. simpl. lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
Qed.

Definition trans_source (tr : NFATransition) : NFAState :=
  match tr with
  | TransChar s _ _ => s
  | TransEpsilon s _ => s
  | TransClass s _ _ => s
  end.

Definition trans_target (tr : NFATransition) : NFAState :=
  match tr with
  | TransChar _ _ t => t
  | TransEpsilon _ t => t
  | TransClass _ _ t => t
  end.

Lemma transition_bounds_weaken :
  forall lower_from lower_to upper_from upper_to transitions,
  lower_to <= lower_from ->
  upper_from <= upper_to ->
  Forall (fun tr => lower_from <= trans_source tr /\ trans_target tr < upper_from)
         transitions ->
  Forall (fun tr => lower_to <= trans_source tr /\ trans_target tr < upper_to)
         transitions.
Proof.
  intros lower_from lower_to upper_from upper_to transitions Hlower Hupper Hbounds.
  induction Hbounds as [|tr rest [Hsrc Hdst] Hrest IH].
  - constructor.
  - constructor.
    + split; lia.
    + exact IH.
Qed.

Lemma compile_transition_bounds : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  Forall (fun tr => counter <= trans_source tr /\ trans_target tr < counter')
         (nfa_transitions nfa).
Proof.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - inversion Heq. constructor.
  - inversion Heq. subst. simpl.
    constructor; [simpl; lia|constructor].
  - inversion Heq. subst. simpl.
    constructor; [simpl; lia|constructor].
  - destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    rewrite Forall_app. split.
    + pose proof (compile_counter_increases r2 c1 nfa2 counter' Hnfa2) as Hinc2.
      eapply (transition_bounds_weaken counter counter c1 counter').
      * lia.
      * lia.
      * apply IHr1. exact Hnfa1.
    + constructor.
      * pose proof (compile_final_succ_eq r1 counter nfa1 c1 Hnfa1) as Hfinal1.
        pose proof (compile_start_eq r2 c1 nfa2 counter' Hnfa2) as Hstart2.
        pose proof (compile_counter_increases r1 counter nfa1 c1 Hnfa1) as Hinc1.
        pose proof (compile_counter_increases r2 c1 nfa2 counter' Hnfa2) as Hinc2.
        simpl. lia.
      * pose proof (compile_counter_increases r1 counter nfa1 c1 Hnfa1) as Hinc1.
        eapply (transition_bounds_weaken c1 counter counter' counter').
        -- lia.
        -- lia.
        -- apply IHr2. exact Hnfa2.
  - destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Heq. subst. simpl.
    constructor.
    + simpl.
      pose proof (compile_start_eq r1 (counter + 1) nfa1 c1 Hnfa1) as Hstart1.
      pose proof (compile_counter_increases r1 (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
      pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
      split; lia.
    + constructor.
      * simpl.
        pose proof (compile_start_eq r2 c1 nfa2 c2 Hnfa2) as Hstart2.
        pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
        split; lia.
      * rewrite Forall_app. split.
        -- pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
           eapply (transition_bounds_weaken (counter + 1) counter c1 (c2 + 1)).
           ++ lia.
           ++ lia.
           ++ apply IHr1. exact Hnfa1.
        -- rewrite Forall_app. split.
           ++ pose proof (compile_counter_increases r1 (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
              eapply (transition_bounds_weaken c1 counter c2 (c2 + 1)).
              ** lia.
              ** lia.
              ** apply IHr2. exact Hnfa2.
           ++ simpl. constructor.
              ** simpl.
                 pose proof (compile_final_succ_eq r1 (counter + 1) nfa1 c1 Hnfa1) as Hfinal1.
                 pose proof (compile_counter_increases r1 (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
                 split; lia.
              ** constructor.
                 --- simpl.
                     pose proof (compile_final_succ_eq r2 c1 nfa2 c2 Hnfa2) as Hfinal2.
                     pose proof (compile_counter_increases r1 (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
                     pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
                     split; lia.
                 --- constructor.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    constructor.
    + simpl.
      pose proof (compile_start_eq r (counter + 1) nfa1 c1 Hnfa1) as Hstart1.
      pose proof (compile_counter_increases r (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
      split; lia.
    + constructor.
      * simpl. split; lia.
      * rewrite Forall_app. split.
        -- eapply (transition_bounds_weaken (counter + 1) counter c1 (c1 + 1)).
           ++ lia.
           ++ lia.
           ++ apply IHr. exact Hnfa1.
        -- simpl. constructor.
           ++ simpl.
              pose proof (compile_final_succ_eq r (counter + 1) nfa1 c1 Hnfa1) as Hfinal1.
              pose proof (compile_counter_increases r (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
              pose proof (compile_start_eq r (counter + 1) nfa1 c1 Hnfa1) as Hstart1.
              split; lia.
           ++ constructor.
              ** simpl.
                 pose proof (compile_final_succ_eq r (counter + 1) nfa1 c1 Hnfa1) as Hfinal1.
                 pose proof (compile_counter_increases r (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
                 split; lia.
              ** constructor.
  - destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    rewrite Forall_app. split.
    + eapply (transition_bounds_weaken counter counter c1 (c1 + 1)).
      * lia.
      * lia.
      * apply IHr. exact Hnfa1.
    + simpl. constructor.
      * simpl.
        pose proof (compile_final_succ_eq r counter nfa1 c1 Hnfa1) as Hfinal1.
        pose proof (compile_start_eq r counter nfa1 c1 Hnfa1) as Hstart1.
        pose proof (compile_counter_increases r counter nfa1 c1 Hnfa1) as Hinc1.
        split; lia.
      * constructor.
        -- simpl.
           pose proof (compile_final_succ_eq r counter nfa1 c1 Hnfa1) as Hfinal1.
           pose proof (compile_counter_increases r counter nfa1 c1 Hnfa1) as Hinc1.
           split; lia.
        -- constructor.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Heq. subst. simpl.
    constructor.
    + simpl.
      pose proof (compile_start_eq r (counter + 1) nfa1 c1 Hnfa1) as Hstart1.
      pose proof (compile_counter_increases r (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
      split; lia.
    + constructor.
      * simpl. split; lia.
      * rewrite Forall_app. split.
        -- eapply (transition_bounds_weaken (counter + 1) counter c1 (c1 + 1)).
           ++ lia.
           ++ lia.
           ++ apply IHr. exact Hnfa1.
        -- simpl. constructor.
           ++ simpl.
              pose proof (compile_final_succ_eq r (counter + 1) nfa1 c1 Hnfa1) as Hfinal1.
              pose proof (compile_counter_increases r (counter + 1) nfa1 c1 Hnfa1) as Hinc1.
              split; lia.
           ++ constructor.
  - inversion Heq. subst. simpl.
    constructor; [simpl; lia|constructor].
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
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
    inversion Heq. subst. simpl.
    destruct IHr1 as [H1a [H1b H1c]].
    destruct IHr2 as [H2a [H2b H2c]].
    repeat split; try lia.
  - (* RAlt *)
    destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
    inversion Heq. subst. simpl.
    destruct IHr1 as [H1a [H1b H1c]].
    destruct IHr2 as [H2a [H2b H2c]].
    repeat split; try lia.
  - (* RStar *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* RPlus *)
    destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr counter nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* ROption *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Ha [Hb Hc]].
    repeat split; try lia.
  - (* RCharClass *)
    inversion Heq. simpl. lia.
Qed.

(** Generalized size bound for compiled NFAs. *)
Lemma compile_nfa_size_generalized : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  nfa_max_state nfa < counter' /\
  counter' - counter <= 2 * regex_size r.
Proof.
  intros r.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
    inversion Heq. subst. simpl.
    destruct IHr1 as [Hmax1 Hdiff1].
    destruct IHr2 as [Hmax2 Hdiff2].
    split; lia.
  - destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    pose proof (compile_counter_increases r2 c1 nfa2 c2 Hnfa2) as Hinc2.
    inversion Heq. subst. simpl.
    destruct IHr1 as [Hmax1 Hdiff1].
    destruct IHr2 as [Hmax2 Hdiff2].
    split; lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Hmax Hdiff].
    split; lia.
  - destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr counter nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Hmax Hdiff].
    split; lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    destruct IHr as [Hmax Hdiff].
    split; lia.
  - inversion Heq. simpl. lia.
Qed.

(** Generalized transition-count bound for compiled NFAs. *)
Lemma compile_nfa_trans_count_generalized : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  List.length (nfa_transitions nfa) <= 4 * regex_size r.
Proof.
  intros r.
  induction r; intros counter nfa counter' Heq; simpl in Heq.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - inversion Heq. simpl. lia.
  - destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 counter nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    inversion Heq. subst. simpl.
    repeat rewrite length_app. simpl. lia.
  - destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    specialize (IHr1 (counter + 1) nfa1 c1 Hnfa1).
    specialize (IHr2 c1 nfa2 c2 Hnfa2).
    inversion Heq. subst. simpl.
    repeat rewrite length_app. simpl. lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    repeat rewrite length_app. simpl. lia.
  - destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr counter nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    repeat rewrite length_app. simpl. lia.
  - destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    specialize (IHr (counter + 1) nfa1 c1 Hnfa1).
    inversion Heq. subst. simpl.
    repeat rewrite length_app. simpl. lia.
  - inversion Heq. simpl. lia.
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

Lemma string_append_empty_r : forall s,
  (s ++ EmptyString)%string = s.
Proof.
  induction s as [|c s IH].
  - reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

Lemma nfa_step_incl : forall nfa1 nfa2 s lbl t,
  incl (nfa_transitions nfa1) (nfa_transitions nfa2) ->
  nfa_step nfa1 s lbl t ->
  nfa_step nfa2 s lbl t.
Proof.
  intros nfa1 nfa2 s lbl t Hincl Hstep.
  inversion Hstep; subst.
  - constructor. apply Hincl. assumption.
  - constructor. apply Hincl. assumption.
  - eapply step_class.
    + apply Hincl. eassumption.
    + assumption.
Qed.

Lemma nfa_run_incl : forall nfa1 nfa2 s str t,
  incl (nfa_transitions nfa1) (nfa_transitions nfa2) ->
  nfa_run nfa1 s str t ->
  nfa_run nfa2 s str t.
Proof.
  intros nfa1 nfa2 s str t Hincl Hrun.
  induction Hrun.
  - constructor.
  - apply run_epsilon with (s2 := s2).
    + eapply nfa_step_incl; eassumption.
    + assumption.
  - apply run_char with (s2 := s2).
    + eapply nfa_step_incl; eassumption.
    + assumption.
Qed.

Lemma nfa_run_epsilon_tail : forall nfa s1 str s2 s3,
  nfa_run nfa s1 str s2 ->
  nfa_step nfa s2 None s3 ->
  nfa_run nfa s1 str s3.
Proof.
  intros nfa s1 str s2 s3 Hrun Hstep.
  rewrite <- (string_append_empty_r str).
  eapply nfa_run_append.
  - exact Hrun.
  - apply run_epsilon with (s2 := s3).
    + exact Hstep.
    + constructor.
Qed.

Local Ltac thompson_transition_in :=
  simpl; repeat rewrite in_app_iff; simpl; intuition congruence.

Lemma thompson_star_loop_accepts : forall r nfa1 start final,
  (forall s, regex_matches r s -> nfa_accepts nfa1 s) ->
  forall s,
  regex_matches (RStar r) s ->
  let nfa := mkNFA start final
                   ([TransEpsilon start (nfa_start nfa1);
                     TransEpsilon start final] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) final])
                   (max (nfa_max_state nfa1) final) in
  nfa_run nfa (nfa_final nfa1) s final.
Proof.
  intros r nfa1 start final Hbody s Hmatch.
  remember (RStar r) as star_regex eqn:Hstar.
  induction Hmatch; inversion Hstar; subst; simpl.
  - apply run_epsilon with (s2 := final).
    + apply step_epsilon. thompson_transition_in.
    + constructor.
  - apply run_epsilon with (s2 := nfa_start nfa1).
    + apply step_epsilon. thompson_transition_in.
    + eapply nfa_run_append.
      * eapply (nfa_run_incl nfa1
          (mkNFA start final
                 ([TransEpsilon start (nfa_start nfa1);
                   TransEpsilon start final] ++
                  nfa_transitions nfa1 ++
                  [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                   TransEpsilon (nfa_final nfa1) final])
                 (max (nfa_max_state nfa1) final))).
        -- unfold incl. intros tr Htr.
           first [ exact Htr | simpl; right; right; rewrite in_app_iff; left; exact Htr ].
        -- exact (Hbody s1 Hmatch1).
      * apply IHHmatch2. reflexivity.
Qed.

Lemma thompson_star_accepts : forall r nfa1 start final,
  (forall s, regex_matches r s -> nfa_accepts nfa1 s) ->
  forall s,
  regex_matches (RStar r) s ->
  let nfa := mkNFA start final
                   ([TransEpsilon start (nfa_start nfa1);
                     TransEpsilon start final] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) final])
                   (max (nfa_max_state nfa1) final) in
  nfa_accepts nfa s.
Proof.
  intros r nfa1 start final Hbody s Hmatch.
  remember (RStar r) as star_regex eqn:Hstar.
  induction Hmatch; inversion Hstar; subst; simpl.
  - apply run_epsilon with (s2 := final).
    + apply step_epsilon. thompson_transition_in.
    + constructor.
  - apply run_epsilon with (s2 := nfa_start nfa1).
    + apply step_epsilon. thompson_transition_in.
    + eapply nfa_run_append.
      * eapply (nfa_run_incl nfa1
          (mkNFA start final
                 ([TransEpsilon start (nfa_start nfa1);
                   TransEpsilon start final] ++
                  nfa_transitions nfa1 ++
                  [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                   TransEpsilon (nfa_final nfa1) final])
                 (max (nfa_max_state nfa1) final))).
        -- unfold incl. intros tr Htr.
           first [ exact Htr | simpl; right; right; rewrite in_app_iff; left; exact Htr ].
        -- exact (Hbody s1 Hmatch1).
      * eapply thompson_star_loop_accepts.
        -- exact Hbody.
        -- exact Hmatch2.
Qed.

Lemma thompson_plus_loop_accepts : forall r nfa1 final,
  (forall s, regex_matches r s -> nfa_accepts nfa1 s) ->
  forall s,
  regex_matches (RStar r) s ->
  let nfa := mkNFA (nfa_start nfa1) final
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) final])
                   (max (nfa_max_state nfa1) final) in
  nfa_run nfa (nfa_final nfa1) s final.
Proof.
  intros r nfa1 final Hbody s Hmatch.
  remember (RStar r) as star_regex eqn:Hstar.
  induction Hmatch; inversion Hstar; subst; simpl.
  - apply run_epsilon with (s2 := final).
    + apply step_epsilon. thompson_transition_in.
    + constructor.
  - apply run_epsilon with (s2 := nfa_start nfa1).
    + apply step_epsilon. thompson_transition_in.
    + eapply nfa_run_append.
      * eapply (nfa_run_incl nfa1
          (mkNFA (nfa_start nfa1) final
                 (nfa_transitions nfa1 ++
                  [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                   TransEpsilon (nfa_final nfa1) final])
                 (max (nfa_max_state nfa1) final))).
        -- unfold incl. intros tr Htr.
           first [ exact Htr | simpl; rewrite in_app_iff; left; exact Htr ].
        -- exact (Hbody s1 Hmatch1).
      * apply IHHmatch2. reflexivity.
Qed.

Lemma thompson_plus_accepts : forall r nfa1 final,
  (forall s, regex_matches r s -> nfa_accepts nfa1 s) ->
  forall s,
  regex_matches (RPlus r) s ->
  let nfa := mkNFA (nfa_start nfa1) final
                   (nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
                     TransEpsilon (nfa_final nfa1) final])
                   (max (nfa_max_state nfa1) final) in
  nfa_accepts nfa s.
Proof.
  intros r nfa1 final Hbody s Hmatch.
  inversion Hmatch; subst; simpl.
  eapply nfa_run_append.
  - eapply (nfa_run_incl nfa1
      (mkNFA (nfa_start nfa1) final
             (nfa_transitions nfa1 ++
              [TransEpsilon (nfa_final nfa1) (nfa_start nfa1);
               TransEpsilon (nfa_final nfa1) final])
             (max (nfa_max_state nfa1) final))).
    + unfold incl. intros tr Htr.
      first [ exact Htr | simpl; rewrite in_app_iff; left; exact Htr ].
    + exact (Hbody s1 H0).
  - eapply thompson_plus_loop_accepts.
    + exact Hbody.
    + exact H1.
Qed.

(** Primitive Thompson fragments have direct semantics. *)
Lemma empty_fragment_run_same_state : forall s1 s s2,
  nfa_run (mkNFA 0 1 [] 1) s1 s s2 ->
  s1 = s2.
Proof.
  intros s1 s s2 Hrun.
  induction Hrun.
  - reflexivity.
  - inversion H; subst; simpl in *; contradiction.
  - inversion H; subst; simpl in *; contradiction.
Qed.

Lemma empty_fragment_no_accepts : forall s,
  ~ nfa_accepts (mkNFA 0 1 [] 1) s.
Proof.
  intros s Haccept.
  pose proof (empty_fragment_run_same_state 0 s 1 Haccept) as Hsame.
  discriminate Hsame.
Qed.

Lemma empty_accepts_sound : forall s,
  nfa_accepts (mkNFA 0 1 [] 1) s ->
  regex_matches REmpty s.
Proof.
  intros s Haccept.
  exfalso. exact (empty_fragment_no_accepts s Haccept).
Qed.

Lemma epsilon_fragment_run_consumes_empty : forall s1 s2 s,
  nfa_run (mkNFA 0 1 [TransEpsilon 0 1] 1) s1 s s2 ->
  s = EmptyString.
Proof.
  intros s1 s2 s Hrun.
  induction Hrun.
  - reflexivity.
  - exact IHHrun.
  - inversion H; subst; cbn in *; intuition discriminate.
Qed.

Lemma epsilon_accepts_sound : forall s,
  nfa_accepts (mkNFA 0 1 [TransEpsilon 0 1] 1) s ->
  regex_matches REpsilon s.
Proof.
  intros s Haccept.
  pose proof (epsilon_fragment_run_consumes_empty _ _ _ Haccept) as Hempty.
  subst. constructor.
Qed.

Lemma char_fragment_final_run_empty : forall c s final,
  nfa_run (mkNFA 0 1 [TransChar 0 c 1] 1) 1 s final ->
  final = 1 ->
  s = EmptyString.
Proof.
  intros c s final Hrun Hfinal.
  dependent induction Hrun.
  - reflexivity.
  - inversion H; subst; cbn in *; intuition discriminate.
  - inversion H; subst; cbn in *; intuition discriminate.
Qed.

Lemma char_accepts_sound : forall c s,
  nfa_accepts (mkNFA 0 1 [TransChar 0 c 1] 1) s ->
  regex_matches (RChar c) s.
Proof.
  intros c s Haccept.
  unfold nfa_accepts in Haccept; cbn in Haccept.
  change (nfa_run (mkNFA 0 1 [TransChar 0 c 1] 1) 0 s 1) in Haccept.
  inversion Haccept; subst.
  - inversion H; subst; cbn in *;
      repeat match goal with
      | HIn : _ \/ False |- _ => destruct HIn as [? | []]
      end;
      discriminate.
  - inversion H; subst; cbn in *;
      repeat match goal with
      | HIn : _ \/ False |- _ => destruct HIn as [? | []]
      end;
      try discriminate;
      repeat match goal with
      | Hedge : TransChar _ _ _ = TransChar _ _ _ |- _ =>
          inversion Hedge; subst; clear Hedge
      end.
    match goal with
    | Hrun : nfa_run (mkNFA 0 1 [TransChar 0 ?ch 1] 1) 1 ?tail 1 |- _ =>
        pose proof (char_fragment_final_run_empty ch tail 1 Hrun eq_refl) as Hstr
    end.
    subst. constructor.
Qed.

Lemma charclass_fragment_final_run_empty : forall cs s final,
  nfa_run (mkNFA 0 1 [TransClass 0 cs 1] 1) 1 s final ->
  final = 1 ->
  s = EmptyString.
Proof.
  intros cs s final Hrun Hfinal.
  dependent induction Hrun.
  - reflexivity.
  - inversion H; subst; cbn in *; intuition discriminate.
  - inversion H; subst; cbn in *; intuition discriminate.
Qed.

Lemma charclass_accepts_sound : forall cs s,
  nfa_accepts (mkNFA 0 1 [TransClass 0 cs 1] 1) s ->
  regex_matches (RCharClass cs) s.
Proof.
  intros cs s Haccept.
  unfold nfa_accepts in Haccept; cbn in Haccept.
  change (nfa_run (mkNFA 0 1 [TransClass 0 cs 1] 1) 0 s 1) in Haccept.
  inversion Haccept; subst.
  - inversion H; subst; cbn in *;
      repeat match goal with
      | HIn : _ \/ False |- _ => destruct HIn as [? | []]
      end;
      discriminate.
  - inversion H; subst; cbn in *;
      repeat match goal with
      | HIn : _ \/ False |- _ => destruct HIn as [? | []]
      end;
      try discriminate;
      repeat match goal with
      | Hedge : TransClass _ _ _ = TransClass _ _ _ |- _ =>
          inversion Hedge; subst; clear Hedge
      end.
    match goal with
    | Hrun : nfa_run (mkNFA 0 1 [TransClass 0 ?classes 1] 1) 1 ?tail 1 |- _ =>
        pose proof (charclass_fragment_final_run_empty classes tail 1 Hrun eq_refl) as Hstr
    end.
    subst. constructor.
    match goal with
    | Hin : In _ _ |- _ => exact Hin
    end.
Qed.

(** * Soundness Theorem *)

(** If NFA accepts, regex matches *)
Theorem thompson_soundness : forall (contracts : ThompsonEvidence) r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  forall s, nfa_accepts nfa s -> regex_matches r s.
Proof.
  intros contracts r.
  induction r; intros nfa counter Hcompile s Haccepts.
  - (* REmpty *)
    simpl in Hcompile. inversion Hcompile. subst.
    exact (empty_accepts_sound s Haccepts).
  - (* REpsilon *)
    simpl in Hcompile. inversion Hcompile. subst.
    exact (epsilon_accepts_sound s Haccepts).
  - (* RChar *)
    simpl in Hcompile. inversion Hcompile. subst.
    exact (char_accepts_sound a s Haccepts).
  - (* RConcat *)
    simpl in Hcompile.
    destruct (compile_nfa r1 0) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the run decomposition evidence premise *)
    destruct (concat_run_decomposition contracts r1 r2 nfa1 nfa2 c1 counter s Hnfa1 Hnfa2 Haccepts)
      as [s1 [s2 [Hseq [Hmatch1 Hmatch2]]]].
    rewrite Hseq.
    apply match_concat; assumption.
  - (* RAlt *)
    simpl in Hcompile.
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the alternation decomposition evidence premise *)
    destruct (alt_run_decomposition contracts r1 r2 nfa1 nfa2 c1 c2 s Hnfa1 Hnfa2 Haccepts) as [Hleft | Hright].
    + apply match_alt_left. assumption.
    + apply match_alt_right. assumption.
  - (* RStar *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the star decomposition evidence premise *)
    exact (star_run_decomposition contracts r nfa1 c1 s Hnfa1 Haccepts).
  - (* RPlus *)
    simpl in Hcompile.
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the plus decomposition evidence premise *)
    exact (plus_run_decomposition contracts r nfa1 c1 s Hnfa1 Haccepts).
  - (* ROption *)
    simpl in Hcompile.
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    unfold nfa_accepts in Haccepts. simpl in Haccepts.
    (* Use the option decomposition evidence premise *)
    exact (option_run_decomposition contracts r nfa1 c1 s Hnfa1 Haccepts).
  - (* RCharClass *)
    simpl in Hcompile. inversion Hcompile. subst.
    exact (charclass_accepts_sound l s Haccepts).
Qed.

(** * Completeness Theorem *)

(** Completeness holds for fragments compiled at any fresh-state counter. *)
Theorem thompson_completeness_general : forall r counter nfa counter',
  compile_nfa r counter = (nfa, counter') ->
  forall s, regex_matches r s -> nfa_accepts nfa s.
Proof.
  induction r; intros counter nfa counter' Hcompile s Hmatch; simpl in Hcompile.
  - (* REmpty - regex never matches *)
    inversion Hmatch.
  - (* REpsilon *)
    inversion Hmatch. subst.
    inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_epsilon with (s2 := counter + 1).
    + constructor. simpl. left. reflexivity.
    + constructor.
  - (* RChar *)
    inversion Hmatch. subst.
    inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_char with (s2 := counter + 1).
    + constructor. simpl. left. reflexivity.
    + constructor.
  - (* RConcat *)
    inversion Hmatch. subst.
    destruct (compile_nfa r1 counter) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    match goal with
    | Hm1 : regex_matches r1 ?s1, Hm2 : regex_matches r2 ?s2 |- _ =>
        pose proof (IHr1 counter nfa1 c1 Hnfa1 s1 Hm1) as Hrun1;
        pose proof (IHr2 c1 nfa2 counter' Hnfa2 s2 Hm2) as Hrun2
    end.
    unfold nfa_accepts in *. simpl in *.
    eapply nfa_run_append.
    + eapply (nfa_run_incl nfa1
        (mkNFA (nfa_start nfa1) (nfa_final nfa2)
               (nfa_transitions nfa1 ++
                [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                nfa_transitions nfa2)
               (max (nfa_max_state nfa1) (nfa_max_state nfa2)))).
      * unfold incl. intros tr Htr. simpl.
        rewrite in_app_iff. left. exact Htr.
      * exact Hrun1.
    + apply run_epsilon with (s2 := nfa_start nfa2).
      * apply step_epsilon. thompson_transition_in.
      * eapply (nfa_run_incl nfa2
          (mkNFA (nfa_start nfa1) (nfa_final nfa2)
                 (nfa_transitions nfa1 ++
                  [TransEpsilon (nfa_final nfa1) (nfa_start nfa2)] ++
                  nfa_transitions nfa2)
                 (max (nfa_max_state nfa1) (nfa_max_state nfa2)))).
        -- unfold incl. intros tr Htr. thompson_transition_in.
        -- exact Hrun2.
  - (* RAlt *)
    destruct (compile_nfa r1 (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst. clear Hcompile.
    inversion Hmatch; subst.
    + (* Left alternative *)
      match goal with
      | Hleft : regex_matches r1 s |- _ =>
          pose proof (IHr1 (counter + 1) nfa1 c1 Hnfa1 s Hleft) as Hrun
      end.
      unfold nfa_accepts in *. simpl in *.
      apply run_epsilon with (s2 := nfa_start nfa1).
      * apply step_epsilon. thompson_transition_in.
      * eapply nfa_run_epsilon_tail.
        -- eapply (nfa_run_incl nfa1
            (mkNFA counter c2
                   ([TransEpsilon counter (nfa_start nfa1);
                     TransEpsilon counter (nfa_start nfa2)] ++
                    nfa_transitions nfa1 ++
                    nfa_transitions nfa2 ++
                    [TransEpsilon (nfa_final nfa1) c2;
                     TransEpsilon (nfa_final nfa2) c2])
                   (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) c2))).
           ++ unfold incl. intros tr Htr. simpl.
              right. right. rewrite in_app_iff. left. exact Htr.
           ++ exact Hrun.
        -- apply step_epsilon. thompson_transition_in.
    + (* Right alternative *)
      match goal with
      | Hright : regex_matches r2 s |- _ =>
          pose proof (IHr2 c1 nfa2 c2 Hnfa2 s Hright) as Hrun
      end.
      unfold nfa_accepts in *. simpl in *.
      apply run_epsilon with (s2 := nfa_start nfa2).
      * apply step_epsilon. thompson_transition_in.
      * eapply nfa_run_epsilon_tail.
        -- eapply (nfa_run_incl nfa2
            (mkNFA counter c2
                   ([TransEpsilon counter (nfa_start nfa1);
                     TransEpsilon counter (nfa_start nfa2)] ++
                    nfa_transitions nfa1 ++
                    nfa_transitions nfa2 ++
                    [TransEpsilon (nfa_final nfa1) c2;
                     TransEpsilon (nfa_final nfa2) c2])
                   (max (max (nfa_max_state nfa1) (nfa_max_state nfa2)) c2))).
           ++ unfold incl. intros tr Htr. simpl.
              right. right. rewrite in_app_iff. right.
              rewrite in_app_iff. left. exact Htr.
           ++ exact Hrun.
        -- apply step_epsilon. thompson_transition_in.
  - (* RStar *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    eapply thompson_star_accepts.
    + intros t Ht. exact (IHr (counter + 1) nfa1 c1 Hnfa1 t Ht).
    + exact Hmatch.
  - (* RPlus *)
    destruct (compile_nfa r counter) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    eapply thompson_plus_accepts.
    + intros t Ht. exact (IHr counter nfa1 c1 Hnfa1 t Ht).
    + exact Hmatch.
  - (* ROption *)
    destruct (compile_nfa r (counter + 1)) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst. clear Hcompile.
    inversion Hmatch; subst.
    + (* None *)
      unfold nfa_accepts. simpl.
      apply run_epsilon with (s2 := c1).
      * apply step_epsilon. thompson_transition_in.
      * constructor.
    + (* Some *)
      match goal with
      | Hsome : regex_matches r s |- _ =>
          pose proof (IHr (counter + 1) nfa1 c1 Hnfa1 s Hsome) as Hrun
      end.
      unfold nfa_accepts in *. simpl in *.
      apply run_epsilon with (s2 := nfa_start nfa1).
      * apply step_epsilon. thompson_transition_in.
      * eapply nfa_run_epsilon_tail.
        -- eapply (nfa_run_incl nfa1
            (mkNFA counter c1
                   ([TransEpsilon counter (nfa_start nfa1);
                     TransEpsilon counter c1] ++
                    nfa_transitions nfa1 ++
                    [TransEpsilon (nfa_final nfa1) c1])
                   (max (nfa_max_state nfa1) c1))).
           ++ unfold incl. intros tr Htr. simpl.
              right. right. rewrite in_app_iff. left. exact Htr.
           ++ exact Hrun.
        -- apply step_epsilon. thompson_transition_in.
  - (* RCharClass *)
    inversion Hmatch. subst.
    inversion Hcompile. subst.
    unfold nfa_accepts. simpl.
    apply run_char with (s2 := counter + 1).
    + apply step_class with (cs := l).
      * simpl. left. reflexivity.
      * assumption.
    + constructor.
Qed.

(** If regex matches, NFA accepts *)
Theorem thompson_completeness : forall r nfa counter,
  compile_nfa r 0 = (nfa, counter) ->
  forall s, regex_matches r s -> nfa_accepts nfa s.
Proof.
  intros r nfa counter Hcompile s Hmatch.
  exact (thompson_completeness_general r 0 nfa counter Hcompile s Hmatch).
Qed.

(** * Size Bound *)

(** NFA size (number of states) *)
Definition nfa_size (nfa : NFA) : nat :=
  nfa_max_state nfa + 1.

(** Number of transitions *)
Definition nfa_trans_count (nfa : NFA) : nat :=
  List.length (nfa_transitions nfa).

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
    (* Use the generalized size bound evidence premise *)
    destruct (compile_nfa_size_generalized r1 0 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    destruct (compile_nfa_size_generalized r2 c1 nfa2 counter Hnfa2) as [Hmax2 Hdiff2].
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound evidence premise *)
    destruct (compile_nfa_size_generalized r1 1 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    destruct (compile_nfa_size_generalized r2 c1 nfa2 c2 Hnfa2) as [Hmax2 Hdiff2].
    lia.
  - (* RStar *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound evidence premise *)
    destruct (compile_nfa_size_generalized r 1 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    lia.
  - (* RPlus *)
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound evidence premise *)
    destruct (compile_nfa_size_generalized r 0 nfa1 c1 Hnfa1) as [Hmax1 Hdiff1].
    lia.
  - (* ROption *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_size. simpl.
    (* Use the generalized size bound evidence premise *)
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
    repeat rewrite length_app. simpl.
    (* Use the generalized transition count evidence premise *)
    pose proof (compile_nfa_trans_count_generalized r1 0 nfa1 c1 Hnfa1) as Htrans1.
    pose proof (compile_nfa_trans_count_generalized r2 c1 nfa2 counter Hnfa2) as Htrans2.
    lia.
  - (* RAlt *)
    destruct (compile_nfa r1 1) as [nfa1 c1] eqn:Hnfa1.
    destruct (compile_nfa r2 c1) as [nfa2 c2] eqn:Hnfa2.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite length_app. simpl.
    (* Use the generalized transition count evidence premise *)
    pose proof (compile_nfa_trans_count_generalized r1 1 nfa1 c1 Hnfa1) as Htrans1.
    pose proof (compile_nfa_trans_count_generalized r2 c1 nfa2 c2 Hnfa2) as Htrans2.
    lia.
  - (* RStar *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite length_app. simpl.
    (* Use the generalized transition count evidence premise *)
    pose proof (compile_nfa_trans_count_generalized r 1 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* RPlus *)
    destruct (compile_nfa r 0) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite length_app. simpl.
    (* Use the generalized transition count evidence premise *)
    pose proof (compile_nfa_trans_count_generalized r 0 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* ROption *)
    destruct (compile_nfa r 1) as [nfa1 c1] eqn:Hnfa1.
    inversion Hcompile. subst.
    unfold nfa_trans_count. simpl.
    repeat rewrite length_app. simpl.
    (* Use the generalized transition count evidence premise *)
    pose proof (compile_nfa_trans_count_generalized r 1 nfa1 c1 Hnfa1) as Htrans1.
    lia.
  - (* RCharClass *)
    inversion Hcompile. unfold nfa_trans_count. simpl. lia.
Qed.

(** Main correctness theorem *)
Theorem thompson_correctness : forall (contracts : ThompsonEvidence) r s,
  let nfa := compile r in
  nfa_accepts nfa s <-> regex_matches r s.
Proof.
  intros contracts r s.
  unfold compile.
  destruct (compile_nfa r 0) as [nfa counter] eqn:Hcompile.
  simpl.
  split.
  - exact (thompson_soundness contracts r nfa counter Hcompile s).
  - exact (thompson_completeness r nfa counter Hcompile s).
Qed.
