(** * Algorithm State Model - Position Skipping Optimization

    This module contains the AlgoState inductive type and its properties.

    The AlgoState relation models the execution state of the sequential rule
    application algorithm, capturing the invariant that at any point in the
    execution, no rules in the list match at positions before the current
    search position.

    This model allows us to prove that find_first_match returning a position
    implies the multi-rule invariant holds.

    Part of: Liblevenshtein.Phonetic.Verification.Invariants
*)

Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Types.
From Liblevenshtein.Phonetic.Verification Require Import Auxiliary.Lib.
From Liblevenshtein.Phonetic.Verification Require Import Core.Rules.
From Liblevenshtein.Phonetic.Verification Require Import Invariants.InvariantProperties.
Import ListNotations.

(** * Contracts for Algorithm State *)

(** Axiom: find_first_match for a rule in a rule list implies AlgoState.

    This axiom captures the execution semantics of the sequential rule application
    algorithm. When find_first_match finds position pos for rule r_head that is
    in the rules list, it implies:
    1. The algorithm searched positions 0 through pos-1 without any rule matching
    2. At position pos, rule r_head (or an earlier rule) matched

    This is exactly the AlgoState invariant at position pos. The axiom bridges
    the semantic gap between the abstract find_first_match function and the
    concrete algorithm execution trace. *)
Definition find_first_match_implies_algo_state_contract : Prop :=
  forall rules r_head s pos,
  (forall r, In r rules -> wf_rule r) ->
  In r_head rules ->
  find_first_match r_head s (length s) = Some pos ->
  AlgoState rules s pos.

(** * Algorithm State Model *)

(** The AlgoState inductive type is already defined in Auxiliary.Types.v:

    Inductive AlgoState : list RewriteRule -> PhoneticString -> nat -> Prop :=
    | algo_init : forall rules s,
        (* Initial state: start searching from position 0 *)
        AlgoState rules s 0

    | algo_step_no_match : forall rules s pos,
        (* Current state at position pos *)
        AlgoState rules s pos ->
        (* No rules in the list match at position pos *)
        (forall r, In r rules -> can_apply_at r s pos = false) ->
        (* Advance to next position *)
        AlgoState rules s (pos + 1)

    | algo_step_match_restart : forall rules r s pos s',
        (* Current state at position pos *)
        AlgoState rules s pos ->
        (* Rule r from the list matches at pos *)
        In r rules ->
        can_apply_at r s pos = true ->
        (* Apply the rule *)
        apply_rule_at r s pos = Some s' ->
        (* Restart from position 0 with transformed string *)
        AlgoState rules s' 0.
*)

(** * Main Theorem: AlgoState Maintains Invariant *)

(** Theorem: AlgoState maintains the no_rules_match_before invariant

    This is the key theorem connecting the algorithm's execution state
    to the no-match invariant. It proves that at any state in the algorithm's
    execution, no rules match at positions before the current position.
*)
Theorem algo_state_maintains_invariant :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    AlgoState rules s pos ->
    no_rules_match_before rules s pos.
Proof.
  intros rules s pos H_wf H_state.
  induction H_state.

  - (* Base case: position 0 *)
    apply no_rules_match_before_zero.

  - (* Step: no match at pos *)
    (* IHH_state gives us: no_rules_match_before rules s pos *)
    (* H gives us: forall r, In r rules -> can_apply_at r s pos = false *)
    (* Need: no_rules_match_before rules s (pos + 1) *)
    apply no_rules_match_before_step; auto.

  - (* Step: match and restart to position 0 *)
    (* After restart, pos = 0, invariant holds trivially *)
    apply no_rules_match_before_zero.
Qed.

(** * Connection to find_first_match *)

(** Lemma: If find_first_match finds a position for a rule in the list,
    then there exists an AlgoState at that position

    This lemma represents a fundamental gap in the proof. It requires
    reasoning about the full execution semantics of apply_rules_seq,
    which cannot be derived from find_first_match alone.

    The issue is that find_first_match only tells us about one rule's
    behavior, but AlgoState requires knowing that ALL rules in the list
    were checked at each position before the match position and all failed.

    This is exactly the semantic gap identified in the Axiom 1 critical
    analysis documents.
*)
Lemma find_first_match_implies_algo_state :
  forall (contract : find_first_match_implies_algo_state_contract) rules r_head s pos,
    (forall r, In r rules -> wf_rule r) ->
    In r_head rules ->
    find_first_match r_head s (length s) = Some pos ->
    AlgoState rules s pos.
Proof.
  intros contract rules r_head s pos H_wf H_in H_find.
  exact (contract rules r_head s pos H_wf H_in H_find).
Qed.

(** * Helper Lemmas for State Construction *)

(** These lemmas would be useful if we had the full execution trace,
    but they are not sufficient to bridge the semantic gap.
*)

(** Lemma: AlgoState is preserved when advancing to a position where no rules match *)
Lemma algo_state_advance_when_no_match :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    AlgoState rules s pos ->
    (forall r, In r rules -> can_apply_at r s pos = false) ->
    AlgoState rules s (pos + 1).
Proof.
  intros rules s pos H_wf H_state H_no_match.
  apply algo_step_no_match; assumption.
Qed.

(** Lemma: AlgoState can be established at position 0 for any string *)
Lemma algo_state_init :
  forall rules s,
    AlgoState rules s 0.
Proof.
  intros rules s.
  apply algo_init.
Qed.

(** Lemma: If AlgoState holds and a rule matches, applying it leads to a new AlgoState at 0 *)
Lemma algo_state_restart_after_match :
  forall rules r s pos s',
    (forall r0, In r0 rules -> wf_rule r0) ->
    AlgoState rules s pos ->
    In r rules ->
    can_apply_at r s pos = true ->
    apply_rule_at r s pos = Some s' ->
    AlgoState rules s' 0.
Proof.
  intros rules r s pos s' H_wf H_state H_in H_can_apply H_apply.
  apply (algo_step_match_restart rules r s pos s'); assumption.
Qed.

(** * Relationship Between AlgoState and SearchInvariant *)

(** Lemma: AlgoState implies SearchInvariant *)
Lemma algo_state_implies_search_invariant :
  forall rules s pos,
    (forall r, In r rules -> wf_rule r) ->
    AlgoState rules s pos ->
    SearchInvariant rules s pos.
Proof.
  intros rules s pos H_wf H_state.
  apply search_inv_intro.
  apply (algo_state_maintains_invariant rules s pos H_wf H_state).
Qed.

(** * Documentation of the Semantic Gap *)

(** The fundamental challenge in completing the AlgoState verification:

    To prove find_first_match_implies_algo_state, we need to show that
    when find_first_match r_head s (length s) = Some pos, we can construct
    an AlgoState trace showing that all rules in the list were checked at
    each position before pos and all failed.

    What we know from find_first_match:
    - r_head matches at position pos
    - r_head doesn't match at any position < pos
    - The search started from position 0

    What we DON'T know (but need for AlgoState):
    - Whether other rules in the list were checked before r_head
    - Whether those other rules matched at positions < pos
    - The execution order of rule checks at each position
    - Whether the algorithm restarted during the search to position pos

    This gap is inherent in the design of find_first_match, which is
    intentionally abstract and doesn't expose the full execution trace
    of apply_rules_seq.

    Possible solutions:
    1. Axiom: Accept find_first_match_in_algorithm_implies_no_earlier_matches
       as an axiom about the algorithm's execution semantics
    2. Refinement: Add execution trace information to find_first_match
    3. Alternative model: Reason directly about apply_rules_seq instead
       of going through find_first_match

    The current approach uses option 1 (axiom) as documented in
    Invariants.NoMatch.v, which is the minimal assumption needed.
*)
