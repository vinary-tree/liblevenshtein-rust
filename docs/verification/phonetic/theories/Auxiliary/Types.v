(** * Type Definitions and Predicates - Position Skipping Optimization

    This module contains core type definitions, predicates, and axioms
    used throughout the position skipping optimization proof.

    Part of: Liblevenshtein.Phonetic.Verification.Auxiliary
*)

Require Import String List Arith Ascii Bool Nat Lia.
Require Import PhoneticRewrites.rewrite_rules.
Import ListNotations.

(** * Core Type Definitions and Predicates *)

(** ** Basic Matching Predicate *)

(** Check if a rule can apply at a position without allocating result.
    This is a lightweight check that combines context and pattern matching.
*)
Definition can_apply_at (r : RewriteRule) (s : PhoneticString) (pos : nat) : bool :=
  if context_matches (context r) s pos then
    pattern_matches_at (pattern r) s pos
  else
    false.

(** ** No-Match Predicates *)

(** No rules in the list match before a given position.
    This is the central invariant maintained by the search algorithm.
*)
Definition no_rules_match_before (rules : list RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  forall r, In r rules -> forall p, (p < max_pos)%nat -> can_apply_at r s p = false.

(** Variant with pattern length constraint: only positions where patterns fit *)
Definition no_rules_match_before_with_space (rules : list RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  forall r, In r rules ->
    forall p, (p < max_pos)%nat ->
      (p + length (pattern r) <= max_pos)%nat ->
      can_apply_at r s p = false.

(** ** Rule Matching Predicates *)

(** Represents that a rule matches at some position in the range [0, max_pos) *)
Definition rule_matches_somewhere (r : RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  exists pos, (pos < max_pos)%nat /\ can_apply_at r s pos = true.

(** No rule in the list matches anywhere in the range [0, max_pos) *)
Definition no_rules_match_anywhere (rules : list RewriteRule) (s : PhoneticString) (max_pos : nat) : Prop :=
  forall r, In r rules -> ~rule_matches_somewhere r s max_pos.

(** ** Context and Position-Dependent Properties *)

(** Context is preserved at positions before the transformation point *)
Definition context_preserved_at_earlier_positions (ctx : Context) (s s' : PhoneticString) (transform_pos : nat) : Prop :=
  forall check_pos,
    (check_pos < transform_pos)%nat ->
    context_matches ctx s check_pos = context_matches ctx s' check_pos.

(** Check if a context depends on position.
    Only Final context depends on position (string length).
*)
Definition position_dependent_context (ctx : Context) : bool :=
  match ctx with
  | Final => true  (* Depends on string length *)
  | Initial => false  (* Position 0 is invariant *)
  | BeforeVowel _ => false  (* Depends only on local structure *)
  | AfterConsonant _ => false
  | BeforeConsonant _ => false
  | AfterVowel _ => false
  | Anywhere => false
  end.

(** ** Search Invariants *)

(** The SearchInvariant represents the execution state of sequential search.
    It states that we've checked all positions before 'pos' for all rules
    and found no matches.

    This is the key predicate for modeling algorithm execution and proving
    that find_first_match's behavior implies no_rules_match_before.
*)
Inductive SearchInvariant : list RewriteRule -> PhoneticString -> nat -> Prop :=
| search_inv_intro : forall rules s pos,
    no_rules_match_before rules s pos ->
    SearchInvariant rules s pos.

(** ** Algorithm State Model *)

(** Models the execution state of the sequential search algorithm.
    This inductive type captures how the algorithm advances through positions
    and restarts after applying a rule.

    Used to prove that the algorithm maintains the no_rules_match_before invariant.
*)
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

(** * Decidable Equality for RewriteRule *)

Definition RewriteRule_eq_dec (r1 r2 : RewriteRule) : {r1 = r2} + {r1 <> r2}.
Proof.
  destruct r1 as [id1 name1 pat1 repl1 ctx1 wt1].
  destruct r2 as [id2 name2 pat2 repl2 ctx2 wt2].
  destruct (Nat.eq_dec id1 id2); [| right; congruence].
  destruct (string_eq_dec name1 name2); [| right; congruence].
  destruct (PhoneticString_eq_dec pat1 pat2); [| right; congruence].
  destruct (PhoneticString_eq_dec repl1 repl2); [| right; congruence].
  destruct (Context_eq_dec ctx1 ctx2); [| right; congruence].
  destruct (Q_leibniz_eq_dec wt1 wt2); [| right; congruence].
  left. subst. reflexivity.
Defined.
