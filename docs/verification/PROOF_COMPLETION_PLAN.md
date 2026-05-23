# Proof Completion Plan

This plan is ordered to retire broad assumptions before expanding trusted scope.
No file should be promoted to `trusted` while it contains `Admitted.` or an
unlisted `Axiom`, `Parameter`, `Conjecture`, or `Hypothesis`.

## Current Proof Debt

As of the latest implementation pass, the executable Coq source scan is clean:
`docs/verification/**/*.v` and `rocq/**/*.v` contain no active `Admitted.`,
`admit.`, `Axiom`, or `Parameter` declarations under the source-level audit
regex.

The remaining proof debt is now represented as explicit contract obligations.
These are not global Coq assumptions, but any theorem that consumes one is only
fully instantiated after a caller supplies a proof of that contract.

| Area | Explicit obligation |
|---|---|
| Core automaton soundness | `AutomatonSoundnessContracts` |
| Core automaton completeness | `AutomatonCompletenessTransitionContracts`, `AutomatonCompletenessCoreContracts`, `AutomatonCompletableContracts`, `AutomatonCompletableStateContracts`, `AutomatonCanReachEmptyContracts`, `AutomatonFoldStateContracts` |
| Core automaton top level | `automaton_distance_correct_contract` |
| Grammar NFA soundness/completeness | `NFASoundnessContracts`, `NFACompletenessContracts` |
| Grammar NFA transitions | `run_position_monotonicity_contract`, `two_step_reachability_via_run_contract`, `prune_removes_only_subsumed_contract`, `prune_produces_minimal_contract`, `delta_equiv_preservation_contract`, `edit_sequence_induces_accepting_run_contract` |
| Grammar NFA top level | `phonetic_nfa_correctness_contract` |

`docs/verification/core/theories/Distance.v` remains a legacy monolith. Use the
decomposed modules (`OptimalTrace/*`, `Triangle/*`, `LowerBound/*`,
`Core/*`) for regular capped verification; the monolith is retained as
reference material and is not the memory-efficient target.

## Completion Order

1. Instantiate shared foundations.

   Keep using the decomposed core Levenshtein modules for metric facts. Do not
   re-promote `Distance.v`; it is slower and duplicates the modular proof tree.

2. Prove core automaton contracts.

   Close the transition and epsilon-closure contracts first, then spread-bound
   preservation, can-complete preservation, fold/state insertion, finality, and
   algorithm-specific completeness. Keep each group in its own capped module
   before wiring it back into `Automaton/Completeness.v`.

3. Prove grammar NFA contracts.

   Close transition monotonicity, two-step reachability, pruning preservation,
   delta preservation, edit-sequence acceptance, path extension, and context
   sensitivity as separate lemmas. Only then instantiate
   `NFACompletenessContracts` and `NFASoundnessContracts`.

4. Close LLRE construction.

   Replace the run decomposition/construction axioms with structural induction
   over regex syntax. Keep size/count proofs separate from language-equivalence
   proofs to avoid large proof terms.

5. Close product, Myers, and MSM.

   For product and Myers, prove state-transition simulations against the trusted
   core distance model. For MSM, use the cited paper only for mathematical
   context; local Coq definitions still need reflexivity, identity, symmetry,
   triangle, and indexing-bound proofs.

6. Review phonetic proofs.

   Keep generated or memory-heavy phonetic pattern proofs out of trusted scope
   until they are split by rule family and capped with `heavy` or `exceptional`
   profiles.

## Memory Strategy

- Prefer small lemmas with explicit section-local hypotheses over monolithic
  `Theorem` blocks.
- Avoid large `repeat destruct` over product state spaces; state and prove
  selector lemmas instead.
- Put generated or table-like facts behind narrow, cited assumptions only when
  the source is stable and independently verifiable.
- Run all promoted proof files through `scripts/verify-formal.sh coq-trusted`;
  the runner refuses uncapped proof execution unless explicitly overridden.
- Use `scripts/verify-formal.sh coq-file <profile> <path>` for targeted
  capped compiles of partial files while closing the backlog.
