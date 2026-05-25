# Proof Completion Plan

This plan is ordered to retire broad assumptions before expanding trusted scope.
No file should be promoted to `trusted` while it contains `Admitted.` or an
unlisted `Axiom`, `Parameter`, `Conjecture`, or `Hypothesis`.

## Current Proof Debt

As of the latest implementation pass, the executable Coq source scan is clean:
`docs/verification/**/*.v` and `rocq/**/*.v` contain no active `Admitted.`,
`admit.`, `Axiom`, or `Parameter` declarations under the source-level audit
regex.

The explicit proof-obligation audit is now clean:
`scripts/verify-formal.sh audit-contracts-tsv` reports only its header. The
remaining parameterized theorems use neutral `Evidence` records or explicit
`premise` parameters instead of `_contract`, `Contracts`, or `_ax` proof-debt
names. These evidence parameters are not global Coq assumptions or admissions;
they make each conditional theorem's external proof requirements explicit until
the corresponding subsystem is promoted into trusted scope.

The latest passes removed local contract-shaped gaps by replacing them with
direct proofs, model-accurate conditional theorems, or the strongest statement
supported by the current executable model:

- `NFA/Operations.v`: `can_apply_chars_match_contract`,
  `context_matches_monotone_contract`, and `phonetic_path_cheaper_contract`.
- `Layers/Layer1.v`: `layer1_score_decreases_contract`.
- `Layers/Layer2.v`: `layer2_progress_contract`; the current abstract parser
  always returns `None`, so the proved theorem is now `layer2_no_parse_results`.
- `NFA/Optimality.v`: `optimal_correction_exists_contract` and the false
  `phonetic_optimal_contract` over the legacy empty extractor.
- `NFA/Correctness.v`: `phonetic_nfa_correctness_contract`, replaced by the
  specialized soundness/completeness theorem with explicit edit applicability.
- `NFA/Layer1Integration.v`: phonetic layer completeness/soundness contracts,
  replaced by calls to the existing NFA completeness and soundness theorems.
- `NFA/Transitions.v`: `prune_removes_only_subsumed_contract` and
  `prune_produces_minimal_contract`, replaced by structural induction over
  `prune_subsumed_positions`.
- `NFA/Types.v`: `prune_state_satisfies_spec_contract`, replaced by a direct
  proof of inclusion, boundedness preservation, acceptance preservation, and
  removed-position subsumption for `prune_state`.
- `NFA/Automaton.v`: `prune_preserves_acceptance_contract`, replaced by the
  concrete pruning acceptance theorem from `NFA/Types.v`; and
  `distance_zero_exact_match_contract` and `empty_target_empty_input_contract`,
  replaced by direct proofs for the current empty `standard_ops` model.
- `NFA/Automaton.v`: `state_size_bounded_contract`,
  `distance_monotone_contract`, `operations_monotone_contract`, and
  `phonetic_accepts_more_contract`, replaced by direct theorems that match the
  current simplified model: pruning does not increase list length, the empty
  `standard_ops` model accepts only empty target/input, and the "phone"→"fone"
  phonetic example remains unaccepted until dynamic standard matches are added.
- `NFA/StateSpace.v`: `state_space_bounded_contract` and
  `pruned_state_space_contract`, replaced by provable invariants for
  well-formed state error bounds and pruning length monotonicity. The former
  O(n²) cardinality statement needs additional no-duplicate and bounded-index
  invariants before it can be reinstated.
- `ProductState.v`: `product_soundness_contract`, replaced by a constructive
  extraction of the NFA word consumed by any accepting product run; and
  `epsilon_closure_sound_contract`, replaced by a bounded-state epsilon-closure
  theorem matching the transition semantics.
- `ProductState.v`: the false `product_completeness_contract`, whose premise
  did not relate the NFA word to `pattern` or `input`, was replaced by exact
  empty-string completeness; `subsumption_reachability_contract` was replaced
  by an error-monotone step/run simulation proof.
- `Core/EditSequence.v`: the Levenshtein side of `EditSequence` evidence
  (`optimal_lev_seq_exists_contract`, `lev_seq_compose_contract`, and
  `lev_seq_cost_ge_distance_contract`) was retired by reusing the existing
  trace-composition proof `lev_distance_triangle_inequality`. The remaining
  explicit evidence record is now Damerau-only.
- `NFA/Completeness.v`: `phonetic_ceil_cost_equals_one_ax` was replaced by a
  direct proof using `Qceiling_resp_le` and `Qceiling_Z`.
- `NFA/Completeness.v`: the context-sensitive and context-match completeness
  contract fields were retired. The replacement theorems now state exactly
  what the executable model proves: successful `can_apply` constructs the
  normalized output context, and length, character, and context checks imply
  `can_apply = true`.
- `LLRE/SymbolExpansion.v`: `star_soundness_contract`,
  `plus_soundness_contract`, `symbol_soundness_contract`,
  `expansion_preserves_language_contract`, and `expand_respects_depth_contract`
  were replaced by depth-indexed soundness/completeness proofs and an induction
  on the source expansion depth.
- `LLRE/SymbolExpansion.v`: broad symbol-depth, lookup, looked-up-pattern
  expansion, and PSymbol termination contracts were removed. The remaining
  termination theorem now states the proven executable fact for symbol-free
  patterns instead of assuming undefined symbols terminate.
- `LLRE/ThompsonConstruction.v`: generalized `compile_nfa` state and
  transition-count bounds were removed from `ThompsonEvidence` and proved
  directly by induction over regex syntax.
- `LLRE/ThompsonConstruction.v`: primitive fragment soundness for `REmpty`,
  `REpsilon`, `RChar`, and `RCharClass` was removed from `ThompsonEvidence` and
  proved directly by local one-transition/empty-transition run lemmas.
- `LLRE/ThompsonConstruction.v`: Thompson construction/completeness obligations
  for concat, alternation, star, plus, option, and character classes were removed
  from `ThompsonEvidence`. A new arbitrary-counter completeness theorem proves
  them locally using transition-inclusion and loop-run lemmas.
- `LLRE/ThompsonConstruction.v`: added memory-small run-trace infrastructure
  (`nfa_run_trace`, trace completeness/soundness, and trace append) plus
  proved compile start/final and per-transition state interval bounds. These
  lemmas are the foundation for replacing Thompson accepted-run decomposition
  evidence used by soundness.
- `LLRE/ThompsonConstruction.v`: strengthened the per-transition bounds to full
  source/target interval bounds, added step-level interval corollaries, and
  proved arbitrary-counter concat run splitting into left and right sub-NFA
  accepting runs. The remaining Thompson soundness evidence then needed the same
  splitter treatment for alternation, option, star, and plus.
- `LLRE/ThompsonConstruction.v`: added proof-only constructors for the
  alternation and option Thompson NFAs, a generic dead-state run lemma, and a
  `compile_final_ge_counter` bound. These are kept separate from soundness
  wiring so the next splitter proofs can reuse them without increasing proof
  term size in the existing capped compile.
- `LLRE/ThompsonConstruction.v`: proved accepted-run splitters for alternation
  and option using branch-state interval classifiers and final-dead lemmas.
  Thompson soundness is now arbitrary-counter, so concat, alternation, and
  option soundness use local sub-NFA accepting-run proofs plus the induction
  hypothesis instead of `ThompsonEvidence`.
- `LLRE/ThompsonConstruction.v`: proved star/plus loop step classifiers,
  body-run decomposition, and accepted-run soundness for Kleene star and plus.
  `ThompsonEvidence` is retired; Thompson soundness and correctness are now
  unconditional local theorems.
- `MSM/Core/MsmDistance.v`: `msm_reflexive_diagonal_proof` was removed from
  `MsmDistanceEvidence`. Reflexivity for identical series is now proved locally
  with a memory-small row-diagonal invariant showing each diagonal cell is bounded
  by the zero-cost Move predecessor, then using non-negativity.
- `Myers/Equivalence.v`: the placeholder full-equivalence contracts were
  replaced by direct model-accurate lemmas: shifted initialization decoding,
  vacuous zero-width step preservation, and empty-text equivalence against the
  current placeholder DP model.
- `LLRE/SymbolExpansion.v`: the broad cycle-detection contract was replaced by
  a bounded DFS theorem matching the executable fuel semantics of
  `has_cycle_from`.
- Grammar core/layer/NFA wrapper contracts that depended on missing enumerators,
  composed automata, Viterbi paths, or arbitrary pipeline layers were narrowed
  to explicit witness or precondition theorems.
- `NFA/Completeness.v`: removed the false broad theorem that `"phone"`→`"fone"`
  is accepted by the current automata. The replacement theorem records the
  executable fact that both the standard and phonetic automata currently reject
  that example until dynamic standard matches are added.
- `NFA/Soundness.v`: removed false or derivable soundness fields from
  `NFASoundnessEvidence`. Path extraction cost is now proved for the current
  empty legacy extractor, phonetic automaton soundness is derived from the
  general acceptance bridge plus a proved `phonetic_automaton_wf`, the phonetic
  witness theorem was weakened from "all edits are phonetic" to "some accepted
  edit is phonetic", and empty-output consume-y reasoning is proved for
  well-formed operations.
- `NFA/Completeness.v`: removed `path_extension_from_operation_bridge` from
  `NFACompletenessEvidence`. The current `valid_path` model directly admits a
  singleton accepting witness at `String.length target`, so only the
  valid-path-to-acceptance bridge remains; the full edit-sequence acceptance
  theorem is now derived from that narrower bridge.
- `NFA/Completeness.v` and `NFA/Soundness.v`: retired
  `NFACompletenessEvidence` and `NFASoundnessEvidence`. The current
  path/edit-sequence models are not anchored enough to prove broad acceptance
  equivalence, so the remaining broad NFA theorems were narrowed to explicit
  executable acceptance or edit-sequence witness hypotheses. Downstream NFA
  correctness and Layer 1 integration theorems now expose those hypotheses
  directly instead of hiding them in evidence records.
- `Automaton/Soundness.v`: removed `initial_state_no_special_proof` from
  `AutomatonSoundnessEvidence`; the transposition initial state is
  `[std_pos 0 0]`, so non-specialness is now a direct lemma.
- `MSM/Core/CFunction.v`: removed an unused contract-shaped triangle helper;
  `MSM/Metric/MainTheorem.v`: renamed the already-proved reverse triangle lemma
  so it no longer appears as an axiom-shaped theorem.
- `MSM/Indexing/QuantizationBounds.v`: removed the broad quantization and trie
  no-false-negative premises for the placeholder `quantize` implementation.
  Replacement theorems state the true current-model facts: all quantized values
  are zero, equal-length quantized Levenshtein distance is zero, and same-length
  trie thresholds are sufficient.
- `MSM/Indexing/LowerBounds.v`: removed broad L1/combined lower-bound premises
  that are not justified by the executable DP model. Replacement theorems prove
  only the empty-side L1, length, combined lower-bound, and pruning cases.
- `MSM/Core/MsmDistance.v`: removed the false split/merge upper-bound evidence
  and the unused broad length lower-bound evidence. Empty-distance behavior is
  now proved directly by computation.
- `MSM/Metric/TriangleInequality.v` and `MSM/Metric/MainTheorem.v`: removed the
  false empty-middle triangle case and added a concrete counterexample. The
  metric theorem is now correctly stated over non-empty time series, where the
  narrowed triangle theorem applies.
- `Phonetic/Auxiliary/Types.v`, `Phonetic/Invariants/AlgoState.v`, and
  `phonetic/position_skipping_proof.v`: removed unused legacy wrappers for the
  false broad claim that single-rule `find_first_match` alone proves a
  multi-rule no-earlier-match invariant.
- `Phonetic/Invariants/NoMatch.v` and
  `Phonetic/Invariants/InvariantProperties.v`: replaced the contract records
  with proved preservation theorems that require the caller to supply the
  execution no-match invariant and the pattern-fit bound.
- `phonetic/rewrite_rules.v`: removed the abstract `RewriteRule` evidence
  shell; the concrete zompist rule set and rule proofs remain in
  `phonetic/zompist_rules.v`.
- `MSM/Metric/MainTheorem.v`: removed `MsmMetricContracts`, which only
  repackaged lower-level MSM distance, symmetry, and triangle obligations.
- MSM, grammar NFA, LLRE Thompson, and core automaton records were renamed from
  contract/axiom-shaped proof-debt names to neutral evidence records. False or
  uninstantiated broad statements now remain conditional rather than being
  reported as completed unconditional proofs.
- `Core/EditSequence.v`, `Core/DamerauLevDistance.v`, and
  `Composition/DamerauComposition.v`: removed the uninstantiated
  `DLEditSequenceEvidence` route to a triangle theorem. The executable
  Damerau recurrence is the restricted adjacent-transposition/OSA variant, so
  the unconditional triangle inequality is false; `DamerauComposition.v` now
  records the checked `ab -> ba -> bca` counterexample instead of a broken proof
  shell.
- `scripts/verify-formal.sh`: added `audit-contracts` and
  `audit-contracts-tsv` modes so explicit `_contract`, `Contracts`, and `_ax`
  proof obligations are visible alongside the Coq-native admission audit.
- `scripts/verify-formal.sh`: added `audit-evidence` and
  `audit-evidence-tsv` modes, and wired the evidence audit into `trusted` and
  `coq-trusted`, so neutral `Evidence` records and explicit `_proof`,
  `_bridge`, and `_premise` parameters remain visible before promotion.
- `Automaton/Completeness.v`: removed directly provable or unused
  completeness evidence: algorithm/query-length transition fields, run
  algorithm preservation, the subsumed-witness premise, the empty-remaining
  can-reach evidence record, fold-state finality, non-empty run epsilon-closure,
  and unused can-complete preservation fields. The replacements are local
  lemmas with capped compile coverage.
- `Automaton/Soundness.v`: removed the unused special-origin transition
  preservation field from `AutomatonSoundnessEvidence`; the remaining soundness
  fields are the ones still consumed by downstream proofs.
- `Core/MergeSplitDistance.v` and `Automaton/Soundness.v`: proved
  `lev_distance_ms_bound` from optimal merge-split edit sequences. The
  `lev_distance_ms_bound_proof` field was removed from
  `AutomatonSoundnessEvidence`; MergeAndSplit Levenshtein fallback soundness now
  uses the local sequence-simulation theorem directly.
- `Automaton/Completeness.v`: removed the false broad
  `subsumption_preserves_nonspecial` evidence field. The only caller now uses
  the explicit antichain invariant that existing positions are non-special;
  this matches the executable subsumption definitions, where Standard
  subsumption does not inspect `is_special`.
- `Automaton/Completeness.v`: removed unused transition/spread fields from
  `AutomatonCompletenessCoreEvidence` (`automaton_step_std_trans_proof`,
  `automaton_step_std_ms_proof`,
  `automaton_step_std_trans_position_incl_proof`,
  `automaton_step_spread_bound_proof`, `spread_bound_preserved`, and
  `spread_bound_through_closure_and_insert_proof`). The remaining fields are
  referenced by active proofs.
- `Automaton/MainTheorem.v`: removed `automaton_distance_correct_premise`.
  Reported Standard automaton distances are now bounded directly from the
  accepting-distance witness, standard run reachability, and the existing
  reachable-position error bound.
- `Automaton/Completeness.v`: removed
  `automaton_final_state_accepts_proof`; final-state acceptance is now derived
  from `position_contained_from_run` plus query-length preservation. Also
  removed `fold_state_insert_spread_bound_ms_proof`; the folded MergeAndSplit
  spread bound follows from origin tracking and monotonicity of the folded
  minimum.
- `MSM/Core/MsmDefinitions.v`, `MSM/Core/MsmDistance.v`,
  `MSM/Metric/Identity.v`, and `MSM/Metric/MainTheorem.v`: corrected the MSM
  identity target from raw Coq list equality over `Q` to pointwise rational
  setoid equality `series_Qeq`. The old statement is not provable over QArith:
  `[1#1]` and `[2#2]` have zero MSM distance but are not Leibniz-equal.
- `MSM/Core/MsmDistance.v`: narrowed `MsmDistanceEvidence` further to the
  non-empty DP identity case. Empty/empty is immediate, and empty/non-empty
  mismatch cases are now proved locally from `c > 0` and positivity of
  `inject_Z (Z.of_nat (length _)) * c`.
- `Automaton/Completeness.v`: narrowed `can_reach_higher_index` to the only
  model-accurate form used by callers, requiring the original `can_reach`
  witness to end at `term_index = length query` with bounded final errors. The
  broader non-final statement was false.
- `Automaton/Completeness.v`: proved the narrowed `can_reach_higher_index`
  obligation locally by induction over `can_reach` and removed the standalone
  `AutomatonCompletableEvidence` record. Ahead-in-query positions now simulate
  consumed dictionary characters with `INSERT` steps paid for by the saved
  error budget.
- `Automaton/Completeness.v`: removed the unused
  `AutomatonCompletableStateEvidence` record and its dead wrapper lemmas. The
  only reusable subsumption helper now uses the local Standard proof directly
  instead of a broad algorithm-parameterized evidence field.
- `Automaton/Completeness.v`: removed the unused
  `AutomatonCompletenessTransitionEvidence` route and the dead
  `reachable_implies_contained_aux` wrapper. The exact epsilon-closure field was
  over-strong for antichain-filtered states; active completeness relies on the
  narrower `position_contained_from_run` field in core evidence.
- `Automaton/Completeness.v`: removed the false exact fold-state inclusion
  surfaces (`AutomatonFoldStateEvidence`, the Standard-to-Transposition fold
  field, and both special-algorithm spread fields). The Standard-to-Transposition
  and Standard-to-MergeSplit acceptance bridges now use proved Standard
  soundness plus `damerau_lev_le_standard`/`ms_le_standard`, then discharge the
  target with the existing algorithm-specific completeness contracts.
- `Automaton/Completeness.v`: deleted unused core completeness fields for
  exact transition production, epsilon-closure final inclusion, and local
  distance tracking, plus the dead `automaton_finds_distance` corollary. The
  active core record now tracks only position containment and
  algorithm-specific completeness.
- `Automaton/Completeness.v`: proved the remaining exact-match transition
  success obligation from `position_contained_from_run`, Standard run
  reachability, Standard non-special preservation, and the characteristic-vector
  spread/window lemmas. The core completeness record no longer contains a
  transition-success field.
- `ASSUMPTIONS.tsv`: split the broad MSM metric candidate into narrow allowed
  assumptions for MSM identity, symmetry, and the non-empty-domain triangle
  theorem, all cited to Stefan et al. MSM reflexivity has since been retired
  from the allowed set after the local row-diagonal proof. `ThompsonEvidence`
  has also been retired after local Kleene star/plus decomposition proofs.
- `ASSUMPTIONS.tsv`: retired stale candidates for the old grammar
  Levenshtein-triangle shell and the now-removed
  `automaton_run_nonempty_epsilon_closed` wrapper.

| Area | Remaining unallowlisted evidence surface |
|---|---|
| Core automaton soundness | `AutomatonSoundnessEvidence` |
| Core automaton completeness | `AutomatonCompletenessCoreEvidence` |

`docs/verification/core/theories/Distance.v` remains a legacy monolith. Use the
decomposed modules (`OptimalTrace/*`, `Triangle/*`, `LowerBound/*`,
`Core/*`) for regular capped verification; the monolith is retained as
reference material and is not the memory-efficient target.

## Completion Order

1. Instantiate shared foundations.

   Keep using the decomposed core Levenshtein modules for metric facts. Do not
   re-promote `Distance.v`; it is slower and duplicates the modular proof tree.

2. Prove core automaton contracts.

   Close the remaining core record by proving or further decomposing the active
   `position_contained_from_run`, `transposition_completeness`, and
   `merge_split_completeness` fields. Exact antichain inclusion has been removed
   because it is false for the executable filtering model.

3. Rebuild grammar NFA equivalence on traced runs.

   The evidence records are retired. To restore unconditional NFA equivalence,
   introduce a generated-run path relation that is anchored to
   `run_automaton_from`, prove it erases to `accepts`, and prove traced runs
   produce edit sequences with bounded cost. Only then strengthen the narrowed
   witness-based theorems back to unconditional completeness/soundness.

4. Keep LLRE construction closed.

   Thompson construction no longer has evidence parameters. Keep size/count
   proofs separate from language-equivalence proofs and continue compiling the
   file with the standard capped profile before promoting related changes.

5. Close product and MSM.

   For product, prove state-transition simulations against the trusted core
   distance model. For MSM, use the cited paper only for mathematical context;
   local Coq definitions still need reflexivity, identity, symmetry, and
   non-empty-domain triangle proofs. Do not reinstate all-list metric claims
   without changing the empty-series model; the current file contains a proved
   counterexample for an empty middle series.

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
