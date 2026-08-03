# Levenshtein Verification Proof Index

**Generated:** 2025-12-01
**Status:** Historical index. For current trusted scope and proof-gap status,
use `FORMAL_VERIFICATION_MANIFEST.tsv` and `README_FORMAL_GATES.md`.

## Quick Statistics

| Category | Core (Modular) | Core (Distance.v.bak) | Phonetic | Total |
|----------|----------------|----------------------|----------|-------|
| Modules | 31 | 1 (deprecated backup) | 14 | 46 |
| Theorems | 6 | 3 | 4 | 13 |
| Lemmas | ~220 | ~80 | ~95 | ~395 |
| Definitions | ~45 | ~10 | ~15 | ~70 |
| Axioms | 0 | 0 | 2 | 2 |

**Note:** Distance.v.bak is a deprecated backup. All proofs have been extracted into the modular structure.

---

## 1. Main Theorems (Top-Level Results)

### Generic elastic walker

| Theorem | Location | Description |
|---|---|---|
| `tree_bound_lower_bounds_every_terminal` | `elastic/theories/WalkerSoundness.v` | Local K1 terminal bounds and K2 child inflation lift to every terminal in a recursive dictionary subtree. |
| `rejected_subtree_contains_no_qualifying_terminal` | `elastic/theories/WalkerSoundness.v` | A subtree rejected above the inclusive cutoff contains no qualifying exact result. |
| `k1_k2_imply_no_false_negatives` | `elastic/theories/WalkerSoundness.v` | The complete recursive DFS walker retains every exact terminal whose distance is inside the cutoff. |
| `walk_tree_sound` | `elastic/theories/WalkerSoundness.v` | Every emitted value belongs to the source tree and satisfies the cutoff. |

### Representation-preserving subsumption fallback

| Theorem | Location | Description |
|---|---|---|
| `shared_subsumes_is_legacy_subsumes` | `core/theories/Conformance/SubsumptionFallback.v` | For every carrier predicate, factoring the shared cost guard out of the three legacy mode branches is extensionally exact. |
| `transposition_mixed_continuations_never_subsume` | `core/theories/Conformance/SubsumptionFallback.v` | Normal and in-progress OSA states remain incomparable in both directions. |
| `merge_split_requires_same_index_and_kind` | `core/theories/Conformance/SubsumptionFallback.v` | Every shared merge/split dominance result preserves index and continuation kind. |

### Exact multi-kind Dyck correction and binary persistence

| Theorem | Location | Description |
|---|---|---|
| `correction_target_is_dyck` | `core/theories/Conformance/DyckCorrection.v` | Every reconstruction branch produces a kind-sensitive balanced target. |
| `zero_cost_correction_is_balanced_identity` | `core/theories/Conformance/DyckCorrection.v` | A zero-cost witness preserves the source exactly and proves it is Dyck. |
| `every_source_has_a_correction` | `core/theories/Conformance/DyckCorrection.v` | Deleting every token supplies a total upper-bound witness. |
| `interval_recurrence_is_globally_exact` | `core/theories/Conformance/DyckCorrection.v` | With strict subinterval minima already filled, the least runtime branch cost is equivalent to the global minimum over all correction trees. |
| `nonempty_dyck_first_pair_decomposition` | `core/theories/Conformance/DyckCorrection.v` | Every nonempty typed-Dyck word has the first-pair decomposition enumerated by the interval recurrence. |
| `diagnostic_rename_is_semantics_preserving` | `core/theories/Conformance/OperationSetSerialization.v` | Operation behavior depends on its applicability tag, never its diagnostic name. |
| `accepted_envelope_is_exact_and_bounded` | `core/theories/Conformance/OperationSetSerialization.v` | Acceptance implies the magic/version/flags contract, exact consumption, semantic validation, and resource bounds. |
| `trailing_payload_bytes_are_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | An envelope with bytes beyond its declared payload cannot be accepted. |
| `accepted_protobuf_is_preflight_bounded` | `core/theories/Conformance/OperationSetSerialization.v` | Every allocation-bearing protobuf count is within policy before semantic admission. |
| `protobuf_over_limit_never_reaches_prost` | `core/theories/Conformance/OperationSetSerialization.v` | An operation, pair, or text count above policy cannot pass the pre-allocation gate. |
| `protobuf_weight_bits_round_trip_exactly` | `core/theories/Conformance/OperationSetSerialization.v` | The protobuf fixed64 weight field preserves all 64 IEEE-754 bits. |
| `trailing_compressed_data_is_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | A gzip wrapper that does not consume the complete supplied input cannot be accepted. |
| `decompression_over_limit_is_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | Inflated bytes above the inner-format ceiling are rejected before semantic decode. |

### Class-A alignment presets

| Theorem | Location | Description |
|---|---|---|
| `hamming_triangle` | `core/theories/Conformance/ClassAPresets.v` | Coordinate mismatch triangle inequalities lift inductively to equal-length sequences. |
| `reverse_script_preserves_cost` | `core/theories/Conformance/ClassAPresets.v` | Reversing an indel script and swapping insert/delete preserves cost. |
| `reverse_script_swaps_consumption` | `core/theories/Conformance/ClassAPresets.v` | The inverse script exchanges source and target consumption. |
| `indel_length_lower_bounds` | `core/theories/Conformance/ClassAPresets.v` | Either directional length difference is bounded by script cost. |
| `bounded_skip_exact_length_difference` | `core/theories/Conformance/ClassAPresets.v` | A match/source-delete path costs exactly source length minus target length. |
| `validated_total_bounds_every_prefix` | `core/theories/Conformance/ClassAPresets.v` | A complete aggregate below the resource ceiling bounds every operation prefix. |

### `PositionKind` and monomorphized variants

| Theorem | Location | Description |
|---|---|---|
| `full_key_injective` | `core/theories/Conformance/PositionKindVariant.v` | Equality of `(term_index, num_errors, kind, aux)` implies equality of positions, justifying binary-search uniqueness. |
| `dispatch_equivalence` | `core/theories/Conformance/PositionKindVariant.v` | Runtime per-position and selected static variant policies are extensionally equal for every built-in algorithm. |
| `osa_mixed_continuations_do_not_subsume` | `core/theories/Conformance/PositionKindVariant.v` | A normal OSA position cannot prune a pending adjacent-transposition continuation. |
| `merge_split_requires_strictly_fewer_errors` | `core/theories/Conformance/PositionKindVariant.v` | Every successful merge/split dominance decision has strict accumulated-cost improvement. |
| `standard_subsumption_never_reverses_error_order` | `core/theories/Conformance/PositionKindVariant.v` | Standard dominance cannot hold when the alleged dominator has greater accumulated cost. |

### Unrestricted Damerau streaming refinement

| Theorem | Location | Description |
|---|---|---|
| `entry_preserves_budget` | `damerau/theories/DamerauStreaming.v` | Every guarded macro entry remains inside the configured edit budget. |
| `entry_creates_valid_pending` | `damerau/theories/DamerauStreaming.v` | Entry creates a pending continuation whose positive delta fits the one-byte payload. |
| `extend_preserves_delta_and_adds_one` | `damerau/theories/DamerauStreaming.v` | An interior dictionary unit preserves origin/delta and charges exactly one insertion. |
| `pending_has_no_epsilon_successor` | `damerau/theories/DamerauStreaming.v` | A pending macro cannot double-charge prepaid query-interior deletions. |
| `resolve_advances_exact_endpoint` | `damerau/theories/DamerauStreaming.v` | Resolution advances from the stored origin by exactly `$`\delta+1`$` and preserves cost. |
| `macro_cost_equivalent` | `damerau/theories/DamerauStreaming.v` | The streaming charge equals the Lowrance–Wagner macro term. |
| `mixed_continuations_never_subsume` | `damerau/theories/DamerauStreaming.v` | Normal and pending residual languages are incomparable in both directions. |
| `pending_subsumption_requires_same_key` | `damerau/theories/DamerauStreaming.v` | Pending dominance exposes non-greater cost and equality of origin and delta. |
| `frontier_quadratic_bound` | `damerau/theories/DamerauStreaming.v` | At most `$`k`$` diagonals times `$`k`$` deltas yields the `$`k^2`$` frontier envelope. |

### Discrete Fréchet kernel and bottleneck properties

| Theorem | Location | Description |
|---|---|---|
| `interval_frechet_step_admissible` | `frechet/theories/Metric/FrechetProperties.v` | Exact point-to-bin minima and monotone min/max recurrence lower-bound every represented scalar cell. |
| `point_interval_frechet_step_exact` | `frechet/theories/Metric/FrechetProperties.v` | A point bin reproduces the scalar bottleneck recurrence exactly. |
| `endpoint_bound_admissible` | `frechet/theories/Metric/FrechetProperties.v` | The maximum of the two coupling-pinned endpoint links is a candidate lower bound. |
| `one_sided_hausdorff_admissible` | `frechet/theories/Metric/FrechetProperties.v` | Coverage of every source point by a bounded coupling implies the one-sided Hausdorff bound. |
| `bottleneck_triangle_composition_step` | `frechet/theories/Metric/FrechetProperties.v` | Pointwise triangle bounds survive one minimax coupling-composition step. |
| `bottleneck_zero_identifies_each_link` | `frechet/theories/Metric/FrechetProperties.v` | A non-negative zero bottleneck forces both prefix and current link to zero. |

### ERP kernel and quotient properties

| Theorem | Location | Description |
|---|---|---|
| `interval_dist_admissible` | `erp/theories/Metric/ErpProperties.v` | Scalar-to-bin distance lower-bounds every concrete realization. |
| `interval_dist_degenerate` | `erp/theories/Metric/ErpProperties.v` | A point bin reproduces scalar absolute distance exactly. |
| `script_gap_mass_bound` | `erp/theories/Metric/ErpProperties.v` | Gap-mass potential difference is bounded by the cost of any ERP edit script. |
| `erp_candidate_lower_bound` | `erp/theories/Metric/ErpProperties.v` | K4 candidate bound for the source and target projected from any alignment. |
| `zero_cost_alignment_has_quotient_identity` | `erp/theories/Metric/ErpProperties.v` | Every zero-cost alignment has equal normal forms after removing the fixed gap value. |

### Core Verification - Levenshtein Distance Properties

| Theorem | Location | Description |
|---------|----------|-------------|
| **trace_cost_lower_bound** | `LowerBound/MainTheorem.v:42` | Any valid trace with NoDup and monotonicity has cost >= lev_distance. The fundamental lower bound theorem. |
| **lev_distance_identity** | `Core/MetricProperties.v:21` | d(A, A) = 0. A string has zero distance to itself. |
| **lev_distance_symmetry** | `Core/MetricProperties.v:41` | d(A, B) = d(B, A). Edit distance is symmetric. |
| **lev_distance_triangle_inequality** | `Triangle/TriangleInequality.v:145`  | $`d(A, C) \le  d(A, B) + d(B, C).`$ Triangle inequality for edit distance. |
| **lev_distance_upper_bound** | `Core/MetricProperties.v:92` | d(A, B) <= max(\|A\|, \|B\|). Distance bounded by longer string. |
| **trace_composition_cost_bound** | `Composition/CostBounds.v:1313` | $`\mathrm{cost}(T_1 \circ T_2) \le \mathrm{cost}(T_1) + \mathrm{cost}(T_2)`$. Key lemma for triangle inequality. |
| **distance_equals_min_trace_cost** | `Distance.v.bak:7876` | Distance equals minimum trace cost over all valid traces. |

### Phonetic Verification - Position Skipping Optimization

| Theorem | Location | Description |
|---------|----------|-------------|
| **position_skipping_conditionally_safe** | `Position_Skipping_Proof.v:518` | Position skipping is safe for restricted rule sets with position-independent contexts. |
| **position_skip_safe_for_local_contexts** | `Position_Skipping_Proof.v:365` | Position skipping preserves semantics when contexts don't depend on absolute position. |
| **apply_rules_seq_opt_terminates** | `Core/Rules.v:75` | The optimized algorithm always terminates with sufficient fuel. |
| **pattern_overlap_preservation** | `Patterns/PatternOverlap.v` | When a pattern overlaps a transformation region and fails to match originally, it fails after transformation. (612-line proof) |

---

## 2. Supporting Theorems & Key Lemmas

### Tier 1: Metric Space Foundations

| Name | Type | Location | Description |
|------|------|----------|-------------|
| lev_distance_length_diff_lower | Lemma | `Core/MetricProperties.v:199` | Distance is at least the difference in lengths |
| abs_diff_succ_bound | Lemma | `Core/MetricProperties.v:155` | Bound on abs_diff with successor |

### Tier 2: Algorithm Correctness

| Name | Type | Location | Description |
|------|------|----------|-------------|
| lev_distance_unfold | Lemma | `Core/LevDistance.v:61` | Unfolding lemma matching recursive definition |
| lev_distance_empty_left | Lemma | `Core/LevDistance.v:81` | Base case: distance from empty string on left |
| lev_distance_empty_right | Lemma | `Core/LevDistance.v:89` | Base case: distance from empty string on right |
| lev_distance_cons | Lemma | `Core/LevDistance.v:98` | Recursive case for cons patterns |
| lev_distance_nil_nil | Lemma | `LowerBound/Definitions.v:22` | Base: empty to empty is 0 |
| lev_distance_nil_l | Lemma | `LowerBound/Definitions.v:25` | Base: empty to any on left |
| lev_distance_nil_r | Lemma | `LowerBound/Definitions.v:28` | Base: any to empty on right |
| lev_distance_cons_cons | Lemma | `LowerBound/Definitions.v:31` | Cons case for both strings |

### Tier 3: Min Function Properties

| Name | Type | Location | Description |
|------|------|----------|-------------|
| min3_lower_bound | Lemma | `Core/MinLemmas.v:19` | min3 returns value <= all inputs |
| min3_comm_12 | Lemma | `Core/MinLemmas.v:37` | min3 commutative in first two args |
| subst_cost_eq | Lemma | `Core/MinLemmas.v:78` | subst_cost is 0 for identical chars |
| subst_cost_neq | Lemma | `Core/MinLemmas.v:93` | subst_cost is 1 for different chars |
| subst_cost_bound | Lemma | `Core/MinLemmas.v:107` | subst_cost bounded by 1 |

### Tier 4: Trace Validity

| Name | Type | Location | Description |
|------|------|----------|-------------|
| is_valid_trace_aux_implies_monotonic | Lemma | `Trace/TraceBasics.v:126` | BRIDGE: is_valid_trace_aux implies monotonicity |
| is_valid_trace_implies_NoDup | Lemma | `Trace/TraceBasics.v:225` | Valid traces have NoDup |
| is_valid_trace_implies_monotonic | Lemma | `Trace/TraceBasics.v:237` | Valid traces are monotonic |
| compatible_pairs_monotonic_helper | Lemma | `Trace/TraceBasics.v:55` | Compatible pairs enforce order |
| forallb_compatible_monotonic | Lemma | `Trace/TraceBasics.v:73` | forallb compatible implies monotonicity |

### Tier 5: Touched Positions

| Name | Type | Location | Description |
|------|------|----------|-------------|
| touched_in_A_length | Lemma | `Trace/TouchedPositions.v:36` | Length of touched_in_A equals trace length |
| touched_in_B_length | Lemma | `Trace/TouchedPositions.v:47` | Length of touched_in_B equals trace length |
| In_touched_in_A_exists_pair | Lemma | `Trace/TouchedPositions.v:58` | If i in touched_in_A, exists j with (i,j) in T |
| In_pair_implies_touched_A | Lemma | `Trace/TouchedPositions.v:84` | If (i,j) in T, then i in touched_in_A |
| In_pair_implies_touched_B | Lemma | `Trace/TouchedPositions.v:97` | If (i,j) in T, then j in touched_in_B |

### Tier 6: Cardinality & NoDup

| Name | Type | Location | Description |
|------|------|----------|-------------|
| NoDup_split | Lemma | `Cardinality/NoDupInclusion.v:18` | Split list with NoDup at element |
| incl_length_NoDup | Lemma | `Cardinality/NoDupInclusion.v:50` | Inclusion with NoDup implies length ordering |
| NoDup_list_inter | Lemma | `Cardinality/NoDupInclusion.v:132` | NoDup preserved by list_inter |
| list_inter_length_bound | Lemma | `Cardinality/NoDupInclusion.v:143` | Length of intersection is bounded |
| NoDup_incl_exclusion | Lemma | `Cardinality/NoDupInclusion.v:155` | Inclusion-exclusion: $`\lvert l1\rvert + \lvert l2\rvert \le n + \lvert l1 \cap l2\rvert`$ |

### Tier 6.5: Trace Composition Infrastructure

| Name | Type | Location | Description |
|------|------|----------|-------------|
| fold_left_triangle_bound | Lemma | `Composition/CostBounds.v:728` | Pointwise bound implies fold_left bound |
| fold_left_sum_map_eq | Lemma | `Composition/CostBounds.v:752` | Fold over composed function equals fold over map |
| fold_left_sum_bound_subset | Lemma | `Composition/CostBounds.v:766` | Sum over subset is bounded by superset sum |
| fold_left_pair_let_body_eq | Lemma | `Composition/CostBounds.v:789` | Equivalence of let-pattern forms in fold_left |
| witness_to_T1_injective | Lemma | `Composition/CostBounds.v:426` | witness_to_T1 is injective on composed trace |
| witness_to_T2_injective | Lemma | `Composition/CostBounds.v:477` | witness_to_T2 is injective on composed trace |
| map_injective_on_list_NoDup | Lemma | `Composition/CostBounds.v:528` | Injective map preserves NoDup |
| touched_comp_A_length_le | Lemma | `Composition/CostBounds.v:847` | touched_in_A of composition bounded by T1 |
| touched_comp_C_length_le | Lemma | `Composition/CostBounds.v:862` | touched_in_C of composition bounded by T2 |
| composition_size_pigeonhole | Lemma | `Composition/CostBounds.v:1072` | Pigeonhole bound on composition size |
| trace_composition_delete_insert_bound | Lemma | `Composition/CostBounds.v:1089` | Delete/insert cost bound for composition |
| change_cost_compose_bound | Lemma | `Composition/CostBounds.v:1170` | Change cost triangle inequality for composition |

### Tier 7: Has Predicates

| Name | Type | Location | Description |
|------|------|----------|-------------|
| monotonicity_eliminates_cross_matching | Lemma | `LowerBound/HasPredicates.v:33` | Monotonicity eliminates cross-matching |
| monotonic_cross_matching_impossible | Lemma | `LowerBound/HasPredicates.v:98` | Cross-matching impossible with monotonicity |
| touched_in_A_1_implies_pair | Lemma | `LowerBound/HasPredicates.v:53` | Extract (1, j) from touched_in_A containing 1 |
| valid_trace_indices_ge1 | Lemma | `LowerBound/HasPredicates.v:79` | Pairs in valid trace have indices >= 1 |

### Tier 8: Shift Operations

| Name | Type | Location | Description |
|------|------|----------|-------------|
| shift_trace_11_length | Lemma | `LowerBound/ShiftTrace11Lemmas.v:21` | Length of shift_trace_11 when (1,1) present |
| shift_trace_A_length_no_A1 | Lemma | `LowerBound/ShiftTraceA.v:46` | shift_trace_A preserves length when has_A1=false |
| shift_trace_B_length_no_B1 | Lemma | `LowerBound/ShiftTraceB.v:39` | shift_trace_B preserves length when has_B1=false |
| shift_trace_11_valid | Lemma | `LowerBound/ShiftTrace11Lemmas.v:86` | Validity of shift_trace_11 |
| shift_trace_A_valid | Lemma | `LowerBound/ShiftTraceA.v:156` | Validity of shift_trace_A |
| shift_trace_B_valid | Lemma | `LowerBound/ShiftTraceB.v:104` | Validity of shift_trace_B |

### Tier 9: NoDup Preservation

| Name | Type | Location | Description |
|------|------|----------|-------------|
| shift_trace_A_NoDup_A | Lemma | `LowerBound/NoDupPreservation.v:95` | NoDup preserved for A under shift_trace_A |
| shift_trace_B_NoDup_B | Lemma | `LowerBound/NoDupPreservation.v:184` | NoDup preserved for B under shift_trace_B |
| shift_trace_11_NoDup_A | Lemma | `LowerBound/ShiftTrace11Lemmas.v:266` | NoDup preserved for shift_trace_11 on A |
| shift_trace_11_NoDup_B | Lemma | `LowerBound/ShiftTrace11Lemmas.v:306` | NoDup preserved for shift_trace_11 on B |

### Tier 10: Monotonicity Preservation

| Name | Type | Location | Description |
|------|------|----------|-------------|
| shift_trace_A_monotonic | Lemma | `LowerBound/MonotonicityLemmas.v:89` | Monotonicity preserved for shift_trace_A |
| shift_trace_B_monotonic | Lemma | `LowerBound/MonotonicityLemmas.v:106` | Monotonicity preserved for shift_trace_B |
| shift_trace_11_monotonic | Lemma | `LowerBound/MonotonicityLemmas.v:123` | Monotonicity preserved for shift_trace_11 |

### Tier 11: Pigeonhole Bounds

| Name | Type | Location | Description |
|------|------|----------|-------------|
| NoDup_length_le_range | Lemma | `LowerBound/PigeonholeBounds.v:116` | Pigeonhole: NoDup list in [a,b] has length <= b-a+1 |
| NoDup_A_bound | Lemma | `LowerBound/PigeonholeBounds.v:135` | NoDup + validity + no A1 implies \|T\| <= \|s1'\| |
| NoDup_B_bound | Lemma | `LowerBound/PigeonholeBounds.v:160` | NoDup + validity + no B1 implies \|T\| <= \|s2'\| |

### Tier 12: Cost Analysis

| Name | Type | Location | Description |
|------|------|----------|-------------|
| trace_cost_fold_cons | Lemma | `LowerBound/TraceCostFold.v:29` | Accumulator property for fold_left |
| trace_cost_fold_shift_all_ge2 | Lemma | `LowerBound/TraceCostFold.v:50` | Cost equality after shift when indices >= 2 |
| change_cost_shift_11 | Lemma | `LowerBound/TraceCostFold.v:79` | Cost decomposition for shift_trace_11 |
| change_cost_shift_A | Lemma | `LowerBound/ShiftTraceA.v:142` | Cost equality for shift_trace_A |
| change_cost_shift_B | Lemma | `LowerBound/ShiftTraceB.v:90` | Cost equality for shift_trace_B |

---

## 3. Core Definitions

### Foundation Types

| Name | Type | Location | Description |
|------|------|----------|-------------|
| Char | Definition | `Core/Definitions.v:18` | Characters as Coq's ascii type |
| Matrix | Definition | `Core/Definitions.v:24` | DP matrix: nested list for 2D array |
| Trace | Definition | `Trace/TraceBasics.v:20` | List of pairs (i, j) representing alignment |
| SearchInvariant | Inductive | `Auxiliary/Types.v:82` | Execution state of sequential search |
| AlgoState | Inductive | `Auxiliary/Types.v:95` | Execution state of search algorithm |

### Core Functions

| Name | Type | Location | Description |
|------|------|----------|-------------|
| min3 | Definition | `Core/Definitions.v:29` | Minimum of three natural numbers |
| subst_cost | Definition | `Core/Definitions.v:41` | Substitution cost: 0 if match, 1 otherwise |
| lev_distance_pair | Function | `Core/LevDistance.v:36` | Levenshtein distance with well-founded recursion |
| lev_distance | Definition | `Core/LevDistance.v:55` | Wrapper with standard signature |
| optimal_trace_pair | Function | `OptimalTrace/Construction.v:28` | Optimal trace via DP backtracking |

### Trace Operations

| Name | Type | Location | Description |
|------|------|----------|-------------|
| touched_in_A | Definition | `Trace/TouchedPositions.v:20` | Positions in A touched by trace |
| touched_in_B | Definition | `Trace/TouchedPositions.v:27` | Positions in B touched by trace |
| trace_cost | Definition | `Trace/TraceCost.v:22` | Cost according to Wagner-Fischer |
| valid_pair | Definition | `Trace/TraceBasics.v:25` | Check if pair valid for lengths |
| trace_monotonic | Definition | `Trace/TraceBasics.v:48` | Trace preserves order |

### Shift Operations

| Name | Type | Location | Description |
|------|------|----------|-------------|
| shift_trace_11 | Definition | `LowerBound/ShiftTrace11.v:20` | Filter out (1,1) and shift indices |
| shift_trace_A | Definition | `LowerBound/ShiftTraceA.v:28` | Filter pairs with i>1 and shift |
| shift_trace_B | Definition | `LowerBound/ShiftTraceB.v:21` | Filter pairs with j>1 and shift |

### Predicates

| Name | Type | Location | Description |
|------|------|----------|-------------|
| has_pair_11 | Definition | `LowerBound/HasPredicates.v:19` | Check if (1,1) in trace |
| has_A1 | Definition | `LowerBound/HasPredicates.v:23` | Check if 1 in touched_in_A |
| has_B1 | Definition | `LowerBound/HasPredicates.v:27` | Check if 1 in touched_in_B |
| simple_valid_trace | Definition | `LowerBound/Definitions.v:66` | Simple validity check |
| can_apply_at | Definition | `Auxiliary/Types.v:20` | Check if rule can apply at position |
| no_rules_match_before | Definition | `Auxiliary/Types.v:31` | No rules match before position |

---

## 4. Phonetic Verification - Supporting Lemmas

### Find First Match Lemmas

| Name | Type | Location | Description |
|------|------|----------|-------------|
| find_first_match_from_lower_bound | Lemma | `Auxiliary/Lib.v:44` | Search only from start_pos onward |
| find_first_match_some_implies_can_apply | Lemma | `Auxiliary/Lib.v:287` | Some result implies can_apply_at true |
| find_first_match_is_first | Lemma | `Auxiliary/Lib.v:376` | Found position has no earlier match |
| find_first_match_from_skip_one | Lemma | `Position_Skipping_Proof.v:42` | Skip single non-matching position |
| find_first_match_from_skip_range | Lemma | `Position_Skipping_Proof.v:55` | Skip range of non-matching positions |

### Context Preservation

| Name | Type | Location | Description |
|------|------|----------|-------------|
| apply_rule_at_preserves_prefix | Lemma | `Patterns/PatternHelpers_Basic.v:19` | Preserves phones before match position |
| initial_context_preserved | Lemma | `Patterns/PatternHelpers_Basic.v:71` | Initial context preserved at earlier positions |
| before_vowel_context_preserved | Lemma | `Patterns/PatternHelpers_Basic.v:85` | BeforeVowel context preserved |
| after_consonant_context_preserved | Lemma | `Patterns/PatternHelpers_Basic.v:138` | AfterConsonant context preserved |

### Pattern Matching

| Name | Type | Location | Description |
|------|------|----------|-------------|
| pattern_matches_at_has_mismatch | Lemma | `Patterns/PatternMatching_Induction.v:25` | False match implies mismatch position exists |
| pattern_has_leftmost_mismatch | Lemma | `Patterns/PatternMatching_Positioning.v:25` | Mismatch has leftmost (first) position |
| leftmost_mismatch_before_transformation | Lemma | `Patterns/PatternOverlap.v:44` | Leftmost mismatch before transformation |

### Invariant Maintenance

| Name | Type | Location | Description |
|------|------|----------|-------------|
| algo_state_maintains_invariant | Theorem | `Invariants/AlgoState.v:61` | AlgoState maintains no_rules_match_before |
| search_invariant_init | Lemma | `Invariants/InvariantProperties.v:125` | Search invariant holds at position 0 |
| search_invariant_step_all_rules | Lemma | `Invariants/InvariantProperties.v:179` | Invariant extends when all rules don't match |
| no_rules_match_before_first_match_preserved | Theorem | `Position_Skipping_Proof.v:111` | Multi-rule invariant for position-independent contexts |

---

## 5. Axioms & Semantic Gaps

| Name | Status | Location | Description |
|------|--------|----------|-------------|
| rule_id_unique | Axiom | `Auxiliary/Types.v:127` | rule_id uniquely identifies rules in Zompist phonetic system. Closed-world semantics for finite rule set. |
| find_first_match_in_algorithm_implies_no_earlier_matches | Axiom | `Auxiliary/Types.v:142` | If find_first_match finds position for rule, no rules matched before. Semantic bridge. |
| find_first_match_implies_algo_state | Admitted | `Invariants/AlgoState.v:100` | SEMANTIC GAP: Connects find_first_match result to AlgoState existence. |

---

## 6. Module-by-Module Reference

### Core Theories (`docs/verification/core/theories/`)

#### Core/
- `Definitions.v` - Base types: Char, Matrix, min3, subst_cost
- `LevDistance.v` - Main lev_distance function with well-founded recursion
- `MinLemmas.v` - Properties of min3 and subst_cost
- `MetricProperties.v` - Metric space: identity, symmetry, upper bound

#### Trace/
- `TraceBasics.v` - Trace type, validity, monotonicity
- `TouchedPositions.v` - touched_in_A, touched_in_B projections
- `TraceCost.v` - trace_cost function and bounds
- `TraceComposition.v` - compose_trace operation

#### Cardinality/
- `NoDupInclusion.v` - NoDup lemmas, list_inter, inclusion-exclusion
- `NoDupPreservation.v` - NoDup preservation under trace operations

#### Triangle/
- `SubstCostTriangle.v` - Substitution cost triangle inequality
- `TriangleInequality.v` - **lev_distance_triangle_inequality theorem**

#### Composition/
- `WitnessLemmas.v` - Witness construction for trace composition
- `CompositionNoDup.v` - NoDup preservation for composed traces
- `CompositionValidity.v` - Validity preservation for composed traces
- `CostBounds.v` - **trace_composition_cost_bound theorem** and helper lemmas

#### OptimalTrace/
- `Construction.v` - optimal_trace_pair construction via DP
- `Validity.v` - Validity proof for optimal traces
- `CostEquality.v` - trace_cost(optimal_trace) = lev_distance

#### DPMatrix/
- `MatrixOps.v` - Matrix initialization and update operations
- `SnocLemmas.v` - Suffix (snoc) lemmas for lev_distance
- `Correctness.v` - Wagner-Fischer DP matrix correctness

#### MainTheorems.v
- Consolidated exports of all main theorems

#### LowerBound/ (12 modules)
- `Definitions.v` - Trace types and base lemmas
- `HasPredicates.v` - has_A1, has_B1, has_pair_11
- `ShiftTrace11.v` - shift_trace_11 operation
- `ShiftTraceA.v` - shift_trace_A operation
- `ShiftTraceB.v` - shift_trace_B operation
- `BoundHelpers.v` - Validity bound helpers
- `PigeonholeBounds.v` - Pigeonhole principle bounds
- `NoDupPreservation.v` - NoDup preservation under shifts
- `ShiftTrace11Lemmas.v` - shift_trace_11 validity and NoDup
- `MonotonicityLemmas.v` - Monotonicity preservation
- `TraceCostFold.v` - trace_cost_fold and cost decomposition
- `MainTheorem.v` - **trace_cost_lower_bound theorem**

### Phonetic Theories (`docs/verification/phonetic/theories/`)

#### Auxiliary/
- `Types.v` - can_apply_at, SearchInvariant, AlgoState, axioms
- `Lib.v` - find_first_match_from, arithmetic helpers, search lemmas

#### Core/
- `Rules.v` - apply_rules_seq_opt, termination theorem

#### Invariants/
- `AlgoState.v` - algo_state_maintains_invariant
- `InvariantProperties.v` - Invariant initialization and stepping
- `NoMatch.v` - No-match preservation lemmas
- `SearchInvariant.v` - SearchInvariant lemmas

#### Patterns/
- `PatternHelpers_Basic.v` - Prefix preservation, context preservation
- `PatternMatching_Properties.v` - Pattern matching properties
- `PatternMatching_Induction.v` - Nested induction for mismatch
- `PatternMatching_Positioning.v` - Leftmost mismatch analysis
- `PatternOverlap.v` - **pattern_overlap_preservation theorem**
- `Preservation.v` - Context preservation definitions

#### Main Entry Point
- `Position_Skipping_Proof.v` - **position_skipping_conditionally_safe theorem**

---

## 7. Dependency Graph (Simplified)

```
                     trace_cost_lower_bound
                              |
            +-----------------+------------------+
            |                 |                  |
    change_cost_shift_*   NoDup_*_bound    shift_trace_*_monotonic
            |                 |                  |
    trace_cost_fold     pigeonhole         shift_trace_*_valid
            |                 |                  |
        subst_cost      touched_in_*       trace_monotonic
            |                 |                  |
         min3              Trace            valid_pair
            |                 |                  |
          Char          list (nat*nat)         nat
```

```
              position_skipping_conditionally_safe
                              |
            +-----------------+------------------+
            |                 |                  |
    no_rules_match_*   pattern_overlap_*   apply_rules_seq_opt
            |                 |                  |
    search_invariant    leftmost_mismatch   find_first_match_from
            |                 |                  |
    algo_state          context_preserved    can_apply_at
            |                 |                  |
    AlgoState           apply_rule_at       RewriteRule
```

---

## 8. Extraction Status (Distance.v.bak → Modular)

Distance.v.bak is now a **deprecated backup**. All key theorems have been extracted into the modular structure.

### ✅ Fully Extracted (Now in Modular Files)

| Original (Distance.v.bak) | Extracted To | Status |
|---------------------------|--------------|--------|
| compose_trace | `Trace/TraceComposition.v` | ✅ Complete |
| compose_trace_valid | `Composition/CompositionValidity.v` | ✅ Complete |
| compose_trace_cost | `Composition/CostBounds.v:trace_composition_cost_bound` | ✅ Complete |
| subst_cost_triangle | `Triangle/SubstCostTriangle.v` | ✅ Complete |
| lev_distance_triangle_inequality | `Triangle/TriangleInequality.v:145` | ✅ Complete |
| witness_injectivity | `Composition/CostBounds.v:witness_to_T1/T2_injective` | ✅ Complete |
| fold_left_triangle_bound | `Composition/CostBounds.v:728` | ✅ Complete |
| change_cost_compose_bound | `Composition/CostBounds.v:1170` | ✅ Complete |
| composition_size_pigeonhole | `Composition/CostBounds.v:1072` | ✅ Complete |

### Remaining Items in Distance.v.bak

These items remain in the backup file but are not critical for the modular build:

| Name | Type | Line | Description |
|------|------|------|-------------|
| distance_equals_min_trace_cost | Theorem | 7876 | Distance equals minimum cost (verified via optimal trace) |
| dp_matrix_correctness | Theorem | 8349 | DP matrix correctness (in DPMatrix/Correctness.v) |

### Decomposition Summary

- **31 modular files** now compile with no Admitted lemmas
- **Distance.v.bak** kept as reference but not used in build
- All metric space properties proven in modular structure
- Triangle inequality chain: `trace_cost_lower_bound` → `trace_composition_cost_bound` → `lev_distance_triangle_inequality`
