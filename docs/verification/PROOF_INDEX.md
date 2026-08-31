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

### Lazy weighted temporal frontiers

| Theorem | Location | Description |
|---|---|---|
| `epsilon_run_composition` | `temporal_automata/theories/LazyWeightedFrontier.v` | A zero-target-consumption path prefixes any suffix run without changing its consumed target word. |
| `epsilon_dominance_is_residual_simulation` | `temporal_automata/theories/LazyWeightedFrontier.v` | A witnessed epsilon-reachability cost makes antichain pruning safe for every future target suffix. |
| `canonical_frontier_permutation_invariant` | `temporal_automata/theories/LazyWeightedFrontier.v` | Canonical state membership is independent of candidate generation order. |
| `canonical_frontier_is_antichain` | `temporal_automata/theories/LazyWeightedFrontier.v` | No retained representative strictly dominates another retained representative. |
| `canonical_interning_is_permutation_sound` | `temporal_automata/theories/LazyWeightedFrontier.v` | Canonical frontiers generated in different orders are safe to reuse under one compact state identity. |
| `exact_canonical_key_reuse_is_sound` | `temporal_automata/theories/LazyWeightedFrontier.v` | Exact equality of canonical context and positions preserves every future run and final cost. |
| `interval_additive_step_is_lower_simulation` | `temporal_automata/theories/LazyWeightedFrontier.v` | Lower predecessor and local costs refine one additive ERP/MSM/TWED/DTW recurrence cell. |
| `interval_bottleneck_step_is_lower_simulation` | `temporal_automata/theories/LazyWeightedFrontier.v` | Lower predecessor and link costs refine one discrete-Fréchet bottleneck cell. |
| `exact_leaf_verification_has_no_false_positives` | `temporal_automata/theories/LazyWeightedFrontier.v` | Exact leaf rescoring is the authority boundary that removes abstract false positives. |
| `page_then_resume_equals_uninterrupted` | `temporal_automata/theories/LazyWeightedFrontier.v` | Every page prefix followed by its continuation reconstructs the uninterrupted observation sequence. |
| `complete_if_and_only_if_exhausted` | `temporal_automata/theories/LazyWeightedFrontier.v` | Complete—including complete empty—is constructible exactly after search exhaustion. |
| `generational_retention_is_prefix_independent` | `temporal_automata/theories/LazyWeightedFrontier.v` | Two bounded frontier generations plus a bounded cache retain memory independent of consumed-prefix length. |
| `sparse_transition_storage_is_observation_bounded` | `temporal_automata/theories/LazyWeightedFrontier.v` | Sparse generated-target storage grows only with distinct transitions actually observed. |
| `push_reused_state_preserves_valid_references` | `temporal_automata/theories/LazyWeightedFrontier.v` | Reusing an interned residual adds a valid frame identifier without allocating an arena state. |
| `push_fresh_state_preserves_valid_references` | `temporal_automata/theories/LazyWeightedFrontier.v` | Appending a fresh residual before its frame preserves every existing and new arena reference. |
| `pop_frame_preserves_valid_references` | `temporal_automata/theories/LazyWeightedFrontier.v` | Frame-only DFS pop leaves every remaining append-only arena identifier valid. |
| `interned_arena_retention_is_history_independent` | `temporal_automata/theories/LazyWeightedFrontier.v` | The configured state ceiling bounds the append-only arena independently of visited-node count; this is not a live-depth-only bound. |
| `rejected_child_preflight_is_atomic` | `temporal_automata/theories/LazyWeightedFrontier.v` | A prospective child above the scratch ceiling leaves retained product state unchanged. |
| `equal_observation_has_equal_transition` | `temporal_automata/theories/LazyProductOperations.v` | State-relative equal observations induce exactly equal transitions when the concrete transition factors through the observation. |
| `complete_cache_eviction_is_transparent` | `temporal_automata/theories/LazyProductOperations.v` | Evicting a complete transition entry and recomputing the same successor changes no observation. |
| `descend_preserves_snapshot_revision` | `temporal_automata/theories/LazyProductOperations.v` | Every successful zipper descent remains scoped to its parent's immutable dictionary revision. |
| `descend_materializes_path_append` | `temporal_automata/theories/LazyProductOperations.v` | Delayed reconstruction from a shared reverse parent spine equals eager path append. |
| `iterative_release_steps_bounded` | `temporal_automata/theories/LazyProductOperations.v` | Releasing a zipper parent spine examines at most one node per retained spine node. |
| `iterative_release_stops_at_shared_suffix` | `temporal_automata/theories/LazyProductOperations.v` | A failed unique-owner unwrap stops before consuming the shared suffix. |
| `iterative_release_drains_unique_spine` | `temporal_automata/theories/LazyProductOperations.v` | A uniquely owned parent spine is drained completely by one loop iteration per node. |
| `erase_path_preserves_successful_descent` | `temporal_automata/theories/LazyProductOperations.v` | Consuming a zipper into an opaque traversal focus may erase path-only context without changing any successful native descent. |
| `erase_path_preserves_absent_descent` | `temporal_automata/theories/LazyProductOperations.v` | Opaque path erasure cannot introduce a dictionary edge that the captured focus did not have. |
| `product_child_components` | `temporal_automata/theories/LazyProductOperations.v` | Every constructed product child consists of a dictionary descent and live query transition on the same label. |
| `product_child_preserves_snapshot_revision` | `temporal_automata/theories/LazyProductOperations.v` | A product transition cannot cross dictionary snapshot revisions. |
| `absent_dictionary_child_prunes_product` | `temporal_automata/theories/LazyProductOperations.v` | A missing dictionary edge creates no Cartesian product state. |
| `dead_query_child_prunes_product` | `temporal_automata/theories/LazyProductOperations.v` | A dead query transition prunes its dictionary child product. |
| `query_first_child_is_product_equivalent` | `temporal_automata/theories/LazyProductOperations.v` | Projecting through the query machine before constructing an owned dictionary child preserves the synchronized product child exactly. |
| `rejected_projection_constructs_no_child` | `temporal_automata/theories/LazyProductOperations.v` | A query-rejected edge constructs zero owned child foci. |
| `live_projection_constructs_one_child` | `temporal_automata/theories/LazyProductOperations.v` | A live projected edge constructs exactly one owned child focus. |
| `admitted_final_is_within_cutoff` | `temporal_automata/theories/LazyProductOperations.v` | A finite exact finalizer score becomes a public range result only when it is no greater than the configured cutoff. |
| `over_cutoff_final_is_rejected` | `temporal_automata/theories/LazyProductOperations.v` | Closing trailing query-only operations cannot leak a finite over-cutoff score into public results. |
| `completed_schedule_permutation_preserves_membership` | `temporal_automata/theories/LazyProductOperations.v` | Completed unordered schedulers may reorder work without changing result membership. |

### Sparse timestamped-TWED product

| Theorem | Location | Description |
|---|---|---|
| `same_typed_label_reflects_every_component` | `twed/theories/Metric/TimestampedProductIndex.v` | Typed token equality includes value bin, time bin, and canonical unit. |
| `exact_state_equalb_reflects_complete_residual` | `twed/theories/Metric/TimestampedProductIndex.v` | Exact reuse compares previous typed label, every sparse row/cost-bit anchor, and final-cost bits. |
| `collision_checked_reuse_is_exact` | `twed/theories/Metric/TimestampedProductIndex.v` | A fingerprint bucket can return a reused state only after complete residual equality. |
| `equal_fingerprint_does_not_authorize_unequal_reuse` | `twed/theories/Metric/TimestampedProductIndex.v` | A fingerprint collision alone cannot authorize reuse. |
| `omitted_cell_has_exact_subsumption_witness` | `twed/theories/Metric/TimestampedProductIndex.v` | Every omitted row has the exact vertical query-deletion equality witness used by canonicalization. |
| `enumerated_sparse_positions_are_strictly_ordered` | `twed/theories/Metric/TimestampedProductIndex.v` | Enumerating retained canonical cells produces strictly increasing explicit row positions. |
| `canonical_sparse_map_reconstructs_dense_exactly` | `twed/theories/Metric/TimestampedProductIndex.v` | Reintroducing exact vertical omissions reconstructs every dense recurrence cell. |
| `canonical_sparse_residual_is_exact_and_no_larger_than_dense` | `twed/theories/Metric/TimestampedProductIndex.v` | Sparse reconstruction is exact and retains no more explicit positions than the dense column. |
| `absent_transition_is_recomputed_on_demand` | `twed/theories/Metric/TimestampedProductIndex.v` | An unobserved state/token pair is computed lazily rather than materialized in advance. |
| `exact_cached_transition_refines_recomputation` | `twed/theories/Metric/TimestampedProductIndex.v` | A cache hit is transparent when it stores the exact complete successor, including a dead successor. |
| `cursor_page_then_resume_equals_uninterrupted_edges` | `twed/theories/Metric/TimestampedProductIndex.v` | A DFS edge page followed by its cursor suffix equals the immutable uninterrupted edge sequence. |
| `cursor_advance_preserves_edge_accounting` | `twed/theories/Metric/TimestampedProductIndex.v` | Cursor advance transfers edges from remaining to consumed without loss or duplication. |
| `zipper_push_preserves_revision_and_arena_reference` | `twed/theories/Metric/TimestampedProductIndex.v` | Descending pushes a child on the same captured revision with an in-bounds compact state ID. |
| `zipper_pop_preserves_revision_and_arena_references` | `twed/theories/Metric/TimestampedProductIndex.v` | Iterative pop preserves all remaining revision and arena-reference invariants. |
| `retained_product_memory_is_bounded_by_explicit_ceilings` | `twed/theories/Metric/TimestampedProductIndex.v` | Frames, append-only residuals, sparse positions, and observed cache entries are bounded by their explicit ceilings, not by live depth alone. |

### Complete elastic snapshots

| Theorem | Location | Description |
|---|---|---|
| `semantics_require_checksum` | `temporal_automata/theories/ElasticSnapshot.v` | No abstract semantic phase is available before checksum verification. |
| `acceptance_requires_checksum` | `temporal_automata/theories/ElasticSnapshot.v` | An accepted snapshot necessarily passed checksum verification. |
| `visible_manifest_names_sealed_generation` | `temporal_automata/theories/ElasticSnapshot.v` | Publish-last ordering prevents a visible manifest from naming an unsealed generation. |
| `exact_key_bijection_permutation` | `temporal_automata/theories/ElasticSnapshot.v` | Duplicate-free bucket and terminal key sets with equal membership are permutations. |
| `exact_key_bijection_cardinality` | `temporal_automata/theories/ElasticSnapshot.v` | The exact finite key bijection implies equal bucket and terminal cardinality. |
| `changed_manifest_invalidates_semantic_identity` | `temporal_automata/theories/ElasticSnapshot.v` | Unequal canonical manifest bytes cannot retain abstract identity equality. |

These theorems form an abstract protocol/finite-set island. Cryptographic
collision resistance, filesystem behavior, persistent-trie correctness, and
Rust correspondence remain explicit trust boundaries.

### Replayable exact-range certificates

| Theorem | Location | Description |
|---|---|---|
| `binding_eqb_true_iff` | `temporal_automata/theories/RangeCertificates.v` | Boolean binding equality is exact for snapshot, query words, and cutoff. |
| `replay_binds_exact_query_cutoff_and_snapshot` | `temporal_automata/theories/RangeCertificates.v` | Successful replay is bound to the recomputed query, cutoff, and snapshot identity. |
| `replay_reproduces_canonical_evidence` | `temporal_automata/theories/RangeCertificates.v` | Successful replay reproduces the complete canonical evidence stream. |
| `replay_validates_every_k1_through_k4_decision` | `temporal_automata/theories/RangeCertificates.v` | Replay revalidates every K1 through K4 decision rather than trusting recorded bounds. |
| `replay_survivors_are_exactly_recomputed_survivors` | `temporal_automata/theories/RangeCertificates.v` | Replayed survivor IDs equal exact recomputation. |
| `any_evidence_mutation_is_rejected` | `temporal_automata/theories/RangeCertificates.v` | Any changed evidence field or order fails replay. |
| `any_binding_mutation_is_rejected` | `temporal_automata/theories/RangeCertificates.v` | Any changed query/cutoff/snapshot binding fails replay. |
| `replay_never_exceeds_any_declared_ceiling` | `temporal_automata/theories/RangeCertificates.v` | Accepted replay respects record, path, work, and witness ceilings. |
| `record_limit_violation_fails_closed` | `temporal_automata/theories/RangeCertificates.v` | Record-limit overflow cannot replay successfully. |
| `path_limit_violation_fails_closed` | `temporal_automata/theories/RangeCertificates.v` | Path-byte overflow cannot replay successfully. |
| `work_limit_violation_fails_closed` | `temporal_automata/theories/RangeCertificates.v` | Work-unit overflow cannot replay successfully. |
| `witness_limit_violation_fails_closed` | `temporal_automata/theories/RangeCertificates.v` | Witness-byte overflow cannot replay successfully. |
| `equal_inputs_construct_equal_certificates` | `temporal_automata/theories/RangeCertificates.v` | Equal canonical inputs construct structurally equal certificates. |
| `certificate_is_issued_only_after_exhaustion` | `temporal_automata/theories/RangeCertificates.v` | A complete certificate is issued only in the exhausted phase. |
| `running_query_cannot_issue_complete_empty` | `temporal_automata/theories/RangeCertificates.v` | A running query cannot issue complete-empty evidence. |
| `failed_query_cannot_issue_complete_empty` | `temporal_automata/theories/RangeCertificates.v` | A failed query cannot issue complete-empty evidence. |

### Reusable exact-workspace resources and classifications

| Theorem | Location | Description |
|---|---|---|
| `plan_first_peak_is_max` | `temporal_automata/theories/ExactWorkspaceResources.v` | Plan-first construction peaks at the larger of plan construction and the retained plan plus frontier. |
| `accepted_construction_preflight_is_within_limit` | `temporal_automata/theories/ExactWorkspaceResources.v` | Successful exact-boundary preflight implies the construction peak is within the ceiling. |
| `rejected_construction_preflight_exceeds_limit` | `temporal_automata/theories/ExactWorkspaceResources.v` | Rejected exact-boundary preflight means the declared peak exceeds the ceiling. |
| `post_construction_peak_reduces_to_plan_or_live_state` | `temporal_automata/theories/ExactWorkspaceResources.v` | The session peak is the larger of plan construction and retained workspace plus later live state. |
| `accepted_post_construction_state_is_within_limit` | `temporal_automata/theories/ExactWorkspaceResources.v` | Accepted later arena/queue state plus retained workspace remains within the ceiling. |
| `candidate_reuse_preserves_retained_storage` | `temporal_automata/theories/ExactWorkspaceResources.v` | Reset-and-reuse cannot grow logical workspace retention with candidate count or length. |
| `structural_impossibility_is_no_finite_alignment` | `temporal_automata/theories/ExactWorkspaceResources.v` | Structural impossibility takes priority over numeric TOP for every cutoff and observation. |
| `finite_cutoff_top_is_above` | `temporal_automata/theories/ExactWorkspaceResources.v` | TOP is safely outside a finite cutoff. |
| `top_cutoff_top_fails_closed` | `temporal_automata/theories/ExactWorkspaceResources.v` | TOP under an unbounded cutoff is ambiguous with overflow and fails closed. |

### Approximate MSM evidence boundary

| Theorem | Location | Description |
|---|---|---|
| `exhaustive_tag_if_and_only_if_full_reranking` | `temporal_automata/theories/ApproximateMsmEvidence.v` | The strict API constructs exhaustive evidence exactly when candidate and exact-decision coverage both equal index size. |
| `classified_success_proves_recall_if_and_only_if_full_reranking` | `temporal_automata/theories/ApproximateMsmEvidence.v` | Runtime recall evidence is equivalent to full exact reranking. |
| `empty_advisory_never_proves_absence` | `temporal_automata/theories/ApproximateMsmEvidence.v` | Empty heuristic advice remains structurally unable to prove absence. |
| `every_mapped_emission_is_exact` | `temporal_automata/theories/ApproximateMsmEvidence.v` | Every emitted candidate receives its score from the exact verifier authority. |
| `proper_candidate_pool_has_no_recall_certificate` | `temporal_automata/theories/ApproximateMsmEvidence.v` | Inspecting a proper heuristic subset cannot produce a recall certificate. |

### Generalized finite-lookback scheduling

| Obligation | Location | Description |
|---|---|---|
| `finite_lookback_rows_are_stream_length_independent` | `verus/generalized_automaton.rs` | The $`r+1`$ retained rows depend on maximum target consumption rather than consumed stream length, and contain every target-consuming predecessor. |
| `generalized_predecessor_is_topologically_earlier` | `verus/generalized_automaton.rs` | Every non-zero operation reaches a lexicographically later alignment coordinate; source-only predecessors are earlier in the current scratch row. |

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

### Ordered costs and binary64 roundoff

| Theorem | Location | Description |
|---|---|---|
| `weighted_l1_associative` | `core/theories/Conformance/CostMonoid.v` | Mathematical non-negative real addition with explicit top is associative; this is not a bitwise `f64` claim. |
| `binary64_round_error_components` | `core/theories/Conformance/WeightedCostFloat.v` | Flocq decomposes round-to-nearest-even binary64 error into bounded relative and gradual-underflow components. |
| `binary64_round_absolute_error` | `core/theories/Conformance/WeightedCostFloat.v` | The component theorem implies a one-round absolute-plus-relative bound. |
| `two_rounded_additions_reassociation_envelope` | `core/theories/Conformance/WeightedCostFloat.v` | Any rounder satisfying the one-round contract has a proved symbolic three-term reassociation envelope. |
| `binary64_three_term_reassociation_envelope` | `core/theories/Conformance/WeightedCostFloat.v` | Instantiating the composition theorem with Flocq's binary64 rounder bounds finite three-term regrouping error without asserting exact associativity. |

### Exact multi-kind Dyck correction and binary persistence

| Theorem | Location | Description |
|---|---|---|
| `correction_target_is_dyck` | `core/theories/Conformance/DyckCorrection.v` | Every reconstruction branch produces a kind-sensitive balanced target. |
| `zero_cost_correction_is_balanced_identity` | `core/theories/Conformance/DyckCorrection.v` | A zero-cost witness preserves the source exactly and proves it is Dyck. |
| `every_source_has_a_correction` | `core/theories/Conformance/DyckCorrection.v` | Deleting every token supplies a total upper-bound witness. |
| `interval_recurrence_is_globally_exact` | `core/theories/Conformance/DyckCorrection.v` | With strict subinterval minima already filled, the least runtime branch cost is equivalent to the global minimum over all correction trees. |
| `finite_functional_minimum` | `core/theories/Conformance/DyckCorrection.v` | A nonempty finite family with one natural cost per descriptor has a constructively selected least cost; no classical choice is used. |
| `every_source_has_an_exact_minimum` | `core/theories/Conformance/DyckCorrection.v` | Strong interval-length induction establishes an attained exact correction minimum for every source. |
| `strict_subintervals_always_minimized` | `core/theories/Conformance/DyckCorrection.v` | The runtime fill-order premise follows for every interval rather than remaining a caller assumption. |
| `correction_target_length_is_bounded` | `core/theories/Conformance/DyckCorrection.v` | Every reconstructed target has length at most twice the source length, justifying the exhaustive-oracle cutoff. |
| `nonempty_dyck_first_pair_decomposition` | `core/theories/Conformance/DyckCorrection.v` | Every nonempty typed-Dyck word has the first-pair decomposition enumerated by the interval recurrence. |
| `correction_tree_is_standard_alignment` | `core/theories/Conformance/DyckCorrection.v` | Every reconstruction tree denotes an ordinary unit-cost Levenshtein alignment with exactly the same cost. |
| `standard_alignment_normalizes_to_correction_tree` | `core/theories/Conformance/DyckCorrection.v` | Every standard alignment to a typed-Dyck target normalizes to a reconstruction tree at no greater cost. |
| `correction_minimum_equals_dyck_levenshtein_minimum` | `core/theories/Conformance/DyckCorrection.v` | The algorithm-shaped minimum is extensionally equal to the independent standard-Levenshtein minimum over the typed Dyck language. |
| `interval_recurrence_is_exact_standard_dyck_distance` | `core/theories/Conformance/DyckCorrection.v` | The increasing-interval invariant refines the runtime recurrence directly to the independent language-distance specification. |
| `interval_recurrence_is_unconditionally_exact_standard_dyck_distance` | `core/theories/Conformance/DyckCorrection.v` | Finite descriptor enumeration and strong induction discharge the interval premise and prove end-to-end exactness against ordinary Levenshtein semantics. |
| `every_source_has_an_exact_standard_dyck_distance` | `core/theories/Conformance/DyckCorrection.v` | Every source has an attained minimum distance to the complete typed-Dyck language. |
| `diagnostic_rename_is_semantics_preserving` | `core/theories/Conformance/OperationSetSerialization.v` | Operation behavior depends on its applicability tag, never its diagnostic name. |
| `accepted_envelope_is_exact_and_bounded` | `core/theories/Conformance/OperationSetSerialization.v` | Acceptance implies the magic/version/flags contract, exact consumption, semantic validation, and resource bounds. |
| `trailing_payload_bytes_are_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | An envelope with bytes beyond its declared payload cannot be accepted. |
| `accepted_protobuf_is_preflight_bounded` | `core/theories/Conformance/OperationSetSerialization.v` | Every allocation-bearing protobuf count is within policy before semantic admission. |
| `protobuf_over_limit_never_reaches_prost` | `core/theories/Conformance/OperationSetSerialization.v` | An operation, pair, or text count above policy cannot pass the pre-allocation gate. |
| `protobuf_weight_bits_round_trip_exactly` | `core/theories/Conformance/OperationSetSerialization.v` | The protobuf fixed64 weight field preserves all 64 IEEE-754 bits. |
| `trailing_compressed_data_is_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | A gzip wrapper that does not consume the complete supplied input cannot be accepted. |
| `decompression_over_limit_is_rejected` | `core/theories/Conformance/OperationSetSerialization.v` | Inflated bytes above the inner-format ceiling are rejected before semantic decode. |
| `accepted_bincode_bytes_have_the_exact_runtime_envelope` | `core/theories/Conformance/OperationSetByteParsers.v` | Successful executable parsing derives magic, little-endian version and flags, exact declared payload consumption, and the payload ceiling from concrete bytes. |
| `parsed_varint_is_bounded_and_consumes_one_prefix` | `core/theories/Conformance/OperationSetByteParsers.v` | Every successful protobuf varint consumes one prefix of at most ten bytes and fits `uint64`. |
| `parsed_length_delimited_field_consumes_exactly_its_prefix_and_body` | `core/theories/Conformance/OperationSetByteParsers.v` | A successful length-delimited parse partitions the input exactly into length prefix, body, and unconsumed suffix. |
| `accepted_protobuf_bytes_are_wire_parsed_before_allocation` | `core/theories/Conformance/OperationSetByteParsers.v` | Successful nested wire preflight implies concrete parse evidence and every payload, operation, name, pair, and text bound. |
| `accepted_gzip_adapter_observation_is_complete_bounded_and_inner_valid` | `core/theories/Conformance/OperationSetByteParsers.v` | Given the explicit flate2 observation boundary, admission implies gzip identity, complete input consumption, bounded output, valid checksum observation, and inner acceptance. |
| `concrete_bincode_and_validated_payload_refine_abstract_admission` | `core/theories/Conformance/OperationSetByteParsers.v` | Concrete header parsing composed with bounded semantic payload validation satisfies the abstract bincode admission contract. |
| `concrete_protobuf_and_validated_message_refine_abstract_admission` | `core/theories/Conformance/OperationSetByteParsers.v` | Concrete nested wire preflight composed with supported-format and semantic validation satisfies the abstract protobuf admission contract. |
| `concrete_gzip_adapter_refines_abstract_admission` | `core/theories/Conformance/OperationSetByteParsers.v` | The crate-owned checks on a trusted decompressor observation refine the abstract single-member gzip admission contract. |

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
| `resolve_advances_exact_endpoint` | `damerau/theories/DamerauStreaming.v` | Resolution advances from the stored origin by exactly $`\delta+1`$ and preserves cost. |
| `macro_cost_equivalent` | `damerau/theories/DamerauStreaming.v` | The streaming charge equals the Lowrance–Wagner macro term. |
| `mixed_continuations_never_subsume` | `damerau/theories/DamerauStreaming.v` | Normal and pending residual languages are incomparable in both directions. |
| `pending_subsumption_requires_same_key` | `damerau/theories/DamerauStreaming.v` | Pending dominance exposes non-greater cost and equality of origin and delta. |
| `frontier_quadratic_bound` | `damerau/theories/DamerauStreaming.v` | At most $`k`$ diagonals times $`k`$ deltas yields the $`k^2`$ frontier envelope. |

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

### Unit-grid and explicit-timestamp TWED

| Theorem | Location | Description |
|---|---|---|
| `match_interval_admissible` | `twed/theories/Metric/TwedProperties.v` | The separable interval match leaf lower-bounds every represented concrete unit-grid TWED match. |
| `twed_step_monotone` | `twed/theories/Metric/TwedProperties.v` | Ordering every predecessor and local leaf orders the complete additive recurrence cell. |
| `twed_length_lower_bound` | `twed/theories/Metric/TwedProperties.v` | The unavoidable deletion count makes the gap-penalty length bound admissible for every well-formed script. |
| `physical_delete_is_nonnegative` | `twed/theories/Metric/TwedProperties.v` | Monotone physical timestamps and validated parameters make an explicit-time deletion leaf nonnegative. |
| `unit_elapsed_physical_delete_is_unit_grid` | `twed/theories/Metric/TwedProperties.v` | A one-unit physical timestamp step reproduces the unit-grid deletion leaf exactly. |
| `physical_match_is_nonnegative` | `twed/theories/Metric/TwedProperties.v` | Every explicit-time match leaf is nonnegative under positive stiffness. |
| `validated_timestamp_step_has_nonnegative_elapsed_time` | `twed/theories/Metric/TwedProperties.v` | A strictly increasing online timestamp produces a nonnegative elapsed-time term. |

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
