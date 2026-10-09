# Vinary Tree C interop binding

This package exposes the language-native representation of the stable Vinary Tree resource ABI. It is the neutral handoff layer used by dictionary, automaton, and WFST packages; it owns no algorithm-specific policy.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | C |
| Languages/runtime | C17 and C23 |
| Support tier | Tier 1 |
| Distribution | CMake package `liblevenshtein` and `pkg-config` module `liblevenshtein` |
| Native boundary | The public C header calls the `llev_*` ABI exported by the shared or static native library. |
| Canonical facade source | [`include/liblevenshtein.h`](../../include/liblevenshtein.h) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/c/tests/cross_project_snapshot.c`](../../bindings/c/tests/cross_project_snapshot.c). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
cc -std=c17 -Wall -Wextra -Werror -Iinclude -I../libdictenstein/include -I../vinary-tree-interop/include bindings/c/tests/cross_project_snapshot.c -Ltarget/debug -lliblevenshtein -L../libdictenstein/target/debug -llibdictenstein -Wl,-rpath,"$PWD/target/debug" -Wl,-rpath,"$PWD/../libdictenstein/target/debug" -o target/c-cross-project-snapshot && target/c-cross-project-snapshot
```

Examples deliberately construct or receive resources through public project
packages. They never import private Rust modules, depend on object layout, or
reach behind the stable C/resource ABIs.

## Public API and data model

The idiomatic facade groups the stable surface into these concepts:

| Concept | Semantics |
|---|---|
| Dictionary resource | A retained `vt.dictionary.v1` capability. Construction and mutation belong to a producer such as libdictenstein. |
| Transducer | Immutable query configuration plus a retained dictionary provider; construction is constant-time with respect to dictionary size. |
| Query cursor | A one-shot traversal over the immutable dictionary revision captured at query start. |
| Match/batch | Owned matches are stable host values; a borrowed batch is valid only inside its documented callback or lease interval. |

### Automaton selection

| Algorithm | Edit semantics | Metric? | Typical use |
|---|---|---:|---|
| Standard | Insert, delete, and substitute | yes | General spelling correction |
| Transposition | Optimal string alignment with adjacent swaps | no | Typographical swaps when metric-tree laws are unnecessary |
| Merge and split | Standard edits plus symmetric two-to-one and one-to-two edits | yes | Optical character recognition and segmentation errors |
| Damerau-Levenshtein | Unrestricted, history-composable adjacent transpositions | yes | True Damerau matching and metric indexes |

The transposition and unrestricted Damerau variants are deliberately distinct:
for example, optimal string alignment assigns distance 3 from `CA` to `ABC`,
while unrestricted Damerau-Levenshtein assigns distance 2. Select the algorithm
when constructing the transducer; all query domains and snapshot laws remain the
same.

All strings and arrays use pointer-plus-length descriptors; embedded zero bytes and empty terms are valid where the function contract permits them. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `llev_abi_version` | `llev_abi_version` | ABI compatibility and feature discovery |
| `llev_affine_costs_validate` | `llev_affine_costs_validate` | project ABI operation |
| `llev_affine_gap_distance` | `llev_affine_gap_distance` | project ABI operation |
| `llev_affine_gap_distance_bytes` | `llev_affine_gap_distance_bytes` | project ABI operation |
| `llev_affine_gap_distance_bytes_threshold` | `llev_affine_gap_distance_bytes_threshold` | project ABI operation |
| `llev_affine_gap_distance_threshold` | `llev_affine_gap_distance_threshold` | project ABI operation |
| `llev_affine_gap_distance_u64` | `llev_affine_gap_distance_u64` | project ABI operation |
| `llev_affine_gap_distance_u64_threshold` | `llev_affine_gap_distance_u64_threshold` | project ABI operation |
| `llev_api_revision` | `llev_api_revision` | ABI compatibility and feature discovery |
| `llev_approx_msm_index_free` | `llev_approx_msm_index_free` | project ABI operation |
| `llev_approx_msm_index_freeze` | `llev_approx_msm_index_freeze` | project ABI operation |
| `llev_approx_msm_index_insert` | `llev_approx_msm_index_insert` | project ABI operation |
| `llev_approx_msm_index_new` | `llev_approx_msm_index_new` | project ABI operation |
| `llev_approx_msm_index_query_knn` | `llev_approx_msm_index_query_knn` | project ABI operation |
| `llev_build_features` | `llev_build_features` | ABI compatibility and feature discovery |
| `llev_cost_cursor_free` | `llev_cost_cursor_free` | project ABI operation |
| `llev_cost_cursor_next_batch` | `llev_cost_cursor_next_batch` | project ABI operation |
| `llev_cost_cursor_reduce` | `llev_cost_cursor_reduce` | project ABI operation |
| `llev_cost_cursor_release_batch` | `llev_cost_cursor_release_batch` | project ABI operation |
| `llev_damerau_distance` | `llev_damerau_distance` | standalone exact or thresholded distance |
| `llev_damerau_distance_bytes` | `llev_damerau_distance_bytes` | standalone exact or thresholded distance |
| `llev_damerau_distance_bytes_threshold` | `llev_damerau_distance_bytes_threshold` | standalone exact or thresholded distance |
| `llev_damerau_distance_threshold` | `llev_damerau_distance_threshold` | standalone exact or thresholded distance |
| `llev_damerau_distance_u64` | `llev_damerau_distance_u64` | standalone exact or thresholded distance |
| `llev_damerau_distance_u64_threshold` | `llev_damerau_distance_u64_threshold` | standalone exact or thresholded distance |
| `llev_distance` | `llev_distance` | standalone exact or thresholded distance |
| `llev_distance_bytes` | `llev_distance_bytes` | standalone exact or thresholded distance |
| `llev_distance_bytes_threshold` | `llev_distance_bytes_threshold` | standalone exact or thresholded distance |
| `llev_distance_threshold` | `llev_distance_threshold` | standalone exact or thresholded distance |
| `llev_distance_u64` | `llev_distance_u64` | standalone exact or thresholded distance |
| `llev_distance_u64_threshold` | `llev_distance_u64_threshold` | standalone exact or thresholded distance |
| `llev_generalized_automaton_evaluate_utf8` | `llev_generalized_automaton_evaluate_utf8` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_automaton_free` | `llev_generalized_automaton_free` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_automaton_new` | `llev_generalized_automaton_new` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_online_advance` | `llev_generalized_online_advance` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_online_free` | `llev_generalized_online_free` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_online_new_utf8` | `llev_generalized_online_new_utf8` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_generalized_online_observation` | `llev_generalized_online_observation` | runtime generalized-automaton lifecycle and prefix evaluation |
| `llev_hamming_distance` | `llev_hamming_distance` | project ABI operation |
| `llev_hamming_distance_bytes` | `llev_hamming_distance_bytes` | project ABI operation |
| `llev_hamming_distance_bytes_threshold` | `llev_hamming_distance_bytes_threshold` | project ABI operation |
| `llev_hamming_distance_threshold` | `llev_hamming_distance_threshold` | project ABI operation |
| `llev_hamming_distance_u64` | `llev_hamming_distance_u64` | project ABI operation |
| `llev_hamming_distance_u64_threshold` | `llev_hamming_distance_u64_threshold` | project ABI operation |
| `llev_indel_distance` | `llev_indel_distance` | project ABI operation |
| `llev_indel_distance_bytes` | `llev_indel_distance_bytes` | project ABI operation |
| `llev_indel_distance_bytes_threshold` | `llev_indel_distance_bytes_threshold` | project ABI operation |
| `llev_indel_distance_threshold` | `llev_indel_distance_threshold` | project ABI operation |
| `llev_indel_distance_u64` | `llev_indel_distance_u64` | project ABI operation |
| `llev_indel_distance_u64_threshold` | `llev_indel_distance_u64_threshold` | project ABI operation |
| `llev_jaro_similarity_utf8` | `llev_jaro_similarity_utf8` | project ABI operation |
| `llev_keogh_plan_bounds_at` | `llev_keogh_plan_bounds_at` | project ABI operation |
| `llev_keogh_plan_free` | `llev_keogh_plan_free` | project ABI operation |
| `llev_keogh_plan_new` | `llev_keogh_plan_new` | project ABI operation |
| `llev_keogh_plan_score` | `llev_keogh_plan_score` | project ABI operation |
| `llev_last_error_message` | `llev_last_error_message` | typed failure diagnostics |
| `llev_merge_and_split_distance` | `llev_merge_and_split_distance` | standalone merge-and-split distance |
| `llev_merge_and_split_distance_bytes` | `llev_merge_and_split_distance_bytes` | standalone merge-and-split distance |
| `llev_merge_and_split_distance_bytes_threshold` | `llev_merge_and_split_distance_bytes_threshold` | standalone merge-and-split distance |
| `llev_merge_and_split_distance_threshold` | `llev_merge_and_split_distance_threshold` | standalone merge-and-split distance |
| `llev_merge_and_split_distance_u64` | `llev_merge_and_split_distance_u64` | standalone merge-and-split distance |
| `llev_merge_and_split_distance_u64_threshold` | `llev_merge_and_split_distance_u64_threshold` | standalone merge-and-split distance |
| `llev_operation_costs_preset` | `llev_operation_costs_preset` | project ABI operation |
| `llev_operation_costs_validate` | `llev_operation_costs_validate` | project ABI operation |
| `llev_owned_bytes_free` | `llev_owned_bytes_free` | versioned compiled-byte ownership |
| `llev_owned_string_free` | `llev_owned_string_free` | owned result-string release |
| `llev_phonetic_articulatory_distance` | `llev_phonetic_articulatory_distance` | articulatory phonetic distance |
| `llev_phonetic_articulatory_edit_distance` | `llev_phonetic_articulatory_edit_distance` | articulatory phonetic distance |
| `llev_phonetic_candidates_free` | `llev_phonetic_candidates_free` | owned normalized-dictionary candidate release |
| `llev_phonetic_chars_with_features` | `llev_phonetic_chars_with_features` | IPA feature classification and relations |
| `llev_phonetic_dictionary_free` | `llev_phonetic_dictionary_free` | normalized phonetic dictionary construction, query, and updates |
| `llev_phonetic_dictionary_new` | `llev_phonetic_dictionary_new` | normalized phonetic dictionary construction, query, and updates |
| `llev_phonetic_dictionary_query` | `llev_phonetic_dictionary_query` | normalized phonetic dictionary construction, query, and updates |
| `llev_phonetic_dictionary_update` | `llev_phonetic_dictionary_update` | normalized phonetic dictionary construction, query, and updates |
| `llev_phonetic_expand` | `llev_phonetic_expand` | bounded reverse phonetic expansion |
| `llev_phonetic_expand_feature_based` | `llev_phonetic_expand_feature_based` | IPA feature-driven expansion |
| `llev_phonetic_expand_with_costs` | `llev_phonetic_expand_with_costs` | bounded reverse phonetic expansion |
| `llev_phonetic_feature_relation` | `llev_phonetic_feature_relation` | IPA feature classification and relations |
| `llev_phonetic_feature_set_distance` | `llev_phonetic_feature_set_distance` | IPA feature classification and relations |
| `llev_phonetic_features` | `llev_phonetic_features` | IPA feature classification and relations |
| `llev_phonetic_grep_distance_config` | `llev_phonetic_grep_distance_config` | word-boundary phonetic search and configuration |
| `llev_phonetic_grep_free` | `llev_phonetic_grep_free` | word-boundary phonetic search and configuration |
| `llev_phonetic_grep_matches` | `llev_phonetic_grep_matches` | word-boundary phonetic search and configuration |
| `llev_phonetic_grep_new` | `llev_phonetic_grep_new` | word-boundary phonetic search and configuration |
| `llev_phonetic_grep_scan_line` | `llev_phonetic_grep_scan_line` | word-boundary phonetic search and configuration |
| `llev_phonetic_grep_scan_text` | `llev_phonetic_grep_scan_text` | word-boundary phonetic search and configuration |
| `llev_phonetic_online_free` | `llev_phonetic_online_free` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_matches_free` | `llev_phonetic_online_matches_free` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_new` | `llev_phonetic_online_new` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_normalized_query` | `llev_phonetic_online_normalized_query` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_scan` | `llev_phonetic_online_scan` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_stream_feed` | `llev_phonetic_online_stream_feed` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_stream_finish` | `llev_phonetic_online_stream_finish` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_stream_free` | `llev_phonetic_online_stream_free` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_online_stream_new` | `llev_phonetic_online_stream_new` | character-level phonetic search and scanner lifecycle |
| `llev_phonetic_pattern_compile_llre` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `llev_phonetic_pattern_compile_regex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `llev_phonetic_pattern_free` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `llev_phonetic_pattern_from_bytes` | `llev_phonetic_pattern_from_bytes` | versioned compiled phonetic-pattern bytes |
| `llev_phonetic_pattern_load_llre_file` | `llev_phonetic_pattern_load_llre_file` | trusted .llre loading with native imports |
| `llev_phonetic_pattern_matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `llev_phonetic_pattern_size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `llev_phonetic_pattern_to_bytes` | `llev_phonetic_pattern_to_bytes` | versioned compiled phonetic-pattern bytes |
| `llev_phonetic_rules_apply` | `llev_phonetic_rules_apply` | phonetic rule-set lifecycle and rewriting |
| `llev_phonetic_rules_builtin` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `llev_phonetic_rules_free` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `llev_phonetic_rules_from_bytes` | `llev_phonetic_rules_from_bytes` | versioned compiled phonetic-rule bytes |
| `llev_phonetic_rules_len` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `llev_phonetic_rules_load_file` | `llev_phonetic_rules_load_file` | trusted .llev loading with native includes |
| `llev_phonetic_rules_parse` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `llev_phonetic_rules_to_bytes` | `llev_phonetic_rules_to_bytes` | versioned compiled phonetic-rule bytes |
| `llev_phonetic_similar_chars` | `llev_phonetic_similar_chars` | IPA feature classification and relations |
| `llev_phonetic_syllable_boundaries` | `llev_phonetic_syllable_boundaries` | syllable count and boundary heuristics |
| `llev_phonetic_syllable_count` | `llev_phonetic_syllable_count` | syllable count and boundary heuristics |
| `llev_phonetic_token_free` | `llev_phonetic_token_free` | token-sequence phonetic matching and detail ownership |
| `llev_phonetic_token_matches_free` | `llev_phonetic_token_matches_free` | token-sequence phonetic matching and detail ownership |
| `llev_phonetic_token_new` | `llev_phonetic_token_new` | token-sequence phonetic matching and detail ownership |
| `llev_phonetic_token_scan` | `llev_phonetic_token_scan` | token-sequence phonetic matching and detail ownership |
| `llev_phonetic_transducer_feed` | `llev_phonetic_transducer_feed` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_transducer_finish` | `llev_phonetic_transducer_finish` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_transducer_free` | `llev_phonetic_transducer_free` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_transducer_new` | `llev_phonetic_transducer_new` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_transducer_normalize` | `llev_phonetic_transducer_normalize` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_transducer_reset` | `llev_phonetic_transducer_reset` | incremental phonetic rewriting and lifecycle |
| `llev_phonetic_voicing_pair` | `llev_phonetic_voicing_pair` | IPA feature classification and relations |
| `llev_query_cache_clear` | `llev_query_cache_clear` | project ABI operation |
| `llev_query_cache_free` | `llev_query_cache_free` | project ABI operation |
| `llev_query_cache_new` | `llev_query_cache_new` | project ABI operation |
| `llev_query_cache_query_bytes` | `llev_query_cache_query_bytes` | project ABI operation |
| `llev_query_cache_query_u64` | `llev_query_cache_query_u64` | project ABI operation |
| `llev_query_cache_query_utf8` | `llev_query_cache_query_utf8` | project ABI operation |
| `llev_query_cache_reset_stats` | `llev_query_cache_reset_stats` | project ABI operation |
| `llev_query_cache_stats` | `llev_query_cache_stats` | project ABI operation |
| `llev_query_cursor_free` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `llev_query_cursor_next_batch` | `llev_query_cursor_next_batch` | streaming result traversal and batch leases |
| `llev_query_cursor_reduce` | `llev_query_cursor_reduce` | streaming result traversal and batch leases |
| `llev_query_cursor_release_batch` | `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `llev_soft_dtw_gradient` | `llev_soft_dtw_gradient` | project ABI operation |
| `llev_source_filter_index_free` | `llev_source_filter_index_free` | project ABI operation |
| `llev_source_filter_index_freeze` | `llev_source_filter_index_freeze` | project ABI operation |
| `llev_source_filter_index_insert` | `llev_source_filter_index_insert` | project ABI operation |
| `llev_source_filter_index_new` | `llev_source_filter_index_new` | project ABI operation |
| `llev_source_filter_index_query` | `llev_source_filter_index_query` | project ABI operation |
| `llev_source_filter_utf8` | `llev_source_filter_utf8` | project ABI operation |
| `llev_specialized_cursor_free` | `llev_specialized_cursor_free` | project ABI operation |
| `llev_specialized_cursor_next_batch` | `llev_specialized_cursor_next_batch` | project ABI operation |
| `llev_specialized_cursor_reduce` | `llev_specialized_cursor_reduce` | project ABI operation |
| `llev_specialized_cursor_release_batch` | `llev_specialized_cursor_release_batch` | project ABI operation |
| `llev_string_array_free` | `llev_string_array_free` | legacy owned-string plumbing |
| `llev_string_dup` | `llev_string_dup` | legacy owned-string plumbing |
| `llev_string_free` | `llev_string_free` | legacy owned-string plumbing |
| `llev_temporal_alignment_free` | `llev_temporal_alignment_free` | project ABI operation |
| `llev_temporal_alignment_new` | `llev_temporal_alignment_new` | project ABI operation |
| `llev_temporal_alignment_page` | `llev_temporal_alignment_page` | project ABI operation |
| `llev_temporal_alignment_replay` | `llev_temporal_alignment_replay` | project ABI operation |
| `llev_temporal_certificate_evidence_at` | `llev_temporal_certificate_evidence_at` | project ABI operation |
| `llev_temporal_certificate_free` | `llev_temporal_certificate_free` | project ABI operation |
| `llev_temporal_certificate_info` | `llev_temporal_certificate_info` | project ABI operation |
| `llev_temporal_certificate_matches` | `llev_temporal_certificate_matches` | project ABI operation |
| `llev_temporal_certificate_query_bits` | `llev_temporal_certificate_query_bits` | project ABI operation |
| `llev_temporal_certificate_verify` | `llev_temporal_certificate_verify` | project ABI operation |
| `llev_temporal_distance` | `llev_temporal_distance` | project ABI operation |
| `llev_temporal_index_cursor_free` | `llev_temporal_index_cursor_free` | project ABI operation |
| `llev_temporal_index_cursor_next_batch` | `llev_temporal_index_cursor_next_batch` | project ABI operation |
| `llev_temporal_index_free` | `llev_temporal_index_free` | project ABI operation |
| `llev_temporal_index_freeze` | `llev_temporal_index_freeze` | project ABI operation |
| `llev_temporal_index_insert` | `llev_temporal_index_insert` | project ABI operation |
| `llev_temporal_index_new` | `llev_temporal_index_new` | project ABI operation |
| `llev_temporal_index_query_certified` | `llev_temporal_index_query_certified` | project ABI operation |
| `llev_temporal_index_query_erp_automaton_range` | `llev_temporal_index_query_erp_automaton_range` | project ABI operation |
| `llev_temporal_index_query_knn` | `llev_temporal_index_query_knn` | project ABI operation |
| `llev_temporal_index_query_range` | `llev_temporal_index_query_range` | project ABI operation |
| `llev_temporal_knn_cursor_free` | `llev_temporal_knn_cursor_free` | project ABI operation |
| `llev_temporal_knn_cursor_next_batch` | `llev_temporal_knn_cursor_next_batch` | project ABI operation |
| `llev_temporal_lower_bound` | `llev_temporal_lower_bound` | project ABI operation |
| `llev_temporal_online_advance` | `llev_temporal_online_advance` | project ABI operation |
| `llev_temporal_online_free` | `llev_temporal_online_free` | project ABI operation |
| `llev_temporal_online_new` | `llev_temporal_online_new` | project ABI operation |
| `llev_temporal_online_observation` | `llev_temporal_online_observation` | project ABI operation |
| `llev_temporal_online_scratch_bytes` | `llev_temporal_online_scratch_bytes` | project ABI operation |
| `llev_timestamped_twed_alignment_new` | `llev_timestamped_twed_alignment_new` | project ABI operation |
| `llev_timestamped_twed_alignment_replay` | `llev_timestamped_twed_alignment_replay` | project ABI operation |
| `llev_timestamped_twed_cursor_free` | `llev_timestamped_twed_cursor_free` | project ABI operation |
| `llev_timestamped_twed_cursor_next_batch` | `llev_timestamped_twed_cursor_next_batch` | project ABI operation |
| `llev_timestamped_twed_distance` | `llev_timestamped_twed_distance` | project ABI operation |
| `llev_timestamped_twed_index_free` | `llev_timestamped_twed_index_free` | project ABI operation |
| `llev_timestamped_twed_index_freeze` | `llev_timestamped_twed_index_freeze` | project ABI operation |
| `llev_timestamped_twed_index_insert` | `llev_timestamped_twed_index_insert` | project ABI operation |
| `llev_timestamped_twed_index_new` | `llev_timestamped_twed_index_new` | project ABI operation |
| `llev_timestamped_twed_index_query_knn` | `llev_timestamped_twed_index_query_knn` | project ABI operation |
| `llev_timestamped_twed_index_query_range` | `llev_timestamped_twed_index_query_range` | project ABI operation |
| `llev_transducer_free` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `llev_transducer_new` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `llev_transducer_query_affine` | `llev_transducer_query_affine` | domain-preserving dictionary query |
| `llev_transducer_query_bytes` | `llev_transducer_query_bytes` | domain-preserving dictionary query |
| `llev_transducer_query_contextual_utf8` | `llev_transducer_query_contextual_utf8` | domain-preserving dictionary query |
| `llev_transducer_query_filtered_utf8` | `llev_transducer_query_filtered_utf8` | domain-preserving dictionary query |
| `llev_transducer_query_pattern` | `llev_transducer_query_pattern` | phonetic-pattern dictionary query |
| `llev_transducer_query_pruned_utf8` | `llev_transducer_query_pruned_utf8` | domain-preserving dictionary query |
| `llev_transducer_query_u64` | `llev_transducer_query_u64` | domain-preserving dictionary query |
| `llev_transducer_query_utf8` | `llev_transducer_query_utf8` | domain-preserving dictionary query |
| `llev_transducer_query_weighted` | `llev_transducer_query_weighted` | domain-preserving dictionary query |
| `llev_transducer_snapshot` | `llev_transducer_snapshot` | transducer lifecycle, snapshot, or domain metadata |
| `llev_transducer_unit_domain` | `llev_transducer_unit_domain` | transducer lifecycle, snapshot, or domain metadata |
| `llev_true_damerau_distance` | `llev_true_damerau_distance` | standalone true-Damerau distance |
| `llev_true_damerau_distance_bytes` | `llev_true_damerau_distance_bytes` | standalone true-Damerau distance |
| `llev_true_damerau_distance_bytes_threshold` | `llev_true_damerau_distance_bytes_threshold` | standalone true-Damerau distance |
| `llev_true_damerau_distance_threshold` | `llev_true_damerau_distance_threshold` | standalone true-Damerau distance |
| `llev_true_damerau_distance_u64` | `llev_true_damerau_distance_u64` | standalone true-Damerau distance |
| `llev_true_damerau_distance_u64_threshold` | `llev_true_damerau_distance_u64_threshold` | standalone true-Damerau distance |
| `llev_twed_length_lower_bound` | `llev_twed_length_lower_bound` | project ABI operation |
| `llev_universal_automaton_evaluate` | `llev_universal_automaton_evaluate` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_automaton_free` | `llev_universal_automaton_free` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_automaton_new` | `llev_universal_automaton_new` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_online_advance` | `llev_universal_online_advance` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_online_free` | `llev_universal_online_free` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_online_new` | `llev_universal_online_new` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_universal_online_observation` | `llev_universal_online_observation` | universal-automaton lifecycle, policies, and prefix evaluation |
| `llev_vector_box_box_lower_bound` | `llev_vector_box_box_lower_bound` | project ABI operation |
| `llev_vector_frechet_ground_distance` | `llev_vector_frechet_ground_distance` | project ABI operation |
| `llev_vector_frechet_ground_online_new` | `llev_vector_frechet_ground_online_new` | project ABI operation |
| `llev_vector_frechet_online_advance` | `llev_vector_frechet_online_advance` | project ABI operation |
| `llev_vector_frechet_online_free` | `llev_vector_frechet_online_free` | project ABI operation |
| `llev_vector_frechet_online_new` | `llev_vector_frechet_online_new` | project ABI operation |
| `llev_vector_frechet_online_observation` | `llev_vector_frechet_online_observation` | project ABI operation |
| `llev_vector_frechet_online_scratch_bytes` | `llev_vector_frechet_online_scratch_bytes` | project ABI operation |
| `llev_vector_metric_free` | `llev_vector_metric_free` | project ABI operation |
| `llev_vector_metric_new` | `llev_vector_metric_new` | project ABI operation |
| `llev_vector_point_box_lower_bound` | `llev_vector_point_box_lower_bound` | project ABI operation |
| `llev_vector_temporal_candidate_lower_bound` | `llev_vector_temporal_candidate_lower_bound` | project ABI operation |
| `llev_vector_temporal_distance` | `llev_vector_temporal_distance` | project ABI operation |
| `llev_vector_twed_interval_lower_bound` | `llev_vector_twed_interval_lower_bound` | project ABI operation |
| `llev_wallbreaker_cursor_cancel` | `llev_wallbreaker_cursor_cancel` | project ABI operation |
| `llev_wallbreaker_cursor_free` | `llev_wallbreaker_cursor_free` | project ABI operation |
| `llev_wallbreaker_cursor_next_batch` | `llev_wallbreaker_cursor_next_batch` | project ABI operation |
| `llev_wallbreaker_cursor_release_batch` | `llev_wallbreaker_cursor_release_batch` | project ABI operation |
| `llev_wallbreaker_free` | `llev_wallbreaker_free` | project ABI operation |
| `llev_wallbreaker_new_utf8` | `llev_wallbreaker_new_utf8` | project ABI operation |
| `llev_wallbreaker_query_utf8` | `llev_wallbreaker_query_utf8` | project ABI operation |
| `llev_wallbreaker_split_utf8` | `llev_wallbreaker_split_utf8` | project ABI operation |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `LlevStatus` | Typed native status or error carrier | Public facade type |
| `LlevAlgorithm` | Edit-distance algorithm selection | Public facade type |
| `LlevQueryOrder` | Result traversal ordering | Public facade type |
| `LlevPhoneticRuleSetKind` | Built-in phonetic rule-set selection | Public facade type |
| `LlevOperationApplicability` | Generalized-operation applicability selection | Public facade type |
| `LlevUniversalVariant` | Universal edit-automaton variant selection | Public facade type |
| `llev_query_cursor_next_batch` | One-shot owned-result iteration | Public facade protocol |
| `llev_query_cursor_reduce` | Bounded batch/reducer traversal | Public facade protocol |

Native operations omitted from the public-symbol table are deliberately
encapsulated by the facade. The generated completeness matrix records every
such operation with its reviewed rationale; an unreasoned absence fails CI.

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | The facade iterator protocol | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | The facade batch/reducer protocol | It amortizes the foreign boundary and keeps borrowed views inside one lexical lease. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Balance every successful constructor/retain with its documented free/release function and release each cursor batch before advancing.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Functions return `LlevStatus`; inspect the enum first and copy `llev_last_error_message()` before making another call on that thread.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Independent handles are reentrant. A query cursor and its current lease are single-consumer resources.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Reuse transducers for repeated queries against the same resource.
- Prefer streaming cursors to whole-result materialization.
- Prefer batch/reducer APIs when per-match boundary crossings dominate.
- Keep Unicode, byte, and token domains explicit to avoid transcoding.
- Measure native, WASM, and WASI paths independently; they have different
  startup and marshalling costs but identical query semantics.

No host wrapper should cache unbounded query results. Applications that add a
memo use a revision key and a hard entry/weight bound; eviction may be
approximate because all values remain derivable from the retained snapshot.

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model](../../docs/security/binding-trust-model.md).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md); never infer compatibility from a
package version alone.

When loading fails, check—in order—the documented runtime/toolchain version,
CPU/OS artifact, native-access permission, loader search path, dependent
interop package pin, and process-wide JavaScript runtime identity. When a query
fails after construction, report the typed status and copied diagnostic before
reducing the case to the smallest dictionary/query pair.

## Maintainer checklist

1. Update the machine-readable binding model before changing a public symbol.
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

<!-- END GENERATED BINDING OPERATIONS -->
