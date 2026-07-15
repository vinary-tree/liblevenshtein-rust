# Examples & Tutorials

This directory holds the **numbered tutorial series** for `liblevenshtein` — a guided
path from a first spell checker to a complete phonetic spellcheck application — plus an
index of **every runnable example** in [`examples/`](../../examples/). Each tutorial is
grounded in a real, compiling example, so every snippet you read is copied or condensed
from code you can run.

New to the library? Start with the main [README](../../README.md) for the conceptual
overview (Levenshtein automata, the dictionary family, feature flags), then follow the
tutorials below in order.

---

## Tutorial series

A progressive, eight-part walkthrough. Each part explains one concept, walks through its
backing example in a few annotated snippets, gives the exact `cargo run` command, and
embeds the relevant architecture diagrams.

| # | Tutorial | You'll learn | Backing example |
|---|---|---|---|
| 01 | [Getting Started](01-getting-started/README.md) | Build a spell checker from a dictionary + algorithm + transducer; `query` vs `query_with_distance` | `spell_checker.rs` |
| 02 | [Dictionaries](02-dictionaries/README.md) | Pick a backend for your access pattern; mutate a `DynamicDawg` at runtime; serialize to disk | `dynamic_dictionary.rs`, `serialization.rs` |
| 03 | [Algorithms & Ordering](03-algorithms/README.md) | `Standard` / `Transposition` / `MergeAndSplit`; distance-first ordered results and lazy top-$`k`$ | `ordered_query_demo.rs` |
| 04 | [Queries & Unicode](04-queries/README.md) | Unicode matching and zero-cost custom substitutions (diacritics, case-folding, kana) | `unicode_diacritics.rs` |
| 05 | [Values & Fuzzy Maps](05-values/README.md) | Attach values to terms; filter, prioritize, and prune by value during traversal | `fuzzy_maps_code_completion.rs` |
| 06 | [Contextual Completion](06-contextual/README.md) | Incremental drafts, checkpoints/undo, and hierarchical scope visibility | `contextual_completion.rs` |
| 07 | [Performance & Concurrency](07-performance/README.md) | Benchmark on a real 124k-word dictionary; backend trade-offs; the lock-free read model | `real_world_benchmark.rs` |
| 08 | [Real-World: Phonetic Spellcheck](08-real-world/README.md) | A complete app: phonetic normalization × edit distance, dual-index dictionary, formally verified rules | `phonetic_spellcheck/` |

---

## All runnable examples

The full set of programs under [`examples/`](../../examples/), grouped by theme. Run any
of them with `cargo run --example <name>`; entries that need Cargo **features** list them
in the *Features* column (omit the column when none are required). A handful are standalone
Cargo packages or analysis harnesses rather than library demos — noted in their purpose.

> **crates.io note.** Examples requiring `pathmap-backend` use a git dependency and must be
> built from source (they are unavailable from a plain `crates.io` install).

### Getting started

| Example | Purpose | Features |
|---|---|---|
| [`spell_checker.rs`](../../examples/spell_checker.rs) | Minimal fuzzy spell checker: dictionary + transducer, `query` and `query_with_distance`, Standard vs Transposition | — |
| [`builder_demo.rs`](../../examples/builder_demo.rs) | The `TransducerBuilder` API for fluent transducer construction | — |
| [`batch_operations.rs`](../../examples/batch_operations.rs) | Bulk insert / contains / remove operations on a `DynamicDawg` | — |

### Dictionaries & backends

| Example | Purpose | Features |
|---|---|---|
| [`dynamic_dictionary.rs`](../../examples/dynamic_dictionary.rs) | Runtime insert/remove on a `DynamicDawg` with a live, shared transducer (incl. concurrent reads) | — |
| [`dynamic_dawg_demo.rs`](../../examples/dynamic_dawg_demo.rs) | Online modifications of a `DynamicDawg` (byte alphabet) | — |
| [`dynamic_dawg_unicode.rs`](../../examples/dynamic_dawg_unicode.rs) | `DynamicDawgChar` with full Unicode (`char`) support | — |
| [`suffix_automaton_demo.rs`](../../examples/suffix_automaton_demo.rs) | Substring matching with a `SuffixAutomaton` | — |
| [`substring_search.rs`](../../examples/substring_search.rs) | Comprehensive approximate substring search over suffix automata | — |
| [`test_backend_comparison.rs`](../../examples/test_backend_comparison.rs) | Side-by-side `contains` behavior across dictionary backends | — |
| [`custom_sync_strategy.rs`](../../examples/custom_sync_strategy.rs) | A custom backend declaring its own `SyncStrategy` for the transducer | `pathmap-backend` |

### Queries, algorithms & substitutions

| Example | Purpose | Features |
|---|---|---|
| [`ordered_query_demo.rs`](../../examples/ordered_query_demo.rs) | `query_ordered`: distance-first, lexicographic results; lazy top-$`k`$ and distance-bounded queries | — |
| [`ordered_query_benchmark.rs`](../../examples/ordered_query_benchmark.rs) | Micro-benchmark: ordered vs unordered query iterators | — |
| [`unicode_diacritics.rs`](../../examples/unicode_diacritics.rs) | `SubstitutionSetChar` presets (Latin diacritics, Greek/Cyrillic case-fold, kana) + custom sets | — |
| [`custom_substitutions.rs`](../../examples/custom_substitutions.rs) | Build and combine byte-level `SubstitutionSet`s for domain-specific matching | — |
| [`phonetic_matching.rs`](../../examples/phonetic_matching.rs) | Restricted substitutions for sound-alike matching (no rules feature needed) | — |
| [`code_completion_demo.rs`](../../examples/code_completion_demo.rs) | Code completion via prefix matching and result filtering | — |
| [`position_skip_test.rs`](../../examples/position_skip_test.rs) | Correctness check that automaton position-skipping preserves results | — |
| [`trace_za_query.rs`](../../examples/trace_za_query.rs) | Manual, step-by-step trace of automaton states for a tiny query (`"za"`) | — |

### Values, fuzzy maps & caching

| Example | Purpose | Features |
|---|---|---|
| [`fuzzy_maps_code_completion.rs`](../../examples/fuzzy_maps_code_completion.rs) | Terms→scope-ID fuzzy map; filter, prioritize, and prune matches by value | `pathmap-backend` |
| [`fuzzy_cache_basic.rs`](../../examples/fuzzy_cache_basic.rs) | Wrapping a dictionary in a cache-eviction decorator | `pathmap-backend` |
| [`mork_fuzzy_query.rs`](../../examples/mork_fuzzy_query.rs) | Zero-plumbing fuzzy queries over a bare, borrowed `PathMap` (MORK-style) | `pathmap-backend` |

### Contextual completion

| Example | Purpose | Features |
|---|---|---|
| [`contextual_completion.rs`](../../examples/contextual_completion.rs) | `DynamicContextualCompletionEngine`: drafts, checkpoints/undo, hierarchical scope visibility | `pathmap-backend` |
| [`hierarchical_scope_completion.rs`](../../examples/hierarchical_scope_completion.rs) | Lexical-scope completion built on fuzzy maps | `pathmap-backend` |
| [`advanced_contextual_filtering.rs`](../../examples/advanced_contextual_filtering.rs) | Bitmap-based node masking for fast contextual filtering | — |
| [`contextual_filtering_optimization.rs`](../../examples/contextual_filtering_optimization.rs) | Efficient contextual filtering via sub-trie construction | — |

### Phonetic matching

| Example | Purpose | Features |
|---|---|---|
| [`phonetic_spellcheck/`](../../examples/phonetic_spellcheck/README.md) | Standalone project: `PhoneticNormalizedDictionary` over ~124k words (fuzzy, regex, pattern expansion) | `phonetic-rules`, `pathmap-backend`, `embedded-rules` |
| [`phonetic_fuzzy_matching.rs`](../../examples/phonetic_fuzzy_matching.rs) | Comprehensive phonetic rewrite × Levenshtein fuzzy matching | `phonetic-rules` |
| [`phonetic_rewrite.rs`](../../examples/phonetic_rewrite.rs) | Apply `.llev` phonetic rewrite rules to transform text | `phonetic-rules` |
| [`phonetic_iteration_analysis.rs`](../../examples/phonetic_iteration_analysis.rs) | Measure iteration counts inside `apply_rules_seq()` | `phonetic-rules` |
| [`phonetic_slice_analysis.rs`](../../examples/phonetic_slice_analysis.rs) | Profile slice-copying overhead in phonetic rewriting | — |

### Performance & profiling

| Example | Purpose | Features |
|---|---|---|
| [`real_world_benchmark.rs`](../../examples/real_world_benchmark.rs) | Benchmark backends on a real English dictionary: build, `contains`, fuzzy `query` | — |
| [`profile.rs`](../../examples/profile.rs) | Representative workload for flame-graph profiling / regression hunting | — |
| [`profile_workload.rs`](../../examples/profile_workload.rs) | Mixed-operation workload for flamegraph capture | — |
| [`scientific_eval.rs`](../../examples/scientific_eval.rs) | Instrumented evaluation harness (custom allocator + `PriorityQueryIterator`) for metrics | — |
| [`simd_prototype.rs`](../../examples/simd_prototype.rs) | Prototype illustrating the SIMD acceleration concepts | — |
| [`parallel_workspace_indexing.rs`](../../examples/parallel_workspace_indexing.rs) | Parallel per-document dictionary construction with binary-tree reduction | — |
| [`msm_experiment.rs`](../../examples/msm_experiment.rs) | Deterministic Move–Split–Merge (time-series) optimization harness | — |

### Serialization

| Example | Purpose | Features |
|---|---|---|
| [`serialization.rs`](../../examples/serialization.rs) | Save/load a `DoubleArrayTrie` via bincode and JSON, then verify fuzzy queries round-trip | `serialization` |

---

[← Documentation Index](../../README.md)
