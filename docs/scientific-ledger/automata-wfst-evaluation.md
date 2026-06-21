# Automata and WFST Scientific Evaluation Ledger

This ledger is the repository-local index for the pgmcp experiment records under
root work item `lev-phonetic-wfst-scientific-evaluation`.

The structured pgmcp records are the source of truth for raw samples,
pre-registered criteria, statistical decisions, and linked work items. This file
summarizes the accepted engineering decisions so reviewers can connect the code,
rules, and verification gates to the empirical evidence without querying pgmcp.

## Terms

- **Native automata**: liblevenshtein query iterators and transducer traversal
  paths implemented directly in this crate.
- **Phonetic automata**: rule-driven normalizers, LLev/LLRE loaders, phonetic
  NFAs, and dictionary-product traversal used for phonetic lookup.
- **WFST**: weighted finite-state transducer. In the current architecture,
  query-side labels are inputs and dictionary-side labels are outputs for
  edit-distance correction transducers.
- **Lattice graph engine**: the crate layer that owns graph construction,
  traversal, Viterbi or n-best extraction, composition, and lazy/eager
  expansion semantics for WFST-like graphs.
- **Conformance gate**: a deterministic experiment with an absolute threshold,
  normally `correctness_pass_rate >= 1.0`, used when a paired statistical test
  is not appropriate because the evidence is a committed regression suite.

## Evaluation Rules

1. A change is accepted only when the pgmcp experiment status is `decided` and
   its verdict supports the treatment or architectural boundary.
2. Runtime and space decisions require both the primary metric and a correctness
   or candidate-preservation gate.
3. Deterministic conformance cases use absolute-threshold criteria instead of
   Welch tests when all samples have zero variance.
4. Heavy commands are run through `systemd-run --user --scope` with
   `MemoryMax` and `MemorySwapMax=0`.
5. External benchmark validity is represented by the BENCH-005 gate, which
   includes Birkbeck/Fawthrop spelling correction, CMUdict phonetic homophones,
   Pizza&Chili-style text throughput/RSS, and OpenSLR/LibriSpeech-style WFST
   lexicon fixtures.

## Native Levenshtein Automata

| ID | pgmcp experiment | Decision | Time evidence | Space evidence | Correctness evidence |
|----|------------------|----------|---------------|----------------|----------------------|
| LEV-001 | `lev-001-query-arena` | Accept arena-backed reusable query state. | p95 latency improved, Welch p < 1e-6, Cohen's d about -3.20 for the preferred arena treatment. | Allocation count improved in the same decision family. | Candidate ordering preserved by the pre-measurement correctness suite. |
| LEV-002 | `lev-002-parent-path-arena-indices` | Keep the arena-index implementation for latency and allocation-count benefit, but do not claim byte reduction on unordered queries. | Mean latency improved from 135.4 us to 82.1 us on `lev_unordered_1k_d2`. | Allocation count improved from 2972 to 643; allocated bytes increased from 188,836 to 189,530, so the pre-registered byte-reduction hypothesis was inconclusive. | Result count stayed 124 in both arms. |
| LEV-007 | `lev-007-ordered-query-arena` | Accept ordered top-k arena/index parent paths. | Birkbeck/Fawthrop ordered p95 latency improved, Welch p < 1e-6, Cohen's d about -2.16. | Allocation count improved; allocated bytes increased because of arena vector storage. | Recall@5 stayed 50/51 and ordering semantics were preserved. |
| LEV-008 | `lev-008-lazy-path-arena-capacity-tuning` | Do not adopt lazy path-arena initialization. | Mean elapsed time increased from 79.6 us to 81.8 us. | Allocated bytes increased from 189,530 to 190,010 and allocation count increased from 643 to 647. | Result count stayed 124, so the treatment was correct but not beneficial. |
| LEV-005 | `lev-005-priority-query-comparator-cache` | Accept cached priority-query term keys. | Elapsed time improved, Welch p < 1e-6, Cohen's d about -2.41. | Allocation count fell from about 3647 to 1505; allocated bytes fell from about 99,953 to 90,105. | `transducer::priority_query` tests passed and every sample produced 25 results. |
| LEV-006 | `lev-006-transition-index-of-match-slice-scan` | Accept bounded slice `.position()` transition scan. | `lev_unordered_1k_d2` elapsed time improved, Welch p = 0.000057, Cohen's d about -0.80. | No allocation increase was recorded for the local transition treatment. | `transducer::transition` tests passed and every sample produced 124 results. |
| LEV-010 | `lev_010_myers_transposition_hybrid_long_case` | Keep the exact OSA hybrid for `myers_transposition_distance`. | Length-12+ cases improved in the Criterion comparison. | Stack rows are used for short/mid strings; heap rows remain for larger inputs. | Exact char-level transposition semantics are preserved by the shared DP cross-checks. |

## Phonetic Automata and LLev Rules

| ID | pgmcp experiment | Decision | Time evidence | Space evidence | Correctness evidence |
|----|------------------|----------|---------------|----------------|----------------------|
| PHON-001/002 | `phon-001-002-trie-product-phonetic-regex` | Accept trie-product traversal for standard-Levenshtein phonetic regex products. | Elapsed time improved, Welch p < 1e-6, Cohen's d about -19.13. | Allocated bytes fell from about 45.3 MB to 0.42 MB; allocation count also dropped sharply. | Result-count sequences matched for all 51 paired samples and focused dictionary tests passed. |
| PHON-008 | `phon-008-llev-zompist-recall-parity-root-cause` | Treat the original CMUdict smoke parity as a rule/oracle/ranking issue, not an unresolved automata traversal bug. | Current diagnostic separated top-k/ranking misses from coverage misses. | Full CMUdict extension did not introduce a measured time/space regression in the accepted root-cause evidence. | First 2048 CMUdict cases had 3960/3960 expected terms in full results; all 195 top-k misses were classified as `top_k_ceiling` or `ambiguous_query_pronunciation_ranking`; zero normalized-index/query bugs were found. |
| PHON-009 | `phon-009-llev-extension-order` | Keep LLev English extensions before primary rules where whole-word extension coverage is required. | Targeted benchmark accepted the extension-order treatment. | No adverse space result was recorded for the targeted rule-order treatment. | Explicit homophone/name cases improved from 36/51 to 51/51 when extensions were prepended. |
| PHON-010 | `phon-010-llev-apply-multisymbol-output-conformance-gate` | Accept multi-symbol output expansion in `RuleSetChar::apply` and `apply_full`. | Deterministic gate; runtime not the primary metric. | Deterministic gate; space not the primary metric. | Capped conformance test passed with `correctness_pass_rate = 1.0` under `MemoryMax=1G`. |
| PHON-011 | `phon-011-llre-import-composite-llev-symbols-conformance-gate` | Accept composite LLev import preservation in the LLRE loader. | Deterministic gate; runtime not the primary metric. | Deterministic gate; space not the primary metric. | Loader conformance passed with `correctness_pass_rate = 1.0`, covering composite imports, aliased references, and range preservation. |
| PHON-012 | `phon-012-llev-compound-context-integration-conformance-gate` | Accept integration coverage for compound contexts such as `x -> gz / [aeiou]_[aeiou]` before fallback `x -> ks`. | Deterministic gate; runtime not the primary metric. | Deterministic gate; space not the primary metric. | LLev integration subset passed 62/62 under `MemoryMax=1G`. |
| PHON-013 | `phon-013-utf8-multichar-substitution-runtime-gate` | Accept UTF-8 multi-character substitution support. | UTF-8 `contains_str` averaged about 14.04 ns/op, below the 25 ns/op threshold. | The ASCII hot-table path remains separate from the generalized UTF-8 path. | Conformance gate accepted UTF-8 generalized substitution behavior. |

## Academic Benchmark Gate

| ID | pgmcp experiment | Decision | Evidence |
|----|------------------|----------|----------|
| BENCH-005 | `bench-005-academic-corpus-gate` | Require external-validity checks for accepted automata/WFST treatments. | Accepted with `multi_suite_regression_count` improvement, Welch p < 1e-6. The gate covers Birkbeck/Fawthrop, Mitton-style spelling corpora, CMUdict homophones, text throughput/RSS, and OpenSLR/LibriSpeech-style WFST lexicon fixtures. |
| MSM-011 | `msm-011-ucr-archive-exact-1nn-academic-benchmark` | Accept the official UCR/aeon univariate archive slice as paired benchmark evidence for exact MSM 1-NN. | On 51 datasets bounded by `train * test * length^2 <= 1e9`, exact MSM 1-NN reached `11653/13754 = 0.847244` accuracy versus majority baseline `5664/13754 = 0.411807`. pgmcp's paired-binary endpoint computed McNemar evidence from `control_only=415`, `treatment_only=6404`, `n_discordant=6819`, `p=0.0`; the previous 139-sample bucket run is invalidated. |

## WFST and Cross-Crate Boundary

| ID | pgmcp experiment | Decision | Evidence |
|----|------------------|----------|----------|
| DUAL-003 | `dual-003-wfst-label-semantics` | Accept duallity label-semantics fix for standard Levenshtein WFSTs. | Semantic mismatches fell from 45/51 to 0/51, Welch p < 1e-6, Cohen's d about -3.83. |
| DUAL-008 | `dual-008-rewrite-wfst-multisymbol-output-conformance-gate` | Accept complete char/epsilon `RewriteWfst` chains for multi-symbol input and output rewrites. | Feature-enabled duallity suite passed under `MemoryMax=3G` with `correctness_pass_rate = 1.0`. |
| DUAL-009 | `dual-009-phonetic-nfa-statesource-conformance-gate` | Accept `PhoneticNfaWfst` `StateSource` computed-state semantics. | Start-state and LazyWfst-equivalence tests passed under `MemoryMax=3G` with `correctness_pass_rate = 1.0`. |
| DUAL-010 | `dual-010-phonetic-wfst-boundary-semantics` and `dual-010-phonetic-wfst-boundary-conformance-gate` | Keep dictionary-integrated `PhoneticWfst` as a dictionary-side scorer/acceptor boundary; use `RewriteWfst` or `PhoneticNfaWfst` plus Levenshtein WFST composition for query-to-dictionary transduction. | The boundary decision is supported by accepted conformance gates for label semantics, rewrite chains, and NFA state-source expansion. |
| WFST-004 | `wfst-004-lattice-wfst-ownership-boundary` | Use `lling-llang` as the lattice graph engine and WFST algorithm crate; use `duallity` as the liblevenshtein adapter; keep `llattice` as algebraic lattice traits; keep `libgrammstein` as language-model/weight infrastructure. | Observational boundary gate accepted 5/5 cross-crate checks, including selected lling-llang lattice/WFST tests, llattice tests, duallity tests, and libgrammstein checks. |

## Current Operational Gates

Run expensive gates under memory caps and serialized builds. The exact command
varies by target; the following are the canonical shapes used by the accepted
experiments:

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env CARGO_BUILD_JOBS=1 cargo test -j1 --all-features -- --test-threads=1
```

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env CARGO_BUILD_JOBS=1 cargo bench -j1 --features rand --bench substitution_set_microbench
```

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  make -C docs/verification/grammar -j1
```

The full core Rocq suite is a proof-compilation memory hotspot: capped runs at
2 GiB, 4 GiB, and 8 GiB reached progressively later proof files and were killed
inside the corresponding `systemd-run` unit. That result is recorded in
`docs/verification/SUMMARY.md` and is separate from runtime automata
performance.
