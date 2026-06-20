---
title: MSM Automata Scientific Evaluation
date: 2026-06-19
pgmcp_root: msm-automata-scientific-evaluation
scope: liblevenshtein-rust time_series MSM automata and dependent integration audit
---

# MSM Automata Scientific Evaluation

This ledger records the 2026-06-19 scientific evaluation of the
Move-Split-Merge (MSM) time-series automata in `liblevenshtein-rust`. MSM is the
metric from Stefan et al., "The Move-Split-Merge Metric for Time Series"; this
work also used the UCR and UEA archive papers as the benchmark plan references:

- UCR Time Series Archive: https://arxiv.org/abs/1810.07758
- UEA multivariate archive: https://arxiv.org/abs/1811.00075
- Multiverse multivariate archive: https://arxiv.org/abs/2603.20352

The evaluation was tracked in pgmcp under
`msm-automata-scientific-evaluation`, with protocol-backed experiments
`msm-002`, `msm-003`, `msm-004`, `msm-006`, `msm-007`, `msm-008`,
`msm-009`, and `msm-010`.

## Method

Heavy commands were run with:

```text
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 ...
```

No benchmark corpora were placed in `/tmp`. The deterministic local harness is
`examples/msm_experiment.rs`; Criterion coverage is in
`benches/msm_benchmarks.rs`. The harness accepts UCR `.txt` splits and
UEA/tslearn-style `.ts` splits under a caller-supplied dataset directory, so
large public corpora can be cached under ignored `target/msm-corpora/` or an XDG
cache instead of being committed or staged in tmpfs.

For the accepted timing experiments, each arm used 51 measured samples after
warm-up. The deterministic synthetic exact-transducer workload indexes 512
prefix-sharing length-48 series, then queries exact range or exact kNN with a
stable checksum.

## Retained Results

| Experiment | Decision | Main Result | Retained Commit |
| --- | --- | --- | --- |
| `msm-002-exact-verification-dp` | accepted | Exact range mean latency improved from `3.510 ms` to `3.077 ms`; pgmcp `p=7.51e-50`. | `046c5ab` |
| `msm-003-interval-column-reuse` | accepted | Same exact range treatment accepted; reusable interval columns and precomputed bin intervals retained. | `046c5ab` |
| `msm-004-best-first-knn` | accepted | Exact kNN mean latency improved from `7.034 ms` to `0.980 ms`; pgmcp `p=3.38e-80`. | `046c5ab` |
| `msm-005-bin-path-storage` | partially accepted | Precomputed bin bounds retained; path/bucket interning was not retained because it was not isolated as a bottleneck. | `046c5ab` |
| `msm-008-approximate-msm-ann` | accepted as an opt-in approximate API | `ApproxMsmIndex` uses PAA feature ranking plus exact MSM reranking; deterministic harness coverage now checks a recall floor without changing exact `MsmTransducer` semantics. | this commit |
| academic UCR/UEA harness | accepted as adapter coverage | `examples/msm_experiment.rs` loads UCR `.txt` and UEA-style `.ts` train/test splits and reports MSM 1-NN latency, accuracy, and per-case outcomes; repo-local tests cover both parsers and deterministic 1-NN outcomes. | this commit |

The retained code changes are:

- `MsmConfig::distance_with_cutoff`, an exact two-row DP with safe row-min
  early abandonment for finite cutoffs.
- Exact transducer final verification now uses cutoff-aware exact DP instead of
  always allocating the full DP matrix.
- Exact range traversal reuses one interval DP column per trie depth.
- `MsmTransducer` precomputes quantization bin intervals for hot trie traversal.
- `MsmTransducer::search_knn` now uses one exact best-first branch-and-bound
  traversal instead of repeated threshold-doubling range searches.
- `ApproxMsmIndex` is retained as an explicit approximate index, separate from
  exact retrieval, with PAA candidate generation and exact MSM reranking.
- The MSM experiment harness supports UCR `.txt` and UEA-style `.ts` split
  formats for standard time-series benchmark evaluation without committing
  large corpora to the repository.

## Non-Retained Or Boundary Decisions

| Experiment | Decision | Evidence |
| --- | --- | --- |
| `msm-006-legacy-automaton-audit` | rejected for hot path | Wavefront direct distance was about `1.33x` optimized two-row DP; legacy `MsmState` automaton was about `90x` optimized two-row DP on the 24-point direct-distance workload. |
| `msm-007-subtree-pruning` | not retained | `DynamicDawg` exposes cloned `DictionaryNode` traversal nodes but no stable mutable per-node metadata slot. External subtree metadata would add memory and lookup work without evidence that it improves the current exact traversal. |
| `msm-009-adaptive-msm-cost` | boundary accepted, no current-repo code change | `adaptive-msm` already has the right crate boundary: a learner over `MsmConfig` without forcing `liblevenshtein` to depend on `lling-llang`. `liblevenshtein-rust` keeps the metric/config surface; learned-cost quality belongs in the adaptive crate's labeled-corpus evaluation. |

## Correctness Gates

The accepted changes passed:

```text
cargo test time_series::msm::tests::test_cutoff
cargo test --test msm_transducer_tests
cargo test --test time_series_msm_tests
cargo test --example msm_experiment
git diff --check
```

All Rust commands above were run under the `systemd-run` memory cap. The
transducer tests include brute-force exact range checks, brute-force kNN checks,
out-of-range quantization bins, quantization collisions, and concurrent
read-only queries. The example-harness tests cover the opt-in approximate PAA
recall gate plus UCR `.txt` and UEA-style `.ts` split parsing with deterministic
1-NN outcomes.

## Dependent Audit

The pgmcp dependent audit found no code change to retain in this session:

- `pgmcp/src/fuzzy/time_series.rs` already uses `search_with_lb_parallel`.
- `pgmcp/src/fuzzy/trajectory_index.rs` already combines adaptive MSM cost
  calibration with exact lower-bound-pruned MSM retrieval.
- `adaptive-msm` already breaks the `liblevenshtein -> lling-llang ->
  liblevenshtein` cycle by living above both crates.

Further dependent work should be benchmarked in each dependent repository with
write permissions and its own memory-capped experiment protocol.
