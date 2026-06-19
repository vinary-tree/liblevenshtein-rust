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
`msm-002`, `msm-003`, `msm-004`, `msm-006`, `msm-007`, and `msm-010`.

## Method

Heavy commands were run with:

```text
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 ...
```

No benchmark corpora were placed in `/tmp`. The deterministic local harness is
`examples/msm_experiment.rs`; Criterion coverage is in
`benches/msm_benchmarks.rs`.

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
| `msm-005-bin-path-storage` | partially accepted | Precomputed bin bounds retained; path/bucket interning deferred because it was not isolated as the bottleneck. | `046c5ab` |

The retained code changes are:

- `MsmConfig::distance_with_cutoff`, an exact two-row DP with safe row-min
  early abandonment for finite cutoffs.
- Exact transducer final verification now uses cutoff-aware exact DP instead of
  always allocating the full DP matrix.
- Exact range traversal reuses one interval DP column per trie depth.
- `MsmTransducer` precomputes quantization bin intervals for hot trie traversal.
- `MsmTransducer::search_knn` now uses one exact best-first branch-and-bound
  traversal instead of repeated threshold-doubling range searches.

## Rejected Or Deferred Results

| Experiment | Decision | Evidence |
| --- | --- | --- |
| `msm-006-legacy-automaton-audit` | rejected for hot path | Wavefront direct distance was about `1.33x` optimized two-row DP; legacy `MsmState` automaton was about `90x` optimized two-row DP on the 24-point direct-distance workload. |
| `msm-007-subtree-pruning` | deferred | `DynamicDawg` exposes cloned `DictionaryNode` traversal nodes but no stable mutable per-node metadata slot. External subtree metadata would need extra memory and lookup work, so it needs corpus evidence first. |
| `msm-008-approximate-msm-ann` | deferred | Exact changes produced large wins without recall risk. Approximate ANN/SAX/PAA needs an explicit opt-in API and UCR/UEA/Multiverse recall@k evidence before adding dependencies. |
| `msm-009-adaptive-msm-cost` | deferred | `adaptive-msm` already has the right crate boundary: learner over `MsmConfig` without `lling-llang`, and optional WFST above `lling-llang`. Learned-c quality belongs in corpus-labeled adaptive-msm/pgmcp evaluation. |
| academic UCR/UEA/Multiverse benchmark | deferred | No corpus was present locally; `target/` was already 11G. A future run should cache archives under ignored `target/msm-corpora/` or XDG cache, never `/tmp`. |

## Correctness Gates

The accepted changes passed:

```text
cargo test time_series::msm::tests::test_cutoff
cargo test --test msm_transducer_tests
cargo test --test time_series_msm_tests
git diff --check
```

All Rust commands above were run under the `systemd-run` memory cap. The
transducer tests include brute-force exact range checks, brute-force kNN checks,
out-of-range quantization bins, quantization collisions, and concurrent
read-only queries.

## Dependent Audit

The pgmcp dependent audit found no code change to retain in this session:

- `pgmcp/src/fuzzy/time_series.rs` already uses `search_with_lb_parallel`.
- `pgmcp/src/fuzzy/trajectory_index.rs` already combines adaptive MSM cost
  calibration with exact lower-bound-pruned MSM retrieval.
- `adaptive-msm` already breaks the `liblevenshtein -> lling-llang ->
  liblevenshtein` cycle by living above both crates.

Further dependent work should be benchmarked in each dependent repository with
write permissions and its own memory-capped experiment protocol.
