# Phase 12: prefix-shared fzf scoring

Status: pre-registered on 2026-08-02, before the benchmark in
`duallity/benches/fzf_trie_vs_flat.rs` was executed.

## Question and measurement boundary

The implementation shares one incremental `FuzzyMatchV2` dynamic-programming
column for every visited dictionary edge. The control computes one independent
matrix per dictionary term. The deterministic corpus contains 1,000 and 10,000
Rust-like paths with deliberately shared directory and module prefixes.

The primary work measure is the number of dynamic-programming columns. Timing
is secondary because it includes dictionary traversal, allocation, and host
noise. The implementation must first pass score-for-score differential tests
and top-$`k`$ set equality; a faster wrong scorer is rejected.

## Pre-registered hypotheses

| ID | Hypothesis | Decision rule |
|---|---|---|
| FZF-H1 | Prefix sharing reduces constructed DP columns. | Accept only if trie columns are at most 70% of the sum of candidate character counts at 10,000 terms. |
| FZF-H2 | Prefix sharing improves wall-clock throughput. | Accept only if the median trie time is at least 10% lower than flat scoring at 10,000 terms. Otherwise retain the semantically useful WFST surface but make no speed claim. |
| FZF-H3 | Claude's active-only upper bound is admissible. | Reject if any descendant can start a later alignment whose score exceeds that formula. The committed regression is a decisive counterexample. |
| FZF-H4 | Reachable-character summaries justify their memory and maintenance cost. | Implement only after a separate benchmark predicts at least 20% additional subtree rejection on the representative path corpus. The current correct unstarted-alignment term is global without subtree metadata, so no pruning claim is pre-registered for the baseline. |

## Correctness gate

Before timing, all of the following must pass: upstream score fixtures,
example tests, generated descendant-bound tests, generated trie/flat top-$`k`$
equality, integration through the liblevenshtein DFS visitor, Arctic transition
telescoping, and the Rocq, Verus, Dafny, SMT, and TLA+ artifacts registered in
the formal-verification manifest.

## Results

Measured on 2026-08-02 with:

```text
cargo bench --bench fzf_trie_vs_flat -- --quick
```

The host was Linux 7.1.5 on an AMD Ryzen Threadripper PRO 5975WX with 32
online cores. The toolchain was `rustc 1.97.1` and `cargo 1.97.1`. Criterion's
`--quick` mode was used for this gate; timing should therefore be read as a
large-effect diagnostic, not a publication-quality microbenchmark.

| Terms | Method | DP columns | Median time | 95% interval reported by Criterion |
|---:|---|---:|---:|---:|
| 1,000 | independent | 56,000 | 1.2920 ms | 1.2837--1.3250 ms |
| 1,000 | prefix-shared | 11,648 | 12.211 ms | 12.159--12.416 ms |
| 10,000 | independent | 560,000 | 12.987 ms | 12.983--13.001 ms |
| 10,000 | prefix-shared | 67,952 | 604.16 ms | 599.55--605.32 ms |

At 10,000 terms, prefix sharing constructs 12.13% as many DP columns, an
87.87% reduction. It nevertheless takes 46.52 times as long in this end-to-end
implementation. Both corpus sizes visited every accepted candidate, and the
sound local-alignment upper bound pruned zero prefixes.

## Verdicts

| ID | Verdict | Consequence |
|---|---|---|
| FZF-H1 | accepted | Prefix sharing has a measured work-count benefit; the 12.13% result is below the 70% gate. |
| FZF-H2 | rejected | Do not claim a runtime speedup. Retain the scorer and WFST for their semantic and compositional value. |
| FZF-H3 | rejected | The active-only formula is unsound; retain the unstarted-alignment alternative and its regression/property/formal checks. |
| FZF-H4 | gate not met | Do not implement reachable-character summaries. No separate evidence predicts the required 20% additional rejection, and the baseline observed zero pruned prefixes. |

The implementation ships no assertion that fewer DP columns imply lower wall
time. A future optimization must pre-register and isolate traversal,
path-materialization, and DP-allocation costs before revisiting FZF-H2.
