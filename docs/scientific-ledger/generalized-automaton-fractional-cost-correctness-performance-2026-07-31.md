---
title: Generalized Automaton Fractional-Cost Boundary Performance
date: 2026-07-31
pgmcp_experiment: generalized-automaton-fractional-cost-correctness-performance
pgmcp_root: extending-liblevenshtein-automaton-families-4bb97598
scope: Phase 2 prime fractional-cost boundary
---

# Generalized Automaton Fractional-Cost Boundary Performance

This ledger records the preregistered boundary benchmark for the repaired
fractional-cost semantics. Its primary input applies six substitutions of cost
`0.15` under a budget of 1. Exact scaled arithmetic computes cost `0.90`, so the
word is accepted. A companion example applies seven substitutions, computes
`1.05`, and is rejected. The old truncating accumulator incorrectly treated
both paths as free.

## Hypothesis and method

The experiment was registered in pgmcp as
`generalized-automaton-fractional-cost-correctness-performance` before the new
benchmark was measured. The primary scenario is
`fractional_cost/boundary/six_accept`.

| Field | Preregistered value |
| --- | --- |
| Control | Archived pre-repair library, using the same new benchmark harness |
| Treatment | Exact `CostScale` implementation |
| Expected direction | Treatment slower |
| Primary test | One-sided Welch greater, `alpha = 0.05`, Cohen's `d >= 0.5` |
| Practical envelope | Equivalence within `-200%..+200%`, or less than three-times latency |
| Samples | 60 per arm, 3 s warm-up, 8 s measurement |
| Isolation | CPU 0 pinned, performance governor, separate target directories |

The wider envelope reflects the fact that the control is semantically invalid:
its work is artificially reduced by zeroing every fractional step. The envelope
is a resource guard, not permission to restore truncation.

## Result

| Arm | Mean (ns) | Median (ns) | Standard deviation (ns) | Samples |
| --- | ---: | ---: | ---: | ---: |
| Truncating control | 510.196 | 508.325 | 5.188 | 60 |
| Exact scaled treatment | 381.797 | 383.002 | 4.626 | 60 |

The exact implementation was **25.2% faster**. Because this is opposite the
preregistered expected direction, pgmcp rejected the slowdown hypothesis.

| Statistic | pgmcp result |
| --- | ---: |
| Welch statistic | -143.0906 |
| One-sided greater `p` | 1.0 |
| Cohen's `d` | -26.1246 |
| Mean-difference 95% interval (ns) | [-130.1765, -126.6221] |
| Equivalence-envelope `p` | 0.0, passed |
| Mann-Whitney `p` | 0.0 |
| Cliff's delta | -1.0 |

The benchmark establishes that the exact boundary does not impose the expected
runtime penalty. It does not by itself prove correctness; correctness follows
from the scaled-integer representation, the six-versus-seven example tests,
randomized refinement tests, and the associated Rocq, Verus, Z3, and cvc5
obligations. The speedup is likely caused by removal of spurious zero-cost
paths, but that causal explanation remains an inference.

## Executable invariant

For decimal weights exactly representable by the selected scale, the executable
boundary property is:

```math
\operatorname{within}(n,w,k)
\quad\Longleftrightarrow\quad
n\,\operatorname{scaled}(w) \le kS.
```

The implementation tests this invariant at the discontinuity: `n = 6` is
within budget and `n = 7` is not. Property tests generalize it across generated
operation sets and operation orderings. The formal models prove exact rescaling,
monotone accumulation, and the absence of an integer-denominator shortcut in
the subsumption certificate.

## Decision

| ID | Verdict | Justification |
| --- | --- | --- |
| `P2P-FRACTIONAL-SEMANTICS` | retained | The treatment restores the configured operation costs exactly. |
| `P2P-FRACTIONAL-PERFORMANCE` | accepted within resource gate | The treatment is faster and well within the preregistered three-times envelope. |
| `P2P-TRUNCATION` | permanently rejected | Truncation violates the distance algebra and makes positive-cost operations free. |

## Reproduction

```console
CARGO_NET_OFFLINE=true RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  taskset -c 0 cargo bench --bench generalized_automaton_benchmarks -- \
  fractional_cost/boundary/six_accept --warm-up-time 3 \
  --measurement-time 8 --sample-size 60 --noplot
```

The complete samples and frozen decision are retained in pgmcp experiment 156.
Transient benchmark logs and isolated target directories are intentionally not
version-controlled.
