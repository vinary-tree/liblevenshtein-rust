# Metric automata domain and capability contract

This is the source-mapped domain inventory for pgmcp task `cbc-01-domain`.
It applies to the current worktree; the [baseline audit](CBC_BASELINE_EVIDENCE.md)
distinguishes baseline evidence from new campaign files. A *metric domain*
names admitted values and their equality. A metric theorem about ideal costs
does not by itself establish the same laws for binary64 results or prove that
an indexed Rust search implements that distance.

## Exported metric claims

| Family and source admission | Mathematical domain and equality | Ideal-arithmetic statement and evidence boundary |
|---|---|---|
| Unit Levenshtein over ordinary finite strings; [ORC worked derivation](../theory/ordered-residual-calculus.md#111-standard-levenshtein) | Character sequences with ordinary sequence equality and unit insertion, deletion, and substitution costs | The minimum unit edit-script cost is a metric. The chosen automaton still needs its own residual, result, and source correspondence proof. |
| Weighted edits; [weighted cost algebra](../../src/cost/weighted.rs) and [ORC metric qualification](../theory/ordered-residual-calculus.md#121-script-metrics-and-alignment-restrictions) | A fixed alphabet and fixed lawful edit operation costs. Equality and zero-cost script behavior must be specified for the particular configuration. | Nonnegative weights alone do not grant a metric. Symmetry, separation, and triangle composition are additional obligations; the binary64 cost monoid is not an associative real semiring. |
| Scalar ERP; [raw recurrence](../../src/time_series/kernels/erp.rs), [quotient constructor](../../src/time_series/metric_domains.rs#L38), and [metric configuration](../../src/time_series/metric_domains.rs#L224) | Finite scalar sequences, including empty, modulo insertion or deletion of the one fixed finite gap value. `ErpQuotientSeries` removes every gap sample before scoring and indexing. | Raw ERP is a pseudometric; the induced ideal-real distance on the fixed-gap quotient is a metric. [ErpProperties.v](erp/theories/Metric/ErpProperties.v) covers local quotient and bound lemmas; binary64 and indexed-search correspondence remain separate. |
| Vector ERP; [fixed-channel ground metric](../../src/time_series/vector.rs#L258), [fixed vector gap and canonical series](../../src/time_series/vector.rs#L844) | Finite typed vector sequences, including empty, on one immutable ordered channel/unit layout, fixed positive weights/scales, and one fixed gap vector; equality is after removal of gap samples. | A metric ground cost induces the ideal ERP quotient metric. The [Verus vector kernel](verus/vector_kernels.rs) checks selected ground and interval laws; the sequence recurrence and rounded-score transfer require their own instance proof. |
| Scalar discrete Fréchet; [raw recurrence](../../src/time_series/kernels/frechet.rs), [stutter class](../../src/time_series/metric_domains.rs#L356), [quotient index](../../src/time_series/metric_domains.rs#L273) | Nonempty finite scalar paths modulo consecutive equal-sample stutters. The constructor collapses each maximal run. | Raw path distance is a pseudometric; the ideal bottleneck distance on stutter classes is a metric. [FrechetProperties.v](frechet/theories/Metric/FrechetProperties.v) proves selected recurrence and triangle lemmas; machine and index correspondence remain separate. |
| Vector discrete Fréchet; [stutter path](../../src/time_series/vector.rs#L1778) and [metric constructor](../../src/time_series/vector.rs#L1851) | Nonempty paths of fixed positive dimension, modulo consecutive identical *whole points*, under one sealed `GroundMetric`. Channel coordinates are never flattened into time. `VectorFrechetPath` retains dimension but no channel-layout identity, so a fixed-channel interpretation also requires a stable caller mapping. | If the chosen ground cost is a metric and the sequence recurrence is the discrete Fréchet recurrence, the ideal quotient path distance is a metric. Each ground implementation, layout binding, and binary64 computation retain their own proof obligations. |
| Fixed positive channel point metric; [channel constructor](../../src/time_series/vector.rs#L138) and [fixed layout/scales](../../src/time_series/vector.rs#L258) | Finite equal-dimensional vectors under one nonempty ordered channel/unit schema, fixed positive finite scale and weight per coordinate, and immutable fold-local provenance. Missing coordinates and pair-dependent renormalization are outside this domain. | A positive weighted sum of scaled coordinate absolute differences is an ideal-real metric. [Verus vector kernel](verus/vector_kernels.rs) checks the typed-channel lifting and a missing-channel control; rounded-score laws and every source primitive still require correspondence. |
| Built-in `L1GroundMetric`, `L2GroundMetric`, and `LinfGroundMetric`; [sealed implementations](../../src/time_series/vector.rs#L545) | Finite vector samples of the same positive dimension, with ordinary coordinate equality. The public `GroundMetric::distance` method relies on this domain precondition; `VectorFrechetMetric::distance_bounded` checks dimension before calling it. | The ideal coordinate norms are metrics at fixed dimension. The actual implementations use left-to-right binary64 accumulation, `hypot`, and maximum respectively; no one ideal proof certifies all three operation graphs. |
| Scalar MSM; [metric configuration](../../src/time_series/msm.rs#L569), [metric kernel](../../src/time_series/msm_kernel.rs#L461) | Nonempty finite scalar sequences with fixed finite split/merge cost strictly above zero. The raw zero-cost configuration is outside the metric-qualified type. | The standard ideal MSM recurrence is metric on this domain under its source theorem. The [manifest](FORMAL_VERIFICATION_MANIFEST.tsv) still marks the broader MSM proof tree partial; its source and binary64 closure cannot be inferred from the marker. |
| Unit-grid TWED; [validated configuration](../../src/time_series/kernels/twed.rs#L241) and [metric kernel marker](../../src/time_series/kernels/twed.rs#L825) | Finite scalar sequences on one common unit grid with fixed synthetic zero predecessor, finite positive stiffness, and finite nonnegative gap penalty. | The ideal recurrence has the qualified metric law under its source hypotheses. [TwedProperties.v](twed/theories/Metric/TwedProperties.v) proves recurrence and interval islands; the whole executable and rounded-score law remain separate. |
| Scalar physical-time TWED; [raw constructor](../../src/time_series/timestamped_twed.rs#L160), [strict-origin wrapper](../../src/time_series/timestamped_twed.rs#L279), [metric configuration](../../src/time_series/timestamped_twed.rs#L333) | Nonempty finite values; finite strictly increasing timestamps; one fixed unit and origin; first timestamp **strictly later** than origin; positive stiffness and nonnegative gap penalty. | [TimestampedMetricTransfer.v](twed/theories/Metric/TimestampedMetricTransfer.v) proves a conditional strict-origin transfer. [TwedSourceMetric.v](twed/theories/Metric/TwedSourceMetric.v) proves only the sample/time product metric. The full sequence-level source theorem and binary64 transfer remain open. |
| Vector physical-time TWED; [metric and series constructor](../../src/time_series/vector.rs#L1155) | Nonempty typed vector series on one fixed positive channel metric, sentinel, unit, origin, and strictly increasing physical timestamps with the first **strictly later** than origin; positive stiffness and nonnegative gap penalty. | The ideal metric claim needs the same strict-origin source theorem, lifted through the fixed ground metric. The previous `try_series` admitted an origin-equal first sample; the current worktree rejects it with `TimestampNotAfterOrigin`, checked by a [regression](../../tests/proptest_vector_kernels.rs#L168). The typed/rounded source correspondence remains open. |

The generic [metric kernel marker](../../src/time_series/elastic/mod.rs#L425)
and [audited-index marker](../../src/time_series/metric_domains.rs#L91)
are admission gates, not proof objects. They expose only the listed reviewed
implementations; neither establishes that a runtime score obeys exact real
metric axioms. No triangle-dependent pruning rule may be invoked from a
marker alone without the concrete numeric and source obligations.

## Explicit nonmetric controls

| Control | Source and failure of a broad metric claim |
|---|---|
| Raw scalar/vector ERP | Inserting the configured gap value or vector costs zero, so distinct raw sequences can have zero distance. Only the fixed-gap canonical quotient has separation. |
| Raw scalar/vector Fréchet | Repeating a point consecutively costs zero. Only nonempty stutter classes have the stated quotient identity. |
| Origin-equal physical TWED | With zero gap penalty, inserting a sentinel-valued sample at the physical origin can cost zero. The scalar [strict wrapper](../../src/time_series/timestamped_twed.rs#L279) and vector `try_series` guard exclude that member. The [transfer proof](twed/theories/Metric/TimestampedMetricTransfer.v) checks both the scalar 1-by-2 recurrence and `generic_origin_equal_cumulative_recurrence_is_zero` for any nonnegative reflexive point cost, including a vector ground metric. |
| Arbitrary weighted edits | Asymmetric, zero-separating, or triangle-violating operation costs break metricity. The generic weighted cost monoid does not validate those semantic conditions. |
| Banded/general DTW | [DtwConfig](../../src/time_series/kernels/dtw.rs#L182) and [DtwTransducer](../../src/time_series/kernels/dtw.rs#L510) are exact nonmetric search capabilities. A concrete triangle counterexample appears in [NotAMetric.v](dtw/theories/NotAMetric.v); they do not implement `MetricElasticKernel`. |
| Soft-DTW | [SoftDtwConfig](../../src/time_series/kernels/soft_dtw.rs#L27) is a soft alignment loss, potentially negative, with path multiplicity. It is analysis-only and is not a metric or an exact min-cost trie kernel. |
| Vector MSM | [VectorMsmSupportDecision](../../src/time_series/vector.rs#L1760) is explicitly unsupported pending a canonical vector betweenness and metric/interval proof. |

For ERP, the scalar index binds the gap by exact bits while
`MetricErpConfig::distance` compares finite gaps by numeric equality. This
means signed zeros are accepted by the direct distance but distinguished by
the index scope check. Both represent the same mathematical zero gap; the
difference is an API scope choice that must be specified before a common
source-refinement theorem is stated. It does not license an equality or cache
key substitution without that theorem.

The vector TWED admission repair adds one equality comparison during series
construction. It does not change the recurrence, index traversal, or retained
search state. Its effect is to reject an input outside the claimed metric
domain before that input can enter a metric-labelled operation.

## Admission rule for later proofs

A concrete metric or pruning certificate must name the constructor path, the
domain identity (ordinary sequence or a particular quotient), all fixed
parameters and provenance, the ideal measure theorem, the actual binary64
operation graph, and the exported operation's observation profile. A theorem
for a raw pseudometric, a different gap, a different unit/origin, or a
different channel layout cannot fill that certificate's metric premise.
