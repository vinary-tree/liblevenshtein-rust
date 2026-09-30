# Metric automata domain and capability contract

This is the source-mapped domain inventory for pgmcp task `cbc-01-domain`.
It applies to the current source revision; the [baseline audit](CBC_BASELINE_EVIDENCE.md)
distinguishes baseline evidence from new campaign files. A *metric domain*
names admitted values and their equality. A metric theorem about ideal costs
does not by itself establish the same laws for binary64 results or prove that
an indexed Rust search implements that distance.

## Exported metric claims

| Family and source admission | Mathematical domain and equality | Ideal-arithmetic statement and evidence boundary |
|---|---|---|
| Unit Levenshtein over ordinary finite strings; [selector and metric classification](../../src/transducer/algorithm.rs#L16), [unit transition](../../src/transducer/variants/standard.rs), and [ORC worked derivation](../theory/ordered-residual-calculus.md#111-standard-levenshtein) | Finite character, byte, or unit sequences under one alphabet and ordinary sequence equality, with unit insertion, deletion, and unrestricted substitution in both directions; full-string distance rather than substring observation. | The minimum unit edit-script cost is a metric. The chosen automaton still needs its own residual, result, and source correspondence proof. |
| Unit symmetric MergeAndSplit; [selector](../../src/transducer/algorithm.rs#L44), [variant](../../src/transducer/variants/merge_split.rs), and [transition](../../src/transducer/transition.rs) | Finite strings under ordinary sequence equality, with unit insert/delete/substitute and symmetric unit merge/split operations; fixed operation repertoire and unrestricted edit-script interpretation. | `Algorithm::is_metric` declares this metric. The ideal shortest-script argument needs recurrence-to-script equivalence and the source transition/subsumption proof before metric-tree use; a marker alone is not that proof. |
| Unit unrestricted Damerau–Levenshtein; [selector and bounded-history contract](../../src/transducer/algorithm.rs#L49), [history-carrying variant](../../src/transducer/variants/damerau.rs), and [transition](../../src/transducer/transition.rs) | Finite strings with ordinary sequence equality and unit insertion, deletion, substitution, and unrestricted adjacent transposition. The compact pending-transition representation supports a maximum distance of `u8::MAX`; this is an implementation cutoff, not a change to the ideal distance. | `Algorithm::is_metric` declares the unrestricted shortest-script measure metric; the recurrence and finite-budget source correspondence remain distinct. `to_operation_set` keeps only the local repertoire and produces OSA semantics in the generalized automaton, so that projection cannot inherit this metric claim. |
| Float-weighted edit configurations; [validity predicate](../../src/transducer/costs_f64.rs#L256), [floating transition](../../src/transducer/transition_f64.rs), [floating query](../../src/transducer/query_f64.rs), [weighted cost algebra](../../src/cost/weighted.rs), and [ORC metric qualification](../theory/ordered-residual-calculus.md#121-script-metrics-and-alignment-restrictions) | A fixed alphabet, fixed edit repertoire, and fixed `OperationCostsF64` values. `is_valid` *tests* for finite nonnegative costs and zero match cost, but public fields and `custom()` permit invalid values, and the float-query constructors do not call this predicate. Lawful costs are presently a caller precondition. | A metric requires a separate proof for the *selected recurrence*: symmetry, separation, and an alignment-composition or unrestricted-script theorem. The binary64 cost monoid is not an associative real semiring, and even passing `is_valid` would not be a metric certificate. |
| Scalar ERP; [raw recurrence](../../src/time_series/kernels/erp.rs), [quotient constructor](../../src/time_series/metric_domains.rs#L38), and [metric configuration](../../src/time_series/metric_domains.rs#L224) | Finite scalar sequences, including empty, modulo insertion or deletion of the one fixed finite gap value. `ErpQuotientSeries` removes every gap sample before scoring and indexing. | Raw ERP is a pseudometric; the induced ideal-real distance on the fixed-gap quotient is a metric. [ErpProperties.v](erp/theories/Metric/ErpProperties.v) covers local quotient and bound lemmas; binary64 and indexed-search correspondence remain separate. |
| Vector ERP; [fixed-channel ground metric](../../src/time_series/vector.rs#L264), [fixed vector gap and canonical series](../../src/time_series/vector.rs#L851), and [rolling recurrence](../../src/time_series/vector.rs#L1028) | Finite typed vector sequences, including empty, on one immutable ordered channel/unit layout, fixed positive weights/scales, and one fixed gap vector; equality is after removal of gap samples. Each `VectorErpSeries` retains shared exact gap identity and metric validation rejects a series canonicalized under another gap. | A metric ground cost induces the ideal ERP quotient metric. The [Verus vector kernel](verus/vector_kernels.rs) checks selected ground and interval laws; the sequence recurrence and rounded-score transfer require their own instance proof. |
| Scalar discrete Fréchet; [raw recurrence](../../src/time_series/kernels/frechet.rs), [stutter class](../../src/time_series/metric_domains.rs#L356), [quotient index](../../src/time_series/metric_domains.rs#L273) | Nonempty finite scalar paths modulo consecutive equal-sample stutters. The constructor collapses each maximal run. | Raw path distance is a pseudometric; the ideal bottleneck distance on stutter classes is a metric. [FrechetProperties.v](frechet/theories/Metric/FrechetProperties.v) proves selected recurrence and triangle lemmas; machine and index correspondence remain separate. |
| Vector discrete Fréchet; [stutter path](../../src/time_series/vector.rs#L1798) and [metric constructor](../../src/time_series/vector.rs#L1871) | Nonempty paths of fixed positive dimension, modulo consecutive identical *whole points*, under one sealed `GroundMetric`. Channel coordinates are never flattened into time. `VectorFrechetPath` retains dimension but no channel-layout identity, so a fixed-channel interpretation also requires a stable caller mapping. | If the chosen ground cost is a metric and the sequence recurrence is the discrete Fréchet recurrence, the ideal quotient path distance is a metric. Each ground implementation, layout binding, and binary64 computation retain their own proof obligations. |
| Fixed positive channel point metric; [channel constructor](../../src/time_series/vector.rs#L139) and [fixed layout/scales](../../src/time_series/vector.rs#L264) | Finite equal-dimensional vectors under one nonempty ordered channel/unit schema, fixed positive finite scale and weight per coordinate, and immutable fold-local provenance. Missing coordinates and pair-dependent renormalization are outside this domain. | A positive weighted sum of scaled coordinate absolute differences is an ideal-real metric. [Verus vector kernel](verus/vector_kernels.rs) checks the typed-channel lifting and a missing-channel control; rounded-score laws and every source primitive still require correspondence. |
| Built-in `L1GroundMetric`, `L2GroundMetric`, and `LinfGroundMetric`; [sealed implementations](../../src/time_series/vector.rs#L547) | Finite vector samples of the same positive dimension, with ordinary coordinate equality. The public `GroundMetric::distance` method relies on this domain precondition; `VectorFrechetMetric::distance_bounded` checks dimension before calling it. | The ideal coordinate norms are metrics at fixed dimension. The actual implementations use left-to-right binary64 accumulation, `hypot`, and maximum respectively; no one ideal proof certifies all three operation graphs. |
| Scalar MSM; [metric configuration](../../src/time_series/msm.rs#L569), [metric kernel](../../src/time_series/msm_kernel.rs#L461), and [column recurrence](../../src/time_series/msm_kernel.rs#L525) | Nonempty finite scalar sequences with fixed finite split/merge cost strictly above zero. The raw zero-cost configuration is outside the metric-qualified type. | The standard ideal MSM recurrence is metric on this domain under its source theorem. The [manifest](FORMAL_VERIFICATION_MANIFEST.tsv) still marks the broader MSM proof tree partial; its source and binary64 closure cannot be inferred from the marker. |
| Unit-grid TWED; [validated configuration](../../src/time_series/kernels/twed.rs#L247), [column recurrence](../../src/time_series/kernels/twed.rs#L413), and [metric kernel marker](../../src/time_series/kernels/twed.rs#L825) | Finite scalar sequences on one common unit grid with fixed synthetic zero predecessor, finite positive stiffness, and finite nonnegative gap penalty. | The ideal recurrence has the qualified metric law under its source hypotheses. [TwedProperties.v](twed/theories/Metric/TwedProperties.v) proves recurrence and interval islands; the whole executable and rounded-score law remain separate. |
| Scalar physical-time TWED; [raw constructor](../../src/time_series/timestamped_twed.rs#L160), [strict-origin wrapper](../../src/time_series/timestamped_twed.rs#L286), [metric configuration](../../src/time_series/timestamped_twed.rs#L338), and [bounded recurrence](../../src/time_series/timestamped_twed.rs#L580) | Nonempty finite values; finite strictly increasing timestamps; one fixed unit and origin; first timestamp **strictly later** than origin; positive stiffness and nonnegative gap penalty. `distance_bounded_strict` requires the wrapper. The raw `distance_bounded`, alignment scoring, and index accept `TimestampedSeries` outside this strict domain, so the metric-named configuration does not close admission on those paths. | [TimestampedMetricTransfer.v](twed/theories/Metric/TimestampedMetricTransfer.v) proves a conditional strict-origin transfer. [TwedSourceMetric.v](twed/theories/Metric/TwedSourceMetric.v) proves only the sample/time product metric. The full sequence-level source theorem, raw-API restriction, and binary64 transfer remain open. |
| Vector physical-time TWED; [metric configuration](../../src/time_series/vector.rs#L1175) and [series constructor](../../src/time_series/vector.rs#L1227) | Nonempty typed vector series on one fixed positive channel metric, sentinel, unit, origin, and strictly increasing physical timestamps with the first **strictly later** than origin; positive stiffness and nonnegative gap penalty. | The ideal metric claim needs the same strict-origin source theorem, lifted through the fixed ground metric. The previous `try_series` admitted an origin-equal first sample; the current source rejects it with `TimestampNotAfterOrigin`, checked by a [regression](../../tests/proptest_vector_kernels.rs#L198). The typed/rounded source correspondence remains open. |

The generic [metric kernel marker](../../src/time_series/elastic/mod.rs#L425)
is an **open public trait**: downstream kernels can implement it, so it is a
declaration rather than a closed admission gate. The
[audited-index marker](../../src/time_series/metric_domains.rs#L91) and
[vector ground trait](../../src/time_series/vector.rs#L31) are sealed to
listed implementations. None is a proof object or establishes that a runtime
score obeys exact real metric axioms. Triangle-dependent pruning requires the
specific domain, numeric, recurrence, and source obligations even when a
marker is present.

`Algorithm::is_metric` classifies its built-in unit-cost selector under its
ordinary full-string semantics. [Custom substitution policies](../../src/transducer/substitution_policy.rs#L67)
can declare directed zero-cost equivalence. If dictionary `a` is allowed to
match query `b` at zero cost but the reverse pair is not, the two directions
cost zero and one respectively; ordinary-string separation also fails. Thus
the selector Boolean is not a metric certificate for an arbitrarily
configured transducer, float-weighted query, or substring observation. The separate
[generalized operation-set engine](../../src/transducer/generalized/automaton.rs)
uses its own exact scaled-integer cost representation and needs its own
admission and recurrence proof.

## Explicit nonmetric controls

| Control | Source and failure of a broad metric claim |
|---|---|
| Raw scalar/vector ERP | Inserting the configured gap value or vector costs zero, so distinct raw sequences can have zero distance. Only the fixed-gap canonical quotient has separation. |
| Raw scalar/vector Fréchet | Repeating a point consecutively costs zero. Only nonempty stutter classes have the stated quotient identity. |
| `Algorithm::Transposition` (OSA) | The restricted recurrence can violate triangle inequality: `CA` to `AC` costs one, `AC` to `ABC` costs one, while `CA` to `ABC` costs three. `Algorithm::is_metric` returns false. Projecting true Damerau through `to_operation_set` yields this local repertoire and does not retain unrestricted edit history. |
| Origin-equal physical TWED | With zero gap penalty, inserting a sentinel-valued sample at the physical origin can cost zero. The scalar [strict wrapper](../../src/time_series/timestamped_twed.rs#L279) and vector `try_series` guard exclude that member. The [transfer proof](twed/theories/Metric/TimestampedMetricTransfer.v) checks both the scalar 1-by-2 recurrence and `generic_origin_equal_cumulative_recurrence_is_zero` for any nonnegative reflexive point cost, including a vector ground metric. |
| Arbitrary weighted edits | Asymmetric or zero-separating costs can fail metric laws. A restricted alignment recurrence can also violate triangle even with symmetric positive costs: substitution weights `a↔b=1`, `b↔c=1`, and `a↔c=3`, with insertion/deletion ten, produce a direct `a↔c` cost three versus two through `b`. Unrestricted shortest scripts can apply two substitutions and restore the triangle bound, so the operation table alone does not decide that case. `OperationCostsF64::is_valid` checks finite nonnegative numbers and zero match cost, not metricity. |
| Banded/general DTW | [DtwConfig](../../src/time_series/kernels/dtw.rs#L182) and [DtwTransducer](../../src/time_series/kernels/dtw.rs#L510) are exact nonmetric search capabilities. A concrete triangle counterexample appears in [NotAMetric.v](dtw/theories/NotAMetric.v); they do not implement `MetricElasticKernel`. |
| Soft-DTW | [SoftDtwConfig](../../src/time_series/kernels/soft_dtw.rs#L27) is a soft alignment loss, potentially negative, with path multiplicity. It is analysis-only and is not a metric or an exact min-cost trie kernel. |
| Vector MSM | [VectorMsmSupportDecision](../../src/time_series/vector.rs#L1787) is explicitly unsupported pending a canonical vector betweenness and metric/interval proof. |

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

The vector ERP gap-scope repair stores a shared pointer to the fixed gap in
each canonical series. Same-metric comparisons check pointer identity in
constant time; separately constructed but equal gaps are compared by value.
Different-gap representatives are rejected before the exact recurrence or
candidate bound runs. Construction adds one shared allocation per metric;
each series stores one pointer and increments its reference count, and each
comparison checks one pointer per operand after layout validation. It does
not copy the gap for every series. The allocation is not charged to the
current series resource ledger, so the later resource contract must account
for it. The
[regression](../../tests/proptest_vector_kernels.rs) covers the prior
cross-gap zero-score counterexample and equal-gap representatives constructed
through separate configurations. No measured time or space improvement is
claimed.

## Admission rule for later proofs

A concrete metric or pruning certificate must name the constructor path, the
domain identity (ordinary sequence or a particular quotient), all fixed
parameters and provenance, the ideal measure theorem, the actual binary64
operation graph, and the exported operation's observation profile. A theorem
for a raw pseudometric, a different gap, a different unit/origin, or a
different channel layout cannot fill that certificate's metric premise.
