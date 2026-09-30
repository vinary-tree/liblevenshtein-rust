# TWED source metric contract and anchored transfer

**Epic:** `metric-automata-cbc-theory`, task `cbc-06-twed-source-contract`.
**Status:** source and implementation specification. The sample-space product metric is mechanized; the sequence-level source metric laws remain open.

## Primary theorem and its scope

Marteau's [Time Warp Edit Distance technical report](https://arxiv.org/pdf/0802.3522), Proposition 1, states metricity for its recurrence with a metric on the combined sample/time space, nonnegative gap penalty, and positive stiffness in the standard product distance. Series have strictly increasing timestamps. The source recurrence gives unavailable positive-axis cells infinite cost, so two nonempty series first enter through a match. This boundary convention matters: cumulative one-empty boundaries define a different raw measure.

For the scalar library specialization fix one origin, one time unit, stiffness $`\nu>0`$, gap penalty $`\lambda\ge0`$, ground metric $`\rho(u,v)=|u-v|`$, and finite sample/time values. A physical series $`[(u_1,t_1),\ldots,(u_m,t_m)]`$ has $`t_1>t_0`$ and $`t_{i+1}>t_i`$. Its empty series is allowed under the cumulative convention. These conditions describe ideal real arithmetic and do not assert binary64 metricity.

The source proof concerns a metric on the sample/time product. The local specialization is

```math
d((u,t),(v,s))=\rho(u,v)+\nu|t-s|.
```

With $`\nu>0`$ and scalar absolute ground distance, this is a metric: nonnegative terms are symmetric, zero forces equality in both coordinates, and the triangle inequalities add. [TwedSourceMetric.v](../verification/twed/theories/Metric/TwedSourceMetric.v) checks all five product sample metric laws. A sequence-level proof must still match the source normalization and boundary conventions before importing Proposition 1.

## Library recurrence and anchor map

Let $`a_0=b_0=(0,t_0)`$ be the shared sentinel. The ideal recurrence is

```math
\begin{aligned}
e_i &= d(a_i,a_{i-1})+\lambda,\\
f_j &= d(b_j,b_{j-1})+\lambda,\\
v_{ij} &= d(a_i,b_j)+d(a_{i-1},b_{j-1}),\\
D(0,0)&=0,\\
D(i,0)&=D(i-1,0)+e_i,\\
D(0,j)&=D(0,j-1)+f_j,\\
D(i,j)&=\min\{D(i-1,j)+e_i,\;D(i,j-1)+f_j,\;D(i-1,j-1)+v_{ij}\}.
\end{aligned}
```

The Rust scorer is [timestamped_twed.rs](../../src/time_series/timestamped_twed.rs). Its binary64 operation order, overflow behavior, and cutoff tags require separate machine correspondence. [ORC `11.5](ordered-residual-calculus.md#115-twed-including-explicit-physical-timestamps) states the ideal recurrence and rolling-column state.

Map each input time to $`1+t_i-t_0`$ and prepend a common anchor $`(0,1)`$ to both series. Empty input becomes the one-anchor series. The condition $`t_1>t_0`$ makes the mapped timestamps strictly increasing.

The source's unavailable positive axes force the anchors to match at zero cost. Afterward, its deletion, insertion, and match charges equal the corresponding $`e_i`$, $`f_j`$, and $`v_{ij}`$: common shifts preserve time differences. The source grid at $`(i+1,j+1)`$ equals $`D(i,j)`$. [TimestampedMetricTransfer.v](../verification/twed/theories/Metric/TimestampedMetricTransfer.v) proves recurrence-grid correspondence, physical-charge equalities, strict timestamp admission, and anchor injectivity.

The transfer represents source-infinite positive axes as absent `option` values. It proves the anchored grid is the unique grid satisfying those boundaries and the source recurrence. Its final metric-pullback theorem is **conditional** on four source-distance metric premises. The source theorem must still be mechanized and instantiated for the product metric and anchored sequences.

## Domain counterexample

The raw Rust constructor admits a first timestamp equal to the origin. With $`\lambda=0`$, origin zero, stiffness one, and

```math
X=((1,1)),\qquad Y=((0,0),(1,1)),
```

the cumulative recurrence assigns zero to unequal series. The first sample of $`Y`$ deletes at zero cost, and the remaining samples match at zero cost. The transfer proof checks distinctness and the 1-by-2 score. The [strict-origin wrapper](../../src/time_series/timestamped_twed.rs) excludes $`Y`$. It proves domain admission, not the source triangle inequality.

## Local proof obligations

| Obligation | Required construction | Failure to exclude |
|---|---|---|
| Product sample metric | Derive value/time laws from $`\rho`$ and $`\nu>0`$ | Rounded positive time cost confused with ideal zero |
| Source alignment semantics | Define valid paths with predecessor context and infinite axes | Path using unavailable boundary |
| Recurrence equivalence | Show path/DP equality both ways and minimum attainment | Altered anchor or missing predecessor term |
| Nonnegative and reflexive | Sum nonnegative local costs and use diagonal zero matches | Negative stiffness or gap |
| Symmetry | Reverse every edit with correct context and boundaries | Directional penalty |
| Separation | Strict timestamps prohibit zero-cost insertion/deletion | Origin-equal zero-cost deletion |
| Triangle | Compose lawful paths through an intermediate series with all terminal-edit cases | Treating predecessor cost as memoryless |
| Transfer closure | Instantiate `strict_origin_pullback_is_metric` | Source metricity left as a `Context` premise |

The source report proves triangle inequality by induction on the three lengths and a final-edit case analysis. A Rocq proof must reconstruct all cases with predecessor indices and base cases. A prose statement that edits compose does not discharge the case analysis.

## Status

- **Checked:** product sample metric, anchored recurrence uniqueness, translated charges, injective domain map, strict-wrapper admission, and raw-domain counterexample.
- **Open for ideal metricity:** local source TWED metric theorem under matching boundary/domain/parameter conventions.
- **Open for executable qualification:** machine-score and source-to-model correspondence for the named Rust operation.
